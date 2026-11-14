"""A wedged /proc/<pid>/cmdline read must bound the wait, not hang the scan.

Covers the hardening for the real incident: a stuck mm lock on a host made a
synchronous open()/read() of /proc/<pid>/cmdline block forever in
uninterruptible D-state, parking Hermes gateway startup (foreign_state_db_holders
runs on the gateway's main thread during state.db structural maintenance).
"""

import os
import threading
import time
from types import SimpleNamespace

import pytest

import hermes_state_holders


@pytest.fixture(autouse=True)
def _fast_timeout(monkeypatch):
    """Shrink the bound to keep these tests fast without changing behavior."""
    monkeypatch.setattr(hermes_state_holders, "_PROC_ARGV_READ_TIMEOUT_S", 0.05)


def _hang_forever(pid):
    """Stand-in for a ``_read_proc_argv`` stuck in a D-state read: never returns."""
    threading.Event().wait()  # blocks the calling (daemon) thread indefinitely


def test_read_proc_argv_bounded_returns_promptly_on_a_hung_read(monkeypatch):
    """A pid whose cmdline read never completes resolves as timed-out, fast."""
    monkeypatch.setattr(hermes_state_holders, "_read_proc_argv", _hang_forever)

    started = time.monotonic()
    argv, timed_out = hermes_state_holders._read_proc_argv_bounded(4242)
    elapsed = time.monotonic() - started

    assert (argv, timed_out) == (None, True)
    # Bounded by the (shrunk) timeout, not by the hang -- this is the whole point.
    assert elapsed < 1.0


def test_read_proc_argv_bounded_passes_through_a_clean_read(monkeypatch):
    """A normal, fast read is returned unchanged with timed_out=False."""
    monkeypatch.setattr(
        hermes_state_holders, "_read_proc_argv", lambda pid: ["hermes", "gateway"]
    )

    argv, timed_out = hermes_state_holders._read_proc_argv_bounded(4242)

    assert argv == ["hermes", "gateway"]
    assert timed_out is False


def test_read_proc_argv_bounded_passes_through_a_clean_none(monkeypatch):
    """A process that's simply gone (OSError inside _read_proc_argv -> None) stays None, not a timeout."""
    monkeypatch.setattr(hermes_state_holders, "_read_proc_argv", lambda pid: None)

    argv, timed_out = hermes_state_holders._read_proc_argv_bounded(4242)

    assert (argv, timed_out) == (None, False)


@pytest.mark.platforms("linux")
def test_foreign_state_db_holders_treats_a_wedged_pid_as_uninspectable_not_a_hang(
    tmp_path, monkeypatch
):
    """Full scan: a pid whose /proc/<pid>/fd listing fails AND whose cmdline read
    hangs must come back as an uninspectable (fail-closed) holder within the
    bounded timeout -- it must never make the whole scan block.
    """
    db_path = tmp_path / "state.db"
    db_path.touch()

    proc_root = tmp_path / "proc"
    # A live, healthy pid with no fd entries pointing at our db (not a holder).
    (proc_root / "111" / "fd").mkdir(parents=True)
    # The wedged pid: its /proc/<pid>/fd dir itself is unreadable (as happens for
    # some kernel-stuck processes), forcing the _read_proc_argv fallback.
    wedged_pid = 999

    projected_os = SimpleNamespace(**vars(os))
    projected_os.getpid = lambda: 1  # neither 111 nor 999
    monkeypatch.setattr(hermes_state_holders, "os", projected_os)

    real_listdir = os.listdir

    def _listdir(path):
        if path == "/proc":
            return ["111", str(wedged_pid)]
        if path == f"/proc/{wedged_pid}/fd":
            raise OSError("stuck kernel lock: fd dir unreadable")
        mapped = path.replace("/proc", str(proc_root)) if isinstance(path, str) else path
        return real_listdir(mapped)

    monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)

    # The cmdline read for the wedged pid never returns -- exactly the incident.
    def _read_proc_argv(pid):
        if pid == wedged_pid:
            threading.Event().wait()
        return None

    monkeypatch.setattr(hermes_state_holders, "_read_proc_argv", _read_proc_argv)

    started = time.monotonic()
    holders = hermes_state_holders.foreign_state_db_holders(db_path)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, "the scan must not block on the wedged pid's hung read"
    pids = [pid for pid, _ in holders]
    assert wedged_pid in pids
    description = next(desc for pid, desc in holders if pid == wedged_pid)
    assert "uninspectable" in description
    assert "D-state" in description


@pytest.mark.platforms("linux")
def test_live_writer_holds_db_stays_fail_closed_on_a_wedged_holder(tmp_path, monkeypatch):
    """live_writer_holds_db must still refuse (assume held) when the holder scan
    reports a timed-out/uninspectable pid -- the fix bounds the WAIT, it must
    not weaken the existing fail-closed guarantee structural maintenance relies on.
    """
    db_path = tmp_path / "state.db"
    db_path.touch()

    monkeypatch.setattr(
        hermes_state_holders,
        "foreign_state_db_holders",
        lambda path: [(999, "uninspectable holder: /proc/999/cmdline read exceeded 2s (process may be wedged in D-state)")],
    )

    def _connect_repair_durable(*_args, **_kwargs):
        raise AssertionError("must not even attempt the lock probe when a holder is already known")

    assert hermes_state_holders.live_writer_holds_db(
        db_path, connect_repair_durable=_connect_repair_durable
    ) is True
