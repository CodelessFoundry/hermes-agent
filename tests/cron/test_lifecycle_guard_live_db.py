"""Regression tests for raw lifecycle-guard reads of live SQLite files."""

from __future__ import annotations

from pathlib import Path

from cron import lifecycle_guard
from hermes_cli.sqlite_safe_read import connect_tracked


def test_referenced_script_read_refuses_live_sqlite_connection(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    conn = connect_tracked(db)
    try:
        def must_not_open(*_args, **_kwargs):
            raise AssertionError("raw-opened a live SQLite database")

        monkeypatch.setattr(lifecycle_guard.os, "open", must_not_open)

        text, unsafe = lifecycle_guard._read_referenced_script(db)

        assert text is None
        assert unsafe is True
    finally:
        conn.close()


def test_referenced_script_read_preserves_normal_script_behavior(tmp_path):
    script = tmp_path / "script.sh"
    script.write_text("echo ok\n", encoding="utf-8")

    text, unsafe = lifecycle_guard._read_referenced_script(script)

    assert text == "echo ok\n"
    assert unsafe is False


def test_invalid_referenced_script_path_is_still_tolerated():
    text, unsafe = lifecycle_guard._read_referenced_script(
        Path("/tmp/hermes\x00binary")
    )

    assert text is None
    assert unsafe is False


def test_offline_oversized_sqlite_file_is_not_a_referenced_script(tmp_path):
    """A SQLite database is data, not a script: it must be sniffed out by its magic
    header and skipped like any other binary format, never fail closed on size.

    A cron job routinely tells the agent to query its own state database with the
    `sqlite3 <path>` CLI (e.g. an intel-pipeline job reading its own intel.db); when the
    db exceeds the 1 MiB referenced-script cap, the guard must not treat the data file as
    an unscannable "script" and log/refuse on it (t_8db3f6ed).
    """
    db = tmp_path / "intel.db"
    # Real SQLite header followed by enough padding to clear the 1 MiB cap, so the old
    # behavior (fail closed on size) and the fixed behavior (skip as binary) are
    # distinguishable by more than just the header bytes.
    db.write_bytes(b"SQLite format 3\x00" + b"\x00" * (2 * 1024 * 1024))

    text, unsafe = lifecycle_guard._read_referenced_script(db)

    assert text is None
    assert unsafe is False


def test_offline_oversized_non_sqlite_file_still_fails_closed(tmp_path):
    """Control for the fix above: an oversized file that is NOT a recognized binary
    format (i.e. could plausibly be a real shell/Python script) must still fail closed,
    unchanged from before."""
    script = tmp_path / "big.sh"
    script.write_bytes(b"#!/bin/sh\n" + b"echo hi\n" * 200_000)

    text, unsafe = lifecycle_guard._read_referenced_script(script)

    assert text is None
    assert unsafe is True
