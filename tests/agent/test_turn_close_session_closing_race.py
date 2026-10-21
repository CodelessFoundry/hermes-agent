"""Regression for t_1c12e7e5: turn/logical-call scope close must not be fenced out by a
concurrent close_session() that already flipped session.closing.

Root cause (see agent/relay_runtime.py _close_turn_scope / _finish_logical_calls): both call
sites passed allow_closing=False (the default) to _close_scope_handle(). _session_context()
raises "Hermes Relay session is closing" when session.closing is True and allow_closing is not
set -- so if close_session() (e.g. CLI shutdown finalizing the session) wins the race and flips
closing=True before the turn's own end_turn()/_finish_logical_calls() pop their scopes, those
pops are rejected outright, never even attempted. That produces the exact paired warning seen in
production:
  - the turn: "Hermes Relay turn finalization failed: turn scope close failed: Hermes Relay
    session is closing"
  - the session close, which then force-drains the turn's never-popped scope as an orphan:
    "Hermes Relay drained N orphaned scope(s) before closing ScopeHandle(...)"

t_41c1e219's fix only added a bounded (2s, fail-open) await before closing a background-review
fork's session -- it narrowed the window for ONE caller, but did not touch the structural fence,
so the warning recurred within one cycle on both profiles (t_873f6e87, t_1c12e7e5). The real fix
is allow_closing=True on the turn's own close-path calls, matching what close_session() already
does for ITS OWN pop -- a scope already on the stack must always be allowed to finish popping,
regardless of which thread flipped session.closing first.
"""

from __future__ import annotations

import logging
import tempfile

import pytest

from agent import relay_runtime


def test_end_turn_closes_cleanly_when_session_already_closing(monkeypatch, caplog):
    pytest.importorskip("nemo_relay")

    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp())
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-t1c12e7e5",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="task-1",
    )
    lease.host.retain_managed_execution("test.turn_close_session_closing_race")

    # A nested logical-LLM-call scope under the turn, like a background-review fork's own API
    # call sharing the parent's session_id.
    logical = lease.host.run_in_session(
        lease.session,
        lease.host.relay.scope.push,
        "logical-llm",
        lease.host.relay.ScopeType.Custom,
        handle=turn.handle,
        input={},
        metadata={"hermes.test": True},
    )
    turn.logical_llm_calls["api-1"] = logical

    # Simulate the race: another thread's close_session() wins and flips session.closing=True
    # BEFORE this turn gets to pop its own scopes (e.g. CLI shutdown finalizing a one-shot
    # session while a background-review fork is still mid-flight on the same session_id).
    with lease.session.lock:
        lease.session.closing = True

    with caplog.at_level(logging.WARNING, logger="agent.relay_runtime"):
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "turn finalization failed" not in joined
    assert "session is closing" not in joined
    assert "drained" not in joined
    assert turn.logical_llm_calls == {}
    assert turn.closed is True

    lease.host.release_managed_execution("test.turn_close_session_closing_race")
    relay_runtime._reset_for_tests()
