from types import SimpleNamespace
from typing import Any

import pytest

import cli


@pytest.fixture(autouse=True)
def reset_single_query_finalize_state(monkeypatch):
    monkeypatch.setattr(cli, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli, "_cleanup_done", False)




def test_finalize_single_query_releases_lease_before_cleanup(monkeypatch):
    """Settlement (finalize hook) runs before the release, the release before
    cleanup; a cleanup failure can never skip or precede the release."""
    calls = []
    fake_cli = SimpleNamespace(_release_active_session=lambda: calls.append("release"))

    def cleanup(**kwargs):
        calls.append("cleanup")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        cli,
        "_notify_single_query_session_finalize",
        lambda _cli: calls.append("finalize"),
    )
    monkeypatch.setattr(cli, "_run_cleanup", cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        cli._finalize_single_query(fake_cli)

    assert calls == ["finalize", "release", "cleanup"]


def test_finalize_single_query_runs_cleanup_when_finalize_hook_fails(monkeypatch):
    calls = []
    fake_agent = SimpleNamespace(session_id="agent-session", platform="cli")
    fake_cli = SimpleNamespace(
        agent=fake_agent,
        session_id="cli-session",
        _release_active_session=lambda: calls.append("release"),
    )

    def invoke_hook(name, **kwargs):
        calls.append("finalize")
        raise RuntimeError("hook failed")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **kwargs: calls.append("cleanup"))

    cli._finalize_single_query(fake_cli)

    assert calls == ["finalize", "release", "cleanup"]


def test_finalize_settles_session_before_a_successor_can_take_over(tmp_path, monkeypatch):
    """P1 handoff race: a successor that acquires the session DURING the
    predecessor's exit linger must never receive the predecessor's stale
    ``cli_close`` end-stamp — settlement (flush incl. end_session) must complete
    before the lease is released.

    Exercises the real seam with a real SessionDB and a real lease: the linger is
    patched to play the successor (acquire the freed lease, reopen the row), and
    the final assertion is that the successor's open interval survives the
    predecessor's whole finalize. On the unsafe order (release → linger → flush)
    the flush's unconditional ``end_session`` stamps the successor's interval and
    this test fails."""
    from hermes_cli import active_sessions
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "s1"
    db.create_session(sid, "cli")
    lease_a, message = active_sessions.try_acquire_active_session(
        session_id=sid, surface="cli", config={}, metadata={"live_session_id": sid}
    )
    assert lease_a is not None, message

    agent_a = SimpleNamespace(
        session_id=sid, _session_db=db, _session_messages=[],
        _persist_disabled=False, _persist_session=lambda *a, **k: None,
    )
    cli_a = SimpleNamespace(
        agent=agent_a, session_id=sid, conversation_history=[], _session_db=db,
        _release_active_session=lease_a.release,
    )

    taken_over: dict[str, Any] = {}

    def linger_with_takeover(_cli):
        # The successor arrives mid-linger. Early admission must hold (the point of
        # the PR)…
        lease_b, msg_b = active_sessions.try_acquire_active_session(
            session_id=sid, surface="cli", config={}, metadata={"live_session_id": sid}
        )
        assert lease_b is not None, f"successor could not acquire during the linger: {msg_b}"
        taken_over["lease_b"] = lease_b
        # …and it resumes the row (the predecessor's settlement already ended it).
        db.reopen_session(sid)

    monkeypatch.setattr(cli, "_wait_for_oneshot_background_completions", linger_with_takeover)
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _c: None)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_k: None)

    cli._finalize_single_query(cli_a)

    row = db.get_session(sid)
    assert row is not None and row["ended_at"] is None, (
        "successor's open interval was end-stamped by the predecessor's finalize"
    )
    taken_over["lease_b"].release()


def test_notify_single_query_session_finalize_uses_agent_session(monkeypatch):
    calls = []
    fake_agent = SimpleNamespace(session_id="agent-session", platform="cli")
    fake_cli = SimpleNamespace(agent=fake_agent, session_id="cli-session")

    def invoke_hook(name, **kwargs):
        calls.append((name, kwargs))

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)

    cli._notify_single_query_session_finalize(fake_cli)

    assert calls == [
        (
            "on_session_finalize",
            {
                "session_id": "agent-session",
                "platform": "cli",
                "reason": "shutdown",
            },
        )
    ]


# ── background-review / session-close race (t_41c1e219) ─────────────────────
#
# Evidence: harness log pair
#   agent.relay_runtime: Hermes Relay drained N orphaned scope(s) before closing ScopeHandle(...)
#   agent.relay_runtime: Hermes Relay turn finalization failed: turn scope close failed: Hermes
#   Relay session is closing
# A one-shot (kanban/cron/CLI -Q) turn's post-turn background review runs on a daemon thread
# that opens its own Relay turn scope under the SAME session_id as the parent. Nothing used to
# wait for it before the parent process finalized (and hard-closed) that Relay session, so the
# review's scope got force-drained mid-flight and its own end_turn() then found the session
# already closing. Fix: await/cancel the in-flight review (the same bounded, fail-open primitive
# already used at the START of every live turn) before notifying session finalize.

def test_notify_single_query_session_finalize_awaits_background_review_first(monkeypatch):
    """The in-flight review must be cancelled/awaited BEFORE the Relay session finalize fires —
    otherwise close_session can race the review's own end_turn() (the orphaned-scope class of bug)."""
    calls = []
    fake_agent = SimpleNamespace(session_id="agent-session", platform="cli")
    fake_cli = SimpleNamespace(agent=fake_agent, session_id="cli-session")

    monkeypatch.setattr(
        cli, "_await_background_review_before_session_close",
        lambda agent: calls.append(("await_review", agent)),
    )
    def _fake_notify_session_finalize(**kwargs):
        calls.append(("finalize", kwargs))
        return []  # the real one yields stderr messages; _notify_single_query_session_finalize iterates it

    monkeypatch.setattr(cli, "_notify_session_finalize", _fake_notify_session_finalize)

    cli._notify_single_query_session_finalize(fake_cli)

    assert [c[0] for c in calls] == ["await_review", "finalize"]
    assert calls[0][1] is fake_agent


def test_await_background_review_before_session_close_calls_cancel_primitive(monkeypatch):
    """Delegates to the existing cancel_background_review_for_live_turn primitive (already used
    at live-turn start) rather than reinventing cancellation/wait semantics."""
    fake_agent = SimpleNamespace(session_id="agent-session")
    calls = []
    monkeypatch.setattr(
        "agent.background_review.cancel_background_review_for_live_turn",
        lambda agent: calls.append(agent),
    )

    cli._await_background_review_before_session_close(fake_agent)

    assert calls == [fake_agent]


def test_await_background_review_before_session_close_is_noop_for_none_agent():
    # Must not raise — the single-query path can have no agent on early-exit failures.
    cli._await_background_review_before_session_close(None)


def test_await_background_review_before_session_close_never_raises(monkeypatch):
    """Best-effort: a broken review-cancellation path must never block process exit."""
    def _boom(agent):
        raise RuntimeError("review cancellation exploded")

    monkeypatch.setattr("agent.background_review.cancel_background_review_for_live_turn", _boom)

    cli._await_background_review_before_session_close(SimpleNamespace())  # must not raise


def test_human_single_query_main_finalizes_after_query(monkeypatch):
    calls = []

    import cli as cli_mod

    class _Console:
        def print(self, *_args, **_kwargs):
            calls.append("query-label")

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = _Console()
            self.session_id = "single-query-session"
            self.agent = SimpleNamespace(
                session_id="single-query-session",
                platform="cli",
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _show_security_advisories(self):
            calls.append("advisories")

        def chat(self, query, images=None):
            calls.append(("chat", query, images))
            self._last_turn_result = {"final_response": "done", "completed": True}
            return "done"

        def _print_exit_summary(self, clear_screen=True):
            calls.append("summary")

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )

    # The non-quiet one-shot path exits with the turn's outcome (0 here), like ``-Q``.
    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert exc_info.value.code == 0
    assert ("chat", "hello", None) in calls
    assert calls[-1] == ("finalize", "single-query-session")


def test_quiet_single_query_main_finalizes_while_preserving_exit_code(monkeypatch):
    calls = []

    import cli as cli_mod

    def run_conversation(*, user_message, conversation_history):
        calls.append(("run", user_message, conversation_history))
        return {
            "final_response": "",
            "error": "provider failed",
            "failed": True,
        }

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.provider = "test-provider"
            self.model = "test-model"
            self.session_id = "quiet-session"
            self.conversation_history = []
            self._active_agent_route_signature = "same-route"
            self.agent = SimpleNamespace(
                session_id="quiet-session",
                platform="cli",
                quiet_mode=False,
                suppress_status_output=False,
                stream_delta_callback=object(),
                tool_gen_callback=object(),
                run_conversation=run_conversation,
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _ensure_runtime_credentials(self):
            calls.append("credentials")
            return True

        def _resolve_turn_agent_config(self, effective_query):
            calls.append(("resolve", effective_query))
            return {
                "signature": "same-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **kwargs):
            calls.append(("init", kwargs))
            return True

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 1
    assert ("claim", "cli", True) in calls
    assert ("run", "hello", []) in calls
    assert calls[-1] == ("finalize", "quiet-session")
