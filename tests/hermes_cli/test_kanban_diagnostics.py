"""Tests for hermes_cli.kanban_diagnostics — rule-engine that produces
structured distress signals (diagnostics) for kanban tasks.

These tests exercise each rule in isolation using minimal in-memory
task/event/run fixtures (no DB) plus a few integration-style cases
that round-trip through the real kanban_db to make sure the rule
engine works on sqlite3.Row objects as well as dataclasses.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _run(outcome="completed", run_id=1, error=None):
    return {
        "id": run_id,
        "outcome": outcome,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------
















def test_running_with_open_parents_fires_only_while_running():
    """A running card whose parent is not terminal is flagged; the same graph
    on a ready/todo card (the gate is holding it) and a done parent are not."""
    graph = {"parents": [{"id": "t_parent", "title": "p", "status": "todo"}], "children": []}
    diags = kd.compute_task_diagnostics(_task(status="running", started_at=100), [], [], graph=graph)
    assert [d.kind for d in diags] == ["running_with_open_parents"]
    assert diags[0].data["open_parents"] == [{"id": "t_parent", "status": "todo"}]
    assert "hermes kanban unlink t_parent t_demo00" in diags[0].actions[0].payload["command"]
    assert kd.compute_task_diagnostics(_task(status="todo"), [], [], graph=graph) == []
    done_graph = {"parents": [{"id": "t_parent", "title": "p", "status": "done"}], "children": []}
    assert kd.compute_task_diagnostics(_task(status="running"), [], [], graph=done_graph) == []


def test_stuck_in_blocked_fires_past_threshold():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=now - 3600 * 48, reason="needs approval"),
    ]
    diags = kd.compute_task_diagnostics(
        task, events, [], now=now,
    )
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["age_hours"] >= 48


# ---------------------------------------------------------------------------
# blocked_stale_despite_comments — complement to _rule_stuck_in_blocked
#
# _rule_stuck_in_blocked is fully silenced by ANY comment after the block,
# which hides a long-blocked card that gets periodic "I re-checked, nothing
# changed" audit comments instead of an actual resolution. Reconstructed
# shaped exactly like the real live case, harness:t_1c68c712: blocked
# 2026-10-11, re-audited (comments) ~3 days and ~10 days later, 17+ days
# stale and never once flagged.
# ---------------------------------------------------------------------------


def test_blocked_stale_despite_comments_fires_on_audit_shaped_history():
    """Reconstructed t_1c68c712 shape: blocked, then two comments each a real
    audit re-check days apart, status never changes. The existing
    stuck_in_blocked rule would stay silent forever here (a comment exists
    after the block) — this rule must fire instead."""
    now = int(time.time())
    blocked_ts = now - 20 * 86400
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=blocked_ts, reason="needs_input"),
        _event("commented", ts=blocked_ts + 3 * 86400),   # re-audit #1
        _event("commented", ts=blocked_ts + 10 * 86400),  # re-audit #2
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stale = [d for d in diags if d.kind == "blocked_stale_despite_comments"]
    assert len(stale) == 1
    d = stale[0]
    assert d.data["comment_count"] == 2
    assert d.data["age_days"] >= 20
    # _rule_stuck_in_blocked must stay silent (a comment exists after the
    # block) — this rule is additive, not a replacement.
    assert "stuck_in_blocked" not in [x.kind for x in diags]


def test_blocked_stale_despite_comments_not_fired_by_tight_exchange():
    """A real back-and-forth (comments close together) looks like it may
    have resolved something — must NOT fire, even past the age threshold."""
    now = int(time.time())
    blocked_ts = now - 20 * 86400
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=blocked_ts),
        _event("commented", ts=blocked_ts + 3600),   # 1h later
        _event("commented", ts=blocked_ts + 7200),   # 2h later — tight pair
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    assert "blocked_stale_despite_comments" not in [d.kind for d in diags]


def test_blocked_stale_despite_comments_requires_age_threshold():
    """Same audit shape, but the block is only 3 days old — too young to
    fire regardless of comment spacing."""
    now = int(time.time())
    blocked_ts = now - 3 * 86400
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=blocked_ts),
        _event("commented", ts=blocked_ts + 86400),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    assert "blocked_stale_despite_comments" not in [d.kind for d in diags]


def test_blocked_stale_despite_comments_single_comment_still_fires():
    """A single early comment followed by long silence is vacuously 'every
    comment spaced apart' (zero pairs to compare) — it should fire, not be
    treated as a special case."""
    now = int(time.time())
    blocked_ts = now - 20 * 86400
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=blocked_ts),
        _event("commented", ts=blocked_ts + 60),  # 1 minute after block
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    assert "blocked_stale_despite_comments" in [d.kind for d in diags]


# ---------------------------------------------------------------------------
# stale_untouched_todo — Scott's literal ask (t_c09b7e63 ask 2)
#
# Scoped narrowly: status=todo, no open parent link, zero
# claimed/spawned/commented events ever, age >= 14 days. Must NOT fire on a
# root card that's legitimately waiting on an in-flight child (the t_01697e00
# shape) — that card has plenty of historical claimed/spawned events from the
# day it was worked and decomposed, even though nothing has happened on it
# since.
# ---------------------------------------------------------------------------


def test_stale_untouched_todo_fires_when_genuinely_untouched():
    now = int(time.time())
    created_at = now - 20 * 86400
    task = _task(status="todo", created_at=created_at)
    events = [_event("created", ts=created_at)]
    graph = {"parents": [], "children": []}
    diags = kd.compute_task_diagnostics(task, events, [], now=now, graph=graph)
    stale = [d for d in diags if d.kind == "stale_untouched_todo"]
    assert len(stale) == 1
    assert stale[0].data["age_days"] >= 20


def test_stale_untouched_todo_does_not_fire_with_open_parent():
    """A todo card gated by a real open parent link must not fire — the
    dependency gate already explains the silence."""
    now = int(time.time())
    created_at = now - 20 * 86400
    task = _task(status="todo", created_at=created_at)
    events = [_event("created", ts=created_at)]
    graph = {"parents": [{"id": "t_parent", "title": "p", "status": "todo"}], "children": []}
    diags = kd.compute_task_diagnostics(task, events, [], now=now, graph=graph)
    assert "stale_untouched_todo" not in [d.kind for d in diags]


def test_stale_untouched_todo_does_not_fire_on_worked_then_waiting_shape():
    """Reconstructed t_01697e00 shape: a root card with NO parent of its own,
    worked and decomposed the same day it was created, then 6+ weeks of
    silence while its last open CHILD resolves. It has no open parent link
    (it's a root card) but it DOES have historical claimed/spawned events —
    the event-history check must exclude it, not the parent-link check."""
    now = int(time.time())
    created_at = now - 45 * 86400
    task = _task(status="todo", created_at=created_at)
    events = [
        _event("created", ts=created_at),
        _event("claimed", ts=created_at + 10),
        _event("spawned", ts=created_at + 15),
        _event("completed", ts=created_at + 3600),  # decomposed same day
    ]
    graph = {"parents": [], "children": [{"id": "t_1c68c712", "title": "c", "status": "blocked"}]}
    diags = kd.compute_task_diagnostics(task, events, [], now=now, graph=graph)
    assert "stale_untouched_todo" not in [d.kind for d in diags]


def test_stale_untouched_todo_requires_age_threshold():
    now = int(time.time())
    created_at = now - 5 * 86400
    task = _task(status="todo", created_at=created_at)
    events = [_event("created", ts=created_at)]
    graph = {"parents": [], "children": []}
    diags = kd.compute_task_diagnostics(task, events, [], now=now, graph=graph)
    assert "stale_untouched_todo" not in [d.kind for d in diags]


def test_stale_untouched_todo_silent_without_graph_context():
    """Graph-aware, same convention as _rule_running_with_open_parents: no
    graph passed means the parent set is unknown, so stay silent rather than
    risk a false positive."""
    now = int(time.time())
    created_at = now - 20 * 86400
    task = _task(status="todo", created_at=created_at)
    events = [_event("created", ts=created_at)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    assert "stale_untouched_todo" not in [d.kind for d in diags]






# ---------------------------------------------------------------------------
# Severity sorting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Integration — runs through real kanban_db so sqlite.Row fields work
# ---------------------------------------------------------------------------


def test_engine_works_on_sqlite_row_objects(kanban_home):
    """Regression: the rule functions must handle sqlite3.Row (which
    supports mapping access but not attribute access and isn't a dict)
    as well as dataclass Task / plain dict. The API layer passes Row
    objects directly.
    """
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="p", assignee="w")
        real = kb.create_task(conn, title="r", assignee="x", created_by="w")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="with phantom", created_cards=[real, "t_deadbeef1"],
            )
        # Pull Row objects the way the API helper does.
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (parent,),
        ).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        diags = kd.compute_task_diagnostics(row, events, runs)
        assert len(diags) == 1
        assert diags[0].kind == "hallucinated_cards"
        assert "t_deadbeef1" in diags[0].data["phantom_ids"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Error-tolerance: a broken rule shouldn't 500 the whole compute call
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# stranded_in_ready
#
# Surfaces ready tasks that nobody has claimed within the threshold.
# Identity-agnostic by design: catches typo'd assignees, deleted profiles,
# down external worker pools, and misconfigured dispatchers in one rule.
# ---------------------------------------------------------------------------


def test_stranded_in_ready_fires_when_age_exceeds_threshold():
    """Default threshold = 30 min. A ready task promoted 45 min ago
    with no claim should fire as a warning."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    # 45 min = 2700s, threshold = 1800s.
    events = [_event("created", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["age_seconds"] == 45 * 60
    assert stranded[0].data["assignee"] == "demo"




# ---------------------------------------------------------------------------
# triage_aux_unavailable rule — auto-decompose aware
# ---------------------------------------------------------------------------


def _triage_task():
    return _task(id="t_triage1", status="triage")








def test_severity_at_or_above_uses_threshold_semantics():
    assert kd.severity_at_or_above("warning", "warning") is True
    assert kd.severity_at_or_above("error", "warning") is True
    assert kd.severity_at_or_above("critical", "warning") is True
    assert kd.severity_at_or_above("critical", "error") is True
    assert kd.severity_at_or_above("warning", "error") is False
    assert kd.severity_at_or_above("error", "critical") is False
    assert kd.severity_at_or_above("mystery", "warning") is False
    assert kd.severity_at_or_above("warning", None) is True
