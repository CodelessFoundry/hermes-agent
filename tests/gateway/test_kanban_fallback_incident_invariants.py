"""Regression tests for incident t_d79e0587 (2026-11-14): a "home-channel fallback"
(commit 79e27143583, task t_507b5157) flooded the owner's Signal.

The gateway restarted at 15:27, loading a notifier change that auto-subscribed his
Signal (plus the Discord home channel) to every card on every board that EVER had a
blocked/crashed/etc event (~260 tasks, most long archived or done). Three independent
bugs combined:

1. The fallback read the plain per-platform home channel, which includes Signal — the
   owner's one direct personal line, which must NEVER be auto-subscribed to a card.
2. The new subscription's cursor started at 0 ("replay everything"), instead of the
   caught-up default every other subscription path in this codebase uses — so each of
   the ~260 tasks dumped its full event history on first delivery.
3. The fallback candidate query matched on event-kind history alone with no
   ``tasks.status`` filter, so an already-archived/done task (whose last event long
   predated this feature) matched forever; the notifier's own "unsubscribe on archive"
   step then deleted the row it had just created, so the SAME task re-qualified and
   got rebuilt — and its full history re-delivered — on every subsequent tick.

The fallback was reverted (79e27143583 reverted by eecca331394) rather than patched
live, because getting all three invariants right under incident pressure was exactly
how the original bug shipped. These tests pin the three invariants directly against
the correctness-fixed-but-unwired primitives kept in ``kanban_db_notify`` and against
``gateway.config``'s new blocklist, so any future orphan-event path (wired through
``add_fallback_notify_subs``/``tasks_needing_fallback_subscription`` or not) that
reintroduces any one of the three bugs fails here first.
"""

import time

from gateway.config import GatewayConfig, HomeChannel, NEVER_AUTO_SUBSCRIBE_PLATFORMS, Platform, PlatformConfig
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


def _board_conn(tmp_path, monkeypatch):
    db_path = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    return kbc.connect()


# --- Bug 1: the home channel being Signal -----------------------------------------


def test_signal_is_excluded_from_the_blocklist_by_construction():
    """The policy is a blocklist, not an allowlist: Signal must be in it, and the
    exclusion must actually be enforced by the sanctioned read helper."""
    assert Platform.SIGNAL in NEVER_AUTO_SUBSCRIBE_PLATFORMS

    config = GatewayConfig(platforms={
        Platform.SIGNAL: PlatformConfig(
            enabled=True,
            home_channel=HomeChannel(platform=Platform.SIGNAL, chat_id="+491****7890", name="Home"),
        ),
    })
    # The plain accessor still works (manual /kanban notify-subscribe relies on it).
    assert config.get_home_channel(Platform.SIGNAL) is not None
    # The unattended-fallback accessor must refuse it outright.
    assert config.get_unattended_fallback_home_channel(Platform.SIGNAL) is None


def test_non_signal_platform_still_available_to_unattended_fallback():
    """The blocklist must not become a de facto allowlist that strands every
    platform — only Signal is excluded; everything else still resolves."""
    config = GatewayConfig(platforms={
        Platform.DISCORD: PlatformConfig(
            enabled=True,
            home_channel=HomeChannel(platform=Platform.DISCORD, chat_id="123456", name="Home"),
        ),
    })
    assert config.get_unattended_fallback_home_channel(Platform.DISCORD) is not None


# --- Bug 2: history replay (cursor must start caught up, never 0) -----------------


def test_fallback_sub_creation_never_replays_history(tmp_path, monkeypatch):
    """A task that already has events when the fallback sub is created must start
    caught up — the triggering event (and nothing before it) is what gets delivered
    going forward, not the task's entire history."""
    conn = _board_conn(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="old stuck task", assignee="worker")
        # Simulate a task with real history before anyone ever subscribed to it.
        for _ in range(5):
            kb.block_task(conn, tid, reason="retry", kind="transient")
            kb.unblock_task(conn, tid)
        kb.block_task(conn, tid, reason="need Scott's approval", kind="needs_input")

        max_event_id = conn.execute(
            "SELECT MAX(id) FROM task_events WHERE task_id = ?", (tid,),
        ).fetchone()[0]
        assert max_event_id and max_event_id > 1, "fixture should have produced real history"

        created = kbn.add_fallback_notify_subs(
            conn, task_id=tid, routes=[("discord", "home-chat-1")], notifier_profile="tess",
        )
        assert created is True

        subs = kbn.list_notify_subs(conn, task_id=tid)
        assert len(subs) == 1
        # Caught up, not 0 — the incident's exact defect was a hardcoded last_event_id=0.
        assert subs[0]["last_event_id"] == max_event_id
    finally:
        conn.close()


# --- Bug 3: re-subscribing after a claim (archived/settled tasks must never re-qualify) --


def test_archived_task_does_not_requalify_for_fallback(tmp_path, monkeypatch):
    """A task that is archived now (even though it was blocked once, long ago) must
    never be treated as a fresh zero-sub candidate — this is exactly the create/
    unsubscribe-on-archive/recreate loop that produced 664 subscribe events."""
    conn = _board_conn(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="ancient blocked task", assignee="worker")
        kb.block_task(conn, tid, reason="need Scott's approval", kind="needs_input")
        kb.archive_task(conn, tid)

        candidates = kbn.tasks_needing_fallback_subscription(
            db_path=kb.kanban_db_path(), kinds=("blocked", "crashed", "timed_out"),
        )
        assert tid not in candidates
    finally:
        conn.close()


def test_currently_blocked_task_still_qualifies(tmp_path, monkeypatch):
    """Sanity check on the other side of the fix: a task that is genuinely stuck right
    now, with zero subscribers, must still qualify — the status filter must not
    over-correct into never firing at all."""
    conn = _board_conn(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="genuinely stuck task", assignee="worker")
        kb.block_task(conn, tid, reason="need Scott's approval", kind="needs_input")

        candidates = kbn.tasks_needing_fallback_subscription(
            db_path=kb.kanban_db_path(), kinds=("blocked", "crashed", "timed_out"),
        )
        assert tid in candidates

        # Once subscribed, it must drop out of the candidate set immediately — no
        # re-create-on-next-tick for a task that is simply sitting blocked and watched.
        kbn.add_fallback_notify_subs(conn, task_id=tid, routes=[("discord", "home-chat-1")])
        candidates_after = kbn.tasks_needing_fallback_subscription(
            db_path=kb.kanban_db_path(), kinds=("blocked", "crashed", "timed_out"),
        )
        assert tid not in candidates_after
    finally:
        conn.close()
