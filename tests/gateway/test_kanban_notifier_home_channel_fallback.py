"""Home-channel fallback for board-wide silent timeouts (t_507b5157).

Root cause: a task created programmatically (worker, orchestrator, cron — anything
that isn't a human typing ``/kanban create`` in a live chat) never gets a
``kanban_notify_subs`` row, because the only auto-subscribe path
(``slash_commands.py::_kanban_auto_subscribe``) only fires on that one interactive
command. A ``blocked``/``review_requested``/etc. event on such a task has nowhere to
go — it sits in ``task_events`` inert. These tests pin the fix: the notifier now
falls back to the configured home channel(s) for exactly that case.
"""

import asyncio

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


def _make_runner(adapter, *, home_chat_id="+491234567890"):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    runner.config = GatewayConfig(platforms={
        Platform.TELEGRAM: PlatformConfig(
            enabled=True,
            home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id=home_chat_id, name="Home"),
        ),
    })
    return runner


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _create_unsubscribed_task(*, kind: str, assignee: str = "worker") -> str:
    """A task created the way every worker/orchestrator task is: no human in the loop,
    so no ``_kanban_auto_subscribe`` ever ran — zero rows in ``kanban_notify_subs``."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title=f"orphaned {kind}", assignee=assignee)
        if kind == "blocked":
            kb.block_task(conn, tid, reason="need Scott's approval", kind="needs_input")
        elif kind == "completed":
            kb.complete_task(conn, tid, summary="done")
        else:
            raise ValueError(kind)
        return tid
    finally:
        conn.close()


def test_blocked_needs_input_with_zero_subs_reaches_home_channel(tmp_path, monkeypatch):
    """The motivating case: an approval-gate block with no subscriber must still page
    the home channel, not time out silently."""
    db_path = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_unsubscribed_task(kind="blocked")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["chat_id"] == "+491234567890"
    assert tid in adapter.sent[0]["text"]

    # A one-time subscription row now exists (so retries/GC see this task as covered);
    # notify-list style readers must see it, matching the pattern other auto-subscribe
    # callers already use (no new, third mechanism).
    conn = kbc.connect()
    try:
        subs = kbn.list_notify_subs(conn, task_id=tid)
    finally:
        conn.close()
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "+491234567890"
    assert subs[0]["delivery_mode"] == "notify"  # passive push only, never a session wake


def test_completed_with_zero_subs_does_not_spam_home_channel(tmp_path, monkeypatch):
    """Routine completion of a programmatically-created task must stay silent — the
    fallback is scoped to kinds that need a human, not every board event."""
    db_path = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _create_unsubscribed_task(kind="completed")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == []


def test_no_home_channel_configured_leaves_task_unsubscribed(tmp_path, monkeypatch):
    """No enabled platform has a home channel: the fallback must not invent a
    destination or write a subscription row pointing nowhere."""
    db_path = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_unsubscribed_task(kind="blocked")

    adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == []
    conn = kbc.connect()
    try:
        subs = kbn.list_notify_subs(conn, task_id=tid)
    finally:
        conn.close()
    assert subs == []


def test_real_subscription_wins_the_race_no_duplicate(tmp_path, monkeypatch):
    """A task that already has a real (non-home-channel) subscriber must never also get
    a fallback row — the race guard in add_fallback_notify_subs must hold."""
    db_path = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="already watched", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.block_task(conn, tid, reason="need Scott's approval", kind="needs_input")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    conn = kbc.connect()
    try:
        subs = kbn.list_notify_subs(conn, task_id=tid)
    finally:
        conn.close()
    assert len(subs) == 1
    assert subs[0]["chat_id"] == "chat-1"
