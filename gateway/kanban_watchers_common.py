"""Plumbing shared by the kanban notifier and dispatcher loops.

Thread offload, board enumeration, live-config coercers and the singleton
dispatcher lock live here so the notifier, dispatcher and mixin modules read
them from one place.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextvars import Context
from pathlib import Path
from typing import Any, Callable, Optional

# Keep the logger name run.py used so extracted log records are unchanged.
logger = logging.getLogger("gateway.run")


def _run_in_fresh_context(func: Callable[..., Any], /, *args: Any) -> Any:
    """Run *func* in an empty ``Context`` so request-local ContextVars stay behind.

    ``asyncio.to_thread`` copies the caller's context; a lingering
    ``delegate_task`` child marker would make ``write_txn`` false-trip for
    these process-owned writers. An empty Context keeps the DB guard intact
    for real children without exempting dispatcher writes.
    """
    return Context().run(func, *args)


async def _to_thread_process_service(func: Callable[..., Any], /, *args: Any) -> Any:
    """Offload blocking process-service work without inheriting request ContextVars."""
    return await asyncio.to_thread(_run_in_fresh_context, func, *args)


async def _tracked_to_thread_process_service(gateway: Any, func: Callable[..., Any], /, *args: Any) -> Any:
    """Like :func:`_to_thread_process_service`, but visible to shutdown's drain for the
    call's whole duration (same registration ``run_turn_hygiene`` uses for detached
    compaction workers — see its ``_track_deferred_agent_worker`` call).

    The kanban watcher loop (notifier/dispatcher) runs on the gateway's event loop, so its
    OWN asyncio Task is in ``_background_tasks`` and gets cancelled on shutdown — but
    cancelling that Task does not stop a blocking call already handed to the loop's default
    executor (``asyncio.to_thread``'s pool, not ``self._executor``). ``auto_decompose_tick``
    is the sharp edge: it makes a real outbound auxiliary-LLM call (tens of seconds) to
    decompose a triage card, on the SAME default executor ``_active_cron_job_count`` /
    ``_active_api_run_count`` / ``_active_deferred_agent_worker_count`` were each added to
    cover for their own categories (cron ticks, API-server runs, detached hygiene workers) —
    none of the three counts this fourth category. A SIGUSR1 restart arriving mid-tick saw
    drain report 0 everywhere, closed state.db, and the tick's eventual completion/retry
    landed its write against the closed handle (teardown/worker race, #94736; root-caused in
    t_6b3334b8 against a live Nov 7 occurrence: a decompose started at 03:02:34, survived the
    03:02:58 close with a mid-flight '[Errno 9] Bad file descriptor' when final-cleanup
    reaped the shared aux HTTP client out from under it, and the warning landed at 03:03:31
    once the immediate retry finished). Wrapping the call site's future as a deferred agent
    worker (``agent=None``: this is raw thread work, not an interruptible agent turn — the
    same no-op-safe shape ``_interrupt_deferred_agent_workers`` already special-cases) makes
    the shutdown gate wait for it like it already waits for cron/API/hygiene work, instead of
    skipping straight to close.
    """
    future = asyncio.ensure_future(_to_thread_process_service(func, *args))
    track = getattr(gateway, "_track_deferred_agent_worker", None)
    if callable(track):
        track(future, None)
    return await future


async def _to_thread_dedicated(executor: Any, func: Callable[..., Any], /, *args: Any) -> Any:
    """Like :func:`_to_thread_process_service`, but runs on *executor* instead of the loop's
    shared default ``ThreadPoolExecutor`` (``asyncio.to_thread``'s pool).

    Added 11 Nov (t_67e88746): the embedded kanban dispatcher's tick was silently starved for
    ~88 minutes because it shared that default pool (``min(32, cpu_count + 4)`` — 8 threads
    on a 4-core box) with a long list of unrelated gateway subsystems (session housekeeping,
    MCP reconcile, channel-directory builds, image staging, etc.) that also offload bare
    blocking work via ``asyncio.to_thread()`` with no timeout. No exception and no log line
    were produced — the dispatcher's "stuck: N consecutive ticks" WARNING only fires AFTER a
    tick completes and updates its own counter, so a tick that never got a thread never even
    starts that clock. Giving the dispatcher watcher its own dedicated executor (see
    ``_kanban_dispatcher_watcher``) makes it immune to being starved by anything else in the
    gateway, at the cost of nothing shared — the dispatcher's own tick work is already
    strictly sequential (zombie reap, then auto-decompose, then dispatch), so it never needs
    more than one thread at a time.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _run_in_fresh_context, func, *args)


async def _tracked_to_thread_dedicated(gateway: Any, executor: Any, func: Callable[..., Any],
                                        /, *args: Any) -> Any:
    """Like :func:`_tracked_to_thread_process_service`, but offloads onto *executor* instead
    of the shared default pool — see :func:`_to_thread_dedicated` for why."""
    future = asyncio.ensure_future(_to_thread_dedicated(executor, func, *args))
    track = getattr(gateway, "_track_deferred_agent_worker", None)
    if callable(track):
        track(future, None)
    return await future


def _list_boards(kb: Any) -> list:
    """Enumerate live boards; fall back to the default board when listing fails."""
    try:
        return kb.list_boards(include_archived=False)
    except Exception:
        return [kb.read_board_metadata(kb.DEFAULT_BOARD)]


def _board_slugs(kb: Any) -> list:
    return [b.get("slug") or kb.DEFAULT_BOARD for b in _list_boards(kb)]


def _positive_int_setting(kanban_cfg: dict, key: str) -> Optional[int]:
    """Parse an optional ``kanban.<key>`` int cap; None when unset or invalid (< 1 is invalid)."""
    raw = kanban_cfg.get(key)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("kanban dispatcher: invalid kanban.%s=%r; ignoring", key, raw)
        return None
    if value < 1:
        logger.warning("kanban dispatcher: kanban.%s=%r is below 1; ignoring", key, raw)
        return None
    logger.info("kanban dispatcher: %s=%d", key, value)
    return value


def _resolve_auto_decompose_settings(load_config: Callable[[], Any]) -> tuple[bool, int]:
    """Live (enabled, per_tick) auto-decompose settings, re-read every dispatcher tick.

    Fails safe: a config read error returns ``(False, 3)`` rather than
    re-enabling a feature the user turned off. ``per_tick`` is clamped to ``>= 1``.

    Read fresh from config on every dispatcher tick (#49638) so that flipping ``kanban.auto_decompose:
    false`` to STOP runaway fan-out takes effect on the next tick instead of requiring a gateway restart.
    Auto-decompose is a safety toggle — a user who sees it create and launch tasks they didn't intend
    reaches for this flag to halt it, and a stale boot-captured value silently ignoring that change is the
    bug reported in #49638.
    """
    try:
        cfg = load_config()
    except Exception:
        return False, 3
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    try:
        per_tick = int(kcfg.get("auto_decompose_per_tick", 3) or 3)
    except (TypeError, ValueError):
        per_tick = 3
    return bool(kcfg.get("auto_decompose", True)), max(per_tick, 1)


def _gc_retention_days() -> int:
    """``kanban.done_sub_retention_days`` (default 30; 0 disables), re-read per sweep; fails safe to 30."""
    try:
        from hermes_cli.config import load_config

        return int(((load_config() or {}).get("kanban") or {}).get("done_sub_retention_days", 30))
    except Exception:
        return 30


def _kanban_dispatch_allowed() -> bool:
    """False while the global emergency stop (`hermes pause`) is engaged.

    Checked every tick before spawning, so a pause applies on the next tick;
    in-flight workers are never touched. Fails open if estop is unimportable.
    """
    try:
        from agent.estop import check_paused
    except ImportError:
        return True
    return not check_paused("kanban", logger)


def _acquire_singleton_lock(lock_path) -> tuple[Optional[object], str]:
    """Take the exclusive, non-blocking advisory lock for the sole dispatcher.

    Only one gateway machine-wide may run the embedded dispatcher: concurrent
    dispatchers double reclaim frequency and claim events, and with
    ``wal_autocheckpoint=0`` concurrent manual checkpoints can corrupt index
    pages. ``dispatch_in_gateway`` is the primary control; this is the backstop.

    Returns ``(handle, "held")`` (release via :func:`_release_singleton_lock`),
    ``(None, "contended")`` when another process holds it (caller must NOT
    dispatch), or ``(None, "unavailable")`` when locking cannot be performed
    (caller falls back to config control).
    """
    try:
        from gateway.status import _try_acquire_file_lock  # deferred; same package
    except ImportError:
        return None, "unavailable"
    try:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(str(lock_path), "a+", encoding="utf-8")  # windows-footgun: ok (append-mode lock handle, write not read)
    except OSError:
        return None, "unavailable"
    if not _try_acquire_file_lock(handle):
        handle.close()
        return None, "contended"
    return handle, "held"


def _release_singleton_lock(handle) -> None:
    """Release a lock acquired via :func:`_acquire_singleton_lock`."""
    if handle is None:
        return
    with contextlib.suppress(Exception):
        from gateway.status import _release_file_lock

        _release_file_lock(handle)
    with contextlib.suppress(Exception):
        handle.close()
