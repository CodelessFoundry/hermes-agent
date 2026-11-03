"""The ``gateway.ready`` frame send (handle_ws's very first write) can lose the exact same
benign-disconnect race as any later send (t_ee5439b9), just one send earlier in the connection's
life: the peer vanishes between ``ws.accept()`` and this write (reconnect-storm double-connect,
tab closed instantly, etc). Before t_3366ecde, ``handle_ws`` logged this unconditionally at ERROR
whenever ``write_async`` returned False, regardless of *why* -- so the already-correct DEBUG/
WARNING split inside ``_safe_send_many`` (t_ee5439b9) got silently overridden back to ERROR for
every ready-frame send, benign or not.

``WSTransport.closed_benign`` exposes the classification so the ready-frame caller can match
``_safe_send_many``'s own verdict: benign peer-already-gone -> DEBUG (routine, not actionable);
anything else -> stays ERROR, because a genuinely live client that never got its ready handshake
is user-visible and must keep paging.
"""

from __future__ import annotations

import asyncio
import logging

from starlette.websockets import WebSocketDisconnect

from tui_gateway.ws import WSTransport


class _FakeWS:
    def __init__(self, to_raise: Exception) -> None:
        self._to_raise = to_raise

    async def send_text(self, line: str) -> None:
        raise self._to_raise


def _send_ready_frame(exc: Exception) -> WSTransport:
    async def _run() -> WSTransport:
        loop = asyncio.get_running_loop()
        ws = _FakeWS(exc)
        transport = WSTransport(ws, loop, peer="10.2.2.240:54892")
        ready_ok = await transport.write_async({"jsonrpc": "2.0", "method": "event"})
        assert ready_ok is False  # the frame genuinely failed to go out either way
        return transport

    return asyncio.run(_run())


def test_benign_disconnect_on_ready_frame_is_flagged_benign() -> None:
    transport = _send_ready_frame(WebSocketDisconnect(code=1006))
    assert transport.closed is True
    assert transport.closed_benign is True


def test_asgi_send_after_close_on_ready_frame_is_flagged_benign() -> None:
    exc = RuntimeError(
        "Unexpected ASGI message 'websocket.send', after sending 'websocket.close' or response "
        "already completed."
    )
    transport = _send_ready_frame(exc)
    assert transport.closed is True
    assert transport.closed_benign is True


def test_real_transport_fault_on_ready_frame_is_not_flagged_benign() -> None:
    """Control case: a genuine fault (client was live, send still failed) must NOT be classified
    benign -- this is the shape that should keep paging at ERROR in handle_ws."""
    transport = _send_ready_frame(ConnectionResetError("reset"))
    assert transport.closed is True
    assert transport.closed_benign is False


def test_fresh_transport_defaults_to_not_benign() -> None:
    """Before any send ever happens, closed_benign must default False (not closed yet, and not
    mistakenly inheriting a stale benign flag from a prior connection/instance)."""
    transport = WSTransport(_FakeWS(RuntimeError("unused")), asyncio.new_event_loop(), peer="x")
    assert transport.closed is False
    assert transport.closed_benign is False
