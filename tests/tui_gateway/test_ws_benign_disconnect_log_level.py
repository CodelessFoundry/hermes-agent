"""A WS send racing a client disconnect is routine, not a backend fault (t_ee5439b9).

``_safe_send_many`` can lose the race between a background/concurrent send and the read loop
noticing the peer left: either Starlette raises ``WebSocketDisconnect`` on the send, or uvicorn's
ASGI layer raises a ``RuntimeError`` because it already tore the connection down. Both shapes mean
"the client is gone" -- ``handle_ws``'s own disconnect teardown (session detach/reap) runs
independently of this send failing, so there is nothing actionable here. They must log at DEBUG,
not WARNING; anything else sending through the same except block (a genuine transport fault) must
still log at WARNING so a real problem isn't silenced.
"""

from __future__ import annotations

import asyncio
import logging

from starlette.websockets import WebSocketDisconnect

from tui_gateway.ws import WSTransport, _is_benign_disconnect_send_error


class _FakeWS:
    """A send() that raises whatever the test wants to simulate the race."""

    def __init__(self, to_raise: Exception) -> None:
        self._to_raise = to_raise

    async def send_text(self, line: str) -> None:
        raise self._to_raise


def test_classifier_flags_websocket_disconnect() -> None:
    assert _is_benign_disconnect_send_error(WebSocketDisconnect(code=1006)) is True


def test_classifier_flags_asgi_send_after_close_runtime_error() -> None:
    exc = RuntimeError(
        "Unexpected ASGI message 'websocket.send', after sending 'websocket.close' or response "
        "already completed."
    )
    assert _is_benign_disconnect_send_error(exc) is True


def test_classifier_does_not_flag_an_unrelated_runtime_error() -> None:
    # Same exception class, unrelated message -- must NOT be swallowed as benign.
    assert _is_benign_disconnect_send_error(RuntimeError("event loop is closed")) is False


def test_classifier_does_not_flag_other_exception_types() -> None:
    assert _is_benign_disconnect_send_error(ConnectionResetError("reset")) is False
    assert _is_benign_disconnect_send_error(OSError("broken pipe")) is False


def _run_send_and_capture(exc: Exception, caplog) -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        ws = _FakeWS(exc)
        transport = WSTransport(ws, loop, peer="10.2.2.240:54892")
        with caplog.at_level(logging.DEBUG, logger="tui_gateway.ws"):
            await transport._safe_send_many(["frame"])
        # Either way the transport must be marked dead -- no retry against a gone socket.
        assert transport.closed is True

    asyncio.run(_run())


def test_websocket_disconnect_logs_at_debug_not_warning(caplog) -> None:
    _run_send_and_capture(WebSocketDisconnect(code=1006), caplog)
    levels = [r.levelname for r in caplog.records if r.name == "tui_gateway.ws"]
    assert "WARNING" not in levels
    assert "DEBUG" in levels


def test_asgi_send_after_close_runtime_error_logs_at_debug_not_warning(caplog) -> None:
    exc = RuntimeError(
        "Unexpected ASGI message 'websocket.send', after sending 'websocket.close' or response "
        "already completed."
    )
    _run_send_and_capture(exc, caplog)
    levels = [r.levelname for r in caplog.records if r.name == "tui_gateway.ws"]
    assert "WARNING" not in levels
    assert "DEBUG" in levels


def test_an_unrelated_send_failure_still_warns(caplog) -> None:
    """Control case: a real transport fault must still surface at WARNING."""
    _run_send_and_capture(ConnectionResetError("reset"), caplog)
    levels = [r.levelname for r in caplog.records if r.name == "tui_gateway.ws"]
    assert "WARNING" in levels
