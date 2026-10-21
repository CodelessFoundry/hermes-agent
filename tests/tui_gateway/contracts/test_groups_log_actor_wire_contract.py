"""Regression for t_eebd7b6b: ``groups.log`` violated its wire contract for any room event
whose actor was a room member.

``gateway/hosted_room_discussion.py::_member_actor`` always sets ``actor.profile`` (and sets
``display_name``/``connection_id`` when present) — see ``gateway/hosted_rooms.py::_OPTIONAL_ACTOR_FIELDS``,
the table that lets those fields onto an event's actor in the first place. ``RoomActor`` in
``tui_gateway/contracts/groups_bot_relay.py`` only declared ``kind``/``id``, so every
``groups.log`` (and ``groups.send``/``groups.replicate``) result carrying a member-authored event
failed ``check_result`` under ``Result``'s ``extra="forbid"``. It stayed invisible because every
existing ``groups.*`` test either used a ``kind: "user"``/``"system"`` actor (no optional fields)
or called ``srv._methods[...]`` directly, which bypasses ``rpc_dispatch.handle_request`` and so
never runs ``check_result`` at all.
"""

import pytest
from pydantic import ValidationError

from tui_gateway.contracts.groups_bot_relay import GroupsLogResult
from tui_gateway.contracts.registry import METHODS, check_result

_BASE_EVENT = {
    "room_id": "room-1", "seq": 1, "event_id": "evt-1", "kind": "message.user",
    "authority_epoch": 1, "payload": {"text": "hi"}, "created_at": 123.0, "idempotent": False,
}
_PAGE = {"cursor": 1, "latest_seq": 1, "has_more": False, "authority": {"gateway_id": "install-1", "epoch": 1}}


@pytest.mark.parametrize("actor", [
    {"kind": "member", "id": "member-1", "profile": "bot-a"},  # always set by _member_actor
    {"kind": "member", "id": "member-1", "profile": "bot-a", "display_name": "Bot A"},
    {"kind": "member", "id": "member-1", "profile": "bot-a", "connection_id": "conn-1"},
    {"kind": "user", "id": "desktop-user"},  # the shape every pre-existing test used
])
def test_groups_log_accepts_every_actor_shape_read_events_can_produce(actor):
    page = {**_PAGE, "events": [{**_BASE_EVENT, "actor": actor}]}
    check_result(METHODS["groups.log"], page)  # raises under HERMES_TEST_ISOLATION on drift
    assert GroupsLogResult.model_validate(page).events[0].actor.id == actor["id"]


def test_groups_log_keeps_strict_result_validation():
    bad_page = {**_PAGE, "events": [{**_BASE_EVENT, "actor": {"kind": "member", "id": "m-1", "unknown_field": "x"}}]}
    with pytest.raises(ValidationError):
        GroupsLogResult.model_validate(bad_page)
