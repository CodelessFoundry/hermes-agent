"""Kanban decomposer — fan a triage task out into a graph of child tasks.

Invoked by ``hermes kanban decompose [task_id | --all]`` and the gateway
dispatcher's auto-decompose path. Reads the profile roster (with
descriptions), asks the auxiliary LLM for a task graph in JSON, then
atomically creates the children, links them under the root, and flips the
root ``triage -> todo``. The root stays alive as parent of every leaf child so
it wakes back up when the graph completes and its assignee (the orchestrator
profile) can judge completion and add more work.

Mirrors ``kanban_specify`` (lazy aux import, lenient parse, never raises on
expected failures). ``fanout=false`` collapses to the ``specify`` behaviour
(tighten + promote, no children), making ``decompose`` a strict superset.
Unknown assignees are rewritten to ``default_assignee`` — a child NEVER ends
up with ``assignee=None``.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import sys
import types
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import profiles as profiles_mod
from hermes_cli.kanban_specify import (
    _call_aux, _extract_json_blob, _load_triage_task, _task_prompt_fields, _title_body,
)
from hermes_cli.kanban_specify import _profile_author as _specify_author

logger = logging.getLogger(__name__)


# --- live-grant skip (t_322addbd) ------------------------------------------
# Scott's "at minimum" ask out of t_07532d30 round 2: don't fan a triage card
# out into children when it is already holding a live, unspent board grant.
# A grant lives on the ORIGINAL task_id's event history
# (board_grants.find_board_grant reads task_events WHERE task_id=<that id>);
# decomposing moves the actual work onto fresh child task_ids that have no
# history of their own, so the grant becomes invisible and the child just
# re-asks Scott for something he already approved. Skipping decomposition
# entirely is lower-risk than trying to carry the grant forward (see
# grant_inheritance.py in ~/.hermes/infra/policy for that alternative, which
# this card deliberately does NOT touch or depend on).
#
# ~/.hermes/infra/policy is an OPTIONAL install, not a hermes-agent
# dependency, so this is a soft integration: if board_grants isn't
# importable, the check below always returns "no skip" and decomposition
# proceeds exactly as it did before this existed. It can only ever cause a
# card to decompose a little later than it otherwise would (once the grant
# that would have been wasted has expired or been spent) — never cause a
# card to decompose that previously wouldn't have, and never touch who may
# approve anything.
_POLICY_DIR = os.path.expanduser("~/.hermes/infra/policy")


def _board_grants_module():
    """Best-effort import of the optional infra-policy ``board_grants``
    helper. Returns ``None`` (never raises) when it isn't installed."""
    if _POLICY_DIR not in sys.path:
        sys.path.insert(0, _POLICY_DIR)
    try:
        return importlib.import_module("board_grants")
    except Exception:
        logger.debug(
            "decompose: board_grants not importable (%s) — skipping the "
            "live-grant pre-decomposition check", _POLICY_DIR, exc_info=True,
        )
        return None


def _last_block_reason(task_id: str) -> Optional[str]:
    """The ``reason`` text of ``task_id``'s most recent ``blocked`` /
    ``block_loop_detected`` event, or ``None`` if it was never blocked.
    Read through the SAME board connection ``decompose_task`` itself uses
    (``kbc.connect_closing()``), so this always looks at the board the
    decomposer is about to act on rather than re-deriving board context
    separately."""
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind IN ('blocked', 'block_loop_detected') "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        return (json.loads(row[0]) or {}).get("reason") or ""
    except Exception:
        return ""


def _live_grant_skip_reason(task_id: str) -> Optional[str]:
    """Non-``None`` (a human-readable reason) when ``task_id``'s last block
    reason carries an ``APPROVAL-REQUEST`` token Scott has already answered
    with a live, unspent board grant — meaning decomposition should be
    skipped. Fails open (returns ``None``, i.e. "don't skip") on anything
    that stops it from answering with confidence: no ``board_grants``
    module, no block history, no parseable token, or an error reading the
    board — in every one of those cases decomposition proceeds exactly as
    it always has, nothing new is prevented.
    """
    bg = _board_grants_module()
    if bg is None:
        return None
    reason = _last_block_reason(task_id)
    if not reason:
        return None
    try:
        # Reuse board_grants' own token parser rather than re-implementing it
        # (same reuse pattern grant_inheritance.py uses for this module).
        tokens = bg._parse_tokens(reason)
    except Exception:
        logger.warning("decompose: failed to parse block tokens for %s", task_id, exc_info=True)
        return None
    for verb, kind, ident in tokens:
        action = types.SimpleNamespace(verb=verb, kind=kind, id=ident, target_host=None)
        try:
            # audit=False: this is a HYPOTHETICAL pre-decomposition question
            # ("would a grant be minted"), not a real evaluate()/consume()
            # decision -- same contract board_grants.find_board_grant
            # documents for the approval clock's own pre-checks.
            grant = bg.find_board_grant(action, task_id=task_id, run_id=None, audit=False)
        except Exception:
            logger.warning(
                "decompose: find_board_grant raised while checking %s for a "
                "live grant on %s/%s:%s — not skipping", task_id, verb, kind, ident,
                exc_info=True,
            )
            continue
        if grant is not None:
            scope = f"{verb}/{kind}" + (f":{ident}" if ident else "")
            return (
                f"task holds a live, unspent board grant for {scope} (from its "
                "last block reason) — skipping decomposition so the already-"
                "approved request isn't fanned out into children that would "
                "have to ask again"
            )
    return None


_SYSTEM_PROMPT = """You are the Kanban decomposer for the Hermes Agent board.

A user dropped a rough idea into the Triage column. Your job is to break it
into a small graph of concrete child tasks and route each one to the best-
matching profile from the available roster.

You will be given:
  - The original task title and body
  - The list of available profiles (each with name + description)
  - The fallback "default_assignee" used when no profile fits

Output a single JSON object with this exact shape:

  {
    "fanout": true,
    "rationale": "<one sentence on why this decomposition>",
    "tasks": [
      {
        "title": "<concrete task title, imperative voice, <= 80 chars>",
        "body":  "<detailed spec for the worker on this child task>",
        "assignee": "<profile name from the roster, or null for default>",
        "parents": [<int>, ...]
      },
      ...
    ]
  }

Rules:
  - "parents" is a list of INDICES (0-based) into this same "tasks" list,
    expressing actual data dependencies. Tasks with no parents run in
    PARALLEL. Tasks with parents wait until every parent completes.
  - Prefer parallelism. If two tasks can be done independently, give
    them no parents so the dispatcher fans them out at once.
  - Use 2-6 tasks for normal work. Don't create 20 tiny tasks. Don't
    cram everything into 1 task.
  - Pick assignees from the roster by matching the task to the profile's
    DESCRIPTION (not just the name). When nothing matches well, use null
    and the system will route to the default_assignee.
  - Each child task body is what a fresh worker will read with no other
    context — be specific about goal, approach, and acceptance criteria.

When the task is genuinely a single unit of work (no useful decomposition),
return:

  {
    "fanout": false,
    "rationale": "<one sentence>",
    "title": "<tightened title>",
    "body":  "<concrete spec for a single worker>",
    "assignee": "<profile name from the roster, or null for default>"
  }

In that case the task stays as one work item, just with a tightened spec and
a concrete assignee. If no profile fits, use null and the system will route to
the default_assignee.

No preamble, no closing remarks, no code fences. Output only the JSON object.
"""


_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Body:
{body}

Available profiles (assignees you may pick from):
{roster}

Default assignee (used when no profile fits a task): {default_assignee}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class DecomposeOutcome:
    """Result of decomposing a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    fanout: bool = False
    child_ids: list[str] | None = None
    new_title: Optional[str] = None


def _profile_author() -> str:
    """Mirror of ``hermes_cli.kanban._profile_author``."""
    return _specify_author("decomposer")


def _resolve_profile_from_cfg(cfg: dict, key: str, *, fallback: Optional[str] = None) -> str:
    """``kanban.<key>`` if it names an existing profile, else ``fallback``
    (the root task's own assignee) if that does, else the active default
    profile — so a task is never stranded for lack of an owner.
    ``orchestrator_profile`` owns the root after fan-out; ``default_assignee``
    catches children the decomposer can't route.

    The root's assignee sits before the active profile because the decomposer
    runs inside whatever profile hosts the dispatcher — an operator's
    credential-less incognito profile, say — and that profile must never
    silently become the owner of work the card was assigned away from (#114294).
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get(key) or "").strip()
    for candidate in (explicit, (fallback or "").strip()):
        if candidate:
            try:
                if profiles_mod.profile_exists(candidate):
                    return candidate
            except Exception:
                pass
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return "default"


def _build_roster() -> tuple[list[dict], set[str]]:
    """``(roster_for_prompt, valid_assignee_names)``; entries are
    ``{name, description, has_description}``."""
    try:
        all_profiles = profiles_mod.list_profiles()
    except Exception as exc:
        logger.warning("decompose: failed to list profiles: %s", exc)
        return [], set()
    roster = []
    for p in all_profiles:
        desc = (p.description or "").strip()
        roster.append({
            "name": p.name,
            "description": desc or f"(no description; profile named {p.name!r})",
            "has_description": bool(desc),
        })
    return roster, {p.name for p in all_profiles}


def _format_roster(roster: list[dict]) -> str:
    if not roster:
        return "  (no profiles installed — decomposer cannot route work)"
    return "\n".join(
        f"  - {entry['name']}{'' if entry['has_description'] else ' ⚠ undescribed'}: {entry['description']}"
        for entry in roster
    )


def _normalize_assignee_choice(assignee: object, *, default_assignee: str, valid_names: set[str]) -> str:
    """A valid assignee, else ``default_assignee`` — promoted work is never
    left unassigned."""
    if not isinstance(assignee, str) or not assignee.strip():
        return default_assignee
    chosen = assignee.strip()
    return chosen if chosen in valid_names else default_assignee


@dataclass
class _Routing:
    """Config-derived routing context for one decomposition."""

    orchestrator: str
    default_assignee: str
    auto_promote: bool
    roster: list[dict]
    valid_names: set[str]


def _load_routing(*, root_assignee: Optional[str] = None) -> _Routing:
    from hermes_cli.config import load_config_readonly
    try:
        cfg = load_config_readonly()
    except Exception:  # decompose_task promises ok=False, never a raise, on config trouble
        cfg = {}
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    roster, valid_names = _build_roster()
    return _Routing(
        orchestrator=_resolve_profile_from_cfg(cfg, "orchestrator_profile", fallback=root_assignee),
        default_assignee=_resolve_profile_from_cfg(cfg, "default_assignee", fallback=root_assignee),
        auto_promote=bool(kanban_cfg.get("auto_promote_children", True)),
        roster=roster,
        valid_names=valid_names,
    )


def _apply_single(task: kb.Task, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    """``fanout=false``: single-task spec promotion (same effect as specify)."""
    title_val, body_val = _title_body(parsed)
    assignee_val = None
    if not task.assignee:
        assignee_val = _normalize_assignee_choice(
            parsed.get("assignee"), default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
    if title_val is None and body_val is None:
        return DecomposeOutcome(task.id, False, "decomposer returned fanout=false with no title/body")
    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn, task.id, title=title_val, body=body_val, assignee=assignee_val, author=author,
        )
    if not ok:
        return DecomposeOutcome(task.id, False, "task moved out of triage before promotion")
    return DecomposeOutcome(task.id, True, "single task (no fanout)", fanout=False, new_title=title_val)


def _clean_children(task_id: str, raw_tasks: list, routing: _Routing) -> tuple[list[dict], str]:
    """Validate/normalise the LLM's ``tasks`` list; ``(children, "")`` or ``([], reason)``.
    Unknown assignees route to the default; never assignee=None."""
    children: list[dict] = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return [], f"tasks[{idx}] is not an object"
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return [], f"tasks[{idx}].title is missing or empty"
        body = entry.get("body")
        assignee = entry.get("assignee")
        chosen = _normalize_assignee_choice(
            assignee, default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
        if isinstance(assignee, str) and assignee.strip() and assignee.strip() not in routing.valid_names:
            logger.info(
                "decompose: task %s child %d picked unknown assignee %r — "
                "routing to default_assignee %r",
                task_id, idx, assignee, routing.default_assignee,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        children.append({
            "title": title.strip()[:200],
            "body": body.strip() if isinstance(body, str) else "",
            "assignee": chosen,
            # Drop non-int, out-of-range and self parent indices.
            "parents": [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx],
        })
    return children, ""


def _apply_fanout(task_id: str, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(task_id, False, "decomposer returned fanout=true with empty tasks list")
    children, reason = _clean_children(task_id, raw_tasks, routing)
    if reason:
        return DecomposeOutcome(task_id, False, reason)
    try:
        with kbc.connect_closing() as conn:
            child_ids = decompose_triage_task(
                conn,
                task_id,
                root_assignee=routing.orchestrator,
                children=children,
                author=author,
                auto_promote=routing.auto_promote,
            )
    except ValueError as exc:
        return DecomposeOutcome(task_id, False, f"DB rejected graph: {exc}")
    except Exception as exc:
        logger.exception("decompose: DB error on task %s", task_id)
        return DecomposeOutcome(task_id, False, f"DB error: {type(exc).__name__}")
    if child_ids is None:
        return DecomposeOutcome(task_id, False, "task already decomposed or moved out of triage")
    return DecomposeOutcome(
        task_id, True, f"decomposed into {len(child_ids)} children", fanout=True, child_ids=child_ids,
    )


def decompose_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks. Expected failures
    (not in triage, no aux client, API error, malformed/empty reply) surface
    as ``ok=False``."""
    task, reason = _load_triage_task(task_id)
    if task is None:
        return DecomposeOutcome(task_id, False, reason)

    # t_322addbd: don't fan a card out while it's still holding an approval
    # Scott already gave it. Checked BEFORE the aux LLM call (not just before
    # decompose_triage_task) so a card stuck here on repeated auto-decompose
    # ticks doesn't burn an LLM call every time for nothing.
    skip_reason = _live_grant_skip_reason(task_id)
    if skip_reason:
        logger.info("decompose: skipping %s — %s", task_id, skip_reason)
        return DecomposeOutcome(task_id, False, skip_reason)

    routing = _load_routing(root_assignee=task.assignee)
    raw, reason = _call_aux(
        "decompose", task_id, aux_task="kanban_decomposer", system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(
            **_task_prompt_fields(task),
            roster=_format_roster(routing.roster),
            default_assignee=routing.default_assignee,
        ),
        max_tokens=4000, timeout=timeout or 180, log=logger,
    )
    if raw is None:
        return DecomposeOutcome(task_id, False, reason)

    parsed = _extract_json_blob(raw, _FENCE_RE)
    if parsed is None:
        return DecomposeOutcome(task_id, False, "LLM returned malformed JSON")

    audit_author = author or _profile_author()
    if not parsed.get("fanout"):
        return _apply_single(task, parsed, routing, audit_author)
    return _apply_fanout(task_id, parsed, routing, audit_author)


#: Per-run bookkeeping events, never a board decision. ``gateway/session_kanban.py``
#: appends ``worker_bound``/``owner_admitted`` as a run starts and ALWAYS appends
#: ``worker_result`` when it ends — including the very run that just wrote
#: ``block_loop_detected`` mid-turn before exiting. ``kanban_db_dispatch.py`` adds
#: ``claimed``/``spawned``/``heartbeat`` the same way. None of these represent a
#: human or the board acting on the card; they are the dispatcher narrating its
#: own process lifecycle and must never be read as "something moved this card on".
_RUN_LIFECYCLE_EVENT_KINDS = frozenset({
    "claimed", "spawned", "owner_admitted", "worker_bound", "heartbeat", "worker_result",
})


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column, EXCLUDING cards that
    landed there via the unblock-loop breaker (``block_loop_detected``).

    Added 2026-10-22 (t_48b8d912). ``_route_block`` in kanban_db.py routes a
    card to ``triage`` at ``BLOCK_RECURRENCE_LIMIT`` purely to break an
    infinite block->unblock->re-block cycle — its own docstring says this is
    "for a human", not for the LLM decomposer. But this function previously
    returned EVERY ``status='triage'`` row with no regard for why it got
    there, so the auto-decompose tick swept up loop-broken cards exactly like
    genuinely-underspecified ones.

    Measured live on t_ab79a0c2 (a fully-specified, Scott-approved, 6-minute
    reversible UDM-SE test): the card hit the loop breaker twice while
    structurally blocked on an unrelated grant-matching bug, and both times
    the auto-decomposer rewrote its title and body and added stop
    conditions the ORIGINAL approved instructions never had — which is what
    actually stalled the approved test, on top of the grant bug itself.
    A card stuck on an approval/grant problem needs that problem fixed (or a
    human decision), not an LLM's best guess at a different task.

    The most recent *substantive* ``task_events`` row is the signal: right
    after ``_route_block`` writes ``block_loop_detected`` the card sits in
    triage with that as its newest decision-relevant event, until something
    else (a comment, an unblock, a manual promote) moves it on. A genuinely
    fresh triage card's newest substantive event is ``created``/``commented``
    /etc. — untouched by this.

    FIXED 2026-11-09 (t_42a8ab05, real incident t_982a6f2f on the tech
    board): the original check used the literal last ``task_events`` row
    (``ORDER BY id DESC LIMIT 1``), which assumed ``block_loop_detected``
    would still be it. It never is in production — every run, including the
    one that just wrote ``block_loop_detected``, ends by appending
    ``worker_result`` (see ``_RUN_LIFECYCLE_EVENT_KINDS`` above), so the real
    newest row for a just-loop-broken card is always ``worker_result``, not
    ``block_loop_detected``. The exclusion was therefore a near-total no-op:
    it only fired if nothing ever ran that card's dispatcher lifecycle again,
    which does not happen. t_982a6f2f hit the loop breaker, its run ended
    (appending ``worker_result`` as always), and three minutes later the
    unprotected tick swept it into the decomposer anyway — fanning it into 4
    child tasks and burying the still-unresolved
    ``APPROVAL-REQUEST: create/dns_record`` decision inside a rewritten
    child body. Fixed by skipping run-lifecycle bookkeeping kinds and asking
    for the newest event that is NOT one of them, so a card whose last
    *meaningful* event is ``block_loop_detected`` is excluded regardless of
    how many bookkeeping rows the dispatcher appended afterward.

    This only narrows the AUTOMATIC tick's candidate pool. A human can still
    force it via ``hermes kanban decompose <id>`` (calls ``decompose_task``
    directly, bypassing this list) if a loop-broken card genuinely does need
    re-specification.
    """
    placeholders = ",".join("?" * len(_RUN_LIFECYCLE_EVENT_KINDS))
    with kbc.connect_closing() as conn:
        rows = kb.list_tasks(conn, status="triage", tenant=tenant, limit=1000)
        ids = []
        for row in rows:
            event = conn.execute(
                f"SELECT kind FROM task_events WHERE task_id = ? "
                f"AND kind NOT IN ({placeholders}) "
                "ORDER BY id DESC LIMIT 1",
                (row.id, *_RUN_LIFECYCLE_EVENT_KINDS),
            ).fetchone()
            if event and event["kind"] == "block_loop_detected":
                logger.debug(
                    "kanban auto-decompose: skipping %s, triage via "
                    "block_loop_detected (needs a human, not a rewrite)",
                    row.id,
                )
                continue
            ids.append(row.id)
    return ids
