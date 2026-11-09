"""hermes cron list/doctor must stop reporting a Bot Chat delivery as "still in progress"
once its live mailbox receipt has actually settled (t_f3f9e75d, backup-watchdog).

``_record_delivery_verification`` stamps ``last_delivery_queued`` once, at the moment a job's
OWN run ends. The target profile's agent can take seconds to minutes longer to actually process
and reply to the admitted Bot Chat turn, by which point the firing job's process has already
exited -- nothing re-checked the receipt afterward, so the stale snapshot stuck around forever.
"""
import uuid

from tools import bot_live_delivery as mailbox


def _seed_receipt(home, delivery_id, *, status):
    with mailbox._locked(home) as root:
        mailbox._write(root / f"{delivery_id}.json", dict(
            id=delivery_id, delivery_id=delivery_id, status=status,
            profile_home=str(home), session_id="local-bot", message="payload",
            admission_id="adm-" + delivery_id[:8], reply="done" if status == "settled" else "",
            error="", reason=""))


def test_refresh_clears_a_settled_receipt_and_marks_the_job_ok(tmp_path, monkeypatch):
    from cron import jobs as J
    from cron import scheduler_delivery as delivery

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = J.create_job(prompt=None, schedule="every 1h", name="watchdog", script="true",
                        no_agent=True, deliver="bot-chat")
    delivery_id = uuid.uuid4().hex
    J.update_job(job["id"], {
        "last_status": "delivery_queued",
        "last_delivery_queued": {"bot-chat:(own)": {"status": "queued", "delivery_id": delivery_id}},
    })
    _seed_receipt(tmp_path.resolve(), delivery_id, status="settled")

    refreshed = delivery.refresh_queued_bot_chat_deliveries(J.get_job(job["id"]) or {})

    assert refreshed is not None
    assert refreshed["last_delivery_queued"] is None
    assert refreshed["last_status"] == "ok"
    # Persisted, not just returned -- a second independent read agrees.
    persisted = J.get_job(job["id"])
    assert persisted is not None
    assert persisted["last_status"] == "ok"


def test_refresh_leaves_a_genuinely_still_queued_receipt_alone(tmp_path, monkeypatch):
    from cron import jobs as J
    from cron import scheduler_delivery as delivery

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = J.create_job(prompt=None, schedule="every 1h", name="watchdog", script="true",
                        no_agent=True, deliver="bot-chat")
    delivery_id = uuid.uuid4().hex
    receipt = {"bot-chat:(own)": {"status": "queued", "delivery_id": delivery_id}}
    J.update_job(job["id"], {"last_status": "delivery_queued", "last_delivery_queued": receipt})
    _seed_receipt(tmp_path.resolve(), delivery_id, status="queued")  # owner hasn't replied yet

    assert delivery.refresh_queued_bot_chat_deliveries(J.get_job(job["id"]) or {}) is None
    persisted = J.get_job(job["id"])
    assert persisted is not None
    assert persisted["last_delivery_queued"] == receipt


def test_list_jobs_self_heals_a_settled_bot_chat_delivery(tmp_path, monkeypatch):
    """The actual chokepoint both ``cron list`` and ``cron doctor`` read through."""
    from cron import jobs as J

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = J.create_job(prompt=None, schedule="every 1h", name="watchdog", script="true",
                        no_agent=True, deliver="bot-chat")
    delivery_id = uuid.uuid4().hex
    J.update_job(job["id"], {
        "last_status": "delivery_queued",
        "last_delivery_queued": {"bot-chat:(own)": {"status": "queued", "delivery_id": delivery_id}},
    })
    _seed_receipt(tmp_path.resolve(), delivery_id, status="settled")

    [listed] = [j for j in J.list_jobs() if j["id"] == job["id"]]
    assert listed["last_status"] == "ok"
    assert listed["last_delivery_queued"] is None


def test_doctor_flags_a_bot_chat_delivery_stuck_well_past_grace(monkeypatch):
    from datetime import timedelta
    from hermes_cli import cron as cron_cli
    from hermes_time import now

    stale_run = (now() - timedelta(hours=2)).isoformat()
    job = {"id": "de706118e09e", "name": "backup-watchdog", "last_run_at": stale_run,
           "last_delivery_queued": {"bot-chat:default": {"status": "queued", "delivery_id": "a" * 32}}}
    issues = cron_cli._cron_doctor_issues_for_job(job)
    assert any("stuck 'queued'" in issue for issue in issues)


def test_doctor_does_not_flag_a_freshly_queued_delivery(monkeypatch):
    from hermes_cli import cron as cron_cli
    from hermes_time import now

    job = {"id": "de706118e09e", "name": "backup-watchdog", "last_run_at": now().isoformat(),
           "last_delivery_queued": {"bot-chat:default": {"status": "queued", "delivery_id": "a" * 32}}}
    issues = cron_cli._cron_doctor_issues_for_job(job)
    assert not any("stuck 'queued'" in issue for issue in issues)
