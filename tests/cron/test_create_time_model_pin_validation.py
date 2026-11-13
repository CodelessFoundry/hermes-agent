"""A malformed --model/--provider pin must be rejected at creation time, not only when the
job finally fires (12 Nov retro, t_be7b266e / t_5d757b4d): a cron job that schedules its own
follow-up with model='sonnet-5' (missing the 'claude-' vendor prefix anthropic requires) used
to create successfully and only fail 9 hours later when it ran. create_job/update_job now run
the same cron.scheduler._validate_resolved_model guard before persisting."""
import pytest

from cron import jobs


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestCreateTimeModelPinValidation:
    def test_explicit_provider_with_malformed_pin_rejected_at_creation(self, tmp_cron_dir):
        """The exact incident shape, but with --provider also explicit: must never reach disk."""
        with pytest.raises(RuntimeError, match="sonnet-5"):
            jobs.create_job(
                prompt="Verify Prepper's first ASSESSMENT brief (17:20 re-check)",
                schedule="in 30m",
                model="sonnet-5",
                provider="anthropic",
            )
        # Nothing was persisted: the 9-hour silent-death window this closes.
        assert jobs.load_jobs() == []

    def test_implicit_provider_from_config_model_provider_rejected_at_creation(
        self, tmp_cron_dir, monkeypatch,
    ):
        """The REAL 12 Nov shape: --model only, no --provider. Provider comes from config.yaml's
        model.provider (anthropic), exactly like the dispatcher resolved it 9 hours later."""
        (tmp_cron_dir / "config.yaml").write_text(
            "model:\n  default: claude-opus-5-5\n  provider: anthropic\n", encoding="utf-8",
        )
        monkeypatch.setattr("cron.jobs.get_hermes_home", lambda: tmp_cron_dir)
        with pytest.raises(RuntimeError, match="sonnet-5"):
            jobs.create_job(
                prompt="Verify Prepper's first ASSESSMENT brief (17:20 re-check)",
                schedule="in 30m",
                model="sonnet-5",
            )
        assert jobs.load_jobs() == []

    def test_correctly_prefixed_pin_is_created_normally(self, tmp_cron_dir):
        job = jobs.create_job(
            prompt="fine", schedule="in 30m", model="claude-sonnet-5", provider="anthropic",
        )
        assert job["model"] == "claude-sonnet-5"
        assert jobs.load_jobs() != []

    def test_non_anthropic_provider_is_not_checked(self, tmp_cron_dir):
        """Only anthropic has a required-prefix entry; any other provider's bare model id is a
        job for ITS provider's own rule (none exists today), not this guard."""
        job = jobs.create_job(
            prompt="fine", schedule="in 30m", model="sonnet-5", provider="openrouter",
        )
        assert job["model"] == "sonnet-5"

    def test_unpinned_job_is_not_checked_at_creation(self, tmp_cron_dir):
        """No explicit model pin: nothing for this creation-time guard to validate (the job
        inherits cron.model / model.default at run time, which the dispatch-time guard still
        re-checks every tick regardless)."""
        job = jobs.create_job(prompt="fine", schedule="in 30m")
        assert job["model"] is None

    def test_update_job_edit_to_malformed_pin_rejected(self, tmp_cron_dir):
        """`hermes cron edit --model` is the other place this typo gets typed in; same guard."""
        job = jobs.create_job(
            prompt="fine", schedule="in 30m", model="claude-sonnet-5", provider="anthropic",
        )
        with pytest.raises(RuntimeError, match="sonnet-5"):
            jobs.update_job(job["id"], {"model": "sonnet-5"})
        # The bad edit never landed: the stored record keeps the last-good pin.
        assert jobs.get_job(job["id"])["model"] == "claude-sonnet-5"

    def test_update_job_pinned_true_with_live_main_model_still_validated(
        self, tmp_cron_dir, monkeypatch,
    ):
        """``pinned=True`` auto-fills model/provider from the live main agent model; this guard
        covers that path too (belt-and-suspenders — a live main model should already be sane)."""
        monkeypatch.setattr(jobs, "_main_model_pin", lambda: ("anthropic", "claude-sonnet-5"))
        job = jobs.create_job(prompt="fine", schedule="in 30m", pinned=True)
        assert job["model"] == "claude-sonnet-5"
