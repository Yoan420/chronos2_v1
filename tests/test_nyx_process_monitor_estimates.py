"""Deterministic tests for observed-reference and explicitly assumed ETAs."""
from datetime import datetime, timezone
import math

import pytest

from nyx_process_monitor.estimates import Estimator


NOW = datetime(2026, 9, 22, 9, tzinfo=timezone.utc).timestamp()


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def active(elapsed=600, **extra):
    return {"id": "job", "title": "unknown.py", "status": "running", "started_at": iso(NOW-elapsed),
            "elapsed_seconds": elapsed, "processes": [{"pid": 10}], "zones": [], **extra}


@pytest.fixture
def estimator(tmp_path):
    return Estimator(tmp_path)


def test_generic_fallback_is_numeric_explicit_and_wide(estimator):
    result = estimator.estimate(active(600), NOW)
    assert result["remaining_seconds"] == 3000
    assert result["low_seconds"] == 750
    assert result["high_seconds"] == 12000
    assert result["confidence"] == "very_low"
    assert result["method"] == "uncalibrated_planning_assumption"
    assert "HYPOTHÈSE NON CALIBRÉE" in result["basis"]
    assert "pas une borne garantie" in result["basis"]
    assert result["estimated_end_at"] == iso(NOW+3000)
    assert result["as_of"] == iso(NOW)


def test_preflight_benchmark_countdown_is_not_a_fake_work_counter(estimator):
    job = active(600, planning_estimate={"total_seconds": 86400, "basis": "Deux recettes quotidiennes, mesures avant lancement."})
    first = estimator.estimate(job, NOW)
    second = estimator.estimate(job, NOW + 60)
    assert first["remaining_seconds"] == 85800
    assert second["remaining_seconds"] == 85740
    assert first["method"] == "preflight_benchmark"
    assert first["confidence"] == "low"
    assert first["low_seconds"] == 42600
    assert first["high_seconds"] == 172200
    assert "pas une progression mesurée" in first["basis"]
    assert job.get('progress') is None


def test_preflight_benchmark_overrun_never_claims_finished(estimator):
    job = active(90000, planning_estimate={"total_seconds": 86400, "basis": "Mesure initiale."})
    result = estimator.estimate(job, NOW)
    assert result["remaining_seconds"] > 0
    assert result["method"] == "preflight_benchmark_overrun"
    assert result["confidence"] == "very_low"
    assert "dépassée" in result["basis"]


@pytest.mark.parametrize('duration', [True, float('nan'), float('inf'), -1, 0, 1e12])
def test_invalid_preflight_benchmark_uses_explicit_generic_fallback(estimator, duration):
    result = estimator.estimate(active(planning_estimate={"total_seconds": duration, "basis": "Invalid"}), NOW)
    assert result['method'] == 'uncalibrated_planning_assumption'


@pytest.mark.parametrize("elapsed,expected", [(0, 3600), (1800, 1800), (3600, 3600), (7200, 7200)])
def test_generic_convention_never_counts_down_to_false_zero(estimator, elapsed, expected):
    assert estimator.estimate(active(elapsed), NOW)["remaining_seconds"] == expected


def interaction(reference=1800, current_elapsed=600):
    started = NOW - reference - current_elapsed
    return active(reference+current_elapsed, title="SolarWind — interaction Kalman", zones=[
        {"zone": "DE", "status": "complete", "phase": "complete", "updated_at": iso(started+reference)},
        {"zone": "NL", "status": "running", "phase": "kalman_replay", "updated_at": iso(started+reference)},
    ])


def test_observed_de_duration_calibrates_nl_without_percent_claim(estimator):
    result = estimator.estimate(interaction(), NOW)
    assert result["method"] == "observed_sequential_zone_reference"
    assert result["remaining_seconds"] == 1200
    assert result["confidence"] == "low"
    assert "Référence observée DE : 30 min" in result["basis"]
    assert "pas mesure des recalibrations" in result["basis"]
    assert result["low_seconds"] == 300
    assert result["high_seconds"] == 3900


def test_overrun_revision_stays_positive_and_explains_convention(estimator):
    result = estimator.estimate(interaction(current_elapsed=2400), NOW)
    assert result["method"] == "observed_zone_reference_overrun"
    assert result["remaining_seconds"] == 450
    assert result["confidence"] == "very_low"
    assert "dépassée de 10 min" in result["basis"]
    assert "estimation révisée" in result["basis"]
    assert "Marge conventionnelle" in result["basis"]
    assert result["low_seconds"] > 0
    assert result["estimated_end_at"] > iso(NOW)


def test_reference_exactly_reached_never_claims_done(estimator):
    result = estimator.estimate(interaction(current_elapsed=1800), NOW)
    assert result["remaining_seconds"] > 0
    assert result["method"] == "observed_zone_reference_overrun"


def test_other_experiment_does_not_inherit_solarwind_country_timing(estimator):
    job = interaction()
    job["title"] = "run_nuclear_kalman.py"
    assert estimator.estimate(job, NOW)["method"] == "uncalibrated_planning_assumption"


def test_zone_counter_is_not_an_exact_time_progress_counter(estimator):
    job = active(600, progress={"completed": 1, "total": 2, "percent": None})
    assert estimator.estimate(job, NOW)["method"] == "uncalibrated_planning_assumption"


def test_explicit_exact_counter_is_extrapolated(estimator):
    job = active(900, progress={"completed": 30, "total": 90, "exact": True})
    result = estimator.estimate(job, NOW)
    assert result["method"] == "exact_work_counter"
    assert result["remaining_seconds"] == 1800
    assert result["confidence"] == "medium"
    assert "peuvent varier" in result["basis"]


def test_completed_but_live_process_still_has_numeric_positive_estimate(estimator):
    result = estimator.estimate(active(600, status="complete"), NOW)
    assert result["remaining_seconds"] > 0
    assert result["remaining_seconds"] == 120
    assert result["method"] == "process_finalization_assumption"
    assert result["confidence"] == "very_low"


def test_finished_and_inactive_are_not_live_etas(estimator):
    done = estimator.estimate({"status": "complete", "processes": []}, NOW)
    assert done["remaining_seconds"] == 0
    assert done["method"] == "complete"
    absent = estimator.estimate({"status": "absent", "processes": []}, NOW)
    assert absent["remaining_seconds"] is None
    assert absent["estimated_end_at"] is None


@pytest.mark.parametrize("malformed", [None, "not a timestamp", {}, float("nan"), float("inf")])
def test_malformed_fields_fall_back_safely(estimator, malformed):
    job = active()
    job.update(started_at=malformed, elapsed_seconds=malformed,
               progress={"kind": "exact", "completed": malformed, "total": 100}, zones=malformed)
    result = estimator.estimate(job, NOW)
    assert math.isfinite(result["remaining_seconds"])
    assert result["remaining_seconds"] == 3600
    assert result["method"] == "uncalibrated_planning_assumption"


def test_future_or_incoherent_zone_timestamps_do_not_calibrate(estimator):
    job = interaction()
    job["zones"][0]["updated_at"] = iso(NOW+60)
    assert estimator.estimate(job, NOW)["method"] == "uncalibrated_planning_assumption"


def test_now_supports_epoch_datetime_and_iso_consistently(estimator):
    expected = estimator.estimate(active(), NOW)
    assert estimator.estimate(active(), iso(NOW)) == expected
    assert estimator.estimate(active(), datetime.fromtimestamp(NOW, timezone.utc)) == expected


def test_input_untouched_and_root_remains_empty(estimator, tmp_path):
    import copy
    job = interaction()
    before = copy.deepcopy(job)
    estimator.estimate(job, NOW)
    assert job == before
    assert list(tmp_path.iterdir()) == []


def test_partial_control_or_report_phase_not_misread_as_replay(estimator):
    job = interaction()
    job["zones"][1]["phase"] = "report"
    assert estimator.estimate(job, NOW)["method"] == "uncalibrated_planning_assumption"


def test_fallback_anchor_counts_down_without_a_continuously_moving_horizon(estimator):
    job = active(7200)
    first = estimator.estimate(job, NOW)
    later = estimator.estimate(job, NOW+60)
    assert first["remaining_seconds"] == 7200
    assert later["remaining_seconds"] == 7140
    assert first["estimated_end_at"] == later["estimated_end_at"]
    assert later["low_seconds"] == first["low_seconds"]-60
    assert later["high_seconds"] == first["high_seconds"]-60


def test_fallback_anchor_revises_explicitly_after_expiry(estimator):
    job = active(600)
    estimator.estimate(job, NOW)
    revised = estimator.estimate(job, NOW+3000)
    assert revised["remaining_seconds"] > 0
    assert revised["method"].endswith("_revised")
    assert "révision explicite n°1" in revised["basis"]
    subsequent = estimator.estimate(job, NOW+3060)
    assert subsequent["remaining_seconds"] == revised["remaining_seconds"]-60
    assert subsequent["estimated_end_at"] == revised["estimated_end_at"]


def test_overrun_anchor_counts_down_then_revises(estimator):
    job = interaction(current_elapsed=2400)
    first = estimator.estimate(job, NOW)
    second = estimator.estimate(job, NOW+60)
    assert first["remaining_seconds"] == 450
    assert second["remaining_seconds"] == 390
    assert first["estimated_end_at"] == second["estimated_end_at"]
    revised = estimator.estimate(job, NOW+451)
    assert revised["remaining_seconds"] > 0
    assert revised["method"] == "observed_zone_reference_overrun_revised"


def test_reused_job_id_new_start_does_not_reuse_old_anchor(estimator):
    estimator.estimate(active(7200), NOW)
    fresh = estimator.estimate(active(600), NOW)
    assert fresh["remaining_seconds"] == 3000
