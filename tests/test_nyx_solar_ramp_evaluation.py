"""Synthetic evaluator tests only: no network, provider, model or production writes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nyx_solar_ramp.evaluation import _average_precision, _block_draws, evaluate


CONFIG = {"seed": 1729, "bootstrap_samples": 20, "block_days": 2}
VARIANTS = ("baseline", "control", "local", "regional", "ramps", "interactions", "governed")


def predictions(days=4, zones=("BE", "DE", "FR", "NL")):
    hours = pd.date_range("2026-09-01", periods=24 * days, freq="h", tz="UTC")
    base = pd.MultiIndex.from_product([zones, hours], names=["zone", "timestamp_utc"]).to_frame(index=False)
    base["forecast_origin_utc"] = base.timestamp_utc.dt.floor("D") - pd.Timedelta(hours=18)
    base["actual"] = np.where(base.timestamp_utc.dt.hour.eq(18), 350., 50.)
    base["spike_label"] = (base.actual >= 300).astype(float)
    base["statistical_spike_label"] = base.spike_label
    base["ramp_label"] = base.spike_label
    base["evaluation_phase"] = np.where(base.timestamp_utc.lt(hours[len(hours) // 2]), "selection", "final_diagnostic")
    frames = []
    for variant in VARIANTS:
        frame = base.copy()
        frame["variant"] = variant
        offset = 20 if variant in ("baseline", "governed") else 10
        frame["forecast"] = frame.actual - offset
        frame["candidate_forecast"] = frame.actual - (5 if variant == "local" else offset)
        frame["q10"], frame["q90"] = frame.forecast - 30, frame.forecast + 30
        frame["candidate_q10"] = frame.candidate_forecast - 30
        frame["candidate_q90"] = frame.candidate_forecast + 30
        frame["risk_probability"] = np.nan if variant == "baseline" else np.where(frame.spike_label.eq(1), .8, .1)
        frame["alert_threshold"] = np.nan if variant == "baseline" else .5
        frame["alert"] = frame.risk_probability.ge(.5)
        frame["expert_ready"] = variant != "baseline"
        frame["applied_correction"] = 0. if variant == "baseline" else 10.
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def score(result, variant="local", zone="ALL", phase="all"):
    return next(row for row in result["scores"]
                if (row["variant"], row["zone"], row["phase"]) == (variant, zone, phase))


def test_metrics_are_json_safe_readonly_and_never_promote():
    frame = predictions()
    original = frame.copy(deep=True)
    result = evaluate(frame, CONFIG)
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(frame, original)
    local = score(result)
    assert local["mae"] == 5
    assert local["candidate_mae"] == 5
    assert local["rmse"] == 5
    assert local["bias"] == -5
    assert local["baseline_mae"] == 10
    assert local["spike_underestimation"] == 5
    assert local["precision"] == local["recall"] == local["pr_auc"] == 1
    assert local["false_alerts_per_1000"] == 0
    assert local["interval_coverage_80"] == 1
    assert local["pinball_50"] == 2.5
    assert score(result, "baseline")["n_probability"] == 0
    assert score(result, "baseline")["brier"] is None
    assert result["decision"]["retain_baseline"] is True
    assert result["decision"]["promotion_allowed"] is False
    assert result["decision"]["production_pit_verified"] is False
    assert result["protocol"]["final_diagnostic_is_virgin_test"] is False


def test_unknown_actual_is_not_a_negative_label_and_input_is_unchanged():
    frame = predictions(days=2, zones=("FR",))
    unknown = frame.timestamp_utc.eq(frame.timestamp_utc.min())
    frame.loc[unknown, "actual"] = np.nan
    frame.loc[unknown, "spike_label"] = 0.
    result = evaluate(frame, {**CONFIG, "bootstrap_samples": 0})
    assert score(result)["n_actual"] == 47
    assert score(result)["n_probability"] == 47
    assert score(result)["probability_coverage"] == 1
    assert result["selection"]["empirical_best_candidate"] == "local"


def test_common_oos_uses_all_five_risk_models_not_native_coverage():
    frame = predictions(days=2, zones=("FR",))
    removed = frame.variant.eq("local") & frame.timestamp_utc.eq(frame.timestamp_utc.min())
    frame.loc[removed, "risk_probability"] = np.nan
    result = evaluate(frame, CONFIG)
    assert score(result)["probability_coverage"] == pytest.approx(47 / 48)
    assert score(result, "control")["probability_coverage"] == 1
    assert result["protocol"]["common_oos_n_targets"] == 47
    assert {score(result, v, phase="common_oos")["n"] for v in VARIANTS} == {47}
    assert score(result, "local", phase="common_oos")["probability_coverage"] == 1


def test_missing_risk_variant_has_no_fabricated_common_sample():
    frame = predictions(days=2, zones=("FR",))
    frame = frame[~frame.variant.eq("interactions")]
    result = evaluate(frame, CONFIG)
    assert result["protocol"]["common_oos_n_targets"] == 0
    assert not any(row["phase"] == "common_oos" for row in result["scores"])


def test_reduced_candidate_coverage_is_not_selected_as_a_gain():
    frame = predictions(days=2, zones=("FR",))
    local = frame.variant.eq("local")
    frame.loc[local & frame.timestamp_utc.dt.hour.ne(0), "candidate_forecast"] = np.nan
    result = evaluate(frame, CONFIG)
    assert result["selection"]["empirical_best_candidate"] != "local"
    pair = next(row for row in result["paired_bootstrap"] if row["variant"] == "local"
                and row["zone"] == "ALL" and row["phase"] == "selection"
                and row["prediction"] == "candidate_forecast")
    assert pair["complete_point_pairing"] is False
    assert pair["paired_coverage"] < 1


def test_final_diagnostic_outcomes_do_not_select_the_candidate():
    frame = predictions(days=4, zones=("FR",))
    first = evaluate(frame, CONFIG)
    frame.loc[frame.variant.eq("local") & frame.evaluation_phase.eq("final_diagnostic"), "candidate_forecast"] += 1e6
    second = evaluate(frame, CONFIG)
    assert first["selection"] == second["selection"]
    assert first["selection"]["empirical_best_candidate"] == "local"


def test_physical_utc_episode_and_one_hour_early_matching_at_dst():
    frame = predictions(days=1, zones=("FR",))
    frame = frame[frame.timestamp_utc.dt.hour.lt(4)].copy()
    frame["timestamp_utc"] += pd.Timedelta(days=54)  # 2026-10-25 fall-back.
    frame["forecast_origin_utc"] += pd.Timedelta(days=54)
    spike = frame.timestamp_utc.dt.hour.isin([0, 1])  # Both are local 02:00.
    frame["actual"] = np.where(spike, 350., 50.)
    frame["spike_label"] = spike.astype(float)
    frame["alert"] = spike & frame.timestamp_utc.dt.hour.eq(1)
    result = evaluate(frame, CONFIG)
    row = score(result)
    assert row["n_spike_episodes"] == 1
    assert row["episode_recall_exact"] == 1
    assert row["first_alert_timing_mean_hours"] == 1
    # Change to one alarm immediately before a two-hour episode.
    frame["actual"] = np.where(frame.timestamp_utc.dt.hour.isin([1, 2]), 350., 50.)
    frame["spike_label"] = (frame.actual >= 300).astype(float)
    frame["alert"] = frame.timestamp_utc.dt.hour.eq(0)
    row = score(evaluate(frame, CONFIG))
    assert row["episode_recall_exact"] == 0
    assert row["episode_recall_early_1h"] == 1
    assert row["first_alert_timing_mean_hours"] == -1


def test_bootstrap_is_paired_deterministic_and_keeps_zone_day_clusters():
    frame = predictions()
    first, second = evaluate(frame, CONFIG), evaluate(frame.sample(frac=1, random_state=4), CONFIG)
    assert first["paired_bootstrap"] == second["paired_bootstrap"]
    rows = [row for row in first["paired_bootstrap"] if row["variant"] == "local"
            and row["prediction"] == "candidate_forecast" and row["phase"] == "all"]
    assert len(rows) == 5
    for row in rows:
        assert row["delta_mae"] == row["delta_mae_ci95_low"] == row["delta_mae_ci95_high"] == -15
        assert row["delta_spike_mae"] == -15
    all_row = next(row for row in rows if row["zone"] == "ALL")
    assert all_row["n"] == 4 * 24 * 4


def test_moving_blocks_do_not_cross_calendar_gaps():
    days = pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-10", "2026-01-11"])
    draws = _block_draws(days, 20, 2, np.random.default_rng(1))
    assert set(map(tuple, draws.reshape(-1, 2))).issubset({(0, 1), (2, 3)})


def test_average_precision_ties_are_not_order_dependent():
    y, p = np.array([0., 1., 0., 1.]), np.array([.8, .8, .2, .2])
    assert _average_precision(y, p) == .5
    assert _average_precision(y[::-1], p[::-1]) == .5
    assert _average_precision(np.zeros(4), p) is None


@pytest.mark.parametrize("change,error", [
    ("duplicate", "Duplicate"), ("keys", "target keys differ"),
    ("actual", "actual values disagree"), ("probability", "outside"),
])
def test_bad_contract_is_rejected(change, error):
    frame = predictions(days=1, zones=("FR",))
    local = frame.index[frame.variant.eq("local")][0]
    if change == "duplicate":
        frame = pd.concat([frame, frame.iloc[[local]]], ignore_index=True)
    elif change == "keys":
        frame = frame.drop(index=local)
    elif change == "actual":
        frame.loc[local, "actual"] += 1
    else:
        frame.loc[local, "risk_probability"] = 1.1
    with pytest.raises(ValueError, match=error):
        evaluate(frame, CONFIG)


def test_empty_input_is_an_explicit_nonpromotion():
    result = evaluate(pd.DataFrame(), CONFIG)
    assert result["status"] == "insufficient_data"
    assert result["decision"]["retain_baseline"]
    json.dumps(result, allow_nan=False)


def test_secondary_regimes_are_not_treated_as_calibration_targets():
    result = evaluate(predictions(days=1, zones=("FR",)), CONFIG)
    row = score(result)
    for name in ("statistical_spike_label", "ramp_label"):
        metrics = row[name + "_regime"]
        assert "brier" not in metrics and "log_loss" not in metrics
        assert metrics["n_events"] == 1
        assert metrics["mae"] == 5


def test_live_historical_is_scored_separately_from_annual_and_common_oos():
    frame = predictions(days=2, zones=("FR",))
    last_day = frame.timestamp_utc.dt.day.eq(2)
    frame.loc[last_day, "evaluation_phase"] = "live_historical"
    result = evaluate(frame, CONFIG)
    assert score(result)["n"] == 24
    assert score(result, phase="live_historical")["n"] == 24
    assert score(result, phase="common_oos")["n"] == 24
