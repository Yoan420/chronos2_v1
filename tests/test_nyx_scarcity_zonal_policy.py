"""Inference and fixed-alpha invariants; no fitting, network or runtime install."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from nyx_scarcity import policy as base
from nyx_scarcity_zonal import policy as zonal


class _Classifier:
    def predict(self, matrix, *, output_margin):
        assert output_margin is True
        return matrix[:, 0]


class _Calibrator:
    def predict_proba(self, matrix):
        probability = matrix[:, 0]
        return np.column_stack([1-probability, probability])


class _Severity:
    def predict(self, matrix):
        return np.zeros(len(matrix))


def inference():
    zones = ("FR", "DE", "BE", "NL")
    state = {"features": ["feature_shared_signal"], "zones": zones,
             "thresholds": {z: 50. for z in zones}, "classifier": _Classifier(),
             "calibrator": _Calibrator(), "severity": _Severity(), "tail_log_errors": np.array([0.]),
             "zone_offsets": {"FR": -3., "DE": 0., "BE": -.5, "NL": 0.}}
    current = pd.DataFrame({"zone": zones, "timestamp_utc": pd.Timestamp("2026-09-14 17:00", tz="UTC"),
                            "forecast_origin_utc": pd.Timestamp("2026-09-13 06:00", tz="UTC"),
                            "feature_shared_signal": .9}, index=[19, 4, 12, 1])
    return state, current, {"threads": 1, "probability_gate": .6}


def saved_result(days=10, *, start="2026-01-01", zones=("FR", "BE")):
    start_day = pd.Timestamp(start).tz_localize("Europe/Paris")
    end_day = (pd.Timestamp(start)+pd.Timedelta(days=days)).tz_localize("Europe/Paris")
    times = pd.date_range(start_day, end_day, freq="h", inclusive="left").tz_convert("UTC")
    civil = times.tz_convert("Europe/Paris").tz_localize(None).normalize()
    origins = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    labels = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC")
    high = np.isin(times.tz_convert("Europe/Paris").hour, [18, 19, 20, 21])
    frame = pd.concat([pd.DataFrame({
        "zone": zone, "timestamp_utc": times, "forecast_origin_utc": origins, "label_available_at_utc": labels,
        "forecast": 100., "q10": 80., "q90": 125., "actual": np.where(high, 200., 100.),
        "feature_signal": high.astype(float), "feature_eligible": True, "label_eligible": True,
        "forecast_eligible": True, "feature_available_at_utc": origins, "sample": "evaluation",
        "benchmark_forecast": 140., "expert_ready": True, "bounded_correction": high.astype(float)*100,
        "raw_correction": high.astype(float)*100, "spike_probability": np.where(high, .8, .1),
        "probability_gate": .6, "threshold_eur_mwh": 50., "selected_weight": 0., "applied_correction": 0.,
        "candidate_forecast": 100., "candidate_q10": 80., "candidate_q90": 125.,
        "gate_reason": "no_candidate_passes_guards", "interval_status": "baseline_preserved",
    }) for zone in zones], ignore_index=True)
    settings = base._parameters({"feature_columns": ["feature_signal"], "required_feature_columns": ["feature_signal"],
                                 "interval_minimum_rows": 20, "threads": 1})
    return base.PolicyResult(frame, pd.DataFrame({"status": ["frozen_test_model"]}),
                             pd.DataFrame({"reason": ["strict_governance_unchanged"]}), {"config": settings})


def test_predict_state_needs_no_labels_and_uses_country_offsets():
    state, current, parameters = inference()
    before = current.copy(deep=True)
    probability, raw, threshold, diagnostic = zonal.predict_state(state, current, parameters)
    expected = expit(logit(.9)+np.array([-3., 0., -.5, 0.]))
    np.testing.assert_allclose(probability, expected)
    np.testing.assert_allclose(threshold, 50.)
    assert raw[0] == 0 and (raw[1:] == 50.).all()
    assert diagnostic.index.equals(current.index)
    assert set(diagnostic).isdisjoint({"actual", "_error", "benchmark_forecast"})
    np.testing.assert_allclose(diagnostic.probability_before_zonal, .9)
    pd.testing.assert_frame_equal(current, before)


def test_predict_state_is_invariant_to_actual_storm_and_future_metadata():
    state, current, parameters = inference()
    original = zonal.predict_state(state, current, parameters)
    current["actual"] = "poisoned observed price"
    current["_error"] = np.inf
    current["benchmark_forecast"] = 1e10
    current["label_available_at_utc"] = pd.Timestamp("2099-01-01", tz="UTC")
    modified = zonal.predict_state(state, current, parameters)
    for before, after in zip(original[:3], modified[:3]):
        np.testing.assert_array_equal(before, after)
    pd.testing.assert_frame_equal(original[3], modified[3])


def test_predict_state_numeric_boundary_clipping_is_explicit_and_finite():
    state, current, parameters = inference()
    current["feature_shared_signal"] = [0., 1., 0., 1.]
    probability, raw, threshold, audit = zonal.predict_state(state, current, parameters)
    assert np.isfinite(np.column_stack([probability, raw, threshold])).all()
    assert audit.shared_probability_numeric_clip_count_in_batch.eq(4).all()
    assert ((probability > 0) & (probability < 1)).all()


def test_predict_state_respects_strict_probability_gate():
    state, current, parameters = inference()
    state["zone_offsets"] = {z: 0. for z in state["zones"]}
    current["feature_shared_signal"] = [.59, .6, .61, .5]
    _, raw, _, _ = zonal.predict_state(state, current, parameters)
    assert raw.tolist() == [0., 0., 50., 0.]


@pytest.mark.parametrize("probability", [np.nan, np.inf, -.1, 1.1])
def test_predict_state_invalid_shared_probability_fails_closed(probability):
    state, current, parameters = inference()
    current.loc[current.index[0], "feature_shared_signal"] = probability
    with pytest.raises(base.ScarcityPolicyError, match="probabilities"):
        zonal.predict_state(state, current, parameters)


def test_fixed_alpha_changes_only_explicit_output_fields_and_preserves_strict_diagnostics():
    result = saved_result()
    original = result.predictions.copy(deep=True)
    evaluated = zonal.fixed_conservative_forecast(result)
    output = evaluated.predictions
    np.testing.assert_allclose(output.candidate_forecast, output.forecast+.25*output.bounded_correction)
    np.testing.assert_allclose(output.applied_correction, .25*output.bounded_correction)
    np.testing.assert_allclose(output.strict_governed_forecast, original.candidate_forecast)
    np.testing.assert_allclose(output.strict_governed_weight, original.selected_weight)
    assert output.strict_governance_reason.equals(original.gate_reason)
    pd.testing.assert_frame_equal(result.predictions, original)
    pd.testing.assert_frame_equal(evaluated.folds, result.folds)
    pd.testing.assert_frame_equal(evaluated.governance, result.governance)
    assert evaluated.audit["fixed_alpha"] == .25
    assert evaluated.audit["annual_non_regression_guaranteed"] is False
    assert evaluated.audit["strict_governor_enforced_in_this_point_forecast"] is False
    assert evaluated.audit["amplitude_selected_on_evaluation"] is False


@pytest.mark.parametrize("alpha", [0., .1, .5, 1., np.nan, "0.25", True])
def test_alpha_is_predeclared_not_a_tunable_hindsight_weight(alpha):
    with pytest.raises(base.ScarcityPolicyError, match="predeclared at 25%"):
        zonal.fixed_conservative_forecast(saved_result(days=2), alpha=alpha)


def test_intervals_calibrate_own_fixed_policy_errors_not_strict_governor_errors(monkeypatch):
    result = saved_result()
    captured = []
    implementation = base._intervals
    def inspect(current, past, weight, settings):
        assert weight == .25
        if len(past):
            assert past._day.lt(current._day.iloc[0]).all()
            assert past.label_available_at_utc.le(current.forecast_origin_utc.iloc[0]).all()
            np.testing.assert_allclose(past[base._column(.25)], past.forecast+.25*past.bounded_correction)
            assert past.strict_governed_forecast.eq(100.).all()
            captured.append(len(past))
        return implementation(current, past, weight, settings)
    monkeypatch.setattr(base, "_intervals", inspect)
    output = zonal.fixed_conservative_forecast(result).predictions
    assert captured
    mature_active = output.timestamp_utc.ge(pd.Timestamp("2026-01-07", tz="Europe/Paris")) & output.applied_correction.gt(0)
    np.testing.assert_allclose(output.loc[mature_active, "candidate_q90"], 200.)
    assert output.loc[mature_active, "interval_status"].eq("empirical_oos_regime_errors").all()
    assert (output.loc[mature_active, "candidate_q90"]-output.loc[mature_active, "candidate_q10"]).ne(45.).all()


def test_future_label_changes_leave_point_forecasts_and_prior_intervals_unchanged():
    result = saved_result()
    before = zonal.fixed_conservative_forecast(result).predictions
    poisoned = result.predictions.copy(deep=True)
    boundary = pd.Timestamp("2026-01-07", tz="Europe/Paris")
    poisoned.loc[poisoned.timestamp_utc.ge(boundary), "actual"] = 10000.
    poisoned["benchmark_forecast"] = -10000.
    after = zonal.fixed_conservative_forecast(replace(result, predictions=poisoned)).predictions
    np.testing.assert_array_equal(before.candidate_forecast, after.candidate_forecast)
    unchanged = before.timestamp_utc.lt(boundary+pd.DateOffset(days=1))
    columns = ["candidate_forecast", "candidate_q10", "candidate_q90", "selected_weight", "interval_status"]
    pd.testing.assert_frame_equal(before.loc[unchanged, columns], after.loc[unchanged, columns])
    # Once those labels are genuinely available, only subsequent intervals may change.
    later = before.timestamp_utc.ge(boundary+pd.DateOffset(days=1)) & before.applied_correction.gt(0)
    assert after.loc[later, "candidate_q90"].gt(before.loc[later, "candidate_q90"]).any()


def test_delayed_labels_cannot_enter_fixed_policy_intervals_before_publication():
    result = saved_result(days=8)
    source = result.predictions.copy(deep=True)
    target = source.timestamp_utc.ge(pd.Timestamp("2026-01-04", tz="Europe/Paris")) & source.timestamp_utc.lt(pd.Timestamp("2026-01-05", tz="Europe/Paris"))
    source.loc[target, "label_available_at_utc"] = pd.Timestamp("2026-01-07 18:00", tz="Europe/Paris")
    original = zonal.fixed_conservative_forecast(replace(result, predictions=source)).predictions
    modified = source.copy(deep=True)
    modified.loc[target, "actual"] = 9000.
    after = zonal.fixed_conservative_forecast(replace(result, predictions=modified)).predictions
    unchanged = source.forecast_origin_utc.lt(pd.Timestamp("2026-01-07 18:00", tz="Europe/Paris"))
    pd.testing.assert_frame_equal(original.loc[unchanged, ["candidate_q10", "candidate_q90"]],
                                  after.loc[unchanged, ["candidate_q10", "candidate_q90"]])


def test_no_point_change_preserves_original_baseline_intervals():
    result = saved_result(days=4)
    data = result.predictions.copy(deep=True)
    unavailable = data.zone.eq("FR")
    data.loc[unavailable, "expert_ready"] = False
    data.loc[unavailable, ["raw_correction", "bounded_correction"]] = 0.
    output = zonal.fixed_conservative_forecast(replace(result, predictions=data)).predictions
    inactive = output.applied_correction.eq(0)
    np.testing.assert_allclose(output.loc[inactive, "candidate_q10"], output.loc[inactive, "q10"])
    np.testing.assert_allclose(output.loc[inactive, "candidate_q90"], output.loc[inactive, "q90"])
    assert output.loc[unavailable, "selected_weight"].eq(0).all()
    assert output.loc[unavailable, "gate_reason"].eq("expert_unavailable_preserve_nyx").all()


def test_live_day_unknown_labels_is_predicted_and_not_used_to_calibrate_itself():
    result = saved_result(days=6)
    data = result.predictions.copy(deep=True)
    live = data.timestamp_utc.ge(pd.Timestamp("2026-01-06", tz="Europe/Paris"))
    data.loc[live, "actual"] = np.nan
    data.loc[live, "label_available_at_utc"] = pd.NaT
    data.loc[live, "sample"] = "live"
    output = zonal.fixed_conservative_forecast(replace(result, predictions=data)).predictions
    assert output.loc[live, "actual"].isna().all()
    assert np.isfinite(output.loc[live, ["candidate_forecast", "candidate_q10", "candidate_q90"]]).all().all()
    assert output.candidate_q10.le(output.candidate_forecast).all()
    assert output.candidate_forecast.le(output.candidate_q90).all()


def test_repeated_dst_hour_is_preserved_and_intervals_remain_ordered():
    result = saved_result(days=7, start="2025-10-23")
    original = result.predictions.iloc[::-1].copy()
    original.index = pd.Index(range(1000, 1000+len(original)), name="original_rows")
    output = zonal.fixed_conservative_forecast(replace(result, predictions=original)).predictions
    pd.testing.assert_series_equal(output.timestamp_utc, original.timestamp_utc)
    pd.testing.assert_series_equal(output.zone, original.zone)
    local = output.timestamp_utc.dt.tz_convert("Europe/Paris")
    repeated = local.dt.strftime("%Y-%m-%d").eq("2025-10-26") & local.dt.hour.eq(2) & output.zone.eq("FR")
    assert repeated.sum() == 2
    assert output.candidate_q10.le(output.candidate_forecast).all()
    assert output.candidate_forecast.le(output.candidate_q90).all()
