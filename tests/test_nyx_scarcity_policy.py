from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nyx_scarcity.policy import (
    ScarcityPolicyError, _column, _eligible_oos, _fit, _govern, _parameters,
    _prepare, run_policy,
)


def panel(days=124, *, start="2026-01-01", zones=("DE", "BE")):
    a = pd.Timestamp(start).tz_localize("Europe/Paris")
    b = (pd.Timestamp(start) + pd.Timedelta(days=days)).tz_localize("Europe/Paris")
    times = pd.date_range(a, b, freq="h", inclusive="left").tz_convert("UTC")
    civil = times.tz_convert("Europe/Paris").tz_localize(None).normalize()
    origin = (civil - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    labels = (civil - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC")
    high = np.isin(times.tz_convert("Europe/Paris").hour, [18, 19, 20, 21])
    return pd.concat([pd.DataFrame({
        "zone": zone, "timestamp_utc": times, "forecast_origin_utc": origin,
        "label_available_at_utc": labels, "forecast": 100., "q10": 80., "q90": 125.,
        "actual": np.where(high, 250., 100.), "feature_stress": high.astype(float),
        "feature_optional": np.nan, "feature_eligible": True, "label_eligible": True,
        "forecast_eligible": True, "feature_available_at_utc": origin,
        "benchmark_forecast": 140., "sample": "evaluation",
    }) for zone in zones], ignore_index=True)


def config(**kwargs):
    return {"feature_columns": ["feature_stress", "feature_optional"],
            "required_feature_columns": ["feature_stress"], "max_iter": 5,
            "min_samples_leaf": 20, **kwargs}


def test_pooled_expert_progressive_training_and_causal_governance():
    data = panel()
    result = run_policy(data, config())
    predicted = result.predictions
    assert len(predicted) == len(data)
    pd.testing.assert_series_equal(predicted.timestamp_utc, data.timestamp_utc)
    pd.testing.assert_series_equal(predicted.zone, data.zone)
    warmup = predicted.timestamp_utc < pd.Timestamp("2026-04-02", tz="Europe/Paris")
    assert not predicted.loc[warmup, "expert_ready"].any()
    assert predicted.loc[warmup, "candidate_forecast"].eq(100).all()
    assert result.audit["changed_forecast_rows"] > 0
    assert result.audit["pooled_training_zones"] == ["BE", "DE"]
    assert not result.audit["all_trained_folds_have_365_days"]
    trained = result.folds.loc[result.folds.status.eq("trained")]
    assert trained.training_days.min() >= 90
    assert (trained.max_label_available_at_utc <= trained.fit_cutoff_utc).all()
    assert (trained.model_training_end_day < trained.calibration_start_day).all()
    selected = result.governance.loc[result.governance.selected & result.governance.weight.gt(0)]
    assert not selected.empty
    assert selected.oos_days.ge(28).all()
    assert selected.mae_gain_lower_bound_eur_mwh.ge(0).all()
    assert selected.tail_mae_gain_eur_mwh.gt(.5).all()
    assert (selected.max_oos_label_available_at_utc <= selected.decision_cutoff_utc).all()
    assert (selected.oos_last_day < selected.delivery_day).all()


def test_missing_required_features_and_late_features_fall_back_without_dropping_rows():
    data = panel()
    late = data.index[-20]
    missing = data.index[-19]
    data.loc[late, "feature_available_at_utc"] += pd.Timedelta(seconds=1)
    data.loc[missing, "feature_stress"] = np.nan
    result = run_policy(data, config()).predictions
    for row in (late, missing):
        assert result.loc[row, "candidate_forecast"] == data.loc[row, "forecast"]
        assert not result.loc[row, "expert_ready"]
        assert result.loc[row, "gate_reason"] == "required_features_unavailable"
        assert result.loc[row, "selected_weight"] == 0.


def test_live_day_has_no_label_and_keeps_finite_ordered_predictions():
    data = panel()
    final = data.timestamp_utc >= pd.Timestamp("2026-05-04", tz="Europe/Paris")
    data.loc[final, "actual"] = np.nan
    data.loc[final, "label_available_at_utc"] = pd.NaT
    result = run_policy(data, config()).predictions
    assert result.loc[final, "actual"].isna().all()
    assert result.loc[final, "expert_ready"].all()
    assert np.isfinite(result[["candidate_q10", "candidate_forecast", "candidate_q90"]]).all().all()
    assert result.candidate_q10.le(result.candidate_forecast).all()
    assert result.candidate_forecast.le(result.candidate_q90).all()


def test_cap_and_intervals_are_separate_from_common_quantile_translation():
    data = panel()
    result = run_policy(data, config(correction_clip_eur_mwh=60.)).predictions
    assert result.raw_correction.max() >= 149.
    assert result.bounded_correction.max() <= 60.
    assert result.applied_correction.max() <= 60.
    changed = result.applied_correction.gt(0)
    assert changed.any()
    assert result.loc[changed, "interval_status"].str.startswith("empirical_oos").all()
    # A separate empirical error distribution, not original fixed width.
    assert (result.loc[changed, "candidate_q90"] - result.loc[changed, "candidate_q10"]).ne(45.).any()


def test_future_labels_and_storm_do_not_change_prediction_prefix():
    data = panel(days=132)
    poison = data.copy()
    day = pd.Timestamp("2026-05-07", tz="Europe/Paris")
    poison.loc[poison.timestamp_utc >= day, "actual"] = -100000.
    poison["benchmark_forecast"] = 1e8
    a, b = run_policy(data, config()), run_policy(poison, config())
    before = data.timestamp_utc < day + pd.DateOffset(days=1)
    columns = ["candidate_forecast", "candidate_q10", "candidate_q90", "spike_probability", "raw_correction", "selected_weight", "threshold_eur_mwh", "gate_reason"]
    pd.testing.assert_frame_equal(a.predictions.loc[before, columns], b.predictions.loc[before, columns])
    keep = a.governance.delivery_day.le(str(day.date()))
    pd.testing.assert_frame_equal(a.governance.loc[keep], b.governance.loc[keep])


def test_threshold_excludes_calibration_labels():
    p = _parameters(config())
    data = _prepare(panel(days=105), p)
    day = "2026-04-11"
    cutoff = pd.Timestamp("2026-04-10 08:00", tz="Europe/Paris").tz_convert("UTC")
    _, original = _fit(data, day, cutoff, p, ("BE", "DE"))
    changed = data.copy()
    changed.loc[changed._day.ge("2026-03-14"), "_error"] = 10000.
    _, modified = _fit(changed, day, cutoff, p, ("BE", "DE"))
    assert original["thresholds_eur_mwh"] == modified["thresholds_eur_mwh"]


def test_clock_and_order_preserved_across_dst_and_shuffled_input():
    data = panel(days=4, start="2026-10-23")
    data = data.sample(frac=1, random_state=14)
    result = run_policy(data, config()).predictions
    assert result.index.equals(data.index)
    pd.testing.assert_series_equal(result.timestamp_utc, data.timestamp_utc)
    assert len(data.loc[data.zone.eq("DE")]) == 97
    assert result.candidate_forecast.eq(data.forecast).all()


@pytest.mark.parametrize("changes", [
    {"trainnig_days": 100}, {"minimum_training_days": 89}, {"training_window_days": 730},
    {"feature_columns": ["actual"]}, {"feature_columns": ["feature_storm"]},
    {"candidate_weights": [.25, 1.]}, {"probability_gate": .4},
])
def test_invalid_settings_rejected(changes):
    with pytest.raises(ScarcityPolicyError):
        run_policy(panel(days=1), config(**changes))


def test_invalid_origins_and_publication_times_rejected():
    data = panel(days=1)
    data.loc[0, "forecast_origin_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ScarcityPolicyError, match="08:00"):
        run_policy(data, config())
    data = panel(days=1)
    data.loc[0, "label_available_at_utc"] = data.loc[0, "forecast_origin_utc"]
    with pytest.raises(ScarcityPolicyError, match="strictly after"):
        run_policy(data, config())


def test_one_class_is_explicit_identity_fallback():
    data = panel(days=100)
    data["actual"] = 100.
    result = run_policy(data, config())
    assert not result.predictions.expert_ready.any()
    assert result.predictions.candidate_forecast.eq(100.).all()
    assert "insufficient_training_classes" in set(result.folds.reason)


def test_all_hour_guard_can_reject_tail_improvement():
    p = _parameters(config(governance_minimum_changed_days=1, governance_minimum_tail_rows=1))
    days = pd.date_range("2026-01-01", periods=30).strftime("%Y-%m-%d")
    past = pd.DataFrame({"_day": days, "actual": [250.] * 5 + [100.] * 25,
                         "forecast": 100., "threshold_eur_mwh": 100.,
                         "label_available_at_utc": pd.Timestamp("2026-02-01", tz="UTC")})
    for w in p["candidate_weights"]:
        past[_column(w)] = 100 + w * 150
    weight, _, records = _govern(past, "DE", "2026-02-03", pd.Timestamp("2026-02-02", tz="UTC"), p)
    assert weight == 0.
    nonzero = [r for r in records if r["weight"]]
    assert all(r["tail_mae_gain_eur_mwh"] > 0 for r in nonzero)
    assert all(r["reason"] == "all_hour_mae_guard" for r in nonzero)


def test_oos_labels_not_yet_published_cannot_enter_governance():
    p = _parameters(config())
    old = _prepare(panel(days=2, zones=("DE",)), p)
    old["expert_ready"] = True
    cutoff = pd.Timestamp("2026-01-02 08:00", tz="Europe/Paris").tz_convert("UTC")
    old.loc[old._day.eq("2026-01-02"), "label_available_at_utc"] = cutoff + pd.Timedelta(hours=1)
    selected = _eligible_oos([old], "DE", "2026-01-03", cutoff, p)
    assert selected._day.unique().tolist() == ["2026-01-01"]


def test_strict_365_training_never_claims_a_synthetic_preceding_year():
    result = run_policy(panel(days=100, zones=("DE",)), config(minimum_training_days=365))
    assert result.audit["trained_folds"] == 0
    assert result.predictions.candidate_forecast.eq(100.).all()


def test_fit_window_is_capped_to_365_physical_civil_days():
    p = _parameters(config(max_iter=1, minimum_training_days=365))
    data = _prepare(panel(days=390, zones=("DE",)), p)
    day = "2027-01-25"
    cutoff = pd.Timestamp("2027-01-24 08:00", tz="Europe/Paris").tz_convert("UTC")
    _, fold = _fit(data, day, cutoff, p, ("DE",))
    assert fold["training_days"] == 365
    assert fold["full_365_day_training"]
    assert fold["training_start_day"] == "2026-01-25"
    assert fold["training_expected_hours"] == 8760


def test_dst_25_hour_oos_day_must_be_complete_before_governance():
    p = _parameters(config())
    data = _prepare(panel(days=4, start="2026-10-23", zones=("DE",)), p)
    data["expert_ready"] = True
    cutoff = pd.Timestamp("2026-10-27 08:00", tz="Europe/Paris").tz_convert("UTC")
    all_days = _eligible_oos([data], "DE", "2026-10-28", cutoff, p)
    assert len(all_days.loc[all_days._day.eq("2026-10-25")]) == 25
    missing = data.drop(data.loc[data._day.eq("2026-10-25")].index[0])
    selected = _eligible_oos([missing], "DE", "2026-10-28", cutoff, p)
    assert "2026-10-25" not in selected._day.tolist()


def test_original_input_dtypes_values_and_index_remain_exact():
    data = panel(days=1).sample(frac=1, random_state=5)
    data["feature_stress"] = data.feature_stress.astype("int32")
    data["sample"] = data["sample"].astype("category")
    data["forecast_origin_utc"] = data.forecast_origin_utc.dt.tz_convert("Europe/Paris")
    result = run_policy(data, config()).predictions
    pd.testing.assert_frame_equal(result[data.columns], data, check_exact=True)
