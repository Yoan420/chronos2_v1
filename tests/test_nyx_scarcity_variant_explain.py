from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from nyx_scarcity.variant_explain import ShapCollector, VariantExplainError, explain_xgb


FEATURES = ["feature_stress", "feature_renewable", "feature_optional"]
ZONES = ("BE", "DE")
PARAMETERS = {"feature_columns": FEATURES, "required_feature_columns": FEATURES[:2], "threads": 1}


def frame(day="2026-09-14", *, zones=ZONES):
    start = pd.Timestamp(day).tz_localize("Europe/Paris")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    times = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    cutoff = (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    n = len(times)
    return pd.concat([pd.DataFrame({
        "zone": z, "timestamp_utc": times, "forecast_origin_utc": cutoff,
        "feature_stress": np.linspace(-2, 2, n), "feature_renewable": np.cos(np.arange(n)),
        "feature_optional": [np.nan if i % 3 == 0 else float(i % 4) for i in range(n)],
        "feature_eligible": True, "forecast_eligible": True, "feature_available_at_utc": cutoff,
        "actual": 200., "benchmark_forecast": 175., "label_eligible": True,
    }) for z in zones], ignore_index=True)


@pytest.fixture(scope="module")
def fitted():
    # Tests use the explicitly installed private runtime, never install packages.
    from nyx_scarcity.variant_runtime import ensure_runtime
    try:
        xgb = ensure_runtime()
    except ValueError as exc:
        pytest.skip(str(exc))
    rng = np.random.default_rng(728)
    X = rng.normal(size=(320, 5))
    X[:, 3] = rng.integers(0, 2, len(X))
    X[:, 4] = 1 - X[:, 3]
    X[::9, 2] = np.nan
    y = (X[:, 0] - .7 * X[:, 1] + .35 * X[:, 4] + rng.normal(0, .35, len(X)) > .6).astype(int)
    clf = xgb.XGBClassifier(n_estimators=14, max_depth=2, learning_rate=.2,
                            objective="binary:logistic", tree_method="hist", n_jobs=1,
                            scale_pos_weight=4, random_state=7)
    clf.fit(X[:240], y[:240])
    margins = clf.predict(X[240:], output_margin=True)
    calibrator = LogisticRegression(random_state=7).fit(margins.reshape(-1, 1), y[240:])
    return {"classifier": clf, "calibrator": calibrator, "calibration_input": "raw_margin",
            "features": FEATURES.copy(), "zones": ZONES,
            "model_feature_names": FEATURES + [f"zone__{z}" for z in ZONES],
            "fit_day": "2026-09-08", "fit_cutoff": pd.Timestamp("2026-09-07T06:00Z"),
            "model_training_end_day": "2026-08-10", "calibration_end_day": "2026-09-07",
            "max_label_available_at_utc": pd.Timestamp("2026-09-06T16:00Z"),
            "target_kind": "positive_residual_tail"}


def collector(**kwargs):
    return ShapCollector("xgb_residual_tail", "2025-09-15", "2026-09-14", live_day="2026-09-15", **kwargs)


def test_true_tree_shap_reconstructs_raw_and_calibrated_margins_and_probabilities(fitted):
    c = collector()
    original = frame()
    untouched = original.copy(deep=True)
    c.observe(fitted, original, PARAMETERS)
    long, obs = c.frames()
    assert len(obs) == 48 and len(long) == 48 * 5
    assert obs.status.eq("explained").all()
    sums = long.groupby("sample_id")[["shap_value_raw", "shap_value_calibrated"]].sum()
    aligned = obs.set_index("sample_id").loc[sums.index]
    np.testing.assert_allclose(sums.shap_value_raw + aligned.base_value_raw, aligned.raw_margin, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(sums.shap_value_calibrated + aligned.base_value_calibrated,
                               aligned.calibrated_margin, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(expit(obs.raw_margin), obs.prob_raw, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(expit(obs.calibrated_margin), obs.prob_calibrated, rtol=1e-10, atol=1e-10)
    slope = fitted["calibrator"].coef_[0, 0]
    np.testing.assert_allclose(long.shap_value_calibrated, slope * long.shap_value_raw)
    assert set(long.feature) == set(FEATURES + ["zone__BE", "zone__DE"])
    assert "actual" not in long and "benchmark_forecast" not in obs
    assert obs.fit_cutoff_utc.le(obs.forecast_origin_utc).all()
    pd.testing.assert_frame_equal(original, untouched)


def test_summary_is_json_safe_and_labels_explanation_scope(fitted):
    c = collector()
    c.observe(fitted, frame(), PARAMETERS)
    c.observe(fitted, frame("2026-09-15"), PARAMETERS)
    result = c.summary()
    json.dumps(result, allow_nan=False)
    assert result["status"] == "complete"
    assert result["audit"]["reconstruction_verified"] is True
    assert result["audit"]["not_final_price_contributions"]
    assert result["audit"]["not_causal_effects"]
    assert result["sampling"]["selection_uses_outcomes"] is False
    assert len(result["global_importance"]) == 5
    assert len(result["local_cases"]) == 4
    assert result["sampling"]["explained_rows"] == 96
    expected = c.frames()[0].assign(v=lambda x: x.shap_value_calibrated.abs()).groupby("feature").v.mean()
    for row in result["global_importance"]:
        assert row["mean_abs_shap_calibrated"] == pytest.approx(expected[row["feature"]])


def test_calls_exact_pred_contribs_on_current_callback_only(fitted, monkeypatch):
    import xgboost as xgb
    original = xgb.Booster.predict
    calls = []
    def spy(self, data, **kwargs):
        calls.append((data.num_row(), kwargs.copy()))
        return original(self, data, **kwargs)
    monkeypatch.setattr(xgb.Booster, "predict", spy)
    c = collector()
    c.observe(fitted, frame(), PARAMETERS)
    contribution_calls = [k for n, k in calls if k.get("pred_contribs")]
    assert len(contribution_calls) == 1
    assert contribution_calls[0]["approx_contribs"] is False
    assert all(n == 48 for n, _ in calls)


def test_sampling_is_predeclared_calendar_based_not_observed_error_or_storm():
    a = ShapCollector("a", "2026-09-01", "2026-09-14", live_day="2026-09-15", sample_days_last=2)
    b = ShapCollector("a", "2026-09-01", "2026-09-14", live_day="2026-09-15", sample_days_last=2)
    for day in pd.date_range("2026-09-01", "2026-09-15", freq="D").strftime("%Y-%m-%d"):
        data = frame(day)
        other = data.assign(actual=[object()] * len(data), benchmark_forecast="DO NOT READ", label_eligible=False)
        a.observe(None, data, PARAMETERS)
        b.observe(None, other, PARAMETERS)
    left, right = a.frames()[1], b.frames()[1]
    pd.testing.assert_frame_equal(left, right)
    assert len(left) == (2 + 2 * 24 + 24) * 2
    assert a.summary()["status"] == "unavailable"
    sparse = left.loc[left.sampling_rule.eq("earlier_weekly_fixed_hour")]
    assert set(sparse.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")) == {"2026-09-01", "2026-09-08"}
    assert sparse.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour.eq(19).all()


@pytest.mark.parametrize("feature", ["actual", "benchmark_forecast", "feature_storm", "feature_actual", "label_available_at_utc", "candidate_forecast"])
def test_forbidden_features_rejected_before_any_explanation(feature, fitted):
    state = {**fitted, "features": [feature]}
    with pytest.raises(VariantExplainError, match="exclude labels"):
        collector().observe(state, frame(), {"feature_columns": [feature], "required_feature_columns": []})


def test_required_features_cannot_use_labels_even_for_sample_eligibility(fitted):
    with pytest.raises(VariantExplainError, match="required feature contract"):
        collector().observe(fitted, frame(), {**PARAMETERS, "required_feature_columns": ["actual"]})


def test_feature_order_identity_checked(fitted):
    bad = {**fitted, "model_feature_names": list(reversed(fitted["model_feature_names"]))}
    with pytest.raises(VariantExplainError, match="feature names"):
        collector().observe(bad, frame(), PARAMETERS)


def test_future_fitted_state_rejected_before_predict(fitted, monkeypatch):
    import xgboost as xgb
    def forbidden(*args, **kwargs):
        raise AssertionError("No inference permitted with a future fitted model")
    monkeypatch.setattr(xgb.Booster, "predict", forbidden)
    bad = {**fitted, "fit_day": "2026-09-16", "fit_cutoff": pd.Timestamp("2026-09-15T06:00Z")}
    with pytest.raises(VariantExplainError, match="after an explained origin"):
        collector().observe(bad, frame(), PARAMETERS)


def test_mixed_origins_and_wrong_future_delivery_rejected(fitted):
    mixed = pd.concat([frame(), frame("2026-09-15")], ignore_index=True)
    with pytest.raises(VariantExplainError, match="one current"):
        collector().observe(fitted, mixed, PARAMETERS)
    wrong = frame()
    wrong.loc[0, "timestamp_utc"] += pd.Timedelta(days=1)
    with pytest.raises(VariantExplainError, match="one current"):
        collector().observe(fitted, wrong, PARAMETERS)


@pytest.mark.parametrize("change", [
    {"max_label_available_at_utc": pd.Timestamp("2026-09-08T16:00Z")},
    {"calibration_end_day": "2026-09-08"},
    {"model_training_end_day": "2026-09-08"},
])
def test_fitted_state_causality_metadata_is_checked(fitted, change):
    with pytest.raises(VariantExplainError):
        collector().observe({**fitted, **change}, frame(), PARAMETERS)


def test_missing_late_features_flagged_without_fabricated_explanations(fitted):
    data = frame()
    data.loc[0, "feature_stress"] = np.nan
    data.loc[1, "feature_eligible"] = False
    data.loc[2, "feature_available_at_utc"] += pd.Timedelta(seconds=1)
    c = collector()
    c.observe(fitted, data, PARAMETERS)
    long, obs = c.frames()
    assert len(obs) == 48 and len(long) == 45 * 5
    assert obs.status.eq("not_explained").sum() == 3
    assert c.summary()["status"] == "partial"
    assert c.summary()["sampling"]["skipped_by_reason"] == {"features_unavailable_at_origin": 3}


def test_non_xgboost_does_not_relabel_importance_as_shap():
    state = {"classifier": HistGradientBoostingClassifier()}
    c = collector()
    c.observe(state, frame(), PARAMETERS)
    assert c.frames()[0].empty
    result = c.summary()
    assert result["method"] == "no_tree_shap_available"
    assert not result["audit"]["reconstruction_verified"]
    assert result["sampling"]["skipped_by_reason"] == {"non_xgboost_classifier_no_fabricated_shap": 48}


def test_duplicate_callback_cannot_explain_history_again_with_later_state(fitted):
    c = collector()
    c.observe(fitted, frame(), PARAMETERS)
    with pytest.raises(VariantExplainError, match="revisited"):
        c.observe(fitted, frame(), PARAMETERS)


def test_dst_fall_day_retains_25_physical_hours(fitted):
    c = ShapCollector("xgb", "2026-01-01", "2026-10-25")
    c.observe(fitted, frame("2026-10-25"), PARAMETERS)
    long, obs = c.frames()
    assert len(obs) == 50 and len(long) == 250
    assert not obs.duplicated(["zone", "timestamp_utc"]).any()


def test_clipped_probability_platt_rejected(fitted):
    with pytest.raises(VariantExplainError, match="without clipping"):
        collector().observe({**fitted, "calibration_input": "clipped_logit_probability"}, frame(), PARAMETERS)


def test_multi_input_calibrator_rejected(fitted):
    calibrator = LogisticRegression().fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 1, 1])
    with pytest.raises(VariantExplainError, match="one-dimensional"):
        collector().observe({**fitted, "calibrator": calibrator}, frame(), PARAMETERS)


def test_corrupted_contributions_fail_closed(fitted, monkeypatch):
    import xgboost as xgb
    original = xgb.Booster.predict
    def corrupt(self, data, **kwargs):
        result = original(self, data, **kwargs)
        if kwargs.get("pred_contribs"):
            result = result.copy()
            result[:, 0] += 1.
        return result
    monkeypatch.setattr(xgb.Booster, "predict", corrupt)
    with pytest.raises(VariantExplainError, match="do not reconstruct"):
        collector().observe(fitted, frame(), PARAMETERS)


def test_negative_platt_slope_preserves_signed_true_contributions(fitted):
    state = deepcopy(fitted)
    state["calibrator"].coef_[0, 0] *= -1
    c = collector()
    c.observe(state, frame(), PARAMETERS)
    long, _ = c.frames()
    np.testing.assert_allclose(long.shap_value_calibrated,
                               state["calibrator"].coef_[0, 0] * long.shap_value_raw)
    assert c.summary()["audit"]["reconstruction_verified"]


def test_early_stopping_tree_range_matches_issued_classifier(fitted):
    state = deepcopy(fitted)
    state["classifier"].get_booster().set_attr(best_iteration="0")
    c = collector()
    data = frame()
    c.observe(state, data, PARAMETERS)
    X = np.column_stack([data[FEATURES].to_numpy(float), *(data.zone.eq(z).to_numpy(float) for z in ZONES)])
    expected = state["classifier"].predict(X, output_margin=True)
    np.testing.assert_allclose(c.frames()[1].raw_margin, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kwargs", [{"sample_days_last": 0}, {"historical_stride_days": 0}, {"historical_hour": 24}])
def test_sampling_parameters_bounded(kwargs):
    with pytest.raises(VariantExplainError):
        collector(**kwargs)


def test_summary_and_frames_do_not_write_files(fitted, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = collector()
    c.observe(fitted, frame(), PARAMETERS)
    c.frames()
    json.dumps(c.summary(), allow_nan=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["xgb_weighted_fixed", "xgb_weighted_dwt"])
def test_actual_variant_daily_callback_collects_matching_oos_probabilities(fitted, name):
    from nyx_scarcity.variant_policy import run_variant_policy
    from test_nyx_scarcity_variants_policy import panel, settings, variant
    data = panel(days=125)
    end = data.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d").max()
    c = ShapCollector(name, "2026-01-01", end)
    result = run_variant_policy(data, settings(max_iter=2, refit_every_days=60), variant(name), on_predict=c.observe)
    long, obs = c.frames()
    assert not long.empty and not obs.empty
    issued = result.predictions[["zone", "timestamp_utc", "spike_probability"]]
    joined = obs.merge(issued, on=["zone", "timestamp_utc"], validate="one_to_one")
    np.testing.assert_allclose(joined.prob_calibrated, joined.spike_probability, atol=1e-10, rtol=1e-10)
    assert c.summary()["audit"]["reconstruction_verified"]
    if name.endswith("dwt"):
        assert "feature_tail_threshold_eur_mwh" in set(long.feature)
