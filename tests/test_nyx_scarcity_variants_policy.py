from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nyx_scarcity import policy as base
from nyx_scarcity import variant_policy as variants


def panel(days=135, zones=("DE", "BE")):
    first = pd.Timestamp("2026-01-01", tz="Europe/Paris")
    final = first + pd.DateOffset(days=days)
    index = pd.date_range(first, final, freq="h", inclusive="left").tz_convert("UTC")
    civil = index.tz_convert("Europe/Paris").tz_localize(None).normalize()
    origins = (civil - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    labels = (civil - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC")
    high = np.isin(index.tz_convert("Europe/Paris").hour, [18, 19, 20, 21])
    return pd.concat([pd.DataFrame({
        "zone": zone, "timestamp_utc": index, "forecast_origin_utc": origins,
        "label_available_at_utc": labels, "forecast": 100., "q10": 80., "q90": 120.,
        "actual": np.where(high, 250., 100.), "feature_stress": high.astype("int32"),
        "feature_optional": np.nan, "feature_eligible": True, "label_eligible": True,
        "forecast_eligible": True, "benchmark_forecast": 120.,
    }) for zone in zones], ignore_index=True)


def settings(**kwargs):
    return {"feature_columns": ["feature_stress", "feature_optional"],
            "required_feature_columns": ["feature_stress"], "max_iter": 3,
            "min_samples_leaf": 20, "threads": 1, **kwargs}


def variant(name):
    weighted, kind = variants.VARIANTS[name]
    return {"id": name, "weighted": weighted, "threshold_kind": kind}


def test_callback_seam_default_is_exactly_the_existing_hgb_pipeline():
    data = panel(days=7)
    a = base.run_policy(data, settings())
    b = base.run_policy(data, settings(), fit_callback=base._fit, predict_callback=base._predict)
    pd.testing.assert_frame_equal(a.predictions, b.predictions, check_exact=True)
    pd.testing.assert_frame_equal(a.folds, b.folds, check_exact=True)
    pd.testing.assert_frame_equal(a.governance, b.governance, check_exact=True)
    assert a.audit == b.audit


@pytest.mark.parametrize("name", list(variants.VARIANTS))
def test_all_four_real_xgboost_ablations_preserve_inputs_and_guards(name):
    data = panel()
    observed = []
    def callback(state, current, pars):
        observed.append((state, current, pars))
        assert current.actual.isna().all() and current._error.isna().all()
        assert current._features_valid.all()
        assert current.forecast_origin_utc.ge(state["fit_cutoff"]).all()
        assert state["calibration_input"] == "raw_margin"
        matrix = base._matrix(current, state["features"], state["zones"])
        assert matrix.shape[1] == len(state["model_feature_names"])
    result = variants.run_variant_policy(data, settings(), variant(name), on_predict=callback)
    pd.testing.assert_frame_equal(result.predictions[data.columns], data, check_exact=True)
    assert result.predictions.spike_probability.dropna().between(0, 1).all()
    assert result.audit["variant"]["id"] == name
    assert result.audit["trained_folds"] > 0 and observed
    assert result.predictions.applied_correction.abs().max() <= 400.
    assert result.predictions.candidate_q10.le(result.predictions.candidate_forecast).all()
    assert result.predictions.candidate_forecast.le(result.predictions.candidate_q90).all()
    trained = result.folds.loc[result.folds.status.eq("trained")]
    assert trained.minimum_eligible_training_days.ge(90).all()
    assert not trained.calibration_sample_weight_used.any()
    assert (trained.max_label_available_at_utc <= trained.fit_cutoff_utc).all()
    if "unweighted" in name:
        assert trained.scale_pos_weight.eq(1.).all()
    else:
        expected = np.minimum(30., np.sqrt(trained.training_negative_rows / trained.training_tail_rows))
        np.testing.assert_allclose(trained.scale_pos_weight, expected)
    if name.endswith("dwt"):
        assert variants.THRESHOLD_FEATURE in result.predictions
        assert trained.training_days.min() >= 120
    else:
        assert variants.THRESHOLD_FEATURE not in result.predictions


def test_causal_dwt_uses_high_volatility_weight_on_global_quantile():
    data = panel(days=100, zones=("DE",))
    civil = data.timestamp_utc.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    high = data.feature_stress.eq(1)
    # Old large errors set a higher global than local quantile. Following a
    # calmer middle segment, recent volatility rises but stays below old tail.
    age = (civil - civil.min()).dt.days
    amplitude = np.where(age < 40, 500., np.where(age < 70, 60., 150.))
    data["actual"] = 100 + np.where(high, amplitude, 0.)
    thresholds, audit = variants.causal_thresholds(data, settings())
    valid = thresholds.dwt_available
    calculated = np.maximum(50., thresholds.dwt_weight * thresholds.dwt_global_quantile
                            + (1 - thresholds.dwt_weight) * thresholds.dwt_local_quantile)
    np.testing.assert_allclose(thresholds.loc[valid, variants.THRESHOLD_FEATURE], calculated.loc[valid])
    distinguish = valid & thresholds.dwt_global_quantile.ne(thresholds.dwt_local_quantile) & thresholds.dwt_weight.ne(.5)
    assert distinguish.any()
    reversed_formula = thresholds.dwt_weight * thresholds.dwt_local_quantile + (1 - thresholds.dwt_weight) * thresholds.dwt_global_quantile
    assert (thresholds.loc[distinguish, variants.THRESHOLD_FEATURE] != reversed_formula.loc[distinguish]).any()
    assert audit["causality_violations"] == 0
    assert not audit["normalization_range_includes_current_sigma"]


def test_dwt_future_labels_and_future_volatility_cannot_change_a_prefix():
    data = panel(days=95, zones=("DE",))
    changed = data.copy()
    start = pd.Timestamp("2026-03-15", tz="Europe/Paris")
    changed.loc[changed.timestamp_utc >= start, "actual"] = np.arange(sum(changed.timestamp_utc >= start)) * 1e5
    a, _ = variants.causal_thresholds(data, settings())
    b, _ = variants.causal_thresholds(changed, settings())
    before = data.timestamp_utc < start + pd.DateOffset(days=1)
    pd.testing.assert_frame_equal(a.loc[before], b.loc[before], check_exact=True)


def test_dwt_late_publication_is_unavailable_until_its_explicit_time():
    data = panel(days=50, zones=("DE",))
    delayed = data.copy()
    late_rows = delayed.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d").eq("2026-02-05")
    delayed.loc[late_rows, "actual"] = 100000.
    delayed.loc[late_rows, "label_available_at_utc"] = pd.Timestamp("2026-02-15 18:00", tz="Europe/Paris")
    alternative = delayed.copy()
    alternative.loc[late_rows, "actual"] = -100000.
    a, _ = variants.causal_thresholds(delayed, settings())
    b, _ = variants.causal_thresholds(alternative, settings())
    before = data.forecast_origin_utc < pd.Timestamp("2026-02-15 18:00", tz="Europe/Paris")
    pd.testing.assert_frame_equal(a.loc[before], b.loc[before], check_exact=True)
    # Missing a complete day of the most recent30 causes explicit threshold
    # fallback; no interpolation makes it appear historically known.
    gap = (data.timestamp_utc >= pd.Timestamp("2026-02-06", tz="Europe/Paris")) & before
    assert not a.loc[gap, "dwt_available"].any()


def test_fixed_weights_do_not_use_validation_class_counts():
    p = base._parameters(settings())
    data = base._prepare(panel(days=105), p)
    day = "2026-04-11"
    cutoff = pd.Timestamp("2026-04-10 08:00", tz="Europe/Paris").tz_convert("UTC")
    _, first = variants._fit(data, day, cutoff, p, ("BE", "DE"), variant=variant("xgb_weighted_fixed"))
    changed = data.copy()
    changed.loc[changed._day.ge("2026-03-14"), "_error"] = 100000.
    _, second = variants._fit(changed, day, cutoff, p, ("BE", "DE"), variant=variant("xgb_weighted_fixed"))
    assert first["scale_pos_weight"] == second["scale_pos_weight"]
    assert first["thresholds_eur_mwh"] == second["thresholds_eur_mwh"]
    assert second["reason"] == "insufficient_chronological_calibration_events"


def test_dwt_threshold_updates_between_model_refits():
    data = panel(days=138, zones=("DE",))
    # Seven all-hour high-error days cross the global95th percentile just
    # after the May7 fit, so u must change before the May14 refit.
    cutoff = pd.Timestamp("2026-05-01", tz="Europe/Paris")
    affected = data.timestamp_utc.ge(cutoff)
    data.loc[affected, "actual"] = 500.
    result = variants.run_variant_policy(data, settings(), variant("xgb_unweighted_dwt"))
    predictions = result.predictions
    ready = predictions.expert_ready
    # Values used at inference are the threshold of that origin, not a stale
    # threshold dictionary from the weekly model fit.
    np.testing.assert_allclose(predictions.loc[ready, "threshold_eur_mwh"], predictions.loc[ready, variants.THRESHOLD_FEATURE])
    daily = predictions.loc[ready].groupby(["expert_fit_day", predictions.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")], observed=True).threshold_eur_mwh.first()
    assert daily.groupby(level=0).nunique().max() > 1


def test_weighted_dwt_policy_is_causal_under_future_label_poisoning():
    data = panel(days=136, zones=("DE",))
    poison_day = pd.Timestamp("2026-05-11", tz="Europe/Paris")
    changed = data.copy()
    changed.loc[changed.timestamp_utc >= poison_day, "actual"] = 10000.
    a = variants.run_variant_policy(data, settings(), variant("xgb_weighted_dwt"))
    b = variants.run_variant_policy(changed, settings(), variant("xgb_weighted_dwt"))
    before = data.timestamp_utc < poison_day + pd.DateOffset(days=1)
    fields = ["spike_probability", "raw_correction", "candidate_forecast", "candidate_q10", "candidate_q90", "threshold_eur_mwh"]
    pd.testing.assert_frame_equal(a.predictions.loc[before, fields], b.predictions.loc[before, fields], check_exact=True)


@pytest.mark.parametrize("bad", [
    {"id": "xgb_weighted_fixed", "weighted": False, "threshold_kind": "fixed"},
    {"id": "xgb_weighted_fixed", "weighted": 1, "threshold_kind": "fixed"},
    {"id": "xgb_unweighted_fixed", "weighted": False, "threshold_kind": "fixed", "foo": 1},
])
def test_mislabelled_or_unknown_variants_are_rejected(bad):
    with pytest.raises(base.ScarcityPolicyError):
        variants.validate_variant(bad)
