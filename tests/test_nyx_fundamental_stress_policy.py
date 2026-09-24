"""Isolated tests; no API calls, full backtests, source writes or installations."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nyx_scarcity import policy as base
from nyx_fundamental_stress import policy as fundamental


FEATURE = "feature_fundamental_signal"
NAMES = [FEATURE, fundamental.LOCAL_PRESSURE, fundamental.PEER_PRESSURE, fundamental.RESIDUAL_RAMP]


class Classifier:
    fitted = []
    def __init__(self, **kwargs):
        self.parameters = kwargs
    def fit(self, X, y):
        self.fitted.append((X.copy(), y.copy(), self.parameters.copy()))
        return self
    def predict(self, X, *, output_margin):
        assert output_margin is True
        return X[:, 0]


class Platt:
    fitted = []
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y):
        self.fitted.append((X.copy(), y.copy()))
        return self
    def predict_proba(self, X):
        return np.column_stack([1-X[:, 0], X[:, 0]])


class SignedMedian:
    fitted = []
    value = 25.
    def __init__(self, **kwargs):
        self.parameters = kwargs
    def fit(self, X, y):
        self.fitted.append((X.copy(), y.copy(), self.parameters.copy()))
        return self
    def predict(self, X):
        return np.full(len(X), self.value)


@pytest.fixture
def models(monkeypatch):
    for cls in (Classifier, Platt, SignedMedian):
        cls.fitted = []
    monkeypatch.setattr(fundamental, "_xgb_classifier", lambda: Classifier)
    monkeypatch.setattr(fundamental, "LogisticRegression", Platt)
    monkeypatch.setattr(fundamental, "HistGradientBoostingRegressor", SignedMedian)


def panel(days=105, zones=("FR", "DE", "BE", "NL")):
    start = pd.Timestamp("2026-01-01", tz="Europe/Paris")
    end = start+pd.DateOffset(days=days)
    stamps = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    civil = stamps.tz_convert("Europe/Paris").tz_localize(None).normalize()
    hour = stamps.tz_convert("Europe/Paris").hour
    signal = np.where(np.isin(hour, [18, 19, 20, 21]), .8, .1)
    return pd.concat([pd.DataFrame({
        "zone": zone, "timestamp_utc": stamps,
        "forecast_origin_utc": (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC"),
        "label_available_at_utc": (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC"),
        "forecast": 100., "q10": 75., "q90": 130.,
        "actual": 100+np.where(signal>.5, 120., np.where(hour==12, 0., -30.)),
        "benchmark_forecast": 999., "feature_eligible": True, "forecast_eligible": True,
        "label_eligible": True, FEATURE: signal,
        fundamental.LOCAL_PRESSURE: 1.+hour/24., fundamental.PEER_PRESSURE: 1.+hour/24.,
        fundamental.RESIDUAL_RAMP: np.where(hour>=18, 3., -1.),
    }) for zone in zones], ignore_index=True)


def parameters():
    return base._parameters({"feature_columns": NAMES, "required_feature_columns": [FEATURE],
        "max_iter": 60, "min_samples_leaf": 40, "threads": 1})


def fitted(data=None, *, variant="fundamental", day="2026-04-10", callback=None):
    source = panel() if data is None else data
    p = parameters()
    prepared = base._prepare(source, p)
    cutoff = (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    return fundamental._fit(prepared, day, cutoff, p, tuple(sorted(source.zone.unique())),
                            variant=variant, on_last_fit=callback)


def inference(variant="fundamental"):
    current = panel(days=1).loc[lambda d: d.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour.eq(19)].copy()
    current = current[["zone", "timestamp_utc", "forecast_origin_utc", *NAMES]]
    current[FEATURE] = .1
    current[fundamental.LOCAL_PRESSURE] = 2.
    current[fundamental.PEER_PRESSURE] = 2.
    current[fundamental.RESIDUAL_RAMP] = 1.
    state = {"variant": variant, "features": list(NAMES), "zones": tuple(sorted(current.zone.unique())),
        "classifier": Classifier(), "calibrator": Platt(), "signed_model": SignedMedian(),
        "thresholds": {z: 50. for z in current.zone}, "risk_probability_gate_by_zone": {z: .05 for z in current.zone},
        "physical_pressure_q90_by_zone": {z: 1.5 for z in current.zone},
        "fit_day": "2026-01-01", "fit_cutoff": current.forecast_origin_utc.iloc[0]}
    return state, current


def test_signed_model_trains_on_all_core_errors_including_negative_and_zero(models):
    state, audit = fitted()
    X, y, kwargs = SignedMedian.fitted[-1]
    assert audit["status"] == "trained"
    assert len(y) == audit["signed_model_training_rows"]
    assert (y < 0).any() and (y == 0).any() and (y > 0).any()
    assert len(y) > audit["training_tail_rows"]
    assert kwargs["loss"] == "quantile" and kwargs["quantile"] == .5 and not kwargs["early_stopping"]
    assert Classifier.fitted[-1][2]["scale_pos_weight"] == 1.
    assert Classifier.fitted[-1][2]["n_estimators"] == 60
    assert state["calibration_input"] == "raw_margin"
    assert "tail_log_errors" not in state


def test_event_and_physical_thresholds_exclude_calibration_and_future(models):
    source = panel()
    first, audit = fitted(source)
    changed = source.copy()
    civil = source.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    recent = civil.ge(audit["calibration_start_day"])
    changed.loc[recent, [fundamental.LOCAL_PRESSURE, fundamental.PEER_PRESSURE]] = 1e8
    future = civil.ge("2026-04-10")
    changed.loc[future, "actual"] = -1e9
    second, after = fitted(changed)
    assert first["thresholds"] == second["thresholds"]
    assert first["risk_probability_gate_by_zone"] == second["risk_probability_gate_by_zone"]
    assert first["physical_pressure_q90_by_zone"] == second["physical_pressure_q90_by_zone"]
    np.testing.assert_array_equal(SignedMedian.fitted[-1][1], SignedMedian.fitted[-2][1])
    assert after["max_label_available_at_utc"] <= after["fit_cutoff_utc"]


def test_delayed_unpublished_labels_are_never_fitted(models):
    source = panel()
    stamp = source.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    delayed = stamp.eq("2026-02-01")
    source.loc[delayed, "actual"] = 987654.
    source.loc[delayed, "label_available_at_utc"] = pd.Timestamp("2026-04-12", tz="UTC")
    _, record = fitted(source)
    assert record["status"] == "trained"
    assert not np.isclose(SignedMedian.fitted[-1][1], 987554.).any()


def test_callback_receives_trained_state_and_parameter_copy(models):
    observed = []
    state, record = fitted(callback=lambda s, p: observed.append((s, p)))
    assert observed[0][0] is state and observed[0][1]["max_iter"] == 60
    assert record["risk_probability_gate_by_zone"] == state["risk_probability_gate_by_zone"]


@pytest.mark.parametrize("day", ["2026-01-01", "2026-03-15"])
def test_warmup_returns_identity_without_fitting(models, day):
    state, audit = fitted(day=day)
    assert state is None and audit["reason"] == "insufficient_training_days"
    assert not SignedMedian.fitted


def test_per_country_coverage_blocks_pooled_shortcut(models):
    source = panel()
    source.loc[source.zone.eq("NL") & source.timestamp_utc.lt(pd.Timestamp("2026-03-01", tz="UTC")), "feature_eligible"] = False
    state, audit = fitted(source)
    assert state is None and audit["reason"] == "insufficient_eligible_training_days"


def test_insufficient_calibration_events_preserves_old_fit_guard(models):
    source = panel()
    recent = source.timestamp_utc.ge(pd.Timestamp("2026-03-13", tz="Europe/Paris"))
    source.loc[recent, "actual"] = 100.
    state, audit = fitted(source)
    assert state is None and audit["reason"] == "insufficient_chronological_calibration_events"


def test_risk_gate_is_country_core_prevalence_not_point_six_or_probability_times_tail():
    state, current = inference()
    probability, raw, threshold, details = fundamental.predict_state(state, current, {"threads": 1})
    np.testing.assert_allclose(probability, .1)
    np.testing.assert_allclose(raw, 25.)
    np.testing.assert_allclose(threshold, 50.)
    np.testing.assert_allclose(details.risk_probability_gate, .05)
    assert details.physical_gate_passed.all()
    assert details.proposal_reason.eq("fundamental_proposal_ready").all()


def test_empirical_zero_and_one_priors_are_explicit_and_gate_is_strict():
    state, current = inference()
    state["risk_probability_gate_by_zone"].update(FR=.1, BE=0., NL=1.)
    _, raw, _, details = fundamental.predict_state(state, current, {"threads": 1})
    assert dict(zip(current.zone, raw)) == {"FR": 0., "DE": 25., "BE": 25., "NL": 0.}
    assert details.loc[current.zone.eq("BE"), "risk_probability_gate"].iloc[0] == 0.


@pytest.mark.parametrize("median", [-20., 0., 700.])
def test_signed_median_not_positive_tail_and_raw_clipping_is_left_to_base(median):
    state, current = inference()
    state["signed_model"].value = median
    _, raw, _, details = fundamental.predict_state(state, current, {"threads": 1})
    np.testing.assert_allclose(raw, max(0., median))
    np.testing.assert_allclose(details.predicted_signed_residual_median, median)


def test_physical_gates_are_target_country_specific_and_strict():
    state, current = inference()
    current.loc[current.zone.eq("FR"), fundamental.RESIDUAL_RAMP] = 0.
    current.loc[current.zone.eq("DE"), fundamental.LOCAL_PRESSURE] = 1.5
    current.loc[current.zone.eq("BE"), fundamental.LOCAL_PRESSURE] = -100.
    current.loc[current.zone.eq("NL"), fundamental.PEER_PRESSURE] = 1.5
    _, raw, _, details = fundamental.predict_state(state, current, {"threads": 1})
    assert dict(zip(current.zone, raw)) == {"FR": 0., "DE": 0., "BE": 25., "NL": 0.}
    assert details.loc[current.zone.eq("DE"), "proposal_reason"].eq("physical_stress_gate_closed").all()


def test_missing_physical_gate_inputs_are_not_zero_imputed():
    state, current = inference()
    current.loc[current.zone.eq("FR"), fundamental.RESIDUAL_RAMP] = np.nan
    _, raw, _, details = fundamental.predict_state(state, current, {"threads": 1})
    assert raw[0] == 0
    assert details.iloc[0].proposal_reason == "physical_gate_inputs_unavailable"


def test_calendar_control_never_reads_physical_gate_inputs():
    state, current = inference("calendar")
    state["features"] = [FEATURE]
    current = current.drop(columns=[fundamental.LOCAL_PRESSURE, fundamental.PEER_PRESSURE, fundamental.RESIDUAL_RAMP])
    state["physical_pressure_q90_by_zone"] = {}
    _, raw, _, details = fundamental.predict_state(state, current, {"threads": 1})
    assert (raw == 25).all() and details.physical_gate_passed.all()


def test_predict_is_invariant_to_actual_nyx_quantiles_storm_and_label_metadata():
    state, current = inference()
    before = fundamental.predict_state(state, current, {"threads": 1})
    for column in ["actual", "forecast", "q10", "q90", "benchmark_forecast", "_error", "label_available_at_utc"]:
        current[column] = "poison"
    after = fundamental.predict_state(state, current, {"threads": 1})
    for a, b in zip(before[:3], after[:3]):
        np.testing.assert_array_equal(a, b)
    pd.testing.assert_frame_equal(before[-1], after[-1])


@pytest.mark.parametrize("prior", [-.1, np.nan, np.inf, 1.1])
def test_invalid_or_missing_saved_core_prior_fails(prior):
    state, current = inference()
    state["risk_probability_gate_by_zone"]["FR"] = prior
    with pytest.raises(base.ScarcityPolicyError, match="prevalences"):
        fundamental.predict_state(state, current, {"threads": 1})


@pytest.mark.parametrize("probability", [np.nan, np.inf, -.1, 1.1])
def test_invalid_model_probability_fails(probability):
    state, current = inference()
    current[FEATURE] = probability
    with pytest.raises(base.ScarcityPolicyError, match="probabilities"):
        fundamental.predict_state(state, current, {"threads": 1})


def test_future_state_cannot_be_used_for_past_forecast():
    state, current = inference()
    state["fit_cutoff"] += pd.Timedelta(days=1)
    with pytest.raises(base.ScarcityPolicyError, match="future fitted"):
        fundamental.predict_state(state, current, {"threads": 1})


@pytest.mark.parametrize("bad", ["forecast", "feature_baseline_forecast", "feature_fundamental_nyx", "feature_fundamental_actual", "feature_fundamental_q90", "feature_fundamental_price"])
def test_forbidden_model_inputs_rejected(bad):
    state, current = inference()
    current[bad] = 1.
    state["features"].append(bad)
    with pytest.raises(base.ScarcityPolicyError, match="Only explicit"):
        fundamental.predict_state(state, current, {"threads": 1})


def feature_builder(source, variant):
    out = source.copy(deep=True)
    names = NAMES if variant == "fundamental" else [FEATURE]
    return out, names, [FEATURE], {"variant": variant, "electricity_price_inputs": False}


@pytest.mark.parametrize("variant", ["fundamental", "calendar"])
def test_run_preserves_originals_and_exposes_correct_functional(models, monkeypatch, variant):
    monkeypatch.setattr(fundamental, "make_fundamental_features", feature_builder)
    source = panel(days=92)
    original = source.copy(deep=True)
    observed = []
    result = fundamental.run_fundamental_policy(source, {"threads": 1}, variant, on_last_fit=lambda s, p: observed.append(s))
    pd.testing.assert_frame_equal(source, original, check_exact=True)
    pd.testing.assert_frame_equal(result.predictions[source.columns], original, check_exact=True)
    assert result.audit["trained_folds"] and observed
    assert result.audit["strict_governor_enforced_in_this_point_forecast"]
    assert not result.audit["positive_tail_only_fit"] and not result.audit["probability_times_tail_correction"]
    assert result.predictions.candidate_forecast.eq(result.predictions.forecast).all()
    ready = result.predictions.expert_ready
    np.testing.assert_allclose(result.predictions.loc[ready, "probability_gate"], 1/6)
    assert result.predictions.candidate_q10.le(result.predictions.candidate_forecast).all()
    assert result.predictions.candidate_forecast.le(result.predictions.candidate_q90).all()


def test_calendar_can_run_with_missing_physical_sources_without_mutating_audit(models, monkeypatch):
    monkeypatch.setattr(fundamental, "make_fundamental_features", feature_builder)
    source = panel(days=92)
    source["feature_eligible"] = False
    source["feature_available_at_utc"] = pd.NaT
    result = fundamental.run_fundamental_policy(source, {"threads": 1}, "calendar")
    assert result.predictions.expert_ready.any()
    assert not result.predictions.feature_eligible.any()
    assert result.predictions.feature_available_at_utc.isna().all()
    pd.testing.assert_frame_equal(result.predictions[source.columns], source, check_exact=True)


def test_fundamental_keeps_physical_source_eligibility(models, monkeypatch):
    monkeypatch.setattr(fundamental, "make_fundamental_features", feature_builder)
    source = panel(days=92)
    source["feature_eligible"] = False
    result = fundamental.run_fundamental_policy(source, {"threads": 1}, "fundamental")
    assert not result.predictions.expert_ready.any()
    assert result.predictions.candidate_forecast.eq(source.forecast).all()


@pytest.mark.parametrize("variant,settings", [("other", {}), ("calendar", {"threads": 3})])
def test_unknown_variant_or_excess_threads_fail(models, monkeypatch, variant, settings):
    monkeypatch.setattr(fundamental, "make_fundamental_features", feature_builder)
    with pytest.raises(base.ScarcityPolicyError):
        fundamental.run_fundamental_policy(panel(days=1), settings, variant)
