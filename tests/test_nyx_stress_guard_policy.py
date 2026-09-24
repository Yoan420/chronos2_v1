"""Chronological policy and interval-envelope checks; no real neural/XGB fit."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nyx_scarcity import policy as base
from nyx_stress_guard import policy


FEATURE = "feature_fundamental_signal"
RANK = "feature_fundamental_stress_core_rank_local_pressure"
NAMES = [FEATURE, RANK]
ZONES = ("BE", "DE", "FR", "NL")


def make_panel(days=105):
    start = pd.Timestamp("2026-01-01", tz="Europe/Paris")
    end = start+pd.DateOffset(days=days)
    times = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    day = times.tz_convert("Europe/Paris").tz_localize(None).normalize()
    hour = times.tz_convert("Europe/Paris").hour
    signal = np.where(np.isin(hour, [18, 19, 20, 21]), .8, .1)
    return pd.concat([pd.DataFrame({
        "zone": zone, "timestamp_utc": times,
        "forecast_origin_utc": (day-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC"),
        "label_available_at_utc": (day-pd.Timedelta(days=1)+pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC"),
        "forecast": 100., "q10": 75., "q90": 130.,
        "actual": 100+np.where(signal>.5, 120., np.where(hour == 12, 0., -30.)),
        "benchmark_forecast": 999., "feature_eligible": True, "forecast_eligible": True,
        "label_eligible": True, FEATURE: signal,
    }) for zone in ZONES], ignore_index=True)


def pars():
    return base._parameters({"feature_columns": NAMES, "required_feature_columns": [FEATURE],
        "max_iter": 60, "min_samples_leaf": 40, "threads": 1})


def origin(day="2026-04-10"):
    return (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")


class Classifier:
    fitted = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def fit(self, X, y):
        self.fitted.append((X.copy(), y.copy(), self.kwargs.copy()))
        return self
    def predict(self, X, *, output_margin):
        assert output_margin
        return X[:, 0]


class Platt:
    fitted = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def fit(self, X, y):
        self.fitted.append((X.copy(), y.copy()))
        return self
    def predict_proba(self, X):
        p = np.clip(X[:, 0], 0, 1)
        return np.column_stack([1-p, p])


@pytest.fixture
def mocks(monkeypatch):
    seen = {"references": [], "cdf": []}
    Classifier.fitted.clear()
    Platt.fitted.clear()
    monkeypatch.setattr(policy, "_xgb_classifier", lambda: Classifier)
    monkeypatch.setattr(policy, "LogisticRegression", Platt)
    def build(panel):
        frame = panel.copy()
        frame[RANK] = np.nan
        return frame, NAMES.copy(), [FEATURE], {"synthetic": True}
    def reference(core, zones, *, cutoff):
        seen["references"].append(core.copy(deep=True))
        return {"maximum": float(core[FEATURE].max()), "cutoff": cutoff, "zones": tuple(zones)}
    def apply(frame, state):
        out = frame.copy()
        out[RANK] = out[FEATURE]/(state["maximum"]+1.)
        return out
    real_fit = policy.fit_distributions
    def cdf(X, errors, zones, **kwargs):
        seen["cdf"].append((X.copy(), errors.copy(), zones.copy(), kwargs.copy()))
        return real_fit(X, errors, zones, **kwargs)
    monkeypatch.setattr(policy, "make_stress_features", build)
    monkeypatch.setattr(policy, "fit_reference", reference)
    monkeypatch.setattr(policy, "apply_reference", apply)
    monkeypatch.setattr(policy, "fit_distributions", cdf)
    return seen


def fit(frame=None, *, day="2026-04-10", callback=None):
    panel = make_panel() if frame is None else frame
    augmented, _, _, _ = policy.make_stress_features(panel)
    p = pars()
    prepared = base._prepare(augmented, p)
    state, record = policy.fit_model(prepared, day, origin(day), p, ZONES, on_last_fit=callback)
    return state, record, p, prepared


def test_fit_has_chronological_core_calibration_and_price_blind_reference(mocks):
    state, record, p, prepared = fit()
    assert record["status"] == "trained"
    reference = mocks["references"][-1]
    core_x, errors, zones, kwargs = mocks["cdf"][-1]
    assert reference._day.lt(record["calibration_start_day"]).all()
    assert reference.label_available_at_utc.le(record["fit_cutoff_utc"]).all()
    assert len(errors) == len(reference) == record["cdf_core_rows"]
    assert (errors < 0).any() and (errors == 0).any() and (errors >= 1).any()
    assert kwargs["kind"] == "empirical"
    assert core_x.shape[1] == len(NAMES)+len(ZONES)
    assert np.isfinite(core_x).all() and (core_x >= 0).all() and (core_x <= 1).all()
    assert len(Platt.fitted[-1][0]) == record["calibration_rows"]
    assert Classifier.fitted[-1][2]["scale_pos_weight"] == 1.
    assert state["probability_gate"] == .5 and not record["country_specific_gate"]


def test_future_and_calibration_feature_changes_do_not_fit_the_reference_or_cdf(mocks):
    frame = make_panel()
    first, record, _, _ = fit(frame)
    reference = mocks["references"][-1].copy()
    cdf = deepcopy(mocks["cdf"][-1])
    days = frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    frame.loc[days.ge(record["calibration_start_day"]), FEATURE] = 1e9
    frame.loc[days.ge("2026-04-10"), "actual"] = -1e9
    second, after, _, _ = fit(frame)
    assert first["reference"] == second["reference"]
    assert first["thresholds"] == second["thresholds"]
    pd.testing.assert_frame_equal(reference, mocks["references"][-1])
    np.testing.assert_array_equal(cdf[0], mocks["cdf"][-1][0])
    np.testing.assert_array_equal(cdf[1], mocks["cdf"][-1][1])
    assert after.get("cdf_model_training_end_day", after["model_training_end_day"]) < after["calibration_start_day"]


def test_late_and_ineligible_core_labels_are_not_used(mocks):
    frame = make_panel()
    _, before, _, _ = fit(frame)
    frame.loc[:4, "label_available_at_utc"] = pd.Timestamp("2027-01-01", tz="UTC")
    frame.loc[5:9, "label_eligible"] = False
    _, after, _, _ = fit(frame)
    assert after["cdf_core_rows"] == before["cdf_core_rows"]-10
    assert mocks["references"][-1].label_available_at_utc.le(origin()).all()


@pytest.mark.parametrize("condition,reason", [("warmup", "insufficient_training_days"),
    ("features", "insufficient_training_days"), ("no_cal_events", "insufficient_chronological_calibration_events")])
def test_fallbacks_are_explicit(condition, reason, mocks):
    frame = make_panel()
    day = "2026-02-01" if condition == "warmup" else "2026-04-10"
    if condition == "features":
        frame.loc[frame.zone.eq("DE"), "feature_eligible"] = False
    if condition == "no_cal_events":
        days = frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
        frame.loc[days.ge("2026-03-13"), "actual"] = 0.
    state, record, _, _ = fit(frame, day=day)
    assert state is None and record["reason"] == reason


def test_prediction_needs_only_exogenous_features_and_identical_half_gate_all_zones(mocks):
    state, _, p, _ = fit()
    frame = make_panel()
    current = frame.loc[frame.forecast_origin_utc.eq(origin()) & frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour.eq(19),
                        [*policy.KEYS, FEATURE]].copy()
    for probability in (.49, .5, np.nextafter(.5, 1.), .8):
        current[FEATURE] = probability
        prob, raw, threshold, detail = policy.predict_model(state, current, p)
        assert prob.shape == raw.shape == (4,)
        assert detail.strong_risk_gate.eq(probability > .5).all()
        assert detail.physical_gate_passed.eq(True).all()
        assert (raw >= threshold).all() if probability > .5 else (raw == 0).all()
        assert (detail.mixture_error_q50 >= threshold).all() if probability > .5 else (detail.mixture_error_q50 < threshold).all()
    baseline = policy.predict_model(state, current, p)
    current["actual"], current["forecast"], current["benchmark_forecast"], current["q10"], current["q90"] = 9999., -9999., 1e8, -1e8, 1e8
    same = policy.predict_model(state, current, p)
    for i in range(3):
        np.testing.assert_array_equal(baseline[i], same[i])


@pytest.mark.parametrize("invalid", [pd.NaT, pd.Timestamp("2027-01-01", tz="UTC"), pd.Timestamp("2026-01-01")])
def test_invalid_or_future_model_cutoff_fails(invalid, mocks):
    state, _, p, _ = fit()
    current = make_panel().loc[lambda f: f.forecast_origin_utc.eq(origin()), [*policy.KEYS, FEATURE]]
    state["fit_cutoff"] = invalid
    with pytest.raises(base.ScarcityPolicyError):
        policy.predict_model(state, current, p)


def sample_decisions(days=35):
    frame = make_panel(days=days)
    hours = frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    proposed = hours.isin([18, 19, 20, 21])
    frame["expert_ready"] = True
    frame["raw_correction"] = np.where(proposed, 500., 0.)
    frame["bounded_correction"] = np.where(proposed, 400., 0.)
    frame["selected_weight"] = .25
    frame["applied_correction"] = np.where(proposed, 100., 0.)
    frame["candidate_forecast"] = frame.forecast+frame.applied_correction
    frame["candidate_q10"], frame["candidate_q90"] = frame.q10, np.maximum(frame.q90, frame.candidate_forecast)
    frame["mixture_error_q10"], frame["mixture_error_q50"], frame["mixture_error_q90"] = -10., np.where(proposed, 500., 10.), 800.
    frame["gate_reason"] = "synthetic_governor_choice"
    frame["proposal_reason"] = np.where(proposed, "probability_above_half_coherent_median", "probability_not_above_half")
    return frame


@pytest.mark.parametrize("direct", [False, True])
def test_envelope_then_calibration_never_reduces_pathwise_baseline_coverage(direct):
    frame = sample_decisions()
    out, audit = policy.finalize_intervals(frame, direct=direct)
    active = out.intervention_active
    expected = 400. if direct else 100.
    assert out.loc[active, "applied_correction"].eq(expected).all()
    assert out.loc[~active, "selected_weight"].eq(0.).all()
    assert (out.candidate_q10 <= frame.q10).all() and (out.candidate_q90 >= frame.q90).all()
    base_covered = frame.actual.between(frame.q10, frame.q90)
    assert out.loc[base_covered, "actual"].between(out.loc[base_covered, "candidate_q10"], out.loc[base_covered, "candidate_q90"]).all()
    assert (np.diff(out[["candidate_q10", "candidate_forecast", "candidate_q90"]], axis=1) >= 0).all()
    assert "interval_calibration_status" in out and "interval_calibration_source" in out
    assert out.interval_calibration_cutoff_utc.eq(out.forecast_origin_utc).all()
    assert not audit["p50_modified"]


def test_interval_cap_matches_point_correction_cap_instead_of_hardcoded400():
    frame = sample_decisions(days=1)
    frame["bounded_correction"] = np.where(frame.raw_correction.gt(0), 600., 0.)
    frame["raw_correction"] = np.where(frame.raw_correction.gt(0), 700., 0.)
    frame["mixture_error_q50"] = np.where(frame.raw_correction.gt(0), 700., 0.)
    out, _ = policy.finalize_intervals(frame, direct=True, correction_clip_eur_mwh=600.)
    assert out.loc[out.intervention_active, "candidate_forecast"].eq(700.).all()
    assert out.loc[out.intervention_active, "candidate_q90"].eq(700.).all()


def test_interval_only_ablation_preserves_all_point_decisions_and_base_coverage():
    frame = sample_decisions()
    out, audit = policy.recalibrate_previous(frame)
    pd.testing.assert_series_equal(out.candidate_forecast, frame.candidate_forecast)
    pd.testing.assert_series_equal(out.applied_correction, frame.applied_correction)
    pd.testing.assert_series_equal(out.selected_weight, frame.selected_weight)
    assert (out.candidate_q10 <= frame.q10).all() and (out.candidate_q90 >= frame.q90).all()
    assert not audit["p50_modified"]


def test_policy_interval_replay_does_not_use_current_or_future_labels():
    frame = sample_decisions(days=15)
    first, _ = policy.finalize_intervals(frame, direct=True)
    changed = frame.copy()
    boundary = origin("2026-01-13")
    later = changed.forecast_origin_utc.ge(boundary)
    changed.loc[later, "actual"] = -1e9
    second, _ = policy.finalize_intervals(changed, direct=True)
    pd.testing.assert_series_equal(first.candidate_forecast, second.candidate_forecast)
    pd.testing.assert_frame_equal(first.loc[~later, ["candidate_q10", "candidate_q90"]],
                                 second.loc[~later, ["candidate_q10", "candidate_q90"]])


def test_full_synthetic_replay_keeps_governor_and_direct_separate(mocks):
    panel = make_panel(days=99)
    out = policy.run_stress_policy(panel, pars())
    assert set(out) == {"direct", "governed"}
    for decision, result in out.items():
        pd.testing.assert_frame_equal(result.predictions[panel.columns], panel, check_exact=True)
        assert result.audit["decision_policy"] == decision
        assert result.audit["strict_governor_enforced_in_this_point_forecast"] == (decision == "governed")
        assert result.predictions.probability_gate.eq(.5).all()
    assert out["direct"].predictions.intervention_active.any()
    assert not out["governed"].predictions.intervention_active.any()  # Too few previous issued OOS days.
    assert out["governed"].predictions.candidate_forecast.equals(panel.forecast)
