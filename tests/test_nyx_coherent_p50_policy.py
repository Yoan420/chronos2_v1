"""Causality, coherence and output-policy tests without operational writes."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from nyx_scarcity import policy as base
from nyx_coherent_p50 import policy
from test_nyx_fundamental_stress_policy import panel, parameters, NAMES, FEATURE


def fixture_data(days=99):
    original = panel(days=days, zones=("FR", "DE"))
    p = parameters()
    prepared = base._prepare(original, p)
    source = original.copy()
    source["expert_ready"] = False
    source["expert_fit_day"] = None
    source["spike_probability"] = np.nan
    source["threshold_eur_mwh"] = np.nan
    source["risk_probability_gate"] = np.nan
    source["physical_gate_passed"] = False
    source["predicted_signed_residual_median"] = -10.
    folds = []
    for i in range(0, days, 7):
        day = (pd.Timestamp("2026-01-01")+pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        cutoff = (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
        split = (pd.Timestamp(day)-pd.Timedelta(days=28)).strftime("%Y-%m-%d")
        train = prepared.loc[prepared._day.lt(day) & prepared.label_available_at_utc.le(cutoff)]
        core = train.loc[train._day.lt(split)]
        ready = i >= 90
        thresholds = {z: max(50., float(core.loc[core.zone.eq(z), "_error"].quantile(.95))) for z in ("FR", "DE")} if ready else {}
        folds.append({"fit_day": day, "fit_cutoff_utc": cutoff, "training_rows": len(train),
            "training_start_day": "2026-01-01", "training_days": i, "full_365_day_training": False,
            "calibration_start_day": split, "signed_model_training_rows": len(core),
            "status": "trained" if ready else "fallback", "reason": "" if ready else "insufficient_training_days",
            "thresholds_eur_mwh": thresholds})
        if ready:
            local_days = source.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
            until = (pd.Timestamp(day)+pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            mask = local_days.ge(day) & local_days.lt(until)
            source.loc[mask, "expert_ready"] = True
            source.loc[mask, "expert_fit_day"] = day
            source.loc[mask, "spike_probability"] = source.loc[mask, FEATURE]
            source.loc[mask, "threshold_eur_mwh"] = source.loc[mask, "zone"].map(thresholds)
            source.loc[mask, "risk_probability_gate"] = 1/6
            source.loc[mask, "physical_gate_passed"] = True
    return original, source, pd.DataFrame(folds), p


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(policy, "make_fundamental_features", lambda frame, variant:
        (frame.copy(), list(NAMES), [FEATURE], {"synthetic_test": True}))
    return fixture_data()


def test_cdf_fit_uses_exact_core_and_keeps_signed_errors(synthetic, monkeypatch):
    original, source, folds, p = synthetic
    prepared = base._prepare(original, p)
    frozen = folds.loc[folds.status.eq("trained")].iloc[0]
    seen = {}
    def fit(X, errors, zones, **kwargs):
        seen.update(X=X, errors=errors, zones=zones)
        return {"test": True}
    monkeypatch.setattr(policy, "fit_distributions", fit)
    state, record = policy._fit(prepared, frozen.fit_day, frozen.fit_cutoff_utc, p, ("DE", "FR"),
        source_folds=folds.set_index("fit_day", drop=False), kind="empirical")
    assert len(seen["errors"]) == frozen.signed_model_training_rows
    assert (seen["errors"] < 0).any() and (seen["errors"] >= 1).any()
    assert record["cdf_model_training_end_day"] < frozen.calibration_start_day
    assert record["cdf_max_label_available_at_utc"] <= frozen.fit_cutoff_utc
    assert not record["cdf_uses_calibration_labels"]
    assert state["features"] == NAMES


@pytest.mark.parametrize("column", ["training_rows", "signed_model_training_rows", "calibration_start_day", "thresholds_eur_mwh"])
def test_changed_frozen_training_contract_is_rejected(synthetic, column):
    original, source, folds, p = synthetic
    at = folds.loc[folds.status.eq("trained")].index[0]
    frozen = folds.loc[at].copy()
    if column == "thresholds_eur_mwh":
        folds.at[at, column] = {"FR": 50., "DE": 50.}
    elif column == "calibration_start_day":
        folds.at[at, column] = "2026-01-01"
    else:
        folds.at[at, column] += 1
    with pytest.raises(base.ScarcityPolicyError):
        policy._fit(base._prepare(original, p), frozen.fit_day, frozen.fit_cutoff_utc, p, ("DE", "FR"),
            source_folds=folds.set_index("fit_day", drop=False), kind="empirical")


def test_delayed_label_cannot_silently_enter_core(synthetic):
    original, source, folds, p = synthetic
    original.loc[0, "label_available_at_utc"] = pd.Timestamp("2027-01-01", tz="UTC")
    frozen = folds.loc[folds.status.eq("trained")].iloc[0]
    with pytest.raises(base.ScarcityPolicyError, match="core differs"):
        policy._fit(base._prepare(original, p), frozen.fit_day, frozen.fit_cutoff_utc, p, ("DE", "FR"),
            source_folds=folds.set_index("fit_day", drop=False), kind="empirical")


def test_complete_replay_keeps_inputs_probabilities_and_corrects_majority_spikes(synthetic):
    original, source, folds, p = synthetic
    result = policy.run_coherent_policy(original, source, folds, p, "empirical")
    direct, governed = result["direct"], result["governed"]
    for model in (direct, governed):
        pd.testing.assert_frame_equal(model.predictions[original.columns], original, check_exact=True)
        pd.testing.assert_series_equal(model.predictions.spike_probability, source.spike_probability)
        assert not model.audit["detector_retrained"]
        assert not model.audit["zero_atom_assumption"]
    out = direct.predictions
    high = out.expert_ready & out.spike_probability.gt(.5)
    assert high.any() and out.loc[high, "mixture_error_q50"].ge(out.loc[high, "threshold_eur_mwh"]).all()
    assert out.loc[high, "candidate_forecast"].ge(220.).all()
    assert out.loc[high, "previous_signed_error_median"].lt(0).all()
    assert out.loc[~out.expert_ready, "candidate_forecast"].eq(out.loc[~out.expert_ready, "forecast"]).all()
    assert not direct.audit["strict_governor_enforced_in_this_point_forecast"]
    assert governed.audit["strict_governor_enforced_in_this_point_forecast"]
    assert (np.diff(out[["candidate_q10", "candidate_forecast", "candidate_q90"]], axis=1) >= 0).all()


def minimal_result():
    out = pd.DataFrame({"forecast": [100., 100., 100.], "q10": [60., 60., 60.], "q90": [130., 130., 130.],
        "raw_correction": [500., 50., 0.], "bounded_correction": [400., 50., 0.], "expert_ready": [True]*3,
        "selected_weight": [.25, 1., 1.], "mixture_error_q10": [-20., -30., -50.],
        "mixture_error_q50": [500., 50., -20.], "mixture_error_q90": [800., 100., 30.],
        "proposal_reason": ["ready", "ready", "coherent_median_not_positive"], "gate_reason": ["governed"]*3})
    return base.PolicyResult(out, pd.DataFrame(), pd.DataFrame(), {"config": {"correction_clip_eur_mwh": 400.}})


@pytest.mark.parametrize("direct", [True, False])
def test_final_quantiles_are_function_interpolation_and_clip_monotone(direct):
    result = policy._finalize(minimal_result(), direct=direct)
    out = result.predictions
    assert out.loc[0, "candidate_forecast"] == (500. if direct else 200.)
    assert out.loc[0, "candidate_q90"] == (500. if direct else 222.5)
    assert out.loc[2, "selected_weight"] == 0
    assert out.loc[2, "candidate_q10"] == 60. and out.loc[2, "candidate_q90"] == 130.
    assert out.loc[2, "candidate_forecast"] == 100.
    assert result.audit["probability_is_raw_expert_probability_not_final_distribution_probability"]
    assert not result.audit["interval_finite_sample_coverage_guaranteed"]


def test_source_prediction_mutation_is_rejected(synthetic):
    original, source, folds, p = synthetic
    source.loc[0, "forecast"] += 10
    with pytest.raises(AssertionError):
        policy.run_coherent_policy(original, source, folds, p, "empirical")


def test_future_state_and_input_price_dependence(synthetic):
    original, source, folds, p = synthetic
    frozen = folds.loc[folds.status.eq("trained")].iloc[0]
    state, _ = policy._fit(base._prepare(original, p), frozen.fit_day, frozen.fit_cutoff_utc, p, ("DE", "FR"),
        source_folds=folds.set_index("fit_day", drop=False), kind="empirical")
    mask = source.expert_fit_day.eq(frozen.fit_day)
    current = original.loc[mask, [*policy.KEYS, *NAMES]].copy()
    detector = source.loc[mask].copy()
    baseline = policy.predict_state(state, current, p, detector)
    detector["actual"], detector["forecast"], detector["benchmark_forecast"] = 9999., -8888., 0.
    same = policy.predict_state(state, current, p, detector)
    np.testing.assert_array_equal(baseline[1], same[1])
    state["fit_cutoff"] = pd.Timestamp("2027-01-01", tz="UTC")
    with pytest.raises(base.ScarcityPolicyError, match="precede"):
        policy.predict_state(state, current, p, detector)


@pytest.mark.parametrize("kind", ["unknown", "fixed", "calendar"])
def test_unregistered_kind_rejected(synthetic, kind):
    with pytest.raises(base.ScarcityPolicyError):
        policy.run_coherent_policy(*synthetic[:3], synthetic[3], kind)
