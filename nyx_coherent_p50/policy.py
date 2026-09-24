"""Causal distribution replay: a frozen detector, two coherent residual regimes.

The classifier probabilities are read from the verified prior replay, not
retrained or recalibrated. New conditional distributions see fundamentals only.
All current/future labels remain outside fitting and governance. Operational
outputs interpolate quantile functions, not arithmetic CDF mixtures.
"""
from __future__ import annotations

from dataclasses import replace
import json
import numpy as np
import pandas as pd

from nyx_scarcity import policy as base
from nyx_fundamental_stress.features import make_fundamental_features
from nyx_fundamental_stress.policy import _check_features
from .distribution import fit_distributions, predict_quantiles

KEYS = ["zone", "timestamp_utc", "forecast_origin_utc"]
KINDS = ("forest", "empirical")


def _mapping(value):
    return json.loads(value) if isinstance(value, str) else dict(value)


def _source_rows(source, current):
    rows = source.reindex(pd.MultiIndex.from_frame(current[KEYS])).copy()
    rows.index = current.index
    if rows.zone.isna().any():
        raise base.ScarcityPolicyError("A current identity is absent from the frozen detector replay.")
    return rows


def _fit(data, day, cutoff, p, zones, *, source_folds, kind, on_last_fit=None):
    """Train distributions on the exact historical core used by the detector."""
    _check_features(p["feature_columns"])
    if day not in source_folds.index:
        raise base.ScarcityPolicyError("Refit schedule differs from the frozen detector.")
    frozen = source_folds.loc[day].to_dict()
    record = {**frozen, "variant": kind, "detector_retrained": False,
              "distribution_training_scope": "same_core_as_frozen_detector_excluding_28_day_calibration"}
    if pd.Timestamp(frozen["fit_cutoff_utc"]) != cutoff:
        raise base.ScarcityPolicyError("Frozen fit cutoff differs from D-1 08 h.")
    if frozen["status"] != "trained":
        return None, record
    first = max(str(data._day.min()), (pd.Timestamp(day)-pd.Timedelta(days=p["training_window_days"])).strftime("%Y-%m-%d"))
    split = (pd.Timestamp(day)-pd.Timedelta(days=p["calibration_days"])).strftime("%Y-%m-%d")
    usable = (data._day.ge(first) & data._day.lt(day) & data._features_valid
              & data._label_valid & data.label_available_at_utc.le(cutoff))
    train = data.loc[usable]
    core = train.loc[train._day.lt(split)].copy()
    if (len(train) != int(frozen["training_rows"]) or len(core) != int(frozen["signed_model_training_rows"])
            or first != frozen["training_start_day"] or split != frozen["calibration_start_day"]):
        raise base.ScarcityPolicyError("The CDF training core differs from the frozen classifier core.")
    thresholds = _mapping(frozen["thresholds_eur_mwh"])
    for zone in zones:
        errors = core.loc[core.zone.eq(zone), "_error"].to_numpy(float)
        if not len(errors):
            raise base.ScarcityPolicyError("A training country is absent.")
        expected = max(p["minimum_threshold_eur_mwh"], float(np.quantile(errors, p["threshold_quantile"])))
        if zone not in thresholds or not np.isclose(expected, thresholds[zone], rtol=0, atol=1e-10):
            raise base.ScarcityPolicyError("Frozen event threshold no longer matches historical labels.")
    X = base._matrix(core, p["feature_columns"], zones)
    normalized = core._error.to_numpy(float)/core.zone.map(thresholds).to_numpy(float)
    distributions = fit_distributions(X, normalized, core.zone.to_numpy(), kind=kind, threads=p["threads"])
    state = {"distributions": distributions, "features": list(p["feature_columns"]), "zones": tuple(zones),
             "thresholds": thresholds, "kind": kind, "fit_day": day, "fit_cutoff": cutoff}
    record.update(cdf_core_rows=len(core), cdf_normal_rows=int((normalized < 1).sum()),
                  cdf_spike_rows=int((normalized >= 1).sum()), cdf_negative_error_rows=int((normalized < 0).sum()),
                  cdf_max_label_available_at_utc=core.label_available_at_utc.max(),
                  cdf_model_training_end_day=str(core._day.max()),
                  cdf_uses_calibration_labels=False, cdf_support_partition="normalized_error <1 versus >=1")
    if on_last_fit is not None:
        on_last_fit(state, dict(p))
    return state, record


def predict_state(state, current, p, detector_rows):
    """Pure exogenous distribution inference using already issued detector probabilities."""
    _check_features(state["features"])
    if state["kind"] not in KINDS or not current.zone.isin(state["zones"]).all():
        raise base.ScarcityPolicyError("Unknown distribution variant or country.")
    if (not detector_rows[KEYS].reset_index(drop=True).equals(current[KEYS].reset_index(drop=True))
            or not detector_rows.expert_ready.eq(True).all()):
        raise base.ScarcityPolicyError("CDF inference cannot manufacture detector readiness or change identities.")
    timestamps = base._utc(current.timestamp_utc, "timestamp_utc")
    origins = base._utc(current.forecast_origin_utc, "forecast_origin_utc")
    civil = timestamps.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    expected = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
    fitted = pd.Timestamp(state["fit_cutoff"])
    if fitted.tzinfo is None or not origins.eq(expected).all() or origins.lt(fitted).any():
        raise base.ScarcityPolicyError("Current CDF inference cannot precede its fitted D-1 08 h state.")
    if "expert_fit_day" in detector_rows and not detector_rows.expert_fit_day.eq(state["fit_day"]).all():
        raise base.ScarcityPolicyError("CDF and detector states must share the same historical fit date.")
    probability = detector_rows.spike_probability.to_numpy(float)
    threshold = detector_rows.threshold_eur_mwh.to_numpy(float)
    prior = detector_rows.risk_probability_gate.to_numpy(float)
    if (not np.allclose(threshold, current.zone.map(state["thresholds"]).to_numpy(float), rtol=0, atol=1e-10)
            or not np.isfinite(prior).all() or ((prior < 0)|(prior > 1)).any()):
        raise base.ScarcityPolicyError("Invalid or mismatched frozen detector thresholds.")
    X = base._matrix(current, state["features"], state["zones"])
    quantiles, distribution_audit = predict_quantiles(state["distributions"], X, current.zone.to_numpy(),
        probability, threshold, levels=(.1, .5, .9))
    if (quantiles.shape != (len(current), 3) or not np.isfinite(quantiles).all()
            or (np.diff(quantiles, axis=1) < 0).any()
            or ((probability > .5) & (quantiles[:, 1] < threshold)).any()
            or ((probability <= .5) & (quantiles[:, 1] >= threshold)).any()):
        raise base.ScarcityPolicyError("Residual CDF violates its quantile/support coherence contract.")
    physical = detector_rows.physical_gate_passed.eq(True).to_numpy()
    risk = probability > prior
    positive = quantiles[:, 1] > 0.
    raw = np.where(physical & risk & positive, quantiles[:, 1], 0.)
    detail = current[KEYS].copy()
    for i, level in enumerate((10, 50, 90)):
        detail[f"mixture_error_q{level}"] = quantiles[:, i]
    detail["risk_probability_gate"] = prior
    detail["physical_gate_passed"] = physical
    detail["risk_above_core_prevalence"] = risk
    detail["proposal_reason"] = np.select([~physical, ~risk, ~positive],
        ["physical_stress_gate_closed", "risk_not_above_core_prevalence", "coherent_median_not_positive"],
        default="coherent_p50_proposal_ready")
    for name, values in distribution_audit.items():
        detail["cdf_"+name] = values
    return probability, raw, threshold, detail


def _finalize(result, *, direct):
    out = result.predictions.copy(deep=True)
    proposed = out.expert_ready & out.raw_correction.gt(0)
    weights = np.where(proposed, 1. if direct else out.selected_weight.to_numpy(float), 0.)
    out["selected_weight"] = weights
    out["applied_correction"] = weights*out.bounded_correction
    out["candidate_forecast"] = out.forecast+out.applied_correction
    out["candidate_q10"], out["candidate_q90"] = out.q10.copy(), out.q90.copy()
    active = weights > 0
    cap = result.audit["config"]["correction_clip_eur_mwh"]
    for level in (10, 90):
        base_quantile = out[f"q{level}"].to_numpy(float)
        expert_quantile = out.forecast.to_numpy(float)+np.minimum(out[f"mixture_error_q{level}"].to_numpy(float), cap)
        out.loc[active, f"candidate_q{level}"] = ((1-weights)*base_quantile+weights*expert_quantile)[active]
    out["interval_status"] = np.where(active, "monotone_quantile_function_interpolation_not_cdf_mixture",
                                      "baseline_preserved_no_intervention")
    inactive_ready = out.expert_ready & ~proposed
    out.loc[inactive_ready, "gate_reason"] = out.loc[inactive_ready, "proposal_reason"]
    if direct:
        out.loc[proposed, "gate_reason"] = "direct_coherent_p50_after_physical_and_risk_gates"
    ordered = out[["candidate_q10", "candidate_forecast", "candidate_q90"]].to_numpy(float)
    if not np.isfinite(ordered).all() or (np.diff(ordered, axis=1) < 0).any():
        raise base.ScarcityPolicyError("Final quantile interpolation is nonfinite or crossed.")
    audit = {**result.audit, "decision_policy": "direct" if direct else "governed",
             "strict_governor_enforced_in_this_point_forecast": not direct,
             "changed_forecast_rows": int(out.applied_correction.ne(0).sum()),
             "probability_is_raw_expert_probability_not_final_distribution_probability": True,
             "point_functional": "positive_mixture_median_after_risk_physical_gate_and_quantile_interpolation_weight",
             "interval_method": "Qfinal(t)=(1-w)*Q_NYX(t)+w*(NYX_p50+min(Qerror(t),400))",
             "interval_finite_sample_coverage_guaranteed": False}
    return replace(result, predictions=out, audit=audit)


def run_coherent_policy(panel, source_predictions, source_folds, settings, kind, on_last_fit=None):
    if kind not in KINDS or (on_last_fit is not None and not callable(on_last_fit)):
        raise base.ScarcityPolicyError("Use forest or empirical and a callable fit observer.")
    pd.testing.assert_frame_equal(panel, source_predictions[panel.columns], check_exact=True)
    augmented, features, required, feature_audit = make_fundamental_features(panel, variant="fundamental")
    pd.testing.assert_frame_equal(augmented[features], source_predictions[features], check_exact=True)
    p = base._parameters({**settings, "feature_columns": features, "required_feature_columns": required})
    if p["threads"] > 2:
        raise base.ScarcityPolicyError("At most two threads per CDF worker.")
    source = source_predictions.set_index(KEYS, drop=False)
    folds = source_folds.set_index("fit_day", drop=False)
    if source.index.has_duplicates or folds.index.has_duplicates:
        raise base.ScarcityPolicyError("Duplicate source detector identities or refit days.")
    details = []
    def predict(state, current, parameters):
        probability, raw, threshold, detail = predict_state(state, current, parameters, _source_rows(source, current))
        details.append(detail)
        return probability, raw, threshold
    result = base.run_policy(augmented, p, fit_callback=lambda data, day, cutoff, parameters, zones:
        _fit(data, day, cutoff, parameters, zones, source_folds=folds, kind=kind, on_last_fit=on_last_fit),
        predict_callback=predict)
    out = result.predictions.copy(deep=True)
    if not out.expert_ready.equals(source_predictions.expert_ready):
        raise base.ScarcityPolicyError("CDF replay changed the source detector's readiness coverage.")
    if not np.allclose(out.spike_probability, source_predictions.spike_probability, equal_nan=True, rtol=0, atol=0):
        raise base.ScarcityPolicyError("The frozen detector probabilities have changed.")
    if details:
        diagnostics = pd.concat(details, ignore_index=True).set_index(KEYS)
        if diagnostics.index.has_duplicates:
            raise base.ScarcityPolicyError("Duplicate CDF diagnostics.")
        identity = pd.MultiIndex.from_frame(out[KEYS])
        for column in diagnostics:
            out[column] = diagnostics[column].reindex(identity).to_numpy()
    else:
        for column in ("mixture_error_q10", "mixture_error_q50", "mixture_error_q90", "risk_probability_gate"):
            out[column] = np.nan
        out["physical_gate_passed"], out["risk_above_core_prevalence"] = False, False
        out["proposal_reason"] = "expert_warmup_or_unavailable"
    out["probability_gate"] = out.risk_probability_gate
    out["mixture_raw_p50_eur_mwh"] = out.forecast+out.mixture_error_q50
    out["previous_signed_error_median"] = source_predictions.predicted_signed_residual_median
    out["cdf_kind"] = kind
    audit = {**result.audit, "engine": "nyx_coherent_p50_v1", "variant": kind,
             "detector_retrained": False, "detector_probabilities_preserved_exactly": True,
             "features": feature_audit, "electricity_prices_or_nyx_used_as_model_inputs": False,
             "conditional_distribution": "(1-p)*F(error<u|X)+p*F(error>=u|X)",
             "conditional_support": "r=error/u; normal r<1, spike r>=1",
             "quantiles": "left generalized inverse of discrete weighted CDF; no linear interpolation",
             "calibration_days_used_for_cdf_fit": False, "zero_atom_assumption": False,
             "conditional_cdf_uses_all_signed_core_errors": True, "probability_times_tail_correction": False,
             "limitations": ["Already examined exploratory year; no independent prospective validation.",
                 "Progressive 90-to-365-day training cap, not a full 365-day training prefix for every evaluation day.",
                 "Historical as-of feature queries do not independently certify publication at D-1 08 h.",
                 "Missing temperature revisions and incomplete supply proxies; no qualified JAO import domain.",
                 "Only upward point corrections; cannot fix overpredictions or negative spikes.",
                 "Probability calibration and interval coverage must be evaluated, not assumed.",
                 "Positive gating, cap and governance alter the raw expert distribution; p is not final-distribution risk."]}
    result = replace(result, predictions=out, audit=audit)
    return {"direct": _finalize(result, direct=True), "governed": _finalize(result, direct=False)}
