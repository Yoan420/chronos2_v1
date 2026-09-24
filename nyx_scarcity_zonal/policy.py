"""Predeclared zonal challengers; frozen operational predictions stay immutable."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nyx_scarcity import policy as base
from nyx_scarcity import variant_policy as regional
from .calibration import fit_zone_offsets, apply_zone_offsets
from .features import make_zonal_features

VARIANTS = {
    "regional_hiercal": {"context": False, "hierarchical": True},
    "zonal_context": {"context": True, "hierarchical": False},
    "zonal_hiercal": {"context": True, "hierarchical": True},
}
FIXED_VARIANT = {"id": "xgb_unweighted_fixed", "weighted": False, "threshold_kind": "fixed"}


def _interior(probability):
    values = np.asarray(probability, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise base.ScarcityPolicyError("Nonfinite/out-of-range shared probabilities.")
    clipped = np.clip(values, 1e-12, 1-1e-12)
    return clipped, int(np.count_nonzero(clipped != values))


def _fit(data, day, cutoff, p, zones, *, variant, penalty, offset_bound, callback=None):
    def post_fit(state, core, calibration, current, parameters):
        X = base._matrix(calibration, state["features"], state["zones"])
        with threadpool_limits(limits=min(parameters["threads"], 2)):
            margin = state["classifier"].predict(X, output_margin=True)
            shared = state["calibrator"].predict_proba(np.asarray(margin).reshape(-1, 1))[:, 1]
        shared, clips = _interior(shared)
        events = calibration._error.to_numpy(float) >= calibration.zone.map(state["thresholds"]).to_numpy(float)
        if VARIANTS[variant]["hierarchical"]:
            offsets, audit = fit_zone_offsets(calibration, shared, events, cutoff=cutoff,
                current_day=day, penalty=penalty, max_abs=offset_bound)
        else:
            offsets = {z: 0. for z in ("FR", "DE", "BE", "NL")}
            audit = {"method": "shared_platt_unchanged", "offsets": offsets}
        state.update(zonal_variant=variant, zone_offsets=offsets,
                     zonal_calibration_audit={**audit, "probability_clip_epsilon": 1e-12,
                                              "clipped_shared_calibration_probabilities": clips})
        if callback is not None:
            callback(state, parameters)
    state, record = regional._fit(data, day, cutoff, p, zones,
                                 variant=FIXED_VARIANT, on_fit=post_fit)
    record["zonal_variant"] = variant
    if state is not None:
        record["zonal_calibration"] = state["zonal_calibration_audit"]
    return state, record


def predict_state(state: dict, current: pd.DataFrame, settings: dict) -> tuple:
    """Pure model inference; no observation, target, governance or file reads."""
    X = base._matrix(current, state["features"], state["zones"])
    threshold = current.zone.map(state["thresholds"]).to_numpy(float)
    with threadpool_limits(limits=min(settings["threads"], 2)):
        margins = state["classifier"].predict(X, output_margin=True)
        shared = state["calibrator"].predict_proba(np.asarray(margins).reshape(-1, 1))[:, 1]
        severity = state["severity"].predict(X)
    shared, clipped = _interior(shared)
    adjusted = apply_zone_offsets(shared, current.zone.to_numpy(), state["zone_offsets"])
    level = np.where(adjusted > .5, 1-.5/np.maximum(adjusted, .5), 0.)
    error_quantile = np.quantile(state["tail_log_errors"], level)
    amplitude = threshold*np.exp(np.clip(severity+error_quantile, -20, 20))
    raw = np.where(adjusted > max(.5, settings["probability_gate"]), np.maximum(threshold, amplitude), 0.)
    diagnostic = current[["zone", "timestamp_utc", "forecast_origin_utc"]].copy()
    diagnostic["probability_before_zonal"] = shared
    diagnostic["zonal_log_odds_offset"] = current.zone.map(state["zone_offsets"]).to_numpy(float)
    diagnostic["shared_probability_numeric_clip_count_in_batch"] = clipped
    return adjusted, raw, threshold, diagnostic


def run_zonal_policy(panel: pd.DataFrame, settings: dict, variant: str, *,
                     offset_penalty: float = 1., offset_bound: float = 3.,
                     on_last_fit: Callable | None = None) -> base.PolicyResult:
    if variant not in VARIANTS:
        raise base.ScarcityPolicyError("Unknown predeclared zonal variant.")
    if not np.isfinite(offset_penalty) or offset_penalty <= 0 or not np.isfinite(offset_bound) or offset_bound <= 0:
        raise base.ScarcityPolicyError("Positive finite calibration penalty/bound required.")
    regional._xgb_classifier()
    p = base._parameters(settings)
    if p["threads"] > 2:
        raise base.ScarcityPolicyError("At most two threads per zonal worker.")
    augmented, options = panel.copy(deep=True), dict(settings)
    feature_audit = {"kind": "frozen_original_regional_features"}
    if VARIANTS[variant]["context"]:
        augmented, names, required, feature_audit = make_zonal_features(panel)
        options.update(feature_columns=names, required_feature_columns=required)
    diagnostics = []
    def predict(state, current, pars):
        probability, raw, threshold, details = predict_state(state, current, pars)
        diagnostics.append(details)
        return probability, raw, threshold
    result = base.run_policy(augmented, options,
        fit_callback=lambda data, day, cutoff, pars, zones: _fit(data, day, cutoff, pars, zones,
            variant=variant, penalty=offset_penalty, offset_bound=offset_bound, callback=on_last_fit),
        predict_callback=predict)
    predictions = result.predictions.copy()
    keys = ["zone", "timestamp_utc", "forecast_origin_utc"]
    if diagnostics:
        details = pd.concat(diagnostics, ignore_index=True).set_index(keys)
        if details.index.has_duplicates:
            raise base.ScarcityPolicyError("Duplicate model-inference audit identities.")
        identity = pd.MultiIndex.from_frame(predictions[keys])
        for column in details:
            predictions[column] = details[column].reindex(identity).to_numpy()
    predictions["zonal_variant"] = variant
    return replace(result, predictions=predictions, audit={**result.audit,
        "engine": "nyx_scarcity_zonal_v1", "variant": variant, "zonal_features": feature_audit,
        "zonal_offset_penalty": offset_penalty, "zonal_offset_bound": offset_bound,
        "hierarchical_calibration": VARIANTS[variant]["hierarchical"],
        "severity_algorithm_changed": False, "severity_inputs_changed": VARIANTS[variant]["context"],
        "severity_refitted_in_each_fold": True,
        "probability_adjustment_changes_gate_and_tail_quantile": True,
        "native_threshold_kind": "fixed_per_training_core",
        "governance_policy_changed": False, "predetermined_fixed_shrinkage_is_separate": True})


def fixed_conservative_forecast(result: base.PolicyResult, *, alpha: float = .25) -> base.PolicyResult:
    """A predeclared point policy, with causal empirical intervals of its own.

    This is NOT the strict annual-guarded model. Both policies are kept and
    reported independently, and neither is activated in the operational path.
    """
    if alpha != .25:
        raise base.ScarcityPolicyError("The zonal conservative candidate is predeclared at 25%, not selected ex post.")
    original = result.predictions
    p = base._parameters(result.audit["config"])
    data = base._prepare(original, p)
    oos, blocks = [], []
    for day, source in data.groupby("_day", sort=True):
        current = source.copy()
        cutoff = current.forecast_origin_utc.iloc[0]
        current["strict_governed_forecast"] = current.candidate_forecast
        current["strict_governed_weight"] = current.selected_weight
        current["strict_governance_reason"] = current.gate_reason
        current["selected_weight"] = np.where(current.expert_ready, alpha, 0.)
        current["applied_correction"] = current.selected_weight * current.bounded_correction
        current["candidate_forecast"] = current.forecast + current.applied_correction
        current[base._column(alpha)] = current.forecast + alpha*current.bounded_correction
        current["_tail_gate"] = current.expert_ready & current.bounded_correction.gt(0)
        current["gate_reason"] = np.where(~current.expert_ready, "expert_unavailable_preserve_nyx",
            np.where(current._tail_gate, "experimental_fixed_25_percent_zonal_probability_gate", "probability_below_gate"))
        current["candidate_q10"], current["candidate_q90"] = current.q10, current.q90
        current["interval_status"] = "baseline_preserved_no_point_change"
        for zone, part in current.groupby("zone", sort=True):
            active = part.applied_correction.gt(0)
            if not active.any():
                continue
            past = base._eligible_oos(oos, zone, day, cutoff, p)
            lo, hi, status = base._intervals(part, past, alpha, p)
            indices = part.index[active]
            current.loc[indices, "candidate_q10"] = lo[active.to_numpy()]
            current.loc[indices, "candidate_q90"] = hi[active.to_numpy()]
            current.loc[indices, "interval_status"] = status[active.to_numpy()]
        blocks.append(current)
        if current.expert_ready.any():
            oos.append(current)
        first = (pd.Timestamp(day)-pd.Timedelta(days=p["governance_lookback_days"])).strftime("%Y-%m-%d")
        oos = [item for item in oos if str(item._day.iloc[0]) >= first]
    computed = pd.concat(blocks).sort_values("_row").reset_index(drop=True)
    out = original.copy(deep=True)
    revised = ["selected_weight", "applied_correction", "candidate_forecast", "candidate_q10", "candidate_q90",
               "gate_reason", "interval_status", "strict_governed_forecast", "strict_governed_weight", "strict_governance_reason"]
    for column in revised:
        out[column] = computed[column].to_numpy()
    audit = {**result.audit, "decision_policy": "predeclared_fixed_25_percent_with_zonal_probability_gate",
             "strict_governor_enforced_in_this_point_forecast": False, "fixed_alpha": alpha,
             "annual_non_regression_guaranteed": False, "amplitude_selected_on_evaluation": False,
             "changed_forecast_rows": int(out.applied_correction.ne(0).sum()),
             "intervals": "past-known prequential own-policy signed errors; baseline preserved on unchanged points"}
    return replace(result, predictions=out, audit=audit)
