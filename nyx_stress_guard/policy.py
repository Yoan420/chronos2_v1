"""Predeclared physical-profile challenger; no operational model is changed.

One calibrated probability, a coherent empirical residual CDF, and the same
structural p>0.5 gate in every zone. No manually fitted country pressure gate.
The primary decision is the existing causal governor; direct is diagnostic.
"""
from __future__ import annotations

from dataclasses import replace
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from nyx_scarcity import policy as base
from nyx_scarcity.variant_policy import _xgb_classifier
from nyx_coherent_p50.distribution import fit_distributions, predict_quantiles
from nyx_fundamental_stress.policy import _check_features
from .features import make_stress_features, fit_reference, apply_reference
from .intervals import calibrate_intervals

KEYS = ["zone", "timestamp_utc", "forecast_origin_utc"]
EVENT_GATE = .5
LOGGER = logging.getLogger(__name__)


def fit_model(data, day, cutoff, p, zones, *, on_last_fit=None):
    """Explicit chronological core/calibration split; all labels as-of cutoff."""
    first = max(str(data._day.min()), (pd.Timestamp(day)-pd.Timedelta(days=365)).strftime("%Y-%m-%d"))
    days = max(0, (pd.Timestamp(day)-pd.Timestamp(first)).days)
    eligible = (data._day.ge(first) & data._day.lt(day) & data._features_valid
                & data._label_valid & data.label_available_at_utc.le(cutoff))
    train = data.loc[eligible].copy()
    expected_zone = base._hours(first, day)
    coverage = {z: float(train.zone.eq(z).sum()/expected_zone) if expected_zone else 0. for z in zones}
    distinct = {z: int(train.loc[train.zone.eq(z), "_day"].nunique()) for z in zones}
    record = dict(fit_day=day, fit_cutoff_utc=cutoff, training_start_day=first,
        training_end_day=(pd.Timestamp(day)-pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        training_days=days, full_365_day_training=days == 365, training_rows=len(train),
        training_expected_hours=expected_zone*len(zones), training_coverage_by_zone=coverage,
        eligible_training_days_by_zone=distinct, max_label_available_at_utc=train.label_available_at_utc.max(),
        status="fallback", reason="", zones=list(zones))
    if days < p["minimum_training_days"] or min(distinct.values()) < p["minimum_training_days"]:
        record["reason"] = "insufficient_training_days"
    elif min(coverage.values()) < p["minimum_training_coverage"]:
        record["reason"] = "insufficient_training_coverage"
    if record["reason"]:
        return None, record
    split = (pd.Timestamp(day)-pd.Timedelta(days=p["calibration_days"])).strftime("%Y-%m-%d")
    core, cal = train.loc[train._day.lt(split)].copy(), train.loc[train._day.ge(split)].copy()
    if core.empty or cal.empty:
        record["reason"] = "empty_chronological_split"
        return None, record
    thresholds = {z: max(p["minimum_threshold_eur_mwh"], float(np.quantile(
        core.loc[core.zone.eq(z), "_error"], p["threshold_quantile"]))) for z in zones}
    y = core._error.to_numpy(float) >= core.zone.map(thresholds).to_numpy(float)
    cy = cal._error.to_numpy(float) >= cal.zone.map(thresholds).to_numpy(float)
    record.update(thresholds_eur_mwh=thresholds, model_training_end_day=str(core._day.max()),
        calibration_start_day=split, calibration_rows=len(cal), training_tail_rows=int(y.sum()),
        calibration_tail_rows=int(cy.sum()), cdf_core_rows=len(core),
        cdf_max_label_available_at_utc=core.label_available_at_utc.max(),
        cdf_uses_calibration_labels=False, reference_fit_scope="core_features_only")
    if min(int(y.sum()), int((~y).sum())) < p["minimum_tail_training_rows"]:
        record["reason"] = "insufficient_training_classes"
        return None, record
    if min(int(cy.sum()), int((~cy).sum())) < p["minimum_calibration_class_rows"] or cy.sum() < p["minimum_tail_calibration_rows"]:
        record["reason"] = "insufficient_chronological_calibration_events"
        return None, record
    reference = fit_reference(core, zones, cutoff=cutoff)
    core_x, cal_x = apply_reference(core, reference), apply_reference(cal, reference)
    features = p["feature_columns"]
    _check_features(features)
    X, V = base._matrix(core_x, features, zones), base._matrix(cal_x, features, zones)
    parameters = dict(n_estimators=p["max_iter"], learning_rate=p["learning_rate"],
        max_depth=0, max_leaves=p["max_leaf_nodes"], grow_policy="lossguide",
        min_child_weight=p["min_samples_leaf"]/4., reg_lambda=p["l2_regularization"],
        scale_pos_weight=1., objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        n_jobs=min(p["threads"], 2), random_state=p["random_state"], subsample=1., colsample_bytree=1., verbosity=0)
    with threadpool_limits(limits=min(p["threads"], 2)):
        classifier = _xgb_classifier()(**parameters).fit(X, y)
        margin = classifier.predict(V, output_margin=True)
        calibrator = LogisticRegression(C=1., solver="lbfgs", random_state=p["random_state"]).fit(
            np.asarray(margin).reshape(-1, 1), cy)
        cdfs = fit_distributions(X, core._error.to_numpy(float)/core.zone.map(thresholds).to_numpy(float),
                                core.zone.to_numpy(), kind="empirical", threads=min(p["threads"], 2))
    state = dict(classifier=classifier, calibrator=calibrator, distributions=cdfs, reference=reference,
        features=list(features), zones=tuple(zones), thresholds=thresholds, fit_day=day, fit_cutoff=cutoff,
        probability_gate=EVENT_GATE, amplitude_kind="empirical", parameters=dict(p))
    record.update(status="trained", reason="", xgboost_parameters=parameters,
        probability_calibration="unweighted_chronological_shared_platt_on_raw_margin",
        probability_gate=EVENT_GATE, country_specific_gate=False)
    if on_last_fit is not None:
        on_last_fit(state, dict(p))
    return state, record


def predict_model(state, current, p):
    """Inputs need no observed price, NYX level or benchmark for expert inference."""
    _check_features(state["features"])
    origin = base._utc(current.forecast_origin_utc, "forecast_origin_utc")
    timestamp = base._utc(current.timestamp_utc, "timestamp_utc")
    day = timestamp.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    expected = (day-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
    fitted = pd.Timestamp(state["fit_cutoff"])
    if (pd.isna(fitted) or fitted.tzinfo is None
            or not origin.eq(expected).all() or origin.lt(fitted).any()
            or not current.zone.isin(state["zones"]).all()):
        raise base.ScarcityPolicyError("Known countries and D-1 08h origin after fitted state required.")
    transformed = apply_reference(current, state["reference"])
    X = base._matrix(transformed, state["features"], state["zones"])
    thresholds = current.zone.map(state["thresholds"]).to_numpy(float)
    with threadpool_limits(limits=min(p["threads"], 2)):
        margin = state["classifier"].predict(X, output_margin=True)
        probability = state["calibrator"].predict_proba(np.asarray(margin).reshape(-1, 1))[:, 1]
        quantiles, diagnostics = predict_quantiles(state["distributions"], X, current.zone.to_numpy(), probability, thresholds)
    strong = probability > EVENT_GATE
    if (not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any()
            or (strong & (quantiles[:, 1] < thresholds)).any()
            or (~strong & (quantiles[:, 1] >= thresholds)).any()):
        raise base.ScarcityPolicyError("Invalid conditional mixture quantiles.")
    raw = np.where(strong, quantiles[:, 1], 0.)
    detail = current[KEYS].copy()
    for i, q in enumerate((10, 50, 90)):
        detail[f"mixture_error_q{q}"] = quantiles[:, i]
    detail["strong_risk_gate"] = strong
    detail["physical_gate_passed"] = True  # Availability checked before callback; no hand-tuned pressure gate.
    detail["proposal_reason"] = np.where(strong, "probability_above_half_coherent_median", "probability_not_above_half")
    for name in state["features"]:
        if "_core_rank" in name or "reference" in name:
            detail[name] = transformed[name].to_numpy()
    for name, values in diagnostics.items():
        detail["cdf_"+name] = values
    return probability, raw, thresholds, detail


def finalize_intervals(frame, *, direct=False, interval_settings=None, correction_clip_eur_mwh=400.):
    """Preserve baseline coverage pathwise, then widen on preceding OOS errors."""
    out = frame.copy(deep=True)
    if not np.isfinite(correction_clip_eur_mwh) or correction_clip_eur_mwh <= 0:
        raise base.ScarcityPolicyError("A finite positive correction clip is required.")
    proposed = out.expert_ready & out.raw_correction.gt(0)
    weights = np.where(proposed, 1. if direct else out.selected_weight.to_numpy(float), 0.)
    out["selected_weight"] = weights
    out["applied_correction"] = weights*out.bounded_correction
    out["candidate_forecast"] = out.forecast+out.applied_correction
    out["intervention_active"] = out.applied_correction.gt(0)
    out["candidate_q10"], out["candidate_q90"] = out.q10.copy(), out.q90.copy()
    active = out.intervention_active.to_numpy()
    for q in (10, 90):
        nyx = out[f"q{q}"].to_numpy(float)
        expert = out.forecast.to_numpy(float)+np.minimum(out[f"mixture_error_q{q}"].to_numpy(float), correction_clip_eur_mwh)
        interpolated = (1-weights)*nyx+weights*expert
        envelope = np.minimum(nyx, interpolated) if q == 10 else np.maximum(nyx, interpolated)
        out.loc[active, f"candidate_q{q}"] = envelope[active]
    out["precalibration_q10"], out["precalibration_q90"] = out.candidate_q10, out.candidate_q90
    if direct:
        out.loc[proposed, "gate_reason"] = "direct_strong_risk_diagnostic"
    inactive_ready = out.expert_ready & ~proposed
    out.loc[inactive_ready, "gate_reason"] = out.loc[inactive_ready, "proposal_reason"]
    out, audit = calibrate_intervals(out, settings=interval_settings)
    validate_output(out)
    return out, audit


def validate_output(out):
    values = out[["candidate_q10", "candidate_forecast", "candidate_q90"]].to_numpy(float)
    if (not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any()
            or (out.candidate_q10 > out.q10+1e-9).any() or (out.candidate_q90 < out.q90-1e-9).any()
            or not out.selected_weight.isin([0., .25, .5, 1.]).all() or out.bounded_correction.lt(0).any()
            or not np.allclose(out.applied_correction, out.selected_weight*out.bounded_correction, rtol=0, atol=1e-9)
            or not np.allclose(out.candidate_forecast, out.forecast+out.applied_correction, rtol=0, atol=1e-9)):
        raise base.ScarcityPolicyError("Ordered baseline-containing intervals and saved correction identity required.")


def recalibrate_previous(frame, *, interval_settings=None):
    """Interval-only ablation, preserving ALL previously issued point decisions."""
    if "interval_calibration_status" in frame or "precalibration_q10" in frame or "precalibration_q90" in frame:
        raise base.ScarcityPolicyError("Use the uncalibrated sealed previous P50 snapshot; repeated wrapper calibration is forbidden.")
    out = frame.copy(deep=True)
    out["intervention_active"] = out.applied_correction.gt(0)
    out["candidate_q10"] = np.minimum(out.q10, out.candidate_q10)
    out["candidate_q90"] = np.maximum(out.q90, out.candidate_q90)
    out["precalibration_q10"], out["precalibration_q90"] = out.candidate_q10, out.candidate_q90
    out, audit = calibrate_intervals(out, settings=interval_settings)
    if not out.candidate_forecast.equals(frame.candidate_forecast):
        raise base.ScarcityPolicyError("Interval-only ablation changed a point prediction.")
    validate_output(out)
    return out, audit


def run_stress_policy(panel, settings, *, interval_settings=None, on_last_fit=None):
    augmented, names, required, feature_audit = make_stress_features(panel)
    p = base._parameters({**settings, "probability_gate": EVENT_GATE, "feature_columns": names,
                          "required_feature_columns": required})
    if p["threads"] > 2:
        raise base.ScarcityPolicyError("At most two CPU threads for the isolated challenger.")
    details = []
    def predict(state, current, parameters):
        probability, raw, threshold, detail = predict_model(state, current, parameters)
        details.append(detail)
        return probability, raw, threshold
    result = base.run_policy(augmented, p,
        fit_callback=lambda data, day, cutoff, parameters, zones: fit_model(
            data, day, cutoff, parameters, zones, on_last_fit=on_last_fit), predict_callback=predict)
    frame = result.predictions.copy(deep=True)
    frame["proposal_reason"] = "expert_warmup_or_unavailable"
    frame["physical_gate_passed"] = False
    frame["strong_risk_gate"] = False
    for q in (10, 50, 90):
        frame[f"mixture_error_q{q}"] = np.nan
    if details:
        detail = pd.concat(details, ignore_index=True).set_index(KEYS)
        if detail.index.has_duplicates:
            raise base.ScarcityPolicyError("Duplicate prediction identities.")
        keys = pd.MultiIndex.from_frame(frame[KEYS])
        for column in detail:
            frame[column] = detail[column].reindex(keys).to_numpy()
    frame["mixture_raw_p50_eur_mwh"] = frame.forecast+frame.mixture_error_q50
    audit = {**result.audit, "engine": "nyx_stress_guard_v1", "features": feature_audit,
        "hypothesis": "new_physical_profile_representation_same_strong_risk_gate_all_zones",
        "amplitude": "two_regime_country_empirical_CDF_with_normalized_regional_fallback",
        "point_functional": "coherent_mixture_median_p_above_half_then_prequential_governor",
        "probability_gate": EVENT_GATE, "hand_tuned_country_physical_gate": False,
        "price_features_used": False, "classifier_refitted": True, "primary_policy": "governed",
        "interval_method": "baseline_union_then_chronological_same_intervention_state_expansion",
        "coverage_non_decrease_vs_baseline_pathwise": True, "coverage_target_guaranteed": False,
        "prospective_validation_completed": False, "exploratory_year_already_examined": True,
        "limitations": ["Exploratory previously examined year, not independent validation.",
            "Selected supply is incomplete mixed Pmax/generation, not a complete reserve/import margin.",
            "No interpretable JAO domain at 08h; weather and multivintage gaps remain explicit.",
            "Progressive training capped365, not a full prior365 before every evaluation day.",
            "Interval union may widen substantially; nominal and conditional coverage are not guaranteed.",
            "Only upward corrections; no guarantee of annual non-regression."]}
    outputs = {}
    for decision in ("direct", "governed"):
        LOGGER.info("[StressGuard] Chronological interval calibration: %s (%d rows)", decision, len(frame))
        predictions, interval_audit = finalize_intervals(frame, direct=decision == "direct", interval_settings=interval_settings,
            correction_clip_eur_mwh=p["correction_clip_eur_mwh"])
        pd.testing.assert_frame_equal(predictions[panel.columns], panel, check_exact=True)
        outputs[decision] = replace(result, predictions=predictions, audit={**audit, "decision_policy": decision,
            "strict_governor_enforced_in_this_point_forecast": decision == "governed", "interval_calibration": interval_audit,
            "changed_forecast_rows": int(predictions.intervention_active.sum())})
        LOGGER.info("[StressGuard] %s complete: %d intervention hours", decision, int(predictions.intervention_active.sum()))
    return outputs
