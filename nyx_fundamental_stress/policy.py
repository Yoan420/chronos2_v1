"""Isolated physical-risk detector and signed-residual median expert.

The event classifier sees only fundamental (or calendar-control) features.
NYX is used in the historical training TARGET, never as a model input. The
second estimator learns the conditional median of ALL signed errors, including
negative errors, not a positive-tail mixture. A positive proposed correction
requires both above-core-prevalence risk and the predeclared physical gate.
The existing prequential governor, clipping and interval policy are unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from nyx_scarcity import policy as base
from nyx_scarcity.variant_policy import _xgb_classifier
from .features import make_fundamental_features


VARIANTS = ("fundamental", "calendar")
LOCAL_PRESSURE = "feature_fundamental_local_pressure"
PEER_PRESSURE = "feature_fundamental_peer_pressure"
RESIDUAL_RAMP = "feature_fundamental_local_residual_ramp_3h_gw_per_hour"
PHYSICAL_QUANTILE = .9
PHYSICAL_RULE = {"DE": "local_pressure_and_positive_ramp", "FR": "local_pressure_and_positive_ramp",
                 "BE": "peer_pressure", "NL": "peer_pressure"}


def _check_features(names):
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(not isinstance(c, str) or not c.startswith("feature_fundamental_")
                   or any(token in c.lower() for token in
                          ("nyx", "baseline", "quantile", "forecast", "actual", "storm", "candidate", "error", "price", "benchmark", "label", "available_at", "q10", "q90"))
                   for c in names)):
        raise base.ScarcityPolicyError("Only explicit fundamental/calendar features without electricity-price or forecast inputs are allowed.")


def _fit(data, day, cutoff, parameters, zones, *, variant, on_last_fit=None):
    """Train only on eligible, already published days before this origin."""
    p = parameters
    _check_features(p["feature_columns"])
    first = max(str(data._day.min()), (pd.Timestamp(day)-pd.Timedelta(days=p["training_window_days"])).strftime("%Y-%m-%d"))
    days = max(0, (pd.Timestamp(day)-pd.Timestamp(first)).days)
    past = data.loc[data._day.ge(first) & data._day.lt(day)].copy()
    valid = past._features_valid & past._label_valid & past.label_available_at_utc.le(cutoff)
    train = past.loc[valid].copy()
    expected_per_zone = base._hours(first, day)
    coverage = {z: float(train.zone.eq(z).sum()/expected_per_zone) if expected_per_zone else 0. for z in zones}
    eligible_days = {z: int(train.loc[train.zone.eq(z), "_day"].nunique()) for z in zones}
    record = {"variant": variant, "fit_day": day, "fit_cutoff_utc": cutoff,
        "training_start_day": first, "training_end_day": (pd.Timestamp(day)-pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "training_days": days, "full_365_day_training": days == 365, "training_rows": len(train),
        "training_expected_hours": expected_per_zone*len(zones),
        "training_coverage": len(train)/(expected_per_zone*len(zones)) if expected_per_zone and zones else 0.,
        "training_coverage_by_zone": coverage, "eligible_training_days_by_zone": eligible_days,
        "minimum_eligible_training_days": min(eligible_days.values()),
        "max_label_available_at_utc": train.label_available_at_utc.max(), "zones": list(zones),
        "status": "fallback", "reason": ""}
    if days < p["minimum_training_days"]:
        record["reason"] = "insufficient_training_days"
    elif min(eligible_days.values()) < p["minimum_training_days"]:
        record["reason"] = "insufficient_eligible_training_days"
    elif min(coverage.values()) < p["minimum_training_coverage"]:
        record["reason"] = "insufficient_training_coverage"
    if record["reason"]:
        return None, record
    split = (pd.Timestamp(day)-pd.Timedelta(days=p["calibration_days"])).strftime("%Y-%m-%d")
    core, cal = train.loc[train._day.lt(split)].copy(), train.loc[train._day.ge(split)].copy()
    if core.empty or cal.empty:
        record["reason"] = "empty_chronological_split"
        return None, record
    thresholds, priors, physical = {}, {}, {}
    for zone in zones:
        local = core.loc[core.zone.eq(zone)]
        errors = local._error.to_numpy(float)
        if not len(errors):
            record["reason"] = "missing_zone_training"
            return None, record
        thresholds[zone] = max(p["minimum_threshold_eur_mwh"], float(np.quantile(errors, p["threshold_quantile"])))
        priors[zone] = float(np.mean(errors >= thresholds[zone]))
        if variant == "fundamental":
            feature = LOCAL_PRESSURE if PHYSICAL_RULE[zone].startswith("local_") else PEER_PRESSURE
            values = local[feature].dropna().to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                record["reason"] = "physical_core_threshold_unavailable"
                return None, record
            physical[zone] = float(np.quantile(values, PHYSICAL_QUANTILE))
    y = core._error.to_numpy(float) >= core.zone.map(thresholds).to_numpy(float)
    cy = cal._error.to_numpy(float) >= cal.zone.map(thresholds).to_numpy(float)
    record.update(thresholds_eur_mwh=thresholds, calibration_start_day=split,
        model_training_end_day=str(core._day.max()), calibration_rows=len(cal), training_tail_rows=int(y.sum()),
        training_negative_rows=int((~y).sum()), calibration_tail_rows=int(cy.sum()),
        risk_probability_gate_by_zone=priors, physical_pressure_q90_by_zone=physical,
        zero_core_event_prevalence_zones=[z for z in zones if priors[z] == 0.],
        signed_model_training_rows=len(core), signed_model_negative_error_rows=int(core._error.lt(0).sum()),
        signed_model_zero_error_rows=int(core._error.eq(0).sum()), signed_model_positive_error_rows=int(core._error.gt(0).sum()))
    if min(int(y.sum()), int((~y).sum())) < p["minimum_tail_training_rows"]:
        record["reason"] = "insufficient_training_classes"
        return None, record
    if min(int(cy.sum()), int((~cy).sum())) < p["minimum_calibration_class_rows"] or cy.sum() < p["minimum_tail_calibration_rows"]:
        record["reason"] = "insufficient_chronological_calibration_events"
        return None, record
    X, V = base._matrix(core, p["feature_columns"], zones), base._matrix(cal, p["feature_columns"], zones)
    xgb_parameters = {"n_estimators": p["max_iter"], "learning_rate": p["learning_rate"],
        "max_depth": 0, "max_leaves": p["max_leaf_nodes"], "grow_policy": "lossguide",
        "min_child_weight": p["min_samples_leaf"]/4., "reg_lambda": p["l2_regularization"],
        "scale_pos_weight": 1., "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
        "n_jobs": min(p["threads"], 2), "random_state": p["random_state"], "subsample": 1., "colsample_bytree": 1., "verbosity": 0}
    common = {key: p[key] for key in ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "random_state")}
    with threadpool_limits(limits=min(p["threads"], 2)):
        classifier = _xgb_classifier()(**xgb_parameters).fit(X, y)
        margin = classifier.predict(V, output_margin=True)
        calibrator = LogisticRegression(C=1., solver="lbfgs", random_state=p["random_state"]).fit(np.asarray(margin).reshape(-1, 1), cy)
        signed_model = HistGradientBoostingRegressor(loss="quantile", quantile=.5, early_stopping=False, **common).fit(X, core._error.to_numpy(float))
    state = {"classifier": classifier, "calibrator": calibrator, "signed_model": signed_model,
        "features": list(p["feature_columns"]), "zones": zones,
        "model_feature_names": [*p["feature_columns"], *("zone__"+z for z in zones)],
        "thresholds": thresholds, "risk_probability_gate_by_zone": priors, "physical_pressure_q90_by_zone": physical,
        "variant": variant, "fit_day": day, "fit_cutoff": cutoff, "calibration_input": "raw_margin",
        "target_kind": "positive_large_actual_minus_frozen_nyx_error", "scale_pos_weight": 1.}
    record.update(status="trained", reason="", xgboost_parameters=xgb_parameters,
        probability_calibration="unweighted_chronological_shared_platt_on_raw_xgb_margin",
        calibration_sample_weight_used=False, signed_model_quantile=.5, signed_model_calibration_offset_used=False,
        signed_model_target="all_signed_actual_minus_frozen_nyx_errors", physical_threshold_fit_scope="core_only_excluding_calibration")
    if on_last_fit is not None:
        on_last_fit(state, dict(p))
    return state, record


def predict_state(state: dict, current: pd.DataFrame, parameters: dict):
    """Inference needs only declared exogenous features, country and identity."""
    _check_features(state["features"])
    variant = state["variant"]
    if variant not in VARIANTS or not current.zone.isin(state["zones"]).all():
        raise base.ScarcityPolicyError("Known fixed variant and training-country identities required.")
    timestamps = base._utc(current.timestamp_utc, "timestamp_utc")
    origins = base._utc(current.forecast_origin_utc, "forecast_origin_utc")
    civil = timestamps.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    expected = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
    fitted = pd.Timestamp(state["fit_cutoff"])
    if (fitted.tzinfo is None or pd.isna(fitted) or not origins.eq(expected).all()
            or origins.lt(fitted.tz_convert("UTC")).any()):
        raise base.ScarcityPolicyError("Inference must preserve D-1 08:00 and cannot use a future fitted state.")
    X = base._matrix(current, state["features"], state["zones"])
    threshold = current.zone.map(state["thresholds"]).to_numpy(float)
    prior = current.zone.map(state["risk_probability_gate_by_zone"]).to_numpy(float)
    if not np.isfinite(threshold).all() or (threshold <= 0).any() or not np.isfinite(prior).all() or ((prior < 0)|(prior > 1)).any():
        raise base.ScarcityPolicyError("Finite positive error thresholds and core event prevalences in [0,1] required.")
    with threadpool_limits(limits=min(parameters["threads"], 2)):
        margin = state["classifier"].predict(X, output_margin=True)
        probability = state["calibrator"].predict_proba(np.asarray(margin).reshape(-1, 1))[:, 1]
        median = np.asarray(state["signed_model"].predict(X), dtype=float)
    probability = np.asarray(probability, dtype=float)
    if (probability.shape != (len(current),) or median.shape != (len(current),)
            or not np.isfinite(median).all() or not np.isfinite(probability).all()
            or ((probability < 0)|(probability > 1)).any()):
        raise base.ScarcityPolicyError("Finite aligned signed medians and event probabilities required.")
    physical_gate = np.ones(len(current), dtype=bool)
    physical_available = np.ones(len(current), dtype=bool)
    physical_threshold = np.full(len(current), np.nan)
    if variant == "fundamental":
        physical_threshold = current.zone.map(state["physical_pressure_q90_by_zone"]).to_numpy(float)
        if not np.isfinite(physical_threshold).all():
            raise base.ScarcityPolicyError("Finite fitted core physical thresholds required.")
        physical_gate[:] = False
        for zone in state["zones"]:
            mask = current.zone.eq(zone).to_numpy()
            if not mask.any():
                continue
            required = [LOCAL_PRESSURE, RESIDUAL_RAMP] if PHYSICAL_RULE[zone].startswith("local_") else [PEER_PRESSURE]
            values = current.reindex(columns=required).to_numpy(float)
            finite = np.isfinite(values).all(axis=1)
            gate = values[:, 0] > physical_threshold
            if len(required) == 2:
                gate &= values[:, 1] > 0.
            physical_available[mask], physical_gate[mask] = finite[mask], (finite & gate)[mask]
    risk = probability > prior
    positive = median > 0.
    raw = np.where(risk & physical_gate & positive, median, 0.)
    reason = np.select([~physical_available, ~physical_gate, ~risk, ~positive],
        ["physical_gate_inputs_unavailable", "physical_stress_gate_closed", "risk_not_above_core_prevalence", "signed_median_not_positive"],
        default="fundamental_proposal_ready" if variant == "fundamental" else "calendar_proposal_ready")
    diagnostic = current[["zone", "timestamp_utc", "forecast_origin_utc"]].copy()
    diagnostic["risk_probability_gate"] = prior
    diagnostic["risk_above_core_prevalence"] = risk
    diagnostic["physical_gate_inputs_available"] = physical_available
    diagnostic["physical_gate_passed"] = physical_gate
    diagnostic["physical_pressure_core_q90"] = physical_threshold
    diagnostic["predicted_signed_residual_median"] = median
    diagnostic["positive_signed_residual_median"] = np.maximum(median, 0.)
    diagnostic["proposal_reason"] = reason
    return probability, raw, threshold, diagnostic


def run_fundamental_policy(panel: pd.DataFrame, settings: dict, variant: str,
                           on_last_fit: Callable | None = None) -> base.PolicyResult:
    """Return an isolated strict-governed challenger with original rows intact."""
    if variant not in VARIANTS or (on_last_fit is not None and not callable(on_last_fit)):
        raise base.ScarcityPolicyError("Use fundamental/calendar and an optional callable fit observer.")
    augmented, features, required, feature_audit = make_fundamental_features(panel, variant=variant)
    _check_features(features)
    if variant == "calendar":
        # Calendar values are known independently of physical-store availability.
        # Only the calculation copy changes; original audit columns are restored
        # in the returned panel. Forecast/label eligibility remains binding.
        augmented["feature_eligible"] = True
        augmented["feature_available_at_utc"] = augmented.forecast_origin_utc
    options = {"max_iter": 60, "min_samples_leaf": 40, **settings,
               "feature_columns": features, "required_feature_columns": required}
    p = base._parameters(options)
    if p["threads"] > 2:
        raise base.ScarcityPolicyError("At most two threads per fundamental worker.")
    _xgb_classifier()
    diagnostics = []
    def predict(state, current, parameters):
        probability, raw, threshold, detail = predict_state(state, current, parameters)
        diagnostics.append(detail)
        return probability, raw, threshold
    result = base.run_policy(augmented, p,
        fit_callback=lambda data, day, cutoff, pars, zones: _fit(data, day, cutoff, pars, zones,
            variant=variant, on_last_fit=on_last_fit), predict_callback=predict)
    predictions = result.predictions.copy()
    keys = ["zone", "timestamp_utc", "forecast_origin_utc"]
    if diagnostics:
        details = pd.concat(diagnostics, ignore_index=True).set_index(keys)
        if details.index.has_duplicates:
            raise base.ScarcityPolicyError("Duplicate fundamental inference identities.")
        identity = pd.MultiIndex.from_frame(predictions[keys])
        for column in details:
            predictions[column] = details[column].reindex(identity).to_numpy()
    else:
        for column in ("risk_probability_gate", "predicted_signed_residual_median", "positive_signed_residual_median", "physical_pressure_core_q90"):
            predictions[column] = np.nan
        for column in ("risk_above_core_prevalence", "physical_gate_inputs_available", "physical_gate_passed"):
            predictions[column] = False
        predictions["proposal_reason"] = "expert_warmup_or_unavailable"
    predictions["probability_gate"] = predictions.risk_probability_gate
    inactive = predictions.expert_ready & predictions.raw_correction.le(0)
    predictions.loc[inactive, "gate_reason"] = predictions.loc[inactive, "proposal_reason"]
    predictions["fundamental_variant"] = variant
    # Concatenate original columns verbatim, including their index and dtypes.
    predictions = pd.concat([panel.copy(deep=True), predictions.drop(columns=list(panel.columns))], axis=1)
    audit = {**result.audit, "engine": "nyx_fundamental_stress_v1", "variant": variant,
        "features": feature_audit, "electricity_prices_or_nyx_used_as_model_inputs": False,
        "nyx_used_in_training_target_only": True, "signed_model_fit_scope": "all_eligible_core_signed_errors_including_negative_and_zero",
        "signed_model_calibration_offset_used": False, "point_functional": "positive_part_of_conditional_median_signed_error_then_risk_and_physical_gates_then_existing_governor",
        "probability_calibration": "unweighted_chronological_shared_platt_on_raw_xgb_margin",
        "risk_gate": "calibrated_event_probability_strictly_greater_than_country_core_event_prevalence",
        "zero_core_prevalence_policy": "retain empirical zero explicitly; no hidden probability floor; physical and positive-median gates still apply",
        "legacy_constant_probability_gate_unused": p["probability_gate"], "risk_gate_uses_future_event_rate": False,
        "physical_gate": PHYSICAL_RULE if variant == "fundamental" else "always_true_calendar_control",
        "physical_pressure_quantile": PHYSICAL_QUANTILE, "physical_threshold_scope": "model_core_excluding_calibration_and_current_future",
        "hypothesis_origin": "predeclared after exploratory inspection of this evaluation year; not an independent confirmatory holdout",
        "governance_policy_changed": False, "strict_governor_enforced_in_this_point_forecast": True,
        "annual_non_regression_guaranteed": False, "severity_model_changed": True,
        "positive_tail_only_fit": False, "probability_times_tail_correction": False}
    audit["calendar_ignores_physical_availability_only_in_working_copy"] = variant == "calendar"
    return replace(result, predictions=predictions, audit=audit)


__all__ = ["run_fundamental_policy", "predict_state"]
