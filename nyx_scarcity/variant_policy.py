"""Four predeclared XGBoost scarcity ablations, with the original governor.

Only classification and (optionally) the positive-error target threshold change.
Weighted fits use sqrt(n_negative / n_positive), capped at 30, on the earlier
core block only. Unweighted Platt calibration uses a later chronological block
and raw XGBoost margins: SHAP-to-calibrated-log-odds remains affine and exact.

The causal dynamic weighted threshold (DWT) is a residual-target adaptation:
u_D = max(floor, w_D * Q_global + (1-w_D) * Q_local), with windows365/30days.
w_D is the clipped min/max normalisation of current30day error volatility
against volatility statistics computed at PREVIOUS origins. High volatility
therefore favours the GLOBAL quantile. It is not a look-ahead normalisation or
an assertion that this adaptation exactly reproduces another paper's protocol.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from . import policy as base


THRESHOLD_FEATURE = "feature_tail_threshold_eur_mwh"
VARIANTS = {
    "xgb_unweighted_fixed": (False, "fixed"), "xgb_weighted_fixed": (True, "fixed"),
    "xgb_unweighted_dwt": (False, "dwt"), "xgb_weighted_dwt": (True, "dwt"),
}


def validate_variant(variant: dict) -> dict:
    if not isinstance(variant, dict) or set(variant) != {"id", "weighted", "threshold_kind"}:
        raise base.ScarcityPolicyError("Variant requires exactly id, weighted and threshold_kind.")
    if not isinstance(variant["id"], str) or not isinstance(variant["threshold_kind"], str):
        raise base.ScarcityPolicyError("Variant id and threshold_kind must be strings.")
    expected = VARIANTS.get(variant["id"])
    if type(variant["weighted"]) is not bool or expected != (variant["weighted"], variant["threshold_kind"]):
        raise base.ScarcityPolicyError("Variant flags must match one of the four predeclared ablation IDs.")
    return dict(variant)


def _xgb_classifier():
    # Require the pinned project-private runtime. Never silently substitute
    # HGB, a shared XGBoost installation or another package version.
    try:
        from .variant_runtime import ensure_runtime
        ensure_runtime()
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise base.ScarcityPolicyError("XGBoost runtime is unavailable; prepare the isolated variant runtime first.") from exc
    return XGBClassifier


def causal_thresholds(panel: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, dict]:
    """One threshold per zone/origin, computed from genuinely prior known errors."""
    p = base._parameters(settings)
    data = base._prepare(panel, p)
    out = pd.DataFrame(index=np.arange(len(panel)), columns=[THRESHOLD_FEATURE, "dwt_global_quantile", "dwt_local_quantile", "dwt_sigma30", "dwt_weight"])
    out = out.astype(float)
    out["dwt_available"] = False
    out["dwt_label_max_available_at_utc"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["dwt_history_days"] = 0
    out["dwt_normalisation_reference_days"] = 0
    diagnostics = []
    for zone, rows in data.groupby("zone", sort=True):
        sigmas: list[tuple[str, float]] = []
        for day, current in rows.groupby("_day", sort=True):
            cutoff = current.forecast_origin_utc.iloc[0]
            start = (pd.Timestamp(day) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
            local_start = (pd.Timestamp(day) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            known = rows._day.ge(start) & rows._day.lt(day) & rows._label_valid & rows.label_available_at_utc.le(cutoff)
            known &= base._flag(rows, "forecast_eligible")
            history = rows.loc[known]
            local = history.loc[history._day.ge(local_start)]
            targets = current._row.to_numpy(int)
            out.loc[targets, "dwt_history_days"] = history._day.nunique()
            if history._day.nunique() < 30 or local._day.nunique() < 30:
                continue
            # Missing hours count towards coverage; no interpolation or label
            # imputation is used to manufacture volatility observations.
            expected = base._hours(local_start, day)
            if len(local) / expected < p["minimum_training_coverage"]:
                continue
            global_q = float(np.quantile(history._error.to_numpy(float), p["threshold_quantile"]))
            local_q = float(np.quantile(local._error.to_numpy(float), p["threshold_quantile"]))
            sigma = float(local._error.std(ddof=0))
            prior = [value for date, value in sigmas if start <= date < day]
            if prior and max(prior) - min(prior) > 1e-12:
                weight = float(np.clip((sigma - min(prior)) / (max(prior) - min(prior)), 0., 1.))
                normalization = "strictly_prior_sigma_range"
            else:
                weight, normalization = .5, "no_prior_sigma_range_fixed_half"
            threshold = max(p["minimum_threshold_eur_mwh"], weight * global_q + (1 - weight) * local_q)
            out.loc[targets, [THRESHOLD_FEATURE, "dwt_global_quantile", "dwt_local_quantile", "dwt_sigma30", "dwt_weight"]] = [threshold, global_q, local_q, sigma, weight]
            out.loc[targets, "dwt_available"] = True
            out.loc[targets, "dwt_label_max_available_at_utc"] = history.label_available_at_utc.max()
            out.loc[targets, "dwt_normalisation_reference_days"] = len(prior)
            diagnostics.append({"zone": zone, "day": day, "cutoff": cutoff,
                                "max_used_label_available_at_utc": history.label_available_at_utc.max(),
                                "normalisation": normalization})
            # Append after computing the weight: today's sigma cannot affect
            # its own normalisation, let alone a threshold in the past.
            sigmas.append((day, sigma))
            sigmas = [(date, value) for date, value in sigmas if date >= start]
    out.index = panel.index.copy()
    audit = {"kind": "causal_dynamic_weighted_residual_threshold", "global_days": 365,
             "local_days": 30, "minimum_history_days": 30, "quantile": p["threshold_quantile"],
             "floor_eur_mwh": p["minimum_threshold_eur_mwh"],
             "formula": "max(floor, w * global_quantile + (1-w) * local_quantile)",
             "volatility_weight": "clip((sigma30 - min(prior_sigmas30))/(max(prior_sigmas30)-min(prior_sigmas30)),0,1)",
             "degenerate_reference_weight": .5, "normalization_range_includes_current_sigma": False,
             "labels_required": "delivery day < current day AND publication <= current origin",
             "available_rows": int(out.dwt_available.sum()), "unavailable_rows": int((~out.dwt_available).sum()),
             "daily_origins": len(diagnostics), "updated_between_weekly_refits": True,
             "causality_violations": int(sum(r["max_used_label_available_at_utc"] > r["cutoff"] for r in diagnostics))}
    return out, audit


def _fit(data, day, cutoff, p, zones, *, variant, on_fit=None):
    first = max(str(data._day.min()), (pd.Timestamp(day) - pd.Timedelta(days=365)).strftime("%Y-%m-%d"))
    days = max(0, (pd.Timestamp(day) - pd.Timestamp(first)).days)
    past = data.loc[data._day.ge(first) & data._day.lt(day)].copy()
    valid = past._features_valid & past._label_valid & past.label_available_at_utc.le(cutoff)
    train = past.loc[valid]
    expected_by_zone = {}
    threshold_start = {}
    for zone in zones:
        start = first
        if variant["threshold_kind"] == "dwt":
            available = past.loc[past.zone.eq(zone) & past[THRESHOLD_FEATURE].notna(), "_day"]
            start = str(available.min()) if len(available) else day
        threshold_start[zone] = start
        expected_by_zone[zone] = base._hours(start, day)
    expected = sum(expected_by_zone.values())
    coverage_by_zone = {z: float(train.zone.eq(z).sum() / n) if n else 0. for z, n in expected_by_zone.items()}
    record = {"variant_id": variant["id"], "fit_day": day, "fit_cutoff_utc": cutoff,
              "training_start_day": first, "training_end_day": (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
              "training_days": days, "full_365_day_training": days == 365,
              "training_rows": len(train), "training_expected_hours": expected,
              "training_coverage": len(train) / expected if expected else 0.,
              "training_coverage_by_zone": coverage_by_zone, "threshold_support_start_by_zone": threshold_start,
              "max_label_available_at_utc": train.label_available_at_utc.max(), "zones": list(zones),
              "status": "fallback", "reason": ""}
    eligible_days = {z: int(train.loc[train.zone.eq(z), "_day"].nunique()) for z in zones}
    record["eligible_training_days_by_zone"] = eligible_days
    record["minimum_eligible_training_days"] = min(eligible_days.values())
    if days < p["minimum_training_days"]:
        record["reason"] = "insufficient_training_days"
    elif min(eligible_days.values()) < p["minimum_training_days"]:
        record["reason"] = "insufficient_eligible_training_days"
    elif min(coverage_by_zone.values()) < p["minimum_training_coverage"]:
        record["reason"] = "insufficient_training_coverage"
    if record["reason"]:
        return None, record
    split = (pd.Timestamp(day) - pd.Timedelta(days=p["calibration_days"])).strftime("%Y-%m-%d")
    core, cal = train.loc[train._day.lt(split)].copy(), train.loc[train._day.ge(split)].copy()
    if core.empty or cal.empty:
        record["reason"] = "empty_chronological_split"
        return None, record
    thresholds = {}
    if variant["threshold_kind"] == "fixed":
        for zone in zones:
            errors = core.loc[core.zone.eq(zone), "_error"].to_numpy(float)
            if not len(errors):
                record["reason"] = "missing_zone_training"
                return None, record
            thresholds[zone] = max(p["minimum_threshold_eur_mwh"], float(np.quantile(errors, p["threshold_quantile"])))
        core_u, cal_u = core.zone.map(thresholds).to_numpy(float), cal.zone.map(thresholds).to_numpy(float)
    else:
        core_u, cal_u = core[THRESHOLD_FEATURE].to_numpy(float), cal[THRESHOLD_FEATURE].to_numpy(float)
    y, cy = core._error.to_numpy(float) >= core_u, cal._error.to_numpy(float) >= cal_u
    positive, negative = int(y.sum()), int((~y).sum())
    scale = min(30., float(np.sqrt(negative / positive))) if variant["weighted"] and positive else 1.
    record.update(thresholds_eur_mwh=thresholds, threshold_kind=variant["threshold_kind"],
                  calibration_start_day=split, model_training_end_day=str(core._day.max()),
                  calibration_rows=len(cal), training_tail_rows=positive,
                  training_negative_rows=negative, calibration_tail_rows=int(cy.sum()), scale_pos_weight=scale)
    if min(positive, negative) < p["minimum_tail_training_rows"]:
        record["reason"] = "insufficient_training_classes"
        return None, record
    if min(int(cy.sum()), int((~cy).sum())) < p["minimum_calibration_class_rows"] or cy.sum() < p["minimum_tail_calibration_rows"]:
        record["reason"] = "insufficient_chronological_calibration_events"
        return None, record
    X, V = base._matrix(core, p["feature_columns"], zones), base._matrix(cal, p["feature_columns"], zones)
    xgb_parameters = {"n_estimators": p["max_iter"], "learning_rate": p["learning_rate"],
                      "max_depth": 0, "max_leaves": p["max_leaf_nodes"], "grow_policy": "lossguide",
                      "min_child_weight": p["min_samples_leaf"] / 4., "reg_lambda": p["l2_regularization"],
                      "scale_pos_weight": scale, "objective": "binary:logistic", "eval_metric": "logloss",
                      "tree_method": "hist", "n_jobs": min(p["threads"], 2), "random_state": p["random_state"],
                      "subsample": 1., "colsample_bytree": 1., "verbosity": 0}
    common = {key: p[key] for key in ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "random_state")}
    with threadpool_limits(limits=min(p["threads"], 2)):
        classifier = _xgb_classifier()(**xgb_parameters).fit(X, y)
        margins = classifier.predict(V, output_margin=True)
        calibrator = LogisticRegression(C=1., solver="lbfgs", random_state=p["random_state"]).fit(np.asarray(margins).reshape(-1, 1), cy)
        severity = HistGradientBoostingRegressor(loss="quantile", quantile=.5, early_stopping=False, **common).fit(X[y], np.log(core._error.to_numpy(float)[y] / core_u[y]))
        severity_location = severity.predict(V[cy])
    tail_log_errors = np.log(cal._error.to_numpy(float)[cy] / cal_u[cy]) - severity_location
    state = {"classifier": classifier, "calibrator": calibrator, "severity": severity,
             "thresholds": thresholds, "threshold_kind": variant["threshold_kind"], "tail_log_errors": tail_log_errors,
             "zones": zones, "features": p["feature_columns"], "fit_day": day, "fit_cutoff": cutoff,
             "model_feature_names": [*p["feature_columns"], *("zone__" + z for z in zones)],
             "calibration_input": "raw_margin", "target_kind": "positive_large_actual_minus_frozen_nyx_error",
             "variant_id": variant["id"], "scale_pos_weight": scale}
    record.update(status="trained", reason="", xgboost_parameters=xgb_parameters,
                  probability_calibration="unweighted_chronological_platt_on_raw_xgboost_margin",
                  calibration_sample_weight_used=False, median_model="zero_atom_plus_positive_tail_mixture",
                  tail_distribution_rows=len(tail_log_errors))
    if on_fit is not None:
        current = data.loc[data._day.eq(day)].copy()
        current[["actual", "_error"]] = np.nan
        current["_label_valid"] = False
        on_fit(state, core.copy(), cal.copy(), current, dict(p))
    return state, record


def _predict(state, current, p, *, on_predict=None):
    X = base._matrix(current, state["features"], state["zones"])
    u = current[THRESHOLD_FEATURE].to_numpy(float) if state["threshold_kind"] == "dwt" else current.zone.map(state["thresholds"]).to_numpy(float)
    with threadpool_limits(limits=min(p["threads"], 2)):
        margins = state["classifier"].predict(X, output_margin=True)
        probability = state["calibrator"].predict_proba(np.asarray(margins).reshape(-1, 1))[:, 1]
        location = state["severity"].predict(X)
    level = np.where(probability > .5, 1 - .5 / np.maximum(probability, .5), 0.)
    log_error = np.quantile(state["tail_log_errors"], level)
    tail = u * np.exp(np.clip(location + log_error, -20, 20))
    raw = np.where(probability > max(.5, p["probability_gate"]), np.maximum(u, tail), 0.)
    if on_predict is not None:
        observed = current.copy()
        observed[["actual", "_error"]] = np.nan
        observed["_label_valid"] = False
        on_predict(state, observed, dict(p))
    return probability, raw, u


def run_variant_policy(panel: pd.DataFrame, settings: dict, variant: dict, on_fit=None, *, on_predict=None) -> base.PolicyResult:
    variant = validate_variant(variant)
    _xgb_classifier()  # Missing runtime fails before any training or callback.
    p = base._parameters(settings)
    if p["threads"] > 2:
        raise base.ScarcityPolicyError("XGBoost ablations are bounded to at most two threads.")
    for callback in (on_fit, on_predict):
        if callback is not None and not callable(callback):
            raise base.ScarcityPolicyError("Variant observation callbacks must be callable.")
    augmented, options, threshold_audit = panel.copy(deep=True), dict(settings), {"kind": "constant_per_fit_core_only"}
    if variant["threshold_kind"] == "dwt":
        if THRESHOLD_FEATURE in panel:
            raise base.ScarcityPolicyError("DWT threshold features must be produced by the causal builder, not supplied externally.")
        columns, threshold_audit = causal_thresholds(panel, settings)
        augmented = pd.concat([augmented, columns], axis=1)
        options["feature_columns"] = [*p["feature_columns"], THRESHOLD_FEATURE]
        options["required_feature_columns"] = [*p["required_feature_columns"], THRESHOLD_FEATURE]
    fitted = base.run_policy(augmented, options,
        fit_callback=lambda data, day, cutoff, pars, zones: _fit(data, day, cutoff, pars, zones, variant=variant, on_fit=on_fit),
        predict_callback=lambda state, frame, pars: _predict(state, frame, pars, on_predict=on_predict))
    audit = {**fitted.audit, "engine": "nyx_scarcity_xgboost_ablation_v1", "variant": variant,
             "classification_backend": "xgboost", "severity_backend": "unchanged_hgb_conditional_median",
             "calibration_input": "raw_margin", "probability_calibration": "unweighted_chronological_platt_on_raw_xgboost_margin",
             "class_weight_formula": "min(30, sqrt(n_negative_core/n_positive_core))" if variant["weighted"] else "1",
             "class_weight_scope": "model core only, never chronological calibration or governance",
             "threshold_fitted_on": "causal_each_row_origin_prior_labels" if variant["threshold_kind"] == "dwt" else "model_core_excluding_calibration",
             "threshold_audit": threshold_audit,
             "dwt_warmup_coverage_policy": "Structural leading threshold warmup excluded from coverage denominator; at least90 eligible training days required per zone, usually120 source days with DWT.",
             "xgb_child_weight_mapping": "min_child_weight=min_samples_leaf/4 is a Hessian proxy, not the HGB minimum row constraint",
             "controls_unchanged": ["probability_gate", "correction_clip_eur_mwh", "candidate_weights", "governance", "severity_model", "empirical_oos_intervals"],
             "daily_prediction_callback_labels_masked": True}
    predictions = fitted.predictions.copy()
    predictions["variant_id"] = variant["id"]
    return replace(fitted, predictions=predictions, audit=audit)


__all__ = ["VARIANTS", "THRESHOLD_FEATURE", "validate_variant", "causal_thresholds", "run_variant_policy"]
