"""Research-only, causal upper-tail expert after an immutable NYX forecast.

The classifier estimates Pr(error >= u | x), without class reweighting. Its
probabilities are Platt-calibrated on a strictly later, held-out block. A
conditional median severity model and held-out log severity residuals define
the positive tail. The proposed correction is the median of a mixture with a
point mass at zero: zero for p <= .5, otherwise tail_quantile(1 - .5 / p).
This is a deliberately conservative positive-error model, NOT a conditional
mean relabelled as a median, nor a full fitted electricity-price distribution.

Daily per-zone governance uses only previously issued candidate forecasts and
labels available at the new origin. Intervals use separate empirical signed
OOS errors; empirical coverage is not a finite-sample or future guarantee.
No Storm forecasts, economic PnL, orders, or operational state are involved.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


LOGGER = logging.getLogger(__name__)


class ScarcityPolicyError(ValueError):
    """The fixed causal experiment contract is invalid."""


@dataclass(frozen=True)
class PolicyResult:
    predictions: pd.DataFrame
    folds: pd.DataFrame
    governance: pd.DataFrame
    audit: dict[str, Any]


def _parameters(settings: Mapping[str, Any]) -> dict[str, Any]:
    p = dict(settings)
    integers = {
        "training_window_days": (365, 365, 365), "minimum_training_days": (90, 90, 365),
        "refit_every_days": (7, 1, 365), "calibration_days": (28, 7, 90),
        "max_iter": (80, 1, 10000), "max_leaf_nodes": (15, 2, 255),
        "min_samples_leaf": (30, 2, 10000), "random_state": (1729, 0, 2**31 - 1),
        "minimum_tail_training_rows": (30, 2, 10000),
        "minimum_calibration_class_rows": (5, 2, 10000),
        "minimum_tail_calibration_rows": (10, 2, 10000),
        "threads": (1, 1, 32),
        "governance_lookback_days": (90, 1, 365), "governance_minimum_days": (28, 2, 365),
        "governance_minimum_changed_days": (5, 1, 365),
        "governance_minimum_tail_rows": (12, 1, 10000), "interval_minimum_rows": (120, 20, 100000),
    }
    for key, (default, low, high) in integers.items():
        value = p.get(key, default)
        if isinstance(value, bool) or not np.isfinite(float(value)) or int(value) != float(value) or not low <= int(value) <= high:
            raise ScarcityPolicyError(f"{key}: integer in [{low}, {high}] required.")
        p[key] = int(value)
    numbers = {
        "minimum_training_coverage": (.9, .5, 1.), "threshold_quantile": (.95, .8, .999),
        "minimum_threshold_eur_mwh": (50., 1., 10000.), "probability_gate": (.6, .5, 1.),
        "correction_clip_eur_mwh": (400., 1., 10000.), "learning_rate": (.06, .00001, 1.),
        "l2_regularization": (10., 0., 100000.), "governance_minimum_mae_gain_eur_mwh": (0., 0., 10000.),
        "governance_minimum_tail_gain_eur_mwh": (.5, 0., 10000.),
        "governance_uncertainty_z": (1., 0., 10.),
    }
    unknown = set(p).difference({*integers, *numbers, "timezone", "candidate_weights", "feature_columns", "required_feature_columns"})
    if unknown:
        raise ScarcityPolicyError(f"Unknown policy settings: {sorted(unknown)}.")
    for key, (default, low, high) in numbers.items():
        value = float(p.get(key, default))
        if not np.isfinite(value) or not low <= value <= high:
            raise ScarcityPolicyError(f"{key}: finite value in [{low}, {high}] required.")
        p[key] = value
    if p["calibration_days"] >= p["minimum_training_days"]:
        raise ScarcityPolicyError("Calibration must leave an earlier model-training block.")
    if p["governance_minimum_days"] > p["governance_lookback_days"]:
        raise ScarcityPolicyError("Governance minimum exceeds its lookback.")
    p["timezone"] = p.get("timezone", "Europe/Paris")
    if p["timezone"] != "Europe/Paris":
        raise ScarcityPolicyError("The qualified origin is D-1 08:00 Europe/Paris civil.")
    weights = [float(v) for v in p.get("candidate_weights", [0., .25, .5, 1.])]
    if (not weights or weights[0] != 0 or len(set(weights)) != len(weights) or len(weights) > 5
            or any(not np.isfinite(v) or not 0 <= v <= 1 for v in weights)):
        raise ScarcityPolicyError("candidate_weights must start with identity and be unique fixed weights in [0,1].")
    p["candidate_weights"] = weights
    features = p.get("feature_columns")
    forbidden = {"actual", "forecast", "q10", "q90", "zone", "timestamp_utc", "label_available_at_utc", "forecast_origin_utc"}
    if (not isinstance(features, list) or not features or len(set(features)) != len(features)
            or any(not isinstance(v, str) or v in forbidden or any(t in v.lower() for t in
                   ("storm", "actual", "label", "candidate_", "eligible", "available_at")) for v in features)):
        raise ScarcityPolicyError("Explicit unique numeric feature_columns must exclude labels, Storm and metadata.")
    required = p.get("required_feature_columns", features)
    if not isinstance(required, list) or len(set(required)) != len(required) or not set(required).issubset(features):
        raise ScarcityPolicyError("required_feature_columns must be a unique feature subset.")
    p["feature_columns"], p["required_feature_columns"] = list(features), list(required)
    return p


def _utc(values: pd.Series, name: str, *, missing: bool = False) -> pd.Series:
    if any(pd.Timestamp(v).tzinfo is None for v in values.dropna()):
        raise ScarcityPolicyError(f"{name}: explicit timezone required.")
    parsed = pd.to_datetime(values, utc=True, format="mixed", errors="raise")
    if not missing and parsed.isna().any():
        raise ScarcityPolicyError(f"{name}: missing timestamp.")
    return parsed


def _flag(data: pd.DataFrame, name: str) -> pd.Series:
    if name not in data:
        return pd.Series(True, index=data.index)
    if not data[name].dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ScarcityPolicyError(f"{name}: explicit boolean values required.")
    return data[name].fillna(False).astype(bool)


def _prepare(panel: pd.DataFrame, p: Mapping[str, Any]) -> pd.DataFrame:
    required = {"zone", "timestamp_utc", "forecast_origin_utc", "label_available_at_utc", "actual", "forecast", "q10", "q90", *p["feature_columns"]}
    if panel.empty or panel.columns.has_duplicates or required.difference(panel):
        raise ScarcityPolicyError(f"Empty panel, duplicate columns or missing inputs: {sorted(required.difference(panel))}.")
    data = panel.copy(deep=True).reset_index(drop=True)
    data["_row"] = np.arange(len(data))
    if data.zone.isna().any() or not data.zone.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ScarcityPolicyError("Non-empty explicit country codes required.")
    for key in ("timestamp_utc", "forecast_origin_utc", "label_available_at_utc"):
        data[key] = _utc(data[key], key, missing=key == "label_available_at_utc")
    if data.duplicated(["zone", "timestamp_utc"]).any() or not data.timestamp_utc.eq(data.timestamp_utc.dt.floor("h")).all():
        raise ScarcityPolicyError("Unique physical hourly zone/timestamp keys required.")
    civil = data.timestamp_utc.dt.tz_convert(p["timezone"]).dt.tz_localize(None).dt.normalize()
    expected_origin = (civil - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).dt.tz_localize(p["timezone"]).dt.tz_convert("UTC")
    if not data.forecast_origin_utc.eq(expected_origin).all():
        raise ScarcityPolicyError("Every origin must equal D-1 08:00 civil (including DST).")
    data["_day"] = civil.dt.strftime("%Y-%m-%d")
    civil_start = civil.dt.tz_localize(p["timezone"]).dt.tz_convert("UTC")
    civil_end = (civil + pd.Timedelta(days=1)).dt.tz_localize(p["timezone"]).dt.tz_convert("UTC")
    data["_physical_day_hours"] = (civil_end - civil_start).dt.total_seconds() / 3600
    for key in ("forecast", "q10", "q90", "actual", *p["feature_columns"]):
        data[key] = pd.to_numeric(data[key], errors="raise").astype(float)
        if np.isinf(data[key]).any():
            raise ScarcityPolicyError(f"{key}: infinite values forbidden.")
    # Consolidate the numeric blocks once; hundreds of audited features must
    # not fragment every subsequent daily slice or produce warning floods.
    data = data.copy()
    if data[["forecast", "q10", "q90"]].isna().any().any() or not (data.q10.le(data.forecast) & data.forecast.le(data.q90)).all():
        raise ScarcityPolicyError("Finite ordered baseline quantiles required.")
    known = data.actual.notna()
    if (known & (data.label_available_at_utc.isna() | data.label_available_at_utc.le(data.forecast_origin_utc))).any():
        raise ScarcityPolicyError("Observed labels need an explicit publication time strictly after their own forecast origin.")
    eligible = _flag(data, "feature_eligible") & _flag(data, "forecast_eligible")
    eligible &= data[p["required_feature_columns"]].notna().all(axis=1)
    if "feature_available_at_utc" in data:
        available = _utc(data.feature_available_at_utc, "feature_available_at_utc", missing=True)
        eligible &= available.notna() & available.le(data.forecast_origin_utc)
    data["_features_valid"] = eligible
    data["_label_valid"] = _flag(data, "label_eligible") & known
    data["_error"] = data.actual - data.forecast
    return data.sort_values(["_day", "zone", "timestamp_utc"]).reset_index(drop=True)


def _hours(first: str, end_exclusive: str) -> int:
    a = pd.Timestamp(first).tz_localize("Europe/Paris").tz_convert("UTC")
    b = pd.Timestamp(end_exclusive).tz_localize("Europe/Paris").tz_convert("UTC")
    return int((b - a).total_seconds() / 3600)


def _matrix(frame: pd.DataFrame, features: list[str], zones: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([frame[features].to_numpy(float), *(frame.zone.eq(z).to_numpy(float) for z in zones)])


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values)).reshape(-1, 1)


def _fit(data: pd.DataFrame, day: str, cutoff: pd.Timestamp, p: Mapping[str, Any], zones: tuple[str, ...]):
    first = max(str(data._day.min()), (pd.Timestamp(day) - pd.Timedelta(days=p["training_window_days"])).strftime("%Y-%m-%d"))
    days = max(0, (pd.Timestamp(day) - pd.Timestamp(first)).days)
    past = data.loc[data._day.ge(first) & data._day.lt(day)].copy()
    usable = past._features_valid & past._label_valid & past.label_available_at_utc.le(cutoff)
    train = past.loc[usable]
    expected = _hours(first, day) * len(zones) if days else 0
    coverage = len(train) / expected if expected else 0.
    zone_expected = expected / len(zones) if expected else 0
    zone_coverage = {z: float(train.zone.eq(z).sum() / zone_expected) if zone_expected else 0. for z in zones}
    record = {"fit_day": day, "fit_cutoff_utc": cutoff, "training_start_day": first,
              "training_end_day": (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
              "training_days": days, "full_365_day_training": days == 365, "training_rows": len(train),
              "training_expected_hours": expected, "training_coverage": coverage,
              "training_coverage_by_zone": zone_coverage,
              "max_label_available_at_utc": train.label_available_at_utc.max(),
              "status": "fallback", "reason": "", "zones": list(zones)}
    if days < p["minimum_training_days"]:
        record["reason"] = "insufficient_training_days"
    elif coverage < p["minimum_training_coverage"] or min(zone_coverage.values()) < p["minimum_training_coverage"]:
        record["reason"] = "insufficient_training_coverage"
    if record["reason"]:
        return None, record
    calibration_start = (pd.Timestamp(day) - pd.Timedelta(days=p["calibration_days"])).strftime("%Y-%m-%d")
    core, calibration = train.loc[train._day.lt(calibration_start)], train.loc[train._day.ge(calibration_start)]
    if core.empty or calibration.empty:
        record["reason"] = "empty_chronological_split"
        return None, record
    thresholds = {}
    for zone in zones:
        errors = core.loc[core.zone.eq(zone), "_error"].to_numpy(float)
        if not len(errors):
            record["reason"] = "missing_zone_training"
            return None, record
        thresholds[zone] = max(p["minimum_threshold_eur_mwh"], float(np.quantile(errors, p["threshold_quantile"])))
    core_u = core.zone.map(thresholds).to_numpy(float)
    cal_u = calibration.zone.map(thresholds).to_numpy(float)
    y = core._error.to_numpy(float) >= core_u
    cal_y = calibration._error.to_numpy(float) >= cal_u
    record.update(thresholds_eur_mwh=thresholds, calibration_start_day=calibration_start,
                  model_training_end_day=str(core._day.max()), calibration_rows=len(calibration),
                  training_tail_rows=int(y.sum()), calibration_tail_rows=int(cal_y.sum()))
    if min(int(y.sum()), int((~y).sum())) < p["minimum_tail_training_rows"]:
        record["reason"] = "insufficient_training_classes"
        return None, record
    if min(int(cal_y.sum()), int((~cal_y).sum())) < p["minimum_calibration_class_rows"] or cal_y.sum() < p["minimum_tail_calibration_rows"]:
        record["reason"] = "insufficient_chronological_calibration_events"
        return None, record
    common = {key: p[key] for key in ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "random_state")}
    common["early_stopping"] = False
    X, V = _matrix(core, p["feature_columns"], zones), _matrix(calibration, p["feature_columns"], zones)
    with threadpool_limits(limits=p["threads"]):
        classifier = HistGradientBoostingClassifier(**common).fit(X, y)
        calibrator = LogisticRegression(C=1., solver="lbfgs", random_state=p["random_state"]).fit(_logit(classifier.predict_proba(V)[:, 1]), cal_y)
        # No class weights or resampling: there is no uncorrected prior shift.
        severity = HistGradientBoostingRegressor(loss="quantile", quantile=.5, **common).fit(
            X[y], np.log(core._error.to_numpy(float)[y] / core_u[y]))
        estimated_log_ratio = severity.predict(V[cal_y])
    tail_log_errors = np.log(calibration._error.to_numpy(float)[cal_y] / cal_u[cal_y]) - estimated_log_ratio
    state = {"classifier": classifier, "calibrator": calibrator, "severity": severity,
             "thresholds": thresholds, "tail_log_errors": tail_log_errors, "zones": zones,
             "features": p["feature_columns"], "fit_day": day, "fit_cutoff": cutoff}
    record.update(status="trained", reason="", probability_calibration="chronological_platt_no_class_weighting",
                  median_model="zero_atom_plus_positive_tail_mixture", tail_distribution_rows=len(tail_log_errors))
    return state, record


def _predict(state: Mapping[str, Any], frame: pd.DataFrame, p: Mapping[str, Any]):
    X = _matrix(frame, state["features"], state["zones"])
    u = frame.zone.map(state["thresholds"]).to_numpy(float)
    with threadpool_limits(limits=p["threads"]):
        probability = state["calibrator"].predict_proba(_logit(state["classifier"].predict_proba(X)[:, 1]))[:, 1]
        location = state["severity"].predict(X)
    level = np.where(probability > .5, 1 - .5 / np.maximum(probability, .5), 0.)
    residual_quantile = np.quantile(state["tail_log_errors"], level)
    tail_median = u * np.exp(np.clip(location + residual_quantile, -20, 20))
    raw = np.where(probability > max(.5, p["probability_gate"]), np.maximum(u, tail_median), 0.)
    return probability, raw, u


def _column(weight: float) -> str:
    return "_forecast_weight_" + format(weight, ".12g").replace(".", "p")


def _eligible_oos(records: list[pd.DataFrame], zone: str, day: str, cutoff: pd.Timestamp, p: Mapping[str, Any]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    first = (pd.Timestamp(day) - pd.Timedelta(days=p["governance_lookback_days"])).strftime("%Y-%m-%d")
    past = pd.concat(records, ignore_index=True)
    past = past.loc[past.zone.eq(zone) & past._day.ge(first) & past._day.lt(day)].copy()
    if past.empty:
        return past
    past["_label_known_now"] = past._label_valid & past.label_available_at_utc.le(cutoff)
    summary = past.groupby("_day", sort=False).agg(
        rows=("timestamp_utc", "size"), expected=("_physical_day_hours", "first"),
        labels_known=("_label_known_now", "all"), any_ready=("expert_ready", "any"))
    valid_days = summary.index[summary.rows.eq(summary.expected) & summary.labels_known & summary.any_ready]
    return past.loc[past._day.isin(valid_days)].copy()


def _govern(past: pd.DataFrame, zone: str, day: str, cutoff: pd.Timestamp, p: Mapping[str, Any]):
    base = {"zone": zone, "delivery_day": day, "decision_cutoff_utc": cutoff,
            "oos_days": int(past._day.nunique()) if len(past) else 0,
            "oos_last_day": str(past._day.max()) if len(past) else None,
            "max_oos_label_available_at_utc": past.label_available_at_utc.max() if len(past) else pd.NaT}
    if base["oos_days"] < p["governance_minimum_days"]:
        return 0., "governance_oos_warmup", [{**base, "weight": 0., "selected": True, "reason": "governance_oos_warmup"}]
    baseline_error = np.abs(past.actual - past.forecast)
    tail = (past.actual - past.forecast).ge(past.threshold_eur_mwh) & past.threshold_eur_mwh.notna()
    records, best, best_tail = [], 0., p["governance_minimum_tail_gain_eur_mwh"]
    for weight in p["candidate_weights"]:
        if weight == 0:
            continue
        changed = ~np.isclose(past[_column(weight)], past.forecast, atol=1e-9, rtol=0.)
        gain = baseline_error - np.abs(past.actual - past[_column(weight)])
        daily = gain.groupby(past._day).agg(["sum", "size"])
        # Hour-weighted MAE matches evaluation even on 23/25-hour civil days.
        # Uncertainty is clustered by day, never by pooled zone-hour samples.
        mean_gain = float(gain.mean())
        influence = daily["sum"] - mean_gain * daily["size"]
        standard_error = np.sqrt(len(daily) / (len(daily) - 1) * float(np.square(influence).sum())) / len(gain)
        lower = mean_gain - p["governance_uncertainty_z"] * standard_error
        tail_gain = float(gain.loc[tail].mean()) if tail.any() else 0.
        changed_days = int(past.loc[changed, "_day"].nunique())
        reason = "qualified"
        if changed_days < p["governance_minimum_changed_days"]:
            reason = "insufficient_changed_days"
        elif int(tail.sum()) < p["governance_minimum_tail_rows"]:
            reason = "insufficient_tail_events"
        elif lower < p["governance_minimum_mae_gain_eur_mwh"]:
            reason = "all_hour_mae_guard"
        elif tail_gain <= p["governance_minimum_tail_gain_eur_mwh"]:
            reason = "tail_gain_guard"
        records.append({**base, "weight": weight, "changed_days": changed_days, "tail_rows": int(tail.sum()),
                        "mean_mae_gain_eur_mwh": mean_gain, "mae_gain_lower_bound_eur_mwh": lower,
                        "tail_mae_gain_eur_mwh": tail_gain, "reason": reason})
        if reason == "qualified" and tail_gain > best_tail + 1e-12:
            best, best_tail = weight, tail_gain
    records.append({**base, "weight": 0., "reason": "identity"})
    for record in records:
        record["selected"] = record["weight"] == best
    return best, "tail_expert_selected" if best else "no_candidate_passes_guards", records


def _intervals(current: pd.DataFrame, past: pd.DataFrame, weight: float, p: Mapping[str, Any]):
    lo, hi = current.q10.to_numpy(float).copy(), current.q90.to_numpy(float).copy()
    status = np.full(len(current), "baseline_preserved", dtype=object)
    if weight == 0:
        return lo, hi, status
    for gate in (False, True):
        selected = current._tail_gate.eq(gate).to_numpy() & current.expert_ready.to_numpy(bool)
        if not selected.any():
            continue
        sample = past.loc[past._tail_gate.eq(gate)] if len(past) else past
        label = "empirical_oos_regime_errors"
        if len(sample) < p["interval_minimum_rows"]:
            sample, label = past, "empirical_oos_all_hour_errors"
        center = current.candidate_forecast.to_numpy(float)[selected]
        if len(sample) >= p["interval_minimum_rows"]:
            errors = (sample.actual - sample[_column(weight)]).to_numpy(float)
            lower, upper = np.quantile(errors, [.1, .9])
            lo[selected], hi[selected] = np.minimum(center + lower, center), np.maximum(center + upper, center)
            status[selected] = label
        else:
            lo[selected], hi[selected] = np.minimum(lo[selected], center), np.maximum(hi[selected], center)
            status[selected] = "baseline_envelope_uncalibrated"
    return lo, hi, status


def run_policy(panel: pd.DataFrame, settings: dict[str, Any], *, fit_callback=None, predict_callback=None) -> PolicyResult:
    """Return predictions in original row order; fit/calibrate/govern strictly as-of."""
    p = _parameters(settings)
    fit_model = _fit if fit_callback is None else fit_callback
    predict_model = _predict if predict_callback is None else predict_callback
    if not callable(fit_model) or not callable(predict_model):
        raise ScarcityPolicyError("Model callbacks must be callable.")
    data = _prepare(panel, p)
    zones = tuple(sorted(data.zone.unique()))
    LOGGER.info("[NYX scarcity] policy start: rows=%s, zones=%s, days=%s, rolling<=365, minimum=%s",
                len(data), ",".join(zones), data._day.nunique(), p["minimum_training_days"])
    state, fitted_at, oos, blocks, folds, governance = None, None, [], [], [], []
    for day, block in data.groupby("_day", sort=True):
        current = block.copy()
        cutoff = current.forecast_origin_utc.iloc[0]
        if fitted_at is None or (pd.Timestamp(day) - pd.Timestamp(fitted_at)).days >= p["refit_every_days"]:
            state, fold = fit_model(data, day, cutoff, p, zones)
            folds.append(fold)
            fitted_at = day
            LOGGER.info("[NYX scarcity] refit %s: %s (%s), training_days=%s, rows=%s",
                        day, fold["status"], fold["reason"], fold["training_days"], fold["training_rows"])
        n = len(current)
        probability, raw, threshold = np.full(n, np.nan), np.zeros(n), np.full(n, np.nan)
        ready = current._features_valid.to_numpy(bool) & (state is not None)
        if ready.any():
            probability[ready], raw[ready], threshold[ready] = predict_model(state, current.loc[ready], p)
        current["spike_probability"], current["raw_correction"] = probability, raw
        current["probability_gate"] = p["probability_gate"]
        current["threshold_eur_mwh"], current["expert_ready"] = threshold, ready
        current["expert_fit_day"] = state["fit_day"] if state else None
        current["_tail_gate"] = ready & (raw > 0)
        clipped = np.clip(raw, 0., p["correction_clip_eur_mwh"])
        current["bounded_correction"] = clipped
        for weight in p["candidate_weights"]:
            current[_column(weight)] = current.forecast + weight * clipped
        current["selected_weight"], current["applied_correction"] = 0., 0.
        current["candidate_forecast"] = current.forecast
        current["candidate_q10"], current["candidate_q90"] = current.q10, current.q90
        current["interval_status"] = "baseline_preserved"
        current["gate_reason"] = "expert_warmup_or_unavailable"
        for zone, part in current.groupby("zone", sort=True):
            past = _eligible_oos(oos, zone, day, cutoff, p)
            weight, reason, records = _govern(past, zone, day, cutoff, p)
            governance.extend(records)
            idx = part.index
            current.loc[idx, "selected_weight"] = np.where(part.expert_ready, weight, 0.)
            current.loc[idx, "applied_correction"] = weight * clipped[current.index.get_indexer(idx)]
            current.loc[idx, "candidate_forecast"] = current.loc[idx, _column(weight)]
            reasons = np.where(~part._features_valid, "required_features_unavailable",
                               np.where(~part.expert_ready, "expert_warmup_or_unavailable",
                                        np.where(~part._tail_gate, "probability_below_gate", reason)))
            current.loc[idx, "gate_reason"] = reasons
            lo, hi, interval_status = _intervals(current.loc[idx], past, weight, p)
            current.loc[idx, "candidate_q10"], current.loc[idx, "candidate_q90"] = lo, hi
            current.loc[idx, "interval_status"] = interval_status
        blocks.append(current)
        # Persist unselected fixed candidates too, but only ones produced by the
        # model available at this origin. Current labels cannot govern today.
        if current.expert_ready.any():
            oos.append(current)
        first = (pd.Timestamp(day) - pd.Timedelta(days=p["governance_lookback_days"])).strftime("%Y-%m-%d")
        oos = [part for part in oos if str(part._day.iloc[0]) >= first]
    predictions = pd.concat(blocks).sort_values("_row")
    internal = [c for c in predictions if c.startswith("_") and c not in panel]
    predictions = predictions.drop(columns=internal).reset_index(drop=True)
    predictions.index = panel.index.copy()
    # The frozen input contract is byte/value/dtype strict at the runner seam.
    # Normalise only the working copy; return original columns unchanged.
    additions = predictions.drop(columns=list(panel.columns))
    predictions = pd.concat([panel.copy(deep=True), additions], axis=1)
    fitted = pd.DataFrame(folds)
    trained = fitted.loc[fitted.status.eq("trained")]
    values = predictions[["candidate_q10", "candidate_forecast", "candidate_q90"]].to_numpy(float)
    if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
        raise ScarcityPolicyError("Expert produced non-finite or crossing quantiles.")
    audit = {
        "schema_version": 1, "engine": "nyx_scarcity_hurdle_v1", "diagnostic_only": True,
        "production_modified": False, "activation_performed": False, "orders_placed": False,
        "storm_used_for_training_or_governance": False, "economic_pnl_used": False,
        "target": "positive_large_actual_minus_frozen_nyx_error", "pooled_training_zones": list(zones),
        "training_window_cap_days": 365, "minimum_training_days": p["minimum_training_days"],
        "refit_every_days": p["refit_every_days"], "refit_is_daily": p["refit_every_days"] == 1,
        "calibration_days": p["calibration_days"], "governance_lookback_days": p["governance_lookback_days"],
        "trained_folds": len(trained), "fallback_folds": len(fitted) - len(trained),
        "actual_training_days_min": int(trained.training_days.min()) if len(trained) else None,
        "actual_training_days_max": int(trained.training_days.max()) if len(trained) else None,
        "all_trained_folds_have_365_days": bool(len(trained) and trained.full_365_day_training.all()),
        "probability_calibration": "chronological_heldout_platt_without_class_weights",
        "threshold_fitted_on": "model_training_block_excluding_calibration_and_future",
        "point_functional": "median_of_zero_atom_positive_tail_mixture_then_governed_shrinkage",
        "interval_method": "separate_empirical_prequential_signed_errors_by_predicted_regime_or_all_hours",
        "identity_interval_policy": "weight_zero_preserves_baseline_p10_p90_even_when_spike_probability_is_high",
        "interval_finite_sample_coverage_guaranteed": False, "annual_non_regression_guaranteed": False,
        "governance_predictions_are_prequential_oos": True, "governance_is_daily_by_zone": True,
        "changed_forecast_rows": int(predictions.applied_correction.ne(0).sum()),
        "expert_ready_rows": int(predictions.expert_ready.sum()), "decision_rows": len(predictions),
        "label_availability": "explicit_input_timestamps_not_certified_publication_vintages",
        "limitations": ["Positive tail only; not an estimator of negative spikes.",
                       "The zero-atom bulk is a modelling approximation, not a complete calibrated residual distribution.",
                       "Historical MAE guards do not guarantee future annual non-regression.",
                       "OOS empirical intervals have neither exchangeability nor guaranteed regime coverage.",
                       "The rolling window is capped at 365 days at each fit; a weekly fitted model is reused between scheduled fits.",
                       "Progressive warmup must not be described as 365 training days in every fold."],
        "config": p,
    }
    LOGGER.info("[NYX scarcity] policy complete: trained_folds=%s, expert_ready=%s, changed=%s",
                audit["trained_folds"], audit["expert_ready_rows"], audit["changed_forecast_rows"])
    return PolicyResult(predictions, fitted, pd.DataFrame(governance), audit)
