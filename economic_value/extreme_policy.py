"""Causal, research-only fundamental tail expert with baseline-relative governance.

Fits use a fixed 365-civil-day history.  Policies are evaluated prequentially:
only candidate predictions actually produced before their labels became known
may enter governance.  Storm is never a predictor or a governance comparator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


POLICIES = ("baseline", "reduce_opposite", "blend_tail")
FEATURE_METADATA = {"feature_eligible", "feature_available_at_utc", "feature_pit_certified"}


class ExtremePolicyError(ValueError):
    """Invalid data/policy contract; no operational model is modified."""


@dataclass(frozen=True)
class ExtremePolicyResult:
    decisions: pd.DataFrame
    folds: pd.DataFrame
    governance: pd.DataFrame
    audit: dict[str, Any]


def _utc(values: pd.Series, name: str) -> pd.Series:
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert("UTC")
    for value in values.dropna():
        if pd.Timestamp(value).tzinfo is None:
            raise ExtremePolicyError(f"{name}: explicit timezone required.")
    return pd.to_datetime(values, format="mixed", utc=True, errors="raise")


def _flag(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[name]
    if not values.dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ExtremePolicyError(f"{name}: explicit boolean values required.")
    return values.fillna(False).astype(bool)


def _number(config: Mapping[str, Any], key: str, default: float, *, low: float = 0, high: float = np.inf) -> float:
    value = float(config.get(key, default))
    if not np.isfinite(value) or value < low or value > high:
        raise ExtremePolicyError(f"{key}: must lie in [{low}, {high}].")
    return value


def _integer(config: Mapping[str, Any], key: str, default: int, *, low: int = 1) -> int:
    value = _number(config, key, default, low=low)
    if value != int(value):
        raise ExtremePolicyError(f"{key}: integer required.")
    return int(value)


def _parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    p: dict[str, Any] = dict(config)
    p["training_days"] = _integer(config, "training_days", 365)
    if p["training_days"] != 365:
        raise ExtremePolicyError("This experiment requires exactly 365 civil training days; shorter histories may not be substituted.")
    p["minimum_training_calendar_days"] = _integer(config, "minimum_training_calendar_days", 365)
    if p["minimum_training_calendar_days"] != 365:
        raise ExtremePolicyError("minimum_training_calendar_days must be 365; missing feature hours are governed separately by coverage.")
    p["minimum_training_coverage"] = _number(config, "minimum_training_coverage", .95, low=.5, high=1)
    p["refit_every_days"] = _integer(config, "refit_every_days", 7)
    p["extreme_move_eur_mwh"] = _number(config, "extreme_move_eur_mwh", 50, low=.01)
    p["extreme_probability_threshold"] = _number(config, "extreme_probability_threshold", .6, low=.5, high=1)
    p["minimum_amplitude_events"] = _integer(config, "minimum_amplitude_events", 30, low=2)
    p["max_iter"] = _integer(config, "max_iter", 60)
    p["learning_rate"] = _number(config, "learning_rate", .06, low=.00001, high=1)
    p["max_leaf_nodes"] = _integer(config, "max_leaf_nodes", 15, low=2)
    p["min_samples_leaf"] = _integer(config, "min_samples_leaf", 40, low=2)
    p["l2_regularization"] = _number(config, "l2_regularization", 10)
    p["random_state"] = _integer(config, "random_state", 1729, low=0)
    p["reduction_fraction"] = _number(config, "reduction_fraction", .5, high=1)
    p["expert_blend_weight"] = _number(config, "expert_blend_weight", .5, high=1)
    p["governance_lookback_days"] = _integer(config, "governance_lookback_days", 60)
    p["governance_minimum_days"] = _integer(config, "governance_minimum_days", 28, low=2)
    if p["governance_minimum_days"] > p["governance_lookback_days"]:
        raise ExtremePolicyError("Governance minimum days exceed its lookback window.")
    p["governance_minimum_changed_days"] = _integer(config, "governance_minimum_changed_days", 7)
    p["governance_minimum_gain_eur_mwh"] = _number(config, "governance_minimum_gain_eur_mwh", .05)
    p["governance_regularization_eur_mwh"] = _number(config, "governance_regularization_eur_mwh", .02)
    p["governance_uncertainty_z"] = _number(config, "governance_uncertainty_z", 1.96)
    p["transaction_cost_eur_mwh"] = _number(config, "transaction_cost_eur_mwh", .5)
    p["slippage_eur_mwh"] = _number(config, "slippage_eur_mwh", .5)
    p["signal_threshold_eur_mwh"] = _number(config, "signal_threshold_eur_mwh", 5)
    p["cost_eur_mwh"] = p["transaction_cost_eur_mwh"] + p["slippage_eur_mwh"]
    p["hurdle_eur_mwh"] = p["signal_threshold_eur_mwh"] + p["cost_eur_mwh"]
    p["timezone"] = str(config.get("timezone", "Europe/Paris"))
    if p["timezone"] != "Europe/Paris":
        raise ExtremePolicyError("The qualified source contract requires Europe/Paris and the fixed D-1 08:00 civil cutoff.")
    p["candidate_policies"] = list(config.get("candidate_policies", POLICIES))
    if (not p["candidate_policies"] or p["candidate_policies"][0] != "baseline"
            or len(set(p["candidate_policies"])) != len(p["candidate_policies"])
            or not set(p["candidate_policies"]).issubset(POLICIES)):
        raise ExtremePolicyError(f"candidate_policies must begin with baseline and select uniquely from {POLICIES}.")
    return p


def _prepare(frame: pd.DataFrame, features: list[str], p: Mapping[str, Any], *, evaluation: bool) -> pd.DataFrame:
    result = frame.copy(deep=True)
    required = ["timestamp_utc", "zone", "reference_price", "actual", "forecast_origin_utc"]
    if evaluation:
        required += ["forecast"]
    missing = [column for column in required if column not in result]
    if missing:
        raise ExtremePolicyError(f"Missing columns: {missing}.")
    for column in ("timestamp_utc", "forecast_origin_utc"):
        result[column] = _utc(result[column], column)
    if result[["timestamp_utc", "forecast_origin_utc", "zone"]].isna().any().any():
        raise ExtremePolicyError("Delivery, zone and forecast origin must be present.")
    result["zone"] = result["zone"].astype(str).str.upper()
    if result.duplicated(["zone", "timestamp_utc"]).any():
        raise ExtremePolicyError("Duplicate zone/delivery interval; supply one baseline per zone.")
    for column in features:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="raise")
    for column in ("reference_price", "actual", "forecast"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="raise")
    if "label_available_at_utc" in result:
        result["label_available_at_utc"] = _utc(result["label_available_at_utc"], "label_available_at_utc")
    else:
        # No implicit publication assumption. A reader may supply an audited
        # conservative availability hypothesis, but this engine never creates it.
        result["label_available_at_utc"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    result["_label_eligible"] = _flag(result, "label_eligible", True)
    if "reference_available_at_utc" in result:
        reference_known = _utc(result["reference_available_at_utc"], "reference_available_at_utc") <= result["forecast_origin_utc"]
    else:
        reference_known = pd.Series(True, index=result.index)
    result["_reference_eligible"] = _flag(result, "reference_eligible", True) & reference_known & np.isfinite(result["reference_price"])
    result["_feature_eligible"] = _flag(result, "feature_eligible", False)
    if "feature_available_at_utc" in result:
        result["_feature_eligible"] &= _utc(result["feature_available_at_utc"], "feature_available_at_utc") <= result["forecast_origin_utc"]
    if features:
        # Missing optional features are handled natively by HGB. Infinite values
        # are not silently imputed, and core coverage comes from the source audit.
        result["_feature_eligible"] &= ~np.isinf(result[features]).any(axis=1)
        required_features = list(p.get("required_feature_columns", []))
        if not set(required_features).issubset(features):
            raise ExtremePolicyError("required_feature_columns must be a subset of feature_columns.")
        if required_features:
            result["_feature_eligible"] &= np.isfinite(result[required_features]).all(axis=1)
        result["_feature_eligible"] &= result[features].notna().any(axis=1)
    else:
        result["_feature_eligible"] = False
    result["duration_hours"] = pd.to_numeric(result.get("duration_hours", pd.Series(1., index=result.index)), errors="raise")
    if not result["duration_hours"].eq(1).all():
        raise ExtremePolicyError("The current tail expert contract requires one-hour physical intervals.")
    local = result["timestamp_utc"].dt.tz_convert(p["timezone"])
    result["_day"] = local.dt.strftime("%Y-%m-%d")
    if not local.dt.minute.eq(0).all() or not local.dt.second.eq(0).all():
        raise ExtremePolicyError("Delivery timestamps must be aligned to physical hours.")
    if not (result["forecast_origin_utc"] < result["timestamp_utc"]).all():
        raise ExtremePolicyError("Forecast origin must precede delivery.")
    expected_origins = (local.dt.tz_localize(None).dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).dt.tz_localize(p["timezone"]).dt.tz_convert("UTC")
    if not result["forecast_origin_utc"].eq(expected_origins).all():
        raise ExtremePolicyError("Both training and evaluation require the fixed D-1 08:00 civil information cutoff.")
    if evaluation:
        result["_baseline_eligible"] = _flag(result, "forecast_eligible", True) & result["_reference_eligible"] & np.isfinite(result["forecast"])
    return result.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)


def _day_hours(day: str, timezone: str) -> int:
    start = pd.Timestamp(day).tz_localize(timezone)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(timezone)
    return int((end.tz_convert("UTC") - start.tz_convert("UTC")).total_seconds() / 3600)


def _fit(history: pd.DataFrame, features: list[str], day: str, cutoff: pd.Timestamp, zone: str, p: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    first = (pd.Timestamp(day) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    last = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    window = history.loc[history["_day"].between(first, last)].copy()
    first_utc = pd.Timestamp(first).tz_localize(p["timezone"]).tz_convert("UTC")
    end_utc = pd.Timestamp(day).tz_localize(p["timezone"]).tz_convert("UTC")
    expected_hours = int((end_utc - first_utc).total_seconds() / 3600)
    known_labels = window["label_available_at_utc"].notna() & (window["label_available_at_utc"] <= cutoff)
    usable = (known_labels & window["_label_eligible"] & window["_feature_eligible"] & window["_reference_eligible"]
              & np.isfinite(window["actual"]) & (window["forecast_origin_utc"] < cutoff))
    train = window.loc[usable]
    coverage = len(train) / expected_hours
    fold: dict[str, Any] = {
        "zone": zone, "fit_delivery_day": day, "fit_cutoff_utc": cutoff,
        "training_start_day": first, "training_end_day": last,
        "training_calendar_days_required": 365, "training_calendar_days_present": int(window["_day"].nunique()),
        "training_expected_hours": expected_hours, "training_valid_hours": len(train),
        "training_coverage": coverage, "labels_unavailable_at_cutoff": int((~known_labels).sum()),
        "features": list(features), "feature_count": len(features),
        "status": "unavailable", "reason": "",
        "max_training_label_available_at_utc": train["label_available_at_utc"].max() if len(train) else pd.NaT,
    }
    if not features:
        fold["reason"] = "no_fundamental_features"
    elif window["_day"].nunique() != 365:
        fold["reason"] = "insufficient_365_day_calendar"
    elif coverage < p["minimum_training_coverage"]:
        fold["reason"] = "insufficient_training_feature_or_label_coverage"
    if fold["reason"]:
        return None, fold
    X = train[features].to_numpy(dtype=float)
    delta = (train["actual"] - train["reference_price"]).to_numpy(dtype=float)
    threshold = p["extreme_move_eur_mwh"]
    classes = np.where(delta >= threshold, 1, np.where(delta <= -threshold, -1, 0))
    common = {
        "max_iter": p["max_iter"], "learning_rate": p["learning_rate"],
        "max_leaf_nodes": p["max_leaf_nodes"], "min_samples_leaf": p["min_samples_leaf"],
        "l2_regularization": p["l2_regularization"], "random_state": p["random_state"],
        "early_stopping": False,
    }
    state: dict[str, Any] = {"fit_day": day, "fit_cutoff": cutoff, "features": features, "fold": fold}
    values, counts = np.unique(classes, return_counts=True)
    state["class_probabilities"] = {int(value): float(count / len(classes)) for value, count in zip(values, counts)}
    state["classifier"] = None
    with threadpool_limits(limits=1):
        if len(values) > 1:
            state["classifier"] = HistGradientBoostingClassifier(**common).fit(X, classes)
        for side, code in (("up", 1), ("down", -1)):
            selected = classes == code
            amplitudes = np.abs(delta[selected])
            average = float(amplitudes.mean()) if len(amplitudes) else float(threshold)
            cap = max(float(np.quantile(amplitudes, .995)), threshold) if len(amplitudes) else float(threshold)
            regressor = None
            if len(amplitudes) >= p["minimum_amplitude_events"]:
                regressor = HistGradientBoostingRegressor(**common).fit(X[selected], amplitudes)
            state[side] = {"regressor": regressor, "mean": average, "cap": cap}
            fold[f"{side}_training_events"] = int(selected.sum())
            fold[f"{side}_amplitude_method"] = "hgb" if regressor is not None else "training_events_mean_or_zero_probability"
    calm = delta[classes == 0]
    state["calm_mean"] = float(calm.mean()) if len(calm) else 0.
    fold["status"] = "trained"
    fold["reason"] = ""
    fold["classifier_method"] = "hgb" if state["classifier"] is not None else "single_observed_training_class"
    fold["training_regime_counts"] = {str(int(value)): int(count) for value, count in zip(values, counts)}
    return state, fold


def _predict(state: Mapping[str, Any], frame: pd.DataFrame, p: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = frame[state["features"]].to_numpy(dtype=float)
    probabilities = {-1: np.zeros(len(frame)), 0: np.zeros(len(frame)), 1: np.zeros(len(frame))}
    with threadpool_limits(limits=1):
        if state["classifier"] is None:
            for code, probability in state["class_probabilities"].items():
                probabilities[code][:] = probability
        else:
            values = state["classifier"].predict_proba(X)
            for offset, code in enumerate(state["classifier"].classes_):
                probabilities[int(code)] = values[:, offset]
        amplitudes = {}
        for side in ("up", "down"):
            info = state[side]
            values = info["regressor"].predict(X) if info["regressor"] is not None else np.full(len(frame), info["mean"])
            amplitudes[side] = np.clip(values, p["extreme_move_eur_mwh"], info["cap"])
    expected = probabilities[1] * amplitudes["up"] - probabilities[-1] * amplitudes["down"] + probabilities[0] * state["calm_mean"]
    return probabilities[1], probabilities[-1], expected


def _net_per_mw(position: pd.Series | np.ndarray, delta: pd.Series | np.ndarray, duration: pd.Series | np.ndarray, cost: float) -> np.ndarray:
    return (np.asarray(position) * np.asarray(delta) - np.abs(np.asarray(position)) * cost) * np.asarray(duration)


def _govern(oof: list[pd.DataFrame], day: str, cutoff: pd.Timestamp, zone: str, p: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    base_record: dict[str, Any] = {"zone": zone, "delivery_day": day, "decision_cutoff_utc": cutoff}
    if not oof:
        return "baseline", [{**base_record, "candidate": "baseline", "reason": "governance_oof_warmup", "oof_complete_days": 0}], "governance_oof_warmup"
    past = pd.concat(oof, ignore_index=True)
    first = (pd.Timestamp(day) - pd.Timedelta(days=p["governance_lookback_days"])).strftime("%Y-%m-%d")
    valid = (past["_day"].ge(first) & past["_day"].lt(day) & past["expert_eligible"]
             & past["_baseline_eligible"] & past["_label_eligible"] & np.isfinite(past["actual"])
             & past["label_available_at_utc"].notna() & (past["label_available_at_utc"] <= cutoff)
             & (past["policy_available_at_utc"] < past["label_available_at_utc"]))
    past = past.loc[valid].copy()
    counts = past.groupby("_day")["duration_hours"].sum()
    complete_days = [date for date, hours in counts.items() if hours == _day_hours(date, p["timezone"])]
    past = past.loc[past["_day"].isin(complete_days)]
    if len(complete_days) < p["governance_minimum_days"]:
        return "baseline", [{**base_record, "candidate": "baseline", "reason": "governance_oof_warmup", "oof_complete_days": len(complete_days)}], "governance_oof_warmup"
    base_pnl = _net_per_mw(past["baseline_position_fraction"], past["actual"] - past["reference_price"], past["duration_hours"], p["cost_eur_mwh"])
    records = []
    best = "baseline"
    best_score = p["governance_minimum_gain_eur_mwh"]
    for candidate in p["candidate_policies"]:
        if candidate == "baseline":
            continue
        fraction = past[f"candidate_{candidate}_fraction"]
        gains = _net_per_mw(fraction, past["actual"] - past["reference_price"], past["duration_hours"], p["cost_eur_mwh"]) - base_pnl
        values = pd.DataFrame({"day": past["_day"], "gain": gains, "hours": past["duration_hours"], "changed": ~np.isclose(fraction, past["baseline_position_fraction"])}).groupby("day").agg(gain=("gain", "sum"), hours=("hours", "sum"), changed=("changed", "any"))
        gain_per_hour = values["gain"] / values["hours"]
        mean = float(gain_per_hour.mean())
        standard_error = float(gain_per_hour.std(ddof=1) / np.sqrt(len(values)))
        margin = p["governance_uncertainty_z"] * standard_error
        score = mean - margin - p["governance_regularization_eur_mwh"]
        changed_days = int(values["changed"].sum())
        sufficient = changed_days >= p["governance_minimum_changed_days"]
        record = {**base_record, "candidate": candidate, "oof_complete_days": len(values),
                  "oof_first_day": min(complete_days), "oof_last_day": max(complete_days),
                  "max_oof_label_available_at_utc": past["label_available_at_utc"].max(),
                  "changed_days": changed_days, "mean_gain_eur_mwh": mean,
                  "standard_error_eur_mwh": standard_error, "uncertainty_margin_eur_mwh": margin,
                  "regularized_lower_bound_eur_mwh": score,
                  "reason": "qualified" if sufficient and score > p["governance_minimum_gain_eur_mwh"] else "insufficient_changed_days" if not sufficient else "insufficient_baseline_relative_evidence"}
        records.append(record)
        if sufficient and score > best_score:
            best, best_score = candidate, score
    for record in records:
        record["selected"] = record["candidate"] == best
    reason = "governed_candidate_selected" if best != "baseline" else "insufficient_baseline_relative_evidence"
    return best, records, reason


def run_extreme_policy(history: pd.DataFrame, evaluation_panel: pd.DataFrame, config: Mapping[str, Any]) -> ExtremePolicyResult:
    """Return causal per-zone position fractions; no trading or activation.

    ``history`` is independently materialised ex-ante fundamental data and labels,
    not baseline forecasts. ``evaluation_panel`` contains one fixed baseline per
    zone. Its realised values are accessed only after their explicit availability
    timestamps when governing subsequent decisions. Output retains every row.
    """
    p = _parameters(config)
    if evaluation_panel.empty:
        raise ExtremePolicyError("Evaluation panel is empty.")
    features = list(config.get("feature_columns", sorted(column for column in history.columns if column.startswith("feature_") and column not in FEATURE_METADATA)))
    if len(set(features)) != len(features) or any(not name.startswith("feature_") or name in FEATURE_METADATA for name in features):
        raise ExtremePolicyError("Feature names must be unique and explicitly prefixed feature_.")
    forbidden = ("storm", "benchmark", "actual", "realised", "realized", "observed", "target", "label", "pnl", "profit", "baseline_forecast")
    if any(any(word in name.lower() for word in forbidden) for name in features):
        raise ExtremePolicyError("A target, realised outcome, baseline forecast or Storm/benchmark variable cannot enter the expert features.")
    history_data = _prepare(history, features, p, evaluation=False)
    evaluation = _prepare(evaluation_panel, features, p, evaluation=True)
    decisions = []
    folds: list[dict[str, Any]] = []
    governance_records: list[dict[str, Any]] = []
    for zone, evaluation_zone in evaluation.groupby("zone", sort=True):
        history_zone = history_data.loc[history_data["zone"].eq(zone)]
        state = None
        last_attempt: str | None = None
        last_fold: dict[str, Any] = {}
        oof: list[pd.DataFrame] = []
        for day, current in evaluation_zone.groupby("_day", sort=True):
            current = current.copy().reset_index(drop=True)
            if current["forecast_origin_utc"].nunique() != 1:
                raise ExtremePolicyError(f"{zone}/{day}: one common forecast cutoff is required for the 24-hour horizon.")
            cutoff = current["forecast_origin_utc"].iloc[0]
            expected_origin = (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(p["timezone"]).tz_convert("UTC")
            if cutoff != expected_origin:
                raise ExtremePolicyError(f"{zone}/{day}: the fixed information cutoff is D-1 08:00 civil.")
            if last_attempt is None or (pd.Timestamp(day) - pd.Timestamp(last_attempt)).days >= p["refit_every_days"]:
                state, last_fold = _fit(history_zone, features, day, cutoff, zone, p)
                last_attempt = day
                folds.append(last_fold)
                if p.get("verbose", True):
                    print(f"[Extreme/{zone}] refit {day}: {last_fold['status']}; window={last_fold['training_start_day']}..{last_fold['training_end_day']}; hours={last_fold['training_valid_hours']}/{last_fold['training_expected_hours']}; {last_fold['reason']}", flush=True)
            selected, records, governance_reason = _govern(oof, day, cutoff, zone, p)
            governance_records.extend(records)
            baseline_edge = current["forecast"] - current["reference_price"]
            current["baseline_position_fraction"] = np.where(current["_baseline_eligible"] & (np.abs(baseline_edge) > p["hurdle_eur_mwh"]), np.sign(baseline_edge), 0.)
            current["extreme_probability_up"] = np.nan
            current["extreme_probability_down"] = np.nan
            current["extreme_expected_edge"] = np.nan
            current["expert_eligible"] = current["_feature_eligible"] & current["_reference_eligible"] & current["_baseline_eligible"] & (state is not None)
            if state is not None and current["expert_eligible"].any():
                valid = current["expert_eligible"]
                up, down, expected = _predict(state, current.loc[valid], p)
                current.loc[valid, "extreme_probability_up"] = up
                current.loc[valid, "extreme_probability_down"] = down
                current.loc[valid, "extreme_expected_edge"] = expected
            tail = (current["expert_eligible"] & (current[["extreme_probability_up", "extreme_probability_down"]].max(axis=1) >= p["extreme_probability_threshold"])
                    & (np.abs(current["extreme_expected_edge"]) > p["hurdle_eur_mwh"]))
            expert = np.sign(current["extreme_expected_edge"])
            base = current["baseline_position_fraction"]
            opposite = tail & (base * expert < 0)
            current["candidate_baseline_fraction"] = base
            current["candidate_reduce_opposite_fraction"] = np.where(opposite, base * p["reduction_fraction"], base)
            current["candidate_blend_tail_fraction"] = np.where(tail, (1 - p["expert_blend_weight"]) * base + p["expert_blend_weight"] * expert, base)
            current["policy_position_fraction"] = current[f"candidate_{selected}_fraction"]
            current["selected_policy"] = selected
            current.loc[~current["expert_eligible"], "selected_policy"] = "baseline"
            current["governance_weight"] = np.where(current["selected_policy"].eq("baseline"), 0., 1.)
            current["policy_available_at_utc"] = cutoff
            current["policy_reason"] = governance_reason
            current.loc[~tail & current["expert_eligible"] & ~current["selected_policy"].eq("baseline"), "policy_reason"] = "tail_trigger_inactive_baseline_position_retained"
            current.loc[~current["_feature_eligible"], "policy_reason"] = "fundamental_features_unavailable_baseline_fallback"
            if state is None:
                current["policy_reason"] = last_fold.get("reason", "training_unavailable_baseline_fallback")
            current.loc[~current["_baseline_eligible"], "policy_reason"] = "baseline_or_reference_unavailable"
            current["training_start_day"] = last_fold.get("training_start_day")
            current["training_end_day"] = last_fold.get("training_end_day")
            current["fit_cutoff_utc"] = last_fold.get("fit_cutoff_utc", pd.NaT)
            current["fit_age_civil_days"] = (pd.Timestamp(day) - pd.Timestamp(last_attempt)).days
            current["oof_prediction"] = current["expert_eligible"]
            if (np.abs(current["policy_position_fraction"]) > 1 + 1e-12).any() or current["policy_position_fraction"].isna().any():
                raise ExtremePolicyError("Policy produced a position outside the fixed [-1, 1] allocation.")
            decisions.append(current)
            oof.append(current)
            # Retain only a bounded daily history; calibration never reaches back
            # beyond the configured past-only governance window.
            retain_from = (pd.Timestamp(day) - pd.Timedelta(days=p["governance_lookback_days"])).strftime("%Y-%m-%d")
            oof = [block for block in oof if block["_day"].iloc[0] >= retain_from]
    output = pd.concat(decisions, ignore_index=True).sort_values(["timestamp_utc", "zone"]).reset_index(drop=True)
    trained = [fold for fold in folds if fold["status"] == "trained"]
    audit = {
        "schema_version": 1, "diagnostic_only": True, "production_modified": False,
        "activation_performed": False, "orders_placed": False, "training_days": 365,
        "minimum_training_coverage": p["minimum_training_coverage"], "refit_every_days": p["refit_every_days"],
        "refit_is_daily": p["refit_every_days"] == 1, "governance_is_daily": True,
        "cutoff": "D-1 08:00 Europe/Paris civil" if p["timezone"] == "Europe/Paris" else f"D-1 08:00 {p['timezone']} civil",
        "feature_columns": features, "benchmark_used_as_feature": False,
        "benchmark_used_for_governance": False, "governance_comparator": "fixed_baseline_only",
        "baseline_forecasts_used_for_training": False, "governance_predictions_are_prequential_oof": True,
        "governance_oof_before_evaluation_invented": False,
        "training_label_availability_is_explicit": "label_available_at_utc" in history.columns,
        "label_availability_assumption_created_by_engine": False,
        "optional_missing_features_handling": "native HGB NaN support; core validity supplied by feature_eligible",
        "trained_folds": len(trained), "unavailable_folds": len(folds) - len(trained),
        "decision_rows": len(output), "expert_eligible_rows": int(output["expert_eligible"].sum()),
        "changed_position_rows": int((~np.isclose(output["policy_position_fraction"], output["baseline_position_fraction"])).sum()),
        "fallback_reasons": output.loc[~output["expert_eligible"], "policy_reason"].value_counts().to_dict(),
        "probabilities_calibrated": False, "confidence_bound_is_diagnostic_not_guarantee": True,
        "candidate_policies": p["candidate_policies"], "config": p,
        "limitations": [
            "Tail labels use historical realised minus reference price; no realised value of the predicted day enters its decision.",
            "A 365-day fit window rolls at the configured refit cadence, not at every daily inference when cadence exceeds one day.",
            "Fewer than 365 historical dates causes baseline fallback; missing hours are admitted only within the declared coverage tolerance.",
            "Probability outputs are uncalibrated classifier scores, not guaranteed probabilities of profitable trading.",
            "Governance uses a regularized daily-clustered uncertainty margin, not proof of positive future EVA or statistical significance after model selection.",
            "Source vintage and baseline OOF quality remain upstream audit responsibilities; explicit availability timestamps do not certify unproven source publication history.",
            "Storm is excluded from expert training and policy selection and can only be scored downstream on the same paired sample.",
        ],
    }
    return ExtremePolicyResult(output, pd.DataFrame(folds), pd.DataFrame(governance_records), audit)
