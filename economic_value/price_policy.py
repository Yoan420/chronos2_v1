"""Isolated causal residual-price expert layered on one frozen baseline.

The output is a price forecast, not a prescribed trading position. Selection uses
past OOF price errors and simulated net PnL versus the same baseline only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


METADATA = {"feature_eligible", "feature_pit_certified", "feature_available_at_utc"}


class PricePolicyError(ValueError):
    """The experiment's temporal or fixed-policy contract is invalid."""


@dataclass(frozen=True)
class PricePolicyResult:
    decisions: pd.DataFrame
    folds: pd.DataFrame
    governance: pd.DataFrame
    audit: dict[str, Any]


def _utc(series: pd.Series, name: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC")
    if any(pd.Timestamp(value).tzinfo is None for value in series.dropna()):
        raise PricePolicyError(f"{name}: an explicit timezone is required.")
    return pd.to_datetime(series, utc=True, format="mixed", errors="raise")


def _flag(data: pd.DataFrame, name: str, default: bool) -> pd.Series:
    if name not in data:
        return pd.Series(default, index=data.index, dtype=bool)
    values = data[name]
    if not values.dropna().map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise PricePolicyError(f"{name}: explicit booleans are required.")
    return values.fillna(False).astype(bool)


def _number(config: Mapping[str, Any], key: str, default: float, low: float = 0., high: float = np.inf) -> float:
    value = float(config.get(key, default))
    if not np.isfinite(value) or not low <= value <= high:
        raise PricePolicyError(f"{key}: expected a finite number in [{low}, {high}].")
    return value


def _integer(config: Mapping[str, Any], key: str, default: int, low: int = 1, high: int = 100000) -> int:
    value = _number(config, key, default, low, high)
    if value != int(value):
        raise PricePolicyError(f"{key}: integer required.")
    return int(value)


def _parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    p = dict(config)
    p["training_window_days"] = _integer(config, "training_window_days", 365, 1, 365)
    if p["training_window_days"] != 365:
        raise PricePolicyError("This experiment fixes the maximum training window at 365 civil days; disclose partial warmup with minimum_training_days.")
    p["minimum_training_days"] = _integer(config, "minimum_training_days", 90, 90, 365)
    if p["minimum_training_days"] > p["training_window_days"]:
        raise PricePolicyError("minimum_training_days exceeds the bounded training window.")
    p["minimum_training_coverage"] = _number(config, "minimum_training_coverage", .95, .5, 1.)
    for key, default in (("refit_every_days", 7), ("max_iter", 60), ("max_leaf_nodes", 15), ("min_samples_leaf", 40),
                         ("governance_lookback_days", 60), ("governance_minimum_days", 28), ("governance_minimum_changed_days", 7)):
        p[key] = _integer(config, key, default, 2 if key in {"max_leaf_nodes", "min_samples_leaf", "governance_minimum_days"} else 1)
    if p["governance_minimum_days"] > p["governance_lookback_days"]:
        raise PricePolicyError("Governance minimum days exceed lookback days.")
    p["random_state"] = _integer(config, "random_state", 1729, 0)
    for key, default in (("minimum_residual_eur_mwh", 10.), ("correction_clip_eur_mwh", 100.), ("l2_regularization", 10.),
                         ("governance_minimum_gain_eur_mwh", .05), ("governance_regularization_eur_mwh", .02),
                         ("governance_uncertainty_z", 1.96), ("governance_minimum_mae_gain_eur_mwh", 0.),
                         ("signal_threshold_eur_mwh", 5.), ("transaction_cost_eur_mwh", .5), ("slippage_eur_mwh", .5)):
        p[key] = _number(config, key, default)
    p["learning_rate"] = _number(config, "learning_rate", .06, .00001, 1.)
    if p["correction_clip_eur_mwh"] <= 0:
        raise PricePolicyError("correction_clip_eur_mwh must be strictly positive.")
    weights = [float(value) for value in config.get("candidate_weights", [0., .25, .5])]
    if not weights or weights[0] != 0 or len(set(weights)) != len(weights) or len(weights) > 4 or any(not np.isfinite(value) or not 0 <= value <= 1 for value in weights):
        raise PricePolicyError("candidate_weights must start with zero and contain at most four unique fixed weights in [0,1].")
    p["candidate_weights"] = weights
    p["timezone"] = str(config.get("timezone", "Europe/Paris"))
    if p["timezone"] != "Europe/Paris":
        raise PricePolicyError("The qualified contract uses D-1 08:00 Europe/Paris civil.")
    p["cost_eur_mwh"] = p["transaction_cost_eur_mwh"] + p["slippage_eur_mwh"]
    p["hurdle_eur_mwh"] = p["signal_threshold_eur_mwh"] + p["cost_eur_mwh"]
    p["use_past_residual_features"] = config.get("use_past_residual_features", False)
    if not isinstance(p["use_past_residual_features"], bool):
        raise PricePolicyError("use_past_residual_features must be an explicit boolean.")
    return p


def _candidate_column(weight: float) -> str:
    return "candidate_forecast_weight_" + format(weight, ".12g").replace(".", "p")


def _hours(day: str) -> int:
    start = pd.Timestamp(day).tz_localize("Europe/Paris")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    return int((end.tz_convert("UTC") - start.tz_convert("UTC")).total_seconds() / 3600)


def _prepare(panel: pd.DataFrame, p: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    data = panel.copy(deep=True)
    required = {"timestamp_utc", "zone", "forecast", "actual", "reference_price", "forecast_origin_utc"}
    if required.difference(data):
        raise PricePolicyError(f"Missing columns: {sorted(required.difference(data))}.")
    data["timestamp_utc"] = _utc(data["timestamp_utc"], "timestamp_utc")
    data["forecast_origin_utc"] = _utc(data["forecast_origin_utc"], "forecast_origin_utc")
    if data[["timestamp_utc", "forecast_origin_utc", "zone"]].isna().any().any():
        raise PricePolicyError("Delivery timestamp, zone and forecast origin must be present.")
    data["zone"] = data["zone"].astype(str).str.upper()
    if data.duplicated(["timestamp_utc", "zone"]).any():
        raise PricePolicyError("Only one frozen baseline is allowed per zone/delivery interval.")
    local = data["timestamp_utc"].dt.tz_convert("Europe/Paris")
    origin = (local.dt.tz_localize(None).dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
    if not data["forecast_origin_utc"].eq(origin).all():
        raise PricePolicyError("Every historical/evaluation forecast must use D-1 08:00 civil.")
    if not local.dt.minute.eq(0).all() or not local.dt.second.eq(0).all() or not local.dt.microsecond.eq(0).all():
        raise PricePolicyError("Physical hourly delivery timestamps are required.")
    data["duration_hours"] = pd.to_numeric(data.get("duration_hours", pd.Series(1., index=data.index)), errors="raise")
    if not data["duration_hours"].eq(1).all():
        raise PricePolicyError("The residual-price POC requires one-hour physical intervals.")
    data["_day"] = local.dt.strftime("%Y-%m-%d")
    for name in ("forecast", "actual", "reference_price"):
        data[name] = pd.to_numeric(data[name], errors="raise")
    data["baseline_forecast"] = data["forecast"]
    data["_baseline_valid"] = np.isfinite(data["baseline_forecast"]) & _flag(data, "forecast_eligible", True)
    data["_label_valid"] = np.isfinite(data["actual"]) & _flag(data, "label_eligible", True)
    data["_reference_valid"] = np.isfinite(data["reference_price"]) & _flag(data, "reference_eligible", True)
    if "reference_available_at_utc" in data:
        data["_reference_valid"] &= _utc(data["reference_available_at_utc"], "reference_available_at_utc") <= data["forecast_origin_utc"]
    if "label_available_at_utc" in data:
        data["label_available_at_utc"] = _utc(data["label_available_at_utc"], "label_available_at_utc")
    else:
        data["label_available_at_utc"] = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
    if not isinstance(p.get("feature_columns"), (list, tuple)) or not p["feature_columns"]:
        raise PricePolicyError("An explicit nonempty audited feature_columns allowlist is required.")
    features = list(p["feature_columns"])
    forbidden = ("storm", "benchmark", "actual", "realised", "realized", "observed", "target", "label", "pnl", "profit", "baseline_forecast")
    if any(not isinstance(name, str) for name in features) or len(features) != len(set(features)) or any(not name.startswith("feature_") or name in METADATA or any(word in name.lower() for word in forbidden) for name in features):
        raise PricePolicyError("Only qualified fundamental feature_* columns may be selected; no targets, Storm, or externally injected baseline context.")
    for name in features:
        if name not in data:
            data[name] = np.nan
        data[name] = pd.to_numeric(data[name], errors="raise")
    data["_features_valid"] = _flag(data, "feature_eligible", False)
    if "feature_available_at_utc" in data:
        data["_features_valid"] &= _utc(data["feature_available_at_utc"], "feature_available_at_utc") <= data["forecast_origin_utc"]
    if not features:
        data["_features_valid"] = False
    else:
        data["_features_valid"] &= ~np.isinf(data[features]).any(axis=1) & data[features].notna().any(axis=1)
    core = list(p.get("required_feature_columns", []))
    if not set(core).issubset(features):
        raise PricePolicyError("Required fundamental features must belong to the explicit feature allowlist.")
    if core:
        data["_features_valid"] &= np.isfinite(data[core]).all(axis=1)
    # These contexts are derived from the given ex-ante baseline, never supplied
    # externally under arbitrary feature names and never fitted using Storm.
    data["_context_baseline_forecast"] = data["baseline_forecast"]
    data["_context_baseline_edge"] = (data["baseline_forecast"] - data["reference_price"]).where(data["_reference_valid"])
    if {"q10", "q90"}.issubset(data):
        low, high = pd.to_numeric(data["q10"], errors="raise"), pd.to_numeric(data["q90"], errors="raise")
        valid_interval = np.isfinite(low) & np.isfinite(high) & (low <= data["baseline_forecast"]) & (data["baseline_forecast"] <= high)
        data["_context_interval_width"] = (high - low).where(valid_interval)
    else:
        data["_context_interval_width"] = np.nan
    model_features = [*features, "_context_baseline_forecast", "_context_baseline_edge", "_context_interval_width"]
    data = data.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    if p["use_past_residual_features"]:
        for span in (7, 30):
            name = f"_context_known_residual_mean_{span}d"
            data[name] = np.nan
            for _, indices in data.groupby("zone", sort=False).groups.items():
                zone = data.loc[indices]
                for day, current_indices in zone.groupby("_day", sort=True).groups.items():
                    cutoff = data.loc[current_indices, "forecast_origin_utc"].iloc[0]
                    first = (pd.Timestamp(day) - pd.Timedelta(days=span)).strftime("%Y-%m-%d")
                    past = zone.loc[zone["_day"].ge(first) & zone["_day"].lt(day) & zone["_label_valid"] & zone["_baseline_valid"]
                                    & zone["label_available_at_utc"].notna() & (zone["label_available_at_utc"] <= cutoff)]
                    if len(past):
                        data.loc[current_indices, name] = float((past["actual"] - past["baseline_forecast"]).mean())
            model_features.append(name)
    return data, model_features


def _fit(zone: pd.DataFrame, day: str, cutoff: pd.Timestamp, features: list[str], p: Mapping[str, Any]) -> tuple[HistGradientBoostingRegressor | None, dict[str, Any]]:
    earliest = (pd.Timestamp(day) - pd.Timedelta(days=p["training_window_days"])).strftime("%Y-%m-%d")
    last = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    window = zone.loc[zone["_day"].ge(earliest) & zone["_day"].lt(day)].copy()
    first = max(earliest, str(zone["_day"].min()))
    calendar_days = max(0, (pd.Timestamp(day) - pd.Timestamp(first)).days)
    expected = int((pd.Timestamp(day).tz_localize("Europe/Paris").tz_convert("UTC")
                    - pd.Timestamp(first).tz_localize("Europe/Paris").tz_convert("UTC")).total_seconds() / 3600) if calendar_days else 0
    known = window["label_available_at_utc"].notna() & (window["label_available_at_utc"] <= cutoff)
    valid = (known & window["_label_valid"] & window["_baseline_valid"] & window["_features_valid"]
             & (window["forecast_origin_utc"] < window["label_available_at_utc"]))
    train = window.loc[valid]
    coverage = len(train) / expected if expected else 0.
    fold: dict[str, Any] = {
        "zone": str(zone["zone"].iloc[0]), "fit_delivery_day": day, "fit_cutoff_utc": cutoff,
        "training_start_day": first if calendar_days else None, "training_end_day": last if calendar_days else None,
        "training_days": calendar_days, "training_calendar_days_present": int(window["_day"].nunique()),
        "training_calendar_days_usable": int(train["_day"].nunique()), "minimum_training_days": p["minimum_training_days"],
        "training_window_cap_days": p["training_window_days"], "training_window_is_365_calendar_days": calendar_days == 365,
        "full_365_day_training": False,
        "training_expected_hours": expected, "training_rows": len(train), "training_coverage": coverage,
        "max_training_label_available_at_utc": train["label_available_at_utc"].max() if len(train) else pd.NaT,
        "feature_count": len(features), "features": list(features), "status": "baseline_fallback", "reason": "",
    }
    if calendar_days < p["minimum_training_days"]:
        fold["reason"] = "insufficient_training_days_warmup"
    elif window["_day"].nunique() != calendar_days:
        fold["reason"] = "nonconsecutive_training_calendar"
    elif train["_day"].nunique() != calendar_days:
        fold["reason"] = "training_calendar_has_unavailable_days"
    elif coverage < p["minimum_training_coverage"]:
        fold["reason"] = "insufficient_training_hour_coverage"
    if fold["reason"]:
        return None, fold
    regressor = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=p["max_iter"], learning_rate=p["learning_rate"],
        max_leaf_nodes=p["max_leaf_nodes"], min_samples_leaf=p["min_samples_leaf"],
        l2_regularization=p["l2_regularization"], random_state=p["random_state"], early_stopping=False,
    )
    with threadpool_limits(limits=1):
        regressor.fit(train[features].to_numpy(dtype=float), (train["actual"] - train["baseline_forecast"]).to_numpy(dtype=float))
    fold["status"] = "trained"
    fold["full_365_day_training"] = calendar_days == 365
    return regressor, fold


def _positions(forecast: pd.Series | np.ndarray, reference: pd.Series | np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
    edge = np.asarray(forecast) - np.asarray(reference)
    return np.where(np.isfinite(edge) & (np.abs(edge) > p["hurdle_eur_mwh"]), np.sign(edge), 0.)


def _net_pnl(forecast: pd.Series | np.ndarray, reference: pd.Series | np.ndarray, actual: pd.Series | np.ndarray, duration: pd.Series | np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
    positions = _positions(forecast, reference, p)
    return (positions * (np.asarray(actual) - np.asarray(reference)) - np.abs(positions) * p["cost_eur_mwh"]) * np.asarray(duration)


def _govern(oof: list[pd.DataFrame], day: str, cutoff: pd.Timestamp, zone: str, p: Mapping[str, Any]) -> tuple[float, str, list[dict[str, Any]]]:
    metadata = {"zone": zone, "delivery_day": day, "decision_cutoff_utc": cutoff}
    if not oof:
        return 0., "governance_oof_warmup", [{**metadata, "weight": 0., "oof_complete_days": 0, "reason": "governance_oof_warmup"}]
    past = pd.concat(oof, ignore_index=True)
    first = (pd.Timestamp(day) - pd.Timedelta(days=p["governance_lookback_days"])).strftime("%Y-%m-%d")
    valid = (past["_day"].ge(first) & past["_day"].lt(day) & past["expert_ready"] & past["_baseline_valid"]
             & past["_reference_valid"] & past["_label_valid"] & past["label_available_at_utc"].notna()
             & (past["label_available_at_utc"] <= cutoff) & (past["expert_available_at_utc"] < past["label_available_at_utc"]))
    past = past.loc[valid].copy()
    day_hours = past.groupby("_day")["duration_hours"].sum()
    complete = [date for date, hours in day_hours.items() if hours == _hours(date)]
    past = past.loc[past["_day"].isin(complete)]
    if len(complete) < p["governance_minimum_days"]:
        return 0., "governance_oof_warmup", [{**metadata, "weight": 0., "oof_complete_days": len(complete), "reason": "governance_oof_warmup"}]
    base_pnl = _net_pnl(past["baseline_forecast"], past["reference_price"], past["actual"], past["duration_hours"], p)
    baseline_error = np.abs(past["actual"] - past["baseline_forecast"])
    selected = 0.
    best = p["governance_minimum_gain_eur_mwh"]
    records = []
    for weight in p["candidate_weights"]:
        if weight == 0:
            continue
        candidate = past[_candidate_column(weight)]
        gain = _net_pnl(candidate, past["reference_price"], past["actual"], past["duration_hours"], p) - base_pnl
        mae_gain = (baseline_error - np.abs(past["actual"] - candidate)) * past["duration_hours"]
        table = pd.DataFrame({"day": past["_day"], "gain": gain, "mae_gain": mae_gain, "hours": past["duration_hours"],
                              "changed": ~np.isclose(candidate, past["baseline_forecast"], rtol=0., atol=1e-9)}).groupby("day").agg(
                                  gain=("gain", "sum"), mae_gain=("mae_gain", "sum"), hours=("hours", "sum"), changed=("changed", "any"))
        rates = table["gain"] / table["hours"]
        mean_gain = float(rates.mean())
        standard_error = float(rates.std(ddof=1) / np.sqrt(len(table)))
        lower = mean_gain - p["governance_uncertainty_z"] * standard_error - p["governance_regularization_eur_mwh"]
        mean_mae_gain = float(table["mae_gain"].sum() / table["hours"].sum())
        changed = int(table["changed"].sum())
        if changed < p["governance_minimum_changed_days"]:
            reason = "insufficient_changed_forecast_days"
        elif mean_mae_gain < p["governance_minimum_mae_gain_eur_mwh"]:
            reason = "mae_non_regression_failed"
        elif lower < p["governance_minimum_gain_eur_mwh"]:
            reason = "insufficient_net_economic_gain"
        else:
            reason = "qualified"
        records.append({**metadata, "weight": weight, "oof_complete_days": len(complete), "oof_first_day": min(complete), "oof_last_day": max(complete),
                        "max_oof_label_available_at_utc": past["label_available_at_utc"].max(), "changed_forecast_days": changed,
                        "mean_net_gain_eur_mwh": mean_gain, "standard_error_eur_mwh": standard_error,
                        "net_gain_lower_bound_eur_mwh": lower, "mean_mae_gain_eur_mwh": mean_mae_gain, "reason": reason})
        if reason == "qualified" and (selected == 0 or lower > best):
            selected, best = weight, lower
    for record in records:
        record["selected"] = record["weight"] == selected
    return selected, "governed_price_correction_selected" if selected else "no_candidate_satisfies_both_governance_guards", records


def run_price_policy(panel: pd.DataFrame, config: Mapping[str, Any]) -> PricePolicyResult:
    """Walk forward across a frozen baseline panel and return corrected prices.

    A short explicit warmup is diagnostic only. Setting minimum_training_days=365
    requires the full window and leaves this first 365-day panel at baseline.
    Quantile recalibration/translation and subsequent trading simulation belong
    to the dedicated caller, not to this point-price correction layer.
    """
    if panel.empty:
        raise PricePolicyError("The price-correction panel is empty.")
    p = _parameters(config)
    data, features = _prepare(panel, p)
    blocks, fold_records, governance_records = [], [], []
    for zone, zone_data in data.groupby("zone", sort=True):
        model = None
        last_attempt = None
        last_fold: dict[str, Any] = {}
        fit_id = None
        oof: list[pd.DataFrame] = []
        for day, current in zone_data.groupby("_day", sort=True):
            current = current.copy().reset_index(drop=True)
            cutoff = current["forecast_origin_utc"].iloc[0]
            if last_attempt is None or (pd.Timestamp(day) - pd.Timestamp(last_attempt)).days >= p["refit_every_days"]:
                model, last_fold = _fit(zone_data, day, cutoff, features, p)
                last_attempt = day
                fit_id = f"{zone}_{day}"
                last_fold["expert_fit_id"] = fit_id
                fold_records.append(last_fold)
                if p.get("verbose", True):
                    print(f"[PriceExpert/{zone}] fit {day}: {last_fold['status']}; past_days={last_fold['training_days']}/{p['training_window_days']}; rows={last_fold['training_rows']}; {last_fold['reason']}", flush=True)
            selected, governance_reason, records = _govern(oof, day, cutoff, zone, p)
            governance_records.extend(records)
            current["raw_residual_prediction"] = np.nan
            current["expert_ready"] = current["_features_valid"] & current["_baseline_valid"] & (model is not None)
            if model is not None and current["expert_ready"].any():
                valid = current["expert_ready"]
                with threadpool_limits(limits=1):
                    predictions = model.predict(current.loc[valid, features].to_numpy(dtype=float))
                if not np.isfinite(predictions).all():
                    raise PricePolicyError("The residual regressor produced non-finite predictions.")
                current.loc[valid, "raw_residual_prediction"] = predictions
            raw = current["raw_residual_prediction"]
            correction = np.where(current["expert_ready"] & (np.abs(raw) >= p["minimum_residual_eur_mwh"]), np.clip(raw, -p["correction_clip_eur_mwh"], p["correction_clip_eur_mwh"]), 0.)
            for weight in p["candidate_weights"]:
                current[_candidate_column(weight)] = current["baseline_forecast"] + weight * correction
            current["selected_weight"] = np.where(current["expert_ready"], selected, 0.)
            current["applied_correction"] = current["selected_weight"] * correction
            current["candidate_forecast"] = current["baseline_forecast"] + current["applied_correction"]
            current["expert_available_at_utc"] = cutoff
            current["expert_fit_id"] = fit_id
            current["fit_cutoff_utc"] = last_fold.get("fit_cutoff_utc", pd.NaT)
            current["fit_age_civil_days"] = (pd.Timestamp(day) - pd.Timestamp(last_attempt)).days
            current["training_days"] = last_fold.get("training_days", 0)
            current["full_365_day_training"] = bool(last_fold.get("full_365_day_training", False))
            current["reason"] = governance_reason
            current.loc[current["expert_ready"] & (np.abs(raw) < p["minimum_residual_eur_mwh"]), "reason"] = "predicted_residual_below_fixed_trigger"
            current.loc[~current["_features_valid"], "reason"] = "fundamental_features_unavailable_baseline_fallback"
            if model is None:
                current["reason"] = last_fold.get("reason", "training_unavailable")
            current.loc[~current["_baseline_valid"], "reason"] = "baseline_forecast_unavailable"
            if not np.all(np.abs(current["applied_correction"]) <= p["correction_clip_eur_mwh"] * max(p["candidate_weights"]) + 1e-9):
                raise PricePolicyError("Correction exceeded the fixed experiment bound.")
            blocks.append(current)
            # No expert prediction exists during warmup. Keeping such blocks in
            # the OOF buffer would only incur repeated concatenation before the
            # same expert_ready=False rows are discarded by governance.
            if current["expert_ready"].any():
                oof.append(current)
            first = (pd.Timestamp(day) - pd.Timedelta(days=p["governance_lookback_days"])).strftime("%Y-%m-%d")
            oof = [block for block in oof if block["_day"].iloc[0] >= first]
    decisions = pd.concat(blocks, ignore_index=True).sort_values(["timestamp_utc", "zone"]).reset_index(drop=True)
    folds, governance = pd.DataFrame(fold_records), pd.DataFrame(governance_records)
    trained = folds.loc[folds["status"].eq("trained")]
    audit = {
        "schema_version": 1, "diagnostic_only": True, "production_modified": False, "activation_performed": False, "orders_placed": False,
        "target": "actual_minus_frozen_nuclear_kalman_forecast", "output_kind": "corrected_price_not_position_override",
        "training_window_cap_days": p["training_window_days"], "minimum_training_days": p["minimum_training_days"],
        "actual_training_days_min": int(trained["training_days"].min()) if len(trained) else None,
        "actual_training_days_max": int(trained["training_days"].max()) if len(trained) else None,
        "all_trained_folds_have_365_days": bool(len(trained) and trained["full_365_day_training"].all()),
        "trained_folds": len(trained), "fallback_folds": int(folds["status"].ne("trained").sum()),
        "refit_every_days": p["refit_every_days"], "governance_is_daily": True, "cutoff": "D-1 08:00 Europe/Paris civil",
        "feature_columns": features, "baseline_forecast_is_ex_ante_context": True, "benchmark_used_as_feature": False,
        "benchmark_used_for_governance": False, "governance_comparator": "frozen_baseline_only",
        "governance_predictions_are_prequential_oof": True, "pre_evaluation_oof_invented": False,
        "label_availability_assumption_created_by_engine": False, "label_availability_column_supplied": "label_available_at_utc" in panel,
        "use_past_residual_features": p["use_past_residual_features"], "past_residual_features_evaluated_at_each_historical_origin": p["use_past_residual_features"],
        "quantiles_modified_by_engine": False, "annual_non_regression_guaranteed": False,
        "changed_forecast_rows": int((decisions["applied_correction"].abs() > 1e-9).sum()),
        "expert_ready_rows": int(decisions["expert_ready"].sum()), "decision_rows": len(decisions),
        "config": p,
        "limitations": [
            "The 365-day window is a maximum, not an invented amount of baseline history: each fit discloses its actual calendar span.",
            "Warmup and unavailable input hours keep the frozen baseline; they remain in the full comparison calendar.",
            "No realised value of the prediction day, no Storm forecast, and no annual evaluation statistic controls its forecast.",
            "Governance requires historical OOF MAE non-regression and a regularized net-PnL gain; neither guarantees annual or future non-regression.",
            "Baseline neural OOF and source PIT quality remain upstream audit responsibilities; this residual walk-forward does not certify them.",
            "Corrected price quantiles require separately disclosed translation or causal recalibration; this engine does not claim calibrated uncertainty.",
            "Daily-clustered uncertainty margins are conservative diagnostics, not a formal statistical guarantee after repeated candidate selection.",
        ],
    }
    return PricePolicyResult(decisions, folds, governance, audit)
