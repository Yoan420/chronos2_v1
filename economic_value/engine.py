"""Fixed-policy economic value diagnostics, isolated from operational forecasting.

Prices are EUR/MWh, positions MW, and settlement energy is MW * elapsed hours.
No parameter is fitted to the evaluation sample.  A quoted reference is only
usable when its availability timestamp is no later than the forecast origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


STRATEGIES = ("no_forecast", "benchmark", "model")
_KEYS = ["model", "zone", "strategy"]


class EconomicValueError(ValueError):
    """An input or policy contract would make the comparison ambiguous."""


@dataclass(frozen=True)
class SimulationResult:
    rows: pd.DataFrame
    metrics: pd.DataFrame
    daily: pd.DataFrame
    breakdowns: dict[str, pd.DataFrame]
    audit: dict[str, Any]


def _aware_utc(values: pd.Series, name: str) -> pd.Series:
    # utc=True alone would silently reinterpret naive local timestamps as UTC.
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert("UTC")
    for value in values.dropna():
        if pd.Timestamp(value).tzinfo is None:
            raise EconomicValueError(f"{name}: timestamp sans fuseau horaire.")
    return pd.to_datetime(values, utc=True, errors="raise", format="mixed")


def _strict_bool(values: pd.Series, name: str) -> pd.Series:
    valid = values.dropna().map(lambda x: isinstance(x, (bool, np.bool_)))
    if not valid.all():
        raise EconomicValueError(f"{name}: booleens explicites requis, pas de texte ni de 0/1.")
    return values.fillna(False).astype(bool)


def _finite_nonnegative(config: Mapping[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if not np.isfinite(value) or value < 0:
        raise EconomicValueError(f"{key}: nombre fini positif ou nul requis.")
    return value


def _calendar_hours(day: str, timezone: str) -> float:
    start = pd.Timestamp(day).tz_localize(timezone)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(timezone)
    return (end.tz_convert("UTC") - start.tz_convert("UTC")).total_seconds() / 3600


def _prepare(frame: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    data = frame.copy(deep=True)
    if "timestamp_utc" not in data and "delivery_start_utc" in data:
        data = data.rename(columns={"delivery_start_utc": "timestamp_utc"})
    required = {
        "timestamp_utc", "zone", "model", "forecast", "actual", "benchmark_forecast",
        "reference_price", "reference_available_at_utc", "forecast_origin_utc",
        "reference_eligible", "forecast_eligible", "duration_hours",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise EconomicValueError(f"Colonnes manquantes: {', '.join(missing)}.")
    if data.empty:
        raise EconomicValueError("Le panel economique est vide.")
    capacities = {str(k).upper(): float(v) for k, v in config.get("zone_capacity_mw", {}).items()}
    if not capacities or any(not np.isfinite(v) or v <= 0 for v in capacities.values()):
        raise EconomicValueError("zone_capacity_mw doit allouer explicitement une puissance positive par pays.")
    cap_limit = _finite_nonnegative(config, "portfolio_capacity_mw", 100.0)
    if sum(capacities.values()) > cap_limit + 1e-9:
        raise EconomicValueError("Les allocations depassent la puissance totale du portefeuille.")
    data["zone"] = data["zone"].astype(str).str.upper()
    data["model"] = data["model"].astype(str)
    governed = config.get("governed_models", [])
    if not isinstance(governed, list) or len(set(governed)) != len(governed) or set(governed) - set(data["model"]):
        raise EconomicValueError("governed_models doit nommer explicitement les alternatives presentes.")
    if "policy_position_fraction" in data and not governed:
        raise EconomicValueError("Positions externes interdites sans governed_models explicite.")
    if "PORTFOLIO" in capacities or not set(data["zone"]).issubset(capacities):
        raise EconomicValueError("Pays non alloue ou nom reserve PORTFOLIO.")
    for col in ("timestamp_utc", "reference_available_at_utc", "forecast_origin_utc"):
        data[col] = _aware_utc(data[col], col)
    if governed:
        selected = data["model"].isin(governed)
        if not {"policy_position_fraction", "policy_available_at_utc"}.issubset(data):
            raise EconomicValueError("Positions gouvernees et date de disponibilite obligatoires.")
        fractions = pd.to_numeric(data.loc[selected, "policy_position_fraction"], errors="raise")
        if (~np.isfinite(fractions) | (fractions.abs() > 1)).any():
            raise EconomicValueError("Fraction de position gouvernee hors limite [-1, 1].")
        available = _aware_utc(data.loc[selected, "policy_available_at_utc"], "policy_available_at_utc")
        if (available.isna() | (available > data.loc[selected, "forecast_origin_utc"])).any():
            raise EconomicValueError("Decision gouvernee non disponible au cutoff.")
        data.loc[selected, "policy_position_fraction"] = fractions
    if data["timestamp_utc"].isna().any():
        raise EconomicValueError("Horodatage de livraison manquant.")
    if data.duplicated(["timestamp_utc", "zone", "model"]).any():
        raise EconomicValueError("Doublon livraison/pays/modele: un intervalle ne peut etre negocie deux fois.")
    for col in ("forecast", "actual", "benchmark_forecast", "reference_price", "duration_hours"):
        data[col] = pd.to_numeric(data[col], errors="raise")
    for col in ("q10", "q90", "benchmark_q10", "benchmark_q90"):
        if col not in data:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="raise")
    duration = data["duration_hours"]
    if (~np.isfinite(duration) | (duration <= 0) | (duration > 24)).any():
        raise EconomicValueError("duration_hours doit etre explicite, finie et comprise entre 0 et 24 heures.")
    for _, group in data.groupby(["model", "zone"], sort=False):
        ordered = group.sort_values("timestamp_utc")
        ends = ordered["timestamp_utc"] + pd.to_timedelta(ordered["duration_hours"], unit="h")
        if (ends.iloc[:-1].to_numpy() > ordered["timestamp_utc"].iloc[1:].to_numpy()).any():
            raise EconomicValueError("Intervalles de livraison qui se chevauchent.")
    for col in ("reference_eligible", "forecast_eligible"):
        data[col] = _strict_bool(data[col], col)
    if "benchmark_eligible" not in data:
        # This flag may be supplied by the reader; finite benchmark alone is not
        # independent evidence of when a revised benchmark became available.
        data["benchmark_eligible"] = data["forecast_eligible"]
    else:
        data["benchmark_eligible"] = _strict_bool(data["benchmark_eligible"], "benchmark_eligible")
    data["sample"] = data.get("sample", pd.Series("evaluation", index=data.index)).astype(str)
    if not data["sample"].isin(["evaluation", "live"]).all():
        raise EconomicValueError("sample doit valoir evaluation ou live.")
    timezone = str(config.get("timezone", "Europe/Paris"))
    local = data["timestamp_utc"].dt.tz_convert(timezone)
    data["delivery_day"] = local.dt.strftime("%Y-%m-%d")
    data["delivery_hour"] = local.dt.hour
    data["delivery_month"] = local.dt.strftime("%Y-%m")
    data["season"] = np.where(local.dt.month.isin([10, 11, 12, 1, 2, 3]), "winter", "summer")
    # This lab currently scores intervals contained within one civil day. DST
    # duplicate hours remain distinct UTC intervals and are never deduplicated.
    last_instants = data["timestamp_utc"] + pd.to_timedelta(duration, unit="h") - pd.Timedelta(nanoseconds=1)
    if (last_instants.dt.tz_convert(timezone).dt.strftime("%Y-%m-%d") != data["delivery_day"]).any():
        raise EconomicValueError("Un intervalle traverse minuit local; le decouper avant evaluation.")
    data["allocated_capacity_mw"] = data["zone"].map(capacities)
    return data.sort_values(["model", "timestamp_utc", "zone"]).reset_index(drop=True), capacities


def _confidence(
    forecast: pd.Series, low: pd.Series, high: pd.Series, reference: pd.Series,
    hurdle: float, config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(low) & np.isfinite(high) & np.isfinite(forecast) & (low <= forecast) & (forecast <= high)
    edge = forecast - reference
    width = high - low
    ratio = np.divide(np.abs(edge), width, out=np.full(len(edge), np.nan), where=width.to_numpy() > 0)
    middle = _finite_nonnegative(config, "confidence_medium_edge_width_ratio", 0.25)
    high_ratio = _finite_nonnegative(config, "confidence_high_edge_width_ratio", 0.5)
    if high_ratio < middle:
        raise EconomicValueError("Le seuil de confiance high doit etre >= au seuil medium.")
    excludes = ((edge > 0) & (low > reference + hurdle)) | ((edge < 0) & (high < reference - hurdle))
    confidence = np.full(len(edge), "unknown", dtype=object)
    confidence[valid] = "low"
    confidence[valid & (ratio >= middle)] = "medium"
    confidence[valid & excludes & ((ratio >= high_ratio) | (width == 0))] = "high"
    return confidence, valid.to_numpy()


def _make_rows(data: pd.DataFrame, config: Mapping[str, Any], capacities: Mapping[str, float]) -> pd.DataFrame:
    cost = _finite_nonnegative(config, "transaction_cost_eur_mwh", 0.0)
    slippage = _finite_nonnegative(config, "slippage_eur_mwh", 0.0)
    threshold = _finite_nonnegative(config, "signal_threshold_eur_mwh", 0.0)
    hurdle = threshold + cost + slippage
    policy = str(config.get("confidence_filter", "none"))
    if policy not in ("none", "high", "medium_or_high"):
        raise EconomicValueError("confidence_filter accepte none, high ou medium_or_high.")
    observed = np.isfinite(data["actual"])
    known_reference = (
        data["reference_available_at_utc"].notna() & data["forecast_origin_utc"].notna()
        & (data["reference_available_at_utc"] <= data["forecast_origin_utc"])
    )
    before_delivery = data["forecast_origin_utc"].notna() & (data["forecast_origin_utc"] < data["timestamp_utc"])
    exante = (
        np.isfinite(data["reference_price"]) & np.isfinite(data["forecast"])
        & np.isfinite(data["benchmark_forecast"]) & known_reference & before_delivery
        & data["reference_eligible"] & data["forecast_eligible"] & data["benchmark_eligible"]
    )
    model_conf, model_interval = _confidence(data["forecast"], data["q10"], data["q90"], data["reference_price"], hurdle, config)
    benchmark_conf, benchmark_interval = _confidence(data["benchmark_forecast"], data["benchmark_q10"], data["benchmark_q90"], data["reference_price"], hurdle, config)
    if policy != "none":
        # Missing benchmark uncertainty may not silently suppress only Strategy1.
        exante &= model_interval & benchmark_interval
    data = data.copy()
    data["candidate_confidence"] = model_conf
    data["signal_eligible"] = exante
    data["paired_eligible"] = exante & observed & data["sample"].eq("evaluation")
    data["evaluation_status"] = np.select(
        [~exante, data["sample"].eq("live"), ~observed],
        ["ineligible", "live", "pending_observation"], default="evaluated",
    )
    data["ineligibility_reason"] = np.select(
        [~np.isfinite(data["reference_price"]), ~known_reference, ~before_delivery,
         ~data["reference_eligible"], ~data["forecast_eligible"], ~data["benchmark_eligible"],
         ~np.isfinite(data["forecast"]), ~np.isfinite(data["benchmark_forecast"]),
         (policy != "none") & ~(model_interval & benchmark_interval)],
        ["reference_missing", "reference_not_known_at_origin", "forecast_not_before_delivery",
         "reference_not_qualified", "forecast_not_qualified", "benchmark_not_qualified",
         "model_forecast_missing", "benchmark_forecast_missing", "symmetric_uncertainty_missing"],
        default="",
    )
    paired = data.groupby(["model", "sample", "timestamp_utc"], sort=False)
    complete = paired["paired_eligible"].transform("sum").eq(len(capacities))
    complete &= paired["zone"].transform("nunique").eq(len(capacities))
    complete &= np.isclose(paired["duration_hours"].transform("min"), paired["duration_hours"].transform("max"))
    data["portfolio_eligible"] = complete
    upper = float(config.get("spike_high_eur_mwh", 200.0))
    lower = float(config.get("spike_low_eur_mwh", 0.0))
    if not np.isfinite(upper) or not np.isfinite(lower) or lower >= upper:
        raise EconomicValueError("Seuils de spikes invalides.")
    data["spike"] = np.select(
        [~observed, data["actual"] >= upper, data["actual"] <= lower],
        ["unknown", "high", "low"], default="normal",
    )
    if config.get("spike_absolute_move_eur_mwh") is not None:
        move = _finite_nonnegative(config, "spike_absolute_move_eur_mwh", 100.0)
        large = observed & (np.abs(data["actual"] - data["reference_price"]) >= move) & data["spike"].eq("normal")
        data.loc[large, "spike"] = "large_reference_move"
    data["model_forecast"] = data["forecast"]
    result = []
    for strategy in STRATEGIES:
        rows = data.copy()
        rows["strategy"] = strategy
        if strategy == "model":
            conf, valid_interval = model_conf, model_interval
        elif strategy == "benchmark":
            rows["forecast"] = rows["benchmark_forecast"]
            rows["q10"], rows["q90"] = rows["benchmark_q10"], rows["benchmark_q90"]
            conf, valid_interval = benchmark_conf, benchmark_interval
        else:
            rows[["forecast", "q10", "q90"]] = np.nan
            conf, valid_interval = np.full(len(rows), "unknown", dtype=object), np.zeros(len(rows), dtype=bool)
        rows["confidence"] = conf
        rows["valid_quantile_interval"] = valid_interval
        rows["edge_eur_mwh"] = rows["forecast"] - rows["reference_price"]
        active = rows["signal_eligible"] & (np.abs(rows["edge_eur_mwh"]) > hurdle)
        if policy == "high":
            active &= rows["confidence"].eq("high")
        elif policy == "medium_or_high":
            active &= rows["confidence"].isin(["medium", "high"])
        rows["position_mw"] = np.where(active, np.sign(rows["edge_eur_mwh"]) * rows["allocated_capacity_mw"], 0.0)
        if strategy == "model" and config.get("governed_models"):
            governed = rows["model"].isin(config["governed_models"])
            rows.loc[governed, "position_mw"] = np.where(
                rows.loc[governed, "signal_eligible"],
                rows.loc[governed, "policy_position_fraction"] * rows.loc[governed, "allocated_capacity_mw"], 0.0,
            )
            active = rows["signal_eligible"] & rows["position_mw"].ne(0)
        rows["signal"] = np.select([rows["position_mw"] > 0, rows["position_mw"] < 0], ["BUY", "SELL"], default="FLAT")
        rows["energy_mwh"] = rows["position_mw"] * rows["duration_hours"]
        rows["absolute_energy_mwh"] = np.abs(rows["energy_mwh"])
        rows["pnl_gross_eur"] = rows["energy_mwh"] * (rows["actual"] - rows["reference_price"])
        rows["trading_cost_eur"] = rows["absolute_energy_mwh"] * (cost + slippage)
        rows["pnl_net_eur"] = rows["pnl_gross_eur"] - rows["trading_cost_eur"]
        rows.loc[~rows["paired_eligible"], ["pnl_gross_eur", "trading_cost_eur", "pnl_net_eur"]] = np.nan
        realised_edge = rows["actual"] - rows["reference_price"]
        rows["active_trade"] = active & rows["paired_eligible"]
        rows["direction_tie"] = rows["active_trade"] & realised_edge.eq(0)
        rows["direction_hit"] = rows["active_trade"] & (np.sign(rows["position_mw"]) == np.sign(realised_edge)) & ~rows["direction_tie"]
        rows["winning_trade"] = rows["active_trade"] & (rows["pnl_net_eur"] > 0)
        rows["interval_covered"] = rows["valid_quantile_interval"] & (rows["actual"] >= rows["q10"]) & (rows["actual"] <= rows["q90"])
        result.append(rows)
    return pd.concat(result, ignore_index=True).sort_values(["model", "timestamp_utc", "zone", "strategy"]).reset_index(drop=True)


def _sum_or_nan(values: pd.Series) -> float:
    return float(values.sum(min_count=1))


def _aggregate_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    eligible = rows.loc[rows["paired_eligible"]]
    active = eligible.loc[eligible["active_trade"]]
    interval = eligible.loc[eligible["valid_quantile_interval"]]
    pnl = _sum_or_nan(eligible["pnl_net_eur"])
    energy = float(eligible["absolute_energy_mwh"].sum())
    return {
        "pnl_net_eur": pnl,
        "pnl_gross_eur": _sum_or_nan(eligible["pnl_gross_eur"]),
        "trading_cost_eur": _sum_or_nan(eligible["trading_cost_eur"]),
        "absolute_energy_mwh": energy,
        "pnl_per_mwh": pnl / energy if energy > 0 else np.nan,
        "eligible_hours": float(eligible["duration_hours"].sum()),
        "active_hours": float(active["duration_hours"].sum()),
        "eligible_intervals": len(eligible),
        "active_intervals": len(active),
        "direction_ties": int(active["direction_tie"].sum()),
        "hit_ratio_directional": float(active["direction_hit"].mean()) if len(active) else np.nan,
        "winning_trade_ratio": float(active["winning_trade"].mean()) if len(active) else np.nan,
        "quantile_coverage": float(interval["interval_covered"].mean()) if len(interval) else np.nan,
        "quantile_scored_intervals": len(interval),
        "mean_quantile_width_eur_mwh": float((interval["q90"] - interval["q10"]).mean()) if len(interval) else np.nan,
        "mean_pnl_per_interval_eur": float(eligible["pnl_net_eur"].mean()) if len(eligible) else np.nan,
    }


def _drawdown(pnl: pd.Series, capital: float | None) -> tuple[float, float]:
    if not len(pnl.dropna()):
        return np.nan, np.nan
    equity = np.r_[0.0, pnl.dropna().to_numpy(dtype=float).cumsum()]
    running_peak = np.maximum.accumulate(equity)
    absolute = float(np.max(running_peak - equity))
    if capital is None:
        return absolute, np.nan
    wealth = capital + equity
    wealth_peak = np.maximum.accumulate(wealth)
    return absolute, float(np.max((wealth_peak - wealth) / wealth_peak) * 100.0)


def _daily_table(current: pd.DataFrame, calendar: list[str], day_hours: Mapping[str, float], countries: int) -> pd.DataFrame:
    """Vectorised daily aggregation: no repeated 8,760-row scans per day."""
    if not calendar:
        return pd.DataFrame()
    eligible = current["paired_eligible"]
    active = eligible & current["active_trade"]
    interval = eligible & current["valid_quantile_interval"]
    values = pd.DataFrame({
        "delivery_day": current["delivery_day"],
        "pnl_net_eur": current["pnl_net_eur"].where(eligible),
        "pnl_gross_eur": current["pnl_gross_eur"].where(eligible),
        "trading_cost_eur": current["trading_cost_eur"].where(eligible),
        "absolute_energy_mwh": current["absolute_energy_mwh"].where(eligible, 0),
        "eligible_hours": current["duration_hours"].where(eligible, 0),
        "active_hours": current["duration_hours"].where(active, 0),
        "eligible_intervals": eligible.astype(int),
        "active_intervals": active.astype(int),
        "direction_ties": (active & current["direction_tie"]).astype(int),
        "direction_hits": (active & current["direction_hit"]).astype(int),
        "winning_trades": (active & current["winning_trade"]).astype(int),
        "quantile_scored_intervals": interval.astype(int),
        "covered_intervals": (interval & current["interval_covered"]).astype(int),
        "quantile_width": (current["q90"] - current["q10"]).where(interval, 0),
    })
    result = values.groupby("delivery_day", sort=True).sum(min_count=1).reindex(calendar)
    pnl_columns = ["pnl_net_eur", "pnl_gross_eur", "trading_cost_eur"]
    count_columns = [column for column in result.columns if column not in pnl_columns]
    result[count_columns] = result[count_columns].fillna(0)
    energy = result["absolute_energy_mwh"].where(result["absolute_energy_mwh"] > 0)
    trade_count = result["active_intervals"].where(result["active_intervals"] > 0)
    interval_count = result["quantile_scored_intervals"].where(result["quantile_scored_intervals"] > 0)
    result["pnl_per_mwh"] = result["pnl_net_eur"] / energy
    result["hit_ratio_directional"] = result["direction_hits"] / trade_count
    result["winning_trade_ratio"] = result["winning_trades"] / trade_count
    result["quantile_coverage"] = result["covered_intervals"] / interval_count
    result["mean_quantile_width_eur_mwh"] = result["quantile_width"] / interval_count
    result["mean_pnl_per_interval_eur"] = result["pnl_net_eur"] / result["eligible_intervals"].where(result["eligible_intervals"] > 0)
    result["expected_hours"] = pd.Series(day_hours) * countries
    result["complete_day"] = np.isclose(result["eligible_hours"], result["expected_hours"], atol=1e-8, rtol=0)
    return result.rename_axis("delivery_day").reset_index()


def simulate(frame: pd.DataFrame, config: Mapping[str, Any]) -> SimulationResult:
    """Score three symmetric fixed strategies for each *alternative* model.

    ``sample='live'`` rows expose prospective signals, never realised PnL in
    aggregates.  Country scores use their paired samples; portfolio scores need
    every configured country at an interval and never recycle missing capacity.
    The model alternatives are separate counterfactual portfolios, not combined.
    """
    data, capacities = _prepare(frame, config)
    rows = _make_rows(data, config, capacities)
    evaluation = rows.loc[rows["sample"].eq("evaluation")].copy()
    timezone = str(config.get("timezone", "Europe/Paris"))
    if evaluation.empty:
        calendar: list[str] = []
    else:
        first = str(config.get("evaluation_start_day", evaluation["delivery_day"].min()))
        last = str(config.get("evaluation_end_day", evaluation["delivery_day"].max()))
        if first > last or (evaluation["delivery_day"] < first).any() or (evaluation["delivery_day"] > last).any():
            raise EconomicValueError("Les lignes d'evaluation sortent de la fenetre declaree.")
        calendar = pd.date_range(first, last, freq="D").strftime("%Y-%m-%d").tolist()
    day_hours = {day: _calendar_hours(day, timezone) for day in calendar}
    full_hours = float(sum(day_hours.values()))
    capital = config.get("initial_capital_eur")
    if capital is not None and (not np.isfinite(float(capital)) or float(capital) <= 0):
        raise EconomicValueError("initial_capital_eur doit etre strictement positif ou absent.")
    capital = None if capital is None else float(capital)
    daily_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    breakdown_records: dict[str, list[dict[str, Any]]] = {key: [] for key in ("hour", "season", "spike", "confidence", "monthly")}
    grouping = {"hour": "delivery_hour", "season": "season", "spike": "spike", "confidence": "candidate_confidence", "monthly": "delivery_month"}
    for model in sorted(data["model"].unique()):
        model_rows = evaluation.loc[evaluation["model"].eq(model)]
        for zone in [*capacities, "PORTFOLIO"]:
            countries = len(capacities) if zone == "PORTFOLIO" else 1
            capacity = sum(capacities.values()) if zone == "PORTFOLIO" else capacities[zone]
            if zone == "PORTFOLIO":
                selected = model_rows.copy()
                selected["paired_eligible"] &= selected["portfolio_eligible"]
                selected.loc[~selected["paired_eligible"], ["pnl_net_eur", "pnl_gross_eur", "trading_cost_eur"]] = np.nan
            else:
                selected = model_rows.loc[model_rows["zone"].eq(zone)]
            for strategy in STRATEGIES:
                current = selected.loc[selected["strategy"].eq(strategy)]
                keys = {"model": model, "zone": zone, "strategy": strategy}
                series = _daily_table(current, calendar, day_hours, countries)
                if not series.empty:
                    for key, value in keys.items():
                        series[key] = value
                    # Missing days remain explicit NaN gaps, never invented zero
                    # returns. The observed-sample equity resumes after the gap.
                    series["cumulative_pnl_eur"] = series["pnl_net_eur"].cumsum()
                    series["equity_is_complete_to_date"] = series["complete_day"].cummin()
                    daily_records.extend(series.to_dict("records"))
                metric = {**keys, **_aggregate_metrics(current)}
                expected = full_hours * countries
                metric["total_hours"] = float(current["duration_hours"].sum())
                metric["expected_hours"] = expected
                metric["coverage_ratio"] = metric["eligible_hours"] / expected if expected else np.nan
                metric["allocated_capacity_mw"] = capacity
                metric["calendar_days"] = len(calendar)
                metric["annual_window"] = len(calendar) == 365
                metric["annual_fully_observed"] = bool(len(calendar) == 365 and not series.empty and series["complete_day"].all())
                metric["pnl_per_mw_year"] = metric["pnl_net_eur"] / capacity if metric["annual_fully_observed"] else np.nan
                complete_daily = series.loc[series["complete_day"], "pnl_net_eur"] if not series.empty else pd.Series(dtype=float)
                std = float(complete_daily.std(ddof=1))
                metric["sharpe_daily_pnl"] = float(np.sqrt(365) * complete_daily.mean() / std) if len(complete_daily) > 1 and np.isfinite(std) and std > 0 else np.nan
                metric["sharpe_complete_days"] = len(complete_daily)
                pnl_series = series["pnl_net_eur"] if not series.empty else pd.Series(dtype=float)
                # Capital is allocated by the same fixed capacities as exposure.
                allocated_capital = None if capital is None else capital * capacity / sum(capacities.values())
                metric["max_drawdown_eur"], metric["max_drawdown_percent"] = _drawdown(pnl_series, allocated_capital)
                metric["drawdown_sample_complete"] = bool(not series.empty and series["complete_day"].all())
                metric_records.append(metric)
                for name, column in grouping.items():
                    for group, group_rows in current.groupby(column, dropna=False, sort=True):
                        breakdown_records[name].append({**keys, "group": group, **_aggregate_metrics(group_rows)})
    metrics = pd.DataFrame(metric_records)
    daily = pd.DataFrame(daily_records)
    if not metrics.empty:
        benchmark = metrics.loc[metrics["strategy"].eq("benchmark"), ["model", "zone", "pnl_net_eur"]].rename(columns={"pnl_net_eur": "benchmark_pnl_net_eur"})
        metrics = metrics.merge(benchmark, on=["model", "zone"], how="left", validate="many_to_one")
        metrics["economic_value_added_eur"] = metrics["pnl_net_eur"] - metrics["benchmark_pnl_net_eur"]
        metrics["economic_value_added_per_allocated_mw"] = metrics["economic_value_added_eur"] / metrics["allocated_capacity_mw"]
        metrics["economic_value_added_per_mw_year"] = np.where(metrics["annual_fully_observed"], metrics["economic_value_added_eur"] / metrics["allocated_capacity_mw"], np.nan)
    if not daily.empty:
        benchmark_daily = daily.loc[daily["strategy"].eq("benchmark"), ["model", "zone", "delivery_day", "pnl_net_eur"]].rename(columns={"pnl_net_eur": "benchmark_pnl_net_eur"})
        daily = daily.merge(benchmark_daily, on=["model", "zone", "delivery_day"], how="left", validate="many_to_one")
        daily["economic_value_added_eur"] = daily["pnl_net_eur"] - daily["benchmark_pnl_net_eur"]
    breakdowns = {name: pd.DataFrame(records) for name, records in breakdown_records.items()}
    for table in breakdowns.values():
        if table.empty:
            continue
        benchmark = table.loc[table["strategy"].eq("benchmark"), ["model", "zone", "group", "pnl_net_eur"]].rename(columns={"pnl_net_eur": "benchmark_pnl_net_eur"})
        # All strategies use the candidate's ex-ante confidence bucket, hence
        # the same hours even when the benchmark has no probabilistic forecast.
        table["economic_value_added_eur"] = table["pnl_net_eur"].to_numpy() - table[["model", "zone", "group"]].merge(benchmark, on=["model", "zone", "group"], how="left", validate="many_to_one")["benchmark_pnl_net_eur"].to_numpy()
    audit = {
        "schema_version": 1, "fixed_policy": not bool(config.get("governed_models")),
        "parameters_fitted_on_evaluation": bool(config.get("governed_models")),
        "governed_models": config.get("governed_models", []),
        "policy_note": "External governed decisions use historical observations only; inspect their training/fold audit." if config.get("governed_models") else "Fixed symmetric forecast-to-position rule.",
        "forecast_pipeline_modified": False, "timezone": timezone,
        "evaluation_start_day": calendar[0] if calendar else None,
        "evaluation_end_day": calendar[-1] if calendar else None,
        "evaluation_calendar_days": len(calendar), "live_rows_excluded_from_metrics": int(data["sample"].eq("live").sum()),
        "strategies": list(STRATEGIES), "zone_capacity_mw": capacities,
        "portfolio_capacity_mw": float(sum(capacities.values())),
        "model_alternatives_are_not_cumulative": True,
        "portfolio_requires_all_allocated_zones_at_each_interval": True,
        "paired_input_rows": int(rows.loc[rows["strategy"].eq("model"), "paired_eligible"].sum()),
        "ineligible_reasons": rows.loc[rows["strategy"].eq("model") & rows["evaluation_status"].eq("ineligible"), "ineligibility_reason"].value_counts().to_dict(),
        "signal_threshold_including_costs_eur_mwh": float(config.get("signal_threshold_eur_mwh", 0)) + float(config.get("transaction_cost_eur_mwh", 0)) + float(config.get("slippage_eur_mwh", 0)),
        "confidence_is_probability": False, "quantile_nominal_coverage": 0.8,
        "confidence_breakdown_basis": "candidate_confidence, identical timestamps for all strategies",
        "sharpe_definition": "sqrt(365) * mean(complete daily net PnL) / sample_std(ddof=1); diagnostic PnL ratio, not invested-capital return Sharpe",
        "drawdown_definition": "Chronological daily cumulative matched-sample net PnL, including initial equity 0; missing observations remain gaps",
        "spikes_are_ex_post_analysis_only": True,
        "winter_months": [10, 11, 12, 1, 2, 3], "summer_months": [4, 5, 6, 7, 8, 9],
        "limitations": [
            "Economic Value Added here means incremental simulated net PnL versus the benchmark, not corporate EVA after capital charges.",
            "A non-executable reference price supports a diagnostic signal experiment only, not a tradable realised PnL claim.",
            "Signals are fixed MW positions; no execution, liquidity, market impact, funding, collateral, or sourcing obligation is modelled beyond configured linear costs.",
            "BUY and SELL are hypothetical relative-value positions, not trade recommendations.",
            "Confidence is an ex-ante interval/edge heuristic, not a calibrated probability of profit.",
            "Daily PnL Sharpe is descriptive; serial dependence and tail risks invalidate automatic significance or conventional return-Sharpe interpretations.",
            "Exactly 365 complete paired civil days are required for EUR/MW/year claims; partial samples are not annualised.",
        ],
    }
    return SimulationResult(rows=rows, metrics=metrics, daily=daily, breakdowns=breakdowns, audit=audit)
