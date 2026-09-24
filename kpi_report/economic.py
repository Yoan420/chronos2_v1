"""Fixed-policy, non-executable previous-day economic diagnostics for KPI.

The original economic lab's policy and execution arithmetic are reused without
fitting anything. These are alternative portfolios, not cumulative positions.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from economic_value.engine import _aware_utc, _make_rows, _prepare, _strict_bool
from economic_value.runner import engine_config, load_config, validate_config
from .metrics import ALL_ZONE, RESERVED_MODELS, STORM_ID, KPIError, _references, _selection, _validate


DEFAULT_CONFIG = Path(__file__).resolve().parents[1]/"config"/"economic_value.yaml"
KEYS = ["zone", "timestamp_utc"]


def _policy(config: Mapping | None, config_path: str | Path | None) -> tuple[dict, dict]:
    source = Path(config_path).resolve() if config_path is not None else DEFAULT_CONFIG
    if config is None:
        raw = source.read_bytes()
        settings = load_config(source)
        if source.read_bytes() != raw:
            raise KPIError("Economic configuration changed while being read.")
        source_audit = {"source_config_path": str(source), "source_config_sha256": hashlib.sha256(raw).hexdigest()}
    else:
        settings = deepcopy(dict(config))
        validate_config(settings)
        source_audit = {"source_config_path": None, "source_config_sha256": None}
        if config_path is not None:
            raw = source.read_bytes()
            if load_config(source) != settings or source.read_bytes() != raw:
                raise KPIError("Supplied economic settings differ from their source file.")
            source_audit = {"source_config_path": str(source), "source_config_sha256": hashlib.sha256(raw).hexdigest()}
    if settings["strategy"].get("governed_models"):
        raise KPIError("KPI economic diagnostics only accept the fixed original policy.")
    parameter_sha = hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return settings, {**source_audit, "parameters_sha256": parameter_sha}


def _origin(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    local = index.tz_convert(timezone).tz_localize(None)
    return (local.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize(timezone).tz_convert("UTC")


def _lagged_references(references: pd.DataFrame, keys: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Mirror the original proxy's civil-hour/DST rule, without reading caches."""
    output = []
    for zone, block in keys.groupby("zone", sort=True):
        target = references.xs(zone, level="zone")["actual"]
        local = target.index.tz_convert(timezone).tz_localize(None)
        counts = pd.Series(1, index=local).groupby(level=0).sum()
        unique = ~local.duplicated(keep=False)
        lookup = pd.Series(target.to_numpy()[unique], index=local[unique])
        timestamp_lookup = pd.Series(target.index[unique], index=local[unique])
        source_local = pd.DatetimeIndex(block.timestamp_utc).tz_convert(timezone).tz_localize(None)-pd.Timedelta(days=1)
        values = lookup.reindex(source_local).to_numpy(float)
        available = (source_local.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=18)).tz_localize(timezone).tz_convert("UTC")
        current = block.copy()
        current["reference_price"] = values
        current["reference_source_timestamp_utc"] = pd.to_datetime(timestamp_lookup.reindex(source_local).to_numpy(), utc=True)
        current["reference_available_at_utc"] = available
        current["reference_eligible"] = np.isfinite(values)
        current["reference_missing_reason"] = np.where(
            np.isfinite(values), "", np.where(counts.reindex(source_local).fillna(0).to_numpy() > 1,
                                              "ambiguous_previous_civil_hour", "missing_previous_civil_hour"))
        output.append(current)
    if not output:
        return keys.assign(reference_price=np.nan, reference_source_timestamp_utc=pd.NaT,
                           reference_available_at_utc=pd.NaT, reference_eligible=False, reference_missing_reason="")
    return pd.concat(output, ignore_index=True)


def _row(zone: str, model: str, rows: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    result = {"zone": zone, "model_id": model, "status": "no_common_economic_support",
              "n_hours": 0, "n_country_hours": 0, "potential_energy_mwh": 0.,
              "pnl_net_eur": None, "storm_pnl_net_eur": None, "gain_vs_storm_eur": None,
              "gain_vs_storm_per_potential_mwh": None, "pnl_per_potential_mwh": None,
              "no_forecast_pnl_eur": None, "active_intervals": 0, "buy_intervals": 0,
              "sell_intervals": 0, "absolute_energy_mwh": 0.}
    if rows.empty:
        return result
    if len(rows) != len(benchmark) or set(map(tuple, rows[KEYS].to_numpy())) != set(map(tuple, benchmark[KEYS].to_numpy())):
        raise KPIError("Economic model and Storm support differ.")
    pnl = float(rows.pnl_net_eur.sum())
    storm_pnl = float(benchmark.pnl_net_eur.sum())
    denominator = float((rows.allocated_capacity_mw*rows.duration_hours).sum())
    result.update({"status": "ok", "n_hours": int(rows.timestamp_utc.nunique()), "n_country_hours": int(len(rows)),
                   "potential_energy_mwh": denominator, "pnl_net_eur": pnl, "storm_pnl_net_eur": storm_pnl,
                   "gain_vs_storm_eur": pnl-storm_pnl,
                   "gain_vs_storm_per_potential_mwh": (pnl-storm_pnl)/denominator,
                   "pnl_per_potential_mwh": pnl/denominator, "no_forecast_pnl_eur": 0.,
                   "active_intervals": int(rows.position_mw.ne(0).sum()),
                   "buy_intervals": int(rows.position_mw.gt(0).sum()),
                   "sell_intervals": int(rows.position_mw.lt(0).sum()),
                   "absolute_energy_mwh": float(rows.absolute_energy_mwh.sum())})
    return result


def compute_economic_kpis(
    frame: pd.DataFrame, *, end_day: str, days: int = 365,
    models: list[str] | None = None, zones: list[str] | None = None,
    config: Mapping | None = None, config_path: str | Path | None = None,
) -> dict:
    """Compare net simulated PnL under the already declared fixed EVA policy.

    Source observations before the scored window remain available solely for
    D-1 reference construction. A first day without that history is excluded,
    never replaced by the same-day target. ALL needs all selected countries at
    each physical hour. Country allocations always use ALL configured countries,
    even if the caller only selects FR; 100 MW is never reassigned to France.
    """
    settings, config_audit = _policy(config, config_path)
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise KPIError("days must be a positive integer.")
    try:
        end = date.fromisoformat(end_day)
    except (ValueError, TypeError) as exc:
        raise KPIError("end_day must be an ISO civil date.") from exc
    if end.isoformat() != end_day:
        raise KPIError("end_day must be an ISO civil date.")
    start = end-timedelta(days=days-1)
    timezone = settings["timezone"]
    lower = pd.Timestamp(start, tz=timezone).tz_convert("UTC")
    upper = pd.Timestamp(end+timedelta(days=1), tz=timezone).tz_convert("UTC")
    data = _validate(frame)
    selected_models = _selection(models, data.model_id, "models")
    selected_zones = _selection(zones, data.zone, "zones")
    if RESERVED_MODELS.intersection(selected_models) or ALL_ZONE in selected_zones:
        raise KPIError("Reserved model or zone identifier.")
    if set(selected_zones)-set(settings["zones"]):
        raise KPIError("Economic countries must have an explicit existing fixed allocation.")
    capacities = {zone: float(settings["portfolio"]["capacity_mw"])/len(settings["zones"]) for zone in settings["zones"]}
    policy = engine_config(settings)
    policy["zone_capacity_mw"] = {zone: capacities[zone] for zone in selected_zones}
    policy.update(evaluation_start_day=start.isoformat(), evaluation_end_day=end.isoformat())
    extra = [name for name in ("forecast_origin_utc", "forecast_eligible", "benchmark_eligible", "sample") if name in frame]
    if extra:
        meta = frame[["model_id", *KEYS, *extra]].copy()
        meta["timestamp_utc"] = _aware_utc(meta.timestamp_utc, "timestamp_utc")
        if "forecast_origin_utc" in meta:
            meta["forecast_origin_utc"] = _aware_utc(meta.forecast_origin_utc, "forecast_origin_utc")
        for name in ("forecast_eligible", "benchmark_eligible"):
            if name in meta:
                meta[name] = _strict_bool(meta[name], name)
        for name in extra:
            if meta.groupby(["model_id", *KEYS])[name].nunique(dropna=False).gt(1).any():
                raise KPIError(f"Conflicting duplicate economic eligibility: {name}.")
        data = data.merge(meta.drop_duplicates(["model_id", *KEYS]), on=["model_id", *KEYS], validate="one_to_one")
    data = data.loc[data.model_id.isin(selected_models) & data.zone.isin(selected_zones)].copy()
    refs = _references(data)
    scored = data.loc[data.timestamp_utc.ge(lower) & data.timestamp_utc.lt(upper)].copy()
    common_keys = []
    coverage = []
    for zone in selected_zones:
        current = scored.loc[scored.zone.eq(zone)]
        wide = current.pivot(index="timestamp_utc", columns="model_id", values="forecast").reindex(columns=selected_models)
        zone_refs = refs.xs(zone, level="zone") if zone in refs.index.get_level_values("zone") else pd.DataFrame(columns=["actual", "storm"], index=wide.index)
        wide = wide.join(zone_refs)
        common = wide.dropna(subset=selected_models+["actual", "storm"]) if selected_models else wide.iloc[:0]
        common_keys.append(pd.DataFrame({"zone": zone, "timestamp_utc": common.index}))
        coverage.append({"zone": zone, "price_common_hours": int(len(common)), "reference_eligible_hours": 0,
                         "economic_common_hours": 0, "missing_previous_civil_hour": 0,
                         "ambiguous_previous_civil_hour": 0, "execution_ineligible_hours": 0,
                         "allocated_capacity_mw": capacities[zone]})
    keys = pd.concat(common_keys, ignore_index=True) if common_keys else pd.DataFrame(columns=KEYS)
    lagged = _lagged_references(refs, keys, timezone)
    candidate = scored.drop(columns=["actual", "storm"]).merge(refs.reset_index(), on=KEYS, validate="many_to_one")
    candidate = candidate.merge(lagged, on=KEYS, how="inner", validate="many_to_one")
    assumed_origin = "forecast_origin_utc" not in candidate
    assumed_qualification = "forecast_eligible" not in candidate or "benchmark_eligible" not in candidate
    expected_origins = _origin(pd.DatetimeIndex(candidate.timestamp_utc), timezone)
    if assumed_origin:
        candidate["forecast_origin_utc"] = expected_origins
    for name in ("forecast_eligible", "benchmark_eligible"):
        if name not in candidate:
            candidate[name] = True
    candidate["forecast_eligible"] &= candidate.forecast_origin_utc.le(expected_origins)
    candidate["duration_hours"] = 1.
    if "sample" not in candidate:
        candidate["sample"] = "evaluation"
    candidate = candidate.rename(columns={"model_id": "model", "storm": "benchmark_forecast"})
    outputs, first_benchmark = [], None
    keep = ["model", *KEYS, "paired_eligible", "position_mw", "pnl_net_eur", "absolute_energy_mwh",
            "allocated_capacity_mw", "duration_hours"]
    for model in selected_models:
        block = candidate.loc[candidate.model.eq(model)].copy()
        if block.empty:
            continue
        prepared, model_capacities = _prepare(block, policy)
        simulation = _make_rows(prepared, policy, model_capacities)
        outputs.append(simulation.loc[simulation.strategy.eq("model"), keep].copy())
        if first_benchmark is None:
            first_benchmark = simulation.loc[simulation.strategy.eq("benchmark"), keep].copy()
    model_rows = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=keep)
    benchmark = first_benchmark if first_benchmark is not None else pd.DataFrame(columns=keep)
    if len(model_rows):
        groups = model_rows.groupby(KEYS)
        qualified = groups.paired_eligible.sum().eq(len(selected_models)) & groups.model.nunique().eq(len(selected_models))
        safe_keys = qualified.loc[qualified].reset_index()[KEYS]
        model_rows = model_rows.merge(safe_keys, on=KEYS, how="inner", validate="many_to_one")
        benchmark = benchmark.merge(safe_keys, on=KEYS, how="inner", validate="one_to_one")
    for record in coverage:
        selected = lagged.loc[lagged.zone.eq(record["zone"])]
        record["reference_eligible_hours"] = int(selected.reference_eligible.sum())
        for reason in ("missing_previous_civil_hour", "ambiguous_previous_civil_hour"):
            record[reason] = int(selected.reference_missing_reason.eq(reason).sum())
        record["economic_common_hours"] = int(benchmark.loc[benchmark.zone.eq(record["zone"]), "timestamp_utc"].nunique())
        record["execution_ineligible_hours"] = record["reference_eligible_hours"]-record["economic_common_hours"]
    portfolio_hours = benchmark.groupby("timestamp_utc").zone.nunique()
    portfolio_hours = portfolio_hours.loc[portfolio_hours.eq(len(selected_zones))].index
    rows = []
    for zone in [*selected_zones, ALL_ZONE]:
        selected_benchmark = benchmark.loc[benchmark.timestamp_utc.isin(portfolio_hours)] if zone == ALL_ZONE else benchmark.loc[benchmark.zone.eq(zone)]
        for model in selected_models:
            selected = model_rows.loc[model_rows.model.eq(model)]
            selected = selected.loc[selected.timestamp_utc.isin(portfolio_hours)] if zone == ALL_ZONE else selected.loc[selected.zone.eq(zone)]
            rows.append(_row(zone, model, selected, selected_benchmark))
        rows.append(_row(zone, STORM_ID, selected_benchmark, selected_benchmark))
    coverage.append({"zone": ALL_ZONE, "economic_common_hours": int(len(portfolio_hours)),
                     "economic_common_country_hours": int(len(portfolio_hours)*len(selected_zones)),
                     "allocated_capacity_mw": float(sum(capacities[zone] for zone in selected_zones)),
                     "requires_all_selected_zones": True})
    audit = {**config_audit, "reference_kind": settings["reference"]["kind"], "executable_reference": False,
             "reference_pit_certified": False, "forecast_and_storm_issue_pit_certified": False,
             "forecast_origin_assumed_when_absent": assumed_origin,
             "forecast_qualification_assumed_when_absent": assumed_qualification,
             "sample_assumed_evaluation_when_absent": "sample" not in frame,
             "reference_availability_assumption": "source delivery D-1 published source D-2 18:00 civil; not observed issue evidence",
             "cutoff": "D-1 08:00 Europe/Paris", "dst_reference_policy": "abstain for missing or ambiguous previous civil hour",
             "reference_observation_source": "shared actual values in the supplied panel, including available pre-window rows",
             "portfolio_capacity_mw": float(settings["portfolio"]["capacity_mw"]), "zone_capacity_mw": capacities,
             "selected_allocated_capacity_mw": float(sum(capacities[zone] for zone in selected_zones)),
             "capacity_is_never_redistributed_when_filtering": True,
             "strategy": settings["strategy"], "signal_hurdle_eur_mwh": sum(float(settings["strategy"][key]) for key in ("signal_threshold_eur_mwh", "transaction_cost_eur_mwh", "slippage_eur_mwh")),
             "net_cost_eur_mwh": float(settings["strategy"]["transaction_cost_eur_mwh"]+settings["strategy"]["slippage_eur_mwh"]),
             "gain_per_mwh_denominator": "sum(fixed allocated MW * eligible physical interval duration); same for model and Storm, not traded volumes",
             "model_alternatives_are_not_cumulative": True, "no_forecast_position_mw": 0.,
             "parameters_fitted_on_evaluation": False, "annualisation_performed": False,
             "orders_placed": False, "production_modified": False,
             "warning": "Diagnostic hypothetical PnL: the previous-day delivery price is not an executable quote for the predicted delivery."}
    return {"schema_version": 1, "period": {"start_day": start.isoformat(), "end_day": end.isoformat(), "days": days, "timezone": timezone},
            "rows": rows, "coverage": coverage, "audit": audit}
