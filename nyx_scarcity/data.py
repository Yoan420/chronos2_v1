"""Read-only, as-of fundamentals for the isolated NYX scarcity challenger.

The baseline is the published nuclear Kalman report, not a newly reconstructed
forecast. Capacity coverage is a proxy; neither imports nor realised generation
are invented. Historical query-as-of evidence is not publication certification.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from economic_value.data import EconomicDataError, _aware, _origin, _stable_bytes, load_report_panel


class ScarcityDataError(EconomicDataError):
    pass


ZONES = ("FR", "DE", "BE", "NL")
TIMEZONES = {"Europe/Paris", "Europe/Berlin", "Europe/Brussels", "Europe/Amsterdam"}


def source_registry() -> dict[str, dict[str, Any]]:
    """Exact identities; overrides may relocate a file, never redefine a series."""
    result = {}
    for zone in ZONES:
        z = zone.lower()
        result[f"{z}_residual_load"] = {
            "path": "data/pit/nuclear_forecast/residual_load_market_features.parquet",
            "column": f"{z}_residual_load_fcst", "series": f"power.{z}.residual.load.hourly.gw.fcst",
            "feature": f"feature_{z}_residual_load_gw", "unit": "GW", "kind": "day_ahead_forecast"}
        result[f"{z}_gas_available"] = {
            "path": f"data/pit/marginal_cost_expert_v2/capacities/{z}_gas_available_gw.parquet",
            "column": "value", "series": f"power.nrjscan.{z}.3mv.availability.pmax.fuel.nat_gas.gw",
            "feature": f"feature_{z}_gas_available_gw", "unit": "GW", "kind": "capacity_forecast_proxy"}
        for technology in ("wind", "solar"):
            key = f"{z}_{technology}_generation"
            series = ("power.nl.prod.total.wind.mw.ecmwf_avg.pointconnect.6h.cache"
                      if key == "nl_wind_generation" else f"power.{z}.generation.{technology}.hourly.gw.fcst")
            result[key] = {"path": f"data/pit/kalman_weather/{key}_fcst.parquet", "column": "value",
                           "series": series, "feature": f"feature_{key}_gw", "unit": "GW",
                           "kind": "day_ahead_forecast", "required_materialized_scale": 0.001 if key == "nl_wind_generation" else 1.0}
        result[f"{z}_temperature"] = {
            "path": f"data/pit/kalman_weather/{z}_temperature_fcst.parquet", "column": "value",
            "series": f"meteo.nrjscan.{z}.t_2m.index.fcst.d", "feature": f"feature_{z}_temperature_c",
            "unit": "degC", "kind": "daily_forecast_temperature_index"}
    result["fr_nuclear_generation"] = {
        "path": "data/pit/nuclear_forecast/fr_nuclear_generation_fcst_gw.parquet", "column": "value",
        "series": "power.fr.generation.nuclear.gw.fcst", "feature": "feature_fr_nuclear_generation_gw",
        "unit": "GW", "kind": "day_ahead_forecast"}
    for zone, technology in (("de", "coal"), ("de", "lignite"), ("nl", "coal"), ("be", "nuclear"), ("nl", "nuclear")):
        key = f"{zone}_{technology}_available"
        category = "type" if technology == "nuclear" else "fuel"
        result[key] = {"path": f"data/pit/marginal_cost_expert/capacities/{key}_gw.parquet",
                       "column": "value", "series": f"power.nrjscan.{zone}.3mv.availability.pmax.{category}.{technology}.gw",
                       "feature": f"feature_{key}_gw", "unit": "GW", "kind": "capacity_forecast_proxy"}
    for key, column, series, source_time, unit in (
        ("ttf", "ttf_m1_eur_mwh_th", "gas.ttf.price.everyday.month.1.ice.eurmwh", "ttf_source_value_time_utc", "EUR/MWh_th"),
        ("eua", "eua_first_dec_eur_tco2", "carbon.eu.price.everyday.eua.ice.1st.dec", "eua_source_value_time_utc", "EUR/tCO2"),
    ):
        result[key] = {"path": "data/pit/marginal_cost_expert/fuel/market_fuel_features.parquet", "column": column,
                       "series": series, "feature": f"feature_{key}", "unit": unit,
                       "source_time_column": source_time, "kind": "market_observation_known_before_cutoff"}
    return result


def _path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ScarcityDataError("A source override must be a nonempty path string.")
    source = (root / value).resolve()
    allowed = [root / "data" / "pit", root / "runs" / "experiments" / "nyx_scarcity_v1" / "source_refresh"]
    if not source.is_relative_to(root) or not any(source.is_relative_to(folder) for folder in allowed):
        raise ScarcityDataError("Scarcity inputs must remain inside data/pit or the isolated scarcity source_refresh directory.")
    return source


def _read_source(root: Path, key: str, spec: dict, expected: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Latest eligible vintage for each physical hour; never interpolate holes."""
    path = _path(root, spec["path"])
    sidecar = Path(str(path) + ".audit.json")
    columns = [spec["feature"], spec["feature"] + "_revision_delta", spec["feature"] + "_vintage_range_proxy"]
    if not path.exists() and not sidecar.exists():
        return pd.DataFrame(np.nan, index=expected, columns=columns), {
            "key": key, "path": str(path), "series": spec["series"], "status": "missing_source",
            "missing_hours": len(expected), "production_pit_evidence": False}
    raw, evidence = _stable_bytes(path)
    audit_raw, audit_evidence = _stable_bytes(sidecar)
    audit = json.loads(audit_raw)
    if evidence["sha256"] != (audit.get("sha256") or audit.get("output_sha256") or audit.get("parquet_sha256")):
        raise ScarcityDataError(f"{key}: source checksum differs from its audit.")
    identity = audit.get("series")
    if isinstance(identity, dict):
        identity = identity.get(spec["column"])
    if identity != spec["series"]:
        raise ScarcityDataError(f"{key}: source series outside the exact feature allowlist.")
    if audit.get("cutoff_time") != "08:00" or audit.get("cutoff_timezone") not in TIMEZONES:
        raise ScarcityDataError(f"{key}: explicit D-1 08:00 civil contract required.")
    if audit.get("causality_violations", 0):
        raise ScarcityDataError(f"{key}: upstream causality violation.")
    if "required_materialized_scale" in spec and audit.get("value_scale") != spec["required_materialized_scale"]:
        raise ScarcityDataError(f"{key}: audited materialized units differ from GW.")
    if audit.get("unit") not in (None, spec["unit"]):
        raise ScarcityDataError(f"{key}: audited unit differs from the registry.")
    frame = pd.read_parquet(BytesIO(raw))
    required = {"value_time_utc", "snapshot_time_utc", "revision_time_utc", spec["column"]}
    if spec.get("source_time_column"):
        required.add(spec["source_time_column"])
    if required.difference(frame):
        raise ScarcityDataError(f"{key}: incomplete PIT columns: {sorted(required.difference(frame))}.")
    delivery = _aware(frame.value_time_utc, key + "/delivery")
    snapshot = _aware(frame.snapshot_time_utc, key + "/snapshot")
    revision = _aware(frame.revision_time_utc, key + "/revision")
    cutoff = _origin(delivery, "Europe/Paris")
    if not delivery.equals(delivery.floor("h")):
        raise ScarcityDataError(f"{key}: only physical hourly delivery timestamps are allowed.")
    if "cutoff_time_utc" in frame and not _aware(frame.cutoff_time_utc, key + "/cutoff").equals(cutoff):
        raise ScarcityDataError(f"{key}: per-row cutoff differs from D-1 08:00 civil.")
    values = pd.to_numeric(frame[spec["column"]], errors="raise").to_numpy(float)
    if np.isinf(values).any() or (spec["kind"] == "capacity_forecast_proxy" and np.nanmin(values, initial=0) < 0):
        raise ScarcityDataError(f"{key}: invalid nonfinite/negative capacity input.")
    late = (snapshot > cutoff) | (revision > cutoff)
    if spec.get("source_time_column"):
        times = _aware(frame[spec["source_time_column"]], key + "/observation")
        late |= times > cutoff
    table = pd.DataFrame({"delivery": delivery, "snapshot": snapshot, "revision": revision, "value": values})
    keys = ["delivery", "snapshot", "revision"]
    if table.groupby(keys, dropna=False).value.nunique(dropna=False).gt(1).any():
        raise ScarcityDataError(f"{key}: conflicting values for a PIT identity.")
    eligible = table.loc[delivery.isin(expected) & ~late].drop_duplicates(keys).sort_values(keys)
    latest = eligible.drop_duplicates("delivery", keep="last").set_index("delivery")
    selected = pd.DataFrame(index=expected)
    selected[columns[0]] = latest.value.reindex(expected)
    # Two different forecasts for the SAME delivery hour, both before 08:00.
    # An independently selected D-1 forecast for another day is not a revision.
    grouped = eligible.groupby("delivery", sort=False)
    difference = grouped.value.diff()
    eligible = eligible.assign(revision_delta=difference)
    changes = eligible.drop_duplicates("delivery", keep="last").set_index("delivery").revision_delta
    spread = grouped.value.max() - grouped.value.min()
    counts = grouped.size()
    selected[columns[1]] = changes.reindex(expected)
    selected[columns[2]] = spread.where(counts.ge(2)).reindex(expected)
    evidence.update({"key": key, "status": "loaded", "audit": audit_evidence, "series": spec["series"],
                     "unit": spec["unit"], "information_type": spec["kind"],
                     "finite_hours": int(selected[columns[0]].notna().sum()),
                     "missing_hours": int(selected[columns[0]].isna().sum()),
                     "first_selected_utc": str(latest.index.min()) if len(latest) else None,
                     "last_selected_utc": str(latest.index.max()) if len(latest) else None,
                     "late_rows_excluded": int((delivery.isin(expected) & late).sum()),
                     "eligible_multivintage_hours": int(counts.ge(2).sum()),
                     "selected_origin_digest": hashlib.sha256(latest[["snapshot", "revision"]].to_json(date_format="iso").encode()).hexdigest(),
                     "selection": "latest snapshot then revision <= delivery D-1 08:00 civil",
                     "provider_revision_timestamp_available": audit.get("provider_revision_timestamp_available", False),
                     "revision_time_semantics": audit.get("revision_time_semantics"),
                     "approximation": audit.get("approximation"), "fill_or_interpolation": audit.get("fill_or_interpolation"),
                     "production_pit_evidence": False})
    return selected, evidence


def _daily_ramp(values: pd.Series, index: pd.DatetimeIndex, hours: int) -> pd.Series:
    """A within-origin future profile ramp, never cross-day mixed forecasts."""
    local_days = index.tz_convert("Europe/Paris").strftime("%Y-%m-%d")
    return values.groupby(local_days).diff(hours) / hours


def _positive_setting(settings: dict, key: str, default: float, maximum: float | None = None) -> float:
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 or (maximum and value > maximum):
        raise ScarcityDataError(f"{key}: finite positive value required" + (f" <= {maximum}" if maximum else ""))
    return float(value)


def load_inputs(config: dict, *, root: Path) -> tuple[pd.DataFrame, dict]:
    """365 frozen reporting days plus the unlabelled live day; no extra year."""
    root = Path(root).resolve()
    zones = config.get("zones", list(ZONES))
    if not isinstance(zones, list) or not zones or len(zones) != len(set(zones)) or set(zones) - set(ZONES):
        raise ScarcityDataError("Choose unique supported FR, DE, BE, NL zones.")
    if config.get("baseline_model", "nuclear_kalman") != "nuclear_kalman":
        raise ScarcityDataError("This experiment isolates the published nuclear_kalman baseline.")
    if config.get("timezone", "Europe/Paris") != "Europe/Paris":
        raise ScarcityDataError("The challenger uses the explicit Europe/Paris civil cutoff.")
    settings = config.get("data", {})
    allowed = {"source_overrides", "ccgt_efficiency", "ocgt_efficiency", "gas_emissions_tco2_mwh_th", "minimum_available_gas_gw"}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ScarcityDataError("Unknown data setting; sources/features must use the explicit allowlist.")
    ccgt = _positive_setting(settings, "ccgt_efficiency", .58, 1)
    ocgt = _positive_setting(settings, "ocgt_efficiency", .39, 1)
    emission = _positive_setting(settings, "gas_emissions_tco2_mwh_th", .202)
    floor = _positive_setting(settings, "minimum_available_gas_gw", .1)
    registry = source_registry()
    overrides = settings.get("source_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - set(registry):
        raise ScarcityDataError("Unknown source override: no native CGC, JAO, Storm or realised power price is registered as a feature.")
    for key, path in overrides.items():
        registry[key]["path"] = path
    panel, baseline_audit = load_report_panel(root, zones, ["nuclear_kalman"],
        delivery_day=config.get("delivery_day"), end_day=config.get("end_day"), timezone="Europe/Paris")
    panel = panel.copy()
    expected = pd.DatetimeIndex(sorted(panel.timestamp_utc.unique())).tz_convert("UTC")
    raw_frames, sources = [], {}
    for key, spec in registry.items():
        block, sources[key] = _read_source(root, key, spec, expected)
        raw_frames.append(block)
    common = pd.concat(raw_frames, axis=1)
    # Preserve all raw levels, but only actual multi-vintage information produces
    # revision/range features. No stochastic ensemble is fabricated.
    rl = [f"feature_{z.lower()}_residual_load_gw" for z in ZONES]
    gas = [f"feature_{z.lower()}_gas_available_gw" for z in ZONES]
    common["feature_regional_residual_load_gw"] = common[rl].sum(axis=1, min_count=4)
    common["feature_regional_gas_available_gw"] = common[gas].sum(axis=1, min_count=4)
    common["feature_regional_residual_spread_gw"] = common[rl].max(axis=1, skipna=False) - common[rl].min(axis=1, skipna=False)
    # A residual-load/gas-only stress ratio is deliberately not a reserve margin:
    # coal, hydro, imports, cogeneration and fleet coverage are incomplete.
    stress = pd.DataFrame({z: common[f"feature_{z.lower()}_residual_load_gw"] /
                           common[f"feature_{z.lower()}_gas_available_gw"].clip(lower=floor)
                           for z in ZONES})
    common["feature_regional_residual_to_gas_ratio_proxy"] = stress.mean(axis=1, skipna=False)
    common["feature_regional_simultaneous_gas_stress_proxy"] = stress.gt(1).sum(axis=1).where(stress.notna().all(axis=1)) / 4
    for typ, efficiency in (("ccgt", ccgt), ("ocgt", ocgt)):
        common[f"feature_clean_gas_cost_{typ}_proxy_eur_mwh"] = (common.feature_ttf + emission * common.feature_eua) / efficiency
    common["feature_gas_merit_slope_proxy_eur_mwh"] = common.feature_clean_gas_cost_ocgt_proxy_eur_mwh - common.feature_clean_gas_cost_ccgt_proxy_eur_mwh
    common["feature_hour_sin"] = np.sin(expected.tz_convert("Europe/Paris").hour * np.pi / 12)
    common["feature_hour_cos"] = np.cos(expected.tz_convert("Europe/Paris").hour * np.pi / 12)
    common["feature_weekday"] = expected.tz_convert("Europe/Paris").dayofweek
    common["feature_month"] = expected.tz_convert("Europe/Paris").month
    base_features = list(common)
    outputs = []
    for zone in zones:
        z = zone.lower()
        block = panel.loc[panel.zone.eq(zone)].set_index("timestamp_utc").sort_index().copy()
        block = block.join(common, validate="one_to_one")
        local_residual = block[f"feature_{z}_residual_load_gw"]
        local_gas = block[f"feature_{z}_gas_available_gw"]
        block["feature_local_residual_load_gw"] = local_residual
        block["feature_local_gas_available_gw"] = local_gas
        block["feature_local_residual_to_gas_ratio_proxy"] = local_residual / local_gas.clip(lower=floor)
        block["feature_local_gas_zero_available"] = local_gas.eq(0).astype(float).where(local_gas.notna())
        block["feature_local_gas_minus_residual_proxy_gw"] = local_gas - local_residual
        selected_supply = {
            "FR": ["feature_fr_gas_available_gw", "feature_fr_nuclear_generation_gw"],
            "DE": ["feature_de_gas_available_gw", "feature_de_coal_available_gw", "feature_de_lignite_available_gw"],
            "BE": ["feature_be_gas_available_gw", "feature_be_nuclear_available_gw"],
            "NL": ["feature_nl_gas_available_gw", "feature_nl_coal_available_gw", "feature_nl_nuclear_available_gw"],
        }[zone]
        block["feature_local_selected_supply_proxy_gw"] = block[selected_supply].sum(axis=1, min_count=len(selected_supply))
        block["feature_local_selected_supply_minus_residual_proxy_gw"] = block.feature_local_selected_supply_proxy_gw - local_residual
        # Wind/solar are already represented in residual load. They enter only
        # as profile descriptors, never subtracted for a second time.
        for tech in ("wind", "solar"):
            values = block[f"feature_{z}_{tech}_generation_gw"]
            block[f"feature_local_{tech}_generation_gw"] = values
            block[f"feature_local_{tech}_ramp_1h_gw_per_hour"] = _daily_ramp(values, block.index, 1)
            block[f"feature_local_{tech}_ramp_3h_gw_per_hour"] = _daily_ramp(values, block.index, 3)
        for hours in (1, 3):
            ramp = _daily_ramp(local_residual, block.index, hours)
            block[f"feature_local_residual_ramp_{hours}h_gw_per_hour"] = ramp
            block[f"feature_local_residual_ramp_{hours}h_to_gas_proxy"] = ramp / local_gas.clip(lower=floor)
        block["feature_local_residual_vs_region_gw"] = local_residual - block.feature_regional_residual_load_gw / 4
        block["feature_local_temperature_c"] = block[f"feature_{z}_temperature_c"]
        block["feature_baseline_forecast"] = block.forecast
        block["feature_baseline_interval_width"] = block.q90 - block.q10
        block["feature_baseline_upper_distance"] = block.q90 - block.forecast
        block["feature_cgc_minus_baseline_proxy_eur_mwh"] = block.feature_clean_gas_cost_ccgt_proxy_eur_mwh - block.forecast
        local_days = block.index.tz_convert("Europe/Paris").tz_localize(None).normalize()
        block["label_available_at_utc"] = (local_days - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC")
        block["label_availability_assumed"] = True
        block["label_eligible"] = np.isfinite(block.actual)
        block["forecast_eligible"] = np.isfinite(block.forecast)
        # Conservative bound: all selected exogenous vintages are no later
        # than this instant; it is not a recovered original publication time.
        block["feature_available_at_utc"] = block.forecast_origin_utc
        block["diagnostic_only"] = True
        required = rl + ["feature_fr_nuclear_generation_gw", "feature_local_gas_available_gw",
                         "feature_local_wind_generation_gw", "feature_local_solar_generation_gw", "feature_baseline_forecast"]
        block["feature_eligible"] = block[required].notna().all(axis=1) & local_gas.ge(0)
        block["scarcity_eligible"] = block.feature_eligible
        block["eligibility_reason"] = np.where(block.feature_eligible, "ready", "missing_required_fundamentals_or_baseline")
        outputs.append(block.rename_axis("timestamp_utc").reset_index())
    result = pd.concat(outputs, ignore_index=True).sort_values(["timestamp_utc", "zone"]).reset_index(drop=True)
    # Only columns created in this function can be used as features. Inherited
    # report fields (actual, Storm, sample) are never implicitly selected.
    added_features = [c for c in outputs[0] if c.startswith("feature_") and c not in {"feature_eligible", "feature_available_at_utc"}]
    feature_columns = [c for c in added_features if c in base_features or c not in panel.columns]
    if len(feature_columns) != len(set(feature_columns)):
        raise ScarcityDataError("Duplicate feature identities.")
    missing_flags = {c + "__missing": result[c].isna().astype(float) for c in feature_columns}
    result = pd.concat([result, pd.DataFrame(missing_flags)], axis=1)
    feature_columns += list(missing_flags)
    if np.isinf(result[feature_columns].to_numpy(float)).any():
        raise ScarcityDataError("Nonfinite derived feature; refusing model input.")
    # Detect a concurrent source refresh across this multi-file capture.
    for item in sources.values():
        for evidence in (item, item.get("audit", {})):
            if evidence.get("sha256") and hashlib.sha256(Path(evidence["path"]).read_bytes()).hexdigest() != evidence["sha256"]:
                raise ScarcityDataError("Source changed during scarcity input assembly.")
    coverage = {c: {"finite_rows": int(result[c].notna().sum()), "missing_rows": int(result[c].isna().sum())} for c in feature_columns}
    eligibility = {}
    for zone in zones:
        b = result.loc[result.zone.eq(zone)]
        days = b.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
        eligibility[zone] = {"eligible_rows": int(b.feature_eligible.sum()), "fallback_rows": int((~b.feature_eligible).sum()),
                             "fallback_days": sorted(days.loc[~b.feature_eligible].unique().tolist())}
    audit = {"schema_version": 1, "baseline": baseline_audit, "sources": sources,
             "delivery_day": baseline_audit["delivery_day"], "evaluation_start_day": baseline_audit["evaluation_start_day"],
             "evaluation_end_day": baseline_audit["evaluation_end_day"], "evaluation_days": 365,
             "zones": zones, "rows": len(result), "feature_columns": feature_columns,
             "required_feature_columns": required,
             "feature_coverage": coverage, "eligibility": eligibility,
             "forecast_cutoff": "D-1 08:00 Europe/Paris", "training_history": "only the 365 reporting days; no invented prior baseline year",
             "label_availability": "assumed D-1 18:00 Europe/Paris; not provider publication timestamp",
             "capacity_semantics": "daily available-Pmax forecasts, not generation; unverified national fleet scope; gas fuel aggregate is not added to CCGT/GT components",
             "margin_semantics": "gas-minus-residual proxy only; no full reserve margin, import capacity, reserve commitment or deliverable flexibility claim",
             "selected_supply_proxy": "FR gas Pmax + nuclear generation forecast; DE gas + coal + lignite Pmax; BE gas + nuclear Pmax; NL gas + coal + nuclear Pmax. Incomplete, mixed capacity/generation approximation, never a full national balance.",
             "renewable_accounting": "no subtraction from residual load; separate wind/solar only for levels and ramps",
             "ramp_semantics": "within the same delivery-day vintage; residual ramp / available gas is not a measured GW/hour ramp capability",
             "ratio_denominator_floor_gw": floor,
             "zero_capacity_semantics": "A valid zero capacity remains zero and eligible; only ratio denominators use the stated floor, never imputed physical capacity.",
             "uncertainty_semantics": "baseline P10-P90 plus past eligible vintage range; no genuine meteorological ensemble assumed",
             "clean_gas_cost": {"native_saturn_cgc_used": False, "formula": "(TTF_M1 + EUA_first_Dec * gas_emission_factor) / efficiency",
                                "ccgt_efficiency": ccgt, "ocgt_efficiency": ocgt, "emissions_tco2_mwh_th": emission,
                                "variable_om_included": False, "contract_mismatch": "M1 gas / first December EUA are cost proxies, not same-hour executable costs"},
             "jao": {"status": "excluded_fail_closed", "reason": "08:00 source/publication qualification incomplete; no RAM/PTDF reconstructed or imported"},
             "diagnostic_only": True, "production_pit_evidence": False, "forecast_pit_certified": False,
             "benchmark_pit_certified": False, "promotion_eligible": False, "production_modified": False,
             "fill_or_interpolation": "none; optional NaN plus missing flags; required gaps force baseline fallback"}
    return result, audit


__all__ = ["ScarcityDataError", "source_registry", "load_inputs"]
