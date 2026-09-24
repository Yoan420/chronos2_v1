"""Isolated V2 supply-source qualification and causal fuel collection.

Pmax/Pnom are not automatically national totals. ``full`` availability formulae
are forbidden: their priority branch may contain realised availability. Only
source-state queries at civil D-1 08:00 enter the bank; public observations may
be archived separately as retrospective checks, never as forecast features.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import MarginalCostDataError, _load_source, expected_hours, _cutoffs
from .sources import ROOT, URL, TIMEZONE, _client, civil_cutoff, sha256, _write_json, _daily_values, _one_day

V2_ROOT = ROOT / "data/pit/marginal_cost_expert_v2"
PROBE_DAYS = ["2024-09-10", "2024-11-15", "2025-01-15", "2025-04-01", "2025-09-01",
              "2025-12-01", "2026-06-24", "2026-06-25", "2026-06-26", "2026-09-09"]
MARKET_SOURCES = {
    "api2_usd_t": "coal.api2.price.everyday.month.1.ice.usdt",
    "eurusd_usd_per_eur": "forex.everyday.eurusd.close",
}
_THREAD_CLIENT = threading.local()


def _pooled_client():
    # Reuse HTTPS/TLS connections within each of the at most two workers.
    if not hasattr(_THREAD_CLIENT, "client"):
        _THREAD_CLIENT.client = _client()
    return _THREAD_CLIENT.client


def isolated_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    root = V2_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("V2 collection must remain under data/pit/marginal_cost_expert_v2")
    return resolved


def validate_physical_identity(series: str) -> None:
    """Storm is a provider tag only for explicit physical forecast series."""
    series = series.lower()
    if re.search(r"(?:^|[._])(?:full|obs|observed|realised|realized)(?:$|[._])", series):
        raise ValueError("Observed/full series cannot supply forecast features")
    if re.search(r"price|spread|chronos|kalman|mkonline|q50", series):
        raise ValueError("Electricity prices or model predictions cannot enter supply features")
    if "storm" in series and not re.fullmatch(r"power\.[a-z]{2}\.avail\.aggr\.[a-z.]+\.mw\.h\.fct\.utc\.3mv\.storm", series):
        raise ValueError("Only explicitly identified physical availability forecasts may use Storm's provider tag")


def api2_thermal_cost(api2_usd_t: Any, eurusd: Any, *, net_calorific_mwh_t: float) -> Any:
    """USD/t / (USD/EUR) / (MWh_th/t); never apply API2 to lignite."""
    if not np.isfinite(net_calorific_mwh_t) or net_calorific_mwh_t <= 0:
        raise ValueError("Explicit positive net calorific content required")
    price, fx = np.asarray(api2_usd_t, dtype=float), np.asarray(eurusd, dtype=float)
    if (np.isfinite(price) & (price < 0)).any() or (np.isfinite(fx) & (fx <= 0)).any():
        raise ValueError("Invalid coal or FX quotation")
    return price / fx / net_calorific_mwh_t


def _numeric(raw: Any) -> pd.Series:
    if raw is None:
        return pd.Series(dtype=float)
    if not isinstance(raw, pd.Series):
        raise ValueError("Expected Saturn Series")
    value = pd.to_numeric(raw, errors="coerce")
    if (raw.notna() & value.isna()).any() or np.isinf(value).any():
        raise ValueError("Malformed/nonfinite source value")
    if raw.index.has_duplicates:
        raise ValueError("Duplicated source timestamps")
    return value.astype(float).sort_index()


def select_previous_close(raw: Any, cutoff: pd.Timestamp, *, maximum_age_hours: float = 176) -> tuple[float, pd.Timestamp]:
    """Exclude same-civil-day close labels even when an as-of service returns one.

    Daily market labels denote trading dates, not actual publication times.
    Conservatively use the previous trading date, with a separate age audit.
    """
    values = _numeric(raw)
    index = pd.DatetimeIndex(values.index)
    if index.tz is None:
        index = index.tz_localize(TIMEZONE, ambiguous="raise", nonexistent="raise")
    index = index.tz_convert("UTC")
    if index.hasnans:
        raise ValueError("Missing source timestamp")
    values.index = index
    cutoff = pd.Timestamp(cutoff).tz_convert("UTC")
    local_date = cutoff.tz_convert(TIMEZONE).normalize()
    selected = values[(index < local_date.tz_convert("UTC")) & (index <= cutoff)].dropna()
    if selected.empty:
        raise ValueError("No previous close known by origin")
    stamp, value = selected.index[-1], float(selected.iloc[-1])
    if (cutoff - stamp).total_seconds() / 3600 > maximum_age_hours:
        raise ValueError("Previous close exceeds maximum permitted age")
    return value, stamp


def _probe_one(alias: str, spec: Mapping[str, Any], days: Sequence[str]) -> dict[str, Any]:
    client = _client()
    name = str(spec["series"])
    if spec["kind"] != "market":
        validate_physical_identity(name)
    records = []
    metadata = client.metadata(name, all=True)
    for day_string in days:
        day = pd.Timestamp(day_string)
        cutoff = civil_cutoff(day)
        try:
            raw = client.get(name, from_value_date=day - pd.Timedelta(days=9 if spec["kind"] == "market" else 1),
                             to_value_date=day + pd.Timedelta(days=1), revision_date=cutoff)
            values = _numeric(raw)
            source_stamp = None
            if spec["kind"] == "market":
                val, source_stamp = select_previous_close(raw, cutoff)
                values = pd.Series([val])
            elif spec["kind"] == "daily":
                if pd.DatetimeIndex(values.index).tz is not None:
                    raise ValueError("Expected naive civil daily labels")
                values = values.loc[values.index == day]
            else:
                idx = pd.DatetimeIndex(values.index)
                if idx.tz is None:
                    raise ValueError("Hourly UTC forecast must have aware timestamps")
                start = day.tz_localize(TIMEZONE).tz_convert("UTC")
                end = (day + pd.Timedelta(days=1)).tz_localize(TIMEZONE).tz_convert("UTC")
                values = values.loc[(idx >= start) & (idx < end)]
            finite = values.dropna()
            records.append({"day": day_string, "cutoff_utc": cutoff.isoformat(), "count": len(values),
                            "finite": len(finite), "min": float(finite.min()) if len(finite) else None,
                            "mean": float(finite.mean()) if len(finite) else None,
                            "max": float(finite.max()) if len(finite) else None,
                            "source_value_time_utc": source_stamp.isoformat() if source_stamp is not None else None})
        except Exception as exc:
            records.append({"day": day_string, "error": f"{type(exc).__name__}: {exc}"})
    return {"alias": alias, **dict(spec), "metadata": metadata, "probes": records,
            "national_coverage_attested": False, "provider_revision_timestamp_available": False}


def probe_catalog() -> dict[str, dict[str, Any]]:
    out = {}
    for zone, fuel in [("BE", "nuclear"), ("DE", "coal"), ("DE", "lignite"), ("NL", "coal")]:
        z = zone.lower()
        for measure in ["availability.pmax", "installedcapacity.pnom"]:
            alias = f"{z}_{fuel}_{measure.split('.')[-1]}"
            out[alias] = {"series": f"power.nrjscan.{z}.3mv.{measure}.fuel.{fuel}.gw", "kind": "daily", "unit": "GW"}
        out[f"{z}_{fuel}_forecast"] = {"series": f"power.{z}.avail.aggr.{fuel}.mw.h.fct.utc.3mv.storm", "kind": "hourly", "unit": "MW"}
    out["be_nuclear_production"] = {"series": "power.be.prod.nuclear.nuclear.mw.h.fct.utc.3mv", "kind": "hourly", "unit": "MW"}
    for zone in ["FR", "DE", "BE", "NL"]:
        for measure in ["availability.pmax", "installedcapacity.pnom"]:
            out[f"{zone.lower()}_gas_{measure.split('.')[-1]}"] = {
                "series": f"power.nrjscan.{zone.lower()}.3mv.{measure}.fuel.nat_gas.gw", "kind": "daily", "unit": "GW"}
    out.update({alias: {"series": name, "kind": "market", "unit": "USD/t" if alias.startswith("api2") else "USD/EUR"}
                for alias, name in MARKET_SOURCES.items()})
    return out


def probe_sources(output_dir: str | Path, *, aliases: Sequence[str] | None = None, workers: int = 2) -> Path:
    if not 1 <= workers <= 2:
        raise ValueError("At most two Saturn workers permitted")
    root = isolated_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    specs = probe_catalog()
    aliases = list(aliases or specs)
    if set(aliases).difference(specs):
        raise ValueError("Unknown source alias")
    destination = root / "source_probes.json"
    payload = {"schema_version": 2, "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
               "days": PROBE_DAYS, "records": {}, "production_pit_evidence": False}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_probe_one, alias, specs[alias], PROBE_DAYS): alias for alias in aliases}
        for future in as_completed(futures):
            alias = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"alias": alias, "error": f"{type(exc).__name__}: {exc}"}
            payload["records"][alias] = record
            _write_json(destination, payload)
            finite = sum(bool(row.get("finite")) for row in record.get("probes", []))
            print(f"[Supply V2] probe {alias}: {finite}/{len(PROBE_DAYS)} dates avec valeurs", flush=True)
    return destination


def _market_day(day_string: str) -> dict[str, Any]:
    cutoff = civil_cutoff(pd.Timestamp(day_string))
    client = _pooled_client()
    record: dict[str, Any] = {"day": day_string, "cutoff_utc": cutoff.isoformat()}
    for alias, name in MARKET_SOURCES.items():
        for attempt in range(3):
            try:
                raw = client.get(name, from_value_date=cutoff - pd.Timedelta(days=10),
                                 to_value_date=cutoff, revision_date=cutoff)
                value, stamp = select_previous_close(raw, cutoff)
                record[alias] = value
                record[alias + "_source_time_utc"] = stamp.isoformat()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    return record


def _validate_market_record(record: Mapping[str, Any], day: str) -> None:
    cutoff = civil_cutoff(pd.Timestamp(day))
    if record.get("day") != day or record.get("series") != MARKET_SOURCES or record.get("cutoff_utc") != cutoff.isoformat():
        raise ValueError("Coal/FX checkpoint identity or cutoff mismatch")
    previous_date_boundary = cutoff.tz_convert(TIMEZONE).normalize().tz_convert("UTC")
    for alias in MARKET_SOURCES:
        value = float(record[alias])
        stamp = pd.Timestamp(record[alias + "_source_time_utc"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid coal/FX checkpoint value")
        if stamp.tzinfo is None or stamp >= previous_date_boundary or stamp > cutoff:
            raise ValueError("Coal/FX checkpoint violates previous-close causality")
        if (cutoff - stamp).total_seconds() / 3600 > 176:
            raise ValueError("Stale coal/FX checkpoint")


def materialize_coal_fx(start_day: str, end_day: str, output_dir: str | Path, *, workers: int = 2,
                        net_calorific_mwh_t: float = 6.978) -> dict[str, Any]:
    if not 1 <= workers <= 2:
        raise ValueError("At most two Saturn workers permitted")
    expected_hours(start_day, end_day)  # dates/DST validation before requests
    root = isolated_path(output_dir)
    checkpoints = root / "coal_fx_daily"
    checkpoints.mkdir(parents=True, exist_ok=True)
    days = [str(day.date()) for day in pd.date_range(start_day, end_day, freq="D")]
    records, missing = {}, []
    for day in days:
        path = checkpoints / (day + ".json")
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            _validate_market_record(record, day)
            records[day] = record
        else:
            missing.append(day)
    begun, errors = time.monotonic(), {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_market_day, day): day for day in missing}
        for number, future in enumerate(as_completed(futures), 1):
            day = futures[future]
            try:
                record = future.result()
                record["series"] = MARKET_SOURCES
                _validate_market_record(record, day)
                _write_json(checkpoints / (day + ".json"), record)
                records[day] = record
            except Exception as exc:
                errors[day] = f"{type(exc).__name__}: {exc}"
            if number % 30 == 0 or number == len(missing):
                elapsed = time.monotonic() - begun
                remaining = (len(missing) - number) * elapsed / number
                progress = {"complete_days": len(records), "requested_days": len(days), "failures": errors,
                            "elapsed_seconds": elapsed, "estimated_remaining_seconds": remaining}
                _write_json(root / "collection_progress.json", progress)
                print(f"[Supply V2] coal/FX {len(records)}/{len(days)}; ETA {remaining / 60:.1f} min; erreurs={len(errors)}", flush=True)
    frames = []
    for day in days:
        if day not in records:
            continue
        record = records[day]
        index = expected_hours(day, day)
        block = pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": pd.Timestamp(record["cutoff_utc"]),
                              "revision_time_utc": pd.Timestamp(record["cutoff_utc"])})
        for alias in MARKET_SOURCES:
            block[alias] = record[alias]
            block[alias + "_source_time_utc"] = pd.Timestamp(record[alias + "_source_time_utc"])
        block["coal_eur_mwh_th"] = api2_thermal_cost(block.api2_usd_t, block.eurusd_usd_per_eur,
                                                    net_calorific_mwh_t=net_calorific_mwh_t)
        frames.append(block)
    if not frames:
        raise ValueError("No coal/FX days available; no empty artifact published")
    output = root / "coal_fx_features.parquet"
    temporary = output.with_suffix(".tmp.parquet")
    pd.concat(frames, ignore_index=True).to_parquet(temporary, index=False)
    temporary.replace(output)
    audit = {"schema_version": 2, "sha256": sha256(output), "series": MARKET_SOURCES,
             "cutoff_time": "08:00", "cutoff_timezone": TIMEZONE, "causality_violations": 0,
             "start_day": start_day, "end_day": end_day, "days": len(records), "missing_days": sorted(set(days) - records.keys()),
             "errors": errors, "net_calorific_mwh_t": net_calorific_mwh_t,
             "conversion": "API2 M1 USD/t / previous-close EURUSD USD/EUR / MWh_th/t",
             "energy_content_basis": "6000 kcal/kg NAR, using international-table calorie 4186.8 J; proxy not plant-specific",
             "energy_content_source": "https://www.globalcoal.com/coaltrading/financialcoaltrading.cfm",
             "same_day_market_closes_excluded": True, "maximum_source_age_hours": 176,
             "provider_revision_timestamp_available": False, "revision_time_semantics": "query_asof_cutoff, not provider insertion timestamp",
             "production_pit_evidence": False, "promotion_eligible": False,
             "lignite_not_mapped_to_api2": True}
    _write_json(output.with_suffix(".parquet.audit.json"), audit)
    return audit


def materialize_gas_capacity(zone: str, start_day: str, end_day: str, output_dir: str | Path) -> dict[str, Any]:
    """Collect disjoint natural-gas fuel aggregate, not a sum across type/fuel axes."""
    if zone not in {"FR", "DE", "BE", "NL"}:
        raise ValueError("Unsupported zone")
    expected_hours(start_day, end_day)
    root = isolated_path(output_dir) / "capacities"
    root.mkdir(parents=True, exist_ok=True)
    series = f"power.nrjscan.{zone.lower()}.3mv.availability.pmax.fuel.nat_gas.gw"
    output = root / f"{zone.lower()}_gas_available_gw.parquet"
    audit_path = output.with_suffix(".parquet.audit.json")
    if output.exists() or audit_path.exists():
        if not output.is_file() or not audit_path.is_file():
            raise ValueError("Incomplete gas source bundle")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("sha256") != sha256(output) or audit.get("series") != series:
            raise ValueError("Invalid gas source bundle")
        if audit.get("start_day") == start_day and audit.get("end_day") == end_day:
            return {**audit, "reused": True}
        raise ValueError("Gas bundle date contract differs; use a new isolated output directory")
    client = _pooled_client()
    days = pd.date_range(start_day, end_day, freq="D")
    raw = client.block_staircase(series, from_value_date=days[0], to_value_date=days[-1],
                                 revision_freq={"days": 1}, revision_time={"hour": 8}, revision_tz=TIMEZONE,
                                 maturity_offset={"days": 1}, maturity_time={"hour": 0})
    daily = _daily_values(raw).reindex(days)
    if (daily.dropna() < 0).any() or np.isinf(daily).any():
        raise ValueError("Invalid natural-gas capacity")
    probes, errors = [], {}
    for text in PROBE_DAYS:
        day = pd.Timestamp(text)
        if day not in days:
            continue
        state = _one_day(client, series, day)
        bulk = daily.get(day, np.nan)
        if (state is None) != (not np.isfinite(bulk)) or (state is not None and not np.isclose(state, bulk, rtol=0, atol=1e-9)):
            raise ValueError(f"Natural-gas staircase/state mismatch at {day.date()}")
        probes.append({"day": text, "state_value_gw": state})
    for day in daily[daily.isna()].index:
        try:
            state = _one_day(client, series, day)
            if state is not None:
                daily.loc[day] = state
        except Exception as exc:
            errors[str(day.date())] = f"{type(exc).__name__}: {exc}"
    frames = []
    for day, value in daily.dropna().items():
        hours = expected_hours(str(day.date()), str(day.date()))
        frames.append(pd.DataFrame({"value_time_utc": hours, "snapshot_time_utc": civil_cutoff(day),
                                    "revision_time_utc": civil_cutoff(day), "value": value}))
    if not frames:
        raise ValueError("Natural-gas capacity has no usable PIT history")
    temporary = output.with_suffix(".tmp.parquet")
    pd.concat(frames, ignore_index=True).to_parquet(temporary, index=False)
    temporary.replace(output)
    audit = {"schema_version": 2, "sha256": sha256(output), "series": series, "zone": zone,
             "start_day": start_day, "end_day": end_day, "requested_days": len(days),
             "complete_days": int(daily.notna().sum()), "missing_days": [str(day.date()) for day in daily[daily.isna()].index],
             "cutoff_time": "08:00", "cutoff_timezone": TIMEZONE, "causality_violations": 0,
             "unit": "GW", "information_type": "capacity_forecast", "probes": probes, "errors": errors,
             "capacity_scope": "provider natural-gas fuel aggregate; national fleet coverage not attested",
             "transformation": "daily Pmax broadcast to each physical hour; no imputation of missing days",
             "type_split_observed": False, "do_not_add": ["type.ccgt", "type.gt", "type.chp"],
             "provider_revision_timestamp_available": False, "revision_time_semantics": "query_asof_cutoff",
             "production_pit_evidence": False, "national_coverage_attested": False, "promotion_eligible": False}
    _write_json(audit_path, audit)
    print(f"[Supply V2] {zone} gaz aggrege: {audit['complete_days']}/{len(days)} jours", flush=True)
    return audit


def load_supply_inputs(config: Mapping[str, Any], *, project_root: str | Path, start_day: str, end_day: str,
                       zones: Sequence[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load an explicit dynamic-column source map; never manufacture unknown MW.

    Qualification is deliberately separate from numerical completeness: an
    incomplete national fleet may have 100% timestamps yet remain unqualified.
    """
    root = Path(project_root).resolve()
    timezone = str(config.get("timezone", TIMEZONE))
    index = expected_hours(start_day, end_day, timezone)
    sources = {}
    if config.get("source_catalog"):
        import yaml
        path = Path(config["source_catalog"])
        path = path if path.is_absolute() else root / path
        catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
        sources.update(catalog["sources"])
    sources.update(config.get("sources", {}))
    audit, frames = {}, []
    selected = list(zones or config["zones"])
    for zone in selected:
        zcfg = config["zones"][zone]
        frame = pd.DataFrame({"delivery_start_utc": index, "zone": zone})
        frame["demand_basis"] = zcfg.get("demand_basis", "residual")
        if frame.demand_basis.iloc[0] == "residual" and not zcfg.get("residual_netting"):
            raise MarginalCostDataError("Residual demand requires explicit netting components")
        provenance = {}
        for field, mapping in zcfg["fields"].items():
            unit = mapping.get("unit")
            expected_unit = ("MW" if field.endswith("_mw") else
                             "EUR/MWh_th" if field.endswith("_eur_mwh_th") else
                             "EUR/tCO2" if field.endswith("_eur_tco2") else
                             "USD/EUR" if field == "eurusd_usd_per_eur" else None)
            if expected_unit is not None and unit != expected_unit:
                raise MarginalCostDataError(f"{zone}/{field}: output unit must be {expected_unit}")
            if "constant" in mapping:
                if not mapping.get("assumption"):
                    raise MarginalCostDataError("Unqualified physical/fuel constant forbidden")
                frame[field] = float(mapping["constant"])
                provenance[field] = {"assumption": mapping["assumption"], "unit": unit}
                continue
            alias = mapping["source"]
            spec = sources[alias]
            if spec["information_type"] != "market_observation_known_before_cutoff":
                validate_physical_identity(spec["series"])
            # V1 loader intentionally rejects Storm names. Do not mask series
            # identity to bypass its guard: V2 forecast-provider inputs require
            # their own reader before activation, even after qualification.
            if "storm" in spec.get("series", ""):
                raise MarginalCostDataError("Physical Storm-tagged forecast not yet activated; preserve exact provenance")
            values, source_audit = _load_source(alias, spec, root=root, expected=index, timezone=timezone,
                                                allow_missing_hours=bool(config.get("allow_missing_hours", False)))
            if spec.get("expected_audit"):
                source_path = Path(spec["path"])
                source_path = source_path if source_path.is_absolute() else root / source_path
                source_audit_path = Path(spec.get("audit_path", str(source_path) + ".audit.json"))
                source_audit_path = source_audit_path if source_audit_path.is_absolute() else root / source_audit_path
                source_contract = json.loads(source_audit_path.read_text(encoding="utf-8"))
                for key, expected_value in spec["expected_audit"].items():
                    if source_contract.get(key) != expected_value:
                        raise MarginalCostDataError(f"{alias}: source derivation contract mismatch for {key}")
            scale = mapping.get("scale", 1.0)
            source_unit = spec.get("unit")
            valid_units = {( "GW", "MW", 1000.0), ("MW", "MW", 1.0),
                           ("EUR/MWh_th", "EUR/MWh_th", 1.0), ("EUR/tCO2", "EUR/tCO2", 1.0),
                           ("USD/EUR", "USD/EUR", 1.0), ("USD/t", "USD/t", 1.0)}
            if (source_unit, unit, float(scale)) not in valid_units:
                raise MarginalCostDataError(f"Undeclared unit conversion {source_unit}->{unit}, scale={scale}")
            frame[field] = values.to_numpy() * float(scale)
            provenance[field] = {**source_audit, "source_alias": alias, "unit": unit, "scale": scale}
        numeric = list(zcfg["fields"])
        for field in numeric:
            values = pd.to_numeric(frame[field], errors="raise")
            if np.isinf(values).any():
                raise MarginalCostDataError(f"{zone}/{field}: infinite physical/fuel input")
            if field.endswith("_available_mw") and values.lt(0).any():
                raise MarginalCostDataError(f"{zone}/{field}: negative capacity")
        frame["inputs_complete"] = np.isfinite(frame[numeric]).all(axis=1)
        qualification = zcfg.get("qualification", {})
        qualified = qualification.get("national_fleet_attested") is True and qualification.get("historical_pit_attested") is True
        frame["inputs_qualified"] = frame.inputs_complete & qualified
        audit[zone] = {"fields": provenance, "qualification": qualification,
                       "complete_hours": int(frame.inputs_complete.sum()), "qualified_hours": int(frame.inputs_qualified.sum()),
                       "residual_netting": zcfg.get("residual_netting", [])}
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), {"schema_version": 2, "zones": audit,
                                                  "production_ready": False, "promotion_eligible": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["probe", "coal-fx", "gas-capacities"])
    parser.add_argument("--start-day", default="2024-09-10")
    parser.add_argument("--end-day", default="2026-09-09")
    parser.add_argument("--output-dir", default=str(V2_ROOT))
    parser.add_argument("--aliases", nargs="+")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--zones", nargs="+", default=["FR", "DE", "BE", "NL"])
    args = parser.parse_args(argv)
    if args.action == "probe":
        print(probe_sources(args.output_dir, aliases=args.aliases, workers=args.workers))
    elif args.action == "coal-fx":
        print(json.dumps(materialize_coal_fx(args.start_day, args.end_day, args.output_dir, workers=args.workers), indent=2))
    else:
        if not 1 <= args.workers <= 2:
            raise ValueError("At most two Saturn workers permitted")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(materialize_gas_capacity, z, args.start_day, args.end_day, args.output_dir) for z in args.zones]
            for future in as_completed(futures):
                print(json.dumps(future.result()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
