"""Read-only, DST-strict adapters for an economic-value research snapshot.

Published forecasts are not retroactively certified point-in-time evidence.
Storm remains a forecast comparator, never an executable market quotation.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from marginal_cost_expert.evaluation import _array, _missing_pair_hours, _trace_index, physical_index


class EconomicDataError(ValueError):
    pass


MODEL_LABELS = {
    "autonomous": "Correcteur résiduel P50",
    "kalman": "Correcteur résiduel + Kalman gouverné P50",
    "nuclear_autonomous": "Chronos-2 + nucl. + correcteur P50",
    "nuclear_kalman": "Chronos-2 + nucl. + correcteur + Kalman P50",
}
TARGET_PATHS = {
    "FR": "data/cache/fr/target__5dcf0bdf8c.csv.gz",
    "DE": "data/cache/de/target__48c616a566.csv.gz",
    "BE": "data/cache/be/target__b9c1dde640.csv.gz",
    "NL": "data/cache/nl/target__57b345e449.csv.gz",
}


def _stable_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Reject a source rewritten while a report is being captured."""
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise EconomicDataError(f"Source modified during capture; retry after its writer finishes: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise EconomicDataError(f"Source content changed during capture: {path}")
    return raw, {"path": str(path.resolve()), "sha256": digest,
                 "modified_at_utc": pd.Timestamp(after.st_mtime, unit="s", tz="UTC").isoformat()}


def _day(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise EconomicDataError("Dates must be local calendar dates YYYY-MM-DD.")
    return str(pd.Timestamp(value).date())


def _aware(values: Any, name: str) -> pd.DatetimeIndex:
    stamps = [pd.Timestamp(v) for v in values]
    if any(pd.isna(v) or v.tzinfo is None for v in stamps):
        raise EconomicDataError(f"{name}: explicit timezone-aware timestamps required.")
    return pd.DatetimeIndex(pd.to_datetime(stamps, utc=True))


def _plots(document: str):
    """Decode JSON plot arguments only. Never evaluate embedded JavaScript."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"Plotly\.newPlot\(", document):
        tail = document[match.end():].lstrip()
        try:
            _, end = decoder.raw_decode(tail)
            tail = tail[end:].lstrip()
            if not tail.startswith(","):
                continue
            traces, _ = decoder.raw_decode(tail[1:].lstrip())
        except (ValueError, TypeError):
            continue
        if isinstance(traces, list) and all(isinstance(t, dict) for t in traces):
            yield traces


def _gap_evidence(path: Path, zone: str) -> tuple[pd.DatetimeIndex, list[dict]]:
    missing, evidence = _missing_pair_hours(path, zone, None)
    if len(missing):
        return missing, evidence
    # Autonomous exports do not carry an audit of their own. The same-day,
    # same-country Kalman export points to the common statistics archive.
    sibling = path.parent.parent / "kalman" / path.name.replace("_autonomous.html", "_kalman.html")
    if path.parent.name == "autonomous" and sibling.is_file():
        missing, extra = _missing_pair_hours(sibling, zone, None)
        evidence += extra
    return missing, evidence


def _series(trace: dict, timezone: str, missing: pd.DatetimeIndex | None = None) -> pd.Series:
    index = _trace_index(trace["x"], timezone=timezone, missing=missing)
    values = _array(trace["y"])
    if len(values) != len(index) or np.isinf(values).any():
        raise EconomicDataError("Invalid numeric trace length or infinite prices.")
    return pd.Series(values, index=index)


def read_report(path: str | Path, *, zone: str, model: str,
                timezone: str = "Europe/Paris") -> tuple[pd.DataFrame, dict]:
    """Read full, paired, and delivery traces, preserving published quantiles."""
    path = Path(path).resolve()
    if model not in MODEL_LABELS:
        raise EconomicDataError(f"Unknown model: {model}")
    raw, source = _stable_bytes(path)
    missing, evidence = _gap_evidence(path, zone)
    label = MODEL_LABELS[model]
    histories, pairs, forecasts, observations = [], [], [], []
    for traces in _plots(raw.decode("utf-8-sig")):
        selected = [t for t in traces if t.get("name") == label and len(t.get("x", [])) >= 23]
        if not selected:
            continue
        if len(selected) != 1:
            raise EconomicDataError("Ambiguous named forecast trace.")
        storm = [t for t in traces if str(t.get("name", "")).startswith("Storm officiel")
                 and str(t.get("name", "")).endswith("P50")]
        if len(storm) > 1:
            raise EconomicDataError("Ambiguous Storm forecast trace.")
        gap = missing if storm else None
        main = _series(selected[0], timezone, gap)
        block = pd.DataFrame({"forecast": main, "q10": np.nan, "q90": np.nan,
                              "actual": np.nan, "benchmark_forecast": np.nan})
        for field, names in (("q10", {"P10", "P10–P90", "Intervalle P10–P90"}), ("q90", {"P90"})):
            quantiles = [t for t in traces if t.get("name") in names]
            if len(quantiles) > 1:
                raise EconomicDataError(f"Ambiguous {field} trace.")
            if quantiles:
                quantile = _series(quantiles[0], timezone, gap)
                if not quantile.index.equals(main.index):
                    raise EconomicDataError(f"{field} and forecast axes disagree.")
                block[field] = quantile
        observed = [t for t in traces if t.get("name") in {"Observé", "Prix observé"}]
        if len(observed) > 1:
            raise EconomicDataError("Ambiguous observed trace.")
        if observed:
            actual = _series(observed[0], timezone, gap)
            block["actual"] = actual.reindex(main.index)
            observations.append(actual)
        if storm:
            comparator = _series(storm[0], timezone, gap)
            if not comparator.index.equals(main.index):
                raise EconomicDataError("Storm and model axes disagree.")
            block["benchmark_forecast"] = comparator
            pairs.append(block)
        elif len(main) > 168:
            histories.append(block)
        else:
            forecasts.append(block)
    if not histories:
        raise EconomicDataError(f"No full hourly backtest trace for {zone}/{model}: {path}")
    result = max(histories, key=len).copy()
    # Paired plots contain refreshed observations; preserve their values.
    for block in sorted(histories, key=len, reverse=True)[1:] + pairs + forecasts:
        common = result.index.intersection(block.index)
        lhs, rhs = result.loc[common, "forecast"], block.loc[common, "forecast"]
        valid = lhs.notna() & rhs.notna()
        if not np.allclose(lhs[valid], rhs[valid], atol=1e-6, rtol=0):
            raise EconomicDataError("Report forecast traces disagree on overlapping physical hours.")
        result = block.combine_first(result)
    if observations:
        # Recent observed context in the delivery chart may refresh labels,
        # but is never used as a predictor or as a market quote.
        for actual in observations:
            valid = actual.index.intersection(result.index)
            result.loc[valid, "actual"] = actual.reindex(valid).combine_first(result.loc[valid, "actual"])
    csv = path.with_suffix(".csv")
    csv_source = None
    if csv.is_file():
        csv_raw, csv_source = _stable_bytes(csv)
        frame = pd.read_csv(BytesIO(csv_raw))
        required = {"delivery_start_utc", "q50", "zone"}
        if required.difference(frame) or not frame.zone.astype(str).eq(zone).all():
            raise EconomicDataError(f"Export CSV schema/zone mismatch: {csv}")
        index = _aware(frame.delivery_start_utc, "CSV delivery")
        if index.has_duplicates or not index.equals(index.floor("h")):
            raise EconomicDataError("Export CSV is not unique physical hourly data.")
        expected_day = path.parents[2].name
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", expected_day) or not index.equals(
                physical_index(expected_day, expected_day, timezone)):
            raise EconomicDataError("Export CSV must contain exactly its declared complete delivery day.")
        current = pd.DataFrame(index=index)
        for dest, src in (("forecast", "q50"), ("q10", "q10"), ("q90", "q90")):
            current[dest] = pd.to_numeric(frame[src], errors="raise").to_numpy() if src in frame else np.nan
        if np.isinf(current.to_numpy(float)).any():
            raise EconomicDataError("Infinite delivery forecast or quantile.")
        if "forecast_variant" in frame and not frame.forecast_variant.astype(str).eq(model).all():
            raise EconomicDataError("CSV forecast variant differs from its report directory.")
        common = current.index.intersection(result.index)
        if not np.allclose(result.loc[common, "forecast"], current.loc[common, "forecast"], atol=1e-5, rtol=0):
            raise EconomicDataError("HTML and delivery CSV forecasts differ; publication may be in progress.")
        # Keep the report's forecast values (and comparison rounding) intact.
        result = result.combine_first(current)
    invalid_q = result.q10.notna() & result.q90.notna() & (
        result.q10.gt(result.q90) | result.q10.gt(result.forecast) | result.q90.lt(result.forecast))
    result.loc[invalid_q, ["q10", "q90"]] = np.nan
    result = result.sort_index().rename_axis("timestamp_utc").reset_index()
    source.update({"zone": zone, "model": model, "label": label, "time_axis_audits": evidence,
                   "allowed_missing_pair_utc": [v.isoformat() for v in missing],
                   "csv_source": csv_source, "quantile_source": "published_P10_P90_traces_not_reconstructed",
                   "invalid_quantile_rows_discarded": int(invalid_q.sum()),
                   "forecast_pit_certified": False, "benchmark_pit_certified": False,
                   "provenance": "published_report_replay_and_latest_dashboard_not_certified_execution_vintages"})
    return result, source


def _complete_last_day(frame: pd.DataFrame, timezone: str) -> str:
    local_days = frame.timestamp_utc.dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
    for day in sorted(local_days.unique(), reverse=True):
        block = frame.loc[local_days.eq(day)].set_index("timestamp_utc")
        index = physical_index(day, day, timezone)
        if block.index.equals(index) and np.isfinite(block[["forecast", "actual"]].to_numpy(float)).all():
            return day
    raise EconomicDataError("No complete observed forecast day in the selected report.")


def _origin(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    local = index.tz_convert(timezone).tz_localize(None).normalize()
    return (local - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(timezone).tz_convert("UTC")


def load_report_panel(root: str | Path, zones: Sequence[str], models: Sequence[str],
                      delivery_day: str | None = None, end_day: str | None = None,
                      *, timezone: str = "Europe/Paris") -> tuple[pd.DataFrame, dict]:
    """Freeze one COMMON export date and a common last-365-calendar-day window.

    Missing hours remain rows with NaN, not shortened observation-count windows.
    Current-day rows are in ``audit['live_rows']`` even when already evaluated.
    Future rows are additionally returned with ``sample='live'``.
    """
    root = Path(root).resolve()
    zones, models = [str(z).upper() for z in zones], list(models)
    if not zones or not models or len(zones) != len(set(zones)) or len(models) != len(set(models)):
        raise EconomicDataError("Choose nonempty unique countries and models.")
    if set(zones) - set(TARGET_PATHS) or set(models) - set(MODEL_LABELS):
        raise EconomicDataError("Unsupported country or model.")
    exports = root / "runs" / "exports"
    availability, common = {}, None
    for zone in zones:
        for model in models:
            paths = {p.parents[2].name: p for p in exports.glob(
                f"????-??-??/{zone.lower()}/{model}/forecast_{zone.lower()}_????-??-??_{model}.html")}
            paths = {day: path for day, path in paths.items() if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day)}
            availability[(zone, model)] = paths
            common = set(paths) if common is None else common.intersection(paths)
    if not common:
        raise EconomicDataError("No common delivery-date export for all selected models/countries.")
    chosen = _day(delivery_day) if delivery_day is not None else max(common)
    if chosen not in common:
        absent = [f"{z}/{m}" for (z, m), paths in availability.items() if chosen not in paths]
        raise EconomicDataError(f"Incomplete export {chosen}: {', '.join(absent)}")
    loaded, audits, complete = {}, [], []
    for key, paths in availability.items():
        frame, audit = read_report(paths[chosen], zone=key[0], model=key[1], timezone=timezone)
        loaded[key] = frame
        audit["last_complete_observed_day"] = _complete_last_day(frame, timezone)
        audits.append(audit)
        complete.append(audit["last_complete_observed_day"])
    latest = min(complete)
    final = _day(end_day) if end_day is not None else latest
    if final > latest:
        raise EconomicDataError(f"Evaluation end {final} exceeds common complete observed day {latest}.")
    first = str((pd.Timestamp(final) - pd.Timedelta(days=364)).date())
    index = physical_index(first, final, timezone)
    outputs, current, shared_audits = [], [], {}
    # One frozen label/comparator per country, never a different settlement
    # price for each competing strategy. Latest report publication has priority.
    for zone in zones:
        ordered = sorted([a for a in audits if a["zone"] == zone], key=lambda a: (a["modified_at_utc"], a["model"]))
        labels, storm = pd.Series(dtype=float), pd.Series(dtype=float)
        for item in ordered:
            block = loaded[(zone, item["model"])].set_index("timestamp_utc")
            labels = block.actual.combine_first(labels)
            storm = block.benchmark_forecast.combine_first(storm)
        shared_audits[zone] = {"label_policy": "latest_selected_report_publication_then_older_missing_fallback",
                              "priority_reports": [a["path"] for a in reversed(ordered)],
                              "settlement_shared_across_models": True,
                              "storm_is_forecast_not_execution_price": True,
                              "actual_revision_differences_by_model": {},
                              "storm_revision_differences_by_model": {}}
        for model in models:
            source = loaded[(zone, model)].set_index("timestamp_utc")
            compared = source.actual.sub(labels.reindex(source.index)).abs()
            shared_audits[zone]["actual_revision_differences_by_model"][model] = int(compared.gt(1e-6).sum())
            compared_storm = source.benchmark_forecast.sub(storm.reindex(source.index)).abs()
            shared_audits[zone]["storm_revision_differences_by_model"][model] = int(compared_storm.gt(1e-6).sum())
            source["actual"] = labels.reindex(source.index)
            source["benchmark_forecast"] = storm.reindex(source.index)
            block = source.reindex(index).copy()
            block["sample"] = "evaluation"
            day_index = physical_index(chosen, chosen, timezone)
            day_block = source.reindex(day_index).copy()
            if chosen > final:
                future = day_block.copy()
                future["sample"] = "live"
                block = pd.concat([block, future])
            for target in (block, day_block):
                target["zone"], target["model"] = zone, model
                target["duration_hours"] = 1.0
                target["forecast_origin_utc"] = _origin(target.index, timezone)
                target["forecast_eligible"] = np.isfinite(target.forecast)
                target["forecast_pit_certified"] = False
                target["benchmark_pit_certified"] = False
            day_block["sample"] = "live"
            current.append(day_block.rename_axis("timestamp_utc").reset_index())
            outputs.append(block.rename_axis("timestamp_utc").reset_index())
    panel = pd.concat(outputs, ignore_index=True).sort_values(["timestamp_utc", "zone", "model"]).reset_index(drop=True)
    if panel.duplicated(["timestamp_utc", "zone", "model"]).any():
        raise EconomicDataError("Duplicate panel identities.")
    # Validate every source again after all reads to detect a concurrent batch
    # spanning more than one file capture (there is no production write lock).
    for item in audits:
        for evidence in [item, item.get("csv_source")] + item.get("time_axis_audits", []):
            if evidence and evidence.get("path") and evidence.get("sha256"):
                if hashlib.sha256(Path(evidence["path"]).read_bytes()).hexdigest() != evidence["sha256"]:
                    raise EconomicDataError(f"Source changed during panel assembly: {evidence['path']}")
    audit = {"schema_version": 1, "delivery_day": chosen, "selection": "latest_common_export_date",
             "latest_export_by_model": {f"{z}/{m}": max(paths) for (z, m), paths in availability.items()},
             "evaluation_start_day": first, "evaluation_end_day": final, "evaluation_days": 365,
             "physical_hours_per_model": len(index), "timezone": timezone,
             "forecast_cutoff": "D-1 08:00 civil (assumed, not per-row issue evidence)",
             "forecast_pit_certified": False, "benchmark_pit_certified": False,
             "diagnostic_only": True, "production_modified": False, "sources": audits,
             "shared_observations": shared_audits,
             "missing_evaluation_forecasts": int(panel.loc[panel["sample"].eq("evaluation"), "forecast"].isna().sum()),
             "live_rows": json.loads(pd.concat(current, ignore_index=True).to_json(orient="records", date_format="iso"))}
    return panel, audit


def load_reference_proxy(root: str | Path, panel: pd.DataFrame, *, timezone: str = "Europe/Paris",
                         target_paths: Mapping[str, str | Path] | None = None) -> tuple[pd.DataFrame, dict]:
    """Previous CIVIL day's same-hour DA price; not a tradable quote.

    Both ambiguous source hours and missing DST hours abstain (no averaging,
    UTC-minus-24 substitution, interpolation, or same-day actual fallback).
    Latest target revisions do NOT prove the historical publication vintage.
    """
    root = Path(root).resolve()
    paths = dict(TARGET_PATHS if target_paths is None else target_paths)
    if {"timestamp_utc", "zone"}.difference(panel):
        raise EconomicDataError("Proxy requires timestamp_utc and zone.")
    timestamps = _aware(panel.timestamp_utc, "Panel")
    keys = pd.DataFrame({"timestamp_utc": timestamps, "zone": panel.zone.to_numpy()}).drop_duplicates()
    if not timestamps.equals(timestamps.floor("h")):
        raise EconomicDataError("Only physical hourly delivery is supported.")
    output, source_audits = [], {}
    for zone, block in keys.groupby("zone", sort=True):
        if zone not in paths:
            raise EconomicDataError(f"No explicitly configured canonical target for {zone}.")
        path = Path(paths[zone])
        path = path if path.is_absolute() else root / path
        raw, audit = _stable_bytes(path)
        target = pd.read_csv(BytesIO(raw), compression="gzip" if path.suffix == ".gz" else None)
        if set(target) != {"timestamp", "value"}:
            raise EconomicDataError("Unexpected canonical target schema.")
        target_index = _aware(target.timestamp, "Target")
        numeric = pd.to_numeric(target.value, errors="raise").to_numpy(float)
        if target_index.has_duplicates or not target_index.equals(target_index.floor("h")) or np.isinf(numeric).any():
            raise EconomicDataError("Invalid canonical target physical grid/values.")
        target_local = target_index.tz_convert(timezone).tz_localize(None)
        # Count by the local identity before lookup, including repeated autumn
        # hours whose prices happen to be equal: identity remains ambiguous.
        counts = pd.Series(1, index=target_local).groupby(level=0).sum()
        unique = ~target_local.duplicated(keep=False)
        lookup = pd.Series(numeric[unique], index=target_local[unique])
        time_lookup = pd.Series(target_index[unique], index=target_local[unique])
        index = pd.DatetimeIndex(block.timestamp_utc)
        source_local = index.tz_convert(timezone).tz_localize(None) - pd.Timedelta(days=1)
        values = lookup.reindex(source_local).to_numpy()
        source_time = time_lookup.reindex(source_local).to_numpy()
        available_local = source_local.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
        available = available_local.tz_localize(timezone).tz_convert("UTC")
        result = block.copy()
        result["reference_price"] = values
        result["reference_source_timestamp_utc"] = pd.to_datetime(source_time, utc=True)
        result["reference_available_at_utc"] = available
        result["reference_eligible"] = np.isfinite(values)
        result["reference_pit_certified"] = False
        result["availability_assumed"] = True
        result["reference_kind"] = "lagged_day_ahead_proxy"
        result["reference_missing_reason"] = np.where(
            np.isfinite(values), "", np.where(counts.reindex(source_local).fillna(0).to_numpy() > 1,
                                              "ambiguous_previous_civil_hour", "missing_previous_civil_hour"))
        output.append(result)
        audit.update({"missing_reference_hours": int((~np.isfinite(values)).sum()),
                      "dst_policy": "abstain_when_previous_local_hour_missing_or_ambiguous",
                      "latest_revision_cache": True})
        source_audits[zone] = audit
    return pd.concat(output, ignore_index=True), {
        "kind": "lagged_day_ahead_proxy", "executable_reference": False,
        "reference_pit_certified": False, "availability_assumed": True,
        "publication_time_assumption": "source delivery D-1: source D-2 18:00 civil, not observed issue evidence",
        "known_source_price_is_not_an_executable_quote_for_next_delivery": True,
        "sources": source_audits,
    }
