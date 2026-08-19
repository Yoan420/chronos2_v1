#!/usr/bin/env python
"""Audit non-Storm FR day-ahead price forecasts in Saturn without bulk download.

The audit reads the catalogue, metadata, revision timestamps, value interval,
and five one-day point-in-time snapshots.  It never downloads full histories.
Any series whose identifier, catalogue source, type, or selected metadata
contains ``storm`` is excluded before snapshot access.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import threading
from typing import Any

import numpy as np
import pandas as pd

from chronos2_modular.saturn import create_saturn_client


TIMEZONE = "Europe/Paris"
LOCAL = threading.local()
SNAPSHOT_DAYS = (
    "2024-01-15",
    "2024-08-01",
    "2025-05-15",
    "2025-07-15",
    "2026-08-11",
)
SAFE_METADATA_KEYS = (
    "provider",
    "vendor",
    "source",
    "owner",
    "author",
    "description",
    "license",
    "licence",
    "copyright",
    "supervision_status",
    "tzaware",
    "index_type",
    "value_type",
)


def _client(args: argparse.Namespace):
    client = getattr(LOCAL, "client", None)
    if client is None:
        client = create_saturn_client(args.saturn_url, args.author)
        LOCAL.client = client
    return client


def _candidate(name: str) -> bool:
    low = name.lower()
    france = bool(re.search(r"(^|[._-])fr([._-]|$)|france|french", low))
    price = "price" in low or "eurmwh" in low or "eur.mwh" in low
    forecast = any(
        token in low
        for token in ("fcst", "forecast", "flowspot", "opti-price", "model")
    )
    hourly = any(token in low for token in (".h.", "hour", "hourly", ".1h"))
    return france and price and forecast and hourly and "power" in low


def _provider(name: str) -> str:
    low = name.lower()
    for token, label in (
        ("pointcarbon", "Point Carbon"),
        ("opti-price", "Opti-price"),
        ("mkonline", "MKOnline"),
        ("kpler", "Kpler"),
        ("3mv", "3MV / Opti-price"),
    ):
        if token in low:
            return label
    return "unknown"


def _nature(name: str, catalog_type: str) -> str:
    low = name.lower()
    tags: list[str] = [catalog_type]
    if ".cache" in low:
        tags.append("historical cache")
    if "flowspot" in low:
        tags.append("flow-based spot price forecast")
    elif "opti-price" in low:
        tags.append("optimisation price forecast")
    elif "ecop" in low:
        tags.append("deterministic EC operational price forecast")
    elif "phx" in low:
        tags.append("PHX day-ahead price forecast")
    else:
        tags.append("day-ahead price forecast")
    if "highload" in low:
        tags.append("high-load scenario")
    if "lowload" in low:
        tags.append("low-load scenario")
    if ".pnl" in low or ".pnl." in low:
        tags.append("PnL derivative; not a direct forecast candidate")
    if ".old" in low:
        tags.append("legacy/old")
    return "; ".join(tags)


def _safe_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    out: dict[str, Any] = {}
    for key in SAFE_METADATA_KEYS:
        if key in metadata:
            value = metadata[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
            else:
                out[key] = str(value)
    return out


def _license_status(metadata: dict[str, Any], provider: str) -> str:
    declared = next(
        (
            str(metadata[key])
            for key in ("license", "licence", "copyright")
            if metadata.get(key) not in (None, "")
        ),
        None,
    )
    if declared:
        return declared
    return (
        f"not declared in Saturn metadata; vendor/internal entitlement for {provider} "
        "must be confirmed before production use"
    )


def _day_bounds(day_raw: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    day = pd.Timestamp(day_raw)
    start = day.tz_localize(TIMEZONE)
    end = (day + pd.Timedelta(days=1)).tz_localize(TIMEZONE)
    cutoff = (
        day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    ).tz_localize(TIMEZONE)
    return start.tz_convert("UTC"), end.tz_convert("UTC"), cutoff.tz_convert("UTC")


def _snapshot(client: Any, name: str, day: str) -> dict[str, Any]:
    start, end, cutoff = _day_bounds(day)
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    try:
        series = client.get(
            name,
            revision_date=cutoff,
            from_value_date=start - pd.Timedelta(hours=2),
            to_value_date=end + pd.Timedelta(hours=2),
            _keep_nans=True,
        )
        if series is None:
            return {
                "day": day,
                "cutoff_utc": str(cutoff),
                "status": "missing",
                "coverage": 0.0,
                "n_expected": int(len(expected)),
                "n_finite": 0,
            }
        values = pd.to_numeric(series, errors="coerce")
        index = pd.DatetimeIndex(series.index)
        if index.tz is None:
            index = index.tz_localize(TIMEZONE).tz_convert("UTC")
            timezone_contract = "naive interpreted Europe/Paris"
        else:
            index = index.tz_convert("UTC")
            timezone_contract = str(series.index.tz)
        normalized = pd.Series(values.to_numpy(dtype=float), index=index)
        if normalized.index.has_duplicates:
            normalized = normalized.groupby(level=0).last()
        selected = normalized.reindex(expected)
        finite = np.isfinite(selected.to_numpy(dtype=float))
        return {
            "day": day,
            "cutoff_utc": str(cutoff),
            "status": "complete" if bool(finite.all()) else "incomplete",
            "coverage": float(finite.mean()),
            "n_expected": int(len(expected)),
            "n_finite": int(finite.sum()),
            "timezone_contract": timezone_contract,
            "first_returned_utc": str(index.min()) if len(index) else None,
            "last_returned_utc": str(index.max()) if len(index) else None,
        }
    except Exception as exc:
        return {
            "day": day,
            "cutoff_utc": str(cutoff),
            "status": "error",
            "coverage": 0.0,
            "n_expected": int(len(expected)),
            "n_finite": 0,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }


def _audit_one(
    name: str,
    catalog_source: str,
    catalog_type: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    client = _client(args)
    metadata = _safe_metadata(client.metadata(name, all=True))
    guard_text = " ".join(
        [name, catalog_source, catalog_type, *[str(value) for value in metadata.values()]]
    ).lower()
    if "storm" in guard_text:
        return {
            "series": name,
            "status": "excluded_storm_guard",
        }
    revisions = client.insertion_dates(
        name,
        from_insertion_date=pd.Timestamp("2023-12-31", tz="UTC"),
        to_insertion_date=pd.Timestamp("2026-08-12", tz="UTC"),
    )
    revisions = [] if revisions is None else list(revisions)
    # Formula-backed Saturn series can expose ``interval`` as a raw HTTP
    # response rather than the interval object returned for primary series.
    # Interval metadata is informative only: never let it suppress the PIT
    # snapshot audit that determines whether a candidate is usable.
    try:
        interval_raw = client.interval(name)
        interval = str(interval_raw) if interval_raw is not None else None
        interval_error = None
    except Exception as exc:
        interval = None
        interval_error = f"{type(exc).__name__}: {str(exc)[:300]}"
    provider = _provider(name)
    snapshots = [_snapshot(client, name, day) for day in SNAPSHOT_DAYS]
    complete = sum(item["coverage"] >= 0.999 for item in snapshots)
    minimum = min(float(item["coverage"]) for item in snapshots)
    return {
        "series": name,
        "status": "audited",
        "catalog_source": catalog_source,
        "catalog_type": catalog_type,
        "provider_inferred_from_identifier": provider,
        "nature": _nature(name, catalog_type),
        "metadata": metadata,
        "license_status": _license_status(metadata, provider),
        "revision_count_2023_12_31_to_2026_08_12": int(len(revisions)),
        "first_revision_utc": str(min(revisions)) if revisions else None,
        "last_revision_utc": str(max(revisions)) if revisions else None,
        "value_interval": interval,
        "value_interval_error": interval_error,
        "snapshots": snapshots,
        "complete_snapshot_count": int(complete),
        "minimum_snapshot_coverage": minimum,
        "coverage_gate_passed": bool(complete == len(SNAPSHOT_DAYS)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saturn-url",
        default="https://saturn-energyscan.gem.myengie.com//api",
    )
    parser.add_argument("--author", default="BQ6757")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional exact series identifiers to audit after catalogue filtering.",
    )
    parser.add_argument(
        "--output",
        default="runs/tmp/saturn_fr_price_nonstorm/catalog_audit.json",
    )
    args = parser.parse_args()
    catalogue = create_saturn_client(args.saturn_url, args.author).catalog()
    requested = set(args.only or [])
    candidates: list[tuple[str, str, str]] = []
    excluded_storm_names: list[str] = []
    for source, items in catalogue.items():
        source_label = str(source)
        for item in items:
            name = str(item[0])
            catalog_type = str(item[1]) if len(item) > 1 else "unknown"
            # Explicit exact identifiers extend the heuristic discovery set;
            # they still pass the same hard Storm guard before any access.
            if not _candidate(name) and name not in requested:
                continue
            if "storm" in f"{source_label} {catalog_type} {name}".lower():
                excluded_storm_names.append(name)
                continue
            candidates.append((name, source_label, catalog_type))
    candidates.sort(key=lambda values: values[0].lower())
    if args.only:
        candidates = [item for item in candidates if item[0] in requested]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(_audit_one, name, source, kind, args): name
            for name, source, kind in candidates
        }
        for number, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "series": name,
                    "status": "audit_error",
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
            results.append(result)
            print(
                f"AUDIT {number}/{len(candidates)} | {name} | "
                f"{result.get('status')} | gate={result.get('coverage_gate_passed')}",
                flush=True,
            )
    results.sort(key=lambda item: str(item["series"]).lower())
    passed = [
        item["series"]
        for item in results
        if item.get("coverage_gate_passed") is True
    ]
    payload = {
        "protocol": {
            "catalogue_read_only": True,
            "bulk_history_downloaded": False,
            "snapshot_days": list(SNAPSHOT_DAYS),
            "forecast_origin": "D-1 08:00 Europe/Paris civil time",
            "storm_guard": "case-insensitive exclusion on identifier/source/type/selected metadata",
            "external_price_forecasts": True,
            "autonomous_model": False,
        },
        "candidate_count": len(candidates),
        "storm_names_excluded_before_audit": sorted(excluded_storm_names),
        "coverage_gate_passed": passed,
        "candidates": results,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "candidate_count": len(candidates),
        "coverage_gate_passed": passed,
        "storm_names_excluded_count": len(excluded_storm_names),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
