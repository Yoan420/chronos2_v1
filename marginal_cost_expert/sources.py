"""Read-only Saturn collection into the isolated marginal-cost research bank.

Daily Pmax is an available-capacity forecast, not realised production and not
an hourly availability curve. Broadcasting it is an explicit approximation.
The bulk block-staircase request uses the same D-1 civil 08:00 contract as
independent state-query probes. Missing days are queried individually; no
previous finite value or observed series substitutes for an absent vintage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://saturn-energyscan.gem.myengie.com//api"
TIMEZONE = "Europe/Paris"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allowed_output(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    allowed = ((ROOT / "data/pit/marginal_cost_expert").resolve(),
               (ROOT / "runs/experiments/marginal_cost_expert_v1").resolve())
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise ValueError("Marginal-cost collection must stay in its isolated research namespace")
    return resolved


def capacity_catalog(zones: list[str]) -> dict[str, dict[str, str]]:
    out = {}
    for zone in zones:
        z = zone.lower()
        for kind in ("ccgt", "gt"):
            out[f"{z}_{kind}_available_gw"] = {
                "series": f"power.nrjscan.{z}.3mv.availability.pmax.type.{kind}.gw",
                "zone": zone, "role": "capacity_forecast", "unit": "GW",
            }
        if zone in {"BE", "NL"}:
            out[f"{z}_nuclear_available_gw"] = {
                "series": f"power.nrjscan.{z}.3mv.availability.pmax.type.nuclear.gw",
                "zone": zone, "role": "capacity_forecast", "unit": "GW",
            }
        if zone in {"FR", "DE", "NL"}:
            for kind in (("coal", "lignite") if zone == "DE" else ("coal",)):
                out[f"{z}_{kind}_available_gw"] = {
                    "series": f"power.nrjscan.{z}.3mv.availability.pmax.fuel.{kind}.gw",
                    "zone": zone, "role": "capacity_forecast", "unit": "GW",
                }
    return out


def snapshot_residual_formulas(output_dir: str | Path, zones: list[str] | None = None) -> Path:
    """Archive current provider formulas as semantic evidence, not vintage proof."""
    import requests
    root = allowed_output(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for zone in zones or ["FR", "DE", "BE", "NL"]:
        series = f"power.{zone.lower()}.residual.load.hourly.gw.fcst"
        response = requests.get(URL + "/series/formula", params={"name": series}, timeout=40)
        response.raise_for_status()
        formula = response.json()
        if not isinstance(formula, str) or not formula.startswith("("):
            raise ValueError(f"Provider formula not returned for {series}")
        records[zone] = {"series": series, "formula": formula,
                         "formula_sha256": hashlib.sha256(formula.encode("utf-8")).hexdigest()}
    path = root / "residual_formula_evidence.json"
    _write_json(path, {"schema_version": 1, "endpoint": URL + "/series/formula",
                       "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                       "historical_formula_version_attested": False, "zones": records})
    return path


def civil_cutoff(day: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(day).normalize() - pd.Timedelta(days=1)
            + pd.Timedelta(hours=8)).tz_localize(TIMEZONE).tz_convert("UTC")


def _client():
    from tshistory_lite import Client
    client = Client(URL, author="BQ6757")
    client.session.request = partial(client.session.request, timeout=55)
    return client


def _daily_values(raw: Any) -> pd.Series:
    if raw is None:
        return pd.Series(dtype=float)
    if not isinstance(raw, pd.Series):
        raise ValueError("Saturn returned a non-Series capacity response")
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        raise ValueError("Daily CET capacity contract unexpectedly became timezone-aware")
    if not idx.equals(idx.normalize()) or idx.has_duplicates:
        raise ValueError("Daily capacity source must have unique local midnight labels")
    return pd.Series(pd.to_numeric(raw, errors="coerce").to_numpy(float), index=idx).sort_index()


def _one_day(client: Any, series: str, day: pd.Timestamp) -> float | None:
    raw = client.get(series, from_value_date=day - pd.Timedelta(days=1),
                     to_value_date=day + pd.Timedelta(days=1), revision_date=civil_cutoff(day))
    daily = _daily_values(raw)
    value = daily.get(day, np.nan)
    if not np.isfinite(value):
        return None
    if value < 0:
        raise ValueError("Negative available capacity returned by Saturn")
    return float(value)


def _write_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _existing(output: Path, audit_path: Path, series: str) -> pd.Series:
    if not output.exists() and not audit_path.exists():
        return pd.Series(dtype=float)
    if not output.is_file() or not audit_path.is_file():
        raise ValueError(f"Incomplete isolated source bundle: {output}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("sha256") != sha256(output) or audit.get("series") != series:
        raise ValueError(f"Invalid isolated capacity cache: {output}")
    if audit.get("cutoff_time") != "08:00" or audit.get("cutoff_timezone") != TIMEZONE:
        raise ValueError("Capacity cache cutoff contract changed")
    frame = pd.read_parquet(output)
    days = pd.to_datetime(frame.value_time_utc, utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
    grouped = frame.groupby(days).value
    if grouped.nunique(dropna=False).gt(1).any():
        raise ValueError("Daily broadcast cache contains multiple daily values")
    return grouped.first().astype(float)


def materialize_capacity(alias: str, spec: dict[str, str], *, start_day: str,
                         end_day: str, output_dir: Path) -> dict[str, Any]:
    output_dir = allowed_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{alias}.parquet"
    audit_path = output.with_suffix(".parquet.audit.json")
    start, end = pd.Timestamp(start_day), pd.Timestamp(end_day)
    if start.tzinfo is not None or end.tzinfo is not None or start != start.normalize() or end != end.normalize() or start > end:
        raise ValueError("Expected inclusive naive civil start/end days")
    days = pd.date_range(start, end, freq="D")
    existing = _existing(output, audit_path, spec["series"])
    missing = days.difference(existing.index)
    if not len(missing):
        return {"alias": alias, "status": "reused", "days": len(days), "path": str(output)}
    begun = time.monotonic()
    client = _client()
    request = {"revision_freq": {"days": 1}, "revision_time": {"hour": 8},
               "revision_tz": TIMEZONE, "maturity_offset": {"days": 1}, "maturity_time": {"hour": 0}}
    errors: dict[str, str] = {}
    bulk_error = None
    probes = []
    try:
        raw = client.block_staircase(spec["series"], from_value_date=missing.min(),
                                     to_value_date=missing.max(), **request)
        bulk = _daily_values(raw).reindex(missing)
        # Independent state checks cover both endpoints and internal origins,
        # including DST-adjacent origins where those fall in this request.
        sample = sorted(set([missing[0], missing[len(missing)//2], missing[-1]]) |
                        {d for d in missing if d.month in {3, 10} and d.weekday() == 0 and d.day >= 25})
        for day in sample:
            value = _one_day(client, spec["series"], day)
            b = bulk.get(day, np.nan)
            if value is not None and np.isfinite(b) and not np.isclose(value, b, rtol=0, atol=1e-9):
                raise ValueError(f"Block staircase disagrees with independent state at {day.date()}")
            probes.append({"day": str(day.date()), "bulk_value": float(b) if np.isfinite(b) else None,
                           "state_value": value})
            if value is None and np.isfinite(b):
                raise ValueError(f"Block staircase invents absent state at {day.date()}")
    except Exception as exc:
        bulk_error = f"{type(exc).__name__}: {exc}"
        bulk = pd.Series(np.nan, index=missing)
    selected = bulk[np.isfinite(bulk)].copy()
    if (selected < 0).any():
        raise ValueError("Negative available capacity in bulk response")
    absent = missing.difference(selected.index)
    for number, day in enumerate(absent, 1):
        value = None
        for attempt in range(3):
            try:
                value = _one_day(client, spec["series"], day)
                break
            except Exception as exc:
                errors[str(day.date())] = f"{type(exc).__name__}: {exc}"
                if attempt < 2:
                    time.sleep(1 + attempt)
        if value is not None:
            selected.loc[day] = value
            errors.pop(str(day.date()), None)
        else:
            errors.setdefault(str(day.date()), "No finite same-day value in state at D-1 08:00")
        if number % 30 == 0 or number == len(absent):
            print(f"[MarginalCost/{alias}] missing-vintage checks {number}/{len(absent)}; complete={len(selected)}/{len(missing)}", flush=True)
    merged = (selected.copy() if existing.empty else
              existing.copy() if selected.empty else pd.concat([existing, selected])).sort_index()
    if merged.index.has_duplicates:
        raise ValueError("Capacity update attempted to overwrite a valid existing day")
    blocks = []
    retrieved = pd.Timestamp.now(tz="UTC")
    for day, value in merged.items():
        index = pd.date_range(day.tz_localize(TIMEZONE), (day + pd.Timedelta(days=1)).tz_localize(TIMEZONE),
                              freq="h", inclusive="left").tz_convert("UTC")
        blocks.append(pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": civil_cutoff(day),
                                    "revision_time_utc": civil_cutoff(day), "value": float(value),
                                    "downloaded_at_utc": retrieved}))
    frame = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(
        columns=["value_time_utc", "snapshot_time_utc", "revision_time_utc", "value", "downloaded_at_utc"])
    temp = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(output)
    missing_final = days.difference(merged.index)
    audit = {"schema_version": 1, "alias": alias, "series": spec["series"], "unit": spec["unit"],
             "information_type": "capacity_forecast", "cutoff_time": "08:00", "cutoff_timezone": TIMEZONE,
             "daily_broadcast": True, "fill_or_interpolation": "daily_capacity_forecast_broadcast; no missing-day substitution",
             "approximation": "Daily Pmax forecast applied uniformly to physical hours; not hourly Pmax, not realised generation; provider fleet scope unverified",
             "start_day": start_day, "end_day": end_day, "requested_days": len(days),
             "days": int(days.isin(merged.index).sum()), "complete": not len(missing_final), "rows": len(frame),
             "missing_days": [str(d.date()) for d in missing_final], "errors": errors,
             "provider_revision_timestamp_available": False, "revision_time_semantics": "query_asof_cutoff",
             "production_pit_evidence": False, "promotion_eligible": False,
             "block_staircase_request": request, "block_error": bulk_error, "independent_probes": probes,
             "sha256": sha256(output), "elapsed_seconds": round(time.monotonic()-begun, 3)}
    _write_json(audit_path, audit)
    print(f"[MarginalCost/{alias}] {audit['days']}/{len(days)} days; {audit['elapsed_seconds']:.1f}s; missing={len(missing_final)}", flush=True)
    return {"alias": alias, "status": "complete" if audit["complete"] else "incomplete",
            "days": audit["days"], "missing_days": audit["missing_days"], "path": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-day", required=True)
    parser.add_argument("--end-day", required=True)
    parser.add_argument("--zones", nargs="+", default=["FR", "DE", "BE", "NL"])
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--output-dir", default=str(ROOT / "data/pit/marginal_cost_expert/capacities"))
    args = parser.parse_args(argv)
    if set(args.zones).difference({"FR", "DE", "BE", "NL"}):
        raise ValueError("Unsupported capacity source zone")
    specs = capacity_catalog(args.zones)
    root = allowed_output(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f"[MarginalCost] {len(specs)} primary capacity sources, {args.start_day}..{args.end_day}, workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(materialize_capacity, alias, spec, start_day=args.start_day,
                            end_day=args.end_day, output_dir=root): alias for alias, spec in specs.items()}
        for job in as_completed(jobs):
            alias = jobs[job]
            try:
                results[alias] = job.result()
            except Exception as exc:
                results[alias] = {"alias": alias, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                print(f"[MarginalCost/{alias}] FAILED: {exc}", flush=True)
            _write_json(root / "collection_progress.json", results)
    return 0 if all(v["status"] in {"complete", "reused"} for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
