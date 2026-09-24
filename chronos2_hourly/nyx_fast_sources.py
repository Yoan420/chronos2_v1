"""Bounded source capture for the isolated NYX/Test2 live candidate.

Existing SolarWind and weather histories are donors, never output locations.
Only the missing suffix (at most 31 civil days) is queried, as of D-1 08:00.
Historical query-as-of is NOT evidence of original provider publication.
"""
from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pandas as pd

from materialize_saturn_kalman_weather import SeriesPlan, build_plan, build_command, ZONE_TIMEZONES
from . import solar_wind_sources as wind
from . import solar_cwe_sources as solar
from .process_lock import exclusive_process_lock

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/experiments/n2"


def safe(path):
    path = Path(path).absolute()
    if (path.resolve() != path or OUTPUT.resolve() != OUTPUT
            or not path.is_relative_to(OUTPUT) or path == OUTPUT):
        raise ValueError("Live outputs must be real children of the accelerated n2 namespace")
    return path


def atomic_json(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def plans(output, include_befr=False):
    output = safe(output)
    solar_plans = [p for p in build_plan(("FR", "DE", "BE", "NL"), output) if p.kind == "solar"]
    wind_zones = ("DE", "NL", "BE", "FR") if include_befr else ("DE", "NL")
    return tuple(solar_plans + [SeriesPlan(z, "wind", f"{z.lower()}_wind_generation_fcst",
                 f"power.{z.lower()}.generation.wind.hourly.gw.fcst", ZONE_TIMEZONES[z], ZONE_TIMEZONES[z],
                 output / f"{z.lower()}_wind_generation_fcst.parquet", unit="GW") for z in wind_zones])


def inspect(plan, first, last, *, legacy=False):
    return wind._inspect(plan, first, last, legacy=legacy, wind_dst_policy="duplicate",
                         wind_gap_policy=wind.NL_SPRING_GAP_POLICY)


def _seed(plan, first, last):
    # Do not silently fall back if a canonical SolarWind donor exists but fails.
    canonical = ROOT / "data/pit/solar_wind_v1" / plan.output.name
    candidate = canonical if canonical.exists() else ROOT / "data/pit/kalman_weather" / plan.output.name
    if candidate.resolve() != candidate or not candidate.is_file():
        raise ValueError(f"Missing unredirected source donor: {candidate}")
    donor = replace(plan, output=candidate)
    return donor, inspect(donor, first, last, legacy=candidate != canonical)


def validate_capture(output, *, start_day, delivery_day, include_befr=False):
    """Read-only complete validation; all eight curves only for BE/FR extension."""
    first, last = solar._day(start_day), solar._day(delivery_day)
    records = {}
    for p in plans(output, include_befr):
        safe(p.output)
        safe(p.audit_path)
        records[p.alias] = inspect(p, first, last)
    if any(not r["complete"] for r in records.values()):
        raise ValueError("Live source suffix incomplete")
    return records


def _wind_suffix(plan, start, last, first):
    args = Namespace(series=plan.series, alias=plan.alias, timezone=plan.timezone,
        cutoff_timezone=plan.timezone, cutoff_time="08:00", naive_timezone=plan.naive_timezone,
        saturn_url="https://saturn-energyscan.gem.myengie.com//api", author="BQ6757",
        request_timeout_seconds=60., retries=3, request_padding_hours=8,
        incomplete_dst_policy="duplicate", daily_broadcast=False, hourly_on_the_hour=False,
        allow_incomplete_days=False, value_scale=1., wind_gap_policy=wind.NL_SPRING_GAP_POLICY)
    previous = json.loads(plan.audit_path.read_text(encoding="utf-8"))
    frames = [pd.read_parquet(plan.output)]
    repairs, substitutions = list(previous["dst_duplicate_repairs"]), list(previous.get("source_substitutions", []))
    for day in pd.date_range(start, last):
        directory = safe(plan.output.parent / "days" / plan.alias)
        day_plan = replace(plan, output=directory / f"{day.date()}.parquet")
        safe(day_plan.output)
        safe(day_plan.audit_path)
        if day_plan.output.exists() or day_plan.audit_path.exists():
            record = inspect(day_plan, day, day)
            if record["start_day"] != str(day.date()) or record["end_day"] != str(day.date()):
                raise ValueError("Day checkpoint date mismatch")
            frame = pd.read_parquet(day_plan.output)
            daily_repairs, daily_subs = record["dst_duplicate_repairs"], record["source_substitutions"]
        else:
            frame = wind._download_wind_day(day_plan, day, args)
            hours = pd.DatetimeIndex(frame.value_time_utc)
            daily_repairs = [{**r, "evidence": "raw_Saturn_singleton_at_query_asof"}
                for r in frame.attrs.get("dst_repairs", [])
                if pd.DatetimeIndex(pd.to_datetime(r["physical_hours_utc"], utc=True)).isin(hours).all()]
            daily_subs = frame.attrs.get("source_substitutions", [])
            metadata = wind._wind_metadata(day_plan, frame, day, day, daily_repairs,
                wind_dst_policy="duplicate", source_substitutions=daily_subs,
                wind_gap_policy=wind.NL_SPRING_GAP_POLICY if daily_subs else None)
            directory.mkdir(parents=True, exist_ok=True)
            wind._publish_wind(day_plan, frame, metadata, wind_gap_policy=wind.NL_SPRING_GAP_POLICY)
            inspect(day_plan, day, day)
        frames.append(frame)
        repairs.extend(daily_repairs)
        substitutions.extend(daily_subs)
        print(f"[live sources] {plan.alias} {day.date()} captured", flush=True)
    merged = pd.concat(frames, ignore_index=True).sort_values("value_time_utc").reset_index(drop=True)
    if merged.value_time_utc.duplicated().any():
        raise ValueError("Duplicate source hours during suffix assembly")
    metadata = wind._wind_metadata(plan, merged, first, last, repairs, wind_dst_policy="duplicate",
        previous=previous, source_substitutions=substitutions,
        wind_gap_policy=wind.NL_SPRING_GAP_POLICY if substitutions else None)
    wind._publish_wind(plan, merged, metadata, wind_gap_policy=wind.NL_SPRING_GAP_POLICY)
    inspect(plan, first, last)


def ensure_live_sources(output, *, start_day, delivery_day, include_befr=False):
    """Copy audited histories and append an audited bounded suffix, serially."""
    output = safe(output)
    first, last = solar._day(start_day), solar._day(delivery_day)
    if first > last:
        raise ValueError("Reversed source dates")
    cutoff = (last - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    if cutoff > pd.Timestamp.now(tz="UTC"):
        raise ValueError("D-1 08:00 cutoff has not occurred")
    with exclusive_process_lock(output / "capture.lock"):
        prepared = []
        for plan in plans(output, include_befr):
            safe(plan.output)
            safe(plan.audit_path)
            exists = plan.output.exists() or plan.audit_path.exists()
            donor, record = (plan, inspect(plan, first, last)) if exists else _seed(plan, first, last)
            suffix = pd.Timestamp(record["end_day"]) + pd.Timedelta(days=1)
            if (last - suffix).days + 1 > 31:
                raise ValueError(f"{plan.alias}: suffix exceeds 31 days")
            prepared.append((plan, donor, record, suffix, exists))
        for plan, donor, record, suffix, exists in prepared:
            safe(plan.output)
            safe(plan.audit_path)
            if not exists:
                output.mkdir(parents=True, exist_ok=True)
                if plan.output.exists() or plan.audit_path.exists():
                    raise ValueError("Source destination appeared during seed copy")
                shutil.copy2(donor.output, plan.output)
                shutil.copy2(donor.audit_path, plan.audit_path)
                if solar._sha(plan.output) != record["sha256"] or solar._sha(plan.audit_path) != record["audit_sha256"]:
                    raise ValueError("Donor changed during copy")
                if plan.kind == "wind" and "dst_duplicate_repairs" not in record["audit"]:
                    wind._annotate_wind_seed(plan, record, wind_dst_policy="duplicate")
                inspect(plan, first, min(last, pd.Timestamp(record["end_day"])))
            if suffix <= last:
                if plan.kind == "wind":
                    _wind_suffix(plan, suffix, last, first)
                else:
                    command = build_command(plan, start_day=suffix, end_day=last, day_workers=1, merge_existing=True)
                    command[0:1] = [sys.executable, "-B", "-u"]
                    subprocess.run(command, cwd=ROOT, shell=False, check=True)
            if not exists and (solar._sha(donor.output) != record["sha256"] or solar._sha(donor.audit_path) != record["audit_sha256"]):
                raise ValueError("Read-only donor changed during capture")
        result = validate_capture(output, start_day=start_day, delivery_day=delivery_day, include_befr=include_befr)
        # One receipt per schema prevents BE/FR extension from overwriting six-source provenance.
        atomic_json(output / ("receipt_cwe.json" if include_befr else "receipt_denl.json"), {
            "captured_utc": pd.Timestamp.now(tz="UTC").isoformat(), "delivery_day": delivery_day,
            "sources": result, "production_modified": False, "production_pit_evidence": False,
            "purpose": "isolated live candidate; bounded query-asof reconstruction"})
        return result
