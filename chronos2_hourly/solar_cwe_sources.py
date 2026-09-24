"""Four isolated generation-forecast sources, with bounded suffix-only sync.

Saturn query-as-of timestamps are not provider publication/capture evidence.
No operational cache is modified and no missing historical prefix is fetched.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from materialize_saturn_kalman_weather import build_plan, build_command, reusable_output, incremental_extension_start
from .process_lock import exclusive_process_lock

ROOT = Path(__file__).resolve().parents[1]
ZONES = ("FR", "DE", "BE", "NL")
SOLAR_SERIES = {f"{z.lower()}_solar_generation_fcst": f"power.{z.lower()}.generation.solar.hourly.gw.fcst" for z in ZONES}
MAX_SUFFIX_DAYS = 31
COLUMNS = ("value_time_utc", "snapshot_time_utc", "revision_time_utc", "value", "downloaded_at_utc")


class SolarCweSourceError(ValueError):
    pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _day(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise SolarCweSourceError("Exact naive civil YYYY-MM-DD dates required")
    return day


def _safe(value):
    raw = Path(value)
    raw = (ROOT/raw).absolute() if not raw.is_absolute() else raw.absolute()
    allowed = ROOT.absolute()/"data/pit/solar_cwe"
    if allowed.resolve() != allowed or raw.resolve() != raw or not raw.is_relative_to(allowed):
        raise SolarCweSourceError("Solar source outputs must stay under data/pit/solar_cwe without aliases/junctions")
    return raw


def _inspect(plan, required_start, required_end):
    """Validate the entire saved cache, including deeper-than-required history."""
    try:
        before = {p: _sha(p) for p in (plan.output, plan.audit_path)}
        metadata = json.loads(plan.audit_path.read_text(encoding="utf-8"))
        first, last = _day(metadata["start_day"]), _day(metadata["end_day"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SolarCweSourceError(f"{plan.alias}: missing/invalid parquet and audit pair") from exc
    if first > required_start:
        raise SolarCweSourceError(f"{plan.alias}: historical prefix missing before {first.date()}; no prefix backfill is authorized")
    if last < first:
        raise SolarCweSourceError(f"{plan.alias}: reversed source dates")
    # reusable_output deliberately requires exact artifact_start, not the
    # shorter requested anchor. Full physical coverage is checked to last.
    valid, reason = reusable_output(plan, start_day=first, end_day=last)
    if not valid:
        raise SolarCweSourceError(f"{plan.alias}: invalid saved source: {reason}")
    if (metadata.get("causal_contract") != "Saturn state queried as-of D-1 civil cutoff"
            or metadata.get("snapshot_time_semantics") != "query_asof_cutoff"
            or not str(metadata.get("revision_time_semantics", "")).startswith("query_asof_cutoff")
            or metadata.get("provider_revision_timestamp_available") is not False
            or metadata.get("unit", "GW") != "GW"):
        raise SolarCweSourceError(f"{plan.alias}: explicit query-asof generation-GW provenance required")
    frame = pd.read_parquet(plan.output)
    if tuple(frame.columns) != COLUMNS:
        raise SolarCweSourceError(f"{plan.alias}: canonical five-column PIT schema required")
    for column in (c for c in COLUMNS if c != "value"):
        if not isinstance(frame[column].dtype, pd.DatetimeTZDtype) or frame[column].isna().any():
            raise SolarCweSourceError(f"{plan.alias}: explicit aware complete timestamps required: {column}")
    if frame.downloaded_at_utc.lt(frame.snapshot_time_utc).any():
        raise SolarCweSourceError(f"{plan.alias}: download timestamp precedes its as-of query")
    values = pd.to_numeric(frame.value, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise SolarCweSourceError(f"{plan.alias}: finite nonnegative generation GW required")
    if any(_sha(path) != digest for path, digest in before.items()):
        raise SolarCweSourceError(f"{plan.alias}: source changed during verification")
    return {"path": str(plan.output), "sha256": before[plan.output], "audit_path": str(plan.audit_path),
            "audit_sha256": before[plan.audit_path], "alias": plan.alias, "series": plan.series, "unit": "GW",
            "start_day": str(first.date()), "end_day": str(last.date()), "required_start_day": str(required_start.date()),
            "required_end_day": str(required_end.date()), "complete": last >= required_end, "audit": metadata,
            "specification": {"alias": plan.alias, "series": plan.series, "unit": "GW", "daily_broadcast": False},
            "provider_revision_timestamp_available": False, "production_pit_evidence": False,
            "pit_evidence_level": "query_asof_cutoff_only", "fill_or_interpolation": metadata.get("fill_or_interpolation")}


def _seed(plan, first, last):
    refresh = ROOT/"runs/experiments/nyx_scarcity_v1/source_refresh"
    candidates = sorted(refresh.glob(f"*/{plan.alias}.parquet"), reverse=True)
    candidates.append(ROOT/"data/pit/kalman_weather"/plan.output.name)
    errors = []
    for path in candidates:
        if not path.is_file():
            continue
        if path.resolve() != path.absolute() or not path.resolve().is_relative_to(ROOT.resolve()):
            continue
        candidate = replace(plan, output=path)
        try:
            record = _inspect(candidate, first, last)
            return candidate, record
        except SolarCweSourceError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors[:2]) or "no local audited seed found"
    raise SolarCweSourceError(f"{plan.alias}: no reusable seed; {detail}")


def ensure_solar_sources(*, output_root: Path, start_day: str, end_day: str, workers=2, sync=False) -> dict:
    """Return four alias-keyed source records; ``sync=False`` never writes.

    Sync copies only verified immutable seeds into the isolated directory and
    fetches at most 31 missing suffix days per series. Existing invalid caches,
    gaps and missing historical prefixes fail closed instead of being rebuilt.
    ``start_day`` may be later than the source's saved start day.
    """
    output = _safe(output_root)
    first, last = _day(start_day), _day(end_day)
    if first > last or type(workers) is not int or not 1 <= workers <= 2 or type(sync) is not bool:
        raise SolarCweSourceError("Ordered dates, 1-2 workers, and boolean sync required")
    cutoff = (last-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    if cutoff > pd.Timestamp.now(tz="UTC"):
        raise SolarCweSourceError("Requested final D-1 08:00 cutoff has not occurred")
    plans = tuple(p for p in build_plan(ZONES, output) if p.kind == "solar")
    if {p.alias: p.series for p in plans} != SOLAR_SERIES:
        raise SolarCweSourceError("Weather materializer changed the four-series solar allowlist")
    for plan in plans:
        _safe(plan.output)
        _safe(plan.audit_path)
    if not sync:
        results = {}
        for plan in plans:
            record = _inspect(plan, first, last)
            if not record["complete"]:
                raise SolarCweSourceError(f"{plan.alias}: missing suffix after {record['end_day']}; rerun with sync=True for bounded isolated extension")
            results[plan.alias] = record
        return results
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(output/".sync.lock"):
        prepared = []
        # Validate all four prefixes and download bounds before copying or
        # requesting anything. Never mix a partial pair with a fallback seed.
        for plan in plans:
            exists = plan.output.exists() or plan.audit_path.exists()
            source_plan, record = (plan, _inspect(plan, first, last)) if exists else _seed(plan, first, last)
            suffix, reason = incremental_extension_start(source_plan, start_day=_day(record["start_day"]), end_day=last)
            if not record["complete"] and suffix is None:
                raise SolarCweSourceError(f"{plan.alias}: non-extendable source: {reason}")
            if suffix is not None and (last-suffix).days+1 > MAX_SUFFIX_DAYS:
                raise SolarCweSourceError(f"{plan.alias}: suffix exceeds {MAX_SUFFIX_DAYS} days; no long backfill authorized")
            prepared.append((plan, source_plan, record, suffix, exists))
        results = {}
        for plan, source_plan, source_record, suffix, exists in prepared:
            if not exists:
                if plan.output.exists() or plan.audit_path.exists():
                    raise SolarCweSourceError(f"{plan.alias}: destination appeared during seed copy")
                shutil.copy2(source_plan.output, plan.output)
                shutil.copy2(source_plan.audit_path, plan.audit_path)
                copied = _inspect(plan, first, last)
                if copied["sha256"] != source_record["sha256"] or copied["audit_sha256"] != source_record["audit_sha256"]:
                    raise SolarCweSourceError(f"{plan.alias}: seed changed during copy")
            if suffix is not None:
                command = build_command(plan, start_day=suffix, end_day=last, day_workers=workers, merge_existing=True)
                subprocess.run(command, cwd=ROOT, check=True, shell=False)
            record = _inspect(plan, first, last)
            if not record["complete"]:
                raise SolarCweSourceError(f"{plan.alias}: materialized suffix remains incomplete")
            if not exists and (_sha(source_plan.output) != source_record["sha256"] or _sha(source_plan.audit_path) != source_record["audit_sha256"]):
                raise SolarCweSourceError(f"{plan.alias}: canonical seed changed during isolated sync")
            record["seed_path"] = str(source_plan.output)
            record["downloaded_suffix_start_day"] = str(suffix.date()) if suffix is not None else None
            results[plan.alias] = record
        return results


__all__ = ["SOLAR_SERIES", "SolarCweSourceError", "ensure_solar_sources"]
