"""Isolated exact-series SolarCWE + DE/NL wind query-as-of laboratory inputs.

Wind's explicit covariate-only DST policy copies a *nonzero* singleton autumn
label to its two physical folds, recording both hours and its value. It never
zero-fills wind. Query cutoffs are not provider publication evidence.
"""
from __future__ import annotations

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
import pandas as pd

from materialize_saturn_kalman_weather import SeriesPlan, build_command, build_plan
from . import solar_cwe_sources as solar
from .process_lock import exclusive_process_lock

ROOT = Path(__file__).resolve().parents[1]
SOLAR_SERIES = dict(solar.SOLAR_SERIES)
WIND_SERIES = {f"{z}_wind_generation_fcst": f"power.{z}.generation.wind.hourly.gw.fcst" for z in ("de", "nl")}
GENERATION_SERIES = {**SOLAR_SERIES, **WIND_SERIES}
WIND_FILL_POLICY = "duplicate_singleton_autumn_fold_covariate"
MAX_WIND_BACKFILL_DAYS = 900
MAX_SUFFIX_DAYS = 31
NL_SPRING_GAP_POLICY = "nl_ecmwf_spring_2025_2026"
NL_ECMWF_COMPONENT = "power.nrjscan.nl.prod.total.wind.mw.ecmwf_avg.pointconnect.6h.cache"
NL_SPRING_GAP_HOURS = {day: pd.Timestamp(f"{day}T02:00:00Z") for day in ("2025-03-30", "2026-03-29")}
SUBSTITUTION_PROVENANCE = ("Saturn ECMWF component queried at the identical D-1 08:00 local cutoff; "
                           "explicit user-approved missing-hour substitution, not the native blend or publication evidence")


class SolarWindSourceError(ValueError):
    pass


def _validate_gap_policy(policy):
    if policy is not None and policy != NL_SPRING_GAP_POLICY:
        raise SolarWindSourceError("Unknown wind_gap_policy; only the two explicitly approved NL spring hours may be substituted")


def _authorized_gap_hour(plan, day, policy):
    _validate_gap_policy(policy)
    if (policy == NL_SPRING_GAP_POLICY and plan.alias == "nl_wind_generation_fcst"
            and plan.series == WIND_SERIES[plan.alias] and plan.timezone == "Europe/Amsterdam"
            and plan.naive_timezone == "Europe/Amsterdam"):
        return NL_SPRING_GAP_HOURS.get(str(day.date()))
    return None


def _safe(value):
    path = Path(value)
    path = (ROOT / path).absolute() if not path.is_absolute() else path.absolute()
    allowed = ROOT.absolute() / "data/pit/solar_wind_v1"
    if allowed.resolve() != allowed or path.resolve() != path or not path.is_relative_to(allowed):
        raise SolarWindSourceError("Source outputs must stay under data/pit/solar_wind_v1 without path aliases")
    return path


def _plans(output):
    plans = [p for p in build_plan(solar.ZONES, output) if p.kind == "solar"]
    for zone, timezone in (("DE", "Europe/Berlin"), ("NL", "Europe/Amsterdam")):
        alias = f"{zone.lower()}_wind_generation_fcst"
        plans.append(SeriesPlan(zone, "wind", alias, WIND_SERIES[alias], timezone, timezone,
                                Path(output) / f"{alias}.parquet", unit="GW"))
    if {p.alias: p.series for p in plans} != GENERATION_SERIES:
        raise SolarWindSourceError("Exact six-series generation allowlist changed")
    return tuple(plans)


def _repair_rows(plan, frame, *, evidence):
    """Conservatively expose legacy equal-fold copies; do not attest raw data."""
    stamps = pd.DatetimeIndex(frame.value_time_utc)
    wall = stamps.tz_convert(plan.timezone).tz_localize(None)
    rows = []
    for stamp in wall[wall.duplicated(keep=False)].unique():
        positions = np.flatnonzero(wall == stamp)
        values = frame.value.iloc[positions].to_numpy(float)
        if len(positions) == 2 and values[0] == values[1]:
            rows.append({"policy": "duplicate", "local_timestamp": stamp.isoformat(),
                         "duplicated_value": float(values[0]),
                         "physical_hours_utc": [t.isoformat() for t in stamps[positions]],
                         "evidence": evidence})
    return rows


def _validate_repairs(plan, frame, metadata):
    repairs = metadata.get("dst_duplicate_repairs")
    if not isinstance(repairs, list) or metadata.get("dst_duplicate_repair_count") != len(repairs):
        raise SolarWindSourceError(f"{plan.alias}: missing/inconsistent wind DST repair audit")
    index = pd.DatetimeIndex(frame.value_time_utc)
    identities = set()
    for repair in repairs:
        try:
            stamp = pd.Timestamp(repair["local_timestamp"])
            value = float(repair["duplicated_value"])
            hours = pd.DatetimeIndex(pd.to_datetime(repair["physical_hours_utc"], utc=True))
            expected = pd.DatetimeIndex(sorted((
                stamp.tz_localize(plan.timezone, ambiguous=True, nonexistent="raise").tz_convert("UTC"),
                stamp.tz_localize(plan.timezone, ambiguous=False, nonexistent="raise").tz_convert("UTC"))))
        except (KeyError, TypeError, ValueError) as exc:
            raise SolarWindSourceError(f"{plan.alias}: invalid wind DST repair") from exc
        identity = tuple(t.isoformat() for t in hours)
        positions = index.isin(hours)
        if (repair.get("policy") != "duplicate" or stamp.tzinfo is not None or not np.isfinite(value)
                or value < 0 or len(expected.unique()) != 2 or not hours.equals(expected)
                or identity in identities or int(positions.sum()) != 2
                or not bool((frame.value.loc[positions].to_numpy(float) == value).all())
                or not repair.get("evidence")):
            raise SolarWindSourceError(f"{plan.alias}: unproven wind DST fold identity/value")
        identities.add(identity)


def _validate_substitutions(plan, frame, metadata, *, wind_gap_policy):
    _validate_gap_policy(wind_gap_policy)
    substitutions = metadata.get("source_substitutions", [])
    count = metadata.get("source_substitution_count", 0)
    if (not isinstance(substitutions, list) or type(count) is not int or count != len(substitutions)
            or frame.attrs.get("source_substitutions", []) != substitutions):
        raise SolarWindSourceError(f"{plan.alias}: missing/inconsistent source substitution audit")
    if substitutions and (wind_gap_policy != NL_SPRING_GAP_POLICY or metadata.get("wind_gap_policy") != wind_gap_policy):
        raise SolarWindSourceError(f"{plan.alias}: strict wind gap policy rejects cached substitutions")
    seen = set()
    for row in substitutions:
        try:
            day = solar._day(row["delivery_day"])
            hour = pd.Timestamp(row["value_time_utc"])
            cutoff = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(plan.timezone).tz_convert("UTC")
            raw, scaled = float(row["raw_value_mw"]), float(row["scaled_value_gw"])
            downloaded = pd.Timestamp(row["fallback_downloaded_at_utc"])
            selected = frame.loc[frame.value_time_utc.eq(hour)]
            valid = (row["policy"] == wind_gap_policy and row["alias"] == plan.alias
                     and row["native_series"] == plan.series and row["fallback_series"] == NL_ECMWF_COMPONENT
                     and hour == _authorized_gap_hour(plan, day, wind_gap_policy) and hour.tzinfo is not None
                     and row["local_time"] == hour.tz_convert(plan.timezone).isoformat()
                     and row["query_cutoff_utc"] == cutoff.isoformat()
                     and row["query_cutoff_local"] == cutoff.tz_convert(plan.timezone).isoformat()
                     and row["native_missing"] is True and row["value_scale"] == .001
                     and np.isfinite(raw) and raw >= 0 and np.isfinite(scaled) and scaled == raw * .001
                     and downloaded.tzinfo is not None and downloaded >= cutoff
                     and row["provenance"] == SUBSTITUTION_PROVENANCE
                     and row["provider_revision_timestamp_available"] is False
                     and row["production_pit_evidence"] is False
                     and len(selected) == 1 and float(selected.value.iloc[0]) == scaled
                     and selected.snapshot_time_utc.iloc[0] == cutoff and selected.revision_time_utc.iloc[0] == cutoff
                     and selected.downloaded_at_utc.iloc[0] == downloaded and hour not in seen)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
            raise SolarWindSourceError(f"{plan.alias}: invalid source substitution provenance") from exc
        if not valid:
            raise SolarWindSourceError(f"{plan.alias}: unproven source substitution scope/cutoff/value")
        seen.add(hour)


def _inspect_wind(plan, first, last, *, legacy, wind_dst_policy, wind_gap_policy=None):
    try:
        hashes = {p: solar._sha(p) for p in (plan.output, plan.audit_path)}
        audit = json.loads(plan.audit_path.read_text(encoding="utf-8"))
        start, end = solar._day(audit["start_day"]), solar._day(audit["end_day"])
        frame = pd.read_parquet(plan.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SolarWindSourceError(f"{plan.alias}: missing/invalid parquet and audit pair") from exc
    expected = {"series": plan.series, "alias": plan.alias, "timezone": plan.timezone,
                "naive_timezone": plan.naive_timezone, "cutoff_timezone": plan.timezone,
                "cutoff_time": "08:00", "value_scale": 1., "daily_broadcast": False,
                "incomplete_dst_policy": wind_dst_policy,
                "causal_contract": "Saturn state queried as-of D-1 civil cutoff",
                "snapshot_time_semantics": "query_asof_cutoff", "provider_revision_timestamp_available": False}
    if any(audit.get(key) != value for key, value in expected.items()):
        raise SolarWindSourceError(f"{plan.alias}: exact series/timezone/cutoff/wind DST provenance required")
    if (start > first or end < start or audit.get("sha256") != hashes[plan.output]
            or audit.get("unit", "GW") != "GW"
            or not str(audit.get("revision_time_semantics", "")).startswith("query_asof_cutoff")):
        raise SolarWindSourceError(f"{plan.alias}: missing prefix or invalid source checksum/provenance")
    if tuple(frame.columns) != solar.COLUMNS:
        raise SolarWindSourceError(f"{plan.alias}: canonical five-column PIT schema required")
    for name in (c for c in solar.COLUMNS if c != "value"):
        if not isinstance(frame[name].dtype, pd.DatetimeTZDtype) or frame[name].isna().any():
            raise SolarWindSourceError(f"{plan.alias}: complete aware timestamps required")
    stamps = pd.DatetimeIndex(frame.value_time_utc)
    expected_grid = pd.date_range(start.tz_localize(plan.timezone).tz_convert("UTC"),
                                 (end + pd.Timedelta(days=1)).tz_localize(plan.timezone).tz_convert("UTC"),
                                 freq="h", inclusive="left")
    if not stamps.equals(expected_grid):
        raise SolarWindSourceError(f"{plan.alias}: exact physical hourly coverage required")
    civil = stamps.tz_convert(plan.timezone).tz_localize(None).normalize()
    cutoffs = (civil - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(plan.timezone).tz_convert("UTC")
    values = pd.to_numeric(frame.value, errors="raise").to_numpy(float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not pd.DatetimeIndex(frame.snapshot_time_utc).equals(cutoffs)
            or not pd.DatetimeIndex(frame.revision_time_utc).equals(cutoffs)
            or frame.downloaded_at_utc.lt(frame.snapshot_time_utc).any()
            or audit.get("days") != (end - start).days + 1 or audit.get("rows") != len(stamps)
            or audit.get("first_delivery_utc") != stamps[0].isoformat()
            or audit.get("last_delivery_utc") != stamps[-1].isoformat()):
        raise SolarWindSourceError(f"{plan.alias}: invalid generation, physical bounds, or civil 08:00 cutoff")
    if not legacy:
        expected_fill = WIND_FILL_POLICY if wind_dst_policy == "duplicate" else "none"
        if audit.get("wind_dst_policy") != wind_dst_policy or audit.get("fill_or_interpolation") != expected_fill:
            raise SolarWindSourceError(f"{plan.alias}: explicit audited wind DST policy required")
        _validate_repairs(plan, frame, audit)
        if wind_dst_policy == "raise" and audit["dst_duplicate_repairs"]:
            raise SolarWindSourceError(f"{plan.alias}: strict wind policy forbids any DST repairs")
    _validate_substitutions(plan, frame, audit, wind_gap_policy=wind_gap_policy)
    if any(solar._sha(path) != digest for path, digest in hashes.items()):
        raise SolarWindSourceError(f"{plan.alias}: source changed during verification")
    return {"path": str(plan.output), "sha256": hashes[plan.output], "audit_path": str(plan.audit_path),
            "audit_sha256": hashes[plan.audit_path], "alias": plan.alias, "series": plan.series, "unit": "GW",
            "start_day": str(start.date()), "end_day": str(end.date()), "required_start_day": str(first.date()),
            "required_end_day": str(last.date()), "complete": end >= last, "audit": audit,
            "specification": {"alias": plan.alias, "series": plan.series, "unit": "GW", "daily_broadcast": False},
            "provider_revision_timestamp_available": False, "production_pit_evidence": False,
            "pit_evidence_level": "query_asof_cutoff_only", "fill_or_interpolation": audit.get("fill_or_interpolation")}


def _inspect(plan, first, last, *, legacy=False, wind_dst_policy="duplicate", wind_gap_policy=None):
    try:
        record = (_inspect_wind(plan, first, last, legacy=legacy, wind_dst_policy=wind_dst_policy, wind_gap_policy=wind_gap_policy)
                  if plan.kind == "wind" else solar._inspect(plan, first, last))
    except solar.SolarCweSourceError as exc:
        raise SolarWindSourceError(str(exc)) from exc
    if plan.kind == "wind" and not legacy:
        audit = record["audit"]
        record["dst_duplicate_repair_count"] = audit["dst_duplicate_repair_count"]
        record["dst_duplicate_repairs"] = audit["dst_duplicate_repairs"]
        record["source_substitution_count"] = audit.get("source_substitution_count", 0)
        record["source_substitutions"] = audit.get("source_substitutions", [])
    record["source_role"] = plan.kind
    return record


def _seed(plan, first, last, *, wind_dst_policy="duplicate"):
    candidates = []
    if plan.kind == "solar":
        candidates.append(ROOT / "data/pit/solar_cwe" / plan.output.name)
    refresh = ROOT / "runs/experiments/nyx_scarcity_v1/source_refresh"
    candidates.extend(sorted(refresh.glob(f"*/{plan.output.name}"), reverse=True))
    candidates.append(ROOT / "data/pit/kalman_weather" / plan.output.name)
    for path in candidates:
        if not path.is_file() or path.resolve() != path.absolute() or not path.is_relative_to(ROOT.absolute()):
            continue
        candidate = replace(plan, output=path)
        try:
            record = _inspect(candidate, first, last, legacy=True, wind_dst_policy=wind_dst_policy)
        except SolarWindSourceError:
            continue
        return candidate, record
    if plan.kind == "solar":
        raise SolarWindSourceError(f"{plan.alias}: no immutable audited solar seed covering required prefix")
    # In particular, the old NL 100% ECMWF MW curve is NOT an eligible seed.
    return None, None


def _write_audit(plan, audit):
    temporary = plan.audit_path.with_name(f".{plan.audit_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(plan.audit_path)


def _annotate_wind_seed(plan, seed_record, *, wind_dst_policy="duplicate"):
    audit = json.loads(plan.audit_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(plan.output)
    repairs = (_repair_rows(plan, frame, evidence="legacy_duplicate_policy_and_equal_cached_folds; raw_singleton_not_requeried")
               if wind_dst_policy == "duplicate" else [])
    audit.update(wind_dst_policy=wind_dst_policy, fill_or_interpolation=WIND_FILL_POLICY if wind_dst_policy == "duplicate" else "none",
                 dst_duplicate_repairs=repairs, dst_duplicate_repair_count=len(repairs),
                 unit="GW", production_pit_evidence=False,
                 seed_provenance={k: seed_record[k] for k in ("path", "sha256", "audit_path", "audit_sha256")})
    _write_audit(plan, audit)


def _wind_metadata(plan, frame, first, last, repairs, *, wind_dst_policy, previous=None,
                   source_substitutions=None, wind_gap_policy=None):
    civil = frame.value_time_utc.dt.tz_convert(plan.timezone).dt.tz_localize(None).dt.normalize()
    return {**(previous or {}), "schema_version": 1, "series": plan.series, "alias": plan.alias,
            "timezone": plan.timezone, "naive_timezone": plan.naive_timezone, "cutoff_timezone": plan.timezone,
            "cutoff_time": "08:00", "value_scale": 1., "daily_broadcast": False, "unit": "GW",
            "incomplete_dst_policy": wind_dst_policy, "wind_dst_policy": wind_dst_policy, "request_padding_hours": 8,
            "start_day": str(civil.min().date()), "end_day": str(civil.max().date()), "days": int(civil.nunique()),
            "rows": len(frame), "first_delivery_utc": frame.value_time_utc.iloc[0].isoformat(),
            "last_delivery_utc": frame.value_time_utc.iloc[-1].isoformat(),
            "fill_or_interpolation": WIND_FILL_POLICY if wind_dst_policy == "duplicate" else "none",
            "dst_duplicate_repairs": repairs, "dst_duplicate_repair_count": len(repairs),
            "wind_gap_policy": wind_gap_policy, "source_substitutions": source_substitutions or [],
            "source_substitution_count": len(source_substitutions or []),
            "causal_contract": "Saturn state queried as-of D-1 civil cutoff", "snapshot_time_semantics": "query_asof_cutoff",
            "revision_time_semantics": "query_asof_cutoff; provider insertion timestamp unavailable",
            "provider_revision_timestamp_available": False, "production_pit_evidence": False,
            "requested_start_day": str(first.date()), "requested_end_day": str(last.date()),
            "requested_days": (last - first).days + 1}


def _publish_wind(plan, frame, metadata, *, wind_gap_policy=None):
    _validate_repairs(plan, frame, metadata)
    # Bind the substitution inventory to the checksummed parquet too, so a
    # missing/edited sidecar inventory cannot silently relabel mixed sources.
    frame.attrs["source_substitutions"] = metadata.get("source_substitutions", [])
    _validate_substitutions(plan, frame, metadata, wind_gap_policy=wind_gap_policy)
    temporary = plan.output.with_name(f".{plan.output.name}.{uuid.uuid4().hex}.tmp.parquet")
    frame.loc[:, list(solar.COLUMNS)].to_parquet(temporary, index=False)
    metadata["sha256"] = solar._sha(temporary)
    temporary.replace(plan.output)
    _write_audit(plan, metadata)


def _fetch_ecmwf_hour(args, hour, cutoff):
    """Read the approved component at the identical cutoff, never latest."""
    from materialize_saturn_daily_asof import _client

    raw = _client(args).get(NL_ECMWF_COMPONENT, from_value_date=hour,
                            to_value_date=hour + pd.Timedelta(hours=1), revision_date=cutoff, nocache=True)
    if not isinstance(raw, pd.Series) or not isinstance(raw.index, pd.DatetimeIndex) or raw.index.tz is None:
        raise SolarWindSourceError("Approved ECMWF component requires explicit UTC-aware hourly labels")
    selected = raw.loc[raw.index.tz_convert("UTC") == hour]
    if len(selected) != 1:
        raise SolarWindSourceError("Approved ECMWF component has no unique value at the authorized hour")
    value = float(pd.to_numeric(selected, errors="raise").iloc[0])
    if not np.isfinite(value) or value < 0:
        raise SolarWindSourceError("Approved ECMWF component is not a finite nonnegative MW forecast")
    return value, pd.Timestamp.now(tz="UTC")


def _download_wind_day(plan, day, args):
    from materialize_saturn_daily_asof import _civil_cutoff, _one_day, _physical_utc_index

    gap_policy = getattr(args, "wind_gap_policy", None)
    allowed = _authorized_gap_hour(plan, day, gap_policy)
    if allowed is None:
        return _one_day(day, args)
    # Relax the native downloader only for these two pre-authorized NL dates.
    # Its returned finite physical products are checked before any substitution.
    native_args = Namespace(**{**vars(args), "allow_incomplete_days": True})
    frame = _one_day(day, native_args)
    expected = _physical_utc_index(day, timezone=plan.timezone)
    actual = pd.DatetimeIndex(frame.value_time_utc)
    if (actual.has_duplicates or not actual.isin(expected).all()
            or not np.isfinite(frame.value.to_numpy(float)).all() or frame.value.lt(0).any()):
        raise SolarWindSourceError("Native wind response has duplicate, extra, or invalid physical products")
    missing = expected.difference(actual)
    if not len(missing):
        return frame  # A finite native forecast always wins; no fallback query.
    if not missing.equals(pd.DatetimeIndex([allowed])):
        raise SolarWindSourceError("Native wind gaps differ from the single explicitly authorized spring hour")
    cutoff = _civil_cutoff(day, timezone=plan.timezone, cutoff_time="08:00").tz_convert("UTC")
    if (args.cutoff_time != "08:00" or args.cutoff_timezone != plan.timezone
            or not frame.snapshot_time_utc.eq(cutoff).all() or not frame.revision_time_utc.eq(cutoff).all()):
        raise SolarWindSourceError("Native and component forecasts must share the exact D-1 08:00 civil cutoff")
    raw_mw, downloaded = _fetch_ecmwf_hour(args, allowed, cutoff)
    scaled_gw = raw_mw * .001
    record = {"policy": gap_policy, "alias": plan.alias, "native_series": plan.series,
              "fallback_series": NL_ECMWF_COMPONENT, "delivery_day": str(day.date()),
              "value_time_utc": allowed.isoformat(), "local_time": allowed.tz_convert(plan.timezone).isoformat(),
              "query_cutoff_utc": cutoff.isoformat(), "query_cutoff_local": cutoff.tz_convert(plan.timezone).isoformat(),
              "raw_value_mw": raw_mw, "value_scale": .001, "scaled_value_gw": scaled_gw,
              "native_missing": True, "fallback_downloaded_at_utc": downloaded.isoformat(),
              "provenance": SUBSTITUTION_PROVENANCE, "provider_revision_timestamp_available": False,
              "production_pit_evidence": False}
    attrs = dict(frame.attrs)
    row = pd.DataFrame({"value_time_utc": [allowed], "snapshot_time_utc": [cutoff], "revision_time_utc": [cutoff],
                        "value": [scaled_gw], "downloaded_at_utc": [downloaded]})
    completed = pd.concat([frame, row], ignore_index=True).sort_values("value_time_utc").reset_index(drop=True)
    completed.attrs = {**attrs, "source_substitutions": [record]}
    return completed


def _checkpoint_day(plan, day, args):
    """Persist complete day/audit pairs so interrupted backfills are reusable."""
    gap_policy = getattr(args, "wind_gap_policy", None)
    contract = {"schema_version": 1, "series": plan.series, "alias": plan.alias,
                "timezone": plan.timezone, "naive_timezone": plan.naive_timezone,
                "cutoff_time": args.cutoff_time, "wind_dst_policy": args.incomplete_dst_policy,
                "request_padding_hours": args.request_padding_hours, "value_scale": 1.}
    if _authorized_gap_hour(plan, day, gap_policy) is not None:
        # Unaffected historic day keys remain byte-for-byte reusable.
        contract["wind_gap_policy"] = gap_policy
        contract["fallback_series"] = NL_ECMWF_COMPONENT
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]
    directory = _safe(plan.output.parent / "_wind_days" / plan.alias / identity)
    checkpoint = replace(plan, output=directory / f"{day.date()}.parquet")
    _safe(checkpoint.output)
    _safe(checkpoint.audit_path)
    if checkpoint.output.exists() or checkpoint.audit_path.exists():
        record = _inspect(checkpoint, day, day, wind_dst_policy=args.incomplete_dst_policy, wind_gap_policy=gap_policy)
        if record["start_day"] != str(day.date()) or record["end_day"] != str(day.date()):
            raise SolarWindSourceError(f"{plan.alias}: day checkpoint has unexpected bounds")
        frame = pd.read_parquet(checkpoint.output)
        frame.attrs["dst_repairs"] = record["dst_duplicate_repairs"]
        frame.attrs["source_substitutions"] = record["source_substitutions"]
        return frame
    frame = _download_wind_day(plan, day, args)
    delivery = pd.DatetimeIndex(frame.value_time_utc)
    repairs = [{**r, "evidence": "raw_Saturn_singleton_at_query_asof"}
               for r in frame.attrs.get("dst_repairs", [])
               if bool(pd.DatetimeIndex(pd.to_datetime(r["physical_hours_utc"], utc=True)).isin(delivery).all())]
    substitutions = frame.attrs.get("source_substitutions", [])
    metadata = _wind_metadata(plan, frame, day, day, repairs, wind_dst_policy=args.incomplete_dst_policy,
                              source_substitutions=substitutions, wind_gap_policy=gap_policy if substitutions else None)
    directory.mkdir(parents=True, exist_ok=True)
    _publish_wind(checkpoint, frame, metadata, wind_gap_policy=gap_policy)
    _inspect(checkpoint, day, day, wind_dst_policy=args.incomplete_dst_policy, wind_gap_policy=gap_policy)
    frame.attrs["dst_repairs"] = repairs
    return frame


def _materialize_wind(plan, first, last, workers, *, merge_existing, wind_dst_policy="duplicate", wind_gap_policy=None):
    """Reuse the day downloader, retaining generic-duplicate repair evidence."""
    args = Namespace(series=plan.series, alias=plan.alias, timezone=plan.timezone,
                     cutoff_timezone=plan.timezone, cutoff_time="08:00", naive_timezone=plan.naive_timezone,
                     saturn_url="https://saturn-energyscan.gem.myengie.com//api", author="BQ6757",
                     request_timeout_seconds=60., retries=3, request_padding_hours=8,
                     incomplete_dst_policy=wind_dst_policy, daily_broadcast=False, hourly_on_the_hour=False,
                     allow_incomplete_days=False, value_scale=1., wind_gap_policy=wind_gap_policy)
    frames, repairs, substitutions, previous_audit = [], [], [], {}
    if merge_existing:
        previous_audit = json.loads(plan.audit_path.read_text(encoding="utf-8"))
        frames.append(pd.read_parquet(plan.output))
        repairs.extend(previous_audit["dst_duplicate_repairs"])
        substitutions.extend(previous_audit.get("source_substitutions", []))
    days = list(pd.date_range(first, last))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_checkpoint_day, plan, day, args): day for day in days}
        for number, future in enumerate(as_completed(pending), 1):
            try:
                frame = future.result()
            except Exception as exc:
                for item in pending:
                    item.cancel()
                raise SolarWindSourceError(f"{plan.alias}: exact native wind acquisition failed: {exc}") from exc
            frames.append(frame)
            substitutions.extend(frame.attrs.get("source_substitutions", []))
            delivery = pd.DatetimeIndex(frame.value_time_utc)
            for repair in frame.attrs.get("dst_repairs", []):
                hours = pd.DatetimeIndex(pd.to_datetime(repair["physical_hours_utc"], utc=True))
                if bool(hours.isin(delivery).all()):
                    repairs.append({**repair, "evidence": "raw_Saturn_singleton_at_query_asof"})
            if number == 1 or number % 25 == 0 or number == len(days):
                print(f"[{plan.alias}] {number}/{len(days)} days complete", flush=True)
    frame = pd.concat(frames, ignore_index=True).sort_values("value_time_utc").reset_index(drop=True)
    if frame.value_time_utc.duplicated().any() or not np.isfinite(frame.value.to_numpy(float)).all() or frame.value.lt(0).any():
        raise SolarWindSourceError(f"{plan.alias}: duplicate physical hours or invalid generation")
    metadata = _wind_metadata(plan, frame, first, last, repairs, wind_dst_policy=wind_dst_policy, previous=previous_audit,
                              source_substitutions=substitutions, wind_gap_policy=wind_gap_policy if substitutions else None)
    _publish_wind(plan, frame, metadata, wind_gap_policy=wind_gap_policy)


def ensure_solar_wind_sources(*, output_root: Path, start_day: str, end_day: str, workers=2,
                              sync=False, wind_dst_policy="duplicate", wind_gap_policy=None) -> dict:
    """Return six alias-keyed audited records; ``sync=False`` never writes.

    Only the two requested native wind series permit bounded initial history
    acquisition. Solar histories must be copied from immutable audited seeds.
    All suffixes are bounded to 31 days; every cache remains lab-isolated.
    """
    output = _safe(output_root)
    first, last = solar._day(start_day), solar._day(end_day)
    if first > last or type(workers) is not int or not 1 <= workers <= 2 or type(sync) is not bool:
        raise SolarWindSourceError("Ordered dates, 1-2 workers, and boolean sync required")
    if wind_dst_policy not in {"raise", "duplicate"}:
        raise SolarWindSourceError("wind_dst_policy must be 'raise' or 'duplicate'; no wind zero-fill policy exists")
    _validate_gap_policy(wind_gap_policy)
    cutoff = (last - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    if cutoff > pd.Timestamp.now(tz="UTC"):
        raise SolarWindSourceError("Requested D-1 08:00 cutoff has not occurred")
    plans = _plans(output)
    for plan in plans:
        _safe(plan.output)
        _safe(plan.audit_path)
    if not sync:
        results = {plan.alias: _inspect(plan, first, last, wind_dst_policy=wind_dst_policy, wind_gap_policy=wind_gap_policy) for plan in plans}
        if any(not row["complete"] for row in results.values()):
            raise SolarWindSourceError("Missing suffix; sync=True is required for isolated extension")
        return results
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(output / ".sync.lock"):
        prepared = []
        for plan in plans:
            exists = plan.output.exists() or plan.audit_path.exists()
            seed, record = ((plan, _inspect(plan, first, last, wind_dst_policy=wind_dst_policy, wind_gap_policy=wind_gap_policy)) if exists
                            else _seed(plan, first, last, wind_dst_policy=wind_dst_policy))
            suffix = (pd.Timestamp(record["end_day"]) + pd.Timedelta(days=1)) if record else first
            if record and suffix > last:
                suffix = None
            limit = MAX_WIND_BACKFILL_DAYS if record is None else MAX_SUFFIX_DAYS
            if suffix is not None and (last - suffix).days + 1 > limit:
                raise SolarWindSourceError(f"{plan.alias}: requested collection exceeds {limit} days")
            prepared.append((plan, seed, record, suffix, exists))
        results = {}
        for plan, seed, seed_record, suffix, exists in prepared:
            if not exists and seed is not None:
                if plan.output.exists() or plan.audit_path.exists():
                    raise SolarWindSourceError(f"{plan.alias}: destination appeared during copy")
                shutil.copy2(seed.output, plan.output)
                shutil.copy2(seed.audit_path, plan.audit_path)
                if solar._sha(plan.output) != seed_record["sha256"] or solar._sha(plan.audit_path) != seed_record["audit_sha256"]:
                    raise SolarWindSourceError(f"{plan.alias}: seed changed during copy")
                if plan.kind == "wind":
                    _annotate_wind_seed(plan, seed_record, wind_dst_policy=wind_dst_policy)
                _inspect(plan, first, min(last, pd.Timestamp(seed_record["end_day"])), wind_dst_policy=wind_dst_policy)
            if suffix is not None:
                if plan.kind == "wind":
                    _materialize_wind(plan, suffix, last, workers, merge_existing=seed_record is not None,
                                      wind_dst_policy=wind_dst_policy, wind_gap_policy=wind_gap_policy)
                else:
                    command = build_command(plan, start_day=suffix, end_day=last, day_workers=workers, merge_existing=True)
                    subprocess.run(command, cwd=ROOT, check=True, shell=False)
            record = _inspect(plan, first, last, wind_dst_policy=wind_dst_policy, wind_gap_policy=wind_gap_policy)
            if not record["complete"]:
                raise SolarWindSourceError(f"{plan.alias}: materialized cache remains incomplete")
            if not exists and seed is not None and (solar._sha(seed.output) != seed_record["sha256"] or solar._sha(seed.audit_path) != seed_record["audit_sha256"]):
                raise SolarWindSourceError(f"{plan.alias}: immutable seed changed during sync")
            record["seed_path"] = str(seed.output) if seed is not None else None
            record["downloaded_suffix_start_day"] = str(suffix.date()) if suffix is not None else None
            record["initial_native_wind_backfill"] = seed_record is None
            results[plan.alias] = record
        return results


__all__ = ["SOLAR_SERIES", "WIND_SERIES", "GENERATION_SERIES", "SolarWindSourceError", "ensure_solar_wind_sources"]
