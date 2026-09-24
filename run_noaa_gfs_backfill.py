#!/usr/bin/env python
"""Resumable daily NOAA GFS archive backfill, with a verified final aggregate."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import ssl
import sys
import time
from typing import Any, Sequence
import uuid

import httpx
import numpy as np
import pandas as pd

from auxiliary_lab import noaa_gfs as gfs
from auxiliary_lab.weather import ZONE_POINTS, delivery_utc_index

VERSION = 1
VARIABLES = ("temperature_2m_c", "wind_speed_100m_ms", "shortwave_radiation_wm2")
TIME_COLUMNS = ("delivery_start_utc", "run_init_utc", "cutoff_utc", "publication_max_utc")
UNITS = dict(zip(VARIABLES, ("degC", "m/s", "W/m2")))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise gfs.GfsError(f"Oversized backfill metadata: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise gfs.GfsError(f"Expected a metadata object: {path}")
    return value


@contextmanager
def _run_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "backfill.lock"
    try:
        with path.open("xb") as stream:
            identity = os.fstat(stream.fileno())
            stream.write(f"pid={os.getpid()}\n".encode("ascii"))
    except FileExistsError as exc:
        raise gfs.GfsError(f"A backfill already reserved {root}; lock: {path}") from exc
    try:
        yield
    finally:
        gfs._unlink_owned(path, identity)


def _contract(start_day: str, end_day: str, zones: Sequence[str], tolerance: float) -> dict[str, Any]:
    return {
        "schema_version": VERSION, "kind": "noaa_gfs_daily_backfill_contract",
        "start_day": start_day, "end_day": end_day, "zones": list(zones),
        "materializer_schema_version": gfs.SCHEMA_VERSION,
        "materializer_source_sha256": file_sha256(Path(gfs.__file__)),
        "cutoff_time": "08:00", "cutoff_timezone": "Europe/Paris",
        "run_policy": "D-1 00:00 UTC; no cycle substitution",
        "radiation_negative_tolerance_wm2": tolerance,
        "points": {z: [vars(p) for p in ZONE_POINTS[z]] for z in zones},
        "feature_units": UNITS,
    }


def _read_frame(path: Path, *, days: Sequence[str], zones: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    columns = [*TIME_COLUMNS, "delivery_day", *[f"{z.lower()}_gfs_{v}" for z in zones for v in VARIABLES]]
    if set(frame.columns) != set(columns):
        raise gfs.GfsError(f"Weather columns do not match the frozen zones/features: {path}")
    expected = delivery_utc_index(days[0])
    for day in days[1:]:
        expected = expected.append(delivery_utc_index(day))
    if len(frame) != len(expected) or not np.array_equal(pd.DatetimeIndex(frame["delivery_start_utc"]).asi8, expected.asi8):
        raise gfs.GfsError(f"Incomplete, duplicate or unordered physical-hour timeline: {path}")
    for column in TIME_COLUMNS:
        if str(frame[column].dtype) != "datetime64[ns, UTC]" or frame[column].isna().any():
            raise gfs.GfsError(f"Invalid UTC nanosecond datetime column: {column}")
    if frame["delivery_day"].tolist() != expected.tz_convert("Europe/Paris").strftime("%Y-%m-%d").tolist():
        raise gfs.GfsError("Delivery-day labels disagree with physical hours.")
    for day in days:
        subset = frame.loc[frame["delivery_day"] == day]
        run, cutoff = gfs.issue_times(day)
        if not (subset["run_init_utc"].eq(run).all() and subset["cutoff_utc"].eq(cutoff).all() and subset["publication_max_utc"].between(run, cutoff).all()):
            raise gfs.GfsError(f"The fixed run or publication-before-cutoff contract failed: {day}")
    for zone in zones:
        for variable, low, high in ((VARIABLES[0], -123.15, 76.85), (VARIABLES[1], 0, math.hypot(200, 200)), (VARIABLES[2], 0, 1600)):
            values = frame[f"{zone.lower()}_gfs_{variable}"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or np.any(values < low - 1e-10) or np.any(values > high + 1e-10):
                raise gfs.GfsError(f"Invalid physical values: {zone}/{variable}")
    return frame


def validate_partition(path: str | Path, day: str, contract: dict[str, Any], *, expected_manifest_sha256: str | None = None) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Read-only validation of a daily data/manifest pair, including provenance."""
    path = Path(path).resolve()
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise gfs.GfsError(f"Incomplete daily partition: {path}")
    manifest_sha = file_sha256(manifest_path)
    if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        raise gfs.GfsError(f"Registered daily manifest SHA mismatch: {manifest_path}")
    audit = _read_json(manifest_path)
    for key in ("zones", "materializer_source_sha256", "cutoff_time", "cutoff_timezone", "run_policy", "points", "feature_units", "radiation_negative_tolerance_wm2"):
        if audit.get(key) != contract[key]:
            raise gfs.GfsError(f"Daily partition differs from frozen contract: {day}/{key}")
    expected_rows = len(delivery_utc_index(day))
    for key, value in {"schema_version": contract["materializer_schema_version"], "start_day": day, "end_day": day, "day_count": 1, "row_count": expected_rows, "forecast_endpoint_count": expected_rows + 1, "evidence_kind": "historical_archive_publication", "local_prospective_capture": False, "production_pit_evidence": False, "production_pipeline_evidence": False, "promotion_eligible": False}.items():
        if audit.get(key) != value:
            raise gfs.GfsError(f"Invalid daily manifest field: {day}/{key}")
    if Path(audit.get("output_path", "")).resolve() != path:
        raise gfs.GfsError("Daily manifest references a different data file.")
    data_sha = file_sha256(path)
    if audit.get("dataset_sha256") != data_sha or audit.get("output_sha256") != data_sha:
        raise gfs.GfsError(f"Daily Parquet SHA mismatch: {path}")
    frame = _read_frame(path, days=[day], zones=contract["zones"])
    sources = audit.get("sources", [])
    run, cutoff = gfs.issue_times(day)
    first = int((delivery_utc_index(day)[0] - run) / pd.Timedelta(hours=1))
    hours = list(range(first, first + expected_rows + 1))
    if len(sources) != len(hours) or [s.get("forecast_hour") for s in sources] != hours:
        raise gfs.GfsError("Archived forecast endpoints do not match the delivery grid.")
    for source, hour in zip(sources, hours):
        if source.get("url") != gfs.object_url(run, hour) or source.get("run_init_utc") != run.isoformat():
            raise gfs.GfsError("Archived forecast endpoint identity mismatch.")
        published = pd.Timestamp(source["publication_max_utc"])
        if published.tzinfo is None or not run <= published <= cutoff:
            raise gfs.GfsError("Archived endpoint publication is outside its causal interval.")
    publication = max(pd.Timestamp(source["publication_max_utc"]) for source in sources)
    if not frame["publication_max_utc"].eq(publication).all():
        raise gfs.GfsError("Hourly publication disagrees with archived endpoint provenance.")
    record = {"day": day, "output_path": str(path), "manifest_path": str(manifest_path), "dataset_sha256": data_sha, "manifest_sha256": manifest_sha, "row_count": len(frame)}
    return frame, audit, record


def _publish_seed(seed: Path, output: Path, day: str, contract: dict[str, Any]) -> None:
    _, audit, original = validate_partition(seed, day, contract)
    manifest = output.with_suffix(".manifest.json")
    with gfs._output_reservations(output, manifest):
        staged = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
        staged_manifest = manifest.with_name(manifest.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            shutil.copyfile(seed, staged)
            if file_sha256(staged) != original["dataset_sha256"]:
                raise gfs.GfsError("Seed copy SHA mismatch.")
            audit = {**audit, "output_path": str(output.resolve()), "seed_provenance": original}
            staged_manifest.write_bytes(gfs._json_bytes(audit))
            gfs._publish_pair(staged, staged_manifest, output, manifest)
        finally:
            for temporary in (staged, staged_manifest):
                if temporary.exists():
                    temporary.unlink()


def _disk_check(output_root: Path, cache_dir: Path, *, new_days: int, min_free_gib: float, estimated_raw_mb_per_day: float) -> dict[str, Any]:
    def existing_parent(path: Path) -> Path:
        while not path.exists():
            path = path.parent
        return path
    margin = int(min_free_gib * 1024**3)
    raw_estimate = int(new_days * estimated_raw_mb_per_day * 1_000_000)
    output_free = shutil.disk_usage(existing_parent(output_root)).free
    cache_free = shutil.disk_usage(existing_parent(cache_dir)).free
    if output_free < margin + max(8 * 1024**2, new_days * 1024**2) or cache_free < margin + raw_estimate:
        raise gfs.GfsError(f"Insufficient free disk: output={output_free}, cache={cache_free}; raw estimate={raw_estimate}, reserved margin={margin}. No data deleted.")
    return {"free_output_bytes": output_free, "free_cache_bytes": cache_free, "estimated_raw_bytes": raw_estimate, "minimum_free_bytes": margin}


def _journal(root: Path, event: str, **details: Any) -> None:
    value = {"at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "event": event, **details}
    with (root / "journal.jsonl").open("ab") as target:
        target.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        target.flush()
        os.fsync(target.fileno())


def _publication_artifacts(output: Path) -> list[str]:
    """Locate daily publication state without removing or repairing anything."""
    manifest = output.with_suffix(".manifest.json")
    candidates = [output, manifest]
    for destination in (output, manifest):
        candidates.append(destination.with_name(destination.name + ".publish.lock"))
        candidates.extend(destination.parent.glob(destination.name + ".*.tmp"))
    return sorted({str(path) for path in candidates if path.exists() or path.is_symlink()})


def _retryable_protocol_error(error: httpx.RemoteProtocolError) -> bool:
    # The materializer already bounds and wraps other network failures in
    # GfsError. Never unwrap that contract or retry TLS/HTTP-status failures.
    pending: list[BaseException] = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (ssl.SSLError, httpx.HTTPStatusError, gfs.GfsError)):
            return False
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
    return True


def _materialize_day(root: Path, *, day_retries: int, source_sha256: str, kwargs: dict[str, Any]) -> None:
    """Retry only pre-publication protocol interruptions, preserving raw cache."""
    output = Path(kwargs["output_path"])
    day = kwargs["start_day"]
    for attempt in range(day_retries + 1):
        artifacts = _publication_artifacts(output)
        if artifacts:
            raise gfs.GfsError(f"Daily publication state is present; no automatic retry or cleanup: {artifacts}")
        if file_sha256(Path(gfs.__file__)) != source_sha256:
            raise gfs.GfsError("Materializer source changed before a daily attempt.")
        try:
            gfs.materialize_noaa_gfs_weather(**kwargs)
            return
        except httpx.RemoteProtocolError as exc:
            if not _retryable_protocol_error(exc):
                raise
            artifacts = _publication_artifacts(output)
            if artifacts:
                _journal(root, "day_retry_refused", day=day, failed_attempt=attempt + 1,
                         error_type=type(exc).__name__, publication_artifacts=artifacts)
                raise gfs.GfsError("Protocol interruption left daily publication state; refusing retry and automatic cleanup.") from exc
            if attempt == day_retries:
                _journal(root, "day_retries_exhausted", day=day, failed_attempt=attempt + 1,
                         day_retries=day_retries, error_type=type(exc).__name__, error=str(exc))
                raise
            delay = min(2 ** attempt, 8)
            _journal(root, "day_retry_scheduled", day=day, failed_attempt=attempt + 1,
                     next_attempt=attempt + 2, maximum_attempts=day_retries + 1,
                     wait_seconds=delay, error_type=type(exc).__name__, error=str(exc),
                     raw_cache_reused=str(kwargs["cache_dir"]))
            print(f"[GFS backfill] {day}: interrupted response; retry {attempt + 2}/{day_retries + 1} in {delay}s, reusing verified raw cache", flush=True)
            time.sleep(delay)


def validate_aggregate(output_root: str | Path) -> dict[str, Any]:
    """Read-only check of aggregate and every registered daily partition."""
    root = Path(output_root).resolve()
    contract = _read_json(root / "contract.json")
    aggregate = root / "aggregate" / "noaa_gfs_weather.parquet"
    manifest = _read_json(aggregate.with_suffix(".manifest.json"))
    days = pd.date_range(contract["start_day"], contract["end_day"], freq="D").strftime("%Y-%m-%d").tolist()
    records = manifest.get("partitions", [])
    if manifest.get("kind") != "noaa_gfs_verified_daily_aggregate" or manifest.get("complete") is not True or [r.get("day") for r in records] != days:
        raise gfs.GfsError("Aggregate has no complete partition timeline.")
    partition_frames = []
    for record in records:
        daily_frame, _, actual = validate_partition(root / "partitions" / f"{record['day']}.parquet", record["day"], contract, expected_manifest_sha256=record["manifest_sha256"])
        if actual != record:
            raise gfs.GfsError("Aggregate partition registration differs from its data.")
        partition_frames.append(daily_frame)
    data_sha = file_sha256(aggregate)
    if manifest.get("dataset_sha256") != data_sha or manifest.get("output_sha256") != data_sha:
        raise gfs.GfsError("Aggregate Parquet SHA mismatch.")
    frame = _read_frame(aggregate, days=days, zones=contract["zones"])
    if any(manifest.get(k) != contract[k] for k in ("start_day", "end_day", "zones", "cutoff_time", "cutoff_timezone", "materializer_source_sha256")) or any(manifest.get(key) is not False for key in ("production_pit_evidence", "production_pipeline_evidence", "promotion_eligible", "local_prospective_capture")) or manifest.get("row_count") != len(frame) or manifest.get("day_count") != len(days):
        raise gfs.GfsError("Aggregate metadata differs from the frozen contract.")
    try:
        pd.testing.assert_frame_equal(frame, pd.concat(partition_frames, ignore_index=True))
    except AssertionError as exc:
        raise gfs.GfsError("Aggregate values differ from their verified daily partitions.") from exc
    return {"status": "complete", "output_path": str(aggregate), "manifest_path": str(aggregate.with_suffix(".manifest.json")), "dataset_sha256": data_sha, "day_count": len(days), "row_count": len(frame)}


def _aggregate(root: Path, contract: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    output = root / "aggregate" / "noaa_gfs_weather.parquet"
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        return validate_aggregate(root)
    frames = [validate_partition(record["output_path"], record["day"], contract, expected_manifest_sha256=record["manifest_sha256"])[0] for record in records]
    frame = pd.concat(frames, ignore_index=True)
    with gfs._output_reservations(output, manifest):
        staged = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
        staged_manifest = manifest.with_name(manifest.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            frame.to_parquet(staged, index=False)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(staged))
            data_sha = file_sha256(staged)
            audit = {**contract, "kind": "noaa_gfs_verified_daily_aggregate", "complete": True, "timeline_verified": True, "output_path": str(output), "output_sha256": data_sha, "dataset_sha256": data_sha, "row_count": len(frame), "day_count": len(records), "partitions": records, "evidence_kind": "historical_archive_publication", "local_prospective_capture": False, "production_pit_evidence": False, "production_pipeline_evidence": False, "promotion_eligible": False, "source": "NOAA GFS original daily 00Z forecasts", "source_license": "NOAA NODD public use with attribution"}
            staged_manifest.write_bytes(gfs._json_bytes(audit))
            gfs._publish_pair(staged, staged_manifest, output, manifest)
        finally:
            for temporary in (staged, staged_manifest):
                if temporary.exists():
                    temporary.unlink()
    return validate_aggregate(root)


def run_backfill(*, start_day: str, end_day: str, output_root: str | Path, cache_dir: str | Path, zones: Sequence[str] = gfs.DEFAULT_ZONES, seed_directory: str | Path | None = None, max_new_days: int | None = None, workers: int = 2, min_free_gib: float = 20, estimated_raw_mb_per_day: float = 90, timeout_seconds: float = 45, retries: int = 2, ca_bundle: str | None = None, radiation_tolerance: float = 0.5, dry_run: bool = False, day_retries: int = 2) -> dict[str, Any]:
    gfs.issue_times(start_day)
    gfs.issue_times(end_day)
    days = pd.date_range(start_day, end_day, freq="D").strftime("%Y-%m-%d").tolist()
    zones = tuple(z.upper() for z in zones)
    if not 1 <= len(days) <= 3660 or not zones or len(set(zones)) != len(zones) or any(z not in gfs.DEFAULT_ZONES for z in zones):
        raise gfs.GfsError("Use 1..3660 days and unique supported zones.")
    if not 1 <= workers <= 4 or (max_new_days is not None and max_new_days < 0) or not math.isfinite(min_free_gib) or min_free_gib < 1 or not math.isfinite(estimated_raw_mb_per_day) or estimated_raw_mb_per_day <= 0 or not 0 <= radiation_tolerance <= 1:
        raise gfs.GfsError("Invalid worker, daily limit, disk margin or radiation tolerance.")
    if isinstance(day_retries, bool) or not isinstance(day_retries, int) or not 0 <= day_retries <= 4:
        raise gfs.GfsError("day_retries must be an integer between 0 and 4.")
    root, cache = Path(output_root).resolve(), Path(cache_dir).resolve()
    seed = Path(seed_directory).resolve(strict=True) if seed_directory else None
    if seed is not None and not seed.is_dir():
        raise gfs.GfsError("The seed directory must be a directory.")
    contract = _contract(start_day, end_day, zones, radiation_tolerance)
    with _run_lock(root):
        contract_path = root / "contract.json"
        if contract_path.exists() and _read_json(contract_path) != contract:
            raise gfs.GfsError("Resume contract changed (dates, zones, source version or scientific settings); use a new root.")
        inventory_path = root / "inventory.json"
        inventory = _read_json(inventory_path) if inventory_path.exists() else {}
        records, missing, seed_days = {}, [], []
        for day in days:
            output = root / "partitions" / f"{day}.parquet"
            if output.exists() or output.with_suffix(".manifest.json").exists():
                _, _, record = validate_partition(output, day, contract, expected_manifest_sha256=inventory.get(day, {}).get("manifest_sha256"))
                records[day] = record
            elif day in inventory:
                raise gfs.GfsError(f"Registered partition disappeared: {day}")
            elif seed and ((seed / f"{day}.parquet").exists() or (seed / f"{day}.manifest.json").exists()):
                validate_partition(seed / f"{day}.parquet", day, contract)
                seed_days.append(day)
            else:
                missing.append(day)
        planned = len(missing) if max_new_days is None else min(max_new_days, len(missing))
        disk = _disk_check(root, cache, new_days=planned, min_free_gib=min_free_gib, estimated_raw_mb_per_day=estimated_raw_mb_per_day)
        plan = {"status": "planned", "total_days": len(days), "verified_days": len(records), "seed_days": seed_days, "new_days_this_run": planned, "remaining_new_days": len(missing), "disk": disk, "day_retries": day_retries}
        if dry_run:
            return plan
        if not contract_path.exists():
            gfs._write(contract_path, gfs._json_bytes(contract))
        inventory.update(records)
        gfs._write(inventory_path, gfs._json_bytes(inventory))
        _journal(root, "started", **plan)
        durations, new_count = [], 0
        try:
            for day in days:
                if day in records or (day not in seed_days and max_new_days is not None and new_count >= max_new_days):
                    continue
                started = time.monotonic()
                _disk_check(root, cache, new_days=1, min_free_gib=min_free_gib, estimated_raw_mb_per_day=estimated_raw_mb_per_day)
                if file_sha256(Path(gfs.__file__)) != contract["materializer_source_sha256"]:
                    raise gfs.GfsError("Materializer source changed during backfill.")
                output = root / "partitions" / f"{day}.parquet"
                if day in seed_days:
                    _publish_seed(seed / f"{day}.parquet", output, day, contract)
                    event = "seed_imported"
                else:
                    _materialize_day(root, day_retries=day_retries,
                                     source_sha256=contract["materializer_source_sha256"], kwargs={
                                         "start_day": day, "end_day": day, "output_path": output,
                                         "cache_dir": cache, "zones": zones, "workers": workers,
                                         "timeout_seconds": timeout_seconds, "retries": retries,
                                         "ca_bundle": ca_bundle, "radiation_tolerance": radiation_tolerance,
                                     })
                    new_count += 1
                    event = "day_completed"
                _, _, record = validate_partition(output, day, contract)
                records[day] = record
                inventory[day] = record
                gfs._write(inventory_path, gfs._json_bytes(inventory))
                elapsed = time.monotonic() - started
                if event == "day_completed":
                    durations.append(elapsed)
                remaining = len(days) - len(records)
                eta = (sum(durations) / len(durations)) * remaining if durations else None
                _journal(root, event, **record, elapsed_seconds=elapsed, estimated_remaining_seconds=eta)
                eta_text = "unknown" if eta is None else f"{eta / 3600:.2f}h"
                print(f"[GFS backfill] {day}: {event}; {len(records)}/{len(days)} days; {elapsed:.1f}s; ETA(full)={eta_text}", flush=True)
            if len(records) == len(days):
                result = _aggregate(root, contract, [records[day] for day in days])
                _journal(root, "aggregate_completed", **result)
                return result
            result = {"status": "partial", "verified_days": len(records), "total_days": len(days), "new_days_completed": new_count, "remaining_days": [day for day in days if day not in records], "output_root": str(root)}
            _journal(root, "paused_at_daily_limit", **result)
            return result
        except BaseException as exc:
            _journal(root, "failed", error_type=type(exc).__name__, error=str(exc), verified_days=len(records))
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-day", required=True)
    parser.add_argument("--end-day", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--seed-directory")
    parser.add_argument("--dependency-directory")
    parser.add_argument("--zones", nargs="+", default=list(gfs.DEFAULT_ZONES))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-new-days", type=int)
    parser.add_argument("--min-free-gib", type=float, default=20)
    parser.add_argument("--estimated-raw-mb-per-day", type=float, default=90)
    parser.add_argument("--timeout-seconds", type=float, default=45)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--day-retries", type=int, default=2,
                        help="Daily retries for direct protocol interruptions only (0..4); raw cache is reused.")
    parser.add_argument("--ca-bundle")
    parser.add_argument("--radiation-tolerance", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args())
    dependencies = args.pop("dependency_directory")
    if dependencies:
        path = Path(dependencies).resolve(strict=True)
        if not path.is_dir():
            parser.error("--dependency-directory must be a directory")
        sys.path.insert(0, str(path))
    print(json.dumps(run_backfill(**args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
