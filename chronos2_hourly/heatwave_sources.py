"""Isolated, resumable daily forecast-temperature inputs, frozen at D-1 08 h.

The five Saturn curves are daily T2m indices in Celsius, not hourly weather,
observations, daily maxima or daily minima. Broadcasting a published daily
index onto physical delivery hours adds no artificial intraday information.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .process_lock import exclusive_process_lock


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = "Europe/Paris"
COLUMNS = ["value_time_utc", "snapshot_time_utc", "revision_time_utc", "value", "downloaded_at_utc"]
ZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin", "BE": "Europe/Brussels",
         "NL": "Europe/Amsterdam", "ES": "Europe/Madrid"}
TEMPERATURE_SOURCES = {
    f"{zone.lower()}_temperature_fcst": {
        "alias": f"{zone.lower()}_temperature_fcst", "zone": zone,
        "series": f"meteo.nrjscan.{zone.lower()}.t_2m.index.fcst.d",
        "unit": "degC", "timezone": timezone, "naive_timezone": timezone,
        "information_type": "daily_forecast_temperature_index", "daily_broadcast": True,
        "actual_weather_used": False, "hourly_temperature_information": False,
        "maximum_minimum_temperature_information": False,
        "unit_evidence": "existing Saturn weather catalogue contract; provider metadata does not expose units",
    } for zone, timezone in ZONES.items()
}


class HeatwaveSourceError(ValueError):
    """An isolated temperature source does not satisfy its causal contract."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def _rename_new_directory(stage: Path, destination: Path) -> None:
    """Bounded retry for transient Windows scanners; never replace a bundle."""
    for attempt in range(6):
        if destination.exists():
            raise HeatwaveSourceError(f"Immutable temperature destination already exists: {destination}.")
        try:
            stage.rename(destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(min(.2 * 2 ** attempt, 1.5))


def _bounds(first, last) -> pd.DatetimeIndex:
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    if any(pd.isna(d) or d.tzinfo is not None or d != d.normalize() for d in (first, last)) or first > last:
        raise HeatwaveSourceError("Inclusive naive civil dates start_day <= end_day are required.")
    if (last - first).days >= 3660:
        raise HeatwaveSourceError("Temperature materialization is limited to 3660 days.")
    return pd.date_range(first, last, freq="D")


def _physical(first, last) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(first).tz_localize(TIMEZONE),
        (pd.Timestamp(last) + pd.Timedelta(days=1)).tz_localize(TIMEZONE),
        freq="h", inclusive="left").tz_convert("UTC")


def _cutoff(day) -> pd.Timestamp:
    return (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(TIMEZONE).tz_convert("UTC")


def _safe_output(path) -> Path:
    path = Path(path).expanduser().resolve()
    allowed = (ROOT / "data/pit/heatwave").resolve()
    if not allowed.is_relative_to(ROOT.resolve()) or not path.is_relative_to(allowed):
        raise HeatwaveSourceError("Temperature source writes must remain under data/pit/heatwave.")
    return path


def _spec(zone: str) -> dict:
    alias = f"{str(zone).lower()}_temperature_fcst"
    if alias not in TEMPERATURE_SOURCES:
        raise HeatwaveSourceError("Only forecast-temperature zones FR, DE, BE, NL and ES are supported.")
    return TEMPERATURE_SOURCES[alias]


def _frame_contract(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if list(frame.columns) != COLUMNS or frame.empty:
        raise HeatwaveSourceError("A nonempty canonical five-column PIT frame is required.")
    for name in (c for c in COLUMNS if c != "value"):
        if not isinstance(frame[name].dtype, pd.DatetimeTZDtype) or frame[name].isna().any():
            raise HeatwaveSourceError(f"Explicit timezone-aware timestamps required: {name}.")
    stamps = pd.DatetimeIndex(frame.value_time_utc).tz_convert("UTC")
    days = stamps.tz_convert(TIMEZONE).tz_localize(None).normalize()
    if not stamps.equals(_physical(days.min(), days.max())):
        raise HeatwaveSourceError("Temperature physical hours must be unique, ordered and complete (23/24/25 h).")
    cutoffs = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(TIMEZONE).tz_convert("UTC")
    for name in ("snapshot_time_utc", "revision_time_utc"):
        if not pd.DatetimeIndex(frame[name]).tz_convert("UTC").equals(cutoffs):
            raise HeatwaveSourceError("Temperature as-of snapshot/revision must equal D-1 08:00 civil.")
    if (pd.DatetimeIndex(frame.downloaded_at_utc).tz_convert("UTC") < cutoffs).any():
        raise HeatwaveSourceError("A temperature source download cannot predate its query cutoff.")
    if pd.api.types.is_bool_dtype(frame.value.dtype):
        raise HeatwaveSourceError("Boolean indicators are not Celsius temperature forecasts.")
    values = pd.to_numeric(frame.value, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < -90) or np.any(values > 65):
        raise HeatwaveSourceError("Temperature must be finite degrees Celsius in [-90, 65]; no Kelvin or missing-data fill.")
    if pd.Series(values).groupby(days).nunique().gt(1).any():
        raise HeatwaveSourceError("Daily forecast temperature must be constant within each civil day.")
    return days


def _read_source(path: Path, spec: dict, *, seed=False):
    from materialize_saturn_kalman_weather import SeriesPlan, reusable_output

    sidecar = path.with_name(path.name + ".audit.json")
    hashes = {p: _sha(p) for p in (path, sidecar)}
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    plan = SeriesPlan(spec["zone"], "temperature_2m", spec["alias"], spec["series"],
                      spec["timezone"], spec["naive_timezone"], path, True, 1.0, "degC")
    first, last = _bounds(metadata.get("start_day"), metadata.get("end_day"))[[0, -1]]
    valid, reason = reusable_output(plan, start_day=first, end_day=last)
    if not valid:
        raise HeatwaveSourceError(f"Temperature source audit refused: {reason}.")
    required = {"schema_version": 1,
        "causal_contract": "Saturn state queried as-of D-1 civil cutoff",
        "snapshot_time_semantics": "query_asof_cutoff",
        "revision_time_semantics": "query_asof_cutoff; provider insertion timestamp is not returned by Client.get(revision_date=...)",
        "provider_revision_timestamp_available": False,
        "fill_or_interpolation": "daily_value_broadcast_to_physical_hours"}
    enriched = dict(unit="degC", information_type=spec["information_type"],
                    actual_weather_used=False, hourly_temperature_information=False,
                    maximum_minimum_temperature_information=False, production_pit_evidence=False,
                    production_pipeline_evidence=False, promotion_eligible=False)
    # Old generic Saturn sidecars omit these fields. A present contradictory
    # declaration must still fail; a seed can never silently turn GW/K into °C.
    required.update({key: value for key, value in enriched.items() if not seed or key in metadata})
    for name, expected in required.items():
        if metadata.get(name) != expected or (isinstance(expected, bool) and metadata.get(name) is not expected):
            raise HeatwaveSourceError(f"Temperature source semantic contract mismatch: {name}.")
    frame = pd.read_parquet(path)
    days = _frame_contract(frame)
    if any(_sha(p) != digest for p, digest in hashes.items()):
        raise HeatwaveSourceError("Temperature source changed while reading.")
    for name in (c for c in COLUMNS if c != "value"):
        frame[name] = frame[name].astype("datetime64[ns, UTC]")
    return frame, metadata, days, hashes


def audit_temperature_store(path, zone: str, start_day, end_day) -> dict:
    """Read-only fail-closed audit, including bytes, source identity and cutoff."""
    spec, requested = _spec(zone), _bounds(start_day, end_day)
    path = Path(path).resolve()
    frame, metadata, days, hashes = _read_source(path, spec)
    missing = requested.difference(days.unique())
    if len(missing):
        raise HeatwaveSourceError(f"Missing temperature forecast days: {missing.strftime('%Y-%m-%d').tolist()[:10]}.")
    return {**metadata, "ready": True, "blockers": [], "path": str(path),
            "audit_path": str(path.with_name(path.name + ".audit.json")),
            "audit_sha256": hashes[path.with_name(path.name + ".audit.json")],
            "requested_start_day": requested[0].date().isoformat(),
            "requested_end_day": requested[-1].date().isoformat(), "requested_days": len(requested),
            "covered_hours": len(_physical(requested[0], requested[-1])), "missing_hours": 0,
            "cutoff_violations": 0}


def _metadata(frame, spec, **extra) -> dict:
    days = _frame_contract(frame)
    stamps = pd.DatetimeIndex(frame.value_time_utc).tz_convert("UTC")
    return {"schema_version": 1, **deepcopy(spec), "cutoff_timezone": spec["timezone"],
        "cutoff_time": "08:00", "value_scale": 1.0, "incomplete_dst_policy": "duplicate",
        "causal_contract": "Saturn state queried as-of D-1 civil cutoff",
        "snapshot_time_semantics": "query_asof_cutoff",
        "revision_time_semantics": "query_asof_cutoff; provider insertion timestamp is not returned by Client.get(revision_date=...)",
        "provider_revision_timestamp_available": False,
        "fill_or_interpolation": "daily_value_broadcast_to_physical_hours",
        "production_pit_evidence": False, "production_pipeline_evidence": False, "promotion_eligible": False,
        "start_day": days.min().date().isoformat(), "end_day": days.max().date().isoformat(),
        "days": int(days.nunique()), "rows": len(frame),
        "first_delivery_utc": stamps[0].isoformat(), "last_delivery_utc": stamps[-1].isoformat(), **extra}


def _publish_frame(directory: Path, frame, spec, **extra) -> Path:
    directory = _safe_output(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".temperature_", dir=directory.parent) as temporary:
        stage = Path(temporary)
        path = stage / (spec["alias"] + ".parquet")
        frame.loc[:, COLUMNS].to_parquet(path, index=False)
        metadata = _metadata(frame, spec, sha256=_sha(path), **extra)
        _write_json(path.with_name(path.name + ".audit.json"), metadata)
        _read_source(path, spec)
        _rename_new_directory(stage, directory)
    return directory / (spec["alias"] + ".parquet")


def _frozen_seed(directory: Path, spec: dict, seed_root: Path):
    seed_dir = _safe_output(directory / "seed")
    name = spec["alias"] + ".parquet"
    if not seed_dir.exists():
        original = seed_root / name
        _, _, _, hashes = _read_source(original, spec, seed=True)
        with tempfile.TemporaryDirectory(prefix=".seed_", dir=directory) as temporary:
            stage = Path(temporary)
            for path, digest in hashes.items():
                shutil.copy2(path, stage / path.name)
                if _sha(stage / path.name) != digest or _sha(path) != digest:
                    raise HeatwaveSourceError("Incumbent temperature seed changed during immutable copy.")
            _write_json(stage / "origin.json", {"source_path": str(original.resolve()),
                "sha256": hashes[original], "audit_sha256": hashes[original.with_name(original.name + ".audit.json")]})
            _rename_new_directory(stage, seed_dir)
    frame, metadata, days, hashes = _read_source(seed_dir / name, spec, seed=True)
    provenance = json.loads((seed_dir / "origin.json").read_text())
    if provenance.get("sha256") != hashes[seed_dir / name] or provenance.get("audit_sha256") != hashes[seed_dir / (name + ".audit.json")]:
        raise HeatwaveSourceError("Frozen temperature seed provenance differs from its immutable bytes.")
    return frame, days, {**provenance, "frozen_path": str(seed_dir / name)}


def _fetch_day(spec: dict, day: pd.Timestamp) -> pd.DataFrame:
    from materialize_saturn_daily_asof import _one_day
    args = SimpleNamespace(saturn_url="https://saturn-energyscan.gem.myengie.com//api", author="BQ6757",
        request_timeout_seconds=30, series=spec["series"], alias=spec["alias"],
        timezone=spec["timezone"], cutoff_timezone=spec["timezone"], cutoff_time="08:00",
        naive_timezone=spec["naive_timezone"], incomplete_dst_policy="duplicate", request_padding_hours=8,
        retries=2, daily_broadcast=True, hourly_on_the_hour=False, value_scale=1.0, allow_incomplete_days=False)
    frame = _one_day(day, args)
    _frame_contract(frame)
    if not pd.DatetimeIndex(frame.value_time_utc).equals(_physical(day, day)):
        raise HeatwaveSourceError("Fetched temperature does not cover the requested civil day.")
    return frame


def _materialize_one(output: Path, spec: dict, requested, seed_root: Path) -> dict:
    directory = _safe_output(output / spec["alias"])
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(directory / "materialize.lock"):
        bundle = _safe_output(directory / "bundles" / f"{requested[0].date()}_{requested[-1].date()}")
        path = bundle / (spec["alias"] + ".parquet")
        if not bundle.exists():
            seed, seed_days, provenance = _frozen_seed(directory, spec, seed_root)
            blocks = [seed.loc[seed_days.isin(requested)].copy()]
            missing = requested.difference(seed_days.unique())
            for number, day in enumerate(missing, start=1):
                day_dir = _safe_output(directory / "days" / day.date().isoformat())
                day_path = day_dir / (spec["alias"] + ".parquet")
                if not day_dir.exists():
                    _publish_frame(day_dir, _fetch_day(spec, day), spec)
                audit_temperature_store(day_path, spec["zone"], day, day)
                block, _, _, _ = _read_source(day_path, spec)
                if not pd.DatetimeIndex(block.value_time_utc).equals(_physical(day, day)):
                    raise HeatwaveSourceError("Daily temperature checkpoint spans the wrong day.")
                blocks.append(block)
                print(f"[Temperature/{spec['zone']}] {number}/{len(missing)} missing days complete: {day.date()}", flush=True)
            result = pd.concat(blocks, ignore_index=True).sort_values("value_time_utc").reset_index(drop=True)
            for name in (c for c in COLUMNS if c != "value"):
                result[name] = result[name].astype("datetime64[ns, UTC]")
            if not pd.DatetimeIndex(result.value_time_utc).equals(_physical(requested[0], requested[-1])):
                raise HeatwaveSourceError("Refusing to publish incomplete temperature range.")
            _publish_frame(bundle, result, spec, seed_provenance=provenance,
                           seed_hours_reused=len(blocks[0]), materialized_missing_days=len(missing))
        audit = audit_temperature_store(path, spec["zone"], requested[0], requested[-1])
        return {"path": str(path), "specification": deepcopy(spec), "audit": audit}


def materialize_temperature_sources(output_root, start_day, end_day, workers=2, *, seed_root=None) -> dict:
    """Freeze five forecast-only series; resume only valid isolated checkpoints.

    Existing operational caches are read-only seeds. Each final date range is
    immutable and is reused without refreshing or changing its historical bytes.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise HeatwaveSourceError("workers must be an integer from 1 to 4.")
    output, requested = _safe_output(output_root), _bounds(start_day, end_day)
    seed_root = Path(seed_root).resolve() if seed_root is not None else ROOT / "data/pit/kalman_weather"
    output.mkdir(parents=True, exist_ok=True)
    results, failures = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(_materialize_one, output, spec, requested, seed_root): alias
                for alias, spec in TEMPERATURE_SOURCES.items()}
        for job in as_completed(jobs):
            try:
                results[jobs[job]] = job.result()
            except Exception as exc:
                failures.append(f"{jobs[job]}: {type(exc).__name__}: {exc}")
    if failures:
        raise HeatwaveSourceError("Temperature sources incomplete; valid isolated checkpoints preserved. " + "; ".join(failures))
    return {alias: results[alias] for alias in TEMPERATURE_SOURCES}
