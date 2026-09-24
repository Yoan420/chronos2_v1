"""Isolated CWE nuclear capacity forecasts; never substitute Pmax for generation.

FR generation is pinned by the caller.  This module freezes the existing BE/NL
capacity archives, queries only absent civil days at D-1 08:00, and publishes
immutable complete range bundles.  It never updates the incumbent source files.
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

import numpy as np
import pandas as pd

from .process_lock import exclusive_process_lock


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = "Europe/Paris"
FORECAST_TYPE = "available_capacity_forecast_daily_broadcast"
FILL_POLICY = "daily_capacity_forecast_broadcast; no missing-day substitution"
COLUMNS = ["value_time_utc", "snapshot_time_utc", "revision_time_utc", "value", "downloaded_at_utc"]
CWE_CAPACITY_SOURCES = {
    f"{zone.lower()}_nuclear_available_gw": {
        "alias": f"{zone.lower()}_nuclear_available_gw", "zone": zone,
        "series": f"power.nrjscan.{zone.lower()}.3mv.availability.pmax.type.nuclear.gw",
        "unit": "GW", "forecast_type": FORECAST_TYPE, "information_type": "capacity_forecast",
        "daily_broadcast": True, "maximum_plausible_gw": 10. if zone == "BE" else 5.,
        "seed_relative_path": f"data/pit/marginal_cost_expert/capacities/{zone.lower()}_nuclear_available_gw.parquet",
    }
    for zone in ("BE", "NL")
}
STRUCTURAL_ZERO_COUNTRIES = {
    "DE": {"value_gw": 0., "effective_from": "2023-04-16", "model_channel": False,
           "reason": "Last power reactors closed on 2023-04-15, before this experiment's support.",
           "source": "https://www.bundeswirtschaftsministerium.de/Redaktion/DE/Pressemitteilungen/2023/04/20230413-deutschland-beendet-das-zeitalter-der-atomkraft.html"},
    "AT": {"value_gw": 0., "model_channel": False, "reason": "No nuclear power plant in operation.",
           "source": "https://www.iaea.org/sites/default/files/joint_convention_8th_national_report_of_austria_at1.pdf"},
    "LU": {"value_gw": 0., "model_channel": False, "reason": "No domestic nuclear power plant.",
           "source": "https://mint.gouvernement.lu/dam-assets/publications/guide-manuel/PNOS-final.pdf"},
}


class NuclearCweSourceError(ValueError):
    """CWE inputs cannot satisfy the audited capacity-forecast contract."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: dict) -> None:
    # Used only inside private staging directories or for new immutable days.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def _bounds(start_day, end_day) -> pd.DatetimeIndex:
    first, last = pd.Timestamp(start_day), pd.Timestamp(end_day)
    if any(pd.isna(d) or d.tzinfo is not None or d != d.normalize() for d in (first, last)) or first > last:
        raise NuclearCweSourceError("Inclusive naive civil dates start_day <= end_day are required.")
    if first < pd.Timestamp("2023-04-16"):
        raise NuclearCweSourceError("The structural-zero Germany contract starts on 2023-04-16.")
    return pd.date_range(first, last, freq="D")


def _physical(first, last) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(first).tz_localize(TIMEZONE),
                         (pd.Timestamp(last)+pd.Timedelta(days=1)).tz_localize(TIMEZONE),
                         freq="h", inclusive="left").tz_convert("UTC")


def _cutoff(day) -> pd.Timestamp:
    return (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize(TIMEZONE).tz_convert("UTC")


def _safe_output(path: Path | str) -> Path:
    target = Path(path).expanduser().resolve()
    allowed = (ROOT / "data/pit/nuclear_cwe").resolve()
    if not allowed.is_relative_to(ROOT.resolve()):
        raise NuclearCweSourceError("The isolated CWE source root cannot redirect outside the project.")
    if target != allowed and not target.is_relative_to(allowed):
        raise NuclearCweSourceError("CWE source writes must remain under data/pit/nuclear_cwe.")
    return target


def _specification(specification: dict) -> dict:
    if not isinstance(specification, dict):
        raise NuclearCweSourceError("An explicit source specification is required.")
    expected = CWE_CAPACITY_SOURCES.get(specification.get("alias"))
    if expected is None or specification != expected:
        raise NuclearCweSourceError("Only the explicitly catalogued BE/NL nuclear forecast Pmax sources are allowed.")
    return expected


def _frame_contract(frame: pd.DataFrame, spec: dict, metadata: dict) -> pd.DatetimeIndex:
    if list(frame.columns) != COLUMNS or frame.empty:
        raise NuclearCweSourceError("Nonempty canonical five-column PIT frame required.")
    for name in (c for c in COLUMNS if c != "value"):
        if not isinstance(frame[name].dtype, pd.DatetimeTZDtype) or frame[name].isna().any():
            raise NuclearCweSourceError(f"Explicit aware timestamps required: {name}.")
    stamps = pd.DatetimeIndex(frame.value_time_utc).tz_convert("UTC")
    if stamps.has_duplicates or not stamps.is_monotonic_increasing or not stamps.equals(stamps.floor("h")):
        raise NuclearCweSourceError("Unique increasing physical hourly products are required.")
    local_days = stamps.tz_convert(TIMEZONE).tz_localize(None).normalize()
    first, last = local_days.min(), local_days.max()
    if not stamps.equals(_physical(first, last)):
        raise NuclearCweSourceError("The capacity source has missing physical hours or days.")
    cutoffs = (local_days-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize(TIMEZONE).tz_convert("UTC")
    for column in ("snapshot_time_utc", "revision_time_utc"):
        if not pd.DatetimeIndex(frame[column]).tz_convert("UTC").equals(cutoffs):
            raise NuclearCweSourceError("Pmax snapshot/revision must equal the civil D-1 08:00 query cutoff.")
    if (pd.DatetimeIndex(frame.downloaded_at_utc).tz_convert("UTC") < cutoffs).any():
        raise NuclearCweSourceError("A source download cannot predate its as-of query cutoff.")
    values = pd.to_numeric(frame.value, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > spec["maximum_plausible_gw"]).any():
        raise NuclearCweSourceError("Pmax values must be finite nonnegative GW in the country-specific plausible range.")
    if pd.Series(values).groupby(local_days).nunique().gt(1).any():
        raise NuclearCweSourceError("Daily capacity broadcast must be constant within each civil day.")
    actual = {"start_day": first.date().isoformat(), "end_day": last.date().isoformat(),
              "days": int(local_days.nunique()), "rows": len(frame)}
    if any(metadata.get(name) != value for name, value in actual.items()):
        raise NuclearCweSourceError("Source audit delivery span/row count disagrees with the Parquet.")
    return local_days


def _read_source(path: Path, spec: dict, *, seed: bool = False):
    sidecar = path.with_name(path.name + ".audit.json")
    before = {_path: _sha(_path) for _path in (path, sidecar)}
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    contract = {"schema_version": 1, "alias": spec["alias"], "series": spec["series"], "unit": "GW",
                "information_type": "capacity_forecast", "daily_broadcast": True,
                "cutoff_time": "08:00", "cutoff_timezone": TIMEZONE, "complete": True,
                "provider_revision_timestamp_available": False, "production_pit_evidence": False,
                "revision_time_semantics": "query_asof_cutoff", "fill_or_interpolation": FILL_POLICY}
    if not seed:
        contract.update(forecast_type=FORECAST_TYPE, national_fleet_attested=False)
    for key, value in contract.items():
        if metadata.get(key) != value or (isinstance(value, bool) and metadata.get(key) is not value):
            raise NuclearCweSourceError(f"Source audit contract mismatch: {key}.")
    if metadata.get("sha256") != before[path]:
        raise NuclearCweSourceError("Source Parquet SHA256 mismatch.")
    frame = pd.read_parquet(path)
    local_days = _frame_contract(frame, spec, metadata)
    # Arrow may restore millisecond precision whereas a fresh query has seconds
    # or nanoseconds. Normalize only storage dtype after awareness validation;
    # otherwise concatenating equivalent UTC timezones can become object dtype.
    for name in (c for c in COLUMNS if c != "value"):
        frame[name] = frame[name].astype("datetime64[ns, UTC]")
    if any(_sha(p) != digest for p, digest in before.items()):
        raise NuclearCweSourceError("Source changed during reading.")
    return frame, metadata, local_days, before


def audit_cwe_source(path: Path | str, specification: dict, start_day, end_day) -> dict:
    """Read-only fail-closed audit; incomplete/invalid bundles raise ValueError."""
    spec = _specification(specification)
    requested = _bounds(start_day, end_day)
    path = Path(path).resolve()
    frame, metadata, local_days, hashes = _read_source(path, spec)
    missing = requested.difference(local_days.unique())
    if len(missing):
        raise NuclearCweSourceError(f"Missing requested capacity days: {[d.date().isoformat() for d in missing[:10]]}.")
    expected = _physical(requested[0], requested[-1])
    return {**metadata, "path": str(path), "audit_path": str(path.with_name(path.name + ".audit.json")),
            "requested_start_day": requested[0].date().isoformat(), "requested_end_day": requested[-1].date().isoformat(),
            "requested_days": len(requested), "covered_hours": len(expected), "expected_hours": len(expected),
            "missing_hours": 0, "cutoff_violations": 0, "audit_sha256": hashes[path.with_name(path.name + ".audit.json")]}


def _frozen_seed(directory: Path, spec: dict):
    seed = _safe_output(directory / "seed")
    name = spec["alias"] + ".parquet"
    if not seed.exists():
        original = ROOT / spec["seed_relative_path"]
        _, _, _, hashes = _read_source(original, spec, seed=True)
        with tempfile.TemporaryDirectory(prefix=".seed_", dir=directory) as temporary:
            staging = Path(temporary)
            for path, digest in hashes.items():
                shutil.copy2(path, staging / path.name)
                if _sha(staging / path.name) != digest or _sha(path) != digest:
                    raise NuclearCweSourceError("Incumbent capacity seed changed while copying.")
            _json(staging / "seed_origin.json", {"source_path": str(original.resolve()),
                  "source_sha256": hashes[original], "source_audit_sha256": hashes[original.with_name(original.name + ".audit.json")],
                  "copied_at_utc": pd.Timestamp.now(tz="UTC").isoformat()})
            staging.rename(seed)
    frame, metadata, _, hashes = _read_source(seed / name, spec, seed=True)
    origin = json.loads((seed / "seed_origin.json").read_text(encoding="utf-8"))
    if origin.get("source_sha256") != hashes[seed / name] or origin.get("source_audit_sha256") != hashes[seed / (name + ".audit.json")]:
        raise NuclearCweSourceError("Frozen seed provenance disagrees with its immutable bytes.")
    return frame, {**origin, "frozen_path": str(seed / name), "start_day": metadata["start_day"], "end_day": metadata["end_day"]}


def _fetch_day(spec: dict, day: pd.Timestamp) -> dict:
    # Pure fetching helpers from the already audited daily-capacity materializer;
    # its writer is NOT called because it targets the incumbent namespace.
    from marginal_cost_expert.sources import _client, _one_day
    error = None
    for attempt in range(2):
        try:
            value = _one_day(_client(), spec["series"], day)
            if value is None:
                raise NuclearCweSourceError("No finite same-day forecast Pmax at D-1 08:00.")
            return {"schema_version": 1, "alias": spec["alias"], "series": spec["series"],
                    "day": day.date().isoformat(), "cutoff_utc": _cutoff(day).isoformat(),
                    "value_gw": float(value), "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "forecast_type": FORECAST_TYPE, "provider_revision_timestamp_available": False}
        except Exception as exc:
            error = exc
            if attempt == 0:
                time.sleep(1)
    raise NuclearCweSourceError(f"{spec['alias']}/{day.date()}: {error}") from error


def _day_frame(record: dict, spec: dict, day: pd.Timestamp) -> pd.DataFrame:
    expected = {"schema_version": 1, "alias": spec["alias"], "series": spec["series"],
                "day": day.date().isoformat(), "cutoff_utc": _cutoff(day).isoformat(),
                "forecast_type": FORECAST_TYPE, "provider_revision_timestamp_available": False}
    if any(record.get(k) != v for k, v in expected.items()):
        raise NuclearCweSourceError("Immutable day checkpoint identity/cutoff mismatch.")
    value = record.get("value_gw")
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= spec["maximum_plausible_gw"]:
        raise NuclearCweSourceError("Day checkpoint requires finite nonnegative capacity in GW.")
    downloaded = pd.Timestamp(record.get("downloaded_at_utc"))
    if pd.isna(downloaded) or downloaded.tzinfo is None or downloaded < _cutoff(day):
        raise NuclearCweSourceError("Day checkpoint download timestamp is absent, naive or pre-cutoff.")
    frame = pd.DataFrame({"value_time_utc": _physical(day, day), "snapshot_time_utc": _cutoff(day),
                          "revision_time_utc": _cutoff(day), "value": float(value),
                          "downloaded_at_utc": downloaded.tz_convert("UTC")})
    for name in (c for c in COLUMNS if c != "value"):
        frame[name] = frame[name].astype("datetime64[ns, UTC]")
    return frame


def _record_sha(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _materialize_one(output: Path, spec: dict, requested: pd.DatetimeIndex) -> dict:
    directory = _safe_output(output / spec["alias"])
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(directory / "materialize.lock"):
        bundle = _safe_output(directory / "bundles" / f"{requested[0].date()}_{requested[-1].date()}")
        target = bundle / (spec["alias"] + ".parquet")
        if bundle.exists():
            audit = audit_cwe_source(target, spec, requested[0], requested[-1])
            return {"path": str(target), "specification": deepcopy(spec), "audit": audit}
        frame, provenance = _frozen_seed(directory, spec)
        local = pd.DatetimeIndex(frame.value_time_utc).tz_convert(TIMEZONE).tz_localize(None).normalize()
        blocks = [frame.loc[local.isin(requested)].copy()]
        missing = requested.difference(local.unique())
        day_dir = _safe_output(directory / "days")
        day_dir.mkdir(exist_ok=True)
        evidence = []
        errors = []
        for day in missing:
            checkpoint = _safe_output(day_dir / (day.date().isoformat() + ".json"))
            try:
                if checkpoint.exists():
                    before = _sha(checkpoint)
                    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
                    record = envelope.get("record", {})
                    if envelope.get("sha256") != _record_sha(record) or _sha(checkpoint) != before:
                        raise NuclearCweSourceError("Immutable day checkpoint SHA256 mismatch.")
                else:
                    record = _fetch_day(spec, day)
                    _day_frame(record, spec, day)
                    with tempfile.TemporaryDirectory(prefix=".day_", dir=day_dir) as temporary:
                        staged = Path(temporary) / checkpoint.name
                        _json(staged, {"record": record, "sha256": _record_sha(record)})
                        staged.rename(checkpoint)
                blocks.append(_day_frame(record, spec, day))
                evidence.append({**record, "checkpoint_sha256": _sha(checkpoint)})
            except Exception as exc:
                errors.append(f"{day.date()}: {type(exc).__name__}: {exc}")
        if errors:
            raise NuclearCweSourceError(f"{spec['alias']}: no complete bundle published; valid day checkpoints preserved. " + " | ".join(errors))
        merged = pd.concat(blocks, ignore_index=True).sort_values("value_time_utc").reset_index(drop=True)
        metadata = {"schema_version": 1, "alias": spec["alias"], "zone": spec["zone"], "series": spec["series"],
                    "unit": "GW", "forecast_type": FORECAST_TYPE, "information_type": "capacity_forecast",
                    "daily_broadcast": True, "cutoff_time": "08:00", "cutoff_timezone": TIMEZONE,
                    "snapshot_time_semantics": "query_asof_cutoff", "revision_time_semantics": "query_asof_cutoff",
                    "provider_revision_timestamp_available": False, "production_pit_evidence": False,
                    "promotion_eligible": False, "national_fleet_attested": False, "complete": True,
                    "fill_or_interpolation": FILL_POLICY, "start_day": requested[0].date().isoformat(),
                    "end_day": requested[-1].date().isoformat(), "days": len(requested), "rows": len(merged),
                    "source_seed": provenance, "seed_hours_reused": len(blocks[0]), "queried_missing_days": len(missing),
                    "daily_query_evidence": evidence, "structural_zero_countries": deepcopy(STRUCTURAL_ZERO_COUNTRIES),
                    "approximation": "Forecast Pmax is available nuclear capacity, not predicted generation; one daily value is broadcast to 23/24/25 physical hours. National fleet coverage and original publication timestamps are not certified."}
        _frame_contract(merged, spec, metadata)
        bundle.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".bundle_", dir=bundle.parent) as temporary:
            staging = Path(temporary)
            temporary_source = staging / target.name
            merged.to_parquet(temporary_source, index=False)
            metadata["sha256"] = _sha(temporary_source)
            _json(staging / (target.name + ".audit.json"), metadata)
            audit_cwe_source(temporary_source, spec, requested[0], requested[-1])
            staging.rename(bundle)
        return {"path": str(target), "specification": deepcopy(spec),
                "audit": audit_cwe_source(target, spec, requested[0], requested[-1])}


def materialize_cwe_sources(output_root: Path | str, start_day, end_day, workers: int = 2) -> dict:
    """Freeze BE/NL Pmax sources and query only missing days; FR is caller-owned.

    Results map alias to ``path``, ``specification`` and ``audit``. Completed range
    bundles never change; successful missing-day checkpoints survive interruption.
    A corrupt existing bundle/checkpoint fails closed, without overwriting it.
    """
    if type(workers) is not int or not 1 <= workers <= 2:
        raise NuclearCweSourceError("Use one or two Saturn workers.")
    requested = _bounds(start_day, end_day)
    if _cutoff(requested[-1]) > pd.Timestamp.now(tz="UTC"):
        raise NuclearCweSourceError("The requested final D-1 08:00 cutoff has not been reached.")
    output = _safe_output(output_root)
    output.mkdir(parents=True, exist_ok=True)
    results, errors = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_materialize_one, output, spec, requested): alias for alias, spec in CWE_CAPACITY_SOURCES.items()}
        for future in as_completed(futures):
            alias = futures[future]
            try:
                results[alias] = future.result()
            except Exception as exc:
                errors.append(f"{alias}: {type(exc).__name__}: {exc}")
    if errors:
        raise NuclearCweSourceError("CWE source materialization incomplete; valid isolated artifacts preserved. " + " | ".join(errors))
    return {alias: results[alias] for alias in CWE_CAPACITY_SOURCES}
