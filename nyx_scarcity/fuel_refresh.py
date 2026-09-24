"""Bounded TTF/EUA extension in a brand-new scarcity-owned source snapshot.

The existing fuel materializer appends only the missing delivery suffix and
requeries twenty overlap days to verify immutable history and rolling features.
Its stored CCGT assumptions are preserved solely to satisfy that source bundle
contract. The challenger recomputes its own cost proxies from raw TTF/EUA;
this module neither invents native Saturn CGC nor converts CO2 a second time.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd

from economic_value.data import _day, _stable_bytes, load_report_panel
from materialize_saturn_kalman_fuel import FuelMaterializationError, OUTPUT_NAME, WARMUP_DAYS, _load_existing
from .data import ScarcityDataError, _path, _read_source, source_registry


LOGGER = logging.getLogger(__name__)
MAX_SUFFIX_DAYS = 30


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _namespace(root: Path) -> Path:
    directory = root / "runs" / "experiments" / "nyx_scarcity_v1" / "source_refresh"
    if directory.resolve() != directory or not directory.is_relative_to(root):
        raise ScarcityDataError("Fuel refresh cannot follow aliases outside its isolated namespace.")
    return directory


def _refresh_parameters(config: dict) -> dict[str, int]:
    values = config.get("refresh", {})
    specs = {"workers": (2, 2), "max_suffix_days": (30, MAX_SUFFIX_DAYS),
             "retries": (2, 3), "request_timeout_seconds": (45, 120)}
    if not isinstance(values, dict) or set(values).difference(specs):
        raise ScarcityDataError("Unknown fuel refresh setting.")
    result = {}
    for name, (default, maximum) in specs.items():
        value = values.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ScarcityDataError(f"refresh.{name} must be an integer from 1 to {maximum}.")
        result[name] = value
    return result


def _assumptions(audit: dict) -> dict[str, float]:
    values = audit.get("ccgt_assumptions")
    expected = {"efficiency", "emission_tco2_mwh", "vom_eur_mwh"}
    if not isinstance(values, dict) or set(values) != expected:
        raise ScarcityDataError("The fuel seed must explicitly declare its three stored CCGT assumptions.")
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ScarcityDataError(f"Invalid seed CCGT assumption: {name}.")
    if not 0 < values["efficiency"] <= 1:
        raise ScarcityDataError("Invalid seed efficiency.")
    return {"ccgt_efficiency": float(values["efficiency"]),
            "ccgt_emission_tco2_mwh": float(values["emission_tco2_mwh"]),
            "ccgt_vom_eur_mwh": float(values["vom_eur_mwh"])}


def _unchanged(evidence: dict) -> bool:
    path = Path(evidence["path"])
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == evidence["sha256"]


def _existing(*args, **kwargs):
    try:
        return _load_existing(*args, **kwargs)
    except FuelMaterializationError as exc:
        raise ScarcityDataError(str(exc)) from exc


def refresh_fuels(config: dict, *, root: Path) -> dict:
    """Read validated seeds; append at most thirty days to private copies only.

    Returns the copied config and an explicit readiness flag. A failed command
    leaves only an isolated diagnostic directory, never a modified live cache.
    """
    root = Path(root).resolve()
    namespace = _namespace(root)
    refresh_settings = _refresh_parameters(config)
    registry = source_registry()
    overrides = config.get("data", {}).get("source_overrides", {})
    if not isinstance(overrides, dict) or set(overrides).difference(registry):
        raise ScarcityDataError("Fuel source overrides must use the exact audited registry.")
    paths = {key: _path(root, overrides.get(key, registry[key]["path"])) for key in ("ttf", "eua")}
    if paths["ttf"] != paths["eua"]:
        raise ScarcityDataError("TTF and EUA must come from one jointly audited fuel seed bundle.")
    source = paths["ttf"]
    sidecar = Path(str(source) + ".audit.json")
    raw, source_evidence = _stable_bytes(source)
    audit_raw, audit_evidence = _stable_bytes(sidecar)
    seed_audit = json.loads(audit_raw)
    parameters = _assumptions(seed_audit)
    first = pd.Timestamp(_day(seed_audit.get("start_day")))
    # Full existing validation includes schema, exact hourly timeline, source
    # publication cutoff, identities, derived formulae and manifest SHA.
    seed, _, last = _existing(source, sidecar, requested_start=first, **parameters)
    if not _unchanged(source_evidence) or not _unchanged(audit_evidence):
        raise ScarcityDataError("Fuel seed changed concurrently during validation.")
    chosen = config.get("delivery_day")
    if chosen is None:
        _, baseline = load_report_panel(root, config.get("zones", ["FR", "DE", "BE", "NL"]),
                                        ["nuclear_kalman"], end_day=config.get("end_day"))
        chosen = baseline["delivery_day"]
    chosen = _day(chosen)
    final = pd.Timestamp(chosen)
    if final < first:
        raise ScarcityDataError("Requested fuel day precedes the immutable seed.")
    suffix_days = max(0, (final - last).days)
    if suffix_days > refresh_settings["max_suffix_days"]:
        raise ScarcityDataError(f"Fuel suffix is {suffix_days} days; Refresh is limited to {refresh_settings['max_suffix_days']}, not a full backfill.")
    expected_seed = pd.DatetimeIndex(pd.to_datetime(seed.value_time_utc, utc=True))
    for key in ("ttf", "eua"):
        spec = {**registry[key], "path": str(source.relative_to(root))}
        selected, evidence = _read_source(root, key, spec, expected_seed)
        if evidence.get("sha256") != source_evidence["sha256"] or selected[spec["feature"]].isna().any():
            raise ScarcityDataError(f"{key}: incomplete or concurrently changed seed selection.")
    # Nothing is created before all source identities and bounds are checked.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    directory = namespace / stamp / "fuel"
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / OUTPUT_NAME
    output_audit = Path(str(output) + ".audit.json")
    output.write_bytes(raw)
    output_audit.write_bytes(audit_raw)
    _existing(output, output_audit, requested_start=first, **parameters)
    copied = deepcopy(config)
    copied["delivery_day"] = chosen
    copied.setdefault("data", {}).setdefault("source_overrides", {}).update(
        {key: str(output.relative_to(root)) for key in ("ttf", "eua")})
    saved = directory / "config.json"
    audit_path = directory / "fuel_refresh_audit.json"
    log = directory / "fuel_materializer.log"
    _write_json(saved, copied)
    audit = {"schema_version": 1, "status": "running", "delivery_day": chosen,
             "source_dir": str(directory), "saved_config": str(saved), "audit_path": str(audit_path),
             "source_path": str(output), "log": str(log), "seed": source_evidence,
             "seed_audit": audit_evidence, "seed_start_day": str(first.date()), "seed_end_day": str(last.date()),
             "requested_suffix_days": suffix_days, "maximum_suffix_days": refresh_settings["max_suffix_days"],
             "hard_maximum_suffix_days": MAX_SUFFIX_DAYS,
             "overlap_query_days": WARMUP_DAYS if suffix_days else 0,
             "maximum_series_day_queries": 2 * (suffix_days + WARMUP_DAYS) if suffix_days else 0,
             "maximum_attempts_per_query": refresh_settings["retries"],
             "request_timeout_seconds": refresh_settings["request_timeout_seconds"],
             "series_workers": 1, "day_workers": 1, "skip_residual_load": True,
             "seed_ccgt_assumptions_preserved": seed_audit["ccgt_assumptions"],
             "native_saturn_cgc_used": False, "uses_only_raw_ttf_eua_for_challenger_cost": True,
             "cutoff": "D-1 08:00 Europe/Paris civil", "required_sources_complete": False,
             "original_sources_modified": False, "production_modified": False,
             "diagnostic_only": True, "production_pit_evidence": False, "promotion_eligible": False}
    _write_json(audit_path, audit)
    LOGGER.info("[Scarcity fuel] seed=%s..%s, append=%s days, overlap check=%s days, isolated=%s",
                first.date(), last.date(), suffix_days, audit["overlap_query_days"], directory)
    try:
        if suffix_days:
            command = [sys.executable, str(root / "materialize_saturn_kalman_fuel.py"),
                       "--output-dir", str(directory), "--start-day", str(first.date()), "--end-day", chosen,
                       "--skip-residual-load", "--series-workers", "1", "--day-workers", "1",
                       "--retries", str(refresh_settings["retries"]),
                       "--request-timeout-seconds", str(refresh_settings["request_timeout_seconds"]),
                       "--ccgt-efficiency", str(parameters["ccgt_efficiency"]),
                       "--ccgt-emission-tco2-mwh", str(parameters["ccgt_emission_tco2_mwh"]),
                       "--ccgt-vom-eur-mwh", str(parameters["ccgt_vom_eur_mwh"])]
            audit["command"] = command
            _write_json(audit_path, audit)
            timeout = 2 * (suffix_days + WARMUP_DAYS) * (refresh_settings["request_timeout_seconds"] * refresh_settings["retries"] + 15) + 120
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(command, cwd=root, shell=False, stdout=stream,
                                           stderr=subprocess.STDOUT, check=False, timeout=timeout)
            audit["returncode"] = completed.returncode
            if completed.returncode:
                raise ScarcityDataError(f"Fuel materializer returned {completed.returncode}; inspect the private log.")
        else:
            audit.update(command=[], returncode=0)
        frame, _, produced_end = _existing(output, output_audit, requested_start=first, **parameters)
        if produced_end != max(last, final):
            raise ScarcityDataError("Fuel output expanded outside the requested suffix or is incomplete.")
        prefix = frame.loc[pd.to_datetime(frame.value_time_utc, utc=True).isin(expected_seed)].reset_index(drop=True)
        pd.testing.assert_frame_equal(seed.reset_index(drop=True), prefix, check_dtype=False, check_exact=True)
        audit["seed_history_semantically_unchanged"] = True
        stop = (final + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
        expected = pd.date_range(first.tz_localize("Europe/Paris"), stop, inclusive="left", freq="h").tz_convert("UTC")
        validations = {}
        for key in ("ttf", "eua"):
            spec = {**registry[key], "path": str(output.relative_to(root))}
            selected, evidence = _read_source(root, key, spec, expected)
            missing = int(selected[spec["feature"]].isna().sum())
            if missing:
                raise ScarcityDataError(f"{key}: {missing} missing hours after strict source-time validation.")
            validations[key] = evidence
        audit.update(status="complete", required_sources_complete=True, sources=validations)
    except KeyboardInterrupt:
        audit.update(status="interrupted", required_sources_complete=False)
        raise
    except (OSError, ValueError, AssertionError, FuelMaterializationError, subprocess.SubprocessError) as exc:
        audit.update(status="failed", required_sources_complete=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        stable = _unchanged(source_evidence) and _unchanged(audit_evidence)
        if not stable:
            audit.update(status="failed", required_sources_complete=False,
                         error="An original fuel seed changed concurrently; private snapshot is not approved.")
        audit["original_sources_unchanged"] = stable
        audit["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(audit_path, audit)
    LOGGER.info("[Scarcity fuel] %s: %s", audit["status"], audit_path)
    return {**audit, "config": copied}


__all__ = ["refresh_fuels", "MAX_SUFFIX_DAYS"]
