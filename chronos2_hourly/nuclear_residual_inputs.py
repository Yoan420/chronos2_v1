"""Verified residual-load forecasts for the isolated nuclear experiment.

Legacy sparse vintages are never interpolated, shifted, or rewritten here.
The existing Saturn as-of bank validator also validates its narrowly scoped
DST repair ledger. Its full audit accompanies the experimental input.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from materialize_saturn_kalman_fuel import (
    MAX_TOTAL_WORKERS, RESIDUAL_LOAD_OUTPUT_NAME,
    _load_existing_residual_load_market,
    _materialize_residual_load_market_features,
)

ROOT = Path(__file__).resolve().parents[1]


def _range(start_day: Any, end_day: Any) -> tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    start, end = pd.Timestamp(start_day), pd.Timestamp(end_day)
    for value in (start, end):
        if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
            raise ValueError("Residual bank dates must be naive local calendar dates.")
    if start > end:
        raise ValueError("Residual bank start_day must not be after end_day.")
    expected = pd.date_range(
        start.tz_localize("Europe/Paris"),
        (end + pd.Timedelta(days=1)).tz_localize("Europe/Paris"),
        freq="h", inclusive="left",
    ).tz_convert("UTC")
    return start, end, expected


def audit_residual_bank(path: str | Path, start_day: Any, end_day: Any) -> dict[str, Any]:
    """Read-only SHA, provenance, civil-cutoff and physical-grid validation."""
    start, end, expected = _range(start_day, end_day)
    source = Path(path).resolve()
    audit: dict[str, Any] = {
        "path": str(source), "complete": False, "blockers": [],
        "start_day": start.date().isoformat(), "end_day": end.date().isoformat(),
        "expected_hours": len(expected), "covered_hours": 0,
        "missing_hour_count": len(expected), "missing_hours": [],
        "missing_days": [], "source_provenance_valid": False,
        "provider_revision_timestamp_available": False,
        "production_pit_evidence": False, "source_audit": None,
    }
    if not source.is_file():
        audit["blockers"].append(f"Residual forecast bank absent: {source}")
        return audit
    try:
        frame, native, _ = _load_existing_residual_load_market(
            source, source.with_name(source.name + ".audit.json"),
            requested_start=start, source_mode="saturn", vintage_root=None,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        audit["blockers"].append(f"Residual forecast bank refused: {exc}")
        return audit
    available = pd.DatetimeIndex(pd.to_datetime(frame["value_time_utc"], utc=True))
    missing = expected.difference(available)
    audit.update({
        "source_provenance_valid": True, "source_audit": native,
        "covered_hours": len(expected) - len(missing),
        "missing_hour_count": len(missing),
        "missing_hours": [stamp.isoformat() for stamp in missing],
        "missing_days": sorted(set(missing.tz_convert("Europe/Paris").strftime("%Y-%m-%d"))),
        "complete": len(missing) == 0,
    })
    if len(missing):
        audit["blockers"].append(
            f"Residual forecast bank incomplete: {len(missing)} physical hours missing."
        )
    return audit


def _destination(path: str | Path) -> Path:
    target = Path(path).resolve()
    allowed = (ROOT / "data" / "pit" / "nuclear_forecast").resolve()
    if target.name != RESIDUAL_LOAD_OUTPUT_NAME or allowed not in target.parents:
        raise ValueError(
            "Residual bank writes require data/pit/nuclear_forecast/"
            + RESIDUAL_LOAD_OUTPUT_NAME + " (or a subdirectory)."
        )
    return target


def _copy_verified_seed(seed: Path, target: Path, native: dict[str, Any]) -> None:
    """Copy bytes into an empty isolated destination; never reseal bad inputs."""
    audit_source = seed.with_name(seed.name + ".audit.json")
    expected = str(native["sha256"])
    pending = target.with_name(target.name + f".{uuid4().hex}.tmp")
    pending_audit = pending.with_name(pending.name + ".audit.json")
    try:
        with seed.open("rb") as src, pending.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        with audit_source.open("rb") as src, pending_audit.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        copied_audit = json.loads(pending_audit.read_text(encoding="utf-8"))
        if (
            hashlib.sha256(pending.read_bytes()).hexdigest() != expected
            or copied_audit != native
        ):
            raise ValueError("Residual seed changed during snapshot copy.")
        if target.exists() or target.with_name(target.name + ".audit.json").exists():
            raise ValueError("Residual destination appeared during snapshot copy.")
        pending.replace(target)
        pending_audit.replace(target.with_name(target.name + ".audit.json"))
    finally:
        pending.unlink(missing_ok=True)
        pending_audit.unlink(missing_ok=True)


def ensure_residual_bank(
    *, path: str | Path, seed_path: str | Path | None,
    start_day: Any, end_day: Any, workers: int, allow_sync: bool = True,
) -> dict[str, Any]:
    """Reuse a verified bank and download only its missing suffix, in isolation.

    ``allow_sync=False`` is strictly read-only: no seed copy or network request.
    An invalid existing bank or seed is refused, never silently rebuilt.
    """
    start, end, _ = _range(start_day, end_day)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    target = _destination(path)
    current = audit_residual_bank(target, start, end)
    if current["complete"] or not allow_sync:
        return current
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name + ".lock")
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Residual bank synchronization already locked: {lock}") from exc
    try:
        with handle:
            json.dump({"pid": os.getpid(), "path": str(target)}, handle)
        current = audit_residual_bank(target, start, end)
        if current["complete"]:
            return current
        sidecar = target.with_name(target.name + ".audit.json")
        if target.exists() or sidecar.exists():
            if not current["source_provenance_valid"]:
                raise ValueError("; ".join(current["blockers"]))
        elif seed_path is not None:
            seed = Path(seed_path).resolve()
            if seed == target:
                raise ValueError("Residual seed must differ from its isolated destination.")
            if seed.exists() or seed.with_name(seed.name + ".audit.json").exists():
                seed_audit = audit_residual_bank(seed, start, end)
                if not seed_audit["source_provenance_valid"]:
                    raise ValueError("Residual seed refused: " + "; ".join(seed_audit["blockers"]))
                _copy_verified_seed(seed, target, seed_audit["source_audit"])
        _materialize_residual_load_market_features(
            start_day=start, end_day=end, output_dir=target.parent,
            source_mode="saturn", series_workers=1,
            day_workers=min(workers, MAX_TOTAL_WORKERS),
        )
        result = audit_residual_bank(target, start, end)
        if not result["complete"]:
            raise ValueError("; ".join(result["blockers"]))
        return result
    finally:
        lock.unlink(missing_ok=True)


__all__ = ["audit_residual_bank", "ensure_residual_bank"]
