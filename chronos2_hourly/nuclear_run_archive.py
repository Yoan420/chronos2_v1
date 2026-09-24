"""Immutable report-only nuclear results; loading never calls a model.

This is not a training checkpoint or a production promotion. It serializes
the already computed result and retains its frozen input provenance so that
new observations and Storm can be attached by the reporting layer alone.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .atomic_directory import AtomicDirectoryStaging


class NuclearRunArchiveError(ValueError):
    """Missing, modified or incompatible frozen report result."""


_FRAMES = (
    "raw_history", "residual_statistics", "source_forecast", "covariates",
    "residual_daily_audit", "kalman_backtest", "kalman_forecast",
)
_FILES = {f"{name}.parquet" for name in _FRAMES} | {"audits.json"}
_TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin", "BE": "Europe/Brussels",
              "NL": "Europe/Amsterdam", "ES": "Europe/Madrid"}
_TRANSIENT = {"raw_future_cache_hit", "residual_result_cache_hit", "rolling_refit_cache",
              "daily_chronos_cache", "daily_residual_cache"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any, *, identity: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, identity=identity) for key, item in value.items()
                if not identity or key not in _TRANSIENT}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_value(item, identity=identity) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (date, datetime, pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _frame_digest(frame: pd.DataFrame) -> str:
    # Arrow roundtrips list-valued audit columns as ndarrays. Normalize just
    # their hash representation, never the serialized training/prediction data.
    hashed = frame.copy(deep=False)
    for column in frame.select_dtypes(include="object").columns:
        hashed = hashed.assign(**{column: frame[column].map(
            lambda value: _json_bytes(value).decode("utf-8"))})
    digest = hashlib.sha256(_json_bytes([(str(c), str(t)) for c, t in frame.dtypes.items()]))
    digest.update(pd.util.hash_pandas_object(hashed, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _location(workdir: str | Path) -> tuple[Path, Path]:
    workspace = Path(workdir).expanduser().resolve()
    destination = (workspace / "report_only" / "frozen_result").resolve()
    if not destination.is_relative_to(workspace) or destination == workspace:
        raise NuclearRunArchiveError("Frozen result directory escapes the run workspace.")
    if (workspace / "artifact_checksums.json").exists():
        raise NuclearRunArchiveError("A frozen report bundle cannot modify a sealed live archive.")
    return workspace, destination


def _source_contract(workspace: Path) -> dict[str, Any]:
    manifest_path, resolved_path = workspace / "input_snapshot.json", workspace / "resolved_config.yaml"
    if not manifest_path.is_file() or not resolved_path.is_file():
        raise NuclearRunArchiveError("Frozen result requires input_snapshot.json and resolved_config.yaml.")
    manifest_sha, resolved_sha = _sha(manifest_path), _sha(resolved_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["resolved_config_sha256"] != resolved_sha:
            raise NuclearRunArchiveError("Frozen resolved_config.yaml checksum mismatch.")
        files = manifest["files"]
        if not isinstance(files, list) or not files:
            raise NuclearRunArchiveError("Frozen input snapshot contains no sources.")
        pinned = {}
        for item in files:
            path = Path(item["snapshot"]).resolve()
            if not path.is_relative_to(workspace / "snapshot") or not path.is_file():
                raise NuclearRunArchiveError(f"Missing or unsafe frozen snapshot source: {path}")
            if str(path) in pinned or _sha(path) != item["sha256"]:
                raise NuclearRunArchiveError(f"Frozen snapshot source checksum mismatch: {path}")
            pinned[str(path)] = item["sha256"]
        if _sha(manifest_path) != manifest_sha or _sha(resolved_path) != resolved_sha:
            raise NuclearRunArchiveError("Frozen input contract changed during verification.")
        return {"input_snapshot_sha256": manifest_sha, "resolved_config_sha256": resolved_sha,
                "snapshot_identity": manifest["identity"], "snapshot_files": pinned}
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise NuclearRunArchiveError("Malformed frozen input snapshot contract.") from exc


def _indexed(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.has_duplicates:
        raise NuclearRunArchiveError(f"{name}: nonempty frame with unique columns required.")
    result = frame.copy(deep=False)
    key = next((column for column in ("delivery_start_utc", "timestamp") if column in frame), None)
    values = frame[key] if key else frame.index
    try:
        if any(pd.Timestamp(value).tzinfo is None for value in values):
            raise ValueError("timezone missing")
        index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    except (ValueError, TypeError) as exc:
        raise NuclearRunArchiveError(f"{name}: explicit timezone-aware timestamps required.") from exc
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise NuclearRunArchiveError(f"{name}: unique increasing physical timestamps required.")
    result.index = index
    return result


def _quantiles(frame: pd.DataFrame, prefix: str, name: str) -> None:
    columns = [f"{prefix}{q}" for q in ("q10", "q50", "q90")]
    if not set(columns).issubset(frame):
        raise NuclearRunArchiveError(f"{name}: missing quantiles {columns}.")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
        raise NuclearRunArchiveError(f"{name}: nonfinite or crossed quantiles.")


def _validate(frames: Mapping[str, pd.DataFrame], audits: Mapping[str, Any], sources: Mapping[str, Any]) -> None:
    audit = audits.get("result", {})
    identity = sources["snapshot_identity"]
    zone, delivery = audit.get("zone"), audit.get("delivery_day")
    if zone not in _TIMEZONES or zone != identity.get("zone") or delivery != identity.get("delivery_day"):
        raise NuclearRunArchiveError("Frozen result zone/delivery differs from the input snapshot.")
    if audit.get("engine") != "nuclear_forecast_v1" or not isinstance(audits.get("kalman_replay"), dict):
        raise NuclearRunArchiveError("Nuclear engine and Kalman replay audits are required.")
    if not audit.get("source_hashes", {}).get("engine_file_sha256"):
        raise NuclearRunArchiveError("Frozen result lacks its original engine source SHA.")
    day, timezone = pd.Timestamp(delivery).date(), _TIMEZONES[zone]
    grid = lambda start, end: pd.date_range(pd.Timestamp(start, tz=timezone),
        pd.Timestamp(end, tz=timezone), freq="h", inclusive="left").tz_convert("UTC")
    history = grid(day - timedelta(days=730), day)
    evaluation = grid(day - timedelta(days=365), day)
    future = grid(day, day + timedelta(days=1))
    expected = {"raw_history": history, "residual_statistics": history,
                "source_forecast": future, "kalman_backtest": evaluation, "kalman_forecast": future,
                "covariates": history.append(future)}
    indexed = {name: _indexed(frames[name], name) for name in expected}
    for name, index in expected.items():
        if not indexed[name].index.equals(index):
            raise NuclearRunArchiveError(f"{name}: wrong 730/365-day or physical forecast coverage.")
    _quantiles(indexed["raw_history"], "", "raw_history")
    for name in ("residual_statistics", "source_forecast"):
        for model in ("chronos2__", "residual_corrected__"):
            _quantiles(indexed[name], model, name)
    for name in ("kalman_backtest", "kalman_forecast"):
        _quantiles(indexed[name], "residual_kalman__", name)
    for name in ("raw_history", "residual_statistics", "kalman_backtest"):
        if "actual" not in indexed[name] or not np.isfinite(pd.to_numeric(indexed[name].actual, errors="coerce")).all():
            raise NuclearRunArchiveError(f"{name}: finite historical observations required.")
    for name in ("source_forecast", "kalman_forecast"):
        if "actual" in indexed[name] and indexed[name].actual.notna().any():
            raise NuclearRunArchiveError(f"{name}: observed delivery labels cannot enter frozen predictions.")
    for q in ("q10", "q50", "q90"):
        column = "residual_corrected__" + q
        if column not in indexed["kalman_forecast"] or not np.array_equal(
            indexed["kalman_forecast"][column].to_numpy(), indexed["source_forecast"][column].to_numpy()):
            raise NuclearRunArchiveError("Frozen Kalman forecast does not retain the autonomous upstream.")
    daily = frames["residual_daily_audit"]
    expected_days = pd.date_range(day - timedelta(days=730), day).strftime("%Y-%m-%d").tolist()
    if "delivery_day" not in daily or daily.delivery_day.astype(str).tolist() != expected_days:
        raise NuclearRunArchiveError("Frozen residual daily audit must cover all 731 delivery days.")


def _collect(result: Any) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    try:
        frames = {name: getattr(result, name) for name in _FRAMES[:5]}
        frames.update(kalman_backtest=result.kalman_view.backtest, kalman_forecast=result.kalman_view.forecast)
        audits = _json_value({"result": result.audit, "kalman_replay": result.kalman_view.replay.audit})
        return frames, audits
    except AttributeError as exc:
        raise NuclearRunArchiveError("A completed nuclear result with Kalman replay is required.") from exc


def _identity(frames: Mapping[str, pd.DataFrame], audits: Mapping[str, Any], sources: Mapping[str, Any]) -> dict:
    return {"sources": sources, "frames": {name: _frame_digest(frame) for name, frame in frames.items()},
            "result_audit_sha256": hashlib.sha256(_json_bytes(_json_value(audits, identity=True))).hexdigest()}


def load_nuclear_result_bundle(*, workdir: str | Path) -> SimpleNamespace:
    """Load validated forecast/history for report rendering only, no fitting."""
    workspace, directory = _location(workdir)
    if not directory.is_dir():
        raise NuclearRunArchiveError("Frozen nuclear result absent: run the nuclear pipeline once before Report.")
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "nuclear_report_frozen_result":
            raise NuclearRunArchiveError("Unknown frozen nuclear report bundle schema.")
        if set(manifest["files"]) != _FILES or {path.name for path in directory.iterdir()} != _FILES | {"manifest.json"}:
            raise NuclearRunArchiveError("Frozen nuclear result file inventory mismatch.")
        sources = _source_contract(workspace)
        if manifest["identity"]["sources"] != sources:
            raise NuclearRunArchiveError("Frozen result source snapshot/configuration changed.")
        for name, digest in manifest["files"].items():
            path = directory / name
            if not path.is_file() or path.resolve().parent != directory or _sha(path) != digest:
                raise NuclearRunArchiveError(f"Frozen result checksum mismatch: {name}")
        frames = {name: pd.read_parquet(directory / f"{name}.parquet") for name in _FRAMES}
        audits = json.loads((directory / "audits.json").read_text(encoding="utf-8"))
        _validate(frames, audits, sources)
        if _identity(frames, audits, sources) != manifest["identity"]:
            raise NuclearRunArchiveError("Frozen result semantic identity mismatch.")
        view = SimpleNamespace(backtest=frames.pop("kalman_backtest"), forecast=frames.pop("kalman_forecast"),
                               replay=SimpleNamespace(audit=audits["kalman_replay"]))
        return SimpleNamespace(**frames, kalman_view=view, audit=audits["result"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise NuclearRunArchiveError(f"Incomplete frozen nuclear result: {directory}") from exc


def save_nuclear_result_bundle(result: Any, *, workdir: str | Path) -> Path:
    """Seal a completed result once, or reuse an identical immutable bundle."""
    workspace, directory = _location(workdir)
    sources = _source_contract(workspace)
    frames, audits = _collect(result)
    _validate(frames, audits, sources)
    identity = _identity(frames, audits, sources)
    if directory.exists():
        retained = load_nuclear_result_bundle(workdir=workspace)
        old_frames, old_audits = _collect(retained)
        if _identity(old_frames, old_audits, sources) != identity:
            raise NuclearRunArchiveError("Refusing to overwrite a divergent frozen nuclear result.")
        return directory
    directory.parent.mkdir(parents=True, exist_ok=True)
    with AtomicDirectoryStaging(directory.parent, prefix=".frozen_result-") as publication:
        staging = publication.path
        for name, frame in frames.items():
            frame.to_parquet(staging / f"{name}.parquet", index=True)
        (staging / "audits.json").write_bytes(_json_bytes(audits))
        manifest = {"schema_version": 1, "artifact_type": "nuclear_report_frozen_result",
                    "report_only": True, "identity": identity,
                    "files": {name: _sha(staging / name) for name in sorted(_FILES)}}
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        if _source_contract(workspace) != sources:
            raise NuclearRunArchiveError("Frozen inputs changed while archiving the result.")
        publication.publish(directory)
    load_nuclear_result_bundle(workdir=workspace)
    return directory


__all__ = ["NuclearRunArchiveError", "save_nuclear_result_bundle", "load_nuclear_result_bundle"]
