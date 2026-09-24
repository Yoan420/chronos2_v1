"""Stable, isolated daily nuclear replay epochs and semantic namespaces.

Only the first delivery of an epoch clips residual training at a cold-start
anchor. Later deliveries keep that anchor, so an unchanged historical forecast
has the same inputs and remains reusable as the reporting window advances.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


class NuclearIncrementalError(ValueError):
    """The persisted daily computation contract is invalid."""


def _resolved(path: str | Path) -> Path:
    result = str(Path(path).expanduser().resolve())
    # Windows may retain its extended-length prefix while another thread is
    # creating a parent directory. Both spellings identify the same path.
    if result.startswith("\\\\?\\UNC\\"):
        result = "\\\\" + result[8:]
    elif result.startswith("\\\\?\\"):
        result = result[4:]
    return Path(result)


def _day(value: Any) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is not None or value.time() != datetime.min.time():
            raise NuclearIncrementalError("Incremental delivery must be a civil date.")
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise NuclearIncrementalError("Incremental delivery must respect YYYY-MM-DD.") from exc
    if parsed.isoformat() != str(value):
        raise NuclearIncrementalError("Incremental delivery must respect YYYY-MM-DD.")
    return parsed


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _read_epoch(path: Path, contract_digest: str | None) -> tuple[date, date]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise NuclearIncrementalError("Incremental epoch must be an object.")
        stored_digest = record.get("contract_digest", "")
        valid_digest = (isinstance(stored_digest, str) and len(stored_digest) == 64
                        and all(character in "0123456789abcdef" for character in stored_digest))
        if (set(record) != {"schema_version", "contract_digest", "first_delivery_day", "anchor_day"}
                or record["schema_version"] != 1 or not valid_digest
                or (contract_digest is not None and stored_digest != contract_digest)
                or (contract_digest is None and path.parent.name not in {stored_digest, stored_digest[:32]})):
            raise NuclearIncrementalError("Incremental epoch identity mismatch.")
        first, anchor = _day(record["first_delivery_day"]), _day(record["anchor_day"])
        if anchor != first - timedelta(days=730):
            raise NuclearIncrementalError("Incremental epoch has an invalid cold-start anchor.")
        return first, anchor
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise NuclearIncrementalError(f"Invalid incremental epoch: {path}") from exc


def _ensure_epoch(directory: Path, contract_digest: str, delivery_day: date) -> tuple[date, date]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "epoch.json"
    if _resolved(path).parent != _resolved(directory):
        raise NuclearIncrementalError("Incremental epoch file escapes its namespace.")
    if not path.exists():
        descriptor = {"schema_version": 1, "contract_digest": contract_digest,
                      "first_delivery_day": delivery_day.isoformat(),
                      "anchor_day": (delivery_day - timedelta(days=730)).isoformat()}
        fd, temporary = tempfile.mkstemp(prefix=".epoch-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(descriptor, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # Link publication is atomic and never replaces another run's
                # committed anchor, including two concurrent first deliveries.
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            Path(temporary).unlink(missing_ok=True)
    return _read_epoch(path, contract_digest)


def resolve_incremental_epoch(
    cache_root: str | Path, delivery_day: str | date, contract: Mapping[str, Any],
) -> tuple[Path, date]:
    """Return the immutable semantic namespace and its original history anchor.

    A request earlier than the first delivery gets a separate dated backfill
    namespace. It cannot move the production anchor or alter later caches.
    """
    root, requested = _resolved(cache_root), _day(delivery_day)
    contract_digest = _digest(contract)
    # A bounded directory name leaves room for daily files on Windows. The
    # full SHA remains in the descriptor and a prefix collision is refused.
    directory = _resolved(root / "epochs" / contract_digest[:32])
    if not directory.is_relative_to(root):
        raise NuclearIncrementalError(f"Incremental namespace {directory} escapes its configured cache root {root}.")
    first, anchor = _ensure_epoch(directory, contract_digest, requested)
    if requested < first:
        directory = _resolved(directory / "backfills" / requested.isoformat())
        if not directory.is_relative_to(root):
            raise NuclearIncrementalError("Incremental backfill escapes its configured cache root.")
        _, anchor = _ensure_epoch(directory, contract_digest, requested)
    return directory, anchor


_LOCATION_KEYS = {"file", "pit_file", "pit_files", "cache_dir", "pit_vintage_dir",
                  "project_root", "directory", "output_dir", "source_path", "runtime_as_of",
                  "thread_count", "threads", "workers", "origin_batch_size", "model_batch_size"}


def _semantic(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _semantic(item) for key, item in value.items() if key not in _LOCATION_KEYS}
    if isinstance(value, (tuple, list)):
        return [_semantic(item) for item in value]
    return value


def prepare_incremental_settings(
    config: dict[str, Any], delivery_day: str | date,
) -> tuple[Path | None, date, date]:
    """Resolve and record the raw-history scope before preparing or fitting.

    Snapshot locations and the rolling delivery date do not define a new model
    recipe. Per-day data digests remain responsible for detecting revised input
    values. Missing mode means the existing full replay for old snapshots.
    """
    requested = _day(delivery_day)
    experiment = config.setdefault("nuclear_experiment", {})
    mode = experiment.get("mode", "full")
    if mode not in {"incremental", "full"}:
        raise NuclearIncrementalError("Nuclear computation mode must be incremental or full.")
    if mode == "full":
        start = requested - timedelta(days=730)
        return None, start, start
    cache_root = experiment.get("incremental_cache_dir")
    if not cache_root:
        raise NuclearIncrementalError("Incremental mode requires incremental_cache_dir.")
    package = Path(__file__).resolve().parent
    implementations = (Path(__file__), package / "nuclear_forecast.py", package / "nuclear_preparation.py",
                       package / "chronos_adapter.py", package / "models" / "residual_corrector.py",
                       package.parent / "run_chronos2_hourly.py")
    versions = {}
    for name in ("numpy", "pandas", "catboost", "torch", "chronos-forecasting"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    contract = {"schema_version": 1, "model": _semantic(config.get("model", {})),
                "hourly": _semantic(config.get("hourly", {})),
                "data": _semantic(config.get("data", {})),
                "zones": _semantic(config.get("zones", {})),
                "input_protocol": experiment.get("input_protocol"),
                "implementations": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in implementations}, "dependencies": versions}
    namespace, anchor = resolve_incremental_epoch(cache_root, requested, contract)
    raw_start = max(anchor, requested - timedelta(days=1095))
    experiment.update(history_anchor_day=anchor.isoformat(), raw_history_start_day=raw_start.isoformat(),
                      incremental_namespace=str(namespace))
    return namespace, anchor, raw_start


def retained_source_start(cache_root: str | Path, delivery_day: str | date) -> date:
    """Read existing epoch anchors to retain source support during daily sync.

    Epochs from older recipes can only widen collection (up to 1095 days); they
    cannot select or validate a model cache. Every path is derived locally.
    """
    root, requested = _resolved(cache_root), _day(delivery_day)
    earliest = requested - timedelta(days=730)
    if not root.exists():
        return earliest
    for path in root.glob("*/*/epochs/*/epoch.json"):
        if not _resolved(path).is_relative_to(root):
            raise NuclearIncrementalError("Incremental epoch escapes the source cache root.")
        first, anchor = _read_epoch(path, None)
        if first <= requested:
            earliest = min(earliest, anchor)
    return max(earliest, requested - timedelta(days=1095))


__all__ = ["NuclearIncrementalError", "prepare_incremental_settings", "resolve_incremental_epoch",
           "retained_source_start"]
