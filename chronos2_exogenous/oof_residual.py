"""Blocked/prequential LoRA predictions for the final residual corrector.

The final LoRA checkpoint cannot be replayed on the days that helped fit it.
This module therefore trains one adapter per chronological block, always on a
365-day window strictly preceding that block.  The resulting 365 OOF delivery
days are used only to fit the small, safe residual corrector.  The frozen
rolling-365 promotion holdout remains completely outside this workflow.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import stat
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd

from .evaluation import build_inference_input, _predict_quantiles_batch
from .lora_finetune import (
    ExogenousFineTuneConfig,
    ExogenousFineTuneError,
    PipelineLoader,
    _default_pipeline_loader,
    _pin_adapter_base,
    _schema_payload,
    build_fit_inputs,
    load_config,
    read_panel,
    resolve_local_model_source,
    sha256_directory,
    validate_panel,
    verify_bundle,
)
from .production import fit_oof_residual_corrector


TRAINING_DAYS = 365
OOF_DAYS = 365
HOLDOUT_DAYS = 365
SIDECAR_SCHEMA_VERSION = 2
SHARED_CACHE_SCHEMA_VERSION = 1
SHARED_CACHE_DIRECTORY_NAME = "shared_oof_fold_checkpoints"
SHARED_CACHE_CONTRACT_NAME = "cache_contract.json"
SHARED_CACHE_FOLD_SEAL_NAME = "fold_seal.json"
SHARED_CACHE_CLAIM_SUFFIX = ".claim.json"
DEFAULT_SHARED_CACHE_WAIT_SECONDS = 12 * 60 * 60
DEFAULT_SHARED_CACHE_POLL_SECONDS = 5.0


class ExogenousOofResidualError(ExogenousFineTuneError):
    """Raised when the prequential calibration contract cannot be proved."""


@dataclass(frozen=True)
class OofFold:
    index: int
    fit_origins: tuple[pd.Timestamp, ...]
    train_origins: tuple[pd.Timestamp, ...]
    validation_origins: tuple[pd.Timestamp, ...]
    prediction_origins: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class OofPlan:
    folds: tuple[OofFold, ...]
    calibration_origins: tuple[pd.Timestamp, ...]
    holdout_origins: tuple[pd.Timestamp, ...]
    required_origins: int


@dataclass(frozen=True)
class ResidualCalibrationArtifacts:
    directory: Path
    predictions_path: Path
    audit_path: Path
    corrector_path: Path
    manifest_path: Path
    folds: int
    shared_checkpoint_cache_directory: Path | None = None
    shared_checkpoint_cache_contract_sha256: str | None = None


@dataclass(frozen=True)
class _SharedFoldCheckpointCache:
    """One immutable identity namespace in the shared fold cache."""

    root: Path
    contract_directory: Path
    contract_sha256: str
    contract: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or _is_link_or_reparse(path):
        raise ExogenousOofResidualError(f"{label} absent, lien ou non regulier: {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExogenousOofResidualError(f"{label} illisible: {path}.") from exc
    if not isinstance(payload, dict):
        raise ExogenousOofResidualError(f"{label} doit etre un objet JSON.")
    return payload


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink() or os.path.islink(path):
        return True
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _assert_regular_checkpoint_tree(path: Path) -> None:
    """Reject links/reparse points before hashing or copying a checkpoint."""

    if not path.is_dir() or _is_link_or_reparse(path):
        raise ExogenousOofResidualError(
            f"Checkpoint cache absent, lien ou reparse interdit: {path}."
        )
    files = 0
    for directory, directory_names, file_names in os.walk(path, followlinks=False):
        parent = Path(directory)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = parent / name
            if _is_link_or_reparse(child):
                raise ExogenousOofResidualError(
                    f"Lien/reparse interdit dans le checkpoint cache: {child}."
                )
        for name in file_names:
            child = parent / name
            if not child.is_file() or _is_link_or_reparse(child):
                raise ExogenousOofResidualError(
                    f"Fichier non regulier dans le checkpoint cache: {child}."
                )
            files += 1
    if files == 0:
        raise ExogenousOofResidualError(f"Checkpoint cache vide: {path}.")


def _fold_ranges(fold: OofFold) -> dict[str, dict[str, Any]]:
    return {
        "fit_origins": _range(fold.fit_origins),
        "train_origins": _range(fold.train_origins),
        "validation_origins": _range(fold.validation_origins),
        "prediction_origins": _range(fold.prediction_origins),
    }


def _fold_manifest_core(
    fold: OofFold,
    *,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "chronos2_exogenous_lora_blocked_oof_fold",
        "fold_index": fold.index,
        "recipe": dict(recipe),
        "recipe_sha256": recipe_sha256,
        **_fold_ranges(fold),
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "holdout_used_for_fit": False,
        "checkpoint_sha256": checkpoint_sha256,
        "predictions_sha256": None,
    }


def _zone_source_cache_identity(
    run_directory: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    """Authenticate a PrepareZones clone and infer its common cache root."""

    provenance = manifest.get("zone_artifact_provenance")
    if provenance is None:
        return None
    if not isinstance(provenance, Mapping):
        raise ExogenousOofResidualError(
            "Provenance PrepareZones invalide; cache OOF partage refuse."
        )
    seal = _read_json_object(
        run_directory / "zone_artifact_manifest.json",
        label="sceau PrepareZones",
    )
    expected_pairs = {
        "schema_version": 1,
        "kind": "chronos2_exogenous_zone_artifact_copy",
        "zone": manifest.get("zone"),
        "source_artifact_path": provenance.get("source_artifact_path"),
        "source_experiment_manifest_sha256": provenance.get(
            "source_experiment_manifest_sha256"
        ),
        "source_tree_sha256": provenance.get("source_tree_sha256"),
        "copy_mode": "physical_complete",
        "shared_hardlinks_with_source": False,
    }
    mismatches = [
        key for key, value in expected_pairs.items() if seal.get(key) != value
    ]
    if provenance.get("kind") != "chronos2_exogenous_zone_artifact" or mismatches:
        raise ExogenousOofResidualError(
            "Sceau PrepareZones divergent; cache OOF partage refuse: "
            + ", ".join(mismatches or ["zone_artifact_provenance.kind"])
            + "."
        )
    source_value = provenance.get("source_artifact_path")
    if not isinstance(source_value, str) or not source_value.strip():
        raise ExogenousOofResidualError(
            "source_artifact_path absent de la provenance PrepareZones."
        )
    source = Path(source_value).expanduser().resolve()
    source_manifest_path = source / "experiment_manifest.json"
    expected_source_sha = provenance.get("source_experiment_manifest_sha256")
    if (
        not isinstance(expected_source_sha, str)
        or len(expected_source_sha) != 64
        or not source_manifest_path.is_file()
        or _sha256_file(source_manifest_path) != expected_source_sha
    ):
        raise ExogenousOofResidualError(
            "Artefact multi-zone source absent ou modifie; cache OOF partage refuse."
        )
    source_manifest = verify_bundle(source)
    if source_manifest.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
        raise ExogenousOofResidualError(
            "Checkpoint ancre divergent entre la copie zone et sa source multi-zone."
        )
    identity = {
        "source_artifact_path": str(source),
        "source_experiment_manifest_sha256": expected_source_sha,
        "source_tree_sha256": provenance.get("source_tree_sha256"),
    }
    return source.parent / SHARED_CACHE_DIRECTORY_NAME, identity


def _prepare_shared_fold_cache(
    *,
    requested_root: str | Path | None,
    config: ExogenousFineTuneConfig,
    run_directory: Path,
    manifest: Mapping[str, Any],
    plan: OofPlan,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    panel_audit: Mapping[str, Any],
    block_days: int,
) -> _SharedFoldCheckpointCache | None:
    source_identity: dict[str, Any] | None = None
    if requested_root is None:
        inferred = _zone_source_cache_identity(run_directory, manifest)
        if inferred is None:
            return None
        root, source_identity = inferred
    else:
        root = Path(requested_root).expanduser().resolve()
        inferred = _zone_source_cache_identity(run_directory, manifest)
        if inferred is not None:
            _inferred_root, source_identity = inferred
    root = root.resolve()
    try:
        root.relative_to(config.project_root.resolve())
    except ValueError as exc:
        raise ExogenousOofResidualError(
            "Le cache de checkpoints OOF doit rester sous project_root."
        ) from exc
    if root == config.project_root.resolve():
        raise ExogenousOofResidualError(
            "La racine du projet ne peut pas servir directement de cache OOF."
        )
    if root.exists() and (not root.is_dir() or _is_link_or_reparse(root)):
        raise ExogenousOofResidualError(
            f"Racine de cache OOF non sure: {root}."
        )

    upstream = panel_audit.get("upstream_panel_audit")
    if not isinstance(upstream, Mapping):
        raise ExogenousOofResidualError(
            "Audit amont du panel de calibration absent."
        )
    split_contract = {
        "final_checkpoint_splits": manifest.get("splits"),
        "required_origins": plan.required_origins,
        "calibration_origins": _range(plan.calibration_origins),
        "holdout_origins": _range(plan.holdout_origins),
        "folds": [
            {"fold_index": fold.index, **_fold_ranges(fold)} for fold in plan.folds
        ],
    }
    contract: dict[str, Any] = {
        "schema_version": SHARED_CACHE_SCHEMA_VERSION,
        "kind": "chronos2_exogenous_shared_oof_fold_checkpoint_cache",
        "scope": "checkpoints_only_no_predictions_no_corrector",
        "deployment_checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "training_panel_sha256": manifest.get("panel_sha256"),
        "calibration_panel_sha256": upstream.get("panel_sha256"),
        "calibration_panel_audit_sha256": upstream.get("panel_audit_sha256"),
        "fold_candidate_recipe": dict(recipe),
        "fold_candidate_recipe_sha256": recipe_sha256,
        "block_days": int(block_days),
        "split_contract": split_contract,
        "source_training_identity": source_identity,
    }
    contract_sha = _json_sha256(contract)
    # Keep the Windows path comfortably below MAX_PATH; the complete digest is
    # still stored and revalidated inside the deterministic contract.
    contract_directory = root / contract_sha[:24]
    contract_path = contract_directory / SHARED_CACHE_CONTRACT_NAME
    if not contract_directory.exists():
        root.mkdir(parents=True, exist_ok=True)
        staging = root / f".contract.tmp-{uuid.uuid4().hex[:8]}"
        try:
            staging.mkdir(parents=False, exist_ok=False)
            (staging / "folds").mkdir()
            (staging / "claims").mkdir()
            _write_json_atomic(staging / SHARED_CACHE_CONTRACT_NAME, contract)
            try:
                os.replace(staging, contract_directory)
            except OSError:
                if not contract_directory.is_dir():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    if not contract_directory.is_dir() or _is_link_or_reparse(contract_directory):
        raise ExogenousOofResidualError(
            f"Namespace de cache OOF non sur: {contract_directory}."
        )
    cached_contract = _read_json_object(contract_path, label="contrat cache OOF")
    if cached_contract != contract or _json_sha256(cached_contract) != contract_sha:
        raise ExogenousOofResidualError(
            "Le contrat du cache OOF partage diverge de son identite SHA-256."
        )
    folds_path = contract_directory / "folds"
    if not folds_path.is_dir() or _is_link_or_reparse(folds_path):
        raise ExogenousOofResidualError("Repertoire folds du cache OOF invalide.")
    claims_path = contract_directory / "claims"
    if not claims_path.is_dir() or _is_link_or_reparse(claims_path):
        raise ExogenousOofResidualError("Repertoire claims du cache OOF invalide.")
    return _SharedFoldCheckpointCache(
        root=root,
        contract_directory=contract_directory,
        contract_sha256=contract_sha,
        contract=contract,
    )


def _cache_fold_identity(
    cache: _SharedFoldCheckpointCache, fold: OofFold
) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema_version": SHARED_CACHE_SCHEMA_VERSION,
        "contract_sha256": cache.contract_sha256,
        "fold_index": fold.index,
        **_fold_ranges(fold),
    }
    return _json_sha256(payload), payload


def _cached_fold_directory(
    cache: _SharedFoldCheckpointCache,
    fold: OofFold,
    timezone_name: str,
) -> Path:
    return cache.contract_directory / "folds" / _fold_name(fold, timezone_name)


def _process_observation(process_id: int) -> tuple[bool | None, str | None]:
    """Return (alive, creation identity) without signalling the process."""

    if process_id <= 0:
        return False, None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetProcessTimes.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
            ]
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            # PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE; the latter is
            # required for the non-invasive zero-timeout liveness check.
            handle = kernel32.OpenProcess(0x00101000, False, int(process_id))
            if not handle:
                error = ctypes.get_last_error()
                return (False, None) if error == 87 else (None, None)
            try:
                creation = wintypes.FILETIME()
                exit_time = wintypes.FILETIME()
                kernel_time = wintypes.FILETIME()
                user_time = wintypes.FILETIME()
                if not kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel_time),
                    ctypes.byref(user_time),
                ):
                    return None, None
                identity = (
                    int(creation.dwHighDateTime) << 32
                ) | int(creation.dwLowDateTime)
                wait_result = int(kernel32.WaitForSingleObject(handle, 0))
                if wait_result == 258:
                    return True, f"windows-filetime:{identity}"
                if wait_result == 0:
                    return False, f"windows-filetime:{identity}"
                return None, None
            finally:
                kernel32.CloseHandle(handle)
        except Exception:  # pragma: no cover - platform API failure is fail-closed.
            return None, None

    proc_stat = Path("/proc") / str(process_id) / "stat"
    if Path("/proc").is_dir():
        if not proc_stat.exists():
            return False, None
        try:
            suffix = proc_stat.read_text(encoding="utf-8").rsplit(")", 1)[1]
            fields = suffix.split()
            return True, f"proc-start-ticks:{fields[19]}"
        except (OSError, IndexError):
            return None, None
    if process_id == os.getpid():
        return True, f"process-local:{process_id}"
    return None, None


def _claim_path(
    cache: _SharedFoldCheckpointCache,
    fold: OofFold,
    timezone_name: str,
) -> Path:
    return (
        cache.contract_directory
        / "claims"
        / f"{_fold_name(fold, timezone_name)}{SHARED_CACHE_CLAIM_SUFFIX}"
    )


def _read_fold_claim(path: Path) -> dict[str, Any]:
    try:
        payload = _read_json_object(path, label="claim fold OOF")
    except ExogenousOofResidualError:
        try:
            age = max(0.0, time.time() - path.stat().st_mtime)
        except OSError:
            raise
        if age < 2.0:
            # The winner may be between O_EXCL and fsync. The caller will poll.
            return {}
        raise
    return payload


def _remove_dead_fold_claim(
    *,
    path: Path,
    cache: _SharedFoldCheckpointCache,
    fold: OofFold,
) -> bool:
    claim = _read_fold_claim(path)
    if not claim:
        return False
    identity_sha, _identity = _cache_fold_identity(cache, fold)
    required = {
        "schema_version": SHARED_CACHE_SCHEMA_VERSION,
        "kind": "chronos2_exogenous_shared_oof_fold_claim",
        "contract_sha256": cache.contract_sha256,
        "fold_identity_sha256": identity_sha,
    }
    mismatches = [key for key, value in required.items() if claim.get(key) != value]
    token = claim.get("token")
    host = claim.get("host")
    process_id = claim.get("process_id")
    process_identity = claim.get("process_start_identity")
    created_at = claim.get("created_at_unix")
    lease_expires_at = claim.get("lease_expires_at_unix")
    if (
        mismatches
        or not isinstance(token, str)
        or not token
        or not isinstance(host, str)
        or type(process_id) is not int
        or not isinstance(process_identity, str)
        or not process_identity
        or isinstance(created_at, bool)
        or not isinstance(created_at, (int, float))
        or not np.isfinite(float(created_at))
        or isinstance(lease_expires_at, bool)
        or not isinstance(lease_expires_at, (int, float))
        or not np.isfinite(float(lease_expires_at))
        or float(lease_expires_at) <= float(created_at)
    ):
        raise ExogenousOofResidualError(
            "Claim fold OOF malforme ou divergent; suppression automatique refusee."
        )
    if host.casefold() != socket.gethostname().casefold():
        return False
    alive, observed_identity = _process_observation(process_id)
    if alive is None:
        return False
    if alive and observed_identity == process_identity:
        return False
    # The PID is absent or has been reused. Atomically quarantine the exact
    # dead owner's claim before deleting it; a live claimant is never stolen.
    quarantine = path.with_name(f".{path.name}.dead-{uuid.uuid4().hex[:8]}")
    try:
        os.replace(path, quarantine)
    except FileNotFoundError:
        return True
    quarantined = _read_json_object(quarantine, label="claim fold OOF mort")
    if quarantined.get("token") != token:
        raise ExogenousOofResidualError(
            "Le claim fold OOF a change pendant sa mise en quarantaine."
        )
    quarantine.unlink()
    return True


@contextmanager
def _shared_fold_fit_claim(
    *,
    cache: _SharedFoldCheckpointCache,
    fold: OofFold,
    timezone_name: str,
    wait_seconds: float,
    poll_seconds: float,
):
    """Elect exactly one fold trainer; waiters reuse its sealed publication."""

    if not np.isfinite(wait_seconds) or wait_seconds < 0:
        raise ExogenousOofResidualError("shared_cache_wait_seconds invalide.")
    if not np.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ExogenousOofResidualError("shared_cache_poll_seconds invalide.")
    alive, process_identity = _process_observation(os.getpid())
    if alive is not True or not process_identity:
        raise ExogenousOofResidualError(
            "Identite du processus indisponible; claim cache OOF refuse."
        )
    path = _claim_path(cache, fold, timezone_name)
    token = uuid.uuid4().hex
    identity_sha, _identity = _cache_fold_identity(cache, fold)
    now = time.time()
    claim = {
        "schema_version": SHARED_CACHE_SCHEMA_VERSION,
        "kind": "chronos2_exogenous_shared_oof_fold_claim",
        "contract_sha256": cache.contract_sha256,
        "fold_identity_sha256": identity_sha,
        "token": token,
        "host": socket.gethostname(),
        "process_id": os.getpid(),
        "process_start_identity": process_identity,
        "created_at_unix": now,
        "lease_expires_at_unix": now + max(float(wait_seconds) * 2.0, 86400.0),
    }
    encoded = (
        json.dumps(claim, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    deadline = time.monotonic() + float(wait_seconds)
    owns_claim = False
    while True:
        if _validate_cached_fold(
            cache=cache, fold=fold, timezone_name=timezone_name
        ) is not None:
            yield False
            return
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                0o600,
            )
        except FileExistsError:
            if _remove_dead_fold_claim(
                path=path, cache=cache, fold=fold
            ):
                continue
            if time.monotonic() >= deadline:
                raise ExogenousOofResidualError(
                    "Attente bornee du fold partage expiree; le claim vivant ou "
                    "non verifiable est conserve (fail-closed)."
                )
            time.sleep(min(float(poll_seconds), max(0.0, deadline - time.monotonic())))
            continue
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            owns_claim = True
            break
        except Exception:
            try:
                path.unlink(missing_ok=True)
            finally:
                raise

    try:
        yield True
    finally:
        if owns_claim:
            if not path.exists():
                raise ExogenousOofResidualError(
                    "Le claim du fold a disparu pendant l'entrainement."
                )
            current = _read_json_object(path, label="claim fold OOF possede")
            if current.get("token") != token:
                raise ExogenousOofResidualError(
                    "Le claim du fold a ete remplace pendant l'entrainement."
                )
            path.unlink()


def _validate_cached_fold(
    *,
    cache: _SharedFoldCheckpointCache,
    fold: OofFold,
    timezone_name: str,
) -> tuple[Path, dict[str, Any]] | None:
    directory = _cached_fold_directory(cache, fold, timezone_name)
    if not directory.exists():
        return None
    if not directory.is_dir() or _is_link_or_reparse(directory):
        raise ExogenousOofResidualError(
            f"Fold cache absent, lien ou non regulier: {directory}."
        )
    children = {child.name for child in directory.iterdir()}
    if children != {"checkpoint", SHARED_CACHE_FOLD_SEAL_NAME}:
        raise ExogenousOofResidualError(
            f"Fold cache contient des artefacts non autorises: {directory}."
        )
    checkpoint = directory / "checkpoint"
    _assert_regular_checkpoint_tree(checkpoint)
    checkpoint_sha = sha256_directory(checkpoint)
    identity_sha, identity = _cache_fold_identity(cache, fold)
    expected = {
        "schema_version": SHARED_CACHE_SCHEMA_VERSION,
        "kind": "chronos2_exogenous_shared_oof_fold_checkpoint",
        "status": "sealed",
        "contract_sha256": cache.contract_sha256,
        "fold_identity_sha256": identity_sha,
        "fold_identity": identity,
        "checkpoint_relative_path": "checkpoint",
        "checkpoint_sha256": checkpoint_sha,
        "contains_predictions": False,
        "contains_corrector": False,
    }
    seal_path = directory / SHARED_CACHE_FOLD_SEAL_NAME
    seal = _read_json_object(seal_path, label="sceau fold cache OOF")
    if seal != expected:
        raise ExogenousOofResidualError(
            f"Sceau du fold cache OOF divergent: {directory}."
        )
    return checkpoint, seal


def _assert_physical_checkpoint_copy(source: Path, destination: Path) -> None:
    source_files = {
        path.relative_to(source).as_posix(): path
        for path in source.rglob("*")
        if path.is_file()
    }
    destination_files = {
        path.relative_to(destination).as_posix(): path
        for path in destination.rglob("*")
        if path.is_file()
    }
    if set(source_files) != set(destination_files):
        raise ExogenousOofResidualError("Copie du checkpoint cache incomplete.")
    for relative, source_path in source_files.items():
        destination_path = destination_files[relative]
        try:
            if os.path.samefile(source_path, destination_path):
                raise ExogenousOofResidualError(
                    f"Hardlink interdit pour le checkpoint cache: {destination_path}."
                )
        except OSError as exc:
            raise ExogenousOofResidualError(
                f"Impossible de verifier la copie physique: {destination_path}."
            ) from exc


def _publish_fold_to_shared_cache(
    *,
    cache: _SharedFoldCheckpointCache,
    fold: OofFold,
    timezone_name: str,
    checkpoint: Path,
) -> dict[str, Any]:
    _assert_regular_checkpoint_tree(checkpoint)
    source_sha = sha256_directory(checkpoint)
    existing = _validate_cached_fold(
        cache=cache, fold=fold, timezone_name=timezone_name
    )
    if existing is not None:
        if existing[1]["checkpoint_sha256"] != source_sha:
            raise ExogenousOofResidualError(
                "Checkpoint fold divergent sous un contrat cache identique; "
                "reutilisation refusee."
            )
        return existing[1]

    destination = _cached_fold_directory(cache, fold, timezone_name)
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    identity_sha, identity = _cache_fold_identity(cache, fold)
    seal = {
        "schema_version": SHARED_CACHE_SCHEMA_VERSION,
        "kind": "chronos2_exogenous_shared_oof_fold_checkpoint",
        "status": "sealed",
        "contract_sha256": cache.contract_sha256,
        "fold_identity_sha256": identity_sha,
        "fold_identity": identity,
        "checkpoint_relative_path": "checkpoint",
        "checkpoint_sha256": source_sha,
        "contains_predictions": False,
        "contains_corrector": False,
    }
    try:
        staging.mkdir(parents=False, exist_ok=False)
        shutil.copytree(checkpoint, staging / "checkpoint", copy_function=shutil.copy2)
        _assert_regular_checkpoint_tree(staging / "checkpoint")
        if sha256_directory(staging / "checkpoint") != source_sha:
            raise ExogenousOofResidualError(
                "Checkpoint modifie pendant sa publication dans le cache OOF."
            )
        _assert_physical_checkpoint_copy(checkpoint, staging / "checkpoint")
        _write_json_atomic(staging / SHARED_CACHE_FOLD_SEAL_NAME, seal)
        try:
            os.replace(staging, destination)
        except OSError:
            if not destination.is_dir():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    published = _validate_cached_fold(
        cache=cache, fold=fold, timezone_name=timezone_name
    )
    if published is None or published[1]["checkpoint_sha256"] != source_sha:
        raise ExogenousOofResidualError(
            "Publication concurrente divergente dans le cache OOF."
        )
    return published[1]


def _materialize_fold_from_shared_cache(
    *,
    cache: _SharedFoldCheckpointCache,
    fold: OofFold,
    fold_directory: Path,
    timezone_name: str,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
) -> dict[str, Any] | None:
    cached = _validate_cached_fold(
        cache=cache, fold=fold, timezone_name=timezone_name
    )
    if cached is None:
        return None
    source_checkpoint, seal = cached
    staging = fold_directory.parent / f".{fold_directory.name}.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=False, exist_ok=False)
        shutil.copytree(
            source_checkpoint,
            staging / "checkpoint",
            copy_function=shutil.copy2,
        )
        _assert_regular_checkpoint_tree(staging / "checkpoint")
        if sha256_directory(staging / "checkpoint") != seal["checkpoint_sha256"]:
            raise ExogenousOofResidualError(
                "Checkpoint cache modifie pendant sa copie vers la zone."
            )
        _assert_physical_checkpoint_copy(source_checkpoint, staging / "checkpoint")
        payload = {
            **_fold_manifest_core(
                fold,
                recipe=recipe,
                recipe_sha256=recipe_sha256,
                checkpoint_sha256=str(seal["checkpoint_sha256"]),
            ),
            "checkpoint_materialization": {
                "mode": "physical_copy_from_sealed_shared_cache",
                "cache_contract_sha256": cache.contract_sha256,
                "fold_identity_sha256": seal["fold_identity_sha256"],
                "fold_seal_sha256": _sha256_file(
                    _cached_fold_directory(cache, fold, timezone_name)
                    / SHARED_CACHE_FOLD_SEAL_NAME
                ),
                "predictions_shared": False,
                "corrector_shared": False,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(staging / "fold_manifest.json", payload)
        os.replace(staging, fold_directory)
        return payload
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _range(origins: Sequence[pd.Timestamp]) -> dict[str, Any]:
    return {
        "count": len(origins),
        "first_utc": origins[0].isoformat() if origins else None,
        "last_utc": origins[-1].isoformat() if origins else None,
    }


def build_oof_plan(
    origins: Sequence[str | pd.Timestamp],
    *,
    timezone_name: str,
    cutoff_local_time: str,
    validation_days: int,
    block_days: int = 30,
    training_days: int = TRAINING_DAYS,
    oof_days: int = OOF_DAYS,
    holdout_days: int = HOLDOUT_DAYS,
) -> OofPlan:
    """Build the deterministic rolling plan without reading model labels."""

    if min(training_days, oof_days, holdout_days, block_days) <= 0:
        raise ExogenousOofResidualError("Les tailles de fenetre doivent etre positives.")
    if not 0 < int(validation_days) < int(training_days):
        raise ExogenousOofResidualError(
            "validation_days doit etre strictement compris entre 0 et training_days."
        )
    parsed = pd.DatetimeIndex(pd.to_datetime(list(origins), utc=True, errors="coerce"))
    if parsed.isna().any():
        raise ExogenousOofResidualError("Origine invalide dans le panel OOF.")
    parsed = parsed.sort_values().drop_duplicates()
    required = int(training_days + oof_days + holdout_days)
    if len(parsed) < required:
        missing = required - len(parsed)
        raise ExogenousOofResidualError(
            "Historique insuffisant pour un correcteur LoRA reellement OOF: "
            f"{len(parsed)} origines disponibles, {required} requises "
            f"({training_days} amorcage + {oof_days} OOF + {holdout_days} holdout); "
            f"il manque {missing} jours."
        )
    selected = parsed[-required:]
    local = selected.tz_convert(timezone_name)
    try:
        cutoff_hour, cutoff_minute = (
            int(part) for part in str(cutoff_local_time).split(":")
        )
    except Exception as exc:
        raise ExogenousOofResidualError("cutoff_local_time invalide.") from exc
    if any(
        timestamp.hour != cutoff_hour or timestamp.minute != cutoff_minute
        for timestamp in local
    ):
        raise ExogenousOofResidualError(
            f"Toutes les origines doivent etre a {cutoff_local_time} local."
        )
    local_days = local.normalize().tz_localize(None)
    if len(local_days) > 1:
        deltas = np.diff(local_days.to_numpy(dtype="datetime64[D]")).astype(int)
        if not bool(np.all(deltas == 1)):
            raise ExogenousOofResidualError(
                "Les origines du plan OOF doivent etre journalieres et consecutives."
            )

    calibration_start = training_days
    calibration_end = calibration_start + oof_days
    calibration = tuple(selected[calibration_start:calibration_end])
    holdout = tuple(selected[calibration_end:])
    folds: list[OofFold] = []
    for offset in range(0, oof_days, block_days):
        predicted = tuple(calibration[offset : offset + block_days])
        fit = tuple(selected[offset : offset + training_days])
        if fit[-1] >= predicted[0]:  # pragma: no cover - construction guard.
            raise ExogenousOofResidualError("Le fit d'un fold chevauche sa prediction.")
        train = fit[:-validation_days]
        validation = fit[-validation_days:]
        folds.append(
            OofFold(
                index=len(folds) + 1,
                fit_origins=fit,
                train_origins=train,
                validation_origins=validation,
                prediction_origins=predicted,
            )
        )
    return OofPlan(
        folds=tuple(folds),
        calibration_origins=calibration,
        holdout_origins=holdout,
        required_origins=required,
    )


def _origin_inventory(path: Path, origin_column: str) -> pd.DatetimeIndex:
    if not path.is_file():
        raise ExogenousOofResidualError(
            f"Panel de calibration introuvable: {path}. Construisez-le avec "
            "Exogenous.ps1 -Action CalibrationPanel."
        )
    suffixes = "".join(path.suffixes).casefold()
    if suffixes.endswith(".parquet"):
        frame = pd.read_parquet(path, columns=[origin_column])
    else:
        frame = pd.read_csv(path, usecols=[origin_column])
    origins = pd.to_datetime(frame[origin_column], utc=True, errors="coerce")
    if origins.isna().any():
        raise ExogenousOofResidualError("Le panel contient des origines invalides.")
    return pd.DatetimeIndex(origins.drop_duplicates()).sort_values()


def _fold_name(fold: OofFold, timezone_name: str) -> str:
    first = fold.prediction_origins[0].tz_convert(timezone_name).date()
    last = fold.prediction_origins[-1].tz_convert(timezone_name).date()
    return f"fold_{fold.index:03d}_{first:%Y%m%d}_{last:%Y%m%d}"


def _assert_final_bundle_matches_recipe(
    config: ExogenousFineTuneConfig,
    run_directory: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, str, str, Mapping[str, Any]]:
    schema_path = run_directory / "schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExogenousOofResidualError(f"Schema LoRA illisible: {schema_path}.") from exc
    expected_schema = _schema_payload(config)
    if schema != expected_schema:
        raise ExogenousOofResidualError(
            "Le schema du checkpoint final differe de la configuration OOF."
        )
    declared_panel_sha = manifest.get("panel_sha256")
    if declared_panel_sha != _sha256_file(config.panel_path):
        raise ExogenousOofResidualError(
            "Le panel 365+365 original ne correspond plus au checkpoint final."
        )
    training = manifest.get("training")
    expected_training = {
        "learning_rate": config.learning_rate,
        "num_steps": config.num_steps,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "lora_config": dict(config.lora_config),
    }
    if not isinstance(training, Mapping) or any(
        training.get(key) != value for key, value in expected_training.items()
    ):
        raise ExogenousOofResidualError(
            "Les hyperparametres du checkpoint final different de la recette OOF."
        )
    if int(manifest.get("training_window_days", -1)) != TRAINING_DAYS or int(
        manifest.get("evaluation_days", -1)
    ) != HOLDOUT_DAYS:
        raise ExogenousOofResidualError(
            "Le checkpoint final doit conserver le contrat 365 fit + 365 holdout."
        )
    source = Path(resolve_local_model_source(config)).resolve()
    base_hash = sha256_directory(source)
    if manifest.get("base_model_snapshot_sha256") != base_hash:
        raise ExogenousOofResidualError(
            "Le snapshot Chronos-2 local differe de celui du checkpoint final."
        )
    deployment_checkpoint_sha = str(manifest.get("checkpoint_sha256", ""))
    if len(deployment_checkpoint_sha) != 64:
        raise ExogenousOofResidualError("SHA du checkpoint LoRA final absent.")
    recipe = {
        "schema_version": 1,
        "base_model_snapshot_sha256": base_hash,
        "schema_sha256": _sha256_file(schema_path),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "context_length": config.context_length,
        "prediction_length": config.prediction_length,
        "training_window_days": TRAINING_DAYS,
        "validation_days": config.validation_days,
        "training": expected_training,
        "cross_learning": False,
    }
    return source, base_hash, deployment_checkpoint_sha, recipe


def _assert_plan_matches_final_split(
    plan: OofPlan,
    *,
    manifest: Mapping[str, Any],
    validation_days: int,
) -> None:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ExogenousOofResidualError("Splits du checkpoint final absents.")
    train = plan.calibration_origins[:-validation_days]
    validation = plan.calibration_origins[-validation_days:]
    expected = {
        "train": _range(train),
        "validation": _range(validation),
        "evaluation_holdout": _range(plan.holdout_origins),
    }
    for name, contract in expected.items():
        declared = splits.get(name)
        if not isinstance(declared, Mapping) or any(
            declared.get(key) != value for key, value in contract.items()
        ):
            raise ExogenousOofResidualError(
                "Le panel de calibration n'est pas aligne sur le split scelle du "
                f"checkpoint final ({name})."
            )


def _fit_fold_locally(
    *,
    fold: OofFold,
    fold_directory: Path,
    panel: pd.DataFrame,
    config: ExogenousFineTuneConfig,
    model_source: Path,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    pipeline_loader: PipelineLoader,
) -> dict[str, Any]:
    train_inputs = build_fit_inputs(panel, fold.train_origins, config)
    validation_inputs = build_fit_inputs(panel, fold.validation_origins, config)
    variates = len(config.target_columns) + len(config.covariate_columns)
    if config.batch_size < variates:
        raise ExogenousOofResidualError(
            f"batch_size={config.batch_size} < {variates} variates par groupe."
        )
    staging = fold_directory.parent / f".{fold_directory.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    load_kwargs: dict[str, Any] = {
        "device_map": config.device_map,
        "local_files_only": True,
    }
    random.seed(config.seed)
    np.random.seed(config.seed)
    try:
        base = pipeline_loader(model_source, load_kwargs)
        fitted = base.fit(
            inputs=train_inputs,
            validation_inputs=validation_inputs,
            prediction_length=config.prediction_length,
            finetune_mode="lora",
            lora_config=dict(config.lora_config),
            context_length=config.context_length,
            min_past=config.context_length,
            learning_rate=config.learning_rate,
            num_steps=config.num_steps,
            batch_size=config.batch_size,
            output_dir=staging,
            finetuned_ckpt_name="checkpoint",
            seed=config.seed,
            data_seed=config.seed,
            remove_printer_callback=True,
        )
        checkpoint_staging = staging / "checkpoint"
        if not checkpoint_staging.is_dir():
            checkpoint_staging.mkdir(parents=True, exist_ok=True)
            fitted.save_pretrained(checkpoint_staging)
        _pin_adapter_base(checkpoint_staging, model_source)
        if not (checkpoint_staging / "adapter_config.json").is_file():
            raise ExogenousOofResidualError(
                "Fold refuse: adapter_config.json absent (fallback full fine-tuning)."
            )
        payload = {
            **_fold_manifest_core(
                fold,
                recipe=recipe,
                recipe_sha256=recipe_sha256,
                checkpoint_sha256=sha256_directory(checkpoint_staging),
            ),
            "checkpoint_materialization": {
                "mode": "locally_fitted",
                "predictions_shared": False,
                "corrector_shared": False,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(staging / "fold_manifest.json", payload)
        os.replace(staging, fold_directory)
        return payload
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _fit_fold(
    *,
    fold: OofFold,
    fold_directory: Path,
    panel: pd.DataFrame,
    config: ExogenousFineTuneConfig,
    model_source: Path,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    pipeline_loader: PipelineLoader,
    shared_cache: _SharedFoldCheckpointCache | None = None,
    shared_cache_wait_seconds: float = DEFAULT_SHARED_CACHE_WAIT_SECONDS,
    shared_cache_poll_seconds: float = DEFAULT_SHARED_CACHE_POLL_SECONDS,
) -> dict[str, Any]:
    manifest_path = fold_directory / "fold_manifest.json"
    checkpoint = fold_directory / "checkpoint"
    if manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExogenousOofResidualError(
                f"Manifeste fold illisible: {manifest_path}."
            ) from exc
        if not checkpoint.is_dir():
            raise ExogenousOofResidualError(
                f"Checkpoint fold absent ou divergent: {fold_directory}."
            )
        _assert_regular_checkpoint_tree(checkpoint)
        checkpoint_sha = sha256_directory(checkpoint)
        expected_core = _fold_manifest_core(
            fold,
            recipe=recipe,
            recipe_sha256=recipe_sha256,
            checkpoint_sha256=checkpoint_sha,
        )
        immutable_keys = tuple(key for key in expected_core if key != "predictions_sha256")
        if any(cached.get(key) != expected_core[key] for key in immutable_keys):
            raise ExogenousOofResidualError(
                f"Cache fold incompatible avec la recette ou le plan: {fold_directory}."
            )
        if shared_cache is not None:
            _publish_fold_to_shared_cache(
                cache=shared_cache,
                fold=fold,
                timezone_name=config.timezone,
                checkpoint=checkpoint,
            )
        return cached
    if fold_directory.exists():
        raise ExogenousOofResidualError(
            f"Cache fold partiel sans manifeste: {fold_directory}."
        )

    if shared_cache is not None:
        materialized = _materialize_fold_from_shared_cache(
            cache=shared_cache,
            fold=fold,
            fold_directory=fold_directory,
            timezone_name=config.timezone,
            recipe=recipe,
            recipe_sha256=recipe_sha256,
        )
        if materialized is not None:
            return materialized

    if shared_cache is None:
        return _fit_fold_locally(
            fold=fold,
            fold_directory=fold_directory,
            panel=panel,
            config=config,
            model_source=model_source,
            recipe=recipe,
            recipe_sha256=recipe_sha256,
            pipeline_loader=pipeline_loader,
        )

    with _shared_fold_fit_claim(
        cache=shared_cache,
        fold=fold,
        timezone_name=config.timezone,
        wait_seconds=float(shared_cache_wait_seconds),
        poll_seconds=float(shared_cache_poll_seconds),
    ) as elected_to_fit:
        materialized = _materialize_fold_from_shared_cache(
            cache=shared_cache,
            fold=fold,
            fold_directory=fold_directory,
            timezone_name=config.timezone,
            recipe=recipe,
            recipe_sha256=recipe_sha256,
        )
        if materialized is not None:
            return materialized
        if not elected_to_fit:
            raise ExogenousOofResidualError(
                "Le fold partage a ete annonce publie mais reste introuvable."
            )
        payload = _fit_fold_locally(
            fold=fold,
            fold_directory=fold_directory,
            panel=panel,
            config=config,
            model_source=model_source,
            recipe=recipe,
            recipe_sha256=recipe_sha256,
            pipeline_loader=pipeline_loader,
        )
        _publish_fold_to_shared_cache(
            cache=shared_cache,
            fold=fold,
            timezone_name=config.timezone,
            checkpoint=fold_directory / "checkpoint",
        )
        return payload


def _load_fold_pipeline(
    checkpoint: Path,
    *,
    config: ExogenousFineTuneConfig,
    pipeline_loader: PipelineLoader,
) -> Any:
    return pipeline_loader(
        checkpoint,
        {
            "device_map": config.device_map,
            "local_files_only": True,
            "import_allowlist": ["chronos.chronos2.model"],
        },
    )


def _predict_fold(
    *,
    fold: OofFold,
    fold_directory: Path,
    fold_manifest: dict[str, Any],
    panel: pd.DataFrame,
    config: ExogenousFineTuneConfig,
    item_id: str,
    target_column: str,
    batch_size: int,
    pipeline_loader: PipelineLoader,
) -> pd.DataFrame:
    prediction_path = fold_directory / "oof_predictions.csv.gz"
    declared_sha = fold_manifest.get("predictions_sha256")
    if prediction_path.is_file():
        actual_sha = _sha256_file(prediction_path)
        if declared_sha is not None and declared_sha != actual_sha:
            raise ExogenousOofResidualError(
                f"Predictions fold divergentes: {prediction_path}."
            )
        if declared_sha is None:
            fold_manifest["predictions_sha256"] = actual_sha
            _write_json_atomic(fold_directory / "fold_manifest.json", fold_manifest)
        return pd.read_csv(prediction_path)

    selected = panel.loc[
        panel[config.origin_column].isin(fold.prediction_origins)
        & panel[config.item_column].astype(str).eq(item_id)
    ]
    present = pd.DatetimeIndex(
        selected[config.origin_column].drop_duplicates()
    ).sort_values()
    expected = pd.DatetimeIndex(fold.prediction_origins)
    if not present.equals(expected):
        raise ExogenousOofResidualError(
            f"Le fold {fold.index} ne couvre pas toutes les origines de {item_id}."
        )
    try:
        target_index = config.target_columns.index(target_column)
    except ValueError as exc:
        raise ExogenousOofResidualError(f"Target inconnue: {target_column}.") from exc
    samples_by_horizon: dict[int, list[tuple[pd.Timestamp, dict[str, Any], pd.DatetimeIndex, np.ndarray]]] = {
        23: [],
        24: [],
        25: [],
    }
    for origin in fold.prediction_origins:
        group = selected.loc[selected[config.origin_column].eq(origin)]
        payload, horizon, actual = build_inference_input(group, config)
        samples_by_horizon[len(horizon)].append((origin, payload, horizon, actual))
    pipeline = _load_fold_pipeline(
        fold_directory / "checkpoint", config=config, pipeline_loader=pipeline_loader
    )
    rows_by_origin: dict[pd.Timestamp, pd.DataFrame] = {}
    for horizon_hours, samples in samples_by_horizon.items():
        if not samples:
            continue
        outputs = _predict_quantiles_batch(
            pipeline,
            [sample[1] for sample in samples],
            prediction_length=horizon_hours,
            context_length=config.context_length,
            batch_size=batch_size,
        )
        for (origin, _payload, horizon, actual), values in zip(
            samples, outputs, strict=True
        ):
            rows_by_origin[pd.Timestamp(origin)] = pd.DataFrame(
                {
                    "delivery_start_utc": horizon,
                    "forecast_origin_utc": pd.DatetimeIndex([origin] * len(horizon)),
                    "actual": actual[target_index],
                    "candidate_q10": values[target_index, :, 0],
                    "candidate_q50": values[target_index, :, 1],
                    "candidate_q90": values[target_index, :, 2],
                }
            )
    result = pd.concat(
        [rows_by_origin[pd.Timestamp(origin)] for origin in fold.prediction_origins],
        ignore_index=True,
    )
    temporary = prediction_path.with_name(
        f".{prediction_path.name}.tmp-{uuid.uuid4().hex}"
    )
    result.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, prediction_path)
    fold_manifest["predictions_sha256"] = _sha256_file(prediction_path)
    _write_json_atomic(fold_directory / "fold_manifest.json", fold_manifest)
    return result


def _delivery_day(origin: pd.Timestamp, timezone_name: str) -> str:
    return str((origin.tz_convert(timezone_name) + pd.DateOffset(days=1)).date())


def _authenticate_completed_calibration_manifest(
    output_directory: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Authenticate every final artifact before allowing a completed resume."""

    expected_header = {
        "schema_version": 1,
        "purpose": "chronos2_exogenous_lora_residual_calibration",
        "status": "complete",
    }
    failures = [
        key for key, value in expected_header.items() if manifest.get(key) != value
    ]
    if failures:
        raise ExogenousOofResidualError(
            "Manifeste de calibration complete invalide: " + ", ".join(failures) + "."
        )
    artifacts = (
        (
            "predictions_relative_path",
            "predictions_sha256",
            "oof_predictions_365.csv.gz",
        ),
        (
            "oof_audit_relative_path",
            "oof_audit_sha256",
            "oof_predictions_365.csv.gz.audit.json",
        ),
        (
            "corrector_relative_path",
            "corrector_sha256",
            "residual_corrector.json",
        ),
    )
    for relative_key, sha_key, expected_relative in artifacts:
        relative = manifest.get(relative_key)
        declared_sha = manifest.get(sha_key)
        if (
            relative != expected_relative
            or not isinstance(declared_sha, str)
            or len(declared_sha) != 64
            or any(character not in "0123456789abcdef" for character in declared_sha)
        ):
            raise ExogenousOofResidualError(
                f"Manifeste complete: identite {relative_key}/{sha_key} invalide."
            )
        path = (output_directory / expected_relative).resolve()
        if path.parent != output_directory.resolve() or not path.is_file():
            raise ExogenousOofResidualError(
                f"Manifeste complete: artefact final absent: {path}."
            )
        if _sha256_file(path) != declared_sha:
            raise ExogenousOofResidualError(
                f"Manifeste complete: SHA divergent pour {expected_relative}; "
                "re-scellage silencieux refuse."
            )


def _commit_completed_calibration_manifest(
    path: Path,
    completed: Mapping[str, Any],
    *,
    existing_manifest: Mapping[str, Any] | None,
) -> None:
    """Publish once; an already-complete seal is comparison-only."""

    if existing_manifest is not None and existing_manifest.get("status") == "complete":
        expected = dict(completed)
        expected["completed_at_utc"] = existing_manifest.get("completed_at_utc")
        if expected != dict(existing_manifest):
            raise ExogenousOofResidualError(
                "Le manifeste complete diverge apres reprise; re-scellage refuse."
            )
        return
    _write_json_atomic(path, completed)


def _reuse_corrector_if_exact(
    path: Path,
    *,
    predictions_path: Path,
    audit_path: Path,
    completed_manifest: Mapping[str, Any] | None,
) -> bool:
    if not path.is_file():
        return False
    if completed_manifest is None or completed_manifest.get("status") != "complete":
        raise ExogenousOofResidualError(
            "Correcteur residuel existant sans manifeste complete authentifie; "
            "reprise et re-scellage refuses. Utilisez un nouveau repertoire de sortie."
        )
    declared_sha = completed_manifest.get("corrector_sha256")
    if not isinstance(declared_sha, str) or _sha256_file(path) != declared_sha:
        raise ExogenousOofResidualError(
            "Correcteur residuel divergent de son manifeste complete; "
            "re-scellage silencieux refuse."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExogenousOofResidualError(f"Correcteur existant illisible: {path}.") from exc
    expected = {
        "oof_training_predictions_sha256": _sha256_file(predictions_path),
        "oof_training_audit_sha256": _sha256_file(audit_path),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ExogenousOofResidualError(
            f"Correcteur existant incompatible avec l'evidence OOF: {path}."
        )
    return True


def calibrate_residual_corrector(
    config_or_path: ExogenousFineTuneConfig | str | Path,
    *,
    run_directory: str | Path | None = None,
    panel_path: str | Path | None = None,
    panel_audit_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    shared_checkpoint_cache_directory: str | Path | None = None,
    item_id: str,
    target_column: str | None = None,
    block_days: int = 30,
    batch_size: int = 64,
    pipeline_loader: PipelineLoader | None = None,
    progress: Callable[[int, int, OofFold, str], None] | None = None,
) -> ResidualCalibrationArtifacts:
    """Train fold-specific LoRA adapters, seal OOF rows, then fit the corrector."""

    base_config = (
        config_or_path
        if isinstance(config_or_path, ExogenousFineTuneConfig)
        else load_config(config_or_path)
    )
    if int(batch_size) <= 0:
        raise ExogenousOofResidualError("batch_size doit etre positif.")
    run_dir = (
        Path(run_directory).expanduser().resolve()
        if run_directory is not None
        else base_config.output_directory
    )
    if (panel_path is None) != (panel_audit_path is None):
        raise ExogenousOofResidualError(
            "panel_path et panel_audit_path doivent etre fournis ensemble."
        )
    config = (
        replace(
            base_config,
            panel_path=Path(panel_path).expanduser().resolve(),
            panel_audit_path=Path(panel_audit_path).expanduser().resolve(),
        )
        if panel_path is not None and panel_audit_path is not None
        else base_config
    )
    inventory = _origin_inventory(config.panel_path, config.origin_column)
    plan = build_oof_plan(
        inventory,
        timezone_name=config.timezone,
        cutoff_local_time=config.cutoff_local_time,
        validation_days=config.validation_days,
        block_days=int(block_days),
    )
    manifest = verify_bundle(run_dir)
    model_source, _base_hash, deployment_sha, recipe = _assert_final_bundle_matches_recipe(
        base_config, run_dir, manifest
    )
    _assert_plan_matches_final_split(
        plan, manifest=manifest, validation_days=config.validation_days
    )
    # Reuse the existing strict PIT/group/DST validator over the exact 1095-day
    # suffix.  The trainer still retains its original 365-day recipe.
    validation_config = replace(
        config,
        training_window_days=TRAINING_DAYS + OOF_DAYS,
        evaluation_days=HOLDOUT_DAYS,
    )
    panel, _split, panel_audit = validate_panel(
        read_panel(config.panel_path), validation_config
    )
    validated_plan = build_oof_plan(
        pd.DatetimeIndex(panel[config.origin_column].drop_duplicates()),
        timezone_name=config.timezone,
        cutoff_local_time=config.cutoff_local_time,
        validation_days=config.validation_days,
        block_days=int(block_days),
    )
    if validated_plan != plan:
        raise ExogenousOofResidualError("Le plan change apres validation complete du panel.")
    items = tuple(sorted(panel[config.item_column].astype(str).unique()))
    zone = str(item_id).strip()
    if zone not in items:
        raise ExogenousOofResidualError(
            f"Item {zone!r} absent du panel; disponibles={list(items)}."
        )
    selected_target = target_column or config.target_columns[0]
    if selected_target not in config.target_columns:
        raise ExogenousOofResidualError(f"Target inconnue: {selected_target}.")
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else run_dir / "residual_calibration" / zone.casefold()
    )
    try:
        output.relative_to(config.project_root)
    except ValueError as exc:
        raise ExogenousOofResidualError(
            "La sortie de calibration doit rester sous project_root."
        ) from exc
    calibration_manifest_path = output / "calibration_manifest.json"
    existing_calibration_manifest: dict[str, Any] | None = None
    if calibration_manifest_path.is_file():
        existing_calibration_manifest = _read_json_object(
            calibration_manifest_path,
            label="manifeste de calibration residuelle",
        )
        status = existing_calibration_manifest.get("status")
        if status == "complete":
            _authenticate_completed_calibration_manifest(
                output, existing_calibration_manifest
            )
        elif status != "running":
            raise ExogenousOofResidualError(
                f"Statut du manifeste de calibration invalide: {status!r}."
            )
    output.mkdir(parents=True, exist_ok=True)
    folds_directory = output / "folds"
    folds_directory.mkdir(exist_ok=True)
    recipe_sha = _json_sha256(recipe)
    shared_cache = _prepare_shared_fold_cache(
        requested_root=shared_checkpoint_cache_directory,
        config=config,
        run_directory=run_dir,
        manifest=manifest,
        plan=plan,
        recipe=recipe,
        recipe_sha256=recipe_sha,
        panel_audit=panel_audit,
        block_days=int(block_days),
    )
    plan_contract = {
        "schema_version": 1,
        "purpose": "chronos2_exogenous_lora_residual_calibration",
        "item_id": zone,
        "target_column": selected_target,
        "block_days": int(block_days),
        "training_days": TRAINING_DAYS,
        "oof_days": OOF_DAYS,
        "holdout_days": HOLDOUT_DAYS,
        "required_origins": plan.required_origins,
        "calibration_origins": _range(plan.calibration_origins),
        "holdout_origins": _range(plan.holdout_origins),
        "fold_count": len(plan.folds),
        "recipe_sha256": recipe_sha,
        "panel_sha256": panel_audit["upstream_panel_audit"]["panel_sha256"],
        "panel_audit_sha256": panel_audit["upstream_panel_audit"][
            "panel_audit_sha256"
        ],
        "deployment_checkpoint_sha256": deployment_sha,
        "shared_fold_checkpoint_cache_enabled": shared_cache is not None,
        "shared_fold_checkpoint_cache_contract_sha256": (
            shared_cache.contract_sha256 if shared_cache is not None else None
        ),
        "shared_fold_checkpoint_cache_scope": (
            "checkpoints_only_no_predictions_no_corrector"
            if shared_cache is not None
            else None
        ),
        "production_pit_evidence": bool(panel_audit["production_pit_evidence"]),
        "holdout_label_binding": panel_audit["evaluation_label_binding"],
        "holdout_targets_used_for_fit": False,
        "promotion_eligible": False,
    }
    if existing_calibration_manifest is not None:
        existing = existing_calibration_manifest
        immutable_keys = tuple(plan_contract)
        if any(existing.get(key) != plan_contract[key] for key in immutable_keys):
            raise ExogenousOofResidualError(
                "Le cache de calibration existe avec un contrat different; "
                "utilisez un nouveau repertoire de sortie."
            )
    else:
        _write_json_atomic(
            calibration_manifest_path,
            {**plan_contract, "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat()},
        )

    loader = pipeline_loader or _default_pipeline_loader
    fold_rows: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    for number, fold in enumerate(plan.folds, start=1):
        if progress is not None:
            progress(number, len(plan.folds), fold, "fit_or_resume")
        directory = folds_directory / _fold_name(fold, config.timezone)
        fold_manifest = _fit_fold(
            fold=fold,
            fold_directory=directory,
            panel=panel,
            config=config,
            model_source=model_source,
            recipe=recipe,
            recipe_sha256=recipe_sha,
            pipeline_loader=loader,
            shared_cache=shared_cache,
        )
        if progress is not None:
            progress(number, len(plan.folds), fold, "predict_or_resume")
        predictions = _predict_fold(
            fold=fold,
            fold_directory=directory,
            fold_manifest=fold_manifest,
            panel=panel,
            config=config,
            item_id=zone,
            target_column=selected_target,
            batch_size=int(batch_size),
            pipeline_loader=loader,
        )
        fold_rows.append(predictions)
        refreshed = json.loads(
            (directory / "fold_manifest.json").read_text(encoding="utf-8")
        )
        fold_audits.append(
            {
                "fold_index": fold.index,
                "fit_origins": refreshed["fit_origins"],
                "train_origins": refreshed["train_origins"],
                "validation_origins": refreshed["validation_origins"],
                "prediction_origins": refreshed["prediction_origins"],
                "checkpoint_sha256": refreshed["checkpoint_sha256"],
                "predictions_sha256": refreshed["predictions_sha256"],
                "refit_uses_only_strictly_prior_days": True,
                "checkpoint_materialization": refreshed.get(
                    "checkpoint_materialization"
                ),
            }
        )

    evidence = pd.concat(fold_rows, ignore_index=True)
    evidence["delivery_start_utc"] = pd.to_datetime(
        evidence["delivery_start_utc"], utc=True
    )
    evidence["forecast_origin_utc"] = pd.to_datetime(
        evidence["forecast_origin_utc"], utc=True
    )
    evidence = evidence.sort_values("delivery_start_utc", kind="stable").reset_index(
        drop=True
    )
    if evidence["delivery_start_utc"].duplicated().any():
        raise ExogenousOofResidualError("Heure OOF dupliquee entre deux folds.")
    observed_origins = pd.DatetimeIndex(
        evidence["forecast_origin_utc"].drop_duplicates()
    ).sort_values()
    if not observed_origins.equals(pd.DatetimeIndex(plan.calibration_origins)):
        raise ExogenousOofResidualError("Les folds ne couvrent pas les 365 origines OOF.")
    predictions_path = output / "oof_predictions_365.csv.gz"
    if predictions_path.exists():
        cached = pd.read_csv(predictions_path)
        for column in ("delivery_start_utc", "forecast_origin_utc"):
            cached[column] = pd.to_datetime(cached[column], utc=True, errors="coerce")
        try:
            pd.testing.assert_frame_equal(
                cached.reset_index(drop=True),
                evidence.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise ExogenousOofResidualError(
                "Evidence OOF consolidee existante divergente."
            ) from exc
        if cached.isna().any().any():
            raise ExogenousOofResidualError("Evidence OOF consolidee invalide.")
    else:
        temporary = predictions_path.with_name(
            f".{predictions_path.name}.tmp-{uuid.uuid4().hex}"
        )
        evidence.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, predictions_path)

    checkpoint_set_sha = _json_sha256(
        [
            {"fold_index": item["fold_index"], "checkpoint_sha256": item["checkpoint_sha256"]}
            for item in fold_audits
        ]
    )
    audit_path = output / "oof_predictions_365.csv.gz.audit.json"
    audit = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "purpose": "chronos2_exogenous_blocked_prequential_oof",
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "candidate_model": "chronos2_exogenous",
        # Deployment identity anchor only. OOF rows are produced by the fold
        # checkpoints below, never by this final checkpoint on its fit days.
        "candidate_checkpoint_sha256": deployment_sha,
        "candidate_checkpoint_role": "deployment_identity_anchor_not_oof_predictor",
        "deployment_checkpoint_used_for_oof": False,
        "fold_checkpoints_are_origin_specific": True,
        "fold_checkpoint_set_sha256": checkpoint_set_sha,
        "fold_candidate_recipe_sha256": recipe_sha,
        "fold_count": len(fold_audits),
        "folds": fold_audits,
        "shared_fold_checkpoint_cache": {
            "enabled": shared_cache is not None,
            "contract_sha256": (
                shared_cache.contract_sha256 if shared_cache is not None else None
            ),
            "scope": (
                "checkpoints_only_no_predictions_no_corrector"
                if shared_cache is not None
                else None
            ),
            "zone_predictions_shared": False,
            "zone_corrector_shared": False,
        },
        "training_days": OOF_DAYS,
        "fold_lookback_days": TRAINING_DAYS,
        "training_start_day": _delivery_day(plan.calibration_origins[0], config.timezone),
        "training_end_day": _delivery_day(plan.calibration_origins[-1], config.timezone),
        "holdout_start_day": _delivery_day(plan.holdout_origins[0], config.timezone),
        "predictions_sha256": _sha256_file(predictions_path),
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "future_actuals_used_as_features": False,
        "holdout_used_for_fit": False,
        "holdout_targets_used_for_fit": False,
        "holdout_label_binding": panel_audit["evaluation_label_binding"],
        "selection_frozen_before_oof": True,
        "panel_sha256": panel_audit["upstream_panel_audit"]["panel_sha256"],
        "panel_audit_sha256": panel_audit["upstream_panel_audit"][
            "panel_audit_sha256"
        ],
        "production_pit_evidence": bool(panel_audit["production_pit_evidence"]),
        "promotion_eligible": False,
    }
    if audit_path.is_file():
        existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing_audit != audit:
            raise ExogenousOofResidualError("Sidecar OOF existant divergent.")
    else:
        _write_json_atomic(audit_path, audit)

    corrector_path = output / "residual_corrector.json"
    if not _reuse_corrector_if_exact(
        corrector_path,
        predictions_path=predictions_path,
        audit_path=audit_path,
        completed_manifest=(
            existing_calibration_manifest
            if existing_calibration_manifest is not None
            and existing_calibration_manifest.get("status") == "complete"
            else None
        ),
    ):
        fit_oof_residual_corrector(
            oof_predictions_path=predictions_path,
            oof_audit_path=audit_path,
            holdout_start_day=audit["holdout_start_day"],
            output_path=corrector_path,
            timezone_name=config.timezone,
        )
    completed = {
        **plan_contract,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_relative_path": predictions_path.relative_to(output).as_posix(),
        "predictions_sha256": _sha256_file(predictions_path),
        "oof_audit_relative_path": audit_path.relative_to(output).as_posix(),
        "oof_audit_sha256": _sha256_file(audit_path),
        "corrector_relative_path": corrector_path.relative_to(output).as_posix(),
        "corrector_sha256": _sha256_file(corrector_path),
        "fold_checkpoint_set_sha256": checkpoint_set_sha,
    }
    _commit_completed_calibration_manifest(
        calibration_manifest_path,
        completed,
        existing_manifest=existing_calibration_manifest,
    )
    return ResidualCalibrationArtifacts(
        directory=output,
        predictions_path=predictions_path,
        audit_path=audit_path,
        corrector_path=corrector_path,
        manifest_path=calibration_manifest_path,
        folds=len(plan.folds),
        shared_checkpoint_cache_directory=(
            shared_cache.root if shared_cache is not None else None
        ),
        shared_checkpoint_cache_contract_sha256=(
            shared_cache.contract_sha256 if shared_cache is not None else None
        ),
    )


__all__ = [
    "ExogenousOofResidualError",
    "OofFold",
    "OofPlan",
    "ResidualCalibrationArtifacts",
    "build_oof_plan",
    "calibrate_residual_corrector",
]
