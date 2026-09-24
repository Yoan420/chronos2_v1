"""Prospective, fail-closed precommitment for a final LoRA shadow epoch.

The rolling evaluation and the first live-shadow forecast meet at a delicate
temporal boundary.  For a shadow delivery day ``J`` the model is run at
``J-1 08:00`` local time, while the final score for the rolling window ending
on ``J-1`` can legitimately be materialised later.  This module separates the
two facts without weakening either one:

* phase A freezes the checkpoint, OOF residual corrector, exact 365-day
  qualification window and first shadow day before the first forecast origin;
* phase B may attach/finalise rolling metrics later, but only for the hashes and
  window committed in phase A.

An epoch freeze is *not* promotion evidence.  In particular it cannot turn a
research PIT panel into production PIT evidence and it cannot repair a missed
first shadow day.  The ordinary final-shadow and governance validators remain
the authority for scoring and promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .final_pipeline import (
    FINAL_EVIDENCE_NAME,
    FINAL_MANIFEST_NAME,
    _normalise_evidence,
    _validate_oof_chain,
)
from .governance import validate_experiment_manifest
from .lora_finetune import sha256_directory, verify_bundle
from .production import _validate_final_pipeline_evidence


EPOCH_SCHEMA_VERSION = 1
EPOCH_KIND = "chronos2_exogenous_prospective_shadow_epoch_freeze"
EPOCH_MANIFEST_NAME = "shadow_epoch_freeze.json"
EXPERIMENT_COPY_NAME = "experiment_manifest_at_freeze.json"
SCHEMA_COPY_NAME = "schema.json"
CORRECTOR_COPY_NAME = "residual_corrector.json"
OOF_AUDIT_COPY_NAME = "oof_audit.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ShadowEpochError(RuntimeError):
    """Raised when a prospective shadow epoch cannot be safely opened."""


@dataclass(frozen=True)
class ShadowEpochAssessment:
    """Read-only assessment of the next immutable shadow boundary."""

    zone: str
    timezone: str
    cutoff_local_time: str
    holdout_start_day: str
    holdout_end_day: str
    holdout_days: int
    first_shadow_day: str
    first_shadow_origin_utc: str
    freeze_deadline_utc: str
    minimum_shadow_end_day: str
    assessed_at_utc: str
    requested_first_shadow_day: str
    continuity_ready: bool
    freeze_window_open: bool
    checkpoint_verified: bool
    residual_corrector_verified: bool
    historical_training_pit_ready: bool
    residual_calibration_pit_ready: bool
    production_pit_ready: bool
    final_backtest_already_sealed: bool
    two_phase_finalisation_required: bool
    shadow_days_required: int
    blockers: tuple[str, ...]

    @property
    def ready_to_freeze(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ready_to_freeze"] = self.ready_to_freeze
        # The freeze only authorises prospective collection.  Promotion still
        # requires final rolling metrics, 30 observed days and governance.
        result["promotion_eligible"] = False
        return result


def _utc_now() -> pd.Timestamp:
    """Non-injectable production clock seam; tests monkeypatch this symbol."""

    return pd.Timestamp.now(tz="UTC")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ShadowEpochError(f"{label} absent: {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShadowEpochError(f"{label} illisible: {path}.") from exc
    if not isinstance(payload, dict):
        raise ShadowEpochError(f"{label} doit etre un objet JSON.")
    return payload


def _digest(value: object, *, label: str) -> str:
    result = str(value).strip().lower() if isinstance(value, str) else ""
    if _SHA256.fullmatch(result) is None:
        raise ShadowEpochError(f"{label} doit etre un SHA-256.")
    return result


def _parse_cutoff(value: object) -> tuple[int, int, str]:
    cutoff = str(value).strip()
    try:
        hour, minute = (int(part) for part in cutoff.split(":"))
    except (TypeError, ValueError) as exc:
        raise ShadowEpochError("cutoff_local_time invalide.") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ShadowEpochError("cutoff_local_time invalide.")
    return hour, minute, f"{hour:02d}:{minute:02d}"


def _local_cutoff(
    day_value: date,
    timezone_value: str | ZoneInfo,
    *,
    hour: int,
    minute: int,
) -> pd.Timestamp:
    """Build a local wall-clock cutoff without DST elapsed-time drift."""

    try:
        return pd.Timestamp(
            datetime.combine(day_value, time(hour=hour, minute=minute))
        ).tz_localize(timezone_value)
    except Exception as exc:
        raise ShadowEpochError(
            "Heure limite locale ambigue ou inexistante pour "
            f"{day_value.isoformat()}."
        ) from exc


def _declared_holdout(
    experiment: Mapping[str, Any], schema: Mapping[str, Any]
) -> dict[str, Any]:
    splits = experiment.get("splits")
    holdout = splits.get("evaluation_holdout") if isinstance(splits, Mapping) else None
    if not isinstance(holdout, Mapping):
        raise ShadowEpochError("evaluation_holdout gele absent du manifeste.")
    if holdout.get("count") != 365:
        raise ShadowEpochError("Le pre-engagement exige exactement 365 jours de holdout.")
    first_origin = pd.to_datetime(holdout.get("first_utc"), utc=True, errors="coerce")
    last_origin = pd.to_datetime(holdout.get("last_utc"), utc=True, errors="coerce")
    if pd.isna(first_origin) or pd.isna(last_origin):
        raise ShadowEpochError("Bornes UTC du holdout invalides.")
    timezone_name = str(schema.get("timezone", "")).strip()
    try:
        local_zone = ZoneInfo(timezone_name)
    except Exception as exc:
        raise ShadowEpochError("Timezone du schema invalide.") from exc
    hour, minute, cutoff = _parse_cutoff(schema.get("cutoff_local_time"))
    first_local = pd.Timestamp(first_origin).tz_convert(local_zone)
    last_local = pd.Timestamp(last_origin).tz_convert(local_zone)
    for name, value in (("premiere", first_local), ("derniere", last_local)):
        if (
            value.hour != hour
            or value.minute != minute
            or value.second != 0
            or value.microsecond != 0
        ):
            raise ShadowEpochError(
                f"Origine {name} du holdout differente de {cutoff} local."
            )
    origin_days = (last_local.date() - first_local.date()).days + 1
    if origin_days != 365:
        raise ShadowEpochError("Les origines du holdout ne couvrent pas 365 jours consecutifs.")
    holdout_start = first_local.date() + timedelta(days=1)
    holdout_end = last_local.date() + timedelta(days=1)
    first_shadow = holdout_end + timedelta(days=1)
    shadow_origin_local = _local_cutoff(
        first_shadow - timedelta(days=1),
        local_zone,
        hour=hour,
        minute=minute,
    )
    return {
        "timezone": timezone_name,
        "cutoff_local_time": cutoff,
        "holdout_start_day": holdout_start.isoformat(),
        "holdout_end_day": holdout_end.isoformat(),
        "holdout_days": 365,
        "first_origin_utc": pd.Timestamp(first_origin).isoformat(),
        "last_origin_utc": pd.Timestamp(last_origin).isoformat(),
        "first_shadow_day": first_shadow.isoformat(),
        "first_shadow_origin_utc": shadow_origin_local.tz_convert("UTC").isoformat(),
    }


def _validated_label_binding(
    experiment: Mapping[str, Any], oof_audit: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Validate the optional two-phase label contract without inventing one."""

    raw = experiment.get("evaluation_label_binding")
    if raw is None:
        # Backward compatibility: bundles trained before schema v1 did not
        # carry this field.  A newer OOF calibration may nevertheless emit
        # the canonical *empty* binding.  It is semantically identical to the
        # former fully-observed contract, whereas any unresolved cell remains
        # fail-closed.
        oof_binding = oof_audit.get("holdout_label_binding")
        if oof_binding is None:
            return None
        if (
            isinstance(oof_binding, Mapping)
            and oof_binding.get("schema_version") == 1
            and oof_binding.get("policy")
            == "last_evaluation_physical_horizon_only"
            and oof_binding.get("allow_unresolved_final_evaluation_day") is False
            and oof_binding.get("unresolved_groups") == []
            and oof_binding.get("unresolved_cells") == 0
            and oof_binding.get("labels_used_for_fit") is False
            and oof_binding.get("train_validation_labels_all_finite") is True
            and oof_binding.get("resolution_required_before_evaluation") is False
            and _SHA256.fullmatch(
                str(oof_binding.get("input_contract_sha256", "")).strip().lower()
            )
            is not None
            and oof_audit.get("holdout_targets_used_for_fit") is False
        ):
            return None
        else:
            raise ShadowEpochError(
                "Le correcteur declare un binding holdout absent du bundle LoRA."
            )
    if not isinstance(raw, Mapping):
        raise ShadowEpochError("evaluation_label_binding invalide.")
    binding = dict(raw)
    if (
        binding.get("schema_version") != 1
        or binding.get("policy") != "last_evaluation_physical_horizon_only"
        or type(binding.get("allow_unresolved_final_evaluation_day")) is not bool
        or type(binding.get("unresolved_cells")) is not int
        or int(binding.get("unresolved_cells", -1)) < 0
        or binding.get("labels_used_for_fit") is not False
        or binding.get("train_validation_labels_all_finite") is not True
        or not isinstance(binding.get("unresolved_groups"), list)
    ):
        raise ShadowEpochError("Contrat two-phase des labels holdout invalide.")
    _digest(
        binding.get("input_contract_sha256"),
        label="evaluation_label_binding.input_contract_sha256",
    )
    if bool(binding["unresolved_cells"]) != bool(
        binding.get("resolution_required_before_evaluation")
    ):
        raise ShadowEpochError("Statut de resolution des labels holdout incoherent.")
    if int(binding["unresolved_cells"]) and not binding[
        "allow_unresolved_final_evaluation_day"
    ]:
        raise ShadowEpochError("Labels holdout absents sans opt-in explicite.")
    if oof_audit.get("holdout_label_binding") != binding or oof_audit.get(
        "holdout_targets_used_for_fit"
    ) is not False:
        raise ShadowEpochError(
            "Le correcteur OOF n'est pas lie au meme holdout non resolu."
        )
    return binding


def _resolve_inputs(
    *,
    run_directory: str | Path,
    residual_corrector_path: str | Path,
    oof_audit_path: str | Path,
    zone: str,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_dir = Path(run_directory).expanduser().resolve()
    corrector_path = Path(residual_corrector_path).expanduser().resolve()
    oof_path = Path(oof_audit_path).expanduser().resolve()
    experiment = verify_bundle(run_dir)
    if not isinstance(experiment, dict):
        experiment = dict(experiment)
    canonical_zone = str(zone).strip().upper()
    validate_experiment_manifest(experiment, zone=canonical_zone)
    schema = _read_json(run_dir / "schema.json", label="schema LoRA")
    window = _declared_holdout(experiment, schema)
    try:
        _validate_oof_chain(
            corrector_path=corrector_path,
            oof_audit_path=oof_path,
            experiment=experiment,
            first_holdout_day=str(window["holdout_start_day"]),
        )
    except Exception as exc:
        raise ShadowEpochError("Correcteur OOF incompatible avec le holdout gele.") from exc
    oof_audit = _read_json(oof_path, label="audit OOF")
    _validated_label_binding(experiment, oof_audit)
    return run_dir, corrector_path, oof_path, experiment, schema, window


def assess_shadow_epoch(
    *,
    run_directory: str | Path,
    residual_corrector_path: str | Path,
    oof_audit_path: str | Path,
    zone: str,
    requested_first_shadow_day: str | date | None = None,
    assessed_at_utc: str | pd.Timestamp | None = None,
    shadow_days_required: int = 30,
) -> ShadowEpochAssessment:
    """Assess continuity and freeze timing without writing any file."""

    if isinstance(shadow_days_required, bool) or shadow_days_required < 2:
        raise ShadowEpochError("shadow_days_required doit etre >= 2.")
    (
        _run_dir,
        _corrector_path,
        oof_path,
        experiment,
        schema,
        window,
    ) = _resolve_inputs(
        run_directory=run_directory,
        residual_corrector_path=residual_corrector_path,
        oof_audit_path=oof_audit_path,
        zone=zone,
    )
    expected_first = date.fromisoformat(str(window["first_shadow_day"]))
    if requested_first_shadow_day is None:
        requested_first = expected_first
    else:
        try:
            requested_first = pd.Timestamp(requested_first_shadow_day).date()
        except Exception as exc:
            raise ShadowEpochError("requested_first_shadow_day invalide.") from exc
    now = pd.to_datetime(
        assessed_at_utc if assessed_at_utc is not None else _utc_now(),
        utc=True,
        errors="coerce",
    )
    if pd.isna(now):
        raise ShadowEpochError("assessed_at_utc invalide.")
    origin = pd.Timestamp(window["first_shadow_origin_utc"])
    continuity = requested_first == expected_first
    freeze_open = pd.Timestamp(now) < origin
    oof_audit = _read_json(oof_path, label="audit OOF")
    label_binding = _validated_label_binding(experiment, oof_audit)
    training_pit = experiment.get("production_pit_evidence") is True
    calibration_pit = oof_audit.get("production_pit_evidence") is True
    blockers: list[str] = []
    if not continuity:
        blockers.append(
            "continuite invalide: le premier shadow doit etre le lendemain exact "
            f"du holdout ({expected_first.isoformat()})"
        )
    if not freeze_open:
        blockers.append(
            "fenetre de gel manquee: l'epoch devait etre scelle avant l'origine "
            f"{origin.isoformat()}"
        )
    if not training_pit:
        blockers.append("production_pit_evidence=false pour le panel train/holdout")
    if not calibration_pit:
        blockers.append("production_pit_evidence=false pour le panel OOF du correcteur")
    shadow_end = expected_first + timedelta(days=int(shadow_days_required) - 1)
    final_sealed = experiment.get("production_pipeline_evidence") is True
    return ShadowEpochAssessment(
        zone=str(zone).strip().upper(),
        timezone=str(schema["timezone"]),
        cutoff_local_time=str(window["cutoff_local_time"]),
        holdout_start_day=str(window["holdout_start_day"]),
        holdout_end_day=str(window["holdout_end_day"]),
        holdout_days=365,
        first_shadow_day=expected_first.isoformat(),
        first_shadow_origin_utc=origin.isoformat(),
        freeze_deadline_utc=origin.isoformat(),
        minimum_shadow_end_day=shadow_end.isoformat(),
        assessed_at_utc=pd.Timestamp(now).isoformat(),
        requested_first_shadow_day=requested_first.isoformat(),
        continuity_ready=continuity,
        freeze_window_open=freeze_open,
        checkpoint_verified=True,
        residual_corrector_verified=True,
        historical_training_pit_ready=training_pit,
        residual_calibration_pit_ready=calibration_pit,
        production_pit_ready=bool(training_pit and calibration_pit),
        final_backtest_already_sealed=final_sealed,
        two_phase_finalisation_required=not final_sealed,
        shadow_days_required=int(shadow_days_required),
        blockers=tuple(blockers),
    )


def freeze_shadow_epoch(
    *,
    run_directory: str | Path,
    residual_corrector_path: str | Path,
    oof_audit_path: str | Path,
    zone: str,
    output_directory: str | Path,
    requested_first_shadow_day: str | date | None = None,
    shadow_days_required: int = 30,
) -> Path:
    """Atomically freeze a self-contained phase-A epoch before first origin.

    The default is intentionally production-strict.  Research candidates with
    false PIT evidence are reported by :func:`assess_shadow_epoch`, but cannot
    be frozen through this production precommitment path.
    """

    assessed = assess_shadow_epoch(
        run_directory=run_directory,
        residual_corrector_path=residual_corrector_path,
        oof_audit_path=oof_audit_path,
        zone=zone,
        requested_first_shadow_day=requested_first_shadow_day,
        shadow_days_required=shadow_days_required,
    )
    if not assessed.ready_to_freeze:
        raise ShadowEpochError(
            "Epoch shadow refuse: " + "; ".join(assessed.blockers)
        )
    (
        run_dir,
        corrector_path,
        oof_path,
        experiment,
        _schema,
        window,
    ) = _resolve_inputs(
        run_directory=run_directory,
        residual_corrector_path=residual_corrector_path,
        oof_audit_path=oof_audit_path,
        zone=zone,
    )
    label_binding = _validated_label_binding(
        experiment, _read_json(oof_path, label="audit OOF")
    )
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists():
        raise ShadowEpochError(
            f"Epoch shadow deja present: {destination}; aucun overwrite n'est permis."
        )
    staging = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=False, exist_ok=False)
        copies = {
            "experiment_manifest_at_freeze": (
                run_dir / "experiment_manifest.json",
                staging / EXPERIMENT_COPY_NAME,
            ),
            "schema": (run_dir / "schema.json", staging / SCHEMA_COPY_NAME),
            "residual_corrector": (corrector_path, staging / CORRECTOR_COPY_NAME),
            "oof_audit": (oof_path, staging / OOF_AUDIT_COPY_NAME),
        }
        for source, target in copies.values():
            shutil.copy2(source, target)
        file_references = {
            name: {
                "relative_path": target.name,
                "sha256": _sha256_file(target),
            }
            for name, (_source, target) in copies.items()
        }
        # Re-read the non-injectable production clock after every potentially
        # slow copy/hash operation.  The initial assessment is not a commit
        # timestamp and must not allow a freeze that crosses the deadline.
        origin = pd.Timestamp(window["first_shadow_origin_utc"])
        commit_checked_at = pd.to_datetime(_utc_now(), utc=True, errors="coerce")
        if pd.isna(commit_checked_at) or pd.Timestamp(commit_checked_at) >= origin:
            raise ShadowEpochError(
                "Epoch shadow refuse: la fenetre de gel s'est fermee avant le commit."
            )
        payload: dict[str, Any] = {
            "schema_version": EPOCH_SCHEMA_VERSION,
            "kind": EPOCH_KIND,
            "freeze_created_at_utc": pd.Timestamp(commit_checked_at).isoformat(),
            "zone": assessed.zone,
            "candidate_model": str(
                experiment.get("experiment_id", experiment.get("model_id", ""))
            ).strip(),
            "checkpoint_sha256": experiment["checkpoint_sha256"],
            "schema_sha256": experiment["schema_sha256"],
            "qualification_holdout": {
                "start_day": assessed.holdout_start_day,
                "end_day": assessed.holdout_end_day,
                "days": 365,
                "first_origin_utc": window["first_origin_utc"],
                "last_origin_utc": window["last_origin_utc"],
            },
            "shadow": {
                "first_day": assessed.first_shadow_day,
                "first_origin_utc": assessed.first_shadow_origin_utc,
                "required_days": assessed.shadow_days_required,
                "minimum_end_day": assessed.minimum_shadow_end_day,
            },
            "phase_contract": {
                "candidate_checkpoint_frozen_before_first_origin": True,
                "residual_corrector_frozen_before_first_origin": True,
                "qualification_window_predeclared_before_first_origin": True,
                "rolling_metrics_may_be_finalized_after_first_origin": True,
                "rolling_window_or_hashes_may_change_after_freeze": False,
                "retrospective_shadow_reconstruction_allowed": False,
                "promotion_eligible_from_epoch_freeze_alone": False,
                "late_labels_may_only_fill_predeclared_nan_cells": True,
            },
            "evaluation_label_binding": label_binding,
            "production_pit_evidence": {
                "training_panel": assessed.historical_training_pit_ready,
                "residual_calibration_panel": assessed.residual_calibration_pit_ready,
                "combined": assessed.production_pit_ready,
            },
            "final_backtest_status_at_freeze": (
                "sealed"
                if assessed.final_backtest_already_sealed
                else "metrics_finalisation_pending"
            ),
            "files": file_references,
            "checkpoint_directory_sha256_verified": (
                sha256_directory(run_dir / "checkpoint")
                == experiment["checkpoint_sha256"]
            ),
            "promotion_eligible": False,
        }
        payload["contract_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        _write_json(staging / EPOCH_MANIFEST_NAME, payload)
        os.replace(staging, destination)
        # A second check after the atomic rename closes the remaining TOCTOU
        # interval.  Since the destination was required not to exist, removing
        # it here cannot destroy a previous epoch.
        committed_at = pd.to_datetime(_utc_now(), utc=True, errors="coerce")
        if pd.isna(committed_at) or pd.Timestamp(committed_at) >= origin:
            shutil.rmtree(destination, ignore_errors=True)
            raise ShadowEpochError(
                "Epoch shadow refuse: le commit atomique a depasse l'origine."
            )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    verify_shadow_epoch(destination, run_directory=run_dir)
    return destination / EPOCH_MANIFEST_NAME


def verify_shadow_epoch(
    epoch_directory: str | Path,
    *,
    run_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a phase-A capsule and, optionally, its live checkpoint identity."""

    directory = Path(epoch_directory).expanduser().resolve()
    payload = _read_json(directory / EPOCH_MANIFEST_NAME, label="manifest epoch")
    if (
        payload.get("schema_version") != EPOCH_SCHEMA_VERSION
        or payload.get("kind") != EPOCH_KIND
    ):
        raise ShadowEpochError("Contrat d'epoch shadow inconnu.")
    contract = _digest(payload.get("contract_sha256"), label="contract_sha256")
    unsigned = dict(payload)
    unsigned.pop("contract_sha256", None)
    if hashlib.sha256(_canonical_bytes(unsigned)).hexdigest() != contract:
        raise ShadowEpochError("contract_sha256 divergent.")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ShadowEpochError("References de fichiers de l'epoch absentes.")
    expected_files = {
        "experiment_manifest_at_freeze": EXPERIMENT_COPY_NAME,
        "schema": SCHEMA_COPY_NAME,
        "residual_corrector": CORRECTOR_COPY_NAME,
        "oof_audit": OOF_AUDIT_COPY_NAME,
    }
    resolved: dict[str, Path] = {}
    for name, expected_name in expected_files.items():
        reference = files.get(name)
        if not isinstance(reference, Mapping):
            raise ShadowEpochError(f"Reference {name} absente.")
        relative = reference.get("relative_path")
        if relative != expected_name:
            raise ShadowEpochError(f"Chemin {name} non canonique.")
        path = (directory / expected_name).resolve()
        if path.parent != directory or not path.is_file():
            raise ShadowEpochError(f"Copie {name} absente ou hors epoch.")
        if _sha256_file(path) != _digest(reference.get("sha256"), label=f"{name}.sha256"):
            raise ShadowEpochError(f"SHA de {name} divergent.")
        resolved[name] = path
    experiment = _read_json(
        resolved["experiment_manifest_at_freeze"], label="manifeste experience gele"
    )
    schema = _read_json(resolved["schema"], label="schema gele")
    zone = str(payload.get("zone", "")).strip().upper()
    validate_experiment_manifest(experiment, zone=zone)
    window = _declared_holdout(experiment, schema)
    qualification = payload.get("qualification_holdout")
    shadow = payload.get("shadow")
    expected_qualification = {
        "start_day": window["holdout_start_day"],
        "end_day": window["holdout_end_day"],
        "days": 365,
        "first_origin_utc": window["first_origin_utc"],
        "last_origin_utc": window["last_origin_utc"],
    }
    if qualification != expected_qualification or not isinstance(shadow, Mapping):
        raise ShadowEpochError("Fenetre qualification/shadow divergente du manifeste gele.")
    expected_shadow = {
        "first_day": window["first_shadow_day"],
        "first_origin_utc": window["first_shadow_origin_utc"],
        "required_days": int(shadow.get("required_days", 0)),
        "minimum_end_day": (
            date.fromisoformat(str(window["first_shadow_day"]))
            + timedelta(days=int(shadow.get("required_days", 0)) - 1)
        ).isoformat()
        if isinstance(shadow.get("required_days"), int)
        and not isinstance(shadow.get("required_days"), bool)
        and int(shadow.get("required_days")) >= 2
        else "invalid",
    }
    if dict(shadow) != expected_shadow:
        raise ShadowEpochError("Dates shadow divergentes du holdout gele.")
    created = pd.to_datetime(payload.get("freeze_created_at_utc"), utc=True, errors="coerce")
    origin = pd.Timestamp(window["first_shadow_origin_utc"])
    if pd.isna(created) or pd.Timestamp(created) >= origin:
        raise ShadowEpochError("L'epoch n'a pas ete gelee avant la premiere origine.")
    try:
        _validate_oof_chain(
            corrector_path=resolved["residual_corrector"],
            oof_audit_path=resolved["oof_audit"],
            experiment=experiment,
            first_holdout_day=str(window["holdout_start_day"]),
        )
    except Exception as exc:
        raise ShadowEpochError("Chaine OOF gelee invalide.") from exc
    pit = payload.get("production_pit_evidence")
    oof = _read_json(resolved["oof_audit"], label="audit OOF gele")
    expected_binding = _validated_label_binding(experiment, oof)
    if payload.get("evaluation_label_binding") != expected_binding:
        raise ShadowEpochError("Binding des labels holdout divergent de l'epoch.")
    expected_pit = {
        "training_panel": experiment.get("production_pit_evidence") is True,
        "residual_calibration_panel": oof.get("production_pit_evidence") is True,
        "combined": bool(
            experiment.get("production_pit_evidence") is True
            and oof.get("production_pit_evidence") is True
        ),
    }
    if pit != expected_pit or expected_pit["combined"] is not True:
        raise ShadowEpochError("Preuve PIT production de l'epoch incomplete.")
    phase = payload.get("phase_contract")
    expected_phase = {
        "candidate_checkpoint_frozen_before_first_origin": True,
        "residual_corrector_frozen_before_first_origin": True,
        "qualification_window_predeclared_before_first_origin": True,
        "rolling_metrics_may_be_finalized_after_first_origin": True,
        "rolling_window_or_hashes_may_change_after_freeze": False,
        "retrospective_shadow_reconstruction_allowed": False,
        "promotion_eligible_from_epoch_freeze_alone": False,
        "late_labels_may_only_fill_predeclared_nan_cells": True,
    }
    if phase != expected_phase or payload.get("promotion_eligible") is not False:
        raise ShadowEpochError("Contrat de phase de l'epoch invalide.")
    if payload.get("checkpoint_sha256") != experiment.get("checkpoint_sha256") or payload.get(
        "schema_sha256"
    ) != experiment.get("schema_sha256"):
        raise ShadowEpochError("Identite checkpoint/schema de l'epoch divergente.")
    if run_directory is not None:
        live = verify_bundle(Path(run_directory).expanduser().resolve())
        if live.get("checkpoint_sha256") != payload["checkpoint_sha256"] or live.get(
            "schema_sha256"
        ) != payload["schema_sha256"]:
            raise ShadowEpochError("Le bundle courant differe de l'epoch gelee.")
        if live.get("splits", {}).get("evaluation_holdout") != experiment.get(
            "splits", {}
        ).get("evaluation_holdout"):
            raise ShadowEpochError("Le holdout courant differe de l'epoch gelee.")
    return payload


def validate_finalisation_against_epoch(
    epoch_directory: str | Path,
    *,
    run_directory: str | Path,
) -> dict[str, Any]:
    """Bind a later FinalBacktest to the immutable phase-A precommitment.

    This is the phase-B check.  It deliberately does not evaluate performance
    gates and therefore still returns ``promotion_eligible=False``.  Its sole
    job is proving that late metric finalisation did not swap the candidate,
    corrector, schema or predeclared rolling window.
    """

    directory = Path(epoch_directory).expanduser().resolve()
    epoch = verify_shadow_epoch(directory, run_directory=run_directory)
    run_dir = Path(run_directory).expanduser().resolve()
    current = verify_bundle(run_dir)
    frozen_binding = epoch.get("evaluation_label_binding")
    if current.get("evaluation_label_binding") != frozen_binding:
        raise ShadowEpochError(
            "Phase B refusee: binding des labels courant different du gel phase A."
        )
    resolution: Mapping[str, Any] | None = None
    if isinstance(frozen_binding, Mapping) and int(
        frozen_binding.get("unresolved_cells", 0)
    ):
        raw_resolution = current.get("evaluation_label_resolution")
        if not isinstance(raw_resolution, Mapping):
            raise ShadowEpochError(
                "Phase B incomplete: les labels holdout predeclares n'ont pas ete "
                "lies a un panel resolu."
            )
        resolution = raw_resolution
        expected_resolution_values = {
            "frozen_panel_sha256": _read_json(
                directory / EXPERIMENT_COPY_NAME,
                label="manifeste experience gele",
            ).get("panel_sha256"),
            "frozen_panel_audit_sha256": _read_json(
                directory / EXPERIMENT_COPY_NAME,
                label="manifeste experience gele",
            ).get("panel_audit_sha256"),
            "input_contract_sha256": frozen_binding.get("input_contract_sha256"),
            "resolved_cells": frozen_binding.get("unresolved_cells"),
            "all_other_values_identical": True,
            "labels_used_for_fit": False,
            "promotion_eligible": False,
        }
        if any(
            resolution.get(key) != value
            for key, value in expected_resolution_values.items()
        ):
            raise ShadowEpochError(
                "Phase B refusee: resolution des labels differente du pre-engagement."
            )
        for path_key, sha_key, label in (
            ("resolved_panel_path", "resolved_panel_sha256", "panel resolu"),
            (
                "resolved_panel_audit_path",
                "resolved_panel_audit_sha256",
                "sidecar du panel resolu",
            ),
        ):
            path_value = resolution.get(path_key)
            if not isinstance(path_value, str) or not path_value.strip():
                raise ShadowEpochError(f"Phase B: chemin {label} absent.")
            resolved_path = Path(path_value).expanduser().resolve()
            if not resolved_path.is_file() or _sha256_file(resolved_path) != _digest(
                resolution.get(sha_key), label=sha_key
            ):
                raise ShadowEpochError(f"Phase B: {label} absent ou divergent.")
    if current.get("production_pipeline_evidence") is not True or current.get(
        "candidate_output_stage"
    ) != "exogenous_residual_corrected":
        raise ShadowEpochError(
            "Phase B incomplete: FinalBacktest du pipeline corrige non scelle."
        )
    files = epoch["files"]
    corrector_sha = files["residual_corrector"]["sha256"]
    oof_sha = files["oof_audit"]["sha256"]
    if current.get("residual_corrector_sha256") != corrector_sha or current.get(
        "oof_training_audit_sha256"
    ) != oof_sha:
        raise ShadowEpochError(
            "Phase B refusee: correcteur/OOF different du pre-engagement."
        )
    evidence = current.get("evaluation_evidence")
    if not isinstance(evidence, Mapping):
        raise ShadowEpochError("Phase B: evaluation_evidence finale absente.")
    relative_text = evidence.get("relative_path")
    if not isinstance(relative_text, str) or not relative_text.strip():
        raise ShadowEpochError("Phase B: chemin de preuve finale absent.")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ShadowEpochError("Phase B: chemin de preuve finale non sur.")
    expected_evidence_relative = Path("final_pipeline") / FINAL_EVIDENCE_NAME
    if relative != expected_evidence_relative:
        raise ShadowEpochError("Phase B: chemin de preuve finale non canonique.")
    evidence_path = (run_dir / relative).resolve()
    if run_dir not in evidence_path.parents or not evidence_path.is_file():
        raise ShadowEpochError("Phase B: preuve finale absente/hors bundle.")
    if _sha256_file(evidence_path) != _digest(
        evidence.get("sha256"), label="evaluation_evidence.sha256"
    ):
        raise ShadowEpochError("Phase B: SHA de la preuve finale divergent.")
    try:
        _validate_final_pipeline_evidence(current, rolling_path=evidence_path)
    except Exception as exc:
        raise ShadowEpochError(
            "Phase B: contrat du pipeline final courant invalide."
        ) from exc
    qualification = epoch["qualification_holdout"]
    frozen_schema = _read_json(directory / SCHEMA_COPY_NAME, label="schema gele")
    timezone_name = str(frozen_schema["timezone"])
    start_local = pd.Timestamp(str(qualification["start_day"])).tz_localize(
        timezone_name
    )
    end_exclusive_local = (
        pd.Timestamp(str(qualification["end_day"])) + pd.Timedelta(days=1)
    ).tz_localize(timezone_name)
    expected_rows = int(
        (
            end_exclusive_local.tz_convert("UTC")
            - start_local.tz_convert("UTC")
        )
        / pd.Timedelta(hours=1)
    )
    if evidence.get("physical_days") != 365 or evidence.get("rows") != expected_rows:
        raise ShadowEpochError(
            "Phase B: preuve finale differente des 365 jours physiques predeclares "
            f"({expected_rows} heures attendues)."
        )
    try:
        frame = _normalise_evidence(
            pd.read_csv(evidence_path), label="preuve finale phase B"
        )
    except Exception as exc:
        raise ShadowEpochError("Phase B: preuve finale illisible ou invalide.") from exc
    delivery = pd.DatetimeIndex(frame["delivery_start_utc"])
    origins = pd.DatetimeIndex(frame["forecast_origin_utc"])
    if delivery.isna().any() or origins.isna().any() or len(frame) != expected_rows:
        raise ShadowEpochError("Phase B: timeline finale invalide.")
    ordered = frame.sort_values("delivery_start_utc", kind="stable")
    timeline = pd.DatetimeIndex(ordered["delivery_start_utc"])
    if timeline.duplicated().any() or not all(
        delta == pd.Timedelta(hours=1) for delta in timeline[1:] - timeline[:-1]
    ):
        raise ShadowEpochError("Phase B: timeline UTC finale non continue.")
    local_days = pd.Index(timeline.tz_convert(timezone_name).date)
    actual_start = min(local_days).isoformat()
    actual_end = max(local_days).isoformat()
    if actual_start != qualification["start_day"] or actual_end != qualification["end_day"]:
        raise ShadowEpochError(
            "Phase B: fenetre finale differente du holdout predeclare."
        )
    cutoff_hour, cutoff_minute, cutoff = _parse_cutoff(
        frozen_schema["cutoff_local_time"]
    )
    for day_value in sorted(set(local_days)):
        mask = np.asarray(local_days == day_value)
        observed_delivery = timeline[mask]
        day = date.fromisoformat(day_value.isoformat())
        day_start = pd.Timestamp(day).tz_localize(timezone_name)
        day_end = pd.Timestamp(day + timedelta(days=1)).tz_localize(timezone_name)
        expected_delivery = pd.date_range(
            day_start.tz_convert("UTC"),
            day_end.tz_convert("UTC"),
            freq="h",
            inclusive="left",
        )
        if not observed_delivery.equals(expected_delivery):
            raise ShadowEpochError(
                f"Phase B: grille physique/DST invalide pour {day.isoformat()}."
            )
        day_origins = pd.DatetimeIndex(ordered.loc[mask, "forecast_origin_utc"]).unique()
        if len(day_origins) != 1:
            raise ShadowEpochError(
                f"Phase B: origine non unique pour {day.isoformat()}."
            )
        expected_origin = _local_cutoff(
            day - timedelta(days=1),
            timezone_name,
            hour=cutoff_hour,
            minute=cutoff_minute,
        ).tz_convert("UTC")
        if pd.Timestamp(day_origins[0]) != expected_origin:
            raise ShadowEpochError(
                f"Phase B: origine de {day.isoformat()} differente de D-1 {cutoff}."
            )
    if evidence.get("first_delivery_utc") not in (None, timeline[0].isoformat()) or evidence.get(
        "last_delivery_utc"
    ) not in (None, timeline[-1].isoformat()):
        raise ShadowEpochError("Phase B: bornes de preuve finale divergentes.")
    final_reference = current.get("final_pipeline_evaluation")
    if not isinstance(final_reference, Mapping):
        raise ShadowEpochError("Phase B: manifeste FinalBacktest absent.")
    final_relative = final_reference.get("relative_path")
    if not isinstance(final_relative, str) or not final_relative.strip():
        raise ShadowEpochError("Phase B: chemin du manifeste FinalBacktest absent.")
    final_relative_path = Path(final_relative)
    expected_manifest_relative = Path("final_pipeline") / FINAL_MANIFEST_NAME
    if (
        final_relative_path.is_absolute()
        or ".." in final_relative_path.parts
        or final_relative_path != expected_manifest_relative
    ):
        raise ShadowEpochError("Phase B: chemin du manifeste FinalBacktest non canonique.")
    final_path = (run_dir / final_relative_path).resolve()
    if (
        run_dir not in final_path.parents
        or not final_path.is_file()
        or _sha256_file(final_path)
        != _digest(final_reference.get("sha256"), label="final_pipeline_evaluation.sha256")
    ):
        raise ShadowEpochError("Phase B: manifeste FinalBacktest absent ou divergent.")
    final_manifest = _read_json(final_path, label="manifeste FinalBacktest")
    final_window = final_manifest.get("window")
    final_artifacts = final_manifest.get("artifacts")
    final_predictions = (
        final_artifacts.get("predictions")
        if isinstance(final_artifacts, Mapping)
        else None
    )
    if (
        final_manifest.get("schema_version") != 1
        or final_manifest.get("kind")
        != "chronos2_exogenous_final_pipeline_evaluation"
        or final_manifest.get("zone") != epoch["zone"]
        or final_manifest.get("experiment_id") != epoch["candidate_model"]
        or final_manifest.get("production_pipeline_evidence") is not True
        or not isinstance(final_window, Mapping)
        or final_window.get("physical_days") != 365
        or final_window.get("physical_hours") != expected_rows
        or final_window.get("first_delivery_utc") != timeline[0].isoformat()
        or final_window.get("last_delivery_utc") != timeline[-1].isoformat()
        or not isinstance(final_predictions, Mapping)
        or final_predictions.get("relative_path") != FINAL_EVIDENCE_NAME
        or final_predictions.get("sha256") != evidence.get("sha256")
    ):
        raise ShadowEpochError(
            "Phase B: contenu du manifeste FinalBacktest non canonique ou divergent."
        )
    return {
        "schema_version": 1,
        "kind": "chronos2_exogenous_shadow_epoch_phase_b_validation",
        "zone": epoch["zone"],
        "shadow_epoch_contract_sha256": epoch["contract_sha256"],
        "checkpoint_sha256": epoch["checkpoint_sha256"],
        "residual_corrector_sha256": corrector_sha,
        "oof_training_audit_sha256": oof_sha,
        "holdout_start_day": qualification["start_day"],
        "holdout_end_day": qualification["end_day"],
        "first_shadow_day": epoch["shadow"]["first_day"],
        "final_backtest_sha256": final_reference["sha256"],
        "evaluation_label_resolution": dict(resolution) if resolution else None,
        "ready_for_final_shadow_scoring": True,
        "promotion_eligible": False,
    }


def validate_shadow_delivery_against_epoch(
    epoch_directory: str | Path,
    *,
    run_directory: str | Path,
    journal_path: str | Path,
    delivery_day: str | date,
) -> dict[str, Any]:
    """Preflight one shadow day against the frozen boundary and journal.

    A missing first day cannot be repaired retrospectively.  Before the costly
    panel materialisation this check therefore accepts only the frozen first
    day, an already-issued day (idempotence/actual resolution), or the exact
    calendar day following the latest issued forecast.
    """

    epoch = verify_shadow_epoch(epoch_directory, run_directory=run_directory)
    raw_day = delivery_day.isoformat() if isinstance(delivery_day, date) else str(delivery_day)
    try:
        requested = date.fromisoformat(raw_day)
    except ValueError as exc:
        raise ShadowEpochError("delivery_day doit respecter YYYY-MM-DD.") from exc
    if raw_day != requested.isoformat():
        raise ShadowEpochError("delivery_day doit respecter exactement YYYY-MM-DD.")
    first = date.fromisoformat(str(epoch["shadow"]["first_day"]))
    journal = Path(journal_path).expanduser().resolve()
    if not journal.exists():
        if requested != first:
            raise ShadowEpochError(
                "Premier shadow non conforme au pre-engagement: attendu "
                f"{first.isoformat()}, recu {requested.isoformat()}."
            )
        return {
            "schema_version": 1,
            "kind": "chronos2_exogenous_shadow_epoch_delivery_preflight",
            "delivery_day": requested.isoformat(),
            "status": "first_forecast",
            "first_shadow_day": first.isoformat(),
            "next_new_delivery_day": first.isoformat(),
            "promotion_eligible": False,
        }
    if not journal.is_file():
        raise ShadowEpochError(f"Journal shadow invalide: {journal}.")
    required = (
        "delivery_start_utc",
        "forecast_origin_utc",
        "record_kind",
        "checkpoint_sha256",
    )
    try:
        history = pd.read_csv(journal, usecols=list(required))
    except Exception as exc:
        raise ShadowEpochError("Journal shadow illisible ou de schema inconnu.") from exc
    forecasts = history.loc[history["record_kind"].eq("forecast")].copy()
    if forecasts.empty:
        raise ShadowEpochError("Journal shadow existant sans emission forecast.")
    delivery = pd.to_datetime(
        forecasts["delivery_start_utc"], utc=True, errors="coerce"
    )
    origins = pd.to_datetime(
        forecasts["forecast_origin_utc"], utc=True, errors="coerce"
    )
    if delivery.isna().any() or origins.isna().any():
        raise ShadowEpochError("Journal shadow: timestamps forecast invalides.")
    if set(forecasts["checkpoint_sha256"].astype(str)) != {
        str(epoch["checkpoint_sha256"])
    }:
        raise ShadowEpochError("Journal shadow: checkpoint different de l'epoch.")
    frozen_schema = _read_json(
        Path(epoch_directory).expanduser().resolve() / SCHEMA_COPY_NAME,
        label="schema gele",
    )
    timezone_name = str(frozen_schema["timezone"])
    cutoff_hour, cutoff_minute, cutoff = _parse_cutoff(
        frozen_schema["cutoff_local_time"]
    )
    forecasts["delivery_start_utc"] = delivery
    forecasts["forecast_origin_utc"] = origins
    forecasts["delivery_day"] = delivery.dt.tz_convert(timezone_name).dt.date
    issued_days = sorted(set(forecasts["delivery_day"]))
    expected_days = [first + timedelta(days=index) for index in range(len(issued_days))]
    if issued_days != expected_days:
        raise ShadowEpochError(
            "Journal shadow: les emissions ne forment pas une suite continue depuis "
            f"{first.isoformat()}."
        )
    for issued_day in issued_days:
        day_rows = forecasts.loc[forecasts["delivery_day"].eq(issued_day)]
        day_delivery = pd.DatetimeIndex(day_rows["delivery_start_utc"]).sort_values()
        if day_delivery.duplicated().any():
            raise ShadowEpochError(
                f"Journal shadow: heures forecast dupliquees le {issued_day}."
            )
        start = pd.Timestamp(issued_day).tz_localize(timezone_name)
        end = pd.Timestamp(issued_day + timedelta(days=1)).tz_localize(timezone_name)
        expected_delivery = pd.date_range(
            start.tz_convert("UTC"),
            end.tz_convert("UTC"),
            freq="h",
            inclusive="left",
        )
        if not day_delivery.equals(expected_delivery):
            raise ShadowEpochError(
                f"Journal shadow: grille physique/DST incomplete le {issued_day}."
            )
        day_origins = pd.DatetimeIndex(day_rows["forecast_origin_utc"]).unique()
        expected_origin = _local_cutoff(
            issued_day - timedelta(days=1),
            timezone_name,
            hour=cutoff_hour,
            minute=cutoff_minute,
        ).tz_convert("UTC")
        if len(day_origins) != 1 or pd.Timestamp(day_origins[0]) != expected_origin:
            raise ShadowEpochError(
                f"Journal shadow: origine differente de D-1 {cutoff} le {issued_day}."
            )
    next_day = issued_days[-1] + timedelta(days=1)
    if requested in issued_days:
        status = "existing_day_resolution_or_idempotent_replay"
    elif requested == next_day:
        status = "next_forecast"
    else:
        raise ShadowEpochError(
            "Livraison shadow non continue: jours admis="
            f"{issued_days[0].isoformat()}..{issued_days[-1].isoformat()} ou "
            f"{next_day.isoformat()}, recu={requested.isoformat()}."
        )
    return {
        "schema_version": 1,
        "kind": "chronos2_exogenous_shadow_epoch_delivery_preflight",
        "delivery_day": requested.isoformat(),
        "status": status,
        "first_shadow_day": first.isoformat(),
        "last_issued_delivery_day": issued_days[-1].isoformat(),
        "next_new_delivery_day": next_day.isoformat(),
        "promotion_eligible": False,
    }


def earliest_new_epoch_plan(
    *,
    timezone_name: str,
    cutoff_local_time: str = "08:00",
    assessed_at_utc: str | pd.Timestamp | None = None,
    shadow_days_required: int = 30,
) -> dict[str, Any]:
    """Return concrete dates for the next *new/rebaselined* candidate epoch.

    This helper does not claim that the necessary candidate, OOF corrector or
    source captures exist.  It only computes the earliest still-open temporal
    boundary and the exact window a new bundle would have to seal.
    """

    if isinstance(shadow_days_required, bool) or shadow_days_required < 2:
        raise ShadowEpochError("shadow_days_required doit etre >= 2.")
    try:
        zone = ZoneInfo(str(timezone_name))
    except Exception as exc:
        raise ShadowEpochError("Timezone invalide.") from exc
    hour, minute, cutoff = _parse_cutoff(cutoff_local_time)
    now = pd.to_datetime(
        assessed_at_utc if assessed_at_utc is not None else _utc_now(),
        utc=True,
        errors="coerce",
    )
    if pd.isna(now):
        raise ShadowEpochError("assessed_at_utc invalide.")
    local_now = pd.Timestamp(now).tz_convert(zone)
    next_origin_day = local_now.date()
    cutoff_today = _local_cutoff(
        next_origin_day,
        zone,
        hour=hour,
        minute=minute,
    )
    # A precommitment must be strictly earlier than the origin.  Once today's
    # cutoff is reached, the next usable origin is tomorrow.
    if local_now >= cutoff_today:
        next_origin_day += timedelta(days=1)
    first_shadow = next_origin_day + timedelta(days=1)
    holdout_end = first_shadow - timedelta(days=1)
    holdout_start = holdout_end - timedelta(days=364)
    first_origin = _local_cutoff(
        next_origin_day,
        zone,
        hour=hour,
        minute=minute,
    )
    shadow_end = first_shadow + timedelta(days=int(shadow_days_required) - 1)
    return {
        "schema_version": 1,
        "kind": "chronos2_exogenous_earliest_new_shadow_epoch_plan",
        "assessed_at_utc": pd.Timestamp(now).isoformat(),
        "timezone": str(timezone_name),
        "cutoff_local_time": cutoff,
        "new_or_rebaselined_bundle_required": True,
        "holdout_start_day": holdout_start.isoformat(),
        "holdout_end_day": holdout_end.isoformat(),
        "holdout_days": 365,
        "freeze_must_complete_before_utc": first_origin.tz_convert("UTC").isoformat(),
        "first_shadow_day": first_shadow.isoformat(),
        "minimum_shadow_end_day": shadow_end.isoformat(),
        "earliest_governance_day": (shadow_end + timedelta(days=1)).isoformat(),
        "rolling_metrics_may_finalize_after_first_shadow_origin": True,
        "model_corrector_and_window_must_not_change_after_freeze": True,
        "promotion_eligible": False,
    }


__all__ = [
    "EPOCH_KIND",
    "EPOCH_MANIFEST_NAME",
    "EPOCH_SCHEMA_VERSION",
    "ShadowEpochAssessment",
    "ShadowEpochError",
    "assess_shadow_epoch",
    "earliest_new_epoch_plan",
    "freeze_shadow_epoch",
    "validate_finalisation_against_epoch",
    "validate_shadow_delivery_against_epoch",
    "verify_shadow_epoch",
]
