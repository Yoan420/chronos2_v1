"""Reproducible, read-only comparison of two final LoRA pipeline proofs.

The comparison is deliberately separated from promotion and activation.  It
prefers the sealed output of :mod:`chronos2_exogenous.final_pipeline`; when
that proof cannot yet exist it can compare two sealed raw evaluations while
marking the result research-only.  In both cases it rebuilds paired
rank-8-versus-rank-16 evidence and applies the existing rolling-365 governance
thresholds with rank 8 as the incumbent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from .evaluation import compute_metrics
from .final_pipeline import (
    ACTUAL_PAIRING_ATOL_EUR_MWH,
    FINAL_AUDIT_NAME,
    FINAL_EVIDENCE_NAME,
    FINAL_MANIFEST_NAME,
    FINAL_METRICS_NAME,
    FINAL_REPORT_NAME,
    INCUMBENT_COLUMNS,
    INCUMBENT_MODEL,
    _validate_oof_chain,
)
from .governance import (
    GovernancePolicy,
    _evaluate_gate,
    _normalise_window,
    _window_metrics,
    load_policy,
    validate_experiment_manifest,
)
from .lora_finetune import EVALUATION_COLUMNS
from .production import (
    BASE_MODEL,
    OUTPUT_MODEL,
    RUNTIME_SCHEMA_VERSION,
    _apply_residual_corrector,
    _validate_final_pipeline_evidence,
)


COMPARISON_SCHEMA_VERSION = 1
COMPARISON_JSON_NAME = "rank8_vs_rank16_comparison.json"
COMPARISON_REPORT_NAME = "rank8_vs_rank16_report.html"
COMPARISON_DAILY_NAME = "rank8_vs_rank16_daily.csv"
FINAL_KIND = "chronos2_exogenous_final_pipeline_evaluation"
RAW_KIND = "chronos2_exogenous_lora_rolling365_evaluation"
RAW_MANIFEST_NAME = "evaluation_manifest.json"
RAW_EVIDENCE_NAME = "evaluation_predictions.csv.gz"
RAW_METRICS_NAME = "evaluation_metrics.json"
RAW_REPORT_NAME = "evaluation_report.html"
RAW_DAILY_NAME = "evaluation_daily.csv.gz"


class CandidateComparisonError(RuntimeError):
    """Raised before publication when the two proofs are not comparable."""


@dataclass(frozen=True)
class CandidateProof:
    """One verified raw or final LoRA proof and its immutable identity."""

    rank: int
    evidence_stage: str
    zone: str
    experiment_id: str
    run_directory: Path
    final_directory: Path
    manifest_path: Path
    predictions_path: Path
    metrics_path: Path
    manifest_sha256: str
    predictions_sha256: str
    metrics_sha256: str
    frame: pd.DataFrame
    declared_metrics: Mapping[str, Any]
    input_contract_sha256: str | None = None
    operational_selection_blockers: tuple[str, ...] = ()
    operational_lineage_hashes: tuple[tuple[Path, str], ...] = ()


@dataclass(frozen=True)
class CandidateComparisonArtifacts:
    """Atomically published comparison artifacts."""

    output_directory: Path
    comparison_path: Path
    report_path: Path
    daily_path: Path
    decision: str
    winner: str
    comparison: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CandidateComparisonError(f"{label} absent: {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateComparisonError(f"{label} illisible: {path}.") from exc
    if not isinstance(payload, dict):
        raise CandidateComparisonError(f"{label} doit etre un objet JSON.")
    return payload


def _json_text(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _resolve_final_directory(source: str | Path) -> Path:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.name not in {
            FINAL_MANIFEST_NAME,
            FINAL_EVIDENCE_NAME,
            FINAL_METRICS_NAME,
        }:
            raise CandidateComparisonError(
                "Entree candidat invalide; fournissez le run, final_pipeline, "
                f"{FINAL_MANIFEST_NAME}, {FINAL_EVIDENCE_NAME} ou "
                f"{FINAL_METRICS_NAME}: {path}."
            )
        return path.parent
    if not path.is_dir():
        raise CandidateComparisonError(f"Entree candidat absente: {path}.")
    if (path / FINAL_MANIFEST_NAME).is_file():
        return path
    nested = path / "final_pipeline"
    if (nested / FINAL_MANIFEST_NAME).is_file():
        return nested
    raise CandidateComparisonError(
        f"Aucune preuve finale scellee trouvee sous {path}."
    )


def _source_stage(source: str | Path) -> str:
    """Detect an explicitly selected proof stage; prefer final for a run dir."""

    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.name in {FINAL_MANIFEST_NAME, FINAL_EVIDENCE_NAME, FINAL_METRICS_NAME}:
            return "final_corrected"
        if path.name in {RAW_MANIFEST_NAME, RAW_EVIDENCE_NAME, RAW_METRICS_NAME}:
            return "raw_lora"
        raise CandidateComparisonError(f"Artefact candidat non reconnu: {path}.")
    if not path.is_dir():
        raise CandidateComparisonError(f"Entree candidat absente: {path}.")
    if (path / FINAL_MANIFEST_NAME).is_file() or (
        path / "final_pipeline" / FINAL_MANIFEST_NAME
    ).is_file():
        return "final_corrected"
    if (path / RAW_MANIFEST_NAME).is_file():
        return "raw_lora"
    raise CandidateComparisonError(f"Aucune preuve LoRA reconnue sous {path}.")


def _safe_artifact_path(final_directory: Path, relative_value: object) -> Path:
    if not isinstance(relative_value, str) or not relative_value.strip():
        raise CandidateComparisonError("Chemin relatif d'artefact absent.")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CandidateComparisonError(
            f"Chemin d'artefact hors preuve interdit: {relative_value!r}."
        )
    resolved = (final_directory / relative).resolve()
    if resolved != final_directory and final_directory not in resolved.parents:
        raise CandidateComparisonError(
            f"Chemin d'artefact hors preuve interdit: {relative_value!r}."
        )
    return resolved


def _artifact_from_manifest(
    manifest: Mapping[str, Any],
    *,
    final_directory: Path,
    name: str,
    expected_name: str,
) -> tuple[Path, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CandidateComparisonError("Manifeste final: artifacts absent.")
    reference = artifacts.get(name)
    if not isinstance(reference, Mapping):
        raise CandidateComparisonError(f"Manifeste final: artefact {name} absent.")
    path = _safe_artifact_path(final_directory, reference.get("relative_path"))
    if path.name != expected_name:
        raise CandidateComparisonError(
            f"Manifeste final: {name} doit pointer vers {expected_name}."
        )
    expected_sha = reference.get("sha256")
    if (
        not _is_sha256(expected_sha)
        or not path.is_file()
        or _sha256(path) != expected_sha
    ):
        raise CandidateComparisonError(
            f"Manifeste final: SHA-256 divergent ou artefact absent pour {name}."
        )
    return path, expected_sha


def _numeric_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return left == right


def _verify_declared_metrics(
    declared: Mapping[str, Any], recomputed: Mapping[str, Any], *, rank: int
) -> None:
    for key, expected in recomputed.items():
        # The final-pipeline publisher intentionally relabels these three raw
        # evaluator descriptors after applying the residual corrector.
        if key in {"comparison_scope", "baseline_label", "candidate_label"}:
            continue
        if key not in declared or not _numeric_equal(declared[key], expected):
            raise CandidateComparisonError(
                f"Rang {rank}: metrique publiee {key} absente ou divergente."
            )


def _resolve_audit_source(
    value: object, *, run_directory: Path, label: str
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CandidateComparisonError(f"{label}: path absent.")
    source = Path(value).expanduser()
    if not source.is_absolute():
        source = run_directory / source
    return source.resolve()


def _operational_selection_blockers(
    *,
    rank: int,
    run_directory: Path,
    final_manifest: Mapping[str, Any],
    experiment: Mapping[str, Any],
    audit_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    frame: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[tuple[Path, str], ...]]:
    """Return every reason a final proof cannot support operational selection.

    Core artifact integrity is checked before this function.  The checks here
    deliberately downgrade an otherwise readable final comparison to a
    research preference when PIT or the canonical OOF/incumbent lineage cannot
    be independently reproduced from the sealed manifests and audit.
    """

    blockers: list[str] = []
    lineage_hashes: dict[Path, str] = {audit_path: _sha256(audit_path)}

    def block(message: str) -> None:
        if message not in blockers:
            blockers.append(message)

    if experiment.get("production_pit_evidence") is not True:
        block("production_pit_evidence n'est pas true")
    try:
        validate_experiment_manifest(experiment, zone=str(experiment.get("zone", "")))
    except Exception as exc:
        block(f"contrat causal/freeze du manifeste invalide ({exc})")

    expected_detail: Mapping[str, object] = {
        "comparison_scope": "paired_operational_final_pipelines",
        "baseline_output_stage": INCUMBENT_MODEL,
        "candidate_output_stage": OUTPUT_MODEL,
        "paired_same_input_contract": True,
        "paired_same_evaluation_window": True,
        "baseline_residual_corrector_applied": True,
        "candidate_residual_corrector_applied": True,
        "rolling_evaluation_days": 365,
        "promotion_eligible": True,
        "paired_delivery_hours": len(frame),
        "paired_forecast_origins": True,
        "paired_actuals": True,
    }
    details: list[tuple[str, Mapping[str, Any]]] = []
    for label, owner in (
        ("manifeste final", final_manifest),
        ("manifeste d'experience", experiment),
    ):
        detail = owner.get("production_pipeline_evidence_detail")
        if not isinstance(detail, Mapping):
            block(f"{label}: production_pipeline_evidence_detail absent")
        else:
            details.append((label, detail))
            mismatches = [
                key
                for key, expected in expected_detail.items()
                if type(detail.get(key)) is not type(expected)
                or detail.get(key) != expected
            ]
            if mismatches:
                block(f"{label}: chaine finale non canonique ({', '.join(mismatches)})")
    if len(details) == 2 and dict(details[0][1]) != dict(details[1][1]):
        block("detail du pipeline final divergent entre les manifestes")

    if (
        final_manifest.get("comparison_scope")
        != "paired_operational_final_pipelines"
        or final_manifest.get("baseline_output_stage") != INCUMBENT_MODEL
    ):
        block("manifeste final: scope ou incumbent non canonique")
    runtime = experiment.get("production_runtime")
    expected_runtime = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "layout": "per_zone",
        "cross_learning": False,
        "target": "target",
    }
    if not isinstance(runtime, Mapping) or dict(runtime) != expected_runtime:
        block("production_runtime per_zone canonique absent")

    try:
        _validate_final_pipeline_evidence(experiment, rolling_path=predictions_path)
    except Exception as exc:
        block(f"preuve pipeline finale non reproductible ({exc})")

    try:
        audit = _read_json(audit_path, label=f"audit final rang {rank}")
    except CandidateComparisonError as exc:
        block(str(exc))
        return tuple(blockers), tuple(sorted(lineage_hashes.items(), key=lambda item: str(item[0])))
    if (
        audit.get("schema_version") != 1
        or audit.get("purpose") != "chronos2_exogenous_final_pipeline_rolling365"
        or audit.get("zone") != experiment.get("zone")
    ):
        block("audit final: identite/purpose non canonique")
    if audit.get("production_pit_evidence_value") is not True:
        block("audit final: preuve PIT production absente")
    audit_detail = audit.get("production_pipeline_evidence_detail")
    if not isinstance(audit_detail, Mapping) or (
        details and dict(audit_detail) != dict(details[0][1])
    ):
        block("audit final: detail du pipeline divergent")
    required_checks: Mapping[str, object] = {
        "bundle_verified": True,
        "exact_365_physical_days": True,
        "continuous_hourly_utc": True,
        "dst_days_verified": True,
        "same_delivery_timeline": True,
        "same_forecast_origins": True,
        "same_actuals": True,
        "corrector_strictly_pre_holdout_oof": True,
        "production_pit_evidence_unchanged": True,
    }
    checks = audit.get("checks")
    if not isinstance(checks, Mapping):
        block("audit final: checks absents")
    else:
        mismatches = [
            key
            for key, expected in required_checks.items()
            if checks.get(key) is not expected
        ]
        if mismatches:
            block("audit final: checks incomplets (" + ", ".join(mismatches) + ")")
        dynamic_hours = checks.get("exact_physical_hours_for_365_local_days")
        declared_hours = checks.get("expected_physical_hours")
        legacy_hours_ok = (
            len(frame) == 8760 and checks.get("exact_8760_physical_hours") is True
        )
        if dynamic_hours is not True and not legacy_hours_ok:
            block("audit final: grille horaire physique/DST non prouvee")
        if dynamic_hours is True and declared_hours != len(frame):
            block("audit final: nombre d'heures physiques divergent")
        tolerance = checks.get("actual_pairing_atol_eur_mwh")
        maximum = checks.get("actual_pairing_max_delta_eur_mwh")
        try:
            tolerance_value = float(tolerance)
            maximum_value = float(maximum)
        except (TypeError, ValueError):
            block("audit final: tolerance d'appariement invalide")
        else:
            if (
                not math.isclose(
                    tolerance_value,
                    ACTUAL_PAIRING_ATOL_EUR_MWH,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or not math.isfinite(maximum_value)
                or maximum_value < 0.0
                or maximum_value > tolerance_value
            ):
                block("audit final: appariement des actuals non prouve")
    if audit.get("final_evidence_sha256") != _sha256(predictions_path):
        block("audit final: SHA des predictions finales divergent")

    audit_metrics = audit.get("metrics")
    declared_metrics = _read_json(
        metrics_path, label=f"metriques finales rang {rank}"
    )
    if not isinstance(audit_metrics, Mapping) or any(
        key not in audit_metrics
        or not _numeric_equal(audit_metrics[key], declared_metrics.get(key))
        for key in (
            "baseline_mae_eur_mwh",
            "candidate_mae_eur_mwh",
            "mae_gain_eur_mwh",
        )
    ):
        block("audit final: metriques headline divergentes")

    raw_path_verified: Path | None = None
    raw_reference = audit.get("raw_candidate_evidence")
    if not isinstance(raw_reference, Mapping):
        block("audit final: preuve LoRA brute absente")
    else:
        try:
            raw_path = _safe_artifact_path(
                run_directory, raw_reference.get("relative_path")
            )
            raw_sha = raw_reference.get("sha256")
            if not _is_sha256(raw_sha) or not raw_path.is_file() or _sha256(raw_path) != raw_sha:
                block("audit final: preuve LoRA brute non verifiable")
            else:
                raw_path_verified = raw_path
                lineage_hashes[raw_path] = str(raw_sha)
        except CandidateComparisonError:
            block("audit final: preuve LoRA brute non verifiable")

    source_specs = (
        ("residual_corrector", "residual_corrector_sha256", None),
        ("oof_training_audit", "oof_training_audit_sha256", None),
        ("incumbent_statistics", None, INCUMBENT_MODEL),
    )
    resolved_sources: dict[str, Path] = {}
    for name, experiment_sha_key, expected_model in source_specs:
        reference = audit.get(name)
        if not isinstance(reference, Mapping):
            block(f"audit final: {name} absent")
            continue
        if expected_model is not None and reference.get("model") != expected_model:
            block(f"audit final: modèle {name} non canonique")
        expected_sha = reference.get("sha256")
        if not _is_sha256(expected_sha):
            block(f"audit final: SHA {name} invalide")
            continue
        if experiment_sha_key is not None and experiment.get(
            experiment_sha_key
        ) != expected_sha:
            block(f"audit final: {name} non lie au manifeste d'experience")
        if name == "incumbent_statistics":
            final_reference = experiment.get("final_pipeline_evaluation")
            if not isinstance(final_reference, Mapping) or final_reference.get(
                "incumbent_statistics_sha256"
            ) != expected_sha:
                block("audit final: incumbent non lie a la preuve finale")
        try:
            source_path = _resolve_audit_source(
                reference.get("path"), run_directory=run_directory, label=name
            )
        except CandidateComparisonError:
            block(f"audit final: path {name} invalide")
            continue
        if not source_path.is_file() or _sha256(source_path) != expected_sha:
            block(f"audit final: {name} absent ou modifie")
            continue
        resolved_sources[name] = source_path
        lineage_hashes[source_path] = str(expected_sha)

    validated_corrector: Mapping[str, Any] | None = None
    if {"residual_corrector", "oof_training_audit"}.issubset(resolved_sources):
        local_delivery = pd.DatetimeIndex(frame["delivery_start_utc"]).tz_convert(
            str(declared_metrics.get("timezone"))
        )
        first_holdout_day = local_delivery[0].date().isoformat()
        try:
            validated_corrector = _validate_oof_chain(
                corrector_path=resolved_sources["residual_corrector"],
                oof_audit_path=resolved_sources["oof_training_audit"],
                experiment=experiment,
                first_holdout_day=first_holdout_day,
            )
        except Exception as exc:
            block(f"chaine correcteur/OOF non reproductible ({exc})")
    else:
        block("chaine correcteur/OOF non reproductible")

    schema_path = run_directory / "schema.json"
    schema: Mapping[str, Any] | None = None
    if (
        not schema_path.is_file()
        or not _is_sha256(experiment.get("schema_sha256"))
        or _sha256(schema_path) != experiment.get("schema_sha256")
    ):
        block("schema LoRA absent, modifie ou non lie")
    else:
        try:
            schema = _read_json(schema_path, label=f"schema LoRA rang {rank}")
            lineage_hashes[schema_path] = str(experiment["schema_sha256"])
        except CandidateComparisonError as exc:
            block(str(exc))

    if (
        raw_path_verified is not None
        and validated_corrector is not None
        and schema is not None
    ):
        try:
            raw_frame = pd.read_csv(raw_path_verified)
            if tuple(raw_frame.columns) != EVALUATION_COLUMNS:
                raise CandidateComparisonError("schema raw non canonique")
            raw_frame = _normalise_window(
                raw_frame,
                timezone_name=str(schema.get("timezone")),
                required_days=365,
                end_day=None,
                phase=f"rank{rank}_raw_lineage",
            )
            for column in ("delivery_start_utc", "forecast_origin_utc"):
                if not pd.DatetimeIndex(raw_frame[column]).equals(
                    pd.DatetimeIndex(frame[column])
                ):
                    raise CandidateComparisonError(f"{column} raw/final divergent")
            if not np.array_equal(
                raw_frame["actual"].to_numpy(float),
                frame["actual"].to_numpy(float),
                equal_nan=False,
            ):
                raise CandidateComparisonError("actual raw/final divergent")
            timestamp_column = schema.get("timestamp_column")
            if not isinstance(timestamp_column, str) or not timestamp_column:
                raise CandidateComparisonError("timestamp_column du schema absent")
            horizon = pd.DataFrame(
                {timestamp_column: pd.DatetimeIndex(frame["delivery_start_utc"])}
            )
            raw_candidate = np.stack(
                [
                    raw_frame[f"candidate_{quantile}"].to_numpy(float)
                    for quantile in ("q10", "q50", "q90")
                ],
                axis=0,
            )[np.newaxis, :, :]
            corrected, _ = _apply_residual_corrector(
                raw_candidate,
                horizon=horizon,
                corrector=validated_corrector,
                schema=schema,
            )
            for index, quantile in enumerate(("q10", "q50", "q90")):
                if not np.allclose(
                    corrected[0, index, :],
                    frame[f"candidate_{quantile}"].to_numpy(float),
                    rtol=0.0,
                    atol=1e-10,
                ):
                    raise CandidateComparisonError(
                        f"candidate_{quantile} final != LoRA brut + correcteur"
                    )
        except Exception as exc:
            block(f"transformation LoRA brut + correcteur non reproductible ({exc})")
    else:
        block("transformation LoRA brut + correcteur non reproductible")

    incumbent_path = resolved_sources.get("incumbent_statistics")
    incumbent_reference = audit.get("incumbent_statistics")
    if incumbent_path is not None and isinstance(incumbent_reference, Mapping):
        try:
            incumbent_all = pd.read_csv(incumbent_path)
            missing = sorted(set(INCUMBENT_COLUMNS).difference(incumbent_all.columns))
            if missing:
                raise CandidateComparisonError(
                    "colonnes incumbent absentes: " + ", ".join(missing)
                )
            incumbent = incumbent_all.loc[:, list(INCUMBENT_COLUMNS)].copy()
            incumbent["delivery_start_utc"] = pd.to_datetime(
                incumbent["delivery_start_utc"], utc=True, errors="coerce"
            )
            incumbent["forecast_origin_utc"] = pd.to_datetime(
                incumbent["forecast_origin_utc"], utc=True, errors="coerce"
            )
            deliveries = pd.DatetimeIndex(frame["delivery_start_utc"])
            incumbent = incumbent.loc[
                incumbent["delivery_start_utc"].isin(deliveries)
            ].sort_values("delivery_start_utc", kind="stable")
            if (
                len(incumbent) != len(frame)
                or incumbent["delivery_start_utc"].duplicated().any()
                or not pd.DatetimeIndex(incumbent["delivery_start_utc"]).equals(
                    deliveries
                )
                or not pd.DatetimeIndex(incumbent["forecast_origin_utc"]).equals(
                    pd.DatetimeIndex(frame["forecast_origin_utc"])
                )
            ):
                raise CandidateComparisonError("timeline/origines incumbent divergentes")
            incumbent_actual = pd.to_numeric(
                incumbent["actual"], errors="coerce"
            ).to_numpy(float)
            if not np.allclose(
                incumbent_actual,
                frame["actual"].to_numpy(float),
                rtol=0.0,
                atol=ACTUAL_PAIRING_ATOL_EUR_MWH,
            ):
                raise CandidateComparisonError("actuals incumbent divergents")
            for quantile in ("q10", "q50", "q90"):
                incumbent_values = pd.to_numeric(
                    incumbent[f"residual_corrected__{quantile}"], errors="coerce"
                ).to_numpy(float)
                if not np.allclose(
                    incumbent_values,
                    frame[f"baseline_{quantile}"].to_numpy(float),
                    rtol=0.0,
                    atol=1e-10,
                ):
                    raise CandidateComparisonError(
                        f"incumbent residual_corrected__{quantile} divergent"
                    )
            if (
                incumbent_reference.get("source_rows") != len(incumbent_all)
                or incumbent_reference.get("paired_rows") != len(incumbent)
            ):
                raise CandidateComparisonError("comptages incumbent divergents")
        except Exception as exc:
            block(f"incumbent residual_corrected non reproductible ({exc})")
    else:
        block("incumbent residual_corrected non reproductible")
    return tuple(blockers), tuple(
        sorted(lineage_hashes.items(), key=lambda item: str(item[0]))
    )


def _read_final_candidate(
    source: str | Path,
    *,
    expected_rank: int,
    policy: GovernancePolicy,
) -> CandidateProof:
    final_directory = _resolve_final_directory(source)
    run_directory = final_directory.parent
    manifest_path = final_directory / FINAL_MANIFEST_NAME
    manifest = _read_json(manifest_path, label=f"manifeste final rang {expected_rank}")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != FINAL_KIND:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: manifeste final non reconnu."
        )
    if manifest.get("production_pipeline_evidence") is not True:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: production_pipeline_evidence doit etre true."
        )
    if manifest.get("candidate_output_stage") != OUTPUT_MODEL:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: seule la sortie LoRA corrigee est comparable."
        )
    zone = manifest.get("zone")
    experiment_id = manifest.get("experiment_id")
    if not isinstance(zone, str) or not zone.strip():
        raise CandidateComparisonError(f"Rang {expected_rank}: zone absente.")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise CandidateComparisonError(
            f"Rang {expected_rank}: experiment_id absent."
        )
    zone = zone.strip().upper()

    paths: dict[str, tuple[Path, str]] = {}
    for name, filename in (
        ("predictions", FINAL_EVIDENCE_NAME),
        ("metrics", FINAL_METRICS_NAME),
        ("report", FINAL_REPORT_NAME),
        ("audit", FINAL_AUDIT_NAME),
    ):
        paths[name] = _artifact_from_manifest(
            manifest,
            final_directory=final_directory,
            name=name,
            expected_name=filename,
        )
    predictions_path, predictions_sha = paths["predictions"]
    metrics_path, metrics_sha = paths["metrics"]

    experiment_path = run_directory / "experiment_manifest.json"
    experiment = _read_json(
        experiment_path, label=f"manifeste d'experience rang {expected_rank}"
    )
    training = experiment.get("training")
    lora_config = training.get("lora_config") if isinstance(training, Mapping) else None
    declared_rank = lora_config.get("r") if isinstance(lora_config, Mapping) else None
    if isinstance(declared_rank, bool) or not isinstance(declared_rank, int):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: training.lora_config.r absent ou invalide."
        )
    if declared_rank != expected_rank:
        raise CandidateComparisonError(
            f"Candidat annonce rang {expected_rank}, artefact declare rang {declared_rank}."
        )
    if experiment.get("zone") != zone or experiment.get("experiment_id") != experiment_id:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: identite zone/experience divergente."
        )
    if (
        experiment.get("production_pipeline_evidence") is not True
        or experiment.get("candidate_output_stage") != OUTPUT_MODEL
    ):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: manifeste d'experience non finalise."
        )
    final_reference = experiment.get("final_pipeline_evaluation")
    evidence_reference = experiment.get("evaluation_evidence")
    if not isinstance(final_reference, Mapping) or not isinstance(
        evidence_reference, Mapping
    ):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: references de preuve finale absentes."
        )
    if final_reference.get("sha256") != _sha256(manifest_path):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: SHA du manifeste final non lie a l'experience."
        )
    if evidence_reference.get("sha256") != predictions_sha:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: SHA des predictions finales non lie a l'experience."
        )

    try:
        raw = pd.read_csv(predictions_path)
    except Exception as exc:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: predictions finales illisibles."
        ) from exc
    if tuple(raw.columns) != EVALUATION_COLUMNS:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: schema exact des predictions finales invalide."
        )
    try:
        frame = _normalise_window(
            raw,
            timezone_name=policy.timezone,
            required_days=policy.rolling_evaluation_days,
            end_day=None,
            phase=f"rank{expected_rank}",
        )
    except Exception as exc:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: preuve rolling365 invalide: {exc}"
        ) from exc
    if len(frame) != len(raw):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: la preuve contient des heures hors rolling365."
        )

    window = manifest.get("window")
    if not isinstance(window, Mapping):
        raise CandidateComparisonError(f"Rang {expected_rank}: window absente.")
    expected_first = pd.Timestamp(frame["delivery_start_utc"].iloc[0])
    expected_last = pd.Timestamp(frame["delivery_start_utc"].iloc[-1])
    try:
        declared_first = pd.Timestamp(str(window.get("first_delivery_utc")))
        declared_last = pd.Timestamp(str(window.get("last_delivery_utc")))
    except Exception as exc:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: bornes temporelles invalides."
        ) from exc
    if (
        window.get("physical_days") != policy.rolling_evaluation_days
        or window.get("physical_hours") != len(frame)
        or declared_first != expected_first
        or declared_last != expected_last
    ):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: fenetre declaree divergente des predictions."
        )

    declared_metrics = _read_json(
        metrics_path, label=f"metriques finales rang {expected_rank}"
    )
    recomputed_metrics, _ = compute_metrics(frame, timezone_name=policy.timezone)
    _verify_declared_metrics(
        declared_metrics, recomputed_metrics, rank=expected_rank
    )
    blockers, lineage_hashes = _operational_selection_blockers(
        rank=expected_rank,
        run_directory=run_directory,
        final_manifest=manifest,
        experiment=experiment,
        audit_path=paths["audit"][0],
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        frame=frame,
    )
    return CandidateProof(
        rank=expected_rank,
        evidence_stage="final_corrected",
        zone=zone,
        experiment_id=experiment_id,
        run_directory=run_directory,
        final_directory=final_directory,
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        manifest_sha256=_sha256(manifest_path),
        predictions_sha256=predictions_sha,
        metrics_sha256=metrics_sha,
        frame=frame,
        declared_metrics=declared_metrics,
        operational_selection_blockers=blockers,
        operational_lineage_hashes=lineage_hashes,
    )


def _read_raw_candidate(
    source: str | Path,
    *,
    expected_rank: int,
    policy: GovernancePolicy,
) -> CandidateProof:
    path = Path(source).expanduser().resolve()
    run_directory = path.parent if path.is_file() else path
    manifest_path = run_directory / RAW_MANIFEST_NAME
    manifest = _read_json(
        manifest_path, label=f"manifeste d'evaluation brute rang {expected_rank}"
    )
    if manifest.get("format_version") != 1 or manifest.get("kind") != RAW_KIND:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: manifeste d'evaluation brute non reconnu."
        )
    comparison = manifest.get("comparison")
    if not isinstance(comparison, Mapping) or (
        comparison.get("same_inputs") is not True
        or comparison.get("cross_learning") is not False
        or comparison.get("residual_corrector_applied") is not False
    ):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: contrat raw LoRA/base non apparié."
        )
    zone = manifest.get("item_id")
    experiment_id = manifest.get("experiment_id")
    input_contract = manifest.get("input_contract_sha256")
    if not isinstance(zone, str) or not zone.strip():
        raise CandidateComparisonError(f"Rang {expected_rank}: zone raw absente.")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise CandidateComparisonError(
            f"Rang {expected_rank}: experiment_id raw absent."
        )
    if not _is_sha256(input_contract):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: input_contract_sha256 raw absent."
        )
    zone = zone.strip().upper()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: artefacts raw absents."
        )

    def raw_artifact(name: str, expected_name: str) -> tuple[Path, str]:
        reference = artifacts.get(name)
        if not isinstance(reference, Mapping):
            raise CandidateComparisonError(
                f"Rang {expected_rank}: artefact raw {name} absent."
            )
        artifact = _safe_artifact_path(run_directory, reference.get("relative_path"))
        expected_sha = reference.get("sha256")
        if (
            artifact.name != expected_name
            or not artifact.is_file()
            or not _is_sha256(expected_sha)
            or _sha256(artifact) != expected_sha
        ):
            raise CandidateComparisonError(
                f"Rang {expected_rank}: artefact raw {name} absent ou SHA-256 divergent."
            )
        return artifact, expected_sha

    predictions_path, predictions_sha = raw_artifact("evidence", RAW_EVIDENCE_NAME)
    metrics_path, metrics_sha = raw_artifact("metrics", RAW_METRICS_NAME)
    raw_artifact("report", RAW_REPORT_NAME)
    raw_artifact("daily", RAW_DAILY_NAME)

    experiment_path = run_directory / "experiment_manifest.json"
    experiment = _read_json(
        experiment_path, label=f"manifeste d'experience rang {expected_rank}"
    )
    training = experiment.get("training")
    lora_config = training.get("lora_config") if isinstance(training, Mapping) else None
    declared_rank = lora_config.get("r") if isinstance(lora_config, Mapping) else None
    if isinstance(declared_rank, bool) or not isinstance(declared_rank, int):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: training.lora_config.r absent ou invalide."
        )
    if declared_rank != expected_rank:
        raise CandidateComparisonError(
            f"Candidat annonce rang {expected_rank}, artefact declare rang {declared_rank}."
        )
    if experiment.get("experiment_id") != experiment_id:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: identite d'experience raw divergente."
        )
    if experiment.get("production_pipeline_evidence") is not False:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: une preuve raw ne peut pas revendiquer le pipeline final."
        )
    evidence_reference = experiment.get("evaluation_evidence")
    if (
        not isinstance(evidence_reference, Mapping)
        or evidence_reference.get("sha256") != predictions_sha
    ):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: predictions raw non liees a l'experience."
        )
    if manifest.get("bundle_manifest_sha256") != _sha256(experiment_path):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: manifeste raw non lie au bundle LoRA courant."
        )
    try:
        raw = pd.read_csv(predictions_path)
    except Exception as exc:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: predictions raw illisibles."
        ) from exc
    if tuple(raw.columns) != EVALUATION_COLUMNS:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: schema exact des predictions raw invalide."
        )
    try:
        frame = _normalise_window(
            raw,
            timezone_name=policy.timezone,
            required_days=policy.rolling_evaluation_days,
            end_day=None,
            phase=f"rank{expected_rank}_raw",
        )
    except Exception as exc:
        raise CandidateComparisonError(
            f"Rang {expected_rank}: preuve raw rolling365 invalide: {exc}"
        ) from exc
    if len(frame) != len(raw):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: la preuve raw contient des heures hors rolling365."
        )
    window = manifest.get("window")
    if not isinstance(window, Mapping):
        raise CandidateComparisonError(f"Rang {expected_rank}: window raw absente.")
    if (
        window.get("physical_days") != policy.rolling_evaluation_days
        or window.get("physical_hours") != len(frame)
        or pd.Timestamp(str(window.get("first_delivery_utc")))
        != pd.Timestamp(frame["delivery_start_utc"].iloc[0])
        or pd.Timestamp(str(window.get("last_delivery_utc")))
        != pd.Timestamp(frame["delivery_start_utc"].iloc[-1])
    ):
        raise CandidateComparisonError(
            f"Rang {expected_rank}: fenetre raw declaree divergente."
        )
    declared_metrics = _read_json(
        metrics_path, label=f"metriques raw rang {expected_rank}"
    )
    if declared_metrics.get("comparison_scope") != "raw_chronos2_same_exogenous_inputs":
        raise CandidateComparisonError(
            f"Rang {expected_rank}: scope des metriques raw divergent."
        )
    recomputed_metrics, _ = compute_metrics(frame, timezone_name=policy.timezone)
    _verify_declared_metrics(declared_metrics, recomputed_metrics, rank=expected_rank)
    return CandidateProof(
        rank=expected_rank,
        evidence_stage="raw_lora",
        zone=zone,
        experiment_id=experiment_id,
        run_directory=run_directory,
        final_directory=run_directory,
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        manifest_sha256=_sha256(manifest_path),
        predictions_sha256=predictions_sha,
        metrics_sha256=metrics_sha,
        frame=frame,
        declared_metrics=declared_metrics,
        input_contract_sha256=input_contract,
    )


def _read_candidate(
    source: str | Path,
    *,
    expected_rank: int,
    policy: GovernancePolicy,
) -> CandidateProof:
    stage = _source_stage(source)
    if stage == "final_corrected":
        return _read_final_candidate(
            source, expected_rank=expected_rank, policy=policy
        )
    return _read_raw_candidate(source, expected_rank=expected_rank, policy=policy)


def _assert_same_values(
    rank8: CandidateProof, rank16: CandidateProof, column: str
) -> None:
    left = rank8.frame[column]
    right = rank16.frame[column]
    if column in {"delivery_start_utc", "forecast_origin_utc"}:
        equal = pd.DatetimeIndex(left).equals(pd.DatetimeIndex(right))
    else:
        equal = np.array_equal(
            left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=False
        )
    if not equal:
        raise CandidateComparisonError(
            f"Comparaison non appariee: {column} differe entre rang 8 et rang 16."
        )


def _source_payload(proof: CandidateProof) -> dict[str, Any]:
    return {
        "rank": proof.rank,
        "evidence_stage": proof.evidence_stage,
        "zone": proof.zone,
        "experiment_id": proof.experiment_id,
        "run_directory": str(proof.run_directory),
        "evidence_manifest": {
            "path": str(proof.manifest_path),
            "sha256": proof.manifest_sha256,
        },
        "predictions": {
            "path": str(proof.predictions_path),
            "sha256": proof.predictions_sha256,
        },
        "metrics": {
            "path": str(proof.metrics_path),
            "sha256": proof.metrics_sha256,
        },
        "input_contract_sha256": proof.input_contract_sha256,
        "operational_selection_ready": (
            proof.evidence_stage == "final_corrected"
            and not proof.operational_selection_blockers
        ),
        "operational_selection_blockers": list(
            proof.operational_selection_blockers
        ),
        "operational_lineage": [
            {"path": str(path), "sha256": digest}
            for path, digest in proof.operational_lineage_hashes
        ],
    }


def _render_report(
    payload: Mapping[str, Any], daily: pd.DataFrame, *, title: str
) -> str:
    metrics = payload["paired_metrics"]
    governance = payload["governance"]
    gate = governance["gate"]
    winner = str(payload["winner"])
    evidence_stage = str(payload["evidence_stage"])
    is_final = evidence_stage == "final_corrected"
    selection_eligible = payload.get("candidate_selection_eligible") is True
    if is_final and selection_eligible:
        winner_label = "Rang 16 retenu" if winner == "rank16" else "Rang 8 conserve"
    else:
        winner_label = (
            "Avantage exploratoire au rang 16"
            if winner == "rank16"
            else "Avantage exploratoire au rang 8"
        )
    winner_class = "good" if winner == "rank16" else "warn"

    def fmt(value: object, digits: int = 3) -> str:
        numeric = float(value)
        return "—" if not math.isfinite(numeric) else f"{numeric:.{digits}f}"

    checks = "".join(
        "<tr>"
        f"<td>{html.escape(str(name))}</td>"
        f"<td>{html.escape(str(record['operator']))} {html.escape(str(record['threshold']))}</td>"
        f"<td>{html.escape(str(record['observed']))}</td>"
        f"<td class={'good' if record['passes'] else 'bad'}>{'OK' if record['passes'] else 'ECHEC'}</td>"
        "</tr>"
        for name, record in gate["checks"].items()
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.delivery_day))}</td>"
        f"<td>{int(row.hours)}</td>"
        f"<td>{row.observed_mean_price_eur_mwh:.2f}</td>"
        f"<td>{row.baseline_hourly_mae_eur_mwh:.3f}</td>"
        f"<td>{row.candidate_hourly_mae_eur_mwh:.3f}</td>"
        f"<td>{row.baseline_hourly_mae_eur_mwh-row.candidate_hourly_mae_eur_mwh:.3f}</td>"
        "</tr>"
        for row in daily.itertuples(index=False)
    )
    reasons = gate.get("reasons") or []
    reason_text = (
        "Tous les seuils rolling365 sont franchis."
        if not reasons
        else "; ".join(html.escape(str(value)) for value in reasons)
    )
    if is_final and selection_eligible:
        evidence_label = "pipelines LoRA corrigés finaux et chaîne vérifiée"
    elif is_final:
        evidence_label = (
            "pipelines LoRA corrigés, mais chaîne opérationnelle incomplète — "
            "résultat exploratoire"
        )
    else:
        evidence_label = "sorties LoRA brutes — résultat exploratoire"
    gate_title = (
        "Seuils de gouvernance"
        if is_final and selection_eligible
        else "Seuils de gouvernance appliqués à titre exploratoire"
    )
    readiness = payload.get("selection_readiness")
    readiness_blockers = (
        readiness.get("blockers", []) if isinstance(readiness, Mapping) else []
    )
    eligibility_text = (
        ""
        if selection_eligible
        else " Sélection opérationnelle interdite : "
        + "; ".join(html.escape(str(value)) for value in readiness_blockers)
        + "."
    )
    return f"""<!doctype html>
<html lang="fr" data-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--bg:#f3f6fb;--card:#fff;--text:#172033;--muted:#68758a;--line:#dce4ef;--r8:#b36a00;--r16:#087fc1;--good:#117a4b;--bad:#c43d48;--warn:#9a6200}}
[data-theme=dark]{{--bg:#0d1421;--card:#172131;--text:#eaf0f8;--muted:#a9b6ca;--line:#304057;--r8:#ffb347;--r16:#62c8ff;--good:#61d79b;--bad:#ff7e87;--warn:#ffd166}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}}main{{max-width:1180px;margin:auto;padding:26px}}header{{display:flex;justify-content:space-between;gap:18px}}h1{{margin:0 0 6px;font-size:25px}}h2{{margin-top:26px}}button,.card,.notice,details{{background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:11px}}button{{padding:8px 12px;height:38px;cursor:pointer}}.muted{{color:var(--muted)}}.notice{{padding:13px 15px;border-left:4px solid var(--r16);margin:18px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card{{padding:15px}}.value{{font-size:23px;font-weight:700;margin-top:5px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.warn{{color:var(--warn)}}details{{padding:12px 14px;margin-top:16px}}summary{{font-weight:650;cursor:pointer}}.table{{max-height:520px;overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:var(--card)}}@media(max-width:650px){{main{{padding:15px}}header{{display:block}}button{{margin-top:10px}}}}
</style></head><body><main><header><div><h1>{html.escape(title)}</h1>
<div class="muted">Comparaison strictement appariée · {html.escape(str(payload['zone']))} · {evidence_label} · 365 jours / {int(metrics['physical_hours'])} heures</div></div><button id="theme">🌙 Mode nuit</button></header>
<div class="notice"><strong class="{winner_class}">{winner_label}</strong> — {reason_text}<br><span class="muted">Cette comparaison ne promeut et n'active aucun modèle.{eligibility_text}</span></div>
<section class="cards"><div class="card"><div class="muted">MAE rang 8</div><div class="value">{fmt(metrics['baseline_mae_eur_mwh'])}</div><div>EUR/MWh</div></div>
<div class="card"><div class="muted">MAE rang 16</div><div class="value">{fmt(metrics['candidate_mae_eur_mwh'])}</div><div>EUR/MWh</div></div>
<div class="card"><div class="muted">Gain du rang 16</div><div class="value {'good' if float(metrics['mae_gain_eur_mwh']) > 0 else 'bad'}">{fmt(metrics['mae_gain_eur_mwh'])}</div><div>{fmt(100*float(metrics['mae_relative_gain']),2)} %</div></div>
<div class="card"><div class="muted">Jours gagnés rang 16</div><div class="value">{int(metrics['candidate_better_hourly_mae_days'])}</div><div>sur 365</div></div></section>
<h2>{gate_title}</h2><div class="table"><table><thead><tr><th>Contrôle</th><th>Seuil</th><th>Observé</th><th>Résultat</th></tr></thead><tbody>{checks}</tbody></table></div>
<details><summary>Détail des 365 journées</summary><div class="table"><table><thead><tr><th>Jour</th><th>Heures</th><th>Observé</th><th>MAE rang 8</th><th>MAE rang 16</th><th>Gain rang 16</th></tr></thead><tbody>{rows}</tbody></table></div></details>
<p class="muted">Généré le {html.escape(str(payload['created_at_utc']))}. Fenêtre, heures, origines, prix observés et incumbent d'origine ont été vérifiés identiques.</p>
</main><script>const r=document.documentElement,b=document.getElementById('theme');function a(t){{r.dataset.theme=t;b.textContent=t==='dark'?'☀️ Mode jour':'🌙 Mode nuit'}}a(localStorage.getItem('chronos2-rank-comparison-theme')||'light');b.onclick=()=>{{const t=r.dataset.theme==='dark'?'light':'dark';localStorage.setItem('chronos2-rank-comparison-theme',t);a(t)}};</script></body></html>"""


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def compare_final_candidates(
    *,
    rank8_source: str | Path,
    rank16_source: str | Path,
    policy_path: str | Path,
    output_directory: str | Path,
    expected_zone: str | None = None,
    overwrite: bool = False,
    report_title: str = "Chronos-2 + LoRA — comparaison rang 8 / rang 16",
) -> CandidateComparisonArtifacts:
    """Compare final corrected LoRA proofs without any deployment mutation."""

    policy_source = Path(policy_path).expanduser().resolve()
    policy = load_policy(policy_source)
    rank8 = _read_candidate(rank8_source, expected_rank=8, policy=policy)
    rank16 = _read_candidate(rank16_source, expected_rank=16, policy=policy)
    if rank8.evidence_stage != rank16.evidence_stage:
        raise CandidateComparisonError(
            "Stages incompatibles: comparez deux preuves finales corrigees ou "
            "deux preuves LoRA brutes, jamais un melange."
        )
    if rank8.zone != rank16.zone:
        raise CandidateComparisonError(
            f"Zones divergentes: rang 8={rank8.zone}, rang 16={rank16.zone}."
        )
    if expected_zone is not None and rank8.zone != str(expected_zone).strip().upper():
        raise CandidateComparisonError(
            f"Zone attendue {expected_zone!r}, preuves reçues pour {rank8.zone}."
        )
    for column in (
        "delivery_start_utc",
        "forecast_origin_utc",
        "actual",
        "baseline_q10",
        "baseline_q50",
        "baseline_q90",
    ):
        _assert_same_values(rank8, rank16, column)
    if (
        rank8.evidence_stage == "raw_lora"
        and rank8.input_contract_sha256 != rank16.input_contract_sha256
    ):
        raise CandidateComparisonError(
            "Comparaison raw non appariee: input_contract_sha256 differe."
        )

    paired = pd.DataFrame(
        {
            "delivery_start_utc": rank8.frame["delivery_start_utc"],
            "forecast_origin_utc": rank8.frame["forecast_origin_utc"],
            "actual": rank8.frame["actual"],
            "baseline_q10": rank8.frame["candidate_q10"],
            "baseline_q50": rank8.frame["candidate_q50"],
            "baseline_q90": rank8.frame["candidate_q90"],
            "candidate_q10": rank16.frame["candidate_q10"],
            "candidate_q50": rank16.frame["candidate_q50"],
            "candidate_q90": rank16.frame["candidate_q90"],
        },
        columns=EVALUATION_COLUMNS,
    )
    normalised = _normalise_window(
        paired,
        timezone_name=policy.timezone,
        required_days=policy.rolling_evaluation_days,
        end_day=None,
        phase="rank8_vs_rank16",
    )
    paired_metrics, daily = compute_metrics(
        normalised, timezone_name=policy.timezone
    )
    is_final = rank8.evidence_stage == "final_corrected"
    scope = (
        "paired_final_lora_rank8_vs_rank16"
        if is_final
        else "paired_raw_lora_rank8_vs_rank16"
    )
    paired_metrics.update(
        {
            "comparison_scope": scope,
            "baseline_label": (
                "Chronos-2 + LoRA rang 8 + correcteur residuel"
                if is_final
                else "Chronos-2 + LoRA rang 8 brut"
            ),
            "candidate_label": (
                "Chronos-2 + LoRA rang 16 + correcteur residuel"
                if is_final
                else "Chronos-2 + LoRA rang 16 brut"
            ),
        }
    )
    governance_metrics = _window_metrics(
        normalised, phase="rolling365", policy=policy
    )
    gate = _evaluate_gate(governance_metrics, policy=policy, phase="rolling365")
    winner = "rank16" if gate.passes else "rank8"
    selection_blockers = [
        f"rang {proof.rank}: {reason}"
        for proof in (rank8, rank16)
        for reason in proof.operational_selection_blockers
    ]
    if not is_final:
        selection_blockers.append(
            "deux preuves finales corrigées et canoniques sont obligatoires"
        )
    selection_eligible = is_final and not selection_blockers
    if selection_eligible:
        decision = "select_rank16" if gate.passes else "retain_rank8"
    else:
        decision = (
            "research_preference_rank16"
            if gate.passes
            else "research_preference_rank8"
        )
    created_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "kind": "chronos2_exogenous_candidate_comparison",
        "created_at_utc": created_at,
        "zone": rank8.zone,
        "evidence_stage": rank8.evidence_stage,
        "comparison_scope": scope,
        "pairing_checks": {
            "same_zone": True,
            "same_rolling_window": True,
            "same_physical_hours": True,
            "same_delivery_timeline": True,
            "same_forecast_origins": True,
            "same_actuals": True,
            "same_original_incumbent": True,
        },
        "sources": {
            "rank8": _source_payload(rank8),
            "rank16": _source_payload(rank16),
        },
        "policy": {
            "path": str(policy_source),
            "sha256": _sha256(policy_source),
            "snapshot": policy.to_dict(),
        },
        "paired_metrics": paired_metrics,
        "governance": {
            "rank8_role": "incumbent",
            "rank16_role": "challenger",
            "metrics": asdict(governance_metrics),
            "gate": asdict(gate),
        },
        "decision": decision,
        "winner": winner,
        "candidate_selection_eligible": selection_eligible,
        "selection_readiness": {
            "passes": selection_eligible,
            "blockers": selection_blockers,
        },
        "promotion_eligible": False,
        "promotion_performed": False,
        "activation_performed": False,
    }
    if not selection_eligible:
        payload["limitations"] = (
            [
                "Sorties LoRA brutes sans correcteur residuel final.",
                "Comparaison exploratoire uniquement; production_pipeline_evidence=false.",
                "FinalBacktest apparie requis avant selection operationnelle.",
            ]
            if not is_final
            else [
                "Les predictions finales restent comparables pour la recherche.",
                "La selection operationnelle est interdite tant que PIT et lineage "
                "correcteur/OOF/incumbent ne sont pas integralement verifies.",
            ]
        )

    destination = Path(output_directory).expanduser().resolve()
    for source_root in (rank8.run_directory, rank16.run_directory):
        if _paths_overlap(destination, source_root):
            raise CandidateComparisonError(
                "Le dossier de comparaison doit etre distinct des artefacts candidats."
            )
    if destination.exists() and not overwrite:
        raise CandidateComparisonError(
            f"Comparaison deja publiee: {destination}; overwrite explicite requis."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    backup: Path | None = None
    source_hashes = {
        rank8.manifest_path: rank8.manifest_sha256,
        rank8.predictions_path: rank8.predictions_sha256,
        rank8.metrics_path: rank8.metrics_sha256,
        rank16.manifest_path: rank16.manifest_sha256,
        rank16.predictions_path: rank16.predictions_sha256,
        rank16.metrics_path: rank16.metrics_sha256,
    }
    for proof in (rank8, rank16):
        for source_path, source_sha in proof.operational_lineage_hashes:
            source_hashes[source_path] = source_sha
    try:
        staging.mkdir(parents=False, exist_ok=False)
        daily_path = staging / COMPARISON_DAILY_NAME
        daily.to_csv(daily_path, index=False, lineterminator="\n")
        report_path = staging / COMPARISON_REPORT_NAME
        report_path.write_text(
            _render_report(payload, daily, title=report_title), encoding="utf-8"
        )
        payload["outputs"] = {
            "daily": {
                "relative_path": COMPARISON_DAILY_NAME,
                "sha256": _sha256(daily_path),
            },
            "report": {
                "relative_path": COMPARISON_REPORT_NAME,
                "sha256": _sha256(report_path),
            },
        }
        comparison_path = staging / COMPARISON_JSON_NAME
        comparison_path.write_text(_json_text(payload), encoding="utf-8")
        for source_path, expected_sha in source_hashes.items():
            if _sha256(source_path) != expected_sha:
                raise CandidateComparisonError(
                    f"Artefact candidat modifie pendant la comparaison: {source_path}."
                )
        if destination.exists():
            backup = destination.parent / f".{destination.name}.backup-{uuid4().hex}"
            os.replace(destination, backup)
        os.replace(staging, destination)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
            backup = None
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
            backup = None
        raise
    return CandidateComparisonArtifacts(
        output_directory=destination,
        comparison_path=destination / COMPARISON_JSON_NAME,
        report_path=destination / COMPARISON_REPORT_NAME,
        daily_path=destination / COMPARISON_DAILY_NAME,
        decision=decision,
        winner=winner,
        comparison=payload,
    )


__all__ = [
    "COMPARISON_DAILY_NAME",
    "COMPARISON_JSON_NAME",
    "COMPARISON_REPORT_NAME",
    "CandidateComparisonArtifacts",
    "CandidateComparisonError",
    "compare_final_candidates",
]
