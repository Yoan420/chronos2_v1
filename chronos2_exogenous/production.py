"""Conditional, opt-in runtime for a promoted Chronos-2 exogenous bundle.

Nothing in this module is imported by the incumbent forecast launcher.  A
candidate must first pass the sealed promotion contract, then be registered
with ``enabled_by_default=false``.  Daily inference is only performed by an
explicit call to :func:`run_registered_candidate` (or its dedicated CLI).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .governance import validate_final_shadow_manifest, verify_promotion_bundle
from .lora_finetune import sha256_directory


REGISTRY_SCHEMA_VERSION = 1
RUNTIME_SCHEMA_VERSION = 1
OUTPUT_MODEL = "exogenous_residual_corrected"
BASE_MODEL = "chronos2_exogenous"
QUANTILES = ("q10", "q50", "q90")
PROMOTED_SHADOW_ISSUED_COLUMNS = (
    "delivery_start_utc",
    "forecast_origin_utc",
    "actual",
    "candidate_q10",
    "candidate_q50",
    "candidate_q90",
    "item_id",
    "target_column",
    "checkpoint_sha256",
    "input_contract_sha256",
    "panel_contract_sha256",
    "forecast_created_at_utc",
    "forecast_record_sha256",
)
ADAPTER_IMPORT_ALLOWLIST = ["chronos.chronos2.model"]
SAFE_DERIVED_FEATURES = frozenset(
    {
        "intercept",
        "local_hour_sin",
        "local_hour_cos",
        "local_dow_sin",
        "local_dow_cos",
        "local_doy_sin",
        "local_doy_cos",
        "is_weekend",
    }
)
FORBIDDEN_CORRECTOR_FEATURE_TOKENS = (
    "actual",
    "observed",
    "realized",
    "realised",
    "target",
    "storm",
    "mkonline",
)
FORBIDDEN_CORRECTOR_FEATURES = frozenset(
    {
        "delivery_start_utc",
        "forecast_origin_utc",
        "candidate_q10",
        "candidate_q50",
        "candidate_q90",
        "baseline_q10",
        "baseline_q50",
        "baseline_q90",
    }
)


class ExogenousProductionError(RuntimeError):
    """Raised when a candidate cannot safely enter the opt-in runtime."""


@dataclass(frozen=True)
class PromotedBundle:
    alias: str
    bundle_path: Path
    candidate_id: str
    candidate_model: str
    zone: str
    bundle_manifest_sha256: str
    artifact_checksums_sha256: str
    checkpoint_path: Path
    schema_path: Path
    experiment_manifest_path: Path
    residual_corrector_path: Path
    oof_audit_path: Path
    rolling_predictions_path: Path


@dataclass(frozen=True)
class ExogenousRunResult:
    output_directory: Path
    forecast_path: Path
    backtest_path: Path
    manifest_path: Path
    delivery_day: str
    zone: str
    model: str = OUTPUT_MODEL


@dataclass(frozen=True)
class PromotedShadowHistory:
    """Final-pipeline history rebuilt from sealed prospective shadow rows."""

    predictions: pd.DataFrame
    predictions_path: Path
    manifest_path: Path
    predictions_sha256: str
    manifest_sha256: str
    delivery_days: tuple[str, ...]
    issued_history_path: Path
    issued_history_sha256: str


PipelineLoader = Callable[[Path, Mapping[str, Any]], Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExogenousProductionError(f"{label} illisible: {path}") from exc
    if not isinstance(payload, dict):
        raise ExogenousProductionError(f"{label} doit etre un objet JSON: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _unique_file(root: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise ExogenousProductionError(
            f"{label}: un fichier unique est requis dans {root}; trouves={len(matches)}."
        )
    return matches[0]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_civil_delivery_index(
    delivery_day: object,
    *,
    timezone_name: str,
) -> pd.DatetimeIndex:
    """Return every physical hour of one local civil delivery day in UTC."""

    try:
        parsed_day = pd.Timestamp(delivery_day).date()
        zone = ZoneInfo(str(timezone_name))
        start_local = pd.Timestamp(parsed_day).tz_localize(zone)
        end_local = pd.Timestamp(parsed_day + timedelta(days=1)).tz_localize(zone)
    except Exception as exc:
        raise ExogenousProductionError(
            "Jour ou timezone de livraison invalide pour la timeline civile."
        ) from exc
    expected = pd.date_range(
        start=start_local,
        end=end_local,
        inclusive="left",
        freq="h",
    ).tz_convert("UTC")
    if len(expected) not in {23, 24, 25}:
        raise ExogenousProductionError(
            "La timeline civile attendue ne contient pas 23/24/25 heures."
        )
    return expected


def _validate_corrector_feature_columns(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ExogenousProductionError("feature_columns du correcteur invalides.")
    features = tuple(str(feature).strip() for feature in value)
    if any(not feature for feature in features) or len(set(features)) != len(features):
        raise ExogenousProductionError("feature_columns du correcteur invalides.")
    forbidden = [
        feature
        for feature in features
        if feature.casefold() in FORBIDDEN_CORRECTOR_FEATURES
        or any(
            token in feature.casefold()
            for token in FORBIDDEN_CORRECTOR_FEATURE_TOKENS
        )
    ]
    if forbidden:
        raise ExogenousProductionError(
            "Features label/prediction interdites dans le correcteur: "
            + ", ".join(forbidden)
        )
    return features


def _validate_corrector(
    path: Path,
    *,
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _json(path, label="correcteur residuel exogene")
    if (
        payload.get("research_only") is True
        or payload.get("kind") == "research_validation_corrector"
        or payload.get("fit_protocol")
        == "post_training_validation30_in_sample_not_oof"
    ):
        raise ExogenousProductionError(
            "Correcteur validation30 research-only interdit en production; "
            "une preuve blocked_prequential_oof_rolling365 est obligatoire."
        )
    required: Mapping[str, object] = {
        "schema_version": 1,
        "model_kind": "linear_shift_v1",
        "base_model": BASE_MODEL,
        "output_model": OUTPUT_MODEL,
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "training_days": 365,
        "selection_frozen_before_holdout": True,
        "holdout_used_for_fit": False,
        "future_actuals_used_as_features": False,
        "oof_audit_required": True,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
    }
    failures = [
        f"{key}={payload.get(key)!r}, attendu={expected!r}"
        for key, expected in required.items()
        if type(payload.get(key)) is not type(expected) or payload.get(key) != expected
    ]
    if failures:
        raise ExogenousProductionError(
            "Contrat du correcteur residuel invalide: " + "; ".join(failures)
        )
    oof_sha = payload.get("oof_training_predictions_sha256")
    if not _is_sha256(oof_sha):
        raise ExogenousProductionError(
            "oof_training_predictions_sha256 absent du correcteur residuel."
        )
    oof_audit_sha = payload.get("oof_training_audit_sha256")
    candidate_checkpoint = payload.get("candidate_checkpoint_sha256")
    if not _is_sha256(oof_audit_sha):
        raise ExogenousProductionError(
            "oof_training_audit_sha256 absent du correcteur residuel."
        )
    if not _is_sha256(candidate_checkpoint) or candidate_checkpoint != experiment.get(
        "checkpoint_sha256"
    ):
        raise ExogenousProductionError(
            "Le correcteur OOF ne correspond pas au checkpoint candidat evalue."
        )
    if payload.get("candidate_model") != BASE_MODEL:
        raise ExogenousProductionError(
            "Le correcteur OOF ne cible pas chronos2_exogenous."
        )
    features = _validate_corrector_feature_columns(payload.get("feature_columns"))
    coefficients = payload.get("coefficients")
    means = payload.get("feature_means")
    scales = payload.get("feature_scales")
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != len(features)
        or not isinstance(means, list)
        or len(means) != len(features)
        or not isinstance(scales, list)
        or len(scales) != len(features)
    ):
        raise ExogenousProductionError(
            "feature_columns/coefficients/normalisation du correcteur invalides."
        )
    numeric = np.asarray([coefficients, means, scales], dtype=float)
    if not np.isfinite(numeric).all() or bool((numeric[2] <= 0.0).any()):
        raise ExogenousProductionError("Parametres residuels non finis/invalides.")
    clip = payload.get("maximum_absolute_shift_eur_mwh")
    if isinstance(clip, bool) or not isinstance(clip, (int, float)):
        raise ExogenousProductionError("Clip du correcteur residuel invalide.")
    if not math.isfinite(float(clip)) or not 0.0 < float(clip) <= 50.0:
        raise ExogenousProductionError("Clip residuel attendu dans ]0, 50].")
    expected_sha = experiment.get("residual_corrector_sha256")
    if not _is_sha256(expected_sha) or expected_sha != _sha256(path):
        raise ExogenousProductionError(
            "Le correcteur ne correspond pas a celui evalue dans le manifeste."
        )
    if experiment.get("candidate_output_stage") != OUTPUT_MODEL:
        raise ExogenousProductionError(
            "Le holdout de promotion doit evaluer exogenous_residual_corrected."
        )
    return payload


def _validate_final_pipeline_evidence(
    experiment: Mapping[str, Any], *, rolling_path: Path | None = None
) -> None:
    """Prevent raw base/LoRA evidence from being relabelled as final pipelines."""

    detail = experiment.get("production_pipeline_evidence_detail")
    if not isinstance(detail, Mapping):
        raise ExogenousProductionError(
            "production_pipeline_evidence_detail final absent."
        )
    expected: Mapping[str, object] = {
        "comparison_scope": "paired_operational_final_pipelines",
        "baseline_output_stage": "residual_corrected",
        "candidate_output_stage": OUTPUT_MODEL,
        "paired_same_input_contract": True,
        "paired_same_evaluation_window": True,
        "baseline_residual_corrector_applied": True,
        "candidate_residual_corrector_applied": True,
        "rolling_evaluation_days": 365,
        "promotion_eligible": True,
    }
    failures = [
        f"{key}={detail.get(key)!r}, attendu={value!r}"
        for key, value in expected.items()
        if type(detail.get(key)) is not type(value) or detail.get(key) != value
    ]
    if failures:
        raise ExogenousProductionError(
            "Preuve du pipeline final invalide: " + "; ".join(failures)
        )
    if experiment.get("candidate_output_stage") != OUTPUT_MODEL:
        raise ExogenousProductionError(
            "candidate_output_stage ne designe pas le pipeline exogene corrige."
        )
    if rolling_path is not None:
        evidence = experiment.get("evaluation_evidence")
        if not isinstance(evidence, Mapping) or evidence.get("sha256") != _sha256(
            rolling_path
        ):
            raise ExogenousProductionError(
                "Les predictions rolling ne correspondent pas a la preuve du pipeline final."
            )


def validate_promoted_bundle(
    bundle_path: str | Path,
    *,
    alias: str,
    expected_zone: str | None = None,
) -> PromotedBundle:
    """Verify promotion gates and resolve immutable runtime artefacts."""

    bundle = Path(bundle_path).expanduser().resolve()
    manifest = verify_promotion_bundle(bundle)
    if manifest.get("decision") != "promote":
        raise ExogenousProductionError(
            f"Bundle non promu (decision={manifest.get('decision')!r})."
        )
    decision_path = bundle / "promotion_decision.json"
    decision = _json(decision_path, label="decision de promotion")
    required_true = (
        "production_pit_evidence",
        "production_pit_gate_passes",
        "production_pipeline_evidence",
        "production_pipeline_gate_passes",
        "shadow_evidence_verified",
    )
    false_flags = [name for name in required_true if decision.get(name) is not True]
    rolling_gate = decision.get("rolling365_gate")
    shadow_gate = decision.get("live_shadow_gate")
    if not isinstance(rolling_gate, Mapping) or rolling_gate.get("passes") is not True:
        false_flags.append("rolling365_gate.passes")
    if not isinstance(shadow_gate, Mapping) or shadow_gate.get("passes") is not True:
        false_flags.append("live_shadow_gate.passes")
    if false_flags:
        raise ExogenousProductionError(
            "Bundle refuse par les gates production: " + ", ".join(false_flags)
        )
    if decision.get("production_activation_performed") is not False:
        raise ExogenousProductionError("Une gate ne peut pas pre-activer la production.")
    zone = str(manifest.get("zone", "")).strip().upper()
    if expected_zone is not None and zone != str(expected_zone).strip().upper():
        raise ExogenousProductionError(
            f"Bundle {zone} incompatible avec la zone attendue {expected_zone}."
        )

    checkpoint = bundle / "artifacts" / "checkpoint"
    schema = _unique_file(bundle / "artifacts" / "schema", "*.json", label="schema")
    corrector = _unique_file(
        bundle / "artifacts" / "residual_corrector",
        "*.json",
        label="correcteur residuel",
    )
    oof_audit = _unique_file(
        bundle / "artifacts" / "oof_audit",
        "*.json",
        label="sidecar audit OOF",
    )
    experiment_manifest = _unique_file(
        bundle / "evidence" / "experiment_manifest",
        "*.json",
        label="manifeste d'experience",
    )
    rolling = _unique_file(
        bundle / "evidence" / "rolling365_predictions",
        "*",
        label="predictions rolling365",
    )
    if not checkpoint.is_dir() or not any(checkpoint.rglob("*")):
        raise ExogenousProductionError("Checkpoint du candidat absent ou vide.")
    experiment = _json(experiment_manifest, label="manifeste d'experience")
    if experiment.get("production_pit_evidence") is not True:
        raise ExogenousProductionError(
            "Le manifeste d'experience ne fournit pas de PIT production."
        )
    if experiment.get("production_pipeline_evidence") is not True:
        raise ExogenousProductionError(
            "Le manifeste d'experience n'evalue pas le pipeline final complet."
        )
    if experiment.get("evaluation_role") != "primary_predeclared":
        raise ExogenousProductionError(
            "Seule une evaluation primary_predeclared peut alimenter la promotion."
        )
    _validate_final_pipeline_evidence(experiment, rolling_path=rolling)
    expected_base_hash = experiment.get("base_model_snapshot_sha256")
    if not _is_sha256(expected_base_hash):
        raise ExogenousProductionError(
            "base_model_snapshot_sha256 absent du manifeste d'experience."
        )
    adapter_config = _json(
        checkpoint / "adapter_config.json", label="configuration de l'adaptateur"
    )
    base_value = adapter_config.get("base_model_name_or_path")
    base_snapshot = (
        Path(base_value).expanduser().resolve()
        if isinstance(base_value, str) and base_value.strip()
        else None
    )
    if base_snapshot is None or not base_snapshot.is_dir():
        raise ExogenousProductionError(
            "Le snapshot Chronos-2 local epingle par l'adaptateur est absent."
        )
    if sha256_directory(base_snapshot) != expected_base_hash:
        raise ExogenousProductionError(
            "Le snapshot Chronos-2 local differe de celui utilise au fine-tuning."
        )
    runtime = experiment.get("production_runtime")
    if not isinstance(runtime, Mapping):
        raise ExogenousProductionError("production_runtime absent du manifeste.")
    expected_runtime: Mapping[str, object] = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "layout": "per_zone",
        "cross_learning": False,
        "target": "target",
    }
    runtime_failures = [
        key
        for key, expected in expected_runtime.items()
        if type(runtime.get(key)) is not type(expected) or runtime.get(key) != expected
    ]
    if runtime_failures:
        raise ExogenousProductionError(
            "Runtime v1 incompatible: " + ", ".join(runtime_failures)
        )
    corrector_payload = _validate_corrector(corrector, experiment=experiment)
    oof_audit_sha256 = _sha256(oof_audit)
    if (
        corrector_payload.get("oof_training_audit_sha256") != oof_audit_sha256
        or experiment.get("oof_training_audit_sha256") != oof_audit_sha256
    ):
        raise ExogenousProductionError(
            "Le sidecar OOF scelle ne correspond pas au correcteur/manifeste."
        )
    oof_payload = _json(oof_audit, label="sidecar audit OOF scelle")
    if (
        oof_payload.get("predictions_sha256")
        != corrector_payload.get("oof_training_predictions_sha256")
        or oof_payload.get("candidate_checkpoint_sha256")
        != experiment.get("checkpoint_sha256")
        or oof_payload.get("refit_uses_only_strictly_prior_days") is not True
        or oof_payload.get("same_day_actual_excluded_from_fit") is not True
    ):
        raise ExogenousProductionError(
            "Le sidecar OOF scelle ne prouve pas le protocole prequential du candidat."
        )
    safe_alias = str(alias).strip()
    if not safe_alias or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in safe_alias):
        raise ExogenousProductionError(f"Alias invalide: {alias!r}.")
    return PromotedBundle(
        alias=safe_alias,
        bundle_path=bundle,
        candidate_id=str(manifest["candidate_id"]),
        candidate_model=str(manifest["candidate_model"]),
        zone=zone,
        bundle_manifest_sha256=_sha256(bundle / "bundle_manifest.json"),
        artifact_checksums_sha256=_sha256(bundle / "artifact_checksums.json"),
        checkpoint_path=checkpoint,
        schema_path=schema,
        experiment_manifest_path=experiment_manifest,
        residual_corrector_path=corrector,
        oof_audit_path=oof_audit,
        rolling_predictions_path=rolling,
    )


def register_promoted_bundle(
    bundle_path: str | Path,
    *,
    registry_path: str | Path,
    alias: str,
    expected_zone: str | None = None,
) -> PromotedBundle:
    """Record a verified candidate without enabling or executing it."""

    promoted = validate_promoted_bundle(
        bundle_path, alias=alias, expected_zone=expected_zone
    )
    registry = Path(registry_path).expanduser().resolve()
    if registry.exists():
        payload = _json(registry, label="registre exogene")
        if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ExogenousProductionError("Version du registre exogene incompatible.")
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise ExogenousProductionError("Entrees du registre exogene invalides.")
    else:
        payload = {"schema_version": REGISTRY_SCHEMA_VERSION, "entries": {}}
        entries = payload["entries"]
    assert isinstance(entries, dict)
    record = {
        "alias": promoted.alias,
        "bundle_path": str(promoted.bundle_path),
        "candidate_id": promoted.candidate_id,
        "candidate_model": promoted.candidate_model,
        "zone": promoted.zone,
        "bundle_manifest_sha256": promoted.bundle_manifest_sha256,
        "artifact_checksums_sha256": promoted.artifact_checksums_sha256,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "enabled_by_default": False,
        "incumbent_modified": False,
        "execution_requires_explicit_alias": True,
    }
    existing = entries.get(promoted.alias)
    if existing is not None:
        stable_keys = (
            "bundle_path",
            "candidate_id",
            "zone",
            "bundle_manifest_sha256",
            "artifact_checksums_sha256",
        )
        if not isinstance(existing, Mapping) or any(
            existing.get(key) != record[key] for key in stable_keys
        ):
            raise ExogenousProductionError(
                f"Alias deja enregistre pour un autre bundle: {promoted.alias}."
            )
        return promoted
    entries[promoted.alias] = record
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    registry.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry.with_name(f".{registry.name}.tmp-{uuid4().hex}")
    _write_json(temporary, payload)
    os.replace(temporary, registry)
    return promoted


def load_registered_bundle(
    *,
    registry_path: str | Path,
    alias: str,
    expected_zone: str | None = None,
) -> PromotedBundle:
    """Resolve an explicit alias and reverify the complete sealed bundle."""

    registry = Path(registry_path).expanduser().resolve()
    payload = _json(registry, label="registre exogene")
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ExogenousProductionError("Version du registre exogene incompatible.")
    entries = payload.get("entries")
    if not isinstance(entries, Mapping) or alias not in entries:
        raise ExogenousProductionError(f"Alias exogene non enregistre: {alias!r}.")
    record = entries[alias]
    if not isinstance(record, Mapping):
        raise ExogenousProductionError(f"Entree de registre invalide: {alias!r}.")
    if record.get("enabled_by_default") is not False:
        raise ExogenousProductionError(
            "Le registre conditionnel ne doit jamais activer un candidat par defaut."
        )
    promoted = validate_promoted_bundle(
        str(record.get("bundle_path", "")),
        alias=alias,
        expected_zone=expected_zone,
    )
    expected_hashes = {
        "bundle_manifest_sha256": promoted.bundle_manifest_sha256,
        "artifact_checksums_sha256": promoted.artifact_checksums_sha256,
    }
    mismatches = [
        key for key, value in expected_hashes.items() if record.get(key) != value
    ]
    if mismatches:
        raise ExogenousProductionError(
            "Le bundle enregistre a change: " + ", ".join(mismatches)
        )
    return promoted


def _read_panel(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ExogenousProductionError(f"Format de panel live non supporte: {path.name}")


def _verify_mutated_target_cache_against_live_panel(
    frame: pd.DataFrame,
    *,
    source_path: Path,
    timestamp_column: str,
    item_column: str,
    target_column: str,
    zone: str,
    sealed_source_sha256: str,
    current_source_sha256: str,
) -> dict[str, Any]:
    """Accept a target-cache append only when every sealed value is unchanged.

    The canonical target cache is mutable and normally receives new auction
    observations after a prospective panel has been sealed.  A byte SHA alone
    would reject that harmless append.  On mismatch, compare every historical
    target embedded in the sealed live context with the current canonical
    cache.  Missing or revised values remain a hard failure.
    """

    selected = frame.loc[
        frame[item_column].astype(str).str.upper().eq(zone)
        & frame["phase"].eq("context"),
        [timestamp_column, target_column],
    ].rename(columns={target_column: "panel_value"})
    if selected.empty:
        raise ExogenousProductionError(
            f"SHA du panel cible divergent: aucune valeur scellee pour {zone}."
        )
    selected["panel_value"] = pd.to_numeric(
        selected["panel_value"], errors="coerce"
    )
    if selected.isna().any().any() or not np.isfinite(
        selected["panel_value"].to_numpy(dtype=float)
    ).all():
        raise ExogenousProductionError(
            f"SHA du panel cible divergent: valeurs scellees {zone} invalides."
        )
    grouped = selected.groupby(timestamp_column, sort=True)["panel_value"].agg(
        ["min", "max"]
    )
    if bool(((grouped["max"] - grouped["min"]).abs() > 1e-9).any()):
        raise ExogenousProductionError(
            f"SHA du panel cible divergent: valeurs scellees {zone} incoherentes."
        )
    expected = grouped["min"].astype(float)

    try:
        current = pd.read_csv(source_path, usecols=["timestamp", "value"])
        current_timestamps = pd.to_datetime(
            current["timestamp"], utc=True, errors="coerce", format="mixed"
        )
        current_values = pd.to_numeric(current["value"], errors="coerce").astype(
            float
        )
    except Exception as exc:
        raise ExogenousProductionError(
            f"SHA du panel cible divergent: cache canonique {zone} illisible."
        ) from exc
    if (
        current_timestamps.isna().any()
        or current_timestamps.duplicated().any()
        or not np.isfinite(current_values.to_numpy(dtype=float)).all()
    ):
        raise ExogenousProductionError(
            f"SHA du panel cible divergent: cache canonique {zone} invalide."
        )
    current_series = pd.Series(
        current_values.to_numpy(dtype=float),
        index=pd.DatetimeIndex(current_timestamps),
    ).sort_index()
    observed = current_series.reindex(expected.index)
    if observed.isna().any():
        first_missing = observed.index[observed.isna()][0]
        raise ExogenousProductionError(
            f"SHA du panel cible divergent: le cache canonique {zone} ne contient "
            f"plus l'heure scellee {first_missing.isoformat()}."
        )

    observed_values = observed.to_numpy(dtype=float)
    expected_values = expected.to_numpy(dtype=float)
    deltas = observed_values - expected_values
    max_abs_delta = float(np.max(np.abs(deltas))) if len(deltas) else 0.0

    # EUPHEMIA publishes prices to the cent.  Re-aggregation from 15-minute
    # MTUs can produce quarter-cent representations of the same hourly market
    # price, so equivalence is accepted only after half-up cent rounding.
    def market_cent(values: np.ndarray) -> np.ndarray:
        magnitudes = np.floor(np.abs(values) * 100.0 + 0.5 + 1e-10) / 100.0
        return np.copysign(magnitudes, values)

    market_delta = np.abs(
        market_cent(observed_values) - market_cent(expected_values)
    )
    if bool((market_delta > 1e-9).any()):
        raise ExogenousProductionError(
            f"SHA du panel cible divergent: le cache canonique {zone} a modifie "
            "des valeurs utilisees par le panel au-dela de la precision marche "
            f"de 0,01 EUR/MWh (ecart brut max={max_abs_delta:.12g} EUR/MWh)."
        )
    return {
        "mode": "panel_target_equivalence",
        "verified_timestamps": int(len(expected)),
        "first_timestamp": expected.index.min().isoformat(),
        "last_timestamp": expected.index.max().isoformat(),
        "max_abs_delta": max_abs_delta,
        "market_precision_eur_mwh": 0.01,
        "rounding_equivalent_timestamps": int((np.abs(deltas) > 1e-9).sum()),
        "sealed_source_sha256": sealed_source_sha256,
        "current_source_sha256": current_source_sha256,
    }


def _validate_live_panel(
    path: Path,
    *,
    audit_path: Path,
    schema: Mapping[str, Any],
    bundle: PromotedBundle,
    experiment: Mapping[str, Any],
    delivery_day: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, dict[str, Any]]:
    if not path.is_file() or not audit_path.is_file():
        raise ExogenousProductionError("Panel live ou audit PIT absent.")
    audit = _json(audit_path, label="audit du panel live")
    if audit.get("panel_sha256") != _sha256(path):
        raise ExogenousProductionError("Checksum du panel live divergent.")
    if audit.get("purpose") != "prospective_shadow":
        raise ExogenousProductionError(
            "Le panel live doit declarer purpose=prospective_shadow."
        )
    if audit.get("layout") != "per_zone" or audit.get("zones") != [bundle.zone]:
        raise ExogenousProductionError(
            "Le sidecar live doit couvrir exactement une zone en layout per_zone."
        )
    if audit.get("production_ready") is not True:
        raise ExogenousProductionError(
            "Panel live non certifie production_ready par la banque PIT."
        )
    pit = audit.get("production_pit_evidence")
    if not isinstance(pit, Mapping) or not pit or any(value is not True for value in pit.values()):
        raise ExogenousProductionError("Preuves PIT production incompletes dans le panel live.")

    raw = _read_panel(path)
    timestamp_col = str(schema["timestamp_column"])
    origin_col = str(schema["origin_column"])
    item_col = str(schema["item_column"])
    available_col = str(schema["feature_available_at_column"])
    targets = tuple(map(str, schema["target_columns"]))
    known = tuple(map(str, schema["known_future_covariates"]))
    past_only = tuple(map(str, schema["past_only_covariates"]))
    required = {
        timestamp_col,
        origin_col,
        item_col,
        available_col,
        "phase",
        "delivery_day",
        *targets,
        *known,
        *past_only,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ExogenousProductionError(
            "Colonnes absentes du panel live: " + ", ".join(missing)
        )
    frame = raw.copy()
    for column in (timestamp_col, origin_col, available_col):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if frame[column].isna().any():
            raise ExogenousProductionError(f"Timestamp invalide dans {column}.")
    origins = pd.DatetimeIndex(frame[origin_col].drop_duplicates())
    if len(origins) != 1:
        raise ExogenousProductionError("Le panel live doit contenir une origine unique.")
    origin = origins[0]
    timezone_name = str(schema["timezone"])
    origin_local = origin.tz_convert(timezone_name)
    cutoff_h, cutoff_m = map(int, str(schema["cutoff_local_time"]).split(":"))
    if origin_local.hour != cutoff_h or origin_local.minute != cutoff_m:
        raise ExogenousProductionError("Origine live differente du cutoff scelle.")
    try:
        requested_day = pd.Timestamp(delivery_day).date()
    except ValueError as exc:
        raise ExogenousProductionError("delivery_day doit respecter YYYY-MM-DD.") from exc
    if (origin_local + pd.DateOffset(days=1)).date() != requested_day:
        raise ExogenousProductionError("Le panel ne correspond pas a la livraison D+1.")
    declared_origin = pd.to_datetime(
        audit.get("forecast_origin_utc"), utc=True, errors="coerce"
    )
    if pd.isna(declared_origin) or pd.Timestamp(declared_origin) != origin:
        raise ExogenousProductionError(
            "forecast_origin_utc du sidecar live divergent du panel."
        )
    if audit.get("delivery_day") != requested_day.isoformat():
        raise ExogenousProductionError(
            "delivery_day du sidecar live divergent de la commande."
        )
    if audit.get("forecast_origin_timezone") != timezone_name:
        raise ExogenousProductionError(
            "forecast_origin_timezone du sidecar incompatible avec le schema."
        )
    created = pd.to_datetime(audit.get("created_at_utc"), utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    deadline = origin + pd.Timedelta(hours=4)
    if pd.isna(created) or not (origin <= pd.Timestamp(created) <= min(deadline, now)):
        raise ExogenousProductionError(
            "Le panel live n'a pas ete capture dans la fenetre prospective D-1."
        )
    if audit.get("horizon_actuals_present") is not False or audit.get(
        "first_publication_with_actuals_must_be_rejected"
    ) is not True:
        raise ExogenousProductionError(
            "Le sidecar live n'atteste pas l'absence d'actuals au premier forecast."
        )
    pack = audit.get("pack")
    if not isinstance(pack, str) or not pack.strip():
        raise ExogenousProductionError("Pack exogene absent du sidecar live.")
    if pack.strip() != experiment.get("panel_pack"):
        raise ExogenousProductionError(
            "Le pack exogene live differe de celui utilise au fine-tuning."
        )

    def hashes(value: object, *, label: str) -> dict[str, str]:
        if not isinstance(value, Mapping) or not value:
            raise ExogenousProductionError(f"{label} absent/vide du sidecar live.")
        result = {str(name): str(digest) for name, digest in value.items()}
        invalid = [name for name, digest in result.items() if not _is_sha256(digest)]
        if invalid:
            raise ExogenousProductionError(
                f"{label}: SHA-256 invalides: {invalid}."
            )
        return result

    bank_root = audit.get("exogenous_banks")
    bank = bank_root.get(bundle.zone) if isinstance(bank_root, Mapping) else None
    if not isinstance(bank, Mapping) or bank.get("production_ready") is not True:
        raise ExogenousProductionError(
            "Banque exogene de zone absente ou non production-ready."
        )
    source_hashes = hashes(bank.get("source_hashes"), label="source_hashes")
    source_audit_hashes = hashes(
        bank.get("source_audit_hashes"), label="source_audit_hashes"
    )
    external_sources = set(source_hashes).difference({"deterministic_calendar"})
    if set(source_audit_hashes) != external_sources:
        raise ExogenousProductionError(
            "Chaque source exogene live doit avoir un sidecar scelle."
        )
    source_evidence = bank.get("production_pit_evidence")
    if (
        not isinstance(source_evidence, Mapping)
        or set(map(str, source_evidence)) != set(source_hashes)
        or any(value is not True for value in source_evidence.values())
    ):
        raise ExogenousProductionError(
            "Preuve prospective absente pour une source exogene live."
        )
    source_timezones = bank.get("source_cutoff_timezones")
    if not isinstance(source_timezones, Mapping) or set(
        map(str, source_timezones)
    ) != set(source_hashes):
        raise ExogenousProductionError(
            "source_cutoff_timezones ne couvre pas toutes les sources live."
        )
    for name, source_timezone in source_timezones.items():
        try:
            ZoneInfo(str(source_timezone))
        except Exception as exc:
            raise ExogenousProductionError(
                f"Timezone source invalide pour {name}."
            ) from exc
    if bank.get("forecast_origin_timezone") != timezone_name:
        raise ExogenousProductionError(
            "Le fuseau d'origine de la banque live differe du schema."
        )
    expected_hash_root = experiment.get("source_hashes")
    expected_hashes = (
        expected_hash_root.get(bundle.zone)
        if isinstance(expected_hash_root, Mapping)
        else None
    )
    if not isinstance(expected_hashes, Mapping) or set(map(str, expected_hashes)) != set(
        source_hashes
    ):
        raise ExogenousProductionError(
            "Le schema de sources live differe de celui utilise au fine-tuning."
        )
    expected_audit_root = experiment.get("source_audit_hashes")
    expected_audits = (
        expected_audit_root.get(bundle.zone)
        if isinstance(expected_audit_root, Mapping)
        else None
    )
    if not isinstance(expected_audits, Mapping) or set(map(str, expected_audits)) != set(
        source_audit_hashes
    ):
        raise ExogenousProductionError(
            "Le schema de sidecars live differe de celui du fine-tuning."
        )
    expected_timezone_root = experiment.get("source_cutoff_timezones")
    expected_timezones = (
        expected_timezone_root.get(bundle.zone)
        if isinstance(expected_timezone_root, Mapping)
        else None
    )
    if not isinstance(expected_timezones, Mapping) or {
        str(name): str(value) for name, value in expected_timezones.items()
    } != {str(name): str(value) for name, value in source_timezones.items()}:
        raise ExogenousProductionError(
            "Les fuseaux de cutoff live different de ceux du fine-tuning."
        )
    delivery_timezones = audit.get("delivery_timezones")
    delivery_timezone = (
        delivery_timezones.get(bundle.zone)
        if isinstance(delivery_timezones, Mapping)
        else None
    )
    try:
        ZoneInfo(str(delivery_timezone))
    except Exception as exc:
        raise ExogenousProductionError(
            "Timezone de livraison absente/invalide dans le sidecar live."
        ) from exc
    target_sources = audit.get("target_sources")
    target_source = (
        target_sources.get(bundle.zone)
        if isinstance(target_sources, Mapping)
        else None
    )
    if not isinstance(target_source, Mapping) or not _is_sha256(
        target_source.get("source_sha256")
    ):
        raise ExogenousProductionError(
            "Identite SHA de la cible canonique absente du sidecar live."
        )
    target_contracts = audit.get("target_contracts")
    target_contract = (
        target_contracts.get(bundle.zone)
        if isinstance(target_contracts, Mapping)
        else None
    )
    if audit.get("canonical_target_contracts_verified") is not True or not isinstance(
        target_contract, Mapping
    ):
        raise ExogenousProductionError(
            "Contrat de cible canonique live absent/non verifie."
        )
    if not isinstance(target_contract.get("series"), str) or not str(
        target_contract.get("series")
    ).strip():
        raise ExogenousProductionError("Serie de cible canonique live absente.")
    target_source_path = target_source.get("source_path")
    target_cache_path = target_contract.get("cache_path")
    if (
        not isinstance(target_source_path, str)
        or not isinstance(target_cache_path, str)
        or Path(target_source_path).expanduser().resolve()
        != Path(target_cache_path).expanduser().resolve()
    ):
        raise ExogenousProductionError(
            "Le cache cible du panel ne correspond pas au contrat canonique."
        )
    resolved_target_path = Path(target_source_path).expanduser().resolve()
    if not resolved_target_path.is_file():
        raise ExogenousProductionError(
            "Le cache cible canonique courant lie au SHA du panel est absent."
        )
    sealed_target_sha = str(target_source["source_sha256"])
    current_target_sha = _sha256(resolved_target_path)
    if current_target_sha == sealed_target_sha:
        target_cache_verification: dict[str, Any] = {
            "mode": "exact_file_sha256",
            "verified_timestamps": None,
            "max_abs_delta": 0.0,
            "sealed_source_sha256": sealed_target_sha,
            "current_source_sha256": current_target_sha,
        }
    else:
        if len(targets) != 1:
            raise ExogenousProductionError(
                "SHA du panel cible divergent: runtime per_zone exige une cible unique."
            )
        target_cache_verification = _verify_mutated_target_cache_against_live_panel(
            frame,
            source_path=resolved_target_path,
            timestamp_column=timestamp_col,
            item_column=item_col,
            target_column=targets[0],
            zone=bundle.zone,
            sealed_source_sha256=sealed_target_sha,
            current_source_sha256=current_target_sha,
        )
    training_target_sources = experiment.get("target_sources")
    training_target_contracts = experiment.get("target_contracts")
    training_target_source = (
        training_target_sources.get(bundle.zone)
        if isinstance(training_target_sources, Mapping)
        else None
    )
    training_target_contract = (
        training_target_contracts.get(bundle.zone)
        if isinstance(training_target_contracts, Mapping)
        else None
    )
    if (
        not isinstance(training_target_source, Mapping)
        or not isinstance(training_target_contract, Mapping)
        or Path(str(training_target_source.get("source_path", ""))).expanduser().resolve()
        != resolved_target_path
        or training_target_contract.get("series") != target_contract.get("series")
        or Path(str(training_target_contract.get("cache_path", ""))).expanduser().resolve()
        != resolved_target_path
    ):
        raise ExogenousProductionError(
            "La cible canonique live differe de celle utilisee au fine-tuning."
        )
    if bool((frame[available_col] > frame[origin_col]).any()):
        raise ExogenousProductionError("Une feature du panel est disponible apres le cutoff.")
    items = tuple(map(str, frame[item_col].drop_duplicates()))
    if items != (bundle.zone,):
        raise ExogenousProductionError(
            f"Runtime per_zone attendu pour {bundle.zone}; items={items}."
        )
    frame = frame.sort_values([item_col, timestamp_col], kind="stable")
    phase_values = set(frame["phase"].drop_duplicates().tolist())
    if phase_values != {"context", "horizon"}:
        raise ExogenousProductionError(
            "Phases du panel live invalides: exactement context/horizon sont requis."
        )
    context = frame.loc[frame["phase"].eq("context")].copy()
    horizon = frame.loc[frame["phase"].eq("horizon")].copy()
    context_length = int(schema["context_length"])
    if len(context) != context_length:
        raise ExogenousProductionError("Longueur de contexte live incompatible.")
    expected_horizon = _expected_civil_delivery_index(
        requested_day,
        timezone_name=str(delivery_timezone),
    )
    actual_horizon = pd.DatetimeIndex(horizon[timestamp_col])
    if not actual_horizon.equals(expected_horizon):
        raise ExogenousProductionError(
            "Index civil de livraison incomplet, excedentaire ou DST incoherent."
        )
    expected_context = pd.date_range(
        end=expected_horizon[0] - pd.Timedelta(hours=1),
        periods=context_length,
        freq="h",
    )
    actual_context = pd.DatetimeIndex(context[timestamp_col])
    if not actual_context.equals(expected_context):
        raise ExogenousProductionError(
            "Index de contexte live non exactement adjacent a la livraison."
        )
    expected_timeline = expected_context.append(expected_horizon)
    timestamps = pd.DatetimeIndex(frame[timestamp_col])
    if not timestamps.equals(expected_timeline):
        raise ExogenousProductionError(
            "Partition context/horizon du panel live incoherente avec la timeline attendue."
        )
    context_numeric = context[[*targets, *known, *past_only]].apply(
        pd.to_numeric, errors="coerce"
    )
    future_numeric = horizon[list(known)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(context_numeric.to_numpy(float)).all() or not np.isfinite(
        future_numeric.to_numpy(float)
    ).all():
        raise ExogenousProductionError("Contexte ou covariables futures incompletes.")
    horizon_targets = horizon[list(targets)].apply(pd.to_numeric, errors="coerce")
    if bool(horizon_targets.notna().any().any()):
        raise ExogenousProductionError(
            "Les cibles futures doivent etre vides au moment de l'inference."
        )
    if set(map(str, frame["delivery_day"].unique())) != {requested_day.isoformat()}:
        raise ExogenousProductionError("delivery_day du panel live incoherent.")
    return context, horizon, origin, target_cache_verification


def _default_pipeline_loader(checkpoint: Path, kwargs: Mapping[str, Any]) -> Any:
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(checkpoint, **dict(kwargs))


def _base_prediction(
    *,
    pipeline: Any,
    context: pd.DataFrame,
    horizon: pd.DataFrame,
    schema: Mapping[str, Any],
) -> np.ndarray:
    targets = tuple(map(str, schema["target_columns"]))
    known = tuple(map(str, schema["known_future_covariates"]))
    past_only = tuple(map(str, schema["past_only_covariates"]))
    prepared = {
        "target": context[list(targets)].to_numpy(dtype=np.float32).T,
        "past_covariates": {
            name: context[name].to_numpy(dtype=np.float32)
            for name in (*known, *past_only)
        },
        "future_covariates": {
            name: horizon[name].to_numpy(dtype=np.float32) for name in known
        },
    }
    if not hasattr(pipeline, "predict_quantiles"):
        raise ExogenousProductionError(
            "Le runtime Chronos-2 doit exposer predict_quantiles."
        )
    quantiles, _ = pipeline.predict_quantiles(
        [prepared],
        prediction_length=len(horizon),
        quantile_levels=[0.1, 0.5, 0.9],
        context_length=int(schema["context_length"]),
        cross_learning=False,
        batch_size=max(32, len(targets) + len(known) + len(past_only)),
        limit_prediction_length=False,
    )
    if not isinstance(quantiles, Sequence) or len(quantiles) != 1:
        raise ExogenousProductionError("Sortie quantile Chronos-2 inattendue.")
    raw = quantiles[0]
    values = raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw)
    if values.ndim == 2:
        values = values[np.newaxis, :, :]
    expected_shape = (len(targets), len(horizon), 3)
    if values.ndim != 3 or values.shape != expected_shape:
        raise ExogenousProductionError(
            "Dimensions quantiles Chronos-2 inattendues: "
            f"{values.shape}, attendu={expected_shape}."
        )
    selected = np.transpose(values, (0, 2, 1))
    if not np.isfinite(selected).all():
        raise ExogenousProductionError("Prediction Chronos-2 non finie.")
    if bool((selected[:, 0, :] > selected[:, 1, :]).any()) or bool(
        (selected[:, 1, :] > selected[:, 2, :]).any()
    ):
        raise ExogenousProductionError("Croisement de quantiles Chronos-2.")
    return selected


def _corrector_design(
    horizon: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    timezone_name: str,
    timestamp_column: str,
) -> np.ndarray:
    features = tuple(map(str, manifest["feature_columns"]))
    timestamp = pd.DatetimeIndex(horizon[timestamp_column]).tz_convert(timezone_name)
    derived: dict[str, np.ndarray] = {
        "intercept": np.ones(len(horizon)),
        "local_hour_sin": np.sin(2 * np.pi * timestamp.hour / 24.0),
        "local_hour_cos": np.cos(2 * np.pi * timestamp.hour / 24.0),
        "local_dow_sin": np.sin(2 * np.pi * timestamp.dayofweek / 7.0),
        "local_dow_cos": np.cos(2 * np.pi * timestamp.dayofweek / 7.0),
        "local_doy_sin": np.sin(2 * np.pi * (timestamp.dayofyear - 1) / 365.2425),
        "local_doy_cos": np.cos(2 * np.pi * (timestamp.dayofyear - 1) / 365.2425),
        "is_weekend": np.asarray(timestamp.dayofweek >= 5, dtype=float),
    }
    columns: list[np.ndarray] = []
    for feature in features:
        if feature in SAFE_DERIVED_FEATURES:
            values = derived[feature]
        elif feature in horizon.columns:
            values = pd.to_numeric(horizon[feature], errors="coerce").to_numpy(float)
        else:
            raise ExogenousProductionError(
                f"Feature du correcteur absente du panel live: {feature}."
            )
        if not np.isfinite(values).all():
            raise ExogenousProductionError(f"Feature residuelle non finie: {feature}.")
        columns.append(np.asarray(values, dtype=float))
    return np.column_stack(columns)


def fit_oof_residual_corrector(
    *,
    oof_predictions_path: str | Path,
    oof_audit_path: str | Path,
    holdout_start_day: str,
    output_path: str | Path,
    timezone_name: str = "Europe/Paris",
    feature_columns: Sequence[str] = (
        "intercept",
        "local_hour_sin",
        "local_hour_cos",
        "local_dow_sin",
        "local_dow_cos",
    ),
    ridge_alpha: float = 1.0,
    maximum_absolute_shift_eur_mwh: float = 20.0,
) -> Path:
    """Fit the mandatory residual stage from 365 strictly pre-holdout OOF days."""

    source = Path(oof_predictions_path).expanduser().resolve()
    audit_source = Path(oof_audit_path).expanduser().resolve()
    if not source.is_file() or not audit_source.is_file():
        raise ExogenousProductionError(
            "Predictions OOF et sidecar audit OOF explicite sont obligatoires."
        )
    oof_audit = _json(audit_source, label="audit OOF du correcteur")
    sidecar_schema_version = oof_audit.get("schema_version")
    if type(sidecar_schema_version) is not int or sidecar_schema_version not in {1, 2}:
        raise ExogenousProductionError(
            "schema_version du sidecar OOF doit valoir 1 ou 2."
        )
    expected_audit: Mapping[str, object] = {
        "purpose": "chronos2_exogenous_blocked_prequential_oof",
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "candidate_model": BASE_MODEL,
        "training_days": 365,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "future_actuals_used_as_features": False,
        "holdout_used_for_fit": False,
        "selection_frozen_before_oof": True,
    }
    audit_failures = [
        f"{key}={oof_audit.get(key)!r}, attendu={expected!r}"
        for key, expected in expected_audit.items()
        if type(oof_audit.get(key)) is not type(expected)
        or oof_audit.get(key) != expected
    ]
    if audit_failures:
        raise ExogenousProductionError(
            "Contrat du sidecar OOF invalide: " + "; ".join(audit_failures)
        )
    if oof_audit.get("predictions_sha256") != _sha256(source):
        raise ExogenousProductionError(
            "Le sidecar OOF n'est pas lie aux predictions fournies."
        )
    candidate_checkpoint_sha256 = oof_audit.get("candidate_checkpoint_sha256")
    if not _is_sha256(candidate_checkpoint_sha256):
        raise ExogenousProductionError(
            "candidate_checkpoint_sha256 absent/invalide dans le sidecar OOF."
        )
    suffixes = "".join(source.suffixes).lower()
    frame = pd.read_parquet(source) if suffixes.endswith(".parquet") else pd.read_csv(source)
    required = {"delivery_start_utc", "forecast_origin_utc", "actual", "candidate_q50"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ExogenousProductionError("OOF correcteur incomplet: " + ", ".join(missing))
    validated_features = _validate_corrector_feature_columns(list(feature_columns))
    if not math.isfinite(float(ridge_alpha)) or float(ridge_alpha) < 0.0:
        raise ExogenousProductionError("ridge_alpha doit etre fini et positif ou nul.")
    clip = float(maximum_absolute_shift_eur_mwh)
    if not math.isfinite(clip) or not 0.0 < clip <= 50.0:
        raise ExogenousProductionError("Clip residuel attendu dans ]0, 50].")

    delivery = pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="coerce")
    origins = pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="coerce")
    if delivery.isna().any() or origins.isna().any():
        raise ExogenousProductionError("Timestamps OOF invalides.")
    if delivery.duplicated().any():
        raise ExogenousProductionError("Heures OOF dupliquees.")
    order = np.argsort(delivery.to_numpy())
    frame = frame.iloc[order].reset_index(drop=True)
    delivery = pd.DatetimeIndex(delivery.iloc[order])
    origins = pd.DatetimeIndex(origins.iloc[order])
    local = delivery.tz_convert(timezone_name)
    days = tuple(dict.fromkeys(local.date))
    if len(days) != 365 or any(
        days[index] != days[0] + pd.Timedelta(days=index)
        for index in range(len(days))
    ):
        raise ExogenousProductionError("Le correcteur exige exactement 365 jours consecutifs.")
    holdout_start = pd.Timestamp(holdout_start_day).date()
    if days[-1] >= holdout_start:
        raise ExogenousProductionError("Le fit OOF chevauche le holdout de promotion.")
    expected_audit_dates = {
        "training_start_day": str(days[0]),
        "training_end_day": str(days[-1]),
        "holdout_start_day": str(holdout_start),
    }
    date_failures = [
        f"{key}={oof_audit.get(key)!r}, attendu={expected!r}"
        for key, expected in expected_audit_dates.items()
        if oof_audit.get(key) != expected
    ]
    if date_failures:
        raise ExogenousProductionError(
            "Fenetre du sidecar OOF incoherente: " + "; ".join(date_failures)
        )
    expected_parts: list[pd.DatetimeIndex] = []
    expected_origins: list[pd.Timestamp] = []
    for day in days:
        start = pd.Timestamp(day, tz=timezone_name)
        end = pd.Timestamp(day + pd.Timedelta(days=1), tz=timezone_name)
        expected = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")
        expected_parts.append(expected)
        origin_day = day - pd.Timedelta(days=1)
        wall_clock_origin = pd.Timestamp(
            f"{origin_day:%Y-%m-%d} 08:00", tz=timezone_name
        )
        expected_origins.extend([wall_clock_origin] * len(expected))
    expected_delivery = expected_parts[0].append(expected_parts[1:])
    if not delivery.equals(expected_delivery):
        raise ExogenousProductionError("Timeline OOF incomplete ou DST incoherente.")
    expected_origin_index = pd.DatetimeIndex(expected_origins).tz_convert("UTC")
    if not origins.equals(expected_origin_index):
        raise ExogenousProductionError("Origines OOF attendues exactement a D-1 08:00 local.")

    fold_checkpoint_set_sha256: str | None = None
    fold_candidate_recipe_sha256: str | None = None
    if sidecar_schema_version == 2:
        v2_expected: Mapping[str, object] = {
            "candidate_checkpoint_role": "deployment_identity_anchor_not_oof_predictor",
            "deployment_checkpoint_used_for_oof": False,
            "fold_checkpoints_are_origin_specific": True,
            "fold_lookback_days": 365,
        }
        v2_failures = [
            f"{key}={oof_audit.get(key)!r}, attendu={expected!r}"
            for key, expected in v2_expected.items()
            if type(oof_audit.get(key)) is not type(expected)
            or oof_audit.get(key) != expected
        ]
        folds = oof_audit.get("folds")
        declared_fold_count = oof_audit.get("fold_count")
        if not isinstance(folds, list) or not folds:
            v2_failures.append("folds absents")
            folds = []
        if type(declared_fold_count) is not int or declared_fold_count != len(folds):
            v2_failures.append("fold_count incoherent")
        fold_candidate_recipe_sha256 = oof_audit.get(
            "fold_candidate_recipe_sha256"
        )
        if not _is_sha256(fold_candidate_recipe_sha256):
            v2_failures.append("fold_candidate_recipe_sha256 invalide")
        fold_checkpoint_set_sha256 = oof_audit.get("fold_checkpoint_set_sha256")
        if not _is_sha256(fold_checkpoint_set_sha256):
            v2_failures.append("fold_checkpoint_set_sha256 invalide")
        if v2_failures:
            raise ExogenousProductionError(
                "Contrat fold-specific OOF v2 invalide: " + "; ".join(v2_failures)
            )

        distinct_origins = pd.DatetimeIndex(origins.drop_duplicates()).sort_values()
        covered_origins: list[pd.Timestamp] = []
        checkpoint_contract: list[dict[str, object]] = []
        for expected_fold_index, raw_fold in enumerate(folds, start=1):
            if not isinstance(raw_fold, Mapping):
                raise ExogenousProductionError("Fold OOF v2 non objet.")
            if raw_fold.get("fold_index") != expected_fold_index:
                raise ExogenousProductionError("Index des folds OOF v2 non consecutif.")
            if raw_fold.get("refit_uses_only_strictly_prior_days") is not True:
                raise ExogenousProductionError(
                    f"Fold {expected_fold_index}: refit non strictement anterieur."
                )
            checkpoint_sha = raw_fold.get("checkpoint_sha256")
            predictions_sha = raw_fold.get("predictions_sha256")
            if not _is_sha256(checkpoint_sha) or not _is_sha256(predictions_sha):
                raise ExogenousProductionError(
                    f"Fold {expected_fold_index}: SHA checkpoint/predictions invalide."
                )
            ranges: dict[str, tuple[pd.DatetimeIndex, int]] = {}
            for name in (
                "fit_origins",
                "train_origins",
                "validation_origins",
                "prediction_origins",
            ):
                raw_range = raw_fold.get(name)
                if not isinstance(raw_range, Mapping):
                    raise ExogenousProductionError(
                        f"Fold {expected_fold_index}: plage {name} absente."
                    )
                count = raw_range.get("count")
                first = pd.to_datetime(raw_range.get("first_utc"), utc=True, errors="coerce")
                last = pd.to_datetime(raw_range.get("last_utc"), utc=True, errors="coerce")
                if (
                    type(count) is not int
                    or count <= 0
                    or pd.isna(first)
                    or pd.isna(last)
                ):
                    raise ExogenousProductionError(
                        f"Fold {expected_fold_index}: plage {name} invalide."
                    )
                first_day = pd.Timestamp(first).tz_convert(timezone_name).normalize()
                last_day = pd.Timestamp(last).tz_convert(timezone_name).normalize()
                expanded_days = pd.date_range(first_day, last_day, freq="D")
                if len(expanded_days) != count:
                    raise ExogenousProductionError(
                        f"Fold {expected_fold_index}: plage {name} non consecutive."
                    )
                cutoff_hour = expected_origin_index[0].tz_convert(timezone_name).hour
                cutoff_minute = expected_origin_index[0].tz_convert(timezone_name).minute
                expanded = pd.DatetimeIndex(
                    [
                        pd.Timestamp(
                            f"{day:%Y-%m-%d} {cutoff_hour:02d}:{cutoff_minute:02d}",
                            tz=timezone_name,
                        )
                        for day in expanded_days
                    ]
                )
                ranges[name] = (expanded.tz_convert("UTC"), count)
            fit_range, fit_count = ranges["fit_origins"]
            train_range, train_count = ranges["train_origins"]
            validation_range, validation_count = ranges["validation_origins"]
            prediction_range, _prediction_count = ranges["prediction_origins"]
            if fit_count != 365 or train_count + validation_count != 365:
                raise ExogenousProductionError(
                    f"Fold {expected_fold_index}: lookback doit contenir 365 origines."
                )
            combined_fit = train_range.append(validation_range)
            if not combined_fit.equals(fit_range):
                raise ExogenousProductionError(
                    f"Fold {expected_fold_index}: train+validation != fit."
                )
            prediction_first_local = prediction_range[0].tz_convert(timezone_name)
            expected_fit_day = prediction_first_local.date() - pd.Timedelta(days=1)
            expected_fit_end = pd.Timestamp(
                f"{expected_fit_day:%Y-%m-%d} "
                f"{cutoff_hour:02d}:{cutoff_minute:02d}",
                tz=timezone_name,
            )
            if fit_range[-1] != expected_fit_end.tz_convert("UTC"):
                raise ExogenousProductionError(
                    f"Fold {expected_fold_index}: le fit ne finit pas a J-1 du bloc."
                )
            if fit_range[-1] >= prediction_range[0]:
                raise ExogenousProductionError(
                    f"Fold {expected_fold_index}: fuite temporelle fit/prediction."
                )
            covered_origins.extend(pd.Timestamp(value) for value in prediction_range)
            checkpoint_contract.append(
                {
                    "fold_index": expected_fold_index,
                    "checkpoint_sha256": checkpoint_sha,
                }
            )
        if not pd.DatetimeIndex(covered_origins).equals(distinct_origins):
            raise ExogenousProductionError(
                "Les folds OOF v2 ne couvrent pas exactement les 365 origines."
            )
        computed_checkpoint_set_sha256 = hashlib.sha256(
            json.dumps(
                checkpoint_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if computed_checkpoint_set_sha256 != fold_checkpoint_set_sha256:
            raise ExogenousProductionError(
                "fold_checkpoint_set_sha256 ne correspond pas aux folds declares."
            )

    numeric = frame[["actual", "candidate_q50"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ExogenousProductionError("Actuals ou predictions OOF non finis.")
    design = _corrector_design(
        frame.assign(delivery_start_utc=delivery),
        manifest={"feature_columns": list(validated_features)},
        timezone_name=timezone_name,
        timestamp_column="delivery_start_utc",
    )
    means = design.mean(axis=0)
    scales = design.std(axis=0)
    for index, feature in enumerate(validated_features):
        if feature == "intercept":
            means[index], scales[index] = 0.0, 1.0
        elif scales[index] <= 1e-12:
            scales[index] = 1.0
    normalised = (design - means) / scales
    target = numeric["actual"].to_numpy(float) - numeric["candidate_q50"].to_numpy(float)
    penalty = np.eye(normalised.shape[1]) * float(ridge_alpha)
    for index, feature in enumerate(validated_features):
        if feature == "intercept":
            penalty[index, index] = 0.0
    coefficients = np.linalg.lstsq(
        normalised.T @ normalised + penalty,
        normalised.T @ target,
        rcond=None,
    )[0]
    fitted = np.clip(normalised @ coefficients, -clip, clip)
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise ExogenousProductionError(f"Correcteur immutable deja existant: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_kind": "linear_shift_v1",
        "base_model": BASE_MODEL,
        "output_model": OUTPUT_MODEL,
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "training_days": 365,
        "training_start_day": str(days[0]),
        "training_end_day": str(days[-1]),
        "holdout_start_day": str(holdout_start),
        "selection_frozen_before_holdout": True,
        "holdout_used_for_fit": False,
        "future_actuals_used_as_features": False,
        "oof_audit_required": True,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "oof_training_predictions_sha256": _sha256(source),
        "oof_training_audit_sha256": _sha256(audit_source),
        "oof_sidecar_schema_version": int(sidecar_schema_version),
        "candidate_model": BASE_MODEL,
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        "oof_fold_checkpoint_set_sha256": fold_checkpoint_set_sha256,
        "oof_fold_candidate_recipe_sha256": fold_candidate_recipe_sha256,
        "feature_columns": list(validated_features),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge_alpha": float(ridge_alpha),
        "maximum_absolute_shift_eur_mwh": clip,
        "fit_mae_before": float(np.mean(np.abs(target))),
        "fit_mae_after": float(np.mean(np.abs(target - fitted))),
    }
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
    _write_json(temporary, payload)
    temporary.replace(destination)
    return destination


def _apply_residual_corrector(
    base: np.ndarray,
    *,
    horizon: pd.DataFrame,
    corrector: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    design = _corrector_design(
        horizon,
        manifest=corrector,
        timezone_name=str(schema["timezone"]),
        timestamp_column=str(schema["timestamp_column"]),
    )
    means = np.asarray(corrector["feature_means"], dtype=float)
    scales = np.asarray(corrector["feature_scales"], dtype=float)
    design = (design - means) / scales
    coefficients = np.asarray(corrector["coefficients"], dtype=float)
    shift = design @ coefficients
    clip = float(corrector["maximum_absolute_shift_eur_mwh"])
    shift = np.clip(shift, -clip, clip)
    corrected = base + shift[np.newaxis, np.newaxis, :]
    if not np.isfinite(corrected).all():
        raise ExogenousProductionError("Prediction corrigee non finie.")
    return corrected, shift


def load_promoted_shadow_history(
    bundle: PromotedBundle,
) -> PromotedShadowHistory:
    """Load the bundle's sealed final-pipeline prospective shadow history.

    Only the v4 evidence comparing ``residual_corrected`` with the already
    corrected ``exogenous_residual_corrected`` candidate is accepted.  Raw v3
    adapter forecasts are deliberately unusable here, which prevents both a
    missing correction and an accidental second application of that stage.
    """

    sealed = verify_promotion_bundle(bundle.bundle_path)
    expected_identity = {
        "candidate_id": bundle.candidate_id,
        "candidate_model": bundle.candidate_model,
        "zone": bundle.zone,
        "decision": "promote",
    }
    mismatches = [
        key for key, expected in expected_identity.items()
        if sealed.get(key) != expected
    ]
    if mismatches:
        raise ExogenousProductionError(
            "Identite du bundle shadow divergente: " + ", ".join(mismatches)
        )
    if _sha256(bundle.bundle_path / "bundle_manifest.json") != (
        bundle.bundle_manifest_sha256
    ) or _sha256(bundle.bundle_path / "artifact_checksums.json") != (
        bundle.artifact_checksums_sha256
    ):
        raise ExogenousProductionError(
            "Empreinte du bundle shadow differente du registre charge."
        )

    role_predictions_path = _unique_file(
        bundle.bundle_path / "evidence" / "shadow_predictions",
        "*",
        label="predictions shadow scellees",
    )
    role_manifest_path = _unique_file(
        bundle.bundle_path / "evidence" / "shadow_manifest",
        "*.json",
        label="manifeste shadow scelle",
    )
    lineage_root = bundle.bundle_path / "evidence" / "shadow_lineage"
    predictions_path = _unique_file(
        lineage_root,
        "shadow_final_evidence.csv.gz",
        label="predictions shadow finales auto-contenues",
    )
    manifest_path = _unique_file(
        lineage_root,
        "shadow_final_manifest.json",
        label="manifeste shadow final auto-contenu",
    )
    predictions_sha256 = _sha256(predictions_path)
    manifest_sha256 = _sha256(manifest_path)
    if (
        _sha256(role_predictions_path) != predictions_sha256
        or _sha256(role_manifest_path) != manifest_sha256
    ):
        raise ExogenousProductionError(
            "Les roles shadow du bundle different de leur provenance auto-contenue."
        )
    experiment = _json(
        bundle.experiment_manifest_path,
        label="manifeste d'experience",
    )
    try:
        frame = pd.read_csv(predictions_path)
    except Exception as exc:
        raise ExogenousProductionError(
            f"Predictions shadow illisibles: {predictions_path}."
        ) from exc
    # Passing the archived path, rather than only the parsed mapping, forces
    # the validator to rehash every raw/incumbent/corrector/OOF lineage copy
    # embedded in the promotion bundle.
    validated_manifest = validate_final_shadow_manifest(
        manifest_path,
        experiment_manifest=experiment,
        zone=bundle.zone,
        expected_rows=len(frame),
    )
    if validated_manifest.get("predictions_sha256") != predictions_sha256:
        raise ExogenousProductionError(
            "Le SHA des predictions shadow ne correspond pas au manifeste."
        )
    observed = validated_manifest.get("observed_governance_evidence")
    if not isinstance(observed, Mapping) or (
        observed.get("relative_path") != predictions_path.name
        or observed.get("sha256") != predictions_sha256
        or observed.get("rows") != len(frame)
    ):
        raise ExogenousProductionError(
            "Le manifeste shadow ne designe pas exactement son evidence observee."
        )
    expected_pipeline: Mapping[str, object] = {
        "format_version": 4,
        "kind": "chronos2_exogenous_final_pipeline_shadow",
        "comparison_scope": "paired_operational_final_pipelines",
        "baseline_output_stage": "residual_corrected",
        "candidate_output_stage": OUTPUT_MODEL,
        "residual_corrector_applied": True,
    }
    pipeline_mismatches = [
        key for key, expected in expected_pipeline.items()
        if type(validated_manifest.get(key)) is not type(expected)
        or validated_manifest.get(key) != expected
    ]
    if pipeline_mismatches:
        raise ExogenousProductionError(
            "Le shadow promu n'evalue pas le pipeline final: "
            + ", ".join(pipeline_mismatches)
        )
    shadow_corrector = validated_manifest.get("residual_corrector")
    if (
        not isinstance(shadow_corrector, Mapping)
        or shadow_corrector.get("sha256")
        != _sha256(bundle.residual_corrector_path)
    ):
        raise ExogenousProductionError(
            "Le correcteur du shadow final differe du correcteur promu."
        )

    shadow_columns = (
        "delivery_start_utc",
        "forecast_origin_utc",
        "actual",
        *(f"baseline_{quantile}" for quantile in QUANTILES),
        *(f"candidate_{quantile}" for quantile in QUANTILES),
    )
    if tuple(frame.columns) != shadow_columns:
        raise ExogenousProductionError(
            "Schema exact de l'evidence shadow finale invalide."
        )
    delivery = pd.DatetimeIndex(
        pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="coerce")
    )
    origins = pd.DatetimeIndex(
        pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="coerce")
    )
    if delivery.isna().any() or origins.isna().any():
        raise ExogenousProductionError("Timestamps shadow invalides.")
    if delivery.has_duplicates:
        raise ExogenousProductionError("Heures shadow dupliquees.")
    if bool(
        (delivery.minute != 0).any()
        or (delivery.second != 0).any()
        or (delivery.microsecond != 0).any()
    ):
        raise ExogenousProductionError("Heures shadow non alignees.")
    numeric_columns = [
        "actual",
        *(f"baseline_{quantile}" for quantile in QUANTILES),
        *(f"candidate_{quantile}" for quantile in QUANTILES),
    ]
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ExogenousProductionError("Valeurs shadow absentes ou non finies.")
    for prefix in ("baseline", "candidate"):
        quantiles = numeric.loc[
            :, [f"{prefix}_{quantile}" for quantile in QUANTILES]
        ].to_numpy(dtype=float)
        if bool(
            ((quantiles[:, 0] > quantiles[:, 1])
             | (quantiles[:, 1] > quantiles[:, 2])).any()
        ):
            raise ExogenousProductionError(
                f"Quantiles shadow croises pour {prefix}."
            )

    temporal = validated_manifest.get("temporal_attachment_audit")
    assert isinstance(temporal, list)  # guaranteed by validate_shadow_manifest
    temporal_keys = {
        (
            pd.Timestamp(pd.to_datetime(row["delivery_start_utc"], utc=True)),
            pd.Timestamp(pd.to_datetime(row["forecast_origin_utc"], utc=True)),
        )
        for row in temporal
    }
    prediction_keys = set(zip(delivery, origins, strict=True))
    if temporal_keys != prediction_keys:
        raise ExogenousProductionError(
            "Les lignes shadow different de leur preuve temporelle horaire."
        )

    schema = _json(bundle.schema_path, label="schema du candidat")
    timezone_name = str(schema.get("timezone", "")).strip()
    try:
        ZoneInfo(timezone_name)
    except Exception as exc:
        raise ExogenousProductionError(
            "Timezone du schema shadow absente ou invalide."
        ) from exc
    order = np.argsort(delivery.asi8)
    frame = frame.iloc[order].reset_index(drop=True)
    numeric = numeric.iloc[order].reset_index(drop=True)
    delivery = delivery[order]
    origins = origins[order]
    local_days = tuple(dict.fromkeys(delivery.tz_convert(timezone_name).date))
    if not local_days:
        raise ExogenousProductionError("L'evidence shadow finale est vide.")
    for day_local in local_days:
        mask = delivery.tz_convert(timezone_name).date == day_local
        observed_hours = delivery[mask]
        expected_hours = _expected_civil_delivery_index(
            day_local,
            timezone_name=timezone_name,
        )
        if not observed_hours.equals(expected_hours):
            raise ExogenousProductionError(
                f"Grille shadow DST incomplete pour {day_local}."
            )
        expected_origin = pd.Timestamp(
            datetime.combine(
                day_local - timedelta(days=1),
                datetime.min.replace(hour=8).time(),
            ),
            tz=ZoneInfo(timezone_name),
        ).tz_convert("UTC")
        if not bool((origins[mask] == expected_origin).all()):
            raise ExogenousProductionError(
                f"Origine shadow differente de D-1 08:00 pour {day_local}."
            )

    issued_reference = validated_manifest.get("issued_shadow_history")
    if not isinstance(issued_reference, Mapping):
        raise ExogenousProductionError(
            "Le manifeste shadow final ne contient pas l'historique emis."
        )
    issued_relative_text = issued_reference.get("relative_path")
    if not isinstance(issued_relative_text, str) or not issued_relative_text.strip():
        raise ExogenousProductionError(
            "Chemin de l'historique shadow emis absent."
        )
    issued_relative = Path(issued_relative_text)
    if issued_relative.is_absolute() or ".." in issued_relative.parts:
        raise ExogenousProductionError(
            "Chemin de l'historique shadow emis hors bundle."
        )
    issued_history_path = (lineage_root / issued_relative).resolve()
    lineage_resolved = lineage_root.resolve()
    if (
        issued_history_path == lineage_resolved
        or lineage_resolved not in issued_history_path.parents
        or not issued_history_path.is_file()
    ):
        raise ExogenousProductionError(
            "Historique shadow emis absent du bundle auto-contenu."
        )
    issued_history_sha256 = _sha256(issued_history_path)
    if (
        not _is_sha256(issued_reference.get("sha256"))
        or issued_reference["sha256"] != issued_history_sha256
    ):
        raise ExogenousProductionError(
            "Le SHA de l'historique shadow emis est divergent."
        )
    try:
        issued_frame = pd.read_csv(issued_history_path)
    except Exception as exc:
        raise ExogenousProductionError(
            f"Historique shadow emis illisible: {issued_history_path}."
        ) from exc
    if tuple(issued_frame.columns) != PROMOTED_SHADOW_ISSUED_COLUMNS:
        raise ExogenousProductionError(
            "Schema exact de l'historique shadow emis invalide."
        )
    if type(issued_reference.get("rows")) is not int or (
        issued_reference["rows"] != len(issued_frame)
        or len(issued_frame) == 0
    ):
        raise ExogenousProductionError(
            "Nombre de lignes de l'historique shadow emis divergent."
        )
    if (
        issued_reference.get("candidate_output_stage") != OUTPUT_MODEL
        or issued_reference.get("actual_nullable") is not True
    ):
        raise ExogenousProductionError(
            "Contrat de sortie de l'historique shadow emis invalide."
        )

    issued_delivery = pd.DatetimeIndex(
        pd.to_datetime(
            issued_frame["delivery_start_utc"], utc=True, errors="coerce"
        )
    )
    issued_origins = pd.DatetimeIndex(
        pd.to_datetime(
            issued_frame["forecast_origin_utc"], utc=True, errors="coerce"
        )
    )
    issued_created = pd.DatetimeIndex(
        pd.to_datetime(
            issued_frame["forecast_created_at_utc"], utc=True, errors="coerce"
        )
    )
    if (
        issued_delivery.isna().any()
        or issued_origins.isna().any()
        or issued_created.isna().any()
    ):
        raise ExogenousProductionError(
            "Timestamps de l'historique shadow emis invalides."
        )
    if issued_delivery.has_duplicates:
        raise ExogenousProductionError(
            "Heures de l'historique shadow emis dupliquees."
        )
    if bool(
        (issued_delivery.minute != 0).any()
        or (issued_delivery.second != 0).any()
        or (issued_delivery.microsecond != 0).any()
    ):
        raise ExogenousProductionError(
            "Heures de l'historique shadow emis non alignees."
        )
    issued_quantile_columns = [
        f"candidate_{quantile}" for quantile in QUANTILES
    ]
    issued_numeric = issued_frame.loc[:, issued_quantile_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    issued_quantiles = issued_numeric.to_numpy(dtype=float)
    if not np.isfinite(issued_quantiles).all() or bool(
        ((issued_quantiles[:, 0] > issued_quantiles[:, 1])
         | (issued_quantiles[:, 1] > issued_quantiles[:, 2])).any()
    ):
        raise ExogenousProductionError(
            "Quantiles de l'historique shadow emis absents/non finis/croises."
        )
    issued_actual_source = issued_frame["actual"]
    issued_actual = pd.to_numeric(issued_actual_source, errors="coerce")
    if bool(
        (issued_actual_source.notna() & issued_actual.isna()).any()
        or np.isinf(issued_actual.to_numpy(dtype=float)).any()
    ):
        raise ExogenousProductionError(
            "Actual nullable de l'historique shadow emis invalide."
        )

    target_columns = schema.get("target_columns")
    if not isinstance(target_columns, list) or len(target_columns) != 1:
        raise ExogenousProductionError(
            "Le schema shadow doit declarer une cible unique."
        )
    expected_target = str(target_columns[0]).strip()
    if (
        not expected_target
        or not issued_frame["item_id"].astype(str).str.strip().eq(bundle.zone).all()
        or not issued_frame["target_column"]
        .astype(str)
        .str.strip()
        .eq(expected_target)
        .all()
    ):
        raise ExogenousProductionError(
            "Zone ou cible de l'historique shadow emis divergente."
        )
    expected_checkpoint_sha = experiment.get("checkpoint_sha256")
    if not _is_sha256(expected_checkpoint_sha) or not issued_frame[
        "checkpoint_sha256"
    ].astype(str).eq(str(expected_checkpoint_sha)).all():
        raise ExogenousProductionError(
            "Checkpoint de l'historique shadow emis divergent."
        )
    for column in (
        "input_contract_sha256",
        "panel_contract_sha256",
        "forecast_record_sha256",
    ):
        if not issued_frame[column].map(_is_sha256).all():
            raise ExogenousProductionError(
                f"Empreinte invalide dans l'historique shadow emis: {column}."
            )
    if not bool(
        (
            (issued_created >= issued_origins)
            & (issued_created <= issued_origins + pd.Timedelta(hours=4))
        ).all()
    ):
        raise ExogenousProductionError(
            "Historique shadow emis hors fenetre prospective."
        )

    issued_order = np.argsort(issued_delivery.asi8)
    issued_frame = issued_frame.iloc[issued_order].reset_index(drop=True)
    issued_numeric = issued_numeric.iloc[issued_order].reset_index(drop=True)
    issued_actual = issued_actual.iloc[issued_order].reset_index(drop=True)
    issued_delivery = issued_delivery[issued_order]
    issued_origins = issued_origins[issued_order]
    issued_created = issued_created[issued_order]
    issued_local_days = tuple(
        dict.fromkeys(issued_delivery.tz_convert(timezone_name).date)
    )
    for day_local in issued_local_days:
        mask = issued_delivery.tz_convert(timezone_name).date == day_local
        expected_hours = _expected_civil_delivery_index(
            day_local,
            timezone_name=timezone_name,
        )
        if not issued_delivery[mask].equals(expected_hours):
            raise ExogenousProductionError(
                f"Grille shadow emise DST incomplete pour {day_local}."
            )
        expected_origin = pd.Timestamp(
            datetime.combine(
                day_local - timedelta(days=1),
                datetime.min.replace(hour=8).time(),
            ),
            tz=ZoneInfo(timezone_name),
        ).tz_convert("UTC")
        if not bool((issued_origins[mask] == expected_origin).all()):
            raise ExogenousProductionError(
                f"Origine shadow emise differente de D-1 08:00 pour {day_local}."
            )

    declared_days = issued_reference.get("delivery_days")
    expected_day_strings = tuple(day.isoformat() for day in issued_local_days)
    if not isinstance(declared_days, list) or tuple(declared_days) != expected_day_strings:
        raise ExogenousProductionError(
            "Jours declares de l'historique shadow emis divergents."
        )
    try:
        declared_first = pd.Timestamp(issued_reference["first_delivery_utc"])
        declared_last = pd.Timestamp(issued_reference["last_delivery_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExogenousProductionError(
            "Bornes de l'historique shadow emis invalides."
        ) from exc
    if (
        declared_first.tzinfo is None
        or declared_last.tzinfo is None
        or declared_first.tz_convert("UTC") != issued_delivery[0]
        or declared_last.tz_convert("UTC") != issued_delivery[-1]
    ):
        raise ExogenousProductionError(
            "Bornes de l'historique shadow emis divergentes."
        )

    # The observed scoring proof must be an exact subset of the issued
    # history.  Its candidate is already residual-corrected; no correction is
    # applied here or later while the bridge is assembled.
    observed_frame = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "forecast_origin_utc": origins,
            "actual": numeric["actual"].to_numpy(dtype=float),
            **{
                f"candidate_{quantile}": numeric[
                    f"candidate_{quantile}"
                ].to_numpy(dtype=float)
                for quantile in QUANTILES
            },
        }
    )
    issued_for_overlap = pd.DataFrame(
        {
            "delivery_start_utc": issued_delivery,
            "forecast_origin_utc": issued_origins,
            "actual": issued_actual.to_numpy(dtype=float),
            **{
                f"candidate_{quantile}": issued_numeric[
                    f"candidate_{quantile}"
                ].to_numpy(dtype=float)
                for quantile in QUANTILES
            },
        }
    )
    overlap = observed_frame.merge(
        issued_for_overlap,
        on=["delivery_start_utc", "forecast_origin_utc"],
        how="left",
        suffixes=("_observed", "_issued"),
        validate="one_to_one",
    )
    if len(overlap) != len(observed_frame) or overlap[
        "candidate_q50_issued"
    ].isna().any():
        raise ExogenousProductionError(
            "L'historique emis ne couvre pas toute la preuve shadow observee."
        )
    for column in ("actual", *issued_quantile_columns):
        if not np.allclose(
            pd.to_numeric(
                overlap[f"{column}_observed"], errors="coerce"
            ).to_numpy(dtype=float),
            pd.to_numeric(
                overlap[f"{column}_issued"], errors="coerce"
            ).to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ExogenousProductionError(
                "Preuve shadow observee et historique emis divergents."
            )

    output = pd.DataFrame(
        {
            "delivery_start_utc": issued_delivery,
            "forecast_origin_utc": issued_origins,
        }
    )
    for quantile in QUANTILES:
        output[f"{OUTPUT_MODEL}__{quantile}"] = issued_numeric[
            f"candidate_{quantile}"
        ].to_numpy(dtype=float)
    return PromotedShadowHistory(
        predictions=output,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        predictions_sha256=predictions_sha256,
        manifest_sha256=manifest_sha256,
        delivery_days=expected_day_strings,
        issued_history_path=issued_history_path,
        issued_history_sha256=issued_history_sha256,
    )


def _read_rolling(
    path: Path, *, zone: str, experiment: Mapping[str, Any]
) -> pd.DataFrame:
    _validate_final_pipeline_evidence(experiment, rolling_path=path)
    suffixes = "".join(path.suffixes).lower()
    frame = pd.read_parquet(path) if suffixes.endswith(".parquet") else pd.read_csv(path)
    required = {
        "delivery_start_utc",
        "actual",
        *(f"baseline_{quantile}" for quantile in QUANTILES),
        *(f"candidate_{quantile}" for quantile in QUANTILES),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ExogenousProductionError(
            "Evidence rolling incompatible: " + ", ".join(missing)
        )
    output = pd.DataFrame(
        {
            "delivery_start_utc": frame["delivery_start_utc"],
            "actual": frame["actual"],
        }
    )
    if "forecast_origin_utc" in frame:
        output["forecast_origin_utc"] = frame["forecast_origin_utc"]
    for quantile in QUANTILES:
        output[f"residual_corrected__{quantile}"] = frame[f"baseline_{quantile}"]
        output[f"{OUTPUT_MODEL}__{quantile}"] = frame[f"candidate_{quantile}"]
    output["zone"] = zone
    return output


def _seal_output(directory: Path) -> None:
    artifacts: list[dict[str, object]] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == "artifact_checksums.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "role": "run_artifact",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        directory / "artifact_checksums.json",
        {"algorithm": "sha256", "artifacts": artifacts},
    )


def _validate_sealed_run_files(directory: Path) -> None:
    checksum_path = directory / "artifact_checksums.json"
    payload = _json(checksum_path, label="checksums du run candidat")
    if payload.get("algorithm") != "sha256" or not isinstance(
        payload.get("artifacts"), list
    ):
        raise ExogenousProductionError("Contrat de checksums du run candidat invalide.")
    declared: set[str] = set()
    for raw in payload["artifacts"]:
        if not isinstance(raw, Mapping):
            raise ExogenousProductionError("Entree de checksum du run invalide.")
        relative = str(raw.get("path", ""))
        candidate = (directory / relative).resolve()
        if (
            not relative
            or relative in declared
            or directory not in candidate.parents
            or not candidate.is_file()
        ):
            raise ExogenousProductionError(
                f"Artefact du run candidat absent/invalide: {relative!r}."
            )
        declared.add(relative)
        if (
            raw.get("role") != "run_artifact"
            or raw.get("size_bytes") != candidate.stat().st_size
            or raw.get("sha256") != _sha256(candidate)
        ):
            raise ExogenousProductionError(
                f"Artefact du run candidat modifie: {relative}."
            )
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != checksum_path.name
    }
    if actual != declared:
        raise ExogenousProductionError(
            "Le contenu du run candidat differe de son manifeste de checksums."
        )


def validate_registered_candidate_run(
    *,
    registry_path: str | Path,
    alias: str,
    output_directory: str | Path,
    delivery_day: str,
    expected_zone: str,
) -> ExogenousRunResult:
    """Revalidate and reuse one immutable daily candidate without inference."""

    zone = str(expected_zone).strip().upper()
    bundle = load_registered_bundle(
        registry_path=registry_path,
        alias=alias,
        expected_zone=zone,
    )
    output = Path(output_directory).expanduser().resolve()
    if not output.is_dir():
        raise ExogenousProductionError(f"Run candidat absent: {output}.")
    _validate_sealed_run_files(output)
    manifest_path = output / "run_manifest.json"
    manifest = _json(manifest_path, label="manifeste du run candidat")
    expected = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "run_type": "opt_in_exogenous_day_ahead",
        "forecast_status": "conditional_promoted_candidate",
        "candidate_model": OUTPUT_MODEL,
        "native_model": OUTPUT_MODEL,
        "baseline_model": "residual_corrected",
        "zone": zone,
        "delivery_day": delivery_day,
        "promotion_alias": alias,
        "promotion_candidate_id": bundle.candidate_id,
        "promotion_bundle_manifest_sha256": bundle.bundle_manifest_sha256,
        "promotion_bundle_checksums_sha256": bundle.artifact_checksums_sha256,
        "production_pit_evidence": True,
        "residual_corrector_recalibrated_oof": True,
        "storm_used_for_prediction": False,
        "mkonline_used_for_prediction": False,
        "launcher_mode_agnostic": True,
    }
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise ExogenousProductionError(
            "Identite du run candidat divergente: " + ", ".join(mismatches)
        )
    forecast_path = output / f"forecast_hourly_{zone.lower()}.csv"
    backtest_path = output / "backtest_hourly_oof.csv.gz"
    forecast = pd.read_csv(forecast_path)
    required = {
        "delivery_start_utc",
        "forecast_origin_utc",
        *(f"{BASE_MODEL}__{quantile}" for quantile in QUANTILES),
        *(f"{OUTPUT_MODEL}__{quantile}" for quantile in QUANTILES),
    }
    missing = sorted(required.difference(forecast.columns))
    if missing:
        raise ExogenousProductionError(
            "Forecast candidat incomplet: " + ", ".join(missing)
        )
    delivery = pd.DatetimeIndex(
        pd.to_datetime(forecast["delivery_start_utc"], utc=True, errors="raise")
    )
    schema = _json(bundle.schema_path, label="schema du candidat")
    timezone_name = str(schema["timezone"])
    expected_delivery = _expected_civil_delivery_index(
        delivery_day,
        timezone_name=timezone_name,
    )
    if not delivery.equals(expected_delivery):
        raise ExogenousProductionError("Timeline du forecast candidat invalide.")
    values = forecast.loc[
        :,
        [
            *(f"{BASE_MODEL}__{quantile}" for quantile in QUANTILES),
            *(f"{OUTPUT_MODEL}__{quantile}" for quantile in QUANTILES),
        ],
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ExogenousProductionError("Forecast candidat non fini.")
    for model in (BASE_MODEL, OUTPUT_MODEL):
        quantiles = values[
            [f"{model}__{quantile}" for quantile in QUANTILES]
        ].to_numpy(dtype=float)
        if bool(((quantiles[:, 0] > quantiles[:, 1]) | (quantiles[:, 1] > quantiles[:, 2])).any()):
            raise ExogenousProductionError(
                f"Quantiles croises dans le run candidat: {model}."
            )
    # The sealed checksum is necessary but not sufficient: make sure the
    # rolling evidence remains parseable before exposing it to Statistics.
    backtest = pd.read_csv(backtest_path)
    rolling_required = {
        "delivery_start_utc",
        "actual",
        *(f"{OUTPUT_MODEL}__{quantile}" for quantile in QUANTILES),
    }
    if rolling_required.difference(backtest.columns):
        raise ExogenousProductionError("Backtest candidat scelle incomplet.")
    return ExogenousRunResult(
        output_directory=output,
        forecast_path=forecast_path,
        backtest_path=backtest_path,
        manifest_path=manifest_path,
        delivery_day=delivery_day,
        zone=zone,
    )


def run_registered_candidate(
    *,
    registry_path: str | Path,
    alias: str,
    live_panel_path: str | Path,
    live_panel_audit_path: str | Path,
    delivery_day: str,
    output_directory: str | Path,
    device_map: str = "auto",
    pipeline_loader: PipelineLoader | None = None,
) -> ExogenousRunResult:
    """Run a real forecast only through an explicit registered alias."""

    bundle = load_registered_bundle(
        registry_path=registry_path, alias=alias, expected_zone=None
    )
    schema = _json(bundle.schema_path, label="schema du candidat")
    experiment = _json(
        bundle.experiment_manifest_path, label="manifeste d'experience"
    )
    corrector = _validate_corrector(
        bundle.residual_corrector_path, experiment=experiment
    )
    panel_path = Path(live_panel_path).expanduser().resolve()
    audit_path = Path(live_panel_audit_path).expanduser().resolve()
    context, horizon, origin, target_cache_verification = _validate_live_panel(
        panel_path,
        audit_path=audit_path,
        schema=schema,
        bundle=bundle,
        experiment=experiment,
        delivery_day=delivery_day,
    )
    loader = pipeline_loader or _default_pipeline_loader
    pipeline = loader(
        bundle.checkpoint_path,
        {
            "device_map": device_map,
            "local_files_only": True,
            # PEFT delegates the adapter base architecture import to the
            # Transformers dynamic loader.  Keep the allowlist identical to
            # the verified fine-tuning checkpoint loader.
            "import_allowlist": list(ADAPTER_IMPORT_ALLOWLIST),
        },
    )
    base = _base_prediction(
        pipeline=pipeline, context=context, horizon=horizon, schema=schema
    )
    corrected, shift = _apply_residual_corrector(
        base, horizon=horizon, corrector=corrector, schema=schema
    )
    if corrected.shape[0] != 1:
        raise ExogenousProductionError("Runtime v1 per_zone exige une cible unique.")
    timestamp_column = str(schema["timestamp_column"])
    forecast = pd.DataFrame(
        {
            "delivery_start_utc": pd.DatetimeIndex(horizon[timestamp_column]),
            "forecast_origin_utc": origin,
            "zone": bundle.zone,
            "residual_shift_eur_mwh": shift,
        }
    )
    for index, quantile in enumerate(QUANTILES):
        forecast[f"{BASE_MODEL}__{quantile}"] = base[0, index, :]
        forecast[f"{OUTPUT_MODEL}__{quantile}"] = corrected[0, index, :]

    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise ExogenousProductionError(
            f"Sortie immutable deja existante: {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    try:
        staging.mkdir(parents=False, exist_ok=False)
        forecast_name = f"forecast_hourly_{bundle.zone.lower()}.csv"
        forecast.to_csv(staging / forecast_name, index=False)
        backtest = _read_rolling(
            bundle.rolling_predictions_path,
            zone=bundle.zone,
            experiment=experiment,
        )
        backtest.to_csv(staging / "backtest_hourly_oof.csv.gz", index=False)
        panel_copy_name = "live_panel" + "".join(panel_path.suffixes)
        audit_copy_name = "live_panel.audit.json"
        shutil.copy2(panel_path, staging / panel_copy_name)
        shutil.copy2(audit_path, staging / audit_copy_name)
        shutil.copy2(bundle.schema_path, staging / "exogenous_schema.json")
        shutil.copy2(
            bundle.residual_corrector_path,
            staging / "exogenous_residual_corrector.json",
        )
        shutil.copy2(
            bundle.oof_audit_path,
            staging / "exogenous_residual_corrector_oof_audit.json",
        )
        source_checksums = bundle.bundle_path / "artifact_checksums.json"
        shutil.copy2(source_checksums, staging / "promotion_bundle_checksums.json")
        manifest = {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "run_type": "opt_in_exogenous_day_ahead",
            "forecast_status": "conditional_promoted_candidate",
            "candidate_model": OUTPUT_MODEL,
            "native_model": OUTPUT_MODEL,
            "baseline_model": "residual_corrected",
            "zone": bundle.zone,
            "delivery_day": delivery_day,
            "forecast_origin_utc": origin.isoformat(),
            "forecast_hours": len(forecast),
            "promotion_alias": alias,
            "promotion_candidate_id": bundle.candidate_id,
            "promotion_bundle_path": str(bundle.bundle_path),
            "promotion_bundle_manifest_sha256": bundle.bundle_manifest_sha256,
            "promotion_bundle_checksums_sha256": bundle.artifact_checksums_sha256,
            "live_panel_path": panel_copy_name,
            "live_panel_source_path": str(panel_path),
            "live_panel_sha256": _sha256(panel_path),
            "live_panel_audit_path": audit_copy_name,
            "live_panel_audit_source_path": str(audit_path),
            "live_panel_audit_sha256": _sha256(audit_path),
            "target_cache_verification": target_cache_verification,
            "production_pit_evidence": True,
            "residual_corrector_recalibrated_oof": True,
            "storm_used_for_prediction": False,
            "mkonline_used_for_prediction": False,
            "incumbent_modified": False,
            "launcher_mode_agnostic": True,
            "explicit_opt_in_required": True,
            "sha256_manifest": "artifact_checksums.json",
        }
        _write_json(staging / "run_manifest.json", manifest)
        _seal_output(staging)
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return ExogenousRunResult(
        output_directory=output,
        forecast_path=output / f"forecast_hourly_{bundle.zone.lower()}.csv",
        backtest_path=output / "backtest_hourly_oof.csv.gz",
        manifest_path=output / "run_manifest.json",
        delivery_day=delivery_day,
        zone=bundle.zone,
    )


__all__ = [
    "BASE_MODEL",
    "ExogenousProductionError",
    "ExogenousRunResult",
    "OUTPUT_MODEL",
    "PromotedBundle",
    "PromotedShadowHistory",
    "load_promoted_shadow_history",
    "load_registered_bundle",
    "fit_oof_residual_corrector",
    "register_promoted_bundle",
    "run_registered_candidate",
    "validate_registered_candidate_run",
    "validate_promoted_bundle",
]
