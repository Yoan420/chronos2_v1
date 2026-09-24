"""Leakage-safe rolling evaluation for the Chronos-2 exogenous LoRA POC.

The evaluator deliberately compares the frozen base model and the LoRA
adapter with byte-identical, origin-by-origin inputs.  It does not call the
residual corrector, MKOnline or Storm.  Civil delivery days keep their real
23/24/25-hour shape; no interpolation or duplicated clock hour is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .lora_finetune import (
    EVALUATION_COLUMNS,
    ExogenousFineTuneConfig,
    ExogenousFineTuneError,
    OriginSplit,
    PipelineLoader,
    bind_resolved_evaluation_panel,
    load_checkpoint,
    load_config,
    publish_evaluation_evidence,
    read_panel,
    resolve_local_model_source,
    validate_panel,
    verify_bundle,
)


QUANTILE_LEVELS = (0.1, 0.5, 0.9)
BASELINE_LABEL = "Chronos-2 base (mêmes entrées exogènes)"
CANDIDATE_LABEL = "Chronos-2 + adaptateur LoRA exogène"


class ExogenousEvaluationError(ExogenousFineTuneError):
    """Raised when paired inference or reporting violates the POC contract."""


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Paths and headline metrics published by :func:`run_evaluation`."""

    evidence_path: Path
    daily_path: Path
    metrics_path: Path
    report_path: Path
    manifest_path: Path
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class ShadowArtifacts:
    """Append-only shadow journal and optional observed governance evidence."""

    journal_path: Path
    observed_evidence_path: Path | None
    manifest_path: Path | None
    appended_rows: int


SHADOW_INPUT_COLUMNS = (
    "delivery_start_utc",
    "forecast_origin_utc",
    "item_id",
    "target_column",
    "actual",
    "baseline_q10",
    "baseline_q50",
    "baseline_q90",
    "candidate_q10",
    "candidate_q50",
    "candidate_q90",
    "input_contract_sha256",
    "checkpoint_sha256",
    "panel_sha256",
    "panel_audit_sha256",
    "panel_contract_sha256",
    "panel_created_at_utc",
    "panel_provenance_json",
)

SHADOW_JOURNAL_COLUMNS = SHADOW_INPUT_COLUMNS + (
    "captured_at_utc",
    "revision",
    "record_kind",
    "previous_record_sha256",
    "record_sha256",
)
SHADOW_MAX_FORECAST_LATENCY_HOURS = 4
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> pd.Timestamp:
    """Single non-injectable production clock seam, monkeypatched only in tests."""

    return pd.Timestamp.now(tz="UTC")


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value).strip().lower() if isinstance(value, str) else ""
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ExogenousEvaluationError(f"Shadow: {label} doit etre un SHA-256.")
    return digest


def _normalise_hash_mapping(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ExogenousEvaluationError(
            f"Shadow: {label} doit etre une table de hashes non vide."
        )
    result: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        name = str(raw_name).strip()
        if not name or name in result:
            raise ExogenousEvaluationError(f"Shadow: cle invalide dans {label}.")
        result[name] = _require_sha256(raw_digest, label=f"{label}.{name}")
    return dict(sorted(result.items()))


def _normalise_panel_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalise one immutable forecast-panel identity."""

    required = {
        "schema_version",
        "purpose",
        "panel_sha256",
        "panel_audit_sha256",
        "panel_created_at_utc",
        "zone",
        "delivery_day",
        "forecast_origin_utc",
        "forecast_origin_timezone",
        "delivery_timezone",
        "pack",
        "production_pit_evidence",
        "source_hashes",
        "source_audit_hashes",
        "source_cutoff_timezones",
        "target_source_sha256",
        "horizon_actuals_present",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ExogenousEvaluationError(
            f"Shadow: identite panel incomplete: {missing}."
        )
    if value.get("schema_version") != 1 or value.get("purpose") != "prospective_shadow":
        raise ExogenousEvaluationError("Shadow: contrat/purpose du panel invalide.")
    zone = str(value.get("zone", "")).strip().upper()
    if re.fullmatch(r"[A-Z]{2,8}", zone) is None:
        raise ExogenousEvaluationError("Shadow: zone du panel invalide.")
    try:
        delivery_day = pd.Timestamp(str(value["delivery_day"])).date().isoformat()
        if str(value["delivery_day"]) != delivery_day:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ExogenousEvaluationError("Shadow: delivery_day du panel invalide.") from exc
    created = pd.to_datetime(value["panel_created_at_utc"], utc=True, errors="coerce")
    origin = pd.to_datetime(value["forecast_origin_utc"], utc=True, errors="coerce")
    if pd.isna(created) or pd.isna(origin):
        raise ExogenousEvaluationError("Shadow: timestamps de provenance panel invalides.")
    forecast_timezone = str(value["forecast_origin_timezone"]).strip()
    delivery_timezone = str(value["delivery_timezone"]).strip()
    try:
        ZoneInfo(forecast_timezone)
        ZoneInfo(delivery_timezone)
    except Exception as exc:
        raise ExogenousEvaluationError("Shadow: timezone de provenance invalide.") from exc
    if type(value["production_pit_evidence"]) is not bool or type(
        value["horizon_actuals_present"]
    ) is not bool:
        raise ExogenousEvaluationError(
            "Shadow: drapeaux de provenance panel non booleens."
        )
    source_hashes = _normalise_hash_mapping(
        value["source_hashes"], label="source_hashes"
    )
    source_audit_hashes = _normalise_hash_mapping(
        value["source_audit_hashes"], label="source_audit_hashes"
    )
    if not set(source_audit_hashes).issubset(source_hashes):
        raise ExogenousEvaluationError(
            "Shadow: source_audit_hashes contient une source absente du panel."
        )
    raw_timezones = value["source_cutoff_timezones"]
    if not isinstance(raw_timezones, Mapping) or set(map(str, raw_timezones)) != set(
        source_hashes
    ):
        raise ExogenousEvaluationError(
            "Shadow: source_cutoff_timezones ne couvre pas exactement les sources."
        )
    source_timezones: dict[str, str] = {}
    for name in source_hashes:
        timezone_name = str(raw_timezones[name]).strip()
        try:
            ZoneInfo(timezone_name)
        except Exception as exc:
            raise ExogenousEvaluationError(
                f"Shadow: timezone source invalide pour {name}."
            ) from exc
        source_timezones[name] = timezone_name
    external_sources = set(source_hashes).difference({"deterministic_calendar"})
    if bool(value["production_pit_evidence"]) and set(
        source_audit_hashes
    ) != external_sources:
        raise ExogenousEvaluationError(
            "Shadow: preuve de production sans sidecar pour chaque source exogene."
        )
    normalised = {
        "schema_version": 1,
        "purpose": "prospective_shadow",
        "panel_sha256": _require_sha256(
            value["panel_sha256"], label="panel_sha256"
        ),
        "panel_audit_sha256": _require_sha256(
            value["panel_audit_sha256"], label="panel_audit_sha256"
        ),
        "panel_created_at_utc": pd.Timestamp(created).isoformat(),
        "zone": zone,
        "delivery_day": delivery_day,
        "forecast_origin_utc": pd.Timestamp(origin).isoformat(),
        "forecast_origin_timezone": forecast_timezone,
        "delivery_timezone": delivery_timezone,
        "pack": str(value["pack"]).strip(),
        "production_pit_evidence": bool(value["production_pit_evidence"]),
        "source_hashes": source_hashes,
        "source_audit_hashes": source_audit_hashes,
        "source_cutoff_timezones": dict(sorted(source_timezones.items())),
        "target_source_sha256": _require_sha256(
            value["target_source_sha256"], label="target_source_sha256"
        ),
        "horizon_actuals_present": bool(value["horizon_actuals_present"]),
    }
    digest = hashlib.sha256(
        _canonical_json_text(normalised).encode("utf-8")
    ).hexdigest()
    declared = value.get("panel_contract_sha256")
    if declared is not None and _require_sha256(
        declared, label="panel_contract_sha256"
    ) != digest:
        raise ExogenousEvaluationError(
            "Shadow: panel_contract_sha256 divergent de la provenance."
        )
    return {**normalised, "panel_contract_sha256": digest}


def _load_shadow_panel_evidence(
    panel_path: Path,
    audit_path: Path,
    *,
    audit_payload: Mapping[str, Any],
    config: ExogenousFineTuneConfig,
    zone: str,
    origin: pd.Timestamp,
) -> dict[str, Any]:
    """Bind an explicit panel sidecar to bytes and causal shadow semantics."""

    if not audit_path.is_file():
        raise ExogenousEvaluationError(
            f"Shadow: sidecar audit du panel obligatoire et absent: {audit_path}."
        )
    if audit_payload.get("purpose") != "prospective_shadow":
        raise ExogenousEvaluationError(
            "Shadow: le sidecar doit declarer purpose=prospective_shadow."
        )
    declared_panel_hash = _require_sha256(
        audit_payload.get("panel_sha256"), label="audit.panel_sha256"
    )
    actual_panel_hash = _sha256_file(panel_path)
    if declared_panel_hash != actual_panel_hash:
        raise ExogenousEvaluationError(
            "Shadow: le sidecar audit n'est pas lie aux octets du panel."
        )
    zones = audit_payload.get("zones")
    canonical_zone = str(zone).strip().upper()
    if not isinstance(zones, list) or [str(item).strip().upper() for item in zones] != [
        canonical_zone
    ]:
        raise ExogenousEvaluationError(
            "Shadow: le sidecar doit couvrir exactement la zone demandee."
        )
    if audit_payload.get("layout") != "per_zone":
        raise ExogenousEvaluationError("Shadow: seul le layout per_zone est admis.")
    declared_origin = pd.to_datetime(
        audit_payload.get("forecast_origin_utc"), utc=True, errors="coerce"
    )
    if pd.isna(declared_origin) or pd.Timestamp(declared_origin) != pd.Timestamp(origin):
        raise ExogenousEvaluationError(
            "Shadow: forecast_origin_utc du sidecar ne correspond pas au panel."
        )
    expected_day = (
        pd.Timestamp(origin).tz_convert(config.timezone) + pd.DateOffset(days=1)
    ).date().isoformat()
    if audit_payload.get("delivery_day") != expected_day:
        raise ExogenousEvaluationError(
            "Shadow: delivery_day du sidecar ne correspond pas a l'origine D-1."
        )
    if audit_payload.get("forecast_origin_timezone") != config.timezone:
        raise ExogenousEvaluationError(
            "Shadow: forecast_origin_timezone incompatible avec la configuration."
        )
    created = pd.to_datetime(
        audit_payload.get("created_at_utc"), utc=True, errors="coerce"
    )
    if pd.isna(created) or pd.Timestamp(created) > _utc_now():
        raise ExogenousEvaluationError(
            "Shadow: created_at_utc du sidecar invalide ou futur."
        )
    horizon_actuals = audit_payload.get("horizon_actuals_present")
    if type(horizon_actuals) is not bool:
        raise ExogenousEvaluationError(
            "Shadow: horizon_actuals_present doit etre un booleen explicite."
        )
    if not horizon_actuals:
        deadline = pd.Timestamp(origin) + pd.Timedelta(
            hours=SHADOW_MAX_FORECAST_LATENCY_HOURS
        )
        if not (pd.Timestamp(origin) <= pd.Timestamp(created) <= deadline):
            raise ExogenousEvaluationError(
                "Shadow: panel de forecast cree hors fenetre prospective."
            )
    production = audit_payload.get("production_pit_evidence")
    if not isinstance(production, Mapping) or type(production.get(canonical_zone)) is not bool:
        raise ExogenousEvaluationError(
            "Shadow: production_pit_evidence de zone absent/non booleen."
        )
    bank_root = audit_payload.get("exogenous_banks")
    bank = bank_root.get(canonical_zone) if isinstance(bank_root, Mapping) else None
    if not isinstance(bank, Mapping):
        raise ExogenousEvaluationError("Shadow: audit de banque exogene absent.")
    if type(bank.get("production_ready")) is not bool or bool(
        bank["production_ready"]
    ) != bool(production[canonical_zone]):
        raise ExogenousEvaluationError(
            "Shadow: drapeaux production du panel incoherents."
        )
    source_hashes = _normalise_hash_mapping(
        bank.get("source_hashes"), label="audit.source_hashes"
    )
    source_audit_hashes = _normalise_hash_mapping(
        bank.get("source_audit_hashes"), label="audit.source_audit_hashes"
    )
    if not set(source_audit_hashes).issubset(source_hashes):
        raise ExogenousEvaluationError(
            "Shadow: les sidecars sources ne correspondent pas aux sources du panel."
        )
    source_timezones = bank.get("source_cutoff_timezones")
    if not isinstance(source_timezones, Mapping):
        raise ExogenousEvaluationError(
            "Shadow: source_cutoff_timezones absent du sidecar panel."
        )
    target_root = audit_payload.get("target_sources")
    target = target_root.get(canonical_zone) if isinstance(target_root, Mapping) else None
    if not isinstance(target, Mapping):
        raise ExogenousEvaluationError("Shadow: identite de la cible absente.")
    delivery_timezones = audit_payload.get("delivery_timezones")
    if not isinstance(delivery_timezones, Mapping):
        raise ExogenousEvaluationError("Shadow: delivery_timezones absent.")
    payload = {
        "schema_version": 1,
        "purpose": "prospective_shadow",
        "panel_sha256": actual_panel_hash,
        "panel_audit_sha256": _sha256_file(audit_path),
        "panel_created_at_utc": pd.Timestamp(created).isoformat(),
        "zone": canonical_zone,
        "delivery_day": expected_day,
        "forecast_origin_utc": pd.Timestamp(origin).isoformat(),
        "forecast_origin_timezone": config.timezone,
        "delivery_timezone": delivery_timezones.get(canonical_zone),
        "pack": audit_payload.get("pack"),
        "production_pit_evidence": bool(production[canonical_zone]),
        "source_hashes": source_hashes,
        "source_audit_hashes": source_audit_hashes,
        "source_cutoff_timezones": source_timezones,
        "target_source_sha256": target.get("source_sha256"),
        "horizon_actuals_present": horizon_actuals,
    }
    return _normalise_panel_evidence(payload)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _horizon_mask(group: pd.DataFrame, config: ExogenousFineTuneConfig) -> np.ndarray:
    origin = pd.Timestamp(group[config.origin_column].iloc[0]).tz_convert(config.timezone)
    delivery_day = (origin + pd.DateOffset(days=1)).date()
    timestamps = pd.DatetimeIndex(group[config.timestamp_column]).tz_convert(
        config.timezone
    )
    return np.asarray([timestamp.date() == delivery_day for timestamp in timestamps])


def build_inference_input(
    group: pd.DataFrame,
    config: ExogenousFineTuneConfig,
    *,
    allow_missing_actual: bool = False,
) -> tuple[dict[str, Any], pd.DatetimeIndex, np.ndarray]:
    """Build one raw Chronos input while proving that no horizon label enters it.

    The panel's horizon target is first overwritten with ``NaN``.  Chronos'
    raw-dict API expects the target and past covariates to contain context only,
    so the masked horizon is then excluded from those arrays.  Known-future
    covariates are the sole arrays copied from the D+1 suffix.
    """

    ordered = group.sort_values(config.timestamp_column, kind="stable").reset_index(
        drop=True
    )
    horizon_mask = _horizon_mask(ordered, config)
    horizon_hours = int(horizon_mask.sum())
    if horizon_hours not in {23, 24, 25}:
        raise ExogenousEvaluationError(
            f"Inférence: horizon civil invalide ({horizon_hours} heures)."
        )
    positions = np.flatnonzero(horizon_mask)
    if not np.array_equal(
        positions, np.arange(len(ordered) - horizon_hours, len(ordered))
    ):
        raise ExogenousEvaluationError("Inférence: la livraison D+1 n'est pas un suffixe.")
    if len(ordered) - horizon_hours != config.context_length:
        raise ExogenousEvaluationError(
            "Inférence: longueur de contexte différente du contrat."
        )

    target_frame = ordered.loc[:, list(config.target_columns)].astype(float).copy()
    actual = target_frame.loc[horizon_mask].to_numpy(dtype=float).T
    target_frame.loc[horizon_mask, :] = np.nan
    if target_frame.loc[horizon_mask].notna().any().any():  # pragma: no cover
        raise ExogenousEvaluationError("Inférence: échec du masquage des labels futurs.")
    target_context = target_frame.loc[~horizon_mask].to_numpy(dtype=np.float32).T
    if not np.isfinite(target_context).all():
        raise ExogenousEvaluationError("Inférence: target contexte non finie.")
    actual_finite = np.isfinite(actual)
    if allow_missing_actual:
        if bool(actual_finite.any()) and not bool(actual_finite.all()):
            raise ExogenousEvaluationError(
                "Shadow: actual doit être entièrement publié ou entièrement absent."
            )
    elif not bool(actual_finite.all()):
        raise ExogenousEvaluationError("Inférence: label d'évaluation non fini.")

    past_covariates: dict[str, np.ndarray] = {}
    for column in config.past_only_covariates:
        values = ordered.loc[~horizon_mask, column].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ExogenousEvaluationError(
                f"Inférence: covariable passée {column!r} non finie."
            )
        past_covariates[column] = values

    future_covariates: dict[str, np.ndarray] = {}
    for column in config.known_future_covariates:
        past_values = ordered.loc[~horizon_mask, column].to_numpy(dtype=np.float32)
        future_values = ordered.loc[horizon_mask, column].to_numpy(dtype=np.float32)
        if not np.isfinite(past_values).all() or not np.isfinite(future_values).all():
            raise ExogenousEvaluationError(
                f"Inférence: covariable connue-future {column!r} non finie."
            )
        past_covariates[column] = past_values
        future_covariates[column] = future_values

    payload = {
        "target": target_context,
        "past_covariates": past_covariates,
        "future_covariates": future_covariates,
    }
    horizon = pd.DatetimeIndex(ordered.loc[horizon_mask, config.timestamp_column])
    return payload, horizon, actual


def _clone_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target": np.asarray(payload["target"]).copy(),
        "past_covariates": {
            str(key): np.asarray(value).copy()
            for key, value in dict(payload["past_covariates"]).items()
        },
        "future_covariates": {
            str(key): np.asarray(value).copy()
            for key, value in dict(payload["future_covariates"]).items()
        },
    }


def _input_sha256(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for namespace, values in (
        ("target", {"target": payload["target"]}),
        ("past", payload["past_covariates"]),
        ("future", payload["future_covariates"]),
    ):
        for name, value in sorted(dict(values).items()):
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(str(namespace).encode("utf-8"))
            digest.update(str(name).encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _predict_quantiles(
    pipeline: Any,
    payload: Mapping[str, Any],
    *,
    prediction_length: int,
    context_length: int,
    batch_size: int,
) -> np.ndarray:
    return _predict_quantiles_batch(
        pipeline,
        [payload],
        prediction_length=prediction_length,
        context_length=context_length,
        batch_size=batch_size,
    )[0]


def _predict_quantiles_batch(
    pipeline: Any,
    payloads: Sequence[Mapping[str, Any]],
    *,
    prediction_length: int,
    context_length: int,
    batch_size: int,
) -> list[np.ndarray]:
    """Predict an equal-horizon batch without enabling cross-series learning."""

    if not hasattr(pipeline, "predict_quantiles"):
        raise ExogenousEvaluationError(
            "Le pipeline doit exposer predict_quantiles (Chronos-2)."
        )
    required_batch_size = max(
        int(np.asarray(payload["target"]).shape[0])
        + len(dict(payload["past_covariates"]))
        for payload in payloads
    )
    if int(batch_size) < required_batch_size:
        raise ExogenousEvaluationError(
            f"batch_size={batch_size} < {required_batch_size} variates par input."
        )
    quantiles, _ = pipeline.predict_quantiles(
        [_clone_input(payload) for payload in payloads],
        prediction_length=int(prediction_length),
        quantile_levels=list(QUANTILE_LEVELS),
        batch_size=int(batch_size),
        context_length=int(context_length),
        cross_learning=False,
        limit_prediction_length=False,
    )
    if not isinstance(quantiles, Sequence) or len(quantiles) != len(payloads):
        raise ExogenousEvaluationError(
            "Chronos-2: une sortie est requise pour chaque origine du batch."
        )
    outputs: list[np.ndarray] = []
    for raw in quantiles:
        values = _to_numpy(raw)
        if values.ndim == 2:
            values = values[np.newaxis, :, :]
        if values.ndim != 3 or values.shape[1:] != (
            prediction_length,
            len(QUANTILE_LEVELS),
        ):
            raise ExogenousEvaluationError(
                "Chronos-2: forme quantile inattendue; attendu "
                f"(targets,{prediction_length},3), reçu {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ExogenousEvaluationError("Chronos-2: quantiles NaN/infini.")
        if not (
            np.all(values[:, :, 0] <= values[:, :, 1])
            and np.all(values[:, :, 1] <= values[:, :, 2])
        ):
            raise ExogenousEvaluationError(
                "Chronos-2: croisement de quantiles; aucune correction silencieuse appliquée."
            )
        outputs.append(values)
    return outputs


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _predict_batches_with_cache(
    pipeline: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    prediction_length: int,
    context_length: int,
    batch_size: int,
    chunk_size: int,
    cache_directory: Path | None,
    cache_label: str,
    model_identity: str,
) -> list[np.ndarray]:
    """Run equal-horizon chunks and atomically cache only fully verified chunks."""

    if chunk_size <= 0:
        raise ExogenousEvaluationError("inference_chunk_size doit être positif.")
    results: list[np.ndarray] = []
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start : start + chunk_size]
        contract = hashlib.sha256(
            (
                "chronos2_exogenous_cache_v1\n"
                + model_identity
                + f"\nprediction_length={prediction_length}"
                + f"\ncontext_length={context_length}"
                + f"\nquantiles={QUANTILE_LEVELS}"
                + "\ncross_learning=false"
                + "\n"
                + "\n".join(_input_sha256(payload) for payload in chunk)
            ).encode("utf-8")
        ).hexdigest()
        cache_path = (
            cache_directory
            / f"{cache_label}_h{prediction_length}_{start:04d}_{contract[:16]}.npz"
            if cache_directory is not None
            else None
        )
        cached: list[np.ndarray] | None = None
        if cache_path is not None and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as bundle:
                    if str(bundle["contract"].item()) != contract:
                        raise ValueError("contract")
                    values = np.asarray(bundle["values"], dtype=float)
                if values.shape[0] != len(chunk):
                    raise ValueError("count")
                cached = [values[index] for index in range(len(chunk))]
            except Exception as exc:
                raise ExogenousEvaluationError(
                    f"Cache d'inférence invalide: {cache_path}."
                ) from exc
        if cached is None:
            cached = _predict_quantiles_batch(
                pipeline,
                chunk,
                prediction_length=prediction_length,
                context_length=context_length,
                batch_size=batch_size,
            )
            if cache_path is not None:
                _atomic_npz(
                    cache_path,
                    contract=np.asarray(contract),
                    values=np.stack(cached, axis=0),
                )
        results.extend(cached)
    return results


def _select_item(
    panel: pd.DataFrame,
    config: ExogenousFineTuneConfig,
    item_id: str | None,
) -> str:
    items = tuple(sorted(panel[config.item_column].astype(str).unique()))
    if item_id is None:
        if len(items) != 1:
            raise ExogenousEvaluationError(
                "Plusieurs items dans le panel; --item-id est obligatoire: "
                + ", ".join(items)
            )
        return items[0]
    selected = str(item_id)
    if selected not in items:
        raise ExogenousEvaluationError(
            f"Item {selected!r} absent; disponibles={list(items)}."
        )
    return selected


def evaluate_holdout(
    panel: pd.DataFrame,
    split: OriginSplit,
    config: ExogenousFineTuneConfig,
    *,
    baseline_pipeline: Any,
    candidate_pipeline: Any,
    item_id: str | None = None,
    target_column: str | None = None,
    batch_size: int = 64,
    inference_chunk_size: int = 32,
    cache_directory: str | Path | None = None,
    baseline_identity: str = "chronos2_base",
    candidate_identity: str = "chronos2_lora",
    progress: Callable[[int, int, pd.Timestamp], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run paired, origin-isolated inference over exactly the frozen holdout."""

    if int(batch_size) <= 0:
        raise ExogenousEvaluationError("batch_size doit être positif.")
    minimum_batch_size = len(config.target_columns) + len(config.covariate_columns)
    if int(batch_size) < minimum_batch_size:
        raise ExogenousEvaluationError(
            f"batch_size={batch_size} < {minimum_batch_size} variates par input."
        )
    if len(split.evaluation) != config.evaluation_days:
        raise ExogenousEvaluationError(
            f"Holdout {len(split.evaluation)} jours != {config.evaluation_days}."
        )
    if config.evaluation_days != 365:
        raise ExogenousEvaluationError(
            "Le rapport de gouvernance exige exactement 365 jours."
        )
    selected_item = _select_item(panel, config, item_id)
    selected_target = target_column or config.target_columns[0]
    if selected_target not in config.target_columns:
        raise ExogenousEvaluationError(
            f"Target {selected_target!r} absente de {list(config.target_columns)}."
        )
    target_index = config.target_columns.index(selected_target)

    evaluation_origins = pd.DatetimeIndex(split.evaluation)
    selected = panel.loc[
        panel[config.origin_column].isin(evaluation_origins)
        & panel[config.item_column].astype(str).eq(selected_item)
    ]
    present_origins = pd.DatetimeIndex(
        selected[config.origin_column].drop_duplicates()
    ).sort_values()
    if not present_origins.equals(evaluation_origins.sort_values()):
        raise ExogenousEvaluationError(
            "Le panel sélectionné ne couvre pas toutes les origines du holdout."
        )

    samples_by_horizon: dict[int, list[dict[str, Any]]] = {23: [], 24: [], 25: []}
    input_hashes_by_origin: dict[pd.Timestamp, str] = {}
    total = len(evaluation_origins)
    for origin in evaluation_origins:
        group = selected.loc[selected[config.origin_column].eq(origin)]
        payload, horizon, actual = build_inference_input(group, config)
        input_digest = _input_sha256(payload)
        samples_by_horizon[len(horizon)].append(
            {
                "origin": pd.Timestamp(origin),
                "payload": payload,
                "horizon": horizon,
                "actual": actual,
                "input_sha256": input_digest,
            }
        )
        input_hashes_by_origin[pd.Timestamp(origin)] = input_digest

    cache_root = (
        Path(cache_directory).expanduser().resolve()
        if cache_directory is not None
        else None
    )
    prediction_pairs: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    for horizon_hours in (23, 24, 25):
        samples = samples_by_horizon[horizon_hours]
        if not samples:
            continue
        payloads = [sample["payload"] for sample in samples]
        baseline_outputs = _predict_batches_with_cache(
            baseline_pipeline,
            payloads,
            prediction_length=horizon_hours,
            context_length=config.context_length,
            batch_size=batch_size,
            chunk_size=inference_chunk_size,
            cache_directory=cache_root,
            cache_label="baseline",
            model_identity=baseline_identity,
        )
        candidate_outputs = _predict_batches_with_cache(
            candidate_pipeline,
            payloads,
            prediction_length=horizon_hours,
            context_length=config.context_length,
            batch_size=batch_size,
            chunk_size=inference_chunk_size,
            cache_directory=cache_root,
            cache_label="candidate",
            model_identity=candidate_identity,
        )
        for sample, baseline, candidate in zip(
            samples, baseline_outputs, candidate_outputs, strict=True
        ):
            if _input_sha256(sample["payload"]) != sample["input_sha256"]:
                raise ExogenousEvaluationError("Un pipeline a muté son entrée.")
            prediction_pairs[sample["origin"]] = (baseline, candidate)

    rows: list[pd.DataFrame] = []
    for number, origin in enumerate(evaluation_origins, start=1):
        matches = [
            sample
            for samples in samples_by_horizon.values()
            for sample in samples
            if sample["origin"] == pd.Timestamp(origin)
        ]
        if len(matches) != 1:  # pragma: no cover - guarded by origin uniqueness.
            raise ExogenousEvaluationError("Mapping origine/prédiction ambigu.")
        sample = matches[0]
        horizon = sample["horizon"]
        actual = sample["actual"]
        baseline, candidate = prediction_pairs[pd.Timestamp(origin)]
        if baseline.shape[0] != len(config.target_columns) or candidate.shape[0] != len(
            config.target_columns
        ):
            raise ExogenousEvaluationError(
                "Le nombre de targets prédit diffère du schéma du panel."
            )
        rows.append(
            pd.DataFrame(
                {
                    "delivery_start_utc": horizon,
                    "forecast_origin_utc": pd.DatetimeIndex([origin] * len(horizon)),
                    "actual": actual[target_index],
                    "baseline_q10": baseline[target_index, :, 0],
                    "baseline_q50": baseline[target_index, :, 1],
                    "baseline_q90": baseline[target_index, :, 2],
                    "candidate_q10": candidate[target_index, :, 0],
                    "candidate_q50": candidate[target_index, :, 1],
                    "candidate_q90": candidate[target_index, :, 2],
                },
                columns=EVALUATION_COLUMNS,
            )
        )
        if progress is not None:
            progress(number, total, pd.Timestamp(origin))

    evidence = pd.concat(rows, ignore_index=True)
    evidence = evidence.sort_values("delivery_start_utc", kind="stable").reset_index(
        drop=True
    )
    contract_digest = hashlib.sha256(
        "\n".join(input_hashes_by_origin[pd.Timestamp(origin)] for origin in evaluation_origins).encode(
            "ascii"
        )
    ).hexdigest()
    audit = {
        "item_id": selected_item,
        "target_column": selected_target,
        "evaluation_days": total,
        "rows": len(evidence),
        "input_contract_sha256": contract_digest,
        "paired_same_inputs": True,
        "cross_learning": False,
        "residual_corrector_applied": False,
        "mkonline_applied": False,
        "storm_used": False,
        "quantile_levels": list(QUANTILE_LEVELS),
        "batched_by_physical_horizon": True,
        "inference_chunk_size": int(inference_chunk_size),
        "cache_enabled": cache_root is not None,
    }
    return evidence, audit


def _prepare_shadow_panel(
    frame: pd.DataFrame, config: ExogenousFineTuneConfig
) -> pd.DataFrame:
    """Validate the causal subset needed for live shadow inference.

    Unlike the training validator, this permits a wholly missing D+1 target.
    Context targets and every requested covariate remain strict.
    """

    required = {
        config.timestamp_column,
        config.origin_column,
        config.item_column,
        config.feature_available_at_column,
        *config.target_columns,
        *config.covariate_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ExogenousEvaluationError(f"Shadow: colonnes absentes: {missing}.")
    panel = frame.loc[:, list(dict.fromkeys(required))].copy()
    for column in (
        config.timestamp_column,
        config.origin_column,
        config.feature_available_at_column,
    ):
        panel[column] = pd.to_datetime(panel[column], errors="coerce", utc=True)
        if panel[column].isna().any():
            raise ExogenousEvaluationError(f"Shadow: timestamp invalide dans {column}.")
    panel[config.item_column] = panel[config.item_column].astype(str)
    if bool(
        (
            panel[config.feature_available_at_column]
            > panel[config.origin_column]
        ).any()
    ):
        raise ExogenousEvaluationError("Shadow: feature disponible après l'origine.")
    for column in (*config.target_columns, *config.covariate_columns):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    for (origin, item), group in panel.groupby(
        [config.origin_column, config.item_column], sort=True, observed=True
    ):
        ordered = group.sort_values(config.timestamp_column)
        horizon_mask = _horizon_mask(ordered, config)
        if int(horizon_mask.sum()) not in {23, 24, 25}:
            raise ExogenousEvaluationError(
                f"Shadow {origin}/{item}: horizon hors 23/24/25."
            )
        if len(ordered) - int(horizon_mask.sum()) != config.context_length:
            raise ExogenousEvaluationError(
                f"Shadow {origin}/{item}: contexte différent de {config.context_length}."
            )
        targets = ordered.loc[:, list(config.target_columns)].to_numpy(float)
        if not np.isfinite(targets[~horizon_mask]).all():
            raise ExogenousEvaluationError(
                f"Shadow {origin}/{item}: target contexte manquante."
            )
        horizon_finite = np.isfinite(targets[horizon_mask])
        if bool(horizon_finite.any()) and not bool(horizon_finite.all()):
            raise ExogenousEvaluationError(
                f"Shadow {origin}/{item}: actual D+1 partiellement publiée."
            )
        if not np.isfinite(
            ordered.loc[:, list(config.covariate_columns)].to_numpy(float)
        ).all():
            raise ExogenousEvaluationError(
                f"Shadow {origin}/{item}: covariable manquante/non finie."
            )
    return panel


def daily_inference(
    panel: pd.DataFrame,
    config: ExogenousFineTuneConfig,
    *,
    baseline_pipeline: Any,
    candidate_pipeline: Any,
    origins: Sequence[str | pd.Timestamp] | None = None,
    item_id: str | None = None,
    target_column: str | None = None,
    batch_size: int = 64,
    checkpoint_sha256: str,
    panel_evidence: Mapping[str, Any],
    progress: Callable[[int, int, pd.Timestamp], None] | None = None,
) -> pd.DataFrame:
    """Infer one or more shadow days; actual may be wholly unavailable."""

    validated = _prepare_shadow_panel(panel, config)
    selected_item = _select_item(validated, config, item_id)
    selected_target = target_column or config.target_columns[0]
    if selected_target not in config.target_columns:
        raise ExogenousEvaluationError(f"Shadow: target inconnue {selected_target!r}.")
    target_index = config.target_columns.index(selected_target)
    available_origins = pd.DatetimeIndex(
        validated[config.origin_column].drop_duplicates()
    ).sort_values()
    if origins:
        requested = pd.DatetimeIndex(pd.to_datetime(list(origins), utc=True)).sort_values()
        missing = requested.difference(available_origins)
        if len(missing):
            raise ExogenousEvaluationError(
                "Shadow: origines absentes: "
                + ", ".join(timestamp.isoformat() for timestamp in missing)
            )
    else:
        requested = available_origins[-1:]
    provenance = _normalise_panel_evidence(panel_evidence)
    evidence_origin = pd.Timestamp(provenance["forecast_origin_utc"])
    if len(requested) != 1 or pd.Timestamp(requested[0]) != evidence_origin:
        raise ExogenousEvaluationError(
            "Shadow: la provenance du panel doit identifier exactement l'origine inferee."
        )
    if str(provenance["zone"]) != str(selected_item).strip().upper():
        raise ExogenousEvaluationError(
            "Shadow: la zone de provenance differe de l'item infere."
        )
    provenance_json = _canonical_json_text(provenance)
    selected = validated.loc[
        validated[config.origin_column].isin(requested)
        & validated[config.item_column].eq(selected_item)
    ]
    frames: list[pd.DataFrame] = []
    for number, origin in enumerate(requested, start=1):
        group = selected.loc[selected[config.origin_column].eq(origin)]
        payload, horizon, actual = build_inference_input(
            group, config, allow_missing_actual=True
        )
        input_digest = _input_sha256(payload)
        baseline = _predict_quantiles(
            baseline_pipeline,
            payload,
            prediction_length=len(horizon),
            context_length=config.context_length,
            batch_size=batch_size,
        )
        candidate = _predict_quantiles(
            candidate_pipeline,
            payload,
            prediction_length=len(horizon),
            context_length=config.context_length,
            batch_size=batch_size,
        )
        if baseline.shape[0] != len(config.target_columns) or candidate.shape[0] != len(
            config.target_columns
        ):
            raise ExogenousEvaluationError("Shadow: nombre de targets prédit invalide.")
        frames.append(
            pd.DataFrame(
                {
                    "delivery_start_utc": horizon,
                    "forecast_origin_utc": pd.DatetimeIndex([origin] * len(horizon)),
                    "item_id": selected_item,
                    "target_column": selected_target,
                    "actual": actual[target_index],
                    "baseline_q10": baseline[target_index, :, 0],
                    "baseline_q50": baseline[target_index, :, 1],
                    "baseline_q90": baseline[target_index, :, 2],
                    "candidate_q10": candidate[target_index, :, 0],
                    "candidate_q50": candidate[target_index, :, 1],
                    "candidate_q90": candidate[target_index, :, 2],
                    "input_contract_sha256": input_digest,
                    "checkpoint_sha256": str(checkpoint_sha256),
                    "panel_sha256": provenance["panel_sha256"],
                    "panel_audit_sha256": provenance["panel_audit_sha256"],
                    "panel_contract_sha256": provenance[
                        "panel_contract_sha256"
                    ],
                    "panel_created_at_utc": provenance["panel_created_at_utc"],
                    "panel_provenance_json": provenance_json,
                },
                columns=SHADOW_INPUT_COLUMNS,
            )
        )
        if progress is not None:
            progress(number, len(requested), pd.Timestamp(origin))
    return pd.concat(frames, ignore_index=True)


def build_actual_resolution_input(
    panel: pd.DataFrame,
    config: ExogenousFineTuneConfig,
    *,
    journal_path: str | Path,
    origins: Sequence[str | pd.Timestamp],
    item_id: str | None = None,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Attach newly published actuals to prior forecasts without model inference."""

    journal = Path(journal_path).expanduser().resolve()
    if not journal.is_file():
        raise ExogenousEvaluationError(
            "Shadow: actual disponible mais aucun journal de forecast antérieur scellé."
        )
    history = pd.read_csv(journal)
    if tuple(history.columns) != SHADOW_JOURNAL_COLUMNS:
        raise ExogenousEvaluationError("Shadow: journal existant de schéma inconnu.")
    for column in ("delivery_start_utc", "forecast_origin_utc", "captured_at_utc"):
        history[column] = pd.to_datetime(history[column], utc=True, errors="coerce")
    validated = _prepare_shadow_panel(panel, config)
    selected_item = _select_item(validated, config, item_id)
    selected_target = target_column or config.target_columns[0]
    if selected_target not in config.target_columns:
        raise ExogenousEvaluationError(f"Shadow: target inconnue {selected_target!r}.")
    target_index = config.target_columns.index(selected_target)
    requested = pd.DatetimeIndex(pd.to_datetime(list(origins), utc=True)).sort_values()
    records: list[dict[str, Any]] = []
    for origin in requested:
        group = validated.loc[
            validated[config.origin_column].eq(origin)
            & validated[config.item_column].eq(selected_item)
        ]
        if group.empty:
            raise ExogenousEvaluationError(f"Shadow: origine absente {origin.isoformat()}.")
        _, horizon, actual = build_inference_input(
            group, config, allow_missing_actual=True
        )
        if not np.isfinite(actual).all():
            raise ExogenousEvaluationError(
                f"Shadow: actual non publiée pour {origin.isoformat()}."
            )
        for step, timestamp in enumerate(horizon):
            matches = history.loc[
                history["delivery_start_utc"].eq(timestamp)
                & history["forecast_origin_utc"].eq(origin)
                & history["item_id"].astype(str).eq(selected_item)
                & history["target_column"].astype(str).eq(selected_target)
            ]
            if matches.empty:
                raise ExogenousEvaluationError(
                    "Shadow: actual disponible sans forecast horaire antérieur scellé: "
                    f"{timestamp.isoformat()}."
                )
            sealed = matches.iloc[-1]
            records.append(
                {
                    "delivery_start_utc": timestamp,
                    "forecast_origin_utc": origin,
                    "item_id": selected_item,
                    "target_column": selected_target,
                    "actual": float(actual[target_index, step]),
                    "baseline_q10": sealed["baseline_q10"],
                    "baseline_q50": sealed["baseline_q50"],
                    "baseline_q90": sealed["baseline_q90"],
                    "candidate_q10": sealed["candidate_q10"],
                    "candidate_q50": sealed["candidate_q50"],
                    "candidate_q90": sealed["candidate_q90"],
                    "input_contract_sha256": sealed["input_contract_sha256"],
                    "checkpoint_sha256": sealed["checkpoint_sha256"],
                    "panel_sha256": sealed["panel_sha256"],
                    "panel_audit_sha256": sealed["panel_audit_sha256"],
                    "panel_contract_sha256": sealed["panel_contract_sha256"],
                    "panel_created_at_utc": sealed["panel_created_at_utc"],
                    "panel_provenance_json": sealed["panel_provenance_json"],
                }
            )
    return pd.DataFrame(records, columns=SHADOW_INPUT_COLUMNS)


def _validate_shadow_provenance_rows(
    frame: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Revalidate embedded panel identities before hash-sealing journal rows."""

    identities: dict[str, dict[str, Any]] = {}
    for index, row in frame.iterrows():
        try:
            raw = json.loads(str(row["panel_provenance_json"]))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ExogenousEvaluationError(
                f"Shadow: panel_provenance_json invalide a la ligne {index}."
            ) from exc
        if not isinstance(raw, Mapping):
            raise ExogenousEvaluationError(
                f"Shadow: provenance panel non objet a la ligne {index}."
            )
        evidence = _normalise_panel_evidence(raw)
        canonical = _canonical_json_text(evidence)
        if str(row["panel_provenance_json"]) != canonical:
            raise ExogenousEvaluationError(
                "Shadow: provenance panel non canonique dans l'entree."
            )
        for column in (
            "panel_sha256",
            "panel_audit_sha256",
            "panel_contract_sha256",
        ):
            if _require_sha256(row[column], label=column) != evidence[column]:
                raise ExogenousEvaluationError(
                    f"Shadow: {column} divergent de la provenance embarquee."
                )
        created = pd.to_datetime(
            row["panel_created_at_utc"], utc=True, errors="coerce"
        )
        if pd.isna(created) or pd.Timestamp(created) != pd.Timestamp(
            evidence["panel_created_at_utc"]
        ):
            raise ExogenousEvaluationError(
                "Shadow: panel_created_at_utc divergent de la provenance."
            )
        origin = pd.Timestamp(row["forecast_origin_utc"])
        if origin != pd.Timestamp(evidence["forecast_origin_utc"]):
            raise ExogenousEvaluationError(
                "Shadow: origine de ligne divergente de la provenance panel."
            )
        if str(row["item_id"]).strip().upper() != evidence["zone"]:
            raise ExogenousEvaluationError(
                "Shadow: item de ligne divergent de la zone du panel."
            )
        delivery_day = (
            pd.Timestamp(row["delivery_start_utc"])
            .tz_convert(evidence["delivery_timezone"])
            .date()
            .isoformat()
        )
        if delivery_day != evidence["delivery_day"]:
            raise ExogenousEvaluationError(
                "Shadow: heure de livraison hors du jour scelle par le panel."
            )
        identities[evidence["panel_contract_sha256"]] = evidence
    return identities


def _canonical_record_hash(row: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for column in SHADOW_JOURNAL_COLUMNS:
        if column == "record_sha256":
            continue
        value = row.get(column)
        if pd.isna(value):
            payload[column] = None
        elif isinstance(value, pd.Timestamp):
            payload[column] = value.isoformat()
        elif isinstance(value, np.generic):
            payload[column] = value.item()
        else:
            payload[column] = value
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def append_shadow_predictions(
    run_directory: str | Path,
    predictions: pd.DataFrame,
    *,
    journal_path: str | Path | None = None,
) -> ShadowArtifacts:
    """Append new forecasts or later actual resolutions to a hash-chained log.

    Re-running the same forecast is idempotent.  Changing a forecast for an
    already logged bundle/origin is refused.  When actual prices become known,
    a second immutable revision is appended and an observed-only evidence file
    is rebuilt from the latest revisions.
    """

    run_dir = Path(run_directory).expanduser().resolve()
    bundle = verify_bundle(run_dir)
    journal = (
        Path(journal_path).expanduser().resolve()
        if journal_path is not None
        else run_dir / "shadow_predictions.csv.gz"
    )
    journal.parent.mkdir(parents=True, exist_ok=True)
    if tuple(predictions.columns) != SHADOW_INPUT_COLUMNS:
        raise ExogenousEvaluationError(
            "Shadow: schéma exact requis: " + ", ".join(SHADOW_INPUT_COLUMNS)
        )
    incoming = predictions.copy()
    for column in (
        "delivery_start_utc",
        "forecast_origin_utc",
        "panel_created_at_utc",
    ):
        incoming[column] = pd.to_datetime(incoming[column], utc=True, errors="coerce")
        if incoming[column].isna().any():
            raise ExogenousEvaluationError(f"Shadow: timestamp invalide dans {column}.")
    if incoming.duplicated(
        ["delivery_start_utc", "forecast_origin_utc", "item_id", "target_column"]
    ).any():
        raise ExogenousEvaluationError("Shadow: clés horaires dupliquées dans l'entrée.")
    if incoming["checkpoint_sha256"].nunique() != 1 or str(
        incoming["checkpoint_sha256"].iloc[0]
    ) != str(bundle["checkpoint_sha256"]):
        raise ExogenousEvaluationError("Shadow: checksum checkpoint différent du bundle.")
    if incoming[["item_id", "target_column"]].drop_duplicates().shape[0] != 1:
        raise ExogenousEvaluationError("Shadow: un journal est limité à un item/target.")
    incoming_panel_identities = _validate_shadow_provenance_rows(incoming)
    if len(incoming_panel_identities) != 1:
        raise ExogenousEvaluationError(
            "Shadow: une emission doit provenir d'un unique panel scelle."
        )
    for column in ("input_contract_sha256", "checkpoint_sha256"):
        for value in incoming[column].drop_duplicates():
            _require_sha256(value, label=column)

    if journal.exists():
        existing = pd.read_csv(journal)
        if tuple(existing.columns) != SHADOW_JOURNAL_COLUMNS:
            raise ExogenousEvaluationError("Shadow: journal existant de schéma inconnu.")
        for column in (
            "delivery_start_utc",
            "forecast_origin_utc",
            "panel_created_at_utc",
            "captured_at_utc",
        ):
            existing[column] = pd.to_datetime(existing[column], utc=True, errors="coerce")
        if existing[["item_id", "target_column"]].drop_duplicates().shape[0] != 1:
            raise ExogenousEvaluationError("Shadow: journal multi-item interdit.")
        if (
            str(existing["item_id"].iloc[0]) != str(incoming["item_id"].iloc[0])
            or str(existing["target_column"].iloc[0])
            != str(incoming["target_column"].iloc[0])
        ):
            raise ExogenousEvaluationError("Shadow: item/target différent du journal.")
        for index, row in existing.iterrows():
            if _canonical_record_hash(row) != str(row["record_sha256"]):
                raise ExogenousEvaluationError(
                    f"Shadow: hash de ligne historique invalide à l'index {index}."
                )
            expected_previous = "" if index == 0 else str(existing.iloc[index - 1]["record_sha256"])
            actual_previous = (
                "" if pd.isna(row["previous_record_sha256"]) else str(row["previous_record_sha256"])
            )
            if actual_previous != expected_previous:
                raise ExogenousEvaluationError("Shadow: chaîne de hashes historique rompue.")
    else:
        existing = pd.DataFrame(columns=SHADOW_JOURNAL_COLUMNS)

    key_columns = [
        "delivery_start_utc",
        "forecast_origin_utc",
        "item_id",
        "target_column",
    ]
    forecast_columns = [
        "baseline_q10",
        "baseline_q50",
        "baseline_q90",
        "candidate_q10",
        "candidate_q50",
        "candidate_q90",
    ]
    appended: list[dict[str, Any]] = []
    previous_hash: str | None = (
        str(existing.iloc[-1]["record_sha256"]) if len(existing) else None
    )
    captured = _utc_now()
    for row in incoming.sort_values("delivery_start_utc").to_dict(orient="records"):
        matches = existing
        for column in key_columns:
            matches = matches.loc[matches[column].eq(row[column])]
        latest = matches.iloc[-1] if len(matches) else None
        if latest is not None:
            if any(
                not math.isclose(
                    float(latest[column]), float(row[column]), rel_tol=0.0, abs_tol=1e-9
                )
                for column in forecast_columns
            ):
                raise ExogenousEvaluationError(
                    "Shadow: forecast différent pour une clé déjà scellée."
                )
            if str(latest["input_contract_sha256"]) != str(
                row["input_contract_sha256"]
            ):
                raise ExogenousEvaluationError("Shadow: input hash différent pour une clé scellée.")
            if str(latest["panel_contract_sha256"]) != str(
                row["panel_contract_sha256"]
            ):
                raise ExogenousEvaluationError(
                    "Shadow: identite panel differente pour une cle scellee."
                )
            old_actual = float(latest["actual"]) if pd.notna(latest["actual"]) else np.nan
            new_actual = float(row["actual"]) if pd.notna(row["actual"]) else np.nan
            if np.isfinite(old_actual):
                if np.isfinite(new_actual) and not math.isclose(
                    old_actual, new_actual, rel_tol=0.0, abs_tol=1e-9
                ):
                    raise ExogenousEvaluationError("Shadow: actual publiée a changé.")
                continue
            if not np.isfinite(new_actual):
                continue
            revision = int(latest["revision"]) + 1
            record_kind = "actual_resolution"
            # Resolution is a pure label attachment.  Forecasts and their
            # input/checkpoint fingerprints are copied from the already
            # hash-sealed record, never trusted from a recomputation.
            for column in (
                *forecast_columns,
                "input_contract_sha256",
                "checkpoint_sha256",
                "panel_sha256",
                "panel_audit_sha256",
                "panel_contract_sha256",
                "panel_created_at_utc",
                "panel_provenance_json",
            ):
                row[column] = latest[column]
        else:
            if pd.notna(row["actual"]):
                raise ExogenousEvaluationError(
                    "Shadow: actual déjà disponible sans forecast antérieur scellé; "
                    "reconstruction rétrospective interdite."
                )
            origin = pd.Timestamp(row["forecast_origin_utc"])
            deadline = origin + pd.Timedelta(
                hours=SHADOW_MAX_FORECAST_LATENCY_HOURS
            )
            if captured < origin or captured > deadline:
                raise ExogenousEvaluationError(
                    "Shadow: première émission hors fenêtre prospective "
                    f"[{origin.isoformat()}, {deadline.isoformat()}]; "
                    "un replay tardif ne constitue pas une preuve live."
                )
            evidence = incoming_panel_identities[str(row["panel_contract_sha256"])]
            if evidence["horizon_actuals_present"]:
                raise ExogenousEvaluationError(
                    "Shadow: premiere emission issue d'un panel contenant deja les actuals."
                )
            panel_created = pd.Timestamp(row["panel_created_at_utc"])
            if not (origin <= panel_created <= captured):
                raise ExogenousEvaluationError(
                    "Shadow: panel cree hors sequence origine -> capture forecast."
                )
            revision = 1
            record_kind = "forecast"
        record = {
            **row,
            "captured_at_utc": captured,
            "revision": revision,
            "record_kind": record_kind,
            "previous_record_sha256": previous_hash,
            "record_sha256": "",
        }
        record["record_sha256"] = _canonical_record_hash(record)
        previous_hash = str(record["record_sha256"])
        appended.append(record)

    appended_frame = pd.DataFrame(appended, columns=SHADOW_JOURNAL_COLUMNS)
    if existing.empty:
        combined = appended_frame.copy()
    elif appended_frame.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, appended_frame], ignore_index=True)
    if appended or not journal.exists():
        _atomic_csv_gz(journal, combined.loc[:, list(SHADOW_JOURNAL_COLUMNS)])

    latest = combined.drop_duplicates(key_columns, keep="last")
    observed = latest.loc[
        latest["actual"].notna() & latest["record_kind"].eq("actual_resolution")
    ].copy()
    observed_path: Path | None = None
    if len(observed):
        observed_path = journal.with_name("shadow_observed_evidence.csv.gz")
        evidence = observed.loc[:, list(EVALUATION_COLUMNS)].sort_values(
            "delivery_start_utc"
        )
        _atomic_csv_gz(observed_path, evidence)

    shadow_manifest_path: Path | None = None
    if observed_path is not None:
        temporal_audit: list[dict[str, Any]] = []
        for _, observed_row in observed.iterrows():
            history = combined
            for column in key_columns:
                history = history.loc[history[column].eq(observed_row[column])]
            forecasts = history.loc[history["record_kind"].eq("forecast")]
            resolutions = history.loc[history["record_kind"].eq("actual_resolution")]
            if forecasts.empty or resolutions.empty:
                raise ExogenousEvaluationError(
                    "Shadow: preuve temporelle forecast/résolution incomplète."
                )
            forecast_time = pd.Timestamp(forecasts.iloc[0]["captured_at_utc"])
            actual_time = pd.Timestamp(resolutions.iloc[-1]["captured_at_utc"])
            if forecast_time >= actual_time:
                raise ExogenousEvaluationError(
                    "Shadow: actual non strictement postérieure au forecast scellé."
                )
            temporal_audit.append(
                {
                    "delivery_start_utc": pd.Timestamp(
                        observed_row["delivery_start_utc"]
                    ).isoformat(),
                    "forecast_origin_utc": pd.Timestamp(
                        observed_row["forecast_origin_utc"]
                    ).isoformat(),
                    "forecast_deadline_utc": (
                        pd.Timestamp(observed_row["forecast_origin_utc"])
                        + pd.Timedelta(hours=SHADOW_MAX_FORECAST_LATENCY_HOURS)
                    ).isoformat(),
                    "forecast_created_at_utc": forecast_time.isoformat(),
                    "actual_attached_at_utc": actual_time.isoformat(),
                    "forecast_record_sha256": str(forecasts.iloc[0]["record_sha256"]),
                    "resolution_record_sha256": str(
                        resolutions.iloc[-1]["record_sha256"]
                    ),
                    "panel_contract_sha256": str(
                        forecasts.iloc[0]["panel_contract_sha256"]
                    ),
                }
            )
        item = str(incoming["item_id"].iloc[0]).strip().upper()
        target = str(incoming["target_column"].iloc[0]).strip().lower()
        target_match = re.fullmatch(r"target_([a-z]{2,8})", target)
        zone = target_match.group(1).upper() if target_match else item
        if re.fullmatch(r"[A-Z]{2,8}", zone) is None:
            raise ExogenousEvaluationError(
                f"Shadow: zone impossible à déduire de item={item!r}, target={target!r}."
            )
        panel_identities = _validate_shadow_provenance_rows(observed)
        shadow_panel_evidence = [
            panel_identities[digest] for digest in sorted(panel_identities)
        ]
        shadow_manifest_path = journal.with_name("shadow_manifest.json")
        shadow_manifest = {
            "format_version": 3,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "kind": "chronos2_exogenous_append_only_shadow",
            "candidate_output_stage": "chronos2_exogenous",
            "residual_corrector_applied": False,
            # Fields below intentionally mirror governance.validate_shadow_manifest.
            "candidate_model": str(
                bundle.get("experiment_id", bundle.get("model_id"))
            ).strip(),
            "zone": zone,
            "checkpoint_sha256": bundle["checkpoint_sha256"],
            "schema_sha256": bundle["schema_sha256"],
            "candidate_frozen_before_shadow": True,
            "actuals_attached_after_forecast_freeze": True,
            "prospective_capture_deadline_enforced": True,
            "prospective_capture_deadline_hours": (
                SHADOW_MAX_FORECAST_LATENCY_HOURS
            ),
            "forecast_artifact_checksums_valid": True,
            "storm_used_for_prediction": False,
            "mkonline_used_for_prediction": False,
            # This digest targets the exact observed CSV passed to governance,
            # never the wider append-only audit journal.
            "predictions_sha256": _sha256_file(observed_path),
            "forecast_created_at_utc": min(
                record["forecast_created_at_utc"] for record in temporal_audit
            ),
            "actual_attached_at_utc": max(
                record["actual_attached_at_utc"] for record in temporal_audit
            ),
            "all_actuals_attached_after_corresponding_forecast": True,
            "shadow_panel_production_ready": bool(
                all(
                    evidence["production_pit_evidence"]
                    for evidence in shadow_panel_evidence
                )
            ),
            "shadow_panel_evidence": shadow_panel_evidence,
            "temporal_attachment_audit": temporal_audit,
            "journal": {
                "relative_path": journal.name,
                "sha256": _sha256_file(journal),
                "records": len(combined),
                "unique_hours": int(len(latest)),
                "last_record_sha256": previous_hash,
            },
            "observed_governance_evidence": {
                "relative_path": observed_path.name,
                "sha256": _sha256_file(observed_path),
                "rows": len(observed),
            },
        }
        _atomic_text(shadow_manifest_path, _json_text(shadow_manifest))
    return ShadowArtifacts(
        journal_path=journal,
        observed_evidence_path=observed_path,
        manifest_path=shadow_manifest_path,
        appended_rows=len(appended),
    )


def _pinball(actual: np.ndarray, forecast: np.ndarray, quantile: float) -> float:
    error = actual - forecast
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def compute_metrics(
    evidence: pd.DataFrame,
    *,
    timezone_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute paired hourly/probabilistic and local-day metrics."""

    frame = evidence.copy()
    frame["delivery_start_utc"] = pd.to_datetime(
        frame["delivery_start_utc"], utc=True, errors="raise"
    )
    actual = frame["actual"].to_numpy(dtype=float)
    local = pd.DatetimeIndex(frame["delivery_start_utc"]).tz_convert(timezone_name)
    frame["delivery_day"] = [timestamp.date().isoformat() for timestamp in local]

    summary: dict[str, Any] = {
        "schema_version": 1,
        "comparison_scope": "raw_chronos2_same_exogenous_inputs",
        "baseline_label": BASELINE_LABEL,
        "candidate_label": CANDIDATE_LABEL,
        "timezone": timezone_name,
        "physical_days": int(frame["delivery_day"].nunique()),
        "physical_hours": int(len(frame)),
        "first_delivery_utc": frame["delivery_start_utc"].iloc[0].isoformat(),
        "last_delivery_utc": frame["delivery_start_utc"].iloc[-1].isoformat(),
        "actual_mean_price_eur_mwh": float(np.mean(actual)),
    }
    if summary["physical_days"] != 365:
        raise ExogenousEvaluationError(
            f"Métriques: {summary['physical_days']} jours != rolling 365."
        )

    for prefix in ("baseline", "candidate"):
        q50 = frame[f"{prefix}_q50"].to_numpy(dtype=float)
        summary[f"{prefix}_mae_eur_mwh"] = float(np.mean(np.abs(actual - q50)))
        summary[f"{prefix}_rmse_eur_mwh"] = float(
            np.sqrt(np.mean(np.square(actual - q50)))
        )
        summary[f"{prefix}_mean_price_eur_mwh"] = float(np.mean(q50))
        pinballs: list[float] = []
        for label, level in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9)):
            value = _pinball(actual, frame[f"{prefix}_{label}"].to_numpy(float), level)
            summary[f"{prefix}_pinball_{label}"] = value
            pinballs.append(value)
        summary[f"{prefix}_pinball_mean"] = float(np.mean(pinballs))
        summary[f"{prefix}_q10_q90_coverage"] = float(
            np.mean(
                (actual >= frame[f"{prefix}_q10"].to_numpy(float))
                & (actual <= frame[f"{prefix}_q90"].to_numpy(float))
            )
        )
        summary[f"{prefix}_q10_q90_mean_width_eur_mwh"] = float(
            np.mean(
                frame[f"{prefix}_q90"].to_numpy(float)
                - frame[f"{prefix}_q10"].to_numpy(float)
            )
        )

    baseline_mae = summary["baseline_mae_eur_mwh"]
    candidate_mae = summary["candidate_mae_eur_mwh"]
    summary["mae_gain_eur_mwh"] = float(baseline_mae - candidate_mae)
    summary["mae_relative_gain"] = float(
        (baseline_mae - candidate_mae) / baseline_mae if baseline_mae else 0.0
    )

    daily_rows: list[dict[str, Any]] = []
    for day, group in frame.groupby("delivery_day", sort=True):
        day_actual = group["actual"].to_numpy(float)
        baseline = group["baseline_q50"].to_numpy(float)
        candidate = group["candidate_q50"].to_numpy(float)
        row: dict[str, Any] = {
            "delivery_day": str(day),
            "hours": int(len(group)),
            "observed_mean_price_eur_mwh": float(np.mean(day_actual)),
            "baseline_mean_price_eur_mwh": float(np.mean(baseline)),
            "candidate_mean_price_eur_mwh": float(np.mean(candidate)),
            "baseline_hourly_mae_eur_mwh": float(
                np.mean(np.abs(day_actual - baseline))
            ),
            "candidate_hourly_mae_eur_mwh": float(
                np.mean(np.abs(day_actual - candidate))
            ),
            "baseline_daily_mean_abs_error_eur_mwh": float(
                abs(np.mean(day_actual) - np.mean(baseline))
            ),
            "candidate_daily_mean_abs_error_eur_mwh": float(
                abs(np.mean(day_actual) - np.mean(candidate))
            ),
        }
        for prefix in ("baseline", "candidate"):
            row[f"{prefix}_pinball_mean"] = float(
                np.mean(
                    [
                        _pinball(
                            day_actual,
                            group[f"{prefix}_{label}"].to_numpy(float),
                            level,
                        )
                        for label, level in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9))
                    ]
                )
            )
            row[f"{prefix}_q10_q90_coverage"] = float(
                np.mean(
                    (day_actual >= group[f"{prefix}_q10"].to_numpy(float))
                    & (day_actual <= group[f"{prefix}_q90"].to_numpy(float))
                )
            )
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)
    summary["candidate_better_hourly_mae_days"] = int(
        (
            daily["candidate_hourly_mae_eur_mwh"]
            < daily["baseline_hourly_mae_eur_mwh"]
        ).sum()
    )
    summary["baseline_better_hourly_mae_days"] = int(
        (
            daily["baseline_hourly_mae_eur_mwh"]
            < daily["candidate_hourly_mae_eur_mwh"]
        ).sum()
    )
    summary["equal_hourly_mae_days"] = int(
        np.isclose(
            daily["baseline_hourly_mae_eur_mwh"],
            daily["candidate_hourly_mae_eur_mwh"],
        ).sum()
    )
    summary["dst_days"] = daily.loc[daily["hours"].ne(24), ["delivery_day", "hours"]].to_dict(
        orient="records"
    )
    return summary, daily


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", " ")
    return f"{float(value):.{digits}f}"


def _svg_chart(
    daily: pd.DataFrame,
    columns: Sequence[tuple[str, str, str]],
    *,
    title: str,
    height: int = 250,
) -> str:
    width = 1000
    left, right, top, bottom = 58, 18, 28, 35
    values = np.concatenate(
        [daily[column].to_numpy(dtype=float) for column, _, _ in columns]
    )
    finite = values[np.isfinite(values)]
    low = float(np.min(finite)) if len(finite) else 0.0
    high = float(np.max(finite)) if len(finite) else 1.0
    if math.isclose(low, high):
        low -= 0.5
        high += 0.5
    padding = (high - low) * 0.06
    low -= padding
    high += padding
    inner_w = width - left - right
    inner_h = height - top - bottom

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + (inner_w * index / max(len(daily) - 1, 1))
        y = top + inner_h * (high - value) / (high - low)
        return x, y

    elements = [
        f'<h3>{html.escape(title)}</h3>',
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
    ]
    for tick in range(5):
        value = low + (high - low) * tick / 4
        y = top + inner_h * (4 - tick) / 4
        elements.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>'
            f'<text class="axis" x="{left-7}" y="{y+4:.1f}" text-anchor="end">{value:.1f}</text>'
        )
    for column, label, css_class in columns:
        points = " ".join(
            f"{x:.2f},{y:.2f}"
            for index, value in enumerate(daily[column].to_numpy(float))
            for x, y in [point(index, value)]
        )
        elements.append(
            f'<polyline class="series {css_class}" points="{points}" fill="none"/>'
        )
    first_day = html.escape(str(daily["delivery_day"].iloc[0]))
    last_day = html.escape(str(daily["delivery_day"].iloc[-1]))
    elements.extend(
        [
            f'<text class="axis" x="{left}" y="{height-8}">{first_day}</text>',
            f'<text class="axis" x="{width-right}" y="{height-8}" text-anchor="end">{last_day}</text>',
            "</svg>",
            '<div class="legend">',
            *[
                f'<span><i class="swatch {css_class}"></i>{html.escape(label)}</span>'
                for _, label, css_class in columns
            ],
            "</div>",
        ]
    )
    return "".join(elements)


def render_report(
    metrics: Mapping[str, Any],
    daily: pd.DataFrame,
    audit: Mapping[str, Any],
    *,
    title: str = "POC Chronos-2 exogène — rolling 365 jours",
) -> str:
    """Return a fully self-contained HTML report with a persistent night mode."""

    gain = float(metrics["mae_gain_eur_mwh"])
    status_class = "good" if gain > 0 else "bad" if gain < 0 else "neutral"
    status_text = (
        "Le LoRA améliore le MAE brut"
        if gain > 0
        else "Le LoRA dégrade le MAE brut"
        if gain < 0
        else "MAE brut inchangé"
    )
    mae_chart = _svg_chart(
        daily,
        (
            ("baseline_hourly_mae_eur_mwh", "Chronos-2 base", "baseline"),
            ("candidate_hourly_mae_eur_mwh", "LoRA exogène", "candidate"),
        ),
        title="MAE horaire moyen par journée (EUR/MWh)",
    )
    price_chart = _svg_chart(
        daily,
        (
            ("observed_mean_price_eur_mwh", "Observé", "observed"),
            ("baseline_mean_price_eur_mwh", "Chronos-2 base", "baseline"),
            ("candidate_mean_price_eur_mwh", "LoRA exogène", "candidate"),
        ),
        title="Prix moyen journalier (EUR/MWh)",
    )
    table_rows = "".join(
        "<tr>"
        + f"<td>{html.escape(str(row.delivery_day))}</td>"
        + f"<td>{int(row.hours)}</td>"
        + f"<td>{row.observed_mean_price_eur_mwh:.2f}</td>"
        + f"<td>{row.baseline_mean_price_eur_mwh:.2f}</td>"
        + f"<td>{row.candidate_mean_price_eur_mwh:.2f}</td>"
        + f"<td>{row.baseline_hourly_mae_eur_mwh:.2f}</td>"
        + f"<td>{row.candidate_hourly_mae_eur_mwh:.2f}</td>"
        + "</tr>"
        for row in daily.itertuples(index=False)
    )
    title_safe = html.escape(title)
    audit_json = html.escape(_json_text(dict(audit)))
    return f"""<!doctype html>
<html lang="fr" data-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title_safe}</title>
<style>
:root{{--bg:#f3f6fb;--card:#fff;--text:#152033;--muted:#637087;--border:#dbe3ef;--grid:#dfe6f0;--baseline:#e58a18;--candidate:#087ec1;--observed:#263648;--good:#117a4b;--bad:#c33d45;--shadow:0 8px 28px rgba(23,40,70,.08)}}
[data-theme="dark"]{{--bg:#0c1320;--card:#151f2f;--text:#eaf0f8;--muted:#a8b5c8;--border:#2b3a50;--grid:#304057;--baseline:#ffb347;--candidate:#58c3ff;--observed:#e9eef6;--good:#58d493;--bad:#ff7b82;--shadow:none}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}} main{{max-width:1220px;margin:auto;padding:26px}} header{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}} h1{{font-size:25px;margin:0 0 6px}} h2{{margin:29px 0 12px;font-size:19px}} h3{{margin:0 0 8px;font-size:15px}} .muted{{color:var(--muted)}} button{{border:1px solid var(--border);background:var(--card);color:var(--text);padding:8px 12px;border-radius:9px;cursor:pointer}} .notice{{margin:18px 0;padding:13px 15px;border-left:4px solid var(--candidate);background:var(--card);border-radius:7px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}} .card,.chartbox,details{{background:var(--card);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow)}} .card{{padding:15px}} .card .value{{font-size:23px;font-weight:700;margin-top:5px}} .good{{color:var(--good)}} .bad{{color:var(--bad)}} .neutral{{color:var(--muted)}} .chartgrid{{display:grid;grid-template-columns:1fr;gap:14px}} .chartbox{{padding:14px}} .chart{{width:100%;height:auto}} .grid{{stroke:var(--grid);stroke-width:1}} .axis{{fill:var(--muted);font-size:12px}} .series{{stroke-width:2.1;vector-effect:non-scaling-stroke}} .series.baseline,.swatch.baseline{{stroke:var(--baseline);background:var(--baseline)}} .series.candidate,.swatch.candidate{{stroke:var(--candidate);background:var(--candidate)}} .series.observed,.swatch.observed{{stroke:var(--observed);background:var(--observed)}} .legend{{display:flex;flex-wrap:wrap;gap:16px;color:var(--muted)}} .swatch{{display:inline-block;width:18px;height:3px;margin:0 6px 3px 0}} table{{width:100%;border-collapse:collapse}} th,td{{padding:8px 10px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{position:sticky;top:0;background:var(--card)}} .tablewrap{{max-height:480px;overflow:auto}} details{{padding:12px 14px;margin-top:14px}} summary{{cursor:pointer;font-weight:650}} pre{{white-space:pre-wrap;overflow:auto;color:var(--muted)}} @media(max-width:650px){{main{{padding:15px}}header{{display:block}}button{{margin-top:10px}}}}
</style></head><body><main>
<header><div><h1>{title_safe}</h1><div class="muted">Comparaison causale, origine par origine, sur exactement {metrics['physical_days']} jours / {metrics['physical_hours']} heures physiques.</div></div><button id="theme" type="button">🌙 Mode nuit</button></header>
<div class="notice"><strong>Périmètre :</strong> Chronos-2 brut avec exactement les mêmes entrées exogènes pour la base et le candidat. Aucun correcteur résiduel, MKOnline ou Storm n'est appliqué dans ce rapport.</div>
<section class="cards">
<div class="card"><div class="muted">MAE base</div><div class="value">{_fmt(metrics['baseline_mae_eur_mwh'])}</div><div>EUR/MWh</div></div>
<div class="card"><div class="muted">MAE LoRA</div><div class="value">{_fmt(metrics['candidate_mae_eur_mwh'])}</div><div>EUR/MWh</div></div>
<div class="card"><div class="muted">Gain MAE</div><div class="value {status_class}">{_fmt(gain)}</div><div>{_fmt(100*metrics['mae_relative_gain'],2)} % — {status_text}</div></div>
<div class="card"><div class="muted">Prix moyen observé</div><div class="value">{_fmt(metrics['actual_mean_price_eur_mwh'],2)}</div><div>EUR/MWh</div></div>
</section>
<h2>Performance probabiliste</h2><section class="cards">
<div class="card"><div class="muted">Pinball moyen base</div><div class="value">{_fmt(metrics['baseline_pinball_mean'])}</div></div>
<div class="card"><div class="muted">Pinball moyen LoRA</div><div class="value">{_fmt(metrics['candidate_pinball_mean'])}</div></div>
<div class="card"><div class="muted">Couverture q10–q90 base</div><div class="value">{_fmt(100*metrics['baseline_q10_q90_coverage'],1)} %</div><div>cible nominale 80 %</div></div>
<div class="card"><div class="muted">Couverture q10–q90 LoRA</div><div class="value">{_fmt(100*metrics['candidate_q10_q90_coverage'],1)} %</div><div>cible nominale 80 %</div></div>
</section>
<h2>Évolution quotidienne</h2><section class="chartgrid"><div class="chartbox">{mae_chart}</div><div class="chartbox">{price_chart}</div></section>
<h2>Prix moyens sur la période</h2><section class="cards">
<div class="card"><div class="muted">Observé</div><div class="value">{_fmt(metrics['actual_mean_price_eur_mwh'],2)}</div><div>EUR/MWh</div></div>
<div class="card"><div class="muted">Base</div><div class="value">{_fmt(metrics['baseline_mean_price_eur_mwh'],2)}</div><div>EUR/MWh</div></div>
<div class="card"><div class="muted">LoRA</div><div class="value">{_fmt(metrics['candidate_mean_price_eur_mwh'],2)}</div><div>EUR/MWh</div></div>
<div class="card"><div class="muted">Jours MAE favorable au LoRA</div><div class="value">{metrics['candidate_better_hourly_mae_days']} / 365</div></div>
</section>
<details><summary>Détail des 365 journées</summary><div class="tablewrap"><table><thead><tr><th>Jour</th><th>Heures</th><th>Observé</th><th>Base</th><th>LoRA</th><th>MAE base</th><th>MAE LoRA</th></tr></thead><tbody>{table_rows}</tbody></table></div></details>
<details><summary>Audit d'inférence</summary><pre>{audit_json}</pre></details>
<p class="muted">Généré le {datetime.now(timezone.utc).isoformat()} · Les jours DST restent à 23/25 heures, sans interpolation.</p>
</main><script>
const root=document.documentElement,button=document.getElementById('theme');
function apply(theme){{root.dataset.theme=theme;button.textContent=theme==='dark'?'☀️ Mode jour':'🌙 Mode nuit';}}
const saved=localStorage.getItem('chronos2-exogenous-theme');apply(saved||((matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light'));
button.addEventListener('click',()=>{{const next=root.dataset.theme==='dark'?'light':'dark';localStorage.setItem('chronos2-exogenous-theme',next);apply(next);}});
</script></body></html>"""


def _verify_manifest_split(
    manifest: Mapping[str, Any], split: OriginSplit
) -> None:
    declared = manifest.get("splits", {}).get("evaluation_holdout", {})
    origins = pd.DatetimeIndex(split.evaluation).sort_values()
    expected = {
        "count": len(origins),
        "first_utc": origins[0].isoformat() if len(origins) else None,
        "last_utc": origins[-1].isoformat() if len(origins) else None,
    }
    if any(declared.get(key) != value for key, value in expected.items()):
        raise ExogenousEvaluationError(
            "Le split du panel ne correspond pas au holdout gelé du checkpoint."
        )


def _load_base_pipeline(
    config: ExogenousFineTuneConfig,
    pipeline_loader: PipelineLoader | None,
) -> Any:
    source = resolve_local_model_source(config) if pipeline_loader is None else config.model_id
    kwargs: dict[str, Any] = {
        "device_map": config.device_map,
        "local_files_only": config.local_files_only,
    }
    if config.model_revision and not Path(str(source)).is_dir():
        kwargs["revision"] = config.model_revision
    if pipeline_loader is not None:
        return pipeline_loader(source, kwargs)
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(source, **kwargs)


def run_evaluation(
    config_or_path: ExogenousFineTuneConfig | str | Path,
    *,
    run_directory: str | Path | None = None,
    panel_path: str | Path | None = None,
    panel_audit_path: str | Path | None = None,
    item_id: str | None = None,
    target_column: str | None = None,
    batch_size: int = 64,
    inference_chunk_size: int = 32,
    overwrite: bool = False,
    pipeline_loader: PipelineLoader | None = None,
    baseline_pipeline: Any | None = None,
    candidate_pipeline: Any | None = None,
    progress: Callable[[int, int, pd.Timestamp], None] | None = None,
    report_title: str = "POC Chronos-2 exogène — rolling 365 jours",
) -> EvaluationArtifacts:
    """Evaluate, validate and atomically publish the complete POC evidence."""

    config = (
        config_or_path
        if isinstance(config_or_path, ExogenousFineTuneConfig)
        else load_config(config_or_path)
    )
    run_dir = (
        Path(run_directory).expanduser().resolve()
        if run_directory is not None
        else config.output_directory
    )
    manifest = verify_bundle(run_dir)
    if (
        manifest.get("production_pipeline_evidence") is True
        or manifest.get("candidate_output_stage")
        == "exogenous_residual_corrected"
    ):
        raise ExogenousEvaluationError(
            "Backtest brut refuse: la preuve du pipeline final est deja "
            "scellee dans ce run. Creez un nouveau repertoire de candidat."
        )
    declared_panel_sha256 = manifest.get("panel_sha256")
    if not isinstance(declared_panel_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", declared_panel_sha256
    ) is None:
        raise ExogenousEvaluationError(
            "Le bundle ne scelle pas panel_sha256; backtest formel interdit."
        )
    current_panel_sha256 = _sha256_file(config.panel_path)
    if current_panel_sha256 != declared_panel_sha256:
        raise ExogenousEvaluationError(
            "Le panel courant diffère du panel scellé avant entraînement "
            f"(attendu={declared_panel_sha256}, reçu={current_panel_sha256})."
        )
    if (panel_path is None) != (panel_audit_path is None):
        raise ExogenousEvaluationError(
            "Backtest: --panel et --panel-audit doivent etre fournis ensemble."
        )
    frozen_panel = read_panel(config.panel_path)
    label_resolution: dict[str, Any] | None = None
    if panel_path is not None and panel_audit_path is not None:
        resolved_config = replace(
            config,
            panel_path=Path(panel_path).expanduser().resolve(),
            panel_audit_path=Path(panel_audit_path).expanduser().resolve(),
        )
        resolved_panel = read_panel(resolved_config.panel_path)
        try:
            validated, split, panel_audit, label_resolution = (
                bind_resolved_evaluation_panel(
                    frozen_frame=frozen_panel,
                    resolved_frame=resolved_panel,
                    frozen_config=config,
                    resolved_config=resolved_config,
                    experiment_manifest=manifest,
                )
            )
        except ExogenousFineTuneError as exc:
            raise ExogenousEvaluationError(str(exc)) from exc
        label_resolution.update(
            {
                "resolved_panel_path": str(resolved_config.panel_path),
                "resolved_panel_audit_path": str(resolved_config.panel_audit_path),
            }
        )
        existing_resolution = manifest.get("evaluation_label_resolution")
        if existing_resolution is not None and existing_resolution != label_resolution:
            raise ExogenousEvaluationError(
                "Une resolution holdout differente est deja scellee dans le bundle."
            )
    else:
        validated, split, panel_audit = validate_panel(frozen_panel, config)
        binding = panel_audit.get("evaluation_label_binding", {})
        if int(binding.get("unresolved_cells", 0)):
            raise ExogenousEvaluationError(
                "Backtest differe requis: le panel gele contient encore la target "
                "du dernier jour holdout non resolue; fournissez un panel resolu "
                "et son sidecar sans modifier le panel gele."
            )
    _verify_manifest_split(manifest, split)

    outputs = {
        "evidence": run_dir / "evaluation_predictions.csv.gz",
        "daily": run_dir / "evaluation_daily.csv.gz",
        "metrics": run_dir / "evaluation_metrics.json",
        "report": run_dir / "evaluation_report.html",
        "manifest": run_dir / "evaluation_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise ExogenousEvaluationError(
            "Artefacts d'évaluation déjà présents; --overwrite requis: "
            + ", ".join(str(path) for path in existing)
        )

    base = (
        baseline_pipeline
        if baseline_pipeline is not None
        else _load_base_pipeline(config, pipeline_loader)
    )
    candidate = (
        candidate_pipeline
        if candidate_pipeline is not None
        else load_checkpoint(
            run_dir,
            pipeline_loader=pipeline_loader,
            device_map=config.device_map,
        )
    )
    evidence, inference_audit = evaluate_holdout(
        validated,
        split,
        config,
        baseline_pipeline=base,
        candidate_pipeline=candidate,
        item_id=item_id,
        target_column=target_column,
        batch_size=batch_size,
        inference_chunk_size=inference_chunk_size,
        cache_directory=run_dir / "evaluation_cache",
        baseline_identity=(
            f"{resolve_local_model_source(config)}@{config.model_revision or 'local'}"
            if pipeline_loader is None
            else f"injected:{config.model_id}@{config.model_revision or 'local'}"
        ),
        candidate_identity=str(manifest["checkpoint_sha256"]),
        progress=progress,
    )
    metrics, daily = compute_metrics(evidence, timezone_name=config.timezone)
    combined_audit = {
        **inference_audit,
        "panel_pit_audit_passed": bool(panel_audit.get("pit_audit_passed")),
        "production_pit_evidence": bool(manifest.get("production_pit_evidence")),
        "dst_days": metrics["dst_days"],
    }
    report = render_report(metrics, daily, combined_audit, title=report_title)

    # The canonical publisher performs an independent strict check of the 365
    # physical days, D-1 cut-offs, UTC continuity and quantile order.
    evidence_path = publish_evaluation_evidence(
        run_dir, evidence.loc[:, list(EVALUATION_COLUMNS)], overwrite=overwrite
    )
    if label_resolution is not None:
        experiment_path = run_dir / "experiment_manifest.json"
        refreshed = json.loads(experiment_path.read_text(encoding="utf-8"))
        previous_resolution = refreshed.get("evaluation_label_resolution")
        if previous_resolution is not None and previous_resolution != label_resolution:
            raise ExogenousEvaluationError(
                "Une resolution holdout differente est deja scellee dans le bundle."
            )
        refreshed["evaluation_label_resolution"] = label_resolution
        _atomic_text(experiment_path, _json_text(refreshed))
        manifest = refreshed
    _atomic_csv_gz(outputs["daily"], daily)
    _atomic_text(outputs["metrics"], _json_text(metrics))
    _atomic_text(outputs["report"], report)

    evaluation_manifest = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "chronos2_exogenous_lora_rolling365_evaluation",
        "experiment_id": manifest.get("experiment_id"),
        "evaluation_role": manifest.get("evaluation_role"),
        "item_id": inference_audit["item_id"],
        "target_column": inference_audit["target_column"],
        "comparison": {
            "baseline": BASELINE_LABEL,
            "candidate": CANDIDATE_LABEL,
            "same_inputs": True,
            "cross_learning": False,
            "residual_corrector_applied": False,
        },
        "window": {
            "physical_days": metrics["physical_days"],
            "physical_hours": metrics["physical_hours"],
            "first_delivery_utc": metrics["first_delivery_utc"],
            "last_delivery_utc": metrics["last_delivery_utc"],
            "dst_days": metrics["dst_days"],
        },
        "input_contract_sha256": inference_audit["input_contract_sha256"],
        "evaluation_label_resolution": label_resolution,
        "artifacts": {
            name: {
                "relative_path": path.name,
                "sha256": _sha256_file(path),
            }
            for name, path in outputs.items()
            if name != "manifest"
        },
        "bundle_manifest_sha256": _sha256_file(run_dir / "experiment_manifest.json"),
    }
    _atomic_text(outputs["manifest"], _json_text(evaluation_manifest))
    return EvaluationArtifacts(
        evidence_path=evidence_path,
        daily_path=outputs["daily"],
        metrics_path=outputs["metrics"],
        report_path=outputs["report"],
        manifest_path=outputs["manifest"],
        metrics=metrics,
    )


def run_shadow(
    config_or_path: ExogenousFineTuneConfig | str | Path,
    *,
    run_directory: str | Path | None = None,
    panel_path: str | Path | None = None,
    panel_audit_path: str | Path | None = None,
    journal_path: str | Path | None = None,
    origins: Sequence[str | pd.Timestamp] | None = None,
    item_id: str | None = None,
    target_column: str | None = None,
    batch_size: int = 64,
    pipeline_loader: PipelineLoader | None = None,
    baseline_pipeline: Any | None = None,
    candidate_pipeline: Any | None = None,
    progress: Callable[[int, int, pd.Timestamp], None] | None = None,
) -> ShadowArtifacts:
    """Run isolated daily shadow forecasts and append them to the bundle log."""

    config = (
        config_or_path
        if isinstance(config_or_path, ExogenousFineTuneConfig)
        else load_config(config_or_path)
    )
    run_dir = (
        Path(run_directory).expanduser().resolve()
        if run_directory is not None
        else config.output_directory
    )
    bundle = verify_bundle(run_dir)
    resolved_panel = Path(panel_path or config.panel_path).expanduser().resolve()
    resolved_panel_audit = (
        Path(panel_audit_path).expanduser().resolve()
        if panel_audit_path is not None
        else (
            config.panel_audit_path
            if panel_path is None
            else resolved_panel.with_suffix(resolved_panel.suffix + ".audit.json")
        )
    )
    if not resolved_panel_audit.is_file():
        raise ExogenousEvaluationError(
            "Shadow: sidecar audit explicite obligatoire: "
            f"{resolved_panel_audit}."
        )
    try:
        raw_panel_audit = json.loads(
            resolved_panel_audit.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ExogenousEvaluationError(
            f"Shadow: sidecar panel illisible: {resolved_panel_audit}."
        ) from exc
    if not isinstance(raw_panel_audit, Mapping):
        raise ExogenousEvaluationError("Shadow: le sidecar panel doit etre un objet JSON.")
    source_panel = read_panel(resolved_panel)
    validated = _prepare_shadow_panel(source_panel, config)
    selected_item = _select_item(validated, config, item_id)
    selected_target = target_column or config.target_columns[0]
    available_origins = pd.DatetimeIndex(
        validated[config.origin_column].drop_duplicates()
    ).sort_values()
    requested = (
        pd.DatetimeIndex(pd.to_datetime(list(origins), utc=True)).sort_values()
        if origins
        else available_origins[-1:]
    )
    missing_origins = requested.difference(available_origins)
    if len(missing_origins):
        raise ExogenousEvaluationError(
            "Shadow: origines absentes: "
            + ", ".join(timestamp.isoformat() for timestamp in missing_origins)
        )
    if len(requested) != 1:
        raise ExogenousEvaluationError(
            "Shadow: un sidecar panel doit identifier une unique origine."
        )
    panel_evidence = _load_shadow_panel_evidence(
        resolved_panel,
        resolved_panel_audit,
        audit_payload=raw_panel_audit,
        config=config,
        zone=selected_item,
        origin=pd.Timestamp(requested[0]),
    )
    forecast_origins: list[pd.Timestamp] = []
    resolution_origins: list[pd.Timestamp] = []
    for origin in requested:
        group = validated.loc[
            validated[config.origin_column].eq(origin)
            & validated[config.item_column].eq(selected_item)
        ]
        _, _, actual = build_inference_input(
            group, config, allow_missing_actual=True
        )
        if np.isfinite(actual).all():
            resolution_origins.append(pd.Timestamp(origin))
        else:
            forecast_origins.append(pd.Timestamp(origin))
    if bool(resolution_origins) != bool(panel_evidence["horizon_actuals_present"]):
        raise ExogenousEvaluationError(
            "Shadow: horizon_actuals_present du sidecar ne correspond pas au panel."
        )

    resolved_journal = (
        Path(journal_path).expanduser().resolve()
        if journal_path is not None
        else run_dir / "shadow_predictions.csv.gz"
    )
    # Preflight every actual resolution before any new forecast is appended;
    # this keeps a mixed invocation atomic with respect to the no-retrospective rule.
    resolutions = (
        build_actual_resolution_input(
            validated,
            config,
            journal_path=resolved_journal,
            origins=resolution_origins,
            item_id=selected_item,
            target_column=selected_target,
        )
        if resolution_origins
        else None
    )

    results: list[ShadowArtifacts] = []
    if forecast_origins:
        base = (
            baseline_pipeline
            if baseline_pipeline is not None
            else _load_base_pipeline(config, pipeline_loader)
        )
        candidate = (
            candidate_pipeline
            if candidate_pipeline is not None
            else load_checkpoint(
                run_dir,
                pipeline_loader=pipeline_loader,
                device_map=config.device_map,
            )
        )
        predictions = daily_inference(
            validated,
            config,
            baseline_pipeline=base,
            candidate_pipeline=candidate,
            origins=forecast_origins,
            item_id=selected_item,
            target_column=selected_target,
            batch_size=batch_size,
            checkpoint_sha256=str(bundle["checkpoint_sha256"]),
            panel_evidence=panel_evidence,
            progress=progress,
        )
        results.append(
            append_shadow_predictions(
                run_dir, predictions, journal_path=resolved_journal
            )
        )
    if resolutions is not None:
        results.append(
            append_shadow_predictions(
                run_dir, resolutions, journal_path=resolved_journal
            )
        )
    if not results:  # pragma: no cover - at least one requested origin exists.
        raise ExogenousEvaluationError("Shadow: aucune origine traitée.")
    final = results[-1]
    return ShadowArtifacts(
        journal_path=final.journal_path,
        observed_evidence_path=final.observed_evidence_path,
        manifest_path=final.manifest_path,
        appended_rows=sum(result.appended_rows for result in results),
    )


__all__ = [
    "BASELINE_LABEL",
    "CANDIDATE_LABEL",
    "EvaluationArtifacts",
    "ExogenousEvaluationError",
    "QUANTILE_LEVELS",
    "SHADOW_INPUT_COLUMNS",
    "SHADOW_JOURNAL_COLUMNS",
    "ShadowArtifacts",
    "append_shadow_predictions",
    "build_actual_resolution_input",
    "build_inference_input",
    "compute_metrics",
    "daily_inference",
    "evaluate_holdout",
    "render_report",
    "run_evaluation",
    "run_shadow",
]
