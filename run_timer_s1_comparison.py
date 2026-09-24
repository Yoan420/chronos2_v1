#!/usr/bin/env python
"""Run an auditable Timer-S1 versus Chronos-2 comparison.

The command is deliberately action-based: displaying help or loading this
module never downloads a model.  Timer-S1 generation, the target-only
Chronos-2 denominator, and comparison publication are separate explicit
actions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import date
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.chronos_adapter import (
    ChronosDeliveryPlan,
    execute_grouped_chronos_backtest,
    generate_delivery_plans,
)
from chronos2_hourly.timer_s1_adapter import (
    DEFAULT_TIMER_S1_MODEL_ID,
    DEFAULT_TIMER_S1_REVISION,
    load_timer_s1_runtime,
    run_timer_s1_backtest,
)
from chronos2_hourly.timer_s1_comparison import (
    SourceProtocol,
    compare_same_downstream_features,
    compare_target_only_native,
    compare_target_only_same_downstream_features,
    json_safe,
    legacy_target_hashes_by_timestamp_unit,
    load_source_protocol,
    paired_daily_mae_bootstrap,
    read_quantile_artifact,
    sha256_file,
    sha256_target_series,
    validate_timer_oof,
)


LOGGER = logging.getLogger("timer_s1_comparison")
SCRIPT_VERSION = "1.0.1"
GIB = 1024**3
TIMER_MIN_AVAILABLE_RAM_GIB = 24.0
TIMER_RECOMMENDED_GPU_GIB = 40.0
POST_PUBLICATION_START_DAY = date(2026, 4, 10)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Comparaison FR, DST-safe et sans fuite, entre Timer-S1 et le "
            "run Chronos-2 gele. Aucune action n'est lancee implicitement."
        )
    )
    parser.add_argument(
        "--config",
        default="config/timer_s1_comparison.yaml",
        help="Configuration YAML de l'experience.",
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--plan-only",
        action="store_true",
        help="Valide le protocole et ecrit le plan; ne charge aucun modele.",
    )
    actions.add_argument(
        "--generate-timer",
        action="store_true",
        help="Genere/reprend l'OOF Timer-S1 target-only.",
    )
    actions.add_argument(
        "--generate-chronos-target-only",
        action="store_true",
        help="Genere/reprend le denominateur Chronos-2 target-only.",
    )
    actions.add_argument(
        "--compare-only",
        action="store_true",
        help="Compare des OOF deja generes; aucun modele n'est charge.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Remplace comparison.output_directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Interdit tout telechargement lors d'une action de generation.",
    )
    parser.add_argument(
        "--allow-low-memory",
        action="store_true",
        help=(
            "Contourne explicitement la garde memoire Timer-S1; risque "
            "d'arret OOM."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Remplace comparison.threads pour le correcteur residuel.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: la racine YAML doit etre un mapping.")
    return dict(value)


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} doit etre un mapping YAML.")
    return dict(value)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} doit etre un entier strictement positif.")
    parsed = int(value)
    if parsed != value or parsed < 1:
        raise ValueError(f"{name} doit etre un entier strictement positif.")
    return parsed


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolved_settings(
    config_path: Path,
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    project_value = config.get("project_root", "..")
    project_root = _resolve_path(str(project_value), base=config_path.parent)
    comparison = _mapping(config.get("comparison"), name="comparison")
    source_run = _resolve_path(
        str(
            comparison.get(
                "source_run",
                "runs/chronos2_hourly_fr_residual_extended_v1",
            )
        ),
        base=project_root,
    )
    output_value = args.output_dir or comparison.get(
        "output_directory", "runs/experiments/timer_s1_fr"
    )
    output_dir = _resolve_path(str(output_value), base=project_root)
    timer_value = (
        output_dir / "timer_s1_oof.csv.gz"
        if args.output_dir
        else comparison.get("timer_oof_file", output_dir / "timer_s1_oof.csv.gz")
    )
    chronos_value = (
        output_dir / "chronos2_target_only_oof.csv.gz"
        if args.output_dir
        else comparison.get(
            "chronos_target_only_oof_file",
            output_dir / "chronos2_target_only_oof.csv.gz",
        )
    )
    timer_file = _resolve_path(str(timer_value), base=project_root)
    chronos_file = _resolve_path(str(chronos_value), base=project_root)
    settings = {
        "project_root": project_root,
        "source_run": source_run,
        "output_dir": output_dir,
        "timer_oof_file": timer_file,
        "chronos_target_only_oof_file": chronos_file,
        "timezone": str(comparison.get("timezone", "Europe/Paris")),
        "extended_days": _positive_int(
            comparison.get("extended_days", 223), name="extended_days"
        ),
        "calibration_days": _positive_int(
            comparison.get("calibration_days", 365), name="calibration_days"
        ),
        "evaluation_days": _positive_int(
            comparison.get("evaluation_days", 365), name="evaluation_days"
        ),
        "context_length": _positive_int(
            comparison.get("context_length", 2048), name="context_length"
        ),
        "checkpoint_days": _positive_int(
            comparison.get("checkpoint_days", 7), name="checkpoint_days"
        ),
        "bootstrap_samples": _positive_int(
            comparison.get("bootstrap_samples", 20_000),
            name="bootstrap_samples",
        ),
        "seed": int(comparison.get("seed", 42)),
        "threads": int(
            args.threads
            if args.threads is not None
            else comparison.get("threads", -1)
        ),
        "timer": _mapping(config.get("timer_s1"), name="timer_s1"),
        "chronos": _mapping(
            config.get("chronos2_target_only"),
            name="chronos2_target_only",
        ),
    }
    frozen_contract = {
        "timezone": "Europe/Paris",
        "extended_days": 223,
        "calibration_days": 365,
        "evaluation_days": 365,
        "context_length": 2048,
    }
    mismatches = {
        key: (settings[key], expected)
        for key, expected in frozen_contract.items()
        if settings[key] != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}={observed!r} (attendu {expected!r})"
            for key, (observed, expected) in mismatches.items()
        )
        raise ValueError(
            "Le protocole de comparaison est gele; configuration refusee: "
            + details
            + "."
        )
    return settings


def _load_protocol(settings: Mapping[str, Any]) -> SourceProtocol:
    return load_source_protocol(
        settings["source_run"],
        timezone=str(settings["timezone"]),
        extended_days=int(settings["extended_days"]),
        calibration_days=int(settings["calibration_days"]),
        evaluation_days=int(settings["evaluation_days"]),
    )


def _plans_for_protocol(protocol: SourceProtocol) -> tuple[ChronosDeliveryPlan, ...]:
    expected_index = protocol.full_oof_index
    local_dates = pd.Index(expected_index.tz_convert(protocol.timezone).date)
    days = local_dates.unique().tolist()
    plans = generate_delivery_plans(
        days[0],
        days[-1],
        forecast_origin_local_time="08:00",
        forecast_days_before=1,
        timezone=protocol.timezone,
    )
    plan_index = plans[0].delivery_index_utc
    if len(plans) > 1:
        plan_index = plan_index.append(
            [plan.delivery_index_utc for plan in plans[1:]]
        )
    plan_index = pd.DatetimeIndex(plan_index, name="delivery_start_utc")
    if not plan_index.equals(expected_index):
        raise ValueError(
            "Les plans D-1 08:00 ne reproduisent pas exactement l'index OOF gele."
        )
    frozen_origins = pd.DatetimeIndex(
        pd.to_datetime(
            pd.concat(
                [
                    protocol.chronos_extended["forecast_origin_utc"],
                    protocol.chronos_main["forecast_origin_utc"],
                ]
            ),
            utc=True,
        )
    )
    planned_origins = pd.DatetimeIndex(
        [
            plan.forecast_origin_utc
            for plan in plans
            for _ in range(plan.horizon)
        ]
    )
    if not planned_origins.equals(frozen_origins):
        raise ValueError(
            "Les origines D-1 08:00 ne reproduisent pas les origines OOF gelees."
        )
    return plans


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _resource_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "accelerate",
                "chronos-forecasting",
                "pandas",
                "numpy",
            )
        },
        "memory": {},
        "cuda_devices": [],
    }
    try:
        import psutil

        virtual = psutil.virtual_memory()
        snapshot["memory"] = {
            "total_gib": float(virtual.total / GIB),
            "available_gib": float(virtual.available / GIB),
        }
    except Exception as exc:  # pragma: no cover - host dependent
        snapshot["memory"] = {"inspection_error": str(exc)}
    try:
        import torch

        for position in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(position)
            snapshot["cuda_devices"].append(
                {
                    "index": position,
                    "name": properties.name,
                    "total_gib": float(properties.total_memory / GIB),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    except Exception as exc:  # pragma: no cover - host dependent
        snapshot["cuda_inspection_error"] = str(exc)
    return snapshot


def _require_transformers_457() -> str:
    installed = _package_version("transformers")
    if installed is None:
        raise RuntimeError(
            "Timer-S1 exige transformers 4.57.x; le paquet est absent. "
            "Installez requirements_timer_s1.txt."
        )
    try:
        from packaging.version import Version

        parsed = Version(installed)
        compatible = Version("4.57.1") <= parsed < Version("4.58")
    except Exception:
        compatible = installed.startswith("4.57.")
    if not compatible:
        raise RuntimeError(
            "Timer-S1 exige transformers>=4.57.1,<4.58; "
            f"version detectee: {installed}."
        )
    return installed


def _require_timer_memory(
    snapshot: Mapping[str, Any],
    *,
    allow_low_memory: bool,
) -> None:
    cuda = list(snapshot.get("cuda_devices", ()))
    available_ram = float(
        _mapping(snapshot.get("memory"), name="resource_snapshot.memory").get(
            "available_gib", 0.0
        )
    )
    sufficient_gpu = any(
        float(item.get("total_gib", 0.0)) >= TIMER_RECOMMENDED_GPU_GIB
        for item in cuda
    )
    sufficient_cpu = available_ram >= TIMER_MIN_AVAILABLE_RAM_GIB
    if sufficient_gpu or sufficient_cpu:
        return
    message = (
        "Ressources insuffisantes pour charger prudemment Timer-S1 BF16: "
        f"GPU recommande >= {TIMER_RECOMMENDED_GPU_GIB:.0f} GiB, ou RAM "
        f"systeme disponible >= {TIMER_MIN_AVAILABLE_RAM_GIB:.0f} GiB. "
        f"RAM disponible detectee={available_ram:.1f} GiB, "
        f"GPU CUDA detecte={bool(cuda)}."
    )
    if not allow_low_memory:
        raise RuntimeError(message + " Utilisez --allow-low-memory a vos risques.")
    LOGGER.warning("%s Garde contournee explicitement.", message)


def _atomic_write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(json_safe(value), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.csv.gz" if path.name.endswith(".gz") else ".tmp.csv"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        output = frame.copy()
        if isinstance(output.index, pd.DatetimeIndex):
            index_name = output.index.name or "delivery_start_utc"
            output.index.name = index_name
            output = output.reset_index()
        compression = "gzip" if path.name.endswith(".gz") else None
        output.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _expected_frame(
    plans: Sequence[ChronosDeliveryPlan],
    protocol: SourceProtocol,
) -> pd.DataFrame:
    pieces = []
    for plan in plans:
        index = plan.delivery_index_utc
        pieces.append(
            pd.DataFrame(
                {
                    "forecast_origin_utc": plan.forecast_origin_utc,
                    "actual": protocol.target.loc[index].to_numpy(dtype=float),
                },
                index=index,
            )
        )
    result = pd.concat(pieces)
    result.index.name = "delivery_start_utc"
    return result


def _validate_generated_frame(
    frame: pd.DataFrame,
    plans: Sequence[ChronosDeliveryPlan],
    protocol: SourceProtocol,
    *,
    name: str,
) -> pd.DataFrame:
    expected = _expected_frame(plans, protocol)
    if not frame.index.equals(expected.index):
        raise ValueError(f"{name}: couverture horaire differente des plans attendus.")
    observed_origins = pd.DatetimeIndex(
        pd.to_datetime(frame["forecast_origin_utc"], utc=True)
    ).as_unit("ns")
    expected_origins = pd.DatetimeIndex(
        pd.to_datetime(expected["forecast_origin_utc"], utc=True)
    ).as_unit("ns")
    if not observed_origins.equals(expected_origins):
        raise ValueError(f"{name}: origines differentes du protocole gele.")
    if not np.allclose(
        frame["actual"].to_numpy(dtype=float),
        expected["actual"].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-5,
    ):
        raise ValueError(f"{name}: valeurs reelles differentes de la cible gelee.")
    return frame


def _plan_chunks(
    plans: Sequence[ChronosDeliveryPlan],
    *,
    days: int,
) -> list[tuple[ChronosDeliveryPlan, ...]]:
    return [tuple(plans[start : start + days]) for start in range(0, len(plans), days)]


def _checkpoint_path(
    directory: Path,
    chunk: Sequence[ChronosDeliveryPlan],
) -> Path:
    first = chunk[0].delivery_date.strftime("%Y%m%d")
    last = chunk[-1].delivery_date.strftime("%Y%m%d")
    return directory / f"{first}_{last}.csv.gz"


def _sidecar_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.name + ".manifest.json")


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: manifest JSON invalide.")
    return dict(value)


def _require_generation_sidecar(
    artifact: Path,
    *,
    model_name: str,
    model_id: str,
    revision: str,
    context_length: int,
    inference_contract: Mapping[str, Any],
    protocol: SourceProtocol,
) -> dict[str, Any]:
    """Authenticate an existing forecast before reuse or comparison."""

    sidecar_path = _sidecar_path(artifact)
    if not sidecar_path.is_file():
        raise ValueError(
            f"{artifact}: sidecar de provenance absent ({sidecar_path.name}). "
            "Le CSV ne sera ni reutilise ni recertifie."
        )
    sidecar = _read_json_mapping(sidecar_path)
    expected = {
        "model_name": model_name,
        "model_id": model_id,
        "revision": revision,
        "context_length": int(context_length),
        "forecast_mode": "strict_native_target_only",
        "native_covariates": [],
        "source_feature_manifest_sha256": protocol.feature_manifest_sha256,
    }
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise ValueError(
                f"{sidecar_path}: {key}={sidecar.get(key)!r}, attendu {value!r}."
            )
    canonical_target_hash = sha256_target_series(protocol.target)
    recorded_target_hash = sidecar.get("source_target_sha256")
    if recorded_target_hash != canonical_target_hash:
        legacy_hashes = legacy_target_hashes_by_timestamp_unit(protocol.target)
        legacy_unit = next(
            (
                unit
                for unit, candidate in legacy_hashes.items()
                if candidate == recorded_target_hash
            ),
            None,
        )
        if legacy_unit is None:
            raise ValueError(
                f"{sidecar_path}: source_target_sha256="
                f"{recorded_target_hash!r}, attendu {canonical_target_hash!r}."
            )
        LOGGER.warning(
            "%s: empreinte cible historique pandas datetime64[%s] acceptee; "
            "normalisation auditee vers UTC datetime64[ns].",
            sidecar_path,
            legacy_unit,
        )
        sidecar = dict(sidecar)
        sidecar["source_target_sha256_recorded"] = recorded_target_hash
        sidecar["source_target_sha256"] = canonical_target_hash
        sidecar["source_target_hash_compatibility"] = {
            "status": "accepted_legacy_datetime_unit",
            "legacy_datetime_unit": legacy_unit,
            "canonical_datetime_unit": "ns",
        }
    observed_contract = sidecar.get("inference_contract")
    if not isinstance(observed_contract, Mapping):
        raise ValueError(f"{sidecar_path}: inference_contract absent.")
    for key, value in inference_contract.items():
        if observed_contract.get(key) != value:
            raise ValueError(
                f"{sidecar_path}: inference_contract.{key}="
                f"{observed_contract.get(key)!r}, attendu {value!r}."
            )
    expected_hash = sidecar.get("artifact_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{sidecar_path}: artifact_sha256 absent ou invalide.")
    actual_hash = sha256_file(artifact)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{artifact}: SHA-256 different du sidecar; artefact refuse."
        )
    return sidecar


def _checkpoint_manifest(
    *,
    artifact: Path,
    model_name: str,
    model_id: str,
    revision: str,
    context_length: int,
    inference_contract: Mapping[str, Any],
    protocol: SourceProtocol,
    plans: Sequence[ChronosDeliveryPlan],
) -> dict[str, Any]:
    return {
        "script": "run_timer_s1_comparison.py",
        "script_version": SCRIPT_VERSION,
        "artifact_scope": "complete_local_day_checkpoint",
        "model_name": model_name,
        "model_id": model_id,
        "revision": revision,
        "forecast_mode": "strict_native_target_only",
        "native_covariates": [],
        "inference_contract": dict(inference_contract),
        "context_length": int(context_length),
        "first_delivery_day": str(plans[0].delivery_date),
        "last_delivery_day": str(plans[-1].delivery_date),
        "delivery_days": len(plans),
        "delivery_hours": int(sum(plan.horizon for plan in plans)),
        "source_run": str(protocol.source_run),
        "source_feature_manifest_sha256": protocol.feature_manifest_sha256,
        "source_target_sha256": sha256_target_series(protocol.target),
        "pit_covariate_diagnostics": dict(protocol.pit_covariate_diagnostics),
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
    }


def _generator_manifest(
    *,
    model_name: str,
    model_id: str,
    revision: str,
    inference_contract: Mapping[str, Any],
    protocol: SourceProtocol,
    settings: Mapping[str, Any],
    plans: Sequence[ChronosDeliveryPlan],
    output_path: Path,
    checkpoint_directory: Path,
    resource_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    horizons = pd.Series([plan.horizon for plan in plans]).value_counts().sort_index()
    return {
        "script": "run_timer_s1_comparison.py",
        "script_version": SCRIPT_VERSION,
        "model_name": model_name,
        "model_id": model_id,
        "revision": revision,
        "forecast_mode": "strict_native_target_only",
        "inference_contract": dict(inference_contract),
        "native_covariates": [],
        "context_length": int(settings["context_length"]),
        "timezone": protocol.timezone,
        "forecast_origin_contract": "delivery_day_minus_1_at_08:00_local",
        "delivery_day_contract": "complete_Europe_Paris_day_23_24_25_hours",
        "first_delivery_day": str(plans[0].delivery_date),
        "last_delivery_day": str(plans[-1].delivery_date),
        "delivery_days": len(plans),
        "delivery_hours": int(sum(plan.horizon for plan in plans)),
        "horizon_day_counts": {
            str(int(horizon)): int(count) for horizon, count in horizons.items()
        },
        "source_run": str(protocol.source_run),
        "source_feature_manifest_sha256": protocol.feature_manifest_sha256,
        "source_target_sha256": sha256_target_series(protocol.target),
        "pit_covariate_diagnostics": dict(protocol.pit_covariate_diagnostics),
        "checkpoint_directory": str(checkpoint_directory),
        "checkpoint_days": int(settings["checkpoint_days"]),
        "output": str(output_path),
        "artifact_sha256": sha256_file(output_path),
        "resource_snapshot": resource_snapshot,
    }


def _generate_resumable(
    *,
    model_name: str,
    model_id: str,
    revision: str,
    inference_contract: Mapping[str, Any],
    plans: Sequence[ChronosDeliveryPlan],
    protocol: SourceProtocol,
    settings: Mapping[str, Any],
    output_path: Path,
    checkpoint_directory: Path,
    resource_snapshot: Mapping[str, Any],
    generate_chunk: Callable[[tuple[ChronosDeliveryPlan, ...]], pd.DataFrame],
) -> Path:
    if output_path.is_file():
        _require_generation_sidecar(
            output_path,
            model_name=model_name,
            model_id=model_id,
            revision=revision,
            context_length=int(settings["context_length"]),
            inference_contract=inference_contract,
            protocol=protocol,
        )
        complete = read_quantile_artifact(output_path)
        _validate_generated_frame(
            complete, plans, protocol, name=f"{model_name} OOF final existant"
        )
        LOGGER.info("OOF %s deja complet et valide: %s", model_name, output_path)
        return output_path
    else:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        completed: list[pd.DataFrame] = []
        chunks = _plan_chunks(plans, days=int(settings["checkpoint_days"]))
        for number, chunk in enumerate(chunks, start=1):
            checkpoint = _checkpoint_path(checkpoint_directory, chunk)
            if checkpoint.is_file():
                _require_generation_sidecar(
                    checkpoint,
                    model_name=model_name,
                    model_id=model_id,
                    revision=revision,
                    context_length=int(settings["context_length"]),
                    inference_contract=inference_contract,
                    protocol=protocol,
                )
                frame = read_quantile_artifact(checkpoint)
                _validate_generated_frame(
                    frame,
                    chunk,
                    protocol,
                    name=f"checkpoint {checkpoint.name}",
                )
                LOGGER.info(
                    "Reprise %s: checkpoint %d/%d deja valide.",
                    model_name,
                    number,
                    len(chunks),
                )
            else:
                frame = generate_chunk(chunk)
                _validate_generated_frame(
                    frame,
                    chunk,
                    protocol,
                    name=f"checkpoint {checkpoint.name}",
                )
                _atomic_write_frame(frame, checkpoint)
                _atomic_write_json(
                    _checkpoint_manifest(
                        artifact=checkpoint,
                        model_name=model_name,
                        model_id=model_id,
                        revision=revision,
                        context_length=int(settings["context_length"]),
                        inference_contract=inference_contract,
                        protocol=protocol,
                        plans=chunk,
                    ),
                    _sidecar_path(checkpoint),
                )
                LOGGER.info(
                    "%s: checkpoint atomique %d/%d publie (%s a %s).",
                    model_name,
                    number,
                    len(chunks),
                    chunk[0].delivery_date,
                    chunk[-1].delivery_date,
                )
            completed.append(frame)
        complete = pd.concat(completed).sort_index(kind="stable")
        _validate_generated_frame(
            complete, plans, protocol, name=f"{model_name} OOF final"
        )
        _atomic_write_frame(complete, output_path)

    manifest = _generator_manifest(
        model_name=model_name,
        model_id=model_id,
        revision=revision,
        inference_contract=inference_contract,
        protocol=protocol,
        settings=settings,
        plans=plans,
        output_path=output_path,
        checkpoint_directory=checkpoint_directory,
        resource_snapshot=resource_snapshot,
    )
    manifest_path = _sidecar_path(output_path)
    _atomic_write_json(manifest, manifest_path)
    LOGGER.info("Manifest publie: %s", manifest_path)
    return output_path


def _timer_dtype_name(configured: str) -> str:
    name = configured.strip().casefold()
    aliases = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float32": "float32",
        "fp32": "float32",
    }
    if name not in aliases:
        raise ValueError("timer_s1.torch_dtype doit valoir bfloat16 ou float32.")
    return aliases[name]


def _timer_dtype(torch_module: Any, configured: str) -> Any:
    return getattr(torch_module, _timer_dtype_name(configured))


def generate_timer(
    protocol: SourceProtocol,
    plans: Sequence[ChronosDeliveryPlan],
    settings: Mapping[str, Any],
    args: argparse.Namespace,
) -> Path:
    timer = dict(settings["timer"])
    resources = _resource_snapshot()
    model_id = str(timer.get("model_id", DEFAULT_TIMER_S1_MODEL_ID))
    revision = str(timer.get("revision", DEFAULT_TIMER_S1_REVISION))
    local_only = bool(args.local_files_only or timer.get("local_files_only", False))
    batch_size = _positive_int(timer.get("batch_size", 1), name="timer_s1.batch_size")
    model_kwargs = _mapping(timer.get("model_kwargs"), name="timer_s1.model_kwargs")
    if model_kwargs.get("use_cache", False) not in (False, 0):
        raise ValueError("Timer-S1 comparison impose model_kwargs.use_cache=false.")
    model_kwargs["use_cache"] = False
    dtype_name = _timer_dtype_name(str(timer.get("torch_dtype", "bfloat16")))
    inference_contract = {
        "revin": True,
        "use_cache": False,
        "torch_dtype": dtype_name,
        "model_kwargs": model_kwargs,
        "quantile_indices": {"q10": 0, "q50": 4, "q90": 8},
    }
    runtime_holder: list[Any] = []

    def generate_chunk(chunk: tuple[ChronosDeliveryPlan, ...]) -> pd.DataFrame:
        if not runtime_holder:
            _require_transformers_457()
            _require_timer_memory(
                resources,
                allow_low_memory=bool(args.allow_low_memory),
            )
            import torch

            dtype = _timer_dtype(torch, dtype_name)
            LOGGER.info(
                "Chargement explicite de Timer-S1 (%s, revision %s).",
                model_id,
                revision,
            )
            runtime_holder.append(
                load_timer_s1_runtime(
                    model_id=model_id,
                    revision=revision,
                    device_map=timer.get("device_map", "auto"),
                    local_files_only=local_only,
                    torch_dtype=dtype,
                    model_kwargs=model_kwargs,
                )
            )
        return run_timer_s1_backtest(
            chunk,
            data=protocol.target,
            context_length=int(settings["context_length"]),
            batch_size=batch_size,
            runtime=runtime_holder[0],
        )

    output_path = Path(settings["timer_oof_file"])
    return _generate_resumable(
        model_name="timer_s1",
        model_id=model_id,
        revision=revision,
        inference_contract=inference_contract,
        plans=plans,
        protocol=protocol,
        settings=settings,
        output_path=output_path,
        checkpoint_directory=Path(settings["output_dir"]) / "checkpoints" / "timer_s1",
        resource_snapshot=resources,
        generate_chunk=generate_chunk,
    )


def _resolve_chronos_source(
    model_id: str,
    revision: str,
    *,
    local_files_only: bool,
) -> str:
    if not local_files_only:
        return model_id
    from huggingface_hub import snapshot_download

    return str(
        Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_files_only=True,
            )
        ).resolve()
    )


def _load_chronos_target_only_runtime(
    config: Mapping[str, Any],
    *,
    force_local_only: bool,
) -> tuple[Any, str, str, dict[str, Any]]:
    import torch
    from chronos import Chronos2Pipeline
    from chronos2_modular.common import resolve_device

    model_id = str(config.get("model_id", "amazon/chronos-2"))
    revision_value = config.get("revision")
    if revision_value in (None, ""):
        raise ValueError(
            "chronos2_target_only.revision doit etre un commit immuable."
        )
    revision = str(revision_value)
    local_only = bool(force_local_only or config.get("local_files_only", True))
    device, dtype = resolve_device(str(config.get("device", "auto")))
    source = _resolve_chronos_source(
        model_id, revision, local_files_only=local_only
    )
    kwargs: dict[str, Any] = {
        "device_map": device,
        "local_files_only": local_only,
    }
    if source == model_id:
        kwargs["revision"] = revision
    try:
        pipeline = Chronos2Pipeline.from_pretrained(source, dtype=dtype, **kwargs)
    except TypeError:
        pipeline = Chronos2Pipeline.from_pretrained(
            source, torch_dtype=dtype, **kwargs
        )
    resources = _resource_snapshot()
    resources["chronos_resolved_source"] = source
    resources["chronos_device"] = device
    resources["chronos_dtype"] = str(dtype)
    return pipeline, model_id, revision, resources


def _chronos_target_only_executor(
    *,
    protocol: SourceProtocol,
    pipeline: Any,
    context_length: int,
    batch_size: int,
) -> Callable[..., pd.DataFrame]:
    target = protocol.target

    def execute(
        *,
        plans: Sequence[ChronosDeliveryPlan],
        horizon: int,
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for start in range(0, len(plans), batch_size):
            batch = tuple(plans[start : start + batch_size])
            contexts: list[np.ndarray] = []
            actuals: list[np.ndarray] = []
            for plan in batch:
                position = target.index.get_loc(plan.delivery_start_utc)
                if not isinstance(position, (int, np.integer)):
                    raise ValueError("Origine Chronos-2 target-only ambigue.")
                position = int(position)
                if position < context_length:
                    raise ValueError(
                        f"Contexte < {context_length} h pour {plan.delivery_date}."
                    )
                context = target.iloc[position - context_length : position]
                expected_end = plan.delivery_start_utc - pd.Timedelta(hours=1)
                if context.index[-1] != expected_end or len(context) != context_length:
                    raise ValueError("Contexte Chronos-2 non contigu au delivery day.")
                contexts.append(context.to_numpy(dtype=np.float32))
                actuals.append(
                    target.loc[plan.delivery_index_utc].to_numpy(dtype=float)
                )

            prediction = pipeline.predict_quantiles(
                inputs=contexts,
                prediction_length=int(horizon),
                quantile_levels=[0.1, 0.5, 0.9],
                batch_size=batch_size,
                context_length=context_length,
                cross_learning=False,
            )
            quantile_tensors = prediction[0] if isinstance(prediction, tuple) else prediction
            if len(quantile_tensors) != len(batch):
                raise ValueError("Chronos-2 a retourne un nombre de series inattendu.")
            for plan, actual, tensor in zip(batch, actuals, quantile_tensors):
                value = tensor.detach().float().cpu().numpy()
                if value.shape == (1, horizon, 3):
                    value = value[0]
                if value.shape != (horizon, 3):
                    raise ValueError(
                        "Chronos-2 target-only doit retourner [H,3] par serie; "
                        f"recu {value.shape}."
                    )
                rows.append(
                    pd.DataFrame(
                        {
                            "delivery_start_utc": plan.delivery_index_utc,
                            "q10": value[:, 0],
                            "q50": value[:, 1],
                            "q90": value[:, 2],
                            "actual": actual,
                        }
                    )
                )
        return pd.concat(rows, ignore_index=True)

    return execute


def generate_chronos_target_only(
    protocol: SourceProtocol,
    plans: Sequence[ChronosDeliveryPlan],
    settings: Mapping[str, Any],
    args: argparse.Namespace,
) -> Path:
    chronos_config = dict(settings["chronos"])
    model_id = str(chronos_config.get("model_id", "amazon/chronos-2"))
    revision_value = chronos_config.get("revision")
    if revision_value in (None, ""):
        raise ValueError(
            "chronos2_target_only.revision doit etre un commit immuable."
        )
    revision = str(revision_value)
    resources = _resource_snapshot()
    batch_size = _positive_int(
        chronos_config.get("batch_size", 8),
        name="chronos2_target_only.batch_size",
    )
    executor_holder: list[Callable[..., pd.DataFrame]] = []

    def generate_chunk(chunk: tuple[ChronosDeliveryPlan, ...]) -> pd.DataFrame:
        if not executor_holder:
            pipeline, loaded_id, loaded_revision, loaded_resources = (
                _load_chronos_target_only_runtime(
                    chronos_config,
                    force_local_only=bool(args.local_files_only),
                )
            )
            if loaded_id != model_id or loaded_revision != revision:
                raise ValueError("Le runtime Chronos-2 ne respecte pas le pin config.")
            resources.update(loaded_resources)
            executor_holder.append(
                _chronos_target_only_executor(
                    protocol=protocol,
                    pipeline=pipeline,
                    context_length=int(settings["context_length"]),
                    batch_size=batch_size,
                )
            )
        return execute_grouped_chronos_backtest(chunk, executor_holder[0])

    output_path = Path(settings["chronos_target_only_oof_file"])
    return _generate_resumable(
        model_name="chronos2_target_only",
        model_id=model_id,
        revision=revision,
        inference_contract={
            "cross_learning": False,
            "quantile_levels": [0.1, 0.5, 0.9],
        },
        plans=plans,
        protocol=protocol,
        settings=settings,
        output_path=output_path,
        checkpoint_directory=(
            Path(settings["output_dir"]) / "checkpoints" / "chronos2_target_only"
        ),
        resource_snapshot=resources,
        generate_chunk=generate_chunk,
    )


def _current_reference_metrics(protocol: SourceProtocol) -> dict[str, float]:
    evaluation = protocol.current_backtest.loc[protocol.evaluation_index]
    actual = protocol.target.loc[protocol.evaluation_index].to_numpy(dtype=float)
    result = {}
    for model, column in (
        ("chronos2_native", "chronos2__q50"),
        ("chronos2_current_corrected", "residual_corrected__q50"),
    ):
        prediction = pd.to_numeric(evaluation[column], errors="coerce").to_numpy(
            dtype=float
        )
        result[f"{model}_mae"] = float(np.mean(np.abs(prediction - actual)))
    return result


def write_plan(
    protocol: SourceProtocol,
    plans: Sequence[ChronosDeliveryPlan],
    settings: Mapping[str, Any],
) -> Path:
    resources = _resource_snapshot()
    sensitivity_days = pd.Index(
        protocol.evaluation_index.tz_convert(protocol.timezone).date
    )
    sensitivity_mask = np.asarray(
        sensitivity_days >= POST_PUBLICATION_START_DAY, dtype=bool
    )
    transformers_version = resources["packages"].get("transformers")
    compatible = False
    if transformers_version:
        try:
            from packaging.version import Version

            parsed = Version(transformers_version)
            compatible = Version("4.57.1") <= parsed < Version("4.58")
        except Exception:
            compatible = str(transformers_version).startswith("4.57.")
    horizons = pd.Series([plan.horizon for plan in plans]).value_counts().sort_index()
    plan = {
        "script_version": SCRIPT_VERSION,
        "status": "validated_no_model_loaded",
        "no_download_performed": True,
        "source_run": str(protocol.source_run),
        "source_feature_manifest_sha256": protocol.feature_manifest_sha256,
        "source_target_sha256": sha256_target_series(protocol.target),
        "source_residual_recipe_sha256": protocol.residual_recipe_sha256,
        "protocol": {
            "timezone": protocol.timezone,
            "forecast_origin": "D-1 08:00 Europe/Paris",
            "context_length": int(settings["context_length"]),
            "first_delivery_day": str(plans[0].delivery_date),
            "last_delivery_day": str(plans[-1].delivery_date),
            "delivery_days": len(plans),
            "delivery_hours": int(sum(plan.horizon for plan in plans)),
            "horizon_day_counts": {
                str(int(horizon)): int(count)
                for horizon, count in horizons.items()
            },
            "calibration_days": int(settings["calibration_days"]),
            "sealed_evaluation_days": int(settings["evaluation_days"]),
            "primary_evaluation_window": "sealed_365_delivery_days",
            "post_publication_sensitivity": {
                "first_included_local_day": str(POST_PUBLICATION_START_DAY),
                "planned_hours": int(sensitivity_mask.sum()),
                "planned_days": int(
                    pd.Index(sensitivity_days[sensitivity_mask]).nunique()
                ),
                "reason": "Timer-S1 arXiv v3 is dated 2026-04-09",
                "role": "secondary_sensitivity_not_primary_metric",
                "caveat": (
                    "Reduces pre-publication contamination risk but does not "
                    "prove absence of training-data contamination."
                ),
            },
            "bootstrap_samples": int(settings["bootstrap_samples"]),
            "bootstrap_seed": int(settings["seed"]),
        },
        "fairness": {
            "timer_s1_native_input": "target_only",
            "chronos2_current_native_input": "target_plus_17_context_covariates_and_12_known_future_covariates",
            "native_backbone_information_parity": False,
            "pit_covariate_diagnostics": dict(
                protocol.pit_covariate_diagnostics
            ),
            "strict_native_denominator": "target_only_for_both_models",
            "same_downstream_system_track": {
                "feature_manifest_exact": True,
                "feature_count": len(protocol.feature_names),
                "feature_names": list(protocol.feature_names),
                "expected_residual_meta_features": protocol.expected_meta_features,
                "claim": "same_downstream_features_not_native_information_parity",
            },
        },
        "current_reference_metrics": _current_reference_metrics(protocol),
        "artifacts": {
            "timer_s1_oof": str(settings["timer_oof_file"]),
            "chronos2_target_only_oof": str(
                settings["chronos_target_only_oof_file"]
            ),
            "comparison_directory": str(Path(settings["output_dir"]) / "comparison"),
        },
        "runtime": {
            "timer_transformers_requirement": ">=4.57.1,<4.58",
            "timer_transformers_compatible": compatible,
            "timer_recommended_gpu_gib": TIMER_RECOMMENDED_GPU_GIB,
            "timer_cpu_guard_available_ram_gib": TIMER_MIN_AVAILABLE_RAM_GIB,
            "resource_snapshot": resources,
        },
    }
    path = Path(settings["output_dir"]) / "timer_s1_plan.json"
    _atomic_write_json(plan, path)
    return path


def _checksums(directory: Path, paths: Sequence[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(directory)).replace("\\", "/"): sha256_file(path)
        for path in sorted(paths, key=lambda value: str(value))
    }


def _post_publication_sensitivity(
    predictions: pd.DataFrame,
    *,
    timezone: str,
    pairs: Mapping[str, tuple[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Score the subset after the public Timer-S1 v3 manuscript date."""

    if not isinstance(predictions.index, pd.DatetimeIndex) or predictions.index.tz is None:
        raise ValueError("Sensibilite post-publication: index UTC requis.")
    local_days = pd.Index(predictions.index.tz_convert(timezone).date)
    mask = np.asarray(local_days >= POST_PUBLICATION_START_DAY, dtype=bool)
    subset = predictions.loc[mask]
    if subset.empty:
        raise ValueError("La fenetre d'evaluation ne couvre pas le 2026-04-10.")
    actual = pd.to_numeric(subset["actual"], errors="coerce")
    model_columns = sorted(
        column for column in subset.columns if column.endswith("__q50")
    )
    metrics = []
    n_days = int(
        pd.Index(subset.index.tz_convert(timezone).date).nunique()
    )
    for column in model_columns:
        prediction = pd.to_numeric(subset[column], errors="coerce")
        error = prediction.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
        if not np.isfinite(error).all():
            raise ValueError(f"Sensibilite post-publication non finie: {column}.")
        metrics.append(
            {
                "model": column[: -len("__q50")],
                "n_hours": len(subset),
                "n_days": n_days,
                "mae_q50": float(np.mean(np.abs(error))),
            }
        )
    paired: dict[str, Any] = {}
    for label, (baseline_model, candidate_model) in pairs.items():
        baseline_column = f"{baseline_model}__q50"
        candidate_column = f"{candidate_model}__q50"
        if baseline_column not in subset or candidate_column not in subset:
            continue
        paired[label] = paired_daily_mae_bootstrap(
            actual,
            pd.to_numeric(subset[baseline_column], errors="coerce"),
            pd.to_numeric(subset[candidate_column], errors="coerce"),
            timezone=timezone,
            samples=bootstrap_samples,
            seed=seed,
        )
    return {
        "role": "secondary_sensitivity_not_primary_metric",
        "timer_s1_public_reference": "arXiv v3 dated 2026-04-09",
        "first_included_local_day": str(POST_PUBLICATION_START_DAY),
        "last_included_local_day": str(
            subset.index[-1].tz_convert(timezone).date()
        ),
        "interpretation": (
            "This subset reduces pre-publication benchmark-contamination risk; "
            "it does not prove absence of training-data contamination."
        ),
        "metrics": metrics,
        "paired_daily_mae": paired,
    }


def publish_comparison(
    protocol: SourceProtocol,
    settings: Mapping[str, Any],
) -> Path:
    timer_path = Path(settings["timer_oof_file"])
    if not timer_path.is_file():
        raise FileNotFoundError(
            f"OOF Timer-S1 absent: {timer_path}. Lancez --generate-timer."
        )
    timer_config = dict(settings["timer"])
    timer_model_kwargs = _mapping(
        timer_config.get("model_kwargs"), name="timer_s1.model_kwargs"
    )
    if timer_model_kwargs.get("use_cache", False) not in (False, 0):
        raise ValueError("Timer-S1 comparison impose model_kwargs.use_cache=false.")
    timer_model_kwargs["use_cache"] = False
    timer_sidecar = _require_generation_sidecar(
        timer_path,
        model_name="timer_s1",
        model_id=str(timer_config.get("model_id", DEFAULT_TIMER_S1_MODEL_ID)),
        revision=str(
            timer_config.get("revision", DEFAULT_TIMER_S1_REVISION)
        ),
        context_length=int(settings["context_length"]),
        inference_contract={
            "revin": True,
            "use_cache": False,
            "torch_dtype": _timer_dtype_name(
                str(timer_config.get("torch_dtype", "bfloat16"))
            ),
            "model_kwargs": timer_model_kwargs,
            "quantile_indices": {"q10": 0, "q50": 4, "q90": 8},
        },
        protocol=protocol,
    )
    timer_oof = read_quantile_artifact(timer_path)
    validate_timer_oof(timer_oof, protocol)
    result = compare_same_downstream_features(
        protocol,
        timer_oof,
        timer_provenance=timer_sidecar,
        threads=int(settings["threads"]),
        bootstrap_samples=int(settings["bootstrap_samples"]),
        seed=int(settings["seed"]),
    )

    directory = Path(settings["output_dir"]) / "comparison"
    directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    frame_artifacts = {
        "system_predictions.csv.gz": result.predictions,
        "system_metrics.csv": result.metrics,
    }
    for name, frame in frame_artifacts.items():
        path = directory / name
        _atomic_write_frame(frame, path)
        artifacts.append(path)

    json_artifacts: dict[str, Any] = {
        "system_metrics.json": result.metrics.to_dict(orient="records"),
        "paired_tests.json": result.paired_tests,
        "feature_parity.json": result.feature_parity,
        "corrector_diagnostics.json": result.corrector_diagnostics,
        "post_publication_sensitivity_system.json": (
            _post_publication_sensitivity(
                result.predictions,
                timezone=protocol.timezone,
                pairs={
                    "native_asymmetric_timer_minus_chronos": (
                        "chronos2_native",
                        "timer_s1_native",
                    ),
                    "same_corrector_asymmetric_timer_minus_chronos": (
                        "chronos2_refit_same_corrector",
                        "timer_s1_refit_same_corrector",
                    ),
                    "challenger_timer_minus_frozen_current": (
                        "chronos2_frozen_current_corrected",
                        "timer_s1_refit_same_corrector",
                    ),
                },
                bootstrap_samples=int(settings["bootstrap_samples"]),
                seed=int(settings["seed"]),
            )
        ),
    }

    chronos_path = Path(settings["chronos_target_only_oof_file"])
    strict_status: dict[str, Any]
    if chronos_path.is_file():
        chronos_config = dict(settings["chronos"])
        chronos_revision = chronos_config.get("revision")
        if chronos_revision in (None, ""):
            raise ValueError(
                "chronos2_target_only.revision doit etre un commit immuable."
            )
        chronos_sidecar = _require_generation_sidecar(
            chronos_path,
            model_name="chronos2_target_only",
            model_id=str(chronos_config.get("model_id", "amazon/chronos-2")),
            revision=str(chronos_revision),
            context_length=int(settings["context_length"]),
            inference_contract={
                "cross_learning": False,
                "quantile_levels": [0.1, 0.5, 0.9],
            },
            protocol=protocol,
        )
        chronos_oof = read_quantile_artifact(chronos_path)
        strict_metrics, strict_paired = compare_target_only_native(
            protocol,
            chronos_target_only_oof=chronos_oof,
            timer_oof=timer_oof,
            chronos_provenance=chronos_sidecar,
            timer_provenance=timer_sidecar,
            bootstrap_samples=int(settings["bootstrap_samples"]),
            seed=int(settings["seed"]),
        )
        strict_metrics_path = directory / "strict_target_only_metrics.csv"
        _atomic_write_frame(strict_metrics, strict_metrics_path)
        artifacts.append(strict_metrics_path)
        json_artifacts["strict_target_only_metrics.json"] = strict_metrics.to_dict(
            orient="records"
        )
        json_artifacts["strict_target_only_paired_test.json"] = strict_paired
        strict_common = compare_target_only_same_downstream_features(
            protocol,
            chronos_target_only_oof=chronos_oof,
            timer_oof=timer_oof,
            chronos_provenance=chronos_sidecar,
            timer_provenance=timer_sidecar,
            threads=int(settings["threads"]),
            bootstrap_samples=int(settings["bootstrap_samples"]),
            seed=int(settings["seed"]),
        )
        strict_common_predictions_path = (
            directory / "strict_common_predictions.csv.gz"
        )
        strict_common_metrics_path = directory / "strict_common_metrics.csv"
        _atomic_write_frame(
            strict_common.predictions, strict_common_predictions_path
        )
        _atomic_write_frame(strict_common.metrics, strict_common_metrics_path)
        artifacts.extend(
            [strict_common_predictions_path, strict_common_metrics_path]
        )
        json_artifacts.update(
            {
                "strict_common_metrics.json": strict_common.metrics.to_dict(
                    orient="records"
                ),
                "strict_common_paired_tests.json": strict_common.paired_tests,
                "strict_common_feature_parity.json": strict_common.feature_parity,
                "strict_common_corrector_diagnostics.json": (
                    strict_common.corrector_diagnostics
                ),
                "post_publication_sensitivity_strict_common.json": (
                    _post_publication_sensitivity(
                        strict_common.predictions,
                        timezone=protocol.timezone,
                        pairs={
                            "strict_native_timer_minus_chronos": (
                                "chronos2_target_only_native",
                                "timer_s1_target_only_native",
                            ),
                            "strict_common_corrector_timer_minus_chronos": (
                                "chronos2_target_only_same_corrector",
                                "timer_s1_target_only_same_corrector",
                            ),
                        },
                        bootstrap_samples=int(settings["bootstrap_samples"]),
                        seed=int(settings["seed"]),
                    )
                ),
            }
        )
        strict_status = {
            "status": "complete",
            "native_information_parity": True,
            "strict_common_downstream_status": "complete",
            "chronos_target_only_oof": str(chronos_path),
            "chronos_target_only_sidecar_sha256": sha256_file(
                _sidecar_path(chronos_path)
            ),
        }
    else:
        strict_status = {
            "status": "missing_chronos2_target_only_oof",
            "native_information_parity": None,
            "required_action": "--generate-chronos-target-only",
            "expected_path": str(chronos_path),
        }
    json_artifacts["strict_target_only_status.json"] = strict_status

    for name, value in json_artifacts.items():
        path = directory / name
        _atomic_write_json(value, path)
        artifacts.append(path)

    manifest = {
        "script": "run_timer_s1_comparison.py",
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "source_run": str(protocol.source_run),
        "source_feature_manifest_sha256": protocol.feature_manifest_sha256,
        "source_target_sha256": sha256_target_series(protocol.target),
        "source_residual_recipe_sha256": protocol.residual_recipe_sha256,
        "pit_covariate_diagnostics": dict(protocol.pit_covariate_diagnostics),
        "timer_oof": str(timer_path),
        "timer_oof_sha256": sha256_file(timer_path),
        "timer_oof_sidecar_sha256": sha256_file(_sidecar_path(timer_path)),
        "timer_provenance": timer_sidecar,
        "chronos2_target_only_oof": (
            str(chronos_path) if chronos_path.is_file() else None
        ),
        "strict_target_only_status": strict_status,
        "system_comparison_claim": "same_downstream_features_not_native_information_parity",
        "strict_common_comparison_claim": (
            "target_only_backbones_plus_identical_downstream_features"
            if chronos_path.is_file()
            else None
        ),
        "evaluation_start_utc": str(protocol.evaluation_index[0]),
        "evaluation_end_utc": str(protocol.evaluation_index[-1]),
        "evaluation_hours": len(protocol.evaluation_index),
        "evaluation_days": int(settings["evaluation_days"]),
        "bootstrap_samples": int(settings["bootstrap_samples"]),
        "seed": int(settings["seed"]),
        "artifacts": [path.name for path in artifacts],
    }
    manifest_path = directory / "run_manifest.json"
    _atomic_write_json(manifest, manifest_path)
    artifacts.append(manifest_path)
    checksum_path = directory / "artifact_checksums.json"
    _atomic_write_json(_checksums(directory, artifacts), checksum_path)
    LOGGER.info("Comparaison publiee: %s", directory)
    return directory


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config_path = Path(args.config).expanduser().resolve()
    config = _load_yaml(config_path)
    settings = _resolved_settings(config_path, config, args)
    protocol = _load_protocol(settings)
    plans = _plans_for_protocol(protocol)

    if args.plan_only:
        path = write_plan(protocol, plans, settings)
        LOGGER.info("Plan valide sans chargement de modele: %s", path)
    elif args.generate_timer:
        generate_timer(protocol, plans, settings, args)
    elif args.generate_chronos_target_only:
        generate_chronos_target_only(protocol, plans, settings, args)
    elif args.compare_only:
        publish_comparison(protocol, settings)
    else:  # pragma: no cover - argparse enforces an action
        raise AssertionError("Action CLI absente.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrompu; les checkpoints atomiques deja publies sont conserves.")
        raise SystemExit(130) from None
