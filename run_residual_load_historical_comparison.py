#!/usr/bin/env python
"""Recalculate the complete Saturn-vs-Chronos residual-load price benchmark.

This is deliberately separate from the normal ``Forecast.ps1 -Action Run``
path.  It rebuilds both historical branches from the same sealed target and
feature contract, then publishes ordinary hourly HTML reports from paired
FINAL365 predictions.  The only permitted branch difference is the ten
residual-load treatment columns (five raw values and five oracle mirrors).

The expensive upstream and price-model steps are checkpointed.  No historical
Statistics file from production is reused.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import date, timedelta
import gzip
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.chronos_adapter import (
    execute_grouped_chronos_backtest,
    generate_delivery_plans,
    load_chronos_oof,
    make_existing_forecasting_executor,
    normalize_chronos_future,
    run_existing_live_forecast,
)
from chronos2_hourly.chronos_residual_load import (
    COUNTRY_ALIASES,
    COUNTRY_OBSERVED_SERIES,
    EXPECTED_ALIASES,
    MODEL_ID,
    MODEL_REVISION,
    RESIDUAL_LOAD_TREATMENT_COLUMNS,
)
from chronos2_hourly.historical_blend_replay import (
    publish_historical_blend_replay,
)
from chronos2_hourly.historical_price_replay import (
    generate_checkpointed_price_replay,
    replay_identity,
    select_replay_days,
)
from chronos2_hourly.historical_residual_comparison import (
    publish_historical_residual_comparison,
)
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_modular.common import (
    build_zone_configs,
    deep_get,
    load_yaml,
    set_reproducibility,
)
from chronos2_modular.data import prepare_zone_data
from chronos2_modular.forecasting import load_model
from chronos2_modular.saturn import create_saturn_client, sync_vintage_series


LOGGER = logging.getLogger("residual_load_historical_comparison")
SCRIPT_VERSION = "1.0.0-full-symmetric-recalculation"
PRICE_EXECUTION_PROTOCOL_VERSION = (
    "hourly_oof_residual_source_comparison.v1"
)
DEFAULT_CONFIG = "config/residual_load_historical_comparison.yaml"
BRANCHES = ("saturn", "chronos2")
SEALED_WINDOWS = {
    "extended": (date(2024, 1, 2), date(2024, 8, 11), 223),
    "oof": (date(2024, 8, 12), date(2026, 8, 11), 730),
    "final": (date(2025, 8, 12), date(2026, 8, 11), 365),
}
STAGES = (
    "plan",
    "sync-observed",
    "residual-replay",
    "price-replay",
    "downstream",
    "blend",
    "report",
    "all",
)


class HistoricalComparisonRunnerError(RuntimeError):
    """Raised when an experiment stage would break paired comparability."""


@dataclass(frozen=True)
class ZoneSpec:
    code: str
    timezone: str
    base_config: Path
    sealed_extended_run: Path
    sealed_blend_run: Path | None


@dataclass(frozen=True)
class Protocol:
    project_root: Path
    config_path: Path
    config: Mapping[str, Any]
    experiment_root: Path
    export_root: Path
    timezone: str
    cutoff_clock: str
    extended_start: date
    extended_end: date
    oof_start: date
    oof_end: date
    final_start: date
    final_end: date
    model_id: str
    model_revision: str
    context_length: int
    maximum_gap_hours: int
    origin_batch_size: int
    model_batch_size: int
    residual_batch_size: int
    observed_series: Mapping[str, str]
    zones: Mapping[str, ZoneSpec]

    @property
    def residual_end(self) -> date:
        # The base runner also needs a recalculated live day.
        return self.oof_end + timedelta(days=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalcule integralement les deux branches residual_load, leurs "
            "OOF prix et les rapports HTML standards de comparaison."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, default="plan")
    parser.add_argument(
        "--zones",
        nargs="+",
        default=["FR", "DE", "BE", "NL", "ES"],
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--chunk-days", type=int, default=14)
    parser.add_argument("--history-chunk-days", type=int, default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-observed-sync",
        action="store_true",
        help="Utilise uniquement les stores observes deja presents.",
    )
    parser.add_argument(
        "--export-delivery-day",
        default=None,
        help=(
            "Dossier journalier d'export. Par defaut, utilise le jour live "
            "du benchmark (2026-08-12 pour le protocole actuel)."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} doit etre un mapping.")
    return value


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _parse_day(value: Any, *, name: str) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is not None:
        raise ValueError(f"{name} doit etre une date locale naive.")
    if timestamp != timestamp.normalize():
        raise ValueError(f"{name} doit etre au format YYYY-MM-DD.")
    return timestamp.date()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(path: Path) -> str:
    """Hash logical bytes, ignoring the gzip container timestamp."""

    digest = hashlib.sha256()
    opener = gzip.open if path.name.lower().endswith(".gz") else Path.open
    if opener is gzip.open:
        handle_context = gzip.open(path, "rb")
    else:
        handle_context = path.open("rb")
    with handle_context as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _implementation_hashes(
    protocol: Protocol,
    scope: str,
) -> dict[str, str]:
    paths_by_scope = {
        "upstream": (
            "chronos2_hourly/chronos_residual_load.py",
            "chronos2_hourly/historical_residual_load.py",
        ),
        "price": (
            "chronos2_modular/forecasting.py",
            "chronos2_hourly/chronos_adapter.py",
            "chronos2_hourly/historical_price_replay.py",
        ),
        "downstream": (
            "run_residual_load_historical_comparison.py",
            "run_chronos2_hourly.py",
            "run_extended_residual_hourly.py",
            "chronos2_hourly/historical_blend_replay.py",
            "chronos2_hourly/historical_residual_comparison.py",
            "chronos2_hourly/reporting.py",
            "chronos2_hourly/models/residual_corrector.py",
            "chronos2_hourly/models/blended_residual_corrector.py",
        ),
    }
    if scope not in paths_by_scope:
        raise ValueError(f"Scope d'implementation inconnu: {scope}.")
    result: dict[str, str] = {}
    for relative in paths_by_scope[scope]:
        path = protocol.project_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = _sha256(path)
    return result


def _execution_signature(
    protocol: Protocol,
    requested_device: str,
    *,
    scope: str,
) -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device = (
            str(torch.cuda.get_device_name(0)) if cuda_available else None
        )
        torch_version = str(torch.__version__)
    except (ImportError, RuntimeError):
        cuda_available = False
        cuda_device = None
        torch_version = "not-installed"
    selected_device = (
        "cuda"
        if requested_device == "cuda"
        or (requested_device == "auto" and cuda_available)
        else "cpu"
    )
    if requested_device == "cuda" and not cuda_available:
        raise HistoricalComparisonRunnerError(
            "--device cuda demande mais CUDA n'est pas disponible."
        )
    signature = {
        "requested_device": requested_device,
        "selected_device": selected_device,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "python": platform.python_version(),
        "torch": torch_version,
        "transformers": _distribution_version("transformers"),
        "chronos_forecasting": _distribution_version("chronos-forecasting"),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "implementation_sha256": _implementation_hashes(protocol, scope),
    }
    if scope == "price":
        signature["executor_protocol_version"] = (
            PRICE_EXECUTION_PROTOCOL_VERSION
        )
        signature["executor_variant"] = (
            "hourly_oof_residual_source_comparison"
        )
        signature["with_covariates"] = True
    return signature


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path, date)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
    )


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        yaml.safe_dump(
            _json_safe(value),
            allow_unicode=True,
            sort_keys=False,
        ),
    )


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        pd.read_parquet(temporary).head(1)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(compressed)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        pd.read_csv(temporary, nrows=1)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_protocol(config_path: Path, project_root: Path) -> Protocol:
    config = load_yaml(config_path)
    raw_protocol = _mapping(config.get("protocol"), name="protocol")
    raw_model = _mapping(config.get("model"), name="model")
    raw_outputs = _mapping(config.get("outputs"), name="outputs")
    raw_zones = _mapping(config.get("zones"), name="zones")
    timezone = str(raw_protocol.get("timezone", "Europe/Paris"))
    cutoff_clock = str(raw_protocol.get("forecast_origin_local_time", "08:00"))
    if cutoff_clock != "08:00":
        raise ValueError("Le protocole compare exige le cutoff civil D-1 08:00.")
    extended_start = _parse_day(
        raw_protocol["extended_start_day"], name="extended_start_day"
    )
    extended_end = _parse_day(
        raw_protocol["extended_end_day"], name="extended_end_day"
    )
    oof_start = _parse_day(raw_protocol["oof_start_day"], name="oof_start_day")
    oof_end = _parse_day(raw_protocol["oof_end_day"], name="oof_end_day")
    final_start = _parse_day(
        raw_protocol["final_start_day"], name="final_start_day"
    )
    final_end = _parse_day(raw_protocol["final_end_day"], name="final_end_day")
    counts = {
        "expected_extended_days": (extended_end - extended_start).days + 1,
        "expected_oof_days": (oof_end - oof_start).days + 1,
        "expected_final_days": (final_end - final_start).days + 1,
    }
    for key, actual in counts.items():
        if int(raw_protocol[key]) != actual:
            raise ValueError(f"{key}: {raw_protocol[key]} != {actual}.")
    observed_windows = {
        "extended": (extended_start, extended_end, counts["expected_extended_days"]),
        "oof": (oof_start, oof_end, counts["expected_oof_days"]),
        "final": (final_start, final_end, counts["expected_final_days"]),
    }
    if observed_windows != SEALED_WINDOWS:
        raise ValueError(
            f"Le benchmark exige les fenetres scellees {SEALED_WINDOWS}, "
            f"recu {observed_windows}."
        )
    if raw_protocol.get("preserve_saturn_availability_mask") is not True:
        raise ValueError("preserve_saturn_availability_mask doit rester true.")
    if not (extended_end + timedelta(days=1) == oof_start):
        raise ValueError("EXT223 et OOF730 ne sont pas contigus.")
    if not (oof_start <= final_start <= final_end == oof_end):
        raise ValueError("FINAL365 doit etre la fin exacte de OOF730.")

    zones: dict[str, ZoneSpec] = {}
    for raw_code, raw_value in raw_zones.items():
        code = str(raw_code).upper()
        settings = _mapping(raw_value, name=f"zones.{code}")
        base_config = _resolve(settings["base_config"], base=project_root)
        sealed_run = _resolve(settings["sealed_extended_run"], base=project_root)
        blend_value = settings.get("sealed_blend_run")
        blend_run = (
            _resolve(blend_value, base=project_root)
            if blend_value not in (None, "")
            else None
        )
        if not base_config.is_file() or not sealed_run.is_dir():
            raise FileNotFoundError(
                f"{code}: config/run scelle absent ({base_config}, {sealed_run})."
            )
        if blend_run is not None and not blend_run.is_dir():
            raise FileNotFoundError(blend_run)
        zones[code] = ZoneSpec(
            code=code,
            timezone=str(settings["timezone"]),
            base_config=base_config,
            sealed_extended_run=sealed_run,
            sealed_blend_run=blend_run,
        )
    observed = {
        str(alias): str(series)
        for alias, series in _mapping(
            config.get("observed_series"), name="observed_series"
        ).items()
    }
    expected_observed = {
        COUNTRY_ALIASES[country]: series
        for country, series in COUNTRY_OBSERVED_SERIES.items()
    }
    if observed != expected_observed:
        raise ValueError(
            "observed_series doit etre exactement le mapping scelle des cinq "
            f"series .obs: {expected_observed}."
        )

    model_id = str(raw_model.get("model_id", MODEL_ID))
    model_revision = str(raw_model.get("revision", MODEL_REVISION))
    context_length = int(raw_model.get("context_length", 2048))
    maximum_gap_hours = int(raw_model.get("maximum_internal_gap_hours", 6))
    if (
        timezone != "Europe/Paris"
        or model_id != MODEL_ID
        or model_revision != MODEL_REVISION
        or context_length != 2048
        or maximum_gap_hours != 6
    ):
        raise ValueError(
            "Le protocole scelle exige Europe/Paris, amazon/chronos-2, "
            f"revision={MODEL_REVISION}, contexte=2048 et gap=6h."
        )

    return Protocol(
        project_root=project_root,
        config_path=config_path,
        config=config,
        experiment_root=_resolve(raw_outputs["experiment_root"], base=project_root),
        export_root=_resolve(raw_outputs["export_root"], base=project_root),
        timezone=timezone,
        cutoff_clock=cutoff_clock,
        extended_start=extended_start,
        extended_end=extended_end,
        oof_start=oof_start,
        oof_end=oof_end,
        final_start=final_start,
        final_end=final_end,
        model_id=model_id,
        model_revision=model_revision,
        context_length=context_length,
        maximum_gap_hours=maximum_gap_hours,
        origin_batch_size=int(raw_model.get("origin_batch_size", 12)),
        model_batch_size=int(raw_model.get("model_batch_size", 128)),
        residual_batch_size=int(raw_model.get("residual_batch_size", 64)),
        observed_series=observed,
        zones=zones,
    )


def _selected_zones(protocol: Protocol, requested: Sequence[str]) -> tuple[ZoneSpec, ...]:
    codes = tuple(dict.fromkeys(str(value).upper() for value in requested))
    unknown = [code for code in codes if code not in protocol.zones]
    if unknown:
        raise ValueError(f"Zones inconnues: {unknown}.")
    return tuple(protocol.zones[code] for code in codes)


def _cutoff_for_day(day: date, timezone: str = "Europe/Paris") -> pd.Timestamp:
    previous = day - timedelta(days=1)
    return (
        (pd.Timestamp(previous) + pd.Timedelta(hours=8))
        .tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    ).tz_convert("UTC")


def _protocol_manifest(protocol: Protocol, zones: Sequence[ZoneSpec]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment_id": str(protocol.config.get("experiment_id")),
        "config": str(protocol.config_path),
        "config_sha256": _sha256(protocol.config_path),
        "zones": [zone.code for zone in zones],
        "branches": list(BRANCHES),
        "treatment_columns": list(RESIDUAL_LOAD_TREATMENT_COLUMNS),
        "upstream": {
            "control": "sealed Saturn residual-load values",
            "challenger": "Chronos-2 historical rolling-origin PIT replay",
            "observed_series": dict(protocol.observed_series),
            "cutoff": "D-1 08:00 Europe/Paris",
            "model_id": protocol.model_id,
            "model_revision": protocol.model_revision,
            "context_length": protocol.context_length,
            "maximum_internal_gap_hours": protocol.maximum_gap_hours,
            "foundation_model_pretraining_note": (
                "rolling-origin PIT replay; upstream foundation-model "
                "pretraining cutoff is not asserted as OOF"
            ),
        },
        "windows": {
            "extended": [protocol.extended_start, protocol.extended_end],
            "oof": [protocol.oof_start, protocol.oof_end],
            "sealed_final": [protocol.final_start, protocol.final_end],
            "residual_replay_end": protocol.residual_end,
        },
        "downstream_policy": (
            "full symmetric recalculation: price Chronos, LEAR, CatBoost, "
            "ensemble, EXT223 residual corrector and frozen MKOnline blend"
        ),
        "statistics_history_reused": False,
    }


def _source_model_frame(zone: ZoneSpec) -> pd.DataFrame:
    path = zone.sealed_extended_run / "inputs" / "model_covariates_with_future.csv.gz"
    frame = pd.read_csv(path)
    if "timestamp" not in frame:
        raise HistoricalComparisonRunnerError(f"{path}: timestamp absent.")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("timestamp"), utc=True, errors="raise"),
        name="timestamp",
    )
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise HistoricalComparisonRunnerError(f"{path}: timeline invalide.")
    missing = sorted(set(EXPECTED_ALIASES) - set(frame.columns))
    if missing:
        raise HistoricalComparisonRunnerError(f"{path}: aliases absents {missing}.")
    for alias in EXPECTED_ALIASES:
        mirror = f"known_{alias}_oracle"
        if mirror not in frame:
            raise HistoricalComparisonRunnerError(f"{path}: {mirror} absent.")
        _assert_equal_numeric(
            pd.to_numeric(frame[alias], errors="coerce"),
            pd.to_numeric(frame[mirror], errors="coerce"),
            name=f"{path.name}/{alias}/oracle_mirror",
        )
    return frame


def _source_target_file(zone: ZoneSpec) -> Path:
    path = zone.sealed_extended_run / "inputs" / "aligned_inputs.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _pit_rows(values: pd.Series, *, timezone: str) -> pd.DataFrame:
    finite = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    delivery = finite.index.tz_convert("UTC")
    local_days = pd.DatetimeIndex(delivery.tz_convert(timezone).date)
    cutoffs = (
        local_days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    ).tz_localize(timezone, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    return pd.DataFrame(
        {
            "value_time_utc": delivery,
            "snapshot_time_utc": cutoffs,
            "revision_time_utc": cutoffs,
            "value": finite.to_numpy(dtype=float),
        }
    )


def _branch_pit_dir(protocol: Protocol, branch: str, zone: str) -> Path:
    return protocol.experiment_root / "pit" / branch / zone.lower()


def _validate_branch_pit(
    protocol: Protocol,
    zone: ZoneSpec,
    branch: str,
) -> dict[str, Any]:
    root = _branch_pit_dir(protocol, branch, zone.code)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise HistoricalComparisonRunnerError(f"{manifest_path}: JSON invalide.")
    expected_source = (
        "saturn" if branch == "saturn" else "chronos2_historical_replay"
    )
    source_frame = (
        zone.sealed_extended_run
        / "inputs"
        / "model_covariates_with_future.csv.gz"
    )
    if (
        manifest.get("zone") != zone.code
        or manifest.get("branch") != branch
        or manifest.get("residual_load_source") != expected_source
        or manifest.get("source_model_covariates_sha256") != _sha256(source_frame)
    ):
        raise HistoricalComparisonRunnerError(
            f"{zone.code}/{branch}: manifeste PIT obsolete ou incompatible."
        )
    pit_files = _mapping(manifest.get("pit_files"), name="pit_files")
    for alias in EXPECTED_ALIASES:
        entry = _mapping(pit_files.get(alias), name=f"pit_files.{alias}")
        path = root / f"{alias}.parquet"
        if (
            Path(str(entry.get("path", ""))).resolve() != path.resolve()
            or entry.get("sha256") != _sha256(path)
        ):
            raise HistoricalComparisonRunnerError(
                f"{zone.code}/{branch}/{alias}: PIT non scelle."
            )
    if branch == "chronos2":
        upstream = _mapping(manifest.get("upstream_replay"), name="upstream_replay")
        replay_manifest_path = Path(str(upstream.get("manifest_path", "")))
        predictions_path = Path(str(upstream.get("predictions_path", "")))
        if (
            not replay_manifest_path.is_file()
            or upstream.get("manifest_sha256") != _sha256(replay_manifest_path)
            or not predictions_path.is_file()
            or upstream.get("predictions_sha256") != _sha256(predictions_path)
            or upstream.get("model_id") != protocol.model_id
            or upstream.get("model_revision") != protocol.model_revision
            or upstream.get("start_day_local") != str(protocol.extended_start)
            or upstream.get("end_day_local") != str(protocol.residual_end)
        ):
            raise HistoricalComparisonRunnerError(
                f"{zone.code}: provenance du replay residual_load invalide."
            )
    return manifest


def _materialize_branch_pit(
    protocol: Protocol,
    zones: Sequence[ZoneSpec],
    replay_wide: pd.DataFrame,
    *,
    replay_manifest_path: Path,
) -> None:
    if not replay_manifest_path.is_file():
        raise FileNotFoundError(replay_manifest_path)
    replay_manifest_sha256 = _sha256(replay_manifest_path)
    replay_manifest = json.loads(
        replay_manifest_path.read_text(encoding="utf-8")
    )
    predictions = _mapping(
        _mapping(replay_manifest.get("artifacts"), name="replay.artifacts").get(
            "predictions"
        ),
        name="replay.artifacts.predictions",
    )
    for zone in zones:
        source = _source_model_frame(zone)
        index = source.index.intersection(replay_wide.index)
        if index.empty:
            raise HistoricalComparisonRunnerError(
                f"{zone.code}: aucun timestamp commun avec le replay upstream."
            )
        zone_audit: dict[str, Any] = {
            "zone": zone.code,
            "source_model_covariates": str(
                zone.sealed_extended_run
                / "inputs"
                / "model_covariates_with_future.csv.gz"
            ),
            "source_model_covariates_sha256": _sha256(
                zone.sealed_extended_run
                / "inputs"
                / "model_covariates_with_future.csv.gz"
            ),
            "aliases": {},
        }
        for alias in EXPECTED_ALIASES:
            control_values = pd.to_numeric(source[alias], errors="coerce")
            replay_values = pd.to_numeric(
                replay_wide[alias].reindex(source.index), errors="coerce"
            )
            challenger_values = replay_values.where(control_values.notna())
            if not control_values.notna().equals(challenger_values.notna()):
                raise HistoricalComparisonRunnerError(
                    f"{zone.code}/{alias}: masque Saturn non preserve."
                )
            control_frame = _pit_rows(control_values, timezone=zone.timezone)
            challenger_frame = _pit_rows(
                challenger_values, timezone=zone.timezone
            )
            control_path = _branch_pit_dir(protocol, "saturn", zone.code) / f"{alias}.parquet"
            challenger_path = _branch_pit_dir(protocol, "chronos2", zone.code) / f"{alias}.parquet"
            _atomic_parquet(control_frame, control_path)
            _atomic_parquet(challenger_frame, challenger_path)
            zone_audit["aliases"][alias] = {
                "available_hours": int(control_values.notna().sum()),
                "control_sha256": _sha256(control_path),
                "challenger_sha256": _sha256(challenger_path),
                "availability_mask_identical": True,
            }
        for branch in BRANCHES:
            root = _branch_pit_dir(protocol, branch, zone.code)
            manifest = {
                **zone_audit,
                "branch": branch,
                "residual_load_source": (
                    "saturn" if branch == "saturn" else "chronos2_historical_replay"
                ),
                "pit_files": {
                    alias: {
                        "path": str(root / f"{alias}.parquet"),
                        "sha256": _sha256(root / f"{alias}.parquet"),
                    }
                    for alias in EXPECTED_ALIASES
                },
            }
            if branch == "chronos2":
                manifest["upstream_replay"] = {
                    "manifest_path": str(replay_manifest_path),
                    "manifest_sha256": replay_manifest_sha256,
                    "predictions_path": str(predictions["path"]),
                    "predictions_sha256": str(predictions["sha256"]),
                    "model_id": replay_manifest.get("model_id"),
                    "model_revision": replay_manifest.get("model_revision"),
                    "start_day_local": replay_manifest.get("start_day_local"),
                    "end_day_local": replay_manifest.get("end_day_local"),
                    "observed_sources": replay_manifest.get("observed_sources"),
                }
            _write_json(root / "manifest.json", manifest)


def _generated_base_config(
    protocol: Protocol,
    zone: ZoneSpec,
    branch: str,
    *,
    local_files_only: bool,
) -> tuple[dict[str, Any], Path]:
    config = copy.deepcopy(load_yaml(zone.base_config))
    model = config.setdefault("model", {})
    if not isinstance(model, dict):
        raise TypeError(f"{zone.base_config}: model doit etre un mapping.")
    model.update(
        {
            "model_id": protocol.model_id,
            "revision": protocol.model_revision,
            "context_length": protocol.context_length,
            "origin_batch_size": protocol.origin_batch_size,
            "model_batch_size": protocol.model_batch_size,
            "local_files_only": bool(local_files_only),
        }
    )
    data = config.setdefault("data", {})
    if not isinstance(data, dict):
        raise TypeError(f"{zone.base_config}: data doit etre un mapping.")
    data["project_root"] = str(protocol.project_root)
    data["pit_vintage_dir"] = str(_branch_pit_dir(protocol, branch, zone.code))
    data["pit_files"] = {alias: f"{alias}.parquet" for alias in EXPECTED_ALIASES}
    data["runtime_as_of"] = _cutoff_for_day(
        protocol.residual_end, zone.timezone
    ).tz_convert(zone.timezone).isoformat()
    zones = config.get("zones")
    if not isinstance(zones, dict) or zone.code not in zones:
        raise KeyError(f"{zone.base_config}: zones.{zone.code} absent.")
    target = zones[zone.code].setdefault("target", {})
    if not isinstance(target, dict):
        raise TypeError(f"{zone.base_config}: target doit etre un mapping.")
    target.update(
        {
            "file": str(_source_target_file(zone)),
            "timestamp_col": "timestamp",
            "value_col": "target",
            "source": "file",
        }
    )
    report = config.setdefault("report", {})
    if not isinstance(report, dict):
        raise TypeError(f"{zone.base_config}: report doit etre un mapping.")
    report["enabled"] = False
    output = config.setdefault("output", {})
    if not isinstance(output, dict):
        raise TypeError(f"{zone.base_config}: output doit etre un mapping.")
    output["directory"] = str(
        protocol.experiment_root / "runs" / branch / "base" / zone.code.lower()
    )
    config["historical_residual_comparison"] = {
        "experiment_id": str(protocol.config.get("experiment_id")),
        "branch": branch,
        "residual_load_source": (
            "saturn" if branch == "saturn" else "chronos2_historical_replay"
        ),
        "model_revision": protocol.model_revision,
        "pit_manifest": str(
            _branch_pit_dir(protocol, branch, zone.code) / "manifest.json"
        ),
        "statistics_history_reused": False,
    }
    path = (
        protocol.experiment_root
        / "configs"
        / f"{branch}_{zone.code.lower()}_base.yaml"
    )
    _write_yaml(path, config)
    return config, path


def _generated_extended_config(
    protocol: Protocol,
    zone: ZoneSpec,
    branch: str,
    *,
    threads: int,
) -> Path:
    base_run = protocol.experiment_root / "runs" / branch / "base" / zone.code.lower()
    price_dir = protocol.experiment_root / "price_replay" / branch / zone.code.lower()
    output = protocol.experiment_root / "runs" / branch / "extended" / zone.code.lower()
    base_config = load_yaml(zone.base_config)
    zone_config = copy.deepcopy(_mapping(base_config.get("zones"), name="zones")[zone.code])
    config = {
        "extended_residual": {
            "zone": zone.code,
            "timezone": zone.timezone,
            "source_run": str(base_run),
            "extended_oof_file": str(price_dir / "chronos_price_ext223.csv.gz"),
            "chronos_oof_file": str(price_dir / "chronos_price_oof730.csv.gz"),
            "chronos_live_file": str(price_dir / "chronos_price_live.csv"),
            "schema": "chronos_only",
            "recipe": "blend_cat_hgb_w0.50",
            "threads": threads,
        },
        "zones": {zone.code: zone_config},
        "output": {"directory": str(output)},
        "report": {
            "filename": f"historical_residual_load_{branch}_{zone.code.lower()}_autonomous.html",
            "title": (
                f"{zone.code} - recalcul historique residual_load - {branch}"
            ),
            "forecast_history_hours": 168,
            "extreme_threshold": 150.0,
        },
        "historical_residual_comparison": {
            "experiment_id": str(protocol.config.get("experiment_id")),
            "branch": branch,
            "residual_load_source": (
                "saturn" if branch == "saturn" else "chronos2_historical_replay"
            ),
            "model_id": protocol.model_id,
            "model_revision": protocol.model_revision,
            "pit_manifest": str(
                _branch_pit_dir(protocol, branch, zone.code) / "manifest.json"
            ),
            "price_replay_manifest": str(
                price_dir / "chronos_price_all953.csv.gz.manifest.json"
            ),
            "statistics_history_reused": False,
        },
    }
    path = (
        protocol.experiment_root
        / "configs"
        / f"{branch}_{zone.code.lower()}_extended.yaml"
    )
    _write_yaml(path, config)
    return path


def _observed_store_path(protocol: Protocol, alias: str) -> Path:
    return protocol.experiment_root / "observed_vintages" / f"{alias}.parquet"


def _sync_observed_stage(protocol: Protocol) -> None:
    saturn = _mapping(protocol.config.get("saturn"), name="saturn")
    client = create_saturn_client(
        str(saturn["url"]),
        os.getenv("SATURN_AUTHOR") or str(saturn["author"]),
    )
    margin_days = int(saturn.get("observed_history_margin_days", 14))
    first_cutoff = _cutoff_for_day(protocol.extended_start, protocol.timezone)
    last_cutoff = _cutoff_for_day(protocol.residual_end, protocol.timezone)
    value_start = first_cutoff - pd.Timedelta(
        hours=protocol.context_length + 24 * margin_days
    )
    revision_start = value_start - pd.Timedelta(days=2)
    history_chunk_days = int(saturn.get("history_chunk_days", 60))
    results: list[dict[str, Any]] = []
    for alias, series_name in protocol.observed_series.items():
        LOGGER.info("Synchronisation historique observee %s...", alias)
        result = sync_vintage_series(
            client,
            zone="ALL",
            alias=alias,
            series_name=series_name,
            path=_observed_store_path(protocol, alias),
            revision_start=revision_start,
            revision_end=last_cutoff,
            value_start=value_start,
            value_end=last_cutoff,
            timezone="UTC",
            naive_timezone="UTC",
            incomplete_dst_policy="raise",
            overlap_days=2,
            chunk_days=history_chunk_days,
            retries=int(saturn.get("retries", 3)),
            full=False,
        )
        results.append(
            {
                **dict(result.__dict__),
                "sha256": _sha256(Path(result.path)),
            }
        )
    _write_json(
        protocol.experiment_root / "observed_vintages" / "sync_manifest.json",
        {
            "status": "complete",
            "revision_start_utc": revision_start,
            "revision_end_utc": last_cutoff,
            "value_start_utc": value_start,
            "value_end_utc": last_cutoff,
            "series": results,
        },
    )


def _load_observed_stores(protocol: Protocol) -> dict[str, pd.DataFrame]:
    stores: dict[str, pd.DataFrame] = {}
    for alias, series_name in protocol.observed_series.items():
        path = _observed_store_path(protocol, alias)
        if not path.is_file():
            raise FileNotFoundError(
                f"Store observe absent pour {alias}: {path}. "
                "Lancez d'abord --stage sync-observed."
            )
        stores[series_name] = pd.read_parquet(path)
    return stores


def _residual_replay_stage(
    protocol: Protocol,
    zones: Sequence[ZoneSpec],
    *,
    device: str,
    local_files_only: bool,
    chunk_days: int,
    resume: bool,
) -> None:
    from chronos2_hourly.historical_residual_load import (
        replay_historical_residual_load,
        replay_to_wide,
        select_observed_context_asof,
    )

    stores = _load_observed_stores(protocol)
    execution_signature = _execution_signature(
        protocol, device, scope="upstream"
    )
    sample_days = tuple(
        dict.fromkeys(
            (
                protocol.extended_start,
                date(2024, 3, 31),
                date(2024, 10, 27),
                protocol.final_start,
                protocol.residual_end,
            )
        )
    )
    preflight: list[dict[str, Any]] = []
    for day in sample_days:
        cutoff = _cutoff_for_day(day, protocol.timezone)
        for alias, series_name in protocol.observed_series.items():
            _context, audit = select_observed_context_asof(
                stores[series_name],
                series_name=series_name,
                request_cutoff_utc=cutoff,
                context_length=protocol.context_length,
                maximum_gap_hours=protocol.maximum_gap_hours,
            )
            preflight.append(
                {
                    "delivery_day_local": day,
                    "alias": alias,
                    "series": series_name,
                    **dict(audit),
                }
            )
    LOGGER.info(
        "Preflight upstream valide: %d contextes echantillons.", len(preflight)
    )
    replay = replay_historical_residual_load(
        stores,
        start_day=protocol.extended_start,
        end_day=protocol.residual_end,
        device=device,
        local_files_only=local_files_only,
        batch_size=protocol.residual_batch_size,
        context_length=protocol.context_length,
        maximum_gap_hours=protocol.maximum_gap_hours,
        model_id=protocol.model_id,
        model_revision=protocol.model_revision,
        execution_signature=execution_signature,
        checkpoint_dir=protocol.experiment_root / "residual_replay" / "checkpoints",
        checkpoint_days=chunk_days,
        resume=resume,
    )
    replay_root = protocol.experiment_root / "residual_replay"
    replay_root.mkdir(parents=True, exist_ok=True)
    predictions_path = replay_root / "predictions.csv.gz"
    audits_path = replay_root / "audits.csv.gz"
    preflight_path = replay_root / "preflight.csv.gz"
    _write_csv_gzip(replay.predictions, predictions_path)
    _write_csv_gzip(replay.audits, audits_path)
    _write_csv_gzip(pd.DataFrame(preflight), preflight_path)
    wide = replay_to_wide(replay)
    manifest = {
        "status": "complete",
        "schema_version": 1,
        "replay_type": "chronos2_historical_residual_load",
        "model_id": protocol.model_id,
        "model_revision": protocol.model_revision,
        "execution_signature": execution_signature,
        "start_day_local": protocol.extended_start,
        "end_day_local": protocol.residual_end,
        "n_delivery_days": int(replay.predictions["delivery_day_local"].nunique()),
        "n_prediction_rows": len(replay.predictions),
        "n_audit_rows": len(replay.audits),
        "request_policy": "D-1 08:00 Europe/Paris",
        "observed_sources": {
            alias: {
                "series": series_name,
                "path": str(_observed_store_path(protocol, alias)),
                "sha256": _sha256(_observed_store_path(protocol, alias)),
            }
            for alias, series_name in protocol.observed_series.items()
        },
        "artifacts": {
            "predictions": {
                "path": str(predictions_path),
                "sha256": _sha256(predictions_path),
            },
            "audits": {"path": str(audits_path), "sha256": _sha256(audits_path)},
            "preflight": {
                "path": str(preflight_path),
                "sha256": _sha256(preflight_path),
            },
        },
        "checkpoint_paths": [str(path) for path in replay.checkpoint_paths],
        "foundation_model_pretraining_note": (
            "rolling-origin PIT replay; foundation-model pretraining cutoff "
            "is not asserted as OOF"
        ),
    }
    replay_manifest_path = replay_root / "manifest.json"
    _write_json(replay_manifest_path, manifest)
    _materialize_branch_pit(
        protocol,
        zones,
        wide,
        replay_manifest_path=replay_manifest_path,
    )


def _input_schema_sha256(data: Any) -> str:
    payload = {
        "known_future_columns": list(data.known_future_columns),
        "covariate_columns": list(data.covariates.columns),
        "model_context_columns": list(data.model_context_covariates.columns),
        "model_context_dtypes": [
            str(data.model_context_covariates[column].dtype)
            for column in data.model_context_covariates
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_equal_numeric(
    left: pd.DataFrame | pd.Series,
    right: pd.DataFrame | pd.Series,
    *,
    name: str,
    atol: float = 0.0,
) -> None:
    if not left.index.equals(right.index):
        raise HistoricalComparisonRunnerError(f"{name}: index differents.")
    if isinstance(left, pd.DataFrame):
        if not isinstance(right, pd.DataFrame) or list(left.columns) != list(right.columns):
            raise HistoricalComparisonRunnerError(f"{name}: schemas differents.")
        left_values = left.apply(pd.to_numeric, errors="coerce").to_numpy(float)
        right_values = right.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    else:
        left_values = pd.to_numeric(left, errors="coerce").to_numpy(float)
        right_values = pd.to_numeric(right, errors="coerce").to_numpy(float)
    finite_left = np.isfinite(left_values)
    finite_right = np.isfinite(right_values)
    if not np.array_equal(finite_left, finite_right):
        raise HistoricalComparisonRunnerError(f"{name}: masques differents.")
    if finite_left.any() and not np.allclose(
        left_values[finite_left],
        right_values[finite_right],
        rtol=0.0,
        atol=atol,
    ):
        raise HistoricalComparisonRunnerError(f"{name}: valeurs differentes.")


def _assert_branch_input_parity(control: Any, challenger: Any, *, zone: str) -> None:
    _assert_equal_numeric(control.target, challenger.target, name=f"{zone}/target", atol=2e-5)
    if list(control.known_future_columns) != list(challenger.known_future_columns):
        raise HistoricalComparisonRunnerError(
            f"{zone}: known_future_columns differents."
        )
    if _input_schema_sha256(control) != _input_schema_sha256(challenger):
        raise HistoricalComparisonRunnerError(
            f"{zone}: hash du schema de features different."
        )
    if list(control.model_context_covariates.columns) != list(
        challenger.model_context_covariates.columns
    ):
        raise HistoricalComparisonRunnerError(f"{zone}: schema model_context different.")
    treatment = set(RESIDUAL_LOAD_TREATMENT_COLUMNS)
    non_treatment = [
        column
        for column in control.model_context_covariates.columns
        if column not in treatment
    ]
    _assert_equal_numeric(
        control.model_context_covariates.loc[:, non_treatment],
        challenger.model_context_covariates.loc[:, non_treatment],
        name=f"{zone}/non_treatment",
    )
    differences = 0
    for alias in EXPECTED_ALIASES:
        raw_control = pd.to_numeric(
            control.model_context_covariates[alias], errors="coerce"
        )
        raw_challenger = pd.to_numeric(
            challenger.model_context_covariates[alias], errors="coerce"
        )
        mirror = f"known_{alias}_oracle"
        _assert_equal_numeric(
            raw_control,
            control.model_context_covariates[mirror],
            name=f"{zone}/{alias}/control_mirror",
        )
        _assert_equal_numeric(
            raw_challenger,
            challenger.model_context_covariates[mirror],
            name=f"{zone}/{alias}/challenger_mirror",
        )
        if not raw_control.notna().equals(raw_challenger.notna()):
            raise HistoricalComparisonRunnerError(
                f"{zone}/{alias}: masque de traitement different."
            )
        common = raw_control.notna()
        differences += int(
            (~np.isclose(
                raw_control.loc[common].to_numpy(float),
                raw_challenger.loc[common].to_numpy(float),
                rtol=0.0,
                atol=1e-12,
            )).sum()
        )
    if differences == 0:
        raise HistoricalComparisonRunnerError(
            f"{zone}: aucune valeur de traitement ne differe entre les branches."
        )


def _assert_sealed_reference_fidelity(data: Any, zone: ZoneSpec) -> None:
    reference = _source_model_frame(zone)
    observed = data.model_context_covariates.copy()
    observed.index = observed.index.tz_convert("UTC")
    reference.index = reference.index.tz_convert("UTC")
    _assert_equal_numeric(
        reference,
        observed,
        name=f"{zone.code}/sealed_model_context",
        atol=2e-6,
    )
    input_manifest = pd.read_csv(
        zone.sealed_extended_run / "inputs" / "input_manifest.csv"
    )
    declared_known = input_manifest.loc[
        input_manifest["known_future"].astype(str).str.lower().eq("true"),
        "alias",
    ].astype(str).tolist()
    if declared_known != list(data.known_future_columns):
        raise HistoricalComparisonRunnerError(
            f"{zone.code}: contrat known_future different du run scelle."
        )


def _price_source_hashes(
    protocol: Protocol,
    zone: ZoneSpec,
    branch: str,
    config_path: Path,
) -> dict[str, str]:
    root = _branch_pit_dir(protocol, branch, zone.code)
    _validate_branch_pit(protocol, zone, branch)
    hashes = {
        "generated_config": _sha256(config_path),
        "sealed_target": _sha256(_source_target_file(zone)),
        "pit_manifest": _sha256(root / "manifest.json"),
        **{
            f"pit_{alias}": _sha256(root / f"{alias}.parquet")
            for alias in EXPECTED_ALIASES
        },
    }
    inputs_dir = (
        protocol.experiment_root
        / "price_replay"
        / branch
        / zone.code.lower()
        / "inputs"
    )
    if not inputs_dir.is_dir():
        raise FileNotFoundError(inputs_dir)
    for path in sorted(inputs_dir.rglob("*")):
        if path.is_file():
            relative = path.relative_to(inputs_dir).as_posix()
            hashes[f"materialized_input:{relative}"] = _content_sha256(path)
    return hashes


def _validate_price_replay_outputs(
    protocol: Protocol,
    zone: ZoneSpec,
    branch: str,
    config_path: Path,
    *,
    device: str,
) -> dict[str, Any]:
    price_dir = (
        protocol.experiment_root
        / "price_replay"
        / branch
        / zone.code.lower()
    )
    outputs_path = price_dir / "outputs_manifest.json"
    if not outputs_path.is_file():
        raise FileNotFoundError(outputs_path)
    outputs = json.loads(outputs_path.read_text(encoding="utf-8"))
    identity = _mapping(outputs.get("identity"), name="outputs.identity")
    plans = generate_delivery_plans(
        protocol.extended_start,
        protocol.oof_end,
        forecast_origin_local_time=protocol.cutoff_clock,
        timezone=zone.timezone,
    )
    feature_schema = str(identity.get("feature_schema_sha256", ""))
    if len(feature_schema) != 64:
        raise HistoricalComparisonRunnerError(
            f"{zone.code}/{branch}: feature_schema_sha256 invalide."
        )
    expected_identity = replay_identity(
        plans,
        zone=zone.code,
        timezone=zone.timezone,
        model_id=protocol.model_id,
        model_revision=protocol.model_revision,
        residual_load_source=(
            "saturn" if branch == "saturn" else "chronos2_historical_replay"
        ),
        source_hashes=_price_source_hashes(
            protocol, zone, branch, config_path
        ),
        feature_schema_sha256=feature_schema,
        execution_signature=_execution_signature(
            protocol, device, scope="price"
        ),
    )
    if dict(identity) != expected_identity:
        raise HistoricalComparisonRunnerError(
            f"{zone.code}/{branch}: replay prix obsolete; relancez PriceReplay."
        )

    canonical = {
        "all953": price_dir / "chronos_price_all953.csv.gz",
        "ext223": price_dir / "chronos_price_ext223.csv.gz",
        "oof730": price_dir / "chronos_price_oof730.csv.gz",
        "live": price_dir / "chronos_price_live.csv",
    }
    for key, path in canonical.items():
        entry = _mapping(outputs.get(key), name=f"outputs.{key}")
        if (
            Path(str(entry.get("path", ""))).resolve() != path.resolve()
            or entry.get("sha256") != _sha256(path)
        ):
            raise HistoricalComparisonRunnerError(
                f"{zone.code}/{branch}: artefact prix {key} non scelle."
            )

    sidecar_path = canonical["all953"].with_name(
        canonical["all953"].name + ".manifest.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        any(sidecar.get(key) != value for key, value in expected_identity.items())
        or sidecar.get("status") != "complete"
        or int(sidecar.get("completed_days", -1)) != len(plans)
        or sidecar.get("output_sha256") != _sha256(canonical["all953"])
    ):
        raise HistoricalComparisonRunnerError(
            f"{zone.code}/{branch}: sidecar prix incomplet ou obsolete."
        )
    replay = load_chronos_oof(canonical["all953"])
    ext_plans = plans[:223]
    oof_plans = plans[223:]
    for path, subset in (
        (canonical["ext223"], ext_plans),
        (canonical["oof730"], oof_plans),
    ):
        expected = select_replay_days(replay, subset)
        observed = load_chronos_oof(path)
        try:
            pd.testing.assert_frame_equal(
                observed,
                expected,
                check_exact=False,
                rtol=0.0,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise HistoricalComparisonRunnerError(
                f"{zone.code}/{branch}: split prix divergent: {path}."
            ) from exc
    live_plan = generate_delivery_plans(
        protocol.residual_end,
        protocol.residual_end,
        forecast_origin_local_time=protocol.cutoff_clock,
        timezone=zone.timezone,
    )[0]
    normalize_chronos_future(pd.read_csv(canonical["live"]), live_plan)
    live_sidecar_path = canonical["live"].with_name(
        canonical["live"].name + ".manifest.json"
    )
    live_sidecar = json.loads(live_sidecar_path.read_text(encoding="utf-8"))
    if (
        live_sidecar.get("identity") != expected_identity
        or live_sidecar.get("sha256") != _sha256(canonical["live"])
        or live_sidecar.get("delivery_day_local")
        != live_plan.delivery_date.isoformat()
        or live_sidecar.get("forecast_origin_utc")
        != str(live_plan.forecast_origin_utc)
    ):
        raise HistoricalComparisonRunnerError(
            f"{zone.code}/{branch}: sidecar live invalide."
        )
    return outputs


def _price_replay_stage(
    protocol: Protocol,
    zones: Sequence[ZoneSpec],
    *,
    device: str,
    local_files_only: bool,
    chunk_days: int,
    resume: bool,
) -> None:
    set_reproducibility(42)
    execution_signature = _execution_signature(protocol, device, scope="price")
    all_plans_by_zone: dict[str, Sequence[Any]] = {}
    for zone in zones:
        LOGGER.info("Preparation des deux branches prix %s...", zone.code)
        prepared: dict[str, tuple[Mapping[str, Any], Path, Any]] = {}
        for branch in BRANCHES:
            config, config_path = _generated_base_config(
                protocol,
                zone,
                branch,
                local_files_only=local_files_only,
            )
            zone_configs = build_zone_configs(config, [zone.code], None, None)
            if len(zone_configs) != 1:
                raise HistoricalComparisonRunnerError(
                    f"{zone.code}/{branch}: selection de zone invalide."
                )
            inputs_dir = (
                protocol.experiment_root
                / "price_replay"
                / branch
                / zone.code.lower()
                / "inputs"
            )
            data = prepare_zone_data(
                zone_configs[0], config, config_path.parent, False, inputs_dir
            )
            prepared[branch] = (config, config_path, data)
        _assert_branch_input_parity(
            prepared["saturn"][2], prepared["chronos2"][2], zone=zone.code
        )
        _assert_sealed_reference_fidelity(prepared["saturn"][2], zone)

        plans = generate_delivery_plans(
            protocol.extended_start,
            protocol.oof_end,
            forecast_origin_local_time=protocol.cutoff_clock,
            timezone=zone.timezone,
        )
        all_plans_by_zone[zone.code] = plans
        ext_plans = [
            plan
            for plan in plans
            if protocol.extended_start <= plan.delivery_date <= protocol.extended_end
        ]
        oof_plans = [
            plan
            for plan in plans
            if protocol.oof_start <= plan.delivery_date <= protocol.oof_end
        ]
        if len(ext_plans) != 223 or len(oof_plans) != 730:
            raise HistoricalComparisonRunnerError(
                f"{zone.code}: plans EXT/OOF inattendus."
            )
        for branch in BRANCHES:
            config, config_path, data = prepared[branch]
            price_dir = (
                protocol.experiment_root
                / "price_replay"
                / branch
                / zone.code.lower()
            )
            all_path = price_dir / "chronos_price_all953.csv.gz"
            runtime_holder: dict[str, Any] = {}

            def get_runtime() -> Any:
                if "runtime" not in runtime_holder:
                    runtime_holder["runtime"] = load_model(
                        config,
                        device,
                        local_files_only,
                    )
                return runtime_holder["runtime"]

            def execute_chunk(chunk: Sequence[Any]) -> pd.DataFrame:
                executor = make_existing_forecasting_executor(
                    data=data,
                    runtime=get_runtime(),
                    context_length=protocol.context_length,
                    origin_batch_size=protocol.origin_batch_size,
                    model_batch_size=protocol.model_batch_size,
                    with_covariates=True,
                    variant="hourly_oof_residual_source_comparison",
                )
                return execute_grouped_chronos_backtest(chunk, executor)

            source_hashes = _price_source_hashes(
                protocol, zone, branch, config_path
            )
            identity = replay_identity(
                plans,
                zone=zone.code,
                timezone=zone.timezone,
                model_id=protocol.model_id,
                model_revision=protocol.model_revision,
                residual_load_source=(
                    "saturn" if branch == "saturn" else "chronos2_historical_replay"
                ),
                source_hashes=source_hashes,
                feature_schema_sha256=_input_schema_sha256(data),
                execution_signature=execution_signature,
            )
            LOGGER.info("Replay prix %s/%s...", branch, zone.code)
            replay = generate_checkpointed_price_replay(
                plans,
                target=data.target,
                execute_chunk=execute_chunk,
                output_path=all_path,
                identity=identity,
                chunk_days=chunk_days,
                resume=resume,
            )
            ext = select_replay_days(replay, ext_plans).reset_index()
            oof = select_replay_days(replay, oof_plans).reset_index()
            _write_csv_gzip(ext, price_dir / "chronos_price_ext223.csv.gz")
            _write_csv_gzip(oof, price_dir / "chronos_price_oof730.csv.gz")

            future_index = data.model_context_covariates.index.difference(
                data.target.index, sort=False
            ).tz_convert("UTC")
            live_plans = generate_delivery_plans(
                protocol.residual_end,
                protocol.residual_end,
                forecast_origin_local_time=protocol.cutoff_clock,
                timezone=zone.timezone,
            )
            if len(live_plans) != 1 or not future_index.equals(
                live_plans[0].delivery_index_utc
            ):
                raise HistoricalComparisonRunnerError(
                    f"{zone.code}/{branch}: horizon live different du protocole."
                )
            live_path = price_dir / "chronos_price_live.csv"
            live_manifest = price_dir / "chronos_price_live.csv.manifest.json"
            live_complete = False
            if resume and live_path.is_file() and live_manifest.is_file():
                payload = json.loads(live_manifest.read_text(encoding="utf-8"))
                live_complete = (
                    payload.get("identity") == identity
                    and payload.get("sha256") == _sha256(live_path)
                    and payload.get("delivery_day_local")
                    == live_plans[0].delivery_date.isoformat()
                    and payload.get("forecast_origin_utc")
                    == str(live_plans[0].forecast_origin_utc)
                )
                if live_complete:
                    try:
                        normalize_chronos_future(
                            pd.read_csv(live_path), live_plans[0]
                        )
                    except Exception:
                        live_complete = False
            if not live_complete:
                live = run_existing_live_forecast(
                    live_plans[0],
                    data=data,
                    runtime=get_runtime(),
                    context_length=protocol.context_length,
                    model_batch_size=protocol.model_batch_size,
                    with_covariates=True,
                    variant="hourly_live_residual_source_comparison",
                )
                _write_csv(live.reset_index(), live_path)
                _write_json(
                    live_manifest,
                    {
                        "identity": identity,
                        "sha256": _sha256(live_path),
                        "rows": len(live),
                        "delivery_day_local": live_plans[
                            0
                        ].delivery_date.isoformat(),
                        "forecast_origin_utc": str(
                            live_plans[0].forecast_origin_utc
                        ),
                    },
                )
            _write_json(
                price_dir / "outputs_manifest.json",
                {
                    "identity": identity,
                    "all953": {"path": str(all_path), "sha256": _sha256(all_path)},
                    "ext223": {
                        "path": str(price_dir / "chronos_price_ext223.csv.gz"),
                        "sha256": _sha256(price_dir / "chronos_price_ext223.csv.gz"),
                    },
                    "oof730": {
                        "path": str(price_dir / "chronos_price_oof730.csv.gz"),
                        "sha256": _sha256(price_dir / "chronos_price_oof730.csv.gz"),
                    },
                    "live": {"path": str(live_path), "sha256": _sha256(live_path)},
                },
            )


def _run_contract(
    protocol: Protocol,
    zone: ZoneSpec,
    branch: str,
    *,
    native_model: str = "residual_corrected",
) -> dict[str, Any]:
    pit_manifest = _branch_pit_dir(protocol, branch, zone.code) / "manifest.json"
    price_manifest = (
        protocol.experiment_root
        / "price_replay"
        / branch
        / zone.code.lower()
        / "outputs_manifest.json"
    )
    return {
        "protocol_config_sha256": _sha256(protocol.config_path),
        "historical_comparison_experiment_id": str(
            protocol.config.get("experiment_id")
        ),
        "zone": zone.code,
        "timezone": zone.timezone,
        "comparison_branch": branch,
        "run_type": "historical_residual_load_full_recalculation",
        "residual_load_source": (
            "saturn" if branch == "saturn" else "chronos2_historical_replay"
        ),
        "native_model": native_model,
        "model_id": protocol.model_id,
        "model_revision": protocol.model_revision,
        "model_context_length": protocol.context_length,
        "implementation_sha256": _implementation_hashes(
            protocol, "downstream"
        ),
        "pit_manifest": str(pit_manifest),
        "pit_manifest_sha256": _sha256(pit_manifest),
        "price_replay_manifest": str(price_manifest),
        "price_replay_manifest_sha256": _sha256(price_manifest),
        "statistics_history_reused": False,
        "downstream_recalculated": [
            "chronos2_price_ext223",
            "chronos2_price_oof730",
            "lear",
            "catboost",
            "convex_ensemble",
            "extended_residual_corrector",
        ],
        "comparison_windows": {
            "extended": [str(protocol.extended_start), str(protocol.extended_end)],
            "oof": [str(protocol.oof_start), str(protocol.oof_end)],
            "sealed_final": [str(protocol.final_start), str(protocol.final_end)],
        },
    }


def _seal_run_manifest(
    run_dir: Path,
    *,
    contract: Mapping[str, Any],
) -> None:
    manifest_path = run_dir / "run_manifest.json"
    checksum_path = run_dir / "artifact_checksums.json"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(
            f"Run non scellable, manifestes absents: {run_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"{manifest_path}: objet JSON attendu.")
    manifest.update(dict(contract))
    _write_json(manifest_path, manifest)

    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    if not isinstance(checksums, dict) or not isinstance(
        checksums.get("artifacts"), list
    ):
        raise TypeError(f"{checksum_path}: manifeste de checksums invalide.")
    found = False
    for entry in checksums["artifacts"]:
        if not isinstance(entry, dict):
            continue
        raw = Path(str(entry.get("path", "")))
        candidate = raw if raw.is_absolute() else run_dir / raw
        if candidate.resolve() == manifest_path.resolve():
            entry["size_bytes"] = manifest_path.stat().st_size
            entry["sha256"] = _sha256(manifest_path)
            found = True
    if not found:
        checksums["artifacts"].append(
            {
                "path": "run_manifest.json",
                "role": "run_artifact",
                "size_bytes": manifest_path.stat().st_size,
                "sha256": _sha256(manifest_path),
            }
        )
    checksums["output_directory"] = str(run_dir)
    _write_json(checksum_path, checksums)


def _artifact_checksums_valid(
    run_dir: Path,
    *,
    project_root: Path | None = None,
) -> bool:
    checksum_path = run_dir / "artifact_checksums.json"
    if not checksum_path.is_file():
        return False
    try:
        payload = json.loads(checksum_path.read_text(encoding="utf-8"))
        entries = payload.get("artifacts")
        declared_output = Path(str(payload.get("output_directory", "")))
        if (
            str(payload.get("algorithm", "")).lower() != "sha256"
            or not isinstance(entries, list)
            or not entries
            or not declared_output.is_absolute()
            or declared_output.resolve() != run_dir.resolve()
        ):
            return False
        manifest_declared = False
        for entry in entries:
            if not isinstance(entry, Mapping):
                return False
            raw = Path(str(entry.get("path", "")))
            if not str(raw):
                return False
            if raw.is_absolute():
                path = raw
            elif entry.get("role") == "source_code":
                if project_root is None:
                    return False
                path = project_root / raw
            else:
                path = run_dir / raw
            if not path.is_file():
                return False
            if int(entry.get("size_bytes", -1)) != path.stat().st_size:
                return False
            if str(entry.get("sha256", "")).lower() != _sha256(path):
                return False
            if path.resolve() == (run_dir / "run_manifest.json").resolve():
                manifest_declared = True
        return manifest_declared
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _completed_run(
    run_dir: Path,
    *,
    expected_contract: Mapping[str, Any],
    project_root: Path,
) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    normalized_contract = _json_safe(dict(expected_contract))
    if any(manifest.get(key) != value for key, value in normalized_contract.items()):
        return False
    return _artifact_checksums_valid(run_dir, project_root=project_root)


def _run_command(command: Sequence[str], *, cwd: Path) -> None:
    LOGGER.info("Execution: %s", json.dumps(list(command), ensure_ascii=False))
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "42"},
    )
    if completed.returncode != 0:
        raise HistoricalComparisonRunnerError(
            f"Commande en echec (code={completed.returncode}): {command[1]}"
        )


def _publish_staging_directory(
    staging: Path,
    output: Path,
    *,
    overwrite: bool,
) -> None:
    if not staging.is_dir() or staging.parent != output.parent:
        raise ValueError("Le staging doit etre un dossier frere de la sortie.")
    previous: Path | None = None
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        previous = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
        output.replace(previous)
    try:
        os.replace(staging, output)
    except Exception:
        if previous is not None and previous.exists() and not output.exists():
            os.replace(previous, output)
        raise
    if previous is not None:
        shutil.rmtree(previous)


def _run_base_downstream(
    protocol: Protocol,
    zone: ZoneSpec,
    branch: str,
    *,
    config_path: Path,
    local_files_only: bool,
    overwrite: bool,
) -> Path:
    output = protocol.experiment_root / "runs" / branch / "base" / zone.code.lower()
    expected_source = (
        "saturn" if branch == "saturn" else "chronos2_historical_replay"
    )
    contract = _run_contract(protocol, zone, branch)
    if not overwrite and _completed_run(
        output,
        expected_contract=contract,
        project_root=protocol.project_root,
    ):
        LOGGER.info("Run base deja complet: %s", output)
        return output
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Run base incomplet/existant: {output}. Utilisez --overwrite."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    price_dir = (
        protocol.experiment_root
        / "price_replay"
        / branch
        / zone.code.lower()
    )
    command = [
        sys.executable,
        str(protocol.project_root / "run_chronos2_hourly.py"),
        "--config",
        str(config_path),
        "--zone",
        zone.code,
        "--output-dir",
        str(staging),
        "--chronos-oof-file",
        str(price_dir / "chronos_price_oof730.csv.gz"),
        "--chronos-live-file",
        str(price_dir / "chronos_price_live.csv"),
    ]
    if local_files_only:
        command.append("--local-files-only")
    try:
        _run_command(command, cwd=protocol.project_root)
        _seal_run_manifest(
            staging,
            contract=contract,
        )
        _publish_staging_directory(staging, output, overwrite=overwrite)
        # The base runner writes its checksum manifest while the artifacts are
        # still in the sibling staging directory.  Seal once more after the
        # atomic move so ``output_directory`` names the immutable publication.
        _seal_run_manifest(
            output,
            contract=contract,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _run_extended_downstream(
    protocol: Protocol,
    zone: ZoneSpec,
    branch: str,
    *,
    config_path: Path,
    threads: int,
    overwrite: bool,
) -> Path:
    output = protocol.experiment_root / "runs" / branch / "extended" / zone.code.lower()
    expected_source = (
        "saturn" if branch == "saturn" else "chronos2_historical_replay"
    )
    contract = _run_contract(protocol, zone, branch)
    if not overwrite and _completed_run(
        output,
        expected_contract=contract,
        project_root=protocol.project_root,
    ):
        LOGGER.info("Run EXT deja complet: %s", output)
        return output
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Run EXT incomplet/existant: {output}. Utilisez --overwrite."
        )
    command = [
        sys.executable,
        str(protocol.project_root / "run_extended_residual_hourly.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
        "--threads",
        str(threads),
    ]
    if overwrite:
        command.append("--overwrite")
    _run_command(command, cwd=protocol.project_root)
    _seal_run_manifest(
        output,
        contract=contract,
    )
    return output


def _downstream_stage(
    protocol: Protocol,
    zones: Sequence[ZoneSpec],
    *,
    device: str,
    local_files_only: bool,
    threads: int,
    overwrite: bool,
) -> None:
    for zone in zones:
        for branch in BRANCHES:
            _config, base_config_path = _generated_base_config(
                protocol,
                zone,
                branch,
                local_files_only=local_files_only,
            )
            _validate_price_replay_outputs(
                protocol,
                zone,
                branch,
                base_config_path,
                device=device,
            )
            _run_base_downstream(
                protocol,
                zone,
                branch,
                config_path=base_config_path,
                local_files_only=local_files_only,
                overwrite=overwrite,
            )
            extended_config_path = _generated_extended_config(
                protocol, zone, branch, threads=threads
            )
            _run_extended_downstream(
                protocol,
                zone,
                branch,
                config_path=extended_config_path,
                threads=threads,
                overwrite=overwrite,
            )


def _blend_stage(
    protocol: Protocol,
    zones: Sequence[ZoneSpec],
    *,
    overwrite: bool,
) -> None:
    for zone in zones:
        if zone.sealed_blend_run is None:
            continue
        for branch in BRANCHES:
            autonomous = (
                protocol.experiment_root
                / "runs"
                / branch
                / "extended"
                / zone.code.lower()
            )
            output = (
                protocol.experiment_root
                / "runs"
                / branch
                / "blend"
                / zone.code.lower()
            )
            expected_source = (
                "saturn" if branch == "saturn" else "chronos2_historical_replay"
            )
            contract = _run_contract(
                protocol, zone, branch, native_model="mkonline_blend"
            )
            reference_forecast = (
                zone.sealed_blend_run
                / f"forecast_hourly_{zone.code.lower()}.csv"
            )
            contract["blend_sources_sha256"] = {
                "autonomous_run_manifest": _sha256(
                    autonomous / "run_manifest.json"
                ),
                "autonomous_checksum_manifest": _sha256(
                    autonomous / "artifact_checksums.json"
                ),
                "autonomous_backtest": _sha256(
                    autonomous / "backtest_hourly_oof.csv.gz"
                ),
                "autonomous_forecast": _sha256(
                    autonomous / f"forecast_hourly_{zone.code.lower()}.csv"
                ),
                "reference_run_manifest": _sha256(
                    zone.sealed_blend_run / "run_manifest.json"
                ),
                "reference_metrics": _sha256(
                    zone.sealed_blend_run / "metrics_hourly.json"
                ),
                "reference_backtest": _sha256(
                    zone.sealed_blend_run / "backtest_hourly_oof.csv.gz"
                ),
                "reference_forecast": _sha256(reference_forecast),
            }
            if not overwrite and _completed_run(
                output,
                expected_contract=contract,
                project_root=protocol.project_root,
            ):
                LOGGER.info("Run blend deja complet: %s", output)
                continue
            publish_historical_blend_replay(
                autonomous,
                zone.sealed_blend_run,
                output,
                residual_load_source=expected_source,
                overwrite=overwrite,
                title=(
                    f"{zone.code} - recalcul historique residual_load - "
                    f"{branch} MKOnline blend"
                ),
            )
            _seal_run_manifest(output, contract=contract)


def _publish_comparison_with_optional_replace(
    control: Path,
    challenger: Path,
    output: Path,
    *,
    zone: str,
    title: str,
    overwrite: bool,
) -> Any:
    previous: Path | None = None
    if output.exists():
        manifest_path = output / "run_manifest.json"
        if not overwrite and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            control_entry = manifest.get("control")
            challenger_entry = manifest.get("challenger")
            current = bool(
                manifest.get("run_type")
                == "historical_residual_load_source_comparison"
                and manifest.get("statistics_history_reused") is False
                and isinstance(control_entry, Mapping)
                and isinstance(challenger_entry, Mapping)
                and Path(str(control_entry.get("run", ""))).resolve()
                == control.resolve()
                and Path(str(challenger_entry.get("run", ""))).resolve()
                == challenger.resolve()
                and control_entry.get("run_manifest_sha256")
                == _sha256(control / "run_manifest.json")
                and control_entry.get("checksum_manifest_sha256")
                == _sha256(control / "artifact_checksums.json")
                and challenger_entry.get("run_manifest_sha256")
                == _sha256(challenger / "run_manifest.json")
                and challenger_entry.get("checksum_manifest_sha256")
                == _sha256(challenger / "artifact_checksums.json")
                and _artifact_checksums_valid(output)
            )
            if current:
                LOGGER.info("Comparaison deja publiee: %s", output)
                return None
        if not overwrite:
            raise FileExistsError(output)
        previous = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
        output.replace(previous)
    try:
        result = publish_historical_residual_comparison(
            control,
            challenger,
            output,
            zone=zone,
            title=title,
        )
    except Exception:
        if previous is not None and previous.exists() and not output.exists():
            previous.replace(output)
        raise
    if previous is not None:
        shutil.rmtree(previous)
    return result


def _export_comparison(
    comparison_dir: Path,
    protocol: Protocol,
    zone: ZoneSpec,
    variant: str,
    *,
    export_day: date,
    overwrite: bool,
) -> Path:
    report_candidates = list(comparison_dir.glob("*.html"))
    if len(report_candidates) != 1:
        raise HistoricalComparisonRunnerError(
            f"{comparison_dir}: un rapport HTML exact attendu."
        )
    forecast_path = comparison_dir / f"forecast_hourly_{zone.code.lower()}.csv"
    forecast = pd.read_csv(forecast_path)
    index = pd.DatetimeIndex(
        pd.to_datetime(forecast["delivery_start_utc"], utc=True, errors="raise")
    )
    days = pd.Index(index.tz_convert(zone.timezone).date).unique().tolist()
    if days != [export_day]:
        raise HistoricalComparisonRunnerError(
            f"{zone.code}/{variant}: le forecast est pour {days}, pas {export_day}."
        )
    destination = (
        protocol.export_root
        / export_day.isoformat()
        / zone.code.lower()
        / variant
    )
    stem = f"forecast_{zone.code.lower()}_{export_day.isoformat()}_{variant}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        shutil.copy2(report_candidates[0], staging / f"{stem}.html")
        timing_columns = [
            column
            for column in (
                "delivery_start_utc",
                "delivery_start_local",
                "utc_offset",
                "utc_offset_minutes",
                "fold",
                "local_date",
                "local_hour",
                "delivery_hour_position",
                "hours_in_local_day",
            )
            if column in forecast
        ]
        exported = forecast.loc[:, timing_columns].copy()
        exported.insert(0, "zone", zone.code)
        exported.insert(1, "forecast_variant", variant)
        exported.insert(2, "source_model", "chronos_residual_load")
        exported.insert(
            3, "residual_load_source", "chronos2_historical_replay"
        )
        exported.insert(4, "uses_mkonline", variant == "blend")
        for quantile in ("q10", "q50", "q90"):
            if quantile not in forecast:
                raise HistoricalComparisonRunnerError(
                    f"{forecast_path}: {quantile} absent."
                )
            exported[quantile] = pd.to_numeric(
                forecast[quantile], errors="raise"
            )
        exported["price_eur_mwh"] = exported["q50"]
        exported.to_csv(
            staging / f"{stem}.csv", index=False, lineterminator="\n"
        )
        if destination.is_dir() and not overwrite:
            expected = {f"{stem}.html", f"{stem}.csv"}
            observed = {
                path.name for path in destination.iterdir() if path.is_file()
            }
            current = observed == expected and all(
                _sha256(destination / name) == _sha256(staging / name)
                for name in expected
            )
            if current:
                LOGGER.info("Export deja publie et identique: %s", destination)
                shutil.rmtree(staging)
                return destination
            raise FileExistsError(
                f"Export obsolete/incomplet: {destination}. "
                "Utilisez --overwrite."
            )
        _publish_staging_directory(staging, destination, overwrite=overwrite)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _report_stage(
    protocol: Protocol,
    zones: Sequence[ZoneSpec],
    *,
    export_day: date,
    overwrite: bool,
) -> None:
    summary: list[dict[str, Any]] = []
    for zone in zones:
        variants = ["autonomous"]
        if zone.sealed_blend_run is not None:
            variants.append("blend")
        for variant in variants:
            source_kind = "extended" if variant == "autonomous" else "blend"
            control = (
                protocol.experiment_root
                / "runs"
                / "saturn"
                / source_kind
                / zone.code.lower()
            )
            challenger = (
                protocol.experiment_root
                / "runs"
                / "chronos2"
                / source_kind
                / zone.code.lower()
            )
            output = (
                protocol.experiment_root
                / "comparison"
                / zone.code.lower()
                / variant
            )
            result = _publish_comparison_with_optional_replace(
                control,
                challenger,
                output,
                zone=zone.code,
                title=(
                    f"Forecast {zone.code} {export_day.isoformat()} - "
                    + (
                        "autonome sans MKOnline"
                        if variant == "autonomous"
                        else "blend MKOnline"
                    )
                    + " - comparaison residual_load Saturn vs Chronos-2"
                ),
                overwrite=overwrite,
            )
            _export_comparison(
                output,
                protocol,
                zone,
                variant,
                export_day=export_day,
                overwrite=overwrite,
            )
            metrics = pd.read_csv(output / "metrics_hourly.csv")
            values = metrics.set_index("model")["mae"].to_dict()
            summary.append(
                {
                    "zone": zone.code,
                    "variant": variant,
                    "saturn_mae": float(values["saturn_residual_load"]),
                    "chronos2_mae": float(values["chronos_residual_load"]),
                    "chronos2_minus_saturn_mae": float(
                        values["chronos_residual_load"]
                        - values["saturn_residual_load"]
                    ),
                    "report": str(
                        next(
                            (
                                protocol.export_root
                                / export_day.isoformat()
                                / zone.code.lower()
                                / variant
                            ).glob("*.html")
                        )
                    ),
                    "newly_published": result is not None,
                }
            )
    summary_frame = pd.DataFrame(summary)
    _write_csv_gzip(
        summary_frame,
        protocol.experiment_root / "comparison" / "metrics_summary.csv.gz",
    )
    _write_json(
        protocol.experiment_root / "comparison" / "manifest.json",
        {
            "status": "complete",
            "statistics_history_reused": False,
            "export_day": export_day,
            "reports": summary,
        },
    )


def _print_plan(protocol: Protocol, zones: Sequence[ZoneSpec]) -> None:
    payload = _protocol_manifest(protocol, zones)
    payload["estimated_work"] = {
        "residual_load_trajectories": 954 * 5,
        "price_trajectories": 953 * len(zones) * 2,
        "base_downstream_runs": len(zones) * 2,
        "extended_downstream_runs": len(zones) * 2,
        "blend_replays": 2
        * sum(zone.sealed_blend_run is not None for zone in zones),
        "reports": sum(
            1 + int(zone.sealed_blend_run is not None) for zone in zones
        ),
    }
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.chunk_days < 1:
        raise ValueError("--chunk-days doit etre >= 1.")
    project_root = Path(__file__).resolve().parent
    config_path = _resolve(args.config, base=project_root)
    protocol = _load_protocol(config_path, project_root)
    zones = _selected_zones(protocol, args.zones)
    _print_plan(protocol, zones)
    if args.stage == "plan":
        return 0
    protocol.experiment_root.mkdir(parents=True, exist_ok=True)
    # ``protocol.json`` describes the immutable experiment configuration, not
    # the country subset selected for this particular invocation.  Keeping all
    # configured zones prevents a later partial stage from silently narrowing
    # the provenance of already-published countries.
    _write_json(
        protocol.experiment_root / "protocol.json",
        _protocol_manifest(protocol, tuple(protocol.zones.values())),
    )
    local_files_only = not bool(args.allow_model_download)
    resume = not bool(args.no_resume)
    requested = (
        (
            "sync-observed",
            "residual-replay",
            "price-replay",
            "downstream",
            "blend",
            "report",
        )
        if args.stage == "all"
        else (args.stage,)
    )
    if "sync-observed" in requested and not args.skip_observed_sync:
        if args.history_chunk_days is not None:
            mutable = copy.deepcopy(dict(protocol.config))
            mutable.setdefault("saturn", {})["history_chunk_days"] = int(
                args.history_chunk_days
            )
            protocol = Protocol(**{**protocol.__dict__, "config": mutable})
        _sync_observed_stage(protocol)
    if "residual-replay" in requested:
        _residual_replay_stage(
            protocol,
            zones,
            device=args.device,
            local_files_only=local_files_only,
            chunk_days=args.chunk_days,
            resume=resume,
        )
    if "price-replay" in requested:
        _price_replay_stage(
            protocol,
            zones,
            device=args.device,
            local_files_only=local_files_only,
            chunk_days=args.chunk_days,
            resume=resume,
        )
    if "downstream" in requested:
        _downstream_stage(
            protocol,
            zones,
            device=args.device,
            local_files_only=local_files_only,
            threads=args.threads,
            overwrite=args.overwrite,
        )
    if "blend" in requested:
        _blend_stage(protocol, zones, overwrite=args.overwrite)
    if "report" in requested:
        export_day = (
            _parse_day(args.export_delivery_day, name="--export-delivery-day")
            if args.export_delivery_day
            else protocol.residual_end
        )
        _report_stage(
            protocol,
            zones,
            export_day=export_day,
            overwrite=args.overwrite,
        )
    print(f"Experience: {protocol.experiment_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Execution interrompue.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("Echec du benchmark residual_load: %s", exc)
        raise SystemExit(1)
