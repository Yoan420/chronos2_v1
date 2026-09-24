#!/usr/bin/env python
"""Build the isolated causal residual-load input-correction challenger.

This runner is intentionally disconnected from ``Forecast.ps1 -Action Run``.
It reads the five production PIT stores and the local observed-vintage stores,
then publishes only below ``runs/experiments``.  It never synchronises Saturn
itself and never mutates a production configuration.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.residual_load_input_corrector import (
    RESIDUAL_LOAD_ALIASES,
    RESIDUAL_LOAD_OBS_SERIES_BY_ALIAS,
    conservative_label_end_utc,
    generate_prequential_residual_load_corrections,
    scheduled_live_origin_utc,
)


LOGGER = logging.getLogger("residual_load_input_correction")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "residual_load_input_correction.yaml"
STAGES = ("plan", "build", "report", "all")
SYNC_OBSERVED_COMMAND = (
    "& '.\\Forecast.ps1' -Action ResidualCompare "
    "-ResidualComparisonStage SyncObserved -Countries FR,DE,BE,NL,ES"
)


class ResidualLoadInputRunnerError(RuntimeError):
    """Raised when the offline challenger contract would be violated."""


@dataclass(frozen=True)
class AliasSpec:
    alias: str
    country: str
    forecast_path: Path
    observed_path: Path
    observed_series: str


@dataclass(frozen=True)
class RunnerConfig:
    project_root: Path
    source_path: Path
    raw: Mapping[str, Any]
    experiment_id: str
    output_root: Path
    corrected_directory: str
    audit_directory: str
    report_directory: str
    timezone: str
    origin_clock: str
    history_start_day: date
    evaluation_start_day: date
    evaluation_end_day: date
    final_start_day: date
    final_end_day: date
    label_delay_days: int
    forecast_fill_limit_hours: int
    forecast_minimum_coverage: float
    refit_every_days: int
    cold_start_policy: str
    minimum_training_rows: int
    max_abs_correction_gw: float | None
    forecast_change_lags_hours: tuple[int, ...]
    error_lags_hours: tuple[int, ...]
    error_rolling_windows_hours: tuple[int, ...]
    minimum_scoring_coverage: float
    minimum_scoring_ramps: int
    aliases: tuple[AliasSpec, ...]


@dataclass(frozen=True)
class BuildResult:
    output_root: Path
    manifest_path: Path
    checksum_manifest_path: Path
    corrected_paths: Mapping[str, Path]
    audit_path: Path


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} doit etre un mapping.")
    return value


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _day(value: Any, *, name: str) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is not None or timestamp != timestamp.normalize():
        raise ValueError(f"{name} doit etre une date locale YYYY-MM-DD.")
    return timestamp.date()


def _positive_hours(value: Any, *, name: str, minimum: int = 1) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} doit etre une liste d'heures.")
    result = tuple(dict.fromkeys(int(item) for item in value))
    if not result or any(item < minimum for item in result):
        raise ValueError(f"{name} doit contenir des entiers >= {minimum}.")
    return result


def _safe_subdirectory(value: Any, *, name: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or path in {Path("."), Path("")}:
        raise ValueError(f"{name} doit etre un sous-dossier relatif sur.")
    return path.as_posix()


def _confined_output(path: Path, *, project_root: Path) -> Path:
    allowed = (project_root / "runs" / "experiments").resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(
            f"La sortie doit rester sous runs/experiments, recu: {resolved}"
        ) from exc
    if not relative.parts:
        raise ValueError("La racine runs/experiments elle-meme est interdite.")
    return resolved


def load_config(
    path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> RunnerConfig:
    """Load and strictly validate the isolated experiment configuration."""

    root = Path(project_root).expanduser().resolve()
    source = _resolve(path, base=root)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    config = _mapping(payload, name="configuration")
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("schema_version doit valoir 1.")
    experiment_id = str(config.get("experiment_id", "")).strip()
    if experiment_id != "residual_load_input_correction_v1":
        raise ValueError("experiment_id doit rester residual_load_input_correction_v1.")

    protocol = _mapping(config.get("protocol"), name="protocol")
    timezone = str(protocol.get("timezone", ""))
    origin_clock = str(protocol.get("forecast_origin_local_time", ""))
    if timezone != "Europe/Paris" or origin_clock != "08:00":
        raise ValueError("Le protocole exige Europe/Paris et D-1 08:00.")
    history_start = _day(protocol.get("history_start_day"), name="history_start_day")
    evaluation_start = _day(
        protocol.get("evaluation_start_day"), name="evaluation_start_day"
    )
    evaluation_end = _day(protocol.get("evaluation_end_day"), name="evaluation_end_day")
    final_start = _day(protocol.get("final_start_day"), name="final_start_day")
    final_end = _day(protocol.get("final_end_day"), name="final_end_day")
    if not (history_start < evaluation_start <= final_start <= final_end <= evaluation_end):
        raise ValueError("Fenêtres history/evaluation/final incoherentes.")
    label_delay = int(protocol.get("label_delay_days", 0))
    if label_delay != 2:
        raise ValueError("label_delay_days doit rester egal a 2 (dernier label D-2).")

    inputs = _mapping(config.get("inputs"), name="inputs")
    forecast_root = _resolve(inputs.get("forecast_vintage_root", ""), base=root)
    observed_root = _resolve(inputs.get("observed_vintage_root", ""), base=root)
    fill_limit = int(inputs.get("forecast_fill_limit_hours", 0))
    if fill_limit < 0 or fill_limit > 6:
        raise ValueError("forecast_fill_limit_hours doit etre compris entre 0 et 6.")
    minimum_forecast_coverage = float(inputs.get("forecast_minimum_coverage", 0.5))
    if not np.isfinite(minimum_forecast_coverage) or not (
        0.0 < minimum_forecast_coverage <= 1.0
    ):
        raise ValueError("forecast_minimum_coverage doit etre dans ]0, 1].")
    raw_aliases = _mapping(inputs.get("aliases"), name="inputs.aliases")
    if set(raw_aliases) != set(RESIDUAL_LOAD_ALIASES):
        raise ValueError("inputs.aliases doit contenir exactement les cinq residual_load.")
    aliases: list[AliasSpec] = []
    for alias in RESIDUAL_LOAD_ALIASES:
        item = _mapping(raw_aliases[alias], name=f"inputs.aliases.{alias}")
        observed_series = str(item.get("observed_series", ""))
        expected_series = RESIDUAL_LOAD_OBS_SERIES_BY_ALIAS[alias]
        if observed_series != expected_series:
            raise ValueError(f"{alias}: observed_series doit etre {expected_series}.")
        aliases.append(
            AliasSpec(
                alias=alias,
                country=str(item.get("country", "")).upper(),
                forecast_path=_resolve(item.get("forecast_file", ""), base=forecast_root),
                observed_path=_resolve(item.get("observed_file", ""), base=observed_root),
                observed_series=observed_series,
            )
        )

    model = _mapping(config.get("model"), name="model")
    if (
        str(model.get("recipe")) != "blend_cat_hgb_w0.50"
        or float(model.get("catboost_weight", -1)) != 0.5
        or float(model.get("hist_gradient_boosting_weight", -1)) != 0.5
    ):
        raise ValueError("La recette doit rester CatBoost/HGB 50/50.")
    refit_days = int(model.get("refit_every_days", 1))
    if refit_days < 1:
        raise ValueError("refit_every_days doit etre >= 1.")
    cold_start_policy = str(model.get("cold_start_policy", "raw_passthrough"))
    if cold_start_policy != "raw_passthrough":
        raise ValueError(
            "Le challenger exige cold_start_policy: raw_passthrough."
        )
    raw_minimum_training_rows = model.get("minimum_training_rows", 48)
    minimum_training_rows = int(raw_minimum_training_rows)
    if (
        isinstance(raw_minimum_training_rows, (bool, np.bool_))
        or minimum_training_rows != raw_minimum_training_rows
        or minimum_training_rows < 48
    ):
        raise ValueError("minimum_training_rows doit etre >= 48.")
    raw_clip = model.get("max_abs_correction_gw", 8.0)
    clip = None if raw_clip is None else float(raw_clip)
    if clip is not None and (not np.isfinite(clip) or clip <= 0):
        raise ValueError("max_abs_correction_gw doit etre fini et > 0, ou null.")

    outputs = _mapping(config.get("outputs"), name="outputs")
    output_root = _confined_output(
        _resolve(outputs.get("experiment_root", ""), base=root), project_root=root
    )
    expected_output_root = (
        root / "runs" / "experiments" / experiment_id
    ).resolve()
    if output_root != expected_output_root:
        raise ValueError(
            "outputs.experiment_root doit etre exactement "
            f"{expected_output_root}."
        )
    output_directories = {
        "corrected": _safe_subdirectory(
            outputs.get("corrected_directory", "corrected"),
            name="outputs.corrected_directory",
        ),
        "audit": _safe_subdirectory(
            outputs.get("audit_directory", "audit"),
            name="outputs.audit_directory",
        ),
        "report": _safe_subdirectory(
            outputs.get("report_directory", "report"),
            name="outputs.report_directory",
        ),
    }
    directory_paths = {
        name: Path(value) for name, value in output_directories.items()
    }
    for left_name, left in directory_paths.items():
        for right_name, right in directory_paths.items():
            if left_name >= right_name:
                continue
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "Les dossiers corrected/audit/report doivent etre disjoints "
                    f"et non imbriques: {left_name}={left}, {right_name}={right}."
                )

    report_config = _mapping(config.get("report", {}), name="report")
    minimum_scoring_coverage = float(
        report_config.get("minimum_scoring_coverage", 0.95)
    )
    minimum_scoring_ramps = int(report_config.get("minimum_scoring_ramps", 720))
    if not np.isfinite(minimum_scoring_coverage) or not (
        0.0 < minimum_scoring_coverage <= 1.0
    ):
        raise ValueError("minimum_scoring_coverage doit etre dans ]0, 1].")
    if minimum_scoring_ramps < 1:
        raise ValueError("minimum_scoring_ramps doit etre >= 1.")
    return RunnerConfig(
        project_root=root,
        source_path=source,
        raw=config,
        experiment_id=experiment_id,
        output_root=output_root,
        corrected_directory=output_directories["corrected"],
        audit_directory=output_directories["audit"],
        report_directory=output_directories["report"],
        timezone=timezone,
        origin_clock=origin_clock,
        history_start_day=history_start,
        evaluation_start_day=evaluation_start,
        evaluation_end_day=evaluation_end,
        final_start_day=final_start,
        final_end_day=final_end,
        label_delay_days=label_delay,
        forecast_fill_limit_hours=fill_limit,
        forecast_minimum_coverage=minimum_forecast_coverage,
        refit_every_days=refit_days,
        cold_start_policy=cold_start_policy,
        minimum_training_rows=minimum_training_rows,
        max_abs_correction_gw=clip,
        forecast_change_lags_hours=_positive_hours(
            model.get("forecast_change_lags_hours", (1, 2, 24, 48, 168)),
            name="forecast_change_lags_hours",
        ),
        error_lags_hours=_positive_hours(
            model.get("error_lags_hours", (48, 72, 168, 336)),
            name="error_lags_hours",
            minimum=48,
        ),
        error_rolling_windows_hours=_positive_hours(
            model.get("error_rolling_windows_hours", (24, 72, 168)),
            name="error_rolling_windows_hours",
        ),
        minimum_scoring_coverage=minimum_scoring_coverage,
        minimum_scoring_ramps=minimum_scoring_ramps,
        aliases=tuple(aliases),
    )


# Backward-compatible private spelling useful to small integration tests.
_load_config = load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, date)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = json.dumps(
        _json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, raw)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if path.name.lower().endswith(".gz"):
        raw = gzip.compress(raw, compresslevel=9, mtime=0)
    _atomic_bytes(path, raw)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename deliberately short: pytest and Codex
    # worktrees can already consume most of Windows' legacy MAX_PATH budget.
    fd, temporary_name = tempfile.mkstemp(
        prefix=".pq-", suffix=".parquet", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        pd.read_parquet(temporary).head(1)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sync_command(config: RunnerConfig) -> str:
    forecast = config.project_root / "Forecast.ps1"
    return (
        f"& '{forecast}' -Action ResidualCompare "
        "-ResidualComparisonStage SyncObserved -Countries FR,DE,BE,NL,ES"
    )


def plan(config: RunnerConfig) -> dict[str, Any]:
    """Return a read-only plan; Parquet contents are deliberately not opened."""

    days = (config.evaluation_end_day - config.evaluation_start_day).days + 1
    refits = math.ceil(days / config.refit_every_days)
    sources = []
    missing_forecasts: list[str] = []
    missing_observed: list[str] = []
    for item in config.aliases:
        forecast_exists = item.forecast_path.is_file()
        observed_exists = item.observed_path.is_file()
        if not forecast_exists:
            missing_forecasts.append(item.alias)
        if not observed_exists:
            missing_observed.append(item.alias)
        sources.append(
            {
                "alias": item.alias,
                "forecast_path": item.forecast_path,
                "forecast_exists": forecast_exists,
                "observed_path": item.observed_path,
                "observed_exists": observed_exists,
            }
        )
    return {
        "schema_version": 1,
        "stage": "plan",
        "experiment_id": config.experiment_id,
        "output_root": config.output_root,
        "production_changed": False,
        "writes_data_pit": False,
        "writes_runs_live": False,
        "protocol": {
            "forecast_origin": "D-1 08:00 Europe/Paris",
            "latest_training_label": "D-2 final physical hour",
            "forecast_fill_limit_hours": config.forecast_fill_limit_hours,
            "forecast_minimum_coverage": config.forecast_minimum_coverage,
            "evaluation_start_day": config.evaluation_start_day,
            "evaluation_end_day": config.evaluation_end_day,
            "refit_every_days": config.refit_every_days,
            "cold_start_policy": config.cold_start_policy,
            "minimum_training_rows": config.minimum_training_rows,
        },
        "sources": sources,
        "missing_forecasts": missing_forecasts,
        "missing_observed_vintages": missing_observed,
        "observed_sync_required": bool(missing_observed),
        "observed_sync_command": _sync_command(config) if missing_observed else None,
        "estimated_work": {
            "delivery_days": days,
            "refits": refits,
            "country_component_fits": refits * len(RESIDUAL_LOAD_ALIASES) * 2,
        },
    }


build_plan = plan


def _normalize_vintages(frame: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    required = {"value_time_utc", "snapshot_time_utc", "revision_time_utc", "value"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ResidualLoadInputRunnerError(f"{path}: colonnes PIT absentes: {missing}.")
    result = frame.copy().reset_index(drop=True)
    if "downloaded_at_utc" not in result:
        result["downloaded_at_utc"] = pd.NaT
    for column in (
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "downloaded_at_utc",
    ):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    if result[["value_time_utc", "snapshot_time_utc", "revision_time_utc"]].isna().any().any():
        raise ResidualLoadInputRunnerError(f"{path}: timestamps PIT invalides.")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    if bool(np.isinf(result["value"].to_numpy(dtype=float)).any()):
        raise ResidualLoadInputRunnerError(f"{path}: valeurs infinies.")
    result["_row_order"] = np.arange(len(result), dtype=np.int64)
    return result


def _origin_by_value_time(index: pd.DatetimeIndex, *, timezone: str) -> pd.Series:
    local_days = index.tz_convert(timezone).date
    mapping = {
        day: scheduled_live_origin_utc(day, timezone=timezone)
        for day in dict.fromkeys(local_days)
    }
    return pd.Series([mapping[day] for day in local_days], index=index, dtype="datetime64[ns, UTC]")


def _full_index(config: RunnerConfig) -> pd.DatetimeIndex:
    start = local_delivery_day_index(config.history_start_day, timezone=config.timezone)[0]
    end = local_delivery_day_index(config.evaluation_end_day, timezone=config.timezone)[-1]
    return pd.date_range(start, end, freq="h", tz="UTC", name="value_time_utc")


def _select_forecast_asof(
    store: pd.DataFrame,
    *,
    alias: str,
    grid: pd.DatetimeIndex,
    timezone: str,
    fill_limit: int,
    minimum_coverage: float = 1.0,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    frame = store.copy()
    row_origins = _origin_by_value_time(pd.DatetimeIndex(frame["value_time_utc"]), timezone=timezone)
    frame["cutoff_utc"] = row_origins.to_numpy()
    eligible = frame.loc[
        frame["snapshot_time_utc"].le(frame["cutoff_utc"])
        & frame["revision_time_utc"].le(frame["cutoff_utc"])
    ]
    selected = (
        eligible.sort_values(
            ["value_time_utc", "revision_time_utc", "snapshot_time_utc", "downloaded_at_utc", "_row_order"],
            kind="stable",
            na_position="first",
        )
        .drop_duplicates("value_time_utc", keep="last")
        .set_index("value_time_utc")
        .sort_index()
    )
    envelope = selected.index.union(grid).sort_values()
    columns = ["value", "snapshot_time_utc", "revision_time_utc"]
    aligned = selected[columns].reindex(envelope)
    source_time = pd.Series(selected.index, index=selected.index).reindex(envelope)
    if fill_limit:
        aligned = aligned.ffill(limit=fill_limit)
        source_time = source_time.ffill(limit=fill_limit)
    aligned = aligned.reindex(grid)
    source_time = source_time.reindex(grid)
    missing = aligned["value"].isna()
    coverage = float((~missing).mean())
    if coverage < minimum_coverage:
        examples = [str(value) for value in grid[missing.to_numpy()][:8]]
        raise ResidualLoadInputRunnerError(
            f"{alias}: couverture PIT {coverage:.3%} < {minimum_coverage:.3%}; "
            f"{int(missing.sum())} heures absentes apres ffill causal "
            f"limite={fill_limit}; exemples={examples}."
        )
    origins = _origin_by_value_time(grid, timezone=timezone)
    violations = (
        aligned["snapshot_time_utc"].gt(origins)
        | aligned["revision_time_utc"].gt(origins)
    )
    if bool(violations.any()):
        raise AssertionError(f"{alias}: une vintage future a traverse la selection.")
    exact = grid.isin(selected.index) & selected.reindex(grid)["value"].notna().to_numpy()
    filled = source_time.notna().to_numpy() & (
        pd.DatetimeIndex(source_time).to_numpy() != grid.to_numpy()
    )
    audit = pd.DataFrame(
        {
            "value_time_utc": grid,
            "snapshot_time_utc": aligned["snapshot_time_utc"].to_numpy(),
            "revision_time_utc": aligned["revision_time_utc"].to_numpy(),
            "source_value_time_utc": source_time.to_numpy(),
            "forecast_origin_utc": origins.to_numpy(),
            "filled": filled,
            "missing": missing.to_numpy(),
        }
    )
    values = aligned["value"].astype(float).rename(alias)
    values.index = grid
    return values, audit, {
        "raw_rows": len(frame),
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "materialized_rows": len(values),
        "coverage": coverage,
        "missing_rows": int(missing.sum()),
        "filled_rows": int(filled.sum()),
        "maximum_selected_snapshot_time_utc": aligned["snapshot_time_utc"].max(),
        "maximum_selected_revision_time_utc": aligned["revision_time_utc"].max(),
        "cutoff_violations": 0,
    }


def _select_observations_asof(
    stores: Mapping[str, pd.DataFrame],
    *,
    origin_utc: pd.Timestamp,
    label_end_utc: pd.Timestamp,
    grid: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    training_grid = grid[grid <= label_end_utc]
    values: dict[str, pd.Series] = {}
    audits: list[dict[str, Any]] = []
    for alias in RESIDUAL_LOAD_ALIASES:
        frame = stores[alias]
        eligible = frame.loc[
            frame["snapshot_time_utc"].le(origin_utc)
            & frame["revision_time_utc"].le(origin_utc)
            & frame["value_time_utc"].le(label_end_utc)
        ]
        selected = (
            eligible.sort_values(
                ["value_time_utc", "revision_time_utc", "snapshot_time_utc", "downloaded_at_utc", "_row_order"],
                kind="stable",
                na_position="first",
            )
            .drop_duplicates("value_time_utc", keep="last")
        )
        series = selected.set_index("value_time_utc")["value"].reindex(training_grid)
        values[alias] = series
        max_snapshot = selected["snapshot_time_utc"].max() if not selected.empty else pd.NaT
        max_revision = selected["revision_time_utc"].max() if not selected.empty else pd.NaT
        if (pd.notna(max_snapshot) and max_snapshot > origin_utc) or (
            pd.notna(max_revision) and max_revision > origin_utc
        ):
            raise AssertionError(f"{alias}: observation post-origine selectionnee.")
        audits.append(
            {
                "alias": alias,
                "origin_utc": origin_utc,
                "label_end_utc": label_end_utc,
                "selected_rows": len(selected),
                "available_labels": int(series.notna().sum()),
                "maximum_selected_snapshot_time_utc": max_snapshot,
                "maximum_selected_revision_time_utc": max_revision,
                "cutoff_violations": 0,
            }
        )
    observations = pd.DataFrame(values, index=training_grid)
    observations.index.name = "value_time_utc"
    return observations, audits


def _common_scoring_origin(stores: Mapping[str, pd.DataFrame]) -> pd.Timestamp:
    maxima: list[pd.Timestamp] = []
    for frame in stores.values():
        eligible_at = pd.concat(
            [frame["snapshot_time_utc"], frame["revision_time_utc"]], axis=1
        ).max(axis=1)
        if eligible_at.empty or pd.isna(eligible_at.max()):
            raise ResidualLoadInputRunnerError("Store observe vide ou sans vintage valide.")
        maxima.append(pd.Timestamp(eligible_at.max()).tz_convert("UTC"))
    return min(maxima)


def _publish_staged_directory(staging: Path, final: Path, *, overwrite: bool) -> None:
    if final.exists() and not overwrite:
        raise FileExistsError(f"Sortie existante; utilisez --overwrite: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    try:
        if final.exists():
            backup = final.with_name(f".{final.name}.{uuid.uuid4().hex}.backup")
            os.replace(final, backup)
        os.replace(staging, final)
    except Exception:
        if backup is not None and backup.exists() and not final.exists():
            os.replace(backup, final)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup)


def _checksum_manifest(root: Path, paths: Sequence[Path]) -> dict[str, Any]:
    artifacts = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {"algorithm": "sha256", "artifacts": artifacts}


def build(
    config: RunnerConfig,
    *,
    threads: int = -1,
    overwrite: bool = False,
    correction_generator: Callable[..., Any] | None = None,
) -> BuildResult:
    """Build all five corrections in a staging tree, then publish atomically."""

    if threads == 0 or threads < -1:
        raise ValueError("--threads doit valoir -1 ou un entier positif.")
    current_plan = plan(config)
    if current_plan["missing_forecasts"]:
        raise FileNotFoundError(
            "Forecasts PIT bruts absents: " + ", ".join(current_plan["missing_forecasts"])
        )
    if current_plan["missing_observed_vintages"]:
        raise FileNotFoundError(
            "Vintages observees locales absentes: "
            + ", ".join(current_plan["missing_observed_vintages"])
            + ". Lancez d'abord: "
            + _sync_command(config)
        )
    if config.output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Sortie experimentale existante: {config.output_root}; utilisez --overwrite."
        )

    grid = _full_index(config)
    forecast_columns: dict[str, pd.Series] = {}
    forecast_audits: dict[str, pd.DataFrame] = {}
    forecast_meta: dict[str, Any] = {}
    observed_stores: dict[str, pd.DataFrame] = {}
    source_manifest: dict[str, Any] = {}
    for item in config.aliases:
        forecast_store = _normalize_vintages(pd.read_parquet(item.forecast_path), path=item.forecast_path)
        observed_store = _normalize_vintages(pd.read_parquet(item.observed_path), path=item.observed_path)
        values, audit, metadata = _select_forecast_asof(
            forecast_store,
            alias=item.alias,
            grid=grid,
            timezone=config.timezone,
            fill_limit=config.forecast_fill_limit_hours,
            minimum_coverage=config.forecast_minimum_coverage,
        )
        forecast_columns[item.alias] = values
        forecast_audits[item.alias] = audit
        forecast_meta[item.alias] = metadata
        observed_stores[item.alias] = observed_store
        source_manifest[item.alias] = {
            "forecast_path": item.forecast_path,
            "forecast_sha256": _sha256(item.forecast_path),
            "observed_path": item.observed_path,
            "observed_sha256": _sha256(item.observed_path),
            "observed_series": item.observed_series,
        }
    forecasts = pd.DataFrame(forecast_columns, index=grid)

    generator = correction_generator or generate_prequential_residual_load_corrections
    day_count = (config.evaluation_end_day - config.evaluation_start_day).days + 1
    results: list[Any] = []
    observation_audits: list[dict[str, Any]] = []
    refit_rows: list[dict[str, Any]] = []
    current = config.evaluation_start_day
    while current <= config.evaluation_end_day:
        group_end = min(
            current + timedelta(days=config.refit_every_days - 1),
            config.evaluation_end_day,
        )
        origin = scheduled_live_origin_utc(current, timezone=config.timezone)
        label_end = conservative_label_end_utc(current, timezone=config.timezone)
        observations, audits = _select_observations_asof(
            observed_stores,
            origin_utc=origin,
            label_end_utc=label_end,
            grid=grid,
        )
        for item in audits:
            observation_audits.append(
                {**item, "refit_delivery_day_local": current, "group_end_day_local": group_end}
            )
        LOGGER.info("Refit %s -> %s (origine %s)", current, group_end, origin)
        result = generator(
            forecasts,
            observations,
            start_day=current,
            end_day=group_end,
            max_abs_correction_gw=config.max_abs_correction_gw,
            cold_start_policy=config.cold_start_policy,
            minimum_training_rows=config.minimum_training_rows,
            thread_count=threads,
            timezone=config.timezone,
            forecast_change_lags_hours=config.forecast_change_lags_hours,
            error_lags_hours=config.error_lags_hours,
            error_rolling_windows_hours=config.error_rolling_windows_hours,
            refit_every_days=config.refit_every_days,
        )
        results.append(result)
        block_status_by_day: dict[date, dict[str, str]] = {}
        result_diagnostics = getattr(result, "diagnostics", {})
        for block in result_diagnostics.get("blocks", ()):
            block_day = _day(
                block.get("prediction_delivery_day_local"),
                name="prediction_delivery_day_local",
            )
            model_diagnostics = _mapping(
                block.get("models", {}),
                name="prequential_block.models",
            )
            block_status_by_day[block_day] = {
                alias: str(
                    _mapping(
                        model_diagnostics.get(alias, {}),
                        name=f"prequential_block.models.{alias}",
                    ).get("status", "unknown")
                )
                for alias in RESIDUAL_LOAD_ALIASES
            }
        group_index = pd.date_range(
            local_delivery_day_index(current, timezone=config.timezone)[0],
            local_delivery_day_index(group_end, timezone=config.timezone)[-1],
            freq="h",
            tz="UTC",
        )
        for timestamp in group_index:
            day = timestamp.tz_convert(config.timezone).date()
            row = {
                "value_time_utc": timestamp,
                "delivery_day_local": day,
                "forecast_origin_utc": scheduled_live_origin_utc(
                    day,
                    timezone=config.timezone,
                ),
                "refit_delivery_day_local": current,
                "refit_origin_utc": origin,
                "reused_model": day != current,
            }
            statuses = block_status_by_day.get(day, {})
            for alias in RESIDUAL_LOAD_ALIASES:
                row[f"{alias}__corrector_status"] = statuses.get(
                    alias,
                    "unknown",
                )
            refit_rows.append(row)
        current = group_end + timedelta(days=1)

    raw = pd.concat([item.raw for item in results]).sort_index()
    correction = pd.concat([item.correction for item in results]).sort_index()
    corrected = pd.concat([item.corrected for item in results]).sort_index()
    expected_index = pd.date_range(
        local_delivery_day_index(config.evaluation_start_day, timezone=config.timezone)[0],
        local_delivery_day_index(config.evaluation_end_day, timezone=config.timezone)[-1],
        freq="h",
        tz="UTC",
    )
    for name, frame in (("raw", raw), ("correction", correction), ("corrected", corrected)):
        if not frame.index.equals(expected_index) or tuple(frame.columns) != tuple(RESIDUAL_LOAD_ALIASES):
            raise ResidualLoadInputRunnerError(f"Sortie {name} incomplete ou schema divergent.")
        if bool(np.isinf(frame.to_numpy(dtype=float)).any()):
            raise ResidualLoadInputRunnerError(f"Sortie {name} contient des infinis.")
    raw_missing = raw.isna()
    if not correction.isna().equals(raw_missing) or not corrected.isna().equals(
        raw_missing
    ):
        raise ResidualLoadInputRunnerError(
            "correction/corrected doivent preserver exactement le masque NaN brut."
        )

    scoring_origin = _common_scoring_origin(observed_stores)
    scoring, scoring_audits = _select_observations_asof(
        observed_stores,
        origin_utc=scoring_origin,
        label_end_utc=expected_index[-1],
        grid=grid,
    )
    scoring = scoring.reindex(expected_index)
    refit_frame = pd.DataFrame(refit_rows).set_index("value_time_utc").reindex(expected_index)
    audit = pd.DataFrame(index=expected_index)
    audit.index.name = "value_time_utc"
    for alias in RESIDUAL_LOAD_ALIASES:
        audit[f"{alias}__raw"] = raw[alias]
        audit[f"{alias}__correction"] = correction[alias]
        audit[f"{alias}__corrected"] = corrected[alias]
        audit[f"{alias}__observed"] = scoring[alias]
        source = forecast_audits[alias].set_index("value_time_utc").reindex(expected_index)
        audit[f"{alias}__raw_snapshot_time_utc"] = source["snapshot_time_utc"]
        audit[f"{alias}__raw_revision_time_utc"] = source["revision_time_utc"]
        audit[f"{alias}__raw_filled"] = source["filled"].astype(bool)
        audit[f"{alias}__raw_missing"] = source["missing"].astype(bool)
        audit[f"{alias}__corrector_status"] = refit_frame[
            f"{alias}__corrector_status"
        ]
    audit["forecast_origin_utc"] = refit_frame["forecast_origin_utc"]
    audit["refit_origin_utc"] = refit_frame["refit_origin_utc"]
    audit["reused_model"] = refit_frame["reused_model"].astype(bool)

    experiments_root = (config.project_root / "runs" / "experiments").resolve()
    experiments_root.mkdir(parents=True, exist_ok=True)
    staging = experiments_root / f".{config.experiment_id}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        corrected_paths: dict[str, Path] = {}
        origins = _origin_by_value_time(expected_index, timezone=config.timezone)
        for alias in RESIDUAL_LOAD_ALIASES:
            pit = pd.DataFrame(
                {
                    "value_time_utc": expected_index,
                    "snapshot_time_utc": origins.to_numpy(),
                    "revision_time_utc": origins.to_numpy(),
                    "value": corrected[alias].to_numpy(dtype=float),
                }
            )
            path = staging / config.corrected_directory / f"{alias}.parquet"
            _atomic_parquet(pit, path)
            corrected_paths[alias] = path
        audit_path = staging / config.audit_directory / "prequential_corrections.parquet"
        _atomic_parquet(audit.reset_index(), audit_path)
        _atomic_csv(
            audit.reset_index(),
            staging / config.audit_directory / "prequential_corrections.csv.gz",
        )
        _atomic_csv(
            pd.DataFrame(observation_audits),
            staging / config.audit_directory / "refit_observation_selection.csv",
        )
        _atomic_csv(
            pd.DataFrame(scoring_audits),
            staging / config.audit_directory / "scoring_observation_selection.csv",
        )
        manifest_path = staging / "build_manifest.json"
        implementation = config.project_root / "chronos2_hourly" / "residual_load_input_corrector.py"
        _atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "complete",
                "run_type": "offline_residual_load_input_correction",
                "experiment_id": config.experiment_id,
                "production_changed": False,
                "writes_data_pit": False,
                "writes_runs_live": False,
                "config": {"path": config.source_path, "sha256": _sha256(config.source_path)},
                "implementation_sha256": {
                    "runner": _sha256(Path(__file__).resolve()),
                    "corrector": _sha256(implementation),
                },
                "protocol": {
                    "forecast_origin": "D-1 08:00 Europe/Paris",
                    "latest_training_label": "D-2 final physical hour",
                    "input_contract": "caller_materialized_asof",
                    "history_start_day": config.history_start_day,
                    "evaluation_start_day": config.evaluation_start_day,
                    "evaluation_end_day": config.evaluation_end_day,
                    "forecast_fill_limit_hours": config.forecast_fill_limit_hours,
                    "forecast_minimum_coverage": config.forecast_minimum_coverage,
                    "refit_every_days": config.refit_every_days,
                    "refit_count": math.ceil(day_count / config.refit_every_days),
                    "cold_start_policy": config.cold_start_policy,
                    "minimum_training_rows": config.minimum_training_rows,
                    "recipe": "blend_cat_hgb_w0.50",
                    "minimum_scoring_coverage": config.minimum_scoring_coverage,
                    "minimum_scoring_ramps": config.minimum_scoring_ramps,
                },
                "sources": source_manifest,
                "forecast_selection": forecast_meta,
                "cold_start_passthrough_hours": {
                    alias: int(
                        audit[f"{alias}__corrector_status"]
                        .eq("cold_start_raw_passthrough")
                        .sum()
                    )
                    for alias in RESIDUAL_LOAD_ALIASES
                },
                "observation_selection": {
                    "per_refit_audit": f"{config.audit_directory}/refit_observation_selection.csv",
                    "scoring_as_of_utc": scoring_origin,
                    "scoring_only_not_used_for_fit": True,
                },
                "artifacts": {
                    alias: f"{config.corrected_directory}/{alias}.parquet"
                    for alias in RESIDUAL_LOAD_ALIASES
                },
            },
        )
        artifact_paths = [path for path in staging.rglob("*") if path.is_file()]
        checksum_path = staging / "artifact_checksums.json"
        _atomic_json(checksum_path, _checksum_manifest(staging, artifact_paths))
        _publish_staged_directory(staging, config.output_root, overwrite=overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    final_corrected = {
        alias: config.output_root / config.corrected_directory / f"{alias}.parquet"
        for alias in RESIDUAL_LOAD_ALIASES
    }
    return BuildResult(
        output_root=config.output_root,
        manifest_path=config.output_root / "build_manifest.json",
        checksum_manifest_path=config.output_root / "artifact_checksums.json",
        corrected_paths=final_corrected,
        audit_path=config.output_root / config.audit_directory / "prequential_corrections.parquet",
    )


def _metrics_for_alias(
    frame: pd.DataFrame,
    *,
    alias: str,
    scope: str,
    timezone: str,
    regime: str = "all",
    selection_mask: pd.Series | None = None,
    minimum_coverage: float = 0.0,
    minimum_ramps: int = 1,
) -> dict[str, Any]:
    raw = pd.to_numeric(frame[f"{alias}__raw"], errors="coerce")
    corrected = pd.to_numeric(frame[f"{alias}__corrected"], errors="coerce")
    observed = pd.to_numeric(frame[f"{alias}__observed"], errors="coerce")
    valid = raw.notna() & corrected.notna() & observed.notna()
    local_day = pd.Series(frame.index.tz_convert(timezone).date, index=frame.index)
    raw_ramp = raw.groupby(local_day).diff()
    corrected_ramp = corrected.groupby(local_day).diff()
    observed_ramp = observed.groupby(local_day).diff()
    ramp_valid = raw_ramp.notna() & corrected_ramp.notna() & observed_ramp.notna()
    eligible_count = len(frame)
    if selection_mask is not None:
        aligned_mask = selection_mask.reindex(frame.index).fillna(False).astype(bool)
        eligible_count = int(aligned_mask.sum())
        valid &= aligned_mask
        ramp_valid &= aligned_mask
    coverage = float(valid.sum() / eligible_count) if eligible_count else 0.0
    if not bool(valid.any()) or coverage < minimum_coverage:
        raise ResidualLoadInputRunnerError(
            f"{alias}/{scope}/{regime}: couverture scorée {coverage:.3%} "
            f"< {minimum_coverage:.3%}."
        )
    if int(ramp_valid.sum()) < minimum_ramps:
        raise ResidualLoadInputRunnerError(
            f"{alias}/{scope}/{regime}: {int(ramp_valid.sum())} rampes scorables "
            f"< {minimum_ramps}."
        )
    raw_mae = float((raw[valid] - observed[valid]).abs().mean())
    corrected_mae = float((corrected[valid] - observed[valid]).abs().mean())
    raw_ramp_mae = float((raw_ramp[ramp_valid] - observed_ramp[ramp_valid]).abs().mean())
    corrected_ramp_mae = float(
        (corrected_ramp[ramp_valid] - observed_ramp[ramp_valid]).abs().mean()
    )
    return {
        "scope": scope,
        "regime": regime,
        "alias": alias,
        "country": alias[:2].upper(),
        "n_hours": int(valid.sum()),
        "coverage": coverage,
        "n_ramps": int(ramp_valid.sum()),
        "raw_mae_gw": raw_mae,
        "corrected_mae_gw": corrected_mae,
        "mae_gain_gw": raw_mae - corrected_mae,
        "raw_bias_gw": float((raw[valid] - observed[valid]).mean()),
        "corrected_bias_gw": float((corrected[valid] - observed[valid]).mean()),
        "raw_ramp_mae_gw": raw_ramp_mae,
        "corrected_ramp_mae_gw": corrected_ramp_mae,
        "ramp_mae_gain_gw": raw_ramp_mae - corrected_ramp_mae,
    }


def _large_observed_ramp_mask(
    frame: pd.DataFrame,
    *,
    alias: str,
    timezone: str,
) -> tuple[pd.Series, float] | None:
    """Select the top observed within-day ramps for regime-shift scoring."""

    observed = pd.to_numeric(frame[f"{alias}__observed"], errors="coerce")
    local_day = pd.Series(frame.index.tz_convert(timezone).date, index=frame.index)
    absolute_ramp = observed.groupby(local_day).diff().abs()
    finite = absolute_ramp.dropna()
    if finite.empty:
        return None
    threshold = float(finite.quantile(0.90))
    return absolute_ramp.ge(threshold) & absolute_ramp.notna(), threshold


def _verify_checksums(root: Path) -> None:
    path = root / "artifact_checksums.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("artifacts", []):
        artifact = (root / str(item["path"])).resolve()
        try:
            artifact.relative_to(root.resolve())
        except ValueError as exc:
            raise ResidualLoadInputRunnerError("Chemin de checksum non confine.") from exc
        if not artifact.is_file() or _sha256(artifact) != item.get("sha256"):
            raise ResidualLoadInputRunnerError(f"Checksum invalide: {artifact}")


def _verify_build_seal(config: RunnerConfig) -> Mapping[str, Any]:
    """Bind reporting to the exact config and implementation used at build."""

    manifest_path = config.output_root / "build_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        name="build_manifest",
    )
    if payload.get("status") != "complete" or payload.get("experiment_id") != (
        config.experiment_id
    ):
        raise ResidualLoadInputRunnerError("Build manifest incomplet ou incompatible.")
    recorded_config = _mapping(payload.get("config"), name="build_manifest.config")
    if recorded_config.get("sha256") != _sha256(config.source_path):
        raise ResidualLoadInputRunnerError(
            "La configuration a change depuis le build; reconstruisez avant report."
        )
    implementation = _mapping(
        payload.get("implementation_sha256"),
        name="build_manifest.implementation_sha256",
    )
    expected_implementation = {
        "runner": _sha256(Path(__file__).resolve()),
        "corrector": _sha256(
            config.project_root
            / "chronos2_hourly"
            / "residual_load_input_corrector.py"
        ),
    }
    if any(
        implementation.get(name) != digest
        for name, digest in expected_implementation.items()
    ):
        raise ResidualLoadInputRunnerError(
            "L'implementation a change depuis le build; reconstruisez avant report."
        )
    return payload


def report(config: RunnerConfig, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Publish JSON/CSV MAE, bias and ramp-MAE comparisons by alias."""

    if not config.output_root.is_dir():
        raise FileNotFoundError(
            f"Build absent: {config.output_root}. Lancez d'abord --stage build."
        )
    _verify_checksums(config.output_root)
    _verify_build_seal(config)
    audit_path = config.output_root / config.audit_directory / "prequential_corrections.parquet"
    audit = pd.read_parquet(audit_path)
    index = pd.to_datetime(audit.pop("value_time_utc"), utc=True)
    audit.index = pd.DatetimeIndex(index)
    scopes = [("evaluation", config.evaluation_start_day, config.evaluation_end_day)]
    if (config.final_start_day, config.final_end_day) != (
        config.evaluation_start_day,
        config.evaluation_end_day,
    ):
        scopes.append(("final", config.final_start_day, config.final_end_day))
    rows: list[dict[str, Any]] = []
    for scope, start_day, end_day in scopes:
        start = local_delivery_day_index(start_day, timezone=config.timezone)[0]
        end = local_delivery_day_index(end_day, timezone=config.timezone)[-1]
        part = audit.loc[(audit.index >= start) & (audit.index <= end)]
        for alias in RESIDUAL_LOAD_ALIASES:
            rows.append(
                _metrics_for_alias(
                    part,
                    alias=alias,
                    scope=scope,
                    timezone=config.timezone,
                    minimum_coverage=config.minimum_scoring_coverage,
                    minimum_ramps=config.minimum_scoring_ramps,
                )
            )
            large_ramps = _large_observed_ramp_mask(
                part,
                alias=alias,
                timezone=config.timezone,
            )
            if large_ramps is not None:
                mask, threshold = large_ramps
                regime_metrics = _metrics_for_alias(
                    part,
                    alias=alias,
                    scope=scope,
                    timezone=config.timezone,
                    regime="observed_ramp_top10pct",
                    selection_mask=mask,
                    minimum_coverage=config.minimum_scoring_coverage,
                    minimum_ramps=max(1, math.ceil(config.minimum_scoring_ramps / 10)),
                )
                regime_metrics["observed_ramp_threshold_gw"] = threshold
                rows.append(regime_metrics)
            status_column = f"{alias}__corrector_status"
            if status_column in part:
                for status in ("fitted", "cold_start_raw_passthrough"):
                    status_mask = part[status_column].eq(status)
                    if not bool(status_mask.any()):
                        continue
                    rows.append(
                        _metrics_for_alias(
                            part,
                            alias=alias,
                            scope=scope,
                            timezone=config.timezone,
                            regime=status,
                            selection_mask=status_mask,
                            minimum_coverage=config.minimum_scoring_coverage,
                            minimum_ramps=max(
                                1,
                                math.ceil(config.minimum_scoring_ramps / 10),
                            ),
                        )
                    )
    metrics = pd.DataFrame(rows)
    report_root = config.output_root / config.report_directory
    if report_root.exists() and not overwrite:
        raise FileExistsError(f"Rapport existant; utilisez --overwrite: {report_root}")
    staging = config.output_root / f".{Path(config.report_directory).name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        csv_path = staging / "metrics_by_alias.csv"
        json_path = staging / "report.json"
        _atomic_csv(metrics, csv_path)
        _atomic_json(
            json_path,
            {
                "schema_version": 1,
                "status": "complete",
                "experiment_id": config.experiment_id,
                "definitions": {
                    "bias": "forecast_minus_observed_gw",
                    "ramp": "physical_hour_difference_within_local_delivery_day",
                    "observed_ramp_top10pct": (
                        "hours whose absolute observed within-day ramp is at or "
                        "above the alias/scope 90th percentile"
                    ),
                    "cold_start_raw_passthrough": (
                        "raw forecast left unchanged until the alias reaches "
                        "the causal minimum_training_rows threshold"
                    ),
                    "scoring_observations": "latest common local vintage; never used for fit",
                },
                "metrics": metrics.to_dict(orient="records"),
            },
        )
        _atomic_json(
            staging / "report_manifest.json",
            {
                "status": "complete",
                "source_build_manifest_sha256": _sha256(config.output_root / "build_manifest.json"),
                "artifacts": {
                    "metrics_by_alias.csv": _sha256(csv_path),
                    "report.json": _sha256(json_path),
                },
            },
        )
        _publish_staged_directory(staging, report_root, overwrite=overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    all_artifacts = [
        path
        for path in config.output_root.rglob("*")
        if path.is_file() and path.name != "artifact_checksums.json"
    ]
    _atomic_json(
        config.output_root / "artifact_checksums.json",
        _checksum_manifest(config.output_root, all_artifacts),
    )
    return report_root / "report.json", report_root / "metrics_by_alias.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correcteur causal offline des cinq inputs residual_load_fcst."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", choices=STAGES, default="plan")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        config = load_config(args.config)
        plan_payload = plan(config)
        print(json.dumps(_json_safe(plan_payload), ensure_ascii=False, indent=2))
        if args.stage == "plan":
            return 0
        if args.stage in {"build", "all"}:
            result = build(
                config,
                threads=args.threads,
                overwrite=bool(args.overwrite),
            )
            print(f"Build: {result.manifest_path}")
        if args.stage in {"report", "all"}:
            report_json, report_csv = report(
                config,
                overwrite=bool(args.overwrite),
            )
            print(f"Rapport JSON: {report_json}")
            print(f"Rapport CSV: {report_csv}")
        return 0
    except Exception as exc:
        LOGGER.exception("Echec du challenger residual_load input: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
