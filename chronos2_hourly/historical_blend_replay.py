"""Reapply a sealed MKOnline blend to a recalculated autonomous branch.

The MKOnline primary series and its already-selected convex weight are held
fixed.  Only the autonomous Chronos price branch is replaced, which makes the
result suitable for the residual-load source A/B comparison without learning
anything from the sealed final year.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd

from .reporting import write_hourly_html_report


QUANTILES = ("q10", "q50", "q90")
FINAL_HOURS = 8760


class HistoricalBlendReplayError(RuntimeError):
    """Raised when the frozen blend cannot be replayed exactly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalBlendReplayError(f"JSON illisible: {path}") from exc
    if not isinstance(value, dict):
        raise HistoricalBlendReplayError(f"{path}: objet JSON attendu.")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _read_timestamped(path: Path, timestamp_column: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if timestamp_column not in frame:
        raise HistoricalBlendReplayError(
            f"{path}: colonne {timestamp_column!r} absente."
        )
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop(timestamp_column), utc=True, errors="raise"),
        name=timestamp_column,
    )
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise HistoricalBlendReplayError(f"{path}: timeline invalide.")
    return frame


def _weights(reference_run: Path) -> tuple[float, float, dict[str, Any]]:
    metrics = _read_json(reference_run / "metrics_hourly.json")
    diagnostics = metrics.get("training_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    benchmark = diagnostics.get("zone_benchmark", {})
    if not isinstance(benchmark, Mapping):
        benchmark = {}
    raw = benchmark.get("weights", {})
    if not isinstance(raw, Mapping):
        raw = {}
    manifest = _read_json(reference_run / "run_manifest.json")
    autonomous = raw.get("autonomous", manifest.get("autonomous_weight"))
    primary = raw.get("mkonline_primary", manifest.get("mkonline_weight"))
    if autonomous is None and primary is not None:
        autonomous = 1.0 - float(primary)
    if primary is None and autonomous is not None:
        primary = 1.0 - float(autonomous)
    if autonomous is None or primary is None:
        raise HistoricalBlendReplayError(
            f"{reference_run}: poids MKOnline figes absents."
        )
    autonomous_value = float(autonomous)
    primary_value = float(primary)
    if (
        autonomous_value <= 0.0
        or primary_value <= 0.0
        or not np.isclose(
            autonomous_value + primary_value, 1.0, rtol=0.0, atol=1e-12
        )
    ):
        raise HistoricalBlendReplayError("Poids MKOnline invalides.")
    return autonomous_value, primary_value, manifest


def _quantiles(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [f"{prefix}__{quantile}" for quantile in QUANTILES]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise HistoricalBlendReplayError(f"Quantiles absents: {missing}.")
    result = frame.loc[:, columns].rename(
        columns={f"{prefix}__{quantile}": quantile for quantile in QUANTILES}
    )
    result = result.apply(pd.to_numeric, errors="coerce")
    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise HistoricalBlendReplayError("Quantiles autonomes non finis.")
    if not ((values[:, 0] <= values[:, 1]) & (values[:, 1] <= values[:, 2])).all():
        raise HistoricalBlendReplayError("Quantiles autonomes croises.")
    return result


def _blend(
    autonomous: pd.DataFrame,
    primary: pd.Series,
    *,
    autonomous_weight: float,
    primary_weight: float,
) -> pd.DataFrame:
    primary = pd.to_numeric(primary, errors="coerce").reindex(autonomous.index)
    if not np.isfinite(primary.to_numpy(dtype=float)).all():
        raise HistoricalBlendReplayError("Primaire MKOnline incomplet.")
    q50 = autonomous_weight * autonomous["q50"] + primary_weight * primary
    shift = q50 - autonomous["q50"]
    result = pd.DataFrame(index=autonomous.index)
    for quantile in QUANTILES:
        result[quantile] = autonomous[quantile] + shift
    result["shift"] = shift
    result["primary"] = primary
    return result


def _metric_rows(backtest: pd.DataFrame, evaluation_index: pd.DatetimeIndex) -> list[dict[str, Any]]:
    actual = pd.to_numeric(backtest.loc[evaluation_index, "actual"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for model in ("residual_corrected", "mkonline_primary", "mkonline_blend"):
        column = f"{model}__q50"
        prediction = pd.to_numeric(
            backtest.loc[evaluation_index, column], errors="coerce"
        )
        scored = actual.notna() & prediction.notna()
        rows.append(
            {
                "model": model,
                "mae": float((prediction.loc[scored] - actual.loc[scored]).abs().mean()),
                "n_scored": int(scored.sum()),
                "n_expected": int(len(evaluation_index)),
                "prediction_coverage": float(prediction.notna().mean()),
                "score_coverage": float(scored.mean()),
            }
        )
    return rows


def _write_checksums(
    directory: Path,
    *,
    source_paths: Mapping[str, Path],
    published_directory: Path | None = None,
) -> None:
    artifacts: list[dict[str, Any]] = []
    for role, path in sorted(source_paths.items()):
        artifacts.append(
            {
                "path": str(path),
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "artifact_checksums.json":
            artifacts.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "role": "output_artifact",
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        directory / "artifact_checksums.json",
        {
            "algorithm": "sha256",
            "output_directory": str(published_directory or directory),
            "artifacts": artifacts,
        },
    )


def publish_historical_blend_replay(
    autonomous_run: str | Path,
    reference_blend_run: str | Path,
    output_dir: str | Path,
    *,
    residual_load_source: str,
    overwrite: bool = False,
    title: str | None = None,
) -> Path:
    """Publish a recalculated blend run using a frozen primary and weight."""

    autonomous_run = Path(autonomous_run).expanduser().resolve()
    reference_run = Path(reference_blend_run).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output in {autonomous_run, reference_run}:
        raise ValueError("La sortie blend doit differer de ses sources.")
    auto_manifest = _read_json(autonomous_run / "run_manifest.json")
    zone = str(auto_manifest.get("zone", "")).upper()
    timezone = str(auto_manifest.get("timezone", ""))
    if zone not in {"FR", "NL"} or not timezone:
        raise HistoricalBlendReplayError(
            "Le replay MKOnline historique est reserve aux runs FR/NL audites."
        )
    reference_weight_auto, reference_weight_primary, reference_manifest = _weights(
        reference_run
    )
    if str(reference_manifest.get("zone", "")).upper() != zone:
        raise HistoricalBlendReplayError("Zones autonome/MKOnline differentes.")

    auto_backtest = _read_timestamped(
        autonomous_run / "backtest_hourly_oof.csv.gz", "delivery_start_utc"
    )
    reference_backtest = _read_timestamped(
        reference_run / "backtest_hourly_oof.csv.gz", "delivery_start_utc"
    )
    final_mask = auto_backtest["residual_corrected__q50"].notna()
    evaluation_index = auto_backtest.index[final_mask]
    if len(evaluation_index) != FINAL_HOURS:
        raise HistoricalBlendReplayError(
            f"Holdout autonome inattendu: {len(evaluation_index)} != {FINAL_HOURS}."
        )
    if not evaluation_index.isin(reference_backtest.index).all():
        raise HistoricalBlendReplayError("Primaire MKOnline sans couverture FINAL365.")
    actual_auto = pd.to_numeric(
        auto_backtest.loc[evaluation_index, "actual"], errors="coerce"
    )
    actual_reference = pd.to_numeric(
        reference_backtest.loc[evaluation_index, "actual"], errors="coerce"
    )
    if not np.allclose(
        actual_auto.to_numpy(dtype=float),
        actual_reference.to_numpy(dtype=float),
        rtol=0.0,
        atol=2e-5,
    ):
        raise HistoricalBlendReplayError("Cibles autonome/MKOnline differentes.")
    primary_final = pd.to_numeric(
        reference_backtest.loc[evaluation_index, "mkonline_primary__q50"],
        errors="coerce",
    )
    blended_final = _blend(
        _quantiles(auto_backtest.loc[evaluation_index], "residual_corrected"),
        primary_final,
        autonomous_weight=reference_weight_auto,
        primary_weight=reference_weight_primary,
    )

    auto_forecast_path = autonomous_run / f"forecast_hourly_{zone.lower()}.csv"
    reference_forecast_path = reference_run / f"forecast_hourly_{zone.lower()}.csv"
    auto_forecast = _read_timestamped(auto_forecast_path, "delivery_start_utc")
    reference_forecast = _read_timestamped(reference_forecast_path, "delivery_start_utc")
    if not auto_forecast.index.equals(reference_forecast.index):
        raise HistoricalBlendReplayError("Timelines live autonome/MKOnline differentes.")
    primary_live = pd.to_numeric(
        reference_forecast["mkonline_primary__q50"], errors="coerce"
    )
    blended_live = _blend(
        _quantiles(auto_forecast, "residual_corrected"),
        primary_live,
        autonomous_weight=reference_weight_auto,
        primary_weight=reference_weight_primary,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        shutil.copytree(autonomous_run, staging, dirs_exist_ok=True)
        backtest = auto_backtest.copy()
        for quantile in QUANTILES:
            backtest[f"mkonline_blend__{quantile}"] = np.nan
            backtest.loc[evaluation_index, f"mkonline_blend__{quantile}"] = (
                blended_final[quantile].to_numpy(dtype=float)
            )
        backtest["mkonline_primary__q50"] = np.nan
        backtest.loc[evaluation_index, "mkonline_primary__q50"] = (
            primary_final.to_numpy(dtype=float)
        )
        backtest["mkonline_blend_shift"] = np.nan
        backtest.loc[evaluation_index, "mkonline_blend_shift"] = (
            blended_final["shift"].to_numpy(dtype=float)
        )
        backtest.reset_index().to_csv(
            staging / "backtest_hourly_oof.csv.gz",
            index=False,
            compression="gzip",
        )

        forecast = auto_forecast.copy()
        for quantile in QUANTILES:
            forecast[f"mkonline_blend__{quantile}"] = blended_live[
                quantile
            ].to_numpy(dtype=float)
            forecast[quantile] = blended_live[quantile].to_numpy(dtype=float)
        forecast["mkonline_primary__q50"] = primary_live.to_numpy(dtype=float)
        forecast["mkonline_blend_shift"] = blended_live["shift"].to_numpy(
            dtype=float
        )
        forecast["price_eur_mwh"] = forecast["q50"]
        forecast.reset_index().to_csv(
            staging / auto_forecast_path.name, index=False
        )

        metric_rows = _metric_rows(backtest, evaluation_index)
        pd.DataFrame(metric_rows).to_csv(
            staging / "metrics_hourly.csv", index=False
        )
        metrics = _read_json(autonomous_run / "metrics_hourly.json")
        diagnostics = metrics.setdefault("training_diagnostics", {})
        diagnostics["mkonline_blend"] = {
            "enabled": True,
            "weights_retrained": False,
            "autonomous_weight": reference_weight_auto,
            "mkonline_primary_weight": reference_weight_primary,
            "reference_run": str(reference_run),
            "reference_run_manifest_sha256": _sha256(
                reference_run / "run_manifest.json"
            ),
        }
        metrics["metrics"] = metric_rows
        metrics["forecast_diagnostics"] = {
            **dict(metrics.get("forecast_diagnostics", {})),
            "model": "mkonline_blend",
        }
        _write_json(staging / "metrics_hourly.json", metrics)

        manifest = dict(auto_manifest)
        manifest.update(
            {
                "script_version": "1.0.0-historical-residual-load-blend-replay",
                "source_run": str(autonomous_run),
                "residual_load_source": str(residual_load_source),
                "native_model": "mkonline_blend",
                "baseline_model": "residual_corrected",
                "mkonline_weights_retrained": False,
                "mkonline_autonomous_weight": reference_weight_auto,
                "mkonline_primary_weight": reference_weight_primary,
                "mkonline_reference_run": str(reference_run),
                "mkonline_reference_backtest_sha256": _sha256(
                    reference_run / "backtest_hourly_oof.csv.gz"
                ),
                "n_evaluation_hours": len(evaluation_index),
                "sha256_manifest": "artifact_checksums.json",
            }
        )
        _write_json(staging / "run_manifest.json", manifest)
        report_path = staging / f"historical_residual_load_{zone.lower()}_blend.html"
        write_hourly_html_report(
            staging,
            output_path=report_path,
            title=title
            or f"{zone} - replay historique residual_load - MKOnline blend",
            native_model="mkonline_blend",
            baseline_model="residual_corrected",
            zone=zone,
            timezone=timezone,
        )
        _write_checksums(
            staging,
            published_directory=output,
            source_paths={
                "autonomous_run_manifest": autonomous_run / "run_manifest.json",
                "autonomous_checksum_manifest": autonomous_run
                / "artifact_checksums.json",
                "autonomous_backtest": autonomous_run
                / "backtest_hourly_oof.csv.gz",
                "autonomous_forecast": auto_forecast_path,
                "reference_blend_manifest": reference_run / "run_manifest.json",
                "reference_blend_metrics": reference_run / "metrics_hourly.json",
                "reference_blend_backtest": reference_run
                / "backtest_hourly_oof.csv.gz",
                "reference_blend_forecast": reference_forecast_path,
            },
        )
        previous: Path | None = None
        if output.exists():
            if not overwrite:
                raise FileExistsError(output)
            previous = output.with_name(
                f".{output.name}.previous-{uuid.uuid4().hex}"
            )
            output.replace(previous)
        try:
            os.replace(staging, output)
        except Exception:
            if previous is not None and previous.exists() and not output.exists():
                os.replace(previous, output)
            raise
        if previous is not None:
            shutil.rmtree(previous)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


__all__ = ["HistoricalBlendReplayError", "publish_historical_blend_replay"]
