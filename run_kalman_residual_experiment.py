"""Publish a causal residual-plus-Kalman rolling-365 challenger.

The script reads already-issued live archives without modifying them.  It
replays the governed Kalman family on the realised residual-corrected history,
scores the last 365 local delivery days, applies the resulting filtered state
to the next delivery day, and publishes a self-contained experimental bundle.

Storm columns are copied only as report/evaluation comparators.  They are
never passed to :func:`chronos2_hourly.kalman_residual.replay_kalman_overlay`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd

from chronos2_hourly.kalman_residual import (
    DEFAULT_MODEL_KEY,
    KalmanReplayResult,
    KalmanResidualConfig,
    build_operational_kalman_view,
)
from chronos2_hourly.reporting import write_hourly_html_report


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DELIVERY_DAY = date(2026, 8, 28)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs" / "experiments" / "residual_kalman_rolling365_v1"
)
SUPPORTED_ZONES: tuple[str, ...] = ("FR", "DE", "BE", "NL", "ES")
ZONE_TIMEZONES: dict[str, str] = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}
ROLLING_DAYS = 365
INPUT_FILES: tuple[str, ...] = (
    "aligned_inputs.csv.gz",
    "model_covariates_with_future.csv.gz",
    "input_coverage.csv",
    "input_manifest.csv",
)
OPTIONAL_INPUT_FILES: tuple[str, ...] = (
    "future_input_coverage.csv",
    "storm_dashboard_official_forecast.parquet",
    "storm_dashboard_official_forecast_audit.json",
)


class KalmanExperimentError(RuntimeError):
    """Raised when an immutable source or rolling-365 contract is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KalmanExperimentError(f"JSON illisible: {path}") from exc
    if not isinstance(value, dict):
        raise KalmanExperimentError(f"{path}: objet JSON attendu.")
    return value


def normalize_zones(raw: Sequence[str]) -> tuple[str, ...]:
    """Normalize CLI zone tokens while rejecting ambiguity and duplicates."""

    tokens: list[str] = []
    for item in raw:
        tokens.extend(part.strip().upper() for part in str(item).split(","))
    zones = tuple(token for token in tokens if token)
    if not zones:
        raise ValueError("Au moins une zone est requise.")
    if len(zones) != len(set(zones)):
        raise ValueError("La liste des zones contient un doublon.")
    unknown = sorted(set(zones).difference(SUPPORTED_ZONES))
    if unknown:
        raise ValueError(f"Zones non supportees: {unknown}.")
    return zones


def source_run_for(
    zone: str,
    delivery_day: date,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve the existing live archive for a zone and delivery day."""

    token = delivery_day.isoformat()
    if zone == "FR":
        relative = Path("runs") / "live" / f"fr_day_ahead_{token}"
    elif zone == "NL":
        relative = (
            Path("runs")
            / "live"
            / "nl_mkonline_v1"
            / f"nl_day_ahead_{token}"
        )
    else:
        relative = (
            Path("runs") / "live" / zone.lower() / f"{zone.lower()}_day_ahead_{token}"
        )
    return (project_root / relative).resolve()


def _validate_source(
    source: Path,
    *,
    zone: str,
    delivery_day: date,
) -> tuple[dict[str, Any], dict[str, str]]:
    required = [
        source / "run_manifest.json",
        source / "artifact_checksums.json",
        source / "metrics_hourly.json",
        source / "statistics_history_hourly.csv.gz",
        source / "statistics_history_audit.json",
        source / f"forecast_hourly_{zone.lower()}.csv",
        *(source / "inputs" / name for name in INPUT_FILES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Artefacts source absents: {missing}")
    manifest = _read_json(source / "run_manifest.json")
    if str(manifest.get("zone", "")).upper() != zone:
        raise KalmanExperimentError(f"{source}: run_manifest.zone incoherent.")
    timezone = str(manifest.get("timezone", ""))
    if timezone != ZONE_TIMEZONES[zone]:
        raise KalmanExperimentError(f"{source}: timezone incoherente ({timezone!r}).")
    forecast_start = pd.Timestamp(manifest.get("forecast_start_utc"))
    if forecast_start.tzinfo is None:
        forecast_start = forecast_start.tz_localize("UTC")
    if forecast_start.tz_convert(timezone).date() != delivery_day:
        raise KalmanExperimentError(
            f"{source}: le forecast ne correspond pas au {delivery_day}."
        )
    checksum_manifest = _read_json(source / "artifact_checksums.json")
    if checksum_manifest.get("algorithm") != "sha256" or not isinstance(
        checksum_manifest.get("artifacts"), list
    ):
        raise KalmanExperimentError(f"{source}: manifeste SHA-256 invalide.")
    declared = {
        str(item.get("path")): str(item.get("sha256", "")).lower()
        for item in checksum_manifest["artifacts"]
        if isinstance(item, Mapping)
        and not Path(str(item.get("path", ""))).is_absolute()
    }
    for path in required:
        if path.name == "artifact_checksums.json":
            continue
        relative = path.relative_to(source).as_posix()
        expected_hash = declared.get(relative)
        observed_hash = _sha256(path)
        if expected_hash != observed_hash:
            raise KalmanExperimentError(
                f"{source}: checksum source divergent ou absent pour {relative}."
            )
    hashes = {str(path.resolve()): _sha256(path) for path in required}
    return manifest, hashes


def _assert_sources_unchanged(hashes: Mapping[str, str]) -> None:
    changed = [
        path
        for path, digest in hashes.items()
        if not Path(path).is_file() or _sha256(Path(path)) != digest
    ]
    if changed:
        raise KalmanExperimentError(
            f"Une archive source a change pendant le run: {changed}"
        )


def _metric_row(
    actual: pd.Series,
    prediction: pd.Series,
    *,
    model: str,
) -> dict[str, Any]:
    actual_values = pd.to_numeric(actual, errors="coerce")
    prediction_values = pd.to_numeric(prediction, errors="coerce")
    paired = actual_values.notna() & prediction_values.notna()
    error = prediction_values.loc[paired] - actual_values.loc[paired]
    return {
        "model": model,
        "n": int(paired.sum()),
        "coverage": float(paired.mean()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(error.mean()),
        "prediction_mean_eur_mwh": float(prediction_values.loc[paired].mean()),
        "observed_mean_eur_mwh": float(actual_values.loc[paired].mean()),
        "mean_price_absolute_error_eur_mwh": float(
            abs(prediction_values.loc[paired].mean() - actual_values.loc[paired].mean())
        ),
    }


def _metrics_payload(
    evaluation: pd.DataFrame,
    *,
    start_day: date,
    end_day: date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    actual = evaluation["actual"]
    rows = [
        _metric_row(
            actual,
            evaluation[f"{model}__q50"],
            model=model,
        )
        for model in ("residual_corrected", DEFAULT_MODEL_KEY)
    ]
    for storm_model in ("storm_dashboard_official", "storm_evaluation_only"):
        column = f"{storm_model}__q50"
        if column in evaluation and pd.to_numeric(
            evaluation[column], errors="coerce"
        ).notna().any():
            rows.append(_metric_row(actual, evaluation[column], model=storm_model))
            break
    metrics = pd.DataFrame(rows)
    by_model = metrics.set_index("model")
    baseline = by_model.loc["residual_corrected"]
    kalman = by_model.loc[DEFAULT_MODEL_KEY]
    baseline_mae = float(baseline["mae"])
    mae_improvement_pct = (
        None
        if baseline_mae <= np.finfo(float).eps
        else float(100.0 * (baseline_mae - float(kalman["mae"])) / baseline_mae)
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "scope": "last_365_complete_local_delivery_days",
        "evaluation_start_local_date": start_day.isoformat(),
        "evaluation_end_local_date": end_day.isoformat(),
        "evaluation_days": ROLLING_DAYS,
        "evaluation_hours": int(len(evaluation)),
        "metrics": rows,
        "paired_delta_vs_residual_corrected": {
            "mae_eur_mwh": float(kalman["mae"] - baseline["mae"]),
            "rmse_eur_mwh": float(kalman["rmse"] - baseline["rmse"]),
            "bias_eur_mwh": float(kalman["bias"] - baseline["bias"]),
            "mae_improvement_pct": mae_improvement_pct,
        },
        "storm_used_as_input": False,
    }
    return metrics, payload


@dataclass(frozen=True)
class KalmanMaterialization:
    """In-memory causal overlay shared by experiments and live exports."""

    replay: KalmanReplayResult
    statistics: pd.DataFrame
    backtest: pd.DataFrame
    forecast: pd.DataFrame
    metrics: pd.DataFrame
    metrics_payload: Mapping[str, Any]
    evaluation_start_day: date
    evaluation_end_day: date
    evaluation_index: pd.DatetimeIndex
    future_index: pd.DatetimeIndex


def build_kalman_materialization(
    *,
    statistics: pd.DataFrame,
    source_forecast: pd.DataFrame,
    covariates: pd.DataFrame,
    timezone: str,
    delivery_day: date,
    config: KalmanResidualConfig | None = None,
    training_lookback_days: int | None = None,
    rolling_refit_workers: int = 1,
    rolling_refit_cache_dir: str | Path | None = None,
) -> KalmanMaterialization:
    """Build one FINAL365 Kalman view without writing or mutating a source.

    Callers are responsible for supplying a disposable, already-refreshed
    Statistics snapshot.  This makes the same causal transformation usable by
    the immutable experiment publisher and by the atomic forecast exporter.
    """

    selected_config = config or KalmanResidualConfig()
    operational = build_operational_kalman_view(
        statistics=statistics,
        source_forecast=source_forecast,
        covariates=covariates,
        timezone=timezone,
        delivery_day=delivery_day,
        config=selected_config,
        training_lookback_days=training_lookback_days,
        rolling_refit_workers=rolling_refit_workers,
        rolling_refit_cache_dir=rolling_refit_cache_dir,
    )
    replay = operational.replay
    statistics_with_kalman = operational.statistics
    backtest = operational.backtest
    forecast = operational.forecast
    start_day = operational.evaluation_start_day
    end_day = operational.evaluation_end_day
    expected_evaluation = operational.evaluation_index
    expected_future = operational.future_index
    metrics, metrics_payload = _metrics_payload(
        backtest,
        start_day=start_day,
        end_day=end_day,
    )
    return KalmanMaterialization(
        replay=replay,
        statistics=statistics_with_kalman,
        backtest=backtest,
        forecast=forecast,
        metrics=metrics,
        metrics_payload=metrics_payload,
        evaluation_start_day=start_day,
        evaluation_end_day=end_day,
        evaluation_index=expected_evaluation,
        future_index=expected_future,
    )


def _csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["object"]):
        result[column] = result[column].map(
            lambda value: (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            )
        )
    return result


def _copy_inputs(source: Path, staging: Path) -> list[str]:
    destination = staging / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for filename in (*INPUT_FILES, *OPTIONAL_INPUT_FILES):
        path = source / "inputs" / filename
        if not path.is_file():
            if filename in INPUT_FILES:
                raise FileNotFoundError(path)
            continue
        shutil.copy2(path, destination / filename)
        copied.append(f"inputs/{filename}")
    return copied


def _write_checksums(
    staging: Path,
    *,
    published_directory: Path,
    source_hashes: Mapping[str, str],
) -> None:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "artifact_checksums.json":
            artifacts.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "role": "output_artifact",
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    for path, digest in sorted(source_hashes.items()):
        source_path = Path(path)
        artifacts.append(
            {
                "path": path,
                "role": "immutable_source_artifact",
                "size_bytes": source_path.stat().st_size,
                "sha256": digest,
            }
        )
    _write_json(
        staging / "artifact_checksums.json",
        {
            "algorithm": "sha256",
            "output_directory": str(published_directory),
            "artifacts": artifacts,
        },
    )


def _publish_staging(staging: Path, output: Path, *, overwrite: bool) -> None:
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


def run_zone_experiment(
    zone: str,
    *,
    delivery_day: date,
    output_root: Path,
    config: KalmanResidualConfig,
    overwrite: bool = False,
    source_run: Path | None = None,
    training_lookback_days: int | None = None,
    rolling_refit_workers: int = 1,
) -> Path:
    """Replay and publish one immutable experimental zone bundle."""

    zone = normalize_zones((zone,))[0]
    source = (source_run or source_run_for(zone, delivery_day)).resolve()
    output = (output_root / delivery_day.isoformat() / zone.lower()).resolve()
    if (
        output == source
        or output in source.parents
        or source in output.parents
    ):
        raise ValueError(
            "La sortie experimentale et l'archive source doivent etre disjointes."
        )
    source_manifest, source_hashes = _validate_source(
        source,
        zone=zone,
        delivery_day=delivery_day,
    )
    timezone = ZONE_TIMEZONES[zone]
    statistics = pd.read_csv(source / "statistics_history_hourly.csv.gz")
    forecast_path = source / f"forecast_hourly_{zone.lower()}.csv"
    source_forecast = pd.read_csv(forecast_path)
    materialized = build_kalman_materialization(
        statistics=statistics,
        source_forecast=source_forecast,
        covariates=pd.read_csv(
            source / "inputs" / "model_covariates_with_future.csv.gz"
        ),
        timezone=timezone,
        delivery_day=delivery_day,
        config=config,
        training_lookback_days=training_lookback_days,
        rolling_refit_workers=rolling_refit_workers,
        rolling_refit_cache_dir=(
            output_root
            / "_rolling_refit_cache"
            / zone.lower()
        ),
    )
    cache_audit = materialized.replay.audit.get("rolling_refit_cache", {})
    if isinstance(cache_audit, Mapping) and cache_audit.get("enabled"):
        print(
            f"[KALMAN-CACHE] {zone}/experiment: "
            f"hits={cache_audit.get('history_hits', 0)}, "
            f"refits={cache_audit.get('history_fitted_days', 0)}, "
            f"future_hits={cache_audit.get('future_hits', 0)}, "
            f"future_refits={cache_audit.get('future_fitted_days', 0)}",
            flush=True,
        )
    replay = materialized.replay
    statistics_with_kalman = materialized.statistics
    backtest = materialized.backtest
    forecast = materialized.forecast
    metrics = materialized.metrics
    metrics_payload = materialized.metrics_payload
    start_day = materialized.evaluation_start_day
    end_day = materialized.evaluation_end_day

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{zone.lower()}.tmp-", dir=output.parent))
    try:
        copied_inputs = _copy_inputs(source, staging)
        backtest.to_csv(
            staging / "backtest_hourly_oof.csv.gz",
            index=False,
            compression="gzip",
        )
        statistics_with_kalman.to_csv(
            staging / "statistics_history_hourly.csv.gz",
            index=False,
            compression="gzip",
        )
        forecast.to_csv(staging / forecast_path.name, index=False)
        metrics.to_csv(staging / "metrics_comparison_rolling365.csv", index=False)
        _write_json(staging / "metrics_comparison_rolling365.json", metrics_payload)
        metrics.to_csv(staging / "metrics_hourly.csv", index=False)

        source_metrics = _read_json(source / "metrics_hourly.json")
        training = source_metrics.get("training_diagnostics")
        if not isinstance(training, dict):
            training = {}
            source_metrics["training_diagnostics"] = training
        training.update(
            {
                "evaluation_start_local_date": start_day.isoformat(),
                "evaluation_end_local_date": end_day.isoformat(),
                "evaluation_days": ROLLING_DAYS,
                "evaluation_hours": int(len(backtest)),
                "kalman_overlay": dict(replay.audit),
            }
        )
        source_metrics["metrics"] = metrics.to_dict(orient="records")
        _write_json(staging / "metrics_hourly.json", source_metrics)

        daily_audit = _csv_safe(replay.daily_audit)
        state_audit = _csv_safe(replay.state_audit)
        candidate_predictions = replay.candidate_predictions.copy()
        candidate_predictions.index.name = "delivery_start_utc"
        daily_audit.to_csv(staging / "kalman_daily_audit.csv", index=False)
        state_audit.to_csv(
            staging / "kalman_state_audit.csv.gz",
            index=False,
            compression="gzip",
        )
        candidate_predictions.reset_index().to_csv(
            staging / "kalman_candidate_hourly.csv.gz",
            index=False,
            compression="gzip",
        )
        if not replay.future_candidate_predictions.empty:
            future_candidates = replay.future_candidate_predictions.copy()
            future_candidates.index.name = "delivery_start_utc"
            future_candidates.reset_index().to_csv(
                staging / "kalman_future_candidate_hourly.csv",
                index=False,
            )
        _write_json(staging / "kalman_config.json", asdict(config))
        kalman_audit = {
            **dict(replay.audit),
            "publication_status": "experimental_challenger",
            "production_changed": False,
            "source_run": str(source),
            "source_statistics_sha256": source_hashes[
                str((source / "statistics_history_hourly.csv.gz").resolve())
            ],
            "source_forecast_sha256": source_hashes[str(forecast_path.resolve())],
            "future_delivery_day": delivery_day.isoformat(),
            "future_hours": len(forecast),
            "future_audit": dict(replay.future_audit),
            "rolling365_hours": len(backtest),
            "storm_used_as_input": False,
            "mkonline_used_as_input": False,
        }
        _write_json(staging / "kalman_filter_audit.json", kalman_audit)

        history_audit = _read_json(source / "statistics_history_audit.json")
        history_audit["statistics_history_path"] = "statistics_history_hourly.csv.gz"
        history_audit["statistics_history_sha256"] = _sha256(
            staging / "statistics_history_hourly.csv.gz"
        )
        history_audit["kalman_overlay"] = {
            "status": "complete",
            "model": DEFAULT_MODEL_KEY,
            "source_candidate": "residual_corrected",
            "candidate_frozen_before_storm_attachment": True,
            "storm_used_as_input": False,
            "audit_path": "kalman_filter_audit.json",
        }
        _write_json(staging / "statistics_history_audit.json", history_audit)

        manifest = dict(source_manifest)
        manifest.update(
            {
                "script_version": "1.0.0-residual-kalman-rolling365-experiment",
                "run_type": "experimental_challenger",
                "experiment": "residual_kalman_rolling365_v1",
                "source_run": str(source),
                "source_run_manifest_sha256": source_hashes[
                    str((source / "run_manifest.json").resolve())
                ],
                "delivery_day": delivery_day.isoformat(),
                "native_model": DEFAULT_MODEL_KEY,
                "baseline_model": "residual_corrected",
                "forecast_path": forecast_path.name,
                "n_forecast_hours": len(forecast),
                "n_evaluation_days": ROLLING_DAYS,
                "n_evaluation_hours": int(len(backtest)),
                "evaluation_start_local_date": start_day.isoformat(),
                "evaluation_end_local_date": end_day.isoformat(),
                "production_changed": False,
                "storm_used_for_prediction": False,
                "storm_used_as_input": False,
                "mkonline_used_as_input": False,
                "kalman_statistics_scope": {
                    "source": "statistics_history_with_evaluation_only_storm",
                    "candidate_model": DEFAULT_MODEL_KEY,
                    "upstream_model": "residual_corrected",
                    "scope": "last_365_complete_local_delivery_days",
                },
                "copied_reporting_inputs": copied_inputs,
                "sha256_manifest": "artifact_checksums.json",
            }
        )
        _write_json(staging / "run_manifest.json", manifest)

        report_path = staging / f"residual_kalman_{zone.lower()}_rolling365.html"
        write_hourly_html_report(
            staging,
            output_path=report_path,
            title=(
                f"{zone} — Correcteur résiduel + Kalman gouverné — "
                "rolling 365 jours"
            ),
            native_model=DEFAULT_MODEL_KEY,
            baseline_model="residual_corrected",
            zone=zone,
            timezone=timezone,
        )
        _write_checksums(
            staging,
            published_directory=output,
            source_hashes=source_hashes,
        )
        _assert_sources_unchanged(source_hashes)
        _publish_staging(staging, output, overwrite=overwrite)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay causal du correcteur residuel + Kalman et publication de "
            "rapports comparables sur les 365 derniers jours."
        )
    )
    parser.add_argument(
        "--delivery-day",
        type=date.fromisoformat,
        default=DEFAULT_DELIVERY_DAY,
        help="Jour de livraison live au format YYYY-MM-DD (defaut: 2026-08-28).",
    )
    parser.add_argument(
        "--zones",
        nargs="+",
        default=list(SUPPORTED_ZONES),
        help=(
            "Zones parmi FR DE BE NL ES; les listes separees par virgule "
            "sont acceptees."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Racine de publication experimentale.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remplace atomiquement un bundle experimental existant.",
    )
    parser.add_argument(
        "--training-lookback-days",
        type=int,
        default=None,
        help=(
            "Refit causal glissant optionnel; par exemple 365. Le cache "
            "incremental persistant est alors active."
        ),
    )
    parser.add_argument(
        "--rolling-refit-workers",
        type=int,
        choices=range(1, 9),
        default=1,
        metavar="1..8",
        help="Nombre de processus pour les seuls jours absents du cache.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.training_lookback_days is not None and args.training_lookback_days < 1:
        raise ValueError("--training-lookback-days doit etre strictement positif.")
    zones = normalize_zones(args.zones)
    config = KalmanResidualConfig()
    outputs: list[Path] = []
    for zone in zones:
        print(f"[KALMAN] {zone}: replay causal et publication rolling-365...")
        outputs.append(
            run_zone_experiment(
                zone,
                delivery_day=args.delivery_day,
                output_root=args.output_root.expanduser().resolve(),
                config=config,
                overwrite=bool(args.overwrite),
                training_lookback_days=args.training_lookback_days,
                rolling_refit_workers=args.rolling_refit_workers,
            )
        )
    print("[KALMAN] Termine:")
    for output in outputs:
        print(f"  - {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DELIVERY_DAY",
    "DEFAULT_OUTPUT_ROOT",
    "KalmanExperimentError",
    "build_parser",
    "main",
    "normalize_zones",
    "run_zone_experiment",
    "source_run_for",
]
