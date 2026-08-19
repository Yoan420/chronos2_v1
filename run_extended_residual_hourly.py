#!/usr/bin/env python
"""Build the frozen EXT223 residual run and the standard hourly HTML report.

This runner consumes only native Chronos-2 OOF/live forecasts and the causal
feature materialisation of an existing hourly run.  The annual evaluation
model is fitted on EXT223 + the first 365 existing OOF days, then evaluated on
the untouched final 365 days.  A separate operational model is fitted on
EXT223 + all 730 observed OOF days for the strictly later live delivery day.

No external or legacy price forecast is read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd

from chronos2_hourly.features import build_history_future_feature_matrix
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.models import BlendedResidualCorrector, ResidualCorrector
from chronos2_hourly.reporting import write_hourly_html_report
from chronos2_modular.common import load_yaml
from evaluate_hourly_backtest import evaluate


LOGGER = logging.getLogger("extended_residual_hourly")
SCRIPT_VERSION = "1.0.0-extended-oof-residual"
TIMEZONE = "Europe/Paris"
RECIPE = "blend_cat_hgb_w0.50"
SCHEMA = "chronos_only"
CALIBRATION_DAYS = 365
FINAL_DAYS = 365
EXTENDED_DAYS = 223
EXTENDED_HOURS = 5351
EXTENDED_START_DAY = "2024-01-02"
EXTENDED_END_DAY = "2024-08-11"
EXPECTED_META_FEATURES = 175 # 250 ? 
QUANTILES = ("q10", "q50", "q90")
CHRONOS_EXPERT_COLUMNS = tuple(f"chronos2__{name}" for name in QUANTILES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correcteur residuel FR EXT223, recette CatBoost/HistGBR 50/50, "
            "artefacts horaires et rapport HTML standard."
        )
    )
    parser.add_argument(
        "--config",
        default="chronos2_hourly_fr_residual_extended_v1.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Remplace atomiquement un run existant, seulement apres avoir "
            "construit et valide le nouveau dossier."
        ),
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} doit etre un mapping YAML.")
    return value


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_no_forbidden_forecast(frame: pd.DataFrame, *, name: str) -> None:
    forbidden = [column for column in frame if "storm" in str(column).casefold()]
    if forbidden:
        raise ValueError(f"{name}: forecast interdit detecte: {forbidden}.")


def _read_timestamped(
    path: Path,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if timestamp_column not in frame:
        raise ValueError(f"{path}: colonne {timestamp_column!r} absente.")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop(timestamp_column), utc=True, errors="raise"),
        name=timestamp_column,
    )
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path}: timestamps dupliques ou non tries.")
    _assert_no_forbidden_forecast(frame, name=str(path))
    return frame


def _validate_complete_days(
    index: pd.DatetimeIndex,
    *,
    expected_days: int,
    timezone: str = TIMEZONE,
) -> list[Any]:
    local_dates = pd.Index(index.tz_convert(timezone).date)
    days = local_dates.unique().tolist()
    if len(days) != expected_days:
        raise ValueError(
            f"Nombre de jours locaux inattendu: {len(days)} != {expected_days}."
        )
    if days != pd.date_range(days[0], days[-1], freq="D").date.tolist():
        raise ValueError("La plage de jours locaux n'est pas continue.")
    for day in days:
        observed = index[np.asarray(local_dates == day)]
        expected = local_delivery_day_index(day, timezone=timezone)
        if not observed.equals(expected):
            raise ValueError(f"Jour local incomplet ou DST invalide: {day}.")
    return days


def _numeric_quantiles(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = [column for column in QUANTILES if column not in frame]
    if missing:
        raise ValueError(f"{name}: quantiles absents: {missing}.")
    result = frame.loc[:, list(QUANTILES)].apply(pd.to_numeric, errors="coerce")
    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: quantiles manquants ou non finis.")
    if not ((values[:, 0] <= values[:, 1]) & (values[:, 1] <= values[:, 2])).all():
        raise ValueError(f"{name}: quantiles croises.")
    return result.astype(float)


def _load_chronos(
    path: Path,
    *,
    require_actual: bool,
    expected_days: int | None = None,
    timezone: str = TIMEZONE,
) -> pd.DataFrame:
    frame = _read_timestamped(path, timestamp_column="delivery_start_utc")
    quantiles = _numeric_quantiles(frame, name=str(path))
    frame.loc[:, list(QUANTILES)] = quantiles
    if "forecast_origin_utc" not in frame:
        raise ValueError(f"{path}: forecast_origin_utc absent.")
    origin = pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="raise")
    if not np.asarray(origin < frame.index).all():
        raise ValueError(f"{path}: origine non causale detectee.")
    frame["forecast_origin_utc"] = origin.to_numpy()
    if require_actual:
        if "actual" not in frame:
            raise ValueError(f"{path}: actual absent.")
        frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
        if not np.isfinite(frame["actual"].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: cible actual invalide.")
    if expected_days is not None:
        _validate_complete_days(
            frame.index,
            expected_days=expected_days,
            timezone=timezone,
        )
    return frame


def _load_external(path: Path, *, timezone: str = TIMEZONE) -> pd.DataFrame:
    frame = _load_chronos(path, require_actual=True, timezone=timezone)
    days = _validate_complete_days(
        frame.index,
        expected_days=EXTENDED_DAYS,
        timezone=timezone,
    )
    expected_index = pd.DatetimeIndex([], tz="UTC")
    for day in pd.date_range(EXTENDED_START_DAY, EXTENDED_END_DAY, freq="D"):
        expected_index = expected_index.append(
            local_delivery_day_index(day.date(), timezone=timezone)
        )
    if len(frame) != len(expected_index) or not frame.index.equals(expected_index):
        raise ValueError(
            "Le bloc EXT ne correspond pas a la timeline physique de la zone: "
            f"recu={len(frame)}, attendu={len(expected_index)}."
        )
    if str(days[0]) != EXTENDED_START_DAY or str(days[-1]) != EXTENDED_END_DAY:
        raise ValueError(f"Plage EXT inattendue: {days[0]}..{days[-1]}.")
    return frame


def _load_features(
    source_run: Path,
    *,
    timezone: str = TIMEZONE,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    inputs = source_run / "inputs"
    aligned = _read_timestamped(
        inputs / "aligned_inputs.csv.gz",
        timestamp_column="timestamp",
    )
    if "target" not in aligned:
        raise ValueError("aligned_inputs.csv.gz ne contient pas target.")
    target = pd.to_numeric(aligned.pop("target"), errors="coerce").astype(float)
    if target.isna().any() or not np.isfinite(target.to_numpy()).all():
        raise ValueError("La cible historique contient une valeur invalide.")

    context = _read_timestamped(
        inputs / "model_covariates_with_future.csv.gz",
        timestamp_column="timestamp",
    )
    feature_manifest = pd.read_csv(source_run / "feature_manifest.csv")
    if "feature" not in feature_manifest:
        raise ValueError("feature_manifest.csv ne contient pas feature.")
    expected_features = feature_manifest["feature"].astype(str).tolist()
    forbidden = [name for name in expected_features if "storm" in name.casefold()]
    if forbidden:
        raise ValueError(f"Feature interdite detectee dans le manifeste: {forbidden}.")
    input_columns = [name for name in expected_features if name in context]
    if not input_columns:
        raise ValueError("Aucune covariable de base ne peut etre reconstruite.")
    covariates = context.loc[:, input_columns].apply(pd.to_numeric, errors="coerce")
    missing_history = target.index.difference(covariates.index)
    if len(missing_history):
        raise ValueError(
            f"Les covariables omettent {len(missing_history)} heures historiques."
        )
    future_index = covariates.index.difference(target.index, sort=False)
    if not len(future_index):
        raise ValueError("Aucun horizon futur dans les inputs du run source.")
    historical = covariates.loc[target.index]
    future = covariates.loc[future_index]
    features = build_history_future_feature_matrix(
        target,
        historical,
        future,
        price_lags=(24, 48, 168, 336),
        rolling_windows=(24, 72, 168),
        timezone=timezone,
        scope="all",
    )
    if list(features.columns) != expected_features:
        raise RuntimeError(
            "La reconstruction des features differe du manifeste du run source."
        )
    features.index = features.index.tz_convert("UTC")
    features.index.name = "delivery_start_utc"
    history_features = features.loc[target.index]
    future_features = features.loc[future_index]
    return target, history_features, future_features


def _base_and_experts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = frame.loc[:, list(QUANTILES)].copy()
    experts = base.rename(
        columns={name: f"chronos2__{name}" for name in QUANTILES}
    )
    return base, experts


def _new_corrector(
    *,
    threads: int,
    timezone: str = TIMEZONE,
    primary_country: str = "FR",
) -> BlendedResidualCorrector:
    builder_options = {
        "timezone": timezone,
        "include_calendar": True,
        "include_rich_calendar": True,
        "rich_calendar_countries": ("FR", "DE", "BE", "ES", "NL"),
        "rich_calendar_primary_country": primary_country,
        "include_daily_profiles": True,
        "include_fundamental_interactions": False,
        "include_missing_indicators": False,
        "exclude_historical_prices": True,
        "exclude_day_of_year": True,
    }
    cat = ResidualCorrector(
        backend="catboost",
        feature_builder_options=builder_options,
        iterations=700,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=15.0,
        min_samples_leaf=30,
        random_state=42,
        thread_count=threads,
        verbose=False,
        max_abs_correction=None,
    )
    hgb = ResidualCorrector(
        backend="sklearn",
        feature_builder_options=builder_options,
        iterations=550,
        depth=5,
        learning_rate=0.035,
        l2_leaf_reg=60.0,
        min_samples_leaf=30,
        sklearn_early_stopping=False,
        random_state=42,
        max_abs_correction=None,
    )
    return BlendedResidualCorrector(
        {"cat_v1": cat, "hgb31": hgb},
        {"cat_v1": 0.5, "hgb31": 0.5},
        max_abs_correction=40.0,
    )


def _fit_inputs(
    external: pd.DataFrame,
    existing: pd.DataFrame,
    history_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    if len(external.index.intersection(existing.index)):
        raise ValueError("EXT et OOF existant se chevauchent.")
    index = external.index.append(existing.index)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("EXT + OOF n'est pas strictement chronologique.")
    X = history_features.loc[index]
    y = pd.concat([external["actual"], existing["actual"]]).astype(float)
    external_base, external_experts = _base_and_experts(external)
    existing_base, existing_experts = _base_and_experts(existing)
    base = pd.concat([external_base, existing_base])
    experts = pd.concat([external_experts, existing_experts])
    if not (X.index.equals(y.index) and X.index.equals(base.index) and X.index.equals(experts.index)):
        raise RuntimeError("EXT + OOF n'est pas aligne pour l'entrainement.")
    return X, y, base, experts


def _allclose_or_raise(
    left: pd.DataFrame | pd.Series,
    right: pd.DataFrame | pd.Series,
    *,
    name: str,
    atol: float = 1e-10,
) -> None:
    if not left.index.equals(right.index):
        raise ValueError(f"{name}: index differents.")
    if not np.allclose(
        left.to_numpy(dtype=float),
        right.to_numpy(dtype=float),
        rtol=1e-10,
        atol=atol,
        equal_nan=True,
    ):
        raise ValueError(f"{name}: valeurs differentes.")


def _validate_source_artifacts(
    *,
    source_backtest: pd.DataFrame,
    source_forecast: pd.DataFrame,
    chronos_oof: pd.DataFrame,
    chronos_live: pd.DataFrame,
    timezone: str = TIMEZONE,
) -> pd.DataFrame:
    if "fold_id" not in source_backtest:
        raise ValueError("Le backtest source ne contient pas fold_id.")
    existing = source_backtest.loc[source_backtest["fold_id"].notna()].copy()
    _validate_complete_days(
        existing.index,
        expected_days=CALIBRATION_DAYS + FINAL_DAYS,
        timezone=timezone,
    )
    if not existing.index.equals(chronos_oof.index):
        raise ValueError("Le fichier Chronos OOF ne correspond pas au backtest source.")
    _allclose_or_raise(
        existing.loc[:, list(CHRONOS_EXPERT_COLUMNS)],
        chronos_oof.loc[:, list(QUANTILES)].set_axis(
            CHRONOS_EXPERT_COLUMNS, axis=1
        ).astype(np.float32).astype(float),
        name="Chronos OOF",
    )
    _allclose_or_raise(
        existing["actual"],
        chronos_oof["actual"].astype(np.float32).astype(float),
        name="cible Chronos OOF",
    )

    if "delivery_start_utc" not in source_forecast:
        raise ValueError("Le forecast source ne contient pas delivery_start_utc.")
    forecast = source_forecast.copy()
    forecast.index = pd.DatetimeIndex(
        pd.to_datetime(forecast["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if forecast.index.has_duplicates or not forecast.index.is_monotonic_increasing:
        raise ValueError("Le forecast source est duplique ou non trie.")
    if not forecast.index.equals(chronos_live.index):
        raise ValueError("Le fichier Chronos live ne correspond pas au forecast source.")
    _allclose_or_raise(
        forecast.loc[:, list(CHRONOS_EXPERT_COLUMNS)],
        chronos_live.loc[:, list(QUANTILES)].set_axis(
            CHRONOS_EXPERT_COLUMNS, axis=1
        ),
        name="Chronos live",
    )
    if existing.index[-1] >= chronos_live.index[0]:
        raise ValueError("Le jour live ne commence pas apres l'OOF observe.")
    return existing


def _native_frame_from_source(existing: pd.DataFrame) -> pd.DataFrame:
    """Return the exact native Chronos values used by the screened v1 run."""

    required = {"actual", *CHRONOS_EXPERT_COLUMNS}
    missing = sorted(required.difference(existing.columns))
    if missing:
        raise ValueError(f"Backtest source incomplet: {missing}.")
    native = existing.loc[:, [*CHRONOS_EXPERT_COLUMNS, "actual"]].rename(
        columns={f"chronos2__{name}": name for name in QUANTILES}
    )
    return native.apply(pd.to_numeric, errors="coerce")


def _metric_rows(backtest: pd.DataFrame, final_index: pd.DatetimeIndex) -> list[dict[str, Any]]:
    frame = backtest.loc[final_index]
    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for model in ("lear", "catboost", "chronos2", "ensemble", "residual_corrected"):
        column = f"{model}__q50"
        if column not in frame:
            continue
        prediction = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        available = np.isfinite(prediction)
        scored = available & np.isfinite(actual)
        rows.append(
            {
                "model": model,
                "mae": float(np.mean(np.abs(actual[scored] - prediction[scored])))
                if scored.any()
                else None,
                "n_scored": int(scored.sum()),
                "n_expected": int(len(frame)),
                "prediction_coverage": float(available.mean()),
                "score_coverage": float(scored.mean()),
            }
        )
    return rows


def _copy_source_contract(source_run: Path, staging: Path) -> None:
    shutil.copytree(source_run / "inputs", staging / "inputs")
    for filename in (
        "feature_manifest.csv",
        "ensemble_weights.csv",
        "pit_feature_coverage_by_hour.csv",
        "pit_feature_coverage_summary.csv",
    ):
        source = source_run / filename
        if source.is_file():
            shutil.copy2(source, staging / filename)


def _artifact_checksums(
    staging: Path,
    *,
    final_output: Path,
    config_path: Path,
    source_paths: Mapping[str, Path],
) -> None:
    checksum_path = staging / "artifact_checksums.json"
    entries: list[dict[str, Any]] = []
    for role, path in {"source_config": config_path, **source_paths}.items():
        entries.append(
            {
                "path": str(path),
                "role": role,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    project_root = Path(__file__).resolve().parent
    source_code = [
        Path(__file__).resolve(),
        (project_root / "evaluate_hourly_backtest.py").resolve(),
        (project_root / "generate_hourly_html_report.py").resolve(),
    ]
    source_code.extend(
        path.resolve() for path in (project_root / "chronos2_hourly").rglob("*.py")
    )
    for path in sorted(set(source_code)):
        entries.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "role": "source_code",
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        entries.append(
            {
                "path": path.relative_to(staging).as_posix(),
                "role": "materialized_input"
                if "inputs" in path.relative_to(staging).parts
                else "run_artifact",
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    _write_json(
        checksum_path,
        {
            "algorithm": "sha256",
            "output_directory": str(final_output),
            "artifacts": entries,
        },
    )


def _publish_staging(
    staging: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """Publish a completed run, restoring the previous one on rename failure."""

    staging = staging.resolve()
    output_dir = output_dir.resolve()
    if not staging.is_dir():
        raise FileNotFoundError(staging)
    if staging.parent != output_dir.parent or not output_dir.name:
        raise ValueError("Le staging et le run final doivent partager leur parent.")
    previous: Path | None = None
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Le dossier de sortie existe deja: {output_dir}. "
                "Utilisez --overwrite ou choisissez --output-dir."
            )
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
        previous = output_dir.with_name(
            f".{output_dir.name}.previous-{uuid.uuid4().hex}"
        )
        output_dir.replace(previous)
    try:
        staging.replace(output_dir)
    except Exception:
        if previous is not None and previous.exists() and not output_dir.exists():
            previous.replace(output_dir)
        raise
    if previous is not None:
        shutil.rmtree(previous)


def _run(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    source_run: Path,
    extended_path: Path,
    chronos_oof_path: Path,
    chronos_live_path: Path,
    staging: Path,
    final_output: Path,
    threads: int,
    zone: str = "FR",
    timezone: str = TIMEZONE,
) -> tuple[float, Path]:
    LOGGER.info("Chargement et validation des artefacts figes...")
    external = _load_external(extended_path, timezone=timezone)
    chronos_oof = _load_chronos(
        chronos_oof_path,
        require_actual=True,
        expected_days=CALIBRATION_DAYS + FINAL_DAYS,
        timezone=timezone,
    )
    chronos_live = _load_chronos(
        chronos_live_path,
        require_actual=False,
        timezone=timezone,
    )
    source_backtest = _read_timestamped(
        source_run / "backtest_hourly_oof.csv.gz",
        timestamp_column="delivery_start_utc",
    )
    forecast_filename = f"forecast_hourly_{zone.lower()}.csv"
    source_forecast = pd.read_csv(source_run / forecast_filename)
    _assert_no_forbidden_forecast(source_forecast, name="forecast source")
    existing = _validate_source_artifacts(
        source_backtest=source_backtest,
        source_forecast=source_forecast,
        chronos_oof=chronos_oof,
        chronos_live=chronos_live,
        timezone=timezone,
    )
    existing_native = _native_frame_from_source(existing)
    target, history_features, future_features = _load_features(
        source_run,
        timezone=timezone,
    )
    if not source_backtest.index.isin(target.index).all():
        raise ValueError("Le backtest source depasse la cible materialisee.")

    local_days = pd.Index(existing.index.tz_convert(timezone).date)
    days = local_days.unique().tolist()
    calibration_index = existing.index[
        np.asarray(local_days.isin(days[:CALIBRATION_DAYS]), dtype=bool)
    ]
    final_index = existing.index[
        np.asarray(local_days.isin(days[CALIBRATION_DAYS:]), dtype=bool)
    ]
    if external.index[-1] >= calibration_index[0]:
        raise ValueError("EXT doit finir strictement avant la calibration existante.")

    LOGGER.info("Entrainement evaluation EXT223 + CAL365 (holdout final scelle)...")
    X_eval, y_eval, base_eval, experts_eval = _fit_inputs(
        external,
        existing_native.loc[calibration_index],
        history_features,
    )
    evaluation_model = _new_corrector(
        threads=threads,
        timezone=timezone,
        primary_country=zone,
    ).fit(
        X_eval,
        y_eval,
        base_eval,
        experts_eval,
    )
    if len(evaluation_model.feature_columns_) != EXPECTED_META_FEATURES:
        raise RuntimeError(
            "Schema de meta-features inattendu: "
            f"{len(evaluation_model.feature_columns_)} != {EXPECTED_META_FEATURES}."
        )
    final_base, final_experts = _base_and_experts(
        existing_native.loc[final_index]
    )
    final_prediction = evaluation_model.predict(
        history_features.loc[final_index],
        final_base,
        final_experts,
    )

    LOGGER.info("Entrainement operationnel EXT223 + OOF730 pour le jour live...")
    X_live_fit, y_live_fit, base_live_fit, experts_live_fit = _fit_inputs(
        external,
        existing_native,
        history_features,
    )
    live_model = _new_corrector(
        threads=threads,
        timezone=timezone,
        primary_country=zone,
    ).fit(
        X_live_fit,
        y_live_fit,
        base_live_fit,
        experts_live_fit,
    )
    live_base, live_experts = _base_and_experts(chronos_live)
    if not future_features.index.equals(chronos_live.index):
        raise ValueError("Les features futures ne correspondent pas au Chronos live.")
    live_prediction = live_model.predict(
        future_features,
        live_base,
        live_experts,
    )

    _copy_source_contract(source_run, staging)
    shutil.copy2(extended_path, staging / "inputs" / "chronos_oof_extended.csv.gz")
    shutil.copy2(chronos_oof_path, staging / "chronos_oof_hourly.csv.gz")
    shutil.copy2(chronos_live_path, staging / "chronos_live_hourly.csv")

    backtest = source_backtest.copy()
    for quantile in QUANTILES:
        backtest[f"residual_corrected__{quantile}"] = np.nan
        backtest.loc[final_index, f"residual_corrected__{quantile}"] = (
            final_prediction[quantile].to_numpy(dtype=float)
        )
    backtest["residual_correction"] = np.nan
    backtest.loc[final_index, "residual_correction"] = (
        final_prediction["q50"].to_numpy(dtype=float)
        - final_base["q50"].to_numpy(dtype=float)
    )
    backtest.reset_index().to_csv(
        staging / "backtest_hourly_oof.csv.gz",
        index=False,
        compression="gzip",
    )

    forecast = source_forecast.copy()
    forecast.index = pd.DatetimeIndex(
        pd.to_datetime(forecast["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    for quantile in QUANTILES:
        values = live_prediction.loc[forecast.index, quantile].to_numpy(dtype=float)
        forecast[quantile] = values
        forecast[f"residual_corrected__{quantile}"] = values
    forecast["price_eur_mwh"] = forecast["q50"]
    forecast["residual_correction"] = (
        forecast["q50"].to_numpy(dtype=float)
        - live_base.loc[forecast.index, "q50"].to_numpy(dtype=float)
    )
    forecast.reset_index(drop=True).to_csv(
        staging / forecast_filename,
        index=False,
    )

    metric_rows = _metric_rows(backtest, final_index)
    pd.DataFrame(metric_rows).to_csv(staging / "metrics_hourly.csv", index=False)
    source_metrics = json.loads(
        (source_run / "metrics_hourly.json").read_text(encoding="utf-8")
    )
    ensemble_weights = source_metrics.get("ensemble_weights", {})
    training_diagnostics = {
        "n_rows": int(len(history_features)),
        "n_target_observed": int(target.notna().sum()),
        "n_oof_common": int(len(chronos_oof)),
        "n_evaluation": int(len(final_index)),
        "metric_scope": f"sealed_final_{FINAL_DAYS}_delivery_days",
        "evaluation_start_local_date": str(days[CALIBRATION_DAYS]),
        "evaluation_end_local_date": str(days[-1]),
        "training_start_utc": str(history_features.index[0]),
        "training_end_utc": str(history_features.index[-1]),
        "residual_correction": {
            "enabled": True,
            "base_model": "chronos2",
            "recipe": RECIPE,
            "uses_legacy_forecast": False,
            "evaluation_fit_extended_days": EXTENDED_DAYS,
            "evaluation_fit_existing_days": CALIBRATION_DAYS,
            "evaluation_predicted_days": FINAL_DAYS,
            "live_fit_extended_days": EXTENDED_DAYS,
            "live_fit_existing_days": CALIBRATION_DAYS + FINAL_DAYS,
            "evaluation_model": evaluation_model.diagnostics(),
            "live_model": live_model.diagnostics(),
        },
    }
    forecast_diagnostics = {
        "n_forecast_hours": int(len(forecast)),
        "forecast_start_utc": str(forecast.index[0]),
        "forecast_end_utc": str(forecast.index[-1]),
        "residual_base_model": "chronos2",
        "residual_recipe": RECIPE,
    }
    _write_json(
        staging / "metrics_hourly.json",
        {
            "metrics": metric_rows,
            "ensemble_weights": ensemble_weights,
            "training_diagnostics": training_diagnostics,
            "forecast_diagnostics": forecast_diagnostics,
        },
    )

    recipe_payload = {
        "name": RECIPE,
        "schema": SCHEMA,
        "base_model": "chronos2",
        "weights": {"cat_v1": 0.5, "hgb31": 0.5},
        "final_clip_eur_mwh": 40.0,
        "expected_meta_features": EXPECTED_META_FEATURES,
        "external_price_forecasts_loaded": [],
        "components": {
            "cat_v1": {
                "backend": "catboost",
                "loss": "MAE", # RMSE ?? 
                "iterations": 700,
                "depth": 6,
                "learning_rate": 0.03,
                "l2_leaf_reg": 15.0,
                "random_seed": 42,
                "has_time": True,
            },
            "hgb31": {
                "backend": "HistGradientBoostingRegressor",
                "loss": "absolute_error",
                "max_iter": 550,
                "max_leaf_nodes": 31,
                "learning_rate": 0.035,
                "min_samples_leaf": 30,
                "l2_regularization": 60.0,
                "early_stopping": False,
                "random_state": 42,
            },
        },
        "protocol": {
            "extended": [EXTENDED_START_DAY, EXTENDED_END_DAY],
            "calibration": [str(days[0]), str(days[CALIBRATION_DAYS - 1])],
            "sealed_final": [str(days[CALIBRATION_DAYS]), str(days[-1])],
            "evaluation_fit_days": EXTENDED_DAYS + CALIBRATION_DAYS,
            "live_fit_days": EXTENDED_DAYS + CALIBRATION_DAYS + FINAL_DAYS,
            "final_targets_used_for_evaluation_fit": False,
        },
    }
    _write_json(staging / "extended_residual_recipe.json", recipe_payload)

    cat_model = evaluation_model.components_["cat_v1"]
    importance = np.asarray(cat_model.model_.feature_importances_, dtype=float)
    pd.DataFrame(
        {
            "feature": cat_model.feature_columns_,
            "importance": importance,
            "component": "cat_v1",
        }
    ).sort_values("importance", ascending=False).to_csv(
        staging / "residual_feature_importance.csv",
        index=False,
    )

    source_manifest = json.loads(
        (source_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    zones_config = _mapping(config.get("zones", {}), name="zones")
    zone_config = _mapping(zones_config.get(zone, {}), name=f"zones.{zone}")
    target_config = _mapping(
        zone_config.get("target", {}), name=f"zones.{zone}.target"
    )
    target_series = str(target_config.get("series", "")).strip()
    if not target_series:
        raise ValueError(f"zones.{zone}.target.series doit etre explicite.")
    run_manifest = dict(source_manifest)
    run_manifest.update(
        {
            "script_version": SCRIPT_VERSION,
            "config": str(config_path),
            "source_run": str(source_run),
            "model_id": "chronos2_native_plus_extended_residual",
            "zone": zone,
            "timezone": timezone,
            "target_series": target_series,
            "residual_recipe": RECIPE,
            "residual_schema": SCHEMA,
            "uses_legacy_price_forecast": False,
            "external_price_forecasts_loaded": [],
            "prediction_inputs": [
                "chronos2_native",
                "supervised_hourly_experts",
                "pit_residual_load_covariates",
            ],
            "evaluation_only_comparators": [],
            "storm_used_as_feature": False,
            "n_extended_oof_hours": int(len(external)),
            "n_chronos_oof_hours": int(len(chronos_oof)),
            "n_forecast_hours": int(len(chronos_live)),
            "forecast_start_utc": str(chronos_live.index[0]),
            "forecast_end_utc": str(chronos_live.index[-1]),
            "sha256_manifest": "artifact_checksums.json",
        }
    )
    _write_json(staging / "run_manifest.json", run_manifest)

    summary, daily, monthly, hourly = evaluate(
        backtest.reset_index(),
        baseline="ensemble__q50",
        candidate="residual_corrected__q50",
        actual="actual",
        timezone=timezone,
        bootstrap_samples=20_000,
        seed=42,
    )
    _write_json(staging / "evaluation_summary.json", summary)
    daily.to_csv(staging / "evaluation_by_day.csv", index=False)
    monthly.to_csv(staging / "evaluation_by_month.csv", index=False)
    hourly.to_csv(staging / "evaluation_by_hour.csv", index=False)

    report_config = _mapping(config.get("report", {}), name="report")
    filename = str(
        report_config.get(
            "filename",
            f"chronos2_hourly_{zone.lower()}_residual_extended_v1.html",
        )
    )
    report_path = staging / filename
    LOGGER.info("Generation du rapport HTML standard...")
    write_hourly_html_report(
        staging,
        output_path=report_path,
        title=report_config.get(
            "title",
            f"Chronos-2 horaire {zone} - correcteur residual EXT223",
        ),
        native_model="residual_corrected",
        baseline_model="ensemble",
        zone=zone,
        timezone=timezone,
        extreme_threshold=float(report_config.get("extreme_threshold", 150.0)),
        history_hours=int(report_config.get("forecast_history_hours", 168)),
    )
    _artifact_checksums(
        staging,
        final_output=final_output,
        config_path=config_path,
        source_paths={
            "source_extended_oof": extended_path,
            "source_chronos_oof": chronos_oof_path,
            "source_chronos_live": chronos_live_path,
            "source_backtest": source_run / "backtest_hourly_oof.csv.gz",
        },
    )
    candidate_mae = float(summary["candidate_mae"])
    return candidate_mae, report_path


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent
    settings = _mapping(config.get("extended_residual", {}), name="extended_residual")
    zone = str(settings.get("zone", "FR")).strip().upper()
    zones_config = _mapping(config.get("zones", {}), name="zones")
    zone_config = _mapping(zones_config.get(zone, {}), name=f"zones.{zone}")
    timezone = str(
        settings.get("timezone", zone_config.get("timezone", TIMEZONE))
    )
    if not zone or not timezone:
        raise ValueError("extended_residual.zone/timezone doivent etre explicites.")
    if settings.get("recipe", RECIPE) != RECIPE:
        raise ValueError(f"La recette de production est figee a {RECIPE}.")
    if settings.get("schema", SCHEMA) != SCHEMA:
        raise ValueError(f"Le schema de production est fige a {SCHEMA}.")
    source_run = _resolve(settings["source_run"], base=config_dir)
    extended_path = _resolve(settings["extended_oof_file"], base=config_dir)
    chronos_oof_path = _resolve(
        settings.get("chronos_oof_file", source_run / "chronos_oof_hourly.csv.gz"),
        base=config_dir,
    )
    chronos_live_path = _resolve(
        settings.get("chronos_live_file", source_run / "chronos_live_hourly.csv"),
        base=config_dir,
    )
    configured_output = _mapping(config.get("output", {}), name="output").get(
        "directory",
        "runs/chronos2_hourly_fr_residual_extended_v1",
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _resolve(configured_output, base=config_dir)
    )
    threads = int(args.threads if args.threads is not None else settings.get("threads", -1))
    if output_dir == source_run:
        raise ValueError("output.directory doit differer du run source.")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Le dossier de sortie existe deja: {output_dir}. "
            "Utilisez --overwrite ou choisissez --output-dir."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        candidate_mae, staging_report = _run(
            config_path=config_path,
            config=config,
            source_run=source_run,
            extended_path=extended_path,
            chronos_oof_path=chronos_oof_path,
            chronos_live_path=chronos_live_path,
            staging=staging,
            final_output=output_dir,
            threads=threads,
            zone=zone,
            timezone=timezone,
        )
        report_name = staging_report.name
        _publish_staging(staging, output_dir, overwrite=args.overwrite)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    report_path = output_dir / report_name
    LOGGER.info("MAE annuelle residual_corrected: %.6f EUR/MWh", candidate_mae)
    print(f"Run horaire : {output_dir}")
    print(
        "Forecast horaire : "
        f"{output_dir / f'forecast_hourly_{zone.lower()}.csv'}"
    )
    print(f"Rapport HTML : {report_path}")
    print(f"MAE annuelle : {candidate_mae:.6f} EUR/MWh")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Execution interrompue.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("Echec du runner EXT223 : %s", exc)
        raise SystemExit(1)
