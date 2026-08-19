#!/usr/bin/env python
"""Run the leakage-safe French hourly ensemble.

The target, metric and exported forecast are always hourly.  Chronos calls are
grouped by the physical length of the Europe/Paris delivery day (23/24/25),
while LEAR and CatBoost consume the same point-in-time feature matrix.  The
last configured evaluation period is never used to learn ensemble weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.chronos_adapter import (
    ChronosDeliveryPlan,
    execute_grouped_chronos_backtest,
    generate_delivery_plans,
    load_chronos_oof,
    make_existing_forecasting_executor,
    normalize_chronos_future,
    run_existing_live_forecast,
)
from chronos2_hourly.features import build_history_future_feature_matrix
from chronos2_hourly.fundamental_features import (
    SupplyStackColumns,
    SupplyStackParameters,
    build_fr_supply_stack_features,
)
from chronos2_hourly.models import HourlyCatBoost, HourlyLEAR, ResidualCorrector
from chronos2_hourly.oof_pipeline import (
    HourlyOOFConfig,
    HourlyOOFPipeline,
)
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_modular.common import (
    LOGGER,
    SCRIPT_VERSION,
    build_zone_configs,
    deep_get,
    load_yaml,
    resolve_path,
    set_reproducibility,
)
from chronos2_modular.data import prepare_zone_data
from chronos2_modular.forecasting import load_model
from chronos2_modular.saturn import sync_saturn_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forecast horaire FR : Chronos-2 + LEAR + CatBoost + "
            "ensemble convexe OOF, avec journées DST 23/24/25 h."
        )
    )
    parser.add_argument("--config", default="chronos2_hourly_fr.yaml")
    parser.add_argument(
        "--zone",
        default="FR",
        help="Zone unique a executer (FR par defaut, p. ex. BE, DE, ES, NL).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--refresh-data", action="store_true")
    full_refresh = parser.add_mutually_exclusive_group()
    full_refresh.add_argument("--full-data-refresh", action="store_true")
    full_refresh.add_argument(
        "--full-target-refresh",
        action="store_true",
        help=(
            "Reconstruit intégralement uniquement le cache de la cible; "
            "les séries PIT restent incrémentales."
        ),
    )
    parser.add_argument(
        "--skip-pit-refresh",
        action="store_true",
        help=(
            "N'actualise aucun historique PIT pendant une actualisation "
            "Saturn (utile avec --full-target-refresh)."
        ),
    )
    parser.add_argument("--data-as-of", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--chronos-oof-file",
        default=None,
        help="Réutilise un CSV/Parquet Chronos OOF déjà validable.",
    )
    parser.add_argument(
        "--chronos-live-file",
        default=None,
        help="Réutilise un CSV/Parquet Chronos pour le prochain jour.",
    )
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
        raise TypeError(f"{name} doit être un mapping YAML.")
    return value


def _utc_index(frame: pd.DataFrame | pd.Series, *, name: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{name}: DatetimeIndex timezone-aware requis.")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name}: index unique et croissant requis.")


def _to_utc_series(series: pd.Series, *, name: str) -> pd.Series:
    _utc_index(series, name=name)
    result = pd.to_numeric(series, errors="coerce").astype(float).copy()
    result.index = result.index.tz_convert("UTC")
    result.index.name = "delivery_start_utc"
    result.name = name
    if result.isna().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name}: la cible contient une valeur manquante/non finie.")
    return result


def _to_utc_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    _utc_index(frame, name=name)
    result = frame.copy()
    result.index = result.index.tz_convert("UTC")
    result.index.name = "delivery_start_utc"
    return result


def _feature_inputs(
    data: Any,
    config: Mapping[str, Any],
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return target, historical covariates, future covariates and features."""

    target = _to_utc_series(data.target, name="target")
    model_covariates = _to_utc_frame(
        data.model_context_covariates,
        name="model_context_covariates",
    )
    missing_target_hours = target.index.difference(model_covariates.index)
    if len(missing_target_hours):
        raise ValueError(
            "Les covariables ne couvrent pas tout l'historique cible: "
            f"{len(missing_target_hours)} heure(s) absente(s)."
        )

    configured_columns = deep_get(
        config,
        "hourly.feature_engineering.covariate_columns",
    )
    if configured_columns is None:
        selected_columns = list(dict.fromkeys(data.known_future_columns))
    else:
        if not isinstance(configured_columns, Sequence) or isinstance(
            configured_columns, (str, bytes)
        ):
            raise TypeError(
                "hourly.feature_engineering.covariate_columns doit être une liste."
            )
        selected_columns = [str(value) for value in configured_columns]
    if not selected_columns:
        raise ValueError("Aucune covariable future explicite n'est disponible.")
    missing_columns = [
        column for column in selected_columns if column not in model_covariates
    ]
    if missing_columns:
        raise ValueError(f"Covariables configurées absentes: {missing_columns}.")
    base = model_covariates.loc[:, selected_columns].copy()

    supply_config = _mapping(
        deep_get(config, "hourly.supply_stack", {}),
        name="hourly.supply_stack",
    )
    if bool(supply_config.get("enabled", False)):
        columns_config = _mapping(
            supply_config.get("columns", {}),
            name="hourly.supply_stack.columns",
        )
        parameters_config = _mapping(
            supply_config.get("parameters", {}),
            name="hourly.supply_stack.parameters",
        )
        supply_columns = SupplyStackColumns(**dict(columns_config))
        supply_parameters = SupplyStackParameters(**dict(parameters_config))
        supply_features = build_fr_supply_stack_features(
            model_covariates,
            columns=supply_columns,
            parameters=supply_parameters,
            rolling_windows=tuple(
                int(value)
                for value in supply_config.get("rolling_windows", (3, 6, 24))
            ),
            nan_policy=str(supply_config.get("nan_policy", "raise")),
        )
        collisions = sorted(set(base.columns).intersection(supply_features.columns))
        if collisions:
            raise ValueError(f"Collision de features supply-stack: {collisions}.")
        base = pd.concat([base, supply_features], axis=1)

    future_index = model_covariates.index.difference(target.index, sort=False)
    if future_index.empty:
        raise ValueError("Aucun horizon futur dans model_context_covariates.")
    if future_index[0] != target.index[-1] + pd.Timedelta(hours=1):
        raise ValueError("Le futur ne commence pas exactement après la cible.")
    historical_covariates = base.loc[target.index].copy()
    future_covariates = base.loc[future_index].copy()
    missing_future = future_covariates.isna().sum()
    if bool((missing_future > 0).any()):
        details = ", ".join(
            f"{column}={int(count)}"
            for column, count in missing_future.items()
            if count
        )
        raise ValueError(
            "Couverture PIT future bloquante: valeurs manquantes dans " + details
        )
    finite_future = np.isfinite(
        future_covariates.select_dtypes(include=[np.number, "bool"]).to_numpy(
            dtype=float
        )
    )
    if not bool(finite_future.all()):
        raise ValueError("Couverture PIT future bloquante: valeur infinie détectée.")

    price_lags = tuple(
        int(value)
        for value in deep_get(
            config,
            "hourly.feature_engineering.target_lags",
            (24, 48, 168),
        )
    )
    rolling_windows = tuple(
        int(value)
        for value in deep_get(
            config,
            "hourly.feature_engineering.target_rolling_windows",
            (24, 168),
        )
    )
    timezone = str(
        deep_get(config, "hourly.feature_engineering.timezone", data.timezone)
    )
    all_features = build_history_future_feature_matrix(
        target,
        historical_covariates,
        future_covariates,
        price_lags=price_lags,
        rolling_windows=rolling_windows,
        timezone=timezone,
        scope="all",
    )
    return target, historical_covariates, future_covariates, all_features


def _desired_plans(
    target: pd.Series,
    config: Mapping[str, Any],
    *,
    timezone: str,
) -> tuple[ChronosDeliveryPlan, ...]:
    windows = int(deep_get(config, "backtest.windows", 730))
    if windows < 2:
        raise ValueError("backtest.windows doit être >= 2 jours.")
    end_date = target.index[-1].tz_convert(timezone).date()
    start_date = end_date - timedelta(days=windows - 1)
    plans = generate_delivery_plans(
        start_date,
        end_date,
        forecast_origin_local_time=str(
            deep_get(config, "data.forecast_origin_local_time", "08:00")
        ),
        timezone=timezone,
    )
    target_set = set(target.index)
    missing = [
        plan.delivery_date
        for plan in plans
        if any(timestamp not in target_set for timestamp in plan.delivery_index_utc)
    ]
    if missing:
        raise ValueError(
            "La cible ne couvre pas les plans Chronos demandés. "
            f"Premiers jours absents: {missing[:3]}."
        )
    return plans


def _require_exact_oof_plans(
    frame: pd.DataFrame,
    plans: Sequence[ChronosDeliveryPlan],
    target: pd.Series,
) -> pd.DataFrame:
    expected = plans[0].delivery_index_utc.append(
        [plan.delivery_index_utc for plan in plans[1:]]
    )
    expected = pd.DatetimeIndex(expected, name="delivery_start_utc")
    if not frame.index.equals(expected):
        raise ValueError(
            "L'artefact Chronos OOF ne couvre pas exactement la fenêtre "
            f"demandée: expected={len(expected)}, received={len(frame)}."
        )
    expected_actual = target.loc[expected].to_numpy(dtype=float)
    artifact_actual = frame["actual"].to_numpy(dtype=float)
    # CSV round-trips of float32 targets can move the last decimal by about
    # 1e-5 EUR/MWh.  This tolerance is five hundred thousand times smaller
    # than a one-cent price move and still rejects any material target drift.
    actual_tolerance = 5e-5
    finite = np.isfinite(artifact_actual) & np.isfinite(expected_actual)
    mismatch = (
        np.isfinite(artifact_actual) != np.isfinite(expected_actual)
    ) | (finite & (np.abs(artifact_actual - expected_actual) > actual_tolerance))
    if mismatch.any():
        finite_mismatch = mismatch & finite
        maximum_difference = (
            float(
                np.max(
                    np.abs(
                        artifact_actual[finite_mismatch]
                        - expected_actual[finite_mismatch]
                    )
                )
            )
            if finite_mismatch.any()
            else float("nan")
        )
        examples = [
            str(expected[position])
            for position in np.flatnonzero(mismatch)[:3]
        ]
        raise ValueError(
            "La cible de l'artefact Chronos OOF diffère de la cible horaire "
            f"canonique sur {int(mismatch.sum())} heure(s); "
            f"écart maximal={maximum_difference:.6g}; exemples={examples}."
        )
    return frame


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


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


def _json_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_artifact_checksums(
    output_dir: Path,
    *,
    config_path: Path,
) -> Path:
    """Freeze hashes for the config, materialized inputs and run outputs."""

    checksum_path = output_dir / "artifact_checksums.json"
    entries: list[dict[str, Any]] = []
    if config_path.is_file():
        entries.append(
            {
                "path": str(config_path),
                "role": "source_config",
                "size_bytes": int(config_path.stat().st_size),
                "sha256": _sha256_file(config_path),
            }
        )
    project_root = Path(__file__).resolve().parent
    source_paths = {
        Path(__file__).resolve(),
        (project_root / "evaluate_hourly_backtest.py").resolve(),
        (project_root / "generate_hourly_html_report.py").resolve(),
    }
    for package in ("chronos2_hourly", "chronos2_modular"):
        source_paths.update(
            path.resolve()
            for path in (project_root / package).rglob("*.py")
        )
    for path in sorted(source_paths):
        if not path.is_file():
            continue
        entries.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "role": "source_code",
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        entries.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "role": (
                    "materialized_input"
                    if "inputs" in path.relative_to(output_dir).parts
                    else "run_artifact"
                ),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    _json_dump(
        checksum_path,
        {
            "algorithm": "sha256",
            "output_directory": str(output_dir),
            "artifacts": entries,
        },
    )
    return checksum_path


def _expert_factories(config: Mapping[str, Any], *, timezone: str):
    lear_config = dict(
        _mapping(deep_get(config, "hourly.lear", {}), name="hourly.lear")
    )
    catboost_config = dict(
        _mapping(
            deep_get(config, "hourly.catboost", {}),
            name="hourly.catboost",
        )
    )
    lear_config.setdefault("hour_column", "calendar_local_hour")
    lear_config.setdefault("timezone", timezone)
    catboost_config.setdefault("hour_column", "calendar_local_hour")
    catboost_config.setdefault("timezone", timezone)

    def new_lear() -> HourlyLEAR:
        return HourlyLEAR(**lear_config)

    def new_catboost() -> HourlyCatBoost:
        return HourlyCatBoost(**catboost_config)

    return new_lear, new_catboost


def _residual_corrector_factory(
    config: Mapping[str, Any],
    *,
    timezone: str,
):
    raw = dict(
        _mapping(
            deep_get(config, "hourly.residual_correction", {}),
            name="hourly.residual_correction",
        )
    )
    enabled = bool(raw.pop("enabled", False))
    base_model = str(raw.pop("base_model", "chronos2"))
    builder_options = dict(
        _mapping(
            raw.pop("feature_builder", {}),
            name="hourly.residual_correction.feature_builder",
        )
    )
    builder_options.setdefault("timezone", timezone)
    if not enabled:
        return None, base_model

    def new_residual_corrector() -> ResidualCorrector:
        return ResidualCorrector(
            feature_builder_options=builder_options,
            **raw,
        )

    # Instantiate once now so invalid YAML fails before the expensive OOF fit.
    new_residual_corrector()
    return new_residual_corrector, base_model


def _trim_to_complete_delivery_days(
    X: pd.DataFrame,
    target: pd.Series,
    *,
    timezone: str,
) -> tuple[pd.DataFrame, pd.Series, int]:
    if not X.index.equals(target.index):
        raise ValueError("Features et cible ne sont pas strictement alignées.")
    local_dates = pd.Index(X.index.tz_convert(timezone).date)
    unique_dates = local_dates.unique().tolist()
    complete_dates: list[object] = []
    for delivery_date in unique_dates:
        observed = X.index[local_dates == delivery_date]
        expected = local_delivery_day_index(delivery_date, timezone=timezone)
        if observed.equals(expected):
            complete_dates.append(delivery_date)
        elif complete_dates:
            raise ValueError(
                "Journée incomplète au milieu de l'historique: "
                f"{delivery_date}."
            )
    if not complete_dates:
        raise ValueError("Aucune journée de livraison complète dans l'historique.")
    expected_dates = pd.date_range(
        complete_dates[0], complete_dates[-1], freq="D"
    ).date.tolist()
    if complete_dates != expected_dates:
        raise ValueError("Les journées complètes de l'historique ne sont pas continues.")
    mask = np.asarray(local_dates.isin(complete_dates), dtype=bool)
    return X.loc[mask].copy(), target.loc[mask].copy(), len(complete_dates)


def _oof_configuration(
    config: Mapping[str, Any],
    *,
    n_complete_days: int,
    timezone: str,
) -> HourlyOOFConfig:
    raw = _mapping(deep_get(config, "hourly.oof", {}), name="hourly.oof")
    n_splits = int(raw.get("n_splits", 5))
    oof_days = int(deep_get(config, "backtest.windows", 730))
    gap_days = int(raw.get("gap_days", 1))
    if oof_days % n_splits:
        raise ValueError(
            "backtest.windows doit être divisible par hourly.oof.n_splits "
            "pour des folds journaliers de taille identique."
        )
    test_days = oof_days // n_splits
    min_train_days = n_complete_days - oof_days - gap_days
    configured_minimum = int(raw.get("min_train_days", 1))
    if min_train_days < configured_minimum:
        raise ValueError(
            "Historique insuffisant avant la fenêtre OOF: "
            f"{min_train_days} jours disponibles < {configured_minimum}."
        )
    evaluation_days = raw.get("evaluation_days", 365)
    if evaluation_days is not None:
        evaluation_days = int(evaluation_days)
        if evaluation_days >= oof_days:
            raise ValueError(
                "evaluation_days doit être strictement inférieur à "
                "backtest.windows afin de conserver un bloc d'apprentissage "
                "des poids."
            )
    return HourlyOOFConfig(
        n_splits=n_splits,
        split_on_delivery_days=True,
        min_train_days=min_train_days,
        test_days=test_days,
        gap_days=gap_days,
        evaluation_days=evaluation_days,
        ensemble_minimum_rows=int(
            raw.get(
                "ensemble_minimum_rows",
                deep_get(config, "hourly.ensemble.minimum_rows", 720),
            )
        ),
        ensemble_weight_l2=float(
            raw.get(
                "ensemble_weight_l2",
                deep_get(config, "hourly.ensemble.weight_l2", 1e-6),
            )
        ),
        timezone=timezone,
        require_single_future_day=True,
    )


def _historical_feature_coverage(
    feature_frame: pd.DataFrame,
    *,
    timezone: str,
    evaluation_days: int | None,
    feature_substrings: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit point-in-time feature availability by local delivery hour."""

    if not isinstance(feature_frame.index, pd.DatetimeIndex):
        raise TypeError("feature_frame doit avoir un DatetimeIndex.")
    needles = tuple(str(value).lower() for value in feature_substrings if value)
    columns = [
        column
        for column in feature_frame.columns
        if any(needle in str(column).lower() for needle in needles)
    ]
    detail_columns = [
        "scope",
        "feature",
        "local_hour",
        "n_expected",
        "n_available",
        "coverage",
    ]
    summary_columns = [
        "scope",
        "feature",
        "n_hours",
        "n_days",
        "n_incomplete_days",
        "coverage",
        "minimum_hour_coverage",
        "worst_local_hour",
    ]
    if not columns:
        return pd.DataFrame(columns=detail_columns), pd.DataFrame(
            columns=summary_columns
        )

    local = feature_frame.index.tz_convert(timezone)
    local_dates = pd.Index(local.date)
    unique_dates = local_dates.unique().tolist()
    scopes: dict[str, np.ndarray] = {
        "oof_total": np.ones(len(feature_frame), dtype=bool)
    }
    if evaluation_days is not None:
        evaluation_dates = set(unique_dates[-evaluation_days:])
        evaluation_mask = np.asarray(local_dates.isin(evaluation_dates))
        scopes["calibration"] = ~evaluation_mask
        scopes["evaluation"] = evaluation_mask

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for scope, scope_mask in scopes.items():
        scoped_dates = local_dates[scope_mask]
        for column in columns:
            values = pd.to_numeric(feature_frame[column], errors="coerce").to_numpy(
                dtype=float
            )
            available = np.isfinite(values)
            hourly_coverages: list[tuple[int, float]] = []
            for hour in range(24):
                mask = scope_mask & (local.hour.to_numpy() == hour)
                n_expected = int(mask.sum())
                if not n_expected:
                    continue
                n_available = int((available & mask).sum())
                coverage = float(n_available / n_expected)
                hourly_coverages.append((hour, coverage))
                detail_rows.append(
                    {
                        "scope": scope,
                        "feature": column,
                        "local_hour": hour,
                        "n_expected": n_expected,
                        "n_available": n_available,
                        "coverage": coverage,
                    }
                )
            scoped_available = available[scope_mask]
            day_complete = pd.Series(
                scoped_available,
                index=pd.Index(scoped_dates),
            ).groupby(level=0, sort=False).all()
            worst_hour, minimum_coverage = min(
                hourly_coverages,
                key=lambda item: item[1],
            )
            summary_rows.append(
                {
                    "scope": scope,
                    "feature": column,
                    "n_hours": int(scope_mask.sum()),
                    "n_days": int(day_complete.size),
                    "n_incomplete_days": int((~day_complete).sum()),
                    "coverage": float(scoped_available.mean()),
                    "minimum_hour_coverage": float(minimum_coverage),
                    "worst_local_hour": int(worst_hour),
                }
            )
    return (
        pd.DataFrame(detail_rows, columns=detail_columns),
        pd.DataFrame(summary_rows, columns=summary_columns),
    )


def _load_or_run_chronos_oof(
    *,
    source_path: Path | None,
    plans: Sequence[ChronosDeliveryPlan],
    target: pd.Series,
    data: Any,
    runtime: Any | None,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if source_path is not None:
        frame = load_chronos_oof(source_path)
    else:
        if runtime is None:
            raise RuntimeError("Le runtime Chronos-2 n'a pas été chargé.")
        context_length = int(deep_get(config, "model.context_length", 2048))
        target_utc = data.target.index.tz_convert("UTC")
        first_origin = int(target_utc.get_indexer(plans[0].delivery_index_utc)[0])
        if first_origin < context_length:
            raise ValueError(
                "Historique insuffisant pour le premier plan Chronos: "
                f"origin={first_origin}, context_length={context_length}."
            )
        executor = make_existing_forecasting_executor(
            data=data,
            runtime=runtime,
            context_length=context_length,
            origin_batch_size=int(
                deep_get(config, "model.origin_batch_size", 12)
            ),
            model_batch_size=int(
                deep_get(config, "model.model_batch_size", 128)
            ),
            with_covariates=True,
            variant="hourly_oof",
        )
        frame = execute_grouped_chronos_backtest(plans, executor)
    return _require_exact_oof_plans(frame, plans, target)


def _load_or_run_chronos_live(
    *,
    source_path: Path | None,
    plan: ChronosDeliveryPlan,
    data: Any,
    runtime: Any | None,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if source_path is not None:
        frame = normalize_chronos_future(_read_frame(source_path), plan)
    else:
        if runtime is None:
            raise RuntimeError("Le runtime Chronos-2 n'a pas été chargé.")
        frame = run_existing_live_forecast(
            plan,
            data=data,
            runtime=runtime,
            context_length=int(deep_get(config, "model.context_length", 2048)),
            model_batch_size=int(
                deep_get(config, "model.model_batch_size", 128)
            ),
            with_covariates=True,
            variant="hourly_live",
        )
    return frame


def _expand_chronos_history(
    training_index: pd.DatetimeIndex,
    chronos_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    unexpected = chronos_oof.index.difference(training_index)
    if len(unexpected):
        raise ValueError(
            f"Chronos OOF contient {len(unexpected)} heure(s) hors historique."
        )
    quantiles = pd.DataFrame(
        np.nan,
        index=training_index,
        columns=["q10", "q50", "q90"],
        dtype=float,
    )
    quantiles.loc[chronos_oof.index] = chronos_oof[
        ["q10", "q50", "q90"]
    ].to_numpy(dtype=float)
    origins = pd.Series(
        pd.NaT,
        index=training_index,
        dtype="datetime64[ns, UTC]",
        name="chronos_origin",
    )
    origins.loc[chronos_oof.index] = pd.DatetimeIndex(
        chronos_oof["forecast_origin_utc"]
    )
    return quantiles, origins


def _write_results(
    *,
    output_dir: Path,
    target: pd.Series,
    chronos_oof: pd.DataFrame,
    pipeline: HourlyOOFPipeline,
    forecast: Any,
    feature_frame: pd.DataFrame,
    config_path: Path,
    runtime: Any | None,
    data: Any,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    training = pipeline.training_result_
    backtest = training.oof_predictions.copy()
    backtest["ensemble__q10"] = training.ensemble_oof_predictions["q10"]
    backtest["ensemble__q50"] = training.ensemble_oof_predictions["q50"]
    backtest["ensemble__q90"] = training.ensemble_oof_predictions["q90"]
    if training.residual_oof_predictions is not None:
        for quantile in ("q10", "q50", "q90"):
            backtest[f"residual_corrected__{quantile}"] = (
                training.residual_oof_predictions[quantile]
            )
        residual_base_column = (
            "chronos2__q50"
            if pipeline.residual_base_model == "chronos2"
            else "ensemble__q50"
        )
        backtest["residual_correction"] = (
            backtest["residual_corrected__q50"]
            - backtest[residual_base_column]
        )
    backtest["actual"] = target
    backtest["fold_id"] = training.fold_id
    origin = pd.Series(pd.NaT, index=backtest.index, dtype="datetime64[ns, UTC]")
    origin.loc[chronos_oof.index] = pd.DatetimeIndex(
        chronos_oof["forecast_origin_utc"]
    )
    backtest["forecast_origin_utc"] = origin
    backtest.reset_index().to_csv(
        output_dir / "backtest_hourly_oof.csv.gz",
        index=False,
        compression="gzip",
    )

    training.metrics.reset_index().to_csv(
        output_dir / "metrics_hourly.csv", index=False
    )
    weights_frame = training.ensemble_weights.rename("weight").rename_axis(
        "model"
    ).reset_index()
    weights_frame.to_csv(output_dir / "ensemble_weights.csv", index=False)

    evaluation_corrector = getattr(
        pipeline,
        "residual_evaluation_corrector_",
        None,
    )
    if evaluation_corrector is not None:
        model = evaluation_corrector.model_
        importance = getattr(model, "feature_importances_", None)
        if importance is not None:
            pd.DataFrame(
                {
                    "feature": evaluation_corrector.feature_columns_,
                    "importance": np.asarray(importance, dtype=float),
                }
            ).sort_values("importance", ascending=False).to_csv(
                output_dir / "residual_feature_importance.csv",
                index=False,
            )

    forecast_frame = forecast.delivery_metadata.copy()
    for column in ("q10", "q50", "q90"):
        forecast_frame[column] = forecast.predictions[column]
    forecast_frame["price_eur_mwh"] = forecast.predictions["q50"]
    for column in forecast.expert_predictions.columns:
        forecast_frame[column] = forecast.expert_predictions[column]
    forecast_filename = f"forecast_hourly_{str(data.zone).lower()}.csv"
    forecast_frame.reset_index(drop=True).to_csv(
        output_dir / forecast_filename, index=False
    )

    pd.DataFrame(
        {
            "feature": feature_frame.columns,
            "dtype": [str(feature_frame[column].dtype) for column in feature_frame],
            "historical_non_missing": [
                int(feature_frame[column].notna().sum()) for column in feature_frame
            ],
        }
    ).to_csv(output_dir / "feature_manifest.csv", index=False)

    _json_dump(
        output_dir / "metrics_hourly.json",
        {
            "metrics": training.metrics.reset_index().to_dict("records"),
            "ensemble_weights": training.ensemble_weights.to_dict(),
            "training_diagnostics": training.diagnostics,
            "forecast_diagnostics": forecast.diagnostics,
        },
    )
    _json_dump(
        output_dir / "run_manifest.json",
        {
            "script_version": SCRIPT_VERSION,
            "config": str(config_path),
            "model_id": getattr(runtime, "model_id", "external_chronos_files"),
            "zone": data.zone,
            "timezone": data.timezone,
            "target_contract": "hourly_utc_no_interpolation",
            "delivery_horizon": "dynamic_23_24_25",
            "sha256_manifest": "artifact_checksums.json",
            "n_training_hours": len(target),
            "n_chronos_oof_hours": len(chronos_oof),
            "n_forecast_hours": len(forecast.predictions),
            "forecast_start_utc": str(forecast.predictions.index[0]),
            "forecast_end_utc": str(forecast.predictions.index[-1]),
            "active_features": list(feature_frame.columns),
            "input_diagnostics": data.diagnostics,
        },
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent
    if args.data_as_of is not None:
        data_config = config.setdefault("data", {})
        if not isinstance(data_config, dict):
            raise TypeError("data doit être un mapping YAML.")
        data_config["runtime_as_of"] = args.data_as_of
    set_reproducibility(int(deep_get(config, "model.seed", 42)))

    requested_zone = str(args.zone).strip().upper()
    if not requested_zone:
        raise ValueError("--zone ne peut pas etre vide.")
    zones = build_zone_configs(config, [requested_zone], None, None)
    if len(zones) != 1 or zones[0].zone != requested_zone:
        raise ValueError(
            "Le runner horaire attend exactement une zone: "
            f"{requested_zone}."
        )
    zone = zones[0]
    project_root = resolve_path(
        deep_get(config, "data.project_root", "."), config_dir
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else resolve_path(
            deep_get(config, "output.directory", "runs/chronos2_hourly_fr"),
            project_root,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    refresh_requested = bool(
        args.refresh_data
        or args.full_data_refresh
        or args.full_target_refresh
        or args.data_as_of is not None
    )
    if refresh_requested:
        if args.full_target_refresh and args.skip_pit_refresh:
            LOGGER.info(
                "Reconstruction intégrale de la cible Saturn; "
                "historiques PIT ignorés."
            )
        elif args.full_target_refresh:
            LOGGER.info(
                "Reconstruction intégrale de la cible Saturn et "
                "actualisation incrémentale des historiques PIT..."
            )
        else:
            LOGGER.info("Actualisation des séries et vintages Saturn...")
        manifest = sync_saturn_data(
            zones,
            config,
            config_dir,
            full=bool(args.full_data_refresh),
            full_target=bool(args.full_target_refresh),
            skip_pit=bool(args.skip_pit_refresh),
            as_of=args.data_as_of,
        )
        manifest.to_csv(output_dir / "saturn_sync_manifest.csv", index=False)

    data = prepare_zone_data(
        zone,
        config,
        config_dir,
        False,
        output_dir / "inputs",
    )
    target, _historical_covariates, future_covariates, all_features = (
        _feature_inputs(data, config)
    )
    training_features = all_features.loc[target.index].copy()
    training_features, training_target, n_complete_days = (
        _trim_to_complete_delivery_days(
            training_features,
            target,
            timezone=zone.timezone,
        )
    )
    future_features = all_features.loc[future_covariates.index].copy()
    oof_config = _oof_configuration(
        config,
        n_complete_days=n_complete_days,
        timezone=zone.timezone,
    )
    plans = _desired_plans(target, config, timezone=zone.timezone)

    live_dates = pd.Index(
        future_features.index.tz_convert(zone.timezone).date
    ).unique()
    if len(live_dates) != 1:
        raise ValueError("L'horizon live doit couvrir un seul jour local.")
    live_plan = generate_delivery_plans(
        live_dates[0],
        live_dates[0],
        forecast_origin_local_time=str(
            deep_get(config, "data.forecast_origin_local_time", "08:00")
        ),
        timezone=zone.timezone,
    )[0]
    if not future_features.index.equals(live_plan.delivery_index_utc):
        raise ValueError(
            "L'index futur préparé ne correspond pas au jour de livraison "
            "23/24/25 attendu."
        )

    oof_path = (
        Path(args.chronos_oof_file).expanduser().resolve()
        if args.chronos_oof_file
        else None
    )
    live_path = (
        Path(args.chronos_live_file).expanduser().resolve()
        if args.chronos_live_file
        else None
    )
    runtime = None
    if oof_path is None or live_path is None:
        runtime = load_model(config, args.device, args.local_files_only)

    chronos_oof = _load_or_run_chronos_oof(
        source_path=oof_path,
        plans=plans,
        target=target,
        data=data,
        runtime=runtime,
        config=config,
    )
    chronos_live = _load_or_run_chronos_live(
        source_path=live_path,
        plan=live_plan,
        data=data,
        runtime=runtime,
        config=config,
    )
    chronos_history, chronos_origins = _expand_chronos_history(
        training_features.index,
        chronos_oof,
    )

    data_quality = _mapping(
        deep_get(config, "hourly.data_quality", {}),
        name="hourly.data_quality",
    )
    coverage_features = training_features.copy()
    coverage_features.index = coverage_features.index.tz_convert("UTC")
    coverage_features = coverage_features.loc[chronos_oof.index]
    coverage_detail, coverage_summary = _historical_feature_coverage(
        coverage_features,
        timezone=zone.timezone,
        evaluation_days=oof_config.evaluation_days,
        feature_substrings=tuple(
            data_quality.get(
                "audit_feature_substrings",
                ("residual_load", "nuclear", "net_export", "firm_margin"),
            )
        ),
    )
    coverage_detail.to_csv(
        output_dir / "pit_feature_coverage_by_hour.csv",
        index=False,
    )
    coverage_summary.to_csv(
        output_dir / "pit_feature_coverage_summary.csv",
        index=False,
    )
    evaluation_coverage = coverage_summary.loc[
        coverage_summary["scope"].eq("evaluation")
    ]
    warning_threshold = data_quality.get(
        "warn_below_hourly_coverage",
        0.9,
    )
    if warning_threshold is not None and not evaluation_coverage.empty:
        weak = evaluation_coverage.loc[
            evaluation_coverage["minimum_hour_coverage"]
            < float(warning_threshold)
        ]
        for row in weak.itertuples(index=False):
            LOGGER.warning(
                "Couverture PIT faible: %s, heure locale %02d, %.1f%%.",
                row.feature,
                row.worst_local_hour,
                100.0 * row.minimum_hour_coverage,
            )
    minimum_coverage = data_quality.get("minimum_hourly_coverage")
    if minimum_coverage is not None and not evaluation_coverage.empty:
        failing = evaluation_coverage.loc[
            evaluation_coverage["minimum_hour_coverage"]
            < float(minimum_coverage)
        ]
        if not failing.empty:
            examples = ", ".join(
                f"{row.feature}@h{row.worst_local_hour:02d}="
                f"{row.minimum_hour_coverage:.1%}"
                for row in failing.head(5).itertuples(index=False)
            )
            raise ValueError(
                "Couverture PIT horaire sous le seuil configurÃ©: " + examples
            )

    new_lear, new_catboost = _expert_factories(
        config, timezone=zone.timezone
    )
    new_residual_corrector, residual_base_model = (
        _residual_corrector_factory(config, timezone=zone.timezone)
    )
    pipeline = HourlyOOFPipeline(
        config=oof_config,
        lear_factory=new_lear,
        catboost_factory=new_catboost,
        residual_corrector_factory=new_residual_corrector,
        residual_base_model=residual_base_model,
    ).fit(
        training_features,
        training_target,
        chronos_oof=chronos_history,
        chronos_origin=chronos_origins,
        chronos_is_oof=True,
    )
    live_origin = pd.Series(
        pd.DatetimeIndex(chronos_live["forecast_origin_utc"]),
        index=chronos_live.index,
        name="chronos_origin",
    )
    forecast = pipeline.predict(
        future_features,
        chronos_future=chronos_live[["q10", "q50", "q90"]],
        chronos_origin=live_origin,
    )

    chronos_oof.reset_index().to_csv(
        output_dir / "chronos_oof_hourly.csv.gz",
        index=False,
        compression="gzip",
    )
    chronos_live.reset_index().to_csv(
        output_dir / "chronos_live_hourly.csv", index=False
    )
    _write_results(
        output_dir=output_dir,
        target=training_target,
        chronos_oof=chronos_oof,
        pipeline=pipeline,
        forecast=forecast,
        feature_frame=training_features,
        config_path=config_path,
        runtime=runtime,
        data=data,
    )
    report_path: Path | None = None
    report_settings = _mapping(
        deep_get(config, "report", {}),
        name="report",
    )
    if bool(report_settings.get("enabled", False)):
        from chronos2_hourly.reporting import write_hourly_html_report

        report_filename = str(
            report_settings.get(
                "filename",
                f"chronos2_hourly_{data.zone.lower()}.html",
            )
        )
        configured_report_path = Path(report_filename).expanduser()
        if not configured_report_path.is_absolute():
            configured_report_path = output_dir / configured_report_path
        report_path = write_hourly_html_report(
            output_dir,
            output_path=configured_report_path,
            title=report_settings.get("title"),
            native_model=report_settings.get("native_model"),
            baseline_model=report_settings.get("baseline_model"),
            zone=data.zone,
            timezone=data.timezone,
            extreme_threshold=float(
                report_settings.get("extreme_threshold", 150.0)
            ),
            history_hours=int(
                report_settings.get("forecast_history_hours", 168)
            ),
        )
    checksum_manifest = _write_artifact_checksums(
        output_dir,
        config_path=config_path,
    )

    metrics = pipeline.training_result_.metrics
    primary_model = (
        "residual_corrected"
        if "residual_corrected" in metrics.index
        else "ensemble"
    )
    primary_mae = float(metrics.loc[primary_model, "mae"])
    LOGGER.info(
        "MAE %s sur holdout %s: %.3f EUR/MWh",
        primary_model,
        pipeline.training_result_.diagnostics["metric_scope"],
        primary_mae,
    )
    forecast_path = output_dir / f"forecast_hourly_{data.zone.lower()}.csv"
    print(f"Forecast horaire : {forecast_path.resolve()}")
    if report_path is not None:
        print(f"Rapport HTML : {report_path.resolve()}")
    print(f"Checksums SHA256 : {checksum_manifest.resolve()}")
    print(f"MAE holdout {primary_model} : {primary_mae:.3f} EUR/MWh")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Exécution interrompue.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("Échec du pipeline horaire : %s", exc)
        raise SystemExit(1)
