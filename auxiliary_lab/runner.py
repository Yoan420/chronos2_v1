"""Training, calibration, evaluation, comparison and inference engine."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid

import joblib
import numpy as np
import pandas as pd

from chronos2_hourly.kalman_covariates import KalmanCovariateConfig
from chronos2_hourly.kalman_residual import (
    EXOGENOUS_FILTER_GROUPS,
    KalmanResidualConfig,
    build_operational_kalman_view,
    replay_kalman_overlay,
)
from chronos2_hourly.models.blended_residual_corrector import BlendedResidualCorrector
from chronos2_hourly.models.residual_corrector import ResidualCorrector

from .config import (
    AuxiliaryLabConfig,
    KalmanCovariateSource,
    MODEL_NAMES,
    SUPPORTED_METRICS,
    ModelConfig,
    load_lab_config,
)
from .data import (
    BlendDataset,
    KalmanDataset,
    QUANTILES,
    ResidualDataset,
    _read_indexed,
    _issued_history,
    join_kalman_additional_sources,
    load_blend_dataset,
    load_kalman_dataset,
    load_residual_dataset,
)
from .metrics import DaySplit, apply_horizon, daily_metrics, metrics_row
from .report import write_report


class AuxiliaryLabError(RuntimeError):
    """Raised when an experiment cannot be completed safely."""


MAX_PARAMETER_TRIALS = 1_000


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(config: AuxiliaryLabConfig) -> tuple[Path, ...]:
    candidates = (
        config.source_run / "backtest_hourly_oof.csv.gz",
        config.source_run / "statistics_history_hourly.csv.gz",
        config.source_run / "inputs" / "aligned_inputs.csv.gz",
        config.source_run / "inputs" / "model_covariates_with_future.csv.gz",
        config.source_run / "run_manifest.json",
    )
    additional = tuple(
        source.path
        for source in config.models["kalman"].options.get(
            "additional_sources", ()
        )
    )
    residual_prefix = config.models["residual_corrector"].options[
        "prequential_bridge"
    ].get("history_prefix_path")
    return tuple(dict.fromkeys(
        path
        for path in (*candidates, *additional, residual_prefix)
        if path is not None and path.is_file()
    ))


def _hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in paths}


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("scikit-learn", "catboost", "pykalman", "joblib", "plotly"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def _parameter_trials(model: ModelConfig) -> list[dict[str, Any]]:
    keys = sorted(model.parameter_grid)
    if not keys:
        return [dict(model.fixed_parameters)]
    trials: list[dict[str, Any]] = []
    for values in itertools.product(*(model.parameter_grid[key] for key in keys)):
        params = dict(model.fixed_parameters)
        params.update(dict(zip(keys, values, strict=True)))
        trials.append(params)
        if len(trials) > MAX_PARAMETER_TRIALS:
            raise AuxiliaryLabError(
                f"La grille depasse {MAX_PARAMETER_TRIALS} configurations; "
                "reduisez parameter_grid."
            )
    return trials


def _phase_mask(split: DaySplit, index: pd.DatetimeIndex, timezone_name: str, phase: str) -> np.ndarray:
    return split.phase_mask(index, timezone=timezone_name, phase=phase)


def _prediction_frame(
    *,
    actual: pd.Series,
    prediction: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=prediction.index)
    frame["actual"] = actual.loc[prediction.index].to_numpy(dtype=float)
    for quantile in QUANTILES:
        frame[quantile] = prediction[quantile].to_numpy(dtype=float)
        frame[f"baseline_{quantile}"] = baseline.loc[prediction.index, quantile].to_numpy(dtype=float)
    return frame


def _scored(
    frame: pd.DataFrame,
    *,
    config: AuxiliaryLabConfig,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    selected = apply_horizon(
        frame,
        timezone=config.timezone,
        horizon_hours=config.split.horizon_hours,
    )
    requested = tuple(dict.fromkeys(("mae", "rmse", "bias", *config.metrics)))
    return selected, metrics_row(selected, requested=requested)


def _metric_records(
    frame: pd.DataFrame,
    *,
    config: AuxiliaryLabConfig,
    model: str,
    phase: str,
) -> list[dict[str, Any]]:
    candidate, candidate_metrics = _scored(frame, config=config)
    period = {
        "start_utc": candidate.index.min().isoformat(),
        "end_utc": candidate.index.max().isoformat(),
        "n_local_days": int(
            pd.Index(candidate.index.tz_convert(config.timezone).date).nunique()
        ),
    }
    baseline = candidate.rename(
        columns={
            "q10": "candidate_q10",
            "q50": "candidate_q50",
            "q90": "candidate_q90",
            "baseline_q10": "q10",
            "baseline_q50": "q50",
            "baseline_q90": "q90",
        }
    )
    _, baseline_metrics = _scored(baseline, config=config)
    return [
        {
            "model": model,
            "phase": phase,
            "role": "candidate",
            **period,
            **candidate_metrics,
        },
        {
            "model": model,
            "phase": phase,
            "role": "baseline",
            **period,
            **baseline_metrics,
        },
    ]


def _prediction_records(
    frame: pd.DataFrame,
    *,
    config: AuxiliaryLabConfig,
    model: str,
    phase: str,
    configuration_id: str,
) -> pd.DataFrame:
    selected, _ = _scored(frame, config=config)
    output = selected.reset_index()
    output.insert(1, "model", model)
    output.insert(2, "phase", phase)
    output.insert(3, "configuration_id", configuration_id)
    return output


def _new_residual(
    model_config: ModelConfig,
    params: Mapping[str, Any],
    *,
    seed: int,
    weights: Mapping[str, float] | None,
) -> ResidualCorrector | BlendedResidualCorrector:
    shared: dict[str, Any] = {}
    component_overrides: dict[str, dict[str, Any]] = {}
    for raw_name, value in params.items():
        name = str(raw_name)
        if name.startswith("components."):
            parts = name.split(".", 2)
            if len(parts) != 3 or not parts[1] or not parts[2]:
                raise ValueError(
                    f"Parametre composant invalide: {name!r}; utilisez "
                    "components.<nom>.<parametre>."
                )
            component_overrides.setdefault(parts[1], {})[parts[2]] = value
        else:
            shared[name] = value
    shared.setdefault("random_state", seed)
    if model_config.options["recipe"] == "single":
        if component_overrides:
            raise ValueError("Les overrides components.* exigent recipe: blend.")
        return ResidualCorrector(**shared)
    unknown_components = sorted(
        set(component_overrides).difference(model_config.options["components"])
    )
    if unknown_components:
        raise ValueError(f"Composants residuels inconnus: {unknown_components}.")
    components: dict[str, ResidualCorrector] = {}
    for name, component_parameters in model_config.options["components"].items():
        selected = {
            **shared,
            **dict(component_parameters),
            **component_overrides.get(str(name), {}),
        }
        selected.setdefault("random_state", seed)
        components[str(name)] = ResidualCorrector(**selected)
    if weights is None:
        raise ValueError("Une recette residuelle blend exige des poids.")
    return BlendedResidualCorrector(
        components,
        weights,
        max_abs_correction=model_config.options["blend_max_abs_correction"],
    )


def _residual_permutation_importance(
    model: ResidualCorrector | BlendedResidualCorrector,
    dataset: ResidualDataset,
    validation_mask: np.ndarray,
    *,
    seed: int,
) -> pd.DataFrame:
    X = dataset.X.loc[validation_mask]
    base = dataset.base.loc[validation_mask]
    experts = dataset.experts.loc[validation_mask]
    actual = dataset.actual.loc[validation_mask].to_numpy(dtype=float)
    reference = model.predict(X, base, experts)["q50"].to_numpy(dtype=float)
    reference_mae = float(np.mean(np.abs(actual - reference)))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for column in X.columns[:100]:
        permuted = X.copy()
        permuted[column] = rng.permutation(permuted[column].to_numpy(copy=True))
        prediction = model.predict(permuted, base, experts)["q50"].to_numpy(dtype=float)
        permuted_mae = float(np.mean(np.abs(actual - prediction)))
        rows.append(
            {
                "feature": str(column),
                "reference_mae": reference_mae,
                "permuted_mae": permuted_mae,
                "importance_mae_increase": permuted_mae - reference_mae,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["importance_mae_increase", "feature"], ascending=[False, True]
    )
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)


def _minimum_residual_training_rows(
    model: ResidualCorrector | BlendedResidualCorrector,
) -> int:
    if isinstance(model, ResidualCorrector):
        return int(model.min_training_rows)
    return max(
        int(component.min_training_rows)
        for component in model.components.values()
    )


def _prequential_residual_history(
    *,
    dataset: ResidualDataset,
    model_config: ModelConfig,
    selected_params: Mapping[str, Any],
    selected_weights: Mapping[str, float] | None,
    issued_history: pd.DataFrame,
    timezone_name: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build a strictly label-causal upstream history for the Kalman bridge.

    The selected residual recipe is frozen, but every fitted instance only
    reads labels from local delivery days strictly before its forecast block.
    Before the recipe has enough observed rows, the output is the identity
    Chronos forecast.  Complete, already-issued residual days replace the
    generated values on validation/test only.
    """

    bridge = dict(model_config.options["prequential_bridge"])
    cadence_days = int(bridge["refit_cadence_days"])
    lookback_days = int(bridge["training_lookback_days"])
    if bridge["cold_start_policy"] != "identity":  # guarded by config parser
        raise AuxiliaryLabError("Politique cold-start prequentielle non supportee.")

    index = dataset.X.index
    local_day_values = np.asarray(index.tz_convert(timezone_name).date, dtype=object)
    days = tuple(pd.Index(local_day_values).unique())
    phase_by_day = {
        **{day: "train" for day in dataset.split.train_days},
        **{day: "validation" for day in dataset.split.validation_days},
        **{day: "test" for day in dataset.split.test_days},
    }
    prototype = _new_residual(
        model_config,
        selected_params,
        seed=seed,
        weights=selected_weights,
    )
    minimum_rows = _minimum_residual_training_rows(prototype)
    generated = dataset.base.loc[:, list(QUANTILES)].copy()
    audit_rows: list[dict[str, Any]] = []

    # Resolve complete sealed evaluation days before fitting anything.  These
    # forecasts are already issued and need neither a reconstructed residual
    # model nor access to an evaluation label.
    published_columns = [f"residual_corrected__{q}" for q in QUANTILES]
    evaluation_days = tuple(
        [*dataset.split.validation_days, *dataset.split.test_days]
    )
    published_by_day: dict[object, pd.DataFrame] = {}
    published_origin_by_day: dict[object, pd.Timestamp] = {}
    published_status_by_day = {
        day: "columns_absent" for day in evaluation_days
    }
    if set(published_columns).issubset(issued_history.columns):
        if "forecast_origin_utc" not in issued_history.columns:
            raise AuxiliaryLabError(
                "L'overlay residual_corrected publie exige la colonne "
                "forecast_origin_utc."
            )
        published = issued_history.loc[:, published_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        for day in evaluation_days:
            day_mask = np.asarray(local_day_values == day, dtype=bool)
            day_index = index[day_mask]
            values = published.reindex(day_index)
            finite = np.isfinite(values.to_numpy(dtype=float)).all()
            present = bool(day_index.isin(published.index).all())
            if not (present and finite):
                published_status_by_day[day] = "not_complete_day"
                continue
            raw_origins = issued_history["forecast_origin_utc"].reindex(day_index)
            parsed_day_origins: list[pd.Timestamp] = []
            invalid_origin = False
            for raw_origin in raw_origins:
                try:
                    parsed_origin = pd.Timestamp(raw_origin)
                except (TypeError, ValueError):
                    invalid_origin = True
                    break
                if (
                    parsed_origin.tzinfo is None
                    or parsed_origin.utcoffset() is None
                ):
                    invalid_origin = True
                    break
                parsed_day_origins.append(parsed_origin.tz_convert("UTC"))
            origins = pd.DatetimeIndex(parsed_day_origins)
            expected_origin = (
                pd.Timestamp(day)
                - pd.Timedelta(days=1)
                + pd.Timedelta(hours=8)
            ).tz_localize(timezone_name).tz_convert("UTC")
            late_origin = origins > expected_origin
            contract_mismatch = origins != expected_origin
            if invalid_origin or len(origins) != len(day_index):
                raise AuxiliaryLabError(
                    "forecast_origin_utc absent, invalide ou sans offset "
                    f"explicite pour l'overlay publie du {day}."
                )
            if bool(late_origin.any()):
                raise AuxiliaryLabError(
                    "forecast_origin_utc posterieur au cutoff civil D-1 "
                    f"08:00 pour l'overlay publie du {day}."
                )
            if bool(contract_mismatch.any()):
                raise AuxiliaryLabError(
                    "forecast_origin_utc ne respecte pas l'origine "
                    f"contractuelle D-1 08:00 pour l'overlay publie du {day}."
                )
            renamed = values.copy()
            renamed.columns = list(QUANTILES)
            if bool(
                (
                    (renamed["q10"] > renamed["q50"])
                    | (renamed["q50"] > renamed["q90"])
                ).any()
            ):
                raise AuxiliaryLabError(
                    "Quantiles residual_corrected publies croises pour la "
                    f"journee {day}."
                )
            published_by_day[day] = renamed
            published_origin_by_day[day] = origins.max()
            published_status_by_day[day] = "complete_day_used"

    cursor = 0
    refit_number = 0
    while cursor < len(days):
        block_start = days[cursor]
        if block_start in published_by_day:
            values = published_by_day[block_start]
            generated.loc[values.index, list(QUANTILES)] = values.to_numpy(
                dtype=float
            )
            audit_rows.append(
                {
                    "delivery_day": str(block_start),
                    "phase": phase_by_day[block_start],
                    "generation_source": "published_issued_history",
                    "output_source": "published_issued_history",
                    "published_overlay_status": "complete_day_used",
                    "hours": len(values),
                    "block_start_day": str(block_start),
                    "block_position": 0,
                    "refit_at_block_start": False,
                    "refit_number": None,
                    "training_rows": 0,
                    "minimum_training_rows": minimum_rows,
                    "training_lookback_days": lookback_days,
                    "fit_start_day": None,
                    "fit_end_day": None,
                    "fit_first_timestamp_utc": None,
                    "fit_last_timestamp_utc": None,
                    "backend": "published_issued_history",
                    "published_max_origin_utc": published_origin_by_day[
                        block_start
                    ].isoformat(),
                    "published_origin_causality_violations": 0,
                    "published_origin_contract_mismatches": 0,
                    "causality_violations": 0,
                }
            )
            cursor += 1
            continue
        lower_day = (pd.Timestamp(block_start) - pd.Timedelta(days=lookback_days)).date()
        training_mask = np.asarray(
            (local_day_values >= lower_day) & (local_day_values < block_start),
            dtype=bool,
        )
        training_mask &= np.isfinite(dataset.actual.to_numpy(dtype=float))
        training_rows = int(training_mask.sum())

        if training_rows < minimum_rows:
            block_days = (block_start,)
            generation_source = "identity_chronos_cold_start"
            fitted_backend = "identity"
            did_refit = False
        else:
            candidate_block_days = days[cursor : cursor + cadence_days]
            # Never spend a refit/prediction on a later day that will be
            # replaced by a complete issued forecast.
            first_published = next(
                (
                    position
                    for position, candidate_day in enumerate(candidate_block_days)
                    if candidate_day in published_by_day
                ),
                len(candidate_block_days),
            )
            block_days = candidate_block_days[:first_published]
            if not block_days:  # guarded by the early branch above
                raise RuntimeError("Bloc prequentiel vide inattendu.")
            block_mask = np.asarray(
                pd.Index(local_day_values).isin(block_days), dtype=bool
            )
            fitted = _new_residual(
                model_config,
                selected_params,
                seed=seed,
                weights=selected_weights,
            )
            try:
                fitted.fit(
                    dataset.X.loc[training_mask],
                    dataset.actual.loc[training_mask],
                    dataset.base.loc[training_mask],
                    dataset.experts.loc[training_mask],
                )
                prediction = fitted.predict(
                    dataset.X.loc[block_mask],
                    dataset.base.loc[block_mask],
                    dataset.experts.loc[block_mask],
                )
            except Exception as exc:
                raise AuxiliaryLabError(
                    "Echec du refit prequentiel residual -> Kalman au "
                    f"bloc {block_start}: {type(exc).__name__}: {exc}"
                ) from exc
            generated.loc[block_mask, list(QUANTILES)] = prediction.loc[
                :, list(QUANTILES)
            ].to_numpy(dtype=float)
            generation_source = "prequential_refit"
            fitted_backend = str(fitted.backend_)
            did_refit = True
            refit_number += 1

        training_index = index[training_mask]
        training_days = local_day_values[training_mask]
        fit_start_day = training_days.min() if training_rows else None
        fit_end_day = training_days.max() if training_rows else None
        fit_first_utc = training_index.min() if training_rows else None
        fit_last_utc = training_index.max() if training_rows else None
        for offset, forecast_day in enumerate(block_days):
            day_mask = np.asarray(local_day_values == forecast_day, dtype=bool)
            violations = int(
                fit_end_day is not None and not (fit_end_day < forecast_day)
            )
            audit_rows.append(
                {
                    "delivery_day": str(forecast_day),
                    "phase": phase_by_day[forecast_day],
                    "generation_source": generation_source,
                    "output_source": generation_source,
                    "published_overlay_status": published_status_by_day.get(
                        forecast_day, "outside_evaluation"
                    ),
                    "hours": int(day_mask.sum()),
                    "block_start_day": str(block_start),
                    "block_position": int(offset),
                    "refit_at_block_start": bool(did_refit and offset == 0),
                    "refit_number": refit_number if did_refit else None,
                    "training_rows": training_rows,
                    "minimum_training_rows": minimum_rows,
                    "training_lookback_days": lookback_days,
                    "fit_start_day": (
                        str(fit_start_day) if fit_start_day is not None else None
                    ),
                    "fit_end_day": (
                        str(fit_end_day) if fit_end_day is not None else None
                    ),
                    "fit_first_timestamp_utc": (
                        fit_first_utc.isoformat() if fit_first_utc is not None else None
                    ),
                    "fit_last_timestamp_utc": (
                        fit_last_utc.isoformat() if fit_last_utc is not None else None
                    ),
                    "backend": fitted_backend,
                    "published_max_origin_utc": None,
                    "published_origin_causality_violations": 0,
                    "published_origin_contract_mismatches": 0,
                    "causality_violations": violations,
                }
            )
        cursor += len(block_days)

    audit = pd.DataFrame(audit_rows)
    violation_count = int(audit["causality_violations"].sum())
    if violation_count:
        raise AuxiliaryLabError(
            f"Le replay prequentiel contient {violation_count} fuite(s) de labels."
        )

    published_overlay_days = len(published_by_day)
    published_overlay_hours = sum(len(values) for values in published_by_day.values())

    generated_values = generated.to_numpy(dtype=float)
    if not np.isfinite(generated_values).all():
        raise AuxiliaryLabError("Le replay prequentiel a produit des valeurs non finies.")
    if bool(((generated["q10"] > generated["q50"]) | (generated["q50"] > generated["q90"])).any()):
        raise AuxiliaryLabError("Le replay prequentiel a produit des quantiles croises.")

    downstream = pd.DataFrame(
        {
            "actual": dataset.actual,
            "residual_corrected__q10": generated["q10"],
            "residual_corrected__q50": generated["q50"],
            "residual_corrected__q90": generated["q90"],
            "chronos2__q50": dataset.base["q50"],
            "residual_correction": generated["q50"] - dataset.base["q50"],
        },
        index=index,
    )
    downstream.index.name = "delivery_start_utc"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "blocked_prequential_residual_then_issued_evaluation_overlay",
        "selected_recipe_is_frozen": True,
        "fit_label_rule": "local_delivery_day < block_start_day",
        "refit_cadence_days": cadence_days,
        "training_lookback_days": lookback_days,
        "cold_start_policy": "identity_chronos_q10_q50_q90",
        "minimum_training_rows": minimum_rows,
        "local_days": len(days),
        "hours": len(index),
        "refits": refit_number,
        "identity_cold_start_days": int(
            audit["generation_source"].eq("identity_chronos_cold_start").sum()
        ),
        "generated_prequential_days": int(
            audit["generation_source"].eq("prequential_refit").sum()
        ),
        "published_overlay_scope": ["validation", "test"],
        "published_overlay_requires_complete_physical_day": True,
        "published_origin_column": "forecast_origin_utc",
        "published_origin_rule": (
            "forecast_origin_utc == civil cutoff D-1 08:00"
        ),
        "published_max_origin_utc": (
            max(published_origin_by_day.values()).isoformat()
            if published_origin_by_day
            else None
        ),
        "published_origin_causality_violations": 0,
        "published_origin_contract_mismatches": 0,
        "published_overlay_days": published_overlay_days,
        "published_overlay_hours": published_overlay_hours,
        "causality_violations": violation_count,
        "audit_csv": "kalman_upstream_prequential_audit.csv",
    }
    return downstream, audit, summary


def _run_residual(
    config: AuxiliaryLabConfig,
    model_config: ModelConfig,
    output: Path,
    *,
    build_kalman_history: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = load_residual_dataset(config, model_config)
    train_mask = _phase_mask(dataset.split, dataset.X.index, config.timezone, "train")
    validation_mask = _phase_mask(dataset.split, dataset.X.index, config.timezone, "validation")
    test_mask = _phase_mask(dataset.split, dataset.X.index, config.timezone, "test")
    parameter_trials = _parameter_trials(model_config)
    weight_candidates: Sequence[Mapping[str, float] | None] = (
        model_config.options["weight_candidates"]
        if model_config.options["recipe"] == "blend"
        else (None,)
    )
    trials = [
        (parameters, weights)
        for parameters in parameter_trials
        for weights in weight_candidates
    ]
    if len(trials) > MAX_PARAMETER_TRIALS:
        raise AuxiliaryLabError(
            f"La grille residuelle depasse {MAX_PARAMETER_TRIALS} essais "
            "apres combinaison avec weight_candidates."
        )
    rows: list[dict[str, Any]] = []
    fitted_trials: dict[str, ResidualCorrector | BlendedResidualCorrector] = {}
    validation_frames: dict[str, pd.DataFrame] = {}
    for number, (params, weights) in enumerate(trials, start=1):
        trial_id = f"residual_{number:04d}"
        try:
            fitted = _new_residual(
                model_config,
                params,
                seed=config.random_seed,
                weights=weights,
            ).fit(
                dataset.X.loc[train_mask],
                dataset.actual.loc[train_mask],
                dataset.base.loc[train_mask],
                dataset.experts.loc[train_mask],
            )
            prediction = fitted.predict(
                dataset.X.loc[validation_mask],
                dataset.base.loc[validation_mask],
                dataset.experts.loc[validation_mask],
            )
            frame = _prediction_frame(
                actual=dataset.actual,
                prediction=prediction,
                baseline=dataset.base.loc[validation_mask],
            )
            _, metrics = _scored(frame, config=config)
            rows.append(
                {
                    "model": "residual_corrector",
                    "configuration_id": trial_id,
                    "status": "complete",
                    "objective": config.objective,
                    "objective_value": metrics[config.objective],
                    "parameters": json.dumps(
                        _json_safe({"model": params, "weights": weights}), sort_keys=True
                    ),
                    "backend": fitted.backend_,
                    "n_training_rows": fitted.n_training_rows_,
                    "error": "",
                }
            )
            fitted_trials[trial_id] = fitted
            validation_frames[trial_id] = frame
        except Exception as exc:
            rows.append(
                {
                    "model": "residual_corrector",
                    "configuration_id": trial_id,
                    "status": "failed",
                    "objective": config.objective,
                    "objective_value": np.inf,
                    "parameters": json.dumps(
                        _json_safe({"model": params, "weights": weights}), sort_keys=True
                    ),
                    "backend": "",
                    "n_training_rows": int(train_mask.sum()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    leaderboard = pd.DataFrame(rows)
    successful = leaderboard.loc[leaderboard["status"] == "complete"]
    if successful.empty:
        details = "; ".join(leaderboard["error"].astype(str).head(3))
        raise AuxiliaryLabError(
            "Toutes les configurations du correcteur residuel ont echoue: "
            + details
        )
    selected_row = successful.sort_values(["objective_value", "configuration_id"]).iloc[0]
    selected_id = str(selected_row["configuration_id"])
    leaderboard["selected"] = leaderboard["configuration_id"].eq(selected_id)
    selected_params, selected_weights = trials[
        int(selected_id.rsplit("_", 1)[1]) - 1
    ]

    evaluation_mask = train_mask | validation_mask
    evaluation_model = _new_residual(
        model_config,
        selected_params,
        seed=config.random_seed,
        weights=selected_weights,
    ).fit(
        dataset.X.loc[evaluation_mask],
        dataset.actual.loc[evaluation_mask],
        dataset.base.loc[evaluation_mask],
        dataset.experts.loc[evaluation_mask],
    )
    test_prediction = evaluation_model.predict(
        dataset.X.loc[test_mask], dataset.base.loc[test_mask], dataset.experts.loc[test_mask]
    )
    test_frame = _prediction_frame(
        actual=dataset.actual,
        prediction=test_prediction,
        baseline=dataset.base.loc[test_mask],
    )
    validation_frame = validation_frames[selected_id]

    downstream_history: pd.DataFrame | None = None
    downstream_audit: pd.DataFrame | None = None
    downstream_audit_summary: dict[str, Any] | None = None
    if build_kalman_history:
        (
            downstream_history,
            downstream_audit,
            downstream_audit_summary,
        ) = _prequential_residual_history(
            dataset=dataset,
            model_config=model_config,
            selected_params=selected_params,
            selected_weights=selected_weights,
            issued_history=_issued_history(config.source_run),
            timezone_name=config.timezone,
            seed=config.random_seed,
        )

    full_model = _new_residual(
        model_config,
        selected_params,
        seed=config.random_seed,
        weights=selected_weights,
    ).fit(
        dataset.X,
        dataset.actual,
        dataset.base,
        dataset.experts,
    )
    model_dir = output / "models" / "residual_corrector"
    model_dir.mkdir(parents=True, exist_ok=True)
    if downstream_history is not None:
        downstream_history.reset_index().to_csv(
            model_dir / "kalman_upstream_history.csv.gz",
            index=False,
            compression="gzip",
        )
        assert downstream_audit is not None
        assert downstream_audit_summary is not None
        downstream_audit.to_csv(
            model_dir / "kalman_upstream_prequential_audit.csv", index=False
        )
        _write_json(
            model_dir / "kalman_upstream_prequential_audit.json",
            downstream_audit_summary,
        )
    importance = _residual_permutation_importance(
        fitted_trials[selected_id],
        dataset,
        validation_mask,
        seed=config.random_seed,
    )
    importance.to_csv(model_dir / "feature_importance.csv", index=False)
    joblib.dump(evaluation_model, model_dir / "model_evaluation.joblib")
    joblib.dump(full_model, model_dir / "model_full.joblib")
    residual_manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_kind": "residual_corrector",
        "selected_configuration_id": selected_id,
        "parameters": selected_params,
        "recipe": model_config.options["recipe"],
        "weights": selected_weights,
        "base_model": dataset.base_model,
        "expert_models": dataset.expert_models,
        "feature_columns": list(
            full_model.x_columns_in_
            if isinstance(full_model, ResidualCorrector)
            else next(iter(full_model.components_.values())).x_columns_in_
        ),
        "artifact": "model_full.joblib",
        "artifact_sha256": _sha256(model_dir / "model_full.joblib"),
        "evaluation_artifact": "model_evaluation.joblib",
        "evaluation_artifact_sha256": _sha256(
            model_dir / "model_evaluation.joblib"
        ),
        "serialization_warning": (
            "Ne charger que des artefacts locaux dont le SHA-256 est verifie."
        ),
    }
    history_prefix_path = model_config.options["prequential_bridge"].get(
        "history_prefix_path"
    )
    if history_prefix_path is not None:
        residual_manifest["history_prefix"] = {
            "path": str(history_prefix_path),
            "sha256_at_training": _sha256(history_prefix_path),
            "schema": (
                "delivery_start_utc, forecast_origin_utc, q10, q50, q90, actual"
            ),
            "origin_rule": "forecast_origin_utc == civil cutoff D-1 08:00",
            "origin_contract_mismatches": 0,
        }
    if downstream_history is not None:
        downstream_path = model_dir / "kalman_upstream_history.csv.gz"
        downstream_audit_path = (
            model_dir / "kalman_upstream_prequential_audit.csv"
        )
        downstream_audit_json_path = (
            model_dir / "kalman_upstream_prequential_audit.json"
        )
        residual_manifest.update(
            {
                "kalman_upstream_history": downstream_path.name,
                "kalman_upstream_history_sha256": _sha256(downstream_path),
                "kalman_upstream_protocol": downstream_audit_summary[
                    "protocol"
                ],
                "kalman_upstream_prequential_audit": downstream_audit_path.name,
                "kalman_upstream_prequential_audit_sha256": _sha256(
                    downstream_audit_path
                ),
                "kalman_upstream_prequential_audit_json": (
                    downstream_audit_json_path.name
                ),
                "kalman_upstream_prequential_audit_json_sha256": _sha256(
                    downstream_audit_json_path
                ),
                "kalman_upstream_prequential_summary": downstream_audit_summary,
            }
        )
    _write_json(model_dir / "manifest.json", residual_manifest)
    predictions = pd.concat(
        [
            _prediction_records(validation_frame, config=config, model="residual_corrector", phase="validation", configuration_id=selected_id),
            _prediction_records(test_frame, config=config, model="residual_corrector", phase="test", configuration_id=selected_id),
        ],
        ignore_index=True,
    )
    metrics = pd.DataFrame(
        _metric_records(validation_frame, config=config, model="residual_corrector", phase="validation")
        + _metric_records(test_frame, config=config, model="residual_corrector", phase="test")
    )
    daily = pd.concat(
        [
            daily_metrics(apply_horizon(validation_frame, timezone=config.timezone, horizon_hours=config.split.horizon_hours), timezone=config.timezone).assign(model="residual_corrector", phase="validation"),
            daily_metrics(apply_horizon(test_frame, timezone=config.timezone, horizon_hours=config.split.horizon_hours), timezone=config.timezone).assign(model="residual_corrector", phase="test"),
        ],
        ignore_index=True,
    )
    return leaderboard, predictions, metrics, daily


def _blend_prediction(
    dataset: BlendDataset,
    mask: np.ndarray,
    *,
    weight: float,
    max_abs_shift: float | None,
) -> pd.DataFrame:
    block = dataset.frame.loc[mask]
    autonomous_q50 = block[f"{dataset.autonomous_model}__q50"].to_numpy(dtype=float)
    primary = block[dataset.primary_column].to_numpy(dtype=float)
    shift = weight * (primary - autonomous_q50)
    if max_abs_shift is not None:
        shift = np.clip(shift, -max_abs_shift, max_abs_shift)
    prediction = pd.DataFrame(index=block.index)
    baseline = pd.DataFrame(index=block.index)
    for quantile in QUANTILES:
        values = block[f"{dataset.autonomous_model}__{quantile}"].to_numpy(dtype=float)
        prediction[quantile] = values + shift
        baseline[quantile] = values
    return _prediction_frame(actual=block["actual"], prediction=prediction, baseline=baseline)


def _best_blend_weight(
    dataset: BlendDataset,
    mask: np.ndarray,
    *,
    step: float,
    max_abs_shift: float | None,
) -> float:
    weights = np.arange(0.0, 1.0 + step / 2.0, step, dtype=float)
    weights = np.unique(np.clip(np.r_[weights, 1.0], 0.0, 1.0))
    losses = []
    for weight in weights:
        frame = _blend_prediction(dataset, mask, weight=float(weight), max_abs_shift=max_abs_shift)
        losses.append(float(np.mean(np.abs(frame["actual"] - frame["q50"]))))
    return float(weights[int(np.argmin(losses))])


def _blend_cap(params: Mapping[str, Any]) -> float | None:
    unknown = sorted(set(params).difference({"max_abs_shift_eur_mwh"}))
    if unknown:
        raise ValueError(f"Parametres MKOnline inconnus: {unknown}.")
    value = params.get("max_abs_shift_eur_mwh")
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("max_abs_shift_eur_mwh doit etre positif ou null.")
    return result


def _run_blend(
    config: AuxiliaryLabConfig,
    model_config: ModelConfig,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = load_blend_dataset(config, model_config)
    train_mask = _phase_mask(dataset.split, dataset.frame.index, config.timezone, "train")
    validation_mask = _phase_mask(dataset.split, dataset.frame.index, config.timezone, "validation")
    test_mask = _phase_mask(dataset.split, dataset.frame.index, config.timezone, "test")
    step = float(model_config.options["weight_step"])
    trials = _parameter_trials(model_config)
    rows: list[dict[str, Any]] = []
    validation_frames: dict[str, pd.DataFrame] = {}
    weights: dict[str, float] = {}
    for number, params in enumerate(trials, start=1):
        trial_id = f"mkonline_{number:04d}"
        try:
            cap = _blend_cap(params)
            weight = _best_blend_weight(dataset, train_mask, step=step, max_abs_shift=cap)
            frame = _blend_prediction(dataset, validation_mask, weight=weight, max_abs_shift=cap)
            _, metric = _scored(frame, config=config)
            rows.append(
                {
                    "model": "mkonline_blend",
                    "configuration_id": trial_id,
                    "status": "complete",
                    "objective": config.objective,
                    "objective_value": metric[config.objective],
                    "parameters": json.dumps(_json_safe(params), sort_keys=True),
                    "calibrated_weight_mkonline": weight,
                    "n_training_rows": int(train_mask.sum()),
                    "error": "",
                }
            )
            validation_frames[trial_id] = frame
            weights[trial_id] = weight
        except Exception as exc:
            rows.append(
                {
                    "model": "mkonline_blend",
                    "configuration_id": trial_id,
                    "status": "failed",
                    "objective": config.objective,
                    "objective_value": np.inf,
                    "parameters": json.dumps(_json_safe(params), sort_keys=True),
                    "calibrated_weight_mkonline": np.nan,
                    "n_training_rows": int(train_mask.sum()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    leaderboard = pd.DataFrame(rows)
    successful = leaderboard.loc[leaderboard["status"] == "complete"]
    if successful.empty:
        details = "; ".join(leaderboard["error"].astype(str).head(3))
        raise AuxiliaryLabError(
            "Toutes les configurations du blend MKOnline ont echoue: " + details
        )
    selected_row = successful.sort_values(["objective_value", "configuration_id"]).iloc[0]
    selected_id = str(selected_row["configuration_id"])
    leaderboard["selected"] = leaderboard["configuration_id"].eq(selected_id)
    params = trials[int(selected_id.rsplit("_", 1)[1]) - 1]
    cap = _blend_cap(params)
    evaluation_mask = train_mask | validation_mask
    evaluation_weight = _best_blend_weight(dataset, evaluation_mask, step=step, max_abs_shift=cap)
    test_frame = _blend_prediction(dataset, test_mask, weight=evaluation_weight, max_abs_shift=cap)
    full_weight = _best_blend_weight(
        dataset,
        np.ones(len(dataset.frame), dtype=bool),
        step=step,
        max_abs_shift=cap,
    )
    model_dir = output / "models" / "mkonline_blend"
    _write_json(
        model_dir / "model.json",
        {
            "schema_version": 1,
            "model_kind": "mkonline_blend",
            "selected_configuration_id": selected_id,
            "parameters": params,
            "autonomous_model": dataset.autonomous_model,
            "primary_column": dataset.primary_column,
            "weight_step": step,
            "evaluation_weight_mkonline": evaluation_weight,
            "full_weight_mkonline": full_weight,
            "full_weight_autonomous": 1.0 - full_weight,
            "quantile_policy": "common additive q50 shift applied to autonomous q10/q50/q90",
        },
    )
    validation_frame = validation_frames[selected_id]
    predictions = pd.concat(
        [
            _prediction_records(validation_frame, config=config, model="mkonline_blend", phase="validation", configuration_id=selected_id),
            _prediction_records(test_frame, config=config, model="mkonline_blend", phase="test", configuration_id=selected_id),
        ],
        ignore_index=True,
    )
    metrics = pd.DataFrame(
        _metric_records(validation_frame, config=config, model="mkonline_blend", phase="validation")
        + _metric_records(test_frame, config=config, model="mkonline_blend", phase="test")
    )
    daily = pd.concat(
        [
            daily_metrics(apply_horizon(validation_frame, timezone=config.timezone, horizon_hours=config.split.horizon_hours), timezone=config.timezone).assign(model="mkonline_blend", phase="validation"),
            daily_metrics(apply_horizon(test_frame, timezone=config.timezone, horizon_hours=config.split.horizon_hours), timezone=config.timezone).assign(model="mkonline_blend", phase="test"),
        ],
        ignore_index=True,
    )
    return leaderboard, predictions, metrics, daily


def _kalman_config(params: Mapping[str, Any]) -> KalmanResidualConfig:
    allowed = {item.name for item in fields(KalmanResidualConfig)}
    unknown = sorted(set(params).difference(allowed))
    if unknown:
        raise ValueError(f"Parametres Kalman inconnus: {unknown}.")
    selected = dict(params)
    if "candidate_kinds" in selected:
        selected["candidate_kinds"] = tuple(selected["candidate_kinds"])
    result = KalmanResidualConfig(**selected)
    result.validate()
    return result


def _kalman_frame(
    dataset: KalmanDataset,
    prediction: pd.DataFrame,
    mask: np.ndarray,
) -> pd.DataFrame:
    index = dataset.history.index[mask]
    base = dataset.history.loc[index, [f"{dataset.upstream_model}__{q}" for q in QUANTILES]].copy()
    base.columns = list(QUANTILES)
    candidate = prediction.loc[index, [f"residual_kalman__{q}" for q in QUANTILES]].copy()
    candidate.columns = list(QUANTILES)
    return _prediction_frame(
        actual=dataset.history.loc[index, "actual"],
        prediction=candidate,
        baseline=base,
    )


def _final_kalman_coefficients(state_audit: pd.DataFrame) -> pd.DataFrame:
    """Expose final standardized exogenous states for interpretation, not causality."""

    columns = [
        "filter_kind",
        "group",
        "feature",
        "standardized_coefficient",
        "absolute_standardized_coefficient",
        "final_local_day",
        "interpretation",
    ]
    rows: list[dict[str, Any]] = []
    if state_audit.empty:
        return pd.DataFrame(columns=columns)
    for filter_kind, block in state_audit.groupby("filter_kind", sort=True):
        if str(filter_kind) not in EXOGENOUS_FILTER_GROUPS:
            continue
        final = block.iloc[-1]
        state = final.get("state_after")
        if not isinstance(state, Mapping):
            continue
        for feature, value in state.items():
            feature_name = str(feature)
            if not feature_name.startswith("covariate::"):
                continue
            coefficient = float(value)
            rows.append(
                {
                    "filter_kind": str(filter_kind),
                    "group": EXOGENOUS_FILTER_GROUPS[str(filter_kind)],
                    "feature": feature_name.removeprefix("covariate::"),
                    "standardized_coefficient": coefficient,
                    "absolute_standardized_coefficient": abs(coefficient),
                    "final_local_day": str(final["local_day"]),
                    "interpretation": (
                        "association conditionnelle du filtre; ne prouve pas une causalite"
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["absolute_standardized_coefficient", "filter_kind", "feature"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def _run_kalman(
    config: AuxiliaryLabConfig,
    model_config: ModelConfig,
    output: Path,
    *,
    history_override: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rolling_cache_root = (
        config.project_root
        / "runs"
        / "cache"
        / "kalman_rolling"
        / "auxiliary_lab"
        / config.experiment_id
    )
    dataset = load_kalman_dataset(
        config,
        model_config,
        history_override=history_override,
    )
    validation_mask = _phase_mask(dataset.split, dataset.history.index, config.timezone, "validation")
    test_mask = _phase_mask(dataset.split, dataset.history.index, config.timezone, "test")
    selection_mask = ~test_mask
    selection_history = dataset.history.loc[selection_mask]
    selection_covariates = dataset.covariates.loc[selection_mask]
    trials = _parameter_trials(model_config)
    rows: list[dict[str, Any]] = []
    validation_frames: dict[str, pd.DataFrame] = {}
    for number, params in enumerate(trials, start=1):
        trial_id = f"kalman_{number:04d}"
        try:
            selected_config = _kalman_config(params)
            replay = replay_kalman_overlay(
                selection_history.reset_index(),
                timezone=config.timezone,
                upstream_model=dataset.upstream_model,
                output_model="residual_kalman",
                evaluation_start_day=str(dataset.split.validation_days[0]),
                covariates=selection_covariates.reset_index().rename(
                    columns={"delivery_start_utc": "timestamp"}
                ),
                config=selected_config,
                covariate_config=dataset.covariate_config,
                training_lookback_days=model_config.options.get(
                    "training_lookback_days"
                ),
                rolling_refit_workers=int(
                    model_config.options.get("rolling_refit_workers", 1)
                ),
                rolling_refit_cache_dir=rolling_cache_root,
            )
            selection_validation_mask = _phase_mask(
                dataset.split,
                selection_history.index,
                config.timezone,
                "validation",
            )
            selection_dataset = KalmanDataset(
                history=selection_history,
                covariates=selection_covariates,
                split=dataset.split,
                upstream_model=dataset.upstream_model,
                covariate_config=dataset.covariate_config,
                coverage=dataset.coverage,
                source_audit=dataset.source_audit,
            )
            frame = _kalman_frame(
                selection_dataset,
                replay.predictions,
                selection_validation_mask,
            )
            _, metric = _scored(frame, config=config)
            rows.append(
                {
                    "model": "kalman",
                    "configuration_id": trial_id,
                    "status": "complete",
                    "objective": config.objective,
                    "objective_value": metric[config.objective],
                    "parameters": json.dumps(_json_safe(params), sort_keys=True),
                    "selected_filter_counts": json.dumps(replay.audit.get("selected_filter_counts", {}), sort_keys=True),
                    "candidate_feature_counts": json.dumps(
                        replay.audit.get("candidate_feature_counts", {}),
                        sort_keys=True,
                    ),
                    "covariate_feature_count": int(
                        len(dataset.covariate_config.feature_columns)
                    ),
                    "minimum_covariate_coverage": float(
                        dataset.coverage.loc[
                            dataset.coverage["column"].isin(
                                dataset.covariate_config.feature_columns
                            ),
                            "coverage",
                        ].min()
                    ),
                    "n_training_rows": int((~(validation_mask | test_mask)).sum()),
                    "error": "",
                }
            )
            validation_frames[trial_id] = frame
        except Exception as exc:
            rows.append(
                {
                    "model": "kalman",
                    "configuration_id": trial_id,
                    "status": "failed",
                    "objective": config.objective,
                    "objective_value": np.inf,
                    "parameters": json.dumps(_json_safe(params), sort_keys=True),
                    "selected_filter_counts": "{}",
                    "candidate_feature_counts": "{}",
                    "covariate_feature_count": int(
                        len(dataset.covariate_config.feature_columns)
                    ),
                    "minimum_covariate_coverage": float(
                        dataset.coverage.loc[
                            dataset.coverage["column"].isin(
                                dataset.covariate_config.feature_columns
                            ),
                            "coverage",
                        ].min()
                    ),
                    "n_training_rows": int((~(validation_mask | test_mask)).sum()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    leaderboard = pd.DataFrame(rows)
    successful = leaderboard.loc[leaderboard["status"] == "complete"]
    if successful.empty:
        details = "; ".join(leaderboard["error"].astype(str).head(3))
        raise AuxiliaryLabError(
            "Toutes les configurations Kalman ont echoue: " + details
        )
    selected_row = successful.sort_values(["objective_value", "configuration_id"]).iloc[0]
    selected_id = str(selected_row["configuration_id"])
    leaderboard["selected"] = leaderboard["configuration_id"].eq(selected_id)
    leaderboard["selection_protocol"] = (
        f"daily_refit_trailing_{model_config.options.get('training_lookback_days')}_days"
        if model_config.options.get("training_lookback_days") is not None
        else "causal_expanding_validation"
    )
    selected_params = trials[int(selected_id.rsplit("_", 1)[1]) - 1]
    training_lookback_days = model_config.options.get("training_lookback_days")
    final_evaluation_start_day = (
        dataset.split.test_days[0]
        if training_lookback_days is not None
        else dataset.split.validation_days[0]
    )
    replay = replay_kalman_overlay(
        dataset.history.reset_index(),
        timezone=config.timezone,
        upstream_model=dataset.upstream_model,
        output_model="residual_kalman",
        evaluation_start_day=str(final_evaluation_start_day),
        covariates=dataset.covariates.reset_index().rename(
            columns={"delivery_start_utc": "timestamp"}
        ),
        config=_kalman_config(selected_params),
        covariate_config=dataset.covariate_config,
        training_lookback_days=training_lookback_days,
        rolling_refit_workers=int(
            model_config.options.get("rolling_refit_workers", 1)
        ),
        rolling_refit_cache_dir=rolling_cache_root,
    )
    cache_audit = replay.audit.get("rolling_refit_cache", {})
    if isinstance(cache_audit, Mapping) and cache_audit.get("enabled"):
        print(
            f"[AUX-LAB] kalman cache: hits="
            f"{cache_audit.get('history_hits', 0)}, refits="
            f"{cache_audit.get('history_fitted_days', 0)}, future_hits="
            f"{cache_audit.get('future_hits', 0)}, future_refits="
            f"{cache_audit.get('future_fitted_days', 0)}",
            flush=True,
        )
    # Hyper-parameters are screened once on the causal validation replay.  The
    # selected configuration is then evaluated again with the exact operational
    # contract (for example a fresh D-365..D-1 fit for every delivery day).
    validation_frame = (
        validation_frames[selected_id]
        if training_lookback_days is not None
        else _kalman_frame(dataset, replay.predictions, validation_mask)
    )
    test_frame = _kalman_frame(dataset, replay.predictions, test_mask)
    model_dir = output / "models" / "kalman"
    _write_json(
        model_dir / "model.json",
        {
            "schema_version": 1,
            "model_kind": "kalman",
            "selected_configuration_id": selected_id,
            "parameters": asdict(_kalman_config(selected_params)),
            "covariates": dataset.covariate_config.to_dict(),
            "covariate_feature_count": int(
                len(dataset.covariate_config.feature_columns)
            ),
            "candidate_feature_counts": replay.audit.get(
                "candidate_feature_counts", {}
            ),
            "training_lookback_days": training_lookback_days,
            "rolling_refit_workers": int(
                model_config.options.get("rolling_refit_workers", 1)
            ),
            "rolling_refit_cache_policy": (
                "persistent_content_addressed_daily_fits"
                if training_lookback_days is not None
                else "not_applicable_expanding_replay"
            ),
            "selection_protocol": (
                f"daily_refit_trailing_{training_lookback_days}_days"
                if training_lookback_days is not None
                else "causal_expanding_validation"
            ),
            "final_evaluation_protocol": (
                (
                    f"validation_selection_and_sealed_test_daily_refit_"
                    f"trailing_{training_lookback_days}_days"
                )
                if training_lookback_days is not None
                else "causal_expanding_replay"
            ),
            "additional_sources": [
                {
                    "path": source.path,
                    "timestamp_column": source.timestamp_column,
                    "columns": dict(source.columns),
                    "origin_column": source.origin_column,
                    "revision_column": source.revision_column,
                    "cutoff_column": source.cutoff_column,
                    "cutoff_time": source.cutoff_time,
                    "sha256_at_training": _sha256(source.path),
                }
                for source in model_config.options["additional_sources"]
            ],
            "source_mapping_audit": [asdict(item) for item in dataset.source_audit],
            "upstream_model": dataset.upstream_model,
            "timezone": config.timezone,
            "serialization": "configuration plus causal replay; no unsafe pickle",
        },
    )
    replay.daily_audit.to_csv(model_dir / "kalman_daily_audit.csv", index=False)
    replay.state_audit.to_csv(model_dir / "kalman_state_audit.csv.gz", index=False, compression="gzip")
    coefficients = _final_kalman_coefficients(replay.state_audit)
    coefficients.to_csv(
        model_dir / "kalman_final_coefficients.csv", index=False
    )
    dataset.coverage.to_csv(
        model_dir / "kalman_covariate_coverage.csv", index=False
    )
    _write_json(model_dir / "kalman_replay_audit.json", dict(replay.audit))
    predictions = pd.concat(
        [
            _prediction_records(validation_frame, config=config, model="kalman", phase="validation", configuration_id=selected_id),
            _prediction_records(test_frame, config=config, model="kalman", phase="test", configuration_id=selected_id),
        ],
        ignore_index=True,
    )
    metrics = pd.DataFrame(
        _metric_records(validation_frame, config=config, model="kalman", phase="validation")
        + _metric_records(test_frame, config=config, model="kalman", phase="test")
    )
    daily = pd.concat(
        [
            daily_metrics(apply_horizon(validation_frame, timezone=config.timezone, horizon_hours=config.split.horizon_hours), timezone=config.timezone).assign(model="kalman", phase="validation"),
            daily_metrics(apply_horizon(test_frame, timezone=config.timezone, horizon_hours=config.split.horizon_hours), timezone=config.timezone).assign(model="kalman", phase="test"),
        ],
        ignore_index=True,
    )
    return leaderboard, predictions, metrics, daily


def _resolved_config(config: AuxiliaryLabConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "experiment_id": config.experiment_id,
        "source_run": str(config.source_run),
        "output_directory": str(config.output_directory),
        "timezone": config.timezone,
        "split": asdict(config.split),
        "objective": config.objective,
        "metrics": list(config.metrics),
        "random_seed": config.random_seed,
        "report": {"enabled": config.report_enabled, "embed_plotly": config.report_embed_plotly},
        "models": {
            name: {
                "enabled": model.enabled,
                "fixed_parameters": model.fixed_parameters,
                "parameter_grid": model.parameter_grid,
                **(
                    {
                        "upstream_model": model.options["upstream_model"],
                        "covariates": model.options["covariate_config"].to_dict(),
                        "additional_sources": model.options["additional_sources"],
                        "training_lookback_days": model.options[
                            "training_lookback_days"
                        ],
                        "rolling_refit_workers": model.options[
                            "rolling_refit_workers"
                        ],
                    }
                    if name == "kalman"
                    else dict(model.options)
                ),
            }
            for name, model in config.models.items()
        },
    }


def _publish_directory(staging: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Experience deja presente: {output}; utilisez --overwrite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous: Path | None = None
    if output.exists():
        previous = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
        os.replace(output, previous)
    try:
        os.replace(staging, output)
    except Exception:
        if previous is not None and previous.exists() and not output.exists():
            os.replace(previous, output)
        raise
    if previous is not None:
        shutil.rmtree(previous)


def train_experiment(
    config_or_path: AuxiliaryLabConfig | str | Path,
    *,
    models: Sequence[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Tune, evaluate and persist selected auxiliary models atomically."""

    config = (
        config_or_path
        if isinstance(config_or_path, AuxiliaryLabConfig)
        else load_lab_config(config_or_path)
    )
    selected_models = tuple(models or config.enabled_models)
    unknown = sorted(set(selected_models).difference(MODEL_NAMES))
    disabled = sorted(name for name in selected_models if not config.models[name].enabled)
    if unknown or disabled or not selected_models:
        raise ValueError(f"Selection de modeles invalide: unknown={unknown}, disabled={disabled}.")
    chain_residual_to_kalman = (
        "residual_corrector" in selected_models
        and "kalman" in selected_models
        and str(config.models["kalman"].options["upstream_model"])
        == "residual_corrected"
    )
    execution_models = selected_models
    if chain_residual_to_kalman:
        execution_models = tuple(
            ["residual_corrector"]
            + [name for name in selected_models if name != "residual_corrector"]
        )
    if (
        config.output_directory == config.source_run
        or config.output_directory in config.source_run.parents
        or config.source_run in config.output_directory.parents
    ):
        raise ValueError("La sortie experimentale doit etre disjointe de la source.")
    if config.output_directory.exists() and not overwrite:
        raise FileExistsError(
            f"Experience deja presente: {config.output_directory}; utilisez --overwrite."
        )
    source_hashes = _hashes(_source_files(config))
    config.output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{config.experiment_id}.tmp-",
            dir=config.output_directory.parent,
        )
    )
    try:
        shutil.copy2(config.source_path, staging / "input_config.yaml")
        _write_json(staging / "resolved_config.json", _resolved_config(config))
        all_leaderboards: list[pd.DataFrame] = []
        all_predictions: list[pd.DataFrame] = []
        all_metrics: list[pd.DataFrame] = []
        all_daily: list[pd.DataFrame] = []
        kalman_history: pd.DataFrame | None = None
        for name in execution_models:
            print(f"[AUX-LAB] {name}: calibration et evaluation...", flush=True)
            if name == "residual_corrector":
                leaderboard, predictions, metrics, daily = _run_residual(
                    config,
                    config.models[name],
                    staging,
                    build_kalman_history=chain_residual_to_kalman,
                )
                if chain_residual_to_kalman:
                    kalman_history = _read_indexed(
                        staging
                        / "models"
                        / "residual_corrector"
                        / "kalman_upstream_history.csv.gz",
                        timestamp_column="delivery_start_utc",
                    )
            elif name == "mkonline_blend":
                leaderboard, predictions, metrics, daily = _run_blend(
                    config, config.models[name], staging
                )
            elif name == "kalman":
                leaderboard, predictions, metrics, daily = _run_kalman(
                    config,
                    config.models[name],
                    staging,
                    history_override=kalman_history,
                )
            else:  # pragma: no cover - guarded by MODEL_NAMES
                raise AssertionError(name)
            all_leaderboards.append(leaderboard)
            all_predictions.append(predictions)
            all_metrics.append(metrics)
            all_daily.append(daily)
        leaderboard = pd.concat(all_leaderboards, ignore_index=True)
        predictions = pd.concat(all_predictions, ignore_index=True)
        metrics = pd.concat(all_metrics, ignore_index=True)
        daily = pd.concat(all_daily, ignore_index=True)
        for phase, phase_frame in predictions.groupby("phase", sort=True):
            supports = {
                model: tuple(
                    pd.to_datetime(block["delivery_start_utc"], utc=True).astype(str)
                )
                for model, block in phase_frame.groupby("model", sort=True)
            }
            if len({support for support in supports.values()}) > 1:
                bounds = {
                    model: (values[0], values[-1], len(values))
                    for model, values in supports.items()
                }
                raise AuxiliaryLabError(
                    f"Les modeles n'utilisent pas le meme support {phase}: {bounds}."
                )
        split_manifest: dict[str, Any] = {}
        first_model = execution_models[0]
        for phase, block in predictions.loc[
            predictions["model"] == first_model
        ].groupby("phase", sort=True):
            timeline = pd.DatetimeIndex(
                pd.to_datetime(block["delivery_start_utc"], utc=True)
            )
            split_manifest[str(phase)] = {
                "start_utc": timeline.min().isoformat(),
                "end_utc": timeline.max().isoformat(),
                "hours": int(len(timeline)),
                "local_days": int(
                    pd.Index(timeline.tz_convert(config.timezone).date).nunique()
                ),
            }
        _write_json(staging / "splits.json", split_manifest)
        _write_json(
            staging / "dataset_manifest.json",
            {
                "schema_version": 1,
                "source_run": str(config.source_run),
                "source_files_sha256": source_hashes,
                "timezone": config.timezone,
                "shared_validation_test_support": True,
                "residual_history_prefix": (
                    {
                        "path": config.models["residual_corrector"].options[
                            "prequential_bridge"
                        ]["history_prefix_path"],
                        "sha256": source_hashes[
                            str(
                                config.models["residual_corrector"].options[
                                    "prequential_bridge"
                                ]["history_prefix_path"].resolve()
                            )
                        ],
                        "origin_rule": (
                            "forecast_origin_utc == civil cutoff D-1 08:00"
                        ),
                    }
                    if config.models["residual_corrector"].options[
                        "prequential_bridge"
                    ]["history_prefix_path"] is not None
                    else None
                ),
                "kalman_upstream_source": (
                    "prequential_lab_residual_with_issued_evaluation_overlay"
                    if chain_residual_to_kalman
                    else "published_source_history"
                ),
                "horizon_hours": config.split.horizon_hours,
                "splits_path": "splits.json",
                "kalman_covariates": (
                    {
                        "contract": config.models[
                            "kalman"
                        ].options["covariate_config"].to_dict(),
                        "training_lookback_days": config.models[
                            "kalman"
                        ].options["training_lookback_days"],
                        "additional_sources": [
                            {
                                "path": source.path,
                                "timestamp_column": source.timestamp_column,
                                "columns": dict(source.columns),
                                "origin_column": source.origin_column,
                                "revision_column": source.revision_column,
                                "cutoff_column": source.cutoff_column,
                                "cutoff_time": source.cutoff_time,
                                "sha256": source_hashes[str(source.path.resolve())],
                            }
                            for source in config.models[
                                "kalman"
                            ].options["additional_sources"]
                        ],
                        "coverage_path": (
                            "models/kalman/kalman_covariate_coverage.csv"
                            if "kalman" in execution_models
                            else None
                        ),
                    }
                    if config.models["kalman"].enabled
                    else None
                ),
            },
        )
        leaderboard.to_csv(staging / "leaderboard.csv", index=False)
        predictions.to_csv(staging / "predictions.csv.gz", index=False, compression="gzip")
        metrics.to_csv(staging / "metrics.csv", index=False)
        daily.to_csv(staging / "metrics_by_day.csv", index=False)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "experiment_id": config.experiment_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(config.source_run),
            "source_files_sha256": source_hashes,
            "models": list(execution_models),
            "objective": config.objective,
            "timezone": config.timezone,
            "split": asdict(config.split),
            "selection_uses_test": False,
            "kalman_upstream_source": (
                "prequential_lab_residual_with_issued_evaluation_overlay"
                if chain_residual_to_kalman
                else "published_source_history"
            ),
            "storm_used_for_training_or_selection": False,
            "production_modified": False,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "packages": _package_versions(),
            },
        }
        _write_json(staging / "run_manifest.json", manifest)
        if config.report_enabled:
            importance_path = (
                staging
                / "models"
                / "residual_corrector"
                / "feature_importance.csv"
            )
            write_report(
                staging / "report.html",
                experiment_id=config.experiment_id,
                manifest=manifest,
                leaderboard=leaderboard,
                metrics=metrics,
                predictions=predictions,
                daily=daily,
                embed_plotly=config.report_embed_plotly,
                feature_importance=(
                    pd.read_csv(importance_path)
                    if importance_path.is_file()
                    else None
                ),
                kalman_coefficients=(
                    pd.read_csv(
                        staging
                        / "models"
                        / "kalman"
                        / "kalman_final_coefficients.csv"
                    )
                    if (
                        staging
                        / "models"
                        / "kalman"
                        / "kalman_final_coefficients.csv"
                    ).is_file()
                    else None
                ),
            )
        current_hashes = _hashes(_source_files(config))
        if current_hashes != source_hashes:
            raise AuxiliaryLabError("Les artefacts source ont change pendant l'experience.")
        artifacts = [path for path in staging.rglob("*") if path.is_file()]
        _write_json(
            staging / "checksums.json",
            {
                "algorithm": "sha256",
                "artifacts": {
                    path.relative_to(staging).as_posix(): _sha256(path)
                    for path in artifacts
                },
            },
        )
        _publish_directory(staging, config.output_directory, overwrite=overwrite)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return config.output_directory


def _read_experiment(path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    leaderboard = pd.read_csv(path / "leaderboard.csv")
    predictions = pd.read_csv(path / "predictions.csv.gz")
    predictions["delivery_start_utc"] = pd.to_datetime(predictions["delivery_start_utc"], utc=True)
    metrics = pd.read_csv(path / "metrics.csv")
    return manifest, leaderboard, predictions, metrics


def evaluate_run(run_directory: str | Path, *, output_directory: str | Path | None = None) -> Path:
    """Recompute metrics from frozen predictions without loading a model."""

    run = Path(run_directory).expanduser().resolve()
    manifest, leaderboard, predictions, _ = _read_experiment(run)
    timezone_name = str(manifest["timezone"])
    requested = tuple(dict.fromkeys(("mae", "rmse", "bias", *SUPPORTED_METRICS)))
    metric_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    for (model, phase), raw in predictions.groupby(["model", "phase"], sort=True):
        frame = raw.set_index("delivery_start_utc")
        candidate = metrics_row(frame, requested=requested)
        baseline_frame = frame.rename(
            columns={"q10": "candidate_q10", "q50": "candidate_q50", "q90": "candidate_q90", "baseline_q10": "q10", "baseline_q50": "q50", "baseline_q90": "q90"}
        )
        baseline = metrics_row(baseline_frame, requested=requested)
        period = {
            "start_utc": frame.index.min().isoformat(),
            "end_utc": frame.index.max().isoformat(),
            "n_local_days": int(
                pd.Index(frame.index.tz_convert(timezone_name).date).nunique()
            ),
        }
        metric_rows.extend(
            [
                {
                    "model": model,
                    "phase": phase,
                    "role": "candidate",
                    **period,
                    **candidate,
                },
                {
                    "model": model,
                    "phase": phase,
                    "role": "baseline",
                    **period,
                    **baseline,
                },
            ]
        )
        daily_rows.append(daily_metrics(frame, timezone=timezone_name).assign(model=model, phase=phase))
    metrics = pd.DataFrame(metric_rows)
    daily = pd.concat(daily_rows, ignore_index=True)
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else run / "evaluation"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    daily.to_csv(output / "metrics_by_day.csv", index=False)
    write_report(
        output / "report.html",
        experiment_id=str(manifest["experiment_id"]),
        manifest=manifest,
        leaderboard=leaderboard,
        metrics=metrics,
        predictions=predictions,
        daily=daily,
        embed_plotly=True,
        feature_importance=(
            pd.read_csv(
                run / "models" / "residual_corrector" / "feature_importance.csv"
            )
            if (
                run / "models" / "residual_corrector" / "feature_importance.csv"
            ).is_file()
            else None
        ),
        kalman_coefficients=(
            pd.read_csv(
                run / "models" / "kalman" / "kalman_final_coefficients.csv"
            )
            if (
                run / "models" / "kalman" / "kalman_final_coefficients.csv"
            ).is_file()
            else None
        ),
    )
    return output


def compare_runs(
    runs: Sequence[str | Path],
    *,
    output_directory: str | Path,
) -> Path:
    """Create a cross-experiment table and HTML report from frozen outputs."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    leaderboards: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    daily: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for raw_run in runs:
        run = Path(raw_run).expanduser().resolve()
        manifest, leaderboard, prediction, metrics = _read_experiment(run)
        experiment = str(manifest["experiment_id"])
        rows.append(
            metrics.assign(
                experiment_id=experiment,
                model=experiment + " / " + metrics["model"].astype(str),
            )
        )
        leaderboards.append(
            leaderboard.assign(
                experiment_id=experiment,
                model=experiment + " / " + leaderboard["model"].astype(str),
            )
        )
        predictions.append(prediction.assign(model=experiment + " / " + prediction["model"].astype(str)))
        day_path = run / "metrics_by_day.csv"
        day_frame = pd.read_csv(day_path)
        daily.append(
            day_frame.assign(
                model=experiment + " / " + day_frame["model"].astype(str)
            )
        )
        manifests.append(manifest)
    combined_metrics = pd.concat(rows, ignore_index=True)
    combined_leaderboard = pd.concat(leaderboards, ignore_index=True)
    combined_predictions = pd.concat(predictions, ignore_index=True)
    combined_daily = pd.concat(daily, ignore_index=True)
    combined_metrics.to_csv(output / "comparison_metrics.csv", index=False)
    combined_leaderboard.to_csv(output / "comparison_leaderboard.csv", index=False)
    manifest = {
        "experiment_id": "comparison",
        "source_run": "multiple frozen experiments",
        "objective": manifests[0].get("objective", "mae") if manifests else "mae",
        "runs": [str(Path(item).resolve()) for item in runs],
    }
    write_report(
        output / "comparison_report.html",
        experiment_id="comparison",
        manifest=manifest,
        leaderboard=combined_leaderboard,
        metrics=combined_metrics,
        predictions=combined_predictions,
        daily=combined_daily,
        embed_plotly=True,
        feature_importance=None,
    )
    return output


def _load_artifact_manifest(artifact: Path) -> tuple[Path, dict[str, Any]]:
    directory = artifact if artifact.is_dir() else artifact.parent
    manifest_path = directory / "manifest.json"
    if artifact.name == "model.json":
        manifest_path = artifact
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuxiliaryLabError(f"Manifest d'artefact invalide: {manifest_path}.")
    return directory, value


def _future_source(source_run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasts = sorted(source_run.glob("forecast_hourly_*.csv"))
    if len(forecasts) != 1:
        raise AuxiliaryLabError(f"{source_run}: un unique forecast_hourly_*.csv est requis.")
    forecast = _read_indexed(forecasts[0], timestamp_column="delivery_start_utc")
    covariates = _read_indexed(
        source_run / "inputs" / "model_covariates_with_future.csv.gz",
        timestamp_column="timestamp",
    )
    future_covariates = covariates.loc[covariates.index.intersection(forecast.index, sort=False)]
    if not future_covariates.index.equals(forecast.index):
        raise AuxiliaryLabError("Les covariates futures ne couvrent pas exactement le forecast.")
    return forecast, future_covariates


def predict_artifact(
    artifact: str | Path,
    *,
    source_run: str | Path,
    output_path: str | Path,
) -> Path:
    """Load a selected lab artefact and infer on an existing run's future day."""

    artifact_path = Path(artifact).expanduser().resolve()
    directory, manifest = _load_artifact_manifest(artifact_path)
    source = Path(source_run).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    kind = str(manifest.get("model_kind", ""))
    forecast, future_covariates = _future_source(source)
    if kind == "residual_corrector":
        model_path = (directory / str(manifest["artifact"])).resolve()
        if not model_path.is_relative_to(directory.resolve()):
            raise AuxiliaryLabError("Le chemin du modele residuel sort de son bundle.")
        if _sha256(model_path) != str(manifest["artifact_sha256"]):
            raise AuxiliaryLabError("SHA-256 du modele residuel invalide.")
        model = joblib.load(model_path)
        features = future_covariates.loc[:, list(manifest["feature_columns"])]
        base_model = str(manifest["base_model"])
        base = forecast.loc[:, [f"{base_model}__{q}" for q in QUANTILES]].copy()
        base.columns = list(QUANTILES)
        experts = forecast.loc[:, [f"{name}__{q}" for name in manifest["expert_models"] for q in QUANTILES]].copy()
        prediction = model.predict(features, base, experts)
    elif kind == "mkonline_blend":
        autonomous = str(manifest["autonomous_model"])
        primary = str(manifest["primary_column"])
        weight = float(manifest["full_weight_mkonline"])
        cap = manifest.get("parameters", {}).get("max_abs_shift_eur_mwh")
        base = forecast.loc[:, [f"{autonomous}__{q}" for q in QUANTILES]].copy()
        base.columns = list(QUANTILES)
        shift = weight * (
            pd.to_numeric(forecast[primary], errors="raise").to_numpy(dtype=float)
            - base["q50"].to_numpy(dtype=float)
        )
        if cap is not None:
            shift = np.clip(shift, -float(cap), float(cap))
        prediction = base.add(shift, axis=0)
    elif kind == "kalman":
        statistics = pd.read_csv(source / "statistics_history_hourly.csv.gz")
        source_forecast_path = next(iter(sorted(source.glob("forecast_hourly_*.csv"))))
        source_forecast = pd.read_csv(source_forecast_path)
        base_covariates = _read_indexed(
            source / "inputs" / "model_covariates_with_future.csv.gz",
            timestamp_column="timestamp",
        )
        configured_sources = tuple(
            KalmanCovariateSource(
                path=Path(str(item["path"])).expanduser().resolve(),
                timestamp_column=str(item["timestamp_column"]),
                columns={
                    str(column): str(alias)
                    for column, alias in item["columns"].items()
                },
                origin_column=str(item["origin_column"]),
                revision_column=(
                    str(item["revision_column"])
                    if item.get("revision_column") is not None
                    else None
                ),
                cutoff_column=(
                    str(item["cutoff_column"])
                    if item.get("cutoff_column") is not None
                    else None
                ),
                cutoff_time=str(item.get("cutoff_time", "08:00")),
            )
            for item in manifest.get("additional_sources", [])
        )
        combined_covariates, _ = join_kalman_additional_sources(
            base_covariates,
            configured_sources,
            timezone=str(manifest["timezone"]),
        )
        covariates = combined_covariates.reset_index().rename(
            columns={"delivery_start_utc": "timestamp"}
        )
        covariate_config = KalmanCovariateConfig.from_mapping(
            manifest.get("covariates")
        )
        local_day = forecast.index[0].tz_convert(str(manifest["timezone"])).date()
        view = build_operational_kalman_view(
            statistics=statistics,
            source_forecast=source_forecast,
            covariates=covariates,
            timezone=str(manifest["timezone"]),
            delivery_day=local_day,
            config=_kalman_config(manifest["parameters"]),
            covariate_config=covariate_config,
            upstream_model=str(manifest["upstream_model"]),
            training_lookback_days=manifest.get("training_lookback_days"),
            rolling_refit_workers=int(manifest.get("rolling_refit_workers", 1)),
            rolling_refit_cache_dir=(
                output.parent
                / ".kalman_rolling_cache"
                / "auxiliary_lab_inference"
            ),
        )
        prediction = view.forecast.set_index("delivery_start_utc")
        prediction.index = pd.to_datetime(prediction.index, utc=True)
        prediction = prediction.loc[:, [f"residual_kalman__{q}" for q in QUANTILES]]
        prediction.columns = list(QUANTILES)
    else:
        raise AuxiliaryLabError(f"Type d'artefact inconnu: {kind!r}.")
    result = prediction.copy()
    result.index.name = "delivery_start_utc"
    result = result.reset_index()
    result.insert(1, "model_kind", kind)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output


__all__ = [
    "AuxiliaryLabError",
    "compare_runs",
    "evaluate_run",
    "predict_artifact",
    "train_experiment",
]
