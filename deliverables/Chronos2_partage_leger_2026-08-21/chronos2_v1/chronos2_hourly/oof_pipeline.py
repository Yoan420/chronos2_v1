"""Bounded, leakage-aware orchestration for the hourly supervised experts.

This module does not import or call the Chronos runtime.  Chronos-2 forecasts
are an external input and must already be out-of-fold, timestamp-aligned and
accompanied by their forecast-origin timestamps.  The orchestrator creates
purged expanding-window OOF predictions for LEAR and CatBoost, learns convex
stacking weights, refits both supervised experts on all available history and
finally produces one forecast for every physical hour of a 23/24/25-hour local
delivery day.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .hourly_contract import (
    HourlyTargetContractError,
    build_delivery_metadata,
    local_delivery_day_index,
)
from .models import (
    HourlyCatBoost,
    HourlyLEAR,
    LeakageRiskError,
    NonNegativeOOFEnsemble,
    ResidualCorrector,
    purged_expanding_splits,
)


class QuantileForecaster(Protocol):
    """Minimal interface required from a supervised expert."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileForecaster": ...

    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...


ForecasterFactory = Callable[[], QuantileForecaster]
ResidualCorrectorFactory = Callable[[], ResidualCorrector]


@dataclass(frozen=True)
class HourlyOOFConfig:
    """Finite time-split and stacking settings for one training run."""

    n_splits: int = 5
    min_train_size: int | None = None
    test_size: int | None = None
    gap: int = 24
    split_on_delivery_days: bool = False
    min_train_days: int | None = None
    test_days: int | None = None
    gap_days: int = 1
    evaluation_days: int | None = 365
    ensemble_minimum_rows: int = 48
    ensemble_weight_l2: float = 1e-6
    timezone: str = "Europe/Paris"
    require_single_future_day: bool = True

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError("n_splits doit être >= 1.")
        if self.min_train_size is not None and self.min_train_size < 2:
            raise ValueError("min_train_size doit être >= 2.")
        if self.test_size is not None and self.test_size < 1:
            raise ValueError("test_size doit être >= 1.")
        if self.gap < 0:
            raise ValueError("gap doit être positif ou nul.")
        if self.min_train_days is not None and self.min_train_days < 1:
            raise ValueError("min_train_days doit être >= 1.")
        if self.test_days is not None and self.test_days < 1:
            raise ValueError("test_days doit être >= 1.")
        if self.gap_days < 0:
            raise ValueError("gap_days doit être positif ou nul.")
        if self.evaluation_days is not None and self.evaluation_days < 1:
            raise ValueError("evaluation_days doit être >= 1 ou None.")
        if self.ensemble_minimum_rows < 2:
            raise ValueError("ensemble_minimum_rows doit être >= 2.")


@dataclass(frozen=True)
class HourlyOOFTrainingResult:
    """Auditable outputs of the OOF and final-refit stages."""

    oof_predictions: pd.DataFrame = field(repr=False)
    ensemble_oof_predictions: pd.DataFrame = field(repr=False)
    fold_id: pd.Series = field(repr=False)
    metrics: pd.DataFrame
    ensemble_weights: pd.Series
    diagnostics: dict[str, Any]
    residual_oof_predictions: pd.DataFrame | None = field(
        default=None,
        repr=False,
    )


@dataclass(frozen=True)
class HourlyForecastResult:
    """Final hourly forecast and the expert-level audit trail."""

    predictions: pd.DataFrame
    expert_predictions: pd.DataFrame = field(repr=False)
    delivery_metadata: pd.DataFrame
    diagnostics: dict[str, Any]


def _require_hourly_index(
    frame: pd.DataFrame,
    *,
    name: str,
) -> pd.DatetimeIndex:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} doit être un pandas.DataFrame.")
    if frame.empty:
        raise ValueError(f"{name} est vide.")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} doit avoir un DatetimeIndex.")
    index = frame.index
    if index.tz is None:
        raise HourlyTargetContractError(
            f"{name}: l'index doit être timezone-aware."
        )
    if index.has_duplicates:
        raise HourlyTargetContractError(
            f"{name}: les timestamps de livraison doivent être uniques."
        )
    if not index.is_monotonic_increasing:
        raise HourlyTargetContractError(
            f"{name}: les timestamps doivent être triés chronologiquement."
        )
    utc = index.tz_convert("UTC")
    aligned = (
        (utc.minute == 0)
        & (utc.second == 0)
        & (utc.microsecond == 0)
        & (utc.nanosecond == 0)
    )
    if not bool(np.all(aligned)):
        raise HourlyTargetContractError(
            f"{name}: tous les timestamps doivent être alignés sur l'heure."
        )
    return index


def purged_daily_expanding_splits(
    delivery_index: pd.DatetimeIndex,
    *,
    n_splits: int = 5,
    min_train_days: int | None = None,
    test_days: int | None = None,
    gap_days: int = 1,
    timezone: str = "Europe/Paris",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding OOF splits on complete local delivery days.

    Unlike row-based splitting, no 23/24/25-hour delivery day can be shared by
    train and test or cut at a fold boundary.  The purge is expressed in local
    calendar days and the input must contain a consecutive sequence of whole
    delivery days.
    """

    if not isinstance(delivery_index, pd.DatetimeIndex):
        raise TypeError("delivery_index doit être un pandas.DatetimeIndex.")
    if delivery_index.tz is None:
        raise HourlyTargetContractError(
            "delivery_index doit être timezone-aware."
        )
    if delivery_index.empty or delivery_index.has_duplicates:
        raise HourlyTargetContractError(
            "delivery_index doit être non vide et sans doublon."
        )
    if not delivery_index.is_monotonic_increasing:
        raise HourlyTargetContractError(
            "delivery_index doit être trié chronologiquement."
        )
    if n_splits < 1 or gap_days < 0:
        raise ValueError("n_splits doit être >= 1 et gap_days >= 0.")

    local = delivery_index.tz_convert(timezone)
    local_dates = pd.Index(local.date)
    unique_days = local_dates.unique().tolist()
    if not unique_days:
        raise ValueError("Aucune journée de livraison.")

    # Each date must be represented by its exact canonical set of physical
    # hours.  This validates normal as well as spring/autumn DST days.
    utc_ns = delivery_index.tz_convert("UTC").asi8
    for delivery_date in unique_days:
        positions = np.flatnonzero(local_dates == delivery_date)
        observed = utc_ns[positions]
        expected = local_delivery_day_index(
            delivery_date, timezone=timezone
        ).asi8
        if not np.array_equal(observed, expected):
            raise HourlyTargetContractError(
                "Le split journalier exige des journées complètes; "
                f"{delivery_date} contient {len(observed)} heure(s), "
                f"attendu {len(expected)}."
            )

    expected_dates = pd.date_range(
        start=pd.Timestamp(unique_days[0]),
        periods=len(unique_days),
        freq="D",
    ).date.tolist()
    if unique_days != expected_dates:
        raise HourlyTargetContractError(
            "Les journées de livraison doivent être calendaires et consécutives."
        )

    n_days = len(unique_days)
    if test_days is None:
        test_days = max(1, n_days // (n_splits + 1))
    if min_train_days is None:
        min_train_days = n_days - n_splits * test_days - gap_days
    if min_train_days < 1 or test_days < 1:
        raise ValueError("min_train_days et test_days doivent être positifs.")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        train_end_day = min_train_days + fold * test_days
        test_start_day = train_end_day + gap_days
        test_end_day = min(test_start_day + test_days, n_days)
        if train_end_day <= 0 or test_start_day >= test_end_day:
            raise ValueError(
                "Paramètres journaliers incompatibles avec l'historique."
            )
        train_days = set(unique_days[:train_end_day])
        test_days_set = set(unique_days[test_start_day:test_end_day])
        train = np.flatnonzero(local_dates.isin(train_days))
        test = np.flatnonzero(local_dates.isin(test_days_set))
        if not len(train) or not len(test):
            raise RuntimeError("Split journalier interne vide.")
        splits.append((train, test))
    return splits


def _final_delivery_day_mask(
    delivery_index: pd.DatetimeIndex,
    *,
    evaluation_days: int,
    timezone: str,
) -> tuple[np.ndarray, list[object]]:
    """Return the exact final complete local days used as sealed holdout."""

    local_dates = pd.Index(delivery_index.tz_convert(timezone).date)
    unique_days = local_dates.unique().tolist()
    if len(unique_days) < evaluation_days:
        raise ValueError(
            f"Historique trop court pour un holdout de {evaluation_days} jours."
        )
    selected = unique_days[-evaluation_days:]
    mask = np.asarray(local_dates.isin(selected), dtype=bool)
    utc_ns = delivery_index.tz_convert("UTC").asi8
    for delivery_date in selected:
        observed = utc_ns[local_dates == delivery_date]
        expected = local_delivery_day_index(
            delivery_date, timezone=timezone
        ).asi8
        if not np.array_equal(observed, expected):
            raise HourlyTargetContractError(
                "Le holdout final doit contenir des journées complètes; "
                f"journée incomplète: {delivery_date}."
            )
    return mask, selected


def _same_instants(left: pd.Index, right: pd.Index) -> bool:
    if not isinstance(left, pd.DatetimeIndex) or not isinstance(
        right, pd.DatetimeIndex
    ):
        return False
    if left.tz is None or right.tz is None or len(left) != len(right):
        return False
    return bool(
        np.array_equal(
            left.tz_convert("UTC").asi8,
            right.tz_convert("UTC").asi8,
        )
    )


def _coerce_target(
    target: pd.Series | Sequence[float] | np.ndarray,
    index: pd.DatetimeIndex,
) -> pd.Series:
    raw = np.asarray(target)
    if raw.ndim != 1 or len(raw) != len(index):
        raise ValueError("y doit être unidimensionnel et aligné avec X.")
    return pd.Series(
        pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float),
        index=index,
        name="target",
    )


def _normalize_chronos_quantiles(
    values: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
    *,
    name: str,
    allow_missing: bool,
) -> pd.DataFrame:
    """Validate q10/q50/q90 without silently reindexing external forecasts."""

    if not isinstance(values, pd.DataFrame):
        raise TypeError(f"{name} doit être un pandas.DataFrame.")
    if not _same_instants(values.index, expected_index):
        raise LeakageRiskError(
            f"{name}: l'index doit être strictement aligné, dans le même ordre, "
            "avec les heures de livraison."
        )
    if {"q10", "q50", "q90"}.issubset(values.columns):
        columns = ["q10", "q50", "q90"]
    elif {
        "chronos2__q10",
        "chronos2__q50",
        "chronos2__q90",
    }.issubset(values.columns):
        columns = ["chronos2__q10", "chronos2__q50", "chronos2__q90"]
    else:
        raise ValueError(
            f"{name} doit contenir q10/q50/q90 (ou chronos2__q10/q50/q90)."
        )
    work = values.loc[:, columns].apply(pd.to_numeric, errors="coerce").copy()
    work.columns = ["q10", "q50", "q90"]
    work.index = expected_index
    array = work.to_numpy(dtype=float)
    if np.isinf(array).any():
        raise ValueError(f"{name} contient des valeurs infinies.")
    if not allow_missing and not np.isfinite(array).all():
        raise ValueError(f"{name} contient des prévisions manquantes.")
    finite_rows = np.isfinite(array).all(axis=1)
    if finite_rows.any():
        ordered = array[finite_rows]
        if np.any(ordered[:, 0] > ordered[:, 1]) or np.any(
            ordered[:, 1] > ordered[:, 2]
        ):
            raise ValueError(f"{name}: quantiles croisés détectés.")
    return work


def _normalize_origins(
    origins: pd.Series,
    expected_index: pd.DatetimeIndex,
    *,
    name: str,
    required_mask: np.ndarray,
) -> pd.Series:
    if not isinstance(origins, pd.Series):
        raise TypeError(
            f"{name} doit être une Series indexée par heure de livraison."
        )
    if not _same_instants(origins.index, expected_index):
        raise LeakageRiskError(
            f"{name}: index non aligné avec les heures de livraison."
        )
    parsed = pd.to_datetime(origins.to_numpy(), utc=True, errors="coerce")
    if len(parsed) != len(expected_index):
        raise ValueError(f"{name}: longueur incohérente.")
    required = np.asarray(required_mask, dtype=bool)
    if required.shape != (len(expected_index),):
        raise ValueError("Masque de disponibilité interne incohérent.")
    if pd.isna(parsed[required]).any():
        raise LeakageRiskError(
            f"{name}: origin timestamp manquant pour une prévision utilisée."
        )
    delivery_ns = expected_index.tz_convert("UTC").asi8
    origin_ns = parsed.asi8
    if np.any(origin_ns[required] >= delivery_ns[required]):
        raise LeakageRiskError(
            f"{name}: chaque prévision doit être produite strictement avant "
            "le début de livraison."
        )
    return pd.Series(parsed, index=expected_index, name=name)


def _prefixed(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    expected = ["q10", "q50", "q90"]
    missing = [column for column in expected if column not in predictions]
    if missing:
        raise RuntimeError(
            f"{model_name} n'a pas produit les quantiles attendus: {missing}."
        )
    result = predictions.loc[:, expected].copy()
    result.columns = [f"{model_name}__{column}" for column in expected]
    return result


def _mae_diagnostics(
    target: pd.Series,
    predictions: pd.DataFrame,
    *,
    expected_mask: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected = np.asarray(expected_mask, dtype=bool)
    target_values = target.to_numpy(dtype=float)
    target_available = expected & np.isfinite(target_values)
    model_names = [
        column.removesuffix("__q50")
        for column in predictions.columns
        if column.endswith("__q50")
    ]
    for model_name in model_names:
        values = pd.to_numeric(
            predictions[f"{model_name}__q50"], errors="coerce"
        ).to_numpy(dtype=float)
        prediction_available = expected & np.isfinite(values)
        scored = prediction_available & np.isfinite(target_values)
        rows.append(
            {
                "model": model_name,
                "mae": (
                    float(np.mean(np.abs(target_values[scored] - values[scored])))
                    if scored.any()
                    else float("nan")
                ),
                "n_scored": int(scored.sum()),
                "n_expected": int(expected.sum()),
                "prediction_coverage": (
                    float(prediction_available.sum() / expected.sum())
                    if expected.any()
                    else float("nan")
                ),
                "score_coverage": (
                    float(scored.sum() / target_available.sum())
                    if target_available.any()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).set_index("model")


class HourlyOOFPipeline:
    """Train OOF experts, convex stacker and final full-history experts."""

    def __init__(
        self,
        *,
        config: HourlyOOFConfig | None = None,
        lear_factory: ForecasterFactory | None = None,
        catboost_factory: ForecasterFactory | None = None,
        residual_corrector_factory: ResidualCorrectorFactory | None = None,
        residual_base_model: str = "chronos2",
    ) -> None:
        self.config = config or HourlyOOFConfig()
        self.lear_factory = lear_factory or (lambda: HourlyLEAR())
        self.catboost_factory = catboost_factory or (
            lambda: HourlyCatBoost(backend="auto")
        )
        if residual_base_model not in {"chronos2", "ensemble"}:
            raise ValueError(
                "residual_base_model doit valoir 'chronos2' ou 'ensemble'."
            )
        self.residual_corrector_factory = residual_corrector_factory
        self.residual_base_model = residual_base_model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[float] | np.ndarray,
        *,
        chronos_oof: pd.DataFrame,
        chronos_origin: pd.Series,
        chronos_is_oof: bool = False,
    ) -> "HourlyOOFPipeline":
        """Fit the bounded OOF stack and refit experts on full history.

        With a sealed holdout, ``chronos_is_oof=True`` also certifies that the
        external Chronos forecasts for that holdout came from a model frozen
        before its first delivery day; this provenance cannot be reconstructed
        from forecast values alone.
        """

        index = _require_hourly_index(X, name="X")
        if not chronos_is_oof:
            raise LeakageRiskError(
                "Les prédictions Chronos doivent être explicitement certifiées "
                "out-of-fold avec chronos_is_oof=True (et, pour un holdout "
                "final, issues d'un modèle gelé avant ce holdout)."
            )
        target = _coerce_target(y, index)
        chronos = _normalize_chronos_quantiles(
            chronos_oof,
            index,
            name="chronos_oof",
            allow_missing=True,
        )
        if self.config.split_on_delivery_days:
            splits = purged_daily_expanding_splits(
                index,
                n_splits=self.config.n_splits,
                min_train_days=self.config.min_train_days,
                test_days=self.config.test_days,
                gap_days=self.config.gap_days,
                timezone=self.config.timezone,
            )
        else:
            splits = purged_expanding_splits(
                len(X),
                n_splits=self.config.n_splits,
                min_train_size=self.config.min_train_size,
                test_size=self.config.test_size,
                gap=self.config.gap,
            )

        evaluation_mask = np.zeros(len(X), dtype=bool)
        evaluation_dates: list[object] = []
        if self.config.evaluation_days is not None:
            evaluation_mask, evaluation_dates = _final_delivery_day_mask(
                index,
                evaluation_days=self.config.evaluation_days,
                timezone=self.config.timezone,
            )
            if not target.iloc[evaluation_mask].notna().all():
                raise ValueError(
                    "Toutes les cibles du holdout final doivent être observées; "
                    "aucune imputation n'est autorisée pour la MAE publiée."
                )

        expert_columns = [
            f"{model}__{quantile}"
            for model in ("lear", "catboost", "chronos2")
            for quantile in ("q10", "q50", "q90")
        ]
        oof = pd.DataFrame(np.nan, index=index, columns=expert_columns, dtype=float)
        fold_id = pd.Series(pd.NA, index=index, dtype="Int64", name="fold_id")
        expected_mask = np.zeros(len(X), dtype=bool)

        for fold, (train_positions, test_positions) in enumerate(splits, start=1):
            # Once the sealed evaluation period starts, no later OOF fold may
            # train on an earlier target from that same holdout.  The purge is
            # applied relative to the first evaluation row/day, so altering a
            # holdout target cannot change any published holdout prediction.
            if evaluation_mask[test_positions].any():
                first_evaluation_position = int(np.flatnonzero(evaluation_mask)[0])
                if self.config.split_on_delivery_days:
                    evaluation_start = pd.Timestamp(evaluation_dates[0])
                    train_cutoff = (
                        evaluation_start
                        - pd.Timedelta(days=self.config.gap_days)
                    ).date()
                    local_train_dates = pd.Index(
                        index[train_positions]
                        .tz_convert(self.config.timezone)
                        .date
                    )
                    train_positions = train_positions[
                        np.asarray(local_train_dates < train_cutoff)
                    ]
                else:
                    train_cutoff = max(
                        0, first_evaluation_position - self.config.gap
                    )
                    train_positions = train_positions[
                        train_positions < train_cutoff
                    ]
                if not len(train_positions):
                    raise ValueError(
                        "Le holdout et sa purge ne laissent aucun train OOF."
                    )
            if train_positions.max() >= test_positions.min():
                raise LeakageRiskError(
                    "Split OOF non causal: le train chevauche le test."
                )
            if expected_mask[test_positions].any():
                raise RuntimeError("Une heure a été affectée à plusieurs folds OOF.")
            train_target = target.iloc[train_positions]
            if int(train_target.notna().sum()) < 2:
                raise ValueError(
                    f"Fold {fold}: moins de deux cibles observées au train."
                )
            lear = self.lear_factory()
            catboost = self.catboost_factory()
            lear.fit(X.iloc[train_positions], train_target)
            catboost.fit(X.iloc[train_positions], train_target)
            learner_predictions = _prefixed(
                lear.predict(X.iloc[test_positions]), "lear"
            )
            catboost_predictions = _prefixed(
                catboost.predict(X.iloc[test_positions]), "catboost"
            )
            oof.iloc[
                test_positions,
                [oof.columns.get_loc(column) for column in learner_predictions.columns],
            ] = learner_predictions.to_numpy(dtype=float)
            oof.iloc[
                test_positions,
                [
                    oof.columns.get_loc(column)
                    for column in catboost_predictions.columns
                ],
            ] = catboost_predictions.to_numpy(dtype=float)
            expected_mask[test_positions] = True
            fold_id.iloc[test_positions] = fold

        if self.config.evaluation_days is not None and not np.all(
            expected_mask[evaluation_mask]
        ):
            missing_hours = int((~expected_mask & evaluation_mask).sum())
            raise ValueError(
                "Les splits OOF ne couvrent pas intégralement le holdout final: "
                f"{missing_hours} heure(s) manquante(s). Ajustez n_splits, "
                "min_train/test_size ou min_train_days/test_days."
            )

        chronos_array = chronos.to_numpy(dtype=float)
        if not np.isfinite(chronos_array[expected_mask]).all():
            raise LeakageRiskError(
                "Chronos-2 ne couvre pas intégralement les heures des folds OOF."
            )
        oof.loc[:, ["chronos2__q10", "chronos2__q50", "chronos2__q90"]] = (
            chronos_array
        )
        origin = _normalize_origins(
            chronos_origin,
            index,
            name="chronos_origin",
            required_mask=expected_mask,
        )

        all_oof_common = expected_mask & target.notna().to_numpy()
        ensemble_fit_mask = all_oof_common & ~evaluation_mask
        if self.config.evaluation_days is None:
            ensemble_fit_mask = all_oof_common
        if int(ensemble_fit_mask.sum()) < self.config.ensemble_minimum_rows:
            raise ValueError(
                "Pas assez de lignes OOF communes pour le stacker: "
                f"{int(ensemble_fit_mask.sum())} < "
                f"{self.config.ensemble_minimum_rows}."
            )
        oof_ensemble_fit = oof.loc[ensemble_fit_mask]
        self.ensemble_ = NonNegativeOOFEnsemble(
            minimum_rows=self.config.ensemble_minimum_rows,
            weight_l2=self.config.ensemble_weight_l2,
        ).fit(
            oof_ensemble_fit,
            target.loc[ensemble_fit_mask],
            is_oof=True,
            prediction_origin=origin.loc[ensemble_fit_mask],
            delivery_start=index[ensemble_fit_mask],
        )
        ensemble_oof = pd.DataFrame(
            np.nan,
            index=index,
            columns=["q10", "q50", "q90"],
            dtype=float,
        )
        # Once the weights have been learned on observed OOF targets, a blend
        # can still be produced for an OOF delivery whose target is missing.
        # Keeping availability separate from scoreability makes the reported
        # coverage diagnostically useful.
        blendable = expected_mask & np.isfinite(
            oof.to_numpy(dtype=float)
        ).all(axis=1)
        ensemble_oof.loc[blendable] = self.ensemble_.predict(
            oof.loc[blendable]
        ).to_numpy()

        metric_mask = (
            evaluation_mask
            if self.config.evaluation_days is not None
            else expected_mask
        )

        metric_input = oof.copy()
        metric_input["ensemble__q10"] = ensemble_oof["q10"]
        metric_input["ensemble__q50"] = ensemble_oof["q50"]
        metric_input["ensemble__q90"] = ensemble_oof["q90"]
        residual_oof: pd.DataFrame | None = None
        residual_diagnostics: dict[str, Any] = {
            "enabled": self.residual_corrector_factory is not None,
            "base_model": self.residual_base_model,
        }
        if self.residual_corrector_factory is not None:
            if self.config.evaluation_days is None:
                raise LeakageRiskError(
                    "Le correcteur rÃ©siduel exige un holdout final scellÃ©; "
                    "configurez evaluation_days pour sÃ©parer son apprentissage "
                    "de sa mesure."
                )
            residual_base = (
                chronos
                if self.residual_base_model == "chronos2"
                else ensemble_oof
            )
            residual_fit_mask = ensemble_fit_mask & np.isfinite(
                residual_base.to_numpy(dtype=float)
            ).all(axis=1)
            residual_eval_mask = metric_mask & expected_mask & np.isfinite(
                residual_base.to_numpy(dtype=float)
            ).all(axis=1)
            if not np.all(residual_eval_mask[metric_mask]):
                raise LeakageRiskError(
                    "La base du correcteur rÃ©siduel ne couvre pas tout le "
                    "holdout final."
                )

            # ResidualCorrector deliberately enforces an explicit UTC schema.
            # Work on UTC-indexed copies even if callers represented the same
            # instants in another aware timezone.
            residual_X = X.copy()
            residual_X.index = index.tz_convert("UTC")
            residual_target = target.copy()
            residual_target.index = residual_X.index
            residual_experts = oof.copy()
            residual_experts.index = residual_X.index
            residual_base_utc = residual_base.copy()
            residual_base_utc.index = residual_X.index

            self.residual_evaluation_corrector_ = (
                self.residual_corrector_factory().fit(
                    residual_X.loc[residual_fit_mask],
                    residual_target.loc[residual_fit_mask],
                    residual_base_utc.loc[residual_fit_mask],
                    residual_experts.loc[residual_fit_mask],
                )
            )
            corrected_evaluation = self.residual_evaluation_corrector_.predict(
                residual_X.loc[residual_eval_mask],
                residual_base_utc.loc[residual_eval_mask],
                residual_experts.loc[residual_eval_mask],
            )
            residual_oof = pd.DataFrame(
                np.nan,
                index=index,
                columns=["q10", "q50", "q90"],
                dtype=float,
            )
            residual_oof.loc[residual_eval_mask] = (
                corrected_evaluation.to_numpy(dtype=float)
            )
            metric_input["residual_corrected__q10"] = residual_oof["q10"]
            metric_input["residual_corrected__q50"] = residual_oof["q50"]
            metric_input["residual_corrected__q90"] = residual_oof["q90"]
            residual_diagnostics.update(
                {
                    "n_evaluation_fit": int(residual_fit_mask.sum()),
                    "n_evaluation_predicted": int(residual_eval_mask.sum()),
                    "evaluation_backend": self.residual_evaluation_corrector_.backend_,
                    "evaluation_meta_features": int(
                        len(self.residual_evaluation_corrector_.feature_columns_)
                    ),
                }
            )
        metrics = _mae_diagnostics(
            target,
            metric_input,
            expected_mask=metric_mask,
        )

        # The annual metric above is now frozen.  A separate operational
        # corrector may use every observed OOF residual (including the former
        # evaluation year) only for forecasts strictly after training_end.
        if self.residual_corrector_factory is not None:
            self.residual_corrector_ = self.residual_corrector_factory().fit(
                residual_X.loc[all_oof_common],
                residual_target.loc[all_oof_common],
                residual_base_utc.loc[all_oof_common],
                residual_experts.loc[all_oof_common],
            )
            residual_diagnostics.update(
                {
                    "n_live_fit": int(all_oof_common.sum()),
                    "live_backend": self.residual_corrector_.backend_,
                    "live_meta_features": int(
                        len(self.residual_corrector_.feature_columns_)
                    ),
                }
            )

        # The final experts see all historical targets only after every OOF
        # prediction and every stacking weight have been frozen.
        self.lear_ = self.lear_factory().fit(X, target)
        self.catboost_ = self.catboost_factory().fit(X, target)
        self.feature_columns_ = list(X.columns)
        self.training_end_utc_ = index[-1].tz_convert("UTC")
        self.training_result_ = HourlyOOFTrainingResult(
            oof_predictions=oof,
            ensemble_oof_predictions=ensemble_oof,
            fold_id=fold_id,
            metrics=metrics,
            ensemble_weights=self.ensemble_.weights_.copy(),
            residual_oof_predictions=residual_oof,
            diagnostics={
                "n_rows": int(len(X)),
                "n_target_observed": int(target.notna().sum()),
                "n_oof_expected": int(expected_mask.sum()),
                "n_oof_common": int(all_oof_common.sum()),
                "n_ensemble_fit": int(ensemble_fit_mask.sum()),
                "n_evaluation": int(metric_mask.sum()),
                "oof_coverage": float(
                    all_oof_common.sum() / expected_mask.sum()
                ),
                "n_splits": int(len(splits)),
                "gap_rows": int(self.config.gap),
                "gap_days": int(self.config.gap_days),
                "split_scope": (
                    "complete_local_delivery_days"
                    if self.config.split_on_delivery_days
                    else "ordered_rows"
                ),
                "training_start_utc": str(index[0].tz_convert("UTC")),
                "training_end_utc": str(self.training_end_utc_),
                "metric_scope": (
                    f"sealed_final_{self.config.evaluation_days}_delivery_days"
                    if self.config.evaluation_days is not None
                    else "all_oof_rows_no_sealed_holdout"
                ),
                "evaluation_start_local_date": (
                    str(evaluation_dates[0]) if evaluation_dates else None
                ),
                "evaluation_end_local_date": (
                    str(evaluation_dates[-1]) if evaluation_dates else None
                ),
                "residual_correction": residual_diagnostics,
            },
        )
        self.is_fitted_ = True
        return self

    def _validate_future_day(self, X_future: pd.DataFrame) -> pd.DataFrame:
        index = _require_hourly_index(X_future, name="X_future")
        if index[0].tz_convert("UTC") <= self.training_end_utc_:
            raise LeakageRiskError(
                "X_future doit commencer strictement après la dernière cible "
                "utilisée pour le réentraînement final."
            )
        missing = [
            column for column in self.feature_columns_ if column not in X_future.columns
        ]
        if missing:
            raise ValueError(f"Features futures absentes: {missing}.")
        if not self.config.require_single_future_day:
            return build_delivery_metadata(index, timezone=self.config.timezone)

        local = index.tz_convert(self.config.timezone)
        local_dates = pd.Index(local.date).unique()
        if len(local_dates) != 1:
            raise HourlyTargetContractError(
                "X_future doit représenter une unique journée locale."
            )
        canonical = local_delivery_day_index(
            local_dates[0], timezone=self.config.timezone
        )
        if not np.array_equal(index.tz_convert("UTC").asi8, canonical.asi8):
            raise HourlyTargetContractError(
                "X_future doit contenir exactement toutes les heures physiques "
                "de la journée locale (23, 24 ou 25)."
            )
        return build_delivery_metadata(index, timezone=self.config.timezone)

    def predict(
        self,
        X_future: pd.DataFrame,
        *,
        chronos_future: pd.DataFrame,
        chronos_origin: pd.Series,
    ) -> HourlyForecastResult:
        """Predict every physical hour of one future local delivery day."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("HourlyOOFPipeline doit être entraîné avant predict().")
        metadata = self._validate_future_day(X_future)
        index = X_future.index
        chronos = _normalize_chronos_quantiles(
            chronos_future,
            index,
            name="chronos_future",
            allow_missing=False,
        )
        required = np.ones(len(X_future), dtype=bool)
        origin = _normalize_origins(
            chronos_origin,
            index,
            name="chronos_origin",
            required_mask=required,
        )

        lear = _prefixed(self.lear_.predict(X_future), "lear")
        catboost = _prefixed(self.catboost_.predict(X_future), "catboost")
        chronos_prefixed = chronos.copy()
        chronos_prefixed.columns = [
            "chronos2__q10",
            "chronos2__q50",
            "chronos2__q90",
        ]
        experts_core = pd.concat([lear, catboost, chronos_prefixed], axis=1)
        experts = experts_core.copy()
        ensemble_final = self.ensemble_.predict(experts)
        final = ensemble_final
        residual_forecast_diagnostics: dict[str, Any] = {
            "enabled": self.residual_corrector_factory is not None,
            "base_model": self.residual_base_model,
        }
        if self.residual_corrector_factory is not None:
            utc_index = index.tz_convert("UTC")
            residual_X = X_future.copy()
            residual_X.index = utc_index
            residual_experts = experts_core.copy()
            residual_experts.index = utc_index
            residual_base = (
                chronos
                if self.residual_base_model == "chronos2"
                else ensemble_final
            ).copy()
            residual_base.index = utc_index
            corrected = self.residual_corrector_.predict(
                residual_X,
                residual_base,
                residual_experts,
            )
            correction = corrected["q50"] - residual_base["q50"]
            final = corrected
            for quantile in ("q10", "q50", "q90"):
                experts[f"ensemble_uncorrected__{quantile}"] = ensemble_final[
                    quantile
                ].to_numpy(dtype=float)
                experts[f"residual_corrected__{quantile}"] = corrected[
                    quantile
                ].to_numpy(dtype=float)
            experts["residual_correction"] = correction.to_numpy(dtype=float)
            residual_forecast_diagnostics.update(
                {
                    "backend": self.residual_corrector_.backend_,
                    "meta_features": int(
                        len(self.residual_corrector_.feature_columns_)
                    ),
                    "correction_mean": float(correction.mean()),
                    "correction_min": float(correction.min()),
                    "correction_max": float(correction.max()),
                }
            )

        # Canonical user-facing timeline is UTC even when feature inputs used a
        # different aware timezone representation.
        utc_index = index.tz_convert("UTC")
        experts.index = utc_index
        final.index = utc_index
        final.index.name = "delivery_start_utc"
        experts.index.name = "delivery_start_utc"
        return HourlyForecastResult(
            predictions=final,
            expert_predictions=experts,
            delivery_metadata=metadata,
            diagnostics={
                "hours_in_local_day": int(len(final)),
                "local_date": str(metadata["local_date"].iloc[0]),
                "chronos_origin_min_utc": str(origin.min()),
                "chronos_origin_max_utc": str(origin.max()),
                "ensemble_weights": self.ensemble_.weights_.to_dict(),
                "residual_correction": residual_forecast_diagnostics,
            },
        )
