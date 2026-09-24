"""Pure auxiliary calculations for the isolated, prospective rank-16 trial.

The old, inspected LoRA evaluation is calibration data here, never a new
holdout.  Only the residual corrector is reconstructed prequentially; this is
not neural OOF and conveys no production or promotion eligibility.  No file,
checkpoint, activation setting or production Kalman implementation is changed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd


RAW_MODEL = "chronos2_exogenous"
RESIDUAL_MODEL = "lora16_residual"
KALMAN_MODEL = "lora16_residual_kalman"
QUANTILES = ("q10", "q50", "q90")
FEATURE_COLUMNS = (
    "intercept", "local_hour_sin", "local_hour_cos", "local_dow_sin", "local_dow_cos",
)
MARKET_COLUMNS = tuple(f"{zone}_residual_load_fcst" for zone in ("fr", "de", "be", "nl", "es"))
STANDARD_CANDIDATES = ("linear_bias", "linear_harmonic", "linear_market", "linear_scale", "ukf_scale")


class ProspectiveAuxiliaryError(ValueError):
    """An auxiliary trial calculation cannot establish its causal contract."""


@dataclass(frozen=True)
class ResidualRecipe:
    ridge_alpha: float = 1.0
    maximum_shift_eur_mwh: float = 20.0
    minimum_training_days: int = 30
    lookback_days: int = 365

    def validate(self) -> None:
        if asdict(self) != asdict(ResidualRecipe()) or any(
            isinstance(value, (bool, np.bool_)) for value in asdict(self).values()
        ):
            raise ProspectiveAuxiliaryError("La recette prospective est fixe: ridge=1, clip=20, minimum=30, lookback=365.")
        if type(self.minimum_training_days) is not int or type(self.lookback_days) is not int:
            raise ProspectiveAuxiliaryError("Les nombres de jours doivent etre des entiers.")


@dataclass(frozen=True)
class ResidualFit:
    coefficients: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    timezone: str
    target_day: str
    training_start_day: str | None
    training_end_day: str | None
    training_days: int
    training_hours: int
    identity_cold_start: bool
    recipe: ResidualRecipe

    def to_audit(self) -> dict[str, Any]:
        return {**asdict(self), "feature_columns": list(FEATURE_COLUMNS),
                "fit_protocol": "strict_prior_calendar_ridge_selected_checkpoint_trial",
                "neural_oof": False, "target_observations_used": 0}


@dataclass(frozen=True)
class TrialChainsResult:
    corrected_future: pd.DataFrame
    kalman_future: pd.DataFrame
    prequential_history: pd.DataFrame
    audit: Mapping[str, Any]


def _index(values: Any, *, label: str) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(values)
        if index.tz is None or index.hasnans:
            raise ValueError("timezone absente ou NaT")
        return index.tz_convert("UTC").as_unit("ns")
    except (TypeError, ValueError) as exc:
        raise ProspectiveAuxiliaryError(f"{label}: timestamps timezone-aware requis.") from exc


def _day(value: str | date | pd.Timestamp) -> date:
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp != stamp.normalize():
            raise ValueError("jour civil attendu")
        return stamp.date()
    except (TypeError, ValueError) as exc:
        raise ProspectiveAuxiliaryError("target_day doit identifier un jour civil.") from exc


def _expected_index(start: date, end_exclusive: date, timezone: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start, tz=timezone), pd.Timestamp(end_exclusive, tz=timezone),
                         freq="h", inclusive="left").tz_convert("UTC")


def _normalise_raw(raw: pd.DataFrame, *, timezone: str, history: bool) -> pd.DataFrame:
    if raw.columns.has_duplicates:
        raise ProspectiveAuxiliaryError("Colonnes dupliquees dans les donnees brutes.")
    required = ["delivery_start_utc", *(f"{RAW_MODEL}__{q}" for q in QUANTILES)]
    if history:
        required.append("actual")
    missing = set(required).difference(raw.columns)
    if missing:
        raise ProspectiveAuxiliaryError(f"Colonnes brutes absentes: {sorted(missing)}.")
    # A future actual is not selected, parsed, inspected or copied.
    columns = list(dict.fromkeys([*required, *(c for c in MARKET_COLUMNS if c in raw),
                                 *(c for c in ("forecast_origin_utc",) if c in raw)]))
    frame = raw.loc[:, columns].copy()
    if frame.empty:
        raise ProspectiveAuxiliaryError("Historique ou jour futur vide.")
    frame["delivery_start_utc"] = _index(frame["delivery_start_utc"], label="delivery_start_utc")
    frame = frame.sort_values("delivery_start_utc", kind="stable").reset_index(drop=True)
    index = pd.DatetimeIndex(frame["delivery_start_utc"])
    local_days = index.tz_convert(timezone).date
    if index.has_duplicates or not index.equals(_expected_index(local_days[0], local_days[-1] + timedelta(days=1), timezone)):
        raise ProspectiveAuxiliaryError("Timeline incomplete: jours consecutifs et heures physiques exactes 23/24/25 requis.")
    if not history and local_days[0] != local_days[-1]:
        raise ProspectiveAuxiliaryError("Le forecast futur doit couvrir un seul jour civil.")
    numeric = [*(f"{RAW_MODEL}__{q}" for q in QUANTILES), *(c for c in MARKET_COLUMNS if c in frame)]
    if history:
        numeric.append("actual")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ProspectiveAuxiliaryError("Quantiles, labels historiques ou covariables non finis.")
    values = frame[[f"{RAW_MODEL}__{q}" for q in QUANTILES]].to_numpy(float)
    if (values[:, :-1] > values[:, 1:]).any():
        raise ProspectiveAuxiliaryError("Croisement des quantiles bruts.")
    if "forecast_origin_utc" in frame:
        origins = _index(frame["forecast_origin_utc"], label="forecast_origin_utc")
        expected = pd.DatetimeIndex([
            pd.Timestamp(f"{d - timedelta(days=1)} 08:00", tz=timezone) for d in local_days
        ]).tz_convert("UTC")
        if not origins.equals(expected):
            raise ProspectiveAuxiliaryError("Origines attendues exactement a D-1 08:00 local.")
        frame["forecast_origin_utc"] = origins
    return frame


def _design(index: pd.DatetimeIndex, timezone: str) -> np.ndarray:
    local = index.tz_convert(timezone)
    return np.column_stack((np.ones(len(local)), np.sin(2 * np.pi * local.hour / 24),
                            np.cos(2 * np.pi * local.hour / 24),
                            np.sin(2 * np.pi * local.dayofweek / 7),
                            np.cos(2 * np.pi * local.dayofweek / 7)))


def _fit_validated(frame: pd.DataFrame, *, target: date, timezone: str,
                   recipe: ResidualRecipe, require_full_window: bool) -> ResidualFit:
    if len(frame):
        days = pd.DatetimeIndex(frame["delivery_start_utc"]).tz_convert(timezone).date
        if days[-1] >= target:
            raise ProspectiveAuxiliaryError("Le fit doit exclure les labels du jour cible et futurs.")
        if days[-1] != target - timedelta(days=1):
            raise ProspectiveAuxiliaryError("Le fit doit finir exactement a D-1; historique perime ou gap.")
        frame = frame.loc[days >= target - timedelta(days=recipe.lookback_days)]
        days = pd.DatetimeIndex(frame["delivery_start_utc"]).tz_convert(timezone).date
        training_days = len(set(days))
    else:
        days, training_days = np.asarray([], dtype=object), 0
    if require_full_window and training_days != recipe.lookback_days:
        raise ProspectiveAuxiliaryError("Le forecast prospectif exige 365 jours complets strictement anterieurs.")
    cold = training_days < recipe.minimum_training_days
    means, scales, coefficients = np.zeros(5), np.ones(5), np.zeros(5)
    if not cold:
        design = _design(pd.DatetimeIndex(frame["delivery_start_utc"]), timezone)
        means, scales = design.mean(axis=0), design.std(axis=0)
        means[0], scales[0] = 0.0, 1.0
        scales[scales <= 1e-12] = 1.0
        normalised = (design - means) / scales
        residual = frame["actual"].to_numpy(float) - frame[f"{RAW_MODEL}__q50"].to_numpy(float)
        penalty = np.eye(5) * recipe.ridge_alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.lstsq(normalised.T @ normalised + penalty,
                                       normalised.T @ residual, rcond=None)[0]
    return ResidualFit(tuple(coefficients), tuple(means), tuple(scales), timezone,
                       target.isoformat(), str(days[0]) if training_days else None,
                       str(days[-1]) if training_days else None, training_days, len(frame), cold, recipe)


def fit_residual_corrector(raw_history: pd.DataFrame, *, target_day: str | date,
                           timezone: str = "Europe/Paris", recipe: ResidualRecipe = ResidualRecipe(),
                           require_full_window: bool = False) -> ResidualFit:
    recipe.validate()
    frame = _normalise_raw(raw_history, timezone=timezone, history=True)
    return _fit_validated(frame, target=_day(target_day), timezone=timezone,
                          recipe=recipe, require_full_window=require_full_window)


def _apply_validated(frame: pd.DataFrame, fit: ResidualFit) -> pd.DataFrame:
    fit.recipe.validate()
    parameters = np.asarray([fit.coefficients, fit.feature_means, fit.feature_scales], dtype=float)
    if parameters.shape != (3, 5) or not np.isfinite(parameters).all() or (parameters[2] <= 0).any():
        raise ProspectiveAuxiliaryError("Parametres residuels invalides.")
    index = pd.DatetimeIndex(frame["delivery_start_utc"])
    if set(index.tz_convert(fit.timezone).date) != {_day(fit.target_day)}:
        raise ProspectiveAuxiliaryError("Le fit residuel est lie a un unique jour cible.")
    if fit.training_end_day is not None and _day(fit.training_end_day) >= _day(fit.target_day):
        raise ProspectiveAuxiliaryError("Fit residuel non causal.")
    shift = ((_design(index, fit.timezone) - parameters[1]) / parameters[2]) @ parameters[0]
    shift = np.clip(shift, -fit.recipe.maximum_shift_eur_mwh, fit.recipe.maximum_shift_eur_mwh)
    result = frame.copy()
    for q in QUANTILES:
        result[f"{RESIDUAL_MODEL}__{q}"] = result[f"{RAW_MODEL}__{q}"].to_numpy(float) + shift
    result["residual_correction"] = shift
    return result


def apply_residual_corrector(raw_future: pd.DataFrame, fit: ResidualFit) -> pd.DataFrame:
    return _apply_validated(_normalise_raw(raw_future, timezone=fit.timezone, history=False), fit)


def build_prequential_residual_history(raw_history: pd.DataFrame, *, timezone: str = "Europe/Paris",
                                      recipe: ResidualRecipe = ResidualRecipe()) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    recipe.validate()
    frame = _normalise_raw(raw_history, timezone=timezone, history=True)
    days = pd.DatetimeIndex(frame["delivery_start_utc"]).tz_convert(timezone).date
    outputs, audits = [], []
    for day in dict.fromkeys(days):
        prior = frame.loc[days < day]
        fit = _fit_validated(prior, target=day, timezone=timezone, recipe=recipe, require_full_window=False)
        block = _apply_validated(frame.loc[days == day], fit)
        outputs.append(block)
        audits.append(fit.to_audit())
    return pd.concat(outputs, ignore_index=True), audits


def _fit_kalman_future(history: pd.DataFrame, future: pd.DataFrame, *, timezone: str,
                       config: Any) -> Mapping[str, Any]:
    # Lazy, numerical-only import: no Chronos/torch pipeline is loaded here.
    from chronos2_hourly import kalman_residual as engine
    from chronos2_hourly.kalman_covariates import KalmanCovariateConfig

    engine._require_pykalman()
    config.validate()
    covariate_config = KalmanCovariateConfig(history_missing_policy="complete_trailing", minimum_history_coverage=1.0)
    for label, frame in (("historique", history), ("futur", future)):
        missing = set(MARKET_COLUMNS).difference(frame.columns)
        if missing:
            raise ProspectiveAuxiliaryError(f"Covariables Kalman {label} absentes: {sorted(missing)}.")
    def covariates(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[["delivery_start_utc", *MARKET_COLUMNS]].rename(columns={"delivery_start_utc": "timestamp"})
    training, columns, coverage = engine._normalise_input(
        history.drop(columns=list(MARKET_COLUMNS)), upstream_model=RESIDUAL_MODEL, timezone=timezone,
        covariates=covariates(history), covariate_config=covariate_config)
    target = engine._normalise_future_input(
        future.drop(columns=list(MARKET_COLUMNS)), upstream_model=RESIDUAL_MODEL, timezone=timezone,
        covariates=covariates(future), covariate_columns=columns, covariate_config=covariate_config)
    days = tuple(dict.fromkeys(training["_local_day"]))
    target_day = target["_local_day"].iloc[0]
    if days[-1] != target_day - timedelta(days=1):
        raise ProspectiveAuxiliaryError("La calibration Kalman doit etre exactement adjacente au jour futur.")
    for day, block in training.groupby("_local_day", sort=True):
        engine._require_complete_local_day(block, local_day=day, timezone=timezone, context="Essai prospectif")
    feature_columns = {kind: engine._candidate_market_feature_columns(
        kind, covariate_config=covariate_config, covariate_columns=columns) for kind in config.candidate_kinds}
    fitted = engine._fit_rolling_target_day(
        training_frame=training, training_days=days, target_block=target, target_day=target_day,
        timezone=timezone, upstream_model=RESIDUAL_MODEL, output_model=KALMAN_MODEL,
        config=config, covariate_columns=columns, candidate_feature_columns=feature_columns)
    return {**fitted, "covariate_audit": coverage, "covariate_columns": list(columns)}


def _audit_values(value: Any) -> Any:
    """Keep engine's not-applicable state fields valid in strict JSON audits."""
    if isinstance(value, Mapping):
        return {str(key): _audit_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_values(item) for item in value]
    if isinstance(value, np.generic):
        return _audit_values(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    return value


def forecast_trial_chains(raw_history: pd.DataFrame, raw_future: pd.DataFrame, *,
                          timezone: str = "Europe/Paris", recipe: ResidualRecipe = ResidualRecipe(),
                          kalman_config: Any = None) -> TrialChainsResult:
    from chronos2_hourly.kalman_residual import KalmanResidualConfig

    recipe.validate()
    config = kalman_config or KalmanResidualConfig()
    config.validate()
    if tuple(config.candidate_kinds) != STANDARD_CANDIDATES:
        raise ProspectiveAuxiliaryError("L'essai reprend exactement les cinq candidats Kalman standard.")
    history = _normalise_raw(raw_history, timezone=timezone, history=True)
    future = _normalise_raw(raw_future, timezone=timezone, history=False)
    target_day = pd.Timestamp(future["delivery_start_utc"].iloc[0]).tz_convert(timezone).date()
    fit = _fit_validated(history, target=target_day, timezone=timezone, recipe=recipe, require_full_window=True)
    corrected = _apply_validated(future, fit)
    prequential, residual_audits = build_prequential_residual_history(history, timezone=timezone, recipe=recipe)
    days = pd.DatetimeIndex(prequential["delivery_start_utc"]).tz_convert(timezone).date
    calibration = prequential.loc[days >= target_day - timedelta(days=recipe.lookback_days)].copy()
    kalman = _fit_kalman_future(calibration, corrected, timezone=timezone, config=config)
    predictions = kalman["predictions"].reset_index()
    result = corrected.merge(predictions, on="delivery_start_utc", how="left", validate="one_to_one")
    if len(result) != len(corrected) or not np.isfinite(result[[f"{KALMAN_MODEL}__{q}" for q in QUANTILES]].to_numpy(float)).all():
        raise ProspectiveAuxiliaryError("Sortie Kalman incomplete ou non finie.")
    audit = {
        "kind": "chronos2_lora16_prospective_auxiliary_trial",
        "research_only": True, "diagnostic_only": True, "production_pit_evidence": False,
        "promotion_eligible": False, "neural_oof": False,
        "historical_predictions_role": "previously_inspected_calibration_not_holdout",
        "residual_recipe_frozen_coefficients_refitted_from_strict_past": True,
        "historical_corrector_protocol": "prequential_residual_only_selected_neural_checkpoint",
        "historical_metrics_are_independent_test": False, "target_observations_used": 0,
        "recipe": asdict(recipe), "residual_fit": fit.to_audit(), "prequential_residual_fits": residual_audits,
        "kalman_configuration": asdict(config), "kalman_daily_audit": kalman["daily_audit"],
        "kalman_window_audit": kalman["window_audit"], "kalman_state_audit": kalman["state_audit"],
        "kalman_market_scalers": kalman["market_scalers"],
        "kalman_covariate_audit": kalman["covariate_audit"],
        "kalman_covariate_columns": kalman["covariate_columns"],
    }
    return TrialChainsResult(corrected, result, prequential, _audit_values(audit))


__all__ = [
    "RAW_MODEL", "RESIDUAL_MODEL", "KALMAN_MODEL", "MARKET_COLUMNS", "STANDARD_CANDIDATES",
    "ProspectiveAuxiliaryError", "ResidualRecipe", "ResidualFit", "TrialChainsResult",
    "fit_residual_corrector", "apply_residual_corrector", "build_prequential_residual_history",
    "forecast_trial_chains",
]
