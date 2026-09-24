"""Causal preprocessing correction of the five residual-load forecasts.

This module learns ``observed residual load - raw residual-load forecast`` in
GW, independently for each country, before the forecasts are passed to the
hourly price model.  The production recipe deliberately mirrors the price
residual layer: one CatBoost model and one
``HistGradientBoostingRegressor``-backed :class:`ResidualCorrector`, blended
50/50 and clipped only after blending.

The public API is intentionally data-only.  Callers provide already
materialised forecast and observation frames; this module performs no network
request and writes no file.  All feature engineering is on a strict UTC hourly
timeline.  Observed errors may enter a feature only at least 48 hours later.
For live and prequential use an additional D-2 embargo is enforced, so the
model fitted for delivery day D never sees a label delivered on D-1 or D.
Missing raw forecasts are never forward-filled: each country is fitted and
predicted only on its present raw values, and every returned frame preserves
the original per-country NaN mask.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
import inspect
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .features import HourlyFeatureContractError, validate_utc_hourly_index
from .hourly_contract import local_delivery_day_index
from .models import BlendedResidualCorrector, ResidualCorrector


DELIVERY_TIMEZONE = "Europe/Paris"
RESIDUAL_LOAD_COUNTRIES: tuple[str, ...] = ("fr", "de", "be", "nl", "es")
RESIDUAL_LOAD_ALIASES: tuple[str, ...] = tuple(
    f"{country}_residual_load_fcst" for country in RESIDUAL_LOAD_COUNTRIES
)
RESIDUAL_LOAD_OBS_SERIES_BY_ALIAS: Mapping[str, str] = MappingProxyType(
    {
        f"{country}_residual_load_fcst":
            f"power.{country}.residual.load.entsoe.hourly.gw.obs"
        for country in RESIDUAL_LOAD_COUNTRIES
    }
)
RESIDUAL_LOAD_ALIAS_BY_OBS_SERIES: Mapping[str, str] = MappingProxyType(
    {series: alias for alias, series in RESIDUAL_LOAD_OBS_SERIES_BY_ALIAS.items()}
)

# Readable aliases for integration code that uses the vocabulary of the live
# residual-load provider.
EXPECTED_ALIASES = RESIDUAL_LOAD_ALIASES
OBSERVED_SERIES_BY_ALIAS = RESIDUAL_LOAD_OBS_SERIES_BY_ALIAS

DEFAULT_FORECAST_CHANGE_LAGS_HOURS: tuple[int, ...] = (1, 2, 24, 48, 168)
DEFAULT_ERROR_LAGS_HOURS: tuple[int, ...] = (48, 72, 168, 336)
DEFAULT_ERROR_ROLLING_WINDOWS_HOURS: tuple[int, ...] = (24, 72, 168)
MINIMUM_ERROR_LAG_HOURS = 48


class ResidualLoadInputCorrectionError(ValueError):
    """Raised when input correction could violate its causal contract."""


CorrectorFactory = Callable[..., Any]


@dataclass(frozen=True)
class ResidualLoadCorrectionResult:
    """One correction result, kept in GW throughout.

    ``raw``, ``correction`` and ``corrected`` have the five canonical aliases
    as columns and exactly the requested UTC index.  ``component_corrections``
    maps component names (normally ``cat_v1`` and ``hgb31``) to their unblended
    correction frames.  ``diagnostics`` contains fit cutoffs and per-component
    model diagnostics; it never contains model objects.
    """

    raw: pd.DataFrame
    correction: pd.DataFrame
    corrected: pd.DataFrame
    component_corrections: Mapping[str, pd.DataFrame]
    diagnostics: Mapping[str, Any]

    def to_frame(self) -> pd.DataFrame:
        """Return a flat audit frame with raw/correction/corrected columns."""

        parts: list[pd.DataFrame] = []
        for label, frame in (
            ("raw", self.raw),
            ("correction", self.correction),
            ("corrected", self.corrected),
        ):
            renamed = frame.rename(
                columns={alias: f"{alias}__{label}" for alias in RESIDUAL_LOAD_ALIASES}
            )
            parts.append(renamed)
        for component, frame in self.component_corrections.items():
            parts.append(
                frame.rename(
                    columns={
                        alias: f"{alias}__component_{component}"
                        for alias in RESIDUAL_LOAD_ALIASES
                    }
                )
            )
        result = pd.concat(parts, axis=1)
        result.index.name = self.raw.index.name
        return result


def _validate_utc_index(
    index: pd.Index,
    *,
    name: str,
    require_contiguous: bool,
) -> pd.DatetimeIndex:
    try:
        return validate_utc_hourly_index(
            index,
            name=name,
            require_contiguous=require_contiguous,
        )
    except HourlyFeatureContractError as exc:
        raise ResidualLoadInputCorrectionError(str(exc)) from exc


def _as_utc_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except Exception as exc:
        raise ResidualLoadInputCorrectionError(
            f"{name} n'est pas un timestamp valide: {value!r}."
        ) from exc
    if pd.isna(result) or result.tzinfo is None:
        raise ResidualLoadInputCorrectionError(
            f"{name} doit être un timestamp timezone-aware."
        )
    return result.tz_convert("UTC")


def _as_local_date(value: str | date | pd.Timestamp, *, timezone: str) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ResidualLoadInputCorrectionError("delivery_day est NaT.")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(timezone).tz_localize(None)
    return timestamp.date()


def scheduled_live_origin_utc(
    delivery_day: str | date | pd.Timestamp,
    *,
    timezone: str = DELIVERY_TIMEZONE,
) -> pd.Timestamp:
    """Return the D-1 08:00 local day-ahead origin in UTC."""

    day = _as_local_date(delivery_day, timezone=timezone)
    local_label = pd.Timestamp(day - timedelta(days=1)) + pd.Timedelta(hours=8)
    try:
        return local_label.tz_localize(timezone).tz_convert("UTC")
    except (TypeError, ValueError, KeyError) as exc:
        raise ResidualLoadInputCorrectionError(
            f"Fuseau de livraison invalide: {timezone!r}."
        ) from exc


def conservative_label_end_utc(
    delivery_day: str | date | pd.Timestamp,
    *,
    timezone: str = DELIVERY_TIMEZONE,
) -> pd.Timestamp:
    """Return the last deliverable UTC hour of D-2.

    The boundary is expressed as one hour before local midnight starting D-1,
    so it stays correct on 23/25-hour DST days.
    """

    day = _as_local_date(delivery_day, timezone=timezone)
    start_d_minus_one = pd.Timestamp(day - timedelta(days=1)).tz_localize(timezone)
    return start_d_minus_one.tz_convert("UTC") - pd.Timedelta(hours=1)


def _positive_unique_hours(
    values: Sequence[int],
    *,
    name: str,
    minimum: int = 1,
) -> tuple[int, ...]:
    result: list[int] = []
    for raw in values:
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"{name} doit contenir des heures entières >= {minimum}.")
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} doit contenir des heures entières >= {minimum}."
            ) from exc
        if value != raw or value < minimum:
            raise ValueError(f"{name} doit contenir des heures entières >= {minimum}.")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError(f"{name} ne doit pas être vide.")
    return tuple(result)


def _forecast_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    require_contiguous: bool = True,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} doit être un pandas.DataFrame.")
    index = _validate_utc_index(
        frame.index,
        name=f"{name}.index",
        require_contiguous=require_contiguous,
    )
    missing = [alias for alias in RESIDUAL_LOAD_ALIASES if alias not in frame]
    if missing:
        raise ResidualLoadInputCorrectionError(
            f"Aliases de residual load absents de {name}: {missing}."
        )
    selected = frame.loc[:, list(RESIDUAL_LOAD_ALIASES)]
    result = selected.apply(
        pd.to_numeric,
        errors="coerce",
    )
    invalid = selected.notna() & result.isna()
    if bool(invalid.to_numpy().any()):
        columns = invalid.columns[invalid.any()].tolist()
        raise TypeError(f"{name} contient des valeurs non numériques: {columns}.")
    if bool(np.isinf(result.to_numpy(dtype=float)).any()):
        raise ResidualLoadInputCorrectionError(
            f"{name} contient des prévisions GW infinies."
        )
    result.index = index.copy()
    result.index.name = frame.index.name or "delivery_start_utc"
    return result.astype(float)


def _observation_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} doit être un pandas.DataFrame.")
    index = _validate_utc_index(
        frame.index,
        name=f"{name}.index",
        require_contiguous=False,
    )
    renamed = frame.rename(columns=RESIDUAL_LOAD_ALIAS_BY_OBS_SERIES)
    missing = [alias for alias in RESIDUAL_LOAD_ALIASES if alias not in renamed]
    if missing:
        expected_series = [RESIDUAL_LOAD_OBS_SERIES_BY_ALIAS[a] for a in missing]
        raise ResidualLoadInputCorrectionError(
            f"Observations absentes de {name}: aliases={missing}, "
            f"séries .obs admises={expected_series}."
        )
    selected = renamed.loc[:, list(RESIDUAL_LOAD_ALIASES)]
    result = selected.apply(
        pd.to_numeric,
        errors="coerce",
    )
    invalid = selected.notna() & result.isna()
    if bool(invalid.to_numpy().any()):
        columns = invalid.columns[invalid.any()].tolist()
        raise TypeError(f"{name} contient des valeurs non numériques: {columns}.")
    if bool(np.isinf(result.to_numpy(dtype=float)).any()):
        raise ResidualLoadInputCorrectionError(f"{name} contient des infinis.")
    result.index = index.copy()
    result.index.name = frame.index.name or "delivery_start_utc"
    return result.astype(float)


def _normalise_clips(
    value: float | Mapping[str, float | None] | None,
) -> dict[str, float | None]:
    if isinstance(value, Mapping):
        unexpected = sorted(set(value) - set(RESIDUAL_LOAD_ALIASES))
        missing = sorted(set(RESIDUAL_LOAD_ALIASES) - set(value))
        if unexpected or missing:
            raise ValueError(
                "max_abs_correction_gw doit mapper exactement les cinq aliases; "
                f"missing={missing}, unexpected={unexpected}."
            )
        raw = {alias: value[alias] for alias in RESIDUAL_LOAD_ALIASES}
    else:
        raw = {alias: value for alias in RESIDUAL_LOAD_ALIASES}
    result: dict[str, float | None] = {}
    for alias, bound in raw.items():
        if bound is None:
            result[alias] = None
            continue
        numeric = float(bound)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(
                f"Clip GW invalide pour {alias}: il doit être fini et > 0."
            )
        result[alias] = numeric
    return result


def _base_quantiles(values: pd.Series) -> pd.DataFrame:
    """Make the required degenerate q10=q50=q90 base forecast."""

    numeric = values.astype(float)
    return pd.DataFrame(
        {"q10": numeric, "q50": numeric, "q90": numeric},
        index=values.index,
    )


def build_residual_load_input_features(
    forecasts: pd.DataFrame,
    *,
    error_history: pd.DataFrame | None = None,
    target_index: pd.DatetimeIndex | None = None,
    forecast_change_lags_hours: Sequence[int] = DEFAULT_FORECAST_CHANGE_LAGS_HOURS,
    error_lags_hours: Sequence[int] = DEFAULT_ERROR_LAGS_HOURS,
    error_rolling_windows_hours: Sequence[int] = (
        DEFAULT_ERROR_ROLLING_WINDOWS_HOURS
    ),
) -> pd.DataFrame:
    """Build causal raw features consumed by ``ResidualMetaFeatureBuilder``.

    Forecast changes use only the current value and earlier delivery hours.
    A missing forecast, or a change touching one, remains NaN; no forward fill
    or cross-hour repair is performed.
    Error-memory lookups use exact UTC offsets and every configured lag is
    validated to be at least 48 hours.  Missing historical memory is encoded
    as zero together with an explicit availability/count feature; it is never
    backfilled from a later observation.

    ``forecasts`` may include history before ``target_index``.  This is useful
    at prediction time so the first delivery hour receives real backward
    changes.  Only rows in ``target_index`` are returned.
    """

    forecast = _forecast_frame(forecasts, name="forecasts")
    change_lags = _positive_unique_hours(
        forecast_change_lags_hours,
        name="forecast_change_lags_hours",
    )
    memory_lags = _positive_unique_hours(
        error_lags_hours,
        name="error_lags_hours",
        minimum=MINIMUM_ERROR_LAG_HOURS,
    )
    windows = _positive_unique_hours(
        error_rolling_windows_hours,
        name="error_rolling_windows_hours",
    )
    if target_index is None:
        selected_index = forecast.index
    else:
        selected_index = _validate_utc_index(
            target_index,
            name="target_index",
            require_contiguous=True,
        )
        if not selected_index.isin(forecast.index).all():
            raise ResidualLoadInputCorrectionError(
                "target_index doit être entièrement contenu dans forecasts.index."
            )

    derived: dict[str, pd.Series] = {
        alias: forecast[alias] for alias in RESIDUAL_LOAD_ALIASES
    }
    for alias in RESIDUAL_LOAD_ALIASES:
        values = forecast[alias]
        for lag in change_lags:
            change = values - values.shift(lag)
            derived[f"{alias}__change_{lag}h"] = change
        ramp = values - values.shift(1)
        derived[f"{alias}__ramp_up_1h"] = ramp.clip(lower=0.0)
        derived[f"{alias}__ramp_down_1h"] = (-ramp).clip(lower=0.0)

    if error_history is None:
        errors = pd.DataFrame(
            np.nan,
            index=forecast.index,
            columns=list(RESIDUAL_LOAD_ALIASES),
            dtype=float,
        )
    else:
        errors = _observation_frame(error_history, name="error_history")
        # The frame is interpreted as already-computed errors, not levels.
        # Reindexing only introduces missing past memory and never propagates.
        errors = errors.reindex(forecast.index)

    # Reindex on the full UTC grid before rolling so a window denotes hours,
    # not merely a count of whichever observations happened to be present.
    grid_start = min(forecast.index[0], errors.index[0])
    grid_end = max(forecast.index[-1], errors.index[-1])
    grid = pd.date_range(grid_start, grid_end, freq="h", tz="UTC")
    errors_on_grid = errors.reindex(grid)
    for alias in RESIDUAL_LOAD_ALIASES:
        values = errors_on_grid[alias]
        for lag in memory_lags:
            lookup_index = forecast.index - pd.Timedelta(hours=lag)
            lagged = values.reindex(lookup_index)
            lagged.index = forecast.index
            available = lagged.notna().astype(float)
            derived[f"{alias}__error_lag_{lag}h"] = lagged.fillna(0.0)
            derived[f"{alias}__error_lag_{lag}h_available"] = available

        # Every rolling statistic ends at t-48h, including its right edge.
        lookup_index = forecast.index - pd.Timedelta(hours=MINIMUM_ERROR_LAG_HOURS)
        for window in windows:
            rolling = values.rolling(window=window, min_periods=1)
            mean = rolling.mean().reindex(lookup_index)
            mean.index = forecast.index
            count = rolling.count().reindex(lookup_index)
            count.index = forecast.index
            derived[
                f"{alias}__error_mean_{window}h_lag{MINIMUM_ERROR_LAG_HOURS}h"
            ] = mean.fillna(0.0)
            derived[
                f"{alias}__error_count_{window}h_lag{MINIMUM_ERROR_LAG_HOURS}h"
            ] = count.fillna(0.0)

    result = pd.DataFrame(derived, index=forecast.index).loc[selected_index]
    if bool(np.isinf(result.to_numpy(dtype=float)).any()):
        raise RuntimeError("La construction des features a produit des infinis.")
    result.index.name = forecast.index.name
    return result.astype(float)


def make_default_residual_load_corrector(
    alias: str,
    max_abs_correction_gw: float | None,
    *,
    thread_count: int = -1,
    timezone: str = DELIVERY_TIMEZONE,
) -> BlendedResidualCorrector:
    """Create the frozen CatBoost/HGB 50/50 recipe for one alias."""

    if alias not in RESIDUAL_LOAD_ALIASES:
        raise ValueError(f"Alias residual-load inconnu: {alias!r}.")
    primary_country = alias.split("_", 1)[0].upper()
    builder_options = {
        "timezone": timezone,
        "include_calendar": True,
        "include_rich_calendar": True,
        "rich_calendar_countries": ("FR", "DE", "BE", "ES", "NL"),
        "rich_calendar_primary_country": primary_country,
        "include_daily_profiles": True,
        "profile_columns": RESIDUAL_LOAD_ALIASES,
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
        thread_count=thread_count,
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
        max_abs_correction=max_abs_correction_gw,
    )


def _invoke_factory(
    factory: CorrectorFactory,
    *,
    alias: str,
    clip: float | None,
) -> Any:
    """Invoke injectable factories with a documented or minimal test shape."""

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None
    candidates = (
        ((), {"alias": alias, "max_abs_correction_gw": clip}),
        ((alias, clip), {}),
        ((alias,), {}),
        ((), {}),
    )
    for args, kwargs in candidates:
        if signature is not None:
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
        model = factory(*args, **kwargs)
        missing = [
            method
            for method in ("fit", "predict_correction")
            if not callable(getattr(model, method, None))
        ]
        if missing:
            raise TypeError(
                f"La factory de {alias} a retourné un objet sans méthodes {missing}."
            )
        return model
    raise TypeError(
        "corrector_factory doit accepter (), (alias), (alias, clip) ou les "
        "arguments nommés alias/max_abs_correction_gw."
    )


class ResidualLoadInputCorrector:
    """Fit and apply five independent causal residual-load correctors.

    Parameters
    ----------
    max_abs_correction_gw:
        One symmetric clip shared by all countries, ``None`` for no clip, or
        an exact mapping of the five aliases to per-country clips.
    corrector_factory:
        Injection point for tests or challengers.  It may accept no argument,
        ``alias``, ``(alias, clip)``, or the equivalent named arguments, and
        must return an object exposing ``fit`` and ``predict_correction``.
    cold_start_policy:
        ``raise`` rejects an alias without enough causal training pairs.
        ``raw_passthrough`` emits a zero correction until the threshold is met.
    minimum_training_rows:
        Minimum number of finite raw/observation pairs required per alias.
    """

    def __init__(
        self,
        *,
        max_abs_correction_gw: float | Mapping[str, float | None] | None = 8.0,
        corrector_factory: CorrectorFactory | None = None,
        cold_start_policy: str = "raise",
        minimum_training_rows: int = 48,
        thread_count: int = -1,
        timezone: str = DELIVERY_TIMEZONE,
        forecast_change_lags_hours: Sequence[int] = (
            DEFAULT_FORECAST_CHANGE_LAGS_HOURS
        ),
        error_lags_hours: Sequence[int] = DEFAULT_ERROR_LAGS_HOURS,
        error_rolling_windows_hours: Sequence[int] = (
            DEFAULT_ERROR_ROLLING_WINDOWS_HOURS
        ),
    ) -> None:
        self.max_abs_correction_gw = _normalise_clips(max_abs_correction_gw)
        self.corrector_factory = corrector_factory
        if cold_start_policy not in {"raise", "raw_passthrough"}:
            raise ValueError(
                "cold_start_policy doit valoir raise ou raw_passthrough."
            )
        try:
            parsed_minimum_training_rows = int(minimum_training_rows)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "minimum_training_rows doit etre un entier >= 2."
            ) from exc
        if (
            isinstance(minimum_training_rows, (bool, np.bool_))
            or parsed_minimum_training_rows != minimum_training_rows
            or parsed_minimum_training_rows < 2
        ):
            raise ValueError("minimum_training_rows doit etre un entier >= 2.")
        self.cold_start_policy = cold_start_policy
        self.minimum_training_rows = parsed_minimum_training_rows
        self.thread_count = int(thread_count)
        self.timezone = str(timezone)
        self.forecast_change_lags_hours = _positive_unique_hours(
            forecast_change_lags_hours,
            name="forecast_change_lags_hours",
        )
        self.error_lags_hours = _positive_unique_hours(
            error_lags_hours,
            name="error_lags_hours",
            minimum=MINIMUM_ERROR_LAG_HOURS,
        )
        self.error_rolling_windows_hours = _positive_unique_hours(
            error_rolling_windows_hours,
            name="error_rolling_windows_hours",
        )

    def _new_model(self, alias: str) -> Any:
        clip = self.max_abs_correction_gw[alias]
        if self.corrector_factory is not None:
            return _invoke_factory(self.corrector_factory, alias=alias, clip=clip)
        return make_default_residual_load_corrector(
            alias,
            clip,
            thread_count=self.thread_count,
            timezone=self.timezone,
        )

    def fit(
        self,
        forecasts: pd.DataFrame,
        observations: pd.DataFrame,
        *,
        label_end_utc: Any | None = None,
        forecast_origin_utc: Any | None = None,
    ) -> "ResidualLoadInputCorrector":
        """Fit on labels delivered no later than ``label_end_utc``.

        ``observations`` may use either the five forecast aliases or the five
        canonical Saturn ``.obs`` series names.  Observations beyond the label
        boundary are discarded *before* error-memory features are built.
        Supplying an origin additionally proves that every retained label is
        strictly earlier than the decision time.
        """

        forecast = _forecast_frame(forecasts, name="forecasts")
        observed = _observation_frame(observations, name="observations")
        if label_end_utc is None:
            label_end = min(forecast.index[-1], observed.index[-1])
        else:
            label_end = _as_utc_timestamp(label_end_utc, name="label_end_utc")
        origin = None
        if forecast_origin_utc is not None:
            origin = _as_utc_timestamp(
                forecast_origin_utc,
                name="forecast_origin_utc",
            )
            if label_end >= origin:
                raise ResidualLoadInputCorrectionError(
                    "label_end_utc doit être strictement antérieur à "
                    "forecast_origin_utc."
                )

        training_index = forecast.index[forecast.index <= label_end]
        if len(training_index) < 2:
            raise ResidualLoadInputCorrectionError(
                "Moins de deux lignes de forecast sont disponibles avant le cutoff."
            )
        training_forecast = forecast.loc[training_index]
        training_observed = observed.reindex(training_index)
        eligible_by_alias = {
            alias: training_forecast[alias].notna()
            & training_observed[alias].notna()
            for alias in RESIDUAL_LOAD_ALIASES
        }
        insufficient = {
            alias: int(eligible_by_alias[alias].sum())
            for alias in RESIDUAL_LOAD_ALIASES
            if eligible_by_alias[alias].sum() < self.minimum_training_rows
        }
        if insufficient and self.cold_start_policy == "raise":
            raise ResidualLoadInputCorrectionError(
                "Chaque alias doit avoir assez de couples raw/observation "
                "présents avant le cutoff; "
                f"minimum={self.minimum_training_rows}, insuffisant={insufficient}."
            )

        # Only eligible errors can feed any lag/rolling memory.
        errors = training_observed - training_forecast
        features = build_residual_load_input_features(
            training_forecast,
            error_history=errors,
            forecast_change_lags_hours=self.forecast_change_lags_hours,
            error_lags_hours=self.error_lags_hours,
            error_rolling_windows_hours=self.error_rolling_windows_hours,
        )
        models: dict[str, Any] = {}
        fit_indices: dict[str, pd.DatetimeIndex] = {}
        feature_columns: dict[str, tuple[str, ...]] = {}
        fit_features: dict[str, pd.DataFrame] = {}
        for alias in RESIDUAL_LOAD_ALIASES:
            if alias in insufficient:
                continue
            eligible = eligible_by_alias[alias]
            alias_index = training_index[eligible.to_numpy()]
            alias_features = features.loc[alias_index]
            usable_columns = tuple(
                column
                for column in alias_features.columns
                if alias_features[column].notna().any()
            )
            if not usable_columns:
                raise ResidualLoadInputCorrectionError(
                    f"Aucune feature utilisable pour {alias}."
                )
            model = self._new_model(alias)
            model.fit(
                alias_features.loc[:, list(usable_columns)],
                training_observed.loc[alias_index, alias],
                _base_quantiles(training_forecast.loc[alias_index, alias]),
                None,
            )
            models[alias] = model
            fit_indices[alias] = alias_index.copy()
            feature_columns[alias] = usable_columns
            fit_features[alias] = alias_features.loc[
                :,
                list(usable_columns),
            ].copy()

        self.models_ = models
        self.cold_start_aliases_ = tuple(
            alias for alias in RESIDUAL_LOAD_ALIASES if alias in insufficient
        )
        self.eligible_training_rows_by_alias_ = {
            alias: int(eligible_by_alias[alias].sum())
            for alias in RESIDUAL_LOAD_ALIASES
        }
        self.fit_indices_by_alias_ = fit_indices
        self.feature_columns_by_alias_ = feature_columns
        self.fit_features_by_alias_ = fit_features
        # Forecasts after the label embargo contain no target information and
        # remain valid causal context for delivery-time changes.  In
        # particular, D-1 forecasts bridge the intentional D-2 -> D label gap
        # of a live prediction.
        self.forecast_context_ = forecast.copy()
        self.forecast_history_ = training_forecast.copy()
        self.error_history_ = errors.copy()
        self.training_index_ = training_index.copy()
        self.label_end_utc_ = label_end
        self.forecast_origin_utc_ = origin
        self.is_fitted_ = True
        return self

    def fit_live(
        self,
        forecasts: pd.DataFrame,
        observations: pd.DataFrame,
        *,
        delivery_day: str | date | pd.Timestamp,
        runtime_cutoff_utc: Any | None = None,
    ) -> "ResidualLoadInputCorrector":
        """Fit a live D model with the scheduled D-1 08 and D-2 embargo.

        The effective observation cutoff is the earlier of an optional runtime
        cutoff and scheduled D-1 08:00 local.  The stricter delivery embargo
        then removes every label after the final UTC hour of local D-2.
        """

        scheduled = scheduled_live_origin_utc(delivery_day, timezone=self.timezone)
        if runtime_cutoff_utc is None:
            effective_cutoff = scheduled
        else:
            runtime = _as_utc_timestamp(
                runtime_cutoff_utc,
                name="runtime_cutoff_utc",
            )
            effective_cutoff = min(runtime, scheduled)
        embargo_end = conservative_label_end_utc(
            delivery_day,
            timezone=self.timezone,
        )
        label_end = min(effective_cutoff, embargo_end)
        result = self.fit(
            forecasts,
            observations,
            label_end_utc=label_end,
            forecast_origin_utc=scheduled,
        )
        self.delivery_day_local_ = _as_local_date(
            delivery_day,
            timezone=self.timezone,
        )
        self.scheduled_origin_utc_ = scheduled
        self.runtime_cutoff_utc_ = effective_cutoff
        self.embargo_end_utc_ = embargo_end
        return result

    def _prediction_features(self, forecast: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("ResidualLoadInputCorrector doit être entraîné avant predict().")
        if forecast.index[0] <= self.forecast_history_.index[-1]:
            raise ResidualLoadInputCorrectionError(
                "La prédiction doit commencer strictement après l'historique de fit."
            )
        context_history = self.forecast_context_.loc[
            self.forecast_context_.index < forecast.index[0]
        ]
        if context_history.empty:
            raise ResidualLoadInputCorrectionError(
                "Aucun contexte forecast ne précède le bloc de prédiction."
            )
        overlap = self.forecast_context_.index.intersection(forecast.index)
        if len(overlap):
            expected = self.forecast_context_.loc[overlap]
            received = forecast.loc[overlap]
            if not np.array_equal(
                expected.to_numpy(dtype=float),
                received.to_numpy(dtype=float),
                equal_nan=True,
            ):
                raise ResidualLoadInputCorrectionError(
                    "Les forecasts du bloc diffèrent du contexte fourni au fit."
                )
        gap = forecast.index[0] - context_history.index[-1]
        if gap != pd.Timedelta(hours=1):
            raise ResidualLoadInputCorrectionError(
                "Contexte forecast et prédiction doivent former une timeline UTC "
                f"continue; gap reçu={gap}."
            )
        context = pd.concat([context_history, forecast])
        return build_residual_load_input_features(
            context,
            error_history=self.error_history_,
            target_index=forecast.index,
            forecast_change_lags_hours=self.forecast_change_lags_hours,
            error_lags_hours=self.error_lags_hours,
            error_rolling_windows_hours=self.error_rolling_windows_hours,
        )

    def predict(self, forecasts: pd.DataFrame) -> ResidualLoadCorrectionResult:
        """Correct present raw values and preserve every per-alias NaN exactly."""

        forecast = _forecast_frame(forecasts, name="forecasts")
        features = self._prediction_features(forecast)
        correction = pd.DataFrame(
            np.nan,
            index=forecast.index,
            columns=list(RESIDUAL_LOAD_ALIASES),
            dtype=float,
        )
        components: dict[str, dict[str, pd.Series]] = {}
        model_diagnostics: dict[str, Any] = {}
        for alias in self.cold_start_aliases_:
            available = forecast[alias].notna()
            correction.loc[available, alias] = 0.0
            model_diagnostics[alias] = {
                "status": "cold_start_raw_passthrough",
                "fit_rows_with_raw_and_observation": (
                    self.eligible_training_rows_by_alias_[alias]
                ),
                "minimum_training_rows": self.minimum_training_rows,
                "prediction_rows_with_raw": int(available.sum()),
                "prediction_rows_missing_raw": int((~available).sum()),
            }
        for alias, model in self.models_.items():
            available = forecast[alias].notna()
            prediction_index = forecast.index[available.to_numpy()]
            model_diagnostics.setdefault(alias, {}).update(
                {
                    "status": "fitted",
                    "fit_rows_with_raw_and_observation": len(
                        self.fit_indices_by_alias_[alias]
                    ),
                    "prediction_rows_with_raw": len(prediction_index),
                    "prediction_rows_missing_raw": int((~available).sum()),
                }
            )
            fitted_components = getattr(model, "components_", {})
            for component_name in fitted_components:
                components.setdefault(component_name, {})
            if len(prediction_index) == 0:
                diagnostics_method = getattr(model, "diagnostics", None)
                if callable(diagnostics_method):
                    model_diagnostics[alias]["fit"] = diagnostics_method()
                continue

            alias_features = features.loc[
                prediction_index,
                list(self.feature_columns_by_alias_[alias]),
            ]
            base = _base_quantiles(forecast.loc[prediction_index, alias])
            model_features = alias_features
            model_base = base
            anchor_count = 0
            # ResidualMetaFeatureBuilder rejects an input column that is
            # entirely NaN *within one prediction batch*.  If a cross-country
            # forecast is absent for the whole day, prepend the minimum set of
            # earlier fitted rows needed only to freeze the already-known
            # schema, then discard their predictions.  No target-day NaN is
            # filled and no later row is consulted.
            if isinstance(model, (ResidualCorrector, BlendedResidualCorrector)):
                empty_columns = [
                    column
                    for column in alias_features.columns
                    if not alias_features[column].notna().any()
                ]
                if empty_columns:
                    fitted_features = self.fit_features_by_alias_[alias]
                    anchor_indices = sorted(
                        {
                            fitted_features[column].first_valid_index()
                            for column in empty_columns
                        }
                    )
                    if any(index is None for index in anchor_indices):
                        raise RuntimeError(
                            f"Schéma de prediction impossible à ancrer pour {alias}."
                        )
                    anchor_index = pd.DatetimeIndex(anchor_indices)
                    anchors = fitted_features.loc[anchor_index]
                    model_features = pd.concat([anchors, alias_features])
                    model_base = pd.concat(
                        [
                            _base_quantiles(
                                self.forecast_history_.loc[anchor_index, alias]
                            ),
                            base,
                        ]
                    )
                    anchor_count = len(anchor_index)
            model_diagnostics[alias]["schema_anchor_rows"] = anchor_count
            predicted = model.predict_correction(model_features, model_base, None)
            if isinstance(predicted, pd.Series):
                if not predicted.index.equals(model_features.index):
                    raise RuntimeError(
                        f"Le correcteur {alias} a retourné un index différent."
                    )
                all_values = predicted.to_numpy(dtype=float)
            else:
                all_values = np.asarray(predicted, dtype=float)
            if (
                all_values.shape != (len(model_features),)
                or not np.isfinite(all_values).all()
            ):
                raise RuntimeError(
                    f"Le correcteur {alias} a retourné une correction invalide."
                )
            values = all_values[anchor_count:]
            clip = self.max_abs_correction_gw[alias]
            if clip is not None:
                values = np.clip(values, -clip, clip)
            correction.loc[prediction_index, alias] = values

            weights = getattr(model, "weights", {})
            for component_name, component in fitted_components.items():
                component_prediction = component.predict_correction(
                    model_features,
                    model_base,
                    None,
                )
                if isinstance(component_prediction, pd.Series):
                    if not component_prediction.index.equals(model_features.index):
                        raise RuntimeError(
                            f"La composante {component_name}/{alias} a retourné "
                            "un index différent."
                        )
                    all_component_values = component_prediction.to_numpy(dtype=float)
                else:
                    all_component_values = np.asarray(
                        component_prediction,
                        dtype=float,
                    )
                if (
                    all_component_values.shape != (len(model_features),)
                    or not np.isfinite(all_component_values).all()
                ):
                    raise RuntimeError(
                        f"La composante {component_name}/{alias} a retourné "
                        "une correction invalide."
                    )
                component_values = all_component_values[anchor_count:]
                components.setdefault(component_name, {})[alias] = pd.Series(
                    component_values,
                    index=prediction_index,
                )
                # Include the configured blend weight next to prediction stats.
                model_diagnostics.setdefault(alias, {}).setdefault(
                    "component_prediction",
                    {},
                )[component_name] = {
                    "weight": float(weights.get(component_name, np.nan)),
                    "mean_gw": float(np.mean(component_values)),
                    "max_abs_gw": float(np.max(np.abs(component_values))),
                }
            diagnostics_method = getattr(model, "diagnostics", None)
            if callable(diagnostics_method):
                model_diagnostics.setdefault(alias, {})["fit"] = diagnostics_method()

        corrected = forecast + correction
        component_frames = {
            name: pd.DataFrame(values, index=forecast.index).reindex(
                columns=list(RESIDUAL_LOAD_ALIASES)
            )
            for name, values in components.items()
        }
        for component_frame in component_frames.values():
            for alias in self.cold_start_aliases_:
                available = forecast[alias].notna()
                component_frame.loc[available, alias] = 0.0
        raw_missing = forecast.isna()
        if not correction.isna().equals(raw_missing):
            raise RuntimeError(
                "Le masque NaN de correction diffère du masque raw."
            )
        if not corrected.isna().equals(raw_missing):
            raise RuntimeError(
                "Le masque NaN corrigé diffère du masque raw."
            )
        for component_name, component_frame in component_frames.items():
            if not component_frame.isna().equals(raw_missing):
                raise RuntimeError(
                    f"Le masque NaN de la composante {component_name} diffère "
                    "du masque raw."
                )
        diagnostics: dict[str, Any] = {
            "mode": "live" if hasattr(self, "delivery_day_local_") else "fit_predict",
            "input_contract": "caller_materialized_asof",
            "label_end_utc": self.label_end_utc_.isoformat(),
            "training_start_utc": self.training_index_[0].isoformat(),
            "training_end_utc": self.training_index_[-1].isoformat(),
            "training_rows": len(self.training_index_),
            "minimum_error_lag_hours": MINIMUM_ERROR_LAG_HOURS,
            "minimum_training_rows": self.minimum_training_rows,
            "cold_start_policy": self.cold_start_policy,
            "cold_start_aliases": list(self.cold_start_aliases_),
            "clips_gw": dict(self.max_abs_correction_gw),
            "models": model_diagnostics,
        }
        if hasattr(self, "delivery_day_local_"):
            diagnostics.update(
                {
                    "delivery_day_local": self.delivery_day_local_.isoformat(),
                    "scheduled_origin_utc": self.scheduled_origin_utc_.isoformat(),
                    "runtime_cutoff_utc": self.runtime_cutoff_utc_.isoformat(),
                    "embargo_end_utc": self.embargo_end_utc_.isoformat(),
                }
            )
        return ResidualLoadCorrectionResult(
            raw=forecast,
            correction=correction,
            corrected=corrected,
            component_corrections=MappingProxyType(component_frames),
            diagnostics=MappingProxyType(diagnostics),
        )

    def correct_live(
        self,
        forecasts: pd.DataFrame,
        observations: pd.DataFrame,
        *,
        delivery_day: str | date | pd.Timestamp,
        runtime_cutoff_utc: Any | None = None,
    ) -> ResidualLoadCorrectionResult:
        """Fit with the live embargo and correct exactly one full local day."""

        day = _as_local_date(delivery_day, timezone=self.timezone)
        expected = local_delivery_day_index(day, timezone=self.timezone)
        all_forecasts = _forecast_frame(forecasts, name="forecasts")
        missing = expected.difference(all_forecasts.index)
        if len(missing):
            raise ResidualLoadInputCorrectionError(
                f"Le jour {day} est incomplet: {len(missing)} heure(s) absente(s)."
            )
        prediction = all_forecasts.loc[expected]
        self.fit_live(
            all_forecasts,
            observations,
            delivery_day=day,
            runtime_cutoff_utc=runtime_cutoff_utc,
        )
        return self.predict(prediction)


def _concat_results(
    results: Sequence[ResidualLoadCorrectionResult],
) -> ResidualLoadCorrectionResult:
    raw = pd.concat([result.raw for result in results])
    correction = pd.concat([result.correction for result in results])
    corrected = pd.concat([result.corrected for result in results])
    component_names = tuple(
        dict.fromkeys(
            name
            for result in results
            for name in result.component_corrections
        )
    )
    components = {
        name: pd.concat(
            [
                result.component_corrections.get(
                    name,
                    pd.DataFrame(
                        np.nan,
                        index=result.raw.index,
                        columns=list(RESIDUAL_LOAD_ALIASES),
                    ),
                )
                for result in results
            ]
        )
        for name in component_names
    }
    block_diagnostics = tuple(dict(result.diagnostics) for result in results)
    diagnostics = MappingProxyType(
        {
            "mode": "prequential",
            "input_contract": "caller_materialized_asof",
            "blocks": block_diagnostics,
            "block_count": len(results),
            "minimum_error_lag_hours": MINIMUM_ERROR_LAG_HOURS,
        }
    )
    return ResidualLoadCorrectionResult(
        raw=raw,
        correction=correction,
        corrected=corrected,
        component_corrections=MappingProxyType(components),
        diagnostics=diagnostics,
    )


def generate_prequential_residual_load_corrections(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    start_day: str | date | pd.Timestamp,
    end_day: str | date | pd.Timestamp,
    max_abs_correction_gw: float | Mapping[str, float | None] | None = 8.0,
    corrector_factory: CorrectorFactory | None = None,
    cold_start_policy: str = "raise",
    minimum_training_rows: int = 48,
    thread_count: int = -1,
    refit_every_days: int = 1,
    timezone: str = DELIVERY_TIMEZONE,
    forecast_change_lags_hours: Sequence[int] = (
        DEFAULT_FORECAST_CHANGE_LAGS_HOURS
    ),
    error_lags_hours: Sequence[int] = DEFAULT_ERROR_LAGS_HOURS,
    error_rolling_windows_hours: Sequence[int] = (
        DEFAULT_ERROR_ROLLING_WINDOWS_HOURS
    ),
) -> ResidualLoadCorrectionResult:
    """Generate strictly prequential corrections by full local-day blocks.

    By default each 23/24/25-hour block gets a fresh corrector.  A larger
    ``refit_every_days`` reuses the last fitted model between refit dates; that
    model has strictly less information than was available at later origins
    and is therefore still causal.  Every refit origin is local D-1 08:00 and
    its labels end at local D-2 23:00 (or the corresponding DST last hour).
    No fitted object or observed error is ever imported from a later block.
    This proves delivery-time causality, not complete
    vintage-selection causality: callers must materialise ``observations``
    point-in-time (snapshot/revision <= each refit origin) before calling it.
    Diagnostics record this contract as ``caller_materialized_asof``.
    """

    forecast = _forecast_frame(forecasts, name="forecasts")
    observed = _observation_frame(observations, name="observations")
    first = _as_local_date(start_day, timezone=timezone)
    last = _as_local_date(end_day, timezone=timezone)
    if last < first:
        raise ValueError("end_day doit être >= start_day.")
    if isinstance(refit_every_days, (bool, np.bool_)):
        raise ValueError("refit_every_days doit être un entier >= 1.")
    try:
        refit_interval = int(refit_every_days)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("refit_every_days doit être un entier >= 1.") from exc
    if refit_interval != refit_every_days or refit_interval < 1:
        raise ValueError("refit_every_days doit être un entier >= 1.")

    results: list[ResidualLoadCorrectionResult] = []
    current = first
    corrector: ResidualLoadInputCorrector | None = None
    refit_day: date | None = None
    day_number = 0
    while current <= last:
        block_index = local_delivery_day_index(current, timezone=timezone)
        if len(block_index) not in {23, 24, 25}:
            raise RuntimeError(f"Bloc local {current} invalide: {len(block_index)} h.")
        missing = block_index.difference(forecast.index)
        if len(missing):
            raise ResidualLoadInputCorrectionError(
                f"Le bloc local {current} est incomplet: {len(missing)} heure(s)."
            )
        if corrector is None or day_number % refit_interval == 0:
            corrector = ResidualLoadInputCorrector(
                max_abs_correction_gw=max_abs_correction_gw,
                corrector_factory=corrector_factory,
                cold_start_policy=cold_start_policy,
                minimum_training_rows=minimum_training_rows,
                thread_count=thread_count,
                timezone=timezone,
                forecast_change_lags_hours=forecast_change_lags_hours,
                error_lags_hours=error_lags_hours,
                error_rolling_windows_hours=error_rolling_windows_hours,
            )
            corrector.fit_live(
                forecast,
                observed,
                delivery_day=current,
                runtime_cutoff_utc=scheduled_live_origin_utc(
                    current,
                    timezone=timezone,
                ),
            )
            refit_day = current
        assert corrector is not None and refit_day is not None
        block = corrector.predict(forecast.loc[block_index])
        block_diagnostics = dict(block.diagnostics)
        block_diagnostics.update(
            {
                "prediction_delivery_day_local": current.isoformat(),
                "prediction_origin_utc": scheduled_live_origin_utc(
                    current,
                    timezone=timezone,
                ).isoformat(),
                "refit_delivery_day_local": refit_day.isoformat(),
                "refit_origin_utc": scheduled_live_origin_utc(
                    refit_day,
                    timezone=timezone,
                ).isoformat(),
                "reused_model": current != refit_day,
            }
        )
        results.append(
            ResidualLoadCorrectionResult(
                raw=block.raw,
                correction=block.correction,
                corrected=block.corrected,
                component_corrections=block.component_corrections,
                diagnostics=MappingProxyType(block_diagnostics),
            )
        )
        current += timedelta(days=1)
        day_number += 1

    if not results:
        raise RuntimeError("Aucun bloc préquentiel n'a été généré.")
    return _concat_results(results)


__all__ = [
    "CorrectorFactory",
    "DEFAULT_ERROR_LAGS_HOURS",
    "DEFAULT_ERROR_ROLLING_WINDOWS_HOURS",
    "DEFAULT_FORECAST_CHANGE_LAGS_HOURS",
    "DELIVERY_TIMEZONE",
    "EXPECTED_ALIASES",
    "MINIMUM_ERROR_LAG_HOURS",
    "OBSERVED_SERIES_BY_ALIAS",
    "RESIDUAL_LOAD_ALIASES",
    "RESIDUAL_LOAD_ALIAS_BY_OBS_SERIES",
    "RESIDUAL_LOAD_COUNTRIES",
    "RESIDUAL_LOAD_OBS_SERIES_BY_ALIAS",
    "ResidualLoadCorrectionResult",
    "ResidualLoadInputCorrectionError",
    "ResidualLoadInputCorrector",
    "build_residual_load_input_features",
    "conservative_label_end_utc",
    "generate_prequential_residual_load_corrections",
    "make_default_residual_load_corrector",
    "scheduled_live_origin_utc",
]
