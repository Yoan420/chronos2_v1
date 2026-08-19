"""Leakage-aware nonlinear correction of hourly quantile forecasts.

The corrector learns the residual ``actual - base_q50`` and applies the same
predicted shift to every base quantile.  Applying one common shift preserves
the interval width and the quantile order by construction.

Only information available at forecast time belongs in ``X`` and
``expert_predictions``.  In particular, expert predictions used for fitting
must be out-of-fold (or come from a genuinely sealed earlier model).  The
builder removes historical target-price features by default and derives daily
profiles only from day-ahead-known residual-load forecasts and Chronos
quantiles; the observed target is used solely as the regression label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from chronos2_hourly.features import (
    HourlyFeatureContractError,
    build_calendar_features,
    validate_utc_hourly_index,
)

from .base import OptionalDependencyError, require_frame
from .catboost_hourly import catboost_available


QUANTILE_COLUMNS: tuple[str, str, str] = ("q10", "q50", "q90")

DEFAULT_RESIDUAL_LOAD_PATTERNS: tuple[str, ...] = (
    r"(?i)(?:^|_)residual[_\- ]?(?:load|demand)(?:$|_(?:fcst|forecast)(?:_oracle)?$)",
    r"(?i)(?:^|_)charge[_\- ]?residuelle(?:$|_(?:fcst|forecast)(?:_oracle)?$)",
)

# These patterns deliberately target lags/rolling statistics of the observed
# target.  They do not remove forward-looking neighbour price forecasts.
DEFAULT_HISTORICAL_PRICE_PATTERNS: tuple[str, ...] = (
    r"(?i)(?:^|__)(?:price|target_price)_(?:lag|rolling)_",
    r"(?i)(?:^|__)(?:target|actual)(?:_price)?_(?:lag|rolling)_",
    r"(?i)(?:^|__)(?:historical|history)_(?:target_)?price(?:$|__|_)",
)

DEFAULT_DAY_OF_YEAR_PATTERNS: tuple[str, ...] = (
    r"(?i)day_?of_?year",
    r"(?i)dayofyear",
    r"(?i)(?:^|_)doy_(?:sin|cos)(?:$|__)",
    r"(?i)(?:^|__)season(?:al)?_day(?:$|__|_)",
)

FRENCH_FUNDAMENTAL_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "load": (
        re.compile(r"(?i)^known_fr_load_(?:fcst|forecast)(?:_oracle)?$"),
    ),
    "wind": (
        re.compile(
            r"(?i)^known_fr_(?:generation_)?wind(?:_generation)?_"
            r"(?:fcst|forecast)(?:_oracle)?$"
        ),
    ),
    "solar": (
        re.compile(
            r"(?i)^known_fr_(?:generation_)?solar(?:_generation)?_"
            r"(?:fcst|forecast)(?:_oracle)?$"
        ),
    ),
    "hydro_ror": (
        re.compile(
            r"(?i)^known_fr_(?:gma_)?(?:generation_)?hydro_ror(?:_generation)?_"
            r"(?:fcst|forecast)(?:_oracle)?$"
        ),
    ),
    "residual_load": (
        re.compile(
            r"(?i)^known_fr_residual_load_(?:fcst|forecast)(?:_oracle)?$"
        ),
    ),
    "nuclear": (
        re.compile(
            r"(?i)^known_fr_(?:generation_)?nuclear(?:_generation)?_"
            r"(?:fcst|forecast)(?:_long)?(?:_oracle)?$"
        ),
    ),
}


class ResidualCorrectionError(ValueError):
    """Raised when residual-correction inputs violate their causal schema."""


def _normalise_patterns(
    patterns: Sequence[str],
    *,
    name: str,
) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for value in patterns:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} doit contenir des expressions non vides.")
        try:
            compiled.append(re.compile(value))
        except re.error as exc:
            raise ValueError(f"Expression régulière invalide dans {name}: {value!r}.") from exc
    return tuple(compiled)


def _matches(column: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(column) is not None for pattern in patterns)


def _validate_index(index: pd.Index, *, name: str) -> pd.DatetimeIndex:
    try:
        # OOF folds may leave chronological gaps, hence no contiguity
        # requirement here.  Sorting, uniqueness, UTC and hourly alignment
        # remain strict and are never silently repaired.
        return validate_utc_hourly_index(
            index,
            name=name,
            require_contiguous=False,
        )
    except HourlyFeatureContractError as exc:
        raise ResidualCorrectionError(str(exc)) from exc


def _require_same_index(
    frame: pd.DataFrame,
    expected: pd.DatetimeIndex,
    *,
    name: str,
) -> None:
    _validate_index(frame.index, name=f"{name}.index")
    if not frame.index.equals(expected):
        raise ResidualCorrectionError(
            f"{name}.index doit être exactement égal à X.index; aucun "
            "réalignement implicite n'est autorisé."
        )


def _numeric_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    require_frame(frame, name=name)
    invalid = [
        column
        for column in frame.columns
        if not (
            pd.api.types.is_numeric_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        )
    ]
    if invalid:
        raise TypeError(
            f"{name} doit contenir uniquement des colonnes numériques; "
            f"colonnes invalides: {invalid}."
        )
    result = frame.astype(float)
    if bool(np.isinf(result.to_numpy(dtype=float, copy=False)).any()):
        raise ResidualCorrectionError(f"{name} contient des valeurs infinies.")
    return result


def _validate_base_predictions(
    predictions: pd.DataFrame,
    *,
    expected_index: pd.DatetimeIndex | None = None,
    name: str = "base_predictions",
) -> pd.DataFrame:
    require_frame(predictions, name=name)
    if expected_index is None:
        index = _validate_index(predictions.index, name=f"{name}.index")
    else:
        index = expected_index
        _require_same_index(predictions, expected_index, name=name)
    missing = [column for column in QUANTILE_COLUMNS if column not in predictions]
    if missing:
        raise ResidualCorrectionError(
            f"Quantiles absents de {name}: {missing}; attendu q10/q50/q90."
        )
    result = predictions.loc[:, list(QUANTILE_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ResidualCorrectionError(
            f"Les quantiles de {name} doivent tous être finis."
        )
    if bool(((result["q10"] > result["q50"]) | (result["q50"] > result["q90"])).any()):
        raise ResidualCorrectionError(
            f"Les quantiles de {name} se croisent; il faut q10 <= q50 <= q90."
        )
    result.index = index.copy()
    return result.astype(float)


def _validate_expert_predictions(
    predictions: pd.DataFrame,
    *,
    expected_index: pd.DatetimeIndex,
    name: str = "expert_predictions",
) -> pd.DataFrame:
    require_frame(predictions, name=name)
    _require_same_index(predictions, expected_index, name=name)
    renamed: dict[str, str] = {}
    groups: dict[str, set[str]] = {}
    for raw_column in predictions.columns:
        if not isinstance(raw_column, str):
            raise TypeError(f"Les colonnes de {name} doivent être des chaînes.")
        if raw_column in QUANTILE_COLUMNS:
            model, quantile = "expert", raw_column
            renamed[raw_column] = f"expert__{quantile}"
        elif "__" in raw_column:
            model, quantile = raw_column.rsplit("__", 1)
            if not model or quantile not in QUANTILE_COLUMNS:
                raise ResidualCorrectionError(
                    f"Colonne experte invalide: {raw_column!r}; utilisez "
                    "<modèle>__q10, <modèle>__q50 ou <modèle>__q90."
                )
            renamed[raw_column] = raw_column
        else:
            # A bare expert name follows the existing ensemble convention and
            # represents a median-only forecast.
            model, quantile = raw_column, "q50"
            renamed[raw_column] = f"{model}__q50"
        if not model:
            raise ResidualCorrectionError(f"Nom d'expert vide dans {raw_column!r}.")
        if quantile in groups.setdefault(model, set()):
            raise ResidualCorrectionError(
                f"Prédiction dupliquée pour l'expert {model!r}/{quantile}."
            )
        groups[model].add(quantile)
    missing_median = sorted(model for model, labels in groups.items() if "q50" not in labels)
    if missing_median:
        raise ResidualCorrectionError(
            f"Prévision q50 absente pour les experts {missing_median}."
        )

    numeric = predictions.rename(columns=renamed).apply(pd.to_numeric, errors="coerce")
    if not numeric.columns.is_unique:
        duplicates = numeric.columns[numeric.columns.duplicated()].tolist()
        raise ResidualCorrectionError(
            f"Colonnes expertes dupliquées après normalisation: {duplicates}."
        )
    if bool(np.isinf(numeric.to_numpy(dtype=float, copy=False)).any()):
        raise ResidualCorrectionError(f"{name} contient des valeurs infinies.")
    empty = [column for column in numeric if numeric[column].notna().sum() == 0]
    if empty:
        raise ResidualCorrectionError(
            f"Colonnes expertes entièrement manquantes: {empty}."
        )
    return numeric.astype(float)


def _normalise_correction_bounds(
    correction_clip: float | Sequence[float] | None,
    max_abs_correction: float | None,
) -> tuple[float, float] | None:
    if correction_clip is None:
        if max_abs_correction is None:
            return None
        maximum = float(max_abs_correction)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("max_abs_correction doit être fini et strictement positif.")
        return (-maximum, maximum)
    if isinstance(correction_clip, (int, float, np.integer, np.floating)):
        maximum = float(correction_clip)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("correction_clip scalaire doit être fini et positif.")
        return (-maximum, maximum)
    values = tuple(float(value) for value in correction_clip)
    if len(values) != 2:
        raise ValueError("correction_clip doit contenir exactement (minimum, maximum).")
    lower, upper = values
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("Bornes correction_clip invalides.")
    return (lower, upper)


def apply_residual_correction(
    base_predictions: pd.DataFrame,
    correction: pd.Series | Sequence[float] | np.ndarray,
    *,
    correction_scale: float = 1.0,
    correction_clip: float | Sequence[float] | None = None,
    max_abs_correction: float | None = None,
) -> pd.DataFrame:
    """Apply one scaled/clipped shift to q10, q50 and q90.

    ``max_abs_correction`` is a convenience symmetric bound.  An explicit
    ``correction_clip`` scalar (symmetric) or ``(lower, upper)`` pair takes
    precedence when both are supplied.
    """

    base = _validate_base_predictions(base_predictions)
    scale = float(correction_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("correction_scale doit être fini et positif ou nul.")
    if isinstance(correction, pd.Series):
        if not correction.index.equals(base.index):
            raise ResidualCorrectionError(
                "correction.index doit être exactement égal à base_predictions.index."
            )
        raw = pd.to_numeric(correction, errors="coerce").to_numpy(dtype=float)
    else:
        raw = pd.to_numeric(pd.Series(np.asarray(correction)), errors="coerce").to_numpy(
            dtype=float
        )
    if raw.ndim != 1 or len(raw) != len(base):
        raise ResidualCorrectionError(
            "correction doit être unidimensionnelle et de même longueur que la base."
        )
    if not np.isfinite(raw).all():
        raise ResidualCorrectionError("correction doit contenir des valeurs finies.")
    applied = raw * scale
    bounds = _normalise_correction_bounds(correction_clip, max_abs_correction)
    if bounds is not None:
        applied = np.clip(applied, bounds[0], bounds[1])

    result = base.add(applied, axis=0)
    # This is an invariant rather than a repair: one common shift must not
    # alter the order or the interval width.
    if bool(((result["q10"] > result["q50"]) | (result["q50"] > result["q90"])).any()):
        raise RuntimeError("La correction commune a violé l'ordre des quantiles.")
    result.attrs["residual_correction"] = pd.Series(
        applied,
        index=result.index,
        name="residual_correction",
    )
    result.attrs["correction_scale"] = scale
    result.attrs["correction_clip"] = bounds
    return result


class ResidualMetaFeatureBuilder:
    """Build and freeze causal features for :class:`ResidualCorrector`.

    Raw numeric ``X`` columns and expert quantiles are retained.  Daily
    statistics/ramp features are generated for residual-load inputs and
    Chronos quantiles.  They are causal for a day-ahead auction because those
    full delivery-day profiles are known at the forecast origin; no actual
    target value is read by this builder.
    """

    def __init__(
        self,
        *,
        timezone: str = "Europe/Paris",
        include_calendar: bool = True,
        include_rich_calendar: bool = False,
        rich_calendar_countries: Sequence[str] = ("FR", "DE", "BE", "ES", "NL"),
        rich_calendar_primary_country: str = "FR",
        include_daily_profiles: bool = True,
        include_fundamental_interactions: bool = True,
        include_missing_indicators: bool = True,
        profile_columns: Sequence[str] | None = None,
        residual_load_patterns: Sequence[str] = DEFAULT_RESIDUAL_LOAD_PATTERNS,
        chronos_prefixes: Sequence[str] = ("chronos",),
        exclude_historical_prices: bool = True,
        historical_price_patterns: Sequence[str] = DEFAULT_HISTORICAL_PRICE_PATTERNS,
        exclude_day_of_year: bool = True,
        day_of_year_patterns: Sequence[str] = DEFAULT_DAY_OF_YEAR_PATTERNS,
        exclude_columns: Sequence[str] = (),
        exclude_patterns: Sequence[str] = (),
    ) -> None:
        self.timezone = str(timezone)
        self.include_calendar = bool(include_calendar)
        self.include_rich_calendar = bool(include_rich_calendar)
        countries = tuple(dict.fromkeys(str(value).upper() for value in rich_calendar_countries))
        primary = str(rich_calendar_primary_country).upper()
        if not countries or any(not value for value in countries) or not primary:
            raise ValueError("Les pays du calendrier riche ne peuvent pas être vides.")
        if primary not in countries:
            countries = (*countries, primary)
        self.rich_calendar_countries = countries
        self.rich_calendar_primary_country = primary
        self.include_daily_profiles = bool(include_daily_profiles)
        self.include_fundamental_interactions = bool(
            include_fundamental_interactions
        )
        self.include_missing_indicators = bool(include_missing_indicators)
        self.profile_columns = None if profile_columns is None else tuple(profile_columns)
        if self.profile_columns is not None and any(
            not isinstance(value, str) or not value for value in self.profile_columns
        ):
            raise ValueError("profile_columns doit contenir des noms non vides.")
        self.residual_load_patterns = tuple(residual_load_patterns)
        self._residual_patterns = _normalise_patterns(
            self.residual_load_patterns,
            name="residual_load_patterns",
        )
        self.chronos_prefixes = tuple(str(value).lower() for value in chronos_prefixes)
        if any(not value for value in self.chronos_prefixes):
            raise ValueError("chronos_prefixes contient un préfixe vide.")
        self.exclude_historical_prices = bool(exclude_historical_prices)
        self.historical_price_patterns = tuple(historical_price_patterns)
        self._historical_price_patterns = _normalise_patterns(
            self.historical_price_patterns,
            name="historical_price_patterns",
        )
        self.exclude_day_of_year = bool(exclude_day_of_year)
        self.day_of_year_patterns = tuple(day_of_year_patterns)
        self._day_of_year_patterns = _normalise_patterns(
            self.day_of_year_patterns,
            name="day_of_year_patterns",
        )
        self.exclude_columns = tuple(exclude_columns)
        if any(not isinstance(value, str) or not value for value in self.exclude_columns):
            raise ValueError("exclude_columns doit contenir des noms non vides.")
        self.exclude_patterns = tuple(exclude_patterns)
        self._exclude_patterns = _normalise_patterns(
            self.exclude_patterns,
            name="exclude_patterns",
        )

    def _is_excluded(self, column: str) -> bool:
        if column in self.exclude_columns or _matches(column, self._exclude_patterns):
            return True
        if self.exclude_historical_prices and _matches(
            column,
            self._historical_price_patterns,
        ):
            return True
        if self.exclude_day_of_year and _matches(column, self._day_of_year_patterns):
            return True
        return False

    def _calendar(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        # Deterministic calendar features can safely be generated on a
        # contiguous envelope when OOF rows have gaps.
        envelope = pd.date_range(index[0], index[-1], freq="h", tz="UTC")
        calendar = build_calendar_features(envelope, timezone=self.timezone).loc[index]
        return calendar.loc[:, [column for column in calendar if not self._is_excluded(column)]]

    def _rich_calendar(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        # Import lazily: `holidays` remains optional unless this feature family
        # is requested.
        from chronos2_modular.exogenous_extensions import rich_calendar_frame

        try:
            local = index.tz_convert(self.timezone)
        except (TypeError, ValueError, KeyError) as exc:
            raise ResidualCorrectionError(
                f"Fuseau de livraison invalide: {self.timezone!r}."
            ) from exc

        # rich_calendar_frame compares adjacent local days for DST.  A live
        # frame often contains one day only, so evaluate on a padded local
        # envelope and select the requested timestamps afterwards.
        first_day = local[0].date() - timedelta(days=1)
        last_exclusive = local[-1].date() + timedelta(days=2)
        expanded = pd.date_range(
            pd.Timestamp(first_day).tz_localize(self.timezone),
            pd.Timestamp(last_exclusive).tz_localize(self.timezone),
            freq="h",
            inclusive="left",
        )
        rich = rich_calendar_frame(
            expanded,
            list(self.rich_calendar_countries),
            self.rich_calendar_primary_country,
        ).loc[local]
        rich.index = index.copy()

        # Compute this flag from the timezone itself, independent of whether
        # neighbouring rows happened to be present in the caller's frame.
        transition_by_day: dict[object, float] = {}
        for day in dict.fromkeys(local.date):
            start = pd.Timestamp(day).tz_localize(self.timezone)
            next_start = pd.Timestamp(day + timedelta(days=1)).tz_localize(self.timezone)
            transition_by_day[day] = float(start.utcoffset() != next_start.utcoffset())
        rich["known_cal_dst_transition_day_oracle"] = np.asarray(
            [transition_by_day[day] for day in local.date],
            dtype=np.float32,
        )
        return rich.loc[:, [column for column in rich if not self._is_excluded(column)]]

    def _daily_profiles(
        self,
        frame: pd.DataFrame,
        *,
        source_columns: Sequence[str],
    ) -> pd.DataFrame:
        if not source_columns:
            return pd.DataFrame(index=frame.index.copy())
        derived: dict[str, pd.Series] = {}
        local_days = pd.Series(
            frame.index.tz_convert(self.timezone).date,
            index=frame.index,
            name="local_delivery_day",
        )
        for column in source_columns:
            values = frame[column].astype(float)
            grouped = values.groupby(local_days, sort=False)
            mean = grouped.transform("mean")
            minimum = grouped.transform("min")
            maximum = grouped.transform("max")
            std = grouped.transform(lambda part: part.std(ddof=0))
            ramp = grouped.diff()
            position_in_day = grouped.cumcount()
            ramp = ramp.mask(position_in_day.eq(0), 0.0)
            ramp_2h = grouped.diff(2).mask(position_in_day.lt(2), 0.0)
            ramp_up = ramp.clip(lower=0.0)
            ramp_down = (-ramp).clip(lower=0.0)
            derived[f"{column}__day_mean"] = mean
            derived[f"{column}__day_std"] = std
            derived[f"{column}__day_min"] = minimum
            derived[f"{column}__day_max"] = maximum
            derived[f"{column}__day_range"] = maximum - minimum
            derived[f"{column}__day_centered"] = values - mean
            derived[f"{column}__ramp_1h"] = ramp
            derived[f"{column}__abs_ramp_1h"] = ramp.abs()
            derived[f"{column}__ramp_2h"] = ramp_2h
            derived[f"{column}__abs_ramp_2h"] = ramp_2h.abs()
            derived[f"{column}__day_ramp_up_max"] = ramp_up.groupby(
                local_days,
                sort=False,
            ).transform("max")
            derived[f"{column}__day_ramp_down_max"] = ramp_down.groupby(
                local_days,
                sort=False,
            ).transform("max")
            derived[f"{column}__day_ramp_abs_max"] = ramp.abs().groupby(
                local_days,
                sort=False,
            ).transform("max")
        return pd.DataFrame(derived, index=frame.index.copy())

    @staticmethod
    def _first_matching_column(
        frame: pd.DataFrame,
        patterns: Sequence[re.Pattern[str]],
    ) -> str | None:
        return next(
            (
                column
                for column in frame
                if any(pattern.search(column) is not None for pattern in patterns)
            ),
            None,
        )

    def _fundamental_interactions(
        self,
        selected_x: pd.DataFrame,
    ) -> tuple[pd.DataFrame, tuple[str, ...]]:
        """Build nonlinear FR supply/demand variables known before auction."""

        columns = {
            family: self._first_matching_column(selected_x, patterns)
            for family, patterns in FRENCH_FUNDAMENTAL_PATTERNS.items()
        }
        load_column = columns["load"]
        wind_column = columns["wind"]
        solar_column = columns["solar"]
        if load_column is None or wind_column is None or solar_column is None:
            return pd.DataFrame(index=selected_x.index.copy()), ()

        load = selected_x[load_column]
        wind = selected_x[wind_column]
        solar = selected_x[solar_column]
        variable_renewables = wind + solar
        net_load = load - variable_renewables
        denominator = load.abs().clip(lower=1.0)
        values: dict[str, pd.Series] = {
            "fr_variable_renewables_fcst": variable_renewables,
            "fr_net_load_wind_solar_fcst": net_load,
            "fr_variable_renewables_share": variable_renewables / denominator,
            "fr_wind_share": wind / denominator,
            "fr_solar_share": solar / denominator,
            "fr_wind_solar_interaction": wind * solar,
        }
        profile_columns = [
            load_column,
            wind_column,
            solar_column,
            *values.keys(),
        ]

        hydro_column = columns["hydro_ror"]
        if hydro_column is not None:
            hydro = selected_x[hydro_column]
            values["fr_variable_renewables_hydro_fcst"] = (
                variable_renewables + hydro
            )
            values["fr_net_load_including_hydro_fcst"] = net_load - hydro
            values["fr_hydro_ror_share"] = hydro / denominator
            profile_columns.extend(
                [
                    hydro_column,
                    "fr_variable_renewables_hydro_fcst",
                    "fr_net_load_including_hydro_fcst",
                    "fr_hydro_ror_share",
                ]
            )

        residual_column = columns["residual_load"]
        if residual_column is not None:
            values["fr_residual_load_identity_gap"] = (
                selected_x[residual_column] - net_load
            )
            profile_columns.append("fr_residual_load_identity_gap")

        nuclear_column = columns["nuclear"]
        if nuclear_column is not None:
            nuclear = selected_x[nuclear_column]
            values["fr_net_load_after_nuclear_fcst"] = net_load - nuclear
            values["fr_nuclear_share_of_load"] = nuclear / denominator
            if residual_column is not None:
                values["fr_residual_load_after_nuclear_fcst"] = (
                    selected_x[residual_column] - nuclear
                )
            profile_columns.extend(
                [
                    nuclear_column,
                    "fr_net_load_after_nuclear_fcst",
                    "fr_nuclear_share_of_load",
                ]
            )
            if residual_column is not None:
                profile_columns.append("fr_residual_load_after_nuclear_fcst")

        # Hinge variables focus capacity on the oversupply regimes that
        # dominate negative-price errors, without reading any realised price.
        for threshold in (30.0, 20.0, 10.0, 0.0):
            label = str(int(threshold)).replace("-", "minus")
            values[f"fr_net_load_below_{label}_gw"] = (
                threshold - net_load
            ).clip(lower=0.0)

        result = pd.DataFrame(values, index=selected_x.index.copy())
        return result, tuple(dict.fromkeys(profile_columns))

    def _missing_indicators(self, selected_x: pd.DataFrame) -> pd.DataFrame:
        if not self.include_missing_indicators:
            return pd.DataFrame(index=selected_x.index.copy())
        columns = [
            column
            for column in selected_x
            if column.startswith("known_") and column.endswith("_oracle")
        ]
        return pd.DataFrame(
            {
                f"{column}__missing": selected_x[column].isna().astype(float)
                for column in columns
            },
            index=selected_x.index.copy(),
        )

    def _chronos_spreads(self, experts: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=experts.index.copy())
        models: dict[str, dict[str, str]] = {}
        for column in experts:
            model, quantile = column.rsplit("__", 1)
            if model.lower().startswith(self.chronos_prefixes):
                models.setdefault(model, {})[quantile] = column
        for model, columns in models.items():
            if set(QUANTILE_COLUMNS).issubset(columns):
                q10 = experts[columns["q10"]]
                q50 = experts[columns["q50"]]
                q90 = experts[columns["q90"]]
                result[f"{model}__interval_width"] = q90 - q10
                result[f"{model}__lower_width"] = q50 - q10
                result[f"{model}__upper_width"] = q90 - q50
                result[f"{model}__interval_asymmetry"] = (q90 - q50) - (q50 - q10)
        return result

    def _expert_disagreement(self, experts: pd.DataFrame) -> pd.DataFrame:
        """Summarise disagreement without reading any realised price."""

        q50_columns = [
            column
            for column in experts
            if column.endswith("__q50")
            and not column.lower().startswith("base__")
        ]
        result = pd.DataFrame(index=experts.index.copy())
        if len(q50_columns) < 2:
            return result
        medians = experts.loc[:, q50_columns]
        result["expert_q50_mean"] = medians.mean(axis=1)
        result["expert_q50_std"] = medians.std(axis=1, ddof=0)
        result["expert_q50_min"] = medians.min(axis=1)
        result["expert_q50_max"] = medians.max(axis=1)
        result["expert_q50_range"] = (
            result["expert_q50_max"] - result["expert_q50_min"]
        )
        chronos_columns = [
            column
            for column in q50_columns
            if column.rsplit("__", 1)[0].lower().startswith(self.chronos_prefixes)
        ]
        if chronos_columns:
            chronos_column = chronos_columns[0]
            for column in q50_columns:
                if column == chronos_column:
                    continue
                model = column.rsplit("__", 1)[0]
                result[f"chronos_minus_{model}__q50"] = (
                    experts[chronos_column] - experts[column]
                )
        return result

    def _residual_load_interactions(self, selected_x: pd.DataFrame) -> pd.DataFrame:
        """Compare the French day-ahead curve with neighbouring systems."""

        by_country: dict[str, str] = {}
        for column in selected_x:
            if not _matches(column, self._residual_patterns):
                continue
            match = re.search(
                r"(?i)(?:^|_)(fr|de|be|nl|es)_residual[_-]?load",
                column,
            )
            if match is not None:
                by_country.setdefault(match.group(1).lower(), column)
        result = pd.DataFrame(index=selected_x.index.copy())
        fr_column = by_country.get("fr")
        neighbour_columns = [
            by_country[country]
            for country in ("de", "be", "nl", "es")
            if country in by_country
        ]
        if fr_column is None or not neighbour_columns:
            return result
        neighbours = selected_x.loc[:, neighbour_columns]
        result["residual_load_neighbour_mean"] = neighbours.mean(axis=1)
        result["residual_load_neighbour_std"] = neighbours.std(axis=1, ddof=0)
        result["residual_load_neighbour_min"] = neighbours.min(axis=1)
        result["residual_load_neighbour_max"] = neighbours.max(axis=1)
        result["residual_load_neighbour_range"] = (
            result["residual_load_neighbour_max"]
            - result["residual_load_neighbour_min"]
        )
        result["fr_minus_neighbour_mean_residual_load"] = (
            selected_x[fr_column] - result["residual_load_neighbour_mean"]
        )
        for country in ("de", "be", "nl", "es"):
            if country in by_country:
                result[f"fr_minus_{country}_residual_load"] = (
                    selected_x[fr_column] - selected_x[by_country[country]]
                )
        return result

    def _build(
        self,
        X: pd.DataFrame,
        expert_predictions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, tuple[str, ...]]:
        require_frame(X, name="X")
        index = _validate_index(X.index, name="X.index")
        if any(not isinstance(column, str) for column in X.columns):
            raise TypeError("Les noms de colonnes de X doivent être des chaînes.")
        selected_x_columns = tuple(column for column in X if not self._is_excluded(column))
        if not selected_x_columns:
            raise ResidualCorrectionError("Toutes les colonnes de X ont été exclues.")
        selected_x = _numeric_frame(X.loc[:, list(selected_x_columns)], name="X sélectionné")
        empty = [column for column in selected_x if selected_x[column].notna().sum() == 0]
        if empty:
            raise ResidualCorrectionError(
                f"Features X entièrement manquantes dans l'échantillon: {empty}."
            )
        experts = _validate_expert_predictions(
            expert_predictions,
            expected_index=index,
        )
        collisions = sorted(set(selected_x).intersection(experts))
        if collisions:
            raise ResidualCorrectionError(
                f"Collision entre X et les prédictions expertes: {collisions}."
            )
        parts = [selected_x, experts]
        fundamental_profiles: tuple[str, ...] = ()
        if self.include_fundamental_interactions:
            fundamentals, fundamental_profiles = self._fundamental_interactions(
                selected_x
            )
            if not fundamentals.empty:
                parts.append(fundamentals)
        missing_indicators = self._missing_indicators(selected_x)
        if not missing_indicators.empty:
            parts.append(missing_indicators)
        load_interactions = self._residual_load_interactions(selected_x)
        if not load_interactions.empty:
            parts.append(load_interactions)
        disagreement = self._expert_disagreement(experts)
        if not disagreement.empty:
            parts.append(disagreement)
        spreads = self._chronos_spreads(experts)
        if not spreads.empty:
            parts.append(spreads)

        if self.include_calendar:
            calendar = self._calendar(index)
            # Existing hourly X matrices already carry these deterministic
            # columns.  Keep their original copy and generate only missing
            # fields, avoiding duplicate schemas.
            calendar = calendar.loc[
                :,
                [column for column in calendar if column not in X.columns and column not in experts],
            ]
            if not calendar.empty:
                parts.append(calendar)
        if self.include_rich_calendar:
            rich = self._rich_calendar(index)
            rich = rich.loc[
                :,
                [column for column in rich if column not in X.columns and column not in experts],
            ]
            if not rich.empty:
                parts.append(rich)

        base = pd.concat(parts, axis=1)
        if not base.columns.is_unique:
            duplicates = base.columns[base.columns.duplicated()].tolist()
            raise ResidualCorrectionError(f"Meta-features dupliquées: {duplicates}.")

        if self.include_daily_profiles:
            if self.profile_columns is not None:
                missing_profile = [column for column in self.profile_columns if column not in base]
                if missing_profile:
                    raise ResidualCorrectionError(
                        f"profile_columns absentes des entrées: {missing_profile}."
                    )
                profile_columns = list(self.profile_columns)
            else:
                profile_columns = [
                    column
                    for column in selected_x
                    if _matches(column, self._residual_patterns)
                ]
                profile_columns.extend(
                    column
                    for column in experts
                    if column.rsplit("__", 1)[0].lower().startswith(self.chronos_prefixes)
                )
                profile_columns.extend(
                    column
                    for column in spreads
                    if column.endswith("__interval_width")
                )
                profile_columns.extend(fundamental_profiles)
            profiles = self._daily_profiles(base, source_columns=profile_columns)
            if not profiles.empty:
                parts.append(profiles)

        result = pd.concat(parts, axis=1).astype(float)
        if not result.columns.is_unique:
            duplicates = result.columns[result.columns.duplicated()].tolist()
            raise ResidualCorrectionError(f"Meta-features dupliquées: {duplicates}.")
        if bool(np.isinf(result.to_numpy(dtype=float, copy=False)).any()):
            raise ResidualCorrectionError("Les meta-features contiennent des infinis.")
        excluded = tuple(column for column in X if column not in selected_x_columns)
        return result, excluded

    @staticmethod
    def _schema_error(
        received: Sequence[str],
        expected: Sequence[str],
        *,
        name: str,
    ) -> ResidualCorrectionError:
        missing = [column for column in expected if column not in received]
        unexpected = [column for column in received if column not in expected]
        order_changed = not missing and not unexpected and tuple(received) != tuple(expected)
        return ResidualCorrectionError(
            f"Schéma {name} différent de fit: missing={missing}, "
            f"unexpected={unexpected}, order_changed={order_changed}."
        )

    def fit(
        self,
        X: pd.DataFrame,
        expert_predictions: pd.DataFrame,
    ) -> "ResidualMetaFeatureBuilder":
        features, excluded = self._build(X, expert_predictions)
        self.x_columns_in_ = tuple(X.columns)
        self.expert_columns_in_ = tuple(expert_predictions.columns)
        self.feature_columns_ = tuple(features.columns)
        self.excluded_columns_ = excluded
        self.is_fitted_ = True
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        expert_predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        features, excluded = self._build(X, expert_predictions)
        self.x_columns_in_ = tuple(X.columns)
        self.expert_columns_in_ = tuple(expert_predictions.columns)
        self.feature_columns_ = tuple(features.columns)
        self.excluded_columns_ = excluded
        self.is_fitted_ = True
        return features

    def transform(
        self,
        X: pd.DataFrame,
        expert_predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("ResidualMetaFeatureBuilder doit être ajusté avant transform().")
        if tuple(X.columns) != self.x_columns_in_:
            raise self._schema_error(X.columns, self.x_columns_in_, name="X")
        if tuple(expert_predictions.columns) != self.expert_columns_in_:
            raise self._schema_error(
                expert_predictions.columns,
                self.expert_columns_in_,
                name="expert_predictions",
            )
        features, _ = self._build(X, expert_predictions)
        if tuple(features.columns) != self.feature_columns_:
            raise self._schema_error(
                features.columns,
                self.feature_columns_,
                name="meta-features",
            )
        return features

    def get_feature_names_out(self) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("ResidualMetaFeatureBuilder n'est pas encore ajusté.")
        return np.asarray(self.feature_columns_, dtype=object)


def build_residual_meta_features(
    X: pd.DataFrame,
    expert_predictions: pd.DataFrame,
    **builder_options: Any,
) -> pd.DataFrame:
    """Stateless convenience wrapper around ``fit_transform``."""

    return ResidualMetaFeatureBuilder(**builder_options).fit_transform(
        X,
        expert_predictions,
    )


class ResidualCorrector:
    """CatBoost residual model with a deterministic sklearn fallback."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        feature_builder: ResidualMetaFeatureBuilder | None = None,
        feature_builder_options: Mapping[str, Any] | None = None,
        correction_scale: float = 1.0,
        correction_clip: float | Sequence[float] | None = None,
        max_abs_correction: float | None = 30.0,
        min_training_rows: int = 48,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.035,
        l2_leaf_reg: float = 8.0,
        min_samples_leaf: int = 30,
        sklearn_early_stopping: bool | str = "auto",
        random_state: int = 42,
        thread_count: int = -1,
        verbose: bool | int = False,
    ) -> None:
        if backend not in {"auto", "catboost", "sklearn"}:
            raise ValueError("backend doit valoir auto, catboost ou sklearn.")
        if feature_builder is not None and feature_builder_options:
            raise ValueError(
                "Fournissez feature_builder ou feature_builder_options, pas les deux."
            )
        if feature_builder is not None and not isinstance(
            feature_builder,
            ResidualMetaFeatureBuilder,
        ):
            raise TypeError("feature_builder doit être un ResidualMetaFeatureBuilder.")
        scale = float(correction_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("correction_scale doit être fini et positif ou nul.")
        self.backend = backend
        self.feature_builder = feature_builder
        self.feature_builder_options = dict(feature_builder_options or {})
        self.correction_scale = scale
        self.correction_clip = correction_clip
        self.max_abs_correction = max_abs_correction
        self.correction_bounds_ = _normalise_correction_bounds(
            correction_clip,
            max_abs_correction,
        )
        if min_training_rows < 2:
            raise ValueError("min_training_rows doit être >= 2.")
        if iterations < 1 or depth < 1 or depth > 16:
            raise ValueError("iterations doit être >= 1 et depth compris entre 1 et 16.")
        if learning_rate <= 0.0 or l2_leaf_reg < 0.0 or min_samples_leaf < 1:
            raise ValueError(
                "learning_rate doit être positif, l2_leaf_reg >= 0 et "
                "min_samples_leaf >= 1."
            )
        if sklearn_early_stopping not in {True, False, "auto"}:
            raise ValueError(
                "sklearn_early_stopping doit valoir true, false ou 'auto'."
            )
        self.min_training_rows = int(min_training_rows)
        self.iterations = int(iterations)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.min_samples_leaf = int(min_samples_leaf)
        self.sklearn_early_stopping = sklearn_early_stopping
        self.random_state = int(random_state)
        self.thread_count = int(thread_count)
        self.verbose = verbose

    def _resolve_backend(self) -> str:
        available = catboost_available()
        if self.backend == "catboost" and not available:
            raise OptionalDependencyError(
                "CatBoost a été demandé pour ResidualCorrector mais n'est pas "
                "installé. Exécutez `python -m pip install catboost>=1.2` ou "
                "utilisez backend='sklearn'."
            )
        if self.backend == "auto":
            return "catboost" if available else "sklearn"
        return self.backend

    def _new_model(self) -> Any:
        if self.backend_ == "catboost":
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                loss_function="MAE",
                eval_metric="MAE",
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                random_seed=self.random_state,
                thread_count=self.thread_count,
                allow_writing_files=False,
                verbose=self.verbose,
                has_time=True,
                nan_mode="Min",
            )
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="absolute_error",
                        learning_rate=self.learning_rate,
                        max_iter=self.iterations,
                        max_leaf_nodes=max(3, 2**self.depth - 1),
                        min_samples_leaf=self.min_samples_leaf,
                        l2_regularization=self.l2_leaf_reg,
                        early_stopping=self.sklearn_early_stopping,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    @staticmethod
    def _target(
        y: pd.Series | Sequence[float] | np.ndarray,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise TypeError("y doit être unidimensionnel.")
            if not y.index.equals(index):
                raise ResidualCorrectionError("y.index doit être exactement égal à X.index.")
            raw = y.iloc[:, 0].to_numpy(copy=False)
        elif isinstance(y, pd.Series):
            if not y.index.equals(index):
                raise ResidualCorrectionError("y.index doit être exactement égal à X.index.")
            raw = y.to_numpy(copy=False)
        else:
            raw = np.asarray(y)
        if raw.ndim != 1 or len(raw) != len(index):
            raise ResidualCorrectionError(
                "y doit être unidimensionnel et de même longueur que X."
            )
        target = pd.to_numeric(pd.Series(raw, index=index), errors="coerce").astype(float)
        if bool(np.isinf(target.to_numpy(dtype=float)).any()):
            raise ResidualCorrectionError("y contient des valeurs infinies.")
        return target

    @staticmethod
    def _combined_predictions(
        base: pd.DataFrame,
        experts: pd.DataFrame | None,
    ) -> pd.DataFrame:
        base_meta = base.rename(columns={column: f"base__{column}" for column in QUANTILE_COLUMNS})
        if experts is None:
            return base_meta
        validated = _validate_expert_predictions(
            experts,
            expected_index=base.index,
        )
        collisions = sorted(set(base_meta).intersection(validated))
        if collisions:
            raise ResidualCorrectionError(
                "expert_predictions utilise le préfixe réservé 'base': "
                f"{collisions}."
            )
        return pd.concat([base_meta, validated], axis=1)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[float] | np.ndarray,
        base_predictions: pd.DataFrame,
        expert_predictions: pd.DataFrame | None = None,
    ) -> "ResidualCorrector":
        require_frame(X, name="X")
        index = _validate_index(X.index, name="X.index")
        base = _validate_base_predictions(
            base_predictions,
            expected_index=index,
        )
        combined = self._combined_predictions(base, expert_predictions)
        builder = self.feature_builder or ResidualMetaFeatureBuilder(
            **self.feature_builder_options
        )
        meta = builder.fit_transform(X, combined)
        target = self._target(y, index)
        residual = target - base["q50"]
        valid = residual.notna().to_numpy()
        if int(valid.sum()) < self.min_training_rows:
            raise ResidualCorrectionError(
                "Pas assez de résidus observés pour entraîner le correcteur: "
                f"{int(valid.sum())} < {self.min_training_rows}."
            )
        self.backend_ = self._resolve_backend()
        self.model_ = self._new_model()
        self.model_.fit(meta.loc[valid], residual.loc[valid].to_numpy(dtype=float))
        self.feature_builder_ = builder
        self.feature_columns_ = tuple(meta.columns)
        self.x_columns_in_ = tuple(X.columns)
        self.expert_columns_in_ = tuple(combined.columns)
        self.n_training_rows_ = int(valid.sum())
        self.residual_training_mae_before_ = float(residual.loc[valid].abs().mean())
        in_sample = np.asarray(self.model_.predict(meta.loc[valid]), dtype=float)
        if in_sample.shape != (self.n_training_rows_,) or not np.isfinite(in_sample).all():
            raise RuntimeError("Le backend a produit une correction d'entraînement invalide.")
        self.residual_training_mae_after_ = float(
            np.mean(np.abs(residual.loc[valid].to_numpy(dtype=float) - in_sample))
        )
        self.is_fitted_ = True
        return self

    def _meta_for_predict(
        self,
        X: pd.DataFrame,
        base_predictions: pd.DataFrame,
        expert_predictions: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("ResidualCorrector doit être entraîné avant predict().")
        require_frame(X, name="X")
        index = _validate_index(X.index, name="X.index")
        base = _validate_base_predictions(
            base_predictions,
            expected_index=index,
        )
        combined = self._combined_predictions(base, expert_predictions)
        meta = self.feature_builder_.transform(X, combined)
        return meta, base

    def predict_correction(
        self,
        X: pd.DataFrame,
        base_predictions: pd.DataFrame,
        expert_predictions: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Return the exact scaled/clipped shift that would be applied."""

        meta, _ = self._meta_for_predict(X, base_predictions, expert_predictions)
        raw = np.asarray(self.model_.predict(meta), dtype=float)
        if raw.shape != (len(meta),) or not np.isfinite(raw).all():
            raise RuntimeError("Le backend a produit une correction invalide.")
        applied = raw * self.correction_scale
        if self.correction_bounds_ is not None:
            applied = np.clip(
                applied,
                self.correction_bounds_[0],
                self.correction_bounds_[1],
            )
        result = pd.Series(applied, index=meta.index, name="residual_correction")
        result.attrs["raw_correction"] = pd.Series(
            raw,
            index=meta.index,
            name="raw_residual_correction",
        )
        return result

    def predict(
        self,
        X: pd.DataFrame,
        base_predictions: pd.DataFrame,
        expert_predictions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        meta, base = self._meta_for_predict(X, base_predictions, expert_predictions)
        raw = np.asarray(self.model_.predict(meta), dtype=float)
        if raw.shape != (len(meta),) or not np.isfinite(raw).all():
            raise RuntimeError("Le backend a produit une correction invalide.")
        result = apply_residual_correction(
            base,
            raw,
            correction_scale=self.correction_scale,
            correction_clip=self.correction_bounds_,
            max_abs_correction=None,
        )
        result.attrs["backend"] = self.backend_
        result.attrs["n_meta_features"] = len(self.feature_columns_)
        return result


__all__ = [
    "DEFAULT_DAY_OF_YEAR_PATTERNS",
    "DEFAULT_HISTORICAL_PRICE_PATTERNS",
    "DEFAULT_RESIDUAL_LOAD_PATTERNS",
    "QUANTILE_COLUMNS",
    "ResidualCorrectionError",
    "ResidualCorrector",
    "ResidualMetaFeatureBuilder",
    "apply_residual_correction",
    "build_residual_meta_features",
]
