"""Causal heat signals from daily national temperature *forecast* indices.

These are not observed temperatures, daily maxima, or official heatwave alerts.
Every source must already identify physical hours and pre-cutoff PIT vintages.
No source fetching, interpolation, model fitting or file writing occurs here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

COUNTRIES = ("FR", "DE", "BE", "NL", "ES")
TIME_COLUMNS = ("snapshot_time_utc", "value_time_utc", "revision_time_utc")
HEAT_FRACTION_ALIAS = "heat_fraction_fcst"
COOLING_MEAN_ALIAS = "cooling_degree_mean_fcst_c"


class HeatwaveFeatureError(ValueError):
    """An incomplete or noncausal temperature feature contract was refused."""


@dataclass(frozen=True)
class HeatwaveFeatureConfig:
    countries: tuple[str, ...] = COUNTRIES
    floor_thresholds_c: Mapping[str, float] = field(default_factory=lambda: {
        "FR": 20.0, "DE": 20.0, "BE": 20.0, "NL": 20.0, "ES": 24.0})
    lookback_days: int = 365
    threshold_quantile: float = 0.9
    minimum_history_days: int = 60
    streak_clip_days: int = 7
    persistent_days: int = 3
    cooling_threshold_c: float = 22.0
    include_cooling_mean: bool = True

    def validate(self) -> None:
        if tuple(self.countries) != COUNTRIES:
            raise HeatwaveFeatureError("This experiment requires exactly FR, DE, BE, NL, ES.")
        if set(self.floor_thresholds_c) != set(COUNTRIES) or any(
                isinstance(v, bool) or not np.isfinite(float(v)) or not -20 <= float(v) <= 45
                for v in self.floor_thresholds_c.values()):
            raise HeatwaveFeatureError("One finite plausible fixed temperature floor per country is required.")
        for name in ("lookback_days", "minimum_history_days", "streak_clip_days", "persistent_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise HeatwaveFeatureError(f"{name}: positive integer required.")
        if self.lookback_days != 365 or self.minimum_history_days > self.lookback_days:
            raise HeatwaveFeatureError("The threshold history must be rolling365; minimum_history_days <=365.")
        if self.streak_clip_days > 30 or self.persistent_days > self.streak_clip_days:
            raise HeatwaveFeatureError("persistent_days <= streak_clip_days <=30 required.")
        if isinstance(self.threshold_quantile, bool) or not 0 < float(self.threshold_quantile) < 1:
            raise HeatwaveFeatureError("threshold_quantile must be strictly between zero and one.")
        if isinstance(self.cooling_threshold_c, bool) or not np.isfinite(float(self.cooling_threshold_c)):
            raise HeatwaveFeatureError("cooling_threshold_c must be finite.")
        if not isinstance(self.include_cooling_mean, bool):
            raise HeatwaveFeatureError("include_cooling_mean must be boolean.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["countries"] = list(self.countries)
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None) -> "HeatwaveFeatureConfig":
        raw = dict(value or {})
        if "countries" in raw:
            raw["countries"] = tuple(raw["countries"])
        try:
            result = cls(**raw)
            result.validate()
        except (TypeError, ValueError) as exc:
            raise HeatwaveFeatureError(f"Invalid heat feature configuration: {exc}") from exc
        return result


def _config(value: HeatwaveFeatureConfig | Mapping[str, Any] | None) -> HeatwaveFeatureConfig:
    result = value if isinstance(value, HeatwaveFeatureConfig) else HeatwaveFeatureConfig.from_mapping(value)
    result.validate()
    return result


def temperature_aliases(config: HeatwaveFeatureConfig | Mapping[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(f"{country.lower()}_temperature_fcst" for country in _config(config).countries)


def heatwave_feature_aliases(config: HeatwaveFeatureConfig | Mapping[str, Any] | None = None) -> tuple[str, ...]:
    selected = _config(config)
    return (*temperature_aliases(selected),
            *(f"{c.lower()}_heat_excess_fcst_c" for c in selected.countries),
            *(f"{c.lower()}_heat_streak_fcst_days" for c in selected.countries),
            HEAT_FRACTION_ALIAS, *((COOLING_MEAN_ALIAS,) if selected.include_cooling_mean else ()))


def feature_metadata(config: HeatwaveFeatureConfig | Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    selected = _config(config)
    result = {}
    for country in selected.countries:
        for suffix, unit, semantic in (("temperature_fcst", "degC", "daily_national_temperature_forecast_index"),
                ("heat_excess_fcst_c", "degC", "excess_above_causal_rolling_threshold"),
                ("heat_streak_fcst_days", "day", "consecutive_positive_forecast_excess_days_clipped")):
            result[f"{country.lower()}_{suffix}"] = {"country": country, "unit": unit,
                "semantic": semantic, "daily_broadcast": True, "observed_temperature": False}
    result[HEAT_FRACTION_ALIAS] = {"country": "regional", "unit": "fraction",
        "semantic": "fraction_of_five_countries_with_persistent_forecast_heat", "daily_broadcast": True}
    if selected.include_cooling_mean:
        result[COOLING_MEAN_ALIAS] = {"country": "regional", "unit": "degC",
            "semantic": "unweighted_mean_forecast_cooling_degrees", "daily_broadcast": True}
    return result


def heatwave_features_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _float32_rounding_radius(values: np.ndarray) -> np.ndarray:
    """Half the adjacent binary32 bin width, expressed in binary64 units."""
    rounded = np.asarray(values, dtype=np.float32)
    centre = rounded.astype(np.float64)
    above = np.nextafter(rounded, np.float32(np.inf)).astype(np.float64)
    below = np.nextafter(rounded, np.float32(-np.inf)).astype(np.float64)
    return 0.5 * np.maximum(above - centre, centre - below)


def validate_heatwave_aggregates(
    frame: pd.DataFrame, config: HeatwaveFeatureConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate heat aggregates allowing only bounded binary32 representation.

    Zone preparation casts source features to float32. A fraction such as 3/5
    then becomes 0.6000000238418579; recomputing CDD from rounded temperatures
    can also differ from the separately rounded, originally computed aggregate.
    Fraction tolerance is half one float32 output bin. Cooling tolerance is the
    mean of the five temperature rounding radii plus its output radius: max()
    is 1-Lipschitz. A separate binary64 arithmetic bound covers recomputation.
    This does not relax the source PIT check or rewrite any feature values.
    """
    selected = _config(config)
    streak_columns = [f"{c.lower()}_heat_streak_fcst_days" for c in selected.countries]
    needed = [*temperature_aliases(selected), *streak_columns, HEAT_FRACTION_ALIAS]
    if selected.include_cooling_mean:
        needed.append(COOLING_MEAN_ALIAS)
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.has_duplicates or not set(needed).issubset(frame):
        raise HeatwaveFeatureError("Aggregate validation requires all raw temperature, streak and regional columns.")
    values = frame[needed].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all():
        raise HeatwaveFeatureError("Aggregate validation requires finite forecast features.")
    fraction = values[HEAT_FRACTION_ALIAS].to_numpy(float)
    streaks = values[streak_columns].to_numpy(float)
    if ((streaks < 0) | (streaks > selected.streak_clip_days) | (streaks != np.floor(streaks))).any():
        raise HeatwaveFeatureError("Aggregate validation requires bounded integer forecast streaks.")
    expected_fraction = np.mean(streaks >= selected.persistent_days, axis=1)
    fraction_tolerance = _float32_rounding_radius(fraction) + 8 * np.finfo(float).eps
    fraction_delta = np.abs(expected_fraction - fraction)
    if ((fraction < 0) | (fraction > 1)).any() or (fraction_delta > fraction_tolerance).any():
        raise HeatwaveFeatureError("Regional persistent-heat fraction differs from country streaks beyond float32 rounding.")
    audit = {"policy": "bounded_binary32_input_output_rounding_plus_binary64_arithmetic",
             "feature_values_modified": False, "fraction_max_abs_difference": float(fraction_delta.max()),
             "fraction_max_absolute_tolerance": float(fraction_tolerance.max())}
    if selected.include_cooling_mean:
        temperatures = values[list(temperature_aliases(selected))].to_numpy(float)
        cooling = values[COOLING_MEAN_ALIAS].to_numpy(float)
        expected_cooling = np.maximum(temperatures - selected.cooling_threshold_c, 0).mean(axis=1)
        magnitude = np.maximum.reduce((np.abs(temperatures).mean(axis=1), np.abs(cooling),
                                       np.full(len(cooling), max(abs(selected.cooling_threshold_c), 1.))))
        cooling_tolerance = (_float32_rounding_radius(temperatures).mean(axis=1)
                             + _float32_rounding_radius(cooling) + 8 * np.finfo(float).eps * magnitude)
        cooling_delta = np.abs(expected_cooling - cooling)
        if (cooling < 0).any() or (cooling_delta > cooling_tolerance).any():
            raise HeatwaveFeatureError("Regional cooling degrees differ from temperatures beyond float32 rounding.")
        audit.update(cooling_max_abs_difference=float(cooling_delta.max()),
                     cooling_max_absolute_tolerance=float(cooling_tolerance.max()))
    return audit


def _day(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise HeatwaveFeatureError("An explicit timezone-naive civil date is required.")
    return result


def _aware(values: Any, name: str) -> pd.DatetimeIndex:
    if any(pd.isna(value) or pd.Timestamp(value).tzinfo is None for value in values):
        raise HeatwaveFeatureError(f"{name}: explicit timezone-aware timestamps required.")
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def _select_daily(frame: pd.DataFrame, alias: str, timezone: str, end: pd.Timestamp) -> tuple[pd.Series, dict]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.has_duplicates:
        raise HeatwaveFeatureError(f"{alias}: nonempty DataFrame with unique columns required.")
    value_column = alias if alias in frame else "value"
    if not {*TIME_COLUMNS, value_column}.issubset(frame):
        raise HeatwaveFeatureError(f"{alias}: narrow or wide standard PIT columns required.")
    delivery = _aware(frame.value_time_utc, f"{alias}/delivery")
    snapshot = _aware(frame.snapshot_time_utc, f"{alias}/snapshot")
    revision = _aware(frame.revision_time_utc, f"{alias}/revision")
    if not delivery.equals(delivery.floor("h")):
        raise HeatwaveFeatureError(f"{alias}: physical HH:00 delivery hours required.")
    local_day = delivery.tz_convert(timezone).tz_localize(None).normalize()
    cutoff = (local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(timezone).tz_convert("UTC")
    raw = pd.DataFrame({"delivery": delivery, "snapshot": snapshot, "revision": revision,
                        "value": pd.to_numeric(frame[value_column], errors="coerce").to_numpy(float)})
    within = local_day <= end
    eligible = raw.loc[within & (snapshot <= cutoff) & (revision <= cutoff)]
    if eligible.empty:
        raise HeatwaveFeatureError(f"{alias}: no eligible pre-08 forecast rows.")
    keys = ["delivery", "snapshot", "revision"]
    tied = eligible.loc[eligible.duplicated(keys, keep=False)]
    if not tied.empty and tied.groupby(keys, dropna=False).value.nunique(dropna=False).gt(1).any():
        raise HeatwaveFeatureError(f"{alias}: contradictory values for one PIT identity.")
    chosen = eligible.sort_values(keys, kind="stable").drop_duplicates("delivery", keep="last")
    series = pd.Series(chosen.value.to_numpy(), index=pd.DatetimeIndex(chosen.delivery), name=alias)
    # A day represented only by post-cutoff vintages is a missing day, not
    # permission to silently shorten the supplied calibration prefix.
    first_day = local_day[within].min()
    expected = pd.date_range(first_day.tz_localize(timezone), (end + pd.Timedelta(days=1)).tz_localize(timezone),
                             freq="h", inclusive="left").tz_convert("UTC")
    if not series.index.equals(expected) or not np.isfinite(series.to_numpy(float)).all():
        raise HeatwaveFeatureError(f"{alias}: incomplete physical-hour PIT support; no filling is allowed.")
    if ((series < -60) | (series > 60)).any():
        raise HeatwaveFeatureError(f"{alias}: temperature outside plausible [-60,60] degC.")
    groups = series.groupby(series.index.tz_convert(timezone).tz_localize(None).normalize())
    if (groups.max() - groups.min() > 1e-9).any():
        raise HeatwaveFeatureError(f"{alias}: expected one national daily index broadcast unchanged; not Tmax/hourly weather.")
    daily = groups.first()
    daily.index = pd.DatetimeIndex(daily.index)
    return daily, {"alias": alias, "first_day": str(first_day.date()), "last_day": str(end.date()),
                   "selected_hours": len(series), "selected_days": len(daily),
                   "post_cutoff_rows_excluded": int((within & ((snapshot > cutoff) | (revision > cutoff))).sum()),
                   "fill_policy": "none", "selection": "latest eligible pre-civil-D-1-08 snapshot/revision"}


def build_heatwave_features(
    raw_sources: Mapping[str, pd.DataFrame] | pd.DataFrame, *, start_day: Any, end_day: Any,
    timezone: str = "Europe/Paris", config: HeatwaveFeatureConfig | Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build hourly-broadcast forecast signals using only preceding daily indices.

    ``start_day`` crops output, not the threshold's history. Supply the audited
    raw prefix before that day. Daily thresholds use [D-365,D), and streaks use
    each previous day's own threshold; subsequent forecasts cannot revise them.
    Output snapshot/revision identify the derived *query cutoff*, not a provider
    publication timestamp. Source provenance remains in the returned audit.
    """
    selected = _config(config)
    start, end = _day(start_day), _day(end_day)
    if start > end:
        raise HeatwaveFeatureError("start_day must not be after end_day.")
    if timezone != "Europe/Paris":
        raise HeatwaveFeatureError("The shared experiment uses the Europe/Paris civil 08:00 cutoff.")
    values, source_audits = {}, {}
    for country, alias in zip(selected.countries, temperature_aliases(selected)):
        source = (raw_sources if isinstance(raw_sources, pd.DataFrame)
                  else raw_sources.get(alias, raw_sources.get(country)))
        values[alias], source_audits[country] = _select_daily(source, alias, timezone, end)
    indices = [value.index for value in values.values()]
    if any(not index.equals(indices[0]) for index in indices[1:]) or indices[0][0] > start:
        raise HeatwaveFeatureError("All five countries require the same complete raw prefix through end_day.")
    daily = pd.DataFrame(values)
    records = []
    for country, alias in zip(selected.countries, temperature_aliases(selected)):
        source = daily[alias]
        threshold = source.rolling(f"{selected.lookback_days}D", closed="left").quantile(selected.threshold_quantile)
        count = source.rolling(f"{selected.lookback_days}D", closed="left").count().fillna(0).astype(int)
        threshold = threshold.where(count >= selected.minimum_history_days,
                                    float(selected.floor_thresholds_c[country])).clip(lower=float(selected.floor_thresholds_c[country]))
        excess = (source - threshold).clip(lower=0.0)
        streak_values, streak = [], 0
        for positive in excess.gt(0):
            streak = min(streak + 1, selected.streak_clip_days) if positive else 0
            streak_values.append(float(streak))
        daily[f"{country.lower()}_heat_excess_fcst_c"] = excess
        daily[f"{country.lower()}_heat_streak_fcst_days"] = streak_values
        for d in daily.index[daily.index >= start]:
            records.append({"day": str(d.date()), "country": country, "temperature_fcst_c": float(source.loc[d]),
                "prior_days": int(count.loc[d]), "threshold_fcst_c": float(threshold.loc[d]),
                "threshold_source": "preceding_365_days_quantile_and_fixed_floor" if count.loc[d] >= selected.minimum_history_days else "fixed_floor_warmup",
                "excess_fcst_c": float(excess.loc[d]),
                "streak_fcst_days": int(daily.loc[d, f"{country.lower()}_heat_streak_fcst_days"])})
    streaks = [f"{c.lower()}_heat_streak_fcst_days" for c in selected.countries]
    daily[HEAT_FRACTION_ALIAS] = daily[streaks].ge(selected.persistent_days).mean(axis=1)
    if selected.include_cooling_mean:
        daily[COOLING_MEAN_ALIAS] = daily[list(temperature_aliases(selected))].sub(selected.cooling_threshold_c).clip(lower=0).mean(axis=1)
    hours = pd.date_range(start.tz_localize(timezone), (end + pd.Timedelta(days=1)).tz_localize(timezone),
                          freq="h", inclusive="left").tz_convert("UTC")
    local_days = hours.tz_convert(timezone).tz_localize(None).normalize()
    cutoffs = (local_days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(timezone).tz_convert("UTC")
    output = daily.loc[local_days, list(heatwave_feature_aliases(selected))].reset_index(drop=True)
    output.insert(0, "revision_time_utc", cutoffs)
    output.insert(0, "value_time_utc", hours)
    output.insert(0, "snapshot_time_utc", cutoffs)
    audit = {"schema_version": 1, "engine": "heatwave_features_v1", "feature_config": selected.to_dict(),
             "features_sha256": heatwave_features_sha256(), "feature_aliases": list(heatwave_feature_aliases(selected)),
             "feature_metadata": feature_metadata(selected), "source_audits": source_audits,
             "raw_prefix_start_day": str(daily.index[0].date()), "start_day": str(start.date()), "end_day": str(end.date()),
             "hours": len(output), "days": (end - start).days + 1, "daily_heat_audit": records,
             "threshold_history": "calendar [D-365,D), quantile linear interpolation; current/future day excluded",
             "streak_rule": "consecutive prior/current forecast excess >0, clipped; no future persistence confirmation",
             "temperature_semantics": "daily national forecast index, not hourly temperature, Tmax, Tmin or observed heatwave",
             "metadata_semantics": "derived query cutoff D-1 civil 08:00; not provider publication timestamp",
             "source_availability_rule": "each source day forecast vintage available no later than that day's D-1 08:00",
             "future_data_used_for_past_features": False, "observed_temperature_used": False,
             "production_pit_evidence": False, "production_modified": False}
    audit["recipe_sha256"] = hashlib.sha256(json.dumps({"code": audit["features_sha256"],
        "config": selected.to_dict()}, sort_keys=True).encode()).hexdigest()
    return output, audit
