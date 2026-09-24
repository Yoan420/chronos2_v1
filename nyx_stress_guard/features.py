"""Price-blind, same-origin profile features and CORE-only physical ranks.

The daily profile is a set of forecasts already available at D-1 08:00, not
the realised next day. Regional synchrony is a descriptor, never a feasible
import calculation, and wind/solar are not subtracted from residual load again.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nyx_fundamental_stress.features import (
    PREFIX as BASE_PREFIX, SUPPLY_COMPONENTS, ZONES, _profile_ramp,
    make_fundamental_features,
)


PREFIX = "feature_fundamental_stress_"
MIN_REFERENCE_OBSERVATIONS = 30
MIN_REFERENCE_DAYS = 20
REFERENCE_SOURCES = {
    PREFIX+"core_rank_local_pressure": BASE_PREFIX+"local_pressure",
    PREFIX+"core_rank_peer_pressure": BASE_PREFIX+"peer_pressure",
    PREFIX+"core_rank_selected_supply": BASE_PREFIX+"local_selected_supply_proxy_gw",
    PREFIX+"core_rank_local_residual_ramp_3h": BASE_PREFIX+"local_residual_ramp_3h_gw_per_hour",
    PREFIX+"core_rank_regional_residual_ramp_3h": PREFIX+"regional_residual_ramp_3h_gw_per_hour",
    PREFIX+"core_rank_residual_rising_count_3h": PREFIX+"residual_rising_count_3h",
}


def _identity(frame: pd.DataFrame):
    if (not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.has_duplicates
            or not {"zone", "timestamp_utc", "forecast_origin_utc"}.issubset(frame)):
        raise ValueError("Nonempty unique-column frame with country/hour/origin identities required.")
    if not frame.zone.map(lambda z: isinstance(z, str) and z in ZONES).all():
        raise ValueError("Only exact FR, DE, BE, NL countries are supported.")
    times = []
    for name in ("timestamp_utc", "forecast_origin_utc"):
        if frame[name].isna().any() or any(pd.Timestamp(t).tzinfo is None for t in frame[name]):
            raise ValueError(f"{name}: complete timezone-aware timestamps required.")
        times.append(pd.to_datetime(frame[name], utc=True, errors="raise").reset_index(drop=True))
    timestamps, origins = times
    if (not timestamps.eq(timestamps.dt.floor("h")).all()
            or pd.DataFrame({"zone": frame.zone.to_numpy(), "timestamp": timestamps}).duplicated().any()):
        raise ValueError("Unique physical hourly country/delivery identities required.")
    civil = timestamps.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    expected = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
    if not origins.eq(expected).all():
        raise ValueError("D-1 08:00 Europe/Paris civil origin required.")
    return timestamps, origins, civil


def _read(frame: pd.DataFrame, name: str) -> pd.Series:
    values = (pd.to_numeric(frame[name], errors="raise").astype(float).reset_index(drop=True)
              if name in frame else pd.Series(np.nan, index=range(len(frame)), dtype=float))
    if np.isinf(values.to_numpy()).any():
        raise ValueError(f"{name}: infinite physical input forbidden.")
    return values


def make_stress_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    """Extend the frozen fundamental bank, keeping every original field intact.

    Reference-rank columns are placeholders until fit_reference(CORE) followed
    by apply_reference. They are never computed from the full replay panel.
    """
    augmented, original_features, original_required, inherited = make_fundamental_features(panel)
    frame = augmented.reset_index(drop=True)
    timestamps, origins, civil = _identity(frame)
    zones = frame.zone.reset_index(drop=True)
    raw_names = [f"feature_{z.lower()}_{suffix}" for suffix in
                 ("residual_load_gw", "wind_generation_gw", "solar_generation_gw") for z in ZONES]
    raw_names += list(dict.fromkeys(c for components in SUPPLY_COMPONENTS.values() for c in components))
    raw = pd.DataFrame({name: _read(frame, name) for name in raw_names})
    # A regional physical source is identical across country views at one
    # origin/hour. Missing copies also disagree: a country cannot silently see
    # a different version of an otherwise shared regional forecast.
    keyed = pd.concat([pd.DataFrame({"timestamp": timestamps, "origin": origins}), raw], axis=1)
    if keyed.groupby(["timestamp", "origin"], sort=False)[raw_names].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Conflicting shared physical source across country views.")

    def countries(driver):
        suffix = "residual_load_gw" if driver == "residual" else f"{driver}_generation_gw"
        return pd.DataFrame({z: raw[f"feature_{z.lower()}_{suffix}"] for z in ZONES})

    profiles = {driver: countries(driver) for driver in ("residual", "wind", "solar")}
    supply = (_read(frame, BASE_PREFIX+"local_selected_supply_proxy_gw")
              + _read(frame, BASE_PREFIX+"peer_selected_supply_proxy_gw"))
    denominator = supply.clip(lower=1.)
    new: dict[str, pd.Series] = {}
    for hours in (1, 3):
        ramps = {driver: pd.DataFrame({z: _profile_ramp(values[z], zones, timestamps, origins, hours)
                                      for z in ZONES}) for driver, values in profiles.items()}
        for driver in profiles:
            value = ramps[driver].sum(axis=1, min_count=4)
            new[f"regional_{driver}_ramp_{hours}h_gw_per_hour"] = value
            new[f"regional_{driver}_ramp_{hours}h_normalized_per_hour"] = value/denominator
        rising = ramps["residual"].gt(0)
        wind_falling, solar_falling = ramps["wind"].lt(0), ramps["solar"].lt(0)
        for name, mask, valid in (
            ("residual_rising_count", rising, ramps["residual"].notna().all(axis=1)),
            ("wind_falling_count", wind_falling, ramps["wind"].notna().all(axis=1)),
            ("solar_falling_count", solar_falling, ramps["solar"].notna().all(axis=1)),
            ("residual_rise_and_renewable_fall_count", rising & (ramps["wind"]+ramps["solar"]).lt(0),
             pd.concat(list(ramps.values()), axis=1).notna().all(axis=1)),
        ):
            new[f"{name}_{hours}h"] = mask.sum(axis=1).astype(float).where(valid)
        # Coincident directions are supplied separately from magnitudes. They
        # are not a causal attribution of the residual ramp to renewables.
        for scope in ("local", "peer"):
            r = _read(frame, BASE_PREFIX+f"{scope}_residual_ramp_{hours}h_gw_per_hour")
            w = _read(frame, BASE_PREFIX+f"{scope}_wind_ramp_{hours}h_gw_per_hour")
            s = _read(frame, BASE_PREFIX+f"{scope}_solar_ramp_{hours}h_gw_per_hour")
            den = _read(frame, BASE_PREFIX+f"{scope}_selected_supply_proxy_gw").clip(lower=1.)
            new[f"{scope}_renewable_fall_ramp_{hours}h_normalized_per_hour"] = (-(w+s)).clip(lower=0)/den
            new[f"{scope}_solar_fall_with_residual_rise_{hours}h"] = (
                (r.gt(0) & s.lt(0)).astype(float).where(r.notna() & s.notna()))

    # Profiles must be complete physical civil days: 23/24/25 hours. Missing
    # physical inputs invalidate whole-day shape descriptors, never get filled.
    full = pd.Series(False, index=frame.index)
    profile_names = [f"{scope}_{driver}" for scope in ("local", "peer") for driver in ("residual", "wind", "solar")]
    profile_values = {}
    for name in profile_names:
        scope, driver = name.split("_", 1)
        suffix = "residual_load_gw" if driver == "residual" else f"{driver}_generation_gw"
        profile_values[name] = _read(frame, BASE_PREFIX+scope+"_"+suffix)
        for descriptor in ("day_rank", "minus_day_median_gw", "below_day_peak_gw"):
            new[name+"_"+descriptor] = pd.Series(np.nan, index=frame.index, dtype=float)
    daily_groups = pd.DataFrame({"zone": zones, "origin": origins, "civil": civil}).groupby(["zone", "origin"], sort=False).groups
    for _, positions in daily_groups.items():
        day = civil.loc[positions].iloc[0]
        expected = pd.date_range(day.tz_localize("Europe/Paris"),
                                 (day+pd.Timedelta(days=1)).tz_localize("Europe/Paris"),
                                 inclusive="left", freq="h").tz_convert("UTC")
        actual = pd.DatetimeIndex(timestamps.loc[positions]).sort_values()
        complete = actual.equals(expected) and raw.loc[positions].notna().all().all()
        if not complete:
            continue
        full.loc[positions] = True
        for name in profile_names:
            values = profile_values[name].loc[positions]
            # Mid-CDF rank, consistent with the CORE reference: a constant
            # profile has rank .5; equal values receive the same rank.
            new[name+"_day_rank"].loc[positions] = (values.rank(method="average")-.5)/len(values)
            new[name+"_minus_day_median_gw"].loc[positions] = values-values.median()
            new[name+"_below_day_peak_gw"].loc[positions] = values.max()-values
    new["complete_physical_profile"] = pd.Series(1., index=frame.index).where(full)
    added = pd.DataFrame({PREFIX+name: value for name, value in new.items()}, index=frame.index)
    for name in REFERENCE_SOURCES:
        added[name] = np.nan
    flags = added.isna().astype(float).rename(columns=lambda c: c+"__missing")
    added = pd.concat([added, flags], axis=1)
    if np.isinf(added.to_numpy(float)).any():
        raise ValueError("Nonfinite derived physical feature.")
    added.index = panel.index.copy()
    result = pd.concat([augmented, added], axis=1)
    names = [*original_features, *added.columns]
    required = [*original_required, PREFIX+"complete_physical_profile"]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature identities.")
    pd.testing.assert_frame_equal(result[panel.columns], panel, check_exact=True)
    audit = {"schema_version": 1, "representation": "same_origin_synchronous_physical_profiles_v1",
             "inherited_fundamental_audit": inherited, "feature_columns": names,
             "required_feature_columns": required, "new_feature_columns": list(added),
             "reference_rank_columns": list(REFERENCE_SOURCES), "reference_rank_sources": dict(REFERENCE_SOURCES),
             "input_source_columns_read": sorted(set(inherited["input_source_columns_read"]) | set(raw_names)),
             "original_columns_unchanged": True, "source_reads_performed": False,
             "electricity_price_forecast_quantile_actual_storm_or_lag_used": False,
             "network_inputs_used": False, "forecast_revision_features_used": False,
             "reference_fit_performed": False, "reference_rank_placeholders_are_not_required": True,
             "profile_shape_scope": "Only the complete physical 23/24/25-hour delivery-day profile at one D-1 08:00 civil origin, never realised next-day values.",
             "complete_profile_rows": int(full.sum()), "incomplete_profile_rows": int((~full).sum()),
             "missing_profile_policy": "Missing any regional RL/wind/solar/selected-supply constituent or physical delivery hour leaves daily shape NaN and forces abstention.",
             "renewable_accounting": "Separate descriptors only; never subtracted from residual load again.",
             "synchrony_semantics": "Sign coincidence across four countries, not power flows or causal attribution.",
             "regional_ratio_denominator_floor_gw": 1., "physical_gate_applied": False,
             "coverage": {c: {"finite_rows": int(added[c].notna().sum()), "missing_rows": int(added[c].isna().sum())} for c in added}}
    return result, names, required, audit


def fit_reference(core: pd.DataFrame, zones, *, cutoff) -> dict:
    """Fit six price-blind empirical distributions on CORE, never CAL/CURRENT.

    Country/hour references require 30 finite observations and 20 distinct
    days. Otherwise the country-pooled distribution uses the same minimum;
    absent support stays unavailable. Hourly observations are not treated as
    independent evidence for confidence intervals (none are calculated).
    """
    timestamps, origins, civil = _identity(core)
    allowed = list(zones)
    if not allowed or len(set(allowed)) != len(allowed) or set(allowed)-set(ZONES) or not core.zone.isin(allowed).all():
        raise ValueError("Unique supported reference countries required.")
    boundary = pd.Timestamp(cutoff)
    if pd.isna(boundary) or boundary.tzinfo is None:
        raise ValueError("Timezone-aware CORE cutoff required.")
    boundary = boundary.tz_convert("UTC")
    cutoff_local = boundary.tz_convert("Europe/Paris")
    if cutoff_local.hour != 8 or any((cutoff_local.minute, cutoff_local.second, cutoff_local.microsecond, cutoff_local.nanosecond)):
        raise ValueError("Reference cutoff must be civil 08:00.")
    if origins.ge(boundary).any() or timestamps.ge(boundary).any():
        raise ValueError("CORE reference contains data at or after its causal cutoff.")
    if not set(REFERENCE_SOURCES.values()).issubset(core):
        raise ValueError("Missing declared physical reference columns.")
    local_hours = timestamps.dt.tz_convert("Europe/Paris").dt.hour
    countries = core.zone.reset_index(drop=True)
    values = {name: _read(core, name) for name in REFERENCE_SOURCES.values()}
    references, counts = {}, {}

    def sample(series, where):
        where = where & series.notna()
        n, days = int(where.sum()), int(civil.loc[where].nunique())
        ok = n >= MIN_REFERENCE_OBSERVATIONS and days >= MIN_REFERENCE_DAYS
        return {"values": np.sort(series.loc[where].to_numpy(float)) if ok else np.array([], dtype=float),
                "observations": n, "days": days, "available": bool(ok)}

    for name, source in REFERENCE_SOURCES.items():
        references[name], counts[name] = {}, {}
        for zone in allowed:
            own = countries.eq(zone)
            pooled = sample(values[source], own)
            hourly = {str(hour): sample(values[source], own & local_hours.eq(hour)) for hour in range(24)}
            references[name][zone] = {"pooled": pooled, "hourly": hourly}
            counts[name][zone] = {"pooled": {k: v for k, v in pooled.items() if k != "values"},
                                  "available_hour_groups": [int(h) for h, s in hourly.items() if s["available"]]}
    return {"schema_version": 1, "kind": "core_country_hour_mid_ecdf_v1", "zones": allowed,
            "sources": dict(REFERENCE_SOURCES), "references": references,
            "audit": {"cutoff_utc": boundary.isoformat(), "core_rows": len(core),
                      "core_first_day": str(civil.min().date()), "core_last_day": str(civil.max().date()),
                      "max_forecast_origin_utc": origins.max().isoformat(),
                      "max_delivery_utc": timestamps.max().isoformat(), "fit_scope": "caller-supplied CORE only; no calibration/current rows",
                      "min_observations": MIN_REFERENCE_OBSERVATIONS, "min_days": MIN_REFERENCE_DAYS,
                      "selection": "country-hour if supported, else country-pooled if supported, else NaN",
                      "rank_formula": "(count(reference < value) + .5 * count(reference == value)) / n",
                      "support": counts, "labels_or_electricity_prices_read": False}}


def apply_reference(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Apply one frozen CORE reference to CORE, chronological CAL or inference.

    Only the six reserved rank columns and their missing flags can change.
    The policy owns the split and rejects future fitted states at inference.
    """
    timestamps, _, _ = _identity(frame)
    if (state.get("kind") != "core_country_hour_mid_ecdf_v1" or state.get("sources") != REFERENCE_SOURCES
            or not frame.zone.isin(state.get("zones", [])).all()):
        raise ValueError("Known country identities and explicit frozen physical reference required.")
    reserved = list(REFERENCE_SOURCES) + [name+"__missing" for name in REFERENCE_SOURCES]
    if not set(reserved).issubset(frame) or not set(REFERENCE_SOURCES.values()).issubset(frame):
        raise ValueError("Construct the declared stress features before applying their reference.")
    output = frame.copy(deep=True)
    countries = frame.zone.reset_index(drop=True)
    hours = timestamps.dt.tz_convert("Europe/Paris").dt.hour
    for name, source in REFERENCE_SOURCES.items():
        current = _read(frame, source)
        ranks = np.full(len(frame), np.nan, dtype=float)
        for zone in state["zones"]:
            byzone = state["references"][name][zone]
            for hour in range(24):
                positions = np.flatnonzero((countries.eq(zone) & hours.eq(hour) & current.notna()).to_numpy())
                if not len(positions):
                    continue
                sample = byzone["hourly"][str(hour)]
                if not sample["available"]:
                    sample = byzone["pooled"]
                if not sample["available"]:
                    continue
                support = np.asarray(sample["values"], dtype=float)
                if (not len(support) or not np.isfinite(support).all() or np.any(np.diff(support) < 0)
                        or len(support) != sample["observations"] or sample["days"] < MIN_REFERENCE_DAYS
                        or len(support) < MIN_REFERENCE_OBSERVATIONS):
                    raise ValueError("Invalid frozen physical reference support.")
                points = current.iloc[positions].to_numpy(float)
                lo, hi = np.searchsorted(support, points, side="left"), np.searchsorted(support, points, side="right")
                ranks[positions] = (lo+hi)/(2.*len(support))
        output[name] = ranks
        output[name+"__missing"] = np.isnan(ranks).astype(float)
    unchanged = [name for name in frame if name not in reserved]
    pd.testing.assert_frame_equal(output[unchanged], frame[unchanged], check_exact=True)
    return output


__all__ = ["PREFIX", "REFERENCE_SOURCES", "make_stress_features", "fit_reference", "apply_reference"]
