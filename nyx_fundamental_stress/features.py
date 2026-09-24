"""Build causal local/peer stress features without reading electricity prices.

No forecasting, realised or benchmark electricity price, quantile, past price,
or baseline-derived feature is inspected. Clean gas costs are explicitly
permitted fundamental fuel/carbon proxies, not electricity-price forecasts.
Selected supply is an incomplete mixed-Pmax/generation proxy, NOT a reserve
margin, a complete merit order, or a physically feasible import capacity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PREFIX = "feature_fundamental_"
ZONES = ("FR", "DE", "BE", "NL")
SUPPLY_COMPONENTS = {
    "FR": ("feature_fr_gas_available_gw", "feature_fr_nuclear_generation_gw"),
    "DE": ("feature_de_gas_available_gw", "feature_de_coal_available_gw", "feature_de_lignite_available_gw"),
    "BE": ("feature_be_gas_available_gw", "feature_be_nuclear_available_gw"),
    "NL": ("feature_nl_gas_available_gw", "feature_nl_coal_available_gw", "feature_nl_nuclear_available_gw"),
}
CALENDAR_NAMES = [PREFIX+name for name in ("hour_sin", "hour_cos", "weekday", "month")]
DENOMINATOR_FLOOR_GW = 1.


def _utc(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any() or any(pd.Timestamp(value).tzinfo is None for value in values):
        raise ValueError(f"{name}: complete timezone-aware timestamps required.")
    return pd.to_datetime(values, utc=True, errors="raise")


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[name], errors="raise").astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError(f"{name}: infinite physical input is forbidden.")
    return values


def _agree(cached: pd.Series, derived: pd.Series, name: str) -> None:
    complete = cached.notna() & derived.notna()
    if not np.allclose(cached.loc[complete], derived.loc[complete], rtol=0, atol=1e-8):
        raise ValueError(f"{name}: cached physical feature disagrees with its source components.")


def _profile_ramp(values: pd.Series, zones: pd.Series, timestamps: pd.Series,
                  origins: pd.Series, hours: int) -> pd.Series:
    """A profile difference at exact h-hours, never a row shift across vintages.

    These are differences between forecasts for two delivery hours, both in
    the same already-issued day-ahead profile. They are not realised ramps.
    First-day hours, missing timestamps or different origins remain unknown.
    """
    key = pd.MultiIndex.from_arrays([zones, origins, timestamps], names=["zone", "origin", "delivery"])
    lag_key = pd.MultiIndex.from_arrays([zones, origins, timestamps-pd.Timedelta(hours=hours)],
                                       names=key.names)
    previous = pd.Series(values.to_numpy(), index=key).reindex(lag_key).to_numpy()
    return pd.Series((values.to_numpy()-previous)/hours, index=values.index)


def make_fundamental_features(panel: pd.DataFrame, variant: str = "fundamental") -> tuple[pd.DataFrame, list[str], list[str], dict]:
    """Return unchanged source columns plus physical/calendar features and audit.

    Both variants construct exactly the same added columns. Only their model
    allowlists/required subsets differ: calendar never requires a physical
    source to be populated. Every new feature is an explicit physical mapping
    or deterministic function of the current forecast origin/profile.
    """
    if variant not in {"fundamental", "calendar"}:
        raise ValueError("Variant must be fundamental or calendar.")
    if (not isinstance(panel, pd.DataFrame) or panel.empty or panel.columns.has_duplicates
            or not {"zone", "timestamp_utc", "forecast_origin_utc"}.issubset(panel)):
        raise ValueError("A nonempty panel with unique columns and country/hour/origin identities is required.")
    if any(str(name).startswith(PREFIX) for name in panel.columns):
        raise ValueError("Reserved feature_fundamental_ namespace already exists; original columns cannot be overwritten.")
    original_columns = list(panel.columns)
    frame = panel.copy(deep=True).reset_index(drop=True)
    if not frame.zone.map(lambda value: isinstance(value, str) and value in ZONES).all():
        raise ValueError("Only exact FR, DE, BE and NL country codes are supported.")
    timestamps, origins = _utc(frame.timestamp_utc, "timestamp_utc"), _utc(frame.forecast_origin_utc, "forecast_origin_utc")
    if (not timestamps.eq(timestamps.dt.floor("h")).all()
            or pd.DataFrame({"zone": frame.zone, "timestamp": timestamps}).duplicated().any()):
        raise ValueError("Unique physical hourly country/delivery keys are required.")
    local = timestamps.dt.tz_convert("Europe/Paris")
    civil = local.dt.tz_localize(None).dt.normalize()
    expected_origin = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
    if not origins.eq(expected_origin).all():
        raise ValueError("Only D-1 08:00 Europe/Paris civil forecast origins are accepted.")
    used: set[str] = set()

    def read(name):
        used.add(name)
        return _numeric(frame, name)

    def countries(suffix):
        return pd.DataFrame({zone: read(f"feature_{zone.lower()}_{suffix}") for zone in ZONES})

    def own(values):
        selected = pd.Series(np.nan, index=frame.index, dtype=float)
        for zone in ZONES:
            where = frame.zone.eq(zone)
            selected.loc[where] = values.loc[where, zone]
        return selected

    def peer_sum(values):
        selected = pd.Series(np.nan, index=frame.index, dtype=float)
        for zone in ZONES:
            where = frame.zone.eq(zone)
            selected.loc[where] = values.loc[where, [z for z in ZONES if z != zone]].sum(axis=1, min_count=3)
        return selected

    def cached(name, derived, *, complete_components=False):
        if name not in frame:
            return derived
        previous = read(name)
        _agree(previous, derived, name)
        # A present saved aggregate cannot hide a missing constituent source.
        return derived if complete_components else previous.where(previous.notna(), derived)

    residual, gas = countries("residual_load_gw"), countries("gas_available_gw")
    supply = pd.DataFrame(index=frame.index)
    for zone, components in SUPPLY_COMPONENTS.items():
        values = pd.DataFrame({name: read(name) for name in components})
        if values.lt(0).any().any():
            raise ValueError(f"{zone}: available-Pmax/nuclear-generation supply inputs cannot be negative.")
        supply[zone] = values.sum(axis=1, min_count=len(components))
    local_r = cached("feature_local_residual_load_gw", own(residual), complete_components=True)
    local_g = cached("feature_local_gas_available_gw", own(gas), complete_components=True)
    local_s = cached("feature_local_selected_supply_proxy_gw", own(supply), complete_components=True)
    local_margin = cached("feature_local_selected_supply_minus_residual_proxy_gw", local_s-local_r, complete_components=True)
    peer_r, peer_s = peer_sum(residual), peer_sum(supply)
    local_denominator = local_s.clip(lower=DENOMINATOR_FLOOR_GW)
    peer_denominator = peer_s.clip(lower=DENOMINATOR_FLOOR_GW)
    pressures = residual/supply.clip(lower=DENOMINATOR_FLOOR_GW)
    peer_stressed, peer_max = pd.Series(np.nan, index=frame.index), pd.Series(np.nan, index=frame.index)
    for zone in ZONES:
        where = frame.zone.eq(zone)
        others = pressures.loc[where, [z for z in ZONES if z != zone]]
        peer_stressed.loc[where] = others.gt(1.).sum(axis=1).where(others.notna().all(axis=1))
        peer_max.loc[where] = others.max(axis=1, skipna=False)
    features = {
        "local_residual_load_gw": local_r, "local_gas_available_gw": local_g,
        "local_selected_supply_proxy_gw": local_s,
        "local_selected_supply_minus_residual_proxy_gw": local_margin,
        "local_selected_non_gas_supply_proxy_gw": local_s-local_g,
        "local_pressure": local_r/local_denominator,
        "local_margin_ratio_proxy": local_margin/local_denominator,
        "local_gas_share_selected_supply_proxy": local_g/local_denominator,
        "peer_residual_load_gw": peer_r, "peer_selected_supply_proxy_gw": peer_s,
        "peer_selected_supply_minus_residual_proxy_gw": peer_s-peer_r,
        "peer_pressure": peer_r/peer_denominator,
        "local_minus_peer_pressure": local_r/local_denominator-peer_r/peer_denominator,
        "peer_stressed_count": peer_stressed, "peer_max_pressure": peer_max,
        "local_positive_residual_share_regional_proxy": local_r.clip(lower=0)/residual.clip(lower=0).sum(axis=1, min_count=4).clip(lower=DENOMINATOR_FLOOR_GW),
    }
    profiles = {"residual": (local_r, peer_r)}
    for driver in ("wind", "solar"):
        profile = countries(f"{driver}_generation_gw")
        local_value = cached(f"feature_local_{driver}_generation_gw", own(profile))
        peer_value = peer_sum(profile)
        features[f"local_{driver}_generation_gw"] = local_value
        features[f"peer_{driver}_generation_gw"] = peer_value
        features[f"local_{driver}_to_selected_supply_proxy"] = local_value/local_denominator
        profiles[driver] = (local_value, peer_value)
    for driver, (local_profile, peer_profile) in profiles.items():
        for hours in (1, 3):
            local_ramp = _profile_ramp(local_profile, frame.zone, timestamps, origins, hours)
            peer_ramp = _profile_ramp(peer_profile, frame.zone, timestamps, origins, hours)
            features[f"local_{driver}_ramp_{hours}h_gw_per_hour"] = local_ramp
            features[f"local_{driver}_ramp_{hours}h_normalized_proxy_per_hour"] = local_ramp/local_denominator
            features[f"peer_{driver}_ramp_{hours}h_gw_per_hour"] = peer_ramp
            features[f"peer_{driver}_ramp_{hours}h_normalized_proxy_per_hour"] = peer_ramp/peer_denominator
    temperatures = countries("temperature_c")
    features["local_temperature_c"] = cached("feature_local_temperature_c", own(temperatures))
    features["peer_temperature_mean_c"] = peer_sum(temperatures)/3
    ccgt, ocgt = read("feature_clean_gas_cost_ccgt_proxy_eur_mwh"), read("feature_clean_gas_cost_ocgt_proxy_eur_mwh")
    features["clean_gas_cost_ccgt_proxy_eur_mwh"] = ccgt
    features["clean_gas_cost_ocgt_proxy_eur_mwh"] = ocgt
    features["gas_merit_slope_proxy_eur_mwh"] = cached("feature_gas_merit_slope_proxy_eur_mwh", ocgt-ccgt, complete_components=True)
    features.update(hour_sin=np.sin(local.dt.hour*np.pi/12), hour_cos=np.cos(local.dt.hour*np.pi/12),
                    weekday=local.dt.dayofweek.astype(float), month=local.dt.month.astype(float))
    physical = pd.DataFrame({PREFIX+name: values for name, values in features.items()}, index=frame.index).astype(float)
    if np.isinf(physical.to_numpy()).any():
        raise ValueError("Physical feature construction overflowed finite arithmetic.")
    # Calendar is deterministic and complete; pointless constant calendar
    # missing flags are not introduced into either comparator's feature list.
    optional_flags = physical.drop(columns=CALENDAR_NAMES).isna().astype(float).rename(columns=lambda name: name+"__missing")
    added = pd.concat([physical, optional_flags], axis=1)
    physical_required = [PREFIX+name for name in ("local_residual_load_gw", "local_selected_supply_proxy_gw",
                                                 "peer_residual_load_gw", "peer_selected_supply_proxy_gw")]
    names = list(added) if variant == "fundamental" else list(CALENDAR_NAMES)
    required = physical_required if variant == "fundamental" else list(CALENDAR_NAMES[:3])
    eligible = added[required].notna().all(axis=1)
    added.index = panel.index.copy()
    augmented = pd.concat([panel.copy(deep=True), added], axis=1)
    pd.testing.assert_frame_equal(augmented[original_columns], panel, check_exact=True)
    audit = {"schema_version": 1, "representation": "price_blind_local_peer_fundamentals_v1", "variant": variant,
             "rows": len(panel), "feature_columns": names, "required_feature_columns": required,
             "all_constructed_feature_columns": list(added), "physical_required_feature_columns": physical_required,
             "calendar_feature_columns": list(CALENDAR_NAMES), "original_columns_unchanged": True,
             "model_fit_performed": False, "source_reads_performed": False,
             "electricity_price_forecast_quantile_actual_storm_or_lag_used": False,
             "input_source_columns_read": sorted(used), "strict_cutoff": "D-1 08:00 Europe/Paris civil, DST-aware",
             "selected_supply_components": {zone: list(parts) for zone, parts in SUPPLY_COMPONENTS.items()},
             "supply_semantics": "FR available gas Pmax + nuclear generation forecast; DE available gas/coal/lignite Pmax; BE available gas/nuclear Pmax; NL available gas/coal/nuclear Pmax. Incomplete mixed-generation/Pmax fleet proxy, not full reserve margin or feasible imports.",
             "missing_component_policy": "NaN propagates through local/peer sums; no partial sum or nuclear zero imputation",
             "ratio_denominator_floor_gw": DENOMINATOR_FLOOR_GW,
             "pressure_formula": "residual GW / max(selected-supply proxy GW, 1)",
             "peer_pressure_formula": "sum residual of the other three countries / max(sum their selected supply, 1)",
             "peer_stressed_count_rule": "number of the other three individual pressure proxies strictly >1; NaN if any missing",
             "ramps": "exact 1h/3h physical timestamp difference / hours; same country, same origin and therefore same civil delivery day; missing lag remains NaN",
             "gas_costs": "saved TTF/EUA-based clean-gas engineering proxies permitted; no CGC-minus-NYX or executable electricity price used",
             "network_inputs_used": False, "forecast_revision_features_used": False,
             "temperature_policy": "saved optional forecasts only; gaps/multivintage gaps are not filled or reconstructed",
             "calendar_eligibility_independent_of_physical_gaps": True,
             "eligible_rows": int(eligible.sum()), "missing_required_rows": int((~eligible).sum()),
             "coverage": {name: {"finite_rows": int(added[name].notna().sum()), "missing_rows": int(added[name].isna().sum())} for name in physical.columns}}
    return augmented, names, required, audit
