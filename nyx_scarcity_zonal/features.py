"""Compact physical-context representation of an immutable CWE forecast panel.

Selected supply deliberately mixes available thermal Pmax and FR nuclear
generation forecasts. It is an incomplete fleet proxy, not a reserve margin or
a feasible network/import balance. Only saved input forecasts are used; neither
observed electricity prices nor Storm can enter the returned feature allowlist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ZONES = ("FR", "DE", "BE", "NL")
PREFIX = "feature_zonal_"
SUPPLY_COMPONENTS = {
    "FR": ("feature_fr_gas_available_gw", "feature_fr_nuclear_generation_gw"),
    "DE": ("feature_de_gas_available_gw", "feature_de_coal_available_gw", "feature_de_lignite_available_gw"),
    "BE": ("feature_be_gas_available_gw", "feature_be_nuclear_available_gw"),
    "NL": ("feature_nl_gas_available_gw", "feature_nl_coal_available_gw", "feature_nl_nuclear_available_gw"),
}
DENOMINATOR_FLOOR_GW = 1.


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[name], errors="raise").astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError(f"{name}: infinite physical input is not supported.")
    return values


def _utc(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any() or any(pd.Timestamp(v).tzinfo is None for v in values):
        raise ValueError(f"{name} requires complete explicitly timezone-aware timestamps.")
    return pd.to_datetime(values, utc=True, errors="raise")


def _agree(existing: pd.Series, derived: pd.Series, name: str) -> None:
    common = existing.notna() & derived.notna()
    if not np.allclose(existing.loc[common], derived.loc[common], rtol=0, atol=1e-8):
        raise ValueError(f"{name}: cached local feature disagrees with its physical source/arithmetic.")


def make_zonal_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    """Add only explicit new columns; preserve every original value and index.

    Every ratio uses a fixed 1 GW denominator floor, never fitted statistics.
    Peer physical proxies exclude the target country. Peer NYX prices require
    three distinct other-country rows with exactly the same UTC origin/hour.
    Missing source components remain unknown: sums never invent zero nuclear.
    """
    identities = {"zone", "timestamp_utc", "forecast_origin_utc", "forecast"}
    if (not isinstance(panel, pd.DataFrame) or panel.empty or panel.columns.has_duplicates
            or not identities.issubset(panel)):
        raise ValueError("A nonempty forecast panel with unique columns and zone/time/origin/forecast is required.")
    if any(str(name).startswith(PREFIX) for name in panel.columns):
        raise ValueError("The reserved feature_zonal_ namespace already exists; original columns cannot be overwritten.")
    original_columns = list(panel.columns)
    frame = panel.copy(deep=True).reset_index(drop=True)
    if not frame.zone.map(lambda x: isinstance(x, str) and x in ZONES).all():
        raise ValueError("Only exact FR, DE, BE and NL country identifiers are supported.")
    timestamps = _utc(frame.timestamp_utc, "timestamp_utc")
    origins = _utc(frame.forecast_origin_utc, "forecast_origin_utc")
    if not timestamps.eq(timestamps.dt.floor("h")).all() or pd.DataFrame({"zone": frame.zone, "timestamp": timestamps}).duplicated().any():
        raise ValueError("Each country/physical delivery hour must be unique and hourly.")
    civil = timestamps.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    expected_origin = (civil - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
    if not origins.eq(expected_origin).all():
        raise ValueError("Every frozen forecast must use its D-1 08:00 civil origin.")
    used: set[str] = set()

    def read(name):
        used.add(name)
        return _numeric(frame, name)

    def local_from_sources(suffix):
        source = pd.DataFrame({z: read(f"feature_{z.lower()}_{suffix}") for z in ZONES})
        return source

    residual = local_from_sources("residual_load_gw")
    gas = local_from_sources("gas_available_gw")
    supply = pd.DataFrame(index=frame.index)
    for zone, components in SUPPLY_COMPONENTS.items():
        values = pd.DataFrame({name: read(name) for name in components})
        if values.lt(0).any().any():
            raise ValueError(f"{zone}: selected supply components cannot be negative.")
        supply[zone] = values.sum(axis=1, min_count=len(components))

    def own(matrix):
        values = np.full(len(frame), np.nan)
        for zone in ZONES:
            mask = frame.zone.eq(zone).to_numpy()
            values[mask] = matrix.loc[mask, zone]
        return pd.Series(values, index=frame.index)

    def peers(matrix):
        values = np.full(len(frame), np.nan)
        for zone in ZONES:
            mask = frame.zone.eq(zone).to_numpy()
            values[mask] = matrix.loc[mask, [z for z in ZONES if z != zone]].sum(axis=1, min_count=3)
        return pd.Series(values, index=frame.index)

    def validated_local(name, derived, *, require_sources=False):
        if name not in frame:
            return derived
        cached = read(name)
        _agree(cached, derived, name)
        # A saved total cannot conceal a missing constituent when completeness
        # of the physical construction is part of the new feature contract.
        return derived if require_sources else cached.where(cached.notna(), derived)

    local_r = validated_local("feature_local_residual_load_gw", own(residual), require_sources=True)
    local_g = validated_local("feature_local_gas_available_gw", own(gas), require_sources=True)
    local_s = validated_local("feature_local_selected_supply_proxy_gw", own(supply), require_sources=True)
    local_margin = validated_local("feature_local_selected_supply_minus_residual_proxy_gw", local_s-local_r, require_sources=True)
    peer_r, peer_s = peers(residual), peers(supply)
    denominator, peer_denominator = local_s.clip(lower=DENOMINATOR_FLOOR_GW), peer_s.clip(lower=DENOMINATOR_FLOOR_GW)
    features = {
        "local_residual_load_gw": local_r,
        "local_gas_available_gw": local_g,
        "local_selected_supply_proxy_gw": local_s,
        "local_selected_supply_minus_residual_proxy_gw": local_margin,
        "local_residual_to_selected_supply_proxy": local_r / denominator,
        "local_margin_to_selected_supply_proxy": local_margin / denominator,
        "local_gas_share_selected_supply_proxy": local_g / denominator,
        "peer_residual_load_gw": peer_r,
        "peer_selected_supply_proxy_gw": peer_s,
        "peer_selected_supply_minus_residual_proxy_gw": peer_s-peer_r,
        "peer_residual_to_selected_supply_proxy": peer_r/peer_denominator,
        "peer_margin_to_selected_supply_proxy": (peer_s-peer_r)/peer_denominator,
        "local_minus_peer_pressure_proxy": local_r/denominator-peer_r/peer_denominator,
        "local_positive_residual_share_regional_proxy": local_r.clip(lower=0)/residual.clip(lower=0).sum(axis=1, min_count=4).clip(lower=DENOMINATOR_FLOOR_GW),
    }
    for technology in ("wind", "solar"):
        local_value = validated_local(f"feature_local_{technology}_generation_gw", own(local_from_sources(f"{technology}_generation_gw")))
        features[f"local_{technology}_generation_gw"] = local_value
        features[f"local_{technology}_to_selected_supply_proxy"] = local_value/denominator
    for driver in ("wind", "solar", "residual"):
        for hours in (1, 3):
            ramp = read(f"feature_local_{driver}_ramp_{hours}h_gw_per_hour")
            features[f"local_{driver}_ramp_{hours}h_to_selected_supply_proxy_per_hour"] = ramp/denominator
    features["local_temperature_c"] = validated_local("feature_local_temperature_c", own(local_from_sources("temperature_c")))
    forecast = read("forecast")
    width, upper = read("q90")-read("q10"), read("q90")-forecast
    for name, values in (("forecast", forecast), ("interval_width", width), ("upper_distance", upper)):
        features[f"baseline_{name}"] = validated_local(f"feature_baseline_{name}", values, require_sources=True)
    if width.dropna().lt(0).any():
        raise ValueError("Baseline q90 cannot be below q10.")
    ccgt, ocgt = read("feature_clean_gas_cost_ccgt_proxy_eur_mwh"), read("feature_clean_gas_cost_ocgt_proxy_eur_mwh")
    features["clean_gas_cost_ccgt_proxy_eur_mwh"] = ccgt
    features["clean_gas_cost_ocgt_proxy_eur_mwh"] = ocgt
    features["gas_merit_slope_proxy_eur_mwh"] = validated_local("feature_gas_merit_slope_proxy_eur_mwh", ocgt-ccgt, require_sources=True)
    features["cgc_minus_baseline_proxy_eur_mwh"] = ccgt-forecast
    local_time = timestamps.dt.tz_convert("Europe/Paris")
    features.update(hour_sin=np.sin(local_time.dt.hour*np.pi/12), hour_cos=np.cos(local_time.dt.hour*np.pi/12),
                    weekday=local_time.dt.dayofweek.astype(float), month=local_time.dt.month.astype(float))
    peer_lookup = pd.DataFrame({"origin": origins, "timestamp": timestamps, "zone": frame.zone, "forecast": forecast})
    wide = peer_lookup.pivot(index=["origin", "timestamp"], columns="zone", values="forecast").reindex(columns=ZONES)
    index = pd.MultiIndex.from_arrays([origins, timestamps], names=["origin", "timestamp"])
    paired = wide.reindex(index).reset_index(drop=True)
    peer_forecast = peers(paired)/3
    features["peer_baseline_mean_forecast"] = peer_forecast
    features["local_minus_peer_baseline_forecast"] = forecast-peer_forecast
    added = pd.DataFrame({PREFIX+name: values for name, values in features.items()}, index=frame.index).astype(float)
    if np.isinf(added.to_numpy()).any():
        raise ValueError("Zonal feature arithmetic produced infinity.")
    main_features = list(added)
    flags = added.isna().astype(float).rename(columns=lambda name: name+"__missing")
    added = pd.concat([added, flags], axis=1)
    required = [PREFIX+name for name in ("local_residual_load_gw", "local_selected_supply_proxy_gw",
                                        "peer_residual_load_gw", "peer_selected_supply_proxy_gw", "baseline_forecast")]
    eligible = added[required].notna().all(axis=1)
    added.index = panel.index.copy()
    augmented = pd.concat([panel.copy(deep=True), added], axis=1)
    pd.testing.assert_frame_equal(augmented[original_columns], panel, check_exact=True)
    audit = {"schema_version": 1, "representation": "local_peer_selected_supply_v1", "rows": len(panel),
             "feature_columns": list(added), "required_feature_columns": required, "raw_feature_count": len(main_features),
             "original_columns_unchanged": True, "fit_performed": False, "sources_read": False,
             "observed_price_or_storm_used": False, "strict_origin": "D-1 08:00 Europe/Paris civil, DST-aware",
             "source_input_columns": sorted(used), "selected_supply_components": {z: list(v) for z, v in SUPPLY_COMPONENTS.items()},
             "supply_semantics": "FR gas available Pmax + nuclear generation forecast; DE gas/coal/lignite available Pmax; BE gas/nuclear available Pmax; NL gas/coal/nuclear available Pmax. Incomplete selected fleet proxy; neither full reserve margin nor feasible imports.",
             "missing_supply_component_policy": "propagate NaN; never partial sum or zero nuclear imputation",
             "ratio_denominator_floor_gw": DENOMINATOR_FLOOR_GW,
             "peer_price_policy": "mean of exactly the other three countries at identical UTC origin and physical delivery timestamp; incomplete peers remain NaN",
             "ramps": "saved within-origin forecast-profile 1h/3h ramps divided by selected-supply proxy; no cross-day interpolation",
             "candidate_inputs_exclude_raw_four_country_vectors": True,
             "eligible_rows": int(eligible.sum()), "missing_required_rows": int((~eligible).sum()),
             "coverage": {name: {"finite_rows": int(added[name].notna().sum()), "missing_rows": int(added[name].isna().sum())} for name in main_features}}
    return augmented, list(added), required, audit
