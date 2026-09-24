"""Same-vintage physical solar/ramp inputs for a diagnostic-only expert.

The source panel is preserved exactly. No labels, realised power, Storm or
electricity-price columns are inspected. An explicitly opted-in, separate
``baseline`` group may read the two saved baseline feature columns only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PREFIX = "solarx_"
ZONES = ("FR", "DE", "BE", "NL")
TIMEZONE = "Europe/Paris"
SUPPLY_COMPONENTS = {
    "FR": ("feature_fr_gas_available_gw", "feature_fr_nuclear_generation_gw"),
    "DE": ("feature_de_gas_available_gw", "feature_de_coal_available_gw", "feature_de_lignite_available_gw"),
    "BE": ("feature_be_gas_available_gw", "feature_be_nuclear_available_gw"),
    "NL": ("feature_nl_gas_available_gw", "feature_nl_coal_available_gw", "feature_nl_nuclear_available_gw"),
}
SUPPLY_FLOOR_GW = 1.0


def _utc(values: pd.Series, name: str, *, missing: bool = False) -> pd.Series:
    if (not missing and values.isna().any()) or any(
        pd.Timestamp(value).tzinfo is None for value in values.dropna()
    ):
        raise ValueError(f"{name}: explicit timezone-aware timestamps required")
    return pd.to_datetime(values, utc=True, errors="raise")


def _ramp(values, zones, origins, timestamps, hours):
    """Exact physical lag in one already-issued profile, never a row shift."""
    keys = pd.MultiIndex.from_arrays([zones, origins, timestamps])
    lag = pd.MultiIndex.from_arrays([zones, origins, timestamps-pd.Timedelta(hours=hours)])
    previous = pd.Series(values.to_numpy(), index=keys).reindex(lag).to_numpy()
    return pd.Series((values.to_numpy()-previous)/hours, index=values.index)


def build_features(panel: pd.DataFrame, *, include_baseline: bool = False):
    """Return ``(unchanged_panel_plus_features, disjoint_feature_groups, audit)``.

    Consumers concatenate predeclared groups for ablations. ``solarx_eligible``
    is a gate, never a model input: all four countries' solar and residual-load
    forecasts must be finite and (when provided) available by the origin.
    Optional supply/weather gaps and the first hours' unavailable ramps remain
    NaN. Baseline inputs require explicit opt-in and stay in their own group.
    """
    identity = {"zone", "timestamp_utc", "forecast_origin_utc"}
    if not isinstance(panel, pd.DataFrame) or panel.empty or panel.columns.has_duplicates or not identity.issubset(panel):
        raise ValueError("Nonempty panel with unique columns and zone/timestamp/origin required")
    if type(include_baseline) is not bool:
        raise ValueError("include_baseline must be an explicit boolean")
    if any(str(name).startswith(PREFIX) for name in panel.columns):
        raise ValueError("Reserved solarx_ columns already exist")
    original_columns = list(panel.columns)
    frame = panel.reset_index(drop=True)
    if frame.zone.isna().any() or not frame.zone.isin(ZONES).all():
        raise ValueError("Only FR, DE, BE and NL zones are supported")
    timestamps = _utc(frame.timestamp_utc, "timestamp_utc")
    origins = _utc(frame.forecast_origin_utc, "forecast_origin_utc")
    if not timestamps.eq(timestamps.dt.floor("h")).all():
        raise ValueError("Only aligned physical hourly delivery timestamps are supported")
    local = timestamps.dt.tz_convert(TIMEZONE)
    civil = local.dt.tz_localize(None).dt.normalize()
    expected = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).dt.tz_localize(TIMEZONE).dt.tz_convert("UTC")
    if not origins.eq(expected).all():
        raise ValueError("Every origin must equal D-1 08:00 Europe/Paris civil, including DST")
    keys = pd.DataFrame({"zone": frame.zone, "timestamp": timestamps, "origin": origins})
    if keys.duplicated().any():
        raise ValueError("Duplicate physical zone/origin/delivery identity")

    read_columns: set[str] = set()

    def read(name, *, nonnegative=False):
        read_columns.add(name)
        if name not in frame:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        values = pd.to_numeric(frame[name], errors="raise").astype(float)
        if np.isinf(values.to_numpy()).any():
            raise ValueError(f"{name}: infinite input is forbidden")
        if nonnegative and values.lt(0).any():
            raise ValueError(f"{name}: negative generation/available capacity is forbidden")
        return values

    def countries(suffix, *, nonnegative=False):
        return pd.DataFrame({z: read(f"feature_{z.lower()}_{suffix}", nonnegative=nonnegative) for z in ZONES})

    def own(values):
        result = pd.Series(np.nan, index=frame.index, dtype=float)
        for zone in ZONES:
            mask = frame.zone.eq(zone)
            result.loc[mask] = values.loc[mask, zone]
        return result

    def peers(values):
        result = pd.Series(np.nan, index=frame.index, dtype=float)
        for zone in ZONES:
            mask = frame.zone.eq(zone)
            result.loc[mask] = values.loc[mask, [z for z in ZONES if z != zone]].sum(axis=1, min_count=3)
        return result

    residual = countries("residual_load_gw")
    solar = countries("solar_generation_gw", nonnegative=True)
    wind = countries("wind_generation_gw", nonnegative=True)
    gas = countries("gas_available_gw", nonnegative=True)
    temperature = countries("temperature_c")
    supply = pd.DataFrame({z: pd.DataFrame({c: read(c, nonnegative=True) for c in components}).sum(
        axis=1, min_count=len(components)) for z, components in SUPPLY_COMPONENTS.items()})
    lr, pr = own(residual), peers(residual)
    ls, ps = own(solar), peers(solar)
    lw, pw = own(wind), peers(wind)
    lc, pc = own(supply), peers(supply)
    ld, pdn = lc.clip(lower=SUPPLY_FLOOR_GW), pc.clip(lower=SUPPLY_FLOOR_GW)
    lp, pp = lr/ld, pr/pdn

    values = {}
    groups = {name: [] for name in ("calendar", "controls", "baseline", "local_solar", "regional_solar", "ramps", "interactions")}

    def add(group, name, value):
        name = PREFIX+name
        values[name] = value
        groups[group].append(name)

    for name, value in {
        "hour_sin": np.sin(local.dt.hour*np.pi/12), "hour_cos": np.cos(local.dt.hour*np.pi/12),
        "weekday": local.dt.dayofweek, "month": local.dt.month,
    }.items():
        add("calendar", name, value)
    for name, value in {
        "local_residual_gw": lr, "peer_residual_gw": pr,
        "local_wind_gw": lw, "peer_wind_gw": pw,
        "local_temperature_c": own(temperature), "peer_temperature_mean_c": peers(temperature)/3,
        "local_supply_proxy_gw": lc, "peer_supply_proxy_gw": pc,
        "local_gas_proxy_gw": own(gas), "peer_gas_proxy_gw": peers(gas),
        "local_pressure_proxy": lp, "peer_pressure_proxy": pp,
        "local_supply_minus_residual_proxy_gw": lc-lr, "peer_supply_minus_residual_proxy_gw": pc-pr,
        "clean_gas_cost_ccgt_proxy_eur_mwh": read("feature_clean_gas_cost_ccgt_proxy_eur_mwh"),
        "clean_gas_cost_ocgt_proxy_eur_mwh": read("feature_clean_gas_cost_ocgt_proxy_eur_mwh"),
    }.items():
        add("controls", name, value)
    if include_baseline:
        add("baseline", "baseline_forecast", read("feature_baseline_forecast"))
        add("baseline", "baseline_interval_width", read("feature_baseline_interval_width", nonnegative=True))
    add("local_solar", "local_solar_gw", ls)
    add("local_solar", "local_solar_to_supply_proxy", ls/ld)
    add("regional_solar", "peer_solar_gw", ps)
    add("regional_solar", "peer_solar_to_supply_proxy", ps/pdn)
    for side, profiles in (("local", (ls, lr, lw)), ("peer", (ps, pr, pw))):
        for driver, profile in zip(("solar", "residual", "wind"), profiles):
            for hours in (1, 3):
                ramp = _ramp(profile, frame.zone, origins, timestamps, hours)
                # Positive solar drop is loss of forecast generation, not an
                # observed ramp or a change between different issue vintages.
                kind = "drop" if driver == "solar" else "ramp"
                # Non-solar ramps belong in every comparator: the incremental
                # ramps ablation must add solar information only.
                add("ramps" if driver == "solar" else "controls",
                    f"{side}_{driver}_{kind}_{hours}h_gwph", (-ramp).clip(lower=0) if driver == "solar" else ramp)
    ldrop = values[PREFIX+"local_solar_drop_1h_gwph"]
    pdrop = values[PREFIX+"peer_solar_drop_1h_gwph"]
    for name, value in {
        "local_drop_x_residual_gw2ph": ldrop*lr.clip(lower=0),
        "peer_drop_x_residual_gw2ph": pdrop*pr.clip(lower=0),
        "local_drop_x_local_pressure_proxy_gwph": ldrop*lp.clip(lower=0),
        "local_drop_x_peer_pressure_proxy_gwph": ldrop*pp.clip(lower=0),
        "peer_drop_x_local_pressure_proxy_gwph": pdrop*lp.clip(lower=0),
        "peer_drop_x_peer_pressure_proxy_gwph": pdrop*pp.clip(lower=0),
        "local_drop_to_supply_proxy_per_hour": ldrop/ld,
        "peer_drop_to_supply_proxy_per_hour": pdrop/pdn,
    }.items():
        add("interactions", name, value)

    added = pd.DataFrame(values, index=frame.index).astype(float)
    if np.isinf(added.to_numpy()).any():
        raise ValueError("Physical feature arithmetic overflowed finite values")
    source_complete = solar.notna().all(axis=1) & residual.notna().all(axis=1)
    available = pd.Series(True, index=frame.index)
    if "feature_available_at_utc" in frame:
        read_columns.add("feature_available_at_utc")
        times = _utc(frame.feature_available_at_utc, "feature_available_at_utc", missing=True)
        available = times.notna() & times.le(origins)
    eligible = source_complete & available
    added[PREFIX+"eligible"] = eligible
    added.index = panel.index.copy()
    augmented = pd.concat([panel.copy(deep=True), added], axis=1)
    pd.testing.assert_frame_equal(augmented[original_columns], panel, check_exact=True)
    names = [name for group in groups.values() for name in group]
    audit = {
        "schema_version": 1, "representation": "solar_ramp_same_origin_v1", "rows": len(panel),
        "feature_groups": groups, "feature_columns": names, "feature_count": len(names),
        "ablation_contract": "Residual/wind ramps are controls in every variant; the ramps group adds only four local/peer 1h/3h solar drops",
        "input_source_columns_read": sorted(read_columns), "original_columns_unchanged": True,
        "cutoff": "D-1 08:00 Europe/Paris civil, DST-aware; aware UTC physical-hour joins",
        "ramps": "Exact 1h/3h delivery difference / hours within the same zone and forecast origin; solar drop=max(-ramp,0); no cross-day/origin lag or interpolation",
        "peer_aggregation": "Other three countries only, min_count=3; no incomplete sum",
        "renewable_accounting": "Residual load is already net of renewables; solar/wind are separate inputs, never subtracted again; gross load is not reconstructed",
        "supply_semantics": "Incomplete mixed available-Pmax/nuclear-generation proxy, not a reserve margin or feasible imports",
        "supply_components": {z: list(c) for z, c in SUPPLY_COMPONENTS.items()},
        "supply_denominator_floor_gw": SUPPLY_FLOOR_GW,
        "interactions": "Products of known forecast profiles, positive residual load, pressure proxies, and solar-drop/supply ratios; no fitted thresholds",
        "baseline_opt_in": include_baseline, "baseline_group_separate": True,
        "labels_actual_storm_or_realised_power_used": False, "forecast_revision_features_used": False,
        "model_fit_performed": False, "source_reads_performed": False,
        "eligibility": "All four solar and residual-load forecasts finite and optional feature availability <= origin; optional missing ramps/controls stay NaN",
        "eligible_rows": int(eligible.sum()), "missing_required_rows": int((~source_complete).sum()),
        "unavailable_rows": int((~available).sum()),
        "coverage": {name: {"finite_rows": int(added[name].notna().sum()), "missing_rows": int(added[name].isna().sum())} for name in names},
        "diagnostic_only": True, "production_modified": False,
        "pit_limitation": "This builder validates origins and saved availability, not provider publication/capture evidence; upstream snapshot audits remain required",
    }
    return augmented, groups, audit
