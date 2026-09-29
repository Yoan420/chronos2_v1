"""Physical forecast shapes from one delivery day's supplied profiles.

Daily aggregates use only forecasts sharing one supplied daily snapshot.
The caller binds actual source revisions, including current-fit recovery.
Ramps never borrow a following day's forecast profile. FR residual load
already nets wind, solar and run-of-river; only nuclear is deducted here.
"""
import numpy as np
import pandas as pd

from .nyx_local_squared_price_model import frame_hash


def build_forecast_profile_features(features, *, profile_history_contract=None):
    contract = dict(profile_history_contract or {})
    if contract:
        from .nyx_annual_live_preflight import validate_profile_history_contract
        # The source and bundle validators bind this ceiling to the outer run.
        validate_profile_history_contract(contract, pd.Timestamp(contract.get("profile_revision_ceiling_utc")))
    index = features.index
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC" or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing or not index.equals(index.floor("h"))):
        raise ValueError("Unique ordered UTC forecast hours required")
    days = index.tz_convert("Europe/Paris").date
    expected_by_day = {}
    for day in np.unique(days):
        # Compute both civil midnights independently: adding 24 physical hours
        # to an aware midnight would give the wrong DST day boundary.
        start = pd.Timestamp(day)
        stop = start+pd.Timedelta(days=1)
        expected_by_day[day] = len(pd.date_range(start, stop, inclusive="left", freq="h", tz="Europe/Paris"))
    expected_hours = pd.Series(days, index=index).map(expected_by_day)
    output = {}

    def series(column):
        value = features[column].astype(float).copy()
        value = value.where(features[column+"__available"].eq(1))
        if np.isinf(value.to_numpy()).any():
            raise ValueError("Infinite forecast profile")
        return value

    def shape(name, value, daily_max=False):
        grouped = value.groupby(days)
        for hours in (1, 3):
            # At DST folds, offsets are physical hours; boundaries stay civil.
            lag = grouped.shift(hours)
            times = pd.Series(index, index=index).groupby(days).shift(hours)
            exact = (pd.Series(index, index=index)-times).eq(pd.Timedelta(hours=hours))
            output[name+f"_ramp_{hours}h_gw"] = (value-lag).where(exact)
        complete = grouped.transform("count").eq(expected_hours)
        output[name+"_daily_range_gw"] = (grouped.transform("max")-grouped.transform("min")).where(complete)
        if daily_max:
            output[name+"_daily_max_gw"] = grouped.transform("max").where(complete)

    residuals = {}
    for zone in ("fr", "de", "be", "nl"):
        for field, suffix in (("residual", "residual_load_fcst"), ("solar", "solar_generation_fcst")):
            value = series(f"known__{zone}_{suffix}")
            shape(f"extra_profile_{zone}_{field}", value)
            if field == "residual":
                residuals[zone] = value
    regional = pd.DataFrame(residuals).sum(axis=1, min_count=4)
    shape("extra_profile_cwe_residual", regional, daily_max=True)
    dispatchable = residuals["fr"]-series("known__fr_nuclear_generation_fcst_gw")
    output["extra_profile_fr_residual_after_nuclear_gw"] = dispatchable
    groups = dispatchable.groupby(days)
    complete = groups.transform("count").eq(expected_hours)
    output["extra_profile_fr_residual_after_nuclear_daily_max_gw"] = groups.transform("max").where(complete)
    previous = groups.shift(3)
    timestamps = pd.Series(index, index=index)
    exact = (timestamps-timestamps.groupby(days).shift(3)).eq(pd.Timedelta(hours=3))
    output["extra_profile_fr_residual_after_nuclear_ramp_3h_gw"] = (dispatchable-previous).where(exact)
    frame = pd.DataFrame(output, index=index)
    if frame.shape[1] != 31:
        raise ValueError("Unexpected forecast-shape feature schema")
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("Nonfinite forecast-shape arithmetic")
    for name in tuple(frame.columns):
        frame[name+"__available"] = frame[name].notna().astype(float)
    return frame, {"protocol": "nyx_known_forecast_daily_profiles_v1", "value_features": 31,
        "total_features": 62, "feature_sha256": frame_hash(frame), "source_sha256": frame_hash(features),
        "source": ("Daily forecast snapshots; historical recovery uses the outer forecast cutoff; "
                   "internal origins are logical reconstruction origins" if contract else
                   "Immutable forecast features at each delivery day's own D-1 08h origin"),
        **contract,
        "ramps_cross_civil_day": False, "hours_are_physical": True,
        "run_of_river_deducted_again": False, "Storm_used_as_input": False,
        "generation_observations_used": False, "daily_aggregates_require_all_supplied_rows_finite": True,
        "daily_aggregates_require_expected_civil_hours": True, "expected_civil_day_hours": [23, 24, 25]}
