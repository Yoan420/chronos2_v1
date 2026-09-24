"""Native-QH source contracts, causal vintage selection, units and DST."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from nyx_intrahour.features import IntrahourFeatureError, build_hourly_features


ALIAS = "fr_residual"
DAY = "2026-09-16"
SOURCE = {"alias": ALIAS, "zone": "FR", "driver": "residual_load", "unit": "GW",
          "series": "forecast.residual_load.fr.native15min", "native_resolution_minutes": 15,
          "is_forecast": True, "interpolation": "none",
          "native_resolution_evidence": "Provider metadata declares native quarter-hour delivery products."}
PREFIX = f"feature_intrahour_{ALIAS}__"


def support(day):
    start = pd.Timestamp(day).tz_localize("Europe/Paris")
    finish = (pd.Timestamp(day)+pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    index = pd.date_range(start, finish, freq="15min", inclusive="left").tz_convert("UTC")
    cutoff = (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    return index, cutoff


def raw(day=DAY, *, pattern=(1.0, 2.0, 3.0, 4.0), alias=ALIAS):
    index, cutoff = support(day)
    return pd.DataFrame({"source_alias": alias, "value_time_utc": index,
                         "snapshot_time_utc": cutoff, "revision_time_utc": cutoff,
                         "value": np.tile(pattern, len(index)//4)})


def build(frame, *, source=None, day=DAY):
    return build_hourly_features(frame, [source or SOURCE], day, day)


def test_exact_features_native_quarters_and_no_input_mutation():
    frame, source = raw(), deepcopy(SOURCE)
    before = frame.copy(deep=True)
    output, audit = build_hourly_features(frame, [source], DAY, DAY)
    assert output.index.name == "timestamp_utc" and str(output.index.tz) == "UTC"
    assert len(output) == 24
    assert output[PREFIX+"mean_gw"].eq(2.5).all()
    np.testing.assert_allclose(output[PREFIX+"std_gw"], np.sqrt(1.25))
    assert output[PREFIX+"range_gw"].eq(3).all()
    assert output[PREFIX+"ramp_gw_per_hour"].eq(4).all()
    assert output[PREFIX+"max_deviation_gw"].eq(1.5).all()
    assert output[PREFIX+"min_deviation_gw"].eq(-1.5).all()
    assert output[f"data_{ALIAS}__complete"].all()
    assert output.forecast_origin_utc.eq(support(DAY)[1]).all()
    assert audit["complete_hours"] == 24
    assert audit["by_source"][ALIAS]["varying_hours"] == 24
    assert audit["production_pit_evidence"] is False
    json.dumps(audit, allow_nan=False)
    pd.testing.assert_frame_equal(frame, before)
    assert source == SOURCE


@pytest.mark.parametrize("late_column", ["snapshot_time_utc", "revision_time_utc"])
def test_both_vintage_timestamps_must_meet_cutoff(late_column):
    base = raw()
    future = base.assign(value=999.0)
    future[late_column] += pd.Timedelta(nanoseconds=1)
    output, audit = build(pd.concat([future, base], ignore_index=True))
    assert output[PREFIX+"mean_gw"].eq(2.5).all()
    assert audit["late_rows_excluded"] == 96
    assert audit["by_source"][ALIAS]["missing_quarters"] == 0


def test_latest_revision_precedes_snapshot_in_selection_order():
    current = raw().assign(snapshot_time_utc=support(DAY)[1]-pd.Timedelta(minutes=10))
    older_revision_later_snapshot = raw().assign(value=90.0,
        revision_time_utc=support(DAY)[1]-pd.Timedelta(minutes=20))
    output, audit = build(pd.concat([older_revision_later_snapshot, current], ignore_index=True))
    assert output[PREFIX+"mean_gw"].eq(2.5).all()
    assert audit["selection_order"] == ["revision_time_utc", "snapshot_time_utc"]


def test_latest_snapshot_breaks_same_revision_tie():
    new = raw()
    old = new.assign(value=99.0, snapshot_time_utc=new.snapshot_time_utc-pd.Timedelta(minutes=1))
    output, _ = build(pd.concat([new, old], ignore_index=True))
    assert output[PREFIX+"mean_gw"].eq(2.5).all()


@pytest.mark.parametrize("missing_value", [np.nan, np.inf, -np.inf])
def test_latest_nonfinite_does_not_fall_back_to_older_finite(missing_value):
    new = raw()
    old = new.assign(snapshot_time_utc=new.snapshot_time_utc-pd.Timedelta(hours=1),
                     revision_time_utc=new.revision_time_utc-pd.Timedelta(hours=1))
    new.loc[0,"value"] = missing_value
    output, audit = build(pd.concat([old, new], ignore_index=True))
    assert output.loc[output.index[0],output.columns.str.startswith(PREFIX)].isna().all()
    assert not output[f"data_{ALIAS}__complete"].iloc[0]
    assert output[f"data_{ALIAS}__complete"].iloc[1:].all()
    assert audit["by_source"][ALIAS]["nonfinite_quarters"] == 1
    assert audit["by_source"][ALIAS]["absent_quarters"] == 0
    assert audit["missing_quarters"] == 1 and audit["complete_hours"] == 23


@pytest.mark.parametrize("day,hours,cutoff", [
    ("2026-03-29",23,"2026-03-28T07:00Z"),
    ("2026-10-25",25,"2026-10-24T06:00Z"),
    ("2026-03-30",24,"2026-03-29T06:00Z"),
    ("2026-10-26",24,"2026-10-25T07:00Z")])
def test_civil_delivery_day_and_previous_day_cutoff_follow_dst(day,hours,cutoff):
    output,audit = build(raw(day),day=day)
    assert len(output)==hours and output.index.is_unique
    assert output.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert output.forecast_origin_utc.eq(pd.Timestamp(cutoff)).all()
    assert audit["expected_quarters"]==hours*4
    assert audit["complete_hours"]==hours
    if hours==25:
        assert (output.index.tz_convert("Europe/Paris").hour==2).sum()==2
    if hours==23:
        assert not (output.index.tz_convert("Europe/Paris").hour==2).any()


def test_mw_sources_convert_every_feature_to_gw():
    expected,_ = build(raw())
    actual,audit = build(raw().assign(value=raw().value*1000),source={**SOURCE,"unit":"MW"})
    pd.testing.assert_frame_equal(actual,expected)
    assert audit["sources"][0]["unit"]=="MW"
    assert audit["feature_contract"]["output_unit"]=="GW"


def test_flat_native_profile_remains_valid_and_is_not_claimed_interpolated():
    output,audit = build(raw(pattern=(-2,-2,-2,-2)))
    assert output[PREFIX+"mean_gw"].eq(-2).all()
    assert output[PREFIX+"std_gw"].eq(0).all()
    assert audit["complete_hours"]==24
    assert audit["by_source"][ALIAS]["varying_hours"]==0
    assert audit["native_resolution_independently_verified"] is False


def test_missing_quarter_is_not_interpolated_or_replaced_by_zero():
    frame = raw().drop(index=1)
    output,audit = build(frame)
    assert output.loc[output.index[0],output.columns.str.startswith(PREFIX)].isna().all()
    assert audit["by_source"][ALIAS]["absent_quarters"]==1
    assert audit["by_source"][ALIAS]["nonfinite_quarters"]==0
    assert audit["complete_hours"]==23


def test_empty_table_retains_full_support_with_missing_features():
    output,audit = build(raw().iloc[:0])
    assert len(output)==24
    assert output.loc[:,output.columns.str.startswith(PREFIX)].isna().all().all()
    assert not output[f"data_{ALIAS}__complete"].any()
    assert audit["missing_quarters"]==96


def test_identical_duplicate_is_deduplicated_but_conflicting_tie_is_rejected():
    frame = raw()
    output,audit = build(pd.concat([frame,frame.iloc[[0]]],ignore_index=True))
    assert audit["identical_duplicates_removed"]==1
    assert audit["expected_quarters"]==96 and audit["complete_hours"]==24
    for conflicting in [9.0,np.nan]:
        with pytest.raises(IntrahourFeatureError,match="Conflicting"):
            build(pd.concat([frame,frame.iloc[[0]].assign(value=conflicting)],ignore_index=True))


@pytest.mark.parametrize("change", [
    {"native_resolution_minutes":60}, {"interpolation":"linear"},
    {"is_forecast":False}, {"is_forecast":"true"}, {"native_resolution_evidence":" "},
    {"driver":"price"}, {"unit":"EUR/MWh"}, {"series":""}, {"zone":"UK"},
    {"alias":"bad-name"}])
def test_non_native_or_unverified_source_manifests_are_rejected(change):
    with pytest.raises(IntrahourFeatureError):
        build(raw(),source={**SOURCE,**change})


@pytest.mark.parametrize("column", ["value_time_utc","snapshot_time_utc","revision_time_utc"])
def test_naive_and_missing_identity_timestamps_are_rejected(column):
    naive = raw()
    naive[column] = naive[column].dt.tz_localize(None)
    with pytest.raises(IntrahourFeatureError,match="timezone-aware"):
        build(naive)
    missing = raw()
    missing.loc[0,column] = pd.NaT
    with pytest.raises(IntrahourFeatureError,match="nonmissing"):
        build(missing)


def test_quarter_alignment_includes_seconds_and_nanoseconds():
    for offset in [pd.Timedelta(minutes=1),pd.Timedelta(seconds=1),pd.Timedelta(nanoseconds=1)]:
        frame = raw()
        frame.loc[0,"value_time_utc"] += offset
        with pytest.raises(IntrahourFeatureError,match="aligned"):
            build(frame)


def test_multiple_sources_days_and_exact_window_do_not_cross_delivery_boundaries():
    second = {**SOURCE,"alias":"de_wind","zone":"DE","driver":"wind"}
    days = ["2026-09-15",DAY,"2026-09-17","2026-09-18"]
    frames = [raw(day,alias=alias) for day in days for alias in [ALIAS,"de_wind"]]
    frames[2] = frames[2].iloc[1:]  # FR first quarter on requested first day.
    result,audit = build_hourly_features(pd.concat(frames,ignore_index=True),[SOURCE,second],DAY,"2026-09-17")
    assert len(result)==48 and audit["days"]==2
    assert audit["out_of_window_rows"]==384
    assert audit["expected_source_quarters"]==384
    assert audit["complete_hours"]==47
    assert audit["by_source"]["de_wind"]["complete_hours"]==48
    assert result.forecast_origin_utc.iloc[0]==support(DAY)[1]
    assert result.forecast_origin_utc.iloc[-1]==support("2026-09-17")[1]


def test_undeclared_aliases_duplicate_declarations_and_invalid_date_windows_fail():
    with pytest.raises(IntrahourFeatureError,match="declared"):
        build(raw(alias="other"))
    with pytest.raises(IntrahourFeatureError,match="Duplicate source alias"):
        build_hourly_features(raw(),[SOURCE,SOURCE],DAY,DAY)
    with pytest.raises(IntrahourFeatureError,match="after"):
        build_hourly_features(raw(),[SOURCE],"2026-09-17",DAY)
    with pytest.raises(IntrahourFeatureError,match="YYYY-MM-DD"):
        build_hourly_features(raw(),[SOURCE],"16/09/2026",DAY)
