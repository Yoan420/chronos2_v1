from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import heatwave_features as hf


def sources(start="2024-06-30", end="2024-09-11", values=None):
    index = pd.date_range(pd.Timestamp(start, tz="Europe/Paris"),
        (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize("Europe/Paris"), freq="h", inclusive="left")
    days = index.tz_localize(None).normalize()
    cutoffs = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    output = {}
    for country in hf.COUNTRIES:
        vector = [float(values(country, d)) if values is not None else 25. for d in days]
        output[country] = pd.DataFrame({"snapshot_time_utc": cutoffs, "revision_time_utc": cutoffs,
            "value_time_utc": index.tz_convert("UTC"), "value": vector})
    return output


def build(raw, start="2024-09-09", end="2024-09-11", **kwargs):
    return hf.build_heatwave_features(raw, start_day=start, end_day=end, **kwargs)


def test_builder_is_pure_daily_indices_and_wide_schema():
    raw = sources()
    originals = deepcopy(raw)
    output, audit = build(raw)
    assert len(output) == 72
    assert list(output.columns) == [*hf.TIME_COLUMNS, *hf.heatwave_feature_aliases()]
    assert len(hf.heatwave_feature_aliases()) == 17
    assert audit["raw_prefix_start_day"] == "2024-06-30"
    assert audit["future_data_used_for_past_features"] is False
    assert audit["observed_temperature_used"] is False
    assert audit["production_pit_evidence"] is False
    assert (output.fr_heat_excess_fcst_c == 0).all()  # Previous 71+ days at 25C.
    assert (output.cooling_degree_mean_fcst_c == 3).all()
    for country in raw:
        pd.testing.assert_frame_equal(raw[country], originals[country])


def test_fixed_floor_startup_streak_and_fraction():
    raw = sources("2024-07-01", "2024-07-10")
    output, audit = build(raw, "2024-07-01", "2024-07-10")
    daily = output.iloc[::24]
    assert daily.fr_heat_excess_fcst_c.tolist() == [5.] * 10
    assert daily.es_heat_excess_fcst_c.tolist() == [1.] * 10
    assert daily.fr_heat_streak_fcst_days.tolist() == [1., 2., 3., 4., 5., 6., 7., 7., 7., 7.]
    assert daily.heat_fraction_fcst.tolist() == [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.]
    assert {r["threshold_source"] for r in audit["daily_heat_audit"]} == {"fixed_floor_warmup"}


def test_threshold_excludes_current_day_and_uses_exact_365_prior_days():
    first, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-02")
    raw = sources(str(first.date()), str(end.date()), lambda c, d: 21. + (d - first).days / 100.)
    output, audit = build(raw, str(end.date()), str(end.date()))
    row = next(r for r in audit["daily_heat_audit"] if r["country"] == "FR")
    prior = np.array([21. + i / 100. for i in range((end - first).days - 365, (end - first).days)])
    assert row["prior_days"] == 365
    assert row["threshold_fcst_c"] == pytest.approx(np.quantile(prior, .9))
    assert row["excess_fcst_c"] == pytest.approx(21. + (end - first).days / 100 - np.quantile(prior, .9))


def test_exact_minimum_days_threshold_transition():
    raw = sources("2024-01-01", "2024-03-05")
    output, audit = build(raw, "2024-02-29", "2024-03-01")
    fr = [r for r in audit["daily_heat_audit"] if r["country"] == "FR"]
    assert [r["prior_days"] for r in fr] == [59, 60]
    assert [r["threshold_fcst_c"] for r in fr] == [20., 25.]


def test_later_temperature_changes_do_not_revise_earlier_features():
    raw = sources()
    before, _ = build(raw)
    changed = deepcopy(raw)
    for frame in changed.values():
        mask = frame.value_time_utc >= pd.Timestamp("2024-09-11", tz="Europe/Paris")
        frame.loc[mask, "value"] = 40.
    after, _ = build(changed)
    pd.testing.assert_frame_equal(before.iloc[:48], after.iloc[:48])
    assert after.fr_heat_excess_fcst_c.iloc[-1] == 15.
    assert before.fr_heat_excess_fcst_c.iloc[-1] == 0.


def test_future_raw_rows_beyond_requested_end_are_not_used():
    raw = sources(end="2024-09-15")
    full, _ = build(raw)
    truncated = {c: f.loc[f.value_time_utc < pd.Timestamp("2024-09-12", tz="Europe/Paris")].copy() for c, f in raw.items()}
    cut, _ = build(truncated)
    pd.testing.assert_frame_equal(full, cut)


@pytest.mark.parametrize("day, hours, cutoff", [("2026-03-29", 23, "2026-03-28T08:00:00+01:00"),
    ("2026-03-30", 24, "2026-03-29T08:00:00+02:00"),
    ("2025-10-26", 25, "2025-10-25T08:00:00+02:00"),
    ("2025-10-27", 24, "2025-10-26T08:00:00+01:00")])
def test_civil_cutoff_and_physical_dst_hours(day, hours, cutoff):
    raw = sources(day, day)
    for frame in raw.values():
        for name in hf.TIME_COLUMNS:
            frame[name] = frame[name].dt.tz_convert("Europe/Paris")
    output, audit = build(raw, day, day)
    assert len(output) == hours
    assert not output.value_time_utc.duplicated().any()
    assert (output.snapshot_time_utc == pd.Timestamp(cutoff).tz_convert("UTC")).all()


@pytest.mark.parametrize("fault", ["missing_hour", "intraday_change", "nan", "infinity", "units", "naive",
    "duplicate_conflict", "latest_nan", "late_only", "late_first_prefix", "different_prefix", "missing_country"])
def test_unsafe_raw_inputs_refused(fault):
    raw = sources()
    frame = raw["FR"]
    if fault == "missing_hour": raw["FR"] = frame.drop(index=50)
    elif fault == "intraday_change": frame.loc[50, "value"] = 24.
    elif fault == "nan": frame.loc[50, "value"] = np.nan
    elif fault == "infinity": frame.loc[50, "value"] = np.inf
    elif fault == "units": frame.loc[:, "value"] = 300.
    elif fault == "naive": frame["value_time_utc"] = frame.value_time_utc.dt.tz_localize(None)
    elif fault == "duplicate_conflict":
        duplicate = frame.iloc[[50]].copy(); duplicate["value"] = 30.
        raw["FR"] = pd.concat([frame, duplicate], ignore_index=True)
    elif fault == "latest_nan":
        frame["snapshot_time_utc"] -= pd.Timedelta(minutes=1)
        frame["revision_time_utc"] -= pd.Timedelta(minutes=1)
        duplicate = frame.iloc[[50]].copy(); duplicate["value"] = np.nan
        duplicate["snapshot_time_utc"] += pd.Timedelta(minutes=1)
        duplicate["revision_time_utc"] += pd.Timedelta(minutes=1)
        raw["FR"] = pd.concat([frame, duplicate], ignore_index=True)
    elif fault == "late_only": frame["revision_time_utc"] += pd.Timedelta(minutes=1)
    elif fault == "late_first_prefix": frame.loc[:23, "revision_time_utc"] += pd.Timedelta(minutes=1)
    elif fault == "different_prefix": raw["FR"] = frame.iloc[24:].copy()
    elif fault == "missing_country": del raw["FR"]
    with pytest.raises(hf.HeatwaveFeatureError):
        build(raw)


def test_post_cutoff_revision_is_excluded_without_using_its_value():
    raw = sources()
    baseline, _ = build(raw)
    duplicate = raw["FR"].iloc[[-1]].copy()
    duplicate["revision_time_utc"] += pd.Timedelta(seconds=1)
    duplicate["value"] = 60.
    raw["FR"] = pd.concat([raw["FR"], duplicate], ignore_index=True)
    output, audit = build(raw)
    pd.testing.assert_frame_equal(output, baseline)
    assert audit["source_audits"]["FR"]["post_cutoff_rows_excluded"] == 1


def test_wide_source_and_alias_mapping_are_equivalent():
    raw = sources()
    wide = raw["FR"][list(hf.TIME_COLUMNS)].copy()
    aliases = {}
    for country, alias in zip(hf.COUNTRIES, hf.temperature_aliases()):
        wide[alias] = raw[country].value
        aliases[alias] = raw[country]
    left, _ = build(wide)
    right, _ = build(aliases)
    pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize("config", [{"countries": ["FR"]}, {"lookback_days": 364}, {"minimum_history_days": True},
    {"threshold_quantile": 1.}, {"threshold_quantile": np.nan}, {"floor_thresholds_c": {"FR": 20.}},
    {"persistent_days": 8}, {"streak_clip_days": 50}, {"include_cooling_mean": "yes"}, {"unknown": 1}])
def test_configuration_refuses_ambiguous_or_unsafe_parameters(config):
    with pytest.raises(hf.HeatwaveFeatureError):
        hf.HeatwaveFeatureConfig.from_mapping(config)


def test_optional_cooling_and_metadata():
    config = hf.HeatwaveFeatureConfig.from_mapping({"include_cooling_mean": False})
    output, audit = build(sources(), config=config)
    assert hf.COOLING_MEAN_ALIAS not in output
    assert len(hf.heatwave_feature_aliases(config)) == 16
    assert hf.feature_metadata()["fr_temperature_fcst"]["semantic"] == "daily_national_temperature_forecast_index"
    assert all(r["prior_days"] >= 60 for r in audit["daily_heat_audit"])


def test_binary32_preparation_of_fraction_and_cooling_is_valid_without_rewriting():
    raw = sources("2024-07-01", "2024-07-03", lambda c, d: {
        "FR": 26.2345699, "DE": 24.1234567, "BE": 25.8765432, "NL": 19.9876543, "ES": 22.1234987}[c])
    frame, _ = build(raw, "2024-07-01", "2024-07-03")
    floats = frame[list(hf.heatwave_feature_aliases())].astype("float32")
    saved = floats.copy(deep=True)
    assert floats.heat_fraction_fcst.iloc[-1] == np.float32(.6)
    audit = hf.validate_heatwave_aggregates(floats)
    assert 0 < audit["fraction_max_abs_difference"] < 3e-8
    assert audit["fraction_max_absolute_tolerance"] < 6e-8
    assert 0 < audit["cooling_max_abs_difference"] < 2e-6
    assert audit["cooling_max_absolute_tolerance"] < 2e-6
    pd.testing.assert_frame_equal(floats, saved)
    # Parquet/report reload may promote the binary32 numbers back to binary64.
    hf.validate_heatwave_aggregates(floats.astype("float64"))


@pytest.mark.parametrize("alias, error", [(hf.HEAT_FRACTION_ALIAS, 1e-6), (hf.COOLING_MEAN_ALIAS, 1e-4)])
def test_aggregate_discrepancies_exceeding_binary32_bound_still_fail(alias, error):
    frame, _ = build(sources())
    frame = frame[list(hf.heatwave_feature_aliases())].astype("float32").astype("float64")
    frame.loc[0, alias] += error
    with pytest.raises(hf.HeatwaveFeatureError, match="beyond float32 rounding"):
        hf.validate_heatwave_aggregates(frame)


def test_zero_cooling_and_fraction_have_only_machine_rounding_not_a_broad_epsilon():
    frame, _ = build(sources(values=lambda c, d: 18.))
    audit = hf.validate_heatwave_aggregates(frame)
    assert audit["fraction_max_absolute_tolerance"] < 1e-12
    frame.loc[0, hf.HEAT_FRACTION_ALIAS] = 1e-10
    with pytest.raises(hf.HeatwaveFeatureError, match="fraction"):
        hf.validate_heatwave_aggregates(frame)
