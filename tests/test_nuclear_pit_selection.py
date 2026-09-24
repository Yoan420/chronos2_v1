"""Civil DST selection and real preparation without network or inference."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nuclear_preparation import (
    NuclearPreparationError, _civil_cutoffs, _require_complete,
    _select_target_context, _strict_selection, prepare_nuclear_zone_data,
)
from chronos2_modular.common import SeriesSpec, ZoneConfig


ALIAS = "fr_residual_load_fcst"
SPEC = SeriesSpec(
    alias=ALIAS, source="pit_parquet", fill_method="none",
    known_future=True, future_strategies=("oracle",),
)


def physical(day: str | pd.Timestamp, timezone="Europe/Paris"):
    day = pd.Timestamp(day)
    return pd.date_range(day.tz_localize(timezone), (day + pd.Timedelta(days=1)).tz_localize(timezone),
                         freq="h", inclusive="left").tz_convert("UTC")


def make_frame(index, value=30.0):
    cutoff = _civil_cutoffs(index, timezone="Europe/Paris", runtime_as_of="2027-01-01T00:00:00Z")
    return pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": cutoff,
                         "revision_time_utc": cutoff, "value": value})


@pytest.mark.parametrize("day,hours", [
    ("2026-03-29", 23), ("2026-03-30", 24),
    ("2026-10-25", 25), ("2026-10-26", 24),
])
def test_dst_cutoff_is_exact_civil_eight_and_excludes_later_revision(day, hours):
    index = physical(day)
    assert len(index) == hours
    expected_cutoff = (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    raw = make_frame(index)
    # Reference is independently constructed on the preceding civil date.
    assert raw.snapshot_time_utc.eq(expected_cutoff).all()
    late = raw.assign(value=999.0, snapshot_time_utc=expected_cutoff + pd.Timedelta(minutes=30),
                      revision_time_utc=expected_cutoff + pd.Timedelta(minutes=30))
    result, audit = _strict_selection(pd.concat([raw, late], ignore_index=True), SPEC,
                                     timezone="Europe/Paris", runtime_as_of="2027-01-01T00:00:00Z")
    assert result.index.equals(index.rename("timestamp"))
    assert result.eq(30).all()
    assert audit["eligible_rows"] == hours


def test_runtime_cap_is_applied_to_both_revision_and_snapshot():
    index = physical("2026-09-09")[:1]
    raw = make_frame(index)
    old = raw.assign(value=20.0, snapshot_time_utc=pd.Timestamp("2026-09-08T05:00Z"),
                     revision_time_utc=pd.Timestamp("2026-09-08T05:00Z"))
    revision_late = old.assign(value=999.0, revision_time_utc=pd.Timestamp("2026-09-08T05:45Z"))
    result, _ = _strict_selection(pd.concat([old, revision_late, raw], ignore_index=True), SPEC,
                                 timezone="Europe/Paris", runtime_as_of="2026-09-08T05:30Z")
    assert result.tolist() == [20.0]


@pytest.mark.parametrize("wide", [False, True])
def test_wide_and_narrow_source_values_have_identical_selection(wide):
    raw = make_frame(physical("2026-09-09"))
    if wide:
        raw = raw.rename(columns={"value": ALIAS}).assign(other_alias=777.0)
    result, audit = _strict_selection(raw, SPEC, timezone="Europe/Paris", runtime_as_of="2026-09-08T06:00Z")
    assert result.eq(30).all()
    assert audit["value_column"] == (ALIAS if wide else "value")


def test_explicit_columns_are_respected():
    raw = make_frame(physical("2026-09-09"))
    renamed = {"value_time_utc": "delivery", "snapshot_time_utc": "seen", "revision_time_utc": "published", "value": "forecast"}
    raw = raw.rename(columns=renamed).assign(value=999)
    spec = replace(SPEC, timestamp_col="delivery", availability_col="seen", revision_col="published", value_col="forecast")
    result, _ = _strict_selection(raw, spec, timezone="Europe/Paris", runtime_as_of="2026-09-08T06:00Z")
    assert result.eq(30).all()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_latest_nonfinite_is_not_replaced_by_older_finite_value(bad_value):
    index = physical("2026-09-09")
    raw = make_frame(index)
    older = raw.assign(snapshot_time_utc=raw.snapshot_time_utc - pd.Timedelta(hours=1),
                       revision_time_utc=raw.revision_time_utc - pd.Timedelta(hours=1))
    raw.loc[0, "value"] = bad_value
    selected, _ = _strict_selection(pd.concat([older, raw], ignore_index=True), SPEC,
                                    timezone="Europe/Paris", runtime_as_of="2026-09-08T06:00Z")
    assert not np.isfinite(selected.iloc[0])
    with pytest.raises(NuclearPreparationError, match=r"fr_residual_load_fcst.*1/24.*2026-09-09"):
        _require_complete(selected, index, alias=ALIAS, timezone="Europe/Paris")


@pytest.mark.parametrize("column", ["value_time_utc", "snapshot_time_utc", "revision_time_utc"])
def test_naive_identity_timestamps_are_rejected(column):
    raw = make_frame(physical("2026-09-09"))
    raw[column] = raw[column].dt.tz_localize(None)
    with pytest.raises(NuclearPreparationError, match="sans fuseau"):
        _strict_selection(raw, SPEC, timezone="Europe/Paris", runtime_as_of="2026-09-08T06:00Z")


def test_conflicting_exact_vintage_ties_are_rejected():
    raw = make_frame(physical("2026-09-09")[:1])
    with pytest.raises(NuclearPreparationError, match="contradictoires"):
        _strict_selection(pd.concat([raw, raw.assign(value=31)], ignore_index=True), SPEC,
                          timezone="Europe/Paris", runtime_as_of="2026-09-08T06:00Z")


@pytest.mark.parametrize("spec", [replace(SPEC, fill_method="ffill"), replace(SPEC, known_future=False),
                                 replace(SPEC, future_strategies=("persistence",))])
def test_implicit_imputation_and_non_forecast_inputs_are_rejected(spec):
    with pytest.raises(NuclearPreparationError):
        _strict_selection(make_frame(physical("2026-09-09")), spec, timezone="Europe/Paris", runtime_as_of="2026-09-08T06:00Z")


def test_missing_hour_diagnostic_includes_alias_counts_and_civil_day():
    index = physical("2026-10-25")
    with pytest.raises(NuclearPreparationError, match=r"fr_residual_load_fcst.*1/25.*2026-10-25"):
        _require_complete(pd.Series(30.0, index=index.delete(4)), index, alias=ALIAS, timezone="Europe/Paris")


@pytest.mark.parametrize("day,expected_future_hours", [("2026-03-30", 24), ("2026-10-25", 25), ("2026-10-26", 24)])
@pytest.mark.parametrize("incremental", [False, True])
def test_real_preparation_isolated_inputs_keep_complete_history_across_dst(tmp_path: Path, monkeypatch, day, expected_future_hours, incremental):
    delivery = pd.Timestamp(day)
    runtime = (delivery - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris")
    index = pd.date_range((delivery - pd.Timedelta(days=820)).tz_localize("Europe/Paris"),
                          (delivery + pd.Timedelta(days=1)).tz_localize("Europe/Paris"), freq="h", inclusive="left").tz_convert("UTC")
    target_path = tmp_path / "target.csv"
    pd.DataFrame({"timestamp": index, "value": 50 + np.sin(np.arange(len(index)) / 24)}).to_csv(target_path, index=False)
    pit_path = tmp_path / "bank.parquet"
    make_frame(index).rename(columns={"value": ALIAS}).to_parquet(pit_path, index=False)
    before = pit_path.read_bytes()
    spec = replace(SPEC, pit_file=str(pit_path))
    zone = ZoneConfig("FR", "Europe/Paris", SeriesSpec("target", source="file", file=str(target_path),
                                                       timestamp_col="timestamp", value_col="value"), {ALIAS: spec})
    config = {"data": {"runtime_as_of": runtime.isoformat(), "source": "cache", "project_root": str(tmp_path),
                       "dynamic_delivery_day_horizon": True, "target_end_policy": "current_day_end", "frequency": "h",
                       "target_interpolation_limit": 0}, "model": {"horizon": 24}}
    if incremental:
        from chronos2_hourly.nuclear_incremental import prepare_incremental_settings
        config["nuclear_experiment"] = {"mode": "incremental", "incremental_cache_dir": str(tmp_path / "daily")}
        prepare_incremental_settings(config, (delivery - pd.Timedelta(days=1)).date())
        prepare_incremental_settings(config, delivery.date())
        monkeypatch.setattr("chronos2_hourly.nuclear_incremental.prepare_incremental_settings",
                            lambda *args: pytest.fail("A pinned scope must not create or recompute an epoch during Report."))
    original = deepcopy(config)
    result = prepare_nuclear_zone_data(zone, config, tmp_path, tmp_path / "prepared")
    assert config == original
    assert zone.covariates[ALIAS].file is None
    assert pit_path.read_bytes() == before
    assert result.model_context_covariates[ALIAS].eq(30.0).all()
    assert len(result.model_context_covariates.loc[physical(day)]) == expected_future_hours
    audit = json.loads((tmp_path / "prepared/pit_selection_audit.json").read_text(encoding="utf-8"))
    assert audit["inputs"][0]["missing_hours"] == 0
    assert audit["inputs"][0]["source_sha256"]
    assert audit["required_start_day"] == (delivery - pd.Timedelta(days=731 if incremental else 730)).date().isoformat()
    assert result.diagnostics["nuclear_pit_selection"]["delivery_day"] == day


def test_failed_preflight_writes_no_selected_files(tmp_path: Path):
    pit_path = tmp_path / "short.parquet"
    make_frame(physical("2026-09-09")).to_parquet(pit_path, index=False)
    zone = ZoneConfig("FR", "Europe/Paris", SeriesSpec("target"), {ALIAS: replace(SPEC, pit_file=str(pit_path))})
    with pytest.raises(NuclearPreparationError, match=ALIAS):
        prepare_nuclear_zone_data(zone, {"data": {"runtime_as_of": "2026-09-08T08:00:00+02:00"}}, tmp_path, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_four_year_target_does_not_dilute_complete_replay_covariate_coverage(tmp_path: Path):
    delivery = pd.Timestamp("2026-09-09")
    target_index = pd.date_range((delivery - pd.DateOffset(years=4)).tz_localize("Europe/Paris"),
                                 delivery.tz_localize("Europe/Paris"), freq="h", inclusive="left").tz_convert("UTC")
    replay_start = delivery - pd.Timedelta(days=730)
    pit_index = pd.date_range(replay_start.tz_localize("Europe/Paris"),
                              (delivery + pd.Timedelta(days=1)).tz_localize("Europe/Paris"),
                              freq="h", inclusive="left").tz_convert("UTC")
    target_path = tmp_path / "target.csv.gz"
    pd.DataFrame({"timestamp": target_index, "value": 50.0}).to_csv(target_path, index=False, compression="gzip")
    source_bytes = target_path.read_bytes()
    pit_path = tmp_path / "bank.parquet"
    make_frame(pit_index).to_parquet(pit_path, index=False)
    zone = ZoneConfig("FR", "Europe/Paris", SeriesSpec("target", file=str(target_path), timestamp_col="timestamp", value_col="value"),
                      {ALIAS: replace(SPEC, pit_file=str(pit_path), minimum_coverage=0.5)})
    config = {"data": {"runtime_as_of": "2026-09-08T08:00:00+02:00", "source": "cache", "project_root": str(tmp_path),
                       "dynamic_delivery_day_horizon": True, "target_end_policy": "current_day_end", "frequency": "h",
                       "target_interpolation_limit": 0}, "model": {"horizon": 24, "context_length": 2048}}
    original = deepcopy(config)
    result = prepare_nuclear_zone_data(zone, config, tmp_path, tmp_path / "prepared")
    historical_physical_hours = len(pit_index) - len(physical(delivery))
    assert len(result.target) == historical_physical_hours + 2048
    assert result.target.index[0] == pit_index[0] - pd.Timedelta(hours=2048)
    assert result.target.index[-1] == physical(delivery)[0] - pd.Timedelta(hours=1)
    assert result.coverage.loc[result.coverage.alias == ALIAS, "coverage_after_fill"].iloc[0] > 0.89
    assert len(result.model_context_covariates[ALIAS].iloc[:2048].dropna()) == 0
    assert config == original
    assert zone.target.file == str(target_path)
    assert target_path.read_bytes() == source_bytes
    audit = result.diagnostics["nuclear_pit_selection"]["target_context"]
    assert audit["context_length"] == 2048
    assert audit["source_rows"] == len(target_index)
    assert audit["selected_rows"] == historical_physical_hours + 2048


def test_target_context_missing_hour_is_rejected_before_prepared_output(tmp_path: Path):
    delivery = pd.Timestamp("2026-09-09")
    pit_index = pd.date_range((delivery - pd.Timedelta(days=730)).tz_localize("Europe/Paris"),
                              (delivery + pd.Timedelta(days=1)).tz_localize("Europe/Paris"),
                              freq="h", inclusive="left").tz_convert("UTC")
    pit_path = tmp_path / "bank.parquet"
    make_frame(pit_index).to_parquet(pit_path, index=False)
    target_index = pd.date_range(pit_index[0] - pd.Timedelta(hours=2048), physical(delivery)[0], freq="h", inclusive="left")
    target_path = tmp_path / "target.csv"
    pd.DataFrame({"timestamp": target_index.delete(100), "value": 50.0}).to_csv(target_path, index=False)
    zone = ZoneConfig("FR", "Europe/Paris", SeriesSpec("target", file=str(target_path), timestamp_col="timestamp", value_col="value"),
                      {ALIAS: replace(SPEC, pit_file=str(pit_path))})
    with pytest.raises(NuclearPreparationError, match="target.*1/"):
        prepare_nuclear_zone_data(zone, {"data": {"runtime_as_of": "2026-09-08T08:00:00+02:00"}}, tmp_path, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("kind", ["naive", "duplicate"])
def test_target_identity_is_checked_before_generic_reader_can_repair_it(tmp_path: Path, kind):
    index = physical("2026-10-25")
    frame = pd.DataFrame({"timestamp": index, "value": 50.0})
    if kind == "naive":
        frame["timestamp"] = frame.timestamp.dt.tz_localize(None)
    else:
        frame = pd.concat([frame, frame.iloc[:1].assign(value=200.0)], ignore_index=True)
    target_path = tmp_path / "target.csv"
    frame.to_csv(target_path, index=False)
    zone = ZoneConfig("FR", "Europe/Paris", SeriesSpec("target", file=str(target_path), timestamp_col="timestamp", value_col="value"), {})
    with pytest.raises(NuclearPreparationError, match="target"):
        _select_target_context(zone, {}, tmp_path, index, pd.Timestamp("2026-10-26"))
