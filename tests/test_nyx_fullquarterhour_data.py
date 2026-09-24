from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.features import build_calendar_features
from chronos2_hourly.models.residual_corrector import ResidualMetaFeatureBuilder
from chronos2_modular.data import calendar_frame
from chronos2_modular.forecasting import build_origin_frames
from nyx_intrahour.data import HOURLY_ALIASES, ZONES
from nyx_quarterhour.data import day_index
from nyx_fullquarterhour.data import (
    CONTEXT_COLUMNS, FUTURE_COLUMNS, ORACLE_COLUMNS, ZONE_TIMEZONES,
    FullChainInputs, calendar, residual_calendar,
)


def fixture(day="2026-06-18", *, history_days=3):
    start = (pd.Timestamp(day)-pd.Timedelta(days=history_days)).tz_localize("Europe/Paris")
    end = (pd.Timestamp(day)+pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    index = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")
    prices = pd.concat([pd.DataFrame({"timestamp_utc": index, "zone": zone,
                          "actual_15m": np.arange(len(index), dtype=float)*.13+i})
                        for i, zone in enumerate(ZONES)], ignore_index=True)
    hours = index[::4]
    baseline = pd.concat([pd.DataFrame({"timestamp_utc": hours, "zone": zone,
        "actual": 50., "nyx_q50": 51.,
        **{f"feature_hourly_{a}": (np.arange(len(hours), dtype=float)+i+1)*.017
           for i, a in enumerate(HOURLY_ALIASES)}}) for zone in ZONES], ignore_index=True)
    return prices, baseline


@pytest.mark.parametrize("day,horizon", [
    ("2026-06-18", 96), ("2026-03-29", 92), ("2025-10-26", 100),
])
def test_matched_physical_context_and_production_schema(day, horizon):
    inputs = FullChainInputs(*fixture(day))
    qc, qf, qa = inputs.build_origin(day, "15min", context_hours=24)
    hc, hf, ha = inputs.build_origin(day, "h", context_hours=24)
    for quarter, hourly, futureq, futureh in zip(qc, hc, qf, hf):
        assert len(quarter) == 96 and len(hourly) == 24
        assert len(futureq) == horizon and len(futureh) == horizon//4
        assert list(quarter.columns) == ["item_id", "timestamp", "target", *CONTEXT_COLUMNS]
        assert list(futureq.columns) == ["item_id", "timestamp", *FUTURE_COLUMNS]
        assert len(CONTEXT_COLUMNS) == 19 and len(FUTURE_COLUMNS) == 13
        assert quarter.timestamp.iloc[0] == hourly.timestamp.iloc[0]
        assert quarter.timestamp.iloc[-1]+pd.Timedelta(minutes=15) == futureq.timestamp.iloc[0]
        assert hourly.timestamp.iloc[-1]+pd.Timedelta(hours=1) == futureh.timestamp.iloc[0]
        assert quarter.timestamp.dt.tz is None
        np.testing.assert_allclose(quarter.target.to_numpy().reshape(-1, 4).mean(axis=1),
                                   hourly.target, rtol=1e-6)
        assert all(dtype == np.dtype("float32") for dtype in quarter.dtypes.iloc[2:])
        assert all(dtype == np.dtype("float32") for dtype in futureq.dtypes.iloc[2:])
        for alias in HOURLY_ALIASES:
            assert alias not in futureq
            np.testing.assert_array_equal(quarter[alias], quarter[f"known_{alias}_oracle"])
            # Forecast fundamentals stay hourly; prices have genuine quarter-hour changes.
            assert futureq[f"known_{alias}_oracle"].groupby(futureq.timestamp.dt.floor("h")).nunique().eq(1).all()
    assert all(a["future_realized_target_used"] is False for a in qa+ha)
    assert all(a["publication_vintage_verified"] is False for a in qa+ha)
    assert all(a["context_hours"] == 24 for a in qa+ha)
    assert all(a["last_price_delivery_day"] == (pd.Timestamp(day)-pd.Timedelta(days=1)).date().isoformat()
               for a in qa+ha)


@pytest.mark.parametrize("day", ["2026-01-01", "2026-03-29", "2025-10-26"])
def test_hourly_known_and_residual_calendars_equal_production(day):
    index = day_index(day, "h")
    for timezone in ZONE_TIMEZONES.values():
        expected = calendar_frame(index.tz_convert(timezone))
        expected.index = index
        pd.testing.assert_frame_equal(calendar(index, timezone=timezone), expected)
        pd.testing.assert_frame_equal(residual_calendar(index, frequency="h", timezone=timezone),
                                       build_calendar_features(index, timezone=timezone))


def test_fractional_calendar_keeps_distinct_year_definitions_and_dst_fold():
    index = day_index("2025-10-26", "15min")
    known = calendar(index)
    residual = residual_calendar(index, frequency="15min")
    assert "known_utc_offset_hours" not in known
    assert known.known_hour_sin.iloc[0] != known.known_hour_sin.iloc[1]
    assert residual.calendar_local_hour.iloc[1] == .25
    assert set(residual.calendar_utc_offset_hours) == {1., 2.}
    assert residual.calendar_dst_fold.sum() == 4
    assert set(residual.calendar_is_dst) == {0, 1}
    doy = index.tz_convert("Europe/Paris").dayofyear[0]
    assert known.known_doy_sin.iloc[0] == np.float32(np.sin(2*np.pi*doy/365.25))
    assert residual.calendar_dayofyear_sin.iloc[0] == np.sin(2*np.pi*(doy-1)/365.2425)


def test_hourly_frames_equal_actual_production_frame_builder():
    prices, baseline = fixture()
    inputs = FullChainInputs(prices, baseline)
    actual_contexts, actual_futures, _ = inputs.build_origin("2026-06-18", "h", context_hours=24)
    for zone, received_context, received_future in zip(ZONES, actual_contexts, actual_futures):
        timezone = ZONE_TIMEZONES[zone]
        target = inputs.hours[zone].copy()
        target.index = target.index.tz_convert(timezone)
        index = target.index
        panel = baseline.loc[baseline.zone.eq(zone)].copy()
        base = panel[[f"feature_hourly_{a}" for a in HOURLY_ALIASES]].copy()
        base.columns = list(HOURLY_ALIASES)
        base.index = index
        model = pd.concat([base, calendar_frame(index), base.rename(columns={a: f"known_{a}_oracle" for a in HOURLY_ALIASES})], axis=1).astype(np.float32)
        data = SimpleNamespace(target=target, model_context_covariates=model,
                               known_future_columns=list(FUTURE_COLUMNS), covariates=base, zone=zone)
        origin = index.get_loc(day_index("2026-06-18", "h")[0])
        expected_context, expected_future, _ = build_origin_frames(
            data, origin, 24, 24, f"{zone}_2026-06-18_h", True)
        pd.testing.assert_frame_equal(received_context, expected_context)
        pd.testing.assert_frame_equal(received_future, expected_future)


def test_price_changes_cannot_change_residual_features_or_delivery_day_inputs():
    prices, baseline = fixture()
    source_copy, baseline_copy = prices.copy(deep=True), baseline.copy(deep=True)
    inputs = FullChainInputs(prices, baseline)
    changed = prices.copy()
    changed.loc[changed.timestamp_utc.ge(day_index("2026-06-18", "15min")[0]), "actual_15m"] += 100000
    second = FullChainInputs(changed, baseline)
    for frequency in ("h", "15min"):
        before, after = (obj.build_origin("2026-06-18", frequency, context_hours=24) for obj in (inputs, second))
        for side in (0, 1):
            for first, last in zip(before[side], after[side]):
                pd.testing.assert_frame_equal(first, last)
        index = day_index("2026-06-18", frequency)
        for zone in ZONES:
            first = inputs.residual_features(index, zone, frequency)
            last = second.residual_features(index, zone, frequency)
            pd.testing.assert_frame_equal(first, last)
            assert len(first.columns) == 25 and set(ORACLE_COLUMNS).issubset(first)
            assert not any("price" in c or c in {"actual", "target"} for c in first)
    pd.testing.assert_frame_equal(prices, source_copy)
    pd.testing.assert_frame_equal(baseline, baseline_copy)


def test_omitted_price_features_have_identical_production_selected_features():
    inputs = FullChainInputs(*fixture())
    index = day_index("2026-06-18", "h")
    reduced = inputs.residual_features(index, "FR", "h")
    complete = reduced.copy()
    # The effective hourly config adds four lags and twelve rolling columns.
    for lag in (24, 48, 168, 336):
        complete[f"price_lag_{lag}h"] = np.nan
    for window in (24, 72, 168):
        for statistic in ("mean", "std", "min", "max"):
            complete[f"price_rolling_{statistic}_{window}h"] = np.nan
    assert len(complete.columns) == 41
    experts = pd.DataFrame({f"{name}__{q}": np.arange(24)+value
                           for name in ("base", "chronos2")
                           for q, value in (("q10", 20), ("q50", 30), ("q90", 45))}, index=index)
    builder = ResidualMetaFeatureBuilder(include_rich_calendar=True)
    expected, excluded = builder._build(complete, experts)
    actual, _ = builder._build(reduced, experts)
    assert len(excluded) == 20
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("defect", ["gap", "duplicate", "nan", "naive", "missing_zone"])
def test_bad_native_panel_refused(defect):
    prices, baseline = fixture()
    if defect == "gap": prices = prices.drop(index=1)
    if defect == "duplicate": prices = pd.concat([prices, prices.iloc[[0]]])
    if defect == "nan": prices.loc[0, "actual_15m"] = np.nan
    if defect == "naive": prices.timestamp_utc = prices.timestamp_utc.dt.tz_localize(None)
    if defect == "missing_zone": prices = prices.loc[prices.zone.ne("DE")]
    with pytest.raises(ValueError): FullChainInputs(prices, baseline)


def test_missing_covariate_or_incomplete_requested_grid_refused_without_fill():
    prices, baseline = fixture()
    missing = baseline.loc[baseline.timestamp_utc.ne(day_index("2026-06-18", "h")[0])]
    inputs = FullChainInputs(prices, missing)
    with pytest.raises(ValueError, match="fundamentals do not cover"):
        inputs.build_origin("2026-06-18", "15min", context_hours=24)
    inputs = FullChainInputs(prices, baseline)
    with pytest.raises(ValueError, match="contiguous"):
        inputs.residual_features(day_index("2026-06-18", "15min").delete(1), "FR", "15min")
    with pytest.raises(ValueError, match="Unknown NYX zone"):
        inputs.residual_features(day_index("2026-06-18", "h"), "ES", "h")


def test_first_real_origin_requires_all_2048_physical_hours():
    prices, baseline = fixture("2025-12-26", history_days=86)  # native archive starts 01/10
    assert prices.timestamp_utc.min().tz_convert("Europe/Paris").date().isoformat() == "2025-10-01"
    inputs = FullChainInputs(prices, baseline)
    for frequency, points in (("h", 2048), ("15min", 8192)):
        with pytest.raises(ValueError, match="insufficient genuine history"):
            inputs.build_origin("2025-12-25", frequency)
        contexts, _, audit = inputs.build_origin("2025-12-26", frequency)
        assert all(len(frame) == points for frame in contexts)
        assert all(row["context_hours"] == 2048 for row in audit)


def test_evaluation_baseline_inherited_and_relabelled_only_from_native_means():
    prices, baseline = fixture()
    inputs = FullChainInputs(prices, baseline)
    frame, audit = inputs.evaluation_baseline(baseline, "2026-06-18", "2026-06-18")
    assert len(frame) == 96 and frame.archived_hourly_actual.eq(50).all()
    for row in frame.itertuples():
        assert row.actual == inputs.hours[row.zone].loc[row.timestamp_utc]
    assert audit["scoring_target"] == "arithmetic_mean_of_four_native_quarter_hour_prices"

