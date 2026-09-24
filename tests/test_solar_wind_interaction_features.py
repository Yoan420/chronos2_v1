import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.solar_wind_interaction_features import (
    MINIMUM_HISTORY_DAYS,
    PROTOCOL_VERSION,
    SolarWindInteractionFeatureError,
    build_interaction,
)


def panel(*, start="2026-01-01", days=20, zone="DE", timezone="Europe/Berlin"):
    first = pd.Timestamp(start)
    index = pd.date_range(first.tz_localize(timezone),
                          (first + pd.Timedelta(days=days)).tz_localize(timezone),
                          freq="h", inclusive="left").tz_convert("UTC")
    hour = index.tz_convert(timezone).hour.to_numpy()
    return pd.DataFrame({f"{zone.lower()}_wind_generation_fcst": 1.0 + hour / 3,
                         f"{zone.lower()}_solar_generation_fcst": np.where((hour >= 8) & (hour <= 17), 6., 0.),
                         f"{zone.lower()}_residual_load_fcst": 10.0 + hour}, index=index)


@pytest.mark.parametrize("zone,timezone", [("DE", "Europe/Berlin"), ("NL", "Europe/Amsterdam")])
def test_formula_single_column_no_input_mutation_and_serializable_audit(zone, timezone):
    source = panel(zone=zone, timezone=timezone)
    source["actual"] = "unused and intentionally not numeric"
    before = source.copy(deep=True)
    result, audit = build_interaction(source, zone, timezone)
    pd.testing.assert_frame_equal(source, before)
    assert list(result) == [f"{zone.lower()}_low_wind_solar_stress"]
    pd.testing.assert_index_equal(result.index, source.index)
    assert result.iloc[:14 * 24, 0].eq(0).all()
    train = source.iloc[:14 * 24]
    current = source.iloc[14 * 24:15 * 24]
    wind, solar, residual = list(source)[:3]
    w75 = np.quantile(train[wind], .75)
    s75 = np.quantile(train.loc[train[solar] > 0, solar], .75)
    r50, r90 = np.quantile(train[residual], [.5, .9])
    expected = ((1-current[wind]/w75).clip(0, 1) * (1-current[solar]/s75).clip(0, 1)
                * ((current[residual]-r50)/(r90-r50)).clip(0, 1))
    np.testing.assert_allclose(result.iloc[14 * 24:15 * 24, 0], expected)
    assert result.iloc[:, 0].between(0, 1).all()
    assert audit["protocol_version"] == PROTOCOL_VERSION
    assert audit["warmup_days"] == MINIMUM_HISTORY_DAYS
    assert audit["active_days"] == 6
    assert audit["normalizations"][14]["solar_positive_q75_gw"] == 6.
    assert audit["normalizations"][14]["solar_scale_positive_sample_hours"] == 14 * 10
    assert audit["normalizations"][14]["wind_scale_sample_hours"] == 14 * 24
    assert "actual" not in audit["source_columns_read"]
    assert json.loads(json.dumps(audit, allow_nan=False)) == audit


def test_current_day_extremes_do_not_change_its_normalization_and_encode_conjunction():
    source = panel(days=15)
    changed = source.copy()
    # Same historical scales; only the forecast delivery profile changes.
    changed.iloc[-24:, :] = [0., 0., 1000.]
    changed.iloc[-23, 0] = 1000.  # abundant wind suppresses the score
    changed.iloc[-22, 1] = 1000.  # abundant solar suppresses the score
    changed.iloc[-21, 2] = -1000.  # low residual load suppresses the score
    result, audit = build_interaction(changed, "DE", "Europe/Berlin")
    _, original_audit = build_interaction(source, "DE", "Europe/Berlin")
    assert result.iloc[-24, 0] == 1.
    assert result.iloc[-23:-20, 0].eq(0).all()
    for key in ("wind_q75_gw", "solar_positive_q75_gw", "residual_q50_gw", "residual_q90_gw"):
        assert audit["normalizations"][-1][key] == original_audit["normalizations"][-1][key]


def test_prefix_invariance_on_future_perturbation_and_append():
    source = panel(days=22)
    original, original_audit = build_interaction(source, "DE", "Europe/Berlin")
    prefix = source.iloc[:18 * 24]
    shorter, shorter_audit = build_interaction(prefix, "DE", "Europe/Berlin")
    changed = source.copy()
    changed.iloc[18 * 24:, 0] *= 20
    changed.iloc[18 * 24:, 1] *= 10
    changed.iloc[18 * 24:, 2] += 1000
    mutated, mutated_audit = build_interaction(changed, "DE", "Europe/Berlin")
    pd.testing.assert_frame_equal(shorter, original.loc[prefix.index])
    pd.testing.assert_frame_equal(shorter, mutated.loc[prefix.index])
    assert shorter_audit["normalizations"] == original_audit["normalizations"][:18]
    assert shorter_audit["normalizations"] == mutated_audit["normalizations"][:18]


@pytest.mark.parametrize("start,target,hours", [
    ("2026-03-10", "2026-03-29", 23), ("2025-10-07", "2025-10-26", 25),
])
@pytest.mark.parametrize("zone,timezone", [("DE", "Europe/Berlin"), ("NL", "Europe/Amsterdam")])
def test_dst_preserves_physical_hours_and_civil_training_dates(start, target, hours, zone, timezone):
    source = panel(start=start, days=21, zone=zone, timezone=timezone)
    result, audit = build_interaction(source, zone, timezone)
    records = {record["delivery_day"]: record for record in audit["normalizations"]}
    assert records[target]["physical_hours"] == hours
    assert records[target]["normalization_training_complete_days"] == 19
    next_day = (pd.Timestamp(target) + pd.Timedelta(days=1)).date().isoformat()
    assert records[next_day]["normalization_training_last_day"] == target
    assert records[next_day]["normalization_training_hours"] == 19 * 24 + hours
    assert len(result) == len(source)
    assert result.index.is_unique


def test_rolling_window_excludes_366th_prior_day():
    source = panel(start="2025-01-01", days=367)
    baseline, audit = build_interaction(source, "DE", "Europe/Berlin")
    changed = source.copy()
    first_day = source.index.tz_convert("Europe/Berlin").date == pd.Timestamp("2025-01-01").date()
    changed.loc[first_day, :] *= 1000
    result, changed_audit = build_interaction(changed, "DE", "Europe/Berlin")
    last_day = source.index.tz_convert("Europe/Berlin").date == pd.Timestamp("2026-01-02").date()
    pd.testing.assert_frame_equal(result.loc[last_day], baseline.loc[last_day])
    assert audit["normalizations"][-1] == changed_audit["normalizations"][-1]
    assert audit["normalizations"][-1]["normalization_training_complete_days"] == 365
    assert audit["normalizations"][-1]["normalization_training_first_day"] == "2025-01-02"


def test_zero_generation_and_flat_residual_are_allowed_only_during_warmup():
    source = panel(days=14)
    source.iloc[:, :] = 0.
    result, audit = build_interaction(source, "DE", "Europe/Berlin")
    assert result.iloc[:, 0].eq(0).all()
    assert audit["active_days"] == 0
    assert all(record["status"] == "warmup_zero_score" for record in audit["normalizations"])
    assert all(record["wind_q75_gw"] is None for record in audit["normalizations"])


@pytest.mark.parametrize("invalid", ["wind_zero", "wind_mostly_zero", "solar_zero", "residual_flat"])
def test_invalid_denominators_fail_closed_after_warmup(invalid):
    source = panel(days=15)
    if invalid == "wind_zero":
        source.iloc[:, 0] = 0.
    elif invalid == "wind_mostly_zero":
        source.iloc[:, 0] = 0.
        source.iloc[::24, 0] = 10.  # positive-only wind filtering would incorrectly pass
    elif invalid == "solar_zero":
        source.iloc[:, 1] = 0.
    else:
        source.iloc[:, 2] = 20.
    with pytest.raises(SolarWindInteractionFeatureError, match="after warmup"):
        build_interaction(source, "DE", "Europe/Berlin")


@pytest.mark.parametrize("column,value", [(0, np.nan), (1, np.inf), (2, -np.inf), (0, -1.), (1, -1.)])
def test_invalid_selected_sources_fail_even_during_warmup(column, value):
    source = panel(days=2)
    source.iloc[0, column] = value
    with pytest.raises(SolarWindInteractionFeatureError):
        build_interaction(source, "DE", "Europe/Berlin")


@pytest.mark.parametrize("invalid", ["naive", "local_timezone", "unsorted", "duplicate_index", "half_hour",
                                      "missing_hour", "missing_day", "partial_first", "partial_last",
                                      "missing_column", "duplicate_column", "empty", "nonnumeric"])
def test_invalid_input_contract_is_rejected(invalid):
    source = panel(days=20)
    if invalid == "naive":
        source.index = source.index.tz_localize(None)
    elif invalid == "local_timezone":
        source.index = source.index.tz_convert("Europe/Berlin")
    elif invalid == "unsorted":
        source = source.iloc[::-1]
    elif invalid == "duplicate_index":
        source = pd.concat([source.iloc[:1], source])
    elif invalid == "half_hour":
        source.index += pd.Timedelta(minutes=30)
    elif invalid == "missing_hour":
        source = source.drop(source.index[30])
    elif invalid == "missing_day":
        source = source.drop(source.index[24:48])
    elif invalid == "partial_first":
        source = source.iloc[1:]
    elif invalid == "partial_last":
        source = source.iloc[:-1]
    elif invalid == "missing_column":
        source = source.iloc[:, :2]
    elif invalid == "duplicate_column":
        source = pd.concat([source, source.iloc[:, :1]], axis=1)
    elif invalid == "empty":
        source = source.iloc[:0]
    else:
        source = source.astype(object)
        source.iloc[0, 0] = "not a forecast"
    with pytest.raises(SolarWindInteractionFeatureError):
        build_interaction(source, "DE", "Europe/Berlin")


@pytest.mark.parametrize("zone,timezone", [("FR", "Europe/Paris"), (None, "Europe/Berlin"),
                                           ("DE", "not/a_timezone"), ("DE", None)])
def test_invalid_zone_or_timezone(zone, timezone):
    with pytest.raises(SolarWindInteractionFeatureError):
        build_interaction(panel(), zone, timezone)
