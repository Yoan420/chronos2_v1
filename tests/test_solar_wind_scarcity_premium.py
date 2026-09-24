"""Behavioral safeguards for the isolated causal scarcity premium experiment."""

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.solar_wind_scarcity_premium import (
    BASIS_COLUMNS,
    build_scarcity_basis,
    fit_nonnegative_premium,
    predict_premium,
    run_premium_backtest,
)


TIMEZONE = "Europe/Berlin"
BASE_QUANTILES = tuple(f"residual_kalman__q{q}" for q in (10, 50, 90))
NEW_QUANTILES = tuple(f"scarcity__q{q}" for q in (10, 50, 90))


def panel(*, start="2026-01-01", days=27, zone="DE", timezone=TIMEZONE):
    first = pd.Timestamp(start)
    index = pd.date_range(
        first.tz_localize(timezone),
        (first + pd.Timedelta(days=days)).tz_localize(timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    hour = index.tz_convert(timezone).hour.to_numpy()
    elapsed_day = np.asarray([(d - first.date()).days for d in index.tz_convert(timezone).date])
    return pd.DataFrame(
        {
            f"{zone.lower()}_wind_generation_fcst": 1.0 + hour / 3 + elapsed_day % 3,
            f"{zone.lower()}_solar_generation_fcst": np.where((hour >= 8) & (hour <= 17), 6.0, 0.0),
            f"{zone.lower()}_residual_load_fcst": 10.0 + hour + elapsed_day % 5,
        },
        index=index,
    )


def day_mask(index, day, timezone=TIMEZONE):
    return index.tz_convert(timezone).date == pd.Timestamp(day).date()


def experiment_inputs(*, start="2026-01-01"):
    source = panel(start=start)
    features, _ = build_scarcity_basis(source, zone="DE", timezone=TIMEZONE)
    civil_days = sorted(set(source.index.tz_convert(TIMEZONE).date))
    historical_index = source.index[
        (source.index.tz_convert(TIMEZONE).date >= civil_days[14])
        & (source.index.tz_convert(TIMEZONE).date < civil_days[-1])
    ]
    forecast_index = source.index[source.index.tz_convert(TIMEZONE).date == civil_days[-1]]
    backtest = pd.DataFrame(
        {BASE_QUANTILES[0]: 20.0, BASE_QUANTILES[1]: 50.0, BASE_QUANTILES[2]: 80.0},
        index=historical_index,
    )
    backtest["actual"] = (
        50.0 + features.loc[historical_index, "scarcity_base"] * 8.0
        + features.loc[historical_index, "scarcity_demand"] * 5.0
    )
    forecast = pd.DataFrame(
        {BASE_QUANTILES[0]: 20.0, BASE_QUANTILES[1]: 50.0, BASE_QUANTILES[2]: 80.0},
        index=forecast_index,
    )
    return backtest, forecast, features


def run_small(backtest, forecast, features, **kwargs):
    options = dict(timezone=TIMEZONE, train_days=5, min_train_days=3, ridge=0.05, cap=40.0)
    options.update(kwargs)
    return run_premium_backtest(
        backtest, forecast, features, **options,
    )


@pytest.mark.parametrize("zone,timezone", [("DE", TIMEZONE), ("NL", "Europe/Amsterdam")])
def test_low_generation_gives_base_premium_even_below_residual_median(zone, timezone):
    source = panel(days=15, zone=zone, timezone=timezone)
    source.iloc[-24:, :] = [0.0, 0.0, -100.0]
    before = source.copy(deep=True)
    features, _ = build_scarcity_basis(source, zone=zone, timezone=timezone)
    pd.testing.assert_frame_equal(source, before)
    pd.testing.assert_index_equal(features.index, source.index)
    assert features.loc[:, list(BASIS_COLUMNS)].iloc[:14 * 24].eq(0).all().all()
    assert features["scarcity_base"].iloc[-24:].eq(1).all()
    assert features["scarcity_demand"].iloc[-24:].eq(0).all()
    assert features["scarcity_extreme"].iloc[-24:].eq(0).all()


def test_declining_generation_increases_premium_with_fixed_historical_scales():
    source = panel(days=15)
    source.iloc[-24:, :] = [0.0, 0.0, 100.0]
    # These hours share exactly the same past normalization and residual load.
    source.iloc[-24:-21, 0] = [3.0, 1.0, 0.0]
    source.iloc[-21:-18, 1] = [3.0, 1.0, 0.0]
    source.iloc[-18, 0] = 1000.0
    source.iloc[-17, 1] = 1000.0
    features, _ = build_scarcity_basis(source, zone="DE", timezone=TIMEZONE)
    premium = predict_premium(features.loc[:, list(BASIS_COLUMNS)].to_numpy(), np.array([4.0, 3.0, 2.0]))
    assert premium[-24] < premium[-23] < premium[-22]
    assert premium[-21] < premium[-20] < premium[-19]
    assert premium[-18] == 0.0
    assert premium[-17] == 0.0
    assert premium[-22] > 0.0


def test_extreme_residual_load_strengthens_the_same_joint_generation_deficit():
    source = panel(days=15)
    source.iloc[-24:, :] = [0.0, 0.0, -100.0]
    source.iloc[-23, 2] = 28.0
    source.iloc[-22, 2] = 1000.0
    features, _ = build_scarcity_basis(source, zone="DE", timezone=TIMEZONE)
    premium = predict_premium(features.loc[:, list(BASIS_COLUMNS)].to_numpy(), np.array([4.0, 3.0, 2.0]))
    assert 0.0 < premium[-24] < premium[-23] < premium[-22]
    assert features["scarcity_extreme"].iloc[-24] == 0.0
    assert features["scarcity_extreme"].iloc[-22] > 0.0


def test_future_feature_changes_and_appended_days_cannot_rewrite_past_features():
    source = panel(days=22)
    prefix = source.iloc[:18 * 24].copy()
    changed = source.copy()
    changed.iloc[18 * 24:, 0] *= 50.0
    changed.iloc[18 * 24:, 1] *= 50.0
    changed.iloc[18 * 24:, 2] += 1000.0
    original, _ = build_scarcity_basis(source, zone="DE", timezone=TIMEZONE)
    mutated, _ = build_scarcity_basis(changed, zone="DE", timezone=TIMEZONE)
    shorter, _ = build_scarcity_basis(prefix, zone="DE", timezone=TIMEZONE)
    pd.testing.assert_frame_equal(original.loc[prefix.index], mutated.loc[prefix.index])
    pd.testing.assert_frame_equal(original.loc[prefix.index], shorter)


@pytest.mark.parametrize("column,value", [(0, -1.0), (1, -1.0), (0, np.nan), (1, np.inf), (2, -np.inf)])
def test_invalid_generation_or_missing_selected_inputs_fail_closed(column, value):
    source = panel()
    source.iloc[0, column] = value
    with pytest.raises(ValueError):
        build_scarcity_basis(source, zone="DE", timezone=TIMEZONE)


@pytest.mark.parametrize("invalid", ["naive", "missing_hour", "duplicate", "partial_day", "missing_column", "missing_day"])
def test_incomplete_or_ambiguous_source_grids_are_rejected(invalid):
    source = panel()
    if invalid == "naive":
        source.index = source.index.tz_localize(None)
    elif invalid == "missing_hour":
        source = source.drop(source.index[30])
    elif invalid == "duplicate":
        source = pd.concat([source.iloc[:1], source])
    elif invalid == "partial_day":
        source = source.iloc[1:]
    elif invalid == "missing_column":
        source = source.iloc[:, :2]
    elif invalid == "missing_day":
        source = source.drop(source.index[24:48])
    with pytest.raises(ValueError):
        build_scarcity_basis(source, zone="DE", timezone=TIMEZONE)


def test_learning_can_select_no_premium_when_baseline_already_overpredicts():
    x = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 1.0]])
    coefficients = fit_nonnegative_premium(x, np.array([-10.0, -20.0, -30.0]), ridge=0.05)
    np.testing.assert_allclose(coefficients, 0.0, atol=1e-12)
    np.testing.assert_allclose(predict_premium(x, coefficients), 0.0, atol=1e-12)


def test_learning_positive_errors_produces_nonnegative_conditional_response():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 1.0]])
    coefficients = fit_nonnegative_premium(x, np.array([0.0, 10.0, 20.0, 35.0]), ridge=0.05)
    assert coefficients.shape == (3,)
    assert np.isfinite(coefficients).all()
    assert (coefficients >= 0).all()
    premium = predict_premium(x, coefficients)
    assert premium[0] == 0.0
    assert 0.0 < premium[1] <= premium[2] <= premium[3]


def test_premium_is_capped_without_creating_a_global_intercept():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    premium = predict_premium(x, np.array([10.0, 30.0, 50.0]), cap=40.0)
    np.testing.assert_allclose(premium, [0.0, 10.0, 40.0])


def test_uncapped_premium_preserves_learned_amplitudes_above_forty():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    premium = predict_premium(x, np.array([10.0, 30.0, 50.0]), cap=None)
    np.testing.assert_allclose(premium, [0.0, 10.0, 120.0])
    np.testing.assert_allclose(predict_premium(x, np.array([10.0, 30.0, 50.0])), premium)
    backtest, forecast, features = experiment_inputs()
    backtest["actual"] += 1000.0
    result = run_small(backtest, forecast, features, cap=None)
    assert result["backtest"]["scarcity_premium"].max() > 40.0
    assert result["forecast"]["scarcity_premium"].max() > 40.0


def test_daily_backtest_has_common_support_causal_training_and_parallel_quantile_shift():
    backtest, forecast, features = experiment_inputs()
    before = [f.copy(deep=True) for f in (backtest, forecast, features)]
    result = run_small(backtest, forecast, features)
    for original, previous in zip((backtest, forecast, features), before):
        pd.testing.assert_frame_equal(original, previous)
    expected_index = backtest.index[3 * 24:]
    pd.testing.assert_index_equal(result["backtest"].index, expected_index)
    pd.testing.assert_index_equal(result["forecast"].index, forecast.index)
    for key, original in (("backtest", backtest), ("forecast", forecast)):
        frame = result[key]
        for base, adjusted in zip(BASE_QUANTILES, NEW_QUANTILES):
            np.testing.assert_allclose(frame[base], original.loc[frame.index, base])
            np.testing.assert_allclose(frame[adjusted] - frame[base], frame["scarcity_premium"])
        assert frame["scarcity_premium"].between(0.0, 40.0).all()
        assert (frame[NEW_QUANTILES[0]] <= frame[NEW_QUANTILES[1]]).all()
        assert (frame[NEW_QUANTILES[1]] <= frame[NEW_QUANTILES[2]]).all()
    assert result["backtest"]["scarcity_premium"].max() > 0
    for audit in result["daily_audit"]:
        if audit.get("train_delivery_max") is not None:
            assert pd.Timestamp(audit["train_delivery_max"]) < pd.Timestamp(audit["delivery_day"])
        assert np.asarray(audit["coefficients"]).min() >= 0.0


def test_changing_labels_for_a_day_cannot_change_that_days_or_prior_predictions():
    backtest, forecast, features = experiment_inputs()
    civil_days = sorted(set(backtest.index.tz_convert(TIMEZONE).date))
    changed_day = civil_days[7]
    original = run_small(backtest, forecast, features)
    altered = backtest.copy()
    affected = altered.index.tz_convert(TIMEZONE).date >= changed_day
    altered.loc[affected, "actual"] += 10000.0
    mutated = run_small(altered, forecast, features)
    unchanged = original["backtest"].index.tz_convert(TIMEZONE).date <= changed_day
    columns = ["scarcity_premium", *NEW_QUANTILES]
    pd.testing.assert_frame_equal(
        original["backtest"].loc[unchanged, columns],
        mutated["backtest"].loc[unchanged, columns],
    )
    # Ensure the perturbation is effective once those labels become available.
    assert not np.allclose(original["forecast"]["scarcity_premium"], mutated["forecast"]["scarcity_premium"])


def test_labels_outside_rolling_training_window_do_not_change_forecast():
    backtest, forecast, features = experiment_inputs()
    original = run_small(backtest, forecast, features)
    altered = backtest.copy()
    cutoff = forecast.index[0].tz_convert(TIMEZONE).date() - pd.Timedelta(days=5)
    old = altered.index.tz_convert(TIMEZONE).date < cutoff
    assert old.any()
    altered.loc[old, "actual"] += 10000.0
    mutated = run_small(altered, forecast, features)
    pd.testing.assert_frame_equal(original["forecast"], mutated["forecast"])


@pytest.mark.parametrize("start,target,hours", [
    ("2026-03-10", "2026-03-29", 23), ("2025-10-07", "2025-10-26", 25),
])
def test_dst_days_keep_every_physical_hour_in_features_and_predictions(start, target, hours):
    backtest, forecast, features = experiment_inputs(start=start)
    result = run_small(backtest, forecast, features)
    assert day_mask(features.index, target).sum() == hours
    assert day_mask(result["backtest"].index, target).sum() == hours
    assert result["backtest"].index.is_unique
    pd.testing.assert_index_equal(result["forecast"].index, forecast.index)
    after = (pd.Timestamp(target) + pd.Timedelta(days=1)).date().isoformat()
    record = next(row for row in result["daily_audit"] if str(row["delivery_day"]) == after)
    assert str(record["train_delivery_max"]) == target


@pytest.mark.parametrize("invalid", ["missing_actual", "missing_feature", "overlap", "forecast_actual", "forecast_gap"])
def test_backtest_rejects_unavailable_labels_or_invalid_forecast_support(invalid):
    backtest, forecast, features = experiment_inputs()
    if invalid == "missing_actual":
        backtest.iloc[0, backtest.columns.get_loc("actual")] = np.nan
    elif invalid == "missing_feature":
        features = features.drop(backtest.index[0])
    elif invalid == "overlap":
        forecast.index = backtest.index[-24:]
    elif invalid == "forecast_actual":
        forecast["actual"] = 100.0
    elif invalid == "forecast_gap":
        forecast.index += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        run_small(backtest, forecast, features)
