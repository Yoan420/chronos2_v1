"""Fast synthetic-only tests for the three isolated scarcity ablations.

No test reads the real parent forecasts, refreshes a source, or launches the
historical runner. Parent frames, probabilities and model code remain untouched.
"""

import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import solar_wind_scarcity_ablation as subject
from chronos2_hourly import solar_wind_scarcity_regime as parent


PEAK_COLUMNS = [
    "own_daily_peak_residual_stress", "own_daily_peak_deficit_stress",
    "other_daily_peak_residual_stress", "other_daily_peak_deficit_stress",
]
TIMEZONE = "Europe/Berlin"
ORIGIN = pd.Timestamp("2026-06-01").date()


def training_data():
    days = np.repeat(pd.date_range(pd.Timestamp(ORIGIN) - pd.Timedelta(days=100), periods=100).date, 4)
    row = np.arange(len(days))
    stress = (row % 13) / 12.0
    features = pd.DataFrame({
        "own_residual_stress": stress,
        "other_residual_stress": np.cos(row / 13.0),
        "country_is_nl": (row % 2).astype(float),
        "own_rl_ramp_next1": np.sin(row / 17.0),
        **{column: 1.0 + stress for column in PEAK_COLUMNS},
    })
    residual = np.where(stress > 0.7, 90.0 + row % 23, -15.0 + row % 19)
    return features, residual, days


def calibration_inputs(*, historical_days=120):
    first = pd.Timestamp(ORIGIN) - pd.Timedelta(days=historical_days)
    physical = pd.date_range(first.tz_localize(TIMEZONE), pd.Timestamp(ORIGIN).tz_localize(TIMEZONE),
                             freq="h", inclusive="left").tz_convert("UTC")
    timestamps = physical.repeat(2)
    zone = np.tile(["DE", "NL"], len(physical))
    hour = timestamps.tz_convert(TIMEZONE).hour.to_numpy()
    day_number = np.asarray([(day - first.date()).days for day in timestamps.tz_convert(TIMEZONE).date])
    probability = 0.03 + 0.5 * hour / 23.0 + 0.1 * (zone == "NL")
    positive = ((hour >= 18) & (day_number % 3 != 0)) | ((zone == "NL") & (hour == 14))
    history = pd.DataFrame({
        "timestamp_utc": timestamps, "zone": zone,
        "fit_origin": timestamps.tz_convert(TIMEZONE).date,
        "spike_probability": probability,
        "residual": np.where(positive, 90.0 + hour, -10.0 + hour),
    })
    test_hours = pd.date_range(pd.Timestamp(ORIGIN).tz_localize(TIMEZONE), periods=4, freq="h").tz_convert("UTC")
    test = pd.DataFrame({"country_is_nl": np.tile([0.0, 1.0], len(test_hours)),
                         "own_residual_stress": np.linspace(0.0, 2.0, 8)},
                        index=test_hours.repeat(2))
    keys = pd.MultiIndex.from_arrays([test.index, np.where(test.country_is_nl, "NL", "DE")],
                                     names=["timestamp_utc", "zone"])
    parent_probability = pd.Series(np.linspace(0.02, 0.72, len(test)), index=keys, name="spike_probability")
    origins = pd.Series([ORIGIN] * len(test), index=keys, name="fit_origin")
    return history, test, parent_probability, origins


def recalibrate(history, test, probability, origins):
    return subject.recalibrate_probability(
        probability, test, history, origin_day=ORIGIN, parent_prediction_origin=origins,
    )


def tiny_variant(variant, features, residual, days):
    return subject.fit_variant(variant, features, residual, features.iloc[-8:].copy(), days,
                               origin_day=ORIGIN, threads=1, iterations=3, seed=42)


def test_hour_local_ablation_drops_exactly_four_peak_features_and_preserves_parent():
    features, _, _ = training_data()
    before = features.copy(deep=True)
    result = subject.select_hour_local_features(features)
    assert list(result) == [column for column in features if column not in PEAK_COLUMNS]
    pd.testing.assert_frame_equal(result, features.drop(columns=PEAK_COLUMNS))
    pd.testing.assert_frame_equal(features, before)
    result.iloc[0, 0] += 999.0
    pd.testing.assert_frame_equal(features, before)


def test_hour_local_variant_delegates_unchanged_parent_recipe_after_only_four_drops(monkeypatch):
    features, residual, days = training_data()
    before = features.copy(deep=True)
    captured = {}

    def spy(train_x, train_residual, test_x, train_days, **kwargs):
        captured.update(train=train_x.copy(), residual=np.asarray(train_residual).copy(),
                        test=test_x.copy(), days=np.asarray(train_days).copy(), kwargs=kwargs)
        return np.zeros((len(test_x), 3)), np.full(len(test_x), 0.1), {"synthetic_parent_spy": True}

    monkeypatch.setattr(parent, "fit_predict", spy)
    quantiles, probability, _ = tiny_variant("regime_hour_local", features, residual, days)
    pd.testing.assert_frame_equal(captured["train"], features.drop(columns=PEAK_COLUMNS))
    pd.testing.assert_frame_equal(captured["test"], features.drop(columns=PEAK_COLUMNS).iloc[-8:])
    np.testing.assert_array_equal(captured["residual"], residual)
    np.testing.assert_array_equal(captured["days"], days)
    assert captured["kwargs"] == {"origin_day": ORIGIN, "threads": 1, "iterations": 3, "seed": 42}
    np.testing.assert_array_equal(quantiles, np.zeros((8, 3)))
    np.testing.assert_array_equal(probability, np.full(8, 0.1))
    pd.testing.assert_frame_equal(features, before)


@pytest.mark.parametrize("missing", PEAK_COLUMNS)
def test_hour_local_refuses_an_incomplete_parent_peak_schema(missing):
    features, _, _ = training_data()
    with pytest.raises(ValueError):
        subject.select_hour_local_features(features.drop(columns=missing))


@pytest.mark.parametrize("residual_value", [-120.0, 120.0])
def test_direct_quantiles_have_no_artificial_positive_or_negative_cap(residual_value):
    features, residual, days = training_data()
    residual[:] = residual_value
    before = features.copy(deep=True)
    quantiles, probability, audit = tiny_variant("direct_quantile", features, residual, days)
    assert quantiles.shape == (8, 3)
    np.testing.assert_allclose(quantiles, residual_value, atol=1e-7, rtol=0)
    assert probability is None
    pd.testing.assert_frame_equal(features, before)
    json.dumps(audit, allow_nan=False)


def test_direct_quantiles_use_the_whole_residual_distribution_without_regime_projection():
    features, residual, days = training_data()
    # Constant design triggers the documented empirical fallback, making the
    # expected quantiles exact rather than relying on a three-tree fit's skill.
    features.iloc[:, :] = 0.0
    residual[:] = np.linspace(-180.0, 240.0, len(residual))
    quantiles, probability, _ = tiny_variant("direct_quantile", features, residual, days)
    expected = np.quantile(residual, [0.1, 0.5, 0.9])
    np.testing.assert_allclose(quantiles, np.repeat(expected[None, :], 8, axis=0), atol=1e-7, rtol=0)
    assert quantiles[0, 0] < -40 and quantiles[0, 2] > 50
    assert probability is None


def test_direct_tiny_catboost_fit_returns_finite_ordered_residual_quantiles():
    features, residual, days = training_data()
    quantiles, probability, _ = tiny_variant("direct_quantile", features, residual, days)
    assert quantiles.shape == (8, 3)
    assert np.isfinite(quantiles).all()
    assert (np.diff(quantiles, axis=1) >= 0).all()
    assert probability is None


@pytest.mark.parametrize("variant", ["direct_quantile", "regime_hour_local"])
@pytest.mark.parametrize("offset", [0, 1, -366])
def test_variants_reject_origin_future_and_out_of_window_labels(variant, offset):
    features, residual, days = training_data()
    days[-1] = (pd.Timestamp(ORIGIN) + pd.Timedelta(days=offset)).date()
    with pytest.raises(ValueError):
        tiny_variant(variant, features, residual, days)


def test_calibration_uses_only_trailing_ninety_common_days_and_never_mutates_parent():
    history, test, probability, origins = calibration_inputs()
    before = [frame.copy(deep=True) for frame in (history, test, probability, origins)]
    result, audit = recalibrate(history, test, probability, origins)
    first = pd.Timestamp(ORIGIN) - pd.Timedelta(days=90)
    old = pd.DatetimeIndex(history.timestamp_utc).tz_convert(TIMEZONE).date < first.date()
    changed = history.copy(deep=True)
    changed.loc[old, "residual"] = 9999.0
    changed.loc[old, "spike_probability"] = 0.999
    mutated, _ = recalibrate(changed, test, probability, origins)
    trailing, _ = recalibrate(history.loc[~old].copy(), test, probability, origins)
    np.testing.assert_allclose(result, mutated, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result, trailing, rtol=0, atol=1e-12)
    assert np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()
    assert audit["method"] == "logistic_oof90"
    assert audit["window_days"] == 90
    assert audit["first_day"] == first.date().isoformat()
    assert audit["last_day"] == (pd.Timestamp(ORIGIN) - pd.Timedelta(days=1)).date().isoformat()
    assert audit["rows"] == audit["expected_rows"] == 90 * 24 * 2 - 2  # Spring DST, both countries.
    assert audit["complete_two_country_panel"] is True
    assert audit["class_weights_used"] is False and audit["C"] == 0.1
    assert audit["slope"] > 0 and np.isfinite(audit["country_offset"])
    pd.testing.assert_frame_equal(history, before[0])
    pd.testing.assert_frame_equal(test, before[1])
    pd.testing.assert_series_equal(probability, before[2])
    pd.testing.assert_series_equal(origins, before[3])
    json.dumps(audit, allow_nan=False)


def test_calibration_preserves_country_and_row_identity_when_test_order_changes():
    history, test, probability, origins = calibration_inputs()
    threshold = np.where(history.zone.eq("DE"), 0.25, 0.55)
    history["residual"] = np.where(history.spike_probability > threshold, 100.0, 0.0)
    probability[:] = 0.4
    expected, _ = recalibrate(history, test, probability, origins)
    assert (expected[test.country_is_nl.to_numpy() == 0] > expected[test.country_is_nl.to_numpy() == 1]).all()
    permutation = np.array([7, 0, 5, 2, 3, 6, 1, 4])
    reordered, _ = recalibrate(history, test.iloc[permutation], probability.iloc[permutation], origins.iloc[permutation])
    np.testing.assert_allclose(reordered, expected[permutation], atol=1e-12, rtol=0)


@pytest.mark.parametrize("fault", ["future_label", "origin_label", "future_fit_origin",
                                   "duplicate_zone_hour", "invalid_zone", "nan_probability",
                                   "large_probability", "negative_probability", "infinite_residual"])
def test_invalid_calibration_labels_or_provenance_fail_closed(fault):
    history, test, probability, origins = calibration_inputs()
    if fault in {"future_label", "origin_label"}:
        row = history.iloc[[-1]].copy()
        extra_day = pd.Timestamp(ORIGIN) + pd.Timedelta(days=int(fault == "future_label"))
        row["timestamp_utc"] = extra_day.tz_localize(TIMEZONE).tz_convert("UTC")
        row["fit_origin"] = extra_day.date()
        history = pd.concat([history, row], ignore_index=True)
    elif fault == "future_fit_origin":
        history.loc[history.index[-1], "fit_origin"] = ORIGIN
    elif fault == "duplicate_zone_hour":
        history = pd.concat([history, history.iloc[[-1]]], ignore_index=True)
    elif fault == "invalid_zone":
        history.loc[history.index[-1], "zone"] = "FR"
    elif fault == "nan_probability":
        history.loc[history.index[-1], "spike_probability"] = np.nan
    elif fault == "large_probability":
        history.loc[history.index[-1], "spike_probability"] = 1.1
    elif fault == "negative_probability":
        history.loc[history.index[-1], "spike_probability"] = -0.1
    else:
        history.loc[history.index[-1], "residual"] = np.inf
    with pytest.raises(ValueError):
        recalibrate(history, test, probability, origins)


@pytest.mark.parametrize("fault", ["probability_order", "probability_array", "origin_order",
                                   "wrong_origin", "wrong_country", "duplicate_key"])
def test_parent_probability_rows_and_origins_require_exact_country_timestamp_alignment(fault):
    history, test, probability, origins = calibration_inputs()
    if fault == "probability_order":
        probability = probability.iloc[::-1]
    elif fault == "probability_array":
        probability = probability.to_numpy()
    elif fault == "origin_order":
        origins = origins.iloc[::-1]
    elif fault == "wrong_origin":
        origins.iloc[0] = (pd.Timestamp(ORIGIN) - pd.Timedelta(days=7)).date()
    elif fault == "wrong_country":
        test.iloc[0, test.columns.get_loc("country_is_nl")] = 1.0
    else:
        probability.index = pd.MultiIndex.from_tuples([probability.index[0]] * len(probability), names=probability.index.names)
    with pytest.raises(ValueError):
        recalibrate(history, test, probability, origins)


def test_sparse_positive_calibration_keeps_original_nonzero_probabilities():
    history, test, probability, origins = calibration_inputs()
    history["residual"] = 0.0
    history.loc[history.index[-3:], "residual"] = 120.0
    result, audit = recalibrate(history, test, probability, origins)
    np.testing.assert_array_equal(result, probability.to_numpy())
    assert (result > 0).all()
    json.dumps(audit, allow_nan=False)


def test_sparse_positive_support_in_one_country_is_not_hidden_by_pooling():
    history, test, probability, origins = calibration_inputs()
    history["residual"] = 0.0
    history.loc[history.loc[history.zone == "DE"].index[-32:], "residual"] = 120.0
    history.loc[history.loc[history.zone == "NL"].index[-4:], "residual"] = 120.0
    result, audit = recalibrate(history, test, probability, origins)
    np.testing.assert_array_equal(result, probability.to_numpy())
    assert audit["method"] == "identity"
    assert audit["fallback_reason"]


def test_nonpositive_calibration_slope_falls_back_instead_of_reversing_parent_risk():
    history, test, probability, origins = calibration_inputs()
    history["residual"] = np.where(history.spike_probability < 0.30, 120.0, 0.0)
    result, audit = recalibrate(history, test, probability, origins)
    np.testing.assert_array_equal(result, probability.to_numpy())
    assert audit["method"] == "identity"
    assert audit["fallback_reason"]


@pytest.mark.parametrize("fault", ["one_missing_hour", "missing_country", "short_history"])
def test_incomplete_ninety_day_common_calibration_support_falls_back_without_imputation(fault):
    history, test, probability, origins = calibration_inputs()
    if fault == "one_missing_hour":
        history = history.drop(history.index[-1])
    elif fault == "missing_country":
        history = history.loc[history.zone == "DE"].copy()
    else:
        cutoff = (pd.Timestamp(ORIGIN) - pd.Timedelta(days=89)).tz_localize(TIMEZONE).tz_convert("UTC")
        history = history.loc[history.timestamp_utc >= cutoff].copy()
    result, audit = recalibrate(history, test, probability, origins)
    np.testing.assert_array_equal(result, probability.to_numpy())
    json.dumps(audit, allow_nan=False)


def test_calibration_is_numerically_finite_for_parent_probabilities_zero_and_one():
    history, test, probability, origins = calibration_inputs()
    probability.iloc[0] = 0.0
    probability.iloc[-1] = 1.0
    result, _ = recalibrate(history, test, probability, origins)
    assert np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()


def synthetic_parent_experts(monkeypatch):
    """Known conditional distributions test the variant without historical fits."""
    history, test_identity, probability, origins = calibration_inputs()
    features, residual, days = training_data()
    test = features.iloc[-len(test_identity):].copy()
    test.index = test_identity.index
    test["country_is_nl"] = test_identity.country_is_nl.to_numpy()
    normal = np.repeat((-10.0 + 20.0 * parent.EXPERT_PROBABILITIES)[None, :], len(test), axis=0)
    spike = np.repeat((80.0 + 100.0 * parent.EXPERT_PROBABILITIES)[None, :], len(test), axis=0)
    captured = {}

    def expert(train, labels, prediction, regime, parameters):
        captured[regime] = np.asarray(labels).copy()
        assert parameters["iterations"] == 3 and parameters["thread_count"] == 1
        assert len(prediction) == len(test)
        return (normal if regime == "normal" else spike).copy(), {"synthetic_expert": regime}

    def forbidden_gate_refit(*args, **kwargs):
        pytest.fail("The calibration ablation must not refit the original gate")

    monkeypatch.setattr(parent, "_expert", expert)
    monkeypatch.setattr(parent, "fit_predict", forbidden_gate_refit)
    parent_quantiles = pd.DataFrame(parent.mixture_quantiles(normal, spike, probability.to_numpy()),
                                    index=probability.index, columns=["q10", "q50", "q90"])
    arguments = dict(variant="regime_calibration_oof90", train_X=features,
                     train_residual=residual, test_X=test, train_days=days, origin_day=ORIGIN,
                     parent_probability=probability, parent_prediction_origin=origins,
                     parent_residual_quantiles=parent_quantiles, calibration_frame=history,
                     threads=1, iterations=3, seed=42)
    return arguments, normal, spike, captured


def test_calibration_variant_reproduces_parent_then_inverts_same_experts_with_new_probability(monkeypatch):
    arguments, normal, spike, captured = synthetic_parent_experts(monkeypatch)
    before = {name: arguments[name].copy(deep=True) for name in
              ("train_X", "test_X", "parent_probability", "parent_prediction_origin",
               "parent_residual_quantiles", "calibration_frame")}
    quantiles, probability, audit = subject.fit_variant(**arguments)
    expected_probability, _ = recalibrate(arguments["calibration_frame"], arguments["test_X"],
                                          arguments["parent_probability"], arguments["parent_prediction_origin"])
    np.testing.assert_allclose(probability, expected_probability, rtol=0, atol=1e-12)
    np.testing.assert_allclose(quantiles, parent.mixture_quantiles(normal, spike, probability), atol=1e-9, rtol=0)
    assert (np.diff(quantiles, axis=1) >= 0).all()
    assert np.isfinite(quantiles).all() and (quantiles[:, 2] > 40).any()
    assert audit["parent_reproduction_max_absolute_difference"] == 0.0
    assert audit["current_gate_refit_performed"] is False
    np.testing.assert_array_equal(captured["normal"], arguments["train_residual"][arguments["train_residual"] <= 50.0])
    np.testing.assert_array_equal(captured["spike"], arguments["train_residual"][arguments["train_residual"] > 50.0])
    for name, original in before.items():
        if isinstance(original, pd.Series):
            pd.testing.assert_series_equal(arguments[name], original)
        else:
            pd.testing.assert_frame_equal(arguments[name], original)


@pytest.mark.parametrize("fault", ["different_values", "row_order", "column_order", "crossed", "nan"])
def test_calibration_variant_rejects_unreproduced_or_unaligned_parent_quantiles_before_calibration(monkeypatch, fault):
    arguments, _, _, _ = synthetic_parent_experts(monkeypatch)
    quantiles = arguments["parent_residual_quantiles"].copy()
    if fault == "different_values":
        quantiles += 1.0
    elif fault == "row_order":
        quantiles = quantiles.iloc[::-1]
    elif fault == "column_order":
        quantiles = quantiles[["q50", "q10", "q90"]]
    elif fault == "crossed":
        quantiles.iloc[0, 0] = quantiles.iloc[0, 2] + 1.0
    else:
        quantiles.iloc[0, 0] = np.nan
    arguments["parent_residual_quantiles"] = quantiles

    def forbidden_calibration(*args, **kwargs):
        pytest.fail("Parent quantile identity/reproduction must be checked before recalibration")

    monkeypatch.setattr(subject, "recalibrate_probability", forbidden_calibration)
    with pytest.raises(ValueError):
        subject.fit_variant(**arguments)
