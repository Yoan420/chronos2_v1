"""Synthetic-only safeguards for the isolated DE/NL scarcity-regime experiment.

These tests never read historical experiment inputs, contact a provider, or
write model artifacts. Small CatBoost fits use one thread and three iterations.
"""

import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import solar_wind_scarcity_regime as subject


TIMEZONES = {"DE": "Europe/Berlin", "NL": "Europe/Amsterdam"}


def panels(*, start="2026-01-01", days=23):
    """Complete physical-hour profiles with variable, strictly positive scales."""
    first = pd.Timestamp(start)
    output = {}
    for zone, timezone in TIMEZONES.items():
        index = pd.date_range(
            first.tz_localize(timezone),
            (first + pd.Timedelta(days=days)).tz_localize(timezone),
            freq="h", inclusive="left",
        ).tz_convert("UTC")
        hour = index.tz_convert(timezone).hour.to_numpy()
        day = np.asarray([(value - first.date()).days for value in index.tz_convert(timezone).date])
        scale = 1.0 if zone == "DE" else 0.6
        prefix = zone.lower()
        output[zone] = pd.DataFrame({
            f"{prefix}_wind_generation_fcst": scale * (2.0 + hour / 4.0 + day % 3),
            f"{prefix}_solar_generation_fcst": scale * np.maximum(0.0, 8.0 - np.abs(hour - 12)),
            f"{prefix}_residual_load_fcst": scale * (15.0 + hour + day % 5),
        }, index=index)
    return output


def local_mask(index, day, timezone="Europe/Berlin"):
    return index.tz_convert(timezone).date == pd.Timestamp(day).date()


def training_data(*, days=100):
    # Four pooled observations per day, two per zone. Keeping this small also
    # makes the chronological calibration holdout easy to inspect.
    dates = np.repeat(pd.date_range("2025-01-01", periods=days).date, 4)
    row = np.arange(len(dates))
    stress = (row % 13) / 12.0
    features = pd.DataFrame({"own_residual_stress": stress,
                             "zone_is_nl": (row % 2).astype(float),
                             "seasonal_signal": np.sin(row / 17.0)})
    residual = np.where(stress > 0.7, 90.0 + row % 23, -15.0 + row % 19)
    origin = pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)
    return features, residual, dates, origin.date()


def small_fit(features, residual, days, origin, *, test=None):
    return subject.fit_predict(
        features, residual, features.iloc[-8:].copy() if test is None else test,
        days, origin_day=origin, threads=1, iterations=3, seed=42,
    )


def test_features_pool_shared_schema_without_mutation_and_mark_warmup_missing():
    source = panels()
    before = {zone: frame.copy(deep=True) for zone, frame in source.items()}
    features, audit = subject.build_features(source)
    assert set(features) == {"DE", "NL"}
    assert list(features["DE"]) == list(features["NL"])
    for zone in source:
        pd.testing.assert_frame_equal(source[zone], before[zone])
        pd.testing.assert_index_equal(features[zone].index, source[zone].index)
        assert features[zone].iloc[:14 * 24].isna().all().all()
        assert np.isfinite(features[zone].iloc[14 * 24:].to_numpy(float)).all()
    json.dumps(audit, allow_nan=False)


def test_feature_prefix_is_invariant_to_appended_or_perturbed_future_days():
    source = panels()
    prefix = {zone: frame.iloc[:18 * 24].copy() for zone, frame in source.items()}
    changed = {zone: frame.copy(deep=True) for zone, frame in source.items()}
    for frame in changed.values():
        frame.iloc[18 * 24:, :2] *= 50.0
        frame.iloc[18 * 24:, 2] += 1000.0
    original, _ = subject.build_features(source)
    shortened, _ = subject.build_features(prefix)
    mutated, _ = subject.build_features(changed)
    for zone in source:
        pd.testing.assert_frame_equal(shortened[zone], original[zone].loc[prefix[zone].index])
        pd.testing.assert_frame_equal(shortened[zone], mutated[zone].loc[prefix[zone].index])


def test_forecast_profile_ramps_never_cross_into_the_next_delivery_day():
    source = panels(days=16)
    original, _ = subject.build_features(source)
    changed = {zone: frame.copy(deep=True) for zone, frame in source.items()}
    for frame in changed.values():
        frame.iloc[-24:, :] = [10000.0, 10000.0, 10000.0]
    modified, _ = subject.build_features(changed)
    ramp_columns = ["own_rl_ramp_previous", "own_rl_ramp_next1", "own_rl_ramp_next2",
                    "own_wind_drop_next2", "own_solar_drop_next2"]
    for zone in source:
        previous_day = local_mask(source[zone].index, "2026-01-15", TIMEZONES[zone])
        pd.testing.assert_frame_equal(original[zone].loc[previous_day, ramp_columns],
                                      modified[zone].loc[previous_day, ramp_columns])
        daily = original[zone].loc[previous_day]
        assert daily["own_rl_ramp_previous"].iloc[0] == 0.0
        assert daily[["own_rl_ramp_next1", "own_rl_ramp_next2",
                      "own_wind_drop_next2", "own_solar_drop_next2"]].iloc[-1].eq(0.0).all()


def test_extreme_current_residual_load_is_unsaturated_with_unchanged_past_normalizers():
    source = panels(days=15)
    _, original_audit = subject.build_features(source)
    source["DE"].iloc[-24:, :] = [0.0, 0.0, 1000.0]
    features, audit = subject.build_features(source)
    assert (features["DE"]["own_residual_stress"].iloc[-24:] > 1.0).all()
    assert (features["DE"]["own_deficit_stress"].iloc[-24:] > 1.0).all()
    pd.testing.assert_series_equal(features["DE"]["own_residual_stress"],
                                   features["NL"]["other_residual_stress"], check_names=False)
    assert audit["normalizations"] == original_audit["normalizations"]
    assert audit["stress_upper_clipped"] is False


def test_feature_builder_ignores_realized_prices_and_targets():
    source = panels()
    expected, _ = subject.build_features(source)
    for frame in source.values():
        frame["actual"] = "not available at forecast time"
        frame["target"] = np.inf
        frame["residual_kalman__q50"] = -1e20
    actual, _ = subject.build_features(source)
    for zone in source:
        pd.testing.assert_frame_equal(expected[zone], actual[zone])


@pytest.mark.parametrize("start,target,hours", [
    ("2026-03-10", "2026-03-29", 23), ("2025-10-07", "2025-10-26", 25),
])
def test_dst_days_preserve_every_physical_hour(start, target, hours):
    source = panels(start=start, days=21)
    features, _ = subject.build_features(source)
    for zone, frame in features.items():
        mask = local_mask(frame.index, target, TIMEZONES[zone])
        assert int(mask.sum()) == hours
        assert frame.index.is_unique
        assert np.isfinite(frame.loc[mask].to_numpy(float)).all()
        pd.testing.assert_index_equal(frame.index, source[zone].index)


@pytest.mark.parametrize("fault", ["missing_hour", "missing_day", "duplicate", "partial_day",
                                   "naive", "local_timezone", "unsorted", "missing_column",
                                   "missing_zone", "nan", "negative_wind"])
def test_invalid_or_unaligned_forecast_grids_fail_closed(fault):
    source = panels()
    frame = source["NL"]
    if fault == "missing_hour":
        frame = frame.drop(frame.index[30])
    elif fault == "missing_day":
        frame = frame.drop(frame.index[24:48])
    elif fault == "duplicate":
        frame = pd.concat([frame.iloc[:1], frame])
    elif fault == "partial_day":
        frame = frame.iloc[:-1]
    elif fault == "naive":
        frame.index = frame.index.tz_localize(None)
    elif fault == "local_timezone":
        frame.index = frame.index.tz_convert("Europe/Amsterdam")
    elif fault == "unsorted":
        frame = frame.iloc[::-1]
    elif fault == "missing_column":
        frame = frame.iloc[:, :2]
    elif fault == "missing_zone":
        source.pop("DE")
    elif fault == "nan":
        frame.iloc[0, 2] = np.nan
    else:
        frame.iloc[0, 0] = -1.0
    source["NL"] = frame
    with pytest.raises(ValueError):
        subject.build_features(source)


def test_mixture_quantiles_invert_cdf_instead_of_averaging_expert_medians():
    # Uniform distributions with disjoint supports admit an exact inverse.
    result = subject.mixture_quantiles(
        np.array([[0.0, 10.0]]), np.array([[100.0, 110.0]]), np.array([0.2]),
        knot_probabilities=np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(result, [[1.25, 6.25, 105.0]], atol=1e-7, rtol=0)
    assert not np.isclose(result[0, 1], 0.8 * 5.0 + 0.2 * 105.0)


@pytest.mark.parametrize("probability,expected", [(0.0, [0., 0., 0.]), (0.2, [0., 0., 100.]),
                                                 (0.8, [0., 100., 100.]), (1.0, [100., 100., 100.])])
def test_mixture_preserves_point_masses_and_pure_regimes(probability, expected):
    result = subject.mixture_quantiles(
        np.zeros((1, 11)), np.full((1, 11), 100.0), np.array([probability]),
    )
    np.testing.assert_allclose(result[0], expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize("fault", ["crossed", "nan", "negative_probability", "large_probability", "shape"])
def test_invalid_mixture_distributions_are_not_repaired_silently(fault):
    normal = np.linspace(0.0, 10.0, 11)[None, :]
    spike = normal + 100.0
    probability = np.array([0.2])
    if fault == "crossed":
        normal[0, 5] = -1.0
    elif fault == "nan":
        spike[0, 5] = np.nan
    elif fault == "negative_probability":
        probability[0] = -0.1
    elif fault == "large_probability":
        probability[0] = 1.1
    else:
        spike = spike[:, :-1]
    with pytest.raises(ValueError):
        subject.mixture_quantiles(normal, spike, probability)


def test_tiny_pooled_fit_returns_finite_ordered_quantiles_and_causal_audit():
    features, residual, days, origin = training_data()
    before = features.copy(deep=True)
    original_labels = residual.copy()
    quantiles, probability, audit = small_fit(features, residual, days, origin)
    assert quantiles.shape == (8, 3)
    assert probability.shape == (8,)
    assert np.isfinite(quantiles).all() and np.isfinite(probability).all()
    assert (np.diff(quantiles, axis=1) >= 0).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert pd.Timestamp(audit["train_last_day"]) < pd.Timestamp(audit["origin_day"])
    gate = audit["gate"]
    assert gate["method"] == "catboost_platt_calibrated"
    assert gate["calibration_window_days"] == 14
    assert gate["calibration_rows"] == 14 * 4
    assert pd.Timestamp(gate["gate_fit_last_day"]) < pd.Timestamp(gate["calibration_first_day"])
    assert pd.Timestamp(gate["calibration_first_day"]) == pd.Timestamp(origin) - pd.Timedelta(days=14)
    assert pd.Timestamp(gate["calibration_last_day"]) == pd.Timestamp(origin) - pd.Timedelta(days=1)
    assert gate["class_weights_used"] is False
    assert audit["correction_clip"] is None
    pd.testing.assert_frame_equal(features, before)
    np.testing.assert_array_equal(residual, original_labels)
    json.dumps(audit, allow_nan=False)


def test_sparse_spikes_use_explicit_empirical_gate_and_spike_expert():
    features, residual, days, origin = training_data()
    residual[:] = np.arange(len(residual)) % 17 - 8.0
    residual[-3:] = [100.0, 120.0, 140.0]
    quantiles, probability, audit = small_fit(features, residual, days, origin)
    assert audit["gate"]["method"] == "empirical_frequency"
    assert audit["experts"]["spike"]["method"] == "empirical_quantiles"
    np.testing.assert_allclose(probability, 3.0 / len(residual))
    assert np.isfinite(quantiles).all()
    assert (np.diff(quantiles, axis=1) >= 0).all()


@pytest.mark.parametrize("label,expected_probability", [(50.0, 0.0), (120.0, 1.0)])
def test_strict_spike_threshold_and_constant_regime_have_no_forty_euro_cap(label, expected_probability):
    features, residual, days, origin = training_data()
    residual[:] = label
    quantiles, probability, audit = small_fit(features, residual, days, origin)
    np.testing.assert_allclose(probability, expected_probability)
    np.testing.assert_allclose(quantiles, label, atol=1e-7, rtol=0)
    assert audit["gate"]["method"] == "empirical_frequency"
    if label > 50:
        assert (quantiles[:, 1] > 40.0).all()


@pytest.mark.parametrize("offset", [0, 1, -366])
def test_origin_future_and_out_of_window_labels_are_rejected(offset):
    features, residual, days, origin = training_data()
    invalid = days.copy()
    invalid[-1] = (pd.Timestamp(origin) + pd.Timedelta(days=offset)).date()
    with pytest.raises(ValueError):
        small_fit(features, residual, invalid, origin)


def test_insufficient_training_days_do_not_silently_fit():
    features, residual, days, origin = training_data(days=89)
    with pytest.raises(ValueError):
        small_fit(features, residual, days, origin)


@pytest.mark.parametrize("fault", ["nan_label", "nan_feature", "infinite_test", "unaligned_labels"])
def test_fit_rejects_nonfinite_or_unaligned_inputs(fault):
    features, residual, days, origin = training_data()
    test = features.iloc[-8:].copy()
    if fault == "nan_label":
        residual[0] = np.nan
    elif fault == "nan_feature":
        features.iloc[0, 0] = np.nan
    elif fault == "infinite_test":
        test.iloc[0, 0] = np.inf
    else:
        residual = residual[:-1]
    with pytest.raises(ValueError):
        small_fit(features, residual, days, origin, test=test)


def test_runner_has_forty_contiguous_weekly_blocks_and_one_fresh_diagnostic_fit():
    import run_solar_wind_scarcity_regime as runner

    schedule = runner.blocks()
    assert len(schedule) == 41
    historical = schedule[:-1]
    assert all(not is_forecast for _, _, is_forecast in historical)
    assert pd.Timestamp(historical[0][0]) == pd.Timestamp("2025-12-21")
    assert pd.Timestamp(historical[-1][1]) == pd.Timestamp("2026-09-22")
    assert sum((stop - origin).days for origin, stop, _ in historical) == 275
    assert all(1 <= (stop - origin).days <= 7 for origin, stop, _ in historical)
    assert all(left[1] == right[0] for left, right in zip(schedule, schedule[1:]))
    assert schedule[-1] == (pd.Timestamp("2026-09-22").date(),
                            pd.Timestamp("2026-09-23").date(), True)
    assert len(runner.expected_index("2025-12-21", "2026-09-22")) == 6599
    assert len(runner.expected_index("2026-09-22", "2026-09-23")) == 24


@pytest.mark.parametrize("fault,message", [
    ("identity", "Invalid completion"), ("status", "Invalid completion"),
    ("days", "evaluation support"), ("hours", "evaluation support"),
    ("forecast_hours", "evaluation support"),
    ("missing_inventory", "exact expected result inventory"),
    ("extra_inventory", "exact expected result inventory"),
])
def test_runner_rejects_invalid_completion_before_source_references(tmp_path, monkeypatch, fault, message):
    import run_solar_wind_scarcity_regime as runner

    inventory = {"experiment.json", "feature_audit.json", "backtest.parquet",
                 "forecast_diagnostic.parquet", "metrics.json", "fit_audits.json", "report.html"}
    for origin, _, _ in runner.blocks():
        inventory.update(f"checkpoints/{origin}{suffix}" for suffix in (".json", ".audit.json", ".parquet"))
    receipt = {"status": "COMPLETE", "identity": "synthetic-test-only",
               "evaluation_days": 275, "hours_per_zone": 6599,
               "diagnostic_forecast_hours_per_zone": 24,
               "files": {name: "not-a-real-source-digest" for name in inventory}}
    if fault == "identity":
        receipt["identity"] = "different"
    elif fault == "status":
        receipt["status"] = "RUNNING"
    elif fault == "days":
        receipt["evaluation_days"] = 274
    elif fault == "hours":
        receipt["hours_per_zone"] = 6600
    elif fault == "forecast_hours":
        receipt["diagnostic_forecast_hours_per_zone"] = 23
    elif fault == "missing_inventory":
        receipt["files"].pop("report.html")
    else:
        receipt["files"]["unapproved-extra.json"] = "not-a-real-source-digest"
    (tmp_path / "completion.json").write_text(json.dumps(receipt), encoding="utf-8")

    def unexpected_reference_access(*args, **kwargs):
        pytest.fail("Invalid completion must be rejected before reading any source reference")

    monkeypatch.setattr(runner, "verify_pins", unexpected_reference_access)
    monkeypatch.setattr(runner, "sha", unexpected_reference_access)
    with pytest.raises(ValueError, match=message):
        runner.verify_completed(tmp_path, "synthetic-test-only", {"sources_and_code_sha256": {}})
