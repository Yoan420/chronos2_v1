from __future__ import annotations

import numpy as np
import pandas as pd

from runs.tmp.screen_public_exogenous_strict import (
    AUTONOMOUS_WEIGHT,
    FAMILY_ORDER,
    MKONLINE_WEIGHT,
    _candidate_passes_b1,
    _candidate_passes_b2,
    _exact_l1_scale,
    _family_features,
    _predeclared_families,
    _read_exogenous_window,
    _score_block,
)


def test_frozen_reference_weights_are_exact() -> None:
    assert AUTONOMOUS_WEIGHT == 0.4977609282924196
    assert MKONLINE_WEIGHT == 0.5022390717075804
    assert AUTONOMOUS_WEIGHT + MKONLINE_WEIGHT == 1.0


def test_exact_l1_scale_is_constrained_optimum() -> None:
    residual = np.array([0.0, 2.0, 4.0, 6.0])
    correction = np.full(4, 4.0)
    scale = _exact_l1_scale(residual, correction)
    grid = np.linspace(0.0, 1.0, 10_001)
    selected = np.mean(np.abs(residual - scale * correction))
    brute_force = np.min(
        np.mean(
            np.abs(residual[:, None] - grid[None, :] * correction[:, None]),
            axis=0,
        )
    )
    assert 0.0 <= scale <= 1.0
    assert selected <= brute_force + 1e-10


def test_predeclared_families_are_name_based_and_exclude_quality_flags() -> None:
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "eco2mix_load_fcst_j1_profile_d2": np.arange(24.0),
            "eco2mix_wind_profile_d2": np.arange(24.0),
            "eco2mix_nuclear_profile_d2": np.arange(24.0),
            "temperature_2m_previous_day2": np.arange(24.0),
            "eco2mix_load_fcst_j1_profile_d2_missing_flag": 0,
            "unclassified_public_signal": np.arange(24.0),
        },
        index=index,
    )
    families, audit = _predeclared_families(frame, minimum_coverage=1.0)
    assert tuple(name for name in FAMILY_ORDER if name in families) == tuple(families)
    assert families["demand"] == ["eco2mix_load_fcst_j1_profile_d2"]
    assert families["renewables"] == ["eco2mix_wind_profile_d2"]
    assert families["dispatchable"] == ["eco2mix_nuclear_profile_d2"]
    assert families["weather"] == ["temperature_2m_previous_day2"]
    assert "eco2mix_load_fcst_j1_profile_d2_missing_flag" not in families["demand"]
    assert audit["unclassified_admitted_columns"] == [
        "unclassified_public_signal"
    ]


def test_family_feature_schema_is_stable_when_missingness_moves() -> None:
    first_index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    second_index = pd.date_range("2025-02-01", periods=2, freq="h", tz="UTC")
    first = pd.DataFrame({"public_load": [1.0, np.nan]}, index=first_index)
    second = pd.DataFrame({"public_load": [2.0, 3.0]}, index=second_index)
    first_features = _family_features(first, ["public_load"])
    second_features = _family_features(second, ["public_load"])
    assert list(first_features.columns) == list(second_features.columns)
    assert "missing__public_load" in second_features


def test_exogenous_reader_filters_at_parquet_boundary(tmp_path) -> None:
    full_index = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    path = tmp_path / "public.parquet"
    pd.DataFrame(
        {
            "delivery_start_utc": full_index,
            "public_load": np.arange(5.0),
        }
    ).to_parquet(path, index=False)
    expected = pd.DatetimeIndex(
        full_index[1:4], name="delivery_start_utc"
    )
    selected = _read_exogenous_window(path, expected)
    assert selected.index.equals(expected)
    assert selected["public_load"].tolist() == [1.0, 2.0, 3.0]


def test_exogenous_reader_accepts_value_time_utc_alias(tmp_path) -> None:
    expected = pd.date_range(
        "2025-01-01", periods=24, freq="h", tz="UTC",
        name="delivery_start_utc",
    )
    path = tmp_path / "weather.parquet"
    pd.DataFrame(
        {
            "value_time_utc": expected,
            "openmeteo_temperature_2m_c__panel_mean": np.arange(24.0),
        }
    ).to_parquet(path, index=False)

    selected = _read_exogenous_window(path, expected)

    assert selected.index.equals(expected)
    assert list(selected) == ["openmeteo_temperature_2m_c__panel_mean"]


def test_exogenous_reader_validates_and_drops_weather_audit_times(tmp_path) -> None:
    expected = pd.date_range(
        "2025-01-01", periods=24, freq="h", tz="UTC",
        name="delivery_start_utc",
    )
    path = tmp_path / "weather.parquet"
    pd.DataFrame(
        {
            "value_time_utc": expected,
            "weather_fixed_lead_reference_time_utc": expected - pd.Timedelta(hours=48),
            "weather_cutoff_time_utc": expected - pd.Timedelta(hours=8),
            "openmeteo_temperature_2m_c__panel_mean": np.arange(24.0),
        }
    ).to_parquet(path, index=False)

    selected = _read_exogenous_window(path, expected)

    assert list(selected) == ["openmeteo_temperature_2m_c__panel_mean"]


def test_b1_gate_requires_threshold_and_both_halves_vs_two_references() -> None:
    index = pd.date_range(
        "2025-04-13T22:00:00Z", periods=60 * 24, freq="h", tz="UTC"
    )
    actual = pd.Series(0.0, index=index)
    baseline = pd.Series(1.0, index=index)
    control = pd.Series(0.8, index=index)
    candidate = pd.Series(
        np.concatenate([np.full(30 * 24, 0.4), np.full(30 * 24, 0.6)]),
        index=index,
    )
    score = _score_block(
        index, actual, baseline, candidate, control=control
    )
    assert _candidate_passes_b1(
        score, minimum_gain=0.05, minimum_gain_vs_calendar=0.01
    )

    candidate.iloc[30 * 24 :] = 0.9
    failed = _score_block(index, actual, baseline, candidate, control=control)
    assert not _candidate_passes_b1(
        failed, minimum_gain=0.05, minimum_gain_vs_calendar=0.01
    )


def test_b2_is_a_veto_with_no_aggregate_only_escape() -> None:
    index = pd.date_range(
        "2025-06-12T22:00:00Z", periods=60 * 24, freq="h", tz="UTC"
    )
    actual = pd.Series(0.0, index=index)
    baseline = pd.Series(1.0, index=index)
    control = pd.Series(0.9, index=index)
    candidate = pd.Series(
        np.concatenate([np.full(30 * 24, 0.4), np.full(30 * 24, 1.1)]),
        index=index,
    )
    score = _score_block(index, actual, baseline, candidate, control=control)
    assert score["all"]["gain_vs_frozen_blend"] > 0.0
    assert score["last30"]["gain_vs_frozen_blend"] < 0.0
    assert not _candidate_passes_b2(score)
