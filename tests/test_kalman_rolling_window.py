from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.kalman_residual as kalman_residual
from chronos2_hourly.kalman_residual import (
    KalmanResidualConfig,
    KalmanResidualError,
    build_operational_kalman_view,
    replay_kalman_overlay,
    validate_operational_kalman_history,
)


TIMEZONE = "Europe/Paris"


def _history(*, start: str, days: int) -> pd.DataFrame:
    local_start = pd.Timestamp(start, tz=TIMEZONE)
    local_end = local_start + pd.DateOffset(days=days)
    index = pd.date_range(
        local_start,
        local_end,
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    local_days = pd.Index(index.tz_convert(TIMEZONE).date)
    unique_days = list(dict.fromkeys(local_days))
    positions = {day: position for position, day in enumerate(unique_days)}
    day_bias = np.asarray([2.0 + positions[day] for day in local_days])
    base = 50.0 + 3.0 * np.sin(2.0 * np.pi * index.hour / 24.0)
    return pd.DataFrame(
        {
            "delivery_start_utc": index,
            "actual": base + day_bias,
            "residual_corrected__q10": base - 8.0,
            "residual_corrected__q50": base,
            "residual_corrected__q90": base + 8.0,
        }
    )


def _config(*, lookback: int = 4) -> KalmanResidualConfig:
    return KalmanResidualConfig(
        governance_lookback_days=min(3, lookback),
        governance_minimum_days=min(2, lookback),
        minimum_gain_eur_mwh=0.0001,
        minimum_relative_gain=0.0,
        candidate_kinds=("linear_bias",),
    )


def _day_mask(frame: pd.DataFrame, day: str) -> pd.Series:
    local = pd.to_datetime(frame["delivery_start_utc"], utc=True).dt.tz_convert(
        TIMEZONE
    )
    return local.dt.date == pd.Timestamp(day).date()


def _prediction_for_day(result, day: str) -> pd.Series:
    local_days = result.predictions.index.tz_convert(TIMEZONE).date
    return result.predictions.loc[
        local_days == pd.Timestamp(day).date(), "residual_kalman__q50"
    ]


def test_rolling_window_rejects_invalid_lookback() -> None:
    history = _history(start="2026-01-01", days=4)
    for invalid in (0, -1, True, 2.5):
        with pytest.raises(KalmanResidualError, match="training_lookback_days"):
            replay_kalman_overlay(
                history,
                timezone=TIMEZONE,
                evaluation_start_day="2026-01-03",
                training_lookback_days=invalid,
                config=_config(),
            )


def test_rolling_window_rejects_invalid_worker_count() -> None:
    history = _history(start="2026-01-01", days=4)
    for invalid in (0, -1, True, 2.5, 9):
        with pytest.raises(KalmanResidualError, match="rolling_refit_workers"):
            replay_kalman_overlay(
                history,
                timezone=TIMEZONE,
                evaluation_start_day="2026-01-03",
                training_lookback_days=2,
                rolling_refit_workers=invalid,
                config=_config(lookback=2),
            )


def test_parallel_rolling_origins_are_bitwise_equivalent() -> None:
    history = _history(start="2026-01-01", days=9)
    arguments = {
        "timezone": TIMEZONE,
        "evaluation_start_day": "2026-01-05",
        "training_lookback_days": 4,
        "config": _config(),
    }

    serial = replay_kalman_overlay(
        history,
        rolling_refit_workers=1,
        **arguments,
    )
    parallel = replay_kalman_overlay(
        history,
        rolling_refit_workers=2,
        **arguments,
    )

    pd.testing.assert_frame_equal(serial.predictions, parallel.predictions)
    pd.testing.assert_frame_equal(
        serial.candidate_predictions,
        parallel.candidate_predictions,
    )
    pd.testing.assert_frame_equal(serial.daily_audit, parallel.daily_audit)
    pd.testing.assert_frame_equal(serial.state_audit, parallel.state_audit)
    assert serial.audit["rolling_refit_workers"] == 1
    assert parallel.audit["rolling_refit_workers"] == 2


def test_persistent_rolling_cache_reuses_every_unchanged_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _history(start="2026-01-01", days=9)
    cache = tmp_path / "rolling-cache"
    arguments = {
        "timezone": TIMEZONE,
        "evaluation_start_day": "2026-01-05",
        "training_lookback_days": 4,
        "rolling_refit_workers": 1,
        "rolling_refit_cache_dir": cache,
        "config": _config(),
    }

    fresh = replay_kalman_overlay(history, **arguments)
    fresh_cache = fresh.audit["rolling_refit_cache"]
    assert fresh_cache["history_hits"] == 0
    assert fresh_cache["history_misses"] == 5
    assert fresh_cache["history_fitted_days"] == 5
    assert fresh_cache["writes"] == 5
    assert len(list(cache.rglob("*.pickle"))) == 5

    def unexpected_fit(**_kwargs):
        raise AssertionError("Un cache complet ne doit lancer aucun refit.")

    monkeypatch.setattr(kalman_residual, "_fit_rolling_target_day", unexpected_fit)
    reused = replay_kalman_overlay(history, **arguments)

    reused_cache = reused.audit["rolling_refit_cache"]
    assert reused_cache["history_hits"] == 5
    assert reused_cache["history_misses"] == 0
    assert reused_cache["history_fitted_days"] == 0
    assert reused.audit["rolling_refit_active_workers"] == 0
    pd.testing.assert_frame_equal(fresh.predictions, reused.predictions)
    pd.testing.assert_frame_equal(
        fresh.candidate_predictions,
        reused.candidate_predictions,
    )
    pd.testing.assert_frame_equal(fresh.daily_audit, reused.daily_audit)
    pd.testing.assert_frame_equal(fresh.state_audit, reused.state_audit)


def test_persistent_rolling_cache_only_fits_the_new_sliding_day(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "rolling-cache"
    common = {
        "timezone": TIMEZONE,
        "training_lookback_days": 4,
        "rolling_refit_workers": 1,
        "rolling_refit_cache_dir": cache,
        "config": _config(),
    }
    first = replay_kalman_overlay(
        _history(start="2026-01-01", days=9),
        evaluation_start_day="2026-01-05",
        **common,
    )
    shifted = replay_kalman_overlay(
        _history(start="2026-01-01", days=10),
        evaluation_start_day="2026-01-06",
        **common,
    )

    assert first.audit["rolling_refit_cache"]["history_fitted_days"] == 5
    shifted_cache = shifted.audit["rolling_refit_cache"]
    assert shifted_cache["history_hits"] == 4
    assert shifted_cache["history_misses"] == 1
    assert shifted_cache["history_fitted_days"] == 1
    assert shifted_cache["writes"] == 1
    assert shifted_cache["stale_entries"] == 0
    assert len(list(cache.rglob("*.pickle"))) == 6


def test_persistent_rolling_cache_promotes_yesterdays_future_fit(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "rolling-cache"
    complete = _history(start="2026-01-01", days=11)
    first_history = complete.loc[~_day_mask(complete, "2026-01-10")].copy()
    first_history = first_history.loc[
        ~_day_mask(first_history, "2026-01-11")
    ].copy()
    first_future = complete.loc[_day_mask(complete, "2026-01-10")].copy()
    second_history = complete.loc[~_day_mask(complete, "2026-01-11")].copy()
    second_future = complete.loc[_day_mask(complete, "2026-01-11")].copy()
    common = {
        "timezone": TIMEZONE,
        "training_lookback_days": 4,
        "rolling_refit_workers": 1,
        "rolling_refit_cache_dir": cache,
        "config": _config(),
    }

    issued = replay_kalman_overlay(
        first_history,
        evaluation_start_day="2026-01-05",
        future_upstream=first_future,
        **common,
    )
    shifted = replay_kalman_overlay(
        second_history,
        evaluation_start_day="2026-01-06",
        future_upstream=second_future,
        **common,
    )

    issued_cache = issued.audit["rolling_refit_cache"]
    assert issued_cache["history_fitted_days"] == 5
    assert issued_cache["future_fitted_days"] == 1
    shifted_cache = shifted.audit["rolling_refit_cache"]
    assert shifted_cache["history_hits"] == 5
    assert shifted_cache["history_fitted_days"] == 0
    assert shifted_cache["future_hits"] == 0
    assert shifted_cache["future_misses"] == 1
    assert shifted_cache["future_fitted_days"] == 1
    np.testing.assert_allclose(
        issued.future_predictions["residual_kalman__q50"],
        _prediction_for_day(shifted, "2026-01-10"),
        rtol=0.0,
        atol=0.0,
    )


def test_persistent_rolling_cache_invalidates_changed_training_data(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "rolling-cache"
    history = _history(start="2026-01-01", days=9)
    arguments = {
        "timezone": TIMEZONE,
        "evaluation_start_day": "2026-01-09",
        "training_lookback_days": 4,
        "rolling_refit_workers": 1,
        "rolling_refit_cache_dir": cache,
        "config": _config(),
    }
    original = replay_kalman_overlay(history, **arguments)
    changed = history.copy()
    changed.loc[_day_mask(changed, "2026-01-08"), "actual"] += 100.0
    invalidated = replay_kalman_overlay(changed, **arguments)

    assert original.audit["rolling_refit_cache"]["history_misses"] == 1
    cache_audit = invalidated.audit["rolling_refit_cache"]
    assert cache_audit["history_hits"] == 0
    assert cache_audit["history_misses"] == 1
    assert cache_audit["stale_entries"] == 1
    assert not np.allclose(
        _prediction_for_day(original, "2026-01-09"),
        _prediction_for_day(invalidated, "2026-01-09"),
    )


def test_corrupt_rolling_cache_entry_is_recomputed_safely(tmp_path: Path) -> None:
    cache = tmp_path / "rolling-cache"
    history = _history(start="2026-01-01", days=9)
    arguments = {
        "timezone": TIMEZONE,
        "evaluation_start_day": "2026-01-09",
        "training_lookback_days": 4,
        "rolling_refit_workers": 1,
        "rolling_refit_cache_dir": cache,
        "config": _config(),
    }
    fresh = replay_kalman_overlay(history, **arguments)
    cache_entry = next(cache.rglob("2026-01-09.pickle"))
    cache_entry.write_bytes(b"not-a-pickle")

    repaired = replay_kalman_overlay(history, **arguments)

    cache_audit = repaired.audit["rolling_refit_cache"]
    assert cache_audit["invalid_entries"] == 1
    assert cache_audit["history_fitted_days"] == 1
    assert cache_audit["write_errors"] == 0
    pd.testing.assert_frame_equal(fresh.predictions, repaired.predictions)


def test_persistent_cache_partitions_distinct_model_contracts(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "rolling-cache"
    history = _history(start="2026-01-01", days=9)
    arguments = {
        "timezone": TIMEZONE,
        "evaluation_start_day": "2026-01-09",
        "training_lookback_days": 4,
        "rolling_refit_workers": 1,
        "rolling_refit_cache_dir": cache,
    }
    first_config = _config()
    second_config = replace(first_config, q_over_r=first_config.q_over_r * 2.0)

    replay_kalman_overlay(history, config=first_config, **arguments)
    replay_kalman_overlay(history, config=second_config, **arguments)
    reused = replay_kalman_overlay(history, config=first_config, **arguments)

    contract_directories = [
        path
        for path in (cache / "schema-1").iterdir()
        if path.is_dir()
    ]
    assert len(contract_directories) == 2
    assert len(list(cache.rglob("2026-01-09.pickle"))) == 2
    assert reused.audit["rolling_refit_cache"]["history_hits"] == 1
    assert reused.audit["rolling_refit_cache"]["history_fitted_days"] == 0


def test_cache_does_not_strengthen_validation_of_an_unused_prefix(
    tmp_path: Path,
) -> None:
    history = _history(start="2026-01-01", days=10).iloc[1:].reset_index(
        drop=True
    )
    arguments = {
        "timezone": TIMEZONE,
        "evaluation_start_day": "2026-01-06",
        "training_lookback_days": 4,
        "rolling_refit_workers": 1,
        "config": _config(),
    }

    uncached = replay_kalman_overlay(history, **arguments)
    cached = replay_kalman_overlay(
        history,
        rolling_refit_cache_dir=tmp_path / "rolling-cache",
        **arguments,
    )

    pd.testing.assert_frame_equal(uncached.predictions, cached.predictions)
    pd.testing.assert_frame_equal(
        uncached.candidate_predictions,
        cached.candidate_predictions,
    )


def test_rolling_prediction_ignores_target_actual_but_uses_it_next_day() -> None:
    history = _history(start="2026-01-01", days=11)
    changed = history.copy()
    changed.loc[_day_mask(changed, "2026-01-09"), "actual"] += 500.0

    original = replay_kalman_overlay(
        history,
        timezone=TIMEZONE,
        evaluation_start_day="2026-01-09",
        training_lookback_days=4,
        config=_config(),
    )
    alternative = replay_kalman_overlay(
        changed,
        timezone=TIMEZONE,
        evaluation_start_day="2026-01-09",
        training_lookback_days=4,
        config=_config(),
    )

    np.testing.assert_allclose(
        _prediction_for_day(original, "2026-01-09"),
        _prediction_for_day(alternative, "2026-01-09"),
        rtol=0.0,
        atol=0.0,
    )
    assert not np.allclose(
        _prediction_for_day(original, "2026-01-10"),
        _prediction_for_day(alternative, "2026-01-10"),
    )
    scored = original.daily_audit.loc[
        original.daily_audit["training_window_complete"]
    ]
    assert (scored["target_observations_assimilated"] == 0).all()
    assert original.audit["target_actuals_assimilated_before_forecast"] == 0


def test_fast_rolling_updates_match_legacy_equations_on_same_window() -> None:
    history = _history(start="2026-01-01", days=6)
    config = KalmanResidualConfig(
        governance_lookback_days=3,
        governance_minimum_days=2,
        minimum_gain_eur_mwh=0.0001,
        minimum_relative_gain=0.0,
        candidate_kinds=(
            "linear_bias",
            "linear_harmonic",
            "linear_market",
            "linear_scale",
            "ekf_scale",
            "ukf_scale",
        ),
    )
    legacy = replay_kalman_overlay(
        history,
        timezone=TIMEZONE,
        evaluation_start_day="2026-01-06",
        config=config,
    )
    rolling = replay_kalman_overlay(
        history,
        timezone=TIMEZONE,
        evaluation_start_day="2026-01-06",
        training_lookback_days=5,
        config=config,
    )
    target = rolling.predictions.index.tz_convert(TIMEZONE).date == pd.Timestamp(
        "2026-01-06"
    ).date()

    np.testing.assert_allclose(
        rolling.predictions.loc[target].filter(like="residual_kalman__"),
        legacy.predictions.loc[target].filter(like="residual_kalman__"),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        rolling.candidate_predictions.loc[target],
        legacy.candidate_predictions.loc[target],
        rtol=1e-8,
        atol=1e-8,
    )


def test_value_older_than_window_has_no_effect_but_in_window_value_does() -> None:
    history = _history(start="2026-01-01", days=10)
    old_changed = history.copy()
    old_changed.loc[_day_mask(old_changed, "2026-01-02"), "actual"] += 1000.0
    recent_changed = history.copy()
    recent_changed.loc[_day_mask(recent_changed, "2026-01-08"), "actual"] += 100.0
    arguments = {
        "timezone": TIMEZONE,
        "evaluation_start_day": "2026-01-09",
        "training_lookback_days": 4,
        "config": _config(),
    }

    original = replay_kalman_overlay(history, **arguments)
    old_result = replay_kalman_overlay(old_changed, **arguments)
    recent_result = replay_kalman_overlay(recent_changed, **arguments)

    np.testing.assert_allclose(
        _prediction_for_day(original, "2026-01-09"),
        _prediction_for_day(old_result, "2026-01-09"),
        rtol=0.0,
        atol=0.0,
    )
    assert not np.allclose(
        _prediction_for_day(original, "2026-01-09"),
        _prediction_for_day(recent_result, "2026-01-09"),
    )
    original_candidate = original.candidate_predictions.loc[
        original.candidate_predictions.index.tz_convert(TIMEZONE).date
        == pd.Timestamp("2026-01-09").date(),
        "linear_bias__q50",
    ]
    recent_candidate = recent_result.candidate_predictions.loc[
        recent_result.candidate_predictions.index.tz_convert(TIMEZONE).date
        == pd.Timestamp("2026-01-09").date(),
        "linear_bias__q50",
    ]
    assert not np.allclose(original_candidate, recent_candidate)


def test_rolling_window_audit_counts_physical_dst_hours_exactly() -> None:
    history = _history(start="2025-03-27", days=7)
    result = replay_kalman_overlay(
        history,
        timezone=TIMEZONE,
        evaluation_start_day="2025-03-30",
        training_lookback_days=3,
        config=_config(lookback=3),
    )
    audit = result.daily_audit.set_index("local_day")

    assert audit.loc["2025-03-30", "hours"] == 23
    assert audit.loc["2025-03-30", "training_window_start"] == "2025-03-27"
    assert audit.loc["2025-03-30", "training_window_end"] == "2025-03-29"
    assert audit.loc["2025-03-30", "training_window_days"] == 3
    assert audit.loc["2025-03-30", "training_window_hours"] == 72
    assert audit.loc["2025-03-31", "training_window_hours"] == 71
    assert result.audit["training_window_hours_min"] == 71
    assert result.audit["training_window_hours_max"] == 72


def test_explicit_365_day_training_window_is_audited_end_to_end() -> None:
    history = _history(start="2025-01-01", days=366)
    result = replay_kalman_overlay(
        history,
        timezone=TIMEZONE,
        evaluation_start_day="2026-01-01",
        training_lookback_days=365,
        config=_config(lookback=365),
    )
    target = result.daily_audit.set_index("local_day").loc["2026-01-01"]

    assert target["training_window_start"] == "2025-01-01"
    assert target["training_window_end"] == "2025-12-31"
    assert target["training_window_days"] == 365
    assert target["training_window_hours"] == 8760
    assert result.audit["training_policy"] == "fixed_length_rolling_local_days"
    assert result.audit["training_lookback_days"] == 365


def test_rolling_window_rejects_a_missing_physical_hour() -> None:
    history = _history(start="2025-03-27", days=5)
    missing = history.drop(index=25).reset_index(drop=True)
    with pytest.raises(KalmanResidualError, match="journee locale incomplete"):
        replay_kalman_overlay(
            missing,
            timezone=TIMEZONE,
            evaluation_start_day="2025-03-30",
            training_lookback_days=3,
            config=_config(lookback=3),
        )


def test_rolling_future_uses_exact_window_and_ignores_future_actual() -> None:
    full = _history(start="2026-01-01", days=9)
    future_mask = _day_mask(full, "2026-01-09")
    history = full.loc[~future_mask].copy()
    future = full.loc[future_mask].copy()
    changed_future = future.copy()
    changed_future["actual"] += 10_000.0
    arguments = {
        "timezone": TIMEZONE,
        "evaluation_start_day": "2026-01-05",
        "training_lookback_days": 4,
        "config": _config(),
    }

    original = replay_kalman_overlay(
        history,
        future_upstream=future,
        **arguments,
    )
    changed = replay_kalman_overlay(
        history,
        future_upstream=changed_future,
        **arguments,
    )

    pd.testing.assert_frame_equal(
        original.future_predictions,
        changed.future_predictions,
    )
    assert original.future_audit["training_window_start"] == "2026-01-05"
    assert original.future_audit["training_window_end"] == "2026-01-08"
    assert original.future_audit["training_window_days"] == 4
    assert original.future_audit["training_window_hours"] == 96
    assert original.future_audit["last_observation_used"] == "2026-01-08"
    assert original.future_audit["observations_assimilated"] == 0
    assert original.future_audit["actual_column_ignored"] is True


def _operational_history(delivery_day: str = "2026-09-08") -> pd.DataFrame:
    start = (pd.Timestamp(delivery_day) - pd.Timedelta(days=730)).date()
    return _history(start=str(start), days=730)


def test_operational_history_reports_exact_required_and_available_period() -> None:
    short = _history(start="2025-08-12", days=392)

    with pytest.raises(KalmanResidualError, match="Le replay rolling exige") as error:
        validate_operational_kalman_history(
            short, timezone=TIMEZONE, delivery_day="2026-09-08"
        )

    message = str(error.value)
    assert "2024-09-08 au 2026-09-07" in message
    assert "2025-08-12 au 2026-09-07" in message
    assert "730 jours civils, 17520 heures physiques" in message
    assert "9408/17520 heures physiques" in message
    assert "Heures manquantes=8112" in message
    assert "2024-09-07T22:00:00+00:00" in message


@pytest.mark.parametrize(
    ("delivery_day", "expected_hours"),
    (("2026-03-30", 17519), ("2026-09-08", 17520), ("2026-10-26", 17521)),
)
def test_operational_history_audits_exact_civil_730_days_across_dst(
    delivery_day: str, expected_hours: int
) -> None:
    history = _operational_history(delivery_day)
    audit = validate_operational_kalman_history(
        history, timezone=TIMEZONE, delivery_day=delivery_day
    )

    assert audit["status"] == "complete"
    assert audit["required_days"] == audit["available_days"] == 730
    assert audit["expected_hours"] == audit["available_hours"] == expected_hours
    assert audit["missing_hours"] == 0
    assert audit["training_lookback_days"] == audit["evaluation_days"] == 365
    assert audit["required_start_day"] == audit["available_start_day"]
    assert audit["required_end_day"] == audit["available_end_day"]
    assert audit["evaluation_start_day"] == str(
        (pd.Timestamp(delivery_day) - pd.Timedelta(days=365)).date()
    )


@pytest.mark.parametrize("missing_utc", ("2025-10-26T00:00:00Z", "2025-10-26T01:00:00Z"))
def test_operational_history_requires_both_physical_hours_of_dst_fold(
    missing_utc: str,
) -> None:
    history = _operational_history()
    missing_timestamp = pd.Timestamp(missing_utc)
    incomplete = history.loc[
        history["delivery_start_utc"] != missing_timestamp
    ].reset_index(drop=True)

    with pytest.raises(KalmanResidualError, match="Heures manquantes=1") as error:
        validate_operational_kalman_history(
            incomplete, timezone=TIMEZONE, delivery_day="2026-09-08"
        )
    assert missing_timestamp.isoformat() in str(error.value)


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("actual", np.nan),
        ("actual", np.inf),
        ("residual_corrected__q10", -np.inf),
        ("residual_corrected__q50", np.nan),
        ("residual_corrected__q90", np.nan),
    ),
)
def test_operational_history_rejects_nonfinite_observations_and_quantiles(
    column: str, value: float
) -> None:
    history = _operational_history()
    history.loc[25, column] = value

    with pytest.raises(KalmanResidualError, match="Valeurs non finies") as error:
        validate_operational_kalman_history(
            history, timezone=TIMEZONE, delivery_day="2026-09-08"
        )
    assert column in str(error.value)
    assert history.loc[25, "delivery_start_utc"].isoformat() in str(error.value)


def test_operational_history_rejects_crossed_upstream_quantiles() -> None:
    history = _operational_history()
    history.loc[25, "residual_corrected__q10"] = (
        history.loc[25, "residual_corrected__q50"] + 1.0
    )
    with pytest.raises(KalmanResidualError, match="quantiles upstream se croisent"):
        validate_operational_kalman_history(
            history, timezone=TIMEZONE, delivery_day="2026-09-08"
        )


def test_operational_history_rejects_duplicate_and_nonhourly_timestamps() -> None:
    history = _operational_history()
    duplicated = pd.concat([history.iloc[:26], history.iloc[25:]], ignore_index=True)
    with pytest.raises(KalmanResidualError, match="doublons=1"):
        validate_operational_kalman_history(
            duplicated, timezone=TIMEZONE, delivery_day="2026-09-08"
        )
    off_hour = history.copy()
    off_hour.loc[25, "delivery_start_utc"] += pd.Timedelta(microseconds=1)
    with pytest.raises(KalmanResidualError, match="manquantes=1, inattendues=1"):
        validate_operational_kalman_history(
            off_hour, timezone=TIMEZONE, delivery_day="2026-09-08"
        )


def test_operational_history_ignores_unused_prefix_and_current_actual_placeholder() -> None:
    history = _history(start="2024-09-07", days=733)
    local_days = history["delivery_start_utc"].dt.tz_convert(TIMEZONE).dt.date
    outside = (local_days < pd.Timestamp("2024-09-08").date()) | (
        local_days >= pd.Timestamp("2026-09-08").date()
    )
    history.loc[outside, "actual"] = np.nan
    history.loc[outside, "residual_corrected__q10"] = np.inf
    # Irrelevant duplicates and ordering outside the required window cannot
    # invalidate the history used by any of the 365 operational origins.
    history = pd.concat([history, history.loc[outside]], ignore_index=True)
    original = history.copy(deep=True)

    audit = validate_operational_kalman_history(
        history, timezone=TIMEZONE, delivery_day="2026-09-08"
    )

    assert audit["available_start_day"] == "2024-09-08"
    assert audit["available_end_day"] == "2026-09-07"
    assert audit["available_hours"] == 17520
    pd.testing.assert_frame_equal(history, original)


def test_operational_view_rejects_incomplete_history_before_any_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = _operational_history().drop(index=25)
    cache_path = tmp_path / "unused-cache"

    def unexpected_replay(*_args, **_kwargs):
        raise AssertionError("L'historique incomplet doit etre refuse avant tout replay.")

    monkeypatch.setattr(kalman_residual, "replay_kalman_overlay", unexpected_replay)
    with pytest.raises(KalmanResidualError, match="Heures manquantes=1"):
        build_operational_kalman_view(
            statistics=history,
            source_forecast=pd.DataFrame(),
            covariates=pd.DataFrame(),
            timezone=TIMEZONE,
            delivery_day="2026-09-08",
            training_lookback_days=365,
            rolling_refit_cache_dir=cache_path,
        )
    assert not cache_path.exists()
