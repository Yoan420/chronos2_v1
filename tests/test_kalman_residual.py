from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.kalman_residual import (
    REQUIRED_MARKET_COVARIATES,
    KalmanReplayResult,
    KalmanResidualConfig,
    KalmanResidualError,
    _governance_choice,
    build_operational_kalman_view,
    replay_kalman_overlay,
)


def _history(
    *,
    days: int = 20,
    timezone: str = "Europe/Paris",
    start: str = "2026-01-01",
    bias: np.ndarray | None = None,
) -> pd.DataFrame:
    local_start = pd.Timestamp(start, tz=timezone)
    local_end = local_start + pd.DateOffset(days=days)
    index = pd.date_range(
        local_start,
        local_end,
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    local_day = pd.Index(index.tz_convert(timezone).date)
    unique_days = list(dict.fromkeys(local_day))
    day_position = {day: position for position, day in enumerate(unique_days)}
    if bias is None:
        bias = np.linspace(0.0, 8.0, len(unique_days))
    hourly_bias = np.asarray([bias[day_position[day]] for day in local_day])
    base = 50.0 + 5.0 * np.sin(2.0 * np.pi * index.hour / 24.0)
    residual_shift = 2.0 + np.cos(2.0 * np.pi * index.hour / 24.0)
    return pd.DataFrame(
        {
            "delivery_start_utc": index,
            "actual": base + hourly_bias,
            "chronos2__q50": base - residual_shift,
            "residual_correction": residual_shift,
            "residual_corrected__q10": base - 7.0,
            "residual_corrected__q50": base,
            "residual_corrected__q90": base + 9.0,
            "forecast_origin_utc": index - pd.Timedelta(hours=16),
        }
    )


def _config(*kinds: str) -> KalmanResidualConfig:
    return KalmanResidualConfig(
        governance_lookback_days=5,
        governance_minimum_days=2,
        minimum_gain_eur_mwh=0.001,
        minimum_relative_gain=0.0,
        candidate_kinds=tuple(kinds or ("linear_bias",)),
    )


def _governance_history(
    *,
    selection_actual: float,
    confirmation_actual: float,
    selection_days: int = 8,
    confirmation_days: int = 2,
) -> pd.DataFrame:
    days = pd.date_range(
        "2026-01-01",
        periods=selection_days + confirmation_days,
        freq="D",
    ).date
    return pd.DataFrame(
        {
            "local_day": days,
            "actual": [selection_actual] * selection_days
            + [confirmation_actual] * confirmation_days,
            "base": 0.0,
            "raw::linear_bias": 10.0,
        }
    )


def _confirmation_config(days: int) -> KalmanResidualConfig:
    return KalmanResidualConfig(
        governance_lookback_days=10,
        governance_minimum_days=4,
        governance_confirmation_days=days,
        governance_weight_step=1.0,
        minimum_gain_eur_mwh=0.1,
        minimum_relative_gain=0.0,
        candidate_kinds=("linear_bias",),
    )


def test_governance_confirmation_rejects_an_unstable_selection_winner() -> None:
    realised = _governance_history(
        selection_actual=10.0,
        confirmation_actual=-10.0,
    )

    kind, weight, losses = _governance_choice(
        realised,
        candidate_kinds=("linear_bias",),
        config=_confirmation_config(2),
    )

    assert (kind, weight) == ("identity", 0.0)
    assert losses["identity"] == 10.0
    assert losses["linear_bias"] == 0.0
    assert losses["confirmation::identity"] == 10.0
    assert losses["confirmation::linear_bias"] == 20.0
    assert losses["confirmation_pass::linear_bias"] == 0.0


def test_governance_confirmation_accepts_a_stable_gain() -> None:
    realised = _governance_history(
        selection_actual=10.0,
        confirmation_actual=10.0,
    )

    kind, weight, losses = _governance_choice(
        realised,
        candidate_kinds=("linear_bias",),
        config=_confirmation_config(2),
    )

    assert (kind, weight) == ("linear_bias", 1.0)
    assert losses["confirmation::identity"] == 10.0
    assert losses["confirmation::linear_bias"] == 0.0
    assert losses["confirmation_gain::linear_bias"] == 10.0
    assert losses["confirmation_pass::linear_bias"] == 1.0


def test_zero_confirmation_preserves_the_legacy_governance_exactly() -> None:
    realised = _governance_history(
        selection_actual=10.0,
        confirmation_actual=10.0,
    )

    kind, weight, losses = _governance_choice(
        realised,
        candidate_kinds=("linear_bias",),
        config=_confirmation_config(0),
    )

    assert (kind, weight) == ("linear_bias", 1.0)
    assert losses == {"identity": 10.0, "linear_bias": 0.0}


@pytest.mark.parametrize("confirmation", [-1, 1.5, True, 4, 10])
def test_governance_confirmation_days_rejects_incoherent_values(
    confirmation: object,
) -> None:
    config = KalmanResidualConfig(
        governance_lookback_days=10,
        governance_minimum_days=4,
        governance_confirmation_days=confirmation,  # type: ignore[arg-type]
    )

    with pytest.raises(KalmanResidualError, match="confirmation_days"):
        config.validate()


@pytest.mark.parametrize("training_lookback_days", [None, 5])
def test_candidate_and_confirmation_diagnostics_are_audited_for_every_replay(
    training_lookback_days: int | None,
) -> None:
    full = _history(days=11, bias=np.full(11, 10.0, dtype=float))
    local_days = pd.to_datetime(
        full["delivery_start_utc"], utc=True
    ).dt.tz_convert("Europe/Paris").dt.date
    future_day = local_days.max()
    history = full.loc[local_days < future_day].copy()
    future = full.loc[local_days == future_day].copy()
    config = KalmanResidualConfig(
        governance_lookback_days=5,
        governance_minimum_days=4,
        governance_confirmation_days=2,
        governance_weight_step=0.5,
        minimum_gain_eur_mwh=0.001,
        minimum_relative_gain=0.0,
        candidate_kinds=("linear_bias",),
    )

    result = replay_kalman_overlay(
        history,
        timezone="Europe/Paris",
        evaluation_start_day="2026-01-06",
        future_upstream=future,
        training_lookback_days=training_lookback_days,
        config=config,
    )

    daily = result.daily_audit.iloc[-1]
    assert set(daily["candidate_trailing_mae"]) == {
        "identity",
        "linear_bias",
    }
    assert "confirmation::linear_bias" in daily["governance_diagnostics"]
    assert set(result.future_audit["candidate_trailing_mae"]) == {
        "identity",
        "linear_bias",
    }
    assert (
        "confirmation::linear_bias"
        in result.future_audit["governance_diagnostics"]
    )


def _split_history_and_future(
    *,
    history_days: int,
    start: str,
    timezone: str = "Europe/Paris",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = _history(
        days=history_days + 1,
        start=start,
        timezone=timezone,
    )
    local_days = pd.to_datetime(
        full["delivery_start_utc"], utc=True
    ).dt.tz_convert(timezone).dt.date
    future_day = local_days.max()
    return (
        full.loc[local_days < future_day].copy(),
        full.loc[local_days == future_day].copy(),
    )


def test_replay_is_strictly_causal_under_future_actual_perturbation() -> None:
    history = _history(days=14)
    changed = history.copy()
    local = pd.to_datetime(changed["delivery_start_utc"], utc=True).dt.tz_convert(
        "Europe/Paris"
    )
    perturbed_day = sorted(pd.Index(local.dt.date).unique())[7]
    changed.loc[local.dt.date == perturbed_day, "actual"] += 500.0

    original_result = replay_kalman_overlay(
        history,
        timezone="Europe/Paris",
        evaluation_start_day="2026-01-04",
        config=_config("linear_bias"),
    )
    changed_result = replay_kalman_overlay(
        changed,
        timezone="Europe/Paris",
        evaluation_start_day="2026-01-04",
        config=_config("linear_bias"),
    )
    original = original_result.predictions["residual_kalman__q50"]
    alternative = changed_result.predictions["residual_kalman__q50"]
    local_index = original.index.tz_convert("Europe/Paris")
    through_perturbed_day = local_index.date <= perturbed_day

    np.testing.assert_allclose(
        original.loc[through_perturbed_day],
        alternative.loc[through_perturbed_day],
        rtol=0.0,
        atol=0.0,
    )
    assert not np.allclose(
        original.loc[~through_perturbed_day],
        alternative.loc[~through_perturbed_day],
    )
    audit = original_result.daily_audit
    has_observation = audit["last_observation_used"].notna()
    assert (
        pd.to_datetime(audit.loc[has_observation, "last_observation_used"]).dt.date
        < pd.to_datetime(audit.loc[has_observation, "local_day"]).dt.date
    ).all()
    assert original_result.audit["causality_violations"] == 0


def test_same_shift_preserves_width_order_and_dst_days() -> None:
    history = _history(days=5, start="2025-03-28")
    result = replay_kalman_overlay(
        history,
        timezone="Europe/Paris",
        evaluation_start_day="2025-03-30",
        config=_config("linear_bias", "linear_harmonic"),
    )
    source = history.copy()
    source.index = pd.to_datetime(source["delivery_start_utc"], utc=True)
    predicted = result.predictions

    np.testing.assert_allclose(
        predicted["residual_kalman__q90"]
        - predicted["residual_kalman__q10"],
        source.loc[predicted.index, "residual_corrected__q90"]
        - source.loc[predicted.index, "residual_corrected__q10"],
    )
    assert bool(
        (
            predicted["residual_kalman__q10"]
            <= predicted["residual_kalman__q50"]
        ).all()
    )
    assert bool(
        (
            predicted["residual_kalman__q50"]
            <= predicted["residual_kalman__q90"]
        ).all()
    )
    assert 23 in set(result.daily_audit["hours"])
    assert result.audit["quantile_crossings"] == 0


def test_identity_gate_keeps_an_already_exact_forecast_unchanged() -> None:
    history = _history(days=10, bias=np.zeros(10, dtype=float))
    result = replay_kalman_overlay(
        history,
        timezone="Europe/Paris",
        evaluation_start_day="2026-01-04",
        config=_config("linear_bias", "linear_market", "ukf_scale"),
    )
    source = history.copy()
    source.index = pd.to_datetime(source["delivery_start_utc"], utc=True)

    np.testing.assert_allclose(
        result.predictions["residual_kalman__q50"],
        source.loc[result.predictions.index, "residual_corrected__q50"],
        rtol=0.0,
        atol=1e-12,
    )
    assert set(result.daily_audit["selected_filter"]) == {"identity"}


def test_future_forecast_uses_history_only_and_ignores_future_actual() -> None:
    history, future = _split_history_and_future(
        history_days=12,
        start="2026-01-01",
    )
    changed_future = future.copy()
    changed_future["actual"] += 10_000.0

    history_only = replay_kalman_overlay(
        history,
        timezone="Europe/Paris",
        evaluation_start_day="2026-01-04",
        config=_config("linear_bias", "linear_harmonic"),
    )
    original = replay_kalman_overlay(
        history,
        timezone="Europe/Paris",
        evaluation_start_day="2026-01-04",
        future_upstream=future,
        config=_config("linear_bias", "linear_harmonic"),
    )
    changed = replay_kalman_overlay(
        history,
        timezone="Europe/Paris",
        evaluation_start_day="2026-01-04",
        future_upstream=changed_future,
        config=_config("linear_bias", "linear_harmonic"),
    )

    pd.testing.assert_frame_equal(
        history_only.predictions,
        original.predictions,
    )
    pd.testing.assert_frame_equal(
        original.future_predictions,
        changed.future_predictions,
    )
    pd.testing.assert_frame_equal(
        original.future_candidate_predictions,
        changed.future_candidate_predictions,
    )
    last_history_day = pd.to_datetime(
        history["delivery_start_utc"], utc=True
    ).dt.tz_convert("Europe/Paris").dt.date.max()
    assert original.future_audit["last_observation_used"] == str(
        last_history_day
    )
    assert original.future_audit["observations_assimilated"] == 0
    assert original.future_audit["actual_column_ignored"] is True
    assert original.audit["future_observations_assimilated"] == 0
    assert original.audit["last_observation_assimilated_day"] == str(
        last_history_day
    )


def test_future_same_shift_preserves_quantiles_on_dst_day() -> None:
    history, future = _split_history_and_future(
        history_days=2,
        start="2025-03-28",
    )
    result = replay_kalman_overlay(
        history,
        timezone="Europe/Paris",
        evaluation_start_day="2025-03-29",
        future_upstream=future.drop(columns="actual"),
        config=_config("linear_bias", "linear_scale", "ukf_scale"),
    )
    source = future.copy()
    source.index = pd.to_datetime(source["delivery_start_utc"], utc=True)
    predicted = result.future_predictions

    assert len(predicted) == 23
    np.testing.assert_allclose(
        predicted["residual_kalman__q90"]
        - predicted["residual_kalman__q10"],
        source.loc[predicted.index, "residual_corrected__q90"]
        - source.loc[predicted.index, "residual_corrected__q10"],
        rtol=0.0,
        atol=1e-12,
    )
    assert (
        predicted["residual_kalman__q10"]
        <= predicted["residual_kalman__q50"]
    ).all()
    assert (
        predicted["residual_kalman__q50"]
        <= predicted["residual_kalman__q90"]
    ).all()
    assert result.future_audit["hours"] == 23
    assert result.future_audit["actual_column_ignored"] is False


@pytest.mark.parametrize("kind", ["linear_scale", "ekf_scale", "ukf_scale"])
def test_scale_filters_remain_finite_and_audited(kind: str) -> None:
    history = _history(days=9)
    result = replay_kalman_overlay(
        history,
        timezone="Europe/Paris",
        evaluation_start_day="2026-01-04",
        config=_config(kind),
    )

    assert np.isfinite(
        result.predictions.filter(like="residual_kalman__").to_numpy(dtype=float)
    ).all()
    assert np.isfinite(
        result.state_audit["minimum_covariance_eigenvalue"].to_numpy(dtype=float)
    ).all()
    assert result.audit["smoother_used"] is False
    assert result.audit["em_used"] is False
    assert result.audit["pykalman_version"] == "0.11.2"


def test_rejects_missing_warmup_and_crossed_upstream_quantiles() -> None:
    history = _history(days=4)
    with pytest.raises(KalmanResidualError, match="warm-up"):
        replay_kalman_overlay(
            history,
            timezone="Europe/Paris",
            evaluation_start_day="2025-01-01",
            config=_config("linear_bias"),
        )
    crossed = history.copy()
    crossed["residual_corrected__q10"] = crossed["residual_corrected__q90"] + 1.0
    with pytest.raises(KalmanResidualError, match="croisent"):
        replay_kalman_overlay(
            crossed,
            timezone="Europe/Paris",
            evaluation_start_day="2026-01-03",
            config=_config("linear_bias"),
        )


def _operational_inputs(
    *,
    start: str = "2025-08-14",
    delivery_day: str = "2026-08-28",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = _history(days=380, start=start)
    local_days = pd.to_datetime(
        full["delivery_start_utc"], utc=True
    ).dt.tz_convert("Europe/Paris").dt.date
    requested_day = pd.Timestamp(delivery_day).date()
    forecast = full.loc[local_days == requested_day].drop(columns="actual").copy()
    covariates = pd.DataFrame({"timestamp": full["delivery_start_utc"]})
    for position, column in enumerate(REQUIRED_MARKET_COVARIATES):
        covariates[column] = 20.0 + position
    return full, forecast, covariates


def _identity_operational_replay(captured: dict[str, object] | None = None):
    def fake_replay(
        history,
        *,
        upstream_model,
        output_model,
        covariates,
        future_upstream,
        future_covariates,
        **_kwargs,
    ):
        history_index = pd.DatetimeIndex(
            pd.to_datetime(history["delivery_start_utc"], utc=True)
        )
        future_index = pd.DatetimeIndex(
            pd.to_datetime(future_upstream["delivery_start_utc"], utc=True)
        )
        if captured is not None:
            captured["last_history_day"] = history_index.tz_convert(
                "Europe/Paris"
            ).date.max()
            captured["history_columns"] = tuple(history.columns)
            captured["future_has_actual"] = "actual" in future_upstream
            captured["covariate_columns"] = tuple(covariates.columns)
            captured["same_future_covariates"] = future_covariates is covariates
        historical = pd.DataFrame(index=history_index)
        future = pd.DataFrame(index=future_index)
        for quantile in ("q10", "q50", "q90"):
            historical[f"{output_model}__{quantile}"] = history[
                f"{upstream_model}__{quantile}"
            ].to_numpy(dtype=float)
            future[f"{output_model}__{quantile}"] = future_upstream[
                f"{upstream_model}__{quantile}"
            ].to_numpy(dtype=float)
        return KalmanReplayResult(
            predictions=historical,
            candidate_predictions=pd.DataFrame(index=history_index),
            daily_audit=pd.DataFrame(),
            state_audit=pd.DataFrame(),
            audit={},
            future_predictions=future,
            future_candidate_predictions=pd.DataFrame(index=future_index),
            future_audit={},
        )

    return fake_replay


def test_operational_view_excludes_observed_delivery_day_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statistics, forecast, covariates = _operational_inputs()
    statistics["storm_dashboard_official__q50"] = 100.0
    statistics["mkonline_blend__q50"] = 101.0
    covariates["known_fr_residual_load_fcst_oracle"] = 999.0
    covariates["unrelated_feature"] = 999.0
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "chronos2_hourly.kalman_residual.replay_kalman_overlay",
        _identity_operational_replay(captured),
    )

    view = build_operational_kalman_view(
        statistics=statistics,
        source_forecast=forecast,
        covariates=covariates,
        timezone="Europe/Paris",
        delivery_day="2026-08-28",
    )

    assert str(captured["last_history_day"]) == "2026-08-27"
    assert captured["future_has_actual"] is False
    assert "storm_dashboard_official__q50" not in captured["history_columns"]
    assert "mkonline_blend__q50" not in captured["history_columns"]
    assert captured["covariate_columns"] == (
        "timestamp",
        *REQUIRED_MARKET_COVARIATES,
    )
    assert captured["same_future_covariates"] is True
    assert view.evaluation_start_day.isoformat() == "2025-08-28"
    assert view.evaluation_end_day.isoformat() == "2026-08-27"
    assert len(view.backtest) == len(view.evaluation_index) == 8760
    assert len(view.forecast) == 24
    statistics_days = pd.to_datetime(
        view.statistics["delivery_start_utc"], utc=True
    ).dt.tz_convert("Europe/Paris").dt.date
    assert statistics_days.max().isoformat() == "2026-08-28"
    realised = view.statistics.loc[
        statistics_days == pd.Timestamp("2026-08-28").date()
    ]
    np.testing.assert_allclose(
        realised["residual_kalman__q50"],
        view.forecast["residual_kalman__q50"],
        rtol=0.0,
        atol=0.0,
    )


def test_operational_view_preserves_empty_current_actual_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statistics, forecast, covariates = _operational_inputs()
    local_days = pd.to_datetime(
        statistics["delivery_start_utc"], utc=True
    ).dt.tz_convert("Europe/Paris").dt.date
    current = local_days == date(2026, 8, 28)
    statistics.loc[current, "actual"] = np.nan
    monkeypatch.setattr(
        "chronos2_hourly.kalman_residual.replay_kalman_overlay",
        _identity_operational_replay(),
    )

    view = build_operational_kalman_view(
        statistics=statistics,
        source_forecast=forecast,
        covariates=covariates,
        timezone="Europe/Paris",
        delivery_day="2026-08-28",
    )

    output_days = pd.to_datetime(
        view.statistics["delivery_start_utc"], utc=True
    ).dt.tz_convert("Europe/Paris").dt.date
    pending = view.statistics.loc[output_days == date(2026, 8, 28)]
    assert len(pending) == 24
    assert pending["actual"].isna().all()
    assert pending["residual_kalman__q50"].notna().all()


def test_operational_view_rejects_partial_current_actual_curve() -> None:
    statistics, forecast, covariates = _operational_inputs()
    local_days = pd.to_datetime(
        statistics["delivery_start_utc"], utc=True
    ).dt.tz_convert("Europe/Paris").dt.date
    current_indices = statistics.index[local_days == date(2026, 8, 28)]
    statistics.loc[current_indices[-1], "actual"] = np.nan

    with pytest.raises(KalmanResidualError, match="complet ou entierement vide"):
        build_operational_kalman_view(
            statistics=statistics,
            source_forecast=forecast,
            covariates=covariates,
            timezone="Europe/Paris",
            delivery_day="2026-08-28",
        )


@pytest.mark.parametrize(
    ("start", "delivery_day", "expected_hours"),
    (
        ("2025-03-16", "2026-03-30", 8759),
        ("2025-10-12", "2026-10-26", 8761),
    ),
)
def test_operational_final365_uses_dynamic_physical_hours(
    monkeypatch: pytest.MonkeyPatch,
    start: str,
    delivery_day: str,
    expected_hours: int,
) -> None:
    statistics, forecast, covariates = _operational_inputs(
        start=start,
        delivery_day=delivery_day,
    )
    monkeypatch.setattr(
        "chronos2_hourly.kalman_residual.replay_kalman_overlay",
        _identity_operational_replay(),
    )

    view = build_operational_kalman_view(
        statistics=statistics,
        source_forecast=forecast,
        covariates=covariates,
        timezone="Europe/Paris",
        delivery_day=delivery_day,
    )

    assert (view.evaluation_end_day - view.evaluation_start_day).days + 1 == 365
    assert len(view.evaluation_index) == len(view.backtest) == expected_hours


def test_operational_view_requires_all_five_future_market_covariates() -> None:
    statistics, forecast, covariates = _operational_inputs()
    incomplete = covariates.drop(columns=REQUIRED_MARKET_COVARIATES[-1])

    with pytest.raises(KalmanResidualError, match="Covariables marche requises"):
        build_operational_kalman_view(
            statistics=statistics,
            source_forecast=forecast,
            covariates=incomplete,
            timezone="Europe/Paris",
            delivery_day="2026-08-28",
        )

    forecast_index = pd.DatetimeIndex(
        pd.to_datetime(forecast["delivery_start_utc"], utc=True)
    )
    missing_hour = covariates.copy()
    on_forecast = pd.to_datetime(
        missing_hour["timestamp"], utc=True
    ).isin(forecast_index)
    missing_hour.loc[on_forecast, REQUIRED_MARKET_COVARIATES[-1]] = np.nan
    with pytest.raises(KalmanResidualError, match="doivent etre completes"):
        build_operational_kalman_view(
            statistics=statistics,
            source_forecast=forecast,
            covariates=missing_hour,
            timezone="Europe/Paris",
            delivery_day="2026-08-28",
        )


def test_operational_view_requires_the_residual_shift_signal() -> None:
    statistics, forecast, covariates = _operational_inputs()
    statistics = statistics.drop(columns=["residual_correction", "chronos2__q50"])
    forecast = forecast.drop(columns=["residual_correction", "chronos2__q50"])

    with pytest.raises(KalmanResidualError, match="residual_correction"):
        build_operational_kalman_view(
            statistics=statistics,
            source_forecast=forecast,
            covariates=covariates,
            timezone="Europe/Paris",
            delivery_day="2026-08-28",
        )


def test_replay_rejects_an_evaluation_start_after_all_history() -> None:
    with pytest.raises(KalmanResidualError, match="posterieur"):
        replay_kalman_overlay(
            _history(days=20),
            timezone="Europe/Paris",
            evaluation_start_day="2027-01-01",
            config=_config("linear_bias"),
        )
