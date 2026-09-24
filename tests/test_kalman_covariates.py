from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.kalman_covariates import (
    DerivedCovariateSpec,
    KalmanCovariateConfig,
    KalmanCovariateError,
    materialize_kalman_covariates,
)
from chronos2_hourly.kalman_residual import (
    KalmanResidualConfig,
    KalmanResidualError,
    replay_kalman_overlay,
)


def _weather_contract(*, coverage: float = 0.0) -> KalmanCovariateConfig:
    return KalmanCovariateConfig(
        input_columns=("temperature_fcst", "wind_generation_fcst"),
        groups={
            "market": ("temperature_fcst",),
            "weather": ("temperature_fcst", "heating_degree_18c"),
            "renewables": ("wind_generation_fcst", "wind_ramp_3h"),
            "fundamentals": (
                "temperature_fcst",
                "heating_degree_18c",
                "wind_generation_fcst",
                "wind_ramp_3h",
            ),
        },
        derived=(
            DerivedCovariateSpec(
                name="heating_degree_18c",
                kind="heating_degree",
                sources=("temperature_fcst",),
                threshold=18.0,
            ),
            DerivedCovariateSpec(
                name="wind_ramp_3h",
                kind="ramp",
                sources=("wind_generation_fcst",),
                periods=3,
            ),
        ),
        minimum_history_coverage=coverage,
        require_future_complete=True,
    )


def _hybrid_contract() -> KalmanCovariateConfig:
    return KalmanCovariateConfig(
        input_columns=(
            "residual_load_fcst",
            "temperature_fcst",
            "wind_generation_fcst",
            "ttf_eur_mwh",
            "eua_eur_tco2",
            "ccgt_marginal_cost",
        ),
        groups={
            "market": ("residual_load_fcst",),
            "weather": ("temperature_fcst",),
            "renewables": ("wind_generation_fcst",),
            "fundamentals": (
                "residual_load_fcst",
                "temperature_fcst",
                "wind_generation_fcst",
            ),
            "fuel": (
                "ttf_eur_mwh",
                "eua_eur_tco2",
                "ccgt_marginal_cost",
            ),
            "market_weather": (
                "residual_load_fcst",
                "temperature_fcst",
                "wind_generation_fcst",
            ),
            "market_weather_fuel": (
                "residual_load_fcst",
                "temperature_fcst",
                "wind_generation_fcst",
                "ttf_eur_mwh",
                "eua_eur_tco2",
                "ccgt_marginal_cost",
            ),
        },
        derived=(),
        require_future_complete=True,
    )


def _history_and_covariates(days: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2026-01-01", periods=24 * days, freq="h", tz="UTC")
    hour = np.arange(len(index), dtype=float)
    base = 65.0 + 8.0 * np.sin(2.0 * np.pi * hour / 24.0)
    temperature = 9.0 + 6.0 * np.sin(2.0 * np.pi * hour / (24.0 * 7.0))
    wind = 6.0 + 2.0 * np.cos(2.0 * np.pi * hour / 24.0)
    history = pd.DataFrame(
        {
            "delivery_start_utc": index,
            "actual": base + 0.25 * np.maximum(18.0 - temperature, 0.0),
            "residual_corrected__q10": base - 10.0,
            "residual_corrected__q50": base,
            "residual_corrected__q90": base + 10.0,
            "residual_correction": np.full(len(index), 1.0),
        }
    )
    covariates = pd.DataFrame(
        {
            "timestamp": index,
            "temperature_fcst": temperature,
            "wind_generation_fcst": wind,
        }
    )
    return history, covariates


def test_intraday_ramp_resets_on_a_23_hour_dst_day() -> None:
    local_index = pd.date_range(
        pd.Timestamp("2025-03-30", tz="Europe/Paris"),
        pd.Timestamp("2025-03-31", tz="Europe/Paris"),
        freq="h",
        inclusive="left",
    )
    assert len(local_index) == 23
    raw = pd.DataFrame(
        {
            "temperature_fcst": np.arange(len(local_index), dtype=float),
            "wind_generation_fcst": np.arange(len(local_index), dtype=float),
        },
        index=local_index.tz_convert("UTC"),
    )

    output = materialize_kalman_covariates(
        raw,
        _weather_contract(),
        timezone="Europe/Paris",
    )

    np.testing.assert_allclose(output["wind_ramp_3h"].iloc[:3], 0.0)
    np.testing.assert_allclose(output["wind_ramp_3h"].iloc[3:], 3.0)
    np.testing.assert_allclose(
        output["heating_degree_18c"],
        np.maximum(18.0 - np.arange(len(local_index), dtype=float), 0.0),
    )
    assert output.attrs["structural_ramp_hours"] == {"wind_ramp_3h": 3}


@pytest.mark.parametrize(
    "column",
    [
        "temperature_actual",
        "price_target",
        "wind_oracle",
        "storm_temperature",
        "mkonline_signal",
        "observed_load",
    ],
)
def test_covariate_contract_rejects_leaking_or_competing_inputs(column: str) -> None:
    with pytest.raises(KalmanCovariateError, match="interdite"):
        KalmanCovariateConfig.from_mapping(
            {
                "input_columns": [column],
                "groups": {"market": [column]},
                "derived": {},
            }
        )


def test_linear_weather_uses_only_the_weather_group() -> None:
    history, covariates = _history_and_covariates()
    config = KalmanResidualConfig(
        candidate_kinds=("linear_weather",),
        governance_lookback_days=5,
        governance_minimum_days=2,
    )
    contract = _weather_contract()
    first = replay_kalman_overlay(
        history,
        timezone="UTC",
        evaluation_start_day="2026-01-05",
        covariates=covariates,
        config=config,
        covariate_config=contract,
    )
    perturbed = covariates.copy()
    perturbed["wind_generation_fcst"] += np.linspace(0.0, 1000.0, len(perturbed))
    second = replay_kalman_overlay(
        history,
        timezone="UTC",
        evaluation_start_day="2026-01-05",
        covariates=perturbed,
        config=config,
        covariate_config=contract,
    )

    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert first.audit["candidate_feature_columns"]["linear_weather"] == [
        "base_level",
        "interval_width",
        "residual_shift",
        "covariate::temperature_fcst",
        "covariate::heating_degree_18c",
    ]
    assert first.audit["candidate_feature_counts"]["linear_weather"] == 8


@pytest.mark.parametrize(
    ("kind", "group"),
    [
        ("linear_fuel", "fuel"),
        ("linear_market_weather", "market_weather"),
        ("linear_market_weather_fuel", "market_weather_fuel"),
    ],
)
def test_hybrid_candidates_use_only_their_configured_group(
    kind: str,
    group: str,
) -> None:
    history, weather = _history_and_covariates()
    position = np.arange(len(weather), dtype=float)
    covariates = weather.assign(
        residual_load_fcst=40.0 + 0.02 * position,
        ttf_eur_mwh=30.0 + 0.01 * position,
        eua_eur_tco2=80.0 + 0.005 * position,
        ccgt_marginal_cost=70.0 + 0.015 * position,
    )
    contract = _hybrid_contract()
    config = KalmanResidualConfig(
        candidate_kinds=(kind,),
        governance_lookback_days=5,
        governance_minimum_days=2,
    )

    result = replay_kalman_overlay(
        history,
        timezone="UTC",
        evaluation_start_day="2026-01-05",
        covariates=covariates,
        config=config,
        covariate_config=contract,
    )

    expected = [
        "base_level",
        "interval_width",
        "residual_shift",
        *(f"covariate::{column}" for column in contract.groups[group]),
    ]
    assert result.audit["candidate_feature_columns"][kind] == expected
    assert result.audit["candidate_feature_counts"][kind] == 6 + len(
        contract.groups[group]
    )


def test_history_coverage_threshold_fails_closed() -> None:
    history, covariates = _history_and_covariates()
    covariates.loc[10, "temperature_fcst"] = np.nan
    config = KalmanResidualConfig(
        candidate_kinds=("linear_weather",),
        governance_lookback_days=5,
        governance_minimum_days=2,
    )

    with pytest.raises(KalmanResidualError, match="Couverture historique"):
        replay_kalman_overlay(
            history,
            timezone="UTC",
            evaluation_start_day="2026-01-05",
            covariates=covariates,
            config=config,
            covariate_config=_weather_contract(coverage=1.0),
        )
