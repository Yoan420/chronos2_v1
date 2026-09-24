from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.models.foundation_moe import (
    FoundationMoEConfig,
    FoundationMoEError,
    FoundationMoEForecaster,
)


def _split(start: str, days: int) -> dict[str, object]:
    timestamps: list[pd.Timestamp] = []
    markets: list[str] = []
    horizons: list[int] = []
    curves: list[str] = []
    actual: list[float] = []
    autonomous: list[float] = []
    chronos: list[float] = []
    ensemble: list[float] = []
    start_timestamp = pd.Timestamp(start, tz="UTC")
    horizon_bias = np.asarray([4.0, -3.0, 2.0, -1.0])
    for day in range(days):
        for horizon in range(4):
            timestamp = start_timestamp + pd.Timedelta(days=day, hours=horizon)
            for market, market_offset in (("FR", 0.0), ("DE", 6.0)):
                value = (
                    50.0
                    + market_offset
                    + 5.0 * np.sin(2.0 * np.pi * horizon / 4.0)
                    + 0.15 * day
                )
                timestamps.append(timestamp)
                markets.append(market)
                horizons.append(horizon)
                curves.append(f"{market}:{timestamp.date()}")
                actual.append(value)
                autonomous.append(value + horizon_bias[horizon])
                chronos.append(value + 0.7 * horizon_bias[horizon])
                ensemble.append(value - 0.4 * horizon_bias[horizon])
    index = pd.DatetimeIndex(timestamps, name="delivery_start_utc")
    experts = pd.DataFrame(
        {
            "autonomous": autonomous,
            "chronos2": chronos,
            "ensemble": ensemble,
        },
        index=index,
    )
    anchor = pd.DataFrame(
        {
            "q10": np.asarray(autonomous) - 8.0,
            "q50": autonomous,
            "q90": np.asarray(autonomous) + 11.0,
        },
        index=index,
    )
    return {
        "experts": experts,
        "anchor": anchor,
        "target": pd.Series(actual, index=index, name="actual"),
        "markets": pd.Series(markets, index=index),
        "horizons": pd.Series(horizons, index=index),
        "curves": pd.Series(curves, index=index),
    }


@pytest.fixture(scope="module")
def fitted_model() -> tuple[FoundationMoEForecaster, dict[str, object]]:
    train = _split("2025-01-01", 24)
    validation = _split("2025-01-25", 8)
    test = _split("2025-02-02", 5)
    config = FoundationMoEConfig(
        hidden_size=24,
        market_embedding_dim=3,
        horizon_embedding_dim=3,
        epochs=45,
        min_epochs=12,
        early_stopping_patience=8,
        learning_rate=0.005,
        tail_weight=0.0,
        underprediction_weight=0.0,
        validation_tail_weight=0.0,
        minimum_training_tokens=64,
        minimum_validation_tokens=16,
        initial_gain=0.2,
        seed=7,
        device="cpu",
    )
    model = FoundationMoEForecaster(config).fit(
        train["experts"],
        train["target"],
        train["anchor"],
        train["markets"],
        train["horizons"],
        train["curves"],
        validation_experts=validation["experts"],
        validation_target=validation["target"],
        validation_anchor_quantiles=validation["anchor"],
        validation_markets=validation["markets"],
        validation_horizon_tokens=validation["horizons"],
        validation_curve_ids=validation["curves"],
        anchor_expert="autonomous",
    )
    return model, test


def _predict(
    model: FoundationMoEForecaster,
    split: dict[str, object],
    *,
    mode: str = "paper_balanced",
):
    return model.predict(
        split["experts"],
        split["anchor"],
        split["markets"],
        split["horizons"],
        split["curves"],
        mode=mode,
    )


def test_horizon_anchor_calibration_reduces_mae_and_preserves_quantiles(
    fitted_model,
) -> None:
    model, test = fitted_model
    forecast = _predict(model, test, mode="anchor_bias")
    target = test["target"].to_numpy(dtype=float)
    anchor = test["anchor"]
    before = np.mean(np.abs(anchor["q50"].to_numpy() - target))
    after = np.mean(np.abs(forecast.predictions["q50"].to_numpy() - target))
    assert after < before * 0.05
    np.testing.assert_allclose(
        forecast.predictions["q50"] - forecast.predictions["q10"],
        anchor["q50"] - anchor["q10"],
    )
    np.testing.assert_allclose(
        forecast.predictions["q90"] - forecast.predictions["q50"],
        anchor["q90"] - anchor["q50"],
    )
    assert (forecast.predictions["q10"] <= forecast.predictions["q50"]).all()
    assert (forecast.predictions["q50"] <= forecast.predictions["q90"]).all()


def test_top2_routing_is_sparse_normalized_and_modes_are_finite(fitted_model) -> None:
    model, test = fitted_model
    balanced = _predict(model, test, mode="paper_balanced")
    downward = _predict(model, test, mode="mae_downward_only")
    weights = balanced.diagnostics[["top1_weight", "top2_weight"]]
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1.0e-6)
    assert balanced.diagnostics["top1_expert"].notna().all()
    assert balanced.diagnostics["top2_expert"].notna().all()
    assert np.isfinite(balanced.predictions.to_numpy()).all()
    assert np.isfinite(downward.predictions.to_numpy()).all()


def test_checkpoint_round_trip_is_prediction_identical(
    fitted_model,
    tmp_path,
) -> None:
    model, test = fitted_model
    expected = _predict(model, test)
    path = model.save(tmp_path / "foundation_moe.pt")
    restored = FoundationMoEForecaster.load(path, device="cpu")
    actual = _predict(restored, test)
    np.testing.assert_allclose(
        actual.predictions.to_numpy(),
        expected.predictions.to_numpy(),
        rtol=0.0,
        atol=0.0,
    )
    pd.testing.assert_frame_equal(actual.diagnostics, expected.diagnostics)


def test_fit_rejects_non_chronological_validation() -> None:
    train = _split("2025-01-01", 6)
    validation = _split("2025-01-04", 3)
    config = FoundationMoEConfig(
        epochs=2,
        min_epochs=1,
        early_stopping_patience=1,
        minimum_training_tokens=16,
        minimum_validation_tokens=8,
    )
    with pytest.raises(FoundationMoEError, match="strictement avant"):
        FoundationMoEForecaster(config).fit(
            train["experts"],
            train["target"],
            train["anchor"],
            train["markets"],
            train["horizons"],
            train["curves"],
            validation_experts=validation["experts"],
            validation_target=validation["target"],
            validation_anchor_quantiles=validation["anchor"],
            validation_markets=validation["markets"],
            validation_horizon_tokens=validation["horizons"],
            validation_curve_ids=validation["curves"],
            anchor_expert="autonomous",
        )


def test_predict_rejects_unknown_market_and_horizon(fitted_model) -> None:
    model, test = fitted_model
    unknown_market = test["markets"].copy()
    unknown_market.iloc[0] = "ES"
    with pytest.raises(FoundationMoEError, match="marches inconnus"):
        model.predict(
            test["experts"],
            test["anchor"],
            unknown_market,
            test["horizons"],
            test["curves"],
        )
    unknown_horizon = test["horizons"].copy()
    unknown_horizon.iloc[0] = 9
    with pytest.raises(FoundationMoEError, match="horizons jamais vus"):
        model.predict(
            test["experts"],
            test["anchor"],
            test["markets"],
            unknown_horizon,
            test["curves"],
        )


def test_top1_ablation_has_one_active_expert() -> None:
    train = _split("2025-03-01", 8)
    validation = _split("2025-03-09", 3)
    test = _split("2025-03-12", 2)
    config = replace(
        FoundationMoEConfig(),
        top_k=1,
        hidden_size=12,
        market_embedding_dim=2,
        horizon_embedding_dim=2,
        epochs=3,
        min_epochs=1,
        early_stopping_patience=1,
        minimum_training_tokens=32,
        minimum_validation_tokens=16,
        device="cpu",
    )
    model = FoundationMoEForecaster(config).fit(
        train["experts"],
        train["target"],
        train["anchor"],
        train["markets"],
        train["horizons"],
        train["curves"],
        validation_experts=validation["experts"],
        validation_target=validation["target"],
        validation_anchor_quantiles=validation["anchor"],
        validation_markets=validation["markets"],
        validation_horizon_tokens=validation["horizons"],
        validation_curve_ids=validation["curves"],
        anchor_expert="autonomous",
    )
    result = _predict(model, test)
    np.testing.assert_allclose(result.diagnostics["top1_weight"], 1.0)
    assert result.diagnostics["top2_expert"].isna().all()
    assert result.diagnostics["top2_weight"].isna().all()
