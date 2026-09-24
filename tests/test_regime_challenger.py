from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.regime_challenger import (
    QUANTILES,
    RESIDUAL_ALIASES,
    RegimeChallengerConfig,
    build_regime_features,
    fit_regime_model,
    load_regime_challenger_config,
    predict_regime_adjustment,
    prequential_regime_predictions,
    previous_local_day_values,
    scheduled_origin_utc,
    select_forecasts_asof,
    summarize_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, **updates: object) -> RegimeChallengerConfig:
    config = load_regime_challenger_config(
        PROJECT_ROOT / "config" / "price_regime_challenger.yaml",
        project_root=PROJECT_ROOT,
    )
    base = replace(
        config,
        source_path=tmp_path / "config.yaml",
        output_root=tmp_path / "challenger",
        n_estimators=12,
        max_depth=4,
        classifier_min_samples_leaf=2,
        regressor_min_samples_leaf=2,
        minimum_training_rows=16,
        minimum_positive_rows=3,
        minimum_training_days=6,
        evaluation_days=8,
        refit_every_days=3,
    )
    return replace(base, **updates)


def _baseline(index: pd.DatetimeIndex, level: float = 60.0) -> pd.DataFrame:
    wave = np.sin(np.arange(len(index)) * 2.0 * np.pi / 24.0)
    return pd.DataFrame(
        {
            "q10": level + wave - 10.0,
            "q50": level + wave,
            "q90": level + wave + 10.0,
        },
        index=index,
    )


def _residual(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert("Europe/Paris")
    result = {}
    for position, alias in enumerate(RESIDUAL_ALIASES):
        result[alias] = (
            20.0
            + position
            + 0.2 * local.hour
            + 0.5 * np.asarray(local.dayofyear)
        )
    return pd.DataFrame(result, index=index)


def test_default_config_is_shadow_and_confined() -> None:
    config = load_regime_challenger_config(
        PROJECT_ROOT / "config" / "price_regime_challenger.yaml",
        project_root=PROJECT_ROOT,
    )
    assert config.challenger_id == "price_regime_shock_v1"
    assert config.label_delay_days == 2
    assert config.gate_probability_threshold == pytest.approx(0.55)
    assert config.output_root.is_relative_to(
        (PROJECT_ROOT / "runs" / "challengers").resolve()
    )


def test_scheduled_origin_is_civil_d_minus_one_0800_across_dst() -> None:
    assert scheduled_origin_utc("2026-03-29") == pd.Timestamp(
        "2026-03-28T07:00:00Z"
    )
    assert scheduled_origin_utc("2026-10-25") == pd.Timestamp(
        "2026-10-24T06:00:00Z"
    )


def test_select_forecasts_asof_uses_latest_eligible_vintage_only() -> None:
    grid = pd.date_range("2026-08-24T22:00:00Z", periods=3, freq="h", tz="UTC")
    cutoff = pd.Timestamp("2026-08-24T06:00:00Z")
    rows = []
    for timestamp in grid:
        rows.extend(
            [
                {
                    "value_time_utc": timestamp,
                    "snapshot_time_utc": cutoff - pd.Timedelta(hours=2),
                    "revision_time_utc": cutoff - pd.Timedelta(hours=1),
                    "value": 10.0,
                },
                {
                    "value_time_utc": timestamp,
                    "snapshot_time_utc": cutoff + pd.Timedelta(minutes=1),
                    "revision_time_utc": cutoff + pd.Timedelta(minutes=1),
                    "value": 999.0,
                },
            ]
        )
    selected, audit = select_forecasts_asof(
        pd.DataFrame(rows), alias=RESIDUAL_ALIASES[0], grid=grid
    )
    assert selected.eq(10.0).all()
    assert audit["cutoff_violations"] == 0
    assert audit["coverage"] == 1.0


def test_previous_local_day_alignment_handles_25_hour_day() -> None:
    index = pd.date_range(
        "2026-10-23T22:00:00Z", "2026-10-25T22:00:00Z", freq="h", tz="UTC"
    )
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    previous = previous_local_day_values(series)
    local = index.tz_convert("Europe/Paris")
    repeated = np.flatnonzero(
        (local.date == date(2026, 10, 25)) & (local.hour == 2)
    )
    assert len(repeated) == 2
    assert np.isfinite(previous.iloc[repeated[0]])
    # D-1 has only one 02:00 occurrence; the second autumn occurrence is
    # intentionally missing rather than borrowing another physical hour.
    assert np.isnan(previous.iloc[repeated[1]])


def test_feature_builder_exposes_fr_solar_day_on_day_jump() -> None:
    index = pd.date_range("2026-08-23T22:00:00Z", periods=48, freq="h", tz="UTC")
    residual = _residual(index)
    local_day = index.tz_convert("Europe/Paris").date
    solar = index.tz_convert("Europe/Paris").hour.isin(range(9, 17))
    residual.loc[(local_day == date(2026, 8, 25)) & solar, RESIDUAL_ALIASES[0]] += 8.0
    features = build_regime_features(residual, _baseline(index))
    day_two = features.loc[
        pd.Series(local_day == date(2026, 8, 25), index=index)
    ]
    assert day_two["fr_residual_load__solar_mean_delta_d1"].dropna().iloc[0] == pytest.approx(8.5)
    assert "fr_vs_neighbours_delta" in features
    assert features["is_solar_hour"].sum() == 16


def test_model_adjusts_only_high_probability_solar_hours(tmp_path: Path) -> None:
    config = _config(tmp_path, gate_probability_threshold=0.25)
    index = pd.date_range("2026-01-01", periods=40 * 24, freq="h", tz="UTC")
    baseline = _baseline(index)
    residual = _residual(index)
    features = build_regime_features(residual, baseline)
    local = index.tz_convert(config.timezone)
    shock = (local.hour >= 11) & (local.hour <= 14) & (local.day % 4 == 0)
    actual = baseline["q50"] + np.where(shock, 55.0, -2.0)
    fitted = fit_regime_model(features, actual, baseline["q50"], config=config)
    predicted = predict_regime_adjustment(fitted, features.tail(48), baseline.tail(48), config=config)
    assert (predicted["shock_premium"] >= 0.0).all()
    assert predicted.loc[
        features.tail(48)["is_solar_hour"].eq(0), "shock_premium"
    ].eq(0.0).all()
    assert (
        predicted["challenger_q10"]
        <= predicted["challenger_q50"]
    ).all()
    assert (
        predicted["challenger_q50"]
        <= predicted["challenger_q90"]
    ).all()
    assert fitted.diagnostics["positive_rows"] >= config.minimum_positive_rows


def test_prequential_blocks_keep_d_minus_two_label_embargo(tmp_path: Path) -> None:
    config = _config(tmp_path, gate_probability_threshold=0.2)
    index = pd.date_range("2026-01-01", periods=24 * 24, freq="h", tz="UTC")
    baseline = _baseline(index)
    features = build_regime_features(_residual(index), baseline)
    local = index.tz_convert(config.timezone)
    actual = baseline["q50"] + np.where(
        (local.hour >= 10) & (local.hour <= 15) & (local.day % 3 == 0),
        45.0,
        0.0,
    )
    predictions, audits = prequential_regime_predictions(
        features, actual, baseline, config=config
    )
    assert not predictions.empty
    assert audits
    for audit in audits:
        assert pd.Timestamp(audit["training_end_day"]) <= (
            pd.Timestamp(audit["block_start_day"]) - pd.Timedelta(days=2)
        )


def test_metric_summary_reports_paired_tail_gain() -> None:
    index = pd.date_range("2026-08-01", periods=24, freq="h", tz="UTC")
    local = index.tz_convert("Europe/Paris")
    baseline = np.full(24, 50.0)
    actual = baseline + np.where(local.hour.isin(range(9, 17)), 40.0, 0.0)
    challenger = baseline + np.where(local.hour.isin(range(9, 17)), 30.0, 0.0)
    hourly = pd.DataFrame(
        {
            "zone": "FR",
            "variant": "autonomous",
            "local_date": local.date.astype(str),
            "local_hour": local.hour,
            "actual": actual,
            "baseline_q10": baseline - 10.0,
            "baseline_q50": baseline,
            "baseline_q90": baseline + 10.0,
            "challenger_q10": challenger - 10.0,
            "challenger_q50": challenger,
            "challenger_q90": challenger + 10.0,
            "shock_probability": np.where(local.hour.isin(range(9, 17)), 0.8, 0.1),
            "shock_premium": challenger - baseline,
            "regime_label": local.hour.isin(range(9, 17)).astype(int),
        }
    )
    metrics, daily = summarize_metrics(hourly, probability_threshold=0.55)
    tail = metrics.loc[metrics["scope"].eq("actual_shock")].iloc[0]
    assert tail["mae_gain"] == pytest.approx(30.0)
    assert tail["regime_recall"] == pytest.approx(1.0)
    # The UTC window crosses local midnight in Europe/Paris.
    assert len(daily) == 2
