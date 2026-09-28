"""CPU Test2 and prior90 behavior without historical archives or provider I/O."""
from datetime import timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import solar_wind_scarcity_regime as historical_test2
from chronos2_hourly.nyx_annual_test2_cpu import fit_test2_origin
from chronos2_hourly.nyx_annual_equal_ensemble import combine_equal_ensemble
from chronos2_hourly.nyx_annual_prior90_cpu import (
    FEATURES, apply_prior90_daily, select_prior90_policy,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "runs/experiments/nyx_improvement_to20260923"


@pytest.mark.parametrize("zone", ["FR", "BE", "NL"])
def test_equal_ensemble_matches_archived_point_when_available(zone):
    composition = ARCHIVE / "price_composition_ensemble_means/oof" / f"{zone}.parquet"
    producers = [
        ARCHIVE / "price_models/hist_residual_400" / f"{zone}_oof.parquet",
        ARCHIVE / "price_models/hist_absolute_400" / f"{zone}_oof.parquet",
        ARCHIVE / "price_models_augmented/hist_residual_400_v1" / f"{zone}_oof.parquet",
    ]
    if not (composition.is_file() and all(path.is_file() for path in producers)):
        pytest.skip("Ignored annual research artifacts unavailable in a clean clone")
    frame = pd.read_parquet(composition)
    residual, absolute, augmented = [pd.read_parquet(path).point for path in producers]
    output = combine_equal_ensemble(
        nyx=frame.nyx_raw, test2=frame.test2_raw,
        price_residual=residual, price_absolute=absolute,
        augmented_residual=augmented)
    assert np.array_equal(output.to_numpy(),
                          frame["three_hgb_mean__clip20__w0p75"].to_numpy())


def test_weekly_test2_fit_matches_original_numerical_fit_and_daily_cutoff():
    rng = np.random.default_rng(20260928)
    origin = pd.Timestamp("2026-09-23")
    train_days = pd.date_range(origin - pd.Timedelta(days=100),
                               origin - pd.Timedelta(days=1), freq="D")
    days = np.repeat(train_days, 24)
    count = len(days)
    baseline = rng.normal(80, 25, count)
    train = pd.DataFrame({
        "own_wind_gw": rng.uniform(0, 15, count),
        "own_residual_stress": rng.normal(0, 1, count),
        "country_is_fr": rng.integers(0, 2, count).astype(float),
        "baseline_p50": baseline,
    })
    spike = rng.random(count) < .11
    residual = 10 * train.own_residual_stress.to_numpy() + rng.normal(0, 6, count)
    residual[spike] += 90
    index = pd.date_range("2026-09-23", "2026-09-24", tz="Europe/Paris",
                          freq="h", inclusive="left").tz_convert("UTC")
    test = pd.DataFrame({
        "own_wind_gw": rng.uniform(0, 15, len(index)),
        "own_residual_stress": rng.normal(0, 1, len(index)),
        "country_is_fr": np.ones(len(index)),
        "baseline_p50": rng.normal(80, 25, len(index)),
    }, index=index)
    frozen = fit_test2_origin(train, residual, days, pair=("BE", "FR"),
                              origin_day=str(origin.date()), iterations=5)
    cutoff = pd.Timestamp("2026-09-22 08:00", tz="Europe/Paris").tz_convert("UTC")
    output = frozen.predict_day(test, test.baseline_p50.rename("nyx__q50"),
                                zone="FR", forecast_issued_at_utc=cutoff)
    original_quantiles, original_probability, _ = historical_test2.fit_predict(
        train, residual, test, days, origin_day=str(origin.date()),
        threads=1, iterations=5)
    assert np.allclose(output.spike_probability, original_probability, rtol=0, atol=1e-14)
    assert np.allclose(output[["test2__q10", "test2__q50", "test2__q90"]].to_numpy(),
                       test.baseline_p50.to_numpy()[:, None] + original_quantiles,
                       rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="D-1"):
        frozen.predict_day(test, test.baseline_p50, zone="FR",
                           forecast_issued_at_utc=cutoff + pd.Timedelta(seconds=1))
    with pytest.raises(ValueError, match="physical hours"):
        frozen.predict_day(test.iloc[:-1], test.baseline_p50.iloc[:-1],
                           zone="FR", forecast_issued_at_utc=cutoff)


def _synthetic_prior90():
    origin = pd.Timestamp("2026-09-23").date()
    index = pd.date_range(str(origin - timedelta(days=90)), str(origin),
                          tz="Europe/Paris", freq="h", inclusive="left").tz_convert("UTC")
    hour = index.tz_convert("Europe/Paris").hour
    active = (hour == 18) | (hour == 19)
    base = np.full(len(index), 100.)
    test2 = np.where(active, 180., 100.)
    frame = pd.DataFrame({
        "ensemble__q50": base,
        "nyx__q50": np.full(len(index), 160.),
        "test2__q50": test2,
        "own_joint_deficit": active.astype(float),
        "own_residual_stress": np.zeros(len(index)),
        "nyx_daily_peak_gap": np.zeros(len(index)),
        "spike_probability": np.full(len(index), .96),
    }, index=index).loc[:, FEATURES]
    actual = pd.Series(np.where(active, 180., 100.), index=index, name="actual")
    return frame, actual, origin


def test_prior90_uses_only_past_and_applies_daily():
    frame, actual, origin = _synthetic_prior90()
    policy = select_prior90_policy(frame, actual, origin_day=str(origin), zone="FR")
    assert policy["selected_rule_index"] is not None
    next_day = origin
    index = pd.date_range(str(next_day), str(next_day + timedelta(days=1)),
                          tz="Europe/Paris", freq="h", inclusive="left").tz_convert("UTC")
    current = frame.iloc[:len(index)].copy()
    current.index = index
    cutoff = pd.Timestamp(f"{next_day - timedelta(days=1)} 08:00",
                          tz="Europe/Paris").tz_convert("UTC")
    output = apply_prior90_daily(current, policy, forecast_issued_at_utc=cutoff)
    assert output.prior90_active.sum() > 0
    assert (output.loc[output.prior90_active, "scarcity_guarded_prior90"]
            == current.loc[output.prior90_active, "test2__q50"]).all()
    with pytest.raises(ValueError, match="D-1"):
        apply_prior90_daily(current, policy, forecast_issued_at_utc=cutoff + pd.Timedelta(seconds=1))
    with pytest.raises(ValueError, match="90 strictly"):
        select_prior90_policy(frame.iloc[1:], actual.iloc[1:], origin_day=str(origin), zone="FR")


@pytest.mark.parametrize("zone,pair", [("FR", "BE_FR"), ("BE", "BE_FR"),
                                      ("NL", "DE_NL")])
def test_archived_prior90_policy_matches_selected_rule_when_available(zone, pair):
    source = ROOT / "runs/experiments/nyx_local_365_to20260923/test2" / pair / "all_oof_predictions.parquet"
    ensemble = ARCHIVE / "price_composition_ensemble_means/oof" / f"{zone}.parquet"
    audit = ARCHIVE / "test2_spike_gate_prior90_v1/audits" / f"{zone}.json"
    if not (source.is_file() and ensemble.is_file() and audit.is_file()):
        pytest.skip("Ignored annual research artifacts unavailable in a clean clone")
    panel = pd.read_parquet(source)
    panel = panel.loc[panel.zone == zone].sort_index()
    ensemble_point = pd.read_parquet(ensemble)["three_hgb_mean__clip20__w0p75"]
    panel["ensemble__q50"] = ensemble_point.loc[panel.index].to_numpy()
    features = panel.loc[:, FEATURES]
    policies = json.loads(audit.read_text())["policies"]
    for archived in (policies[0], policies[len(policies) // 2], policies[-1]):
        origin = pd.Timestamp(archived["origin_day"]).date()
        past = pd.date_range(str(origin - timedelta(days=90)), str(origin),
                             tz={"FR": "Europe/Paris", "BE": "Europe/Brussels",
                                 "NL": "Europe/Amsterdam"}[zone],
                             freq="h", inclusive="left").tz_convert("UTC")
        rebuilt = select_prior90_policy(features.loc[past], panel.actual.loc[past],
                                        origin_day=str(origin), zone=zone)
        assert rebuilt["selected_rule_index"] == archived["selected_rule_index"]
        assert rebuilt["selected_rule"] == archived["selected_rule"]
