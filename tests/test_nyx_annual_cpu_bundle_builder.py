from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_cpu_bundle_builder as builder
from chronos2_hourly.nyx_local_extra_features import FUEL_COLUMNS, FUEL_TIME_COLUMNS, cutoff_times


def inputs():
    index = pd.date_range("2026-10-05", "2026-10-27", freq="h", inclusive="left",
                          tz="Europe/Paris").tz_convert("UTC")
    # Includes a 25-hour autumn day and genuinely absent forecast-day labels.
    day = "2026-10-26"
    origins = cutoff_times(index, "Europe/Paris")
    t = np.arange(len(index), dtype=float)
    covariates = pd.DataFrame({name: 20. + np.sin(t / 20.) * 3. + t / 1000.
                               for name in builder.known_columns()}, index=index)
    for name in covariates:
        if "solar" in name:
            covariates[name] = np.maximum(np.sin(t / 4.), 0.) * 4.
    price_index = index[index.tz_convert("Europe/Paris").date < pd.Timestamp(day).date()]
    quantiles = {zone: pd.DataFrame({"q10": 30. + np.sin(t), "q50": 50. + np.sin(t),
                                    "q90": 70. + np.sin(t)}, index=index)
                 for zone in builder.gate.ZONES}
    prices = {zone: pd.Series(40. + np.sin(np.arange(len(price_index))), index=price_index,
                              name="price_eur_mwh") for zone in builder.gate.ZONES}
    fuel = pd.DataFrame({name: np.ones(len(index)) for name in FUEL_COLUMNS}, index=index)
    for name in FUEL_TIME_COLUMNS:
        fuel[name] = index if name == "value_time_utc" else origins
    schema = builder.gate.load_schema()
    hydro_names = [name for name in schema["families"][builder.POOLED]["columns"]["FR"]
                   if name.startswith("extra_hydro_")]
    hydro = pd.DataFrame(1., index=index, columns=hydro_names)
    jao = pd.DataFrame(2., index=index, columns=builder.JAO_COLUMNS)
    jao["extra_jao__available"] = 1.
    refreshed = jao.copy()
    refreshed[builder.JAO_COLUMNS[0]] = 3.
    capacities = {}
    for name in builder.thermal.SOURCES:
        capacities[name] = pd.DataFrame({"value_time_utc": index,
            "snapshot_time_utc": origins.to_numpy(), "revision_time_utc": origins.to_numpy(),
            "value": np.ones(len(index)), "downloaded_at_utc": origins.to_numpy()})
    exchanges = pd.DataFrame(1., index=index, columns=builder.exchange.COLUMNS)
    return day, dict(prices=prices, nyx_quantiles=quantiles,
        forecast_origins={zone: origins.copy() for zone in builder.gate.ZONES},
        covariates=covariates, fuel=fuel, jao_original=jao, jao_refreshed=refreshed,
        thermal_sources=capacities, hydro_features=hydro, exchange_features=exchanges,
        reference_index=index, feature_index=index[-73:])


def test_future_without_labels_exact_schema_and_separate_jao_branches():
    day, sources = inputs()
    result = builder.build_feature_matrices(day, **sources)
    assert set(result.features) == set(builder.gate.FAMILIES)
    for zone in builder.gate.ZONES:
        assert result.base292[zone].shape[1] == 292
        assert result.augmented334[zone].shape[1] == 334
        assert result.test2_features[zone].shape[1] == 48
        for family, width in builder.gate.FAMILIES.items():
            frame = result.features[family][zone]
            assert frame.shape == (73, width)
            assert frame.index.equals(sources["feature_index"])
        original = result.features[builder.POOLED][zone]
        full = result.features[builder.FULL][zone]
        assert original[builder.JAO_COLUMNS[0]].eq(2.).all()
        assert full[builder.JAO_COLUMNS[0]].eq(3.).all()
        current = original.index.tz_convert("Europe/Paris").date == pd.Timestamp(day).date()
        assert original.loc[current, "price_fr_d1_hour__available"].all()
        assert len(original.index[original.index.tz_convert("Europe/Paris").date ==
                                  pd.Timestamp("2026-10-25").date()]) == 25


def test_delivery_labels_and_late_covariate_origins_are_rejected():
    day, sources = inputs()
    sources["prices"]["FR"] = sources["prices"]["FR"].reindex(sources["reference_index"], fill_value=100.)
    with pytest.raises(ValueError, match="future labels forbidden"):
        builder.build_feature_matrices(day, **sources)
    day, sources = inputs()
    sources["forecast_origins"]["FR"].iloc[-1] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="origin exceeds"):
        builder.build_feature_matrices(day, **sources)


def test_materializer_refuses_missing_receipts_and_no_outputs(tmp_path):
    with pytest.raises(ValueError, match="source receipt missing"):
        builder.materialize_features(tmp_path, "2026-10-26")
    assert not (tmp_path / "features").exists()


def test_revision_snapshots_restore_prices_known_at_each_daily_origin():
    day, sources = inputs()
    selected = "2026-10-25"

    def snapshot(origin_day):
        stop = pd.Timestamp(origin_day, tz="Europe/Paris").tz_convert("UTC")
        result = {zone: values.loc[values.index < stop].copy()
                  for zone, values in sources["prices"].items()}
        if origin_day == selected:
            past_day = result["FR"].index.tz_convert("Europe/Paris").date == pd.Timestamp("2026-10-24").date()
            result["FR"].loc[past_day] = 123.
        return result

    built = builder.build_feature_matrices(day, **sources, price_snapshots=snapshot)
    frame = built.features[builder.POOLED]["FR"]
    selected_hours = frame.index.tz_convert("Europe/Paris").date == pd.Timestamp(selected).date()
    assert frame.loc[selected_hours, "price_fr_d1_hour"].eq(123.).all()
    assert built.audits["countries"]["FR"]["price"]["revised_price_days_rebuilt"] == [selected]


@pytest.mark.skipif(not Path("runs/experiments/nyx_local_365_to20260923/inputs/covariates.parquet").exists(),
                    reason="Optional archived research inputs are not distributed in Git")
def test_local_archives_match_all_twelve_historical_feature_matrices():
    """Arithmetic parity only; old data are not certified prospective inputs."""
    root = Path("runs/experiments/nyx_improvement_to20260923/feature_sets")
    baseline_root = Path("runs/experiments/nyx_local_365_to20260923")
    covariates = pd.read_parquet(baseline_root / "inputs/covariates.parquet")
    index = covariates.index
    baselines = {zone: pd.read_parquet(baseline_root / f"baseline/{zone}.parquet")
                 for zone in builder.gate.ZONES}
    nyx = {zone: frame[["nyx__q10", "nyx__q50", "nyx__q90"]].rename(
        columns=lambda name: name.removeprefix("nyx__")) for zone, frame in baselines.items()}
    origins = {zone: pd.read_parquet(baseline_root / f"chronos/{zone}.parquet").forecast_origin_utc
               for zone in builder.gate.ZONES}
    original = pd.read_parquet(root / "pooled_fundamentals_v1/features_FR.parquet").loc[:, builder.JAO_COLUMNS]
    refreshed = pd.read_parquet(root / "pooled_jao_refresh_v1/features_FR.parquet").loc[:, builder.JAO_COLUMNS]
    built = builder.build_feature_matrices("2026-09-24",
        prices={zone: frame.actual for zone, frame in baselines.items()},
        nyx_quantiles=nyx, forecast_origins=origins, covariates=covariates,
        fuel=pd.read_parquet("data/pit/marginal_cost_expert/fuel/market_fuel_features.parquet"),
        jao_original=original, jao_refreshed=refreshed,
        thermal_sources={name: pd.read_parquet(f"data/pit/marginal_cost_expert/capacities/{name}_available_gw.parquet")
                         for name in builder.thermal.SOURCES},
        hydro_hourly=pd.read_parquet(root / "pooled_fundamentals_v1/hydro_hourly.parquet"),
        exchange_hourly=pd.read_parquet(root / "pooled_exchange_v1/exchange_hourly.parquet"),
        feature_index=index, reference_index=index)
    families = {builder.POOLED: "pooled_fundamentals_v1", builder.FULL: "pooled_jao_refresh_v1",
                builder.COMPACT: "pooled_jao_refresh_v1/compact"}
    for family, directory in families.items():
        for zone in builder.gate.ZONES:
            expected = pd.read_parquet(root / directory / f"features_{zone}.parquet")
            pd.testing.assert_frame_equal(built.features[family][zone], expected, check_exact=True)
