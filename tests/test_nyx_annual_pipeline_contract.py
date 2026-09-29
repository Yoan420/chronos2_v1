"""Real source/feature/reference contracts with explicitly unqualified fixtures.

Only provider transport, pretrained Chronos scores and reference estimators are
test doubles. The calendar, 292/334/449/503/123 calculations, thermal source
formats, Test2 normalizations, prior90 selection, routing, saved model plumbing
and four-country output schemas execute their real implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_saturn_source as saturn
from chronos2_hourly import nyx_annual_cpu_baseline as baseline
from chronos2_hourly import nyx_annual_cpu_bundle_builder as builder
from chronos2_hourly import nyx_annual_cpu_reference_builder as reference
from chronos2_hourly.nyx_local_extra_features import FUEL_COLUMNS, FUEL_TIME_COLUMNS, cutoff_times
from test_nyx_annual_cpu_baseline import Pipeline
from test_nyx_annual_jao_source import FakeClient, _fetch
from test_run_nyx_annual_hydro_source import Session as HydroSession, payload as hydro_payload
from test_run_nyx_annual_exchange_source import Session as ExchangeSession, body as exchange_body


@dataclass(frozen=True)
class FastHGB:
    stop_day_exclusive: str


class FastTest2:
    audit = {"test_fixture": True, "production_qualified": False}

    def predict_day(self, features, nyx, *, zone, forecast_issued_at_utc):
        assert features.shape[1] == 45
        return pd.DataFrame({"test2__q50": nyx + 1., "spike_probability": .1}, index=nyx.index)


def _signal(index, kind):
    t = (index.asi8 - pd.Timestamp("2020-01-01", tz="UTC").value) / 3.6e12
    if "solar" in kind:
        return 6. * np.maximum(0., np.sin(t * 2 * np.pi / 24))
    return 20. + 3. * np.sin(t / 27.) + 2. * np.sin(t / 170.)


def test_saturn_source_to_real_twelve_matrices_and_four_references(tmp_path, monkeypatch):
    import run_nyx_annual_thermal_source as thermal
    import run_nyx_annual_hydro_source as hydro
    import run_nyx_annual_exchange_source as exchange
    from chronos2_hourly import nyx_annual_jao_source as jao
    from run_nyx_annual_auction_prices_source import SERIES

    day = "2026-09-29"  # Worst weekly-anchor offset: exactly 462 baseline days required.
    d = pd.Timestamp(day).date()
    full, current, cutoff = builder.gate.delivery_grid(day)
    raw_index = baseline._grid(d - timedelta(days=834), d + timedelta(days=1)).rename("timestamp_utc")
    ref_index = baseline._grid(d - timedelta(days=462), d + timedelta(days=1)).rename("timestamp_utc")
    quantile_index = baseline._grid(d - timedelta(days=469), d + timedelta(days=1)).rename("timestamp_utc")
    prices_index = baseline._grid(d - timedelta(days=469), d).rename("timestamp_utc")
    contract = saturn.specs()
    aliases = {spec["series"]: alias for alias, spec in contract.items()}

    def fetch(_client, series, first, last, timezone, **kwargs):
        assert kwargs["revision_date"] == cutoff and kwargs["nocache"] is True
        index = pd.date_range(first, last, freq="h", name="timestamp_utc")
        kind = aliases.get(series, "price")
        assert series in aliases or series in SERIES.values()
        return pd.Series(_signal(index, kind), index=index)

    monkeypatch.setattr(saturn, "fetch_saturn_series_from_client", fetch)
    daily_root = tmp_path / "saturn_capture"
    saturn.capture_day(object(), day, daily_root, now_utc=cutoff)
    daily_cov, daily_prices, daily_receipt = saturn.verify_day(daily_root / day, day)
    cov = pd.DataFrame({name: _signal(raw_index, name) for name in saturn.ALIASES}, index=raw_index)
    pd.testing.assert_frame_equal(cov.loc[current].rename_axis("timestamp_utc"), daily_cov, check_freq=False)
    prices = {z: pd.Series(_signal(prices_index, "price"), index=prices_index, name="price_eur_mwh")
              for z in builder.gate.ZONES}
    for zone in builder.gate.ZONES:
        common = daily_prices.index.intersection(prices[zone].index)
        np.testing.assert_array_equal(prices[zone].loc[common], daily_prices.loc[common, zone])
    assert daily_receipt["provider_publication_timestamp_verified"] is False

    nyx, histories = {}, {}
    origins = cutoff_times(raw_index, "Europe/Paris")
    for zone in builder.gate.ZONES:
        center = _signal(quantile_index, "price") + 2.
        quantiles = pd.DataFrame({"q10": center - 10., "q50": center, "q90": center + 10.}, index=quantile_index)
        # The actual current source contract flows through the native Chronos
        # input adapter, including all fourteen named covariates and no labels.
        predicted = baseline.predict_chronos_day(target=daily_prices[zone], covariates=cov,
                                                 zone=zone, delivery_day=day, pipeline=Pipeline())
        assert "actual" not in predicted
        quantiles.loc[current] = predicted.loc[current, ["q10", "q50", "q90"]]
        nyx[zone] = quantiles
        histories[zone] = quantiles.rename(columns=lambda name: "nyx__" + name)
        histories[zone]["actual"] = prices[zone].reindex(quantile_index)
        histories[zone]["forecast_origin_utc"] = origins.loc[quantile_index]

    fuel = pd.DataFrame({name: np.ones(len(ref_index)) for name in FUEL_COLUMNS}, index=ref_index)
    for name in FUEL_TIME_COLUMNS:
        fuel[name] = ref_index if name == "value_time_utc" else origins.loc[ref_index]
    jao_root = tmp_path / "jao_capture"
    jao.capture_day(day=d, cache_root=jao_root, client=FakeClient(_fetch(d)),
                    now_utc=cutoff - pd.Timedelta(minutes=9))
    jao_day, _ = jao.build_strict_history_features(jao_root, current)
    hydro_root, exchange_root = tmp_path / "hydro_capture", tmp_path / "exchange_capture"
    hydro.capture(day, hydro_root, session=HydroSession(hydro_payload(day)),
                  now_utc=cutoff - pd.Timedelta(minutes=5))
    hydro_day, _ = hydro.verify_capture(hydro_root / day, day)
    exchange.capture(day, exchange_root,
                     session=ExchangeSession({z: exchange_body(day, z) for z in ("DE", "FR")}),
                     now_utc=cutoff - pd.Timedelta(minutes=5))
    exchange_day, _ = exchange._verify_capture(exchange_root / day, day)

    def only_captured_day(frame):
        # No historical API evidence exists in this fixture: keep its values
        # missing rather than pretending the current capture was available then.
        result = frame.reindex(full)
        for column in result:
            if column.endswith("__available"):
                result[column] = result[column].fillna(0.)
        return result

    thermal_days, _, _ = thermal.grids(day)
    capacities = {name: thermal.source_hourly(pd.Series(float(i + 1), index=thermal_days,
                                                       name="pmax_gw"), day)
                  for i, name in enumerate(thermal.SOURCES)}
    one_missing_day = capacities[thermal.SOURCES[0]].index.tz_convert("Europe/Paris").date == pd.Timestamp("2026-04-01").date()
    capacities[thermal.SOURCES[0]].loc[one_missing_day, "pmax_gw"] = np.nan

    def snapshots(origin):
        stop = pd.Timestamp(origin, tz="Europe/Paris").tz_convert("UTC")
        return {zone: series.loc[series.index < stop] for zone, series in prices.items()}

    built = builder.build_feature_matrices(day, prices=prices, nyx_quantiles=nyx,
        forecast_origins={z: origins for z in builder.gate.ZONES}, covariates=cov, fuel=fuel,
        jao_original=only_captured_day(jao_day), jao_refreshed=only_captured_day(jao_day),
        hydro_features=only_captured_day(hydro_day), exchange_features=only_captured_day(exchange_day),
        thermal_sources=capacities, price_snapshots=snapshots)
    for zone in builder.gate.ZONES:
        assert tuple(built.base292[zone]) == reference.archived_columns(zone, "hist_residual_400")
        assert tuple(built.augmented334[zone]) == reference.archived_columns(zone, "augmented_hist_residual_400")
        assert built.test2_features[zone].index.equals(ref_index)
        assert np.isfinite(built.test2_features[zone].to_numpy(float)).all()
        assert built.base292[zone].loc[ref_index[0], "past_error_fr_d7_hour__available"]
        assert built.features[builder.FULL][zone].loc[current, "extra_jao__available"].eq(1.).all()
        for family, frames in built.features.items():
            builder._write_frame(tmp_path / f"features/{family}/{zone}.parquet", frames[zone])

    # Write an explicitly unqualified fixture baseline receipt. Actual CPU
    # qualification checks stay active and reject it at the publication gate.
    hashes = {}
    for zone, frame in histories.items():
        relative = f"baseline_history/{zone}.parquet"
        baseline.write_frame(tmp_path / relative, frame)
        hashes[relative] = builder.gate.sha256(tmp_path / relative)
    baseline.write_json(tmp_path / "source_receipts/nyx_quantiles.json", {
        "protocol": builder.gate.SOURCE_PROTOCOL, "source_group": "nyx_quantiles",
        "state": "UNQUALIFIED", "delivery_day": day, "artifact_sha256": hashes,
        "provider_publication_timestamp_verified": False})

    def fit_hgb(matrix, actual, point, **kwargs):
        expected_columns = reference.archived_columns(kwargs["zone"], kwargs["variant"])
        assert tuple(matrix) == expected_columns
        assert actual.iloc[-24:].isna().all()
        return FastHGB(kwargs["stop_day"]), None, {"test_fixture": True, "future_labels_used": False}

    monkeypatch.setattr(reference, "fit_hgb_block", fit_hgb)
    # This deliberately unqualified arithmetic fixture has no source packet.
    monkeypatch.setattr(saturn, "target_history_contract", lambda _: {})
    monkeypatch.setattr(reference, "fit_test2_origin", lambda *args, **kwargs: FastTest2())
    monkeypatch.setattr(reference, "predict_saved_block", lambda fit, matrix, nyx:
                        pd.DataFrame({"point": nyx + 2.}, index=nyx.index))
    receipt = reference.build_cpu_reference_bundle(tmp_path, day, baselines=histories,
        base_features=built.base292, augmented_features=built.augmented334,
        test2_features=built.test2_features, target_snapshots_by_day=snapshots,
        require_verified_sources=False)
    assert receipt["state"] == "UNQUALIFIED"
    for zone in builder.gate.ZONES:
        output = pd.read_parquet(tmp_path / f"reference/{zone}.parquet")
        builder.gate.validate_reference(output, current, cutoff, zone)
    with pytest.raises(ValueError, match="source receipt missing"):
        builder.seal_bundle(tmp_path, day)
    assert not (tmp_path / builder.gate.MATERIALIZATION_PATH).exists()
