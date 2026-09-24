"""Real rolling residual replay, frozen nuclear outputs, no neural runtime."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nuclear_forecast as base
from chronos2_hourly import solar_correction_forecast as solar


def inputs(tmp_path, **kwargs):
    existing = runpy.run_path(str(Path(__file__).with_name("test_solar_cwe_forecast.py")))
    config, data, day, _, expected = existing["inputs"](tmp_path, **kwargs)
    config["nuclear_experiment"]["input_protocol"] = solar.solar_correction_input_protocol()
    work = tmp_path / "runs/experiments/solar_correction_v1/work"
    historical = expected[expected.tz_convert(data.timezone).date < day]
    future = expected[expected.tz_convert(data.timezone).date == day]

    def raw(index, actual=False):
        local_days = index.tz_convert(data.timezone).date
        origins = {delivery: (pd.Timestamp(delivery) - pd.Timedelta(days=1) + pd.Timedelta(hours=8))
                   .tz_localize(data.timezone).tz_convert("UTC") for delivery in pd.Index(local_days).unique()}
        # Non-round float32 deliberately catches any neural adapter recasting,
        # CSV rounding, or generic future q (already corrected) substitution.
        middle = np.asarray(51.12345 + index.hour.to_numpy() / 137, dtype=np.float32)
        frame = pd.DataFrame({"q10": middle - np.float32(3.33333), "q50": middle,
                              "q90": middle + np.float32(8.88888),
                              "forecast_origin_utc": [origins[d] for d in local_days]}, index=index)
        if actual:
            frame["actual"] = data.target.reindex(index).to_numpy(np.float32)
        frame.index.name = "delivery_start_utc"
        return frame

    history, source = raw(historical, True), raw(future)
    source = source.rename(columns={q: f"chronos2__{q}" for q in base.QUANTILES})
    for q in base.QUANTILES:
        source[q] = source[f"residual_corrected__{q}"] = source[f"chronos2__{q}"] + 999.
    incumbent = SimpleNamespace(raw_history=history, source_forecast=source.reset_index(),
        audit={"engine": "nuclear_forecast_v1", "zone": data.zone, "delivery_day": str(day)})
    return config, data, day, work, incumbent


class FakeResidual:
    min_training_rows = 24 * 365
    feature_builder_options = {"exclude_historical_prices": True, "include_calendar": False,
                               "include_daily_profiles": False}
    fits = []

    def fit(self, X, actual, prediction, experts):
        self.feature_columns_ = tuple(f"known_{alias}_oracle" for alias in solar.INPUT_ALIASES)
        assert set(self.feature_columns_).issubset(X)
        self.last = X.index.max()
        type(self).fits.append((X.index[0], self.last))
        return self

    def predict(self, X, prediction, experts):
        assert self.last < X.index.min()
        return prediction.add(X[solar.SOLAR_KNOWN_COLUMNS[0]] * .01, axis=0)


def delegate(**kwargs):
    return SimpleNamespace(forecast=kwargs["source_forecast"].copy(deep=True), captured=kwargs)


def reject_runtime(*args, **kwargs):
    pytest.fail("Frozen correction ablation reached a neural/model computation")


class RejectForecasting:
    def __getattr__(self, name):
        pytest.fail(f"Frozen correction ablation accessed neural forecasting attribute: {name}")


def run(fixture, **kwargs):
    config, data, day, work, incumbent = fixture
    return solar.run_solar_correction_forecast(config=config, data=data, incumbent=incumbent,
        zone=data.zone, delivery_day=day, workdir=work, threads=1, residual_factory=FakeResidual,
        kalman_builder=delegate, runtime_factory=reject_runtime, forecasting_module=RejectForecasting(), **kwargs)


@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL"])
def test_real_730_day_replay_fits_once_for_two_variants_and_preserves_raw_exactly(tmp_path, monkeypatch, zone):
    fixture = inputs(tmp_path, zone=zone)
    config, data, day, work, incumbent = fixture
    before = deepcopy((config, data, incumbent))
    monkeypatch.setattr(base, "run_nuclear_forecast", reject_runtime)
    monkeypatch.setattr(base, "make_existing_forecasting_executor", reject_runtime)
    FakeResidual.fits = []
    results = run(fixture)
    a, b = results["residual"], results["residual_kalman"]
    expected_fits = a.residual_daily_audit.generation_source.eq("daily_prequential_refit").sum()
    assert len(FakeResidual.fits) == expected_fits
    assert len(a.residual_daily_audit) == 731
    assert a.audit["residual_replay_count"] == 1 and a.audit["chronos_runtime_calls"] == 0
    assert a.audit["chronos_recomputed"] is False
    for name in ("raw_history", "source_forecast", "residual_statistics"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name), check_exact=True)
    pd.testing.assert_frame_equal(a.raw_history, incumbent.raw_history, check_exact=True)
    for q in base.QUANTILES:
        pd.testing.assert_series_equal(a.source_forecast[f"chronos2__{q}"], incumbent.source_forecast[f"chronos2__{q}"], check_exact=True)
    assert set(a.covariates) == {"timestamp", *solar.INPUT_ALIASES[:6]}
    assert set(b.covariates) == {"timestamp", *solar.INPUT_ALIASES}
    assert a.kalman_view.captured["config"] == b.kalman_view.captured["config"]
    assert a.kalman_view.captured["covariate_config"].derived == b.kalman_view.captured["covariate_config"].derived
    assert a.audit["solar_kalman_market_features"] == []
    assert b.audit["solar_kalman_market_features"] == list(solar.SOLAR_ALIASES)
    assert a.kalman_view.captured["rolling_refit_cache_dir"] != b.kalman_view.captured["rolling_refit_cache_dir"]
    assert a.audit["frozen_chronos_history_sha256"] == solar.frozen_chronos_sha256(incumbent.raw_history)
    assert config == before[0]
    pd.testing.assert_frame_equal(data.model_context_covariates, before[1].model_context_covariates)
    pd.testing.assert_frame_equal(incumbent.raw_history, before[2].raw_history)
    assert not list(work.rglob("*chronos*"))


def test_daily_cache_reuses_fitted_outputs_without_neural_fallback(tmp_path):
    fixture = inputs(tmp_path)
    FakeResidual.fits = []
    first = run(fixture)
    first_fits = len(FakeResidual.fits)
    second = run(fixture)
    assert len(FakeResidual.fits) == first_fits
    assert second["residual"].audit["daily_residual_cache"]["hits"] == first_fits
    for key in solar.VARIANTS:
        pd.testing.assert_frame_equal(first[key].source_forecast, second[key].source_forecast, check_exact=True)


def test_incremental_next_day_reuses_all_historical_fits(tmp_path):
    first = inputs(tmp_path, raw_days=740)
    second = inputs(tmp_path, day="2026-09-19", raw_days=741)
    for config, *_ in (first, second):
        config["nuclear_experiment"].update(mode="incremental",
            incremental_cache_dir=str(tmp_path / "runs/experiments/solar_correction_v1/daily"))
    FakeResidual.fits = []
    run(first)
    count = len(FakeResidual.fits)
    result = run(second)["residual"]
    assert len(FakeResidual.fits) == count + 1
    assert result.audit["daily_residual_cache"]["misses"] == 1
    assert result.audit["daily_residual_cache"]["hits"] == count


def test_anchored_full_history_used_for_fit_but_outputs_report_last730(tmp_path):
    fixture = inputs(tmp_path, raw_days=740)
    results = run(fixture)
    config, data, day, _, incumbent = fixture
    result = results["residual"]
    assert len(result.raw_history) < len(incumbent.raw_history)
    assert result.audit["raw_history_start_day"] == str(day - timedelta(days=740))
    assert result.raw_history.index[0].tz_convert(data.timezone).date() == day - timedelta(days=730)
    assert result.residual_daily_audit.training_rows.iloc[0] == 24 * 10
    assert len(result.residual_daily_audit) == 731


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2025-10-26", 25)])
def test_dst_physical_hours_remain_exact(tmp_path, day, hours):
    result = run(inputs(tmp_path, day=day))["residual"]
    assert len(result.source_forecast) == hours
    assert pd.DatetimeIndex(result.source_forecast.delivery_start_utc).is_unique


@pytest.mark.parametrize("fault", ["origin", "actual_future", "actual_history", "missing_raw", "missing_raw_future",
    "missing_anchor", "challenger", "zone", "delivery", "future_target", "solar_nan", "solar_negative",
    "cutoff", "protocol", "promotion", "namespace", "foreign_cache", "recipe_exclusion"])
def test_invalid_input_fails_before_fit_or_write(tmp_path, fault):
    fixture = inputs(tmp_path)
    config, data, day, work, incumbent = fixture
    if fault == "origin": incumbent.raw_history.iloc[0, incumbent.raw_history.columns.get_loc("forecast_origin_utc")] += pd.Timedelta(hours=1)
    elif fault == "actual_future": incumbent.source_forecast["actual"] = 1.
    elif fault == "actual_history": incumbent.raw_history.iloc[-1, incumbent.raw_history.columns.get_loc("actual")] += 1
    elif fault == "missing_raw": incumbent.raw_history = incumbent.raw_history.drop(columns="q50")
    elif fault == "missing_raw_future": incumbent.source_forecast = incumbent.source_forecast.drop(columns="chronos2__q50")
    elif fault == "missing_anchor": config["nuclear_experiment"]["raw_history_start_day"] = str(day - timedelta(days=731))
    elif fault == "challenger": incumbent.audit["candidate_engine"] = "solar_cwe_v1"
    elif fault == "zone": incumbent.audit["zone"] = "BE"
    elif fault == "delivery": incumbent.audit["delivery_day"] = str(day - timedelta(days=1))
    elif fault == "future_target": data.target = pd.concat([data.target, pd.Series(50., index=data.model_context_covariates.index[-1:])])
    elif fault in {"solar_nan", "solar_negative"}:
        data.model_context_covariates.loc[data.model_context_covariates.index[-1], [solar.SOLAR_ALIASES[0], solar.SOLAR_KNOWN_COLUMNS[0]]] = np.nan if fault == "solar_nan" else -1.
    elif fault == "cutoff": config["data"]["forecast_origin_local_time"] = "09:00"
    elif fault == "protocol": config["nuclear_experiment"]["input_protocol"] = "other"
    elif fault == "promotion": config["nuclear_experiment"]["promote"] = True
    elif fault == "namespace": fixture = (*fixture[:3], tmp_path / "runs/experiments/solar_cwe_v1/work", incumbent)
    elif fault == "foreign_cache": config["nuclear_experiment"].update(mode="incremental", incremental_cache_dir=str(tmp_path / "runs/cache/foreign"))
    elif fault == "recipe_exclusion": config["hourly"]["residual_correction"]["feature_builder"]["exclude_columns"] = [solar.SOLAR_KNOWN_COLUMNS[0]]
    FakeResidual.fits = []
    with pytest.raises(ValueError):
        run(fixture)
    assert not FakeResidual.fits and not work.exists()


def test_quantile_preservation_guard_rejects_even_one_bit_change(tmp_path, monkeypatch):
    fixture = inputs(tmp_path)
    real_replay = base.causal_residual_replay
    def corrupted(**kwargs):
        stats, source, audit = real_replay(**kwargs)
        values = source.chronos2__q50.to_numpy(copy=True)
        values[0] = np.nextafter(values[0], np.float32(np.inf))
        source["chronos2__q50"] = values
        return stats, source, audit
    monkeypatch.setattr(base, "causal_residual_replay", corrupted)
    with pytest.raises(ValueError, match="altered frozen Chronos"):
        run(fixture)


def test_post_fit_solar_drop_is_rejected(tmp_path, monkeypatch):
    fixture = inputs(tmp_path)
    original = FakeResidual.fit
    def dropped(self, *args):
        original(self, *args)
        self.feature_columns_ = (base.NUCLEAR_KNOWN_COLUMN,)
        return self
    monkeypatch.setattr(FakeResidual, "fit", dropped)
    with pytest.raises(ValueError, match="dropped"):
        run(fixture)
