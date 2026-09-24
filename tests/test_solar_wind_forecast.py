from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nuclear_forecast as base
from chronos2_hourly import solar_wind_forecast as solar
from chronos2_hourly.kalman_covariates import BASE_RESIDUAL_LOAD_COVARIATES, KalmanCovariateConfig


def inputs(tmp_path, *, day="2026-09-18", zone="FR", raw_days=730):
    delivery, tz = date.fromisoformat(day), base.ZONE_TIMEZONES[zone]
    required = pd.date_range(pd.Timestamp(delivery-timedelta(days=raw_days), tz=tz),
                             pd.Timestamp(delivery+timedelta(days=1), tz=tz), freq="h", inclusive="left").tz_convert("UTC")
    index = pd.date_range(required[0]-pd.Timedelta(hours=48), required[-1], freq="h")
    historical = index[index < pd.Timestamp(delivery, tz=tz)]
    context = pd.DataFrame(index=index)
    for number, alias in enumerate(solar.INPUT_ALIASES):
        # Hourly varying solar profiles: unlike fuels/Pmax, no daily broadcast.
        values = (np.maximum(0, np.sin(index.tz_convert(tz).hour.to_numpy()/24*2*np.pi-np.pi/2))*(number+1)
                  if "_solar_" in alias else (number + 3 + np.sin(np.arange(len(index))/17)
                  if "_wind_" in alias else np.full(len(index), 30.+number)))
        context[alias] = context[f"known_{alias}_oracle"] = values
    specs = {alias: {"enabled": True, "source": "pit_parquet", "series": "baseline_pit_"+alias,
                     "future": {"known_future": True, "strategies": ["oracle"]}} for alias in solar.INPUT_ALIASES}
    specs[base.NUCLEAR_ALIAS]["series"] = base.NUCLEAR_SERIES
    for alias, series in solar.GENERATION_SERIES.items():
        specs[alias].update(series=series, unit="GW", semantic="forecast_generation", daily_broadcast=False,
                            include_base_context=True, fill_method="none", fill_limit=0)
    config = {"data": {"project_root": str(tmp_path), "forecast_origin_local_time": "08:00"},
              "zones": {zone: {"covariates": specs}},
              "model": {"model_id": "amazon/chronos-2", "local_files_only": True, "context_length": 48},
              "hourly": {"feature_engineering": {"target_lags": [24], "target_rolling_windows": [24]},
                         "residual_correction": {"enabled": True, "base_model": "chronos2", "backend": "catboost",
                             "min_training_rows": 24*365, "feature_builder": {"exclude_historical_prices": True}}},
              "nuclear_experiment": {"input_protocol": solar.solar_wind_input_protocol(),
                                     "raw_history_start_day": str(delivery-timedelta(days=raw_days))}}
    data = SimpleNamespace(zone=zone, timezone=tz, frequency="h", target=pd.Series(50., index=historical),
        covariates=context.loc[historical, list(solar.INPUT_ALIASES)].copy(), model_context_covariates=context,
        known_future_columns=[f"known_{alias}_oracle" for alias in solar.INPUT_ALIASES])
    return config, data, delivery, tmp_path/"runs/experiments/solar_wind_v1/w", required


class FakeResidual:
    min_training_rows = 24*365
    feature_builder_options = {"exclude_historical_prices": True, "include_calendar": False, "include_daily_profiles": False}

    def fit(self, X, actual, prediction, experts):
        self.feature_columns_ = (*solar.GENERATION_KNOWN_COLUMNS, base.NUCLEAR_KNOWN_COLUMN)
        assert set(self.feature_columns_).issubset(X)
        self.last = X.index.max()
        return self

    def predict(self, X, prediction, experts):
        assert self.last < X.index.min()
        return prediction.add(X[solar.GENERATION_KNOWN_COLUMNS[0]]*.01, axis=0)


def delegate(**kwargs):
    return SimpleNamespace(forecast=kwargs["source_forecast"].copy(), captured=kwargs)


def fake_base(**kwargs):
    data, day = kwargs["data"], kwargs["delivery_day"]
    context = data.model_context_covariates
    index = context.index[context.index >= pd.Timestamp(day-timedelta(days=730), tz=data.timezone)]
    future = index[index >= pd.Timestamp(day, tz=data.timezone)]
    history = index[index < pd.Timestamp(day, tz=data.timezone)]
    covariates = context.loc[index, [*BASE_RESIDUAL_LOAD_COVARIATES, base.NUCLEAR_ALIAS]].rename_axis("timestamp").reset_index()
    forecast = pd.DataFrame({"delivery_start_utc": future, "residual_corrected__q50": 90.})
    stats = pd.DataFrame({"delivery_start_utc": history, "actual": 50.})
    kalman = kwargs["kalman_builder"](statistics=stats, source_forecast=forecast, covariates=covariates,
        covariate_config=base.nuclear_kalman_covariate_config(), config=SimpleNamespace(q_over_r=.001),
        timezone=data.timezone, delivery_day=day, training_lookback_days=365, upstream_model="residual_corrected",
        output_model="residual_kalman", rolling_refit_cache_dir=Path(kwargs["workdir"])/"kalman_cache")
    daily = pd.DataFrame({"generation_source": ["identity_chronos_cold_start", "daily_prequential_refit"],
                         "residual_feature_columns": [[], [*solar.GENERATION_KNOWN_COLUMNS, base.NUCLEAR_KNOWN_COLUMN]]})
    return base.NuclearForecastResult(stats, stats, forecast, covariates, daily, kalman,
        {"engine": "nuclear_forecast_v1", "raw_history_start_day": str(day-timedelta(days=730)), "production_changed": False})


@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL"])
def test_six_hourly_inputs_consumed_three_stages_without_mutating_baseline(tmp_path, monkeypatch, zone):
    config, data, day, work, _ = inputs(tmp_path, zone=zone)
    before, context = deepcopy(config), data.model_context_covariates.copy(deep=True)
    monkeypatch.setattr(base, "run_nuclear_forecast", fake_base)
    result = solar.run_solar_wind_forecast(config=config, data=data, zone=zone, delivery_day=day, workdir=work,
                                          residual_factory=FakeResidual, kalman_builder=delegate)
    captured = result.kalman_view.captured
    assert set(result.covariates) == {"timestamp", *solar.INPUT_ALIASES}
    assert captured["covariate_config"].derived == base.nuclear_kalman_covariate_config().derived
    for alias in solar.GENERATION_ALIASES:
        assert alias in captured["covariate_config"].input_columns
        assert all(alias in group for group in captured["covariate_config"].groups.values())
        assert result.covariates[alias].max() > result.covariates[alias].min()
    assert captured["training_lookback_days"] == 365 and captured["config"].q_over_r == .001
    assert result.audit["additional_raw_input_count"] == 6 and result.audit["new_ramps_or_aggregates"] is False
    assert result.audit["candidate_engine"] == "solar_wind_v1"
    assert result.audit["solar_wind_chronos_known_future_columns"] == list(solar.GENERATION_KNOWN_COLUMNS)
    assert result.audit["production_pit_evidence"] is False
    assert config == before
    pd.testing.assert_frame_equal(data.model_context_covariates, context)
    assert not work.exists()


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2025-10-26", 25)])
@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL"])
def test_physical_dst_and_optional_preanchor_missing_context(tmp_path, day, hours, zone):
    config, data, delivery, _, _ = inputs(tmp_path, day=day, zone=zone)
    for alias in solar.GENERATION_ALIASES:
        names = [alias, f"known_{alias}_oracle"]
        data.model_context_covariates.iloc[:48, data.model_context_covariates.columns.get_indexer(names)] = np.nan
    for frame in (data.target, data.covariates, data.model_context_covariates):
        frame.index = frame.index.tz_convert(data.timezone)
    context, coverage, expected = solar._prepare_inputs(config, data, zone, delivery)
    assert (expected.tz_convert(data.timezone).date == delivery).sum() == hours
    assert all(row["missing_optional_pre_anchor_context_hours"] == 48 for row in coverage.values())
    assert context.index.is_unique


@pytest.mark.parametrize("fault", ["series", "unit", "semantic", "broadcast", "fill", "context_disabled", "future_spec",
    "missing_context", "missing_known", "undeclared", "history_mismatch", "vintage_mismatch", "future_nan", "history_nan",
    "negative", "residual_selection", "residual_exclusion", "historical_prices", "protocol", "foreign_path", "foreign_cache",
    "extra_covariate", "derived_context", "supply_stack", "missing_baseline_future"])
@pytest.mark.parametrize("input_alias", ["fr_solar_generation_fcst", "de_wind_generation_fcst", "nl_wind_generation_fcst"])
def test_invalid_contract_fails_before_any_model_or_write(tmp_path, monkeypatch, fault, input_alias):
    config, data, day, work, _ = inputs(tmp_path)
    alias, known = input_alias, f"known_{input_alias}_oracle"
    spec, context = config["zones"]["FR"]["covariates"][alias], data.model_context_covariates
    changes = {"series": ("series", "power.fr.generation.solar.hourly.gw.obs"), "unit": ("unit", "MW"),
               "semantic": ("semantic", "irradiance"), "broadcast": ("daily_broadcast", True), "fill": ("fill_method", "ffill"),
               "context_disabled": ("include_base_context", False)}
    if fault in changes: spec[changes[fault][0]] = changes[fault][1]
    elif fault == "future_spec": spec["future"]["known_future"] = False
    elif fault == "missing_context": data.model_context_covariates = context.drop(columns=alias)
    elif fault == "missing_known": data.model_context_covariates = context.drop(columns=known)
    elif fault == "undeclared": data.known_future_columns.remove(known)
    elif fault == "missing_baseline_future": data.known_future_columns.remove("known_fr_residual_load_fcst_oracle")
    elif fault == "history_mismatch": data.covariates.iloc[-1, data.covariates.columns.get_loc(alias)] += 1
    elif fault == "vintage_mismatch": context.loc[context.index[-1], alias] += 1
    elif fault == "future_nan": context.loc[context.index[-1], [alias, known]] = np.nan
    elif fault == "history_nan": context.loc[context.index[100], [alias, known]] = np.nan
    elif fault == "negative": context.loc[context.index[-1], [alias, known]] = -1
    elif fault == "residual_selection": config["hourly"]["feature_engineering"]["covariate_columns"] = [base.NUCLEAR_KNOWN_COLUMN]
    elif fault == "residual_exclusion": config["hourly"]["residual_correction"]["feature_builder"]["exclude_columns"] = [known]
    elif fault == "historical_prices": config["hourly"]["residual_correction"]["feature_builder"]["exclude_historical_prices"] = False
    elif fault == "protocol": config["nuclear_experiment"]["input_protocol"] = "old"
    elif fault == "foreign_path": work = tmp_path/"runs/experiments/nuclear_forecast_v1/w"
    elif fault == "foreign_cache": config["nuclear_experiment"].update(mode="incremental", incremental_cache_dir=str(tmp_path/"runs/cache/w"))
    elif fault == "extra_covariate": config["zones"]["FR"]["covariates"]["solar_ramp"] = {"enabled": True}
    elif fault == "derived_context": context["solar_ramp"] = 1.
    elif fault == "supply_stack": config["hourly"]["supply_stack"] = {"enabled": True}
    def forbidden(**kwargs):
        pytest.fail("Invalid SolarWind contract reached the model")
    monkeypatch.setattr(base, "run_nuclear_forecast", forbidden)
    with pytest.raises(ValueError):
        solar.run_solar_wind_forecast(config=config, data=data, zone="FR", delivery_day=day, workdir=work)
    assert not work.exists()


def test_schema_protocol_and_kalman_dimensionality(tmp_path, monkeypatch):
    previous = solar.solar_wind_input_protocol()
    monkeypatch.setitem(solar.GENERATION_SERIES, "fr_solar_generation_fcst", "changed-series")
    assert solar.solar_wind_input_protocol() != previous
    original = base.nuclear_kalman_covariate_config()
    contract = solar.solar_wind_kalman_covariate_config(original)
    assert original.derived == contract.derived and len(contract.input_columns) == 12
    assert contract.minimum_history_coverage == 1 and contract.require_future_complete is True
    assert not set(solar.GENERATION_ALIASES).intersection(original.input_columns)
    huge = KalmanCovariateConfig(input_columns=tuple(f"feature_{i}" for i in range(57)),
        derived=(), groups={"market": tuple(f"feature_{i}" for i in range(57))})
    with pytest.raises(ValueError, match="maximum"):
        solar.solar_wind_kalman_covariate_config(huge)


def test_anchor_support_accepts_extended_history_and_rejects_short_history(tmp_path):
    config, data, day, _, _ = inputs(tmp_path, raw_days=800)
    _, coverage, expected = solar._prepare_inputs(config, data, "FR", day)
    assert expected[0].tz_convert(data.timezone).date() == day-timedelta(days=800)
    assert all(row["required_hours"] == len(expected) for row in coverage.values())
    config["nuclear_experiment"]["raw_history_start_day"] = str(day-timedelta(days=729))
    with pytest.raises(ValueError, match="anchor"):
        solar._prepare_inputs(config, data, "FR", day)


def test_post_fit_feature_drop_is_rejected(tmp_path, monkeypatch):
    config, data, day, work, _ = inputs(tmp_path)
    def dropped(**kwargs):
        result = fake_base(**kwargs)
        result.residual_daily_audit.at[1, "residual_feature_columns"] = [base.NUCLEAR_KNOWN_COLUMN]
        return result
    monkeypatch.setattr(base, "run_nuclear_forecast", dropped)
    with pytest.raises(ValueError, match="dropped"):
        solar.run_solar_wind_forecast(config=config, data=data, zone="FR", delivery_day=day, workdir=work,
                                    residual_factory=FakeResidual, kalman_builder=delegate)


def test_default_kalman_normalizes_timestamp_and_does_not_add_cache_namespace(tmp_path, monkeypatch):
    config, data, day, work, _ = inputs(tmp_path)
    monkeypatch.setattr(base, "run_nuclear_forecast", fake_base)
    monkeypatch.setattr(solar, "build_operational_kalman_view", delegate)
    result = solar.run_solar_wind_forecast(config=config, data=data, zone="FR", delivery_day=day, workdir=work,
                                          residual_factory=FakeResidual)
    assert "timestamp" in result.kalman_view.captured["covariates"]
    cache = str(result.kalman_view.captured["rolling_refit_cache_dir"])
    assert Path(cache.removeprefix("\\\\?\\")) == work/"kalman_cache"
    assert result.audit["solar_wind_kalman_delegate_signature"] is None


def test_real_native_kalman_accepts_solar_columns_before_replay(tmp_path, monkeypatch):
    import chronos2_hourly.kalman_residual as native
    _, data, day, work, expected = inputs(tmp_path)
    history = expected[expected < pd.Timestamp(day, tz=data.timezone)]
    future = expected[expected >= pd.Timestamp(day, tz=data.timezone)]
    def raw(index, actual=False):
        value = pd.DataFrame({"delivery_start_utc": index, "residual_correction": 0.,
            "residual_corrected__q10": 70., "residual_corrected__q50": 90., "residual_corrected__q90": 110.})
        if actual: value["actual"] = 90.
        return value
    class ReachedNativeReplay(Exception): pass
    def validated(*args, **kwargs):
        assert "timestamp" in kwargs["covariates"] and set(solar.GENERATION_ALIASES).issubset(kwargs["covariates"])
        raise ReachedNativeReplay
    monkeypatch.setattr(native, "replay_kalman_overlay", validated)
    covariates = data.model_context_covariates.loc[expected, [*BASE_RESIDUAL_LOAD_COVARIATES, base.NUCLEAR_ALIAS]].rename_axis("timestamp").reset_index()
    hook = solar._SolarWindKalmanBuilder(data.model_context_covariates, native.build_operational_kalman_view, None)
    with pytest.raises(ReachedNativeReplay):
        hook(statistics=raw(history, True), source_forecast=raw(future), covariates=covariates,
            timezone=data.timezone, delivery_day=day, covariate_config=base.nuclear_kalman_covariate_config(),
            upstream_model="residual_corrected", output_model="residual_kalman", training_lookback_days=365,
            rolling_refit_cache_dir=work/"cache")


class FakeForecasting:
    @staticmethod
    def _check_neural_frames(data, origin, context_length, horizon, item, live=False):
        from chronos2_modular.forecasting import build_origin_frames, build_live_frames
        frames = (build_live_frames(data, context_length, horizon, item, True) if live else
                  build_origin_frames(data, origin, context_length, horizon, item, True))
        context, future = frames[:2]
        assert set(solar.GENERATION_ALIASES).issubset(context)
        assert set(solar.GENERATION_KNOWN_COLUMNS).issubset(context) and set(solar.GENERATION_KNOWN_COLUMNS).issubset(future)

    @classmethod
    def run_backtest_variant(cls, **kwargs):
        assert kwargs["with_covariates"] is True
        data, pieces = kwargs["data"], []
        for origin in kwargs["origins"]:
            cls._check_neural_frames(data, origin, kwargs["context_length"], kwargs["horizon"], "test")
            index = data.target.index[origin:origin+kwargs["horizon"]]
            middle = sum(data.model_context_covariates.loc[index, known] for known in solar.GENERATION_KNOWN_COLUMNS)
            pieces.append(pd.DataFrame({"timestamp": index, "q10": middle.to_numpy()-5, "q50": middle.to_numpy(),
                "q90": middle.to_numpy()+5, "actual": data.target.loc[index].to_numpy(), "origin_index": origin}))
        return pd.concat(pieces, ignore_index=True)

    @classmethod
    def run_live_forecast_variant(cls, **kwargs):
        data = kwargs["data"]
        cls._check_neural_frames(data, len(data.target), kwargs["context_length"], kwargs["horizon"], "test", live=True)
        index = pd.date_range(data.target.index[-1]+pd.Timedelta(hours=1), periods=kwargs["horizon"], freq="h")
        middle = sum(data.model_context_covariates.loc[index, known] for known in solar.GENERATION_KNOWN_COLUMNS)
        return pd.DataFrame({"timestamp": index, "q10": middle.to_numpy()-5, "q50": middle.to_numpy(), "q90": middle.to_numpy()+5})


def test_real_730_day_base_engine_and_effective_neural_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("sc")
    config, data, day, work, _ = inputs(root)
    result = solar.run_solar_wind_forecast(config=config, data=data, zone="FR", delivery_day=day, workdir=work,
        runtime_factory=lambda *args: object(), forecasting_module=FakeForecasting, residual_factory=FakeResidual,
        kalman_builder=delegate, threads=1)
    assert len(result.residual_daily_audit) == 731 and len(result.source_forecast) == 24
    expected = sum(data.model_context_covariates[alias].iloc[-24:].to_numpy() for alias in solar.GENERATION_ALIASES)
    np.testing.assert_allclose(result.source_forecast.chronos2__q50, expected, rtol=1e-6, atol=1e-6)
    expected += .01*data.model_context_covariates[solar.GENERATION_ALIASES[0]].iloc[-24:].to_numpy()
    np.testing.assert_allclose(result.source_forecast.residual_corrected__q50, expected, rtol=1e-6, atol=1e-6)
    assert result.audit["lora_used"] is False and result.audit["production_changed"] is False
    assert not (root/"runs/experiments/nuclear_forecast_v1").exists()



@pytest.mark.parametrize("namespace", ["solar_cwe_v1/cache", "nuclear_forecast_v1/cache", "solar_wind_v1/outside"])
def test_incremental_namespace_cannot_escape_own_cache(tmp_path, namespace):
    config, _, _, work, _ = inputs(tmp_path)
    config["nuclear_experiment"].update(mode="incremental",
        incremental_cache_dir=str(tmp_path/"runs/experiments/solar_wind_v1/_daily_cache"),
        incremental_namespace=str(tmp_path/"runs/experiments"/namespace))
    with pytest.raises(ValueError, match="namespace"):
        solar._isolated_paths(config, work)


def test_exact_requested_six_series_and_no_solar_identity():
    from chronos2_hourly.solar_cwe_forecast import solar_cwe_input_protocol
    assert len(solar.GENERATION_SERIES) == 6
    assert len(solar.INPUT_ALIASES) == 12
    assert solar.GENERATION_SERIES["de_wind_generation_fcst"] == "power.de.generation.wind.hourly.gw.fcst"
    assert solar.GENERATION_SERIES["nl_wind_generation_fcst"] == "power.nl.generation.wind.hourly.gw.fcst"
    assert solar.solar_wind_input_protocol() != solar_cwe_input_protocol()
    assert solar.solar_wind_input_protocol().startswith("civil_pit_solar_wind_v1_")


def test_nl_primary_substitution_refused_before_model(tmp_path, monkeypatch):
    config, data, day, work, _ = inputs(tmp_path, zone="NL")
    config["zones"]["NL"]["covariates"]["nl_wind_generation_fcst"]["series"] = "power.nl.prod.total.wind.mw.ecmwf_avg.pointconnect.6h.cache"
    monkeypatch.setattr(base, "run_nuclear_forecast", lambda **kw: pytest.fail("Model reached"))
    with pytest.raises(ValueError, match="exact hourly"):
        solar.run_solar_wind_forecast(config=config, data=data, zone="NL", delivery_day=day, workdir=work)

