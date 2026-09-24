"""The full fuel variant feeds all three stages without touching production."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyx_clean_fuel import forecast as fuel
from chronos2_hourly import nuclear_forecast as base
from chronos2_hourly.kalman_covariates import BASE_RESIDUAL_LOAD_COVARIATES


def plain_windows_path(path):
    return Path(str(path).removeprefix("\\\\?\\"))


def inputs(tmp_path, day="2026-09-18", zone="FR"):
    delivery, tz = date.fromisoformat(day), base.ZONE_TIMEZONES[zone]
    start = pd.Timestamp(delivery - timedelta(days=730), tz=tz)
    stop = pd.Timestamp(delivery + timedelta(days=1), tz=tz)
    required = pd.date_range(start, stop, freq="h", inclusive="left").tz_convert("UTC")
    index = pd.date_range(required[0] - pd.Timedelta(hours=48), required[-1], freq="h")
    historical = index[index < pd.Timestamp(delivery, tz=tz)]
    aliases = (*BASE_RESIDUAL_LOAD_COVARIATES, base.NUCLEAR_ALIAS, *fuel.FUEL_ALIASES)
    context = pd.DataFrame(index=index)
    for number, alias in enumerate(aliases):
        value = 80. + number if alias in fuel.FUEL_ALIASES else 30. + number
        context[alias] = context[f"known_{alias}_oracle"] = value
    specs = {base.NUCLEAR_ALIAS: {"enabled": True, "source": "pit_parquet", "series": base.NUCLEAR_SERIES,
                                  "future": {"known_future": True, "strategies": ["oracle"]}}}
    for alias, schema in fuel.SCHEMA.items():
        specs[alias] = {**deepcopy(schema), "enabled": True, "source": "pit_parquet", "fill_method": "none",
                        "future": {"known_future": True, "strategies": ["oracle"]}}
    cfg = {"data": {"project_root": str(tmp_path), "forecast_origin_local_time": "08:00"},
           "zones": {zone: {"covariates": specs}},
           "model": {"model_id": "amazon/chronos-2", "local_files_only": True, "context_length": 48},
           "hourly": {"feature_engineering": {"target_lags": [24], "target_rolling_windows": [24]},
                      "residual_correction": {"enabled": True, "base_model": "chronos2", "backend": "catboost",
                          "min_training_rows": 24 * 365, "feature_builder": {"exclude_historical_prices": True}}},
           "nuclear_experiment": {"input_protocol": fuel.clean_fuel_input_protocol()}}
    data = SimpleNamespace(zone=zone, timezone=tz, target=pd.Series(50., index=historical),
        covariates=context.loc[historical, list(aliases)].copy(), model_context_covariates=context,
        known_future_columns=[f"known_{alias}_oracle" for alias in aliases])
    work = tmp_path / "runs/experiments/nyx_clean_fuel_full_v1" / zone.lower() / day
    return cfg, data, delivery, work, required


class FakeResidual:
    min_training_rows = 24 * 365
    feature_builder_options = {"exclude_historical_prices": True, "include_calendar": False, "include_daily_profiles": False}

    def fit(self, X, actual, prediction, experts):
        self.feature_columns_ = (*fuel.FUEL_KNOWN_COLUMNS, base.NUCLEAR_KNOWN_COLUMN)
        assert all(column in X for column in self.feature_columns_)
        self.last = X.index.max()
        return self

    def predict(self, X, prediction, experts):
        assert self.last < X.index.min()
        return prediction.add(X.known_ccc_oracle * .01, axis=0)


def delegate(**kwargs):
    return SimpleNamespace(forecast=kwargs["source_forecast"].copy(), captured=kwargs)


def fake_base(**kwargs):
    data, delivery = kwargs["data"], kwargs["delivery_day"]
    context, tz = data.model_context_covariates, data.timezone
    index = context.index[context.index >= pd.Timestamp(delivery - timedelta(days=730), tz=tz)]
    future = index[index >= pd.Timestamp(delivery, tz=tz)]
    history = index[index < pd.Timestamp(delivery, tz=tz)]
    covariates = pd.DataFrame({alias: context.loc[index, f"known_{alias}_oracle"]
        for alias in (*BASE_RESIDUAL_LOAD_COVARIATES, base.NUCLEAR_ALIAS)})
    source = pd.DataFrame({"timestamp": future, "residual_corrected__q50": 80.})
    stats = pd.DataFrame({"timestamp": history, "actual": 50.})
    kalman = kwargs["kalman_builder"](statistics=stats, source_forecast=source,
        covariates=covariates.rename_axis("timestamp").reset_index(), covariate_config=base.nuclear_kalman_covariate_config(),
        config=SimpleNamespace(q_over_r=.001), timezone=tz, delivery_day=delivery, training_lookback_days=365,
        upstream_model="residual_corrected", output_model="residual_kalman", rolling_refit_cache_dir=Path(kwargs["workdir"]) / "kalman_cache")
    daily = pd.DataFrame({"generation_source": ["identity_chronos_cold_start", "daily_prequential_refit"],
        "residual_feature_columns": [[], [*fuel.FUEL_KNOWN_COLUMNS, base.NUCLEAR_KNOWN_COLUMN]]})
    return base.NuclearForecastResult(stats, stats, source, covariates, daily, kalman,
        {"engine": "nuclear_forecast_v1", "raw_history_start_day": str(delivery - timedelta(days=730)), "production_changed": False})


@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL"])
def test_all_countries_keep_recipe_and_consume_fuels_in_every_kalman_group(tmp_path, monkeypatch, zone):
    cfg, data, day, work, _ = inputs(tmp_path, zone=zone)
    saved_cfg, saved_context = deepcopy(cfg), data.model_context_covariates.copy(deep=True)
    monkeypatch.setattr(base, "run_nuclear_forecast", fake_base)
    result = fuel.run_clean_fuel_forecast(config=cfg, data=data, zone=zone, delivery_day=day, workdir=work,
        residual_factory=FakeResidual, kalman_builder=delegate)
    capture = result.kalman_view.captured
    for alias in fuel.FUEL_ALIASES:
        assert alias in result.covariates
        assert alias in capture["covariate_config"].input_columns
        assert all(alias in group for group in capture["covariate_config"].groups.values())
        assert result.covariates[alias].iloc[-1] == data.model_context_covariates[alias].iloc[-1]
    assert capture["training_lookback_days"] == 365
    assert capture["config"].q_over_r == .001
    assert plain_windows_path(capture["rolling_refit_cache_dir"]).parent == work / "kalman_cache"
    assert "timestamp" in result.covariates
    assert result.audit["candidate_engine"] == fuel.ENGINE
    assert result.audit["carbon_added_again"] is False
    assert result.audit["native_costs_rescaled"] is False
    assert result.audit["enforced_price_floor"] is False
    assert result.audit["promotion_eligible"] is False
    assert cfg == saved_cfg
    pd.testing.assert_frame_equal(data.model_context_covariates, saved_context)
    assert not work.exists()


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2025-10-26", 25)])
@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL"])
def test_dst_and_optional_preanchor_missing_context_are_preserved(tmp_path, day, hours, zone):
    cfg, data, delivery, _, _ = inputs(tmp_path, day=day, zone=zone)
    for alias in fuel.FUEL_ALIASES:
        names = [alias, f"known_{alias}_oracle"]
        data.model_context_covariates.iloc[:48, data.model_context_covariates.columns.get_indexer(names)] = np.nan
    for frame in (data.target, data.covariates, data.model_context_covariates):
        frame.index = frame.index.tz_convert(data.timezone)
    result, audit, expected = fuel._prepare_inputs(cfg, data, zone, delivery)
    assert (expected.tz_convert(data.timezone).date == delivery).sum() == hours
    assert all(value["missing_optional_pre_anchor_context_hours"] == 48 for value in audit.values())
    assert result.iloc[:48][list(fuel.FUEL_KNOWN_COLUMNS)].isna().all().all()
    assert not result.index.has_duplicates


@pytest.mark.parametrize("fault", ["series", "source", "disabled", "unit", "semantic", "carbon", "broadcast", "fill",
    "future_spec", "missing_context", "missing_known", "undeclared", "missing_history", "history_mismatch", "vintage_mismatch",
    "future_nan", "history_nan", "intraday_change", "residual_selection", "residual_exclusion", "historical_prices",
    "protocol", "operational_path", "old_lab_path", "foreign_cache", "storm", "actual"])
def test_invalid_source_contract_stops_before_any_base_work(tmp_path, monkeypatch, fault):
    cfg, data, day, work, _ = inputs(tmp_path)
    alias, known = "ccc", "known_ccc_oracle"
    spec = cfg["zones"]["FR"]["covariates"][alias]
    context = data.model_context_covariates
    mutations = {"series": ("series", "wrong"), "source": ("source", "csv"), "disabled": ("enabled", False),
        "unit": ("unit", "EUR/MWh_th"), "semantic": ("semantic", "fuel_only"), "carbon": ("carbon_included", False),
        "broadcast": ("daily_broadcast", False), "fill": ("fill_method", "ffill")}
    if fault in mutations:
        key, value = mutations[fault]
        spec[key] = value
    elif fault == "future_spec": spec["future"]["known_future"] = False
    elif fault == "missing_context": data.model_context_covariates = context.drop(columns=alias)
    elif fault == "missing_known": data.model_context_covariates = context.drop(columns=known)
    elif fault == "undeclared": data.known_future_columns.remove(known)
    elif fault == "missing_history": data.covariates = data.covariates.drop(columns=alias)
    elif fault == "history_mismatch": data.covariates.iloc[-1, data.covariates.columns.get_loc(alias)] += 1
    elif fault == "vintage_mismatch": context.loc[context.index[-1], alias] += 1
    elif fault == "future_nan": context.loc[context.index[-1], [alias, known]] = np.nan
    elif fault == "history_nan": context.loc[context.index[100], [alias, known]] = np.nan
    elif fault == "intraday_change": context.loc[context.index[-1], [alias, known]] += 1
    elif fault == "residual_selection": cfg["hourly"]["feature_engineering"]["covariate_columns"] = [base.NUCLEAR_KNOWN_COLUMN]
    elif fault == "residual_exclusion": cfg["hourly"]["residual_correction"]["feature_builder"]["exclude_columns"] = [known]
    elif fault == "historical_prices": cfg["hourly"]["residual_correction"]["feature_builder"]["exclude_historical_prices"] = False
    elif fault == "protocol": cfg["nuclear_experiment"]["input_protocol"] = "old"
    elif fault == "operational_path": work = tmp_path / "runs/live/fr"
    elif fault == "old_lab_path": work = tmp_path / "runs/experiments/nyx_clean_fuel_v1/test"
    elif fault == "foreign_cache": cfg["nuclear_experiment"].update(mode="incremental", incremental_cache_dir=str(tmp_path / "runs/experiments/nuclear_forecast_v1/cache"))
    elif fault in {"storm", "actual"}: context[fault] = 100.
    def forbidden(**kwargs):
        pytest.fail("An invalid clean-fuel contract launched a model")
    monkeypatch.setattr(base, "run_nuclear_forecast", forbidden)
    with pytest.raises(ValueError):
        fuel.run_clean_fuel_forecast(config=cfg, data=data, zone="FR", delivery_day=day, workdir=work)
    assert not work.exists()


def test_fuel_inputs_do_not_pollute_residual_load_aggregates():
    original = base.nuclear_kalman_covariate_config()
    extended = fuel.clean_fuel_kalman_covariate_config(original)
    assert extended.derived == original.derived
    assert all(not set(fuel.FUEL_ALIASES).intersection(item.sources) for item in extended.derived)
    assert extended.minimum_history_coverage == 1
    assert extended.require_future_complete is True
    assert all(alias not in original.input_columns for alias in fuel.FUEL_ALIASES)


def test_protocol_changes_when_native_formula_schema_changes(monkeypatch):
    before = fuel.protocol()
    monkeypatch.setitem(fuel.SCHEMA, "ccc", {**fuel.SCHEMA["ccc"], "unit": "bad"})
    assert fuel.protocol() != before


def test_fitted_corrector_cannot_drop_any_fuel_input(tmp_path, monkeypatch):
    cfg, data, day, work, _ = inputs(tmp_path)
    def drop(**kwargs):
        result = fake_base(**kwargs)
        result.residual_daily_audit.at[1, "residual_feature_columns"] = [base.NUCLEAR_KNOWN_COLUMN]
        return result
    monkeypatch.setattr(base, "run_nuclear_forecast", drop)
    with pytest.raises(fuel.CleanFuelForecastError, match="dropped"):
        fuel.run_clean_fuel_forecast(config=cfg, data=data, zone="FR", delivery_day=day, workdir=work,
            residual_factory=FakeResidual, kalman_builder=delegate)


def test_default_kalman_cache_stays_short_and_isolated(tmp_path, monkeypatch):
    cfg, data, day, work, _ = inputs(tmp_path)
    monkeypatch.setattr(base, "run_nuclear_forecast", fake_base)
    monkeypatch.setattr(fuel, "build_operational_kalman_view", delegate)
    result = fuel.run_clean_fuel_forecast(config=cfg, data=data, zone="FR", delivery_day=day, workdir=work, residual_factory=FakeResidual)
    assert plain_windows_path(result.kalman_view.captured["rolling_refit_cache_dir"]) == work / "kalman_cache"


def test_real_kalman_delegate_accepts_timestamp_and_fuel_schema_before_expensive_replay(tmp_path, monkeypatch):
    import chronos2_hourly.kalman_residual as native
    cfg, data, day, work, expected = inputs(tmp_path)
    historical = expected[expected < pd.Timestamp(day, tz=data.timezone)]
    future = expected[expected >= pd.Timestamp(day, tz=data.timezone)]
    statistics = pd.DataFrame({"delivery_start_utc": historical, "actual": 90., "residual_correction": 0.,
        "residual_corrected__q10": 70., "residual_corrected__q50": 90., "residual_corrected__q90": 110.})
    forecast = pd.DataFrame({"delivery_start_utc": future, "residual_correction": 0.,
        "residual_corrected__q10": 70., "residual_corrected__q50": 90., "residual_corrected__q90": 110.})
    covariates = data.model_context_covariates.loc[expected, [*BASE_RESIDUAL_LOAD_COVARIATES, base.NUCLEAR_ALIAS]].rename_axis("timestamp").reset_index()
    class ReachedNativeReplay(Exception):
        pass
    def validated(*args, **kwargs):
        assert "timestamp" in kwargs["covariates"]
        assert all(alias in kwargs["covariates"] for alias in fuel.FUEL_ALIASES)
        assert np.isfinite(kwargs["covariates"][list(fuel.FUEL_ALIASES)].to_numpy()).all()
        raise ReachedNativeReplay
    monkeypatch.setattr(native, "replay_kalman_overlay", validated)
    hook = fuel._FuelKalmanBuilder(data.model_context_covariates, native.build_operational_kalman_view, "signature", False)
    with pytest.raises(ReachedNativeReplay):
        hook(statistics=statistics, source_forecast=forecast, covariates=covariates,
            timezone=data.timezone, delivery_day=day, covariate_config=base.nuclear_kalman_covariate_config(),
            upstream_model="residual_corrected", output_model="residual_kalman", training_lookback_days=365,
            rolling_refit_cache_dir=work / "cache")


class FakeForecasting:
    @classmethod
    def run_backtest_variant(cls, **kwargs):
        assert kwargs["with_covariates"] is True
        data, parts = kwargs["data"], []
        for origin in kwargs["origins"]:
            index = data.target.index[origin:origin + kwargs["horizon"]]
            context = data.model_context_covariates.loc[index]
            for alias in fuel.FUEL_ALIASES:
                assert alias in context and f"known_{alias}_oracle" in data.known_future_columns
            middle = sum(context[known] for known in fuel.FUEL_KNOWN_COLUMNS)
            parts.append(pd.DataFrame({"timestamp": index, "q10": middle.to_numpy() - 5, "q50": middle.to_numpy(),
                "q90": middle.to_numpy() + 5, "actual": data.target.loc[index].to_numpy(), "origin_index": origin}))
        return pd.concat(parts, ignore_index=True)

    @classmethod
    def run_live_forecast_variant(cls, **kwargs):
        assert kwargs["with_covariates"] is True
        data = kwargs["data"]
        index = pd.date_range(data.target.index[-1] + pd.Timedelta(hours=1), periods=kwargs["horizon"], freq="h")
        middle = sum(data.model_context_covariates.loc[index, known] for known in fuel.FUEL_KNOWN_COLUMNS)
        return pd.DataFrame({"timestamp": index, "q10": middle.to_numpy() - 5, "q50": middle.to_numpy(), "q90": middle.to_numpy() + 5})


def test_real_three_stage_replay_with_lightweight_model_hooks(tmp_path_factory):
    # Base engine's sealed-cache SHA is deliberately full-length. Keep the
    # fixture root short enough for Windows' 260-character path limitation.
    tmp_path = tmp_path_factory.mktemp("cf")
    cfg, data, day, work, _ = inputs(tmp_path)
    work = tmp_path / "runs/experiments/nyx_clean_fuel_full_v1/w"
    result = fuel.run_clean_fuel_forecast(config=cfg, data=data, zone="FR", delivery_day=day, workdir=work,
        runtime_factory=lambda *args: object(), forecasting_module=FakeForecasting, residual_factory=FakeResidual,
        kalman_builder=delegate, threads=1)
    native_sum = sum(data.model_context_covariates[alias].iloc[-1] for alias in fuel.FUEL_ALIASES)
    assert len(result.residual_daily_audit) == 731
    assert len(result.source_forecast) == 24
    assert result.source_forecast.chronos2__q50.iloc[0] == pytest.approx(native_sum)
    assert result.source_forecast.residual_corrected__q50.iloc[0] == pytest.approx(native_sum + .01 * data.model_context_covariates.ccc.iloc[-1])
    assert all(alias in result.kalman_view.captured["covariates"] for alias in fuel.FUEL_ALIASES)
    assert result.audit["lora_used"] is False
    assert result.audit["storm_used_as_input"] is False
    assert result.audit["production_changed"] is False
    assert not (tmp_path / "runs/live").exists()
    assert not (tmp_path / "runs/experiments/nuclear_forecast_v1").exists()
