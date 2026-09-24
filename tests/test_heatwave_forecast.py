from copy import deepcopy
from datetime import date, timedelta
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import heatwave_forecast as hw, nuclear_forecast as base
from chronos2_hourly.heatwave_features import heatwave_feature_aliases, temperature_aliases, HeatwaveFeatureConfig
from chronos2_hourly.chronos_adapter import generate_delivery_plans
from chronos2_hourly.kalman_covariates import BASE_RESIDUAL_LOAD_COVARIATES


class FakeResidual:
    min_training_rows = 24 * 365
    feature_builder_options = {"exclude_historical_prices": True, "include_calendar": False,
                               "include_daily_profiles": False}

    def fit(self, X, actual, prediction, experts):
        self.feature_columns_ = tuple(f"known_{a}_oracle" for a in (base.NUCLEAR_ALIAS, *heatwave_feature_aliases()))
        self.last_training = X.index.max()
        return self

    def predict(self, X, prediction, experts):
        assert self.last_training < X.index.min()
        return prediction.add(X.known_fr_temperature_fcst_oracle * .1, axis=0)


def inputs(tmp_path, day="2026-09-11", *, native=False, cwe=False):
    delivery = date.fromisoformat(day)
    plans = generate_delivery_plans(delivery - timedelta(days=730), delivery,
        timezone="Europe/Paris", forecast_origin_local_time="08:00")
    required = plans[0].delivery_index_utc
    for plan in plans[1:]:
        required = required.append(plan.delivery_index_utc)
    index = pd.date_range(required[0] - pd.Timedelta(hours=48), required[-1], freq="h")
    target_index = index[index < plans[-1].delivery_index_utc[0]]
    aliases = (*BASE_RESIDUAL_LOAD_COVARIATES, base.NUCLEAR_ALIAS, *heatwave_feature_aliases(),
               *(("be_nuclear_available_gw", "nl_nuclear_available_gw") if cwe else ()))
    context = pd.DataFrame(index=index)
    for i, alias in enumerate(aliases):
        value = (25. if alias in temperature_aliases() else 5. if alias.endswith("_heat_excess_fcst_c")
            else 3. if alias.endswith("_heat_streak_fcst_days") else 1. if alias == "heat_fraction_fcst"
            else 3. if alias == "cooling_degree_mean_fcst_c" else 4. if alias.startswith("be_nuclear")
            else .48 if alias.startswith("nl_nuclear") else 30. + i)
        context[alias] = value
        context[f"known_{alias}_oracle"] = value
    specs = {alias: {"enabled": True, "source": "pit_parquet", "fill_method": "none",
        "series": base.NUCLEAR_SERIES if alias == base.NUCLEAR_ALIAS else "pit_forecast_" + alias,
        "future": {"known_future": True, "strategies": ["oracle"]}} for alias in aliases}
    config = {"data": {"project_root": str(tmp_path), "forecast_origin_local_time": "08:00"},
        "zones": {"FR": {"covariates": specs}},
        "model": {"model_id": "amazon/chronos-2", "local_files_only": True, "context_length": 48},
        "hourly": {"feature_engineering": {"target_lags": [24], "target_rolling_windows": [24]},
            "residual_correction": {"enabled": True, "base_model": "chronos2", "backend": "catboost",
                "min_training_rows": 24 * 365, "feature_builder": {"exclude_historical_prices": True}}},
        "nuclear_experiment": {"input_protocol": hw.heatwave_input_protocol(),
                               "heatwave_features": HeatwaveFeatureConfig().to_dict()}}
    data = SimpleNamespace(zone="FR", timezone="Europe/Paris", target=pd.Series(55., index=target_index),
        covariates=context.loc[target_index, list(aliases)].copy(), model_context_covariates=context,
        known_future_columns=[f"known_{alias}_oracle" for alias in aliases])
    if native:
        for frame in (data.target, data.covariates, data.model_context_covariates):
            frame.index = frame.index.tz_convert("Europe/Paris")
    return config, data, delivery, tmp_path / "runs/experiments/heatwave_v1/fr", plans


def delegate(**kwargs):
    assert "timestamp" in kwargs["covariates"] and "delivery_start_utc" not in kwargs["covariates"]
    return SimpleNamespace(forecast=kwargs["source_forecast"].copy(), captured=kwargs)


def fake_base(**kwargs):
    data, day = kwargs["data"], kwargs["delivery_day"]
    context = data.model_context_covariates
    index = context.index[context.index >= pd.Timestamp(day - timedelta(days=730), tz="Europe/Paris")]
    covariates = pd.DataFrame({a: context.loc[index, f"known_{a}_oracle"]
                              for a in (*BASE_RESIDUAL_LOAD_COVARIATES, base.NUCLEAR_ALIAS)})
    stats = pd.DataFrame({"timestamp": index[:-24], "actual": 55.})
    forecast = pd.DataFrame({"timestamp": index[-24:], "residual_corrected__q50": 80.})
    view = kwargs["kalman_builder"](statistics=stats, source_forecast=forecast,
        covariates=covariates.rename_axis("timestamp").reset_index(),
        covariate_config=base.nuclear_kalman_covariate_config(), config=SimpleNamespace(q_over_r=.001),
        timezone="Europe/Paris", delivery_day=day, training_lookback_days=365,
        upstream_model="residual_corrected", output_model="residual_kalman",
        rolling_refit_cache_dir=Path(kwargs["workdir"]) / "kalman")
    columns = [f"known_{a}_oracle" for a in data.covariates if a not in BASE_RESIDUAL_LOAD_COVARIATES]
    daily = pd.DataFrame({"generation_source": ["identity_chronos_cold_start", "daily_prequential_refit"],
                          "residual_feature_columns": [[], columns]})
    return base.NuclearForecastResult(stats, stats, forecast, covariates, daily, view,
        {"engine": "nuclear_forecast_v1", "raw_history_start_day": str(day - timedelta(days=730)),
         "source_hashes": {"engine_file_sha256": "unchanged"}, "production_changed": False})


@pytest.mark.parametrize("cwe", [False, True])
def test_all_features_consumed_three_stages_without_mutating_config_or_inputs(tmp_path, monkeypatch, cwe):
    config, data, day, work, _ = inputs(tmp_path, cwe=cwe)
    saved_config, saved_data = deepcopy(config), deepcopy(data)
    monkeypatch.setattr(base, "run_nuclear_forecast", fake_base)
    result = hw.run_heatwave_forecast(config=config, data=data, zone="FR", delivery_day=day,
        workdir=work, residual_factory=FakeResidual, kalman_builder=delegate)
    capture = result.kalman_view.captured
    for alias in heatwave_feature_aliases():
        assert alias in capture["covariates"]
        assert alias in capture["covariate_config"].groups["market"]
        assert alias in result.audit["heatwave_chronos_context_columns"]
        assert f"known_{alias}_oracle" in result.audit["heatwave_residual_features"]
    assert capture["training_lookback_days"] == 365
    assert capture["config"].q_over_r == .001
    assert result.audit["engine"] == "nuclear_forecast_v1"
    assert result.audit["candidate_engine"] == hw.HEATWAVE_ENGINE
    assert result.audit["source_hashes"]["engine_file_sha256"] == "unchanged"
    assert result.audit["heatwave_base_variant"] == ("nuclear_cwe" if cwe else "nuclear_fr")
    if cwe:
        assert "be_nuclear_available_gw" in capture["covariate_config"].groups["market"]
    assert config == saved_config
    pd.testing.assert_frame_equal(data.model_context_covariates, saved_data.model_context_covariates)
    assert not work.exists()


@pytest.mark.parametrize("fault", ["source", "disabled", "observed", "fill", "future", "missing_base", "missing_known",
    "missing_historical", "history_mismatch", "undeclared", "nan", "vintage_mismatch", "units", "streak",
    "intraday", "fraction", "cooling", "residual_selection", "residual_excluded", "protocol", "production", "cwe_workdir", "cache"])
def test_unsafe_contract_stops_before_model_or_writes(tmp_path, monkeypatch, fault):
    config, data, day, work, _ = inputs(tmp_path)
    alias, known = "fr_temperature_fcst", "known_fr_temperature_fcst_oracle"
    spec, context = config["zones"]["FR"]["covariates"][alias], data.model_context_covariates
    if fault == "source": spec["source"] = "csv"
    elif fault == "disabled": spec["enabled"] = False
    elif fault == "observed": spec["series"] = "observed_temperature"
    elif fault == "fill": spec["fill_method"] = "ffill"
    elif fault == "future": spec["future"]["known_future"] = False
    elif fault == "missing_base": data.model_context_covariates = context.drop(columns=alias)
    elif fault == "missing_known": data.model_context_covariates = context.drop(columns=known)
    elif fault == "missing_historical": data.covariates = data.covariates.drop(columns=alias)
    elif fault == "history_mismatch": data.covariates.iloc[100, data.covariates.columns.get_loc(alias)] += 1
    elif fault == "undeclared": data.known_future_columns.remove(known)
    elif fault == "nan": context.iloc[-1, context.columns.get_indexer([alias, known])] = np.nan
    elif fault == "vintage_mismatch": context.loc[:, alias] += 1
    elif fault == "units":
        context.loc[:, [alias, known]] = 300.; data.covariates.loc[:, alias] = 300.
    elif fault == "streak":
        a = "fr_heat_streak_fcst_days"; context.loc[:, [a, f"known_{a}_oracle"]] = .5; data.covariates[a] = .5
    elif fault == "intraday":
        context.iloc[-1, context.columns.get_indexer([alias, known])] = 26.
    elif fault in {"fraction", "cooling"}:
        a = "heat_fraction_fcst" if fault == "fraction" else "cooling_degree_mean_fcst_c"
        context.loc[:, [a, f"known_{a}_oracle"]] = 0.; data.covariates[a] = 0.
    elif fault == "residual_selection": config["hourly"]["feature_engineering"]["covariate_columns"] = [base.NUCLEAR_KNOWN_COLUMN]
    elif fault == "residual_excluded": config["hourly"]["residual_correction"]["feature_builder"]["exclude_columns"] = [known]
    elif fault == "protocol": config["nuclear_experiment"]["input_protocol"] = "civil_pit_v2"
    elif fault == "production": work = tmp_path / "runs/live/fr"
    elif fault == "cwe_workdir": work = tmp_path / "runs/experiments/nuclear_cwe_v1/fr"
    elif fault == "cache": config["nuclear_experiment"].update(mode="incremental", incremental_cache_dir=str(tmp_path / "runs/experiments/nuclear_cwe_v1/cache"))
    def forbidden(**kwargs): raise AssertionError("No expensive engine call permitted")
    monkeypatch.setattr(base, "run_nuclear_forecast", forbidden)
    with pytest.raises(ValueError):
        hw.run_heatwave_forecast(config=config, data=data, zone="FR", delivery_day=day, workdir=work)
    assert not work.exists()


@pytest.mark.parametrize("day, hours", [("2026-03-29", 23), ("2025-10-26", 25)])
def test_native_local_and_utc_inputs_preserve_physical_dst(tmp_path, day, hours):
    for native in (False, True):
        config, data, parsed, work, _ = inputs(tmp_path, day, native=native)
        # Optional earlier exogenous history is not backfilled using hindsight.
        for alias in heatwave_feature_aliases():
            data.model_context_covariates.iloc[:48, data.model_context_covariates.columns.get_indexer(
                [alias, f"known_{alias}_oracle"])] = np.nan
        _, aliases, coverage, expected = hw._prepare_inputs(config, data, "FR", parsed)
        assert (expected.tz_convert("Europe/Paris").date == parsed).sum() == hours
        assert coverage["fr_temperature_fcst"]["missing_optional_pre_anchor_context_hours"] == 48
        assert len(aliases) == 17


def test_preparation_accepts_actual_binary32_rounding_with_a_precise_audit(tmp_path):
    config, data, day, _, _ = inputs(tmp_path, native=True)
    context = data.model_context_covariates
    temperatures = [26.2345699, 24.1234567, 25.8765432, 19.9876543, 22.1234987]
    for alias, value in zip(temperature_aliases(), temperatures):
        context.loc[:, [alias, f"known_{alias}_oracle"]] = value
        data.covariates[alias] = value
    for country in ("nl", "es"):
        alias = f"{country}_heat_streak_fcst_days"
        context.loc[:, [alias, f"known_{alias}_oracle"]] = 0.
        data.covariates[alias] = 0.
    for alias, value in (("heat_fraction_fcst", .6),
            ("cooling_degree_mean_fcst_c", np.maximum(np.array(temperatures) - 22, 0).mean())):
        context.loc[:, [alias, f"known_{alias}_oracle"]] = value
        data.covariates[alias] = value
    data.model_context_covariates = context.astype("float32")
    data.covariates = data.covariates.astype("float32")
    _, _, coverage, _ = hw._prepare_inputs(config, data, "FR", day)
    audit = coverage["heat_fraction_fcst"]["aggregate_precision"]
    assert audit["fraction_max_abs_difference"] > 1e-9
    assert audit["cooling_max_abs_difference"] > 1e-9
    assert audit["feature_values_modified"] is False


def test_protocol_pins_both_code_and_config(monkeypatch):
    expected = hashlib.sha256(Path(hw.__file__).read_bytes()).hexdigest()
    assert hw.heatwave_code_sha256() == expected
    original = hw.heatwave_input_protocol()
    assert original != hw.heatwave_input_protocol({"cooling_threshold_c": 21.})
    monkeypatch.setattr(hw, "heatwave_features_sha256", lambda: "new-feature-code")
    assert hw.heatwave_input_protocol() != original


def test_existing_kalman_aggregates_are_not_mixed_with_degrees():
    original = base.nuclear_kalman_covariate_config()
    enriched = hw.heatwave_kalman_covariate_config(original)
    assert enriched.derived == original.derived
    assert len(enriched.groups["market"]) + 6 <= 64
    assert enriched.history_missing_policy == "complete_trailing"
    assert enriched.minimum_history_coverage == 1
    assert all(alias not in original.input_columns for alias in heatwave_feature_aliases())


def test_default_builder_normalizes_timestamp_and_windows_path(tmp_path, monkeypatch):
    config, data, day, work, _ = inputs(tmp_path)
    monkeypatch.setattr(base, "run_nuclear_forecast", fake_base)
    monkeypatch.setattr(hw, "build_operational_kalman_view", delegate)
    result = hw.run_heatwave_forecast(config=config, data=data, zone="FR", delivery_day=day,
        workdir=work, residual_factory=FakeResidual)
    path = result.kalman_view.captured["rolling_refit_cache_dir"]
    assert str(path).endswith(str(work / "kalman"))
    assert result.audit["heatwave_kalman_delegate_signature"] is None


def test_post_fit_drop_is_refused(tmp_path, monkeypatch):
    config, data, day, work, _ = inputs(tmp_path)
    def dropped(**kwargs):
        result = fake_base(**kwargs)
        result.residual_daily_audit.at[1, "residual_feature_columns"] = [base.NUCLEAR_KNOWN_COLUMN]
        return result
    monkeypatch.setattr(base, "run_nuclear_forecast", dropped)
    with pytest.raises(hw.HeatwaveForecastError, match="dropped"):
        hw.run_heatwave_forecast(config=config, data=data, zone="FR", delivery_day=day,
            workdir=work, residual_factory=FakeResidual, kalman_builder=delegate)


class FakeForecasting:
    @classmethod
    def run_backtest_variant(cls, **kwargs):
        assert kwargs["with_covariates"] is True
        data, chunks = kwargs["data"], []
        for origin in kwargs["origins"]:
            index = data.target.index[origin:origin + kwargs["horizon"]]
            middle = sum(data.model_context_covariates.loc[index, f"known_{a}_oracle"]
                         for a in (base.NUCLEAR_ALIAS, *heatwave_feature_aliases()))
            chunks.append(pd.DataFrame({"timestamp": index, "q10": middle.to_numpy() - 5., "q50": middle.to_numpy(),
                "q90": middle.to_numpy() + 5., "actual": data.target.loc[index].to_numpy(), "origin_index": origin}))
        return pd.concat(chunks, ignore_index=True)

    @classmethod
    def run_live_forecast_variant(cls, **kwargs):
        assert kwargs["with_covariates"] is True
        data = kwargs["data"]
        index = pd.date_range(data.target.index[-1] + pd.Timedelta(hours=1), periods=kwargs["horizon"], freq="h")
        middle = sum(data.model_context_covariates.loc[index, f"known_{a}_oracle"]
                     for a in (base.NUCLEAR_ALIAS, *heatwave_feature_aliases()))
        return pd.DataFrame({"timestamp": index, "q10": middle.to_numpy() - 5., "q50": middle.to_numpy(), "q90": middle.to_numpy() + 5.})


def test_full_730_day_engine_is_reused_with_all_new_channels(tmp_path):
    config, data, day, work, _ = inputs(tmp_path)
    result = hw.run_heatwave_forecast(config=config, data=data, zone="FR", delivery_day=day,
        workdir=work, runtime_factory=lambda *args: object(), forecasting_module=FakeForecasting,
        residual_factory=FakeResidual, kalman_builder=delegate, threads=1)
    assert len(result.residual_daily_audit) == 731
    forecast = result.source_forecast
    assert len(forecast) == 24
    expected = 35. + 5 * 25. + 5 * 5. + 5 * 3. + 1. + 3.
    assert forecast.chronos2__q50.iloc[0] == pytest.approx(expected)
    assert forecast.residual_corrected__q50.iloc[0] == pytest.approx(expected + 2.5)
    assert result.kalman_view.captured["training_lookback_days"] == 365
    assert result.audit["production_changed"] is False
    assert result.audit["lora_used"] is False
    assert not (tmp_path / "runs/live").exists()
