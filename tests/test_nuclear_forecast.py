from __future__ import annotations

import copy
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.nuclear_forecast as engine
from chronos2_hourly.chronos_adapter import generate_delivery_plans
from chronos2_hourly.historical_price_replay import HistoricalPriceReplayError
from chronos2_hourly.kalman_covariates import BASE_RESIDUAL_LOAD_COVARIATES


class FakeResidual:
    min_training_rows = 48
    feature_builder_options = {"include_calendar": False, "include_daily_profiles": False}

    def fit(self, X, actual, base, experts):
        self.feature_columns_ = (engine.NUCLEAR_KNOWN_COLUMN,)
        self.fit_index = X.index
        self.shift = float((actual - base.q50).mean())
        return self

    def predict(self, X, base, experts):
        assert self.fit_index.max() < X.index.min()
        return base.add(self.shift + X[engine.NUCLEAR_KNOWN_COLUMN] * 0.01, axis=0)


def _small_inputs(day: str = "2024-04-02", days: int = 5):
    delivery = date.fromisoformat(day)
    plans = generate_delivery_plans(
        delivery - timedelta(days=days), delivery, timezone="Europe/Paris",
        forecast_origin_local_time="08:00",
    )
    pieces = [pd.DataFrame({
        "q10": 30.0, "q50": 40.0, "q90": 50.0,
        "actual": 45.0, "forecast_origin_utc": plan.forecast_origin_utc,
    }, index=plan.delivery_index_utc) for plan in plans]
    history = pd.concat(pieces[:-1])
    future = pieces[-1].drop(columns="actual")
    X = pd.DataFrame({engine.NUCLEAR_KNOWN_COLUMN: 35.0}, index=history.index.append(future.index))
    return history, future, X, delivery


def test_daily_residual_is_causal_preserves_dst_and_nuclear_changes_output():
    history, future, X, delivery = _small_inputs()
    first, forecast, audit = engine.causal_residual_replay(
        raw_history=history, raw_future=future, features=X, timezone="Europe/Paris",
        delivery_day=delivery, residual_factory=FakeResidual,
    )
    assert audit.forecast_hours.tolist() == [24, 24, 24, 23, 24, 24]
    assert audit.generation_source.iloc[:2].eq("identity_chronos_cold_start").all()
    assert audit.causality_violations.sum() == 0
    fit_days = audit.dropna(subset=["fit_end_day"])
    assert (fit_days.fit_end_day < fit_days.delivery_day).all()
    assert "actual" not in forecast
    assert forecast.residual_corrected__q50.iloc[0] == pytest.approx(45.35)
    changed = history.copy()
    last_day = changed.index.tz_convert("Europe/Paris").date == delivery - timedelta(days=1)
    changed.loc[last_day, "actual"] = 5000.0
    second, changed_future, _ = engine.causal_residual_replay(
        raw_history=changed, raw_future=future, features=X, timezone="Europe/Paris",
        delivery_day=delivery, residual_factory=FakeResidual,
    )
    pd.testing.assert_series_equal(first.residual_corrected__q50, second.residual_corrected__q50)
    assert changed_future.residual_corrected__q50.iloc[0] > forecast.residual_corrected__q50.iloc[0]
    changed_X = X.copy()
    changed_X.loc[future.index, engine.NUCLEAR_KNOWN_COLUMN] = 50.0
    _, nuclear_future, _ = engine.causal_residual_replay(
        raw_history=history, raw_future=future, features=changed_X, timezone="Europe/Paris",
        delivery_day=delivery, residual_factory=FakeResidual,
    )
    assert nuclear_future.residual_corrected__q50.iloc[0] != forecast.residual_corrected__q50.iloc[0]


def test_future_actuals_and_noncausal_feature_exclusion_are_rejected():
    history, future, X, delivery = _small_inputs()
    with pytest.raises(engine.NuclearForecastError, match="Future observations"):
        engine.causal_residual_replay(
            raw_history=history, raw_future=future.assign(actual=999), features=X,
            timezone="Europe/Paris", delivery_day=delivery, residual_factory=FakeResidual,
        )
    class DroppedNuclear(FakeResidual):
        feature_builder_options = {"exclude_columns": [engine.NUCLEAR_KNOWN_COLUMN]}
    with pytest.raises(engine.NuclearForecastError, match="retain nuclear"):
        engine.causal_residual_replay(
            raw_history=history, raw_future=future, features=X,
            timezone="Europe/Paris", delivery_day=delivery, residual_factory=DroppedNuclear,
        )


def test_real_residual_factory_consumes_nuclear_with_tiny_catboost_fits():
    from run_chronos2_hourly import _residual_corrector_factory

    history, future, X, delivery = _small_inputs()
    # Nonconstant synthetic residual labels and a genuinely varying nuclear
    # feature exercise CatBoost and the real feature builder in four tiny fits.
    history["actual"] += np.sin(np.arange(len(history), dtype=float) / 5.0)
    X[engine.NUCLEAR_KNOWN_COLUMN] += np.cos(np.arange(len(X), dtype=float) / 7.0)
    factory, base_model = _residual_corrector_factory(
        {"hourly": {"residual_correction": {
            "enabled": True, "base_model": "chronos2", "backend": "catboost",
            "min_training_rows": 48, "iterations": 2, "depth": 2,
            "min_samples_leaf": 2, "thread_count": 1, "verbose": False,
            "feature_builder": {
                "timezone": "Europe/Paris", "include_calendar": True,
                "include_daily_profiles": True, "exclude_historical_prices": True,
            },
        }}},
        timezone="Europe/Paris",
    )
    assert base_model == "chronos2" and factory is not None
    history_before, features_before = history.copy(deep=True), X.copy(deep=True)
    statistics, forecast, audit = engine.causal_residual_replay(
        raw_history=history, raw_future=future, features=X,
        timezone="Europe/Paris", delivery_day=delivery, residual_factory=factory,
    )
    columns = [f"residual_corrected__{q}" for q in engine.QUANTILES]
    assert np.isfinite(statistics[columns].to_numpy(dtype=float)).all()
    assert np.isfinite(forecast[columns].to_numpy(dtype=float)).all()
    assert (forecast.residual_corrected__q10 <= forecast.residual_corrected__q50).all()
    assert (forecast.residual_corrected__q50 <= forecast.residual_corrected__q90).all()
    fitted = audit.loc[audit.generation_source.eq("daily_prequential_refit")]
    assert len(fitted) == 4
    assert fitted.nuclear_feature_used.all()
    assert all(engine.NUCLEAR_KNOWN_COLUMN in columns for columns in fitted.residual_feature_columns)
    assert (fitted.fit_end_day < fitted.delivery_day).all()
    assert "actual" not in forecast
    pd.testing.assert_frame_equal(history, history_before)
    pd.testing.assert_frame_equal(X, features_before)


def test_raw_residual_history_must_be_contiguous_complete_local_days():
    history, future, X, delivery = _small_inputs()
    with pytest.raises(engine.NuclearForecastError, match="complete contiguous"):
        engine.causal_residual_replay(
            raw_history=history.iloc[1:], raw_future=future, features=X,
            timezone="Europe/Paris", delivery_day=delivery, residual_factory=FakeResidual,
        )


def test_raw_origins_cannot_be_backdated_or_late():
    history, future, X, delivery = _small_inputs()
    history["forecast_origin_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(engine.NuclearForecastError, match="D-1 08:00"):
        engine.causal_residual_replay(
            raw_history=history, raw_future=future, features=X,
            timezone="Europe/Paris", delivery_day=delivery, residual_factory=FakeResidual,
        )


def test_nuclear_is_added_to_the_active_market_group_without_mixing_units():
    contract = engine.nuclear_kalman_covariate_config()
    assert engine.NUCLEAR_ALIAS in contract.input_columns
    assert engine.NUCLEAR_ALIAS in contract.groups["market"]
    assert all(engine.NUCLEAR_ALIAS not in item.sources for item in contract.derived)
    assert contract.minimum_history_coverage == 1.0
    assert contract.require_future_complete is True


def _prepared(tmp_path: Path, delivery: date = date(2026, 9, 8)):
    timezone = "Europe/Paris"
    # Three extra context days, then exactly 730 historical and one future day.
    index = pd.date_range(
        pd.Timestamp(delivery - timedelta(days=733), tz=timezone),
        pd.Timestamp(delivery + timedelta(days=1), tz=timezone),
        freq="h", inclusive="left",
    ).tz_convert("UTC")
    future = generate_delivery_plans(
        delivery, delivery, timezone=timezone, forecast_origin_local_time="08:00",
    )[0].delivery_index_utc
    target_index = index[index < future[0]]
    target = pd.Series(55.0 + np.sin(np.arange(len(target_index)) / 20), index=target_index)
    context = pd.DataFrame(index=index)
    aliases = (*BASE_RESIDUAL_LOAD_COVARIATES, engine.NUCLEAR_ALIAS)
    for position, alias in enumerate(aliases):
        context[alias] = 25.0 + position
        context[f"known_{alias}_oracle"] = context[alias]
    data = SimpleNamespace(
        zone="FR", timezone=timezone, target=target,
        covariates=context.loc[target_index, list(aliases)].copy(),
        model_context_covariates=context,
        known_future_columns=[f"known_{alias}_oracle" for alias in aliases],
    )
    config = {
        "data": {"project_root": str(tmp_path), "forecast_origin_local_time": "08:00"},
        "model": {"context_length": 48, "model_id": "amazon/chronos-2", "local_files_only": True},
        "zones": {"FR": {"covariates": {engine.NUCLEAR_ALIAS: {
            "enabled": True, "source": "pit_parquet", "series": engine.NUCLEAR_SERIES,
        }}}},
        "hourly": {
            "feature_engineering": {"target_lags": [24], "target_rolling_windows": [24]},
            "residual_correction": {
                "enabled": True, "base_model": "chronos2", "backend": "catboost",
                "min_training_rows": 48,
                "feature_builder": {"exclude_historical_prices": True},
            },
        },
    }
    return config, data, delivery


class FakeForecasting:
    historical_calls = 0
    live_calls = 0

    @classmethod
    def run_backtest_variant(cls, **kwargs):
        cls.historical_calls += 1
        assert kwargs["with_covariates"] is True
        data = kwargs["data"]
        frames = []
        for origin in kwargs["origins"]:
            index = data.target.index[origin: origin + kwargs["horizon"]]
            middle = data.model_context_covariates.loc[index, engine.NUCLEAR_KNOWN_COLUMN] + 10
            frames.append(pd.DataFrame({
                "timestamp": index, "q10": middle.to_numpy() - 5,
                "q50": middle.to_numpy(), "q90": middle.to_numpy() + 5,
                "actual": data.target.loc[index].to_numpy(),
            }))
        return pd.concat(frames, ignore_index=True)

    @classmethod
    def run_live_forecast_variant(cls, **kwargs):
        cls.live_calls += 1
        assert kwargs["with_covariates"] is True
        data = kwargs["data"]
        index = pd.date_range(data.target.index[-1] + pd.Timedelta(hours=1), periods=kwargs["horizon"], freq="h")
        middle = data.model_context_covariates.loc[index, engine.NUCLEAR_KNOWN_COLUMN] + 10
        return pd.DataFrame({"timestamp": index, "q10": middle.to_numpy() - 5,
                             "q50": middle.to_numpy(), "q90": middle.to_numpy() + 5})


def test_full_engine_new_upstream_is_shared_and_raw_checkpoint_reuse_is_identity_pinned(tmp_path):
    config, data, delivery = _prepared(tmp_path)
    config["nuclear_experiment"] = {"filter_parameters": {"q_over_r": 0.002}}
    original = copy.deepcopy(config)
    target_copy = data.target.copy()
    calls = []
    captured = []
    def runtime_factory(config, device, local_only):
        calls.append((device, local_only))
        return object()
    def kalman_builder(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(forecast=kwargs["source_forecast"].copy())
    workdir = tmp_path / "runs" / "experiments" / "nuclear_test" / "work"
    arguments = dict(
        config=config, data=data, zone="FR", delivery_day=delivery, workdir=workdir,
        runtime_factory=runtime_factory, forecasting_module=FakeForecasting,
        residual_factory=FakeResidual, kalman_builder=kalman_builder,
    )
    FakeForecasting.historical_calls = 0
    result = engine.run_nuclear_forecast(**arguments)
    assert FakeForecasting.historical_calls > 20
    assert len(result.residual_daily_audit) == 731
    assert result.residual_daily_audit.phase.eq("diagnostic_warmup").sum() == 365
    assert result.residual_daily_audit.phase.eq("evaluation").sum() == 365
    assert result.audit["nuclear_units"] == "GW"
    assert result.audit["incumbent_upstream_prefix_used"] is False
    assert result.audit["lora_used"] is False
    pd.testing.assert_frame_equal(captured[0]["statistics"], result.residual_statistics)
    pd.testing.assert_frame_equal(captured[0]["source_forecast"], result.source_forecast)
    assert captured[0]["upstream_model"] == "residual_corrected"
    assert captured[0]["training_lookback_days"] == 365
    assert captured[0]["config"].q_over_r == 0.002
    assert captured[0]["rolling_refit_cache_dir"].is_relative_to(workdir)
    assert engine.NUCLEAR_ALIAS in captured[0]["covariate_config"].groups["market"]
    assert "actual" not in captured[0]["source_forecast"]
    assert config == original
    pd.testing.assert_series_equal(data.target, target_copy)
    first_calls = FakeForecasting.historical_calls
    first_live_calls = FakeForecasting.live_calls
    first_runtime_calls = len(calls)
    resumed = engine.run_nuclear_forecast(**arguments)
    assert FakeForecasting.historical_calls == first_calls
    assert FakeForecasting.live_calls == first_live_calls
    assert len(calls) == first_runtime_calls
    assert resumed.audit["raw_future_cache_hit"] is True
    assert resumed.audit["residual_result_cache_hit"] is True
    pd.testing.assert_frame_equal(resumed.residual_statistics, result.residual_statistics)
    changed_filter = copy.deepcopy(config)
    changed_filter["nuclear_experiment"]["filter_parameters"]["q_over_r"] = 0.003
    with pytest.raises(HistoricalPriceReplayError, match="source_hashes"):
        engine.run_nuclear_forecast(**{**arguments, "config": changed_filter})
    assert FakeForecasting.historical_calls == first_calls
    assert FakeForecasting.live_calls == first_live_calls
    import torch
    assert torch.get_num_threads() == 4
    pd.testing.assert_frame_equal(resumed.raw_history, result.raw_history)
    changed_data = copy.deepcopy(data)
    changed_data.model_context_covariates[engine.NUCLEAR_ALIAS] += 1
    changed_data.model_context_covariates[engine.NUCLEAR_KNOWN_COLUMN] += 1
    with pytest.raises(HistoricalPriceReplayError, match="source_hashes"):
        engine.run_nuclear_forecast(**{**arguments, "data": changed_data})
    cache_forecast = Path(result.audit["result_cache_directory"]) / "residual" / "forecast.parquet"
    cache_forecast.write_bytes(b"corrupt-cache")
    with pytest.raises(engine.NuclearForecastError, match="checksum mismatch"):
        engine.run_nuclear_forecast(**arguments)
    assert FakeForecasting.live_calls == first_live_calls


@pytest.mark.parametrize("fault", ["series", "future_hole", "model_context", "feature_allowlist", "target_horizon", "production_workdir", "competing_input", "lora_model", "invalid_filter", "inactive_nuclear_filter", "residual_exclusion", "unreachable_minimum"])
def test_bad_inputs_fail_before_model_loading_or_output(tmp_path, fault):
    config, data, delivery = _prepared(tmp_path)
    workdir = tmp_path / "runs" / "experiments" / "nuclear_test" / "work"
    if fault == "series":
        config["zones"]["FR"]["covariates"][engine.NUCLEAR_ALIAS]["series"] = "power.fr.generation.nuclear.remit.mw.fcst"
    elif fault == "future_hole":
        data.model_context_covariates.loc[data.model_context_covariates.index[-1], engine.NUCLEAR_KNOWN_COLUMN] = np.nan
    elif fault == "model_context":
        data.model_context_covariates = data.model_context_covariates.drop(columns=engine.NUCLEAR_ALIAS)
    elif fault == "feature_allowlist":
        config["hourly"]["feature_engineering"]["covariate_columns"] = ["known_fr_residual_load_fcst_oracle"]
    elif fault == "target_horizon":
        delivery += timedelta(days=1)
    elif fault == "production_workdir":
        workdir = tmp_path / "runs" / "live" / "danger"
    elif fault == "competing_input":
        data.model_context_covariates["mkonline_price"] = 100.0
    elif fault == "lora_model":
        config["model"]["model_id"] = "local-promoted-lora"
    elif fault == "invalid_filter":
        config["nuclear_experiment"] = {"filter_parameters": {"arbitrary_field": 100}}
    elif fault == "inactive_nuclear_filter":
        config["nuclear_experiment"] = {"filter_parameters": {"candidate_kinds": ["linear_bias"]}}
    elif fault == "residual_exclusion":
        config["hourly"]["residual_correction"]["feature_builder"]["exclude_patterns"] = ["nuclear"]
    elif fault == "unreachable_minimum":
        config["hourly"]["residual_correction"]["min_training_rows"] = 1_000_000
    def forbidden(*args, **kwargs):
        raise AssertionError("No runtime should be loaded")
    with pytest.raises(ValueError):
        engine.run_nuclear_forecast(
            config=config, data=data, zone="FR", delivery_day=delivery,
            workdir=workdir, runtime_factory=forbidden,
        )
    assert not workdir.exists()
