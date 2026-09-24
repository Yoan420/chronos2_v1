from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.nuclear_cwe_forecast as cwe
from chronos2_hourly import nuclear_forecast as base
from chronos2_hourly.chronos_adapter import generate_delivery_plans
from chronos2_hourly.kalman_covariates import BASE_RESIDUAL_LOAD_COVARIATES


class FakeResidual:
    min_training_rows = 24 * 365
    feature_builder_options = {"exclude_historical_prices": True,
                               "include_calendar": False, "include_daily_profiles": False}

    def fit(self, X, actual, prediction, experts):
        self.feature_columns_ = tuple(f"known_{alias}_oracle" for alias in cwe.CWE_NUCLEAR_ALIASES)
        self.last_training = X.index.max()
        return self

    def predict(self, X, prediction, experts):
        assert self.last_training < X.index.min()
        return prediction.add(X.known_be_nuclear_available_gw_oracle * .1, axis=0)


def _inputs(tmp_path, day="2026-09-11"):
    delivery = date.fromisoformat(day)
    plans = generate_delivery_plans(delivery - timedelta(days=730), delivery,
                                    timezone="Europe/Paris", forecast_origin_local_time="08:00")
    required = plans[0].delivery_index_utc
    for plan in plans[1:]:
        required = required.append(plan.delivery_index_utc)
    index = pd.date_range(required[0] - pd.Timedelta(hours=48), required[-1], freq="h")
    target_index = index[index < plans[-1].delivery_index_utc[0]]
    context = pd.DataFrame(index=index)
    all_aliases = (*BASE_RESIDUAL_LOAD_COVARIATES, *cwe.CWE_NUCLEAR_ALIASES)
    for i, alias in enumerate(all_aliases):
        value = (4.0 if alias.startswith("be_nuclear") else .48 if alias.startswith("nl_nuclear") else 30 + i)
        context[alias] = value
        context[f"known_{alias}_oracle"] = value
    specs = {alias: {"enabled": True, "source": "pit_parquet", "series": series,
                     "future": {"known_future": True, "strategies": ["oracle"]}}
             for alias, series in {base.NUCLEAR_ALIAS: base.NUCLEAR_SERIES,
                                   **cwe.CWE_AVAILABILITY_SERIES}.items()}
    config = {
        "data": {"project_root": str(tmp_path), "forecast_origin_local_time": "08:00"},
        "zones": {"FR": {"covariates": specs}},
        "model": {"model_id": "amazon/chronos-2", "local_files_only": True, "context_length": 48},
        "hourly": {"feature_engineering": {"target_lags": [24], "target_rolling_windows": [24]},
                   "residual_correction": {"enabled": True, "base_model": "chronos2",
                       "backend": "catboost", "min_training_rows": 24 * 365,
                       "feature_builder": {"exclude_historical_prices": True}}},
        "nuclear_experiment": {"input_protocol": cwe.nuclear_cwe_input_protocol()},
    }
    data = SimpleNamespace(zone="FR", timezone="Europe/Paris",
        target=pd.Series(55.0, index=target_index),
        covariates=context.loc[target_index, list(all_aliases)].copy(),
        model_context_covariates=context,
        known_future_columns=[f"known_{alias}_oracle" for alias in all_aliases])
    workdir = tmp_path / "runs" / "experiments" / "cwe" / "w"
    return config, data, delivery, workdir, plans


def _delegate(**kwargs):
    return SimpleNamespace(forecast=kwargs["source_forecast"].copy(), captured=kwargs)


def _fake_base(**kwargs):
    data, delivery = kwargs["data"], kwargs["delivery_day"]
    context = data.model_context_covariates
    index = context.index[context.index >= pd.Timestamp(delivery - timedelta(days=730), tz="Europe/Paris")]
    cov = pd.DataFrame({alias: context.loc[index, f"known_{alias}_oracle"]
                       for alias in (*BASE_RESIDUAL_LOAD_COVARIATES, base.NUCLEAR_ALIAS)})
    source = pd.DataFrame({"timestamp": index[-24:], "residual_corrected__q50": 80.})
    stats = pd.DataFrame({"timestamp": index[:-24], "actual": 55.})
    kalman = kwargs["kalman_builder"](
        statistics=stats, source_forecast=source, covariates=cov.rename_axis("timestamp").reset_index(),
        covariate_config=base.nuclear_kalman_covariate_config(), config=SimpleNamespace(q_over_r=.001),
        timezone="Europe/Paris", delivery_day=delivery, training_lookback_days=365,
        upstream_model="residual_corrected", output_model="residual_kalman",
        rolling_refit_cache_dir=Path(kwargs["workdir"]) / "kalman_cache")
    feature_columns = [f"known_{alias}_oracle" for alias in cwe.CWE_NUCLEAR_ALIASES]
    daily = pd.DataFrame({"generation_source": ["identity_chronos_cold_start", "daily_prequential_refit"],
                          "residual_feature_columns": [[], feature_columns]})
    return base.NuclearForecastResult(stats, stats, source, cov, daily, kalman,
        {"engine": "nuclear_forecast_v1", "raw_history_start_day": str(delivery - timedelta(days=730)),
         "source_hashes": {"engine_file_sha256": "unchanged"}, "production_changed": False})


def test_wrapper_enriches_every_kalman_group_preserves_recipe_and_inputs(tmp_path, monkeypatch):
    config, data, day, work, _ = _inputs(tmp_path)
    saved_config, saved_context = deepcopy(config), data.model_context_covariates.copy(deep=True)
    monkeypatch.setattr(base, "run_nuclear_forecast", _fake_base)
    result = cwe.run_nuclear_cwe_forecast(config=config, data=data, zone="FR", delivery_day=day,
        workdir=work, residual_factory=FakeResidual, kalman_builder=_delegate)
    capture = result.kalman_view.captured
    for alias in cwe.CWE_NUCLEAR_ALIASES:
        assert alias in result.covariates
        assert alias in capture["covariate_config"].input_columns
        assert all(alias in group for group in capture["covariate_config"].groups.values())
    assert capture["training_lookback_days"] == 365
    assert capture["config"].q_over_r == .001
    assert capture["upstream_model"] == "residual_corrected"
    assert capture["rolling_refit_cache_dir"].parent == work / "kalman_cache"
    assert result.audit["engine"] == "nuclear_forecast_v1"
    assert result.audit["candidate_engine"] == cwe.CWE_ENGINE
    assert result.audit["nuclear_covariate_semantics"]["be_nuclear_available_gw"]["semantic"] == "forecast_available_capacity"
    assert result.audit["nuclear_covariate_semantics"][base.NUCLEAR_ALIAS]["semantic"] == "forecast_generation"
    assert result.audit["source_hashes"]["engine_file_sha256"] == "unchanged"
    assert config == saved_config
    pd.testing.assert_frame_equal(data.model_context_covariates, saved_context)
    assert not work.exists()


@pytest.mark.parametrize("fault", ["series", "disabled", "source", "generation_label", "units", "broadcast_flag",
    "future_spec", "missing_base", "missing_known", "missing_history_alias", "history_mismatch",
    "undeclared_known", "future_nan", "history_nan",
    "vintage_mismatch", "negative", "mw_units", "intraday_change", "residual_selection",
    "residual_excluded", "late_protocol", "incumbent_workdir", "production_workdir", "foreign_cache"])
def test_invalid_cwe_contract_fails_before_base_engine_or_writes(tmp_path, monkeypatch, fault):
    config, data, day, work, _ = _inputs(tmp_path)
    alias = "be_nuclear_available_gw"
    known = f"known_{alias}_oracle"
    spec = config["zones"]["FR"]["covariates"][alias]
    context = data.model_context_covariates
    if fault == "series": spec["series"] = "power.be.actual.generation"
    elif fault == "disabled": spec["enabled"] = False
    elif fault == "source": spec["source"] = "csv"
    elif fault == "generation_label": spec["semantic"] = "forecast_generation"
    elif fault == "units": spec["unit"] = "MW"
    elif fault == "broadcast_flag": spec["daily_broadcast"] = False
    elif fault == "future_spec": spec["future"]["known_future"] = False
    elif fault == "missing_base": data.model_context_covariates = context.drop(columns=alias)
    elif fault == "missing_known": data.model_context_covariates = context.drop(columns=known)
    elif fault == "missing_history_alias": data.covariates = data.covariates.drop(columns=alias)
    elif fault == "history_mismatch": data.covariates.loc[data.covariates.index[-1], alias] += 1
    elif fault == "undeclared_known": data.known_future_columns.remove(known)
    elif fault == "future_nan": context.loc[context.index[-1], [alias, known]] = np.nan
    elif fault == "history_nan": context.loc[context.index[100], [alias, known]] = np.nan
    elif fault == "vintage_mismatch": context.loc[context.index[-1], alias] = 5.
    elif fault == "negative": context.loc[:, [alias, known]] = -1.
    elif fault == "mw_units": context.loc[:, [alias, known]] = 4000.
    elif fault == "intraday_change": context.loc[context.index[-1], [alias, known]] = 5.
    elif fault == "residual_selection":
        config["hourly"]["feature_engineering"]["covariate_columns"] = [base.NUCLEAR_KNOWN_COLUMN]
    elif fault == "residual_excluded":
        config["hourly"]["residual_correction"]["feature_builder"]["exclude_columns"] = [known]
    elif fault == "late_protocol": config["nuclear_experiment"]["input_protocol"] = "civil_pit_v2"
    elif fault == "incumbent_workdir": work = tmp_path / "runs/experiments/nuclear_forecast_v1/run"
    elif fault == "production_workdir": work = tmp_path / "runs/live/fr"
    elif fault == "foreign_cache":
        config["nuclear_experiment"].update(mode="incremental", incremental_cache_dir=str(tmp_path / "runs/experiments/nuclear_forecast_v1/cache"))
    def forbidden(**kwargs):
        raise AssertionError("No base replay/model should start")
    monkeypatch.setattr(base, "run_nuclear_forecast", forbidden)
    with pytest.raises(ValueError):
        cwe.run_nuclear_cwe_forecast(config=config, data=data, zone="FR", delivery_day=day, workdir=work)
    assert not work.exists()


@pytest.mark.parametrize("day, hours", [("2026-03-29", 23), ("2025-10-26", 25)])
def test_preparation_preserves_dst_and_optional_pre_anchor_gaps(tmp_path, day, hours):
    config, data, delivery, _, plans = _inputs(tmp_path, day)
    for alias in cwe.CWE_AVAILABILITY_SERIES:
        data.model_context_covariates.iloc[:48, data.model_context_covariates.columns.get_indexer(
            [alias, f"known_{alias}_oracle"])] = np.nan
    _, audit, expected = cwe._prepare_inputs(config, data, "FR", delivery)
    assert len(plans[-1].delivery_index_utc) == hours
    assert (expected.tz_convert("Europe/Paris").date == delivery).sum() == hours
    assert audit["be_nuclear_available_gw"]["missing_optional_pre_anchor_context_hours"] == 48
    assert audit["be_nuclear_available_gw"]["missing_required_hours"] == 0


def test_extra_aliases_do_not_change_existing_residual_aggregates():
    original = base.nuclear_kalman_covariate_config()
    configured = cwe.nuclear_cwe_kalman_covariate_config(original)
    assert configured.derived == original.derived
    assert configured.minimum_history_coverage == 1.
    assert configured.require_future_complete is True
    assert len(configured.groups["market"]) == len(original.groups["market"]) + 2
    assert all(alias not in original.input_columns for alias in cwe.CWE_AVAILABILITY_SERIES)


def test_protocol_fingerprints_exact_wrapper_bytes():
    import hashlib
    digest = hashlib.sha256(Path(cwe.__file__).read_bytes()).hexdigest()
    assert cwe.nuclear_cwe_code_sha256() == digest
    assert cwe.nuclear_cwe_input_protocol() == f"civil_pit_cwe_v1_{digest}"


@pytest.mark.parametrize("day, hours", [("2026-03-29", 23), ("2025-10-26", 25)])
def test_native_local_zone_data_keeps_dst_folds(tmp_path, day, hours):
    config, data, delivery, _, _ = _inputs(tmp_path, day)
    # Production prepare_zone_data returns local-zone indices. Keeping them
    # here catches an AmbiguousTimeError that UTC-only synthetic tests conceal.
    for frame in (data.target, data.covariates, data.model_context_covariates):
        frame.index = frame.index.tz_convert("Europe/Paris")
    _, audit, expected = cwe._prepare_inputs(config, data, "FR", delivery)
    assert (expected.tz_convert("Europe/Paris").date == delivery).sum() == hours
    assert all(item["missing_required_hours"] == 0 for item in audit.values())
    actual = cwe._frame(data.model_context_covariates, "native")
    assert actual.index.tz is not None and str(actual.index.tz) == "UTC"
    assert not actual.index.has_duplicates
    assert len(actual) == len(data.model_context_covariates)


def test_standard_kalman_cache_does_not_add_max_path_depth(tmp_path, monkeypatch):
    config, data, day, work, _ = _inputs(tmp_path)
    monkeypatch.setattr(base, "run_nuclear_forecast", _fake_base)
    monkeypatch.setattr(cwe, "build_operational_kalman_view", _delegate)
    result = cwe.run_nuclear_cwe_forecast(config=config, data=data, zone="FR", delivery_day=day,
        workdir=work, residual_factory=FakeResidual)
    assert result.kalman_view.captured["rolling_refit_cache_dir"] == work / "kalman_cache"


def test_fitted_residual_cannot_silently_drop_be_or_nl(tmp_path, monkeypatch):
    config, data, day, work, _ = _inputs(tmp_path)
    def dropped(**kwargs):
        result = _fake_base(**kwargs)
        result.residual_daily_audit.at[1, "residual_feature_columns"] = [base.NUCLEAR_KNOWN_COLUMN]
        return result
    monkeypatch.setattr(base, "run_nuclear_forecast", dropped)
    with pytest.raises(cwe.NuclearCWEForecastError, match="dropped"):
        cwe.run_nuclear_cwe_forecast(config=config, data=data, zone="FR", delivery_day=day,
            workdir=work, residual_factory=FakeResidual, kalman_builder=_delegate)


class FakeForecasting:
    @classmethod
    def run_backtest_variant(cls, **kwargs):
        assert kwargs["with_covariates"] is True
        data = kwargs["data"]
        parts = []
        for origin in kwargs["origins"]:
            index = data.target.index[origin:origin + kwargs["horizon"]]
            context = data.model_context_covariates.loc[index]
            middle = sum(context[f"known_{alias}_oracle"] for alias in cwe.CWE_NUCLEAR_ALIASES)
            parts.append(pd.DataFrame({"timestamp": index, "q10": middle.to_numpy() - 5,
                "q50": middle.to_numpy(), "q90": middle.to_numpy() + 5,
                "actual": data.target.loc[index].to_numpy(), "origin_index": origin}))
        return pd.concat(parts, ignore_index=True)

    @classmethod
    def run_live_forecast_variant(cls, **kwargs):
        assert kwargs["with_covariates"] is True
        data = kwargs["data"]
        index = pd.date_range(data.target.index[-1] + pd.Timedelta(hours=1), periods=kwargs["horizon"], freq="h")
        middle = sum(data.model_context_covariates.loc[index, f"known_{alias}_oracle"]
                     for alias in cwe.CWE_NUCLEAR_ALIASES)
        return pd.DataFrame({"timestamp": index, "q10": middle.to_numpy() - 5,
                             "q50": middle.to_numpy(), "q90": middle.to_numpy() + 5})


def test_replay_pipeline(tmp_path):
    config, data, day, work, _ = _inputs(tmp_path)
    result = cwe.run_nuclear_cwe_forecast(config=config, data=data, zone="FR", delivery_day=day,
        workdir=work, runtime_factory=lambda *args: object(), forecasting_module=FakeForecasting,
        residual_factory=FakeResidual, kalman_builder=_delegate, threads=1)
    forecast = result.source_forecast
    assert len(result.residual_daily_audit) == 731
    assert len(forecast) == 24
    assert forecast.chronos2__q50.iloc[0] == pytest.approx(35. + 4. + .48)
    assert forecast.residual_corrected__q50.iloc[0] == pytest.approx(35. + 4. + .48 + .4)
    assert all(alias in result.kalman_view.captured["covariates"] for alias in cwe.CWE_NUCLEAR_ALIASES)
    assert result.kalman_view.captured["training_lookback_days"] == 365
    assert result.audit["production_changed"] is False
    assert result.audit["lora_used"] is False
    assert result.audit["storm_used_as_input"] is False
    assert not (tmp_path / "runs/live").exists()
