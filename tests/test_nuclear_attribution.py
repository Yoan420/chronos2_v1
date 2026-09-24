from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.nuclear_attribution as module
from chronos2_hourly.nuclear_forecast import NUCLEAR_ALIAS, NUCLEAR_KNOWN_COLUMN, QUANTILES
from chronos2_hourly.report_attribution_cache import sha256


def _inputs(delivery="2026-09-09"):
    day = pd.Timestamp(delivery)
    start = (day - pd.Timedelta(days=370)).tz_localize("Europe/Paris").tz_convert("UTC")
    origin = day.tz_localize("Europe/Paris").tz_convert("UTC")
    stop = (day + pd.Timedelta(days=1)).tz_localize("Europe/Paris").tz_convert("UTC")
    history_index = pd.date_range(start, origin, freq="h", inclusive="left", name="delivery_start_utc")
    future_index = pd.date_range(origin, stop, freq="h", inclusive="left", name="delivery_start_utc")
    history = pd.DataFrame({"q10": 30., "q50": 40., "q90": 50., "actual": 41.}, index=history_index)
    forecast = pd.DataFrame(index=future_index)
    for q, value in zip(QUANTILES, (30., 40., 50.)):
        forecast[f"chronos2__{q}"] = value
        forecast[f"residual_corrected__{q}"] = value + 1
        forecast[q] = value + 1
    target = pd.Series(41., index=history_index, name="target")
    features = pd.DataFrame({NUCLEAR_KNOWN_COLUMN: 45.}, index=history_index.append(future_index))
    data = SimpleNamespace(
        target=target, covariates=pd.DataFrame({NUCLEAR_ALIAS: 45.}, index=history_index),
        model_context_covariates=features.copy(), known_future_columns=[NUCLEAR_KNOWN_COLUMN],
    )
    result = SimpleNamespace(
        raw_history=history, source_forecast=forecast.reset_index(),
        residual_statistics=history.rename(columns={q: f"chronos2__{q}" for q in QUANTILES}).copy(),
        audit={"zone": "FR", "delivery_day": delivery},
    )
    config = {"model": {"seed": 42, "context_length": 168, "model_batch_size": 8},
              "hourly": {"residual_correction": {"enabled": True, "base_model": "chronos2"}}}

    def feature_factory(local_data, resolved):
        assert local_data is not data and resolved is not config
        return local_data.target, features.loc[history_index], features.loc[future_index], features

    return result, data, config, features, feature_factory


@pytest.fixture
def dependencies(monkeypatch):
    calls = {"fits": [], "fit_labels": [], "fit_bases": [], "runtimes": [], "writers": []}

    class Corrector:
        min_training_rows = 720

        def fit(self, X, target, base, experts):
            assert list(experts) == [f"chronos2__{q}" for q in QUANTILES]
            calls["fits"].append(X.index.copy())
            calls["fit_labels"].append(target.copy())
            calls["fit_bases"].append(base.copy())
            self.feature_columns_ = (NUCLEAR_KNOWN_COLUMN,)
            self.training_end = X.index[-1]
            return self

        def predict(self, X, base, experts):
            assert X.index[0] > self.training_end
            return base + 1.

    import run_chronos2_hourly
    monkeypatch.setattr(run_chronos2_hourly, "_residual_corrector_factory", lambda config, timezone: (Corrector, "chronos2"))
    monkeypatch.setattr(module, "set_reproducibility", lambda seed: None)

    def runtime(config, device, local_files_only):
        calls["runtimes"].append((device, local_files_only))
        return object()

    def writer(**kwargs):
        calls["writers"].append(kwargs)
        assert kwargs["include_past_prices"] is True
        assert NUCLEAR_ALIAS in kwargs["required_covariates"]
        assert kwargs["data"].target.index[-1] < kwargs["fresh_future"].index[0]
        pd.DataFrame({"weight_pct": [50.]}).to_csv(
            kwargs["output_dir"] / "variable_attribution_hourly.csv.gz", index=False, compression="gzip",
        )
        (kwargs["output_dir"] / "variable_attribution_audit.json").write_text(json.dumps({
            "status": "complete", "groups": [{"key": "historical_target_price"}],
            "forecast_sha256": sha256(kwargs["forecast_path"]),
        }), encoding="utf-8")
        kwargs["data"].target.iloc[0] = -999.

    return calls, Corrector, runtime, writer


@pytest.mark.parametrize("delivery", ["2026-09-09", "2025-03-31", "2025-10-26"])
def test_only_one_last_365day_fit_then_cached_attribution(tmp_path, dependencies, delivery):
    calls, _, runtime, writer = dependencies
    result, data, config, features, feature_factory = _inputs(delivery)
    before = copy.deepcopy((result.raw_history, result.source_forecast, data.target, features, config))
    path = module.prepare_nuclear_attribution(
        result, data, config, tmp_path, runtime_factory=runtime,
        feature_factory=feature_factory, attribution_writer=writer,
    )
    assert len(calls["fits"]) == len(calls["runtimes"]) == len(calls["writers"]) == 1
    fitted = calls["fits"][0].tz_convert("Europe/Paris")
    assert len(pd.Index(fitted.date).unique()) == 365
    assert fitted[0].date() == (pd.Timestamp(delivery) - pd.Timedelta(days=365)).date()
    assert fitted[-1].date() == (pd.Timestamp(delivery) - pd.Timedelta(days=1)).date()
    assert calls["runtimes"] == [("auto", True)]
    assert (path / "forecast_hourly_fr.csv").is_file()
    assert not (path / "artifact_checksums.json").exists()
    audit = json.loads((path / "variable_attribution_audit.json").read_text())
    assert audit["nuclear_report_preparation"]["residual_refits"] == 1
    again = module.prepare_nuclear_attribution(
        result, data, config, tmp_path, runtime_factory=runtime,
        feature_factory=feature_factory, attribution_writer=writer,
    )
    assert again == path and len(calls["fits"]) == 1
    pd.testing.assert_frame_equal(result.raw_history, before[0])
    pd.testing.assert_frame_equal(result.source_forecast, before[1])
    pd.testing.assert_series_equal(data.target, before[2])
    pd.testing.assert_frame_equal(features, before[3])
    assert config == before[4]


def test_changed_configuration_gets_new_report_cache(tmp_path, dependencies):
    calls, _, runtime, writer = dependencies
    result, data, config, _, feature_factory = _inputs()
    kwargs = dict(runtime_factory=runtime, feature_factory=feature_factory, attribution_writer=writer)
    first = module.prepare_nuclear_attribution(result, data, config, tmp_path, **kwargs)
    config["model"]["seed"] += 1
    second = module.prepare_nuclear_attribution(result, data, config, tmp_path, **kwargs)
    assert first != second and len(calls["fits"]) == 2


@pytest.mark.parametrize("damage", ["missing", "changed"])
@pytest.mark.parametrize("legacy_seal", [False, True])
def test_forecast_copy_is_checked_with_legacy_and_complete_seals(tmp_path, dependencies, damage, legacy_seal):
    calls, _, runtime, writer = dependencies
    result, data, config, _, feature_factory = _inputs()
    kwargs = dict(runtime_factory=runtime, feature_factory=feature_factory, attribution_writer=writer)
    path = module.prepare_nuclear_attribution(result, data, config, tmp_path, **kwargs)
    forecast = path / "forecast_hourly_fr.csv"
    if legacy_seal:
        # Older publications sealed only the attribution CSV and audit. Their
        # forecast companion must still be checked by the nuclear adapter.
        manifest = path / "report_cache_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["artifacts"].pop(forecast.name)
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    if damage == "missing":
        forecast.unlink()
    else:
        forecast.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="forecast copy" if legacy_seal else "divergent"):
        module.prepare_nuclear_attribution(result, data, config, tmp_path, **kwargs)
    assert len(calls["fits"]) == 1


def test_nonreproducing_fit_publishes_no_attribution_and_loads_no_runtime(tmp_path, dependencies):
    calls, Corrector, runtime, writer = dependencies
    result, data, config, _, feature_factory = _inputs()

    class Wrong(Corrector):
        def predict(self, X, base, experts):
            return base + 2.

    with pytest.raises(ValueError, match="does not reproduce"):
        module.prepare_nuclear_attribution(
            result, data, config, tmp_path, runtime_factory=runtime, residual_factory=Wrong,
            feature_factory=feature_factory, attribution_writer=writer,
        )
    assert not calls["runtimes"] and not calls["writers"]
    assert not list(tmp_path.rglob("variable_attribution_audit.json"))


def test_writer_reproduction_failure_publishes_no_cache(tmp_path, dependencies):
    _, _, runtime, _ = dependencies
    result, data, config, _, feature_factory = _inputs()

    def failed(**kwargs):
        raise ValueError("scenario does not reproduce official forecast")

    with pytest.raises(ValueError, match="scenario does not reproduce"):
        module.prepare_nuclear_attribution(
            result, data, config, tmp_path, runtime_factory=runtime,
            feature_factory=feature_factory, attribution_writer=failed,
        )
    assert not list(tmp_path.rglob("report_cache_manifest.json"))


@pytest.mark.parametrize("damage", ["missing_history", "changed_labels", "missing_nuclear", "future_target"])
def test_causal_input_failures_happen_before_fit(tmp_path, dependencies, damage):
    calls, _, runtime, writer = dependencies
    result, data, config, features, feature_factory = _inputs()
    if damage == "missing_history":
        result.raw_history = result.raw_history.drop(result.raw_history.index[-100])
    elif damage == "changed_labels":
        result.raw_history.loc[result.raw_history.index[-100], "actual"] = 999
    elif damage == "missing_nuclear":
        features.iloc[-100, 0] = np.nan
    elif damage == "future_target":
        data.target.loc[features.index[-1]] = 999
    with pytest.raises(ValueError):
        module.prepare_nuclear_attribution(
            result, data, config, tmp_path, runtime_factory=runtime,
            feature_factory=feature_factory, attribution_writer=writer,
        )
    assert not calls["fits"] and not calls["runtimes"]


def test_frozen_recipe_cannot_be_changed_for_the_explanation(tmp_path, dependencies):
    calls, _, runtime, writer = dependencies
    result, data, config, _, feature_factory = _inputs()
    result.audit["residual_recipe"] = {"enabled": True, "base_model": "chronos2", "thread_count": 4, "iterations": 700}
    with pytest.raises(ValueError, match="recipe differs"):
        module.prepare_nuclear_attribution(
            result, data, config, tmp_path, runtime_factory=runtime,
            feature_factory=feature_factory, attribution_writer=writer,
        )
    assert not calls["fits"]


def test_attribution_never_writes_under_live_archives(tmp_path, dependencies):
    _, _, runtime, writer = dependencies
    result, data, config, _, feature_factory = _inputs()
    protected = Path(module.__file__).resolve().parents[1] / "runs/live/not-created-nuclear-test"
    with pytest.raises(ValueError, match="outside live archives"):
        module.prepare_nuclear_attribution(
            result, data, config, protected, runtime_factory=runtime,
            feature_factory=feature_factory, attribution_writer=writer,
        )
    assert not protected.exists()


def test_csv_roundtrip_reuses_exact_frozen_float32_labels_and_raw_quantiles(tmp_path, dependencies):
    calls, _, runtime, writer = dependencies
    result, data, config, _, feature_factory = _inputs()
    # Real run shape: CSV restores short decimals in float64; the original
    # residual Parquet preserves their underlying float32 values precisely.
    result.raw_history["actual"] = 338.8625
    result.residual_statistics["actual"] = np.float32(338.8625)
    data.target[:] = float(np.float32(338.8625))
    for q, value in zip(QUANTILES, (29.54321, 39.65432, 49.76543)):
        result.raw_history[q] = value
        result.residual_statistics[f"chronos2__{q}"] = np.float32(value)
    frozen_before = result.residual_statistics.copy(deep=True)
    path = module.prepare_nuclear_attribution(result, data, config, tmp_path, runtime_factory=runtime,
                                             feature_factory=feature_factory, attribution_writer=writer)
    labels = calls["fit_labels"][0]
    assert labels.dtype == np.dtype("float32")
    assert (labels.to_numpy() == np.float32(338.8625)).all()
    for q in QUANTILES:
        assert calls["fit_bases"][0][q].dtype == np.dtype("float32")
        assert np.array_equal(calls["fit_bases"][0][q].to_numpy(), result.residual_statistics.loc[labels.index, f"chronos2__{q}"].to_numpy())
    pd.testing.assert_frame_equal(result.residual_statistics, frozen_before)
    audit = json.loads((path / "variable_attribution_audit.json").read_text())["nuclear_report_preparation"]
    assert audit["training_actuals_source"] == "residual_statistics_parquet_original_fit_labels"
    assert audit["observation_precision"]["checkpoint_labels_vs_frozen_labels"]["mode"] == "float32_roundtrip"
    assert audit["training_raw_quantiles_source"] == "frozen_residual_cache_chronos2_quantiles"
    assert all(item["mode"] == "float32_roundtrip" for item in audit["training_raw_quantile_precision"].values())


@pytest.mark.parametrize("column", ["actual", "chronos2__q50"])
def test_real_cached_training_value_change_is_rejected_before_fit(tmp_path, dependencies, column):
    calls, _, runtime, writer = dependencies
    result, data, config, _, feature_factory = _inputs()
    result.residual_statistics.loc[result.residual_statistics.index[-100], column] += 0.01
    with pytest.raises(ValueError, match="divergentes"):
        module.prepare_nuclear_attribution(result, data, config, tmp_path, runtime_factory=runtime,
                                          feature_factory=feature_factory, attribution_writer=writer)
    assert not calls["fits"] and not calls["runtimes"]
