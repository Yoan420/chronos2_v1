from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.models.residual_corrector import ResidualCorrector
import nyx_catboost_rmse.core as core


def _config():
    recipe = dict(core.FIXED_RECIPE)
    recipe.update(enabled=True, base_model="chronos2", thread_count=4, verbose=False,
                  feature_builder={"timezone": "Europe/Paris", "include_calendar": True,
                                   "include_daily_profiles": True,
                                   "exclude_historical_prices": True,
                                   "exclude_day_of_year": True})
    return {"hourly": {"residual_correction": recipe}}


def _inputs(day="2026-09-19"):
    target = date.fromisoformat(day)
    index = core._civil_index(target - timedelta(days=366), target + timedelta(days=3), "Europe/Paris")
    local_dates = pd.Series(index.tz_convert("Europe/Paris").date, index=index)
    origins = {d: (pd.Timestamp(d) - pd.Timedelta(days=1) + pd.Timedelta(hours=8))
               .tz_localize("Europe/Paris").tz_convert("UTC") for d in local_dates.unique()}
    raw = pd.DataFrame({"q10": 20., "q50": 30., "q90": 50., "actual": 45.,
                        "forecast_origin_utc": local_dates.map(origins)}, index=index)
    future = raw.loc[local_dates.eq(target)].drop(columns="actual").copy()
    features = pd.DataFrame({"known_fr_nuclear_generation_fcst_gw_oracle": 40.,
                             "price_lag_24": np.arange(len(index), dtype=float)}, index=index)
    return raw, future, features, target


class FakeCorrector(ResidualCorrector):
    """Exercise the causal boundary without any expensive CatBoost fits."""
    def __init__(self):
        super().__init__(**core.FIXED_RECIPE)
        self.model_ = SimpleNamespace(get_params=lambda: {"loss_function": "RMSE"})

    def fit(self, X, y, base, experts):
        assert X.index.equals(y.index) and X.index.equals(base.index)
        assert tuple(experts) == ("chronos2__q10", "chronos2__q50", "chronos2__q90")
        np.testing.assert_array_equal(experts.to_numpy(), base.to_numpy())
        self.training_index = X.index.copy()
        self.shift = float((y - base.q50).mean())
        self.feature_columns_ = ("known_fr_nuclear_generation_fcst_gw_oracle",)
        return self

    def predict_correction(self, X, base, experts):
        assert self.training_index[-1] < X.index[0]
        result = pd.Series(np.clip(self.shift, -40, 40), index=X.index)
        result.attrs["raw_correction"] = pd.Series(self.shift, index=X.index)
        return result

    def predict(self, X, base, experts):
        result = base.add(self.predict_correction(X, base, experts), axis=0)
        result.attrs["nonserializable_inherited_attr"] = pd.Series([1])
        return result


@pytest.fixture
def fake_factory(monkeypatch):
    made = []

    def factory(*args, **kwargs):
        result = FakeCorrector()
        made.append(result)
        return result

    monkeypatch.setattr(core, "make_corrector", factory)
    return made


def _fit(inputs, **kwargs):
    history, future, features, day = inputs
    return core.fit_day(history=history, future=future, features=features,
                        timezone="Europe/Paris", day=day, config=_config(), **kwargs)


def test_factory_changes_only_explicit_loss_metric_and_preserves_builder():
    pytest.importorskip("catboost")
    from run_chronos2_hourly import _residual_corrector_factory

    config = _config()
    before = deepcopy(config)
    factory, _ = _residual_corrector_factory(config, timezone="Europe/Paris")
    existing = factory()
    existing.backend_ = "catboost"
    original = existing._new_model().get_params()
    models = {loss: core.make_corrector(config, "Europe/Paris", 4, loss) for loss in ("MAE", "RMSE")}
    for loss, corrector in models.items():
        corrector.backend_ = "catboost"
        params = corrector._new_model().get_params()
        assert params == {**original, "loss_function": loss, "eval_metric": loss}
        assert corrector.feature_builder_options == existing.feature_builder_options
        assert corrector.feature_builder_options is not existing.feature_builder_options
        assert core._constructor_kwargs(corrector) == core._constructor_kwargs(existing)
    models["RMSE"].feature_builder_options["include_calendar"] = False
    assert config == before
    assert models["MAE"].feature_builder_options["include_calendar"] is True


@pytest.mark.parametrize("key,value", [("backend", "auto"), ("max_abs_correction", 30.),
                                      ("correction_scale", .5), ("iterations", 500),
                                      ("learning_rate", .1), ("correction_clip", 40.)])
def test_factory_rejects_recipe_drift(key, value):
    config = _config()
    config["hourly"]["residual_correction"][key] = value
    with pytest.raises(core.RMSEExperimentError, match="recipe mismatch"):
        core.make_corrector(config, "Europe/Paris", 2)


def test_factory_fails_closed_without_catboost(monkeypatch):
    import chronos2_hourly.models.residual_corrector as residual
    from chronos2_hourly.models.base import OptionalDependencyError

    monkeypatch.setattr(residual, "catboost_available", lambda: False)
    with pytest.raises(OptionalDependencyError):
        core.make_corrector(_config(), "Europe/Paris", 2)


def test_tiny_real_catboost_fit_keeps_inherited_clipping_and_features():
    pytest.importorskip("catboost")
    corrector = core.make_corrector(_config(), "Europe/Paris", 1)
    # Unit-test-only local instance: two trees on 48 synthetic rows, never an
    # annual experiment. Factory recipe validation above still uses 700 trees.
    corrector.iterations = 2
    corrector.min_training_rows = 2
    index = pd.date_range("2026-01-01", periods=72, freq="h", tz="UTC")
    X = pd.DataFrame({"known_fr_nuclear_generation_fcst_gw_oracle": 40 + np.sin(np.arange(72)),
                      "price_lag_24": np.arange(72, dtype=float)}, index=index)
    base = pd.DataFrame({"q10": 20., "q50": 30., "q90": 50.}, index=index)
    experts = base.rename(columns={q: f"chronos2__{q}" for q in core.QUANTILES})
    y = pd.Series(60 + np.cos(np.arange(48)), index=index[:48])
    corrector.fit(X.iloc[:48], y, base.iloc[:48], experts.iloc[:48])
    predicted = corrector.predict(X.iloc[48:], base.iloc[48:], experts.iloc[48:])
    shift = corrector.predict_correction(X.iloc[48:], base.iloc[48:], experts.iloc[48:])
    assert not any("price_lag" in column for column in corrector.feature_columns_)
    assert corrector.model_.get_params()["loss_function"] == "RMSE"
    np.testing.assert_allclose(shift, np.clip(shift.attrs["raw_correction"], -40, 40))
    np.testing.assert_allclose(predicted.q90 - predicted.q10, base.q90.iloc[48:] - base.q10.iloc[48:])


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25), ("2026-09-19", 24)])
def test_fit_day_exact_civil_window_and_dst(day, hours, fake_factory):
    inputs = _inputs(day)
    raw_before = inputs[0].copy(deep=True)
    predicted, audit = _fit(inputs, expected_features=["known_fr_nuclear_generation_fcst_gw_oracle"])
    target = inputs[-1]
    expected = core._civil_index(target - timedelta(days=365), target, "Europe/Paris")
    assert fake_factory[0].training_index.equals(expected)
    assert len(predicted) == hours
    assert audit["training_days"] == 365
    assert audit["training_rows"] == len(expected)
    assert audit["training_start_day"] == str(target - timedelta(days=365))
    assert audit["training_end_day"] == str(target - timedelta(days=1))
    assert audit["current_day_labels_used"] is False
    assert tuple(predicted) == (*core.QUANTILES, "raw_correction", "applied_correction", "forecast_origin_utc")
    assert not predicted.attrs
    json.dumps(audit, allow_nan=False)
    pd.testing.assert_frame_equal(inputs[0], raw_before)


def test_prediction_never_reads_current_or_later_history_labels(fake_factory):
    inputs = _inputs()
    original, _ = _fit(inputs)
    local = inputs[0].index.tz_convert("Europe/Paris").date
    inputs[0].loc[local >= inputs[-1], "actual"] = np.nan
    changed, _ = _fit(inputs)
    pd.testing.assert_frame_equal(original, changed)
    # Conversely a previous-day label belongs to training and may change D.
    inputs[0].loc[local == inputs[-1] - timedelta(days=1), "actual"] = 5000
    causal_change, _ = _fit(inputs)
    assert not original.q50.equals(causal_change.q50)


def test_preclip_diagnostics_are_not_destroyed(fake_factory):
    inputs = _inputs()
    inputs[0]["actual"] = 130.
    predicted, audit = _fit(inputs)
    assert predicted.raw_correction.eq(100).all()
    assert predicted.applied_correction.eq(40).all()
    assert predicted.q50.eq(70).all()
    assert audit["clipped_hours"] == 24


@pytest.mark.parametrize("fault", ["missing_train_hour", "missing_pred_hour", "future_actual", "nan_label",
                                  "crossed_quantile", "wrong_origin", "missing_feature", "duplicate", "timezone"])
def test_malformed_inputs_are_rejected_before_fit(fault, fake_factory):
    history, future, features, day = _inputs()
    if fault == "missing_train_hour":
        history = history.drop(history.index[30])
    elif fault == "missing_pred_hour":
        future = future.iloc[:-1]
    elif fault == "future_actual":
        future["actual"] = 999.
    elif fault == "nan_label":
        history.loc[history.index[30], "actual"] = np.nan
    elif fault == "crossed_quantile":
        future["q10"] = 999.
    elif fault == "wrong_origin":
        future["forecast_origin_utc"] += pd.Timedelta(hours=1)
    elif fault == "missing_feature":
        features = features.drop(future.index[-1])
    elif fault == "duplicate":
        history = pd.concat([history.iloc[:1], history])
    elif fault == "timezone":
        history.index = history.index.tz_convert("Europe/Paris")
    with pytest.raises(ValueError):
        _fit((history, future, features, day))
    assert not fake_factory


def test_schema_difference_is_rejected(fake_factory):
    with pytest.raises(core.RMSEExperimentError, match="schema differs"):
        _fit(_inputs(), expected_features=["different_column"])
