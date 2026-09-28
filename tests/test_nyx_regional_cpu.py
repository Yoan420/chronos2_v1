from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_regional_cpu as cpu
import run_nyx_regional_cpu as runner
from run_nyx_regional_cpu import _validate_activation


def _inputs(zone="FR", day="2026-09-10"):
    origin = pd.Timestamp(day)
    first = origin - pd.Timedelta(days=365)
    source = cpu.grid(first - pd.Timedelta(days=7), origin, zone)
    prices = pd.Series(20. + np.sin(np.arange(len(source)) / 11.),
                       index=source, name="price")
    train = cpu.grid(first, origin, zone)
    future = cpu.grid(origin, origin + pd.Timedelta(days=1), zone)
    features = pd.DataFrame({"known_load": np.sin(np.arange(len(train) + len(future)) / 30.),
                             "known_load__available": 1.}, index=train.append(future))
    return features, prices.reindex(train), prices


def test_inputs_forbid_future_labels_and_storm():
    features, actual, _ = _inputs()
    cpu.validate_inputs(features, actual, zone="FR", delivery_day="2026-09-10")
    with pytest.raises(ValueError, match="Actual prices must be exactly"):
        cpu.validate_inputs(features, actual.reindex(features.index),
                            zone="FR", delivery_day="2026-09-10")
    with pytest.raises(ValueError, match="Storm"):
        cpu.validate_inputs(features.assign(storm_prediction=0.), actual,
                            zone="FR", delivery_day="2026-09-10")


def test_prior_day_mean_requires_complete_dst_day():
    source = cpu.grid("2026-10-24", "2026-10-27", "FR")
    prices = pd.Series(np.arange(len(source), dtype=float), index=source)
    target = cpu.grid("2026-10-26", "2026-10-27", "FR")
    result = cpu.prior_day_mean(prices, target, zone="FR")
    previous = prices.loc[cpu.grid("2026-10-25", "2026-10-26", "FR")]
    assert len(previous) == 25
    assert result.eq(previous.mean()).all()
    with pytest.raises(ValueError, match="complete past days"):
        cpu.prior_day_mean(prices.drop(previous.index[0]), target, zone="FR")


def test_price_candidates_have_fixed_blend_and_cpu_parameters(monkeypatch):
    class FakeRegressor:
        def __init__(self, **params):
            assert params["task_type"] == "CPU"
            self.params = params
            self.tree_count_ = params["iterations"]

        def fit(self, features, labels):
            self.value = float(np.mean(labels))
            return self

        def predict(self, features):
            return np.full(len(features), self.value)

    monkeypatch.setattr(cpu, "CatBoostRegressor", FakeRegressor)
    features, actual, prices = _inputs()
    result = cpu.fit_price_candidates(features, actual, prices,
                                      zone="FR", delivery_day="2026-09-10")
    points = result.predictions
    np.testing.assert_array_equal(points.blend50.to_numpy(),
                                  ((points.absolute + points.residual_prior_day_mean) / 2.).to_numpy())
    assert set(result.models) == {"absolute", "residual_prior_day_mean"}
    assert result.audit["current_or_future_actual_used"] is False
    assert result.audit["parameters"]["task_type"] == "CPU"


def test_negative_fallback_uses_only_past_frequency():
    features, actual, _ = _inputs()
    actual[:] = 10.
    result = cpu.fit_negative_price(features, actual, zone="FR", delivery_day="2026-09-10")
    assert result.model is None
    assert result.predictions.p_negative.eq(0.).all()
    assert result.audit["fallback_reason"] == "one_class"


def test_live_probability_uses_sealed_method_and_prior_prices():
    index = pd.date_range("2026-09-10", periods=2, freq="h", tz="UTC")
    points = pd.DataFrame({"p_negative": [.1, .9],
                           "p_negative_raw": [.4, .6]}, index=index)
    history = pd.Series([-1., 1., -2., 3.])
    np.testing.assert_array_equal(runner._operational_negative_probability(
        points, history, "p_negative").to_numpy(), [.1, .9])
    np.testing.assert_array_equal(runner._operational_negative_probability(
        points, history, "p_negative_raw").to_numpy(), [.4, .6])
    np.testing.assert_array_equal(runner._operational_negative_probability(
        points, history, "history_frequency").to_numpy(), [.5, .5])
    with pytest.raises(ValueError, match="Complete historical prices"):
        runner._operational_negative_probability(points, pd.concat([history, pd.Series([np.nan])]),
                                                 "history_frequency")


def test_activation_requires_pinned_backtest_receipt(tmp_path):
    (tmp_path / "config").mkdir()
    selected = {zone: "absolute" for zone in cpu.ZONES}
    negative_selected = {zone: "p_negative_raw" if zone == "FR" else "p_negative"
                         for zone in cpu.ZONES}
    code = {}
    for relative in ("run_nyx_regional_cpu_backtest.py",
                     "run_nyx_regional_cpu.py",
                     "chronos2_hourly/nyx_regional_cpu.py",
                     "chronos2_hourly/nyx_cpu_live_features.py",
                     "chronos2_hourly/nyx_regional_cpu_sources.py"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative)
        code[relative] = hashlib.sha256(relative.encode()).hexdigest()
    receipt = {"protocol": cpu.PROTOCOL, "passed": True, "selected": selected,
               "negative_selected": negative_selected,
               "causality_passed": True,
               "code_sha256": code,
               "country_metrics": {zone: {"selected": "absolute", "hours": 8760,
                    "storm_common_hours": 8759, "confirmation": {"hours": 3000},
                    "negative_selected": negative_selected[zone],
                    "negative_selection_candidates": {
                        "p_negative": {"brier": .2 if zone == "FR" else .1},
                        "p_negative_raw": {"brier": .1 if zone == "FR" else .2},
                        "history_frequency": {"brier": .3}},
                    "negative_confirmation": {"hours": 3000}} for zone in cpu.ZONES},
               "feature_columns": list(__import__(
                   "chronos2_hourly.nyx_cpu_live_features", fromlist=["FEATURE_COLUMNS"]
               ).FEATURE_COLUMNS), "countries": list(cpu.ZONES)}
    raw = json.dumps(receipt).encode()
    path = tmp_path / "config" / "receipt.json"
    path.write_bytes(raw)
    config = {"schema_version": 1, "protocol": cpu.PROTOCOL, "status": "validated",
              "selected": selected, "negative_selected": negative_selected,
              "backtest_receipt": "config/receipt.json",
              "backtest_sha256": hashlib.sha256(raw).hexdigest()}
    (tmp_path / "config" / "nyx_regional_cpu.json").write_text(json.dumps(config))
    _, blockers = _validate_activation(tmp_path)
    assert blockers == []
    (tmp_path / "run_nyx_regional_cpu.py").write_text("changed")
    _, blockers = _validate_activation(tmp_path)
    assert any("code SHA-256 mismatch" in message for message in blockers)
    (tmp_path / "run_nyx_regional_cpu.py").write_text("run_nyx_regional_cpu.py")
    path.write_bytes(raw + b" ")
    _, blockers = _validate_activation(tmp_path)
    assert any("SHA-256 mismatch" in message for message in blockers)


def test_preflight_reports_missing_saturn_client(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_validate_activation", lambda: (
        {"status": "validated", "selected": {zone: "absolute" for zone in cpu.ZONES},
         "negative_selected": {zone: "p_negative" for zone in cpu.ZONES}}, []))
    monkeypatch.setattr(runner, "preflight_sources", lambda root: {"ready": True})
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: None)
    result = runner.plan(delivery_day="2026-09-10", countries=("FR",),
                         output=tmp_path / "new_output", inspect_sources=True)
    assert result["ready"] is False
    assert any("tshistory_lite" in blocker for blocker in result["blockers"])
