from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nyx_cpu_live_features import FEATURE_COLUMNS
from chronos2_hourly.nyx_regional_cpu import ZONES
import run_nyx_regional_cpu as live
import run_nyx_regional_cpu_backtest as backtest


def test_weekly_selection_boundary_and_comparable_scoring():
    origins = backtest._origin_days()
    assert len(origins) == 173
    assert origins[31:34] == ["2026-04-29", "2026-05-06", "2026-05-07"]
    assert origins[-1] == "2026-09-23"
    assert all(backtest._forecast_stop_day(origin, number) == origins[number + 1]
               for number, origin in enumerate(origins[:-1]))
    assert backtest._forecast_stop_day(origins[-1], len(origins) - 1) == backtest.STOP
    index = pd.date_range("2026-05-06", periods=3, freq="h", tz="UTC")
    actual = pd.Series([0., 10., 20.], index=index)
    points = pd.Series([1., 9., 23.], index=index)
    storm = pd.Series([2., np.nan, 19.], index=index)
    score = backtest._price_score(points, actual, storm)
    assert score["n"] == 2
    assert score["strict_wins"] == 1
    assert score["strict_win_rate"] == .5
    assert score["rmse"] == np.sqrt(5.)


def test_negative_probability_scores_exactly_price_below_zero():
    index = pd.date_range("2026-05-06", periods=4, freq="h", tz="UTC")
    actual = pd.Series([-1., 0., 2., -4.], index=index)
    point = pd.Series([.8, .1, .2, .7], index=index)
    baseline = pd.Series([.5] * 4, index=index)
    score = backtest._probability_score(point, actual, baseline)
    assert score["negative_hours"] == 2
    assert score["observed_frequency"] == .5
    assert score["brier"] == np.mean([.2**2, .1**2, .2**2, .3**2])
    assert score["history_frequency_brier"] == .25


def test_negative_method_selects_lowest_selection_brier_with_fixed_ties():
    index = pd.date_range("2026-04-29", periods=4, freq="h", tz="UTC")
    actual = pd.Series([-1., 2., -3., 4.], index=index)
    frame = pd.DataFrame({"actual": actual,
                          "p_negative": [.2, .8, .2, .8],
                          "p_negative_raw": [.8, .2, .8, .2],
                          "history_frequency": [.5] * 4}, index=index)
    selected, scores = backtest._select_negative_method(frame)
    assert selected == "p_negative_raw"
    assert scores[selected]["brier"] == pytest.approx(.04)
    frame["p_negative"] = frame["p_negative_raw"]
    selected, _ = backtest._select_negative_method(frame)
    assert selected == "p_negative"
    frame["p_negative"] = .2
    frame["p_negative_raw"] = .8
    selected, _ = backtest._select_negative_method(frame)
    assert selected == "history_frequency"


def test_canonical_epex_alignment_counts_price_and_event_drift():
    index = pd.date_range("2026-05-06", periods=4, freq="h", tz="UTC")
    canonical = pd.Series([0., -1., 5., 8.], index=index)
    epex = pd.Series([0., 1., 5. + 2e-9, 7.], index=index)
    audit = backtest._target_alignment(canonical, epex, zone="FR")
    assert audit["hours"] == 4
    assert audit["exact_equal_hours"] == 1
    assert audit["different_hours_gt_1e_9_eur_mwh"] == 3
    assert audit["negative_event_disagreements"] == 1
    assert audit["max_abs_difference_eur_mwh"] == 2.
    assert audit["mean_abs_difference_eur_mwh"] == pytest.approx(.75 + 5e-10)
    incomplete = canonical.drop(index[0])
    with pytest.raises(ValueError, match="alignment has missing"):
        backtest._target_alignment(incomplete, epex, zone="FR")


def test_activation_requires_a_pinned_passed_receipt(tmp_path, monkeypatch):
    root = tmp_path
    config_dir = root / "config"
    config_dir.mkdir()
    config = config_dir / "nyx_regional_cpu.json"
    config.write_text(json.dumps({"schema_version": 1, "protocol": live.PROTOCOL,
                                  "status": "pending_backtest", "selected": {},
                                  "negative_selected": {},
                                  "backtest_receipt": "config/nyx_regional_cpu_backtest_receipt.json",
                                  "backtest_sha256": None}), encoding="utf-8")
    for name in ("run_nyx_regional_cpu_backtest.py",
                 "run_nyx_regional_cpu.py",
                 "chronos2_hourly/nyx_regional_cpu.py",
                 "chronos2_hourly/nyx_cpu_live_features.py",
                 "chronos2_hourly/nyx_regional_cpu_sources.py"):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(name, encoding="utf-8")
    source = root / "data" / "target.parquet"
    source.parent.mkdir()
    source.write_bytes(b"sealed source")
    output = root / "runs" / "evaluation"
    output.mkdir(parents=True)
    (output / "backtest_receipt.json").write_text("{}", encoding="utf-8")
    selected = {zone: "absolute" for zone in ZONES}
    negative_selected = {zone: "p_negative" for zone in ZONES}
    metrics = {zone: {"hours": 8760, "storm_common_hours": 8759,
                      "selected": "absolute",
                      "negative_selected": "p_negative",
                      "negative_selection_candidates": {
                          "p_negative": {"brier": .1},
                          "p_negative_raw": {"brier": .2},
                          "history_frequency": {"brier": .3}},
                      "confirmation": {"n": 3384, "hours": 3384,
                                       "rmse": 1., "storm_rmse": 2.,
                                       "strict_win_rate": .6},
                      "negative_confirmation": {"n": 3384, "hours": 3384,
                                                "brier": .1}}
               for zone in ZONES}
    result = {"protocol": live.PROTOCOL, "passed": True, "selected": selected,
              "negative_selected": negative_selected,
              "countries": metrics, "feature_audits": {"FR": {
                  "feature_columns": list(FEATURE_COLUMNS)}},
              "future_labels_used_for_fit": False, "Storm_used_as_model_input": False,
              "causality_passed": True}
    monkeypatch.setattr(backtest, "ROOT", root)
    monkeypatch.setattr(backtest, "CONFIG", config)
    monkeypatch.setattr(backtest, "CANONICAL_RECEIPT",
                        config_dir / "nyx_regional_cpu_backtest_receipt.json")
    backtest._activate(result, output,
                       {"source_sha256": {str(source): backtest._sha(source)}}, {})
    applied, blockers = live._validate_activation(root)
    assert applied["status"] == "validated"
    assert applied["negative_selected"] == negative_selected
    assert not blockers
    receipt = config_dir / "nyx_regional_cpu_backtest_receipt.json"
    receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
    _, blockers = live._validate_activation(root)
    assert "Backtest receipt SHA-256 mismatch" in blockers
