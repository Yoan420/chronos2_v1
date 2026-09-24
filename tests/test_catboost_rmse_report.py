from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyx_catboost_rmse.report import write_report


def _frame(index, *, actual=100.0, baseline=90.0, challenger=95.0):
    return pd.DataFrame({"actual": actual, "chronos_q50": 80.0, "mae_q50": baseline,
                         "rmse_q50": challenger, "rmse_q10": 70.0, "rmse_q90": 110.0,
                         "raw_correction": 45.0, "applied_correction": 40.0},
                        index=pd.DatetimeIndex(index, name="delivery_start_utc"))


def _run(tmp_path, frames, **metadata):
    paths = write_report(frames, metadata, tmp_path / "reports")
    return paths, json.loads(paths["metrics"].read_text(encoding="utf-8"))


def _local_day(day):
    start = pd.Timestamp(day, tz="Europe/Paris")
    return pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left").tz_convert("UTC")


@pytest.mark.parametrize("day,n_hours", [("2026-03-29", 23), ("2025-10-26", 25)])
def test_dst_days_have_physical_hours_and_can_be_complete(tmp_path, day, n_hours):
    _, payload = _run(tmp_path, {"FR": _frame(_local_day(day))}, planned_zones=["FR"],
                      evaluation_start=day, evaluation_end=day)
    assert payload["annual_complete"] is True
    assert payload["status"] == "COMPLETE"
    assert payload["scope"]["expected_hours"] == n_hours
    assert payload["annual"]["pooled"]["n_hours"] == n_hours
    assert payload["daily"][0]["n_hours"] == n_hours
    hour_two = [row for row in payload["hourly"] if row["hour"] == 2 and row["zone"] == "FR"]
    assert ([row["n_hours"] for row in hour_two] == ([2] if n_hours == 25 else []))


def test_partial_missing_zones_and_empty_input_are_prominent(tmp_path):
    paths, payload = _run(tmp_path, {"FR": _frame(_local_day("2026-09-14"))})
    assert payload["annual_complete"] is False
    assert payload["status"] == "PARTIAL"
    assert payload["scope"]["common_hours"] == 24
    markup = paths["html"].read_text(encoding="utf-8")
    assert "PARTIEL · PAS UN RÉSULTAT ANNUEL COMPLET" in markup
    assert "PAS un résultat annuel complet" in markup
    paths, empty = _run(tmp_path, {})
    assert empty["pending"] is True
    assert empty["annual"]["pooled"]["models"]["rmse_q50"]["rmse"] is None
    assert "EN ATTENTE D'OBSERVATIONS" in paths["html"].read_text(encoding="utf-8")


def test_primary_models_use_identical_population_storm_never_filled(tmp_path):
    index = pd.date_range("2026-09-14", periods=5, freq="h", tz="UTC")
    frame = _frame(index)
    frame.loc[index[0], "mae_q50"] = np.nan
    frame.loc[index[1], "rmse_q50"] = np.inf
    frame["storm_q50"] = [100.0, 100.0, 99.0, np.nan, np.inf]
    _, payload = _run(tmp_path, {"FR": frame}, planned_zones=["FR"])
    primary = payload["annual"]["pooled"]
    assert primary["n_hours"] == 3
    assert {row["n_hours"] for row in primary["models"].values()} == {3}
    assert primary["models"]["rmse_q50"]["mae"] == 5.0
    storm = payload["storm_comparison"]
    assert storm["matched_hours"] == 1
    assert storm["filled_hours"] == 0
    assert {row["n_hours"] for row in storm["pooled"]["models"].values()} == {1}
    assert storm["pooled"]["models"]["storm_q50"]["mae"] == 1.0


def test_missing_storm_is_null_not_substituted(tmp_path):
    _, payload = _run(tmp_path, {"FR": _frame(_local_day("2026-09-14"))})
    storm = payload["storm_comparison"]
    assert storm["matched_hours"] == 0
    assert storm["pooled"]["models"]["storm_q50"]["mae"] is None


def test_pooled_rmse_weights_hours_not_zone_rmse(tmp_path):
    index = pd.date_range("2026-09-14", periods=4, freq="h", tz="UTC")
    frames = {"FR": _frame(index[:1], actual=0, baseline=20, challenger=10),
              "DE": _frame(index[:3], actual=0, baseline=20, challenger=2)}
    _, payload = _run(tmp_path, frames, planned_zones=["FR", "DE"])
    score = payload["annual"]["pooled"]["models"]["rmse_q50"]
    assert score["rmse"] == pytest.approx(np.sqrt((100 + 3 * 4) / 4))
    assert score["rmse"] != pytest.approx((10 + 2) / 2)
    assert score["mae"] == 4.0
    assert score["mean_daily_mae"] == 6.0
    assert score["hour_win_rate"] == 1.0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_actual_is_rejected(tmp_path, value):
    frame = _frame(_local_day("2026-09-14"))
    frame.iloc[0, frame.columns.get_loc("actual")] = value
    with pytest.raises(ValueError, match="finite canonical observations"):
        _run(tmp_path, {"FR": frame})
    assert not (tmp_path / "reports").exists()


def test_common_population_controls_caps_slices_and_chronology(tmp_path):
    index = pd.DatetimeIndex(["2026-09-14T17:00:00Z", "2025-10-26T01:00:00Z", "2026-02-01T12:00:00Z"])
    frame = _frame(index, actual=[300.0, -100.0, 200.0], baseline=[290.0, -110.0, 190.0], challenger=[295.0, -105.0, 195.0])
    frame["raw_correction"] = [41.0, -40.0, 0.0]
    frame["applied_correction"] = [40.0, -40.0, 0.0]
    _, payload = _run(tmp_path, {"FR": frame}, planned_zones=["FR"])
    cap = payload["annual"]["pooled"]["cap_rates"]
    assert cap["raw_abs_gt_40_rate"] == pytest.approx(1 / 3)
    assert cap["applied_abs_eq_40_rate"] == pytest.approx(2 / 3)
    assert [row["month"] for row in payload["monthly"] if row["zone"] == "ALL"] == ["2025-10", "2026-02", "2026-09"]
    assert payload["slices"]["actual_ge_200"]["pooled"]["n_hours"] == 2
    assert payload["slices"]["actual_ge_300"]["pooled"]["n_hours"] == 1
    assert payload["slices"]["actual_le_minus_100"]["pooled"]["n_hours"] == 1
    assert payload["slices"]["september_14_19_local"]["pooled"]["n_hours"] == 1


def test_html_escapes_metadata_and_json_is_strict(tmp_path):
    original = _frame(_local_day("2026-09-14"))
    before = original.copy(deep=True)
    paths, payload = _run(tmp_path, {"FR": original}, note='<script>alert("bad")</script>', optional=np.nan)
    markup = paths["html"].read_text(encoding="utf-8")
    assert "<script" not in markup
    assert "&lt;script&gt;" in markup
    assert "<link " not in markup
    assert payload["metadata"]["optional"] is None
    assert "NaN" not in paths["metrics"].read_text(encoding="utf-8")
    assert "moyenne conditionnelle" in markup and "ni un P50 statistiquement calibré" in markup
    assert "sans recalibrage des intervalles de prévision" in markup
    assert "Aucun Kalman n&#x27;est recalculé" in markup
    assert '<html lang="fr">' in markup
    assert "face au CatBoost MAE archivé, pas face à Storm" in markup
    for english_label in ["Archived CatBoost", "Matched h", "Hour win", "PARTIAL ·", "Time profile", "diagnostic report"]:
        assert english_label not in markup
    pd.testing.assert_frame_equal(original, before)


def test_reject_duplicate_and_naive_indexes(tmp_path):
    index = _local_day("2026-09-14")
    with pytest.raises(ValueError, match="unique"):
        _run(tmp_path, {"FR": _frame(index.append(index[:1]))})
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        _run(tmp_path, {"FR": _frame(index.tz_localize(None))})


def test_reject_symlink_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not permitted by this Windows account")
    with pytest.raises(ValueError, match="symlinks"):
        write_report({}, {}, alias)


def test_reject_nonregular_target_before_updating_either_file(tmp_path):
    destination = tmp_path / "reports"
    destination.mkdir()
    (destination / "catboost_rmse_report.html").mkdir()
    metrics = destination / "catboost_rmse_metrics.json"
    metrics.write_text("unchanged sentinel", encoding="utf-8")
    with pytest.raises(ValueError, match="regular files"):
        write_report({}, {}, destination)
    assert metrics.read_text(encoding="utf-8") == "unchanged sentinel"


def test_missing_forecast_keeps_complete_calendar_partial(tmp_path):
    frame = _frame(_local_day("2026-09-14"))
    frame.iloc[0, frame.columns.get_loc("rmse_q50")] = np.nan
    _, payload = _run(tmp_path, {"FR": frame}, planned_zones=["FR"],
                      evaluation_start="2026-09-14", evaluation_end="2026-09-14")
    assert payload["status"] == "PARTIAL"
    assert payload["scope"]["by_zone"][0]["period_hours"] == 24
    assert payload["scope"]["by_zone"][0]["common_hours"] == 23
    assert payload["scope"]["by_zone"][0]["missing_expected_hours"] == 1


def test_nonfinite_chronos_does_not_change_primary_population(tmp_path):
    frame = _frame(_local_day("2026-09-14"))
    frame.iloc[0, frame.columns.get_loc("chronos_q50")] = np.nan
    _, payload = _run(tmp_path, {"FR": frame})
    scores = payload["annual"]["pooled"]["models"]
    assert scores["rmse_q50"]["n_hours"] == 24
    assert scores["chronos_q50"]["n_hours"] == 24
    assert scores["chronos_q50"]["available"] is False
    assert scores["chronos_q50"]["mae"] is None
