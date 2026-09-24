"""Synthetic report-only checks: no training, source loading or production writes."""

from copy import deepcopy
import json
import re

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_live_reporting as report


def _frame(index, zone, *, history=False):
    frame = pd.DataFrame(index=index)
    frame["zone"] = zone
    point = 180. + np.arange(len(index)) % 24
    for model, shift in (("nyx", 0.), ("test2", 130.)):
        for quantile, offset in (("q10", -25.), ("q50", 0.), ("q90", 90.)):
            frame[f"{model}__{quantile}"] = point + shift + offset
    frame["selected_test2"] = np.arange(len(index)) % 5 == 0
    frame["selected_model"] = np.where(frame.selected_test2, "Test2", "NYX")
    for q in report.QUANTILES:
        frame[f"hybrid__{q}"] = np.where(frame.selected_test2, frame[f"test2__{q}"], frame[f"nyx__{q}"])
    frame["own_joint_deficit"] = .6
    frame["own_residual_stress"] = 1.2
    frame["nyx_daily_peak_gap"] = 23. - np.arange(len(index)) % 24
    frame["reason"] = np.where(frame.selected_test2, "selected safe rule", "NYX fallback")
    if history:
        frame["actual"] = point + 30.
    return frame


def _inputs(day="2026-09-23", zone="DE", history_days=3):
    tz = report.TIMEZONES[zone]
    first = pd.Timestamp(day) - pd.Timedelta(days=history_days)
    index = pd.date_range(first.tz_localize(tz), pd.Timestamp(day).tz_localize(tz),
                          freq="h", inclusive="left").tz_convert("UTC")
    return _frame(index, zone, history=True), _frame(report._day_index(day, tz), zone)


def _prepare(tmp_path, history, forecast, *, day="2026-09-23", zone="DE"):
    return report._prepare(history, forecast, zone=zone, delivery_day=day,
                           output_path=tmp_path / "report.html",
                           metadata={"protocol": "synthetic", "warmup_days": 90})


def test_standard_html_preserves_scope_and_is_not_a_corrector_report(tmp_path):
    history, forecast = _inputs()
    hist_before, future_before = history.copy(deep=True), forecast.copy(deep=True)
    metadata = {"protocol": "synthetic-v1", "asof": "2026-09-22T06:00:00Z", "warmup_days": 90,
                "pair": "DE/NL", "reference": "<script>alert('unsafe')</script>"}
    meta_before = deepcopy(metadata)
    path = report.render_live_report(history, forecast, zone="DE", delivery_day="2026-09-23",
                                    output_path=tmp_path / "new" / "report.html", metadata=metadata)
    document = path.read_text(encoding="utf-8")
    for signature in ('data-report-section="statistics"', 'id="theme-toggle"',
                      'data-report-section="nyx-routing-decisions"',
                      'data-report-section="nyx-live-protocol"', "plotly", "NYX de référence"):
        assert signature in document
    assert "2026-09-20" in document and "2026-09-22" in document
    assert "3 jours civils" in document
    assert "Prévision opérationnelle du" not in document
    assert "365 derniers jours" not in document
    assert 'data-report-section="forecast-components"' not in document
    assert 'data-report-section="kalman-diagnostics"' not in document
    assert "Attribution du prix final indisponible" in document
    assert "La référence utilise uniquement le contexte historique" not in document
    assert "<script>alert('unsafe')</script>" not in document
    payload = re.search(r'id="nyx-live-report-audit">(.*?)</script>', document, re.S).group(1)
    audit = json.loads(payload)
    assert audit["history_rows"] == 72
    assert audit["paired_observed_rows"] == 72
    assert audit["forecast_hours"] == 24
    assert audit["future_actual_included"] is False
    assert audit["live_excluded_from_statistics"] is True
    assert audit["selected_test2_hours"] == 5
    assert audit["metadata"] == metadata
    pd.testing.assert_frame_equal(history, hist_before)
    pd.testing.assert_frame_equal(forecast, future_before)
    assert metadata == meta_before
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("zone", ["DE", "NL", "BE", "FR"])
@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25), ("2026-09-23", 24)])
def test_exact_local_delivery_grids_include_dst(tmp_path, zone, day, hours):
    history, forecast = _inputs(day, zone)
    result, issued, audit = _prepare(tmp_path, history, forecast, day=day, zone=zone)
    assert len(issued) == hours == audit["forecast_hours"]
    assert set(result.forecast_native.timestamp.dt.date) == {pd.Timestamp(day).date()}
    assert result.zone_data.timezone == report.TIMEZONES[zone]


@pytest.mark.parametrize("day,hours", [("2026-03-30", 71), ("2026-10-26", 73)])
def test_historical_dst_grid_keeps_all_physical_hours(tmp_path, day, hours):
    history, forecast = _inputs(day)
    result, _, audit = _prepare(tmp_path, history, forecast, day=day)
    assert len(result.backtest_native) == hours == audit["paired_observed_rows"]


def test_quantiles_are_exact_copies_without_cap_or_forced_increase(tmp_path):
    history, forecast = _inputs()
    forecast.loc[forecast.index[0], [f"test2__{q}" for q in report.QUANTILES]] = [1000., 1200., 2000.]
    forecast.loc[forecast.index[5], [f"test2__{q}" for q in report.QUANTILES]] = [-200., -100., 0.]
    for q in report.QUANTILES:
        forecast[f"hybrid__{q}"] = np.where(forecast.selected_test2, forecast[f"test2__{q}"], forecast[f"nyx__{q}"])
    result, _, _ = _prepare(tmp_path, history, forecast)
    for q in report.QUANTILES:
        np.testing.assert_array_equal(result.forecast_native[q], forecast[f"hybrid__{q}"])
        np.testing.assert_array_equal(result.forecast_baseline[q], forecast[f"nyx__{q}"])
    assert result.forecast_native.q50.iloc[0] == 1200.
    assert result.forecast_native.q50.iloc[5] == -100.


def test_current_actuals_are_rejected_before_any_file_is_written(tmp_path):
    history, forecast = _inputs()
    forecast["actual"] = np.nan
    forecast.loc[forecast.index[-1], "actual"] = 9999.
    with pytest.raises(ValueError, match="Future actual"):
        report.render_live_report(history, forecast, zone="DE", delivery_day="2026-09-23",
            output_path=tmp_path / "absent" / "report.html", metadata={})
    assert not (tmp_path / "absent").exists()


def test_null_current_actuals_and_missing_historical_actuals_do_not_leak(tmp_path):
    history, forecast = _inputs()
    forecast["actual"] = np.nan
    history.loc[history.index[0], "actual"] = np.nan
    result, _, audit = _prepare(tmp_path, history, forecast)
    assert audit["paired_observed_rows"] == 71
    assert len(result.backtest_native) == len(result.backtest_baseline) == 71
    assert len(result.statistics_candidate) == 72
    assert result.zone_data.target.index.max() < forecast.index.min()
    assert "actual" not in result.forecast_native
    assert result.backtest_native.origin_timestamp.isna().all()


@pytest.mark.parametrize("case", ["missing_hour", "duplicate", "unordered", "wrong_day", "naive",
                                  "wrong_zone", "missing_quantile", "nan_quantile", "crossed_quantile",
                                  "wrong_route", "integer_route", "missing_route", "mismatched_label"])
def test_invalid_forecast_fails_closed(tmp_path, case):
    history, forecast = _inputs()
    if case == "missing_hour":
        forecast = forecast.iloc[:-1]
    elif case == "duplicate":
        forecast = pd.concat([forecast, forecast.iloc[-1:]])
    elif case == "unordered":
        forecast = forecast.iloc[::-1]
    elif case == "wrong_day":
        forecast.index = forecast.index + pd.Timedelta(days=1)
    elif case == "naive":
        forecast.index = forecast.index.tz_localize(None)
    elif case == "wrong_zone":
        forecast["zone"] = "NL"
    elif case == "missing_quantile":
        forecast = forecast.drop(columns="test2__q10")
    elif case == "nan_quantile":
        forecast.iloc[0, forecast.columns.get_loc("test2__q10")] = np.nan
    elif case == "crossed_quantile":
        forecast.iloc[0, forecast.columns.get_loc("test2__q10")] = 10000.
    elif case == "wrong_route":
        forecast.iloc[0, forecast.columns.get_loc("hybrid__q50")] += .000001
    elif case == "integer_route":
        forecast["selected_test2"] = forecast.selected_test2.astype(int)
    elif case == "missing_route":
        forecast = forecast.drop(columns="selected_test2")
    else:
        forecast["selected_model"] = "NYX"
    with pytest.raises((ValueError, KeyError)):
        _prepare(tmp_path, history, forecast)


@pytest.mark.parametrize("case", ["future", "hole", "empty", "no_actual", "no_finite_actual", "infinite_actual"])
def test_invalid_history_is_rejected(tmp_path, case):
    history, forecast = _inputs()
    if case == "future":
        history.index = history.index + pd.Timedelta(days=3)
    elif case == "hole":
        history = history.drop(history.index[10])
    elif case == "empty":
        history = history.iloc[:0]
    elif case == "no_actual":
        history = history.drop(columns="actual")
    elif case == "no_finite_actual":
        history["actual"] = np.nan
    else:
        history.iloc[0, history.columns.get_loc("actual")] = np.inf
    with pytest.raises(ValueError):
        _prepare(tmp_path, history, forecast)


def test_provided_origins_are_preserved_and_future_origins_rejected(tmp_path):
    history, forecast = _inputs()
    history["forecast_origin_utc"] = history.index - pd.Timedelta(days=1)
    result, _, _ = _prepare(tmp_path, history, forecast)
    np.testing.assert_array_equal(result.backtest_native.origin_timestamp.dt.tz_convert("UTC"),
                                  history.forecast_origin_utc)
    history["forecast_origin_utc"] = history.index
    with pytest.raises(ValueError, match="origins"):
        _prepare(tmp_path, history, forecast)


def test_utc_columns_supported_but_disagreeing_timestamps_rejected(tmp_path):
    history, forecast = _inputs()
    history = history.rename_axis("delivery_start_utc").reset_index()
    forecast = forecast.rename_axis("timestamp_utc").reset_index()
    _, _, audit = _prepare(tmp_path, history, forecast)
    assert audit["forecast_hours"] == 24
    forecast["delivery_start_utc"] = forecast.timestamp_utc + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="disagree"):
        _prepare(tmp_path, history, forecast)


def test_one_complete_day_is_sufficient(tmp_path):
    history, forecast = _inputs(history_days=1)
    _, _, audit = _prepare(tmp_path, history, forecast)
    assert audit["history_days"] == 1


@pytest.mark.parametrize("parent", ["autonomous", "kalman", "nuclear_kalman", "kalman_hybrid"])
def test_incumbent_namespaces_cannot_be_written(tmp_path, parent):
    history, forecast = _inputs()
    with pytest.raises(ValueError, match="namespace"):
        report.render_live_report(history, forecast, zone="DE", delivery_day="2026-09-23",
                                  output_path=tmp_path / parent / "report.html", metadata={})
    assert not (tmp_path / parent).exists()


def test_existing_output_is_never_overwritten(tmp_path):
    history, forecast = _inputs()
    path = tmp_path / "sentinel.html"
    path.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        report.render_live_report(history, forecast, zone="DE", delivery_day="2026-09-23",
                                  output_path=path, metadata={})
    assert path.read_text(encoding="utf-8") == "keep me"
