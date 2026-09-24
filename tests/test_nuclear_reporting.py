from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly import nuclear_reporting
from chronos2_modular.common import ZoneData
from chronos2_modular.report import build_statistics_records


TIMEZONE = "Europe/Paris"


def _fixture(delivery_day: str = "2026-09-08") -> tuple[SimpleNamespace, ZoneData]:
    day = pd.Timestamp(delivery_day).date()
    history_index = pd.date_range(
        pd.Timestamp(day - timedelta(days=730), tz=TIMEZONE),
        pd.Timestamp(day, tz=TIMEZONE),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC").rename("delivery_start_utc")
    future_index = local_delivery_day_index(day).rename("delivery_start_utc")
    actual = 50.0 + np.sin(np.arange(len(history_index)) / 24.0)
    origin = pd.Timestamp(day - timedelta(days=731), tz=TIMEZONE).tz_convert("UTC")
    raw_history = pd.DataFrame(
        {
            "q10": actual - 3.0,
            "q50": actual + 2.0,
            "q90": actual + 7.0,
            "actual": actual,
            "forecast_origin_utc": origin,
        },
        index=history_index,
    )
    residual = pd.DataFrame(
        {"delivery_start_utc": history_index, "actual": actual, "forecast_origin_utc": origin}
    )
    forecast = pd.DataFrame(
        {
            "delivery_start_utc": future_index,
            "forecast_origin_utc": future_index[0] - pd.Timedelta(hours=16),
        }
    )
    for quantile, offset in (("q10", -5.0), ("q50", 0.0), ("q90", 5.0)):
        residual[f"residual_corrected__{quantile}"] = actual + 1.0 + offset
        residual[f"chronos2__{quantile}"] = actual + 2.0 + offset
        forecast[f"residual_corrected__{quantile}"] = 55.0 + offset
        forecast[f"chronos2__{quantile}"] = 56.0 + offset
    cutoff = pd.Timestamp(day - timedelta(days=365), tz=TIMEZONE).tz_convert("UTC")
    kalman_history = residual.loc[residual["delivery_start_utc"] >= cutoff].copy()
    kalman_forecast = forecast.copy()
    for quantile in ("q10", "q50", "q90"):
        kalman_history[f"residual_kalman__{quantile}"] = (
            kalman_history[f"residual_corrected__{quantile}"] - 0.25
        )
        kalman_forecast[f"residual_kalman__{quantile}"] = (
            kalman_forecast[f"residual_corrected__{quantile}"] - 0.25
        )
    all_index = history_index.append(future_index)
    covariates = pd.DataFrame(
        {"timestamp": all_index, "fr_residual_load_fcst": 30_000.0, "fr_nuclear_fcst": 40_000.0}
    )
    cov_frame = covariates.set_index("timestamp")
    result = SimpleNamespace(
        raw_history=raw_history,
        residual_statistics=residual,
        source_forecast=forecast,
        covariates=covariates,
        kalman_view=SimpleNamespace(backtest=kalman_history, forecast=kalman_forecast),
        audit={"diagnostic_only": True, "production": False},
    )
    data = ZoneData(
        zone="FR",
        timezone=TIMEZONE,
        frequency="h",
        target=pd.Series(actual, index=history_index.tz_convert(TIMEZONE), name="target"),
        covariates=cov_frame.reindex(history_index),
        model_context_covariates=cov_frame,
        known_future_columns=["fr_residual_load_fcst", "fr_nuclear_fcst"],
        coverage=pd.DataFrame(),
        input_manifest=pd.DataFrame({"alias": ["fr_residual_load_fcst"]}),
        diagnostics={},
    )
    return result, data


def _capture_renderer(monkeypatch: pytest.MonkeyPatch) -> list:
    captured = []

    def render(results, config, path):
        captured.extend(results)
        path.write_text(
            "<html><main><h3>Comparaison au modèle prix seul</h3>"
            "<table><tr><th>Gain %</th></tr></table>Covariables natives"
            "<p>Prévision opérationnelle du jour</p></main></html>",
            encoding="utf-8",
        )

    monkeypatch.setattr(nuclear_reporting, "write_html_report", render)
    return captured


def test_report_refuses_different_kalman_upstream_before_any_attribution(tmp_path):
    result, data = _fixture()
    result.kalman_view.forecast["residual_corrected__q50"] += 1.0
    with pytest.raises(ValueError, match="Kalman upstream forecast differs"):
        nuclear_reporting.render_nuclear_reports(
            result, data=data, zone="FR", delivery_day="2026-09-08", output_directory=tmp_path,
        )


def test_float32_replay_labels_are_scored_with_exact_canonical_observations(tmp_path, monkeypatch):
    result, data = _fixture()
    original = data.target.copy()
    result.residual_statistics["actual"] = result.residual_statistics["actual"].astype("float32")
    result.kalman_view.backtest["actual"] = result.kalman_view.backtest["actual"].astype("float32")
    captured = _capture_renderer(monkeypatch)
    paths = nuclear_reporting.render_nuclear_reports(
        result, data=data, zone="FR", delivery_day="2026-09-08", output_directory=tmp_path,
    )
    for rendered in captured:
        index = pd.DatetimeIndex(rendered.backtest_native.timestamp)
        np.testing.assert_array_equal(rendered.backtest_native.actual, original.reindex(index))
    assert result.residual_statistics.actual.dtype == np.dtype("float32")
    assert "historical_observation_precision" in json.loads(paths["audit"].read_text())


def test_both_reports_share_final365_actual_hours_and_unobserved_delivery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, data = _fixture()
    captured = _capture_renderer(monkeypatch)
    source_before = result.residual_statistics.copy(deep=True)
    target_before = data.target.copy(deep=True)
    paths = nuclear_reporting.render_nuclear_reports(
        result,
        data=data,
        zone="FR",
        delivery_day="2026-09-08",
        output_directory=tmp_path,
        source_audit={"vintages_verified": False},
    )

    assert set(paths) == {"autonomous", "kalman", "audit"}
    assert len(captured) == 2
    autonomous, kalman = captured
    assert autonomous.backtest_native["timestamp"].equals(kalman.backtest_native["timestamp"])
    assert autonomous.backtest_native["actual"].equals(kalman.backtest_native["actual"])
    assert autonomous.metrics_native["n"] == kalman.metrics_native["n"] == 8760
    assert autonomous.metrics_native["mae_q50"] == pytest.approx(1.0)
    assert kalman.metrics_native["mae_q50"] == pytest.approx(0.75)
    for report in captured:
        assert report.statistics_candidate["timestamp"].dt.date.nunique() == 366
        assert report.statistics_candidate.tail(24)["actual"].isna().all()
        assert report.metrics_baseline is None
        assert report.backtest_baseline is None
        assert not hasattr(report, "statistics_benchmark")
        assert not hasattr(report, "forecast_benchmark")
        daily = [row for row in build_statistics_records([report]) if row["sample"] == "daily"]
        assert len(daily) == 366
        assert daily[-1]["period_key"] == "2026-09-08"
        assert daily[-1]["n"] == 0
        assert daily[-1]["mae"] is None
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["diagnostic_only"] is True
    assert audit["production"] is False
    assert audit["evaluation_start_day"] == "2025-09-08"
    assert audit["evaluation_end_day"] == "2026-09-07"
    assert audit["warmup"]["residual_days_before_final365"] == 365
    assert audit["delivery_day_placeholder_hours"] == 24
    assert audit["source_audit"] == {"vintages_verified": False}
    for key, label in (
        ("autonomous", nuclear_reporting.NUCLEAR_MODEL_LABEL),
        ("kalman", nuclear_reporting.NUCLEAR_KALMAN_LABEL),
    ):
        rendered = paths[key].read_text(encoding="utf-8")
        assert label in rendered
        assert 'data-diagnostic-only="true"' in rendered
        assert 'data-production="false"' in rendered
        assert "Warm-up" in rendered
        assert "DST" in rendered
        assert "Storm indisponible" in rendered
        assert "Gain %" not in rendered
        assert "Prévision opérationnelle" not in rendered
        assert "LoRA" not in rendered
    pd.testing.assert_frame_equal(result.residual_statistics, source_before)
    pd.testing.assert_series_equal(data.target, target_before)
    assert data.diagnostics == {}
    assert list(data.input_manifest["alias"]) == ["fr_residual_load_fcst"]


@pytest.mark.parametrize("delivery_day, hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_delivery_dst_physical_hours_preserved(
    delivery_day: str, hours: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, data = _fixture(delivery_day)
    captured = _capture_renderer(monkeypatch)
    paths = nuclear_reporting.render_nuclear_reports(
        result, data=data, zone="FR", delivery_day=delivery_day, output_directory=tmp_path
    )
    for report in captured:
        assert len(report.forecast_native) == hours
        assert not report.forecast_native["timestamp"].duplicated().any()
        assert report.statistics_candidate.tail(hours)["actual"].isna().all()
    assert json.loads(paths["audit"].read_text())["delivery_day_placeholder_hours"] == hours


def test_historical_replay_displays_published_delivery_actuals_without_changing_forecasts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, data = _fixture()
    captured = _capture_renderer(monkeypatch)
    future_index = local_delivery_day_index("2026-09-08").tz_convert(TIMEZONE)
    data.target = pd.concat([data.target, pd.Series(53.0, index=future_index)])
    frozen_forecast = result.source_forecast.copy(deep=True)
    paths = nuclear_reporting.render_nuclear_reports(
        result, data=data, zone="FR", delivery_day="2026-09-08", output_directory=tmp_path
    )
    for report in captured:
        assert report.statistics_candidate.tail(24)["actual"].eq(53.0).all()
        daily = [row for row in build_statistics_records([report]) if row["sample"] == "daily"]
        assert len(daily) == 365
        assert daily[-1]["n"] == 24
        assert daily[-1]["period_key"] == "2026-09-08"
        assert report.metrics_native["n"] == 8760  # scoring bounds remain D-365..D-1
    assert json.loads(paths["audit"].read_text())["delivery_day_placeholder_hours"] == 0
    pd.testing.assert_frame_equal(result.source_forecast, frozen_forecast)


@pytest.mark.parametrize(
    "invalid",
    ["duplicate", "missing_hour", "actual_mismatch", "crossing", "missing_kalman"],
)
def test_invalid_or_unpaired_inputs_fail_before_writing(
    invalid: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, data = _fixture()
    captured = _capture_renderer(monkeypatch)
    if invalid == "duplicate":
        result.residual_statistics = pd.concat(
            [result.residual_statistics, result.residual_statistics.iloc[[-1]]]
        )
    elif invalid == "missing_hour":
        result.kalman_view.backtest = result.kalman_view.backtest.iloc[1:]
    elif invalid == "actual_mismatch":
        result.kalman_view.backtest.loc[result.kalman_view.backtest.index[0], "actual"] += 1.0
    elif invalid == "crossing":
        result.source_forecast["residual_corrected__q10"] = 200.0
    else:
        result.kalman_view = None
    with pytest.raises(ValueError):
        nuclear_reporting.render_nuclear_reports(
            result, data=data, zone="FR", delivery_day="2026-09-08", output_directory=tmp_path
        )
    assert not captured
    assert list(tmp_path.iterdir()) == []


def test_autonomous_only_real_standard_html_render(tmp_path: Path) -> None:
    result, data = _fixture()
    result.kalman_view = None
    paths = nuclear_reporting.render_nuclear_reports(
        result,
        data=data,
        zone="FR",
        delivery_day="2026-09-08",
        output_directory=tmp_path,
        include_kalman=False,
        source_audit={"source_description": "</pre><script>not executable</script>"},
    )
    assert set(paths) == {"autonomous", "audit"}
    rendered = paths["autonomous"].read_text(encoding="utf-8")
    assert "plotly-graph-div" in rendered
    assert 'id="statistics"' in rendered
    assert nuclear_reporting.NUCLEAR_MODEL_LABEL in rendered
    assert "Gain %" not in rendered
    assert "Comparaison au modèle prix seul" not in rendered
    assert "<script>not executable</script>" not in rendered
    assert "&lt;script&gt;not executable&lt;/script&gt;" in rendered


@pytest.mark.parametrize("target_timezone", ["UTC", TIMEZONE])
def test_refreshed_canonical_revisions_are_report_only_and_audit_required(tmp_path, monkeypatch, target_timezone):
    from test_nuclear_reporting_refresh import sources, run_refresh, DAY
    sources(monkeypatch, actual_count=24, storm_count=24)
    actuals, snapshot, refresh_audit = run_refresh(tmp_path)
    result, data = _fixture(DAY)
    engine_before = result.residual_statistics.copy(deep=True)
    forecast_before = result.source_forecast.copy(deep=True)
    data.target = actuals.tz_convert(target_timezone)
    with pytest.raises(ValueError, match="FINAL365 observations disagree"):
        nuclear_reporting.render_nuclear_reports(
            result, data=data, zone="FR", delivery_day=DAY, output_directory=tmp_path / "strict",
        )
    captured = _capture_renderer(monkeypatch)
    paths = nuclear_reporting.render_nuclear_reports(
        result, data=data, zone="FR", delivery_day=DAY, output_directory=tmp_path / "refreshed",
        storm_archive=snapshot, observed_source_audit=refresh_audit, operational_layout=True,
    )
    audit = json.loads(paths["audit"].read_text())
    assert audit["historical_observation_precision"]["mode"] == "verified_latest_canonical_reporting_observations"
    assert audit["historical_observation_precision"]["max_absolute_revision_eur_mwh"] > 40
    assert audit["publication_mode"] == "forecast_run"
    assert audit["storm_status"] == "verified_standard_report_comparator"
    for item in captured:
        assert str(item.zone_data.target.index.tz) == TIMEZONE
        assert item.backtest_native.actual.eq(100).all()
        daily = [row for row in build_statistics_records([item]) if row["sample"] == "daily"]
        assert len(daily) == 365
        assert daily[-1]["observed_mean_price"] == 120
        assert daily[-1]["benchmark_mean_price"] == 125
        assert item.statistics_freshness["last_complete_common_day_local"] == DAY
        assert item.statistics_freshness["common_hours_last_day"] == 24
    assert "Variante intégrée au lancement Forecast" in paths["autonomous"].read_text(encoding="utf-8")
    assert "Prévision opérationnelle du jour" in paths["autonomous"].read_text(encoding="utf-8")
    pd.testing.assert_frame_equal(result.residual_statistics, engine_before)
    pd.testing.assert_frame_equal(result.source_forecast, forecast_before)
    assert str(data.target.index.tz) == target_timezone


def test_integrated_reports_have_standard_storm_statistics_calendar_headline_and_hourly(tmp_path, monkeypatch):
    from test_nuclear_reporting_refresh import sources, run_refresh, DAY
    sources(monkeypatch, actual_count=24, storm_count=24)
    actuals, snapshot, refresh_audit = run_refresh(tmp_path)
    result, data = _fixture(DAY)
    data.target = actuals.tz_convert(TIMEZONE)
    paths = nuclear_reporting.render_nuclear_reports(
        result, data=data, zone="FR", delivery_day=DAY, output_directory=tmp_path / "reports",
        storm_archive=snapshot, observed_source_audit=refresh_audit, operational_layout=True,
    )
    for key in ("autonomous", "kalman"):
        document = paths[key].read_text(encoding="utf-8")
        for token in ('id="statistics"', 'data-report-section="average-prices"',
                      'data-report-subsection="mean-price-refresh"', "Storm officiel dashboard",
                      "Prix moyen du jour — Storm", "125.00", "calendar", "hourly-comparison"):
            assert token in document
        assert "verified_standard_report_comparator" in document
        assert "diagnostic uniquement" not in document
        assert "Les Statistics natives restent sur leur support complet" not in document


def test_kalman_only_report_renders_one_variant(tmp_path, monkeypatch):
    result, data = _fixture()
    captured = _capture_renderer(monkeypatch)
    paths = nuclear_reporting.render_nuclear_reports(result, data=data, zone="FR",
        delivery_day="2026-09-08", output_directory=tmp_path, report_variants=("kalman",))
    assert set(paths) == {"kalman", "audit"}
    assert len(captured) == 1
    assert not list(tmp_path.glob("*autonomous.html"))
    assert json.loads(paths["audit"].read_text())["report_variants"] == ["kalman"]
