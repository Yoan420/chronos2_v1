from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.reporting import (
    STORM_AVAILABLE_0800_CONTRACT_ID,
    STORM_BENCHMARK_CONTRACTS,
    STORM_DASHBOARD_CONTRACT_ID,
    _backtest_point_prediction_frame,
    _backtest_prediction_frame,
    _evaluation_mask,
    _expand_report_quantiles,
    _quantile_source_columns,
    build_hourly_zone_result,
)
from chronos2_modular.report import (
    _REPORT_THEME_BOOTSTRAP,
    _REPORT_THEME_CONTROLLER,
    STATISTICS_METRICS,
    STATISTICS_ROLLING_DAYS,
    _comparison_outcome,
    _latest_statistics_window,
    _sample_metrics,
    _statistics_source,
    _statistics_comparison_payload,
    _win_rate_summary,
    average_price_cards,
    average_price_summary,
    build_statistics_records,
    build_statistics_table_html,
    figure_storm_comparison,
)


def test_report_statistics_are_limited_to_latest_365_local_days() -> None:
    timestamps = pd.date_range(
        "2025-01-01", periods=370, freq="D", tz="Europe/Paris"
    )
    source = pd.DataFrame(
        {
            "_timestamp_local": timestamps,
            "q50": np.arange(370, dtype=float),
        }
    )

    limited = _latest_statistics_window(source)

    assert STATISTICS_ROLLING_DAYS == 365
    assert len(limited) == 365
    assert limited["_timestamp_local"].min() == timestamps[-365]
    assert limited["_timestamp_local"].max() == timestamps[-1]


def test_statistics_window_keeps_365_observed_days_plus_current_placeholder() -> None:
    timestamps = pd.date_range(
        "2025-01-01", periods=367, freq="D", tz="Europe/Paris"
    )
    source = pd.DataFrame(
        {
            "_timestamp_local": timestamps,
            "actual": [*np.arange(366, dtype=float), np.nan],
            "q50": np.arange(367, dtype=float),
        }
    )

    limited = _latest_statistics_window(source)

    assert len(limited) == 366
    assert limited["_timestamp_local"].min() == timestamps[1]
    assert limited["_timestamp_local"].max() == timestamps[-1]


def test_average_prices_weight_days_equally_and_pair_storm() -> None:
    local_timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Paris"),
            pd.Timestamp("2026-01-01 01:00:00", tz="Europe/Paris"),
            pd.Timestamp("2026-01-02 00:00:00", tz="Europe/Paris"),
        ]
    )
    candidate = pd.DataFrame(
        {
            "timestamp": local_timestamps,
            "actual": [11.0, 21.0, 45.0],
            "q50": [10.0, 20.0, 40.0],
        }
    )
    storm = pd.DataFrame(
        {
            "timestamp": local_timestamps,
            "actual": [11.0, 21.0, 45.0],
            "q50": [12.0, 22.0, 50.0],
        }
    )
    target_index = pd.date_range(
        "2025-12-31", periods=6, freq="h", tz="Europe/Paris"
    )
    forecast_timestamps = pd.date_range(
        "2026-01-03", periods=3, freq="h", tz="Europe/Paris"
    )
    result = SimpleNamespace(
        zone="FR",
        forecast_native=pd.DataFrame(
            {
                "timestamp": forecast_timestamps,
                "q50": [20.0, 40.0, 60.0],
            }
        ),
        forecast_benchmark=pd.DataFrame(
            {
                "timestamp": forecast_timestamps,
                "q50": [21.0, 39.0, 63.0],
            }
        ),
        forecast_benchmark_label="Storm officiel dashboard",
        backtest_native=candidate,
        statistics_candidate=candidate,
        statistics_benchmark=storm,
        statistics_benchmark_label="Storm officiel dashboard",
        statistics_benchmark_contract={
            "report_label": "Storm officiel dashboard"
        },
        zone_data=SimpleNamespace(
            target=pd.Series(np.arange(6, dtype=float), index=target_index)
        ),
    )

    summary = average_price_summary(result)

    assert summary["live"]["mean"] == pytest.approx(40.0)
    assert summary["live_benchmark"]["mean"] == pytest.approx(41.0)
    assert summary["delta_live_benchmark"] == pytest.approx(-1.0)
    # Equal daily weighting: mean((10 + 20) / 2, 40) = 27.5.
    assert summary["candidate"]["mean"] == pytest.approx(27.5)
    assert summary["benchmark"]["mean"] == pytest.approx(33.5)
    assert summary["delta_candidate_benchmark"] == pytest.approx(-6.0)
    assert summary["candidate"]["days"] == 2
    assert summary["candidate"]["hours"] == 3

    rendered = average_price_cards(result)
    assert rendered.count('data-report-section="average-prices"') == 1
    assert "Prix moyen du jour — modèle" in rendered
    assert "Prix moyen du jour — Storm" in rendered
    assert "Écart du jour modèle − Storm" in rendered
    assert "40.00" in rendered
    assert "41.00" in rendered
    assert "-1.00" in rendered
    assert "mêmes timestamps appariés" in rendered


def test_average_prices_keep_backtest_mean_when_storm_is_unavailable() -> None:
    timestamps = pd.date_range(
        "2026-02-01", periods=4, freq="h", tz="Europe/Madrid"
    )
    candidate = pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual": [20.0, 21.0, 22.0, 23.0],
            "q50": [18.0, 20.0, 22.0, 24.0],
        }
    )
    result = SimpleNamespace(
        zone="ES",
        forecast_native=pd.DataFrame(
            {"timestamp": timestamps, "q50": [30.0, 32.0, 34.0, 36.0]}
        ),
        backtest_native=candidate,
        statistics_candidate=candidate,
        zone_data=SimpleNamespace(
            target=pd.Series(np.arange(4, dtype=float), index=timestamps)
        ),
    )

    summary = average_price_summary(result)
    rendered = average_price_cards(result)

    assert summary["candidate"]["mean"] == pytest.approx(21.0)
    assert summary["benchmark"]["mean"] is None
    assert summary["delta_candidate_benchmark"] is None
    assert summary["live_benchmark"]["mean"] is None
    assert summary["delta_live_benchmark"] is None
    assert "Prix moyen du jour — Storm" in rendered
    assert rendered.count("Indisponible pour cette zone") == 1
    assert "Storm n’est pas disponible pour cette zone" in rendered


def test_statistics_displays_observed_and_storm_freshness() -> None:
    timestamps = pd.date_range(
        "2026-08-26T22:00:00Z", periods=24, freq="h"
    )
    candidate = pd.DataFrame(
        {"timestamp": timestamps, "actual": 100.0, "q50": 101.0}
    )
    storm = pd.DataFrame(
        {"timestamp": timestamps, "actual": 100.0, "q50": 102.0}
    )
    result = SimpleNamespace(
        zone="FR",
        backtest_native=candidate,
        statistics_candidate=candidate,
        statistics_benchmark=storm,
        statistics_candidate_label="Notre modèle",
        statistics_benchmark_label="Storm officiel dashboard",
        statistics_benchmark_contract={
            "report_label": "Storm officiel dashboard",
            "report_note": "Extraction native exacte.",
        },
        statistics_freshness={
            "timezone": "Europe/Paris",
            "actual_extracted_at_utc": "2026-08-26T13:00:00Z",
            "actual_available_end_utc": str(timestamps[-1]),
            "actual_applied_end_utc": str(timestamps[-1]),
            "storm_available": True,
            "storm_extracted_at_utc": "2026-08-26T13:01:00Z",
            "common_delivery_end_utc": str(timestamps[-1]),
            "last_complete_common_day_local": "2026-08-27",
            "common_hours_last_day": 24,
            "expected_common_hours_last_day": 24,
        },
        zone_data=SimpleNamespace(
            target=pd.Series(np.arange(24, dtype=float), index=timestamps)
        ),
    )

    rendered = build_statistics_table_html([result])

    assert "observé disponible jusqu’au 27/08/2026 23:00" in rendered
    assert "Comparaison modèle / observé / Storm actualisée" in rendered
    assert "dernière journée complète : 27/08/2026 (24/24 h)" in rendered
    assert "extraction observé : 26/08/2026 15:00" in rendered
    assert "extraction Storm : 26/08/2026 15:01" in rendered
    assert 'data-report-subsection="mean-price-refresh"' in rendered
    assert "actualisation automatique jusqu’au 27/08/2026 (24/24 h)" in rendered


def test_detailed_report_night_mode_is_persistent_and_rethemes_plotly() -> None:
    assert "chronos2-report-theme" in _REPORT_THEME_BOOTSTRAP
    assert "window.localStorage.getItem" in _REPORT_THEME_BOOTSTRAP
    assert 'document.documentElement.dataset.theme = theme' in (
        _REPORT_THEME_BOOTSTRAP
    )
    assert 'toggle.textContent = dark ? "☀ Mode clair" : "☾ Mode nuit"' in (
        _REPORT_THEME_CONTROLLER
    )
    assert 'window.localStorage.setItem(storageKey, normalized)' in (
        _REPORT_THEME_CONTROLLER
    )
    assert "window.Plotly.relayout" in _REPORT_THEME_CONTROLLER
    assert '"hoverlabel.font.color": colors.text' in _REPORT_THEME_CONTROLLER
    assert "chronos2-theme-change" in _REPORT_THEME_CONTROLLER


def test_statistics_mape_is_percentage_and_ignores_zero_actuals() -> None:
    metrics = _sample_metrics(
        pd.DataFrame(
            {
                "actual": [-10.0, 0.0, 20.0],
                "q50": [-8.0, 999.0, 18.0],
            }
        )
    )

    assert metrics["mape"] == pytest.approx(15.0)
    assert metrics["observed_mean_price"] == pytest.approx(10.0 / 3.0)
    assert metrics["mean_price"] == pytest.approx(((-8.0) + 999.0 + 18.0) / 3)
    assert any(metric["key"] == "mape" for metric in STATISTICS_METRICS)
    assert any(metric["key"] == "mean_price" for metric in STATISTICS_METRICS)


def test_statistics_pending_actual_keeps_forecast_mean_and_blank_metrics() -> None:
    metrics = _sample_metrics(
        pd.DataFrame(
            {
                "actual": [np.nan, np.nan],
                "q50": [10.0, 20.0],
            }
        )
    )

    assert metrics["observed_mean_price"] is None
    assert metrics["mean_price"] == pytest.approx(15.0)
    assert metrics["mae"] is None
    assert metrics["rmse"] is None
    assert metrics["n"] == 0


def test_backtest_report_prefers_model_specific_forecast_origin() -> None:
    delivery = pd.date_range("2026-01-02", periods=2, freq="h", tz="UTC")
    common_origin = pd.Timestamp("2026-01-01 05:00:00+00:00")
    model_origin = pd.Timestamp("2026-01-01 07:00:00+00:00")
    frame = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "actual": [10.0, 20.0],
            "candidate__q10": [5.0, 15.0],
            "candidate__q50": [10.0, 20.0],
            "candidate__q90": [15.0, 25.0],
            "forecast_origin_utc": [common_origin, common_origin],
            "candidate_forecast_origin_utc": [model_origin, model_origin],
        }
    )

    report = _backtest_prediction_frame(
        frame,
        model="candidate",
        row_mask=np.ones(len(frame), dtype=bool),
        timezone="Europe/Paris",
    )

    observed = pd.to_datetime(report["origin_timestamp"], utc=True)
    assert bool((observed == model_origin).all())


def test_point_only_storm_benchmark_does_not_invent_quantiles() -> None:
    delivery = pd.date_range("2026-01-02", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "actual": [10.0, 20.0, 30.0],
            "storm_evaluation_only__q50": [11.0, 18.0, 34.0],
        }
    )

    storm = _backtest_point_prediction_frame(
        frame,
        model="storm_evaluation_only",
        row_mask=np.ones(len(frame), dtype=bool),
        timezone="Europe/Paris",
    )

    assert list(storm.columns) == ["timestamp", "actual", "q50", "point"]
    np.testing.assert_allclose(storm["q50"], [11.0, 18.0, 34.0])
    assert not any(column.startswith("q1") for column in storm if column != "q50")


def test_storm_figure_uses_paired_statistics_history_only() -> None:
    timestamps = pd.date_range(
        "2026-01-01", periods=4, freq="h", tz="Europe/Paris"
    )
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    statistics_candidate = pd.DataFrame(
        {
            "timestamp": timestamps,
            "origin_timestamp": timestamps - pd.Timedelta(days=1),
            "actual": actual,
            "q50": actual + np.array([1.0, -2.0, 3.0, -4.0]),
        }
    )
    storm = pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual": actual,
            "q50": actual + np.array([4.0, -3.0, 2.0, -1.0]),
        }
    )
    # Deliberately different: the graph must not fall back to the native
    # backtest when a dedicated Statistics history is attached.
    sealed_backtest = statistics_candidate.assign(q50=actual + 100.0)
    result = SimpleNamespace(
        zone="FR",
        backtest_native=sealed_backtest,
        statistics_candidate=statistics_candidate,
        statistics_benchmark=storm,
        statistics_candidate_label="Candidat test",
        statistics_benchmark_label="Storm test",
        zone_data=SimpleNamespace(target=pd.Series(actual, index=timestamps)),
    )

    statistics_before = build_statistics_records([result])
    figure = figure_storm_comparison(result)

    assert len(figure.data) == 5
    assert [trace.name for trace in figure.data] == [
        "Observé",
        "Candidat test P50",
        "Storm test P50",
        "Erreur absolue Candidat test",
        "Erreur absolue Storm test",
    ]
    np.testing.assert_allclose(figure.data[1].y, statistics_candidate["q50"])
    np.testing.assert_allclose(figure.data[2].y, storm["q50"])
    np.testing.assert_allclose(figure.data[3].y, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(figure.data[4].y, [4.0, 3.0, 2.0, 1.0])
    assert figure.layout.meta["evaluation_only"] is True
    assert "forecast_native" not in figure.layout.meta["source"]
    assert build_statistics_records([result]) == statistics_before


def test_statistics_win_rate_directions_and_ties_for_every_metric() -> None:
    for metric in STATISTICS_METRICS:
        if metric["higher_is_better"] is None:
            continue
        key = str(metric["key"])
        higher_is_better = bool(metric["higher_is_better"])
        benchmark = 1.0
        better = 2.0 if higher_is_better else 0.0
        worse = 0.0 if higher_is_better else 2.0
        records = [
            {key: better, f"benchmark_{key}": benchmark},
            # Within the shared relative tolerance: a tie, not half a win.
            {key: 1.0 + 5e-10, f"benchmark_{key}": benchmark},
            {key: worse, f"benchmark_{key}": benchmark},
            {key: np.nan, f"benchmark_{key}": benchmark},
        ]

        summary = _win_rate_summary(
            records,
            metric_key=key,
            higher_is_better=higher_is_better,
        )

        assert summary == {
            "wins": 1,
            "ties": 1,
            "losses": 1,
            "comparable_periods": 3,
            "win_rate": 1 / 3,
            "tie_rate": 1 / 3,
            "loss_rate": 1 / 3,
        }
    assert _comparison_outcome(
        np.nan, 1.0, higher_is_better=False
    ) is None


def test_statistics_records_pair_candidate_and_storm_and_render_comparison() -> None:
    timestamps = pd.date_range(
        "2026-01-01", periods=48, freq="h", tz="Europe/Paris"
    )
    actual = np.tile(np.arange(24, dtype=float), 2)
    native = pd.DataFrame(
        {
            "timestamp": timestamps,
            "origin_timestamp": timestamps - pd.Timedelta(days=1),
            "actual": actual,
            "q50": actual + np.tile([1.0, -1.0], 24),
        }
    )
    storm = pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual": actual,
            "q50": actual + 2.0,
        }
    )
    result = SimpleNamespace(
        zone="FR",
        backtest_native=native,
        statistics_benchmark=storm,
        statistics_candidate_label="Candidat test",
        statistics_benchmark_label=(
            "Storm disponible à 08:00 (évaluation uniquement)"
        ),
        statistics_benchmark_contract=dict(
            STORM_BENCHMARK_CONTRACTS[
                STORM_AVAILABLE_0800_CONTRACT_ID
            ]
        ),
        zone_data=SimpleNamespace(
            target=pd.Series(actual, index=timestamps)
        ),
    )

    records = build_statistics_records([result])
    daily = [record for record in records if record["sample"] == "daily"]

    assert len(daily) == 2
    assert all(record["n"] == record["benchmark_n"] == 24 for record in daily)
    assert all(record["mae"] == 1.0 for record in daily)
    assert all(record["benchmark_mae"] == 2.0 for record in daily)
    assert all(record["observed_mean_price"] == 11.5 for record in daily)
    assert all(record["mean_price"] == 11.5 for record in daily)
    assert all(record["benchmark_mean_price"] == 13.5 for record in daily)
    assert all(record["mean_price_absolute_error"] == 0.0 for record in daily)
    assert all(
        record["benchmark_mean_price_absolute_error"] == 2.0
        for record in daily
    )
    assert all(record["mean_price_closest"] == "candidate" for record in daily)
    assert all(record["mean_price_closeness_margin"] == 2.0 for record in daily)

    zones, summaries = _statistics_comparison_payload([result], records)
    assert zones == [
        {
            "key": "FR",
            "candidate_label": "Candidat test",
            "benchmark_label": (
                "Storm disponible à 08:00 (évaluation uniquement)"
            ),
            "benchmark_contract_id": STORM_AVAILABLE_0800_CONTRACT_ID,
            "benchmark_is_official_dashboard": False,
            "has_benchmark": True,
        }
    ]
    comparable_metrics = [
        metric
        for metric in STATISTICS_METRICS
        if metric["higher_is_better"] is not None
    ]
    assert len(summaries) == len(comparable_metrics) * 3
    assert {
        (summary["sample"], summary["metric"])
        for summary in summaries
    } == {
        (sample, str(metric["key"]))
        for sample in ("daily", "weekly", "monthly")
        for metric in comparable_metrics
    }
    daily_mae = next(
        summary
        for summary in summaries
        if summary["sample"] == "daily"
        and summary["metric"] == "mae"
    )
    assert daily_mae["wins"] == 2
    assert daily_mae["ties"] == 0
    assert daily_mae["losses"] == 0
    assert daily_mae["win_rate"] == 1.0

    report = build_statistics_table_html([result])
    assert report.count(
        "Résumé candidat vs Storm disponible à 08:00"
    ) == 1
    assert report.count('data-report-section="statistics"') == 1
    assert "Contrat du benchmark" in report
    assert "cutoff civil J-1 08:00 Europe/Paris" in report
    assert "Win rate vs ${benchmarkLabel}" in report
    assert "Δ candidat − " in report
    assert "Storm disponible à 08:00 (évaluation uniquement)" in report
    assert "ne reproduit pas la métrique" in report
    assert "Storm officiel dashboard" in report
    assert STORM_DASHBOARD_CONTRACT_ID not in report
    assert '"metric": "mae", "wins": 2, "ties": 0' in report
    assert "Prix moyen (EUR/MWh)" in report
    assert '"key": "mean_price"' in report
    assert "Le prix moyen est un niveau" in report
    assert "ne produit donc pas de win rate" in report
    assert report.count('data-report-subsection="mean-price-comparison"') == 1
    assert 'id="statistics-price-sample-select"' in report
    assert "Prix observé vs modèle et Storm" in report
    assert "|Modèle − observé|" in report
    assert "|Storm − observé|" in report
    assert "Forecast le plus proche" in report
    assert "Notre modèle plus proche" in report
    assert "Storm plus proche" in report
    assert "function renderPriceComparisonTable()" in report
    assert 'record.observed_mean_price' in report
    assert report.count('data-report-subsection="mean-price-calendar"') == 1
    assert "Calendrier des écarts du modèle à l’observé" in report
    assert 'id="statistics-price-calendar-container"' in report
    assert "function dailyCalendarHtml(" in report
    assert "function periodCalendarHtml(" in report
    assert "function renderPriceCalendar(" in report
    assert "calendarErrorColor" in report
    assert "Échelle de couleur plafonnée au 95e percentile" in report
    assert (
        "Notre modèle est plus proche de l’observé sur davantage de périodes"
        in report
    )
    assert "Storm est plus proche de l’observé sur davantage de périodes" in report
    assert "le plus souvent plus proche" not in report
    assert "function statisticsPalette()" in report
    assert 'dataset.theme === "dark"' in report
    assert '"chronos2-theme-change"' in report
    assert 'class="statistics-price-color-legend"' in report
    assert "violet" not in report.casefold()
    assert "Écart absolu : faible → élevé" in report
    assert "statistics-price-observed-header" in report
    assert "statistics-price-model-header" in report
    assert "statistics-price-storm-header" in report
    assert "statistics-distance-cell${candidateBest}" in report
    assert "statistics-distance-cell${benchmarkBest}" in report
    assert "palette.distanceLow" in report
    assert "palette.distanceMid" in report
    assert "palette.distanceHigh" in report


def test_statistics_benchmark_coverage_fails_closed() -> None:
    timestamps = pd.date_range(
        "2026-03-29 00:00", periods=23, freq="h", tz="UTC"
    )
    native = pd.DataFrame(
        {
            "timestamp": timestamps,
            "origin_timestamp": timestamps - pd.Timedelta(days=1),
            "actual": np.arange(23, dtype=float),
            "q50": np.arange(23, dtype=float),
        }
    )
    storm = pd.DataFrame(
        {
            "timestamp": timestamps[:-1],
            "actual": np.arange(22, dtype=float),
            "q50": np.arange(22, dtype=float),
        }
    )
    result = SimpleNamespace(
        zone="FR",
        backtest_native=native,
        statistics_benchmark=storm,
        zone_data=SimpleNamespace(
            target=pd.Series(np.arange(23), index=timestamps)
        ),
    )

    with pytest.raises(ValueError, match="couverture"):
        _statistics_source(result)


def test_statistics_without_benchmark_remains_backward_compatible() -> None:
    timestamps = pd.date_range(
        "2026-03-28 23:00", periods=23, freq="h", tz="UTC"
    )
    native = pd.DataFrame(
        {
            "timestamp": timestamps,
            "origin_timestamp": timestamps - pd.Timedelta(days=1),
            "actual": np.arange(23, dtype=float),
            "q50": np.arange(23, dtype=float) + 1.0,
        }
    )
    local_index = timestamps.tz_convert("Europe/Paris")
    result = SimpleNamespace(
        zone="FR",
        backtest_native=native,
        zone_data=SimpleNamespace(
            target=pd.Series(np.arange(23), index=local_index)
        ),
    )

    records = build_statistics_records([result])
    daily = [record for record in records if record["sample"] == "daily"]

    assert len(daily) == 1
    assert daily[0]["n"] == 23
    assert daily[0]["benchmark_n"] == 0
    assert daily[0]["observed_mean_price"] == pytest.approx(11.0)
    assert daily[0]["mean_price"] == pytest.approx(12.0)
    assert daily[0]["mean_price_absolute_error"] == pytest.approx(1.0)
    assert daily[0]["benchmark_mean_price_absolute_error"] is None
    assert daily[0]["mean_price_closest"] == "unavailable"
    report = build_statistics_table_html([result])
    assert "Résumé candidat vs" not in report
    assert '"has_benchmark": false' in report
    assert "Storm indisponible" in report


def test_report_quantile_expansion_preserves_hourly_artifacts() -> None:
    frame = pd.DataFrame(
        {
            "model__q10": [-20.0, 10.0],
            "model__q50": [0.0, 30.0],
            "model__q90": [80.0, 50.0],
        }
    )
    expanded = _expand_report_quantiles(
        frame,
        {
            "q10": "model__q10",
            "q50": "model__q50",
            "q90": "model__q90",
        },
    )

    np.testing.assert_allclose(expanded["q10"], frame["model__q10"])
    np.testing.assert_allclose(expanded["q50"], frame["model__q50"])
    np.testing.assert_allclose(expanded["q90"], frame["model__q90"])
    np.testing.assert_allclose(expanded["point"], frame["model__q50"])
    assert (expanded.filter(regex=r"^q\d+$").diff(axis=1).iloc[:, 1:] >= 0).all().all()


def test_ensemble_report_uses_uncorrected_columns_when_needed() -> None:
    frame = pd.DataFrame(
        {
            "ensemble_uncorrected__q10": [1.0],
            "ensemble_uncorrected__q50": [2.0],
            "ensemble_uncorrected__q90": [3.0],
        }
    )
    assert _quantile_source_columns(
        frame,
        "ensemble",
        allow_final_columns=False,
    ) == {
        "q10": "ensemble_uncorrected__q10",
        "q50": "ensemble_uncorrected__q50",
        "q90": "ensemble_uncorrected__q90",
    }


def test_evaluation_mask_keeps_complete_23_hour_dst_day(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics_hourly.json").write_text(
        json.dumps(
            {
                "training_diagnostics": {
                    "evaluation_start_local_date": "2026-03-29",
                    "evaluation_end_local_date": "2026-03-29",
                }
            }
        ),
        encoding="utf-8",
    )
    timestamps = pd.date_range(
        "2026-03-28 22:00:00+00:00",
        periods=25,
        freq="h",
    )
    frame = pd.DataFrame({"delivery_start_utc": timestamps})

    mask = _evaluation_mask(
        frame,
        run_dir=run_dir,
        timezone="Europe/Paris",
    )

    assert int(mask.sum()) == 23
    selected = timestamps[mask].tz_convert("Europe/Paris")
    assert set(selected.date) == {pd.Timestamp("2026-03-29").date()}


def test_main_backtest_and_probabilities_use_extended_statistics_history(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True)
    sealed_delivery = pd.date_range("2026-08-10T22:00:00Z", periods=24, freq="h")
    extended_delivery = pd.date_range(
        "2026-08-10T22:00:00Z", periods=48, freq="h"
    )

    def predictions(delivery: pd.DatetimeIndex) -> pd.DataFrame:
        actual = np.arange(len(delivery), dtype=float) + 40.0
        return pd.DataFrame(
            {
                "delivery_start_utc": delivery.astype(str),
                "actual": actual,
                "residual_corrected__q10": actual - 5.0,
                "residual_corrected__q50": actual + 1.0,
                "residual_corrected__q90": actual + 7.0,
                "chronos2__q10": actual - 7.0,
                "chronos2__q50": actual + 2.0,
                "chronos2__q90": actual + 9.0,
            }
        )

    predictions(sealed_delivery).to_csv(
        run_dir / "backtest_hourly_oof.csv.gz",
        index=False,
        compression="gzip",
    )
    predictions(extended_delivery).to_csv(
        run_dir / "statistics_history_hourly.csv.gz",
        index=False,
        compression="gzip",
    )
    future = predictions(
        pd.date_range("2026-08-12T22:00:00Z", periods=24, freq="h")
    ).drop(columns="actual")
    future.to_csv(run_dir / "forecast_hourly_fr.csv", index=False)

    aligned = pd.DataFrame(
        {
            "timestamp": extended_delivery.astype(str),
            "target": np.arange(48, dtype=float) + 40.0,
            "known_load": np.arange(48, dtype=float),
        }
    )
    aligned.to_csv(
        inputs / "aligned_inputs.csv.gz", index=False, compression="gzip"
    )
    aligned[["timestamp", "known_load"]].to_csv(
        inputs / "model_covariates_with_future.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame({"alias": ["known_load"], "coverage": [1.0]}).to_csv(
        inputs / "input_coverage.csv", index=False
    )
    pd.DataFrame(
        {"alias": ["known_load"], "known_future": [True]}
    ).to_csv(inputs / "input_manifest.csv", index=False)

    result = build_hourly_zone_result(
        run_dir,
        native_model="residual_corrected",
        baseline_model="chronos2",
        zone="FR",
        timezone="Europe/Paris",
    )

    assert len(result.backtest_native) == 48
    assert len(result.backtest_baseline) == 48
    assert result.backtest_native["timestamp"].max() == pd.Timestamp(
        "2026-08-12 23:00:00+02:00"
    )
    assert result.metrics_native["n"] == 48


def test_statistics_extension_uses_common_native_baseline_support(tmp_path) -> None:
    run_dir = tmp_path / "run"
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True)
    sealed_delivery = pd.date_range("2026-08-10T22:00:00Z", periods=24, freq="h")
    extended_delivery = pd.date_range(
        "2026-08-10T22:00:00Z", periods=48, freq="h"
    )

    def predictions(delivery: pd.DatetimeIndex) -> pd.DataFrame:
        actual = np.arange(len(delivery), dtype=float) + 40.0
        return pd.DataFrame(
            {
                "delivery_start_utc": delivery.astype(str),
                "actual": actual,
                "residual_corrected__q10": actual - 5.0,
                "residual_corrected__q50": actual + 1.0,
                "residual_corrected__q90": actual + 7.0,
                "ensemble__q10": actual - 7.0,
                "ensemble__q50": actual + 2.0,
                "ensemble__q90": actual + 9.0,
            }
        )

    sealed = predictions(sealed_delivery)
    sealed.to_csv(
        run_dir / "backtest_hourly_oof.csv.gz",
        index=False,
        compression="gzip",
    )
    extended = predictions(extended_delivery)
    extended.loc[
        24:,
        ["ensemble__q10", "ensemble__q50", "ensemble__q90"],
    ] = np.nan
    extended.to_csv(
        run_dir / "statistics_history_hourly.csv.gz",
        index=False,
        compression="gzip",
    )
    future = predictions(
        pd.date_range("2026-08-12T22:00:00Z", periods=24, freq="h")
    ).drop(columns="actual")
    future.to_csv(run_dir / "forecast_hourly_fr.csv", index=False)

    aligned = pd.DataFrame(
        {
            "timestamp": extended_delivery.astype(str),
            "target": np.arange(48, dtype=float) + 40.0,
            "known_load": np.arange(48, dtype=float),
        }
    )
    aligned.to_csv(
        inputs / "aligned_inputs.csv.gz", index=False, compression="gzip"
    )
    aligned[["timestamp", "known_load"]].to_csv(
        inputs / "model_covariates_with_future.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame({"alias": ["known_load"], "coverage": [1.0]}).to_csv(
        inputs / "input_coverage.csv", index=False
    )
    pd.DataFrame(
        {"alias": ["known_load"], "known_future": [True]}
    ).to_csv(inputs / "input_manifest.csv", index=False)

    result = build_hourly_zone_result(
        run_dir,
        native_model="residual_corrected",
        baseline_model="ensemble",
        zone="FR",
        timezone="Europe/Paris",
    )

    assert len(result.backtest_native) == 24
    assert len(result.backtest_baseline) == 24
    assert result.backtest_native["timestamp"].equals(
        result.backtest_baseline["timestamp"]
    )
