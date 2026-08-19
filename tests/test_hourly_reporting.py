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
)
from chronos2_modular.report import (
    STATISTICS_METRICS,
    _comparison_outcome,
    _sample_metrics,
    _statistics_source,
    _statistics_comparison_payload,
    _win_rate_summary,
    build_statistics_records,
    build_statistics_table_html,
    figure_storm_comparison,
)


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
    assert any(metric["key"] == "mape" for metric in STATISTICS_METRICS)


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
    assert len(summaries) == len(STATISTICS_METRICS) * 3
    assert {
        (summary["sample"], summary["metric"])
        for summary in summaries
    } == {
        (sample, str(metric["key"]))
        for sample in ("daily", "weekly", "monthly")
        for metric in STATISTICS_METRICS
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
    report = build_statistics_table_html([result])
    assert "Résumé candidat vs" not in report
    assert '"has_benchmark": false' in report


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
