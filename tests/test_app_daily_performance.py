from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import chronos2_hourly.app_service as app_service
from app_multizone import (
    _overall_performance,
    _performance_chart,
    _performance_series,
    _statistics_period_table,
    _weekend_bands,
    _zone_performance_table,
)
from chronos2_hourly.app_service import StatisticsDataset, ZoneStatus


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _statistics_dataset(
    *,
    timezone_name: str = "Europe/Paris",
    start_day: str = "2026-08-17",
    days: int = 2,
    candidate_errors: tuple[float, ...] = (1.0, 1.0),
    benchmark_errors: tuple[float, ...] | None = (2.0, 0.5),
    candidate_label: str = "Chronos-2 corrigé",
    scope_note: str | None = None,
    path: Path | None = None,
) -> StatisticsDataset:
    if len(candidate_errors) != days:
        raise ValueError("candidate_errors must contain one value per day")
    if benchmark_errors is not None and len(benchmark_errors) != days:
        raise ValueError("benchmark_errors must contain one value per day")

    local_index = pd.date_range(
        pd.Timestamp(start_day, tz=timezone_name),
        periods=24 * days,
        freq="h",
    )
    actual = np.linspace(40.0, 80.0, len(local_index))
    candidate_error = np.repeat(np.asarray(candidate_errors, dtype=float), 24)
    frame = pd.DataFrame(
        {
            "timestamp": local_index.tz_convert("UTC"),
            "actual": actual,
            "candidate": actual + candidate_error,
        }
    )
    if benchmark_errors is None:
        frame["benchmark"] = np.nan
        benchmark_column = None
        benchmark_label = None
    else:
        benchmark_error = np.repeat(np.asarray(benchmark_errors, dtype=float), 24)
        frame["benchmark"] = actual + benchmark_error
        benchmark_column = "storm_dashboard_official__q50"
        benchmark_label = "Storm officiel dashboard"

    return StatisticsDataset(
        path=path or Path("statistics_history_hourly.csv.gz"),
        frame=frame,
        candidate_column="residual_corrected__q50",
        candidate_label=candidate_label,
        benchmark_column=benchmark_column,
        benchmark_label=benchmark_label,
        scope_note=scope_note,
    )


def _status(zone: str) -> ZoneStatus:
    timezones = {
        "FR": "Europe/Paris",
        "DE": "Europe/Berlin",
        "BE": "Europe/Brussels",
        "NL": "Europe/Amsterdam",
        "ES": "Europe/Madrid",
    }
    return ZoneStatus(
        code=zone,
        timezone=timezones[zone],
        enabled=True,
        production_ready=True,
        ready=True,
        runner=Path("runner.py"),
        live_config=Path("live.yaml"),
        checks=("ok",),
        blockers=(),
    )


def _performance_workspace_harness(
    statuses,
    archive_specs,
    datasets_by_path,
):
    # AppTest extracts this function as a standalone script, hence local imports.
    from pathlib import Path
    from types import SimpleNamespace

    import streamlit as st

    from app_multizone import _render_performance_workspace

    archive_by_zone = {
        zone: Path(archive_path) for zone, archive_path in archive_specs
    }

    def cached_best_statistics(
        _registry: str,
        _registry_modified_ns: int,
        _project_root: str,
        zone: str,
        variant: str,
    ):
        archive = archive_by_zone[zone]
        statistics_path = archive / "statistics_history_hourly.csv.gz"
        dataset = datasets_by_path[str(statistics_path)][variant]
        artifact = SimpleNamespace(
            zone=zone,
            archive_kind="live_day_ahead",
            archive_path=archive,
            statistics_path=statistics_path,
            statistics_complete=True,
            missing_realized_days=(),
        )
        return artifact, dataset

    _render_performance_workspace(
        st,
        statuses=statuses,
        cached_best_statistics=cached_best_statistics,
    )


def _workspace_args(
    tmp_path: Path,
    *,
    available_zones: tuple[str, ...] = ("FR", "DE", "BE", "NL", "ES"),
    missing_statistics: tuple[str, ...] = (),
) -> tuple[
    list[ZoneStatus],
    tuple[tuple[str, str], ...],
    dict[str, dict[str, StatisticsDataset]],
]:
    statuses = [_status(zone) for zone in ("FR", "DE", "BE", "NL", "ES")]
    archive_specs: list[tuple[str, str]] = []
    datasets: dict[str, dict[str, StatisticsDataset]] = {}
    for zone in available_zones:
        archive = tmp_path / zone.lower() / f"{zone.lower()}_day_ahead_2026-08-19"
        archive.mkdir(parents=True)
        archive_specs.append((zone, str(archive)))
        statistics_path = archive / "statistics_history_hourly.csv.gz"
        if zone in missing_statistics:
            continue
        statistics_path.write_bytes(b"AppTest fixture")
        timezone_name = _status(zone).timezone
        benchmark = None if zone == "ES" else (2.0, 0.5)
        datasets[str(statistics_path)] = {
            "autonomous": _statistics_dataset(
                timezone_name=timezone_name,
                benchmark_errors=benchmark,
                candidate_label=f"Chronos-2 {zone}",
                path=statistics_path,
            ),
            "production": _statistics_dataset(
                timezone_name=timezone_name,
                candidate_errors=(0.25, 0.25),
                benchmark_errors=benchmark,
                candidate_label=(
                    f"Chronos-2 + MKOnline {zone}"
                    if zone in {"FR", "NL"}
                    else f"Chronos-2 {zone}"
                ),
                path=statistics_path,
            ),
        }
    return statuses, tuple(archive_specs), datasets


def test_performance_series_filters_local_days_and_aggregates() -> None:
    dataset = _statistics_dataset(days=3, candidate_errors=(1.0, 2.0, 3.0), benchmark_errors=(3.0, 2.0, 1.0))

    hourly = _performance_series(
        dataset,
        timezone_name="Europe/Paris",
        start_day=date(2026, 8, 18),
        end_day=date(2026, 8, 19),
        frequency="H",
    )
    daily = _performance_series(
        dataset,
        timezone_name="Europe/Paris",
        start_day=date(2026, 8, 18),
        end_day=date(2026, 8, 19),
        frequency="D",
    )

    assert len(hourly) == 48
    assert hourly["local_timestamp"].dt.date.min() == date(2026, 8, 18)
    assert hourly["local_timestamp"].dt.date.max() == date(2026, 8, 19)
    assert len(daily) == 2
    assert daily["local_label"].tolist() == ["18/08/2026", "19/08/2026"]
    assert str(pd.DatetimeIndex(daily["plot_timestamp"]).tz) == "UTC"

    with pytest.raises(ValueError, match="Fréquence inconnue"):
        _performance_series(
            dataset,
            timezone_name="Europe/Paris",
            start_day=date(2026, 8, 18),
            end_day=date(2026, 8, 19),
            frequency="Q",
        )


def test_weekend_bands_preserve_dst_day_lengths() -> None:
    bands = _weekend_bands(
        start_day=date(2025, 10, 25),
        end_day=date(2025, 10, 26),
        timezone_name="Europe/Paris",
    )

    durations = (bands["end"] - bands["start"]).dt.total_seconds() / 3600.0
    assert durations.tolist() == [24.0, 25.0]


def test_overall_performance_computes_kpis_and_daily_win_rate() -> None:
    dataset = _statistics_dataset()

    overview = _overall_performance(
        dataset,
        timezone_name="Europe/Paris",
        start_day=date(2026, 8, 17),
        end_day=date(2026, 8, 18),
    )

    assert overview["candidate_mae"] == pytest.approx(1.0)
    assert overview["benchmark_mae"] == pytest.approx(1.25)
    assert overview["advantage"] == pytest.approx(0.25)
    assert overview["win_rate"] == pytest.approx(0.5)
    assert overview["hours"] == 48
    assert overview["start"] == date(2026, 8, 17)
    assert overview["end"] == date(2026, 8, 18)


def test_overall_performance_uses_the_same_hours_as_storm() -> None:
    dataset = _statistics_dataset()
    dataset.frame.loc[0, "benchmark"] = np.nan
    dataset.frame.loc[0, "candidate"] = dataset.frame.loc[0, "actual"] + 100.0

    overview = _overall_performance(
        dataset,
        timezone_name="Europe/Paris",
        start_day=date(2026, 8, 17),
        end_day=date(2026, 8, 18),
    )

    assert overview["candidate_mae"] == pytest.approx(1.0)
    assert overview["benchmark_mae"] == pytest.approx((23 * 2.0 + 24 * 0.5) / 47)
    assert overview["advantage"] == pytest.approx(
        overview["benchmark_mae"] - overview["candidate_mae"]
    )
    assert overview["hours"] == 47


def test_es_without_storm_stays_na_and_chart_omits_storm() -> None:
    dataset = _statistics_dataset(
        benchmark_errors=None,
        candidate_label="Chronos-2 Espagne",
    )
    overview = _overall_performance(
        dataset,
        timezone_name="Europe/Madrid",
        start_day=date(2026, 8, 17),
        end_day=date(2026, 8, 18),
    )
    table = _zone_performance_table(
        {"ES": dataset},
        timezone_by_zone={"ES": "Europe/Madrid"},
        start_day=date(2026, 8, 17),
        end_day=date(2026, 8, 18),
    )
    series = _performance_series(
        dataset,
        timezone_name="Europe/Madrid",
        start_day=date(2026, 8, 17),
        end_day=date(2026, 8, 18),
        frequency="D",
    )
    chart = _performance_chart(
        series,
        candidate_label=dataset.candidate_label,
        benchmark_label=dataset.benchmark_label,
        weekend_bands=pd.DataFrame(columns=["start", "end"]),
    ).to_dict()

    assert np.isnan(overview["benchmark_mae"])
    assert np.isnan(overview["advantage"])
    assert np.isnan(overview["win_rate"])
    assert np.isnan(table.loc[0, "MAE Storm"])
    assert np.isnan(table.loc[0, "Win rate quotidien"])
    line_layer = chart["layer"][-1]
    assert line_layer["encoding"]["color"]["scale"]["domain"] == [
        "Prix réalisé",
        "Chronos-2 Espagne",
    ]


def test_statistics_table_uses_positive_advantage_for_a_mae_win() -> None:
    dataset = _statistics_dataset()
    overview = _overall_performance(
        dataset,
        timezone_name="Europe/Paris",
        start_day=date(2026, 8, 17),
        end_day=date(2026, 8, 18),
    )

    periods, summary = _statistics_period_table(
        overview,
        metric="mae",
        sample="daily",
        timezone_name="Europe/Paris",
    )

    assert periods["Période"].tolist() == ["2026-08-18", "2026-08-17"]
    assert periods["Avantage modèle"].tolist() == pytest.approx([-0.5, 1.0])
    assert periods["Résultat"].tolist() == ["Perdu", "Gagné"]
    mae = summary.set_index("Statistic").loc["MAE"]
    assert mae["Wins"] == 1
    assert mae["Losses"] == 1
    assert mae["Win rate vs Storm"] == pytest.approx(0.5)


def test_partial_zone_table_keeps_every_available_country() -> None:
    fr = _statistics_dataset(scope_note="Historique partiel FR")
    es = _statistics_dataset(
        timezone_name="Europe/Madrid",
        benchmark_errors=None,
        candidate_label="Chronos-2 Espagne",
    )

    table = _zone_performance_table(
        {"FR": fr, "ES": es},
        timezone_by_zone={"FR": "Europe/Paris", "ES": "Europe/Madrid"},
        start_day=date(2026, 8, 17),
        end_day=date(2026, 8, 18),
    )

    assert table["Pays"].tolist() == ["FR · France", "ES · Espagne"]
    assert table["Périmètre"].tolist() == ["Partiel", "Complet"]
    assert table["Heures"].tolist() == [48, 48]


def test_performance_archive_contract_rejects_storm_in_prediction_inputs() -> None:
    clean_manifest = {
        "storm_used_as_feature": False,
        "storm_loaded_for_prediction": False,
        "storm_used_for_prediction": False,
        "prediction_inputs": ["fr_residual_load_fcst", "calendar_features"],
    }
    app_service._require_prediction_without_storm(
        clean_manifest,
        archive=Path("fr_day_ahead_2026-08-19"),
    )

    contaminated = {
        **clean_manifest,
        "prediction_inputs": [
            "fr_residual_load_fcst",
            "power.price.fr.euromwh.h.fcst.3mv.storm",
        ],
    }
    with pytest.raises(
        app_service.ExistingForecastArchiveError,
        match="Storm apparait dans les prediction_inputs",
    ):
        app_service._require_prediction_without_storm(
            contaminated,
            archive=Path("fr_day_ahead_2026-08-19"),
        )


def test_app_test_default_workspace_filters_kpis_chart_and_tables(
    tmp_path: Path,
) -> None:
    app = AppTest.from_function(
        _performance_workspace_harness,
        args=_workspace_args(tmp_path),
        default_timeout=20,
    ).run()

    assert not app.exception
    assert app.segmented_control(key="performance_variant").value == "autonomous"
    assert app.selectbox(key="performance_zone").value == "FR"
    assert app.segmented_control(key="performance_frequency").value == "H"
    assert app.selectbox(key="performance_statistic").value == "mae"
    assert app.selectbox(key="performance_sample").value == "daily"
    assert app.date_input(key="performance_period_FR").value == (
        date(2026, 8, 17),
        date(2026, 8, 18),
    )

    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics == {
        "MAE modèle": "1.00 EUR/MWh",
        "MAE Storm": "1.25 EUR/MWh",
        "Avantage modèle": "+0.25 EUR/MWh",
        "Win rate quotidien": "50.0%",
        "Heures évaluées": "48",
    }
    assert len(app.get("vega_lite_chart")) == 1
    assert len(app.dataframe) == 2
    assert app.dataframe[0].value["Pays"].tolist() == [
        "FR · France",
        "DE · Allemagne",
        "BE · Belgique",
        "NL · Pays-Bas",
        "ES · Espagne",
    ]
    assert any(
        "uniquement après gel du forecast candidat" in caption.value
        for caption in app.caption
    )


def test_app_test_country_date_mode_and_frequency_filters(
    tmp_path: Path,
) -> None:
    app = AppTest.from_function(
        _performance_workspace_harness,
        args=_workspace_args(tmp_path),
        default_timeout=20,
    ).run()

    app.segmented_control(key="performance_variant").set_value("production")
    app = app.run()
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["MAE modèle"] == "0.25 EUR/MWh"
    assert metrics["Avantage modèle"] == "+1.00 EUR/MWh"
    assert "Chronos-2 + MKOnline FR" in app.dataframe[0].value["Modèle"].tolist()

    app.selectbox(key="performance_zone").set_value("DE")
    app = app.run()
    app.date_input(key="performance_period_DE").set_value(
        (date(2026, 8, 18), date(2026, 8, 18))
    )
    app.segmented_control(key="performance_frequency").set_value("D")
    app.selectbox(key="performance_statistic").set_value("rmse")
    app.selectbox(key="performance_sample").set_value("weekly")
    app = app.run()

    assert app.selectbox(key="performance_zone").value == "DE"
    assert app.date_input(key="performance_period_DE").value == (
        date(2026, 8, 18),
        date(2026, 8, 18),
    )
    assert app.segmented_control(key="performance_frequency").value == "D"
    assert app.selectbox(key="performance_statistic").value == "rmse"
    assert app.selectbox(key="performance_sample").value == "weekly"
    assert {metric.label: metric.value for metric in app.metric}[
        "Heures évaluées"
    ] == "24"
    assert len(app.dataframe[1].value) == 1


def test_app_test_es_displays_na_without_storm(tmp_path: Path) -> None:
    app = AppTest.from_function(
        _performance_workspace_harness,
        args=_workspace_args(tmp_path),
        default_timeout=20,
    ).run()
    app.selectbox(key="performance_zone").set_value("ES")
    app = app.run()

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["MAE modèle"] == "1.00 EUR/MWh"
    assert metrics["MAE Storm"] == "N/A"
    assert metrics["Avantage modèle"] == "N/A"
    assert metrics["Win rate quotidien"] == "N/A"
    es_row = app.dataframe[0].value.set_index("Pays").loc["ES · Espagne"]
    assert pd.isna(es_row["MAE Storm"])
    assert pd.isna(es_row["Win rate quotidien"])
    assert len(app.get("vega_lite_chart")) == 1


def test_app_test_partial_archives_keep_available_country_visible(
    tmp_path: Path,
) -> None:
    app = AppTest.from_function(
        _performance_workspace_harness,
        args=_workspace_args(
            tmp_path,
            available_zones=("FR", "BE", "ES"),
            missing_statistics=("BE",),
        ),
        default_timeout=20,
    ).run()

    assert not app.exception
    assert not app.error
    warnings = " ".join(item.value for item in app.warning)
    assert "Certaines zones sont temporairement indisponibles" in warnings
    assert "BE:" in warnings
    assert app.selectbox(key="performance_zone").value == "FR"
    assert app.dataframe[0].value["Pays"].tolist() == [
        "FR · France",
        "ES · Espagne",
    ]


def test_main_default_workspace_does_not_launch_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import streamlit as st

    statuses = [_status(zone) for zone in ("FR", "DE", "BE", "NL", "ES")]

    def fake_statuses(_registry: str, *, zones=None):
        if zones is None:
            return statuses
        requested = set(zones)
        return [status for status in statuses if status.code in requested]

    def fake_best_statistics(status, *, project_root, variant="production"):
        del project_root
        benchmark = None if status.code == "ES" else (2.0, 0.5)
        dataset = _statistics_dataset(
            timezone_name=status.timezone,
            benchmark_errors=benchmark,
            candidate_label=f"Chronos-2 {status.code}",
        )
        return SimpleNamespace(archive_kind="live_day_ahead"), dataset

    launcher = Mock(side_effect=AssertionError("aucun modèle au chargement"))
    monkeypatch.setattr(app_service, "inspect_zone_statuses", fake_statuses)
    monkeypatch.setattr(
        app_service,
        "load_best_statistics_history",
        fake_best_statistics,
    )
    monkeypatch.setattr(app_service, "launch_zone_forecast", launcher)
    st.cache_data.clear()

    app = AppTest.from_file(
        PROJECT_ROOT / "app_multizone.py",
        default_timeout=40,
    ).run()

    assert not app.exception
    assert app.segmented_control(key="main_workspace").value == "performance"
    assert any(
        "n'entre jamais dans les features du modèle" in caption.value
        for caption in app.caption
    )
    launcher.assert_not_called()
