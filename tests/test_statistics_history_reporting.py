from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.reporting import (
    STORM_AVAILABLE_0800_CONTRACT_ID,
    STORM_BENCHMARK_CONTRACTS,
    STORM_DASHBOARD_CONTRACT_ID,
    build_hourly_zone_result,
    storm_benchmark_contracts,
    write_hourly_html_report,
)
from chronos2_hourly.storm_dashboard import (
    STORM_DASHBOARD_ARTIFACT,
    STORM_DASHBOARD_COLUMN,
)
from chronos2_modular.report import (
    _statistics_source,
    build_statistics_records,
    build_statistics_table_html,
)


def _model_columns(prefix: str, q50: np.ndarray) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__q10": q50 - 5.0,
        f"{prefix}__q50": q50,
        f"{prefix}__q90": q50 + 5.0,
    }


def _write_reporting_run(
    run_dir: Path,
    *,
    duplicate_history_timestamp: bool = False,
    zone: str = "FR",
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    run_dir.mkdir()
    inputs = run_dir / "inputs"
    inputs.mkdir()

    sealed_index = local_delivery_day_index("2026-01-01")
    sealed_actual = np.arange(len(sealed_index), dtype=float) + 40.0
    sealed = pd.DataFrame(
        {
            "delivery_start_utc": sealed_index,
            "forecast_origin_utc": pd.Timestamp("2025-12-31 07:00Z"),
            "mkonline_blend_forecast_origin_utc": pd.Timestamp(
                "2025-12-31 07:00Z"
            ),
            "actual": sealed_actual,
            "storm_evaluation_only__q50": sealed_actual + 3.0,
            **_model_columns("mkonline_blend", sealed_actual + 1.0),
            **_model_columns("residual_corrected", sealed_actual + 2.0),
        }
    )
    sealed.to_csv(
        run_dir / "backtest_hourly_oof.csv.gz",
        index=False,
        compression="gzip",
    )

    forecast_index = local_delivery_day_index("2026-01-04")
    forecast_q50 = np.arange(len(forecast_index), dtype=float) + 60.0
    forecast = pd.DataFrame(
        {
            "delivery_start_utc": forecast_index,
            **_model_columns("mkonline_blend", forecast_q50),
            **_model_columns("residual_corrected", forecast_q50 + 1.0),
        }
    )
    forecast.to_csv(
        run_dir / f"forecast_hourly_{zone.lower()}.csv",
        index=False,
    )

    history_index = local_delivery_day_index("2026-01-02").append(
        local_delivery_day_index("2026-01-03")
    )
    history_actual = np.arange(len(history_index), dtype=float) + 80.0
    history = pd.DataFrame(
        {
            "delivery_start_utc": history_index,
            "forecast_origin_utc": pd.Timestamp("2026-01-01 07:00Z"),
            "mkonline_blend_forecast_origin_utc": pd.Timestamp(
                "2026-01-01 07:00Z"
            ),
            "actual": history_actual,
            "storm_evaluation_only__q50": history_actual + 4.0,
            **_model_columns("mkonline_blend", history_actual + 1.0),
        }
    )
    if duplicate_history_timestamp:
        history = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    history.to_csv(
        run_dir / "statistics_history_hourly.csv.gz",
        index=False,
        compression="gzip",
    )
    (run_dir / "statistics_history_audit.json").write_text(
        json.dumps(
            {
                "report_scope_note": (
                    "Backtest scellé séparé ; Statistics = forecasts live "
                    "réalisés appariés à Storm."
                )
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "metrics_hourly.json").write_text(
        json.dumps(
            {
                "training_diagnostics": {
                    "evaluation_start_local_date": "2026-01-01",
                    "evaluation_end_local_date": "2026-01-01",
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

    context_index = sealed_index.append(forecast_index)
    pd.DataFrame(
        {
            "timestamp": sealed_index,
            "target": sealed_actual,
            "known": 1.0,
        }
    ).to_csv(inputs / "aligned_inputs.csv.gz", index=False, compression="gzip")
    pd.DataFrame(
        {"timestamp": context_index, "known": 1.0}
    ).to_csv(
        inputs / "model_covariates_with_future.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        {
            "alias": ["known"],
            "coverage": [1.0],
            "coverage_after_fill": [1.0],
        }
    ).to_csv(
        inputs / "input_coverage.csv", index=False
    )
    pd.DataFrame({"alias": ["known"], "known_future": [True]}).to_csv(
        inputs / "input_manifest.csv", index=False
    )
    return sealed_index, history_index


def test_reporting_uses_separate_paired_statistics_history_when_present(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    sealed_index, history_index = _write_reporting_run(run_dir)
    sealed_hash = (run_dir / "backtest_hourly_oof.csv.gz").read_bytes()

    result = build_hourly_zone_result(
        run_dir,
        native_model="mkonline_blend",
        baseline_model="residual_corrected",
    )

    assert len(result.backtest_native) == len(sealed_index)
    assert len(result.statistics_candidate) == len(history_index)
    assert len(result.statistics_benchmark) == len(history_index)
    assert result.statistics_benchmark_label == (
        "Storm disponible à 08:00 (évaluation uniquement)"
    )
    assert result.statistics_benchmark_contract["id"] == (
        STORM_AVAILABLE_0800_CONTRACT_ID
    )
    assert result.statistics_benchmark_contract[
        "official_dashboard_metric"
    ] is False
    dashboard_contract = result.statistics_benchmark_contract_catalog[
        STORM_DASHBOARD_CONTRACT_ID
    ]
    assert dashboard_contract == STORM_BENCHMARK_CONTRACTS[
        STORM_DASHBOARD_CONTRACT_ID
    ]
    assert dashboard_contract["artifact_filename"] == (
        STORM_DASHBOARD_ARTIFACT.as_posix()
    )
    assert dashboard_contract["point_column"] == (
        "storm_dashboard_official__q50"
    )
    assert result.statistics_candidate["timestamp"].equals(
        result.statistics_benchmark["timestamp"]
    )
    np.testing.assert_allclose(
        result.statistics_candidate["actual"],
        result.statistics_benchmark["actual"],
    )
    statistics_days = {
        timestamp.date()
        for timestamp in pd.to_datetime(
            result.statistics_candidate["timestamp"], utc=True
        ).dt.tz_convert("Europe/Paris")
    }
    assert statistics_days == {
        pd.Timestamp("2026-01-02").date(),
        pd.Timestamp("2026-01-03").date(),
    }
    assert (run_dir / "backtest_hourly_oof.csv.gz").read_bytes() == sealed_hash

    source = _statistics_source(result)
    assert len(source) == len(history_index)
    records = build_statistics_records([result])
    daily = [record for record in records if record["sample"] == "daily"]
    assert len(daily) == 2
    assert all(record["n"] == record["benchmark_n"] == 24 for record in daily)
    assert all(record["mae"] == 1.0 for record in daily)
    assert all(record["benchmark_mae"] == 4.0 for record in daily)


def test_reporting_keeps_current_day_with_blank_observed_price(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _sealed_index, history_index = _write_reporting_run(run_dir)
    history_path = run_dir / "statistics_history_hourly.csv.gz"
    history = pd.read_csv(history_path)
    current_index = local_delivery_day_index("2026-01-04")
    current_q50 = np.arange(len(current_index), dtype=float) + 60.0
    current = pd.DataFrame(
        {
            "delivery_start_utc": current_index,
            "forecast_origin_utc": pd.Timestamp("2026-01-03 07:00Z"),
            "mkonline_blend_forecast_origin_utc": pd.Timestamp(
                "2026-01-03 07:00Z"
            ),
            "actual": np.nan,
            "storm_evaluation_only__q50": current_q50 + 3.0,
            **_model_columns("mkonline_blend", current_q50),
        }
    )
    pd.concat([history, current], ignore_index=True).to_csv(
        history_path,
        index=False,
        compression="gzip",
    )

    result = build_hourly_zone_result(
        run_dir,
        native_model="mkonline_blend",
        baseline_model="residual_corrected",
    )
    assert len(result.statistics_candidate) == len(history_index) + len(current)
    records = build_statistics_records([result])
    daily = {
        record["period_start"]: record
        for record in records
        if record["sample"] == "daily"
    }
    pending = daily["2026-01-04"]
    assert pending["observed_mean_price"] is None
    assert pending["mean_price"] == pytest.approx(float(np.mean(current_q50)))
    assert pending["benchmark_mean_price"] == pytest.approx(
        float(np.mean(current_q50 + 3.0))
    )
    assert pending["mae"] is None
    assert pending["benchmark_mae"] is None
    assert pending["n"] == pending["benchmark_n"] == 0
    rendered = build_statistics_table_html([result])
    assert '"period_start": "2026-01-04"' in rendered
    assert '"observed_mean_price": null' in rendered
    assert "function formatValue" in rendered


def test_reporting_rejects_duplicate_statistics_history_timestamps(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_reporting_run(run_dir, duplicate_history_timestamp=True)

    with pytest.raises(ValueError, match="dupliques"):
        build_hourly_zone_result(
            run_dir,
            native_model="mkonline_blend",
            baseline_model="residual_corrected",
        )


def test_reporting_prefers_declared_exact_dashboard_storm(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _, history_index = _write_reporting_run(run_dir)
    history_path = run_dir / "statistics_history_hourly.csv.gz"
    history = pd.read_csv(history_path)
    history[STORM_DASHBOARD_COLUMN] = (
        pd.to_numeric(history["actual"], errors="raise") + 2.0
    )
    history.to_csv(history_path, index=False, compression="gzip")
    audit_path = run_dir / "statistics_history_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["storm_primary_report_benchmark"] = STORM_DASHBOARD_COLUMN
    audit["storm_dashboard"] = {
        "coverage": 1.0,
        "missing_hours": 0,
        "available_hours": len(history),
        "expected_hours": len(history),
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    result = build_hourly_zone_result(
        run_dir,
        native_model="mkonline_blend",
        baseline_model="residual_corrected",
    )

    assert len(result.statistics_candidate) == len(history_index)
    assert len(result.statistics_benchmark) == len(history_index)
    assert "officiel dashboard" in result.statistics_benchmark_label
    assert (
        result.statistics_benchmark_contract["id"]
        == STORM_DASHBOARD_CONTRACT_ID
    )
    np.testing.assert_allclose(
        result.statistics_benchmark["q50"]
        - result.statistics_benchmark["actual"],
        2.0,
    )


def test_reporting_rejects_undeclared_dashboard_column(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_reporting_run(run_dir)
    history_path = run_dir / "statistics_history_hourly.csv.gz"
    history = pd.read_csv(history_path)
    history[STORM_DASHBOARD_COLUMN] = history["actual"]
    history.to_csv(history_path, index=False, compression="gzip")

    with pytest.raises(ValueError, match="sans contrat actif"):
        build_hourly_zone_result(
            run_dir,
            native_model="mkonline_blend",
            baseline_model="residual_corrected",
        )


def test_reporting_rejects_incomplete_official_dashboard(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_reporting_run(run_dir)
    history_path = run_dir / "statistics_history_hourly.csv.gz"
    history = pd.read_csv(history_path)
    history[STORM_DASHBOARD_COLUMN] = history["actual"]
    history.loc[0, STORM_DASHBOARD_COLUMN] = np.nan
    history.to_csv(history_path, index=False, compression="gzip")
    audit_path = run_dir / "statistics_history_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["storm_primary_report_benchmark"] = STORM_DASHBOARD_COLUMN
    audit["storm_dashboard"] = {
        "coverage": 1.0,
        "missing_hours": 0,
        "available_hours": len(history),
        "expected_hours": len(history),
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="couvrir exactement"):
        build_hourly_zone_result(
            run_dir,
            native_model="mkonline_blend",
            baseline_model="residual_corrected",
        )


def test_reporting_reads_zone_specific_forecast_filename(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_reporting_run(run_dir, zone="BE")

    result = build_hourly_zone_result(
        run_dir,
        native_model="mkonline_blend",
        baseline_model="residual_corrected",
        zone="BE",
        timezone="Europe/Brussels",
    )

    assert result.zone == "BE"
    assert len(result.forecast_native) == 24
    dashboard_contract = result.statistics_benchmark_contract_catalog[
        STORM_DASHBOARD_CONTRACT_ID
    ]
    assert dashboard_contract == storm_benchmark_contracts(
        "BE", timezone="Europe/Brussels"
    )[STORM_DASHBOARD_CONTRACT_ID]
    assert dashboard_contract["series"] == (
        "power.price.be.euromwh.h.fcst.3mv.storm.da.cache"
    )
    assert dashboard_contract["series_kind"] == "frozen_day_ahead_cache"
    assert dashboard_contract["series_identifier_verified"] is True
    assert dashboard_contract["zone"] == "BE"
    assert dashboard_contract["timezone"] == "Europe/Brussels"


def test_statistics_html_displays_history_scope_note(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_reporting_run(run_dir)
    result = build_hourly_zone_result(
        run_dir,
        native_model="mkonline_blend",
        baseline_model="residual_corrected",
    )

    html = build_statistics_table_html([result])

    assert "Perimetre Statistics" in html
    assert "Backtest scellé séparé" in html
    assert "forecasts live réalisés appariés à Storm" in html


def test_full_hourly_report_renders_evaluation_only_storm_graph(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_reporting_run(run_dir)
    output = write_hourly_html_report(
        run_dir,
        output_path=tmp_path / "report.html",
        native_model="mkonline_blend",
        baseline_model="residual_corrected",
    )

    rendered = output.read_text(encoding="utf-8")
    assert rendered.count('data-report-section="storm-comparison"') == 1
    assert "Comparaison graphique au benchmark actif" in rendered
    assert "Storm disponible à 08:00" in rendered
    assert "Storm officiel dashboard" in rendered
    assert "ne reproduit pas la métrique" in rendered
    assert "il n'entre ni dans les variables, ni dans le modèle" in rendered
    assert "dans la prévision live" in rendered
    assert "statistics_history_or_sealed_backtest" in rendered
