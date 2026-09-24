from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.reporting import (
    KALMAN_DAILY_AUDIT_ARTIFACT,
    KALMAN_FILTER_AUDIT_ARTIFACT,
    KALMAN_STATE_AUDIT_ARTIFACT,
    MODEL_LABELS,
    _attach_kalman_diagnostics,
    _default_models,
    _kalman_evaluation_mask,
)
from chronos2_modular.report import build_kalman_diagnostics_html


TIMEZONE = "Europe/Paris"
EVALUATION_START = "2025-08-28"
EVALUATION_END = "2026-08-27"
CANDIDATES = (
    "linear_bias",
    "linear_harmonic",
    "linear_market",
    "linear_weather",
    "linear_renewables",
    "linear_fundamental",
    "linear_fuel",
    "linear_market_weather",
    "linear_market_weather_fuel",
    "linear_scale",
    "ekf_scale",
    "ukf_scale",
)


def _result() -> SimpleNamespace:
    delivery = pd.date_range(
        "2025-08-27T22:00:00Z",
        periods=8_760,
        freq="h",
        tz="UTC",
    )
    actual = 70.0 + 20.0 * np.sin(np.arange(len(delivery)) * 2.0 * np.pi / 24.0)
    shared = {
        "timestamp": delivery.tz_convert(TIMEZONE),
        "actual": actual,
        "origin_timestamp": delivery.tz_convert(TIMEZONE) - pd.Timedelta(days=1),
    }
    return SimpleNamespace(
        zone="FR",
        backtest_native=pd.DataFrame({**shared, "q50": actual + 1.0}),
        backtest_baseline=pd.DataFrame({**shared, "q50": actual + 2.0}),
    )


def _write_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    all_days = pd.date_range("2025-08-12", "2026-08-27", freq="D")
    selected = np.resize(
        np.asarray(["linear_market", "identity", "ukf_scale"], dtype=object),
        len(all_days),
    )
    last_used = pd.Series(all_days - pd.Timedelta(days=1), dtype="datetime64[ns]")
    last_used.iloc[0] = pd.NaT
    daily = pd.DataFrame(
        {
            "local_day": all_days,
            "hours": 24,
            "selected_filter": selected,
            "selected_weight": np.where(selected == "identity", 0.0, 0.75),
            "baseline_trailing_mae": 2.0,
            "selected_trailing_mae": 1.0,
            "raw_correction_mean": -1.25,
            "applied_correction_mean": np.where(selected == "identity", 0.0, -1.0),
            "applied_correction_abs_max": 1.0,
            "last_observation_used": last_used,
        }
    )
    daily.to_csv(directory / KALMAN_DAILY_AUDIT_ARTIFACT, index=False)
    state = pd.DataFrame(
        {
            "local_day": np.repeat(all_days, len(CANDIDATES)),
            "filter_kind": np.tile(CANDIDATES, len(all_days)),
            "mean_innovation": 0.1,
            "innovation_clips_total": 0,
            "minimum_covariance_eigenvalue": 0.001,
            "covariance_repairs_total": 0,
        }
    )
    state.to_csv(
        directory / KALMAN_STATE_AUDIT_ARTIFACT,
        index=False,
        compression="gzip",
    )
    audit = {
        "schema_version": 1,
        "status": "complete",
        "model_key": "residual_kalman",
        "upstream_model": "residual_corrected",
        "algorithm": "governed_daily_kf_ekf_ukf_filter_update",
        "pykalman_version": "0.11.2",
        "filter_only": True,
        "smoother_used": False,
        "em_used": False,
        "quantile_policy": "same additive shift on q10/q50/q90",
        "config": {
            "governance_lookback_days": 90,
            "governance_confirmation_days": 14,
            "governance_weight_step": 0.05,
            "minimum_gain_eur_mwh": 0.10,
            "minimum_relative_gain": 0.01,
        },
        "candidate_kinds": list(CANDIDATES),
        "covariate_columns": ["fr_residual_load_fcst"],
        "warmup_start_day": "2025-08-12",
        "warmup_end_day": "2025-08-27",
        "warmup_days": 16,
        "evaluation_start_day": EVALUATION_START,
        "evaluation_end_day": EVALUATION_END,
        "evaluation_days": 365,
        "evaluation_hours": 8_760,
        "causality_violations": 0,
        "quantile_crossings": 0,
        "used_for_storm": False,
        "storm_used_as_input": False,
        "selected_filter_counts": {
            "identity": 127,
            "linear_market": 127,
            "ukf_scale": 127,
        },
    }
    (directory / KALMAN_FILTER_AUDIT_ARTIFACT).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_kalman_artifacts_attach_and_render_rolling_365_diagnostics(
    tmp_path: Path,
) -> None:
    result = _result()
    _write_artifacts(tmp_path)

    _attach_kalman_diagnostics(
        result,
        directory=tmp_path,
        native_model="residual_kalman",
        baseline_model="residual_corrected",
        timezone=TIMEZONE,
    )

    diagnostics = result.kalman_diagnostics
    assert diagnostics["audit"]["evaluation_hours"] == 8_760
    assert len(diagnostics["daily"]) == 381
    assert set(diagnostics["state"]["filter_kind"]) == set(CANDIDATES)
    rendered = build_kalman_diagnostics_html(result)
    assert rendered.count('data-report-section="kalman-diagnostics"') == 1
    assert 'data-kalman-window-days="365"' in rendered
    assert "Rolling window appariée de 365 jours" in rendered
    assert "1.000" in rendered
    assert "2.000" in rendered
    assert "+50.00%" in rendered
    assert "EKF — biais + échelle bornée" in rendered
    assert "UKF — biais + échelle bornée" in rendered
    assert "KF linéaire — météo et degrés-jours" in rendered
    assert "KF linéaire — vent et solaire prévus" in rendered
    assert "KF linéaire — fondamentaux météo-énergie" in rendered
    assert "KF linéaire — gaz, CO₂ et dynamique des combustibles" in rendered
    assert "KF linéaire — charges résiduelles et météo" in rendered
    assert "KF linéaire — charges résiduelles, météo et combustibles" in rendered
    assert "76 j sélection + 14 j confirmation" in rendered
    assert "0.10 EUR/MWh et 1.00 %" in rendered
    assert "aucun smoother" in rendered
    assert "aucun EM" in rendered
    assert "Storm est exclusivement un benchmark" in rendered
    assert "fr_residual_load_fcst" in rendered
    assert rendered.count("plotly-graph-div") == 1


def test_kalman_reporting_is_optional_without_artifacts(tmp_path: Path) -> None:
    result = _result()

    _attach_kalman_diagnostics(
        result,
        directory=tmp_path,
        native_model="residual_kalman",
        baseline_model="residual_corrected",
        timezone=TIMEZONE,
    )

    assert not hasattr(result, "kalman_diagnostics")
    assert build_kalman_diagnostics_html(result) == ""
    assert MODEL_LABELS["residual_kalman"] == (
        "Correcteur résiduel + Kalman gouverné"
    )
    assert MODEL_LABELS["residual_kalman_weather"] == (
        "Correcteur résiduel + Kalman météo gouverné"
    )
    assert MODEL_LABELS["residual_kalman_hybrid"] == (
        "Correcteur résiduel + banque Kalman marché-météo-combustibles gouvernée"
    )
    assert _default_models(
        pd.DataFrame(
            {
                "residual_kalman__q50": [1.0],
                "residual_corrected__q50": [1.0],
            }
        )
    ) == ("residual_kalman", "residual_corrected")
    assert _default_models(
        pd.DataFrame(
            {
                "residual_kalman_weather__q50": [1.0],
                "residual_corrected__q50": [1.0],
            }
        )
    ) == ("residual_kalman_weather", "residual_corrected")
    assert _default_models(
        pd.DataFrame(
            {
                "residual_kalman_hybrid__q50": [1.0],
                "residual_corrected__q50": [1.0],
            }
        )
    ) == ("residual_kalman_hybrid", "residual_corrected")


def test_partial_or_noncausal_kalman_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    result = _result()
    pd.DataFrame(
        {
            "local_day": ["2026-08-27"],
            "selected_filter": ["identity"],
            "selected_weight": [0.0],
            "applied_correction_mean": [0.0],
        }
    ).to_csv(tmp_path / KALMAN_DAILY_AUDIT_ARTIFACT, index=False)
    with pytest.raises(FileNotFoundError, match="kalman_filter_audit"):
        _attach_kalman_diagnostics(
            result,
            directory=tmp_path,
            native_model="residual_kalman",
            baseline_model="residual_corrected",
            timezone=TIMEZONE,
        )

    _write_artifacts(tmp_path)
    audit_path = tmp_path / KALMAN_FILTER_AUDIT_ARTIFACT
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["smoother_used"] = True
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="smoother_used"):
        _attach_kalman_diagnostics(
            result,
            directory=tmp_path,
            native_model="residual_kalman",
            baseline_model="residual_corrected",
            timezone=TIMEZONE,
        )


def test_kalman_main_backtest_mask_is_strictly_the_audited_365_days(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path)
    delivery = pd.date_range(
        "2025-08-11T22:00:00Z", periods=9_144, freq="h", tz="UTC"
    )
    frame = pd.DataFrame({"delivery_start_utc": delivery})

    mask = _kalman_evaluation_mask(
        frame,
        run_dir=tmp_path,
        native_model="residual_kalman",
        timezone=TIMEZONE,
    )

    selected = pd.DatetimeIndex(delivery[mask]).tz_convert(TIMEZONE)
    assert len(selected) == 8_760
    assert len(pd.Index(selected.date).unique()) == 365
    assert selected[0].date().isoformat() == EVALUATION_START
    assert selected[-1].date().isoformat() == EVALUATION_END
    assert _kalman_evaluation_mask(
        frame,
        run_dir=tmp_path,
        native_model="residual_corrected",
        timezone=TIMEZONE,
    ).all()
    weather_mask = _kalman_evaluation_mask(
        frame,
        run_dir=tmp_path,
        native_model="residual_kalman_weather",
        timezone=TIMEZONE,
    )
    assert np.array_equal(weather_mask, mask)
    hybrid_mask = _kalman_evaluation_mask(
        frame,
        run_dir=tmp_path,
        native_model="residual_kalman_hybrid",
        timezone=TIMEZONE,
    )
    assert np.array_equal(hybrid_mask, mask)

    audit_path = tmp_path / KALMAN_FILTER_AUDIT_ARTIFACT
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["evaluation_days"] = 364
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="365 jours"):
        _kalman_evaluation_mask(
            frame,
            run_dir=tmp_path,
            native_model="residual_kalman",
            timezone=TIMEZONE,
        )
