from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_kalman_residual_experiment as runner
from chronos2_hourly.kalman_residual import KalmanResidualConfig


def test_normalize_zones_is_strict() -> None:
    assert runner.normalize_zones(("fr,de", "NL")) == ("FR", "DE", "NL")
    with pytest.raises(ValueError, match="doublon"):
        runner.normalize_zones(("FR", "fr"))
    with pytest.raises(ValueError, match="non supportees"):
        runner.normalize_zones(("GB",))


def test_source_mapping_keeps_current_nl_and_fr_layout(tmp_path: Path) -> None:
    delivery_day = date(2026, 8, 28)
    assert runner.source_run_for(
        "FR", delivery_day, project_root=tmp_path
    ) == (tmp_path / "runs/live/fr_day_ahead_2026-08-28").resolve()
    assert runner.source_run_for(
        "NL", delivery_day, project_root=tmp_path
    ) == (
        tmp_path / "runs/live/nl_mkonline_v1/nl_day_ahead_2026-08-28"
    ).resolve()


def test_output_must_be_disjoint_from_immutable_source(tmp_path: Path) -> None:
    delivery_day = date(2026, 8, 28)
    source = (tmp_path / delivery_day.isoformat() / "fr").resolve()
    with pytest.raises(ValueError, match="disjointes"):
        runner.run_zone_experiment(
            "FR",
            delivery_day=delivery_day,
            output_root=tmp_path,
            config=KalmanResidualConfig(),
            source_run=source,
        )


def test_metrics_are_paired_and_report_price_means() -> None:
    evaluation = pd.DataFrame(
        {
            "actual": [10.0, 20.0, 30.0],
            "residual_corrected__q50": [13.0, 23.0, 33.0],
            "residual_kalman__q50": [11.0, 21.0, 31.0],
            "storm_dashboard_official__q50": [9.0, np.nan, 29.0],
        }
    )
    metrics, payload = runner._metrics_payload(
        evaluation,
        start_day=date(2025, 8, 28),
        end_day=date(2026, 8, 27),
    )
    indexed = metrics.set_index("model")
    assert indexed.loc["residual_kalman", "mae"] == pytest.approx(1.0)
    assert indexed.loc["residual_corrected", "mae"] == pytest.approx(3.0)
    assert indexed.loc["storm_dashboard_official", "n"] == 2
    assert indexed.loc["residual_kalman", "observed_mean_eur_mwh"] == 20.0
    assert payload["paired_delta_vs_residual_corrected"]["mae_eur_mwh"] == -2.0
    assert payload["storm_used_as_input"] is False


def test_metrics_payload_keeps_zero_baseline_mae_json_safe() -> None:
    evaluation = pd.DataFrame(
        {
            "actual": [10.0, 20.0],
            "residual_corrected__q50": [10.0, 20.0],
            "residual_kalman__q50": [10.0, 20.0],
        }
    )

    _metrics, payload = runner._metrics_payload(
        evaluation,
        start_day=date(2025, 8, 28),
        end_day=date(2026, 8, 27),
    )

    assert payload["paired_delta_vs_residual_corrected"]["mae_improvement_pct"] is None
    json.dumps(payload, allow_nan=False)


def test_cli_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        runner.main(["--help"])
    assert error.value.code == 0
    assert "365 derniers jours" in capsys.readouterr().out
