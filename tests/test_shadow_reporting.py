from pathlib import Path

import pandas as pd

from chronos2_hourly.shadow_reporting import write_forecast_only_shadow_report


def test_shadow_report_contains_only_j1_forecast(tmp_path: Path) -> None:
    forecast = tmp_path / "forecast.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": ["2026-08-24T22:00:00Z"],
            "residual_corrected__q10": [10.0],
            "residual_corrected__q50": [20.0],
            "residual_corrected__q90": [30.0],
        }
    ).to_csv(forecast, index=False)
    report = write_forecast_only_shadow_report(
        forecast,
        output_path=tmp_path / "report.html",
        zone="DE",
        delivery_day="2026-08-25",
        candidate_model="residual_corrected",
    )

    html = report.read_text(encoding="utf-8")
    assert "FORECAST UNIQUEMENT" in html
    assert "Aucune performance historique Saturn" in html
    assert "20.00" in html
    assert "MAE" not in html
    assert "RMSE" not in html
