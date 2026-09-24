from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import chronos2_hourly.historical_blend_replay as module


def _write_run(path: Path, *, reference: bool) -> None:
    path.mkdir(parents=True)
    (path / "inputs").mkdir()
    index = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    backtest = pd.DataFrame(
        {
            "delivery_start_utc": index,
            "actual": np.arange(6, dtype=float),
            "fold_id": 1,
            "residual_corrected__q10": np.arange(6) - 1.0,
            "residual_corrected__q50": np.arange(6) + 1.0,
            "residual_corrected__q90": np.arange(6) + 3.0,
        }
    )
    if reference:
        backtest["mkonline_primary__q50"] = np.arange(6) - 1.0
    backtest.to_csv(path / "backtest_hourly_oof.csv.gz", index=False)
    forecast = backtest.drop(columns=["actual", "fold_id"]).copy()
    forecast.to_csv(path / "forecast_hourly_fr.csv", index=False)
    (path / "run_manifest.json").write_text(
        json.dumps(
            {
                "zone": "FR",
                "timezone": "Europe/Paris",
                "autonomous_weight": 0.5 if reference else None,
                "mkonline_weight": 0.5 if reference else None,
            }
        ),
        encoding="utf-8",
    )
    (path / "metrics_hourly.json").write_text(
        json.dumps(
            {
                "metrics": [],
                "training_diagnostics": {
                    "zone_benchmark": {
                        "weights": {
                            "autonomous": 0.5,
                            "mkonline_primary": 0.5,
                        }
                    }
                    if reference
                    else {}
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "artifact_checksums.json").write_text(
        json.dumps({"algorithm": "sha256", "artifacts": []}),
        encoding="utf-8",
    )


def test_replays_frozen_blend_without_refitting(tmp_path: Path, monkeypatch) -> None:
    autonomous = tmp_path / "auto"
    reference = tmp_path / "reference"
    output = tmp_path / "output"
    _write_run(autonomous, reference=False)
    _write_run(reference, reference=True)
    monkeypatch.setattr(module, "FINAL_HOURS", 6)
    monkeypatch.setattr(
        module,
        "write_hourly_html_report",
        lambda _run, *, output_path, **_kwargs: output_path.write_text(
            "<html></html>", encoding="utf-8"
        ),
    )

    module.publish_historical_blend_replay(
        autonomous,
        reference,
        output,
        residual_load_source="chronos2_historical_replay",
    )

    frame = pd.read_csv(output / "backtest_hourly_oof.csv.gz")
    np.testing.assert_allclose(frame["mkonline_blend__q50"], np.arange(6))
    np.testing.assert_allclose(
        frame["mkonline_blend__q90"] - frame["mkonline_blend__q10"],
        4.0,
    )
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["mkonline_weights_retrained"] is False
    assert manifest["residual_load_source"] == "chronos2_historical_replay"
    checksums = json.loads((output / "artifact_checksums.json").read_text())
    assert Path(checksums["output_directory"]) == output
