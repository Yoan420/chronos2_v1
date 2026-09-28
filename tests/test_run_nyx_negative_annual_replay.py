from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_negative_probability_cpu as negative
import run_nyx_negative_annual_replay as replay


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_calendar_contains_53_weekly_origins() -> None:
    config = json.loads((ROOT / replay.CONFIG_PATH).read_text(encoding="utf-8"))
    origins = replay.annual_origins(config)
    assert len(origins) == 53
    assert origins[0] == date(2025, 9, 24)
    assert origins[-1] == date(2026, 9, 23)
    assert all((right - left).days == 7 for left, right in zip(origins, origins[1:]))
    assert sum(len(negative.physical_grid(origin, min(origin + pd.Timedelta(days=7), date(2026, 9, 24)), "FR")) for origin in origins) == 8760


def test_clean_clone_preflight_reports_missing_without_writing(tmp_path: Path) -> None:
    config_path = tmp_path / replay.CONFIG_PATH
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes((ROOT / replay.CONFIG_PATH).read_bytes())
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    result = replay.preflight(tmp_path, ("FR",))
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    assert not result["ready"]
    assert result["files"]["sources"]["FR"]["features"]["status"] == "missing"
    assert before == after


def test_metric_and_series_parity_check_detects_a_change() -> None:
    index = pd.date_range("2026-09-23", periods=4, freq="h", tz="UTC")
    observed = pd.Series([-5.0, 0.0, 10.0, -1.0], index=index, name="actual")
    probabilities = pd.DataFrame(
        {
            "p_negative_raw": [0.8, 0.2, 0.1, 0.7],
            "p_negative": [0.75, 0.15, 0.05, 0.65],
            "is_negative_predicted": [True, False, False, True],
        },
        index=index,
    )
    metrics = negative.probability_metrics(observed, probabilities.p_negative)
    tp, fp, fn = metrics["true_positive"], metrics["false_positive"], metrics["false_negative"]
    report = {"models": {"model": {
        "n": 4,
        "negative_hours": metrics["negative_hours"],
        **{key: metrics[key] for key in ("brier", "log_loss", "average_precision")},
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
    }}}
    archived = probabilities.rename(columns={"p_negative_raw": "p_raw"}).copy()
    matched = replay._score_comparison(observed, probabilities, archived, report)
    assert matched["passed"]
    assert matched["series"]["maximum_calibrated_probability_difference"] == 0.0
    changed = archived.copy()
    changed.loc[index[0], "p_negative"] += 0.01
    difference = replay._score_comparison(observed, probabilities, changed, report)
    assert not difference["passed"]
    assert difference["series"]["maximum_calibrated_probability_difference"] == pytest.approx(0.01)


def test_optional_archived_source_preflight() -> None:
    archive = ROOT / "runs/experiments/nyx_improvement_to20260923/negative_prices_v1/plan.json"
    if not archive.is_file():
        pytest.skip("Research archive is absent from this Git checkout")
    result = replay.preflight(ROOT)
    assert result["ready"], result["blockers"]
    assert result["origins_per_country"] == 53
    assert all(item["status"] == "verified" for zone in result["files"]["sources"].values() for item in zone.values())
