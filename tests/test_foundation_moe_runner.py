from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from run_foundation_moe_challenger import (
    Partition,
    _daily_metrics,
    _manifest_entry,
    _metrics,
    _output_directory,
    _paired_bootstrap,
    _partition,
)


def _forecast_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    base = np.linspace(40.0, 63.0, len(index))
    return pd.DataFrame(
        {
            "residual_corrected__q10": base - 5.0,
            "residual_corrected__q50": base,
            "residual_corrected__q90": base + 7.0,
            "chronos2__q50": base + 1.0,
            "ensemble__q50": base + 0.5,
            "catboost__q50": base + 3.0,
            "lear__q50": base - 2.0,
            "actual": base - 1.0,
        },
        index=index,
    )


def test_partition_builds_a_complete_local_curve() -> None:
    result = _partition(
        _forecast_frame(),
        zone="FR",
        timezone="UTC",
        include_target=True,
        live=False,
    )
    assert result.experts.columns.tolist() == [
        "autonomous",
        "chronos2",
        "ensemble",
        "catboost",
        "lear",
    ]
    assert result.horizons.tolist() == list(range(24))
    assert result.curves.nunique() == 1
    assert result.target is not None


def test_manifest_entry_uses_the_run_artifact_role() -> None:
    manifest = {
        "artifacts": [
            {
                "path": "C:/source/backtest_hourly_oof.csv.gz",
                "role": "source_backtest",
            },
            {
                "path": "backtest_hourly_oof.csv.gz",
                "role": "run_artifact",
            },
        ]
    }
    selected = _manifest_entry(
        manifest,
        "backtest_hourly_oof.csv.gz",
        role="run_artifact",
    )
    assert selected["role"] == "run_artifact"


def test_metrics_report_candidate_gain_without_crossings() -> None:
    frame = _forecast_frame()
    partition = _partition(
        frame,
        zone="FR",
        timezone="UTC",
        include_target=True,
        live=False,
    )
    assert partition.target is not None
    candidate = partition.anchor.copy()
    candidate.loc[:, ["q10", "q50", "q90"]] -= 1.0
    result = _metrics(
        partition,
        {"autonomous": partition.anchor, "candidate": candidate},
    )
    row = result.loc[
        result["scope"].eq("ALL") & result["model"].eq("candidate")
    ].iloc[0]
    assert row["mae"] == pytest.approx(0.0)
    assert row["mae_gain_vs_autonomous"] == pytest.approx(1.0)
    assert row["quantile_crossings"] == 0


def test_bootstrap_clusters_markets_by_date_and_uses_temporal_blocks() -> None:
    rows = []
    for date_offset in range(8):
        delivery_date = str((pd.Timestamp("2026-01-01") + pd.Timedelta(days=date_offset)).date())
        for market in ("DE", "FR"):
            rows.append(
                {
                    "delivery_date": delivery_date,
                    "market": market,
                    "curve_id": f"{market}:{delivery_date}",
                    "n_tokens": 24,
                    "autonomous_mae": 2.0,
                    "candidate_mae": 1.5,
                    "mae_gain": 0.5,
                }
            )
    result = _paired_bootstrap(
        pd.DataFrame(rows),
        samples=200,
        seed=7,
        block_length_days=3,
    )
    assert result["unit"] == "delivery_date_circular_moving_block"
    assert result["n_units"] == 8
    assert result["markets_per_date_min"] == 2
    assert result["markets_per_date_max"] == 2
    assert result["block_length_days"] == 3
    assert result["observed_mean_gain"] == pytest.approx(0.5)
    assert result["ci95_lower"] == pytest.approx(0.5)
    assert result["ci95_upper"] == pytest.approx(0.5)


def test_daily_metrics_exposes_shared_delivery_date() -> None:
    partition = _partition(
        _forecast_frame(),
        zone="FR",
        timezone="UTC",
        include_target=True,
        live=False,
    )
    daily = _daily_metrics(partition, partition.anchor)
    assert daily["delivery_date"].tolist() == ["2026-01-01"]


def test_output_directory_is_confined_and_immutable(tmp_path) -> None:
    project = tmp_path / "project"
    experiments = project / "runs" / "experiments"
    experiments.mkdir(parents=True)
    accepted = _output_directory(
        "runs/experiments/new_run",
        project_root=project,
    )
    assert accepted == (experiments / "new_run").resolve()
    accepted.mkdir()
    with pytest.raises(FileExistsError, match="immuable"):
        _output_directory(
            "runs/experiments/new_run",
            project_root=project,
        )
    with pytest.raises(ValueError, match="runs/experiments"):
        _output_directory("runs/live/bad", project_root=project)
