from __future__ import annotations

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from evaluate_hourly_backtest import evaluate


def test_evaluate_uses_complete_physical_days_and_paired_bootstrap() -> None:
    index = local_delivery_day_index("2026-03-28").append(
        [
            local_delivery_day_index("2026-03-29"),
            local_delivery_day_index("2026-03-30"),
        ]
    )
    frame = pd.DataFrame(
        {
            "delivery_start_utc": index,
            "actual": np.zeros(len(index)),
            "baseline__q50": np.ones(len(index)),
            "candidate__q50": np.full(len(index), 0.5),
        }
    )

    summary, daily, monthly, hourly = evaluate(
        frame,
        baseline="baseline__q50",
        candidate="candidate__q50",
        actual="actual",
        timezone="Europe/Paris",
        bootstrap_samples=200,
        seed=42,
    )

    assert summary["n_hours"] == 71
    assert summary["n_local_days"] == 3
    assert summary["baseline_mae"] == 1.0
    assert summary["candidate_mae"] == 0.5
    assert summary["paired_day_bootstrap"]["delta_ci95"] == [-0.5, -0.5]
    assert daily["n_hours"].tolist() == [24, 23, 24]
    assert int(monthly["n_hours"].sum()) == 71
    assert int(hourly["n_hours"].sum()) == 71
