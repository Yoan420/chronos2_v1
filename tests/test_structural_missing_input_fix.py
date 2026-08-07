from __future__ import annotations

import numpy as np
import pandas as pd

from chronos2_structural_market.config import (
    parse_structural_model_config,
)
from chronos2_structural_market.features import (
    _repair_residual_load_for_day,
)


def _config(max_missing: int = 2):
    return parse_structural_model_config(
        {
            "structural_model": {
                "input_repair": {
                    "impute_residual_load": True,
                    "max_residual_load_missing_hours_per_day": max_missing,
                }
            }
        }
    )


def test_repairs_two_internal_missing_hours():
    idx = pd.date_range(
        "2026-07-20",
        periods=24,
        freq="h",
        tz="Europe/Paris",
    )
    values = np.linspace(30.0, 50.0, 24)
    frame = pd.DataFrame(
        {"residual_load_gw": values},
        index=idx,
    )
    frame.iloc[10:12, 0] = np.nan

    repaired, remaining, timestamps = (
        _repair_residual_load_for_day(
            frame,
            _config(2),
        )
    )

    assert remaining == 0
    assert len(timestamps) == 2
    assert repaired["residual_load_gw"].notna().all()


def test_rejects_more_than_two_missing_hours():
    idx = pd.date_range(
        "2026-07-20",
        periods=24,
        freq="h",
        tz="Europe/Paris",
    )
    frame = pd.DataFrame(
        {"residual_load_gw": np.arange(24, dtype=float)},
        index=idx,
    )
    frame.iloc[5:8, 0] = np.nan

    repaired, remaining, timestamps = (
        _repair_residual_load_for_day(
            frame,
            _config(2),
        )
    )

    assert remaining == 3
    assert len(timestamps) == 3
    assert repaired["residual_load_gw"].isna().sum() == 3
