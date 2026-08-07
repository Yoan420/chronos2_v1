
from __future__ import annotations

import pandas as pd

from chronos2_zero_plateau.labels import mark_near_zero_plateaus


def test_requires_three_consecutive_near_zero_hours():
    index = pd.date_range(
        "2026-06-01 08:00",
        periods=12,
        freq="h",
        tz="Europe/Paris",
    )
    price = pd.Series(
        [20, 10, 2, 0, 1, 12, 2, 0, 15, 20, 25, 30],
        index=index,
        dtype=float,
    )

    labelled = mark_near_zero_plateaus(
        price,
        low=-3,
        high=3,
        min_consecutive_hours=3,
        solar_start_hour=8,
        solar_end_hour=19,
    )

    assert labelled["plateau_label"].sum() == 3
    assert (labelled["plateau_label"].iloc[2:5] == 1).all()
    assert (labelled["plateau_label"].iloc[6:8] == 0).all()
