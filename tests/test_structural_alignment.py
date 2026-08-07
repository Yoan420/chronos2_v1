from __future__ import annotations

import pandas as pd

from chronos2_structural_market.alignment import (
    _contiguous_suffix_start,
)


def test_contiguous_suffix_starts_after_last_gap():
    index = pd.date_range(
        "2026-01-01",
        periods=6,
        freq="h",
        tz="Europe/Paris",
    )
    complete = pd.Series(
        [False, True, True, False, True, True],
        index=index,
    )
    assert _contiguous_suffix_start(complete) == index[4]


def test_no_suffix_if_last_hour_missing():
    index = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="h",
        tz="Europe/Paris",
    )
    complete = pd.Series(
        [True, True, False],
        index=index,
    )
    assert _contiguous_suffix_start(complete) is None
