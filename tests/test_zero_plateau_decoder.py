
from __future__ import annotations

import pandas as pd

from chronos2_zero_plateau.decoder import decode_probability_day


def test_decoder_returns_one_coherent_block():
    index = pd.date_range(
        "2026-06-01 00:00",
        periods=24,
        freq="h",
        tz="Europe/Paris",
    )
    values = [0.05] * 24
    values[11:16] = [0.55, 0.78, 0.90, 0.84, 0.67]
    probabilities = pd.Series(values, index=index)

    block = decode_probability_day(
        probabilities,
        min_length=3,
        max_length=8,
        solar_start_hour=8,
        solar_end_hour=19,
        min_mean_probability=0.45,
    )

    assert block.has_block
    assert block.start.hour >= 11
    assert block.end.hour <= 15
    assert len(block.positions) >= 3
