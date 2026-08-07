
from __future__ import annotations

import numpy as np
import pandas as pd

from chronos2_zero_plateau.gate import apply_soft_zero_gate


def test_gate_only_moves_decoded_high_probability_hours():
    frame = pd.DataFrame(
        {
            "point": [20.0, 20.0, 20.0],
            "q10": [10.0, 10.0, 10.0],
            "q50": [20.0, 20.0, 20.0],
            "q90": [30.0, 30.0, 30.0],
            "zero_plateau_probability": [0.9, 0.9, 0.9],
            "zero_plateau_block_flag": [0.0, 1.0, 0.0],
        }
    )
    expert = {"q10": -2.0, "q50": 0.0, "q90": 2.0}

    result = apply_soft_zero_gate(
        frame,
        expert,
        threshold=0.55,
        max_weight=0.65,
    )

    assert np.isclose(result.loc[0, "q50"], 20.0)
    assert result.loc[1, "q50"] < 20.0
    assert np.isclose(result.loc[2, "q50"], 20.0)
    assert result.loc[1, "q10"] <= result.loc[1, "q50"] <= result.loc[1, "q90"]
