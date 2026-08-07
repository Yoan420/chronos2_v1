from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from chronos2_structural_market.residual import restore_final_price_frame


def test_residual_predictions_are_shifted_back_to_price_space():
    index = pd.date_range(
        "2026-08-07 00:00",
        periods=3,
        freq="h",
        tz="Europe/Paris",
    )
    data = SimpleNamespace(
        structural_price_model=pd.Series([50.0, 60.0, 70.0], index=index)
    )
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "actual": [5.0, -2.0, 3.0],
            "point": [4.0, -1.0, 2.0],
            "q10": [1.0, -4.0, 0.0],
            "q50": [4.0, -1.0, 2.0],
            "q90": [7.0, 2.0, 5.0],
        }
    )

    restored = restore_final_price_frame(frame, data)
    assert np.allclose(restored["actual"], [55.0, 58.0, 73.0])
    assert np.allclose(restored["point"], [54.0, 59.0, 72.0])
    assert np.allclose(restored["q50"], [54.0, 59.0, 72.0])
    assert "actual_residual" in restored
    assert "point_residual" in restored
