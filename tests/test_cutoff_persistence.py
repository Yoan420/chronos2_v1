from __future__ import annotations

import numpy as np
import pandas as pd

from chronos2_modular.cutoff_persistence import cutoff_persistence_frame


def test_cutoff_persistence_uses_only_d_minus_1_0800():
    index = pd.date_range(
        "2026-08-07 00:00",
        periods=24,
        freq="h",
        tz="Europe/Paris",
    )
    source = pd.Series(
        [1.0, 2.0, 9.0],
        index=pd.DatetimeIndex(
            [
                "2026-08-06 07:00+02:00",
                "2026-08-06 08:00+02:00",
                "2026-08-06 09:00+02:00",
            ]
        ),
    )

    result = cutoff_persistence_frame(
        source,
        index,
        cutoff_local_time="08:00",
        max_age_hours=72,
    )

    assert np.allclose(result["value"], 2.0)
    assert np.allclose(result["age_hours"], 0.0)


def test_stale_value_is_rejected():
    index = pd.date_range(
        "2026-08-07 00:00",
        periods=24,
        freq="h",
        tz="Europe/Paris",
    )
    source = pd.Series(
        [1.0],
        index=pd.DatetimeIndex(["2026-08-01 08:00+02:00"]),
    )

    result = cutoff_persistence_frame(
        source,
        index,
        cutoff_local_time="08:00",
        max_age_hours=72,
    )

    assert result["value"].isna().all()
