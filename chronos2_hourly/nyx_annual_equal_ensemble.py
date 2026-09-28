"""Fixed equal-ensemble point used before the paired scarcity confirmation.

This is only arithmetic on five aligned forecasts. It does not train or
produce the three historical HGB experts, NYX or Test2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def combine_equal_ensemble(*, nyx: pd.Series, test2: pd.Series,
                           price_residual: pd.Series, price_absolute: pd.Series,
                           augmented_residual: pd.Series) -> pd.Series:
    inputs = (nyx, test2, price_residual, price_absolute, augmented_residual)
    if not all(isinstance(value, pd.Series) for value in inputs):
        raise ValueError("Five aligned forecast Series are required")
    index = nyx.index
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC"
            or not len(index) or index.hasnans or not index.is_unique
            or not index.is_monotonic_increasing):
        raise ValueError("Nonempty ordered unique UTC delivery hours required")
    for value in inputs:
        if (not value.index.equals(index)
                or not np.isfinite(value.to_numpy(dtype=float)).all()
                or any(word in str(value.name).lower()
                       for word in ("actual", "observed", "target", "storm"))):
            raise ValueError("Only finite aligned model forecasts are accepted")
    nyx, test2, residual, absolute, augmented = (value.astype(float) for value in inputs)
    clipped_test2 = nyx + np.clip(test2 - nyx, -20., 20.)
    result = .75 * ((residual + absolute + augmented) / 3.) + .25 * clipped_test2
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite ensemble output")
    return result.rename("nyx_equal_ensemble")
