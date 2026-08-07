
from __future__ import annotations

import numpy as np
import pandas as pd


def gate_weight(
    probability,
    block_flag,
    *,
    threshold: float = 0.55,
    max_weight: float = 0.65,
):
    p = np.asarray(probability, dtype=float)
    flag = np.asarray(block_flag, dtype=float)
    activation = np.clip((p - threshold) / (1 - threshold), 0, 1)
    return max_weight * activation * np.clip(flag, 0, 1)


def apply_soft_zero_gate(
    frame: pd.DataFrame,
    expert_quantiles: dict[str, float],
    *,
    threshold: float = 0.55,
    max_weight: float = 0.65,
) -> pd.DataFrame:
    result = frame.copy()
    weight = gate_weight(
        result["zero_plateau_probability"],
        result["zero_plateau_block_flag"],
        threshold=threshold,
        max_weight=max_weight,
    )
    result["zero_gate_weight"] = weight

    prediction_columns = [
        c
        for c in result.columns
        if c == "point"
        or (c.startswith("q") and len(c) == 3 and c[1:].isdigit())
    ]
    for column in prediction_columns:
        key = "q50" if column == "point" else column
        expert = float(expert_quantiles.get(key, expert_quantiles.get("q50", 0.0)))
        result[f"{column}_pre_gate"] = result[column]
        result[column] = (1 - weight) * pd.to_numeric(
            result[column], errors="coerce"
        ).to_numpy(dtype=float) + weight * expert
    return result
