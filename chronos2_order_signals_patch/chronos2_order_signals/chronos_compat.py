from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from chronos2_modular.common import CALENDAR_COLUMNS, ZoneData
from chronos2_modular.data import calendar_frame


def future_proxy_frame_fixed(
    data: ZoneData,
    future_index: pd.DatetimeIndex,
    origin_position: int,
) -> pd.DataFrame:
    """Version compatible live de future_proxy_frame.

    Pour oracle, la valeur future doit être lue dans
    model_context_covariates, qui contient l'extension live de l'index. Le
    DataFrame data.covariates est limité à l'historique de la cible.
    """
    frame = calendar_frame(future_index)
    for column in data.known_future_columns:
        if column in CALENDAR_COLUMNS:
            continue
        match = re.match(
            r"^known_(.+)_(lag24|lag168|persistence|oracle)$",
            column,
        )
        if not match:
            frame[column] = np.nan
            continue
        alias, strategy = match.groups()

        if strategy == "oracle":
            if column in data.model_context_covariates.columns:
                frame[column] = data.model_context_covariates[
                    column
                ].reindex(future_index).to_numpy()
            elif alias in data.covariates:
                frame[column] = data.covariates[alias].reindex(
                    future_index
                ).to_numpy()
            else:
                frame[column] = np.nan
            continue

        if alias not in data.covariates:
            frame[column] = np.nan
            continue

        series = data.covariates[alias]
        if strategy == "lag24":
            frame[column] = series.reindex(
                future_index - pd.Timedelta(hours=24)
            ).to_numpy()
        elif strategy == "lag168":
            frame[column] = series.reindex(
                future_index - pd.Timedelta(hours=168)
            ).to_numpy()
        elif strategy == "persistence":
            history = series.iloc[:origin_position].dropna()
            frame[column] = (
                float(history.iloc[-1]) if not history.empty else math.nan
            )

    for column in data.known_future_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[data.known_future_columns].astype(np.float32)
