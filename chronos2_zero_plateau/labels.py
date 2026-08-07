
from __future__ import annotations

import numpy as np
import pandas as pd


def mark_near_zero_plateaus(
    price: pd.Series,
    *,
    low: float = -3.0,
    high: float = 3.0,
    min_consecutive_hours: int = 3,
    solar_start_hour: int = 8,
    solar_end_hour: int = 19,
) -> pd.DataFrame:
    series = pd.to_numeric(price, errors="coerce").sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("price doit avoir un DatetimeIndex.")
    if low > high:
        raise ValueError("low doit être <= high.")

    out = pd.DataFrame(
        {
            "price": series,
            "near_zero_hour": False,
            "plateau_label": 0,
            "plateau_event_id": 0,
        },
        index=series.index,
    )
    event_id = 0

    positions_by_day = pd.Series(
        np.arange(len(series)),
        index=series.index,
    ).groupby(series.index.normalize())

    for _, positions in positions_by_day:
        pos = positions.to_numpy(dtype=int)
        idx = series.index[pos]
        values = series.iloc[pos].to_numpy(dtype=float)
        candidate = (
            np.isfinite(values)
            & (values >= low)
            & (values <= high)
            & (idx.hour >= solar_start_hour)
            & (idx.hour <= solar_end_hour)
        )
        out.iloc[pos, out.columns.get_loc("near_zero_hour")] = candidate

        start = None
        for j in range(len(candidate) + 1):
            active = bool(candidate[j]) if j < len(candidate) else False
            if active and start is None:
                start = j
            if not active and start is not None:
                end = j
                if end - start >= min_consecutive_hours:
                    event_id += 1
                    selected = pos[start:end]
                    out.iloc[selected, out.columns.get_loc("plateau_label")] = 1
                    out.iloc[selected, out.columns.get_loc("plateau_event_id")] = event_id
                start = None
    return out


def daily_primary_events(labelled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    local_days = labelled.index.normalize()

    for day in pd.Index(local_days.unique()):
        block = labelled.loc[local_days == day]
        candidates = []
        selected = block.loc[block["plateau_event_id"] > 0]

        for event_id, event in selected.groupby("plateau_event_id"):
            candidates.append(
                {
                    "event_id": int(event_id),
                    "start": event.index.min(),
                    "end": event.index.max(),
                    "duration_hours": int(len(event)),
                }
            )

        if not candidates:
            rows.append(
                {
                    "day": day,
                    "has_plateau": 0,
                    "start": pd.NaT,
                    "end": pd.NaT,
                    "duration_hours": 0,
                }
            )
            continue

        candidates.sort(key=lambda row: (-row["duration_hours"], row["start"]))
        best = candidates[0]
        rows.append(
            {
                "day": day,
                "has_plateau": 1,
                "start": best["start"],
                "end": best["end"],
                "duration_hours": best["duration_hours"],
            }
        )

    return pd.DataFrame(rows)
