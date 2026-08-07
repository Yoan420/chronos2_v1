
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecodedBlock:
    has_block: bool
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    score: float
    mean_probability: float
    positions: tuple[int, ...]


def _logit(values: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(values, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def decode_probability_day(
    probabilities: pd.Series,
    *,
    min_length: int = 3,
    max_length: int = 9,
    solar_start_hour: int = 8,
    solar_end_hour: int = 19,
    min_mean_probability: float = 0.45,
    minimum_score: float = 0.0,
) -> DecodedBlock:
    probs = pd.to_numeric(probabilities, errors="coerce").sort_index()
    eligible = (
        (probs.index.hour >= solar_start_hour)
        & (probs.index.hour <= solar_end_hour)
        & probs.notna().to_numpy()
    )
    values = probs.to_numpy(dtype=float)
    logits = _logit(np.nan_to_num(values, nan=1e-5))
    best = None

    for start in np.flatnonzero(eligible):
        for length in range(min_length, max_length + 1):
            end = int(start) + length
            if end > len(probs):
                break
            pos = np.arange(int(start), end)
            if not np.all(eligible[pos]):
                continue

            idx = probs.index[pos]
            diffs = idx.to_series().diff().dropna()
            if not diffs.empty and not (diffs == pd.Timedelta(hours=1)).all():
                continue

            mean_p = float(np.mean(values[pos]))
            if mean_p < min_mean_probability:
                continue

            score = float(np.sum(logits[pos]))
            if score < minimum_score:
                continue

            candidate = DecodedBlock(
                True,
                idx[0],
                idx[-1],
                score,
                mean_p,
                tuple(int(x) for x in pos),
            )
            if (
                best is None
                or candidate.score > best.score
                or (
                    np.isclose(candidate.score, best.score)
                    and len(candidate.positions) > len(best.positions)
                )
            ):
                best = candidate

    return best or DecodedBlock(False, None, None, 0.0, 0.0, ())


def decode_probability_days(probabilities: pd.Series, **kwargs) -> pd.DataFrame:
    probs = pd.to_numeric(probabilities, errors="coerce").sort_index()
    result = pd.DataFrame(
        {
            "zero_plateau_probability": probs,
            "zero_plateau_block_flag": 0.0,
            "zero_plateau_block_probability": 0.0,
            "zero_plateau_event_probability": 0.0,
            "zero_plateau_block_position": 0.0,
        },
        index=probs.index,
        dtype=float,
    )
    days = probs.index.normalize()

    for day in pd.Index(days.unique()):
        day_pos = np.flatnonzero(days == day)
        day_probs = probs.iloc[day_pos]
        decoded = decode_probability_day(day_probs, **kwargs)
        result.iloc[
            day_pos,
            result.columns.get_loc("zero_plateau_event_probability"),
        ] = float(day_probs.max())

        if not decoded.has_block:
            continue

        abs_pos = [int(day_pos[p]) for p in decoded.positions]
        result.iloc[
            abs_pos,
            result.columns.get_loc("zero_plateau_block_flag"),
        ] = 1.0
        result.iloc[
            day_pos,
            result.columns.get_loc("zero_plateau_block_probability"),
        ] = decoded.mean_probability

        rel = [0.0] if len(abs_pos) == 1 else np.linspace(-1, 1, len(abs_pos))
        result.iloc[
            abs_pos,
            result.columns.get_loc("zero_plateau_block_position"),
        ] = rel

    return result.astype(np.float32)
