from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


LABEL_COLUMNS = (
    "order_ramp_pressure",
    "order_plateau_pressure",
    "order_jump_probability",
    "order_block_pressure",
    "order_gradient_binding_probability",
)


@dataclass(frozen=True)
class LabelSettings:
    jump_threshold: float = 30.0
    jump_softness: float = 12.0
    plateau_scale: float = 5.0
    ramp_min_slope: float = 7.5
    ramp_softness: float = 3.0
    block_boundary_threshold: float = 20.0
    block_boundary_softness: float = 8.0
    gradient_min_slope: float = 5.0
    gradient_softness: float = 2.5
    min_segment_hours: int = 3
    max_segment_hours: int = 6


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _settings(config: Mapping[str, Any] | None) -> LabelSettings:
    raw: Mapping[str, Any] = {}
    if config:
        value = config.get("order_signals", {})
        if isinstance(value, Mapping):
            nested = value.get("labels", {})
            if isinstance(nested, Mapping):
                raw = nested
    kwargs = {
        field: raw.get(field, getattr(LabelSettings(), field))
        for field in LabelSettings.__dataclass_fields__
    }
    return LabelSettings(**kwargs)


def _complete_local_days(target: pd.Series) -> dict[pd.Timestamp, pd.Series]:
    result: dict[pd.Timestamp, pd.Series] = {}
    normalized = target.index.normalize()
    for day, positions in pd.Series(
        np.arange(len(target)), index=normalized
    ).groupby(level=0):
        day_series = target.iloc[positions.to_numpy(dtype=int)]
        if len(day_series) != 24:
            continue
        if day_series.isna().any():
            continue
        result[pd.Timestamp(day)] = day_series
    return result


def _segment_scores(
    prices: np.ndarray,
    previous_price: float,
    next_price: float,
    settings: LabelSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(prices)
    plateau = np.zeros(n, dtype=np.float64)
    ramp = np.zeros(n, dtype=np.float64)
    block = np.zeros(n, dtype=np.float64)
    gradient = np.zeros(n, dtype=np.float64)

    minimum = max(2, int(settings.min_segment_hours))
    maximum = min(n, int(settings.max_segment_hours))

    for length in range(minimum, maximum + 1):
        x = np.arange(length, dtype=np.float64)
        x_centered = x - x.mean()
        x_denom = float(np.sum(x_centered**2))

        for start in range(0, n - length + 1):
            stop = start + length
            segment = prices[start:stop]
            differences = np.diff(segment)

            within_std = float(np.std(segment, ddof=0))
            plateau_score = float(
                np.exp(-within_std / max(settings.plateau_scale, 1e-6))
            )

            positive_share = float(np.mean(differences > 0.0))
            negative_share = float(np.mean(differences < 0.0))
            direction_consistency = max(positive_share, negative_share)
            average_slope = float((segment[-1] - segment[0]) / (length - 1))
            ramp_magnitude = float(
                _sigmoid(
                    (abs(average_slope) - settings.ramp_min_slope)
                    / max(settings.ramp_softness, 1e-6)
                )
            )
            ramp_score = direction_consistency * ramp_magnitude

            left_reference = prices[start - 1] if start > 0 else previous_price
            right_reference = prices[stop] if stop < n else next_price
            left_jump = abs(float(segment[0] - left_reference))
            right_jump = abs(float(right_reference - segment[-1]))
            strongest_boundary = max(left_jump, right_jump)
            boundary_score = float(
                _sigmoid(
                    (
                        strongest_boundary
                        - settings.block_boundary_threshold
                    )
                    / max(settings.block_boundary_softness, 1e-6)
                )
            )
            block_score = plateau_score * boundary_score

            slope = float(
                np.sum(x_centered * (segment - segment.mean()))
                / max(x_denom, 1e-9)
            )
            fitted = segment.mean() + slope * x_centered
            residual_ss = float(np.sum((segment - fitted) ** 2))
            total_ss = float(np.sum((segment - segment.mean()) ** 2))
            r_squared = (
                1.0 - residual_ss / total_ss
                if total_ss > 1e-9
                else 0.0
            )
            diff_scale = abs(float(np.mean(differences))) + 1e-6
            smoothness = float(
                np.exp(-float(np.std(differences, ddof=0)) / diff_scale)
            )
            gradient_magnitude = float(
                _sigmoid(
                    (abs(slope) - settings.gradient_min_slope)
                    / max(settings.gradient_softness, 1e-6)
                )
            )
            gradient_score = (
                max(0.0, min(1.0, r_squared))
                * smoothness
                * gradient_magnitude
            )

            plateau[start:stop] = np.maximum(
                plateau[start:stop], plateau_score
            )
            ramp[start:stop] = np.maximum(ramp[start:stop], ramp_score)
            block[start:stop] = np.maximum(block[start:stop], block_score)
            gradient[start:stop] = np.maximum(
                gradient[start:stop], gradient_score
            )

    return ramp, plateau, block, gradient


def build_ex_post_labels(
    target: pd.Series,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Construit les labels ex post à partir des prix Day-Ahead réalisés.

    Les labels utilisent toute la courbe réalisée du jour : ils ne doivent
    jamais être injectés directement dans Chronos-2. Seules leurs prévisions
    strictement out-of-fold sont exportées dans les vintages PIT.
    """
    settings = _settings(config)
    target = pd.to_numeric(target, errors="coerce").sort_index()
    complete_days = _complete_local_days(target)
    rows: list[pd.DataFrame] = []

    for day, day_series in complete_days.items():
        prices = day_series.to_numpy(dtype=np.float64)
        first_timestamp = day_series.index[0]
        last_timestamp = day_series.index[-1]

        before = target.loc[target.index < first_timestamp].dropna()
        after = target.loc[target.index > last_timestamp].dropna()
        previous_price = (
            float(before.iloc[-1]) if not before.empty else float(prices[0])
        )
        next_price = (
            float(after.iloc[0]) if not after.empty else float(prices[-1])
        )

        previous_augmented = np.concatenate([[previous_price], prices])
        absolute_change = np.abs(np.diff(previous_augmented))
        jump = _sigmoid(
            (absolute_change - settings.jump_threshold)
            / max(settings.jump_softness, 1e-6)
        ).astype(np.float64)

        ramp, plateau, block, gradient = _segment_scores(
            prices,
            previous_price,
            next_price,
            settings,
        )

        frame = pd.DataFrame(
            {
                "order_ramp_pressure": ramp,
                "order_plateau_pressure": plateau,
                "order_jump_probability": jump,
                "order_block_pressure": block,
                "order_gradient_binding_probability": gradient,
            },
            index=day_series.index,
        )
        frame["delivery_day"] = day
        rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=[*LABEL_COLUMNS, "delivery_day"])

    result = pd.concat(rows).sort_index()
    for column in LABEL_COLUMNS:
        result[column] = result[column].clip(0.0, 1.0).astype(np.float32)
    return result
