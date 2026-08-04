from __future__ import annotations

import numpy as np
import pandas as pd

from chronos2_order_signals.labels import LABEL_COLUMNS, build_ex_post_labels
from chronos2_order_signals.pit import snapshot_time_for_delivery


def synthetic_target(days: int = 50) -> pd.Series:
    index = pd.date_range(
        "2025-01-01",
        periods=24 * days,
        freq="h",
        tz="Europe/Paris",
    )
    hour = index.hour.to_numpy()
    values = 60.0 + 15.0 * np.sin(2.0 * np.pi * hour / 24.0)

    # Plateau entouré de ruptures.
    day_10 = (index.normalize() == pd.Timestamp("2025-01-11", tz="Europe/Paris"))
    positions = np.flatnonzero(day_10)
    values[positions[8:13]] = 120.0

    # Rampe lisse.
    day_20 = (index.normalize() == pd.Timestamp("2025-01-21", tz="Europe/Paris"))
    positions = np.flatnonzero(day_20)
    values[positions[6:13]] = np.linspace(20.0, 140.0, 7)

    # Jump net.
    day_30 = (index.normalize() == pd.Timestamp("2025-01-31", tz="Europe/Paris"))
    positions = np.flatnonzero(day_30)
    values[positions[12:]] += 100.0

    return pd.Series(values, index=index, name="target")


def test_labels_are_bounded_and_complete() -> None:
    labels = build_ex_post_labels(synthetic_target())
    assert not labels.empty
    for column in LABEL_COLUMNS:
        assert labels[column].notna().all()
        assert labels[column].between(0.0, 1.0).all()


def test_ramp_and_jump_patterns_receive_high_scores() -> None:
    labels = build_ex_post_labels(synthetic_target())

    ramp_day = pd.Timestamp("2025-01-21", tz="Europe/Paris")
    ramp = labels.loc[labels["delivery_day"] == ramp_day]
    assert ramp["order_ramp_pressure"].max() > 0.70
    assert ramp["order_gradient_binding_probability"].max() > 0.50

    jump_day = pd.Timestamp("2025-01-31", tz="Europe/Paris")
    jump = labels.loc[labels["delivery_day"] == jump_day]
    assert jump["order_jump_probability"].max() > 0.95


def test_snapshot_is_exactly_previous_day_at_origin() -> None:
    delivery = pd.Timestamp("2026-08-04", tz="Europe/Paris")
    snapshot = snapshot_time_for_delivery(
        delivery,
        "Europe/Paris",
        8,
        0,
    )
    expected = pd.Timestamp("2026-08-03 06:00:00+00:00")
    assert snapshot == expected
