from __future__ import annotations

import numpy as np
import pandas as pd

from runs.tmp.screen_mkonline_neighbour_convex import (
    GROUPS,
    _assert_predeclared_family,
    _fit_convex_l1,
    _predict_convex,
)
from runs.tmp.screen_mkonline_primary_crossfit import (
    _exact_l1_weight,
    _score,
)


def test_exact_l1_weight_is_constrained_optimum() -> None:
    actual = np.array([0.0, 2.0, 4.0, 6.0])
    autonomous = np.zeros(4)
    expert = np.full(4, 4.0)
    weight = _exact_l1_weight(actual, autonomous, expert)
    assert 0.0 <= weight <= 1.0
    mae = np.mean(np.abs(actual - ((1.0 - weight) * autonomous + weight * expert)))
    grid = np.linspace(0.0, 1.0, 10_001)
    grid_mae = np.min(
        np.mean(
            np.abs(
                actual[:, None]
                - (
                    (1.0 - grid[None, :]) * autonomous[:, None]
                    + grid[None, :] * expert[:, None]
                )
            ),
            axis=0,
        )
    )
    assert mae <= grid_mae + 1e-10


def test_convex_l1_respects_nonnegative_mass_cap_and_improves() -> None:
    baseline = np.array([0.0, 0.0, 0.0, 0.0])
    experts = np.array(
        [
            [2.0, 10.0],
            [2.0, 10.0],
            [2.0, -10.0],
            [2.0, -10.0],
        ]
    )
    actual = np.full(4, 1.0)
    weights, audit = _fit_convex_l1(actual, baseline, experts, 0.5)
    assert audit["success"] is True
    assert bool((weights >= -1e-12).all())
    assert weights.sum() <= 0.5 + 1e-10
    prediction = _predict_convex(baseline, experts, weights)
    assert np.mean(np.abs(actual - prediction)) < np.mean(np.abs(actual - baseline))


def test_convex_recipe_family_is_predeclared_with_es() -> None:
    _assert_predeclared_family()
    assert GROUPS == {
        "core_de_be_nl": ("de", "be", "nl"),
        "coupled_at_be_ch_de_nl": ("at", "be", "ch", "de", "nl"),
        "all_with_es": ("at", "be", "ch", "de", "es", "nl"),
    }


def test_b1_gate_requires_aggregate_threshold_and_positive_halves() -> None:
    index = pd.date_range("2025-04-13T22:00:00Z", periods=60 * 24, freq="h")
    actual = np.zeros(len(index))
    baseline = np.ones(len(index))
    candidate = np.concatenate(
        [np.zeros(30 * 24), np.full(30 * 24, 0.5)]
    )
    score = _score(index, actual, baseline, candidate)
    assert score["all"]["gain"] >= 0.75
    assert score["first30"]["gain"] > 0.0
    assert score["last30"]["gain"] > 0.0
    assert score["passes"] is True
