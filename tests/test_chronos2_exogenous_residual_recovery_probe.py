from __future__ import annotations

import pandas as pd
import pytest

from run_chronos2_exogenous_residual_recovery_probe import (
    _affine_parameters,
    _expected_cutoffs,
    _metrics,
    _zero_intercept_scale,
)


def test_pre_oof_calibration_helpers_are_deterministic() -> None:
    proxy = pd.Series([0.0, 1.0, 2.0, 3.0])
    actual_scale = pd.Series([0.0, 2.0, 4.0, 6.0])
    assert _zero_intercept_scale(actual_scale, proxy) == pytest.approx(2.0)

    actual_affine = pd.Series([3.0, 5.0, 7.0, 9.0])
    intercept, slope = _affine_parameters(actual_affine, proxy)
    assert intercept == pytest.approx(3.0)
    assert slope == pytest.approx(2.0)
    assert _metrics(actual_affine, intercept + slope * proxy)["mae"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_expected_cutoff_is_civil_d_minus_one_0800_across_dst() -> None:
    deliveries = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-03-30T23:00:00Z"),
            pd.Timestamp("2024-10-26T22:00:00Z"),
        ]
    )
    cutoffs = _expected_cutoffs(deliveries, "Europe/Amsterdam")
    assert cutoffs[0] == pd.Timestamp("2024-03-30T07:00:00Z")
    assert cutoffs[1] == pd.Timestamp("2024-10-26T06:00:00Z")
