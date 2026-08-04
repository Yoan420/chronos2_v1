from __future__ import annotations

import numpy as np
import pandas as pd

from chronos2_modular.regime import (
    REGIME_COLUMNS,
    build_market_regime_frame,
    regime_at_origin,
)


CONFIG = {
    "market_regime": {
        "enabled": True,
        "short_window_hours": 168,
        "long_window_hours": 720,
        "short_min_periods": 72,
        "long_min_periods": 168,
        "negative_threshold": 0.0,
        "spike_threshold": 150.0,
    }
}


def make_target(periods: int = 1000) -> pd.Series:
    index = pd.date_range(
        "2025-01-01",
        periods=periods,
        freq="h",
        tz="Europe/Paris",
    )
    values = 50 + 20 * np.sin(np.arange(periods) * 2 * np.pi / 24)
    return pd.Series(values, index=index, dtype=float)


def test_regime_is_strictly_causal() -> None:
    target = make_target()
    original = build_market_regime_frame(target, CONFIG)

    changed = target.copy()
    changed.iloc[800:] = 10_000.0
    modified = build_market_regime_frame(changed, CONFIG)

    # Modifier target[t:] ne doit pas changer la feature au timestamp t.
    pd.testing.assert_series_equal(
        original.iloc[800],
        modified.iloc[800],
        check_names=False,
    )


def test_origin_regime_uses_last_known_price() -> None:
    target = make_target()
    origin = 900

    expected = build_market_regime_frame(
        pd.concat(
            [
                target.iloc[:origin],
                pd.Series(
                    [np.nan],
                    index=[target.index[origin]],
                ),
            ]
        ),
        CONFIG,
    ).iloc[-1]

    actual = regime_at_origin(target.iloc[:origin], CONFIG)

    for column in REGIME_COLUMNS:
        assert np.isclose(
            actual[column],
            expected[column],
            equal_nan=True,
        )


def test_future_regime_is_constant_by_design() -> None:
    target = make_target()
    values = regime_at_origin(target, CONFIG)

    for column in REGIME_COLUMNS:
        assert np.isfinite(values[column])
