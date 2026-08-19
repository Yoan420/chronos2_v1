from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.models import BlendedResidualCorrector, ResidualCorrector


def _frame(periods: int = 120) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    signal = np.sin(np.arange(periods) * 2.0 * np.pi / 24.0)
    X = pd.DataFrame({"safe_signal": signal}, index=index)
    base = pd.DataFrame(
        {"q10": 40.0, "q50": 50.0, "q90": 60.0},
        index=index,
    )
    target = pd.Series(50.0 + 8.0 * signal, index=index)
    return X, target, base


def _component(*, iterations: int) -> ResidualCorrector:
    return ResidualCorrector(
        backend="sklearn",
        feature_builder_options={
            "include_calendar": False,
            "include_rich_calendar": False,
            "include_daily_profiles": False,
        },
        iterations=iterations,
        depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        sklearn_early_stopping=False,
        max_abs_correction=None,
    )


def test_blend_is_weighted_before_single_final_clip() -> None:
    X, target, base = _frame()
    model = BlendedResidualCorrector(
        {"short": _component(iterations=20), "long": _component(iterations=40)},
        {"short": 0.25, "long": 0.75},
        max_abs_correction=3.0,
    ).fit(X.iloc[:96], target.iloc[:96], base.iloc[:96])

    index = X.index[96:]
    component = {
        name: fitted.predict_correction(X.loc[index], base.loc[index]).to_numpy()
        for name, fitted in model.components_.items()
    }
    expected = np.clip(0.25 * component["short"] + 0.75 * component["long"], -3.0, 3.0)
    correction = model.predict_correction(X.loc[index], base.loc[index])
    corrected = model.predict(X.loc[index], base.loc[index])

    np.testing.assert_allclose(correction, expected)
    np.testing.assert_allclose(corrected["q50"], base.loc[index, "q50"] + expected)
    np.testing.assert_allclose(corrected["q50"] - corrected["q10"], 10.0)
    np.testing.assert_allclose(corrected["q90"] - corrected["q50"], 10.0)


def test_blend_validates_component_weights() -> None:
    component = _component(iterations=5)
    with pytest.raises(ValueError, match="memes noms"):
        BlendedResidualCorrector({"a": component}, {"b": 1.0})
    with pytest.raises(ValueError, match="somme"):
        BlendedResidualCorrector({"a": component}, {"a": 0.5})

