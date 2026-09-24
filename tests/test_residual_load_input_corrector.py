from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly import residual_load_input_corrector as correction


ALIASES = correction.RESIDUAL_LOAD_ALIASES


def _frames(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    position = np.arange(len(index), dtype=float)
    forecasts = pd.DataFrame(
        {
            alias: 20.0 + offset + position / 100.0
            for offset, alias in enumerate(ALIASES)
        },
        index=index,
    )
    forecasts.index.name = "delivery_start_utc"
    observations = forecasts + 1.0
    return forecasts, observations


def _history_through_day(
    delivery_day: date,
    *,
    history_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = (
        pd.Timestamp(delivery_day - timedelta(days=history_days))
        .tz_localize("Europe/Paris")
        .tz_convert("UTC")
    )
    stop = (
        pd.Timestamp(delivery_day + timedelta(days=1))
        .tz_localize("Europe/Paris")
        .tz_convert("UTC")
    )
    index = pd.date_range(start, stop, freq="h", inclusive="left", tz="UTC")
    return _frames(index)


class _Component:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict_correction(self, X, base_predictions, expert_predictions=None):
        return pd.Series(self.value, index=X.index)


class _FakeCorrector:
    created: list["_FakeCorrector"] = []

    def __init__(self, alias: str, clip: float | None, value: float = 12.0) -> None:
        self.alias = alias
        self.clip = clip
        self.value = float(value)
        self.fit_index: pd.DatetimeIndex | None = None
        self.fit_target: pd.Series | None = None
        self.predict_features: pd.DataFrame | None = None
        self.components_ = {
            "cat_v1": _Component(4.0),
            "hgb31": _Component(20.0),
        }
        self.weights = {"cat_v1": 0.5, "hgb31": 0.5}
        type(self).created.append(self)

    def fit(self, X, y, base_predictions, expert_predictions=None):
        self.fit_index = pd.DatetimeIndex(X.index).copy()
        self.fit_target = pd.Series(y, index=X.index).copy()
        return self

    def predict_correction(self, X, base_predictions, expert_predictions=None):
        self.predict_features = X.copy()
        return pd.Series(self.value, index=X.index)

    def diagnostics(self):
        return {
            "alias": self.alias,
            "fit_rows": 0 if self.fit_index is None else len(self.fit_index),
        }


@pytest.fixture(autouse=True)
def _clear_fake_instances() -> None:
    _FakeCorrector.created.clear()


def _factory(alias: str, max_abs_correction_gw: float | None) -> _FakeCorrector:
    return _FakeCorrector(alias, max_abs_correction_gw)


def test_default_recipe_is_exact_50_50_and_clips_only_after_blending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component_calls: list[dict[str, object]] = []

    class CapturedComponent:
        def __init__(self, **kwargs) -> None:
            component_calls.append(dict(kwargs))

    class CapturedBlend:
        def __init__(self, components, weights, *, max_abs_correction) -> None:
            self.components = components
            self.weights = weights
            self.max_abs_correction = max_abs_correction

    monkeypatch.setattr(correction, "ResidualCorrector", CapturedComponent)
    monkeypatch.setattr(correction, "BlendedResidualCorrector", CapturedBlend)

    model = correction.make_default_residual_load_corrector(
        "fr_residual_load_fcst",
        6.5,
        thread_count=3,
    )

    assert set(model.components) == {"cat_v1", "hgb31"}
    assert model.weights == {"cat_v1": 0.5, "hgb31": 0.5}
    assert model.max_abs_correction == pytest.approx(6.5)
    assert [call["backend"] for call in component_calls] == ["catboost", "sklearn"]
    assert all(call["max_abs_correction"] is None for call in component_calls)
    assert component_calls[0]["thread_count"] == 3
    assert component_calls[1]["sklearn_early_stopping"] is False


def test_features_require_canonical_schema_and_contiguous_utc_index() -> None:
    index = pd.date_range("2026-01-01", periods=72, freq="h", tz="UTC")
    forecasts, _ = _frames(index)
    errors = pd.DataFrame(0.0, index=index, columns=ALIASES)
    errors.loc[index[0], ALIASES[0]] = 123.0

    features = correction.build_residual_load_input_features(
        forecasts,
        error_history=errors,
        error_lags_hours=(48,),
        error_rolling_windows_hours=(24,),
    )

    assert features.index.equals(index)
    assert features.index.tz is not None
    assert features.loc[index[48], f"{ALIASES[0]}__error_lag_48h"] == 123.0
    assert features.loc[index[48], f"{ALIASES[0]}__error_lag_48h_available"] == 1.0
    with pytest.raises(correction.ResidualLoadInputCorrectionError, match="UTC|fuseau"):
        correction.build_residual_load_input_features(
            forecasts.set_axis(index.tz_localize(None))
        )
    with pytest.raises(correction.ResidualLoadInputCorrectionError, match="Aliases"):
        correction.build_residual_load_input_features(
            forecasts.drop(columns=ALIASES[-1])
        )
    with pytest.raises(ValueError, match="48"):
        correction.build_residual_load_input_features(
            forecasts,
            error_lags_hours=(24,),
        )


def test_correct_live_uses_d_minus_one_forecast_context_but_never_its_labels() -> None:
    delivery_day = date(2026, 4, 2)
    forecasts, observations = _history_through_day(delivery_day)
    local_dates = forecasts.index.tz_convert("Europe/Paris").date
    d_minus_one = delivery_day - timedelta(days=1)
    observations.loc[local_dates == d_minus_one, :] = 1_000_001.0
    observations.loc[local_dates == delivery_day, :] = 2_000_001.0

    result = correction.ResidualLoadInputCorrector(
        max_abs_correction_gw=3.0,
        corrector_factory=_factory,
    ).correct_live(
        forecasts,
        observations,
        delivery_day=delivery_day,
    )

    expected_index = local_delivery_day_index(delivery_day, timezone="Europe/Paris")
    assert result.raw.index.equals(expected_index)
    assert len(_FakeCorrector.created) == len(ALIASES)
    expected_label_end = correction.conservative_label_end_utc(delivery_day)
    for model in _FakeCorrector.created:
        assert model.fit_index is not None
        assert model.fit_index.max() == expected_label_end
        assert model.fit_target is not None
        assert model.fit_target.max() < 1_000_000.0
        assert model.predict_features is not None

    # The first D ramp is computed against the immediately preceding D-1
    # forecast hour, proving D-1 remains raw context despite the label embargo.
    first = expected_index[0]
    previous = first - pd.Timedelta(hours=1)
    expected_ramp = forecasts.loc[first, ALIASES[0]] - forecasts.loc[previous, ALIASES[0]]
    assert _FakeCorrector.created[0].predict_features.loc[
        first, f"{ALIASES[0]}__change_1h"
    ] == pytest.approx(expected_ramp)


def test_prequential_generation_refits_each_day_without_future_labels() -> None:
    start_day = date(2026, 2, 5)
    end_day = start_day + timedelta(days=1)
    forecasts, observations = _history_through_day(end_day, history_days=8)
    observations.loc[:, :] = forecasts + 1.0

    result = correction.generate_prequential_residual_load_corrections(
        forecasts,
        observations,
        start_day=start_day,
        end_day=end_day,
        max_abs_correction_gw=4.0,
        corrector_factory=_factory,
    )

    expected = local_delivery_day_index(start_day, timezone="Europe/Paris").append(
        local_delivery_day_index(end_day, timezone="Europe/Paris")
    )
    assert result.raw.index.equals(expected)
    assert len(_FakeCorrector.created) == 2 * len(ALIASES)
    expected_ends = [
        correction.conservative_label_end_utc(day)
        for day in (start_day, end_day)
        for _ in ALIASES
    ]
    assert [model.fit_index.max() for model in _FakeCorrector.created] == expected_ends
    assert result.diagnostics["mode"] == "prequential"
    assert result.diagnostics["block_count"] == 2


@pytest.mark.parametrize(
    ("delivery_day", "expected_hours"),
    [(date(2025, 3, 30), 23), (date(2025, 10, 26), 25)],
)
def test_complete_dst_days_are_preserved(
    delivery_day: date,
    expected_hours: int,
) -> None:
    forecasts, observations = _history_through_day(delivery_day)
    result = correction.ResidualLoadInputCorrector(
        corrector_factory=_factory,
    ).correct_live(forecasts, observations, delivery_day=delivery_day)
    expected = local_delivery_day_index(delivery_day, timezone="Europe/Paris")

    assert len(result.raw) == expected_hours
    assert result.raw.index.equals(expected)
    assert result.corrected.index.equals(expected)
    assert not result.corrected.isna().any().any()


def test_result_exposes_raw_clipped_correction_corrected_and_components() -> None:
    delivery_day = date(2026, 2, 5)
    forecasts, observations = _history_through_day(delivery_day)
    result = correction.ResidualLoadInputCorrector(
        max_abs_correction_gw=3.0,
        corrector_factory=_factory,
    ).correct_live(forecasts, observations, delivery_day=delivery_day)

    np.testing.assert_allclose(result.correction.to_numpy(), 3.0)
    pd.testing.assert_frame_equal(result.corrected, result.raw + 3.0)
    assert set(result.component_corrections) == {"cat_v1", "hgb31"}
    np.testing.assert_allclose(
        result.component_corrections["cat_v1"].to_numpy(), 4.0
    )
    np.testing.assert_allclose(
        result.component_corrections["hgb31"].to_numpy(), 20.0
    )
    flat = result.to_frame()
    for alias in ALIASES:
        assert {
            f"{alias}__raw",
            f"{alias}__correction",
            f"{alias}__corrected",
            f"{alias}__component_cat_v1",
            f"{alias}__component_hgb31",
        }.issubset(flat.columns)


def test_nan_forecasts_are_never_filled_and_masks_are_preserved_per_alias() -> None:
    delivery_day = date(2026, 2, 5)
    forecasts, observations = _history_through_day(delivery_day, history_days=8)
    prediction_index = local_delivery_day_index(
        delivery_day,
        timezone="Europe/Paris",
    )
    label_end = correction.conservative_label_end_utc(delivery_day)

    excluded_training_rows: dict[str, set[pd.Timestamp]] = {}
    missing_prediction_rows: dict[str, pd.Timestamp] = {}
    for offset, alias in enumerate(ALIASES):
        raw_missing = label_end - pd.Timedelta(hours=20 + 2 * offset)
        observation_missing = raw_missing + pd.Timedelta(hours=1)
        forecasts.loc[raw_missing, alias] = np.nan
        observations.loc[observation_missing, alias] = np.nan
        excluded_training_rows[alias] = {raw_missing, observation_missing}

        prediction_missing = prediction_index[2 + offset]
        forecasts.loc[prediction_missing, alias] = np.nan
        missing_prediction_rows[alias] = prediction_missing
    # A country may be unavailable for the complete delivery block.  Its own
    # corrector must not be called at all, while other aliases remain usable.
    forecasts.loc[prediction_index, ALIASES[-1]] = np.nan

    engineered = correction.build_residual_load_input_features(forecasts)
    first_alias_missing = missing_prediction_rows[ALIASES[0]]
    assert np.isnan(engineered.loc[first_alias_missing, ALIASES[0]])
    assert np.isnan(
        engineered.loc[
            first_alias_missing + pd.Timedelta(hours=1),
            f"{ALIASES[0]}__change_1h",
        ]
    )

    result = correction.ResidualLoadInputCorrector(
        max_abs_correction_gw=3.0,
        corrector_factory=_factory,
    ).correct_live(
        forecasts,
        observations,
        delivery_day=delivery_day,
    )

    for alias, model in zip(ALIASES, _FakeCorrector.created):
        assert model.fit_index is not None
        assert excluded_training_rows[alias].isdisjoint(model.fit_index)
        expected_prediction_index = prediction_index[
            forecasts.loc[prediction_index, alias].notna().to_numpy()
        ]
        if len(expected_prediction_index):
            assert model.predict_features is not None
            assert model.predict_features.index.equals(expected_prediction_index)
        else:
            assert model.predict_features is None

    raw_mask = result.raw.isna()
    assert result.correction.isna().equals(raw_mask)
    assert result.corrected.isna().equals(raw_mask)
    for component in result.component_corrections.values():
        assert component.isna().equals(raw_mask)


def test_cold_start_alias_is_raw_passthrough_until_labels_are_available() -> None:
    delivery_day = date(2025, 8, 12)
    forecasts, observations = _history_through_day(delivery_day, history_days=8)
    cold_alias = "nl_residual_load_fcst"
    observations.loc[:, cold_alias] = np.nan

    result = correction.ResidualLoadInputCorrector(
        max_abs_correction_gw=3.0,
        corrector_factory=_factory,
        cold_start_policy="raw_passthrough",
        minimum_training_rows=48,
    ).correct_live(
        forecasts,
        observations,
        delivery_day=delivery_day,
    )

    np.testing.assert_allclose(result.correction[cold_alias].to_numpy(), 0.0)
    pd.testing.assert_series_equal(
        result.corrected[cold_alias],
        result.raw[cold_alias],
    )
    assert len(_FakeCorrector.created) == len(ALIASES) - 1
    assert result.diagnostics["models"][cold_alias]["status"] == (
        "cold_start_raw_passthrough"
    )
    assert cold_alias in result.diagnostics["cold_start_aliases"]
    for component in result.component_corrections.values():
        np.testing.assert_allclose(component[cold_alias].to_numpy(), 0.0)
