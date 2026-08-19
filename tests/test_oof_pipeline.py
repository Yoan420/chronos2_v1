from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.models import (
    HourlyCatBoost,
    HourlyLEAR,
    LeakageRiskError,
)
from chronos2_hourly.oof_pipeline import (
    HourlyOOFConfig,
    HourlyOOFPipeline,
    purged_daily_expanding_splits,
)


def feature_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert("Europe/Paris")
    position = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "delivery_hour": local.hour,
            "residual_load": 40.0 + 3.0 * np.sin(position / 13.0),
            "firm_margin": 8.0 + 2.0 * np.cos(position / 17.0),
        },
        index=index,
    )


def price_target(frame: pd.DataFrame) -> pd.Series:
    values = (
        35.0
        + 1.2 * frame["residual_load"].to_numpy()
        - 0.8 * frame["firm_margin"].to_numpy()
        + 0.15 * frame["delivery_hour"].to_numpy()
    )
    return pd.Series(values, index=frame.index)


def chronos_frame(target: pd.Series) -> pd.DataFrame:
    median = target.to_numpy() + 1.5 * np.sin(np.arange(len(target)) / 8.0)
    return pd.DataFrame(
        {"q10": median - 9.0, "q50": median, "q90": median + 9.0},
        index=target.index,
    )


def origin_series(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index - pd.Timedelta(hours=12), index=index)


def small_pipeline() -> HourlyOOFPipeline:
    config = HourlyOOFConfig(
        n_splits=3,
        min_train_size=96,
        test_size=40,
        gap=4,
        evaluation_days=None,
        ensemble_minimum_rows=40,
    )
    return HourlyOOFPipeline(
        config=config,
        lear_factory=lambda: HourlyLEAR(
            min_samples_per_hour=10_000,
            alpha=0.001,
        ),
        catboost_factory=lambda: HourlyCatBoost(
            backend="sklearn",
            min_samples_per_hour=10_000,
            iterations=12,
            depth=3,
            min_samples_leaf=5,
        ),
    )


class _SimpleQuantileForecaster:
    """Fast deterministic expert used to isolate orchestration tests."""

    def __init__(self, bias: float) -> None:
        self.bias = float(bias)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_SimpleQuantileForecaster":
        observed = pd.to_numeric(y, errors="coerce").dropna()
        self.location_ = float(observed.median()) + self.bias
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        signal = 0.2 * pd.to_numeric(X["residual_load"]).to_numpy(dtype=float)
        median = self.location_ + signal
        return pd.DataFrame(
            {"q10": median - 8.0, "q50": median, "q90": median + 8.0},
            index=X.index,
        )


class _MedianResidualCorrector:
    """Tiny deterministic corrector used to test orchestration leakage."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_predictions: pd.DataFrame,
        expert_predictions: pd.DataFrame | None = None,
    ) -> "_MedianResidualCorrector":
        self.shift_ = float((y - base_predictions["q50"]).median())
        self.backend_ = "test"
        self.feature_columns_ = tuple(X.columns)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        base_predictions: pd.DataFrame,
        expert_predictions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return base_predictions.loc[:, ["q10", "q50", "q90"]] + self.shift_


def complete_day_index(start: str, n_days: int) -> pd.DatetimeIndex:
    days = pd.date_range(start, periods=n_days, freq="D")
    parts = [local_delivery_day_index(day) for day in days]
    return parts[0].append(parts[1:])


class HourlyOOFPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = pd.date_range(
            "2025-01-01", periods=240, freq="h", tz="UTC"
        )
        cls.features = feature_frame(cls.index)
        cls.target = price_target(cls.features)

    def test_fit_builds_complete_oof_diagnostics_and_final_experts(self) -> None:
        model = small_pipeline().fit(
            self.features,
            self.target,
            chronos_oof=chronos_frame(self.target),
            chronos_origin=origin_series(self.index),
            chronos_is_oof=True,
        )
        result = model.training_result_

        self.assertEqual(result.diagnostics["n_oof_expected"], 120)
        self.assertEqual(result.diagnostics["n_oof_common"], 120)
        self.assertEqual(result.fold_id.notna().sum(), 120)
        self.assertEqual(
            set(result.metrics.index), {"lear", "catboost", "chronos2", "ensemble"}
        )
        self.assertTrue(np.isfinite(result.metrics["mae"]).all())
        self.assertAlmostEqual(float(result.ensemble_weights.sum()), 1.0, places=8)
        self.assertTrue((result.ensemble_weights >= 0).all())
        self.assertTrue(hasattr(model.lear_, "is_fitted_"))
        self.assertTrue(hasattr(model.catboost_, "is_fitted_"))

    def test_chronos_oof_contract_is_mandatory(self) -> None:
        chronos = chronos_frame(self.target)
        origins = origin_series(self.index)
        with self.assertRaises(LeakageRiskError):
            small_pipeline().fit(
                self.features,
                self.target,
                chronos_oof=chronos,
                chronos_origin=origins,
                chronos_is_oof=False,
            )

        shifted = chronos.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=1)
        with self.assertRaisesRegex(LeakageRiskError, "strictement aligné"):
            small_pipeline().fit(
                self.features,
                self.target,
                chronos_oof=shifted,
                chronos_origin=origins,
                chronos_is_oof=True,
            )

        invalid_origins = origins.copy()
        # This position belongs to the first test fold.
        invalid_origins.iloc[100] = self.index[100]
        with self.assertRaisesRegex(LeakageRiskError, "strictement avant"):
            small_pipeline().fit(
                self.features,
                self.target,
                chronos_oof=chronos,
                chronos_origin=invalid_origins,
                chronos_is_oof=True,
            )

    def test_predict_preserves_spring_and_autumn_physical_hours(self) -> None:
        model = small_pipeline().fit(
            self.features,
            self.target,
            chronos_oof=chronos_frame(self.target),
            chronos_origin=origin_series(self.index),
            chronos_is_oof=True,
        )
        for delivery_date, expected_hours in (
            ("2026-03-29", 23),
            ("2026-10-25", 25),
        ):
            future_index = local_delivery_day_index(delivery_date)
            future_features = feature_frame(future_index)
            proxy_target = price_target(future_features)
            forecast = model.predict(
                future_features,
                chronos_future=chronos_frame(proxy_target),
                chronos_origin=origin_series(future_index),
            )

            self.assertEqual(len(forecast.predictions), expected_hours)
            self.assertEqual(
                forecast.diagnostics["hours_in_local_day"], expected_hours
            )
            self.assertTrue(forecast.predictions.index.tz is not None)
            self.assertEqual(str(forecast.predictions.index.tz), "UTC")
            self.assertTrue(np.isfinite(forecast.predictions.to_numpy()).all())
            self.assertEqual(
                forecast.delivery_metadata["hours_in_local_day"].unique().tolist(),
                [expected_hours],
            )

        past_index = local_delivery_day_index("2024-10-27")
        past_features = feature_frame(past_index)
        past_proxy = price_target(past_features)
        with self.assertRaisesRegex(LeakageRiskError, "strictement après"):
            model.predict(
                past_features,
                chronos_future=chronos_frame(past_proxy),
                chronos_origin=origin_series(past_index),
            )

    def test_future_day_must_be_complete(self) -> None:
        model = small_pipeline().fit(
            self.features,
            self.target,
            chronos_oof=chronos_frame(self.target),
            chronos_origin=origin_series(self.index),
            chronos_is_oof=True,
        )
        future_index = local_delivery_day_index("2026-03-29")[:-1]
        future_features = feature_frame(future_index)
        proxy = price_target(future_features)
        with self.assertRaisesRegex(ValueError, "exactement toutes les heures"):
            model.predict(
                future_features,
                chronos_future=chronos_frame(proxy),
                chronos_origin=origin_series(future_index),
            )


class DailySplitAndSealedHoldoutTests(unittest.TestCase):
    def test_daily_splits_use_complete_midnight_boundaries_and_day_purge(self) -> None:
        # Includes the 23-hour Europe/Paris spring-transition day.
        index = complete_day_index("2026-03-15", 30)
        splits = purged_daily_expanding_splits(
            index,
            n_splits=3,
            min_train_days=14,
            test_days=5,
            gap_days=1,
        )
        local = index.tz_convert("Europe/Paris")

        for train, test in splits:
            train_dates = set(local[train].date)
            test_dates = set(local[test].date)
            self.assertFalse(train_dates.intersection(test_dates))
            self.assertEqual(local[test[0]].hour, 0)
            self.assertEqual(local[test[-1]].hour, 23)
            self.assertEqual(
                (min(test_dates) - max(train_dates)).days,
                2,
            )
            for delivery_date in test_dates:
                observed = index[np.asarray(local.date == delivery_date)]
                expected = local_delivery_day_index(delivery_date)
                self.assertTrue(np.array_equal(observed.asi8, expected.asi8))

        with self.assertRaisesRegex(ValueError, "journées complètes"):
            purged_daily_expanding_splits(
                index[1:],
                n_splits=2,
                min_train_days=10,
                test_days=5,
                gap_days=1,
            )

    def test_holdout_targets_cannot_change_ensemble_weights(self) -> None:
        index = complete_day_index("2025-01-01", 30)
        features = feature_frame(index)
        target = price_target(features)
        chronos = chronos_frame(target)
        origins = origin_series(index)
        config = HourlyOOFConfig(
            n_splits=3,
            split_on_delivery_days=True,
            min_train_days=14,
            test_days=5,
            gap_days=1,
            evaluation_days=10,
            ensemble_minimum_rows=48,
        )

        def new_pipeline() -> HourlyOOFPipeline:
            return HourlyOOFPipeline(
                config=config,
                lear_factory=lambda: _SimpleQuantileForecaster(-2.0),
                catboost_factory=lambda: _SimpleQuantileForecaster(3.0),
            )

        original = new_pipeline().fit(
            features,
            target,
            chronos_oof=chronos,
            chronos_origin=origins,
            chronos_is_oof=True,
        )
        changed_target = target.copy()
        evaluation_dates = set(index.tz_convert("Europe/Paris").date[-10 * 24 :])
        evaluation_mask = np.asarray(
            pd.Index(index.tz_convert("Europe/Paris").date).isin(evaluation_dates)
        )
        changed_target.iloc[evaluation_mask] += 1_000.0
        changed = new_pipeline().fit(
            features,
            changed_target,
            chronos_oof=chronos,
            chronos_origin=origins,
            chronos_is_oof=True,
        )

        np.testing.assert_allclose(
            original.training_result_.ensemble_weights.to_numpy(),
            changed.training_result_.ensemble_weights.to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(
            original.training_result_.diagnostics["n_ensemble_fit"], 5 * 24
        )
        self.assertEqual(
            original.training_result_.diagnostics["n_evaluation"], 10 * 24
        )
        self.assertEqual(
            original.training_result_.diagnostics["metric_scope"],
            "sealed_final_10_delivery_days",
        )
        self.assertTrue(
            (original.training_result_.metrics["n_expected"] == 10 * 24).all()
        )
        self.assertFalse(
            np.allclose(
                original.training_result_.metrics["mae"],
                changed.training_result_.metrics["mae"],
            )
        )

    def test_residual_holdout_predictions_never_use_holdout_targets(self) -> None:
        index = complete_day_index("2025-01-01", 30)
        features = feature_frame(index)
        target = price_target(features)
        chronos = chronos_frame(target)
        origins = origin_series(index)
        config = HourlyOOFConfig(
            n_splits=3,
            split_on_delivery_days=True,
            min_train_days=14,
            test_days=5,
            gap_days=1,
            evaluation_days=10,
            ensemble_minimum_rows=48,
        )

        def new_pipeline() -> HourlyOOFPipeline:
            return HourlyOOFPipeline(
                config=config,
                lear_factory=lambda: _SimpleQuantileForecaster(-2.0),
                catboost_factory=lambda: _SimpleQuantileForecaster(3.0),
                residual_corrector_factory=_MedianResidualCorrector,
                residual_base_model="chronos2",
            )

        original = new_pipeline().fit(
            features,
            target,
            chronos_oof=chronos,
            chronos_origin=origins,
            chronos_is_oof=True,
        )
        changed_target = target.copy()
        evaluation_days = set(index.tz_convert("Europe/Paris").date[-10 * 24 :])
        evaluation_mask = np.asarray(
            pd.Index(index.tz_convert("Europe/Paris").date).isin(evaluation_days)
        )
        changed_target.iloc[evaluation_mask] += 1_000.0
        changed = new_pipeline().fit(
            features,
            changed_target,
            chronos_oof=chronos,
            chronos_origin=origins,
            chronos_is_oof=True,
        )

        original_residual = original.training_result_.residual_oof_predictions
        changed_residual = changed.training_result_.residual_oof_predictions
        self.assertIsNotNone(original_residual)
        self.assertIsNotNone(changed_residual)
        np.testing.assert_allclose(
            original_residual.loc[evaluation_mask].to_numpy(),
            changed_residual.loc[evaluation_mask].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertIn("residual_corrected", original.training_result_.metrics.index)
        self.assertNotEqual(
            float(original.training_result_.metrics.loc["residual_corrected", "mae"]),
            float(changed.training_result_.metrics.loc["residual_corrected", "mae"]),
        )


if __name__ == "__main__":
    unittest.main()
