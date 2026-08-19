from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from chronos2_hourly.models import (
    HourlyCatBoost,
    HourlyLEAR,
    LeakageRiskError,
    NonNegativeOOFEnsemble,
    OptionalDependencyError,
    RollingMedianCalibrator,
    purged_expanding_splits,
)
from chronos2_hourly.models.base import make_prediction_frame


def synthetic_hourly_frame(n_rows: int = 360) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2025-09-20", periods=n_rows, freq="h", tz="UTC")
    local_hour = index.tz_convert("Europe/Paris").hour
    x1 = np.linspace(-2.0, 2.0, n_rows)
    x2 = np.sin(np.arange(n_rows) * 2.0 * np.pi / 24.0)
    frame = pd.DataFrame(
        {
            "delivery_hour": local_hour,
            "residual_load": x1,
            "renewable_ramp": x2,
        },
        index=index,
    )
    target = pd.Series(
        55.0 + 8.0 * x1 - 4.0 * x2 + 0.15 * (local_hour - 12.0),
        index=index,
    )
    return frame, target


class HourlyLEARTests(unittest.TestCase):
    def test_fit_predict_contract_and_dst_duplicate_hour(self) -> None:
        frame, target = synthetic_hourly_frame()
        model = HourlyLEAR(
            feature_columns=[
                "delivery_hour",
                "residual_load",
                "renewable_ramp",
            ],
            min_samples_per_hour=8,
            alpha=0.001,
        ).fit(frame, target)

        # 00:00 UTC and 01:00 UTC are both local 02:00 on the autumn DST day.
        dst_index = pd.date_range(
            "2025-10-26 00:00", periods=4, freq="h", tz="UTC"
        )
        dst = pd.DataFrame(
            {
                "delivery_hour": dst_index.tz_convert("Europe/Paris").hour,
                "residual_load": [0.1, 0.2, 0.3, 0.4],
                "renewable_ramp": [0.0, 0.1, 0.0, -0.1],
            },
            index=dst_index,
        )
        prediction = model.predict(dst)

        self.assertEqual(prediction.columns.tolist(), ["q10", "q50", "q90"])
        self.assertTrue(prediction.index.equals(dst_index))
        self.assertTrue(np.isfinite(prediction.to_numpy()).all())
        self.assertTrue((prediction["q10"] <= prediction["q50"]).all())
        self.assertTrue((prediction["q50"] <= prediction["q90"]).all())
        self.assertEqual(dst["delivery_hour"].tolist()[:2], [2, 2])

    def test_feature_schema_is_frozen(self) -> None:
        frame, target = synthetic_hourly_frame(120)
        model = HourlyLEAR(min_samples_per_hour=1000).fit(frame, target)
        with self.assertRaisesRegex(ValueError, "Features absentes"):
            model.predict(frame.drop(columns="residual_load"))


class HourlyCatBoostTests(unittest.TestCase):
    def test_sklearn_fallback_is_fully_testable(self) -> None:
        frame, target = synthetic_hourly_frame(180)
        model = HourlyCatBoost(
            backend="sklearn",
            min_samples_per_hour=10_000,
            iterations=20,
            depth=3,
            min_samples_leaf=5,
        ).fit(frame, target)
        prediction = model.predict(frame.tail(12))

        self.assertEqual(model.backend_, "sklearn")
        self.assertEqual(prediction.shape, (12, 3))
        self.assertTrue(np.isfinite(prediction.to_numpy()).all())
        self.assertTrue((prediction["q10"] <= prediction["q50"]).all())
        self.assertTrue((prediction["q50"] <= prediction["q90"]).all())

    def test_explicit_catboost_backend_has_actionable_error(self) -> None:
        frame, target = synthetic_hourly_frame(60)
        with patch(
            "chronos2_hourly.models.catboost_hourly.catboost_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(OptionalDependencyError, "pip install catboost"):
                HourlyCatBoost(backend="catboost").fit(frame, target)


class EnsembleTests(unittest.TestCase):
    def _oof_frame(self) -> tuple[pd.DataFrame, pd.Series]:
        n_rows = 240
        index = pd.date_range("2025-01-01", periods=n_rows, freq="h", tz="UTC")
        learner = 40.0 + np.linspace(-12.0, 15.0, n_rows)
        chronos = learner + 4.0 * np.sin(np.arange(n_rows) / 9.0)
        weak = 20.0 + 0.2 * learner
        target = 0.75 * learner + 0.25 * chronos
        frame = pd.DataFrame(index=index)
        for name, median, width in (
            ("lear", learner, 8.0),
            ("chronos2", chronos, 10.0),
            ("weak", weak, 20.0),
        ):
            frame[f"{name}__q10"] = median - width
            frame[f"{name}__q50"] = median
            frame[f"{name}__q90"] = median + width
        return frame, pd.Series(target, index=index)

    def test_refuses_predictions_not_declared_oof(self) -> None:
        frame, target = self._oof_frame()
        with self.assertRaises(LeakageRiskError):
            NonNegativeOOFEnsemble().fit(frame, target)

    def test_nonnegative_oof_weights_include_external_chronos(self) -> None:
        frame, target = self._oof_frame()
        origins = frame.index - pd.Timedelta(hours=12)
        model = NonNegativeOOFEnsemble().fit(
            frame,
            target,
            is_oof=True,
            prediction_origin=origins,
            delivery_start=frame.index,
        )
        prediction = model.predict(frame.tail(24))

        self.assertIn("chronos2", model.weights_.index)
        self.assertAlmostEqual(float(model.weights_.sum()), 1.0, places=8)
        self.assertTrue((model.weights_ >= 0).all())
        self.assertLess(float(model.weights_["weak"]), 0.05)
        self.assertEqual(prediction.columns.tolist(), ["q10", "q50", "q90"])
        self.assertTrue((prediction["q10"] <= prediction["q50"]).all())
        self.assertTrue((prediction["q50"] <= prediction["q90"]).all())

    def test_purged_splits_are_strictly_chronological(self) -> None:
        splits = purged_expanding_splits(
            160,
            n_splits=3,
            min_train_size=70,
            test_size=20,
            gap=4,
        )
        for train, test in splits:
            self.assertLess(int(train.max()), int(test.min()) - 4 + 1)
            self.assertEqual(len(set(train).intersection(test)), 0)


class PredictionFrameTests(unittest.TestCase):
    def test_non_crossing_repair_preserves_dedicated_median(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        raw = np.asarray(
            [
                [80.0, 50.0, 40.0],
                [60.0, 50.0, 70.0],
            ]
        )

        repaired = make_prediction_frame(raw, index, (0.1, 0.5, 0.9))

        np.testing.assert_allclose(repaired["q50"].to_numpy(), [50.0, 50.0])
        self.assertTrue((repaired["q10"] <= repaired["q50"]).all())
        self.assertTrue((repaired["q50"] <= repaired["q90"]).all())


class CalibrationTests(unittest.TestCase):
    def test_fit_transform_never_uses_current_or_future_residual(self) -> None:
        index = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        prediction = pd.DataFrame(
            {"q10": 5.0, "q50": 10.0, "q90": 15.0}, index=index
        )
        # A shock appears on row 3.  Its own correction must still be based on
        # rows 1-2 only, so q50 must remain equal to 10.
        target = pd.Series([10.0, 10.0, 110.0, 10.0, 10.0], index=index)
        calibrator = RollingMedianCalibrator(
            window=3,
            min_periods=1,
            by_delivery_hour=False,
            max_abs_correction=None,
        )
        calibrated = calibrator.fit_transform(prediction, target)

        self.assertEqual(float(calibrated.iloc[0]["q50"]), 10.0)
        self.assertEqual(float(calibrated.iloc[2]["q50"]), 10.0)
        self.assertEqual(float(calibrated.iloc[3]["q50"]), 10.0)
        self.assertTrue((calibrated["q10"] <= calibrated["q50"]).all())
        self.assertTrue((calibrated["q50"] <= calibrated["q90"]).all())


if __name__ == "__main__":
    unittest.main()
