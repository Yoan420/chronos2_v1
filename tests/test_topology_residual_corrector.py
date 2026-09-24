from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from chronos2_hourly.models.topology_residual_corrector import (
    BASE_FEATURE_COLUMNS,
    TopologyResidualCorrectionError,
    TopologyResidualCorrector,
)
from chronos2_hourly.topology_context import (
    TOPOLOGY_CONTEXT_COLUMNS,
    build_topology_context,
)


def _problem(
    periods: int = 576,
    *,
    radius: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    step = np.arange(periods, dtype=float)
    daily = np.sin(step * 2.0 * np.pi / 24.0)
    weekly = np.cos(step * 2.0 * np.pi / (24.0 * 7.0))
    residual = pd.DataFrame(
        {
            "FR": 48.0 + 8.0 * daily + 2.0 * weekly,
            "DE": 53.0 + 4.0 * daily - 3.0 * weekly,
            "BE": 46.0 + 6.0 * daily + 4.0 * weekly,
            "NL": 44.0 + 3.0 * daily - 2.0 * weekly,
            "ES": 38.0 + 10.0 * daily + weekly,
        },
        index=index,
    )
    prices = pd.DataFrame(
        {
            "FR": 52.0 + 10.0 * daily + 2.0 * weekly,
            "DE": 56.0 + 9.0 * daily - weekly,
            "BE": 54.0 + 8.0 * daily + weekly,
            "NL": 51.0 + 7.0 * daily - 2.0 * weekly,
            "ES": 49.0 + 11.0 * daily + 3.0 * weekly,
        },
        index=index,
    )
    if radius == 0:
        residual_input = residual[["FR"]]
        price_input = prices[["FR"]]
    else:
        residual_input = residual
        price_input = prices
    context = build_topology_context(
        residual_input,
        price_input,
        target_zone="FR",
        radius=radius,
    )
    base_median = 45.0 + 0.24 * residual["FR"].to_numpy()
    base = pd.DataFrame(
        {
            "q10": base_median - 7.0,
            "q50": base_median,
            "q90": base_median + 9.0,
        },
        index=index,
    )
    pool = context["topology__residual_load__pool_mean"].to_numpy()
    correction = 0.55 * (pool - residual["FR"].to_numpy()) + 2.5 * daily
    target = pd.Series(base_median + correction, index=index, name="actual")
    return context, base, target


def _model(**overrides: object) -> TopologyResidualCorrector:
    options: dict[str, object] = {
        "learning_rate": 0.08,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 16,
        "l2_regularization": 2.0,
        "min_training_rows": 96,
    }
    options.update(overrides)
    return TopologyResidualCorrector(**options)


class TopologyResidualCorrectorTests(unittest.TestCase):
    def test_hgb_absolute_error_improves_residual_and_is_deterministic(self) -> None:
        context, base, target = _problem()
        train = slice(0, 480)
        test = slice(480, None)
        first = _model().fit(context.iloc[train], target.iloc[train], base.iloc[train])
        second = _model().fit(context.iloc[train], target.iloc[train], base.iloc[train])

        first_prediction = first.predict(context.iloc[test], base.iloc[test])
        second_prediction = second.predict(context.iloc[test], base.iloc[test])
        baseline_mae = float(
            np.mean(np.abs(target.iloc[test].to_numpy() - base.iloc[test]["q50"].to_numpy()))
        )
        corrected_mae = float(
            np.mean(
                np.abs(
                    target.iloc[test].to_numpy()
                    - first_prediction["q50"].to_numpy()
                )
            )
        )

        self.assertIsInstance(first.estimator_, HistGradientBoostingRegressor)
        self.assertEqual(first.estimator_.loss, "absolute_error")
        self.assertFalse(first.estimator_.early_stopping)
        self.assertEqual(first.estimator_.random_state, 120)
        self.assertLess(corrected_mae, baseline_mae * 0.45)
        np.testing.assert_allclose(first_prediction, second_prediction, rtol=0, atol=0)
        self.assertEqual(first.hyperparameter_sha256(), second.hyperparameter_sha256())

    def test_common_shift_preserves_width_order_and_clip(self) -> None:
        context, base, target = _problem()
        model = _model(correction_clip=0.75).fit(
            context.iloc[:480],
            target.iloc[:480],
            base.iloc[:480],
        )
        future_context = context.iloc[480:]
        future_base = base.iloc[480:]

        correction = model.predict_correction(future_context, future_base)
        result = model.predict(future_context, future_base)

        self.assertLessEqual(float(correction.abs().max()), 0.75)
        np.testing.assert_allclose(result["q50"] - result["q10"], 7.0)
        np.testing.assert_allclose(result["q90"] - result["q50"], 9.0)
        self.assertTrue((result["q10"] <= result["q50"]).all())
        self.assertTrue((result["q50"] <= result["q90"]).all())
        np.testing.assert_allclose(
            result["q50"] - future_base["q50"],
            correction,
        )

    def test_native_nan_path_has_no_imputer_and_is_audited(self) -> None:
        context, base, target = _problem()
        context = context.copy()
        context.iloc[100:110, context.columns.get_loc("topology__residual_load__local")] = np.nan
        context.iloc[200, :10] = np.nan
        model = _model().fit(
            context.iloc[:480],
            target.iloc[:480],
            base.iloc[:480],
        )
        prediction = model.predict(context.iloc[480:], base.iloc[480:])
        audit = model.audit_metadata()

        self.assertIsInstance(model.estimator_, HistGradientBoostingRegressor)
        self.assertTrue(np.isfinite(prediction.to_numpy()).all())
        self.assertEqual(audit["missing_policy"], "native_nan_no_interpolation")
        self.assertGreater(
            audit["fit_missing_values_by_feature"]["topology__residual_load__local"],
            0,
        )
        self.assertGreater(
            audit["fit_missing_values_by_feature"]["topology__price_da_lag24h__local"],
            0,
        )

    def test_base_level_and_widths_are_internal_estimator_features(self) -> None:
        context, base, target = _problem()
        model = _model().fit(
            context.iloc[:480],
            target.iloc[:480],
            base.iloc[:480],
        )
        audit = model.audit_metadata()

        self.assertEqual(tuple(context.columns), TOPOLOGY_CONTEXT_COLUMNS)
        self.assertEqual(model.context_feature_names_in_, TOPOLOGY_CONTEXT_COLUMNS)
        self.assertEqual(model.feature_names_in_[-3:], BASE_FEATURE_COLUMNS)
        self.assertEqual(len(model.feature_names_in_), 25)
        self.assertEqual(audit["feature_names"][-3:], list(BASE_FEATURE_COLUMNS))

    def test_autonomous_or_mkonline_blended_base_is_accepted_as_object(self) -> None:
        context, base, target = _problem()
        blended = base.copy()
        blended[["q10", "q50", "q90"]] += 0.3
        blended.attrs["forecast_source"] = "MKOnline blend"
        model = _model().fit(
            context.iloc[:480],
            target.iloc[:480],
            blended.iloc[:480],
        )

        result = model.predict(context.iloc[480:], blended.iloc[480:])

        self.assertEqual(result.shape, (96, 3))
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertNotIn("mkonline", " ".join(model.feature_names_in_).lower())

    def test_storm_and_mkonline_features_are_rejected(self) -> None:
        context, base, target = _problem()
        for column in ("storm_q50", "mk_online_signal"):
            with self.subTest(column=column):
                bad = context.iloc[:480].copy()
                bad[column] = 1.0
                with self.assertRaisesRegex(
                    TopologyResidualCorrectionError,
                    "Storm/MKOnline",
                ):
                    _model().fit(bad, target.iloc[:480], base.iloc[:480])

    def test_radius_drift_is_rejected_even_with_same_column_schema(self) -> None:
        context_zero, base, target = _problem(radius=0)
        context_one, _, _ = _problem(radius=1)
        model = _model().fit(
            context_zero.iloc[:480],
            target.iloc[:480],
            base.iloc[:480],
        )
        with self.assertRaisesRegex(
            TopologyResidualCorrectionError,
            "diffère du contexte",
        ):
            model.predict(context_one.iloc[480:], base.iloc[480:])

    def test_missing_metadata_schema_drift_and_crossed_base_fail_closed(self) -> None:
        context, base, target = _problem()
        no_metadata = context.iloc[:480].copy()
        no_metadata.attrs.clear()
        with self.assertRaisesRegex(
            TopologyResidualCorrectionError,
            "build_topology_context",
        ):
            _model().fit(no_metadata, target.iloc[:480], base.iloc[:480])

        reordered = context.iloc[:480, ::-1].copy()
        with self.assertRaisesRegex(
            TopologyResidualCorrectionError,
            "ordre_exact_requis",
        ):
            _model().fit(reordered, target.iloc[:480], base.iloc[:480])

        tampered = context.iloc[:480].copy()
        tampered.attrs["topology_context"] = dict(
            tampered.attrs["topology_context"]
        )
        tampered.attrs["topology_context"]["included_zones"] = ["FR", "NL"]
        with self.assertRaisesRegex(
            TopologyResidualCorrectionError,
            "masque topologique",
        ):
            _model().fit(tampered, target.iloc[:480], base.iloc[:480])

        crossed = base.iloc[:480].copy()
        crossed.iloc[0, crossed.columns.get_loc("q10")] = crossed.iloc[0]["q90"] + 1.0
        with self.assertRaisesRegex(TopologyResidualCorrectionError, "se croisent"):
            _model().fit(context.iloc[:480], target.iloc[:480], crossed)

    def test_no_implicit_identity_fallback(self) -> None:
        context, base, target = _problem(periods=120)
        model = _model(min_training_rows=100)
        with self.assertRaisesRegex(RuntimeError, "entraîné"):
            model.predict(context.iloc[-24:], base.iloc[-24:])
        sparse_target = target.copy()
        sparse_target.iloc[50:] = np.nan
        with self.assertRaisesRegex(
            TopologyResidualCorrectionError,
            "aucun fallback implicite",
        ):
            model.fit(context, sparse_target, base)

    def test_global_hyperparameters_are_radius_independent_and_auditable(self) -> None:
        radius_zero = _model()
        radius_one = _model()
        self.assertEqual(radius_zero.hyperparameters(), radius_one.hyperparameters())
        self.assertEqual(radius_zero.hyperparameter_sha256(), radius_one.hyperparameter_sha256())
        parameters = radius_zero.hyperparameters()
        self.assertEqual(parameters["loss"], "absolute_error")
        self.assertEqual(parameters["random_state"], 120)
        self.assertFalse(parameters["early_stopping"])
        self.assertEqual(parameters["correction_clip"], [-30.0, 30.0])


if __name__ == "__main__":
    unittest.main()
