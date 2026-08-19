from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from chronos2_hourly.models import (
    OptionalDependencyError,
    ResidualCorrectionError,
    ResidualCorrector,
    ResidualMetaFeatureBuilder,
    apply_residual_correction,
)


def _inputs(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    position = np.arange(len(index), dtype=float)
    residual = 45.0 + 8.0 * np.sin(position * 2.0 * np.pi / 24.0)
    X = pd.DataFrame(
        {
            "known_fr_residual_load_fcst_oracle": residual,
            "known_de_residual_load_fcst_oracle": residual + 3.0,
            "safe_fuel_cost": 25.0 + 0.01 * position,
            "price_lag_24h": 50.0 + 0.1 * position,
            "price_rolling_mean_168h": 48.0,
            "calendar_dayofyear_sin": np.sin(position / 365.0),
            "known_doy_cos": np.cos(position / 365.0),
        },
        index=index,
    )
    chronos_q50 = 55.0 + 0.2 * residual
    experts = pd.DataFrame(
        {
            "chronos2__q10": chronos_q50 - 9.0,
            "chronos2__q50": chronos_q50,
            "chronos2__q90": chronos_q50 + 11.0,
            "lear__q50": chronos_q50 - 1.0,
        },
        index=index,
    )
    return X, experts


def _base(index: pd.DatetimeIndex, median: np.ndarray | float) -> pd.DataFrame:
    values = np.broadcast_to(np.asarray(median, dtype=float), (len(index),)).copy()
    return pd.DataFrame(
        {"q10": values - 7.0, "q50": values, "q90": values + 9.0},
        index=index,
    )


class ResidualMetaFeatureBuilderTests(unittest.TestCase):
    def test_builds_profiles_and_excludes_price_history_and_doy(self) -> None:
        index = pd.date_range("2025-01-01", periods=72, freq="h", tz="UTC")
        X, experts = _inputs(index)
        original_x = X.copy(deep=True)
        original_experts = experts.copy(deep=True)
        builder = ResidualMetaFeatureBuilder(
            include_calendar=False,
            include_daily_profiles=True,
        )

        result = builder.fit_transform(X, experts)

        self.assertNotIn("price_lag_24h", result)
        self.assertNotIn("price_rolling_mean_168h", result)
        self.assertNotIn("calendar_dayofyear_sin", result)
        self.assertNotIn("known_doy_cos", result)
        self.assertIn("safe_fuel_cost", result)
        self.assertIn("chronos2__q50", result)
        self.assertIn(
            "known_fr_residual_load_fcst_oracle__day_mean",
            result,
        )
        self.assertIn(
            "known_fr_residual_load_fcst_oracle__ramp_2h",
            result,
        )
        self.assertIn("chronos2__q50__day_ramp_abs_max", result)
        self.assertIn("chronos2__interval_width", result)
        self.assertIn("chronos2__interval_width__day_mean", result)
        self.assertIn("expert_q50_range", result)
        self.assertIn("fr_minus_de_residual_load", result)
        pd.testing.assert_frame_equal(X, original_x)
        pd.testing.assert_frame_equal(experts, original_experts)

    def test_price_and_day_of_year_exclusions_are_configurable(self) -> None:
        index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        X, experts = _inputs(index)
        result = ResidualMetaFeatureBuilder(
            include_calendar=False,
            include_daily_profiles=False,
            exclude_historical_prices=False,
            exclude_day_of_year=False,
        ).fit_transform(X, experts)

        self.assertIn("price_lag_24h", result)
        self.assertIn("price_rolling_mean_168h", result)
        self.assertIn("calendar_dayofyear_sin", result)
        self.assertIn("known_doy_cos", result)

    def test_builds_renewable_supply_interactions_and_missing_flags(self) -> None:
        index = pd.date_range("2025-04-01", periods=48, freq="h", tz="UTC")
        X, experts = _inputs(index)
        X["known_fr_load_fcst_oracle"] = 50.0
        X["known_fr_wind_generation_fcst_oracle"] = 10.0
        X["known_fr_solar_generation_fcst_oracle"] = 5.0
        X["known_fr_hydro_ror_generation_fcst_oracle"] = 2.0
        X["known_fr_nuclear_generation_fcst_long_oracle"] = 30.0
        X.loc[index[0], "known_fr_solar_generation_fcst_oracle"] = np.nan
        builder = ResidualMetaFeatureBuilder(include_calendar=False)

        result = builder.fit_transform(X, experts)

        self.assertEqual(float(result.iloc[1]["fr_variable_renewables_fcst"]), 15.0)
        self.assertEqual(float(result.iloc[1]["fr_net_load_wind_solar_fcst"]), 35.0)
        self.assertEqual(
            float(result.iloc[1]["fr_net_load_including_hydro_fcst"]),
            33.0,
        )
        self.assertEqual(float(result.iloc[1]["fr_variable_renewables_share"]), 0.3)
        self.assertEqual(float(result.iloc[1]["fr_net_load_after_nuclear_fcst"]), 5.0)
        self.assertIn("fr_net_load_wind_solar_fcst__day_min", result)
        missing = "known_fr_solar_generation_fcst_oracle__missing"
        self.assertEqual(result[missing].iloc[:2].tolist(), [1.0, 0.0])

        complete_live = X.iloc[-24:].fillna(0.0)
        live = builder.transform(complete_live, experts.loc[complete_live.index])
        self.assertIn(missing, live)
        self.assertEqual(float(live[missing].sum()), 0.0)

    def test_ramps_restart_at_each_local_day_and_cover_dst_fallback(self) -> None:
        local = pd.date_range(
            "2025-10-26",
            "2025-10-27",
            freq="h",
            inclusive="left",
            tz="Europe/Paris",
        )
        index = local.tz_convert("UTC")
        X, experts = _inputs(index)
        result = ResidualMetaFeatureBuilder(
            include_calendar=False,
        ).fit_transform(X, experts)

        self.assertEqual(len(result), 25)
        self.assertEqual(
            float(result.iloc[0]["known_fr_residual_load_fcst_oracle__ramp_1h"]),
            0.0,
        )
        self.assertEqual(
            result.iloc[:2][
                "known_fr_residual_load_fcst_oracle__ramp_2h"
            ].tolist(),
            [0.0, 0.0],
        )
        repeated_hour = local.hour == 2
        self.assertEqual(int(repeated_hour.sum()), 2)
        self.assertTrue(
            np.isfinite(
                result.loc[
                    repeated_hour,
                    "known_fr_residual_load_fcst_oracle__ramp_2h",
                ].to_numpy()
            ).all()
        )

    def test_rich_calendar_is_identical_for_live_day_and_history_slice(self) -> None:
        local_history = pd.date_range(
            "2026-03-28",
            "2026-03-31",
            freq="h",
            inclusive="left",
            tz="Europe/Paris",
        )
        history_index = local_history.tz_convert("UTC")
        live_mask = local_history.date == pd.Timestamp("2026-03-29").date()
        live_index = history_index[live_mask]
        history_x, history_experts = _inputs(history_index)
        live_x = history_x.loc[live_index].copy()
        live_experts = history_experts.loc[live_index].copy()
        options = dict(
            include_calendar=False,
            include_rich_calendar=True,
            include_daily_profiles=False,
        )

        history = ResidualMetaFeatureBuilder(**options).fit_transform(
            history_x,
            history_experts,
        )
        live = ResidualMetaFeatureBuilder(**options).fit_transform(
            live_x,
            live_experts,
        )
        calendar_columns = [
            column for column in live if column.startswith("known_cal_")
        ]

        pd.testing.assert_frame_equal(
            history.loc[live_index, calendar_columns],
            live.loc[:, calendar_columns],
        )
        self.assertEqual(
            live["known_cal_dst_transition_day_oracle"].unique().tolist(),
            [1.0],
        )

    def test_transform_rejects_index_and_schema_drift(self) -> None:
        index = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
        X, experts = _inputs(index)
        builder = ResidualMetaFeatureBuilder(include_calendar=False).fit(X, experts)

        with self.assertRaisesRegex(ResidualCorrectionError, "Schéma X différent"):
            builder.transform(X.drop(columns="safe_fuel_cost"), experts)
        shifted_experts = experts.copy()
        shifted_experts.index = shifted_experts.index + pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ResidualCorrectionError, "exactement égal"):
            builder.transform(X, shifted_experts)


class ResidualCorrectionApplicationTests(unittest.TestCase):
    def test_common_shift_preserves_width_order_and_applies_scale_clip(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        base = _base(index, 50.0)

        result = apply_residual_correction(
            base,
            pd.Series([20.0, -20.0], index=index),
            correction_scale=0.5,
            max_abs_correction=3.0,
        )

        self.assertEqual(result["q50"].tolist(), [53.0, 47.0])
        np.testing.assert_allclose(result["q50"] - result["q10"], 7.0)
        np.testing.assert_allclose(result["q90"] - result["q50"], 9.0)
        self.assertTrue((result["q10"] <= result["q50"]).all())
        self.assertTrue((result["q50"] <= result["q90"]).all())

    def test_crossed_base_and_misaligned_correction_are_rejected(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        crossed = _base(index, 50.0)
        crossed.loc[index[0], "q10"] = 60.0
        with self.assertRaisesRegex(ResidualCorrectionError, "se croisent"):
            apply_residual_correction(crossed, [0.0, 0.0])
        with self.assertRaisesRegex(ResidualCorrectionError, "exactement égal"):
            apply_residual_correction(
                _base(index, 50.0),
                pd.Series([0.0, 0.0], index=index + pd.Timedelta(hours=1)),
            )


class ResidualCorrectorTests(unittest.TestCase):
    def test_sklearn_early_stopping_is_explicit_and_validated(self) -> None:
        model = ResidualCorrector(
            backend="sklearn",
            sklearn_early_stopping=False,
        )
        model.backend_ = "sklearn"
        pipeline = model._new_model()

        self.assertFalse(pipeline.named_steps["model"].early_stopping)
        with self.assertRaisesRegex(ValueError, "sklearn_early_stopping"):
            ResidualCorrector(
                backend="sklearn",
                sklearn_early_stopping="sometimes",
            )

    def test_sklearn_fallback_improves_systematic_residual(self) -> None:
        index = pd.date_range("2025-01-01", periods=360, freq="h", tz="UTC")
        local_hour = index.tz_convert("Europe/Paris").hour.to_numpy(dtype=float)
        signal = np.sin(np.arange(len(index)) * 2.0 * np.pi / 24.0)
        residual_load = 50.0 + 10.0 * signal
        X = pd.DataFrame(
            {
                "known_fr_residual_load_fcst_oracle": residual_load,
                "delivery_hour": local_hour,
            },
            index=index,
        )
        base_median = 60.0 + 0.15 * residual_load
        base = _base(index, base_median)
        target = pd.Series(base_median + 5.0 * signal, index=index)
        train = slice(0, 288)
        test = slice(288, None)
        model = ResidualCorrector(
            backend="sklearn",
            feature_builder_options={"include_rich_calendar": False},
            min_training_rows=48,
            iterations=80,
            depth=4,
            learning_rate=0.08,
            min_samples_leaf=8,
            max_abs_correction=8.0,
        ).fit(X.iloc[train], target.iloc[train], base.iloc[train])

        corrected = model.predict(X.iloc[test], base.iloc[test])
        baseline_mae = float(
            np.mean(np.abs(target.iloc[test].to_numpy() - base.iloc[test]["q50"].to_numpy()))
        )
        corrected_mae = float(
            np.mean(
                np.abs(
                    target.iloc[test].to_numpy()
                    - corrected["q50"].to_numpy()
                )
            )
        )

        self.assertEqual(model.backend_, "sklearn")
        self.assertLess(corrected_mae, baseline_mae * 0.45)
        self.assertTrue((corrected["q10"] <= corrected["q50"]).all())
        self.assertTrue((corrected["q50"] <= corrected["q90"]).all())

    def test_explicit_catboost_backend_has_actionable_error(self) -> None:
        index = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
        X = pd.DataFrame({"residual_load": [1.0, 2.0, 3.0, 4.0]}, index=index)
        base = _base(index, 50.0)
        target = pd.Series([51.0, 52.0, 53.0, 54.0], index=index)
        with patch(
            "chronos2_hourly.models.residual_corrector.catboost_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(OptionalDependencyError, "pip install catboost"):
                ResidualCorrector(
                    backend="catboost",
                    feature_builder_options={
                        "include_calendar": False,
                        "include_daily_profiles": False,
                    },
                    min_training_rows=2,
                ).fit(X, target, base)


if __name__ == "__main__":
    unittest.main()
