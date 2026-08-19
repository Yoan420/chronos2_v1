from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from chronos2_hourly.features import (
    HourlyFeatureContractError,
    build_calendar_features,
    build_history_future_feature_matrix,
    build_hourly_feature_matrix,
    validate_utc_hourly_index,
)


def utc_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


class UtcIndexContractTests(unittest.TestCase):
    def test_rejects_non_utc_duplicates_and_gaps(self) -> None:
        paris = pd.date_range(
            "2026-01-01",
            periods=3,
            freq="h",
            tz="Europe/Paris",
        )
        with self.assertRaisesRegex(HourlyFeatureContractError, "UTC"):
            validate_utc_hourly_index(paris)

        base = utc_index("2026-01-01", 3)
        duplicated = base.insert(2, base[1])
        with self.assertRaisesRegex(HourlyFeatureContractError, "dupliqué"):
            validate_utc_hourly_index(duplicated)

        gap = base.delete(1)
        with self.assertRaisesRegex(HourlyFeatureContractError, "continu"):
            validate_utc_hourly_index(gap)


class CalendarFeatureTests(unittest.TestCase):
    def test_autumn_fold_keeps_both_local_02_hours(self) -> None:
        index = pd.date_range(
            "2024-10-26 22:00",
            "2024-10-27 22:00",
            freq="h",
            tz="UTC",
        )
        features = build_calendar_features(index, timezone="Europe/Paris")
        repeated = features.loc[features["calendar_local_hour"].eq(2)]

        self.assertEqual(len(features), 25)
        self.assertEqual(repeated["calendar_dst_fold"].tolist(), [0, 1])
        self.assertEqual(
            repeated["calendar_utc_offset_hours"].tolist(), [2.0, 1.0]
        )
        self.assertEqual(repeated["calendar_is_dst"].tolist(), [1, 0])
        self.assertEqual(
            repeated["calendar_hour_sin"].iloc[0],
            repeated["calendar_hour_sin"].iloc[1],
        )


class HourlyFeatureMatrixTests(unittest.TestCase):
    def test_lags_and_rolling_statistics_are_strictly_causal(self) -> None:
        index = utc_index("2026-01-01", 200)
        target = pd.Series(np.arange(200, dtype=float), index=index)
        covariates = pd.DataFrame(
            {"load_forecast": 1000.0 + np.arange(200)},
            index=index,
        )

        features = build_hourly_feature_matrix(
            target,
            covariates,
            price_lags=(24, 48, 168),
            rolling_windows=(3,),
            rolling_statistics=("mean", "std", "min", "max"),
        )

        self.assertEqual(features["price_lag_24h"].iloc[50], 26.0)
        self.assertEqual(features["price_lag_48h"].iloc[50], 2.0)
        self.assertEqual(features["price_lag_168h"].iloc[180], 12.0)
        self.assertEqual(features["price_rolling_mean_3h"].iloc[50], 45.0)
        self.assertEqual(features["price_rolling_min_3h"].iloc[50], 44.0)
        self.assertEqual(features["price_rolling_max_3h"].iloc[50], 46.0)
        self.assertAlmostEqual(
            features["price_rolling_std_3h"].iloc[50],
            np.std([44.0, 45.0, 46.0], ddof=0),
        )
        self.assertNotIn("price", features.columns)

        mutated = target.copy()
        mutated.iloc[10:] = -99999.0
        changed = build_hourly_feature_matrix(
            mutated,
            covariates,
            price_lags=(24, 48, 168),
            rolling_windows=(3,),
        )
        # target[t] cannot affect any feature at or before t.
        assert_frame_equal(features.iloc[:11], changed.iloc[:11])

    def test_mutating_delivery_day_prices_cannot_change_same_day_features(
        self,
    ) -> None:
        index = pd.date_range(
            "2024-10-23 22:00",
            "2024-10-29 22:00",
            freq="h",
            tz="UTC",
            inclusive="left",
        )
        target = pd.Series(np.arange(len(index), dtype=float), index=index)
        covariates = pd.DataFrame({"load": 1000.0}, index=index)
        baseline = build_hourly_feature_matrix(
            target,
            covariates,
            price_lags=(1, 24, 48),
            rolling_windows=(3, 24),
        )
        local_days = pd.Series(
            index.tz_convert("Europe/Paris").date,
            index=index,
        )

        for delivery_day in (
            pd.Timestamp("2024-10-26").date(),
            pd.Timestamp("2024-10-27").date(),
        ):
            with self.subTest(delivery_day=delivery_day):
                day_mask = local_days.eq(delivery_day).to_numpy()
                mutated_target = target.copy()
                mutated_target.loc[day_mask] += 100000.0
                mutated = build_hourly_feature_matrix(
                    mutated_target,
                    covariates,
                    price_lags=(1, 24, 48),
                    rolling_windows=(3, 24),
                )
                generated = [
                    column
                    for column in baseline.columns
                    if column.startswith("price_")
                ]
                assert_frame_equal(
                    baseline.loc[day_mask, generated],
                    mutated.loc[day_mask, generated],
                    check_exact=True,
                )

    def test_rolling_is_frozen_on_d_and_ends_at_d_minus_1_2300(self) -> None:
        index = pd.date_range(
            "2026-01-01 23:00",
            periods=73,
            freq="h",
            tz="UTC",
        )
        target = pd.Series(np.arange(len(index), dtype=float), index=index)
        covariates = pd.DataFrame({"load": 1000.0}, index=index)
        features = build_hourly_feature_matrix(
            target,
            covariates,
            price_lags=(24,),
            rolling_windows=(3,),
            rolling_statistics=("mean", "min", "max"),
        )
        local_days = pd.Series(
            index.tz_convert("Europe/Paris").date,
            index=index,
        )
        delivery_day = pd.Timestamp("2026-01-03").date()
        day_mask = local_days.eq(delivery_day).to_numpy()
        previous = target.loc[local_days.lt(delivery_day)].iloc[-3:]

        self.assertEqual(
            features.loc[day_mask, "price_rolling_mean_3h"].unique().tolist(),
            [float(previous.mean())],
        )
        self.assertEqual(
            features.loc[day_mask, "price_rolling_min_3h"].unique().tolist(),
            [float(previous.min())],
        )
        self.assertEqual(
            features.loc[day_mask, "price_rolling_max_3h"].unique().tolist(),
            [float(previous.max())],
        )
        last_previous_local = previous.index[-1].tz_convert("Europe/Paris")
        self.assertEqual(last_previous_local.hour, 23)
        self.assertEqual(
            last_previous_local.date(),
            pd.Timestamp("2026-01-02").date(),
        )

    def test_historical_autumn_day_masks_25th_hour_lag24(self) -> None:
        index = pd.date_range(
            "2024-10-25 22:00",
            "2024-10-28 23:00",
            freq="h",
            tz="UTC",
            inclusive="left",
        )
        target = pd.Series(np.arange(len(index), dtype=float), index=index)
        covariates = pd.DataFrame({"load": 1000.0}, index=index)
        features = build_hourly_feature_matrix(
            target,
            covariates,
            price_lags=(24,),
            rolling_windows=(),
        )
        local_days = pd.Series(
            index.tz_convert("Europe/Paris").date,
            index=index,
        )
        autumn_mask = local_days.eq(pd.Timestamp("2024-10-27").date()).to_numpy()
        autumn_lag = features.loc[autumn_mask, "price_lag_24h"]

        self.assertEqual(len(autumn_lag), 25)
        self.assertTrue(autumn_lag.iloc[:24].notna().all())
        self.assertTrue(np.isnan(autumn_lag.iloc[24]))

    def test_missing_inputs_are_not_interpolated(self) -> None:
        index = utc_index("2026-01-01", 30)
        target = pd.Series(np.arange(30, dtype=float), index=index)
        target.iloc[5] = np.nan
        covariates = pd.DataFrame({"load": np.arange(30.0)}, index=index)
        covariates.iloc[12, 0] = np.nan

        features = build_hourly_feature_matrix(
            target,
            covariates,
            price_lags=(1,),
            rolling_windows=(3,),
        )

        self.assertTrue(np.isnan(features["load"].iloc[12]))
        self.assertTrue(np.isnan(features["price_lag_1h"].iloc[6]))
        self.assertTrue(
            features["price_rolling_mean_3h"].iloc[6:9].isna().all()
        )

    def test_covariate_index_must_match_without_reindexing(self) -> None:
        index = utc_index("2026-01-01", 10)
        target = pd.Series(np.arange(10.0), index=index)
        covariates = pd.DataFrame(
            {"load": np.arange(9.0)},
            index=index[:-1],
        )
        with self.assertRaisesRegex(HourlyFeatureContractError, "exactement"):
            build_hourly_feature_matrix(target, covariates)


class HistoryFutureFeatureTests(unittest.TestCase):
    def test_future_prices_are_explicitly_masked_on_25_hour_day(self) -> None:
        future_index = pd.date_range(
            "2024-10-26 22:00",
            "2024-10-27 22:00",
            freq="h",
            tz="UTC",
        )
        history_index = pd.date_range(
            end=future_index[0] - pd.Timedelta(hours=1),
            periods=200,
            freq="h",
            tz="UTC",
        )
        history_target = pd.Series(
            np.arange(200, dtype=float),
            index=history_index,
            name="price",
        )
        history_covariates = pd.DataFrame(
            {"load": np.arange(200, dtype=float)},
            index=history_index,
        )
        future_covariates = pd.DataFrame(
            {"load": 1000.0 + np.arange(25, dtype=float)},
            index=future_index,
        )

        future = build_history_future_feature_matrix(
            history_target,
            history_covariates,
            future_covariates,
            price_lags=(24,),
            rolling_windows=(3,),
            scope="future",
        )

        self.assertEqual(len(future), 25)
        np.testing.assert_allclose(
            future["price_lag_24h"].iloc[:24].to_numpy(),
            np.arange(176.0, 200.0),
        )
        # The 25th row's 24-hour lag belongs to the forecast day itself.  It
        # remains missing instead of reading the future realised price.
        self.assertTrue(np.isnan(future["price_lag_24h"].iloc[24]))
        self.assertEqual(
            future["price_rolling_mean_3h"].unique().tolist(), [198.0]
        )
        self.assertEqual(future["load"].iloc[-1], 1024.0)
        repeated = future.loc[future["calendar_local_hour"].eq(2)]
        self.assertEqual(repeated["calendar_dst_fold"].tolist(), [0, 1])

    def test_history_future_boundary_must_be_contiguous(self) -> None:
        history_index = utc_index("2026-01-01", 10)
        history_target = pd.Series(np.arange(10.0), index=history_index)
        history_covariates = pd.DataFrame(
            {"load": np.arange(10.0)}, index=history_index
        )
        future_index = pd.date_range(
            history_index[-1] + pd.Timedelta(hours=2),
            periods=3,
            freq="h",
            tz="UTC",
        )
        future_covariates = pd.DataFrame(
            {"load": np.arange(3.0)}, index=future_index
        )

        with self.assertRaisesRegex(HourlyFeatureContractError, "exactement"):
            build_history_future_feature_matrix(
                history_target,
                history_covariates,
                future_covariates,
            )


if __name__ == "__main__":
    unittest.main()
