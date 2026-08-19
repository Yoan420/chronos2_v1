from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import (
    HourlyTargetContractError,
    aggregate_quarter_hour_prices,
    coerce_hourly_target,
    delivery_day_metadata,
    local_delivery_day_index,
    validate_hourly_target,
)


def quarter_hour_index_for_local_day(day: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(day, tz="Europe/Paris")
    end = start + pd.DateOffset(days=1)
    return pd.date_range(
        start.tz_convert("UTC"),
        end.tz_convert("UTC"),
        freq="15min",
        inclusive="left",
    )


class LocalDeliveryDayTests(unittest.TestCase):
    def test_local_days_contain_23_24_or_25_distinct_hours(self) -> None:
        normal = local_delivery_day_index("2024-02-15")
        spring = local_delivery_day_index("2024-03-31")
        autumn = local_delivery_day_index("2024-10-27")

        self.assertEqual(len(normal), 24)
        self.assertEqual(len(spring), 23)
        self.assertEqual(len(autumn), 25)
        for index in (normal, spring, autumn):
            self.assertEqual(str(index.tz), "UTC")
            self.assertFalse(index.has_duplicates)
            self.assertTrue(
                bool((index.to_series().diff().dropna() == pd.Timedelta("1h")).all())
            )

    def test_autumn_metadata_disambiguates_repeated_local_hour(self) -> None:
        metadata = delivery_day_metadata("2024-10-27")
        repeated = metadata.loc[metadata["local_hour"].eq(2)]

        self.assertEqual(len(metadata), 25)
        self.assertEqual(repeated["fold"].tolist(), [0, 1])
        self.assertEqual(repeated["utc_offset"].tolist(), ["+02:00", "+01:00"])
        self.assertEqual(
            repeated["utc_offset_minutes"].tolist(), [120, 60]
        )
        self.assertEqual(metadata["hours_in_local_day"].unique().tolist(), [25])
        self.assertTrue(metadata["delivery_start_utc"].is_unique)


class QuarterHourAggregationTests(unittest.TestCase):
    def test_coerce_target_keeps_hourly_output_for_qh_input(self) -> None:
        index = quarter_hour_index_for_local_day("2024-10-27")
        qh = pd.Series(np.arange(len(index), dtype=float), index=index)

        target = coerce_hourly_target(
            qh,
            input_resolution="quarter_hour",
        )

        self.assertEqual(len(target), 25)
        self.assertEqual(str(target.index.tz), "UTC")
        self.assertTrue((target.index.minute == 0).all())

    def test_qh_prices_are_aggregated_in_utc_across_both_dst_days(self) -> None:
        for day, expected_quarters, expected_hours in (
            ("2024-03-31", 92, 23),
            ("2024-10-27", 100, 25),
        ):
            with self.subTest(day=day):
                index = quarter_hour_index_for_local_day(day)
                self.assertEqual(len(index), expected_quarters)
                values = pd.Series(np.arange(len(index), dtype=float), index=index)

                hourly = aggregate_quarter_hour_prices(values)

                self.assertEqual(len(hourly), expected_hours)
                self.assertEqual(str(hourly.index.tz), "UTC")
                self.assertFalse(hourly.index.has_duplicates)
                np.testing.assert_allclose(
                    hourly.to_numpy(),
                    np.arange(1.5, expected_quarters, 4.0),
                )

    def test_missing_quarter_is_never_interpolated(self) -> None:
        index = quarter_hour_index_for_local_day("2024-02-15")
        values = pd.Series(10.0, index=index)
        missing_timestamp = index[6]
        incomplete = values.drop(missing_timestamp)

        with self.assertRaises(HourlyTargetContractError):
            aggregate_quarter_hour_prices(incomplete)

        hourly = aggregate_quarter_hour_prices(
            incomplete,
            incomplete="nan",
        )
        affected_hour = missing_timestamp.floor("h")
        self.assertTrue(pd.isna(hourly.loc[affected_hour]))
        self.assertEqual(int(hourly.isna().sum()), 1)

        dropped = aggregate_quarter_hour_prices(
            incomplete,
            incomplete="drop",
        )
        self.assertNotIn(affected_hour, dropped.index)
        self.assertEqual(len(dropped), 23)

    def test_naive_index_is_rejected(self) -> None:
        naive = pd.Series(
            [1.0, 2.0, 3.0, 4.0],
            index=pd.date_range("2024-01-01", periods=4, freq="15min"),
        )
        with self.assertRaises(HourlyTargetContractError):
            aggregate_quarter_hour_prices(naive)


class HourlyTargetValidationTests(unittest.TestCase):
    def test_existing_hourly_target_matches_qh_mean(self) -> None:
        index = quarter_hour_index_for_local_day("2024-10-27")
        qh = pd.Series(np.arange(len(index), dtype=float), index=index)
        expected = aggregate_quarter_hour_prices(qh)
        existing_local = expected.copy()
        existing_local.index = existing_local.index.tz_convert("Europe/Paris")

        report = validate_hourly_target(existing_local, qh)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.compared_hours, 25)
        self.assertEqual(report.mismatched_hours, 0)
        self.assertEqual(report.max_abs_error, 0.0)

    def test_validation_reports_mismatch_and_incomplete_hour(self) -> None:
        index = quarter_hour_index_for_local_day("2024-02-15")
        qh = pd.Series(np.arange(len(index), dtype=float), index=index)
        existing = aggregate_quarter_hour_prices(qh)
        existing.iloc[3] += 2.0
        qh = qh.drop(index[20])

        report = validate_hourly_target(existing, qh)

        self.assertFalse(report.is_valid)
        self.assertEqual(report.mismatched_hours, 1)
        self.assertEqual(report.incomplete_quarter_hours, 1)
        self.assertEqual(report.missing_in_derived, 1)
        with self.assertRaises(HourlyTargetContractError):
            report.raise_if_invalid()


if __name__ == "__main__":
    unittest.main()
