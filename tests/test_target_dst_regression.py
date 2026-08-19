from __future__ import annotations

import sys
import types
import unittest
from collections.abc import Callable

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from chronos2_hourly.hourly_contract import (
    coerce_hourly_target,
    local_delivery_day_index,
)
from chronos2_modular.data import regularize_target
from chronos2_modular.saturn import normalize_saturn_series


PARIS = "Europe/Paris"

SPRING_DAYS = (
    "2023-03-26",
    "2024-03-31",
    "2025-03-30",
    "2026-03-29",
)
FALL_DAYS = (
    "2023-10-29",
    "2024-10-27",
    "2025-10-26",
)

MULTI_YEAR_START_UTC = "2023-01-01 00:00"
MULTI_YEAR_END_UTC = "2026-06-01 23:00"
MULTI_YEAR_END_QH_UTC = "2026-06-01 23:45"


def _physical_grid(
    *,
    frequency: str,
    start: str = MULTI_YEAR_START_UTC,
    end: str | None = None,
) -> pd.DatetimeIndex:
    """Return the canonical sequence of physical delivery instants."""

    if end is None:
        end = (
            MULTI_YEAR_END_QH_UTC
            if frequency.lower() in {"15min", "15t"}
            else MULTI_YEAR_END_UTC
        )
    return pd.date_range(start, end, freq=frequency, tz="UTC")


def _series_on_index(
    index: pd.DatetimeIndex,
    *,
    name: str = "target",
) -> pd.Series:
    # Unique values make a swapped autumn fold observable in the assertions.
    return pd.Series(
        np.arange(len(index), dtype=float),
        index=index,
        name=name,
    )


def _utc_naive_series(
    index_utc: pd.DatetimeIndex,
    *,
    name: str = "target",
) -> pd.Series:
    """Mimic a Saturn series whose UTC offsets were stripped in transport."""

    if str(index_utc.tz) != "UTC":
        raise AssertionError("The fixture must be expressed in UTC.")
    return _series_on_index(index_utc.tz_localize(None), name=name)


def _local_naive_series(
    index_utc: pd.DatetimeIndex,
    *,
    name: str = "target",
) -> pd.Series:
    """Mimic Saturn civil labels: spring gap and repeated autumn hour."""

    if str(index_utc.tz) != "UTC":
        raise AssertionError("The fixture must be expressed in UTC.")
    local_naive = index_utc.tz_convert(PARIS).tz_localize(None)
    return _series_on_index(local_naive, name=name)


def _rotate_rows(series: pd.Series) -> pd.Series:
    """Make input non-monotonic without splitting a DST transition block."""

    pivot = len(series) // 3
    return pd.concat((series.iloc[pivot:], series.iloc[:pivot]))


def _assert_physical_round_trip(
    testcase: unittest.TestCase,
    normalized: pd.Series,
    expected_utc: pd.DatetimeIndex,
) -> None:
    testcase.assertFalse(normalized.index.has_duplicates)
    testcase.assertTrue(normalized.index.is_monotonic_increasing)
    testcase.assertTrue(
        normalized.index.tz_convert("UTC").equals(expected_utc),
        msg=(
            "La normalisation Saturn n'a pas restauré les instants UTC "
            "originaux."
        ),
    )
    np.testing.assert_array_equal(
        normalized.to_numpy(),
        np.arange(len(expected_utc), dtype=float),
    )


class SaturnLocalNaiveDstRegressionTests(unittest.TestCase):
    """Regression tests for the civil-naive shape returned by Saturn."""

    def test_multi_year_hourly_grid_has_no_spring_04_hole(self) -> None:
        """Reproduce the four spring 04:00 failures from the user trace."""

        expected_utc = _physical_grid(frequency="h")
        raw = _local_naive_series(expected_utc)

        # No timezone hint on purpose: the Saturn boundary must distinguish
        # this DST-shaped local grid from a continuous UTC-naive grid.
        normalized = normalize_saturn_series(raw, "target", PARIS)
        target, frequency, cleaning = regularize_target(
            normalized,
            PARIS,
            "h",
            0,
        )

        self.assertEqual(frequency.lower(), "h")
        self.assertEqual(cleaning["missing_before_interpolation"], 0)
        self.assertEqual(cleaning["missing_after_interpolation"], 0)
        _assert_physical_round_trip(self, target, expected_utc)

        for day in SPRING_DAYS:
            with self.subTest(day=day):
                expected_day = local_delivery_day_index(day, timezone=PARIS)
                actual_day = target.loc[
                    target.index.tz_convert(PARIS).date
                    == pd.Timestamp(day).date()
                ]
                self.assertEqual(len(actual_day), 23)
                self.assertTrue(
                    actual_day.index.tz_convert("UTC").equals(expected_day)
                )
                spring_04 = pd.Timestamp(f"{day} 04:00", tz=PARIS)
                self.assertIn(spring_04, actual_day.index)

        for day in FALL_DAYS:
            with self.subTest(day=day):
                expected_day = local_delivery_day_index(day, timezone=PARIS)
                actual_day = target.loc[
                    target.index.tz_convert(PARIS).date
                    == pd.Timestamp(day).date()
                ]
                repeated_02 = actual_day.index[
                    actual_day.index.tz_convert(PARIS).hour == 2
                ]
                self.assertEqual(len(actual_day), 25)
                self.assertEqual(len(repeated_02), 2)
                self.assertTrue(
                    actual_day.index.tz_convert("UTC").equals(expected_day)
                )

    def test_multi_year_quarter_hour_grid_round_trips_and_aggregates(self) -> None:
        expected_qh_utc = _physical_grid(frequency="15min")
        raw = _local_naive_series(expected_qh_utc)

        normalized = normalize_saturn_series(raw, "target", PARIS)
        _assert_physical_round_trip(self, normalized, expected_qh_utc)

        hourly = coerce_hourly_target(
            normalized,
            input_resolution="quarter_hour",
        )
        expected_hourly = pd.Series(
            np.arange(len(expected_qh_utc), dtype=float),
            index=expected_qh_utc,
        ).resample("h").mean()

        self.assertTrue(hourly.index.equals(expected_hourly.index))
        np.testing.assert_array_equal(
            hourly.to_numpy(),
            expected_hourly.to_numpy(),
        )

        for day in SPRING_DAYS:
            with self.subTest(day=day):
                local = hourly.index.tz_convert(PARIS)
                self.assertEqual(
                    int((local.date == pd.Timestamp(day).date()).sum()),
                    23,
                )
        for day in FALL_DAYS:
            with self.subTest(day=day):
                local = hourly.index.tz_convert(PARIS)
                self.assertEqual(
                    int((local.date == pd.Timestamp(day).date()).sum()),
                    25,
                )

    def test_hourly_descending_and_unsorted_inputs_restore_same_instants(
        self,
    ) -> None:
        expected_utc = _physical_grid(
            frequency="h",
            start="2024-01-01 00:00",
            end="2024-12-31 23:00",
        )
        ordered = _local_naive_series(expected_utc)
        cases: dict[str, Callable[[pd.Series], pd.Series]] = {
            "descending": lambda values: values.iloc[::-1],
            "rotated_unsorted": _rotate_rows,
        }

        for label, reorder in cases.items():
            with self.subTest(order=label):
                normalized = normalize_saturn_series(
                    reorder(ordered),
                    "target",
                    PARIS,
                )
                _assert_physical_round_trip(self, normalized, expected_utc)

    def test_quarter_hour_descending_and_unsorted_inputs_restore_same_instants(
        self,
    ) -> None:
        expected_utc = _physical_grid(
            frequency="15min",
            start="2024-01-01 00:00",
            end="2024-12-31 23:45",
        )
        ordered = _local_naive_series(expected_utc)
        cases: dict[str, Callable[[pd.Series], pd.Series]] = {
            "descending": lambda values: values.iloc[::-1],
            "rotated_unsorted": _rotate_rows,
        }

        for label, reorder in cases.items():
            with self.subTest(order=label):
                normalized = normalize_saturn_series(
                    reorder(ordered),
                    "target",
                    PARIS,
                )
                _assert_physical_round_trip(self, normalized, expected_utc)
                hourly = coerce_hourly_target(
                    normalized,
                    input_resolution="quarter_hour",
                )
                self.assertEqual(len(hourly), len(expected_utc) // 4)

    def test_random_fallback_order_is_rejected_instead_of_swapping_prices(
        self,
    ) -> None:
        expected_utc = _physical_grid(
            frequency="h",
            start="2024-10-26 00:00",
            end="2024-10-28 23:00",
        )
        raw = _local_naive_series(expected_utc).sample(
            frac=1.0,
            random_state=42,
        )

        with self.assertRaisesRegex(ValueError, "ordre source insuffisant"):
            normalize_saturn_series(raw, "target", PARIS)

    def test_explicit_local_naive_mode_rejects_nonexistent_hour(self) -> None:
        raw = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.DatetimeIndex(
                (
                    "2024-03-31 01:00",
                    "2024-03-31 02:00",
                    "2024-03-31 03:00",
                )
            ),
        )

        with self.assertRaises(ValueError):
            normalize_saturn_series(
                raw,
                "target",
                PARIS,
                naive_timezone=PARIS,
            )


class SaturnUtcNaiveControlTests(unittest.TestCase):
    """A continuous UTC-naive grid must not be mistaken for civil time."""

    def test_multi_year_hourly_grid_remains_utc(self) -> None:
        expected_utc = _physical_grid(frequency="h")
        normalized = normalize_saturn_series(
            _utc_naive_series(expected_utc),
            "target",
            PARIS,
            naive_timezone="UTC",
        )

        _assert_physical_round_trip(self, normalized, expected_utc)

        for day, expected_hours in (
            *((day, 23) for day in SPRING_DAYS),
            *((day, 25) for day in FALL_DAYS),
        ):
            with self.subTest(day=day):
                local = normalized.index.tz_convert(PARIS)
                self.assertEqual(
                    int((local.date == pd.Timestamp(day).date()).sum()),
                    expected_hours,
                )

    def test_multi_year_quarter_hour_grid_remains_utc(self) -> None:
        expected_utc = _physical_grid(frequency="15min")
        normalized = normalize_saturn_series(
            _utc_naive_series(expected_utc),
            "target",
            PARIS,
            naive_timezone="UTC",
        )

        _assert_physical_round_trip(self, normalized, expected_utc)
        hourly = coerce_hourly_target(
            normalized,
            input_resolution="quarter_hour",
        )
        self.assertEqual(len(hourly), len(expected_utc) // 4)


class TimezoneAwareDstControlTests(unittest.TestCase):
    """Timezone-aware Saturn input is already unambiguous."""

    def test_aware_hourly_utc_and_paris_inputs_are_identical(self) -> None:
        expected_utc = _physical_grid(
            frequency="h",
            start="2024-01-01 00:00",
            end="2024-12-31 23:00",
        )

        for source_timezone in ("UTC", PARIS):
            with self.subTest(source_timezone=source_timezone):
                raw = _series_on_index(
                    expected_utc.tz_convert(source_timezone)
                )
                normalized = normalize_saturn_series(
                    raw,
                    "target",
                    PARIS,
                )
                _assert_physical_round_trip(self, normalized, expected_utc)

    def test_aware_quarter_hour_utc_and_paris_inputs_are_identical(
        self,
    ) -> None:
        expected_utc = _physical_grid(
            frequency="15min",
            start="2024-01-01 00:00",
            end="2024-12-31 23:45",
        )

        for source_timezone in ("UTC", PARIS):
            with self.subTest(source_timezone=source_timezone):
                raw = _series_on_index(
                    expected_utc.tz_convert(source_timezone)
                )
                normalized = normalize_saturn_series(
                    raw,
                    "target",
                    PARIS,
                )
                _assert_physical_round_trip(self, normalized, expected_utc)
                hourly = coerce_hourly_target(
                    normalized,
                    input_resolution="quarter_hour",
                )
                self.assertEqual(len(hourly), len(expected_utc) // 4)


if __name__ == "__main__":
    unittest.main()
