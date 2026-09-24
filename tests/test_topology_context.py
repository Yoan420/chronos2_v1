from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from chronos2_hourly.topology_context import (
    CALENDAR_FEATURE_COLUMNS,
    DIRECT_NEIGHBORS,
    TOPOLOGY_CONTEXT_COLUMNS,
    TOPOLOGY_FEATURE_COLUMNS,
    TOPOLOGY_ZONES,
    TopologyContextError,
    build_topology_context,
    topology_mask,
    zones_within_radius,
)


def _input_frames(
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    step = np.arange(len(index), dtype=float)
    residual = pd.DataFrame(
        {
            "FR": 10.0 + step,
            "DE": 20.0 + step,
            "BE": 30.0 + step,
            "NL": 40.0 + step,
            "ES": 50.0 + step,
        },
        index=index,
    )
    prices = pd.DataFrame(
        {
            zone: 100.0 * position + step
            for position, zone in enumerate(TOPOLOGY_ZONES, start=1)
        },
        index=index,
    )
    return residual, prices


class TopologyGraphTests(unittest.TestCase):
    def test_induced_graph_is_exact_and_symmetric(self) -> None:
        expected = {
            "FR": {"DE", "BE", "ES"},
            "DE": {"FR", "BE", "NL"},
            "BE": {"FR", "DE", "NL"},
            "NL": {"DE", "BE"},
            "ES": {"FR"},
        }
        self.assertEqual(
            {zone: set(neighbours) for zone, neighbours in DIRECT_NEIGHBORS.items()},
            expected,
        )
        for zone, neighbours in DIRECT_NEIGHBORS.items():
            for neighbour in neighbours:
                self.assertIn(zone, DIRECT_NEIGHBORS[neighbour])

    def test_radius_zero_and_one_masks_are_deterministic(self) -> None:
        self.assertEqual(zones_within_radius("fr", 0), ("FR",))
        self.assertEqual(zones_within_radius("FR", 1), ("FR", "DE", "BE", "ES"))
        self.assertEqual(zones_within_radius("NL", 1), ("NL", "DE", "BE"))
        expected = pd.Series(
            [1, 1, 1, 0, 1],
            index=pd.Index(TOPOLOGY_ZONES, name="zone"),
            dtype="int8",
            name="topology_mask",
        )
        assert_series_equal(topology_mask("FR", 1), expected)
        with self.assertRaisesRegex(TopologyContextError, "0 ou 1"):
            zones_within_radius("FR", 2)


class TopologyContextBuilderTests(unittest.TestCase):
    def test_schema_pooling_metadata_and_input_immutability(self) -> None:
        index = pd.date_range("2026-01-01", periods=72, freq="h", tz="UTC")
        residual, prices = _input_frames(index)
        original_residual = residual.copy(deep=True)
        original_prices = prices.copy(deep=True)

        context = build_topology_context(
            residual,
            prices,
            target_zone="FR",
            radius=1,
        )

        self.assertEqual(tuple(context.columns), TOPOLOGY_CONTEXT_COLUMNS)
        self.assertEqual(len(TOPOLOGY_FEATURE_COLUMNS), 10)
        self.assertEqual(len(CALENDAR_FEATURE_COLUMNS), 12)
        self.assertEqual(context.shape[1], 22)
        np.testing.assert_allclose(
            context["topology__residual_load__pool_mean"],
            27.5 + np.arange(len(index), dtype=float),
        )
        self.assertTrue(
            (context["topology__residual_load__pool_count"] == 4.0).all()
        )
        self.assertTrue(
            (context["topology__residual_load__pool_coverage"] == 1.0).all()
        )
        metadata = context.attrs["topology_context"]
        self.assertEqual(metadata["target_zone"], "FR")
        self.assertEqual(metadata["radius"], 1)
        self.assertEqual(metadata["neighbours"], ["DE", "BE", "ES"])
        self.assertEqual(metadata["included_zones"], ["FR", "DE", "BE", "ES"])
        self.assertEqual(metadata["residual_load_source"], "sealed_pit")
        self.assertEqual(metadata["missing_policy"], "native_nan_no_interpolation")
        assert_frame_equal(residual, original_residual)
        assert_frame_equal(prices, original_prices)

    def test_radius_zero_is_local_model_not_identity(self) -> None:
        index = pd.date_range("2026-02-01", periods=48, freq="h", tz="UTC")
        residual, prices = _input_frames(index)
        context = build_topology_context(
            residual[["FR"]],
            prices[["FR"]],
            target_zone="FR",
            radius=0,
        )

        np.testing.assert_allclose(
            context["topology__residual_load__pool_mean"],
            context["topology__residual_load__local"],
        )
        np.testing.assert_allclose(
            context["topology__residual_load__local_minus_pool"],
            0.0,
        )
        self.assertEqual(context.attrs["topology_context"]["neighbours"], [])
        self.assertTrue(
            (context["topology__residual_load__pool_count"] == 1.0).all()
        )

    def test_sparse_pool_is_cross_sectional_and_never_interpolates(self) -> None:
        index = pd.date_range("2026-01-01", periods=72, freq="h", tz="UTC")
        residual, prices = _input_frames(index)
        residual.loc[index[10], ["FR", "DE"]] = np.nan
        residual.loc[index[11], ["FR", "DE", "BE", "ES"]] = np.nan
        prices.loc[index[0], "DE"] = np.nan

        context = build_topology_context(
            residual,
            prices,
            target_zone="FR",
            radius=1,
        )

        self.assertTrue(np.isnan(context.loc[index[10], "topology__residual_load__local"]))
        self.assertEqual(
            context.loc[index[10], "topology__residual_load__pool_count"],
            2.0,
        )
        self.assertAlmostEqual(
            context.loc[index[10], "topology__residual_load__pool_mean"],
            (40.0 + 60.0) / 2.0,
        )
        self.assertTrue(
            np.isnan(context.loc[index[11], "topology__residual_load__pool_mean"])
        )
        self.assertEqual(
            context.loc[index[11], "topology__residual_load__pool_count"],
            0.0,
        )
        self.assertEqual(
            context.loc[index[11], "topology__residual_load__pool_coverage"],
            0.0,
        )
        audit = context.attrs["topology_missing_audit"]
        self.assertEqual(audit["residual_load_input"]["FR"], 2)
        self.assertGreaterEqual(audit["price_lag24"]["DE"], 25)

    def test_radius_one_ignores_non_neighboring_remote_zone(self) -> None:
        index = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
        residual, prices = _input_frames(index)
        baseline = build_topology_context(
            residual,
            prices,
            target_zone="DE",
            radius=1,
        )
        changed_residual = residual.copy()
        changed_prices = prices.copy()
        changed_residual["ES"] = changed_residual["ES"] * 10_000.0
        changed_prices["ES"] = changed_prices["ES"] * 10_000.0
        changed = build_topology_context(
            changed_residual,
            changed_prices,
            target_zone="DE",
            radius=1,
        )
        assert_frame_equal(baseline, changed)

    def test_physical_lag_and_dst_guard_are_exact_on_fallback(self) -> None:
        index = pd.date_range(
            "2025-10-24 00:00",
            periods=96,
            freq="h",
            tz="UTC",
        )
        residual, prices = _input_frames(index)
        context = build_topology_context(
            residual[["FR"]],
            prices[["FR"]],
            target_zone="FR",
            radius=0,
        )
        lag = context["topology__price_da_lag24h__local"]
        local = index.tz_convert("Europe/Paris")
        for position in range(len(index)):
            if position < 24 or local[position - 24].date() >= local[position].date():
                self.assertTrue(np.isnan(lag.iloc[position]))
            else:
                self.assertEqual(lag.iloc[position], prices["FR"].iloc[position - 24])

        fallback = local.date == pd.Timestamp("2025-10-26").date()
        self.assertEqual(int(fallback.sum()), 25)
        repeated = fallback & (local.hour == 2)
        self.assertEqual(int(repeated.sum()), 2)
        self.assertEqual(
            context.loc[repeated, "calendar_dst_fold"].tolist(),
            [0, 1],
        )
        self.assertEqual(
            context.loc[repeated, "calendar_utc_offset_hours"].tolist(),
            [2.0, 1.0],
        )

    def test_spring_day_has_23_distinct_utc_rows(self) -> None:
        index = pd.date_range(
            "2026-03-28 00:00",
            periods=72,
            freq="h",
            tz="UTC",
        )
        residual, prices = _input_frames(index)
        context = build_topology_context(
            residual[["FR"]],
            prices[["FR"]],
            target_zone="FR",
            radius=0,
        )
        local = index.tz_convert("Europe/Paris")
        spring = local.date == pd.Timestamp("2026-03-29").date()
        self.assertEqual(int(spring.sum()), 23)
        self.assertEqual(len(context.loc[spring]), 23)
        self.assertFalse(context.index.has_duplicates)

    def test_invalid_inputs_fail_closed(self) -> None:
        index = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
        residual, prices = _input_frames(index)
        with self.assertRaisesRegex(TopologyContextError, "aucun fallback"):
            build_topology_context(
                residual.drop(columns="ES"),
                prices,
                target_zone="FR",
                radius=1,
            )
        bad = residual.copy()
        bad["storm_q50"] = 1.0
        with self.assertRaisesRegex(TopologyContextError, "Storm/MKOnline"):
            build_topology_context(
                bad,
                prices,
                target_zone="FR",
                radius=1,
            )
        shifted = prices.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=1)
        with self.assertRaisesRegex(TopologyContextError, "exactement égal"):
            build_topology_context(
                residual,
                shifted,
                target_zone="FR",
                radius=1,
            )
        gap = residual.drop(index=index[12])
        with self.assertRaisesRegex(TopologyContextError, "continu"):
            build_topology_context(
                gap,
                prices.drop(index=index[12]),
                target_zone="FR",
                radius=1,
            )


if __name__ == "__main__":
    unittest.main()
