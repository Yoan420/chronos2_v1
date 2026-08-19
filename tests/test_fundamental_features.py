from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal

from chronos2_hourly.fundamental_features import (
    SupplyStackColumns,
    SupplyStackParameters,
    build_fr_supply_stack_features,
    required_supply_stack_columns,
)


def _input_frame(periods: int = 8) -> pd.DataFrame:
    index = pd.date_range(
        "2026-10-24 20:00",
        periods=periods,
        freq="h",
        tz="Europe/Paris",
    )
    step = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "load_forecast": 100.0 + step,
            "wind_forecast": 20.0 + 0.5 * step,
            "solar_forecast": 10.0 - 0.25 * step,
            "nuclear_available": np.full(periods, 55.0),
            "thermal_available": np.full(periods, 25.0),
            "hydro_firm": np.full(periods, 5.0),
            "import_headroom": np.full(periods, 10.0),
            "export_headroom": np.full(periods, 12.0),
            "pumping_capacity": np.full(periods, 8.0),
            "nuclear_minimum": np.full(periods, 40.0),
            "run_of_river": np.full(periods, 3.0),
            "ttf_eur_mwh_th": np.full(periods, 30.0),
            "eua_eur_tco2": np.full(periods, 80.0),
            "api2_usd_tonne": np.full(periods, 120.0),
            "eurusd_usd_per_eur": np.full(periods, 1.2),
        },
        index=index,
    )


class FundamentalFeatureTests(unittest.TestCase):
    def test_core_economic_formulas_and_index_contract(self) -> None:
        frame = _input_frame()
        original = frame.copy(deep=True)
        params = SupplyStackParameters(
            ccgt_efficiency=0.60,
            ocgt_efficiency=0.40,
            coal_efficiency=0.40,
            ccgt_emission_tco2_mwh=0.35,
            ocgt_emission_tco2_mwh=0.55,
            coal_emission_tco2_mwh=0.90,
            ccgt_vom_eur_mwh=3.0,
            ocgt_vom_eur_mwh=5.0,
            coal_vom_eur_mwh=4.0,
            coal_mwh_th_per_tonne=8.0,
        )

        result = build_fr_supply_stack_features(
            frame,
            parameters=params,
            rolling_windows=(3,),
        )

        assert_index_equal(result.index, frame.index, exact=True)
        assert_frame_equal(frame, original)
        self.assertAlmostEqual(result["residual_load"].iloc[0], 70.0)
        self.assertAlmostEqual(result["firm_available_supply"].iloc[0], 95.0)
        self.assertAlmostEqual(result["firm_margin"].iloc[0], 25.0)
        self.assertAlmostEqual(result["negative_pressure"].iloc[0], -47.0)
        self.assertAlmostEqual(
            result["coal_fuel_cost_eur_mwh_th"].iloc[0],
            12.5,
        )
        self.assertAlmostEqual(result["ccgt_marginal_cost"].iloc[0], 81.0)
        self.assertAlmostEqual(result["ocgt_marginal_cost"].iloc[0], 124.0)
        self.assertAlmostEqual(result["coal_marginal_cost"].iloc[0], 107.25)

    def test_future_mutation_cannot_change_past_features(self) -> None:
        frame = _input_frame(periods=10)
        baseline = build_fr_supply_stack_features(
            frame,
            rolling_windows=(3, 6),
        )
        changed = frame.copy()
        changed.iloc[7:, :] = changed.iloc[7:, :] * 100.0
        mutated = build_fr_supply_stack_features(
            changed,
            rolling_windows=(3, 6),
        )

        assert_frame_equal(
            baseline.iloc[:7],
            mutated.iloc[:7],
            check_exact=True,
        )
        self.assertTrue(baseline["residual_load_ramp_1"].iloc[0] != baseline["residual_load_ramp_1"].iloc[0])
        self.assertTrue(
            baseline["residual_load_trailing_mean_3"].iloc[:2].isna().all()
        )

    def test_nan_propagates_without_implicit_fill(self) -> None:
        frame = _input_frame()
        frame.iloc[3, frame.columns.get_loc("load_forecast")] = np.nan

        result = build_fr_supply_stack_features(
            frame,
            rolling_windows=(3,),
            nan_policy="propagate",
        )

        self.assertTrue(np.isnan(result["residual_load"].iloc[3]))
        self.assertTrue(np.isnan(result["firm_margin"].iloc[3]))
        self.assertTrue(np.isnan(result["residual_load_ramp_1"].iloc[3]))
        self.assertTrue(np.isnan(result["residual_load_ramp_1"].iloc[4]))
        self.assertTrue(
            result["residual_load_trailing_mean_3"].iloc[3:6].isna().all()
        )
        self.assertFalse(np.isnan(result["residual_load"].iloc[4]))
        self.assertAlmostEqual(result["ccgt_marginal_cost"].iloc[3], 30.0 / 0.58 + 80.0 * 0.36 + 3.0)

    def test_raise_policy_reports_missing_inputs(self) -> None:
        frame = _input_frame()
        frame.iloc[2, frame.columns.get_loc("nuclear_available")] = np.nan

        with self.assertRaisesRegex(
            ValueError,
            r"nuclear_available=1",
        ):
            build_fr_supply_stack_features(frame, nan_policy="raise")

    def test_direct_coal_price_and_custom_columns(self) -> None:
        frame = _input_frame().rename(
            columns={
                "load_forecast": "fr_load",
                "ttf_eur_mwh_th": "gas",
                "eua_eur_tco2": "carbon",
            }
        )
        frame["coal_direct"] = 14.0
        frame = frame.drop(columns=["api2_usd_tonne", "eurusd_usd_per_eur"])
        columns = SupplyStackColumns(
            load_forecast="fr_load",
            ttf_eur_mwh_th="gas",
            eua_eur_tco2="carbon",
            api2_usd_tonne=None,
            eurusd_usd_per_eur=None,
            coal_eur_mwh_th="coal_direct",
        )

        result = build_fr_supply_stack_features(
            frame,
            columns=columns,
            rolling_windows=(),
        )

        self.assertAlmostEqual(
            result["coal_fuel_cost_eur_mwh_th"].iloc[0],
            14.0,
        )
        self.assertIn("coal_direct", required_supply_stack_columns(columns))
        self.assertNotIn("api2_usd_tonne", required_supply_stack_columns(columns))

    def test_missing_or_malformed_columns_fail_loudly(self) -> None:
        missing = _input_frame().drop(columns=["hydro_firm"])
        with self.assertRaisesRegex(KeyError, "hydro_firm"):
            build_fr_supply_stack_features(missing)

        malformed = _input_frame()
        malformed["ttf_eur_mwh_th"] = malformed["ttf_eur_mwh_th"].astype(
            object
        )
        malformed.iloc[1, malformed.columns.get_loc("ttf_eur_mwh_th")] = "bad"
        with self.assertRaisesRegex(TypeError, "ttf_eur_mwh_th"):
            build_fr_supply_stack_features(malformed)

    def test_dst_fallback_keeps_both_local_hours(self) -> None:
        frame = _input_frame(periods=10)
        self.assertEqual(frame.index[6].hour, frame.index[7].hour)
        self.assertNotEqual(frame.index[6].utcoffset(), frame.index[7].utcoffset())

        result = build_fr_supply_stack_features(
            frame,
            rolling_windows=(3,),
        )

        assert_index_equal(result.index, frame.index, exact=True)
        self.assertEqual(len(result), len(frame))


if __name__ == "__main__":
    unittest.main()
