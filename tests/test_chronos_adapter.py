from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from chronos2_hourly.chronos_adapter import (
    ChronosAdapterError,
    build_delivery_plan,
    execute_chronos_live_forecast,
    execute_grouped_chronos_backtest,
    generate_delivery_plans,
    load_chronos_oof,
    make_existing_forecasting_executor,
    make_existing_live_forecast_executor,
    normalize_chronos_future,
    normalize_chronos_oof,
    run_existing_live_forecast,
)


def _raw_predictions(
    plans,
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    frames = []
    for plan in plans:
        n = plan.horizon
        actual = np.linspace(40.0, 80.0, n)
        frames.append(
            pd.DataFrame(
                {
                    timestamp_column: plan.delivery_index_utc.tz_convert(
                        "Europe/Paris"
                    ),
                    "q10": actual - 10.0,
                    "q50": actual,
                    "q90": actual + 10.0,
                    "actual": actual + 1.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class DeliveryPlanTests(unittest.TestCase):
    def test_delivery_plans_cover_dst_days_without_duplicate_utc_hours(self) -> None:
        cases = (
            ("2024-03-31", 23),
            ("2024-06-15", 24),
            ("2024-10-27", 25),
        )
        for day, expected_horizon in cases:
            with self.subTest(day=day):
                plan = build_delivery_plan(day)
                self.assertEqual(plan.horizon, expected_horizon)
                self.assertEqual(str(plan.delivery_index_utc.tz), "UTC")
                self.assertFalse(plan.delivery_index_utc.has_duplicates)
                self.assertTrue(
                    bool(
                        (
                            plan.delivery_index_utc.to_series().diff().dropna()
                            == pd.Timedelta(hours=1)
                        ).all()
                    )
                )
                self.assertTrue(plan.forecast_origin_utc < plan.delivery_start_utc)

    def test_generate_plans_is_inclusive_and_consecutive(self) -> None:
        plans = generate_delivery_plans("2024-03-30", "2024-04-01")

        self.assertEqual([plan.horizon for plan in plans], [24, 23, 24])
        joined = plans[0].delivery_index_utc.append(
            [plan.delivery_index_utc for plan in plans[1:]]
        )
        self.assertTrue(
            bool((joined[1:] - joined[:-1] == pd.Timedelta(hours=1)).all())
        )


class OOFNormalizationTests(unittest.TestCase):
    def _valid_frame(self, periods: int = 5) -> pd.DataFrame:
        delivery = pd.date_range(
            "2026-01-01",
            periods=periods,
            freq="h",
            tz="UTC",
        )
        return pd.DataFrame(
            {
                "timestamp": delivery.tz_convert("Europe/Paris"),
                "forecast_origin": pd.Timestamp("2025-12-31 07:00", tz="UTC"),
                "0.1": np.arange(periods, dtype=float),
                "0.5": np.arange(periods, dtype=float) + 1.0,
                "0.9": np.arange(periods, dtype=float) + 2.0,
                "target": np.arange(periods, dtype=float) + 1.5,
            }
        )

    def test_normalizes_aliases_to_strict_utc_contract(self) -> None:
        result = normalize_chronos_oof(self._valid_frame())

        self.assertEqual(str(result.index.tz), "UTC")
        self.assertEqual(result.index.name, "delivery_start_utc")
        self.assertEqual(
            list(result.columns),
            ["forecast_origin_utc", "q10", "q50", "q90", "actual"],
        )
        self.assertTrue(bool((result["forecast_origin_utc"] < result.index).all()))

    def test_rejects_duplicate_gap_naive_and_late_origin(self) -> None:
        duplicate = pd.concat(
            [self._valid_frame(), self._valid_frame().iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ChronosAdapterError, "duplicate"):
            normalize_chronos_oof(duplicate)

        gap = self._valid_frame().drop(index=2).reset_index(drop=True)
        with self.assertRaisesRegex(ChronosAdapterError, "gap"):
            normalize_chronos_oof(gap)

        naive = self._valid_frame()
        naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
        with self.assertRaisesRegex(ChronosAdapterError, "timezone-naive"):
            normalize_chronos_oof(naive)

        late = self._valid_frame()
        late["forecast_origin"] = late["timestamp"]
        with self.assertRaisesRegex(ChronosAdapterError, "strictly before"):
            normalize_chronos_oof(late)

    def test_rejects_crossing_or_nonfinite_quantiles(self) -> None:
        crossing = self._valid_frame()
        crossing.loc[2, "0.1"] = 100.0
        with self.assertRaisesRegex(ChronosAdapterError, "Crossing"):
            normalize_chronos_oof(crossing)

        missing = self._valid_frame()
        missing.loc[1, "target"] = np.nan
        with self.assertRaisesRegex(ChronosAdapterError, "non-finite"):
            normalize_chronos_oof(missing)

    def test_loads_csv_and_parquet_then_applies_same_validation(self) -> None:
        frame = self._valid_frame()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "chronos_oof.csv"
            frame.to_csv(csv_path, index=False)
            csv_result = load_chronos_oof(csv_path)

            parquet_path = root / "chronos_oof.parquet"
            try:
                frame.to_parquet(parquet_path, index=False)
            except (ImportError, ModuleNotFoundError):
                self.skipTest("No Parquet engine is installed.")
            parquet_result = load_chronos_oof(parquet_path)

        assert_frame_equal(csv_result, parquet_result)


class GroupExecutionTests(unittest.TestCase):
    def test_groups_calls_by_horizon_and_restores_chronological_utc(self) -> None:
        plans = generate_delivery_plans("2024-03-30", "2024-04-01")
        calls: list[tuple[int, int]] = []

        def fake_executor(*, plans, horizon):
            calls.append((horizon, len(plans)))
            return _raw_predictions(plans)

        result = execute_grouped_chronos_backtest(plans, fake_executor)

        self.assertEqual(calls, [(23, 1), (24, 2)])
        self.assertEqual(len(result), 24 + 23 + 24)
        self.assertEqual(str(result.index.tz), "UTC")
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertTrue(bool((result["forecast_origin_utc"] < result.index).all()))

    def test_group_executor_must_return_exact_plan_coverage(self) -> None:
        plans = generate_delivery_plans("2024-03-30", "2024-03-31")

        def incomplete_executor(*, plans, horizon):
            return _raw_predictions(plans).iloc[:-1]

        with self.assertRaisesRegex(ChronosAdapterError, "exactly cover"):
            execute_grouped_chronos_backtest(plans, incomplete_executor)

    def test_existing_forecasting_factory_delegates_fixed_horizon_calls(self) -> None:
        plan = build_delivery_plan("2024-06-15")
        target_index = pd.date_range(
            plan.delivery_start_utc - pd.Timedelta(days=10),
            plan.delivery_index_utc[-1],
            freq="h",
            tz="UTC",
        )
        data = SimpleNamespace(
            target=pd.Series(np.arange(len(target_index)), index=target_index)
        )
        calls = []

        class FakeForecasting:
            @staticmethod
            def run_backtest_variant(**kwargs):
                calls.append(kwargs)
                origin = kwargs["origins"][0]
                horizon = kwargs["horizon"]
                index = data.target.index[origin : origin + horizon]
                values = np.arange(horizon, dtype=float) + 50.0
                return pd.DataFrame(
                    {
                        "timestamp": index,
                        "q10": values - 5.0,
                        "q50": values,
                        "q90": values + 5.0,
                        "actual": values + 1.0,
                    }
                )

        executor = make_existing_forecasting_executor(
            data=data,
            runtime=object(),
            context_length=48,
            origin_batch_size=4,
            model_batch_size=32,
            forecasting_module=FakeForecasting,
        )
        result = execute_grouped_chronos_backtest((plan,), executor)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["horizon"], 24)
        self.assertEqual(calls[0]["origins"], [10 * 24])
        self.assertEqual(len(result), 24)


class LiveExecutionTests(unittest.TestCase):
    def test_future_normalization_needs_no_actual_and_preserves_dst_plan(self) -> None:
        for day, expected_horizon in (
            ("2024-03-31", 23),
            ("2024-10-27", 25),
        ):
            with self.subTest(day=day):
                plan = build_delivery_plan(day)
                raw = _raw_predictions((plan,)).drop(columns=["actual"])

                result = normalize_chronos_future(raw, plan)

                self.assertEqual(len(result), expected_horizon)
                self.assertEqual(
                    list(result.columns),
                    ["forecast_origin_utc", "q10", "q50", "q90"],
                )
                self.assertTrue(result.index.equals(plan.delivery_index_utc))
                self.assertTrue(
                    bool(
                        result["forecast_origin_utc"]
                        .eq(plan.forecast_origin_utc)
                        .all()
                    )
                )

    def test_future_normalization_rejects_missing_or_extra_hour(self) -> None:
        plan = build_delivery_plan("2024-10-27")
        incomplete = _raw_predictions((plan,)).drop(columns=["actual"]).iloc[:-1]
        with self.assertRaisesRegex(ChronosAdapterError, "exactly cover"):
            normalize_chronos_future(incomplete, plan)

        extra = pd.concat(
            [
                _raw_predictions((plan,)).drop(columns=["actual"]),
                pd.DataFrame(
                    {
                        "timestamp": [plan.delivery_index_utc[-1] + pd.Timedelta("1h")],
                        "q10": [1.0],
                        "q50": [2.0],
                        "q90": [3.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ChronosAdapterError, "exactly cover"):
            normalize_chronos_future(extra, plan)

    def test_injected_live_executor_receives_variable_horizon(self) -> None:
        plan = build_delivery_plan("2024-03-31")
        calls = []

        def fake_executor(*, plan, horizon):
            calls.append((plan.delivery_date, horizon))
            return _raw_predictions((plan,)).drop(columns=["actual"])

        result = execute_chronos_live_forecast(plan, fake_executor)

        self.assertEqual(calls, [(plan.delivery_date, 23)])
        self.assertEqual(len(result), 23)

    def test_existing_live_wrapper_calls_legacy_runner_without_model(self) -> None:
        for day, expected_horizon in (
            ("2024-03-31", 23),
            ("2024-06-15", 24),
            ("2024-10-27", 25),
        ):
            with self.subTest(day=day):
                plan = build_delivery_plan(day)
                target_index = pd.date_range(
                    plan.delivery_start_utc - pd.Timedelta(days=5),
                    plan.delivery_start_utc - pd.Timedelta(hours=1),
                    freq="h",
                    tz="UTC",
                )
                data = SimpleNamespace(
                    target=pd.Series(
                        np.arange(len(target_index), dtype=float),
                        index=target_index,
                    )
                )
                calls = []

                class FakeForecasting:
                    @staticmethod
                    def run_live_forecast_variant(**kwargs):
                        calls.append(kwargs)
                        horizon = kwargs["horizon"]
                        future = pd.date_range(
                            data.target.index[-1] + pd.Timedelta(hours=1),
                            periods=horizon,
                            freq="h",
                            tz="UTC",
                        )
                        values = np.arange(horizon, dtype=float) + 50.0
                        return pd.DataFrame(
                            {
                                "timestamp": future,
                                "q10": values - 5.0,
                                "q50": values,
                                "q90": values + 5.0,
                            }
                        )

                result = run_existing_live_forecast(
                    plan,
                    data=data,
                    runtime=object(),
                    context_length=48,
                    model_batch_size=32,
                    forecasting_module=FakeForecasting,
                )

                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["horizon"], expected_horizon)
                self.assertEqual(len(result), expected_horizon)
                self.assertTrue(result.index.equals(plan.delivery_index_utc))

    def test_live_factory_rejects_target_not_ending_before_plan(self) -> None:
        plan = build_delivery_plan("2024-06-15")
        bad_index = pd.date_range(
            plan.delivery_start_utc - pd.Timedelta(days=2),
            plan.delivery_start_utc - pd.Timedelta(hours=2),
            freq="h",
            tz="UTC",
        )
        data = SimpleNamespace(target=pd.Series(1.0, index=bad_index))

        class MustNotRun:
            @staticmethod
            def run_live_forecast_variant(**kwargs):
                raise AssertionError("Legacy runner must not be called.")

        executor = make_existing_live_forecast_executor(
            data=data,
            runtime=object(),
            context_length=48,
            model_batch_size=32,
            forecasting_module=MustNotRun,
        )
        with self.assertRaisesRegex(ChronosAdapterError, "one hour before"):
            execute_chronos_live_forecast(plan, executor)


if __name__ == "__main__":
    unittest.main()
