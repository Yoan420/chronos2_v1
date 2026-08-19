from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from chronos2_hourly.hourly_contract import local_delivery_day_index
from run_chronos2_hourly import (
    _feature_inputs,
    _oof_configuration,
    _trim_to_complete_delivery_days,
    main,
    parse_args,
)


class HourlyRunnerTests(unittest.TestCase):
    def test_target_refresh_cli_can_skip_pit(self) -> None:
        argv = [
            "run_chronos2_hourly.py",
            "--full-target-refresh",
            "--skip-pit-refresh",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        self.assertTrue(args.full_target_refresh)
        self.assertTrue(args.skip_pit_refresh)
        self.assertFalse(args.full_data_refresh)
        self.assertEqual(args.zone, "FR")

    def test_zone_cli_accepts_one_explicit_market(self) -> None:
        with patch("sys.argv", ["run_chronos2_hourly.py", "--zone", "BE"]):
            args = parse_args()
        self.assertEqual(args.zone, "BE")

    def test_main_runs_end_to_end_with_external_chronos_artifacts(self) -> None:
        history = local_delivery_day_index("2026-01-01").append(
            [
                local_delivery_day_index(
                    pd.Timestamp("2026-01-01") + pd.Timedelta(days=offset)
                )
                for offset in range(1, 8)
            ]
        )
        future = local_delivery_day_index("2026-01-09")
        target = pd.Series(
            50.0 + 5.0 * np.sin(np.arange(len(history)) / 12.0),
            index=history.tz_convert("Europe/Paris"),
        )
        combined = history.append(future)
        covariates = pd.DataFrame(
            {"known_load": 30.0 + np.cos(np.arange(len(combined)) / 8.0)},
            index=combined.tz_convert("Europe/Paris"),
        )
        data = SimpleNamespace(
            zone="FR",
            timezone="Europe/Paris",
            target=target,
            model_context_covariates=covariates,
            known_future_columns=["known_load"],
            diagnostics={},
        )

        oof_index = local_delivery_day_index("2026-01-05").append(
            [
                local_delivery_day_index("2026-01-06"),
                local_delivery_day_index("2026-01-07"),
                local_delivery_day_index("2026-01-08"),
            ]
        )
        actual = pd.Series(target.to_numpy(), index=history).loc[oof_index]
        oof_origin = [
            pd.Timestamp(delivery_date).tz_localize("Europe/Paris")
            - pd.Timedelta(days=1)
            + pd.Timedelta(hours=8)
            for delivery_date in oof_index.tz_convert("Europe/Paris").date
        ]
        # Every row of a delivery day shares its D-1 08:00 forecast origin.
        oof = pd.DataFrame(
            {
                "delivery_start_utc": oof_index,
                "forecast_origin_utc": pd.DatetimeIndex(oof_origin).tz_convert(
                    "UTC"
                ),
                "q10": actual.to_numpy() - 8.0,
                "q50": actual.to_numpy() + 0.5,
                "q90": actual.to_numpy() + 8.0,
                "actual": actual.to_numpy(),
            }
        )
        live_origin = pd.Timestamp(
            "2026-01-08 08:00", tz="Europe/Paris"
        ).tz_convert("UTC")
        live = pd.DataFrame(
            {
                "delivery_start_utc": future,
                "forecast_origin_utc": live_origin,
                "q10": 42.0,
                "q50": 50.0,
                "q90": 58.0,
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            output = root / "output"
            oof_path = root / "oof.csv"
            live_path = root / "live.csv"
            config_path.write_text(
                """
model:
  seed: 42
  context_length: 24
data:
  project_root: .
  frequency: h
  forecast_origin_local_time: "08:00"
backtest:
  windows: 4
hourly:
  feature_engineering:
    target_lags: [24]
    target_rolling_windows: [24]
  oof:
    n_splits: 2
    min_train_days: 2
    gap_days: 1
    evaluation_days: 2
    ensemble_minimum_rows: 2
  lear:
    min_samples_per_hour: 1000
    alpha: 0.001
  catboost:
    backend: sklearn
    min_samples_per_hour: 1000
    iterations: 5
    depth: 2
    min_samples_leaf: 2
  ensemble:
    minimum_rows: 2
  supply_stack:
    enabled: false
output:
  directory: output
zones:
  FR:
    enabled: true
    timezone: Europe/Paris
    target:
      file: placeholder.csv
    covariates: {}
""".strip(),
                encoding="utf-8",
            )
            oof.to_csv(oof_path, index=False)
            live.to_csv(live_path, index=False)
            argv = [
                "run_chronos2_hourly.py",
                "--config",
                str(config_path),
                "--output-dir",
                str(output),
                "--chronos-oof-file",
                str(oof_path),
                "--chronos-live-file",
                str(live_path),
            ]
            with patch("sys.argv", argv), patch(
                "run_chronos2_hourly.set_reproducibility"
            ), patch(
                "run_chronos2_hourly.prepare_zone_data", return_value=data
            ):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            forecast = pd.read_csv(output / "forecast_hourly_fr.csv")
            metrics = pd.read_csv(output / "metrics_hourly.csv")
            self.assertEqual(len(forecast), 24)
            self.assertEqual(
                set(metrics["model"]),
                {"lear", "catboost", "chronos2", "ensemble"},
            )
            self.assertTrue((output / "backtest_hourly_oof.csv.gz").exists())
            self.assertTrue((output / "artifact_checksums.json").exists())
            self.assertTrue((output / "pit_feature_coverage_summary.csv").exists())

    def test_feature_inputs_keep_a_23_hour_live_day(self) -> None:
        future_index = local_delivery_day_index("2024-03-31")
        history_index = pd.date_range(
            end=future_index[0] - pd.Timedelta(hours=1),
            periods=24 * 10,
            freq="h",
            tz="UTC",
        )
        target = pd.Series(
            50.0 + np.sin(np.arange(len(history_index))),
            index=history_index.tz_convert("Europe/Paris"),
        )
        combined = history_index.append(future_index)
        covariates = pd.DataFrame(
            {"known_load": 40.0 + np.arange(len(combined)) / 100.0},
            index=combined.tz_convert("Europe/Paris"),
        )
        data = SimpleNamespace(
            target=target,
            model_context_covariates=covariates,
            known_future_columns=["known_load"],
            timezone="Europe/Paris",
        )
        config = {
            "hourly": {
                "feature_engineering": {
                    "target_lags": [24],
                    "target_rolling_windows": [24],
                },
                "supply_stack": {"enabled": False},
            }
        }

        result_target, history, future, features = _feature_inputs(data, config)

        self.assertEqual(len(future), 23)
        self.assertTrue(future.index.equals(future_index))
        self.assertTrue(result_target.index.equals(history.index))
        self.assertEqual(len(features), len(history_index) + 23)
        self.assertFalse(future.isna().any().any())

    def test_trim_and_oof_config_use_only_complete_delivery_days(self) -> None:
        complete = local_delivery_day_index("2024-10-26").append(
            [
                local_delivery_day_index("2024-10-27"),
                local_delivery_day_index("2024-10-28"),
                local_delivery_day_index("2024-10-29"),
                local_delivery_day_index("2024-10-30"),
                local_delivery_day_index("2024-10-31"),
                local_delivery_day_index("2024-11-01"),
                local_delivery_day_index("2024-11-02"),
            ]
        )
        partial = pd.DatetimeIndex([complete[0] - pd.Timedelta(hours=1)]).append(
            complete
        )
        features = pd.DataFrame({"x": np.arange(len(partial))}, index=partial)
        target = pd.Series(np.arange(len(partial), dtype=float), index=partial)

        trimmed_x, trimmed_y, days = _trim_to_complete_delivery_days(
            features,
            target,
            timezone="Europe/Paris",
        )

        self.assertEqual(days, 8)
        self.assertEqual(len(trimmed_x), len(complete))
        self.assertTrue(trimmed_x.index.equals(trimmed_y.index))
        config = {
            "backtest": {"windows": 4},
            "hourly": {
                "oof": {
                    "n_splits": 2,
                    "min_train_days": 2,
                    "gap_days": 1,
                    "evaluation_days": 2,
                    "ensemble_minimum_rows": 2,
                }
            },
        }
        oof = _oof_configuration(
            config,
            n_complete_days=days,
            timezone="Europe/Paris",
        )
        self.assertTrue(oof.split_on_delivery_days)
        self.assertEqual(oof.min_train_days, 3)
        self.assertEqual(oof.test_days, 2)
        self.assertEqual(oof.evaluation_days, 2)


if __name__ == "__main__":
    unittest.main()
