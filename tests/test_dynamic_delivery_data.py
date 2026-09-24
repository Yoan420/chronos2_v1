from __future__ import annotations

import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from chronos2_modular.common import SeriesSpec, ZoneConfig
from chronos2_modular.data import (
    _clip_target_to_operational_cutoff,
    prepare_zone_data,
)
from chronos2_modular.forecasting import future_proxy_frame


class DynamicDeliveryDataTests(unittest.TestCase):
    def test_quarter_hour_cutoff_keeps_the_fourth_mtu_of_2300(self) -> None:
        index = pd.date_range(
            "2024-04-01 00:00",
            "2024-04-01 23:45",
            freq="15min",
            tz="Europe/Paris",
        )
        raw = pd.Series(np.arange(len(index), dtype=float), index=index)
        zone = ZoneConfig(
            zone="FR",
            timezone="Europe/Paris",
            target=SeriesSpec(alias="target", series="price_qh"),
            covariates={},
        )
        config = {
            "data": {
                "runtime_as_of": "2024-04-01T08:00:00+02:00",
                "target_end_policy": "current_day_end",
                "target_input_resolution": "quarter_hour",
                "frequency": "h",
            }
        }

        clipped = _clip_target_to_operational_cutoff(
            raw,
            zone,
            zone.target,
            config,
            require_complete=True,
        )

        self.assertEqual(len(clipped), 96)
        self.assertEqual(clipped.index[-1].minute, 45)

    def _prepare_for_last_day(self, last_day: str):
        last_midnight = pd.Timestamp(last_day, tz="Europe/Paris")
        next_midnight = last_midnight + pd.DateOffset(days=1)
        index = pd.date_range(
            last_midnight - pd.DateOffset(days=15),
            next_midnight,
            inclusive="left",
            freq="h",
        )
        target = pd.Series(np.arange(len(index), dtype=float), index=index)
        covariate_index = pd.date_range(
            index[0],
            next_midnight + pd.DateOffset(days=2),
            inclusive="left",
            freq="h",
        )
        covariate = pd.Series(1.0, index=covariate_index)
        zone = ZoneConfig(
            zone="FR",
            timezone="Europe/Paris",
            target=SeriesSpec(alias="target", series="price"),
            covariates={
                "forecast": SeriesSpec(
                    alias="forecast",
                    series="forecast",
                    known_future=True,
                    future_strategies=("oracle",),
                    fill_method="none",
                    minimum_coverage=1.0,
                )
            },
        )
        config = {
            "data": {
                "frequency": "h",
                "target_input_resolution": "hourly",
                "target_end_policy": "full_series",
                "target_interpolation_limit": 0,
                "dynamic_delivery_day_horizon": True,
                "require_all_covariates": True,
                "require_complete_future_covariates": True,
            }
        }

        def loader(_zone, spec, *_args, **_kwargs):
            return (
                (target if spec.alias == "target" else covariate),
                {"source": "test"},
            )

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        with patch(
            "chronos2_modular.data.load_input_series",
            side_effect=loader,
        ):
            return prepare_zone_data(
                zone,
                config,
                Path(temporary.name),
                False,
                Path(temporary.name) / "output",
            )

    def test_spring_delivery_day_has_23_future_hours(self) -> None:
        data = self._prepare_for_last_day("2024-03-30")
        future = data.model_context_covariates.index.difference(data.target.index)
        self.assertEqual(len(future), 23)
        self.assertEqual(
            future.tz_convert("Europe/Paris").normalize().nunique(),
            1,
        )

    def test_autumn_delivery_day_has_25_future_hours(self) -> None:
        data = self._prepare_for_last_day("2024-10-26")
        future = data.model_context_covariates.index.difference(data.target.index)
        self.assertEqual(len(future), 25)
        self.assertEqual(future.tz_convert("Europe/Paris").hour.tolist().count(2), 2)

    def test_normal_delivery_day_has_24_known_future_oracle_values(self) -> None:
        data = self._prepare_for_last_day("2024-04-01")
        future = data.model_context_covariates.index.difference(data.target.index)

        proxy = future_proxy_frame(data, future, len(data.target))

        self.assertEqual(len(future), 24)
        self.assertEqual(proxy["known_forecast_oracle"].notna().sum(), 24)
        np.testing.assert_array_equal(
            proxy["known_forecast_oracle"].to_numpy(),
            np.ones(24, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
