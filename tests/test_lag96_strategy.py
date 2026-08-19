from __future__ import annotations

import sys
import tempfile
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

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_modular.common import (
    SUPPORTED_FUTURE_LAG_HOURS,
    SeriesSpec,
    ZoneConfig,
    ZoneData,
    build_zone_configs,
    parse_future_lag_hours,
    parse_series_spec,
)
from chronos2_modular.data import prepare_zone_data
from chronos2_modular.feature_selection import filter_zone_data
from chronos2_modular.forecasting import future_proxy_frame


PARIS = "Europe/Paris"
LAG_HOURS = 96
PROXY_COLUMN = "known_signal_lag96"


def _observed_net_exports_config(strategy: str) -> dict:
    return {
        "data": {"default_fill_limit": 0},
        "zones": {
            "FR": {
                "enabled": True,
                "timezone": PARIS,
                "target": {"series": "price"},
                "covariates": {
                    "fr_net_exports": {
                        "enabled": True,
                        "series": (
                            "power.net.fr.exports.core.entsoe.hourly.gw.obs"
                        ),
                        "include_base_context": False,
                        "future": {
                            "known_future": False,
                            "strategies": [strategy],
                        },
                    }
                },
            }
        },
    }


def _target_and_signal(last_context_day: str) -> tuple[pd.Series, pd.Series]:
    last_midnight = pd.Timestamp(last_context_day, tz=PARIS)
    next_midnight = last_midnight + pd.DateOffset(days=1)
    index = pd.date_range(
        last_midnight - pd.DateOffset(days=15),
        next_midnight,
        inclusive="left",
        freq="h",
    )
    target = pd.Series(
        np.arange(len(index), dtype=float),
        index=index,
        name="target",
    )
    # Unique timestamp-derived values make any accidental local-day shift
    # immediately visible, including around both DST transitions.
    utc_hours = (
        index.tz_convert("UTC").asi8 // pd.Timedelta(hours=1).value
    ).astype(float)
    signal = pd.Series(utc_hours, index=index, name="signal")
    return target, signal


def _lag96_zone(*, minimum_coverage: float = 1.0) -> ZoneConfig:
    return ZoneConfig(
        zone="FR",
        timezone=PARIS,
        target=SeriesSpec(alias="target", series="price"),
        covariates={
            "signal": SeriesSpec(
                alias="signal",
                series="signal.obs",
                include_base_context=False,
                fill_method="none",
                fill_limit=0,
                minimum_coverage=minimum_coverage,
                future_strategies=("lag96",),
            )
        },
        include_calendar=False,
    )


def _lag96_config() -> dict:
    return {
        "model": {"horizon": 24},
        "data": {
            "frequency": "h",
            "target_input_resolution": "hourly",
            "target_end_policy": "full_series",
            "target_interpolation_limit": 0,
            "dynamic_delivery_day_horizon": True,
            "require_all_covariates": True,
            "require_complete_future_covariates": True,
            "minimum_future_coverage": 1.0,
        },
    }


def _prepare_lag96(
    last_context_day: str,
    *,
    signal_transform=None,
    minimum_coverage: float = 1.0,
) -> tuple[ZoneData, pd.Series]:
    target, signal = _target_and_signal(last_context_day)
    if signal_transform is not None:
        signal = signal_transform(signal)

    def loader(_zone, spec, *_args, **_kwargs):
        return (
            (target if spec.alias == "target" else signal),
            {"source": "test"},
        )

    temporary = tempfile.TemporaryDirectory()
    with temporary:
        with patch(
            "chronos2_modular.data.load_input_series",
            side_effect=loader,
        ):
            result = prepare_zone_data(
                _lag96_zone(minimum_coverage=minimum_coverage),
                _lag96_config(),
                Path(temporary.name),
                False,
                Path(temporary.name) / "output",
            )
    return result, signal


class Lag96ParsingTests(unittest.TestCase):
    def test_lag96_is_parsed_as_a_supported_physical_lag(self) -> None:
        spec = parse_series_spec(
            "signal",
            {
                "series": "signal.obs",
                "future": {
                    "known_future": False,
                    "strategies": ["LaG96"],
                },
            },
            0,
        )

        self.assertEqual(spec.future_strategies, ("lag96",))
        self.assertEqual(parse_future_lag_hours("lag96"), 96)

    def test_unsupported_or_malformed_lags_are_rejected(self) -> None:
        invalid = (
            "lag0",
            "lag72",
            "lag096",
            "lag-96",
            "lag96.0",
            "lag_96",
            "lagfoo",
        )
        for strategy in invalid:
            with self.subTest(strategy=strategy):
                self.assertIsNone(parse_future_lag_hours(strategy))
                with self.assertRaisesRegex(
                    ValueError,
                    "strat.gies futures inconnues",
                ):
                    parse_series_spec(
                        "signal",
                        {
                            "series": "signal.obs",
                            "future": {"strategies": [strategy]},
                        },
                        0,
                    )

    def test_every_supported_observed_net_export_lag_below_48_is_rejected(
        self,
    ) -> None:
        unsafe = [hours for hours in SUPPORTED_FUTURE_LAG_HOURS if hours < 48]
        self.assertTrue(unsafe, "Le test doit exercer au moins un lag < 48 h.")
        for hours in unsafe:
            with self.subTest(hours=hours):
                with self.assertRaisesRegex(ValueError, f"lag{hours}"):
                    build_zone_configs(
                        _observed_net_exports_config(f"lag{hours}"),
                        ["FR"],
                        None,
                        None,
                    )

    def test_observed_net_export_lag48_and_lag96_are_allowed(self) -> None:
        for strategy in ("lag48", "lag96"):
            with self.subTest(strategy=strategy):
                zones = build_zone_configs(
                    _observed_net_exports_config(strategy),
                    ["FR"],
                    None,
                    None,
                )
                self.assertEqual(
                    zones[0].covariates[
                        "fr_net_exports"
                    ].future_strategies,
                    (strategy,),
                )


class Lag96PreparationTests(unittest.TestCase):
    def test_prepare_zone_data_uses_exact_96_physical_hours_on_24_23_25_days(
        self,
    ) -> None:
        cases = (
            ("2024-04-01", 24),
            ("2024-03-30", 23),
            ("2024-10-26", 25),
        )
        for last_context_day, expected_hours in cases:
            with self.subTest(
                last_context_day=last_context_day,
                expected_hours=expected_hours,
            ):
                data, signal = _prepare_lag96(last_context_day)
                future = data.model_context_covariates.index.difference(
                    data.target.index
                )
                actual = data.model_context_covariates.loc[
                    future, PROXY_COLUMN
                ]
                source_index = future - pd.Timedelta(hours=LAG_HOURS)
                expected = signal.reindex(source_index)

                self.assertEqual(len(future), expected_hours)
                self.assertFalse(actual.isna().any())
                np.testing.assert_array_equal(
                    actual.to_numpy(),
                    expected.to_numpy(dtype=np.float32),
                )
                self.assertTrue(
                    (
                        future.tz_convert("UTC")
                        - source_index.tz_convert("UTC")
                        == pd.Timedelta(hours=LAG_HOURS)
                    ).all()
                )
                if expected_hours == 25:
                    self.assertEqual(
                        future.tz_convert(PARIS).hour.tolist().count(2),
                        2,
                    )

    def test_incomplete_lag96_source_tail_blocks_live_forecast(self) -> None:
        delivery_day = "2024-04-02"
        future = local_delivery_day_index(
            delivery_day,
            timezone=PARIS,
        ).tz_convert(PARIS)
        required_sources = future - pd.Timedelta(hours=LAG_HOURS)

        def truncate_tail(signal: pd.Series) -> pd.Series:
            return signal.loc[signal.index <= required_sources[-2]]

        with self.assertRaisesRegex(
            ValueError,
            f"{PROXY_COLUMN}.*95.8%",
        ):
            _prepare_lag96(
                "2024-04-01",
                signal_transform=truncate_tail,
                minimum_coverage=0.50,
            )


class Lag96ForecastingTests(unittest.TestCase):
    def test_future_proxy_frame_reindexes_exactly_96_physical_hours(self) -> None:
        future = local_delivery_day_index(
            "2024-10-27",
            timezone=PARIS,
        ).tz_convert(PARIS)
        source = future - pd.Timedelta(hours=LAG_HOURS)
        signal_index = pd.date_range(
            source[0] - pd.Timedelta(hours=4),
            future[-1],
            freq="h",
        )
        signal = pd.Series(
            np.arange(len(signal_index), dtype=float),
            index=signal_index,
        )
        context_index = signal_index[signal_index < future[0]]
        data = ZoneData(
            zone="FR",
            timezone=PARIS,
            frequency="h",
            target=pd.Series(50.0, index=context_index),
            covariates=pd.DataFrame({"signal": signal}),
            model_context_covariates=pd.DataFrame(index=signal_index),
            known_future_columns=[PROXY_COLUMN],
            coverage=pd.DataFrame(),
            input_manifest=pd.DataFrame(),
            diagnostics={},
        )

        actual = future_proxy_frame(
            data,
            future,
            origin_position=len(context_index),
        )[PROXY_COLUMN]
        expected = signal.reindex(source)

        self.assertEqual(len(actual), 25)
        self.assertEqual(actual.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(
            actual.to_numpy(),
            expected.to_numpy(dtype=np.float32),
        )

    def test_feature_selection_keeps_raw_aliases_for_lag48_and_lag96(
        self,
    ) -> None:
        index = pd.date_range(
            "2026-01-01",
            periods=4,
            freq="h",
            tz=PARIS,
        )
        data = ZoneData(
            zone="FR",
            timezone=PARIS,
            frequency="h",
            target=pd.Series([1.0, 2.0, 3.0, 4.0], index=index),
            covariates=pd.DataFrame(
                {
                    "exports": [10.0, 11.0, 12.0, 13.0],
                    "nuclear": [20.0, 21.0, 22.0, 23.0],
                    "unused": [30.0, 31.0, 32.0, 33.0],
                },
                index=index,
            ),
            model_context_covariates=pd.DataFrame(
                {
                    "known_exports_lag48": [1.0, 2.0, 3.0, 4.0],
                    "known_nuclear_lag96": [5.0, 6.0, 7.0, 8.0],
                    "known_unused_oracle": [9.0, 10.0, 11.0, 12.0],
                },
                index=index,
            ),
            known_future_columns=[
                "known_exports_lag48",
                "known_nuclear_lag96",
                "known_unused_oracle",
            ],
            coverage=pd.DataFrame(),
            input_manifest=pd.DataFrame(),
            diagnostics={},
        )
        config = {
            "feature_selection": {
                "enabled": True,
                "selected_groups": ["lagged"],
                "group_definitions": {
                    "lagged": {
                        "patterns": [
                            "known_*_lag48",
                            "known_*_lag96",
                        ]
                    }
                },
            }
        }

        result = filter_zone_data(data, config)

        self.assertEqual(
            list(result.model_context_covariates.columns),
            ["known_exports_lag48", "known_nuclear_lag96"],
        )
        self.assertEqual(
            result.known_future_columns,
            ["known_exports_lag48", "known_nuclear_lag96"],
        )
        self.assertEqual(
            list(result.covariates.columns),
            ["exports", "nuclear"],
        )


if __name__ == "__main__":
    unittest.main()
