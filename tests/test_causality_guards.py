from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from chronos2_modular.common import (
    SeriesSpec,
    ZoneConfig,
    build_zone_configs,
)
from chronos2_modular.data import (
    prepare_zone_data,
    read_pit_vintage_series,
)
from chronos2_order_signals.labels import build_ex_post_labels


def _minimal_zone_config(net_export_payload: dict) -> dict:
    return {
        "data": {"default_fill_limit": 0},
        "zones": {
            "FR": {
                "enabled": True,
                "timezone": "Europe/Paris",
                "target": {"series": "price"},
                "covariates": {"fr_net_exports": net_export_payload},
            }
        },
    }


class CausalityGuardTests(unittest.TestCase):
    def test_observed_net_exports_lag24_is_rejected(self) -> None:
        config = _minimal_zone_config(
            {
                "enabled": True,
                "series": "power.net.fr.exports.core.entsoe.hourly.gw.obs",
                "include_base_context": False,
                "future": {"known_future": False, "strategies": ["lag24"]},
            }
        )
        with self.assertRaisesRegex(ValueError, "lag24"):
            build_zone_configs(config, ["FR"], None, None)

    def test_observed_net_exports_lag48_without_raw_context_is_allowed(self) -> None:
        config = _minimal_zone_config(
            {
                "enabled": True,
                "series": "power.net.fr.exports.core.entsoe.hourly.gw.obs",
                "include_base_context": False,
                "future": {"known_future": False, "strategies": ["lag48"]},
            }
        )
        zones = build_zone_configs(config, ["FR"], None, None)
        spec = zones[0].covariates["fr_net_exports"]
        self.assertEqual(spec.future_strategies, ("lag48",))
        self.assertFalse(spec.include_base_context)

    def test_net_exports_raw_context_is_not_sent_to_the_model(self) -> None:
        index = pd.date_range(
            "2026-01-01", periods=120, freq="h", tz="Europe/Paris"
        )
        target = pd.Series(np.arange(len(index), dtype=float), index=index)
        exports = pd.Series(np.linspace(-4.0, 4.0, len(index)), index=index)
        zone = ZoneConfig(
            zone="FR",
            timezone="Europe/Paris",
            target=SeriesSpec(alias="target", series="price"),
            covariates={
                "fr_net_exports": SeriesSpec(
                    alias="fr_net_exports",
                    series="power.net.fr.exports.core.entsoe.hourly.gw.obs",
                    include_base_context=False,
                    fill_method="none",
                    minimum_coverage=1.0,
                    future_strategies=("lag48",),
                )
            },
        )
        config = {
            "model": {"horizon": 24},
            "data": {
                "frequency": "h",
                "target_end_policy": "full_series",
                "target_interpolation_limit": 0,
                "require_all_covariates": True,
                "require_complete_future_covariates": True,
            },
        }

        def fake_loader(_zone, spec, *_args, **_kwargs):
            return (
                (target if spec.alias == "target" else exports),
                {"source": "test"},
            )

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "chronos2_modular.data.load_input_series",
                side_effect=fake_loader,
            ):
                data = prepare_zone_data(
                    zone,
                    config,
                    Path(directory),
                    False,
                    Path(directory) / "output",
                )

        self.assertNotIn("fr_net_exports", data.model_context_covariates)
        self.assertIn(
            "known_fr_net_exports_lag48",
            data.model_context_covariates,
        )

    def test_revision_after_cutoff_is_never_selected(self) -> None:
        delivery = pd.Timestamp("2026-01-02 12:00", tz="UTC")
        frame = pd.DataFrame(
            {
                "delivery_utc": [delivery, delivery],
                "availability_utc": pd.to_datetime(
                    ["2026-01-01 06:00Z", "2026-01-01 06:30Z"], utc=True
                ),
                "revision_utc": pd.to_datetime(
                    ["2026-01-01 06:00Z", "2026-01-01 08:00Z"], utc=True
                ),
                "value": [10.0, 999.0],
            }
        )
        config = {
            "data": {
                "forecast_origin_local_time": "08:00",
                "revision_policy": "latest_before_asof",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast.parquet"
            try:
                frame.to_parquet(path, index=False)
            except (ImportError, ModuleNotFoundError):
                frame.to_pickle(path)
                parquet_patch = patch.object(
                    pd,
                    "read_parquet",
                    side_effect=lambda source, **_kwargs: pd.read_pickle(source),
                )
            else:
                parquet_patch = patch.object(
                    pd,
                    "read_parquet",
                    wraps=pd.read_parquet,
                )
            with parquet_patch:
                series, metadata = read_pit_vintage_series(
                    path,
                    SeriesSpec(alias="forecast"),
                    "Europe/Paris",
                    config,
                )
        self.assertEqual(series.iloc[0], 10.0)
        self.assertEqual(metadata["revision_cutoff_violations"], 0)

    def test_next_day_first_price_cannot_change_previous_day_labels(self) -> None:
        index = pd.date_range(
            "2026-01-05", periods=48, freq="h", tz="Europe/Paris"
        )
        values = 50.0 + 10.0 * np.sin(2.0 * np.pi * index.hour / 24.0)
        base = pd.Series(values, index=index, name="target")
        changed = base.copy()
        changed.iloc[24] = 1000.0

        first = build_ex_post_labels(base)
        second = build_ex_post_labels(changed)
        day = pd.Timestamp("2026-01-05", tz="Europe/Paris")
        columns = [column for column in first if column != "delivery_day"]
        pd.testing.assert_frame_equal(
            first.loc[first["delivery_day"].eq(day), columns],
            second.loc[second["delivery_day"].eq(day), columns],
        )

    def test_require_all_covariates_blocks_low_coverage(self) -> None:
        index = pd.date_range(
            "2026-01-01", periods=96, freq="h", tz="Europe/Paris"
        )
        target = pd.Series(np.arange(len(index), dtype=float), index=index)
        covariate = pd.Series(
            np.arange(12, dtype=float),
            index=index[:12],
        )
        zone = ZoneConfig(
            zone="FR",
            timezone="Europe/Paris",
            target=SeriesSpec(alias="target", series="price"),
            covariates={
                "nuclear": SeriesSpec(
                    alias="nuclear",
                    series="nuclear",
                    fill_method="none",
                    minimum_coverage=0.90,
                )
            },
        )
        config = {
            "model": {"horizon": 24},
            "data": {
                "frequency": "h",
                "target_end_policy": "full_series",
                "target_interpolation_limit": 0,
                "require_all_covariates": True,
            },
        }

        def fake_loader(_zone, spec, *_args, **_kwargs):
            if spec.alias == "target":
                return target, {"source": "test"}
            return covariate, {"source": "test"}

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "chronos2_modular.data.load_input_series",
                side_effect=fake_loader,
            ):
                with self.assertRaisesRegex(ValueError, "couverture"):
                    prepare_zone_data(
                        zone,
                        config,
                        Path(directory),
                        False,
                        Path(directory) / "output",
                    )

    def test_future_pit_coverage_is_really_blocking(self) -> None:
        index = pd.date_range(
            "2026-01-01", periods=96, freq="h", tz="Europe/Paris"
        )
        target = pd.Series(np.arange(len(index), dtype=float), index=index)
        # Historique parfait, mais aucune valeur connue pour les 24 heures
        # futures du modèle.
        forecast = pd.Series(10.0, index=index)
        zone = ZoneConfig(
            zone="FR",
            timezone="Europe/Paris",
            target=SeriesSpec(alias="target", series="price"),
            covariates={
                "forecast": SeriesSpec(
                    alias="forecast",
                    series="forecast",
                    fill_method="none",
                    minimum_coverage=1.0,
                    known_future=True,
                    future_strategies=("oracle",),
                )
            },
        )
        config = {
            "model": {"horizon": 24},
            "data": {
                "frequency": "h",
                "target_end_policy": "full_series",
                "target_interpolation_limit": 0,
                "require_all_covariates": True,
                "require_complete_future_covariates": True,
                "minimum_future_coverage": 1.0,
            },
        }

        def fake_loader(_zone, spec, *_args, **_kwargs):
            return (
                (target if spec.alias == "target" else forecast),
                {"source": "test"},
            )

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "chronos2_modular.data.load_input_series",
                side_effect=fake_loader,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Couverture future insuffisante",
                ):
                    prepare_zone_data(
                        zone,
                        config,
                        Path(directory),
                        False,
                        Path(directory) / "output",
                    )


if __name__ == "__main__":
    unittest.main()
