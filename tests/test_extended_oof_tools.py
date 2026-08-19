from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from chronos2_hourly.chronos_adapter import generate_delivery_plans
from generate_chronos_oof_range import (
    _expected_index as chronos_expected_index,
    _local_model_config,
    _select_zone,
    _validate_existing_prefix,
)
import generate_hourly_supervised_oof_block as supervised_oof
from generate_hourly_supervised_oof_block import (
    _active_training_columns,
    _expected_index as supervised_expected_index,
    _resolve_zone_contract,
    _validate_split,
)
from screen_extended_residual_calibration import (
    CHRONOS_EXPERT_COLUMNS,
    _build_meta,
    _fit_frame,
)


class ExtendedOOFToolTests(unittest.TestCase):
    @staticmethod
    def _zone_config(
        zone: str,
        timezone: str,
        *,
        primary_country: str | None = None,
    ) -> dict[str, object]:
        builder: dict[str, object] = {
            "timezone": timezone,
            "rich_calendar_countries": ["FR", "DE", "BE", "ES", "NL"],
        }
        if primary_country is not None:
            builder["rich_calendar_primary_country"] = primary_country
        return {
            "data": {"default_fill_limit": 0},
            "hourly": {
                "residual_correction": {"feature_builder": builder},
            },
            "zones": {
                zone: {
                    "enabled": True,
                    "timezone": timezone,
                    "target": {
                        "series": f"power.price.{zone.lower()}.test",
                        "naive_timezone": "UTC",
                    },
                    "covariates": {},
                }
            },
        }

    def test_chronos_oof_zone_selection_is_not_fr_hardcoded(self) -> None:
        config = self._zone_config("DE", "Europe/Berlin", primary_country="DE")
        selected = _select_zone(config, "de")
        self.assertEqual(selected.zone, "DE")
        self.assertEqual(selected.timezone, "Europe/Berlin")

    def test_supervised_zone_contract_uses_zone_timezone_and_primary_calendar(self) -> None:
        expected = {
            "BE": ("Europe/Brussels", "BE"),
            "DE": ("Europe/Berlin", "DE"),
            "NL": ("Europe/Amsterdam", "NL"),
            "ES": ("Europe/Madrid", "ES"),
        }
        for zone_code, (timezone, primary) in expected.items():
            with self.subTest(zone=zone_code):
                config = self._zone_config(
                    zone_code,
                    timezone,
                    primary_country=primary,
                )
                zone, resolved_timezone, countries, resolved_primary = (
                    _resolve_zone_contract(config, zone_code.lower())
                )
                self.assertEqual(zone.zone, zone_code)
                self.assertEqual(resolved_timezone, timezone)
                self.assertIn(resolved_primary, countries)
                self.assertEqual(resolved_primary, primary)

    def test_supervised_zone_contract_rejects_timezone_drift(self) -> None:
        config = self._zone_config("DE", "Europe/Berlin", primary_country="DE")
        with self.assertRaisesRegex(ValueError, "--timezone"):
            _resolve_zone_contract(
                config,
                "DE",
                timezone_override="Europe/Paris",
            )

    def test_supervised_zone_contract_rejects_primary_calendar_drift(self) -> None:
        config = self._zone_config("DE", "Europe/Berlin", primary_country="DE")
        with self.assertRaisesRegex(ValueError, "--calendar-primary-country"):
            _resolve_zone_contract(
                config,
                "DE",
                primary_country_override="FR",
            )

    def test_meta_builder_receives_non_fr_calendar_contract(self) -> None:
        captured: dict[str, object] = {}

        class StubMetaBuilder:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def fit_transform(
                self,
                X: pd.DataFrame,
                expert_predictions: pd.DataFrame,
            ) -> pd.DataFrame:
                captured["expert_columns"] = tuple(expert_predictions.columns)
                return pd.DataFrame(
                    0.0,
                    index=X.index,
                    columns=[f"meta_{position}" for position in range(188)],
                )

        index = pd.date_range("2024-01-02", periods=2, freq="h", tz="UTC")
        X = pd.DataFrame({"known_value": [1.0, 2.0]}, index=index)
        experts = pd.DataFrame(
            {
                column: np.linspace(10.0, 11.0, len(index))
                for column in supervised_oof.EXPERT_COLUMNS
            },
            index=index,
        )
        with patch.object(
            supervised_oof,
            "ResidualMetaFeatureBuilder",
            StubMetaBuilder,
        ):
            meta = supervised_oof.build_v1_meta_features(
                X,
                experts,
                timezone="Europe/Berlin",
                rich_calendar_countries=("FR", "DE"),
                rich_calendar_primary_country="DE",
            )
        self.assertEqual(meta.shape, (2, 188))
        self.assertEqual(captured["timezone"], "Europe/Berlin")
        self.assertEqual(captured["rich_calendar_countries"], ("FR", "DE"))
        self.assertEqual(captured["rich_calendar_primary_country"], "DE")

    def test_expected_index_preserves_23_and_25_hour_days(self) -> None:
        spring = generate_delivery_plans("2024-03-31", "2024-03-31")
        autumn = generate_delivery_plans("2024-10-27", "2024-10-27")
        self.assertEqual(len(chronos_expected_index(spring)), 23)
        self.assertEqual(len(supervised_expected_index(autumn)), 25)

    def test_split_requires_exact_one_day_purge(self) -> None:
        gap = _validate_split(
            train_end=pd.Timestamp("2023-12-31"),
            test_start=pd.Timestamp("2024-01-02"),
            test_end=pd.Timestamp("2024-08-11"),
            gap_days=1,
        )
        self.assertEqual(gap, [pd.Timestamp("2024-01-01")])
        with self.assertRaisesRegex(ValueError, "train_end_day"):
            _validate_split(
                train_end=pd.Timestamp("2024-01-01"),
                test_start=pd.Timestamp("2024-01-02"),
                test_end=pd.Timestamp("2024-08-11"),
                gap_days=1,
            )

    def test_all_missing_training_columns_are_explicitly_removed(self) -> None:
        X = pd.DataFrame(
            {
                "calendar": [1.0, 2.0, 3.0],
                "fundamental": [np.nan, np.nan, np.nan],
            }
        )
        active, dropped = _active_training_columns(X)
        self.assertEqual(active, ["calendar"])
        self.assertEqual(dropped, ["fundamental"])

    def test_resume_prefix_stops_at_a_complete_day(self) -> None:
        plans = generate_delivery_plans("2024-03-30", "2024-03-31")
        first = plans[0]
        frame = pd.DataFrame(
            {
                "delivery_start_utc": first.delivery_index_utc,
                "forecast_origin_utc": first.forecast_origin_utc,
                "q10": 10.0,
                "q50": 20.0,
                "q90": 30.0,
                "actual": np.arange(first.horizon, dtype=float),
            }
        )
        target_index = chronos_expected_index(plans)
        target = pd.Series(np.arange(len(target_index), dtype=float), index=target_index)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "prefix.csv.gz"
            frame.to_csv(path, index=False, compression="gzip")
            existing, completed = _validate_existing_prefix(path, plans, target)
        self.assertEqual(completed, 1)
        self.assertEqual(len(existing), 24)

    def test_local_model_config_resolves_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            snapshot = Path(root)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            resolved = _local_model_config({"model": {"model_id": str(snapshot)}})
        self.assertEqual(Path(resolved["model"]["model_id"]), snapshot.resolve())
        self.assertTrue(resolved["model"]["local_files_only"])

    def test_chronos_only_meta_schema_is_stable_across_blocks(self) -> None:
        first_index = pd.date_range("2024-01-02", periods=48, freq="h", tz="UTC")
        second_index = pd.date_range("2025-04-14", periods=48, freq="h", tz="UTC")
        columns = {
            "known_fr_residual_load_fcst_oracle": np.arange(48, dtype=float),
            "calendar_local_hour": np.tile(np.arange(24), 2),
            "price_lag_24h": np.linspace(1.0, 2.0, 48),
        }
        X_first = pd.DataFrame(columns, index=first_index)
        X_second = pd.DataFrame(columns, index=second_index)
        experts_first = pd.DataFrame(
            {column: np.linspace(10.0, 20.0, 48) for column in CHRONOS_EXPERT_COLUMNS},
            index=first_index,
        )
        experts_second = pd.DataFrame(
            {column: np.linspace(11.0, 21.0, 48) for column in CHRONOS_EXPERT_COLUMNS},
            index=second_index,
        )
        first = _build_meta(X_first, experts_first, CHRONOS_EXPERT_COLUMNS)
        second = _build_meta(X_second, experts_second, CHRONOS_EXPERT_COLUMNS)
        self.assertTupleEqual(tuple(first.columns), tuple(second.columns))

    def test_extended_fit_requires_same_schema(self) -> None:
        first_index = pd.date_range("2024-01-02", periods=2, freq="h", tz="UTC")
        second_index = pd.date_range("2024-08-12", periods=2, freq="h", tz="UTC")
        external = {
            "meta": pd.DataFrame({"a": [1.0, 2.0]}, index=first_index),
            "residual": pd.Series([0.0, 1.0], index=first_index),
        }
        existing = {
            "meta": pd.DataFrame({"b": [3.0, 4.0]}, index=second_index),
            "residual": pd.Series([1.0, 2.0], index=second_index),
        }
        with self.assertRaisesRegex(RuntimeError, "meta-features"):
            _fit_frame(external, existing, np.asarray([True, True]))


if __name__ == "__main__":
    unittest.main()
