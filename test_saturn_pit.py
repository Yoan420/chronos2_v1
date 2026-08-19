from __future__ import annotations

import tempfile
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest import mock

import pandas as pd

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from chronos2_modular.common import (
    SeriesSpec,
    ZoneConfig,
    parse_series_spec,
)
from chronos2_modular.data import (
    load_input_series,
    read_pit_vintage_series,
    resolve_live_target_cutoff,
)
from chronos2_modular.saturn import (
    cache_path_for_series,
    fetch_saturn_series_from_client,
    read_vintage_store,
    sync_saturn_data,
    sync_vintage_series,
)


class FakeSaturnClient:
    def __init__(
        self,
        *,
        latest: Mapping[str, pd.Series] | None = None,
        histories: Mapping[
            str,
            Mapping[pd.Timestamp, pd.Series],
        ]
        | None = None,
    ) -> None:
        self.latest = dict(latest or {})
        self.histories = {
            name: dict(history)
            for name, history in (histories or {}).items()
        }
        self.get_calls: list[tuple[str, dict[str, Any]]] = []
        self.history_calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, name: str, *args: Any, **kwargs: Any) -> pd.Series:
        self.get_calls.append((name, dict(kwargs)))
        series = self.latest.get(name)
        if series is None:
            return pd.Series(dtype=float, name=name)

        start = kwargs.get("from_value_date")
        end = kwargs.get("to_value_date")
        if start is None:
            start = kwargs.get("from_value", kwargs.get("start"))
        if end is None:
            end = kwargs.get("to_value", kwargs.get("end"))
        if start is None and args:
            start = args[0]
        if end is None and len(args) > 1:
            end = args[1]

        result = series
        if start is not None:
            result = result.loc[result.index >= pd.Timestamp(start)]
        if end is not None:
            result = result.loc[result.index <= pd.Timestamp(end)]
        return result.copy()

    def history(self, name: str, **kwargs: Any) -> dict[Any, Any]:
        self.history_calls.append((name, dict(kwargs)))
        start_revision = pd.Timestamp(
            kwargs["from_insertion_date"]
        )
        end_revision = pd.Timestamp(kwargs["to_insertion_date"])
        start_value = pd.Timestamp(kwargs["from_value_date"])
        end_value = pd.Timestamp(kwargs["to_value_date"])

        result: dict[Any, Any] = {}
        for revision, series in self.histories.get(name, {}).items():
            if not start_revision <= revision <= end_revision:
                continue
            values = series.loc[
                (series.index >= start_value)
                & (series.index <= end_value)
            ]
            if not values.empty:
                result[revision] = values.copy()
        return result


def pit_config() -> dict[str, Any]:
    return {
        "data": {
            "forecast_origin_local_time": "08:00",
            "revision_policy": "latest_before_asof",
        }
    }


def _pickle_to_parquet(
    frame: pd.DataFrame,
    path: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    frame.to_pickle(path)


def _pickle_read_parquet(
    path: Any,
    *args: Any,
    columns: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    return frame.loc[:, columns] if columns is not None else frame


class SaturnAsOfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import pyarrow  # noqa: F401
        except ModuleNotFoundError:
            cls._parquet_patches = (
                mock.patch.object(
                    pd.DataFrame,
                    "to_parquet",
                    _pickle_to_parquet,
                ),
                mock.patch.object(
                    pd,
                    "read_parquet",
                    _pickle_read_parquet,
                ),
            )
            for patcher in cls._parquet_patches:
                patcher.start()
        else:
            cls._parquet_patches = ()

    @classmethod
    def tearDownClass(cls) -> None:
        for patcher in cls._parquet_patches:
            patcher.stop()

    def test_get_receives_revision_date(self) -> None:
        index = pd.date_range(
            "2024-01-02 00:00",
            periods=3,
            freq="h",
            tz="UTC",
        )
        client = FakeSaturnClient(
            latest={"forecast": pd.Series([1.0, 2.0, 3.0], index=index)}
        )
        as_of = pd.Timestamp("2024-01-01 08:00", tz="UTC")

        result = fetch_saturn_series_from_client(
            client,
            "forecast",
            index[0],
            index[-1],
            "Europe/Paris",
            revision_date=as_of,
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(len(client.get_calls), 1)
        self.assertEqual(
            client.get_calls[0][1]["revision_date"],
            as_of,
        )

    def test_network_error_does_not_try_legacy_signatures(self) -> None:
        class FailingClient:
            def __init__(self) -> None:
                self.calls = 0

            def get(self, name: str, **kwargs: Any) -> pd.Series:
                self.calls += 1
                raise ConnectionError("proxy unavailable")

        client = FailingClient()
        start = pd.Timestamp("2024-01-02 00:00", tz="UTC")
        with self.assertRaisesRegex(RuntimeError, "proxy unavailable") as raised:
            fetch_saturn_series_from_client(
                client,
                "forecast",
                start,
                start + pd.Timedelta(hours=2),
                "Europe/Paris",
            )

        self.assertEqual(client.calls, 1)
        self.assertNotIn("unexpected keyword argument", str(raised.exception))

    def test_legacy_signature_is_tried_only_after_keyword_type_error(self) -> None:
        index = pd.date_range("2024-01-02 00:00", periods=3, freq="h", tz="UTC")
        values = pd.Series([1.0, 2.0, 3.0], index=index)

        class LegacyClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def get(self, name: str, **kwargs: Any) -> pd.Series:
                self.calls.append(dict(kwargs))
                if "from_value_date" in kwargs:
                    raise TypeError("got an unexpected keyword argument 'from_value_date'")
                return values

        client = LegacyClient()
        result = fetch_saturn_series_from_client(
            client,
            "forecast",
            index[0],
            index[-1],
            "Europe/Paris",
        )

        self.assertEqual(len(client.calls), 2)
        self.assertIn("from_value_date", client.calls[0])
        self.assertIn("from_value", client.calls[1])
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0])

    def test_non_signature_type_error_does_not_fallback(self) -> None:
        class BrokenClient:
            def __init__(self) -> None:
                self.calls = 0

            def get(self, name: str, **kwargs: Any) -> pd.Series:
                self.calls += 1
                raise TypeError("internal decoding error")

        client = BrokenClient()
        start = pd.Timestamp("2024-01-02 00:00", tz="UTC")
        with self.assertRaisesRegex(RuntimeError, "internal decoding error"):
            fetch_saturn_series_from_client(
                client,
                "forecast",
                start,
                start + pd.Timedelta(hours=2),
                "Europe/Paris",
            )
        self.assertEqual(client.calls, 1)

    def test_naive_timezone_is_parsed_from_series_spec(self) -> None:
        spec = parse_series_spec(
            "target",
            {
                "series": "price",
                "naive_timezone": "UTC",
            },
            default_fill=3,
        )

        self.assertEqual(spec.naive_timezone, "UTC")

    def test_naive_timezone_changes_latest_cache_identity(self) -> None:
        cache_root = Path("cache")
        automatic = cache_path_for_series(
            cache_root,
            "FR",
            SeriesSpec(alias="target", series="price"),
        )
        explicit_utc = cache_path_for_series(
            cache_root,
            "FR",
            SeriesSpec(
                alias="target",
                series="price",
                naive_timezone="UTC",
            ),
        )

        self.assertNotEqual(automatic, explicit_utc)

    def test_fetch_respects_explicit_utc_for_naive_payload(self) -> None:
        index = pd.date_range(
            "2024-07-01 00:00",
            periods=3,
            freq="h",
        )
        client = FakeSaturnClient(
            latest={"price": pd.Series([1.0, 2.0, 3.0], index=index)}
        )

        result = fetch_saturn_series_from_client(
            client,
            "price",
            index[0],
            index[-1],
            "Europe/Paris",
            naive_timezone="UTC",
        )

        self.assertEqual(
            result.index[0],
            pd.Timestamp("2024-07-01 02:00", tz="Europe/Paris"),
        )
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0])

    def test_direct_load_forwards_series_naive_timezone(self) -> None:
        zone = ZoneConfig(
            zone="FR",
            timezone="Europe/Paris",
            target=SeriesSpec(alias="target", series="price"),
            covariates={},
        )
        spec = SeriesSpec(
            alias="load",
            series="load",
            source="saturn",
            naive_timezone="UTC",
            incomplete_dst_policy="duplicate",
        )
        downloaded = pd.Series(
            [10.0, 11.0],
            index=pd.date_range(
                "2024-07-01 02:00",
                periods=2,
                freq="h",
                tz="Europe/Paris",
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "data": {
                    "project_root": str(root),
                    "cache_dir": "cache",
                    "source": "saturn",
                    "start": "2024-07-01T00:00:00Z",
                    "end": "2024-07-01T01:00:00Z",
                    "saturn_url": "https://saturn.invalid",
                    "saturn_author": "test",
                }
            }
            with mock.patch(
                "chronos2_modular.data.fetch_saturn_series",
                return_value=downloaded,
            ) as fetch_mock:
                result, _ = load_input_series(
                    zone,
                    spec,
                    config,
                    root,
                    refresh=True,
                )

        self.assertEqual(result.tolist(), [10.0, 11.0])
        self.assertEqual(
            fetch_mock.call_args.kwargs["naive_timezone"],
            "UTC",
        )
        self.assertEqual(
            fetch_mock.call_args.kwargs["incomplete_dst_policy"],
            "duplicate",
        )

    def test_runtime_as_of_controls_operational_target_day(self) -> None:
        zone = ZoneConfig(
            zone="FR",
            timezone="Europe/Paris",
            target=SeriesSpec(alias="target", series="price"),
            covariates={},
        )
        config = {
            "data": {
                "runtime_as_of": "2024-04-01T05:00:00Z",
                "target_end_policy": "current_day_end",
                "frequency": "h",
            }
        }

        cutoff = resolve_live_target_cutoff(zone, config)

        self.assertEqual(
            cutoff,
            pd.Timestamp("2024-04-01 23:00", tz="Europe/Paris"),
        )

    def test_latest_revision_before_cutoff_is_selected(self) -> None:
        delivery = pd.Timestamp("2024-01-02 11:00", tz="UTC")
        frame = pd.DataFrame(
            {
                "delivery_utc": [delivery, delivery],
                "availability_utc": [
                    pd.Timestamp("2024-01-01 06:30", tz="UTC"),
                    pd.Timestamp("2024-01-01 07:01", tz="UTC"),
                ],
                "revision_utc": [
                    pd.Timestamp("2024-01-01 06:30", tz="UTC"),
                    pd.Timestamp("2024-01-01 07:01", tz="UTC"),
                ],
                "value": [10.0, 20.0],
            }
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forecast.parquet"
            frame.to_parquet(path, index=False)
            series, metadata = read_pit_vintage_series(
                path,
                SeriesSpec(alias="forecast"),
                "Europe/Paris",
                pit_config(),
            )

        self.assertEqual(series.iloc[0], 10.0)
        self.assertEqual(metadata["cutoff_violations"], 0)
        self.assertEqual(metadata["selected_rows"], 1)

    def test_runtime_as_of_caps_a_future_nominal_cutoff(self) -> None:
        delivery = pd.Timestamp("2026-01-02 11:00", tz="UTC")
        frame = pd.DataFrame(
            {
                "delivery_utc": [delivery, delivery],
                "availability_utc": pd.to_datetime(
                    ["2026-01-01 06:00Z", "2026-01-01 06:30Z"], utc=True
                ),
                "revision_utc": pd.to_datetime(
                    ["2026-01-01 06:00Z", "2026-01-01 06:30Z"], utc=True
                ),
                "value": [10.0, 999.0],
            }
        )
        config = pit_config()
        config["data"]["runtime_as_of"] = "2026-01-01T06:15:00Z"

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forecast.parquet"
            frame.to_parquet(path, index=False)
            series, metadata = read_pit_vintage_series(
                path,
                SeriesSpec(alias="forecast"),
                "Europe/Paris",
                config,
            )

        self.assertEqual(series.iloc[0], 10.0)
        self.assertEqual(
            metadata["runtime_as_of_utc"],
            "2026-01-01 06:15:00+00:00",
        )

    def test_null_revision_does_not_resurrect_older_value(self) -> None:
        delivery = pd.Timestamp("2024-01-02 11:00", tz="UTC")
        frame = pd.DataFrame(
            {
                "delivery_utc": [delivery, delivery],
                "availability_utc": [
                    pd.Timestamp("2024-01-01 06:00", tz="UTC"),
                    pd.Timestamp("2024-01-01 06:30", tz="UTC"),
                ],
                "revision_utc": [
                    pd.Timestamp("2024-01-01 06:00", tz="UTC"),
                    pd.Timestamp("2024-01-01 06:30", tz="UTC"),
                ],
                "value": [10.0, float("nan")],
            }
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forecast.parquet"
            frame.to_parquet(path, index=False)
            series, metadata = read_pit_vintage_series(
                path,
                SeriesSpec(alias="forecast"),
                "Europe/Paris",
                pit_config(),
            )

        self.assertTrue(pd.isna(series.iloc[0]))
        self.assertEqual(metadata["selected_null_values"], 1)

    def test_dst_cutoff_is_computed_in_europe_paris(self) -> None:
        delivery = pd.Timestamp("2024-04-01 10:00", tz="UTC")
        frame = pd.DataFrame(
            {
                "delivery_utc": [delivery, delivery],
                "availability_utc": [
                    pd.Timestamp("2024-03-31 05:59", tz="UTC"),
                    pd.Timestamp("2024-03-31 06:01", tz="UTC"),
                ],
                "revision_utc": [
                    pd.Timestamp("2024-03-31 05:59", tz="UTC"),
                    pd.Timestamp("2024-03-31 06:01", tz="UTC"),
                ],
                "value": [1.0, 2.0],
            }
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forecast.parquet"
            frame.to_parquet(path, index=False)
            series, _ = read_pit_vintage_series(
                path,
                SeriesSpec(alias="forecast"),
                "Europe/Paris",
                pit_config(),
            )

        self.assertEqual(series.iloc[0], 1.0)

    def test_incremental_vintage_sync_is_idempotent(self) -> None:
        delivery = pd.Timestamp("2024-01-02 11:00", tz="UTC")
        history = {
            pd.Timestamp("2024-01-01 06:00", tz="UTC"): pd.Series(
                [10.0],
                index=[delivery],
            ),
            pd.Timestamp("2024-01-01 09:00", tz="UTC"): pd.Series(
                [20.0],
                index=[delivery],
            ),
        }
        client = FakeSaturnClient(histories={"forecast": history})

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forecast.parquet"
            first = sync_vintage_series(
                client,
                zone="FR",
                alias="forecast",
                series_name="forecast",
                path=path,
                revision_start=pd.Timestamp(
                    "2024-01-01 00:00", tz="UTC"
                ),
                revision_end=pd.Timestamp(
                    "2024-01-02 00:00", tz="UTC"
                ),
                value_start=pd.Timestamp(
                    "2024-01-01 00:00", tz="UTC"
                ),
                value_end=pd.Timestamp(
                    "2024-01-03 00:00", tz="UTC"
                ),
                timezone="Europe/Paris",
                retries=1,
                full=True,
            )
            second = sync_vintage_series(
                client,
                zone="FR",
                alias="forecast",
                series_name="forecast",
                path=path,
                revision_start=pd.Timestamp(
                    "2024-01-01 00:00", tz="UTC"
                ),
                revision_end=pd.Timestamp(
                    "2024-01-02 00:00", tz="UTC"
                ),
                value_start=pd.Timestamp(
                    "2024-01-01 00:00", tz="UTC"
                ),
                value_end=pd.Timestamp(
                    "2024-01-03 00:00", tz="UTC"
                ),
                timezone="Europe/Paris",
                retries=1,
                full=False,
            )
            stored = read_vintage_store(path)

        self.assertEqual(first.rows_after, 2)
        self.assertEqual(second.rows_after, 2)
        self.assertEqual(len(stored), 2)

    def test_config_sync_updates_latest_and_forecast_stores(self) -> None:
        target_index = pd.date_range(
            "2024-01-01 00:00",
            "2024-01-03 23:00",
            freq="h",
            tz="Europe/Paris",
        )
        delivery = pd.Timestamp("2024-01-03 11:00", tz="UTC")
        revision = pd.Timestamp("2024-01-02 06:00", tz="UTC")
        client = FakeSaturnClient(
            latest={
                "price": pd.Series(
                    range(len(target_index)),
                    index=target_index,
                    dtype=float,
                )
            },
            histories={
                "forecast": {
                    revision: pd.Series([42.0], index=[delivery])
                }
            },
        )
        zone = ZoneConfig(
            zone="FR",
            timezone="Europe/Paris",
            target=SeriesSpec(alias="target", series="price"),
            covariates={
                "forecast": SeriesSpec(
                    alias="forecast",
                    series="forecast",
                    source="pit_parquet",
                )
            },
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "data": {
                    "project_root": str(root),
                    "source": "auto",
                    "cache_dir": "cache",
                    "pit_vintage_dir": str(root / "pit"),
                    "pit_files": {
                        "forecast": "forecast.parquet"
                    },
                    "start": "2024-01-01T00:00:00+01:00",
                    "end": "2024-01-03T23:00:00+01:00",
                    "frequency": "h",
                    "saturn_sync": {
                        "initial_revision_start": (
                            "2024-01-01T00:00:00Z"
                        ),
                        "retries": 1,
                    },
                }
            }
            manifest = sync_saturn_data(
                [zone],
                config,
                root,
                full=True,
                as_of="2024-01-03T08:00:00Z",
                client=client,
            )

            pit_path = root / "pit" / "forecast.parquet"
            self.assertTrue(pit_path.exists())
            self.assertEqual(len(read_vintage_store(pit_path)), 1)

        self.assertEqual(set(manifest["kind"]), {
            "latest",
            "forecast_vintages",
        })
        self.assertEqual(
            client.get_calls[0][1]["revision_date"],
            pd.Timestamp("2024-01-03 08:00", tz="UTC"),
        )
        self.assertLessEqual(
            client.history_calls[0][1]["to_insertion_date"],
            pd.Timestamp("2024-01-03 08:00", tz="UTC"),
        )

    def test_full_target_refresh_keeps_pit_incremental(self) -> None:
        target_index = pd.date_range(
            "2024-01-01 00:00",
            "2024-01-03 23:00",
            freq="h",
            tz="Europe/Paris",
        )
        delivery = pd.Timestamp("2024-01-03 11:00", tz="UTC")
        revision = pd.Timestamp("2024-01-02 06:00", tz="UTC")
        client = FakeSaturnClient(
            latest={
                "price": pd.Series(
                    range(len(target_index)),
                    index=target_index,
                    dtype=float,
                )
            },
            histories={
                "forecast": {
                    revision: pd.Series([42.0], index=[delivery])
                }
            },
        )
        zone = ZoneConfig(
            zone="FR",
            timezone="Europe/Paris",
            target=SeriesSpec(alias="target", series="price"),
            covariates={
                "forecast": SeriesSpec(
                    alias="forecast",
                    series="forecast",
                    source="pit_parquet",
                )
            },
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "data": {
                    "project_root": str(root),
                    "source": "auto",
                    "cache_dir": "cache",
                    "pit_vintage_dir": str(root / "pit"),
                    "pit_files": {"forecast": "forecast.parquet"},
                    "start": "2024-01-01T00:00:00+01:00",
                    "end": "2024-01-03T23:00:00+01:00",
                    "frequency": "h",
                    "saturn_sync": {
                        "initial_revision_start": "2024-01-01T00:00:00Z",
                        "retries": 1,
                    },
                }
            }
            sync_saturn_data(
                [zone],
                config,
                root,
                full=True,
                as_of="2024-01-03T08:00:00Z",
                client=client,
            )
            client.get_calls.clear()
            client.history_calls.clear()

            manifest = sync_saturn_data(
                [zone],
                config,
                root,
                full_target=True,
                as_of="2024-01-03T08:00:00Z",
                client=client,
            )

            target_row = manifest.loc[manifest["alias"] == "target"].iloc[0]
            pit_row = manifest.loc[manifest["alias"] == "forecast"].iloc[0]
            self.assertEqual(int(target_row["rows_before"]), 0)
            self.assertEqual(int(pit_row["rows_before"]), 1)
            self.assertTrue(client.get_calls)
            self.assertTrue(client.history_calls)

    def test_full_target_refresh_can_skip_pit(self) -> None:
        target_index = pd.date_range(
            "2024-01-01 00:00",
            "2024-01-03 23:00",
            freq="h",
            tz="Europe/Paris",
        )
        client = FakeSaturnClient(
            latest={
                "price": pd.Series(
                    range(len(target_index)),
                    index=target_index,
                    dtype=float,
                )
            },
            histories={
                "forecast": {
                    pd.Timestamp("2024-01-02 06:00", tz="UTC"): pd.Series(
                        [42.0],
                        index=[pd.Timestamp("2024-01-03 11:00", tz="UTC")],
                    )
                }
            },
        )
        zone = ZoneConfig(
            zone="FR",
            timezone="Europe/Paris",
            target=SeriesSpec(
                alias="target",
                series="price",
                naive_timezone="UTC",
            ),
            covariates={
                "forecast": SeriesSpec(
                    alias="forecast",
                    series="forecast",
                    source="pit_parquet",
                )
            },
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "data": {
                    "project_root": str(root),
                    "source": "auto",
                    "cache_dir": "cache",
                    "pit_vintage_dir": str(root / "pit"),
                    "pit_files": {"forecast": "forecast.parquet"},
                    "start": "2024-01-01T00:00:00+01:00",
                    "end": "2024-01-03T23:00:00+01:00",
                    "frequency": "h",
                }
            }
            with mock.patch(
                "chronos2_modular.saturn.fetch_saturn_series_from_client",
                wraps=fetch_saturn_series_from_client,
            ) as fetch_mock:
                manifest = sync_saturn_data(
                    [zone],
                    config,
                    root,
                    full_target=True,
                    skip_pit=True,
                    as_of="2024-01-03T08:00:00Z",
                    client=client,
                )

            self.assertEqual(set(manifest["alias"]), {"target"})
            self.assertEqual(
                fetch_mock.call_args.kwargs["naive_timezone"],
                "UTC",
            )
            self.assertTrue(client.get_calls)
            self.assertFalse(client.history_calls)
            self.assertFalse((root / "pit" / "forecast.parquet").exists())


if __name__ == "__main__":
    unittest.main()
