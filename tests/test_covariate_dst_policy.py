from __future__ import annotations

import hashlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from chronos2_modular.common import (
    SeriesSpec,
    build_zone_configs,
    parse_series_spec,
)
from chronos2_modular.saturn import (
    cache_path_for_series,
    fetch_saturn_series_from_client,
    history_to_vintage_frame,
    normalize_saturn_series,
    sync_latest_series,
)


PARIS = "Europe/Paris"
NUCLEAR = "power.fr.generation.nuclear.entsoe.hourly.gw.obs"

SPRING_DAYS = (
    "2022-03-27",
    "2023-03-26",
    "2024-03-31",
    "2025-03-30",
)
FALL_DAYS = (
    "2022-10-30",
    "2023-10-29",
    "2024-10-27",
    "2025-10-26",
)


def _physical_grid(
    start: str = "2022-01-01 00:00",
    end: str = "2025-12-31 23:00",
) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", tz="UTC")


def _local_naive_series(
    expected_utc: pd.DatetimeIndex,
    *,
    lose_second_fall_fold: bool,
) -> pd.Series:
    local_labels = expected_utc.tz_convert(PARIS).tz_localize(None)
    raw = pd.Series(
        np.arange(len(expected_utc), dtype=float),
        index=local_labels,
        name="nuclear",
    )
    if lose_second_fall_fold:
        raw = raw.loc[~raw.index.duplicated(keep="first")]
    return raw


def _local_day(series: pd.Series, day: str) -> pd.Series:
    local_index = series.index.tz_convert(PARIS)
    return series.loc[local_index.date == pd.Timestamp(day).date()]


class CovariateDstPolicyParsingTests(unittest.TestCase):
    def test_parse_duplicate_and_default_raise(self) -> None:
        duplicate = parse_series_spec(
            "nuclear",
            {
                "series": NUCLEAR,
                "incomplete_dst_policy": " DuPliCaTe ",
            },
            0,
        )
        strict = parse_series_spec(
            "nuclear",
            {"series": NUCLEAR},
            0,
        )

        self.assertEqual(duplicate.incomplete_dst_policy, "duplicate")
        self.assertEqual(strict.incomplete_dst_policy, "raise")

    def test_parse_rejects_unknown_policy(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "incomplete_dst_policy inconnue",
        ):
            parse_series_spec(
                "nuclear",
                {
                    "series": NUCLEAR,
                    "incomplete_dst_policy": "interpolate",
                },
                0,
            )

    def test_duplicate_policy_is_forbidden_for_target(self) -> None:
        config = {
            "data": {"default_fill_limit": 0},
            "zones": {
                "FR": {
                    "timezone": PARIS,
                    "target": {
                        "series": "power.price.da.fr.hourly",
                        "incomplete_dst_policy": "duplicate",
                    },
                    "covariates": {},
                }
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            "interdit pour la cible",
        ):
            build_zone_configs(config, None, None, None)


class CovariateDstNormalizationTests(unittest.TestCase):
    def test_multi_year_23_24_source_becomes_physical_23_25_grid(
        self,
    ) -> None:
        expected_utc = _physical_grid()
        raw = _local_naive_series(
            expected_utc,
            lose_second_fall_fold=True,
        )

        with self.assertLogs("chronos2_modular", level="WARNING") as logs:
            normalized = normalize_saturn_series(
                raw,
                NUCLEAR,
                PARIS,
                naive_timezone=PARIS,
                incomplete_dst_policy="duplicate",
            )

        self.assertFalse(normalized.index.has_duplicates)
        self.assertTrue(normalized.index.is_monotonic_increasing)
        self.assertTrue(
            normalized.index.tz_convert("UTC").equals(expected_utc)
        )
        self.assertIn(
            "4 heure(s) automnale(s) singleton dupliquée(s)",
            "\n".join(logs.output),
        )

        for day in SPRING_DAYS:
            with self.subTest(day=day, transition="spring"):
                self.assertEqual(len(_local_day(normalized, day)), 23)

        for day in FALL_DAYS:
            with self.subTest(day=day, transition="fall"):
                local_day = _local_day(normalized, day)
                repeated_02 = local_day.loc[
                    local_day.index.tz_convert(PARIS).hour == 2
                ]

                self.assertEqual(len(local_day), 25)
                self.assertEqual(len(repeated_02), 2)
                self.assertEqual(
                    repeated_02.index[1].tz_convert("UTC")
                    - repeated_02.index[0].tz_convert("UTC"),
                    pd.Timedelta(hours=1),
                )
                self.assertEqual(
                    float(repeated_02.iloc[0]),
                    float(repeated_02.iloc[1]),
                )

    def test_default_strict_policy_still_rejects_singleton_fall_hour(
        self,
    ) -> None:
        expected_utc = pd.date_range(
            "2024-10-26 00:00",
            "2024-10-28 23:00",
            freq="h",
            tz="UTC",
        )
        raw = _local_naive_series(
            expected_utc,
            lose_second_fall_fold=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "apparaît 1 fois au lieu de 2",
        ):
            normalize_saturn_series(
                raw,
                NUCLEAR,
                PARIS,
                naive_timezone=PARIS,
            )

    def test_duplicate_policy_preserves_a_real_two_fold_pair(self) -> None:
        expected_utc = pd.date_range(
            "2024-10-26 00:00",
            "2024-10-28 23:00",
            freq="h",
            tz="UTC",
        )
        raw = _local_naive_series(
            expected_utc,
            lose_second_fall_fold=False,
        )
        raw_pair = raw.loc[
            raw.index == pd.Timestamp("2024-10-27 02:00")
        ].to_numpy(copy=True)
        self.assertEqual(len(raw_pair), 2)
        self.assertNotEqual(float(raw_pair[0]), float(raw_pair[1]))

        normalized = normalize_saturn_series(
            raw,
            NUCLEAR,
            PARIS,
            naive_timezone=PARIS,
            incomplete_dst_policy="duplicate",
        )
        local_day = _local_day(normalized, "2024-10-27")
        normalized_pair = local_day.loc[
            local_day.index.tz_convert(PARIS).hour == 2
        ].to_numpy()

        self.assertTrue(
            normalized.index.tz_convert("UTC").equals(expected_utc)
        )
        np.testing.assert_array_equal(normalized_pair, raw_pair)

    def test_duplicate_policy_does_not_modify_utc_naive_input(self) -> None:
        expected_utc = pd.date_range(
            "2024-01-01 00:00",
            "2024-12-31 23:00",
            freq="h",
            tz="UTC",
        )
        raw = pd.Series(
            np.arange(len(expected_utc), dtype=float),
            index=expected_utc.tz_localize(None),
            name="nuclear",
        )

        normalized = normalize_saturn_series(
            raw,
            NUCLEAR,
            PARIS,
            naive_timezone="UTC",
            incomplete_dst_policy="duplicate",
        )

        self.assertTrue(
            normalized.index.tz_convert("UTC").equals(expected_utc)
        )
        np.testing.assert_array_equal(
            normalized.to_numpy(),
            raw.to_numpy(),
        )

    def test_duplicate_zero_only_records_zero_and_rejects_nonzero(self) -> None:
        expected_utc = pd.date_range(
            "2025-10-25 22:00",
            "2025-10-26 22:00",
            freq="h",
            tz="UTC",
        )
        raw = _local_naive_series(
            expected_utc,
            lose_second_fall_fold=True,
        )
        ambiguous = pd.Timestamp("2025-10-26 02:00")
        raw.loc[ambiguous] = 0.25
        with self.assertRaisesRegex(ValueError, "zero fini prouve"):
            normalize_saturn_series(
                raw,
                "solar",
                PARIS,
                naive_timezone=PARIS,
                incomplete_dst_policy="duplicate_zero_only",
            )

        raw.loc[ambiguous] = 0.0
        normalized = normalize_saturn_series(
            raw,
            "solar",
            PARIS,
            naive_timezone=PARIS,
            incomplete_dst_policy="duplicate_zero_only",
        )

        self.assertTrue(normalized.index.tz_convert("UTC").equals(expected_utc))
        self.assertEqual(len(normalized.attrs["dst_repairs"]), 1)
        repair = normalized.attrs["dst_repairs"][0]
        self.assertEqual(repair["duplicated_value"], 0.0)
        self.assertEqual(len(repair["physical_hours_utc"]), 2)


class CovariateDstPolicyPlumbingTests(unittest.TestCase):
    def test_cache_identity_changes_with_dst_policy(self) -> None:
        strict = SeriesSpec(
            alias="nuclear",
            series=NUCLEAR,
            naive_timezone=PARIS,
            incomplete_dst_policy="raise",
        )
        repaired = SeriesSpec(
            alias="nuclear",
            series=NUCLEAR,
            naive_timezone=PARIS,
            incomplete_dst_policy="duplicate",
        )

        self.assertNotEqual(
            cache_path_for_series(Path("cache"), "FR", strict),
            cache_path_for_series(Path("cache"), "FR", repaired),
        )

    def test_default_policy_preserves_legacy_strict_cache_identity(self) -> None:
        spec = SeriesSpec(
            alias="nuclear",
            series=NUCLEAR,
            naive_timezone=PARIS,
        )
        legacy_identity = f"{NUCLEAR}|naive_timezone={PARIS}"
        digest = hashlib.sha1(legacy_identity.encode("utf-8")).hexdigest()[:10]

        self.assertEqual(
            cache_path_for_series(Path("cache"), "FR", spec),
            Path("cache") / "fr" / f"nuclear__{digest}.csv.gz",
        )

    def test_fetch_passes_policy_to_normalizer(self) -> None:
        raw = pd.Series(
            [40.0],
            index=pd.DatetimeIndex(["2024-01-01 00:00"]),
        )
        expected = pd.Series(
            [40.0],
            index=pd.DatetimeIndex(["2023-12-31 23:00"], tz="UTC").tz_convert(
                PARIS
            ),
            name=NUCLEAR,
        )
        client = Mock()
        client.get.return_value = raw

        with patch(
            "chronos2_modular.saturn.normalize_saturn_series",
            return_value=expected,
        ) as normalizer:
            result = fetch_saturn_series_from_client(
                client,
                NUCLEAR,
                pd.Timestamp("2024-01-01", tz=PARIS),
                pd.Timestamp("2024-01-02", tz=PARIS),
                PARIS,
                naive_timezone=PARIS,
                incomplete_dst_policy="duplicate",
            )

        self.assertIs(result, expected)
        self.assertEqual(
            normalizer.call_args.kwargs["incomplete_dst_policy"],
            "duplicate",
        )
        self.assertEqual(
            normalizer.call_args.kwargs["naive_timezone"],
            PARIS,
        )

    def test_fetch_empty_valid_response_stops_dialect_detection(self) -> None:
        client = Mock()
        client.get.return_value = pd.Series(dtype=float)
        start = pd.Timestamp("2026-09-01 00:00", tz=PARIS)
        end = pd.Timestamp("2026-09-01 23:00", tz=PARIS)
        cutoff = pd.Timestamp("2026-09-01 06:00", tz="UTC")

        with self.assertRaisesRegex(
            RuntimeError,
            r"plage=.*2026-09-01.*cutoff=.*2026-09-01.*reponse vide",
        ):
            fetch_saturn_series_from_client(
                client,
                NUCLEAR,
                start,
                end,
                PARIS,
                revision_date=cutoff,
            )

        self.assertEqual(client.get.call_count, 2)
        args, kwargs = client.get.call_args_list[0]
        self.assertEqual(args, (NUCLEAR,))
        self.assertIn("from_value_date", kwargs)
        self.assertIn("to_value_date", kwargs)
        self.assertNotIn("from_value", kwargs)
        self.assertNotIn("start", kwargs)
        retry_args, retry_kwargs = client.get.call_args_list[1]
        self.assertEqual(retry_args, (NUCLEAR,))
        self.assertEqual(
            retry_kwargs["from_value_date"],
            kwargs["from_value_date"],
        )
        self.assertEqual(
            retry_kwargs["to_value_date"],
            kwargs["to_value_date"],
        )
        self.assertTrue(retry_kwargs["nocache"])

    def test_fetch_same_dialect_nocache_retry_can_recover_empty(self) -> None:
        raw = pd.Series(
            [40.0],
            index=pd.DatetimeIndex(["2026-09-01 00:00"], tz="UTC"),
        )
        client = Mock()
        client.get.side_effect = [pd.Series(dtype=float), raw]

        result = fetch_saturn_series_from_client(
            client,
            NUCLEAR,
            pd.Timestamp("2026-09-01 00:00", tz=PARIS),
            pd.Timestamp("2026-09-01 23:00", tz=PARIS),
            PARIS,
            revision_date=pd.Timestamp("2026-09-01 06:00", tz="UTC"),
        )

        self.assertEqual(client.get.call_count, 2)
        first = client.get.call_args_list[0]
        retry = client.get.call_args_list[1]
        self.assertEqual(first.args, retry.args)
        self.assertEqual(
            first.kwargs["from_value_date"],
            retry.kwargs["from_value_date"],
        )
        self.assertNotIn("from_value", retry.kwargs)
        self.assertTrue(retry.kwargs["nocache"])
        self.assertEqual(float(result.iloc[0]), 40.0)

    def test_fetch_changes_dialect_only_for_unexpected_keyword(self) -> None:
        raw = pd.Series(
            [40.0],
            index=pd.DatetimeIndex(["2024-01-01 00:00"]),
        )

        class Client:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def get(self, _name: str, **kwargs: object) -> pd.Series:
                self.calls.append(dict(kwargs))
                if "from_value_date" in kwargs:
                    raise TypeError(
                        "get() got an unexpected keyword argument "
                        "'from_value_date'"
                    )
                return raw

        client = Client()
        result = fetch_saturn_series_from_client(
            client,
            NUCLEAR,
            pd.Timestamp("2024-01-01", tz=PARIS),
            pd.Timestamp("2024-01-02", tz=PARIS),
            PARIS,
            naive_timezone=PARIS,
        )

        self.assertEqual(len(client.calls), 2)
        self.assertIn("from_value_date", client.calls[0])
        self.assertIn("from_value", client.calls[1])
        self.assertEqual(float(result.iloc[0]), 40.0)

    def test_fetch_internal_type_error_is_terminal_and_never_positional(self) -> None:
        client = Mock()
        client.get.side_effect = TypeError("payload conversion failed")

        with self.assertRaisesRegex(RuntimeError, "payload conversion failed"):
            fetch_saturn_series_from_client(
                client,
                NUCLEAR,
                pd.Timestamp("2024-01-01", tz=PARIS),
                pd.Timestamp("2024-01-02", tz=PARIS),
                PARIS,
            )

        self.assertEqual(client.get.call_count, 1)
        args, _kwargs = client.get.call_args
        self.assertEqual(args, (NUCLEAR,))

    def test_fetch_never_uses_positional_date_fallback(self) -> None:
        client = Mock()
        client.get.side_effect = TypeError(
            "get() got an unexpected keyword argument 'date'"
        )

        with self.assertRaisesRegex(RuntimeError, "unexpected keyword"):
            fetch_saturn_series_from_client(
                client,
                NUCLEAR,
                pd.Timestamp("2024-01-01", tz=PARIS),
                pd.Timestamp("2024-01-02", tz=PARIS),
                PARIS,
            )

        self.assertEqual(client.get.call_count, 3)
        for call in client.get.call_args_list:
            self.assertEqual(call.args, (NUCLEAR,))

    def test_sync_passes_policy_to_fetch_boundary(self) -> None:
        downloaded = pd.Series(
            [40.0, 41.0],
            index=pd.date_range(
                "2024-01-01 00:00",
                periods=2,
                freq="h",
                tz=PARIS,
            ),
            name="nuclear",
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nuclear.csv.gz"
            with patch(
                "chronos2_modular.saturn.fetch_saturn_series_from_client",
                return_value=downloaded,
            ) as fetch:
                result = sync_latest_series(
                    Mock(),
                    zone="FR",
                    alias="nuclear",
                    series_name=NUCLEAR,
                    path=path,
                    start=pd.Timestamp("2024-01-01", tz=PARIS),
                    end=pd.Timestamp("2024-01-02", tz=PARIS),
                    timezone=PARIS,
                    sync_as_of_utc=pd.Timestamp(
                        "2024-01-03",
                        tz="UTC",
                    ),
                    naive_timezone=PARIS,
                    incomplete_dst_policy="duplicate",
                    full=True,
                )

            self.assertTrue(path.exists())
            self.assertEqual(result.rows_after, 2)
            self.assertEqual(
                fetch.call_args.kwargs["incomplete_dst_policy"],
                "duplicate",
            )
            self.assertEqual(
                fetch.call_args.kwargs["naive_timezone"],
                PARIS,
            )

    def test_sync_rejects_duplicate_policy_for_target(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                ValueError,
                "interdit pour la cible",
            ):
                sync_latest_series(
                    Mock(),
                    zone="FR",
                    alias="target",
                    series_name="power.price.da.fr.hourly",
                    path=Path(temporary_directory) / "target.csv.gz",
                    start=pd.Timestamp("2024-01-01", tz=PARIS),
                    end=pd.Timestamp("2024-01-02", tz=PARIS),
                    timezone=PARIS,
                    sync_as_of_utc=pd.Timestamp(
                        "2024-01-03",
                        tz="UTC",
                    ),
                    naive_timezone=PARIS,
                    incomplete_dst_policy="duplicate",
                    full=True,
                )

    def test_vintage_history_respects_explicit_utc_naive_timezone(self) -> None:
        revision = pd.Timestamp("2024-10-26 06:00", tz="UTC")
        # 02:00 is ambiguous in Europe/Paris on this date, but not on the
        # UTC grid actually returned by these Saturn forecast series.
        raw = pd.Series(
            [41.0],
            index=pd.DatetimeIndex(["2024-10-27 02:00"]),
        )

        frame = history_to_vintage_frame(
            {revision: raw},
            "power.fr.load.hourly.gw.fcst",
            PARIS,
            naive_timezone="UTC",
        )

        self.assertEqual(len(frame), 1)
        self.assertEqual(
            frame.loc[0, "value_time_utc"],
            pd.Timestamp("2024-10-27 02:00", tz="UTC"),
        )
        self.assertEqual(frame.loc[0, "revision_time_utc"], revision)


if __name__ == "__main__":
    unittest.main()
