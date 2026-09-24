from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.storm_dashboard import (
    COMPARATOR_AUDIT_NAME,
    STORM_DASHBOARD_ARTIFACT,
    STORM_DASHBOARD_CACHE_SERIES_BY_ZONE,
    STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE,
    STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE,
    STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE,
    STORM_DASHBOARD_PRIMARY_SERIES_BY_ZONE,
    STORM_DASHBOARD_COLUMN,
    STORM_LEGACY_STRICT_COLUMN,
    STORM_STRICT_08_COLUMN,
    build_dashboard_comparator,
    create_report_only_copy,
    fetch_native_dashboard_snapshot,
    load_dashboard_from_basecase_vintages,
    load_materialized_dashboard_comparator,
    normalize_native_dashboard_series,
    storm_dashboard_series,
)


def test_native_dashboard_uses_local_clock_and_never_fabricates_dst_fold() -> None:
    expected = local_delivery_day_index("2025-10-26")
    local = expected.tz_convert("Europe/Paris")
    # Native Saturn exposes one naive local 02:00 rather than two folds.
    naive = pd.DatetimeIndex(local.tz_localize(None)).drop_duplicates()
    raw = pd.Series(np.arange(len(naive), dtype=float), index=naive)
    actual = pd.Series(0.0, index=expected)

    comparator = normalize_native_dashboard_series(
        raw,
        zone="FR",
        expected_index=expected,
        actual=actual,
    )

    assert comparator.audit["series"] == (
        "power.price.fr.euromwh.h.fcst.3mv.storm.da.cache"
    )
    assert comparator.audit["available_hours"] == 24
    assert comparator.audit["missing_hours"] == 1
    assert comparator.audit["dst"]["interpolation"] is False
    assert comparator.audit["dst"]["strict_08_fallback"] is False
    assert comparator.audit["source"]["naive_timezone"] == "Europe/Paris"
    missing = comparator.values[comparator.values.isna()].index
    assert len(missing) == 1
    assert missing[0].tz_convert("Europe/Paris").hour == 2


def test_native_dashboard_rejects_a_non_dst_missing_hour() -> None:
    expected = local_delivery_day_index("2026-01-02")
    local = expected.tz_convert("Europe/Paris").tz_localize(None)
    raw = pd.Series(np.arange(len(local), dtype=float), index=local).drop(
        local[12]
    )

    with pytest.raises(ValueError, match="missing non-DST"):
        normalize_native_dashboard_series(
            raw,
            zone="FR",
            expected_index=expected,
            actual=pd.Series(0.0, index=expected),
        )


def test_dashboard_comparator_accepts_only_explicit_actual_placeholder() -> None:
    expected = local_delivery_day_index("2026-01-02")
    values = pd.Series(80.0, index=expected)
    actual = pd.Series(np.nan, index=expected)

    with pytest.raises(ValueError, match="canonical actuals are incomplete"):
        build_dashboard_comparator(
            values,
            expected_index=expected,
            actual=actual,
            source={"kind": "offline_test"},
        )

    comparator = build_dashboard_comparator(
        values,
        expected_index=expected,
        actual=actual,
        source={"kind": "offline_test"},
        allowed_missing_actual_index=expected,
    )

    assert comparator.audit["actual_available_hours"] == 0
    assert comparator.audit["actual_missing_hours"] == 24
    assert comparator.audit["allowed_missing_actual_hours"] == 24
    assert comparator.audit["metrics"]["n"] == 0


def test_native_dashboard_rejects_wrong_requested_series() -> None:
    expected = local_delivery_day_index("2026-01-02")
    local = expected.tz_convert("Europe/Paris").tz_localize(None)

    with pytest.raises(ValueError, match="requested_series mismatch"):
        normalize_native_dashboard_series(
            pd.Series(np.arange(len(local), dtype=float), index=local),
            zone="FR",
            expected_index=expected,
            actual=pd.Series(0.0, index=expected),
            source={
                "requested_series": "power.price.de.euromwh.h.fcst.3mv.storm"
            },
        )


def test_report_copy_accepts_only_the_audited_native_dst_gap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_native"
    source.mkdir()
    expected = local_delivery_day_index("2025-10-26")
    actual = pd.Series(np.arange(len(expected), dtype=float), index=expected)
    local = expected.tz_convert("Europe/Paris")
    naive = pd.DatetimeIndex(local.tz_localize(None)).drop_duplicates()
    comparator = normalize_native_dashboard_series(
        pd.Series(np.arange(len(naive), dtype=float), index=naive),
        zone="FR",
        expected_index=expected,
        actual=actual,
    )
    pd.DataFrame(
        {
            "delivery_start_utc": expected,
            "actual": actual,
            STORM_LEGACY_STRICT_COLUMN: actual + 3.0,
        }
    ).to_csv(
        source / "statistics_history_hourly.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        {"delivery_start_utc": expected, "q50": actual}
    ).to_csv(source / "forecast_hourly_fr.csv", index=False)
    (source / "statistics_history_audit.json").write_text("{}")
    (source / "run_manifest.json").write_text("{}")

    output, _ = create_report_only_copy(
        source_run_dir=source,
        output_dir=tmp_path / "native_report",
        comparator=comparator,
        regenerate_html=False,
    )
    injected = pd.read_csv(output / "statistics_history_hourly.csv.gz")
    assert injected[STORM_DASHBOARD_COLUMN].isna().sum() == 1


def _basecase_vintages(
    expected: pd.DatetimeIndex,
    actual: pd.Series,
) -> pd.DataFrame:
    cutoff = pd.Timestamp("2026-01-01 23:00:00Z")
    rows: list[dict[str, object]] = []
    vintages = (
        (cutoff - pd.Timedelta(hours=2), cutoff - pd.Timedelta(hours=2), 3.0),
        (
            cutoff - pd.Timedelta(minutes=10),
            cutoff - pd.Timedelta(minutes=5),
            1.0,
        ),
        # Equality on either marker is deliberately ineligible.
        (cutoff - pd.Timedelta(minutes=1), cutoff, 80.0),
        (cutoff, cutoff - pd.Timedelta(minutes=1), 90.0),
        (cutoff + pd.Timedelta(minutes=1), cutoff + pd.Timedelta(minutes=1), 99.0),
    )
    for position, delivery in enumerate(expected):
        for snapshot, revision, shift in vintages:
            rows.append(
                {
                    "value_time_utc": delivery,
                    "snapshot_time_utc": snapshot,
                    "revision_time_utc": revision,
                    "value": float(actual.iloc[position] + shift),
                }
            )
    return pd.DataFrame(rows)


def test_basecase_selector_matches_strict_civil_midnight_contract(
    tmp_path: Path,
) -> None:
    expected = local_delivery_day_index("2026-01-02")
    actual = pd.Series(np.arange(len(expected), dtype=float) + 10.0, index=expected)
    vintages = _basecase_vintages(expected, actual)
    path = tmp_path / "basecase.parquet"
    vintages.to_parquet(path, index=False)

    comparator = load_dashboard_from_basecase_vintages(
        path,
        expected_index=expected,
        actual=actual,
    )

    np.testing.assert_allclose(comparator.values, actual + 1.0)
    assert comparator.audit["coverage"] == 1.0
    assert comparator.audit["metrics"]["mae"] == 1.0
    source = comparator.audit["source"]
    assert source["kind"] == (
        "saturn_basecase_pit_proxy_for_dashboard_cache_gap"
    )
    assert source["series"] == (
        "power.price.fr.euromwh.h.fcst.3mv.storm.da.cache"
    )
    assert source["series_kind"] == "frozen_day_ahead_cache"
    assert source["materialization_series"] == (
        "power.price.fr.euromwh.h.fcst.3mv.storm.da.basecase"
    )
    assert source["comparison_operator"] == "strict_less_than"
    assert source["selected_rows"] == len(expected)
    assert source["snapshot_cutoff_violations"] == 0
    assert source["revision_cutoff_violations"] == 0
    assert source["minimum_cutoff_lag_minutes"] == 5.0


@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL"])
def test_dashboard_native_series_identifiers_are_exact_and_zone_aware(
    zone: str,
) -> None:
    expected = (
        f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm.da.cache"
    )
    assert storm_dashboard_series(zone) == expected
    assert STORM_DASHBOARD_CACHE_SERIES_BY_ZONE[zone] == expected
    assert STORM_DASHBOARD_PRIMARY_SERIES_BY_ZONE[zone] == expected
    assert STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE[zone] == (
        f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm"
    )


@pytest.mark.parametrize(
    ("zone", "timezone", "primary"),
    [
        ("FR", "Europe/Paris", "41377_native"),
        ("DE", "Europe/Berlin", "41376_native"),
        ("BE", "Europe/Brussels", "41378_native"),
        ("NL", "Europe/Amsterdam", "41379_native"),
    ],
)
def test_native_dashboard_fetch_and_dst_contract_are_zone_specific(
    zone: str,
    timezone: str,
    primary: str,
) -> None:
    expected = local_delivery_day_index("2025-10-26", timezone=timezone)
    local_naive = expected.tz_convert(timezone).tz_localize(None).drop_duplicates()
    raw = pd.Series(np.arange(len(local_naive), dtype=float), index=local_naive)
    cache_index = local_naive.tz_localize(
        timezone, ambiguous=False
    ).tz_convert("UTC")
    cache = pd.Series(np.arange(len(cache_index), dtype=float), index=cache_index)

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def get(self, series: str, **kwargs: object) -> pd.Series:
            self.calls.append((series, kwargs))
            return cache if series.endswith(".da.cache") else raw

    client = FakeClient()
    fetched, source = fetch_native_dashboard_snapshot(
        client,
        zone=zone,
        expected_index=expected,
        extracted_at_utc=pd.Timestamp("2026-08-13T12:00:00Z"),
    )

    exact_series = (
        f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm.da.cache"
    )
    fallback_series = f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm"
    assert len(fetched) == len(raw)
    assert client.calls[0][0] == exact_series
    assert client.calls[1][0] == fallback_series
    assert client.calls[0][1]["nocache"] is True
    assert client.calls[0][1]["live"] is False
    assert client.calls[1][1]["nocache"] is True
    assert client.calls[1][1]["live"] is True
    assert source["requested_series"] == exact_series
    assert source["nocache"] is True
    assert source["cache_live_recomputation"] is False
    assert source["fallback_live_recomputation"] is True
    assert source["primary_series"] == exact_series
    assert source["fallback_series"] == fallback_series
    assert source["fallback_primary_series"] == primary
    assert source["naive_timezone"] == timezone
    assert STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE[zone] == primary
    assert STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE[zone] == timezone

    comparator = normalize_native_dashboard_series(
        fetched,
        zone=zone,
        expected_index=expected,
        actual=pd.Series(0.0, index=expected),
        source=source,
    )
    assert comparator.audit["expected_hours"] == 25
    assert comparator.audit["available_hours"] == 24
    assert comparator.audit["missing_hours"] == 1
    assert comparator.audit["period"]["timezone"] == timezone
    assert comparator.audit["dst"]["native_actual_missing_matches_allowed"] is True


def test_dashboard_native_series_rejects_unverified_zone() -> None:
    with pytest.raises(ValueError, match="day-ahead dashboard"):
        storm_dashboard_series("ES")


def test_fr_20260826_uses_dashboard_cache_148_not_native_12881() -> None:
    expected = local_delivery_day_index("2026-08-26")
    cache_mean = 147.99454095833335
    native_mean = 128.80891033333333
    cache = pd.Series(cache_mean, index=expected)
    native_index = expected.tz_convert("Europe/Paris").tz_localize(None)
    native = pd.Series(native_mean, index=native_index)

    class FakeClient:
        def get(self, series: str, **_kwargs: object) -> pd.Series:
            return cache if series.endswith(".da.cache") else native

    raw, source = fetch_native_dashboard_snapshot(
        FakeClient(), zone="FR", expected_index=expected
    )
    comparator = normalize_native_dashboard_series(
        raw,
        zone="FR",
        expected_index=expected,
        actual=pd.Series(0.0, index=expected),
        source=source,
    )

    assert len(comparator.values) == 24
    assert comparator.values.mean() == pytest.approx(cache_mean)
    assert comparator.values.mean() != pytest.approx(native_mean)
    assert source["cache_available_hours"] == 24
    assert source["fallback_used_hours"] == 0


def test_basecase_selector_requires_exact_statistics_timeline(
    tmp_path: Path,
) -> None:
    expected = local_delivery_day_index("2026-01-02")
    actual = pd.Series(np.arange(len(expected), dtype=float), index=expected)
    vintages = _basecase_vintages(expected, actual)
    first_hour = expected[0]
    cutoff = pd.Timestamp("2026-01-01 23:00:00Z")
    # Leave only vintages at/after the strict cutoff for one delivery hour.
    vintages = vintages.loc[
        (vintages["value_time_utc"] != first_hour)
        | (vintages["snapshot_time_utc"] >= cutoff)
        | (vintages["revision_time_utc"] >= cutoff)
    ]
    path = tmp_path / "basecase_missing.parquet"
    vintages.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="exact Statistics timeline"):
        load_dashboard_from_basecase_vintages(
            path,
            expected_index=expected,
            actual=actual,
        )


def test_materialized_dashboard_audits_partial_dst_fold_without_fallback(
    tmp_path: Path,
) -> None:
    expected = local_delivery_day_index("2025-10-26")
    actual = pd.Series(np.arange(len(expected), dtype=float), index=expected)
    # Explicitly omit the second 02:00 local fold from the old formula.
    local = expected.tz_convert("Europe/Paris")
    fold_positions = np.flatnonzero(local.hour == 2)
    keep = np.ones(len(expected), dtype=bool)
    keep[fold_positions[1]] = False
    materialized = tmp_path / "storm.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": expected[keep],
            "value": actual.to_numpy()[keep] + 2.0,
        }
    ).to_csv(materialized, index=False)

    comparator = load_materialized_dashboard_comparator(
        materialized,
        expected_index=expected,
        actual=actual,
        minimum_coverage=0.95,
        maximum_missing_hours=1,
    )

    assert comparator.audit["expected_hours"] == 25
    assert comparator.audit["available_hours"] == 24
    assert comparator.audit["missing_hours"] == 1
    assert comparator.audit["metrics"]["n"] == 24
    assert comparator.audit["metrics"]["mae"] == 2.0
    assert comparator.audit["dst"]["expected_local_day_hour_histogram"] == {
        "25": 1
    }
    assert comparator.audit["dst"]["available_local_day_hour_histogram"] == {
        "24": 1
    }
    assert comparator.audit["dst"]["strict_08_fallback"] is False
    assert pd.isna(comparator.values.iloc[fold_positions[1]])


def test_report_only_copy_preserves_strict_08_and_source_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    expected = local_delivery_day_index("2026-01-02")
    actual = pd.Series(np.arange(len(expected), dtype=float) + 40.0, index=expected)
    history = pd.DataFrame(
        {
            "delivery_start_utc": expected,
            "actual": actual.to_numpy(),
            STORM_LEGACY_STRICT_COLUMN: actual.to_numpy() + 3.0,
        }
    )
    history_path = source / "statistics_history_hourly.csv.gz"
    history.to_csv(history_path, index=False, compression="gzip")
    source_history_bytes = history_path.read_bytes()
    forecast_path = source / "forecast_hourly_fr.csv"
    pd.DataFrame(
        {"delivery_start_utc": expected, "q50": actual.to_numpy()}
    ).to_csv(forecast_path, index=False)
    source_forecast_bytes = forecast_path.read_bytes()
    (source / "statistics_history_audit.json").write_text(
        json.dumps(
            {
                "storm": {"cutoff": "civil D-1 08:00 Europe/Paris"},
                "report_scope_note": "Source scellée.",
            }
        ),
        encoding="utf-8",
    )
    (source / "run_manifest.json").write_text(
        json.dumps({"run_type": "live_statistics_report_snapshot"}),
        encoding="utf-8",
    )

    comparator = build_dashboard_comparator(
        pd.Series(actual.to_numpy() + 1.0, index=expected),
        expected_index=expected,
        actual=actual,
        source={"kind": "offline_test"},
        minimum_coverage=1.0,
        maximum_missing_hours=0,
    )
    output, report = create_report_only_copy(
        source_run_dir=source,
        output_dir=tmp_path / "report_only",
        comparator=comparator,
        regenerate_html=False,
    )

    assert report is None
    assert history_path.read_bytes() == source_history_bytes
    assert forecast_path.read_bytes() == source_forecast_bytes
    injected = pd.read_csv(output / "statistics_history_hourly.csv.gz")
    np.testing.assert_allclose(
        injected[STORM_STRICT_08_COLUMN], actual.to_numpy() + 3.0
    )
    np.testing.assert_allclose(
        injected[STORM_DASHBOARD_COLUMN], actual.to_numpy() + 1.0
    )
    np.testing.assert_allclose(
        injected[STORM_LEGACY_STRICT_COLUMN], actual.to_numpy() + 3.0
    )
    audit = json.loads(
        (output / "statistics_history_audit.json").read_text(encoding="utf-8")
    )
    assert audit["storm_primary_report_benchmark"] == STORM_DASHBOARD_COLUMN
    assert audit["storm_dashboard"]["metrics"]["mae"] == 1.0
    assert audit["storm_strict_08"]["metrics"]["mae"] == 3.0
    assert audit["storm_used_for_prediction"] is False
    copy_audit = json.loads(
        (output / COMPARATOR_AUDIT_NAME).read_text(encoding="utf-8")
    )
    assert copy_audit["source_modified"] is False
    assert copy_audit["used_for_prediction"] is False
    assert (output / STORM_DASHBOARD_ARTIFACT).is_file()
    assert (output / "artifact_checksums.json").is_file()
