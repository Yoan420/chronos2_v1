"""Offline official-Storm report parity without changing frozen predictions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.nuclear_report_benchmark import NuclearBenchmarkError, attach_nuclear_storm
from chronos2_hourly.storm_dashboard import STORM_DASHBOARD_COLUMN
from chronos2_modular.report import _statistics_source, average_price_summary, build_statistics_records


ZONE = "FR"
TZ = "Europe/Paris"
SERIES = "power.price.fr.euromwh.h.fcst.3mv.storm.da.cache"
DAY = pd.Timestamp("2026-09-09")


def fixture_results(*, observed_day=False):
    index = pd.date_range((DAY-pd.Timedelta(days=365)).tz_localize(TZ),
                          (DAY+pd.Timedelta(days=1)).tz_localize(TZ), freq="h", inclusive="left").tz_convert("UTC")
    future_index = local_delivery_day_index(DAY.date(), timezone=TZ)
    actual = np.full(len(index), 50.0)
    if not observed_day:
        actual[index.isin(future_index)] = np.nan
    candidate = pd.DataFrame({"timestamp": index.tz_convert(TZ), "actual": actual, "q50": 52.0})
    return {
        name: SimpleNamespace(zone=ZONE, statistics_candidate=candidate.assign(q50=value),
                              forecast_native=pd.DataFrame({"timestamp": future_index.tz_convert(TZ), "q50": value}),
                              zone_data=SimpleNamespace(target=pd.Series(actual, index=index.tz_convert(TZ))))
        for name, value in (("autonomous", 52.0), ("kalman", 51.0))
    }


def fixture_archive(tmp_path: Path, *, days=365, dst_hole=True):
    directory = tmp_path / "archive"
    (directory / "inputs").mkdir(parents=True)
    index = pd.date_range((DAY-pd.Timedelta(days=days)).tz_localize(TZ), DAY.tz_localize(TZ),
                          freq="h", inclusive="left").tz_convert("UTC")
    values = np.full(len(index), 53.0)
    hole = pd.Timestamp("2025-10-26T00:00:00Z")
    if dst_hole and hole in index:
        values[index.get_loc(hole)] = np.nan
        holes = [hole.isoformat()]
    else:
        holes = []
    path = directory / "inputs/storm_dashboard_official_statistics.parquet"
    pd.DataFrame({"delivery_start_utc": index, STORM_DASHBOARD_COLUMN: values,
                  "actual": -999999.0}).to_parquet(path, index=False)
    audit = {
        "status": "complete", "storm_primary_report_benchmark": STORM_DASHBOARD_COLUMN,
        "storm_dashboard": {
            "role": "evaluation_only_dashboard_comparator", "series": SERIES, "column": STORM_DASHBOARD_COLUMN,
            "used_for_prediction": False, "used_for_live_forecast": False,
            "normalized_artifact_path": "inputs/storm_dashboard_official_statistics.parquet",
            "normalized_artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": {"kind": "saturn_storm_day_ahead_cache_with_native_gap_fallback", "zone": ZONE,
                       "series": SERIES, "primary_series": SERIES, "series_kind": "frozen_day_ahead_cache",
                       "used_for_prediction": False, "fallback_policy": "native_only_where_day_ahead_cache_is_missing",
                       "cache_precedence": True, "fallback_series": "power.price.fr.euromwh.h.fcst.3mv.storm",
                       "extracted_at_utc": "2026-09-08T07:02:00Z"},
            "period": {"start_utc": index[0].isoformat(), "end_utc": index[-1].isoformat(), "timezone": TZ},
            "expected_hours": len(index), "available_hours": len(index)-len(holes), "missing_hours": len(holes),
            "dst": {"interpolation": False, "strict_08_fallback": False, "native_allowed_missing_utc": holes,
                    "native_allowed_missing_hours": len(holes), "native_actual_missing_matches_allowed": True},
        },
    }
    audit_path = directory / "statistics_history_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return directory, path, audit_path, audit


def test_dst_hole_uses_dedicated_hook_and_never_changes_native_statistics(tmp_path):
    archive, path, audit_path, _ = fixture_archive(tmp_path)
    before_files = path.read_bytes(), audit_path.read_bytes()
    prepared = fixture_results()
    originals = {key: deepcopy(result.statistics_candidate) for key, result in prepared.items()}
    result = attach_nuclear_storm(prepared, archive, ZONE, TZ)
    assert result["status"] == "complete"
    assert result["native_statistics_modified"] is False
    for key, item in prepared.items():
        pd.testing.assert_frame_equal(item.statistics_candidate, originals[key])
        assert item.statistics_benchmark.q50.notna().sum() == 8759
        assert item.statistics_benchmark_contract["official_dashboard_metric"] is True
        source = item.hourly_comparison_source
        assert set(source) == {"_timestamp_utc", "_timestamp_local", "actual", "q50", "_benchmark_q50"}
        assert len(source) == 8760 + 24
        assert source.loc[source._timestamp_utc == pd.Timestamp("2025-10-26T00:00Z"), "_benchmark_q50"].isna().all()
        assert source.actual.dropna().eq(50).all()  # Never borrow the cache's deliberately wrong actual column.
        assert source.iloc[-24:].actual.isna().all()
        assert source.iloc[-24:]._benchmark_q50.isna().all()
        contract = item.hourly_comparison_contract
        assert contract["id"] == "storm_official_dashboard"
        assert contract["materialization_audit"]["matched_hours"] == 8759
        assert contract["materialization_audit"]["storm_missing_dst_hours"] == 1
        assert contract["materialization_audit"]["unmatched_observed_hours"] == 1
        assert contract["materialization_audit"]["fully_paired_days"] == 364
        assert contract["materialization_audit"]["partially_paired_days"] == 1
        assert contract["materialization_audit"]["source_materialization"]["source"]["extracted_at_utc"]
    assert (path.read_bytes(), audit_path.read_bytes()) == before_files


def test_standard_statistics_keep_all_days_and_compare_exact_dst_hours(tmp_path):
    archive, _, _, _ = fixture_archive(tmp_path)
    prepared = fixture_results()
    attach_nuclear_storm(prepared, archive, ZONE, TZ)
    for key, candidate_error in (("autonomous", 2.0), ("kalman", 1.0)):
        item = prepared[key]
        source = _statistics_source(item)
        assert len(source) == 8784
        daily = [row for row in build_statistics_records([item]) if row["sample"] == "daily"]
        assert len(daily) == 366
        dst = next(row for row in daily if row["period_key"] == "2025-10-26")
        assert dst["n"] == dst["benchmark_n"] == 24  # physical autumn 25 minus audited hole
        assert dst["mae"] == candidate_error
        assert dst["benchmark_mae"] == 3.0
        assert dst["observed_mean_price"] == 50.0
        assert daily[-1]["n"] == 0
        assert daily[-1]["observed_mean_price"] is None
        assert daily[-1]["benchmark_mean_price"] is None


def test_latest_observed_day_without_storm_remains_visible_without_comparison(tmp_path):
    archive, _, _, _ = fixture_archive(tmp_path)
    prepared = fixture_results(observed_day=True)
    attach_nuclear_storm(prepared, archive, ZONE, TZ)
    daily = [row for row in build_statistics_records([prepared["autonomous"]]) if row["sample"] == "daily"]
    assert len(daily) == 365
    assert daily[-1]["period_key"] == str(DAY.date())
    assert daily[-1]["n"] == 24
    assert daily[-1]["mean_price"] == 52.0
    assert daily[-1]["observed_mean_price"] == 50.0
    assert daily[-1]["benchmark_mean_price"] is None
    assert daily[-1]["benchmark_mae"] is None


@pytest.mark.parametrize("fault", ["absent", "missing", "extra", "naive", "used", "role", "hours", "contract"])
def test_renderer_keeps_strict_coverage_unless_exact_audited_pairing(tmp_path, fault):
    archive, _, _, _ = fixture_archive(tmp_path)
    prepared = fixture_results()
    attach_nuclear_storm(prepared, archive, ZONE, TZ)
    item = prepared["autonomous"]
    if fault == "absent":
        del item.statistics_pairing_audit
    elif fault == "missing":
        item.statistics_pairing_audit["missing_benchmark_utc"] = []
    elif fault == "extra":
        item.statistics_pairing_audit["missing_benchmark_utc"].append("2025-09-09T00:00:00Z")
    elif fault == "naive":
        item.statistics_pairing_audit["missing_benchmark_utc"] = ["2025-10-26T00:00:00"]
    elif fault == "used":
        item.statistics_pairing_audit["used_for_prediction"] = True
    elif fault == "role":
        item.statistics_pairing_audit["role"] = "unverified"
    elif fault == "hours":
        item.statistics_pairing_audit["expected_hours"] -= 1
    else:
        item.statistics_benchmark_contract["official_dashboard_metric"] = False
    with pytest.raises(ValueError, match="couverture"):
        _statistics_source(item)


def add_delivery_snapshot(path, audit_path, audit, *, missing_hours=0):
    old = pd.read_parquet(path)
    delivery = local_delivery_day_index(DAY.date(), timezone=TZ)
    values = np.arange(len(delivery), dtype=float) + 40.0
    values[:missing_hours] = np.nan
    pd.concat([old, pd.DataFrame({"delivery_start_utc": delivery, STORM_DASHBOARD_COLUMN: values})],
              ignore_index=True).to_parquet(path, index=False)
    storm = audit["storm_dashboard"]
    storm["period"]["end_utc"] = delivery[-1].isoformat()
    storm["expected_hours"] += len(delivery)
    storm["available_hours"] += len(delivery)-missing_hours
    storm["missing_hours"] += missing_hours
    storm["normalized_artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    storm["delivery_placeholder"] = {
        "delivery_day_local": str(DAY.date()), "timezone": TZ,
        "allowed_missing_utc": [value.isoformat() for value in delivery[:missing_hours]],
        "allowed_missing_hours": missing_hours, "status": "pending_placeholder" if missing_hours else "complete",
        "used_for_prediction": False,
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")


def test_verified_current_snapshot_populates_day_headline_and_statistics(tmp_path):
    archive, path, audit_path, audit = fixture_archive(tmp_path)
    add_delivery_snapshot(path, audit_path, audit)
    prepared = fixture_results(observed_day=True)
    attach_nuclear_storm(prepared, archive, ZONE, TZ)
    item = prepared["autonomous"]
    summary = average_price_summary(item)
    assert summary["live_benchmark_available"] is True
    assert summary["live"]["mean"] == 52.0
    assert summary["live_benchmark"]["mean"] == 51.5
    assert summary["delta_live_benchmark"] == 0.5
    daily = [row for row in build_statistics_records([item]) if row["sample"] == "daily"]
    assert len(daily) == 365
    assert daily[-1]["benchmark_mean_price"] == 51.5
    assert daily[-1]["observed_mean_price"] == 50.0


def test_partial_current_storm_means_use_exact_same_physical_hours(tmp_path):
    archive, path, audit_path, audit = fixture_archive(tmp_path)
    add_delivery_snapshot(path, audit_path, audit, missing_hours=12)
    prepared = fixture_results(observed_day=True)
    item = prepared["autonomous"]
    item.forecast_native["q50"] = np.arange(24, dtype=float) + 50
    item.statistics_candidate.loc[item.statistics_candidate.index[-24:], "q50"] = item.forecast_native.q50.to_numpy()
    attach_nuclear_storm(prepared, archive, ZONE, TZ)
    summary = average_price_summary(item)
    assert summary["live"]["hours"] == summary["live_benchmark"]["hours"] == 12
    assert summary["live"]["mean"] == 67.5
    assert summary["live_benchmark"]["mean"] == 57.5
    assert summary["delta_live_benchmark"] == 10.0
    daily = [row for row in build_statistics_records([item]) if row["sample"] == "daily"]
    assert daily[-1]["n"] == daily[-1]["benchmark_n"] == 12
    assert daily[-1]["mean_price"] == 67.5
    assert daily[-1]["benchmark_mean_price"] == 57.5


def test_placeholder_cannot_whitelist_missing_storm_from_other_delivery_day(tmp_path):
    archive, path, audit_path, audit = fixture_archive(tmp_path)
    add_delivery_snapshot(path, audit_path, audit, missing_hours=24)
    audit["storm_dashboard"]["delivery_placeholder"]["delivery_day_local"] = "2026-09-08"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(NuclearBenchmarkError, match="dernier jour"):
        attach_nuclear_storm(fixture_results(), archive, ZONE, TZ)


def test_shorter_verified_archive_keeps_full_candidate_support(tmp_path):
    archive, _, _, _ = fixture_archive(tmp_path, days=10, dst_hole=False)
    prepared = fixture_results()
    result = attach_nuclear_storm(prepared, archive, ZONE, TZ)
    assert len(prepared["autonomous"].hourly_comparison_source) == 8784
    assert result["variants"]["autonomous"]["matched_hours"] == 240
    assert result["variants"]["autonomous"]["unmatched_observed_hours"] == 8760-240
    assert prepared["autonomous"].statistics_benchmark.q50.notna().sum() == 240


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_delivery_placeholder_preserves_every_physical_dst_hour(tmp_path, monkeypatch, day, hours):
    monkeypatch.setattr(sys.modules[__name__], "DAY", pd.Timestamp(day))
    archive, _, _, _ = fixture_archive(tmp_path)
    prepared = fixture_results()
    attach_nuclear_storm(prepared, archive, ZONE, TZ)
    source = prepared["autonomous"].hourly_comparison_source
    future = source.loc[source._timestamp_local.dt.date == pd.Timestamp(day).date()]
    assert len(future) == hours
    assert not future._timestamp_utc.duplicated().any()
    assert future.actual.isna().all()
    assert future._benchmark_q50.isna().all()
    assert len(prepared["autonomous"].forecast_native) == hours


def test_observed_delivery_uses_exact_latest_365_days_and_keeps_source_data(tmp_path):
    archive, _, _, _ = fixture_archive(tmp_path)
    prepared = fixture_results(observed_day=True)
    before = prepared["autonomous"].statistics_candidate.copy(deep=True)
    attach_nuclear_storm(prepared, archive, ZONE, TZ)
    frame = prepared["autonomous"].hourly_comparison_source
    assert len(frame) == 8760
    assert frame._timestamp_local.iloc[0].date() == (DAY-pd.Timedelta(days=364)).date()
    assert frame.iloc[-24:].actual.eq(50).all()
    assert frame.iloc[-24:]._benchmark_q50.isna().all()
    pd.testing.assert_frame_equal(prepared["autonomous"].statistics_candidate, before)


@pytest.mark.parametrize("fault", ["sha", "source", "zone", "strict08", "used", "dst", "missingcount", "traversal"])
def test_present_invalid_storm_is_rejected_without_partial_attachment(tmp_path, fault):
    archive, _, audit_path, audit = fixture_archive(tmp_path)
    storm = audit["storm_dashboard"]
    if fault == "sha":
        storm["normalized_artifact_sha256"] = "0"*64
    elif fault == "source":
        storm["source"]["series"] = SERIES.replace(".cache", ".basecase")
    elif fault == "zone":
        storm["source"]["zone"] = "DE"
    elif fault == "strict08":
        storm["dst"]["strict_08_fallback"] = True
    elif fault == "used":
        storm["used_for_prediction"] = True
    elif fault == "dst":
        storm["dst"]["native_allowed_missing_utc"] = []
    elif fault == "missingcount":
        storm["available_hours"] = 8760
    else:
        storm["normalized_artifact_path"] = "../outside.parquet"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    prepared = fixture_results()
    with pytest.raises(NuclearBenchmarkError):
        attach_nuclear_storm(prepared, archive, ZONE, TZ)
    assert all(not hasattr(result, "hourly_comparison_source") for result in prepared.values())


def test_non_dst_missing_hour_cannot_be_whitelisted_as_dst(tmp_path):
    archive, path, audit_path, audit = fixture_archive(tmp_path, dst_hole=False)
    frame = pd.read_parquet(path)
    frame.loc[10, STORM_DASHBOARD_COLUMN] = np.nan
    frame.to_parquet(path, index=False)
    storm = audit["storm_dashboard"]
    storm.update(normalized_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), available_hours=len(frame)-1, missing_hours=1)
    storm["dst"].update(native_allowed_missing_hours=1, native_allowed_missing_utc=[frame.delivery_start_utc.iloc[10].isoformat()])
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(NuclearBenchmarkError, match="automne"):
        attach_nuclear_storm(fixture_results(), archive, ZONE, TZ)


@pytest.mark.parametrize("fault", ["duplicate", "naive", "inf"])
def test_invalid_physical_snapshot_is_rejected_even_with_updated_hash(tmp_path, fault):
    archive, path, audit_path, audit = fixture_archive(tmp_path)
    frame = pd.read_parquet(path)
    if fault == "duplicate":
        frame.loc[1, "delivery_start_utc"] = frame.delivery_start_utc.iloc[0]
    elif fault == "naive":
        frame["delivery_start_utc"] = frame.delivery_start_utc.dt.tz_localize(None)
    else:
        frame.loc[0, STORM_DASHBOARD_COLUMN] = np.inf
    frame.to_parquet(path, index=False)
    audit["storm_dashboard"]["normalized_artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(NuclearBenchmarkError):
        attach_nuclear_storm(fixture_results(), archive, ZONE, TZ)


def test_mismatched_observations_between_variants_are_rejected(tmp_path):
    archive, _, _, _ = fixture_archive(tmp_path)
    prepared = fixture_results()
    prepared["kalman"].statistics_candidate.loc[0, "actual"] = 60
    with pytest.raises(NuclearBenchmarkError, match="memes heures et observations"):
        attach_nuclear_storm(prepared, archive, ZONE, TZ)
    assert not hasattr(prepared["autonomous"], "hourly_comparison_source")


def test_absent_snapshot_is_explicitly_unavailable_without_hooks(tmp_path):
    prepared = fixture_results()
    result = attach_nuclear_storm(prepared, tmp_path, ZONE, TZ)
    assert result["status"] == "unavailable"
    assert not hasattr(prepared["autonomous"], "hourly_comparison_source")


@pytest.mark.parametrize("missing", ["artifact", "audit"])
def test_incomplete_existing_pair_fails_closed(tmp_path, missing):
    archive, path, audit_path, _ = fixture_archive(tmp_path)
    # Test fixtures are disposable; never touch actual archive files.
    (path if missing == "artifact" else audit_path).unlink()
    with pytest.raises(NuclearBenchmarkError):
        attach_nuclear_storm(fixture_results(), archive, ZONE, TZ)
