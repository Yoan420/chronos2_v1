"""Latest report-only source refresh: no network, model or frozen-cache writes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
import chronos2_hourly.nuclear_reporting_refresh as refresh
from chronos2_hourly.reporting_observations import (
    EPEX_REPORTING_POLICY, EPEX_REPORTING_SERIES_BY_ZONE, epex_reporting_identity,
)
from chronos2_hourly.storm_dashboard import STORM_DASHBOARD_COLUMN, storm_dashboard_series


DAY = "2026-09-09"
TZ = "Europe/Paris"
TARGET = "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh"
CONFIG = {"data": {"saturn_url": "https://test.invalid/api", "saturn_author": "test"},
          "zones": {"FR": {"target": {"series": TARGET}}}}


def sources(monkeypatch, *, day=DAY, actual_count=0, storm_count=0, dst_hole=True,
            actual_hole=None, storm_hole=None, wrong_storm_series=False):
    expected, current = refresh._support(day, TZ)
    observed = pd.Series(100.0, index=expected)
    observed.loc[current] = np.nan
    observed.loc[current[:actual_count]] = 120.0
    storm = pd.Series(105.0, index=expected)
    storm.loc[current] = np.nan
    storm.loc[current[:storm_count]] = 125.0
    if dst_hole:
        civil = expected.tz_convert(TZ).tz_localize(None)
        folds = expected[civil.tz_localize(TZ, ambiguous=False).tz_convert("UTC") != expected]
        storm.loc[folds.difference(current)] = np.nan
    if actual_hole is not None:
        observed.iloc[actual_hole] = np.nan
    if storm_hole is not None:
        storm.iloc[storm_hole] = np.nan
    calls = []

    def actual_fetch(client, *, spec, expected_index, extracted_at_utc):
        calls.append(("actual", client, spec, expected_index))
        return observed.copy(), {**epex_reporting_identity(spec.zone, spec.timezone),
                                 "extracted_at_utc": str(extracted_at_utc)}

    def storm_fetch(client, *, zone, expected_index, extracted_at_utc):
        calls.append(("storm", client, zone, expected_index))
        series = storm_dashboard_series(zone)
        return storm.copy(), {"kind": "saturn_storm_day_ahead_cache_with_native_gap_fallback",
                              "series": "wrong" if wrong_storm_series else series,
                              "primary_series": series, "zone": zone,
                              "fallback_series": "power.price.fr.euromwh.h.fcst.3mv.storm",
                              "extracted_at_utc": str(extracted_at_utc), "used_for_prediction": False}

    monkeypatch.setattr(refresh, "_fetch_epex_observed", actual_fetch)
    monkeypatch.setattr(refresh, "fetch_native_dashboard_snapshot", storm_fetch)
    return expected, current, observed, storm, calls


def run_refresh(tmp_path, *, day=DAY):
    return refresh.refresh_nuclear_reporting_sources(
        CONFIG, "FR", TZ, day, tmp_path / "report_only/sources", client=object())


@pytest.mark.parametrize("permanent", [False, True])
def test_publication_retries_staging_without_refetch_and_preserves_permanent_failure(tmp_path, monkeypatch, permanent):
    from chronos2_hourly import atomic_directory as atomic
    from test_atomic_directory import windows_error

    _, _, _, _, calls = sources(monkeypatch)
    rename = atomic._rename_no_replace
    attempts = []

    def locked(source, target):
        attempts.append((source, target))
        assert len(list(source.rglob("*.parquet"))) == 2
        if permanent or len(attempts) < 3:
            raise windows_error(5)
        rename(source, target)

    monkeypatch.setattr(atomic, "_rename_no_replace", locked)
    monkeypatch.setattr(atomic.time, "sleep", lambda _: None)
    if permanent:
        with pytest.raises(atomic.AtomicDirectoryPublishError):
            run_refresh(tmp_path)
        assert len(attempts) == 7
        stage, final = attempts[0]
        assert stage.is_dir() and not final.exists()
        audit = json.loads((stage / "statistics_history_audit.json").read_text())
        assert audit["snapshot_directory"] == str(final)
        # The exact already-fetched snapshot can be published after release,
        # with its original audit paths/hashes and no additional source calls.
        monkeypatch.setattr(atomic, "_rename_no_replace", rename)
        atomic.publish_directory_no_replace(stage, final)
        frame = pd.read_parquet(final / "inputs/observed_latest.parquet")
        values = pd.Series(frame.actual.to_numpy(), index=pd.DatetimeIndex(frame.timestamp))
    else:
        values, final, audit = run_refresh(tmp_path)
        assert len(attempts) == 3
    assert [call[0] for call in calls] == ["actual", "storm"]
    assert len({source for source, _ in attempts}) == 1
    assert refresh.verify_refreshed_observations(values, audit, zone="FR", timezone=TZ, delivery_day=DAY)["status"] == "verified"


def test_latest_refresh_publishes_separate_audited_snapshot_and_keeps_source_inputs(tmp_path, monkeypatch):
    expected, current, observed, storm, calls = sources(monkeypatch)
    before = observed.copy(), storm.copy(), deepcopy(CONFIG)
    frozen = tmp_path / "frozen_prediction.parquet"
    frozen.write_bytes(b"immutable prediction")
    values, directory, audit = run_refresh(tmp_path)
    assert directory.parent == tmp_path / "report_only/sources"
    assert directory.is_dir()
    assert len(list(directory.rglob("*.parquet"))) == 2
    assert values.index.equals(expected) and values.loc[current].isna().all()
    assert audit["status"] == "complete" and audit["old_snapshot_fallback"] is False
    assert audit["storm_dashboard"]["missing_hours"] == 25
    assert audit["storm_dashboard"]["dst"]["native_allowed_missing_hours"] == 1
    assert audit["storm_dashboard"]["delivery_placeholder"]["allowed_missing_hours"] == 24
    assert audit["observed"]["source"]["current_delivery_actual_status"] == "pending_placeholder"
    assert audit["observed"]["series"] == EPEX_REPORTING_SERIES_BY_ZONE["FR"]
    assert audit["schema_version"] == 2 and audit["observation_policy"] == EPEX_REPORTING_POLICY
    assert audit["training_target_series"] == TARGET
    for key, value in epex_reporting_identity("FR", TZ).items():
        assert audit["observed"]["source"][key] == value
    assert calls[0][0] == "actual" and calls[1][0] == "storm"
    assert calls[0][1] is calls[1][1]
    assert calls[0][2].history_contract.target_series == TARGET
    assert frozen.read_bytes() == b"immutable prediction"
    pd.testing.assert_series_equal(observed, before[0])
    pd.testing.assert_series_equal(storm, before[1])
    assert CONFIG == before[2]
    verified = refresh.verify_refreshed_observations(values, audit, zone="FR", timezone=TZ, delivery_day=DAY)
    assert verified["status"] == "verified" and verified["compared_hours"] == len(expected)


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_pending_and_complete_dst_delivery_days_preserve_physical_hours(tmp_path, monkeypatch, day, hours):
    expected, current, *_ = sources(monkeypatch, day=day, actual_count=hours-1, storm_count=hours-1)
    values, directory, audit = run_refresh(tmp_path, day=day)
    assert len(current) == hours
    assert values.loc[current].isna().all()
    assert audit["observed"]["source"]["current_delivery_raw_available_hours"] == hours-1
    assert audit["storm_dashboard"]["delivery_placeholder"]["allowed_missing_hours"] == 1
    assert not values.index.duplicated().any()
    sources(monkeypatch, day=day, actual_count=hours, storm_count=hours)
    values2, directory2, audit2 = run_refresh(tmp_path, day=day)
    assert directory2 != directory and directory.is_dir()
    assert values2.loc[current].eq(120).all()
    assert audit2["observed"]["source"]["current_delivery_actual_status"] == "complete"
    assert audit2["storm_dashboard"]["delivery_placeholder"]["status"] == "complete"


def test_partial_actual_day_is_empty_even_when_storm_is_published(tmp_path, monkeypatch):
    _, current, *_ = sources(monkeypatch, actual_count=12, storm_count=24)
    values, _, audit = run_refresh(tmp_path)
    assert values.loc[current].isna().all()
    assert audit["storm_dashboard"]["actual_missing_hours"] == 24
    assert audit["storm_dashboard"]["delivery_placeholder"]["allowed_missing_hours"] == 0


@pytest.mark.parametrize("fault", ["actual_hole", "storm_hole", "wrong_storm_series"])
def test_invalid_historical_source_fails_without_publishing_or_cache_fallback(tmp_path, monkeypatch, fault):
    sources(monkeypatch, **{fault: True if fault == "wrong_storm_series" else 100})
    with pytest.raises(refresh.NuclearReportingRefreshError):
        run_refresh(tmp_path)
    assert not (tmp_path / "report_only/sources").exists()


def test_complete_autumn_storm_folds_are_retained_not_forced_missing(tmp_path, monkeypatch):
    sources(monkeypatch, actual_count=24, storm_count=24, dst_hole=False)
    _, directory, audit = run_refresh(tmp_path)
    assert audit["storm_dashboard"]["missing_hours"] == 0
    assert audit["storm_dashboard"]["dst"]["native_allowed_missing_hours"] == 0
    frame = pd.read_parquet(directory / "inputs/storm_dashboard_official_statistics.parquet")
    assert frame[STORM_DASHBOARD_COLUMN].notna().all()


def test_client_created_from_config_and_author_override(tmp_path, monkeypatch):
    sources(monkeypatch)
    monkeypatch.setenv("SATURN_AUTHOR", "override")
    seen = []
    monkeypatch.setattr(refresh, "create_saturn_client", lambda url, author: seen.append((url, author)) or object())
    refresh.refresh_nuclear_reporting_sources(CONFIG, "FR", TZ, DAY, tmp_path / "reports")
    assert seen == [("https://test.invalid/api", "override")]


def test_fetch_failure_preserves_existing_snapshot_and_does_not_return_it(tmp_path, monkeypatch):
    sources(monkeypatch)
    _, directory, _ = run_refresh(tmp_path)
    before = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}

    def failure(*args, **kwargs):
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(refresh, "_fetch_epex_observed", failure)
    with pytest.raises(RuntimeError, match="unavailable"):
        run_refresh(tmp_path)
    assert len(list(directory.parent.iterdir())) == 1
    assert before == {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_sealed_artifact_cannot_be_used_as_output_root(tmp_path):
    archive = tmp_path / "sealed"
    archive.mkdir()
    (archive / "artifact_checksums.json").write_text("{}")
    with pytest.raises(refresh.NuclearReportingRefreshError, match="gele"):
        refresh.refresh_nuclear_reporting_sources(CONFIG, "FR", TZ, DAY, archive, client=object())


@pytest.mark.parametrize("fault", ["value", "sha", "path", "zone", "source", "date", "pending", "nan_history"])
def test_observation_verifier_rejects_changed_values_or_invalid_provenance(tmp_path, monkeypatch, fault):
    sources(monkeypatch)
    values, directory, audit = run_refresh(tmp_path)
    if fault == "value":
        values.iloc[10] += 0.01
    elif fault == "sha":
        audit["observed"]["artifact_sha256"] = "0"*64
    elif fault == "path":
        audit["observed"]["artifact_path"] = str(directory / "../other.parquet")
    elif fault == "zone":
        audit["observed"]["zone"] = "DE"
    elif fault == "source":
        audit["observed"]["source"]["used_for_prediction"] = True
    elif fault == "date":
        audit["observed"]["source"]["extracted_at_utc"] = "2026-09-08"
    elif fault == "pending":
        audit["observed"]["source"]["current_delivery_actual_status"] = "complete"
    else:
        path = Path(audit["observed"]["artifact_path"])
        frame = pd.read_parquet(path)
        frame.loc[10, "actual"] = np.nan
        frame.to_parquet(path, index=False)
        values.iloc[10] = np.nan
        audit["observed"]["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(refresh.NuclearReportingRefreshError):
        refresh.verify_refreshed_observations(values, audit, zone="FR", timezone=TZ, delivery_day=DAY)


def test_verifier_allows_older_fit_context_but_never_changes_it(tmp_path, monkeypatch):
    sources(monkeypatch)
    values, _, audit = run_refresh(tmp_path)
    older = pd.Series([5.0], index=[values.index[0]-pd.Timedelta(hours=1)])
    expanded = pd.concat([older, values])
    before = expanded.copy()
    result = refresh.verify_refreshed_observations(expanded, audit, zone="FR", timezone=TZ, delivery_day=DAY)
    assert result["compared_hours"] == len(values)
    pd.testing.assert_series_equal(expanded, before)


def test_new_snapshot_contract_roundtrips_through_benchmark_loader(tmp_path, monkeypatch):
    sources(monkeypatch)
    _, directory, audit = run_refresh(tmp_path)
    from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
    values, source, provenance = _load_verified_snapshot(directory, zone="FR", timezone=TZ)
    assert values.isna().sum() == 25
    assert source["delivery_placeholder"]["allowed_missing_hours"] == 24
    assert provenance["artifact_sha256"] == audit["storm_dashboard"]["normalized_artifact_sha256"]


@pytest.mark.parametrize("current_count", [0, 12, 24])
def test_es_refreshes_and_verifies_observations_without_fabricating_storm(tmp_path, monkeypatch, current_count):
    timezone = "Europe/Madrid"
    target = "power.price.da.es.bzn.hourly.entsoe.utc.cdh.eurmwh"
    config = {"data": CONFIG["data"], "zones": {"ES": {"target": {"series": target}}}}
    expected, current = refresh._support(DAY, timezone)
    raw = pd.Series(50.0, index=expected)
    raw.loc[current] = np.nan
    raw.loc[current[:current_count]] = 60.0
    requests = []

    def fetch_actual(client, *, spec, expected_index, extracted_at_utc):
        assert spec.zone == "ES" and spec.timezone == timezone
        assert spec.history_contract.target_series == target
        requests.append("observed")
        return raw.copy(), {"kind": "saturn_target_latest_extraction", "series": target,
                            "zone": "ES", "nocache": True, "live_recomputation": True,
                            "extracted_at_utc": str(extracted_at_utc), "used_for_prediction": False}

    def forbidden_storm(*args, **kwargs):
        pytest.fail("ES has no verified Storm comparator; no request may be attempted")

    monkeypatch.setattr(refresh, "_fetch_observed", fetch_actual)
    monkeypatch.setattr(refresh, "fetch_native_dashboard_snapshot", forbidden_storm)
    values, directory, audit = refresh.refresh_nuclear_reporting_sources(
        config, "ES", timezone, DAY, tmp_path / "report_only/es", client=object())
    assert requests == ["observed"]
    assert values.index.equals(expected)
    assert values.loc[current].notna().sum() == (24 if current_count == 24 else 0)
    assert audit["status"] == "complete"
    assert audit["schema_version"] == 1 and "observation_policy" not in audit
    assert audit["storm_status"] == "unavailable_zone_not_supported"
    assert "storm_dashboard" not in audit and "storm_primary_report_benchmark" not in audit
    assert not (directory / "inputs/storm_dashboard_official_statistics.parquet").exists()
    assert len(list(directory.rglob("*.parquet"))) == 1
    verified = refresh.verify_refreshed_observations(values, audit, zone="ES", timezone=timezone, delivery_day=DAY)
    assert verified["status"] == "verified" and verified["series"] == target
    from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
    assert _load_verified_snapshot(directory, zone="ES", timezone=timezone) is None
