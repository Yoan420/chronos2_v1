"""Pinned historical NL recovery never invents hours or source timestamps."""
from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import pytest
import yaml

from chronos2_hourly import nyx_annual_saturn_archive as archive


def block(day="2024-08-17", snapshot="2026-08-01T06:00:00Z", *, offset=0.):
    first = pd.Timestamp(day, tz="Europe/Paris")
    stop = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    grid = pd.date_range(first, stop, freq="h", inclusive="left").tz_convert("UTC")
    stamp = pd.Timestamp(snapshot)
    return pd.DataFrame({"value_time_utc": grid, "snapshot_time_utc": stamp,
        "revision_time_utc": stamp - pd.Timedelta(minutes=1),
        "value": np.arange(len(grid), dtype=float) + offset,
        "downloaded_at_utc": stamp + pd.Timedelta(hours=1)})


def fixture_archive(tmp_path, monkeypatch, *frames):
    path = tmp_path / "pinned.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path)
    monkeypatch.setattr(archive, "PINNED_ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


@pytest.mark.parametrize("day,hours", [("2024-08-17", 24), ("2026-03-29", 23), ("2026-10-25", 25)])
def test_one_common_snapshot_preserves_exact_physical_dst_grid(tmp_path, monkeypatch, day, hours):
    data = block(day, "2026-11-01T06:00:00Z")
    path = fixture_archive(tmp_path, monkeypatch, data)
    series, evidence = archive.recover_nl_profile(day, "2026-11-03", archive_path=path)
    assert len(series) == hours and str(series.index.tz) == "UTC" and series.index.is_unique
    np.testing.assert_array_equal(series, data.value)
    assert evidence["snapshot_time_utc"] == "2026-11-01T06:00:00+00:00"
    assert evidence["imputation"] is False and evidence["mosaic"] is False
    assert archive.verify_nl_profile(series, evidence, day, "2026-11-03", path) == evidence


def test_latest_complete_snapshot_is_selected_without_borrowing_from_partial_newer_snapshot(tmp_path, monkeypatch):
    older = block(offset=10.)
    newer = block(snapshot="2026-08-02T06:00:00Z", offset=100.)
    path = fixture_archive(tmp_path, monkeypatch, older, newer.iloc[:12])
    series, evidence = archive.recover_nl_profile("2024-08-17", "2026-09-30", archive_path=path)
    np.testing.assert_array_equal(series, older.value)
    assert evidence["snapshot_time_utc"] == "2026-08-01T06:00:00+00:00"
    # Two incomplete snapshots whose union has every hour still cannot qualify.
    path = fixture_archive(tmp_path, monkeypatch, older.iloc[:12], newer.iloc[12:])
    with pytest.raises(ValueError, match="no complete common snapshot"):
        archive.recover_nl_profile("2024-08-17", "2026-09-30", archive_path=path)


@pytest.mark.parametrize("corruption", ["late_snapshot", "late_revision", "late_download", "duplicate", "missing", "nan"])
def test_unavailable_or_incomplete_snapshot_is_never_relabelled(tmp_path, monkeypatch, corruption):
    data = block(snapshot="2026-09-29T05:00:00Z")
    if corruption == "late_snapshot":
        data["snapshot_time_utc"] = pd.Timestamp("2026-09-29T06:00:01Z")
    elif corruption == "late_revision":
        data.loc[0, "revision_time_utc"] = pd.Timestamp("2026-09-29T05:00:01Z")
    elif corruption == "late_download":
        data.loc[0, "downloaded_at_utc"] = pd.Timestamp("2026-09-29T06:00:01Z")
    elif corruption == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif corruption == "missing":
        data = data.iloc[:-1]
    elif corruption == "nan":
        data.loc[0, "value"] = np.nan
    path = fixture_archive(tmp_path, monkeypatch, data)
    with pytest.raises(ValueError, match="no complete common snapshot"):
        archive.recover_nl_profile("2024-08-17", "2026-09-30", archive_path=path)


def test_hash_and_canonical_source_contract_are_mandatory(tmp_path, monkeypatch):
    path = fixture_archive(tmp_path, monkeypatch, block())
    pin = archive.PINNED_ARCHIVE_SHA256
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        archive.recover_nl_profile("2024-08-17", "2026-09-30", archive_path=path)
    path = fixture_archive(tmp_path, monkeypatch, block())
    assert archive.PINNED_ARCHIVE_SHA256 == pin
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"zones": {"NL": {"covariates": {archive.ALIAS: {
        "series": "unapproved.series", "source": "pit_parquet", "naive_timezone": "UTC", "fill_method": "none"}}}}}))
    monkeypatch.setattr(archive, "CONFIG_PATH", config)
    with pytest.raises(ValueError, match="canonical series"):
        archive.recover_nl_profile("2024-08-17", "2026-09-30", archive_path=path)


def test_verification_recomputes_values_metadata_and_transportable_archive(tmp_path, monkeypatch):
    path = fixture_archive(tmp_path, monkeypatch, block())
    series, evidence = archive.recover_nl_profile("2024-08-17", "2026-09-30", archive_path=path)
    portable = tmp_path / "raw_day" / "nl_repository_vintages.parquet"
    portable.parent.mkdir()
    portable.write_bytes(path.read_bytes())
    assert archive.verify_nl_profile(series, evidence, "2024-08-17", "2026-09-30", portable) == evidence
    changed = series.copy()
    changed.iloc[0] += 1.
    with pytest.raises(ValueError, match="values differ"):
        archive.verify_nl_profile(changed, evidence, "2024-08-17", "2026-09-30", portable)
    for key, value in (("snapshot_time_utc", "2024-08-16T06:00:00+00:00"),
                       ("series", "unapproved.series"), ("imputation", 0)):
        changed_evidence = deepcopy(evidence)
        changed_evidence[key] = value
        with pytest.raises(ValueError, match="evidence differs"):
            archive.verify_nl_profile(series, changed_evidence, "2024-08-17", "2026-09-30", portable)
    with pytest.raises(ValueError, match="live or future"):
        archive.recover_nl_profile("2026-09-30", "2026-09-30", archive_path=portable)


def test_actual_tracked_archive_recovers_august17_but_cannot_qualify_its_old_origin():
    # This 4 MB artifact is Git-tracked; no private server or ignored research
    # dataset is needed for the regression that reproduces the reported gap.
    series, evidence = archive.recover_nl_profile("2024-08-17", "2026-09-30")
    assert len(series) == 24
    assert series.index[0] == pd.Timestamp("2024-08-16T22:00:00Z")
    assert series.index[-1] == pd.Timestamp("2024-08-17T21:00:00Z")
    assert series.min() == pytest.approx(-6.6850862)
    assert series.max() == pytest.approx(11.849862)
    assert evidence["snapshot_time_utc"] == "2026-08-05T12:17:00+00:00"
    assert evidence["downloaded_at_max_utc"] == "2026-08-10T13:04:15.101799+00:00"
    assert archive.verify_nl_profile(series, evidence, "2024-08-17", "2026-09-30") == evidence
    with pytest.raises(ValueError, match="no complete common snapshot"):
        archive.recover_nl_profile("2024-08-17", "2024-08-18")
