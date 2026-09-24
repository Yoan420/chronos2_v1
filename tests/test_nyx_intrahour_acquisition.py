from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyx_intrahour.saturn_sources import ReadOnlySession, SOURCE, day_grid, fetch_day


class FakeClient:
    def __init__(self, raw):
        self.raw, self.calls = raw, []

    def get(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return self.raw


@pytest.mark.parametrize("day,count,cutoff", [
    ("2025-10-26", 100, "2025-10-25T06:00:00Z"),
    ("2026-03-29", 92, "2026-03-28T07:00:00Z"),
    ("2026-03-30", 96, "2026-03-29T06:00:00Z"),
    ("2026-01-15", 96, "2026-01-14T07:00:00Z"),
])
def test_native_physical_quarters_and_previous_civil_cutoff(day, count, cutoff):
    grid, _ = day_grid(day)
    client = FakeClient(pd.Series(np.arange(count, dtype=float), index=grid))
    data, audit = fetch_day(client, day)
    assert audit["status"] == "complete"
    assert len(data) == audit["expected_quarters"] == count
    name, args = client.calls[0]
    assert name == "23259"
    assert args == {"revision_date": pd.Timestamp(cutoff), "from_value_date": grid[0],
                    "to_value_date": grid[-1], "_keep_nans": True}
    assert data.value_time_utc.tolist() == grid.tolist()
    assert data.snapshot_time_utc.eq(pd.Timestamp(cutoff)).all()
    assert data.revision_time_utc.equals(data.snapshot_time_utc)
    assert audit["provider_revision_timestamp_available"] is False
    assert audit["temporal_evidence"] == "retrospective_asof"


def test_missing_and_nan_quarters_are_never_filled():
    grid, _ = day_grid("2026-07-15")
    raw = pd.Series(np.arange(96, dtype=float), index=grid).drop(grid[3])
    raw.loc[grid[7]] = np.nan
    frame, audit = fetch_day(FakeClient(raw), "2026-07-15")
    assert audit["status"] == "incomplete"
    assert audit["missing_quarters"] == audit["nonfinite_quarters"] == 1
    assert len(frame) == 95 and frame.value.isna().sum() == 1
    assert grid[3] not in frame.value_time_utc.tolist()


@pytest.mark.parametrize("kind", ["naive", "duplicate", "hourly", "offgrid"])
def test_no_repair_for_ambiguous_or_non_native_source(kind):
    grid, _ = day_grid("2026-07-15")
    index = {"naive": grid.tz_localize(None), "duplicate": grid.append(grid[:1]),
             "hourly": grid[::4], "offgrid": grid+pd.Timedelta(minutes=1)}[kind]
    data, audit = fetch_day(FakeClient(pd.Series(np.arange(len(index)), index=index)), "2026-07-15")
    assert data.empty and audit["status"] == "error"


def test_empty_and_exception_do_not_fabricate_data_or_leak_errors():
    frame, audit = fetch_day(FakeClient(None), "2026-07-15")
    assert frame.empty and audit["status"] == "empty"

    class Broken:
        def get(self, *args, **kwargs):
            raise RuntimeError("password=super-secret proxy=https://user:secret@host")

    frame, audit = fetch_day(Broken(), "2026-07-15")
    assert frame.empty and audit["error_type"] == "RuntimeError"
    assert "secret" not in json.dumps(audit) and "password" not in json.dumps(audit)


def test_http_mutations_are_forbidden_before_request():
    with ReadOnlySession() as session:
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            with pytest.raises(ValueError, match="GET only"):
                session.request(method, "https://invalid.test")


def test_materializer_isolated_sealed_manifest_and_idempotent_resume(tmp_path, monkeypatch):
    import materialize_nyx_intrahour as module
    calls = []

    class Client:
        session = type("Session", (), {"close": lambda self: None})()

        def get(self, name, **kwargs):
            calls.append(kwargs)
            index = pd.date_range(kwargs["from_value_date"], kwargs["to_value_date"], freq="15min")
            return pd.Series(np.arange(len(index), dtype=float), index=index)

    monkeypatch.setattr(module, "ARCHIVE_ROOT", tmp_path)
    monkeypatch.setattr(module, "make_client", Client)
    destination = tmp_path/"archive"
    first = module.collect("2026-01-15", "2026-01-15", destination, workers=1)
    assert first["summary"]["complete_days"] == 1
    assert first["sources"] == [SOURCE]
    assert first["provider_revision_timestamp_available"] is False
    assert module.sha(destination/first["data_file"]) == first["data_sha256"]
    second = module.collect("2026-01-15", "2026-01-15", destination, workers=1, resume=True)
    assert len(calls) == 1 and second["data_sha256"] == first["data_sha256"]
    with pytest.raises(ValueError, match="Resume identity"):
        module.collect("2026-01-16", "2026-01-16", destination, workers=1, resume=True)
    with pytest.raises(ValueError, match="dedicated child"):
        module.collect("2026-01-15", "2026-01-15", tmp_path.parent/"outside", workers=1)
    with pytest.raises(ValueError, match="one or two"):
        module.collect("2026-01-15", "2026-01-15", tmp_path/"three", workers=3)


@pytest.mark.parametrize("redirect_namespace", [True, False])
def test_redirected_namespace_or_child_fails_before_writing(tmp_path, monkeypatch, redirect_namespace):
    import materialize_nyx_intrahour as module
    namespace = tmp_path / "research"
    destination = namespace / "archive"
    outside = tmp_path / "operational"
    original_resolve = Path.resolve
    redirected = namespace if redirect_namespace else destination

    def resolve(path, *args, **kwargs):
        return outside if path == redirected else original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(module, "ARCHIVE_ROOT", namespace)
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="symlink or junction"):
        module.collect("2026-01-15", "2026-01-15", destination, workers=1)
    assert not namespace.exists() and not outside.exists()


@pytest.mark.parametrize("child", ["days", "native_forecasts.parquet", "manifest.json.tmp", "collection.lock"])
def test_redirected_archive_descendants_are_refused(tmp_path, monkeypatch, child):
    import materialize_nyx_intrahour as module
    namespace = tmp_path/"research"
    archive = namespace/"archive"
    redirected = archive/child
    outside = tmp_path/"production"
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        return outside if path == redirected else original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(module, "ARCHIVE_ROOT", namespace)
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="symlink or junction"):
        module.safe_target(redirected)
    assert not namespace.exists() and not outside.exists()
