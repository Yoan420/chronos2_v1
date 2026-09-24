from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nyx_quarterhour.sources as source


class FakeClient:
    def __init__(self, raw):
        self.raw, self.calls = raw, []

    def get(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return self.raw


@pytest.mark.parametrize("day,count", [("2025-10-26",100), ("2026-03-29",92), ("2026-01-15",96)])
def test_native_prices_keep_dst_and_negative_prices_without_asof_claim(day, count):
    grid = source.quarter_grid(day, day)
    client = FakeClient(pd.Series(np.arange(count, dtype=float)-10, index=grid))
    frame, audit = source.fetch_range(client, "BE", day, day)
    assert len(frame) == count and audit["status"] == "complete"
    assert frame.actual_15m.min() == -10
    assert pd.DatetimeIndex(frame.timestamp_utc).equals(grid)
    name, args = client.calls[0]
    assert name == "60451" and "revision_date" not in args
    assert args == {"from_value_date": grid[0], "to_value_date": grid[-1], "_keep_nans": True}
    assert audit["price_vintage"] == "latest_observations"
    assert audit["provider_revision_timestamp_available"] is False
    hourly = source.hourly_means(frame)
    assert len(hourly) == count//4
    assert hourly.actual_hourly_from_15m.iloc[0] == -8.5


def test_missing_native_quarters_remain_missing_and_invalidate_hour():
    grid = source.quarter_grid("2026-01-15", "2026-01-15")
    raw = pd.Series(np.arange(96,dtype=float),index=grid).drop(grid[1])
    raw.loc[grid[6]] = np.nan
    frame, audit = source.fetch_range(FakeClient(raw), "FR", "2026-01-15", "2026-01-15")
    assert audit["status"] == "partial" and len(frame) == 94
    assert audit["missing_quarters"] == 2 and audit["nonfinite_quarters_excluded"] == 1
    assert len(source.hourly_means(frame)) == 22


@pytest.mark.parametrize("kind", ["naive", "duplicate", "hourly", "offgrid"])
def test_bad_native_clock_is_not_repaired(kind):
    grid = source.quarter_grid("2026-01-15", "2026-01-15")
    index = {"naive":grid.tz_localize(None), "duplicate":grid.append(grid[:1]),
             "hourly":grid[::4], "offgrid":grid+pd.Timedelta(seconds=1)}[kind]
    frame, audit = source.fetch_range(FakeClient(pd.Series(np.arange(len(index)),index=index)), "NL", "2026-01-15", "2026-01-15")
    assert frame.empty and audit["status"] == "error"


def test_failed_request_never_persists_sensitive_error_text():
    class Broken:
        def get(self, *args, **kwargs):
            raise RuntimeError("password=private-secret at https://user:secret@proxy")
    frame, audit = source.fetch_range(Broken(), "DE", "2026-01-15", "2026-01-15")
    assert frame.empty and audit["error_type"] == "RuntimeError"
    assert "secret" not in json.dumps(audit) and "password" not in json.dumps(audit)


def test_http_session_forbids_mutations():
    with source.ReadOnlySession() as session:
        with pytest.raises(ValueError, match="GET"):
            session.request("POST", "https://invalid.test")


@pytest.fixture
def archive(tmp_path, monkeypatch):
    import materialize_nyx_quarterhour as materializer
    monkeypatch.setattr(source,"ARCHIVE_ROOT",tmp_path)

    class Client:
        session = type("Session", (), {"close":lambda self: None})()
        def get(self, name, **kwargs):
            grid = pd.date_range(kwargs["from_value_date"],kwargs["to_value_date"],freq="15min")
            return pd.Series(np.arange(len(grid),dtype=float),index=grid)

    monkeypatch.setattr(materializer,"make_client",Client)
    directory = tmp_path/"archive"
    materializer.collect("2026-01-15", "2026-01-15",directory,workers=1)
    return directory


def test_sealed_archive_contract_and_no_overwrite(archive):
    import materialize_nyx_quarterhour as materializer
    frame, audit = source.read_native_prices(archive/"manifest.json")
    assert len(frame) == 4*96 and audit["status"] == "complete"
    assert audit["production_pit_evidence"] is False
    assert source.digest(archive/"native_prices.parquet") == audit["data_sha256"]
    with pytest.raises(ValueError, match="never overwritten"):
        materializer.collect("2026-01-15", "2026-01-15", archive, workers=1)


@pytest.mark.parametrize("field,value", [("data_file","../other.parquet"), ("unit","MW"),
    ("resolution_minutes",60), ("price_vintage","historical_issued"),
    ("interpolation","ffill"), ("provider_revision_timestamp_available",True), ("data_sha256","bad")])
def test_reader_rejects_manifest_contract_forgery(archive, field, value):
    path = archive/"manifest.json"
    doc = json.loads(path.read_text())
    doc[field] = value
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        source.read_native_prices(path)


def test_reader_rejects_missing_quarter_disguised_as_complete(archive):
    path = archive/"manifest.json"
    doc = json.loads(path.read_text())
    data = archive/doc["data_file"]
    frame = pd.read_parquet(data).iloc[1:]
    frame.to_parquet(data,index=False)
    doc["data_sha256"] = source.digest(data)
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="every physical quarter"):
        source.read_native_prices(path)


@pytest.mark.parametrize("child", ["", "archive", "archive/native_prices.parquet", "archive/manifest.json.tmp", "archive/collection.lock"])
def test_namespace_and_descendant_redirects_are_refused(tmp_path,monkeypatch,child):
    namespace = tmp_path/"research"
    target = namespace/child
    original = Path.resolve
    monkeypatch.setattr(source,"ARCHIVE_ROOT",namespace)
    monkeypatch.setattr(Path,"resolve",lambda p,*a,**k: tmp_path/"operational" if p==target else original(p,*a,**k))
    with pytest.raises(ValueError,match="symlink or junction"):
        source.safe_output(target)
    assert not namespace.exists()
