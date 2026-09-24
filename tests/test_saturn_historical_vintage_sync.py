from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from chronos2_modular import saturn


def _row(*, value_time: str, revision: str, value: float) -> pd.DataFrame:
    timestamp = pd.Timestamp(revision)
    return pd.DataFrame(
        {
            "value_time_utc": [pd.Timestamp(value_time)],
            "snapshot_time_utc": [timestamp],
            "revision_time_utc": [timestamp],
            "value": [value],
            "downloaded_at_utc": [timestamp + pd.Timedelta(hours=1)],
        }
    )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sync(path, *, monkeypatch, downloaded: pd.DataFrame):
    calls: list[dict[str, object]] = []

    def fake_history(_client, _series_name, **kwargs):
        calls.append(dict(kwargs))
        return {"history": "fixture"}

    monkeypatch.setattr(saturn, "fetch_saturn_history", fake_history)
    monkeypatch.setattr(
        saturn,
        "history_to_vintage_frame",
        lambda *_args, **_kwargs: downloaded.copy(),
    )
    result = saturn.sync_vintage_series(
        object(),
        zone="FR",
        alias="fr_residual_load_fcst",
        series_name="power.fr.residual.load.hourly.gw.fcst",
        path=path,
        revision_start=pd.Timestamp("2026-08-14T00:00:00Z"),
        revision_end=pd.Timestamp("2026-08-15T06:00:00Z"),
        value_start=pd.Timestamp("2026-08-14T22:00:00Z"),
        value_end=pd.Timestamp("2026-08-16T21:00:00Z"),
        timezone="Europe/Paris",
        chunk_days=90,
        retries=1,
    )
    return result, calls


def test_historical_asof_queries_requested_window_despite_newer_cache(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "vintages.parquet"
    existing = _row(
        value_time="2026-08-20T00:00:00Z",
        revision="2026-08-19T05:00:00Z",
        value=60.0,
    )
    existing.to_parquet(path, index=False)
    historical = _row(
        value_time="2026-08-16T00:00:00Z",
        revision="2026-08-15T05:00:00Z",
        value=50.0,
    )

    result, calls = _sync(
        path,
        monkeypatch=monkeypatch,
        downloaded=historical,
    )

    assert len(calls) == 1
    assert calls[0]["from_insertion_date"] == pd.Timestamp(
        "2026-08-14T00:00:00Z"
    )
    assert calls[0]["to_insertion_date"] == pd.Timestamp(
        "2026-08-15T06:00:00Z"
    )
    assert result.status == "updated"
    assert result.rows_before == 1
    assert result.rows_downloaded == 1
    assert result.rows_after == 2
    merged = saturn.read_vintage_store(path)
    assert set(merged["revision_time_utc"]) == {
        pd.Timestamp("2026-08-15T05:00:00Z"),
        pd.Timestamp("2026-08-19T05:00:00Z"),
    }


def test_empty_historical_response_is_still_queried_and_cache_is_unchanged(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "vintages.parquet"
    existing = _row(
        value_time="2026-08-20T00:00:00Z",
        revision="2026-08-19T05:00:00Z",
        value=60.0,
    )
    existing.to_parquet(path, index=False)
    before = _sha256(path)

    result, calls = _sync(
        path,
        monkeypatch=monkeypatch,
        downloaded=pd.DataFrame(columns=saturn.VINTAGE_COLUMNS),
    )

    assert len(calls) == 1
    assert result.status == "up_to_date"
    assert result.rows_downloaded == 0
    assert _sha256(path) == before


def test_historical_fetch_error_never_replaces_existing_cache(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "vintages.parquet"
    existing = _row(
        value_time="2026-08-20T00:00:00Z",
        revision="2026-08-19T05:00:00Z",
        value=60.0,
    )
    existing.to_parquet(path, index=False)
    before = _sha256(path)

    def fail(*_args, **_kwargs):
        raise ConnectionError("Saturn unavailable")

    monkeypatch.setattr(saturn, "fetch_saturn_history", fail)
    with pytest.raises(ConnectionError, match="Saturn unavailable"):
        saturn.sync_vintage_series(
            object(),
            zone="FR",
            alias="fr_residual_load_fcst",
            series_name="power.fr.residual.load.hourly.gw.fcst",
            path=path,
            revision_start=pd.Timestamp("2026-08-14T00:00:00Z"),
            revision_end=pd.Timestamp("2026-08-15T06:00:00Z"),
            value_start=pd.Timestamp("2026-08-14T22:00:00Z"),
            value_end=pd.Timestamp("2026-08-16T21:00:00Z"),
            timezone="Europe/Paris",
            retries=1,
        )
    assert _sha256(path) == before
