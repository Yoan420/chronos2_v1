from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos2_modular.common import SeriesSpec, ZoneConfig
from chronos2_modular.saturn import (
    cache_path_for_series,
    sync_latest_series,
    sync_saturn_data,
)


PARIS = "Europe/Paris"
TARGET_SERIES = "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_grid() -> pd.DatetimeIndex:
    return pd.date_range(
        "2026-08-12 00:00",
        "2026-08-19 23:00",
        freq="h",
        tz=PARIS,
    )


def _contaminated_target() -> pd.Series:
    """Mimic the real cache after a D-2-only refresh skipped 72 hours."""

    index = _target_grid()
    local_days = pd.Index(index.date)
    gap = (
        (local_days >= pd.Timestamp("2026-08-15").date())
        & (local_days <= pd.Timestamp("2026-08-17").date())
    )
    kept = index[~gap]
    return pd.Series(
        np.arange(len(kept), dtype=float),
        index=kept,
        name="target",
    )


def _write_latest(path: Path, series: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (
        series.rename("value")
        .to_frame()
        .reset_index(names="timestamp")
        .to_csv(path, index=False, compression="gzip")
    )


def _read_index(path: Path) -> pd.DatetimeIndex:
    frame = pd.read_csv(path)
    return pd.DatetimeIndex(
        pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
    )


def test_strict_target_sync_repairs_internal_72_hour_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.csv.gz"
    _write_latest(path, _contaminated_target())
    complete = pd.Series(
        np.arange(len(_target_grid()), dtype=float) + 100.0,
        index=_target_grid(),
        name="target",
    )
    fetch_calls: list[dict[str, Any]] = []

    def fetch(
        _client: Any,
        _series_name: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        _timezone: str,
        **kwargs: Any,
    ) -> pd.Series:
        fetch_calls.append(
            {"start": start, "end": end, "revision_date": kwargs["revision_date"]}
        )
        return complete.loc[(complete.index >= start) & (complete.index <= end)]

    monkeypatch.setattr(
        "chronos2_modular.saturn.fetch_saturn_series_from_client",
        fetch,
    )
    cutoff = pd.Timestamp("2026-08-19 06:00", tz="UTC")

    result = sync_latest_series(
        object(),
        zone="FR",
        alias="target",
        series_name=TARGET_SERIES,
        path=path,
        start=pd.Timestamp("2026-08-18 00:00", tz=PARIS),
        end=pd.Timestamp("2026-08-19 23:00", tz=PARIS),
        timezone=PARIS,
        sync_as_of_utc=cutoff,
        naive_timezone="UTC",
        require_contiguous_hourly=True,
    )

    assert len(fetch_calls) == 1
    # The requested D-2 lower bound must be moved behind the first missing
    # instant (2026-08-15 00:00 local), otherwise the three-day hole survives.
    assert fetch_calls[0]["start"] <= pd.Timestamp(
        "2026-08-14 23:00", tz=PARIS
    )
    assert fetch_calls[0]["revision_date"] == cutoff
    expected = _target_grid().tz_convert("UTC")
    assert _read_index(path).equals(expected)
    assert result.rows_after == len(expected)


def test_strict_target_sync_rejects_incomplete_download_without_mutating_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.csv.gz"
    existing = _contaminated_target()
    _write_latest(path, existing)
    digest_before = _sha256(path)
    tail = existing.loc[existing.index >= pd.Timestamp("2026-08-18", tz=PARIS)]

    def incomplete_fetch(*_args: Any, **_kwargs: Any) -> pd.Series:
        return tail.copy()

    monkeypatch.setattr(
        "chronos2_modular.saturn.fetch_saturn_series_from_client",
        incomplete_fetch,
    )

    with pytest.raises(
        ValueError,
        match=r"(?i)(cible|target).*(discontinu|manqu|absente)",
    ):
        sync_latest_series(
            object(),
            zone="FR",
            alias="target",
            series_name=TARGET_SERIES,
            path=path,
            start=pd.Timestamp("2026-08-18 00:00", tz=PARIS),
            end=pd.Timestamp("2026-08-19 23:00", tz=PARIS),
            timezone=PARIS,
            sync_as_of_utc=pd.Timestamp("2026-08-19 06:00", tz="UTC"),
            naive_timezone="UTC",
            require_contiguous_hourly=True,
        )

    assert _sha256(path) == digest_before
    assert _read_index(path).equals(existing.index.tz_convert("UTC"))


def test_target_gap_failure_stops_before_any_pit_download(tmp_path: Path) -> None:
    target = SeriesSpec(
        alias="target",
        series=TARGET_SERIES,
        naive_timezone="UTC",
    )
    zone = ZoneConfig(
        zone="FR",
        timezone=PARIS,
        target=target,
        covariates={
            "forecast": SeriesSpec(
                alias="forecast",
                series="power.fr.residual.load.hourly.gw.fcst",
                source="pit_parquet",
            )
        },
    )
    cache_root = tmp_path / "cache"
    target_path = cache_path_for_series(cache_root, zone.zone, target)
    existing = _contaminated_target()
    _write_latest(target_path, existing)
    tail = existing.loc[existing.index >= pd.Timestamp("2026-08-18", tz=PARIS)]

    class IncompleteClient:
        def __init__(self) -> None:
            self.get_calls = 0
            self.history_calls = 0

        def get(self, _name: str, *_args: Any, **_kwargs: Any) -> pd.Series:
            self.get_calls += 1
            return tail.copy()

        def history(self, _name: str, **_kwargs: Any) -> dict[Any, Any]:
            self.history_calls += 1
            raise AssertionError("PIT must not run after a strict target failure")

    client = IncompleteClient()
    config = {
        "data": {
            "project_root": str(tmp_path),
            "source": "auto",
            "cache_dir": "cache",
            "pit_vintage_dir": str(tmp_path / "pit"),
            "pit_files": {"forecast": "forecast.parquet"},
            "start": "2026-08-18T00:00:00+02:00",
            "end": "2026-08-19T23:00:00+02:00",
            "frequency": "h",
            "saturn_sync": {"retries": 1},
        }
    }

    with pytest.raises(
        ValueError,
        match=r"(?i)(cible|target).*(discontinu|manqu|absente)",
    ):
        sync_saturn_data(
            [zone],
            config,
            tmp_path,
            as_of="2026-08-19T06:00:00Z",
            client=client,
        )

    assert client.get_calls >= 1
    assert client.history_calls == 0
    assert not (tmp_path / "pit" / "forecast.parquet").exists()
