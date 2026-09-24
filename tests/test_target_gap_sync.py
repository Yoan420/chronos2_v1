from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos2_modular.common import SeriesSpec, ZoneConfig
from chronos2_modular.saturn import (
    AUDITED_EQUIVALENT_TARGET_FALLBACKS,
    cache_path_for_series,
    sync_latest_series,
    sync_saturn_data,
)


PARIS = "Europe/Paris"
TARGET_SERIES = "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh"
FALLBACK_SERIES = "power.price.da.fr.bzn.hourly.entsoe.eurmwh"


def test_audited_target_completion_sources_are_zone_specific_and_exclude_es() -> None:
    expected = {
        "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh": (
            "power.price.da.fr.bzn.hourly.entsoe.eurmwh",
            "Europe/Paris",
            3,
            True,
        ),
        "power.price.da.de_lu.bzn.hourly.entsoe.utc.cdh.eurmwh": (
            "power.price.da.de_lu.bzn.hourly.entsoe.eurmwh",
            "Europe/Berlin",
            3,
            True,
        ),
        "power.price.da.be.bzn.hourly.entsoe.utc.cdh.eurmwh": (
            "power.price.da.be.bzn.hourly.entsoe.eurmwh",
            "Europe/Brussels",
            3,
            True,
        ),
        "power.price.da.nl.bzn.hourly.entsoe.utc.cdh.eurmwh": (
            "power.price.da.nl.bzn.hourly.entsoe.eurmwh",
            "Europe/Amsterdam",
            3,
            True,
        ),
    }
    assert {
        target: (
            str(config["series"]),
            str(config["naive_timezone"]),
            int(config["request_padding_hours"]),
            bool(config["nocache"]),
        )
        for target, config in AUDITED_EQUIVALENT_TARGET_FALLBACKS.items()
    } == expected
    assert not any(".es." in target for target in expected)


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


def test_historical_replay_never_builds_an_inverted_target_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.csv.gz"
    existing_index = pd.date_range(
        "2026-08-01 00:00",
        "2026-08-24 23:00",
        freq="h",
        tz=PARIS,
    )
    existing = pd.Series(
        np.arange(len(existing_index), dtype=float),
        index=existing_index,
        name="target",
    )
    _write_latest(path, existing)
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch(
        _client: Any,
        _series_name: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        _timezone: str,
        **_kwargs: Any,
    ) -> pd.Series:
        calls.append((start, end))
        return existing.loc[(existing.index >= start) & (existing.index <= end)]

    monkeypatch.setattr(
        "chronos2_modular.saturn.fetch_saturn_series_from_client",
        fetch,
    )

    result = sync_latest_series(
        object(),
        zone="BE",
        alias="target",
        series_name="power.price.da.be.bzn.hourly.entsoe.utc.cdh.eurmwh",
        path=path,
        start=pd.Timestamp("2022-08-15 08:00", tz=PARIS),
        end=pd.Timestamp("2026-08-15 23:00", tz=PARIS),
        timezone=PARIS,
        sync_as_of_utc=pd.Timestamp("2026-08-15 06:00", tz="UTC"),
        require_contiguous_hourly=True,
    )

    assert len(calls) == 1
    assert calls[0][0] <= calls[0][1]
    assert calls[0][1] == pd.Timestamp("2026-08-15 23:00", tz=PARIS)
    assert result.rows_after == len(existing)


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


def test_strict_target_sync_fills_only_an_equivalent_missing_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.csv.gz"
    complete_index = pd.date_range(
        "2026-08-23 00:00",
        "2026-08-31 23:00",
        freq="h",
        tz=PARIS,
    )
    complete = pd.Series(
        np.arange(len(complete_index), dtype=float),
        index=complete_index,
        name="target",
    )
    canonical = complete.loc[complete.index < pd.Timestamp("2026-08-31", tz=PARIS)]
    _write_latest(path, canonical)
    calls: list[tuple[str, pd.Timestamp]] = []

    def fetch(
        _client: Any,
        series_name: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        _timezone: str,
        **kwargs: Any,
    ) -> pd.Series:
        calls.append((series_name, kwargs["revision_date"]))
        source = complete if series_name == FALLBACK_SERIES else canonical
        return source.loc[(source.index >= start) & (source.index <= end)]

    monkeypatch.setattr(
        "chronos2_modular.saturn.fetch_saturn_series_from_client",
        fetch,
    )
    cutoff = pd.Timestamp("2026-08-31 06:00", tz="UTC")
    result = sync_latest_series(
        object(),
        zone="FR",
        alias="target",
        series_name=TARGET_SERIES,
        path=path,
        start=pd.Timestamp("2026-08-23 00:00", tz=PARIS),
        end=pd.Timestamp("2026-08-31 23:00", tz=PARIS),
        timezone=PARIS,
        sync_as_of_utc=cutoff,
        naive_timezone="UTC",
        require_contiguous_hourly=True,
    )

    assert [item[0] for item in calls] == [TARGET_SERIES, FALLBACK_SERIES]
    assert all(item[1] == cutoff for item in calls)
    assert _read_index(path).equals(complete_index.tz_convert("UTC"))
    assert result.equivalent_fallback_series == FALLBACK_SERIES
    assert result.equivalent_fallback_rows == 24
    assert result.equivalent_fallback_validation_paired_hours == 168
    assert result.equivalent_fallback_validation_max_abs_difference_eur_mwh == 0.0
    assert result.equivalent_fallback_validation_tolerance_eur_mwh == 1e-9


def test_empty_canonical_response_reaches_exact_audited_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "target.csv.gz"
    complete_index = pd.date_range(
        "2026-08-23 00:00",
        "2026-08-31 23:00",
        freq="h",
        tz=PARIS,
    )
    complete = pd.Series(
        np.arange(len(complete_index), dtype=float),
        index=complete_index,
        name="target",
    )
    canonical_cache = complete.loc[
        complete.index < pd.Timestamp("2026-08-31", tz=PARIS)
    ]
    _write_latest(path, canonical_cache)

    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def get(self, series_name: str, **kwargs: Any) -> pd.Series:
            self.calls.append((series_name, dict(kwargs)))
            if series_name == TARGET_SERIES:
                # The canonical Saturn dialect is valid, but the auction day
                # has not propagated to this formula yet.
                return pd.Series(dtype=float)
            assert series_name == FALLBACK_SERIES
            query_start = pd.Timestamp(kwargs["from_value_date"])
            query_end = pd.Timestamp(kwargs["to_value_date"])
            return complete.loc[
                (complete.index >= query_start)
                & (complete.index <= query_end)
            ]

    client = Client()
    cutoff = pd.Timestamp("2026-08-31 06:00", tz="UTC")
    result = sync_latest_series(
        client,
        zone="FR",
        alias="target",
        series_name=TARGET_SERIES,
        path=path,
        start=pd.Timestamp("2026-08-29 00:00", tz=PARIS),
        end=pd.Timestamp("2026-08-31 23:00", tz=PARIS),
        timezone=PARIS,
        sync_as_of_utc=cutoff,
        naive_timezone="UTC",
        require_contiguous_hourly=True,
    )

    assert [item[0] for item in client.calls] == [
        TARGET_SERIES,
        TARGET_SERIES,
        FALLBACK_SERIES,
    ]
    assert client.calls[1][1]["nocache"] is True
    assert all("from_value_date" in item[1] for item in client.calls)
    assert all(item[1]["revision_date"] == cutoff for item in client.calls)
    assert client.calls[2][1]["from_value_date"].tzinfo is not None
    assert client.calls[2][1]["nocache"] is True
    assert client.calls[2][1]["to_value_date"] == pd.Timestamp(
        "2026-09-01 02:00",
        tz=PARIS,
    )
    assert _read_index(path).equals(complete_index.tz_convert("UTC"))
    assert result.rows_downloaded == 0
    assert result.equivalent_fallback_series == FALLBACK_SERIES
    assert result.equivalent_fallback_rows == 24
    assert result.equivalent_fallback_validation_paired_hours == 168
    assert result.equivalent_fallback_validation_max_abs_difference_eur_mwh == 0.0


def test_current_canonical_download_replaces_a_revised_fallback_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.csv.gz"
    complete_index = pd.date_range(
        "2026-08-23 00:00",
        "2026-09-01 23:00",
        freq="h",
        tz=PARIS,
    )
    current_fallback = pd.Series(
        np.arange(len(complete_index), dtype=float),
        index=complete_index,
        name="target",
    )
    canonical_end = pd.Timestamp("2026-08-30 23:00", tz=PARIS)
    current_canonical = current_fallback.loc[
        current_fallback.index <= canonical_end
    ]

    # The prior cutoff had already filled 31 August from the official
    # completion formula, whose value was then revised at the new cutoff.  This
    # stale suffix must not participate in the new compatibility check.
    prior_cache = current_fallback.loc[
        current_fallback.index <= pd.Timestamp("2026-08-31 23:00", tz=PARIS)
    ].copy()
    stale_hour = pd.Timestamp("2026-08-31 12:00", tz=PARIS)
    prior_cache.loc[stale_hour] = 184.66
    current_fallback.loc[stale_hour] = 162.4375
    _write_latest(path, prior_cache)

    calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def fetch(
        _client: Any,
        series_name: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        _timezone: str,
        **_kwargs: Any,
    ) -> pd.Series:
        calls.append((series_name, start, end))
        source = (
            current_fallback
            if series_name == FALLBACK_SERIES
            else current_canonical
        )
        return source.loc[(source.index >= start) & (source.index <= end)]

    monkeypatch.setattr(
        "chronos2_modular.saturn.fetch_saturn_series_from_client",
        fetch,
    )
    result = sync_latest_series(
        object(),
        zone="FR",
        alias="target",
        series_name=TARGET_SERIES,
        path=path,
        # Reproduce the operational D-2 sync-only bound.  The strict target
        # path must still fetch enough earlier canonical hours for its 168-hour
        # guard instead of validating against the stale cached suffix.
        start=pd.Timestamp("2026-08-31 00:00", tz=PARIS),
        end=pd.Timestamp("2026-09-01 23:00", tz=PARIS),
        timezone=PARIS,
        sync_as_of_utc=pd.Timestamp("2026-09-01 06:00", tz="UTC"),
        naive_timezone="UTC",
        require_contiguous_hourly=True,
    )

    refreshed = pd.read_csv(path)
    refreshed.index = pd.DatetimeIndex(
        pd.to_datetime(refreshed.pop("timestamp"), utc=True)
    ).tz_convert(PARIS)
    refreshed.index.name = None
    refreshed_values = refreshed["value"].astype(float)
    pd.testing.assert_series_equal(
        refreshed_values.loc[refreshed_values.index <= canonical_end],
        current_canonical.rename("value"),
        check_freq=False,
    )
    pd.testing.assert_series_equal(
        refreshed_values.loc[refreshed_values.index > canonical_end],
        current_fallback.loc[
            (current_fallback.index > canonical_end)
            & (
                current_fallback.index
                <= pd.Timestamp("2026-09-01 23:00", tz=PARIS)
            )
        ].rename("value"),
        check_freq=False,
    )
    assert refreshed_values.loc[stale_hour] == 162.4375
    assert result.equivalent_fallback_rows == 48
    assert result.equivalent_fallback_validation_paired_hours == 168
    assert result.equivalent_fallback_validation_max_abs_difference_eur_mwh == 0.0
    canonical_call = next(item for item in calls if item[0] == TARGET_SERIES)
    assert canonical_call[1] <= (
        canonical_end - pd.Timedelta(hours=167)
    )


@pytest.mark.parametrize(
    ("alias", "series_name", "strict", "with_cache"),
    [
        pytest.param("target", TARGET_SERIES, True, False, id="no-cache"),
        pytest.param("target", TARGET_SERIES, False, True, id="not-strict"),
        pytest.param(
            "target",
            "power.price.da.es.bzn.hourly.entsoe.utc.cdh.eurmwh",
            True,
            True,
            id="no-audited-fallback",
        ),
        pytest.param("price", TARGET_SERIES, True, True, id="not-target"),
    ],
)
def test_empty_canonical_response_is_blocked_outside_audited_strict_path(
    tmp_path: Path,
    alias: str,
    series_name: str,
    strict: bool,
    with_cache: bool,
) -> None:
    path = tmp_path / "target.csv.gz"
    existing_index = pd.date_range(
        "2026-08-23 00:00",
        "2026-08-30 23:00",
        freq="h",
        tz=PARIS,
    )
    existing = pd.Series(
        np.arange(len(existing_index), dtype=float),
        index=existing_index,
        name=alias,
    )
    digest_before: str | None = None
    if with_cache:
        _write_latest(path, existing)
        digest_before = _sha256(path)

    class EmptyClient:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, _series_name: str, **_kwargs: Any) -> pd.Series:
            self.calls += 1
            return pd.Series(dtype=float)

    client = EmptyClient()
    with pytest.raises(
        RuntimeError,
        match=r"Saturn indisponible ou vide.*plage=.*cutoff=.*reponse vide",
    ):
        sync_latest_series(
            client,
            zone="FR",
            alias=alias,
            series_name=series_name,
            path=path,
            start=pd.Timestamp("2026-08-29 00:00", tz=PARIS),
            end=pd.Timestamp("2026-08-31 23:00", tz=PARIS),
            timezone=PARIS,
            sync_as_of_utc=pd.Timestamp("2026-08-31 06:00", tz="UTC"),
            naive_timezone="UTC",
            require_contiguous_hourly=strict,
        )

    assert client.calls == 2
    if with_cache:
        assert digest_before is not None
        assert _sha256(path) == digest_before
    else:
        assert not path.exists()


def test_strict_target_sync_rejects_a_divergent_fallback_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.csv.gz"
    complete_index = pd.date_range(
        "2026-08-23 00:00",
        "2026-08-31 23:00",
        freq="h",
        tz=PARIS,
    )
    complete = pd.Series(
        np.arange(len(complete_index), dtype=float),
        index=complete_index,
        name="target",
    )
    canonical = complete.loc[complete.index < pd.Timestamp("2026-08-31", tz=PARIS)]
    divergent = complete.copy()
    divergent.loc[pd.Timestamp("2026-08-30 12:00", tz=PARIS)] += 0.01
    prior_cache = complete.copy()
    prior_cache.loc[pd.Timestamp("2026-08-31 12:00", tz=PARIS)] += 22.2225
    _write_latest(path, prior_cache)
    digest_before = _sha256(path)

    def fetch(
        _client: Any,
        series_name: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        _timezone: str,
        **_kwargs: Any,
    ) -> pd.Series:
        source = divergent if series_name == FALLBACK_SERIES else canonical
        return source.loc[(source.index >= start) & (source.index <= end)]

    monkeypatch.setattr(
        "chronos2_modular.saturn.fetch_saturn_series_from_client",
        fetch,
    )

    with pytest.raises(ValueError, match=r"fallback.*non equivalent"):
        sync_latest_series(
            object(),
            zone="FR",
            alias="target",
            series_name=TARGET_SERIES,
            path=path,
            start=pd.Timestamp("2026-08-23 00:00", tz=PARIS),
            end=pd.Timestamp("2026-08-31 23:00", tz=PARIS),
            timezone=PARIS,
            sync_as_of_utc=pd.Timestamp("2026-08-31 06:00", tz="UTC"),
            naive_timezone="UTC",
            require_contiguous_hourly=True,
        )

    assert _sha256(path) == digest_before
    assert _read_index(path).equals(prior_cache.index.tz_convert("UTC"))


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
