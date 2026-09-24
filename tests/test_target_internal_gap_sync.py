from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos2_modular.saturn import sync_latest_series


ZONE = "Europe/Berlin"
TARGET = "power.price.da.de_lu.bzn.hourly.entsoe.utc.cdh.eurmwh"
FALLBACK = "power.price.da.de_lu.bzn.hourly.entsoe.eurmwh"
CUTOFF = pd.Timestamp("2026-09-10 06:00Z")
END = pd.Timestamp("2026-09-10 23:00", tz=ZONE)


def _series() -> pd.Series:
    index = pd.date_range("2026-08-29", "2026-09-11 23:00", freq="h", tz=ZONE)
    return pd.Series(np.arange(len(index), dtype=float) / 8, index=index, name="target")


def _write(path: Path, values: pd.Series) -> None:
    values.rename("value").to_frame().reset_index(names="timestamp").to_csv(
        path, index=False, compression="gzip"
    )


def _read(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    return pd.Series(
        frame["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)).tz_convert(ZONE),
        name="target",
    )


def _canonical(complete: pd.Series) -> pd.Series:
    # Exact shape of the failed 2026-09-11 DE launch: 8 September absent,
    # 9 September present, 10 September only available from the official formula.
    return complete.loc[
        (complete.index.date != pd.Timestamp("2026-09-08").date())
        & (complete.index < pd.Timestamp("2026-09-10", tz=ZONE))
    ]


def _install_fetch(monkeypatch: pytest.MonkeyPatch, canonical: pd.Series, fallback: pd.Series):
    calls: list[dict[str, Any]] = []

    def fetch(_client, series_name, start, end, timezone, **kwargs):
        calls.append({"series": series_name, "start": start, "end": end, **kwargs})
        assert timezone == ZONE
        assert kwargs["revision_date"] == CUTOFF
        source = canonical if series_name == TARGET else fallback
        return source.loc[(source.index >= start) & (source.index <= end)].copy()

    monkeypatch.setattr("chronos2_modular.saturn.fetch_saturn_series_from_client", fetch)
    return calls


def _sync(path: Path, **kwargs):
    return sync_latest_series(
        object(), zone="DE", alias="target", series_name=TARGET, path=path,
        start=kwargs.pop("start", pd.Timestamp("2026-09-09", tz=ZONE)), end=END, timezone=ZONE,
        sync_as_of_utc=CUTOFF, naive_timezone="UTC", require_contiguous_hourly=True,
        **kwargs,
    )


def test_internal_day_and_suffix_refresh_at_same_cutoff_with_exact_audit(tmp_path, monkeypatch):
    complete = _series()
    canonical = _canonical(complete)
    fallback = complete.copy()
    # Preserve authoritative values even if the equivalent differs within the
    # allowed numerical tolerance on a canonical island between the gap/tail.
    canonical_hour = pd.Timestamp("2026-09-09 12:00", tz=ZONE)
    fallback.loc[canonical_hour] += 5e-10
    existing = complete.copy()
    existing.loc[existing.index.date == pd.Timestamp("2026-09-08").date()] = -900.0
    existing.loc[existing.index.date == pd.Timestamp("2026-09-10").date()] = -800.0
    existing.loc[existing.index > END] = 777.0
    path = tmp_path / "target.csv.gz"
    _write(path, existing)
    calls = _install_fetch(monkeypatch, canonical, fallback)

    result = _sync(path)

    refreshed = _read(path)
    expected = complete.copy()
    expected.loc[expected.index > END] = 777.0
    pd.testing.assert_series_equal(refreshed, expected, check_names=False, check_freq=False)
    assert refreshed.loc[canonical_hour] == canonical.loc[canonical_hour]
    assert result.equivalent_fallback_rows == 48
    assert result.equivalent_fallback_internal_rows == 24
    assert result.equivalent_fallback_validation_paired_hours == 192
    assert result.equivalent_fallback_validation_max_abs_difference_eur_mwh < 1e-9
    repaired = pd.DatetimeIndex(result.equivalent_fallback_value_times_utc)
    expected_repaired = complete.index[
        complete.index.date == pd.Timestamp("2026-09-08").date()
    ].union(complete.index[complete.index.date == pd.Timestamp("2026-09-10").date()])
    assert repaired.equals(expected_repaired.tz_convert("UTC"))
    # The moving nine-day overlap needs 23 additional canonical hours to prove
    # the full 168 h before this now older hole; only that prefix is refetched.
    assert len(calls) == 3
    assert calls[1]["series"] == TARGET and calls[1]["nocache"] is True
    assert calls[1]["start"] == pd.Timestamp("2026-09-01", tz=ZONE)
    assert calls[1]["end"] == pd.Timestamp("2026-09-01 22:00", tz=ZONE)
    assert calls[2]["series"] == FALLBACK and calls[2]["nocache"] is True
    assert calls[2]["incomplete_dst_policy"] == "raise"


@pytest.mark.parametrize("defect", ["divergent-before", "divergent-after", "missing-gap", "nan-gap", "inf-gap", "missing-prefix"])
def test_unverified_internal_repair_leaves_original_cache_byte_identical(tmp_path, monkeypatch, defect):
    complete = _series()
    canonical = _canonical(complete)
    fallback = complete.copy()
    if defect.startswith("divergent"):
        when = "2026-09-07 12:00" if defect.endswith("before") else "2026-09-09 12:00"
        fallback.loc[pd.Timestamp(when, tz=ZONE)] += 0.01
    elif defect == "missing-gap":
        fallback = fallback.drop(pd.Timestamp("2026-09-08 12:00", tz=ZONE))
    elif defect == "missing-prefix":
        canonical = canonical.loc[canonical.index >= pd.Timestamp("2026-09-02", tz=ZONE)]
    else:
        fallback.loc[pd.Timestamp("2026-09-08 12:00", tz=ZONE)] = np.nan if defect == "nan-gap" else np.inf
    path = tmp_path / "target.csv.gz"
    _write(path, complete)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    _install_fetch(monkeypatch, canonical, fallback)

    with pytest.raises(ValueError, match="(non equivalent|incomplet|insuffisant)"):
        _sync(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_internal_hole_without_admitted_source_still_fails_closed(tmp_path, monkeypatch):
    complete = _series()
    path = tmp_path / "target.csv.gz"
    _write(path, complete)
    before = path.read_bytes()
    calls = _install_fetch(monkeypatch, _canonical(complete), complete)
    with pytest.raises(ValueError, match="canonique courant discontinu"):
        _sync(path, equivalent_target_fallback={}, start=pd.Timestamp("2026-09-01", tz=ZONE))
    assert path.read_bytes() == before
    assert all(call["series"] == TARGET for call in calls)


def test_nonfinite_canonical_internal_values_are_repaired_as_missing(tmp_path, monkeypatch):
    complete = _series()
    canonical = complete.loc[complete.index <= END].copy()
    missing_hour = pd.Timestamp("2026-09-08 12:00", tz=ZONE)
    canonical.loc[missing_hour] = np.inf
    path = tmp_path / "target.csv.gz"
    _write(path, complete)
    _install_fetch(monkeypatch, canonical, complete)

    result = _sync(path)

    assert _read(path).loc[missing_hour] == complete.loc[missing_hour]
    assert result.equivalent_fallback_internal_rows == 1
    assert result.equivalent_fallback_rows == 1
