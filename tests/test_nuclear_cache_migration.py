from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from migrate_nuclear_daily_cache import canonical_raw, publish_epoch


def _legacy_result() -> SimpleNamespace:
    index = pd.date_range("2026-09-07T22:00Z", periods=3, freq="h", name="delivery_start_utc")
    values = np.asarray([91.23456, -10.123456, 17.777777], dtype="float32")
    stats = pd.DataFrame(
        {"chronos2__q10": values - np.float32(5),
         "chronos2__q50": values,
         "chronos2__q90": values + np.float32(5),
         "actual": values + np.float32(1),
         "forecast_origin_utc": pd.Timestamp("2026-09-07T06:00Z")},
        index=index,
    )
    raw = stats.rename(columns={f"chronos2__{q}": q for q in ("q10", "q50", "q90")})
    # Legacy FR refreshed its raw frame from a decimal CSV checkpoint, while
    # the archived corrector retained the neural outputs as exact float32.
    checkpoint = pd.read_csv(StringIO(raw.to_csv()), index_col="delivery_start_utc")
    checkpoint.index = pd.to_datetime(checkpoint.index, utc=True)
    checkpoint["forecast_origin_utc"] = pd.to_datetime(checkpoint.forecast_origin_utc, utc=True)
    future = stats.drop(columns="actual").copy()
    future.index = index + pd.Timedelta(days=1)
    return SimpleNamespace(raw_history=checkpoint,
                           residual_statistics=stats.reset_index(),
                           source_forecast=future.reset_index())


def test_canonical_raw_recovers_exact_float32_from_decimal_checkpoint() -> None:
    result = _legacy_result()
    saved_checkpoint = result.raw_history.copy(deep=True)
    saved_stats = result.residual_statistics.copy(deep=True)
    saved_future = result.source_forecast.copy(deep=True)
    expected = saved_stats.set_index("delivery_start_utc").rename(
        columns={f"chronos2__{q}": q for q in ("q10", "q50", "q90")})
    assert result.raw_history.q50.dtype == np.dtype("float64")
    assert np.any(result.raw_history.q50.to_numpy() != expected.q50.to_numpy(dtype="float64"))

    raw, future = canonical_raw(result)

    pd.testing.assert_frame_equal(raw, expected, check_exact=True)
    pd.testing.assert_frame_equal(
        future,
        saved_future.set_index("delivery_start_utc").rename(
            columns={f"chronos2__{q}": q for q in ("q10", "q50", "q90")}),
        check_exact=True,
    )
    for column in ("q10", "q50", "q90", "actual"):
        assert raw[column].dtype == np.dtype("float32")
    pd.testing.assert_frame_equal(result.raw_history, saved_checkpoint, check_exact=True)
    pd.testing.assert_frame_equal(result.residual_statistics, saved_stats, check_exact=True)
    pd.testing.assert_frame_equal(result.source_forecast, saved_future, check_exact=True)


@pytest.mark.parametrize("column", ["q10", "q50", "q90", "actual"])
def test_canonical_raw_rejects_changed_neural_values_or_observations(column: str) -> None:
    result = _legacy_result()
    result.raw_history.loc[result.raw_history.index[0], column] += 0.1
    with pytest.raises(AssertionError):
        canonical_raw(result)


def test_canonical_raw_rejects_changed_delivery_index() -> None:
    result = _legacy_result()
    result.raw_history.index += pd.Timedelta(hours=1)
    with pytest.raises(AssertionError):
        canonical_raw(result)


def test_canonical_raw_requires_original_float32_training_values() -> None:
    result = _legacy_result()
    result.residual_statistics["chronos2__q50"] = result.residual_statistics.chronos2__q50.astype("float64")
    with pytest.raises(ValueError, match="original float32"):
        canonical_raw(result)


def _epoch(directory: Path, first_day: str, files: dict[str, bytes] | None = None) -> bytes:
    directory.mkdir(parents=True)
    first = date.fromisoformat(first_day)
    record = {"schema_version": 1, "contract_digest": "a" * 64,
              "first_delivery_day": first.isoformat(),
              "anchor_day": (first - timedelta(days=730)).isoformat()}
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode()
    (directory / "epoch.json").write_bytes(encoded)
    for relative, content in (files or {}).items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return encoded


def test_publish_epoch_preserves_populated_epoch_with_different_anchor(tmp_path: Path) -> None:
    staged, destination, backups = tmp_path / "staged", tmp_path / "epochs" / "contract", tmp_path / "backups"
    _epoch(staged, "2026-09-09", {"chronos/day.json": b"imported"})
    original = _epoch(destination, "2026-09-10", {"chronos/day.json": b"existing"})

    with pytest.raises(ValueError, match="occupied epoch has a different anchor"):
        publish_epoch(staged, destination, backups)

    assert (destination / "epoch.json").read_bytes() == original
    assert (destination / "chronos/day.json").read_bytes() == b"existing"
    assert (staged / "chronos/day.json").read_bytes() == b"imported"
    assert not backups.exists()


def test_publish_epoch_backs_up_empty_future_epoch_before_import(tmp_path: Path) -> None:
    staged, destination, backups = tmp_path / "staged", tmp_path / "epochs" / "contract", tmp_path / "backups"
    imported = _epoch(staged, "2026-09-09", {"chronos/day.json": b"verified"})
    original = _epoch(destination, "2026-09-10")

    publish_epoch(staged, destination, backups)

    assert (backups / "contract_empty_epoch" / "epoch.json").read_bytes() == original
    assert (destination / "epoch.json").read_bytes() == imported
    assert (destination / "chronos/day.json").read_bytes() == b"verified"
    assert not staged.exists()


def test_publish_epoch_accepts_identical_repeat_and_preserves_other_entries(tmp_path: Path) -> None:
    destination, backups = tmp_path / "epochs" / "contract", tmp_path / "backups"
    first = tmp_path / "first"
    imported = _epoch(first, "2026-09-09", {"chronos/day.json": b"verified"})
    publish_epoch(first, destination, backups)
    (destination / "newer-day.json").write_bytes(b"later daily run")
    repeated = tmp_path / "repeated"
    _epoch(repeated, "2026-09-09", {"chronos/day.json": b"verified"})

    publish_epoch(repeated, destination, backups)

    assert (destination / "epoch.json").read_bytes() == imported
    assert (destination / "chronos/day.json").read_bytes() == b"verified"
    assert (destination / "newer-day.json").read_bytes() == b"later daily run"
    assert not backups.exists()


def test_publish_epoch_rejects_conflict_without_overwrite_or_partial_copy(tmp_path: Path) -> None:
    staged, destination, backups = tmp_path / "staged", tmp_path / "epochs" / "contract", tmp_path / "backups"
    original = _epoch(destination, "2026-09-09", {"residual/day.parquet": b"original verified bytes"})
    _epoch(staged, "2026-09-09", {"new-day.json": b"new entry", "residual/day.parquet": b"different bytes"})

    with pytest.raises(ValueError, match="Existing cache entry differs"):
        publish_epoch(staged, destination, backups)

    assert (destination / "epoch.json").read_bytes() == original
    assert (destination / "residual/day.parquet").read_bytes() == b"original verified bytes"
    assert not (destination / "new-day.json").exists()
    assert (staged / "residual/day.parquet").read_bytes() == b"different bytes"
    assert not backups.exists()
