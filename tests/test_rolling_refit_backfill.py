from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.rolling_refit_backfill import (
    PitBackfillSource,
    RollingRefitBackfillError,
    reconstruct_pit_backfill_day,
)


TIMEZONE = "Europe/Paris"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _origin(day: date) -> pd.Timestamp:
    return (
        pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    ).tz_localize("Europe/Paris").tz_convert("UTC")


def _sources(
    tmp_path: Path,
    day: date,
    *,
    missing_tail: int = 2,
) -> tuple[list[PitBackfillSource], pd.DataFrame]:
    index = local_delivery_day_index(day, timezone=TIMEZONE).tz_convert("UTC")
    cutoff = _origin(day)
    archived = pd.DataFrame(index=index)
    sources: list[PitBackfillSource] = []
    kept_index = index[: len(index) - missing_tail] if missing_tail else index
    for number, alias in enumerate(("fr_load", "de_load"), start=1):
        value = number * 10.0 + np.arange(len(kept_index), dtype=float)
        early = pd.DataFrame(
            {
                "value_time_utc": kept_index,
                "snapshot_time_utc": cutoff - pd.Timedelta(hours=3),
                "revision_time_utc": cutoff - pd.Timedelta(hours=4),
                "value": value - 1.0,
            }
        )
        selected = pd.DataFrame(
            {
                "value_time_utc": kept_index,
                "snapshot_time_utc": cutoff - pd.Timedelta(hours=2 - number / 2),
                "revision_time_utc": cutoff - pd.Timedelta(minutes=20 - number),
                "value": value,
            }
        )
        too_late = pd.DataFrame(
            {
                "value_time_utc": kept_index,
                "snapshot_time_utc": cutoff + pd.Timedelta(minutes=1),
                "revision_time_utc": cutoff - pd.Timedelta(minutes=1),
                "value": value + 1000.0,
            }
        )
        frame = pd.concat([too_late, early, selected], ignore_index=True)
        path = tmp_path / f"{alias}.parquet"
        frame.to_parquet(path, index=False)
        feature = f"known_{alias}_oracle"
        archived[feature] = np.nan
        archived.loc[kept_index, feature] = value
        sources.append(
            PitBackfillSource(
                alias=alias,
                path=path,
                sha256=_sha(path),
                archived_feature_column=feature,
            )
        )
    return sources, archived


def test_reconstructs_real_daily_maxima_and_preserves_missing_tail(
    tmp_path: Path,
) -> None:
    day = date(2025, 8, 16)
    sources, archived = _sources(tmp_path, day)
    result = reconstruct_pit_backfill_day(
        sources,
        delivery_day=day,
        delivery_timezone=TIMEZONE,
        archived_features=archived,
        serialization_tolerance=1e-12,
    )

    assert len(result.selected_values) == 24
    assert result.selected_values.iloc[-2:].isna().all().all()
    assert result.timestamp_evidence["pit_inputs_present"].eq(True).all()
    assert result.timestamp_evidence["maximum_snapshot_time_utc"].nunique() == 1
    assert result.timestamp_evidence["maximum_revision_time_utc"].nunique() == 1
    assert result.audit["causal_timestamp_scope"] == "delivery_day_all_pit_inputs"
    assert result.audit["network_access"] is False
    assert result.audit["files_written"] is False
    assert [source["missing_physical_hours"] for source in result.audit["sources"]] == [
        2,
        2,
    ]


@pytest.mark.parametrize(
    ("day", "hours"),
    [(date(2026, 3, 29), 23), (date(2026, 10, 25), 25)],
)
def test_backfill_respects_canonical_dst_grid(
    tmp_path: Path, day: date, hours: int
) -> None:
    sources, archived = _sources(tmp_path, day, missing_tail=0)
    result = reconstruct_pit_backfill_day(
        sources,
        delivery_day=day,
        delivery_timezone=TIMEZONE,
        archived_features=archived,
    )
    assert len(result.selected_values) == hours
    assert result.audit["physical_hours"] == hours


def test_archived_value_mismatch_fails_closed(tmp_path: Path) -> None:
    day = date(2025, 8, 16)
    sources, archived = _sources(tmp_path, day)
    archived.iloc[0, 0] += 0.1
    with pytest.raises(RollingRefitBackfillError, match="disagree"):
        reconstruct_pit_backfill_day(
            sources,
            delivery_day=day,
            delivery_timezone=TIMEZONE,
            archived_features=archived,
            serialization_tolerance=1e-5,
        )


def test_archived_missing_mask_mismatch_fails_closed(tmp_path: Path) -> None:
    day = date(2025, 8, 16)
    sources, archived = _sources(tmp_path, day)
    archived.iloc[-1, 0] = 1.0
    with pytest.raises(RollingRefitBackfillError, match="missing masks differ"):
        reconstruct_pit_backfill_day(
            sources,
            delivery_day=day,
            delivery_timezone=TIMEZONE,
            archived_features=archived,
        )


def test_changed_pit_parquet_checksum_is_rejected(tmp_path: Path) -> None:
    day = date(2025, 8, 16)
    sources, archived = _sources(tmp_path, day)
    path = Path(sources[0].path)
    original = pd.read_parquet(path)
    original.loc[0, "value"] += 1.0
    original.to_parquet(path, index=False)
    with pytest.raises(RollingRefitBackfillError, match="checksum mismatch"):
        reconstruct_pit_backfill_day(
            sources,
            delivery_day=day,
            delivery_timezone=TIMEZONE,
            archived_features=archived,
        )


def test_day_without_any_real_pit_input_is_rejected(tmp_path: Path) -> None:
    day = date(2025, 8, 16)
    index = local_delivery_day_index(day, timezone=TIMEZONE).tz_convert("UTC")
    path = tmp_path / "empty.parquet"
    pd.DataFrame(
        {
            "value_time_utc": pd.DatetimeIndex([], tz="UTC"),
            "snapshot_time_utc": pd.DatetimeIndex([], tz="UTC"),
            "revision_time_utc": pd.DatetimeIndex([], tz="UTC"),
            "value": pd.Series([], dtype=float),
        }
    ).to_parquet(path, index=False)
    source = PitBackfillSource(
        alias="fr_load",
        path=path,
        sha256=_sha(path),
        archived_feature_column="known_fr_load_oracle",
    )
    archived = pd.DataFrame(
        {"known_fr_load_oracle": np.nan}, index=index
    )
    with pytest.raises(RollingRefitBackfillError, match="no real selected"):
        reconstruct_pit_backfill_day(
            [source],
            delivery_day=day,
            delivery_timezone=TIMEZONE,
            archived_features=archived,
        )

