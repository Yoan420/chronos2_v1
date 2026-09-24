from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from chronos2_exogenous.feature_bank import ParquetFeatureSource
from chronos2_exogenous.history_recovery import (
    audit_local_history,
    derive_recovery_window,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recovery_window_has_exact_inclusive_1095_days_and_context() -> None:
    window = derive_recovery_window("2026-09-03")
    assert window.required_origins == 1095
    assert window.first_delivery_day == pd.Timestamp("2023-09-05")
    assert window.bank_materialization_start_day == pd.Timestamp("2023-06-09")
    assert window.first_origin_utc == pd.Timestamp("2023-09-04T06:00:00Z")
    assert window.first_context_utc == pd.Timestamp("2023-06-11T14:00:00Z")
    assert window.last_delivery_utc == pd.Timestamp("2026-09-03T21:00:00Z")


def test_local_audit_reports_missing_prefix_without_promoting_pit(
    tmp_path: Path, monkeypatch
) -> None:
    target_path = tmp_path / "target.csv.gz"
    # Use a compact 2+2+2-day contract in this unit test.
    window = derive_recovery_window(
        "2024-01-10", training_days=2, oof_days=2, holdout_days=2, context_length=4
    )
    target_index = pd.date_range(
        window.first_context_utc, window.last_delivery_utc, freq="h"
    )
    pd.DataFrame({"timestamp": target_index, "value": 1.0}).to_csv(
        target_path, index=False, compression="gzip"
    )
    source_path = tmp_path / "feature.parquet"
    source_index = pd.date_range("2024-01-08T23:00:00Z", "2024-01-10T22:00:00Z", freq="h")
    frame = pd.DataFrame(
        {
            "value_time_utc": source_index,
            "cutoff_time_utc": [
                (timestamp.tz_convert("Europe/Paris").normalize() - pd.Timedelta(days=1))
                .replace(hour=8)
                .tz_convert("UTC")
                for timestamp in source_index
            ],
            "revision_time_utc": [
                (timestamp.tz_convert("Europe/Paris").normalize() - pd.Timedelta(days=1))
                .replace(hour=8)
                .tz_convert("UTC")
                for timestamp in source_index
            ],
            "feature": 2.0,
        }
    )
    frame.to_parquet(source_path, index=False)
    audit_path = source_path.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "sha256": _sha(source_path),
                "cutoff_time": "08:00",
                "cutoff_timezone": "Europe/Paris",
                "causality_violations": 0,
                "start_day": "2024-01-09",
                "end_day": "2024-01-10",
            }
        ),
        encoding="utf-8",
    )
    source = ParquetFeatureSource(
        name="test_feature",
        family="weather",
        path=source_path,
        audit_path=audit_path,
        value_columns={"test_feature": "feature"},
        cutoff_column="cutoff_time_utc",
        information_time_columns=("revision_time_utc",),
        age_column="revision_time_utc",
        production_evidence_kind="versioned_revision_history",
    )
    monkeypatch.setattr(
        "chronos2_exogenous.history_recovery.default_project_sources",
        lambda *_args, **_kwargs: (source,),
    )
    result = audit_local_history(
        tmp_path,
        target_paths={"FR": target_path},
        end_day="2024-01-10",
        zones=("FR",),
        training_days=2,
        oof_days=2,
        holdout_days=2,
        context_length=4,
    )
    assert result["research_horizon_ready"] is False
    assert result["production_ready"] is False
    assert result["targets"][0]["covers_context_and_horizons"] is True
    assert result["sources"][0]["missing_day_count"] == 4
    assert result["sources"][0]["contiguous_complete_suffix_start"] == "2024-01-09"
    assert result["sources"][0]["production_pit_evidence"] is False
