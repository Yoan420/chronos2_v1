#!/usr/bin/env python
"""Audit isolated non-Storm external-price PIT materialisations.

The audit is deliberately label-free: it validates only timestamps, physical
hours, finite values, and the D-1 08:00 Europe/Paris as-of contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TIMEZONE = "Europe/Paris"
START_DAY = pd.Timestamp("2024-01-02")
END_DAY = pd.Timestamp("2025-08-11")
EXPECTED_ROWS = 14_111
EXPECTED_DAYS = 588


def _expected_cutoff(local_day: pd.Timestamp) -> pd.Timestamp:
    naive = pd.Timestamp(local_day.date()) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    return naive.tz_localize(TIMEZONE).tz_convert("UTC")


def _audit(path: Path, series: str) -> dict[str, object]:
    # The directory is intentionally named ``nonstorm``; the hard exclusion
    # applies to the Saturn series/provider identifier, not that audit label.
    if "storm" in series.lower():
        raise RuntimeError(f"Storm guard triggered before reading {path}")
    frame = pd.read_parquet(path)
    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
        "downloaded_at_utc",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{path}: missing columns {missing}")
    for column in ("value_time_utc", "snapshot_time_utc", "revision_time_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    delivery = pd.DatetimeIndex(frame["value_time_utc"])
    local = delivery.tz_convert(TIMEZONE)
    local_days = pd.Index(local.date)
    expected_cutoff = pd.Series(
        [_expected_cutoff(day) for day in local.normalize()],
        index=frame.index,
        dtype="datetime64[ns, UTC]",
    )
    actual_snapshot = frame["snapshot_time_utc"]
    actual_revision = frame["revision_time_utc"]
    day_counts = pd.Series(1, index=local_days).groupby(level=0).sum()
    duplicate_rows = int(delivery.duplicated().sum())
    snapshot_mismatch = int((actual_snapshot != expected_cutoff).sum())
    revision_after_cutoff = int((actual_revision > expected_cutoff).sum())
    revision_not_snapshot = int((actual_revision != actual_snapshot).sum())
    finite = np.isfinite(pd.to_numeric(frame["value"], errors="coerce").to_numpy(float))
    first_day = str(min(local_days)) if len(local_days) else None
    last_day = str(max(local_days)) if len(local_days) else None
    counts_histogram = {
        str(int(hours)): int(count)
        for hours, count in day_counts.value_counts().sort_index().items()
    }
    passed = bool(
        len(frame) == EXPECTED_ROWS
        and len(day_counts) == EXPECTED_DAYS
        and first_day == str(START_DAY.date())
        and last_day == str(END_DAY.date())
        and duplicate_rows == 0
        and bool(finite.all())
        and snapshot_mismatch == 0
        and revision_after_cutoff == 0
        and revision_not_snapshot == 0
        and set(day_counts.unique()).issubset({23, 24, 25})
    )
    return {
        "series": series,
        "external_dependency": True,
        "autonomous_model": False,
        "provider_license": "not declared in Saturn metadata; entitlement must be confirmed",
        "path": str(path.resolve()),
        "rows": int(len(frame)),
        "local_days": int(len(day_counts)),
        "first_local_day": first_day,
        "last_local_day": last_day,
        "first_delivery_utc": str(delivery.min()),
        "last_delivery_utc": str(delivery.max()),
        "day_hour_counts": counts_histogram,
        "duplicate_delivery_rows": duplicate_rows,
        "nonfinite_values": int((~finite).sum()),
        "snapshot_cutoff_mismatches": snapshot_mismatch,
        "revision_after_cutoff_violations": revision_after_cutoff,
        "revision_snapshot_mismatches": revision_not_snapshot,
        "cutoff_contract": "D-1 08:00 Europe/Paris civil time",
        "revision_semantics": (
            "revision_time_utc records the explicit Saturn as-of query cutoff; "
            "Saturn get(revision_date=cutoff) enforces source revision <= cutoff"
        ),
        "storm_guard_passed": True,
        "audit_passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcarbon", required=True)
    parser.add_argument("--mkonline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = [
        _audit(
            Path(args.pointcarbon),
            "power.price.fr.euromwh.h.fcst.pointcarbon.flowspot",
        ),
        _audit(
            Path(args.mkonline),
            "power.price.fr.euromwh.h.fcst.mkonline.ecop",
        ),
    ]
    payload = {
        "protocol": {
            "labels_read": False,
            "final_period_read": False,
            "materialized_period": "2024-01-02..2025-08-11",
            "expected_partition": "EXT 5351h + CAL/A/B1/B2 8760h",
            "storm_series_used": False,
        },
        "passed": bool(all(item["audit_passed"] for item in results)),
        "results": results,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["passed"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
