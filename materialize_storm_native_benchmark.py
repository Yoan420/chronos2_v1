#!/usr/bin/env python
"""Materialize one exact native Storm dashboard snapshot for offline scoring.

This command is deliberately separate from the benchmark publisher.  It only
reads the verified Saturn native series, keeps its timezone-naive civil index,
and writes a checksummed snapshot plus provenance.  It never reads target
values and the resulting curve is evaluation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from chronos2_hourly.storm_dashboard import (
    STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE,
    fetch_native_dashboard_snapshot,
    normalize_native_dashboard_series,
    storm_dashboard_series,
)
from chronos2_hourly.zone_benchmark import complete_local_range
from chronos2_modular.saturn import create_saturn_client


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize an exact evaluation-only native Storm snapshot."
    )
    parser.add_argument("--zone", required=True, choices=("DE", "BE", "NL"))
    parser.add_argument("--start-day", required=True)
    parser.add_argument("--end-day", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--saturn-url",
        default="https://saturn-energyscan.gem.myengie.com//api",
    )
    parser.add_argument("--saturn-author", default="BQ6757")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zone = str(args.zone).upper()
    timezone = STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE[zone]
    expected = complete_local_range(
        args.start_day,
        args.end_day,
        timezone=timezone,
    )
    client = create_saturn_client(args.saturn_url, args.saturn_author)
    raw, provenance = fetch_native_dashboard_snapshot(
        client,
        zone=zone,
        expected_index=expected,
    )
    raw_index = pd.DatetimeIndex(raw.index)
    if raw_index.tz is not None:
        raise ValueError("Saturn native Storm index unexpectedly has a timezone")
    if raw_index.has_duplicates or not raw_index.is_monotonic_increasing:
        raise ValueError("Saturn native Storm index is duplicate or unordered")

    # Validate the exact DST/coverage contract without consulting target data.
    synthetic_actual = pd.Series(
        np.zeros(len(expected), dtype=float),
        index=expected,
        name="actual_not_loaded",
    )
    comparator = normalize_native_dashboard_series(
        raw,
        zone=zone,
        expected_index=expected,
        actual=synthetic_actual,
        source=provenance,
    )

    output = Path(args.output).expanduser().resolve()
    sidecar = output.with_suffix(output.suffix + ".audit.json")
    if (output.exists() or sidecar.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output} or {sidecar}; pass --overwrite explicitly"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.stem}.tmp-", dir=output.parent
    ) as temporary:
        temporary_path = Path(temporary)
        staged = temporary_path / output.name
        pd.DataFrame(
            {
                "timestamp": raw_index,
                "value": pd.to_numeric(raw, errors="coerce").to_numpy(float),
            }
        ).to_parquet(staged, index=False)
        audit = {
            "schema_version": 1,
            "role": "evaluation_only_native_storm_dashboard_snapshot",
            "zone": zone,
            "timezone": timezone,
            "series": storm_dashboard_series(zone),
            "start_local_day": str(args.start_day),
            "end_local_day": str(args.end_day),
            "expected_hours": int(len(expected)),
            "raw_rows": int(len(raw)),
            "raw_index_timezone": None,
            "timestamp_column": "timestamp",
            "value_column": "value",
            "used_for_prediction": False,
            "target_values_loaded": False,
            "normalization_contract": {
                "coverage": float(comparator.audit["coverage"]),
                "available_hours": int(comparator.audit["available_hours"]),
                "missing_hours": int(comparator.audit["missing_hours"]),
                "dst": comparator.audit["dst"],
            },
            "saturn": provenance,
            "sha256": _sha256(staged),
        }
        staged_sidecar = temporary_path / sidecar.name
        _write_json(staged_sidecar, audit)
        staged.replace(output)
        staged_sidecar.replace(sidecar)

    print(f"Snapshot: {output}")
    print(f"SHA-256: {_sha256(output)}")
    print(f"Audit: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
