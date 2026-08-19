#!/usr/bin/env python
"""Audit the six direct-primary MKOnline PIT materialisations.

This audit is label-free and refuses formula-wrapper inputs.  It emits one
global provenance manifest with exact commands, dependency metadata, cutoff
checks, DST coverage, and SHA-256 file hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "runs" / "tmp" / "saturn_fr_price_nonstorm"
OUTPUT = DATA_DIR / "mkonline_coupled_primary_materializations_manifest.json"
DEPENDENCY_AUDIT = DATA_DIR / "mkonline_coupled_dependency_audit.json"
TIMEZONE = "Europe/Paris"
START_DAY = "2024-01-02"
END_DAY = "2025-06-12"
EXPECTED_DAYS = 528
EXPECTED_ROWS = 12_671

SPECS = (
    {
        "zone": "DE",
        "terminal": "41550_native",
        "root_formula": "power.price.de.euromwh.h.fcst.mkonline.ecop",
        "alias": "mkonline_ecop_de_primary",
        "filename": "coupled_mkonline_ecop_de_primary_ext_a_b1.parquet",
    },
    {
        "zone": "BE",
        "terminal": "41555_native",
        "root_formula": "power.price.be.euromwh.h.fcst.mkonline.ecop",
        "alias": "mkonline_ecop_be_primary",
        "filename": "coupled_mkonline_ecop_be_primary_ext_a_b1.parquet",
    },
    {
        "zone": "NL",
        "terminal": "41554_native",
        "root_formula": "power.price.nl.euromwh.h.fcst.mkonline.ecop",
        "alias": "mkonline_ecop_nl_primary",
        "filename": "coupled_mkonline_ecop_nl_primary_ext_a_b1.parquet",
    },
    {
        "zone": "AT",
        "terminal": "41552_native",
        "root_formula": "power.price.at.euromwh.h.fcst.mkonline.ecop",
        "alias": "mkonline_ecop_at_primary",
        "filename": "coupled_mkonline_ecop_at_primary_ext_a_b1.parquet",
    },
    {
        "zone": "CH",
        "terminal": "41553_native",
        "root_formula": "power.price.ch.euromwh.h.fcst.mkonline.ecop",
        "alias": "mkonline_ecop_ch_primary",
        "filename": "coupled_mkonline_ecop_ch_primary_ext_a_b1.parquet",
    },
    {
        "zone": "ES",
        "terminal": "58307_native",
        "root_formula": "power.price.es.euromwh.h.fcst.mkonline.monthly",
        "alias": "mkonline_monthly_es_primary",
        "filename": "coupled_mkonline_monthly_es_primary_ext_a_b1.parquet",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_cutoff(local_timestamp: pd.Timestamp) -> pd.Timestamp:
    naive = (
        pd.Timestamp(local_timestamp.date())
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=8)
    )
    return naive.tz_localize(TIMEZONE).tz_convert("UTC")


def _command(spec: dict[str, str], path: Path) -> str:
    return (
        "& 'C:\\Users\\BQ6757\\venvs\\pricefm311\\Scripts\\python.exe' "
        "materialize_saturn_daily_asof.py "
        f"--series '{spec['terminal']}' --alias '{spec['alias']}' "
        f"--start-day {START_DAY} --end-day {END_DAY} "
        f"--output '{path}' --workers 4 --hourly-on-the-hour"
    )


def _audit_one(
    spec: dict[str, str],
    dependency: dict[str, object],
) -> dict[str, object]:
    terminal = spec["terminal"]
    if "storm" in f"{terminal} {spec['root_formula']}".lower():
        raise RuntimeError(f"Storm guard triggered for {terminal}")
    if dependency.get("terminal") != terminal:
        raise RuntimeError(f"Dependency mismatch for {terminal}")
    if dependency.get("terminal_type") != "primary":
        raise RuntimeError(f"Non-primary terminal refused: {terminal}")
    if dependency.get("terminal_formula") is not None:
        raise RuntimeError(f"Formula terminal refused: {terminal}")
    if dependency.get("storm_token_found") is not False:
        raise RuntimeError(f"Unproven Storm guard for {terminal}")
    path = DATA_DIR / spec["filename"]
    frame = pd.read_parquet(path)
    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
        "downloaded_at_utc",
    }
    if set(frame.columns) != required:
        raise RuntimeError(f"Unexpected schema for {path}: {list(frame.columns)}")
    for column in ("value_time_utc", "snapshot_time_utc", "revision_time_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    delivery = pd.DatetimeIndex(frame["value_time_utc"])
    local = delivery.tz_convert(TIMEZONE)
    days = pd.Index(local.date)
    day_counts = pd.Series(1, index=days).groupby(level=0).sum()
    expected_cutoff = pd.Series(
        [_expected_cutoff(timestamp) for timestamp in local],
        index=frame.index,
        dtype="datetime64[ns, UTC]",
    )
    snapshot_mismatch = int((frame["snapshot_time_utc"] != expected_cutoff).sum())
    revision_after_cutoff = int((frame["revision_time_utc"] > expected_cutoff).sum())
    revision_snapshot_mismatch = int(
        (frame["revision_time_utc"] != frame["snapshot_time_utc"]).sum()
    )
    finite = np.isfinite(pd.to_numeric(frame["value"], errors="coerce").to_numpy(float))
    histogram = {
        str(int(hours)): int(count)
        for hours, count in day_counts.value_counts().sort_index().items()
    }
    passed = bool(
        len(frame) == EXPECTED_ROWS
        and len(day_counts) == EXPECTED_DAYS
        and str(days.min()) == START_DAY
        and str(days.max()) == END_DAY
        and int(delivery.duplicated().sum()) == 0
        and bool(finite.all())
        and histogram == {"23": 2, "24": 525, "25": 1}
        and snapshot_mismatch == 0
        and revision_after_cutoff == 0
        and revision_snapshot_mismatch == 0
    )
    return {
        **spec,
        "provider": dependency.get("provider"),
        "source": dependency.get("source"),
        "terminal_metadata_sha256": dependency.get("terminal_metadata_sha256"),
        "path": str(path.resolve()),
        "command": _command(spec, path),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "local_days": int(len(day_counts)),
        "first_local_day": str(days.min()),
        "last_local_day": str(days.max()),
        "first_delivery_utc": str(delivery.min()),
        "last_delivery_utc": str(delivery.max()),
        "day_hour_counts": histogram,
        "duplicate_delivery_rows": int(delivery.duplicated().sum()),
        "nonfinite_values": int((~finite).sum()),
        "snapshot_cutoff_mismatches": snapshot_mismatch,
        "revision_after_cutoff_violations": revision_after_cutoff,
        "revision_snapshot_mismatches": revision_snapshot_mismatch,
        "audit_passed": passed,
    }


def main() -> int:
    dependency_payload = json.loads(DEPENDENCY_AUDIT.read_text(encoding="utf-8"))
    by_root = {item["root"]: item for item in dependency_payload["series"]}
    results = [_audit_one(spec, by_root[spec["root_formula"]]) for spec in SPECS]
    payload = {
        "protocol": {
            "data_source": "direct Saturn primary terminals only",
            "formula_wrappers_materialized": False,
            "cutoff": "D-1 08:00 Europe/Paris civil time",
            "period": f"{START_DAY}..{END_DAY} (EXT+A+B1 only)",
            "B2_read": False,
            "final_read": False,
            "labels_read": False,
            "storm_series_used": False,
            "dependency_audit": str(DEPENDENCY_AUDIT.resolve()),
        },
        "all_passed": bool(all(item["audit_passed"] for item in results)),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["all_passed"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
