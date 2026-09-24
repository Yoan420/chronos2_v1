"""Read four raw primary quarter-hour price series into an isolated research archive."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import json
from pathlib import Path

from filelock import FileLock
import pandas as pd

from nyx_quarterhour.sources import (ROOT, SOURCES, ZONES, civil_day, digest, empty_frame, fetch_range,
                                    make_client, quarter_grid, read_native_prices, safe_output)


def write_json(path: Path, value: dict):
    path = safe_output(path)
    temporary = safe_output(path.with_suffix(path.suffix+".tmp"))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def collect(start_day: str, end_day: str, output: Path, workers: int = 2) -> dict:
    quarter_grid(start_day, end_day)
    if civil_day(start_day) < date(2025, 10, 1):
        raise ValueError("The audited native day-ahead quarter-hour window starts 2025-10-01.")
    if workers not in (1, 2):
        raise ValueError("One or two simultaneous reads are permitted.")
    output = safe_output(output)
    if output.exists():
        raise ValueError("Use a fresh dedicated archive directory; sealed archives are never overwritten.")
    output.mkdir(parents=True)
    with FileLock(str(safe_output(output/"collection.lock")), timeout=0):
        results = {}

        def one(zone):
            client = make_client()
            try:
                return zone, fetch_range(client, zone, start_day, end_day)
            finally:
                client.session.close()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(one, z) for z in ZONES]):
                zone, (frame, audit) = future.result()
                results[zone] = (frame, audit)
                print(json.dumps({k: audit[k] for k in ("zone", "series", "status", "returned_quarters")}), flush=True)
        audits = [results[z][1] for z in ZONES]
        frames = [results[z][0] for z in ZONES if len(results[z][0])]
        table = pd.concat(frames, ignore_index=True).sort_values(["timestamp_utc", "zone"]).reset_index(drop=True) if frames else empty_frame()
        path = safe_output(output/"native_prices.parquet")
        table.to_parquet(path, index=False)
        audit_path = output/"collection_audit.json"
        write_json(audit_path, {"start_day": start_day, "end_day": end_day, "countries": audits})
        status = "complete" if all(a["status"] == "complete" for a in audits) else "partial"
        discovery = ROOT/"tmp/nyx_quarterhour_discovery"
        manifest = {"schema_version": 1, "artifact_type": "nyx_quarterhour_price_observations",
                    "data_file": path.name, "data_sha256": digest(path), "sources": SOURCES,
                    "timezone": "UTC", "unit": "EUR/MWh", "resolution_minutes": 15,
                    "interpolation": "none", "price_vintage": "latest_observations", "status": status,
                    "provider_revision_timestamp_available": False, "production_pit_evidence": False,
                    "start_day": start_day, "end_day": end_day,
                    "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "summary": {"rows": len(table), "countries": {a["zone"]: {k:a.get(k) for k in ("status", "returned_quarters", "expected_quarters", "complete_hours", "varying_hours", "missing_quarters")} for a in audits}},
                    "collection_audit_sha256": digest(audit_path),
                    "evidence_sha256": {str(p.relative_to(ROOT)): digest(p) for p in discovery.glob("*.json")},
                    "acquisition_code_sha256": {"sources.py": digest(ROOT/"nyx_quarterhour/sources.py"), "materialize_nyx_quarterhour.py": digest(Path(__file__))}}
        write_json(output/"manifest.json", manifest)
        read_native_prices(output/"manifest.json")
        print(json.dumps({"manifest": str(output/"manifest.json"), "status": status, "rows": len(table)}), flush=True)
        return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-day", default="2025-10-01")
    p.add_argument("--end-day", default="2026-09-15")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", choices=(1, 2), type=int, default=2)
    a = p.parse_args()
    collect(a.start_day, a.end_day, a.output_dir, a.workers)


if __name__ == "__main__":
    main()
