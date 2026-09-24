"""Collect an isolated, retrospective native-quarter-hour BE solar archive.

No NYX model or production cache is touched. Maximum concurrency is two reads.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import stat

from filelock import FileLock
import pandas as pd

from nyx_intrahour.saturn_sources import SOURCE, civil_day, empty_frame, fetch_day, make_client

ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT = ROOT / "data/pit/nyx_intrahour"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_target(path: Path) -> Path:
    """Reject redirected descendants too, including files and temporary files."""
    namespace = Path(os.path.abspath(ARCHIVE_ROOT))
    lexical = Path(os.path.abspath(path))
    if namespace.resolve() != namespace or not lexical.is_relative_to(namespace):
        raise ValueError("The research path must stay inside its lexical namespace.")
    if lexical.resolve() != lexical:
        raise ValueError("Research descendants must not use a symlink or junction.")
    for part in (lexical, *lexical.parents):
        if part == namespace.parent:
            break
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Research descendants must not use a symlink, junction or reparse point.")
    return lexical


def write_json(path: Path, value: dict):
    path = safe_target(path)
    temporary = safe_target(path.with_suffix(path.suffix+".tmp"))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def collect(start_day: str, end_day: str, output: Path, workers: int = 2, resume: bool = False) -> dict:
    first, last = civil_day(start_day), civil_day(end_day)
    if first > last or (last-first).days > 730:
        raise ValueError("An ordered window of at most 731 days is required.")
    if workers not in (1, 2):
        raise ValueError("Use one or two concurrent read-only queries.")
    namespace = Path(os.path.abspath(ARCHIVE_ROOT))
    if namespace.resolve() != namespace:
        raise ValueError("The research namespace must not resolve through a symlink or junction.")
    lexical_output = Path(os.path.abspath(output))
    output = lexical_output.resolve()
    if output != lexical_output:
        raise ValueError("The archive must not resolve through a symlink or junction.")
    if not output.is_relative_to(namespace) or output == namespace:
        raise ValueError("Use a dedicated child directory of data/pit/nyx_intrahour.")
    if output.exists() and not resume:
        raise ValueError("Output exists; --resume is required for this isolated archive.")
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(safe_target(output / "collection.lock")), timeout=0):
        identity = {"source": SOURCE, "start_day": start_day, "end_day": end_day,
                    "cutoff": "D-1 08:00 Europe/Paris", "interpolation": "none"}
        identity_path = safe_target(output / "collection_identity.json")
        if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf8")) != identity:
            raise ValueError("Resume identity differs from the existing isolated archive.")
        write_json(identity_path, identity)
        days = [(first+timedelta(days=i)).isoformat() for i in range((last-first).days+1)]
        shards = safe_target(output / "days")
        shards.mkdir(exist_ok=True)
        results = {}
        pending = []
        for day in days:
            report, data = safe_target(shards / f"{day}.json"), safe_target(shards / f"{day}.parquet")
            if resume and report.exists() and data.exists():
                saved = json.loads(report.read_text(encoding="utf8"))
                if saved.get("status") != "error" and saved.get("data_sha256") == sha(data):
                    results[day] = saved
                    continue
            pending.append(day)

        def one(day):
            client = make_client()
            try:
                frame, audit = fetch_day(client, day)
            finally:
                client.session.close()
            data = safe_target(shards / f"{day}.parquet")
            frame.to_parquet(data, index=False)
            audit["data_sha256"] = sha(data)
            write_json(shards / f"{day}.json", audit)
            return day, audit

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, day) for day in pending]
            for future in as_completed(futures):
                day, audit = future.result()
                results[day] = audit
                count = len(results)
                if count % 20 == 0 or audit["status"] != "complete" or count == len(days):
                    print(json.dumps({"completed_days": count, "total_days": len(days), "day": day,
                                      "status": audit["status"], "returned_quarters": audit["returned_quarters"]}), flush=True)
        audits = [results[day] for day in days]
        frames = [pd.read_parquet(safe_target(shards/f"{day}.parquet")) for day in days]
        nonempty = [f for f in frames if len(f)]
        table = pd.concat(nonempty, ignore_index=True) if nonempty else empty_frame()
        table = table.sort_values(["value_time_utc", "source_alias"]).reset_index(drop=True)
        if table.duplicated(["source_alias", "value_time_utc", "revision_time_utc"]).any():
            raise ValueError("Overlapping source/day identities are forbidden.")
        data_file = safe_target(output / "native_forecasts.parquet")
        table.to_parquet(data_file, index=False)
        summary = {"days": len(days), "complete_days": sum(a["status"] == "complete" for a in audits),
                   "incomplete_days": sum(a["status"] == "incomplete" for a in audits),
                   "empty_days": sum(a["status"] == "empty" for a in audits),
                   "error_days": sum(a["status"] == "error" for a in audits),
                   "expected_quarters": sum(a["expected_quarters"] for a in audits),
                   "returned_quarters": len(table),
                   "varying_hours": sum(a.get("varying_hours", 0) for a in audits)}
        write_json(output / "collection_audit.json", {"identity": identity, "summary": summary, "days": audits})
        discovery = ROOT / "tmp/nyx_intrahour_discovery"
        evidence = {str(p.relative_to(ROOT)): sha(p) for p in sorted(discovery.glob("*.json"))}
        manifest = {"schema_version": 1, "artifact_type": "nyx_intrahour_forecast_vintages",
                    "data_file": data_file.name, "data_sha256": sha(data_file), "sources": [SOURCE],
                    "temporal_evidence": "retrospective_asof", "provider_revision_timestamp_available": False,
                    "production_pit_evidence": False, "start_day": start_day, "end_day": end_day,
                    "cutoff": "D-1 08:00 Europe/Paris", "timestamp_semantics": "Both snapshot and revision are requested query cutoff times, not provider publication timestamps.",
                    "scope": "Belgian solar fundamental only; no claim of four-country native-quarter-hour coverage.",
                    "interpolation": "none", "summary": summary, "evidence_sha256": evidence,
                    "collection_audit_sha256": sha(output / "collection_audit.json"),
                    "acquisition_code_sha256": {"materialize_nyx_intrahour.py": sha(Path(__file__)),
                                                "nyx_intrahour/saturn_sources.py": sha(ROOT/"nyx_intrahour/saturn_sources.py")}}
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"manifest": str(output / "manifest.json"), **summary}), flush=True)
        return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-day", required=True)
    p.add_argument("--end-day", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    collect(args.start_day, args.end_day, args.output_dir, args.workers, args.resume)


if __name__ == "__main__":
    main()
