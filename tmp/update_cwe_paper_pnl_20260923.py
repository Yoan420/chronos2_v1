"""Migrate only rolling report sections; verify all other report bytes and scores.

Staging is read-only with respect to reports. --publish publishes the previously
verified stage, after rechecking report and source hashes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chronos2_hourly.model_storm_data import load_model_storm_payload
from chronos2_hourly.model_storm_rolling import build_rolling_performance
from chronos2_hourly.model_storm_rolling_view import render_rolling_section

BACKUP = ROOT / "tmp/cwe_pnl_paper_20260923"
BLOCK = re.compile(r'<style>\s*#rolling-performance\{.*?<script id="rolling-performance-data" type="application/json">(.*?)</script><script>.*?</script>', re.S)
DATA = re.compile(r'<script id="rolling-performance-data" type="application/json">(.*?)</script>', re.S)
DAYS = [f"2026-09-{d}" for d in range(19, 25)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_files(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from source_files(item)
    elif isinstance(value, list):
        for item in value:
            yield from source_files(item)
    elif isinstance(value, str) and len(value) < 1024:
        path = Path(value)
        if path.is_absolute() and path.is_file():
            yield path


def check_same(a, b, label):
    if isinstance(a, float) and isinstance(b, (float, int)):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10), (label, a, b)
    else:
        assert a == b, (label, a, b)


def check_accuracy(old, new):
    previous = {z["zone"]: z for z in old["internal_zones"]}
    assert set(previous) == {z["zone"] for z in new["zones"]}
    count = 0
    for zone in new["zones"]:
        original = previous[zone["zone"]]
        for key in ("anchor_day", "invalid_timestamp_rows", "duplicate_hours", "history_status"):
            check_same(original[key], zone[key], (zone["zone"], key))
        for period, window in zone["windows"].items():
            prior = original["windows"][period]
            for key in ("start_day", "end_day", "expected_hours", "paired_hours", "complete_paired_days", "observed_complete_days"):
                check_same(prior[key], window[key], (zone["zone"], period, key))
            for freq, view in window["frequencies"].items():
                oldview = prior["frequencies"][freq]
                check_same(oldview["samples"], view["samples"], (zone["zone"], period, freq, "samples"))
                providers = {p["key"]: p for p in oldview["providers"]}
                for row in view["providers"]:
                    for metric in ("samples", "mae", "bias", "rmse", "hit_rate", "r2"):
                        check_same(providers[row["key"]][metric], row[metric], (zone["zone"], period, freq, row["key"], metric))
                        count += 1
    return count


def check_pnl(data):
    assert data["schema_version"] == 3
    assert data["source_scope"] == "internal_completed"
    assert set(data["strategy_zones"]) == {"quantile_based", "unlimited_bid"}
    assert "scopes" not in data
    qb = {z["zone"]: z for z in data["strategy_zones"]["quantile_based"]}
    for zone in data["strategy_zones"]["unlimited_bid"]:
        qzone = qb[zone["zone"]]
        for period, window in zone["windows"].items():
            qwindow = qzone["windows"][period]
            assert window["pnl_support_days"] == qwindow["pnl_support_days"]
            for comparison in (window, qwindow):
                for freq, view in comparison["frequencies"].items():
                    for row in view["providers"]:
                        assert row["pnl_days"] == comparison["pnl_days"]
                        assert row["trade_days"] <= row["submitted_days"] <= row["pnl_days"]
                        if row["pnl_days"]:
                            assert math.isfinite(row["total_pnl"])
                            check_same(row["daily_pnl"] * row["pnl_days"], row["total_pnl"], "pnl mean")
                        else:
                            assert row["total_pnl"] is row["daily_pnl"] is None


def stage():
    BACKUP.mkdir(parents=True, exist_ok=True)
    records, sources = [], {}
    for day in DAYS:
        path = ROOT / "runs/reports/model_storm" / f"CWE_Model_Storm_{day}.html"
        original = path.read_bytes()
        text = original.decode("utf-8")
        matches = list(BLOCK.finditer(text))
        assert len(matches) == 1, (day, len(matches))
        match = matches[0]
        old = json.loads(match.group(1))
        assert old["schema_version"] != 3, "Use --publish for an existing stage; never migrate a migrated report."
        options = {"history_from_delivery": "2026-09-22"} if day in ("2026-09-20", "2026-09-21") else {}
        payload = load_model_storm_payload(ROOT, day, **options)
        for zone in payload["zones"]:
            for source in (zone["sources"], zone["rolling_history"]["sources"]):
                for file in source_files(source):
                    digest = sha(file.read_bytes())
                    if str(file) in sources:
                        assert sources[str(file)] == digest
                    sources[str(file)] = digest
        updated_data = build_rolling_performance(payload)
        comparisons = check_accuracy(old, updated_data)
        check_pnl(updated_data)
        section = render_rolling_section(updated_data)
        assert section.count('data-rolling-strategy="') == 2
        assert "data-rolling-scope=" not in section
        assert "Cache Only" not in section
        assert 'aria-label="En cours de construction"' in section
        encoded_data = json.loads(DATA.search(section)[1])
        check_pnl(encoded_data)
        prefix, suffix = text[:match.start()], text[match.end():]
        updated = (prefix + section + suffix).encode("utf-8")
        newmatch = BLOCK.search(updated.decode("utf-8"))
        assert updated[:len(prefix.encode("utf-8"))] == prefix.encode("utf-8")
        assert updated.decode("utf-8")[newmatch.end():] == suffix
        backup = BACKUP / path.name
        if backup.exists():
            assert backup.read_bytes() == original, "Existing backup differs."
        else:
            backup.write_bytes(original)
        staged = BACKUP / (path.stem + ".staged.html")
        staged.write_bytes(updated)
        record = {"day": day, "report": str(path), "backup": str(backup), "staged": str(staged),
                  "before_sha256": sha(original), "after_sha256": sha(updated),
                  "outside_rolling_sha256": sha((prefix + suffix).encode("utf-8")),
                  "outside_rolling_unchanged": True, "accuracy_values_unchanged": comparisons,
                  "summary": {strategy: [{"zone": z["zone"], "anchor": z["anchor_day"],
                      "days_365": z["windows"]["365"]["pnl_days"],
                      "90_day": z["windows"]["90"]["frequencies"]["60min"]["providers"]}
                      for z in zones] for strategy, zones in updated_data["strategy_zones"].items()}}
        records.append(record)
        print(json.dumps({"staged": day, "unchanged_accuracy_values": comparisons,
                          "eligible_365": {z["zone"]: z["windows"]["365"]["pnl_days"] for z in updated_data["zones"]}}), flush=True)
    for path, digest in sources.items():
        assert sha(Path(path).read_bytes()) == digest, ("Source changed", path)
    verification = {"published": False, "source_sha256": sources, "reports": records}
    (BACKUP / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"stage_complete": True, "reports": len(records), "source_files_unchanged": len(sources)}), flush=True)


def publish():
    manifest = BACKUP / "verification.json"
    verification = json.loads(manifest.read_text(encoding="utf-8"))
    assert not verification["published"]
    for path, digest in verification["source_sha256"].items():
        assert sha(Path(path).read_bytes()) == digest, ("Source changed", path)
    for record in verification["reports"]:
        assert sha(Path(record["report"]).read_bytes()) == record["before_sha256"]
        assert sha(Path(record["staged"]).read_bytes()) == record["after_sha256"]
    for record in verification["reports"]:
        report = Path(record["report"])
        with tempfile.NamedTemporaryFile(dir=report.parent, suffix=".tmp", delete=False) as out:
            out.write(Path(record["staged"]).read_bytes())
            temporary = out.name
        os.replace(temporary, report)
        assert sha(report.read_bytes()) == record["after_sha256"]
    verification["published"] = True
    manifest.write_text(json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"published": len(verification["reports"]), "verification": str(manifest)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    publish() if args.publish else stage()
