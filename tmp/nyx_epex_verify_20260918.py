"""Read-only verification of the actual NYX EPEX publication."""
import sys
import json
import hashlib
import sqlite3
from html import unescape
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chronos2_hourly.model_storm_data import load_model_storm_payload
from chronos2_hourly.model_storm_report import daily_metrics
from chronos2_hourly.nuclear_reporting_refresh import verify_refreshed_observations

out = ROOT / "tmp/nyx_epex_reference_20260918"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
before = json.loads((out / "frozen_before.json").read_text())
assert len(before) == 36 and all(sha(p) == digest for p, digest in before.items())
run_id = json.loads((out / "launch.json").read_text())["run"]["id"]
with sqlite3.connect((ROOT / "runs/.experiment_console/console.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
    run = json.loads(db.execute("SELECT data FROM runs WHERE id=?", (run_id,)).fetchone()[0])
assert run["status"] == "succeeded", run["status"]
assert run["return_code"] == 0
payload = load_model_storm_payload(ROOT, "2026-09-19")
zones = []
published_files = 0
for item in payload["zones"]:
    zone, tz = item["zone"], item["timezone"]
    assert item["coverage"] == {"storm": 24, "observed": 24, "model": 24}, item
    assert item["status"] == "complete"
    assert item["sources"]["observed"]["actual_reference"] == "EPEX"
    assert item["sources"]["observed"]["policy"] == "epex_only_v1"
    work = ROOT / "runs/experiments/nuclear_forecast_v1/2026-09-19" / zone.lower() / "civil_pit_v2"
    result = json.loads((work / "run_result.json").read_text(encoding="utf-8"))
    assert result["execution"]["forecast_result_reused"] is True
    audit = result["reporting_sources"]
    assert audit["schema_version"] == 2
    assert pd.Timestamp(audit["extracted_at_utc"]) > pd.Timestamp(run["started_at"])
    source = audit["observed"]["source"]
    assert source["series"] == f"power.price.{zone.lower()}.euromwh.h.obs.epex"
    assert source["fallback_policy"] == "none" and source["source_mixing"] is False
    assert "post_auction_fallback" not in source
    frame = pd.read_parquet(audit["observed"]["artifact_path"])
    observed = pd.Series(frame.actual.to_numpy(), index=pd.DatetimeIndex(frame.timestamp))
    verify_refreshed_observations(observed, audit, zone=zone, timezone=tz, delivery_day="2026-09-19")
    assert len(observed) == 8784 and observed.notna().all()
    exports = ROOT / "runs/exports/2026-09-19" / zone.lower()
    manifest = json.loads((exports / "current_nuclear_batch_manifest.json").read_text())
    for export in manifest["exports"]:
        for file in export["files"]:
            assert sha(exports / file["path"]) == file["sha256"]
            published_files += 1
    report_audit = json.loads((exports / "nuclear_kalman/nuclear_report_audit.json").read_text(encoding="utf-8"))
    assert report_audit["actual_price_reference"]["policy"] == "epex_only_v1"
    assert report_audit["same_actual_hours"] is True
    assert report_audit["delivery_day_observed_hours"] == 24
    text = unescape((exports / f"nuclear_kalman/forecast_{zone.lower()}_2026-09-19_nuclear_kalman.html").read_text(encoding="utf-8"))
    assert 'data-report-section="actual-price-reference"' in text and "EPEX via Saturn" in text
    assert "Prix réalisés du jour non validés" not in text
    metrics = daily_metrics(item)
    assert metrics["observed"] is not None
    zones.append({"zone": zone, "coverage": item["coverage"], "series": source["series"],
                  "extracted_at_utc": source["extracted_at_utc"], "daily_mean_prices": metrics})
cwe = ROOT / "runs/reports/model_storm/CWE_Model_Storm_2026-09-19.html"
text = unescape(cwe.read_text(encoding="utf-8"))
assert "Observed (EPEX)" in text and "Realized DA prices and error reference" in text
assert "Today's realized prices are not validated" not in text
assert all(z + ": EPEX" in text for z in ("BE", "DE", "FR", "NL"))
assert published_files == 12
summary = {"run_id": run_id, "status": run["status"], "return_code": run["return_code"],
           "finished_at": run["finished_at"], "unchanged_frozen_files": len(before),
           "verified_published_files": published_files, "zones": zones, "report": str(cwe)}
(out / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
