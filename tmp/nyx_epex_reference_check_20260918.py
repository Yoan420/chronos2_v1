"""Check live EPEX coverage without mutating model inputs or publications."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_nuclear_forecast as runner
from chronos2_hourly.nuclear_reporting_refresh import _support
from chronos2_hourly.reporting_observations import fetch_epex_reporting_observations
from chronos2_modular.saturn import create_saturn_client

out = ROOT / "tmp/nyx_epex_reference_20260918"
out.mkdir(exist_ok=True)
settings = runner.load_settings(ROOT / "config/nuclear_forecast.yaml")
hashes = {}
for path in (ROOT / "runs/experiments/nuclear_forecast_v1/2026-09-19").glob("*/civil_pit_v2/report_only/frozen_result/*"):
    if path.is_file():
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / "frozen_before.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
records = []
for zone in ("BE", "DE", "FR", "NL"):
    config, _, _ = runner.zone_inputs(settings, zone)
    spec = runner.build_zone_configs(config, [zone], None, None)[0]
    expected, current = _support("2026-09-19", spec.timezone)
    client = create_saturn_client(config["data"]["saturn_url"], config["data"]["saturn_author"])
    values, source = fetch_epex_reporting_observations(
        client, zone=zone, timezone=spec.timezone, expected_index=expected,
        extracted_at_utc=pd.Timestamp.now(tz="UTC"))
    values = values.reindex(expected)
    values.rename("actual").to_frame().to_parquet(out / (zone + ".parquet"))
    record = {"zone": zone, "source": source, "expected_hours": len(expected),
              "historical_missing": int(values.reindex(expected.difference(current)).isna().sum()),
              "current_available": int(values.reindex(current).notna().sum()),
              "all_finite": bool(np.isfinite(values.to_numpy(float)).all()),
              "day_min": float(values.reindex(current).min()), "day_max": float(values.reindex(current).max())}
    records.append(record)
    print(json.dumps(record), flush=True)
(out / "source_check.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
assert len(hashes) == 36
assert all(r["historical_missing"] == 0 and r["current_available"] == 24 for r in records)
