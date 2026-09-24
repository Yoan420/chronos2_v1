"""Launch the existing NYX application pipeline once for the EPEX publication."""
import json
from pathlib import Path
import httpx

root = Path(__file__).resolve().parents[1]
with httpx.Client(base_url="http://127.0.0.1:8765", trust_env=False, timeout=60) as client:
    state = client.get("/api/primary-run").raise_for_status().json()
    if state.get("active"):
        raise RuntimeError("An application run is already active; no duplicate launched.")
    bootstrap = client.get("/api/bootstrap").raise_for_status().json()
    response = client.post("/api/primary-run", json={"delivery_day": "2026-09-19"}, headers={
        "X-Console-Token": bootstrap["token"], "Origin": "http://127.0.0.1:8765",
        "Idempotency-Key": "nyx-epex-reference-20260918-v1"}).raise_for_status().json()
    (root / "tmp/nyx_epex_reference_20260918/launch.json").write_text(
        json.dumps(response, indent=2), encoding="utf-8")
    print(json.dumps(response), flush=True)
