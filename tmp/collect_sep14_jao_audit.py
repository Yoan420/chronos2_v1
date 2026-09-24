"""Bounded, TLS-verified public JAO forensic capture; never a forecast feature store.

All four endpoints are deliberately labelled audit-only here. Even the initial
domain is downloaded after the event: only its publication watermark can support
a historical availability claim, and not a reconstruction of missing vintages.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import pandas as pd

from chronos2_hourly.jao_flowbased import (
    JAO_CORE_DATA_URL,
    JaoCoreClient,
    build_windows_trust_context,
    expected_cutoff_utc,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: object) -> str:
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                     default=str).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(raw)
    return sha256(raw)


def main() -> int:
    target = (ROOT / "runs/experiments/nyx_physical_p50_v1/forensics/2026-09-14").resolve()
    target.relative_to(ROOT / "runs/experiments/nyx_physical_p50_v1")
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    output = target / (stamp + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    wire = output / "raw_http"
    wire.mkdir()
    captures = []

    def capture(response: httpx.Response) -> None:
        # The raw JSON body bytes are retained without JSON normalization.
        body = response.read()
        name = f"{len(captures):03d}_{response.request.url.path.rsplit('/', 1)[-1]}.json"
        with (wire / name).open("xb") as handle:
            handle.write(body)
        captures.append({
            "path": str((wire / name).relative_to(output)),
            "sha256": sha256(body), "bytes": len(body),
            "url": str(response.request.url),
            "status_code": response.status_code,
            "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "response_headers": {key: response.headers[key] for key in
                                 ("date", "etag", "last-modified", "content-type")
                                 if key in response.headers},
        })

    context, roots = build_windows_trust_context()
    results = {}
    errors = {}
    start, end = "2026-09-14T16:00:00Z", "2026-09-14T19:00:00Z"
    endpoints = ["initialComputation", "finalComputation", "shadowPrices", "netPos"]
    with httpx.Client(verify=context, timeout=45, follow_redirects=True,
                      headers={"User-Agent": "NYX-readonly-spike-forensics/1"},
                      event_hooks={"response": [capture]}) as transport:
        with JaoCoreClient(client=transport, maximum_retries=1,
                           request_interval_seconds=.75) as client:
            for endpoint in endpoints:
                try:
                    if endpoint in ("shadowPrices", "netPos"):
                        # These audit endpoints return an unpaginated list and
                        # intentionally do not echo Skip/Take like computation.
                        response = transport.get(f"{JAO_CORE_DATA_URL}/{endpoint}",
                                                 params={"FromUtc": start, "ToUtc": end})
                        response.raise_for_status()
                        payload = response.json()
                        rows = payload.get("data")
                        declared = payload.get("totalRowsWithFilter", payload.get("totalRows"))
                        if payload.get("rejected") or not isinstance(rows, list) or (declared is not None and declared != len(rows)):
                            raise ValueError("Unpaginated audit endpoint count or payload invalid.")
                        times = pd.to_datetime([row["dateTimeUtc"] for row in rows], utc=True)
                        if not ((times >= pd.Timestamp(start)) & (times < pd.Timestamp(end))).all():
                            raise ValueError("Audit endpoint returned rows outside exact requested interval.")
                        if declared is None:
                            expected = pd.date_range(start, end, freq="15min", inclusive="left")
                            if endpoint != "netPos" or not times.sort_values().equals(expected):
                                raise ValueError("Missing declared count requires exact complete net-position MTU grid.")
                        rows_name = f"{endpoint}_rows.json"
                        rows_sha = save_json(output / rows_name, rows)
                        audit = {"endpoint": endpoint, "start_utc": start, "end_utc": end,
                                 "last_modified_utc": payload.get("lastModifiedOn"),
                                 "retrieved_at_utc": captures[-1]["retrieved_at_utc"],
                                 "total_rows": len(rows), "rows_file": rows_name,
                                 "rows_sha256": rows_sha, "audit_only": True,
                                 "allowed_as_model_input": False, "post_coupling": True,
                                 "available_after_08_cutoff_by_design": True,
                                 "pagination": "unpaginated_declared_count_verified" if declared is not None
                                 else "unpaginated_complete_15min_grid_verified"}
                        results[endpoint] = audit
                        print(json.dumps({"endpoint": endpoint, "rows": len(rows),
                                          "watermark": audit["last_modified_utc"],
                                          "keys": sorted({k for row in rows for k in row}),
                                          "first": rows[0] if rows else None}, default=str), flush=True)
                        continue
                    fetched = client.fetch(endpoint, start_utc=start, end_utc=end,
                                           filters={"Presolved": True} if endpoint in
                                           ("initialComputation", "finalComputation") else None)
                    rows_name = f"{endpoint}_rows.json"
                    rows_sha = save_json(output / rows_name, list(fetched.rows))
                    audit = fetched.audit_dict()
                    audit.update({"rows_file": rows_name, "rows_sha256": rows_sha,
                                  "audit_only": True, "allowed_as_model_input": False,
                                  "post_coupling": endpoint in ("shadowPrices", "netPos"),
                                  "available_after_08_cutoff_by_design": endpoint != "initialComputation"})
                    results[endpoint] = audit
                    keys = sorted({key for row in fetched.rows for key in row})
                    print(json.dumps({"endpoint": endpoint, "rows": len(fetched.rows),
                                      "watermark": audit["last_modified_utc"],
                                      "keys": keys, "first": fetched.rows[0] if fetched.rows else None},
                                     ensure_ascii=False, default=str), flush=True)
                except Exception as exc:
                    errors[endpoint] = {"type": type(exc).__name__, "message": str(exc)}
                    print(f"{endpoint}: {type(exc).__name__}: {exc}", flush=True)
    manifest = {
        "schema_version": 1, "purpose": "ex_post_spike_forensics_only",
        "delivery_day": "2026-09-14", "interval_start_utc": start,
        "interval_end_utc_exclusive": end, "local_hours": "18:00, 19:00, 20:00 Europe/Paris",
        "historical_forecast_cutoff_utc": expected_cutoff_utc("2026-09-14").isoformat(),
        "api_base_url": JAO_CORE_DATA_URL, "tls_verification": True,
        "windows_root_certificates_loaded": roots,
        "collector_sha256": sha256(Path(__file__).read_bytes()),
        "jao_client_sha256": sha256((ROOT / "chronos2_hourly/jao_flowbased.py").read_bytes()),
        "allowed_as_model_input": False, "production_modified": False,
        "limitations": [
            "Post-event retrieval: not a captured pre-08 vintage.",
            "Final domain and post-coupling prices/net positions forbidden as 08h inputs.",
            "Shadow-price times PTDF descriptors are diagnostics, not proof of a specific plant cause.",
            "Current field conventions, MTU duration and all binding constraints must be verified before spread attribution.",
        ],
        "results": results, "errors": errors, "raw_http": captures,
    }
    save_json(output / "forensic_manifest.json", manifest)
    print(json.dumps({"output": str(output), "complete": not errors,
                      "endpoints": len(results), "errors": errors}, ensure_ascii=False), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
