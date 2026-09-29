"""Audited JAO training-history bootstrap, separate from live capture proof.

Historical initial-computation downloads may train a later fit. Their actual
retrieval timestamps are retained and never presented as old daily captures.
The delivery-day partition must still pass the prospective capture verifier.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from datetime import date
import gzip
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from . import jao_flowbased as jao
from . import nyx_annual_jao_source as live
from .nyx_local_io import publish_bytes, promote_directory_retry
from .process_lock import exclusive_process_lock


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = ROOT / "data/pit/nyx_annual_jao_history"
DEFAULT_LIVE = ROOT / "data/pit/nyx_annual_jao_initial_live"
DEFAULT_LEGACY = ROOT / "data/pit/jao_core_flowbased"
HISTORY_POLICY = "current_fit_snapshot_v1"
LEDGER_PROTOCOL = "nyx_annual_jao_current_fit_history_v1"
require, utc, sha256 = live.require, live._utc, live.sha256


def _read_partition(root: Path, day: date) -> tuple[dict, pd.DataFrame, dict]:
    """Verify identities and hashes, then independently normalise archived RAW."""
    paths = live._paths(root, day)
    raw, audit_path, normalised_path, feature_path = paths
    require(all(path.is_file() for path in paths), f"Missing JAO training partition: {day}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit.get("schema_version") == jao.FLOWBASED_SCHEMA_VERSION
            and audit.get("delivery_day") == day.isoformat()
            and audit.get("publication_stage") in ("initial_computation", "initial_computation_fallback")
            and audit.get("tls_verification") is True
            and audit.get("fallback_reason") in (None, "empty_initial_publication", "api_last_modified_after_cutoff")
            and tuple(audit.get("feature_columns", ())) == jao.FLOWBASED_FEATURE_COLUMNS,
            f"JAO {day}: training archive identity/TLS/schema differs")
    hashes = {raw: audit.get("raw_gzip_sha256"),
              normalised_path: audit.get("normalised_sha256"),
              feature_path: audit.get("features_sha256")}
    require(all(isinstance(digest, str) and len(digest) == 64 and sha256(path) == digest
                for path, digest in hashes.items()), f"JAO {day}: training checksum differs")
    payload = json.loads(gzip.decompress(raw.read_bytes()))
    info, rows = payload.get("fetch"), payload.get("records")
    start, end = jao.local_day_utc_bounds(day)
    require(payload.get("schema_version") == jao.FLOWBASED_SCHEMA_VERSION
            and isinstance(info, dict) and isinstance(rows, list)
            and info.get("api_base_url") == jao.JAO_CORE_DATA_URL
            and info.get("http_method") == "GET"
            and info.get("endpoint") == "initialComputation"
            and info.get("filters") == {"Presolved": True}
            and info.get("to_utc_is_exclusive") is True
            and info.get("total_rows") == len(rows)
            and utc(info.get("start_utc"), "JAO start") == start
            and utc(info.get("end_utc"), "JAO end") == end
            and all(isinstance(row, dict) and row.get("presolved") is True for row in rows),
            f"JAO {day}: RAW response identity differs")
    retrieved = utc(info.get("retrieved_at_utc"), "JAO training retrieval")
    modified = (utc(info["last_modified_utc"], "JAO training watermark")
                if info.get("last_modified_utc") is not None else None)
    pages = tuple(utc(item, "JAO page watermark")
                  for item in info.get("page_last_modified_utc", []))
    audit_modified = (audit.get("raw_api_last_modified_utc")
                      if audit.get("publication_stage") == "initial_computation_fallback"
                      else audit.get("api_last_modified_utc"))
    audit_modified = utc(audit_modified, "JAO audit watermark") if audit_modified is not None else None
    valid_watermark = (not rows and modified is None and not pages) or (
        modified is not None and bool(pages) and max(pages) == modified and modified <= retrieved)
    require(valid_watermark
            and retrieved <= pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=1)
            and utc(audit.get("retrieved_at_utc"), "JAO audit retrieval") == retrieved
            and audit_modified == modified,
            f"JAO {day}: actual retrieval/watermark differs")
    fetched = jao.JaoFetchResult(
        endpoint="initialComputation", start_utc=start, end_utc=end, rows=tuple(rows),
        total_rows=len(rows), last_modified_utc=modified, retrieved_at_utc=retrieved,
        filters={"Presolved": True}, requests=int(info.get("requests", 1)),
        pages=int(info.get("pages", 1)), page_last_modified_utc=pages,
        snapshot_fingerprint_sha256=info.get("snapshot_fingerprint_sha256"),
        snapshot_verification_scans=int(info.get("snapshot_verification_scans", 1)))
    normalised, rebuilt_audit = jao.normalise_initial_computation(fetched, delivery_day=day)
    require(audit.get("mtu_minutes") == rebuilt_audit["mtu_minutes"]
            and int(audit.get("api_rows", 0)) == len(rows)
            and utc(audit.get("cutoff_time_utc"), "JAO audit cutoff") == jao.expected_cutoff_utc(day),
            f"JAO {day}: normalisation metadata differs")
    pd.testing.assert_frame_equal(pd.read_parquet(normalised_path), normalised,
                                  check_dtype=False, check_exact=True)
    item = {"delivery_day": day.isoformat(),
            "cutoff_utc": jao.expected_cutoff_utc(day).isoformat(),
            "retrieved_at_utc": retrieved.isoformat(),
            "last_modified_utc": modified.isoformat() if modified is not None else None,
            "rows": len(rows), "raw_sha256": hashes[raw], "audit_sha256": sha256(audit_path),
            "normalised_sha256": hashes[normalised_path],
            "legacy_features_sha256": hashes[feature_path]}
    require(all(sha256(path) == digest for path, digest in hashes.items())
            and sha256(audit_path) == item["audit_sha256"],
            f"JAO {day}: archive changed during reconstruction")
    return item, normalised, rebuilt_audit


def verify_training_partition(root: Path, day: date) -> dict:
    return _read_partition(Path(root), day)[0]


def _cache_partition(*, day: date, history_root: Path, candidates: tuple[Path, ...],
                     client: jao.JaoCoreClient, tls_trust_source: str) -> Path:
    """Commit a whole daily directory atomically; interrupted downloads are ignored."""
    destination = history_root / "partitions" / day.isoformat()
    if destination.exists():
        verify_training_partition(destination, day)
        return destination
    stage = history_root / "staging" / f"{day}.{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    copied = False
    for candidate in candidates:
        paths = live._paths(candidate, day)
        if not any(path.exists() for path in paths):
            continue
        if not all(path.is_file() for path in paths):
            print(f"JAO {day}: archive locale partielle ignoree; recuperation API", flush=True)
            continue
        # A corrupt complete cache is reported, never silently relabelled.
        verify_training_partition(candidate, day)
        for path in paths:
            live._copy_immutable(path, stage / path.relative_to(candidate))
        copied = True
        break
    if not copied:
        fetched = client.fetch_initial_day(day)
        start, end = jao.local_day_utc_bounds(day)
        require(fetched.endpoint == "initialComputation" and fetched.filters == {"Presolved": True}
                and fetched.start_utc == start and fetched.end_utc == end
                and fetched.total_rows == len(fetched.rows),
                f"JAO {day}: absent or wrong initial-computation history")
        normalised, audit = jao.normalise_initial_computation(fetched, delivery_day=day)
        if normalised.empty:
            # Archive absence faithfully. The annual builder keeps all values
            # unavailable; it never adopts the legacy previous-day fallback.
            grid = pd.date_range(start, end, freq="h", inclusive="left")
            features = pd.DataFrame(np.nan, index=grid, columns=jao.FLOWBASED_FEATURE_COLUMNS)
            features.index.name = "value_time_utc"
            features = features.reset_index()
        else:
            features = jao.build_hourly_flowbased_features(normalised, daily_audit=audit)
        jao.write_daily_flowbased_bundle(stage, delivery_day=day, fetch=fetched,
            normalised=normalised, features=features, audit=audit,
            tls_verification=True, tls_trust_source=tls_trust_source)
    verify_training_partition(stage, day)
    destination.parent.mkdir(parents=True, exist_ok=True)
    promote_directory_retry(stage, destination)
    return destination


def _features(partitions: dict[date, Path], index: pd.DatetimeIndex,
              delivery_day: date) -> tuple[pd.DataFrame, list[dict], dict]:
    output = pd.DataFrame(np.nan, index=index, columns=live.VALUE_COLUMNS)
    output[live.AVAILABLE] = np.int8(0)
    local_days = pd.Index(index.tz_convert("Europe/Paris").date)
    ledger, counts = [], Counter()
    for day, root in sorted(partitions.items()):
        item, normalised, audit = _read_partition(root, day)
        if day == delivery_day:
            require(live.verify_daily_capture(root, day) == item,
                    "JAO delivery snapshot differs from strict live capture")
        ledger.append(item)
        if normalised.empty:
            counts["empty_initial_publication"] += int((local_days == day).sum())
            continue
        mtu = int(audit["mtu_minutes"])
        require(mtu in (15, 60), "JAO unexpected MTU resolution")
        blocks = dict(tuple(normalised.groupby(normalised.delivery_start_utc.dt.floor("h"))))
        for hour in index[local_days == day]:
            block = blocks.get(hour)
            if block is None:
                counts["missing_raw_hour"] += 1
                continue
            expected = pd.date_range(hour, periods=60 // mtu, freq=f"{mtu}min")
            values, reason = live._aggregate_complete_hour(block, expected)
            counts[reason] += 1
            if values is not None:
                output.loc[hour, list(live.VALUE_COLUMNS)] = values
                output.loc[hour, live.AVAILABLE] = np.int8(1)
    return output, ledger, {"imputation": False, "reasons": dict(counts),
                           "available_hours": int(output[live.AVAILABLE].sum())}


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True, check_freq=False)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temp)
        from .nyx_local_io import replace_retry
        replace_retry(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def publish_history(day: str, bundle: Path, *, live_cache_root: Path = DEFAULT_LIVE,
                    history_cache_root: Path = DEFAULT_HISTORY,
                    legacy_cache_root: Path = DEFAULT_LEGACY,
                    client: jao.JaoCoreClient | None = None,
                    tls_trust_source: str = "verified_default") -> dict:
    """Bootstrap current-fit history without relaxing the target-day cutoff."""
    delivery_day, bundle = date.fromisoformat(day), Path(bundle).resolve()
    full, _, cutoff = live.delivery_grid(day)
    live_root, history_root = Path(live_cache_root).resolve(), Path(history_cache_root).resolve()
    require(not bundle.is_relative_to(history_root) and not bundle.is_relative_to(live_root),
            "JAO bundle must be outside source caches")
    receipt_path = bundle / "source_receipts/jao_initial.json"
    previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
    if previous is not None and previous.get("state") == "COMPLETE":
        require(previous.get("history_policy") == HISTORY_POLICY,
                "Completed JAO source has a different policy; select a new bundle")
        verified = verify_bundle_history(bundle, day, previous)
        return {"state": "COMPLETE", "receipt": str(receipt_path), "reused": True, **verified}
    # Refuse a missing/late delivery capture BEFORE historical network work.
    current = live.verify_daily_capture(live_root, delivery_day)
    days = sorted(set(full.tz_convert("Europe/Paris").date))
    require(days[-1] == delivery_day and len(days) >= 2, "JAO incomplete training grid")
    partitions = {delivery_day: live_root}
    with exclusive_process_lock(history_root / "jao_history.lock"):
        context = (nullcontext(client) if client is not None else
                   jao.JaoCoreClient(request_interval_seconds=.65, maximum_retries=4))
        with context as collector:
            for number, origin in enumerate(days[:-1], 1):
                partitions[origin] = _cache_partition(day=origin, history_root=history_root,
                    candidates=(live_root, Path(legacy_cache_root).resolve()), client=collector,
                    tls_trust_source=tls_trust_source)
                if number == 1 or number % 10 == 0 or number == len(days) - 1:
                    print(f"JAO historique: {number}/{len(days)-1} jours verifies", flush=True)
        frame, captures, feature_audit = _features(partitions, full, delivery_day)
    latest = max(utc(item["retrieved_at_utc"], "JAO training retrieval") for item in captures[:-1])
    asof_verified = latest <= cutoff
    artifacts = {}
    prefix = live.SOURCE_SUBDIR
    for origin, source_root in sorted(partitions.items()):
        for source in live._paths(source_root, origin):
            relative = f"{prefix}/captures/{source.relative_to(source_root).as_posix()}"
            artifacts[relative] = live._copy_immutable(source, bundle / relative)
    # Both model assemblies use the same explicitly identified current-fit
    # initial-domain snapshot. These names do not claim separate old vintages.
    for name in (live.FEATURE_NAME, "original_features.parquet", "refreshed_features.parquet"):
        relative = f"{prefix}/{name}"
        _write_frame(bundle / relative, frame)
        artifacts[relative] = sha256(bundle / relative)
    ledger = {"protocol": LEDGER_PROTOCOL, "history_policy": HISTORY_POLICY,
        "delivery_day": day, "captured_days": len(captures), "required_days": len(days),
        "missing_days": [], "captures": captures, "feature_audit": feature_audit,
        "history_available_at_fit_cutoff": bool(asof_verified),
        "training_snapshot_max_retrieved_at_utc": latest.isoformat(),
        "origin_snapshot_capture_verified": False,
        "delivery_snapshot_pre_cutoff_verified": True,
        "original_refreshed_views": "same_current_fit_initial_computation_snapshot"}
    relative = f"{prefix}/{live.LEDGER_NAME}"
    publish_bytes(bundle / relative, live._json_bytes(ledger))
    artifacts[relative] = sha256(bundle / relative)
    receipt = {"protocol": live.SOURCE_PROTOCOL, "source_group": live.SOURCE_GROUP,
        "delivery_day": day, "state": "COMPLETE", "history_policy": HISTORY_POLICY,
        "asof_cutoff_verified": bool(asof_verified), "training_window_complete": True,
        "asof_state_utc": current["retrieved_at_utc"], "cutoff_utc": cutoff.isoformat(),
        "training_snapshot_max_retrieved_at_utc": latest.isoformat(),
        "delivery_snapshot_pre_cutoff_verified": True,
        "origin_snapshot_capture_verified": False, "actual_pre_cutoff_capture_verified": False,
        "provider_publication_timestamp_verified": False,
        "provider_last_modified_utc": current["last_modified_utc"],
        "captured_training_and_delivery_days": len(days),
        "required_training_and_delivery_days": len(days), "missing_days": [],
        "availability_basis": "Historical initial snapshots at real retrieval times; strict pre-cutoff delivery capture",
        "artifact_sha256": artifacts, "model_inputs_complete": False,
        "limitations": ["Historical retrieval is not proof of availability at old backtest origins.",
                        "History first downloaded after this fit cutoff is preparation-only for this delivery day.",
                        "The new training policy requires a separate complete model evaluation."]}
    # Validate the portable packet, not just the original mutable cache, before
    # committing COMPLETE. An interrupted predecessor stays incomplete.
    live.validate_source_receipt(receipt, group=live.SOURCE_GROUP, day=day,
                                 bundle=bundle, cutoff=cutoff, allow_training_bootstrap=True)
    verify_bundle_history(bundle, day, receipt)
    # Receipt is the commit marker; incomplete predecessors may be replaced.
    publish_bytes(receipt_path, live._json_bytes(receipt))
    return {"state": "COMPLETE", "receipt": str(receipt_path), "captured_days": len(days),
            "missing_days": 0, "asof_cutoff_verified": bool(asof_verified),
            "history_policy": HISTORY_POLICY, "feature_artifact": str(bundle / prefix / live.FEATURE_NAME)}


def verify_bundle_history(bundle: Path, day: str, receipt: dict) -> dict:
    """Reconstruct all hourly descriptors from the portable hashed raw archive."""
    bundle, delivery_day = Path(bundle).resolve(), date.fromisoformat(day)
    full, _, cutoff = live.delivery_grid(day)
    require(receipt.get("protocol") == live.SOURCE_PROTOCOL
            and receipt.get("source_group") == live.SOURCE_GROUP
            and receipt.get("delivery_day") == day and receipt.get("state") == "COMPLETE"
            and receipt.get("history_policy") == HISTORY_POLICY,
            "JAO training receipt identity differs")
    artifacts = receipt.get("artifact_sha256", {})
    prefix, captures_root = live.SOURCE_SUBDIR, bundle / live.SOURCE_SUBDIR / "captures"
    days = sorted(set(full.tz_convert("Europe/Paris").date))
    expected = {f"{prefix}/{name}" for name in (live.LEDGER_NAME, live.FEATURE_NAME,
                                               "original_features.parquet", "refreshed_features.parquet")}
    for origin in days:
        expected.update(path.relative_to(bundle).as_posix() for path in live._paths(captures_root, origin))
    require(set(artifacts) == expected, "JAO portable archive inventory differs")
    for relative, digest in artifacts.items():
        path = bundle / relative
        require(path.resolve().is_relative_to(bundle) and path.is_file() and sha256(path) == digest,
                f"JAO bound artifact checksum differs: {relative}")
    computed, captures, audit = _features({origin: captures_root for origin in days}, full, delivery_day)
    ledger = json.loads((bundle / prefix / live.LEDGER_NAME).read_text(encoding="utf-8"))
    latest = max(utc(item["retrieved_at_utc"], "JAO history retrieval") for item in captures[:-1])
    verified = latest <= cutoff
    require(ledger.get("protocol") == LEDGER_PROTOCOL and ledger.get("history_policy") == HISTORY_POLICY
            and ledger.get("delivery_day") == day and ledger.get("captures") == captures
            and ledger.get("feature_audit") == audit and ledger.get("missing_days") == []
            and ledger.get("captured_days") == len(days) and ledger.get("required_days") == len(days)
            and ledger.get("original_refreshed_views") == "same_current_fit_initial_computation_snapshot"
            and ledger.get("training_snapshot_max_retrieved_at_utc") == latest.isoformat()
            and ledger.get("history_available_at_fit_cutoff") is bool(verified)
            and ledger.get("origin_snapshot_capture_verified") is False
            and ledger.get("delivery_snapshot_pre_cutoff_verified") is True,
            "JAO training ledger differs from RAW reconstruction")
    require(receipt.get("asof_cutoff_verified") is bool(verified)
            and receipt.get("training_window_complete") is True
            and receipt.get("delivery_snapshot_pre_cutoff_verified") is True
            and receipt.get("origin_snapshot_capture_verified") is False
            and receipt.get("actual_pre_cutoff_capture_verified") is False
            and receipt.get("provider_publication_timestamp_verified") is False
            and receipt.get("provider_last_modified_utc") == captures[-1]["last_modified_utc"]
            and receipt.get("training_snapshot_max_retrieved_at_utc") == latest.isoformat()
            and receipt.get("asof_state_utc") == captures[-1]["retrieved_at_utc"]
            and receipt.get("cutoff_utc") == cutoff.isoformat()
            and receipt.get("captured_training_and_delivery_days") == len(days)
            and receipt.get("required_training_and_delivery_days") == len(days)
            and receipt.get("missing_days") == [], "JAO training receipt causality differs")
    for name in (live.FEATURE_NAME, "original_features.parquet", "refreshed_features.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(bundle / prefix / name), computed,
                                      check_exact=True, check_freq=False)
    for relative, digest in artifacts.items():
        require(sha256(bundle / relative) == digest, "JAO archive changed during verification")
    return {"passed": True, "daily_raw_captures_recomputed": len(days), "history_policy": HISTORY_POLICY,
            "asof_cutoff_verified": bool(verified), "origin_snapshot_capture_verified": False,
            "delivery_snapshot_pre_cutoff_verified": True,
            "provider_first_publication_certified": False}
