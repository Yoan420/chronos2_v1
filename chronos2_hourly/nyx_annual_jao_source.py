"""Prospective JAO Initial Computation capture for the annual CWE input bundle.

The public JAO API does not expose a historical snapshot-as-of query.  A
``lastModifiedOn`` value from a later download is therefore insufficient:
every adopted daily partition must have been captured before that delivery
day's own D-1 08:00 Europe/Paris cutoff.  No historical research cache is
silently adopted and no final/post-coupling publication is substituted.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from datetime import date, timedelta
import gzip
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from . import jao_flowbased as jao
from .nyx_annual_live_preflight import (
    SOURCE_PROTOCOL,
    delivery_grid,
    sha256,
    validate_source_receipt,
)


SOURCE_GROUP = "jao_initial"
SOURCE_SUBDIR = "source_artifacts/jao_initial"
SOURCE_HANDBOOK = "https://publicationtool.jao.eu/core/CORE_PublicationHandbook"
LEDGER_PROTOCOL = "nyx_annual_jao_prospective_ledger_v1"
FEATURE_NAME = "features_365d_plus_delivery.parquet"
LEDGER_NAME = "snapshot_ledger.json"
VALUE_COLUMNS = tuple("extra_jao_" + name.removeprefix("flowbased_")
                      for name in jao.FLOWBASED_FEATURE_COLUMNS)
AVAILABLE = "extra_jao__available"
FEATURE_COLUMNS = (*VALUE_COLUMNS, AVAILABLE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc(value: object, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    require(not pd.isna(stamp) and stamp.tzinfo is not None,
            f"{label}: UTC-offset timestamp required")
    return stamp.tz_convert("UTC")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2,
                       sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


@contextmanager
def exclusive_cache_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "jao_live_capture.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError(f"Another JAO live capture holds {lock}") from error
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()) + "\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def _paths(root: Path, day: date) -> tuple[Path, Path, Path, Path]:
    token = day.isoformat()
    return (
        root / "raw/initialComputation" / f"{token}.json.gz",
        root / "raw/initialComputation" / f"{token}.audit.json",
        root / "normalised" / f"{token}.parquet",
        root / "daily_features" / f"{token}.parquet",
    )


def _check_fetch(fetch: jao.JaoFetchResult, day: date) -> dict:
    start, end = jao.local_day_utc_bounds(day)
    cutoff = jao.expected_cutoff_utc(day)
    require(fetch.endpoint == "initialComputation"
            and fetch.filters == {"Presolved": True}
            and fetch.start_utc == start and fetch.end_utc == end,
            "JAO live response has wrong endpoint, filter, or civil-day bounds")
    require(fetch.total_rows == len(fetch.rows) and fetch.total_rows > 0,
            "JAO Initial Computation is absent or empty before cutoff")
    retrieved = _utc(fetch.retrieved_at_utc, "JAO capture time")
    modified = _utc(fetch.last_modified_utc, "JAO lastModifiedOn")
    require(modified <= retrieved <= cutoff,
            "JAO initial publication/capture is after D-1 08:00")
    pages = [_utc(value, "JAO page lastModifiedOn")
             for value in fetch.page_last_modified_utc]
    require(pages and max(pages) == modified,
            "JAO page watermarks do not match the final watermark")
    require(all(row.get("presolved") is True for row in fetch.rows),
            "JAO returned an unpresolved row")
    normalised, audit = jao.normalise_initial_computation(fetch, delivery_day=day)
    require(not normalised.empty and audit.get("operational_pit_eligible") is True
            and audit.get("pit_status") == "api_last_modified_pre_cutoff",
            "JAO response is not an actual pre-cutoff initial capture")
    return {"normalised": normalised, "audit": audit,
            "retrieved_at_utc": retrieved.isoformat(),
            "last_modified_utc": modified.isoformat(),
            "cutoff_utc": cutoff.isoformat()}


def capture_day(*, day: date, cache_root: Path, client: jao.JaoCoreClient,
                now_utc: pd.Timestamp | None = None,
                tls_trust_source: str = "verified_default") -> dict:
    """Capture one day only during its actual D-1 publication window.

    A matching immutable cache is reused after the cutoff; a fresh API fetch
    after cutoff is forbidden even if the API reports an old lastModifiedOn.
    """
    cache_root = Path(cache_root).resolve()
    paths = _paths(cache_root, day)
    exists = [path.exists() for path in paths]
    if any(exists):
        require(all(exists), f"Partial JAO prospective capture for {day}")
        return verify_daily_capture(cache_root, day)
    now = _utc(now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC"),
               "Current time")
    cutoff = jao.expected_cutoff_utc(day)
    scheduled = jao.expected_initial_publication_utc(day)
    require(scheduled <= now <= cutoff,
            "Fresh JAO capture requires the D-1 01:15–08:00 local window")
    fetch = client.fetch_initial_day(day)
    verified = _check_fetch(fetch, day)
    # The existing writer archives the actual API result and its audit.  Its
    # imputed flowbased sidecar is retained only as legacy audit material; the
    # annual model source below uses the strict no-imputation RAW builder.
    features = jao.build_hourly_flowbased_features(
        verified["normalised"], daily_audit=verified["audit"])
    metadata = jao.write_daily_flowbased_bundle(
        cache_root, delivery_day=day, fetch=fetch,
        normalised=verified["normalised"], features=features,
        audit=verified["audit"], tls_verification=True,
        tls_trust_source=tls_trust_source)
    require(metadata.get("operational_pit_eligible") is True,
            "JAO archive lost pre-cutoff qualification")
    return verify_daily_capture(cache_root, day)


def verify_daily_capture(cache_root: Path, day: date) -> dict:
    """Verify archived RAW, actual capture time, watermark, and source hashes."""
    root = Path(cache_root).resolve()
    raw, audit_path, normalised, features = _paths(root, day)
    require(all(path.is_file() for path in (raw, audit_path, normalised, features)),
            f"Missing prospective JAO partition for {day}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cutoff = jao.expected_cutoff_utc(day)
    require(audit.get("schema_version") == jao.FLOWBASED_SCHEMA_VERSION
            and audit.get("delivery_day") == day.isoformat()
            and audit.get("publication_stage") == "initial_computation"
            and audit.get("pit_status") == "api_last_modified_pre_cutoff"
            and audit.get("pit_eligible") is True
            and audit.get("operational_pit_eligible") is True
            and audit.get("tls_verification") is True
            and audit.get("fallback_reason") is None
            and tuple(audit.get("feature_columns", ())) == jao.FLOWBASED_FEATURE_COLUMNS,
            f"JAO {day}: audit is not a direct verified initial capture")
    require(_utc(audit.get("cutoff_time_utc"), "JAO audit cutoff") == cutoff,
            f"JAO {day}: audit cutoff differs")
    retrieved = _utc(audit.get("retrieved_at_utc"), "JAO audit capture")
    modified = _utc(audit.get("api_last_modified_utc"), "JAO audit watermark")
    require(modified <= retrieved <= cutoff,
            f"JAO {day}: archive was retrieved or modified after cutoff")
    expected = {
        raw: audit.get("raw_gzip_sha256"),
        normalised: audit.get("normalised_sha256"),
        features: audit.get("features_sha256"),
    }
    require(all(isinstance(digest, str) and len(digest) == 64
                and sha256(path) == digest for path, digest in expected.items()),
            f"JAO {day}: archived partition checksum differs")
    return {
        "delivery_day": day.isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "retrieved_at_utc": retrieved.isoformat(),
        "last_modified_utc": modified.isoformat(),
        "rows": int(audit.get("api_rows", 0)),
        "raw_sha256": expected[raw],
        "audit_sha256": sha256(audit_path),
        "normalised_sha256": expected[normalised],
        "legacy_features_sha256": expected[features],
    }


def inspect_history(cache_root: Path, day: date) -> tuple[list[dict], list[str]]:
    """Require 366 individually captured delivery days; never backfill now."""
    present = []
    missing = []
    for stamp in pd.date_range(day - timedelta(days=365), day, freq="D"):
        candidate = stamp.date()
        paths = _paths(Path(cache_root).resolve(), candidate)
        if not any(path.exists() for path in paths):
            missing.append(candidate.isoformat())
        else:
            present.append(verify_daily_capture(cache_root, candidate))
    return present, missing


def _aggregate_complete_hour(block: pd.DataFrame,
                             expected_mtus: pd.DatetimeIndex,
                             ) -> tuple[np.ndarray | None, str]:
    """Reproduce the 27 annual RAW JAO descriptors without filling gaps."""
    times = pd.DatetimeIndex(block.delivery_start_utc)
    cnec = block.loc[block.cnec.astype(bool)]
    ctimes = pd.DatetimeIndex(cnec.delivery_start_utc).unique().sort_values()
    if not ctimes.equals(expected_mtus):
        return None, "missing_or_partial_cnec_mtu"
    required = ["ram", "fmax", *("ptdf_" + zone for zone in jao.CORE_PTDF_ZONES)]
    values = cnec[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (cnec.fmax.abs() <= 1e-9).any():
        return None, "invalid_raw_numeric_support"
    ram = cnec.ram.astype(float)
    safe_ram = ram.abs().clip(lower=100.)
    ptdf = {zone: cnec["ptdf_" + zone].astype(float)
            for zone in jao.CORE_PTDF_ZONES}
    spreads = {zone: (ptdf["FR"] - ptdf[zone]).abs()
               for zone in ("DE", "BE", "NL")}
    neighbor = pd.concat(list(spreads.values()), axis=1).max(axis=1)
    alegro = (ptdf["ALBE"] - ptdf["ALDE"]).abs()
    ptdf_matrix = np.column_stack([ptdf[zone].to_numpy() for zone in ptdf])
    core_range = pd.Series(np.nanmax(ptdf_matrix, axis=1)
                           - np.nanmin(ptdf_matrix, axis=1), index=cnec.index)
    stress = neighbor / safe_ram * 1000.
    hhis = []
    for stamp in ctimes:
        weights = stress.loc[pd.DatetimeIndex(cnec.delivery_start_utc) == stamp].abs().to_numpy()
        total = float(weights.sum())
        hhis.append(float(np.square(weights / total).sum()) if total > 0 else 0.)
    external = block.loc[block.constraint_kind.eq("external"), "ram"].astype(float)
    external = external.loc[external >= 0.]
    n_mtu = times.nunique()
    quantile = lambda vector, probability: float(pd.Series(vector).quantile(probability))
    descriptor = {
        "flowbased_cnec_mtu_availability": 1.,
        "flowbased_missing_mtu_share": 0.,
        "flowbased_hour_imputed": 0.,
        "flowbased_cnec_count": float(len(cnec) / len(ctimes)),
        "flowbased_external_constraint_count": float(block.constraint_kind.eq("external").sum() / n_mtu),
        "flowbased_equality_constraint_count": float(block.constraint_kind.eq("equality").sum() / n_mtu),
        # A genuine absence of external constraints has value zero in the
        # original descriptor definition; it does not fill missing hours.
        "flowbased_external_ram_min_mw": float(external.min()) if len(external) else 0.,
        "flowbased_external_ram_p10_mw": quantile(external, .1) if len(external) else 0.,
        "flowbased_ram_min_mw": float(ram.min()),
        "flowbased_ram_p05_mw": quantile(ram, .05),
        "flowbased_ram_p10_mw": quantile(ram, .1),
        "flowbased_ram_median_mw": quantile(ram, .5),
        "flowbased_ram_iqr_mw": quantile(ram, .75) - quantile(ram, .25),
        "flowbased_ram_below_500_share": float((ram < 500).mean()),
        "flowbased_ram_below_1000_share": float((ram < 1000).mean()),
        "flowbased_ram_to_fmax_p05": quantile(ram / cnec.fmax, .05),
        "flowbased_ram_to_fmax_median": quantile(ram / cnec.fmax, .5),
        **{"flowbased_fr_" + zone.lower() + "_ptdf_spread_p90": quantile(spreads[zone], .9)
           for zone in spreads},
        "flowbased_fr_neighbor_ptdf_spread_p90": quantile(neighbor, .9),
        "flowbased_alegro_ptdf_spread_p90": quantile(alegro, .9),
        "flowbased_core_ptdf_range_p90": quantile(core_range, .9),
        "flowbased_fr_neighbor_ram_stress_p95_per_gw": quantile(stress, .95),
        "flowbased_alegro_ram_stress_p95_per_gw": quantile(alegro / safe_ram * 1000., .95),
        "flowbased_core_ram_stress_p95_per_gw": quantile(core_range / safe_ram * 1000., .95),
        "flowbased_stress_hhi": float(np.mean(hhis)),
    }
    result = np.array([descriptor[column]
                       for column in jao.FLOWBASED_FEATURE_COLUMNS], dtype=float)
    require(np.isfinite(result).all(), "JAO RAW descriptor is non-finite")
    return result, "available"


def build_strict_history_features(cache_root: Path,
                                  delivery_index: pd.DatetimeIndex,
                                  ) -> tuple[pd.DataFrame, dict]:
    """Read prospective RAW archives and derive exact annual JAO columns."""
    index = delivery_index
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
            and len(index) > 0 and index.is_unique and index.is_monotonic_increasing
            and not index.hasnans and index.equals(index.floor("h")),
            "JAO feature index must be ordered UTC physical hours")
    root = Path(cache_root).resolve()
    output = pd.DataFrame(np.nan, index=index, columns=VALUE_COLUMNS)
    output[AVAILABLE] = np.int8(0)
    local_days = pd.Index(index.tz_convert("Europe/Paris").date)
    counts = Counter()
    files = {}
    for day in sorted(set(local_days)):
        verified = verify_daily_capture(root, day)
        raw_path, audit_path, _, _ = _paths(root, day)
        raw_bytes = raw_path.read_bytes()
        files[str(raw_path)] = verified["raw_sha256"]
        files[str(audit_path)] = verified["audit_sha256"]
        payload = json.loads(gzip.decompress(raw_bytes))
        fetch_info = payload.get("fetch")
        rows = payload.get("records")
        start, end = jao.local_day_utc_bounds(day)
        require(payload.get("schema_version") == jao.FLOWBASED_SCHEMA_VERSION
                and isinstance(fetch_info, dict) and isinstance(rows, list)
                and fetch_info.get("api_base_url") == jao.JAO_CORE_DATA_URL
                and fetch_info.get("endpoint") == "initialComputation"
                and fetch_info.get("filters") == {"Presolved": True}
                and fetch_info.get("to_utc_is_exclusive") is True
                and fetch_info.get("total_rows") == len(rows)
                and _utc(fetch_info.get("start_utc"), "JAO raw start") == start
                and _utc(fetch_info.get("end_utc"), "JAO raw end") == end,
                f"JAO {day}: archived RAW response identity differs")
        retrieved = _utc(fetch_info.get("retrieved_at_utc"), "JAO RAW capture")
        modified = _utc(fetch_info.get("last_modified_utc"), "JAO RAW watermark")
        require(retrieved.isoformat() == verified["retrieved_at_utc"]
                and modified.isoformat() == verified["last_modified_utc"],
                f"JAO {day}: RAW and audit timestamps differ")
        pages = tuple(_utc(item, "JAO RAW page watermark")
                      for item in fetch_info.get("page_last_modified_utc", []))
        require(pages and max(pages) == modified,
                f"JAO {day}: RAW page watermark differs")
        result = jao.JaoFetchResult(
            endpoint="initialComputation", start_utc=start, end_utc=end,
            rows=tuple(rows), total_rows=len(rows), last_modified_utc=modified,
            retrieved_at_utc=retrieved, filters={"Presolved": True},
            requests=int(fetch_info.get("requests", 1)),
            page_last_modified_utc=pages)
        normalised, raw_audit = jao.normalise_initial_computation(
            result, delivery_day=day)
        require(raw_audit.get("operational_pit_eligible") is True,
                f"JAO {day}: RAW loses actual pre-cutoff eligibility")
        mtu_minutes = int(raw_audit["mtu_minutes"])
        stored_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        require(stored_audit.get("mtu_minutes") == mtu_minutes,
                f"JAO {day}: RAW and audit MTU resolution differ")
        blocks = {hour: block for hour, block in
                  normalised.groupby(normalised.delivery_start_utc.dt.floor("h"))}
        requested = index[local_days == day]
        for hour in requested:
            block = blocks.get(hour)
            if block is None:
                counts["missing_raw_hour"] += 1
                continue
            expected_mtus = pd.date_range(hour, periods=60 // mtu_minutes,
                                          freq=f"{mtu_minutes}min")
            values, reason = _aggregate_complete_hour(block, expected_mtus)
            counts[reason] += 1
            if values is not None:
                output.loc[hour, list(VALUE_COLUMNS)] = values
                output.loc[hour, AVAILABLE] = np.int8(1)
    for name, digest in files.items():
        require(sha256(Path(name)) == digest,
                f"JAO archive changed while building descriptors: {name}")
    return output, {
        "available_hours": int(output[AVAILABLE].sum()),
        "reasons": dict(counts),
        "imputation": False,
        "historical_fallback": False,
        "source_sha256": files,
    }


def _copy_immutable(source: Path, destination: Path) -> str:
    digest = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(sha256(destination) == digest,
                f"Existing JAO bundle artifact differs: {destination}")
        return digest
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        require(sha256(temporary) == digest and sha256(source) == digest,
                "JAO source changed during copy")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def publish_jao_receipt(*, day: date, cache_root: Path, bundle: Path) -> dict:
    """Publish a preflight-compatible receipt only for full live lineage."""
    cache_root, bundle = Path(cache_root).resolve(), Path(bundle).resolve()
    full, current, cutoff = delivery_grid(day.isoformat())
    captured, missing = inspect_history(cache_root, day)
    require(any(item["delivery_day"] == day.isoformat() for item in captured),
            "Current JAO day has no direct pre-cutoff capture")
    current_capture = next(item for item in captured
                           if item["delivery_day"] == day.isoformat())
    ledger = {
        "protocol": LEDGER_PROTOCOL,
        "delivery_day": day.isoformat(),
        "model_source": "JAO Core initialComputation; Presolved=true",
        "handbook_url": SOURCE_HANDBOOK,
        "expected_initial_publication_local": "D-1 01:15 Europe/Paris",
        "actual_capture_required_by_local": "D-1 08:00 Europe/Paris",
        "captured_days": len(captured),
        "required_days": 366,
        "missing_days": missing,
        "captures": captured,
        "historical_api_backfill_adopted": False,
        "last_modified_alone_used_as_publication_proof": False,
    }
    artifacts = {}
    ledger_relative = f"{SOURCE_SUBDIR}/{LEDGER_NAME}"
    ledger_path = bundle / ledger_relative
    complete = not missing and len(captured) == 366
    receipt_path = bundle / "source_receipts/jao_initial.json"
    previous = (json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt_path.exists() else None)
    ledger_bytes = _json_bytes(ledger)
    if previous is not None and previous.get("state") == "COMPLETE":
        require(complete and ledger_path.is_file()
                and ledger_path.read_bytes() == ledger_bytes,
                "Completed JAO source ledger is immutable")
    else:
        _atomic_write(ledger_path, ledger_bytes)
    artifacts[ledger_relative] = sha256(ledger_path)

    if complete:
        frame, audit = build_strict_history_features(cache_root, full)
        require(frame.index.equals(full)
                and list(frame.columns) == list(FEATURE_COLUMNS)
                and len(current) in (23, 24, 25)
                and not np.isinf(frame.to_numpy(dtype=float)).any()
                and audit.get("historical_fallback") is False
                and audit.get("imputation") is False,
                "JAO full RAW feature build failed")
        feature_relative = f"{SOURCE_SUBDIR}/{FEATURE_NAME}"
        feature_path = bundle / feature_relative
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = feature_path.with_name(f".{feature_path.stem}.{uuid.uuid4().hex}.parquet")
        try:
            frame.to_parquet(temporary)
            if feature_path.exists():
                require(sha256(feature_path) == sha256(temporary),
                        "Existing JAO feature artifact differs")
            else:
                os.replace(temporary, feature_path)
        finally:
            temporary.unlink(missing_ok=True)
        artifacts[feature_relative] = sha256(feature_path)
        current_raw, current_audit, _, _ = _paths(cache_root, day)
        for source in (current_raw, current_audit):
            relative = f"{SOURCE_SUBDIR}/current/{source.name}"
            artifacts[relative] = _copy_immutable(source, bundle / relative)

    receipt = {
        "protocol": SOURCE_PROTOCOL,
        "source_group": SOURCE_GROUP,
        "delivery_day": day.isoformat(),
        "state": "COMPLETE" if complete else "INCOMPLETE",
        "asof_cutoff_verified": bool(complete),
        "training_window_complete": bool(complete),
        "asof_state_utc": current_capture["retrieved_at_utc"],
        "cutoff_utc": cutoff.isoformat(),
        "availability_basis": "Actual TLS-verified JAO initial API capture before each day-specific D-1 08:00",
        "actual_pre_cutoff_capture_verified": True,
        "provider_publication_timestamp_verified": False,
        "provider_last_modified_utc": current_capture["last_modified_utc"],
        "captured_training_and_delivery_days": len(captured),
        "required_training_and_delivery_days": 366,
        "missing_days": missing,
        "artifact_sha256": artifacts,
        "model_inputs_complete": False,
        "limitations": [
            "JAO lastModifiedOn is not an independent publication timestamp; actual pre-cutoff API capture is the proof of availability.",
            "A clean clone cannot retroactively reconstruct pre-cutoff JAO vintages for the prior 365 days.",
            "Initial RAM is relative to RefProg, not an absolute import capacity.",
        ],
    }
    if complete:
        validate_source_receipt(receipt, group=SOURCE_GROUP,
                                day=day.isoformat(), bundle=bundle, cutoff=cutoff)
    if previous is not None and previous.get("state") == "COMPLETE":
        require(previous == receipt, "Completed JAO source receipt is immutable")
    else:
        _atomic_write(receipt_path, _json_bytes(receipt))
    return {"receipt": str(receipt_path), "state": receipt["state"],
            "captured_days": len(captured), "missing_days": len(missing),
            "feature_artifact": str(bundle / f"{SOURCE_SUBDIR}/{FEATURE_NAME}") if complete else None}
