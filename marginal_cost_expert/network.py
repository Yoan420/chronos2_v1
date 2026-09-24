"""Strict, lossless JAO initial-domain input for the research dispatch expert.

This is deliberately NOT the imputed aggregate feature store used by Kalman.
All presolved constraints and every PTDF column are retained, including virtual
hubs. Historical API last-modified evidence is not a contemporaneous archive.
In particular, an initial domain is RefProg-balanced: its raw RAM must not be
passed to a zero-net-position LP without independently validating its reference.
No reference/boundary positions, later domains or missing hours are invented.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_hourly.jao_flowbased import (
    CORE_PTDF_ZONES, JAO_CORE_DATA_URL, JaoCoreClient,
    expected_cutoff_utc, expected_initial_publication_utc, local_day_utc_bounds,
)

AHC_FIRST_DELIVERY = date(2026, 6, 11)
AHC_HUBS = (
    "DE_DK1_VH", "DE_DK2_BigHub", "DE_NO2_BigHub", "DE_SE4_Baltic",
    "NL_DK1_COBRA", "NL_NO2_NorNed", "PL_LT_BigHub", "PL_SE4_SwePol",
    "RO_BG_VH",
)
DOCUMENTATION = {
    "initial_domain": "https://publicationtool.jao.eu/core/CORE_PublicationHandbook",
    "ahc_go_live": "https://www.jao.eu/news/confirmation-core-advanced-hybrid-coupling-go-live",
    "reference_equations": "https://eepublicdownloads.entsoe.eu/clean-documents/nc-tasks/Core%20DA%20CCM%203rd%20RfA%20-%20Clean%20version.pdf",
}


class NetworkContractError(ValueError):
    """A supplied network archive violates its declared source/schema contract."""


@dataclass(frozen=True)
class NetworkDay:
    constraints: pd.DataFrame
    qualification: pd.DataFrame
    audit: dict[str, Any]
    active_hubs: tuple[str, ...]
    inactive_hubs: tuple[str, ...]

    def active_constraints(self) -> pd.DataFrame:
        """Lossless active-hub view; NOT an assertion of an LP-ready RAM basis.

        Only historically inactive (all-null pre-AHC) columns are removed.
        Missing active values and unavailable rows remain missing, never zero.
        """
        excluded = {f"ptdf_{hub}" for hub in self.inactive_hubs}
        return self.constraints.drop(columns=list(excluded), errors="ignore").copy()


def _utc(value: Any, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise NetworkContractError(f"{name}: invalid timestamp") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise NetworkContractError(f"{name}: explicit timezone required")
    return stamp.tz_convert("UTC")


def _kind(row: Mapping[str, Any]) -> str:
    description = " ".join(str(row.get(key, "")) for key in
                           ("cnecType", "cneStatus", "cneName")).casefold()
    if "equality constraint" in description:
        return "equality_constraint"
    if any(token in description for token in
           ("external constraint", "allocation constraint", "import limit", "export limit")):
        return "external_constraint"
    if row.get("cnec") is True:
        return "cnec"
    # Older production payloads have cnec=null for every record. A documented
    # physical CNE (with EIC and element type) remains a physical constraint;
    # unknown technical rows are retained but do not establish CNEC coverage.
    if row.get("cnec") is None and row.get("cneEic") and row.get("elementType") in {
        "Line", "TieLine", "Transformer", "PST", "Cable",
    }:
        return "cnec"
    return "other_non_cnec"


def _empty_day(day: date, timezone: str, reason: str) -> NetworkDay:
    start, end = local_day_utc_bounds(day, timezone=timezone)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    active = (*CORE_PTDF_ZONES, *(AHC_HUBS if day >= AHC_FIRST_DELIVERY else ()))
    columns = ["delivery_start_utc", "cnec_id", "ram_mw",
               *[f"ptdf_{hub}" for hub in (*CORE_PTDF_ZONES, *AHC_HUBS)]]
    qualification = pd.DataFrame({"delivery_start_utc": hours, "inputs_qualified": False,
                                  "operational_pit_eligible": False, "reason": reason,
                                  "constraint_count": 0, "cnec_count": 0})
    return NetworkDay(pd.DataFrame(columns=columns), qualification, {
        "delivery_day": day.isoformat(), "expected_hours": len(hours),
        "available_hours": 0, "missing_hours": len(hours), "inputs_qualified": False,
        "operational_pit_eligible": False, "domain_reference_qualified": False,
        "usable_zero_based_domain": False, "boundary_qualified": False,
        "blockers": [reason], "imputation_performed": False,
        "publication_timestamp_certified": False,
    }, tuple(active), tuple(hub for hub in AHC_HUBS if hub not in active))


def normalise_network_payload(
    payload: Mapping[str, Any], delivery_day: str | date, *,
    source_audit: Mapping[str, Any] | None = None, timezone: str = "Europe/Paris",
    cutoff_time: str = "08:00", require_operational_capture: bool = False,
) -> NetworkDay:
    """Validate an original API envelope without reducing its physical domain.

    Data defects make affected hours ineligible. Malformed source identity,
    duplicate records or contradictory bounds raise instead of being repaired.
    Full-day qualification requires all 23/24/25 civil hours to be valid.
    Hourly network matrices are required; 15-minute matrices are not averaged.
    """
    day = pd.Timestamp(delivery_day).date()
    source_audit = dict(source_audit or {})
    start, end = local_day_utc_bounds(day, timezone=timezone)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    cutoff = expected_cutoff_utc(day, timezone=timezone, cutoff_time=cutoff_time)
    fetch = payload.get("fetch", {})
    if fetch.get("endpoint") != "initialComputation":
        raise NetworkContractError("Only initialComputation is allowed; later domains are not substitutes")
    if fetch.get("api_base_url") != JAO_CORE_DATA_URL:
        raise NetworkContractError("JAO production Core source identity is required")
    if fetch.get("filters", {}).get("Presolved") is not True:
        raise NetworkContractError("The complete Presolved=true domain is required")
    if _utc(fetch.get("start_utc"), "start_utc") != start or _utc(fetch.get("end_utc"), "end_utc") != end:
        raise NetworkContractError("Fetch bounds do not match the civil delivery day")
    raw = payload.get("records")
    if not isinstance(raw, list) or int(fetch.get("total_rows", -1)) != len(raw):
        raise NetworkContractError("Raw record count differs from declared API total")
    last_modified = _utc(fetch.get("last_modified_utc"), "last_modified_utc")
    retrieved = _utc(fetch.get("retrieved_at_utc"), "retrieved_at_utc")
    if last_modified > retrieved:
        raise NetworkContractError("API modification timestamp is after retrieval")
    watermarks = [_utc(value, "page_last_modified_utc")
                  for value in fetch.get("page_last_modified_utc", [])]
    if watermarks and max(watermarks) != last_modified:
        raise NetworkContractError("Page last-modified evidence contradicts the snapshot watermark")
    fallback = "fallback" in str(source_audit.get("pit_status", "")).casefold()
    fallback = fallback or bool(source_audit.get("causal_fallback_used", False))
    all_columns = sorted({key for row in raw for key in row if key.startswith("ptdf_")})
    all_hubs = tuple(column.removeprefix("ptdf_") for column in all_columns)
    expected = (*CORE_PTDF_ZONES, *(AHC_HUBS if day >= AHC_FIRST_DELIVERY else ()))
    missing_columns = sorted(set(expected) - set(all_hubs))
    inactive = tuple(hub for hub in AHC_HUBS if day < AHC_FIRST_DELIVERY and
                     not any(row.get(f"ptdf_{hub}") is not None for row in raw))
    # Unknown provider hubs remain active, including all-null unexpected columns:
    # they cannot be discarded under an undocumented schema change.
    active = tuple(sorted(set(expected) | (set(all_hubs) - set(inactive))))
    rows = []
    for row in raw:
        stamp = _utc(row.get("dateTimeUtc"), "dateTimeUtc")
        if not start <= stamp < end:
            raise NetworkContractError("A raw constraint lies outside the declared delivery day")
        if row.get("id") is None:
            raise NetworkContractError("Source row id is required; physical-name deduplication is unsafe")
        result = {
            "delivery_start_utc": stamp, "cnec_id": str(row["id"]),
            "ram_mw": row.get("ram"), "constraint_kind": _kind(row),
            "api_cnec": row.get("cnec"),
            "cnec_classification": ("legacy_physical_eic_and_type" if
                row.get("cnec") is None and _kind(row) == "cnec" else "api_or_descriptor"),
            "presolved": row.get("presolved") is True,
            "cne_name": row.get("cneName"), "cne_eic": row.get("cneEic"),
            "contingency_name": row.get("contName"), "contingency_tso": row.get("contTso"),
            "contingencies_json": json.dumps(row.get("contingencies", []), sort_keys=True),
            "direction": row.get("direction"), "hub_from": row.get("hubFrom"),
            "hub_to": row.get("hubTo"), "tso": row.get("tso"),
            "fref_init_mw": row.get("frefInit"), "fref_mw": row.get("fref"),
            "fcore_mw": row.get("fcore"), "fall_mw": row.get("fall"),
            "fuaf_mw": row.get("fuaf"), "fmax_mw": row.get("fmax"),
            "frm_mw": row.get("frm"), "fnrao_mw": row.get("fnrao"),
            "amr_mw": row.get("amr"), "iva_mw": row.get("iva"),
            "cva_mw": row.get("cva"), "lta_margin_mw": row.get("ltaMargin"),
            "ftotal_ltn_mw": row.get("ftotalLtn"),
            **{column: row.get(column) for column in all_columns},
        }
        rows.append(result)
    if not rows:
        result = _empty_day(day, timezone, "empty_original_domain")
        result.audit.update({"cutoff_utc": cutoff.isoformat(), "last_modified_utc": last_modified.isoformat()})
        return result
    frame = pd.DataFrame(rows).sort_values(["delivery_start_utc", "cnec_id"]).reset_index(drop=True)
    if frame.duplicated(["delivery_start_utc", "cnec_id"]).any():
        raise NetworkContractError("Duplicate source ids within a market time unit")
    numeric = ["ram_mw", *all_columns, *[col for col in frame if col.endswith("_mw") and col != "ram_mw"]]
    for column in dict.fromkeys(numeric):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for hub in missing_columns:
        frame[f"ptdf_{hub}"] = np.nan
    numeric_required = ["ram_mw", *[f"ptdf_{hub}" for hub in active]]
    row_valid = np.isfinite(frame[numeric_required].to_numpy(dtype=float)).all(axis=1) & frame.presolved
    frame["source_row_qualified"] = row_valid
    minute_aligned = frame.delivery_start_utc.dt.floor("h").eq(frame.delivery_start_utc)
    hourly_only = bool(minute_aligned.all())
    modified_before_cutoff = bool(last_modified <= cutoff)
    captured_before_cutoff = bool(retrieved <= cutoff)
    snapshot_safe = modified_before_cutoff and not fallback and hourly_only and not missing_columns
    if require_operational_capture:
        snapshot_safe = snapshot_safe and captured_before_cutoff
    grouped = {stamp: group for stamp, group in frame.groupby("delivery_start_utc", sort=False)}
    quality = []
    for stamp in hours:
        group = grouped.get(stamp)
        count = 0 if group is None else len(group)
        cnecs = 0 if group is None else int(group.constraint_kind.eq("cnec").sum())
        complete = bool(group is not None and cnecs > 0 and group.source_row_qualified.all())
        reasons = []
        if not count: reasons.append("missing_original_hour")
        elif not cnecs: reasons.append("no_cnec_for_hour")
        elif not complete: reasons.append("incomplete_ram_or_full_ptdf")
        if fallback: reasons.append("fallback_not_physical_network")
        if not modified_before_cutoff: reasons.append("modified_after_cutoff")
        if not hourly_only: reasons.append("non_hourly_domain_not_aggregated")
        if missing_columns: reasons.append("missing_active_ptdf_columns")
        if require_operational_capture and not captured_before_cutoff: reasons.append("not_captured_at_origin")
        qualified = complete and snapshot_safe
        quality.append({"delivery_start_utc": stamp, "inputs_qualified": qualified,
                        "operational_pit_eligible": qualified and captured_before_cutoff,
                        "reason": "qualified_historical_network_only" if qualified else ";".join(reasons),
                        "constraint_count": count, "cnec_count": cnecs})
    qualification = pd.DataFrame(quality)
    blockers = sorted({reason for row in quality if not row["inputs_qualified"]
                       for reason in str(row["reason"]).split(";")})
    audit = {
        "schema_version": 1, "delivery_day": day.isoformat(), "timezone": timezone,
        "cutoff_utc": cutoff.isoformat(), "endpoint": "initialComputation",
        "presolved_filter": True, "raw_rows": len(raw), "retained_rows": len(frame),
        "expected_hours": len(hours), "available_hours": int(qualification.inputs_qualified.sum()),
        "missing_hours": int((qualification.constraint_count == 0).sum()),
        "inputs_qualified": bool(qualification.inputs_qualified.all()),
        "operational_pit_eligible": bool(qualification.operational_pit_eligible.all()),
        "active_hubs": list(active), "inactive_hubs": list(inactive),
        "full_raw_ptdf_columns": all_columns, "missing_ptdf_columns": missing_columns,
        "last_modified_utc": last_modified.isoformat(), "retrieved_at_utc": retrieved.isoformat(),
        "expected_initial_publication_utc": expected_initial_publication_utc(day, timezone=timezone).isoformat(),
        "publication_timestamp_certified": False,
        "pit_evidence": "captured_before_cutoff" if captured_before_cutoff else "historical_api_last_modified_only",
        "domain_reference": "initial_computation_refprog_balanced_raw_not_translated",
        "domain_reference_qualified": False, "usable_zero_based_domain": False,
        "boundary_qualified": False, "imputation_performed": False,
        "source_audit_pit_status": source_audit.get("pit_status"),
        "legacy_classified_cnec_rows": int(frame.cnec_classification.eq("legacy_physical_eic_and_type").sum()),
        "blockers": blockers,
        "dispatch_blockers": ["initial_ram_reference_not_validated", "full_hub_boundary_positions_not_qualified"],
        "documentation": DOCUMENTATION,
        "limits": ["API lastModifiedOn is a revision watermark, not certified initial public availability",
                   "Initial CNEC/PTDF/RAM are not the final auction domain after NRAO, minRAM, validation and LTN",
                   "D2CF/Refprog reference publication is scheduled 10:30 D-1, after the 08:00 cutoff",
                   "Virtual hubs have no generation/load offers; network and boundary mappings must be explicit"],
    }
    return NetworkDay(frame, qualification, audit, active, inactive)


def load_network_day(
    root: str | Path, delivery_day: str | date, *, timezone: str = "Europe/Paris",
    cutoff_time: str = "08:00", require_operational_capture: bool = False,
) -> NetworkDay:
    """Read original raw/audit only. Missing archives cause explicit abstention."""
    day = pd.Timestamp(delivery_day).date()
    base = Path(root) / "raw" / "initialComputation"
    raw_path, audit_path = base / f"{day}.json.gz", base / f"{day}.audit.json"
    if not raw_path.exists() and not audit_path.exists():
        return _empty_day(day, timezone, "missing_original_archive")
    if not raw_path.is_file() or not audit_path.is_file():
        raise NetworkContractError(f"Incomplete raw/audit archive pair for {day}")
    blob = raw_path.read_bytes()
    audit_blob = audit_path.read_bytes()
    audit = json.loads(audit_blob.decode("utf-8"))
    sha = hashlib.sha256(blob).hexdigest()
    if audit.get("raw_gzip_sha256") != sha:
        raise NetworkContractError(f"Raw checksum mismatch: {raw_path}")
    payload = json.loads(gzip.decompress(blob))
    result = normalise_network_payload(payload, day, source_audit=audit, timezone=timezone,
                                      cutoff_time=cutoff_time, require_operational_capture=require_operational_capture)
    result.audit.update({"raw_path": str(raw_path.resolve()), "source_audit_path": str(audit_path.resolve()),
                         "raw_gzip_sha256": sha, "source_audit_sha256": hashlib.sha256(audit_blob).hexdigest()})
    return result


def audit_network_window(root: str | Path, first_day: str | date, last_day: str | date,
                         *, timezone: str = "Europe/Paris", overlay_root: str | Path | None = None,
                         output_path: str | Path | None = None, cutoff_time: str = "08:00") -> dict[str, Any]:
    """Read/check every raw partition, retaining only small daily diagnostics.

    A calendar window is never compressed by dropping absent days. Contract
    failures are retained as explicit failures instead of stopping the audit.
    """
    start, end = pd.Timestamp(first_day).date(), pd.Timestamp(last_day).date()
    if start > end:
        raise NetworkContractError("Network audit start must not be after its end")
    daily = []
    for stamp in pd.date_range(start, end, freq="D"):
        day = stamp.date()
        selected_root = Path(root)
        if overlay_root is not None:
            directory = Path(overlay_root) / "raw" / "initialComputation"
            if any((directory / f"{day}.{suffix}").exists() for suffix in ("json.gz", "audit.json")):
                selected_root = Path(overlay_root)
        try:
            result = load_network_day(selected_root, day, timezone=timezone, cutoff_time=cutoff_time)
            selected = {key: result.audit.get(key) for key in (
                "delivery_day", "expected_hours", "available_hours", "missing_hours",
                "raw_rows", "inputs_qualified", "operational_pit_eligible", "blockers",
                "last_modified_utc", "retrieved_at_utc", "raw_gzip_sha256", "raw_path",
                "source_audit_path", "source_audit_sha256", "cutoff_utc")}
            selected["active_hubs"] = list(result.active_hubs)
            selected["selected_root"] = str(selected_root.resolve())
            daily.append(selected)
        except (NetworkContractError, OSError, ValueError, KeyError) as exc:
            missing = _empty_day(day, timezone, "invalid_original_archive").audit
            directory = selected_root / "raw" / "initialComputation"
            raw_path, source_audit_path = directory / f"{day}.json.gz", directory / f"{day}.audit.json"
            daily.append({**missing, "error": str(exc),
                          "selected_root": str(selected_root.resolve()),
                          "raw_path": str(raw_path.resolve()),
                          "source_audit_path": str(source_audit_path.resolve()),
                          "raw_gzip_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.is_file() else None,
                          "source_audit_sha256": hashlib.sha256(source_audit_path.read_bytes()).hexdigest() if source_audit_path.is_file() else None})
    report = {"first_day": start.isoformat(), "last_day": end.isoformat(),
            "calendar_days": len(daily), "expected_hours": sum(row["expected_hours"] for row in daily),
            "complete_research_days": sum(bool(row["inputs_qualified"]) for row in daily),
            "qualified_original_hours": sum(int(row.get("available_hours") or 0) for row in daily),
            "operational_capture_days": sum(bool(row["operational_pit_eligible"]) for row in daily),
            "domain_reference_qualified": False, "boundary_qualified": False,
            "imputation_performed": False, "daily": daily,
            "timezone": timezone, "cutoff_time": cutoff_time,
            "network_code_path": str(Path(__file__).resolve()),
            "network_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_root": str(Path(root).resolve()),
            "overlay_root": str(Path(overlay_root).resolve()) if overlay_root else None,
            "overlay_policy": "explicit_precedence_if_either_raw_or_audit_exists_no_corrupt_fallback"}
    if output_path is not None:
        output = Path(output_path).resolve()
        original = Path(__file__).resolve().parents[1] / "data" / "pit" / "jao_core_flowbased"
        if output == original or original in output.parents:
            raise NetworkContractError("The original JAO bank cannot receive experiment audits")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    return report


def archive_initial_probe(client: JaoCoreClient, output_root: str | Path,
                          delivery_day: str | date) -> NetworkDay:
    """Archive one fresh API response in a new isolated location, never overwrite.

    The caller controls/authorizes the requested dates. The old aggregate bank
    must not be passed as output_root; existing raw/audit pairs are refused.
    """
    root = Path(output_root).resolve()
    original = Path(__file__).resolve().parents[1] / "data" / "pit" / "jao_core_flowbased"
    if root == original or original in root.parents:
        raise NetworkContractError("The original JAO bank is read-only for this experiment")
    day = pd.Timestamp(delivery_day).date()
    raw_dir = root / "raw" / "initialComputation"
    raw_path, audit_path = raw_dir / f"{day}.json.gz", raw_dir / f"{day}.audit.json"
    if raw_path.exists() or audit_path.exists():
        raise NetworkContractError("Probe archive already exists; refusing to overwrite")
    fetched = client.fetch_initial_day(day)
    payload = {"schema_version": 1, "source": "JAO Core production initialComputation",
               "fetch": fetched.audit_dict(), "records": list(fetched.rows),
               "license_notice": "JAO public publication tool; preserve source attribution"}
    result = normalise_network_payload(payload, day)
    blob = gzip.compress(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    audit = {**result.audit, "raw_gzip_sha256": hashlib.sha256(blob).hexdigest()}
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves a concurrently created archive too.
    with raw_path.open("xb") as handle:
        handle.write(blob)
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    return load_network_day(root, day)
