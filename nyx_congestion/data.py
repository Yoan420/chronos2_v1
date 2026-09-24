"""Immutable post-coupling labels for INITIAL constraints; never model features.

The learned universe is limited to original presolved initial constraints that
can be identified in the final domain without fuzzy names. Historical API
watermarks qualify retrospective label availability, not contemporaneous
captures; strict capture timestamps are retained separately. No final RAM,
final PTDF, shadow price or same-day label is returned as a feature.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import gzip
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Callable, Mapping

import httpx
import numpy as np
import pandas as pd

from chronos2_hourly.jao_flowbased import (
    JAO_CORE_DATA_URL, JaoCoreClient, build_windows_trust_context,
    expected_cutoff_utc, local_day_utc_bounds,
)
from chronos2_hourly.process_lock import exclusive_process_lock
from marginal_cost_expert.network import AHC_FIRST_DELIVERY, AHC_HUBS


ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = ROOT / "runs" / "experiments" / "nyx_congestion_v1"
LABEL_ROOT = NAMESPACE / "raw" / "labels"
ZONES = ("FR", "DE", "BE", "NL")
ENDPOINTS = ("finalComputation", "shadowPrices")
PTDF_TOLERANCE = 1e-6
ACTIVE_EPSILON = 1e-9
FEATURE_COLUMNS = (
    "feature_cnec_initial_ram_mw", "feature_cnec_initial_fmax_mw", "feature_cnec_initial_frm_mw",
    "feature_cnec_initial_ram_negative", "feature_cnec_initial_ram_to_fmax",
    *("feature_cnec_ptdf_" + z for z in ZONES),
    *("feature_cnec_initial_fr_minus_" + z + "_ptdf" for z in ("DE", "BE", "NL")),
)
_LOCAL = threading.local()


class CongestionDataError(ValueError):
    pass


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _utc(value, name):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise CongestionDataError(name + ": explicit timezone required")
    return t.tz_convert("UTC")


def _day(value) -> str:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise CongestionDataError("Exact timezone-naive civil date required")
    return stamp.strftime("%Y-%m-%d")


def _safe(value: str | Path) -> Path:
    path = Path(value).absolute()
    namespace = NAMESPACE.absolute()
    if path != path.resolve() or namespace != namespace.resolve() or not path.is_relative_to(namespace):
        raise CongestionDataError("Outputs and label sources must remain under nyx_congestion_v1, without aliases")
    return path


def _contract(day, endpoint):
    start, end = local_day_utc_bounds(day)
    return {"schema_version": 1, "endpoint": endpoint, "api_base_url": JAO_CORE_DATA_URL,
            "delivery_day": day, "start_utc": start.isoformat(), "end_utc": end.isoformat(),
            "filters": {"Presolved": True} if endpoint == "finalComputation" else {},
            "role": "post_cutoff_label_only", "allowed_as_model_input": False}


def _validate_bundle(payload, day, endpoint):
    if endpoint not in ENDPOINTS or payload.get("contract") != _contract(day, endpoint):
        raise CongestionDataError("Unexpected label endpoint, range, filters or source identity")
    rows, fetch = payload.get("records"), payload.get("fetch", {})
    if (not isinstance(rows, list) or type(fetch.get("total_rows")) is not int
            or fetch["total_rows"] != len(rows) or fetch.get("complete_response") is not True
            or fetch.get("tls_verified") is not True):
        raise CongestionDataError("A TLS-verified complete exact-count response is required")
    retrieved = _utc(fetch.get("retrieved_at_utc"), "retrieved_at_utc")
    modified = fetch.get("last_modified_utc")
    if modified is not None and _utc(modified, "last_modified_utc") > retrieved:
        raise CongestionDataError("A label watermark cannot be after its retrieval")
    start, end = local_day_utc_bounds(day)
    seen = set()
    for row in rows:
        stamp = _utc(row.get("dateTimeUtc"), "record timestamp")
        if not start <= stamp < end or stamp != stamp.floor("15min"):
            raise CongestionDataError("Label records must be aligned physical MTUs inside requested day")
        if row.get("id") is None or (stamp, str(row["id"])) in seen:
            raise CongestionDataError("Unique source row identities required")
        seen.add((stamp, str(row["id"])))
        if endpoint == "finalComputation" and row.get("presolved") is not True:
            raise CongestionDataError("Final-domain response violated Presolved=true")
    return payload


def _load_endpoint(directory, day, endpoint):
    target = _safe(_safe(directory) / endpoint)
    data, sidecar = _safe(target / "payload.json.gz"), _safe(target / "audit.json")
    if not target.exists():
        return None
    if not data.is_file() or not sidecar.is_file():
        raise CongestionDataError("Incomplete committed label endpoint: " + str(target))
    raw, audit_raw = data.read_bytes(), sidecar.read_bytes()
    audit = json.loads(audit_raw)
    if audit.get("raw_sha256") != _sha(raw):
        raise CongestionDataError("Label source SHA mismatch: " + str(data))
    payload = _validate_bundle(json.loads(gzip.decompress(raw)), day, endpoint)
    return payload, {str(data): _sha(raw), str(sidecar): _sha(audit_raw)}


def _commit(directory, day, endpoint, payload):
    _validate_bundle(payload, day, endpoint)
    directory = _safe(directory)
    destination = _safe(directory / endpoint)
    if destination.exists():
        raise CongestionDataError("Committed label endpoints are immutable")
    temp = _safe(directory / ("." + endpoint + "_" + uuid.uuid4().hex + ".tmp"))
    temp.mkdir(parents=True, exist_ok=False)
    raw = gzip.compress(_json(payload), mtime=0)
    (temp / "payload.json.gz").write_bytes(raw)
    (temp / "audit.json").write_bytes(_json({"schema_version": 1, "contract": payload["contract"],
        "raw_sha256": _sha(raw), "collector_code_sha256": _sha(Path(__file__).read_bytes()),
        "production_modified": False, "allowed_as_model_input": False}))
    # Commit both files in one directory rename. Interrupted temporary folders
    # are ignored, never confused with a completed endpoint or overwritten.
    for attempt in range(8):
        try:
            _safe(temp).rename(_safe(destination))
            break
        except OSError as exc:
            if (not isinstance(exc, PermissionError) and getattr(exc, "winerror", None) not in (5, 32)) or attempt == 7:
                raise
            if destination.exists():
                raise CongestionDataError("Concurrent label endpoint appeared") from exc
            time.sleep(.15 * (attempt + 1))


class LabelClient:
    """One pooled TLS client; final pagination and sparse shadow counts differ."""
    def __init__(self):
        context, _ = build_windows_trust_context()
        self.transport = httpx.Client(verify=context, timeout=45., follow_redirects=True,
            headers={"User-Agent": "NYX-congestion-retrospective-labels/1"})
        self.final = JaoCoreClient(client=self.transport, maximum_retries=2,
                                   request_interval_seconds=.65, page_size=40000)
        self.last = 0.

    def close(self):
        self.transport.close()

    def fetch(self, day, endpoint):
        start, end = local_day_utc_bounds(day)
        if endpoint == "finalComputation":
            time.sleep(max(0., .65 - (time.monotonic() - self.last)))
            fetched = self.final.fetch(endpoint, start_utc=start, end_utc=end, filters={"Presolved": True})
            meta = fetched.audit_dict()
            self.last = time.monotonic()
            fetch = {"total_rows": fetched.total_rows, "last_modified_utc": meta["last_modified_utc"],
                     "retrieved_at_utc": meta["retrieved_at_utc"], "complete_response": True,
                     "tls_verified": True, "requests": fetched.requests,
                     "pagination": "two_consistent_complete_scans_when_paginated",
                     "snapshot_fingerprint_sha256": fetched.snapshot_fingerprint_sha256}
            rows = list(fetched.rows)
        else:
            for attempt in range(3):
                time.sleep(max(0., .65 - (time.monotonic() - self.last)))
                self.last = time.monotonic()
                try:
                    response = self.transport.get(JAO_CORE_DATA_URL + "/shadowPrices",
                        params={"FromUtc": start.isoformat(), "ToUtc": end.isoformat()})
                    response.raise_for_status()
                    body = response.json()
                    rows = body.get("data")
                    count = body.get("totalRowsWithFilter")
                    if (body.get("rejected") or body.get("appliedFilter") not in (None, {})
                            or not isinstance(rows, list) or type(count) is not int or count != len(rows)):
                        raise CongestionDataError("Sparse shadow endpoint complete-response count not proven")
                    fetch = {"total_rows": count, "last_modified_utc": body.get("lastModifiedOn"),
                             "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                             "complete_response": True, "tls_verified": True, "requests": attempt + 1,
                             "pagination": "unpaginated_exact_declared_filtered_count",
                             "raw_http_sha256": _sha(response.content)}
                    break
                except (httpx.HTTPError, CongestionDataError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        return _validate_bundle({"contract": _contract(day, endpoint), "fetch": fetch, "records": rows}, day, endpoint)


def collect_labels(start_day, end_day, *, output_root=LABEL_ROOT, workers=2,
                   client_factory: Callable[[], Any] | None = None, progress=None):
    """Explicit bounded 1..366-day collection, immutable endpoint checkpoints."""
    first, last = _day(start_day), _day(end_day)
    days = pd.date_range(first, last, freq="D").strftime("%Y-%m-%d").tolist()
    if not 1 <= len(days) <= 366 or type(workers) is not int or workers not in (1, 2):
        raise CongestionDataError("Choose 1..366 civil days and one or two workers")
    output = _safe(output_root)
    output.mkdir(parents=True, exist_ok=True)
    clients, guard = [], threading.Lock()
    def one(day):
        directory = _safe(output / day)
        directory.mkdir(exist_ok=True)
        reused = 0
        with exclusive_process_lock(_safe(directory / "capture.lock")):
            for endpoint in ENDPOINTS:
                existing = _load_endpoint(directory, day, endpoint)
                if existing is not None:
                    reused += 1
                    continue
                if not hasattr(_LOCAL, "label_client"):
                    _LOCAL.label_client = (client_factory or LabelClient)()
                    with guard:
                        clients.append(_LOCAL.label_client)
                _commit(directory, day, endpoint, _LOCAL.label_client.fetch(day, endpoint))
            sources = {}
            for endpoint in ENDPOINTS:
                _, evidence = _load_endpoint(directory, day, endpoint)
                sources.update(evidence)
            manifest = _safe(directory / "manifest.json")
            content = {"schema_version": 1, "status": "complete", "day": day, "source_files": sources,
                       "endpoints": list(ENDPOINTS), "allowed_as_model_input": False}
            if manifest.exists():
                if json.loads(manifest.read_bytes()) != content:
                    raise CongestionDataError("Completed label-day manifest changed")
            else:
                with manifest.open("xb") as stream:
                    stream.write(_json(content))
        return {"day": day, "reused_endpoints": reused, "new_endpoints": 2 - reused,
                "manifest_path": str(manifest), "manifest_sha256": _sha(manifest.read_bytes())}
    outcomes, failures = [], []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(one, day): day for day in days}
            for future in as_completed(pending):
                day = pending[future]
                try:
                    outcomes.append(future.result())
                except Exception as exc:
                    failures.append({"day": day, "error": type(exc).__name__ + ": " + str(exc)})
                if progress:
                    progress(f"labels {len(outcomes)+len(failures)}/{len(days)} {day}; failures={len(failures)}")
    finally:
        for client in clients:
            client.close()
    return {"schema_version": 1, "status": "complete" if not failures else "incomplete",
            "output_root": str(output), "first_day": first, "last_day": last, "days": sorted(outcomes, key=lambda x: x["day"]),
            "failures": failures, "production_modified": False, "allowed_as_model_input": False,
            "production_pit_evidence": False}


def _eic(value):
    text = str(value or "").strip().upper()
    return None if text in {"", "NA", "N/A", "NONE", "NAN"} else text


def constraint_identity(row: Mapping[str, Any], *, shadow=False):
    """Physical identity, with sorted distinct contingency EICs; never fuzzy names."""
    eic = _eic(row.get("cnecEic" if shadow else "cneEic"))
    direction = str(row.get("direction") or "").strip().upper()
    if eic is None or direction not in {"DIRECT", "OPPOSITE"}:
        return None
    if shadow:
        branch = _eic(row.get("branchEic"))
        if not branch and row.get("contName"):
            return None
        contingencies = (branch,) if branch else ()
    else:
        source = row.get("contingencies") or []
        if not isinstance(source, list):
            return None
        branches = [_eic(c.get("branchEic")) for c in source]
        if any(c is None for c in branches) or (not branches and row.get("contName")):
            return None
        contingencies = tuple(sorted(set(branches)))
    return eic, direction, contingencies


def _numeric(row, key):
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        value = float(value)
        return value if np.isfinite(value) else math.nan
    except (ValueError, TypeError):
        return math.nan


def _ptdf_agrees(final, shadow):
    hubs = [name[4:] for name in shadow if name.startswith("hub_") and name not in {"hub_From", "hub_To"}]
    if not set(ZONES).issubset(hubs):
        return False
    for hub in hubs:
        a, b = _numeric(final, "ptdf_" + hub), _numeric(shadow, "hub_" + hub)
        before_ahc = _utc(shadow["dateTimeUtc"], "shadow timestamp").tz_convert("Europe/Paris").date() < AHC_FIRST_DELIVERY
        if before_ahc and hub in AHC_HUBS and final.get("ptdf_"+hub) is None and shadow.get("hub_"+hub) is None:
            continue
        if not np.isfinite(a) or not np.isfinite(b) or abs(a-b) > PTDF_TOLERANCE:
            return False
    return True


def label_day(day, final_payload, shadow_payload):
    """Return final-universe hourly labels + whole-zone diagnostic contributions.

    Sparse absence is zero only for a nonempty exact-count complete day response,
    known publication watermark and an unambiguous complete final hourly/MTU
    domain. Unmatched active physical shadow rows invalidate all zeros for the
    affected hour. External constraints remain in the zone diagnostic only.
    """
    day = _day(day)
    final = _validate_bundle(final_payload, day, "finalComputation")
    shadow = _validate_bundle(shadow_payload, day, "shadowPrices")
    first, end = local_day_utc_bounds(day)
    hours = pd.date_range(first, end, freq="h", inclusive="left")
    # The documented SDAC quarter-hour go-live is a data contract, not inferred
    # from missing timestamps. Before it, auction labels are hourly.
    minutes = 15 if day >= "2025-10-01" else 60
    mtus_per_hour = 60 // minutes
    frows, srows = final["records"], shadow["records"]
    final_by_stamp, shadow_by_stamp = {}, {}
    for row in frows:
        final_by_stamp.setdefault(_utc(row["dateTimeUtc"], "final timestamp"), []).append(row)
    for row in srows:
        shadow_by_stamp.setdefault(_utc(row["dateTimeUtc"], "shadow timestamp"), []).append(row)
    final_hourly = all(stamp == stamp.floor("h") for stamp in final_by_stamp)
    expected_all = set(pd.date_range(first, end, freq=f"{minutes}min", inclusive="left"))
    if set(shadow_by_stamp) - expected_all:
        raise CongestionDataError("Shadow granularity differs from explicit SDAC date contract")
    watermarks = [p["fetch"].get("last_modified_utc") for p in (final, shadow)]
    watermark_known = all(value is not None for value in watermarks)
    # Scheduled release time is only a conservative floor; API watermark
    # remains retrospective evidence rather than certified first publication.
    scheduled = (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=13, minutes=30)).tz_localize("Europe/Paris").tz_convert("UTC")
    available = max([scheduled, *[_utc(t, "watermark") for t in watermarks]]) if watermark_known else pd.NaT
    retrieved = max(_utc(p["fetch"]["retrieved_at_utc"], "retrieval") for p in (final, shadow))
    captured = max(retrieved, available) if watermark_known else retrieved
    rows, zones, diagnostics = [], [], []
    for hour in hours:
        expected = pd.date_range(hour, periods=mtus_per_hour, freq=f"{minutes}min")
        maps, contributions, constraint_contributions, unknown, counts = [], [], [], [], []
        for stamp in expected:
            domain = final_by_stamp.get(hour if final_hourly else stamp, [])
            mapping = {}
            for row in domain:
                identity = constraint_identity(row)
                if identity is not None:
                    mapping.setdefault(identity, []).append(row)
            unique = {key: value[0] for key, value in mapping.items() if len(value) == 1}
            labels = {key: 0. for key in unique}
            matched_contributions = {key: {z: 0. for z in ZONES} for key in unique}
            unmatched = []
            contribution = {z: 0. for z in ZONES}
            absolute_contribution = {z: 0. for z in ZONES}
            zone_ok = bool(domain and watermark_known and srows)
            seen = set()
            for row in shadow_by_stamp.get(stamp, []):
                price = _numeric(row, "shadowPrice")
                identity = constraint_identity(row, shadow=True)
                coeff = {z: _numeric(row, "hub_" + z) for z in ZONES}
                if not np.isfinite(price) or price < -ACTIVE_EPSILON or not all(np.isfinite(v) for v in coeff.values()):
                    zone_ok = False
                    unmatched.append("invalid_shadow_numeric")
                    continue
                price = price if price > ACTIVE_EPSILON else 0.
                for z in ZONES:
                    amount = price * (coeff["FR"] - coeff[z])
                    contribution[z] += amount
                    absolute_contribution[z] += abs(amount)
                if identity is None:
                    # Unidentified physical rows cannot be mistaken for known
                    # external allocation constraints and converted into zeros.
                    if _eic(row.get("cnecEic")) is not None:
                        unmatched.append("unidentifiable_physical_shadow")
                    continue
                if identity in seen or identity not in unique or not _ptdf_agrees(unique[identity], row):
                    if identity in seen:
                        zone_ok = False
                    unmatched.append(repr(identity))
                    continue
                seen.add(identity)
                labels[identity] = price if price > ACTIVE_EPSILON else 0.
                matched_contributions[identity] = {z: price * (coeff["FR"]-coeff[z]) for z in ZONES}
            valid = bool(domain and watermark_known and srows and not unmatched)
            maps.append((labels, valid, unique))
            constraint_contributions.append(matched_contributions)
            contributions.append((contribution, absolute_contribution, zone_ok))
            unknown.extend(unmatched)
            counts.append(len(domain))
        keys = set().union(*(set(item[0]) for item in maps))
        for identity in sorted(keys):
            known = all(ok and identity in labels for labels, ok, _ in maps)
            prices = [labels.get(identity, math.nan) for labels, _, _ in maps]
            descriptor = next(domain[identity] for _, _, domain in maps if identity in domain)
            record = {"timestamp_utc": hour, "constraint_identity": identity,
                         "constraint_key": _sha(_json(identity)), "cne_eic": identity[0],
                         "direction": identity[1], "contingency_eics": json.dumps(identity[2]),
                         "cne_name": str(descriptor.get("cneName") or ""),
                         "contingency_name": str(descriptor.get("contName") or ""),
                         "label_active": bool(any(v > ACTIVE_EPSILON for v in prices)) if known else None,
                         "label_shadow_price": float(np.mean(prices)) if known else math.nan,
                         "label_available_at_utc": available if known else pd.NaT,
                         "label_capture_available_at_utc": captured if known else pd.NaT,
                         "label_eligible": known, "label_mtu_count": mtus_per_hour if known else 0}
            for z in ZONES:
                # Use the exact matched shadow coefficients for both covered
                # and total mass; final PTDFs only validate physical identity.
                amounts = [values[identity][z] for values in constraint_contributions] if known else []
                record["label_contribution_"+z+"_fr_eur_mwh"] = float(np.mean(amounts)) if known else math.nan
                record["label_absolute_contribution_"+z+"_fr_eur_mwh"] = float(np.mean(np.abs(amounts))) if known else math.nan
            rows.append(record)
        for zone in ZONES:
            known = all(ok for _, _, ok in contributions)
            zones.append({"zone": zone, "timestamp_utc": hour,
                          "label_directional_contribution_eur_mwh": float(np.mean([v[zone] for v, _, _ in contributions])) if known else math.nan,
                          "label_absolute_contribution_eur_mwh": float(np.mean([v[zone] for _, v, _ in contributions])) if known else math.nan,
                          "label_physical_matching_complete": bool(all(ok for _, ok, _ in maps)),
                          "label_eligible": known, "label_available_at_utc": available if known else pd.NaT})
        diagnostics.append({"timestamp_utc": hour.isoformat(), "complete_label_mtu_count": sum(ok for _, ok, _ in maps),
                            "expected_label_mtu_count": mtus_per_hour, "final_row_counts": counts,
                            "unmatched_physical_shadow_count": len(unknown), "unmatched_examples": unknown[:3]})
    result = pd.DataFrame(rows)
    return result, pd.DataFrame(zones), {"day": day, "mtu_minutes": minutes, "final_domain_hourly": final_hourly,
        "label_watermarks_known": watermark_known, "label_available_at_utc": str(available),
        "label_capture_available_at_utc": str(captured), "hours": diagnostics,
        "shadow_day_nonempty": bool(srows),
        "zero_policy": "Only unique final identity, nonempty complete exact-count shadow day, all expected MTUs and no unmatched physical active shadow; sparse absent MTUs assumed inactive within that proven nonempty day response; wholly empty days unknown without separate publication monitoring proof",
        "label_availability_semantics": "max(final/shadow API watermarks, scheduled post-auction release floor); historical watermark evidence only",
        "production_pit_evidence": False, "allowed_as_model_input": False}


def build_constraint_panel(panel, network_features, network_audit, *, label_root=LABEL_ROOT, progress=None):
    """Join frozen initial-only features to strict final/shadow target labels."""
    from nyx_physical_p50.network import _panel_identity, _load_day
    identity = _panel_identity(panel)
    if (not network_features.index.equals(panel.index) or "network_eligible" not in network_features
            or network_features.network_eligible.isna().any()):
        raise CongestionDataError("Frozen original-index network eligibility required")
    eligible = identity.assign(eligible=network_features.network_eligible.to_numpy(bool))
    if eligible.groupby("timestamp_utc").eligible.nunique().gt(1).any():
        raise CongestionDataError("Country views disagree on original network availability")
    by_hour = eligible.drop_duplicates("timestamp_utc").set_index("timestamp_utc")
    days = {d["delivery_day"]: d for d in network_audit["days"]}
    source_files = dict(network_audit["source_files"])
    output, zone_outputs, day_audits = [], [], []
    directory = _safe(label_root)
    for number, day in enumerate(sorted(eligible.day.unique()), 1):
        if day not in days:
            raise CongestionDataError("Frozen initial-day audit missing")
        record = days[day]
        initial = _load_day(Path(record["selected_root"]), day, cutoff_time="08:00")
        # Raw identity/contingency records are retained because the old reduced
        # network frame deliberately stores contingency details only as JSON.
        original_path = initial.audit.get("raw_path")
        raw_rows = json.loads(gzip.decompress(Path(original_path).read_bytes()))["records"] if original_path else []
        selected = [r for r in raw_rows if _utc(r["dateTimeUtc"], "initial timestamp") in by_hour.index
                    and by_hour.loc[_utc(r["dateTimeUtc"], "initial timestamp"), "eligible"]]
        bundles, evidence = [], {}
        for endpoint in ENDPOINTS:
            loaded = _load_endpoint(directory/day, day, endpoint)
            if loaded is not None:
                bundles.append(loaded[0]); evidence.update(loaded[1])
        source_files.update(evidence)
        if len(bundles) == 2:
            targets, zonal, details = label_day(day, *bundles)
            lookup = {(r.timestamp_utc, r.constraint_key): r._asdict() for r in targets.itertuples(index=False)} if len(targets) else {}
        else:
            lookup, details = {}, {"day": day, "status": "missing_label_endpoint", "production_pit_evidence": False}
        physical_counts = {}
        for row in selected:
            physical_key = constraint_identity(row)
            if physical_key is not None:
                pair = (_utc(row["dateTimeUtc"], "initial timestamp"), _sha(_json(physical_key)))
                physical_counts[pair] = physical_counts.get(pair, 0) + 1
        matched, unidentified, ambiguous, keys_seen = 0, 0, 0, set()
        for row in selected:
            stamp = _utc(row["dateTimeUtc"], "initial timestamp")
            physical_key = constraint_identity(row)
            key = _sha(_json(physical_key)) if physical_key is not None else "unidentified:" + str(row["id"])
            is_ambiguous = physical_key is not None and physical_counts[(stamp, key)] > 1
            identified = physical_key is not None and not is_ambiguous
            target = lookup.get((stamp, key), {}) if identified else {}
            if identified:
                keys_seen.add((stamp, key))
            elif is_ambiguous:
                key += ":ambiguous:" + str(row["id"])
            known = target.get("label_eligible", False)
            matched += bool(known); unidentified += not identified; ambiguous += is_ambiguous
            values = {"timestamp_utc": stamp, "forecast_origin_utc": expected_cutoff_utc(day),
                "constraint_key": key, "source_initial_id": str(row["id"]), "cne_name": str(row.get("cneName") or ""),
                "contingency_name": str(row.get("contName") or ""), "cne_eic": _eic(row.get("cneEic")),
                "direction": str(row.get("direction") or ""), "constraint_identified": identified,
                "constraint_identity_ambiguous": is_ambiguous,
                "label_active": target.get("label_active") if known else None,
                "label_shadow_price": target.get("label_shadow_price", math.nan) if known else math.nan,
                "label_available_at_utc": target.get("label_available_at_utc", pd.NaT) if known else pd.NaT,
                "label_capture_available_at_utc": target.get("label_capture_available_at_utc", pd.NaT) if known else pd.NaT,
                "label_eligible": bool(known), "feature_cnec_initial_ram_mw": _numeric(row, "ram"),
                "feature_cnec_initial_fmax_mw": _numeric(row, "fmax"), "feature_cnec_initial_frm_mw": _numeric(row, "frm"),
                "feature_cnec_initial_ram_negative": float(_numeric(row, "ram") < 0)}
            ram, fmax = _numeric(row, "ram"), _numeric(row, "fmax")
            values["feature_cnec_initial_ram_to_fmax"] = ram/fmax if np.isfinite(fmax) and fmax > 0 else math.nan
            for z in ZONES:
                values["feature_cnec_ptdf_"+z] = _numeric(row, "ptdf_"+z)
            for z in ("DE", "BE", "NL"):
                values["feature_cnec_initial_fr_minus_"+z+"_ptdf"] = _numeric(row, "ptdf_FR")-_numeric(row, "ptdf_"+z)
            output.append(values)
        if len(bundles) == 2 and len(targets):
            active = targets.loc[targets.label_eligible & targets.label_active.eq(True)]
            outside = sum((r.timestamp_utc, r.constraint_key) not in keys_seen for r in active.itertuples(index=False))
        else:
            outside = 0
        if len(bundles) == 2:
            covered = {}
            for pair, target in lookup.items():
                if pair in keys_seen and target["label_eligible"]:
                    covered.setdefault(pair[0], []).append(target)
            zonal = zonal.copy()
            coverage_rows = []
            for row in zonal.itertuples(index=False):
                hour_available = row.timestamp_utc in by_hour.index and bool(by_hour.loc[row.timestamp_utc, "eligible"])
                valid = bool(hour_available and row.label_eligible and row.label_physical_matching_complete)
                records = covered.get(row.timestamp_utc, [])
                signed = sum(r["label_contribution_"+row.zone+"_fr_eur_mwh"] for r in records) if valid else math.nan
                absolute = sum(r["label_absolute_contribution_"+row.zone+"_fr_eur_mwh"] for r in records) if valid else math.nan
                denominator = row.label_absolute_contribution_eur_mwh
                if valid and absolute > denominator + 1e-7:
                    raise CongestionDataError("Covered physical contribution exceeds total shadow contribution")
                coverage_rows.append({"initial_hour_available": hour_available,
                    "initial_covered_signed_contribution_eur_mwh": signed,
                    "initial_covered_absolute_contribution_eur_mwh": absolute,
                    "outside_initial_absolute_contribution_eur_mwh": max(0., denominator-absolute) if valid else math.nan,
                    "initial_absolute_contribution_coverage": absolute/denominator if valid and denominator > 0. else math.nan})
            zonal = pd.concat([zonal.reset_index(drop=True), pd.DataFrame(coverage_rows)], axis=1)
            zone_outputs.append(zonal)
        day_audits.append({**details, "initial_rows": len(selected), "identified_initial_rows": len(selected)-unidentified,
                           "eligible_label_rows": matched, "unmatched_initial_rows": len(selected)-matched,
                           "ambiguous_initial_identity_rows": ambiguous,
                           "active_final_outside_initial_rows": outside})
        if progress:
            progress(f"constraints {number}/{len(days)} {day}: initial={len(selected)}, labels={matched}, active outside={outside}")
    for path, expected in source_files.items():
        if _sha(Path(path).read_bytes()) != expected:
            raise CongestionDataError("Source changed while building constraint panel: " + path)
    frame = pd.DataFrame(output)
    if len(frame):
        frame["label_active"] = pd.array(frame.label_active, dtype="boolean")
        for name in ("timestamp_utc", "forecast_origin_utc", "label_available_at_utc", "label_capture_available_at_utc"):
            frame[name] = pd.to_datetime(frame[name], utc=True)
    zone_frame = pd.concat(zone_outputs, ignore_index=True) if zone_outputs else pd.DataFrame()
    audit = {"schema_version": 1, "feature_columns": list(FEATURE_COLUMNS), "source_files": source_files,
        "rows": len(frame), "label_eligible_rows": int(frame.label_eligible.sum()) if len(frame) else 0,
        "days": day_audits, "active_final_outside_initial_rows": sum(d["active_final_outside_initial_rows"] for d in day_audits),
        "identity": "physical CNE EIC + exact direction + sorted distinct contingency EICs; no fuzzy names or reciprocal-line merging",
        "label_scope": "initial presolved constraints matched unambiguously to the final physical universe",
        "contribution_coverage_reference": "FR; hourly mean of sum absolute lambda*(PTDF_FR-PTDF_zone) over all shadow rows including external constraints; zero denominator means undefined ratio; unqualified initial hours and ambiguous matching have unknown coverage",
        "post_cutoff_columns_allowed_as_features": [], "production_modified": False,
        "production_pit_evidence": False, "independent_prospective_validation": False}
    return frame, zone_frame, audit


__all__ = ["collect_labels", "build_constraint_panel", "label_day", "constraint_identity", "FEATURE_COLUMNS", "CongestionDataError"]
