"""Prospective input captures: read production, write only new private evidence.

No materializer, fitting process, forecast launcher, cache rewrite or historical
price fallback is called. The only API reads are uncached canonical-target
checks. Query-as-of timestamps remain different from publication certification.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from chronos2_exogenous.prospective_inputs import _capture_target
from economic_value.data import _stable_bytes
from nyx_scarcity import data as scarcity_data
from . import ledger
from .features import make_stress_features


ZONES = ("FR", "DE", "BE", "NL")
MAX_REFRESH_DIRECTORIES = 20
MAX_OBSERVATION_DAYS = 31


def _now():
    return pd.Timestamp(datetime.now(timezone.utc))


def _zones(zones):
    if not isinstance(zones, (list, tuple)) or not zones or len(set(zones)) != len(zones) or set(zones)-set(ZONES):
        raise ValueError("Unique supported FR, DE, BE, NL countries required.")
    return list(zones)


def _evidence(root, path):
    path = ledger._path(root, path)
    _, evidence = _stable_bytes(path)
    return evidence


def _canonical_read(root, days, zones):
    """One exact canonical query per country, with global before/after hashes."""
    from run_chronos2_exogenous_panel import _canonical_target_path
    root = Path(root).resolve()
    started = _now()
    expected = pd.DatetimeIndex(sorted(t for day in days for t in ledger._index(day)))
    contracts, pinned = {}, {}
    for zone in zones:
        cache, contract = _canonical_target_path(root, zone)
        if not isinstance(contract.get("series"), str) or not contract["series"]:
            raise ValueError(f"{zone}: exact canonical target series required.")
        cache = ledger._path(root, cache)
        if ledger._path(root, contract["cache_path"]) != cache:
            raise ValueError("Canonical target cache identity disagrees with its contract.")
        contracts[zone] = {**contract, "canonical_cache_sha256": _evidence(root, cache)["sha256"]}
        for name in ("cache_path", "base_config", "live_config"):
            if name in contract:
                pinned[str(ledger._path(root, contract[name]))] = _evidence(root, contract[name])["sha256"]
    results, sources, calls = [], {}, {}
    full_index = pd.date_range(ledger._index(days[0])[0], ledger._index(days[-1])[-1], freq="h")
    for zone in zones:
        values, audit = _capture_target(root=root, zone=zone, start=days[0], end=days[-1], refresh=True)
        contract = contracts[zone]
        if (audit.get("fresh_api_read") is not True or audit.get("nocache") is not True
                or audit.get("fallback_used") is not False or audit.get("source_cache_modified") is not False
                or audit.get("series") != contract["series"]
                or audit.get("canonical_source_cache_sha256") != contract["canonical_cache_sha256"]):
            raise ValueError(f"{zone}: fresh uncached canonical target evidence is incomplete or inconsistent.")
        first = ledger._utc(audit["capture_started_at_utc"])
        last = ledger._utc(audit["capture_completed_at_utc"])
        if not started <= first <= last <= _now():
            raise ValueError("Canonical API receipt is outside its real invocation clock.")
        if not isinstance(values, pd.Series) or values.index.has_duplicates or values.index.tz is None:
            raise ValueError("Canonical target must return unique timezone-aware physical hours.")
        index = pd.DatetimeIndex(values.index).tz_convert("UTC")
        if not index.equals(full_index):
            raise ValueError("Canonical target check returned an incomplete or foreign physical-hour index.")
        numeric = pd.to_numeric(values, errors="raise").to_numpy(float)
        if np.isinf(numeric).any():
            raise ValueError("Infinite canonical target observations are forbidden.")
        selected = pd.Series(numeric, index=index).reindex(expected)
        results.append(pd.DataFrame({"zone": zone, "timestamp_utc": expected, "actual": selected.to_numpy()}))
        sources[zone] = {"series": contract["series"], "canonical_cache_sha256": contract["canonical_cache_sha256"],
                         "canonical_cache_path": contract["cache_path"]}
        calls[zone] = audit
    for path, digest in pinned.items():
        if _evidence(root, path)["sha256"] != digest:
            raise ValueError("Canonical target cache or configuration changed during the multi-country check.")
    received = _now()
    receipt = {"schema_version": 1, "started_at_utc": started.isoformat(), "received_at_utc": received.isoformat(),
               "zones": zones, "fresh_api_read": True, "nocache": True, "target_sources": sources,
               "source_checks": calls, "canonical_cache_modified": False,
               "historical_publication_times_verified": False}
    return pd.concat(results, ignore_index=True), receipt


def fresh_label_check(root, day, zones) -> dict:
    """An API error is an error, never silently interpreted as zero observations."""
    day, zones = ledger._day(day), _zones(zones)
    frame, receipt = _canonical_read(root, [day], zones)
    return {**receipt, "kind": "fresh_canonical_target_check", "delivery_day": day,
            "observed_hours_by_zone": {z: int(frame.loc[frame.zone.eq(z), "actual"].notna().sum()) for z in zones}}


def collect_observations(root, days, zones):
    """Return requested hours including NaNs; the ledger resolves complete days."""
    zones = _zones(zones)
    days = [days] if isinstance(days, str) else list(days)
    days = [ledger._day(day) for day in days]
    if not days or len(set(days)) != len(days) or len(days) > MAX_OBSERVATION_DAYS:
        raise ValueError("Select one to31 unique observation days per request.")
    days.sort()
    if (pd.Timestamp(days[-1])-pd.Timestamp(days[0])).days >= MAX_OBSERVATION_DAYS:
        raise ValueError("Observation query span is bounded to31 civil days.")
    frame, receipt = _canonical_read(root, days, zones)
    return frame, {**receipt, "kind": "canonical_target_observations", "delivery_days": days,
                   "missing_hours": int(frame.actual.isna().sum()), "forecast_recalculated": False}


def _refresh_candidates(root, day):
    """Only20 newest standard private refresh directories, never a broad scan."""
    base = root/"runs/experiments/nyx_scarcity_v1/source_refresh"
    if not base.exists():
        return []
    ledger._path(root, base)
    folders = sorted((p for p in base.iterdir() if re.fullmatch(r"\d{8}T\d{6,12}Z_[a-f0-9]{8}", p.name)), reverse=True)
    candidates = []
    for directory in folders[:MAX_REFRESH_DIRECTORIES]:
        ledger._path(root, directory)
        for relative in ("fuel/fuel_refresh_audit.json", "refresh_audit.json"):
            path = directory/relative
            if not path.exists():
                continue
            raw, evidence = _stable_bytes(ledger._path(root, path))
            audit = json.loads(raw)
            if (audit.get("status") != "complete" or audit.get("required_sources_complete") is not True
                    or audit.get("original_sources_modified") is not False):
                continue
            if ledger._day(audit["delivery_day"]) < day:
                continue
            config_path = ledger._path(root, audit["saved_config"])
            if config_path.parent != path.parent or ledger._path(root, audit["source_dir"]) != path.parent:
                raise ValueError("Completed private refresh manifest points outside its own directory.")
            config_raw, config_evidence = _stable_bytes(config_path)
            config = json.loads(config_raw)
            mapping = config.get("data", {}).get("source_overrides", {})
            if not isinstance(mapping, dict) or set(mapping)-set(scarcity_data.source_registry()):
                raise ValueError("Refresh config may relocate only exact registered physical sources.")
            candidates.append({"overrides": mapping, "audit": evidence, "config": config_evidence})
    return candidates


def _select_sources(root, config, day):
    """Identity+as-of coverage chooses paths, never feature definitions/settings."""
    registry = scarcity_data.source_registry()
    data = config.get("data", {})
    original = data.get("source_overrides", {})
    if not isinstance(original, dict) or set(original)-set(registry):
        raise ValueError("Frozen source configuration contains an unknown source override.")
    candidates = _refresh_candidates(root, day)
    index = ledger._index(day)
    chosen, audits, optional_gaps, missing = {}, {}, {}, []
    for key, spec in registry.items():
        paths = [candidate["overrides"][key] for candidate in candidates if key in candidate["overrides"]]
        paths += [spec["path"]]
        if key in original:
            paths.append(original[key])
        paths = list(dict.fromkeys(paths))
        options = []
        for path in paths:
            values, audit = scarcity_data._read_source(root, key, {**spec, "path": path}, index)
            count = int(values[spec["feature"]].notna().sum())
            options.append((count, path, audit))
            if count == len(index):
                break
        best = next((option for option in options if option[0] == len(index)), max(options, key=lambda option: option[0]))
        chosen[key], audits[key] = best[1], best[2]
        if best[0] != len(index):
            if key.endswith("_temperature") or key in {"ttf", "eua"}:
                optional_gaps[key] = len(index)-best[0]
            else:
                missing.append(key)
    if missing:
        raise ValueError("Prospective physical inputs incomplete for "+day+": "+", ".join(missing)+
                         ". No values were fabricated or refreshed. Run the existing isolated Scarcity.ps1 -Action Refresh -DeliveryDay "+day+
                         " after the operational forecast; RL/FR nuclear still require their normal operational materialization. Then Capture again.")
    resolved = deepcopy(config)
    resolved.setdefault("data", {})["source_overrides"] = chosen
    resolved.update(delivery_day=day, end_day=None, zones=list(ZONES))
    return resolved, {"selected": audits, "optional_missing_hours": optional_gaps,
                      "completed_refresh_manifests_considered": [{"audit": c["audit"], "config": c["config"]} for c in candidates],
                      "discovery_directory_limit": MAX_REFRESH_DIRECTORIES,
                      "selection": "latest compatible completed private refresh, else current exact-registry source, else frozen seed; only full-day qualified paths for required physical inputs"}


def _loaded_evidence(audit):
    """Enumerate existing source artifacts without trusting arbitrary path text."""
    result = {}
    for key, item in audit.get("sources", {}).items():
        if item.get("sha256"):
            result["physical_"+key] = {"path": item["path"], "sha256": item["sha256"]}
        if item.get("audit", {}).get("sha256"):
            result["physical_"+key+"_audit"] = item["audit"]
    for i, item in enumerate(audit.get("baseline", {}).get("sources", [])):
        for j, evidence in enumerate([item, item.get("csv_source"), *item.get("time_axis_audits", [])]):
            if evidence and evidence.get("sha256"):
                result[f"report_{i}_{j}"] = {"path": evidence["path"], "sha256": evidence["sha256"]}
    return result


def capture_inputs(root, snapshot_dir, delivery_day, output_directory):
    """Capture a complete new unobserved day from existing production outputs."""
    from . import runner
    root, day = Path(root).resolve(), ledger._day(delivery_day)
    output = runner.safe_path(root, output_directory)
    if output.exists():
        raise ValueError("Input capture directory already exists; evidence is create-only.")
    snapshot, _, manifest = runner.read_suite(snapshot_dir, root=root)
    runner.verify_result(snapshot, manifest)
    started = _now()
    if started < ledger._origin(day):
        raise ValueError("The requested day information cutoff is still in the future; no backdated capture.")
    before = fresh_label_check(root, day, list(ZONES))
    if any(before["observed_hours_by_zone"].values()):
        raise ValueError("Canonical day-ahead observations already exist; retrospective capture is forbidden.")
    source_raw, source_evidence = _stable_bytes(snapshot/"source_audit.json")
    source_audit = json.loads(source_raw)
    config = source_audit.get("source_config")
    if not isinstance(config, dict) or config.get("baseline_model", "nuclear_kalman") != "nuclear_kalman":
        raise ValueError("A frozen nuclear_kalman source_config is required.")
    resolved, selection = _select_sources(root, config, day)
    all_rows, data_audit = scarcity_data.load_inputs(resolved, root=root)
    if (all_rows.columns.has_duplicates or not {"zone", "timestamp_utc", "forecast_origin_utc", "forecast", "q10", "q90", "actual"}.issubset(all_rows)):
        raise ValueError("Unexpected prospective report/source schema.")
    times = pd.to_datetime(all_rows.timestamp_utc, utc=True, errors="raise")
    current = all_rows.loc[times.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d").eq(day)].copy()
    if current.actual.notna().any():
        raise ValueError("The selected report already contains observations; they cannot be hidden before prospective issuance.")
    if set(current.zone) != set(ZONES) or current.duplicated(["zone", "timestamp_utc"]).any():
        raise ValueError("Complete unique four-country prospective delivery required.")
    for zone, block in current.groupby("zone"):
        if not pd.DatetimeIndex(pd.to_datetime(block.timestamp_utc, utc=True).sort_values()).equals(ledger._index(day)):
            raise ValueError(f"{zone}: complete physical23/24/25-hour input day required.")
    if not pd.to_datetime(current.forecast_origin_utc, utc=True).eq(ledger._origin(day)).all():
        raise ValueError("Prospective source origin differs from D-1 08:00.")
    q = current[["q10", "forecast", "q90"]].to_numpy(float)
    if not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
        raise ValueError("Finite intact ordered published baseline P10/P50/P90 required.")
    feature_columns = list(data_audit["feature_columns"])
    if (len(set(feature_columns)) != len(feature_columns) or not set(feature_columns).issubset(current)
            or any(not c.startswith("feature_") or any(t in c.lower() for t in ("actual", "storm", "benchmark", "label", "target", "oracle", "error")) for c in feature_columns)):
        raise ValueError("Unexpected label/comparator column in the explicitly declared input feature bank.")
    base_columns = ["zone", "timestamp_utc", "forecast_origin_utc", "forecast", "q10", "q90"]
    metadata = [c for c in ("feature_eligible", "feature_available_at_utc", "forecast_eligible") if c in current and c not in feature_columns]
    clean = current[[*base_columns, *feature_columns, *metadata]].sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    if "actual" in clean or "benchmark_forecast" in clean:
        raise ValueError("Labels or Storm escaped prospective sanitisation.")
    checked, _, required, feature_audit = make_stress_features(clean)
    if not checked[required].notna().all().all():
        raise ValueError("Incomplete physical delivery profiles; refusing prospective Capture rather than manufacturing features.")
    original_artifacts = _loaded_evidence(data_audit)
    for key, item in selection["selected"].items():
        if item.get("sha256"):
            original_artifacts["selected_"+key] = {"path": item["path"], "sha256": item["sha256"]}
        if item.get("audit", {}).get("sha256"):
            original_artifacts["selected_"+key+"_audit"] = item["audit"]
    original_artifacts["frozen_source_audit"] = source_evidence
    for i, entry in enumerate(selection["completed_refresh_manifests_considered"]):
        original_artifacts[f"refresh_{i}_audit"] = entry["audit"]
        original_artifacts[f"refresh_{i}_config"] = entry["config"]
    ledger._verify_files(root, original_artifacts)
    after = fresh_label_check(root, day, list(ZONES))
    if any(after["observed_hours_by_zone"].values()):
        raise ValueError("Observations became available during input capture; no prospective input was published.")
    runner.read_suite(snapshot, root=root)
    ledger._verify_files(root, original_artifacts)
    captured = _now()
    output.mkdir(parents=True, exist_ok=False)
    baseline = clean[base_columns]
    feature_frame = clean[["zone", "timestamp_utc", "forecast_origin_utc", *feature_columns, *metadata]]
    for name, frame in (("baseline", baseline), ("features", feature_frame), ("panel", clean)):
        ledger._write_frame(output/f"{name}.parquet", frame)
    ledger._write_json(output/"source_audit.json", {"resolved_input_config": resolved, "source_selection": selection,
        "data": data_audit, "stress_feature_preflight": feature_audit, "snapshot_source_audit": source_evidence})
    ledger._write_json(output/"label_checks.json", {"before": before, "after": after})
    artifacts = {**original_artifacts, **{name: ledger._file(root, output/f"{name}.parquet") for name in ("baseline", "features", "panel")},
                 "capture_audit": ledger._file(root, output/"source_audit.json"), "label_checks": ledger._file(root, output/"label_checks.json")}
    evidence = {"schema_version": 1, "delivery_day": day, "zones": list(ZONES), "rows": len(clean),
        "information_cutoff_utc": ledger._origin(day).isoformat(), "captured_at_utc": captured.isoformat(),
        "artifacts": artifacts, "source_information_times_utc": {"baseline_export_contract": ledger._origin(day).isoformat(),
            **{name: ledger._origin(day).isoformat() for name, item in selection["selected"].items() if item.get("finite_hours", 0)}},
        "source_information_time_semantics": "Latest eligible source query-as-of bound at D-1 08:00, NOT recovered publication time; baseline report origin is assumed by its existing export contract.",
        "publication_evidence": "historical_asof", "source_publication_certified": False,
        "forecast_pit_certified": False, "model_refitted": False, "physical_sources_refreshed": False,
        "observed_or_storm_fields_in_panel": False, "diagnostic_only": True, "production_modified": False,
        "output_directory": str(output), "panel_path": str(output/"panel.parquet")}
    ledger._write_json(output/"input_evidence.json", evidence)
    return clean, evidence


__all__ = ["capture_inputs", "fresh_label_check", "collect_observations"]
