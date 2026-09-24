"""Local append-only prospective decision evidence; never activates production.

Information cutoff, wall-clock issue time and observation receipt are distinct.
Hashes detect accidental changes; this is not an externally notarised clock or
an append-only filesystem against an administrator rewriting the whole ledger.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Mapping
import uuid

import numpy as np
import pandas as pd

from chronos2_hourly.process_lock import exclusive_process_lock


NAMESPACE = Path("runs/experiments/nyx_scarcity_v1/stress_guard")
GRADES = ("strict_08_issue", "pre_observation_asof08")
ZONES = {"FR", "DE", "BE", "NL"}
KEYS = ["zone", "timestamp_utc"]
MAX_RECEIPT_AGE_SECONDS = 120
FLAGS = {"diagnostic_only": True, "production_modified": False, "activation_performed": False,
         "external_timestamp_notarisation": False}


class LedgerError(ValueError):
    pass


def now_utc() -> pd.Timestamp:
    """Actual host clock; no CLI argument can supply an earlier issue time."""
    return pd.Timestamp(datetime.now(timezone.utc))


def _utc(value, label="timestamp"):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise LedgerError(f"{label}: explicit UTC-aware timestamp required.")
    return stamp.tz_convert("UTC")


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise LedgerError("A YYYY-MM-DD civil date is required.")
    return pd.Timestamp(value).date().isoformat()


def _origin(day):
    return (pd.Timestamp(_day(day))-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")


def _deadline(day, grade):
    offset = 8*60 if grade == "strict_08_issue" else 11*60+45
    return (pd.Timestamp(_day(day))-pd.Timedelta(days=1)+pd.Timedelta(minutes=offset)).tz_localize("Europe/Paris").tz_convert("UTC")


def _index(day):
    first = pd.Timestamp(_day(day)).tz_localize("Europe/Paris")
    last = (pd.Timestamp(day)+pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    return pd.date_range(first, last, freq="h", inclusive="left").tz_convert("UTC")


def _clean(value):
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or value is pd.NaT or isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _bytes(value):
    return json.dumps(_clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf8")


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _path(root, value, *, output=False):
    root = Path(root).resolve()
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise LedgerError("Parent traversal is forbidden.")
    raw = raw if raw.is_absolute() else root/raw
    path = raw.resolve()
    base = root/NAMESPACE if output else root
    if not path.is_relative_to(base) or path == base:
        raise LedgerError("Path is outside the isolated stress_guard namespace." if output else "Source must remain inside the project.")
    for parent in [raw, *raw.parents]:
        if parent == root.parent:
            break
        if parent.is_symlink() or hasattr(parent, "is_junction") and parent.is_junction():
            raise LedgerError("Symlink or junction paths are forbidden.")
    return path


def _file(root, value):
    path = _path(root, value)
    if not path.is_file():
        raise LedgerError(f"Sealed input file is missing: {path.name}.")
    return {"path": str(path), "sha256": _digest(path), "bytes": path.stat().st_size}


def _verify_files(root, entries):
    if not isinstance(entries, Mapping) or not entries:
        raise LedgerError("Nonempty sealed artifact mapping required.")
    for name, item in entries.items():
        if not isinstance(item, Mapping) or not re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256", ""))):
            raise LedgerError(f"{name}: valid artifact hash required.")
        path = _path(root, item["path"])
        if not path.is_file() or _digest(path) != item["sha256"]:
            raise LedgerError(f"Source artifact changed or missing: {name}.")


def _write_json(path, value):
    # Create-only: existing evidence can never be replaced by this API.
    with path.open("xb") as stream:
        stream.write(_bytes(value)); stream.flush()
        import os
        os.fsync(stream.fileno())


def _write_frame(path, frame):
    with path.open("xb") as stream:
        frame.to_parquet(stream, index=False)


def _frame_digest(frame):
    return hashlib.sha256(_bytes({"columns": list(frame), "dtypes": [str(t) for t in frame.dtypes],
        "records": frame.to_dict("records")})).hexdigest()


def _last_history_day(path):
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        frame = pd.read_csv(path)
    else:
        return None
    for column in ("timestamp_utc", "delivery_start_utc", "timestamp"):
        if column in frame and frame[column].notna().any():
            stamps = [_utc(v, "history timestamp") for v in frame[column].dropna()]
            return max(stamps).tz_convert("Europe/Paris").date().isoformat()
    raise LedgerError("A tabular history must expose timezone-aware delivery timestamps.")


def freeze_ledger(output_directory, *, root, candidate_files: dict, history_files: dict,
                  explored_through_day: str, zones: list[str], issue_policy="pre_observation_asof08",
                  requested_start_day=None, deadline_local="11:45") -> Path:
    """Freeze the protocol before future origins; no fit, market read or issue."""
    root = Path(root).resolve()
    destination = _path(root, output_directory, output=True)
    if issue_policy not in GRADES or deadline_local != "11:45":
        raise LedgerError("Use a declared issue grade and fixed internal deadline 11:45; this is not a market publication claim.")
    if not zones or len(set(zones)) != len(zones) or set(zones)-ZONES:
        raise LedgerError("Unique supported country identities required.")
    if not isinstance(candidate_files, dict) or not {"model", "config"}.issubset(candidate_files) or not history_files:
        raise LedgerError("Freeze model, config, code and nonempty historical evidence.")
    candidates = {str(k): _file(root, v) for k, v in candidate_files.items()}
    candidates["ledger_code"] = _file(root, Path(__file__))
    history = {str(k): _file(root, v) for k, v in history_files.items()}
    seen = [_day(explored_through_day)]
    for item in history.values():
        day = _last_history_day(Path(item["path"]))
        if day:
            seen.append(day)
    seen_through = max(seen)
    frozen = now_utc()
    start = (pd.Timestamp(seen_through)+pd.Timedelta(days=1)).date().isoformat()
    while _origin(start) <= frozen:
        start = (pd.Timestamp(start)+pd.Timedelta(days=1)).date().isoformat()
    if requested_start_day is not None:
        requested = _day(requested_start_day)
        if requested < start:
            raise LedgerError("Requested start includes a seen day or an origin preceding the real candidate freeze.")
        start = requested
    manifest = {"schema_version": 1, "kind": "nyx_stress_guard_prospective_ledger", "created_at_utc": frozen,
        "zones": sorted(zones), "issue_policy": issue_policy, "information_cutoff_local": "08:00",
        "internal_pre_observation_deadline_local": "11:45", "deadline_is_market_publication_time": False,
        "explored_through_day": seen_through, "start_day": start, "evaluation_days": 365,
        "candidate_files": candidates, "history_files": history, "target_receipt_max_age_seconds": MAX_RECEIPT_AGE_SECONDS,
        "observation_revision_policy": "first_complete_country_day_is_sealed_later_conflicts_rejected", **FLAGS}
    _verify_files(root, candidates); _verify_files(root, history)
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"events").mkdir(); (destination/"artifacts").mkdir(); (destination/"heads").mkdir()
    _write_json(destination/"manifest.json", manifest)
    _write_json(destination/"manifest_seal.json", {"manifest_sha256": _digest(destination/"manifest.json")})
    return destination


def _read(ledger_directory, root):
    directory = _path(root, ledger_directory, output=True)
    if json.loads((directory/"manifest_seal.json").read_text(encoding="utf8")).get("manifest_sha256") != _digest(directory/"manifest.json"):
        raise LedgerError("Frozen ledger manifest changed.")
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf8"))
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "nyx_stress_guard_prospective_ledger"
        or manifest.get("issue_policy") not in GRADES or any(manifest.get(k) != v for k, v in FLAGS.items())
        or manifest.get("information_cutoff_local") != "08:00" or manifest.get("internal_pre_observation_deadline_local") != "11:45"
        or _day(manifest["start_day"]) <= _day(manifest["explored_through_day"])
        or _origin(manifest["start_day"]) <= _utc(manifest["created_at_utc"])):
        raise LedgerError("Invalid prospective freeze contract.")
    seal, previous, events = _digest(directory/"manifest.json"), "", []
    for sequence, path in enumerate(sorted((directory/"events").glob("*.json")), start=1):
        if path.name != f"{sequence:08d}.json":
            raise LedgerError("Missing or reordered append-only journal event.")
        event = json.loads(path.read_text(encoding="utf8"))
        saved = event.pop("event_sha256", None)
        if (saved != hashlib.sha256(_bytes(event)).hexdigest() or event.get("previous_event_sha256") != previous
            or event.get("sequence") != sequence or event.get("ledger_manifest_sha256") != seal):
            raise LedgerError("Journal hash chain or freeze identity changed.")
        event["event_sha256"] = saved
        _verify_files(root, event["artifacts"])
        if event.get("kind") not in {"forecast", "observation_resolution"}:
            raise LedgerError("Unknown append-only event type.")
        events.append(event); previous = saved
    heads = sorted((directory/"heads").glob("*.json"))
    if len(heads) != len(events):
        raise LedgerError("Journal tail/checkpoint count differs; incomplete commit or removed evidence.")
    for sequence, (path, event) in enumerate(zip(heads, events), start=1):
        if path.name != f"{sequence:08d}.json" or json.loads(path.read_text(encoding="utf8")) != {"sequence": sequence, "event_sha256": event["event_sha256"]}:
            raise LedgerError("Append-only journal head checkpoint changed.")
    return directory, manifest, events


def _append(directory, events, event):
    record = {**event, "sequence": len(events)+1, "previous_event_sha256": events[-1]["event_sha256"] if events else "",
        "ledger_manifest_sha256": _digest(directory/"manifest.json")}
    record["event_sha256"] = hashlib.sha256(_bytes(record)).hexdigest()
    _write_json(directory/"events"/f"{record['sequence']:08d}.json", record)
    _write_json(directory/"heads"/f"{record['sequence']:08d}.json", {"sequence": record["sequence"], "event_sha256": record["event_sha256"]})
    events.append(record)
    return record


def _frame(panel, manifest):
    if not isinstance(panel, pd.DataFrame) or panel.empty or panel.columns.has_duplicates:
        raise LedgerError("A nonempty unique-column input panel is required.")
    required = {*KEYS, "forecast_origin_utc", "forecast", "q10", "q90"}
    if not required.issubset(panel):
        raise LedgerError("Input panel lacks the saved baseline and forecast identities.")
    frame = panel.copy(deep=True)
    for col in ("timestamp_utc", "forecast_origin_utc"):
        frame[col] = pd.to_datetime([_utc(v, col) for v in frame[col]], utc=True)
    if not set(frame.zone).issubset(ZONES) or set(frame.zone) != set(manifest["zones"]) or frame.duplicated(KEYS).any():
        raise LedgerError("Countries or physical delivery identities are incomplete/duplicated.")
    days = frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    if days.nunique() != 1:
        raise LedgerError("Issue exactly one complete common delivery day.")
    day = days.iloc[0]
    if day < manifest["start_day"] or day > (pd.Timestamp(manifest["start_day"])+pd.Timedelta(days=364)).date().isoformat():
        raise LedgerError("Delivery is outside the new, predeclared 365-day prospective window.")
    if not frame.forecast_origin_utc.eq(_origin(day)).all():
        raise LedgerError("Every input row must preserve the civil D-1 08:00 information origin.")
    for zone, block in frame.groupby("zone"):
        if not pd.DatetimeIndex(block.timestamp_utc.sort_values()).equals(_index(day)):
            raise LedgerError(f"{zone}: exact 23/24/25-hour DST delivery frame required.")
    label_columns = [column for column in frame if str(column).lower() in {"actual", "target", "observed", "realised", "realized", "label"}
        or str(column).lower().startswith(("actual_", "target_", "observed_", "realised_", "realized_"))]
    if any(frame[column].notna().any() for column in label_columns):
        raise LedgerError("Input already contains observations/targets; retrospective issuance is forbidden.")
    baseline = frame[["q10", "forecast", "q90"]].to_numpy(float)
    if not np.isfinite(baseline).all() or (np.diff(baseline, axis=1) < 0).any():
        raise LedgerError("Finite ordered baseline P10/P50/P90 required.")
    return frame.sort_values(KEYS).reset_index(drop=True), day


def _check_window(day, manifest, stamp):
    current = _utc(stamp)
    grade = manifest["issue_policy"]
    start = (pd.Timestamp(day)-pd.Timedelta(days=1)).tz_localize("Europe/Paris").tz_convert("UTC")
    if grade == "pre_observation_asof08":
        start = _origin(day)
    deadline = _deadline(day, grade)
    valid_end = current <= deadline if grade == "strict_08_issue" else current < deadline
    if current < start or not valid_end or current <= _utc(manifest["created_at_utc"]):
        raise LedgerError("Real wall-clock issue is outside its predeclared grade window; no backdating.")


def validate_input_snapshot(panel, baseline_evidence, feature_evidence, label_check_evidence, ledger_manifest):
    """Pure compatibility preflight; the register separately verifies actual files and clocks."""
    frame, day = _frame(panel, ledger_manifest)
    return {"delivery_day": day, "rows": len(frame), "zones": ledger_manifest["zones"],
        "baseline_evidence_provided": bool(baseline_evidence), "feature_evidence_provided": bool(feature_evidence),
        "label_check_evidence_provided": bool(label_check_evidence), "publication_evidence_created": False}


def _input_evidence(root, evidence, day, current, *, panel=None, manifest=None):
    if not isinstance(evidence, Mapping) or evidence.get("schema_version") != 1:
        raise LedgerError("Explicit input capture evidence required.")
    if _utc(evidence["information_cutoff_utc"]) != _origin(day):
        raise LedgerError("Input information cutoff differs from D-1 08:00.")
    captured = _utc(evidence["captured_at_utc"])
    if captured > current:
        raise LedgerError("Input capture cannot postdate issuance.")
    if evidence.get("publication_evidence") not in {"historical_asof", "prospective_capture"}:
        raise LedgerError("Do not invent source-publication certification.")
    times = evidence.get("source_information_times_utc")
    if not isinstance(times, Mapping) or not times or any(_utc(v) > min(_origin(day), captured) for v in times.values()):
        raise LedgerError("Source information is missing or later than cutoff/capture.")
    artifacts = evidence.get("artifacts", {})
    if not {"baseline", "features", "panel"}.issubset(artifacts):
        raise LedgerError("Hash-pinned baseline, feature and complete panel artifacts required.")
    _verify_files(root, artifacts)
    if panel is not None:
        saved, saved_day = _frame(pd.read_parquet(artifacts["panel"]["path"]), manifest)
        if saved_day != day:
            raise LedgerError("Captured panel delivery differs from the issue request.")
        try:
            pd.testing.assert_frame_equal(saved, panel, check_exact=True)
        except AssertionError as exc:
            raise LedgerError("In-memory inputs differ from the hash-pinned captured panel.") from exc


def _target_sources(receipt, zones):
    sources = receipt.get("target_sources")
    if not isinstance(sources, Mapping) or set(sources) != set(zones):
        raise LedgerError("Canonical target source identities required for every country.")
    for source in sources.values():
        if (not isinstance(source, Mapping) or not isinstance(source.get("series"), str) or not source["series"]
            or not re.fullmatch(r"[a-f0-9]{64}", str(source.get("canonical_cache_sha256", "")))):
            raise LedgerError("Target series identity or cache hash is invalid.")


def _fresh_receipt(receipt, day, zones, started, completed):
    if (not isinstance(receipt, Mapping) or receipt.get("schema_version") != 1
        or receipt.get("kind") != "fresh_canonical_target_check" or receipt.get("delivery_day") != day
        or receipt.get("zones") != zones or receipt.get("fresh_api_read") is not True or receipt.get("nocache") is not True):
        raise LedgerError("A fresh uncached canonical-target receipt is required before and after inference.")
    first, received = _utc(receipt["started_at_utc"]), _utc(receipt["received_at_utc"])
    if not started <= first <= received <= completed:
        raise LedgerError("Target receipt is not bracketed by the real check invocation.")
    counts = receipt.get("observed_hours_by_zone")
    if not isinstance(counts, Mapping) or set(counts) != set(zones) or any(type(v) is not int or v != 0 for v in counts.values()):
        raise LedgerError("A delivery observation is already known; no prospective issue is allowed.")
    _target_sources(receipt, zones)


def issue_forecast(ledger_directory, *, root, panel: pd.DataFrame, input_evidence: dict,
                   predictor: Callable[[pd.DataFrame, dict], pd.DataFrame],
                   fresh_label_check: Callable[[str, list[str]], dict]) -> dict:
    """Register a genuinely new pre-observation forecast using two fresh checks."""
    directory = _path(root, ledger_directory, output=True)
    with exclusive_process_lock(directory/"ledger.lock"):
        directory, manifest, events = _read(directory, root)
        frame, day = _frame(panel, manifest)
        identity = _frame_digest(frame)
        old = [e for e in events if e["kind"] == "forecast" and e["delivery_day"] == day]
        if old:
            if len(old) != 1 or old[0]["input_frame_sha256"] != identity:
                raise LedgerError("Delivery already issued with different immutable inputs.")
            return {"status": "already_issued", "event_sha256": old[0]["event_sha256"], "delivery_day": day, **FLAGS}
        _check_window(day, manifest, now_utc())
        _verify_files(root, manifest["candidate_files"]); _verify_files(root, manifest["history_files"])
        _input_evidence(root, input_evidence, day, now_utc(), panel=frame, manifest=manifest)
        before_start = now_utc(); before = fresh_label_check(day, list(manifest["zones"])); before_end = now_utc()
        _fresh_receipt(before, day, manifest["zones"], before_start, before_end)
        prediction_started = now_utc()
        predicted = predictor(frame.copy(deep=True), dict(manifest))
        prediction_completed = now_utc()
        if not isinstance(predicted, pd.DataFrame):
            raise LedgerError("Predictor must return a DataFrame.")
        result, result_day = _frame(predicted, manifest)
        if result_day != day or not set(frame).issubset(result):
            raise LedgerError("Predictor removed or changed the frozen input frame.")
        try:
            pd.testing.assert_frame_equal(result[list(frame)], frame, check_exact=True)
        except AssertionError as exc:
            raise LedgerError("Predictor changed saved baseline/features/identities.") from exc
        qcols = ["candidate_q10", "candidate_forecast", "candidate_q90"]
        if not set(qcols).issubset(result):
            raise LedgerError("Candidate P10/P50/P90 are required.")
        quantiles = result[qcols].to_numpy(float)
        if not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
            raise LedgerError("Candidate quantiles must be finite and ordered.")
        after_start = now_utc(); after = fresh_label_check(day, list(manifest["zones"])); after_end = now_utc()
        _fresh_receipt(after, day, manifest["zones"], after_start, after_end)
        _verify_files(root, manifest["candidate_files"]); _verify_files(root, manifest["history_files"])
        _input_evidence(root, input_evidence, day, after_end, panel=frame, manifest=manifest)
        attempt = directory/"artifacts"/uuid.uuid4().hex
        attempt.mkdir(exist_ok=False)
        _write_frame(attempt/"input.parquet", frame)
        _write_frame(attempt/"predictions.parquet", result)
        _write_json(attempt/"receipts.json", {"before": before, "after": after, "input_evidence": input_evidence})
        artifacts = {key: _file(root, attempt/name) for key, name in
            (("input", "input.parquet"), ("predictions", "predictions.parquet"), ("receipts", "receipts.json"))}
        issued = now_utc()
        _check_window(day, manifest, issued)
        if not before_end <= prediction_started <= prediction_completed <= after_start <= after_end <= issued:
            raise LedgerError("Non-monotone inference and fresh-observation checks.")
        if any((issued-_utc(r["received_at_utc"])).total_seconds() > MAX_RECEIPT_AGE_SECONDS for r in (before, after)):
            raise LedgerError("Fresh target check expired before publication; rerun safely within the issue window.")
        event = _append(directory, events, {"kind": "forecast", "delivery_day": day, "zones": manifest["zones"],
            "issue_grade": manifest["issue_policy"], "issued_at_utc": issued, "forecast_origin_utc": _origin(day),
            "inference_started_at_utc": prediction_started, "inference_completed_at_utc": prediction_completed,
            "input_frame_sha256": identity, "source_publication_evidence": input_evidence["publication_evidence"],
            "source_publication_certified_by_ledger": False,
            "artifacts": artifacts, **FLAGS})
        return {"status": "issued", "delivery_day": day, "issue_grade": manifest["issue_policy"],
            "issued_at_utc": issued.isoformat(), "event_sha256": event["event_sha256"], **FLAGS}


def resolve_observations(ledger_directory, *, root, observations: pd.DataFrame, observation_evidence: dict) -> dict:
    """Attach only complete new canonical country-days; never rerun a forecast."""
    directory = _path(root, ledger_directory, output=True)
    with exclusive_process_lock(directory/"ledger.lock"):
        directory, manifest, events = _read(directory, root)
        if (not isinstance(observation_evidence, Mapping) or observation_evidence.get("schema_version") != 1
            or observation_evidence.get("kind") != "canonical_target_observations" or observation_evidence.get("fresh_api_read") is not True):
            raise LedgerError("Fresh canonical observation receipt required.")
        received = _utc(observation_evidence["received_at_utc"])
        if received > now_utc():
            raise LedgerError("Observation receipt cannot be in the future.")
        _target_sources(observation_evidence, manifest["zones"])
        if not isinstance(observations, pd.DataFrame) or not {*KEYS, "actual"}.issubset(observations):
            raise LedgerError("Observation frame requires zone, timestamp_utc and actual.")
        frame = observations[[*KEYS, "actual"]].copy(deep=True)
        frame["timestamp_utc"] = pd.to_datetime([_utc(v) for v in frame.timestamp_utc], utc=True)
        if frame.duplicated(KEYS).any() or set(frame.zone)-set(manifest["zones"]):
            raise LedgerError("Duplicated or foreign observation identities.")
        frame["actual"] = pd.to_numeric(frame.actual, errors="raise")
        if np.isinf(frame.actual.to_numpy(float)).any():
            raise LedgerError("Infinite observed prices are invalid.")
        frame["delivery_day"] = frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
        issues = {e["delivery_day"]: e for e in events if e["kind"] == "forecast"}
        resolutions = {(e["delivery_day"], e["zone"]): e for e in events if e["kind"] == "observation_resolution"}
        plans, pending = [], []
        for (day, zone), group in frame.groupby(["delivery_day", "zone"], sort=True):
            if day not in issues:
                raise LedgerError("No pre-existing sealed forecast for this observation day; retrospective results are forbidden.")
            if received <= _utc(issues[day]["issued_at_utc"]):
                raise LedgerError("Observation receipt must strictly postdate the sealed forecast.")
            block = group.drop(columns="delivery_day").sort_values(KEYS).reset_index(drop=True)
            if not pd.DatetimeIndex(block.timestamp_utc).equals(_index(day)) or block.actual.isna().any():
                pending.append({"delivery_day": day, "zone": zone, "observed_hours": int(block.actual.notna().sum()), "expected_hours": len(_index(day))})
                continue
            prior = resolutions.get((day, zone))
            if prior:
                saved = pd.read_parquet(prior["artifacts"]["observations"]["path"])
                try:
                    pd.testing.assert_frame_equal(saved, block, check_exact=True)
                except AssertionError as exc:
                    raise LedgerError("Observed prices changed after first complete resolution; prior evidence is preserved.") from exc
                continue
            plans.append((day, zone, block, issues[day]))
        appended = []
        for day, zone, block, issue in plans:
            attempt = directory/"artifacts"/uuid.uuid4().hex; attempt.mkdir(exist_ok=False)
            _write_frame(attempt/"observations.parquet", block)
            _write_json(attempt/"observation_receipt.json", observation_evidence)
            event = _append(directory, events, {"kind": "observation_resolution", "zone": zone, "delivery_day": day,
                "issue_grade": issue["issue_grade"], "forecast_event_sha256": issue["event_sha256"],
                "observations_received_at_utc": received, "registered_at_utc": now_utc(),
                "artifacts": {"observations": _file(root, attempt/"observations.parquet"), "receipt": _file(root, attempt/"observation_receipt.json")}, **FLAGS})
            appended.append(event["event_sha256"])
        return {"status": "waiting_for_observations" if pending else "resolved" if appended else "no_new_observations",
            "new_complete_country_days": len(appended), "pending": pending, "event_sha256": appended,
            "forecast_recalculated": False, **FLAGS}


def ledger_status(ledger_directory, *, root) -> dict:
    """Read separate grade/country results without loading a model or changing evidence."""
    directory, manifest, events = _read(ledger_directory, root)
    issues = {e["event_sha256"]: e for e in events if e["kind"] == "forecast"}
    scored = {grade: {} for grade in GRADES}
    resolved = set()
    for event in events:
        if event["kind"] != "observation_resolution":
            continue
        issue = issues.get(event["forecast_event_sha256"])
        if issue is None or issue["delivery_day"] != event["delivery_day"] or issue["issue_grade"] != event["issue_grade"]:
            raise LedgerError("Resolution references an inconsistent forecast event.")
        if _utc(event["observations_received_at_utc"]) <= _utc(issue["issued_at_utc"]):
            raise LedgerError("Resolution is not strictly later than the corresponding forecast.")
        prediction = pd.read_parquet(issue["artifacts"]["predictions"]["path"])
        observed = pd.read_parquet(event["artifacts"]["observations"]["path"])
        block = prediction.loc[prediction.zone.eq(event["zone"])].drop(columns="actual", errors="ignore").merge(observed, on=KEYS, validate="one_to_one")
        block["delivery_day"] = event["delivery_day"]
        scored[event["issue_grade"]].setdefault(event["zone"], []).append(block)
        resolved.add((event["delivery_day"], event["zone"]))
    results = {grade: {} for grade in GRADES}
    for grade, countries in scored.items():
        for zone, blocks in countries.items():
            frame = pd.concat(blocks, ignore_index=True)
            error, baseline_error = frame.candidate_forecast-frame.actual, frame.forecast-frame.actual
            results[grade][zone] = {"days": int(frame.delivery_day.nunique()), "hours": len(frame),
                "start_day": frame.delivery_day.min(), "end_day": frame.delivery_day.max(),
                "candidate_mae_eur_mwh": float(error.abs().mean()), "baseline_mae_eur_mwh": float(baseline_error.abs().mean()),
                "candidate_bias_eur_mwh": float(error.mean()),
                "coverage_p10_p90": float(((frame.actual >= frame.candidate_q10) & (frame.actual <= frame.candidate_q90)).mean()),
                "mean_interval_width_eur_mwh": float((frame.candidate_q90-frame.candidate_q10).mean()),
                "independent_annual_validation_complete": frame.delivery_day.nunique() == 365}
    current, emitted = now_utc(), {e["delivery_day"] for e in issues.values()}
    missed = []
    for stamp in pd.date_range(manifest["start_day"], periods=365):
        day = stamp.date().isoformat()
        deadline = _deadline(day, manifest["issue_policy"])
        expired = deadline < current if manifest["issue_policy"] == "strict_08_issue" else deadline <= current
        if expired and day not in emitted:
            missed.append(day)
    pending = [{"delivery_day": day, "zone": zone} for day in sorted(emitted) for zone in manifest["zones"] if (day, zone) not in resolved]
    return {"status": "evaluated" if resolved else "waiting_for_observations" if emitted else "armed_no_forecasts",
        "ledger": str(directory), "start_day": manifest["start_day"], "explored_through_day": manifest["explored_through_day"],
        "issue_policy": manifest["issue_policy"], "forecast_days": len(emitted), "complete_country_days": len(resolved),
        "pending_observations": pending, "missed_emission_days": missed, "evaluation_by_grade_and_zone": results,
        "journal_events": len(events), "journal_head_sha256": events[-1]["event_sha256"] if events else None,
        "forecast_recalculated": False, "models_loaded_for_evaluation": False, **FLAGS}


__all__ = ["LedgerError", "freeze_ledger", "validate_input_snapshot", "issue_forecast", "resolve_observations", "ledger_status"]
