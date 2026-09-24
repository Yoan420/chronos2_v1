"""Bounded Saturn suffix refresh into brand-new challenger-owned copies only.

Operational caches and existing experiments are read-only inputs. Every call
owns a unique snapshot. Its saved config can seed a later, separate snapshot.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from threading import Event, Lock
import uuid

import pandas as pd

from economic_value.data import _aware, _day, _stable_bytes, load_report_panel
from materialize_saturn_kalman_weather import SeriesPlan, build_command, build_plan
from .data import ScarcityDataError, ZONES, _path, _read_source, source_registry


REFRESH_KEYS = tuple([f"{z.lower()}_{technology}_generation" for z in ZONES for technology in ("wind", "solar")]
                     + [f"{z.lower()}_gas_available" for z in ZONES]
                     + ["de_coal_available", "de_lignite_available", "nl_coal_available",
                        "be_nuclear_available", "nl_nuclear_available"])


def _snapshot_root(root: Path) -> Path:
    base = root / "runs" / "experiments" / "nyx_scarcity_v1" / "source_refresh"
    if base.resolve() != base or not base.is_relative_to(root):
        raise ScarcityDataError("The isolated refresh namespace must not redirect through a junction/symlink.")
    return base


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _plans(directory: Path) -> dict[str, SeriesPlan]:
    plans = {p.alias.removesuffix("_fcst"): p for p in build_plan(ZONES, directory) if p.kind in {"wind", "solar"}}
    registry = source_registry()
    for key in REFRESH_KEYS:
        if not key.endswith("_available"):
            continue
        plans[key] = SeriesPlan(zone=key[:2].upper(), kind="capacity", alias=f"{key}_gw", series=registry[key]["series"],
            timezone="Europe/Paris", naive_timezone="Europe/Paris", output=directory / f"{key}_gw.parquet",
            daily_broadcast=True, value_scale=1., unit="GW")
    if set(plans) != set(REFRESH_KEYS):
        raise ScarcityDataError("Unexpected refresh source plan.")
    for key, plan in plans.items():
        if plan.series != registry[key]["series"]:
            raise ScarcityDataError(f"{key}: materializer/catalog identity mismatch.")
    return plans


def _integer(settings: dict, name: str, default: int, maximum: int) -> int:
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ScarcityDataError(f"refresh.{name} must be an integer from 1 to {maximum}.")
    return value


def _transport_failure(log: Path) -> str | None:
    """Classify explicit transport/authentication failures, not missing series.

    Only a bounded tail is inspected. The audit gets a category and log path,
    never copies URLs, credentials or an arbitrary exception payload.
    """
    if not log.is_file():
        return None
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 65536))
        tail = stream.read().decode("utf-8", errors="replace")
    patterns = (
        ("proxy", r"ProxyError|ProxyConnectError|Unable to connect to proxy|407\s+(?:Client Error|Proxy Authentication)"),
        ("tls", r"SSLError|SSLCertVerificationError|CERTIFICATE_VERIFY_FAILED|certificate verify failed"),
        ("authentication", r"(?:401|403)\s+(?:Client Error|Unauthorized|Forbidden)|AuthenticationError|InvalidCredentials|HTTP(?:Error|StatusError).{0,60}\b(?:401|403)\b"),
        ("network", r"ConnectTimeout|ReadTimeout|ConnectionRefusedError|NameResolutionError|NewConnectionError|RemoteProtocolError|WinError\s+(?:10061|10060|11001)"),
    )
    for category, pattern in patterns:
        if re.search(pattern, tail, flags=re.IGNORECASE):
            return category
    return None


def refresh_sources(config: dict, *, root: Path) -> dict:
    """Suffix only, with logs, bounded retries and original-source hash checks.

    Partial snapshots are never marked ready. Missing required features still
    force baseline fallback if their saved configuration is subsequently used.
    """
    root = Path(root).resolve()
    settings = config.get("refresh", {})
    if not isinstance(settings, dict) or set(settings) - {"workers", "max_suffix_days", "retries", "request_timeout_seconds"}:
        raise ScarcityDataError("Unknown refresh option; arbitrary series or commands are forbidden.")
    workers = _integer(settings, "workers", 2, 2)
    max_days = _integer(settings, "max_suffix_days", 30, 60)
    retries = _integer(settings, "retries", 2, 3)
    request_timeout = _integer(settings, "request_timeout_seconds", 45, 120)
    chosen = config.get("delivery_day")
    if chosen is None:
        _, audit = load_report_panel(root, config.get("zones", list(ZONES)), ["nuclear_kalman"], end_day=config.get("end_day"))
        chosen = audit["delivery_day"]
    chosen = _day(chosen)
    final = pd.Timestamp(chosen)
    registry = source_registry()
    overrides = config.get("data", {}).get("source_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - set(registry):
        raise ScarcityDataError("Refresh source overrides must use the exact source registry.")
    base = _snapshot_root(root)
    seeds = {}
    # All seeds must validate before a new directory or command is created.
    for key in REFRESH_KEYS:
        spec = {**registry[key], "path": overrides.get(key, registry[key]["path"])}
        path = _path(root, spec["path"])
        raw, evidence = _stable_bytes(path)
        sidecar = Path(str(path) + ".audit.json")
        audit_raw, audit_evidence = _stable_bytes(sidecar)
        frame = pd.read_parquet(BytesIO(raw))
        expected = _aware(frame.value_time_utc, key + "/seed").unique().sort_values()
        selected, selection_audit = _read_source(root, key, spec, expected)
        if selection_audit.get("sha256") != evidence["sha256"]:
            raise ScarcityDataError(f"{key}: source changed between seed capture and validation.")
        finite = selected[spec["feature"]].dropna()
        if finite.empty:
            raise ScarcityDataError(f"{key}: no validated historical seed; full backfill is outside Refresh scope.")
        last = pd.Timestamp(finite.index.max().tz_convert("Europe/Paris").date())
        last_hours = pd.date_range(last.tz_localize("Europe/Paris"), (last + pd.Timedelta(days=1)).tz_localize("Europe/Paris"),
                                   freq="h", inclusive="left").tz_convert("UTC")
        if finite.reindex(last_hours).isna().any():
            raise ScarcityDataError(f"{key}: the last seed day is incomplete; suffix refresh cannot silently repair an existing day.")
        start = last + pd.Timedelta(days=1)
        requested = max(0, (final - start).days + 1)
        if requested > max_days:
            raise ScarcityDataError(f"{key}: suffix {requested} days exceeds refresh limit {max_days}; refusing full-history download.")
        seeds[key] = {"bytes": raw, "audit_bytes": audit_raw, "spec": spec, "evidence": evidence,
                      "audit_evidence": audit_evidence, "selection_audit": selection_audit,
                      "audit": json.loads(audit_raw), "start": start, "requested_days": requested,
                      "first_day": str(finite.index.min().tz_convert("Europe/Paris").date()), "last_day": str(last.date())}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    directory = base / stamp
    directory.mkdir(parents=True, exist_ok=False)
    plans = _plans(directory)
    copied = deepcopy(config)
    copied["delivery_day"] = chosen
    copied.setdefault("data", {}).setdefault("source_overrides", {})
    for key, seed in seeds.items():
        path = plans[key].output
        path.write_bytes(seed["bytes"])
        Path(str(path) + ".audit.json").write_bytes(seed["audit_bytes"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != seed["evidence"]["sha256"]:
            raise ScarcityDataError(f"{key}: snapshot copy checksum mismatch.")
        copied["data"]["source_overrides"][key] = str(path.relative_to(root))
    saved_config = directory / "config.json"
    audit_path = directory / "refresh_audit.json"
    _write_json(saved_config, copied)
    audit = {"schema_version": 1, "status": "running", "delivery_day": chosen, "source_dir": str(directory),
             "saved_config": str(saved_config), "required_sources_complete": False, "sources": {},
             "original_sources_modified": False, "historical_missing_days_repaired": False,
             "cutoff": "D-1 08:00 civil", "max_processes": workers, "day_workers_per_process": 1,
             "diagnostic_only": True, "production_pit_evidence": False, "promotion_eligible": False}
    _write_json(audit_path, audit)
    stop_requests = Event()
    stop_lock = Lock()
    transport_failure = {}

    def run_one(key: str) -> tuple[str, dict]:
        seed, plan = seeds[key], plans[key]
        log = directory / f"{key}.log"
        record = {"series": plan.series, "seed": seed["evidence"], "seed_audit": seed["audit_evidence"],
                  "source_path": str(plan.output), "requested_start_day": str(seed["start"].date()),
                  "requested_end_day": chosen, "requested_days": seed["requested_days"], "log": str(log)}
        if stop_requests.is_set():
            record.update(status="skipped_transport_failure", command=[],
                          error="No query launched after a prior global transport/authentication failure.")
            return key, record
        try:
            if seed["requested_days"]:
                command = build_command(plan, start_day=seed["start"], end_day=final, day_workers=1, merge_existing=True)
                command[1] = str(root / "materialize_saturn_daily_asof.py")
                command += ["--retries", str(retries), "--request-timeout-seconds", str(request_timeout), "--allow-incomplete-days"]
                record["command"] = command
                timeout = seed["requested_days"] * (request_timeout * retries + 15) + 90
                if stop_requests.is_set():
                    record.update(status="skipped_transport_failure", command=[],
                                  error="No query launched after a prior global transport/authentication failure.")
                    return key, record
                with log.open("w", encoding="utf-8") as stream:
                    completed = subprocess.run(command, cwd=root, shell=False, stdout=stream, stderr=subprocess.STDOUT,
                                               timeout=timeout, check=False)
                record["returncode"] = completed.returncode
                if completed.returncode != 0:
                    raise ScarcityDataError(f"Materializer returned {completed.returncode}; see source log.")
            else:
                record["command"] = []
                record["returncode"] = 0
            # Only the NEW sidecar is amended, preserving the original scope.
            sidecar = Path(str(plan.output) + ".audit.json")
            output_audit = json.loads(sidecar.read_text(encoding="utf-8"))
            output_audit["unit"] = plan.unit
            output_audit["production_pit_evidence"] = False
            output_audit["scarcity_refresh"] = {"seed": seed["evidence"], "seed_audit": seed["audit_evidence"],
                "original_series_unmodified": True, "requested_suffix_only": True,
                "seed_approximation": seed["audit"].get("approximation"),
                "seed_fill_or_interpolation": seed["audit"].get("fill_or_interpolation"),
                "seed_missing_days": seed["audit"].get("missing_days", [])}
            if plan.kind == "capacity":
                output_audit.update(capacity_scope="provider technology/fuel available-Pmax aggregate; national fleet coverage not attested",
                    national_coverage_attested=False, information_type="capacity_forecast",
                    approximation="Daily available Pmax broadcast to physical hours, not generation; national coverage unverified")
                if "_gas_" in key:
                    output_audit["do_not_add"] = ["type.ccgt", "type.gt", "type.chp"]
            _write_json(sidecar, output_audit)
            seed_frame = pd.read_parquet(BytesIO(seed["bytes"]))
            output_frame = pd.read_parquet(plan.output)
            identities = ["value_time_utc", "snapshot_time_utc", "revision_time_utc"]
            old = seed_frame[identities + ["value"]].drop_duplicates().sort_values(identities).reset_index(drop=True)
            prefix = output_frame.loc[pd.to_datetime(output_frame.value_time_utc, utc=True).isin(
                pd.to_datetime(seed_frame.value_time_utc, utc=True)), identities + ["value"]]
            prefix = prefix.drop_duplicates().sort_values(identities).reset_index(drop=True)
            pd.testing.assert_frame_equal(old, prefix, check_dtype=False, check_exact=True)
            record["seed_history_semantically_unchanged"] = True
            start = pd.Timestamp(seed["first_day"]).tz_localize("Europe/Paris")
            stop = (final + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
            expected = pd.date_range(start, stop, inclusive="left", freq="h").tz_convert("UTC")
            spec = {**registry[key], "path": str(plan.output.relative_to(root))}
            values, validated = _read_source(root, key, spec, expected)
            suffix = expected.tz_convert("Europe/Paris").tz_localize(None).normalize() >= seed["start"]
            record["validation"] = validated
            record["missing_suffix_hours"] = int(values.loc[suffix, spec["feature"]].isna().sum())
            end_hours = expected.tz_convert("Europe/Paris").strftime("%Y-%m-%d") == chosen
            record["missing_requested_delivery_hours"] = int(values.loc[end_hours, spec["feature"]].isna().sum())
            record["status"] = "complete" if record["missing_suffix_hours"] == record["missing_requested_delivery_hours"] == 0 else "partial"
        except (OSError, ValueError, AssertionError, subprocess.SubprocessError) as exc:
            record.update(status="failed", error=str(exc))
            category = _transport_failure(log)
            if category:
                record["global_transport_failure"] = category
                with stop_lock:
                    if not stop_requests.is_set():
                        transport_failure.update(category=category, source=key, log=str(log))
                        stop_requests.set()
        return key, record

    pool = ThreadPoolExecutor(max_workers=workers)
    interrupted = False
    futures = {}
    try:
        futures = {pool.submit(run_one, key): key for key in REFRESH_KEYS}
        for future in as_completed(futures):
            key, record = future.result()
            audit["sources"][key] = record
            _write_json(audit_path, audit)
            print(f"[Scarcity Refresh] {key}: {record['status']} ({record['requested_days']} suffix days)", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        stop_requests.set()
        for future, key in futures.items():
            if key not in audit["sources"]:
                audit["sources"][key] = {"status": "cancelled_before_launch" if future.cancel() else "in_flight_at_interruption",
                                        "source_path": str(plans[key].output)}
        audit.update(status="interrupted", required_sources_complete=False,
                     completed_at_utc=datetime.now(timezone.utc).isoformat(),
                     interruption="Pending jobs cancelled; at most two already-started bounded child calls may finish. Snapshot is not ready.")
        _write_json(audit_path, audit)
        raise
    finally:
        pool.shutdown(wait=not interrupted, cancel_futures=interrupted)
    for seed in seeds.values():
        for evidence in (seed["evidence"], seed["audit_evidence"]):
            if hashlib.sha256(Path(evidence["path"]).read_bytes()).hexdigest() != evidence["sha256"]:
                audit.update(status="failed", error="An original source changed concurrently during refresh; snapshot not approved.")
                _write_json(audit_path, audit)
                raise ScarcityDataError(audit["error"])
    statuses = [r["status"] for r in audit["sources"].values()]
    audit["required_sources_complete"] = all(s == "complete" for s in statuses)
    audit["status"] = "complete" if audit["required_sources_complete"] else "partial" if any(s in {"complete", "partial"} for s in statuses) else "failed"
    if transport_failure:
        audit["global_transport_failure"] = transport_failure
        audit["transport_fail_fast"] = True
        audit["queries_skipped_after_transport_failure"] = statuses.count("skipped_transport_failure")
    audit["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(audit_path, audit)
    return {**audit, "config": copied, "audit_path": str(audit_path)}


__all__ = ["refresh_sources", "REFRESH_KEYS"]
