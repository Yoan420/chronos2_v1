"""Create a new private report from already saved forecasts, never re-fit."""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

from .metrics import compute_kpis
from .economic import compute_economic_kpis
from .render import render_kpi


NAMESPACE = Path("runs/reports/kpi")
LOGGER = logging.getLogger(__name__)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_output(root, path):
    root = Path(root).resolve()
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("KPI: parent traversal is forbidden.")
    absolute = path if path.is_absolute() else root/path
    if not absolute.is_relative_to(root/NAMESPACE) or absolute == root/NAMESPACE:
        raise ValueError("KPI outputs must stay in runs/reports/kpi, away from models and operational reports.")
    for ancestor in (absolute, *absolute.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink() or (hasattr(ancestor, "is_junction") and ancestor.is_junction()):
            raise ValueError("KPI outputs cannot follow symbolic links or junctions.")
    if absolute.resolve() != absolute:
        raise ValueError("KPI outputs must not redirect outside their declared path.")
    return absolute


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _publish_latest(root, payload):
    target = safe_output(root, NAMESPACE/"latest.json")
    temporary = safe_output(root, NAMESPACE/("latest_"+uuid.uuid4().hex+".tmp"))
    _write_json(temporary, payload)
    os.replace(temporary, target)


def _coverage_by_zone(result):
    coverage = {r["zone"]: r for r in result["coverage"]}
    fields = ("n_expected_hours", "n_common_hours", "n_complete_days", "n_incomplete_days", "reference_hours")
    coverage["ALL"] = {key: sum(c[key] for c in coverage.values()) for key in fields}
    coverage["ALL"]["definition"] = "Physical country-hours and complete country-days summed across countries."
    return coverage


def produce_report(*, root, end_day=None, models=None, zones=None):
    from .data import load_recent_models, verify_sources
    root = Path(root).resolve()
    LOGGER.info("[KPI] Reading completed local snapshots and production provenance; no forecast or API.")
    frame, catalog, source_audit = load_recent_models(root)
    selected_models = [m["id"] for m in catalog] if models is None else list(models)
    selected_zones = source_audit["zones"] if zones is None else list(zones)
    if not selected_models or len(selected_models) != len(set(selected_models)) or set(selected_models)-{m["id"] for m in catalog}:
        raise ValueError("KPI: unknown, duplicate or empty model selection. Use KPI.ps1 -Action List.")
    if not selected_zones or len(selected_zones) != len(set(selected_zones)) or set(selected_zones)-set(source_audit["zones"]):
        raise ValueError("KPI: unknown, duplicate or empty country selection.")
    if not any(m["kind"] == "production" and m["id"] in selected_models for m in catalog):
        raise ValueError("KPI must retain the current production comparator.")
    end = source_audit["recommended_end_day"] if end_day is None else end_day
    if date.fromisoformat(end).isoformat() != end or end > source_audit["recommended_end_day"]:
        raise ValueError("KPI: end date must not exceed the last common evaluation day "+source_audit["recommended_end_day"]+". Live rows stay excluded.")
    catalog = [m for m in catalog if m["id"] in selected_models]
    economic_config = root/"config/economic_value.yaml"
    economic_sha = digest(economic_config)
    source_audit["economic_config"] = {"path": str(economic_config), "sha256": economic_sha}
    economic_dependencies = {str(path): digest(path) for path in
        (Path(__file__).resolve().parents[1]/"economic_value"/name for name in ("engine.py", "runner.py"))}
    source_audit["economic_dependencies"] = economic_dependencies
    # Narrow once before repeated validation/aggregation of the four periods.
    frame = frame.loc[frame.model_id.isin(selected_models) & frame.zone.isin(selected_zones)].copy()
    periods, full_metrics = {}, {}
    for days in (365, 90, 30, 7):
        LOGGER.info("[KPI] Computing %d-day common support through %s (%d models).", days, end, len(catalog))
        result = compute_kpis(frame, end_day=end, days=days, models=selected_models, zones=selected_zones)
        result["economic"] = compute_economic_kpis(frame, end_day=end, days=days,
            models=selected_models, zones=selected_zones, config_path=economic_config)
        full_metrics[str(days)] = result
        lean = {k: v for k, v in result.items() if k != "daily_rows"}
        lean["coverage"] = _coverage_by_zone(result)
        periods[str(days)] = lean
    if not any(r["n_hours"] for r in periods["365"]["rows"]):
        raise ValueError("KPI: no common evaluation hours. No misleading empty comparison was published.")
    verify_sources(source_audit)
    if digest(economic_config) != economic_sha:
        raise ValueError("KPI: economic assumptions changed during calculation. No report was published.")
    if any(digest(path) != sha for path, sha in economic_dependencies.items()):
        raise ValueError("KPI: economic calculation dependencies changed during calculation.")
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    directory = safe_output(root, NAMESPACE/"snapshots"/stamp)
    directory.mkdir(parents=True, exist_ok=False)
    payload = {"schema_version": 1, "title": "KPI", "catalog": catalog, "zones": selected_zones,
        "generated_at_utc": now.isoformat(), "generated_at_local": now.astimezone(ZoneInfo("Europe/Paris")).strftime("%d/%m/%Y %H:%M %Z"),
        "source_delivery_day": source_audit["source_delivery_day"], "periods": periods,
        "production_modified": False, "models_fitted": False, "forecasts_generated": False,
        "network_requested": False, "independent_validation": False}
    _write_json(directory/"source_audit.json", source_audit)
    _write_json(directory/"kpi_metrics.json", {**payload, "periods": full_metrics})
    report = render_kpi(payload, directory/"KPI.html")
    verify_sources(source_audit)
    if digest(economic_config) != economic_sha:
        raise ValueError("KPI: economic assumptions changed during generation. No latest report was published.")
    if any(digest(path) != sha for path, sha in economic_dependencies.items()):
        raise ValueError("KPI: economic calculation dependencies changed during generation.")
    files = {name: digest(directory/name) for name in ("KPI.html", "kpi_metrics.json", "source_audit.json")}
    code = [*sorted((root/"kpi_report").glob("*.py")), root/"KPI.ps1", root/"run_kpi_report.py"]
    manifest = {"schema_version": 1, "status": "completed", "report": str(report), "snapshot": str(directory),
        "generated_at_utc": now.isoformat(), "end_day": end, "models": selected_models,
        "zones": selected_zones, "files": files,
        "code_sha256": {p.relative_to(root).as_posix(): digest(p) for p in code if p.is_file()},
        "production_modified": False, "models_fitted": False, "forecasts_generated": False,
        "sources_unchanged_during_generation": True}
    _write_json(directory/"report_manifest.json", manifest)
    _publish_latest(root, {"snapshot": str(directory), "manifest_sha256": digest(directory/"report_manifest.json")})
    LOGGER.info("[KPI] Completed: %s", report)
    return manifest


def status(*, root):
    root = Path(root).resolve()
    latest = json.loads(safe_output(root, NAMESPACE/"latest.json").read_text(encoding="utf-8"))
    directory = safe_output(root, latest["snapshot"])
    if digest(directory/"report_manifest.json") != latest["manifest_sha256"]:
        raise ValueError("KPI manifest checksum differs from the published pointer.")
    manifest = json.loads((directory/"report_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or manifest.get("production_modified") is not False
        or set(manifest.get("files", {})) != {"KPI.html", "kpi_metrics.json", "source_audit.json"}
        or any(digest(directory/p) != sha for p, sha in manifest["files"].items())):
        raise ValueError("KPI report is incomplete or changed.")
    return {"status": "completed", "report": str(directory/"KPI.html"), "end_day": manifest["end_day"],
        "models": len(manifest["models"]), "zones": manifest["zones"], "production_modified": False,
        "report_hashes_verified": True}
