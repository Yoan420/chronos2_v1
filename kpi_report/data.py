"""Read-only adapters for the latest completed, sealed NYX research families.

No fitting, refresh, pickle loading, production mutation, or rescoring with a
different observation vintage is performed here. Historical prices remain
readable when the modelling runtime changes: file identities, not today's ML
installation, are the contract for this retrospective KPI report.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import numpy as np
import pandas as pd


class KPIDataError(ValueError):
    pass


NAMESPACE = Path("runs/experiments/nyx_scarcity_v1")
ZONES = ("BE", "DE", "FR", "NL")
FAMILIES = ("fundamental", "coherent_p50", "stress_guard")
KEYS = ["zone", "timestamp_utc"]
BASE_COLUMNS = [*KEYS, "forecast_origin_utc", "sample", "forecast", "actual", "benchmark_forecast"]
SNAPSHOT_NAME = re.compile(r"^\d{8}T\d{6}Z_[0-9a-f]{8}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MODEL_SPECS = (
    ("fundamental", "fundamental_governed", "Fondamental physique · gouverné", "fundamental/predictions.parquet", "gouverné", "experiment"),
    ("fundamental", "fundamental_fixed25", "Fondamental physique · poids fixe 25 %", "fundamental/proposals_25.parquet", "fixe 25 %", "experiment"),
    ("fundamental", "calendar_governed", "Calendrier seul · gouverné (témoin)", "calendar/predictions.parquet", "gouverné", "control"),
    ("fundamental", "calendar_fixed25", "Calendrier seul · poids fixe 25 % (témoin)", "calendar/proposals_25.parquet", "fixe 25 %", "control"),
    ("coherent_p50", "coherent_forest_direct", "P50 cohérent · forêt · direct", "forest/predictions.parquet", "direct", "experiment"),
    ("coherent_p50", "coherent_forest_governed", "P50 cohérent · forêt · gouverné", "forest/governed_predictions.parquet", "gouverné", "experiment"),
    ("coherent_p50", "coherent_empirical_direct", "P50 cohérent · empirique · direct", "empirical/predictions.parquet", "direct", "experiment"),
    ("coherent_p50", "coherent_empirical_governed", "P50 cohérent · empirique · gouverné", "empirical/governed_predictions.parquet", "gouverné", "experiment"),
    ("stress_guard", "stress_guard_direct", "StressGuard physique enrichie · direct", "physics_direct.parquet", "direct", "experiment"),
    ("stress_guard", "stress_guard_governed", "StressGuard physique enrichie · gouverné", "physics_governed.parquet", "gouverné", "experiment"),
    ("stress_guard", "coherent_forest_calibrated", "P50 forêt · intervalles recalibrés (prix identique)", "p50_calibrated.parquet", "intervalles seulement", "control"),
)


def _safe(root: Path, value: str | Path) -> Path:
    root = Path(root).absolute()
    raw = Path(value)
    if ".." in raw.parts:
        raise KPIDataError(f"Parent traversal is forbidden: {value}")
    path = Path(os.path.abspath(raw if raw.is_absolute() else root / raw))
    if not path.is_relative_to(root):
        raise KPIDataError(f"Source is outside the project: {value}")
    for part in [path, *path.parents]:
        if part.exists() or part.is_symlink():
            st = part.lstat()
            if part.is_symlink() or getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024):
                raise KPIDataError(f"Linked or reparse-point source is forbidden: {part}")
        if part == root:
            break
    return path


def _stable_bytes(path: Path) -> tuple[bytes, str]:
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise KPIDataError(f"Cannot read source {path}: {exc}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise KPIDataError(f"Source changed during capture: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    if _digest(path) != digest:
        raise KPIDataError(f"Source content changed during capture: {path}")
    return raw, digest


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _Capture:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.files: dict[str, str] = {}

    def read(self, path, expected=None):
        path = _safe(self.root, path)
        raw, digest = _stable_bytes(path)
        if expected is not None and (not isinstance(expected, str) or not SHA256.fullmatch(expected) or digest != expected):
            raise KPIDataError(f"SHA256 mismatch: {path}")
        previous = self.files.get(str(path))
        if previous is not None and previous != digest:
            raise KPIDataError(f"Source changed during KPI capture: {path}")
        self.files[str(path)] = digest
        return raw

    def json(self, path, expected=None):
        try:
            value = json.loads(self.read(path, expected).decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise KPIDataError(f"Invalid source JSON: {path}") from exc
        if not isinstance(value, dict):
            raise KPIDataError(f"Expected JSON object: {path}")
        return value

    def verify_map(self, directory, mapping, required=()):
        if not isinstance(mapping, dict) or not set(required).issubset(mapping):
            raise KPIDataError(f"Incomplete file seals in {directory}")
        for name, digest in mapping.items():
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise KPIDataError(f"Invalid sealed relative file name: {name}")
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                raise KPIDataError(f"Invalid SHA256 for sealed source: {name}")
            self.read(Path(directory) / name, digest)


def _verify_snapshot(capture: _Capture, family: str, directory: Path):
    directory = _safe(capture.root, directory)
    expected_parent = capture.root / NAMESPACE / family / "snapshots"
    if directory.parent != expected_parent or not SNAPSHOT_NAME.fullmatch(directory.name):
        raise KPIDataError(f"Unexpected snapshot path for {family}: {directory}")
    manifest = capture.json(directory / "manifest.json")
    capture.verify_map(directory, manifest.get("input_files"), {"panel.parquet", "config.json", "source_audit.json"})
    config = capture.json(directory / "config.json")
    if config != manifest.get("config"):
        raise KPIDataError(f"Config / manifest identity mismatch: {directory}")
    if config.get("diagnostic_only") is not True or config.get("production_modified") is not False or config.get("activation_performed") is not False:
        raise KPIDataError(f"Research-only flags invalid: {directory}")
    manifest_sha = capture.files[str(directory / "manifest.json")]
    if family == "stress_guard":
        result = capture.json(directory / "results_manifest.json")
        if result.get("status") != "completed" or result.get("suite_manifest_sha256") != manifest_sha:
            raise KPIDataError(f"Incomplete or unbound result: {directory}")
        capture.verify_map(directory, result.get("result_files"), {s[3] for s in MODEL_SPECS if s[0] == family})
    else:
        variants = ("fundamental", "calendar") if family == "fundamental" else ("forest", "empirical")
        if set(config.get("variants", [])) != set(variants):
            raise KPIDataError(f"Unexpected variants in {directory}")
        comparison = capture.json(directory / "comparison_manifest.json")
        if comparison.get("status") != "completed" or comparison.get("suite_manifest_sha256") != manifest_sha:
            raise KPIDataError(f"Comparison is not completed and bound: {directory}")
        variant_seals = comparison.get("variant_manifests")
        if not isinstance(variant_seals, dict) or set(variant_seals) != set(variants):
            raise KPIDataError(f"Incomplete variant manifest seals: {directory}")
        if not isinstance(comparison.get("comparison_sha256"), str) or not SHA256.fullmatch(comparison["comparison_sha256"]):
            raise KPIDataError(f"Missing comparison content seal: {directory}")
        capture.read(directory / "comparison.json", comparison["comparison_sha256"])
        for variant in variants:
            result = capture.json(directory / variant / "results_manifest.json", variant_seals[variant])
            if (result.get("status") != "completed" or result.get("variant") != variant
                    or result.get("suite_manifest_sha256") != manifest_sha):
                raise KPIDataError(f"Incomplete or unbound variant: {directory / variant}")
            required = {Path(s[3]).name for s in MODEL_SPECS if s[0] == family and Path(s[3]).parts[0] == variant}
            capture.verify_map(directory / variant, result.get("result_files"), required)
    return directory, manifest, capture.json(directory / "source_audit.json")


def _select_snapshot(capture: _Capture, family: str):
    """Latest completed pointer first; bounded dated-directory fallback only.

    A completed pointer whose seals are corrupt is a hard failure, not a quiet
    fallback to an older, better-performing candidate.
    """
    namespace = _safe(capture.root, NAMESPACE / family)
    pointer = namespace / "latest.json"
    pointer_data = capture.json(pointer) if pointer.is_file() else None
    if pointer_data and pointer_data.get("status") == "completed":
        directory, manifest, audit = _verify_snapshot(capture, family, Path(pointer_data["snapshot"]))
        return directory, manifest, audit, "latest_completed_pointer"
    capture.files.pop(str(pointer), None)
    snapshots = _safe(capture.root, namespace / "snapshots")
    if not snapshots.is_dir():
        raise KPIDataError(f"No snapshots for {family}")
    choices = sorted((p for p in snapshots.iterdir() if SNAPSHOT_NAME.fullmatch(p.name)), key=lambda p: p.name, reverse=True)[:20]
    for directory in choices:
        result_name = "results_manifest.json" if family == "stress_guard" else "comparison_manifest.json"
        result_path = _safe(capture.root, directory / result_name)
        if not result_path.is_file():
            continue
        result = capture.json(result_path)
        if result.get("status") != "completed":
            capture.files.pop(str(result_path), None)
            continue
        directory, manifest, audit = _verify_snapshot(capture, family, directory)
        return directory, manifest, audit, "bounded_latest_completed_directory"
    raise KPIDataError(f"No completed {family} snapshot among the last 20 dated directories")


def _normalise(frame, name, *, candidate=False):
    required = [*BASE_COLUMNS, *( ["candidate_forecast"] if candidate else [] )]
    if not set(required).issubset(frame.columns):
        raise KPIDataError(f"Missing prediction columns in {name}")
    frame = frame[required].copy()
    if frame["zone"].isna().any() or set(frame["zone"].astype(str)) != set(ZONES):
        raise KPIDataError(f"Expected exactly BE, DE, FR and NL in {name}")
    frame["zone"] = frame["zone"].astype(str)
    if not isinstance(frame.timestamp_utc.dtype, pd.DatetimeTZDtype):
        raise KPIDataError(f"Explicit aware timestamp dtype required in {name}")
    frame["timestamp_utc"] = frame.timestamp_utc.dt.tz_convert("UTC")
    if frame.timestamp_utc.isna().any() or not frame.timestamp_utc.eq(frame.timestamp_utc.dt.floor("h")).all():
        raise KPIDataError(f"Invalid physical hourly timestamps in {name}")
    if frame.duplicated(KEYS).any() or not frame["sample"].isin(["evaluation", "live"]).all():
        raise KPIDataError(f"Duplicate identities or unknown sample role in {name}")
    if not isinstance(frame.forecast_origin_utc.dtype, pd.DatetimeTZDtype):
        raise KPIDataError(f"Explicit aware forecast origin dtype required in {name}")
    frame["forecast_origin_utc"] = frame.forecast_origin_utc.dt.tz_convert("UTC")
    local = frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    expected_origin = (local - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
    if frame.forecast_origin_utc.isna().any() or not frame.forecast_origin_utc.eq(expected_origin).all():
        raise KPIDataError(f"Forecast origin must be D-1 08:00 local civil time in {name}")
    for key in ["forecast", "actual", "benchmark_forecast", *( ["candidate_forecast"] if candidate else [] )]:
        frame[key] = pd.to_numeric(frame[key], errors="raise").astype(float)
        if np.isinf(frame[key].to_numpy()).any():
            raise KPIDataError(f"Infinite {key} in {name}")
    if frame.forecast.isna().any() or (candidate and frame.candidate_forecast.isna().any()):
        raise KPIDataError(f"Missing baseline or candidate forecasts in {name}")
    return frame.sort_values(KEYS).reset_index(drop=True)


def _same_reference(left, right, name):
    if not left[KEYS + ["forecast_origin_utc", "sample"]].equals(right[KEYS + ["forecast_origin_utc", "sample"]]):
        raise KPIDataError(f"Different physical hours or evaluation/live roles: {name}")
    for col in ("forecast", "actual", "benchmark_forecast"):
        if not np.array_equal(left[col].to_numpy(), right[col].to_numpy(), equal_nan=True):
            raise KPIDataError(f"Different frozen {col} vintage in {name}; no hidden reference replacement allowed")


def _verify_production(capture, source_audit, baseline):
    """Verify the exact published NYX report values and their local identities."""
    from economic_value.data import read_report

    evidence = source_audit.get("source_data_audit", {}).get("baseline", {})
    sources = evidence.get("sources")
    if not isinstance(sources, list) or len(sources) != len(ZONES) or {s.get("zone") for s in sources} != set(ZONES):
        raise KPIDataError("Four explicit production source reports are required")
    parsed_sources = []
    for item in sources:
        if item.get("model") != "nuclear_kalman":
            raise KPIDataError("Production baseline must be nuclear_kalman")
        path = _safe(capture.root, item["path"])
        relative = path.relative_to(capture.root)
        parts = relative.parts
        if (len(parts) != 6 or parts[:2] != ("runs", "exports") or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[2])
                or parts[3] != item["zone"].lower() or parts[4] != "nuclear_kalman"
                or parts[5] != f"forecast_{parts[3]}_{parts[2]}_nuclear_kalman.html"):
            raise KPIDataError(f"Invalid operational report path: {path}")
        capture.read(path, item["sha256"])
        for dependency in item.get("time_axis_audits", []) + ([item["csv_source"]] if item.get("csv_source") else []):
            capture.read(dependency["path"], dependency["sha256"])
        parsed, parsed_audit = read_report(path, zone=item["zone"], model="nuclear_kalman")
        capture.read(path, parsed_audit["sha256"])
        for dependency in parsed_audit.get("time_axis_audits", []) + ([parsed_audit["csv_source"]] if parsed_audit.get("csv_source") else []):
            capture.read(dependency["path"], dependency["sha256"])
        expected = baseline.loc[baseline.zone.eq(item["zone"])].set_index("timestamp_utc")
        actual = parsed.set_index("timestamp_utc").reindex(expected.index)
        for col in ("forecast", "actual", "benchmark_forecast"):
            if not np.array_equal(expected[col].to_numpy(), actual[col].to_numpy(), equal_nan=True):
                raise KPIDataError(f"Published production {col} differs from the frozen baseline for {item['zone']}")
        parsed_sources.append({"zone": item["zone"], "path": str(path), "sha256": item["sha256"], "values_verified": True})
    return {"delivery_day": evidence.get("delivery_day"), "sources": parsed_sources,
            "all_p50_and_references_identical_to_published_production": True,
            "forecast_pit_certified": False, "benchmark_pit_certified": False,
            "description": "NYX nucléaire + Kalman · production (historique figé)"}


def _recommended_end(frame):
    last_days = {}
    for (model, zone), group in frame.groupby(["model_id", "zone"]):
        days = group.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
        complete = []
        for day, rows in group.groupby(days):
            start = pd.Timestamp(day, tz="Europe/Paris")
            end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
            expected = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
            if pd.DatetimeIndex(rows.timestamp_utc).sort_values().equals(expected) and rows.actual.notna().all():
                complete.append(day)
        if not complete:
            raise KPIDataError(f"No complete observed evaluation day for {model}/{zone}")
        last_days[f"{model}/{zone}"] = max(complete)
    return min(last_days.values()), last_days


def load_recent_models(root) -> tuple[pd.DataFrame, list[dict], dict]:
    """Capture twelve configurations on their unchanged common source vintage."""
    capture = _Capture(root)
    selected, provenance = {}, {}
    baseline, baseline_audit = None, None
    for family in FAMILIES:
        directory, manifest, source_audit, selection = _select_snapshot(capture, family)
        panel = _normalise(pd.read_parquet(BytesIO(capture.read(directory / "panel.parquet"))), f"{family}/panel")
        if baseline is None:
            baseline, baseline_audit = panel, source_audit
        else:
            _same_reference(baseline, panel, family)
        selected[family] = directory
        provenance[family] = {"snapshot": str(directory), "selection": selection,
            "manifest_sha256": capture.files[str(directory / "manifest.json")],
            "created_at_utc": manifest.get("created_at_utc"), "diagnostic_only": True,
            "prospective_validation_completed": False, "exploratory_year_already_examined": True}
    production = _verify_production(capture, baseline_audit, baseline)
    base_rows = baseline.loc[baseline["sample"].eq("evaluation")].rename(columns={"benchmark_forecast": "storm"}).copy()
    base_rows["model_id"] = "nuclear_kalman"
    catalog = [{"id": "nuclear_kalman", "label": production["description"], "family": "production",
        "decision": "production", "kind": "production", "source_path": str(selected["fundamental"] / "panel.parquet"),
        "report_path": production["sources"][0]["path"], "production_delivery_day": production["delivery_day"],
        "reports_by_zone": {s["zone"]: s["path"] for s in production["sources"]}}]
    frames, raw_prices = [base_rows], {}
    excluded = {"nuclear_kalman": int(baseline["sample"].eq("live").sum())}
    for family, model_id, label, relative, decision, kind in MODEL_SPECS:
        path = selected[family] / relative
        result = _normalise(pd.read_parquet(BytesIO(capture.read(path))), model_id, candidate=True)
        _same_reference(baseline, result, model_id)
        raw_prices[model_id] = result.candidate_forecast.to_numpy()
        excluded[model_id] = int(result["sample"].eq("live").sum())
        result["forecast"] = result.pop("candidate_forecast")
        result = result.loc[result["sample"].eq("evaluation")].rename(columns={"benchmark_forecast": "storm"})
        result["model_id"] = model_id
        frames.append(result)
        report_name = {"fundamental": "fundamental_comparison.html", "coherent_p50": "coherent_p50_comparison.html",
                       "stress_guard": "stress_guard_comparison.html"}[family]
        report_path = _safe(capture.root, selected[family] / report_name)
        if not report_path.is_file():
            raise KPIDataError(f"Missing completed family report: {report_path}")
        catalog.append({"id": model_id, "label": label, "family": family, "decision": decision, "kind": kind,
            "source_path": str(path), "source_sha256": capture.files[str(path)], "report_path": str(report_path),
            "diagnostic_only": True, "independent_validation": False})
    if not np.array_equal(raw_prices["coherent_forest_direct"], raw_prices["coherent_forest_calibrated"]):
        raise KPIDataError("Interval-only control no longer has the same P50 as the forest direct model")
    catalog[-1]["price_alias_of"] = "coherent_forest_direct"
    combined = pd.concat(frames, ignore_index=True)[["model_id", *KEYS, "forecast", "actual", "storm", "sample", "forecast_origin_utc"]]
    end_day, last_days = _recommended_end(combined)
    start_day = (pd.Timestamp(end_day) - pd.Timedelta(days=364)).strftime("%Y-%m-%d")
    audit = {"schema_version": 1, "project_root": str(capture.root), "selected_families": provenance,
        "sources": [{"path": path, "sha256": digest} for path, digest in sorted(capture.files.items())],
        "production": production, "source_delivery_day": production["delivery_day"],
        "source_files": dict(sorted(capture.files.items())), "zones": list(ZONES), "model_count": len(catalog),
        "source_rows_per_model": len(baseline), "evaluation_rows_per_model": len(base_rows),
        "live_rows_excluded_by_model": excluded, "live_excluded_even_when_observed": True,
        "last_complete_observed_evaluation_day_by_model_zone": last_days,
        "recommended_end_day": end_day, "recommended_start_day": start_day, "evaluation_days": 365,
        "frozen_references_identical_across_models": True, "reference_revisions_substituted": False,
        "forecast_origin_contract": "D-1 08:00 Europe/Paris civil; assumed cutoff, not actual issue-time evidence",
        "production_modified": False, "models_retrained": False, "external_api_used": False,
        "independent_validation": False, "year_already_examined": True,
        "selection_policy": "latest completed snapshot per family, never best score",
        "limitations": ["Replays exploratoires sur une année déjà examinée, pas une validation indépendante.",
            "NYX et Storm sont les vintages historiques figés des rapports publiés ; aucune actualisation réseau.",
            "La livraison live reste exclue, même si son prix observé est déjà connu.",
            "Le témoin recalibré ne modifie que les intervalles : ses KPI de prix sont identiques au P50 forêt direct."]}
    verify_sources(audit)
    return combined, catalog, audit


def verify_sources(audit) -> None:
    """Recheck captured identities only, without rediscovery or any mutation."""
    root = Path(audit["project_root"]).absolute()
    seen = set()
    if not isinstance(audit.get("sources"), list) or not audit["sources"]:
        raise KPIDataError("Missing captured source identities")
    for item in audit["sources"]:
        path = _safe(root, item["path"])
        if str(path) in seen:
            raise KPIDataError(f"Duplicate captured source identity: {path}")
        seen.add(str(path))
        _, digest = _stable_bytes(path)
        if digest != item.get("sha256"):
            raise KPIDataError(f"Source changed after KPI capture: {path}")
