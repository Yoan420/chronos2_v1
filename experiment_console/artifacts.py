"""Read-only adapters for the repository's existing scientific output formats.

Nothing in this module modifies a result, loads a model, or reads source datasets.
An artifact's existence never proves that an execution succeeded. Scope metadata
is deliberately conservative; equal scalar scores do not imply comparable runs.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any


MARKERS = {
    "run_manifest.json", "experiment_manifest.json", "evaluation_manifest.json",
    "status.json", "run_status.json", "metrics_hourly.json", "evaluation_metrics.json",
    "evaluation_summary.json", "metrics.json",
}
METRIC_FILES = ("metrics_hourly.json", "evaluation_metrics.json", "evaluation_summary.json", "metrics.json")
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "checkpoint", "checkpoints", "inputs", "cache", "materialized", "runtime", "tmp", ".experiment_console"}
ARTIFACT_SKIP_DIRS = SKIP_DIRS | {"prepared", "selected_pit", "snapshot", "sources"}
NUMERIC_METRICS = {
    "mae", "rmse", "mse", "bias", "mape", "smape", "wape", "r2",
    "n_scored", "n_expected", "prediction_coverage", "score_coverage",
    "n_hours", "n_local_days", "physical_hours", "physical_days",
}
SCOPE_LABELS = {"period": "période", "target": "cible", "zone": "zone", "horizon": "horizon", "frequency": "fréquence", "coverage": "couverture d'évaluation", "metric_scope": "protocole d'évaluation"}


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return value if isinstance(value, int) else number


def _json(path: Path, warnings: list[str]) -> dict:
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            warnings.append(f"{path.name} : métadonnées trop volumineuses (limite 8 Mo).")
            return {}
        value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=lambda _value: None, parse_float=_number)
        if not isinstance(value, dict):
            warnings.append(f"{path.name} : objet JSON attendu, format non reconnu.")
            return {}
        return value
    except (OSError, ValueError, RecursionError) as exc:
        warnings.append(f"{path.name} : fichier illisible ou incomplet ({type(exc).__name__}).")
        return {}


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except (ValueError, OSError):
        return False


def _linked(path: Path) -> bool:
    """Exclude Windows junctions as well as symbolic links during traversal."""
    try:
        return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 1024)
    except OSError:
        return True


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None and value != "" and value != {} and value != []), None)


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _duration(start: Any, end: Any) -> float | None:
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    try:
        seconds = (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
        return seconds if seconds >= 0 else None
    except (ValueError, TypeError):
        return None


def _kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".log"):
        return "log"
    if name.endswith((".html", ".htm", ".pdf")):
        return "report"
    if name.endswith((".png", ".jpg", ".jpeg", ".svg", ".webp")):
        return "image"
    if name.endswith((".csv", ".csv.gz")) and any(token in name for token in ("forecast", "backtest", "predictions")):
        return "forecast"
    if "metric" in name or "evaluation" in name:
        return "metrics"
    if name.endswith((".json", ".yaml", ".yml")):
        return "metadata"
    return "table"


def list_artifacts(directory: str | Path, *, max_files: int = 500) -> list[dict]:
    """List local results only, excluding model weights and input/cache trees."""
    directory = Path(directory).resolve()
    result = []
    if not directory.is_dir():
        return result
    for current, dirs, files in os.walk(directory, followlinks=False):
        relative_depth = len(Path(current).relative_to(directory).parts)
        dirs[:] = sorted(d for d in dirs if d.lower() not in ARTIFACT_SKIP_DIRS and not _linked(Path(current) / d)) if relative_depth < 2 else []
        # Nested run folders have their own ownership and import identity.
        if Path(current) != directory and MARKERS.intersection(files):
            dirs[:] = []
            continue
        for name in sorted(files):
            path = Path(current) / name
            # Every traversed parent is local and contains no reparse point.
            # Resolving every artifact repeatedly is very costly on Windows.
            if _linked(path):
                continue
            if not name.lower().endswith((".json", ".yaml", ".yml", ".csv", ".csv.gz", ".log", ".txt", ".html", ".htm", ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".webp")):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            result.append({"path": path.relative_to(directory).as_posix(), "name": name, "kind": _kind(path), "size": size})
            if len(result) >= max_files:
                return result
    return result


def _extract_metrics(documents: dict[str, dict]) -> tuple[dict, list[dict]]:
    metrics: dict[str, float | int] = {}
    coverage = []
    for filename in METRIC_FILES:
        data = documents.get(filename, {})
        rows = data.get("metrics", [])
        if isinstance(rows, dict):
            rows = [dict(values, model=model) for model, values in rows.items() if isinstance(values, dict)]
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                model = str(row.get("model") or "result")
                for key, value in row.items():
                    number = _number(value)
                    if number is not None and (key in NUMERIC_METRICS or any(metric in key.lower() for metric in ("mae", "rmse", "bias", "pinball", "coverage"))):
                        metrics[f"{model}.{key}"] = number
                sample = {key: _number(row.get(key)) for key in ("n_scored", "n_expected", "prediction_coverage", "score_coverage")}
                if any(value is not None for value in sample.values()):
                    coverage.append(sample)
        for key, value in data.items():
            number = _number(value)
            if number is not None and (key in NUMERIC_METRICS or any(metric in key.lower() for metric in ("mae", "rmse", "bias", "pinball", "coverage"))):
                # Names retain their original semantics and units.
                metrics.setdefault(key, number)
    return metrics, coverage


def _scope(manifest: dict, evaluation: dict, documents: dict[str, dict], coverage: list[dict]) -> dict:
    hourly = documents.get("metrics_hourly.json", {})
    diagnostics = _dict(hourly.get("training_diagnostics"))
    summary = documents.get("evaluation_summary.json", {})
    eval_metrics = documents.get("evaluation_metrics.json", {})
    window = _dict(evaluation.get("window"))
    start = _first(summary.get("start_utc"), window.get("first_delivery_utc"), eval_metrics.get("first_delivery_utc"))
    end = _first(summary.get("end_utc"), window.get("last_delivery_utc"), eval_metrics.get("last_delivery_utc"))
    basis = "utc"
    if start is None and end is None:
        start, end = diagnostics.get("evaluation_start_local_date"), diagnostics.get("evaluation_end_local_date")
        basis = "local_date"
    period = {"start": start, "end": end, "basis": basis, "timezone": manifest.get("timezone") if basis == "local_date" else "UTC"} if start and end else None
    target_contract = manifest.get("target_contract")
    frequency = _first(manifest.get("frequency"), manifest.get("freq"), evaluation.get("frequency"))
    if frequency is None and target_contract == "hourly_utc_no_interpolation":
        frequency = "1h"
    unique_coverage = {json.dumps(item, sort_keys=True): item for item in coverage}
    # Counts alone do not establish which timestamps were scored. Missing detail
    # remains visible, and compare_scopes never promotes an incomplete contract.
    coverage_value = sorted(unique_coverage.values(), key=lambda item: json.dumps(item, sort_keys=True)) or None
    return {
        "period": period,
        "target": _first(manifest.get("target_column"), evaluation.get("target_column"), manifest.get("target_id")),
        "zone": _first(manifest.get("zone"), evaluation.get("item_id"), manifest.get("item_id")),
        "horizon": _first(manifest.get("evaluation_horizon"), manifest.get("delivery_horizon"), evaluation.get("horizon")),
        "frequency": frequency,
        "coverage": coverage_value,
        "metric_scope": _first(diagnostics.get("metric_scope"), eval_metrics.get("comparison_scope"), evaluation.get("evaluation_role")),
        "target_contract": target_contract,
    }


def inspect_run(directory: str | Path, *, include_artifacts: bool = True) -> dict:
    """Normalize one folder; never infer process success or historical dates."""
    directory = Path(directory).resolve()
    warnings: list[str] = []
    documents = {}
    try:
        for filename in sorted(MARKERS | {"config.json", "effective_config.json"}):
            path = directory / filename
            if path.is_file() and not path.is_symlink() and _inside(path, directory):
                documents[filename] = _json(path, warnings)
    except OSError as exc:
        warnings.append(f"Dossier inaccessible ({type(exc).__name__}).")
    manifest = _first(documents.get("run_manifest.json"), documents.get("experiment_manifest.json"), {}) or {}
    evaluation = documents.get("evaluation_manifest.json", {})
    status = documents.get("status.json") or documents.get("run_status.json", {})
    auxiliary_status = bool(status) and not documents.get("status.json") and bool(documents.get("run_status.json"))
    reported = _first(status.get("status"), manifest.get("status"))
    normalized = {"complete": "succeeded", "completed": "succeeded", "success": "succeeded", "succeeded": "succeeded", "failed": "failed", "error": "failed", "cancelled": "cancelled", "canceled": "cancelled", "interrupted": "interrupted"}.get(str(reported).lower(), "unknown")
    if auxiliary_status and normalized == "succeeded":
        # run_nuclear_forecast.run_progress explicitly defines this as an
        # auxiliary stage signal, not a verified final-publication receipt.
        normalized = "unknown"
        warnings.append("Étape nucléaire déclarée terminée ; run_status.json ne confirme pas à lui seul la publication des résultats finaux.")
    source = "external" if status else "historical"
    if str(reported).lower() in {"running", "starting", "queued", "pending"}:
        warnings.append("Processus externe : statut déclaré dans le fichier, activité actuelle non vérifiée ; suivi en lecture seule.")
    elif normalized == "unknown":
        warnings.append("Fin d'exécution non attestée ; statut inconnu. La présence de résultats ne prouve pas la réussite.")
    start = _first(status.get("started_at_utc"), status.get("started_at"), manifest.get("started_at_utc"))
    end = _first(status.get("finished_at_utc"), status.get("finished_at"), manifest.get("finished_at_utc"))
    config_ref = manifest.get("config")
    config_path = config_ref if isinstance(config_ref, str) else _dict(config_ref).get("path")
    config = _first(documents.get("effective_config.json"), documents.get("config.json"), manifest.get("effective_config"))
    if not isinstance(config, dict):
        config = config_ref if isinstance(config_ref, dict) and "path" not in config_ref else {}
    if config_path and not config:
        warnings.append("Configuration historique effective non disponible ; la référence source peut avoir changé depuis ce run.")
    metrics, coverage = _extract_metrics(documents)
    artifacts = list_artifacts(directory) if include_artifacts else []
    if not documents and not any(item["kind"] == "forecast" for item in artifacts):
        warnings.append("Format de run non reconnu ; fichiers conservés en lecture seule.")
    log_paths = [str(directory / item["path"]) for item in artifacts if item["kind"] == "log"]
    steps = status.get("steps") if isinstance(status.get("steps"), list) else []
    activity = next((f"{step.get('name', 'étape')} {step.get('zone') or ''}".strip() for step in steps if isinstance(step, dict) and step.get("status") in {"running", "starting"}), None)
    if activity is None:
        activity = next((f"{step.get('name', 'étape')} : {step.get('status')}" for step in reversed(steps) if isinstance(step, dict) and step.get("status") not in {"pending", "queued"}), None)
    if activity is None and status.get("phase"):
        activity = " · ".join(str(value) for value in (status.get("zone"), status.get("stage"), status.get("phase")) if value)
    model = _first(manifest.get("model_id"), manifest.get("challenger_id"), manifest.get("model"))
    run_type = _first(manifest.get("run_type"), manifest.get("kind"), evaluation.get("kind"))
    if run_type is None:
        run_type = "evaluation" if metrics else "forecast" if any(item["kind"] == "forecast" for item in artifacts) else "external" if status else "unknown"
    canonical = os.path.normcase(str(directory))
    return {
        "source_key": "historical:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "output_dir": str(directory), "name": directory.name, "source": source,
        "status": normalized, "reported_status": reported, "type": str(run_type),
        "model": model if isinstance(model, str) else None,
        "config_path": config_path, "config": config, "metrics": metrics,
        "scope": _scope(manifest, evaluation, documents, coverage), "artifacts": artifacts,
        "log_paths": log_paths, "started_at": start, "finished_at": end,
        "duration_seconds": _duration(start, end), "warnings": warnings, "activity": activity,
        "return_code": _first(status.get("return_code"), status.get("returncode")),
        "returncode": _first(status.get("return_code"), status.get("returncode")),
        "status_updated_at": status.get("updated_at_utc"),
        "external_run_id": status.get("run_id"), "steps": steps,
    }


def discover_runs(root: str | Path, *, max_candidates: int = 2000, max_depth: int = 12) -> dict:
    """Find supported folders deterministically, tolerating partial/unknown files.

    latest_status.json is a pointer maintained by the pipeline, not a distinct
    run. Only the canonical status.json directory becomes an import candidate.
    """
    root = Path(root).resolve()
    warnings: list[str] = []
    candidates = []
    if not root.is_dir():
        return {"candidates": [], "warnings": [f"Dossier introuvable : {root}"]}

    def onerror(error: OSError) -> None:
        if len(warnings) < 100:
            warnings.append(f"Dossier inaccessible : {error.filename}")

    for current, dirs, files in os.walk(root, topdown=True, onerror=onerror, followlinks=False):
        directory = Path(current)
        depth = len(directory.relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIRS and not _linked(directory / d)) if depth < max_depth else []
        known = bool(MARKERS.intersection(files)) or any(name.startswith(("forecast_hourly_", "backtest_hourly_")) and name.endswith((".csv", ".csv.gz")) for name in files)
        if known:
            candidates.append(inspect_run(directory))
            if len(candidates) >= max_candidates:
                warnings.append(f"Import limité à {max_candidates} dossiers ; préciser un sous-dossier pour la suite.")
                break
        elif "latest_status.json" in files and len(warnings) < 100:
            alias = _json(directory / "latest_status.json", warnings)
            reference = alias.get("status_file")
            if not isinstance(reference, str) or not Path(reference).is_file():
                warnings.append(f"{directory.relative_to(root)} : latest_status.json sans fichier de statut canonique disponible ; alias ignoré.")
        elif files and not dirs and len(warnings) < 100:
            warnings.append(f"Format non reconnu, dossier ignoré : {directory.relative_to(root)}")
    if len(warnings) >= 100:
        warnings.append("Liste des avertissements limitée à 100 entrées ; importer un sous-dossier pour une inspection ciblée.")
    return {"candidates": candidates, "warnings": warnings}


def compare_scopes(runs: list[dict]) -> dict:
    """Check scope equality without ranking or recomputing scientific scores."""
    warnings = []
    dimensions = {}
    if len(runs) < 2:
        warnings.append("Sélectionner au moins deux runs pour comparer leurs périmètres.")
    for key, label in SCOPE_LABELS.items():
        values = [_dict(run.get("scope")).get(key) for run in runs]
        missing = any(value is None or value == {} or value == [] or value == "" for value in values)
        equal = bool(values) and len({json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values}) == 1
        state = "unknown" if missing else "same" if equal else "different"
        dimensions[key] = {"label": label, "status": state, "values": values}
        if missing:
            warnings.append(f"Comparabilité incertaine : {label} non disponible pour au moins un run.")
        elif not equal:
            warnings.append(f"Périmètres différents : {label}. Ne pas interpréter les métriques comme un classement.")
        if key == "coverage" and not missing:
            for value in values:
                if isinstance(value, list) and any(not isinstance(item, dict) or any(item.get(field) is None for field in ("n_scored", "n_expected", "score_coverage")) or item.get("score_coverage") != 1 for item in value):
                    warnings.append("Comparabilité incertaine : couverture partielle ou incomplète ; l'identité des heures évaluées n'est pas attestée.")
                    dimensions[key]["status"] = "unknown" if equal else "different"
                    break
    return {"comparable": len(runs) >= 2 and not warnings, "warnings": warnings, "dimensions": dimensions}


def read_forecast(path: str | Path, limit: int = 2000, *, scope: dict | None = None) -> dict:
    """Read real hourly CSV evidence; never treat price_eur_mwh as an actual.

    In live forecast_hourly_*.csv, price_eur_mwh is the exported point forecast.
    The backtest and exogenous evaluation outputs use the explicit actual field.
    Error is predicted minus observed and exists only for a finite pair.
    """
    path = Path(path)
    warnings: list[str] = []
    points = []
    columns: dict[str, Any] = {}
    truncated = False
    limit = max(1, min(int(limit), 10000))
    bounds = None
    period = _dict(_dict(scope).get("period"))
    is_evaluation = "backtest" in path.name.lower() or "evaluation_predictions" in path.name.lower()
    if is_evaluation and period.get("basis") == "utc":
        try:
            start, end = (datetime.fromisoformat(str(period[key]).replace("Z", "+00:00")) for key in ("start", "end"))
            if start.utcoffset() is not None and end.utcoffset() is not None and start <= end:
                bounds = (start, end)
        except (KeyError, ValueError, TypeError):
            pass
    if is_evaluation and bounds is None:
        warnings.append("Période d'évaluation UTC non disponible : l'aperçu du fichier peut couvrir une autre période que les métriques finales.")
    if not path.name.lower().endswith((".csv", ".csv.gz")):
        return {"points": [], "columns": {}, "warnings": ["Format de prévisions non reconnu (CSV ou CSV.GZ attendu)."], "truncated": False}
    try:
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            names = reader.fieldnames or []
            timestamp = next((key for key in ("delivery_start_utc", "timestamp", "time", "date", "ds") if key in names), None)
            actual = next((key for key in ("actual", "observed", "y_true") if key in names), None)
            predictions = [key for key in names if key == "q50" or key.endswith(("__q50", "_q50"))]
            if not predictions:
                predictions = [key for key in ("predicted", "prediction", "y_pred", "forecast") if key in names]
            columns = {"timestamp": timestamp, "observed": actual, "predictions": predictions, "all": names}
            if timestamp is None or not predictions:
                warnings.append("Colonnes de prévisions non reconnues ; consulter le fichier original.")
                return {"points": [], "columns": columns, "warnings": warnings, "truncated": False}
            if actual is None:
                warnings.append("Observations non disponibles : seules les prévisions sont affichées ; aucune erreur n'est calculée.")
            displayed_rows = 0
            for row in reader:
                if bounds:
                    try:
                        instant = datetime.fromisoformat(str(row.get(timestamp)).replace("Z", "+00:00"))
                        if instant.utcoffset() is None or not bounds[0] <= instant <= bounds[1]:
                            continue
                    except (ValueError, TypeError):
                        continue
                # Native backtest files include training-only rows before OOF
                # predictions begin. They are not forecast evidence.
                if not any(_number(row.get(prediction)) is not None for prediction in predictions):
                    continue
                if displayed_rows >= limit:
                    truncated = True
                    break
                displayed_rows += 1
                observed = _number(row.get(actual)) if actual else None
                for prediction in predictions:
                    predicted = _number(row.get(prediction))
                    if predicted is None and observed is None:
                        continue
                    points.append({"timestamp": row.get(timestamp), "observed": observed, "predicted": predicted, "error": predicted - observed if predicted is not None and observed is not None else None, "model": prediction})
    except (OSError, EOFError, ValueError, csv.Error, UnicodeError) as exc:
        warnings.append(f"Prévisions illisibles ou incomplètes ({type(exc).__name__}).")
    if truncated:
        warnings.append(f"Aperçu limité aux {limit} premières lignes avec prévisions ; les métriques finales restent celles des artefacts d'origine.")
    return {"points": points, "columns": columns, "warnings": warnings, "truncated": truncated, "evaluation_period_applied": bounds is not None}
