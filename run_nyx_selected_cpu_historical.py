"""Local historical CPU replay of the selected FR/BE/NL annual price recipes.

This is an evaluation harness, not the live NYX producer. It needs frozen,
Git-ignored historical feature matrices, NYX baselines and OOF references.
It never downloads data or claims that a clean Git clone can forecast with
these variables. CPU CatBoost fits are new models, not reproduced GPU scores.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, timedelta
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time
import uuid

import numpy as np
import pandas as pd

from chronos2_hourly import nyx_pooled_cpu_price_model as cpu_model
from chronos2_hourly.nyx_pooled_cpu_price_model import (
    PooledConfig,
    ZONES,
    fit_pooled_block,
    grid,
)

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path("runs/experiments/nyx_improvement_to20260923")
BASELINE = Path("runs/experiments/nyx_local_365_to20260923")
FIRST_TRAINING_DAY = "2024-09-23"
ANNUAL_FIRST = date(2025, 9, 24)
ANNUAL_STOP = date(2026, 9, 24)
ANNUAL_GRID = grid(str(ANNUAL_FIRST), str(ANNUAL_STOP))
FULL_SOURCE_GRID = grid("2024-09-09", str(ANNUAL_STOP))
COUNTRIES = ("FR", "BE", "NL")
SELECTED_ARCHIVE = {
    "FR": (ARCHIVE / "rmse_pooled_composition_v1/annual/FR.parquet",
           "residual__disagreement20__w1p0"),
    "BE": (ARCHIVE / "rmse_boosting_2000_composition_v1/annual/BE.parquet",
           "boosting_2000_mean_disagreement20"),
    "NL": (ARCHIVE / "rmse_boosting_2000_composition_v1/annual/NL.parquet",
           "boosting_2000_mean_all"),
}
ROLE_CONFIGS = {
    "fr_residual_1000": PooledConfig(target_mode="residual", iterations=1000),
    "cwe_residual_2000": PooledConfig(target_mode="residual", iterations=2000),
    "cwe_absolute_2000": PooledConfig(target_mode="absolute", iterations=2000),
}
ROLE_COUNTRIES = {
    "fr_residual_1000": ("FR",),
    "cwe_residual_2000": ("BE", "NL"),
    "cwe_absolute_2000": ("BE", "NL"),
}
QUALIFICATION_GATE = {
    "hours": 8759,
    "country_rule": "RMSE strictly below Storm AND strict-win fraction above 0.5",
    "countries": list(COUNTRIES),
    "all_countries_required": True,
    "recipes_frozen_before_CPU_replay": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_path(relative: Path) -> Path:
    path = ROOT / relative
    require(path.is_file(), f"Missing local historical artifact: {relative}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + "." + uuid.uuid4().hex + ".tmp.parquet")
    try:
        frame.to_parquet(temporary)
        digest = sha256(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(relative: Path) -> dict:
    return json.loads(source_path(relative).read_text(encoding="utf-8"))


def _plan(role: str) -> dict:
    name = "rmse_pooled_gpu_v1" if role == "fr_residual_1000" else "rmse_boosting_2000_gpu_v1"
    return _read_json(ARCHIVE / name / "plan.json")


def _feature_path(role: str, zone: str) -> Path:
    if role == "fr_residual_1000":
        folder = "pooled_fundamentals_v1"
    elif role == "cwe_residual_2000":
        folder = "pooled_jao_refresh_v1/compact"
    else:
        folder = "pooled_jao_refresh_v1"
    return ARCHIVE / "feature_sets" / folder / f"features_{zone}.parquet"


def _feature_spec(role: str, zone: str, plan: dict) -> dict:
    return (plan["features"][zone] if role == "fr_residual_1000"
            else plan["features"][ROLE_CONFIGS[role].target_mode][zone])


def _validate_index(index: pd.Index, expected: pd.DatetimeIndex, label: str) -> None:
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
            and index.is_unique and index.is_monotonic_increasing
            and index.equals(expected), f"{label}: physical UTC grid differs")


def _load_baselines(plan_2000: dict) -> tuple[dict, dict, dict]:
    actual, nyx, receipt = {}, {}, {}
    for zone in ZONES:
        relative = BASELINE / "baseline" / f"{zone}.parquet"
        path = source_path(relative)
        digest = sha256(path)
        require(digest == plan_2000["baselines"][zone]["sha256"],
                f"{relative}: historical baseline hash differs")
        frame = pd.read_parquet(path)
        _validate_index(frame.index, FULL_SOURCE_GRID, str(relative))
        require({"actual", "nyx__q50"}.issubset(frame.columns),
                f"{relative}: observation or NYX q50 missing")
        for name in ("actual", "nyx__q50"):
            require(np.isfinite(frame[name].to_numpy(dtype=float)).all(),
                    f"{relative}: nonfinite {name}")
        actual[zone] = frame["actual"].copy().rename("actual")
        nyx[zone] = frame["nyx__q50"].copy().rename("nyx_q50")
        receipt[zone] = {"path": str(relative), "sha256": digest, "hours": len(frame)}
    return actual, nyx, receipt


def _load_features(role: str) -> tuple[dict, dict]:
    plan = _plan(role)
    features, receipt = {}, {}
    for zone in ZONES:
        relative = _feature_path(role, zone)
        path = source_path(relative)
        spec = _feature_spec(role, zone, plan)
        digest = sha256(path)
        require(digest == spec["sha256"], f"{relative}: historical feature hash differs")
        frame = pd.read_parquet(path)
        _validate_index(frame.index, FULL_SOURCE_GRID, str(relative))
        require(list(frame.columns) == spec["columns"],
                f"{relative}: ordered historical feature schema differs")
        require(not any(any(token in name.lower() for token in
                            ("storm", "actual", "observed", "target"))
                        for name in frame.columns),
                f"{relative}: forbidden target or Storm feature")
        features[zone] = frame
        receipt[zone] = {"path": str(relative), "sha256": digest,
                         "hours": len(frame), "columns": len(frame.columns)}
    return features, receipt


def _load_references() -> tuple[dict, dict]:
    paths = {
        "FR": ARCHIVE / "rmse_pooled_composition_v1/annual/FR.parquet",
        "BE": ARCHIVE / "rmse_exchange_composition_v1/oof/BE.parquet",
    }
    reference, receipt = {}, {}
    for zone, relative in paths.items():
        path = source_path(relative)
        digest = sha256(path)
        frame = pd.read_parquet(path, columns=["reference"])
        require(isinstance(frame.index, pd.DatetimeIndex) and str(frame.index.tz) == "UTC"
                and frame.index.is_unique and frame.index.is_monotonic_increasing,
                f"{relative}: invalid UTC index")
        require(ANNUAL_GRID.isin(frame.index).all(),
                f"{relative}: annual reference incomplete")
        series = frame.loc[ANNUAL_GRID, "reference"].astype(float)
        require(np.isfinite(series.to_numpy()).all(),
                f"{relative}: nonfinite annual reference")
        reference[zone] = series
        receipt[zone] = {"path": str(relative), "sha256": digest,
                         "annual_hours": len(series)}
    return reference, receipt


def _load_comparisons() -> tuple[dict, dict]:
    comparisons, receipt = {}, {}
    for zone in COUNTRIES:
        relative = BASELINE / "inputs" / f"{zone}_comparison.parquet"
        path = source_path(relative)
        digest = sha256(path)
        frame = pd.read_parquet(path)
        _validate_index(frame.index, ANNUAL_GRID, str(relative))
        require({"actual", "storm"}.issubset(frame.columns),
                f"{relative}: official observed/Storm columns missing")
        require(np.isfinite(frame["actual"].to_numpy(dtype=float)).all(),
                f"{relative}: nonfinite official observed prices")
        require(np.isfinite(frame["storm"].to_numpy(dtype=float)).sum() >= 8759,
                f"{relative}: Storm annual coverage unexpectedly low")
        comparisons[zone] = frame[["actual", "storm"]].astype(float)
        receipt[zone] = {"path": str(relative), "sha256": digest,
                         "hours": len(frame),
                         "storm_finite_hours": int(np.isfinite(frame["storm"]).sum())}
    return comparisons, receipt


def verify_historical_compositions(references: dict) -> dict:
    """Prove the frozen composition arithmetic against archived GPU OOF points.

    These GPU points are used only here, before CPU fitting. The CPU replay
    never feeds them to a model or uses them in its new score.
    """
    expert_paths = {
        "fr_residual_1000_FR": ARCHIVE / "rmse_pooled_gpu_v1/residual/FR_oof.parquet",
        "cwe_residual_2000_BE": ARCHIVE / "rmse_boosting_2000_gpu_v1/residual/BE_oof.parquet",
        "cwe_absolute_2000_BE": ARCHIVE / "rmse_boosting_2000_gpu_v1/absolute/BE_oof.parquet",
        "cwe_residual_2000_NL": ARCHIVE / "rmse_boosting_2000_gpu_v1/residual/NL_oof.parquet",
        "cwe_absolute_2000_NL": ARCHIVE / "rmse_boosting_2000_gpu_v1/absolute/NL_oof.parquet",
    }
    experts, sources = {}, {}
    for key, relative in expert_paths.items():
        path = source_path(relative)
        frame = pd.read_parquet(path, columns=["point"])
        require(ANNUAL_GRID.isin(frame.index).all(),
                f"{relative}: historical expert annual grid incomplete")
        experts[key] = frame.loc[ANNUAL_GRID, "point"].astype(float)
        sources[key] = {"path": str(relative), "sha256": sha256(path)}
    reconstructed = compose_selected(experts, references)
    results = {}
    for zone, (relative, column) in SELECTED_ARCHIVE.items():
        path = source_path(relative)
        selected = pd.read_parquet(path, columns=[column])
        _validate_index(selected.index, ANNUAL_GRID, str(relative))
        archived = selected[column].to_numpy(dtype=float)
        current = reconstructed[zone].to_numpy(dtype=float)
        require(np.array_equal(archived, current),
                f"{zone}: frozen recipe fails exact archived GPU composition check")
        results[zone] = {"path": str(relative), "sha256": sha256(path),
                         "selected_column": column, "hours": len(archived),
                         "bit_exact": True, "maximum_absolute_difference": 0.0}
    return {"archived_gpu_experts": sources, "archived_selected_series": results,
            "CPU_scores_use_archived_GPU_points": False}


def _historical_gpu_scores(comparisons: dict) -> dict:
    scores = {}
    for zone, (relative, column) in SELECTED_ARCHIVE.items():
        frame = pd.read_parquet(source_path(relative), columns=[column])
        _validate_index(frame.index, ANNUAL_GRID, str(relative))
        scores[zone] = score_selected(frame[column].rename("point"), comparisons[zone])
    return scores


def _markdown_report(receipt: dict) -> str:
    cpu = receipt["scores_on_official_epex_storm_rows"]
    gpu = receipt["historical_gpu_scores_on_same_rows"]
    rows = [
        "# Évaluation historique CPU des variantes CWE retenues",
        "",
        "| Pays | RMSE CPU | RMSE Storm | RMSE GPU historique | Victoires CPU / 8 759 | Victoires GPU / 8 759 | Porte CPU |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for zone in COUNTRIES:
        cp, gp = cpu[zone], gpu[zone]
        rows.append(f"| {zone} | {cp['rmse']:.4f} | {cp['storm_rmse']:.4f} | "
                    f"{gp['rmse']:.4f} | {cp['wins_vs_storm']} | {gp['wins_vs_storm']} | "
                    f"{'passe' if cp['both_criteria_met'] else 'échoue'} |")
    rows += [
        "",
        "Porte fixée avant le replay : pour chacun des trois pays, RMSE CPU strictement "
        "inférieure à Storm et taux de victoires horaires strictes supérieur à 50 % "
        "sur les 8 759 heures communes officielles.",
        "",
        "Verdict global : **" + ("passe" if receipt["qualifies_all_three_countries"]
                                  else "échoue") + "**.",
        "",
        "Limites : les références FR et BE sont des prévisions historiques figées ; "
        "les vintages de publication des variables ne sont pas certifiés ; les matrices, "
        "baselines et références sont des archives locales ignorées par Git. Ce harnais "
        "n'est pas un pipeline de prévision live depuis un clone propre. Les modèles CPU "
        "sont réentraînés et leurs scores ne sont pas les scores GPU archivés.",
        "",
        "Preuves : `receipt.json`, `run_manifest.json`, `checkpoints/` et "
        "`FR.parquet`/`BE.parquet`/`NL.parquet`.",
        "",
    ]
    return "\n".join(rows)


def annual_origins() -> list[tuple[str, str]]:
    origins = []
    day = ANNUAL_FIRST
    while day < ANNUAL_STOP:
        stop = min(day + timedelta(days=7), ANNUAL_STOP)
        origins.append((str(day), str(stop)))
        day += timedelta(days=7)
    require(len(origins) == 53 and origins[-1] == ("2026-09-23", "2026-09-24"),
            "Annual weekly origin schedule differs")
    return origins


def _run_manifest(check: dict, thread_count: int) -> dict:
    return {"protocol": "nyx_selected_cwe_cpu_historical_v1",
            "preflight": check,
            "runner_code_sha256": sha256(Path(__file__)),
            "cpu_model_code_sha256": sha256(Path(cpu_model.__file__)),
            "dependency_versions": {"python": platform.python_version(),
                                    **{name: version(name) for name in
                                       ("catboost", "numpy", "pandas", "pyarrow")}},
            "configs": {role: asdict(config) for role, config in ROLE_CONFIGS.items()},
            "role_countries": {role: list(zones) for role, zones in ROLE_COUNTRIES.items()},
            "first_training_day": FIRST_TRAINING_DAY,
            "origins": [list(pair) for pair in annual_origins()],
            "thread_count": thread_count,
            "qualification_gate": QUALIFICATION_GATE}


def _prepare_output(output_dir: Path, manifest: dict) -> str:
    digest = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    manifest_path = output_dir / "run_manifest.json"
    if output_dir.exists():
        require(manifest_path.is_file(),
                f"Existing output has no run manifest; refusing reuse: {output_dir}")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(old == manifest, "Output source/code/config manifest differs; refusing checkpoint reuse")
    else:
        output_dir.mkdir(parents=True)
        _atomic_json(manifest_path, manifest)
    return digest


def _checkpoint_path(output_dir: Path, role: str, origin: str) -> Path:
    return output_dir / "checkpoints" / role / f"{origin}.json"


def _checkpoint_matrix(role: str, origin: str, stop: str,
                       points: dict[str, pd.Series]) -> dict:
    expected = grid(origin, stop)
    require(set(points) == set(ROLE_COUNTRIES[role]), "Checkpoint country set differs")
    for zone in ROLE_COUNTRIES[role]:
        _validate_index(points[zone].index, expected, f"{role}/{origin}/{zone}")
        require(np.isfinite(points[zone].to_numpy(dtype=float)).all(),
                f"{role}/{origin}/{zone}: nonfinite checkpoint point")
    return {"timestamps_utc": [stamp.isoformat() for stamp in expected],
            "points": {zone: points[zone].astype(float).tolist()
                       for zone in ROLE_COUNTRIES[role]}}


def _write_checkpoint(path: Path, *, role: str, origin: str, stop: str,
                      manifest_sha256: str, points: dict, audit: dict) -> None:
    matrix = _checkpoint_matrix(role, origin, stop, points)
    require(audit["tree_count"] == ROLE_CONFIGS[role].iterations,
            "CPU tree count differs before checkpoint")
    payload = {"protocol": "nyx_selected_cwe_cpu_checkpoint_v1",
               "manifest_sha256": manifest_sha256, "role": role,
               "origin_day": origin, "stop_day_exclusive": stop,
               "config": asdict(ROLE_CONFIGS[role]), **matrix,
               "point_matrix_sha256": hashlib.sha256(_canonical_json(matrix).encode()).hexdigest(),
               "fit_audit": audit}
    _atomic_json(path, payload)


def _read_checkpoint(path: Path, *, role: str, origin: str, stop: str,
                     manifest_sha256: str) -> tuple[dict, dict] | None:
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    require(saved.get("protocol") == "nyx_selected_cwe_cpu_checkpoint_v1"
            and saved.get("manifest_sha256") == manifest_sha256
            and saved.get("role") == role and saved.get("origin_day") == origin
            and saved.get("stop_day_exclusive") == stop
            and saved.get("config") == asdict(ROLE_CONFIGS[role]),
            f"Checkpoint identity differs: {path}")
    matrix = {"timestamps_utc": saved.get("timestamps_utc"), "points": saved.get("points")}
    require(saved.get("point_matrix_sha256")
            == hashlib.sha256(_canonical_json(matrix).encode()).hexdigest(),
            f"Checkpoint point digest differs: {path}")
    require(matrix["timestamps_utc"] == [stamp.isoformat() for stamp in grid(origin, stop)]
            and isinstance(matrix["points"], dict)
            and set(matrix["points"]) == set(ROLE_COUNTRIES[role]),
            f"Checkpoint UTC grid/country set differs: {path}")
    expected = grid(origin, stop)
    points = {zone: pd.Series(matrix["points"][zone], index=expected, dtype=float)
              for zone in ROLE_COUNTRIES[role]}
    _checkpoint_matrix(role, origin, stop, points)
    audit = saved.get("fit_audit")
    require(isinstance(audit, dict)
            and audit.get("origin_day") == origin
            and audit.get("stop_day_exclusive") == stop
            and audit.get("tree_count") == ROLE_CONFIGS[role].iterations
            and isinstance(audit.get("fit_seconds"), (int, float))
            and np.isfinite(audit["fit_seconds"]) and audit["fit_seconds"] > 0,
            f"Checkpoint fit audit differs: {path}")
    return points, audit


def preflight() -> dict:
    """Validate every frozen artifact before any CPU fit or output write."""
    plan_2000 = _plan("cwe_residual_2000")
    _, _, baselines = _load_baselines(plan_2000)
    features = {}
    for role in ROLE_CONFIGS:
        _, features[role] = _load_features(role)
    reference_values, references = _load_references()
    historical_composition_check = verify_historical_compositions(reference_values)
    _, comparisons = _load_comparisons()
    return {
        "protocol": "nyx_selected_cwe_cpu_historical_v1",
        "scope": "local historical CPU evaluation only",
        "clean_git_clone_runnable": False,
        "live_saturn_refresh_implemented_here": False,
        "historical_publication_vintages_certified": False,
        "historical_reference_frozen": True,
        "annual_hours_per_country": len(ANNUAL_GRID),
        "annual_origins": len(annual_origins()),
        "planned_cpu_fits": len(ROLE_CONFIGS) * len(annual_origins()),
        "archived_plan_sha256": {
            "fr_residual_1000": sha256(source_path(ARCHIVE / "rmse_pooled_gpu_v1/plan.json")),
            "cwe_2000": sha256(source_path(ARCHIVE / "rmse_boosting_2000_gpu_v1/plan.json")),
        },
        "baselines": baselines,
        "features": features,
        "references": references,
        "historical_composition_check": historical_composition_check,
        "official_comparisons": comparisons,
    }


def compose_selected(experts: dict[str, pd.Series], references: dict[str, pd.Series]) -> dict:
    """Apply only the fixed, historically selected FR/BE/NL point recipes."""
    fr = experts["fr_residual_1000_FR"]
    be_mean = (experts["cwe_residual_2000_BE"]
               + experts["cwe_absolute_2000_BE"]) / 2.0
    nl_mean = (experts["cwe_residual_2000_NL"]
               + experts["cwe_absolute_2000_NL"]) / 2.0
    require(fr.index.equals(references["FR"].index)
            and be_mean.index.equals(references["BE"].index)
            and fr.index.equals(be_mean.index) and fr.index.equals(nl_mean.index),
            "Expert/reference UTC grids differ")
    result = {
        "FR": pd.Series(np.where(np.abs(fr.to_numpy() - references["FR"].to_numpy()) >= 20.,
                                 fr.to_numpy(), references["FR"].to_numpy()),
                        index=fr.index, name="point"),
        "BE": pd.Series(np.where(np.abs(be_mean.to_numpy() - references["BE"].to_numpy()) >= 20.,
                                 be_mean.to_numpy(), references["BE"].to_numpy()),
                        index=be_mean.index, name="point"),
        "NL": nl_mean.rename("point"),
    }
    require(all(np.isfinite(value.to_numpy()).all() for value in result.values()),
            "Selected point contains nonfinite prices")
    return result


def _paired_metrics(point: pd.Series, comparison: pd.DataFrame) -> dict:
    """Use only official EPEX/Storm paired rows, never training targets."""
    require(point.index.equals(comparison.index), "Official comparison grid differs")
    observed, storm = (comparison[name].to_numpy(dtype=float)
                       for name in ("actual", "storm"))
    forecast = point.to_numpy(dtype=float)
    require(np.isfinite(observed).all() and np.isfinite(forecast).all(),
            "Nonfinite official observation or CPU point")
    paired = np.isfinite(storm)
    require(int(paired.sum()) > 0, "No official Storm pairing")
    error = np.abs(forecast[paired] - observed[paired])
    storm_error = np.abs(storm[paired] - observed[paired])
    rmse = float(np.sqrt(np.mean(np.square(error))))
    storm_rmse = float(np.sqrt(np.mean(np.square(storm_error))))
    wins = int(np.sum(error < storm_error))
    ties = int(np.sum(error == storm_error))
    return {"paired_hours": int(paired.sum()), "rmse": rmse,
            "storm_rmse": storm_rmse, "wins_vs_storm": wins,
            "ties_vs_storm": ties, "strict_win_fraction": wins / int(paired.sum())}


def score_selected(point: pd.Series, comparison: pd.DataFrame) -> dict:
    """Apply the fixed annual qualification gate on 8,759 official paired hours."""
    metrics = _paired_metrics(point, comparison)
    require(metrics["paired_hours"] == QUALIFICATION_GATE["hours"],
            "Official Storm annual pairing changed")
    rmse, storm_rmse = metrics["rmse"], metrics["storm_rmse"]
    wins = metrics["wins_vs_storm"]
    metrics.update({
            "beats_storm_rmse": bool(rmse < storm_rmse),
            "wins_majority": bool(wins / metrics["paired_hours"] > 0.5),
            "both_criteria_met": bool(rmse < storm_rmse
                                      and wins / metrics["paired_hours"] > 0.5)})
    return metrics


def _fit_role(role: str, origins: list[tuple[str, str]], actual: dict,
              nyx: dict, thread_count: int, *, output_dir: Path | None = None,
              manifest_sha256: str | None = None) -> tuple[dict, list]:
    require((output_dir is None) == (manifest_sha256 is None),
            "Checkpoint output and manifest must be supplied together")
    features = None
    pieces = {zone: [] for zone in ROLE_COUNTRIES[role]}
    audits = []
    for origin, stop in origins:
        path = None if output_dir is None else _checkpoint_path(output_dir, role, origin)
        checkpoint = (None if path is None else
                      _read_checkpoint(path, role=role, origin=origin, stop=stop,
                                       manifest_sha256=manifest_sha256))
        if checkpoint is None:
            if features is None:
                features, _ = _load_features(role)
            predictions, raw_audit = fit_pooled_block(
                features, actual, nyx, origin_day=origin, stop_day=stop,
                config=ROLE_CONFIGS[role], initial_training_day=FIRST_TRAINING_DAY,
                thread_count=thread_count)
            role_points = {zone: predictions[zone]["point"] for zone in pieces}
            audit = {"origin_day": origin, "stop_day_exclusive": stop,
                     "fit_seconds": raw_audit["fit_seconds"],
                     "tree_count": raw_audit["tree_count"],
                     "training_features_sha256": raw_audit["training_features_sha256"],
                     "training_target_sha256": raw_audit["training_target_sha256"],
                     "prediction_features_sha256": raw_audit["prediction_features_sha256"],
                     "prediction_base_sha256": raw_audit["prediction_base_sha256"],
                     "prediction_points_sha256": raw_audit["prediction_points_sha256"]}
            if path is not None:
                _write_checkpoint(path, role=role, origin=origin, stop=stop,
                                  manifest_sha256=manifest_sha256,
                                  points=role_points, audit=audit)
            reused = False
        else:
            role_points, audit = checkpoint
            reused = True
        for zone in pieces:
            pieces[zone].append(role_points[zone])
        audits.append({**audit, "reused_checkpoint": reused})
    return ({zone: pd.concat(series) for zone, series in pieces.items()}, audits)


def smoke(role: str, origin: str, thread_count: int) -> dict:
    check = preflight()
    require(role in ROLE_CONFIGS, f"Unknown role: {role}")
    schedule = dict(annual_origins())
    require(origin in schedule, f"Origin must be one of the 53 annual weekly origins: {origin}")
    actual, nyx, _ = _load_baselines(_plan("cwe_residual_2000"))
    started = time.monotonic()
    points, audits = _fit_role(role, [(origin, schedule[origin])], actual, nyx, thread_count)
    return {"preflight": check, "action": "smoke", "role": role,
            "config": asdict(ROLE_CONFIGS[role]),
            "origin_day": origin, "output_hours": {z: len(v) for z, v in points.items()},
            "fit_audit": audits[0], "elapsed_seconds": time.monotonic() - started,
            "annual_score_computed": False,
            "GPU_historical_scores_reproduced": False}


def run_replay(output_dir: Path, thread_count: int, max_origins: int = 53) -> dict:
    """Run or resume the fixed annual replay; N<53 is diagnostic only."""
    require(type(max_origins) is int and 1 <= max_origins <= 53,
            "--max-origins must be between 1 and 53")
    check = preflight()
    manifest = _run_manifest(check, thread_count)
    manifest_digest = _prepare_output(output_dir, manifest)
    completed_path = output_dir / "receipt.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        require(completed.get("manifest_sha256") == manifest_digest
                and completed.get("total_cpu_fits") == 159,
                "Completed receipt differs from current replay manifest")
        for zone in COUNTRIES:
            output_path = output_dir / f"{zone}.parquet"
            require(output_path.is_file()
                    and sha256(output_path) == completed["selected_point_sha256"][zone],
                    f"Completed output differs: {output_path}")
        report_path = output_dir / "rapport.md"
        if not report_path.is_file():
            _atomic_text(report_path, _markdown_report(completed))
        return completed
    actual, nyx, _ = _load_baselines(_plan("cwe_residual_2000"))
    references, _ = _load_references()
    origins = annual_origins()[:max_origins]
    expected = grid(str(ANNUAL_FIRST), origins[-1][1])
    experts, fits = {}, {}
    started = time.monotonic()
    for role in ROLE_CONFIGS:
        points, fits[role] = _fit_role(
            role, origins, actual, nyx, thread_count,
            output_dir=output_dir, manifest_sha256=manifest_digest)
        for zone, series in points.items():
            _validate_index(series.index, expected, f"{role}/{zone}")
            require(np.isfinite(series.to_numpy(dtype=float)).all(),
                    f"{role}/{zone}: nonfinite expert point")
            experts[f"{role}_{zone}"] = series
    selected = compose_selected(
        experts, {zone: series.loc[expected] for zone, series in references.items()})
    comparisons, _ = _load_comparisons()
    new_fits = sum(not audit["reused_checkpoint"] for role in fits for audit in fits[role])
    reused_fits = sum(audit["reused_checkpoint"] for role in fits for audit in fits[role])
    if max_origins < 53:
        provisional = {zone: _paired_metrics(selected[zone], comparisons[zone].loc[expected])
                       for zone in COUNTRIES}
        progress = {"protocol": "nyx_selected_cwe_cpu_historical_v1",
                    "action": "partial_annual_replay", "manifest_sha256": manifest_digest,
                    "origins_completed_in_prefix": max_origins, "origins_required": 53,
                    "physical_hours_scored_in_prefix": len(expected),
                    "fit_count_in_prefix": sum(map(len, fits.values())),
                    "new_cpu_fits_this_invocation": new_fits,
                    "reused_checkpoints_this_invocation": reused_fits,
                    "provisional_metrics_diagnostic_only": provisional,
                    "qualification_gate": QUALIFICATION_GATE,
                    "final_verdict_available": False,
                    "annual_score_computed": False,
                    "elapsed_seconds": time.monotonic() - started}
        _atomic_json(output_dir / "progress.json", progress)
        return progress
    scores = {zone: score_selected(selected[zone], comparisons[zone]) for zone in COUNTRIES}
    gpu_scores = _historical_gpu_scores(comparisons)
    training_comparison_difference = {
        zone: {"different_observation_hours": int(np.sum(
            actual[zone].loc[ANNUAL_GRID].to_numpy() != comparisons[zone]["actual"].to_numpy())),
               "mean_absolute_difference": float(np.mean(np.abs(
                   actual[zone].loc[ANNUAL_GRID].to_numpy()
                   - comparisons[zone]["actual"].to_numpy())))}
        for zone in COUNTRIES
    }
    receipt = {"preflight": check, "action": "full_annual_replay",
               "manifest_sha256": manifest_digest,
               "qualification_gate": QUALIFICATION_GATE,
               "selected_recipes": {
                   "FR": "fr_residual_1000 if abs(expert-reference)>=20 else reference",
                   "BE": "mean(cwe_residual_2000,cwe_absolute_2000) if abs(mean-reference)>=20 else reference",
                   "NL": "mean(cwe_residual_2000,cwe_absolute_2000) on every hour"},
               "model_configs": {role: asdict(config) for role, config in ROLE_CONFIGS.items()},
               "fit_audits": fits, "total_cpu_fits": sum(map(len, fits.values())),
               "new_cpu_fits_this_invocation": new_fits,
               "reused_checkpoints_this_invocation": reused_fits,
               "scores_on_official_epex_storm_rows": scores,
               "historical_gpu_scores_on_same_rows": gpu_scores,
               "baseline_vs_official_observation": training_comparison_difference,
               "elapsed_seconds": time.monotonic() - started,
               "GPU_historical_scores_reproduced": False,
               "final_verdict_available": True,
               "qualifies_all_three_countries": all(v["both_criteria_met"] for v in scores.values())}
    receipt["selected_point_sha256"] = {
        zone: _atomic_parquet(output_dir / f"{zone}.parquet",
                              pd.DataFrame({"point": selected[zone]}))
        for zone in COUNTRIES}
    _atomic_json(completed_path, receipt)
    _atomic_text(output_dir / "rapport.md", _markdown_report(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "smoke", "run"), default="preflight")
    parser.add_argument("--role", choices=tuple(ROLE_CONFIGS), default="fr_residual_1000")
    parser.add_argument("--origin", default=str(ANNUAL_FIRST))
    parser.add_argument("--thread-count", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-origins", type=int, default=53,
                        help="First N weekly origins; N<53 yields provisional metrics only")
    args = parser.parse_args()
    require(1 <= args.thread_count <= 128, "Positive CPU thread count required")
    if args.action == "preflight":
        result = preflight()
    elif args.action == "smoke":
        result = smoke(args.role, args.origin, args.thread_count)
    else:
        require(args.output_dir is not None, "--output-dir required for full replay")
        result = run_replay(args.output_dir.resolve(), args.thread_count, args.max_origins)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
