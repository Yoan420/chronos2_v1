"""Seal the two completed annual CPU replays without fitting or activating NYX.

The price gate was fixed in the price replay before its annual evaluation:
each of FR, BE and NL must beat Storm on RMSE and win strictly more than half
of the same 8,759 physical hours. The negative replay must reproduce its
sealed probabilities and score all 8,760 hours. No additional score threshold
is selected after seeing the negative results.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
from typing import Any, Iterator

import numpy as np
import pandas as pd

import run_nyx_negative_annual_replay as negative_replay
import run_nyx_selected_cpu_historical as price_replay
from chronos2_hourly import nyx_negative_probability_cpu as negative_model
from chronos2_hourly.nyx_annual_cpu_live import (
    COMPOSITIONS, COUNTRIES, NEGATIVE_THREADS, PRICE_EXPERTS, PRICE_THREADS,
    QUALIFICATION_CODE, QUALIFICATION_PROTOCOL, QUALIFICATION_RECEIPT,
    require,
)
from chronos2_hourly.nyx_annual_live_preflight import load_schema, sha256


ROOT = Path(__file__).resolve().parents[1]
FIRST = "2025-09-24"
STOP = "2026-09-24"
PRICE_HOURS = 8759
NEGATIVE_HOURS = 8760
ORIGINS = 53
EXTRA_CODE = (
    "qualify_nyx_annual_cpu.py",
    "run_nyx_selected_cpu_historical.py",
    "run_nyx_negative_annual_replay.py",
    "chronos2_hourly/nyx_annual_cpu_qualification.py",
)
CODE_FILES = (*QUALIFICATION_CODE, *EXTRA_CODE)


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"Expected JSON object: {path.name}")
    return data


def _near(actual: Any, expected: Any, label: str, tolerance: float = 1e-11) -> None:
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(expected, (int, float)) and not isinstance(expected, bool)
            and math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=tolerance,
                             abs_tol=tolerance), f"{label}: score differs")


def _source(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and relative, "Empty replay source path")
    normalized = relative.replace("\\", "/")
    posix, windows = PurePosixPath(normalized), PureWindowsPath(relative)
    require(not posix.is_absolute() and not windows.drive and ".." not in posix.parts,
            f"Replay source path is not repository-relative: {relative}")
    path = (root / Path(*posix.parts)).resolve()
    require(path.is_relative_to((root / "runs").resolve()),
            f"Replay source must stay inside ignored runs/: {relative}")
    return path


def _specs(value: Any) -> Iterator[dict]:
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        else:
            for child in value.values():
                yield from _specs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _specs(child)


def _verify_sources(root: Path, preflight: dict) -> dict[str, str]:
    require(preflight.get("historical_reference_frozen") is True
            and preflight.get("historical_publication_vintages_certified") is False
            and preflight.get("annual_hours_per_country") == NEGATIVE_HOURS
            and preflight.get("annual_origins") == ORIGINS
            and preflight.get("planned_cpu_fits") == 3 * ORIGINS,
            "Price replay preflight scope differs")
    hashes: dict[str, str] = {}
    for spec in _specs(preflight):
        relative, expected = spec["path"], spec["sha256"]
        require(isinstance(expected, str) and len(expected) == 64,
                "Malformed price source SHA-256")
        path = _source(root, relative)
        require(path.is_file() and sha256(path) == expected,
                f"Price replay source changed: {relative}")
        name = path.relative_to(root).as_posix()
        require(name not in hashes or hashes[name] == expected,
                f"Inconsistent price source checksum: {name}")
        hashes[name] = expected
    require(len(hashes) >= 21, "Price replay source evidence incomplete")
    plan_hashes = preflight.get("archived_plan_sha256")
    schema = load_schema(root / "config/nyx_annual_cpu_ordered_features.json")
    require(plan_hashes == {"fr_residual_1000":
            schema["source_plan_sha256"]["fr_residual_1000"],
            "cwe_2000": schema["source_plan_sha256"]["cwe_residual_2000"]},
            "Historical price plans differ from ordered live schema")
    for role, name in (("fr_residual_1000", "rmse_pooled_gpu_v1"),
                       ("cwe_2000", "rmse_boosting_2000_gpu_v1")):
        path = root / "runs/experiments/nyx_improvement_to20260923" / name / "plan.json"
        require(path.is_file() and sha256(path) == plan_hashes[role],
                f"Historical price plan changed: {name}")
    return hashes


def _runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(), **{name: version(name)
            for name in ("catboost", "numpy", "pandas", "pyarrow")}}


def _price_replay(root: Path, folder: Path) -> dict:
    manifest_path, receipt_path = folder / "run_manifest.json", folder / "receipt.json"
    require(manifest_path.is_file() and receipt_path.is_file(),
            "Full annual price replay manifest and receipt are required")
    manifest, receipt = _read(manifest_path), _read(receipt_path)
    manifest_digest = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    require(manifest.get("protocol") == "nyx_selected_cwe_cpu_historical_v1"
            and receipt.get("action") == "full_annual_replay"
            and receipt.get("manifest_sha256") == manifest_digest
            and receipt.get("preflight") == manifest.get("preflight")
            and receipt.get("qualification_gate") == manifest.get("qualification_gate")
            == price_replay.QUALIFICATION_GATE
            and receipt.get("final_verdict_available") is True
            and receipt.get("total_cpu_fits") == 3 * ORIGINS
            and receipt.get("GPU_historical_scores_reproduced") is False,
            "Price replay is partial or its predeclared gate differs")
    require(manifest.get("thread_count") == PRICE_THREADS
            and manifest.get("first_training_day") == price_replay.FIRST_TRAINING_DAY
            and manifest.get("origins") == [list(pair) for pair in price_replay.annual_origins()]
            and manifest.get("configs") == {role: asdict(config)
                    for role, config in price_replay.ROLE_CONFIGS.items()}
            and manifest.get("role_countries") == {role: list(zones)
                    for role, zones in price_replay.ROLE_COUNTRIES.items()},
            "Price replay calendar, 8-thread CPU recipe, or roles differ")
    require(manifest.get("runner_code_sha256") ==
                sha256(root / "run_nyx_selected_cpu_historical.py")
            and manifest.get("cpu_model_code_sha256") ==
                sha256(root / "chronos2_hourly/nyx_pooled_cpu_price_model.py")
            and manifest.get("dependency_versions") == _runtime_versions(),
            "Price replay code or dependency versions differ")
    source_hashes = _verify_sources(root, manifest["preflight"])
    require(receipt.get("selected_recipes") == {
        "FR": "fr_residual_1000 if abs(expert-reference)>=20 else reference",
        "BE": "mean(cwe_residual_2000,cwe_absolute_2000) if abs(mean-reference)>=20 else reference",
        "NL": "mean(cwe_residual_2000,cwe_absolute_2000) on every hour"},
        "Price compositions differ from frozen selected recipes")

    expected_grid = price_replay.grid(FIRST, STOP)
    require(len(expected_grid) == NEGATIVE_HOURS, "Annual physical UTC grid differs")
    audits = receipt.get("fit_audits")
    require(isinstance(audits, dict) and set(audits) == set(PRICE_EXPERTS),
            "Three annual price expert audit lists required")
    experts: dict[str, pd.Series] = {}
    checkpoint_hashes: dict[str, str] = {}
    for role in PRICE_EXPERTS:
        rows = audits[role]
        require(isinstance(rows, list) and len(rows) == ORIGINS,
                f"{role}: exactly 53 audited fits required")
        pieces = {zone: [] for zone in price_replay.ROLE_COUNTRIES[role]}
        for position, (origin, stop) in enumerate(price_replay.annual_origins()):
            path = folder / "checkpoints" / role / f"{origin}.json"
            require(path.is_file(), f"Missing price checkpoint: {role}/{origin}")
            points, audit = price_replay._read_checkpoint(
                path, role=role, origin=origin, stop=stop,
                manifest_sha256=manifest_digest)
            checkpoint_hashes[path.relative_to(folder).as_posix()] = sha256(path)
            row = rows[position]
            require(isinstance(row, dict) and row.get("reused_checkpoint") in (True, False)
                    and {k: v for k, v in row.items() if k != "reused_checkpoint"} == audit,
                    f"Price receipt/checkpoint audit differs: {role}/{origin}")
            for zone in pieces:
                pieces[zone].append(points[zone])
        for zone, blocks in pieces.items():
            series = pd.concat(blocks)
            require(series.index.equals(expected_grid)
                    and np.isfinite(series.to_numpy(float)).all(),
                    f"{role}/{zone}: annual expert grid differs")
            experts[f"{role}_{zone}"] = series

    references = {}
    for zone in ("FR", "BE"):
        spec = manifest["preflight"]["references"][zone]
        frame = pd.read_parquet(_source(root, spec["path"]), columns=["reference"])
        require(expected_grid.isin(frame.index).all(), f"{zone}: reference year incomplete")
        references[zone] = frame.loc[expected_grid, "reference"].astype(float)
    selected = price_replay.compose_selected(experts, references)
    scores, selected_hashes = {}, {}
    require(set(receipt.get("scores_on_official_epex_storm_rows", {})) == set(COUNTRIES)
            and set(receipt.get("selected_point_sha256", {})) == set(COUNTRIES),
            "FR/BE/NL annual price score and output hashes required")
    for zone in COUNTRIES:
        point_path = folder / f"{zone}.parquet"
        expected_hash = receipt["selected_point_sha256"][zone]
        require(point_path.is_file() and sha256(point_path) == expected_hash,
                f"{zone}: price output SHA-256 differs")
        saved = pd.read_parquet(point_path)
        require(saved.index.equals(expected_grid) and list(saved.columns) == ["point"]
                and np.array_equal(saved["point"].to_numpy(float),
                                   selected[zone].to_numpy(float)),
                f"{zone}: final price points differ from 159 CPU checkpoints")
        spec = manifest["preflight"]["official_comparisons"][zone]
        comparison = pd.read_parquet(_source(root, spec["path"]))
        require(comparison.index.equals(expected_grid)
                and {"actual", "storm"} <= set(comparison),
                f"{zone}: official EPEX/Storm annual grid differs")
        calculated = price_replay.score_selected(selected[zone], comparison)
        claimed = receipt["scores_on_official_epex_storm_rows"][zone]
        require(calculated["paired_hours"] == claimed.get("paired_hours") == PRICE_HOURS
                and calculated["wins_vs_storm"] == claimed.get("wins_vs_storm")
                and calculated["ties_vs_storm"] == claimed.get("ties_vs_storm")
                and calculated["both_criteria_met"] is True
                and claimed.get("both_criteria_met") is True,
                f"{zone}: predeclared price qualification gate failed")
        for key in ("rmse", "storm_rmse", "strict_win_fraction"):
            _near(claimed.get(key), calculated[key], f"{zone}/{key}")
        scores[zone] = {"hours": PRICE_HOURS, "storm_common_hours": PRICE_HOURS,
                        "rmse": calculated["rmse"],
                        "storm_rmse": calculated["storm_rmse"],
                        "strict_win_rate": calculated["strict_win_fraction"],
                        "strict_wins": calculated["wins_vs_storm"]}
        selected_hashes[zone] = expected_hash
    require(receipt.get("qualifies_all_three_countries") is True,
            "Price replay does not qualify all three countries")
    require(_verify_sources(root, manifest["preflight"]) == source_hashes
            and all(sha256(folder / name) == digest
                    for name, digest in checkpoint_hashes.items()),
            "Price replay sources or checkpoints changed during qualification")
    checkpoint_tree = hashlib.sha256(json.dumps(
        checkpoint_hashes, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {"metrics": scores, "receipt_sha256": sha256(receipt_path),
            "manifest_sha256": sha256(manifest_path),
            "selected_point_sha256": selected_hashes,
            "checkpoint_tree_sha256": checkpoint_tree,
            "source_count": len(source_hashes),
            "runtime_versions": manifest["dependency_versions"]}


def _negative_replay(root: Path, folder: Path) -> dict:
    receipt_path = folder / "receipt.json"
    require(receipt_path.is_file(), "Complete annual negative replay receipt required")
    receipt = _read(receipt_path)
    require(receipt.get("state") == "COMPLETE"
            and receipt.get("identity") == "nyx_negative_annual_cpu_replay_20260923_v1"
            and receipt.get("countries") == list(COUNTRIES)
            and receipt.get("origins_per_country") == ORIGINS
            and receipt.get("completed_fits") == receipt.get("total_fits") == 3 * ORIGINS
            and receipt.get("same_historical_period") is True
            and receipt.get("independent_validation") is False
            and receipt.get("source_hashes_unchanged") is True
            and receipt.get("all_metrics_and_series_match") is True,
            "Negative replay is partial or its sealed annual comparison failed")
    status = negative_replay.preflight(root, COUNTRIES)
    require(status.get("ready") is True,
            "Negative replay pinned inputs/code runtime are invalid: "
            + "; ".join(status.get("blockers", [])))
    config = _read(root / "config/nyx_negative_annual_replay.json")
    require(config.get("expected_runtime") == negative_replay._runtime_versions()
            and config.get("origin_step_civil_days") == 7
            and config.get("origins_per_country") == ORIGINS
            and negative_model.catboost_parameters(NEGATIVE_THREADS)["thread_count"] == 2,
            "Negative replay CPU recipe or dependency versions differ")
    evidence = config["evidence"]["metrics"]
    report = _read(_source(root, evidence["path"]))
    annual_grid = negative_model.physical_grid(FIRST, STOP, "FR")
    require(len(annual_grid) == NEGATIVE_HOURS
            and set(receipt.get("comparisons", {})) == set(COUNTRIES),
            "Negative replay annual scope differs")
    metrics, output_hashes = {}, {}
    for zone in COUNTRIES:
        expected = negative_model.physical_grid(FIRST, STOP, zone)
        require(expected.equals(annual_grid), "FR/BE/NL physical annual grids differ")
        output_path = folder / f"{zone}_probabilities.parquet"
        require(output_path.is_file(), f"{zone}: negative replay probabilities missing")
        output = pd.read_parquet(output_path)
        require(output.index.equals(expected) and len(output) == NEGATIVE_HOURS,
                f"{zone}: 8,760 physical negative-probability hours required")
        specs = config["sources"][zone]
        baseline = pd.read_parquet(_source(root, specs["baseline"]["path"]))
        archived = pd.read_parquet(_source(root, specs["archived_probabilities"]["path"]))
        require(expected.isin(baseline.index).all() and archived.index.equals(expected),
                f"{zone}: negative score observations/archives incomplete")
        recomputed = negative_replay._score_comparison(
            baseline["actual"].loc[expected], output, archived,
            report["countries"][zone])
        claimed = receipt["comparisons"][zone]
        require(recomputed["passed"] is True and claimed.get("passed") is True
                and claimed.get("fits") == ORIGINS
                and claimed.get("series") == recomputed["series"]
                and claimed.get("archived") == recomputed["archived"]
                and claimed.get("measured") == recomputed["measured"],
                f"{zone}: negative replay series or scores differ")
        measured = recomputed["measured"]
        require(measured["hours"] == NEGATIVE_HOURS
                and 0 <= measured["brier"] <= 1
                and math.isfinite(measured["brier"]),
                f"{zone}: invalid annual Brier or physical coverage")
        metrics[zone] = {"hours": NEGATIVE_HOURS,
                         "negative_hours": measured["negative_hours"],
                         "brier": measured["brier"]}
        output_hashes[zone] = sha256(output_path)
    require(negative_replay.preflight(root, COUNTRIES).get("ready") is True,
            "Negative replay inputs changed during qualification")
    return {"metrics": metrics, "receipt_sha256": sha256(receipt_path),
            "output_sha256": output_hashes,
            "config_sha256": sha256(root / "config/nyx_negative_annual_replay.json"),
            "runtime_versions": config["expected_runtime"]}


def prepare_receipt(root: Path, price_folder: Path, negative_folder: Path) -> dict:
    """Read and independently reconstruct both replay score summaries."""
    root, price_folder, negative_folder = (path.resolve() for path in
                                            (root, price_folder, negative_folder))
    runs = (root / "runs").resolve()
    require(price_folder.is_relative_to(runs)
            and negative_folder.is_relative_to(runs)
            and price_folder != negative_folder,
            "Distinct replay folders inside repository runs/ are required")
    price = _price_replay(root, price_folder)
    negative = _negative_replay(root, negative_folder)
    code_hashes = {name: sha256(root / name) for name in CODE_FILES}
    return {"protocol": QUALIFICATION_PROTOCOL,
            "qualified": False,
            "price_expert_replay_qualified": True,
            "negative_replay_verified": True,
            "full_input_chain_qualified": False,
            "price_experts": list(PRICE_EXPERTS),
            "negative_model_protocol": negative_model.PROTOCOL,
            "compositions": COMPOSITIONS.copy(),
            "first_delivery_day": FIRST,
            "last_delivery_day": str(date.fromisoformat(STOP) - timedelta(days=1)),
            "origins_per_country": ORIGINS,
            "price_threads": PRICE_THREADS, "negative_threads": NEGATIVE_THREADS,
            "replay_receipts_sha256": {
                "price": price["receipt_sha256"],
                "negative": negative["receipt_sha256"]},
            "price_country_metrics": price["metrics"],
            "negative_country_metrics": negative["metrics"],
            "code_sha256": code_hashes,
            "evidence_sha256": {
                "price_run_manifest": price["manifest_sha256"],
                "price_checkpoint_tree": price["checkpoint_tree_sha256"],
                "price_selected_points": price["selected_point_sha256"],
                "negative_probabilities": negative["output_sha256"],
                "negative_replay_config": negative["config_sha256"],
                "ordered_feature_schema": sha256(root / "config/nyx_annual_cpu_ordered_features.json")},
            "runtime_versions": {"price": price["runtime_versions"],
                                 "negative": negative["runtime_versions"]},
            "price_source_artifacts_verified": price["source_count"],
            "retrospective_evaluation": True,
            "historical_publication_vintages_certified": False,
            "negative_replay_execution_code_sha_embedded": False,
            "prospective_input_bundle_required": True,
            "manifest_activated": False}


def seal_receipt(root: Path, price_folder: Path, negative_folder: Path) -> dict:
    """Write the fixed canonical receipt once, only after complete validation."""
    root = root.resolve()
    destination = root / "config" / QUALIFICATION_RECEIPT.name
    require(not destination.exists(), "Canonical CPU qualification receipt already exists")
    price_receipt, negative_receipt = (folder.resolve() / "receipt.json"
                                       for folder in (price_folder, negative_folder))
    first = (sha256(price_receipt), sha256(negative_receipt))
    payload = prepare_receipt(root, price_folder, negative_folder)
    require(first == (sha256(price_receipt), sha256(negative_receipt)),
            "Replay receipt changed during qualification")
    require(all(sha256(root / name) == digest
                for name, digest in payload["code_sha256"].items())
            and all(sha256(price_folder.resolve() / f"{zone}.parquet") == digest
                    for zone, digest in payload["evidence_sha256"]["price_selected_points"].items())
            and all(sha256(negative_folder.resolve() / f"{zone}_probabilities.parquet") == digest
                    for zone, digest in payload["evidence_sha256"]["negative_probabilities"].items()),
            "Code or replay outputs changed before qualification was sealed")
    require(not destination.exists(), "Canonical CPU qualification receipt already exists")
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          indent=2, allow_nan=False) + "\n").encode("utf-8")
    # Exclusive creation refuses an existing qualification, including a race.
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {"price_expert_replay_qualified": True,
            "full_input_chain_qualified": False,
            "qualified": False, "path": str(destination),
            "sha256": sha256(destination), "receipt": payload}
