"""Replay the archived annual negative-price CPU recipe on pinned local inputs.

This is a retrospective audit of the same 2025-09-24 to 2026-09-23 period. It
does not fetch future features, train the price model, or activate NYX forecasts.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
from typing import Any

import numpy as np
import pandas as pd

from chronos2_hourly import nyx_negative_probability_cpu as negative


CONFIG_PATH = Path("config/nyx_negative_annual_replay.json")
SUPPORTED_COUNTRIES = ("FR", "BE", "NL")
METRIC_TOLERANCE = 1e-12
SERIES_TOLERANCE = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if not relative or posix.is_absolute() or windows.drive or ".." in posix.parts or "\\" in relative:
        raise ValueError(f"Source path must be repository-relative: {relative!r}")
    path = (root / Path(*posix.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes checkout: {relative!r}")
    return path


def _file_status(root: Path, spec: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {"path": spec.get("path"), "status": "invalid_spec"}
    expected = spec.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return result
    try:
        path = _source_path(root, spec["path"])
    except (ValueError, KeyError, TypeError) as exc:
        result["detail"] = str(exc)
        return result
    if not path.exists():
        result["status"] = "missing"
    elif not path.is_file():
        result["status"] = "not_a_file"
    elif path.stat().st_size == 0:
        result["status"] = "empty"
    else:
        actual = _sha256(path)
        result["status"] = "verified" if actual == expected else "sha256_mismatch"
        if actual != expected:
            result["actual_sha256"] = actual
    return result


def _load_config(root: Path) -> dict[str, Any]:
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("identity") != "nyx_negative_annual_cpu_replay_20260923_v1"
        or config.get("countries") != list(SUPPORTED_COUNTRIES)
        or config.get("first_origin_day") != "2025-09-24"
        or config.get("stop_day_exclusive") != "2026-09-24"
        or config.get("origin_step_civil_days") != 7
        or config.get("origins_per_country") != 53
    ):
        raise ValueError("Annual replay config identity or fixed calendar changed")
    return config


def annual_origins(config: dict[str, Any]) -> tuple[date, ...]:
    first = date.fromisoformat(config["first_origin_day"])
    stop = date.fromisoformat(config["stop_day_exclusive"])
    step = timedelta(days=config["origin_step_civil_days"])
    origins = []
    origin = first
    while origin < stop:
        origins.append(origin)
        origin += step
    if len(origins) != config["origins_per_country"] or origins[-1] != date(2026, 9, 23):
        raise ValueError("Expected exactly 53 fixed weekly origins ending 2026-09-23")
    return tuple(origins)


def _runtime_versions() -> dict[str, str]:
    return {name: version(name) for name in ("catboost", "numpy", "pandas", "scikit-learn")}


def _check_provenance(root: Path, config: dict[str, Any], countries: tuple[str, ...]) -> None:
    study = config["evidence"]
    plan = json.loads(_source_path(root, study["plan"]["path"]).read_text(encoding="utf-8"))
    seal = json.loads(_source_path(root, study["prediction_seal"]["path"]).read_text(encoding="utf-8"))
    evaluation = json.loads(_source_path(root, study["evaluation_receipt"]["path"]).read_text(encoding="utf-8"))
    report = json.loads(_source_path(root, study["metrics"]["path"]).read_text(encoding="utf-8"))
    contract = plan["contract"]
    expected = negative.catboost_parameters(2)
    if not (
        plan["protocol"] == "nyx_negative_prices_weekly_v1"
        and plan["origins"] == [str(origin) for origin in annual_origins(config)]
        and plan["runtime"] == config["expected_runtime"] == _runtime_versions()
        and contract["feature_columns_per_country"] == negative.FEATURE_COUNT
        and contract["training_civil_days"] == 365
        and contract["classifier_days"] == 337
        and contract["calibration_days"] == 28
        and contract["minimum_calibration_events_per_class"] == 10
        and contract["iterations"] == expected["iterations"]
        and contract["depth"] == expected["depth"]
        and contract["learning_rate"] == expected["learning_rate"]
        and contract["l2_leaf_reg"] == expected["l2_leaf_reg"]
        and contract["seed"] == expected["random_seed"]
        and contract["task_type"] == expected["task_type"] == "CPU"
        and contract["thread_count"] == 2
        and contract["alert_threshold"] == negative.ALERT_THRESHOLD
        and seal["state"] == "NEGATIVE_PROBABILITIES_SEALED_BEFORE_EVALUATION"
        and seal["plan_sha256"] == study["plan"]["sha256"]
        and evaluation["state"] == "COMPLETE"
        and evaluation["metrics_sha256"] == study["metrics"]["sha256"]
        and report["hours_per_country"] == 8760
        and report["retrospective_evaluation"] is True
        and report["independent_validation"] is False
    ):
        raise ValueError("Archived recipe, runtime, calendar, or evaluation seal differs")
    first_source = date.fromisoformat(config["source_first_day"])
    stop = date.fromisoformat(config["stop_day_exclusive"])
    first_eval = date.fromisoformat(config["first_origin_day"])
    for zone in countries:
        source = config["sources"][zone]
        if (
            plan["features"][zone]["sha256"] != source["features"]["sha256"]
            or plan["baselines"][zone]["sha256"] != source["baseline"]["sha256"]
            or seal["files_sha256"][f"annual/{zone}.parquet"] != source["archived_probabilities"]["sha256"]
        ):
            raise ValueError(f"{zone} source identities differ from archived plan/seal")
        features = pd.read_parquet(_source_path(root, source["features"]["path"]))
        baseline = pd.read_parquet(_source_path(root, source["baseline"]["path"]))
        annual = pd.read_parquet(_source_path(root, source["archived_probabilities"]["path"]))
        expected_source = negative.physical_grid(first_source, stop, zone)
        expected_eval = negative.physical_grid(first_eval, stop, zone)
        if not (
            features.index.equals(expected_source)
            and baseline.index.equals(expected_source)
            and annual.index.equals(expected_eval)
            and len(features) == len(baseline) == 17880
            and len(annual) == 8760
            and list(features.columns) == plan["features"][zone]["columns"]
            and len(features.columns) == negative.FEATURE_COUNT
            and {"actual", "nyx__q50"}.issubset(baseline.columns)
            and {"p_raw", "p_negative", "is_negative_predicted"}.issubset(annual.columns)
            and np.isfinite(baseline["actual"].to_numpy(dtype=float)).all()
            and np.isfinite(annual[["p_raw", "p_negative"]].to_numpy(dtype=float)).all()
            and annual["p_raw"].between(0.0, 1.0).all()
            and annual["p_negative"].between(0.0, 1.0).all()
            and zone in report["countries"]
        ):
            raise ValueError(f"{zone} feature, observation, or archived output schema differs")
        # A full-year grid check also catches missing/repeated physical DST hours.
        for origin in annual_origins(config):
            history = negative.physical_grid(origin - timedelta(days=365), origin, zone)
            if not history.isin(features.index).all() or not history.isin(baseline.index).all():
                raise ValueError(f"{zone} history is incomplete at origin {origin}")


def preflight(root: Path, countries: tuple[str, ...] = SUPPORTED_COUNTRIES) -> dict[str, Any]:
    """Verify pinned sources and schemas without fitting or writing anything."""

    root = root.resolve()
    try:
        config = _load_config(root)
        if not countries or len(countries) != len(set(countries)) or set(countries) - set(SUPPORTED_COUNTRIES):
            raise ValueError("Select one or more distinct FR/BE/NL countries")
        files = {"evidence": {}, "sources": {}}
        for key, spec in config["evidence"].items():
            files["evidence"][key] = _file_status(root, spec)
        for zone in countries:
            files["sources"][zone] = {
                key: _file_status(root, spec) for key, spec in config["sources"][zone].items()
            }
        bad = [
            item["path"]
            for group in (files["evidence"], *(files["sources"][zone] for zone in countries))
            for item in group.values()
            if item["status"] != "verified"
        ]
        if bad:
            return {"ready": False, "countries": list(countries), "files": files, "blockers": [f"Unverified source: {item}" for item in bad]}
        _check_provenance(root, config, countries)
        return {"ready": True, "countries": list(countries), "files": files, "origins_per_country": 53, "blockers": []}
    except (OSError, KeyError, TypeError, ValueError, ImportError) as exc:
        return {"ready": False, "countries": list(countries), "blockers": [str(exc)]}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _score_comparison(observed: pd.Series, predictions: pd.DataFrame, archived: pd.DataFrame, report: dict[str, Any]) -> dict[str, Any]:
    measured = negative.probability_metrics(observed, predictions["p_negative"])
    tp, fp, fn = measured["true_positive"], measured["false_positive"], measured["false_negative"]
    measured["precision"] = tp / (tp + fp) if tp + fp else None
    measured["recall"] = tp / (tp + fn) if tp + fn else None
    reference = report["models"]["model"]
    metric_difference = {}
    for name in ("brier", "log_loss", "average_precision", "precision", "recall"):
        actual_value, expected_value = measured[name], reference[name]
        metric_difference[name] = None if actual_value is None and expected_value is None else abs(actual_value - expected_value)
    index_equal = predictions.index.equals(archived.index)
    raw_difference = float(np.max(np.abs(predictions["p_negative_raw"].to_numpy() - archived["p_raw"].to_numpy()))) if index_equal else None
    probability_difference = float(np.max(np.abs(predictions["p_negative"].to_numpy() - archived["p_negative"].to_numpy()))) if index_equal else None
    alert_equal = bool(np.array_equal(predictions["is_negative_predicted"].to_numpy(), archived["is_negative_predicted"].to_numpy())) if index_equal else False
    passed = bool(
        index_equal
        and alert_equal
        and raw_difference <= SERIES_TOLERANCE
        and probability_difference <= SERIES_TOLERANCE
        and all(value is None or value <= METRIC_TOLERANCE for value in metric_difference.values())
        and measured["hours"] == reference["n"]
        and measured["negative_hours"] == reference["negative_hours"]
    )
    return {
        "passed": passed,
        "measured": measured,
        "archived": {key: reference[key] for key in ("n", "negative_hours", "brier", "log_loss", "average_precision", "precision", "recall")},
        "absolute_metric_differences": metric_difference,
        "series": {
            "physical_index_equal": index_equal,
            "maximum_raw_probability_difference": raw_difference,
            "maximum_calibrated_probability_difference": probability_difference,
            "alerts_equal": alert_equal,
        },
    }


def run_replay(root: Path, output: Path, countries: tuple[str, ...] = SUPPORTED_COUNTRIES, threads: int = 2) -> dict[str, Any]:
    """Run 53 past-only CPU fits per country after a complete read-only preflight."""

    negative.catboost_parameters(threads)
    root = root.resolve()
    status = preflight(root, countries)
    if not status["ready"]:
        raise ValueError("Replay preflight failed: " + "; ".join(status["blockers"]))
    if output.exists():
        raise FileExistsError(f"Replay output already exists: {output}")
    config = _load_config(root)
    origins = annual_origins(config)
    report = json.loads(_source_path(root, config["evidence"]["metrics"]["path"]).read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    completed = 0
    total = len(origins) * len(countries)
    comparisons: dict[str, Any] = {}
    try:
        for zone in countries:
            specs = config["sources"][zone]
            features = pd.read_parquet(_source_path(root, specs["features"]["path"]))
            baseline = pd.read_parquet(_source_path(root, specs["baseline"]["path"]))
            archived = pd.read_parquet(_source_path(root, specs["archived_probabilities"]["path"]))
            pieces = []
            fit_seconds = []
            for origin in origins:
                stop = min(origin + timedelta(days=7), date.fromisoformat(config["stop_day_exclusive"]))
                history = negative.physical_grid(origin - timedelta(days=365), origin, zone)
                current = negative.physical_grid(origin, stop, zone)
                block = negative.fit_predict_block(
                    features.loc[history], baseline["actual"].loc[history], features.loc[current],
                    zone=zone, origin_day=origin, stop_day=stop, threads=threads,
                )
                pieces.append(block.probabilities)
                fit_seconds.append(block.audit["fit_seconds"])
                completed += 1
                progress = {"state": "RUNNING", "completed": completed, "total": total, "zone": zone, "origin_day": str(origin)}
                _atomic_json(output / "status.json", progress)
                print(json.dumps(progress), flush=True)
            predictions = pd.concat(pieces)
            expected = negative.physical_grid(config["first_origin_day"], config["stop_day_exclusive"], zone)
            if not predictions.index.equals(expected) or len(predictions) != 8760:
                raise ValueError(f"{zone} replay did not cover each annual physical hour exactly once")
            predictions.to_parquet(output / f"{zone}_probabilities.parquet")
            comparison = _score_comparison(baseline["actual"].loc[expected], predictions, archived, report["countries"][zone])
            comparison["fits"] = len(origins)
            comparison["fit_seconds_total"] = float(sum(fit_seconds))
            comparison["fit_seconds_median"] = float(np.median(fit_seconds))
            comparisons[zone] = comparison
            _atomic_json(output / f"{zone}_comparison.json", comparison)
        postflight = preflight(root, countries)
        all_match = bool(postflight["ready"] and all(item["passed"] for item in comparisons.values()))
        receipt = {
            "state": "COMPLETE" if all_match else "FAILED",
            "identity": config["identity"],
            "countries": list(countries),
            "origins_per_country": len(origins),
            "completed_fits": completed,
            "total_fits": total,
            "same_historical_period": True,
            "independent_validation": False,
            "source_hashes_unchanged": postflight["ready"],
            "all_metrics_and_series_match": all_match,
            "comparisons": comparisons,
        }
        _atomic_json(output / "receipt.json", receipt)
        _atomic_json(output / "status.json", {"state": receipt["state"], "completed": completed, "total": total})
        return receipt
    except Exception as exc:
        _atomic_json(output / "status.json", {"state": "FAILED", "completed": completed, "total": total, "error": str(exc)})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--countries", nargs="+", choices=SUPPORTED_COUNTRIES, default=list(SUPPORTED_COUNTRIES))
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--preflight", action="store_true", help="verify sources and schema without fitting or writing")
    parser.add_argument("--output", type=Path, help="new output directory; required for replay")
    args = parser.parse_args(argv)
    countries = tuple(args.countries)
    if args.preflight:
        status = preflight(args.root, countries)
        print(json.dumps(status, ensure_ascii=False))
        return 0 if status["ready"] else 2
    if args.output is None:
        parser.error("--output is required for replay")
    try:
        result = run_replay(args.root, args.output, countries, args.threads)
    except (OSError, ValueError) as exc:
        print(json.dumps({"state": "FAILED", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"state": result["state"], "receipt": str(args.output / "receipt.json")}), flush=True)
    return 0 if result["state"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
