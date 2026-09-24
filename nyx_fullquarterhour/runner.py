"""Daily native CatBoost and Kalman replay, paired with the same hourly chain."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from nyx_intrahour.data import digest, ZONES, HOURLY_ALIASES
from nyx_intrahour.runner import clean
from nyx_quarterhour.data import day_index
from nyx_quarterhour.evaluation import aggregate_quarterhour_predictions, evaluate_predictions
from .data import FullChainInputs
from .raw import run_raw, verify_files, versions
from .storage import identity, safe_path, write_json as _write_json


def write_json(path, value, *, root):
    # Windows readers/antivirus can briefly hold an existing status file.
    # Only retry sharing/permission failures; all other errors remain failures.
    for attempt in range(6):
        try:
            return _write_json(path, value, root=root)
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2 ** attempt)


def _archived_covariate_config(value):
    from chronos2_hourly.kalman_covariates import KalmanCovariateConfig, DerivedCovariateSpec
    expected = {"input_columns", "groups", "derived", "history_missing_policy", "minimum_history_coverage", "require_future_complete"}
    if set(value) != expected:
        raise ValueError("Complete archived Kalman covariate policy required.")
    options = dict(value)
    options["input_columns"] = tuple(options["input_columns"])
    options["groups"] = {name: tuple(columns) for name, columns in options["groups"].items()}
    options["derived"] = tuple(DerivedCovariateSpec(**dict(item, sources=tuple(item["sources"]))) for item in options["derived"])
    config = KalmanCovariateConfig(**options)
    config.validate()
    return config


def _worker(zone: str, frequency: str, directory_text: str, root_text: str, protocol_id: str):
    from threadpoolctl import threadpool_limits
    from chronos2_hourly.kalman_residual import KalmanResidualConfig
    from .residual import build_meta_features, fit_predict_day
    from .kalman import forecast_day

    root, directory = Path(root_text), Path(directory_text)
    protocol = json.loads((directory / "full_protocol.json").read_text(encoding="utf-8"))["protocol"]
    cfg, recipes = protocol["config"], protocol["recipes"]
    record = recipes[zone]
    recipe = dict(record["residual_recipe"], thread_count=cfg["catboost_threads"])
    options = dict(recipe["feature_builder"])
    tz = options["timezone"]
    work = safe_path(root, directory / "workers" / f"{zone}_{frequency}")
    worker_manifest = safe_path(root, work / "manifest.json")
    if worker_manifest.is_file():
        sealed = json.loads(worker_manifest.read_text(encoding="utf-8"))
        if sealed["status"] != "complete" or sealed["protocol_identity"] != protocol_id:
            raise ValueError("Completed worker protocol mismatch.")
        verify_files(work, sealed["files"])
        return {"zone": zone, "frequency": frequency, "directory": str(work), "runtime_seconds": sealed["runtime_seconds"]}
    prices, baseline = (pd.read_parquet(directory / name) for name in ("native_prices.parquet", "baseline.parquet"))
    inputs = FullChainInputs(prices, baseline)
    raw = pd.read_parquet(directory / f"raw_{frequency}.parquet")
    raw = raw.loc[raw.zone.eq(zone)].set_index("timestamp_utc").sort_index()[["q10", "q50", "q90"]]
    raw.index = pd.DatetimeIndex(raw.index).tz_convert("UTC")
    actual = (inputs.quarters[zone] if frequency == "15min" else inputs.hours[zone]).reindex(raw.index)
    if not np.isfinite(actual.to_numpy(float)).all():
        raise ValueError("Complete native observations are required for this replay.")
    features = inputs.residual_features(raw.index, zone, frequency)
    meta = build_meta_features(features, raw, frequency=frequency, timezone=tz,
                               primary_country=options["rich_calendar_primary_country"], feature_builder_options=options)
    work.mkdir(parents=True, exist_ok=True)
    residual_dir = safe_path(root, work / "residual_daily")
    residual_dir.mkdir(exist_ok=True)
    kalman_dir = safe_path(root, work / "kalman_daily")
    kalman_dir.mkdir(exist_ok=True)
    days = pd.Index(raw.index.tz_convert(tz).date)
    unique_days = days.unique()
    residual_parts, residual_audits = [], []
    began = time.perf_counter()
    with threadpool_limits(limits=1):
        for number, day in enumerate(unique_days):
            target_mask = days == day
            train_mask = (days < day) & (days >= day - pd.Timedelta(days=cfg["training_lookback_days"]))
            path = safe_path(root, residual_dir / f"{day}.parquet")
            record_path = safe_path(root, path.with_suffix(".json"))
            if path.exists() or record_path.exists():
                cached = json.loads(record_path.read_text(encoding="utf-8"))
                if cached["protocol_identity"] != protocol_id or digest(path) != cached["data_sha256"]:
                    raise ValueError("Residual checkpoint identity mismatch.")
                predicted, audit = pd.read_parquet(path), cached["audit"]
            else:
                if not train_mask.any():
                    predicted = raw.loc[target_mask].copy()
                    audit = {"delivery_day": str(day), "frequency": frequency, "training_days": 0, "training_rows": 0,
                             "generation_source": "identity_chronos_cold_start", "target_observations_used": False,
                             "causality_violations": 0, "fit_start_day": None, "fit_end_day": None}
                else:
                    predicted, audit = fit_predict_day(meta.loc[train_mask], actual.loc[train_mask], raw.loc[train_mask],
                                                      meta.loc[target_mask], raw.loc[target_mask], recipe=recipe,
                                                      frequency=frequency, timezone=tz,
                                                      max_lookback_days=cfg["training_lookback_days"], minimum_training_days=cfg["minimum_training_days"])
                # The corrector exposes a convenience Series in attrs. Persist
                # numerical columns and JSON audit separately, not pandas objects
                # in parquet's JSON metadata.
                predicted = predicted.copy()
                predicted.attrs = {}
                predicted.to_parquet(path, index=True)
                write_json(record_path, {"protocol_identity": protocol_id, "data_sha256": digest(path), "audit": audit}, root=root)
            if not predicted.index.equals(raw.index[target_mask]) or list(predicted) != ["q10", "q50", "q90"]:
                raise ValueError("Residual checkpoint grid/schema mismatch.")
            if not np.isfinite(predicted.to_numpy(float)).all():
                raise ValueError("Residual nonfinite output.")
            residual_parts.append(predicted)
            residual_audits.append(audit)
            progress = {"phase": "catboost", "zone": zone, "frequency": frequency, "completed_days": number + 1,
                        "total_days": len(unique_days), "day": str(day), "elapsed_seconds": time.perf_counter() - began}
            write_json(work / "status.json", progress, root=root)
            if (number + 1) % 10 == 0 or number + 1 == len(unique_days):
                print(json.dumps(progress), flush=True)
        corrected = pd.concat(residual_parts).sort_index()
        history = pd.DataFrame({"actual": actual, "chronos2__q50": raw.q50}, index=raw.index)
        for quantile in ("q10", "q50", "q90"):
            history[f"residual_corrected__{quantile}"] = corrected[quantile]
        history["residual_correction"] = corrected.q50 - raw.q50
        for alias in HOURLY_ALIASES:
            history[alias] = features[f"known_{alias}_oracle"]
        filter_options = dict(record["kalman_filter_parameters"])
        filter_options["candidate_kinds"] = tuple(filter_options["candidate_kinds"])
        policy = KalmanResidualConfig(**filter_options)
        covariate_policy = _archived_covariate_config(record["kalman_covariate_config"])
        eval_days = pd.date_range(cfg["start_day"], cfg["end_day"], freq="D")
        kalman_parts, kalman_audits = [], []
        for number, timestamp in enumerate(eval_days):
            day = timestamp.date()
            target_mask = days == day
            future = history.loc[target_mask].drop(columns="actual")
            prior = history.loc[days < day]
            path = safe_path(root, kalman_dir / f"{day}.parquet")
            record_path = safe_path(root, path.with_suffix(".json"))
            if path.exists() or record_path.exists():
                cached = json.loads(record_path.read_text(encoding="utf-8"))
                if cached["protocol_identity"] != protocol_id or digest(path) != cached["data_sha256"]:
                    raise ValueError("Kalman checkpoint identity mismatch.")
                if digest(safe_path(root, kalman_dir / f"{day}.candidates.parquet")) != cached["candidates_sha256"]:
                    raise ValueError("Kalman candidate checkpoint changed.")
                predicted, audit = pd.read_parquet(path), cached["audit"]
            else:
                result = forecast_day(prior, future, frequency=frequency, timezone=tz,
                                      config=policy, covariate_config=covariate_policy, lookback_days=cfg["training_lookback_days"])
                predicted, audit = result.predictions, result.audit
                predicted.to_parquet(path, index=True)
                candidates_path = safe_path(root, kalman_dir / f"{day}.candidates.parquet")
                result.candidate_predictions.to_parquet(candidates_path, index=True)
                write_json(record_path, {"protocol_identity": protocol_id, "data_sha256": digest(path),
                           "candidates_sha256": digest(candidates_path), "audit": audit,
                           "states": result.state_audit.to_dict(orient="records")}, root=root)
            if not predicted.index.equals(raw.index[target_mask]):
                raise ValueError("Kalman grid mismatch.")
            kalman_parts.append(predicted)
            kalman_audits.append(audit)
            progress = {"phase": "kalman", "zone": zone, "frequency": frequency, "completed_days": number + 1,
                        "total_days": len(eval_days), "day": str(day), "elapsed_seconds": time.perf_counter() - began}
            write_json(work / "status.json", progress, root=root)
            print(json.dumps(progress), flush=True)
    corrected.to_parquet(safe_path(root, work / "residual.parquet"), index=True)
    pd.concat(kalman_parts).to_parquet(safe_path(root, work / "kalman.parquet"), index=True)
    write_json(work / "residual_audit.json", residual_audits, root=root)
    write_json(work / "kalman_audit.json", kalman_audits, root=root)
    files = {str(p.relative_to(work)): digest(p) for p in sorted(work.rglob("*")) if p.is_file() and p.name != "status.json" and p.name != "manifest.json"}
    write_json(work / "manifest.json", {"status": "complete", "files": files, "protocol_identity": protocol_id,
               "runtime_seconds": time.perf_counter() - began}, root=root)
    write_json(work / "status.json", {"phase": "complete", "zone": zone, "frequency": frequency}, root=root)
    return {"zone": zone, "frequency": frequency, "directory": str(work), "runtime_seconds": time.perf_counter() - began}


def _evaluate(directory: Path, cfg: dict):
    predictions = {}
    for frequency in ("h", "15min"):
        raw = pd.read_parquet(directory / f"raw_{frequency}.parquet")
        all_stages = {"raw": [], "residual": [], "full": []}
        for zone in ZONES:
            work = directory / "workers" / f"{zone}_{frequency}"
            frames = {"raw": raw.loc[raw.zone.eq(zone)].set_index("timestamp_utc"),
                      "residual": pd.read_parquet(work / "residual.parquet"), "full": pd.read_parquet(work / "kalman.parquet")}
            for stage, frame in frames.items():
                local = frame.index.tz_convert("Europe/Paris")
                selected = frame.loc[(local.date >= pd.Timestamp(cfg["start_day"]).date()) & (local.date <= pd.Timestamp(cfg["end_day"]).date())]
                column = "residual_kalman__q50" if stage == "full" else "q50"
                all_stages[stage].append(pd.DataFrame({"timestamp_utc": selected.index, "zone": zone, "prediction": selected[column].to_numpy(float)}))
        for stage, parts in all_stages.items():
            frame = pd.concat(parts, ignore_index=True)
            if frequency == "15min":
                frame = aggregate_quarterhour_predictions(frame, start_day=cfg["start_day"], end_day=cfg["end_day"])
            name = {("full", "h"): "matched_full_hourly", ("full", "15min"): "full_quarterhour"}.get((stage, frequency), f"{stage}_{'hourly' if frequency == 'h' else 'quarterhour'}")
            predictions[name] = frame
    baseline = pd.read_parquet(directory / "baseline.parquet")
    before = min(pd.Timestamp("2026-03-14", tz="Europe/Paris").tz_convert("UTC"), day_index(cfg["start_day"], "h")[0])
    threshold_training = baseline.loc[baseline.timestamp_utc < before]
    q95 = {zone: float(threshold_training.loc[threshold_training.zone.eq(zone), "training_actual"].quantile(.95)) for zone in ZONES}
    return evaluate_predictions(pd.read_parquet(directory / "scoring_baseline.parquet"), predictions, q95_thresholds=q95,
                                candidate_family="full_quarterhour", matched_control_family="matched_full_hourly",
                                start_day=cfg["start_day"], end_day=cfg["end_day"], bootstrap_repetitions=cfg["bootstrap_repetitions"],
                                seed=cfg["bootstrap_seed"], matched_control_verified=True)


def run_postprocessing(config: dict, *, root: Path, directory: Path) -> Path:
    from .reporting import render_report

    root = root.resolve()
    directory = safe_path(root, directory)
    raw_manifest_path = directory / "raw_outputs.manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    if raw_manifest["status"] != "complete":
        raise ValueError("The raw inference must be complete before any learned postprocessing.")
    verify_files(directory, raw_manifest["files"])
    raw_lock = json.loads((directory / "raw_protocol.json").read_text(encoding="utf-8"))
    raw_protocol = raw_lock["protocol"]
    if raw_protocol["config"] != config or versions() != raw_protocol["versions"]:
        raise ValueError("The declared experiment/dependencies differ from the sealed raw run.")
    input_hashes = json.loads((directory / "inputs.manifest.json").read_text(encoding="utf-8"))["files"]
    verify_files(directory, input_hashes)
    paths = sorted((root / "nyx_fullquarterhour").glob("*.py")) + [root / name for name in (
        "chronos2_hourly/models/residual_corrector.py", "chronos2_hourly/features.py", "chronos2_modular/exogenous_extensions.py",
        "chronos2_hourly/kalman_residual.py", "chronos2_hourly/kalman_covariates.py", "chronos2_hourly/nuclear_forecast.py",
        "nyx_quarterhour/evaluation.py")]
    hashes = {str(p.relative_to(root)): digest(p) for p in paths}
    protocol = {"config": config, "recipes": raw_protocol["recipes"], "code_sha256": hashes,
                "raw_manifest_sha256": digest(raw_manifest_path), "versions": versions(),
                "decision_policy": "preset_2pct_gain_vs_NYX_and_matched_control_with_negative_paired_MAE_CI",
                "history_policy": "same_available_complete_days_capped365_for_both_frequencies",
                "selection_used_test_targets": False, "production_modified": False}
    protocol_id = identity(protocol)
    lock = safe_path(root, directory / "full_protocol.json")
    if lock.exists():
        if json.loads(lock.read_text(encoding="utf-8"))["identity"] != protocol_id:
            raise ValueError("Full-chain protocol changed; refusing mixed-code resume.")
    else:
        write_json(lock, {"identity": protocol_id, "protocol": protocol}, root=root)
    began = time.perf_counter()
    summary = {"schema_version": 1, "config": config, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "postprocessing", "architecture": "full_Chronos2_CatBoost_Kalman_native15_vs_matched_hourly",
               "protocol_identity": protocol_id, "raw_protocol_identity": raw_lock["identity"], "diagnostic_only": True,
               "activation_performed": False, "production_modified": False, "production_pit_evidence": False,
               "short_history": True, "raw_runtime_seconds": raw_manifest["runtime_seconds"],
               "decision": {"encouraging": False, "promotion_allowed": False}}
    write_json(directory / "status.json", summary, root=root)
    tables = {}
    try:
        completed = []
        with ProcessPoolExecutor(max_workers=config["worker_count"]) as pool:
            futures = {pool.submit(_worker, zone, frequency, str(directory), str(root), protocol_id): (zone, frequency)
                       for frequency in ("15min", "h") for zone in ZONES}
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    completed.append(future.result())
                write_json(directory / "status.json", {"status": "postprocessing", "completed_workers": completed,
                           "total_workers": len(futures), "elapsed_seconds": time.perf_counter() - began}, root=root)
        for result in completed:
            work = Path(result["directory"])
            manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
            if manifest["protocol_identity"] != protocol_id:
                raise ValueError("Mixed worker protocols.")
            verify_files(work, manifest["files"])
        result = _evaluate(directory, config)
        verify_files(root, hashes)
        verify_files(directory, raw_manifest["files"])
        verify_files(directory, input_hashes)
        if digest(raw_manifest_path) != protocol["raw_manifest_sha256"] or identity(json.loads(lock.read_text(encoding="utf-8"))["protocol"]) != protocol_id:
            raise ValueError("Protocol identity changed during postprocessing.")
        for name, value in result.items():
            if isinstance(value, pd.DataFrame):
                tables[name] = value
                value.to_parquet(safe_path(root, directory / f"{name}.parquet"), index=False)
                value.to_csv(safe_path(root, directory / f"{name}.csv"), index=False)
            else:
                summary[name] = clean(value)
        summary.update(status=result.get("status", "complete"), workers=completed)
    except Exception as exc:
        summary.update(status="failed", reason=f"{type(exc).__name__}: {exc}", results_valid=False,
                       decision={"encouraging": False, "promotion_allowed": False})
        tables = {}
    summary["postprocessing_runtime_seconds"] = time.perf_counter() - began
    summary["runtime_seconds"] = summary["raw_runtime_seconds"] + summary["postprocessing_runtime_seconds"]
    write_json(directory / "summary.json", summary, root=root)
    safe_path(root, directory / "report.html")
    render_report(directory, summary, tables)
    outputs = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob("*"))
               if p.is_file() and p.name not in {"outputs.manifest.json", "status.json"}}
    write_json(directory / "outputs.manifest.json", {"files": outputs, "protocol_identity": protocol_id}, root=root)
    write_json(directory / "status.json", {"status": summary["status"], "reason": summary.get("reason"), "report": str(directory / "report.html")}, root=root)
    return directory


def run(config: dict, *, root: Path, resume: Path | None = None, stage: str = "all"):
    if stage not in {"all", "raw", "postprocess"}:
        raise ValueError("Unsupported stage.")
    directory = run_raw(config, root=root, resume=resume) if stage != "postprocess" else resume
    if directory is None:
        raise ValueError("Postprocessing requires the sealed raw run directory.")
    return directory if stage == "raw" else run_postprocessing(config, root=root, directory=directory)
