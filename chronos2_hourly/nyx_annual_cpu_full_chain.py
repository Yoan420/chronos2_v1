"""Chronological evaluation and activation of the complete four-country chain.

Every delivery day is predicted from a separately validated producer bundle.
Current-day labels must be absent in each bundle. Official labels and Storm are
opened only after all predictions have been sealed. The old conditional expert
replay cannot be promoted using this code: new baseline/feature/reference
producer evidence is required on every day of a complete 365-day evaluation.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

from chronos2_hourly import nyx_annual_cpu_live as live
from chronos2_hourly.nyx_annual_live_preflight import (
    MATERIALIZER_CODE, delivery_grid, inspect_bundle, sha256,
    TARGET_HISTORY_POLICY, LEGACY_TARGET_HISTORY_POLICY, TARGET_HISTORY_FIELDS,
    validate_target_history_contract,
)
from chronos2_hourly.nyx_annual_nyx_quantiles_gate import validate_nyx_quantiles_source
from chronos2_hourly.nyx_local_io import promote_directory_retry, replace_retry
from chronos2_hourly.nyx_negative_probability_cpu import probability_metrics
from chronos2_hourly.process_lock import exclusive_process_lock

PROTOCOL = "nyx_annual_cpu_full_chain_qualification_v2"
PLAN_PROTOCOL = "nyx_annual_cpu_full_chain_plan_v1"
EVALUATION_DAYS = 365
CODE_HASH_POLICY = "utf8_crlf_to_lf_v1"
DE_EXCEPTION = {"country": "DE", "price_performance_exception_authorized": True,
                "input_chain_exception_authorized": False}
CODE = (*live.QUALIFICATION_CODE,
    "chronos2_hourly/nyx_annual_cpu_reporting.py",
    "chronos2_hourly/nyx_annual_cpu_full_chain.py",
    "evaluate_nyx_annual_cpu_full_chain.py", "qualify_nyx_annual_cpu.py",
    *MATERIALIZER_CODE,
    "chronos2_hourly/nyx_annual_source_validation.py",
    "chronos2_hourly/nyx_annual_saturn_source.py",
    "chronos2_hourly/nyx_annual_jao_source.py",
    "chronos2_hourly/nyx_annual_jao_history.py",
    "chronos2_hourly/nyx_annual_public_history.py",
    "chronos2_hourly/jao_flowbased.py",
    "chronos2_modular/saturn.py",
    "chronos2_modular/common.py", "chronos2_modular/data.py",
    "run_nyx_annual_auction_prices_source.py",
    "run_nyx_annual_exchange_source.py", "run_nyx_annual_hydro_source.py",
    "run_nyx_annual_fuel_source.py", "run_nyx_annual_thermal_source.py",
    "materialize_saturn_kalman_fuel.py", "marginal_cost_expert/sources.py",
    "chronos2_hourly/solar_wind_forecast.py", "chronos2_hourly/chronos_adapter.py",
    "chronos2_hourly_fr_residual_v1.yaml",
    "chronos2_hourly_de_residual_candidate_v1.yaml",
    "chronos2_hourly_be_residual_candidate_v1.yaml",
    "chronos2_hourly_nl_residual_candidate_v1.yaml", "config/kalman_operational.yaml",
    "config/nyx_historical_hgb_feature_schemas.json",
    "config/nyx_annual_cpu_ordered_features.json",
    "export_nyx_annual_comparisons.py",
    "chronos2_hourly/nuclear_reporting_refresh.py",
    "chronos2_hourly/nuclear_report_benchmark.py",
    "chronos2_hourly/process_lock.py",
    "chronos2_hourly/nyx_annual_cpu_baseline.py",
    "chronos2_hourly/nyx_annual_cpu_reference_builder.py")


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    live.require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        replace_retry(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _versions() -> dict:
    return {**live._runtime_versions(), "baseline": {
        name: version(name) for name in ("torch", "transformers", "chronos-forecasting",
            "pykalman", "scipy", "holidays", "joblib", "PyYAML")}}


def _code_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _code(root: Path) -> dict:
    live.require(root.resolve() == Path(__file__).resolve().parents[1],
                 "Full-chain qualification must execute from the checkout being qualified")
    from chronos2_hourly.nyx_annual_cpu_baseline import CODE_FILES as BASELINE_CODE
    from chronos2_hourly.nyx_annual_cpu_reference_builder import CODE_FILES as REFERENCE_CODE
    # Git's Windows checkout newline policy must not change a CPU recipe pin.
    # Source data, receipts, predictions and model weights keep binary SHA-256.
    return {name: _code_sha256(root / name) for name in
            sorted(set((*CODE, *BASELINE_CODE, *REFERENCE_CODE)))}


def _source_packet(bundle: Path, day: str) -> dict:
    from chronos2_hourly.nyx_annual_source_validation import validate_source_packet
    from chronos2_hourly.nyx_annual_cpu_baseline import validate_cpu_baseline_evidence
    from chronos2_hourly.nyx_annual_cpu_reference_builder import validate_cpu_reference_source
    from chronos2_hourly.nyx_annual_saturn_source import target_history_contract
    sources = validate_source_packet(bundle, day)
    live.require(sources.get("passed") is True
        and sources.get("source_snapshot_asof_verified") is True,
        f"{day}: independently checked raw source evidence required")
    contract = target_history_contract(bundle)
    validate_target_history_contract(contract, delivery_grid(day)[2])
    baseline = validate_cpu_baseline_evidence(bundle, day)
    reference = validate_cpu_reference_source(bundle, day)
    for label, producer in (("baseline", baseline), ("reference", reference)):
        declared = {key: producer[key] for key in TARGET_HISTORY_FIELDS if key in producer}
        live.require(declared == contract, f"{day}: {label} target history contract differs from its source")
    return {"sources": sources, "baseline": baseline, "reference": reference, **contract}


def _target_policy(record: dict) -> str:
    """Normalize old strict receipts without silently treating them as current-fit."""
    policy = record.get("target_history_policy", LEGACY_TARGET_HISTORY_POLICY)
    live.require(policy in (LEGACY_TARGET_HISTORY_POLICY, TARGET_HISTORY_POLICY),
                 "Unsupported evaluation target history policy")
    return policy


def _chronos_pin(record: dict) -> dict:
    """Bind evaluated model weights across countries, dates and live forecasts."""
    from chronos2_hourly.nyx_annual_cpu_baseline import chronos_identity_digest
    identity, digest = record.get("chronos_model_identity"), record.get("chronos_model_sha256")
    live.require(chronos_identity_digest(identity) == digest,
                 "Pinned Chronos weight identity checksum differs")
    return {"chronos_model_identity": identity, "chronos_model_sha256": digest}


def _official_comparisons(directory: Path, first: str, stop: str) -> dict:
    # This validator opens the labels: call only after all predictions seal.
    from export_nyx_annual_comparisons import validate_comparisons
    return validate_comparisons(directory, first, stop)


def _days(first: str, stop: str) -> list[str]:
    start, end = date.fromisoformat(first), date.fromisoformat(stop)
    live.require(0 < (end-start).days <= EVALUATION_DAYS,
                 "Evaluation requires 1 to 365 consecutive delivery days")
    return [(start + timedelta(days=i)).isoformat() for i in range((end-start).days)]


def prepare_plan(*, root: Path, bundles: Path, comparisons: Path, output: Path,
                 first: str, stop: str) -> dict:
    """Freeze causal inputs and implementation before any evaluation fitting."""
    live.require(not output.exists(), "Evaluation directory already exists")
    days = _days(first, stop)
    entries, chronos_pin, target_policy = {}, None, None
    # Read-only preflight of every requested day before spending CPU time.
    for day in days:
        bundle = bundles / day
        report = inspect_bundle(bundle, day)
        live.require(report["input_bundle_valid"],
                     f"{day}: producer bundle is not source-qualified: {report.get('checks', [])}")
        validate_nyx_quantiles_source(bundle, day)
        packet = _source_packet(bundle, day)
        daily_pin = _chronos_pin(packet["baseline"])
        daily_policy = _target_policy(packet)
        live.require(target_policy is None or target_policy == daily_policy,
                     f"{day}: target history policies differ across evaluation days")
        target_policy = daily_policy
        live.require(chronos_pin is None or chronos_pin == daily_pin,
                     f"{day}: Chronos model weights differ from other evaluation days")
        chronos_pin = daily_pin
        entries[day] = {"path": str(bundle.resolve()),
                        "sha256": live.bundle_hashes(bundle)}
    observations = {}
    comparisons_receipt = comparisons / "comparisons_receipt.json"
    live.require(comparisons_receipt.is_file(),
        "Official comparison source receipt missing; run export_nyx_annual_comparisons.py")
    for zone in live.COUNTRIES:
        path = comparisons / f"{zone}.parquet"
        live.require(path.is_file(), f"Official EPEX/Storm comparison missing: {path}")
        observations[zone] = {"path": str(path.resolve()), "sha256": sha256(path)}
    payload = {"protocol": PLAN_PROTOCOL, "first_delivery_day": first,
        "stop_day_exclusive": stop, "delivery_days": days,
        "countries": list(live.COUNTRIES), "bundles": entries,
        "comparisons_receipt": {"path": str(comparisons_receipt.resolve()),
                                "sha256": sha256(comparisons_receipt)},
        "comparisons": observations, "code_sha256": _code(root), "code_hash_policy": CODE_HASH_POLICY,
        "runtime_versions": _versions(), "compositions": live.COMPOSITIONS,
        **chronos_pin,
        "target_history_policy": target_policy,
        "de_exception": DE_EXCEPTION, "price_threads": live.PRICE_THREADS,
        "negative_threads": live.NEGATIVE_THREADS,
        "frequency": "daily_retraining", "production_qualification_days": EVALUATION_DAYS}
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", payload)
    return payload


def _plan(root: Path, output: Path) -> dict:
    plan = _read(output / "plan.json")
    _chronos_pin(plan)
    _target_policy(plan)
    live.require(plan.get("protocol") == PLAN_PROTOCOL
        and plan.get("delivery_days") == _days(plan["first_delivery_day"], plan["stop_day_exclusive"])
        and plan.get("countries") == list(live.COUNTRIES)
        and plan.get("compositions") == live.COMPOSITIONS
        and plan.get("code_sha256") == _code(root)
        and plan.get("code_hash_policy") == CODE_HASH_POLICY
        and plan.get("runtime_versions") == _versions()
        and plan.get("de_exception") == DE_EXCEPTION
        and plan.get("price_threads") == live.PRICE_THREADS
        and plan.get("negative_threads") == live.NEGATIVE_THREADS
        and plan.get("frequency") == "daily_retraining"
        and plan.get("production_qualification_days") == EVALUATION_DAYS,
        "Full-chain evaluation implementation, runtime or frozen protocol changed")
    return plan


def _bundle(plan: dict, day: str) -> Path:
    entry = plan["bundles"][day]
    bundle = Path(entry["path"])
    live.require(live.bundle_hashes(bundle) == entry["sha256"],
                 f"{day}: source-qualified bundle changed")
    report = inspect_bundle(bundle, day)
    live.require(report["input_bundle_valid"], f"{day}: producer bundle no longer validates")
    validate_nyx_quantiles_source(bundle, day)
    packet = _source_packet(bundle, day)
    live.require(_target_policy(packet) == _target_policy(plan),
                 f"{day}: target history policy differs from the frozen evaluation plan")
    live.require(_chronos_pin(packet["baseline"]) == _chronos_pin(plan),
                 f"{day}: CPU Chronos weights differ from the frozen evaluation plan")
    return bundle


def _prediction_receipt(output: Path, day: str, plan_hash: str, *, folder: Path | None = None) -> dict:
    folder = folder or output / "predictions" / day
    receipt = _read(folder / "receipt.json")
    live.require(receipt.get("plan_sha256") == plan_hash
        and receipt.get("delivery_day") == day
        and receipt.get("predictions_sealed_before_scoring") is True
        and set(receipt.get("outputs_sha256", {})) == set(live.COUNTRIES),
        f"{day}: prediction seal identity invalid")
    for zone, digest in receipt["outputs_sha256"].items():
        live.require(sha256(folder / f"{zone}.parquet") == digest,
                     f"{day}/{zone}: sealed predictions changed")
    audit = receipt["models"]
    live.require(set(audit["price_experts"]) == set(live.PRICE_EXPERTS)
        and set(audit["negative_countries"]) == set(live.COUNTRIES)
        and audit["compositions"] == live.COMPOSITIONS,
        f"{day}: incomplete four-country model execution")
    for family, item in audit["price_experts"].items():
        live.require(item["training_labels_before_origin"] is True
            and item["training_days"] == 365
            and item["Storm_used_as_input"] is False
            and item["tree_count"] == (1000 if family == "fr_residual_1000" else 2000)
            and item["model"]["sha256"] == sha256(folder / "models" / f"{family}.cbm"),
            f"{day}/{family}: price fit lineage invalid")
    for zone, item in audit["negative_countries"].items():
        live.require(item["forecast_labels_used"] is False and item["storm_used"] is False
            and item["tree_count"] == 120 and item["models_fitted"] == 1
            and item["model"]["sha256"] == sha256(folder / "models" / f"negative_{zone}.cbm"),
            f"{day}/{zone}: classifier fit lineage invalid")
    return receipt


def _predict_day(root: Path, output: Path, plan: dict, day: str, digest: str, bundle: Path) -> None:
    folder = output / "predictions" / day
    with exclusive_process_lock(output / "locks" / f"{day}.lock"):
        if (folder / "receipt.json").is_file():
            _prediction_receipt(output, day, digest)
            return
        attempts = output / "attempts" / day
        attempts.mkdir(parents=True, exist_ok=True)
        if folder.exists():
            # Preserve an unfinished directory written by an older evaluator.
            abandoned = attempts / f"interrupted-{uuid4().hex}"
            live.require(folder.resolve().is_relative_to(output.resolve())
                and abandoned.resolve().is_relative_to(output.resolve()),
                "Interrupted evaluation directory escapes its output root")
            folder.rename(abandoned)
        attempt = attempts / uuid4().hex
        attempt.mkdir()
        frames, audit = live.execute_models(live.load_bundle(bundle, day), day, attempt / "models")
        _bundle(plan, day)
        live.require(_plan(root, output) == plan, "Implementation changed during fit")
        hashes = {}
        for zone, frame in frames.items():
            path = attempt / f"{zone}.parquet"
            frame.to_parquet(path)
            hashes[zone] = sha256(path)
        for family, item in audit["price_experts"].items():
            item["model"]["path"] = str(folder / "models" / f"{family}.cbm")
        for zone, item in audit["negative_countries"].items():
            item["model"]["path"] = str(folder / "models" / f"negative_{zone}.cbm")
        _write(attempt / "receipt.json", {"delivery_day": day,
            "plan_sha256": digest, "models": audit, "outputs_sha256": hashes,
            "predictions_sealed_before_scoring": True})
        _prediction_receipt(output, day, digest, folder=attempt)
        live.require(attempt.resolve().is_relative_to(output.resolve())
            and folder.resolve().is_relative_to(output.resolve()),
            "Evaluation publication path escapes its output root")
        folder.parent.mkdir(parents=True, exist_ok=True)
        promote_directory_retry(attempt, folder)


def predict_plan(*, root: Path, output: Path, max_days: int | None = None) -> dict:
    """Fit chronologically; never opens the comparison files or activates NYX."""
    plan = _plan(root, output)
    digest = sha256(output / "plan.json")
    days = plan["delivery_days"]
    if max_days is not None:
        live.require(max_days > 0, "max_days must be positive")
        days = days[:max_days]
    for position, day in enumerate(days, 1):
        bundle = _bundle(plan, day)
        _predict_day(root, output, plan, day, digest, bundle)
        _write(output / "status.json", {"state": "PREDICTING", "day": day,
            "completed": position, "total": len(plan["delivery_days"])})
    result = {"state": "PREDICTIONS_SEALED" if len(days) == len(plan["delivery_days"]) else "PARTIAL",
              "completed": len(days), "total": len(plan["delivery_days"])}
    _write(output / "status.json", result)
    return result


def score_plan(*, root: Path, output: Path) -> dict:
    """Recheck every causal source/model artifact, then open held-out labels."""
    plan = _plan(root, output)
    digest = sha256(output / "plan.json")
    days = plan["delivery_days"]
    receipt_hashes = {}
    for day in days:
        _bundle(plan, day)
        _prediction_receipt(output, day, digest)
        receipt_hashes[day] = sha256(output / "predictions" / day / "receipt.json")
    # No labels or Storm have been read until every prediction has been verified.
    comparison_receipt = plan["comparisons_receipt"]
    comparison_receipt_path = Path(comparison_receipt["path"])
    live.require(sha256(comparison_receipt_path) == comparison_receipt["sha256"],
                 "Official comparison source receipt changed")
    _official_comparisons(comparison_receipt_path.parent,
                          plan["first_delivery_day"], plan["stop_day_exclusive"])
    grid = pd.DatetimeIndex(np.concatenate([delivery_grid(day)[1].to_numpy() for day in days]))
    price_metrics, negative_metrics, comparison_hashes = {}, {}, {}
    for zone in live.COUNTRIES:
        frames = [pd.read_parquet(output / "predictions" / day / f"{zone}.parquet") for day in days]
        for day, frame in zip(days, frames):
            live.require(frame.index.equals(delivery_grid(day)[1]), f"{day}/{zone}: prediction hour grid invalid")
        prediction = pd.concat(frames)
        spec = plan["comparisons"][zone]
        path = Path(spec["path"])
        live.require(path.resolve() == (comparison_receipt_path.parent / f"{zone}.parquet").resolve(),
                     f"{zone}: official comparison file is outside verified source export")
        live.require(sha256(path) == spec["sha256"], f"{zone}: official comparison changed")
        observed = pd.read_parquet(path)
        live.require(observed.index.equals(grid) and {"actual", "storm"} <= set(observed),
                     f"{zone}: official comparison physical grid invalid")
        actual, storm = (observed[name].to_numpy(float) for name in ("actual", "storm"))
        prices = prediction.price_eur_mwh.to_numpy(float)
        probability = prediction.p_negative.to_numpy(float)
        live.require(np.isfinite(actual).all() and np.isfinite(prices).all()
            and np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
            f"{zone}: nonfinite output or observed label")
        paired = np.isfinite(storm)
        live.require(paired.sum() >= len(grid)-1 and paired.any(), f"{zone}: excessive missing Storm observations")
        error, storm_error = np.abs(prices[paired]-actual[paired]), np.abs(storm[paired]-actual[paired])
        rmse, storm_rmse = float(np.sqrt(np.mean(error**2))), float(np.sqrt(np.mean(storm_error**2)))
        wins = int((error < storm_error).sum())
        price_metrics[zone] = {"hours": int(paired.sum()), "storm_common_hours": int(paired.sum()),
            "rmse": rmse, "storm_rmse": storm_rmse, "strict_wins": wins,
            "strict_win_rate": wins/int(paired.sum()),
            "both_criteria_met": bool(rmse < storm_rmse and wins > paired.sum()/2)}
        negative_metrics[zone] = probability_metrics(observed.actual, prediction.p_negative)
        comparison_hashes[zone] = sha256(path)
        live.require(comparison_hashes[zone] == spec["sha256"],
                     f"{zone}: official comparison changed during scoring")
    live.require(_plan(root, output) == plan, "Implementation changed during scoring")
    live.require(sha256(comparison_receipt_path) == comparison_receipt["sha256"],
                 "Official comparison source receipt changed during scoring")
    for day in days:
        _prediction_receipt(output, day, digest)
        live.require(sha256(output / "predictions" / day / "receipt.json") == receipt_hashes[day],
                     f"{day}: prediction receipt changed during scoring")
        live.require(live.bundle_hashes(Path(plan["bundles"][day]["path"])) == plan["bundles"][day]["sha256"],
                     f"{day}: source inputs changed during scoring")
    performance = all(price_metrics[z]["both_criteria_met"] for z in ("FR", "BE", "NL"))
    complete = len(days) == EVALUATION_DAYS
    return {"protocol": PROTOCOL, "qualified": bool(complete and performance),
        "evaluation_kind": "frozen_recipe_full_chain_replay",
        "retrospective_evaluation": True, "independent_validation": False,
        "full_input_chain_qualified": complete, "price_expert_replay_qualified": performance,
        "negative_replay_verified": True, "first_delivery_day": days[0],
        "last_delivery_day": days[-1], "days_evaluated": len(days), "physical_hours": len(grid),
        "price_experts": list(live.PRICE_EXPERTS), "compositions": live.COMPOSITIONS,
        "negative_model_protocol": live.NEGATIVE_PROTOCOL,
        "price_threads": live.PRICE_THREADS, "negative_threads": live.NEGATIVE_THREADS,
        "frequency": "daily_retraining", "de_exception": DE_EXCEPTION,
        "price_country_metrics": price_metrics, "negative_country_metrics": negative_metrics,
        "code_sha256": plan["code_sha256"], "code_hash_policy": CODE_HASH_POLICY,
        "runtime_versions": plan["runtime_versions"],
        **_chronos_pin(plan),
        "target_history_policy": _target_policy(plan),
        "evidence_sha256": {"evaluation_plan": digest, "prediction_receipts": receipt_hashes,
                            "official_comparisons": comparison_hashes,
                            "official_comparisons_receipt": comparison_receipt["sha256"]},
        "predictions_sealed_before_scoring": True,
        "source_snapshots_asof_verified": True,
        "supplier_first_publication_certified": False}


def verify_receipt(receipt: dict, *, root: Path) -> None:
    """Portable activation checks for a previously evaluated full-chain receipt."""
    _target_policy(receipt)
    days = _days(receipt["first_delivery_day"],
        (date.fromisoformat(receipt["last_delivery_day"]) + timedelta(days=1)).isoformat())
    expected_hours = sum(len(delivery_grid(day)[1]) for day in days)
    live.require(receipt.get("protocol") == PROTOCOL and receipt.get("qualified") is True
        and receipt.get("full_input_chain_qualified") is True
        and receipt.get("negative_replay_verified") is True
        and receipt.get("price_expert_replay_qualified") is True
        and receipt.get("days_evaluated") == len(days) == EVALUATION_DAYS
        and receipt.get("physical_hours") == expected_hours
        and receipt.get("compositions") == live.COMPOSITIONS
        and receipt.get("price_experts") == list(live.PRICE_EXPERTS)
        and receipt.get("negative_model_protocol") == live.NEGATIVE_PROTOCOL
        and receipt.get("price_threads") == live.PRICE_THREADS
        and receipt.get("negative_threads") == live.NEGATIVE_THREADS
        and receipt.get("frequency") == "daily_retraining"
        and receipt.get("de_exception") == DE_EXCEPTION
        and receipt.get("predictions_sealed_before_scoring") is True
        and receipt.get("source_snapshots_asof_verified") is True
        and receipt.get("runtime_versions") == _versions()
        and receipt.get("code_sha256") == _code(root)
        and receipt.get("code_hash_policy") == CODE_HASH_POLICY,
        "Full-chain qualification scope, code or CPU runtime differs")
    _chronos_pin(receipt)
    evidence = receipt["evidence_sha256"]
    live.require(set(evidence["prediction_receipts"]) == set(days)
        and set(evidence["official_comparisons"]) == set(live.COUNTRIES)
        and all(live._digest(value) for value in [evidence["evaluation_plan"],
            evidence["official_comparisons_receipt"],
            *evidence["prediction_receipts"].values(), *evidence["official_comparisons"].values()]),
        "Full-chain daily evaluation evidence incomplete")
    live.require(set(receipt["price_country_metrics"]) == set(live.COUNTRIES)
        and set(receipt["negative_country_metrics"]) == set(live.COUNTRIES),
        "Four-country CPU metrics required")
    for zone in live.COUNTRIES:
        p, n = receipt["price_country_metrics"][zone], receipt["negative_country_metrics"][zone]
        live.require(expected_hours-1 <= p["hours"] == p["storm_common_hours"] <= expected_hours
            and np.isfinite([p["rmse"], p["storm_rmse"], p["strict_win_rate"]]).all()
            and 0 <= p["strict_win_rate"] <= 1,
            f"{zone}: invalid annual price metrics")
        if zone != "DE":
            live.require(p["rmse"] < p["storm_rmse"] and p["strict_win_rate"] > .5,
                         f"{zone}: CPU price does not qualify against Storm")
        live.require(n["hours"] == expected_hours and np.isfinite(n["brier"])
            and 0 <= n["brier"] <= 1, f"{zone}: negative-probability metrics invalid")


def qualify(*, root: Path, output: Path, activate: bool = False) -> dict:
    """Recompute score from evidence and optionally install the verified gate."""
    receipt = score_plan(root=root, output=output)
    _write(output / "qualification.json", receipt)
    if activate:
        verify_receipt(receipt, root=root)
        path = root / "config" / live.QUALIFICATION_RECEIPT.name
        # Keep the old conditional CPU score receipt as a distinct audit artifact.
        if path.is_file():
            previous = path.with_name(f"nyx_annual_cpu_qualification_{sha256(path)[:16]}.json")
            if not previous.exists():
                previous.write_bytes(path.read_bytes())
        _write(path, receipt)
        manifest_path = root / "config" / live.MANIFEST.name
        manifest = _read(manifest_path)
        manifest["cpu_annual_qualification"] = {"path": f"config/{path.name}", "sha256": sha256(path)}
        manifest["forecast_enabled"] = True
        _write(manifest_path, manifest)
    return receipt
