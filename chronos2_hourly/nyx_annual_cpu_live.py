"""Prospective consumer of an externally produced annual NYX CPU bundle.

This module never fetches Saturn or reconstructs historical features. A live
bundle, a pinned annual CPU qualification and an enabled manifest are all
required before fitting or writing a forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from chronos2_hourly.nyx_annual_live_preflight import (
    FAMILIES, MATERIALIZATION_PATH, SOURCE_GROUPS, ZONES, delivery_grid,
    inspect_bundle, sha256,
)
from chronos2_hourly.nyx_annual_nyx_quantiles_gate import validate_nyx_quantiles_source
from chronos2_hourly.nyx_negative_probability_cpu import (
    PROTOCOL as NEGATIVE_PROTOCOL, fit_predict_block as fit_negative_block,
    physical_grid,
)
from chronos2_hourly.nyx_pooled_cpu_price_model import (
    PROTOCOL as PRICE_PROTOCOL, PooledConfig, fit_pooled_block,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "nyx_annual_cwe_historical.json"
QUALIFICATION_RECEIPT = ROOT / "config" / "nyx_annual_cpu_qualification_receipt.json"
PROTOCOL = "nyx_annual_cpu_prospective_consumer_v1"
QUALIFICATION_PROTOCOL = "nyx_annual_cpu_price_qualification_v1"
COUNTRIES = ("FR", "DE", "BE", "NL")
PRICE_EXPERTS = ("fr_residual_1000", "cwe_residual_2000", "cwe_absolute_2000")
COMPOSITIONS = {"FR": "fr_residual_disagreement20",
                "DE": "boosting_mean_disagreement20",
                "BE": "boosting_mean_disagreement20",
                "NL": "boosting_mean_all"}
PRICE_THREADS = 8
NEGATIVE_THREADS = 2
QUALIFICATION_CODE = (
    "run_nyx_annual_cpu_live.py",
    "chronos2_hourly/nyx_annual_cpu_live.py",
    "chronos2_hourly/nyx_annual_live_preflight.py",
    "chronos2_hourly/nyx_annual_nyx_quantiles_gate.py",
    "chronos2_hourly/nyx_pooled_cpu_price_model.py",
    "chronos2_hourly/nyx_negative_probability_cpu.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _runtime_versions() -> dict:
    return {"price": {"python": platform.python_version(),
            **{name: version(name) for name in
               ("catboost", "numpy", "pandas", "pyarrow")}},
            "negative": {name: version(name) for name in
                         ("catboost", "numpy", "pandas", "scikit-learn")}}


def verify_activation(root: Path = ROOT, manifest_path: Path | None = None) -> dict:
    """Fail closed on the fixed production manifest and pinned annual score."""
    manifest_file = manifest_path or root / "config" / MANIFEST.name
    manifest = _json(manifest_file)
    require(manifest.get("schema_version") == 1
            and manifest.get("identity") == "nyx_annual_cwe_2026-09-23_historical_inspection_v1",
            "Annual CWE manifest identity differs")
    require(manifest.get("forecast_enabled") is True,
            "Annual CWE manifest has forecast_enabled:false")
    countries = manifest.get("countries")
    require(isinstance(countries, dict) and all(
        isinstance(countries.get(zone), dict)
        and isinstance(countries[zone].get("price"), dict)
        and countries[zone]["price"].get("composition") == COMPOSITIONS[zone]
        for zone in COUNTRIES), "Frozen FR/DE/BE/NL price compositions differ")
    pin = manifest.get("cpu_annual_qualification")
    require(isinstance(pin, dict)
            and pin.get("path") == f"config/{QUALIFICATION_RECEIPT.name}"
            and isinstance(pin.get("sha256"), str)
            and len(pin["sha256"]) == 64,
            "Pinned annual CPU qualification receipt required")
    receipt_path = root / pin["path"]
    require(receipt_path.is_file() and sha256(receipt_path) == pin["sha256"],
            "Annual CPU qualification receipt missing or SHA-256 differs")
    receipt = _json(receipt_path)
    from chronos2_hourly.nyx_annual_cpu_full_chain import (
        PROTOCOL as FULL_CHAIN_PROTOCOL, _chronos_pin, _target_policy, verify_receipt,
    )
    require(receipt.get("protocol") == FULL_CHAIN_PROTOCOL,
            "Production requires a new full-chain CPU evaluation; legacy expert-only scores cannot activate it")
    verify_receipt(receipt, root=root)
    return {"manifest_sha256": sha256(manifest_file),
            "qualification_sha256": pin["sha256"],
            "qualification_receipt": str(receipt_path),
            "compositions": COMPOSITIONS.copy(),
            **_chronos_pin(receipt),
            "target_history_policy": _target_policy(receipt),
            "de_price_performance_exception_used": not (
                receipt["price_country_metrics"]["DE"]["rmse"]
                < receipt["price_country_metrics"]["DE"]["storm_rmse"]
                and receipt["price_country_metrics"]["DE"]["strict_win_rate"] > .5)}


def recoverable_output(output: Path, delivery_day: str) -> bool:
    """Recognize unfinished outputs from the former direct-write consumer."""
    if not output.is_dir() or output.is_symlink() or (output / "receipt.json").exists():
        return False
    try:
        state = _json(output / "status.json")
        return (state.get("protocol") == PROTOCOL
                and state.get("delivery_day") == delivery_day
                and state.get("status") in ("RUNNING", "FAILED"))
    except (OSError, ValueError, TypeError):
        return False


def preflight(bundle: Path, delivery_day: str, output: Path) -> dict:
    """Read only. The CLI exposes no manifest or qualification override."""
    blockers = []
    try:
        activation = verify_activation()
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        activation = None
        blockers.append(f"Activation: {error}")
    try:
        input_report = inspect_bundle(bundle, delivery_day)
        if not input_report["input_bundle_valid"]:
            blockers.append("Live annual input bundle is incomplete or invalid")
    except (OSError, ValueError, KeyError, TypeError) as error:
        input_report = None
        blockers.append(f"Bundle: {error}")
    try:
        baseline_report = validate_nyx_quantiles_source(bundle, delivery_day)
    except (OSError, ValueError, KeyError, TypeError) as error:
        baseline_report = None
        blockers.append(f"NYX CPU baseline: {error}")
    source_report = None
    if input_report and input_report.get("input_bundle_valid") and baseline_report:
        try:
            from chronos2_hourly.nyx_annual_cpu_full_chain import _chronos_pin, _source_packet, _target_policy
            source_report = _source_packet(bundle, delivery_day)
            if activation is not None:
                require(_target_policy(source_report) == _target_policy(activation),
                        "Live target history policy differs from the annually evaluated recipe")
                require(_chronos_pin(source_report["baseline"]) == _chronos_pin(activation),
                        "Live CPU Chronos weights differ from the annually evaluated model")
        except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
            blockers.append(f"Source and CPU producer evidence: {error}")
    if output.exists() and not recoverable_output(output, delivery_day):
        blockers.append("Output already exists; choose a new immutable run directory")
    if output.resolve() == bundle.resolve() or output.resolve().is_relative_to(bundle.resolve()):
        blockers.append("Output must be outside the immutable input bundle")
    attempts = output.parent / f".{output.name}.attempts"
    if attempts.resolve().is_relative_to(bundle.resolve()):
        blockers.append("Interrupted attempts must be outside the immutable input bundle")
    return {"protocol": PROTOCOL, "delivery_day": delivery_day,
            "bundle": str(bundle.resolve()), "output": str(output.resolve()),
            "requires_external_bundle": True, "saturn_fetched": False,
            "activation": activation, "bundle_inspection": input_report,
            "nyx_cpu_baseline_inspection": baseline_report,
            "source_and_producer_inspection": source_report,
            "ready": not blockers, "blockers": blockers}


def _inside(bundle: Path, relative: str) -> Path:
    path = (bundle / relative).resolve()
    require(path.is_relative_to(bundle.resolve()), "Bundle artifact escapes input directory")
    return path


def bundle_hashes(bundle: Path) -> dict[str, str]:
    """Bind every consumed matrix, receipt and receipt-bound source artifact."""
    relatives = [MATERIALIZATION_PATH]
    relatives += [f"features/{family}/{zone}.parquet"
                 for family in FAMILIES for zone in ZONES]
    relatives += [f"baseline/{zone}.parquet" for zone in ZONES]
    relatives += [f"reference/{zone}.parquet" for zone in COUNTRIES]
    for group in SOURCE_GROUPS:
        relative = f"source_receipts/{group}.json"
        receipt = _json(_inside(bundle, relative))
        relatives.append(relative)
        hashes = receipt.get("artifact_sha256")
        require(isinstance(hashes, dict), f"{group}: missing bound source artifacts")
        relatives.extend(hashes)
    paths = {name: _inside(bundle, name) for name in relatives}
    require(all(path.is_file() for path in paths.values()), "Bundle artifact missing")
    return {name: sha256(path) for name, path in sorted(paths.items())}


@dataclass
class BundleData:
    features: dict[str, dict[str, pd.DataFrame]]
    actual: dict[str, pd.Series]
    nyx: dict[str, pd.Series]
    reference: dict[str, pd.Series]
    current: pd.DatetimeIndex


def load_bundle(bundle: Path, delivery_day: str) -> BundleData:
    """Load only the fixed paths accepted by the structural input gate."""
    _, current, _ = delivery_grid(delivery_day)
    features = {family: {zone: pd.read_parquet(_inside(bundle,
        f"features/{family}/{zone}.parquet")) for zone in ZONES} for family in FAMILIES}
    baseline = {zone: pd.read_parquet(_inside(bundle, f"baseline/{zone}.parquet"))
                for zone in ZONES}
    references = {zone: pd.read_parquet(_inside(bundle, f"reference/{zone}.parquet"))
                  for zone in COUNTRIES}
    return BundleData(features=features,
        actual={zone: baseline[zone]["actual"] for zone in ZONES},
        nyx={zone: baseline[zone]["nyx__q50"] for zone in ZONES},
        reference={zone: references[zone]["reference"] for zone in COUNTRIES},
        current=current)


def compose_price(zone: str, points: dict[str, pd.Series],
                  reference: pd.Series) -> pd.Series:
    """Apply the four frozen country choices without learning on labels."""
    require(zone in COUNTRIES and set(points) == set(PRICE_EXPERTS),
            "Exactly three price experts and an FR/DE/BE/NL composition required")
    index = reference.index
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
            and all(point.index.equals(index) for point in points.values()),
            "Expert/reference physical-hour grids differ")
    arrays = {name: value.to_numpy(dtype=np.float64) for name, value in points.items()}
    base = reference.to_numpy(dtype=np.float64)
    require(np.isfinite(base).all() and all(np.isfinite(value).all()
            for value in arrays.values()), "Finite expert and reference points required")
    with np.errstate(over="ignore", invalid="ignore"):
        if zone == "FR":
            proposed = arrays["fr_residual_1000"]
        else:
            proposed = (arrays["cwe_residual_2000"]
                        + arrays["cwe_absolute_2000"]) / 2.
        if zone in ("FR", "DE", "BE"):
            gap = np.abs(proposed - base)
            result = np.where(gap >= 20., proposed, base)
        else:
            result = proposed
    require(np.isfinite(result).all(), "Nonfinite composed forecast")
    return pd.Series(result, index=index, name="price_eur_mwh")


def _save_classifier(estimator: Any, path: Path) -> dict:
    require(estimator is not None and not path.exists(),
            "A fitted classifier and unused CBM path are required")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, filename = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp.cbm",
                                        dir=path.parent)
    os.close(handle)
    temporary = Path(filename)
    try:
        estimator.save_model(str(temporary), format="cbm")
        require(temporary.stat().st_size > 0, "Empty negative-price CBM")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def execute_models(data: BundleData, delivery_day: str, model_dir: Path,
                   price_fit=fit_pooled_block,
                   negative_fit=fit_negative_block, progress=None) -> tuple[dict, dict]:
    """Exactly three pooled price fits and four country classifier fits."""
    day = pd.Timestamp(delivery_day).date()
    stop = str(day + timedelta(days=1))
    expert_configs = {
        "fr_residual_1000": PooledConfig(target_mode="residual", iterations=1000),
        "cwe_residual_2000": PooledConfig(target_mode="residual", iterations=2000),
        "cwe_absolute_2000": PooledConfig(target_mode="absolute", iterations=2000),
    }
    expert_points, expert_audits = {}, {}
    for family in PRICE_EXPERTS:
        points, audit = price_fit(data.features[family], data.actual, data.nyx,
            origin_day=delivery_day, stop_day=stop, config=expert_configs[family],
            model_path=model_dir / f"{family}.cbm", thread_count=PRICE_THREADS)
        require(set(points) == set(ZONES) and audit.get("tree_count") ==
                expert_configs[family].iterations
                and audit.get("training_days") == 365
                and audit.get("training_labels_before_origin") is True
                and audit.get("Storm_used_as_input") is False,
                f"{family}: pooled CPU fit audit invalid")
        model = audit.get("model", {})
        model_path = model_dir / f"{family}.cbm"
        require(model.get("saved") is True and model.get("sha256") == sha256(model_path),
                f"{family}: pooled CPU model not saved or checksum differs")
        for zone in ZONES:
            frame = points[zone]
            require(isinstance(frame, pd.DataFrame) and list(frame) == ["point"]
                    and frame.index.equals(data.current)
                    and np.isfinite(frame.point.to_numpy(float)).all(),
                    f"{family}/{zone}: invalid expert forecast grid")
        expert_points[family], expert_audits[family] = points, audit
        if progress is not None:
            progress("price_expert", expert=family)
    negative_points, negative_audits = {}, {}
    for zone in COUNTRIES:
        history_index = physical_grid(day - timedelta(days=365), day, zone)
        compact = data.features["cwe_residual_2000"][zone]
        block = negative_fit(compact.loc[history_index],
            data.actual[zone].loc[history_index], compact.loc[data.current],
            zone=zone, origin_day=delivery_day, stop_day=stop,
            threads=NEGATIVE_THREADS)
        probabilities, audit = block.probabilities, block.audit
        require(audit.get("models_fitted") == 1 and audit.get("tree_count") == 120
                and audit.get("forecast_labels_used") is False
                and audit.get("storm_used") is False,
                f"{zone}: negative-price classifier did not fit the fixed recipe; "
                f"fallback={audit.get('fallback_reason')}")
        require(probabilities.index.equals(data.current)
                and {"p_negative_raw", "p_negative", "is_negative_predicted"} <=
                set(probabilities)
                and probabilities.p_negative.between(0., 1.).all()
                and probabilities.p_negative_raw.between(0., 1.).all()
                and (probabilities.is_negative_predicted ==
                     (probabilities.p_negative >= .5)).all(),
                f"{zone}: invalid negative-price probabilities")
        audit["model"] = _save_classifier(block.estimator,
            model_dir / f"negative_{zone}.cbm")
        negative_points[zone], negative_audits[zone] = probabilities, audit
        if progress is not None:
            progress("negative_classifier", zone=zone)
    frames = {}
    for zone in COUNTRIES:
        ref = data.reference[zone]
        require(ref.index.equals(data.current), f"{zone}: reference hour grid differs")
        points = {family: expert_points[family][zone].point for family in PRICE_EXPERTS}
        price = compose_price(zone, points, ref)
        negative = negative_points[zone]
        frame = pd.DataFrame({
            "price_eur_mwh": price,
            "p_negative": negative.p_negative,
            "p_negative_raw": negative.p_negative_raw,
            "is_negative_predicted": negative.is_negative_predicted,
            "reference_eur_mwh": ref,
            **{f"{family}_eur_mwh": point for family, point in points.items()},
            "timestamp_local": [value.isoformat() for value in
                                data.current.tz_convert("Europe/Paris")],
        }, index=data.current)
        frame.index.name = "timestamp_utc"
        frames[zone] = frame
    return frames, {"price_experts": expert_audits,
                    "negative_countries": negative_audits,
                    "compositions": COMPOSITIONS.copy(),
                    "price_model_protocol": PRICE_PROTOCOL,
                    "negative_model_protocol": NEGATIVE_PROTOCOL,
                    "price_threads": PRICE_THREADS,
                    "negative_threads": NEGATIVE_THREADS}
