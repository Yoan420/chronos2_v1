"""Fail-closed contract for the four prospective annual NYX baseline curves.

The archived annual q50 curves were produced by a local GPU research replay.
This validator accepts only a new, explicitly CPU-produced four-country bundle;
it does not infer producer identity from a Parquet filename or a q50 column.
It validates declarations and checksums, not the truth of an external provider's
publication timestamp or the statistical quality of the CPU baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .chronos_adapter import build_delivery_plan
from .nyx_annual_live_preflight import (
    SOURCE_PROTOCOL,
    ZONES,
    delivery_grid,
    sha256,
    validate_source_receipt,
)


PRODUCER_PROTOCOL = "nyx_annual_cpu_nyx_quantiles_v1"
CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
EXPECTED_RECIPE = {
    "chronos": {
        "engine": "nyx_local_chronos_v1",
        "model_id": "amazon/chronos-2",
        "revision": CHRONOS_REVISION,
        "device": "cpu",
        "dtype": "torch.float32",
        "context_length": 2048,
        "target_availability": "day_ahead",
        "forecast_origin_local_time": "08:00",
        "cross_learning": False,
    },
    "residual": {
        "recipe": "interaction40",
        "backend": "catboost",
        "device": "cpu",
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 700,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 15.0,
        "min_training_rows": 720,
        "random_seed": 42,
        "has_time": True,
        "nan_mode": "Min",
        "max_abs_correction": 40.0,
    },
    "kalman": {"training_lookback_days": 365},
}
QUANTILES = ("nyx__q10", "nyx__q50", "nyx__q90")
REQUIRED_COLUMNS = {"actual", "forecast_origin_utc", *QUANTILES}
RUN_STAGES = ("chronos", "residual", "kalman")


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _inside(bundle: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and relative
             and not Path(relative).is_absolute(), "Unsafe baseline artifact path")
    path = (bundle / relative).resolve()
    _require(path.is_relative_to(bundle.resolve()), "Baseline artifact escapes bundle")
    return path


def _expected_origins(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.Index(index.tz_convert("Europe/Paris").date).unique()
    origins = {day: build_delivery_plan(day, timezone="Europe/Paris").forecast_origin_utc
               for day in dates}
    return pd.DatetimeIndex([origins[day] for day in index.tz_convert("Europe/Paris").date])


def validate_curve(frame: pd.DataFrame, *, delivery_day: str, zone: str) -> None:
    """Check complete local-day support, quantiles, labels and exact D-1 origins."""
    full, current, _ = delivery_grid(delivery_day)
    _require(zone in ZONES, "Unknown NYX baseline zone")
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique
             and set(frame.columns) == REQUIRED_COLUMNS,
             f"{zone}: exact NYX quantile/actual/origin columns required")
    index = frame.index
    _require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
             and index.equals(full), f"{zone}: exact 365-day plus delivery UTC grid required")
    values = frame.loc[:, list(QUANTILES)].to_numpy(dtype=float)
    _require(np.isfinite(values).all() and (np.diff(values, axis=1) >= 0).all(),
             f"{zone}: finite ordered NYX quantiles required")
    observed = frame["actual"].to_numpy(dtype=float)
    future = index.isin(current)
    _require(np.isfinite(observed[~future]).all() and np.isnan(observed[future]).all(),
             f"{zone}: observed past and unlabelled future required")
    origins = pd.DatetimeIndex(pd.to_datetime(frame["forecast_origin_utc"], utc=True))
    _require(not origins.hasnans and origins.equals(_expected_origins(index)),
             f"{zone}: each local day requires its exact D-1 08:00 origin")


def validate_nyx_quantiles_source(bundle: str | Path, delivery_day: str) -> dict:
    """Validate the prospective NYX producer receipt and four bound curves.

    A generic `nyx_quantiles` source receipt or preflight-clean Parquet is
    insufficient: this additional contract records the CPU numerical recipe,
    full quantile curves, exact origins, and source-specific producer audits.
    """
    root = Path(bundle).resolve()
    _, _, cutoff = delivery_grid(delivery_day)
    receipt_path = _inside(root, "source_receipts/nyx_quantiles.json")
    _require(receipt_path.is_file(), "NYX quantiles source receipt missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_source_receipt(receipt, group="nyx_quantiles", day=delivery_day,
                            bundle=root, cutoff=cutoff)
    producer = receipt.get("producer")
    _require(isinstance(producer, dict) and producer.get("protocol") == PRODUCER_PROTOCOL,
             "NYX quantiles CPU producer protocol missing")
    _require(producer.get("recipe") == EXPECTED_RECIPE,
             "NYX quantiles CPU numerical recipe differs")
    zone_audits = producer.get("zones")
    _require(isinstance(zone_audits, dict) and set(zone_audits) == set(ZONES),
             "Four NYX producer zone audits required")
    output = {}
    for zone in ZONES:
        item = zone_audits[zone]
        _require(isinstance(item, dict), f"{zone}: producer audit required")
        curve_path = f"baseline/{zone}.parquet"
        audit_path = f"baseline_audits/{zone}.json"
        _require(item.get("curve") == curve_path and item.get("audit") == audit_path,
                 f"{zone}: canonical curve and audit paths required")
        for relative in (curve_path, audit_path):
            _require(relative in receipt["artifact_sha256"],
                     f"{zone}: unbound baseline artifact {relative}")
            path = _inside(root, relative)
            _require(path.is_file() and sha256(path) == receipt["artifact_sha256"][relative],
                     f"{zone}: baseline artifact missing or changed: {relative}")
        audit = json.loads(_inside(root, audit_path).read_text(encoding="utf-8"))
        upstream = audit.get("upstream_receipts")
        _require(audit.get("protocol") == PRODUCER_PROTOCOL
                 and audit.get("zone") == zone
                 and audit.get("delivery_day") == delivery_day
                 and audit.get("recipe") == EXPECTED_RECIPE
                 and audit.get("source_asof_cutoff_utc") == cutoff.isoformat()
                 and audit.get("cpu_retrained") is True
                 and audit.get("reused_archived_gpu_predictions") is False
                 and isinstance(audit.get("chronos_model_sha256"), str)
                 and len(audit["chronos_model_sha256"]) == 64
                 and all(char in "0123456789abcdef" for char in audit["chronos_model_sha256"])
                 and isinstance(upstream, dict) and set(upstream) == set(RUN_STAGES),
                 f"{zone}: complete CPU baseline producer lineage required")
        for stage in RUN_STAGES:
            run = upstream[stage]
            relative = f"baseline_runs/{zone}/{stage}.json"
            _require(isinstance(run, dict) and run.get("path") == relative
                     and run.get("sha256") == receipt["artifact_sha256"].get(relative),
                     f"{zone}: {stage} run receipt not bound to source receipt")
            run_path = _inside(root, relative)
            _require(run_path.is_file() and sha256(run_path) == run["sha256"],
                     f"{zone}: {stage} run receipt missing or changed")
            run_audit = json.loads(run_path.read_text(encoding="utf-8"))
            _require(run_audit.get("zone") == zone
                     and run_audit.get("delivery_day") == delivery_day
                     and run_audit.get("stage") == stage
                     and run_audit.get("device") == "cpu"
                     and run_audit.get("source_asof_cutoff_utc") == cutoff.isoformat()
                     and run_audit.get("complete") is True,
                     f"{zone}: {stage} CPU run audit incomplete")
        frame = pd.read_parquet(_inside(root, curve_path))
        validate_curve(frame, delivery_day=delivery_day, zone=zone)
        output[zone] = {"curve_sha256": receipt["artifact_sha256"][curve_path],
                        "producer_audit_sha256": receipt["artifact_sha256"][audit_path],
                        "hours": len(frame)}
    return {"protocol": PRODUCER_PROTOCOL, "delivery_day": delivery_day,
            "receipt_sha256": sha256(receipt_path), "zones": output,
            "scope": "Producer declarations, artifact integrity and chronology; no independent score or provider publication attestation"}


__all__ = ["EXPECTED_RECIPE", "PRODUCER_PROTOCOL", "validate_curve",
           "validate_nyx_quantiles_source"]
