"""The annual NYX q50 input must carry the intended CPU producer lineage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nyx_annual_live_preflight import ZONES, delivery_grid
from chronos2_hourly.nyx_annual_nyx_quantiles_gate import (
    EXPECTED_RECIPE,
    PRODUCER_PROTOCOL,
    validate_curve,
    validate_nyx_quantiles_source,
)


DAY = "2026-10-25"


def _frame(day: str = DAY) -> pd.DataFrame:
    full, current, _ = delivery_grid(day)
    local_days = full.tz_convert("Europe/Paris").date
    origins = {date: pd.Timestamp(
        f"{date - pd.Timedelta(days=1)} 08:00", tz="Europe/Paris")
        for date in set(local_days)}
    frame = pd.DataFrame({
        "nyx__q10": np.full(len(full), 10.),
        "nyx__q50": np.full(len(full), 20.),
        "nyx__q90": np.full(len(full), 30.),
        "actual": np.full(len(full), 25.),
        "forecast_origin_utc": [origins[date].tz_convert("UTC") for date in local_days],
    }, index=full)
    frame.loc[current, "actual"] = np.nan
    return frame


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(root: Path, day: str = DAY) -> None:
    _, _, cutoff = delivery_grid(day)
    (root / "source_receipts").mkdir()
    (root / "baseline").mkdir()
    (root / "baseline_audits").mkdir()
    (root / "baseline_runs").mkdir()
    hashes = {}
    zones = {}
    for zone in ZONES:
        curve = f"baseline/{zone}.parquet"
        audit = f"baseline_audits/{zone}.json"
        upstream = {}
        (root / "baseline_runs" / zone).mkdir()
        for stage in ("chronos", "residual", "kalman"):
            relative = f"baseline_runs/{zone}/{stage}.json"
            (root / relative).write_text(json.dumps({
                "zone": zone, "delivery_day": day, "stage": stage,
                "device": "cpu", "source_asof_cutoff_utc": cutoff.isoformat(),
                "complete": True,
            }), encoding="utf-8")
            hashes[relative] = _sha(root / relative)
            upstream[stage] = {"path": relative, "sha256": hashes[relative]}
        _frame(day).to_parquet(root / curve)
        (root / audit).write_text(json.dumps({
            "protocol": PRODUCER_PROTOCOL, "zone": zone, "delivery_day": day,
            "recipe": EXPECTED_RECIPE, "source_asof_cutoff_utc": cutoff.isoformat(),
            "cpu_retrained": True, "reused_archived_gpu_predictions": False,
            "chronos_model_sha256": "1" * 64, "upstream_receipts": upstream,
        }), encoding="utf-8")
        hashes[curve], hashes[audit] = _sha(root / curve), _sha(root / audit)
        zones[zone] = {"curve": curve, "audit": audit}
    (root / "source_receipts/nyx_quantiles.json").write_text(json.dumps({
        "protocol": "nyx_annual_cpu_live_source_receipt_v1",
        "source_group": "nyx_quantiles", "delivery_day": day,
        "state": "COMPLETE", "asof_cutoff_verified": True,
        "training_window_complete": True,
        "asof_state_utc": cutoff.isoformat(),
        "artifact_sha256": hashes,
        "producer": {"protocol": PRODUCER_PROTOCOL, "recipe": EXPECTED_RECIPE,
                     "zones": zones},
    }), encoding="utf-8")


def test_four_cpu_curves_and_lineage_pass(tmp_path: Path):
    _bundle(tmp_path)
    result = validate_nyx_quantiles_source(tmp_path, DAY)
    assert set(result["zones"]) == set(ZONES)
    assert all(item["hours"] == len(_frame()) for item in result["zones"].values())


def test_old_origin_or_crossed_quantile_rejected():
    frame = _frame()
    validate_curve(frame, delivery_day=DAY, zone="FR")
    full, current, cutoff = delivery_grid(DAY)
    assert len(current) == 25
    frame.loc[current[0], "forecast_origin_utc"] = cutoff - pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="exact D-1 08:00"):
        validate_curve(frame, delivery_day=DAY, zone="FR")
    frame.loc[current[0], "forecast_origin_utc"] = cutoff
    frame.loc[full[0], "nyx__q10"] = 50.
    with pytest.raises(ValueError, match="ordered NYX quantiles"):
        validate_curve(frame, delivery_day=DAY, zone="FR")


def test_generic_receipt_or_gpu_recipe_is_insufficient(tmp_path: Path):
    _bundle(tmp_path)
    path = tmp_path / "source_receipts/nyx_quantiles.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt.pop("producer")
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="CPU producer protocol missing"):
        validate_nyx_quantiles_source(tmp_path, DAY)
    receipt["producer"] = {"protocol": PRODUCER_PROTOCOL,
                           "recipe": {**EXPECTED_RECIPE,
                                      "residual": {**EXPECTED_RECIPE["residual"], "device": "cuda"}},
                           "zones": {zone: {"curve": f"baseline/{zone}.parquet",
                                            "audit": f"baseline_audits/{zone}.json"}
                                     for zone in ZONES}}
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="CPU numerical recipe differs"):
        validate_nyx_quantiles_source(tmp_path, DAY)


def test_archived_gpu_reuse_or_changed_file_rejected(tmp_path: Path):
    _bundle(tmp_path)
    audit_path = tmp_path / "baseline_audits/FR.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["reused_archived_gpu_predictions"] = True
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    receipt_path = tmp_path / "source_receipts/nyx_quantiles.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_sha256"]["baseline_audits/FR.json"] = _sha(audit_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="CPU baseline producer lineage"):
        validate_nyx_quantiles_source(tmp_path, DAY)
    audit["reused_archived_gpu_predictions"] = False
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        validate_nyx_quantiles_source(tmp_path, DAY)
