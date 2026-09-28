"""Tests for the prospective annual CWE input gate without ignored archives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nyx_annual_live_preflight import (
    FAMILIES,
    MATERIALIZATION_PATH,
    MATERIALIZATION_PROTOCOL,
    MATERIALIZER_CODE,
    SOURCE_GROUPS,
    SOURCE_PROTOCOL,
    ZONES,
    delivery_grid,
    inspect_bundle,
    load_schema,
    materialized_outputs,
    validate_baseline,
    validate_feature_frame,
    validate_materialization_manifest,
    validate_reference,
    validate_source_receipt,
)
from chronos2_hourly.nyx_annual_cpu_live import bundle_hashes


def test_ordered_schemas_are_independent_of_historical_cache():
    schema = load_schema()
    assert set(schema["families"]) == set(FAMILIES)
    for family, width in FAMILIES.items():
        for zone in ZONES:
            assert len(schema["families"][family]["columns"][zone]) == width
    assert (schema["families"]["cwe_residual_2000"]["columns"]["FR"]
            != schema["families"]["cwe_absolute_2000"]["columns"]["FR"])


def test_clean_clone_has_explicit_missing_inputs(tmp_path: Path):
    report = inspect_bundle(tmp_path, "2026-09-29")
    assert report["input_bundle_valid"] is False
    assert len(report["checks"]) == len(SOURCE_GROUPS) + 4 * len(FAMILIES) + 4 + 3 + 1
    assert all(item["passed"] is False for item in report["checks"])
    assert any(item["input"] == "source/saturn" for item in report["checks"])
    assert any(item["input"] == "reference/FR" for item in report["checks"])
    assert any(item["input"] == "materialization/feature_bundle" for item in report["checks"])


def test_physical_dst_grid_and_strict_order_availability():
    full, current, _ = delivery_grid("2026-10-25")
    assert len(current) == 25
    frame = pd.DataFrame({"value": np.ones(len(full)),
                          "value__available": np.ones(len(full))}, index=full)
    validate_feature_frame(frame, ["value", "value__available"], full, "test")
    with pytest.raises(ValueError, match="ordered annual feature"):
        validate_feature_frame(frame, ["value__available", "value"], full, "test")
    frame.iloc[0, 1] = 0
    with pytest.raises(ValueError, match="availability flag/value mismatch"):
        validate_feature_frame(frame, ["value", "value__available"], full, "test")


def test_jao_shared_flag_matches_all_retained_descriptors():
    full, _, _ = delivery_grid("2026-09-29")
    columns = ["extra_jao_ram_p10_mw", "extra_jao_core_ptdf_range_p90",
               "extra_jao__available"]
    frame = pd.DataFrame({columns[0]: np.ones(len(full)),
                          columns[1]: np.ones(len(full)),
                          columns[2]: np.ones(len(full))}, index=full)
    validate_feature_frame(frame, columns, full, "JAO")
    frame.loc[full[0], columns[0]] = np.nan
    frame.loc[full[0], columns[2]] = 0
    validate_feature_frame(frame, columns, full, "JAO")
    frame.loc[full[0], columns[2]] = 1
    with pytest.raises(ValueError, match="availability flag/value mismatch"):
        validate_feature_frame(frame, columns, full, "JAO")


def test_future_observation_and_late_forecast_are_rejected():
    full, current, cutoff = delivery_grid("2026-09-29")
    frame = pd.DataFrame({"actual": np.zeros(len(full)),
                          "nyx__q50": np.zeros(len(full)),
                          "forecast_origin_utc": [cutoff] * len(full)}, index=full)
    frame.loc[current, "actual"] = np.nan
    validate_baseline(frame, full, current, cutoff, "baseline")
    frame.loc[current[0], "actual"] = 0
    with pytest.raises(ValueError, match="future labels absent"):
        validate_baseline(frame, full, current, cutoff, "baseline")
    frame.loc[current[0], "actual"] = np.nan
    frame.loc[current[0], "forecast_origin_utc"] = cutoff + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="after D-1 08:00"):
        validate_baseline(frame, full, current, cutoff, "baseline")
    reference = pd.DataFrame({"reference": np.zeros(len(current)),
                              "forecast_origin_utc": [cutoff] * len(current)},
                             index=current)
    validate_reference(reference, current, cutoff, "reference")
    reference.loc[current[0], "forecast_origin_utc"] = cutoff + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="after D-1 08:00"):
        validate_reference(reference, current, cutoff, "reference")


def test_source_receipt_is_date_bound_and_hash_bound(tmp_path: Path):
    day = "2026-09-29"
    _, _, cutoff = delivery_grid(day)
    artifact = tmp_path / "source.bin"
    artifact.write_bytes(b"future-source-snapshot")
    receipt = {"protocol": "nyx_annual_cpu_live_source_receipt_v1",
               "source_group": "saturn", "delivery_day": day, "state": "COMPLETE",
               "asof_cutoff_verified": True, "training_window_complete": True,
               "asof_state_utc": cutoff.isoformat(),
               "artifact_sha256": {"source.bin": hashlib.sha256(artifact.read_bytes()).hexdigest()}}
    validate_source_receipt(receipt, group="saturn", day=day, bundle=tmp_path, cutoff=cutoff)
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        validate_source_receipt(receipt, group="saturn", day=day, bundle=tmp_path, cutoff=cutoff)
    receipt["artifact_sha256"] = {"../outside": "0" * 64}
    with pytest.raises(ValueError, match="escapes live bundle"):
        validate_source_receipt(receipt, group="saturn", day=day, bundle=tmp_path, cutoff=cutoff)


def _bound_materialization(tmp_path: Path):
    """Small hash graph; frame schemas are validated separately."""
    day = "2026-09-29"
    _, _, cutoff = delivery_grid(day)
    bundle = tmp_path / "bundle"
    code_root = tmp_path / "checkout"
    code_path = code_root / MATERIALIZER_CODE[0]
    code_path.parent.mkdir(parents=True)
    code_path.write_text("# deterministic materializer\n", encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text('{"schema": "test"}\n', encoding="utf-8")
    receipts, sources = {}, {}
    for group in SOURCE_GROUPS:
        relative = f"source_snapshots/{group}.bin"
        source = bundle / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(group.encode("ascii"))
        sources[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        receipt = {"protocol": SOURCE_PROTOCOL, "source_group": group,
                   "delivery_day": day, "state": "COMPLETE",
                   "asof_cutoff_verified": True, "training_window_complete": True,
                   "asof_state_utc": cutoff.isoformat(),
                   "artifact_sha256": {relative: sources[relative]}}
        receipt_path = bundle / "source_receipts" / f"{group}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        receipts[group] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    outputs = {}
    for relative in materialized_outputs():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("ascii"))
        outputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for zone in ZONES:
        path = bundle / "baseline" / f"{zone}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(zone.encode("ascii"))
    manifest = {"protocol": MATERIALIZATION_PROTOCOL, "delivery_day": day,
                "state": "COMPLETE", "asof_cutoff_utc": cutoff.isoformat(),
                "deterministic_transform": True, "future_labels_used": False,
                "storm_used_as_model_input": False, "parameters": {},
                "schema_sha256": hashlib.sha256(schema.read_bytes()).hexdigest(),
                "source_receipts_sha256": receipts,
                "source_artifacts_sha256": sources,
                "transform_code_sha256": {MATERIALIZER_CODE[0]:
                    hashlib.sha256(code_path.read_bytes()).hexdigest()},
                "output_sha256": outputs}
    manifest_path = bundle / MATERIALIZATION_PATH
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return bundle, code_root, schema, manifest, code_path


def test_materialization_binds_all_outputs_sources_and_code(tmp_path: Path):
    bundle, code_root, schema, manifest, code_path = _bound_materialization(tmp_path)
    result = validate_materialization_manifest(bundle, "2026-09-29",
        schema_path=schema, code_root=code_root)
    assert set(result["output_sha256"]) == set(materialized_outputs())
    assert result["manifest_sha256"] == bundle_hashes(bundle)[MATERIALIZATION_PATH]

    output = bundle / materialized_outputs()[0]
    output.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Materialized input missing or changed"):
        validate_materialization_manifest(bundle, "2026-09-29",
            schema_path=schema, code_root=code_root)
    output.write_bytes(materialized_outputs()[0].encode("ascii"))
    code_path.write_text("# replaced code\n", encoding="utf-8")
    with pytest.raises(ValueError, match="transformation code missing or changed"):
        validate_materialization_manifest(bundle, "2026-09-29",
            schema_path=schema, code_root=code_root)


def test_materialization_rejects_unrelated_source_hashes(tmp_path: Path):
    bundle, code_root, schema, manifest, _ = _bound_materialization(tmp_path)
    manifest["source_artifacts_sha256"] = {"unrelated.bin": "0" * 64}
    (bundle / MATERIALIZATION_PATH).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source digest graph differs"):
        validate_materialization_manifest(bundle, "2026-09-29",
            schema_path=schema, code_root=code_root)


def test_materialization_rejects_unapproved_code_even_when_hash_matches(tmp_path: Path):
    bundle, code_root, schema, manifest, _ = _bound_materialization(tmp_path)
    unrelated = code_root / "chronos2_hourly/unrelated.py"
    unrelated.write_text("# unrelated\n", encoding="utf-8")
    manifest["transform_code_sha256"] = {"chronos2_hourly/unrelated.py":
        hashlib.sha256(unrelated.read_bytes()).hexdigest()}
    (bundle / MATERIALIZATION_PATH).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="approved transformation code"):
        validate_materialization_manifest(bundle, "2026-09-29",
            schema_path=schema, code_root=code_root)
