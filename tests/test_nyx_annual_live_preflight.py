"""Tests for the prospective annual CWE input gate without ignored archives."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nyx_annual_live_preflight import (
    FAMILIES,
    SOURCE_GROUPS,
    ZONES,
    delivery_grid,
    inspect_bundle,
    load_schema,
    validate_baseline,
    validate_feature_frame,
    validate_reference,
    validate_source_receipt,
)


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
    assert len(report["checks"]) == len(SOURCE_GROUPS) + 4 * len(FAMILIES) + 4 + 3
    assert all(item["passed"] is False for item in report["checks"])
    assert any(item["input"] == "source/saturn" for item in report["checks"])
    assert any(item["input"] == "reference/FR" for item in report["checks"])


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
