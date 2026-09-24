"""Finalization must prove source identity and the model's float32 conversion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.rolling_capture as capture
from chronos2_hourly.hourly_contract import local_delivery_day_index


def _sealed_inputs(tmp_path, day="2026-09-10", *, source_value=338.8625):
    index = local_delivery_day_index(day, timezone="Europe/Paris")
    source_values = pd.Series(float(source_value), index=index)
    target_path = (tmp_path / "target.csv").resolve()
    pd.DataFrame({"timestamp": index, "value": source_values.to_numpy()}).to_csv(
        target_path, index=False
    )
    pending = tmp_path / "capture" / "fr" / capture.PENDING_DIRNAME / day
    pending.mkdir(parents=True)
    pd.DataFrame({"delivery_start_utc": index, "chronos_q50": 100.0}).to_parquet(
        pending / capture.CANDIDATE_FILENAME, index=False
    )
    manifest = {
        "schema_version": capture.SCHEMA_VERSION,
        "status": "target_pending",
        "source_kind": "issued_live",
        "zone": "FR",
        "timezone": "Europe/Paris",
        "delivery_start_day_local": day,
        "delivery_end_day_local": day,
        "target_available": False,
        "target_series": "price.fr",
    }
    capture._write_json(pending / capture.MANIFEST_FILENAME, manifest)
    capture._write_checksum_manifest(pending, {
        "rolling_refit_target_pending_rows": pending / capture.CANDIDATE_FILENAME,
        "rolling_refit_block_manifest": pending / capture.MANIFEST_FILENAME,
    })
    archive = tmp_path / "issued"
    archive.mkdir()
    (archive / "forecast.csv").write_text("q50\n100\n", encoding="utf-8")
    issued = {
        "zone": "FR",
        "delivery_day_local": (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat(),
        "run_type": "live_day_ahead",
        "forecast_status": "issued_live",
        "target_series": "price.fr",
        "target_source_path": str(target_path),
        "target_source_sha256": capture._sha256(target_path),
        "input_diagnostics": {"target": {"input": str(target_path)}},
        "issued_at_utc": "2026-09-10T06:00:00Z",
    }
    capture._write_json(archive / "run_manifest.json", issued)
    capture._write_json(archive / capture.CHECKSUM_FILENAME, {
        "algorithm": "sha256",
        "artifacts": [
            {"path": name, "sha256": capture._sha256(archive / name), "role": "run_artifact"}
            for name in ["forecast.csv", "run_manifest.json"]
        ],
    })
    return {
        "capture_root": tmp_path / "capture",
        "zone": "FR",
        "delivery_day": day,
        "delivery_timezone": "Europe/Paris",
        "canonical_target": source_values.astype(np.float32),
        "target_series": "price.fr",
        "target_source_path": target_path,
        "target_observation_archive": archive,
        "target_observation_forecast_filename": "forecast.csv",
    }, pending, source_values


@pytest.mark.parametrize("day,hours", [("2026-09-10", 24), ("2026-03-29", 23), ("2026-10-25", 25)])
def test_float32_roundtrip_finalizes_full_physical_day_without_changing_inputs(tmp_path, day, hours):
    kwargs, pending, source_values = _sealed_inputs(tmp_path, day)
    before = {path: path.read_bytes() for path in pending.iterdir()}
    before[kwargs["target_source_path"]] = kwargs["target_source_path"].read_bytes()
    canonical_before = kwargs["canonical_target"].copy()
    assert (source_values - canonical_before).abs().max() > 5e-6

    final, audit = capture.finalize_target_pending_candidate(**kwargs)

    assert audit["status"] == "complete"
    assert audit["hours"] == hours
    precision = audit["target_observation_precision"]
    assert precision["mode"] == "float32_roundtrip"
    assert precision["compared_hours"] == hours
    assert precision["float32_representations_equal"] is True
    assert precision["inputs_modified"] is False
    manifest = json.loads((final / capture.MANIFEST_FILENAME).read_text())
    assert manifest["target_observation_precision"] == precision
    rows = pd.read_csv(final / capture.BLOCK_FILENAME)
    np.testing.assert_array_equal(rows.actual, canonical_before.to_numpy(dtype=float))
    pd.testing.assert_series_equal(kwargs["canonical_target"], canonical_before)
    assert all(path.read_bytes() == payload for path, payload in before.items())
    capture._verify_checksum_manifest(final)


def test_exact_observations_are_audited_and_second_finalization_is_immutable(tmp_path):
    kwargs, _, source_values = _sealed_inputs(tmp_path, source_value=100.0)
    kwargs["canonical_target"] = source_values
    final, audit = capture.finalize_target_pending_candidate(**kwargs)
    assert audit["target_observation_precision"]["mode"] == "exact"
    before = {path: path.read_bytes() for path in final.iterdir()}
    repeated, audit = capture.finalize_target_pending_candidate(**kwargs)
    assert repeated == final
    assert audit["status"] == "already_finalized"
    assert all(path.read_bytes() == payload for path, payload in before.items())


@pytest.mark.parametrize("source_value,changed_value", [
    (100.0, 100.01),  # Actual revision.
    (10.0, 10.0 + 1e-6),  # Smaller than the old tolerance, but a different float32.
    (10000.0, 10000.0 + 1e-4),  # Same float32 but outside the explicit ceiling.
    (100.0, np.nan),
    (100.0, np.inf),
])
def test_invalid_target_is_rejected_without_sealing(tmp_path, source_value, changed_value):
    kwargs, pending, _ = _sealed_inputs(tmp_path, source_value=source_value)
    kwargs["canonical_target"] = kwargs["canonical_target"].astype(float)
    kwargs["canonical_target"].iloc[0] = changed_value
    with pytest.raises(capture.RollingCaptureError, match="checksummed target source|non-finite"):
        capture.finalize_target_pending_candidate(**kwargs)
    assert pending.is_dir()
    assert not (pending.parent.parent / capture.FINAL_DIRNAME).exists()


@pytest.mark.parametrize("corrupted", ["source", "candidate", "archive"])
def test_checksum_guards_still_reject_tampering(tmp_path, corrupted):
    kwargs, pending, _ = _sealed_inputs(tmp_path)
    target = {
        "source": kwargs["target_source_path"],
        "candidate": pending / capture.CANDIDATE_FILENAME,
        "archive": kwargs["target_observation_archive"] / "forecast.csv",
    }[corrupted]
    with target.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(capture.RollingCaptureError, match="checksum"):
        capture.finalize_target_pending_candidate(**kwargs)
    assert not (pending.parent.parent / capture.FINAL_DIRNAME).exists()


def test_missing_physical_hour_is_not_filled(tmp_path):
    kwargs, pending, _ = _sealed_inputs(tmp_path)
    kwargs["canonical_target"] = kwargs["canonical_target"].iloc[:-1]
    with pytest.raises(capture.RollingCaptureError, match="unavailable or non-finite"):
        capture.finalize_target_pending_candidate(**kwargs)
    assert not (pending.parent.parent / capture.FINAL_DIRNAME).exists()
