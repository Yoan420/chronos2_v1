"""A preparation permission must never become an old-origin as-of claim."""
import hashlib

import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_live_preflight as gate


def receipt(tmp_path, *, late=True, group="public_hydro"):
    day = "2026-09-30"
    cutoff = gate.delivery_grid(day)[2]
    raw = tmp_path / "raw.json"
    raw.write_bytes(b"raw historical observations")
    record = {"protocol": gate.SOURCE_PROTOCOL, "source_group": group, "delivery_day": day,
        "state": "COMPLETE", "training_window_complete": True,
        "history_policy": gate.TRAINING_HISTORY_POLICY,
        "asof_cutoff_verified": not late,
        "asof_state_utc": (cutoff - pd.Timedelta(minutes=5)).isoformat(),
        "training_snapshot_max_retrieved_at_utc": (cutoff + pd.Timedelta(hours=1 if late else -1)).isoformat(),
        "delivery_snapshot_pre_cutoff_verified": True,
        "origin_snapshot_capture_verified": False,
        "artifact_sha256": {raw.name: hashlib.sha256(raw.read_bytes()).hexdigest()}}
    return record, dict(group=group, day=day, bundle=tmp_path, cutoff=cutoff)


@pytest.mark.parametrize("group", gate.BOOTSTRAP_GROUPS)
def test_late_history_is_preparable_but_never_qualified(tmp_path, group):
    record, options = receipt(tmp_path, group=group)
    gate.validate_source_receipt(record, **options, allow_training_bootstrap=True)
    with pytest.raises(ValueError, match="history downloaded after"):
        gate.validate_source_receipt(record, **options)
    record["asof_cutoff_verified"] = True
    with pytest.raises(ValueError, match="declaration differs"):
        gate.validate_source_receipt(record, **options, allow_training_bootstrap=True)


def test_historical_capture_does_not_need_its_old_cutoff_for_a_current_fit(tmp_path):
    record, options = receipt(tmp_path, late=False)
    gate.validate_source_receipt(record, **options)
    assert record["origin_snapshot_capture_verified"] is False


def test_bootstrap_never_relaxes_delivery_capture_or_saturn(tmp_path):
    record, options = receipt(tmp_path)
    record["asof_state_utc"] = (options["cutoff"] + pd.Timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="as-of state exceeds"):
        gate.validate_source_receipt(record, **options, allow_training_bootstrap=True)
    record, options = receipt(tmp_path, group="saturn")
    with pytest.raises(ValueError, match="unsupported training history policy"):
        gate.validate_source_receipt(record, **options, allow_training_bootstrap=True)


def test_preparation_keeps_hash_verification(tmp_path):
    record, options = receipt(tmp_path)
    (tmp_path / "raw.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="source artifact missing or changed"):
        gate.validate_source_receipt(record, **options, allow_training_bootstrap=True)


def test_full_chain_refuses_late_history_before_model_evidence(monkeypatch, tmp_path):
    from chronos2_hourly import nyx_annual_source_validation as sources
    from chronos2_hourly import nyx_annual_cpu_full_chain as chain
    from chronos2_hourly import nyx_annual_cpu_baseline as baseline
    monkeypatch.setattr(sources, "validate_source_packet", lambda *a: {
        "passed": True, "source_snapshot_asof_verified": False})
    monkeypatch.setattr(baseline, "validate_cpu_baseline_evidence",
        lambda *a: pytest.fail("Late training snapshot reached qualification of CPU models"))
    with pytest.raises(ValueError, match="raw source evidence required"):
        chain._source_packet(tmp_path, "2026-09-30")
