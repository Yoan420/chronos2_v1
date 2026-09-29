"""Historical profile recovery is bounded and requires its own CPU qualification."""
from __future__ import annotations

import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_live_preflight as gate
from chronos2_hourly import nyx_annual_cpu_full_chain as chain
from chronos2_hourly import nyx_annual_cpu_live as live
from test_nyx_annual_cpu_full_chain import _small_plan, chronos_pin


def contract(day):
    return {"profile_history_policy": gate.PROFILE_HISTORY_POLICY,
            "profile_revision_ceiling_utc": gate.delivery_grid(day)[2].isoformat(),
            "profile_origin_snapshot_verified": False}


def source_receipt(tmp_path, day):
    artifact = tmp_path / "profiles.bin"
    artifact.write_bytes(b"bound canonical profile history")
    return {"protocol": gate.SOURCE_PROTOCOL, "source_group": "saturn",
            "delivery_day": day, "state": "COMPLETE", "asof_cutoff_verified": True,
            "training_window_complete": True,
            "asof_state_utc": gate.delivery_grid(day)[2].isoformat(),
            "artifact_sha256": {"profiles.bin": gate.sha256(artifact)}, **contract(day)}


@pytest.mark.parametrize("change,reason", [
    ({"profile_revision_ceiling_utc": "2026-09-29T06:00:01+00:00"}, "outer forecast cutoff"),
    ({"profile_revision_ceiling_utc": "2024-08-15T06:00:00+00:00"}, "outer forecast cutoff"),
    ({"profile_revision_ceiling_utc": "2026-09-29T06:00:00"}, "outer forecast cutoff"),
    ({"profile_origin_snapshot_verified": True}, "cannot certify old revisions"),
    ({"profile_history_policy": "latest"}, "Unsupported profile history policy"),
    ({"asof_state_utc": "2026-09-28T06:00:00+00:00"}, "source state must equal"),
])
def test_recovery_does_not_relax_outer_source_cutoff(tmp_path, change, reason):
    day = "2026-09-30"
    receipt = source_receipt(tmp_path, day)
    gate.validate_source_receipt(receipt, group="saturn", day=day, bundle=tmp_path,
                                 cutoff=gate.delivery_grid(day)[2])
    receipt.update(change)
    for preparation in (False, True):
        with pytest.raises(ValueError, match=reason):
            gate.validate_source_receipt(receipt, group="saturn", day=day, bundle=tmp_path,
                cutoff=gate.delivery_grid(day)[2], allow_training_bootstrap=preparation)


@pytest.mark.parametrize("missing", list(gate.PROFILE_HISTORY_FIELDS))
def test_incomplete_recovery_declarations_are_rejected(tmp_path, missing):
    day = "2026-09-30"
    receipt = source_receipt(tmp_path, day)
    receipt.pop(missing)
    with pytest.raises(ValueError):
        gate.validate_source_receipt(receipt, group="saturn", day=day, bundle=tmp_path,
                                     cutoff=gate.delivery_grid(day)[2])


def test_legacy_receipt_retains_its_strict_contract(tmp_path):
    day = "2026-09-30"
    receipt = source_receipt(tmp_path, day)
    for key in gate.PROFILE_HISTORY_FIELDS:
        receipt.pop(key)
    gate.validate_source_receipt(receipt, group="saturn", day=day, bundle=tmp_path,
                                 cutoff=gate.delivery_grid(day)[2])


def test_source_and_both_producers_must_agree_on_recovery(tmp_path, monkeypatch):
    from chronos2_hourly import nyx_annual_source_validation as raw
    from chronos2_hourly import nyx_annual_saturn_source as saturn
    from chronos2_hourly import nyx_annual_cpu_baseline as baseline
    from chronos2_hourly import nyx_annual_cpu_reference_builder as reference
    day = "2026-09-30"
    current = contract(day)
    monkeypatch.setattr(raw, "validate_source_packet", lambda *_: {
        "passed": True, "source_snapshot_asof_verified": True})
    monkeypatch.setattr(saturn, "target_history_contract", lambda *_: {})
    monkeypatch.setattr(saturn, "profile_history_contract", lambda *_: current, raising=False)
    monkeypatch.setattr(baseline, "validate_cpu_baseline_evidence", lambda *_: {**chronos_pin(), **current})
    monkeypatch.setattr(reference, "validate_cpu_reference_source", lambda *_: current)
    packet = chain._source_packet(tmp_path, day)
    assert {key: packet[key] for key in gate.PROFILE_HISTORY_FIELDS} == current
    # A missing producer declaration cannot silently downgrade recovered inputs.
    monkeypatch.setattr(reference, "validate_cpu_reference_source", lambda *_: {})
    with pytest.raises(ValueError, match="reference profile history contract differs"):
        chain._source_packet(tmp_path, day)
    monkeypatch.setattr(reference, "validate_cpu_reference_source", lambda *_: current)
    monkeypatch.setattr(baseline, "validate_cpu_baseline_evidence", lambda *_: chronos_pin())
    with pytest.raises(ValueError, match="baseline profile history contract differs"):
        chain._source_packet(tmp_path, day)
    monkeypatch.setattr(baseline, "validate_cpu_baseline_evidence", lambda *_: {**chronos_pin(), **current})
    # Nor can removing the source policy make recovered producer evidence strict.
    monkeypatch.setattr(saturn, "profile_history_contract", lambda *_: {})
    with pytest.raises(ValueError, match="baseline profile history contract differs"):
        chain._source_packet(tmp_path, day)
    monkeypatch.setattr(saturn, "profile_history_contract", lambda *_: current)
    with pytest.raises(ValueError, match="outer forecast cutoff"):
        chain._source_packet(tmp_path, "2026-09-29")


def test_evaluation_pins_recovery_policy_across_distinct_outer_cutoffs(tmp_path, monkeypatch):
    _, initial, observations = _small_plan(tmp_path, monkeypatch)
    first = initial["first_delivery_day"]
    stop = (pd.Timestamp(initial["stop_day_exclusive"]) + pd.Timedelta(days=1)).date().isoformat()
    monkeypatch.setattr(chain, "_source_packet", lambda _, day: {
        "baseline": chronos_pin(), **contract(day)})
    plan = chain.prepare_plan(root=tmp_path, bundles=tmp_path / "bundles", comparisons=observations,
        output=tmp_path / "recovery_evaluation", first=first, stop=stop)
    assert plan["profile_history_policy"] == gate.PROFILE_HISTORY_POLICY
    assert "profile_revision_ceiling_utc" not in plan
    for day in plan["delivery_days"]:
        assert chain._bundle(plan, day) == tmp_path / "bundles" / day
    monkeypatch.setattr(chain, "_source_packet", lambda *_: {"baseline": chronos_pin()})
    with pytest.raises(ValueError, match="profile history policy differs"):
        chain._bundle(plan, first)
    monkeypatch.setattr(chain, "_source_packet", lambda _, day: {
        "baseline": chronos_pin(), **(contract(day) if day == first else {})})
    with pytest.raises(ValueError, match="profile history policies differ across evaluation days"):
        chain.prepare_plan(root=tmp_path, bundles=tmp_path / "bundles", comparisons=observations,
            output=tmp_path / "mixed", first=first, stop=stop)
    assert not (tmp_path / "mixed").exists()


def test_live_refuses_unevaluated_recovery_or_silent_policy_removal(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "verify_activation", lambda: chronos_pin())
    monkeypatch.setattr(live, "inspect_bundle", lambda *_: {"input_bundle_valid": True})
    monkeypatch.setattr(live, "validate_nyx_quantiles_source", lambda *_: {"passed": True})
    monkeypatch.setattr(chain, "_source_packet", lambda _, day: {
        "baseline": chronos_pin(), **contract(day)})
    report = live.preflight(tmp_path / "bundle", "2026-09-30", tmp_path / "output")
    assert not report["ready"]
    assert any("profile history policy differs" in error for error in report["blockers"])
    monkeypatch.setattr(live, "verify_activation", lambda: {
        **chronos_pin(), "profile_history_policy": gate.PROFILE_HISTORY_POLICY})
    assert live.preflight(tmp_path / "bundle", "2026-09-30", tmp_path / "output")["ready"]
    monkeypatch.setattr(chain, "_source_packet", lambda *_: {"baseline": chronos_pin()})
    assert not live.preflight(tmp_path / "bundle", "2026-09-30", tmp_path / "output")["ready"]
