from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from chronos2_hourly.exogenous_research_registry import (
    EXPECTED_FAMILIES,
    ExogenousRegistryError,
    audit_exogenous_research_registry,
    load_exogenous_research_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "exogenous_research_registry.example.yaml"


def _payload() -> dict:
    payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _first_signal(payload: dict) -> dict:
    return payload["families"][0]["signals"][0]


def test_checked_in_registry_covers_roadmap_and_fails_closed() -> None:
    payload, audit = load_exogenous_research_registry(REGISTRY_PATH)

    assert payload["policy"] == {
        "scope": "offline_research_registry_only",
        "fail_closed": True,
        "training_allowed": False,
        "production_writes_allowed": False,
        "storm_mkonline_as_inputs": False,
        "latest_backfill_allowed": False,
        "interpolation_allowed": False,
        "forward_backward_fill_allowed": False,
        "raw_media_as_inputs": False,
        "direct_llm_price_output": False,
    }
    assert {family["id"] for family in payload["families"]} == EXPECTED_FAMILIES
    assert len(audit.signals) == 11
    assert audit.ready_signal_ids == ()
    assert set(audit.blocked_signal_ids) == {
        signal["id"]
        for family in payload["families"]
        for signal in family["signals"]
    }
    assert all(item.blockers for item in audit.signals)
    assert len(audit.registry_sha256) == 64


def test_registry_hash_and_signal_hashes_are_deterministic() -> None:
    first = audit_exogenous_research_registry(_payload())
    second = audit_exogenous_research_registry(_payload())

    assert first.registry_sha256 == second.registry_sha256
    assert [item.declaration_sha256 for item in first.signals] == [
        item.declaration_sha256 for item in second.signals
    ]


def test_late_hour_coverage_repair_is_explicit_and_blocked() -> None:
    payload, audit = load_exogenous_research_registry(REGISTRY_PATH)
    signal = next(
        signal
        for family in payload["families"]
        for signal in family["signals"]
        if signal["id"] == "residual_load_late_hour_coverage_repair"
    )
    decision = audit.signal(signal["id"])

    assert decision.priority == "P0"
    assert decision.status == "blocked"
    assert decision.ready_for_offline_screen is False
    assert signal["operational_source"]["provider"] == "Saturn"
    assert signal["operational_source"]["dataset"] == (
        "exact_residual_load_revision_history"
    )
    pit_note = signal["gates"]["point_in_time_vintages"]["note"].casefold()
    coverage_note = signal["gates"]["zone_coverage"]["note"].casefold()
    assert all(
        forbidden in pit_note
        for forbidden in ("interpolation", "forward-fill", "backward-fill", "latest")
    )
    assert "543" in " ".join(signal["blockers"])
    assert "22:00–23:00" in " ".join(signal["blockers"])
    assert "23/24/25" in coverage_note
    assert "archives" in coverage_note

    with pytest.raises(ExogenousRegistryError, match="offline PIT screen refused"):
        audit.require_screenable(signal["id"])


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("training_allowed", True),
        ("production_writes_allowed", True),
        ("storm_mkonline_as_inputs", True),
        ("latest_backfill_allowed", True),
        ("interpolation_allowed", True),
        ("forward_backward_fill_allowed", True),
        ("raw_media_as_inputs", True),
        ("direct_llm_price_output", True),
        ("fail_closed", False),
    ],
)
def test_policy_cannot_authorize_unsafe_actions(
    field: str,
    unsafe_value: bool,
) -> None:
    payload = _payload()
    payload["policy"][field] = unsafe_value

    with pytest.raises(ExogenousRegistryError, match="policy"):
        audit_exogenous_research_registry(payload)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("timezone", "UTC"),
        ("local_time", "08:01"),
        ("delivery_horizon", "intraday"),
        ("physical_actual_latest_day_offset", -1),
    ],
)
def test_cutoff_is_exactly_day_ahead_d_minus_1_0800_paris(
    field: str,
    unsafe_value: object,
) -> None:
    payload = _payload()
    payload["cutoff"][field] = unsafe_value

    with pytest.raises(ExogenousRegistryError, match="cutoff"):
        audit_exogenous_research_registry(payload)


@pytest.mark.parametrize("token", ["Storm", "mKoNlInE"])
def test_competing_forecasts_are_forbidden_in_operational_lineage(token: str) -> None:
    payload = _payload()
    _first_signal(payload)["operational_source"]["dataset"] = f"hidden_{token}_input"

    with pytest.raises(ExogenousRegistryError, match="forbidden input tokens"):
        audit_exogenous_research_registry(payload)


def test_competing_forecasts_are_forbidden_in_output_schema() -> None:
    payload = _payload()
    _first_signal(payload)["feature_contract"]["output_columns"].append(
        "storm_residual"
    )

    with pytest.raises(ExogenousRegistryError, match="forbidden outputs"):
        audit_exogenous_research_registry(payload)


def test_physical_actuals_must_stop_at_d_minus_2() -> None:
    payload = _payload()
    _first_signal(payload)["feature_contract"][
        "physical_actual_latest_day_offset"
    ] = -1

    with pytest.raises(ExogenousRegistryError, match="D-2"):
        audit_exogenous_research_registry(payload)


def test_non_physical_signal_cannot_declare_an_actual_lag() -> None:
    payload = _payload()
    nwp = payload["families"][1]["signals"][0]
    nwp["feature_contract"]["physical_actual_latest_day_offset"] = -2

    with pytest.raises(ExogenousRegistryError, match="must be null"):
        audit_exogenous_research_registry(payload)


def test_same_day_uncleared_price_is_always_rejected() -> None:
    payload = _payload()
    _first_signal(payload)["feature_contract"][
        "same_day_uncleared_price_allowed"
    ] = True

    with pytest.raises(ExogenousRegistryError, match="uncleared"):
        audit_exogenous_research_registry(payload)


def test_raw_or_non_numeric_feature_contract_is_rejected() -> None:
    payload = _payload()
    _first_signal(payload)["feature_contract"]["numeric_only"] = False

    with pytest.raises(ExogenousRegistryError, match="numeric"):
        audit_exogenous_research_registry(payload)


def test_unproved_signal_cannot_claim_screenable_status() -> None:
    payload = _payload()
    _first_signal(payload)["status"] = "ready_for_offline_screen"
    _first_signal(payload)["blockers"] = []

    with pytest.raises(ExogenousRegistryError, match="every proof gate passed"):
        audit_exogenous_research_registry(payload)


def test_all_proofs_and_open_phase_a_are_required_for_screening() -> None:
    payload = _payload()
    signal = _first_signal(payload)
    # The checked-in YAML intentionally reuses the identical downstream phase
    # gates via an anchor.  Detach the one declaration being promoted here.
    signal["phase_gates"] = copy.deepcopy(signal["phase_gates"])
    signal["status"] = "ready_for_offline_screen"
    signal["blockers"] = []
    for gate_name, gate in signal["gates"].items():
        gate["status"] = "passed"
        if gate_name in {
            "availability_d_minus_1_0800",
            "point_in_time_vintages",
            "zone_coverage",
        }:
            gate["evidence_refs"] = [
                f"artifact:runs/audits/{gate_name}.json#sha256={'a' * 64}"
            ]
        elif gate_name == "license_internal_research":
            gate["evidence_refs"] = ["license:approved-internal-research-v1"]
        else:
            gate["evidence_refs"] = [f"contract:{gate_name}:v1"]
    signal["phase_gates"]["A"] = {
        "status": "open",
        "note": "Toutes les preuves PIT ont été revues; screen A autorisé.",
    }

    audit = audit_exogenous_research_registry(payload)
    selected = audit.require_screenable(signal["id"])

    assert selected.ready_for_offline_screen is True
    assert audit.ready_signal_ids == (signal["id"],)
    assert not selected.blockers


def test_require_screenable_explains_blocked_signal() -> None:
    audit = audit_exogenous_research_registry(_payload())

    with pytest.raises(ExogenousRegistryError, match="offline PIT screen refused"):
        audit.require_screenable("residual_load_quantiles_pit")


def test_passed_proof_gate_requires_an_evidence_reference() -> None:
    payload = _payload()
    gate = _first_signal(payload)["gates"]["availability_d_minus_1_0800"]
    gate["status"] = "passed"
    gate["evidence_refs"] = []

    with pytest.raises(ExogenousRegistryError, match="evidence ref"):
        audit_exogenous_research_registry(payload)


def test_operational_proof_requires_a_hashed_artifact_reference() -> None:
    payload = _payload()
    gate = _first_signal(payload)["gates"]["availability_d_minus_1_0800"]
    gate["status"] = "passed"
    gate["evidence_refs"] = ["audit:unhashed-and-mutable"]

    with pytest.raises(ExogenousRegistryError, match="immutable.*sha256"):
        audit_exogenous_research_registry(payload)


def test_later_phase_cannot_open_before_an_earlier_blocked_phase() -> None:
    payload = _payload()
    _first_signal(payload)["phase_gates"]["B1"] = {
        "status": "open",
        "note": "Unsafe attempt to skip phase A.",
    }

    with pytest.raises(ExogenousRegistryError, match="earlier phase is blocked"):
        audit_exogenous_research_registry(payload)


def test_new_scientific_source_requires_code_review() -> None:
    payload = _payload()
    payload["families"][0]["scientific_basis"][0]["url"] = (
        "https://example.invalid/unreviewed-paper"
    )

    with pytest.raises(ExogenousRegistryError, match="primary-source allowlist"):
        audit_exogenous_research_registry(payload)


def test_family_set_cannot_silently_drift_from_reviewed_roadmap() -> None:
    payload = _payload()
    payload["families"] = payload["families"][:-1]

    with pytest.raises(ExogenousRegistryError, match="missing="):
        audit_exogenous_research_registry(payload)


def test_duplicate_signal_id_is_rejected() -> None:
    payload = _payload()
    duplicate = copy.deepcopy(payload["families"][0]["signals"][0])
    payload["families"][1]["signals"].append(duplicate)

    with pytest.raises(ExogenousRegistryError, match="duplicate signal id"):
        audit_exogenous_research_registry(payload)


def test_unknown_or_missing_schema_keys_fail_closed() -> None:
    payload = _payload()
    _first_signal(payload)["future_shortcut"] = True

    with pytest.raises(ExogenousRegistryError, match="unknown=.*future_shortcut"):
        audit_exogenous_research_registry(payload)


def test_loader_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text("- not\n- a\n- registry\n", encoding="utf-8")

    with pytest.raises(ExogenousRegistryError, match="mapping expected"):
        load_exogenous_research_registry(path)
