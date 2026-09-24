from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

import chronos2_exogenous.activation_contract as activation
from chronos2_exogenous.activation_contract import (
    ExogenousActivationContractError,
    load_activation_contract,
)
from chronos2_exogenous.governance import ExogenousGovernanceError
from chronos2_exogenous.production import ExogenousProductionError, PromotedBundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict[str, object]:
    inactive = {
        "enabled_modes": [],
        "alias": None,
        "candidate_id": None,
        "candidate_model": None,
        "bundle_manifest_sha256": None,
        "artifact_checksums_sha256": None,
        "live_source_manifest_path": None,
        "live_source_manifest_sha256": None,
    }
    return {
        "schema_version": 1,
        "project_root": ".",
        "registry_path": "registry.json",
        "runtime_output_root": "operational-runs",
        "validation_failure_policy": "error",
        "pipelines": {
            "autonomous": {
                "activated_model": "exogenous_residual_corrected",
                "fallback_model": "residual_corrected",
            },
            "kalman": {
                "output_model": "residual_kalman",
                "upstream_model": "exogenous_residual_corrected",
                "fallback_model": "residual_kalman",
                "fallback_upstream_model": "residual_corrected",
            },
        },
        "zones": {
            zone: dict(inactive) for zone in activation.SUPPORTED_ZONES
        },
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "activation.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _promoted(tmp_path: Path, *, zone: str = "FR") -> PromotedBundle:
    bundle = tmp_path / "bundle"
    return PromotedBundle(
        alias=f"lora-{zone.casefold()}-v1",
        bundle_path=bundle,
        candidate_id=f"candidate-{zone.casefold()}-v1",
        candidate_model="chronos2-exogenous-lora-v1",
        zone=zone,
        bundle_manifest_sha256="a" * 64,
        artifact_checksums_sha256="b" * 64,
        checkpoint_path=bundle / "checkpoint",
        schema_path=bundle / "schema.json",
        experiment_manifest_path=bundle / "experiment_manifest.json",
        residual_corrector_path=bundle / "corrector.json",
        oof_audit_path=bundle / "oof.json",
        rolling_predictions_path=bundle / "rolling.csv.gz",
    )


def _enable(payload: dict[str, object], promoted: PromotedBundle, *modes: str) -> None:
    zones = payload["zones"]
    assert isinstance(zones, dict)
    live_manifest = promoted.bundle_path.parent / f"live-{promoted.zone}.json"
    live_manifest.write_text("{}\n", encoding="utf-8")
    zones[promoted.zone] = {
        "enabled_modes": list(modes),
        "alias": promoted.alias,
        "candidate_id": promoted.candidate_id,
        "candidate_model": promoted.candidate_model,
        "bundle_manifest_sha256": promoted.bundle_manifest_sha256,
        "artifact_checksums_sha256": promoted.artifact_checksums_sha256,
        "live_source_manifest_path": str(live_manifest),
        "live_source_manifest_sha256": hashlib.sha256(
            live_manifest.read_bytes()
        ).hexdigest(),
    }


def test_repository_contract_keeps_every_zone_on_explicit_fallback() -> None:
    contract = load_activation_contract(
        PROJECT_ROOT / "config" / "chronos2_exogenous_activation_v1.yaml"
    )

    for zone in activation.SUPPORTED_ZONES:
        autonomous = contract.resolve(zone=zone, mode="autonomous")
        kalman = contract.resolve(zone=zone, mode="kalman")
        assert autonomous.lora_enabled is False
        assert autonomous.selected_model == "residual_corrected"
        assert kalman.lora_enabled is False
        assert kalman.selected_model == "residual_kalman"
        assert kalman.kalman_upstream_model == "residual_corrected"


def test_promoted_bundle_is_pinned_for_autonomous_and_kalman(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    promoted = _promoted(tmp_path)
    payload = _payload()
    _enable(payload, promoted, "autonomous", "kalman")
    calls: list[tuple[Path, str, str]] = []

    def load_registered_bundle(**kwargs: object) -> PromotedBundle:
        calls.append(
            (
                Path(str(kwargs["registry_path"])),
                str(kwargs["alias"]),
                str(kwargs["expected_zone"]),
            )
        )
        return promoted

    monkeypatch.setattr(activation.production, "load_registered_bundle", load_registered_bundle)
    contract = load_activation_contract(_write(tmp_path, payload))

    autonomous = contract.resolve(zone="fr", mode="autonomous")
    kalman = contract.resolve(zone="FR", mode="kalman")

    assert autonomous.lora_enabled is True
    assert autonomous.selected_model == "exogenous_residual_corrected"
    assert autonomous.bundle is promoted
    assert kalman.lora_enabled is True
    assert kalman.selected_model == "residual_kalman"
    assert kalman.kalman_upstream_model == "exogenous_residual_corrected"
    assert kalman.fallback_upstream_model == "residual_corrected"
    assert calls == [(tmp_path / "registry.json", promoted.alias, "FR")]


@pytest.mark.parametrize(
    "production_failure",
    [
        "Bundle non promu (decision='shadow').",
        "Le manifeste d'experience ne fournit pas de PIT production.",
        "Le manifeste d'experience n'evalue pas le pipeline final complet.",
    ],
)
def test_active_shadow_or_missing_production_evidence_never_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    production_failure: str,
) -> None:
    promoted = _promoted(tmp_path)
    payload = _payload()
    _enable(payload, promoted, "autonomous")

    def reject(**_: object) -> PromotedBundle:
        raise ExogenousProductionError(production_failure)

    monkeypatch.setattr(activation.production, "load_registered_bundle", reject)

    with pytest.raises(ExogenousActivationContractError, match="activation LoRA refusee"):
        load_activation_contract(_write(tmp_path, payload))


def test_incomplete_shadow_artifact_error_is_normalised_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    promoted = _promoted(tmp_path)
    payload = _payload()
    _enable(payload, promoted, "kalman")

    def reject(**_: object) -> PromotedBundle:
        raise ExogenousGovernanceError("artifact_checksums.json absent")

    monkeypatch.setattr(activation.production, "load_registered_bundle", reject)

    with pytest.raises(ExogenousActivationContractError, match="activation LoRA refusee"):
        load_activation_contract(_write(tmp_path, payload))


def test_bundle_identity_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    promoted = _promoted(tmp_path)
    payload = _payload()
    _enable(payload, promoted, "kalman")
    other = PromotedBundle(
        **{
            **promoted.__dict__,
            "bundle_manifest_sha256": "c" * 64,
        }
    )
    monkeypatch.setattr(
        activation.production, "load_registered_bundle", lambda **_: other
    )

    with pytest.raises(ExogenousActivationContractError, match="bundle_manifest_sha256"):
        load_activation_contract(_write(tmp_path, payload))


def test_kalman_upstream_and_silent_fallback_cannot_be_relaxed(tmp_path: Path) -> None:
    payload = _payload()
    pipelines = payload["pipelines"]
    assert isinstance(pipelines, dict)
    kalman = pipelines["kalman"]
    assert isinstance(kalman, dict)
    kalman["upstream_model"] = "residual_corrected"

    with pytest.raises(ExogenousActivationContractError, match="Route Kalman invalide"):
        load_activation_contract(_write(tmp_path, payload))

    payload = _payload()
    payload["validation_failure_policy"] = "fallback"
    with pytest.raises(ExogenousActivationContractError, match="fallback silencieux"):
        load_activation_contract(_write(tmp_path, payload))


def test_active_zone_requires_complete_pinned_identity(tmp_path: Path) -> None:
    payload = _payload()
    zones = payload["zones"]
    assert isinstance(zones, dict)
    fr = zones["FR"]
    assert isinstance(fr, dict)
    fr["enabled_modes"] = ["autonomous"]
    fr["alias"] = "shadow-artifact"

    with pytest.raises(ExogenousActivationContractError, match="entierement renseignee"):
        load_activation_contract(_write(tmp_path, payload))


def test_direct_bundle_path_is_not_an_accepted_bypass(tmp_path: Path) -> None:
    payload = _payload()
    payload["bundle_path"] = str(
        PROJECT_ROOT
        / "runs"
        / "experiments"
        / "chronos2_exogenous_lora_poc_v1"
        / "artifact"
    )

    with pytest.raises(ExogenousActivationContractError, match="inattendues"):
        load_activation_contract(_write(tmp_path, payload))
