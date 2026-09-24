"""Fail-closed configuration contract for operational LoRA activation.

This module does not launch forecasts and does not register candidates.  It
only resolves an explicitly enabled autonomous/Kalman route after the sealed
bundle has been revalidated by :mod:`chronos2_exogenous.production`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from . import production
from .governance import ExogenousGovernanceError


ACTIVATION_SCHEMA_VERSION = 1
SUPPORTED_ZONES = ("FR", "DE", "BE", "NL", "ES")
SUPPORTED_MODES = ("autonomous", "kalman")
VALIDATION_FAILURE_POLICY = "error"

LORA_AUTONOMOUS_MODEL = production.OUTPUT_MODEL
INCUMBENT_AUTONOMOUS_MODEL = "residual_corrected"
KALMAN_OUTPUT_MODEL = "residual_kalman"


class ExogenousActivationContractError(RuntimeError):
    """Raised when LoRA activation cannot be proven safe."""


@dataclass(frozen=True)
class AutonomousRoute:
    activated_model: str
    fallback_model: str


@dataclass(frozen=True)
class KalmanRoute:
    output_model: str
    upstream_model: str
    fallback_model: str
    fallback_upstream_model: str


@dataclass(frozen=True)
class ZoneActivation:
    zone: str
    enabled_modes: tuple[str, ...]
    alias: str | None
    candidate_id: str | None
    candidate_model: str | None
    bundle_manifest_sha256: str | None
    artifact_checksums_sha256: str | None
    live_source_manifest_path: Path | None
    live_source_manifest_sha256: str | None

    @property
    def has_pinned_bundle(self) -> bool:
        return self.alias is not None


@dataclass(frozen=True)
class ActivationResolution:
    zone: str
    mode: str
    lora_enabled: bool
    selected_model: str
    fallback_model: str
    kalman_upstream_model: str | None
    fallback_upstream_model: str | None
    bundle: production.PromotedBundle | None
    live_source_manifest_path: Path | None
    live_source_manifest_sha256: str | None
    runtime_output_root: Path
    reason: str


@dataclass(frozen=True)
class ExogenousActivationContract:
    source_path: Path
    project_root: Path
    registry_path: Path
    runtime_output_root: Path
    validation_failure_policy: str
    autonomous: AutonomousRoute
    kalman: KalmanRoute
    zones: Mapping[str, ZoneActivation]
    verified_bundles: Mapping[str, production.PromotedBundle]

    def resolve(self, *, zone: str, mode: str) -> ActivationResolution:
        """Resolve one route; an invalid active bundle raises, never falls back."""

        canonical_zone = str(zone).strip().upper()
        canonical_mode = str(mode).strip().casefold()
        if canonical_zone not in self.zones:
            raise ExogenousActivationContractError(
                f"Zone absente du contrat LoRA: {canonical_zone!r}."
            )
        if canonical_mode not in SUPPORTED_MODES:
            raise ExogenousActivationContractError(
                f"Mode LoRA non supporte: {mode!r}."
            )
        zone_config = self.zones[canonical_zone]
        if canonical_mode not in zone_config.enabled_modes:
            if canonical_mode == "autonomous":
                return ActivationResolution(
                    zone=canonical_zone,
                    mode=canonical_mode,
                    lora_enabled=False,
                    selected_model=self.autonomous.fallback_model,
                    fallback_model=self.autonomous.fallback_model,
                    kalman_upstream_model=None,
                    fallback_upstream_model=None,
                    bundle=None,
                    live_source_manifest_path=None,
                    live_source_manifest_sha256=None,
                    runtime_output_root=self.runtime_output_root,
                    reason="lora_explicitly_disabled",
                )
            return ActivationResolution(
                zone=canonical_zone,
                mode=canonical_mode,
                lora_enabled=False,
                selected_model=self.kalman.fallback_model,
                fallback_model=self.kalman.fallback_model,
                kalman_upstream_model=self.kalman.fallback_upstream_model,
                fallback_upstream_model=self.kalman.fallback_upstream_model,
                bundle=None,
                live_source_manifest_path=None,
                live_source_manifest_sha256=None,
                runtime_output_root=self.runtime_output_root,
                reason="lora_explicitly_disabled",
            )

        promoted = self.verified_bundles.get(canonical_zone)
        if promoted is None:
            raise ExogenousActivationContractError(
                f"{canonical_zone}: mode actif sans bundle verifie au preflight."
            )
        if canonical_mode == "autonomous":
            return ActivationResolution(
                zone=canonical_zone,
                mode=canonical_mode,
                lora_enabled=True,
                selected_model=self.autonomous.activated_model,
                fallback_model=self.autonomous.fallback_model,
                kalman_upstream_model=None,
                fallback_upstream_model=None,
                bundle=promoted,
                live_source_manifest_path=zone_config.live_source_manifest_path,
                live_source_manifest_sha256=zone_config.live_source_manifest_sha256,
                runtime_output_root=self.runtime_output_root,
                reason="promoted_bundle_verified",
            )
        return ActivationResolution(
            zone=canonical_zone,
            mode=canonical_mode,
            lora_enabled=True,
            selected_model=self.kalman.output_model,
            fallback_model=self.kalman.fallback_model,
            kalman_upstream_model=self.kalman.upstream_model,
            fallback_upstream_model=self.kalman.fallback_upstream_model,
            bundle=promoted,
            live_source_manifest_path=zone_config.live_source_manifest_path,
            live_source_manifest_sha256=zone_config.live_source_manifest_sha256,
            runtime_output_root=self.runtime_output_root,
            reason="promoted_bundle_verified",
        )

    def validate_enabled_bundles(self) -> Mapping[str, production.PromotedBundle]:
        """Preflight every enabled zone through the production verifier."""

        verified: dict[str, production.PromotedBundle] = dict(self.verified_bundles)
        for zone, zone_config in self.zones.items():
            if zone_config.enabled_modes and zone not in verified:
                verified[zone] = _load_pinned_bundle(self, zone_config)
        return MappingProxyType(verified)


def _exact_keys(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExogenousActivationContractError(f"{label} doit etre un objet.")
    keys = {str(key) for key in value}
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise ExogenousActivationContractError(
            f"{label}: cles manquantes={missing}, inattendues={extra}."
        )
    return value


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExogenousActivationContractError(f"{label} doit etre une chaine non vide.")
    return value.strip()


def _optional_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, label=label)


def _optional_sha256(value: object, *, label: str) -> str | None:
    digest = _optional_string(value, label=label)
    if digest is None:
        return None
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ExogenousActivationContractError(f"{label} doit etre un SHA-256 minuscule.")
    return digest


def _resolve_path(value: object, *, base: Path, label: str) -> Path:
    raw = _required_string(value, label=label)
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_zone(zone: str, value: object, *, project_root: Path) -> ZoneActivation:
    payload = _exact_keys(
        value,
        {
            "enabled_modes",
            "alias",
            "candidate_id",
            "candidate_model",
            "bundle_manifest_sha256",
            "artifact_checksums_sha256",
            "live_source_manifest_path",
            "live_source_manifest_sha256",
        },
        label=f"zones.{zone}",
    )
    raw_modes = payload["enabled_modes"]
    if not isinstance(raw_modes, list) or any(not isinstance(mode, str) for mode in raw_modes):
        raise ExogenousActivationContractError(
            f"zones.{zone}.enabled_modes doit etre une liste de chaines."
        )
    enabled_modes = tuple(mode.strip().casefold() for mode in raw_modes)
    if (
        any(not mode or mode not in SUPPORTED_MODES for mode in enabled_modes)
        or len(set(enabled_modes)) != len(enabled_modes)
    ):
        raise ExogenousActivationContractError(
            f"zones.{zone}.enabled_modes doit etre un sous-ensemble unique de {SUPPORTED_MODES}."
        )
    alias = _optional_string(payload["alias"], label=f"zones.{zone}.alias")
    candidate_id = _optional_string(
        payload["candidate_id"], label=f"zones.{zone}.candidate_id"
    )
    candidate_model = _optional_string(
        payload["candidate_model"], label=f"zones.{zone}.candidate_model"
    )
    bundle_sha = _optional_sha256(
        payload["bundle_manifest_sha256"],
        label=f"zones.{zone}.bundle_manifest_sha256",
    )
    artifacts_sha = _optional_sha256(
        payload["artifact_checksums_sha256"],
        label=f"zones.{zone}.artifact_checksums_sha256",
    )
    raw_live_manifest = _optional_string(
        payload["live_source_manifest_path"],
        label=f"zones.{zone}.live_source_manifest_path",
    )
    live_manifest = (
        _resolve_path(
            raw_live_manifest,
            base=project_root,
            label=f"zones.{zone}.live_source_manifest_path",
        )
        if raw_live_manifest is not None
        else None
    )
    live_manifest_sha = _optional_sha256(
        payload["live_source_manifest_sha256"],
        label=f"zones.{zone}.live_source_manifest_sha256",
    )
    identity = (alias, candidate_id, candidate_model, bundle_sha, artifacts_sha)
    present = tuple(item is not None for item in identity)
    if any(present) and not all(present):
        raise ExogenousActivationContractError(
            f"zones.{zone}: l'identite du bundle doit etre entierement renseignee."
        )
    if enabled_modes and not all(present):
        raise ExogenousActivationContractError(
            f"zones.{zone}: aucun mode ne peut etre active sans bundle epingle."
        )
    runtime_present = (live_manifest is not None, live_manifest_sha is not None)
    if any(runtime_present) and not all(runtime_present):
        raise ExogenousActivationContractError(
            f"zones.{zone}: manifeste live et SHA doivent etre renseignes ensemble."
        )
    if enabled_modes and not all(runtime_present):
        raise ExogenousActivationContractError(
            f"zones.{zone}: aucun mode ne peut etre active sans manifeste "
            "de sources prospectives epingle."
        )
    if live_manifest is not None:
        if not live_manifest.is_file():
            raise ExogenousActivationContractError(
                f"zones.{zone}: manifeste live introuvable: {live_manifest}."
            )
        assert live_manifest_sha is not None
        if _sha256(live_manifest) != live_manifest_sha:
            raise ExogenousActivationContractError(
                f"zones.{zone}: SHA du manifeste live divergent."
            )
    return ZoneActivation(
        zone=zone,
        enabled_modes=enabled_modes,
        alias=alias,
        candidate_id=candidate_id,
        candidate_model=candidate_model,
        bundle_manifest_sha256=bundle_sha,
        artifact_checksums_sha256=artifacts_sha,
        live_source_manifest_path=live_manifest,
        live_source_manifest_sha256=live_manifest_sha,
    )


def _load_pinned_bundle(
    contract: ExogenousActivationContract,
    zone_config: ZoneActivation,
) -> production.PromotedBundle:
    if not zone_config.has_pinned_bundle:
        raise ExogenousActivationContractError(
            f"{zone_config.zone}: bundle promu non epingle."
        )
    assert zone_config.alias is not None
    try:
        promoted = production.load_registered_bundle(
            registry_path=contract.registry_path,
            alias=zone_config.alias,
            expected_zone=zone_config.zone,
        )
    except (production.ExogenousProductionError, ExogenousGovernanceError) as exc:
        raise ExogenousActivationContractError(
            f"{zone_config.zone}: activation LoRA refusee par le contrat production: {exc}"
        ) from exc
    expected = {
        "candidate_id": zone_config.candidate_id,
        "candidate_model": zone_config.candidate_model,
        "bundle_manifest_sha256": zone_config.bundle_manifest_sha256,
        "artifact_checksums_sha256": zone_config.artifact_checksums_sha256,
    }
    mismatches = [
        field
        for field, expected_value in expected.items()
        if getattr(promoted, field) != expected_value
    ]
    if mismatches:
        raise ExogenousActivationContractError(
            f"{zone_config.zone}: identite du bundle divergent du contrat: "
            + ", ".join(mismatches)
        )
    return promoted


def load_activation_contract(path: str | Path) -> ExogenousActivationContract:
    """Load the structural contract; active bundles are checked on resolution."""

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExogenousActivationContractError(
            f"Configuration d'activation LoRA illisible: {source}."
        ) from exc
    payload = _exact_keys(
        raw,
        {
            "schema_version",
            "project_root",
            "registry_path",
            "runtime_output_root",
            "validation_failure_policy",
            "pipelines",
            "zones",
        },
        label="activation_lora",
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != ACTIVATION_SCHEMA_VERSION
    ):
        raise ExogenousActivationContractError(
            f"schema_version attendu={ACTIVATION_SCHEMA_VERSION}."
        )
    project_root = _resolve_path(
        payload["project_root"], base=source.parent, label="project_root"
    )
    registry_path = _resolve_path(
        payload["registry_path"], base=project_root, label="registry_path"
    )
    runtime_output_root = _resolve_path(
        payload["runtime_output_root"],
        base=project_root,
        label="runtime_output_root",
    )
    live_root = (project_root / "runs" / "live").resolve()
    if runtime_output_root == live_root or live_root in runtime_output_root.parents:
        raise ExogenousActivationContractError(
            "runtime_output_root doit rester hors de runs/live."
        )
    failure_policy = _required_string(
        payload["validation_failure_policy"], label="validation_failure_policy"
    ).casefold()
    if failure_policy != VALIDATION_FAILURE_POLICY:
        raise ExogenousActivationContractError(
            "validation_failure_policy doit etre 'error'; un bundle invalide ne peut pas "
            "declencher un fallback silencieux."
        )

    pipelines = _exact_keys(
        payload["pipelines"], {"autonomous", "kalman"}, label="pipelines"
    )
    autonomous_raw = _exact_keys(
        pipelines["autonomous"],
        {"activated_model", "fallback_model"},
        label="pipelines.autonomous",
    )
    autonomous = AutonomousRoute(
        activated_model=_required_string(
            autonomous_raw["activated_model"],
            label="pipelines.autonomous.activated_model",
        ),
        fallback_model=_required_string(
            autonomous_raw["fallback_model"],
            label="pipelines.autonomous.fallback_model",
        ),
    )
    if autonomous != AutonomousRoute(
        activated_model=LORA_AUTONOMOUS_MODEL,
        fallback_model=INCUMBENT_AUTONOMOUS_MODEL,
    ):
        raise ExogenousActivationContractError(
            "Route autonomous invalide: LoRA doit produire exogenous_residual_corrected "
            "et le fallback doit rester residual_corrected."
        )

    kalman_raw = _exact_keys(
        pipelines["kalman"],
        {
            "output_model",
            "upstream_model",
            "fallback_model",
            "fallback_upstream_model",
        },
        label="pipelines.kalman",
    )
    kalman = KalmanRoute(
        output_model=_required_string(
            kalman_raw["output_model"], label="pipelines.kalman.output_model"
        ),
        upstream_model=_required_string(
            kalman_raw["upstream_model"], label="pipelines.kalman.upstream_model"
        ),
        fallback_model=_required_string(
            kalman_raw["fallback_model"], label="pipelines.kalman.fallback_model"
        ),
        fallback_upstream_model=_required_string(
            kalman_raw["fallback_upstream_model"],
            label="pipelines.kalman.fallback_upstream_model",
        ),
    )
    if kalman != KalmanRoute(
        output_model=KALMAN_OUTPUT_MODEL,
        upstream_model=autonomous.activated_model,
        fallback_model=KALMAN_OUTPUT_MODEL,
        fallback_upstream_model=autonomous.fallback_model,
    ):
        raise ExogenousActivationContractError(
            "Route Kalman invalide: son upstream actif doit etre le modele autonome LoRA "
            "et son upstream de fallback le modele autonome incumbent."
        )

    zones_raw = _exact_keys(payload["zones"], set(SUPPORTED_ZONES), label="zones")
    zones = {
        zone: _parse_zone(zone, zones_raw[zone], project_root=project_root)
        for zone in SUPPORTED_ZONES
    }
    contract = ExogenousActivationContract(
        source_path=source,
        project_root=project_root,
        registry_path=registry_path,
        runtime_output_root=runtime_output_root,
        validation_failure_policy=failure_policy,
        autonomous=autonomous,
        kalman=kalman,
        zones=zones,
        verified_bundles=MappingProxyType({}),
    )
    return replace(
        contract,
        verified_bundles=contract.validate_enabled_bundles(),
    )


__all__ = [
    "ACTIVATION_SCHEMA_VERSION",
    "ActivationResolution",
    "ExogenousActivationContract",
    "ExogenousActivationContractError",
    "SUPPORTED_MODES",
    "SUPPORTED_ZONES",
    "load_activation_contract",
]
