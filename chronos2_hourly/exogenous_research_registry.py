"""Fail-closed registry for causal exogenous-feature research.

The registry is deliberately *not* a feature loader, trainer, or production
switch.  It only validates declarations and answers whether an experiment is
allowed to enter an offline PIT screen.  A signal remains blocked until every
proof gate is explicitly passed and the phase-A gate is opened.

This separation prevents an attractive but non-reproducible data series from
being added to a model configuration before its D-1 08:00 availability,
point-in-time history, licence, lineage, numeric schema, and zone coverage have
all been documented.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = 1
EXPECTED_FAMILIES = frozenset(
    {
        "probabilistic_fundamentals",
        "archived_nwp",
        "outages_nuclear",
        "fuel_carbon_coal",
        "cross_zone_delayed_network",
        "online_ensemble",
        "text_events",
        "weather_images_numeric",
    }
)
SUPPORTED_ZONES = frozenset({"FR", "DE", "BE", "NL", "ES"})
PRIORITIES = frozenset({"P0", "P1", "P2", "P3"})
EXPERIMENT_PHASES = ("A", "B1", "B2", "FINAL", "SHADOW")
SIGNAL_STATUSES = frozenset(
    {"blocked", "ready_for_offline_screen", "rejected"}
)
GATE_STATUSES = frozenset({"passed", "blocked"})
PHASE_GATE_STATUSES = frozenset({"open", "passed", "blocked"})
PROOF_GATES = (
    "availability_d_minus_1_0800",
    "point_in_time_vintages",
    "license_internal_research",
    "causal_lineage",
    "numeric_schema",
    "zone_coverage",
)
FORBIDDEN_INPUT_TOKENS = ("storm", "mkonline")
FEATURE_KINDS = frozenset(
    {
        "probabilistic_numeric",
        "nwp_numeric",
        "outage_numeric",
        "market_numeric",
        "cross_zone_numeric",
        "ensemble_numeric",
        "event_numeric",
        "image_derived_numeric",
    }
)

# Primary research sources already reviewed in the scientific roadmap.  Adding
# a new paper is an explicit code-review action, not an untracked YAML edit.
APPROVED_PRIMARY_SOURCES = frozenset(
    {
        "https://doi.org/10.1016/j.renene.2026.125844",
        "https://arxiv.org/abs/2501.06180",
        "https://doi.org/10.1109/TPWRS.2022.3180119",
        "https://doi.org/10.1038/s41467-026-75433-7",
        "https://doi.org/10.1016/j.eneco.2023.107241",
        "https://doi.org/10.1016/j.apenergy.2022.118752",
        "https://arxiv.org/abs/2508.04875",
        "https://arxiv.org/abs/2606.07014",
        "https://doi.org/10.48550/arXiv.2601.02856",
        "https://arxiv.org/abs/2601.02856",
        "https://arxiv.org/abs/2506.11050",
    }
)

_ROOT_KEYS = {
    "schema_version",
    "registry_id",
    "purpose",
    "cutoff",
    "policy",
    "families",
}
_CUTOFF_KEYS = {
    "timezone",
    "local_time",
    "delivery_horizon",
    "physical_actual_latest_day_offset",
}
_POLICY_KEYS = {
    "scope",
    "fail_closed",
    "training_allowed",
    "production_writes_allowed",
    "storm_mkonline_as_inputs",
    "latest_backfill_allowed",
    "interpolation_allowed",
    "forward_backward_fill_allowed",
    "raw_media_as_inputs",
    "direct_llm_price_output",
}
_FAMILY_KEYS = {"id", "title", "priority", "scientific_basis", "signals"}
_BASIS_KEYS = {"url", "claim"}
_SIGNAL_KEYS = {
    "id",
    "title",
    "zones",
    "phase",
    "status",
    "operational_source",
    "feature_contract",
    "gates",
    "phase_gates",
    "blockers",
}
_SOURCE_KEYS = {"provider", "dataset", "series_or_endpoint"}
_FEATURE_KEYS = {
    "kind",
    "output_columns",
    "numeric_only",
    "uses_physical_actuals",
    "physical_actual_latest_day_offset",
    "same_day_uncleared_price_allowed",
}
_GATE_KEYS = {"status", "evidence_refs", "note"}
_PHASE_GATE_KEYS = {"status", "note"}
_SLUG = re.compile(r"^[a-z][a-z0-9_]*$")
_ARTIFACT_EVIDENCE = re.compile(
    r"^artifact:.+#sha256=[0-9a-fA-F]{64}$"
)


class ExogenousRegistryError(ValueError):
    """Raised when a registry declaration is unsafe or structurally invalid."""


@dataclass(frozen=True)
class ExogenousSignalAudit:
    """One signal's fail-closed research decision."""

    signal_id: str
    family_id: str
    zones: tuple[str, ...]
    priority: str
    phase: str
    status: str
    ready_for_offline_screen: bool
    blockers: tuple[str, ...]
    proof_gates: Mapping[str, str]
    phase_gates: Mapping[str, str]
    declaration_sha256: str

    def require_screenable(self) -> "ExogenousSignalAudit":
        if not self.ready_for_offline_screen:
            detail = "; ".join(self.blockers) or "proof gates are not complete"
            raise ExogenousRegistryError(
                f"{self.signal_id}: offline PIT screen refused: {detail}"
            )
        return self


@dataclass(frozen=True)
class ExogenousRegistryAudit:
    """Validated immutable view of an exogenous research registry."""

    registry_id: str
    registry_sha256: str
    signals: tuple[ExogenousSignalAudit, ...]

    @property
    def ready_signal_ids(self) -> tuple[str, ...]:
        return tuple(
            item.signal_id for item in self.signals if item.ready_for_offline_screen
        )

    @property
    def blocked_signal_ids(self) -> tuple[str, ...]:
        return tuple(
            item.signal_id
            for item in self.signals
            if not item.ready_for_offline_screen
        )

    def signal(self, signal_id: str) -> ExogenousSignalAudit:
        requested = str(signal_id).strip()
        for item in self.signals:
            if item.signal_id == requested:
                return item
        raise ExogenousRegistryError(f"unknown signal_id={signal_id!r}")

    def require_screenable(self, signal_id: str) -> ExogenousSignalAudit:
        return self.signal(signal_id).require_screenable()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExogenousRegistryError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExogenousRegistryError(f"{name} must be a sequence")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ExogenousRegistryError(
            f"{name} keys invalid; missing={missing}, unknown={unknown}"
        )


def _text(value: Any, *, name: str) -> str:
    if value is None:
        raise ExogenousRegistryError(f"{name} must be explicit")
    result = str(value).strip()
    if not result:
        raise ExogenousRegistryError(f"{name} must be non-empty")
    return result


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ExogenousRegistryError(f"{name} must be an explicit boolean")
    return value


def _slug(value: Any, *, name: str) -> str:
    result = _text(value, name=name)
    if _SLUG.fullmatch(result) is None:
        raise ExogenousRegistryError(f"{name} must be a lower-case snake_case id")
    return result


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_policy(payload: Mapping[str, Any]) -> None:
    cutoff = _mapping(payload["cutoff"], name="cutoff")
    _exact_keys(cutoff, _CUTOFF_KEYS, name="cutoff")
    expected_cutoff = {
        "timezone": "Europe/Paris",
        "local_time": "08:00",
        "delivery_horizon": "day_ahead",
        "physical_actual_latest_day_offset": -2,
    }
    if dict(cutoff) != expected_cutoff:
        raise ExogenousRegistryError(
            "cutoff must be exactly D-1 08:00 Europe/Paris and physical "
            "actuals may end no later than D-2"
        )

    policy = _mapping(payload["policy"], name="policy")
    _exact_keys(policy, _POLICY_KEYS, name="policy")
    expected_policy = {
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
    if dict(policy) != expected_policy:
        raise ExogenousRegistryError(
            "policy must remain offline-only, fail-closed, non-training, "
            "non-production, PIT-only, gap-fill-free, numeric-only, and "
            "Storm/MKOnline-free"
        )


def _validate_scientific_basis(
    raw: Any,
    *,
    family_name: str,
) -> None:
    basis = _sequence(raw, name=f"{family_name}.scientific_basis")
    if not basis:
        raise ExogenousRegistryError(
            f"{family_name}.scientific_basis must cite a primary source"
        )
    seen: set[str] = set()
    for position, raw_item in enumerate(basis):
        name = f"{family_name}.scientific_basis[{position}]"
        item = _mapping(raw_item, name=name)
        _exact_keys(item, _BASIS_KEYS, name=name)
        url = _text(item["url"], name=f"{name}.url")
        _text(item["claim"], name=f"{name}.claim")
        if url not in APPROVED_PRIMARY_SOURCES:
            raise ExogenousRegistryError(
                f"{name}.url is not in the reviewed primary-source allowlist: {url}"
            )
        if url in seen:
            raise ExogenousRegistryError(f"{name}.url is duplicated")
        seen.add(url)


def _validate_operational_source(raw: Any, *, signal_name: str) -> Mapping[str, Any]:
    source = _mapping(raw, name=f"{signal_name}.operational_source")
    _exact_keys(source, _SOURCE_KEYS, name=f"{signal_name}.operational_source")
    for key in sorted(_SOURCE_KEYS):
        _text(source[key], name=f"{signal_name}.operational_source.{key}")
    lineage = " ".join(str(source[key]) for key in sorted(_SOURCE_KEYS)).casefold()
    forbidden = [token for token in FORBIDDEN_INPUT_TOKENS if token in lineage]
    if forbidden:
        raise ExogenousRegistryError(
            f"{signal_name}.operational_source contains forbidden input tokens: "
            f"{forbidden}"
        )
    return source


def _validate_feature_contract(raw: Any, *, signal_name: str) -> Mapping[str, Any]:
    feature = _mapping(raw, name=f"{signal_name}.feature_contract")
    _exact_keys(feature, _FEATURE_KEYS, name=f"{signal_name}.feature_contract")
    kind = _text(feature["kind"], name=f"{signal_name}.feature_contract.kind")
    if kind not in FEATURE_KINDS:
        raise ExogenousRegistryError(
            f"{signal_name}.feature_contract.kind={kind!r} is unsupported"
        )
    outputs = _sequence(
        feature["output_columns"],
        name=f"{signal_name}.feature_contract.output_columns",
    )
    if not outputs:
        raise ExogenousRegistryError(
            f"{signal_name}.feature_contract.output_columns must not be empty"
        )
    output_names = [
        _slug(value, name=f"{signal_name}.feature_contract.output_columns")
        for value in outputs
    ]
    if len(output_names) != len(set(output_names)):
        raise ExogenousRegistryError(
            f"{signal_name}.feature_contract.output_columns contains duplicates"
        )
    forbidden = [
        output
        for output in output_names
        if any(token in output.casefold() for token in FORBIDDEN_INPUT_TOKENS)
    ]
    if forbidden:
        raise ExogenousRegistryError(
            f"{signal_name}.feature_contract contains forbidden outputs: {forbidden}"
        )
    if not _bool(
        feature["numeric_only"],
        name=f"{signal_name}.feature_contract.numeric_only",
    ):
        raise ExogenousRegistryError(f"{signal_name}: only numeric features are allowed")
    if _bool(
        feature["same_day_uncleared_price_allowed"],
        name=f"{signal_name}.feature_contract.same_day_uncleared_price_allowed",
    ):
        raise ExogenousRegistryError(
            f"{signal_name}: same-day uncleared prices are forbidden at 08:00"
        )
    uses_actuals = _bool(
        feature["uses_physical_actuals"],
        name=f"{signal_name}.feature_contract.uses_physical_actuals",
    )
    actual_offset = feature["physical_actual_latest_day_offset"]
    if uses_actuals:
        if isinstance(actual_offset, bool) or not isinstance(actual_offset, int):
            raise ExogenousRegistryError(
                f"{signal_name}: physical actual lag must be an integer"
            )
        if actual_offset > -2:
            raise ExogenousRegistryError(
                f"{signal_name}: physical actuals must stop at D-2 or earlier"
            )
    elif actual_offset is not None:
        raise ExogenousRegistryError(
            f"{signal_name}: physical_actual_latest_day_offset must be null "
            "when no physical actual is used"
        )
    return feature


def _validate_gates(raw: Any, *, signal_name: str) -> tuple[dict[str, str], list[str]]:
    gates = _mapping(raw, name=f"{signal_name}.gates")
    _exact_keys(gates, set(PROOF_GATES), name=f"{signal_name}.gates")
    statuses: dict[str, str] = {}
    blocked: list[str] = []
    for gate_name in PROOF_GATES:
        name = f"{signal_name}.gates.{gate_name}"
        gate = _mapping(gates[gate_name], name=name)
        _exact_keys(gate, _GATE_KEYS, name=name)
        status = _text(gate["status"], name=f"{name}.status")
        if status not in GATE_STATUSES:
            raise ExogenousRegistryError(
                f"{name}.status={status!r}; expected {sorted(GATE_STATUSES)}"
            )
        refs = _sequence(gate["evidence_refs"], name=f"{name}.evidence_refs")
        evidence_refs = [
            _text(ref, name=f"{name}.evidence_refs") for ref in refs
        ]
        note = _text(gate["note"], name=f"{name}.note")
        if status == "passed" and not evidence_refs:
            raise ExogenousRegistryError(
                f"{name}: a passed proof gate requires at least one evidence ref"
            )
        if status == "passed":
            if gate_name in {
                "availability_d_minus_1_0800",
                "point_in_time_vintages",
                "zone_coverage",
            } and not any(_ARTIFACT_EVIDENCE.fullmatch(ref) for ref in evidence_refs):
                raise ExogenousRegistryError(
                    f"{name}: operational proof requires an immutable "
                    "artifact:<uri>#sha256=<64 hex> evidence ref"
                )
            if gate_name == "license_internal_research" and not any(
                ref.startswith("license:") and len(ref) > len("license:")
                for ref in evidence_refs
            ):
                raise ExogenousRegistryError(
                    f"{name}: licence proof requires a license:<reference> ref"
                )
            if gate_name in {"causal_lineage", "numeric_schema"} and not any(
                ref.startswith("contract:") and len(ref) > len("contract:")
                for ref in evidence_refs
            ):
                raise ExogenousRegistryError(
                    f"{name}: proof requires a contract:<reference> ref"
                )
        if status == "blocked":
            blocked.append(f"{gate_name}: {note}")
        statuses[gate_name] = status
    return statuses, blocked


def _validate_phase_gates(
    raw: Any,
    *,
    signal_name: str,
) -> tuple[dict[str, str], list[str]]:
    gates = _mapping(raw, name=f"{signal_name}.phase_gates")
    _exact_keys(gates, set(EXPERIMENT_PHASES), name=f"{signal_name}.phase_gates")
    statuses: dict[str, str] = {}
    blocked: list[str] = []
    previous_blocked = False
    for phase in EXPERIMENT_PHASES:
        name = f"{signal_name}.phase_gates.{phase}"
        gate = _mapping(gates[phase], name=name)
        _exact_keys(gate, _PHASE_GATE_KEYS, name=name)
        status = _text(gate["status"], name=f"{name}.status")
        note = _text(gate["note"], name=f"{name}.note")
        if status not in PHASE_GATE_STATUSES:
            raise ExogenousRegistryError(
                f"{name}.status={status!r}; expected "
                f"{sorted(PHASE_GATE_STATUSES)}"
            )
        if previous_blocked and status in {"open", "passed"}:
            raise ExogenousRegistryError(
                f"{name} cannot be {status!r} while an earlier phase is blocked"
            )
        if status == "blocked":
            previous_blocked = True
            blocked.append(f"phase {phase}: {note}")
        statuses[phase] = status
    return statuses, blocked


def _validate_signal(
    raw: Any,
    *,
    family_id: str,
    family_priority: str,
    position: int,
) -> ExogenousSignalAudit:
    name = f"families.{family_id}.signals[{position}]"
    signal = _mapping(raw, name=name)
    _exact_keys(signal, _SIGNAL_KEYS, name=name)
    signal_id = _slug(signal["id"], name=f"{name}.id")
    title = _text(signal["title"], name=f"{name}.title")
    identity = f"{signal_id} {title}".casefold()
    identity_forbidden = [
        token for token in FORBIDDEN_INPUT_TOKENS if token in identity
    ]
    if identity_forbidden:
        raise ExogenousRegistryError(
            f"{name} identity contains forbidden input tokens: "
            f"{identity_forbidden}"
        )

    zones_raw = _sequence(signal["zones"], name=f"{name}.zones")
    zones = tuple(_text(zone, name=f"{name}.zones").upper() for zone in zones_raw)
    if not zones or len(zones) != len(set(zones)):
        raise ExogenousRegistryError(f"{name}.zones must be non-empty and unique")
    unknown_zones = sorted(set(zones) - SUPPORTED_ZONES)
    if unknown_zones:
        raise ExogenousRegistryError(f"{name}.zones unsupported: {unknown_zones}")

    phase = _text(signal["phase"], name=f"{name}.phase")
    if phase not in EXPERIMENT_PHASES:
        raise ExogenousRegistryError(
            f"{name}.phase={phase!r}; expected {list(EXPERIMENT_PHASES)}"
        )
    status = _text(signal["status"], name=f"{name}.status")
    if status not in SIGNAL_STATUSES:
        raise ExogenousRegistryError(
            f"{name}.status={status!r}; expected {sorted(SIGNAL_STATUSES)}"
        )

    _validate_operational_source(signal["operational_source"], signal_name=name)
    _validate_feature_contract(signal["feature_contract"], signal_name=name)
    proof_statuses, proof_blockers = _validate_gates(
        signal["gates"], signal_name=name
    )
    phase_statuses, _all_phase_blockers = _validate_phase_gates(
        signal["phase_gates"], signal_name=name
    )
    declared_blockers = tuple(
        _text(item, name=f"{name}.blockers")
        for item in _sequence(signal["blockers"], name=f"{name}.blockers")
    )

    current_phase_index = EXPERIMENT_PHASES.index(phase)
    earlier_phases = EXPERIMENT_PHASES[:current_phase_index]
    later_phases = EXPERIMENT_PHASES[current_phase_index + 1 :]
    if any(phase_statuses[item] != "passed" for item in earlier_phases):
        raise ExogenousRegistryError(
            f"{name}: every phase before {phase} must be passed"
        )
    if any(phase_statuses[item] != "blocked" for item in later_phases):
        raise ExogenousRegistryError(
            f"{name}: every phase after {phase} must remain blocked"
        )
    proof_complete = all(value == "passed" for value in proof_statuses.values())
    current_phase_open = phase_statuses[phase] in {"open", "passed"}
    if not proof_complete and current_phase_open:
        raise ExogenousRegistryError(
            f"{name}: phase {phase} cannot open before every proof gate passes"
        )
    ready = (
        status == "ready_for_offline_screen"
        and proof_complete
        and current_phase_open
        and phase != "SHADOW"
    )
    active_phase_blockers = [
        f"phase {phase_name}: "
        f"{signal['phase_gates'][phase_name]['note']}"
        for phase_name in EXPERIMENT_PHASES[: current_phase_index + 1]
        if phase_statuses[phase_name] == "blocked"
    ]
    derived_blockers = tuple(proof_blockers + active_phase_blockers)

    if status == "ready_for_offline_screen" and not ready:
        raise ExogenousRegistryError(
            f"{name}: ready_for_offline_screen requires every proof gate passed "
            "and the declared offline phase open"
        )
    if ready and declared_blockers:
        raise ExogenousRegistryError(f"{name}: a screenable signal cannot have blockers")
    if not ready and not declared_blockers:
        raise ExogenousRegistryError(
            f"{name}: every non-screenable signal must declare at least one blocker"
        )
    if status == "blocked" and not derived_blockers:
        raise ExogenousRegistryError(
            f"{name}: blocked status requires a blocked proof or phase gate"
        )

    return ExogenousSignalAudit(
        signal_id=signal_id,
        family_id=family_id,
        zones=zones,
        priority=family_priority,
        phase=phase,
        status=status,
        ready_for_offline_screen=ready,
        blockers=declared_blockers + derived_blockers,
        proof_gates=dict(proof_statuses),
        phase_gates=dict(phase_statuses),
        declaration_sha256=_canonical_sha256(signal),
    )


def audit_exogenous_research_registry(
    payload: Mapping[str, Any],
) -> ExogenousRegistryAudit:
    """Validate one registry mapping and return its deterministic audit.

    Structural contradictions raise :class:`ExogenousRegistryError`.  Missing
    evidence does not: it produces a blocked signal that cannot be screened.
    """

    root = _mapping(payload, name="registry")
    _exact_keys(root, _ROOT_KEYS, name="registry")
    raw_version = root["schema_version"]
    if isinstance(raw_version, bool) or raw_version != SCHEMA_VERSION:
        raise ExogenousRegistryError(
            f"schema_version={raw_version!r}; expected {SCHEMA_VERSION}"
        )
    registry_id = _slug(root["registry_id"], name="registry.registry_id")
    _text(root["purpose"], name="registry.purpose")
    _validate_policy(root)

    families_raw = _sequence(root["families"], name="registry.families")
    audits: list[ExogenousSignalAudit] = []
    family_ids: set[str] = set()
    signal_ids: set[str] = set()
    for position, raw_family in enumerate(families_raw):
        name = f"registry.families[{position}]"
        family = _mapping(raw_family, name=name)
        _exact_keys(family, _FAMILY_KEYS, name=name)
        family_id = _slug(family["id"], name=f"{name}.id")
        if family_id in family_ids:
            raise ExogenousRegistryError(f"duplicate family id: {family_id}")
        family_ids.add(family_id)
        _text(family["title"], name=f"{name}.title")
        priority = _text(family["priority"], name=f"{name}.priority")
        if priority not in PRIORITIES:
            raise ExogenousRegistryError(
                f"{name}.priority={priority!r}; expected {sorted(PRIORITIES)}"
            )
        _validate_scientific_basis(
            family["scientific_basis"], family_name=name
        )
        signals = _sequence(family["signals"], name=f"{name}.signals")
        if not signals:
            raise ExogenousRegistryError(f"{name}.signals must not be empty")
        for signal_position, raw_signal in enumerate(signals):
            audit = _validate_signal(
                raw_signal,
                family_id=family_id,
                family_priority=priority,
                position=signal_position,
            )
            if audit.signal_id in signal_ids:
                raise ExogenousRegistryError(
                    f"duplicate signal id: {audit.signal_id}"
                )
            signal_ids.add(audit.signal_id)
            audits.append(audit)

    missing_families = sorted(EXPECTED_FAMILIES - family_ids)
    unknown_families = sorted(family_ids - EXPECTED_FAMILIES)
    if missing_families or unknown_families:
        raise ExogenousRegistryError(
            "research families must match the reviewed roadmap; "
            f"missing={missing_families}, unknown={unknown_families}"
        )
    if not audits:
        raise ExogenousRegistryError("registry must declare at least one signal")

    return ExogenousRegistryAudit(
        registry_id=registry_id,
        registry_sha256=_canonical_sha256(root),
        signals=tuple(audits),
    )


def load_exogenous_research_registry(
    path: str | Path,
) -> tuple[dict[str, Any], ExogenousRegistryAudit]:
    """Load and audit a YAML registry without accessing any data provider."""

    registry_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExogenousRegistryError(
            f"cannot read exogenous research registry {registry_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ExogenousRegistryError(f"{registry_path}: YAML mapping expected")
    copied = dict(payload)
    return copied, audit_exogenous_research_registry(copied)


__all__ = [
    "APPROVED_PRIMARY_SOURCES",
    "EXPECTED_FAMILIES",
    "ExogenousRegistryAudit",
    "ExogenousRegistryError",
    "ExogenousSignalAudit",
    "audit_exogenous_research_registry",
    "load_exogenous_research_registry",
]
