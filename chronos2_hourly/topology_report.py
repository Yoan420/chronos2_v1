"""Audited HTML reports for PriceFM-inspired topology experiments.

This module is deliberately reporting-only.  It reuses
``write_hourly_html_report`` for the detailed charts, then adds a compact
audit banner sourced from ``topology_evaluation.json``.  It never reads a
PriceFM model, never changes a production configuration and refuses every
source or destination below ``runs/live``.

The implementation is clean-room: PriceFM supplies the methodological idea
(topology-constrained spatial context), but no PriceFM code, weight or dataset
is bundled or loaded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote
from uuid import uuid4

from chronos2_hourly.reporting import write_hourly_html_report


SCHEMA_VERSION = 1
EVALUATION_FILENAME = "topology_evaluation.json"
ANNUAL_METRICS_FILENAME = "annual_metrics.json"
ANNUAL_POLICY_FILENAME = "annual_policy_seal.json"
ANNUAL_STRATEGY_FILENAME = "annual_strategy_hourly.csv.gz"
ANNUAL_STRATEGY_SEAL_FILENAME = "annual_strategy_seal.json"
ANNUAL_REPORT_TYPE = "sealed_annual_365_strategy"
ANNUAL_EXPECTED_DAYS = 365
ANNUAL_EXPECTED_HOURS = 8760
ANNUAL_STRATEGIES = (
    "sequential_governed_strategy",
    "causal_formal_shadow_strategy",
)
SUPPORTED_ZONES = ("FR", "DE", "BE", "NL", "ES")
SUPPORTED_VARIANTS = ("autonomous", "mkonline_blend")
BLEND_ZONES = frozenset({"FR", "NL"})
FORMAL_STAGES = ("b1", "b2", "final")
ZONE_TIMEZONES = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}
VARIANT_MODELS = {
    "autonomous": ("topology_autonomous", "residual_corrected"),
    "mkonline_blend": ("topology_mkonline_blend", "mkonline_blend"),
}
PRICEFM_PAPER_URL = "https://arxiv.org/abs/2508.04875"
PRICEFM_REPOSITORY_URL = "https://github.com/runyao-yu/PriceFM"
PRICEFM_MODEL_URL = "https://huggingface.co/RunyaoYu/PriceFM"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CRITICAL_NAMES = frozenset(
    {
        EVALUATION_FILENAME,
        "artifact_checksums.json",
        "backtest_hourly_oof.csv.gz",
        "metrics_hourly.json",
        "run_manifest.json",
        "statistics_history_audit.json",
        "statistics_history_hourly.csv.gz",
    }
)
_CRITICAL_INPUT_NAMES = frozenset(
    {
        "aligned_inputs.csv.gz",
        "input_coverage.csv",
        "input_manifest.csv",
        "model_covariates_with_future.csv.gz",
    }
)


class TopologyReportError(ValueError):
    """Raised when a topology report would be unsafe or misleading."""


@dataclass(frozen=True)
class BlendWeights:
    """Candidate and current production weights for one blend evaluation."""

    topology_autonomous: float | None
    mkonline_primary: float | None
    previous_autonomous: float | None
    previous_mkonline_primary: float | None
    grid_step: float | None
    fit_method: str | None
    valid: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class TopologyReportRecord:
    """Normalized, display-ready view of one topology evaluation variant."""

    experiment_id: str
    zone: str
    timezone: str
    variant: str
    selected_radius: int | None
    neighbours: tuple[str, ...]
    candidate_model: str
    baseline_model: str
    candidate_mae: float | None
    baseline_mae: float | None
    gain_eur_mwh: float | None
    daily_win_rate: float | None
    gate_passes: bool | None
    gate_reasons: tuple[str, ...]
    pit_min_coverage: float | None
    pit_coverage: tuple[tuple[str, float], ...]
    feature_schema_sha256: str | None
    model_hyperparameters_sha256: str | None
    blend_weights: BlendWeights | None
    decision_stage: str | None
    opened_stages: tuple[str, ...]
    promotion_decision: str | None
    promoted: bool | None
    sequential_decision_complete: bool | None
    complete: bool
    completeness_issues: tuple[str, ...]
    status: str
    fallback_message: str
    raw_evaluation: Mapping[str, Any]
    variant_payload: Mapping[str, Any]


@dataclass(frozen=True)
class TopologyReportArtifact:
    """One generated detailed report and the record used to render it."""

    path: Path
    evaluation_path: Path
    record: TopologyReportRecord
    sha256: str


@dataclass(frozen=True)
class TopologyAnnualStrategyRecord:
    """Display-ready metrics for one sealed mixed annual strategy."""

    name: str
    candidate_mae: float
    baseline_mae: float
    gain_eur_mwh: float
    active_hours: int
    active_days: int
    active_coverage: float
    wins_all_days: int
    ties_all_days: int
    losses_all_days: int
    win_rate_all_days: float
    tie_rate_all_days: float
    loss_rate_all_days: float
    wins_active_days: int
    ties_active_days: int
    losses_active_days: int
    win_rate_active_days: float | None


@dataclass(frozen=True)
class TopologyAnnualComparatorRecord:
    """One reporting-only annual comparator, never a topology input."""

    available: bool
    model: str | None
    mae: float | None
    paired_hours: int | None
    pairing_coverage: float | None
    gain_vs_governed_eur_mwh: float | None
    gain_vs_baseline_eur_mwh: float | None
    reason: str | None


@dataclass(frozen=True)
class TopologyAnnualReportRecord:
    """Strict normalized view of one 365-day annual strategy bundle."""

    zone: str
    timezone: str
    start_local_day: str
    end_local_day: str
    start_utc: str
    end_utc: str
    n_days: int
    n_hours: int
    baseline_model: str
    baseline_mae: float
    sequential_governed_strategy: TopologyAnnualStrategyRecord
    causal_formal_shadow_strategy: TopologyAnnualStrategyRecord
    storm: TopologyAnnualComparatorRecord
    mkonline_production_reference: TopologyAnnualComparatorRecord
    topology_blend_candidate_available: bool
    pure_topology_annual_available: bool
    pure_topology_annual_reason: str
    out_of_sample_scope: str
    annual_metrics_path: Path
    strategy_path: Path
    policy_path: Path
    strategy_seal_path: Path
    raw_metrics: Mapping[str, Any]


@dataclass(frozen=True)
class TopologyAnnualReportArtifact:
    """One generated annual HTML report and its authenticated record."""

    path: Path
    record: TopologyAnnualReportRecord
    sha256: str


@dataclass(frozen=True)
class _SequentialDecision:
    """Normalized state of the fail-closed sequential protocol."""

    modern: bool
    decision_stage: str | None
    metric_stage: str | None
    opened_stages: tuple[str, ...]
    promotion_decision: str | None
    promoted: bool | None
    sequential_complete: bool | None
    gate_passes: bool | None
    gate_reasons: tuple[str, ...]
    issues: tuple[str, ...]


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TopologyReportError(f"{name} doit etre un objet JSON.")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _required_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TopologyReportError(f"{name} doit etre une chaine non vide.")
    return value.strip()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(payload: Any) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _project_root(value: str | Path | None) -> Path:
    return (
        Path(value).expanduser().resolve()
        if value is not None
        else Path(__file__).resolve().parents[1]
    )


def _contains_runs_live(path: Path) -> bool:
    parts = tuple(part.casefold() for part in path.resolve().parts)
    return any(
        parts[index : index + 2] == ("runs", "live")
        for index in range(max(0, len(parts) - 1))
    )


def _assert_experiment_path(path: Path, *, project_root: Path, name: str) -> None:
    resolved = path.expanduser().resolve()
    if _contains_runs_live(resolved):
        raise TopologyReportError(f"{name} ne peut jamais etre sous runs/live.")
    if resolved.is_relative_to(project_root):
        experiments = (project_root / "runs" / "experiments").resolve()
        if not resolved.is_relative_to(experiments):
            raise TopologyReportError(
                f"{name} doit rester sous {experiments} pour un projet reel."
            )


def _variant_section(
    payload: Mapping[str, Any], variant: str
) -> Mapping[str, Any]:
    variants = _optional_mapping(payload.get("variants"))
    if variant in variants:
        return _mapping(variants[variant], name=f"variants.{variant}")
    if variant == "autonomous":
        autonomous = payload.get("autonomous")
        return (
            _mapping(autonomous, name="autonomous")
            if autonomous is not None
            else payload
        )
    blend = _optional_mapping(payload.get("blend_compatibility"))
    if blend.get("supported") is False:
        raise TopologyReportError("Le blend MKOnline est declare non supporte.")
    for key in ("topology_then_mkonline", "evaluation", "mkonline_blend"):
        if key in blend:
            return _mapping(blend[key], name=f"blend_compatibility.{key}")
    raise TopologyReportError(
        "Evaluation mkonline_blend absente de blend_compatibility."
    )


def _segment(
    payload: Mapping[str, Any], section: str, name: str
) -> Mapping[str, Any]:
    values = _optional_mapping(payload.get(section))
    return _optional_mapping(values.get(name))


def _metric_value(
    metrics: Mapping[str, Any], gate: Mapping[str, Any], *keys: str
) -> float | None:
    containers = (
        metrics,
        _optional_mapping(metrics.get("overall")),
        gate,
        _optional_mapping(gate.get("overall")),
    )
    for container in containers:
        for key in keys:
            value = _finite(container.get(key))
            if value is not None:
                return value
    return None


def _reason_list(gate: Mapping[str, Any]) -> tuple[str, ...]:
    reasons = gate.get("reasons", ())
    if isinstance(reasons, str):
        reasons = (reasons,)
    if not isinstance(reasons, Sequence) or isinstance(reasons, (bytes, str)):
        return ()
    return tuple(str(item).strip() for item in reasons if str(item).strip())


def _coverage_leaves(
    value: Any, *, prefix: str = ""
) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    if isinstance(value, Mapping):
        if "coverage" in value:
            coverage = _finite(value.get("coverage"))
            if coverage is not None:
                result.append((prefix or "coverage", coverage))
        else:
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                result.extend(_coverage_leaves(child, prefix=child_prefix))
    else:
        coverage = _finite(value)
        if coverage is not None:
            result.append((prefix or "coverage", coverage))
    return result


def _neighbours(payload: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = (
        payload.get("topology_context"),
        payload.get("context_metadata"),
        payload,
    )
    for candidate in candidates:
        context = _optional_mapping(candidate)
        raw = context.get("neighbours")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values = tuple(
                dict.fromkeys(
                    str(item).strip().upper()
                    for item in raw
                    if str(item).strip()
                )
            )
            return values
    return ()


def _declared_sha(
    payload: Mapping[str, Any], name: str
) -> str | None:
    candidates = (
        payload.get(name),
        _optional_mapping(payload.get("source_hashes")).get(name),
        _optional_mapping(payload.get("topology_context")).get(name),
    )
    for value in candidates:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if _SHA256_RE.fullmatch(normalized):
                return normalized
    return None


def _weight_block(payload: Mapping[str, Any]) -> BlendWeights:
    candidate = _optional_mapping(payload.get("candidate_weights"))
    previous = _optional_mapping(payload.get("previous_production_weights"))
    candidate_topology = _finite(candidate.get("topology_autonomous"))
    candidate_mkonline = _finite(candidate.get("mkonline_primary"))
    previous_autonomous = _finite(previous.get("autonomous"))
    previous_mkonline = _finite(previous.get("mkonline_primary"))
    grid_step = _finite(payload.get("weight_grid_step"))
    fit_method = (
        str(payload.get("weight_fit_method")).strip()
        if payload.get("weight_fit_method") is not None
        else None
    )
    issues: list[str] = []
    pairs = (
        ("poids candidats", candidate_topology, candidate_mkonline),
        ("poids de production", previous_autonomous, previous_mkonline),
    )
    for label, left, right in pairs:
        if left is None or right is None:
            issues.append(f"{label} absents ou non finis")
            continue
        if left < 0.0 or right < 0.0:
            issues.append(f"{label} negatifs")
        if not math.isclose(left + right, 1.0, rel_tol=0.0, abs_tol=1e-9):
            issues.append(f"{label} dont la somme differe de 1")
    if grid_step not in {0.025, 0.05}:
        issues.append("weight_grid_step doit valoir 0.025 ou 0.05")
    if (
        candidate_mkonline is not None
        and grid_step in {0.025, 0.05}
        and not math.isclose(
            candidate_mkonline / grid_step,
            round(candidate_mkonline / grid_step),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        issues.append("poids MKOnline candidat hors grille")
    if fit_method != "constrained_l1_grid":
        issues.append("weight_fit_method doit etre constrained_l1_grid")
    if payload.get("fitted_on") != "A":
        issues.append("les poids doivent etre ajustes sur A")
    if payload.get("final_used_for_tuning") is not False:
        issues.append("le final ne doit jamais regler les poids")
    return BlendWeights(
        topology_autonomous=candidate_topology,
        mkonline_primary=candidate_mkonline,
        previous_autonomous=previous_autonomous,
        previous_mkonline_primary=previous_mkonline,
        grid_step=grid_step,
        fit_method=fit_method,
        valid=not issues,
        issues=tuple(issues),
    )


def _normalize_stage(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().casefold()
    if normalized == "a_identity":
        return "A_identity"
    return normalized


def _normalize_opened_stages(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    normalized: list[str] = []
    for item in value:
        stage = _normalize_stage(item)
        if stage is None or stage == "A_identity":
            return None
        normalized.append(stage)
    return tuple(normalized)


def _sequential_decision(
    variant_payload: Mapping[str, Any], *, variant: str
) -> _SequentialDecision:
    """Validate the modern sequential decision without opening spared stages.

    Older, already-published sidecars only declared a final gate.  They remain
    readable through a narrow legacy branch.  As soon as any modern decision
    field is present, all modern invariants apply and partial evidence is
    reported as incomplete rather than silently interpreted as legacy.
    """

    decision_fields = {
        "opened_stages",
        "decision_stage",
        "promotion_decision",
        "sequential_decision_complete",
        "promoted",
    }
    modern = any(key in variant_payload for key in decision_fields)
    gates = _optional_mapping(variant_payload.get("gates"))
    metrics = _optional_mapping(variant_payload.get("metrics"))

    if not modern:
        final_gate = _optional_mapping(gates.get("final"))
        final_passes = _optional_bool(final_gate.get("passes"))
        opened = tuple(
            stage
            for stage in FORMAL_STAGES
            if stage in gates or stage in metrics
        )
        return _SequentialDecision(
            modern=False,
            decision_stage="final",
            metric_stage="final",
            opened_stages=opened,
            promotion_decision=(
                "promoted"
                if final_passes is True
                else (
                    "fallback_identity"
                    if variant == "autonomous" and final_passes is False
                    else (
                        "fallback_production_blend"
                        if final_passes is False
                        else None
                    )
                )
            ),
            promoted=final_passes,
            sequential_complete=None,
            gate_passes=final_passes,
            gate_reasons=_reason_list(final_gate),
            issues=(),
        )

    issues: list[str] = []
    opened = _normalize_opened_stages(variant_payload.get("opened_stages"))
    if opened is None:
        issues.append("opened_stages absent ou invalide")
        opened = ()
    elif len(set(opened)) != len(opened):
        issues.append("opened_stages contient des doublons")

    decision_stage = _normalize_stage(variant_payload.get("decision_stage"))
    if decision_stage is None:
        issues.append("decision_stage absent ou invalide")
    promotion_decision = variant_payload.get("promotion_decision")
    if not isinstance(promotion_decision, str) or not promotion_decision.strip():
        issues.append("promotion_decision absent ou invalide")
        normalized_promotion: str | None = None
    else:
        normalized_promotion = promotion_decision.strip()
    promoted = _optional_bool(variant_payload.get("promoted"))
    if promoted is None:
        issues.append("promoted doit etre un booleen")
    sequential_complete = _optional_bool(
        variant_payload.get("sequential_decision_complete")
    )
    if sequential_complete is not True:
        issues.append("sequential_decision_complete doit etre true")

    anchors = ("a", "development") if variant == "autonomous" else ("a",)
    valid_prefix = opened[: len(anchors)] == anchors
    formal_opened = opened[len(anchors) :] if valid_prefix else ()
    if not valid_prefix:
        issues.append(
            "opened_stages doit commencer par " + ", ".join(anchors)
        )
    formal_prefix_valid = formal_opened == FORMAL_STAGES[: len(formal_opened)]
    if not formal_prefix_valid:
        issues.append("les stages formels ouverts ne forment pas un prefixe causal")

    selected_arm = variant_payload.get("selected_arm")
    identity_decision = variant == "autonomous" and decision_stage == "A_identity"
    metric_stage: str | None = None
    gate_passes: bool | None = None
    gate_reasons: tuple[str, ...] = ()

    if identity_decision:
        if selected_arm != "identity":
            issues.append("A_identity exige selected_arm=identity")
        if opened != anchors:
            issues.append("A_identity ne doit ouvrir aucun holdout formel")
        if promoted is not False:
            issues.append("A_identity exige promoted=false")
        if normalized_promotion != "fallback_identity":
            issues.append("A_identity exige promotion_decision=fallback_identity")
        present_formal = [
            stage
            for stage in FORMAL_STAGES
            if stage in gates or stage in metrics
        ]
        if present_formal:
            issues.append(
                "A_identity doit laisser absents les stages formels: "
                + ", ".join(present_formal)
            )
        gate_passes = False
    else:
        if variant == "autonomous" and selected_arm == "identity":
            issues.append("selected_arm=identity exige decision_stage=A_identity")
        if not formal_opened or not formal_prefix_valid:
            issues.append("aucune sequence formelle valide ouverte pour la decision")
        else:
            last_opened = formal_opened[-1]
            metric_stage = last_opened
            if decision_stage != last_opened:
                issues.append("decision_stage doit etre le dernier stage formel ouvert")
            for stage in formal_opened[:-1]:
                previous_passes = _optional_bool(
                    _optional_mapping(gates.get(stage)).get("passes")
                )
                if previous_passes is not True:
                    issues.append(f"la gate precedente {stage} doit etre explicitement pass")
            decision_gate = _optional_mapping(gates.get(last_opened))
            gate_passes = _optional_bool(decision_gate.get("passes"))
            gate_reasons = _reason_list(decision_gate)

            if promoted is True:
                if normalized_promotion != "promoted":
                    issues.append("une promotion exige promotion_decision=promoted")
                if formal_opened != FORMAL_STAGES:
                    issues.append("une promotion exige l'ouverture de B1, B2 et final")
                if decision_stage != "final":
                    issues.append("une promotion exige decision_stage=final")
                if gate_passes is not True:
                    issues.append("une promotion exige une gate final explicitement pass")
            elif promoted is False:
                expected_fallback = (
                    "fallback_identity"
                    if variant == "autonomous"
                    else "fallback_production_blend"
                )
                if normalized_promotion != expected_fallback:
                    issues.append(
                        "un rejet exige promotion_decision=" + expected_fallback
                    )
                if gate_passes is not False:
                    issues.append(
                        f"la derniere gate ouverte {last_opened} doit etre explicitement fail"
                    )
                later_stages = FORMAL_STAGES[
                    FORMAL_STAGES.index(last_opened) + 1 :
                ]
                exposed_later = [
                    stage
                    for stage in later_stages
                    if stage in opened or stage in gates or stage in metrics
                ]
                if exposed_later:
                    issues.append(
                        "les stages posterieurs au rejet doivent etre absents: "
                        + ", ".join(exposed_later)
                    )

    if decision_stage == "A_identity" and variant != "autonomous":
        issues.append("A_identity est reserve a la variante autonome")
    if decision_stage not in {"A_identity", *FORMAL_STAGES}:
        issues.append("decision_stage doit valoir A_identity, b1, b2 ou final")

    spared = variant_payload.get("unopened_holdouts_spared")
    if spared is not None and opened:
        normalized_spared = _normalize_opened_stages(spared)
        expected_spared = tuple(stage for stage in FORMAL_STAGES if stage not in opened)
        if normalized_spared != expected_spared:
            issues.append("unopened_holdouts_spared contredit opened_stages")

    return _SequentialDecision(
        modern=True,
        decision_stage=decision_stage,
        metric_stage=metric_stage,
        opened_stages=opened,
        promotion_decision=normalized_promotion,
        promoted=promoted,
        sequential_complete=sequential_complete,
        gate_passes=gate_passes,
        gate_reasons=gate_reasons,
        issues=tuple(issues),
    )


def load_topology_evaluation(
    path: str | Path,
    *,
    variant: str = "autonomous",
) -> TopologyReportRecord:
    """Validate and normalize one topology evaluation sidecar.

    Missing audit evidence does not get invented: it produces an ``incomplete``
    record.  Contradictory numeric claims (for example a wrong MAE gain) fail
    closed because rendering them would be misleading.
    """

    evaluation_path = Path(path).expanduser().resolve()
    if not evaluation_path.is_file():
        raise FileNotFoundError(evaluation_path)
    try:
        raw = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyReportError(
            f"Sidecar topology illisible: {evaluation_path}"
        ) from exc
    payload = _mapping(raw, name=EVALUATION_FILENAME)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise TopologyReportError(
            f"schema_version={payload.get('schema_version')!r}; "
            f"attendu={SCHEMA_VERSION}."
        )
    if variant not in SUPPORTED_VARIANTS:
        raise TopologyReportError(
            f"variant={variant!r}; choix={SUPPORTED_VARIANTS}."
        )
    experiment_id = _required_text(
        payload.get("experiment_id"), name="experiment_id"
    )
    zone = _required_text(payload.get("zone"), name="zone").upper()
    if zone not in SUPPORTED_ZONES:
        raise TopologyReportError(f"Zone non supportee: {zone}.")
    if variant == "mkonline_blend" and zone not in BLEND_ZONES:
        raise TopologyReportError(
            f"Le blend MKOnline est reserve a {sorted(BLEND_ZONES)}."
        )
    local_timezone = _required_text(payload.get("timezone"), name="timezone")
    if local_timezone != ZONE_TIMEZONES[zone]:
        raise TopologyReportError(
            f"Timezone {local_timezone!r} incoherente pour {zone}."
        )

    variant_payload = _variant_section(payload, variant)
    models = VARIANT_MODELS[variant]
    candidate_model = str(
        variant_payload.get("candidate_model")
        or payload.get("candidate_model")
        or models[0]
    ).strip()
    baseline_model = str(
        variant_payload.get("baseline_model")
        or payload.get("baseline_model")
        or models[1]
    ).strip()
    selected_value = variant_payload.get(
        "selected_radius", payload.get("selected_radius")
    )
    selected_radius = (
        int(selected_value)
        if isinstance(selected_value, int) and not isinstance(selected_value, bool)
        else None
    )
    if selected_radius not in {None, 0, 1}:
        raise TopologyReportError("selected_radius doit valoir 0 ou 1.")

    sequential = _sequential_decision(variant_payload, variant=variant)
    decision_metrics = (
        _segment(variant_payload, "metrics", sequential.metric_stage)
        if sequential.metric_stage is not None
        else {}
    )
    decision_gate = (
        _segment(variant_payload, "gates", sequential.metric_stage)
        if sequential.metric_stage is not None
        else {}
    )
    candidate_mae = _metric_value(
        decision_metrics, decision_gate, "candidate_mae", "topology_mae"
    )
    baseline_mae = _metric_value(
        decision_metrics, decision_gate, "baseline_mae", "identity_mae"
    )
    declared_gain = _metric_value(
        decision_metrics, decision_gate, "gain_eur_mwh", "mae_gain_eur_mwh"
    )
    computed_gain = (
        baseline_mae - candidate_mae
        if baseline_mae is not None and candidate_mae is not None
        else None
    )
    if (
        declared_gain is not None
        and computed_gain is not None
        and not math.isclose(
            declared_gain, computed_gain, rel_tol=0.0, abs_tol=1e-8
        )
    ):
        raise TopologyReportError(
            "Le gain MAE declare contredit baseline_mae - candidate_mae."
        )
    gain = computed_gain if computed_gain is not None else declared_gain
    win_rate = _metric_value(
        decision_metrics, decision_gate, "daily_win_rate", "win_rate"
    )
    if win_rate is not None and not 0.0 <= win_rate <= 1.0:
        raise TopologyReportError("daily_win_rate doit etre entre 0 et 1.")
    gate_passes = sequential.gate_passes
    gate_reasons = sequential.gate_reasons

    raw_coverage = variant_payload.get(
        "pit_coverage", payload.get("pit_coverage", {})
    )
    coverage = sorted(_coverage_leaves(raw_coverage))
    invalid_coverage = [
        name for name, value in coverage if not 0.0 <= value <= 1.0
    ]
    if invalid_coverage:
        raise TopologyReportError(
            "Couverture PIT hors [0,1]: " + ", ".join(invalid_coverage)
        )
    min_coverage = min((value for _, value in coverage), default=None)
    neighbours = _neighbours(variant_payload) or _neighbours(payload)
    feature_hash = _declared_sha(payload, "feature_schema_sha256")
    hyperparameter_hash = _declared_sha(
        payload, "model_hyperparameters_sha256"
    )
    blend_weights = (
        _weight_block(variant_payload)
        if variant == "mkonline_blend"
        else None
    )

    issues: list[str] = list(sequential.issues)
    required_values: tuple[tuple[str, Any], ...] = (
        ("selected_radius", selected_radius),
        ("couverture PIT", min_coverage),
        ("feature_schema_sha256", feature_hash),
        ("model_hyperparameters_sha256", hyperparameter_hash),
    )
    if sequential.decision_stage != "A_identity":
        required_values += (
            ("MAE candidat du stage de decision", candidate_mae),
            ("MAE reference du stage de decision", baseline_mae),
            ("gate du stage de decision", gate_passes),
        )
    for label, value in required_values:
        if value is None:
            issues.append(f"{label} absent")
    if selected_radius == 1 and not neighbours:
        issues.append("voisins absents pour delta=1")
    if payload.get("storm_loaded_after_candidate_freeze") is not True:
        issues.append("Storm non prouve apres gel candidat")
    if payload.get("storm_used_as_prediction_input") is not False:
        issues.append("absence de Storm dans les inputs non prouvee")
    mkonline_flag = payload.get(
        "mkonline_used_by_topology",
        payload.get(
            "mkonline_used_as_topology_input",
            payload.get("mkonline_used_as_prediction_input"),
        ),
    )
    if mkonline_flag is not False:
        issues.append("autonomie du correcteur vis-a-vis de MKOnline non prouvee")
    if variant == "mkonline_blend":
        assert blend_weights is not None
        issues.extend(blend_weights.issues)
        required_true = (
            "autonomous_gates_passed_before_mkonline_load",
            "recipe_frozen_before_final",
        )
        for key in required_true:
            if variant_payload.get(key) is not True:
                issues.append(f"{key} doit etre true")
        dependency_hash = variant_payload.get("dependency_manifest_sha256")
        if not (
            isinstance(dependency_hash, str)
            and _SHA256_RE.fullmatch(dependency_hash.strip().lower())
        ):
            issues.append("dependency_manifest_sha256 absent ou invalide")
        if gate_passes is False:
            expected_recommendation = (
                "production_mkonline_blend"
                if sequential.modern
                else "autonomous"
            )
            if variant_payload.get("recommended_variant") != expected_recommendation:
                issues.append(
                    "un blend refuse doit recommander "
                    + expected_recommendation
                )
            if variant_payload.get("production_weights_unchanged") is not True:
                issues.append(
                    "un blend refuse doit conserver les poids de production"
                )

    if sequential.modern and variant == "autonomous":
        development = _optional_mapping(variant_payload.get("development"))
        if development.get("used_for_formal_gate") is not False:
            issues.append("development.used_for_formal_gate doit etre false")

    complete = not issues
    if not complete:
        status = "incomplete"
    elif sequential.promoted is True:
        status = "promoted"
    else:
        status = "fallback"
    if sequential.promoted is True:
        fallback_message = (
            "Candidate topologique retenu par les gates hors echantillon."
            if variant == "autonomous"
            else "Nouveaux poids du blend retenus par sa gate distincte."
        )
    elif variant == "autonomous":
        fallback_message = (
            "Fallback identite : le forecast autonome existant reste inchange."
        )
    else:
        fallback_message = (
            "Fallback vers le blend actuel : les poids MKOnline de production "
            "restent inchanges."
        )

    return TopologyReportRecord(
        experiment_id=experiment_id,
        zone=zone,
        timezone=local_timezone,
        variant=variant,
        selected_radius=selected_radius,
        neighbours=neighbours,
        candidate_model=candidate_model,
        baseline_model=baseline_model,
        candidate_mae=candidate_mae,
        baseline_mae=baseline_mae,
        gain_eur_mwh=gain,
        daily_win_rate=win_rate,
        gate_passes=gate_passes,
        gate_reasons=gate_reasons,
        pit_min_coverage=min_coverage,
        pit_coverage=tuple(coverage),
        feature_schema_sha256=feature_hash,
        model_hyperparameters_sha256=hyperparameter_hash,
        blend_weights=blend_weights,
        decision_stage=sequential.decision_stage,
        opened_stages=sequential.opened_stages,
        promotion_decision=sequential.promotion_decision,
        promoted=sequential.promoted,
        sequential_decision_complete=sequential.sequential_complete,
        complete=complete,
        completeness_issues=tuple(issues),
        status=status,
        fallback_message=fallback_message,
        raw_evaluation=payload,
        variant_payload=variant_payload,
    )


def _critical_files(run_dir: Path) -> tuple[Path, ...]:
    files: set[Path] = set()
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        if (
            path.name in _CRITICAL_NAMES
            or path.name.startswith("forecast_hourly_")
            or path.name.startswith("evaluation_")
        ):
            files.add(path)
    inputs = run_dir / "inputs"
    if inputs.is_dir():
        for name in _CRITICAL_INPUT_NAMES:
            path = inputs / name
            if path.is_file():
                files.add(path)
    return tuple(sorted(files, key=lambda item: str(item).casefold()))


def _snapshot(run_dir: Path) -> dict[Path, str]:
    return {path: _sha256(path) for path in _critical_files(run_dir)}


def _verify_unchanged(before: Mapping[Path, str], run_dir: Path) -> None:
    after = _snapshot(run_dir)
    if before != after:
        changed = sorted(
            str(path)
            for path in set(before).union(after)
            if before.get(path) != after.get(path)
        )
        raise TopologyReportError(
            "Le rendu HTML a modifie des artefacts sources: "
            + ", ".join(changed)
        )


def _fmt(value: float | None, *, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def _status_label(status: str) -> str:
    return {
        "promoted": "Gates validees",
        "fallback": "Fallback actif",
        "incomplete": "Audit incomplet",
        "missing": "Rapport absent",
    }.get(status, status)


def _audit_rows(record: TopologyReportRecord) -> str:
    rows: list[tuple[str, str]] = []
    development = _optional_mapping(record.variant_payload.get("development"))
    development_metrics = _optional_mapping(development.get("metrics"))
    if development_metrics:
        candidate = _metric_value(
            development_metrics, {}, "candidate_mae", "topology_mae"
        )
        baseline = _metric_value(
            development_metrics, {}, "baseline_mae", "identity_mae"
        )
        gain = (
            baseline - candidate
            if baseline is not None and candidate is not None
            else _metric_value(
                development_metrics,
                {},
                "gain_eur_mwh",
                "mae_gain_eur_mwh",
            )
        )
        rows.append(
            (
                "Developpement (diagnostic)",
                "<td>" + html.escape(_fmt(candidate)) + "</td>"
                "<td>" + html.escape(_fmt(baseline)) + "</td>"
                "<td>" + html.escape(_fmt(gain)) + "</td>"
                "<td>DIAGNOSTIC</td>",
            )
        )
    for split in ("A", "B1", "B2", "final"):
        metrics = _segment(record.variant_payload, "metrics", split.lower())
        gate = _segment(record.variant_payload, "gates", split.lower())
        if not metrics and not gate:
            metrics = _segment(record.variant_payload, "metrics", split)
            gate = _segment(record.variant_payload, "gates", split)
        candidate = _metric_value(metrics, gate, "candidate_mae", "topology_mae")
        baseline = _metric_value(metrics, gate, "baseline_mae", "identity_mae")
        gain = (
            baseline - candidate
            if baseline is not None and candidate is not None
            else _metric_value(metrics, gate, "gain_eur_mwh", "mae_gain_eur_mwh")
        )
        passes = _optional_bool(gate.get("passes"))
        if metrics or gate:
            gate_text = "N/A" if passes is None else ("OK" if passes else "NON")
            rows.append(
                (
                    split,
                    "<td>" + html.escape(_fmt(candidate)) + "</td>"
                    "<td>" + html.escape(_fmt(baseline)) + "</td>"
                    "<td>" + html.escape(_fmt(gain)) + "</td>"
                    "<td>" + gate_text + "</td>",
                )
            )
    return "".join(
        f"<tr><th>{html.escape(name)}</th>{cells}</tr>"
        for name, cells in rows
    ) or '<tr><td colspan="5">Aucun detail par split.</td></tr>'


def _banner_html(record: TopologyReportRecord) -> str:
    status_label = _status_label(record.status)
    neighbours = ", ".join(record.neighbours) or "Aucun declare"
    radius = "N/A" if record.selected_radius is None else str(record.selected_radius)
    decision_stage = record.decision_stage or "N/A"
    opened_stages = ", ".join(record.opened_stages) or "Aucun"
    pit_rows = "".join(
        "<tr><td>"
        + html.escape(name)
        + "</td><td>"
        + html.escape(_pct(value))
        + "</td></tr>"
        for name, value in record.pit_coverage
    ) or '<tr><td colspan="2">Couverture absente.</td></tr>'
    gate_reasons = "".join(
        f"<li>{html.escape(reason)}</li>" for reason in record.gate_reasons
    ) or "<li>Aucune raison detaillee.</li>"
    completeness = "".join(
        f"<li>{html.escape(issue)}</li>"
        for issue in record.completeness_issues
    ) or "<li>Audit complet.</li>"
    blend = ""
    if record.blend_weights is not None:
        weights = record.blend_weights
        blend = f"""
        <div class="topology-audit-panel">
          <h3>Poids du blend</h3>
          <table><thead><tr><th>Recette</th><th>Autonome</th><th>MKOnline</th></tr></thead>
          <tbody>
            <tr><td>Candidate</td><td>{html.escape(_fmt(weights.topology_autonomous))}</td><td>{html.escape(_fmt(weights.mkonline_primary))}</td></tr>
            <tr><td>Production actuelle</td><td>{html.escape(_fmt(weights.previous_autonomous))}</td><td>{html.escape(_fmt(weights.previous_mkonline_primary))}</td></tr>
          </tbody></table>
          <p class="topology-audit-small">Ajustement {html.escape(weights.fit_method or 'N/A')} sur A · pas de reglage sur final · pas de remplacement automatique en cas de refus.</p>
        </div>"""
    embedded = {
        "schema_version": SCHEMA_VERSION,
        "record": {
            "experiment_id": record.experiment_id,
            "zone": record.zone,
            "variant": record.variant,
            "selected_radius": record.selected_radius,
            "neighbours": record.neighbours,
            "candidate_mae": record.candidate_mae,
            "baseline_mae": record.baseline_mae,
            "gain_eur_mwh": record.gain_eur_mwh,
            "daily_win_rate": record.daily_win_rate,
            "gate_passes": record.gate_passes,
            "decision_stage": record.decision_stage,
            "opened_stages": record.opened_stages,
            "promotion_decision": record.promotion_decision,
            "promoted": record.promoted,
            "sequential_decision_complete": record.sequential_decision_complete,
            "status": record.status,
            "complete": record.complete,
        },
        "source": record.raw_evaluation,
        "provenance": {
            "relationship": "methodological_inspiration_only_clean_room",
            "pricefm_code_reused": False,
            "pricefm_weights_reused": False,
            "pricefm_data_reused": False,
            "paper": PRICEFM_PAPER_URL,
            "repository": PRICEFM_REPOSITORY_URL,
            "model_card": PRICEFM_MODEL_URL,
        },
    }
    return f"""
<section id="topology-audit" class="topology-audit topology-status-{html.escape(record.status)}" data-report-section="topology-audit" data-topology-status="{html.escape(record.status)}">
  <div class="topology-audit-heading">
    <div><p class="topology-audit-eyebrow">EXPERIENCE PRICEFM-INSPIRED · CLEAN-ROOM</p>
    <h2>Contexte topologique audite — {html.escape(record.zone)}</h2></div>
    <span class="topology-audit-status">{html.escape(status_label)}</span>
  </div>
  <p>{html.escape(record.fallback_message)}</p>
  <div class="topology-audit-kpis">
    <div><span>Variante</span><strong>{html.escape(record.variant)}</strong></div>
    <div><span>Rayon topologique δ</span><strong>{html.escape(radius)}</strong></div>
    <div><span>Stage de decision</span><strong>{html.escape(decision_stage)}</strong></div>
    <div><span>Stages ouverts</span><strong>{html.escape(opened_stages)}</strong></div>
    <div><span>MAE candidate</span><strong>{html.escape(_fmt(record.candidate_mae))} €/MWh</strong></div>
    <div><span>MAE reference</span><strong>{html.escape(_fmt(record.baseline_mae))} €/MWh</strong></div>
    <div><span>Gain MAE</span><strong>{html.escape(_fmt(record.gain_eur_mwh))} €/MWh</strong></div>
    <div><span>Win rate journalier</span><strong>{html.escape(_pct(record.daily_win_rate))}</strong></div>
    <div><span>Couverture PIT minimale</span><strong>{html.escape(_pct(record.pit_min_coverage))}</strong></div>
    <div><span>Voisins réellement injectés</span><strong>{html.escape(neighbours)}</strong></div>
  </div>
  <div class="topology-audit-grid">
    <div class="topology-audit-panel"><h3>Gates par split</h3>
      <table><thead><tr><th>Split</th><th>MAE cand.</th><th>MAE ref.</th><th>Gain</th><th>Gate</th></tr></thead>
      <tbody>{_audit_rows(record)}</tbody></table>
      <h4>Motifs de la gate de decision</h4><ul>{gate_reasons}</ul>
    </div>
    <div class="topology-audit-panel"><h3>Disponibilité point-in-time</h3>
      <table><thead><tr><th>Série / zone</th><th>Couverture</th></tr></thead><tbody>{pit_rows}</tbody></table>
      <p class="topology-audit-small">Storm est evaluation-only et doit etre chargé après gel du candidat. MKOnline n'entre jamais dans le correcteur topologique autonome.</p>
    </div>
    {blend}
    <div class="topology-audit-panel"><h3>Audit et reproductibilité</h3>
      <p><b>Schema features :</b> <code>{html.escape(record.feature_schema_sha256 or 'N/A')}</code></p>
      <p><b>Hyperparamètres :</b> <code>{html.escape(record.model_hyperparameters_sha256 or 'N/A')}</code></p>
      <h4>Complétude</h4><ul>{completeness}</ul>
    </div>
  </div>
  <details><summary>Source scientifique, licence et périmètre</summary>
    <p>Cette intégration reprend uniquement l'idée scientifique d'un contexte spatial contraint par la topologie. Aucun code, poids ou jeu de données PriceFM n'est copié, chargé ou redistribué.</p>
    <p><a href="{PRICEFM_PAPER_URL}">Préprint PriceFM</a> · <a href="{PRICEFM_REPOSITORY_URL}">Dépôt des auteurs</a> · <a href="{PRICEFM_MODEL_URL}">Fiche modèle Hugging Face</a>.</p>
    <p>Le dépôt GitHub ne publie pas de fichier LICENSE à sa racine au moment de l'audit. Toute réutilisation future d'un artefact externe exige donc une revue de licence séparée.</p>
  </details>
  <script type="application/json" id="topology-report-data">{_safe_json(embedded)}</script>
</section>
"""


_BANNER_CSS = """
.topology-audit{border:2px solid #2563eb;background:linear-gradient(135deg,color-mix(in srgb,#2563eb 10%,var(--card,#fff)),var(--card,#fff));}
.topology-audit-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;}
.topology-audit-heading h2{margin:2px 0 0;}.topology-audit-eyebrow{margin:0;color:#1d4ed8;font-size:12px;font-weight:800;letter-spacing:.08em;}
.topology-audit-status{border-radius:999px;padding:7px 12px;font-size:12px;font-weight:800;background:#dbeafe;color:#1e3a8a;white-space:nowrap;}
.topology-status-promoted{border-color:#059669}.topology-status-promoted .topology-audit-status{background:#d1fae5;color:#065f46;}
.topology-status-fallback{border-color:#d97706}.topology-status-fallback .topology-audit-status{background:#fef3c7;color:#92400e;}
.topology-status-incomplete{border-color:#dc2626}.topology-status-incomplete .topology-audit-status{background:#fee2e2;color:#991b1b;}
.topology-audit-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px;margin:18px 0;}
.topology-audit-kpis>div{border:1px solid color-mix(in srgb,#60a5fa 45%,var(--border,#dfe5ea));background:var(--card,#fff);border-radius:10px;padding:12px;}.topology-audit-kpis span{display:block;color:var(--muted,#64748b);font-size:12px;margin-bottom:5px;}.topology-audit-kpis strong{font-size:15px;}
.topology-audit-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px;}.topology-audit-panel{border:1px solid color-mix(in srgb,#60a5fa 35%,var(--border,#dfe5ea));background:var(--card,#fff);border-radius:12px;padding:14px;overflow:auto;}.topology-audit-panel h3{margin-top:0;}
.topology-audit table{width:100%;border-collapse:collapse;font-size:12px}.topology-audit th,.topology-audit td{border-bottom:1px solid var(--border,#e2e8f0);padding:7px;text-align:left;}.topology-audit code{font-size:10px;overflow-wrap:anywhere}.topology-audit-small{font-size:12px;color:var(--muted,#64748b);}.topology-audit details{margin-top:16px;padding-top:12px;border-top:1px solid color-mix(in srgb,#60a5fa 45%,var(--border,#dfe5ea))}.topology-audit summary{cursor:pointer;font-weight:700;}
@media(max-width:720px){.topology-audit-heading{display:block}.topology-audit-status{display:inline-block;margin-top:10px}.topology-audit-grid{grid-template-columns:1fr}}
"""


def _decorate(source: str, record: TopologyReportRecord) -> str:
    if "</style>" not in source or "<main>" not in source:
        raise TopologyReportError(
            "Le HTML courant ne contient pas les points d'insertion attendus."
        )
    result = source.replace("</style>", _BANNER_CSS + "</style>", 1)
    return result.replace("<main>", "<main>" + _banner_html(record), 1)


def write_topology_html_report(
    run_dir: str | Path,
    *,
    variant: str = "autonomous",
    evaluation_path: str | Path | None = None,
    output_path: str | Path | None = None,
    history_hours: int = 168,
    overwrite: bool = False,
    project_root: str | Path | None = None,
    report_writer: Callable[..., Path] = write_hourly_html_report,
) -> TopologyReportArtifact:
    """Render the existing detailed report plus the topology audit banner."""

    root = _project_root(project_root)
    directory = Path(run_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    _assert_experiment_path(directory, project_root=root, name="run_dir")
    audit_path = (
        Path(evaluation_path).expanduser().resolve()
        if evaluation_path is not None
        else directory / EVALUATION_FILENAME
    )
    _assert_experiment_path(
        audit_path, project_root=root, name="evaluation_path"
    )
    record = load_topology_evaluation(audit_path, variant=variant)
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else directory
        / "reports"
        / f"topology_{record.zone.lower()}_{variant}.html"
    )
    _assert_experiment_path(
        destination, project_root=root, name="output_path"
    )
    if destination.suffix.casefold() != ".html":
        raise TopologyReportError("output_path doit etre un fichier .html.")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.stem}.rendering-{uuid4().hex}.html"
    )
    before = _snapshot(directory)
    title = (
        f"Chronos-2 {record.zone} — contexte topologique "
        + ("autonome" if variant == "autonomous" else "puis blend MKOnline")
    )
    try:
        rendered_path = Path(
            report_writer(
                directory,
                output_path=temporary,
                title=title,
                native_model=record.candidate_model,
                baseline_model=record.baseline_model,
                zone=record.zone,
                timezone=record.timezone,
                history_hours=int(history_hours),
            )
        ).resolve()
        if rendered_path != temporary.resolve() or not temporary.is_file():
            raise TopologyReportError(
                "Le renderer n'a pas produit le fichier temporaire attendu."
            )
        _verify_unchanged(before, directory)
        decorated = _decorate(temporary.read_text(encoding="utf-8"), record)
        temporary.write_text(decorated, encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return TopologyReportArtifact(
        path=destination,
        evaluation_path=audit_path,
        record=record,
        sha256=_sha256(destination),
    )


def _relative_href(target: Path, base: Path) -> str:
    relative = Path(os.path.relpath(target, start=base))
    return "/".join(quote(part) for part in relative.parts)


def _index_row(
    artifact: TopologyReportArtifact | None,
    *,
    zone: str,
    variant: str,
    index_directory: Path,
) -> str:
    if artifact is None:
        values = {
            "status": "missing",
            "radius": "N/A",
            "candidate": "N/A",
            "baseline": "N/A",
            "gain": "N/A",
            "win": "N/A",
            "pit": "N/A",
            "weights": "N/A",
            "fallback": "Rapport non produit.",
            "link": "—",
        }
    else:
        record = artifact.record
        weights = record.blend_weights
        weight_text = "N/A"
        if weights is not None:
            weight_text = (
                f"cand. { _fmt(weights.topology_autonomous) } / "
                f"{ _fmt(weights.mkonline_primary) } · prod. "
                f"{ _fmt(weights.previous_autonomous) } / "
                f"{ _fmt(weights.previous_mkonline_primary) }"
            )
        href = _relative_href(artifact.path, index_directory)
        values = {
            "status": record.status,
            "radius": (
                "N/A"
                if record.selected_radius is None
                else str(record.selected_radius)
            ),
            "candidate": _fmt(record.candidate_mae),
            "baseline": _fmt(record.baseline_mae),
            "gain": _fmt(record.gain_eur_mwh),
            "win": _pct(record.daily_win_rate),
            "pit": _pct(record.pit_min_coverage),
            "weights": weight_text,
            "fallback": record.fallback_message,
            "link": f'<a href="{html.escape(href)}">Ouvrir</a>',
        }
    return f"""
<tr data-zone="{html.escape(zone)}" data-variant="{html.escape(variant)}" data-status="{html.escape(values['status'])}">
  <td><b>{html.escape(zone)}</b></td><td>{html.escape(variant)}</td>
  <td><span class="status status-{html.escape(values['status'])}">{html.escape(_status_label(values['status']))}</span></td>
  <td>{html.escape(values['radius'])}</td><td>{html.escape(values['candidate'])}</td>
  <td>{html.escape(values['baseline'])}</td><td>{html.escape(values['gain'])}</td>
  <td>{html.escape(values['win'])}</td><td>{html.escape(values['pit'])}</td>
  <td>{html.escape(values['weights'])}</td><td>{html.escape(values['fallback'])}</td>
  <td>{values['link']}</td>
</tr>"""


def write_topology_report_index(
    artifacts: Sequence[TopologyReportArtifact],
    *,
    output_path: str | Path,
    selected_zones: Sequence[str] | None = None,
    include_blend_slots: bool = True,
    overwrite: bool = False,
    project_root: str | Path | None = None,
) -> Path:
    """Write one offline, filterable index across zones and both variants."""

    root = _project_root(project_root)
    destination = Path(output_path).expanduser().resolve()
    _assert_experiment_path(
        destination, project_root=root, name="index output_path"
    )
    if destination.suffix.casefold() != ".html":
        raise TopologyReportError("L'index doit etre un fichier .html.")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    by_key: dict[tuple[str, str], TopologyReportArtifact] = {}
    for artifact in artifacts:
        key = (artifact.record.zone, artifact.record.variant)
        if key in by_key:
            raise TopologyReportError(f"Rapport duplique pour {key}.")
        if not artifact.path.is_file():
            raise FileNotFoundError(artifact.path)
        by_key[key] = artifact
    if selected_zones is None:
        present = {zone for zone, _ in by_key}
        zones = tuple(zone for zone in SUPPORTED_ZONES if zone in present)
    else:
        normalized = tuple(str(zone).strip().upper() for zone in selected_zones)
        if len(set(normalized)) != len(normalized):
            raise TopologyReportError("selected_zones contient des doublons.")
        if any(zone not in SUPPORTED_ZONES for zone in normalized):
            raise TopologyReportError("selected_zones contient une zone invalide.")
        zones = normalized
    if not zones:
        raise TopologyReportError("Aucune zone a consolider.")
    slots = [(zone, "autonomous") for zone in zones]
    if include_blend_slots:
        slots.extend(
            (zone, "mkonline_blend") for zone in zones if zone in BLEND_ZONES
        )
    rows = "".join(
        _index_row(
            by_key.get((zone, variant)),
            zone=zone,
            variant=variant,
            index_directory=destination.parent,
        )
        for zone, variant in slots
    )
    generated = datetime.now(timezone.utc).isoformat()
    document = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chronos-2 · rapports topologiques</title>
<style>
:root{{--bg:#f4f6f8;--card:#fff;--text:#172033;--muted:#64748b;--line:#dbe2ea;--blue:#159de4}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Arial,sans-serif}}header{{background:#111827;color:#fff;padding:28px 5vw}}header h1{{margin:0 0 8px}}main{{max-width:1500px;margin:24px auto;padding:0 24px}}section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:20px;box-shadow:0 5px 18px rgba(15,23,42,.05)}}.controls{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}}select,input{{border:1px solid #cbd5e1;border-radius:8px;padding:9px;background:#fff}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:1180px}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;font-size:13px;vertical-align:top}}th{{background:#f8fafc;position:sticky;top:0}}.status{{display:inline-block;border-radius:999px;padding:5px 8px;font-size:11px;font-weight:800;white-space:nowrap}}.status-promoted{{background:#d1fae5;color:#065f46}}.status-fallback{{background:#fef3c7;color:#92400e}}.status-incomplete,.status-missing{{background:#fee2e2;color:#991b1b}}a{{color:#0369a1}}.muted{{color:var(--muted)}}
</style></head><body>
<header><h1>Chronos-2 · intégration topologique PriceFM-inspired</h1><p>Index consolidé autonome et blend MKOnline · généré {html.escape(generated)}</p></header>
<main><section><h2>Rapports détaillés</h2>
<div class="controls"><label>Variante <select id="variant"><option value="all">Toutes</option><option value="autonomous">Autonome</option><option value="mkonline_blend">Blend MKOnline</option></select></label><label>Statut <select id="status"><option value="all">Tous</option><option value="promoted">Validé</option><option value="fallback">Fallback</option><option value="incomplete">Incomplet</option><option value="missing">Absent</option></select></label><label>Recherche <input id="search" type="search" placeholder="Pays, statut, modèle…"></label></div>
<div class="table"><table id="reports"><thead><tr><th>Pays</th><th>Variante</th><th>Statut</th><th>δ</th><th>MAE candidate</th><th>MAE référence</th><th>Gain</th><th>Win rate</th><th>PIT min.</th><th>Poids cand. / prod.</th><th>Décision / fallback</th><th>Rapport</th></tr></thead><tbody>{rows}</tbody></table></div>
</section><section><h2>Périmètre scientifique et licence</h2><p>Implémentation clean-room inspirée du masque topologique de PriceFM. Aucun code, poids ou dataset PriceFM n'est utilisé. Les résultats de l'article ne sont pas transposés comme promesse de gain local : chaque pays et chaque variante passe ses propres gates hors échantillon.</p><p><a href="{PRICEFM_PAPER_URL}">Papier</a> · <a href="{PRICEFM_REPOSITORY_URL}">GitHub</a> · <a href="{PRICEFM_MODEL_URL}">Hugging Face</a></p></section></main>
<script>const v=document.getElementById('variant'),s=document.getElementById('status'),q=document.getElementById('search'),rows=[...document.querySelectorAll('#reports tbody tr')];function filter(){{const needle=q.value.trim().toLowerCase();for(const row of rows){{row.hidden=!((v.value==='all'||row.dataset.variant===v.value)&&(s.value==='all'||row.dataset.status===s.value)&&(!needle||row.textContent.toLowerCase().includes(needle)))}}}}[v,s,q].forEach(x=>x.addEventListener(x===q?'input':'change',filter));</script>
</body></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.stem}-{uuid4().hex}.tmp"
    try:
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _required_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TopologyReportError(
            f"{name} doit etre un entier superieur ou egal a {minimum}."
        )
    return value


def _required_finite(value: Any, *, name: str) -> float:
    result = _finite(value)
    if result is None:
        raise TopologyReportError(f"{name} doit etre un nombre fini.")
    return result


def _required_false(value: Any, *, name: str) -> None:
    if value is not False:
        raise TopologyReportError(f"{name} doit etre explicitement false.")


def _required_true(value: Any, *, name: str) -> None:
    if value is not True:
        raise TopologyReportError(f"{name} doit etre explicitement true.")


def _read_required_json(path: Path, *, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyReportError(f"{name} est illisible: {path}.") from exc
    return _mapping(payload, name=name)


def _verify_annual_checksums(
    zone_dir: Path,
    *,
    required_names: Sequence[str],
) -> Mapping[str, Any]:
    manifest_path = zone_dir / "artifact_checksums.json"
    manifest = _read_required_json(
        manifest_path,
        name="artifact_checksums.json annuel",
    )
    if manifest.get("algorithm") != "sha256":
        raise TopologyReportError("Le manifeste annuel doit utiliser sha256.")
    raw_entries = manifest.get("artifacts")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise TopologyReportError("artifact_checksums.json.artifacts est invalide.")
    entries: dict[str, Mapping[str, Any]] = {}
    for position, raw in enumerate(raw_entries):
        entry = _mapping(raw, name=f"artifact_checksums.artifacts[{position}]")
        relative = _required_text(entry.get("path"), name="artifact path")
        normalized = relative.replace("\\", "/")
        if normalized in entries:
            raise TopologyReportError(f"Checksum annuel duplique: {normalized}.")
        candidate = (zone_dir / Path(normalized)).resolve()
        if not candidate.is_relative_to(zone_dir):
            raise TopologyReportError(
                f"Le checksum annuel sort du dossier de zone: {normalized}."
            )
        entries[normalized] = entry
    for name in required_names:
        normalized = name.replace("\\", "/")
        if normalized not in entries:
            raise TopologyReportError(
                f"Artefact annuel non authentifie: {normalized}."
            )
        entry = entries[normalized]
        path = (zone_dir / Path(normalized)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        declared = entry.get("sha256")
        if not isinstance(declared, str) or not _SHA256_RE.fullmatch(
            declared.strip().lower()
        ):
            raise TopologyReportError(f"SHA256 annuel invalide: {normalized}.")
        if _sha256(path) != declared.strip().lower():
            raise TopologyReportError(f"Checksum annuel divergent: {normalized}.")
        declared_size = entry.get("size_bytes")
        if declared_size is not None and _required_int(
            declared_size,
            name=f"size_bytes {normalized}",
        ) != path.stat().st_size:
            raise TopologyReportError(f"Taille annuelle divergente: {normalized}.")
    return manifest


def _annual_daily_outcomes(
    payload: Mapping[str, Any],
    *,
    strategy_name: str,
    active_days: int,
) -> tuple[int, int, int, float, float, float, int, int, int, float | None]:
    outcomes = _mapping(
        payload.get("daily_outcomes"),
        name=f"strategies.{strategy_name}.daily_outcomes",
    )

    def parse(
        key: str,
        *,
        expected_days: int,
    ) -> tuple[int, int, int, float, float, float]:
        section = _mapping(
            outcomes.get(key),
            name=f"strategies.{strategy_name}.daily_outcomes.{key}",
        )
        n_days = _required_int(
            section.get("n_days"),
            name=f"{strategy_name}.{key}.n_days",
        )
        if n_days != expected_days:
            raise TopologyReportError(
                f"{strategy_name}.{key} doit couvrir {expected_days} jours."
            )
        wins = _required_int(section.get("wins"), name=f"{strategy_name}.{key}.wins")
        ties = _required_int(section.get("ties"), name=f"{strategy_name}.{key}.ties")
        losses = _required_int(
            section.get("losses"), name=f"{strategy_name}.{key}.losses"
        )
        if wins + ties + losses != n_days:
            raise TopologyReportError(
                f"{strategy_name}.{key}: wins+ties+losses differe de n_days."
            )
        if n_days == 0:
            for metric in ("win_rate", "tie_rate", "loss_rate"):
                if section.get(metric) is not None:
                    raise TopologyReportError(
                        f"{strategy_name}.{key}.{metric} doit etre null sans jour actif."
                    )
            return wins, ties, losses, 0.0, 0.0, 0.0
        values: list[float] = []
        for metric, count in (
            ("win_rate", wins),
            ("tie_rate", ties),
            ("loss_rate", losses),
        ):
            value = _required_finite(
                section.get(metric), name=f"{strategy_name}.{key}.{metric}"
            )
            expected = count / n_days if n_days else 0.0
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
                raise TopologyReportError(
                    f"{strategy_name}.{key}.{metric} contredit les comptes."
                )
            values.append(value)
        return wins, ties, losses, values[0], values[1], values[2]

    all_values = parse("all_days", expected_days=ANNUAL_EXPECTED_DAYS)
    active_values = parse("active_days", expected_days=active_days)
    active_win_rate = active_values[3] if active_days else None
    return (
        all_values[0],
        all_values[1],
        all_values[2],
        all_values[3],
        all_values[4],
        all_values[5],
        active_values[0],
        active_values[1],
        active_values[2],
        active_win_rate,
    )


def _annual_strategy_record(
    name: str,
    payload: Mapping[str, Any],
    *,
    baseline_mae: float,
) -> TopologyAnnualStrategyRecord:
    full = _mapping(
        payload.get("full_period_metrics"),
        name=f"strategies.{name}.full_period_metrics",
    )
    if _required_int(full.get("n_hours"), name=f"{name}.n_hours") != ANNUAL_EXPECTED_HOURS:
        raise TopologyReportError(f"{name} ne couvre pas 8760 heures.")
    if _required_int(full.get("n_days"), name=f"{name}.n_days") != ANNUAL_EXPECTED_DAYS:
        raise TopologyReportError(f"{name} ne couvre pas 365 jours.")
    candidate_mae = _required_finite(
        full.get("candidate_mae"), name=f"{name}.candidate_mae"
    )
    declared_baseline = _required_finite(
        full.get("baseline_mae"), name=f"{name}.baseline_mae"
    )
    gain = _required_finite(full.get("gain_eur_mwh"), name=f"{name}.gain_eur_mwh")
    if not math.isclose(declared_baseline, baseline_mae, rel_tol=0.0, abs_tol=1e-12):
        raise TopologyReportError(f"{name}: baseline_mae contredit la reference annuelle.")
    if not math.isclose(
        gain,
        declared_baseline - candidate_mae,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TopologyReportError(f"{name}: gain MAE contradictoire.")
    active_hours = _required_int(
        payload.get("active_hours"), name=f"{name}.active_hours"
    )
    active_days = _required_int(payload.get("active_days"), name=f"{name}.active_days")
    if active_hours > ANNUAL_EXPECTED_HOURS or active_days > ANNUAL_EXPECTED_DAYS:
        raise TopologyReportError(f"{name}: activite hors fenetre annuelle.")
    active_coverage = _required_finite(
        payload.get("active_coverage"), name=f"{name}.active_coverage"
    )
    if not math.isclose(
        active_coverage,
        active_hours / ANNUAL_EXPECTED_HOURS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TopologyReportError(f"{name}: active_coverage contradictoire.")
    active_metrics = payload.get("active_only_metrics")
    if active_hours == 0:
        if active_metrics is not None:
            raise TopologyReportError(
                f"{name}: active_only_metrics doit etre null sans heure active."
            )
    else:
        active = _mapping(active_metrics, name=f"{name}.active_only_metrics")
        if _required_int(active.get("n_hours"), name=f"{name}.active.n_hours") != active_hours:
            raise TopologyReportError(f"{name}: n_hours actif contradictoire.")
        if _required_int(active.get("n_days"), name=f"{name}.active.n_days") != active_days:
            raise TopologyReportError(f"{name}: n_days actif contradictoire.")
    outcomes = _annual_daily_outcomes(
        payload,
        strategy_name=name,
        active_days=active_days,
    )
    return TopologyAnnualStrategyRecord(
        name=name,
        candidate_mae=candidate_mae,
        baseline_mae=declared_baseline,
        gain_eur_mwh=gain,
        active_hours=active_hours,
        active_days=active_days,
        active_coverage=active_coverage,
        wins_all_days=outcomes[0],
        ties_all_days=outcomes[1],
        losses_all_days=outcomes[2],
        win_rate_all_days=outcomes[3],
        tie_rate_all_days=outcomes[4],
        loss_rate_all_days=outcomes[5],
        wins_active_days=outcomes[6],
        ties_active_days=outcomes[7],
        losses_active_days=outcomes[8],
        win_rate_active_days=outcomes[9],
    )


def _annual_storm_record(
    payload: Any,
    *,
    governed_model: str,
) -> TopologyAnnualComparatorRecord:
    storm = _mapping(payload, name="storm")
    available = storm.get("available")
    if available is False:
        reason = _required_text(storm.get("reason"), name="storm.reason")
        return TopologyAnnualComparatorRecord(
            available=False,
            model=None,
            mae=None,
            paired_hours=None,
            pairing_coverage=None,
            gain_vs_governed_eur_mwh=None,
            gain_vs_baseline_eur_mwh=None,
            reason=reason,
        )
    if available is not True:
        raise TopologyReportError("storm.available doit etre un booleen explicite.")
    _required_true(storm.get("comparator_only"), name="storm.comparator_only")
    for flag in ("used_for_prediction", "used_for_gate", "used_for_selection", "used_for_promotion"):
        _required_false(storm.get(flag), name=f"storm.{flag}")
    metrics = _mapping(storm.get("metrics"), name="storm.metrics")
    candidate_model = _required_text(
        metrics.get("candidate_model"), name="storm.metrics.candidate_model"
    )
    if candidate_model != governed_model:
        raise TopologyReportError("Storm doit etre apparie a la strategie gouvernee.")
    expected = _required_int(
        metrics.get("n_expected_hours"), name="storm.metrics.n_expected_hours"
    )
    if expected != ANNUAL_EXPECTED_HOURS:
        raise TopologyReportError("Storm doit declarer 8760 heures attendues.")
    paired = _required_int(
        metrics.get("n_paired_hours"), name="storm.metrics.n_paired_hours"
    )
    if paired > expected:
        raise TopologyReportError("Storm a plus d'heures appariees qu'attendues.")
    coverage = _required_finite(
        metrics.get("pairing_coverage"), name="storm.metrics.pairing_coverage"
    )
    if not math.isclose(coverage, paired / expected, rel_tol=0.0, abs_tol=1e-12):
        raise TopologyReportError("Couverture Storm contradictoire.")
    return TopologyAnnualComparatorRecord(
        available=True,
        model=_required_text(
            metrics.get("benchmark_model"), name="storm.metrics.benchmark_model"
        ),
        mae=_required_finite(metrics.get("storm_mae"), name="storm.metrics.storm_mae"),
        paired_hours=paired,
        pairing_coverage=coverage,
        gain_vs_governed_eur_mwh=_required_finite(
            metrics.get("candidate_gain_vs_storm_eur_mwh"),
            name="storm.metrics.candidate_gain_vs_storm_eur_mwh",
        ),
        gain_vs_baseline_eur_mwh=_required_finite(
            metrics.get("baseline_gain_vs_storm_eur_mwh"),
            name="storm.metrics.baseline_gain_vs_storm_eur_mwh",
        ),
        reason=None,
    )


def _annual_mkonline_record(payload: Any) -> TopologyAnnualComparatorRecord:
    reference = _mapping(payload, name="mkonline_production_reference")
    available = reference.get("available")
    _required_false(
        reference.get("topology_blend_candidate_available"),
        name="mkonline_production_reference.topology_blend_candidate_available",
    )
    if available is False:
        reason = _required_text(
            reference.get("reason"), name="mkonline_production_reference.reason"
        )
        return TopologyAnnualComparatorRecord(
            available=False,
            model=None,
            mae=None,
            paired_hours=None,
            pairing_coverage=None,
            gain_vs_governed_eur_mwh=None,
            gain_vs_baseline_eur_mwh=None,
            reason=reason,
        )
    if available is not True:
        raise TopologyReportError(
            "mkonline_production_reference.available doit etre explicite."
        )
    _required_true(
        reference.get("comparator_only"),
        name="mkonline_production_reference.comparator_only",
    )
    _required_false(
        reference.get("weights_recomputed"),
        name="mkonline_production_reference.weights_recomputed",
    )
    weights = _mapping(reference.get("weights"), name="mkonline weights")
    autonomous = _required_finite(weights.get("autonomous"), name="mkonline weight autonomous")
    primary = _required_finite(
        weights.get("mkonline_primary"), name="mkonline weight primary"
    )
    if autonomous < 0.0 or primary < 0.0 or not math.isclose(
        autonomous + primary, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise TopologyReportError("Poids MKOnline de production invalides.")
    metrics = _mapping(reference.get("metrics"), name="mkonline metrics")
    if _required_int(metrics.get("n_hours"), name="mkonline n_hours") != ANNUAL_EXPECTED_HOURS:
        raise TopologyReportError("MKOnline ne couvre pas 8760 heures.")
    if _required_int(metrics.get("n_days"), name="mkonline n_days") != ANNUAL_EXPECTED_DAYS:
        raise TopologyReportError("MKOnline ne couvre pas 365 jours.")
    audit = _mapping(reference.get("audit"), name="mkonline audit")
    if _required_int(audit.get("cutoff_violations"), name="mkonline cutoff_violations") != 0:
        raise TopologyReportError("MKOnline contient une violation de cutoff.")
    for key in (
        "recipe_manifest_sha256",
        "dependency_manifest_sha256",
        "forecast_file_sha256",
    ):
        value = audit.get(key)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.strip().lower()):
            raise TopologyReportError(f"mkonline audit.{key} est invalide.")
    return TopologyAnnualComparatorRecord(
        available=True,
        model="production_mkonline_blend",
        mae=_required_finite(metrics.get("mae"), name="mkonline metrics.mae"),
        paired_hours=ANNUAL_EXPECTED_HOURS,
        pairing_coverage=1.0,
        gain_vs_governed_eur_mwh=_required_finite(
            metrics.get("gain_vs_sequential_governed_eur_mwh"),
            name="mkonline gain_vs_sequential_governed",
        ),
        gain_vs_baseline_eur_mwh=_required_finite(
            metrics.get("gain_vs_residual_corrected_eur_mwh"),
            name="mkonline gain_vs_residual_corrected",
        ),
        reason=None,
    )


def _strict_bool_series(series: Any, *, name: str) -> Any:
    import pandas as pd

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.casefold()
    if not bool(normalized.isin({"true", "false"}).all()):
        raise TopologyReportError(f"{name} doit contenir uniquement true/false.")
    return normalized.eq("true")


def _validate_annual_hourly(
    record: TopologyAnnualReportRecord,
) -> None:
    import numpy as np
    import pandas as pd

    required = {
        "delivery_start_utc",
        "forecast_origin_utc",
        "protocol_stage",
        "actual",
        "candidate_available",
        "candidate_formal_oos",
        "governed_topology_active",
        "governed_reason",
        "formal_shadow_topology_active",
        "formal_shadow_reason",
    }
    for model in (
        "residual_corrected",
        "topology_opened_candidate",
        "sequential_governed_strategy",
        "causal_formal_shadow_strategy",
    ):
        required.update(f"{model}__{quantile}" for quantile in ("q10", "q50", "q90"))
    frame = pd.read_csv(record.strategy_path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TopologyReportError(f"Colonnes annuelles absentes: {missing}.")
    if len(frame) != ANNUAL_EXPECTED_HOURS:
        raise TopologyReportError("annual_strategy_hourly doit contenir 8760 lignes.")
    delivery = pd.DatetimeIndex(
        pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if delivery.has_duplicates or not delivery.is_monotonic_increasing:
        raise TopologyReportError("Timeline annuelle dupliquee ou non ordonnee.")
    if not bool((delivery[1:] - delivery[:-1] == pd.Timedelta(hours=1)).all()):
        raise TopologyReportError("Timeline annuelle UTC discontinue.")
    if delivery[0].isoformat() != pd.Timestamp(record.start_utc).tz_convert("UTC").isoformat():
        raise TopologyReportError("start_utc contredit le CSV annuel.")
    if delivery[-1].isoformat() != pd.Timestamp(record.end_utc).tz_convert("UTC").isoformat():
        raise TopologyReportError("end_utc contredit le CSV annuel.")
    local = delivery.tz_convert(record.timezone)
    local_days = pd.Index(local.strftime("%Y-%m-%d"))
    unique_days = tuple(dict.fromkeys(local_days))
    if len(unique_days) != ANNUAL_EXPECTED_DAYS:
        raise TopologyReportError("Le CSV annuel ne couvre pas 365 jours locaux.")
    if unique_days[0] != record.start_local_day or unique_days[-1] != record.end_local_day:
        raise TopologyReportError("Les bornes locales contredisent le CSV annuel.")
    expected_days = pd.date_range(record.start_local_day, periods=365, freq="D").strftime(
        "%Y-%m-%d"
    )
    if tuple(expected_days) != unique_days:
        raise TopologyReportError("Les 365 jours locaux ne sont pas contigus.")
    counts = pd.Series(local_days).value_counts().value_counts().to_dict()
    if counts != {24: 363, 25: 1, 23: 1}:
        raise TopologyReportError(f"Contrat DST annuel inattendu: {counts}.")
    origins = pd.DatetimeIndex(
        pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="raise")
    )
    expected_origins = pd.DatetimeIndex(
        [
            pd.Timestamp(
                (pd.Timestamp(day) - pd.Timedelta(days=1)).date(),
                tz=record.timezone,
            )
            .replace(hour=8)
            .tz_convert("UTC")
            for day in local_days
        ]
    )
    if not np.array_equal(origins.asi8, expected_origins.asi8):
        raise TopologyReportError("forecast_origin_utc annuel viole D-1 08:00 local.")
    for model in (
        "residual_corrected",
        "sequential_governed_strategy",
        "causal_formal_shadow_strategy",
    ):
        values = frame[[f"{model}__q10", f"{model}__q50", f"{model}__q90"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(values.to_numpy(float)).all():
            raise TopologyReportError(f"{model} contient des quantiles non finis.")
        if bool(((values.iloc[:, 0] > values.iloc[:, 1]) | (values.iloc[:, 1] > values.iloc[:, 2])).any()):
            raise TopologyReportError(f"{model} contient des quantiles croises.")
    actual = pd.to_numeric(frame["actual"], errors="coerce")
    if not np.isfinite(actual.to_numpy(float)).all():
        raise TopologyReportError("actual annuel contient des valeurs non finies.")
    candidate_available = _strict_bool_series(
        frame["candidate_available"], name="candidate_available"
    )
    candidate_formal = _strict_bool_series(
        frame["candidate_formal_oos"], name="candidate_formal_oos"
    )
    governed_active = _strict_bool_series(
        frame["governed_topology_active"], name="governed_topology_active"
    )
    shadow_active = _strict_bool_series(
        frame["formal_shadow_topology_active"],
        name="formal_shadow_topology_active",
    )
    stages = frame["protocol_stage"].astype(str).str.strip().str.casefold()
    if not bool(stages.isin({"seed", "a", "development", "b1", "b2", "final"}).all()):
        raise TopologyReportError("protocol_stage annuel invalide.")
    preformal = stages.isin({"seed", "a", "development"})
    if bool(governed_active[preformal].any()) or bool(shadow_active[preformal].any()):
        raise TopologyReportError("Une strategie annuelle active la topologie avant B1.")
    if bool(governed_active[stages.eq("b1")].any()):
        raise TopologyReportError("La strategie gouvernee ne peut pas activer B1 avant sa gate.")
    if bool((governed_active & ~shadow_active).any()):
        raise TopologyReportError("La strategie gouvernee sort du shadow formel disponible.")
    if bool((shadow_active != candidate_formal).any()):
        raise TopologyReportError("Le shadow formel contredit candidate_formal_oos.")
    if bool((candidate_formal & ~candidate_available).any()):
        raise TopologyReportError("Un candidat formel est marque indisponible.")
    if int(governed_active.sum()) != record.sequential_governed_strategy.active_hours:
        raise TopologyReportError("active_hours gouverne contredit le CSV.")
    if int(shadow_active.sum()) != record.causal_formal_shadow_strategy.active_hours:
        raise TopologyReportError("active_hours shadow contredit le CSV.")
    baseline = frame[
        [f"residual_corrected__{quantile}" for quantile in ("q10", "q50", "q90")]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    for model, active in (
        ("sequential_governed_strategy", governed_active),
        ("causal_formal_shadow_strategy", shadow_active),
    ):
        values = frame[
            [f"{model}__{quantile}" for quantile in ("q10", "q50", "q90")]
        ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.array_equal(values[~active.to_numpy()], baseline[~active.to_numpy()]):
            raise TopologyReportError(f"{model}: fallback non identique au baseline.")
    opened = frame[
        [f"topology_opened_candidate__{quantile}" for quantile in ("q10", "q50", "q90")]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    available = candidate_available.to_numpy()
    if not np.isfinite(opened[available]).all() or not np.isnan(opened[~available]).all():
        raise TopologyReportError("Disponibilite du candidat ouvert contradictoire.")


def load_topology_annual_evaluation(
    zone_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> TopologyAnnualReportRecord:
    """Authenticate and normalize one exact 365-day annual strategy bundle."""

    root = _project_root(project_root)
    directory = Path(zone_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    _assert_experiment_path(directory, project_root=root, name="annual zone_dir")
    required_names = (
        ANNUAL_STRATEGY_FILENAME,
        ANNUAL_STRATEGY_SEAL_FILENAME,
        ANNUAL_POLICY_FILENAME,
        ANNUAL_METRICS_FILENAME,
        "run_manifest.json",
    )
    _verify_annual_checksums(directory, required_names=required_names)
    metrics_path = directory / ANNUAL_METRICS_FILENAME
    policy_path = directory / ANNUAL_POLICY_FILENAME
    strategy_path = directory / ANNUAL_STRATEGY_FILENAME
    strategy_seal_path = directory / ANNUAL_STRATEGY_SEAL_FILENAME
    metrics = _read_required_json(metrics_path, name=ANNUAL_METRICS_FILENAME)
    policy = _read_required_json(policy_path, name=ANNUAL_POLICY_FILENAME)
    strategy_seal = _read_required_json(
        strategy_seal_path, name=ANNUAL_STRATEGY_SEAL_FILENAME
    )
    run_manifest = _read_required_json(directory / "run_manifest.json", name="run_manifest.json")
    if metrics.get("schema_version") != SCHEMA_VERSION:
        raise TopologyReportError("schema_version annuel non supporte.")
    if metrics.get("report_type") != ANNUAL_REPORT_TYPE:
        raise TopologyReportError("report_type annuel invalide.")
    zone = _required_text(metrics.get("zone"), name="annual zone").upper()
    if zone not in SUPPORTED_ZONES:
        raise TopologyReportError(f"Zone annuelle non supportee: {zone}.")
    timezone_name = _required_text(metrics.get("timezone"), name="annual timezone")
    if timezone_name != ZONE_TIMEZONES[zone]:
        raise TopologyReportError("Timezone annuelle contraire a la zone.")
    period = _mapping(metrics.get("period"), name="annual period")
    start_local_day = _required_text(period.get("start_local_day"), name="period.start_local_day")
    end_local_day = _required_text(period.get("end_local_day"), name="period.end_local_day")
    start_utc = _required_text(period.get("start_utc"), name="period.start_utc")
    end_utc = _required_text(period.get("end_utc"), name="period.end_utc")
    n_days = _required_int(period.get("n_local_days"), name="period.n_local_days")
    n_hours = _required_int(period.get("n_hours"), name="period.n_hours")
    if n_days != ANNUAL_EXPECTED_DAYS or n_hours != ANNUAL_EXPECTED_HOURS:
        raise TopologyReportError("Le rapport annuel exige exactement 365 jours et 8760 heures.")
    if metrics.get("n_days") is not None and _required_int(
        metrics.get("n_days"), name="annual n_days"
    ) != n_days:
        raise TopologyReportError("annual n_days contredit period.")
    if metrics.get("n_hours") is not None and _required_int(
        metrics.get("n_hours"), name="annual n_hours"
    ) != n_hours:
        raise TopologyReportError("annual n_hours contredit period et doit valoir 8760.")
    expected_scope = {
        "out_of_sample_scope": "mixed_sequential_governed",
        "formal_candidate_scope": "formal_holdouts_only",
        "selection_A_excluded_from_strategies": True,
        "development_excluded_from_strategies": True,
        "identity_fallback_is_not_candidate_prediction": True,
    }
    for key, expected in expected_scope.items():
        if metrics.get(key) != expected:
            raise TopologyReportError(
                f"Portee hors echantillon ambigue: {key} doit valoir {expected!r}."
            )
    for flag in ("no_fit", "no_predict", "no_refit", "no_new_prediction"):
        _required_true(metrics.get(flag), name=f"annual {flag}")
    for flag in ("used_for_gate", "used_for_selection", "used_for_promotion"):
        _required_false(metrics.get(flag), name=f"annual {flag}")
    if run_manifest.get("rolling365_enabled") not in (None, False):
        raise TopologyReportError("Le rapport annuel ne doit pas annoncer un refit rolling.")
    if run_manifest.get("production_changed") is not False:
        raise TopologyReportError("Le rapport annuel ne doit pas modifier la production.")
    for key, path in (
        ("annual_policy_sha256", policy_path),
        ("annual_strategy_sha256", strategy_path),
        ("annual_strategy_seal_sha256", strategy_seal_path),
    ):
        declared = metrics.get(key)
        if not isinstance(declared, str) or not _SHA256_RE.fullmatch(declared.strip().lower()):
            raise TopologyReportError(f"{key} annuel invalide.")
        if declared.strip().lower() != _sha256(path):
            raise TopologyReportError(f"{key} annuel divergent.")
    for key in (
        "calibration_artifact_manifest_sha256",
        "source_backtest_sha256",
        "source_candidate_prediction_sha256",
        "config_sha256",
    ):
        value = metrics.get(key)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.strip().lower()):
            raise TopologyReportError(f"{key} annuel invalide.")
    seal_strategy_sha = strategy_seal.get(
        "annual_strategy_sha256", strategy_seal.get("prediction_sha256")
    )
    if seal_strategy_sha != _sha256(strategy_path):
        raise TopologyReportError("Le seal annuel contredit annual_strategy_hourly.")
    if policy.get("out_of_sample_scope") not in (None, "mixed_sequential_governed"):
        raise TopologyReportError("Le policy seal contredit la portee hors echantillon.")
    baseline = _mapping(metrics.get("baseline"), name="annual baseline")
    if _required_int(baseline.get("n_hours"), name="baseline.n_hours") != n_hours:
        raise TopologyReportError("Le baseline annuel ne couvre pas 8760 heures.")
    if _required_int(baseline.get("n_days"), name="baseline.n_days") != n_days:
        raise TopologyReportError("Le baseline annuel ne couvre pas 365 jours.")
    baseline_model = _required_text(baseline.get("model"), name="baseline.model")
    if baseline_model != "residual_corrected":
        raise TopologyReportError("Le baseline annuel doit etre residual_corrected.")
    baseline_mae = _required_finite(baseline.get("mae"), name="baseline.mae")
    strategies = _mapping(metrics.get("strategies"), name="annual strategies")
    if set(strategies) != set(ANNUAL_STRATEGIES):
        raise TopologyReportError("Les deux strategies annuelles exactes sont requises.")
    sequential = _annual_strategy_record(
        "sequential_governed_strategy",
        _mapping(strategies["sequential_governed_strategy"], name="sequential strategy"),
        baseline_mae=baseline_mae,
    )
    shadow = _annual_strategy_record(
        "causal_formal_shadow_strategy",
        _mapping(strategies["causal_formal_shadow_strategy"], name="formal shadow strategy"),
        baseline_mae=baseline_mae,
    )
    pure = _mapping(metrics.get("pure_topology_annual"), name="pure_topology_annual")
    _required_false(pure.get("available"), name="pure_topology_annual.available")
    pure_reason = _required_text(pure.get("reason"), name="pure_topology_annual.reason")
    storm = _annual_storm_record(
        metrics.get("storm"), governed_model="sequential_governed_strategy"
    )
    mkonline = _annual_mkonline_record(metrics.get("mkonline_production_reference"))
    record = TopologyAnnualReportRecord(
        zone=zone,
        timezone=timezone_name,
        start_local_day=start_local_day,
        end_local_day=end_local_day,
        start_utc=start_utc,
        end_utc=end_utc,
        n_days=n_days,
        n_hours=n_hours,
        baseline_model=baseline_model,
        baseline_mae=baseline_mae,
        sequential_governed_strategy=sequential,
        causal_formal_shadow_strategy=shadow,
        storm=storm,
        mkonline_production_reference=mkonline,
        topology_blend_candidate_available=False,
        pure_topology_annual_available=False,
        pure_topology_annual_reason=pure_reason,
        out_of_sample_scope="mixed_sequential_governed",
        annual_metrics_path=metrics_path,
        strategy_path=strategy_path,
        policy_path=policy_path,
        strategy_seal_path=strategy_seal_path,
        raw_metrics=metrics,
    )
    _validate_annual_hourly(record)
    return record


def _annual_input_snapshot(record: TopologyAnnualReportRecord) -> dict[Path, str]:
    paths = (
        record.annual_metrics_path,
        record.strategy_path,
        record.policy_path,
        record.strategy_seal_path,
        record.annual_metrics_path.parent / "run_manifest.json",
        record.annual_metrics_path.parent / "artifact_checksums.json",
    )
    return {path: _sha256(path) for path in paths}


def _verify_annual_input_snapshot(before: Mapping[Path, str]) -> None:
    changed = [
        path.name
        for path, digest in before.items()
        if not path.is_file() or _sha256(path) != digest
    ]
    if changed:
        raise TopologyReportError(
            "Le renderer a modifie un artefact annuel scelle: " + ", ".join(changed)
        )


def _build_annual_renderer_view(
    record: TopologyAnnualReportRecord,
    directory: Path,
) -> None:
    import pandas as pd

    frame = pd.read_csv(record.strategy_path)
    keep = [
        "delivery_start_utc",
        "forecast_origin_utc",
        "actual",
        *[
            f"{model}__{quantile}"
            for model in ("residual_corrected", "sequential_governed_strategy")
            for quantile in ("q10", "q50", "q90")
        ],
    ]
    backtest = frame.loc[:, keep].copy()
    backtest.to_csv(directory / "backtest_hourly_oof.csv.gz", index=False)
    forecast_columns = [
        "delivery_start_utc",
        *[
            f"{model}__{quantile}"
            for model in ("residual_corrected", "sequential_governed_strategy")
            for quantile in ("q10", "q50", "q90")
        ],
    ]
    frame.loc[:, forecast_columns].tail(24).to_csv(
        directory / f"forecast_hourly_{record.zone.lower()}.csv",
        index=False,
    )
    (directory / "metrics_hourly.json").write_text(
        json.dumps(
            {
                "metrics": [],
                "training_diagnostics": {
                    "metric_scope": ANNUAL_REPORT_TYPE,
                    "evaluation_start_local_date": record.start_local_day,
                    "evaluation_end_local_date": record.end_local_day,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_type": ANNUAL_REPORT_TYPE,
                "zone": record.zone,
                "timezone": record.timezone,
                "rolling365_enabled": False,
                "production_changed": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(
        {
            "timestamp": frame["delivery_start_utc"],
            "target": frame["actual"],
        }
    ).to_csv(inputs / "aligned_inputs.csv.gz", index=False)
    pd.DataFrame(
        {
            "timestamp": frame["delivery_start_utc"],
            "annual_topology_active": _strict_bool_series(
                frame["governed_topology_active"], name="governed_topology_active"
            ).astype(int),
        }
    ).to_csv(inputs / "model_covariates_with_future.csv.gz", index=False)
    pd.DataFrame(
        [
            {
                "zone": record.zone,
                "alias": "annual_topology_active",
                "series": "sealed_reporting_contract",
                "coverage_exact": 1.0,
                "coverage_after_fill": 1.0,
                "missing_after_fill": 0,
            }
        ]
    ).to_csv(inputs / "input_coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "alias": "annual_topology_active",
                "role": "reporting_audit",
                "series": "sealed_reporting_contract",
                "description": "Indicateur audite de topologie active",
                "future_strategies": "",
                "known_future": False,
                "source": "annual_strategy_hourly",
            }
        ]
    ).to_csv(inputs / "input_manifest.csv", index=False)


def _annual_outcome_rows(strategy: TopologyAnnualStrategyRecord) -> str:
    active_win = _pct(strategy.win_rate_active_days)
    return f"""
<tr><th>Tous les jours</th><td>{strategy.wins_all_days}</td><td>{strategy.ties_all_days}</td><td>{strategy.losses_all_days}</td><td>{html.escape(_pct(strategy.win_rate_all_days))}</td></tr>
<tr><th>Jours topology actifs</th><td>{strategy.wins_active_days}</td><td>{strategy.ties_active_days}</td><td>{strategy.losses_active_days}</td><td>{html.escape(active_win)}</td></tr>"""


def _annual_comparator_html(
    comparator: TopologyAnnualComparatorRecord,
    *,
    label: str,
    gain_label: str,
) -> str:
    if not comparator.available:
        return (
            f"<div class=\"annual-panel\"><h3>{html.escape(label)}</h3>"
            f"<p><strong>N/A</strong> — {html.escape(comparator.reason or 'indisponible')}</p></div>"
        )
    return f"""
<div class="annual-panel"><h3>{html.escape(label)}</h3>
  <p><b>Modèle :</b> {html.escape(comparator.model or 'N/A')}</p>
  <p><b>MAE :</b> {html.escape(_fmt(comparator.mae, digits=6))} €/MWh</p>
  <p><b>{html.escape(gain_label)} :</b> {html.escape(_fmt(comparator.gain_vs_governed_eur_mwh, digits=6))} €/MWh</p>
  <p><b>Couverture appariée :</b> {html.escape(_pct(comparator.pairing_coverage))} ({comparator.paired_hours} h)</p>
</div>"""


def _annual_banner_html(record: TopologyAnnualReportRecord) -> str:
    governed = record.sequential_governed_strategy
    shadow = record.causal_formal_shadow_strategy
    embedded = {
        "schema_version": SCHEMA_VERSION,
        "report_type": ANNUAL_REPORT_TYPE,
        "zone": record.zone,
        "period": {
            "start_local_day": record.start_local_day,
            "end_local_day": record.end_local_day,
            "n_days": record.n_days,
            "n_hours": record.n_hours,
        },
        "out_of_sample_scope": record.out_of_sample_scope,
        "pure_topology_annual_available": False,
        "topology_blend_candidate_available": False,
        "source": record.raw_metrics,
    }
    return f"""
<section id="topology-annual-audit" class="topology-annual" data-report-section="topology-annual-audit" data-zone="{html.escape(record.zone)}">
  <div class="annual-heading"><div><p class="annual-eyebrow">BACKTEST ANNUEL AUDITÉ</p>
  <h2>365 jours scellés — stratégie mixte, sans refit rolling</h2></div><span>8 760 heures exactes</span></div>
  <p>Fenêtre locale {html.escape(record.start_local_day)} → {html.escape(record.end_local_day)}. Le résultat principal combine le modèle actuel et la topologie uniquement après les décisions causales prévues. Il ne s'agit ni d'un réentraînement glissant, ni d'une performance annuelle d'un modèle topologique pur.</p>
  <div class="annual-kpis">
    <div><small>Baseline {html.escape(record.baseline_model)}</small><strong>{html.escape(_fmt(record.baseline_mae, digits=6))} €/MWh</strong></div>
    <div><small>Stratégie séquentielle gouvernée</small><strong>{html.escape(_fmt(governed.candidate_mae, digits=6))} €/MWh</strong><em>gain {html.escape(_fmt(governed.gain_eur_mwh, digits=6))}</em></div>
    <div><small>Shadow causal formel</small><strong>{html.escape(_fmt(shadow.candidate_mae, digits=6))} €/MWh</strong><em>gain {html.escape(_fmt(shadow.gain_eur_mwh, digits=6))}</em></div>
    <div><small>Topologie pure annuelle</small><strong>N/A</strong><em>non identifiable sans biais</em></div>
  </div>
  <div class="annual-grid">
    <div class="annual-panel"><h3>Stratégie séquentielle gouvernée</h3>
      <p>Topologie active : <b>{governed.active_hours} h</b> / 8 760 ({html.escape(_pct(governed.active_coverage))}), sur {governed.active_days} jours.</p>
      <table><thead><tr><th>Périmètre</th><th>Gagnés</th><th>Égalités</th><th>Perdus</th><th>Win rate</th></tr></thead><tbody>{_annual_outcome_rows(governed)}</tbody></table>
      <p class="annual-note">Les égalités incluent les jours où le fallback identité reproduit exactement le baseline.</p>
    </div>
    <div class="annual-panel"><h3>Shadow causal formel</h3>
      <p>Topologie active : <b>{shadow.active_hours} h</b> / 8 760 ({html.escape(_pct(shadow.active_coverage))}), sur {shadow.active_days} jours.</p>
      <table><thead><tr><th>Périmètre</th><th>Gagnés</th><th>Égalités</th><th>Perdus</th><th>Win rate</th></tr></thead><tbody>{_annual_outcome_rows(shadow)}</tbody></table>
      <p class="annual-note">Uniquement les holdouts formels ouverts ; A et development sont exclus.</p>
    </div>
    {_annual_comparator_html(record.storm, label='Storm — référence appariée', gain_label='Gain stratégie gouvernée vs Storm')}
    {_annual_comparator_html(record.mkonline_production_reference, label='MKOnline — blend de production', gain_label='Gain MKOnline vs stratégie gouvernée')}
    <div class="annual-panel"><h3>Topologie pure sur 365 jours</h3><p><strong>N/A</strong> — {html.escape(record.pure_topology_annual_reason)}</p><p class="annual-note">Aucune valeur n'est inventée sur seed, A ou les holdouts non ouverts.</p></div>
    <div class="annual-panel"><h3>Blend topologique</h3><p><code>topology_blend_candidate_available=false</code></p><p>MKOnline, lorsqu'il est présent, reste uniquement le blend de production scellé, avec poids inchangés.</p></div>
  </div>
  <script type="application/json" id="topology-annual-report-data">{_safe_json(embedded)}</script>
</section>"""


_ANNUAL_CSS = """
.topology-annual{border:2px solid #0f766e;background:linear-gradient(135deg,color-mix(in srgb,#14b8a6 10%,var(--card,#fff)),var(--card,#fff));}
.annual-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.annual-heading h2{margin:2px 0 0}.annual-heading>span{background:#ccfbf1;color:#115e59;border-radius:999px;padding:7px 12px;font-size:12px;font-weight:800;white-space:nowrap}.annual-eyebrow{margin:0;color:#0f766e;font-size:12px;font-weight:800;letter-spacing:.08em}.annual-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:18px 0}.annual-kpis>div,.annual-panel{border:1px solid color-mix(in srgb,#2dd4bf 45%,var(--border,#dfe5ea));background:var(--card,#fff);border-radius:12px;padding:14px}.annual-kpis small,.annual-kpis em{display:block;color:var(--muted,#64748b)}.annual-kpis strong{display:block;margin:6px 0;font-size:17px}.annual-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}.annual-panel h3{margin-top:0}.annual-panel table{width:100%;border-collapse:collapse;font-size:12px}.annual-panel th,.annual-panel td{padding:7px;border-bottom:1px solid var(--border,#e2e8f0);text-align:left}.annual-note{font-size:12px;color:var(--muted,#64748b)}.topology-annual code{font-size:11px;overflow-wrap:anywhere}@media(max-width:720px){.annual-heading{display:block}.annual-heading>span{display:inline-block;margin-top:10px}.annual-grid{grid-template-columns:1fr}}
"""


def _decorate_annual(source: str, record: TopologyAnnualReportRecord) -> str:
    if "</style>" not in source or "<main>" not in source:
        raise TopologyReportError(
            "Le HTML courant ne contient pas les points d'insertion annuels."
        )
    result = source.replace("</style>", _ANNUAL_CSS + "</style>", 1)
    return result.replace("<main>", "<main>" + _annual_banner_html(record), 1)


def write_topology_annual_html_report(
    zone_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    project_root: str | Path | None = None,
    report_writer: Callable[..., Path] = write_hourly_html_report,
) -> TopologyAnnualReportArtifact:
    """Render one immutable 365-day mixed-strategy report with the existing engine."""

    if overwrite:
        raise TopologyReportError("overwrite est interdit pour un rapport annuel scelle.")
    root = _project_root(project_root)
    directory = Path(zone_dir).expanduser().resolve()
    record = load_topology_annual_evaluation(directory, project_root=root)
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else directory / "reports" / f"topology_annual_{record.zone.lower()}.html"
    )
    _assert_experiment_path(destination, project_root=root, name="annual output_path")
    if destination.suffix.casefold() != ".html":
        raise TopologyReportError("annual output_path doit etre un fichier .html.")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep internal names independent from the (potentially long) report name.
    # The annual runner already nests these paths deeply on Windows, where the
    # longer stem-based names could exceed MAX_PATH while pandas writes gzip.
    staging = destination.parent / f".ae-{uuid4().hex[:8]}"
    temporary = destination.parent / f".ar-{uuid4().hex[:8]}.html"
    before = _annual_input_snapshot(record)
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _build_annual_renderer_view(record, staging)
        rendered = Path(
            report_writer(
                staging,
                output_path=temporary,
                title=(
                    f"Chronos-2 {record.zone} — 365 jours scellés, "
                    "stratégie mixte sans refit rolling"
                ),
                native_model="sequential_governed_strategy",
                baseline_model="residual_corrected",
                zone=record.zone,
                timezone=record.timezone,
                history_hours=ANNUAL_EXPECTED_HOURS,
            )
        ).resolve()
        if rendered != temporary.resolve() or not temporary.is_file():
            raise TopologyReportError(
                "Le renderer annuel n'a pas produit le fichier temporaire attendu."
            )
        _verify_annual_input_snapshot(before)
        temporary.write_text(
            _decorate_annual(temporary.read_text(encoding="utf-8"), record),
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
        if staging.exists():
            shutil.rmtree(staging)
    return TopologyAnnualReportArtifact(
        path=destination,
        record=record,
        sha256=_sha256(destination),
    )


def _annual_index_row(
    artifact: TopologyAnnualReportArtifact | None,
    *,
    zone: str,
    index_directory: Path,
) -> str:
    if artifact is None:
        return f"""<tr data-zone="{html.escape(zone)}" data-status="missing"><td><b>{html.escape(zone)}</b></td><td colspan="12">N/A — rapport annuel absent</td></tr>"""
    record = artifact.record
    governed = record.sequential_governed_strategy
    shadow = record.causal_formal_shadow_strategy
    storm = record.storm
    mkonline = record.mkonline_production_reference
    storm_text = (
        f"{_fmt(storm.mae, digits=6)} / {_pct(storm.pairing_coverage)}"
        if storm.available
        else f"N/A — {storm.reason}"
    )
    mkonline_text = (
        f"{_fmt(mkonline.mae, digits=6)}"
        if mkonline.available
        else f"N/A — {mkonline.reason}"
    )
    href = _relative_href(artifact.path, index_directory)
    return f"""
<tr data-zone="{html.escape(record.zone)}" data-status="complete">
  <td><b>{html.escape(record.zone)}</b></td><td>{html.escape(record.start_local_day)} → {html.escape(record.end_local_day)}</td>
  <td>{html.escape(_fmt(record.baseline_mae, digits=6))}</td>
  <td>{html.escape(_fmt(governed.candidate_mae, digits=6))}</td><td>{html.escape(_fmt(governed.gain_eur_mwh, digits=6))}</td><td>{html.escape(_pct(governed.active_coverage))}</td><td>{governed.wins_all_days}/{governed.ties_all_days}/{governed.losses_all_days} · actif {html.escape(_pct(governed.win_rate_active_days))}</td>
  <td>{html.escape(_fmt(shadow.candidate_mae, digits=6))}</td><td>{html.escape(_fmt(shadow.gain_eur_mwh, digits=6))}</td><td>{html.escape(_pct(shadow.active_coverage))}</td>
  <td>{html.escape(storm_text)}</td><td>{html.escape(mkonline_text)}</td><td>N/A</td><td><a href="{html.escape(href)}">Ouvrir</a></td>
</tr>"""


def write_topology_annual_report_index(
    artifacts: Sequence[TopologyAnnualReportArtifact],
    *,
    output_path: str | Path,
    selected_zones: Sequence[str] | None = None,
    overwrite: bool = False,
    project_root: str | Path | None = None,
) -> Path:
    """Write the immutable multi-zone index for exact annual strategy reports."""

    if overwrite:
        raise TopologyReportError("overwrite est interdit pour l'index annuel scelle.")
    root = _project_root(project_root)
    destination = Path(output_path).expanduser().resolve()
    _assert_experiment_path(destination, project_root=root, name="annual index output_path")
    if destination.suffix.casefold() != ".html":
        raise TopologyReportError("L'index annuel doit etre un fichier .html.")
    if destination.exists():
        raise FileExistsError(destination)
    by_zone: dict[str, TopologyAnnualReportArtifact] = {}
    for artifact in artifacts:
        zone = artifact.record.zone
        if zone in by_zone:
            raise TopologyReportError(f"Rapport annuel duplique pour {zone}.")
        if not artifact.path.is_file() or _sha256(artifact.path) != artifact.sha256:
            raise TopologyReportError(f"Rapport annuel absent ou modifie pour {zone}.")
        by_zone[zone] = artifact
    if selected_zones is None:
        zones = tuple(zone for zone in SUPPORTED_ZONES if zone in by_zone)
    else:
        zones = tuple(str(zone).strip().upper() for zone in selected_zones)
        if len(set(zones)) != len(zones):
            raise TopologyReportError("selected_zones annuel contient des doublons.")
        if any(zone not in SUPPORTED_ZONES for zone in zones):
            raise TopologyReportError("selected_zones annuel contient une zone invalide.")
    if not zones:
        raise TopologyReportError("Aucune zone annuelle a consolider.")
    rows = "".join(
        _annual_index_row(
            by_zone.get(zone),
            zone=zone,
            index_directory=destination.parent,
        )
        for zone in zones
    )
    generated = datetime.now(timezone.utc).isoformat()
    document = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chronos-2 · backtest annuel topologique</title><style>:root{{--bg:#f4f6f8;--card:#fff;--text:#172033;--line:#dbe2ea}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Arial,sans-serif}}header{{background:#0f172a;color:#fff;padding:28px 5vw}}header h1{{margin:0 0 8px}}main{{max-width:1700px;margin:24px auto;padding:0 24px}}section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 5px 18px rgba(15,23,42,.05)}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:1550px}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;font-size:12px;vertical-align:top}}th{{background:#f8fafc}}a{{color:#0369a1}}code{{font-size:11px}}</style></head><body><header><h1>365 jours scellés — stratégie mixte, sans refit rolling</h1><p>Comparaison sur exactement les mêmes 8 760 heures que les modèles actuels · généré {html.escape(generated)}</p></header><main><section><h2>Résultats annuels par pays</h2><p>La topologie pure annuelle reste <b>N/A</b> : seed, A et les holdouts non ouverts ne sont jamais inventés. MKOnline est uniquement une référence de production post-seal ; <code>topology_blend_candidate_available=false</code>.</p><div class="table"><table><thead><tr><th>Pays</th><th>Période</th><th>MAE baseline</th><th>MAE gouvernée</th><th>Gain</th><th>Topologie active</th><th>W/T/L tous jours · WR actif</th><th>MAE shadow formel</th><th>Gain shadow</th><th>Shadow actif</th><th>Storm MAE / couverture</th><th>MKOnline prod MAE</th><th>Topologie pure 365j</th><th>Rapport</th></tr></thead><tbody>{rows}</tbody></table></div></section></main></body></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.stem}-{uuid4().hex}.tmp"
    try:
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = (
    "ANNUAL_EXPECTED_DAYS",
    "ANNUAL_EXPECTED_HOURS",
    "ANNUAL_METRICS_FILENAME",
    "ANNUAL_POLICY_FILENAME",
    "ANNUAL_REPORT_TYPE",
    "ANNUAL_STRATEGIES",
    "ANNUAL_STRATEGY_FILENAME",
    "ANNUAL_STRATEGY_SEAL_FILENAME",
    "BLEND_ZONES",
    "EVALUATION_FILENAME",
    "PRICEFM_MODEL_URL",
    "PRICEFM_PAPER_URL",
    "PRICEFM_REPOSITORY_URL",
    "SCHEMA_VERSION",
    "SUPPORTED_VARIANTS",
    "SUPPORTED_ZONES",
    "BlendWeights",
    "TopologyReportArtifact",
    "TopologyReportError",
    "TopologyReportRecord",
    "TopologyAnnualComparatorRecord",
    "TopologyAnnualReportArtifact",
    "TopologyAnnualReportRecord",
    "TopologyAnnualStrategyRecord",
    "load_topology_annual_evaluation",
    "load_topology_evaluation",
    "write_topology_annual_html_report",
    "write_topology_annual_report_index",
    "write_topology_html_report",
    "write_topology_report_index",
)
