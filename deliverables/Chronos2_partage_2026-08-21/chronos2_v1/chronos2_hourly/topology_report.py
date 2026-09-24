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
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote
from uuid import uuid4

from chronos2_hourly.reporting import write_hourly_html_report


SCHEMA_VERSION = 1
EVALUATION_FILENAME = "topology_evaluation.json"
SUPPORTED_ZONES = ("FR", "DE", "BE", "NL", "ES")
SUPPORTED_VARIANTS = ("autonomous", "mkonline_blend")
BLEND_ZONES = frozenset({"FR", "NL"})
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

    final_metrics = _segment(variant_payload, "metrics", "final")
    final_gate = _segment(variant_payload, "gates", "final")
    candidate_mae = _metric_value(
        final_metrics, final_gate, "candidate_mae", "topology_mae"
    )
    baseline_mae = _metric_value(
        final_metrics, final_gate, "baseline_mae", "identity_mae"
    )
    declared_gain = _metric_value(
        final_metrics, final_gate, "gain_eur_mwh", "mae_gain_eur_mwh"
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
        final_metrics, final_gate, "daily_win_rate", "win_rate"
    )
    if win_rate is not None and not 0.0 <= win_rate <= 1.0:
        raise TopologyReportError("daily_win_rate doit etre entre 0 et 1.")
    gate_passes = _optional_bool(final_gate.get("passes"))
    gate_reasons = _reason_list(final_gate)

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

    issues: list[str] = []
    required_values = (
        ("selected_radius", selected_radius),
        ("MAE candidat final", candidate_mae),
        ("MAE reference finale", baseline_mae),
        ("gate finale", gate_passes),
        ("couverture PIT", min_coverage),
        ("feature_schema_sha256", feature_hash),
        ("model_hyperparameters_sha256", hyperparameter_hash),
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
        payload.get("mkonline_used_as_prediction_input"),
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
            if variant_payload.get("recommended_variant") != "autonomous":
                issues.append(
                    "un blend refuse doit recommander la variante autonome"
                )
            if variant_payload.get("production_weights_unchanged") is not True:
                issues.append(
                    "un blend refuse doit conserver les poids de production"
                )

    complete = not issues
    if not complete:
        status = "incomplete"
    elif gate_passes:
        status = "promoted"
    else:
        status = "fallback"
    if gate_passes:
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
            "Fallback autonome : les poids MKOnline de production restent "
            "inchanges."
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
      <h4>Motifs de la gate finale</h4><ul>{gate_reasons}</ul>
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
.topology-audit{border:2px solid #2563eb;background:linear-gradient(135deg,#eff6ff,#fff);}
.topology-audit-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;}
.topology-audit-heading h2{margin:2px 0 0;}.topology-audit-eyebrow{margin:0;color:#1d4ed8;font-size:12px;font-weight:800;letter-spacing:.08em;}
.topology-audit-status{border-radius:999px;padding:7px 12px;font-size:12px;font-weight:800;background:#dbeafe;color:#1e3a8a;white-space:nowrap;}
.topology-status-promoted{border-color:#059669}.topology-status-promoted .topology-audit-status{background:#d1fae5;color:#065f46;}
.topology-status-fallback{border-color:#d97706}.topology-status-fallback .topology-audit-status{background:#fef3c7;color:#92400e;}
.topology-status-incomplete{border-color:#dc2626}.topology-status-incomplete .topology-audit-status{background:#fee2e2;color:#991b1b;}
.topology-audit-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px;margin:18px 0;}
.topology-audit-kpis>div{border:1px solid #bfdbfe;background:#fff;border-radius:10px;padding:12px;}.topology-audit-kpis span{display:block;color:#64748b;font-size:12px;margin-bottom:5px;}.topology-audit-kpis strong{font-size:15px;}
.topology-audit-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px;}.topology-audit-panel{border:1px solid #dbeafe;background:#fff;border-radius:12px;padding:14px;overflow:auto;}.topology-audit-panel h3{margin-top:0;}
.topology-audit table{width:100%;border-collapse:collapse;font-size:12px}.topology-audit th,.topology-audit td{border-bottom:1px solid #e2e8f0;padding:7px;text-align:left;}.topology-audit code{font-size:10px;overflow-wrap:anywhere}.topology-audit-small{font-size:12px;color:#64748b;}.topology-audit details{margin-top:16px;padding-top:12px;border-top:1px solid #bfdbfe}.topology-audit summary{cursor:pointer;font-weight:700;}
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


__all__ = (
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
    "load_topology_evaluation",
    "write_topology_html_report",
    "write_topology_report_index",
)
