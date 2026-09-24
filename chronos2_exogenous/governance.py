"""Fail-closed governance for a Chronos-2 exogenous shadow candidate.

The module deliberately has no dependency on the live launcher.  It evaluates
paired forecasts, records a promotion decision, and seals all evidence in a
self-contained bundle.  It never edits a production recipe or activates a
model: a ``promote`` decision only authorises the separate deployment step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


POLICY_SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 1
DECISIONS = frozenset({"reject", "shadow", "promote"})
GOVERNABLE_EVALUATION_ROLE = "primary_predeclared"
REQUIRED_PREDICTION_COLUMNS = (
    "delivery_start_utc",
    "forecast_origin_utc",
    "actual",
    "baseline_q10",
    "baseline_q50",
    "baseline_q90",
    "candidate_q10",
    "candidate_q50",
    "candidate_q90",
)
REQUIRED_EXPERIMENT_FLAGS: Mapping[str, object] = {
    "training_window_days": 365,
    "evaluation_days": 365,
    "cutoff_local_time": "08:00",
    "candidate_frozen_before_evaluation": True,
    "feature_selection_frozen_before_evaluation": True,
    "actual_future_used_as_input": False,
    "storm_used_for_input": False,
    "storm_used_for_selection": False,
    "mkonline_used_for_input": False,
    "pit_audit_passed": True,
}


class ExogenousGovernanceError(RuntimeError):
    """Raised when evidence, policy, or a sealed bundle is invalid."""


@dataclass(frozen=True)
class GovernancePolicy:
    """Predeclared rolling and live-shadow promotion thresholds."""

    schema_version: int = POLICY_SCHEMA_VERSION
    timezone: str = "Europe/Paris"
    rolling_evaluation_days: int = 365
    minimum_mae_gain_eur_mwh: float = 0.05
    minimum_relative_mae_gain: float = 0.005
    minimum_daily_win_rate: float = 0.50
    require_positive_chronological_halves: bool = True
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 20260903
    bootstrap_confidence: float = 0.95
    bootstrap_block_days: int = 7
    require_bootstrap_lower_bound_positive: bool = True
    peak_local_hours: tuple[int, ...] = (7, 8, 17, 18, 19, 20)
    tail_actual_absolute_quantile: float = 0.90
    maximum_peak_mae_relative_degradation: float = 0.02
    maximum_tail_mae_relative_degradation: float = 0.05
    maximum_daily_mean_mae_relative_degradation: float = 0.0
    require_probabilistic_metrics: bool = True
    maximum_pinball_relative_degradation: float = 0.0
    maximum_interval_coverage_error_increase: float = 0.02
    interval_nominal_coverage: float = 0.80
    shadow_required_for_promotion: bool = True
    shadow_evaluation_days: int = 30
    shadow_minimum_mae_gain_eur_mwh: float = 0.0
    shadow_minimum_relative_mae_gain: float = 0.0
    shadow_minimum_daily_win_rate: float = 0.50
    shadow_require_bootstrap_lower_bound_positive: bool = True

    def validate(self) -> "GovernancePolicy":
        if type(self.schema_version) is not int or self.schema_version != POLICY_SCHEMA_VERSION:
            raise ExogenousGovernanceError(
                f"schema_version politique attendu={POLICY_SCHEMA_VERSION}."
            )
        try:
            ZoneInfo(self.timezone)
        except Exception as exc:
            raise ExogenousGovernanceError(
                f"Timezone invalide: {self.timezone!r}."
            ) from exc
        for name in (
            "rolling_evaluation_days",
            "bootstrap_samples",
            "bootstrap_block_days",
            "shadow_evaluation_days",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ExogenousGovernanceError(f"{name} doit etre >= 1.")
        if self.shadow_evaluation_days < 2:
            raise ExogenousGovernanceError(
                "shadow_evaluation_days doit etre >= 2 pour le bootstrap."
            )
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ExogenousGovernanceError("bootstrap_seed doit etre un entier >= 0.")
        for name in (
            "require_positive_chronological_halves",
            "require_bootstrap_lower_bound_positive",
            "require_probabilistic_metrics",
            "shadow_required_for_promotion",
            "shadow_require_bootstrap_lower_bound_positive",
        ):
            if type(getattr(self, name)) is not bool:
                raise ExogenousGovernanceError(f"{name} doit etre un booleen.")
        if self.rolling_evaluation_days != 365:
            raise ExogenousGovernanceError(
                "La gate formelle exige exactement 365 jours."
            )
        for name in (
            "minimum_mae_gain_eur_mwh",
            "minimum_relative_mae_gain",
            "maximum_peak_mae_relative_degradation",
            "maximum_tail_mae_relative_degradation",
            "maximum_daily_mean_mae_relative_degradation",
            "maximum_pinball_relative_degradation",
            "maximum_interval_coverage_error_increase",
            "shadow_minimum_mae_gain_eur_mwh",
            "shadow_minimum_relative_mae_gain",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise ExogenousGovernanceError(f"{name} doit etre numerique.")
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ExogenousGovernanceError(
                    f"{name} doit etre fini et positif ou nul."
                )
        for name in (
            "minimum_daily_win_rate",
            "tail_actual_absolute_quantile",
            "bootstrap_confidence",
            "interval_nominal_coverage",
            "shadow_minimum_daily_win_rate",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise ExogenousGovernanceError(f"{name} doit etre numerique.")
            value = float(raw)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ExogenousGovernanceError(f"{name} doit etre dans ]0, 1[.")
        hours = tuple(int(value) for value in self.peak_local_hours)
        if not hours or len(set(hours)) != len(hours) or any(
            value < 0 or value > 23 for value in hours
        ):
            raise ExogenousGovernanceError(
                "peak_local_hours doit contenir des heures uniques entre 0 et 23."
            )
        return self

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["peak_local_hours"] = list(self.peak_local_hours)
        return payload


@dataclass(frozen=True)
class WindowEvaluation:
    """Metrics recomputed from one exact paired causal window."""

    phase: str
    start_day_local: str
    end_day_local: str
    days: int
    hours: int
    baseline_mae: float
    candidate_mae: float
    mae_gain_eur_mwh: float
    relative_mae_gain: float
    baseline_bias: float
    candidate_bias: float
    daily_win_rate: float
    first_half_gain_eur_mwh: float
    second_half_gain_eur_mwh: float
    bootstrap_ci_lower_eur_mwh: float
    bootstrap_ci_upper_eur_mwh: float
    peak_baseline_mae: float
    peak_candidate_mae: float
    peak_relative_degradation: float
    tail_threshold_abs_actual: float
    tail_baseline_mae: float
    tail_candidate_mae: float
    tail_relative_degradation: float
    daily_mean_baseline_mae: float
    daily_mean_candidate_mae: float
    daily_mean_relative_degradation: float
    baseline_pinball: float
    candidate_pinball: float
    pinball_relative_degradation: float
    baseline_interval_coverage: float
    candidate_interval_coverage: float
    baseline_interval_coverage_error: float
    candidate_interval_coverage_error: float
    interval_coverage_error_increase: float
    causal_origin_violations: int
    quantile_crossings: int


@dataclass(frozen=True)
class GateResult:
    phase: str
    passes: bool
    checks: Mapping[str, Mapping[str, object]]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PromotionDecision:
    """Pure decision; it deliberately performs no deployment mutation."""

    schema_version: int
    candidate_model: str
    zone: str
    decision: str
    rolling365: WindowEvaluation
    rolling365_gate: GateResult
    live_shadow: WindowEvaluation | None
    live_shadow_gate: GateResult | None
    shadow_evidence_verified: bool
    production_pit_evidence: bool
    production_pit_gate_passes: bool
    production_pipeline_evidence: bool
    production_pipeline_gate_passes: bool
    reasons: tuple[str, ...]
    production_activation_performed: bool = False
    incumbent_contract_modified: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return payload


def _strict_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExogenousGovernanceError(f"{name} doit etre un objet.")
    return value


def load_policy(path: str | Path) -> GovernancePolicy:
    """Load JSON/YAML while rejecting unknown keys and silent typos."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is operational
            raise ExogenousGovernanceError("PyYAML est requis pour une politique YAML.") from exc
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    mapping = _strict_mapping(payload, name=str(source))
    allowed = {item.name for item in fields(GovernancePolicy)}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ExogenousGovernanceError(
            f"Parametres de politique inconnus: {', '.join(unknown)}."
        )
    values = dict(mapping)
    if "peak_local_hours" in values:
        raw_hours = values["peak_local_hours"]
        if isinstance(raw_hours, (str, bytes)) or not isinstance(raw_hours, Sequence):
            raise ExogenousGovernanceError("peak_local_hours doit etre une liste.")
        values["peak_local_hours"] = tuple(int(value) for value in raw_hours)
    try:
        return GovernancePolicy(**values).validate()
    except TypeError as exc:
        raise ExogenousGovernanceError(f"Politique invalide: {exc}") from exc


def load_prediction_evidence(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffixes = "".join(source.suffixes).lower()
    if suffixes.endswith(".parquet"):
        frame = pd.read_parquet(source)
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        frame = pd.read_csv(source)
    else:
        raise ExogenousGovernanceError(
            f"Format de predictions non supporte: {source.name}."
        )
    if not isinstance(frame, pd.DataFrame):
        raise ExogenousGovernanceError(f"{source}: tableau attendu.")
    return frame


def validate_experiment_manifest(
    manifest: str | Path | Mapping[str, object],
    *,
    zone: str,
) -> dict[str, object]:
    """Require explicit causality and freeze evidence before scoring."""

    if isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        source = Path(manifest).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = dict(
            _strict_mapping(
                json.loads(source.read_text(encoding="utf-8")), name=str(source)
            )
        )
    model_id = payload.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ExogenousGovernanceError("experiment_manifest.model_id absent.")
    evaluation_role = payload.get("evaluation_role")
    if evaluation_role != GOVERNABLE_EVALUATION_ROLE:
        if evaluation_role == "diagnostic_only":
            raise ExogenousGovernanceError(
                "experiment_manifest.evaluation_role=diagnostic_only: une ablation "
                "observee sur le holdout final ne peut etre ni gouvernee ni promue."
            )
        raise ExogenousGovernanceError(
            "experiment_manifest.evaluation_role doit valoir "
            "'primary_predeclared' pour toute gouvernance."
        )
    declared_zone = payload.get("zone")
    if declared_zone is not None and str(declared_zone).strip().upper() != zone.upper():
        raise ExogenousGovernanceError(
            f"Zone du manifeste {declared_zone!r} incompatible avec {zone}."
        )
    failures: list[str] = []
    for key, expected in REQUIRED_EXPERIMENT_FLAGS.items():
        actual = payload.get(key, object())
        if type(actual) is not type(expected) or actual != expected:
            failures.append(f"{key}={actual!r}, attendu={expected!r}")
    if failures:
        raise ExogenousGovernanceError(
            "Preuves causales/freeze incompletes: " + "; ".join(failures)
        )
    production_pit_evidence = payload.get("production_pit_evidence")
    if type(production_pit_evidence) is not bool:
        raise ExogenousGovernanceError(
            "experiment_manifest.production_pit_evidence doit etre un booleen "
            "explicite; seule la valeur true permet une promotion."
        )
    production_pipeline_evidence = payload.get("production_pipeline_evidence")
    if type(production_pipeline_evidence) is not bool:
        raise ExogenousGovernanceError(
            "experiment_manifest.production_pipeline_evidence doit etre un "
            "booleen explicite; seule une comparaison du pipeline final avec "
            "l'incumbent autonome permet une promotion."
        )
    for key in ("checkpoint_sha256", "schema_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ExogenousGovernanceError(
                f"experiment_manifest.{key} doit etre un SHA-256 hexadecimal."
            )
    return payload


def _validated_shadow_panel_evidence(
    raw: object, *, index: int, expected_zone: str
) -> dict[str, object]:
    """Recompute the canonical panel contract carried by shadow evidence."""

    if not isinstance(raw, Mapping):
        raise ExogenousGovernanceError(
            f"shadow_manifest.shadow_panel_evidence[{index}] invalide."
        )
    required = {
        "schema_version",
        "purpose",
        "panel_sha256",
        "panel_audit_sha256",
        "panel_created_at_utc",
        "zone",
        "delivery_day",
        "forecast_origin_utc",
        "forecast_origin_timezone",
        "delivery_timezone",
        "pack",
        "production_pit_evidence",
        "source_hashes",
        "source_audit_hashes",
        "source_cutoff_timezones",
        "target_source_sha256",
        "horizon_actuals_present",
        "panel_contract_sha256",
    }
    if set(raw) != required:
        missing = sorted(required.difference(raw))
        extra = sorted(set(raw).difference(required))
        raise ExogenousGovernanceError(
            "shadow_manifest: schema de provenance panel non exact "
            f"(absent={missing}, extra={extra})."
        )
    if raw.get("schema_version") != 1 or raw.get("purpose") != "prospective_shadow":
        raise ExogenousGovernanceError(
            "shadow_manifest: purpose/schema de provenance panel invalide."
        )
    zone_value = str(raw.get("zone", "")).strip().upper()
    if zone_value != expected_zone:
        raise ExogenousGovernanceError(
            "shadow_manifest: zone de provenance panel incoherente."
        )

    def digest(value: object, label: str) -> str:
        result = str(value).strip().lower() if isinstance(value, str) else ""
        if re.fullmatch(r"[0-9a-f]{64}", result) is None:
            raise ExogenousGovernanceError(
                f"shadow_manifest.shadow_panel_evidence[{index}].{label} invalide."
            )
        return result

    def hashes(value: object, label: str) -> dict[str, str]:
        if not isinstance(value, Mapping) or not value:
            raise ExogenousGovernanceError(
                f"shadow_manifest.shadow_panel_evidence[{index}].{label} vide."
            )
        result: dict[str, str] = {}
        for name, value_digest in value.items():
            key = str(name).strip()
            if not key or key in result:
                raise ExogenousGovernanceError(
                    f"shadow_manifest: cle invalide dans {label}."
                )
            result[key] = digest(value_digest, f"{label}.{key}")
        return dict(sorted(result.items()))

    created = pd.to_datetime(raw.get("panel_created_at_utc"), utc=True, errors="coerce")
    origin = pd.to_datetime(raw.get("forecast_origin_utc"), utc=True, errors="coerce")
    if pd.isna(created) or pd.isna(origin):
        raise ExogenousGovernanceError(
            "shadow_manifest: timestamps de provenance panel invalides."
        )
    try:
        delivery_day = date.fromisoformat(str(raw.get("delivery_day"))).isoformat()
    except ValueError as exc:
        raise ExogenousGovernanceError(
            "shadow_manifest: delivery_day de provenance invalide."
        ) from exc
    forecast_timezone = str(raw.get("forecast_origin_timezone", "")).strip()
    delivery_timezone = str(raw.get("delivery_timezone", "")).strip()
    try:
        ZoneInfo(forecast_timezone)
        ZoneInfo(delivery_timezone)
    except Exception as exc:
        raise ExogenousGovernanceError(
            "shadow_manifest: timezone de provenance panel invalide."
        ) from exc
    if type(raw.get("production_pit_evidence")) is not bool or type(
        raw.get("horizon_actuals_present")
    ) is not bool:
        raise ExogenousGovernanceError(
            "shadow_manifest: drapeaux de provenance panel non booleens."
        )
    source_hashes = hashes(raw.get("source_hashes"), "source_hashes")
    source_audit_hashes = hashes(
        raw.get("source_audit_hashes"), "source_audit_hashes"
    )
    if not set(source_audit_hashes).issubset(source_hashes):
        raise ExogenousGovernanceError(
            "shadow_manifest: source_audit_hashes hors sources du panel."
        )
    timezone_values = raw.get("source_cutoff_timezones")
    if not isinstance(timezone_values, Mapping) or set(map(str, timezone_values)) != set(
        source_hashes
    ):
        raise ExogenousGovernanceError(
            "shadow_manifest: source_cutoff_timezones incomplet."
        )
    normalised_timezones: dict[str, str] = {}
    for name in source_hashes:
        timezone_name = str(timezone_values[name]).strip()
        try:
            ZoneInfo(timezone_name)
        except Exception as exc:
            raise ExogenousGovernanceError(
                f"shadow_manifest: timezone source invalide pour {name}."
            ) from exc
        normalised_timezones[name] = timezone_name
    external = set(source_hashes).difference({"deterministic_calendar"})
    if bool(raw["production_pit_evidence"]) and set(source_audit_hashes) != external:
        raise ExogenousGovernanceError(
            "shadow_manifest: production=true sans sidecar pour toutes les sources."
        )
    normalised: dict[str, object] = {
        "schema_version": 1,
        "purpose": "prospective_shadow",
        "panel_sha256": digest(raw["panel_sha256"], "panel_sha256"),
        "panel_audit_sha256": digest(
            raw["panel_audit_sha256"], "panel_audit_sha256"
        ),
        "panel_created_at_utc": pd.Timestamp(created).isoformat(),
        "zone": zone_value,
        "delivery_day": delivery_day,
        "forecast_origin_utc": pd.Timestamp(origin).isoformat(),
        "forecast_origin_timezone": forecast_timezone,
        "delivery_timezone": delivery_timezone,
        "pack": str(raw.get("pack", "")).strip(),
        "production_pit_evidence": bool(raw["production_pit_evidence"]),
        "source_hashes": source_hashes,
        "source_audit_hashes": source_audit_hashes,
        "source_cutoff_timezones": dict(sorted(normalised_timezones.items())),
        "target_source_sha256": digest(
            raw["target_source_sha256"], "target_source_sha256"
        ),
        "horizon_actuals_present": bool(raw["horizon_actuals_present"]),
    }
    expected_contract = hashlib.sha256(_canonical_json_bytes(normalised)).hexdigest()
    if digest(raw["panel_contract_sha256"], "panel_contract_sha256") != expected_contract:
        raise ExogenousGovernanceError(
            "shadow_manifest: panel_contract_sha256 divergent."
        )
    return {**normalised, "panel_contract_sha256": expected_contract}


def validate_shadow_manifest(
    manifest: str | Path | Mapping[str, object],
    *,
    experiment_manifest: Mapping[str, object],
    zone: str,
    expected_rows: int | None = None,
) -> dict[str, object]:
    """Bind prospective live evidence to the exact frozen adapter.

    Actual prices may be attached only after the immutable forecast has been
    sealed.  This sidecar prevents a hand-crafted retrospective CSV from
    satisfying the live-shadow gate.
    """

    if isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        source = Path(manifest).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = dict(
            _strict_mapping(
                json.loads(source.read_text(encoding="utf-8")), name=str(source)
            )
        )
    expected_model = str(
        experiment_manifest.get("experiment_id", experiment_manifest["model_id"])
    ).strip()
    expected: Mapping[str, object] = {
        "format_version": 3,
        "kind": "chronos2_exogenous_append_only_shadow",
        "candidate_model": expected_model,
        "candidate_output_stage": "chronos2_exogenous",
        "residual_corrector_applied": False,
        "zone": zone,
        "checkpoint_sha256": experiment_manifest["checkpoint_sha256"],
        "schema_sha256": experiment_manifest["schema_sha256"],
        "candidate_frozen_before_shadow": True,
        "actuals_attached_after_forecast_freeze": True,
        "prospective_capture_deadline_enforced": True,
        "prospective_capture_deadline_hours": 4,
        "forecast_artifact_checksums_valid": True,
        "storm_used_for_prediction": False,
        "mkonline_used_for_prediction": False,
    }
    failures: list[str] = []
    for key, expected_value in expected.items():
        actual = payload.get(key, object())
        if type(actual) is not type(expected_value) or actual != expected_value:
            failures.append(f"{key}={actual!r}, attendu={expected_value!r}")
    if failures:
        raise ExogenousGovernanceError(
            "Manifeste live shadow invalide: " + "; ".join(failures)
        )
    raw_panel_evidence = payload.get("shadow_panel_evidence")
    if not isinstance(raw_panel_evidence, list) or not raw_panel_evidence:
        raise ExogenousGovernanceError(
            "shadow_manifest.shadow_panel_evidence doit contenir les panels scelles."
        )
    panel_evidence: dict[str, dict[str, object]] = {}
    for panel_index, raw_panel in enumerate(raw_panel_evidence):
        verified_panel = _validated_shadow_panel_evidence(
            raw_panel, index=panel_index, expected_zone=zone
        )
        contract = str(verified_panel["panel_contract_sha256"])
        if contract in panel_evidence:
            raise ExogenousGovernanceError(
                "shadow_manifest: identite panel dupliquee."
            )
        panel_evidence[contract] = verified_panel
    panel_ready = payload.get("shadow_panel_production_ready")
    expected_panel_ready = all(
        bool(item["production_pit_evidence"])
        for item in panel_evidence.values()
    )
    if type(panel_ready) is not bool or panel_ready is not expected_panel_ready:
        raise ExogenousGovernanceError(
            "shadow_manifest.shadow_panel_production_ready incoherent."
        )
    predictions_sha256 = payload.get("predictions_sha256")
    if not isinstance(predictions_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", predictions_sha256
    ) is None:
        raise ExogenousGovernanceError(
            "shadow_manifest.predictions_sha256 doit etre un SHA-256 hexadecimal."
        )
    temporal = payload.get("temporal_attachment_audit")
    if not isinstance(temporal, list) or not temporal:
        raise ExogenousGovernanceError(
            "shadow_manifest.temporal_attachment_audit doit contenir la preuve "
            "horaire prospective."
        )
    if expected_rows is not None and len(temporal) != int(expected_rows):
        raise ExogenousGovernanceError(
            "shadow_manifest.temporal_attachment_audit ne couvre pas exactement "
            f"les predictions observees ({len(temporal)} != {expected_rows})."
        )
    seen: set[tuple[str, str]] = set()
    forecast_times: list[pd.Timestamp] = []
    actual_times: list[pd.Timestamp] = []
    for index, raw_record in enumerate(temporal):
        if not isinstance(raw_record, Mapping):
            raise ExogenousGovernanceError(
                f"shadow_manifest.temporal_attachment_audit[{index}] invalide."
            )
        timestamps: dict[str, pd.Timestamp] = {}
        for key in (
            "delivery_start_utc",
            "forecast_origin_utc",
            "forecast_deadline_utc",
            "forecast_created_at_utc",
            "actual_attached_at_utc",
        ):
            value = pd.to_datetime(raw_record.get(key), utc=True, errors="coerce")
            if pd.isna(value):
                raise ExogenousGovernanceError(
                    f"shadow_manifest.temporal_attachment_audit[{index}].{key} "
                    "invalide."
                )
            timestamps[key] = pd.Timestamp(value)
        expected_deadline = timestamps["forecast_origin_utc"] + pd.Timedelta(hours=4)
        if timestamps["forecast_deadline_utc"] != expected_deadline:
            raise ExogenousGovernanceError(
                "Manifeste live shadow: deadline prospective incoherente."
            )
        forecast_time = timestamps["forecast_created_at_utc"]
        if not (
            timestamps["forecast_origin_utc"]
            <= forecast_time
            <= expected_deadline
        ):
            raise ExogenousGovernanceError(
                "Manifeste live shadow: forecast hors fenêtre prospective D-1."
            )
        actual_time = timestamps["actual_attached_at_utc"]
        if actual_time <= forecast_time:
            raise ExogenousGovernanceError(
                "Manifeste live shadow: actual non strictement posterieure au forecast."
            )
        key = (
            timestamps["delivery_start_utc"].isoformat(),
            timestamps["forecast_origin_utc"].isoformat(),
        )
        if key in seen:
            raise ExogenousGovernanceError(
                "Manifeste live shadow: preuve temporelle horaire dupliquee."
            )
        seen.add(key)
        for hash_key in ("forecast_record_sha256", "resolution_record_sha256"):
            digest = raw_record.get(hash_key)
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ExogenousGovernanceError(
                    f"shadow_manifest.temporal_attachment_audit[{index}].{hash_key} "
                    "doit etre un SHA-256."
                )
        panel_contract = raw_record.get("panel_contract_sha256")
        if not isinstance(panel_contract, str) or panel_contract not in panel_evidence:
            raise ExogenousGovernanceError(
                "Manifeste live shadow: panel_contract_sha256 horaire absent/inconnu."
            )
        panel = panel_evidence[panel_contract]
        if bool(panel["horizon_actuals_present"]):
            raise ExogenousGovernanceError(
                "Manifeste live shadow: le panel de forecast contenait deja les actuals."
            )
        if timestamps["forecast_origin_utc"] != pd.Timestamp(
            panel["forecast_origin_utc"]
        ):
            raise ExogenousGovernanceError(
                "Manifeste live shadow: origine differente du panel scelle."
            )
        delivery_day = (
            timestamps["delivery_start_utc"]
            .tz_convert(str(panel["delivery_timezone"]))
            .date()
            .isoformat()
        )
        if delivery_day != panel["delivery_day"]:
            raise ExogenousGovernanceError(
                "Manifeste live shadow: livraison hors jour du panel scelle."
            )
        if pd.Timestamp(panel["panel_created_at_utc"]) > forecast_time:
            raise ExogenousGovernanceError(
                "Manifeste live shadow: panel cree apres le forecast."
            )
        forecast_times.append(forecast_time)
        actual_times.append(actual_time)
    aggregate_forecast = pd.to_datetime(
        payload.get("forecast_created_at_utc"), utc=True, errors="coerce"
    )
    aggregate_actual = pd.to_datetime(
        payload.get("actual_attached_at_utc"), utc=True, errors="coerce"
    )
    if (
        pd.isna(aggregate_forecast)
        or pd.Timestamp(aggregate_forecast) != min(forecast_times)
        or pd.isna(aggregate_actual)
        or pd.Timestamp(aggregate_actual) != max(actual_times)
    ):
        raise ExogenousGovernanceError(
            "Manifeste live shadow: bornes temporelles agregees incoherentes."
        )
    return payload


def validate_final_shadow_manifest(
    manifest: str | Path | Mapping[str, object],
    *,
    experiment_manifest: Mapping[str, object],
    zone: str,
    expected_rows: int | None = None,
) -> dict[str, object]:
    """Validate prospective evidence for the deployable, corrected pipeline.

    A v3 raw-LoRA shadow remains valid audit material, but it must never satisfy
    the promotion gate.  The v4 manifest carries the same prospective temporal
    proof and additionally binds the operational incumbent, the OOF corrector,
    and the final rolling-365 evidence.  When a path is supplied, all copied
    lineage files are verified as a self-contained snapshot.
    """

    source_directory: Path | None = None
    if isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        source = Path(manifest).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        source_directory = source.parent
        payload = dict(
            _strict_mapping(
                json.loads(source.read_text(encoding="utf-8")), name=str(source)
            )
        )

    canonical_zone = str(zone).strip().upper()
    expected_model = str(
        experiment_manifest.get(
            "experiment_id", experiment_manifest.get("model_id", "")
        )
    ).strip()
    expected: Mapping[str, object] = {
        "format_version": 4,
        "kind": "chronos2_exogenous_final_pipeline_shadow",
        "candidate_model": expected_model,
        "zone": canonical_zone,
        "checkpoint_sha256": experiment_manifest.get("checkpoint_sha256"),
        "schema_sha256": experiment_manifest.get("schema_sha256"),
        "candidate_frozen_before_shadow": True,
        "actuals_attached_after_forecast_freeze": True,
        "prospective_capture_deadline_enforced": True,
        "prospective_capture_deadline_hours": 4,
        "forecast_artifact_checksums_valid": True,
        "storm_used_for_prediction": False,
        "mkonline_used_for_prediction": False,
        "comparison_scope": "paired_operational_final_pipelines",
        "baseline_output_stage": "residual_corrected",
        "candidate_output_stage": "exogenous_residual_corrected",
        "baseline_residual_corrector_applied": True,
        "candidate_residual_corrector_applied": True,
        "residual_corrector_applied": True,
        "paired_same_delivery_hours": True,
        "paired_same_forecast_origins": True,
        "paired_same_observed_actuals": True,
    }
    failures = [
        f"{key}={payload.get(key)!r}, attendu={value!r}"
        for key, value in expected.items()
        if type(payload.get(key)) is not type(value) or payload.get(key) != value
    ]
    if failures:
        raise ExogenousGovernanceError(
            "Manifeste shadow du pipeline final invalide: " + "; ".join(failures)
        )
    experiment_expected: Mapping[str, object] = {
        "production_pipeline_evidence": True,
        "candidate_output_stage": "exogenous_residual_corrected",
    }
    experiment_failures = [
        f"{key}={experiment_manifest.get(key)!r}, attendu={value!r}"
        for key, value in experiment_expected.items()
        if type(experiment_manifest.get(key)) is not type(value)
        or experiment_manifest.get(key) != value
    ]
    detail = experiment_manifest.get("production_pipeline_evidence_detail")
    required_detail: Mapping[str, object] = {
        "comparison_scope": "paired_operational_final_pipelines",
        "baseline_output_stage": "residual_corrected",
        "candidate_output_stage": "exogenous_residual_corrected",
        "baseline_residual_corrector_applied": True,
        "candidate_residual_corrector_applied": True,
        "rolling_evaluation_days": 365,
        "promotion_eligible": True,
    }
    if not isinstance(detail, Mapping):
        experiment_failures.append("production_pipeline_evidence_detail absent")
    else:
        experiment_failures.extend(
            f"production_pipeline_evidence_detail.{key}={detail.get(key)!r}, "
            f"attendu={value!r}"
            for key, value in required_detail.items()
            if type(detail.get(key)) is not type(value) or detail.get(key) != value
        )
    if experiment_failures:
        raise ExogenousGovernanceError(
            "Le shadow final n'est pas lie a un FinalBacktest eligible: "
            + "; ".join(experiment_failures)
        )
    pairing_tolerance = payload.get("actual_pairing_atol_eur_mwh")
    pairing_max_delta = payload.get("actual_pairing_max_delta_eur_mwh")
    if (
        type(pairing_tolerance) is not float
        or pairing_tolerance != 5e-5
        or isinstance(pairing_max_delta, bool)
        or not isinstance(pairing_max_delta, (int, float))
        or not math.isfinite(float(pairing_max_delta))
        or not 0.0 <= float(pairing_max_delta) <= pairing_tolerance
    ):
        raise ExogenousGovernanceError(
            "shadow final: tolerance/ecart d'appariement actual invalide."
        )

    def digest(value: object, label: str) -> str:
        result = str(value).strip().lower() if isinstance(value, str) else ""
        if re.fullmatch(r"[0-9a-f]{64}", result) is None:
            raise ExogenousGovernanceError(
                f"shadow final: {label} doit etre un SHA-256."
            )
        return result

    def reference(
        name: str,
        *,
        rows_required: bool = False,
        expected_stage: str | None = None,
    ) -> tuple[dict[str, object], Path | None]:
        raw_reference = payload.get(name)
        if not isinstance(raw_reference, Mapping):
            raise ExogenousGovernanceError(
                f"shadow final: reference {name} absente."
            )
        result = dict(raw_reference)
        relative_text = result.get("relative_path")
        if not isinstance(relative_text, str) or not relative_text.strip():
            raise ExogenousGovernanceError(
                f"shadow final: {name}.relative_path absent."
            )
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ExogenousGovernanceError(
                f"shadow final: chemin hors preuve interdit pour {name}."
            )
        digest(result.get("sha256"), f"{name}.sha256")
        if rows_required:
            rows = result.get("rows")
            if type(rows) is not int or rows <= 0:
                raise ExogenousGovernanceError(
                    f"shadow final: {name}.rows invalide."
                )
            if expected_rows is not None and rows != int(expected_rows):
                raise ExogenousGovernanceError(
                    f"shadow final: {name}.rows={rows}, attendu={expected_rows}."
                )
        if expected_stage is not None and result.get("output_stage") != expected_stage:
            raise ExogenousGovernanceError(
                f"shadow final: {name}.output_stage invalide."
            )
        resolved: Path | None = None
        if source_directory is not None:
            resolved = (source_directory / relative).resolve()
            if resolved != source_directory and source_directory not in resolved.parents:
                raise ExogenousGovernanceError(
                    f"shadow final: {name} echappe au repertoire scelle."
                )
            if not resolved.is_file() or _sha256(resolved) != result["sha256"]:
                raise ExogenousGovernanceError(
                    f"shadow final: SHA/empreinte divergent pour {name}."
                )
        return result, resolved

    predictions_sha = digest(payload.get("predictions_sha256"), "predictions_sha256")
    observed, observed_path = reference(
        "observed_governance_evidence", rows_required=True
    )
    if observed.get("sha256") != predictions_sha:
        raise ExogenousGovernanceError(
            "shadow final: observed_governance_evidence diverge des predictions."
        )
    raw_evidence, raw_evidence_path = reference(
        "raw_shadow_evidence", rows_required=True
    )
    raw_manifest_reference, raw_manifest_path = reference("raw_shadow_manifest")
    if raw_manifest_reference.get("format_version") != 3:
        raise ExogenousGovernanceError(
            "shadow final: le manifeste source doit etre un shadow brut v3."
        )
    raw_journal_reference, raw_journal_path = reference("raw_shadow_journal")
    if (
        type(raw_journal_reference.get("records")) is not int
        or raw_journal_reference["records"] <= 0
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(raw_journal_reference.get("last_record_sha256", "")),
        )
        is None
    ):
        raise ExogenousGovernanceError(
            "shadow final: provenance du journal brut invalide."
        )
    issued, issued_path = reference("issued_shadow_history")
    if (
        type(issued.get("rows")) is not int
        or issued["rows"] <= 0
        or issued.get("candidate_output_stage")
        != "exogenous_residual_corrected"
        or issued.get("actual_nullable") is not True
        or not isinstance(issued.get("delivery_days"), list)
        or not issued["delivery_days"]
    ):
        raise ExogenousGovernanceError(
            "shadow final: issued_shadow_history invalide."
        )
    incumbent, incumbent_path = reference(
        "paired_incumbent_evidence",
        rows_required=True,
        expected_stage="residual_corrected",
    )
    corrector, _corrector_path = reference("residual_corrector")
    oof_audit, _oof_path = reference("oof_training_audit")
    schema_reference, schema_path = reference("schema")
    experiment_reference, experiment_path = reference(
        "source_experiment_manifest"
    )
    rolling = payload.get("rolling365_final_pipeline_evidence")
    if not isinstance(rolling, Mapping):
        raise ExogenousGovernanceError(
            "shadow final: rolling365_final_pipeline_evidence absent."
        )
    rolling_sha = digest(rolling.get("sha256"), "rolling365.sha256")
    rolling_manifest_sha = digest(
        rolling.get("manifest_sha256"), "rolling365.manifest_sha256"
    )
    experiment_rolling = experiment_manifest.get("evaluation_evidence")
    experiment_final = experiment_manifest.get("final_pipeline_evaluation")
    if not isinstance(experiment_rolling, Mapping) or (
        experiment_rolling.get("sha256") != rolling_sha
    ):
        raise ExogenousGovernanceError(
            "shadow final: preuve rolling365 differente du FinalBacktest."
        )
    if not isinstance(experiment_final, Mapping) or (
        experiment_final.get("sha256") != rolling_manifest_sha
    ):
        raise ExogenousGovernanceError(
            "shadow final: manifeste rolling365 different du FinalBacktest."
        )
    residual_sha = digest(
        corrector.get("sha256"), "residual_corrector.sha256"
    )
    oof_sha = digest(oof_audit.get("sha256"), "oof_training_audit.sha256")
    if residual_sha != experiment_manifest.get("residual_corrector_sha256"):
        raise ExogenousGovernanceError(
            "shadow final: correcteur different du correcteur evalue."
        )
    if oof_sha != experiment_manifest.get("oof_training_audit_sha256"):
        raise ExogenousGovernanceError(
            "shadow final: audit OOF different du FinalBacktest."
        )
    if schema_reference.get("sha256") != experiment_manifest.get("schema_sha256"):
        raise ExogenousGovernanceError(
            "shadow final: schema different du schema evalue."
        )
    source_experiment_sha = digest(
        payload.get("source_experiment_manifest_sha256"),
        "source_experiment_manifest_sha256",
    )
    if experiment_reference.get("sha256") != source_experiment_sha:
        raise ExogenousGovernanceError(
            "shadow final: reference du manifeste d'experience divergente."
        )
    # Keep convenient top-level digests for old readers, but require them to be
    # exact aliases rather than a second, potentially divergent source of truth.
    if payload.get("residual_corrector_sha256") != residual_sha or payload.get(
        "oof_training_audit_sha256"
    ) != oof_sha:
        raise ExogenousGovernanceError(
            "shadow final: alias SHA correcteur/OOF divergent."
        )

    lineage = payload.get("derivation_lineage")
    if not isinstance(lineage, Mapping):
        raise ExogenousGovernanceError("shadow final: derivation_lineage absente.")
    expected_lineage: Mapping[str, object] = {
        "raw_shadow_evidence_sha256": raw_evidence["sha256"],
        "raw_shadow_manifest_sha256": raw_manifest_reference["sha256"],
        "raw_shadow_journal_sha256": raw_journal_reference["sha256"],
        "paired_incumbent_sha256": incumbent["sha256"],
        "issued_shadow_history_sha256": issued["sha256"],
        "residual_corrector_sha256": residual_sha,
        "oof_training_audit_sha256": oof_sha,
        "rolling365_final_evidence_sha256": rolling_sha,
        "rolling365_final_manifest_sha256": rolling_manifest_sha,
        "checkpoint_sha256": experiment_manifest["checkpoint_sha256"],
        "schema_sha256": experiment_manifest["schema_sha256"],
        "final_shadow_predictions_sha256": predictions_sha,
        "rows": observed["rows"],
        "issued_rows": issued["rows"],
        "baseline_output_stage": "residual_corrected",
        "candidate_output_stage": "exogenous_residual_corrected",
    }
    if dict(lineage) != dict(expected_lineage):
        raise ExogenousGovernanceError(
            "shadow final: chaine de derivation SHA incoherente."
        )
    derivation_sha = digest(payload.get("derivation_sha256"), "derivation_sha256")
    if derivation_sha != hashlib.sha256(
        _canonical_json_bytes(expected_lineage)
    ).hexdigest():
        raise ExogenousGovernanceError(
            "shadow final: derivation_sha256 divergent."
        )
    transformation = payload.get("transformation")
    if not isinstance(transformation, Mapping):
        raise ExogenousGovernanceError("shadow final: transformation absente.")
    expected_transformation: Mapping[str, object] = {
        "algorithm": "sealed_residual_corrector_linear_shift_v1",
        "raw_candidate_output_stage": "chronos2_exogenous",
        "output_stage": "exogenous_residual_corrected",
    }
    transformation_failures = [
        f"{key}={transformation.get(key)!r}, attendu={value!r}"
        for key, value in expected_transformation.items()
        if transformation.get(key) != value
    ]
    for key in (
        "mean_shift_eur_mwh",
        "maximum_absolute_shift_eur_mwh",
        "issued_mean_shift_eur_mwh",
        "issued_maximum_absolute_shift_eur_mwh",
    ):
        value = transformation.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            transformation_failures.append(f"{key} invalide")
    if transformation_failures:
        raise ExogenousGovernanceError(
            "shadow final: transformation invalide: "
            + "; ".join(transformation_failures)
        )

    # Reuse the mature prospective proof validator.  Only the raw prediction
    # digest differs; all timestamps/panels must be byte-for-byte inherited.
    proxy = dict(payload)
    proxy["format_version"] = 3
    proxy["kind"] = "chronos2_exogenous_append_only_shadow"
    proxy["candidate_output_stage"] = "chronos2_exogenous"
    proxy["residual_corrector_applied"] = False
    proxy["predictions_sha256"] = raw_evidence["sha256"]
    validate_shadow_manifest(
        proxy,
        experiment_manifest=experiment_manifest,
        zone=canonical_zone,
        expected_rows=expected_rows,
    )

    if source_directory is not None:
        assert observed_path is not None
        assert raw_evidence_path is not None
        assert raw_manifest_path is not None
        assert raw_journal_path is not None
        assert issued_path is not None
        assert incumbent_path is not None
        assert _corrector_path is not None
        assert _oof_path is not None
        assert schema_path is not None
        assert experiment_path is not None
        raw_manifest_payload = validate_shadow_manifest(
            raw_manifest_path,
            experiment_manifest=experiment_manifest,
            zone=canonical_zone,
            expected_rows=expected_rows,
        )
        if raw_manifest_payload.get("predictions_sha256") != raw_evidence["sha256"]:
            raise ExogenousGovernanceError(
                "shadow final: le manifeste brut ne designe pas sa copie de preuve."
            )
        inherited = (
            "forecast_created_at_utc",
            "actual_attached_at_utc",
            "shadow_panel_production_ready",
            "shadow_panel_evidence",
            "temporal_attachment_audit",
        )
        if any(raw_manifest_payload.get(key) != payload.get(key) for key in inherited):
            raise ExogenousGovernanceError(
                "shadow final: preuve prospective modifiee par rapport au journal brut."
            )
        try:
            final_frame = pd.read_csv(observed_path)
            raw_frame = pd.read_csv(raw_evidence_path)
            raw_journal_frame = pd.read_csv(raw_journal_path)
            issued_frame = pd.read_csv(issued_path)
            incumbent_frame = pd.read_csv(incumbent_path)
            corrector_payload = json.loads(
                _corrector_path.read_text(encoding="utf-8")
            )
            schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
            copied_experiment = json.loads(
                experiment_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ExogenousGovernanceError(
                "shadow final: preuve de provenance illisible."
            ) from exc
        if not isinstance(corrector_payload, Mapping) or not isinstance(
            schema_payload, Mapping
        ) or not isinstance(copied_experiment, Mapping):
            raise ExogenousGovernanceError(
                "shadow final: provenance correcteur/schema/manifeste invalide."
            )
        if dict(copied_experiment) != dict(experiment_manifest):
            raise ExogenousGovernanceError(
                "shadow final: copie du manifeste d'experience divergente."
            )
        if len(final_frame) != observed["rows"] or len(raw_frame) != raw_evidence[
            "rows"
        ] or len(incumbent_frame) != incumbent["rows"] or len(
            raw_journal_frame
        ) != raw_journal_reference["records"] or len(issued_frame) != issued["rows"]:
            raise ExogenousGovernanceError(
                "shadow final: nombre de lignes des copies incoherent."
            )
        try:
            from .evaluation import (
                SHADOW_JOURNAL_COLUMNS,
                _canonical_record_hash,
                _validate_shadow_provenance_rows,
            )
        except ImportError as exc:  # pragma: no cover - package invariant
            raise ExogenousGovernanceError(
                "shadow final: validateur du journal indisponible."
            ) from exc
        if tuple(raw_journal_frame.columns) != SHADOW_JOURNAL_COLUMNS:
            raise ExogenousGovernanceError(
                "shadow final: schema du journal brut inconnu."
            )
        for column in (
            "delivery_start_utc",
            "forecast_origin_utc",
            "panel_created_at_utc",
            "captured_at_utc",
        ):
            raw_journal_frame[column] = pd.to_datetime(
                raw_journal_frame[column], utc=True, errors="coerce"
            )
            if raw_journal_frame[column].isna().any():
                raise ExogenousGovernanceError(
                    f"shadow final: timestamp journal invalide dans {column}."
                )
        previous_record = ""
        for index, row in raw_journal_frame.iterrows():
            recorded_previous = (
                ""
                if pd.isna(row["previous_record_sha256"])
                else str(row["previous_record_sha256"])
            )
            if recorded_previous != previous_record or _canonical_record_hash(
                row
            ) != str(row["record_sha256"]):
                raise ExogenousGovernanceError(
                    f"shadow final: chaine du journal rompue a la ligne {index}."
                )
            previous_record = str(row["record_sha256"])
        if previous_record != raw_journal_reference["last_record_sha256"]:
            raise ExogenousGovernanceError(
                "shadow final: dernier record du journal divergent."
            )
        journal_keys = [
            "delivery_start_utc",
            "forecast_origin_utc",
            "item_id",
            "target_column",
        ]
        journal_forecasts = raw_journal_frame.loc[
            raw_journal_frame["record_kind"].eq("forecast")
        ].copy()
        if journal_forecasts.empty or journal_forecasts.duplicated(
            journal_keys
        ).any():
            raise ExogenousGovernanceError(
                "shadow final: records forecast bruts absents/dupliques."
            )
        try:
            panel_evidence = _validate_shadow_provenance_rows(journal_forecasts)
        except Exception as exc:
            raise ExogenousGovernanceError(
                "shadow final: provenance panel des emissions invalide."
            ) from exc
        if not panel_evidence or any(
            evidence.get("production_pit_evidence") is not True
            or evidence.get("horizon_actuals_present") is not False
            for evidence in panel_evidence.values()
        ):
            raise ExogenousGovernanceError(
                "shadow final: emission sans preuve PIT production-ready."
            )
        key_columns = ("delivery_start_utc", "forecast_origin_utc", "actual")
        prediction_columns = tuple(
            f"{prefix}_{quantile}"
            for prefix in ("baseline", "candidate")
            for quantile in ("q10", "q50", "q90")
        )
        if any(
            column not in frame
            for frame in (final_frame, raw_frame)
            for column in (*key_columns, *prediction_columns)
        ):
            raise ExogenousGovernanceError(
                "shadow final: cles/quantiles d'appariement absents."
            )
        incumbent_required = (*key_columns, "residual_corrected__q10", "residual_corrected__q50", "residual_corrected__q90")
        if any(column not in incumbent_frame for column in incumbent_required):
            raise ExogenousGovernanceError(
                "shadow final: incumbent apparie incomplet."
            )
        for column in key_columns[:2]:
            final_values = pd.to_datetime(final_frame[column], utc=True, errors="coerce")
            raw_values = pd.to_datetime(raw_frame[column], utc=True, errors="coerce")
            incumbent_values = pd.to_datetime(
                incumbent_frame[column], utc=True, errors="coerce"
            )
            if final_values.isna().any() or not final_values.equals(raw_values) or not final_values.equals(incumbent_values):
                raise ExogenousGovernanceError(
                    "shadow final: heures/origines non appariees dans les copies."
                )
            final_frame[column] = final_values
            raw_frame[column] = raw_values
            incumbent_frame[column] = incumbent_values
        final_actual = pd.to_numeric(final_frame["actual"], errors="coerce").to_numpy(float)
        raw_actual = pd.to_numeric(raw_frame["actual"], errors="coerce").to_numpy(float)
        incumbent_actual = pd.to_numeric(
            incumbent_frame["actual"], errors="coerce"
        ).to_numpy(float)
        tolerance = payload.get("actual_pairing_atol_eur_mwh")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise ExogenousGovernanceError(
                "shadow final: tolerance d'appariement actual invalide."
            )
        if not (
            np.isfinite(final_actual).all()
            and np.allclose(final_actual, raw_actual, rtol=0.0, atol=float(tolerance))
            and np.allclose(
                final_actual, incumbent_actual, rtol=0.0, atol=float(tolerance)
            )
        ):
            raise ExogenousGovernanceError(
                "shadow final: actuals non appariees dans les copies."
            )
        observed_delta = float(np.max(np.abs(final_actual - incumbent_actual)))
        if not math.isclose(
            observed_delta,
            float(pairing_max_delta),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ExogenousGovernanceError(
                "shadow final: actual_pairing_max_delta_eur_mwh divergent."
            )
        for quantile in ("q10", "q50", "q90"):
            final_baseline = pd.to_numeric(
                final_frame[f"baseline_{quantile}"], errors="coerce"
            ).to_numpy(float)
            incumbent_baseline = pd.to_numeric(
                incumbent_frame[f"residual_corrected__{quantile}"],
                errors="coerce",
            ).to_numpy(float)
            if not np.allclose(
                final_baseline, incumbent_baseline, rtol=0.0, atol=1e-12
            ):
                raise ExogenousGovernanceError(
                    "shadow final: baseline differente de l'incumbent scelle."
                )
        required_corrector: Mapping[str, object] = {
            "schema_version": 1,
            "model_kind": "linear_shift_v1",
            "base_model": "chronos2_exogenous",
            "output_model": "exogenous_residual_corrected",
            "fit_protocol": "blocked_prequential_oof_rolling365",
            "holdout_used_for_fit": False,
            "future_actuals_used_as_features": False,
        }
        corrector_failures = [
            f"{key}={corrector_payload.get(key)!r}, attendu={value!r}"
            for key, value in required_corrector.items()
            if type(corrector_payload.get(key)) is not type(value)
            or corrector_payload.get(key) != value
        ]
        features = corrector_payload.get("feature_columns")
        means = corrector_payload.get("feature_means")
        scales = corrector_payload.get("feature_scales")
        coefficients = corrector_payload.get("coefficients")
        if (
            not isinstance(features, list)
            or not features
            or not isinstance(means, list)
            or not isinstance(scales, list)
            or not isinstance(coefficients, list)
            or not len(features) == len(means) == len(scales) == len(coefficients)
        ):
            corrector_failures.append("parametres vectoriels invalides")
        if corrector_failures:
            raise ExogenousGovernanceError(
                "shadow final: correcteur scelle invalide: "
                + "; ".join(corrector_failures)
            )
        timezone_name = str(schema_payload.get("timezone", "")).strip()
        try:
            ZoneInfo(timezone_name)
            timestamp = pd.DatetimeIndex(
                pd.to_datetime(
                    raw_frame["delivery_start_utc"], utc=True, errors="raise"
                )
            ).tz_convert(timezone_name)
        except Exception as exc:
            raise ExogenousGovernanceError(
                "shadow final: timezone/timeline du schema invalide."
            ) from exc
        derived: Mapping[str, np.ndarray] = {
            "intercept": np.ones(len(raw_frame)),
            "local_hour_sin": np.sin(2 * np.pi * timestamp.hour / 24.0),
            "local_hour_cos": np.cos(2 * np.pi * timestamp.hour / 24.0),
            "local_dow_sin": np.sin(2 * np.pi * timestamp.dayofweek / 7.0),
            "local_dow_cos": np.cos(2 * np.pi * timestamp.dayofweek / 7.0),
            "local_doy_sin": np.sin(
                2 * np.pi * (timestamp.dayofyear - 1) / 365.2425
            ),
            "local_doy_cos": np.cos(
                2 * np.pi * (timestamp.dayofyear - 1) / 365.2425
            ),
            "is_weekend": np.asarray(timestamp.dayofweek >= 5, dtype=float),
        }
        try:
            design = np.column_stack([derived[str(feature)] for feature in features])
            means_array = np.asarray(means, dtype=float)
            scales_array = np.asarray(scales, dtype=float)
            coefficients_array = np.asarray(coefficients, dtype=float)
            clip = float(corrector_payload["maximum_absolute_shift_eur_mwh"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExogenousGovernanceError(
                "shadow final: feature/parametre de correction non reproductible."
            ) from exc
        if not (
            np.isfinite(design).all()
            and np.isfinite(means_array).all()
            and np.isfinite(scales_array).all()
            and bool((scales_array > 0.0).all())
            and np.isfinite(coefficients_array).all()
            and math.isfinite(clip)
            and 0.0 < clip <= 50.0
        ):
            raise ExogenousGovernanceError(
                "shadow final: parametres de correction non finis/invalides."
            )
        shift = np.clip(
            ((design - means_array) / scales_array) @ coefficients_array,
            -clip,
            clip,
        )
        for quantile in ("q10", "q50", "q90"):
            raw_candidate = pd.to_numeric(
                raw_frame[f"candidate_{quantile}"], errors="coerce"
            ).to_numpy(float)
            final_candidate = pd.to_numeric(
                final_frame[f"candidate_{quantile}"], errors="coerce"
            ).to_numpy(float)
            expected_candidate = raw_candidate + shift
            if not (
                np.isfinite(final_candidate).all()
                and np.allclose(
                    final_candidate,
                    expected_candidate,
                    rtol=0.0,
                    atol=1e-10,
                )
            ):
                raise ExogenousGovernanceError(
                    "shadow final: candidat different de LoRA + correcteur scelle."
                )
        if not math.isclose(
            float(transformation["mean_shift_eur_mwh"]),
            float(np.mean(shift)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(transformation["maximum_absolute_shift_eur_mwh"]),
            float(np.max(np.abs(shift))),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ExogenousGovernanceError(
                "shadow final: statistiques de transformation divergentes."
            )
        issued_columns = (
            "delivery_start_utc",
            "forecast_origin_utc",
            "actual",
            "candidate_q10",
            "candidate_q50",
            "candidate_q90",
            "item_id",
            "target_column",
            "checkpoint_sha256",
            "input_contract_sha256",
            "panel_contract_sha256",
            "forecast_created_at_utc",
            "forecast_record_sha256",
        )
        if tuple(issued_frame.columns) != issued_columns:
            raise ExogenousGovernanceError(
                "shadow final: schema exact de issued_shadow_history invalide."
            )
        for column in (
            "delivery_start_utc",
            "forecast_origin_utc",
            "forecast_created_at_utc",
        ):
            issued_frame[column] = pd.to_datetime(
                issued_frame[column], utc=True, errors="coerce"
            )
            if issued_frame[column].isna().any():
                raise ExogenousGovernanceError(
                    f"shadow final: timestamp issued invalide dans {column}."
                )
        issued_frame = issued_frame.sort_values(
            "delivery_start_utc", kind="stable"
        ).reset_index(drop=True)
        if issued_frame.duplicated(
            ["delivery_start_utc", "forecast_origin_utc"]
        ).any():
            raise ExogenousGovernanceError(
                "shadow final: heure/origine issued dupliquee."
            )
        if (
            issued_frame["item_id"].astype(str).str.strip().str.upper().nunique()
            != 1
            or issued_frame["item_id"]
            .astype(str)
            .str.strip()
            .str.upper()
            .iloc[0]
            != canonical_zone
            or issued_frame["target_column"].astype(str).str.strip().nunique()
            != 1
        ):
            raise ExogenousGovernanceError(
                "shadow final: zone/item/target de l'historique emis invalide."
            )
        for column in (
            "checkpoint_sha256",
            "input_contract_sha256",
            "panel_contract_sha256",
            "forecast_record_sha256",
        ):
            if not issued_frame[column].astype(str).str.fullmatch(
                r"[0-9a-f]{64}"
            ).all():
                raise ExogenousGovernanceError(
                    f"shadow final: hash issued invalide dans {column}."
                )
        if not issued_frame["checkpoint_sha256"].eq(
            experiment_manifest["checkpoint_sha256"]
        ).all():
            raise ExogenousGovernanceError(
                "shadow final: checkpoint issued divergent."
            )
        deadline = issued_frame["forecast_origin_utc"] + pd.Timedelta(hours=4)
        if not bool(
            (
                (issued_frame["forecast_created_at_utc"] >= issued_frame["forecast_origin_utc"])
                & (issued_frame["forecast_created_at_utc"] <= deadline)
            ).all()
        ):
            raise ExogenousGovernanceError(
                "shadow final: forecast issued hors fenetre prospective."
            )
        issued_delivery = pd.DatetimeIndex(issued_frame["delivery_start_utc"])
        expected_days = list(
            dict.fromkeys(issued_delivery.tz_convert(timezone_name).date.astype(str))
        )
        if issued.get("delivery_days") != expected_days or (
            issued.get("first_delivery_utc") != issued_delivery[0].isoformat()
            or issued.get("last_delivery_utc") != issued_delivery[-1].isoformat()
        ):
            raise ExogenousGovernanceError(
                "shadow final: bornes/jours de issued_shadow_history divergents."
            )
        issued_quantile_columns = [
            f"candidate_{quantile}" for quantile in ("q10", "q50", "q90")
        ]
        issued_numeric = issued_frame[issued_quantile_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        issued_quantiles = issued_numeric.to_numpy(float)
        if not np.isfinite(issued_quantiles).all() or bool(
            (
                (issued_quantiles[:, 0] > issued_quantiles[:, 1])
                | (issued_quantiles[:, 1] > issued_quantiles[:, 2])
            ).any()
        ):
            raise ExogenousGovernanceError(
                "shadow final: quantiles issued absents/non finis/croises."
            )

        raw_latest = raw_journal_frame.drop_duplicates(journal_keys, keep="last")
        raw_forecast_for_join = journal_forecasts.loc[
            :,
            [
                *journal_keys,
                "candidate_q10",
                "candidate_q50",
                "candidate_q90",
                "checkpoint_sha256",
                "input_contract_sha256",
                "panel_contract_sha256",
                "captured_at_utc",
                "record_sha256",
            ],
        ].merge(
            raw_latest.loc[:, [*journal_keys, "actual"]],
            on=journal_keys,
            how="left",
            validate="one_to_one",
        )
        issued_join = issued_frame.merge(
            raw_forecast_for_join,
            on=journal_keys,
            how="left",
            suffixes=("_issued", "_raw"),
            validate="one_to_one",
        )
        if len(issued_join) != len(issued_frame) or issued_join[
            "record_sha256"
        ].isna().any():
            raise ExogenousGovernanceError(
                "shadow final: historique emis sans record forecast brut."
            )
        metadata_pairs = (
            ("checkpoint_sha256_issued", "checkpoint_sha256_raw"),
            ("input_contract_sha256_issued", "input_contract_sha256_raw"),
            ("panel_contract_sha256_issued", "panel_contract_sha256_raw"),
            ("forecast_record_sha256", "record_sha256"),
        )
        if any(
            not issued_join[left].astype(str).equals(
                issued_join[right].astype(str)
            )
            for left, right in metadata_pairs
        ) or not pd.DatetimeIndex(issued_join["forecast_created_at_utc"]).equals(
            pd.DatetimeIndex(issued_join["captured_at_utc"])
        ):
            raise ExogenousGovernanceError(
                "shadow final: provenance issued differente du journal brut."
            )
        issued_actual = pd.to_numeric(
            issued_join["actual_issued"], errors="coerce"
        ).to_numpy(float)
        raw_latest_actual = pd.to_numeric(
            issued_join["actual_raw"], errors="coerce"
        ).to_numpy(float)
        actual_equal = (np.isnan(issued_actual) & np.isnan(raw_latest_actual)) | np.isclose(
            issued_actual,
            raw_latest_actual,
            rtol=0.0,
            atol=5e-5,
            equal_nan=False,
        )
        if not bool(actual_equal.all()):
            raise ExogenousGovernanceError(
                "shadow final: actual nullable issued divergent du journal."
            )
        issued_timestamp = issued_delivery.tz_convert(timezone_name)
        issued_derived: Mapping[str, np.ndarray] = {
            "intercept": np.ones(len(issued_frame)),
            "local_hour_sin": np.sin(2 * np.pi * issued_timestamp.hour / 24.0),
            "local_hour_cos": np.cos(2 * np.pi * issued_timestamp.hour / 24.0),
            "local_dow_sin": np.sin(2 * np.pi * issued_timestamp.dayofweek / 7.0),
            "local_dow_cos": np.cos(2 * np.pi * issued_timestamp.dayofweek / 7.0),
            "local_doy_sin": np.sin(
                2 * np.pi * (issued_timestamp.dayofyear - 1) / 365.2425
            ),
            "local_doy_cos": np.cos(
                2 * np.pi * (issued_timestamp.dayofyear - 1) / 365.2425
            ),
            "is_weekend": np.asarray(
                issued_timestamp.dayofweek >= 5, dtype=float
            ),
        }
        try:
            issued_design = np.column_stack(
                [issued_derived[str(feature)] for feature in features]
            )
        except KeyError as exc:
            raise ExogenousGovernanceError(
                "shadow final: correcteur issued non reproductible."
            ) from exc
        issued_shift = np.clip(
            ((issued_design - means_array) / scales_array) @ coefficients_array,
            -clip,
            clip,
        )
        for quantile in ("q10", "q50", "q90"):
            raw_issued_candidate = pd.to_numeric(
                issued_join[f"candidate_{quantile}_raw"], errors="coerce"
            ).to_numpy(float)
            expected_issued_candidate = raw_issued_candidate + issued_shift
            actual_issued_candidate = pd.to_numeric(
                issued_join[f"candidate_{quantile}_issued"], errors="coerce"
            ).to_numpy(float)
            if not np.allclose(
                actual_issued_candidate,
                expected_issued_candidate,
                rtol=0.0,
                atol=1e-10,
            ):
                raise ExogenousGovernanceError(
                    "shadow final: candidat issued different de LoRA + correcteur."
                )
        if not math.isclose(
            float(transformation.get("issued_mean_shift_eur_mwh")),
            float(np.mean(issued_shift)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(transformation.get("issued_maximum_absolute_shift_eur_mwh")),
            float(np.max(np.abs(issued_shift))),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ExogenousGovernanceError(
                "shadow final: statistiques de correction issued divergentes."
            )
        overlap = final_frame.merge(
            issued_frame,
            on=["delivery_start_utc", "forecast_origin_utc"],
            how="left",
            suffixes=("_observed", "_issued"),
            validate="one_to_one",
        )
        if len(overlap) != len(final_frame) or overlap[
            "candidate_q50_issued"
        ].isna().any():
            raise ExogenousGovernanceError(
                "shadow final: issued ne couvre pas toute la preuve observee."
            )
        for column in ("actual", "candidate_q10", "candidate_q50", "candidate_q90"):
            left = pd.to_numeric(
                overlap[f"{column}_observed"], errors="coerce"
            ).to_numpy(float)
            right = pd.to_numeric(
                overlap[f"{column}_issued"], errors="coerce"
            ).to_numpy(float)
            if not np.allclose(left, right, rtol=0.0, atol=1e-10):
                raise ExogenousGovernanceError(
                    f"shadow final: overlap observed/issued divergent sur {column}."
                )
    return payload


def _parse_day(value: str | date | None, *, name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ExogenousGovernanceError(f"{name} doit respecter YYYY-MM-DD.") from exc


def _expected_utc_hours(day_local: date, timezone_name: str) -> pd.DatetimeIndex:
    tz = ZoneInfo(timezone_name)
    start = pd.Timestamp(datetime.combine(day_local, datetime.min.time()), tz=tz)
    end = pd.Timestamp(
        datetime.combine(day_local + timedelta(days=1), datetime.min.time()), tz=tz
    )
    return pd.date_range(
        start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left"
    )


def _finite_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        bad = int((~np.isfinite(converted.to_numpy(dtype=float))).sum())
        if bad:
            raise ExogenousGovernanceError(
                f"{column}: {bad} valeur(s) absente(s) ou non finie(s)."
            )
        frame[column] = converted.astype(float)


def _normalise_window(
    raw: pd.DataFrame,
    *,
    timezone_name: str,
    required_days: int,
    end_day: str | date | None,
    phase: str,
) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - set(raw.columns))
    if missing:
        raise ExogenousGovernanceError(
            f"{phase}: colonnes obligatoires absentes: {', '.join(missing)}."
        )
    frame = raw.loc[:, list(REQUIRED_PREDICTION_COLUMNS)].copy()
    delivery = pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="coerce")
    origin = pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="coerce")
    if delivery.isna().any() or origin.isna().any():
        raise ExogenousGovernanceError(f"{phase}: timestamps UTC invalides ou absents.")
    if delivery.duplicated().any():
        raise ExogenousGovernanceError(f"{phase}: delivery_start_utc dupliques.")
    aligned = (
        delivery.dt.minute.eq(0)
        & delivery.dt.second.eq(0)
        & delivery.dt.microsecond.eq(0)
    )
    if not bool(aligned.all()):
        raise ExogenousGovernanceError(f"{phase}: timestamps non alignes a l'heure.")
    frame["delivery_start_utc"] = delivery
    frame["forecast_origin_utc"] = origin
    _finite_numeric(frame, REQUIRED_PREDICTION_COLUMNS[2:])
    frame = frame.sort_values("delivery_start_utc").reset_index(drop=True)

    local = frame["delivery_start_utc"].dt.tz_convert(timezone_name)
    frame["delivery_day_local"] = local.dt.date
    frame["local_hour"] = local.dt.hour.astype(int)
    available_days = sorted(set(frame["delivery_day_local"]))
    selected_end = _parse_day(end_day, name=f"{phase}.end_day")
    if selected_end is None:
        if not available_days:
            raise ExogenousGovernanceError(f"{phase}: aucune prediction.")
        selected_end = available_days[-1]
    selected_start = selected_end - timedelta(days=required_days - 1)
    required = [selected_start + timedelta(days=offset) for offset in range(required_days)]
    selected = frame.loc[frame["delivery_day_local"].isin(required)].copy()
    actual_days = sorted(set(selected["delivery_day_local"]))
    if actual_days != required:
        absent = [item.isoformat() for item in required if item not in set(actual_days)]
        raise ExogenousGovernanceError(
            f"{phase}: fenetre non consecutive/incomplete; jours absents={absent[:10]}."
        )
    for day_local in required:
        observed = pd.DatetimeIndex(
            selected.loc[
                selected["delivery_day_local"].eq(day_local), "delivery_start_utc"
            ]
        )
        expected = _expected_utc_hours(day_local, timezone_name)
        if not observed.equals(expected):
            missing_hours = expected.difference(observed)
            extra_hours = observed.difference(expected)
            raise ExogenousGovernanceError(
                f"{phase}/{day_local}: grille DST incomplete; "
                f"missing={len(missing_hours)}, extra={len(extra_hours)}."
            )

    crossings = np.zeros(len(selected), dtype=bool)
    for prefix in ("baseline", "candidate"):
        crossings |= ~(
            selected[f"{prefix}_q10"].le(selected[f"{prefix}_q50"])
            & selected[f"{prefix}_q50"].le(selected[f"{prefix}_q90"])
        ).to_numpy()
    if bool(crossings.any()):
        raise ExogenousGovernanceError(
            f"{phase}: {int(crossings.sum())} croisement(s) de quantiles."
        )

    tz = ZoneInfo(timezone_name)
    cutoffs = pd.Series(
        [
            pd.Timestamp(
                datetime.combine(
                    day_local - timedelta(days=1),
                    datetime.min.replace(hour=8).time(),
                ),
                tz=tz,
            )
            for day_local in selected["delivery_day_local"]
        ],
        index=selected.index,
    ).dt.tz_convert("UTC")
    violations = selected["forecast_origin_utc"].gt(cutoffs)
    if bool(violations.any()):
        first = int(np.flatnonzero(violations.to_numpy())[0])
        raise ExogenousGovernanceError(
            f"{phase}: {int(violations.sum())} origine(s) posterieure(s) au "
            f"cutoff D-1 08:00; premiere ligne={first}."
        )
    selected.attrs["causal_origin_violations"] = 0
    selected.attrs["quantile_crossings"] = 0
    return selected.reset_index(drop=True)


def _safe_relative_delta(candidate: float, baseline: float) -> float:
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise ExogenousGovernanceError("La metrique baseline doit etre strictement positive.")
    return float((candidate - baseline) / baseline)


def _pinball(actual: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = actual - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def _circular_block_bootstrap(
    gains: np.ndarray,
    *,
    samples: int,
    seed: int,
    block_days: int,
    confidence: float,
) -> tuple[float, float]:
    if gains.ndim != 1 or len(gains) < 2:
        raise ExogenousGovernanceError("Deux jours apparies minimum sont requis.")
    rng = np.random.default_rng(int(seed))
    block = min(int(block_days), len(gains))
    block_count = int(math.ceil(len(gains) / block))
    estimates = np.empty(int(samples), dtype=float)
    batch_size = 1_000
    offsets = np.arange(block, dtype=int)
    for start in range(0, int(samples), batch_size):
        stop = min(start + batch_size, int(samples))
        starts = rng.integers(0, len(gains), size=(stop - start, block_count))
        indices = (starts[..., None] + offsets) % len(gains)
        draws = gains[indices.reshape(stop - start, -1)[:, : len(gains)]]
        estimates[start:stop] = draws.mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def _window_metrics(
    frame: pd.DataFrame,
    *,
    phase: str,
    policy: GovernancePolicy,
) -> WindowEvaluation:
    actual = frame["actual"].to_numpy(dtype=float)
    baseline = frame["baseline_q50"].to_numpy(dtype=float)
    candidate = frame["candidate_q50"].to_numpy(dtype=float)
    baseline_abs = np.abs(actual - baseline)
    candidate_abs = np.abs(actual - candidate)
    baseline_mae = float(baseline_abs.mean())
    candidate_mae = float(candidate_abs.mean())
    gain = baseline_mae - candidate_mae

    work = frame[["delivery_day_local", "local_hour"]].copy()
    work["baseline_abs"] = baseline_abs
    work["candidate_abs"] = candidate_abs
    work["actual"] = actual
    work["baseline"] = baseline
    work["candidate"] = candidate
    daily = work.groupby("delivery_day_local", sort=True).agg(
        baseline_mae=("baseline_abs", "mean"),
        candidate_mae=("candidate_abs", "mean"),
        actual_mean=("actual", "mean"),
        baseline_mean=("baseline", "mean"),
        candidate_mean=("candidate", "mean"),
    )
    daily["gain"] = daily["baseline_mae"] - daily["candidate_mae"]
    gains = daily["gain"].to_numpy(dtype=float)
    split = len(gains) // 2
    lower, upper = _circular_block_bootstrap(
        gains,
        samples=policy.bootstrap_samples,
        seed=policy.bootstrap_seed + (1 if phase == "shadow" else 0),
        block_days=policy.bootstrap_block_days,
        confidence=policy.bootstrap_confidence,
    )

    peak = work["local_hour"].isin(policy.peak_local_hours).to_numpy()
    peak_base = float(baseline_abs[peak].mean())
    peak_candidate = float(candidate_abs[peak].mean())
    tail_threshold = float(
        np.quantile(np.abs(actual), policy.tail_actual_absolute_quantile)
    )
    tail = np.abs(actual) >= tail_threshold
    tail_base = float(baseline_abs[tail].mean())
    tail_candidate = float(candidate_abs[tail].mean())

    daily_base = float(
        np.abs(daily["actual_mean"] - daily["baseline_mean"]).mean()
    )
    daily_candidate = float(
        np.abs(daily["actual_mean"] - daily["candidate_mean"]).mean()
    )

    baseline_losses: list[float] = []
    candidate_losses: list[float] = []
    for label, quantile in (("q10", 0.10), ("q50", 0.50), ("q90", 0.90)):
        baseline_losses.append(
            _pinball(actual, frame[f"baseline_{label}"].to_numpy(float), quantile)
        )
        candidate_losses.append(
            _pinball(actual, frame[f"candidate_{label}"].to_numpy(float), quantile)
        )
    baseline_pinball = float(np.mean(baseline_losses))
    candidate_pinball = float(np.mean(candidate_losses))
    baseline_coverage = float(
        np.mean(
            (actual >= frame["baseline_q10"].to_numpy(float))
            & (actual <= frame["baseline_q90"].to_numpy(float))
        )
    )
    candidate_coverage = float(
        np.mean(
            (actual >= frame["candidate_q10"].to_numpy(float))
            & (actual <= frame["candidate_q90"].to_numpy(float))
        )
    )
    baseline_coverage_error = abs(
        baseline_coverage - policy.interval_nominal_coverage
    )
    candidate_coverage_error = abs(
        candidate_coverage - policy.interval_nominal_coverage
    )

    days = sorted(set(frame["delivery_day_local"]))
    return WindowEvaluation(
        phase=phase,
        start_day_local=days[0].isoformat(),
        end_day_local=days[-1].isoformat(),
        days=len(days),
        hours=len(frame),
        baseline_mae=baseline_mae,
        candidate_mae=candidate_mae,
        mae_gain_eur_mwh=float(gain),
        relative_mae_gain=float(gain / baseline_mae),
        baseline_bias=float(np.mean(baseline - actual)),
        candidate_bias=float(np.mean(candidate - actual)),
        daily_win_rate=float(np.mean(daily["candidate_mae"] < daily["baseline_mae"])),
        first_half_gain_eur_mwh=float(gains[:split].mean()),
        second_half_gain_eur_mwh=float(gains[split:].mean()),
        bootstrap_ci_lower_eur_mwh=lower,
        bootstrap_ci_upper_eur_mwh=upper,
        peak_baseline_mae=peak_base,
        peak_candidate_mae=peak_candidate,
        peak_relative_degradation=_safe_relative_delta(peak_candidate, peak_base),
        tail_threshold_abs_actual=tail_threshold,
        tail_baseline_mae=tail_base,
        tail_candidate_mae=tail_candidate,
        tail_relative_degradation=_safe_relative_delta(tail_candidate, tail_base),
        daily_mean_baseline_mae=daily_base,
        daily_mean_candidate_mae=daily_candidate,
        daily_mean_relative_degradation=_safe_relative_delta(
            daily_candidate, daily_base
        ),
        baseline_pinball=baseline_pinball,
        candidate_pinball=candidate_pinball,
        pinball_relative_degradation=_safe_relative_delta(
            candidate_pinball, baseline_pinball
        ),
        baseline_interval_coverage=baseline_coverage,
        candidate_interval_coverage=candidate_coverage,
        baseline_interval_coverage_error=baseline_coverage_error,
        candidate_interval_coverage_error=candidate_coverage_error,
        interval_coverage_error_increase=float(
            candidate_coverage_error - baseline_coverage_error
        ),
        causal_origin_violations=int(frame.attrs.get("causal_origin_violations", 0)),
        quantile_crossings=int(frame.attrs.get("quantile_crossings", 0)),
    )


def _gate_check(
    checks: dict[str, dict[str, object]],
    reasons: list[str],
    *,
    name: str,
    observed: float | int | bool,
    threshold: float | int | bool,
    operator: str,
    passes: bool,
    failure: str,
) -> None:
    checks[name] = {
        "passes": bool(passes),
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
    }
    if not passes:
        reasons.append(failure)


def _evaluate_gate(
    metrics: WindowEvaluation,
    *,
    policy: GovernancePolicy,
    phase: str,
) -> GateResult:
    shadow = phase == "shadow"
    minimum_gain = (
        policy.shadow_minimum_mae_gain_eur_mwh
        if shadow
        else policy.minimum_mae_gain_eur_mwh
    )
    minimum_relative = (
        policy.shadow_minimum_relative_mae_gain
        if shadow
        else policy.minimum_relative_mae_gain
    )
    minimum_win_rate = (
        policy.shadow_minimum_daily_win_rate
        if shadow
        else policy.minimum_daily_win_rate
    )
    require_lower = (
        policy.shadow_require_bootstrap_lower_bound_positive
        if shadow
        else policy.require_bootstrap_lower_bound_positive
    )
    checks: dict[str, dict[str, object]] = {}
    reasons: list[str] = []
    _gate_check(
        checks,
        reasons,
        name="mae_gain",
        observed=metrics.mae_gain_eur_mwh,
        threshold=minimum_gain,
        operator=">=",
        passes=metrics.mae_gain_eur_mwh >= minimum_gain,
        failure=f"gain MAE < {minimum_gain:.3f} EUR/MWh",
    )
    _gate_check(
        checks,
        reasons,
        name="relative_mae_gain",
        observed=metrics.relative_mae_gain,
        threshold=minimum_relative,
        operator=">=",
        passes=metrics.relative_mae_gain >= minimum_relative,
        failure=f"gain MAE relatif < {minimum_relative:.3%}",
    )
    _gate_check(
        checks,
        reasons,
        name="daily_win_rate",
        observed=metrics.daily_win_rate,
        threshold=minimum_win_rate,
        operator=">=",
        passes=metrics.daily_win_rate >= minimum_win_rate,
        failure=f"taux de jours gagnes < {minimum_win_rate:.1%}",
    )
    if policy.require_positive_chronological_halves:
        _gate_check(
            checks,
            reasons,
            name="first_half_gain_positive",
            observed=metrics.first_half_gain_eur_mwh,
            threshold=0.0,
            operator=">",
            passes=metrics.first_half_gain_eur_mwh > 0.0,
            failure="gain non positif dans la premiere moitie chronologique",
        )
        _gate_check(
            checks,
            reasons,
            name="second_half_gain_positive",
            observed=metrics.second_half_gain_eur_mwh,
            threshold=0.0,
            operator=">",
            passes=metrics.second_half_gain_eur_mwh > 0.0,
            failure="gain non positif dans la seconde moitie chronologique",
        )
    if require_lower:
        _gate_check(
            checks,
            reasons,
            name="bootstrap_lower_positive",
            observed=metrics.bootstrap_ci_lower_eur_mwh,
            threshold=0.0,
            operator=">",
            passes=metrics.bootstrap_ci_lower_eur_mwh > 0.0,
            failure="borne basse bootstrap bloc 95% non positive",
        )
    for name, observed, maximum, label in (
        (
            "peak_mae_non_degradation",
            metrics.peak_relative_degradation,
            policy.maximum_peak_mae_relative_degradation,
            "MAE des heures de pointe",
        ),
        (
            "tail_mae_non_degradation",
            metrics.tail_relative_degradation,
            policy.maximum_tail_mae_relative_degradation,
            "MAE des prix extremes",
        ),
        (
            "daily_mean_mae_non_degradation",
            metrics.daily_mean_relative_degradation,
            policy.maximum_daily_mean_mae_relative_degradation,
            "erreur du prix moyen journalier",
        ),
    ):
        _gate_check(
            checks,
            reasons,
            name=name,
            observed=observed,
            threshold=maximum,
            operator="<=",
            passes=observed <= maximum,
            failure=f"{label} degradee de plus de {maximum:.1%}",
        )
    if policy.require_probabilistic_metrics:
        _gate_check(
            checks,
            reasons,
            name="pinball_non_degradation",
            observed=metrics.pinball_relative_degradation,
            threshold=policy.maximum_pinball_relative_degradation,
            operator="<=",
            passes=(
                metrics.pinball_relative_degradation
                <= policy.maximum_pinball_relative_degradation
            ),
            failure="score pinball probabiliste degrade",
        )
        _gate_check(
            checks,
            reasons,
            name="interval_calibration_non_degradation",
            observed=metrics.interval_coverage_error_increase,
            threshold=policy.maximum_interval_coverage_error_increase,
            operator="<=",
            passes=(
                metrics.interval_coverage_error_increase
                <= policy.maximum_interval_coverage_error_increase
            ),
            failure="calibration q10-q90 trop degradee",
        )
    return GateResult(
        phase=phase,
        passes=not reasons,
        checks=checks,
        reasons=tuple(reasons),
    )


def evaluate_promotion(
    *,
    rolling_predictions: pd.DataFrame,
    experiment_manifest: str | Path | Mapping[str, object],
    zone: str,
    policy: GovernancePolicy,
    rolling_end_day: str | date | None = None,
    shadow_predictions: pd.DataFrame | None = None,
    shadow_manifest: str | Path | Mapping[str, object] | None = None,
    shadow_end_day: str | date | None = None,
    shadow_epoch_directory: str | Path | None = None,
) -> PromotionDecision:
    """Recompute every gate and return reject/shadow/promote.

    ``shadow`` means the rolling-365 candidate is eligible for prospective
    shadow execution but has not yet accumulated the required live evidence.
    """

    policy.validate()
    canonical_zone = str(zone).strip().upper()
    if not re.fullmatch(r"[A-Z]{2,8}", canonical_zone):
        raise ExogenousGovernanceError(f"Zone invalide: {zone!r}.")
    manifest = validate_experiment_manifest(
        experiment_manifest, zone=canonical_zone
    )
    label_binding = manifest.get("evaluation_label_binding")
    if (
        isinstance(label_binding, Mapping)
        and label_binding.get("allow_unresolved_final_evaluation_day") is True
    ):
        if shadow_epoch_directory is None:
            raise ExogenousGovernanceError(
                "Gouvernance two-phase refusee: shadow_epoch_directory est "
                "obligatoire et doit etre revalide apres le FinalBacktest."
            )
        if isinstance(experiment_manifest, Mapping):
            raise ExogenousGovernanceError(
                "Gouvernance two-phase refusee: un chemin canonique vers "
                "experiment_manifest.json est obligatoire."
            )
        experiment_source = Path(experiment_manifest).expanduser().resolve()
        if experiment_source.name != "experiment_manifest.json":
            raise ExogenousGovernanceError(
                "Gouvernance two-phase refusee: le manifeste d'experience doit "
                "etre le fichier canonique experiment_manifest.json."
            )
        # Local import avoids a module cycle: shadow_epoch itself relies on the
        # generic governance manifest validator above.
        from .shadow_epoch import (  # pylint: disable=import-outside-toplevel
            ShadowEpochError,
            validate_finalisation_against_epoch,
        )

        try:
            validate_finalisation_against_epoch(
                shadow_epoch_directory,
                run_directory=experiment_source.parent,
            )
        except ShadowEpochError as exc:
            raise ExogenousGovernanceError(
                "Gouvernance two-phase refusee: l'epoch/finalisation n'est pas "
                "valide."
            ) from exc
    candidate_name = manifest.get("experiment_id", manifest["model_id"])
    if not isinstance(candidate_name, str) or not candidate_name.strip():
        raise ExogenousGovernanceError(
            "experiment_manifest.experiment_id doit etre une chaine non vide."
        )
    model_id = candidate_name.strip()
    production_pit_evidence = bool(manifest["production_pit_evidence"])
    production_pipeline_evidence = bool(
        manifest["production_pipeline_evidence"]
    )
    shadow_panel_production_ready = True
    shadow_evidence_verified = False
    verified_shadow_manifest: dict[str, object] | None = None
    if shadow_predictions is None and shadow_manifest is not None:
        raise ExogenousGovernanceError(
            "shadow_manifest fourni sans shadow_predictions."
        )
    if shadow_predictions is not None:
        if shadow_manifest is None:
            raise ExogenousGovernanceError(
                "Un shadow_manifest scelle est obligatoire avec les predictions live."
            )
        verified_shadow_manifest = validate_final_shadow_manifest(
            shadow_manifest,
            experiment_manifest=manifest,
            zone=canonical_zone,
            expected_rows=len(shadow_predictions),
        )
        shadow_panel_production_ready = bool(
            verified_shadow_manifest["shadow_panel_production_ready"]
        )
        production_pit_evidence = bool(
            production_pit_evidence and shadow_panel_production_ready
        )
        shadow_evidence_verified = True
    production_gate_passes = (
        production_pit_evidence and production_pipeline_evidence
    )
    rolling = _normalise_window(
        rolling_predictions,
        timezone_name=policy.timezone,
        required_days=policy.rolling_evaluation_days,
        end_day=rolling_end_day,
        phase="rolling365",
    )
    rolling_metrics = _window_metrics(rolling, phase="rolling365", policy=policy)
    rolling_gate = _evaluate_gate(
        rolling_metrics, policy=policy, phase="rolling365"
    )
    shadow_metrics: WindowEvaluation | None = None
    shadow_gate: GateResult | None = None
    reasons: list[str] = []
    if not rolling_gate.passes:
        decision = "reject"
        reasons.extend(f"rolling365: {reason}" for reason in rolling_gate.reasons)
    elif shadow_predictions is None and policy.shadow_required_for_promotion:
        decision = "shadow"
        reasons.append(
            f"gate rolling365 passee; {policy.shadow_evaluation_days} jours live "
            "shadow scelles sont requis avant promotion"
        )
        if not production_pit_evidence:
            reasons.append(
                "production_pit_evidence=false: promotion interdite; les donnees "
                "JAO historiques actuelles restent reservees au POC/shadow"
            )
        if not production_pipeline_evidence:
            reasons.append(
                "production_pipeline_evidence=false: le pipeline final avec "
                "correcteur residuel OOF et l'incumbent autonome n'ont pas "
                "encore ete compares"
            )
    elif shadow_predictions is not None:
        shadow = _normalise_window(
            shadow_predictions,
            timezone_name=policy.timezone,
            required_days=policy.shadow_evaluation_days,
            end_day=shadow_end_day,
            phase="shadow",
        )
        shadow_metrics = _window_metrics(shadow, phase="shadow", policy=policy)
        rolling_end = date.fromisoformat(rolling_metrics.end_day_local)
        shadow_start = date.fromisoformat(shadow_metrics.start_day_local)
        expected_shadow_start = rolling_end + timedelta(days=1)
        if shadow_start != expected_shadow_start:
            raise ExogenousGovernanceError(
                "Continuite rolling365/shadow invalide: le shadow doit commencer "
                f"le {expected_shadow_start.isoformat()}, immediatement apres la "
                f"fin rolling365 {rolling_end.isoformat()}, mais commence le "
                f"{shadow_start.isoformat()}. Reconstruisez un holdout actualise "
                "ou fournissez un bridge causal scelle."
            )
        assert verified_shadow_manifest is not None
        issued_reference = verified_shadow_manifest.get("issued_shadow_history")
        issued_days = (
            issued_reference.get("delivery_days")
            if isinstance(issued_reference, Mapping)
            else None
        )
        if (
            not isinstance(issued_days, list)
            or not issued_days
            or issued_days[0] != expected_shadow_start.isoformat()
        ):
            raise ExogenousGovernanceError(
                "Continuite rolling365/issued shadow invalide: l'historique emis "
                "doit commencer le lendemain exact du rolling365."
            )
        shadow_gate = _evaluate_gate(shadow_metrics, policy=policy, phase="shadow")
        if not shadow_gate.passes:
            decision = "reject"
        elif not production_gate_passes:
            decision = "shadow"
        else:
            decision = "promote"
        if shadow_gate.passes and production_gate_passes:
            reasons.append("gates rolling365 et live shadow passees")
        elif shadow_gate.passes:
            blockers: list[str] = []
            if not production_pit_evidence:
                blockers.append(
                    "production_pit_evidence=false interdit la promotion"
                )
            if not production_pipeline_evidence:
                blockers.append(
                    "production_pipeline_evidence=false: le pipeline final "
                    "avec correcteur residuel n'est pas encore valide"
                )
            reasons.append(
                "gates rolling365 et live shadow passees, mais " + "; ".join(blockers)
            )
        else:
            reasons.extend(f"shadow: {reason}" for reason in shadow_gate.reasons)
    else:
        if production_gate_passes:
            decision = "promote"
            reasons.append("gate rolling365 passee; shadow non requis par la politique")
        else:
            decision = "shadow"
            blockers: list[str] = []
            if not production_pit_evidence:
                blockers.append(
                    "production_pit_evidence=false interdit la promotion"
                )
            if not production_pipeline_evidence:
                blockers.append(
                    "production_pipeline_evidence=false: le pipeline final "
                    "avec correcteur residuel n'est pas encore valide"
                )
            reasons.append(
                "gate rolling365 passee, mais " + "; ".join(blockers)
            )
    if decision not in DECISIONS:  # pragma: no cover - invariant
        raise AssertionError(decision)
    return PromotionDecision(
        schema_version=1,
        candidate_model=model_id,
        zone=canonical_zone,
        decision=decision,
        rolling365=rolling_metrics,
        rolling365_gate=rolling_gate,
        live_shadow=shadow_metrics,
        live_shadow_gate=shadow_gate,
        shadow_evidence_verified=shadow_evidence_verified,
        production_pit_evidence=production_pit_evidence,
        production_pit_gate_passes=production_pit_evidence,
        production_pipeline_evidence=production_pipeline_evidence,
        production_pipeline_gate_passes=production_pipeline_evidence,
        reasons=tuple(reasons),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trainer_directory_sha256(path: Path) -> str:
    """Reproduce the trainer's relative-path + byte checkpoint identity."""

    digest = hashlib.sha256()
    files = _regular_files(path)
    if not path.is_dir() or not files:
        raise ExogenousGovernanceError(f"Checkpoint absent ou vide: {path}")
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _is_reparse_or_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.stat().st_file_attributes  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False
    return bool(attributes & getattr(os.stat_result, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _regular_files(root: Path) -> list[Path]:
    if _is_reparse_or_link(root):
        raise ExogenousGovernanceError(f"Lien/reparse interdit dans le bundle: {root}")
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    result: list[Path] = []
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in directory_names:
            child = parent / name
            if _is_reparse_or_link(child):
                raise ExogenousGovernanceError(
                    f"Lien/reparse interdit dans le bundle: {child}"
                )
        for name in filenames:
            child = parent / name
            if _is_reparse_or_link(child) or not child.is_file():
                raise ExogenousGovernanceError(
                    f"Artefact non regulier interdit: {child}"
                )
            result.append(child)
    return sorted(result)


def _source_fingerprint(
    paths: Mapping[str, Path],
    *,
    decision: PromotionDecision,
    policy: GovernancePolicy,
) -> str:
    records: list[dict[str, object]] = []
    for role, source in sorted(paths.items()):
        for file_path in _regular_files(source):
            relative = file_path.name if source.is_file() else file_path.relative_to(source).as_posix()
            records.append(
                {
                    "role": role,
                    "path": relative,
                    "size_bytes": file_path.stat().st_size,
                    "sha256": _sha256(file_path),
                }
            )
    payload = {
        "sources": records,
        "decision": decision.to_dict(),
        "policy": policy.to_dict(),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-.").lower()
    if not slug:
        raise ExogenousGovernanceError("Identifiant candidat vide apres normalisation.")
    return slug[:80]


def _copy_source(source: Path, destination: Path) -> None:
    _regular_files(source)
    if source.is_file():
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source, destination / source.name)
    else:
        shutil.copytree(source, destination, symlinks=True)


def _bundle_records(bundle: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in _regular_files(bundle):
        relative = path.relative_to(bundle).as_posix()
        if relative == "artifact_checksums.json":
            continue
        role = (
            "candidate_artifact"
            if relative.startswith("artifacts/")
            else "promotion_evidence"
        )
        records.append(
            {
                "path": relative,
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return sorted(records, key=lambda item: str(item["path"]))


def _records_sha256(records: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(records))).hexdigest()


def seal_promotion_bundle(
    *,
    output_root: str | Path,
    rolling_predictions_path: str | Path,
    experiment_manifest_path: str | Path,
    decision: PromotionDecision,
    policy: GovernancePolicy,
    candidate_artifacts: Mapping[str, str | Path],
    shadow_predictions_path: str | Path | None = None,
    shadow_manifest_path: str | Path | None = None,
    candidate_id: str | None = None,
) -> Path:
    """Copy and checksum all evidence into an immutable candidate directory."""

    if set(candidate_artifacts) < {"checkpoint", "schema"}:
        raise ExogenousGovernanceError(
            "Les roles d'artefact checkpoint et schema sont obligatoires."
        )
    role_pattern = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
    sources: dict[str, Path] = {
        "rolling365_predictions": Path(rolling_predictions_path).expanduser().resolve(),
        "experiment_manifest": Path(experiment_manifest_path).expanduser().resolve(),
    }
    if shadow_predictions_path is not None:
        sources["shadow_predictions"] = Path(shadow_predictions_path).expanduser().resolve()
        if shadow_manifest_path is None:
            raise ExogenousGovernanceError(
                "shadow_manifest_path obligatoire avec shadow_predictions_path."
            )
        sources["shadow_manifest"] = Path(shadow_manifest_path).expanduser().resolve()
        if sources["shadow_manifest"].name == "shadow_final_manifest.json":
            lineage_directory = sources["shadow_manifest"].parent
            required_lineage = {
                "shadow_final_evidence.csv.gz",
                "shadow_final_manifest.json",
                "shadow_final_incumbent.csv.gz",
                "raw_shadow_observed_evidence.csv.gz",
                "raw_shadow_manifest.json",
                "raw_shadow_journal.csv.gz",
                "shadow_final_issued_history.csv.gz",
                "residual_corrector.json",
                "oof_audit.json",
                "schema.json",
                "experiment_manifest.json",
            }
            available_lineage = {
                child.name for child in lineage_directory.iterdir() if child.is_file()
            }
            missing_lineage = sorted(required_lineage.difference(available_lineage))
            if missing_lineage:
                raise ExogenousGovernanceError(
                    "Dossier de provenance shadow final incomplet: "
                    + ", ".join(missing_lineage)
                )
            sources["shadow_lineage"] = lineage_directory
    elif shadow_manifest_path is not None:
        raise ExogenousGovernanceError(
            "shadow_manifest_path fourni sans shadow_predictions_path."
        )
    artifacts: dict[str, Path] = {}
    for role, raw_path in candidate_artifacts.items():
        if not role_pattern.fullmatch(str(role)):
            raise ExogenousGovernanceError(f"Role d'artefact invalide: {role!r}.")
        artifacts[str(role)] = Path(raw_path).expanduser().resolve()
    for source in [*sources.values(), *artifacts.values()]:
        _regular_files(source)

    experiment_payload = validate_experiment_manifest(
        sources["experiment_manifest"], zone=decision.zone
    )
    expected_candidate = str(
        experiment_payload.get("experiment_id", experiment_payload["model_id"])
    ).strip()
    if expected_candidate != decision.candidate_model:
        raise ExogenousGovernanceError(
            "La decision ne correspond pas au candidat du manifeste d'experience."
        )
    checkpoint_source = artifacts["checkpoint"]
    schema_source = artifacts["schema"]
    if not checkpoint_source.is_dir() or not schema_source.is_file():
        raise ExogenousGovernanceError(
            "checkpoint doit etre un repertoire et schema un fichier regulier."
        )
    if _trainer_directory_sha256(checkpoint_source) != experiment_payload.get(
        "checkpoint_sha256"
    ):
        raise ExogenousGovernanceError(
            "Le checkpoint fourni ne correspond pas au checkpoint evalue."
        )
    if _sha256(schema_source) != experiment_payload.get("schema_sha256"):
        raise ExogenousGovernanceError(
            "Le schema fourni ne correspond pas au schema evalue."
        )
    evaluation_evidence = _strict_mapping(
        experiment_payload.get("evaluation_evidence"),
        name="experiment_manifest.evaluation_evidence",
    )
    if evaluation_evidence.get("sha256") != _sha256(
        sources["rolling365_predictions"]
    ):
        raise ExogenousGovernanceError(
            "Le CSV rolling365 ne correspond pas a l'evidence publiee par le trainer."
        )
    if shadow_predictions_path is not None:
        shadow_payload = validate_final_shadow_manifest(
            sources["shadow_manifest"],
            experiment_manifest=experiment_payload,
            zone=decision.zone,
        )
        if shadow_payload.get("predictions_sha256") != _sha256(
            sources["shadow_predictions"]
        ):
            raise ExogenousGovernanceError(
                "Le CSV live shadow ne correspond pas a son manifeste scelle."
            )
        if shadow_payload.get("source_experiment_manifest_sha256") != _sha256(
            sources["experiment_manifest"]
        ):
            raise ExogenousGovernanceError(
                "Le shadow final ne correspond pas au manifeste d'experience fourni."
            )

    root = Path(output_root).expanduser().resolve()
    for source in [*sources.values(), *artifacts.values()]:
        if source.is_dir():
            try:
                root.relative_to(source)
            except ValueError:
                pass
            else:
                raise ExogenousGovernanceError(
                    f"La sortie {root} ne peut pas etre imbriquee dans la source {source}."
                )
    fingerprint = _source_fingerprint(
        {**sources, **{f"artifact:{key}": value for key, value in artifacts.items()}},
        decision=decision,
        policy=policy,
    )
    identity = candidate_id or (
        f"{decision.candidate_model}-{decision.zone}-"
        f"{decision.rolling365.end_day_local}-{fingerprint[:12]}"
    )
    identity = _safe_slug(identity)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / identity
    if destination.exists():
        manifest = verify_promotion_bundle(destination)
        if manifest.get("source_fingerprint_sha256") != fingerprint:
            raise FileExistsError(
                f"Bundle existant avec une autre empreinte: {destination}"
            )
        return destination

    staging = root / f".{identity}.staging-{uuid4().hex}"
    try:
        staging.mkdir(parents=False, exist_ok=False)
        evidence = staging / "evidence"
        evidence.mkdir()
        for role, source in sources.items():
            target = evidence / role
            _copy_source(source, target)
        artifact_root = staging / "artifacts"
        artifact_root.mkdir()
        for role, source in sorted(artifacts.items()):
            _copy_source(source, artifact_root / role)
        _write_json(staging / "policy_snapshot.json", policy.to_dict())
        _write_json(staging / "promotion_decision.json", decision.to_dict())
        bundle_manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_kind": "chronos2_exogenous_promotion_evidence",
            "candidate_id": identity,
            "candidate_model": decision.candidate_model,
            "zone": decision.zone,
            "decision": decision.decision,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_fingerprint_sha256": fingerprint,
            "rolling365_end_day_local": decision.rolling365.end_day_local,
            "shadow_end_day_local": (
                decision.live_shadow.end_day_local
                if decision.live_shadow is not None
                else None
            ),
            "required_artifact_roles": ["checkpoint", "schema"],
            "deployment": {
                "activation_performed": False,
                "incumbent_contract_modified": False,
                "activation_requires_separate_explicit_authorisation": True,
                "recommended_mode": "isolated_shadow_sidecar",
            },
            "sha256_manifest": "artifact_checksums.json",
        }
        _write_json(staging / "bundle_manifest.json", bundle_manifest)
        records = _bundle_records(staging)
        _write_json(
            staging / "artifact_checksums.json",
            {
                "schema_version": 1,
                "algorithm": "sha256",
                "bundle_sha256": _records_sha256(records),
                "artifacts": records,
            },
        )
        verify_promotion_bundle(staging)
        staging.rename(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def verify_promotion_bundle(path: str | Path) -> dict[str, object]:
    """Verify completeness, path confinement, hashes, and decision identity."""

    bundle = Path(path).expanduser().resolve()
    if not bundle.is_dir() or _is_reparse_or_link(bundle):
        raise ExogenousGovernanceError(f"Bundle absent ou non regulier: {bundle}")
    checksum_path = bundle / "artifact_checksums.json"
    if not checksum_path.is_file():
        raise ExogenousGovernanceError("artifact_checksums.json absent.")
    checksum = _strict_mapping(
        json.loads(checksum_path.read_text(encoding="utf-8")),
        name=str(checksum_path),
    )
    if checksum.get("algorithm") != "sha256":
        raise ExogenousGovernanceError("Algorithme de checksum non supporte.")
    raw_records = checksum.get("artifacts")
    if not isinstance(raw_records, list) or not raw_records:
        raise ExogenousGovernanceError("Liste de checksums absente ou vide.")
    listed: set[str] = set()
    normalized: list[dict[str, object]] = []
    for raw in raw_records:
        record = _strict_mapping(raw, name="artifact_checksums.artifact")
        relative_text = record.get("path")
        if not isinstance(relative_text, str):
            raise ExogenousGovernanceError("Chemin de checksum invalide.")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative_text == "artifact_checksums.json":
            raise ExogenousGovernanceError(f"Chemin hors bundle interdit: {relative_text}")
        if relative_text in listed:
            raise ExogenousGovernanceError(f"Checksum duplique: {relative_text}")
        listed.add(relative_text)
        target = (bundle / relative).resolve()
        try:
            target.relative_to(bundle)
        except ValueError as exc:
            raise ExogenousGovernanceError(
                f"Chemin de checksum echappe au bundle: {relative_text}"
            ) from exc
        if not target.is_file() or _is_reparse_or_link(target):
            raise ExogenousGovernanceError(f"Artefact absent/non regulier: {relative_text}")
        size = target.stat().st_size
        digest = _sha256(target)
        if record.get("size_bytes") != size or record.get("sha256") != digest:
            raise ExogenousGovernanceError(f"Checksum divergent: {relative_text}")
        normalized.append(
            {
                "path": relative_text,
                "role": record.get("role"),
                "size_bytes": size,
                "sha256": digest,
            }
        )
    actual_files = {
        item.relative_to(bundle).as_posix()
        for item in _regular_files(bundle)
        if item.name != "artifact_checksums.json"
    }
    if actual_files != listed:
        raise ExogenousGovernanceError(
            "Le bundle contient des fichiers non scelles ou omet des artefacts: "
            f"extra={sorted(actual_files - listed)}, missing={sorted(listed - actual_files)}."
        )
    normalized.sort(key=lambda item: str(item["path"]))
    if checksum.get("bundle_sha256") != _records_sha256(normalized):
        raise ExogenousGovernanceError("bundle_sha256 divergent.")

    manifest_path = bundle / "bundle_manifest.json"
    decision_path = bundle / "promotion_decision.json"
    manifest = dict(
        _strict_mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            name=str(manifest_path),
        )
    )
    decision = _strict_mapping(
        json.loads(decision_path.read_text(encoding="utf-8")), name=str(decision_path)
    )
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ExogenousGovernanceError("Schema du bundle incompatible.")
    declared_id = manifest.get("candidate_id")
    is_internal_staging = isinstance(declared_id, str) and bundle.name.startswith(
        f".{declared_id}.staging-"
    )
    if declared_id != bundle.name and not is_internal_staging:
        raise ExogenousGovernanceError("candidate_id incompatible avec le repertoire.")
    if manifest.get("decision") not in DECISIONS:
        raise ExogenousGovernanceError("Decision du bundle invalide.")
    if manifest.get("decision") != decision.get("decision"):
        raise ExogenousGovernanceError("Decision incoherente entre manifeste et audit.")
    if decision.get("production_activation_performed") is not False:
        raise ExogenousGovernanceError("La gate ne doit jamais activer la production.")
    deployment = _strict_mapping(manifest.get("deployment"), name="deployment")
    if deployment.get("activation_performed") is not False:
        raise ExogenousGovernanceError("Bundle declare une activation non autorisee.")
    required_roles = manifest.get("required_artifact_roles")
    if required_roles != ["checkpoint", "schema"]:
        raise ExogenousGovernanceError("Liste des roles d'artefact obligatoire invalide.")
    for role in required_roles:
        if not (bundle / "artifacts" / str(role)).is_dir():
            raise ExogenousGovernanceError(f"Role d'artefact obligatoire absent: {role}")
    return manifest


__all__ = [
    "ExogenousGovernanceError",
    "GateResult",
    "GovernancePolicy",
    "PromotionDecision",
    "WindowEvaluation",
    "evaluate_promotion",
    "load_policy",
    "load_prediction_evidence",
    "seal_promotion_bundle",
    "validate_experiment_manifest",
    "validate_final_shadow_manifest",
    "validate_shadow_manifest",
    "verify_promotion_bundle",
]
