"""Causal, isolated LoRA fine-tuning support for Chronos-2.

This module deliberately has no import-time dependency on ``chronos`` or
``torch``.  The production pipeline is loaded lazily and can be replaced by a
small fake in unit tests.  Inputs are origin-aware panels: every
``(forecast_origin_utc, item_id)`` group is one exact context + horizon sample.
That shape, combined with ``min_past=context_length``, prevents Chronos-2's
training dataset from drawing artificial, non day-ahead cut-offs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import shutil
import stat
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


class ExogenousFineTuneError(RuntimeError):
    """Raised when the causal fine-tuning contract is not satisfied."""


PipelineLoader = Callable[[str | Path, Mapping[str, Any]], Any]
EVALUATION_ROLES = frozenset({"primary_predeclared", "diagnostic_only"})


@dataclass(frozen=True)
class ExogenousFineTuneConfig:
    """Validated, flat configuration used by the trainer."""

    config_path: Path
    project_root: Path
    experiment_id: str
    evaluation_role: str
    panel_path: Path
    panel_audit_path: Path
    output_directory: Path
    timestamp_column: str
    origin_column: str
    item_column: str
    feature_available_at_column: str
    target_columns: tuple[str, ...]
    known_future_covariates: tuple[str, ...]
    past_only_covariates: tuple[str, ...]
    timezone: str
    cutoff_local_time: str
    frequency: str
    context_length: int
    prediction_length: int
    training_window_days: int
    validation_days: int
    evaluation_days: int
    require_consecutive_origins: bool
    require_complete_known_future: bool
    production_pit_evidence: bool
    model_id: str
    model_revision: str | None
    local_files_only: bool
    device_map: str
    learning_rate: float
    num_steps: int
    batch_size: int
    seed: int
    lora_config: Mapping[str, Any]
    # Explicitly opt in to a two-phase prospective freeze.  The only labels
    # that may then be absent are the complete physical horizon (23/24/25 h)
    # of the very last evaluation origin.  The default deliberately preserves
    # the historical, fully-observed contract.
    allow_unresolved_final_evaluation_day: bool = False

    @property
    def covariate_columns(self) -> tuple[str, ...]:
        return self.past_only_covariates + self.known_future_covariates


@dataclass(frozen=True)
class OriginSplit:
    train: tuple[pd.Timestamp, ...]
    validation: tuple[pd.Timestamp, ...]
    evaluation: tuple[pd.Timestamp, ...]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExogenousFineTuneError(f"{name} doit etre un mapping YAML.")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    name: str,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ExogenousFineTuneError(f"{name}: cles manquantes: {missing}.")
    if unknown:
        raise ExogenousFineTuneError(f"{name}: cles inconnues: {unknown}.")


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExogenousFineTuneError(f"{name} doit etre un entier >= {minimum}.")
    return int(value)


def _name_tuple(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExogenousFineTuneError(f"{name} doit etre une liste de colonnes.")
    result = tuple(str(item).strip() for item in value)
    if (not allow_empty and not result) or any(not item for item in result):
        raise ExogenousFineTuneError(f"{name} contient une colonne vide ou est vide.")
    if len(set(result)) != len(result):
        raise ExogenousFineTuneError(f"{name} contient des doublons.")
    return result


def _resolve_path(root: Path, value: Any, name: str) -> Path:
    raw = Path(str(value)).expanduser()
    path = raw if raw.is_absolute() else root / raw
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ExogenousFineTuneError(
            f"{name} doit rester sous le project_root: {resolved}."
        ) from exc
    return resolved


def load_config(path: str | Path) -> ExogenousFineTuneConfig:
    """Load the strict YAML contract without touching data or the model."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ExogenousFineTuneError(f"Configuration introuvable: {config_path}.")
    raw = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "racine")
    _exact_keys(
        raw,
        required={
            "format_version",
            "experiment_id",
            "evaluation_role",
            "project_root",
            "data",
            "model",
            "training",
            "output",
        },
        optional=set(),
        name="racine",
    )
    if raw["format_version"] != 1:
        raise ExogenousFineTuneError("format_version doit valoir 1.")
    evaluation_role = str(raw["evaluation_role"]).strip()
    if evaluation_role not in EVALUATION_ROLES:
        raise ExogenousFineTuneError(
            "evaluation_role doit valoir 'primary_predeclared' ou "
            "'diagnostic_only'."
        )

    root_value = Path(str(raw["project_root"])).expanduser()
    project_root = (
        root_value if root_value.is_absolute() else config_path.parent / root_value
    ).resolve()
    if not project_root.is_dir():
        raise ExogenousFineTuneError(f"project_root introuvable: {project_root}.")

    data = _mapping(raw["data"], "data")
    model = _mapping(raw["model"], "model")
    training = _mapping(raw["training"], "training")
    output = _mapping(raw["output"], "output")
    _exact_keys(
        data,
        required={
            "panel_path",
            "timestamp_column",
            "origin_column",
            "item_column",
            "feature_available_at_column",
            "target_columns",
            "known_future_covariates",
            "past_only_covariates",
            "timezone",
            "cutoff_local_time",
            "frequency",
            "context_length",
            "prediction_length",
            "training_window_days",
            "validation_days",
            "evaluation_days",
            "require_consecutive_origins",
            "require_complete_known_future",
            "production_pit_evidence",
        },
        optional={"panel_audit_path", "allow_unresolved_final_evaluation_day"},
        name="data",
    )
    _exact_keys(
        model,
        required={"model_id", "local_files_only", "device_map"},
        optional={"revision"},
        name="model",
    )
    _exact_keys(
        training,
        required={
            "finetune_mode",
            "learning_rate",
            "num_steps",
            "batch_size",
            "seed",
            "lora_config",
        },
        optional=set(),
        name="training",
    )
    _exact_keys(output, required={"directory"}, optional=set(), name="output")

    if str(training["finetune_mode"]).lower() != "lora":
        raise ExogenousFineTuneError(
            "Ce POC est fail-closed: training.finetune_mode doit valoir 'lora'."
        )
    for section, key in (
        (data, "require_consecutive_origins"),
        (data, "require_complete_known_future"),
        (data, "production_pit_evidence"),
        (model, "local_files_only"),
    ):
        if not isinstance(section[key], bool):
            raise ExogenousFineTuneError(f"{key} doit etre booleen.")
    unresolved_final_day = data.get(
        "allow_unresolved_final_evaluation_day", False
    )
    if not isinstance(unresolved_final_day, bool):
        raise ExogenousFineTuneError(
            "allow_unresolved_final_evaluation_day doit etre booleen."
        )

    targets = _name_tuple(data["target_columns"], "data.target_columns")
    known = _name_tuple(
        data["known_future_covariates"], "data.known_future_covariates"
    )
    past = _name_tuple(
        data["past_only_covariates"],
        "data.past_only_covariates",
        allow_empty=True,
    )
    overlap = sorted(set(targets) & (set(known) | set(past)))
    overlap += sorted(set(known) & set(past))
    if overlap:
        raise ExogenousFineTuneError(f"Colonnes classees plusieurs fois: {overlap}.")

    forbidden = ("storm", "mkonline")
    forbidden_columns = [
        column
        for column in (*known, *past)
        if any(token in column.lower() for token in forbidden)
    ]
    if forbidden_columns:
        raise ExogenousFineTuneError(
            "Storm et MKOnline sont interdits dans les inputs: "
            f"{forbidden_columns}."
        )
    noncausal_future = [
        column
        for column in known
        if any(
            token in column.lower()
            for token in ("actual", "observed", "realized", "realised")
        )
    ]
    if noncausal_future:
        raise ExogenousFineTuneError(
            "Covariables futures potentiellement realisees/interdites: "
            f"{noncausal_future}."
        )

    cutoff = str(data["cutoff_local_time"])
    try:
        datetime.strptime(cutoff, "%H:%M")
    except ValueError as exc:
        raise ExogenousFineTuneError(
            "data.cutoff_local_time doit suivre HH:MM."
        ) from exc

    training_days = _positive_int(
        data["training_window_days"], "data.training_window_days"
    )
    validation_days = _positive_int(data["validation_days"], "data.validation_days")
    if validation_days >= training_days:
        raise ExogenousFineTuneError(
            "validation_days doit etre strictement inferieur a training_window_days."
        )
    learning_rate = float(training["learning_rate"])
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ExogenousFineTuneError("training.learning_rate doit etre positif.")
    lora = dict(_mapping(training["lora_config"], "training.lora_config"))
    if not lora:
        raise ExogenousFineTuneError("training.lora_config ne doit pas etre vide.")

    revision_raw = str(model.get("revision") or "").strip()
    panel_path = _resolve_path(project_root, data["panel_path"], "data.panel_path")
    audit_value = data.get("panel_audit_path")
    panel_audit_path = (
        _resolve_path(project_root, audit_value, "data.panel_audit_path")
        if audit_value is not None
        else panel_path.with_suffix(panel_path.suffix + ".audit.json")
    )
    return ExogenousFineTuneConfig(
        config_path=config_path,
        project_root=project_root,
        experiment_id=str(raw["experiment_id"]).strip(),
        evaluation_role=evaluation_role,
        panel_path=panel_path,
        panel_audit_path=panel_audit_path,
        output_directory=_resolve_path(project_root, output["directory"], "output.directory"),
        timestamp_column=str(data["timestamp_column"]),
        origin_column=str(data["origin_column"]),
        item_column=str(data["item_column"]),
        feature_available_at_column=str(data["feature_available_at_column"]),
        target_columns=targets,
        known_future_covariates=known,
        past_only_covariates=past,
        timezone=str(data["timezone"]),
        cutoff_local_time=cutoff,
        frequency=str(data["frequency"]),
        context_length=_positive_int(data["context_length"], "data.context_length", 24),
        prediction_length=_positive_int(data["prediction_length"], "data.prediction_length"),
        training_window_days=training_days,
        validation_days=validation_days,
        evaluation_days=_positive_int(data["evaluation_days"], "data.evaluation_days"),
        require_consecutive_origins=bool(data["require_consecutive_origins"]),
        require_complete_known_future=bool(data["require_complete_known_future"]),
        production_pit_evidence=bool(data["production_pit_evidence"]),
        model_id=str(model["model_id"]).strip(),
        model_revision=revision_raw or None,
        local_files_only=bool(model["local_files_only"]),
        device_map=str(model["device_map"]),
        learning_rate=learning_rate,
        num_steps=_positive_int(training["num_steps"], "training.num_steps"),
        batch_size=_positive_int(training["batch_size"], "training.batch_size"),
        seed=_positive_int(training["seed"], "training.seed", 0),
        lora_config=lora,
        allow_unresolved_final_evaluation_day=bool(unresolved_final_day),
    )


def read_panel(path: Path) -> pd.DataFrame:
    """Read an origin panel from Parquet or CSV without implicit index rules."""

    if not path.is_file():
        raise ExogenousFineTuneError(f"Panel causal introuvable: {path}.")
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == ".parquet":
        return pd.read_parquet(path)
    if suffixes and suffixes[-1] in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ExogenousFineTuneError(
        f"Format de panel non gere: {path.name}; utiliser .parquet, .csv ou .csv.gz."
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _verify_mutated_target_cache_against_panel(
    frame: pd.DataFrame,
    config: ExogenousFineTuneConfig,
    *,
    audit_payload: Mapping[str, Any],
    zone: str,
    source_path: Path,
) -> dict[str, Any]:
    """Prove that a mutable target cache still agrees with the sealed panel.

    Operational target caches are append-only in normal use.  Their byte SHA
    therefore changes whenever a newly observed day is published, even though
    every value used to build an older training panel remains unchanged.  On a
    byte mismatch we compare *all* target timestamps used by the sealed panel
    with the current canonical cache.  This preserves the anti-tampering check
    while allowing harmless appends after materialisation.
    """

    timestamp_column = config.timestamp_column
    if timestamp_column not in frame.columns:
        raise ExogenousFineTuneError(
            "Audit PIT refuse: impossible de verifier la cible; colonne timestamp absente."
        )
    layout = str(audit_payload.get("layout", "per_zone"))
    if layout == "per_zone":
        if config.item_column not in frame.columns or "target" not in frame.columns:
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: impossible de verifier la cible {zone} dans le panel."
            )
        item_values = frame[config.item_column].astype(str).str.upper()
        selected = frame.loc[
            item_values.eq(zone), [timestamp_column, "target"]
        ].rename(columns={"target": "panel_value"})
    elif layout == "cwe_wide":
        target_column = f"target_{zone.casefold()}"
        if target_column not in frame.columns:
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: cible {target_column} absente du panel CWE."
            )
        selected = frame.loc[:, [timestamp_column, target_column]].rename(
            columns={target_column: "panel_value"}
        )
    else:
        raise ExogenousFineTuneError(
            f"Audit PIT refuse: layout du panel inconnu {layout!r}."
        )
    if selected.empty:
        raise ExogenousFineTuneError(
            f"Audit PIT refuse: aucune valeur cible {zone} dans le panel."
        )

    selected[timestamp_column] = pd.to_datetime(
        selected[timestamp_column], utc=True, errors="coerce"
    )
    selected["panel_value"] = pd.to_numeric(selected["panel_value"], errors="coerce")
    if selected[timestamp_column].isna().any() or np.isinf(
        selected["panel_value"].to_numpy(dtype=float)
    ).any():
        raise ExogenousFineTuneError(
            f"Audit PIT refuse: valeurs cibles {zone} invalides dans le panel."
        )
    unresolved_values = int(selected["panel_value"].isna().sum())
    if unresolved_values and not config.allow_unresolved_final_evaluation_day:
        raise ExogenousFineTuneError(
            f"Audit PIT refuse: valeurs cibles {zone} invalides dans le panel."
        )
    finite_selected = selected.loc[selected["panel_value"].notna()]
    if finite_selected.empty:
        raise ExogenousFineTuneError(
            f"Audit PIT refuse: aucune cible finie {zone} ne peut etre verifiee."
        )
    grouped = finite_selected.groupby(timestamp_column, sort=True)["panel_value"].agg(
        ["min", "max"]
    )
    internal_delta = (grouped["max"] - grouped["min"]).abs()
    if bool((internal_delta > 1e-9).any()):
        raise ExogenousFineTuneError(
            f"Audit PIT refuse: cible {zone} incoherente entre origines du panel."
        )
    expected = grouped["min"].astype(float)

    try:
        current = pd.read_csv(source_path, usecols=["timestamp", "value"])
        current_timestamps = pd.to_datetime(
            current["timestamp"], utc=True, errors="coerce", format="mixed"
        )
        current_values = pd.to_numeric(current["value"], errors="coerce").astype(
            float
        )
    except Exception as exc:
        raise ExogenousFineTuneError(
            f"Audit PIT refuse: cache cible canonique {zone} illisible."
        ) from exc
    if (
        current_timestamps.isna().any()
        or current_timestamps.duplicated().any()
        or not np.isfinite(current_values.to_numpy(dtype=float)).all()
    ):
        raise ExogenousFineTuneError(
            f"Audit PIT refuse: cache cible canonique {zone} invalide."
        )
    current_series = pd.Series(
        current_values.to_numpy(dtype=float),
        index=pd.DatetimeIndex(current_timestamps),
    ).sort_index()
    observed = current_series.reindex(expected.index)
    if observed.isna().any():
        first_missing = observed.index[observed.isna()][0]
        raise ExogenousFineTuneError(
            f"Audit PIT refuse: le cache cible canonique {zone} ne contient plus "
            f"l'heure du panel {first_missing.isoformat()}."
        )
    observed_values = observed.to_numpy(dtype=float)
    expected_values = expected.to_numpy(dtype=float)
    deltas = observed_values - expected_values
    max_abs_delta = float(np.max(np.abs(deltas))) if len(deltas) else 0.0
    # EUPHEMIA prices are published to the cent.  An hourly cache rebuilt from
    # 15-minute MTUs can legitimately contain quarter-cent averages while an
    # older cache stores the same market price rounded to two decimals.  Treat
    # those representations as equivalent only when half-up cent rounding is
    # identical; this is deliberately much stricter than a generic tolerance.
    def market_cent(values: np.ndarray) -> np.ndarray:
        magnitudes = np.floor(np.abs(values) * 100.0 + 0.5 + 1e-10) / 100.0
        return np.copysign(magnitudes, values)

    market_delta = np.abs(market_cent(observed_values) - market_cent(expected_values))
    if bool((market_delta > 1e-9).any()):
        raise ExogenousFineTuneError(
            f"Audit PIT refuse: le cache cible canonique {zone} a modifie des "
            "valeurs utilisees par le panel au-dela de la precision marche de "
            f"0,01 EUR/MWh (ecart brut max={max_abs_delta:.12g} EUR/MWh)."
        )
    return {
        "mode": "panel_target_equivalence",
        "verified_timestamps": int(len(expected)),
        "first_timestamp": expected.index.min().isoformat(),
        "last_timestamp": expected.index.max().isoformat(),
        "max_abs_delta": max_abs_delta,
        "market_precision_eur_mwh": 0.01,
        "rounding_equivalent_timestamps": int((np.abs(deltas) > 1e-9).sum()),
        "unresolved_panel_values_omitted": unresolved_values,
    }


def load_panel_audit(
    config: ExogenousFineTuneConfig,
    *,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Load and cryptographically bind the upstream PIT audit to its panel."""

    audit_path = config.panel_audit_path
    if not audit_path.is_file():
        raise ExogenousFineTuneError(
            f"Audit PIT du panel obligatoire et introuvable: {audit_path}."
        )
    if not config.panel_path.is_file():
        raise ExogenousFineTuneError(f"Panel causal introuvable: {config.panel_path}.")
    try:
        payload = _mapping(
            json.loads(audit_path.read_text(encoding="utf-8")), "panel_audit"
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ExogenousFineTuneError(f"Audit PIT illisible: {audit_path}.") from exc
    declared_panel_hash = payload.get("panel_sha256")
    actual_panel_hash = _sha256_file(config.panel_path)
    if not _is_sha256(declared_panel_hash) or declared_panel_hash != actual_panel_hash:
        raise ExogenousFineTuneError(
            "Audit PIT refuse: panel_sha256 absent ou divergent du panel charge."
        )

    zones_raw = payload.get("zones")
    if not isinstance(zones_raw, list) or not zones_raw:
        raise ExogenousFineTuneError("Audit PIT refuse: liste zones absente.")
    zones = tuple(str(zone).strip().upper() for zone in zones_raw)
    if any(not zone for zone in zones) or len(set(zones)) != len(zones):
        raise ExogenousFineTuneError("Audit PIT refuse: liste zones invalide.")
    zone_evidence_raw = payload.get("production_pit_evidence")
    if not isinstance(zone_evidence_raw, Mapping):
        raise ExogenousFineTuneError(
            "Audit PIT refuse: production_pit_evidence par zone absent."
        )
    if set(zone_evidence_raw) != set(zones):
        raise ExogenousFineTuneError(
            "Audit PIT refuse: couverture production_pit_evidence incomplete par zone."
        )
    if any(type(zone_evidence_raw[zone]) is not bool for zone in zones):
        raise ExogenousFineTuneError(
            "Audit PIT refuse: chaque preuve de zone doit etre booleenne."
        )
    production_ready = payload.get("production_ready")
    if type(production_ready) is not bool:
        raise ExogenousFineTuneError(
            "Audit PIT refuse: production_ready doit etre booleen."
        )
    effective_evidence = bool(
        production_ready and all(bool(zone_evidence_raw[zone]) for zone in zones)
    )
    if config.production_pit_evidence and not effective_evidence:
        raise ExogenousFineTuneError(
            "La configuration exige production_pit_evidence=true, mais l'audit "
            "lie au panel n'est pas production-ready pour toutes les zones."
        )
    unresolved_policy = payload.get(
        "allow_unresolved_final_evaluation_day", False
    )
    if not isinstance(unresolved_policy, bool):
        raise ExogenousFineTuneError(
            "Audit PIT refuse: allow_unresolved_final_evaluation_day doit etre "
            "booleen."
        )
    if config.allow_unresolved_final_evaluation_day and not unresolved_policy:
        raise ExogenousFineTuneError(
            "Le mode prospectif demande des labels finaux potentiellement absents, "
            "mais le sidecar du panel ne predeclare pas cette politique."
        )

    pack = payload.get("pack")
    if not isinstance(pack, str) or not pack.strip():
        raise ExogenousFineTuneError("Audit PIT refuse: pack exogene absent.")
    if payload.get("canonical_target_contracts_verified") is not True:
        raise ExogenousFineTuneError(
            "Audit PIT refuse: contrats de cibles canoniques non verifies."
        )
    target_sources_raw = payload.get("target_sources")
    target_contracts_raw = payload.get("target_contracts")
    if (
        not isinstance(target_sources_raw, Mapping)
        or set(map(str, target_sources_raw)) != set(zones)
        or not isinstance(target_contracts_raw, Mapping)
        or set(map(str, target_contracts_raw)) != set(zones)
    ):
        raise ExogenousFineTuneError(
            "Audit PIT refuse: provenance des cibles canoniques incomplete."
        )
    target_sources: dict[str, dict[str, str]] = {}
    target_cache_verification: dict[str, dict[str, Any]] = {}
    target_contracts: dict[str, dict[str, Any]] = {}
    for zone in zones:
        source = target_sources_raw.get(zone)
        contract = target_contracts_raw.get(zone)
        if not isinstance(source, Mapping) or not isinstance(contract, Mapping):
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: cible canonique {zone} invalide."
            )
        source_path = source.get("source_path")
        source_sha = source.get("source_sha256")
        series = contract.get("series")
        cache_path = contract.get("cache_path")
        if (
            not isinstance(source_path, str)
            or not source_path.strip()
            or not _is_sha256(source_sha)
            or not isinstance(series, str)
            or not series.strip()
            or not isinstance(cache_path, str)
            or not cache_path.strip()
            or Path(source_path).expanduser().resolve()
            != Path(cache_path).expanduser().resolve()
        ):
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: identite de cible canonique {zone} incoherente."
            )
        resolved_source = Path(source_path).expanduser().resolve()
        if not resolved_source.is_file():
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: cache cible canonique {zone} introuvable."
            )
        current_source_sha = _sha256_file(resolved_source)
        if current_source_sha == source_sha:
            verification = {
                "mode": "exact_file_sha256",
                "verified_timestamps": None,
                "max_abs_delta": 0.0,
            }
        else:
            if panel is None:
                raise ExogenousFineTuneError(
                    f"Audit PIT refuse: SHA du cache cible canonique {zone} divergent "
                    "et aucun panel n'a ete fourni pour verifier une extension sans revision."
                )
            verification = _verify_mutated_target_cache_against_panel(
                panel,
                config,
                audit_payload=payload,
                zone=zone,
                source_path=resolved_source,
            )
            verification["materialization_source_sha256"] = str(source_sha)
            verification["current_source_sha256"] = current_source_sha
        target_cache_verification[zone] = verification
        target_sources[zone] = {
            "source_path": str(resolved_source),
            "source_sha256": str(source_sha),
        }
        target_contracts[zone] = dict(contract)

    banks = payload.get("exogenous_banks")
    if not isinstance(banks, Mapping):
        raise ExogenousFineTuneError("Audit PIT refuse: exogenous_banks absent.")
    source_hashes: dict[str, dict[str, str]] = {}
    source_audit_hashes: dict[str, dict[str, str]] = {}
    source_cutoff_timezones: dict[str, dict[str, str]] = {}
    production_blockers: dict[str, list[str]] = {}
    for zone in zones:
        bank = banks.get(zone)
        if not isinstance(bank, Mapping):
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: exogenous_banks.{zone} absent."
            )
        hashes = bank.get("source_hashes")
        if not isinstance(hashes, Mapping) or not hashes:
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: source_hashes absents pour {zone}."
            )
        normalised_hashes = {str(name): str(value) for name, value in hashes.items()}
        invalid = [name for name, value in normalised_hashes.items() if not _is_sha256(value)]
        if invalid:
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: SHA-256 sources invalides pour {zone}: {invalid}."
            )
        source_hashes[zone] = normalised_hashes
        audit_hashes = bank.get("source_audit_hashes")
        if not isinstance(audit_hashes, Mapping):
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: source_audit_hashes absent pour {zone}."
            )
        normalised_audit_hashes = {
            str(name): str(value) for name, value in audit_hashes.items()
        }
        invalid_audits = [
            name
            for name, value in normalised_audit_hashes.items()
            if name not in normalised_hashes or not _is_sha256(value)
        ]
        if invalid_audits:
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: SHA-256 sidecars invalides pour {zone}: "
                f"{invalid_audits}."
            )
        if bool(zone_evidence_raw[zone]):
            expected_audited = set(normalised_hashes).difference(
                {"deterministic_calendar"}
            )
            if set(normalised_audit_hashes) != expected_audited:
                raise ExogenousFineTuneError(
                    f"Audit PIT refuse: preuve production {zone} sans tous les "
                    "sidecars de source scelles."
                )
        source_audit_hashes[zone] = normalised_audit_hashes
        timezone_values = bank.get("source_cutoff_timezones")
        if not isinstance(timezone_values, Mapping) or set(
            map(str, timezone_values)
        ) != set(normalised_hashes):
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: source_cutoff_timezones incomplet pour {zone}."
            )
        normalised_timezones = {
            str(name): str(value) for name, value in timezone_values.items()
        }
        for name, timezone_name in normalised_timezones.items():
            try:
                ZoneInfo(timezone_name)
            except Exception as exc:
                raise ExogenousFineTuneError(
                    f"Audit PIT refuse: timezone source invalide pour {zone}/{name}."
                ) from exc
        source_cutoff_timezones[zone] = normalised_timezones
        blockers = bank.get("production_blockers", [])
        if not isinstance(blockers, list):
            raise ExogenousFineTuneError(
                f"Audit PIT refuse: production_blockers invalide pour {zone}."
            )
        production_blockers[zone] = [str(value) for value in blockers]

    return {
        "audit_path": str(audit_path),
        "panel_sha256": actual_panel_hash,
        "panel_audit_sha256": _sha256_file(audit_path),
        "zones": list(zones),
        "declared_production_ready": production_ready,
        "zone_production_pit_evidence": {
            zone: bool(zone_evidence_raw[zone]) for zone in zones
        },
        "production_pit_evidence": effective_evidence,
        "allow_unresolved_final_evaluation_day": unresolved_policy,
        "pack": pack.strip(),
        "source_hashes": source_hashes,
        "source_audit_hashes": source_audit_hashes,
        "source_cutoff_timezones": source_cutoff_timezones,
        "target_sources": target_sources,
        "target_cache_verification": target_cache_verification,
        "target_contracts": target_contracts,
        "production_blockers": production_blockers,
    }


def _delivery_mask(
    group: pd.DataFrame,
    config: ExogenousFineTuneConfig,
) -> np.ndarray:
    """Locate the physical D+1 delivery day without assuming 24 civil hours."""

    origin = pd.Timestamp(group[config.origin_column].iloc[0]).tz_convert(
        config.timezone
    )
    delivery_day = (origin + pd.DateOffset(days=1)).date()
    timestamps = pd.DatetimeIndex(group[config.timestamp_column]).tz_convert(
        config.timezone
    )
    return np.asarray([timestamp.date() == delivery_day for timestamp in timestamps])


def _evaluation_label_contract_sha256(
    panel: pd.DataFrame,
    split: OriginSplit,
    config: ExogenousFineTuneConfig,
    unresolved_groups: Sequence[Mapping[str, Any]],
) -> str:
    """Hash every model input while masking only predeclared unresolved cells.

    The ordinary Parquet SHA remains the primary immutable identity.  This
    second digest is solely the bridge to a later panel in which the explicitly
    listed NaNs have become finite observations.  All other values, including
    earlier holdout targets and every covariate, remain inside the digest.
    """

    columns = list(
        dict.fromkeys(
            (
                config.timestamp_column,
                config.origin_column,
                config.item_column,
                config.feature_available_at_column,
                *config.target_columns,
                *config.covariate_columns,
            )
        )
    )
    # The training panel (365 fit + 365 holdout) and the residual-calibration
    # panel (365 warm-up + 365 OOF + the same 365 holdout) have different
    # prefixes.  Their common immutable object is therefore the holdout only.
    # Each complete Parquet remains independently bound by its ordinary SHA.
    selected_origins = set(split.evaluation)
    canonical = panel.loc[
        panel[config.origin_column].isin(selected_origins), columns
    ].copy()
    canonical = canonical.sort_values(
        [config.origin_column, config.item_column, config.timestamp_column],
        kind="stable",
    ).reset_index(drop=True)
    for raw in unresolved_groups:
        origin = pd.to_datetime(raw.get("origin_utc"), utc=True, errors="coerce")
        item = str(raw.get("item_id", ""))
        targets = raw.get("target_columns")
        if pd.isna(origin) or not isinstance(targets, Sequence) or isinstance(
            targets, (str, bytes)
        ):
            raise ExogenousFineTuneError(
                "Contrat de labels: groupe non resolu invalide."
            )
        target_names = tuple(str(value) for value in targets)
        if not target_names or not set(target_names).issubset(config.target_columns):
            raise ExogenousFineTuneError(
                "Contrat de labels: colonnes target non resolues invalides."
            )
        group_mask = canonical[config.origin_column].eq(pd.Timestamp(origin)) & canonical[
            config.item_column
        ].eq(item)
        group = canonical.loc[group_mask]
        if group.empty:
            raise ExogenousFineTuneError(
                "Contrat de labels: origine/item non resolu absent du panel."
            )
        delivery = _delivery_mask(group, config)
        delivery_indices = group.index[np.flatnonzero(delivery)]
        if len(delivery_indices) != int(raw.get("hours", -1)):
            raise ExogenousFineTuneError(
                "Contrat de labels: nombre d'heures non resolues divergent."
            )
        canonical.loc[delivery_indices, list(target_names)] = np.nan

    digest = hashlib.sha256()
    digest.update(b"chronos2-evaluation-label-contract-v1\n")
    digest.update(
        json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    # pandas' deterministic 64-bit row fingerprints keep this operation
    # practical for multi-million-row context panels; the outer SHA-256 binds
    # their ordered byte representation and the explicit schema above.
    row_hashes = pd.util.hash_pandas_object(
        canonical, index=False, categorize=False
    ).to_numpy(dtype="<u8", copy=False)
    digest.update(np.ascontiguousarray(row_hashes).tobytes())
    return digest.hexdigest()


def validate_panel(
    frame: pd.DataFrame,
    config: ExogenousFineTuneConfig,
) -> tuple[pd.DataFrame, OriginSplit, dict[str, Any]]:
    """Validate point-in-time provenance, shape and the frozen outer split."""

    upstream_audit = load_panel_audit(config, panel=frame)
    required = {
        config.timestamp_column,
        config.origin_column,
        config.item_column,
        config.feature_available_at_column,
        *config.target_columns,
        *config.covariate_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ExogenousFineTuneError(f"Panel: colonnes manquantes: {missing}.")
    panel = frame.loc[:, list(dict.fromkeys((*required,)))].copy()
    for column in (
        config.timestamp_column,
        config.origin_column,
        config.feature_available_at_column,
    ):
        panel[column] = pd.to_datetime(panel[column], errors="coerce", utc=True)
        if panel[column].isna().any():
            raise ExogenousFineTuneError(f"Panel: timestamps invalides dans {column}.")
    if panel[config.item_column].isna().any():
        raise ExogenousFineTuneError("Panel: item_id manquant.")
    panel[config.item_column] = panel[config.item_column].astype(str)
    if (panel[config.feature_available_at_column] > panel[config.origin_column]).any():
        offending = panel.loc[
            panel[config.feature_available_at_column] > panel[config.origin_column],
            [config.origin_column, config.feature_available_at_column],
        ].iloc[0]
        raise ExogenousFineTuneError(
            "Fuite PIT: une feature est disponible apres l'origine: "
            f"origin={offending[config.origin_column]}, "
            f"available_at={offending[config.feature_available_at_column]}."
        )

    for column in (*config.target_columns, *config.covariate_columns):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    target_values = panel[list(config.target_columns)].to_numpy(dtype=float)
    if np.isinf(target_values).any():
        raise ExogenousFineTuneError("Panel: les labels target contiennent NaN/infini.")

    origin_values = pd.DatetimeIndex(panel[config.origin_column].drop_duplicates()).sort_values()
    if origin_values.empty:
        raise ExogenousFineTuneError("Panel: aucune origine.")
    local_origins = origin_values.tz_convert(config.timezone)
    expected_h, expected_m = (int(part) for part in config.cutoff_local_time.split(":"))
    if any(ts.hour != expected_h or ts.minute != expected_m for ts in local_origins):
        raise ExogenousFineTuneError(
            f"Panel: toutes les origines doivent etre a {config.cutoff_local_time} "
            f"dans {config.timezone}."
        )
    local_days = pd.DatetimeIndex(local_origins.normalize().tz_localize(None))
    if local_days.duplicated().any():
        raise ExogenousFineTuneError("Panel: plusieurs origines distinctes le meme jour local.")
    if config.require_consecutive_origins and len(local_days) > 1:
        deltas = np.diff(local_days.to_numpy(dtype="datetime64[D]")).astype(int)
        if not np.all(deltas == 1):
            raise ExogenousFineTuneError("Panel: les origines journalieres ne sont pas consecutives.")

    needed = config.training_window_days + config.evaluation_days
    if len(origin_values) < needed:
        raise ExogenousFineTuneError(
            f"Historique insuffisant: {len(origin_values)} origines < {needed} "
            "(training_window_days + evaluation_days)."
        )
    evaluation = tuple(origin_values[-config.evaluation_days :])
    fit_window = origin_values[
        -(config.training_window_days + config.evaluation_days) : -config.evaluation_days
    ]
    train = tuple(fit_window[: -config.validation_days])
    validation = tuple(fit_window[-config.validation_days :])
    split = OriginSplit(train=train, validation=validation, evaluation=evaluation)

    items_reference: tuple[str, ...] | None = None
    known_future_missing = 0
    irregular_deliveries: list[dict[str, Any]] = []
    unresolved_groups: list[dict[str, Any]] = []
    split_phase = {
        **{value: "train" for value in split.train},
        **{value: "validation" for value in split.validation},
        **{value: "evaluation_holdout" for value in split.evaluation},
    }
    groups = panel.groupby(
        [config.origin_column, config.item_column], sort=True, observed=True
    )
    expected_delta = pd.tseries.frequencies.to_offset(config.frequency)
    for (origin, item), group in groups:
        ordered = group.sort_values(config.timestamp_column)
        delivery_mask = _delivery_mask(ordered, config)
        delivery_hours = int(delivery_mask.sum())
        if delivery_hours not in {23, 24, 25}:
            raise ExogenousFineTuneError(
                f"Panel {origin}/{item}: livraison civile de {delivery_hours} heures; "
                "seules 23/24/25 sont admises."
            )
        delivery_positions = np.flatnonzero(delivery_mask)
        if not np.array_equal(
            delivery_positions,
            np.arange(len(ordered) - delivery_hours, len(ordered)),
        ):
            raise ExogenousFineTuneError(
                f"Panel {origin}/{item}: les heures D+1 doivent former le suffixe du groupe."
            )
        context_rows = len(ordered) - delivery_hours
        if context_rows != config.context_length:
            raise ExogenousFineTuneError(
                f"Panel {origin}/{item}: {context_rows} heures de contexte != "
                f"{config.context_length}."
            )
        timestamps = pd.DatetimeIndex(ordered[config.timestamp_column])
        if timestamps.duplicated().any():
            raise ExogenousFineTuneError(f"Panel {origin}/{item}: timestamp duplique.")
        if len(timestamps) > 1 and not all(
            timestamps[index] - timestamps[index - 1] == expected_delta
            for index in range(1, len(timestamps))
        ):
            raise ExogenousFineTuneError(
                f"Panel {origin}/{item}: index non regulier ({config.frequency})."
            )
        future = ordered.iloc[-delivery_hours:]
        context_targets = ordered.iloc[:-delivery_hours][
            list(config.target_columns)
        ].to_numpy(dtype=float)
        if not np.isfinite(context_targets).all():
            raise ExogenousFineTuneError(
                f"Panel {origin}/{item}: target contexte non finie."
            )
        phase = split_phase.get(pd.Timestamp(origin), "unused_prefix")
        future_targets = future[list(config.target_columns)]
        unresolved_targets: list[str] = []
        for target_column in config.target_columns:
            finite = np.isfinite(future_targets[target_column].to_numpy(dtype=float))
            if bool(finite.all()):
                continue
            if bool(finite.any()):
                raise ExogenousFineTuneError(
                    f"Panel {origin}/{item}: target horizon {target_column!r} "
                    "partiellement resolue."
                )
            if not config.allow_unresolved_final_evaluation_day:
                raise ExogenousFineTuneError(
                    "Panel: les labels target contiennent NaN/infini."
                )
            if phase != "evaluation_holdout" or pd.Timestamp(origin) != pd.Timestamp(
                split.evaluation[-1]
            ):
                raise ExogenousFineTuneError(
                    f"Panel {origin}/{item}: label absent hors de la toute derniere "
                    "origine evaluation_holdout."
                )
            unresolved_targets.append(str(target_column))
        if unresolved_targets:
            timestamps_future = pd.DatetimeIndex(future[config.timestamp_column])
            unresolved_groups.append(
                {
                    "origin_utc": pd.Timestamp(origin).isoformat(),
                    "item_id": str(item),
                    "delivery_day": timestamps_future[0]
                    .tz_convert(config.timezone)
                    .date()
                    .isoformat(),
                    "target_columns": unresolved_targets,
                    "hours": int(delivery_hours),
                    "first_delivery_utc": timestamps_future[0].isoformat(),
                    "last_delivery_utc": timestamps_future[-1].isoformat(),
                }
            )
        future_values = future[list(config.known_future_covariates)].to_numpy(
            dtype=float
        )
        missing_count = int((~np.isfinite(future_values)).sum())
        known_future_missing += missing_count
        if config.require_complete_known_future and missing_count:
            raise ExogenousFineTuneError(
                f"Panel {origin}/{item}: {missing_count} valeurs futures PIT manquantes."
            )
        delivery_local = timestamps[-delivery_hours].tz_convert(config.timezone)
        origin_local = pd.Timestamp(origin).tz_convert(config.timezone)
        if delivery_local.date() != (origin_local + pd.DateOffset(days=1)).date():
            raise ExogenousFineTuneError(
                f"Panel {origin}/{item}: livraison {delivery_local.date()} non D+1 "
                f"de l'origine {origin_local.date()}."
            )
        if delivery_hours != config.prediction_length:
            irregular_deliveries.append(
                {
                    "origin_utc": pd.Timestamp(origin).isoformat(),
                    "item_id": str(item),
                    "delivery_day": delivery_local.date().isoformat(),
                    "delivery_hours": delivery_hours,
                    "phase": phase,
                    "eligible_for_fixed_horizon_fit": False,
                }
            )

    item_sets = (
        panel.groupby(config.origin_column, observed=True)[config.item_column]
        .agg(lambda values: tuple(sorted(set(values))))
        .tolist()
    )
    for items in item_sets:
        if items_reference is None:
            items_reference = items
        elif items != items_reference:
            raise ExogenousFineTuneError("Panel: la liste des items change selon l'origine.")

    split_sets = set(split.train) | set(split.validation) | set(split.evaluation)
    selected = panel[panel[config.origin_column].isin(split_sets)].copy()
    if unresolved_groups and not upstream_audit[
        "allow_unresolved_final_evaluation_day"
    ]:
        raise ExogenousFineTuneError(
            "Panel: labels finaux absents sans predeclaration dans le sidecar PIT."
        )
    unresolved_cells = int(
        sum(record["hours"] * len(record["target_columns"]) for record in unresolved_groups)
    )
    label_binding = {
        "schema_version": 1,
        "policy": "last_evaluation_physical_horizon_only",
        "allow_unresolved_final_evaluation_day": bool(
            config.allow_unresolved_final_evaluation_day
        ),
        "unresolved_groups": unresolved_groups,
        "unresolved_cells": unresolved_cells,
        "labels_used_for_fit": False,
        "train_validation_labels_all_finite": True,
        "resolution_required_before_evaluation": bool(unresolved_groups),
        "input_contract_sha256": _evaluation_label_contract_sha256(
            selected, split, config, unresolved_groups
        ),
    }
    audit = {
        "rows": int(len(selected)),
        "items": list(items_reference or ()),
        "origins_total": int(len(origin_values)),
        "origins_used": int(len(split_sets)),
        "known_future_missing": known_future_missing,
        "pit_audit_passed": True,
        "production_pit_evidence": upstream_audit["production_pit_evidence"],
        "upstream_panel_audit": upstream_audit,
        "physical_evaluation_days": len(split.evaluation),
        "evaluation_label_binding": label_binding,
        "irregular_deliveries": irregular_deliveries,
        "irregular_fit_origins": sorted(
            {
                record["origin_utc"]
                for record in irregular_deliveries
                if record["phase"] in {"train", "validation"}
            }
        ),
        "irregular_evaluation_origins": sorted(
            {
                record["origin_utc"]
                for record in irregular_deliveries
                if record["phase"] == "evaluation_holdout"
            }
        ),
        "first_origin_utc": origin_values[0].isoformat(),
        "last_origin_utc": origin_values[-1].isoformat(),
    }
    return selected, split, audit


def bind_resolved_evaluation_panel(
    *,
    frozen_frame: pd.DataFrame,
    resolved_frame: pd.DataFrame,
    frozen_config: ExogenousFineTuneConfig,
    resolved_config: ExogenousFineTuneConfig,
    experiment_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, OriginSplit, dict[str, Any], dict[str, Any]]:
    """Bind late observations without allowing any other panel mutation.

    The frozen and resolved Parquet files keep independent SHA-bound sidecars.
    Only cells explicitly recorded as NaN in the final evaluation horizon may
    transition to finite target values.  No row, covariate, earlier target,
    timestamp or metadata value may change.
    """

    declared_panel_sha = experiment_manifest.get("panel_sha256")
    if not _is_sha256(declared_panel_sha) or _sha256_file(
        frozen_config.panel_path
    ) != declared_panel_sha:
        raise ExogenousFineTuneError(
            "Resolution holdout refusee: le panel non resolu ne correspond plus "
            "au SHA gele par le checkpoint."
        )
    frozen, frozen_split, frozen_audit = validate_panel(
        frozen_frame, frozen_config
    )
    binding_raw = experiment_manifest.get("evaluation_label_binding")
    if not isinstance(binding_raw, Mapping):
        raise ExogenousFineTuneError(
            "Resolution holdout refusee: contrat evaluation_label_binding absent."
        )
    binding = dict(binding_raw)
    if binding != frozen_audit["evaluation_label_binding"]:
        raise ExogenousFineTuneError(
            "Resolution holdout refusee: contrat des NaN divergent du panel gele."
        )
    unresolved_groups = binding.get("unresolved_groups")
    if not isinstance(unresolved_groups, list) or not unresolved_groups:
        raise ExogenousFineTuneError(
            "Resolution holdout refusee: aucune cellule non resolue predeclaree."
        )

    resolved, resolved_split, resolved_audit = validate_panel(
        resolved_frame, resolved_config
    )
    if resolved_split != frozen_split:
        raise ExogenousFineTuneError(
            "Resolution holdout refusee: le split chronologique a change."
        )
    if resolved_audit["evaluation_label_binding"]["unresolved_cells"] != 0:
        raise ExogenousFineTuneError(
            "Resolution holdout refusee: les observations finales restent incompletes."
        )
    resolved_contract = _evaluation_label_contract_sha256(
        resolved, resolved_split, resolved_config, unresolved_groups
    )
    if resolved_contract != binding.get("input_contract_sha256"):
        raise ExogenousFineTuneError(
            "Resolution holdout refusee: une entree, une ligne ou une cible non "
            "autorisee a change."
        )

    key_columns = [
        frozen_config.origin_column,
        frozen_config.item_column,
        frozen_config.timestamp_column,
    ]
    if list(frozen_frame.columns) != list(resolved_frame.columns):
        raise ExogenousFineTuneError(
            "Resolution holdout refusee: le schema du panel a change."
        )
    left = frozen_frame.copy()
    right = resolved_frame.copy()
    for column in (frozen_config.origin_column, frozen_config.timestamp_column):
        left[column] = pd.to_datetime(left[column], utc=True, errors="coerce")
        right[column] = pd.to_datetime(right[column], utc=True, errors="coerce")
    left = left.sort_values(key_columns, kind="stable").reset_index(drop=True)
    right = right.sort_values(key_columns, kind="stable").reset_index(drop=True)
    if len(left) != len(right):
        raise ExogenousFineTuneError(
            "Resolution holdout refusee: le nombre de lignes a change."
        )

    allowed_by_target = {
        target: np.zeros(len(left), dtype=bool) for target in frozen_config.target_columns
    }
    for raw in unresolved_groups:
        origin = pd.Timestamp(pd.to_datetime(raw["origin_utc"], utc=True))
        item = str(raw["item_id"])
        row_mask = left[frozen_config.origin_column].eq(origin) & left[
            frozen_config.item_column
        ].astype(str).eq(item)
        group = left.loc[row_mask]
        delivery_indices = group.index[np.flatnonzero(_delivery_mask(group, frozen_config))]
        if len(delivery_indices) != int(raw["hours"]):
            raise ExogenousFineTuneError(
                "Resolution holdout refusee: horizon physique divergent."
            )
        for target in raw["target_columns"]:
            allowed_by_target[str(target)][delivery_indices] = True

    for target, allowed in allowed_by_target.items():
        left_values = pd.to_numeric(left[target], errors="coerce").to_numpy(float)
        right_values = pd.to_numeric(right[target], errors="coerce").to_numpy(float)
        if allowed.any():
            if not np.isnan(left_values[allowed]).all() or not np.isfinite(
                right_values[allowed]
            ).all():
                raise ExogenousFineTuneError(
                    "Resolution holdout refusee: seules les cellules NaN "
                    "predeclarees peuvent devenir des observations finies."
                )
            left.loc[allowed, target] = np.nan
            right.loc[allowed, target] = np.nan
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise ExogenousFineTuneError(
            "Resolution holdout refusee: les panels different hors des cellules "
            "NaN autorisees."
        ) from exc

    upstream = resolved_audit["upstream_panel_audit"]
    if upstream["target_contracts"] != experiment_manifest.get("target_contracts"):
        raise ExogenousFineTuneError(
            "Resolution holdout refusee: le contrat de cible canonique a change."
        )
    resolution = {
        "schema_version": 1,
        "frozen_panel_sha256": str(declared_panel_sha),
        "frozen_panel_audit_sha256": str(
            experiment_manifest.get("panel_audit_sha256")
        ),
        "resolved_panel_sha256": upstream["panel_sha256"],
        "resolved_panel_audit_sha256": upstream["panel_audit_sha256"],
        "input_contract_sha256": resolved_contract,
        "resolved_cells": int(binding["unresolved_cells"]),
        "all_other_values_identical": True,
        "labels_used_for_fit": False,
        "promotion_eligible": False,
    }
    return resolved, resolved_split, resolved_audit, resolution


def build_fit_inputs(
    panel: pd.DataFrame,
    origins: Sequence[pd.Timestamp],
    config: ExogenousFineTuneConfig,
) -> list[dict[str, Any]]:
    """Build exact-origin raw inputs accepted by ``Chronos2Pipeline.fit``."""

    selected = panel[panel[config.origin_column].isin(origins)]
    inputs: list[dict[str, Any]] = []
    for _, group in selected.groupby(
        [config.origin_column, config.item_column], sort=True, observed=True
    ):
        ordered = group.sort_values(config.timestamp_column)
        delivery_hours = int(_delivery_mask(ordered, config).sum())
        # Chronos2Pipeline.fit has one prediction_length for the whole batch.
        # The physical 23/25-hour days remain in the outer evaluation reserve,
        # but are explicitly excluded from this fixed-24 fit.  No interpolation
        # or duplicated hour is introduced.
        if delivery_hours != config.prediction_length:
            continue
        target = ordered[list(config.target_columns)].to_numpy(dtype=np.float32).T
        past_covariates: dict[str, np.ndarray] = {}
        for column in config.past_only_covariates:
            values = ordered[column].to_numpy(dtype=np.float32)
            # These rows are labels' contemporaneous values and must never be
            # available to the model at the forecast horizon.
            values[-config.prediction_length :] = np.nan
            past_covariates[column] = values
        future_covariates: dict[str, np.ndarray] = {}
        for column in config.known_future_covariates:
            values = ordered[column].to_numpy(dtype=np.float32)
            past_covariates[column] = values
            future_covariates[column] = values[-config.prediction_length :].copy()
        inputs.append(
            {
                "target": target,
                "past_covariates": past_covariates,
                "future_covariates": future_covariates,
            }
        )
    if not inputs:
        raise ExogenousFineTuneError("Aucun exemple construit pour ce split.")
    return inputs


def _default_pipeline_loader(source: str | Path, kwargs: Mapping[str, Any]) -> Any:
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(source, **dict(kwargs))


def resolve_local_model_source(config: ExogenousFineTuneConfig) -> str | Path:
    """Resolve a pinned HF snapshot without allowing an implicit network HEAD.

    ``transformers`` may probe the Hub even with ``local_files_only=True`` when
    the original model id is passed to adapter discovery.  Passing the concrete
    snapshot directory avoids that behavior and also proves which revision was
    used.
    """

    raw_path = Path(config.model_id).expanduser()
    for candidate in (
        raw_path,
        config.project_root / raw_path if not raw_path.is_absolute() else raw_path,
    ):
        if candidate.is_dir():
            return candidate.resolve()
    if not config.local_files_only:
        return config.model_id
    if "/" not in config.model_id:
        raise ExogenousFineTuneError(
            f"Mode local strict: modele local introuvable: {config.model_id}."
        )

    cache_roots: list[Path] = []
    explicit_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit_cache:
        cache_roots.append(Path(explicit_cache).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home).expanduser() / "hub")
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    repository = "models--" + config.model_id.replace("/", "--")
    for cache_root in dict.fromkeys(path.resolve() for path in cache_roots):
        repository_root = cache_root / repository
        revision = config.model_revision
        if revision:
            snapshot = repository_root / "snapshots" / revision
            if snapshot.is_dir() and (snapshot / "config.json").is_file():
                return snapshot.resolve()
            continue
        ref = repository_root / "refs" / "main"
        if ref.is_file():
            commit = ref.read_text(encoding="utf-8").strip()
            snapshot = repository_root / "snapshots" / commit
            if snapshot.is_dir() and (snapshot / "config.json").is_file():
                return snapshot.resolve()
        snapshots_root = repository_root / "snapshots"
        snapshots = (
            [path for path in snapshots_root.iterdir() if path.is_dir()]
            if snapshots_root.is_dir()
            else []
        )
        if len(snapshots) == 1 and (snapshots[0] / "config.json").is_file():
            return snapshots[0].resolve()
    revision_label = config.model_revision or "revision non precisee"
    raise ExogenousFineTuneError(
        "Mode local strict: snapshot Hugging Face absent pour "
        f"{config.model_id}@{revision_label}. Aucun acces reseau ne sera tente."
    )


def _pin_adapter_base(checkpoint: Path, model_source: str | Path) -> None:
    """Make PEFT reload the already-resolved base snapshot, never the Hub id."""

    adapter_path = checkpoint / "adapter_config.json"
    if not adapter_path.is_file() or not Path(model_source).is_dir():
        return
    payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    payload["base_model_name_or_path"] = str(Path(model_source).resolve())
    _write_json(adapter_path, payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_or_reparse(path: Path) -> bool:
    """Return True for symlinks and Windows junction/reparse entries."""

    if path.is_symlink() or os.path.islink(path):
        return True
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _assert_no_reparse_component(path: Path, *, label: str) -> Path:
    """Make a path absolute while preserving and rejecting link components."""

    absolute = path.expanduser().absolute()
    for component in (absolute, *absolute.parents):
        if _is_link_or_reparse(component):
            raise ExogenousFineTuneError(
                f"{label}: lien symbolique/junction/reparse interdit: {component}."
            )
    return absolute.resolve()


def sha256_directory(path: Path) -> str:
    """Hash relative paths and bytes, making a checkpoint tamper-evident."""

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ExogenousFineTuneError(f"Checkpoint vide: {path}.")
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _range_payload(values: Sequence[pd.Timestamp]) -> dict[str, Any]:
    return {
        "count": len(values),
        "first_utc": values[0].isoformat() if values else None,
        "last_utc": values[-1].isoformat() if values else None,
    }


def _trainer_state_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExogenousFineTuneError(
            f"Provenance LoRA: {name} doit etre un entier >= {minimum}."
        )
    return int(value)


def _checkpoint_step_from_name(name: str, field: str) -> int:
    prefix = "checkpoint-"
    suffix = name[len(prefix) :] if name.startswith(prefix) else ""
    if not suffix.isdigit() or int(suffix) < 1:
        raise ExogenousFineTuneError(
            f"Provenance LoRA: {field} ne designe pas un checkpoint-<step> valide."
        )
    return int(suffix)


def _read_trainer_state(
    state_path: Path,
    *,
    run_directory: Path,
    expected_max_steps: int,
) -> dict[str, Any]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExogenousFineTuneError(
            f"Provenance LoRA: trainer_state illisible: {state_path}."
        ) from exc
    if not isinstance(state, Mapping):
        raise ExogenousFineTuneError(
            f"Provenance LoRA: trainer_state doit etre un objet JSON: {state_path}."
        )

    global_step = _trainer_state_int(state.get("global_step"), "global_step")
    declared_max_steps = _trainer_state_int(state.get("max_steps"), "max_steps")
    best_step = _trainer_state_int(
        state.get("best_global_step"), "best_global_step"
    )
    if declared_max_steps != expected_max_steps:
        raise ExogenousFineTuneError(
            "Provenance LoRA: trainer_state.max_steps divergent de "
            f"training.num_steps ({declared_max_steps} != {expected_max_steps})."
        )
    if global_step > expected_max_steps or best_step > global_step:
        raise ExogenousFineTuneError(
            "Provenance LoRA: ordre des etapes incoherent "
            f"(best={best_step}, state={global_step}, max={expected_max_steps})."
        )

    state_parent = state_path.parent.name
    if state_parent.startswith("checkpoint-"):
        state_checkpoint_step = _checkpoint_step_from_name(
            state_parent, "emplacement de trainer_state"
        )
        if state_checkpoint_step != global_step:
            raise ExogenousFineTuneError(
                "Provenance LoRA: global_step divergent du repertoire contenant "
                "trainer_state.json."
            )

    best_metric = state.get("best_metric")
    if (
        isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or not np.isfinite(float(best_metric))
    ):
        raise ExogenousFineTuneError(
            "Provenance LoRA: best_metric absent ou non fini."
        )
    best_eval_loss = float(best_metric)

    raw_checkpoint = state.get("best_model_checkpoint")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
        raise ExogenousFineTuneError(
            "Provenance LoRA: best_model_checkpoint absent."
        )
    # Transformers records the absolute staging path.  That path becomes stale
    # after the atomic rename, so the durable manifest deliberately seals only
    # the local checkpoint directory name.
    checkpoint_name = raw_checkpoint.replace("\\", "/").rstrip("/").split("/")[-1]
    checkpoint_step = _checkpoint_step_from_name(
        checkpoint_name, "best_model_checkpoint"
    )
    if checkpoint_step != best_step:
        raise ExogenousFineTuneError(
            "Provenance LoRA: best_model_checkpoint divergent de best_global_step."
        )
    if not (run_directory / checkpoint_name).is_dir():
        raise ExogenousFineTuneError(
            "Provenance LoRA: checkpoint source du meilleur modele absent du bundle."
        )

    history = state.get("log_history")
    if not isinstance(history, list):
        raise ExogenousFineTuneError(
            "Provenance LoRA: log_history absent ou invalide."
        )
    observed_losses: list[tuple[int, float]] = []
    for record in history:
        if not isinstance(record, Mapping) or "eval_loss" not in record:
            continue
        step_value = record.get("step")
        loss_value = record.get("eval_loss")
        if (
            isinstance(step_value, bool)
            or not isinstance(step_value, int)
            or step_value < 1
            or isinstance(loss_value, bool)
            or not isinstance(loss_value, (int, float))
            or not np.isfinite(float(loss_value))
        ):
            raise ExogenousFineTuneError(
                "Provenance LoRA: entree eval_loss invalide dans log_history."
            )
        observed_losses.append((int(step_value), float(loss_value)))
    if not observed_losses:
        raise ExogenousFineTuneError(
            "Provenance LoRA: aucune eval_loss dans trainer_state."
        )
    matching_losses = [loss for step, loss in observed_losses if step == best_step]
    if not matching_losses or not np.isclose(
        matching_losses[-1], best_eval_loss, rtol=1e-9, atol=1e-12
    ):
        raise ExogenousFineTuneError(
            "Provenance LoRA: best_metric ne correspond pas a eval_loss du best_step."
        )
    if not np.isclose(
        min(loss for _, loss in observed_losses),
        best_eval_loss,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ExogenousFineTuneError(
            "Provenance LoRA: best_metric n'est pas la meilleure eval_loss observee."
        )

    callbacks = state.get("stateful_callbacks")
    if isinstance(callbacks, Mapping) and any(
        "earlystopping" in str(name).replace("_", "").lower()
        for name in callbacks
    ):
        raise ExogenousFineTuneError(
            "Provenance LoRA: EarlyStopping detecte alors que le contrat impose "
            "un budget fixe."
        )

    return {
        "path": state_path,
        "global_step": global_step,
        "best_step": best_step,
        "best_eval_loss": best_eval_loss,
        "checkpoint_source": checkpoint_name,
    }


def _model_selection_provenance(
    run_directory: Path, *, expected_max_steps: int
) -> dict[str, Any]:
    state_paths = sorted(
        run_directory.rglob("trainer_state.json"),
        key=lambda path: path.relative_to(run_directory).as_posix(),
    )
    base = {
        "validation_used": True,
        "early_stopping_used": False,
        "strategy": "best_eval_loss_after_fixed_steps",
        "selection_metric": "eval_loss",
        "trainer_state_available": False,
        "trainer_state_relative_path": None,
        "trainer_state_sha256": None,
        "trainer_state_observed_step": None,
        "best_step": None,
        "best_eval_loss": None,
        "checkpoint_source": None,
        "max_steps_completed": None,
        "max_steps_expected": int(expected_max_steps),
        "max_steps_completion_evidence": None,
    }
    if not state_paths:
        return base

    records = [
        _read_trainer_state(
            path,
            run_directory=run_directory,
            expected_max_steps=expected_max_steps,
        )
        for path in state_paths
    ]
    latest_step = max(record["global_step"] for record in records)
    latest = [record for record in records if record["global_step"] == latest_step]
    signatures = {
        (
            record["best_step"],
            record["best_eval_loss"],
            record["checkpoint_source"],
        )
        for record in latest
    }
    if len(signatures) != 1:
        raise ExogenousFineTuneError(
            "Provenance LoRA: trainer_state concurrents et contradictoires."
        )
    selected = latest[-1]
    selected_path = selected["path"]
    base.update(
        {
            "trainer_state_available": True,
            "trainer_state_relative_path": selected_path.relative_to(
                run_directory
            ).as_posix(),
            "trainer_state_sha256": _sha256_file(selected_path),
            "trainer_state_observed_step": selected["global_step"],
            "best_step": selected["best_step"],
            "best_eval_loss": selected["best_eval_loss"],
            "checkpoint_source": selected["checkpoint_source"],
            # ``Chronos2Pipeline.fit`` returns only after Trainer.train has
            # exhausted max_steps, then reloads the best validation checkpoint
            # and saves the published adapter.  A retained best-checkpoint
            # trainer_state can therefore have global_step < max_steps.
            "max_steps_completed": int(expected_max_steps),
            "max_steps_completion_evidence": (
                "fit_returned_after_fixed_steps_with_matching_trainer_state"
            ),
        }
    )
    return base


def _schema_payload(config: ExogenousFineTuneConfig) -> dict[str, Any]:
    return {
        "format_version": 1,
        "timestamp_column": config.timestamp_column,
        "origin_column": config.origin_column,
        "item_column": config.item_column,
        "feature_available_at_column": config.feature_available_at_column,
        "target_columns": list(config.target_columns),
        "known_future_covariates": list(config.known_future_covariates),
        "past_only_covariates": list(config.past_only_covariates),
        "timezone": config.timezone,
        "cutoff_local_time": config.cutoff_local_time,
        "frequency": config.frequency,
        "context_length": config.context_length,
        "prediction_length": config.prediction_length,
    }


def verify_bundle(run_directory: str | Path) -> dict[str, Any]:
    """Verify schema/checkpoint hashes without loading the neural model."""

    run_dir = _assert_no_reparse_component(
        Path(run_directory), label="Artefact LoRA"
    )
    if not run_dir.is_dir():
        raise ExogenousFineTuneError(f"Artefact LoRA absent: {run_dir}.")
    manifest_path = run_dir / "experiment_manifest.json"
    schema_path = run_dir / "schema.json"
    checkpoint = run_dir / "checkpoint"
    if not checkpoint.is_dir() or _is_link_or_reparse(checkpoint):
        raise ExogenousFineTuneError(
            f"Artefact incomplet, checkpoint absent ou lien/reparse: {checkpoint}."
        )
    for path in (manifest_path, schema_path, checkpoint / "adapter_config.json"):
        if not path.is_file() or _is_link_or_reparse(path):
            raise ExogenousFineTuneError(
                f"Artefact incomplet, absent, non regulier ou lien/reparse: {path}."
            )
    for directory, directory_names, file_names in os.walk(
        checkpoint, followlinks=False
    ):
        parent = Path(directory)
        for name in directory_names:
            child = parent / name
            if _is_link_or_reparse(child):
                raise ExogenousFineTuneError(
                    f"Checkpoint refuse: lien/junction/reparse: {child}."
                )
        for name in file_names:
            child = parent / name
            if _is_link_or_reparse(child) or not child.is_file():
                raise ExogenousFineTuneError(
                    f"Checkpoint refuse: fichier non regulier ou lien/reparse: {child}."
                )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_digest = _sha256_file(schema_path)
    checkpoint_digest = sha256_directory(checkpoint)
    if schema_digest != manifest.get("schema_sha256"):
        raise ExogenousFineTuneError("schema.json checksum mismatch.")
    if checkpoint_digest != manifest.get("checkpoint_sha256"):
        raise ExogenousFineTuneError("checkpoint checksum mismatch.")
    if manifest.get("finetune_mode") != "lora":
        raise ExogenousFineTuneError("Artefact refuse: finetune_mode n'est pas LoRA.")
    training = manifest.get("training")
    if isinstance(training, Mapping) and "model_selection" in training:
        declared_selection = training["model_selection"]
        if not isinstance(declared_selection, Mapping):
            raise ExogenousFineTuneError(
                "Provenance LoRA: training.model_selection doit etre un objet."
            )
        expected_steps = _trainer_state_int(
            training.get("num_steps"), "training.num_steps"
        )
        observed_selection = _model_selection_provenance(
            run_dir, expected_max_steps=expected_steps
        )
        if dict(declared_selection) != observed_selection:
            raise ExogenousFineTuneError(
                "Provenance LoRA: training.model_selection divergent des "
                "trainer_state scelles."
            )
    # A recovered training snapshot carries an optional, checksum-pinned
    # lineage sidecar.  Import lazily to keep the ordinary trainer independent
    # and to avoid a module cycle while the recovery utility itself reuses
    # ``verify_bundle``.
    if "training_snapshot_recovery" in manifest:
        from .training_snapshot_recovery import verify_recovery_reference

        verify_recovery_reference(run_dir, manifest)
    return manifest


def load_checkpoint(
    run_directory: str | Path,
    *,
    pipeline_loader: PipelineLoader | None = None,
    device_map: str = "auto",
) -> Any:
    """Verify then reload the saved LoRA adapter through Chronos-2."""

    run_dir = Path(run_directory).expanduser().resolve()
    manifest = verify_bundle(run_dir)
    expected_base_hash = manifest.get("base_model_snapshot_sha256")
    if not _is_sha256(expected_base_hash):
        raise ExogenousFineTuneError(
            "Artefact refuse: base_model_snapshot_sha256 absent ou invalide."
        )
    adapter_config_path = run_dir / "checkpoint" / "adapter_config.json"
    try:
        adapter_config = _mapping(
            json.loads(adapter_config_path.read_text(encoding="utf-8")),
            "adapter_config",
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ExogenousFineTuneError(
            f"Adapter config illisible: {adapter_config_path}."
        ) from exc
    base_value = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_value, str) or not base_value.strip():
        raise ExogenousFineTuneError(
            "Adapter refuse: base_model_name_or_path absent."
        )
    base_snapshot = Path(base_value).expanduser()
    if not base_snapshot.is_absolute():
        base_snapshot = (run_dir / "checkpoint" / base_snapshot).resolve()
    else:
        base_snapshot = base_snapshot.resolve()
    if not base_snapshot.is_dir():
        raise ExogenousFineTuneError(
            f"Snapshot Chronos-2 epingle absent: {base_snapshot}."
        )
    actual_base_hash = sha256_directory(base_snapshot)
    if actual_base_hash != expected_base_hash:
        raise ExogenousFineTuneError(
            "Snapshot Chronos-2 refuse: checksum divergent du modele de base "
            "utilise pendant le fine-tuning."
        )
    loader = pipeline_loader or _default_pipeline_loader
    return loader(
        run_dir / "checkpoint",
        {
            "device_map": device_map,
            "local_files_only": True,
            # PEFT 0.20 delegates the base architecture import to Transformers'
            # dynamic loader. Keep this adapter-only allowlist as narrow as
            # possible; the ordinary base-model load in ``train_lora`` does
            # not receive it.
            "import_allowlist": ["chronos.chronos2.model"],
        },
    )


EVALUATION_COLUMNS = (
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


def publish_evaluation_evidence(
    run_directory: str | Path,
    frame: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and publish the gate's canonical rolling evaluation evidence.

    Forecast computation intentionally remains outside the trainer.  This
    function accepts only already-produced, paired baseline/candidate rows and
    refuses missing hours, quantile crossings, wrong D-1 cut-offs and anything
    other than the exact physical holdout declared in the manifest.
    """

    run_dir = Path(run_directory).expanduser().resolve()
    manifest = verify_bundle(run_dir)
    if (
        manifest.get("production_pipeline_evidence") is True
        or manifest.get("candidate_output_stage")
        == "exogenous_residual_corrected"
    ):
        raise ExogenousFineTuneError(
            "Evaluation brute refusee: la preuve du pipeline final est deja "
            "scellee. Utilisez un nouveau repertoire de run pour un nouveau "
            "backtest; aucun downgrade/relabel de la preuve finale n'est permis."
        )
    schema = json.loads((run_dir / "schema.json").read_text(encoding="utf-8"))
    if tuple(frame.columns) != EVALUATION_COLUMNS:
        raise ExogenousFineTuneError(
            "Evaluation: schema exact requis: " + ", ".join(EVALUATION_COLUMNS) + "."
        )
    evidence = frame.copy()
    for column in ("delivery_start_utc", "forecast_origin_utc"):
        evidence[column] = pd.to_datetime(evidence[column], errors="coerce", utc=True)
        if evidence[column].isna().any():
            raise ExogenousFineTuneError(f"Evaluation: timestamps invalides dans {column}.")
    numeric_columns = list(EVALUATION_COLUMNS[2:])
    for column in numeric_columns:
        evidence[column] = pd.to_numeric(evidence[column], errors="coerce")
    if not np.isfinite(evidence[numeric_columns].to_numpy(dtype=float)).all():
        raise ExogenousFineTuneError("Evaluation: NaN/infini interdit dans prix ou quantiles.")
    if evidence["delivery_start_utc"].duplicated().any():
        raise ExogenousFineTuneError("Evaluation: heure de livraison dupliquee.")
    evidence = evidence.sort_values("delivery_start_utc", kind="stable").reset_index(drop=True)
    timestamps = pd.DatetimeIndex(evidence["delivery_start_utc"])
    if len(timestamps) > 1 and not np.all(np.diff(timestamps.asi8) == 3_600_000_000_000):
        raise ExogenousFineTuneError("Evaluation: couverture UTC non horaire ou discontinue.")
    for prefix in ("baseline", "candidate"):
        values = evidence[[f"{prefix}_q10", f"{prefix}_q50", f"{prefix}_q90"]].to_numpy(
            dtype=float
        )
        if not (np.all(values[:, 0] <= values[:, 1]) and np.all(values[:, 1] <= values[:, 2])):
            raise ExogenousFineTuneError(f"Evaluation: croisement de quantiles {prefix}.")

    timezone_name = str(schema["timezone"])
    local_delivery = timestamps.tz_convert(timezone_name)
    local_days = pd.Index([timestamp.date() for timestamp in local_delivery])
    unique_days = sorted(set(local_days))
    expected_days = int(manifest["evaluation_days"])
    if len(unique_days) != expected_days:
        raise ExogenousFineTuneError(
            f"Evaluation: {len(unique_days)} jours physiques != {expected_days}."
        )
    if len(unique_days) > 1:
        day_index = pd.DatetimeIndex(unique_days)
        if not np.all(
            np.diff(day_index.to_numpy(dtype="datetime64[D]")).astype(int) == 1
        ):
            raise ExogenousFineTuneError("Evaluation: jours de livraison non consecutifs.")

    cutoff_h, cutoff_m = (int(part) for part in str(schema["cutoff_local_time"]).split(":"))
    physical_day_audit: list[dict[str, Any]] = []
    for day in unique_days:
        mask = np.asarray(local_days == day)
        day_frame = evidence.loc[mask]
        day_origins = pd.DatetimeIndex(day_frame["forecast_origin_utc"].drop_duplicates())
        if len(day_origins) != 1:
            raise ExogenousFineTuneError(
                f"Evaluation {day}: plusieurs origines ou origine manquante."
            )
        origin_local = day_origins[0].tz_convert(timezone_name)
        if (
            origin_local.hour != cutoff_h
            or origin_local.minute != cutoff_m
            or (origin_local + pd.DateOffset(days=1)).date() != day
        ):
            raise ExogenousFineTuneError(
                f"Evaluation {day}: origine non D-1 {schema['cutoff_local_time']}."
            )
        start = pd.Timestamp(day).tz_localize(timezone_name)
        end = start + pd.DateOffset(days=1)
        expected_hours = int(
            (end.tz_convert("UTC") - start.tz_convert("UTC")) / pd.Timedelta(hours=1)
        )
        observed_hours = int(mask.sum())
        if observed_hours != expected_hours:
            raise ExogenousFineTuneError(
                f"Evaluation {day}: {observed_hours} heures != jour civil {expected_hours}."
            )
        physical_day_audit.append(
            {
                "delivery_day": day.isoformat(),
                "hours": observed_hours,
                "forecast_origin_utc": day_origins[0].isoformat(),
            }
        )

    declared_holdout = manifest["splits"]["evaluation_holdout"]
    distinct_origins = pd.DatetimeIndex(
        evidence["forecast_origin_utc"].drop_duplicates()
    ).sort_values()
    if (
        len(distinct_origins) != int(declared_holdout["count"])
        or distinct_origins[0].isoformat() != declared_holdout["first_utc"]
        or distinct_origins[-1].isoformat() != declared_holdout["last_utc"]
    ):
        raise ExogenousFineTuneError(
            "Evaluation: les origines ne correspondent pas au holdout gele du manifest."
        )

    output = run_dir / "evaluation_predictions.csv.gz"
    if output.exists() and not overwrite:
        raise ExogenousFineTuneError(
            f"Evidence deja publiee: {output}; overwrite explicite requis."
        )
    temporary = run_dir / f".{output.name}.tmp-{uuid.uuid4().hex}"
    evidence.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, output)
    manifest["evaluation_evidence"] = {
        "relative_path": output.name,
        "sha256": _sha256_file(output),
        "rows": int(len(evidence)),
        "physical_days": len(unique_days),
        "first_delivery_utc": timestamps[0].isoformat(),
        "last_delivery_utc": timestamps[-1].isoformat(),
        "dst_days": [record for record in physical_day_audit if record["hours"] != 24],
    }
    manifest_tmp = run_dir / f".experiment_manifest.json.tmp-{uuid.uuid4().hex}"
    _write_json(manifest_tmp, manifest)
    os.replace(manifest_tmp, run_dir / "experiment_manifest.json")
    return output


def train_lora(
    config_or_path: ExogenousFineTuneConfig | str | Path,
    *,
    overwrite: bool = False,
    pipeline_loader: PipelineLoader | None = None,
    panel: pd.DataFrame | None = None,
    model_source_override: str | Path | None = None,
) -> Path:
    """Train and publish an immutable, auditable LoRA challenger bundle."""

    config = (
        config_or_path
        if isinstance(config_or_path, ExogenousFineTuneConfig)
        else load_config(config_or_path)
    )
    if pipeline_loader is None and importlib.util.find_spec("peft") is None:
        raise ExogenousFineTuneError(
            "PEFT n'est pas installe. Chronos-2 basculerait silencieusement vers un "
            "fine-tuning complet; installation requise: pip install peft."
        )
    source_panel = read_panel(config.panel_path)
    if panel is not None:
        try:
            pd.testing.assert_frame_equal(
                panel.reset_index(drop=True),
                source_panel.reset_index(drop=True),
                check_dtype=False,
                check_like=False,
            )
        except AssertionError as exc:
            raise ExogenousFineTuneError(
                "Le panel injecte differe du fichier lie a l'audit PIT."
            ) from exc
    validated, split, pit_audit = validate_panel(source_panel, config)
    train_inputs = build_fit_inputs(validated, split.train, config)
    validation_inputs = build_fit_inputs(validated, split.validation, config)

    variates = len(config.target_columns) + len(config.covariate_columns)
    if config.batch_size < variates:
        raise ExogenousFineTuneError(
            f"batch_size={config.batch_size} < {variates} variates par groupe."
        )
    destination = config.output_directory
    if destination.exists() and not overwrite:
        raise ExogenousFineTuneError(
            f"Sortie deja existante: {destination}; utiliser --overwrite explicitement."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    backup: Path | None = None
    loader = pipeline_loader or _default_pipeline_loader
    load_kwargs: dict[str, Any] = {
        "device_map": config.device_map,
        "local_files_only": config.local_files_only,
    }
    if config.model_revision:
        load_kwargs["revision"] = config.model_revision

    random.seed(config.seed)
    np.random.seed(config.seed)
    try:
        model_source: str | Path = (
            resolve_local_model_source(config)
            if model_source_override is None
            else Path(model_source_override).expanduser().resolve()
        )
        if not Path(str(model_source)).is_dir():
            raise ExogenousFineTuneError(
                "Fine-tuning refuse: un snapshot local Chronos-2 est obligatoire "
                "pour etablir base_model_snapshot_sha256."
            )
        # A concrete local snapshot already pins the revision. Passing a Hub
        # revision alongside a filesystem path is unnecessary and can trigger
        # version-dependent adapter discovery behavior.
        if Path(str(model_source)).is_dir():
            load_kwargs.pop("revision", None)
        base = loader(model_source, load_kwargs)
        fit_kwargs = {
            "inputs": train_inputs,
            "validation_inputs": validation_inputs,
            "prediction_length": config.prediction_length,
            "finetune_mode": "lora",
            "lora_config": dict(config.lora_config),
            "context_length": config.context_length,
            "min_past": config.context_length,
            "learning_rate": config.learning_rate,
            "num_steps": config.num_steps,
            "batch_size": config.batch_size,
            "output_dir": staging,
            "finetuned_ckpt_name": "checkpoint",
            "seed": config.seed,
            "data_seed": config.seed,
            "remove_printer_callback": True,
        }
        finetuned = base.fit(**fit_kwargs)
        checkpoint = staging / "checkpoint"
        if not checkpoint.is_dir():
            checkpoint.mkdir(parents=True, exist_ok=True)
            finetuned.save_pretrained(checkpoint)
        _pin_adapter_base(checkpoint, model_source)
        if not (checkpoint / "adapter_config.json").is_file():
            raise ExogenousFineTuneError(
                "Le checkpoint ne contient pas adapter_config.json: fallback full "
                "fine-tuning refuse."
            )

        model_selection = _model_selection_provenance(
            staging, expected_max_steps=config.num_steps
        )

        schema_path = staging / "schema.json"
        _write_json(schema_path, _schema_payload(config))
        manifest = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": config.experiment_id,
            "evaluation_role": config.evaluation_role,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "base_model_resolved_locally": Path(str(model_source)).is_dir(),
            "base_model_snapshot_sha256": sha256_directory(
                Path(str(model_source))
            ),
            "finetune_mode": "lora",
            "training_window_days": config.training_window_days,
            "evaluation_days": config.evaluation_days,
            "cutoff_local_time": config.cutoff_local_time,
            "candidate_frozen_before_evaluation": True,
            "feature_selection_frozen_before_evaluation": True,
            "actual_future_used_as_input": False,
            "storm_used_for_input": False,
            "storm_used_for_selection": False,
            "mkonline_used_for_input": False,
            "pit_audit_passed": bool(pit_audit["pit_audit_passed"]),
            # Distinct from the mechanical available_at <= origin check above.
            # This flag may be true only when the upstream materializer itself
            # supplies archived/vintaged evidence acceptable for production.
            "production_pit_evidence": bool(
                pit_audit["production_pit_evidence"]
            ),
            "production_pit_evidence_detail": {
                "availability_column": config.feature_available_at_column,
                "verified_rows": int(pit_audit["rows"]),
                "late_rows": 0,
                "required_by_config": config.production_pit_evidence,
                "derived_from_panel_audit": bool(
                    pit_audit["production_pit_evidence"]
                ),
            },
            # This trainer compares the raw foundation model and its raw LoRA
            # adaptation. It cannot claim evidence for the operational chain
            # until a future OOF evaluator seals predictions after the final
            # residual corrector and against the true incumbent pipeline.
            # Deliberately not configurable from YAML.
            "production_pipeline_evidence": False,
            "production_pipeline_evidence_detail": {
                "components": [
                    "chronos2_base_raw",
                    "chronos2_exogenous_lora_raw",
                ],
                "missing_components": [
                    "final_residual_corrector_oof",
                    "operational_incumbent_pipeline",
                ],
                "comparison_scope": "raw_chronos2_base_vs_raw_lora",
                "promotion_eligible": False,
            },
            "checkpoint_relative_path": "checkpoint",
            "schema_relative_path": "schema.json",
            "checkpoint_sha256": sha256_directory(checkpoint),
            "schema_sha256": _sha256_file(schema_path),
            "panel_sha256": pit_audit["upstream_panel_audit"]["panel_sha256"],
            "panel_audit_sha256": pit_audit["upstream_panel_audit"][
                "panel_audit_sha256"
            ],
            "source_hashes": pit_audit["upstream_panel_audit"]["source_hashes"],
            "source_audit_hashes": pit_audit["upstream_panel_audit"][
                "source_audit_hashes"
            ],
            "source_cutoff_timezones": pit_audit["upstream_panel_audit"][
                "source_cutoff_timezones"
            ],
            "panel_pack": pit_audit["upstream_panel_audit"]["pack"],
            "target_sources": pit_audit["upstream_panel_audit"]["target_sources"],
            "target_contracts": pit_audit["upstream_panel_audit"][
                "target_contracts"
            ],
            "panel_audit_summary": {
                "audit_path": pit_audit["upstream_panel_audit"]["audit_path"],
                "zones": pit_audit["upstream_panel_audit"]["zones"],
                "declared_production_ready": pit_audit[
                    "upstream_panel_audit"
                ]["declared_production_ready"],
                "zone_production_pit_evidence": pit_audit[
                    "upstream_panel_audit"
                ]["zone_production_pit_evidence"],
                "production_blockers": pit_audit["upstream_panel_audit"][
                    "production_blockers"
                ],
            },
            "evaluation_label_binding": pit_audit[
                "evaluation_label_binding"
            ],
            "splits": {
                "train": _range_payload(split.train),
                "validation": _range_payload(split.validation),
                "evaluation_holdout": _range_payload(split.evaluation),
            },
            "pit_audit": pit_audit,
            "training": {
                "learning_rate": config.learning_rate,
                "num_steps": config.num_steps,
                "batch_size": config.batch_size,
                "seed": config.seed,
                "lora_config": dict(config.lora_config),
                "train_samples": len(train_inputs),
                "validation_samples": len(validation_inputs),
                "model_selection": model_selection,
            },
        }
        _write_json(staging / "experiment_manifest.json", manifest)
        verify_bundle(staging)

        if destination.exists():
            backup = destination.parent / (
                f".{destination.name}.backup-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )
            os.replace(destination, backup)
        os.replace(staging, destination)
        return destination
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise


def config_summary(config: ExogenousFineTuneConfig) -> dict[str, Any]:
    """Return a JSON-safe view used by CLI validation and tests."""

    payload = asdict(config)
    for key in (
        "config_path",
        "project_root",
        "panel_path",
        "panel_audit_path",
        "output_directory",
    ):
        payload[key] = str(payload[key])
    payload["target_columns"] = list(config.target_columns)
    payload["known_future_covariates"] = list(config.known_future_covariates)
    payload["past_only_covariates"] = list(config.past_only_covariates)
    return payload


__all__ = [
    "EVALUATION_ROLES",
    "EVALUATION_COLUMNS",
    "ExogenousFineTuneConfig",
    "ExogenousFineTuneError",
    "OriginSplit",
    "bind_resolved_evaluation_panel",
    "build_fit_inputs",
    "config_summary",
    "load_checkpoint",
    "load_config",
    "load_panel_audit",
    "publish_evaluation_evidence",
    "read_panel",
    "resolve_local_model_source",
    "sha256_directory",
    "train_lora",
    "validate_panel",
    "verify_bundle",
]
