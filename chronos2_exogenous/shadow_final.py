"""Build governance evidence for the *final* prospective LoRA pipeline.

The ordinary shadow evaluator intentionally keeps an append-only journal of
raw Chronos-2/LoRA forecasts.  That journal is useful audit material, but it is
not the model that would be deployed.  This module derives a separate,
immutable snapshot where:

* the candidate is the raw LoRA forecast plus the sealed OOF residual
  corrector;
* the baseline is the genuinely issued ``residual_corrected`` incumbent;
* deliveries, forecast origins and observed prices are paired exactly; and
* every input is bound to the result by SHA-256 provenance.

The raw append-only journal and its v3 manifest are never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .evaluation import (
    SHADOW_JOURNAL_COLUMNS,
    _canonical_record_hash,
    _validate_shadow_provenance_rows,
)
from .final_pipeline import (
    ACTUAL_PAIRING_ATOL_EUR_MWH,
    INCUMBENT_COLUMNS,
    _normalise_evidence,
    _normalise_incumbent,
    _validate_pairing,
)
from .governance import (
    ExogenousGovernanceError,
    validate_experiment_manifest,
    validate_final_shadow_manifest,
    validate_shadow_manifest,
)
from .lora_finetune import EVALUATION_COLUMNS, verify_bundle
from .production import (
    BASE_MODEL,
    OUTPUT_MODEL,
    ExogenousProductionError,
    _apply_residual_corrector,
    _validate_corrector,
    _validate_final_pipeline_evidence,
)


FINAL_SHADOW_DIRECTORY_NAME = "shadow_final"
FINAL_SHADOW_EVIDENCE_NAME = "shadow_final_evidence.csv.gz"
FINAL_SHADOW_MANIFEST_NAME = "shadow_final_manifest.json"
FINAL_SHADOW_INCUMBENT_NAME = "shadow_final_incumbent.csv.gz"
RAW_SHADOW_EVIDENCE_COPY_NAME = "raw_shadow_observed_evidence.csv.gz"
RAW_SHADOW_MANIFEST_COPY_NAME = "raw_shadow_manifest.json"
RESIDUAL_CORRECTOR_COPY_NAME = "residual_corrector.json"
OOF_AUDIT_COPY_NAME = "oof_audit.json"
SCHEMA_COPY_NAME = "schema.json"
EXPERIMENT_MANIFEST_COPY_NAME = "experiment_manifest.json"
RAW_SHADOW_JOURNAL_COPY_NAME = "raw_shadow_journal.csv.gz"
FINAL_SHADOW_ISSUED_NAME = "shadow_final_issued_history.csv.gz"
ISSUED_COLUMNS = (
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


class FinalShadowError(RuntimeError):
    """Raised before publication when final shadow proof is incomplete."""


@dataclass(frozen=True)
class FinalShadowArtifacts:
    """Published final-shadow paths, or a harmless no-observation result."""

    pending: bool
    output_directory: Path
    evidence_path: Path | None
    manifest_path: Path | None
    rows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FinalShadowError(f"{label} absent: {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalShadowError(f"{label} illisible: {path}.") from exc
    if not isinstance(payload, dict):
        raise FinalShadowError(f"{label} doit etre un objet JSON.")
    return payload


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FinalShadowError(f"{label} absent: {path}.")
    suffixes = "".join(path.suffixes).lower()
    if not (suffixes.endswith(".csv") or suffixes.endswith(".csv.gz")):
        raise FinalShadowError(f"{label}: seul CSV/CSV gzip est accepte.")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise FinalShadowError(f"{label} illisible: {path}.") from exc


def _safe_bundle_reference(run_directory: Path, reference: object, *, label: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise FinalShadowError(f"{label}: relative_path absent.")
    target = (run_directory / reference).resolve()
    if target != run_directory and run_directory not in target.parents:
        raise FinalShadowError(f"{label}: chemin hors bundle LoRA.")
    return target


def _validate_final_rolling_lineage(
    run_directory: Path,
    experiment: Mapping[str, Any],
) -> dict[str, str]:
    """Verify the final rolling proof that authorised this shadow phase."""

    if experiment.get("production_pipeline_evidence") is not True:
        raise FinalShadowError(
            "Le shadow final exige d'abord un FinalBacktest scelle."
        )
    try:
        _validate_final_pipeline_evidence(experiment)
    except ExogenousProductionError as exc:
        raise FinalShadowError(
            "Le shadow final exige d'abord un FinalBacktest scelle."
        ) from exc
    rolling = experiment.get("evaluation_evidence")
    if not isinstance(rolling, Mapping):
        raise FinalShadowError("evaluation_evidence finale absente du manifeste.")
    rolling_path = _safe_bundle_reference(
        run_directory, rolling.get("relative_path"), label="rolling365 final"
    )
    rolling_sha = rolling.get("sha256")
    if not _is_sha256(rolling_sha) or not rolling_path.is_file() or _sha256(
        rolling_path
    ) != rolling_sha:
        raise FinalShadowError("SHA de la preuve rolling365 finale divergent.")

    final_reference = experiment.get("final_pipeline_evaluation")
    if not isinstance(final_reference, Mapping):
        raise FinalShadowError("final_pipeline_evaluation absente du manifeste.")
    final_manifest_path = _safe_bundle_reference(
        run_directory,
        final_reference.get("relative_path"),
        label="manifeste FinalBacktest",
    )
    final_manifest_sha = final_reference.get("sha256")
    if not _is_sha256(final_manifest_sha) or not final_manifest_path.is_file() or _sha256(
        final_manifest_path
    ) != final_manifest_sha:
        raise FinalShadowError("SHA du manifeste FinalBacktest divergent.")
    return {
        "relative_path": str(rolling.get("relative_path")),
        "sha256": str(rolling_sha),
        "manifest_relative_path": str(final_reference.get("relative_path")),
        "manifest_sha256": str(final_manifest_sha),
    }


def _validate_corrector_lineage(
    *,
    corrector_path: Path,
    oof_audit_path: Path,
    experiment: Mapping[str, Any],
    first_shadow_day: str | None,
) -> dict[str, Any]:
    try:
        corrector = _validate_corrector(corrector_path, experiment=experiment)
    except ExogenousProductionError as exc:
        raise FinalShadowError("Correcteur residuel OOF refuse.") from exc
    audit_sha = _sha256(oof_audit_path) if oof_audit_path.is_file() else ""
    if (
        not _is_sha256(audit_sha)
        or corrector.get("oof_training_audit_sha256") != audit_sha
        or experiment.get("oof_training_audit_sha256") != audit_sha
    ):
        raise FinalShadowError(
            "Le sidecar OOF ne correspond pas au correcteur et au FinalBacktest."
        )
    audit = _json(oof_audit_path, label="sidecar audit OOF")
    required: Mapping[str, object] = {
        "purpose": "chronos2_exogenous_blocked_prequential_oof",
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "candidate_model": BASE_MODEL,
        "training_days": 365,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "future_actuals_used_as_features": False,
        "holdout_used_for_fit": False,
        "selection_frozen_before_oof": True,
    }
    failures = [
        f"{key}={audit.get(key)!r}, attendu={expected!r}"
        for key, expected in required.items()
        if type(audit.get(key)) is not type(expected) or audit.get(key) != expected
    ]
    if failures:
        raise FinalShadowError("Contrat OOF invalide: " + "; ".join(failures))
    if audit.get("candidate_checkpoint_sha256") != experiment.get(
        "checkpoint_sha256"
    ):
        raise FinalShadowError("Le sidecar OOF cible un autre checkpoint LoRA.")
    if audit.get("predictions_sha256") != corrector.get(
        "oof_training_predictions_sha256"
    ):
        raise FinalShadowError("Le sidecar OOF et le correcteur divergent.")
    if first_shadow_day is not None:
        try:
            training_end = pd.Timestamp(str(corrector["training_end_day"])).date()
            shadow_start = pd.Timestamp(first_shadow_day).date()
        except Exception as exc:
            raise FinalShadowError("Dates OOF/shadow invalides.") from exc
        if training_end >= shadow_start:
            raise FinalShadowError(
                "Le correcteur n'est pas gele avant la premiere livraison shadow."
            )
    return corrector


def _validate_complete_shadow_days(frame: pd.DataFrame, *, schema: Mapping[str, Any]) -> None:
    timezone_name = str(schema.get("timezone", "")).strip()
    cutoff = str(schema.get("cutoff_local_time", "")).strip()
    try:
        local_zone = ZoneInfo(timezone_name)
        cutoff_hour, cutoff_minute = (int(value) for value in cutoff.split(":"))
    except Exception as exc:
        raise FinalShadowError("Timezone/cutoff du schema invalides.") from exc
    deliveries = pd.DatetimeIndex(frame["delivery_start_utc"])
    origins = pd.DatetimeIndex(frame["forecast_origin_utc"])
    local_deliveries = deliveries.tz_convert(local_zone)
    local_days = tuple(dict.fromkeys(local_deliveries.date))
    if not local_days:
        raise FinalShadowError("La preuve shadow observee est vide.")
    if any(
        local_days[index] != local_days[0] + timedelta(days=index)
        for index in range(len(local_days))
    ):
        raise FinalShadowError("Les jours shadow observes ne sont pas consecutifs.")
    for day_local in local_days:
        mask = np.asarray(local_deliveries.date == day_local)
        start = pd.Timestamp(day_local).tz_localize(local_zone)
        end = pd.Timestamp(day_local + timedelta(days=1)).tz_localize(local_zone)
        expected = pd.date_range(
            start.tz_convert("UTC"),
            end.tz_convert("UTC"),
            freq="h",
            inclusive="left",
        )
        if not deliveries[mask].equals(expected):
            raise FinalShadowError(
                f"Jour shadow {day_local}: grille physique/DST incomplete."
            )
        unique_origins = origins[mask].unique()
        if len(unique_origins) != 1:
            raise FinalShadowError(
                f"Jour shadow {day_local}: origine unique requise."
            )
        expected_origin = pd.Timestamp(
            datetime.combine(
                day_local - timedelta(days=1),
                datetime.min.replace(
                    hour=cutoff_hour, minute=cutoff_minute
                ).time(),
            ),
            tz=local_zone,
        ).tz_convert("UTC")
        if pd.Timestamp(unique_origins[0]) != expected_origin:
            raise FinalShadowError(
                f"Jour shadow {day_local}: origine differente de D-1 {cutoff}."
            )


def _load_raw_issued_forecasts(
    journal_path: Path,
    *,
    raw_manifest: Mapping[str, Any],
    experiment: Mapping[str, Any],
    zone: str,
) -> pd.DataFrame:
    """Verify the full append-only chain and return one row per issued hour."""

    journal_reference = raw_manifest.get("journal")
    if not isinstance(journal_reference, Mapping):
        raise FinalShadowError(
            "Le manifeste brut ne scelle pas le journal append-only complet."
        )
    if not journal_path.is_file() or journal_reference.get("sha256") != _sha256(
        journal_path
    ):
        raise FinalShadowError("SHA du journal shadow brut divergent.")
    journal = _read_csv(journal_path, label="journal shadow brut")
    if tuple(journal.columns) != SHADOW_JOURNAL_COLUMNS:
        raise FinalShadowError("Schema du journal shadow brut inconnu.")
    if journal_reference.get("records") != len(journal):
        raise FinalShadowError("Nombre de records du journal shadow divergent.")
    for column in (
        "delivery_start_utc",
        "forecast_origin_utc",
        "panel_created_at_utc",
        "captured_at_utc",
    ):
        journal[column] = pd.to_datetime(journal[column], utc=True, errors="coerce")
        if journal[column].isna().any():
            raise FinalShadowError(f"Journal shadow: timestamp invalide dans {column}.")
    previous = ""
    for index, row in journal.iterrows():
        actual_previous = (
            ""
            if pd.isna(row["previous_record_sha256"])
            else str(row["previous_record_sha256"])
        )
        if actual_previous != previous or _canonical_record_hash(row) != str(
            row["record_sha256"]
        ):
            raise FinalShadowError(
                f"Chaine de hashes du journal rompue a la ligne {index}."
            )
        previous = str(row["record_sha256"])
    if journal_reference.get("last_record_sha256") != previous:
        raise FinalShadowError("Dernier hash du journal shadow divergent.")

    keys = [
        "delivery_start_utc",
        "forecast_origin_utc",
        "item_id",
        "target_column",
    ]
    if journal_reference.get("unique_hours") != len(journal.drop_duplicates(keys)):
        raise FinalShadowError("Nombre d'heures uniques du journal divergent.")
    forecasts = journal.loc[journal["record_kind"].eq("forecast")].copy()
    if forecasts.empty or forecasts.duplicated(keys).any():
        raise FinalShadowError(
            "Le journal doit contenir exactement un record forecast par cle."
        )
    try:
        panel_evidence = _validate_shadow_provenance_rows(forecasts)
    except Exception as exc:
        raise FinalShadowError(
            "Provenance panel des forecasts emis invalide."
        ) from exc
    if not panel_evidence or any(
        evidence.get("production_pit_evidence") is not True
        or evidence.get("horizon_actuals_present") is not False
        for evidence in panel_evidence.values()
    ):
        raise FinalShadowError(
            "Chaque forecast emis exige un panel PIT production-ready sans actual D+1."
        )
    latest = journal.drop_duplicates(keys, keep="last").loc[
        :, [*keys, "actual"]
    ]
    forecasts = forecasts.drop(columns=["actual"]).merge(
        latest, on=keys, how="left", validate="one_to_one"
    )
    if forecasts["item_id"].astype(str).str.strip().str.upper().nunique() != 1 or (
        forecasts["item_id"].astype(str).str.strip().str.upper().iloc[0] != zone
    ):
        raise FinalShadowError("Zone/item du journal shadow incoherent.")
    if forecasts["checkpoint_sha256"].astype(str).nunique() != 1 or (
        str(forecasts["checkpoint_sha256"].iloc[0])
        != experiment.get("checkpoint_sha256")
    ):
        raise FinalShadowError("Checkpoint du journal shadow divergent.")
    for column in (
        "checkpoint_sha256",
        "input_contract_sha256",
        "panel_contract_sha256",
        "record_sha256",
    ):
        if not forecasts[column].astype(str).map(_is_sha256).all():
            raise FinalShadowError(f"Hash invalide dans le journal: {column}.")
    deadline = forecasts["forecast_origin_utc"] + pd.Timedelta(hours=4)
    created = forecasts["captured_at_utc"]
    if not bool(
        (
            (created >= forecasts["forecast_origin_utc"])
            & (created <= deadline)
        ).all()
    ):
        raise FinalShadowError("Forecast du journal hors fenetre prospective.")
    quantile_columns = [f"candidate_{quantile}" for quantile in ("q10", "q50", "q90")]
    forecasts[quantile_columns] = forecasts[quantile_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    quantiles = forecasts[quantile_columns].to_numpy(float)
    if not np.isfinite(quantiles).all() or bool(
        ((quantiles[:, 0] > quantiles[:, 1]) | (quantiles[:, 1] > quantiles[:, 2])).any()
    ):
        raise FinalShadowError("Quantiles LoRA du journal invalides/croises.")
    return forecasts.sort_values("delivery_start_utc", kind="stable").reset_index(
        drop=True
    )


def finalize_shadow_evidence(
    *,
    run_directory: str | Path,
    raw_observed_evidence_path: str | Path,
    raw_shadow_manifest_path: str | Path,
    residual_corrector_path: str | Path,
    oof_audit_path: str | Path,
    incumbent_statistics_path: str | Path,
    zone: str,
    raw_shadow_journal_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    allow_no_observed: bool = False,
) -> FinalShadowArtifacts:
    """Atomically publish paired final-pipeline prospective shadow evidence."""

    run_dir = Path(run_directory).expanduser().resolve()
    raw_path = Path(raw_observed_evidence_path).expanduser().resolve()
    raw_manifest_path = Path(raw_shadow_manifest_path).expanduser().resolve()
    destination = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else raw_path.parent / FINAL_SHADOW_DIRECTORY_NAME
    )
    experiment = verify_bundle(run_dir)
    if not isinstance(experiment, dict):
        experiment = dict(experiment)
    canonical_zone = str(zone).strip().upper()
    validate_experiment_manifest(experiment, zone=canonical_zone)
    rolling_lineage = _validate_final_rolling_lineage(run_dir, experiment)
    schema = _json(run_dir / "schema.json", label="schema LoRA")
    corrector_source = Path(residual_corrector_path).expanduser().resolve()
    oof_source = Path(oof_audit_path).expanduser().resolve()
    # Even a first, still-unobserved emission is allowed only after the model,
    # FinalBacktest and OOF corrector have all been frozen.  Returning pending
    # earlier would permit post-outcome candidate selection.
    _validate_corrector_lineage(
        corrector_path=corrector_source,
        oof_audit_path=oof_source,
        experiment=experiment,
        first_shadow_day=None,
    )
    if not raw_path.exists() and not raw_manifest_path.exists() and allow_no_observed:
        return FinalShadowArtifacts(True, destination, None, None, 0)
    if not raw_path.is_file() or not raw_manifest_path.is_file():
        raise FinalShadowError(
            "La preuve observee et le manifeste shadow brut doivent exister ensemble."
        )

    raw = _normalise_evidence(
        _read_csv(raw_path, label="preuve shadow brute"),
        label="preuve shadow brute",
    )
    _validate_complete_shadow_days(raw, schema=schema)
    raw_manifest = _json(raw_manifest_path, label="manifeste shadow brut")
    try:
        verified_raw_manifest = validate_shadow_manifest(
            raw_manifest,
            experiment_manifest=experiment,
            zone=canonical_zone,
            expected_rows=len(raw),
        )
    except ExogenousGovernanceError as exc:
        raise FinalShadowError("Manifeste shadow brut refuse.") from exc
    raw_sha = _sha256(raw_path)
    if verified_raw_manifest.get("predictions_sha256") != raw_sha:
        raise FinalShadowError(
            "Le manifeste shadow brut ne designe pas la preuve observee fournie."
        )
    observed_reference = verified_raw_manifest.get("observed_governance_evidence")
    if not isinstance(observed_reference, Mapping) or (
        observed_reference.get("sha256") != raw_sha
        or observed_reference.get("rows") != len(raw)
    ):
        raise FinalShadowError(
            "La reference observed_governance_evidence brute est incoherente."
        )
    journal_reference = verified_raw_manifest.get("journal")
    if not isinstance(journal_reference, Mapping):
        raise FinalShadowError(
            "Le manifeste shadow brut ne reference pas son journal complet."
        )
    if raw_shadow_journal_path is None:
        relative_journal = journal_reference.get("relative_path")
        if not isinstance(relative_journal, str) or not relative_journal.strip():
            raise FinalShadowError("Chemin du journal shadow brut absent.")
        relative_path = Path(relative_journal)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise FinalShadowError("Chemin du journal shadow brut non sur.")
        journal_source = (raw_manifest_path.parent / relative_path).resolve()
    else:
        journal_source = Path(raw_shadow_journal_path).expanduser().resolve()
    issued_raw = _load_raw_issued_forecasts(
        journal_source,
        raw_manifest=verified_raw_manifest,
        experiment=experiment,
        zone=canonical_zone,
    )
    _validate_complete_shadow_days(issued_raw, schema=schema)
    raw_keys = ["delivery_start_utc", "forecast_origin_utc"]
    raw_check = raw.merge(
        issued_raw.loc[
            :,
            [
                *raw_keys,
                "actual",
                "candidate_q10",
                "candidate_q50",
                "candidate_q90",
            ],
        ],
        on=raw_keys,
        suffixes=("_observed", "_journal"),
        how="left",
        validate="one_to_one",
    )
    if len(raw_check) != len(raw) or raw_check["actual_journal"].isna().any():
        raise FinalShadowError(
            "Les lignes observees ne sont pas toutes issues du journal scelle."
        )
    for column in ("actual", "candidate_q10", "candidate_q50", "candidate_q90"):
        if not np.allclose(
            raw_check[f"{column}_observed"].to_numpy(float),
            raw_check[f"{column}_journal"].to_numpy(float),
            rtol=0.0,
            atol=1e-9,
        ):
            raise FinalShadowError(
                f"La preuve observee diverge du journal brut sur {column}."
            )

    incumbent_source = Path(incumbent_statistics_path).expanduser().resolve()
    incumbent_all = _normalise_incumbent(
        _read_csv(incumbent_source, label="Statistics incumbent")
    )
    deliveries = pd.DatetimeIndex(raw["delivery_start_utc"])
    incumbent = incumbent_all.loc[
        incumbent_all["delivery_start_utc"].isin(deliveries)
    ].reset_index(drop=True)
    if len(incumbent) != len(raw):
        raise FinalShadowError(
            "L'incumbent ne contient pas exactement toutes les heures shadow observees."
        )
    actual_delta = _validate_pairing(raw, incumbent)

    first_shadow_day = (
        deliveries.tz_convert(str(schema["timezone"]))[0].date().isoformat()
    )
    corrector = _validate_corrector_lineage(
        corrector_path=corrector_source,
        oof_audit_path=oof_source,
        experiment=experiment,
        first_shadow_day=first_shadow_day,
    )

    timestamp_column = str(schema.get("timestamp_column", ""))
    if not timestamp_column:
        raise FinalShadowError("timestamp_column absent du schema LoRA.")
    horizon = pd.DataFrame({timestamp_column: deliveries})
    raw_candidate = np.stack(
        [
            raw[f"candidate_{quantile}"].to_numpy(dtype=float)
            for quantile in ("q10", "q50", "q90")
        ],
        axis=0,
    )[np.newaxis, :, :]
    try:
        corrected, shift = _apply_residual_corrector(
            raw_candidate,
            horizon=horizon,
            corrector=corrector,
            schema=schema,
        )
    except ExogenousProductionError as exc:
        raise FinalShadowError("Application du correcteur OOF impossible.") from exc

    issued_delivery = pd.DatetimeIndex(issued_raw["delivery_start_utc"])
    issued_horizon = pd.DataFrame({timestamp_column: issued_delivery})
    issued_candidate = np.stack(
        [
            issued_raw[f"candidate_{quantile}"].to_numpy(dtype=float)
            for quantile in ("q10", "q50", "q90")
        ],
        axis=0,
    )[np.newaxis, :, :]
    try:
        issued_corrected, issued_shift = _apply_residual_corrector(
            issued_candidate,
            horizon=issued_horizon,
            corrector=corrector,
            schema=schema,
        )
    except ExogenousProductionError as exc:
        raise FinalShadowError(
            "Application du correcteur OOF a l'historique emis impossible."
        ) from exc

    final = pd.DataFrame(
        {
            "delivery_start_utc": raw["delivery_start_utc"],
            "forecast_origin_utc": raw["forecast_origin_utc"],
            "actual": raw["actual"],
            "baseline_q10": incumbent["residual_corrected__q10"],
            "baseline_q50": incumbent["residual_corrected__q50"],
            "baseline_q90": incumbent["residual_corrected__q90"],
            "candidate_q10": corrected[0, 0, :],
            "candidate_q50": corrected[0, 1, :],
            "candidate_q90": corrected[0, 2, :],
        },
        columns=EVALUATION_COLUMNS,
    )
    final = _normalise_evidence(final, label="preuve shadow pipeline final")

    issued = pd.DataFrame(
        {
            "delivery_start_utc": issued_raw["delivery_start_utc"],
            "forecast_origin_utc": issued_raw["forecast_origin_utc"],
            "actual": pd.to_numeric(issued_raw["actual"], errors="coerce"),
            "candidate_q10": issued_corrected[0, 0, :],
            "candidate_q50": issued_corrected[0, 1, :],
            "candidate_q90": issued_corrected[0, 2, :],
            "item_id": issued_raw["item_id"].astype(str),
            "target_column": issued_raw["target_column"].astype(str),
            "checkpoint_sha256": issued_raw["checkpoint_sha256"].astype(str),
            "input_contract_sha256": issued_raw["input_contract_sha256"].astype(str),
            "panel_contract_sha256": issued_raw["panel_contract_sha256"].astype(str),
            "forecast_created_at_utc": issued_raw["captured_at_utc"],
            "forecast_record_sha256": issued_raw["record_sha256"].astype(str),
        },
        columns=ISSUED_COLUMNS,
    )
    overlap = final.merge(
        issued,
        on=["delivery_start_utc", "forecast_origin_utc"],
        how="left",
        suffixes=("_observed", "_issued"),
        validate="one_to_one",
    )
    if len(overlap) != len(final) or overlap["candidate_q50_issued"].isna().any():
        raise FinalShadowError(
            "L'historique final emis ne couvre pas toute la preuve observee."
        )
    for column in ("actual", "candidate_q10", "candidate_q50", "candidate_q90"):
        if not np.allclose(
            overlap[f"{column}_observed"].to_numpy(float),
            overlap[f"{column}_issued"].to_numpy(float),
            rtol=0.0,
            atol=1e-10,
        ):
            raise FinalShadowError(
                f"Historique final emis divergent sur l'overlap {column}."
            )

    source_paths = {
        "preuve shadow brute": raw_path,
        "manifeste shadow brut": raw_manifest_path,
        "Statistics incumbent": incumbent_source,
        "correcteur residuel": corrector_source,
        "audit OOF": oof_source,
        "journal shadow brut": journal_source,
    }
    resolved_destination = destination.resolve()
    for label, source in source_paths.items():
        if source == resolved_destination or resolved_destination in source.parents:
            raise FinalShadowError(f"Sortie shadow imbriquee dans {label}.")

    staging = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    backup: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=False, exist_ok=False)
        evidence_path = staging / FINAL_SHADOW_EVIDENCE_NAME
        final.to_csv(evidence_path, index=False, compression="gzip")
        incumbent_path = staging / FINAL_SHADOW_INCUMBENT_NAME
        incumbent.loc[:, list(INCUMBENT_COLUMNS)].to_csv(
            incumbent_path, index=False, compression="gzip"
        )
        raw_copy = staging / RAW_SHADOW_EVIDENCE_COPY_NAME
        raw_manifest_copy = staging / RAW_SHADOW_MANIFEST_COPY_NAME
        raw_journal_copy = staging / RAW_SHADOW_JOURNAL_COPY_NAME
        issued_path = staging / FINAL_SHADOW_ISSUED_NAME
        corrector_copy = staging / RESIDUAL_CORRECTOR_COPY_NAME
        oof_copy = staging / OOF_AUDIT_COPY_NAME
        schema_copy = staging / SCHEMA_COPY_NAME
        experiment_copy = staging / EXPERIMENT_MANIFEST_COPY_NAME
        shutil.copy2(raw_path, raw_copy)
        shutil.copy2(raw_manifest_path, raw_manifest_copy)
        shutil.copy2(journal_source, raw_journal_copy)
        issued.to_csv(issued_path, index=False, compression="gzip")
        shutil.copy2(corrector_source, corrector_copy)
        shutil.copy2(oof_source, oof_copy)
        shutil.copy2(run_dir / "schema.json", schema_copy)
        shutil.copy2(run_dir / "experiment_manifest.json", experiment_copy)

        evidence_sha = _sha256(evidence_path)
        lineage = {
            "raw_shadow_evidence_sha256": _sha256(raw_copy),
            "raw_shadow_manifest_sha256": _sha256(raw_manifest_copy),
            "raw_shadow_journal_sha256": _sha256(raw_journal_copy),
            "paired_incumbent_sha256": _sha256(incumbent_path),
            "issued_shadow_history_sha256": _sha256(issued_path),
            "residual_corrector_sha256": _sha256(corrector_copy),
            "oof_training_audit_sha256": _sha256(oof_copy),
            "rolling365_final_evidence_sha256": rolling_lineage["sha256"],
            "rolling365_final_manifest_sha256": rolling_lineage[
                "manifest_sha256"
            ],
            "checkpoint_sha256": str(experiment["checkpoint_sha256"]),
            "schema_sha256": str(experiment["schema_sha256"]),
            "final_shadow_predictions_sha256": evidence_sha,
            "rows": int(len(final)),
            "issued_rows": int(len(issued)),
            "baseline_output_stage": "residual_corrected",
            "candidate_output_stage": OUTPUT_MODEL,
        }
        derivation_sha = hashlib.sha256(_json_bytes(lineage)).hexdigest()
        final_manifest: dict[str, Any] = {
            "format_version": 4,
            "kind": "chronos2_exogenous_final_pipeline_shadow",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_model": str(
                experiment.get("experiment_id", experiment.get("model_id"))
            ).strip(),
            "zone": canonical_zone,
            "checkpoint_sha256": experiment["checkpoint_sha256"],
            "schema_sha256": experiment["schema_sha256"],
            "candidate_frozen_before_shadow": True,
            "actuals_attached_after_forecast_freeze": True,
            "prospective_capture_deadline_enforced": True,
            "prospective_capture_deadline_hours": 4,
            "forecast_artifact_checksums_valid": True,
            "storm_used_for_prediction": False,
            "mkonline_used_for_prediction": False,
            "predictions_sha256": evidence_sha,
            "forecast_created_at_utc": verified_raw_manifest[
                "forecast_created_at_utc"
            ],
            "actual_attached_at_utc": verified_raw_manifest[
                "actual_attached_at_utc"
            ],
            "all_actuals_attached_after_corresponding_forecast": True,
            "shadow_panel_production_ready": verified_raw_manifest[
                "shadow_panel_production_ready"
            ],
            "shadow_panel_evidence": verified_raw_manifest[
                "shadow_panel_evidence"
            ],
            "temporal_attachment_audit": verified_raw_manifest[
                "temporal_attachment_audit"
            ],
            "comparison_scope": "paired_operational_final_pipelines",
            "baseline_output_stage": "residual_corrected",
            "candidate_output_stage": OUTPUT_MODEL,
            "baseline_residual_corrector_applied": True,
            "candidate_residual_corrector_applied": True,
            "residual_corrector_applied": True,
            "paired_same_delivery_hours": True,
            "paired_same_forecast_origins": True,
            "paired_same_observed_actuals": True,
            "actual_pairing_atol_eur_mwh": ACTUAL_PAIRING_ATOL_EUR_MWH,
            "actual_pairing_max_delta_eur_mwh": actual_delta,
            "residual_corrector_sha256": _sha256(corrector_copy),
            "oof_training_audit_sha256": _sha256(oof_copy),
            "source_experiment_manifest_sha256": _sha256(
                experiment_copy
            ),
            "observed_governance_evidence": {
                "relative_path": FINAL_SHADOW_EVIDENCE_NAME,
                "sha256": evidence_sha,
                "rows": int(len(final)),
            },
            "raw_shadow_evidence": {
                "relative_path": RAW_SHADOW_EVIDENCE_COPY_NAME,
                "sha256": _sha256(raw_copy),
                "rows": int(len(raw)),
            },
            "raw_shadow_manifest": {
                "relative_path": RAW_SHADOW_MANIFEST_COPY_NAME,
                "sha256": _sha256(raw_manifest_copy),
                "format_version": 3,
            },
            "raw_shadow_journal": {
                "relative_path": RAW_SHADOW_JOURNAL_COPY_NAME,
                "sha256": _sha256(raw_journal_copy),
                "records": int(journal_reference["records"]),
                "last_record_sha256": journal_reference["last_record_sha256"],
            },
            "paired_incumbent_evidence": {
                "relative_path": FINAL_SHADOW_INCUMBENT_NAME,
                "sha256": _sha256(incumbent_path),
                "rows": int(len(incumbent)),
                "output_stage": "residual_corrected",
            },
            "residual_corrector": {
                "relative_path": RESIDUAL_CORRECTOR_COPY_NAME,
                "sha256": _sha256(corrector_copy),
                "fit_protocol": corrector["fit_protocol"],
            },
            "oof_training_audit": {
                "relative_path": OOF_AUDIT_COPY_NAME,
                "sha256": _sha256(oof_copy),
            },
            "schema": {
                "relative_path": SCHEMA_COPY_NAME,
                "sha256": _sha256(schema_copy),
            },
            "source_experiment_manifest": {
                "relative_path": EXPERIMENT_MANIFEST_COPY_NAME,
                "sha256": _sha256(experiment_copy),
            },
            "rolling365_final_pipeline_evidence": rolling_lineage,
            "issued_shadow_history": {
                "relative_path": FINAL_SHADOW_ISSUED_NAME,
                "sha256": _sha256(issued_path),
                "rows": int(len(issued)),
                "first_delivery_utc": pd.Timestamp(
                    issued["delivery_start_utc"].iloc[0]
                ).isoformat(),
                "last_delivery_utc": pd.Timestamp(
                    issued["delivery_start_utc"].iloc[-1]
                ).isoformat(),
                "delivery_days": list(
                    dict.fromkeys(
                        pd.DatetimeIndex(issued["delivery_start_utc"])
                        .tz_convert(str(schema["timezone"]))
                        .date.astype(str)
                    )
                ),
                "candidate_output_stage": OUTPUT_MODEL,
                "actual_nullable": True,
            },
            "transformation": {
                "algorithm": "sealed_residual_corrector_linear_shift_v1",
                "raw_candidate_output_stage": BASE_MODEL,
                "output_stage": OUTPUT_MODEL,
                "mean_shift_eur_mwh": float(np.mean(shift)),
                "maximum_absolute_shift_eur_mwh": float(np.max(np.abs(shift))),
                "issued_mean_shift_eur_mwh": float(np.mean(issued_shift)),
                "issued_maximum_absolute_shift_eur_mwh": float(
                    np.max(np.abs(issued_shift))
                ),
            },
            "derivation_lineage": lineage,
            "derivation_sha256": derivation_sha,
        }
        manifest_path = staging / FINAL_SHADOW_MANIFEST_NAME
        _write_json(manifest_path, final_manifest)
        validate_final_shadow_manifest(
            manifest_path,
            experiment_manifest=experiment,
            zone=canonical_zone,
            expected_rows=len(final),
        )
        if destination.exists():
            backup = destination.parent / f".{destination.name}.backup-{uuid4().hex}"
            os.replace(destination, backup)
        os.replace(staging, destination)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
            backup = None
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise

    return FinalShadowArtifacts(
        False,
        destination,
        destination / FINAL_SHADOW_EVIDENCE_NAME,
        destination / FINAL_SHADOW_MANIFEST_NAME,
        len(final),
    )


__all__ = [
    "FINAL_SHADOW_DIRECTORY_NAME",
    "FINAL_SHADOW_EVIDENCE_NAME",
    "FINAL_SHADOW_ISSUED_NAME",
    "FINAL_SHADOW_MANIFEST_NAME",
    "ISSUED_COLUMNS",
    "RAW_SHADOW_JOURNAL_COPY_NAME",
    "FinalShadowArtifacts",
    "FinalShadowError",
    "finalize_shadow_evidence",
]
