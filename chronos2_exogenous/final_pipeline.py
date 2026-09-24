"""Seal the final operational LoRA pipeline on its untouched 365-day holdout.

The raw LoRA evaluator deliberately stops before the residual correction.  This
module is the complementary, fail-closed publication step: it applies one
already-fitted strictly pre-holdout OOF corrector, pairs the result with the
real operational ``residual_corrected`` incumbent and commits the final
evidence only after every temporal and cryptographic check has passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .evaluation import compute_metrics
from .governance import validate_experiment_manifest
from .lora_finetune import EVALUATION_COLUMNS, verify_bundle
from .production import (
    BASE_MODEL,
    OUTPUT_MODEL,
    RUNTIME_SCHEMA_VERSION,
    ExogenousProductionError,
    _apply_residual_corrector,
    _validate_corrector,
    _validate_final_pipeline_evidence,
)


FINAL_DIRECTORY_NAME = "final_pipeline"
FINAL_EVIDENCE_NAME = "final_pipeline_predictions.csv.gz"
FINAL_METRICS_NAME = "final_pipeline_metrics.json"
FINAL_REPORT_NAME = "final_pipeline_report.html"
FINAL_AUDIT_NAME = "final_pipeline_audit.json"
FINAL_MANIFEST_NAME = "final_pipeline_manifest.json"
EXPECTED_DAYS = 365
# Legacy reference only.  A rolling block of 365 *local* days is not always
# 8,760 physical hours: depending on where the block starts it may contain one
# or three DST transitions and therefore total 8,759 or 8,761 hours.
EXPECTED_HOURS = 8760
ACTUAL_PAIRING_ATOL_EUR_MWH = 5e-5
INCUMBENT_MODEL = "residual_corrected"
INCUMBENT_COLUMNS = (
    "delivery_start_utc",
    "forecast_origin_utc",
    "actual",
    "residual_corrected__q10",
    "residual_corrected__q50",
    "residual_corrected__q90",
)


class FinalPipelineEvaluationError(ExogenousProductionError):
    """Raised before publication when final-pipeline evidence is incomplete."""


@dataclass(frozen=True)
class FinalPipelineArtifacts:
    """Paths and headline metrics committed by the final evaluator."""

    output_directory: Path
    evidence_path: Path
    metrics_path: Path
    report_path: Path
    audit_path: Path
    manifest_path: Path
    experiment_manifest_path: Path
    metrics: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalPipelineEvaluationError(f"{label} illisible: {path}.") from exc
    if not isinstance(payload, dict):
        raise FinalPipelineEvaluationError(f"{label} doit etre un objet JSON.")
    return payload


def _json_text(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(_json_text(payload), encoding="utf-8")


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FinalPipelineEvaluationError(f"{label} absent: {path}.")
    suffixes = "".join(path.suffixes).lower()
    if not (suffixes.endswith(".csv") or suffixes.endswith(".csv.gz")):
        raise FinalPipelineEvaluationError(
            f"{label}: seul un CSV ou CSV gzip est accepte: {path.name}."
        )
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise FinalPipelineEvaluationError(f"{label} illisible: {path}.") from exc


def _safe_relative_path(run_directory: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise FinalPipelineEvaluationError(
            "La preuve brute ne declare pas de relative_path valide."
        )
    candidate = (run_directory / relative).resolve()
    if run_directory != candidate and run_directory not in candidate.parents:
        raise FinalPipelineEvaluationError(
            "La preuve brute reference un chemin hors du bundle LoRA."
        )
    return candidate


def _raw_evidence_reference(
    run_directory: Path, manifest: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    raw_reference = manifest.get("raw_evaluation_evidence")
    if raw_reference is None:
        raw_reference = manifest.get("evaluation_evidence")
    if not isinstance(raw_reference, Mapping):
        raise FinalPipelineEvaluationError(
            "Le manifeste ne contient aucune preuve rolling brute scellee."
        )
    reference = dict(raw_reference)
    path = _safe_relative_path(run_directory, reference.get("relative_path"))
    expected = (run_directory / "evaluation_predictions.csv.gz").resolve()
    if path != expected:
        raise FinalPipelineEvaluationError(
            "La preuve source doit etre evaluation_predictions.csv.gz, produite "
            "par l'evaluateur LoRA brut."
        )
    expected_sha = reference.get("sha256")
    if not isinstance(expected_sha, str) or expected_sha != _sha256(path):
        raise FinalPipelineEvaluationError(
            "Le SHA de evaluation_predictions.csv.gz diverge du manifeste."
        )
    return path, reference


def _validate_late_label_resolution(manifest: Mapping[str, Any]) -> None:
    """Require the sealed NaN->actual bridge before a final-pipeline claim."""

    binding = manifest.get("evaluation_label_binding")
    if not isinstance(binding, Mapping):
        return
    unresolved = binding.get("unresolved_cells", 0)
    if isinstance(unresolved, bool) or not isinstance(unresolved, int) or unresolved < 0:
        raise FinalPipelineEvaluationError(
            "Contrat evaluation_label_binding invalide."
        )
    if unresolved == 0:
        return
    resolution = manifest.get("evaluation_label_resolution")
    if not isinstance(resolution, Mapping):
        raise FinalPipelineEvaluationError(
            "FinalBacktest refuse: les labels holdout predeclares n'ont pas ete "
            "lies a un panel resolu."
        )
    expected: Mapping[str, object] = {
        "schema_version": 1,
        "frozen_panel_sha256": manifest.get("panel_sha256"),
        "frozen_panel_audit_sha256": manifest.get("panel_audit_sha256"),
        "input_contract_sha256": binding.get("input_contract_sha256"),
        "resolved_cells": unresolved,
        "all_other_values_identical": True,
        "labels_used_for_fit": False,
        "promotion_eligible": False,
    }
    failures = [
        key for key, value in expected.items() if resolution.get(key) != value
    ]
    if failures:
        raise FinalPipelineEvaluationError(
            "FinalBacktest refuse: resolution holdout divergente ("
            + ", ".join(failures)
            + ")."
        )
    for path_key, sha_key, label in (
        ("resolved_panel_path", "resolved_panel_sha256", "panel resolu"),
        (
            "resolved_panel_audit_path",
            "resolved_panel_audit_sha256",
            "sidecar du panel resolu",
        ),
    ):
        path_value = resolution.get(path_key)
        expected_sha = resolution.get(sha_key)
        path = (
            Path(path_value).expanduser().resolve()
            if isinstance(path_value, str) and path_value.strip()
            else None
        )
        if (
            path is None
            or not path.is_file()
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or _sha256(path) != expected_sha
        ):
            raise FinalPipelineEvaluationError(
                f"FinalBacktest refuse: {label} absent ou divergent."
            )


def _normalise_evidence(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if tuple(frame.columns) != EVALUATION_COLUMNS:
        raise FinalPipelineEvaluationError(
            f"{label}: schema exact requis: " + ", ".join(EVALUATION_COLUMNS) + "."
        )
    output = frame.copy()
    for column in ("delivery_start_utc", "forecast_origin_utc"):
        output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
        if output[column].isna().any():
            raise FinalPipelineEvaluationError(
                f"{label}: timestamps invalides dans {column}."
            )
    numeric = list(EVALUATION_COLUMNS[2:])
    output[numeric] = output[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(output[numeric].to_numpy(dtype=float)).all():
        raise FinalPipelineEvaluationError(f"{label}: valeurs non finies.")
    if output["delivery_start_utc"].duplicated().any():
        raise FinalPipelineEvaluationError(f"{label}: livraisons dupliquees.")
    output = output.sort_values("delivery_start_utc", kind="stable").reset_index(
        drop=True
    )
    for prefix in ("baseline", "candidate"):
        quantiles = output[
            [f"{prefix}_q10", f"{prefix}_q50", f"{prefix}_q90"]
        ].to_numpy(float)
        if bool(
            ((quantiles[:, 0] > quantiles[:, 1]) | (quantiles[:, 1] > quantiles[:, 2])).any()
        ):
            raise FinalPipelineEvaluationError(
                f"{label}: croisement de quantiles {prefix}."
            )
    return output


def _normalise_incumbent(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(INCUMBENT_COLUMNS).difference(frame.columns))
    if missing:
        raise FinalPipelineEvaluationError(
            "Statistics incumbent incompletes: " + ", ".join(missing) + "."
        )
    output = frame.loc[:, list(INCUMBENT_COLUMNS)].copy()
    output["delivery_start_utc"] = pd.to_datetime(
        output["delivery_start_utc"], utc=True, errors="coerce"
    )
    output["forecast_origin_utc"] = pd.to_datetime(
        output["forecast_origin_utc"], utc=True, errors="coerce"
    )
    if output[["delivery_start_utc", "forecast_origin_utc"]].isna().any().any():
        raise FinalPipelineEvaluationError(
            "Statistics incumbent: timestamps invalides."
        )
    numeric = ["actual", *(f"residual_corrected__{q}" for q in ("q10", "q50", "q90"))]
    output[numeric] = output[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(output[numeric].to_numpy(float)).all():
        raise FinalPipelineEvaluationError(
            "Statistics incumbent: prix ou quantiles non finis."
        )
    if output["delivery_start_utc"].duplicated().any():
        raise FinalPipelineEvaluationError(
            "Statistics incumbent: livraisons dupliquees."
        )
    output = output.sort_values("delivery_start_utc", kind="stable").reset_index(
        drop=True
    )
    quantiles = output[
        [f"residual_corrected__{q}" for q in ("q10", "q50", "q90")]
    ].to_numpy(float)
    if bool(
        ((quantiles[:, 0] > quantiles[:, 1]) | (quantiles[:, 1] > quantiles[:, 2])).any()
    ):
        raise FinalPipelineEvaluationError(
            "Statistics incumbent: quantiles residual_corrected croises."
        )
    return output


def _validate_holdout_timeline(
    frame: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if manifest.get("evaluation_days") != EXPECTED_DAYS:
        raise FinalPipelineEvaluationError(
            "Le bundle LoRA doit declarer exactement 365 jours d'evaluation."
        )
    delivery = pd.DatetimeIndex(frame["delivery_start_utc"])
    if len(delivery) > 1 and not np.all(
        np.diff(delivery.asi8) == 3_600_000_000_000
    ):
        raise FinalPipelineEvaluationError(
            "La timeline finale UTC n'est pas horaire et continue."
        )
    timezone_name = str(schema.get("timezone", ""))
    try:
        zone = ZoneInfo(timezone_name)
    except Exception as exc:
        raise FinalPipelineEvaluationError(
            f"Timezone du schema invalide: {timezone_name!r}."
        ) from exc
    cutoff = str(schema.get("cutoff_local_time", ""))
    try:
        cutoff_hour, cutoff_minute = (int(part) for part in cutoff.split(":"))
    except Exception as exc:
        raise FinalPipelineEvaluationError(
            f"cutoff_local_time invalide: {cutoff!r}."
        ) from exc

    local = delivery.tz_convert(zone)
    days = tuple(dict.fromkeys(local.date))
    if len(days) != EXPECTED_DAYS or any(
        day != days[0] + timedelta(days=index) for index, day in enumerate(days)
    ):
        raise FinalPipelineEvaluationError(
            "Le holdout final ne couvre pas 365 jours civils consecutifs."
        )
    audits: list[dict[str, Any]] = []
    origins = pd.DatetimeIndex(frame["forecast_origin_utc"])
    for day in days:
        mask = np.asarray(local.date == day)
        day_delivery = delivery[mask]
        day_origins = origins[mask].unique()
        if len(day_origins) != 1:
            raise FinalPipelineEvaluationError(
                f"Holdout {day}: une origine unique est requise."
            )
        start = pd.Timestamp(day).tz_localize(zone)
        end = pd.Timestamp(day + timedelta(days=1)).tz_localize(zone)
        expected = pd.date_range(
            start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left"
        )
        if not day_delivery.equals(expected):
            raise FinalPipelineEvaluationError(
                f"Holdout {day}: heures physiques/DST incoherentes."
            )
        origin = pd.Timestamp(day_origins[0]).tz_convert(zone)
        expected_origin_day = day - timedelta(days=1)
        if (
            origin.date() != expected_origin_day
            or origin.hour != cutoff_hour
            or origin.minute != cutoff_minute
            or origin.second != 0
        ):
            raise FinalPipelineEvaluationError(
                f"Holdout {day}: origine attendue a D-1 {cutoff}."
            )
        audits.append(
            {
                "delivery_day": day.isoformat(),
                "hours": len(expected),
                "forecast_origin_utc": origin.tz_convert("UTC").isoformat(),
            }
        )

    expected_physical_hours = sum(int(record["hours"]) for record in audits)
    if len(frame) != expected_physical_hours:
        raise FinalPipelineEvaluationError(
            "Le holdout final contient un nombre d'heures different de la "
            f"grille DST de ses 365 jours ({len(frame)} != "
            f"{expected_physical_hours})."
        )

    holdout = manifest.get("splits", {}).get("evaluation_holdout")
    if not isinstance(holdout, Mapping):
        raise FinalPipelineEvaluationError(
            "Le manifeste ne scelle pas la plage evaluation_holdout."
        )
    distinct_origins = origins.unique().sort_values()
    if (
        holdout.get("count") != EXPECTED_DAYS
        or len(distinct_origins) != EXPECTED_DAYS
        or distinct_origins[0].isoformat() != holdout.get("first_utc")
        or distinct_origins[-1].isoformat() != holdout.get("last_utc")
    ):
        raise FinalPipelineEvaluationError(
            "Les origines finales divergent du holdout gele du manifeste."
        )
    return audits


def _validate_pairing(raw: pd.DataFrame, incumbent: pd.DataFrame) -> float:
    raw_delivery = pd.DatetimeIndex(raw["delivery_start_utc"])
    incumbent_delivery = pd.DatetimeIndex(incumbent["delivery_start_utc"])
    if not raw_delivery.equals(incumbent_delivery):
        raise FinalPipelineEvaluationError(
            "L'incumbent ne couvre pas exactement les memes heures de livraison."
        )
    raw_origins = pd.DatetimeIndex(raw["forecast_origin_utc"])
    incumbent_origins = pd.DatetimeIndex(incumbent["forecast_origin_utc"])
    if not raw_origins.equals(incumbent_origins):
        raise FinalPipelineEvaluationError(
            "Les origines de l'incumbent divergent des origines LoRA."
        )
    raw_actual = raw["actual"].to_numpy(float)
    incumbent_actual = incumbent["actual"].to_numpy(float)
    delta = float(np.max(np.abs(raw_actual - incumbent_actual)))
    if not np.allclose(
        raw_actual,
        incumbent_actual,
        rtol=0.0,
        atol=ACTUAL_PAIRING_ATOL_EUR_MWH,
    ):
        raise FinalPipelineEvaluationError(
            "Les actuals de l'incumbent divergent des actuals LoRA "
            f"(ecart max={delta:.12g} EUR/MWh > "
            f"{ACTUAL_PAIRING_ATOL_EUR_MWH:.12g})."
        )
    return delta


def _validate_oof_chain(
    *,
    corrector_path: Path,
    oof_audit_path: Path,
    experiment: Mapping[str, Any],
    first_holdout_day: str,
) -> dict[str, Any]:
    if not corrector_path.is_file() or not oof_audit_path.is_file():
        raise FinalPipelineEvaluationError(
            "Le correcteur residuel et son sidecar OOF sont obligatoires."
        )
    corrector_sha = _sha256(corrector_path)
    audit_sha = _sha256(oof_audit_path)
    validation_manifest = dict(experiment)
    validation_manifest["candidate_output_stage"] = OUTPUT_MODEL
    validation_manifest["residual_corrector_sha256"] = corrector_sha
    corrector = _validate_corrector(
        corrector_path, experiment=validation_manifest
    )
    if corrector.get("oof_training_audit_sha256") != audit_sha:
        raise FinalPipelineEvaluationError(
            "Le sidecar OOF ne correspond pas au correcteur residuel."
        )
    audit = _json(oof_audit_path, label="sidecar audit OOF")
    sidecar_version = audit.get("schema_version")
    if type(sidecar_version) is not int or sidecar_version not in {1, 2}:
        raise FinalPipelineEvaluationError(
            "schema_version du sidecar OOF doit valoir 1 ou 2."
        )
    if corrector.get("oof_sidecar_schema_version") != sidecar_version:
        raise FinalPipelineEvaluationError(
            "Le correcteur et le sidecar OOF declarent des versions differentes."
        )
    expected: Mapping[str, object] = {
        "purpose": "chronos2_exogenous_blocked_prequential_oof",
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "candidate_model": BASE_MODEL,
        "training_days": 365,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "future_actuals_used_as_features": False,
        "holdout_used_for_fit": False,
        "selection_frozen_before_oof": True,
        "holdout_start_day": first_holdout_day,
    }
    failures = [
        f"{key}={audit.get(key)!r}, attendu={value!r}"
        for key, value in expected.items()
        if type(audit.get(key)) is not type(value) or audit.get(key) != value
    ]
    if failures:
        raise FinalPipelineEvaluationError(
            "Contrat du sidecar OOF invalide: " + "; ".join(failures)
        )
    if audit.get("predictions_sha256") != corrector.get(
        "oof_training_predictions_sha256"
    ):
        raise FinalPipelineEvaluationError(
            "Le sidecar OOF et le correcteur ne lient pas les memes predictions."
        )
    if audit.get("candidate_checkpoint_sha256") != experiment.get(
        "checkpoint_sha256"
    ):
        raise FinalPipelineEvaluationError(
            "Le sidecar OOF ne correspond pas au checkpoint LoRA evalue."
        )
    if corrector.get("holdout_start_day") != first_holdout_day:
        raise FinalPipelineEvaluationError(
            "Le correcteur OOF n'a pas ete gele avant ce holdout."
        )
    for field in ("training_start_day", "training_end_day"):
        if audit.get(field) != corrector.get(field):
            raise FinalPipelineEvaluationError(
                f"Le correcteur et le sidecar OOF divergent sur {field}."
            )
    if sidecar_version == 2:
        expected_v2: Mapping[str, object] = {
            "candidate_checkpoint_role": "deployment_identity_anchor_not_oof_predictor",
            "deployment_checkpoint_used_for_oof": False,
            "fold_checkpoints_are_origin_specific": True,
            "fold_lookback_days": 365,
        }
        v2_failures = [
            f"{key}={audit.get(key)!r}, attendu={value!r}"
            for key, value in expected_v2.items()
            if type(audit.get(key)) is not type(value) or audit.get(key) != value
        ]
        folds = audit.get("folds")
        if not isinstance(folds, list) or not folds:
            v2_failures.append("folds absents")
            folds = []
        if audit.get("fold_count") != len(folds):
            v2_failures.append("fold_count incoherent")
        for hash_field in (
            "fold_checkpoint_set_sha256",
            "fold_candidate_recipe_sha256",
        ):
            value = audit.get(hash_field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                or corrector.get(f"oof_{hash_field}") != value
            ):
                v2_failures.append(f"{hash_field} invalide/non lie")
        if v2_failures:
            raise FinalPipelineEvaluationError(
                "Contrat fold-specific OOF v2 invalide: " + "; ".join(v2_failures)
            )
    try:
        training_start = pd.Timestamp(str(corrector["training_start_day"])).date()
        training_end = pd.Timestamp(str(corrector["training_end_day"])).date()
        holdout_start = pd.Timestamp(first_holdout_day).date()
    except Exception as exc:
        raise FinalPipelineEvaluationError(
            "Dates d'entrainement du correcteur invalides."
        ) from exc
    if training_end >= holdout_start:
        raise FinalPipelineEvaluationError(
            "Le correcteur residuel chevauche le holdout final."
        )
    if (training_end - training_start).days != 364:
        raise FinalPipelineEvaluationError(
            "Le correcteur residuel ne couvre pas exactement 365 jours OOF."
        )
    return corrector


def _render_report(
    metrics: Mapping[str, Any], daily: pd.DataFrame, *, title: str
) -> str:
    gain = float(metrics["mae_gain_eur_mwh"])
    status = (
        "Le pipeline LoRA corrigé améliore l'incumbent"
        if gain > 0
        else "Le pipeline LoRA corrigé dégrade l'incumbent"
        if gain < 0
        else "Performance identique à l'incumbent"
    )
    status_class = "good" if gain > 0 else "bad" if gain < 0 else "muted"
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.delivery_day))}</td>"
        f"<td>{int(row.hours)}</td>"
        f"<td>{row.observed_mean_price_eur_mwh:.2f}</td>"
        f"<td>{row.baseline_mean_price_eur_mwh:.2f}</td>"
        f"<td>{row.candidate_mean_price_eur_mwh:.2f}</td>"
        f"<td>{row.baseline_hourly_mae_eur_mwh:.2f}</td>"
        f"<td>{row.candidate_hourly_mae_eur_mwh:.2f}</td>"
        "</tr>"
        for row in daily.itertuples(index=False)
    )
    title_safe = html.escape(title)

    def fmt(value: object, digits: int = 3) -> str:
        numeric = float(value)
        return "—" if not math.isfinite(numeric) else f"{numeric:.{digits}f}"

    return f"""<!doctype html>
<html lang="fr" data-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title_safe}</title><style>
:root{{--bg:#f3f6fb;--card:#fff;--text:#172033;--muted:#68758a;--line:#dce4ef;--base:#dc7b12;--candidate:#087fc1;--good:#117a4b;--bad:#c43d48}}
[data-theme=dark]{{--bg:#0d1421;--card:#172131;--text:#eaf0f8;--muted:#a9b6ca;--line:#304057;--base:#ffb347;--candidate:#62c8ff;--good:#61d79b;--bad:#ff7e87}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}}main{{max-width:1180px;margin:auto;padding:26px}}header{{display:flex;justify-content:space-between;gap:18px}}h1{{margin:0 0 6px;font-size:25px}}h2{{margin-top:26px}}button,.card,.notice,details{{background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:11px}}button{{padding:8px 12px;height:38px;cursor:pointer}}.muted{{color:var(--muted)}}.notice{{padding:13px 15px;border-left:4px solid var(--candidate);margin:18px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card{{padding:15px}}.value{{font-size:23px;font-weight:700;margin-top:5px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}details{{padding:12px 14px;margin-top:16px}}summary{{font-weight:650;cursor:pointer}}.table{{max-height:520px;overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:var(--card)}}@media(max-width:650px){{main{{padding:15px}}header{{display:block}}button{{margin-top:10px}}}}
</style></head><body><main><header><div><h1>{title_safe}</h1>
<div class="muted">Comparaison finale appariée sur 365 jours / 8 760 heures physiques.</div></div><button id="theme">🌙 Mode nuit</button></header>
<div class="notice"><strong>Périmètre :</strong> incumbent autonome avec correcteur résiduel contre Chronos-2 + LoRA exogène avec son correcteur résiduel OOF gelé. Même fenêtre, mêmes heures, mêmes origines et mêmes prix observés.</div>
<section class="cards"><div class="card"><div class="muted">MAE incumbent</div><div class="value">{fmt(metrics['baseline_mae_eur_mwh'])}</div><div>EUR/MWh</div></div>
<div class="card"><div class="muted">MAE LoRA corrigé</div><div class="value">{fmt(metrics['candidate_mae_eur_mwh'])}</div><div>EUR/MWh</div></div>
<div class="card"><div class="muted">Gain MAE</div><div class="value {status_class}">{fmt(gain)}</div><div>{fmt(100*float(metrics['mae_relative_gain']),2)} % — {status}</div></div>
<div class="card"><div class="muted">Prix moyen observé</div><div class="value">{fmt(metrics['actual_mean_price_eur_mwh'],2)}</div><div>EUR/MWh</div></div></section>
<h2>Prix moyens et probabilités</h2><section class="cards">
<div class="card"><div class="muted">Prix moyen incumbent</div><div class="value">{fmt(metrics['baseline_mean_price_eur_mwh'],2)}</div><div>EUR/MWh</div></div>
<div class="card"><div class="muted">Prix moyen LoRA corrigé</div><div class="value">{fmt(metrics['candidate_mean_price_eur_mwh'],2)}</div><div>EUR/MWh</div></div>
<div class="card"><div class="muted">Pinball incumbent</div><div class="value">{fmt(metrics['baseline_pinball_mean'])}</div></div>
<div class="card"><div class="muted">Pinball LoRA corrigé</div><div class="value">{fmt(metrics['candidate_pinball_mean'])}</div></div></section>
<details><summary>Détail des 365 journées</summary><div class="table"><table><thead><tr><th>Jour</th><th>Heures</th><th>Observé</th><th>Incumbent</th><th>LoRA corrigé</th><th>MAE incumbent</th><th>MAE LoRA</th></tr></thead><tbody>{rows}</tbody></table></div></details>
<p class="muted">Généré le {datetime.now(timezone.utc).isoformat()} · Les journées DST restent à 23/25 heures, sans interpolation.</p>
</main><script>const r=document.documentElement,b=document.getElementById('theme');function a(t){{r.dataset.theme=t;b.textContent=t==='dark'?'☀️ Mode jour':'🌙 Mode nuit'}}a(localStorage.getItem('chronos2-final-theme')||'light');b.onclick=()=>{{const t=r.dataset.theme==='dark'?'light':'dark';localStorage.setItem('chronos2-final-theme',t);a(t)}};</script></body></html>"""


def run_final_pipeline_evaluation(
    *,
    run_directory: str | Path,
    residual_corrector_path: str | Path,
    oof_audit_path: str | Path,
    incumbent_statistics_path: str | Path,
    zone: str,
    overwrite: bool = False,
    report_title: str = "Chronos-2 + LoRA — pipeline final rolling 365 jours",
) -> FinalPipelineArtifacts:
    """Validate, evaluate and atomically commit final LoRA pipeline evidence.

    ``production_pit_evidence`` is deliberately never assigned by this
    function.  The final-pipeline proof and the upstream PIT proof remain two
    independent production gates.
    """

    run_dir = Path(run_directory).expanduser().resolve()
    experiment_path = run_dir / "experiment_manifest.json"
    manifest = verify_bundle(run_dir)
    if not isinstance(manifest, dict):  # defensive: verify_bundle currently returns dict.
        manifest = dict(manifest)
    original_pit_evidence = manifest.get("production_pit_evidence")
    schema = _json(run_dir / "schema.json", label="schema LoRA")
    canonical_zone = str(zone).strip().upper()
    if not canonical_zone or not canonical_zone.isascii() or not canonical_zone.isalpha():
        raise FinalPipelineEvaluationError(f"Zone invalide: {zone!r}.")
    # Re-run the governance-level causal/freeze contract before deriving any
    # stronger final-pipeline claim from the raw evidence.
    validate_experiment_manifest(manifest, zone=canonical_zone)
    _validate_late_label_resolution(manifest)
    targets = schema.get("target_columns")
    if targets != ["target"]:
        raise FinalPipelineEvaluationError(
            "Le runtime production per_zone exige target_columns=['target']."
        )

    destination = run_dir / FINAL_DIRECTORY_NAME
    if destination.exists() and not overwrite:
        raise FinalPipelineEvaluationError(
            f"Evaluation finale deja publiee: {destination}; overwrite explicite requis."
        )
    raw_path, raw_reference = _raw_evidence_reference(run_dir, manifest)
    raw = _normalise_evidence(
        _read_csv(raw_path, label="preuve LoRA brute"), label="preuve LoRA brute"
    )
    physical_day_audit = _validate_holdout_timeline(
        raw, manifest=manifest, schema=schema
    )
    expected_physical_hours = int(
        sum(int(record["hours"]) for record in physical_day_audit)
    )
    incumbent_path = Path(incumbent_statistics_path).expanduser().resolve()
    incumbent_all = _normalise_incumbent(
        _read_csv(incumbent_path, label="Statistics incumbent")
    )
    raw_deliveries = pd.DatetimeIndex(raw["delivery_start_utc"])
    incumbent = incumbent_all.loc[
        incumbent_all["delivery_start_utc"].isin(raw_deliveries)
    ].reset_index(drop=True)
    if len(incumbent) != expected_physical_hours:
        raise FinalPipelineEvaluationError(
            "Statistics incumbent: la sous-fenetre appariee contient "
            f"{len(incumbent)} heures, attendu={expected_physical_hours}."
        )
    actual_max_delta = _validate_pairing(raw, incumbent)

    local_delivery = pd.DatetimeIndex(raw["delivery_start_utc"]).tz_convert(
        str(schema["timezone"])
    )
    first_holdout_day = local_delivery[0].date().isoformat()
    corrector_path = Path(residual_corrector_path).expanduser().resolve()
    audit_source = Path(oof_audit_path).expanduser().resolve()
    corrector = _validate_oof_chain(
        corrector_path=corrector_path,
        oof_audit_path=audit_source,
        experiment=manifest,
        first_holdout_day=first_holdout_day,
    )

    timestamp_column = str(schema["timestamp_column"])
    horizon = pd.DataFrame(
        {timestamp_column: pd.DatetimeIndex(raw["delivery_start_utc"])}
    )
    raw_candidate = np.stack(
        [raw[f"candidate_{quantile}"].to_numpy(float) for quantile in ("q10", "q50", "q90")],
        axis=0,
    )[np.newaxis, :, :]
    try:
        corrected, shift = _apply_residual_corrector(
            raw_candidate, horizon=horizon, corrector=corrector, schema=schema
        )
    except ExogenousProductionError as exc:
        raise FinalPipelineEvaluationError(
            "Application du correcteur final impossible; seules les features "
            "presentes dans la preuve sont utilisables."
        ) from exc
    if corrected.shape != raw_candidate.shape:
        raise FinalPipelineEvaluationError(
            "Le correcteur a produit une forme de prediction inattendue."
        )

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
    final = _normalise_evidence(final, label="pipeline final")
    _validate_holdout_timeline(final, manifest=manifest, schema=schema)
    metrics, daily = compute_metrics(final, timezone_name=str(schema["timezone"]))
    metrics.update(
        {
            "comparison_scope": "paired_operational_final_pipelines",
            "baseline_label": "Autonome + correcteur residuel (incumbent)",
            "candidate_label": "Chronos-2 + LoRA + correcteur residuel",
            "baseline_output_stage": INCUMBENT_MODEL,
            "candidate_output_stage": OUTPUT_MODEL,
            "physical_hours": expected_physical_hours,
            "residual_shift_mean_eur_mwh": float(np.mean(shift)),
            "residual_shift_mean_absolute_eur_mwh": float(np.mean(np.abs(shift))),
            "residual_shift_max_absolute_eur_mwh": float(np.max(np.abs(shift))),
        }
    )

    staging = run_dir / f".{FINAL_DIRECTORY_NAME}.tmp-{uuid4().hex}"
    backup: Path | None = None
    published = False
    manifest_temporary: Path | None = None
    previous_manifest_bytes = experiment_path.read_bytes()
    try:
        staging.mkdir(parents=False, exist_ok=False)
        evidence_path = staging / FINAL_EVIDENCE_NAME
        final.to_csv(evidence_path, index=False, compression="gzip")
        metrics_path = staging / FINAL_METRICS_NAME
        _write_json(metrics_path, metrics)
        report_path = staging / FINAL_REPORT_NAME
        report_path.write_text(
            _render_report(metrics, daily, title=report_title), encoding="utf-8"
        )
        evidence_sha = _sha256(evidence_path)
        corrector_sha = _sha256(corrector_path)
        oof_audit_sha = _sha256(audit_source)
        incumbent_sha = _sha256(incumbent_path)
        dst_days = [record for record in physical_day_audit if record["hours"] != 24]
        detail = {
            "comparison_scope": "paired_operational_final_pipelines",
            "baseline_output_stage": INCUMBENT_MODEL,
            "candidate_output_stage": OUTPUT_MODEL,
            "paired_same_input_contract": True,
            "paired_same_evaluation_window": True,
            "baseline_residual_corrector_applied": True,
            "candidate_residual_corrector_applied": True,
            "rolling_evaluation_days": EXPECTED_DAYS,
            "promotion_eligible": True,
            "paired_delivery_hours": expected_physical_hours,
            "paired_forecast_origins": True,
            "paired_actuals": True,
        }
        final_reference = {
            "relative_path": f"{FINAL_DIRECTORY_NAME}/{FINAL_EVIDENCE_NAME}",
            "sha256": evidence_sha,
            "rows": expected_physical_hours,
            "physical_days": EXPECTED_DAYS,
            "first_delivery_utc": final["delivery_start_utc"].iloc[0].isoformat(),
            "last_delivery_utc": final["delivery_start_utc"].iloc[-1].isoformat(),
            "dst_days": dst_days,
            "baseline_output_stage": INCUMBENT_MODEL,
            "candidate_output_stage": OUTPUT_MODEL,
        }
        audit = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "chronos2_exogenous_final_pipeline_rolling365",
            "zone": canonical_zone,
            "raw_candidate_evidence": {
                "relative_path": str(raw_reference["relative_path"]),
                "sha256": _sha256(raw_path),
            },
            "incumbent_statistics": {
                "path": str(incumbent_path),
                "sha256": incumbent_sha,
                "model": INCUMBENT_MODEL,
                "source_rows": int(len(incumbent_all)),
                "paired_rows": int(len(incumbent)),
            },
            "residual_corrector": {
                "path": str(corrector_path),
                "sha256": corrector_sha,
                "fit_protocol": corrector["fit_protocol"],
                "holdout_used_for_fit": False,
            },
            "oof_training_audit": {
                "path": str(audit_source),
                "sha256": oof_audit_sha,
            },
            "checks": {
                "bundle_verified": True,
                "exact_365_physical_days": True,
                "exact_physical_hours_for_365_local_days": True,
                "expected_physical_hours": expected_physical_hours,
                # Kept as an explicit legacy diagnostic, not as the governing
                # DST invariant.
                "exact_8760_physical_hours": expected_physical_hours == 8760,
                "continuous_hourly_utc": True,
                "dst_days_verified": True,
                "same_delivery_timeline": True,
                "same_forecast_origins": True,
                "same_actuals": True,
                "actual_pairing_atol_eur_mwh": ACTUAL_PAIRING_ATOL_EUR_MWH,
                "actual_pairing_max_delta_eur_mwh": actual_max_delta,
                "corrector_strictly_pre_holdout_oof": True,
                "production_pit_evidence_unchanged": True,
            },
            "production_pit_evidence_value": original_pit_evidence,
            "production_pipeline_evidence_detail": detail,
            "final_evidence_sha256": evidence_sha,
            "metrics": {
                "baseline_mae_eur_mwh": metrics["baseline_mae_eur_mwh"],
                "candidate_mae_eur_mwh": metrics["candidate_mae_eur_mwh"],
                "mae_gain_eur_mwh": metrics["mae_gain_eur_mwh"],
            },
        }
        audit_path = staging / FINAL_AUDIT_NAME
        _write_json(audit_path, audit)
        source_manifest_sha = hashlib.sha256(previous_manifest_bytes).hexdigest()
        final_manifest = {
            "schema_version": 1,
            "kind": "chronos2_exogenous_final_pipeline_evaluation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "zone": canonical_zone,
            "experiment_id": manifest.get("experiment_id"),
            "source_experiment_manifest_sha256": source_manifest_sha,
            "comparison_scope": "paired_operational_final_pipelines",
            "baseline_output_stage": INCUMBENT_MODEL,
            "candidate_output_stage": OUTPUT_MODEL,
            "window": {
                "physical_days": EXPECTED_DAYS,
                "physical_hours": expected_physical_hours,
                "first_delivery_utc": final_reference["first_delivery_utc"],
                "last_delivery_utc": final_reference["last_delivery_utc"],
                "dst_days": dst_days,
            },
            "artifacts": {
                "predictions": {
                    "relative_path": FINAL_EVIDENCE_NAME,
                    "sha256": evidence_sha,
                },
                "metrics": {
                    "relative_path": FINAL_METRICS_NAME,
                    "sha256": _sha256(metrics_path),
                },
                "report": {
                    "relative_path": FINAL_REPORT_NAME,
                    "sha256": _sha256(report_path),
                },
                "audit": {
                    "relative_path": FINAL_AUDIT_NAME,
                    "sha256": _sha256(audit_path),
                },
            },
            "production_pipeline_evidence": True,
            "production_pipeline_evidence_detail": detail,
        }
        final_manifest_path = staging / FINAL_MANIFEST_NAME
        _write_json(final_manifest_path, final_manifest)

        updated_manifest = dict(manifest)
        updated_manifest["zone"] = canonical_zone
        updated_manifest["raw_evaluation_evidence"] = dict(raw_reference)
        updated_manifest["evaluation_evidence"] = final_reference
        updated_manifest["candidate_output_stage"] = OUTPUT_MODEL
        updated_manifest["residual_corrector_sha256"] = corrector_sha
        updated_manifest["oof_training_audit_sha256"] = oof_audit_sha
        updated_manifest["production_pipeline_evidence"] = True
        updated_manifest["production_pipeline_evidence_detail"] = detail
        updated_manifest["production_runtime"] = {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "layout": "per_zone",
            "cross_learning": False,
            "target": "target",
        }
        updated_manifest["final_pipeline_evaluation"] = {
            "relative_path": f"{FINAL_DIRECTORY_NAME}/{FINAL_MANIFEST_NAME}",
            "sha256": _sha256(final_manifest_path),
            "audit_relative_path": f"{FINAL_DIRECTORY_NAME}/{FINAL_AUDIT_NAME}",
            "audit_sha256": _sha256(audit_path),
            "metrics_relative_path": f"{FINAL_DIRECTORY_NAME}/{FINAL_METRICS_NAME}",
            "metrics_sha256": _sha256(metrics_path),
            "incumbent_statistics_sha256": incumbent_sha,
        }
        if updated_manifest.get("production_pit_evidence") is not original_pit_evidence:
            raise FinalPipelineEvaluationError(
                "Invariant viole: production_pit_evidence ne doit jamais etre modifie."
            )
        _validate_final_pipeline_evidence(
            updated_manifest, rolling_path=evidence_path
        )

        manifest_temporary = run_dir / f".experiment_manifest.json.tmp-{uuid4().hex}"
        _write_json(manifest_temporary, updated_manifest)
        if destination.exists():
            backup = run_dir / f".{FINAL_DIRECTORY_NAME}.backup-{uuid4().hex}"
            os.replace(destination, backup)
        os.replace(staging, destination)
        published = True
        try:
            os.replace(manifest_temporary, experiment_path)
            manifest_temporary = None
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            published = False
            if backup is not None and backup.exists():
                os.replace(backup, destination)
                backup = None
            raise
        committed = verify_bundle(run_dir)
        if committed.get("production_pit_evidence") is not original_pit_evidence:
            raise FinalPipelineEvaluationError(
                "Invariant viole apres publication: production_pit_evidence a change."
            )
        _validate_final_pipeline_evidence(
            committed, rolling_path=destination / FINAL_EVIDENCE_NAME
        )
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
            backup = None
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if manifest_temporary is not None and manifest_temporary.exists():
            manifest_temporary.unlink(missing_ok=True)
        if published:
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            restore_temporary = run_dir / f".experiment_manifest.json.restore-{uuid4().hex}"
            restore_temporary.write_bytes(previous_manifest_bytes)
            os.replace(restore_temporary, experiment_path)
        if backup is not None and backup.exists():
            if published and destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            if not destination.exists():
                os.replace(backup, destination)
        raise

    return FinalPipelineArtifacts(
        output_directory=destination,
        evidence_path=destination / FINAL_EVIDENCE_NAME,
        metrics_path=destination / FINAL_METRICS_NAME,
        report_path=destination / FINAL_REPORT_NAME,
        audit_path=destination / FINAL_AUDIT_NAME,
        manifest_path=destination / FINAL_MANIFEST_NAME,
        experiment_manifest_path=experiment_path,
        metrics=metrics,
    )


__all__ = [
    "ACTUAL_PAIRING_ATOL_EUR_MWH",
    "EXPECTED_DAYS",
    "EXPECTED_HOURS",
    "FINAL_AUDIT_NAME",
    "FINAL_DIRECTORY_NAME",
    "FINAL_EVIDENCE_NAME",
    "FINAL_MANIFEST_NAME",
    "FINAL_METRICS_NAME",
    "FINAL_REPORT_NAME",
    "FinalPipelineArtifacts",
    "FinalPipelineEvaluationError",
    "run_final_pipeline_evaluation",
]
