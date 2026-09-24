"""Evaluate a genuine 365-day OOF corrector without publishing production evidence.

This read-only consumer of candidate/calibration artefacts deliberately accepts
diagnostic experiments.  It authenticates the normal blocked-OOF chain; it does
not invoke or relax the promotion, final-pipeline or shadow validators.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from .evaluation import compute_metrics
from .final_pipeline import (
    _json, _json_text, _normalise_evidence, _raw_evidence_reference, _sha256,
    _validate_holdout_timeline, _validate_late_label_resolution, _validate_oof_chain,
)
from .governance import REQUIRED_EXPERIMENT_FLAGS
from .lora_finetune import (
    EVALUATION_COLUMNS, load_config, read_panel, sha256_directory,
    validate_panel, verify_bundle,
)
from .oof_residual import (
    _assert_final_bundle_matches_recipe, _assert_plan_matches_final_split,
    _assert_regular_checkpoint_tree, _authenticate_completed_calibration_manifest,
    _fold_manifest_core, _fold_name, _json_sha256, _range, build_oof_plan,
)
from .production import _apply_residual_corrector, fit_oof_residual_corrector
from .research_validation_corrector import _resolve_without_links


MANIFEST_NAME = "research_oof_evaluation_manifest.json"
EVIDENCE_NAME = "research_oof_holdout_predictions.csv.gz"
METRICS_NAME = "research_oof_holdout_metrics.json"
DAILY_NAME = "research_oof_holdout_daily.csv.gz"
REPORT_NAME = "research_oof_report.html"
OOF_COLUMNS = (
    "delivery_start_utc", "forecast_origin_utc", "actual",
    "candidate_q10", "candidate_q50", "candidate_q90",
)


class ResearchOofEvaluationError(ValueError):
    """The research comparison cannot be authenticated."""


@dataclass(frozen=True)
class ResearchOofEvaluationArtifacts:
    directory: Path
    manifest_path: Path
    predictions_path: Path
    metrics_path: Path
    report_path: Path
    metrics: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResearchOofEvaluationError(message)


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def _normalise_oof(frame: pd.DataFrame) -> pd.DataFrame:
    _require(tuple(frame.columns) == OOF_COLUMNS, "Schema des predictions OOF invalide.")
    paired = frame.copy()
    for quantile in ("q10", "q50", "q90"):
        paired[f"baseline_{quantile}"] = paired[f"candidate_{quantile}"]
    checked = _normalise_evidence(paired.loc[:, EVALUATION_COLUMNS], label="OOF")
    return checked.loc[:, OOF_COLUMNS]


def _same_timeline(left: pd.DataFrame, right: pd.DataFrame, *, label: str) -> None:
    _require(len(left) == len(right), f"{label}: nombre d'heures divergent.")
    for column in ("delivery_start_utc", "forecast_origin_utc"):
        _require(
            pd.DatetimeIndex(left[column]).equals(pd.DatetimeIndex(right[column])),
            f"{label}: {column} divergent.",
        )


def _same_observations(left: pd.DataFrame, right: pd.DataFrame, *, label: str) -> None:
    _same_timeline(left, right, label=label)
    _require(
        np.allclose(left["actual"].to_numpy(float), right["actual"].to_numpy(float),
                    rtol=0., atol=1e-12),
        f"{label}: prix observes divergents.",
    )


def _same_target_contract(left: Mapping[str, Any], right: Mapping[str, Any], *,
                          zone: str, project_root: Path) -> None:
    contracts = [value.get("target_contracts", {}).get(zone, {}) for value in (left, right)]
    for key in ("series", "naive_timezone"):
        _require(isinstance(contracts[0].get(key), str) and bool(contracts[0][key])
                 and contracts[0][key] == contracts[1].get(key),
                 f"Reference: identite cible canonique {key} divergente ou absente.")
    paths = []
    for contract in contracts:
        _require(isinstance(contract.get("cache_path"), str) and bool(contract["cache_path"]),
                 "Reference: cache canonique absent du contrat.")
        path = Path(contract["cache_path"])
        paths.append((path if path.is_absolute() else project_root / path).resolve())
    _require(paths[0] == paths[1], "Reference: cache canonique divergent.")


def _panel_observations(panel: pd.DataFrame, config: Any, zone: str) -> pd.DataFrame:
    selected = panel.loc[panel[config.item_column].astype(str).eq(zone)]
    delivery_days = pd.DatetimeIndex(selected[config.timestamp_column]).tz_convert(
        config.timezone
    ).normalize().tz_localize(None)
    origin_days = pd.DatetimeIndex(selected[config.origin_column]).tz_convert(
        config.timezone
    ).normalize().tz_localize(None)
    horizon = delivery_days == origin_days + pd.Timedelta(days=1)
    return selected.loc[
        horizon, [config.timestamp_column, config.origin_column, "target"]
    ].rename(columns={
        config.timestamp_column: "delivery_start_utc",
        config.origin_column: "forecast_origin_utc", "target": "actual",
    }).sort_values("delivery_start_utc", kind="stable").reset_index(drop=True)


def _load_raw(run: Path, zone: str) -> tuple[dict[str, Any], dict[str, Any], Path, pd.DataFrame]:
    manifest = dict(verify_bundle(run))
    _require(manifest.get("evaluation_role") in {"primary_predeclared", "diagnostic_only"},
             "Role d'evaluation inconnu.")
    _require(all(type(manifest.get(k)) is type(v) and manifest.get(k) == v
                 for k, v in REQUIRED_EXPERIMENT_FLAGS.items()),
             "Contrat causal/freeze du candidat incomplet.")
    _require(manifest.get("zone", zone) == zone, "Zone du bundle divergente.")
    _require(type(manifest.get("production_pit_evidence")) is bool,
             "Preuve PIT booleenne explicite obligatoire.")
    schema = _json(run / "schema.json", label="schema")
    _require(schema.get("target_columns") == ["target"], "Cible canonique target obligatoire.")
    _validate_late_label_resolution(manifest)
    raw_path, _ = _raw_evidence_reference(run, manifest)
    raw = _normalise_evidence(_csv(raw_path), label="holdout brut")
    _validate_holdout_timeline(raw, manifest=manifest, schema=schema)
    return manifest, schema, raw_path, raw


def _render(metrics: Mapping[str, Any], daily: pd.DataFrame, *, title: str) -> str:
    rows = "".join(
        "<tr><td>" + html.escape(name) + "</td><td>" + f"{value['mae_eur_mwh']:.3f}"
        + "</td><td>" + f"{value['rmse_eur_mwh']:.3f}" + "</td><td>"
        + f"{value['mean_price_eur_mwh']:.2f}" + "</td><td>"
        + f"{value['daily_mean_mae_eur_mwh']:.3f}" + "</td><td>"
        + f"{value['pinball_mean']:.3f}" + "</td><td>"
        + f"{value['coverage_q10_q90']:.1%}" + "</td></tr>"
        for name, value in metrics["models"].items()
    )
    day_rows = "".join(
        f"<tr><td>{html.escape(str(row.delivery_day))}</td><td>{row.hours}</td>"
        f"<td>{row.observed_mean_price_eur_mwh:.2f}</td>"
        f"<td>{row.baseline_mean_price_eur_mwh:.2f}</td>"
        f"<td>{row.candidate_mean_price_eur_mwh:.2f}</td>"
        f"<td>{row.baseline_daily_mean_abs_error_eur_mwh:.3f}</td>"
        f"<td>{row.candidate_daily_mean_abs_error_eur_mwh:.3f}</td>"
        f"<td>{row.baseline_hourly_mae_eur_mwh:.3f}</td>"
        f"<td>{row.candidate_hourly_mae_eur_mwh:.3f}</td></tr>"
        for row in daily.itertuples(index=False)
    )
    reference_notes = "".join(
        "<li>" + html.escape(reference["label"]) + ": "
        + ("recalcul explicite sur la cible canonique commune" if reference["actual_policy"] == "canonical_recompute"
           else "observations archivées identiques à la précision numérique près")
        + f" ; {reference['revised_actual_hours']} heure(s) révisée(s), écart maximal "
        + f"{reference['maximum_absolute_actual_revision_eur_mwh']:.6f} EUR/MWh. "
        + f"MAE horaire avec observations archivées : {reference['original_observation_metrics']['mae_eur_mwh']:.6f} ; "
        + f"avec cible commune : {reference['common_observation_metrics']['mae_eur_mwh']:.6f}. "
        + "Les prévisions restent inchangées ; les observations originales sont conservées dans le CSV.</li>"
        for reference in metrics["references"]
    )
    return f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>:root{{color-scheme:light;--bg:#f5f7fa;--fg:#172435;--panel:white;--line:#ccd5df}}
html[data-theme=dark]{{color-scheme:dark;--bg:#101924;--fg:#e5edf5;--panel:#1a2837;--line:#42566d}}
body{{font:16px system-ui;background:var(--bg);color:var(--fg);max-width:1100px;margin:32px auto;padding:0 20px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);font-variant-numeric:tabular-nums}}
th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}
button{{float:right;padding:8px}}.notice{{padding:16px;border:1px solid var(--line);border-radius:8px}}
details{{margin-top:24px}}h1{{font-size:27px}}.scroll{{overflow:auto;max-height:520px}}</style>
<button id="theme">Mode nuit / jour</button><h1>{html.escape(title)}</h1>
<p>{metrics['first_delivery_day']} → {metrics['last_delivery_day']} · {metrics['physical_days']} jours · {metrics['physical_hours']} heures</p>
<p class="notice">Évaluation de recherche uniquement. Correcteur entraîné sur 365 jours réellement hors échantillon,
avec checkpoints distincts par bloc et 365 jours antérieurs de fit par bloc. Aucun prix du holdout n'est utilisé
pour le calibrage. Ce rapport ne qualifie pas les attestations historiques pour la production : aucune promotion,
activation ou validation shadow n'est produite. Les références éventuelles peuvent utiliser des entrées différentes ;
elles sont appariées sur les mêmes heures et origines, puis évaluées sur les mêmes prix observés canoniques.
Un éventuel recalcul des scores historiques est signalé ci-dessous ; les anciens rapports ne sont pas modifiés.</p>
<div class="scroll"><table><thead><tr><th>Modèle</th><th>MAE horaire</th><th>RMSE horaire</th><th>Prix moyen</th><th>MAE prix moyen journalier</th><th>Pinball</th><th>Couverture q10–q90</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>Prix observé moyen : {metrics['observed_mean_price_eur_mwh']:.2f} EUR/MWh.
Gain du correcteur sur le modèle brut : {metrics['corrector_gain_eur_mwh']:+.3f} EUR/MWh de MAE.</p>
<p>La MAE du prix moyen journalier donne le même poids à chacun des 365 jours ; le prix moyen de période
et la MAE horaire pondèrent chaque heure physique de la même façon (y compris les journées de 23/25 heures).</p>
<ul>{reference_notes}</ul>
<details><summary>Statistics — prix moyens et erreurs journalières (EUR/MWh)</summary><div class="scroll"><table><thead><tr><th>Jour</th><th>Heures</th><th>Observé moyen</th><th>Brut moyen</th><th>Corrigé moyen</th><th>Erreur absolue moyenne brut</th><th>Erreur absolue moyenne corrigé</th><th>MAE horaire brut</th><th>MAE horaire corrigé OOF</th></tr></thead><tbody>{day_rows}</tbody></table></div></details>
<script>const key='chronos-research-theme';try{{document.documentElement.dataset.theme=localStorage.getItem(key)||'light'}}catch(e){{}}
document.getElementById('theme').onclick=()=>{{const t=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=t;try{{localStorage.setItem(key,t)}}catch(e){{}}}};</script></html>"""


def evaluate_research_oof(
    *, config_path: str | Path, run_directory: str | Path,
    calibration_directory: str | Path, calibration_panel_path: str | Path,
    calibration_panel_audit_path: str | Path, output_directory: str | Path,
    zone: str, reference_runs: Mapping[str, str | Path] | None = None,
    reference_actual_policy: str = "strict",
    report_title: str = "LoRA météo — correcteur OOF, recherche rolling 365 jours",
) -> ResearchOofEvaluationArtifacts:
    """Authenticate and evaluate, without mutating any input or eligibility flag."""
    config = load_config(config_path)
    _require(reference_actual_policy in {"strict", "canonical_recompute"},
             "Politique des observations de reference inconnue.")
    zone = str(zone).strip().upper()
    run = _resolve_without_links(Path(run_directory), label="bundle candidat")
    calibration = _resolve_without_links(Path(calibration_directory), label="calibration OOF")
    output = _resolve_without_links(Path(output_directory), label="sortie recherche")
    _require(output != config.project_root and config.project_root in output.parents,
             "La sortie recherche doit rester sous project_root.")
    _require(not output.exists(), "Sortie recherche existante; choisissez un nouveau repertoire.")
    _require(not (output == run or run in output.parents or output in run.parents
                  or output == calibration or calibration in output.parents
                  or output in calibration.parents), "La sortie recherche doit etre separee des sources.")
    source_hashes: dict[Path, str] = {}
    checkpoint_hashes: dict[Path, str] = {}

    def bind(path: Path) -> None:
        checked = _resolve_without_links(path, label="source recherche")
        source_hashes[checked] = _sha256(checked)

    experiment, schema, raw_path, raw = _load_raw(run, zone)
    _require(experiment.get("evaluation_role") == config.evaluation_role,
             "Role config/bundle divergent.")
    for path in (config.config_path, config.panel_path, config.panel_audit_path,
                 run / "experiment_manifest.json", run / "schema.json", raw_path):
        bind(path)
    model_source, base_sha, deployment_sha, recipe = _assert_final_bundle_matches_recipe(
        config, run, experiment
    )
    checkpoint_hashes[model_source] = base_sha
    checkpoint_hashes[run / "checkpoint"] = deployment_sha
    calibration_config = replace(
        config, panel_path=Path(calibration_panel_path).expanduser().resolve(),
        panel_audit_path=Path(calibration_panel_audit_path).expanduser().resolve(),
        training_window_days=730, evaluation_days=365,
    )
    bind(calibration_config.panel_path)
    bind(calibration_config.panel_audit_path)
    panel, _, panel_audit = validate_panel(read_panel(calibration_config.panel_path), calibration_config)
    complete_path = calibration / "calibration_manifest.json"
    bind(complete_path)
    complete = _json(complete_path, label="calibration complete")
    _authenticate_completed_calibration_manifest(calibration, complete)
    block_days = complete.get("block_days")
    _require(type(block_days) is int and block_days > 0, "Taille des blocs OOF invalide.")
    plan = build_oof_plan(panel[config.origin_column].drop_duplicates(),
                         timezone_name=config.timezone, cutoff_local_time=config.cutoff_local_time,
                         validation_days=config.validation_days, block_days=block_days)
    _assert_plan_matches_final_split(plan, manifest=experiment, validation_days=config.validation_days)
    upstream = panel_audit["upstream_panel_audit"]
    expected_complete = {
        "item_id": zone, "target_column": "target", "required_origins": 1095,
        "training_days": 365, "oof_days": 365, "holdout_days": 365,
        "fold_count": len(plan.folds), "recipe_sha256": _json_sha256(recipe),
        "panel_sha256": upstream["panel_sha256"],
        "panel_audit_sha256": upstream["panel_audit_sha256"],
        "deployment_checkpoint_sha256": deployment_sha,
        "calibration_origins": _range(plan.calibration_origins),
        "holdout_origins": _range(plan.holdout_origins),
        "production_pit_evidence": panel_audit["production_pit_evidence"],
        "holdout_targets_used_for_fit": False, "promotion_eligible": False,
    }
    _require(all(type(complete.get(k)) is type(v) and complete.get(k) == v
                 for k, v in expected_complete.items()),
             "Contrat de calibration divergent du panel/plan/candidat.")
    _require(upstream["target_contracts"] == experiment["target_contracts"],
             "Contrats des cibles canoniques divergents.")
    _require(panel_audit["evaluation_label_binding"] == experiment["evaluation_label_binding"],
             "Le panel de calibration ne contient pas le meme holdout gele.")
    oof_path = calibration / "oof_predictions_365.csv.gz"
    audit_path = calibration / "oof_predictions_365.csv.gz.audit.json"
    corrector_path = calibration / "residual_corrector.json"
    for path in (oof_path, audit_path, corrector_path):
        bind(path)
    audit = _json(audit_path, label="audit OOF")
    _require(audit.get("schema_version") == 2, "Checkpoints de folds OOF v2 obligatoires.")
    _require(audit.get("panel_sha256") == upstream["panel_sha256"]
             and audit.get("panel_audit_sha256") == upstream["panel_audit_sha256"],
             "Le sidecar OOF ne lie pas le panel fourni.")
    _require(audit.get("production_pit_evidence") is panel_audit["production_pit_evidence"],
             "Preuve PIT OOF divergente du panel.")
    _require(len(audit.get("folds", [])) == len(plan.folds), "Nombre de folds OOF divergent.")
    parts = []
    for fold, declared in zip(plan.folds, audit["folds"], strict=True):
        directory = calibration / "folds" / _fold_name(fold, config.timezone)
        fold_manifest_path = directory / "fold_manifest.json"
        prediction_path = directory / "oof_predictions.csv.gz"
        bind(fold_manifest_path)
        bind(prediction_path)
        checkpoint = directory / "checkpoint"
        _assert_regular_checkpoint_tree(checkpoint)
        checkpoint_hashes[checkpoint] = sha256_directory(checkpoint)
        fold_manifest = _json(fold_manifest_path, label="fold OOF")
        expected = _fold_manifest_core(fold, recipe=recipe, recipe_sha256=_json_sha256(recipe),
                                       checkpoint_sha256=checkpoint_hashes[checkpoint])
        expected["predictions_sha256"] = source_hashes[prediction_path.resolve()]
        _require(all(fold_manifest.get(k) == v for k, v in expected.items()),
                 f"Fold {fold.index}: checkpoint/recette/plages/predictions divergents.")
        for key in ("fold_index", "fit_origins", "train_origins", "validation_origins",
                    "prediction_origins", "checkpoint_sha256", "predictions_sha256"):
            _require(declared.get(key) == expected[key], f"Fold {fold.index}: sidecar {key} divergent.")
        part = _normalise_oof(_csv(prediction_path))
        _require(pd.DatetimeIndex(part["forecast_origin_utc"].drop_duplicates()).equals(
            pd.DatetimeIndex(fold.prediction_origins)), f"Fold {fold.index}: origines incorrectes.")
        parts.append(part)
    oof = _normalise_oof(_csv(oof_path))
    joined = pd.concat(parts, ignore_index=True)
    _same_observations(oof, joined, label="OOF consolide/folds")
    # A resumed fold is read by the OOF writer with pandas' ordinary float
    # parser. Match its existing prediction tolerance. Observations only allow
    # absolute 1e-12 roundtrip noise; timestamps and origins remain exact.
    _require(np.allclose(oof.loc[:, OOF_COLUMNS[3:]].to_numpy(float),
                         joined.loc[:, OOF_COLUMNS[3:]].to_numpy(float),
                         rtol=1e-12, atol=1e-12),
             "Les predictions OOF consolidees divergent des folds.")
    targets = _panel_observations(panel, config, zone)
    _same_observations(oof, targets.loc[targets.forecast_origin_utc.isin(plan.calibration_origins)],
                       label="OOF/panel canonique")
    _same_observations(raw, targets.loc[targets.forecast_origin_utc.isin(plan.holdout_origins)],
                       label="Holdout/panel canonique")
    first_day = pd.Timestamp(raw.delivery_start_utc.iloc[0]).tz_convert(config.timezone).date().isoformat()
    corrector = _validate_oof_chain(corrector_path=corrector_path, oof_audit_path=audit_path,
                                    experiment=experiment, first_holdout_day=first_day)
    horizon = pd.DataFrame({schema["timestamp_column"]: raw.delivery_start_utc})
    values = np.stack([raw[f"candidate_{q}"].to_numpy(float) for q in ("q10", "q50", "q90")])[None]
    corrected, shift = _apply_residual_corrector(values, horizon=horizon, corrector=corrector, schema=schema)
    evidence = raw.loc[:, ["delivery_start_utc", "forecast_origin_utc", "actual"]].copy()
    for index, q in enumerate(("q10", "q50", "q90")):
        evidence[f"raw_{q}"] = raw[f"candidate_{q}"]
        evidence[f"corrected_oof_{q}"] = corrected[0, index]
    paired = raw.copy()
    for q in ("q10", "q50", "q90"):
        paired[f"baseline_{q}"] = evidence[f"raw_{q}"]
        paired[f"candidate_{q}"] = evidence[f"corrected_oof_{q}"]
    summary, daily = compute_metrics(paired, timezone_name=config.timezone)

    def model_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
        actual = frame["actual"].to_numpy(float)
        median = frame[f"{prefix}_q50"].to_numpy(float)
        days = pd.DatetimeIndex(frame["delivery_start_utc"]).tz_convert(config.timezone).date
        daily_means = pd.DataFrame({"day": days, "actual": actual, "prediction": median}).groupby(
            "day", sort=True
        )[["actual", "prediction"]].mean()
        losses = []
        for q, level in (("q10", .1), ("q50", .5), ("q90", .9)):
            error = actual - frame[f"{prefix}_{q}"].to_numpy(float)
            losses.append(float(np.mean(np.maximum(level * error, (level - 1) * error))))
        return {"mae_eur_mwh": float(np.mean(abs(actual - median))),
                "rmse_eur_mwh": float(np.sqrt(np.mean((actual - median) ** 2))),
                "mean_price_eur_mwh": float(np.mean(median)), "pinball_mean": float(np.mean(losses)),
                "daily_mean_mae_eur_mwh": float(abs(daily_means.actual - daily_means.prediction).mean()),
                "coverage_q10_q90": float(np.mean((actual >= frame[f"{prefix}_q10"])
                                                   & (actual <= frame[f"{prefix}_q90"]))) }

    models = {"LoRA brut": model_metrics(evidence, "raw"),
              "LoRA + correcteur OOF": model_metrics(evidence, "corrected_oof")}
    references = []
    for number, (label, reference_path) in enumerate((reference_runs or {}).items(), start=1):
        _require(bool(label.strip()) and label not in models, "Nom de reference vide ou duplique.")
        reference = _resolve_without_links(Path(reference_path), label="reference brute")
        _require(not (output == reference or reference in output.parents or output in reference.parents),
                 "La sortie recherche doit etre separee des references.")
        ref_manifest, _, ref_csv, ref_raw = _load_raw(reference, zone)
        if reference_actual_policy == "canonical_recompute":
            _same_timeline(raw, ref_raw, label=f"Reference {label}")
            _same_target_contract(experiment, ref_manifest, zone=zone, project_root=config.project_root)
        else:
            _same_observations(raw, ref_raw, label=f"Reference {label}")
        for path in (reference / "experiment_manifest.json", reference / "schema.json", ref_csv):
            bind(path)
        checkpoint_hashes[reference / "checkpoint"] = ref_manifest["checkpoint_sha256"]
        prefix = f"reference_{number}"
        evidence[f"{prefix}_original_actual"] = ref_raw["actual"].to_numpy(float)
        for q in ("q10", "q50", "q90"):
            evidence[f"{prefix}_{q}"] = ref_raw[f"candidate_{q}"]
        models[label] = model_metrics(evidence, prefix)
        revisions = abs(raw["actual"].to_numpy(float) - ref_raw["actual"].to_numpy(float))
        references.append({"label": label, "column_prefix": prefix, "run_directory": str(reference),
                           "checkpoint_sha256": ref_manifest["checkpoint_sha256"],
                           "comparison_scope": "different_inputs_same_delivery_origin_common_actual",
                           "actual_policy": reference_actual_policy,
                           "revised_actual_hours": int(np.count_nonzero(revisions > 1e-12)),
                           "maximum_absolute_actual_revision_eur_mwh": float(np.max(revisions)),
                           "original_observation_metrics": model_metrics(ref_raw, "candidate"),
                           "common_observation_metrics": models[label],
                           "input_contracts_declared_identical": False})
    metrics = {"schema_version": 1, "comparison_scope": "research_raw_vs_genuine_oof_corrected",
               "physical_days": summary["physical_days"], "physical_hours": summary["physical_hours"],
               "first_delivery_day": first_day,
               "last_delivery_day": pd.Timestamp(raw.delivery_start_utc.iloc[-1]).tz_convert(config.timezone).date().isoformat(),
               "observed_mean_price_eur_mwh": summary["actual_mean_price_eur_mwh"],
               "corrector_gain_eur_mwh": summary["mae_gain_eur_mwh"],
               "maximum_absolute_shift_eur_mwh": float(np.max(abs(shift))), "models": models,
               "references": references}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    staging.mkdir()
    try:
        # Reconstruct the tiny ridge fit to validate coefficients and all OOF
        # chronology using the existing authoritative fitter; no neural fit.
        check_path = staging / "verified_corrector.json"
        fit_oof_residual_corrector(oof_predictions_path=oof_path, oof_audit_path=audit_path,
                                  holdout_start_day=first_day, output_path=check_path,
                                  timezone_name=config.timezone,
                                  feature_columns=corrector["feature_columns"],
                                  ridge_alpha=corrector["ridge_alpha"],
                                  maximum_absolute_shift_eur_mwh=corrector["maximum_absolute_shift_eur_mwh"])
        _require(_json(check_path, label="correcteur recalcule") == corrector,
                 "Les parametres du correcteur ne correspondent pas au fit OOF scelle.")
        check_path.unlink()
        evidence.to_csv(staging / EVIDENCE_NAME, index=False, compression="gzip")
        daily.to_csv(staging / DAILY_NAME, index=False, compression="gzip")
        (staging / METRICS_NAME).write_text(_json_text(metrics), encoding="utf-8")
        (staging / REPORT_NAME).write_text(_render(metrics, daily, title=report_title), encoding="utf-8")
        for path, digest in source_hashes.items():
            _require(_sha256(path) == digest, f"Source modifiee pendant l'evaluation: {path}.")
        for path, digest in checkpoint_hashes.items():
            _require(sha256_directory(path) == digest, f"Checkpoint modifie pendant l'evaluation: {path}.")
        manifest = {
            "schema_version": 1, "kind": "chronos2_exogenous_research_oof_evaluation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "zone": zone,
            "experiment_id": experiment["experiment_id"], "evaluation_role": experiment["evaluation_role"],
            "research_only": True, "production_pit_evidence": False,
            "production_pipeline_evidence": False, "promotion_eligible": False,
            "candidate_selection_eligible": False, "shadow_eligible": False, "activation_performed": False,
            "genuine_oof_days": 365, "holdout_days": 365, "fold_count": len(plan.folds),
            "observation_pairing": "exact_timestamps_origins_roundtrip_equivalent_actuals",
            "observation_csv_roundtrip_tolerance": {"rtol": 0., "atol": 1e-12},
            "reference_actual_policy": reference_actual_policy,
            "fold_prediction_csv_roundtrip_tolerance": {"rtol": 1e-12, "atol": 1e-12},
            "holdout_used_for_fit": False, "source_artifacts_modified": False,
            "source_production_pit_evidence": experiment["production_pit_evidence"],
            "oof_production_pit_evidence": audit["production_pit_evidence"],
            "candidate_checkpoint_sha256": deployment_sha, "references": references,
            "sources": [{"path": str(p), "sha256": h} for p, h in source_hashes.items()],
            "checkpoints": [{"path": str(p), "sha256": h} for p, h in checkpoint_hashes.items()],
            "outputs": {name: {"relative_path": name, "sha256": _sha256(staging / name)}
                        for name in (EVIDENCE_NAME, METRICS_NAME, DAILY_NAME, REPORT_NAME)},
        }
        (staging / MANIFEST_NAME).write_text(_json_text(manifest), encoding="utf-8")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ResearchOofEvaluationArtifacts(output, output / MANIFEST_NAME, output / EVIDENCE_NAME,
                                         output / METRICS_NAME, output / REPORT_NAME, metrics)


__all__ = ["ResearchOofEvaluationError", "ResearchOofEvaluationArtifacts", "evaluate_research_oof"]
