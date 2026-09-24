"""Production-format HTML for isolated zonal experiments, from frozen inputs only."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
from chronos2_hourly.reporting import (
    STORM_DASHBOARD_CONTRACT_ID, _backtest_prediction_frame, _forecast_prediction_frame,
    _replace_report_labels, storm_benchmark_contracts,
)
from chronos2_modular.common import ZoneData, ZoneRunResult
from chronos2_modular.metrics import compute_metrics, metric_breakdowns
from chronos2_modular.report import write_html_report
from nyx_scarcity.reporting import TIMEZONES, _clean, _prepare


class ZonalReportError(ValueError):
    """A report cannot truthfully reproduce its frozen comparison contract."""


def _read_verified(path: str | Path, expected: str) -> bytes:
    target = Path(path).expanduser().resolve()
    raw = target.read_bytes()
    if not expected or hashlib.sha256(raw).hexdigest() != expected:
        raise ZonalReportError(f"Frozen source checksum mismatch: {target}")
    if target.read_bytes() != raw:
        raise ZonalReportError(f"Source changed during verification: {target}")
    return raw


def _verified_storm(frame: pd.DataFrame, source_audit: dict, zone: str) -> tuple[dict, dict]:
    """Verify original archive, then preserve the narrower frozen-panel mask."""
    data = source_audit.get("source_data_audit", {})
    sources = data.get("baseline", {}).get("sources", [])
    matches = [item for item in sources if item.get("zone") == zone and item.get("model") == "nuclear_kalman"]
    if len(matches) != 1:
        raise ZonalReportError(f"{zone}: exactly one frozen nuclear_kalman source audit is required.")
    source = matches[0]
    proofs = source.get("time_axis_audits", [])
    if len(proofs) != 1:
        raise ZonalReportError(f"{zone}: original report audit identity is missing or ambiguous.")
    report_audit = json.loads(_read_verified(proofs[0]["path"], proofs[0]["sha256"]))
    if report_audit.get("zone") != zone:
        raise ZonalReportError("Original report audit belongs to another zone.")
    snapshot = report_audit.get("storm_hourly_comparison", {}).get("snapshot", {})
    for key in ("artifact", "audit"):
        _read_verified(snapshot[f"{key}_path"], snapshot[f"{key}_sha256"])
    loaded = _load_verified_snapshot(Path(snapshot["archive"]), zone=zone, timezone=TIMEZONES[zone])
    if loaded is None:
        raise ZonalReportError(f"{zone}: original official Storm archive unavailable.")
    values, materialization, provenance = loaded
    if (provenance["artifact_sha256"] != snapshot["artifact_sha256"]
            or provenance["audit_sha256"] != snapshot["audit_sha256"]):
        raise ZonalReportError("Verified archive differs from the original report snapshot.")
    timestamps = pd.DatetimeIndex(frame.timestamp_utc)
    reference = values.reindex(timestamps).to_numpy(float)
    frozen = frame.benchmark_forecast.to_numpy(float)
    available = np.isfinite(frozen)
    if not np.isfinite(reference[available]).all() or not np.allclose(
            frozen[available], reference[available], rtol=0, atol=1e-8):
        raise ZonalReportError(f"{zone}: frozen Storm prices differ from the original verified archive.")
    audit = {
        "status": "verified_frozen_projection", "source_kind": "projection du snapshot comparatif figé",
        "original_archive": provenance, "original_source_materialization": materialization,
        "published_report_source": source, "compared_finite_hours": int(available.sum()),
        "missing_frozen_hours": int((~available).sum()),
        "archive_values_excluded_by_frozen_panel_mask": int((~available & np.isfinite(reference)).sum()),
        "panel_missing_values_filled": False, "used_for_prediction": False,
        "forecast_pit_certified": False, "benchmark_pit_certified": False,
        "verification_tolerance_eur_mwh": 1e-8,
    }
    return audit, materialization


def _output_directory(value: Path) -> Path:
    path = Path(value).expanduser().resolve()
    project = Path(__file__).resolve().parents[1]
    forbidden = (project / "runs" / "exports", project / "runs" / "live", project / "config")
    if path == project or any(path == p or path.is_relative_to(p) for p in forbidden):
        raise ZonalReportError("Experimental reports must not overwrite the project root or production paths.")
    if path.is_relative_to(project) and not any(path.is_relative_to(p) for p in
            (project / "runs" / "experiments", project / "tmp")):
        raise ZonalReportError("Within the project, reports must stay in runs/experiments or private tmp previews.")
    return path


def _raw_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    raw = frame.rename(columns={"timestamp_utc": "delivery_start_utc"}).copy(deep=True)
    for actual, expected in (
        ("candidate_forecast", raw.forecast + raw.applied_correction),
        ("applied_correction", raw.selected_weight * raw.bounded_correction),
    ):
        if not np.allclose(raw[actual], expected, rtol=1e-9, atol=1e-8, equal_nan=True):
            raise ZonalReportError(f"Saved price/correction/weight arithmetic disagrees: {actual}.")
    for name, point, low, high in (
        ("zonal", "candidate_forecast", "candidate_q10", "candidate_q90"),
        ("nyx", "forecast", "q10", "q90"),
    ):
        numbers = raw[[low, point, high]].to_numpy(float)
        if not np.isfinite(numbers).all() or (np.diff(numbers, axis=1) < 0).any():
            raise ZonalReportError(f"{name}: saved finite ordered P10/P50/P90 are required; no intervals are invented.")
        for quantile, column in (("q10", low), ("q50", point), ("q90", high)):
            raw[f"{name}__{quantile}"] = raw[column]
    return raw


def _result(frame: pd.DataFrame, *, source_audit: dict, directory: Path, label: str) -> tuple[ZoneRunResult, dict]:
    zone = str(frame.zone.iloc[0]); timezone = TIMEZONES[zone]
    evaluation = frame.loc[frame["sample"].eq("evaluation")].copy()
    live = frame.loc[frame["sample"].eq("live")].copy()
    declared = source_audit.get("source_data_audit", {}).get("baseline", {})
    start = pd.Timestamp(declared.get("evaluation_start_day", evaluation.local_day.min()))
    end = pd.Timestamp(declared.get("evaluation_end_day", evaluation.local_day.max()))
    if (end-start).days != 364:
        raise ZonalReportError("The frozen evaluation must declare exactly 365 civil days.")
    expected = pd.date_range(start.tz_localize(timezone), (end+pd.Timedelta(days=1)).tz_localize(timezone),
                             freq="h", inclusive="left").tz_convert("UTC")
    if not pd.DatetimeIndex(evaluation.timestamp_utc).equals(expected):
        raise ZonalReportError(f"{zone}: frozen evaluation must preserve all 365 days of physical hours.")
    delivery = (end+pd.Timedelta(days=1)).date()
    if not pd.DatetimeIndex(live.timestamp_utc).equals(local_delivery_day_index(delivery, timezone=timezone)):
        raise ZonalReportError(f"{zone}: a separate complete live delivery day is required.")
    if frame.forecast_origin_utc.isna().any() or (frame.forecast_origin_utc >= frame.timestamp_utc).any():
        raise ZonalReportError("Saved forecast origins must precede delivery; no origin is fabricated.")
    storm_audit, original_materialization = _verified_storm(frame, source_audit, zone)
    raw = _raw_quantiles(frame)
    eval_raw = raw.loc[raw["sample"].eq("evaluation")].reset_index(drop=True)
    future_raw = raw.loc[raw["sample"].eq("live")].reset_index(drop=True)
    paired = np.isfinite(evaluation[["actual", "forecast", "benchmark_forecast", "candidate_forecast"]].to_numpy(float)).all(axis=1)
    if not paired.any():
        raise ZonalReportError("No paired evaluation hour is available.")
    all_native = _backtest_prediction_frame(eval_raw, model="zonal", row_mask=np.ones(len(eval_raw), bool),
                                             timezone=timezone, allow_missing_actual=True)
    all_baseline = _backtest_prediction_frame(eval_raw, model="nyx", row_mask=np.ones(len(eval_raw), bool),
                                               timezone=timezone, allow_missing_actual=True)
    if len(all_native) != len(evaluation) or len(all_baseline) != len(evaluation):
        raise ZonalReportError("Preparing probabilistic frames removed a frozen physical hour.")
    native, baseline = all_native.loc[paired].copy(), all_baseline.loc[paired].copy()
    target = pd.Series(frame.actual.to_numpy(float), index=pd.DatetimeIndex(frame.timestamp_utc).tz_convert(timezone), name="actual")
    features = [column for column in frame if column.startswith("feature_")]
    displayed = features[:12]
    covariates = frame[displayed].copy(); covariates.index = target.index
    coverage = pd.DataFrame({"alias": displayed, "coverage_after_fill": covariates.notna().mean().to_numpy()})
    manifest = pd.DataFrame({"alias": features, "source": "snapshot expérimental figé", "role": "feature de l’expert (pas attribution du prix)"})
    data = ZoneData(zone, timezone, "h", target, covariates, covariates.copy(), [], coverage, manifest,
                    {"diagnostic_only": True, "production": False, "displayed_feature_subset": displayed})
    by_horizon, by_hour = metric_breakdowns(native, baseline, 150.)
    result = ZoneRunResult(zone, compute_metrics(native, 150.), compute_metrics(baseline, 150.),
        native, baseline, by_horizon, by_hour,
        _forecast_prediction_frame(future_raw, model="zonal", timezone=timezone),
        _forecast_prediction_frame(future_raw, model="nyx", timezone=timezone), data, directory)
    result.statistics_candidate = all_native
    result.statistics_candidate_label = label
    missing = pd.DatetimeIndex(evaluation.loc[evaluation.benchmark_forecast.isna(), "timestamp_utc"])
    missing_values = [stamp.isoformat() for stamp in missing]
    projection = {"kind": "verified_original_archive_with_frozen_comparison_projection", "missing_benchmark_utc": missing_values,
        "matched_hours": int(paired.sum()), "expected_comparison_hours": len(evaluation),
        "source_materialization": original_materialization, "frozen_projection": storm_audit,
        "used_for_prediction": False, "panel_missing_values_filled": False}
    contract = deepcopy(storm_benchmark_contracts(zone, timezone=timezone)[STORM_DASHBOARD_CONTRACT_ID])
    note = (f"Évaluation figée du {start.date()} au {end.date()} : {int(paired.sum())} heures communes. "
            "Storm = projection du snapshot comparatif figé, vérifiée contre son archive officielle. "
            "Les trous du snapshot restent vides même si l’archive contient une valeur. Live exclu de Statistics. "
            "La vérification des prix ne certifie pas leur disponibilité à 08 h.")
    contract.update(status="verified_frozen_projection", used_for_prediction=False, report_note=note, materialization_audit=projection)
    result.statistics_benchmark = pd.DataFrame({"timestamp": all_native.timestamp,
        "actual": evaluation.actual.to_numpy(float), "q50": evaluation.benchmark_forecast.to_numpy(float)})
    result.statistics_benchmark_label = "Storm officiel — snapshot figé"
    contract["report_label"] = result.statistics_benchmark_label
    result.statistics_benchmark_contract = contract
    result.statistics_pairing_audit = {"status": "complete", "role": "verified_official_storm_pairing",
        "zone": zone, "used_for_prediction": False, "expected_hours": len(evaluation), "missing_benchmark_utc": missing_values}
    result.statistics_scope_note = note
    result.hourly_comparison_source = pd.DataFrame({"_timestamp_utc": evaluation.timestamp_utc.to_numpy(),
        "_timestamp_local": all_native.timestamp.to_numpy(), "actual": evaluation.actual.to_numpy(float),
        "q50": evaluation.candidate_forecast.to_numpy(float), "_benchmark_q50": evaluation.benchmark_forecast.to_numpy(float)})
    result.hourly_comparison_contract = contract
    result.forecast_benchmark = pd.DataFrame({"timestamp": result.forecast_native.timestamp, "q50": live.benchmark_forecast.to_numpy(float)})
    result.forecast_benchmark_label = result.statistics_benchmark_label
    audit = {"zone": zone, "evaluation_start_day": str(start.date()), "evaluation_end_day": str(end.date()),
        "evaluation_days": 365, "evaluation_hours": len(evaluation), "paired_hours": int(paired.sum()),
        "live_day": str(delivery), "live_observed_hours": int(live.actual.notna().sum()), "live_excluded_from_statistics": True,
        "displayed_feature_count": len(displayed), "model_feature_column_count": len(features),
        "quantiles": "saved P10/P50/P90; intermediate deciles linearly interpolated only for legacy CRPS display",
        "storm": storm_audit, "diagnostic_only": True, "production_modified": False, "activation_performed": False,
        "forecast_pit_certified": False, "benchmark_pit_certified": False,
        "decision_policy": source_audit.get("decision_policy", "not_declared"),
        "strict_governor_enforced": source_audit.get("strict_governor_enforced"),
        "point_metrics_common_support": True, "price_attribution_available": False}
    return result, audit


def _banner(audit: dict, live: pd.DataFrame) -> str:
    def number(value):
        return f"{float(value):.3f}" if pd.notna(value) and np.isfinite(float(value)) else "—"
    columns = [("local_label", "Heure locale"), ("forecast", "NYX"), ("candidate_forecast", "Prix zonal retenu"),
               ("spike_probability", "Probabilité du risque"), ("bounded_correction", "Correction proposée bornée"),
               ("selected_weight", "Poids appliqué"), ("applied_correction", "Correction appliquée"),
               ("actual", "Observé"), ("benchmark_forecast", "Storm figé"), ("gate_reason", "Décision / motif")]
    rows = []
    for record in live.to_dict("records"):
        rows.append("<tr>"+"".join("<td>"+html.escape(str(record.get(key, "—")) if key in {"local_label", "gate_reason"}
            else number(record.get(key, np.nan)))+"</td>" for key, _ in columns)+"</tr>")
    decision = ('<p><strong>Règle expérimentale : correction fixe de 25 % de la proposition bornée si '
        'p &gt; 0,6 et si l’expert est disponible. Le gouverneur annuel strict n’est PAS appliqué à ce candidat. '
        'La sortie strictement gouvernée est conservée et testée séparément. Aucune garantie de non-régression.</strong></p>'
        if audit.get("decision_policy") == "fixed_25_percent_experimental" else '')
    return ('<section data-report-section="zonal-experiment"><h2>EXPÉRIMENTAL — modèle zonal, aucune activation</h2>'
        +decision+
        f'<p>Backtest figé : {audit["evaluation_start_day"]} → {audit["evaluation_end_day"]}, 365 jours représentés, '
        f'{audit["paired_hours"]} heures communes. Livraison {audit["live_day"]} affichée séparément, '
        'exclue des scores même si son observation est connue.</p>'
        '<p>Année déjà examinée ; entraînement progressif 90–365 jours. Aucun résultat ne constitue une preuve '
        'prospective, une certification PIT à 08 h ou une garantie de non-régression. Les prix opérationnels restent inchangés.</p>'
        '<p>Les P10/P50/P90 proviennent du candidat sauvegardé. Les déciles intermédiaires et le CRPS sont interpolés '
        'pour compatibilité d’affichage, pas de nouveaux quantiles appris. Cartes de prix : moyenne des moyennes '
        'journalières ; MAE et comparatif figé : agrégation horaire sur support commun.</p>'
        '<p>La probabilité explique le risque de forte sous-estimation, pas un poids causal des variables dans le prix. '
        'Aucune attribution du prix final n’est fabriquée. Les séries d’entrée montrées sont un échantillon illustratif.</p>'
        '<details><summary>Livraison — décision et prix horaires (hors scores historiques)</summary><div class="table-wrap"><table><thead><tr>'
        + ''.join('<th>'+html.escape(label)+'</th>' for _, label in columns)+'</tr></thead><tbody>'+''.join(rows)
        + '</tbody></table></div></details></section>')


def _index_comparison(comparison: dict | None) -> str:
    if not comparison or not comparison.get("by_zone"):
        return '<p>Comparatif annuel non fourni ; aucun classement reconstruit.</p>'
    labels = {"nuclear_kalman": "NYX nucléaire + Kalman figé", "storm": "Storm figé",
              "regional_25": "Expert régional précédent — 25 %", "zonal_hiercal": "Expert zonal + calibration — 25 %"}
    def number(value, digits=3):
        return f"{float(value):.{digits}f}" if value is not None and np.isfinite(float(value)) else "—"
    sections = []
    for zone, group in sorted(comparison["by_zone"].items()):
        annual, interventions = group.get("annual", {}), group.get("interventions", {})
        rows = []
        for model, label in labels.items():
            score, intervention = annual.get(model, {}), interventions.get(model, {})
            values = [html.escape(label), number(score.get("hours"), 0)]
            values.extend(number(score.get(key)) for key in ("mae_eur_mwh", "rmse_eur_mwh", "bias_eur_mwh",
                "daily_mean_mae_eur_mwh", "mean_forecast_eur_mwh", "mean_observed_eur_mwh"))
            values.extend(number(intervention.get(key), 0) for key in ("active_hours", "worsened_absolute_error_hours"))
            rows.append('<tr>'+''.join('<td>'+value+'</td>' for value in values)+'</tr>')
        regional = interventions.get("regional_25", {}).get("worsened_absolute_error_hours")
        zonal = interventions.get("zonal_hiercal", {}).get("worsened_absolute_error_hours")
        delta = zonal-regional if regional is not None and zonal is not None else None
        headers = ("Modèle", "Heures communes", "MAE", "RMSE", "Biais", "MAE prix moyens journaliers",
                   "Prix moyen prévu", "Prix moyen observé", "Corrections actives", "Corrections aggravant l’erreur")
        sections.append('<section><h2>'+html.escape(str(zone))+'</h2><div class="table-wrap"><table><thead><tr>'
            +''.join('<th>'+html.escape(title)+'</th>' for title in headers)+'</tr></thead><tbody>'+''.join(rows)
            +'</tbody></table></div><p>Corrections aggravant l’erreur absolue : régional 25 % = '+number(regional, 0)
            +', zonal calibré 25 % = '+number(zonal, 0)+' ; écart zonal − régional = '+number(delta, 0)
            +' heure(s). Un écart négatif signifie moins d’aggravations, sans garantie pour les jours futurs.</p></section>')
    return ('<h2>Comparaison annuelle figée — aucune promotion</h2><p>Prix et erreurs en EUR/MWh, sur les mêmes heures '
        'évaluables pour tous les modèles. Les heures sans correction restent incluses. Les prix moyens ci-dessous '
        'sont des moyennes horaires, distinctes des cartes de moyennes journalières des rapports détaillés. '
        'Une correction aggravant l’erreur n’est pas nécessairement un faux positif du classificateur.</p>'+''.join(sections))


def render_zonal_reports(predictions: pd.DataFrame, *, source_audit: dict, output_directory: Path,
                         model_name: str = "nyx_zonal", comparison: dict | None = None) -> dict[str, Path]:
    """Use the unchanged operational renderer inside a separate experiment folder."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_name):
        raise ZonalReportError("A safe model identifier is required.")
    directory = _output_directory(output_directory)
    frame = _prepare(predictions)
    label = f"NYX + expert zonal ({model_name})"
    prepared = {zone: _result(group.reset_index(drop=True), source_audit=source_audit, directory=directory, label=label)
                for zone, group in frame.groupby("zone", sort=True)}
    if not prepared:
        raise ZonalReportError("No zonal predictions to report.")
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}; audits = {}
    for zone, (result, audit) in prepared.items():
        path = directory / f"forecast_{zone.lower()}_{audit['live_day']}_{model_name}.html"
        write_html_report([result], {"report": {"title": f"{zone} — {label} — EXPÉRIMENTAL", "forecast_history_hours": 168}}, path)
        _replace_report_labels(path, native_label=label, baseline_label="NYX nucléaire + Kalman figé")
        document = path.read_text(encoding="utf-8")
        document = document.replace("Prévision opérationnelle du", "Prévision expérimentale du")
        document = document.replace("prévision Day-Ahead opérationnelle", "prévision Day-Ahead expérimentale")
        document = document.replace("Prévision Day-Ahead réelle", "Prévision Day-Ahead expérimentale — live hors scores")
        document = document.replace("covariables actives", "séries affichées (sélection illustrative)")
        live = frame.loc[frame.zone.eq(zone) & frame["sample"].eq("live")]
        document = document.replace("<main>", "<main>"+_banner(audit, live), 1)
        document = document.replace("</main>", '<section><h2>Audit du rapport expérimental</h2><p><a href="zonal_report_audit.json">Audit JSON</a> · <a href="comparison.json">Comparatif figé</a></p><details><summary>Traçabilité et exclusions</summary><pre>'
            +html.escape(json.dumps(_clean(audit), ensure_ascii=False, indent=2, allow_nan=False))+'</pre></details></section></main>', 1)
        path.write_text(document, encoding="utf-8")
        paths[zone] = path; audits[zone] = audit
    audit_path = directory / "zonal_report_audit.json"
    audit_path.write_text(json.dumps(_clean({"reports": audits, "source_audit": source_audit}), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    compare_path = directory / "comparison.json"
    compare_path.write_text(json.dumps(_clean(comparison or {"status": "not_provided"}), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    index = directory / "index.html"
    links = ''.join(f'<li><a href="{html.escape(path.name)}">{zone} — {html.escape(label)}</a></li>' for zone, path in paths.items())
    decision_note = ('<p>Correction expérimentale fixe à 25 % ; le gouverneur annuel strict n’est pas appliqué à ce candidat. '
        'Sa sortie strictement gouvernée est évaluée séparément. Aucune garantie de non-régression.</p>'
        if source_audit.get("decision_policy") == "fixed_25_percent_experimental" else '')
    index.write_text('<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>NYX zonal — rapports expérimentaux</title><style>body{font:15px system-ui;background:#f4f6f8;color:#18212b;max-width:1500px;margin:30px auto;padding:0 20px}'
        'section{background:white;padding:20px;margin:20px 0;border-radius:10px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%}'
        'th,td{padding:10px;text-align:right;border-bottom:1px solid #dfe5ea}th:first-child,td:first-child{text-align:left}th{background:#edf2f7}a{color:#185995}</style></head>'
        '<body><h1>Rapports zonaux — EXPÉRIMENTAL</h1><p>Format opérationnel, sorties isolées. Aucune activation ni promotion ; live exclu des scores figés.</p>'
        +decision_note+
        '<p><a href="../zonal_comparison.html">Ouvrir le comparatif interactif complet : variantes, erreurs, calibration et extrêmes</a></p><ul>'
        +links+'</ul>'+_index_comparison(comparison)+'</body></html>', encoding="utf-8")
    return {**paths, "index": index, "audit": audit_path, "comparison": compare_path}


__all__ = ["ZonalReportError", "render_zonal_reports"]
