"""Read frozen decisions only; real production-format HTML in a private folder."""
from __future__ import annotations

import html
import json
from pathlib import Path
import numpy as np
import pandas as pd

from chronos2_modular.report import write_html_report
from chronos2_hourly.reporting import _replace_report_labels
from nyx_scarcity.reporting import _prepare, _clean
from nyx_scarcity.variant_reporting import build_comparison, render_comparison, _align
from nyx_scarcity_zonal.reporting import _result
from .policy import validate_output

LABELS = {"hgb_v1": "HGB historique figé", "p50_previous": "P50 forêt précédent",
    "empirical_previous": "P50 empirique précédent", "p50_calibrated": "P50 précédent — intervalles recalibrés",
    "physics_direct": "StressGuard — diagnostic direct", "physics_governed": "StressGuard — gouverné (principal)",
    "nuclear_kalman": "NYX nucléaire + Kalman figé", "storm": "Storm officiel figé"}


def _number(value, digits=3):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def calibration_summary(frame, *, common_keys=None):
    """Do not hide cold-start intervals or pool countries/active states implicitly."""
    frame = _prepare(frame)
    mask = frame.in_evaluation_window & np.isfinite(frame[["actual", "forecast", "benchmark_forecast", "candidate_forecast"]].to_numpy(float)).all(axis=1)
    if common_keys is not None:
        mask &= pd.MultiIndex.from_frame(frame[["zone", "timestamp_utc"]]).isin(common_keys)
    frame = frame.loc[mask]
    result = {}
    for zone in ("all", *sorted(frame.zone.unique())):
        group = frame if zone == "all" else frame.loc[frame.zone.eq(zone)]
        result[zone] = {}
        for regime in ("all", "active", "inactive"):
            rows = group if regime == "all" else group.loc[group.applied_correction.gt(0).eq(regime == "active")]
            if rows.empty:
                result[zone][regime] = {"hours": 0}
                continue
            lo, hi = rows.candidate_q10, rows.candidate_q90
            covered = rows.actual.ge(lo) & rows.actual.le(hi)
            nyx_covered = rows.actual.ge(rows.q10) & rows.actual.le(rows.q90)
            alpha = .2
            score = hi-lo+2/alpha*(lo-rows.actual).clip(lower=0)+2/alpha*(rows.actual-hi).clip(lower=0)
            summary = {"hours": len(rows), "coverage": float(covered.mean()),
                "nyx_same_hours_coverage": float(nyx_covered.mean()), "nominal_coverage": .8,
                "below_lower": float(rows.actual.lt(lo).mean()), "above_upper": float(rows.actual.gt(hi).mean()),
                "mean_width": float((hi-lo).mean()), "nyx_mean_width": float((rows.q90-rows.q10).mean()),
                "interval_score_80": float(score.mean()), "mae": float((rows.actual-rows.candidate_forecast).abs().mean())}
            if "interval_calibration_status" in rows:
                summary["calibration_status"] = rows.interval_calibration_status.fillna("missing").value_counts().to_dict()
            result[zone][regime] = summary
    return result


def _interval_table(summary, zone):
    rows = []
    for regime, label in (("all", "Toutes heures"), ("active", "Interventions"), ("inactive", "Sans intervention")):
        s = summary[zone][regime]
        vals = [label, _number(s.get("hours"), 0), _number(100*s["coverage"], 2) if s.get("hours") else "—",
            _number(100*s["nyx_same_hours_coverage"], 2) if s.get("hours") else "—",
            _number(s.get("mean_width")), _number(s.get("nyx_mean_width")), _number(s.get("interval_score_80"))]
        rows.append("<tr>"+"".join("<td>"+html.escape(v)+"</td>" for v in vals)+"</tr>")
    heads = ["Groupe", "Heures", "Couverture %", "NYX mêmes heures %", "Largeur", "Largeur NYX", "Score intervalle80"]
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in heads)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def _banner(zone, variant, summary, live):
    rows = []
    columns = [("local_label", "Heure"), ("spike_probability", "p hausse forte"),
        ("forecast", "NYX"), ("candidate_forecast", "P50"), ("applied_correction", "Correction"),
        ("candidate_q10", "Borne basse"), ("candidate_q90", "Borne haute"),
        ("gate_reason", "Décision"), ("interval_calibration_status", "Calibration")]
    for item in live.to_dict("records"):
        vals = [str(item.get(key, "—")) if key in {"local_label", "gate_reason", "interval_calibration_status"}
                else _number(item.get(key)) for key, _ in columns]
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
    return ('<section data-report-section="stress-guard"><h2>STRESS GUARD — EXPÉRIMENTAL, AUCUNE ACTIVATION</h2>'
        '<p>'+html.escape(LABELS[variant])+'. Le prix opérationnel n’est pas remplacé.</p>'
        '<p>Signal : synchronie régionale, profils prévus de charge résiduelle/vent/solaire, positions dans la journée et '
        'rangs historiques appris sur le core passé uniquement. Pas de prix électrique, NYX ou Storm en entrée du détecteur. '
        'Les disponibilités de parc restent des proxys incomplets ; aucun import JAO non interprétable à 08 h n’est ajouté.</p>'
        '<p>Nouvelle hypothèse pré-déclarée : p &gt; 50 % dans tous les pays, sans porte rigide de pression par pays. '
        'La variante gouvernée décide un poids sur les 90 jours précédents ; la variante directe reste un diagnostic. '
        'La probabilité est celle du risque avant gouvernance, pas une probabilité de la distribution finale.</p>'
        '<p>Événement détecté : sous-prévision de NYX (observé − NYX) supérieure ou égale au seuil appris sur le core passé ; '
        'il ne s’agit pas d’une hausse absolue du prix par rapport à la veille.</p>'
        '<h3>INTERVALLES — CALIBRATION CHRONOLOGIQUE</h3><p>Enveloppe contenant NYX et les quantiles de l’expert pondéré, '
        'puis élargissement unilatéral à partir des erreurs hors échantillon déjà connues. Groupe pays × intervention ; '
        'repli régional sur le même état seulement, sinon amorçage explicite. P50 inchangé par le calibrage. '
        'La couverture ne peut pas baisser par rapport à NYX sur les mêmes heures, mais 80 % n’est pas garanti. '
        'Une couverture supérieure s’évalue aussi avec la largeur et le score d’intervalle ; elle n’est pas gratuite. '
        'Le CRPS du rapport natif est approché à partir de déciles interpolés entre P10, P50 et P90, '
        'pas calculé sur une distribution prédictive complète.</p>'
        +_interval_table(summary, zone)+
        '<p>Année déjà examinée, entraînement progressif plafonné à365 jours : ce replay n’est pas une validation indépendante. '
        'La livraison affichée reste hors Statistics. Le journal prospectif séparé ne comptera que les nouvelles prévisions '
        'enregistrées avant observation, selon un protocole gelé ; aucune journée nouvelle n’est inventée ici.</p>'
        '<details><summary>Prévision live — décisions et intervalles enregistrés</summary><div class="table-wrap"><table><thead><tr>'
        +''.join('<th>'+html.escape(label)+'</th>' for _, label in columns)+'</tr></thead><tbody>'+''.join(rows)
        +'</tbody></table></div></details></section>')


def render_reports(predictions, *, source_audit, model_audit, output_directory, root):
    from .runner import safe_path
    directory = safe_path(root, output_directory)
    for key in ("physics_direct", "physics_governed", "p50_calibrated"):
        validate_output(predictions[key])
    if not predictions["p50_calibrated"].candidate_forecast.equals(predictions["p50_previous"].candidate_forecast):
        raise ValueError("Interval-only control changed a P50.")
    comparison = build_comparison(predictions)
    comparison.update(model_labels=LABELS, primary_variant="physics_governed", primary_selected_before_new_backtest=True,
        prospective_validation_completed=False, exploratory_year_already_examined=True, annual_non_regression_guaranteed=False)
    shared, _, common = _align(predictions)
    common_keys = pd.MultiIndex.from_frame(shared.loc[common, ["zone", "timestamp_utc"]])
    summaries = {name: calibration_summary(frame, common_keys=common_keys) for name, frame in predictions.items()}
    baseline = predictions["physics_governed"].copy(deep=True)
    baseline["candidate_forecast"] = baseline.forecast
    baseline["candidate_q10"], baseline["candidate_q90"] = baseline.q10, baseline.q90
    baseline["applied_correction"] = 0.
    baseline = baseline.drop(columns="interval_calibration_status", errors="ignore")
    summaries["nuclear_kalman"] = calibration_summary(baseline, common_keys=common_keys)
    summary_path = directory/"interval_comparison.json"
    summary_path.write_text(json.dumps(_clean(summaries), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    comparison_path = directory/"comparison.json"
    comparison_path.write_text(json.dumps(_clean(comparison), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    interactive = directory/"stress_guard_comparison.html"
    render_comparison(predictions, comparison, {"models": model_audit, "data": source_audit["source_data_audit"],
        "diagnostic_only": True, "production_modified": False}, interactive)
    document = interactive.read_text(encoding="utf-8")
    document = document.replace("NYX — comparaison contrôlée des experts XGB", "NYX — StressGuard : physique et intervalles")
    document = document.replace("NYX — variantes XGB et seuils causaux", "NYX — StressGuard : expérience pré-déclarée")
    document = document.replace("P.summary.variants.includes('xgb_unweighted_fixed')?'xgb_unweighted_fixed':'hgb_v1'", "P.summary.primary_variant||'physics_governed'")
    document = document.replace("<main>", '<main><section><h2>StressGuard — laboratoire isolé</h2><p>Année exploratoire déjà examinée. '
        'Le témoin P50 recalibré conserve exactement le prix précédent. Le nouveau signal physique est une hypothèse '
        'à tester sur de nouvelles journées, pas un succès acquis. Aucune activation.</p><p><a href="index.html">Rapports et calibration détaillée</a></p></section>', 1)
    interactive.write_text(document, encoding="utf-8")
    paths = {"comparison": comparison_path, "interactive": interactive, "interval_comparison": summary_path}
    report_audit, links = {}, []
    for variant in ("physics_governed", "physics_direct"):
        frame = _prepare(predictions[variant])
        features = [c for c in frame if c.startswith("feature_fundamental_")]
        frame = frame.drop(columns=[c for c in frame if c.startswith("feature_") and c not in features])
        target = safe_path(root, directory/variant)
        target.mkdir(parents=True, exist_ok=True)
        for zone, group in frame.groupby("zone", sort=True):
            result, audit = _result(group.reset_index(drop=True), source_audit={**source_audit,
                "decision_policy": variant, "strict_governor_enforced": variant == "physics_governed"},
                directory=target, label=LABELS[variant])
            audit.update(expert="stress_guard", interval_summary=summaries[variant][zone],
                final_quantile_method="baseline_union_then_chronological_expansion", feature_allowlist=features)
            path = target/f"forecast_{zone.lower()}_{audit['live_day']}_{variant}.html"
            write_html_report([result], {"report": {"title": f"{zone} — {LABELS[variant]}", "forecast_history_hours": 168}}, path)
            _replace_report_labels(path, native_label=LABELS[variant], baseline_label="NYX nucléaire + Kalman figé")
            document = path.read_text(encoding="utf-8")
            for old, new in (("Prévision opérationnelle du", "Prévision expérimentale du"),
                ("prévision Day-Ahead opérationnelle", "prévision Day-Ahead expérimentale"),
                ("Prévision Day-Ahead réelle", "Prévision expérimentale — live hors Statistics"),
                ("covariables actives", "fondamentaux affichés (sélection illustrative)")):
                document = document.replace(old, new)
            document = document.replace("<main>", "<main>"+_banner(zone, variant, summaries[variant], group.loc[group["sample"].eq("live")]), 1)
            document = document.replace("</main>", '<section><p><a href="../index.html">Synthèse StressGuard</a> · '
                '<a href="../stress_guard_comparison.html">Comparatif interactif</a> · '
                '<a href="../report_audit.json">Audit</a></p></section></main>', 1)
            document.encode("utf-8")
            path.write_text(document, encoding="utf-8")
            paths[variant+"_"+zone] = path
            report_audit[variant+"_"+zone] = audit
            links.append('<li><a href="'+html.escape(path.relative_to(directory).as_posix())+'">'+zone+' — '+LABELS[variant]+'</a></li>')
    audit_path = directory/"report_audit.json"
    audit_path.write_text(json.dumps(_clean(report_audit), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    paths["audit"] = audit_path
    rows = []
    annual = comparison["overall"]["annual"]
    for key, label in LABELS.items():
        if key not in annual:
            continue
        vals = [label, _number(annual[key].get("mae_eur_mwh"))]
        s = summaries.get(key, {}).get("all", {}).get("all", {})
        vals.extend([_number(100*s["coverage"], 2) if s else "—", _number(s.get("mean_width")), _number(s.get("interval_score_80"))])
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
    index = directory/"index.html"
    index.write_text('<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>StressGuard — comparaison et calibration</title><style>body{font:15px system-ui;max-width:1450px;margin:30px auto;padding:0 20px;background:#f4f6f8;color:#172433}'
        'section{background:white;padding:20px;margin:20px 0;border-radius:10px}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:right;border-bottom:1px solid #dde3e9}'
        'th:first-child,td:first-child{text-align:left}.table-wrap{overflow:auto}a{color:#185995}</style></head><body><h1>StressGuard — expérimental, hors production</h1>'
        '<p>Signal physique enrichi, calibration chronologique, protocole prospectif séparé. Aucun réglage par pays choisi sur cette année. '
        'Le candidat principal pré-déclaré est gouverné ; l’expérience ne garantit pas qu’il sera meilleur.</p>'
        '<p><a href="stress_guard_comparison.html">Comparatif interactif complet</a> · <a href="interval_comparison.json">Calibration détaillée JSON</a></p>'
        '<section><h2>Année commune et coût des intervalles</h2><div class="table-wrap"><table><thead><tr><th>Modèle</th><th>MAE</th><th>Couverture %</th><th>Largeur moyenne</th><th>Score intervalle80</th></tr></thead><tbody>'
        +''.join(rows)+'</tbody></table></div><p>MAE, largeur et score en EUR/MWh. Même support horaire apparié. Une couverture supérieure avec des bornes '
        'plus larges n’est pas automatiquement une meilleure prévision probabiliste.</p></section><section><h2>Rapports au format habituel</h2><ul>'
        +''.join(links)+'</ul></section><section><h2>Validation indépendante</h2><p>Non réalisée : les observations jusqu’au15septembre sont déjà connues '
        'dans la source. Un journal gelé séparé refuse ces dates et les émissions trop tardives. Les futures journées devront être évaluées '
        'sans retoucher ce candidat. Information à08h et heure d’émission réelle seront distinguées.</p></section></body></html>', encoding="utf-8")
    paths["index"] = index
    return paths
