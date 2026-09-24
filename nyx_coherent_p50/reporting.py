"""Production-format reports for the isolated coherent residual-CDF experiment."""
from __future__ import annotations

import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from chronos2_hourly.reporting import _replace_report_labels
from chronos2_modular.report import write_html_report
from nyx_scarcity.reporting import _clean, _prepare
from nyx_scarcity_zonal.reporting import _result


_NAMESPACE = Path(__file__).resolve().parents[1] / "runs/experiments/nyx_scarcity_v1/coherent_p50"
_PREFIX = "feature_fundamental_"


class P50ReportError(ValueError):
    """Saved forecast arithmetic or isolated report identity is invalid."""


def _number(value, digits=3):
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "—"


def _policy(value):
    if value not in {"direct", "governed"}:
        raise P50ReportError("decision_policy must be direct or governed.")
    return value


def _validate_decisions(frame, policy):
    required = {"mixture_error_q10", "mixture_error_q50", "mixture_error_q90", "mixture_raw_p50_eur_mwh",
        "risk_probability_gate", "physical_gate_passed", "proposal_reason", "threshold_eur_mwh"}
    if not required.issubset(frame):
        raise P50ReportError(f"Missing saved CDF decision columns: {sorted(required-set(frame))}.")
    grid = np.array([0., 1.] if policy == "direct" else [0., .25, .5, 1.])
    weights = frame.selected_weight.to_numpy(float)
    if not np.isfinite(weights).all() or not np.isclose(weights[:, None], grid[None, :], rtol=0, atol=1e-9).any(axis=1).all():
        raise P50ReportError("Saved weights differ from the declared decision policy.")
    qcols = ["mixture_error_q10", "mixture_error_q50", "mixture_error_q90"]
    quantiles = frame[qcols].to_numpy(float)
    present = np.isfinite(quantiles).all(axis=1)
    if np.isinf(quantiles).any() or ((~np.isnan(quantiles).all(axis=1)) & ~present).any():
        raise P50ReportError("Raw mixture quantiles must be jointly available and finite.")
    if ((quantiles[:, 0] > quantiles[:, 1]) | (quantiles[:, 1] > quantiles[:, 2])).any():
        raise P50ReportError("Raw mixture quantiles are crossing.")
    ready = frame.expert_ready.to_numpy(bool)
    if (ready & ~present).any():
        raise P50ReportError("A ready CDF expert must have complete saved quantiles.")
    raw_price = frame.mixture_raw_p50_eur_mwh.to_numpy(float)
    if not np.allclose(raw_price[present], frame.forecast.to_numpy(float)[present]+quantiles[present, 1], rtol=0, atol=1e-8):
        raise P50ReportError("The raw mixture P50 differs from NYX plus its signed-error median.")
    high = present & frame.spike_probability.gt(.5).to_numpy()
    threshold = frame.threshold_eur_mwh.to_numpy(float)
    if (high & (~np.isfinite(threshold) | (threshold <= 0))).any() or (quantiles[high, 1] < threshold[high]-1e-8).any():
        raise P50ReportError("With event probability above one half, the raw P50 must belong to the spike support.")
    active = frame.applied_correction.gt(1e-9).to_numpy()
    if frame.applied_correction.lt(-1e-9).any() or frame.bounded_correction.lt(0).any() or frame.bounded_correction.gt(400.+1e-9).any():
        raise P50ReportError("The positive proposal must remain bounded between zero and 400 EUR/MWh.")
    prior = frame.risk_probability_gate.to_numpy(float)
    risk = np.isfinite(prior) & (prior >= 0) & (prior <= 1) & frame.spike_probability.gt(prior).to_numpy()
    physical = frame.physical_gate_passed.eq(True).fillna(False).to_numpy(bool)
    if (active & ~(ready & present & risk & physical & (quantiles[:, 1] > 0))).any():
        raise P50ReportError("An applied correction fails its saved readiness, probability, physical or positive-median gate.")
    raw = frame.raw_correction.to_numpy(float)
    eligible = ready & present & risk & physical & (quantiles[:, 1] > 0)
    expected_raw = np.where(eligible, quantiles[:, 1], 0.)
    if not np.allclose(raw, expected_raw, rtol=0, atol=1e-8) or not np.allclose(frame.bounded_correction, np.minimum(expected_raw, 400.), rtol=0, atol=1e-8):
        raise P50ReportError("Saved raw or bounded correction differs from the gated positive mixture median.")
    if policy == "direct" and not np.array_equal(weights, eligible.astype(float)):
        raise P50ReportError("Direct decisions must use weight one on every eligible positive proposal and zero otherwise.")
    if not np.allclose(frame.applied_correction, weights*frame.bounded_correction, rtol=0, atol=1e-8):
        raise P50ReportError("Applied correction differs from saved weight times bounded proposal.")
    baseline = frame[["q10", "forecast", "q90"]].to_numpy(float)
    final = frame[["candidate_q10", "candidate_forecast", "candidate_q90"]].to_numpy(float)
    if not np.isfinite(baseline).all() or not np.isfinite(final).all():
        raise P50ReportError("Finite baseline and final P10/P50/P90 are required.")
    expected = baseline.copy()
    changed = weights > 0
    if (changed & ~present).any():
        raise P50ReportError("A nonzero decision weight requires an available residual CDF.")
    capped = frame.forecast.to_numpy(float)[:, None] + np.minimum(quantiles, 400.)
    expected[changed] = (1.-weights[changed, None])*baseline[changed] + weights[changed, None]*capped[changed]
    if not np.allclose(final, expected, rtol=0, atol=1e-8):
        raise P50ReportError("Final quantiles differ from saved quantile-function interpolation; zero weight must preserve NYX.")
    if ((final[:, 0] > final[:, 1]) | (final[:, 1] > final[:, 2])).any():
        raise P50ReportError("Final interpolated quantiles are crossing.")
    return {"policy": policy, "allowed_weights": grid.tolist(), "active_hours_including_live": int(active.sum()),
        "complete_raw_distribution_hours": int(present.sum()), "probability_above_half_hours": int(high.sum()),
        "raw_median_spike_support_checked": True, "saved_probability_and_physical_gates_checked": True,
        "quantile_function_interpolation_checked": True, "thresholds_reestimated_by_report": False,
        "classifier_probability_is_final_decision_distribution_probability": False}


def _decision_text(policy):
    if policy == "governed":
        return ("Version gouvernée : poids 0 %, 25 %, 50 % ou 100 % décidé sur les erreurs hors échantillon déjà "
                "publiées des 90 jours antérieurs. Le gouverneur peut conserver NYX. Aucune garantie de non-régression.")
    return ("Version directe — diagnostic principal : poids de 100 % après les portes de risque, de stress physique "
            "et de médiane positive ; sinon NYX est conservé. Le gouverneur strict n’est PAS appliqué. Aucune activation en production.")


def _decision_table(frame):
    columns = [("local_label", "Heure locale"), ("forecast", "NYX figé"),
        ("spike_probability", "p événement — classificateur"), ("threshold_eur_mwh", "Seuil d’erreur u"),
        ("mixture_error_q10", "Erreur Q10 brute"), ("mixture_error_q50", "Erreur Q50 brute"),
        ("mixture_error_q90", "Erreur Q90 brute"), ("mixture_raw_p50_eur_mwh", "P50 brut NYX + erreur"),
        ("risk_probability_gate", "Prévalence core / seuil p"), ("physical_gate_passed", "Porte physique"),
        ("proposal_reason", "Motif de proposition"), ("bounded_correction", "Correction bornée"),
        ("selected_weight", "Poids"), ("applied_correction", "Correction appliquée"),
        ("candidate_forecast", "P50 final expérimental"), ("actual", "Observé"),
        ("benchmark_forecast", "Storm figé"), ("gate_reason", "Décision / refus")]
    rows = []
    for record in frame.reindex(columns=[key for key, _ in columns]).to_dict("records"):
        cells = []
        for key, _ in columns:
            value = record.get(key)
            if key in {"local_label", "proposal_reason", "gate_reason"}:
                rendered = str(value) if value is not None and not pd.isna(value) else "—"
            elif key == "physical_gate_passed":
                rendered = "—" if value is None or pd.isna(value) else ("Oui" if value == True else "Non")
            else:
                rendered = _number(value)
            cells.append("<td>"+html.escape(rendered)+"</td>")
        rows.append("<tr>"+"".join(cells)+"</tr>")
    return ('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _, label in columns)
            +'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>')


def _banner(audit, group, policy):
    live = group.loc[group["sample"].eq("live")]
    case = group.loc[group.local_day.eq("2026-09-14") & group.local_hour.eq(19) & group["sample"].eq("evaluation")]
    return ('<section data-report-section="coherent-p50-experiment"><h2>P50 COHÉRENT — EXPÉRIMENTAL, AUCUNE ACTIVATION</h2>'
        '<p><strong>'+html.escape(_decision_text(policy))+'</strong></p>'
        f'<p>Évaluation figée : {audit["evaluation_start_day"]} → {audit["evaluation_end_day"]}, 365 jours représentés, '
        f'{audit["paired_hours"]} heures communes avec Storm. Livraison {audit["live_day"]} visible séparément et '
        'exclue des Statistics, même si son observation est publiée.</p>'
        '<p>Le P50 brut provient désormais de la même distribution que le risque de hausse : '
        '<strong>si p &gt; 50 %, sa médiane d’erreur brute est au moins égale au seuil de forte erreur u</strong>. '
        'Cette propriété précède les portes d’intervention, le plafond de correction et la gouvernance.</p>'
        '<details><summary>Méthode, interprétation des quantiles et limites</summary>'
        '<p>Événement : erreur observé − NYX ≥ u, avec u = max(50 EUR/MWh, Q95 des erreurs du core par pays). '
        'La probabilité provient du classificateur fondamental figé, calibré sur un bloc chronologique distinct. '
        'Deux distributions conditionnelles d’erreur normalisée sont combinées : régime ordinaire E/u &lt; 1 '
        'et régime extrême E/u ≥ 1. Leurs masses sont respectivement 1 − p et p. '
        'La médiane est l’inverse de cette CDF, pas un modèle de médiane indépendant, ni p × sévérité, ni une masse artificielle à zéro.</p>'
        '<p>Une hausse est proposée lorsque p dépasse la prévalence historique core du pays, que la porte physique '
        'est ouverte et que Q50 de l’erreur est positive. La porte physique est apprise sur le core : pression locale '
        'au-dessus de Q90 et rampe positive pour FR/DE, pression des voisins au-dessus de Q90 pour BE/NL. '
        'Le plafond unilatéral est de 400 EUR/MWh. p &gt; 50 % est une propriété de la médiane de la CDF, '
        'pas un seuil de déclenchement ajusté sur cette année.</p>'
        '<p><strong>Quantiles finaux :</strong> Qfinal(a) = (1 − w) QNYX(a) + w [NYX + min(Qerreur(a), 400)]. '
        'C’est une interpolation de fonctions quantiles, PAS un mélange des CDF NYX/expert. À poids nul, les '
        'P10/P50/P90 NYX restent inchangés. La probabilité affichée est celle du classificateur avant décision ; '
        'elle ne décrit pas automatiquement la probabilité de la distribution finale après interpolation. '
        'L’ordre des quantiles est vérifié, pas leur calibration future.</p>'
        '<p><strong>Entrées :</strong> seulement les variables explicitement préfixées <code>feature_fundamental_</code> '
        'et l’identité du pays. Aucun prix électrique passé, NYX, quantile NYX ou Storm en entrée des estimateurs. '
        'L’erreur de prix historique sert de cible supervisée ; NYX sert aussi de base au prix corrigé.</p>'
        '<p><strong>Limites :</strong> année déjà examinée, hypothèse exploratoire ; entraînement progressif 90–365 jours ; '
        'offre et disponibilité décrites par des proxys incomplets ; température lacunaire après le 4 septembre ; réseau JAO exclu. '
        'Les requêtes historiques as-of ne certifient pas chaque publication à 08 h. Pas de preuve prospective ni de gain garanti. '
        'Les déciles intermédiaires et le CRPS du moteur historique sont interpolés depuis les P10/P50/P90 sauvegardés, '
        'pas appris séparément. Les cartes de prix utilisent des moyennes de moyennes journalières ; les scores annuels sont horaires.</p>'
        '</details><details><summary>Livraison — mécanisme de correction du P50, hors scores historiques</summary>'
        +_decision_table(live)+'</details><details><summary>Cas historique — 14 septembre à 19 h, inclus dans les scores</summary>'
        +(_decision_table(case) if not case.empty else '<p>Heure absente de ce snapshot ; aucun prix reconstruit.</p>')
        +'</details></section>')


def _index_comparison(comparison):
    if not comparison or not comparison.get("by_zone"):
        return '<p>Comparatif annuel non fourni ; aucun classement inventé.</p>'
    labels = {"nuclear_kalman": "NYX nucléaire + Kalman figé", "storm": "Storm figé", "hgb_v1": "HGB témoin historique",
        "regional_25": "Expert régional précédent — 25 %", "fundamental_old": "Fondamental précédent — gouverné",
        "forest": "P50 forêt conditionnelle — direct principal", "forest_governed": "P50 forêt conditionnelle — gouverné",
        "empirical": "P50 empirique — direct témoin", "empirical_governed": "P50 empirique — gouverné témoin"}
    output = []
    for zone, group in sorted(comparison["by_zone"].items()):
        annual, impacts = group.get("annual", {}), group.get("interventions", {})
        rows = []
        for name, label in labels.items():
            if name not in annual:
                continue
            score, impact = annual[name], impacts.get(name, {})
            cells = [html.escape(label), _number(score.get("hours"), 0)]
            cells.extend(_number(score.get(key)) for key in ("mae_eur_mwh", "rmse_eur_mwh", "bias_eur_mwh",
                "daily_mean_mae_eur_mwh", "mean_forecast_eur_mwh", "mean_observed_eur_mwh"))
            cells.extend(_number(impact.get(key), 0) for key in ("active_hours", "worsened_absolute_error_hours"))
            rows.append('<tr>'+''.join('<td>'+value+'</td>' for value in cells)+'</tr>')
        headings = ("Modèle", "Heures communes", "MAE", "RMSE", "Biais", "MAE prix moyens journaliers",
                    "Prix moyen prévu", "Prix moyen observé", "Corrections actives", "Corrections aggravantes")
        output.append('<section><h2>'+html.escape(str(zone))+'</h2><div class="table-wrap"><table><thead><tr>'
            +''.join('<th>'+html.escape(title)+'</th>' for title in headings)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></section>')
    return ('<h2>Année commune — comparaison sans promotion</h2><p>Prix et erreurs en EUR/MWh sur les mêmes heures. '
        'Les retours à NYX sont inclus. Une correction aggravante augmente l’erreur absolue ; elle ne signifie pas '
        'nécessairement un faux positif du classificateur. Aucun candidat n’est choisi automatiquement.</p>'+''.join(output))


def render_p50_reports(predictions, *, source_audit, output_directory, comparison=None,
                       decision_policy="direct", model_name="coherent_p50"):
    """Render saved coherent-P50 prices with no source refresh, fitting or activation."""
    policy = _policy(decision_policy)
    if not isinstance(model_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", model_name):
        raise P50ReportError("A safe model identifier is required.")
    raw_directory = Path(output_directory).expanduser()
    if ".." in raw_directory.parts:
        raise P50ReportError("Reports may not traverse parent directories.")
    directory = raw_directory.resolve()
    if not directory.is_relative_to(_NAMESPACE.resolve()) or directory == _NAMESPACE.resolve():
        raise P50ReportError("Reports must remain under the isolated coherent_p50 namespace.")
    frame = _prepare(predictions)
    features = [c for c in frame if c.startswith(_PREFIX)]
    if not features:
        raise P50ReportError("No declared fundamental input feature is available.")
    frame = frame.drop(columns=[c for c in frame if c.startswith("feature_") and c not in features])
    _validate_decisions(frame, policy)
    label = "NYX + P50 cohérent — direct expérimental" if policy == "direct" else "NYX + P50 cohérent — gouverné expérimental"
    prepared = {}
    for zone, group in frame.groupby("zone", sort=True):
        result, audit = _result(group.reset_index(drop=True), source_audit=source_audit, directory=directory, label=label)
        result.zone_data.input_manifest["source"] = "fondamentaux du snapshot P50 expérimental figé"
        result.zone_data.input_manifest["role"] = "entrée de l’expert ; jamais un poids d’attribution du prix"
        result.zone_data.diagnostics.update(fundamental_only_display=True, feature_prefix=_PREFIX)
        audit.update(expert="coherent_p50", decision_policy=policy, strict_governor_enforced=policy == "governed",
            allowed_input_features=features, fundamental_only_display=True,
            historical_price_error_used_as_supervised_target=True, attribution_available=False,
            final_quantile_method="interpolation_of_quantile_functions_not_mixture_of_cdfs",
            displayed_probability_scope="frozen_event_classifier_before_gates_and_decision",
            decision_checks=_validate_decisions(group, policy))
        prepared[zone] = result, audit
    if not prepared:
        raise P50ReportError("No predictions to render.")
    directory.mkdir(parents=True, exist_ok=True)
    paths, audits = {}, {}
    for zone, (result, audit) in prepared.items():
        path = directory/f"forecast_{zone.lower()}_{audit['live_day']}_{model_name}.html"
        write_html_report([result], {"report": {"title": f"{zone} — {label}", "forecast_history_hours": 168}}, path)
        _replace_report_labels(path, native_label=label, baseline_label="NYX nucléaire + Kalman figé")
        document = path.read_text(encoding="utf-8")
        for old, new in (("Prévision opérationnelle du", "Prévision expérimentale du"),
            ("prévision Day-Ahead opérationnelle", "prévision Day-Ahead expérimentale"),
            ("Prévision Day-Ahead réelle", "Prévision Day-Ahead expérimentale — live hors scores"),
            ("covariables actives", "séries fondamentales affichées (sélection illustrative)")):
            document = document.replace(old, new)
        document = document.replace("<main>", "<main>"+_banner(audit, frame.loc[frame.zone.eq(zone)], policy), 1)
        document = document.replace("</main>", '<section><h2>Audit P50 cohérent</h2><p><a href="p50_report_audit.json">Audit JSON</a> · '
            '<a href="comparison.json">Comparatif figé</a> · <a href="../coherent_p50_comparison.html">Comparatif interactif complet</a></p>'
            '<details><summary>Sources, masque figé et contrôles</summary><pre>'
            +html.escape(json.dumps(_clean(audit), ensure_ascii=False, indent=2, allow_nan=False))+'</pre></details></section></main>', 1)
        path.write_text(document, encoding="utf-8"); paths[zone] = path; audits[zone] = audit
    audit_path = directory/"p50_report_audit.json"
    audit_path.write_text(json.dumps(_clean({"reports": audits, "source_audit": source_audit}), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    comparison_path = directory/"comparison.json"
    comparison_path.write_text(json.dumps(_clean(comparison or {"status": "not_provided"}), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    links = ''.join('<li><a href="'+html.escape(path.name)+'">'+zone+' — '+html.escape(label)+'</a></li>' for zone, path in paths.items())
    index = directory/"index.html"
    index.write_text('<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>P50 cohérent — expérimental</title><style>body{font:15px system-ui;background:#f4f6f8;color:#18212b;max-width:1550px;margin:30px auto;padding:0 20px}'
        'section{background:white;padding:20px;margin:20px 0;border-radius:10px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%}'
        'th,td{padding:10px;text-align:right;border-bottom:1px solid #dfe5ea}th:first-child,td:first-child{text-align:left}th{background:#edf2f7}a{color:#185995}</style></head><body>'
        '<h1>P50 cohérent — EXPÉRIMENTAL, AUCUNE ACTIVATION</h1><p>'+html.escape(_decision_text(policy))+'</p>'
        '<p>Live exclu des Statistics figées ; probabilité brute du classificateur distincte de la distribution finale après décision.</p>'
        '<p><a href="../coherent_p50_comparison.html">Comparatif interactif complet : forêt, témoin empirique et décisions directe/gouvernée</a></p><ul>'
        +links+'</ul>'+_index_comparison(comparison)+'</body></html>', encoding="utf-8")
    return {**paths, "index": index, "audit": audit_path, "comparison": comparison_path}


__all__ = ["P50ReportError", "render_p50_reports"]
