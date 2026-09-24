"""Frozen fundamental-expert reports using the unchanged production HTML engine."""
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


_NAMESPACE = Path(__file__).resolve().parents[1] / "runs/experiments/nyx_scarcity_v1/fundamental"
_PREFIX = "feature_fundamental_"


class FundamentalReportError(ValueError):
    """Frozen prices, gate information or output identity cannot be verified."""


def _number(value, digits=3) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "—"


def _policy(value: str) -> str:
    if value == "fixed_25":
        value = "fixed25"
    if value not in {"governed", "fixed25"}:
        raise FundamentalReportError("decision_policy must be governed or fixed25.")
    return value


def _validate_decisions(frame: pd.DataFrame, policy: str) -> dict:
    known = frame.selected_weight.dropna().to_numpy(float)
    grid = np.array([0., .25, .5, 1.] if policy == "governed" else [0., .25])
    if len(known) and not np.isclose(known[:, None], grid[None, :], rtol=0, atol=1e-9).any(axis=1).all():
        raise FundamentalReportError("Saved weights differ from the declared decision policy.")
    if frame.applied_correction.dropna().lt(-1e-9).any():
        raise FundamentalReportError("This experiment only applies the positive part of the signed-error median.")
    if frame.bounded_correction.dropna().lt(0).any() or frame.bounded_correction.dropna().gt(400.+1e-9).any():
        raise FundamentalReportError("The saved fundamental proposal must be bounded between 0 and 400 EUR/MWh.")
    active = frame.applied_correction.gt(1e-9)
    probability_verified = "risk_probability_gate" in frame
    physical_verified = "physical_gate_passed" in frame
    if probability_verified:
        threshold = pd.to_numeric(frame.risk_probability_gate, errors="raise")
        valid = np.isfinite(threshold) & threshold.between(0, 1) & frame.spike_probability.gt(threshold)
        if (active & ~valid).any():
            raise FundamentalReportError("An applied correction fails its saved core-prevalence probability gate.")
    if physical_verified:
        valid = frame.physical_gate_passed.eq(True).fillna(False)
        if (active & ~valid).any():
            raise FundamentalReportError("An applied correction fails its saved physical gate.")
    return {"policy": policy, "allowed_weights": grid.tolist(), "active_hours_including_live": int(active.sum()),
            "saved_probability_gate_checked": probability_verified, "saved_physical_gate_checked": physical_verified,
            "thresholds_reestimated_by_report": False}


def _decision_text(policy: str) -> str:
    if policy == "governed":
        return ("Version principale gouvernée : poids 0 %, 25 %, 50 % ou 100 %, choisi à partir des 90 jours "
                "antérieurs disponibles. Le gouverneur strict peut refuser l’intervention et conserver NYX. "
                "Ce contrôle historique ne garantit pas l’absence de régression future.")
    return ("Diagnostic distinct à poids fixe de 25 % après les portes de risque et de stress physique. "
            "Le gouverneur strict n’est PAS appliqué à cette version. Ce n’est ni une activation "
            "ni un candidat automatiquement sélectionné ; aucune garantie de non-régression.")


def _banner(audit: dict, live: pd.DataFrame, policy: str) -> str:
    columns = [("local_label", "Heure locale"), ("forecast", "NYX figé"),
        ("predicted_signed_residual_median", "Médiane d’erreur signée"), ("spike_probability", "Probabilité de forte erreur"),
        ("risk_probability_gate", "Prévalence core / seuil p"), ("physical_gate_passed", "Porte physique"),
        ("proposal_reason", "Motif de proposition"), ("bounded_correction", "Proposition positive bornée"),
        ("selected_weight", "Poids retenu"), ("applied_correction", "Correction appliquée"),
        ("candidate_forecast", "Prix final expérimental"), ("actual", "Observé"),
        ("benchmark_forecast", "Storm figé"), ("gate_reason", "Décision / refus")]
    text_fields = {"local_label", "proposal_reason", "gate_reason"}
    rows = []
    # Never embed the full feature matrix in the decision table.
    for record in live.loc[:, [key for key, _ in columns if key in live]].to_dict("records"):
        cells = []
        for key, _ in columns:
            value = record.get(key)
            if key in text_fields:
                rendered = str(value) if value is not None and not pd.isna(value) else "—"
            elif key == "physical_gate_passed":
                rendered = "—" if value is None or pd.isna(value) else ("Oui" if value == True else "Non")
            else:
                rendered = _number(value)
            cells.append('<td>'+html.escape(rendered)+'</td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    return ('<section data-report-section="fundamental-experiment"><h2>EXPERT FONDAMENTAL — EXPÉRIMENTAL, AUCUNE ACTIVATION</h2>'
        '<p><strong>'+html.escape(_decision_text(policy))+'</strong></p>'
        f'<p>Évaluation figée : {audit["evaluation_start_day"]} → {audit["evaluation_end_day"]}, '
        f'365 jours représentés, {audit["paired_hours"]} heures communes. Livraison {audit["live_day"]} '
        'visible ci-dessous mais exclue des Statistics, même lorsque son observation est déjà publiée.</p>'
        '<p><strong>Entrées de l’expert :</strong> seulement les fondamentaux explicitement préfixés '
        '<code>feature_fundamental_</code>. Aucun prix NYX, quantile NYX, Storm ou prix électrique passé '
        'n’est une variable explicative du classificateur ou du correcteur. Les prix NYX et observés interviennent '
        'néanmoins dans la cible supervisée d’erreur historique, l’évaluation et l’application de la correction.</p>'
        '<p><strong>Détection :</strong> erreur observé − NYX ≥ max(50 EUR/MWh, quantile 95 % des erreurs du bloc '
        'd’entraînement core par pays). Ce bloc est strictement antérieur et exclut les 28 jours de calibration. '
        'L’action exige p strictement supérieure à la prévalence réelle de cet événement dans le core du pays, '
        'ainsi qu’une porte physique. Ce n’est pas un seuil de probabilité universel. La porte physique utilise '
        'le quantile 90 % core : rampe locale FR/DE, indicateurs des pays voisins pour BE/NL.</p>'
        '<p><strong>Amplitude :</strong> HGB prédit la médiane des erreurs signées sur TOUTES les observations du core, '
        'pas uniquement les dépassements positifs. Seule sa partie positive, plafonnée à 400 EUR/MWh, est proposée. '
        'La correction n’est pas le produit p × sévérité de queue. Le poids retenu détermine la correction appliquée à NYX.</p>'
        '<p><strong>Limites :</strong> année déjà examinée ; entraînement progressif 90–365 jours ; proxys incomplets '
        'd’offre et de disponibilité, révisions historiques imparfaitement tracées, couverture de température lacunaire '
        'après le 4 septembre, réseau JAO exclu. Les requêtes historiques as-of ne certifient pas la publication '
        'effective des données à 08 h. Aucune preuve prospective ni garantie de gain n’est créée par ce rapport.</p>'
        '<p>Les P10/P50/P90 sauvegardés sont conservés ; les déciles intermédiaires et le CRPS sont interpolés uniquement '
        'pour l’affichage du moteur historique. Les cartes utilisent des moyennes de moyennes journalières ; le '
        'comparatif annuel utilise des moyennes horaires communes. Aucune attribution causale du prix n’est fabriquée.</p>'
        '<details><summary>Livraison — mécanisme de décision et prix horaires, hors scores historiques</summary>'
        '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _, label in columns)
        +'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></details></section>')


def _index_comparison(comparison: dict | None) -> str:
    if not comparison or not comparison.get("by_zone"):
        return '<p>Comparatif annuel non fourni ; aucun score ou classement inventé.</p>'
    labels = {"nuclear_kalman": "NYX nucléaire + Kalman figé", "storm": "Storm figé",
        "regional_25": "Expert régional précédent — 25 %", "fundamental": "Fondamental gouverné — principal",
        "fundamental_25": "Fondamental 25 % — diagnostic sans gouverneur strict"}
    output = []
    for zone, group in sorted(comparison["by_zone"].items()):
        annual, interventions = group.get("annual", {}), group.get("interventions", {})
        rows = []
        for model, label in labels.items():
            if model == "fundamental_25" and model not in annual:
                continue
            score, impact = annual.get(model, {}), interventions.get(model, {})
            cells = [html.escape(label), _number(score.get("hours"), 0)]
            cells.extend(_number(score.get(key)) for key in ("mae_eur_mwh", "rmse_eur_mwh", "bias_eur_mwh",
                "daily_mean_mae_eur_mwh", "mean_forecast_eur_mwh", "mean_observed_eur_mwh"))
            cells.extend(_number(impact.get(key), 0) for key in ("active_hours", "worsened_absolute_error_hours"))
            rows.append('<tr>'+''.join('<td>'+value+'</td>' for value in cells)+'</tr>')
        headings = ("Modèle", "Heures communes", "MAE", "RMSE", "Biais", "MAE prix moyens journaliers",
                    "Prix moyen prévu", "Prix moyen observé", "Corrections actives", "Corrections aggravantes")
        output.append('<section><h2>'+html.escape(str(zone))+'</h2><div class="table-wrap"><table><thead><tr>'
            +''.join('<th>'+html.escape(title)+'</th>' for title in headings)+'</tr></thead><tbody>'+''.join(rows)
            +'</tbody></table></div></section>')
    return ('<h2>Année commune — comparaison sans promotion</h2><p>Prix et erreurs en EUR/MWh, sur les mêmes heures '
        'disponibles. Les retours à NYX restent inclus. Les corrections aggravantes augmentent l’erreur absolue ; '
        'elles ne sont pas assimilées aux faux positifs du classificateur.</p>'+''.join(output))


def render_fundamental_reports(predictions: pd.DataFrame, *, source_audit: dict, output_directory: Path,
        model_name: str = "fundamental_stress", comparison: dict | None = None,
        decision_policy: str = "governed") -> dict[str, Path]:
    """Render isolated fundamental reports; no source refresh, fit or activation."""
    policy = _policy(decision_policy)
    if not isinstance(model_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", model_name):
        raise FundamentalReportError("A safe model identifier is required.")
    directory = Path(output_directory).expanduser().resolve()
    if not directory.is_relative_to(_NAMESPACE.resolve()) or directory == _NAMESPACE.resolve():
        raise FundamentalReportError("Reports must remain in a dedicated folder under the fundamental experiment namespace.")
    frame = _prepare(predictions)
    feature_columns = [column for column in frame if column.startswith(_PREFIX)]
    if not feature_columns:
        raise FundamentalReportError("No declared fundamental input feature is available for this report.")
    # Remove other models' features from all renderer input/manifest construction.
    frame = frame.drop(columns=[column for column in frame if column.startswith("feature_") and column not in feature_columns])
    _validate_decisions(frame, policy)
    label = "NYX + expert fondamental gouverné" if policy == "governed" else "NYX + expert fondamental — diagnostic 25 %"
    prepared = {}
    for zone, group in frame.groupby("zone", sort=True):
        result, audit = _result(group.reset_index(drop=True), source_audit=source_audit, directory=directory, label=label)
        result.zone_data.input_manifest["source"] = "fondamentaux du snapshot expérimental figé"
        result.zone_data.input_manifest["role"] = "entrée fondamentale de l’expert ; jamais un poids d’attribution du prix"
        result.zone_data.diagnostics.update(fundamental_only_display=True, feature_prefix=_PREFIX)
        audit.update(expert="fundamental_stress", decision_policy=policy, strict_governor_enforced=policy == "governed",
            allowed_input_features=feature_columns, fundamental_only_display=True,
            feature_price_inputs_excluded_by_contract=True, historical_price_error_used_as_supervised_target=True,
            decision_checks=_validate_decisions(group, policy), attribution_available=False)
        prepared[zone] = result, audit
    if not prepared:
        raise FundamentalReportError("No predictions to render.")
    directory.mkdir(parents=True, exist_ok=True)
    paths, audits = {}, {}
    for zone, (result, audit) in prepared.items():
        path = directory/f"forecast_{zone.lower()}_{audit['live_day']}_{model_name}.html"
        write_html_report([result], {"report": {"title": f"{zone} — {label} — EXPÉRIMENTAL", "forecast_history_hours": 168}}, path)
        _replace_report_labels(path, native_label=label, baseline_label="NYX nucléaire + Kalman figé")
        document = path.read_text(encoding="utf-8")
        for old, new in (("Prévision opérationnelle du", "Prévision expérimentale du"),
            ("prévision Day-Ahead opérationnelle", "prévision Day-Ahead expérimentale"),
            ("Prévision Day-Ahead réelle", "Prévision Day-Ahead expérimentale — live hors scores"),
            ("covariables actives", "séries fondamentales affichées (sélection illustrative)")):
            document = document.replace(old, new)
        document = document.replace("<main>", "<main>"+_banner(audit, frame.loc[frame.zone.eq(zone) & frame["sample"].eq("live")], policy), 1)
        document = document.replace("</main>", '<section><h2>Audit fondamental</h2><p><a href="fundamental_report_audit.json">Audit JSON</a> · '
            '<a href="comparison.json">Comparatif figé</a> · <a href="../fundamental_comparison.html">Comparatif interactif complet</a></p>'
            '<details><summary>Sources, masque figé et contrôles</summary><pre>'
            +html.escape(json.dumps(_clean(audit), ensure_ascii=False, indent=2, allow_nan=False))+'</pre></details></section></main>', 1)
        path.write_text(document, encoding="utf-8"); paths[zone] = path; audits[zone] = audit
    audit_path = directory/"fundamental_report_audit.json"
    audit_path.write_text(json.dumps(_clean({"reports": audits, "source_audit": source_audit}), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    comparison_path = directory/"comparison.json"
    comparison_path.write_text(json.dumps(_clean(comparison or {"status": "not_provided"}), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    links = ''.join('<li><a href="'+html.escape(path.name)+'">'+zone+' — '+html.escape(label)+'</a></li>' for zone, path in paths.items())
    index = directory/"index.html"
    index.write_text('<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Expert fondamental — expérimental</title><style>body{font:15px system-ui;background:#f4f6f8;color:#18212b;max-width:1500px;margin:30px auto;padding:0 20px}'
        'section{background:white;padding:20px;margin:20px 0;border-radius:10px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%}'
        'th,td{padding:10px;text-align:right;border-bottom:1px solid #dfe5ea}th:first-child,td:first-child{text-align:left}th{background:#edf2f7}a{color:#185995}</style></head><body>'
        '<h1>Expert fondamental — EXPÉRIMENTAL, AUCUNE ACTIVATION</h1><p>'+html.escape(_decision_text(policy))+'</p>'
        '<p>Live exclu des Statistics figées ; aucun prix explicatif dans l’expert, mais erreurs de prix historiques utilisées comme cibles supervisées.</p>'
        '<p><a href="../fundamental_comparison.html">Comparatif interactif complet : fondamentaux, calendrier témoin et versions de décision</a></p><ul>'
        +links+'</ul>'+_index_comparison(comparison)+'</body></html>', encoding="utf-8")
    return {**paths, "index": index, "audit": audit_path, "comparison": comparison_path}


__all__ = ["FundamentalReportError", "render_fundamental_reports"]
