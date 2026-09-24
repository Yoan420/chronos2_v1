"""A local, self-contained result page for the hourly intrahour experiment."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def render_report(directory: Path, summary: dict, tables: dict[str, pd.DataFrame]) -> Path:
    state = summary.get("status", "unknown")
    labels = {"complete": "Comparaison terminée", "prepared": "Données préparées",
              "data_unavailable": "Données à 15 minutes indisponibles",
              "insufficient_data": "Historique insuffisant", "failed": "Vérification interrompue"}
    decision = summary.get("decision", {})
    encouraging = decision.get("encouraging", False) if isinstance(decision, dict) else False
    message = summary.get("reason") or (
        "Les critères de poursuite sont satisfaits sur ce test rétrospectif. Une validation prospective reste nécessaire."
        if encouraging else "Aucune amélioration suffisante n'est établie pour passer à un modèle prédisant les quarts d'heure.")
    if state == "prepared":
        message = "Les profils ont été contrôlés. Aucun modèle n'a été ajusté dans cette étape."
    warnings = ""
    regimes = decision.get("critical_regimes", {}) if isinstance(decision, dict) else {}
    if state == "complete" and regimes.get("coverage_complete") is False:
        warnings = ('<section><h2>Preuve partielle sur les situations extrêmes</h2>'
                    '<p>Certains régimes de prix négatifs ou élevés ne comptent pas assez de journées. '
                    'Même un gain global ne permettrait pas de conclure à une amélioration dans ces situations. '
                    'Les détails figurent dans regime_checks.csv.</p></section>')
    cards = ""
    if state == "failed":
        tables = {}
    for name, label in [("metrics", "Qualité des prévisions"), ("paired_deltas", "Écarts appariés et incertitude")]:
        frame = tables.get(name, pd.DataFrame())
        if frame.empty:
            continue
        # Show the reserved test first; detailed CSV/parquet retain all groups.
        shown = frame.copy()
        if "stage" in shown and "test" in set(shown.stage):
            shown = shown.loc[shown.stage.eq("test")]
        for col in ("group", "grouping"):
            if col in shown and shown[col].isin(["all", "overall", "pooled", "country", "zone"]).any():
                shown = shown.loc[shown[col].isin(["all", "overall", "pooled", "country", "zone"])]
                break
        family_labels = {"nyx": "NYX actuel", "hourly_control": "Contrôle : moyenne horaire", "intrahour": "Profils à 15 minutes"}
        for column in ("family", "baseline"):
            if column in shown:
                shown[column] = shown[column].map(lambda value: family_labels.get(value, value))
        if name == "metrics":
            columns = {"family": "Variante", "value": "Périmètre", "n": "Points", "mae": "MAE €/MWh", "rmse": "RMSE €/MWh", "bias": "Biais €/MWh"}
            if "value" in shown:
                shown["value"] = shown["value"].replace({"all": "CWE"})
        else:
            columns = {"family": "Variante", "baseline": "Comparée à", "zone": "Pays", "mae_delta": "Écart MAE", "mae_delta_ci_low": "IC 95 % bas", "mae_delta_ci_high": "IC 95 % haut", "rmse_delta": "Écart RMSE"}
            if "zone" in shown:
                shown["zone"] = shown["zone"].replace({"all": "CWE"})
        shown = shown[[c for c in columns if c in shown]].rename(columns=columns)
        cards += f'<section><h2>{label}</h2><div class="table">{shown.head(40).to_html(index=False, escape=True, float_format=lambda v: f"{v:.3f}")}</div><p>Tableaux complets disponibles dans les fichiers de résultats du dossier.</p></section>'
    coverage = summary.get("feature_audit", {})
    if coverage:
        cards = (f'<section><h2>Couverture des profils</h2><p><strong>{coverage.get("complete_hours", 0):,}</strong> heures complètes sur '
                 f'<strong>{coverage.get("expected_hours", 0):,}</strong>. Les données manquantes ne sont pas interpolées.</p></section>')+cards
    selection = summary.get("selection", {})
    if selection.get("status") == "locked":
        first = html.escape(str(selection.get("test_start", "")))
        last = html.escape(str(selection.get("test_end", "")))
        cards = (f'<section><h2>Période évaluée</h2><p>Test du <strong>{first}</strong> au <strong>{last}</strong>, '
                 'sur la Belgique, l’Allemagne, la France et les Pays-Bas. '
                 'Les paramètres sont sélectionnés sur la période antérieure. Cet historique a déjà été '
                 'étudié dans un audit précédent : les résultats restent exploratoires.</p></section>')+cards
    sources = summary.get("native_source_audit", {}).get("sources", [])
    source_note = ""
    if sources:
        names = ", ".join(html.escape(str(s.get("alias", "source inconnue"))) for s in sources)
        source_note = f'<p>Sources à 15 minutes utilisées : <strong>{names}</strong>.</p>'
        if {s.get("alias") for s in sources} == {"be_solar_elia_fcst"}:
            source_note += ('<p>Ce premier pilote utilise uniquement la prévision solaire belge Elia, '
                            'comme information commune aux quatre pays. Il ne mesure pas encore '
                            'l’apport de tous les fondamentaux à 15 minutes.</p>')
    if summary.get("native_source_audit", {}).get("temporal_evidence") == "retrospective_asof":
        source_note += ('<p>Les profils ont été interrogés avec une date historique D−1 à 08:00 Paris. '
                        'L’heure de publication originale du fournisseur n’est pas certifiée par cette archive.</p>')
    payload = html.escape(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    document = f'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NYX — Profils intra-horaires</title><style>
:root{{color-scheme:dark;font-family:Segoe UI,Arial,sans-serif;background:#080d16;color:#e8edf7}}body{{margin:0;background:radial-gradient(ellipse at 10% 0%,#13313e66,transparent 55%),radial-gradient(ellipse at 95% 40%,#39162d55,transparent 50%),#080d16}}main{{max-width:1200px;margin:auto;padding:42px 24px 70px}}.brand{{letter-spacing:.23em;color:#6fe5e8;font-weight:700}}h1{{font-size:clamp(28px,4vw,44px);margin-bottom:12px}}h2{{font-size:20px}}p{{line-height:1.65;color:#bac9dc}}.badge{{display:inline-block;border:1px solid #d6a56166;color:#f1c486;padding:8px 13px;border-radius:6px}}section{{margin-top:24px;padding:22px;background:#111b29cc;border:1px solid #29394c;border-radius:10px}}.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:right;padding:10px;border-bottom:1px solid #29394c;white-space:nowrap}}th{{color:#75d5e1}}th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;word-break:break-word;color:#a7b9cb;font-size:12px}}a{{color:#76dbe1}}strong{{color:#e8edf7}}details{{margin-top:24px}}footer{{margin-top:28px;color:#8394a8;font-size:13px}}</style></head><body><main>
<div class="brand">NYX / RECHERCHE</div><h1>Prévisions horaires, profils à 15 minutes</h1>
<span class="badge">{html.escape(labels.get(state,state))}</span><p>{html.escape(str(message))}</p>
<section><h2>Ce que mesure cette variante</h2><p>NYX complet reste la référence. Le contrôle ajoute uniquement les moyennes horaires des nouveaux fondamentaux. La variante intra-horaire ajoute leurs pentes, amplitudes et dispersions à 15 minutes, sélectionnées au cutoff de prévision. Les trois sorties restent horaires.</p><p>Les prix et fondamentaux futurs réalisés ne sont jamais des entrées. Ce test produit une prévision ponctuelle : il ne crée pas de nouveaux intervalles P10–P90.</p>{source_note}</section>
{warnings}{cards}<details><summary>Protocole, provenance et décision détaillée</summary><pre>{payload}</pre></details>
<footer>Expérience isolée · Aucun modèle activé en production · Les résultats rétrospectifs ne prouvent pas un gain prospectif.</footer>
</main></body></html>'''
    path = directory/"report.html"
    path.write_text(document, encoding="utf-8")
    return path
