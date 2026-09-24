"""Standalone result report for the entire native-frequency architecture."""
import html
import json
from pathlib import Path

import pandas as pd

NAMES = {"nyx": "NYX actuel · historique complet", "matched_full_hourly": "NYX horaire · historique commun",
         "full_quarterhour": "NYX complet à 15 min → heure", "raw_hourly": "Chronos horaire seul",
         "raw_quarterhour": "Chronos à 15 min seul → heure", "residual_hourly": "Chronos + CatBoost horaire",
         "residual_quarterhour": "Chronos + CatBoost à 15 min → heure"}


def render_report(directory: Path, summary: dict, tables: dict[str, pd.DataFrame]) -> Path:
    state = summary.get("status", "unknown")
    decision = summary.get("decision", {})
    if state != "complete":
        message = "Comparaison incomplète : " + str(summary.get("reason") or state)
        tables = {}
    elif decision.get("encouraging"):
        message = "Les critères de poursuite sont satisfaits sur cet essai exploratoire. Une validation prospective reste nécessaire."
    else:
        message = "Les critères fixés avant le calcul ne sont pas tous satisfaits. Aucun nouveau modèle n’est activé."
    sections = ""
    regimes = decision.get("critical_regimes", {})
    if state == "complete" and regimes.get("coverage_complete") is False:
        sections += '<section><h2>Régimes extrêmes : couverture partielle</h2><p>Certains régimes ne comptent pas assez d’heures ou de journées. Une conclusion favorable globale ne constitue pas une preuve dans ces situations ; la couverture détaillée figure dans le protocole ci-dessous.</p></section>'
    metrics = tables.get("metrics", pd.DataFrame())
    if not metrics.empty:
        selected = metrics.loc[metrics["group"].isin(["overall", "country"]), ["family", "value", "n", "mae", "rmse", "bias"]].copy()
        selected["family"] = selected.family.map(lambda name: NAMES.get(name, name))
        selected["value"] = selected.value.replace({"all": "CWE"})
        selected.rename(columns={"family": "Modèle", "value": "Périmètre", "n": "Points", "mae": "MAE €/MWh ↓", "rmse": "RMSE €/MWh ↓", "bias": "Biais €/MWh"}, inplace=True)
        sections += '<section><h2>Chaîne complète et contribution des étapes</h2><div class="table">' + selected.to_html(index=False, escape=True, float_format=lambda value: f"{value:.3f}") + '</div></section>'
    paired = tables.get("paired_deltas", pd.DataFrame())
    if not paired.empty:
        selected = paired.loc[paired.zone.eq("all") & paired.family.eq("full_quarterhour"), ["baseline", "mae_delta", "mae_delta_ci_low", "mae_delta_ci_high", "rmse_delta"]].copy()
        selected["baseline"] = selected.baseline.map(lambda name: NAMES.get(name, name))
        selected.rename(columns={"baseline": "Comparaison du NYX à 15 minutes avec", "mae_delta": "Écart MAE", "mae_delta_ci_low": "IC 95 % bas", "mae_delta_ci_high": "IC 95 % haut", "rmse_delta": "Écart RMSE"}, inplace=True)
        sections += '<section><h2>Écarts appariés et incertitude</h2><p>Un écart négatif favorise la chaîne à 15 minutes. Bootstrap de blocs communs de sept jours, 1 000 répétitions.</p><div class="table">' + selected.to_html(index=False, escape=True, float_format=lambda value: f"{value:.3f}") + '</div></section>'
    cfg = summary.get("config", {})
    raw = html.escape(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    doc = f'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NYX — Chaîne complète à 15 minutes</title><style>
:root{{color-scheme:dark;font-family:Segoe UI,Arial,sans-serif;color:#e8edf7;background:#080d16}}body{{margin:0;background:radial-gradient(ellipse at 8% 0%,#16324088,transparent 55%),radial-gradient(ellipse at 95% 35%,#3f203650,transparent 50%),#080d16}}main{{max-width:1280px;margin:auto;padding:40px 24px 70px}}.brand{{letter-spacing:.22em;color:#6fe5e8;font-weight:700}}h1{{font-size:clamp(28px,4vw,42px)}}h2{{font-size:20px}}p{{color:#bdcbda;line-height:1.65}}.badge{{color:#f1c486;border:1px solid #c9965966;border-radius:6px;padding:8px 12px;display:inline-block}}section{{background:#111b29bb;border:1px solid #29394c;border-radius:10px;padding:22px;margin-top:24px}}.flow{{color:#83e8e7;font-size:19px;font-weight:600;line-height:1.9}}.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:right;white-space:nowrap;padding:10px;border-bottom:1px solid #29394c}}th{{color:#75d5e1}}th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;word-break:break-word;color:#a7b9cb;font-size:12px}}details{{margin-top:25px}}footer{{margin-top:25px;color:#8293a8;font-size:13px}}</style></head><body><main>
<div class="brand">NYX / RECHERCHE</div><h1>Toute la chaîne au quart d’heure</h1><span class="badge">{html.escape(state)}</span><p>{html.escape(message)}</p>
<section><div class="flow">Chronos-2 natif 15 min → CatBoost quotidien → Kalman gouverné quotidien → moyenne des quatre points par heure</div><p>Évaluation du <strong>{html.escape(str(cfg.get('start_day', '')))}</strong> au <strong>{html.escape(str(cfg.get('end_day', '')))}</strong>, sur BE, DE, FR et NL. Les trois quantiles Chronos sont recalculés pour l’apprentissage et le test. CatBoost apprend les erreurs à 15 minutes ; les cinq candidats Kalman assimilent les résidus à 15 minutes après gel des prévisions du jour.</p></section>
<section><h2>Une comparaison à historique égal</h2><p>Les prix natifs disponibles commencent le 1er octobre 2025. Après 2 048 heures de contexte, le rejeu commence le 26 décembre 2025. Le premier jour de test dispose de 174 journées de résidus antérieurs, puis cette durée augmente. Les deux chaînes expérimentales utilisent exactement les mêmes journées, avec un plafond de 365 jours ; elles ne disposent pas de 365 jours complets sur cet essai.</p><p>NYX actuel conserve sa prévision archivée et son historique plus long. C’est la référence opérationnelle distincte. CatBoost et Kalman sont réentraînés au quart d’heure : aucun correcteur horaire n’est simplement appliqué aux nouveaux points.</p></section>
{sections}
<section><h2>Protocole et limites</h2><p>Même Chronos-2 gelé, contexte de 2 048 heures (8 192 quarts), six fondamentaux archivés, entrées historiques et futures de NYX. Les fondamentaux restent horaires et constants sur leurs quatre quarts. Calendriers fractionnaires, changements d’heure et rampes physiques de une et deux heures sont conservés.</p><p>CatBoost : recette MAE de NYX, 700 arbres, profondeur 6, apprentissage 0,03, correction plafonnée à ±40 €/MWh. Kalman : cinq candidats, gouvernance sur 60 jours, correction plafonnée à ±20 €/MWh ; transition une fois par jour, sans division artificielle du bruit quotidien par quatre.</p><p>Seuls les jours précédant D servent à apprendre les corrections de D. Les prix day-ahead de D−1 sont admis selon la convention de NYX. Les observations historiques sont rétrospectives, sans preuve de chaque vintage publié au cutoff. Le recalibrage de Kalman suit la production ; son historique de gouvernance n’est pas une validation imbriquée complète.</p><p>Le test est exploratoire : la période a déjà été consultée. La moyenne des quatre prévisions ponctuelles est évaluée à l’heure ; elle ne prouve pas la calibration d’intervalles horaires. Aucun réglage n’est choisi sur les erreurs de cette période et aucune promotion automatique n’est effectuée.</p></section>
<details><summary>Traçabilité et décision détaillée</summary><pre>{raw}</pre></details><footer>Expérience locale isolée · Modèle opérationnel inchangé</footer></main></body></html>'''
    path = directory / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path
