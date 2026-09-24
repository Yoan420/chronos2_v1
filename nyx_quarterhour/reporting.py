"""Self-contained French result page for the matched resolution experiment."""
import html
import json
from pathlib import Path

import pandas as pd


NAMES = {"nyx":"NYX complet", "hourly_control":"Chronos-2 horaire",
         "native_quarterhour":"Chronos-2 à 15 min → moyenne horaire"}


def render_report(directory: Path, summary: dict, tables: dict[str,pd.DataFrame]) -> Path:
    state = summary.get("status","unknown")
    decision = summary.get("decision",{})
    if state != "complete":
        message = "Comparaison non validée : "+str(summary.get("reason") or state)
        tables = {}
    elif decision.get("encouraging"):
        message = "Les critères de poursuite sont satisfaits sur ce test exploratoire. Une validation prospective reste nécessaire."
    else:
        message = "Les critères fixés avant le calcul ne permettent pas de retenir ce candidat."
    cards = ""
    metrics = tables.get("metrics",pd.DataFrame())
    if not metrics.empty:
        shown = metrics.loc[metrics["group"].isin(["overall","country"]),["family","value","n","mae","rmse","bias"]].copy()
        shown["family"] = shown.family.map(lambda v:NAMES.get(v,v))
        shown["value"] = shown.value.replace({"all":"CWE"})
        shown.rename(columns={"family":"Modèle","value":"Périmètre","n":"Points","mae":"MAE €/MWh ↓","rmse":"RMSE €/MWh ↓","bias":"Biais €/MWh"},inplace=True)
        cards += '<section><h2>Résultats horaires appariés</h2><div class="table">'+shown.to_html(index=False,escape=True,float_format=lambda x:f"{x:.3f}")+'</div></section>'
    contrasts = tables.get("paired_deltas",pd.DataFrame())
    if not contrasts.empty:
        shown = contrasts.loc[contrasts.zone.eq("all"),["family","baseline","mae_delta","mae_delta_ci_low","mae_delta_ci_high","rmse_delta"]].copy()
        for name in ("family","baseline"):
            shown[name] = shown[name].map(lambda v:NAMES.get(v,v))
        shown.rename(columns={"family":"Modèle","baseline":"Comparé à","mae_delta":"Écart MAE","mae_delta_ci_low":"IC 95 % bas","mae_delta_ci_high":"IC 95 % haut","rmse_delta":"Écart RMSE"},inplace=True)
        cards += '<section><h2>Écarts et incertitude</h2><p>Un écart négatif favorise le candidat. Intervalles par blocs communs de sept jours.</p><div class="table">'+shown.to_html(index=False,escape=True,float_format=lambda x:f"{x:.3f}")+'</div></section>'
    regimes = decision.get("critical_regimes",{})
    if state=="complete" and regimes.get("coverage_complete") is False:
        cards += '<section><h2>Régimes critiques : preuve partielle</h2><p>Certains régimes ne réunissent pas assez d’heures et de journées. Aucun gain ne peut y être établi.</p></section>'
    cfg = summary.get("config",{})
    first,last = html.escape(str(cfg.get("start_day",""))),html.escape(str(cfg.get("end_day","")))
    raw = html.escape(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
    doc = f'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NYX — Test des prix à 15 minutes</title><style>
:root{{color-scheme:dark;font-family:Segoe UI,Arial,sans-serif;color:#e8edf7;background:#080d16}}body{{margin:0;background:radial-gradient(ellipse at 8% 0%,#16324088,transparent 55%),radial-gradient(ellipse at 95% 35%,#3f203650,transparent 50%),#080d16}}main{{max-width:1220px;margin:auto;padding:40px 24px 70px}}.brand{{letter-spacing:.22em;color:#6fe5e8;font-weight:700}}h1{{font-size:clamp(28px,4vw,42px)}}h2{{font-size:20px}}p{{color:#bdcbda;line-height:1.65}}.badge{{color:#f1c486;border:1px solid #c9965966;border-radius:6px;padding:8px 12px;display:inline-block}}section{{background:#111b29bb;border:1px solid #29394c;border-radius:10px;padding:22px;margin-top:24px}}.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:right;white-space:nowrap;padding:10px;border-bottom:1px solid #29394c}}th{{color:#75d5e1}}th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;word-break:break-word;color:#a7b9cb;font-size:12px}}details{{margin-top:25px}}footer{{margin-top:25px;color:#8293a8;font-size:13px}}</style></head>
<body><main><div class="brand">NYX / RECHERCHE</div><h1>Prévoir à 15 minutes, évaluer à l’heure</h1>
<span class="badge">{html.escape(state)}</span><p>{html.escape(message)}</p>
<section><h2>Le test</h2><p>Du <strong>{first}</strong> au <strong>{last}</strong>, Belgique, Allemagne, France et Pays-Bas. Même Chronos-2 gelé, mêmes fondamentaux prévus, même durée de contexte : 2 048 heures ou 8 192 quarts d’heure. Les quatre prévisions ponctuelles sont moyennées pour chaque heure physique.</p><p>Le contrôle horaire et le candidat à 15 minutes utilisent Chronos-2 sans correcteur CatBoost ni Kalman. NYX complet demeure une référence séparée. Ce test isole le changement de résolution ; il ne reproduit pas toute la chaîne NYX à 15 minutes.</p></section>
{cards}
<section><h2>Portée des résultats</h2><p>Prix historiques natifs à 15 minutes, observations rétrospectives. Les fondamentaux horaires sont constants sur leurs quatre quarts ; aucune finesse supplémentaire ne leur est attribuée. L’heure de publication historique de chaque observation n’est pas certifiée.</p><p>Les prix day-ahead de D−1 sont admis jusqu’à la fin de D−1 dans les deux contextes. Aucun prix réalisé de D n’entre dans la prévision. L’agrégation de quatre points ne crée pas de nouveaux quantiles horaires calibrés.</p><p>La période a déjà été examinée dans les recherches précédentes. Les résultats restent exploratoires et ne déclenchent aucune activation automatique.</p></section>
<details><summary>Protocole, sources et décision détaillée</summary><pre>{raw}</pre></details>
<footer>Expérience locale isolée · Aucun modèle activé en production</footer></main></body></html>'''
    path = directory/"report.html"
    path.write_text(doc,encoding="utf-8")
    return path
