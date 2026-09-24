"""Offline V2 qualification report; raw diagnostics never labelled forecasts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=7))


def _table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="statistics", na_rep="—", float_format=lambda x: f"{x:.3f}")


def _metrics(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>Aucun échantillon qualifié disponible.</p>"
    labels = {"base": "Modèle actuel figé", "guarded": "Référence conservée (abstention)",
              "storm": "Storm", "raw_zonal_diagnostic": "Simulation zonale NON QUALIFIÉE"}
    names = {"zone": "Pays", "model": "Série", "days": "Jours", "hours": "Heures",
             "mae": "MAE horaire", "rmse": "RMSE", "daily_mean_mae": "MAE prix moyen/jour", "bias": "Biais"}
    return _table(frame.replace({"model": labels})[list(names)].rename(columns=names))


def render_report(snapshot: Path, config: dict, audit: dict) -> Path:
    frame = pd.read_parquet(snapshot / "predictions.parquet")
    tz = config["evaluation"]["timezone"]
    frame["day"] = frame.timestamp.dt.tz_convert(tz).dt.strftime("%Y-%m-%d")
    frame["hour"] = frame.timestamp.dt.tz_convert(tz).dt.hour
    daily = pd.read_parquet(snapshot / "daily_metrics.parquet")
    diagnostics = pd.read_parquet(snapshot / "diagnostic_daily.parquet")
    candidates = pd.read_parquet(snapshot / "candidate_predictions.parquet")
    central = candidates.loc[candidates.candidate_id.eq("central")].copy()
    central["day"] = central.delivery_start_utc.dt.tz_convert(tz).dt.strftime("%Y-%m-%d")
    central = central.loc[central.day.between(audit["evaluation_start"], audit["evaluation_end"])]
    network = json.loads((snapshot / "network_audit.json").read_text(encoding="utf-8"))
    net = pd.DataFrame(network["daily"])
    net = net.loc[net.delivery_day.between(audit["evaluation_start"], audit["evaluation_end"])].copy()
    supply, deficits = [], []
    for zone, data in central.groupby("zone"):
        available = data.raw_price_eur_mwh.notna()
        missing = audit["supply"]["zones"][zone]["qualification"].get("omitted_segments", [])
        supply.append({"Pays": zone, "Heures attendues": len(data), "Données renseignées": int(available.sum()),
                       "Heures expert couplé qualifié": 0,
                       "Déficit pile représentée (%)": 100 * data.loc[available, "shortage_mw"].gt(1e-6).mean(),
                       "Filières omises / non attestées": ", ".join(missing)})
        for day, group in data.groupby("day"):
            ok = group.raw_price_eur_mwh.notna()
            deficits.append({"zone": zone, "day": day, "available_hours": int(ok.sum()),
                             "deficit_share": float(group.loc[ok, "shortage_mw"].gt(1e-6).mean()) if ok.any() else None})
    paired = frame.loc[np.isfinite(frame[["actual", "base", "storm"]]).all(axis=1)].copy()
    hourly = []
    for (zone, hour), data in paired.groupby(["zone", "hour"]):
        hourly.append({"zone": zone, "hour": int(hour), "base": float((data.base-data.actual).abs().mean()),
                       "storm": float((data.storm-data.actual).abs().mean()), "hours": len(data)})
    episode = frame.loc[frame.day.between("2026-06-24", "2026-06-26")]
    episode_rows = []
    for (zone, day), group in episode.groupby(["zone", "day"]):
        ok = group.diagnostic_expert.notna()
        episode_rows.append({"Pays": zone, "Jour": day, "MAE modèle actuel": (group.base-group.actual).abs().mean(),
                             "MAE simulation non qualifiée": (group.loc[ok, "diagnostic_expert"]-group.loc[ok, "actual"]).abs().mean(),
                             "Heures simulation": int(ok.sum()), "Interventions": int(group.weight.gt(0).sum())})
    fields = ["timestamp", "day", "zone", "actual", "base", "storm", "diagnostic_expert", "candidate_id"]
    payload = {"hourly": _records(frame[[key for key in fields if key in frame]]),
               "daily": _records(pd.concat([daily, diagnostics], ignore_index=True)),
               "network": _records(net), "deficits": deficits, "hourly_mae": hourly,
               "audit": audit, "network_summary": {key: value for key, value in network.items() if key != "daily"},
               "assumptions": config.get("assumptions", {})}
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str).replace("</", "<\\/")
    document = """<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coût marginal V2 — offre enrichie / cutoff 08 h</title><style>
:root{color-scheme:light;--bg:#f4f7fb;--panel:#fff;--ink:#172b44;--muted:#53657d;--line:#dce4ee;--warn:#fff4d7}body.night{color-scheme:dark;--bg:#0c1422;--panel:#142136;--ink:#e8eff9;--muted:#b7c8df;--line:#34455f;--warn:#44371a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;align-items:center;gap:20px}h1{font-size:27px;margin:8px 0}h2{font-size:20px}.tag{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:1px}section{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin:18px 0;padding:20px;overflow:auto}.warning{background:var(--warn)}button,select,input{font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}button{cursor:pointer}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.statistics{width:100%;border-collapse:collapse;font-size:13px}.statistics th,.statistics td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:right}.statistics td:nth-child(-n+2),.statistics th:nth-child(-n+2){text-align:left}.plot{min-height:360px}.small{color:var(--muted);font-size:13px}pre{white-space:pre-wrap;word-break:break-word;font-size:12px}a{color:#398bd9}</style>
<script>__PLOTLY__</script></head><body><main><header><div><div class="tag">Laboratoire isolé · production et Complete inchangés</div><h1>Coût marginal V2 — offre enrichie</h1><p>__START__ → __END__ · 365 jours évalués · cutoff strict D−1 à 08 h</p></div><button id="theme">Mode nuit</button></header>
<section class="warning"><strong>Offre améliorée, mais expert couplé encore non qualifié.</strong><p>Le gaz agrégé par combustible remplace la somme CCGT/GT. Le charbon allemand et néerlandais et le lignite allemand sont ajoutés avec leurs coûts distincts. Le parc fournisseur ne couvre toutefois pas toutes les capacités nationales, et les données JAO ne permettent pas encore de reconstituer un domaine complet et validé à 08 h.</p><p><strong>Aucune intervention sur le modèle actuel.</strong> Les prix de la simulation zonale sont uniquement des diagnostics d'une pile partielle sans échanges. La pénalité artificielle de déficit de 4 000 EUR/MWh n'est pas une prévision de pénurie réelle. L'absence de dégradation par abstention n'est pas un gain prédictif.</p></section>
<section><h2>Statistics — modèles conservés sur 365 jours</h2><p class="small">Référence et abstention conservent toutes les heures. Storm n'est pas interpolé ; son heure DST manquante reste absente. Les observations et prévisions sont exactement celles du snapshot V1, figées par pays.</p>__METRICS__</section>
<section><h2>Qualification de l'offre — 365 jours évalués</h2><p class="small">« Données renseignées » signifie seulement que les séries déclarées sont disponibles. Ce n'est pas une attestation de parc national complet. Une disponibilité nulle publiée n'est pas une donnée manquante. Le nucléaire FR est une prévision de production, contrairement au Pmax de disponibilité BE/NL ; ces blocs ne sont pas des offres réelles identifiées. L'hydraulique flexible et le stockage ne sont pas inventés à partir de puissances seules.</p>__SUPPLY__</section>
<section><h2>Diagnostic zonal — comparaison sur les mêmes heures</h2><p class="small">Trois hypothèses de rendement sont départagées chaque jour sur les 365 jours calendaires précédents. Une fenêtre incomplète entraîne une abstention, jamais une compression du calendrier. Le tableau ci-dessous retient uniquement les heures communes à la simulation, au modèle actuel et à Storm ; il ne constitue pas la validation du modèle couplé.</p>__PAIRED__</section>
<section><div class="controls"><label>Pays <select id="zone"></select></label><label>Journée <input id="day" type="date"></label><button id="june">24 juin 2026</button></div><p class="small">La simulation non qualifiée est masquée initialement pour garder les prix usuels lisibles. Cliquer sa légende pour afficher aussi ses valeurs extrêmes ; aucune de ces valeurs n'est supprimée des métriques.</p><div id="forecast" class="plot"></div><div id="daily" class="plot"></div><div id="deficit" class="plot"></div></section>
<section><h2>Épisode du 24–26 juin 2026</h2><p class="small">Épisode identifié a posteriori, pas un test final indépendant. Aucune sélection de scénario à partir du prix du jour et aucun remplacement des mauvaises simulations par la référence.</p>__EPISODE__</section>
<section><h2>Réseau JAO — disponibilité avant 08 h</h2><p id="networkSummary"></p><div id="network" class="plot"></div><p>Les contraintes originales sont conservées, y compris les frontières et les hubs virtuels. Des heures manquent réellement dans les réponses API. Les agrégats interpolés utilisés ailleurs dans le projet ne servent pas de réseau physique.</p><p><strong>Point bloquant :</strong> le RAM initial n'est pas assimilé automatiquement à une contrainte PTDF × position nette ≤ RAM. Sa référence et les positions de toutes les frontières doivent être validées. La publication D2CF/RefProg annoncée à 10 h 30 est exclue du test à 08 h. Aucun réseau libre, aucune position externe implicitement nulle, aucun résultat post-couplage utilisé en entrée.</p></section>
<section><h2>Erreur moyenne selon l'heure civile — modèle actuel / Storm</h2><p class="small">MAE horaire appariée : mêmes dates/heures avec observation pour les deux courbes ; les deux heures physiques du changement d'heure d'automne restent deux observations.</p><div id="hours" class="plot"></div></section>
<section><h2>Note méthodologique</h2><ol><li>La demande résiduelle déjà nette du vent et du solaire n'est pas déduite une seconde fois. Les prévisions et états as-of sont limités à D−1 08 h.</li><li>Coût thermique = (combustible + CO₂ × facteur d'émission) / rendement + coût variable. API2 USD/t est converti par le dernier EUR/USD connu et 6,978 MWh thermiques/t. Le lignite utilise une hypothèse indépendante de 2,3 EUR/MWh thermique.</li><li>L'offre gaz est répartie hypothétiquement en 85 % de blocs efficaces / 15 % de pointe. Ce ne sont pas des centrales individuellement identifiées. Aucun cumul avec les capacités déjà incluses dans l'agrégat gaz.</li><li>Les simulations physiques ne reçoivent aucun prix électrique passé ni prévision Chronos/Storm. Seul le choix du scénario utilise les prix historiques de calibration. Les labels canoniques gelés peuvent contenir des révisions postérieures : ce backtest reste diagnostique.</li><li>Le domaine réseau non qualifié arrête le calcul couplé avant toute intervention. Aucun entraînement de poids ne peut rendre valide un réseau mal défini.</li><li>La prochaine validation exige un parc suffisamment complet, une référence RAM interprétable avant 08 h et toutes les frontières. Elle devra ensuite améliorer les épisodes tendus sans dégrader les scores annuels sur les mêmes heures.</li></ol><p><a href="https://publicationtool.jao.eu/PublicationHandbook/Core_PublicationTool_Handbook_v2.2.pdf">JAO — calendrier et champs publiés</a> · <a href="https://eepublicdownloads.entsoe.eu/clean-documents/nc-tasks/Core%20DA%20CCM%203rd%20RfA%20-%20Clean%20version.pdf">Core CCM — référence des flux et RAM</a> · <a href="https://www.ise.fraunhofer.de/content/dam/ise/en/documents/publications/studies/EN2024_ISE_Study_Levelized_Cost_of_Electricity_Renewable_Energy_Technologies.pdf">Fraunhofer ISE 2024 — hypothèses thermiques</a></p><details><summary>Audits et hypothèses reproductibles</summary><pre id="audit"></pre></details></section>
</main><script>
const payload=__PAYLOAD__, colors={base:'#927bdd',storm:'#5a9dea',actual:'#e56472',diagnostic_expert:'#dfa02d'}, names={base:'Modèle actuel',storm:'Storm',actual:'Observé',diagnostic_expert:'Simulation non qualifiée'};
const zone=document.getElementById('zone'),day=document.getElementById('day'),days=[...new Set(payload.hourly.map(r=>r.day))].sort();
[...new Set(payload.hourly.map(r=>r.zone))].sort().forEach(z=>zone.add(new Option(z,z)));day.min=days[0];day.max=days.at(-1);day.value=days.at(-1);
document.getElementById('audit').textContent=JSON.stringify({audit:payload.audit,network:payload.network_summary,assumptions:payload.assumptions},null,2);
document.getElementById('networkSummary').textContent=`Sur les 730 jours de support : ${payload.network_summary.complete_research_days} jours de données brutes complets. Cela ne qualifie ni la référence du domaine ni les frontières ; 0 heure de prix couplé autorisée.`;
function layout(title,y='EUR/MWh'){let dark=document.body.classList.contains('night');return{title:{text:title},paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:dark?'#e8eff9':'#172b44'},margin:{t:55,r:25,b:55,l:75},xaxis:{gridcolor:dark?'#34455f':'#dce4ee'},yaxis:{title:y,gridcolor:dark?'#34455f':'#dce4ee'},legend:{orientation:'h'},hovermode:'x unified'}}
function draw(){const rows=payload.hourly.filter(r=>r.zone===zone.value&&r.day===day.value);Plotly.react('forecast',['actual','base','storm','diagnostic_expert'].map(k=>({x:rows.map(r=>r.timestamp),y:rows.map(r=>r[k]),type:'scatter',mode:'lines',connectgaps:false,name:names[k],visible:k==='diagnostic_expert'?'legendonly':true,line:{color:colors[k],width:k==='actual'?3:2}})),layout(`${zone.value} — ${day.value} (axe UTC)`),{responsive:true});
Plotly.react('daily',['base','storm','raw_zonal_diagnostic'].map(k=>{let d=payload.daily.filter(r=>r.zone===zone.value&&r.model===k);return{x:d.map(r=>r.day),y:d.map(r=>r.mae),type:'scatter',mode:'lines',connectgaps:false,visible:k==='raw_zonal_diagnostic'?'legendonly':true,name:k==='raw_zonal_diagnostic'?names.diagnostic_expert:names[k],line:{color:k==='raw_zonal_diagnostic'?colors.diagnostic_expert:colors[k]}}}),layout('MAE quotidienne'),{responsive:true});
let d=payload.deficits.filter(r=>r.zone===zone.value);Plotly.react('deficit',[{x:d.map(r=>r.day),y:d.map(r=>r.deficit_share===null?null:100*r.deficit_share),type:'bar',name:'Déficit simulé',marker:{color:'#dfa02d'}}],layout('Déficit de la pile partielle centrale, pas du système réel','% des heures renseignées'),{responsive:true});
const n=payload.network;Plotly.react('network',[{x:n.map(r=>r.delivery_day),y:n.map(r=>r.available_hours||0),customdata:n.map(r=>r.expected_hours),type:'bar',name:'Heures brutes qualifiées',marker:{color:n.map(r=>r.inputs_qualified?'#23a68b':'#dfa02d')},hovertemplate:'%{x}<br>%{y}/%{customdata} heures<extra></extra>'}],layout('Données originales JAO — 365 jours évalués','Heures / jour'),{responsive:true});
const h=payload.hourly_mae.filter(r=>r.zone===zone.value);Plotly.react('hours',['base','storm'].map(k=>({x:h.map(r=>r.hour),y:h.map(r=>r[k]),type:'scatter',mode:'lines+markers',name:names[k],line:{color:colors[k]}})),layout('MAE par heure civile (échantillon apparié)'),{responsive:true});}
zone.onchange=draw;day.onchange=draw;document.getElementById('june').onclick=()=>{if(days.includes('2026-06-24')){day.value='2026-06-24';draw()}};document.getElementById('theme').onclick=()=>{document.body.classList.toggle('night');document.getElementById('theme').textContent=document.body.classList.contains('night')?'Mode jour':'Mode nuit';draw()};draw();
</script></body></html>"""
    substitutions = {"__PLOTLY__": get_plotlyjs(), "__PAYLOAD__": data,
                     "__START__": audit["evaluation_start"], "__END__": audit["evaluation_end"],
                     "__METRICS__": _metrics(pd.read_parquet(snapshot / "metrics.parquet")),
                     "__SUPPLY__": _table(pd.DataFrame(supply)), "__EPISODE__": _table(pd.DataFrame(episode_rows)),
                     "__PAIRED__": _metrics(pd.read_parquet(snapshot / "paired_diagnostic_metrics.parquet"))}
    for marker, replacement in substitutions.items():
        document = document.replace(marker, replacement)
    path = snapshot / "marginal_cost_v2_report.html"
    path.write_text(document, encoding="utf-8")
    return path
