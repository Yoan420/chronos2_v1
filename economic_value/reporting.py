"""Offline economic-value research report; no trading or model execution.

All plotted values come from the archived engine outputs.  In particular, this
module never fills a missing settlement price or turns a confidence heuristic
into a probability of profit.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=10))


def _columnar(frame: pd.DataFrame) -> dict:
    """Keep field names once and intern repeated strings in a portable format."""
    split = json.loads(frame.to_json(orient="split", index=False, date_format="iso", double_precision=10))
    values = split["data"]
    dictionaries = {}
    if values:
        for index in range(len(split["columns"])):
            present = [row[index] for row in values if row[index] is not None]
            if not present or not all(isinstance(value, str) for value in present):
                continue
            distinct = list(dict.fromkeys(present))
            if len(distinct) > len(present) / 2:
                continue
            lookup = {value: code for code, value in enumerate(distinct)}
            dictionaries[str(index)] = distinct
            for row in values:
                if row[index] is not None:
                    row[index] = lookup[row[index]]
    return {"columns": split["columns"], "values": values, "dictionaries": dictionaries}


def _clean(value):
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (Path, datetime, date)):
        return str(value)
    if hasattr(value, "item"):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_extreme_forecasts(rows: pd.DataFrame, policy: Mapping) -> None:
    """Do not describe a position-only experiment if its forecasts changed."""
    baseline_name = policy.get("baseline_model", "nuclear_kalman")
    candidate_name = policy.get("candidate_model", "nuclear_kalman_extreme_governed")
    selected = rows.loc[rows["strategy"].eq("model")]
    keys = ["timestamp_utc", "zone"]
    values = [name for name in ("forecast", "q10", "q90") if name in selected]
    baseline = selected.loc[selected["model"].eq(baseline_name), keys + values]
    candidate = selected.loc[selected["model"].eq(candidate_name), keys + values]
    if candidate.empty:
        return
    merged = candidate.merge(baseline, on=keys, how="left", suffixes=("_policy", "_base"),
                             indicator=True, validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        raise ValueError("Extreme policy report: baseline forecasts missing for candidate hours")
    for name in values:
        left, right = merged[name + "_policy"], merged[name + "_base"]
        equal = left.eq(right) | (left.isna() & right.isna())
        if not equal.all():
            raise ValueError(f"Extreme policy report: {name} changed; this is not a position-only comparison")


def render_report(rows: pd.DataFrame, metrics: pd.DataFrame, daily: pd.DataFrame,
                  breakdowns: Mapping[str, pd.DataFrame], audit: dict,
                  destination: str | Path) -> Path:
    """Render a portable, non-executable strategy comparison from scored data.

    ``rows`` is the long-form output of the evaluation engine.  One candidate
    selection compares three strategies; different candidates are alternatives,
    never simultaneous additional portfolios.  Live rows may be present for the
    trader table, but annual metrics must already exclude them in the engine.
    """
    required = {"timestamp_utc", "delivery_day", "zone", "model", "strategy"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Economic report: missing row fields {sorted(missing)}")
    if not rows["strategy"].dropna().isin(["no_forecast", "benchmark", "model"]).all():
        raise ValueError("Economic report: unknown strategy")
    destination = Path(destination)
    if destination.suffix.lower() != ".html":
        raise ValueError("Economic report destination must have an .html extension")
    extreme_policy = audit.get("extreme_policy")
    has_extreme_policy = isinstance(extreme_policy, Mapping)
    if has_extreme_policy:
        _validate_extreme_forecasts(rows, extreme_policy)
    # Strategy-level hourly paths are archived by the runner.  They are not used
    # by any chart here: only the selected candidate feeds the hourly table.
    # Avoid tripling a 365-day, multi-country, multi-candidate HTML payload.
    table_fields = [field for field in (
        "timestamp_utc", "delivery_day", "zone", "model",
        "forecast", "reference_price", "actual", "q10", "q90",
        "edge_eur_mwh", "confidence", "signal", "position_mw", "pnl_net_eur",
        "sample", "evaluation_status", "paired_eligible", "portfolio_eligible",
        "policy_position_fraction", "baseline_position_fraction",
        "extreme_probability_up", "extreme_probability_down", "extreme_expected_edge",
        "governance_weight", "policy_reason",
    ) if field in rows.columns]
    table_rows = rows.loc[rows["strategy"].eq("model"), table_fields]
    daily_fields = [field for field in (
        "model", "zone", "strategy", "delivery_day", "cumulative_pnl_eur", "complete_day",
    ) if field in daily.columns]
    breakdown_fields = (
        "model", "zone", "strategy", "group", "pnl_net_eur", "economic_value_added_eur",
        "absolute_energy_mwh", "eligible_hours", "active_hours",
    )
    payload = {"rows": _columnar(table_rows), "metrics": _records(metrics),
               "daily": _columnar(daily[daily_fields]),
               "breakdowns": {str(key): _records(value[[field for field in breakdown_fields if field in value]])
                              for key, value in breakdowns.items()},
               "audit": _clean(audit), "report_payload": {
                   "hourly_strategy": "model", "archive_rows": len(rows),
                   "displayed_rows": len(table_rows), "encoding": "columnar_dictionary_v1",
                   "other_strategy_paths": "archived separately"}}
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # Escape all HTML delimiters, including an untrusted source label in audit.
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    data = data.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    document = _TEMPLATE.replace("__EXTREME_SECTION__", _EXTREME_SECTION if has_extreme_policy else "")
    document = document.replace("__PLOTLY__", get_plotlyjs()).replace("__PAYLOAD__", data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


_EXTREME_SECTION = r'''<section id="extremeSection">
<h2>Expert mouvements extrêmes — politique de position gouvernée</h2>
<p><strong>Les forecasts et les P10/P90 du modèle nucléaire + Kalman restent inchangés.</strong> L'expert modifie uniquement la position simulée, sous le contrôle d'une gouvernance. Ce n'est ni un nouveau forecast électrique, ni une activation opérationnelle.</p>
<p id="extremeProtocol" class="small"></p>
<p class="small">Cette année a déjà été examinée : ce résultat est un test de recherche répété, pas une année indépendante laissée intacte. Le cutoff cible demeure 08 h et le PIT historique n'est pas certifié. Les scores de mouvements extrêmes ne sont ni une probabilité calibrée de gain, ni des intervalles de prix P10/P90.</p>
<div class="cards"><div class="card"><div class="label">EVA de la politique vs Storm</div><div id="extremeEva" class="value">—</div><div class="sub">P&amp;L net politique − Storm</div></div><div class="card"><div class="label">Gain net vs règle de base</div><div id="extremeBaselineGain" class="value">—</div><div id="extremePairedNote" class="sub"></div></div><div class="card"><div class="label">Heures appariées avec position modifiée</div><div id="extremeInterventions" class="value">—</div><div class="sub">Heures-pays appariées pour le portefeuille.</div></div><div class="card"><div class="label">Poids moyen sur les heures appariées</div><div id="extremeMeanWeight" class="value">—</div><div class="sub">Un poids positif ne signifie pas toujours un changement de position.</div></div></div>
<div id="extremeComparison" class="scroll"></div>
<div id="extremeWeights" class="plot"></div><div id="extremeBaselineCurve" class="plot"></div><div id="extremeSpikes" class="plot"></div>
<p class="small">Les interventions comparent la fraction de position effectivement retenue à celle de la règle de base, uniquement sur les heures d'évaluation appariées ; elles ne mesurent pas une modification du forecast. Pour le portefeuille, tous les pays alloués doivent être comparables sur l'heure. Les courbes incluent le démarrage et les retours à la règle de base ; les heures non appariées ou live sont exclues du compteur. L'analyse des spikes reste ex post, jamais un filtre décidé en connaissant le prix réalisé.</p>
<details><summary>Bilan audité et paramètres de l'expert</summary><pre id="extremeAudit"></pre></details>
</section>'''


_TEMPLATE = r'''<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Economic Value Added — laboratoire Chronos-2</title>
<style>
:root{color-scheme:light;--bg:#f2f5f9;--panel:#fff;--ink:#14263d;--muted:#52657a;--line:#d5dfeb;--warn:#fff0d2;--warnline:#c18a21;--good:#00766c;--bad:#b3394c;--neutral:#5f6f85;--blue:#246eaa}
body.night{color-scheme:dark;--bg:#0b1422;--panel:#142235;--ink:#e6edf8;--muted:#aec0d8;--line:#35455b;--warn:#392d1b;--warnline:#b79554;--good:#68d5bd;--bad:#ff96a4;--neutral:#b2c2d9;--blue:#80bcff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1540px;margin:auto;padding:26px}h1{font-size:29px;line-height:1.2;margin:9px 0}h2{font-size:20px;margin:0 0 10px}h3{font-size:16px;margin:18px 0 10px}p{margin:8px 0}header{display:flex;justify-content:space-between;align-items:flex-start;gap:18px}.eyebrow{font-size:12px;letter-spacing:1.6px;text-transform:uppercase;color:var(--muted)}section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:21px;margin:17px 0}.warning{background:var(--warn);border:2px solid var(--warnline)}.warning strong{font-size:17px}.small,.note{font-size:13px;color:var(--muted)}.controls{display:flex;align-items:center;gap:15px;flex-wrap:wrap}.controls label{display:flex;gap:8px;align-items:center}select,input,button{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px 11px}button{cursor:pointer}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid var(--blue);outline-offset:3px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin-top:17px}.card{border:1px solid var(--line);border-radius:9px;padding:15px}.card .label{font-size:12px;color:var(--muted)}.card .value{font-size:23px;font-weight:650;margin-top:5px}.card .sub{font-size:12px;color:var(--muted);margin-top:5px}.scroll{overflow:auto}.statistics{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px}.statistics th,.statistics td{text-align:right;border-bottom:1px solid var(--line);padding:10px 11px}.statistics th{font-weight:600;color:var(--muted)}.statistics th:first-child,.statistics td:first-child{text-align:left}.statistics tbody tr:hover{background:var(--bg)}.good{color:var(--good)}.bad{color:var(--bad)}.neutral{color:var(--neutral)}.pill{display:inline-block;padding:2px 7px;border:1px solid currentColor;border-radius:5px;font-size:12px}.plot{height:385px;min-width:0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}.pending{font-size:12px;color:var(--muted)}pre{font-size:12px;white-space:pre-wrap;word-break:break-word}code{font-size:13px}.equation{font-family:ui-monospace,monospace;padding:10px;background:var(--bg);border-radius:7px;overflow:auto}details{margin-top:15px}summary{cursor:pointer;color:var(--blue)}a{color:var(--blue)}.empty{padding:30px;text-align:center;color:var(--muted)}.sticky{position:sticky;top:0;z-index:3;box-shadow:0 3px 10px #00000010}@media(max-width:950px){main{padding:15px}.cards,.grid{grid-template-columns:1fr 1fr}header{flex-wrap:wrap}.plot{height:340px}}@media(max-width:650px){.cards,.grid{grid-template-columns:1fr}.sticky{position:static}h1{font-size:25px}}
</style><script>__PLOTLY__</script></head>
<body><main>
<header><div><div class="eyebrow">Chronos-2 · laboratoire indépendant · portefeuille test</div><h1>Economic Value Added</h1><p id="period" class="note"></p></div><button id="theme" type="button" aria-pressed="false">Mode nuit</button></header>
<section class="warning" role="note"><strong id="referenceWarning">Simulation hypothétique — aucune preuve de P&amp;L exécutable</strong><p id="referenceDescription"></p><p>Ce rapport simule des décisions avec une règle fixée à l'avance. Il ne reconstitue ni ordres exécutés, ni liquidité, ni impact de marché, ni disponibilité réelle d'un prix de transaction. Les signaux BUY / SELL sont des étiquettes de simulation, pas des recommandations de trading.</p><p id="pitWarning">Cutoff cible strict : 08 h, la veille de livraison. Les horodatages connus sont contrôlés, mais l'information disponible à l'époque (PIT) n'est pas certifiée par la seule présence d'un export historique. Des horodatages reconstruits et un benchmark révisé ne prouvent pas que le signal était disponible à 08 h ; consulter l'audit.</p></section>
<section class="sticky"><div class="controls"><label>Périmètre <select id="zone" aria-label="Pays ou portefeuille"></select></label><label>Candidat <select id="model" aria-label="Modèle candidat"></select></label></div><p class="small">Un candidat à la fois, comparé à Storm et à l'absence de position. Les différents modèles sont des alternatives : leurs positions ne sont jamais additionnées.</p><div class="cards"><div class="card"><div class="label">Valeur ajoutée nette vs Storm</div><div id="eva" class="value">—</div><div id="evaSub" class="sub"></div></div><div class="card"><div id="evaMwLabel" class="label">Valeur ajoutée / MW / an</div><div id="evaMw" class="value">—</div><div id="evaMwSub" class="sub">Uniquement sur 365 jours entièrement appariés.</div></div><div class="card"><div class="label">P&amp;L net simulé du candidat</div><div id="net" class="value">—</div><div id="capacity" class="sub"></div></div><div class="card"><div class="label">Couverture de comparaison</div><div id="coverage" class="value">—</div><div id="coverageSub" class="sub"></div></div></div></section>
<section><h2>Statistics — décision économique</h2><p id="metricScope" class="small"></p><div id="metricsTable" class="scroll"></div><p class="small">P&amp;L/MWh : P&amp;L net divisé par le volume absolu simulé, sans compenser achats et ventes. Sans transaction, le ratio est indéfini (—), pas nul. Le Sharpe est un diagnostic sur les P&amp;L quotidiens complets, et non un Sharpe de rendement sur capital investi. Le drawdown est exprimé en EUR ; sans capital initial explicite, son pourcentage n'est pas interprétable.</p><p class="small">Le hit ratio directionnel mesure si le signe du signal actif était correct ; le taux de transactions gagnantes tient compte des coûts. Ces deux taux peuvent différer. Le gain économique affiché est une différence de P&amp;L nets appariés, pas une amélioration de MAE.</p></section>
<section><h2>Trajectoire économique</h2><div id="equity" class="plot"></div><div id="drawdown" class="plot"></div><p class="small">Le cumul porte sur les heures appariées réellement évaluées. Les journées incomplètes sont signalées ; une observation manquante n'est jamais transformée en P&amp;L nul. Le drawdown est l'écart au plus haut cumulé, avec un point de départ à zéro.</p></section>
__EXTREME_SECTION__
<section><h2>Quand le signal apporte-t-il de la valeur ?</h2><div class="grid"><div id="hour" class="plot"></div><div id="season" class="plot"></div><div id="spike" class="plot"></div><div id="confidence" class="plot"></div></div><div id="monthly" class="plot"></div><p class="small">Heure et saison suivent le calendrier de livraison Europe/Paris, y compris les journées de 23 ou 25 heures. Hiver : octobre à mars ; été : avril à septembre. Les groupes de spikes sont des diagnostics définis par les seuils fixés dans la configuration ; ils ne sont pas connus au moment de décider. La confiance est une heuristique fondée sur le signal et, lorsqu'ils existent, les quantiles du modèle : ce n'est pas une probabilité calibrée de gain. Les trois stratégies sont comparées sur les mêmes heures, regroupées selon la confiance du candidat. Des intervalles P10–P90 non disponibles restent vides.</p><details><summary>Valeurs et effectifs des sous-groupes</summary><div id="breakdownTables" class="scroll"></div></details></section>
<section><h2>Table horaire — signal simulé</h2><div class="controls"><label>Journée <input id="day" type="date"></label><label>Pays <select id="traderZone" aria-label="Pays de la table horaire"></select></label></div><p id="dayStatus" class="small"></p><div id="traderTable" class="scroll"></div><p class="small">Edge = forecast − référence. Une position positive correspond à BUY, une position négative à SELL. P&amp;L brut = position (MW) × durée (h) × (observé − référence), puis déduction des coûts. Une heure future dont l'observé n'est pas publié reste en attente ; ses valeurs ne rentrent pas dans les Statistics annuelles.</p></section>
<section><h2>Protocole, limites et reproductibilité</h2><div class="equation">PnL net = Σ [Position_MW × durée_h × (Prix_observé − Prix_référence)] − coûts<br>EVA = PnL net du candidat − PnL net du benchmark, aux mêmes heures</div><ol><li><strong>Strategy 0 — sans forecast :</strong> position nulle. Il s'agit d'une absence de position incrémentale, pas d'une stratégie de sourcing complète d'un portefeuille physique.</li><li id="benchmarkProtocol"><strong>Strategy 1 — Storm :</strong> le forecast Storm passe par la même règle de position, le même prix de référence et les mêmes plafonds que le candidat.</li><li id="candidateProtocol"><strong>Strategy 2 — candidat :</strong> aucune optimisation a posteriori sur cette année. Les données futures, prix observés du jour et scores réalisés ne servent pas à construire son signal.</li><li><strong>Appariement strict :</strong> benchmark, candidat, prix observé et référence doivent être disponibles sur l'heure comparée. Aucun forward-fill entre journées ni interpolation du prix manquant.</li><li><strong>Gestion des risques :</strong> plafond MW, allocation entre pays, zone neutre et coûts proviennent de la configuration archivée. La taille du portefeuille ne constitue pas une capacité réellement négociable garantie.</li><li><strong>Qualification :</strong> performance simulée, biais possibles de révision, éligibilité des sources et nature négociable ou non du prix de référence sont exposés ci-dessous. Aucun changement du pipeline Forecast.</li></ol><p>« Economic Value Added » désigne ici la valeur incrémentale du signal simulé ; ce n'est pas l'indicateur comptable EVA après coût du capital. Les performances historiques, même positives, ne garantissent aucun gain futur.</p><details><summary>Audit complet et configuration de la simulation</summary><pre id="audit"></pre></details></section>
</main><script id="economic-payload" type="application/json">__PAYLOAD__</script><script>
'use strict';
function unpack(packed,defaults={}){
 if(Array.isArray(packed))return packed;
 const columns=packed.columns,dictionaries=packed.dictionaries||{};
 return packed.values.map(values=>{const row={...defaults};columns.forEach((column,index)=>{const value=values[index];row[column]=value===null?null:dictionaries[index]?dictionaries[index][value]:value});return row});
}
const data=JSON.parse(document.getElementById('economic-payload').textContent);
data.rows=unpack(data.rows,{strategy:'model'});
data.daily=unpack(data.daily);
const strategies=['no_forecast','benchmark','model'];
const labels={no_forecast:'0 · Sans forecast (position nulle)',benchmark:'1 · Storm',model:'2 · Candidat'};
const modelLabels={autonomous:'Autonomous',kalman:'Kalman',nuclear_autonomous:'Nucléaire · Autonomous',nuclear_kalman:'Nucléaire · Kalman',nuclear_kalman_extreme_governed:'Nucléaire · Kalman + politique extrêmes (positions)'};
const extremePolicy=data.audit.extreme_policy&&typeof data.audit.extreme_policy==='object'?data.audit.extreme_policy:null;
const confidenceLabels={unknown:'Indisponible',low:'Faible',medium:'Moyenne',high:'Élevée'};
const selected={zone:document.getElementById('zone'),model:document.getElementById('model'),day:document.getElementById('day'),traderZone:document.getElementById('traderZone')};
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const fmt=(v,d=2)=>finite(v)?v.toLocaleString('fr-FR',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
const pct=v=>finite(v)?fmt(v*100,1)+' %':'—';
const money=v=>finite(v)?fmt(v,0)+' €':'—';
const signClass=v=>!finite(v)||v===0?'neutral':v>0?'good':'bad';
const strategyColor=s=>({no_forecast:dark()?'#a6b6cc':'#65748a',benchmark:dark()?'#83b8ff':'#3b70b6',model:dark()?'#5cd8b7':'#008575'})[s];
function dark(){return document.body.classList.contains('night')}
function cell(value,className){const td=document.createElement('td');td.textContent=value;if(className)td.className=className;return td}
function table(headers,values,classes){const t=document.createElement('table');t.className='statistics';const thead=document.createElement('thead'),tr=document.createElement('tr');headers.forEach(h=>{const th=document.createElement('th');th.textContent=h;tr.appendChild(th)});thead.appendChild(tr);t.appendChild(thead);const tbody=document.createElement('tbody');values.forEach((row,i)=>{const r=document.createElement('tr');row.forEach((v,j)=>r.appendChild(cell(v,classes&&classes[i]?classes[i][j]:'')));tbody.appendChild(r)});t.appendChild(tbody);return t}
function replaceTable(id,headers,values,classes){const host=document.getElementById(id);host.replaceChildren();if(!values.length){const p=document.createElement('p');p.className='empty';p.textContent='Aucune donnée disponible pour cette sélection.';host.appendChild(p);return}host.appendChild(table(headers,values,classes))}
function choose(row){return row.zone===selected.zone.value&&row.model===selected.model.value}
function option(select,value,label){select.add(new Option(label||value,value))}
const zones=[...new Set(data.rows.map(r=>r.zone))].sort(),models=[...new Set(data.rows.map(r=>r.model))].sort();
if(data.metrics.some(r=>r.zone==='PORTFOLIO'))option(selected.zone,'PORTFOLIO','Portefeuille agrégé');zones.forEach(z=>option(selected.zone,z));models.forEach(m=>option(selected.model,m,modelLabels[m]||m));zones.forEach(z=>option(selected.traderZone,z));
const days=[...new Set(data.rows.map(r=>r.delivery_day))].sort();if(days.length){selected.day.min=days[0];selected.day.max=days.at(-1);selected.day.value=days.at(-1)}
const evaluationDays=[...new Set(data.rows.filter(r=>r.sample!=='live').map(r=>r.delivery_day))].sort();
document.getElementById('period').textContent=evaluationDays.length?`${evaluationDays[0]} → ${evaluationDays.at(-1)} · ${evaluationDays.length} jours calendaires · décision à 08 h D−1`:'Aucune fenêtre historique disponible';
document.getElementById('audit').textContent=JSON.stringify(data.audit,null,2);
const auditText=JSON.stringify(data.audit).toLowerCase();
const proxy=auditText.includes('lagged_da')||auditText.includes('previous_day')||auditText.includes('lagged_day')||auditText.includes('previous_da')||auditText.includes('proxy');
document.getElementById('referenceWarning').textContent=proxy?'SIMULATION HYPOTHÉTIQUE — référence proxy non négociable':'SIMULATION HISTORIQUE — exécution et gains non démontrés';
document.getElementById('referenceDescription').textContent=proxy?'Le prix de référence est un proxy historique (par exemple, le prix day-ahead de la veille). Ce n’est pas une cotation achetable ou vendable pour la livraison étudiée. Le P&L mesure une valeur conditionnelle du signal ; il ne permet pas d’annoncer un gain réalisable en salle de marché.':'Le P&L suppose que la référence archivée puisse être utilisée pour la décision simulée. Sa présence dans les données ne prouve ni sa négociabilité, ni une exécution à ce prix : consulter la qualification dans l’audit.';
function layout(title,yTitle){const grid=dark()?'#35455b':'#d5dfeb';return {title:{text:title,font:{size:16}},paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:dark()?'#e6edf8':'#14263d'},margin:{t:65,r:22,b:60,l:80},xaxis:{gridcolor:grid,zerolinecolor:grid},yaxis:{title:yTitle,gridcolor:grid,zerolinecolor:grid},legend:{orientation:'h',y:-0.20},hovermode:'x unified'}}
function plot(id,traces,title,yTitle){Plotly.react(id,traces,layout(title,yTitle),{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']})}
function drawMetrics(){
const ms=data.metrics.filter(choose),candidate=ms.find(r=>r.strategy==='model')||{};
const node=document.getElementById('eva');
node.textContent=money(candidate.economic_value_added_eur);
node.className='value '+signClass(candidate.economic_value_added_eur);
document.getElementById('evaSub').textContent='Candidat − benchmark, après coûts, sur les mêmes heures.';
const annualEva=finite(candidate.economic_value_added_per_mw_year);
document.getElementById('evaMw').textContent=annualEva?fmt(candidate.economic_value_added_per_mw_year,2)+' €/MW/an':finite(candidate.economic_value_added_per_allocated_mw)?fmt(candidate.economic_value_added_per_allocated_mw,2)+' €/MW':'—';
document.getElementById('evaMwLabel').textContent=annualEva?'Valeur ajoutée / MW / an':'Valeur ajoutée / MW — échantillon';
document.getElementById('evaMwSub').textContent=annualEva?'365 jours entièrement appariés.':'Échantillon disponible uniquement, sans annualisation.';
document.getElementById('net').textContent=money(candidate.pnl_net_eur);
document.getElementById('capacity').textContent=finite(candidate.allocated_capacity_mw)?'Allocation : '+fmt(candidate.allocated_capacity_mw,1)+' MW':'Voir le plafond MW et l’allocation dans l’audit.';
const expected=candidate.expected_hours??candidate.total_hours;
document.getElementById('coverage').textContent=finite(candidate.eligible_hours)&&finite(expected)?`${fmt(candidate.eligible_hours,0)} / ${fmt(expected,0)}`:'—';
document.getElementById('coverageSub').textContent=selected.zone.value==='PORTFOLIO'?'Heures-pays appariées / attendues (somme des pays, pas des heures uniques).':'Heures appariées / heures physiques attendues.';
document.getElementById('metricScope').textContent=candidate.annual_window&&candidate.annual_fully_observed?'Fenêtre annuelle de 365 jours entièrement appariée. P&L en EUR ; valeur ajoutée nette vs Storm.':'Échantillon incomplet ou fenêtre différente de 365 jours : P&L de l’échantillon seulement, sans extrapolation annuelle ni €/MW/an.';
const ordered=strategies.map(s=>ms.find(r=>r.strategy===s)).filter(Boolean);replaceTable('metricsTable',['Stratégie','P&L brut (€)','Coûts (€)','P&L net (€)','EVA vs Storm (€)','PnL/MWh (€/MWh)','EVA (€/MW/an)','Sharpe quotidien*','Hit directionnel','Transactions gagnantes','Max. drawdown (€)','Heures actives'],ordered.map(r=>[labels[r.strategy],fmt(r.pnl_gross_eur,0),fmt(r.trading_cost_eur,0),fmt(r.pnl_net_eur,0),fmt(r.economic_value_added_eur,0),fmt(r.pnl_per_mwh),fmt(r.economic_value_added_per_mw_year),fmt(r.sharpe_daily_pnl),pct(r.hit_ratio_directional),pct(r.winning_trade_ratio),fmt(r.max_drawdown_eur,0),fmt(r.active_hours,0)]),ordered.map(r=>['','','',signClass(r.pnl_net_eur),signClass(r.economic_value_added_eur)]));}
function drawTrajectory(){const rows=data.daily.filter(choose);const equity=[],drawdowns=[];for(const strategy of strategies){const rs=rows.filter(r=>r.strategy===strategy).sort((a,b)=>a.delivery_day.localeCompare(b.delivery_day));let peak=0;const drawdown=rs.map(r=>{if(!finite(r.cumulative_pnl_eur))return null;peak=Math.max(peak,r.cumulative_pnl_eur);return r.cumulative_pnl_eur-peak});const line={color:strategyColor(strategy),width:strategy==='model'?3:2,dash:strategy==='no_forecast'?'dot':'solid'};equity.push({x:rs.map(r=>r.delivery_day),y:rs.map(r=>r.cumulative_pnl_eur),name:labels[strategy],type:'scatter',mode:'lines',line,connectgaps:false,customdata:rs.map(r=>r.complete_day?'Journée complète':'Journée incomplète — échantillon seulement'),hovertemplate:'%{x}<br>%{y:,.2f} €<br>%{customdata}<extra>%{fullData.name}</extra>'});drawdowns.push({x:rs.map(r=>r.delivery_day),y:drawdown,name:labels[strategy],type:'scatter',mode:'lines',line,connectgaps:false})}plot('equity',equity,'Cumul des P&L nets simulés','EUR');plot('drawdown',drawdowns,'Drawdown du cumul quotidien','EUR sous le plus haut');}
const groupTitles={hour:'Performance par heure de livraison',season:'Hiver (oct.–mars) / été (avr.–sept.)',spike:'Performance pendant les spikes',confidence:'P&L selon la confiance du candidat',monthly:'P&L par mois'};
function groupLabel(group,key){if(key==='confidence')return confidenceLabels[group]||String(group);if(key==='season')return {winter:'Hiver',summer:'Été'}[group]||String(group);if(key==='spike')return {high:'Prix hauts',low:'Prix bas',normal:'Hors spikes',unknown:'Indisponible',large_reference_move:'Fort écart à la référence'}[group]||String(group);return String(group??'Indisponible')}
function drawBreakdowns(){const all=document.getElementById('breakdownTables');all.replaceChildren();for(const key of ['hour','season','spike','confidence','monthly']){const rs=(data.breakdowns[key]||[]).filter(choose);const groups=[...new Set(rs.map(r=>r.group))];const traces=strategies.map(strategy=>{const byGroup=new Map(rs.filter(r=>r.strategy===strategy).map(r=>[r.group,r]));return {x:groups.map(g=>groupLabel(g,key)),y:groups.map(g=>byGroup.get(g)?.pnl_net_eur??null),name:labels[strategy],type:'bar',marker:{color:strategyColor(strategy)},customdata:groups.map(g=>{const r=byGroup.get(g)||{};return [r.eligible_hours??null,r.active_hours??null]}),hovertemplate:'%{x}<br>P&L net %{y:,.2f} €<br>Heures appariées %{customdata[0]}<br>Heures actives %{customdata[1]}<extra>%{fullData.name}</extra>'}});plot(key,traces,groupTitles[key],'P&L net simulé (EUR)');const h=document.createElement('h3');h.textContent=groupTitles[key];all.appendChild(h);if(rs.length)all.appendChild(table(['Groupe','Stratégie','P&L net (€)','EVA vs Storm (€)','Volume absolu (MWh)','Heures appariées','Heures actives'],rs.map(r=>[groupLabel(r.group,key),labels[r.strategy],fmt(r.pnl_net_eur,0),fmt(r.economic_value_added_eur,0),fmt(r.absolute_energy_mwh,1),fmt(r.eligible_hours,0),fmt(r.active_hours,0)])));}}
function extremeNames(){return {baseline:extremePolicy?.baseline_model||'nuclear_kalman',candidate:extremePolicy?.candidate_model||'nuclear_kalman_extreme_governed'}}
function extremeComparisonPaired(){const proof=extremePolicy?.baseline_comparison_paired;return proof===true||(proof&&typeof proof==='object'&&proof[selected.zone.value]===true)}
function drawExtreme(){
 if(!extremePolicy)return;
 const names=extremeNames(),scope=selected.zone.value,paired=extremeComparisonPaired();
 const findMetric=model=>data.metrics.find(r=>r.zone===scope&&r.model===model&&r.strategy==='model');
 const baseline=findMetric(names.baseline),candidate=findMetric(names.candidate);
 const gain=paired&&finite(candidate?.pnl_net_eur)&&finite(baseline?.pnl_net_eur)?candidate.pnl_net_eur-baseline.pnl_net_eur:null;
 const setValue=(id,value)=>{const node=document.getElementById(id);node.textContent=money(value);node.className='value '+signClass(value)};
 setValue('extremeEva',candidate?.economic_value_added_eur);setValue('extremeBaselineGain',gain);
 document.getElementById('extremePairedNote').textContent=paired?'Politique − règle de base, mêmes heures et mêmes coûts.':'Comparaison indisponible : appariement avec la règle de base non attesté.';
 const training=extremePolicy.training_days??365,refit=extremePolicy.refit_every_days??7,warmup=extremePolicy.governance_minimum_days??28,lookback=extremePolicy.governance_lookback_days??60;
 document.getElementById('extremeProtocol').textContent=`Entraînement sur ${training} jours glissants ; réentraînement tous les ${refit} jours. Gouvernance : au moins ${warmup} jours de résultats antérieurs, fenêtre de ${lookback} jours. Pendant le démarrage ou faute de preuve suffisante, retour à la règle de position de base, pas à un nouveau forecast.`;
 document.getElementById('extremeAudit').textContent=JSON.stringify(extremePolicy,null,2);
 document.getElementById('benchmarkProtocol').textContent='Strategy 1 — Storm : règle de position de référence inchangée, mêmes prix de référence, coûts et plafonds. L’expert de mouvements extrêmes intervient uniquement sur la décision du candidat.';
 document.getElementById('candidateProtocol').textContent='Strategy 2 — candidat : forecasts et P10/P90 inchangés ; une politique apprise sur les jours antérieurs ajuste la position. Cette année déjà étudiée reste un test de recherche répété, pas une validation finale indépendante. Les prix du jour évalué ne servent pas à choisir son intervention.';
 const scopeRows=data.rows.filter(r=>r.model===names.candidate&&r.sample==='evaluation'&&(scope==='PORTFOLIO'||r.zone===scope));
 const policyRows=scopeRows.filter(r=>r.paired_eligible===true&&(scope!=='PORTFOLIO'||r.portfolio_eligible===true));
 const summaryEntries=Array.isArray(extremePolicy.summary)?extremePolicy.summary:Object.entries(extremePolicy.summary||{}).map(([zone,value])=>({zone,...value}));
 const summary=summaryEntries.find(r=>r.zone===scope)||{};
 const dayMap=new Map(scopeRows.map(row=>[row.delivery_day,{weights:[],interventions:0,knownPositions:0}]));
 for(const row of policyRows){if(!dayMap.has(row.delivery_day))dayMap.set(row.delivery_day,{weights:[],interventions:0,knownPositions:0});const day=dayMap.get(row.delivery_day);if(finite(row.governance_weight))day.weights.push(row.governance_weight);if(finite(row.policy_position_fraction)&&finite(row.baseline_position_fraction)){day.knownPositions++;if(Math.abs(row.policy_position_fraction-row.baseline_position_fraction)>1e-10)day.interventions++;}}
 const ds=[...dayMap].sort((a,b)=>a[0].localeCompare(b[0]));
 const weights=policyRows.map(r=>r.governance_weight).filter(finite),interventions=ds.reduce((sum,entry)=>sum+entry[1].interventions,0),knownPositions=ds.reduce((sum,entry)=>sum+entry[1].knownPositions,0);
 const meanWeight=finite(summary.mean_governance_weight)?summary.mean_governance_weight:weights.length?weights.reduce((a,b)=>a+b,0)/weights.length:null;
 document.getElementById('extremeMeanWeight').textContent=pct(meanWeight);
 document.getElementById('extremeInterventions').textContent=fmt(finite(summary.intervention_hours)?summary.intervention_hours:knownPositions?interventions:null,0);
 replaceTable('extremeComparison',['Décision','P&L net (€)','EVA vs Storm (€)','Gain vs règle de base (€)','Hit directionnel effectif','Heures actives'],[baseline,candidate].filter(Boolean).map(r=>[r.model===names.baseline?'Règle de base · forecasts nucléaires + Kalman':'Politique extrêmes gouvernée · mêmes forecasts',fmt(r.pnl_net_eur,0),fmt(r.economic_value_added_eur,0),fmt(paired?(r.model===names.baseline?0:gain):null,0),pct(r.hit_ratio_directional),fmt(r.active_hours,0)]));
 const wl=layout('Interventions et poids de gouvernance par journée','Poids moyen (0–1)');wl.yaxis.range=[0,1];wl.yaxis2={title:scope==='PORTFOLIO'?'Heures-pays modifiées':'Heures modifiées',overlaying:'y',side:'right',rangemode:'tozero',showgrid:false};
 Plotly.react('extremeWeights',[{x:ds.map(r=>r[0]),y:ds.map(r=>r[1].knownPositions?r[1].interventions:null),name:'Positions modifiées',type:'bar',yaxis:'y2',opacity:0.45,marker:{color:dark()?'#c1a4ff':'#7752b7'}},{x:ds.map(r=>r[0]),y:ds.map(r=>r[1].weights.length?r[1].weights.reduce((a,b)=>a+b,0)/r[1].weights.length:null),name:'Poids moyen',type:'scatter',mode:'lines',connectgaps:false,line:{color:strategyColor('model'),width:2}}],wl,{responsive:true,displaylogo:false});
 const baseDaily=new Map(data.daily.filter(r=>r.zone===scope&&r.model===names.baseline&&r.strategy==='model').map(r=>[r.delivery_day,r.cumulative_pnl_eur]));
 const candidateDaily=data.daily.filter(r=>r.zone===scope&&r.model===names.candidate&&r.strategy==='model').sort((a,b)=>a.delivery_day.localeCompare(b.delivery_day));
 plot('extremeBaselineCurve',[{x:candidateDaily.map(r=>r.delivery_day),y:candidateDaily.map(r=>paired&&finite(r.cumulative_pnl_eur)&&finite(baseDaily.get(r.delivery_day))?r.cumulative_pnl_eur-baseDaily.get(r.delivery_day):null),name:'Politique − règle de base',type:'scatter',mode:'lines',connectgaps:false,line:{color:strategyColor('model'),width:3}}],'Gain net cumulé vs règle de base — échantillon apparié','EUR');
 const spikeRows=(data.breakdowns.spike||[]).filter(r=>r.zone===scope&&r.strategy==='model'),baseSpikes=new Map(spikeRows.filter(r=>r.model===names.baseline).map(r=>[r.group,r.pnl_net_eur])),candidateSpikes=spikeRows.filter(r=>r.model===names.candidate);
 const spikeGain=candidateSpikes.map(r=>paired&&finite(r.pnl_net_eur)&&finite(baseSpikes.get(r.group))?r.pnl_net_eur-baseSpikes.get(r.group):null);
 plot('extremeSpikes',[{x:candidateSpikes.map(r=>groupLabel(r.group,'spike')),y:spikeGain,name:'Gain net politique − règle de base',type:'bar',marker:{color:spikeGain.map(value=>!finite(value)?strategyColor('no_forecast'):value>=0?(dark()?'#68d5bd':'#00766c'):(dark()?'#ff96a4':'#b3394c'))}}],'Gain vs règle de base pendant les spikes et hors spikes','EUR');
}
function hourLabel(r){const local=new Date(r.timestamp_utc).toLocaleString('fr-FR',{timeZone:'Europe/Paris',hour:'2-digit',minute:'2-digit',timeZoneName:'short'});return local}
function drawTrader(){
 const rs=data.rows.filter(r=>r.model===selected.model.value&&r.zone===selected.traderZone.value&&r.delivery_day===selected.day.value&&r.strategy==='model').sort((a,b)=>String(a.timestamp_utc).localeCompare(String(b.timestamp_utc)));
 const pending=rs.filter(r=>!finite(r.actual)).length,sample=[...new Set(rs.map(r=>r.sample||'historique'))].join(', '),isPolicy=extremePolicy&&selected.model.value===extremeNames().candidate;
 document.getElementById('dayStatus').textContent=`${rs.length} heures physiques · ${pending} observations en attente · échantillon : ${sample}. Les signaux non éligibles sont des abstentions.`+(isPolicy?' Forecast et P10/P90 inchangés ; signal et position correspondent à la décision effective gouvernée. Scores extrêmes non calibrés en probabilité de gain.':'');
 const headers=['Heure locale',isPolicy?'Forecast de base (inchangé)':'Forecast','Référence',isPolicy?'Edge du forecast':'Edge','P10','P90',isPolicy?'Confiance du forecast*':'Confiance*',isPolicy?'Signal effectif simulé':'Signal simulé','Position (MW)','Observé','P&L net (€)','Statut'];
 const optional=isPolicy?[['baseline_position_fraction','Fraction de base',v=>fmt(v,3)],['policy_position_fraction','Fraction retenue',v=>fmt(v,3)],['extreme_probability_up','Score hausse extrême*',pct],['extreme_probability_down','Score baisse extrême*',pct],['extreme_expected_edge','Edge expert estimé',fmt],['governance_weight','Poids gouvernance',pct],['policy_reason','Décision / repli',v=>v??'—']].filter(([field])=>rs.some(r=>Object.hasOwn(r,field))):[];
 headers.push(...optional.map(r=>r[1]));
 replaceTable('traderTable',headers,rs.map(r=>[hourLabel(r),fmt(r.forecast??r.model_forecast),fmt(r.reference_price),fmt(r.edge_eur_mwh),fmt(r.q10),fmt(r.q90),confidenceLabels[r.confidence]||'Indisponible',r.signal||'—',fmt(r.position_mw,1),fmt(r.actual),fmt(r.pnl_net_eur),r.evaluation_status||'—',...optional.map(([field,label,format])=>format(r[field]))]),rs.map(r=>['','','',signClass(r.edge_eur_mwh),'','','',r.signal==='BUY'?'good':r.signal==='SELL'?'bad':'neutral','','',signClass(r.pnl_net_eur)]));
}
function draw(){drawMetrics();drawTrajectory();drawBreakdowns();drawTrader();drawExtreme()}
selected.zone.onchange=()=>{if(zones.includes(selected.zone.value))selected.traderZone.value=selected.zone.value;draw()};selected.model.onchange=draw;selected.day.onchange=drawTrader;selected.traderZone.onchange=drawTrader;
document.getElementById('theme').onclick=()=>{document.body.classList.toggle('night');const isDark=dark(),button=document.getElementById('theme');button.textContent=isDark?'Mode jour':'Mode nuit';button.setAttribute('aria-pressed',String(isDark));try{localStorage.setItem('economic-value-theme',isDark?'night':'day')}catch(e){}draw()};
try{if(localStorage.getItem('economic-value-theme')==='night'){document.body.classList.add('night');document.getElementById('theme').textContent='Mode jour';document.getElementById('theme').setAttribute('aria-pressed','true')}}catch(e){}
draw();
</script></body></html>'''
