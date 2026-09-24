"""Self-contained research report. Rendering never imports a model/client."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=8))


def render_report(frame: pd.DataFrame, metrics: pd.DataFrame, paired_metrics: pd.DataFrame,
                  daily: pd.DataFrame, audit: dict, destination: Path) -> Path:
    """Every chart is sourced from the exact scored, archived predictions."""
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    fields = [c for c in ("timestamp", "day", "zone", "base", "expert", "guarded", "actual", "storm",
                           "weight", "risk", "decision", "mode", "candidate_id", "shortage_mw",
                           "thermal_margin_mw", "net_position_mw", "marginal_regime_proxy") if c in frame]
    payload = {"hourly": _records(frame[fields]), "metrics": _records(metrics),
               "paired": _records(paired_metrics), "daily": _records(daily), "audit": audit}
    data = json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False).replace("</", "<\\/")
    labels = {"base": "Modèle actuel figé", "expert": "Expert coût marginal", "guarded": "Intervention conditionnelle", "storm": "Storm"}
    def table(values: pd.DataFrame) -> str:
        if values.empty:
            return "<p>Comparaison indisponible : couverture commune insuffisante.</p>"
        values = values.copy()
        values["model"] = values.model.map(labels)
        columns = {"zone": "Pays", "model": "Modèle", "days": "Jours", "hours": "Heures", "mae": "MAE horaire",
                   "daily_mean_mae": "MAE prix moyen du jour", "worst10pct_days_mae": "MAE des 10 % pires jours", "bias": "Biais"}
        return values[list(columns)].rename(columns=columns).to_html(index=False, border=0, classes="statistics", float_format=lambda v: f"{v:.3f}")
    diagnostic_rows = []
    for zone, values in audit.get("intervention_diagnostics", {}).items():
        physical = audit.get("physical_diagnostics", {}).get(zone, {})
        diagnostic_rows.append({"Pays": zone, "Gain MAE annuel": values["annual_mae_gain"],
                                "Heures d'intervention": values["active_hours"],
                                "Heures améliorées": values["beneficial_active_hours"],
                                "Heures aggravées": values["harmful_active_hours"],
                                "Heures expert indisponible": physical.get("expert_unavailable_hours"),
                                "Part de déficit de la pile (%)": None if physical.get("shortage_proxy_share") is None else 100 * physical["shortage_proxy_share"]})
    diagnostic_table = pd.DataFrame(diagnostic_rows).to_html(index=False, border=0, classes="statistics", float_format=lambda v: f"{v:.4f}")
    document = """<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Expert indépendant de coût marginal zonal — rolling 365</title><style>
:root{color-scheme:light;--bg:#f4f7fb;--panel:#fff;--ink:#172b44;--muted:#53657d;--line:#dce4ee;--warn:#fff4d7}body.night{color-scheme:dark;--bg:#0c1422;--panel:#142136;--ink:#e8eff9;--muted:#b7c8df;--line:#34455f;--warn:#44371a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;align-items:center;gap:20px}h1{font-size:27px;margin:8px 0}h2{font-size:20px}.tag{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:1px}section{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin:18px 0;padding:20px;overflow:auto}.warning{background:var(--warn)}button,select,input{font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}button{cursor:pointer}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.statistics{width:100%;border-collapse:collapse;font-size:13px}.statistics th,.statistics td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:right}.statistics td:nth-child(-n+2),.statistics th:nth-child(-n+2){text-align:left}.plot{min-height:380px}.small{color:var(--muted);font-size:13px}pre{white-space:pre-wrap;word-break:break-word;font-size:12px}a{color:#3d8bd4}.good{color:#14977b}.bad{color:#db5661}</style>
<script>__PLOTLY__</script></head><body><main><header><div><div class="tag">Laboratoire indépendant · aucun changement de Complete</div><h1>Coût marginal zonal & intervention conditionnelle</h1><p id="period"></p></div><button id="theme">Mode nuit</button></header>
<section class="warning"><strong>Expérience diagnostique — aucune activation opérationnelle.</strong><p id="status"></p><p><strong>Périmètre testé : pile résiduelle partielle gaz + nucléaire.</strong> Le charbon, le lignite, plusieurs autres filières et les échanges ne sont pas encore représentés. Un déficit de cette pile déclenche une pénalité hypothétique de 4 000 EUR/MWh : ce n'est pas une prévision validée de pénurie réelle. Le résultat mesure les limites de ce prototype, pas celles d'un merit order européen complet.</p><p>Les prix ne constituent pas l'identification certifiée de la dernière centrale réelle. Les scénarios de capacités/offres sont des hypothèses. La simulation « zonal_degraded » ne reproduit pas le couplage européen. Les données horaires ne restituent pas les variations au quart d'heure.</p><p>Les labels canoniques historiques peuvent inclure des révisions postérieures ; les états Saturn interrogés as-of ne constituent pas une preuve indépendante de publication fournisseur. Les dates déjà discutées du 24–26 juin sont des diagnostics post-hoc, pas un test final indépendant.</p></section>
<section><h2>Statistics — 365 jours, toutes les heures du modèle actuel</h2><p class="small">MAE et biais en EUR/MWh. Référence et intervention conservent toutes les heures. L'expert seul est évalué uniquement lorsqu'il est disponible : son compteur peut être inférieur à 365 jours ; utiliser alors le tableau apparié pour une comparaison équitable. La MAE du prix moyen est calculée journée par journée. Chaque modèle a son propre ensemble des 10 % de journées aux plus grandes erreurs. La référence est figée par pays, jamais choisie a posteriori à chaque heure.</p>__TABLE__</section>
<section><h2>Interventions et couverture physique</h2><p class="small">Gain positif : amélioration. Les abstentions ne sont pas des gains. Un gain minuscule sur quelques heures ne constitue pas une validation statistique. « Déficit » concerne la pile partielle simulée, pas l'état réel du système électrique.</p>__DIAGNOSTICS__</section>
<section><h2>Comparaison appariée avec Storm</h2><p class="small">Dans chaque pays, les modèles disponibles sont évalués sur exactement les mêmes heures communes avec Storm et l'expert. Un expert indisponible sur tout le pays est exclu de ce tableau. Une heure Storm manquante n'est pas remplacée. Le compteur d'heures peut donc différer du tableau précédent.</p>__PAIRED__</section>
<section><div class="controls"><label>Pays <select id="zone"></select></label><label>Journée <input id="day" type="date"></label><button id="june">Voir le 24 juin 2026</button></div><p class="small">Pour garder les prévisions usuelles lisibles, la courbe de l'expert brut est masquée au départ : cliquer « Coût marginal » dans la légende pour l'afficher, y compris ses valeurs extrêmes. Toutes ses erreurs restent dans les Statistics.</p><div id="forecast" class="plot"></div><div id="weights" class="plot"></div><p id="dayNote" class="small"></p></section>
<section><h2>Erreurs quotidiennes et effet de l'intervention</h2><div id="daily" class="plot"></div><div id="gain" class="plot"></div><p class="small">Gain positif : l'intervention réduit l'erreur absolue. Les interventions inutiles et nuisibles restent incluses. Un poids nul pendant le démarrage ou faute de preuve ne démontre pas un gain prédictif.</p></section>
<section><h2>Lecture méthodologique</h2><ol><li>Les scénarios physiques ne lisent aucun prix électrique : ils combinent demande prévue, offres disponibles et coûts des combustibles.</li><li>Chaque journée choisit son scénario sur les 365 jours calendaires précédents. Les prix du jour prédit ne sont pas utilisés.</li><li>La gouvernance apprend sur les erreurs de prévisions déjà hors échantillon. Elle reste à poids nul au démarrage et ne prend pas une décision à partir du résultat futur.</li><li>Les limites annuelles sont testées après le replay ; elles ne servent pas à revenir en arrière pour remplacer les mauvaises interventions par la référence.</li><li>Prévisions et sources sont archivées ; refaire ce rapport ne réentraîne pas le modèle.</li></ol><p>Aucune garantie de non-dégradation future. Une simulation proche des prix ne prouve ni l'identité d'une centrale marginale, ni une cause de congestion.</p><details><summary>Sources, hypothèses et audits complets</summary><pre id="audit"></pre></details><p><a href="https://arxiv.org/abs/2501.02963">Merit order appris — Ghelasi & Ziel</a> · <a href="https://www.nemo-committee.eu/assets/files/euphemia-public-description.pdf">EUPHEMIA</a> · <a href="https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html">Solveur HiGHS</a></p></section>
</main><script>
const payload=__PAYLOAD__, names={base:'Modèle actuel',expert:'Coût marginal',guarded:'Intervention conditionnelle',actual:'Observé',storm:'Storm'}, colors={base:'#8f78d9',expert:'#e8a239',guarded:'#19ad91',actual:'#ee6473',storm:'#669bdd'};
const zone=document.getElementById('zone'),day=document.getElementById('day'); [...new Set(payload.hourly.map(r=>r.zone))].sort().forEach(z=>zone.add(new Option(z,z)));
const days=[...new Set(payload.hourly.map(r=>r.day))].sort();day.min=days[0];day.max=days.at(-1);day.value=days.at(-1);
document.getElementById('period').textContent=`${days[0]} → ${days.at(-1)} · ${days.length} journées · ${payload.audit.status||'diagnostic'}`;
document.getElementById('status').textContent=payload.audit.result_summary||'Le rapport décrit les résultats expérimentaux, sans promouvoir le candidat.';
document.getElementById('audit').textContent=JSON.stringify(payload.audit,null,2);
function layout(title){let dark=document.body.classList.contains('night');return {title:{text:title},paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:dark?'#e8eff9':'#172b44'},margin:{t:55,r:25,b:55,l:70},xaxis:{gridcolor:dark?'#34455f':'#dce4ee'},yaxis:{gridcolor:dark?'#34455f':'#dce4ee',title:'EUR/MWh'},legend:{orientation:'h'},hovermode:'x unified'}}
function line(rows,key){return{x:rows.map(r=>r.timestamp),y:rows.map(r=>r[key]),type:'scatter',mode:'lines',name:names[key],visible:key==='expert'?'legendonly':true,connectgaps:false,line:{color:colors[key],width:key==='actual'?3:2,dash:key==='expert'?'dot':'solid'}}}
function draw(){const rows=payload.hourly.filter(r=>r.zone===zone.value&&r.day===day.value);Plotly.react('forecast',['actual','base','expert','guarded','storm'].map(k=>line(rows,k)),layout(`Prévisions — ${zone.value} — ${day.value} (axe UTC)`),{responsive:true});let wl=layout('Poids de l’expert / décision par heure');wl.yaxis.title='Poids (0–1)';wl.yaxis.range=[0,1];Plotly.react('weights',[{x:rows.map(r=>r.timestamp),y:rows.map(r=>r.weight),customdata:rows.map(r=>[r.risk,r.decision,r.mode,r.candidate_id]),type:'bar',marker:{color:colors.guarded},hovertemplate:'%{x}<br>Poids %{y:.2f}<br>%{customdata[0]} / %{customdata[1]}<br>%{customdata[2]} / %{customdata[3]}<extra></extra>'}],wl,{responsive:true});document.getElementById('dayNote').textContent=`${rows.length} heures physiques. Modes : ${[...new Set(rows.map(r=>r.mode))].join(', ')}. Régimes : ${[...new Set(rows.map(r=>r.risk))].join(', ')}.`;
const ds=payload.daily.filter(r=>r.zone===zone.value);Plotly.react('daily',['base','expert','guarded','storm'].map(k=>{let x=ds.filter(r=>r.model===k);return{x:x.map(r=>r.day),y:x.map(r=>r.mae),name:names[k],visible:k==='expert'?'legendonly':true,type:'scatter',mode:'lines',line:{color:colors[k]}}}),layout('MAE quotidienne'),{responsive:true});const bs=ds.filter(r=>r.model==='base'),gs=new Map(ds.filter(r=>r.model==='guarded').map(r=>[r.day,r.mae]));const gains=bs.map(r=>r.mae-gs.get(r.day));Plotly.react('gain',[{x:bs.map(r=>r.day),y:gains,type:'bar',name:'Gain',marker:{color:gains.map(x=>x>=0?'#19ad91':'#ee6473')}}],layout('Gain de MAE par journée : référence − intervention'),{responsive:true});}
zone.onchange=draw;day.onchange=draw;document.getElementById('june').onclick=()=>{if(days.includes('2026-06-24')){day.value='2026-06-24';draw()}};document.getElementById('theme').onclick=()=>{document.body.classList.toggle('night');document.getElementById('theme').textContent=document.body.classList.contains('night')?'Mode jour':'Mode nuit';draw()};draw();
</script></body></html>"""
    document = document.replace("__PLOTLY__", get_plotlyjs()).replace("__TABLE__", table(metrics.loc[metrics.model.ne("storm")]))
    document = document.replace("__PAIRED__", table(paired_metrics)).replace("__DIAGNOSTICS__", diagnostic_table).replace("__PAYLOAD__", data)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination
