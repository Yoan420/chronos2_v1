"""Offline research report: synthetic demand response is never an observed forecast."""
from __future__ import annotations

from datetime import date, datetime
import json
import math
from numbers import Integral, Real
from pathlib import Path

EXPERT_ID = "nyx_demand_response"
ZONES = ("FR", "DE", "BE", "NL")
SCORES = ("mae_eur_mwh", "rmse_eur_mwh", "win_rate_hour_pct", "win_rate_day_mae_pct",
          "win_rate_day_mean_price_pct", "mae_day_mean_price_eur_mwh", "mean_price_eur_mwh")


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    if hasattr(value, "item"):
        return _safe(value.item())
    if isinstance(value, Real) and not isinstance(value, bool):
        if math.isinf(float(value)):
            raise ValueError("Infinite report value is forbidden.")
        if math.isnan(float(value)):
            return None
    return value


def _prepared(payload):
    result = _safe(payload)
    if result.get("schema_version") != 1:
        raise ValueError("Demand response report requires schema_version=1.")
    period = result["period"]
    if period.get("days") != 365 or (date.fromisoformat(period["end_day"])-date.fromisoformat(period["start_day"])).days != 364:
        raise ValueError("Exactly 365 calendar evaluation days are required.")
    decision = result["decision"]
    for key in ("qualified_country_hours", "total_country_hours", "interventions"):
        value = decision[key]
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid decision count: {key}.")
    if decision["qualified_country_hours"] > decision["total_country_hours"]:
        raise ValueError("Qualified support exceeds total support.")
    if decision["interventions"] > decision["qualified_country_hours"]:
        raise ValueError("Interventions exceed qualified support.")
    if decision.get("integration_ready") is not False or decision.get("empirical_gain_demonstrated") is not False:
        raise ValueError("This diagnostic report cannot authorize integration or claim a demonstrated gain.")
    rows = result.setdefault("kpi_rows", [])
    keys = [(r["zone"], r["model_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate zone/model KPI identities.")
    if any(r["zone"] not in (*ZONES, "ALL") for r in rows):
        raise ValueError("Only the four CWE countries and their ALL aggregate are supported.")
    for row in result.setdefault("cases", []):
        stamp = datetime.fromisoformat(str(row["timestamp_utc"]).replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("Case timestamps must be timezone-aware.")
        if row["zone"] not in ZONES:
            raise ValueError("Unsupported case country.")
    if not decision["qualified_country_hours"]:
        for row in rows:
            if row["model_id"] == EXPERT_ID and (row.get("n_hours", 0) != 0 or any(row.get(s) is not None for s in SCORES)):
                raise ValueError("An unqualified expert must not inherit baseline KPI scores.")
        if any(row.get("expert") is not None for row in result["cases"]):
            raise ValueError("Unqualified real cases must leave the expert price missing.")
        for zone in ("ALL", *ZONES):
            if (zone, EXPERT_ID) not in keys:
                rows.append({"zone": zone, "model_id": EXPERT_ID, "status": "unqualified",
                             "n_hours": 0, **{s: None for s in SCORES}})
    paired = result.get("paired_expert_evaluation")
    if paired is not None:
        if not decision["qualified_country_hours"] or not isinstance(paired, dict) or not isinstance(paired.get("rows"), list):
            raise ValueError("Paired expert evaluation requires qualified support and explicit KPI rows.")
        if any(paired.get("period", {}).get(k) != period[k] for k in ("start_day", "end_day", "days")):
            raise ValueError("Paired expert period differs from the fixed annual evaluation.")
        paired_keys = [(r["zone"], r["model_id"]) for r in paired["rows"]]
        if len(paired_keys) != len(set(paired_keys)):
            raise ValueError("Duplicate paired KPI identities.")
        for zone in {r["zone"] for r in paired["rows"]}:
            group = [r for r in paired["rows"] if r["zone"] == zone]
            if zone not in (*ZONES, "ALL") or not any(r["model_id"] == EXPERT_ID for r in group):
                raise ValueError("Paired support must contain the expert for each reported zone.")
            counts = {r.get("n_hours") for r in group}
            if len(counts) != 1 or not all(isinstance(n, Integral) and not isinstance(n, bool) and 0 <= n <= decision["qualified_country_hours"] for n in counts):
                raise ValueError("Expert and references must use identical paired hour counts.")
    for name in ("blockers", "sources", "demos", "sensitivity"):
        result.setdefault(name, [])
    result.setdefault("audit", {})
    result["zones"] = list(ZONES)
    result["expert_id"] = EXPERT_ID
    return result


def render_report(payload: dict, destination: Path) -> Path:
    """Validate and write a new report. No source access, model fitting or overwrite."""
    data = _prepared(payload)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for old, new in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        encoded = encoded.replace(old, new)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf8") as handle:
        handle.write(TEMPLATE.replace("@@DATA@@", encoded))
    return destination


TEMPLATE = r'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NYX · Réponse de la demande</title><style>
:root{--bg:#f1f5f9;--paper:#fff;--ink:#182b43;--muted:#52647a;--line:#dae2ec;--accent:#315cac;--price:#713fbd;--volume:#08745f;--warning:#895600;--soft:#fff5dd}:root[data-theme=dark]{--bg:#111b27;--paper:#1a293a;--ink:#edf3fc;--muted:#b8c6d9;--line:#354b64;--accent:#a4c3ff;--price:#d0acff;--volume:#73dbbd;--warning:#ffd283;--soft:#382e1e}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,Segoe UI,sans-serif}main{max-width:1450px;margin:auto;padding:30px}header{display:flex;justify-content:space-between;gap:20px}h1{font-size:30px;line-height:1.2;margin:5px 0 12px}h2{font-size:19px;margin:0 0 12px}h3{font-size:15px;margin:16px 0 10px}.eyebrow{color:var(--accent);font-size:11px;letter-spacing:2px;text-transform:uppercase}.muted,.foot{color:var(--muted)}.foot{font-size:12px}.card,.decision{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:22px;margin:18px 0}.decision{background:var(--soft);border-left:4px solid var(--warning)}.decision h2{color:var(--warning)}button,select{font:inherit;color:var(--ink);background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:8px 12px}.filters{display:flex;gap:12px;align-items:center}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:12px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px;border-bottom:1px solid var(--line)}th{color:var(--muted)}td:first-child,th:first-child{text-align:left}tr.expert td{color:var(--warning)}.wrap td{white-space:normal;vertical-align:top;text-align:left}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.stat{display:inline-block;margin:4px 30px 4px 0}.stat strong{font-size:23px;display:block}a{color:var(--accent)}svg{width:100%;height:auto}svg text{fill:var(--muted);font:11px system-ui}.tip{font-size:12px;min-height:24px;color:var(--muted)}pre{white-space:pre-wrap;word-break:break-word;max-height:400px;overflow:auto;font-size:11px}.badge{font-size:11px;font-weight:700;color:var(--warning)}@media(max-width:800px){main{padding:14px}.grid{grid-template-columns:1fr}header{flex-direction:column}h1{font-size:24px}}
</style></head><body><main><header><div><div class="eyebrow">Laboratoire indépendant · production inchangée</div><h1>Réponse de la demande · mécanisme et faisabilité</h1><p class="muted" id="period"></p></div><div><button id="theme">Mode nuit</button></div></header>
<section class="decision"><h2 id="decision-title">Intégration non autorisée</h2><p id="decision-reason"></p><div><span class="stat"><strong id="qualified">—</strong>heures-pays qualifiées / total</span><span class="stat"><strong id="interventions">—</strong>interventions réelles</span><span class="stat"><strong>Non démontré</strong>gain empirique de l'expert</span></div><p><strong>Une démonstration de mécanisme n'est pas un backtest concluant.</strong> Aucun prix ni volume hypothétique ci-dessous n'est injecté dans les forecasts opérationnels. L'absence d'intervention ne constitue pas une amélioration mesurée.</p></section>
<section class="card"><h2>Ce qui manque pour un test réel</h2><div class="scroll"><table id="blockers" class="wrap"><thead><tr><th>Contrôle</th><th>Constat</th></tr></thead><tbody></tbody></table></div><details><summary>Sources examinées et niveau de preuve</summary><div class="scroll"><table id="sources" class="wrap"><thead><tr><th>Source</th><th>Statut</th><th>Limite / provenance</th></tr></thead><tbody></tbody></table></div></details><p class="foot">La courbe d'achat de l'enchère à venir n'est pas connue à 08 h. Les historiques après enchère peuvent servir de labels ou d'explication, jamais d'entrée contemporaine. Une consommation réalisée inférieure à sa prévision ne mesure pas à elle seule un effacement volontaire.</p></section>
<section class="card"><h2>Références sur les 365 jours · expert non qualifié laissé vide</h2><div class="filters"><label for="zone">Pays</label><select id="zone"></select></div><p id="kpi-support" class="muted"></p><div class="scroll"><table id="kpi"><thead><tr><th>Modèle</th><th>MAE €/MWh</th><th>RMSE €/MWh</th><th>Victoire heure<br>vs Storm</th><th>Victoire MAE jour<br>vs Storm</th><th>Victoire prix moyen jour<br>vs Storm</th><th>MAE prix moyen jour<br>€/MWh</th><th>Prix moyen<br>€/MWh</th><th>Heures évaluées</th></tr></thead><tbody></tbody></table></div><p class="foot">Les références historiques sont figées et les supports affichés, sans sélection de journées favorables. La ligne expert ne recopie jamais les scores de NYX lorsqu'il s'abstient faute de qualification : « — » signifie non évalué, pas zéro erreur. Les journées incomplètes et les changements d'heure suivent les règles du rapport source. Pas de nouveau gain économique revendiqué.</p></section>
<section class="card"><h2>14 septembre 2026 · cas réel, pas scénario fictif</h2><p class="muted">Europe/Paris. Les observations sont explicatives a posteriori ; la colonne expert reste vide quand les données ne permettent pas un prix qualifié.</p><div class="scroll"><table id="cases"><thead><tr><th>Pays / livraison</th><th>Observé €/MWh</th><th>Storm €/MWh</th><th>NYX €/MWh</th><th>Expert €/MWh</th><th>Qualification</th></tr></thead><tbody></tbody></table></div></section>
<section class="card"><div class="badge">DÉMONSTRATION SYNTHÉTIQUE · NON PRÉDICTIVE</div><h2>Pourquoi une demande flexible peut fixer le prix</h2><p>Quand une quantité demandée peut renoncer à acheter à un certain prix, la disposition à payer de cette quantité peut devenir marginale. Les éventuels paliers de 350 / 650 €/MWh sont des hypothèses pédagogiques, pas des seuils estimés sur l'Allemagne ou la Belgique. Un effacement volontaire ne doit pas être confondu avec du délestage involontaire ou une offre absente de notre jeu de données.</p><div class="scroll"><table id="demos"><thead><tr><th>Cas hypothétique</th><th>Demande MW</th><th>Offre MW</th><th>Prix dual €/MWh</th><th>Réduction volontaire MW</th><th>Statut</th></tr></thead><tbody></tbody></table></div><div class="grid"><div><h3>Prix du cas synthétique selon la demande</h3><div id="price-plot"></div><div class="tip" id="price-tip"></div></div><div><h3>Réduction volontaire du cas synthétique</h3><div id="reduction-plot"></div><div class="tip" id="reduction-tip"></div></div></div><p class="foot">Sensibilité du solveur uniquement : toutes les autres hypothèses sont maintenues constantes. Aucun point n'est une prévision de NYX. Les paliers dépendent des offres et des quantités saisies ; sans contrat complet d'entrée, ils ne représentent pas le marché couplé.</p></section>
<section class="card"><h2>Interprétation et protocole de validation</h2><div class="grid"><div><h3>Trois niveaux à ne pas confondre</h3><p><strong>1. Mécanisme :</strong> vérifier sur des exemples contrôlés que des offres d'achat flexibles peuvent déterminer un prix marginal et préserver les bilans.</p><p><strong>2. Proxy de tension :</strong> un modèle ajusté sur les prix peut être utile en prévision, mais ses volumes latents ne prouvent pas des MW réellement effacés.</p><p><strong>3. Demande réelle :</strong> elle exige des observations compatibles, une référence contrefactuelle ou des offres d'achat, et une stratégie d'identification qui traite prix et consommation déterminés conjointement. La prévision de charge peut déjà intégrer une réponse au prix : ne pas la retrancher deux fois.</p></div><div><h3>Conditions d'un futur essai</h3><p>Figer les scénarios et les sources connues avant D−1 08 h, qualifier l'offre, la base de demande et toutes les frontières réseau ; séparer effacement volontaire, report avec rebond et déficit involontaire. Faire un replay chronologique sans lire les prix futurs et conserver les heures manquantes comme manquantes.</p><p>Comparer séparément le candidat brut, son intervention gouvernée et NYX sur les mêmes heures ; publier couverture, abstentions, interventions favorables/défavorables, MAE/RMSE annuelle et pics. Une année déjà explorée et des exemples comme le 14 septembre ne sont pas une validation indépendante. Si aucune heure n'est qualifiée, le résultat est <strong>test bloqué</strong>, pas modèle validé.</p><p>Ce rapport n'entraîne ni n'active de modèle. Les diagnostics restent dans leur dossier d'expérimentation ; les modes opérationnels sont intacts.</p></div></div><details><summary>Provenance et audit</summary><pre id="audit"></pre></details></section>
<noscript>JavaScript est nécessaire pour filtrer les tableaux et afficher les démonstrations.</noscript></main><script id="demand-data" type="application/json">@@DATA@@</script><script>
'use strict';const D=JSON.parse(document.getElementById('demand-data').textContent),$=id=>document.getElementById(id),fmt=(v,n=2)=>Number.isFinite(v)?v.toLocaleString('fr-FR',{minimumFractionDigits:n,maximumFractionDigits:n}):'—',color=k=>getComputedStyle(document.documentElement).getPropertyValue('--'+k).trim();const labels={nuclear_kalman:'NYX opérationnel',nyx:'NYX opérationnel',__storm__:'Storm',storm:'Storm',nyx_demand_response:'Expert réponse de la demande · non qualifié',network_fuel_direct:'Réseau + CGC · direct',nyx_physical_p50:'Réseau + CGC · gouverné',nyx_congestion_calibrated:'Congestion · calibration90 · gouverné'};
function cell(tr,value){const e=document.createElement('td');e.textContent=value;tr.append(e);return e}function table(id,rows,draw){const b=$(id).querySelector('tbody');b.replaceChildren();rows.forEach(r=>{const tr=document.createElement('tr');draw(tr,r);b.append(tr)})}function option(z,text){const e=document.createElement('option');e.value=z;e.textContent=text;$('zone').append(e)}option('ALL','ALL · pays agrégés');D.zones.forEach(z=>option(z,z));
function chart(id,tip,field,label,c){const host=$(id);host.replaceChildren();$(tip).textContent='';const points=D.sensitivity.filter(r=>Number.isFinite(r.demand_mw)&&Number.isFinite(r[field])).slice().sort((a,b)=>a.demand_mw-b.demand_mw);if(!points.length){host.textContent='Aucun scénario synthétique disponible.';return}const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 640 285');svg.setAttribute('role','img');svg.setAttribute('aria-label','Démonstration synthétique : '+label);host.append(svg);let xmin=points[0].demand_mw,xmax=points.at(-1).demand_mw,ymin=Math.min(0,...points.map(r=>r[field])),ymax=Math.max(...points.map(r=>r[field]));if(xmax===xmin)xmax=xmin+1;if(ymax===ymin)ymax=ymin+1;const x=v=>68+(v-xmin)*542/(xmax-xmin),y=v=>228-(v-ymin)*205/(ymax-ymin);function el(tag,attrs,text){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e}for(let i=0;i<5;i++){const v=ymin+(ymax-ymin)*i/4;el('line',{x1:68,x2:610,y1:y(v),y2:y(v),stroke:color('line')});el('text',{x:61,y:y(v)+4,'text-anchor':'end'},fmt(v,0));const w=xmin+(xmax-xmin)*i/4;el('text',{x:x(w),y:249,'text-anchor':'middle'},fmt(w,0))}el('text',{x:338,y:276,'text-anchor':'middle'},'Demande hypothétique (MW)');let path='';points.forEach((r,i)=>{path+=(i?'H'+x(r.demand_mw)+'V'+y(r[field]):'M'+x(r.demand_mw)+','+y(r[field]))});el('path',{d:path,fill:'none',stroke:c,'stroke-width':2});points.forEach(r=>el('circle',{cx:x(r.demand_mw),cy:y(r[field]),r:2,fill:c}));svg.addEventListener('pointermove',e=>{const rect=svg.getBoundingClientRect(),v=xmin+((e.clientX-rect.left)*640/rect.width-68)/542*(xmax-xmin),r=points.reduce((a,b)=>Math.abs(a.demand_mw-v)<Math.abs(b.demand_mw-v)?a:b);$(tip).textContent='Scénario uniquement · demande '+fmt(r.demand_mw,0)+' MW · '+label+' '+fmt(r[field])})}
function render(){const z=$('zone').value,rows=D.kpi_rows.filter(r=>r.zone===z);$('period').textContent=D.period.start_day+' → '+D.period.end_day+' · 365 jours calendaires · cutoff strict 08 h Europe/Paris';$('decision-reason').textContent=D.decision.reason||'Les données ne permettent pas encore une intervention qualifiée.';$('qualified').textContent=fmt(D.decision.qualified_country_hours,0)+' / '+fmt(D.decision.total_country_hours,0);$('interventions').textContent=fmt(D.decision.interventions,0);$('kpi-support').textContent='Références historiques : '+(D.source_report||'rapport source indiqué dans l’audit')+' · les effectifs ci-dessous sont ceux réellement évalués.';table('kpi',rows,(tr,r)=>{if(r.model_id===D.expert_id)tr.className='expert';cell(tr,r.label||labels[r.model_id]||r.model_id);['mae_eur_mwh','rmse_eur_mwh'].forEach(k=>cell(tr,fmt(r[k])));['win_rate_hour_pct','win_rate_day_mae_pct','win_rate_day_mean_price_pct'].forEach(k=>cell(tr,Number.isFinite(r[k])?fmt(r[k])+' %':'—'));['mae_day_mean_price_eur_mwh','mean_price_eur_mwh'].forEach(k=>cell(tr,fmt(r[k])));cell(tr,fmt(r.n_hours,0))});table('cases',D.cases.filter(r=>z==='ALL'||r.zone===z),(tr,r)=>{const stamp=new Intl.DateTimeFormat('fr-FR',{timeZone:'Europe/Paris',dateStyle:'short',timeStyle:'short'}).format(new Date(r.timestamp_utc));cell(tr,r.zone+' · '+stamp);['actual','storm','nyx','expert'].forEach(k=>cell(tr,fmt(r[k])));cell(tr,r.status||'non qualifié')});table('blockers',D.blockers,(tr,r)=>{cell(tr,r.id);cell(tr,r.detail)});table('sources',D.sources,(tr,r)=>{const first=cell(tr,r.name);if(typeof r.url==='string'&&/^https?:\/\//i.test(r.url)){const a=document.createElement('a');a.href=r.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=r.name;first.replaceChildren(a)}cell(tr,r.status);cell(tr,r.detail)});table('demos',D.demos,(tr,r)=>{cell(tr,r.label);['demand_mw','supply_mw','price_eur_mwh','voluntary_reduction_mw'].forEach(k=>cell(tr,fmt(r[k])));cell(tr,r.status||'hypothétique')});chart('price-plot','price-tip','price_eur_mwh','Prix dual (€/MWh)',color('price'));chart('reduction-plot','reduction-tip','voluntary_reduction_mw','Réduction volontaire (MW)',color('volume'));$('audit').textContent=JSON.stringify({snapshot:D.snapshot,period:D.period,source_report:D.source_report,decision:D.decision,audit:D.audit},null,2)}
$('zone').addEventListener('change',render);$('theme').addEventListener('click',()=>{const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';$('theme').textContent=dark?'Mode jour':'Mode nuit';render()});render();
</script></body></html>'''

PAIRED_SECTION = r'''<section class="card"><h2>Expert et références · support strictement apparié</h2><p id="paired-status" class="muted"></p><div class="scroll"><table id="paired-kpi"><thead><tr><th>Modèle</th><th>MAE €/MWh</th><th>RMSE €/MWh</th><th>Victoire heure<br>vs Storm</th><th>Victoire MAE jour<br>vs Storm</th><th>Victoire prix moyen jour<br>vs Storm</th><th>MAE prix moyen jour<br>€/MWh</th><th>Prix moyen<br>€/MWh</th><th>Heures appariées</th></tr></thead><tbody></tbody></table></div><p class="foot">Intersection recalculée avec l'expert : mêmes heures, observations et références pour toutes les lignes de ce tableau. Ne pas comparer directement ces scores de couverture partielle aux références annuelles ci-dessus. Un scénario importé admissible n'est pas une validation indépendante, une preuve de causalité de l'effacement, ni une autorisation de modifier NYX. Les quantiles de scénarios ne sont pas déclarés statistiquement calibrés.</p></section>'''
PAIRED_SCRIPT = r'''function renderPaired(zone){const p=D.paired_expert_evaluation,rows=p?p.rows.filter(r=>r.zone===zone):[];$('paired-kpi').hidden=!p;const n=rows.find(r=>r.model_id===D.expert_id)?.n_hours;$('paired-status').textContent=!p?'Aucune évaluation expert possible : aucune prévision qualifiée sur un support de comparaison.':(Number.isFinite(n)&&n>0?fmt(n,0)+' heures-pays strictement communes sur la fenêtre de 365 jours. Comparaison exploratoire, non validée indépendamment.':'Aucune heure commune pour ce pays : le scénario importé ne permet pas de score apparié.');table('paired-kpi',rows,(tr,r)=>{cell(tr,r.label||(r.model_id===D.expert_id?'Expert réponse de la demande · scénario importé':labels[r.model_id]||r.model_id));['mae_eur_mwh','rmse_eur_mwh'].forEach(k=>cell(tr,fmt(r[k])));['win_rate_hour_pct','win_rate_day_mae_pct','win_rate_day_mean_price_pct'].forEach(k=>cell(tr,Number.isFinite(r[k])?fmt(r[k])+' %':'—'));['mae_day_mean_price_eur_mwh','mean_price_eur_mwh'].forEach(k=>cell(tr,fmt(r[k])));cell(tr,fmt(r.n_hours,0))})}'''
_PAIRED_ANCHOR = '<section class="card"><h2>14 septembre 2026'
_SYNTHETIC_CONNECTIONS = "let path='';points.forEach((r,i)=>{path+=(i?'H'+x(r.demand_mw)+'V'+y(r[field]):'M'+x(r.demand_mw)+','+y(r[field]))});el('path',{d:path,fill:'none',stroke:c,'stroke-width':2});"
TEMPLATE = (TEMPLATE.replace(_PAIRED_ANCHOR, PAIRED_SECTION+_PAIRED_ANCHOR)
    .replace('function render(){', PAIRED_SCRIPT+'\nfunction render(){')
    .replace("const z=$('zone').value,rows=D.kpi_rows.filter(r=>r.zone===z);", "const z=$('zone').value,rows=D.kpi_rows.filter(r=>r.zone===z);renderPaired(z);")
    .replace(_SYNTHETIC_CONNECTIONS, '')
    .replace("r:2,fill:c", "r:4,fill:c")
    .replace('Sensibilité du solveur uniquement :', 'Points testés, sans interpolation. Sensibilité du solveur uniquement :')
    .replace('</style>', '#kpi-support{overflow-wrap:anywhere;word-break:break-word}</style>'))
