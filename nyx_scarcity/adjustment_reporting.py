"""Offline amplitude diagnostics; no fit, threshold selection or activation."""
from __future__ import annotations

from collections.abc import Mapping
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reporting import _STYLE, _clean, _script_json
from .variant_reporting import LABELS, _align


FRACTIONS = {"proposal25": .25, "proposal50": .5, "proposal100": 1.}
_BASE_COLUMNS = ("zone", "timestamp_utc", "forecast_origin_utc", "sample", "local_day", "local_hour",
                 "local_label", "actual", "forecast", "benchmark_forecast")
_MODEL_COLUMNS = ("spike_probability", "probability_gate", "threshold_eur_mwh", "expert_ready",
                  "raw_correction", "bounded_correction", "candidate_forecast", "applied_correction",
                  "selected_weight", "gate_reason", *FRACTIONS)


def _display_payload(projected: pd.DataFrame, summary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(projected, pd.DataFrame) or "variant_id" not in projected:
        raise ValueError("Projected prices require a long DataFrame with variant_id.")
    if summary.get("fractions") != FRACTIONS:
        raise ValueError("The report requires the declared fixed 25%, 50%, 100% diagnostic fractions.")
    variants = list(summary.get("variants", []))
    if not variants or set(projected.variant_id) != set(variants):
        raise ValueError("Projection rows and summary must identify the same variants.")
    required = {"bounded_correction", "candidate_forecast", "applied_correction", *FRACTIONS}
    if not required.issubset(projected):
        raise ValueError(f"Missing projected/correction columns: {sorted(required.difference(projected.columns))}.")
    for name, fraction in FRACTIONS.items():
        values = pd.to_numeric(projected[name], errors="raise")
        expected = pd.to_numeric(projected.forecast, errors="raise") + fraction*pd.to_numeric(projected.bounded_correction, errors="raise")
        if np.isinf(values).any() or not np.allclose(values, expected, rtol=1e-9, atol=1e-8, equal_nan=True):
            raise ValueError(f"{name} differs from NYX + the declared fraction of the bounded correction.")
    base, frames, _ = _align({key: projected.loc[projected.variant_id.eq(key)].copy() for key in variants})
    keep = base.in_window | base["sample"].eq("live")
    return {"base": base.loc[keep, _BASE_COLUMNS].to_dict("records"),
            "variant_values": {key: {column: frame.loc[keep, column].tolist() for column in _MODEL_COLUMNS}
                               for key, frame in frames.items()},
            "summary": summary, "labels": {key: LABELS.get(key, key) for key in variants}}


_SCRIPT = r"""
const P=JSON.parse(document.getElementById('adjustment-data').textContent),$=id=>document.getElementById(id),finite=x=>typeof x==='number'&&Number.isFinite(x);P.base.forEach((r,i)=>r.i=i);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),fmt=(x,n=2)=>finite(x)?x.toLocaleString('fr-FR',{minimumFractionDigits:n,maximumFractionDigits:n}):'—',zone=()=>$('country').value,variant=()=>$('variant').value,alpha=()=>$('alpha').value,label=k=>P.labels[k]||k,dark=()=>document.documentElement.dataset.theme==='dark';
function plot(id,traces,title){const c=getComputedStyle(document.documentElement);Plotly.react(id,traces,{title:{text:title,font:{size:15}},paper_bgcolor:c.getPropertyValue('--panel').trim(),plot_bgcolor:c.getPropertyValue('--panel').trim(),font:{color:c.getPropertyValue('--ink').trim()},margin:{l:65,r:20,t:55,b:80},xaxis:{automargin:true,gridcolor:c.getPropertyValue('--line').trim()},yaxis:{title:'EUR/MWh',automargin:true,gridcolor:c.getPropertyValue('--line').trim()},legend:{orientation:'h',y:-.25},hovermode:'x unified'},{responsive:true,displaylogo:false});}
function table(headers,rows){return '<div class="table-wrap"><table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function scoreRows(s){const rows=[['NYX figé',s.nyx],['Storm',s.storm]];for(const k of P.summary.variants){rows.push([label(k)+' — prix gouverné retenu',s.variants[k].governed]);rows.push([label(k)+' — proposition '+Math.round(P.summary.fractions[alpha()]*100)+' % NON gouvernée',s.variants[k][alpha()]]);}return table(['Modèle / diagnostic','Heures communes','Prix moyen prévu','MAE','RMSE','Biais','MAE des moyennes journalières'],rows.map(([label,v])=>[esc(label),fmt(v.hours,0),...['mean_forecast_eur_mwh','mae_eur_mwh','rmse_eur_mwh','bias_eur_mwh','daily_mean_mae_eur_mwh'].map(k=>fmt(v[k],3))]));}
function dates(){const rows=P.base.filter(r=>r.zone===zone()),days=[...new Set(rows.map(r=>r.local_day))].sort(),observed=rows.filter(r=>finite(r.actual)).map(r=>r.local_day).sort();$('day').innerHTML=days.map(d=>'<option>'+esc(d)+'</option>').join('');$('day').value=observed.at(-1)||days.at(-1)||'';}
function statistics(){const g=P.summary.by_zone[zone()],w=P.summary.windows[zone()];if(!g)return;$('window').textContent='Du '+(w.requested_start_day||w.start_day||'—')+' au '+(w.end_day||'—')+' : '+fmt(w.represented_days,0)+' jours représentés, '+fmt(g.paired_hours,0)+' heures communes, '+fmt(g.unpaired_window_hours,0)+' heures présentes non appariées. Prix moyen observé sur ce même support : '+fmt(g.annual.nyx.mean_observed_eur_mwh)+' EUR/MWh.';$('annual').innerHTML=scoreRows(g.annual);$('tails').innerHTML=['top_1_percent','top_5_percent'].map(k=>'<h3>'+esc(k.replaceAll('_',' '))+' — seuil observé ex post '+fmt(g.tails[k].threshold_eur_mwh)+' EUR/MWh</h3>'+scoreRows(g.tails[k].scores)).join('');const s=g.interventions[variant()][alpha()];$('risk').innerHTML=table(['Heures avec proposition active','Erreur améliorée','Erreur aggravée','Erreur inchangée','Part bénéfique','Correction moyenne','Somme bénéfices d’erreur','Somme aggravations d’erreur','Solde erreurs'],[[fmt(s.active_hours,0),fmt(s.improved_absolute_error_hours,0),fmt(s.harmed_absolute_error_hours,0),fmt(s.tied_hours,0),finite(s.precision_beneficial)?fmt(100*s.precision_beneficial,1)+' %':'—',fmt(s.mean_correction_eur_mwh),fmt(s.total_benefit_eur_mwh),fmt(s.total_harm_eur_mwh),fmt(s.net_gain_eur_mwh)]]);}
function day(){const rows=P.base.filter(r=>r.zone===zone()&&r.local_day===$('day').value),v=P.variant_values[variant()],x=rows.map(r=>r.local_label),colors={actual:dark()?'#f8f4eb':'#23354c',nyx:dark()?'#92b3ff':'#426bb9',storm:dark()?'#ffca87':'#b97816',governed:dark()?'#79dfb4':'#147858',proposal:dark()?'#f5a9cf':'#ae427d'},traces=[];
for(const [field,key,name] of [['actual','actual','Observé'],['forecast','nyx','NYX figé'],['benchmark_forecast','storm','Storm']])traces.push({x,y:rows.map(r=>r[field]),name,mode:'lines+markers',line:{color:colors[key]},connectgaps:false});traces.push({x,y:rows.map(r=>v.candidate_forecast[r.i]),name:'Prix gouverné retenu',mode:'lines+markers',line:{color:colors.governed,width:3},connectgaps:false});traces.push({x,y:rows.map(r=>v[alpha()][r.i]),name:'Proposition '+Math.round(P.summary.fractions[alpha()]*100)+' % — NON gouvernée',mode:'lines+markers',line:{color:colors.proposal,dash:'dot',width:3},connectgaps:false});plot('priceChart',traces,zone()+' '+$('day').value+' — '+label(variant()));
$('dayNote').textContent=rows.some(r=>r.sample==='live')?'Livraison live : comparaison visuelle seulement, exclue des scores historiques même si les observations sont publiées.':'Rejeu historique, pas un nouvel entraînement. Le prix gouverné retenu reste strictement inchangé.';
$('hourly').innerHTML=table(['Heure locale','p(événement)','Seuil de probabilité','NYX','+ ajustement borné 100 %','Prix proposé 100 %','Prix retenu','Ajustement appliqué','Poids retenu','Observé','Storm','Motif de gouvernance'],rows.map(r=>[esc(r.local_label.slice(11)),fmt(v.spike_probability[r.i],3),fmt(v.probability_gate[r.i],3),fmt(r.forecast),fmt(v.bounded_correction[r.i]),fmt(v.proposal100[r.i]),fmt(v.candidate_forecast[r.i]),fmt(v.applied_correction[r.i]),fmt(v.selected_weight[r.i],3),fmt(r.actual),fmt(r.benchmark_forecast),esc(v.gate_reason[r.i])]));
plot('correctionChart',[{x,y:rows.map(r=>v.bounded_correction[r.i]),name:'Proposition bornée 100 % avant gouvernance',type:'bar',marker:{color:colors.proposal}},{x,y:rows.map(r=>v.applied_correction[r.i]),name:'Correction effectivement retenue',type:'bar',marker:{color:colors.governed}}],'Amplitude proposée et amplitude retenue');}
function render(){statistics();day();}
$('country').innerHTML=Object.keys(P.summary.by_zone).map(z=>'<option>'+esc(z)+'</option>').join('');if(P.summary.by_zone.FR)$('country').value='FR';$('variant').innerHTML=P.summary.variants.map(v=>'<option value="'+esc(v)+'">'+esc(label(v))+'</option>').join('');$('variant').value=P.summary.variants.includes('xgb_unweighted_fixed')?'xgb_unweighted_fixed':P.summary.variants[0];$('alpha').value='proposal25';try{document.documentElement.dataset.theme=localStorage.getItem('nyx-adjustment-theme')||'light';}catch(_){}$('theme').onclick=()=>{document.documentElement.dataset.theme=dark()?'light':'dark';try{localStorage.setItem('nyx-adjustment-theme',document.documentElement.dataset.theme);}catch(_){}render();};$('country').onchange=()=>{dates();render();};$('variant').onchange=render;$('alpha').onchange=render;$('day').onchange=day;dates();if(Object.keys(P.summary.by_zone).length)render();
"""


def render_adjustments(projected_long_df: pd.DataFrame, summary: Mapping[str, Any], audit: Mapping[str, Any],
                       path: str | Path) -> Path:
    """Compare fixed-amplitude proposals with the unchanged governed prices."""
    from plotly.offline import get_plotlyjs

    payload = _display_payload(projected_long_df, summary)
    audit_text = html.escape(json.dumps(_clean(audit), ensure_ascii=False, indent=2, default=str, allow_nan=False))
    document = '''<!doctype html><html lang="fr" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NYX — détection et proposition d’ajustement</title><style>''' + _STYLE + '''</style></head><body>
<header><div><small>DIAGNOSTIC SANS RÉENTRAÎNEMENT · AUCUNE ACTIVATION</small><h1>Détecter un risque et proposer un ajustement de prix</h1></div><button id="theme">Mode nuit / jour</button></header><main>
<section><div class="controls"><label>Pays <select id="country"></select></label><label>Expert <select id="variant"></select></label><label>Proposition à afficher <select id="alpha"><option value="proposal25">25 % — diagnostic</option><option value="proposal50">50 % — diagnostic</option><option value="proposal100">100 % — diagnostic</option></select></label></div>
<p class="warning">Les prix opérationnels et les décisions historiques de gouvernance restent inchangés. Les courbes à 25 %, 50 % et 100 % sont des propositions contrefactuelles NON gouvernées : ce ne sont pas de nouveaux forecasts validés. Aucun alpha gagnant n’est sélectionné automatiquement.</p>
<p>Prix proposé = NYX + fraction fixe × correction bornée de l’expert déjà entraîné. La probabilité détecte une sous-estimation extrême de NYX selon son seuil causal natif ; elle ne mesure pas directement le prix ni un gain économique. La correction bornée à 100 % n’est pas nécessairement retenue : le poids et le motif de gouvernance montrent ce qui a réellement été appliqué.</p>
<p>Année déjà examinée, entraînement progressif 90–365 jours, sources et réseau à 08 h soumis aux audits : pas de preuve prospective/PIT indépendante, pas de garantie de non-régression. Les intervalles de prix ne sont pas extrapolés pour ces nouvelles amplitudes.</p></section>
<section data-report-section="adjustment-hourly"><h2>Détection, proposition et prix retenu — heure par heure</h2><label>Livraison <select id="day"></select></label><p id="dayNote"></p><div id="priceChart" class="chart"></div><div id="correctionChart" class="chart"></div><div id="hourly"></div></section>
<section data-report-section="adjustment-statistics"><h2>Année commune — prix gouvernés et amplitudes non gouvernées</h2><p id="window" class="muted"></p><div id="annual"></div><p>Scores sur les mêmes heures avec observation, NYX, Storm et toutes les variantes/propositions disponibles. Les retours à NYX restent inclus. Les observations ou Storm absents restent vides ; la livraison live n’entre pas dans les scores. Les prix moyens journaliers utilisent uniquement les heures communes déclarées.</p></section>
<section data-report-section="adjustment-risk"><h2>Bénéfice sur l’erreur et risque d’une amplitude accrue</h2><div id="risk"></div><p>La « part bénéfique » est la proportion de propositions actives qui réduisent l’erreur absolue ; ce n’est pas la précision du classificateur. Les sommes de bénéfices et d’aggravations portent sur des erreurs horaires : ce ne sont ni un P&amp;L ni une EVA. Un bénéfice historique ne permet pas d’activer automatiquement cette amplitude.</p></section>
<section data-report-section="adjustment-tails"><h2>Queues de prix — diagnostic ex post</h2><p>Top 1 % / 5 % déterminés par les observations, pour l’analyse uniquement. Ces cohortes n’ont servi ni à fabriquer les propositions ni à choisir leur fraction.</p><div id="tails"></div></section>
<section><h2>Méthodologie et traçabilité</h2><p>Aucun entraînement, aucune modification de feature ou de forecast, aucun seuil optimisé après coup. La sélection 25/50/100 % sert uniquement à inspecter une sensibilité à l’amplitude.</p><details><summary>Audit des sources, du calcul et de la configuration</summary><pre>''' + audit_text + '''</pre></details></section></main>
<script id="adjustment-data" type="application/json">''' + _script_json(payload) + '''</script><script>''' + get_plotlyjs().replace('</script', '<\\/script') + '''</script><script>''' + _SCRIPT + '''</script></body></html>'''
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return target


__all__ = ["render_adjustments"]
