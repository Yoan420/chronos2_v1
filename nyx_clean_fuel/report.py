"""Offline, strictly paired clean-fuel residual ablation report.

This module only renders supplied forecasts. It never fits a model, fetches a
benchmark, publishes operational exports, or chooses a production candidate.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


TIMEZONE = "Europe/Paris"
REQUIRED = {"delivery_start_utc", "zone", "actual", "model", "q10", "q50", "q90", "storm_q50", "phase"}


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _metrics(frame: pd.DataFrame, model: str) -> dict[str, Any]:
    actual = frame.actual.to_numpy(float)
    point = frame.q50.to_numpy(float)
    error = point - actual
    storm = frame.storm_q50.to_numpy(float)
    paired = np.isfinite(storm)
    spikes = actual >= 300.0
    result = {
        "model": model, "hours": len(frame), "storm_hours": int(paired.sum()),
        "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)), "mean": float(np.mean(point)),
        "actual_mean": float(np.mean(actual)),
        "win_rate": float(np.mean(np.abs(error[paired]) < np.abs(storm[paired] - actual[paired]) - 1e-9) * 100) if paired.any() else None,
        "tie_rate": float(np.mean(np.isclose(np.abs(error[paired]), np.abs(storm[paired] - actual[paired]), rtol=0, atol=1e-9)) * 100) if paired.any() else None,
        "spike_hours": int(spikes.sum()),
        "spike_mae": float(np.mean(np.abs(error[spikes]))) if spikes.any() else None,
        "spike_rmse": float(np.sqrt(np.mean(error[spikes] ** 2))) if spikes.any() else None,
    }
    return result


def _build_payload(panel: pd.DataFrame, audit: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED.difference(panel.columns)
    if missing:
        raise ValueError(f"Clean-fuel report columns missing: {sorted(missing)}")
    frame = panel.loc[:, sorted(REQUIRED)].copy(deep=True)
    if frame.empty:
        raise ValueError("Clean-fuel report panel is empty")
    if any(pd.Timestamp(value).tzinfo is None for value in frame.delivery_start_utc):
        raise ValueError("Clean-fuel report timestamps must include their timezone")
    frame["delivery_start_utc"] = pd.to_datetime(frame.delivery_start_utc, utc=True, errors="raise")
    if frame.delivery_start_utc.isna().any() or not frame.delivery_start_utc.equals(frame.delivery_start_utc.dt.floor("h")):
        raise ValueError("Clean-fuel report requires finite physical hourly timestamps")
    for key in ("zone", "model", "phase"):
        if frame[key].isna().any() or not frame[key].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"Clean-fuel report requires nonempty text in {key}")
    if not set(frame.phase).issubset({"history", "future"}):
        raise ValueError("Clean-fuel report phases must be history or future")
    if frame.duplicated(["zone", "model", "delivery_start_utc"]).any():
        raise ValueError("Clean-fuel report duplicate zone/model/hour")
    for key in ("actual", "q10", "q50", "q90", "storm_q50"):
        frame[key] = pd.to_numeric(frame[key], errors="raise")
        if np.isinf(frame[key].to_numpy(float)).any():
            raise ValueError(f"Clean-fuel report infinite {key}")
    quantiles = frame[["q10", "q50", "q90"]].to_numpy(float)
    if not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
        raise ValueError("Clean-fuel report nonfinite or crossed quantiles")
    frame["day"] = frame.delivery_start_utc.dt.tz_convert(TIMEZONE).dt.strftime("%Y-%m-%d")
    frame["hour"] = frame.delivery_start_utc.dt.tz_convert(TIMEZONE).dt.hour
    history = frame.loc[frame.phase.eq("history")]
    if history.empty:
        raise ValueError("Clean-fuel report needs historical evaluation forecasts")
    end_day = pd.Timestamp(history.day.max())
    start_day = end_day - pd.Timedelta(days=364)
    partial = bool(audit.get("allow_partial_evaluation", False))
    if partial:
        start_day = max(start_day, pd.Timestamp(history.day.min()))
    history = history.loc[history.day.ge(str(start_day.date()))].copy()
    if not np.isfinite(history.actual.to_numpy(float)).all():
        raise ValueError("Clean-fuel report requires finite historical observations")
    expected = pd.date_range(start_day.tz_localize(TIMEZONE),
                             (end_day + pd.Timedelta(days=1)).tz_localize(TIMEZONE),
                             freq="h", inclusive="left").tz_convert("UTC")
    models = list(dict.fromkeys(frame.model))
    zones = sorted(frame.zone.unique())
    baseline = audit.get("baseline_model", models[0])
    if baseline not in models:
        raise ValueError("Clean-fuel report baseline_model is absent")
    family_mapping = audit.get("baseline_by_model", {})
    if not isinstance(family_mapping, Mapping) or any(
        key not in models or value not in models for key, value in family_mapping.items()
    ):
        raise ValueError("Clean-fuel report baseline_by_model contains an absent model")
    references = {
        model: family_mapping.get(model, model if model in family_mapping.values() else baseline)
        for model in models
    }
    future = frame.loc[frame.phase.eq("future")].copy()
    if not future.empty and future.day.le(str(end_day.date())).any():
        raise ValueError("Clean-fuel report future overlaps evaluated history")
    for zone in zones:
        for phase, subset in (("history", history), ("future", future)):
            zonal = subset.loc[subset.zone.eq(zone)]
            if zonal.empty:
                if phase == "history":
                    raise ValueError(f"{zone}: missing history")
                continue
            reference = zonal.loc[zonal.model.eq(baseline)].sort_values("delivery_start_utc")
            reference_index = pd.DatetimeIndex(reference.delivery_start_utc)
            if phase == "history" and not reference_index.equals(expected):
                raise ValueError(f"{zone}: evaluation must cover the same complete {len(expected)} physical hours")
            if phase == "future":
                future_days = sorted(reference.day.unique())
                complete_future = pd.date_range(pd.Timestamp(future_days[0], tz=TIMEZONE),
                    pd.Timestamp(future_days[-1], tz=TIMEZONE) + pd.DateOffset(days=1),
                    freq="h", inclusive="left").tz_convert("UTC") if future_days else pd.DatetimeIndex([], tz="UTC")
                if not reference_index.equals(complete_future):
                    raise ValueError(f"{zone}: future days must contain every physical hour")
            for model in models:
                candidate = zonal.loc[zonal.model.eq(model)].sort_values("delivery_start_utc")
                if not pd.DatetimeIndex(candidate.delivery_start_utc).equals(reference_index):
                    raise ValueError(f"{zone}/{model}: different comparison hours")
                for field in ("actual", "storm_q50"):
                    if not np.allclose(candidate[field], reference[field], rtol=0, atol=1e-9, equal_nan=True):
                        raise ValueError(f"{zone}/{model}: inconsistent {field} comparator")
    outputs = {}
    for zone in [*zones, "ALL"]:
        selected = history if zone == "ALL" else history.loc[history.zone.eq(zone)]
        scores, hourly, paired_scores, paired_hourly = [], [], [], []
        for model in models:
            rows = selected.loc[selected.model.eq(model)]
            scores.append(_metrics(rows, model))
            for hour, group in rows.groupby("hour"):
                hourly.append({"model": model, "hour": int(hour), "mae": float(np.mean(np.abs(group.q50 - group.actual)))})
            paired = rows.loc[rows.storm_q50.notna()]
            if not paired.empty:
                paired_scores.append(_metrics(paired, model))
                for hour, group in paired.groupby("hour"):
                    paired_hourly.append({"model": model, "hour": int(hour), "mae": float(np.mean(np.abs(group.q50 - group.actual)))})
        comparator = selected.loc[selected.model.eq(baseline)].copy()
        storm = comparator.loc[comparator.storm_q50.notna()].copy()
        if len(storm):
            storm["q50"] = storm.storm_q50
            scores.append(_metrics(storm, "__storm__"))
            paired_scores.append(_metrics(storm, "__storm__"))
            for hour, group in storm.groupby("hour"):
                row = {"model": "__storm__", "hour": int(hour), "mae": float(np.mean(np.abs(group.q50 - group.actual)))}
                hourly.append(row)
                paired_hourly.append(row)
        for rows in (scores, paired_scores):
            lookup = {row["model"]: row for row in rows}
            for row in rows:
                reference_model = references.get(row["model"])
                reference = lookup.get(reference_model)
                row.update(reference_model=reference_model,
                    delta_mae=row["mae"] - reference["mae"] if reference is not None else None,
                    delta_rmse=row["rmse"] - reference["rmse"] if reference is not None else None)
        profiles = []
        if zone != "ALL":
            for model in models:
                source = future.loc[future.zone.eq(zone) & future.model.eq(model)].sort_values("delivery_start_utc")
                base = future.loc[future.zone.eq(zone) & future.model.eq(references[model])].sort_values("delivery_start_utc")
                profiles.extend({"model": model, "time": timestamp.isoformat(),
                    "reference_model": references[model],
                    "label": timestamp.tz_convert(TIMEZONE).strftime("%d/%m %H:%M %z"),
                    "q10": float(low), "q50": float(point), "q90": float(high),
                    "actual": float(observed), "storm": float(storm_price),
                    "correction": float(point - base_price)}
                    for timestamp, low, point, high, observed, storm_price, base_price in zip(
                        source.delivery_start_utc, source.q10, source.q50, source.q90,
                        source.actual, source.storm_q50, base.q50))
        outputs[zone] = {"scores": scores, "hourly": hourly, "future": profiles,
                         "paired_scores": paired_scores, "paired_hourly": paired_hourly,
                         "expected_hours": len(comparator), "storm_hours": len(storm)}
    return _safe({"start_day": str(start_day.date()), "end_day": str(end_day.date()),
        "days": (end_day-start_day).days + 1, "zones": zones, "models": models,
        "baseline_model": baseline, "baseline_by_model": references,
        "labels": dict(audit.get("model_labels", {})),
        "views": outputs, "audit": dict(audit), "diagnostic_only": True,
        "promotion_performed": False})


def render_report(panel: pd.DataFrame, output_path: str | Path, audit: Mapping[str, Any]) -> Path:
    """Validate comparable support, then write one self-contained HTML file."""
    payload = _build_payload(panel, audit)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    title = html.escape(str(audit.get("title", "NYX — correcteur Clean Fuel Costs")))
    document = _TEMPLATE.replace("__TITLE__", title).replace("__PAYLOAD__", encoded)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


_TEMPLATE = r'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>:root{color-scheme:light;--bg:#f2f5f8;--card:#fff;--fg:#13293d;--muted:#516578;--grid:#d8e1e8;--model:#176ba0;--base:#8d55b6;--storm:#c16b06;--good:#0b7d5a;--bad:#bf3b46;--band:#bfd7e8}html[data-theme=dark]{color-scheme:dark;--bg:#111b26;--card:#1b2938;--fg:#e7eef5;--muted:#b6c8d7;--grid:#3c5063;--model:#69c5fa;--base:#d0a8f2;--storm:#ffc36a;--good:#73dec0;--bad:#ffa2aa;--band:#314f67}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}header,.controls{display:flex;align-items:center;gap:15px;flex-wrap:wrap}h1{font-size:27px;margin:0;flex:1}h2{font-size:19px;margin:0 0 15px}p{line-height:1.5}section,.notice{background:var(--card);border:1px solid var(--grid);border-radius:12px;padding:20px;margin:18px 0}.notice{border-left:5px solid var(--storm)}button,select{background:var(--card);color:var(--fg);border:1px solid var(--grid);padding:9px;border-radius:7px}label{color:var(--muted)}.muted{color:var(--muted);font-size:13px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{text-align:right;padding:12px 10px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}th{font-size:12px;color:var(--muted)}.good{color:var(--good)}.bad{color:var(--bad)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid section{min-width:0;margin-top:0}svg{width:100%;height:auto;min-height:230px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.legend{display:flex;gap:17px;flex-wrap:wrap;font-size:13px}.legend span:before{content:'━ ';color:var(--line)}@media(max-width:850px){main{padding:13px}.grid{display:block}h1{font-size:22px}section{padding:13px}}</style></head>
<body><main><header><h1>__TITLE__</h1><button id="theme" type="button">Mode nuit</button></header><p id="period" class="muted"></p>
<div class="notice"><strong>Expérience séparée — aucune promotion en production.</strong><p>Le même correcteur résiduel reçoit en plus les <strong>Clean Gas Costs (CGC)</strong> et <strong>Clean Coal Costs (CCC)</strong>, coûts combustibles + CO₂ exprimés en €/MWh électrique. Il ne s’agit pas d’un expert ajouté après le correcteur. Chronos et la référence opérationnelle ne sont pas remplacés.</p><p class="muted">Comparaison descriptive, pas une validation indépendante après sélection. Les quantiles reçoivent le décalage du correcteur ; leurs intervalles ne sont pas recalibrés par ce rapport. Le seuil « spike » est fixé à un prix observé ≥ 300 €/MWh et sert uniquement à l’évaluation.</p></div>
<div class="controls"><label>Pays <select id="zone"></select></label><label>Candidat <select id="model"></select></label><label>Support <select id="supportMode"><option value="paired">Comparaison Storm — heures communes</option><option value="annual">Année complète — modèles uniquement</option></select></label></div>
<section><h2>Statistics — comparaison des modèles</h2><p id="support" class="muted"></p><div class="scroll"><table><thead><tr><th>Modèle</th><th>Référence de même famille</th><th>MAE €/MWh</th><th>RMSE €/MWh</th><th>Δ MAE vs référence</th><th>Gagnées vs Storm</th><th>Biais €/MWh</th><th>Prix moyen</th><th>Observé moyen</th><th>Heures</th></tr></thead><tbody id="scores"></tbody></table></div><p class="muted">Gain en vert : erreur inférieure à la référence de même famille (autonome ou Kalman). Δ MAE = candidat − référence. Win rate : heures strictement gagnées / heures appariées à Storm ; les égalités ne comptent pas comme victoires. Biais positif = surestimation. Le support annuel conserve toutes les heures des modèles, sans comparer sa MAE à un Storm incomplet.</p></section>
<div class="grid"><section><h2>Erreur absolue moyenne par heure</h2><div id="hourLegend" class="legend"></div><svg id="hourChart" viewBox="0 0 650 290" role="img" aria-label="MAE par heure civile"></svg><p class="muted">Agrégation des mêmes heures physiques. Les deux occurrences de l’heure d’automne restent distinctes dans les métriques.</p></section><section><h2>Performance sur les spikes</h2><div class="scroll"><table><thead><tr><th>Modèle</th><th>Heures ≥ 300</th><th>MAE</th><th>RMSE</th></tr></thead><tbody id="spikes"></tbody></table></div></section></div>
<section><h2>Profil de livraison et ajustement</h2><p id="futureNote" class="muted"></p><div id="futureLegend" class="legend"></div><svg id="futureChart" viewBox="0 0 1200 310" role="img" aria-label="Prix prévu, référence et Storm"></svg><div class="scroll"><table><thead><tr><th>Heure locale</th><th>P10</th><th>P50</th><th>P90</th><th>Storm</th><th>Observé</th><th>Δ P50 vs référence</th></tr></thead><tbody id="future"></tbody></table></div></section>
<section><h2>Construction des coûts propres et hubs</h2><p>CGC = prix du gaz / rendement électrique + coût du CO₂ émis par MWh électrique. CCC = coût du charbon API#2 converti en €/MWh thermique / rendement électrique + coût du CO₂ émis par MWh électrique. Les paramètres et conventions exacts des séries utilisées figurent ci-dessous.</p><p><strong>Le CO₂ est déjà inclus dans les séries CGC/CCC : il n’est pas ajouté une seconde fois.</strong> Les hubs gaz ne sont pas interchangeables ; une série TTF ne constitue pas une observation PEG ou PSV. Aucun hub absent n’est présenté comme utilisé.</p><pre id="costMethod"></pre></section>
<section><details><summary>Sources, paramètres et audit</summary><pre id="audit"></pre></details></section></main>
<script type="application/json" id="data">__PAYLOAD__</script><script>'use strict';const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';const label=m=>D.labels[m]||(m==='__storm__'?'Storm':m),fmt=(v,n=2)=>Number.isFinite(v)?v.toLocaleString('fr-FR',{minimumFractionDigits:n,maximumFractionDigits:n}):'—',css=k=>getComputedStyle(document.documentElement).getPropertyValue('--'+k).trim();function option(select,value,text){let o=document.createElement('option');o.value=value;o.textContent=text;select.appendChild(o)}['ALL',...D.zones].forEach(z=>option($('zone'),z,z==='ALL'?'Tous pays':z));D.models.forEach(m=>option($('model'),m,label(m)));$('model').value=D.models.find(m=>D.baseline_by_model[m]!==m)||D.models[0];function cell(row,text,style){let e=document.createElement('td');e.textContent=text;if(style)e.className=style;row.appendChild(e)}function table(id,rows,fn){$(id).replaceChildren();rows.forEach(r=>{let tr=document.createElement('tr');fn(tr,r);$(id).appendChild(tr)})}function delta(v){return Number.isFinite(v)?v< -1e-9?'good':v>1e-9?'bad':'':''}function svgNode(tag,attrs,text){let e=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,String(v)));if(text!==undefined)e.textContent=text;return e}function legend(id,series){$(id).replaceChildren();series.forEach(s=>{let x=document.createElement('span');x.textContent=s.label;x.style.setProperty('--line',s.color);$(id).appendChild(x)})}function plot(id,labels,series){let el=$(id);el.replaceChildren();let [,,W,H]=el.getAttribute('viewBox').split(' ').map(Number),L=57,R=15,T=23,B=44;let values=series.flatMap(s=>s.values).filter(Number.isFinite);if(!values.length||!labels.length){el.appendChild(svgNode('text',{x:W/2,y:H/2,'text-anchor':'middle',fill:css('muted')},'Données indisponibles'));return}let lo=Math.min(0,...values),hi=Math.max(...values);if(hi===lo)hi=lo+1;let pad=(hi-lo)*.07;hi+=pad;lo-=pad;let x=i=>L+(W-L-R)*i/Math.max(1,labels.length-1),y=v=>T+(hi-v)/(hi-lo)*(H-T-B);for(let i=0;i<=4;i++){let v=lo+(hi-lo)*i/4,yy=y(v);el.appendChild(svgNode('line',{x1:L,x2:W-R,y1:yy,y2:yy,stroke:css('grid')}));el.appendChild(svgNode('text',{x:L-8,y:yy+4,'text-anchor':'end',fill:css('muted'),'font-size':12},fmt(v,0)))}let step=Math.max(1,Math.ceil(labels.length/8));labels.forEach((s,i)=>{if(i%step===0||i===labels.length-1)el.appendChild(svgNode('text',{x:x(i),y:H-15,'text-anchor':'middle',fill:css('muted'),'font-size':11},s))});series.forEach(s=>{let path='',pen=false;s.values.forEach((v,i)=>{if(Number.isFinite(v)){path+=(pen?' L':' M')+x(i)+','+y(v);pen=true}else pen=false});el.appendChild(svgNode('path',{d:path,fill:'none',stroke:s.color,'stroke-width':s.dashed?1.4:2.6,'stroke-dasharray':s.dashed?'5 4':'none'}));s.values.forEach((v,i)=>{if(!Number.isFinite(v))return;let p=svgNode('circle',{cx:x(i),cy:y(v),r:3,fill:s.color});p.appendChild(svgNode('title',{},labels[i]+' · '+s.label+' : '+fmt(v)+' €/MWh'));el.appendChild(p)})})}
function render(){
  const z=$('zone').value,m=$('model').value,v=D.views[z],reference=D.baseline_by_model[m];
  const paired=$('supportMode').value==='paired';
  const rows=paired?v.paired_scores:v.scores.filter(r=>r.model!=='__storm__');
  const hourly=paired?v.paired_hourly:v.hourly.filter(r=>r.model!=='__storm__');
  $('period').textContent=D.start_day+' → '+D.end_day+' · '+D.days+' jours civils · cutoff des sources 08 h · unité €/MWh';
  $('support').textContent=(paired
    ? v.storm_hours.toLocaleString('fr-FR')+' heures-pays strictement communes à tous les modèles et Storm / '+v.expected_hours.toLocaleString('fr-FR')+' attendues. Aucune interpolation du comparateur.'
    : v.expected_hours.toLocaleString('fr-FR')+' heures-pays de l’année complète. Storm n’est pas affiché sur ce support ; le win rate reste calculé sur ses '+v.storm_hours.toLocaleString('fr-FR')+' heures disponibles.')
    +(D.days!==365?' Évaluation partielle de diagnostic, non annuelle.':'');
  table('scores',rows,(tr,r)=>{
    cell(tr,label(r.model));cell(tr,r.reference_model?label(r.reference_model):'—');
    cell(tr,fmt(r.mae),delta(r.delta_mae));cell(tr,fmt(r.rmse),delta(r.delta_rmse));
    cell(tr,fmt(r.delta_mae),delta(r.delta_mae));
    cell(tr,r.model==='__storm__'||!Number.isFinite(r.win_rate)?'—':fmt(r.win_rate)+' %');
    cell(tr,fmt(r.bias));cell(tr,fmt(r.mean));cell(tr,fmt(r.actual_mean));cell(tr,fmt(r.hours,0));
  });
  table('spikes',rows,(tr,r)=>{cell(tr,label(r.model));cell(tr,fmt(r.spike_hours,0));cell(tr,fmt(r.spike_mae));cell(tr,fmt(r.spike_rmse))});
  const specs=[{id:m,color:css('model')},{id:reference,color:css('base')},...(paired?[{id:'__storm__',color:css('storm')}]:[])]
    .filter((s,i,a)=>a.findIndex(x=>x.id===s.id)===i);
  const hs=specs.map(s=>({...s,label:label(s.id),values:Array.from({length:24},(_,h)=>hourly.find(r=>r.model===s.id&&r.hour===h)?.mae??null)}));
  legend('hourLegend',hs);plot('hourChart',Array.from({length:24},(_,h)=>String(h).padStart(2,'0')+'h'),hs);
  const future=v.future.filter(r=>r.model===m),b=v.future.filter(r=>r.model===reference);
  $('futureNote').textContent=z==='ALL'?'Choisir un pays pour le profil de livraison.':future.length
    ?'Écart présenté par rapport à '+label(reference)+' (même famille). Les observations absentes restent vides.'
    :'Aucune journée future fournie : comparaison historique uniquement.';
  const fs=[{label:label(m),color:css('model'),values:future.map(r=>r.q50)},
    {label:label(reference),color:css('base'),values:b.map(r=>r.q50)},
    {label:'Storm',color:css('storm'),values:future.map(r=>r.storm)},
    {label:'P10 / P90 candidat',color:css('muted'),values:future.map(r=>r.q10),dashed:true},
    {label:'P90 candidat',color:css('muted'),values:future.map(r=>r.q90),dashed:true}];
  legend('futureLegend',fs.slice(0,4));plot('futureChart',future.map(r=>r.label),fs);
  table('future',future,(tr,r)=>{cell(tr,r.label);['q10','q50','q90','storm','actual','correction'].forEach(k=>cell(tr,fmt(r[k])))});
  $('audit').textContent=JSON.stringify(D.audit,null,2);
  const method={formules:D.audit.cost_formulas??'Non renseignées dans cet audit.',
    hubs:D.audit.gas_hubs??D.audit.hubs??D.audit.hub_mapping??D.audit.cost_hubs??'Consulter les identifiants des séries dans l’audit des sources.',
    co2_deja_inclus:D.audit.co2_already_included??true};
  $('costMethod').textContent=JSON.stringify(method,null,2);
}
['zone','model','supportMode'].forEach(id=>$(id).addEventListener('change',render));
if(!D.views.ALL.storm_hours)$('supportMode').value='annual';
$('theme').addEventListener('click',()=>{
  const dark=document.documentElement.dataset.theme!=='dark';
  document.documentElement.dataset.theme=dark?'dark':'light';
  $('theme').textContent=dark?'Mode jour':'Mode nuit';render();
});
render();</script></body></html>'''


__all__ = ["render_report"]
