"""Price-expert comparison layered onto the existing offline EVA report."""
from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .reporting import _clean, _columnar, _records, render_report


def _calibration_text(policy: Mapping) -> str:
    model = policy.get("model_audit", {})
    model = model if isinstance(model, Mapping) else {}
    minimum = policy.get("minimum_training_days", model.get("minimum_training_days"))
    maximum = policy.get("training_window_days", model.get("training_window_cap_days"))
    known = isinstance(minimum, (int, float)) and isinstance(maximum, (int, float))
    if known and minimum == maximum:
        text = f"Calibration stricte : au moins {minimum:g} jours historiques requis ; fenêtre plafonnée à {maximum:g} jours. "
    elif known:
        text = f"Calibration progressive : minimum de {minimum:g} jours historiques, fenêtre plafonnée à {maximum:g} jours. "
    else:
        text = "Minimum de calibration et plafond non renseignés : consulter l'audit, sans supposer 90 ou 365 jours disponibles. "
    trained, fallback = model.get("trained_folds"), model.get("fallback_folds")
    low, high = model.get("actual_training_days_min"), model.get("actual_training_days_max")
    if trained == 0:
        text += "Aucun entraînement effectif dans ce replay : 0 bloc entraîné. "
    elif isinstance(trained, (int, float)) and trained > 0:
        text += f"Entraînements effectivement réalisés : {trained:g} blocs"
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            text += f", de {low:g} à {high:g} jours historiques par bloc"
        text += ". "
    else:
        text += "Nombre et profondeur des entraînements effectifs non renseignés. "
    if isinstance(fallback, (int, float)):
        text += f"Blocs en repli : {fallback:g}. "
    text += "Le plafond n'est pas une preuve d'historique disponible : avant l'éligibilité effective, la prévision de base est conservée."
    return text


def _pooled(frame: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
    """Pool country-hour errors, not electricity prices, for the portfolio view."""
    if frame.empty or frame["zone"].eq("PORTFOLIO").any():
        return frame.copy()
    rows = []
    for keys, block in frame.groupby(grouping, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        result = {**dict(zip(grouping, keys)), "zone": "PORTFOLIO"}
        counts = pd.to_numeric(block["hours"], errors="raise")
        result["hours"] = float(counts.sum())
        for name in ("mae_eur_mwh", "rmse_eur_mwh", "bias_eur_mwh"):
            values = pd.to_numeric(block[name], errors="raise")
            if result["hours"] <= 0 or not np.isfinite(values).all():
                result[name] = np.nan
            else:
                squared = values ** 2 if name == "rmse_eur_mwh" else values
                value = float((squared * counts).sum() / result["hours"])
                result[name] = float(np.sqrt(value)) if name == "rmse_eur_mwh" else value
        rows.append(result)
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def _hourly(rows: pd.DataFrame, decisions: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    required = {"timestamp_utc", "zone", "baseline_forecast", "candidate_forecast", "raw_residual_prediction",
                "applied_correction", "selected_weight", "expert_ready", "reason", "forecast_origin_utc"}
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"Price expert report: missing decision fields {sorted(missing)}")
    keys = ["timestamp_utc", "zone"]
    fields = [field for field in ("actual", "q10", "q90", "sample", "paired_eligible", "portfolio_eligible") if field in rows]
    base = rows.loc[rows["strategy"].eq("model") & rows["model"].eq(baseline), keys + ["forecast"] + fields]
    cand = rows.loc[rows["strategy"].eq("model") & rows["model"].eq(candidate), keys + ["forecast"] + fields]
    base = base.rename(columns={name: "baseline_" + name for name in ["forecast"] + fields})
    cand = cand.rename(columns={name: "candidate_" + name for name in ["forecast"] + fields})
    # Candidate/old model prices are taken from the scored archive, not rebuilt.
    scored = base.merge(cand, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not scored["_merge"].eq("both").all():
        raise ValueError("Price expert report: baseline/candidate hourly support differs")
    scored = scored.drop(columns="_merge")
    result = decisions[list(required)].merge(scored, on=keys, how="outer", suffixes=("", "_scored"),
                                             validate="one_to_one", indicator=True)
    if not result["_merge"].eq("both").all():
        raise ValueError("Price expert report: decisions do not match scored hourly support")
    for name in ("baseline_forecast", "candidate_forecast"):
        if not (result[name].eq(result[name + "_scored"]) | (result[name].isna() & result[name + "_scored"].isna())).all():
            raise ValueError(f"Price expert report: {name} disagrees with the scored archive")
    if "baseline_actual" in result and "candidate_actual" in result:
        left, right = result["baseline_actual"], result["candidate_actual"]
        if not (left.eq(right) | (left.isna() & right.isna())).all():
            raise ValueError("Price expert report: baseline/candidate observed labels differ")
    changed = (result["candidate_forecast"] - result["baseline_forecast"]).abs().gt(1e-9)
    finite_prices = np.isfinite(result["candidate_forecast"]) & np.isfinite(result["baseline_forecast"])
    if not np.isclose((result["candidate_forecast"] - result["baseline_forecast"])[finite_prices],
                      result.loc[finite_prices, "applied_correction"], rtol=0, atol=1e-9).all():
        raise ValueError("Price expert report: applied correction disagrees with the candidate price")
    for name in ("q10", "q90"):
        column = "candidate_" + name
        if column in result and result.loc[changed, column].notna().any():
            raise ValueError("Price expert report: modified candidate forecasts cannot retain uncalibrated quantiles")
        if column in result and "baseline_" + name in result:
            left, right = result.loc[~changed, column], result.loc[~changed, "baseline_" + name]
            if not (left.eq(right) | (left.isna() & right.isna())).all():
                raise ValueError("Price expert report: fallback quantiles must match the baseline")
    storm = rows.loc[rows["strategy"].eq("benchmark") & rows["model"].eq(baseline), keys + ["forecast"]]
    result = result.merge(storm.rename(columns={"forecast": "storm_forecast"}), on=keys, how="left", validate="one_to_one")
    result["actual"] = result.get("candidate_actual", result.get("baseline_actual", np.nan))
    result["sample"] = result.get("candidate_sample", "evaluation")
    result["delivery_day"] = pd.to_datetime(result["timestamp_utc"], utc=True).dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    result["raw_price_proposal"] = result["baseline_forecast"] + result["raw_residual_prediction"]
    columns = ["timestamp_utc", "zone", "delivery_day", "sample", "forecast_origin_utc", "baseline_forecast",
               "candidate_forecast", "raw_residual_prediction", "raw_price_proposal", "applied_correction",
               "selected_weight", "expert_ready", "reason", "actual", "storm_forecast"]
    columns += [name for name in ("baseline_q10", "baseline_q90", "candidate_q10", "candidate_q90",
                                  "candidate_paired_eligible", "candidate_portfolio_eligible") if name in result]
    return result[columns].sort_values(keys).reset_index(drop=True)


def render_price_report(rows: pd.DataFrame, metrics: pd.DataFrame, daily: pd.DataFrame,
                        breakdowns: Mapping[str, pd.DataFrame], audit: dict,
                        destination: str | Path, *, forecast_metrics: pd.DataFrame,
                        forecast_daily: pd.DataFrame, decisions: pd.DataFrame,
                        governance: pd.DataFrame) -> Path:
    """Create a separate price forecast/EVA report without changing old reports."""
    price_expert = audit.get("price_expert")
    if not isinstance(price_expert, Mapping):
        raise ValueError("Price expert report requires audit.price_expert")
    baseline = str(price_expert.get("baseline_model", "nuclear_kalman"))
    candidate = str(price_expert.get("candidate_model", "nuclear_kalman_extreme"))
    destination = Path(destination)
    if destination.suffix.lower() != ".html":
        raise ValueError("Price expert report destination must be .html")
    hourly = _hourly(rows, decisions, baseline, candidate)
    forecast_metrics = _pooled(forecast_metrics, ["model", "subset"])
    forecast_daily = _pooled(forecast_daily, ["model", "delivery_day"])
    payload = {"hourly": _columnar(hourly), "metrics": _records(forecast_metrics),
               "daily": _columnar(forecast_daily), "audit": _clean(price_expert),
               "calibration_text": _calibration_text(price_expert),
               "governance_rows": len(governance),
               "governance_columns": list(governance.columns)}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for original, replacement in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        encoded = encoded.replace(original, replacement)
    clean_audit = {key: value for key, value in audit.items() if key != "extreme_policy"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".price_render_", dir=destination.parent) as temporary:
        base_path = render_report(rows, metrics, daily, breakdowns, clean_audit, Path(temporary) / "base.html")
        document = base_path.read_text(encoding="utf-8")
    marker = '<section><h2>Statistics — décision économique</h2>'
    if marker not in document or "</body></html>" not in document:
        raise ValueError("Price expert report: incompatible base report template")
    section = _SECTION.replace("__CALIBRATION_TEXT__", escape(payload["calibration_text"]))
    document = document.replace(marker, section + marker, 1)
    extension = '<script id="price-expert-payload" type="application/json">' + encoded + '</script><script>' + _SCRIPT + '</script>'
    document = document.replace("</body></html>", extension + "</body></html>", 1)
    document = document.replace('<title>Economic Value Added — laboratoire Chronos-2</title>',
                                '<title>Expert de prix extrêmes — prévisions et valeur économique</title>', 1)
    destination.write_text(document, encoding="utf-8")
    return destination


_SECTION = r'''<section id="priceExpertSection">
<h2>Expert de prix extrêmes — nouveau candidat de prévision</h2>
<p><strong>Cette expérience modifie le prix prévu, pas directement la position.</strong> Le candidat ajoute à la prévision nucléaire + Kalman une correction résiduelle pondérée par la gouvernance. La même règle économique transforme ensuite les prévisions de chaque modèle en positions simulées.</p>
<p id="priceCalibration" class="small">__CALIBRATION_TEXT__</p>
<p class="small">La période a déjà été examinée : diagnostic de recherche répété, pas année de validation indépendante. Le cutoff cible reste 08 h ; le PIT historique n'est pas certifié. Cette couche ne fournit pas une attribution causale du rôle des variables ni du prix passé de la prévision de base.</p>
<div class="cards"><div class="card"><div class="label">Gain MAE vs base — période entière</div><div id="priceMaeGain" class="value">—</div><div class="sub">Démarrage inclus ; MAE de base − candidat (EUR/MWh).</div></div><div class="card"><div class="label">Gain net de P&amp;L vs base</div><div id="pricePnlGain" class="value">—</div><div id="pricePairedNote" class="sub"></div></div><div class="card"><div class="label">Heures appariées de prix modifié</div><div id="priceChanged" class="value">—</div><div class="sub">Heures-pays pour le portefeuille.</div></div><div class="card"><div class="label">Poids moyen de la correction</div><div id="priceWeight" class="value">—</div><div class="sub">Calculé sur les heures d'évaluation appariées.</div></div></div>
<p id="priceGates" class="small"></p>
<div class="controls"><label>Périmètre des erreurs <select id="priceSubset"><option value="all">Toute l'année, démarrage inclus</option><option value="high">Prix hauts</option><option value="normal">Prix normaux</option><option value="negative">Prix non positifs (≤ 0)</option><option value="expert_ready">Expert disponible — diagnostic complémentaire</option></select></label></div>
<div id="priceMetrics" class="scroll"></div>
<p class="small">MAE, RMSE et biais en EUR/MWh. Biais = prévision − observé. La vue portefeuille regroupe les erreurs par heure-pays, pas les prix en un prix unique. Le sous-ensemble « expert disponible » est complémentaire : il ne remplace jamais le bilan sur les 365 jours avec démarrage et replis.</p>
<div id="priceMaeCurve" class="plot"></div><div id="pricePnlCurve" class="plot"></div><div id="priceGovernance" class="plot"></div>
<h3>Journée sélectionnée dans la table horaire</h3><p id="priceDayNote" class="small"></p><div id="priceDayForecast" class="plot"></div><div id="priceDayTable" class="scroll"></div>
<p class="small">Prix proposé brut = prévision de base + résiduel brut estimé. La prévision candidate utilise seulement la correction effectivement autorisée. Sur les heures corrigées, P10/P90 sont indisponibles : décaler mécaniquement les intervalles ne constituerait pas une calibration. Les quantiles de base ne sont conservés que lorsque le candidat reste inchangé. Un observé non publié reste vide.</p>
<details><summary>Calibration, gouvernance et limites de l'expérience</summary><pre id="priceExpertAudit"></pre></details>
</section>'''


_SCRIPT = r'''
'use strict';
const priceData=JSON.parse(document.getElementById('price-expert-payload').textContent);
priceData.hourly=unpack(priceData.hourly);priceData.daily=unpack(priceData.daily);
const priceExpert=priceData.audit;
const priceBaseline=priceExpert.baseline_model||'nuclear_kalman',priceCandidate=priceExpert.candidate_model||'nuclear_kalman_extreme';
const priceModelNames={[priceBaseline]:'Nucléaire + Kalman · base',[priceCandidate]:'Nucléaire + Kalman + expert de prix',storm:'Storm'};
modelLabels[priceCandidate]=priceModelNames[priceCandidate];
for(const item of selected.model.options){if(item.value===priceCandidate)item.label=priceModelNames[priceCandidate]}
document.getElementById('priceExpertAudit').textContent=JSON.stringify({...priceExpert,governance_archived_rows:priceData.governance_rows,governance_archived_columns:priceData.governance_columns},null,2);
document.getElementById('priceCalibration').textContent=priceData.calibration_text;
document.getElementById('candidateProtocol').textContent='Strategy 2 — candidat prix : prévision de base + correction résiduelle gouvernée, puis application de la même règle de position que les autres modèles. Les paramètres et profondeurs effectives de calibration figurent dans la section prix ; aucun historique absent n’est inventé. Année déjà étudiée : diagnostic répété, pas test final indépendant.';
function pricePaired(){const value=priceExpert.baseline_comparison_paired;return value===true||(value&&typeof value==='object'&&value[selected.zone.value]===true)}
function priceScope(rows){return rows.filter(r=>r.zone===selected.zone.value)}
function priceNumber(id,value,suffix='',decimals=2){const node=document.getElementById(id);node.textContent=finite(value)?fmt(value,decimals)+suffix:'—';node.className='value '+signClass(value)}
function priceSummary(){const entries=Array.isArray(priceExpert.summary)?priceExpert.summary:Object.entries(priceExpert.summary||{}).map(([zone,value])=>({zone,...value}));return entries.find(r=>r.zone===selected.zone.value)||{}}
function priceModelColor(model){return model===priceBaseline?(dark()?'#bda9ff':'#7956b0'):model===priceCandidate?strategyColor('model'):strategyColor('benchmark')}
function drawPrice(){
 const all=priceScope(priceData.metrics).filter(r=>r.subset==='all'),base=all.find(r=>r.model===priceBaseline),candidate=all.find(r=>r.model===priceCandidate),paired=pricePaired();
 const sameErrors=base&&candidate&&base.hours===candidate.hours,maeGain=sameErrors&&finite(base.mae_eur_mwh)&&finite(candidate.mae_eur_mwh)?base.mae_eur_mwh-candidate.mae_eur_mwh:null;
 priceNumber('priceMaeGain',maeGain,' €/MWh',3);
 const economic=priceScope(data.metrics).filter(r=>r.strategy==='model'),eb=economic.find(r=>r.model===priceBaseline),ec=economic.find(r=>r.model===priceCandidate);
 const pnlGain=paired&&finite(eb?.pnl_net_eur)&&finite(ec?.pnl_net_eur)?ec.pnl_net_eur-eb.pnl_net_eur:null;
 priceNumber('pricePnlGain',pnlGain,' €',0);
 document.getElementById('pricePairedNote').textContent=paired?'Candidat − base, heures appariées et coûts identiques.':'Gain indisponible : appariement économique non attesté.';
 const scope=selected.zone.value,scopeHours=priceData.hourly.filter(r=>r.sample==='evaluation'&&(scope==='PORTFOLIO'||r.zone===scope)),matched=scopeHours.filter(r=>r.candidate_paired_eligible===true&&(scope!=='PORTFOLIO'||r.candidate_portfolio_eligible===true));
 const changed=matched.filter(r=>finite(r.applied_correction)&&Math.abs(r.applied_correction)>1e-9).length,weights=matched.map(r=>r.selected_weight).filter(finite);
 document.getElementById('priceChanged').textContent=matched.length?fmt(changed,0):'—';document.getElementById('priceWeight').textContent=weights.length?pct(weights.reduce((a,b)=>a+b,0)/weights.length):'—';
 const summary=priceSummary(),gate=value=>value===true?'oui':value===false?'non':'indisponible';
 document.getElementById('priceGates').textContent=`Diagnostic annuel — MAE non dégradée : ${gate(summary.annual_non_regression)} ; gain économique vs base : ${gate(summary.annual_eva_gain_pass)}. Ces contrôles de recherche ne constituent pas une autorisation de production.`;
 const subset=document.getElementById('priceSubset').value||'all',tableRows=priceScope(priceData.metrics).filter(r=>r.subset===subset);
 replaceTable('priceMetrics',['Prévision','Périmètre','Heures','MAE','RMSE','Biais'],tableRows.map(r=>[priceModelNames[r.model]||r.model,r.subset,fmt(r.hours,0),fmt(r.mae_eur_mwh,3),fmt(r.rmse_eur_mwh,3),fmt(r.bias_eur_mwh,3)]));
 const forecastDays=priceScope(priceData.daily),baselineDays=new Map(forecastDays.filter(r=>r.model===priceBaseline).map(r=>[r.delivery_day,r])),candidateDays=forecastDays.filter(r=>r.model===priceCandidate).sort((a,b)=>a.delivery_day.localeCompare(b.delivery_day));
 const maeGains=candidateDays.map(r=>{const b=baselineDays.get(r.delivery_day);return b&&r.hours===b.hours&&finite(b.mae_eur_mwh)&&finite(r.mae_eur_mwh)?b.mae_eur_mwh-r.mae_eur_mwh:null});
 plot('priceMaeCurve',[{x:candidateDays.map(r=>r.delivery_day),y:maeGains,name:'MAE base − candidat',type:'bar',marker:{color:maeGains.map(v=>finite(v)&&v<0?(dark()?'#ff96a4':'#b3394c'):(dark()?'#68d5bd':'#00766c'))}}],'Gain de MAE par journée — démarrage inclus','EUR/MWh');
 const ed=priceScope(data.daily).filter(r=>r.strategy==='model'),bd=new Map(ed.filter(r=>r.model===priceBaseline).map(r=>[r.delivery_day,r.cumulative_pnl_eur])),cd=ed.filter(r=>r.model===priceCandidate).sort((a,b)=>a.delivery_day.localeCompare(b.delivery_day));
 plot('pricePnlCurve',[{x:cd.map(r=>r.delivery_day),y:cd.map(r=>paired&&finite(r.cumulative_pnl_eur)&&finite(bd.get(r.delivery_day))?r.cumulative_pnl_eur-bd.get(r.delivery_day):null),name:'P&L candidat − base',type:'scatter',mode:'lines',connectgaps:false,line:{color:strategyColor('model'),width:3}}],'Gain net de P&L cumulé vs base — échantillon apparié','EUR');
 const daysMap=new Map(scopeHours.map(r=>[r.delivery_day,[]]));for(const r of matched)daysMap.get(r.delivery_day).push(r);const wd=[...daysMap].sort((a,b)=>a[0].localeCompare(b[0]));
 const wl=layout('Gouvernance de la correction — heures appariées','Poids moyen (0–1)');wl.yaxis.range=[0,1];wl.yaxis2={title:scope==='PORTFOLIO'?'Heures-pays corrigées':'Heures corrigées',overlaying:'y',side:'right',showgrid:false,rangemode:'tozero'};
 Plotly.react('priceGovernance',[{x:wd.map(r=>r[0]),y:wd.map(r=>r[1].length?r[1].filter(x=>Math.abs(x.applied_correction)>1e-9).length:null),name:'Prix modifiés',type:'bar',yaxis:'y2',opacity:0.4,marker:{color:priceModelColor(priceBaseline)}},{x:wd.map(r=>r[0]),y:wd.map(r=>{const w=r[1].map(x=>x.selected_weight).filter(finite);return w.length?w.reduce((a,b)=>a+b,0)/w.length:null}),name:'Poids retenu',type:'scatter',mode:'lines',connectgaps:false,line:{color:strategyColor('model'),width:2}}],wl,{responsive:true,displaylogo:false});
 drawPriceDay();
}
function drawPriceDay(){
 const rs=priceData.hourly.filter(r=>r.zone===selected.traderZone.value&&r.delivery_day===selected.day.value).sort((a,b)=>a.timestamp_utc.localeCompare(b.timestamp_utc));
 const changed=rs.filter(r=>finite(r.applied_correction)&&Math.abs(r.applied_correction)>1e-9).length,pending=rs.filter(r=>!finite(r.actual)).length;
 document.getElementById('priceDayNote').textContent=`${selected.traderZone.value} · ${selected.day.value} · ${rs.length} heures physiques · ${changed} prix corrigés · ${pending} observations en attente. Courbe brute masquée au départ ; cliquer sa légende pour l'afficher.`;
 const names={actual:'Observé',baseline_forecast:'Base nucléaire + Kalman',candidate_forecast:'Candidat prix extrêmes',storm_forecast:'Storm',raw_price_proposal:'Proposition brute avant gouvernance'},colors={actual:dark()?'#ffc67d':'#a56919',baseline_forecast:priceModelColor(priceBaseline),candidate_forecast:priceModelColor(priceCandidate),storm_forecast:strategyColor('benchmark'),raw_price_proposal:dark()?'#f299c2':'#a73471'};
 plot('priceDayForecast',Object.keys(names).map(key=>({x:rs.map(r=>r.timestamp_utc),y:rs.map(r=>r[key]),name:names[key],type:'scatter',mode:'lines',visible:key==='raw_price_proposal'?'legendonly':true,connectgaps:false,line:{color:colors[key],width:key==='actual'?3:2,dash:key==='raw_price_proposal'?'dot':'solid'}})),'Prévisions et correction du jour — axe UTC','EUR/MWh');
 replaceTable('priceDayTable',['Heure locale','Base','Résiduel brut','Prix proposé brut','Correction appliquée','Candidat','Storm','Observé','Poids','Expert prêt','P10 candidat','P90 candidat','Décision / repli'],rs.map(r=>[hourLabel(r),fmt(r.baseline_forecast),fmt(r.raw_residual_prediction),fmt(r.raw_price_proposal),fmt(r.applied_correction),fmt(r.candidate_forecast),fmt(r.storm_forecast),fmt(r.actual),pct(r.selected_weight),r.expert_ready===true?'Oui':r.expert_ready===false?'Non':'—',fmt(r.candidate_q10),fmt(r.candidate_q90),r.reason??'—']));
}
const priceOriginalDraw=draw;draw=function(){priceOriginalDraw();drawPrice()};
selected.model.onchange=draw;
const priceOriginalDayChange=selected.day.onchange;selected.day.onchange=function(){priceOriginalDayChange();drawPriceDay()};
const priceOriginalTraderChange=selected.traderZone.onchange;selected.traderZone.onchange=function(){priceOriginalTraderChange();drawPriceDay()};
document.getElementById('priceSubset').onchange=drawPrice;
if(models.includes(priceCandidate))selected.model.value=priceCandidate;
draw();
'''
