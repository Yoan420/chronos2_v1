"""Frozen, common-support comparison of governed scarcity expert variants.

Native classifier targets can differ. Native calibration is therefore kept
separate from diagnostic ranking against the common residual event >=50.
No threshold is selected retrospectively, and no forecast is modified here.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .reporting import TIMEZONES, _STYLE, _clean, _prepare, _score, _script_json


LABELS = {"nuclear_kalman": "Nuclear Kalman / NYX figé", "storm": "Storm",
          "hgb_v1": "HGB original — snapshot figé", "xgb_unweighted_fixed": "XGB non pondéré — seuil fixe",
          "xgb_weighted_fixed": "XGB pondéré — seuil fixe", "xgb_unweighted_dwt": "XGB non pondéré — seuil causal dynamique",
          "xgb_weighted_dwt": "XGB pondéré — seuil causal dynamique"}
_SHARED = ("actual", "forecast", "benchmark_forecast", "q10", "q90")
_BASE_PAYLOAD = ("zone", "timestamp_utc", "forecast_origin_utc", "sample", "local_day", "local_hour",
                 "local_label", "actual", "forecast", "benchmark_forecast", "q10", "q90")
_VARIANT_PAYLOAD = ("candidate_forecast", "candidate_q10", "candidate_q90", "spike_probability",
                    "applied_correction", "selected_weight", "expert_ready", "threshold_eur_mwh", "probability_gate")


def _align(predictions: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], np.ndarray]:
    if not isinstance(predictions, Mapping) or not predictions or "hgb_v1" not in predictions:
        raise ValueError("A nonempty variant mapping including the frozen hgb_v1 control is required.")
    if any(not isinstance(key, str) or not key or key in {"nuclear_kalman", "storm"} for key in predictions):
        raise ValueError("Variant identifiers must be nonempty strings distinct from baseline/Storm names.")
    frames = {key: _prepare(value).set_index(["zone", "timestamp_utc"], drop=False)
              for key, value in predictions.items()}
    index = frames["hgb_v1"].index
    for frame in frames.values():
        index = index.union(frame.index)
    index = index.sort_values()
    if index.empty:
        raise ValueError("There are no frozen forecasts to compare.")
    aligned = {key: frame.reindex(index).reset_index(drop=True) for key, frame in frames.items()}
    base = pd.DataFrame({"zone": index.get_level_values("zone"), "timestamp_utc": index.get_level_values("timestamp_utc")})
    for column in ("forecast_origin_utc", "sample", "local_day", "local_hour", "local_label", *_SHARED):
        reference = None
        for key, frame in aligned.items():
            values = frame[column]
            if reference is None:
                reference = values.copy()
                continue
            common = reference.notna() & values.notna()
            if column in _SHARED:
                equal = np.allclose(reference.loc[common].to_numpy(float), values.loc[common].to_numpy(float), rtol=0, atol=1e-9)
            else:
                equal = reference.loc[common].eq(values.loc[common]).all()
            if not equal:
                raise ValueError(f"{key}: frozen source field {column} differs between variants.")
            reference = reference.where(reference.notna(), values)
        base[column] = reference
    base["in_window"] = False
    for zone, group in base.loc[base["sample"].eq("evaluation")].groupby("zone"):
        first = str((pd.Timestamp(group.local_day.max()) - pd.Timedelta(days=364)).date())
        base.loc[group.index, "in_window"] = group.local_day.ge(first)
    # This mask is progressively reduced. A NumPy view would accidentally
    # shrink the declared evaluation window itself and hide missing coverage.
    common = base.in_window.to_numpy(dtype=bool, copy=True)
    for frame in aligned.values():
        common &= frame["sample"].eq("evaluation").to_numpy()
        common &= np.isfinite(frame[["actual", "forecast", "benchmark_forecast", "candidate_forecast"]].to_numpy(float)).all(axis=1)
    for key, frame in aligned.items():
        frame["zone"], frame["timestamp_utc"] = base.zone, base.timestamp_utc
        frame["local_day"], frame["local_hour"] = base.local_day, base.local_hour
        frame["expert_ready"] = frame.expert_ready.astype("boolean").fillna(False).astype(bool)
    return base, aligned, common


def _ranking(events: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    n, positives = len(events), int(events.sum())
    both_classes = bool(positives and positives < n)
    average_precision = float(average_precision_score(events, probabilities)) if positives else None
    roc_auc = float(roc_auc_score(events, probabilities)) if both_classes else None
    if positives:
        precision, recall, _ = precision_recall_curve(events, probabilities)
        take = np.unique(np.linspace(0, len(precision)-1, min(201, len(precision))).astype(int))
        pr = {"precision": precision[take].tolist(), "recall": recall[take].tolist()}
    else:
        pr = {"precision": [], "recall": []}
    if both_classes:
        fpr, tpr, _ = roc_curve(events, probabilities)
        take = np.unique(np.linspace(0, len(fpr)-1, min(201, len(fpr))).astype(int))
        roc = {"fpr": fpr[take].tolist(), "tpr": tpr[take].tolist()}
    else:
        roc = {"fpr": [], "tpr": []}
    return {"hours": n, "positives": positives, "negatives": n-positives,
            "prevalence": positives/n if n else None, "average_precision": average_precision, "roc_auc": roc_auc,
            "pr_curve": pr, "roc_curve": roc, "curve_display_max_points": 201,
            "scores_calculated_on_all_rows": True, "operating_threshold_selected": False}


def _native_classification(frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame.loc[frame.expert_ready & frame.spike_probability.notna()
                         & frame.threshold_eur_mwh.notna() & frame.probability_gate.notna()]
    probabilities = selected.spike_probability.to_numpy(float)
    events = (selected.actual - selected.forecast).ge(selected.threshold_eur_mwh).to_numpy()
    positive = probabilities > selected.probability_gate.to_numpy(float)
    tp, fp = int((positive & events).sum()), int((positive & ~events).sum())
    fn, tn = int((~positive & events).sum()), int((~positive & ~events).sum())
    thresholds = selected.threshold_eur_mwh
    bins = []
    for ordinal in range(10):
        lower, upper = ordinal/10, (ordinal+1)/10
        mask = (probabilities >= lower) & ((probabilities < upper) if ordinal < 9 else (probabilities <= upper))
        if mask.any():
            bins.append({"hours": int(mask.sum()), "mean_probability": float(probabilities[mask].mean()),
                         "event_frequency": float(events[mask].mean())})
    return {**_ranking(events, probabilities), "target": "actual - frozen NYX forecast >= own causal threshold_eur_mwh",
            "cross_variant_calibration_comparable": False,
            "threshold_min": float(thresholds.min()) if len(selected) else None,
            "threshold_median": float(thresholds.median()) if len(selected) else None,
            "threshold_max": float(thresholds.max()) if len(selected) else None,
            "threshold_distinct_values": int(thresholds.nunique()),
            "brier": float(np.mean((probabilities-events)**2)) if len(selected) else None,
            "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None,
            "false_positive_rate": fp/(fp+tn) if fp+tn else None,
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "gates": sorted(float(value) for value in selected.probability_gate.unique()),
            "gate_rule": "strictly greater than supplied causal probability_gate; no ex-post tuning", "reliability_bins": bins}


def _point_scores(base: pd.DataFrame, frames: Mapping[str, pd.DataFrame], mask: np.ndarray) -> dict[str, Any]:
    selected = base.loc[mask]
    scores = {"nuclear_kalman": _score(selected, "forecast"), "storm": _score(selected, "benchmark_forecast")}
    for key, frame in frames.items():
        score = _score(frame.loc[mask], "candidate_forecast")
        scores[key] = score
    for key in frames:
        scores[key]["gain_vs_hgb_mae_eur_mwh"] = (
            scores["hgb_v1"]["mae_eur_mwh"] - scores[key]["mae_eur_mwh"] if len(selected) else None)
        scores[key]["gain_vs_nyx_mae_eur_mwh"] = (
            scores["nuclear_kalman"]["mae_eur_mwh"] - scores[key]["mae_eur_mwh"] if len(selected) else None)
    return scores


def _zone_summary(base: pd.DataFrame, frames: Mapping[str, pd.DataFrame], common: np.ndarray,
                  population: np.ndarray) -> dict[str, Any]:
    selected = common & population
    common_ready = selected.copy()
    for frame in frames.values():
        common_ready &= frame.expert_ready.to_numpy(bool) & frame.spike_probability.notna().to_numpy()
    common_event = (base.actual - base.forecast).ge(50).to_numpy()
    ranking = {key: {**_ranking(common_event[common_ready], frame.spike_probability.loc[common_ready].to_numpy(float)),
                     "native_target_matches_common_on_all_rows": bool(frame.threshold_eur_mwh.loc[common_ready].eq(50).all()) if common_ready.any() else None}
               for key, frame in frames.items()}
    native, intervention, trained = {}, {}, {}
    for key, frame in frames.items():
        native[key] = _native_classification(frame.loc[selected])
        active = selected & frame.applied_correction.notna().to_numpy() & frame.applied_correction.ne(0).to_numpy()
        changed_error = (frame.candidate_forecast - base.actual).abs() - (base.forecast - base.actual).abs()
        own_event = (base.actual-base.forecast).ge(frame.threshold_eur_mwh)
        own_known = active & frame.threshold_eur_mwh.notna().to_numpy()
        intervention[key] = {"active_hours": int(active.sum()),
            "inactive_hours": int((selected & frame.applied_correction.eq(0).to_numpy()).sum()),
            "unknown_intervention_hours": int((selected & frame.applied_correction.isna().to_numpy()).sum()),
            "non_event_interventions_native_threshold": int((own_known & ~own_event.to_numpy()).sum()),
            "non_event_interventions_common_error50": int((active & ~common_event).sum()),
            "worsened_absolute_error_hours": int((active & changed_error.gt(1e-9).to_numpy()).sum()),
            "improved_absolute_error_hours": int((active & changed_error.lt(-1e-9).to_numpy()).sum()),
            "mean_added_absolute_error_on_active_eur_mwh": float(changed_error.loc[active].mean()) if active.any() else None,
            "all_models_on_this_variant_active_subset": _point_scores(base, frames, active),
            "warning": "active subsets differ across variants; non-event does not imply economic harm; use the full annual comparison"}
        ready = population & base.in_window.to_numpy(bool) & frame.expert_ready.to_numpy(bool)
        trained[key] = {"ready_hours_in_window": int(ready.sum()), "ready_hours_on_common_point_support": int((ready & common).sum()),
                        "first_ready_day": str(base.local_day.loc[ready].min()) if ready.any() else None,
                        "common_ready_classification_hours": int(common_ready.sum()),
                        "baseline_fallback_hours_in_common_support": int((selected & frame.candidate_forecast.eq(frame.forecast).to_numpy()).sum())}
    tails = {}
    for percent, q in ((1, .99), (5, .95)):
        threshold = float(base.actual.loc[selected].quantile(q)) if selected.any() else None
        mask = selected & base.actual.ge(threshold).to_numpy() if threshold is not None else selected
        tails[f"top_{percent}_percent"] = {"definition": "ex_post_observed_price_quantile_on_common_support",
            "threshold_eur_mwh": threshold, "ties_may_increase_fraction": True,
            "used_for_training_or_governance": False, "scores": _point_scores(base, frames, mask)}
    return {"represented_window_hours": int((population & base.in_window.to_numpy(bool)).sum()),
            "paired_hours": int(selected.sum()), "unpaired_window_hours": int((population & base.in_window.to_numpy(bool) & ~common).sum()),
            "annual": _point_scores(base, frames, selected), "tails": tails, "interventions": intervention,
            "trained_coverage": trained, "classification_native": native,
            "classification_common_ranking": {"target": "actual - frozen NYX forecast >= 50 EUR/MWh",
                "mask": "common point rows AND every variant expert_ready with finite probability",
                "hours": int(common_ready.sum()), "variants": ranking, "brier_comparison_provided": False,
                "interpretation": "ROC/AP rank scores against one common event; probabilities trained on different native targets are not assumed calibrated for this common target"}}


def _aggregates(base: pd.DataFrame, frames: Mapping[str, pd.DataFrame], common: np.ndarray, group: str) -> list[dict[str, Any]]:
    rows = []
    window = base.loc[base.in_window]
    for (zone, period), indices in window.groupby(["zone", group]).groups.items():
        mask = np.zeros(len(base), dtype=bool)
        mask[np.asarray(indices, dtype=int)] = True
        selected = mask & common
        scores = _point_scores(base, frames, selected)
        for model, score in scores.items():
            rows.append({"zone": zone, group: int(period) if group == "local_hour" else str(period), "model": model,
                         "represented_hours": int(mask.sum()), "paired_hours": int(selected.sum()),
                         "mean_forecast_eur_mwh": score["mean_forecast_eur_mwh"], "mean_observed_eur_mwh": score["mean_observed_eur_mwh"],
                         "mae_eur_mwh": score["mae_eur_mwh"], "bias_eur_mwh": score["bias_eur_mwh"],
                         "gain_vs_hgb_mae_eur_mwh": score.get("gain_vs_hgb_mae_eur_mwh")})
    return rows


def build_comparison(predictions: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    base, frames, common = _align(predictions)
    windows = {}
    for zone, group in base.loc[base.in_window].groupby("zone"):
        end = pd.Timestamp(group.local_day.max())
        first = end - pd.Timedelta(days=364)
        expected = pd.date_range(first.tz_localize(TIMEZONES[zone]), (end+pd.Timedelta(days=1)).tz_localize(TIMEZONES[zone]),
                                 freq="h", inclusive="left").tz_convert("UTC")
        windows[zone] = {"requested_start_day": str(first.date()), "end_day": str(end.date()),
                         "represented_start_day": str(group.local_day.min()), "represented_days": int(group.local_day.nunique()),
                         "requested_days": 365, "expected_hours": len(expected), "represented_hours": len(group),
                         "common_paired_hours": int(common[group.index].sum()),
                         "complete_365_physical_support": len(expected.difference(pd.DatetimeIndex(group.timestamp_utc))) == 0,
                         "complete_365_common_support": len(expected) == int(common[group.index].sum())}
    return {"schema_version": 1, "diagnostic_only": True, "production_activation": False,
            "variants": list(frames), "model_labels": {key: LABELS.get(key, key) for key in ["nuclear_kalman", "storm", *frames]},
            "pairing_policy": "same finite actual, nuclear_kalman, Storm and every variant candidate on identical physical evaluation hours",
            "frozen_source_values_verified": True, "baseline_fallback_rows_included": True,
            "live_rows_excluded": int(base["sample"].eq("live").sum()), "windows": windows,
            "classification_note": "Native targets/thresholds differ: never rank native precision, recall or Brier across unlike events. Common-target ROC/AP are diagnostic ranking only.",
            "dynamic_threshold_note": "Causal DWT targets extreme NYX underprediction (residual), not an absolute electricity-price spike threshold.",
            "overall": _zone_summary(base, frames, common, np.ones(len(base), dtype=bool)),
            "by_zone": {zone: _zone_summary(base, frames, common, base.zone.eq(zone).to_numpy()) for zone in sorted(windows)},
            "daily": _aggregates(base, frames, common, "local_day"), "hourly": _aggregates(base, frames, common, "local_hour")}


def _local_waterfall(case: Mapping[str, Any]) -> dict[str, Any] | None:
    """Retain the exact additive identity when grouping small local effects."""
    required = ("base_value_calibrated", "calibrated_margin", "prob_calibrated", "timestamp_utc", "zone", "contributions")
    if any(key not in case for key in required):
        return None
    contributions = case["contributions"]
    if not isinstance(contributions, list) or any(not isinstance(row, Mapping) or
            "shap_value_calibrated" not in row or "feature" not in row for row in contributions):
        return None
    base, margin, probability = (float(case[key]) for key in required[:3])
    values = np.asarray([float(row["shap_value_calibrated"]) for row in contributions])
    if not np.isfinite([base, margin, probability, *values]).all() or not 0 <= probability <= 1:
        raise ValueError("Local calibrated SHAP values/probability must be finite and valid.")
    reconstructed = base + float(values.sum())
    sigmoid = 1/(1+np.exp(-margin)) if margin >= 0 else np.exp(margin)/(1+np.exp(margin))
    if not np.isclose(reconstructed, margin, rtol=2e-5, atol=2e-5) or not np.isclose(sigmoid, probability, rtol=2e-5, atol=2e-5):
        raise ValueError("Local SHAP contributions do not reconstruct calibrated log-odds/probability.")
    ordered = sorted(contributions, key=lambda row: abs(float(row["shap_value_calibrated"])), reverse=True)
    displayed = [{"feature": str(row["feature"]), "contribution": float(row["shap_value_calibrated"]),
                  "feature_value": row.get("feature_value")} for row in ordered[:10]]
    if len(ordered) > 10:
        displayed.append({"feature": f"Autres ({len(ordered)-10} variables)",
                          "contribution": float(sum(float(row["shap_value_calibrated"]) for row in ordered[10:])),
                          "feature_value": None})
    timestamp = pd.Timestamp(case["timestamp_utc"])
    zone = str(case["zone"]).upper()
    if timestamp.tzinfo is None or zone not in TIMEZONES:
        raise ValueError("Local SHAP cases require an aware timestamp and supported country.")
    local = timestamp.tz_convert(TIMEZONES[zone])
    return {"zone": zone, "timestamp_utc": timestamp.isoformat(), "local_day": str(local.date()),
            "local_label": local.strftime("%Y-%m-%d %H:%M %z"), "base_log_odds": base,
            "calibrated_log_odds": margin, "probability_calibrated": probability,
            "displayed_contributions": displayed, "source_feature_count": len(contributions),
            "reconstructed_log_odds": base + sum(row["contribution"] for row in displayed),
            "all_contributions_retained_in_sum": True}


def _compact_explanations(explanations: Mapping[str, Any] | None) -> dict[str, Any]:
    if explanations is None:
        return {}
    if not isinstance(explanations, Mapping):
        raise ValueError("Explanations must be per-variant summaries, not a raw feature/SHAP table.")
    allowed = ("status", "method", "explained_output", "sampling", "global_importance", "local_cases", "paths", "audit")
    output = {}
    for key, value in explanations.items():
        if not isinstance(value, Mapping):
            continue
        # The runner also retains full SHAP/observation DataFrames for its
        # exports. Only the bounded human-readable summary belongs in HTML.
        if isinstance(value.get("summary"), Mapping):
            value = value["summary"]
        verified = value.get("status") in {"complete", "partial"} and isinstance(value.get("audit"), Mapping) and value["audit"].get("reconstruction_verified") is True
        output[str(key)] = {name: value[name] for name in allowed if name in value
                            and (verified or name not in {"global_importance", "local_cases"})}
        output[str(key)]["numeric_display_verified"] = verified
        if verified:
            output[str(key)]["local_waterfalls"] = [result for case in value.get("local_cases", [])
                                                     if isinstance(case, Mapping) and (result := _local_waterfall(case)) is not None]
    return output


_SCRIPT = r"""
const P=JSON.parse(document.getElementById('variant-data').textContent),$=id=>document.getElementById(id),finite=x=>typeof x==='number'&&Number.isFinite(x);
P.base.forEach((r,i)=>r.i=i);const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),fmt=(x,n=3)=>finite(x)?x.toLocaleString('fr-FR',{minimumFractionDigits:n,maximumFractionDigits:n}):'—';
const zone=()=>$('country').value,variant=()=>$('variant').value,label=k=>P.summary.model_labels[k]||k,group=()=>P.summary.by_zone[zone()],dark=()=>document.documentElement.dataset.theme==='dark';
const palette=()=>({actual:dark()?'#f6f4ee':'#23354c',nuclear_kalman:'#789bec',storm:dark()?'#ffca89':'#bb7214',hgb_v1:dark()?'#c8a2ed':'#8952b8',candidate:dark()?'#75dfb0':'#13774f'});
function plot(id,traces,title,y){const c=getComputedStyle(document.documentElement),horizontal=id.startsWith('shap');Plotly.react(id,traces,{title:{text:title,font:{size:15}},height:id==='shap'?680:id==='shapLocal'?560:undefined,paper_bgcolor:c.getPropertyValue('--panel').trim(),plot_bgcolor:c.getPropertyValue('--panel').trim(),font:{color:c.getPropertyValue('--ink').trim()},margin:{l:60,r:18,t:55,b:75},xaxis:{title:horizontal?y:undefined,automargin:true,gridcolor:c.getPropertyValue('--line').trim()},yaxis:{title:horizontal?undefined:y,automargin:true,gridcolor:c.getPropertyValue('--line').trim()},legend:{orientation:'h',y:-.25},hovermode:'closest'},{responsive:true,displaylogo:false});}
function table(headers,rows){return '<div class="table-wrap"><table><thead><tr>'+headers.map(v=>'<th>'+esc(v)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function scores(s){return table(['Modèle','Heures','Prix prévu moyen','MAE','RMSE','Biais','MAE des moyennes journalières','Gain MAE vs HGB'],Object.entries(s).map(([key,v])=>[esc(label(key)),fmt(v.hours,0),...['mean_forecast_eur_mwh','mae_eur_mwh','rmse_eur_mwh','bias_eur_mwh','daily_mean_mae_eur_mwh','gain_vs_hgb_mae_eur_mwh'].map(k=>fmt(v[k]))]));}
function dates(){const rows=P.base.filter(r=>r.zone===zone()),days=[...new Set(rows.map(r=>r.local_day))].sort(),known=rows.filter(r=>finite(r.actual)).map(r=>r.local_day).sort();$('day').innerHTML=days.map(d=>'<option>'+esc(d)+'</option>').join('');$('day').value=known.at(-1)||days.at(-1)||'';const months=[...new Set(P.summary.daily.filter(r=>r.zone===zone()).map(r=>r.local_day.slice(0,7)))].sort();$('month').innerHTML=months.map(m=>'<option>'+m+'</option>').join('');$('month').value=months.at(-1)||'';}
function overview(){const g=group(),w=P.summary.windows[zone()];if(!g)return;$('support').textContent=`${w.requested_start_day} → ${w.end_day} : ${w.represented_days}/365 jours représentés, ${g.paired_hours} heures communes, ${g.unpaired_window_hours} heures exclues. ${w.complete_365_common_support?'Année complète appariée.':'Support annuel incomplet : ne pas présenter le score comme une année complète.'}`;$('annual').innerHTML=scores(g.annual);$('observed').textContent='Prix moyen observé sur les mêmes heures : '+fmt(g.annual.nuclear_kalman.mean_observed_eur_mwh)+' EUR/MWh.';
$('tails').innerHTML=['top_1_percent','top_5_percent'].map(k=>'<h3>'+esc(k.replaceAll('_',' '))+' — seuil ex post '+fmt(g.tails[k].threshold_eur_mwh)+' EUR/MWh</h3>'+scores(g.tails[k].scores)).join('');const v=variant(),a=g.interventions[v],t=g.trained_coverage[v];$('interventions').innerHTML=table(['Heures entraînées communes','Première date prête','Retours à NYX inclus','Corrections actives','Interventions hors événement ≥50','Heures avec erreur aggravée'],[[fmt(t.ready_hours_on_common_point_support,0),esc(t.first_ready_day||'—'),fmt(t.baseline_fallback_hours_in_common_support,0),fmt(a.active_hours,0),fmt(a.non_event_interventions_common_error50,0),fmt(a.worsened_absolute_error_hours,0)]]);$('active').innerHTML=scores(a.all_models_on_this_variant_active_subset);}
function classifications(){const g=group(),r=g.classification_common_ranking;$('ranking').innerHTML=table(['Variante','Heures prêtes communes','Événements ≥50','Prévalence','Average precision / PR-AUC','ROC-AUC','Cible native =50 partout'],Object.entries(r.variants).map(([k,v])=>[esc(label(k)),fmt(v.hours,0),fmt(v.positives,0),fmt(v.prevalence),fmt(v.average_precision),fmt(v.roc_auc),v.native_target_matches_common_on_all_rows?'Oui':'Non / indisponible']));plot('commonPR',Object.entries(r.variants).map(([k,v])=>({x:v.pr_curve.recall,y:v.pr_curve.precision,name:label(k),mode:'lines'})),'Événement commun : erreur NYX ≥50 — classement diagnostique','Précision');
const n=g.classification_native[variant()];$('native').innerHTML=table(['Heures','Seuil natif min / médian / max','Événements','Brier natif','Précision','Rappel','FPR','AP native','ROC native'],[[fmt(n.hours,0),[n.threshold_min,n.threshold_median,n.threshold_max].map(v=>fmt(v,1)).join(' / '),fmt(n.positives,0),fmt(n.brier),fmt(n.precision),fmt(n.recall),fmt(n.false_positive_rate),fmt(n.average_precision),fmt(n.roc_auc)]]);plot('reliability',[{x:n.reliability_bins.map(v=>v.mean_probability),y:n.reliability_bins.map(v=>v.event_frequency),text:n.reliability_bins.map(v=>v.hours+' heures'),name:'Cible native — '+label(variant()),mode:'lines+markers'},{x:[0,1],y:[0,1],name:'Calibration parfaite',mode:'lines',line:{dash:'dot',color:palette().actual}}],'Calibration sur la seule cible native sélectionnée','Fréquence observée');}
function dailyChart(){const rows=P.base.filter(r=>r.zone===zone()&&r.local_day===$('day').value),v=P.variant_values[variant()],h=P.variant_values.hgb_v1,c=palette(),x=rows.map(r=>r.local_label),traces=[{x,y:rows.map(r=>v.candidate_q90[r.i]),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'},{x,y:rows.map(r=>v.candidate_q10[r.i]),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(50,165,120,.15)',name:'P10–P90 variante',hoverinfo:'skip'}];for(const [key,field] of [['actual','actual'],['nuclear_kalman','forecast'],['storm','benchmark_forecast']])traces.push({x,y:rows.map(r=>r[field]),name:key==='actual'?'Observé':label(key),mode:'lines+markers',line:{color:c[key]},connectgaps:false});if(variant()!=='hgb_v1')traces.push({x,y:rows.map(r=>h.candidate_forecast[r.i]),name:label('hgb_v1'),mode:'lines',line:{color:c.hgb_v1,dash:'dot'},connectgaps:false});traces.push({x,y:rows.map(r=>v.candidate_forecast[r.i]),name:label(variant()),mode:'lines+markers',line:{color:c.candidate,width:3},connectgaps:false});plot('dailyChart',traces,'Prix — '+zone()+' '+$('day').value,'EUR/MWh');$('dayNote').textContent=rows.some(r=>r.sample==='live')?'Livraison live : affichage uniquement, hors métriques historiques.':'Prévisions figées ; pas de remplissage des observations ou de Storm manquants.';const means=P.summary.daily.filter(r=>r.zone===zone()&&r.local_day===$('day').value);$('dailyMeans').innerHTML=table(['Modèle','Heures communes','Prix moyen observé','Prix moyen prévu','MAE horaire'],means.map(r=>[esc(label(r.model)),fmt(r.paired_hours,0),fmt(r.mean_observed_eur_mwh),fmt(r.mean_forecast_eur_mwh),fmt(r.mae_eur_mwh)]));}
function hourly(){const all=P.summary.hourly.filter(r=>r.zone===zone()),models=['nuclear_kalman','storm','hgb_v1',...(variant()==='hgb_v1'?[]:[variant()])];for(const [id,metric,title] of [['hourlyMae','mae_eur_mwh','MAE horaire'],['hourlyMean','mean_forecast_eur_mwh','Prix moyen par heure locale']]){const traces=models.map(k=>{const r=all.filter(x=>x.model===k);return {x:r.map(x=>x.local_hour),y:r.map(x=>x[metric]),name:label(k),mode:'lines+markers',connectgaps:false}});if(id==='hourlyMean'){const r=all.filter(x=>x.model==='nuclear_kalman');traces.push({x:r.map(x=>x.local_hour),y:r.map(x=>x.mean_observed_eur_mwh),name:'Observé',mode:'lines+markers',line:{color:palette().actual},connectgaps:false});}plot(id,traces,title+' — support commun à toutes les variantes','EUR/MWh');}}
function calendar(){const month=$('month').value;if(!month)return;const rows=P.summary.daily.filter(r=>r.zone===zone()&&r.model===variant()&&r.local_day.startsWith(month)),map=Object.fromEntries(rows.map(r=>[r.local_day,r])),scale=Math.max(1,...rows.filter(r=>finite(r.gain_vs_hgb_mae_eur_mwh)).map(r=>Math.abs(r.gain_vs_hgb_mae_eur_mwh))),first=new Date(month+'-01T12:00:00Z'),offset=(first.getUTCDay()+6)%7,last=new Date(Date.UTC(first.getUTCFullYear(),first.getUTCMonth()+1,0)).getUTCDate();let h=['Lun','Mar','Mer','Jeu','Ven','Sam','Dim'].map(v=>'<b>'+v+'</b>').join('')+'<span></span>'.repeat(offset);for(let n=1;n<=last;n++){const date=month+'-'+String(n).padStart(2,'0'),r=map[date],gain=r?.gain_vs_hgb_mae_eur_mwh,a=finite(gain)?(.1+.6*Math.abs(gain)/scale):0,color=gain>=0?'40,150,100':'200,60,80';h+='<button class="day" data-date="'+date+'" style="background:rgba('+color+','+a+')" title="'+esc(date+' · '+(r?.paired_hours??0)+' heures communes')+'">'+n+'<small>'+fmt(gain,2)+'</small></button>';}$('calendar').innerHTML=h;$('calendar').querySelectorAll('[data-date]').forEach(b=>b.onclick=()=>{if([...$('day').options].some(o=>o.value===b.dataset.date)){$('day').value=b.dataset.date;dailyChart();}});}
function shapLocal(){const e=P.explanations[variant()],cases=e?.local_waterfalls||[],item=cases[Number($('localCase').value)];if(!item||item.zone!==zone()){$('localNote').textContent='Aucun cas local vérifié pour le pays sélectionné.';plot('shapLocal',[],'Explication locale indisponible','Log-odds calibrés');return;}const rows=item.displayed_contributions,c=palette();$('localNote').textContent=`${item.local_label} · Base = ${fmt(item.base_log_odds,6)} ; total = ${fmt(item.calibrated_log_odds,6)} log-odds ; probabilité calibrée = ${fmt(item.probability_calibrated*100,3)} %. Base + toutes les contributions = ${fmt(item.reconstructed_log_odds,6)}. Les effets non affichés individuellement sont conservés dans « Autres ».`;plot('shapLocal',[{type:'waterfall',orientation:'h',measure:['absolute',...rows.map(()=> 'relative'),'total'],y:['Base',...rows.map(r=>r.feature),'Total calibré'],x:[item.base_log_odds,...rows.map(r=>r.contribution),0],text:[fmt(item.base_log_odds,5),...rows.map(r=>fmt(r.contribution,5)),fmt(item.calibrated_log_odds,5)],textposition:'auto',increasing:{marker:{color:c.candidate}},decreasing:{marker:{color:dark()?'#ff9aaa':'#bf3b58'}},totals:{marker:{color:c.hgb_v1}},name:'Contributions signées'}],'TreeSHAP local — '+item.local_label,'Log-odds calibrés, pas EUR/MWh');}
function shap(){const e=P.explanations[variant()];if(!e||!e.numeric_display_verified){$('shapText').textContent='Attribution indisponible ou reconstruction non vérifiée : aucune valeur n’est inventée.';plot('shap',[],'Attribution indisponible','');$('shapAudit').textContent=e?JSON.stringify(e,null,2):'';$('localCase').innerHTML='';shapLocal();return;}$('shapText').textContent=`${e.method||'Méthode non déclarée'} — ${e.explained_output||'sortie non déclarée'}. Contributions du classificateur, pas du prix final ni de la correction en EUR/MWh.`;const source=Array.isArray(e.global_importance)?e.global_importance:[],valid=source.filter(r=>finite(r.mean_abs_shap_calibrated??r.mean_abs_shap_raw??r.mean_abs_shap));const top=valid.slice().sort((a,b)=>(b.mean_abs_shap_calibrated??b.mean_abs_shap_raw??b.mean_abs_shap)-(a.mean_abs_shap_calibrated??a.mean_abs_shap_raw??a.mean_abs_shap)).slice(0,20).reverse();plot('shap',[{x:top.map(r=>r.mean_abs_shap_calibrated??r.mean_abs_shap_raw??r.mean_abs_shap),y:top.map(r=>r.feature),type:'bar',orientation:'h',marker:{color:palette().candidate},name:'Moyenne |TreeSHAP|'}],'Importance globale — échantillon audité','Log-odds, pas EUR/MWh');$('shapAudit').textContent=JSON.stringify(e,null,2);const cases=(e.local_waterfalls||[]).map((r,i)=>({...r,i})).filter(r=>r.zone===zone());$('localCase').innerHTML=cases.map(r=>'<option value="'+r.i+'">'+esc(r.local_label)+'</option>').join('');const selected=cases.find(r=>r.local_day===$('day').value)||cases[0];if(selected)$('localCase').value=String(selected.i);shapLocal();}
function render(){overview();classifications();dailyChart();hourly();calendar();shap();}
$('country').innerHTML=Object.keys(P.summary.by_zone).map(z=>'<option>'+esc(z)+'</option>').join('');if(P.summary.by_zone.FR)$('country').value='FR';$('variant').innerHTML=P.summary.variants.map(v=>'<option value="'+esc(v)+'">'+esc(label(v))+'</option>').join('');$('variant').value=P.summary.variants.includes('xgb_unweighted_fixed')?'xgb_unweighted_fixed':'hgb_v1';try{document.documentElement.dataset.theme=localStorage.getItem('nyx-variants-theme')||'light';}catch(_){}$('theme').onclick=()=>{document.documentElement.dataset.theme=dark()?'light':'dark';try{localStorage.setItem('nyx-variants-theme',document.documentElement.dataset.theme);}catch(_){}render();};$('country').onchange=()=>{dates();render();};$('variant').onchange=render;$('day').onchange=()=>{dailyChart();shap();};$('localCase').onchange=shapLocal;$('month').onchange=calendar;dates();if(Object.keys(P.summary.by_zone).length)render();
"""


def render_comparison(predictions: Mapping[str, pd.DataFrame], summary: Mapping[str, Any], audit: Mapping[str, Any],
                      path: str | Path, explanations: Mapping[str, Any] | None = None) -> Path:
    from plotly.offline import get_plotlyjs

    base, frames, _ = _align(predictions)
    if list(frames) != list(summary.get("variants", [])):
        raise ValueError("Comparison summary does not identify the supplied variants in the same order.")
    display = base.in_window | base["sample"].eq("live")
    payload = {"base": base.loc[display, _BASE_PAYLOAD].to_dict("records"),
               "variant_values": {key: {column: frame.loc[display, column].tolist() for column in _VARIANT_PAYLOAD}
                                  for key, frame in frames.items()},
               "summary": summary, "explanations": _compact_explanations(explanations)}
    text_audit = html.escape(json.dumps(_clean(audit), ensure_ascii=False, indent=2, default=str, allow_nan=False))
    document = '''<!doctype html><html lang="fr" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NYX — comparaison contrôlée des experts XGB</title><style>''' + _STYLE + '''</style></head><body>
<header><div><small>EXPÉRIMENTATION ISOLÉE · CONTRÔLE HGB FIGÉ</small><h1>NYX — variantes XGB et seuils causaux</h1></div><button id="theme">Mode nuit / jour</button></header><main>
<section><div class="controls"><label>Pays <select id="country"></select></label><label>Variante <select id="variant"></select></label></div>
<p class="warning">Année historique déjà examinée ; diagnostic, pas validation prospective ni preuve PIT indépendante. Entraînement progressif 90–365 jours et retours à NYX conservés : pas 365 jours d’entraînement avant chaque prévision. Aucune activation ni garantie de non-régression. Les données réseau à 08 h restent soumises aux audits.</p>
<p>Les variantes à seuil dynamique causal (DWT) ciblent une forte erreur positive de NYX, pas un seuil absolu de prix de l’électricité. Aucun seuil opératoire n’est sélectionné après examen des résultats. L’interface n’active pas automatiquement la variante historiquement gagnante.</p><p id="support"></p></section>
<section data-report-section="variant-statistics"><h2>Statistics — année commune et prix moyens</h2><p id="observed"></p><div id="annual"></div><p>EUR/MWh. Observé, NYX, Storm et toutes les variantes utilisent exactement les mêmes heures finies. Les moyennes journalières portent sur ces mêmes heures ; une journée partielle n’est pas un prix moyen journalier complet. Les trous ne sont jamais remplis par zéro.</p></section>
<section data-report-section="variant-daily"><h2>Prévisions journalières figées</h2><label>Livraison <select id="day"></select></label><p id="dayNote"></p><div id="dailyChart" class="chart"></div><h3>Statistics — prix moyens de la journée choisie</h3><div id="dailyMeans"></div></section>
<section data-report-section="variant-hourly"><h2>Profil horaire sur le même support</h2><div id="hourlyMean" class="chart"></div><div id="hourlyMae" class="chart"></div></section>
<section data-report-section="variant-calendar"><h2>Calendrier — gain de MAE contre le HGB original</h2><select id="month"></select><p class="legend">Vert : amélioration ; rouge : dégradation ; — : score absent. Cliquer une date pour les prévisions. Le choix du modèle reste manuel.</p><div id="calendar" class="calendar"></div></section>
<section data-report-section="variant-tails"><h2>Queues de prix — analyse ex post</h2><p>Top 1 % / 5 % des prix observés sur le support commun ; ces cohortes ne sont ni des entrées ni des règles de gouvernance. Les égalités au seuil peuvent accroître la fraction sélectionnée.</p><div id="tails"></div></section>
<section data-report-section="variant-interventions"><h2>Couverture entraînée et interventions de la variante sélectionnée</h2><div id="interventions"></div><p>Une intervention hors événement résiduel ≥50 n’implique pas à elle seule une mauvaise décision. L’aggravation de l’erreur absolue est comptée séparément. Le tableau suivant utilise uniquement les heures d’intervention de la variante choisie : il ne remplace pas la comparaison annuelle et ce sous-ensemble change selon la variante.</p><div id="active"></div></section>
<section data-report-section="variant-common-classification"><h2>Classement sur un événement commun : erreur NYX ≥50 EUR/MWh</h2><p>Intersection des heures où tous les experts sont prêts avec une probabilité finie. Average precision (AP, résumé de la courbe précision-rappel) et ROC-AUC évaluent ici le classement diagnostique. Les probabilités entraînées sur des cibles différentes ne sont pas supposées calibrées pour cet événement commun ; aucun Brier commun n’est présenté.</p><div id="ranking"></div><div id="commonPR" class="chart"></div></section>
<section data-report-section="variant-native-classification"><h2>Calibration native — variante sélectionnée seulement</h2><p>Cible propre : observé − NYX ≥ seuil causal de la variante. Les seuils et les populations entraînées peuvent différer ; ne pas classer les modèles à partir de leurs Brier, précision ou rappel natifs. Les décisions utilisent le gate fourni, strictement dépassé, jamais un seuil optimisé dans ce rapport.</p><div id="native"></div><div id="reliability" class="chart"></div></section>
<section data-report-section="variant-shap"><h2>Explication du classificateur — TreeSHAP</h2><p id="shapText"></p><div id="shap" class="chart" style="height:680px"></div><h3>Explication locale signée — fin d’évaluation / livraison à 19 h</h3><label>Cas du pays sélectionné <select id="localCase"></select></label><p id="localNote"></p><div id="shapLocal" class="chart" style="height:560px"></div><details><summary>Échantillonnage, cas locaux et audit de reconstruction</summary><pre id="shapAudit"></pre></details></section>
<section><h2>Traçabilité et limites</h2><p>Pas d’EVA, de P&amp;L ou de poids causaux inventés. Les observations et Storm ne sont utilisés ici que pour l’évaluation. Les annotations de classement et de queues ne modifient jamais les prévisions.</p><details><summary>Audit complet : sources, modèles et configuration</summary><pre>''' + text_audit + '''</pre></details></section></main>
<script id="variant-data" type="application/json">''' + _script_json(payload) + '''</script><script>''' + get_plotlyjs().replace('</script', '<\\/script') + '''</script><script>''' + _SCRIPT + '''</script></body></html>'''
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return target


__all__ = ["build_comparison", "render_comparison"]
