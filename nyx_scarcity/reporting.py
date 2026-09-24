"""Offline paired diagnostics for an isolated NYX scarcity challenger.

No model fit, external request, feature construction or production mutation is
performed here.  Scores always retain the governed baseline-fallback rows.
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


TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin", "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
POINTS = {"baseline": "forecast", "candidate": "candidate_forecast", "storm": "benchmark_forecast"}
NUMERIC = ("actual", "forecast", "q10", "q90", "benchmark_forecast", "candidate_forecast",
           "candidate_q10", "candidate_q90", "spike_probability", "raw_correction", "applied_correction",
           "selected_weight", "threshold_eur_mwh", "probability_gate", "bounded_correction")
# Only chart/table inputs belong in the HTML. Training frames can contain
# hundreds of fundamental features; embedding them duplicates large datasets
# without serving any report interaction. UTC timestamps retain provenance.
_REPORT_PREDICTION_COLUMNS = (
    "zone", "timestamp_utc", "forecast_origin_utc", "sample", "local_day", "local_label",
    "actual", "forecast", "benchmark_forecast", "candidate_forecast", "q10", "q90",
    "candidate_q10", "candidate_q90", "spike_probability", "raw_candidate_forecast",
    "applied_correction", "selected_weight", "gate_reason",
)


def _prepare(predictions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame) or predictions.columns.has_duplicates:
        raise ValueError("Predictions must be a DataFrame with unique columns.")
    required = {"zone", "timestamp_utc", "sample"}
    if not required.issubset(predictions):
        raise ValueError(f"Missing identity columns: {sorted(required.difference(predictions.columns))}.")
    frame = predictions.copy(deep=True).reset_index(drop=True)
    frame["zone"] = frame.zone.astype(str).str.upper().str.strip()
    if not set(frame.zone).issubset(TIMEZONES):
        raise ValueError("Only FR, DE, BE and NL are supported.")
    if not set(frame["sample"]).issubset({"evaluation", "live"}):
        raise ValueError("sample must be evaluation or live.")
    for column in ("timestamp_utc", "forecast_origin_utc"):
        if column not in frame:
            frame[column] = pd.NaT
            continue
        values = frame[column]
        if column == "timestamp_utc" and values.isna().any():
            raise ValueError("Delivery timestamps cannot be missing.")
        if any(pd.Timestamp(value).tzinfo is None for value in values[values.notna()]):
            raise ValueError(f"{column} requires explicit timezone-aware timestamps.")
        frame[column] = pd.to_datetime(values, utc=True, errors="raise")
    if frame.duplicated(["zone", "timestamp_utc"]).any():
        raise ValueError("Each country/physical delivery hour must appear exactly once.")
    if not frame.timestamp_utc.equals(frame.timestamp_utc.dt.floor("h")):
        raise ValueError("Hourly physical timestamps are required.")
    for column in NUMERIC:
        if column not in frame:
            frame[column] = .6 if column == "probability_gate" else np.nan
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if np.isinf(frame[column]).any():
            raise ValueError(f"{column} contains infinity.")
    for column in ("spike_probability", "selected_weight", "probability_gate"):
        finite = frame[column].dropna()
        if ((finite < 0) | (finite > 1)).any():
            raise ValueError(f"{column} must be between zero and one.")
    for lower, upper in (("q10", "q90"), ("candidate_q10", "candidate_q90")):
        if (frame[lower].notna() & frame[upper].notna() & frame[lower].gt(frame[upper])).any():
            raise ValueError(f"Crossing quantiles: {lower} > {upper}.")
    if "expert_ready" not in frame:
        frame["expert_ready"] = pd.NA
    if any(not isinstance(value, (bool, np.bool_)) for value in frame.expert_ready.dropna()):
        raise ValueError("expert_ready must contain booleans, not truthy strings.")
    frame["expert_ready_available"] = frame.expert_ready.notna()
    frame["expert_ready"] = frame.expert_ready.astype("boolean").fillna(False).astype(bool)
    for column in ("gate_reason", "interval_status"):
        if column not in frame:
            frame[column] = "unavailable"
        frame[column] = frame[column].fillna("unavailable").astype(str)
    frame["local_day"] = ""
    frame["local_hour"] = 0
    frame["local_label"] = ""
    for zone, indices in frame.groupby("zone").groups.items():
        local = frame.loc[indices, "timestamp_utc"].dt.tz_convert(TIMEZONES[zone])
        frame.loc[indices, "local_day"] = local.dt.strftime("%Y-%m-%d")
        frame.loc[indices, "local_hour"] = local.dt.hour.to_numpy()
        frame.loc[indices, "local_label"] = local.dt.strftime("%Y-%m-%d %H:%M %z")
    frame["paired"] = np.isfinite(frame[["actual", *POINTS.values()]].to_numpy(float)).all(axis=1)
    frame["active"] = frame.applied_correction.notna() & frame.applied_correction.ne(0)
    frame["raw_candidate_forecast"] = frame.forecast + frame.raw_correction
    frame["bounded_candidate_forecast"] = frame.forecast + frame.bounded_correction
    if np.isinf(frame[["raw_candidate_forecast", "bounded_candidate_forecast"]].to_numpy(float)).any():
        raise ValueError("Diagnostic price proposals overflow finite numeric range.")
    frame["in_evaluation_window"] = False
    for _, part in frame.loc[frame["sample"].eq("evaluation")].groupby("zone"):
        end = pd.Timestamp(part.local_day.max()).date()
        start = end - timedelta(days=364)
        frame.loc[part.index, "in_evaluation_window"] = part.local_day.ge(str(start))
    return frame.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)


def _finite_mean(values: pd.Series) -> float | None:
    return float(values.mean()) if values.notna().any() else None


def _score(frame: pd.DataFrame, point: str) -> dict[str, Any]:
    if frame.empty:
        return {"hours": 0, "country_days": 0, "mean_forecast_eur_mwh": None, "mean_observed_eur_mwh": None,
                "mae_eur_mwh": None, "rmse_eur_mwh": None, "bias_eur_mwh": None,
                "daily_mean_mae_eur_mwh": None}
    error = frame[point].to_numpy(float) - frame.actual.to_numpy(float)
    daily = frame.groupby(["zone", "local_day"])[[point, "actual"]].mean()
    return {"hours": len(frame), "country_days": len(daily), "mean_forecast_eur_mwh": float(frame[point].mean()),
            "mean_observed_eur_mwh": float(frame.actual.mean()), "mae_eur_mwh": float(np.abs(error).mean()),
            "rmse_eur_mwh": float(np.sqrt(np.mean(error ** 2))), "bias_eur_mwh": float(error.mean()),
            "daily_mean_mae_eur_mwh": float((daily[point] - daily.actual).abs().mean())}


def _cohort(frame: pd.DataFrame) -> dict[str, Any]:
    scores = {name: _score(frame, column) for name, column in POINTS.items()}
    for baseline in ("baseline", "storm"):
        scores[f"gain_{baseline}_minus_candidate"] = {
            key: scores[baseline][key] - scores["candidate"][key] if len(frame) else None
            for key in ("mae_eur_mwh", "rmse_eur_mwh", "daily_mean_mae_eur_mwh")}
    raw = frame.loc[frame.raw_candidate_forecast.notna()]
    scores["raw_ungoverned_diagnostic"] = {"not_a_deployed_forecast": True,
        "raw_candidate": _score(raw, "raw_candidate_forecast"), "baseline_same_hours": _score(raw, "forecast"),
        "candidate_same_hours": _score(raw, "candidate_forecast"), "storm_same_hours": _score(raw, "benchmark_forecast")}
    bounded = frame.loc[frame.bounded_candidate_forecast.notna()]
    scores["bounded_weight_one_diagnostic"] = {"not_a_deployed_forecast": True,
        "bounded_candidate": _score(bounded, "bounded_candidate_forecast"),
        "baseline_same_hours": _score(bounded, "forecast"), "candidate_same_hours": _score(bounded, "candidate_forecast"),
        "storm_same_hours": _score(bounded, "benchmark_forecast")}
    return scores


def _classifier(frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame.loc[frame.spike_probability.notna() & frame.threshold_eur_mwh.notna() & frame.probability_gate.notna()]
    actual = (selected.actual - selected.forecast).ge(selected.threshold_eur_mwh).to_numpy()
    probability = selected.spike_probability.to_numpy(float)
    predicted = probability > selected.probability_gate.to_numpy(float)
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    bins = []
    for lower in np.arange(0., 1., .1):
        mask = (probability >= lower) & ((probability < lower + .1) if lower < .89 else (probability <= 1))
        if mask.any():
            bins.append({"lower": float(lower), "hours": int(mask.sum()),
                         "mean_probability": float(probability[mask].mean()), "event_frequency": float(actual[mask].mean())})
    gates = sorted(float(value) for value in selected.probability_gate.unique())
    return {"hours": len(selected), "event_definition": "actual - baseline forecast >= causal residual threshold_eur_mwh",
            "probability_gates": gates, "probability_rule": "strictly greater than each row's supplied probability_gate; default 0.6 only if column absent",
            "brier": float(np.mean((probability - actual) ** 2)) if len(selected) else None,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "true_negative": int((~predicted & ~actual).sum()), "observed_events": int(actual.sum()), "bins": bins}


def _quantiles(frame: pd.DataFrame) -> dict[str, Any]:
    columns = ["q10", "q90", "candidate_q10", "candidate_q90"]
    common = frame.loc[np.isfinite(frame[columns].to_numpy(float)).all(axis=1)]
    scores = {}
    for name, lower, upper in (("baseline", "q10", "q90"), ("candidate", "candidate_q10", "candidate_q90")):
        row = {"hours": len(common), "coverage_p10_p90": None, "mean_interval_width_eur_mwh": None,
               "pinball_p10_eur_mwh": None, "pinball_p90_eur_mwh": None}
        if len(common):
            row["coverage_p10_p90"] = float((common.actual.ge(common[lower]) & common.actual.le(common[upper])).mean())
            row["mean_interval_width_eur_mwh"] = float((common[upper] - common[lower]).mean())
            for q, column in ((.1, lower), (.9, upper)):
                error = common.actual - common[column]
                row[f"pinball_p{int(q*100)}_eur_mwh"] = float(np.maximum(q * error, (q - 1) * error).mean())
        scores[name] = row
    return {"mask": "paired point support plus all four finite baseline/candidate interval endpoints",
            "nominal_coverage": .8, "scores": scores,
            "excluded_interval_hours": len(frame) - len(common),
            "interval_status_counts": {str(k): int(v) for k, v in frame.interval_status.value_counts().items()}}


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    paired = frame.loc[frame.paired]
    tails = {}
    for percent, quantile in ((1, .99), (5, .95)):
        threshold = float(paired.actual.quantile(quantile)) if len(paired) else None
        cohort = paired.loc[paired.actual.ge(threshold)] if threshold is not None else paired
        tails[f"top_{percent}_percent"] = {"selection": "ex_post_observed_price_quantile_on_common_paired_support",
            "used_for_training_or_governance": False, "threshold_eur_mwh": threshold,
            "ties_may_increase_selected_fraction": True, "scores": _cohort(cohort)}
    paired_active = paired.loc[paired.active]
    reasons = {str(k): int(v) for k, v in frame.gate_reason.value_counts().items()}
    return {"rows": len(frame), "paired_hours": len(paired), "unpaired_hours": len(frame) - len(paired),
            "missing_actual_hours": int(frame.actual.isna().sum()),
            "missing_storm_hours": int(frame.benchmark_forecast.isna().sum()),
            "missing_baseline_hours": int(frame.forecast.isna().sum()),
            "missing_candidate_hours": int(frame.candidate_forecast.isna().sum()),
            "ready_hours": int(frame.expert_ready.sum()),
            "not_ready_hours": int((~frame.expert_ready & frame.expert_ready_available).sum()),
            "readiness_unknown_hours": int((~frame.expert_ready_available).sum()),
            "active_hours": int(frame.active.sum()), "inactive_hours": int(frame.applied_correction.eq(0).sum()),
            "intervention_unknown_hours": int(frame.applied_correction.isna().sum()),
            "exact_baseline_fallback_hours": int((frame.candidate_forecast.notna() & frame.forecast.notna()
                                                  & frame.candidate_forecast.eq(frame.forecast)).sum()),
            "gate_reason_counts": reasons, "annual": _cohort(paired),
            "active_only": {"post_governance_subset_not_a_full_year": True, "scores": _cohort(paired_active)},
            "tails": tails, "classifier": _classifier(paired), "quantiles": _quantiles(paired)}


def _aggregate(frame: pd.DataFrame, grouping: str) -> pd.DataFrame:
    rows = []
    for (zone, period), group in frame.groupby(["zone", grouping], sort=True):
        paired = group.loc[group.paired]
        row = {"zone": zone, grouping: period, "rows": len(group), "paired_hours": len(paired),
               "available_actual_hours": int(group.actual.notna().sum()),
               "available_storm_hours": int(group.benchmark_forecast.notna().sum()),
               "active_hours": int(group.active.sum()), "ready_hours": int(group.expert_ready.sum()),
               "actual_mean_eur_mwh": _finite_mean(paired.actual),
               "actual_mean_available_eur_mwh": _finite_mean(group.actual)}
        for name, column in POINTS.items():
            score = _score(paired, column)
            row[f"{name}_mean_eur_mwh"] = score["mean_forecast_eur_mwh"]
            row[f"{name}_mae_eur_mwh"] = score["mae_eur_mwh"]
        row["gain_baseline_minus_candidate_eur_mwh"] = (
            row["baseline_mae_eur_mwh"] - row["candidate_mae_eur_mwh"] if len(paired) else None)
        rows.append(row)
    columns = ["zone", grouping, "rows", "paired_hours", "available_actual_hours", "available_storm_hours",
               "active_hours", "ready_hours", "actual_mean_eur_mwh", "actual_mean_available_eur_mwh",
               *[f"{name}_{metric}_eur_mwh" for name in POINTS for metric in ("mean", "mae")],
               "gain_baseline_minus_candidate_eur_mwh"]
    return pd.DataFrame(rows, columns=columns)


def evaluate_predictions(predictions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return JSON-compatible metrics and paired daily/hourly DataFrames.

    At most the last 365 *civil evaluation dates* per country are scored. Short
    or incomplete support is disclosed, never called a complete annual test.
    Live rows (even observed) are excluded; missing comparisons remain missing.
    """
    frame = _prepare(predictions)
    evaluation = frame.loc[frame.in_evaluation_window]
    windows = {}
    for zone, group in evaluation.groupby("zone"):
        first, last = pd.Timestamp(group.local_day.min()), pd.Timestamp(group.local_day.max())
        expected = pd.date_range((last - pd.Timedelta(days=364)).tz_localize(TIMEZONES[zone]),
                                 (last + pd.Timedelta(days=1)).tz_localize(TIMEZONES[zone]), freq="h", inclusive="left")
        actual_index = pd.DatetimeIndex(group.timestamp_utc)
        windows[zone] = {"start_day": str(first.date()), "end_day": str(last.date()),
                         "requested_start_day": str((last - pd.Timedelta(days=364)).date()),
                         "requested_days": 365, "represented_days": int(group.local_day.nunique()),
                         "represented_hours": len(group), "expected_365_hours": len(expected),
                         "missing_physical_hours": len(expected.tz_convert("UTC").difference(actual_index)),
                         "complete_365_physical_support": len(expected) == len(group) and set(expected.tz_convert("UTC")) == set(actual_index),
                         "complete_365_paired_support": len(expected) == int(group.paired.sum())}
    metrics = {"schema_version": 1, "diagnostic_only": True, "production_activation": False,
               "pairing_policy": "same finite actual, Storm, baseline and governed candidate on every scored hour",
               "baseline_fallback_rows_included": True, "live_rows_excluded": int(frame["sample"].eq("live").sum()),
               "older_evaluation_rows_excluded": int((frame["sample"].eq("evaluation") & ~frame.in_evaluation_window).sum()),
               "daily_mean_scope": "exactly the same paired physical hours; partial days are disclosed, not full-day prices",
               "windows": windows, "overall": _summary(evaluation),
               "by_zone": {zone: _summary(group) for zone, group in evaluation.groupby("zone")}}
    return metrics, _aggregate(evaluation, "local_day"), _aggregate(evaluation, "local_hour")


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _script_json(value: Any) -> str:
    return (json.dumps(_clean(value), ensure_ascii=False, allow_nan=False, default=str)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


_STYLE = """
:root{--bg:#f1f5fa;--panel:#fff;--ink:#182a40;--muted:#526780;--line:#d5dfeb;--accent:#166bb5;--good:#157a55;--bad:#b53550;--hover:#e9f0f7}
html[data-theme=dark]{--bg:#0c1422;--panel:#142137;--ink:#e4edf8;--muted:#a4b7cd;--line:#30445e;--accent:#6bb5ff;--good:#73dfb5;--bad:#ff90a8;--hover:#21334c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px system-ui,sans-serif}header,main{max-width:1450px;margin:auto;padding:22px}
header{display:flex;justify-content:space-between;gap:16px;align-items:center}h1{font-size:27px;margin:4px 0}h2{font-size:20px}h3{font-size:16px}p{line-height:1.55}small,.muted{color:var(--muted)}
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px;margin-bottom:20px}button,select,input{background:var(--panel);color:var(--ink);border:1px solid var(--line);padding:9px;border-radius:6px;font:inherit}
button{cursor:pointer}select{min-width:110px}.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.card{padding:16px;border:1px solid var(--line);border-radius:8px}.card b{display:block;font-size:23px;padding-top:7px}
.chart{height:380px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}tr:hover{background:var(--hover)}th{color:var(--muted)}
.good{color:var(--good)}.bad{color:var(--bad)}pre{white-space:pre-wrap;word-break:break-word;max-height:550px;overflow:auto;font-size:12px}summary{cursor:pointer;font-weight:600}.calendar{display:grid;grid-template-columns:repeat(7,1fr);gap:5px;max-width:830px;margin:auto}.day{min-height:72px;text-align:left;padding:8px;border-radius:6px;border:1px solid var(--line)}.day small{display:block;color:inherit;font-size:11px}.legend{font-size:12px;color:var(--muted);padding:10px}.warning{border-left:4px solid #d79730;padding-left:12px}
@media(max-width:700px){header,main{padding:12px}section{padding:14px}.chart{height:330px}.day{min-height:55px;padding:4px;font-size:12px}.day small{font-size:10px}}
"""


_SCRIPT = r"""
const D=JSON.parse(document.getElementById('nyx-data').textContent), $=id=>document.getElementById(id), finite=x=>typeof x==='number'&&Number.isFinite(x);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,n=3)=>finite(x)?x.toLocaleString('fr-FR',{maximumFractionDigits:n,minimumFractionDigits:n}):'—';
const dark=()=>document.documentElement.dataset.theme==='dark', color=()=>({actual:dark()?'#f2f5fa':'#222f3e',baseline:dark()?'#8eafff':'#355fb2',candidate:dark()?'#70dfb4':'#147858',storm:dark()?'#ffc680':'#b96e15',raw:dark()?'#dc9aee':'#9940a8'});
const zone=()=>$('country').value, byZone=()=>D.metrics.by_zone[zone()], rows=()=>D.predictions.filter(r=>r.zone===zone());
function plot(id,traces,title,ylabel){const c=getComputedStyle(document.documentElement);Plotly.react(id,traces,{title:{text:title,font:{size:15}},paper_bgcolor:c.getPropertyValue('--panel').trim(),plot_bgcolor:c.getPropertyValue('--panel').trim(),font:{color:c.getPropertyValue('--ink').trim()},margin:{l:60,r:20,t:55,b:65},xaxis:{gridcolor:c.getPropertyValue('--line').trim(),automargin:true},yaxis:{title:ylabel,gridcolor:c.getPropertyValue('--line').trim(),zerolinecolor:c.getPropertyValue('--line').trim()},legend:{orientation:'h',y:-.22},hovermode:'x unified'},{responsive:true,displaylogo:false});}
function table(headers,body){return '<div class="table-wrap"><table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+body.map(row=>'<tr>'+row.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function scoreTable(scores){return table(['Modèle','Prix moyen','MAE horaire','RMSE','Biais','MAE moyenne journalière','Heures'],[['baseline','NYX figé'],['candidate','NYX + expert gouverné'],['storm','Storm']].map(([k,n])=>[n,...['mean_forecast_eur_mwh','mae_eur_mwh','rmse_eur_mwh','bias_eur_mwh','daily_mean_mae_eur_mwh'].map(m=>fmt(scores[k][m])),fmt(scores[k].hours,0)]));}
function dates(){const r=rows(),days=[...new Set(r.map(x=>x.local_day))].sort(),observed=r.filter(x=>finite(x.actual)).map(x=>x.local_day).sort();$('day').innerHTML=days.map(d=>'<option>'+esc(d)+'</option>').join('');$('day').value=observed.at(-1)||days.at(-1)||'';const months=[...new Set(D.daily.filter(x=>x.zone===zone()).map(x=>x.local_day.slice(0,7)))].sort();$('month').innerHTML=months.map(m=>'<option>'+esc(m)+'</option>').join('');$('month').value=months.at(-1)||'';}
function annual(){const m=byZone();if(!m){$('stats').textContent='Aucune donnée pour ce pays.';return;}const w=D.metrics.windows[zone()],a=m.annual, gain=a.gain_baseline_minus_candidate.mae_eur_mwh;
$('support').textContent=w?`${w.start_day} → ${w.end_day} · ${w.represented_days}/365 jours représentés · ${m.paired_hours} heures communes · ${m.unpaired_hours} non appariées. ${w.complete_365_paired_support?'Année complète appariée.':'Couverture annuelle incomplète : ne pas présenter ce score comme une année complète.'}`:'Aucune période d’évaluation.';
$('cards').innerHTML=[['MAE du candidat',fmt(a.candidate.mae_eur_mwh)+' €/MWh'],['Gain vs NYX',fmt(gain)+' €/MWh'],['Prix moyen observé commun',fmt(a.candidate.mean_observed_eur_mwh)+' €/MWh'],['Interventions / retour NYX',`${m.active_hours} / ${m.exact_baseline_fallback_hours}`]].map(([t,v])=>'<div class="card">'+esc(t)+'<b>'+esc(v)+'</b></div>').join('');
$('stats').innerHTML=scoreTable(a);$('coverage').textContent=`Manquants : observation ${m.missing_actual_hours}, Storm ${m.missing_storm_hours}, NYX ${m.missing_baseline_hours}, candidat ${m.missing_candidate_hours}. Expert prêt ${m.ready_hours} h ; non prêt ${m.not_ready_hours} h. Les retours à NYX restent inclus dans tous les scores annuels.`;
const cohorts=[['top_1_percent','Top 1 % des prix observés'],['top_5_percent','Top 5 % des prix observés']];$('tails').innerHTML=cohorts.map(([k,label])=>'<h3>'+esc(label)+' — seuil ex post '+fmt(m.tails[k].threshold_eur_mwh)+' €/MWh</h3>'+scoreTable(m.tails[k].scores)).join('');$('active').innerHTML=scoreTable(m.active_only.scores);
const raw=a.raw_ungoverned_diagnostic;$('raw').innerHTML=table(['Diagnostic non gouverné','Heures','MAE'],[['Proposition brute',fmt(raw.raw_candidate.hours,0),fmt(raw.raw_candidate.mae_eur_mwh)],['NYX sur les mêmes heures',fmt(raw.baseline_same_hours.hours,0),fmt(raw.baseline_same_hours.mae_eur_mwh)],['Candidat gouverné sur les mêmes heures',fmt(raw.candidate_same_hours.hours,0),fmt(raw.candidate_same_hours.mae_eur_mwh)]]);
const q=m.quantiles;$('quantiles').innerHTML=table(['Modèle','Heures communes','Couverture P10–P90 (cible 80 %)','Largeur moyenne','Pinball P10','Pinball P90'],[['baseline','NYX'],['candidate','Candidat']].map(([k,label])=>[label,fmt(q.scores[k].hours,0),finite(q.scores[k].coverage_p10_p90)?fmt(100*q.scores[k].coverage_p10_p90,1)+' %':'—',fmt(q.scores[k].mean_interval_width_eur_mwh),fmt(q.scores[k].pinball_p10_eur_mwh),fmt(q.scores[k].pinball_p90_eur_mwh)]));
const c=m.classifier,gate=c.probability_gates.map(v=>fmt(v,2)).join(' / ');$('classifier').innerHTML=table(['Heures','Événements','Brier','Précision : p > '+gate,'Rappel : p > '+gate],[[fmt(c.hours,0),fmt(c.observed_events,0),fmt(c.brier),fmt(c.precision),fmt(c.recall)]]);plot('calibration',[{x:c.bins.map(b=>b.mean_probability),y:c.bins.map(b=>b.event_frequency),text:c.bins.map(b=>b.hours+' heures'),mode:'markers+lines',name:'Fréquence observée',marker:{color:color().candidate,size:9}},{x:[0,1],y:[0,1],mode:'lines',name:'Calibration parfaite',line:{color:color().actual,dash:'dot'}}],'Probabilité de sous-estimation extrême de NYX','Fréquence observée');
$('gateReasons').textContent=JSON.stringify(m.gate_reason_counts,null,2);}
function dayChart(){const r=rows().filter(r=>r.local_day===$('day').value),c=color(),traces=[];const x=r.map(p=>p.local_label);
for(const [key,lo,hi,name,fill] of [['baseline','q10','q90','NYX','rgba(90,130,210,.12)'],['candidate','candidate_q10','candidate_q90','Candidat','rgba(50,170,120,.15)']]){traces.push({x,y:r.map(p=>p[hi]),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip',connectgaps:false});traces.push({x,y:r.map(p=>p[lo]),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:fill,name:'P10–P90 '+name,hoverinfo:'skip',connectgaps:false});}
for(const [field,key,label] of [['actual','actual','Observé'],['forecast','baseline','NYX figé'],['candidate_forecast','candidate','NYX + expert gouverné'],['benchmark_forecast','storm','Storm']])traces.push({x,y:r.map(p=>p[field]),name:label,mode:'lines+markers',line:{color:c[key],width:key==='candidate'?3:2},marker:{size:4},connectgaps:false});
if($('showRaw').checked)traces.push({x,y:r.map(p=>p.raw_candidate_forecast),name:'Proposition brute — non gouvernée',line:{color:c.raw,dash:'dot'},connectgaps:false});plot('dailyChart',traces,'Prévisions et observations — '+zone()+' '+$('day').value,'EUR/MWh');
$('dayNote').textContent=r.some(p=>p.sample==='live')?'Livraison live : affichage uniquement, exclue des métriques de backtest, même si une observation est déjà disponible.':'Rejeu historique : observations et Storm restent vides lorsqu’ils sont absents.';
$('hourRows').innerHTML=table(['Heure locale','Observé','NYX','Candidat','Storm','p(spike)','Correction','Poids','Décision'],r.map(p=>[esc(p.local_label.slice(11)),fmt(p.actual,2),fmt(p.forecast,2),fmt(p.candidate_forecast,2),fmt(p.benchmark_forecast,2),fmt(p.spike_probability),fmt(p.applied_correction,2),fmt(p.selected_weight,2),esc(p.gate_reason)]));}
function hourly(){const r=D.hourly.filter(p=>p.zone===zone()),c=color();plot('hourlyPrice',[['actual','Observé'],['baseline','NYX'],['candidate','Candidat'],['storm','Storm']].map(([k,n])=>({x:r.map(p=>p.local_hour),y:r.map(p=>p[k+'_mean_eur_mwh']),mode:'lines+markers',name:n,line:{color:c[k]},connectgaps:false})),'Prix moyens par heure locale — mêmes heures communes','EUR/MWh');plot('hourlyMae',[['baseline','NYX'],['candidate','Candidat'],['storm','Storm']].map(([k,n])=>({x:r.map(p=>p.local_hour),y:r.map(p=>p[k+'_mae_eur_mwh']),mode:'lines+markers',name:n,line:{color:c[k]},connectgaps:false})),'MAE par heure locale — mêmes heures communes','EUR/MWh');}
function calendar(){const month=$('month').value;if(!month){$('calendar').textContent='Aucune journée évaluée.';return;}const r=D.daily.filter(p=>p.zone===zone()&&p.local_day.startsWith(month)),map=Object.fromEntries(r.map(p=>[p.local_day,p])),metric=$('calendarMetric').value,gain=metric.startsWith('gain'),scale=Math.max(1,...r.filter(p=>finite(p[metric])).map(p=>Math.abs(p[metric]))),first=new Date(month+'-01T12:00:00Z'),offset=(first.getUTCDay()+6)%7,last=new Date(Date.UTC(first.getUTCFullYear(),first.getUTCMonth()+1,0)).getUTCDate();let h=['Lun','Mar','Mer','Jeu','Ven','Sam','Dim'].map(v=>'<b>'+v+'</b>').join('')+'<span></span>'.repeat(offset);
for(let day=1;day<=last;day++){const date=month+'-'+String(day).padStart(2,'0'),p=map[date],v=p?.[metric],alpha=finite(v)?(.10+.65*Math.abs(v)/scale):0,rgb=gain?(v>=0?'30,150,100':'205,65,85'):'205,85,40';h+='<button class="day" data-date="'+date+'" style="background:rgba('+rgb+','+alpha+')" title="'+esc(p?`${date} — ${p.paired_hours} heures communes, NYX MAE ${fmt(p.baseline_mae_eur_mwh)}, candidat MAE ${fmt(p.candidate_mae_eur_mwh)}`:'Aucune donnée')+'">'+day+'<small>'+fmt(v,2)+'</small></button>';}$('calendar').innerHTML=h;$('calendar').querySelectorAll('[data-date]').forEach(b=>b.onclick=()=>{if([...$('day').options].some(o=>o.value===b.dataset.date)){$('day').value=b.dataset.date;dayChart();$('dailyChart').scrollIntoView({behavior:'smooth',block:'center'});}});}
function render(){annual();dayChart();hourly();calendar();}
const available=Object.keys(D.metrics.by_zone);$('country').innerHTML=['FR','DE','BE','NL'].map(z=>'<option '+(available.includes(z)?'':'disabled')+'>'+z+'</option>').join('');if(available.length)$('country').value=available.includes('FR')?'FR':available[0];
try{document.documentElement.dataset.theme=localStorage.getItem('nyx-scarcity-theme')||'light';}catch(_){}
$('theme').onclick=()=>{document.documentElement.dataset.theme=dark()?'light':'dark';try{localStorage.setItem('nyx-scarcity-theme',document.documentElement.dataset.theme);}catch(_){}render();};$('country').onchange=()=>{dates();render();};$('day').onchange=dayChart;$('showRaw').onchange=dayChart;$('month').onchange=calendar;$('calendarMetric').onchange=calendar;dates();render();
"""


def render_report(predictions: pd.DataFrame, metrics: Mapping[str, Any], daily: pd.DataFrame,
                  hourly: pd.DataFrame, audit: Mapping[str, Any], path: str | Path) -> Path:
    """Render a self-contained Plotly HTML diagnostic, including live display."""
    from plotly.offline import get_plotlyjs

    frame = _prepare(predictions)
    payload = {"predictions": frame.loc[:, _REPORT_PREDICTION_COLUMNS].to_dict("records"), "metrics": metrics,
               "daily": daily.to_dict("records"), "hourly": hourly.to_dict("records")}
    details = "".join('<details><summary>' + label + '</summary><pre>'
                      + html.escape(json.dumps(_clean(audit.get(key, {})), ensure_ascii=False, indent=2, default=str, allow_nan=False))
                      + '</pre></details>' for key, label in
                      (("data", "Sources, rôles des variables et couverture"), ("model", "Expert, gouvernance et limites"),
                       ("config", "Configuration et fenêtres d’entraînement")))
    document = '''<!doctype html><html lang="fr" data-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NYX — expert de rareté gouverné</title><style>''' + _STYLE + '''</style></head><body>
<header><div><small>EXPÉRIMENTATION SÉPARÉE · AUCUNE ACTIVATION</small><h1>NYX — expert de rareté gouverné</h1><p class="muted">Comparaison figée avec NYX et Storm · Jusqu’à 365 jours civils</p></div><button id="theme">Mode nuit / jour</button></header>
<main><section><div class="controls"><label>Pays <select id="country"></select></label></div>
<p class="warning">Diagnostic historique sur une année déjà examinée, sans preuve indépendante de disponibilité live/PIT. L’entraînement est progressif, de 90 à 365 jours : il ne dispose pas de 365 jours avant chaque date évaluée. Aucune activation ni garantie de non-régression. Les lacunes de données et la disponibilité des informations réseau à 08 h restent soumises aux audits.</p>
<p>Les scores utilisent les mêmes heures disposant d’une observation, de Storm, de NYX et du candidat gouverné. Les heures de retour au NYX initial restent dans le test annuel. Aucune donnée manquante n’est transformée en zéro. Ce rapport n’est ni un P&amp;L, ni une EVA, ni une attribution causale des variables.</p>
<p id="support" class="muted"></p><div id="cards" class="grid"></div></section>
<section data-report-section="statistics"><h2>Statistics — prix et erreurs sur heures communes</h2><div id="stats"></div><p id="coverage" class="muted"></p><p class="muted">Prix moyens et MAE en EUR/MWh. La MAE des prix moyens journaliers utilise exactement les heures appariées de chaque journée ; une journée partielle n’est pas un prix moyen journalier complet. Gain positif = erreur réduite.</p></section>
<section data-report-section="daily-forecast"><h2>Prévision journalière</h2><div class="controls"><label>Livraison <select id="day"></select></label><label><input type="checkbox" id="showRaw"> Proposition brute non gouvernée</label></div><p id="dayNote" class="muted"></p><div id="dailyChart" class="chart"></div><details><summary>Détail horaire et décision de gouvernance</summary><div id="hourRows"></div></details></section>
<section data-report-section="hourly-performance"><h2>Profil horaire moyen</h2><div id="hourlyPrice" class="chart"></div><div id="hourlyMae" class="chart"></div><p class="muted">Heures civiles locales ; les deux heures physiques du changement d’heure d’automne restent deux observations distinctes dans les agrégations.</p></section>
<section data-report-section="daily-calendar"><h2>Calendrier des erreurs quotidiennes</h2><div class="controls"><select id="month"></select><select id="calendarMetric"><option value="gain_baseline_minus_candidate_eur_mwh">Gain MAE : NYX − candidat</option><option value="candidate_mae_eur_mwh">MAE du candidat</option></select></div><p class="legend">Gain : vert = amélioration, rouge = dégradation. — = score indisponible. Cliquer une journée pour afficher les prévisions.</p><div id="calendar" class="calendar"></div></section>
<section data-report-section="tails"><h2>Prix extrêmes — cohortes ex post</h2><p>Top 1 % / 5 % définis par les prix observés sur le support apparié, uniquement pour l’analyse après coup. Ces cohortes ne sont pas des signaux utilisables à 08 h ; les égalités au seuil peuvent accroître leur taille.</p><div id="tails"></div></section>
<section data-report-section="active-subset"><h2>Heures d’intervention effective</h2><p>Sous-période où la correction appliquée est non nulle, sélectionnée par la gouvernance : ce résultat ne remplace jamais celui de l’année entière.</p><div id="active"></div><details><summary>Proposition brute non gouvernée : diagnostic, pas un modèle promu</summary><div id="raw"></div></details><details><summary>Comptage des motifs de décision</summary><pre id="gateReasons"></pre></details></section>
<section data-report-section="probabilistic"><h2>Intervalles et détection des dépassements</h2><div id="quantiles"></div><p>Intervalles sur le sous-ensemble commun possédant les quatre bornes P10/P90. Les quantiles manquants ne sont pas reconstruits. La couverture est descriptive, non une garantie de calibration.</p><div id="classifier"></div><p class="muted">Événement : erreur NYX (observé − prévision NYX) supérieure ou égale au seuil causal de résidu fourni par le modèle. Ce n’est pas un seuil de niveau de prix. Décision positive si la probabilité dépasse strictement le gate configuré. Le seuil historique n’est pas remplacé par les seuils ex post des tableaux précédents.</p><div id="calibration" class="chart"></div></section>
<section><h2>Méthode, sources et traçabilité</h2>''' + details + '''</section></main>
<script id="nyx-data" type="application/json">''' + _script_json(payload) + '''</script>
<script>''' + get_plotlyjs().replace('</script', '<\\/script') + '''</script><script>''' + _SCRIPT + '''</script></body></html>'''
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return target


__all__ = ["evaluate_predictions", "render_report"]
