"""Single-file CWE daily dashboard: nuclear Kalman, Storm and observed prices."""
from __future__ import annotations

from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
import re
import tempfile
import os

import pandas as pd
from plotly.offline import get_plotlyjs

from .model_storm_rolling import build_rolling_performance
from .model_storm_rolling_view import render_rolling_section


COLORS = {"observed": "#c084fc", "storm": "#00a8ff", "model": "#e48b23"}
LABELS = {"observed": "Observed", "storm": "Storm", "model": "Model"}
MODEL_RANGE_LABEL = "Model P10–P90"
OBSERVED_REJECTION_NOTE = (
    "Today's realized prices are not validated: sources disagree. "
    "Forecasts remain available; today's scores are not calculated."
)


def _observed_rejected(zone):
    return (zone.get("sources", {}).get("observed", {}).get("current_delivery_actual_reason")
            == "post_auction_source_rejected")


def _reference_label(zone):
    """Use snapshot provenance, never infer a source from today's policy."""
    label = zone.get("sources", {}).get("observed", {}).get("actual_reference_label")
    return label if label in ("EPEX", "ENTSO-E", "ENTSO-E + EPEX") else None


def _observed_label(zone):
    reference = _reference_label(zone)
    return f"Observed ({reference})" if reference else LABELS["observed"]


def _shared_observed_label(zones):
    labels = {_observed_label(zone) for zone in zones}
    return next(iter(labels)) if len(labels) == 1 else "Observed (reference per zone)"


def _reference_note(zones):
    rows = []
    for zone in zones:
        source = zone.get("sources", {}).get("observed", {})
        label = _reference_label(zone) or "reference not identified"
        extracted = source.get("extracted_at_utc")
        suffix = (" · extracted " + pd.Timestamp(extracted).tz_convert(zone["timezone"]).strftime("%Y-%m-%d %H:%M %Z")) if extracted else ""
        rows.append(escape(zone["zone"] + ": " + label + suffix))
    return ('<p class="reference-note" data-report-section="actual-price-reference">'
            '<strong>Realized DA prices and error reference:</strong> ' + '; '.join(rows)
            + '. Model and Storm errors use the same observed hourly prices from the selected snapshot.</p>')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _model_interval(row):
    low, middle, high = (row.get(key) for key in ("model_p10", "model", "model_p90"))
    if all(finite(value) for value in (low, middle, high)) and low <= middle <= high:
        return low, high
    return None, None


def _interval_segments(rows):
    """Never draw a filled polygon across missing or rejected hourly bounds."""
    segments, current = [], []
    for index, row in enumerate(rows):
        low, high = _model_interval(row)
        if low is not None:
            current.append((index, low, high))
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def daily_metrics(zone):
    """A daily mean requires all physical hours; never average a partial day."""
    rows = zone["rows"]
    expected = zone["expected_hours"]
    means = {}
    for key in COLORS:
        values = [row.get(key) for row in rows]
        means[key] = (sum(values) / expected if len(values) == expected and expected > 0
                      and all(finite(value) for value in values) else None)
    forecasts = [means["storm"], means["model"]]
    means["models_mean"] = sum(forecasts) / 2 if all(finite(v) for v in forecasts) else None
    means["models_std"] = abs(forecasts[0] - forecasts[1]) / 2 if means["models_mean"] is not None else None
    return means


def _number(value):
    return f"{value:.2f}" if finite(value) else "—"


def _card(zone, day):
    metrics = daily_metrics(zone)
    observed = metrics["observed"]
    reference = _reference_label(zone)
    realized_label = "Realized DA" + (f" · {reference}" if reference else "")
    rows = [f'<div class="metric observed"><span>{escape(realized_label)}</span><strong>{_number(observed)}'
            + (' €/MWh' if observed is not None else '') + '</strong></div>']
    for key in ("storm", "model"):
        value = metrics[key]
        badge = ""
        if finite(value) and finite(observed):
            delta = value - observed
            style, arrow = ("positive", "↑") if delta > 0 else ("negative", "↓") if delta < 0 else ("neutral", "=")
            badge = f'<span class="delta {style}" title="Forecast minus observed">{arrow} {abs(delta):.2f}</span>'
        rows.append(f'<div class="metric"><span>{LABELS[key]}</span><strong>{_number(value)} {badge}</strong></div>')
    mean = (f'{metrics["models_mean"]:.2f} ± {metrics["models_std"]:.2f} €/MWh'
            if metrics["models_mean"] is not None else '—')
    rows.append(f'<div class="metric models-mean"><span>Models Mean ± std</span><strong>{mean}</strong></div>')
    missing = []
    for key in ("model", "storm", "observed"):
        count = sum(finite(row.get(key)) for row in zone["rows"])
        if count < zone["expected_hours"]:
            if key == "observed" and _observed_rejected(zone):
                rejection_note = OBSERVED_REJECTION_NOTE
                extracted = zone["sources"]["observed"].get("extracted_at_utc")
                if extracted:
                    stamp = pd.Timestamp(extracted).tz_convert(zone["timezone"])
                    rejection_note += " Last source extraction: " + stamp.strftime("%Y-%m-%d %H:%M %Z") + "."
                missing.append(rejection_note)
                continue
            status = zone.get("sources", {}).get(key, {}).get("status")
            suffix = "source invalid" if status == "invalid" else "unavailable" if count == 0 else f'{count}/{zone["expected_hours"]} hours'
            missing.append(f'{LABELS[key]} {suffix}')
    note = escape(" · ".join(missing)) if missing else f'{zone["expected_hours"]} hourly prices · €/MWh'
    return (f'<article class="zone-card" aria-label="{escape(zone["name"])} daily snapshot">'
            f'<h3><span class="flag flag-{escape(zone["zone"].lower())}" aria-hidden="true"></span>'
            f'{escape(zone["name"])}: {escape(day)}</h3>' + ''.join(rows)
            + f'<p class="availability{ " pending" if missing else ""}">{note}</p></article>')


def _chart(payload):
    zones = payload["zones"]
    data, annotations = [], []
    range_legend_shown = False
    layout = dict(height=676, paper_bgcolor="#222729", plot_bgcolor="#222729",
                  font=dict(family="Arial, sans-serif", size=13, color="#e6e4df"),
                  margin=dict(l=86, r=78, t=98, b=110), hovermode="x unified",
                  legend=dict(orientation="h", x=.5, xanchor="center", y=-.14, yanchor="top",
                              groupclick="togglegroup", font=dict(size=13)),
                  hoverlabel=dict(bgcolor="#303638", font=dict(color="#fff")),
                  dragmode="zoom", uirevision="model-storm-daily")
    prices = [row[key] for z in zones for row in z["rows"] for key in COLORS if finite(row.get(key))]
    prices += [value for z in zones for row in z["rows"] for value in _model_interval(row) if finite(value)]
    errors = [row[key] - row["observed"] for z in zones for row in z["rows"]
              for key in ("storm", "model") if finite(row.get(key)) and finite(row.get("observed"))]
    low, high = (min(prices), max(prices)) if prices else (0, 100)
    span = max(high - low, 30)
    price_range = [min(0, low - .08 * span), high + .1 * span]
    bound = max([abs(v) for v in errors] or [30]) * 1.15
    for col, zone in enumerate(zones):
        rows = zone["rows"]
        count = len(rows)
        domain = [col * .25, col * .25 + .226]
        labels = [pd.Timestamp(r["timestamp_utc"]).tz_convert(zone["timezone"]) for r in rows]
        tickvals = [i for i, t in enumerate(labels) if t.hour in (0, 6, 12, 18)]
        ticktext = [labels[i].strftime("%H:%M") + ("<br>" + labels[i].strftime("%b %d, %Y") if i == 0 else "") for i in tickvals]
        tooltip = [t.strftime("%H:%M %Z · %Y-%m-%d") for t in labels]
        annotations.append(dict(x=sum(domain)/2, y=1.035, xref="paper", yref="paper", text=f'<b>{zone["zone"]}</b>',
                                showarrow=False, font=dict(size=16, color="#fff"), yanchor="bottom"))
        for lower in (False, True):
            number = col + 1 + (4 if lower else 0)
            suffix = str(number) if number > 1 else ""
            xref, yref = "x" + suffix, "y" + suffix
            ydomain = [0, .29] if lower else [.365, 1]
            layout["xaxis" + suffix] = dict(domain=domain, anchor=yref, range=[-.1, max(count-1, 1)+.1],
                tickmode="array", tickvals=tickvals, ticktext=ticktext, showticklabels=lower,
                gridcolor="#4a4e50", linecolor="#7f8182", showline=True, mirror=True,
                zeroline=False, tickcolor="#858789", ticks="outside", fixedrange=False)
            layout["yaxis" + suffix] = dict(domain=ydomain, anchor=xref,
                range=[-bound, bound] if lower else price_range, gridcolor="#4a4e50",
                linecolor="#7f8182", showline=True, mirror=True, showticklabels=col in (0, 3),
                side="right" if col == 3 else "left", zeroline=True,
                zerolinecolor="#a9aaab" if lower else "#ccc", zerolinewidth=1,
                title=dict(text=("Error (€/MWh)" if lower else "Price (€/MWh)") if col in (0, 3) else "",
                           font=dict(size=15), standoff=6))
            if not lower:
                for segment in _interval_segments(rows):
                    for upper in (False, True):
                        data.append(dict(type="scatter", mode="lines",
                            x=[point[0] for point in segment],
                            y=[point[2 if upper else 1] for point in segment],
                            xaxis=xref, yaxis=yref, name=MODEL_RANGE_LABEL,
                            legendgroup="model_range", legendrank=1001,
                            showlegend=upper and not range_legend_shown,
                            connectgaps=False, hoverinfo="skip",
                            line=dict(color="rgba(228,139,35,0.40)", width=.6),
                            fill="tonexty" if upper else "none", fillcolor="rgba(228,139,35,0.20)",
                            meta=dict(role="model_interval", quantile="p90" if upper else "p10")))
                        if upper:
                            range_legend_shown = True
            for key in (("storm", "model") if lower else ("observed", "storm", "model")):
                values = [(row[key] - row["observed"] if finite(row.get(key)) and finite(row.get("observed")) else None)
                          if lower else (row[key] if finite(row.get(key)) else None) for row in rows]
                label = _observed_label(zone) if key == "observed" else LABELS[key]
                error_reference = (" vs " + _reference_label(zone)) if lower and _reference_label(zone) else ""
                trace = dict(type="scatter", mode="lines", x=list(range(count)), y=values,
                    xaxis=xref, yaxis=yref, name=_shared_observed_label(zones) if key == "observed" else label, legendgroup=key,
                    showlegend=col == 0 and not lower, connectgaps=False,
                    line=dict(color=COLORS[key], width=2.4 if key == "observed" else 1.8),
                    customdata=tooltip,
                    hovertemplate="%{customdata}<br>" + label + (" error" + error_reference if lower else "")
                                  + ": %{y:.2f} €/MWh<extra></extra>")
                if key == "model" and not lower:
                    trace["text"] = [f"P10–P90: {_number(low)} – {_number(high)} €/MWh"
                                     if low is not None else "P10–P90 unavailable"
                                     for low, high in map(_model_interval, rows)]
                    trace["hovertemplate"] = "%{customdata}<br>Model: %{y:.2f} €/MWh<br>%{text}<extra></extra>"
                data.append(trace)
            if lower and not any(finite(row.get("observed")) for row in rows):
                message = ("Observed prices not validated<br>Sources disagree; no daily scores"
                           if _observed_rejected(zone) else "Observed prices unavailable")
            elif not lower and not any(finite(row.get(key)) for row in rows for key in COLORS):
                message = "Data unavailable"
            elif not lower and not any(finite(row.get("model")) for row in rows):
                message = "Model unavailable"
            else:
                message = ""
            if message:
                annotations.append(dict(x=sum(domain)/2, y=sum(ydomain)/2, xref="paper", yref="paper",
                    text=message, showarrow=False, bgcolor="rgba(34,39,41,.85)", borderpad=6,
                    font=dict(size=13, color="#c5c8ca")))
    layout["annotations"] = annotations
    return dict(data=data, layout=layout)


def _table(payload):
    output = []
    for zone in payload["zones"]:
        for row in zone["rows"]:
            stamp = pd.Timestamp(row["timestamp_utc"]).tz_convert(zone["timezone"]).strftime("%H:%M %Z")
            observed, storm, model = (row.get(k) for k in ("observed", "storm", "model"))
            values = [observed, storm, model,
                      *_model_interval(row),
                      storm-observed if finite(storm) and finite(observed) else None,
                      model-observed if finite(model) and finite(observed) else None]
            output.append(f'<tr data-zone="{escape(zone["zone"])}"><th scope="row">{escape(zone["zone"])}</th>'
                          f'<td title="{escape(row["timestamp_utc"])}">{escape(stamp)}</td>'
                          + ''.join('<td>'+_number(v)+'</td>' for v in values) + '</tr>')
    return ''.join(output)


def render_model_storm_report(payload: dict, output_path: Path) -> Path:
    day = payload["delivery_day"]
    if [z["zone"] for z in payload["zones"]] != ["BE", "DE", "FR", "NL"]:
        raise ValueError("The CWE dashboard requires BE, DE, FR and NL in this order")
    figure = json.dumps(_chart(payload), allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    cards = ''.join(_card(z, day) for z in payload["zones"])
    rolling = render_rolling_section(build_rolling_performance(payload))
    historical = payload.get("history_from_delivery")
    history_note = (f'<p class="history-note">Historical replay · source run <strong>{escape(str(historical))}</strong>. '
                    'Model values come from its archived backtest; Storm and observed prices come from verified local snapshots.</p>'
                    if historical else '')
    availability = ''.join('<li><strong>'+escape(z["name"])+':</strong> '
        + ' · '.join(f'{LABELS[k]} {sum(finite(r.get(k)) for r in z["rows"])}/{z["expected_hours"]} h'
                     for k in ("model", "storm", "observed")) + '</li>' for z in payload["zones"])
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CWE · Model &amp; Storm · __DAY__</title>
<style>
:root{color-scheme:dark;--bg:#222729;--text:#dedbd3;--muted:#9a978e;--blue:#2d5c80;--border:#d5d5d3}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,Helvetica,sans-serif;font-size:14px}
main{padding:16px 8px 8px;min-width:320px}h1{font-weight:400;font-size:28px;line-height:1.4;margin:0 0 10px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
.delivery-date{display:inline-block;border:1px solid #475157;padding:1px 11px;line-height:1.35;white-space:nowrap}
.history-note{color:#c3b99e;font-size:12px;line-height:1.6;margin:0 0 15px}
.panel{border:2px solid var(--border);border-radius:12px;padding:12px 11px;margin-bottom:16px}.panel h2{font-size:24px;font-weight:400;margin:0 0 10px;line-height:1.3}
.snapshot-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:24px}.zone-card{border:3px solid var(--blue);border-radius:9px;padding:24px 24px 18px;min-height:252px;display:flex;flex-direction:column}
.zone-card h3{font-size:18px;line-height:24px;margin:0 0 14px;display:flex;align-items:center;gap:13px;white-space:nowrap}
.flag{display:inline-block;width:34px;height:22px;border-radius:4px;flex:0 0 34px}.flag-be{background:linear-gradient(90deg,#080808 0 33.33%,#ffe526 33.33% 66.66%,#ed3447 66.66%)}
.flag-de{background:linear-gradient(#050505 0 33.33%,#e60013 33.33% 66.66%,#ffdc00 66.66%)}.flag-fr{background:linear-gradient(90deg,#00336b 0 33.33%,#fff 33.33% 66.66%,#e31339 66.66%)}
.flag-nl{background:linear-gradient(#b72037 0 33.33%,#fff 33.33% 66.66%,#254888 66.66%)}
.metric{display:grid;grid-template-columns:minmax(130px,1fr) minmax(160px,1fr);gap:8px;align-items:center;min-height:29px;font-size:13px}.metric>span{color:var(--muted)}.metric strong{white-space:nowrap;font-size:13px}
.metric.observed{font-size:14px;margin-bottom:2px}.metric.observed strong{font-size:14px}.delta{display:inline-flex;align-items:center;justify-content:center;min-width:49px;border-radius:6px;padding:3px 7px;color:white;font-size:11px;line-height:1.2;margin-left:9px}
.positive{background:#8cc446}.negative{background:#df5155}.neutral{background:#65717a}.models-mean{margin-top:5px}.availability{font-size:11px;color:#999c9d;margin:20px 0 0;line-height:1.5}.availability.pending{color:#c3b99e}
.charts-panel{padding:11px 11px 5px}.tabs{display:flex;justify-content:flex-end;border-bottom:1px solid #bcbfbd;gap:0;margin-top:3px;min-height:43px}
button,select{font:inherit}.tab{border:1px solid #62696d;border-bottom:0;border-radius:6px 6px 0 0;background:#222729;color:#dedbd3;font-size:14px;font-weight:700;padding:12px 15px;cursor:pointer;display:flex;align-items:center;gap:9px}
.tab[aria-selected=true]{background:#fff;color:#246591}.tab:hover{background:#3a4246}.tab[aria-selected=true]:hover{background:#fff}.tab svg{width:15px;height:15px;fill:none;stroke:currentColor;stroke-width:1.7}
.tab:focus-visible,select:focus-visible{outline:2px solid #00a8ff;outline-offset:3px}.tab.info{background:#3c3d44;color:#aaaeb1}.tab.info[aria-selected=true]{background:white;color:#246591}
.chart-scroll{overflow-x:auto}.chart{width:100%;min-width:1050px;min-height:676px}.tab-panel[hidden]{display:none}.table-tools{display:flex;justify-content:space-between;align-items:center;padding:18px 4px 12px;gap:15px;color:#bebfbf}
select{background:#2e363a;color:#eee;border:1px solid #7b858b;border-radius:4px;padding:7px 28px 7px 10px}.table-scroll{overflow:auto;max-height:610px}table{border-collapse:collapse;width:100%;font-size:13px}thead{position:sticky;top:0;background:#30383d}
th,td{text-align:right;padding:10px 15px;border-bottom:1px solid #4a5053;font-variant-numeric:tabular-nums}th:first-child,td:nth-child(2){text-align:left}tbody th{font-weight:700;color:#d6dce0}tbody tr:hover{background:#2e3639}
.info-content{padding:24px 15px 30px;max-width:1050px;font-size:14px;line-height:1.7;color:#c1c5c7}.info-content h3{font-size:19px;font-weight:400;color:#eee;margin:0 0 12px}.info-content li{margin:8px 0}.source-note{color:#949b9e;font-size:12px;margin-top:22px}
.footer{font-size:11px;color:#899295;display:flex;justify-content:space-between;padding:0 5px 4px;gap:15px}.no-js{padding:24px;color:#dfc899}
@media(max-width:1500px){.snapshot-grid{gap:16px}.zone-card{padding:22px 17px 16px}.zone-card h3{font-size:16px;gap:9px}.metric{grid-template-columns:minmax(112px,1fr) minmax(125px,1fr);font-size:12px}.metric strong{font-size:12px}.delta{margin-left:4px;min-width:43px;padding:3px 5px}}
@media(max-width:1200px){.snapshot-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.zone-card{min-height:238px}.metric{grid-template-columns:1fr 1fr}.zone-card h3{font-size:18px}}
@media(max-width:600px){main{padding:12px 7px}h1{font-size:23px;gap:6px}.panel h2{font-size:21px}.snapshot-grid{grid-template-columns:1fr;gap:12px}.zone-card{min-height:220px;padding:18px}.tab{font-size:12px;padding:11px}.panel{padding:10px}.table-tools{align-items:flex-start;flex-direction:column}.footer{flex-direction:column;gap:4px}.metric strong{font-size:13px}.chart{min-width:1150px}}
</style>
<script>__PLOTLY__</script></head><body><main>
<h1>CWE Day-Ahead - Daily Models Performance for <time class="delivery-date" datetime="__DAY__">__DAY__</time></h1>
__HISTORY_NOTE__
__REFERENCE_NOTE__
<section class="panel" aria-labelledby="snapshot-title"><h2 id="snapshot-title">Daily Snapshot</h2><div class="snapshot-grid">__CARDS__</div></section>
<section class="panel charts-panel" aria-labelledby="hourly-title"><h2 id="hourly-title">Hourly Forecasts per Zone</h2>
<div class="tabs" role="tablist" aria-label="Hourly forecast views">
<button class="tab" id="tab-graph" role="tab" aria-selected="true" aria-controls="view-graph" data-view="graph">GRAPH <svg viewBox="0 0 18 18" aria-hidden="true"><path d="M2 2v14h14M4 12l4-5 3 3 5-6"/></svg></button>
<button class="tab" id="tab-table" role="tab" aria-selected="false" aria-controls="view-table" tabindex="-1" data-view="table">TABLE <svg viewBox="0 0 18 18" aria-hidden="true"><rect x="2" y="3" width="14" height="12" rx="1"/><path d="M2 7h14M2 11h14M7 3v12M12 3v12"/></svg></button>
<button class="tab info" id="tab-info" role="tab" aria-selected="false" aria-controls="view-info" tabindex="-1" data-view="info">INFO <svg viewBox="0 0 18 18" aria-hidden="true"><circle cx="9" cy="9" r="6"/><path d="M9 8v5M9 5v1"/></svg></button>
</div>
<div class="tab-panel" id="view-graph" role="tabpanel" aria-labelledby="tab-graph"><div class="chart-scroll"><div class="chart" id="forecast-chart" aria-label="Hourly observed, Storm and Model prices with the Model P10–P90 range, and forecast errors for Belgium, Germany, France and the Netherlands"></div></div></div>
<div class="tab-panel" id="view-table" role="tabpanel" aria-labelledby="tab-table" hidden><div class="table-tools"><span>Hourly prices and errors · €/MWh · local delivery time</span><label>Zone <select id="table-zone"><option value="all">All zones</option><option>BE</option><option>DE</option><option>FR</option><option>NL</option></select></label></div>
<div class="table-scroll"><table><thead><tr><th scope="col">Zone</th><th scope="col">Hour</th><th scope="col">__OBSERVED_LABEL__</th><th scope="col">Storm</th><th scope="col">Model</th><th scope="col">Model P10</th><th scope="col">Model P90</th><th scope="col">Storm error</th><th scope="col">Model error</th></tr></thead><tbody>__ROWS__</tbody></table></div></div>
<div class="tab-panel info-content" id="view-info" role="tabpanel" aria-labelledby="tab-info" hidden><h3>Model, Storm and observed prices</h3>
<p><strong>Model</strong> is the nuclear Kalman forecast. <strong>Storm</strong> is the official Storm forecast. <strong>Observed</strong> is the available realized day-ahead price.</p>
<p>The orange <strong>Model P10–P90</strong> band shows the hourly 10th and 90th forecast percentiles. The Model line is the median (P50). Hover over Model or open TABLE to read the bounds; click the range legend to show or hide the band.</p>
<p>Errors and snapshot badges show <strong>forecast − observed</strong>, in €/MWh. Green ↑ means above the observed price; red ↓ means below it. These colors indicate direction, not forecast quality.</p>
<p>Daily means require every physical hour of the delivery day. Missing or incomplete series display —. Models Mean ± std combines the two complete daily forecast means (Model and Storm), using their population standard deviation.</p>
<p>The charts use local delivery times and include all 23, 24 or 25 hours on daylight-saving days. The table distinguishes repeated hours by their timezone. Missing values are never replaced by zero.</p>
<h3>Availability for __DAY__</h3><ul>__AVAILABILITY__</ul>
<p class="source-note">This file contains the results available when it was generated. Germany is included even when its Model forecast is unavailable. Regenerate the report to include newly completed results.</p></div>
<noscript><p class="no-js">JavaScript is required for the interactive charts. Daily snapshot values remain available above.</p></noscript>
</section>__ROLLING_SECTION__<footer class="footer"><span>Model · Storm · Observed — CWE day-ahead</span><span>Generated __GENERATED__ · standalone HTML</span></footer>
</main><script id="chart-data" type="application/json">__FIGURE__</script>
<script>
const figure=JSON.parse(document.getElementById('chart-data').textContent);
Plotly.newPlot('forecast-chart',figure.data,figure.layout,{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:'CWE_Model_Storm___DAY__',scale:2},modeBarButtonsToRemove:['lasso2d','select2d']});
const tabs=[...document.querySelectorAll('[role=tab]')];
function showView(tab){tabs.forEach(item=>{const active=item===tab;item.setAttribute('aria-selected',String(active));item.tabIndex=active?0:-1;document.getElementById('view-'+item.dataset.view).hidden=!active;});if(tab.dataset.view==='graph')Plotly.Plots.resize('forecast-chart');}
tabs.forEach((tab,i)=>{tab.addEventListener('click',()=>showView(tab));tab.addEventListener('keydown',event=>{let next;if(event.key==='ArrowRight')next=(i+1)%tabs.length;else if(event.key==='ArrowLeft')next=(i+tabs.length-1)%tabs.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=tabs.length-1;else return;event.preventDefault();tabs[next].focus();showView(tabs[next]);});});
document.getElementById('table-zone').addEventListener('change',event=>{document.querySelectorAll('#view-table tbody tr').forEach(row=>{row.hidden=event.target.value!=='all'&&row.dataset.zone!==event.target.value;});});
</script></body></html>'''
    # One pass prevents marker-like input text from becoming template syntax.
    replacements = {"__DAY__": escape(day), "__CARDS__": cards, "__ROWS__": _table(payload),
                    "__AVAILABILITY__": availability, "__GENERATED__": escape(str(payload.get("generated_at", datetime.now().isoformat(timespec="minutes")))),
                    "__FIGURE__": figure, "__PLOTLY__": get_plotlyjs(), "__HISTORY_NOTE__": history_note,
                    "__REFERENCE_NOTE__": _reference_note(payload["zones"]),
                    "__ROLLING_SECTION__": rolling,
                    "__OBSERVED_LABEL__": escape(_shared_observed_label(payload["zones"]))}
    template = re.sub('|'.join(map(re.escape, replacements)), lambda match: replacements[match.group()], template)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".model-storm-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(template)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
