"""Offline, accessible rolling comparison tables for the CWE HTML report."""
from __future__ import annotations

from html import escape
from decimal import Decimal, ROUND_HALF_UP
from copy import deepcopy
import json
import math


STRATEGIES = (("quantile_based", "Quantile Based"), ("unlimited_bid", "Unlimited Bid"))

COLUMNS = (
    ("provider", "Provider", "Forecast source"),
    ("mae", "MAE", "Mean absolute error · EUR/MWh · lower is better"),
    ("bias", "BIAS", "Forecast minus observed · EUR/MWh · nearer zero is better"),
    ("rmse", "RMSE", "Root mean squared error · EUR/MWh · lower is better"),
    ("hit_rate", "Hit Rate (±€5)", "Share of absolute errors at most 5 EUR/MWh · higher is better"),
    ("r2", "R²", "1 − squared error / observed variance · higher is better; may be negative"),
    ("daily_pnl", "Daily P&L (sim.)", "Mean daily strategy profit on eligible complete days · EUR/day · higher is better"),
)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _format(key, value):
    if not _finite(value):
        return "—"
    if key == "hit_rate":
        return str((Decimal(str(value)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) + " %"
    return f"{value:.2f}" + (" €" if key == "daily_pnl" else "")


def _best(rows, key, value):
    if key == "daily_pnl" and any(row.get("pnl_comparison_eligible") is False for row in rows):
        return False
    if any(row.get("comparison_eligible") is False for row in rows):
        return False
    values = [row.get(key) for row in rows if _finite(row.get(key))]
    if len(values) < 2 or not _finite(value):
        return False
    if key == "bias":
        values, value = list(map(abs, values)), abs(value)
    optimum = min(values) if key in ("mae", "bias", "rmse") else max(values)
    return math.isclose(value, optimum, rel_tol=1e-10, abs_tol=1e-10)


def _providers(window, frequency):
    return window.get("frequencies", {}).get(frequency, {}).get("providers", []) or [
        {"key": "storm", "label": "Storm"}, {"key": "model", "label": "Model"}]


def _rows(providers):
    result = []
    for row in providers:
        cells = []
        for key, _, _ in COLUMNS[1:]:
            winner = _best(providers, key, row.get(key))
            title = _pnl_title(row) if key == "daily_pnl" else ""
            if winner:
                title = "Best value on the matched sample" + (" · " + title if title else "")
            cells.append('<td data-column="' + key + '"'
                         + (' class="rolling-best"' if winner else '')
                         + (' title="' + escape(str(title), quote=True) + '"' if title else '')
                         + '>' + _format(key, row.get(key)) + '</td>')
        result.append('<tr data-provider="' + escape(row.get("key", ""), quote=True) + '"><th scope="row">'
                      + escape(row.get("label", row.get("key", ""))) + '</th>' + ''.join(cells) + '</tr>')
    return ''.join(result)


def _coverage(window, frequency):
    if not window:
        return "Verified rolling history unavailable."
    samples = window.get("frequencies", {}).get(frequency, {}).get("samples", 0)
    unit = "hourly pairs" if frequency == "60min" else "complete daily pairs"
    return (f"{samples} {unit} · {window.get('paired_hours', 0)}/{window.get('expected_hours', 0)} matched hours"
            f" · {window.get('complete_paired_days', 0)} complete paired days"
            f" · P&L: {window.get('pnl_days', 0)} shared eligible days")


def _pnl_note(window, frequency):
    providers = _providers(window, frequency)
    details = []
    for row in providers:
        label = row.get("label", row.get("key", ""))
        count = window.get("pnl_provider_days", {}).get(row.get("key"), row.get("pnl_days", 0))
        summary = f"{label}: {count} eligible days"
        if _finite(row.get("total_pnl")):
            summary += f", {row['total_pnl']:.2f} € total"
        if row.get("trade_days") is not None:
            summary += f", {row['trade_days']} executed cycles"
        details.append(summary)
        if row.get("pnl_unavailable_reason"):
            details.append(f"{label}: {row['pnl_unavailable_reason']}")
    if any(row.get("pnl_comparison_eligible") is False for row in providers):
        details.append("P&L ranking disabled: comparable support unavailable.")
    return " · ".join(details)


def _pnl_title(row):
    if row.get("pnl_unavailable_reason"):
        return str(row["pnl_unavailable_reason"])
    parts = []
    if _finite(row.get("total_pnl")):
        parts.append(f"Total simulated profit: {row['total_pnl']:.2f} €")
    if row.get("trade_days") is not None:
        parts.append(f"{row['trade_days']} executed cycles")
    if row.get("pnl_days") is not None:
        parts.append(f"{row['pnl_days']} eligible days")
    return " · ".join(parts)


def _strategy_note(strategy, alpha):
    common = "1 MWh storage · 95% efficiency on each leg · €25 per executed cycle · one purchase before one sale per day. "
    if strategy == "quantile_based":
        return common + (f"Quantile Based: α = {alpha:.0%}, NYX native P10/P90; Storm calibrated from past errors "
                         "(at least 60 prior complete days). Both strategies use the same eligible completed history.")
    return common + ("Unlimited Bid: hours chosen from point forecasts; execution valued at observed prices. "
                     "Both strategies use the same eligible completed history, after calibration warmup.")


CSS = r'''
#rolling-performance{margin:28px 0 16px;min-width:0}
#rolling-performance>.rolling-title{display:flex;align-items:center;flex-wrap:wrap;gap:12px;font-size:28px;font-weight:400;margin:0 0 10px}
#rolling-performance .rolling-construction{display:inline-flex;align-items:center;justify-content:center;flex:none;width:34px;height:34px;border:1px solid #b78039;border-radius:7px;background:#f1ae4710;color:#f1ae47;box-shadow:0 0 12px #f1ae4714;vertical-align:middle}
#rolling-performance .rolling-construction svg{width:24px;height:24px}
#rolling-performance .rolling-control{display:flex;align-items:center;flex-wrap:wrap;gap:7px;margin:4px 0}
#rolling-performance .rolling-buttons{display:inline-flex;flex-wrap:wrap;border:1px solid #38658b;border-radius:5px;overflow:hidden}
#rolling-performance .rolling-toggle{padding:11px 14px;border:0;border-right:1px solid #38658b;background:#222729;color:#8ab9dc;cursor:pointer;font-size:12px;font-weight:700;min-height:38px}
#rolling-performance .rolling-toggle:last-child{border-right:0}
#rolling-performance .rolling-toggle[aria-pressed=true]{background:#355e85;color:#fff}
#rolling-performance button:focus-visible{outline:2px solid #00a8ff;outline-offset:-3px}
#rolling-performance .rolling-panel{margin-top:9px}
#rolling-performance .rolling-panel>h3{font-size:24px;font-weight:400;margin:0 0 12px}
#rolling-performance .rolling-subtitle{font-size:18px;margin:0 0 8px}
#rolling-performance .rolling-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px 24px}
#rolling-performance .rolling-zone{min-width:0}
#rolling-performance .rolling-zone h4{display:flex;align-items:center;gap:13px;font-size:15px;font-weight:400;margin:0 0 9px}
#rolling-performance .rolling-table-scroll{overflow:auto;border:1px solid #464e53;border-radius:3px}
#rolling-performance table{min-width:700px;font-size:13px;width:100%;border-collapse:collapse}
#rolling-performance thead{background:#333b40;position:static}
#rolling-performance th,#rolling-performance td{padding:15px 12px;text-align:right;white-space:nowrap;border:1px solid #454c50;border-top:0}
#rolling-performance th:first-child{min-width:116px;text-align:left}
#rolling-performance td:nth-child(2){text-align:right}
#rolling-performance tbody tr:nth-child(even){background:#2b3134}
#rolling-performance tbody tr:hover{background:#303a40}
#rolling-performance .rolling-best{background:#d2eedc!important;color:#123b21!important;font-weight:700}
#rolling-performance .rolling-sort{font:inherit;font-weight:700;color:inherit;background:none;border:0;cursor:pointer;padding:0;min-height:22px;white-space:nowrap}
#rolling-performance .rolling-arrow{display:inline-block;min-width:11px;font-size:12px;color:#a7cbe6;margin-left:4px}
#rolling-performance .rolling-coverage{font-size:11px;color:#c4c9cc;line-height:1.6;margin:7px 0 0}
#rolling-performance .rolling-pnl-note{font-size:11px;font-style:italic;color:#b9b8b1;line-height:1.5;margin:3px 0 0}
#rolling-performance .rolling-explainer{font-size:12px;line-height:1.6;color:#c4c9cc;margin:10px 0 18px}
#rolling-performance .rolling-best-key{display:inline-block;background:#d2eedc;color:#123b21;padding:0 5px;border-radius:2px;font-weight:700}
#rolling-performance .rolling-strategy-note{font-size:12px;color:#c4c9cc;line-height:1.65;margin:8px 0 12px}
#rolling-performance details{margin-top:20px;line-height:1.65;font-size:12px;color:#c4c9cc}
#rolling-performance summary{cursor:pointer;color:#e1dfd8;font-size:13px}
#rolling-performance details p{max-width:1300px}
@media(max-width:1100px){#rolling-performance .rolling-grid{grid-template-columns:1fr}}
@media(max-width:600px){#rolling-performance>.rolling-title{font-size:23px}#rolling-performance .rolling-toggle{padding:10px 9px;font-size:11px}#rolling-performance .rolling-zone h4{font-size:13px;gap:8px}#rolling-performance .rolling-control{font-size:12px}}
'''


JS = r"""
(()=>{
  const root=document.getElementById('rolling-performance');
  const data=JSON.parse(document.getElementById('rolling-performance-data').textContent);
  let period=data.default_window,frequency=data.default_frequency,strategy=data.default_strategy||'unlimited_bid';
  const sorts=new Map();
  const valid=v=>typeof v==='number'&&Number.isFinite(v);
  const format=(key,v)=>!valid(v)?'—':key==='hit_rate'?(v*100).toFixed(1)+' %':v.toFixed(2)+(key==='daily_pnl'?' €':'');
  const columns=['mae','bias','rmse','hit_rate','r2','daily_pnl'];
  const betterLow=key=>['mae','bias','rmse'].includes(key);
  const score=(key,v)=>key==='bias'?Math.abs(v):v;
  function isBest(rows,key,v){
    if(key==='daily_pnl'&&rows.some(row=>row.pnl_comparison_eligible===false))return false;
    if(rows.some(row=>row.comparison_eligible===false))return false;
    const values=rows.map(row=>row[key]).filter(valid).map(value=>score(key,value));
    if(values.length<2||!valid(v))return false;
    const best=betterLow(key)?Math.min(...values):Math.max(...values),current=score(key,v);
    return Math.abs(current-best)<=Math.max(1e-10,Math.abs(best)*1e-10);
  }
  function draw(){
    root.dataset.activeStrategy=strategy;
    root.querySelectorAll('[data-rolling-strategy]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.rollingStrategy===strategy)));
    root.querySelector('.rolling-strategy-note').textContent=data.strategy_notes[strategy];
    root.querySelectorAll('[data-rolling-period]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.rollingPeriod)===period)));
    root.querySelectorAll('[data-rolling-frequency]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.rollingFrequency===frequency)));
    const zones=data.strategy_zones[strategy]||[];
    zones.forEach(zone=>{
      const card=root.querySelector('[data-rolling-zone="'+zone.zone+'"]');
      const win=zone.anchor_day===null?{}:(zone.windows||{})[String(period)]||{},view=(win.frequencies||{})[frequency]||{};
      const rows=(view.providers&&view.providers.length?view.providers:[{key:'storm',label:'Storm'},{key:'model',label:'Model'}]).slice();
      const order=sorts.get(zone.zone);
      if(order)rows.sort((a,b)=>{
        if(order.key==='provider')return order.direction*(a.label||a.key).localeCompare(b.label||b.key);
        const av=a[order.key],bv=b[order.key];
        if(!valid(av)||!valid(bv))return valid(av)?-1:valid(bv)?1:0;
        return order.direction*(score(order.key,av)-score(order.key,bv));
      });
      card.querySelector('.rolling-dates').textContent=win.start_day&&win.end_day?win.start_day+' — '+win.end_day:'No complete observed day';
      card.querySelector('.rolling-coverage').textContent=win.start_day?
        (view.samples||0)+(frequency==='60min'?' hourly pairs':' complete daily pairs')+' · '+(win.paired_hours||0)+'/'+(win.expected_hours||0)+' matched hours · '+(win.complete_paired_days||0)+' complete paired days · P&L: '+(win.pnl_days||0)+' shared eligible days':
        'Verified rolling history unavailable.';
      card.querySelector('.rolling-pnl-note').textContent=win.pnl_notes?.[frequency]||'';
      const body=card.querySelector('tbody');body.replaceChildren();
      rows.forEach(row=>{
        const tr=document.createElement('tr');tr.dataset.provider=row.key;
        const head=document.createElement('th');head.scope='row';head.textContent=row.label||row.key;tr.appendChild(head);
        columns.forEach(key=>{
          const td=document.createElement('td');td.dataset.column=key;td.textContent=row.display_values?.[key]??format(key,row[key]);
          if(key==='daily_pnl'&&row.pnl_tooltip)td.title=row.pnl_tooltip;
          if(isBest(rows,key,row[key])){td.className='rolling-best';td.title='Best value on the matched sample'+(td.title?' · '+td.title:'');}
          tr.appendChild(td);
        });
        body.appendChild(tr);
      });
      card.querySelectorAll('thead th').forEach(th=>{
        const key=th.querySelector('button').dataset.rollingSort,active=order&&order.key===key;
        th.setAttribute('aria-sort',active?(order.direction===1?'ascending':'descending'):'none');
        th.querySelector('.rolling-arrow').textContent=active?(order.direction===1?'↑':'↓'):'↕';
      });
    });
  }
  root.querySelectorAll('[data-rolling-period]').forEach(b=>b.addEventListener('click',()=>{period=Number(b.dataset.rollingPeriod);draw();}));
  root.querySelectorAll('[data-rolling-strategy]').forEach(b=>b.addEventListener('click',()=>{strategy=b.dataset.rollingStrategy;sorts.clear();draw();}));
  root.querySelectorAll('[data-rolling-frequency]').forEach(b=>b.addEventListener('click',()=>{frequency=b.dataset.rollingFrequency;draw();}));
  root.querySelectorAll('[data-rolling-sort]').forEach(b=>b.addEventListener('click',()=>{
    const zone=b.closest('[data-rolling-zone]').dataset.rollingZone,key=b.dataset.rollingSort,previous=sorts.get(zone);
    sorts.set(zone,{key,direction:previous&&previous.key===key?-previous.direction:(key==='provider'||betterLow(key)?1:-1)});draw();
  }));
})();
"""


def render_rolling_section(data: dict) -> str:
    """Render completed-history scores with two paper strategy P&L tabs."""
    data = deepcopy(data)
    period, frequency = data["default_window"], data["default_frequency"]
    strategy = data.get("default_strategy", "unlimited_bid")
    if strategy not in dict(STRATEGIES):
        raise ValueError("Unsupported P&L strategy.")
    strategy_zones = data.get("strategy_zones")
    if not strategy_zones or any(key not in strategy_zones for key, _ in STRATEGIES):
        raise ValueError("Both paper strategy tables are required; legacy P&L cannot be relabelled.")
    data["strategy_notes"] = {key: _strategy_note(key, data.get("quantile_alpha", .8)) for key, _ in STRATEGIES}
    for zones in strategy_zones.values():
        for zone in zones:
            for window in zone.get("windows", {}).values():
                window["pnl_notes"] = {key: _pnl_note(window, key) for key in ("60min", "day")}
                for view in window.get("frequencies", {}).values():
                    for row in view.get("providers", []):
                        row["display_values"] = {key: _format(key, row.get(key)) for key, _, _ in COLUMNS[1:]}
                        row["pnl_tooltip"] = _pnl_title(row)
    controls = '<div class="rolling-control"><span>P&amp;L strategy:</span><div class="rolling-buttons" role="group" aria-label="P&L strategy">'
    for key, label in STRATEGIES:
        controls += (f'<button type="button" class="rolling-toggle" data-rolling-strategy="{key}" '
                     f'aria-pressed="{str(key == strategy).lower()}">{label}</button>')
    controls += '</div></div><p class="rolling-strategy-note">' + escape(data["strategy_notes"][strategy]) + '</p>'
    controls += '<div class="rolling-control"><span>Rolling period:</span><div class="rolling-buttons" role="group" aria-label="Rolling period">'
    for days in data["windows"]:
        controls += (f'<button type="button" class="rolling-toggle" data-rolling-period="{int(days)}" '
                     f'aria-pressed="{str(days == period).lower()}">LAST {int(days)} DAYS</button>')
    controls += '</div></div><div class="rolling-control"><span>Sampling frequency:</span><div class="rolling-buttons" role="group" aria-label="Sampling frequency">'
    for key, label in (("60min", "60-MIN"), ("day", "DAY")):
        controls += (f'<button type="button" class="rolling-toggle" data-rolling-frequency="{key}" '
                     f'aria-pressed="{str(key == frequency).lower()}">{label}</button>')
    controls += '</div></div>'
    headers = ''.join(f'<th scope="col" aria-sort="none" data-column="{key}"><button type="button" class="rolling-sort" '
                      f'data-rolling-sort="{key}" title="{escape(title, quote=True)}">{escape(label)}'
                      '<span class="rolling-arrow" aria-hidden="true">↕</span></button></th>'
                      for key, label, title in COLUMNS)
    cards = []
    for zone in strategy_zones[strategy]:
        window = {} if zone.get("anchor_day", "missing") is None else zone.get("windows", {}).get(str(period), {})
        dates = (str(window["start_day"]) + ' — ' + str(window["end_day"])) if window.get("start_day") else "No complete observed day"
        code = escape(zone["zone"], quote=True)
        cards.append(f'<article class="rolling-zone" data-rolling-zone="{code}">'
                     f'<h4><span class="flag flag-{code.lower()}" aria-hidden="true"></span><span>{escape(zone["name"])}: '
                     f'<span class="rolling-dates">{escape(dates)}</span></span></h4>'
                     '<div class="rolling-table-scroll" role="region" tabindex="0" '
                     f'aria-label="{escape(zone["name"], quote=True)} rolling performance"><table>'
                     f'<thead><tr>{headers}</tr></thead><tbody>{_rows(_providers(window, frequency))}</tbody></table></div>'
                     f'<p class="rolling-coverage">{escape(_coverage(window, frequency))}</p>'
                     f'<p class="rolling-pnl-note">{escape(_pnl_note(window, frequency) if window else "")}</p></article>')
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return ('<style>' + CSS + '</style><section id="rolling-performance" aria-labelledby="rolling-performance-title"'
            f' data-active-strategy="{strategy}">'
            '<h2 class="rolling-title" id="rolling-performance-title">CWE Day-Ahead - Rolling Models Performance'
            '<span class="rolling-construction" role="img" aria-label="En cours de construction" title="En cours de construction">'
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="M5 16v5m14-5v5"/><rect x="3" y="6" width="18" height="10" rx="1.5"/>'
            '<path d="m4 7 6 8m1-8 6 8m1-8 2 3"/></svg></span></h2>'
            + controls + '<div class="panel rolling-panel"><h3>Rolling Performance</h3><h4 class="rolling-subtitle">Summary Tables</h4>'
            '<p class="rolling-explainer"><span class="rolling-best-key">Green = best</span> on the same matched sample; '
            'bias is compared by distance to zero. Ties are highlighted together. Switching strategy changes P&amp;L only.</p>'
            '<div class="rolling-grid" aria-live="polite">' + ''.join(cards) + '</div>'
            '<details><summary>Definitions, coverage and strategy assumptions</summary>'
            '<p><strong>Completed history:</strong> each window ends on the last complete observed local day available at or before the report date, independently per country. '
            'Model and Storm are scored on exactly the same finite observations. Missing prices are never replaced by zero. '
            '60-MIN preserves all physical hours, including 23/25-hour DST days. DAY averages each complete physical day first, '
            'then scores those daily means with equal weight per day; it is not the average hourly MAE.</p>'
            '<p>MAE = mean absolute error; BIAS = mean(forecast − observed); RMSE = root mean squared error, all in EUR/MWh. '
            'Hit Rate (±€5) counts |forecast − observed| ≤ 5 EUR/MWh, not directional accuracy or win rate against Storm. '
            'R² = 1 − Σ(error²)/Σ(observed − mean observed)²; it is unavailable for a constant observed sample.</p>'
            '<p><strong>Paper strategies:</strong> section 6.2 of <cite>arXiv:2609.00089v1</cite>. '
            'A 1 MWh battery buys before it sells, with at most one cycle per day, 95% charging efficiency, 95% discharging efficiency and €25 per executed cycle. '
            'Decisions use forecasts only; profit uses observed prices. Unlimited Bid uses point forecasts. '
            'Quantile Based uses P10/P90 (α = 80%): native NYX quantiles and Storm quantiles calibrated from past hourly errors only. '
            'Storm calibration uses a trailing 365-day window, at least 60 prior complete days and 30 samples per local hour, '
            'with centering on the historical median error to preserve the point forecast. '
            'A day with available inputs and no trade has zero profit; a day with missing inputs has no P&amp;L value.</p>'
            '<p>Daily P&amp;L is total strategy profit divided by eligible complete days shared by both strategies and providers. '
            'Calibration warmup and days with missing quantiles are excluded from both P&amp;L tabs. The eligible count is shown for each provider. '
            'P&amp;L ranking requires the same eligible days for both providers. '
            'DAY retains the same hourly strategy decisions and P&amp;L. Other forecast metrics keep their original paired sample.</p>'
            '<p>These are simulated strategy results, not realized trading gains. Bid acceptance is a price-taking loop approximation; '
            'market-level paradoxical bid rejection is not modelled. The 23/25-hour DST treatment extends the hourly paper setting. '
            'Only locally available verified Model and Storm results are included. Backtest-based history is retrospective, '
            'not proof of prospective execution. Operational price forecasts are not retrained or changed by this report.</p>'
            '</details></div></section><script id="rolling-performance-data" type="application/json">' + encoded + '</script>'
            '<script>' + JS + '</script>')
