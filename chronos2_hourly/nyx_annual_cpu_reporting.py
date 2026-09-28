"""Standalone, dependency-free HTML reports for the four annual CPU forecasts."""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .nyx_annual_live_preflight import delivery_grid


COUNTRIES = {"FR": ("France", "Europe/Paris"), "DE": ("Allemagne", "Europe/Berlin"),
             "BE": ("Belgique", "Europe/Brussels"), "NL": ("Pays-Bas", "Europe/Amsterdam")}


def _price_chart(prices: np.ndarray, local: pd.DatetimeIndex) -> str:
    width, height, left, right, top, bottom = 920, 270, 72, 20, 24, 42
    plot_width, plot_height = width-left-right, height-top-bottom
    scale = max(1., float(np.max(np.abs(prices))))
    normalized = prices/scale
    low, high = float(normalized.min()), float(normalized.max())
    padding = max((high-low)*.12, .01)
    lower, upper = low-padding, high+padding
    x = lambda i: left + (i+.5)*plot_width/len(prices)
    y = lambda p: top+(upper-p)/(upper-lower)*plot_height
    marks = []
    for level in (low, (low+high)/2, high):
        marks.append(f'<line x1="{left}" x2="{width-right}" y1="{y(level):.2f}" y2="{y(level):.2f}" class="grid"/>'
                     f'<text x="{left-10}" y="{y(level)+4:.2f}" text-anchor="end">{level*scale:.2f}</text>')
    if lower <= 0 <= upper:
        marks.append(f'<line x1="{left}" x2="{width-right}" y1="{y(0):.2f}" y2="{y(0):.2f}" class="zero"/>')
    points = ' '.join(f'{x(i):.2f},{y(p):.2f}' for i,p in enumerate(normalized))
    marks.append(f'<polyline points="{points}" class="price-line"/>')
    for i, (price, stamp) in enumerate(zip(prices, local)):
        marks.append(f'<circle cx="{x(i):.2f}" cy="{y(normalized[i]):.2f}" r="3.5" class="{"negative-point" if price < 0 else "price-point"}">'
                     f'<title>{escape(stamp.isoformat())} : {price:.2f} €/MWh</title></circle>')
        if i % 3 == 0 or i == len(prices)-1:
            marks.append(f'<text x="{x(i):.2f}" y="{height-15}" text-anchor="middle">{stamp:%H:%M}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Prix prévus heure par heure">'
            '<title>Prix prévus en euros par mégawattheure</title>' + ''.join(marks) + '</svg>')


def _probability_chart(probabilities: np.ndarray, local: pd.DatetimeIndex) -> str:
    width, height, left, right, top, bottom = 920, 200, 72, 20, 18, 38
    plot_width, plot_height = width-left-right, height-top-bottom
    cell = plot_width/len(probabilities)
    marks = []
    for p in (0., .5, 1.):
        y = top+(1-p)*plot_height
        marks.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.2f}" y2="{y:.2f}" class="{"threshold" if p == .5 else "grid"}"/>'
                     f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end">{p:.0%}</text>')
    for i, (probability, stamp) in enumerate(zip(probabilities, local)):
        x, y, bar_height = left+i*cell+3, top+(1-probability)*plot_height, probability*plot_height
        marks.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell-6:.2f}" height="{bar_height:.2f}" rx="2" '
                     f'class="{"probability-alert" if probability >= .5 else "probability-bar"}"><title>'
                     f'{escape(stamp.isoformat())} : {probability:.1%}</title></rect>')
        if i % 3 == 0 or i == len(probabilities)-1:
            marks.append(f'<text x="{left+(i+.5)*cell:.2f}" y="{height-13}" text-anchor="middle">{stamp:%H:%M}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Probabilité de prix négatif heure par heure">'
            '<title>Probabilité que le prix observé soit inférieur à zéro</title>' + ''.join(marks) + '</svg>')


def render_country_report(frame: pd.DataFrame, *, country: str, delivery_day: str,
                          de_performance_exception: bool = False) -> str:
    """Render exact physical-hour values, including both local DST occurrences."""
    if country not in COUNTRIES:
        raise ValueError("Unknown annual report country")
    _, current, _ = delivery_grid(delivery_day)
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.equals(current):
        raise ValueError("The report requires the exact physical delivery-day UTC grid")
    prices, probabilities = (frame[name].to_numpy(float) for name in ("price_eur_mwh", "p_negative"))
    if not (np.isfinite(prices).all() and np.isfinite(probabilities).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("Finite prices and probabilities in [0,1] required")
    name, timezone = COUNTRIES[country]
    local = current.tz_convert(timezone)
    rows = []
    for stamp, utc, price, probability in zip(local, current, prices, probabilities):
        offset = stamp.strftime("%z")
        offset = offset[:3]+":"+offset[3:]
        rows.append(f'<tr><td>{stamp:%d/%m/%Y %H:%M} <span class="offset">UTC{offset}</span></td>'
                    f'<td>{utc:%Y-%m-%d %H:%M}Z</td><td class="number {"below-zero" if price < 0 else ""}">{price:.2f}</td>'
                    f'<td class="number">{probability:.1%}</td>'
                    f'<td>{"Oui" if probability >= .5 else "—"}</td></tr>')
    note = ('<aside class="exception"><strong>DE : exception de performance.</strong> '
            'Ce modèle est inclus à votre demande alors que tous les critères contre Storm ne sont pas validés. '
            'Les contrôles de la chaîne de données et de calcul restent obligatoires.</aside>'
            if country == "DE" and de_performance_exception else '')
    day_label = pd.Timestamp(delivery_day).strftime("%d/%m/%Y")
    mean_price = float(np.sum(prices/len(prices)))
    return f'''<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NYX — {escape(name)} — {escape(day_label)}</title>
<style>
:root{{color-scheme:light;font-family:Arial,sans-serif;color:#172b42;background:#f4f7fb}}
*{{box-sizing:border-box}}body{{margin:0;padding:28px}}main{{max-width:1120px;margin:auto}}
header{{margin-bottom:24px}}.eyebrow{{color:#3a617b;font-size:13px;letter-spacing:.12em;text-transform:uppercase}}
h1{{font-size:30px;margin:10px 0}}h2{{font-size:19px;margin:0 0 12px}}p{{line-height:1.55}}.muted{{color:#52677b}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}
.card,section{{background:white;border:1px solid #d8e1ec;border-radius:12px;padding:18px}}
.card small{{display:block;color:#52677b;margin-bottom:8px}}.card strong{{font-size:24px;font-variant-numeric:tabular-nums}}
section{{margin:16px 0}}.exception{{border-left:5px solid #b36600;background:#fff2d9;padding:16px;line-height:1.6}}
svg{{display:block;width:100%;height:auto}}svg text{{font:12px Arial,sans-serif;fill:#52677b}}
.grid{{stroke:#e0e7ef;stroke-width:1}}.zero{{stroke:#a24045;stroke-width:1;stroke-dasharray:4 3}}
.price-line{{fill:none;stroke:#2155a3;stroke-width:2.5}}.price-point{{fill:#2155a3}}.negative-point{{fill:#a92f51}}
.threshold{{stroke:#b36600;stroke-width:1;stroke-dasharray:5 4}}.probability-bar{{fill:#328080}}.probability-alert{{fill:#9f3157}}
.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e0e7ef;white-space:nowrap}}
th{{background:#f0f4f9}}.number{{text-align:right;font-variant-numeric:tabular-nums}}.below-zero{{color:#a12f4a;font-weight:bold}}
.offset{{color:#52677b;font-size:11px}}footer{{font-size:12px;color:#52677b;margin:22px 0}}
@media(max-width:700px){{body{{padding:12px}}.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:24px}}}}
@media print{{body{{padding:0;background:white}}section,.cards,.exception{{break-inside:avoid}}table{{font-size:10px}}}}
</style></head><body><main>
<header><div class="eyebrow">NYX · Modèles annuels CPU</div><h1>{escape(name)} · Livraison du {escape(day_label)}</h1>
<p class="muted">Prévisions horaires de prix et de probabilité de prix négatif. {len(frame)} heures physiques · {escape(timezone)}.</p></header>
{note}
<div class="cards"><div class="card"><small>Prix moyen prévu</small><strong>{mean_price:.2f}</strong> €/MWh</div>
<div class="card"><small>Prix minimum / maximum</small><strong>{prices.min():.2f} / {prices.max():.2f}</strong> €/MWh</div>
<div class="card"><small>Probabilité négative maximale</small><strong>{probabilities.max():.1%}</strong></div>
<div class="card"><small>Heures avec probabilité ≥ 50 %</small><strong>{int((probabilities >= .5).sum())} / {len(frame)}</strong></div></div>
<section><h2>Prix prévus · €/MWh</h2>{_price_chart(prices, local)}</section>
<section><h2>Probabilité de prix négatif</h2><p class="muted">Le trait pointillé indique le seuil de 50 %.</p>{_probability_chart(probabilities, local)}</section>
<section><h2>Détail horaire</h2><p class="muted">Les offsets UTC distinguent les heures répétées lors du changement d'heure.</p>
<div class="table-wrap"><table><thead><tr><th>Heure locale</th><th>Heure UTC</th><th class="number">Prix · €/MWh</th>
<th class="number">P(prix &lt; 0)</th><th>Alerte ≥ 50 %</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<footer>Prévision calculée sur CPU par la chaîne annuelle qualifiée. Chaque ligne correspond à une heure physique.</footer>
</main></body></html>'''


def write_country_report(frame: pd.DataFrame, destination: Path, *, country: str,
                         delivery_day: str, de_performance_exception: bool = False) -> Path:
    document = render_country_report(frame, country=country, delivery_day=delivery_day,
                                      de_performance_exception=de_performance_exception)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(document)
    return destination
