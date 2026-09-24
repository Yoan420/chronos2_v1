"""Self-contained comparison of one retrospective delivery day.

This deliberately does not use the prospective reporting ledger: late-created
forecasts must never be counted as forecasts issued before the auction.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_MODELS = {
    "lora16_residual": "Chronos-2 + LoRA rang 16 + correcteur résiduel",
    "lora16_residual_kalman": "Chronos-2 + LoRA rang 16 + correcteur résiduel + Kalman",
    "incumbent_autonomous": "Run existant — autonome",
    "incumbent_kalman": "Run existant — Kalman",
    "chronos2_exogenous": "LoRA rang 16 brut — diagnostic uniquement",
}
_REQUIRED = ("lora16_residual", "lora16_residual_kalman", "chronos2_exogenous")
_COLORS = {
    "lora16_residual": "residual",
    "lora16_residual_kalman": "kalman",
    "incumbent_autonomous": "incumbent",
    "incumbent_kalman": "incumbent-kalman",
    "chronos2_exogenous": "raw",
    "actual": "actual",
}


def _normalise(frame: pd.DataFrame, timezone: str) -> tuple[pd.DataFrame, tuple[str, ...], str]:
    required = {"delivery_start_utc", "zone", "actual"}
    required.update(f"{model}__q{q}" for model in _REQUIRED for q in (10, 50, 90))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Colonnes absentes du rapport rétrospectif : {', '.join(missing)}.")
    if frame.empty:
        raise ValueError("Le rapport rétrospectif exige une journée non vide.")
    result = frame.copy()
    timestamps = [pd.Timestamp(value) for value in result["delivery_start_utc"]]
    if any(pd.isna(value) or value.tzinfo is None for value in timestamps):
        raise ValueError("Les heures de livraison doivent comporter un fuseau horaire explicite.")
    result["delivery_start_utc"] = pd.to_datetime(timestamps, utc=True)
    if any(not isinstance(zone, str) or not zone.strip() for zone in result["zone"]):
        raise ValueError("Chaque ligne doit identifier un pays non vide.")
    if result.duplicated(["zone", "delivery_start_utc"]).any():
        raise ValueError("Heure de livraison dupliquée pour un même pays.")
    days = result["delivery_start_utc"].dt.tz_convert(timezone).dt.date.unique()
    if len(days) != 1:
        raise ValueError("Le rapport rétrospectif doit porter sur une seule journée locale.")
    day = pd.Timestamp(days[0])
    expected = pd.date_range(
        day.tz_localize(timezone), (day + pd.Timedelta(days=1)).tz_localize(timezone),
        freq="h", inclusive="left",
    ).tz_convert("UTC")
    result = result.sort_values(["zone", "delivery_start_utc"]).reset_index(drop=True)
    for zone, group in result.groupby("zone", sort=True):
        if not pd.DatetimeIndex(group["delivery_start_utc"]).equals(expected):
            raise ValueError(f"{zone} : couverture physique incomplète, {len(expected)} heures attendues.")
    models = tuple(model for model in _MODELS if any(f"{model}__q{q}" in result for q in (10, 50, 90)))
    numeric = ["actual"]
    for model in models:
        columns = [f"{model}__q{q}" for q in (10, 50, 90)]
        if any(column not in result for column in columns):
            raise ValueError(f"Les trois quantiles q10/q50/q90 sont requis pour {model}.")
        numeric.extend(columns)
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        values = result[column].to_numpy()
        if np.isinf(values).any():
            raise ValueError(f"Valeur infinie dans {column}.")
        if column != "actual" and any(column.startswith(f"{model}__") for model in _REQUIRED):
            if not np.isfinite(values).all():
                raise ValueError(f"Prévisions non finies dans {column}.")
    for model in models:
        values = result[[f"{model}__q{q}" for q in (10, 50, 90)]].to_numpy()
        complete = np.isfinite(values).all(axis=1)
        if (np.diff(values[complete], axis=1) < 0).any():
            raise ValueError(f"Quantiles inversés pour {model}.")
    return result, models, day.date().isoformat()


def _summarise(group: pd.DataFrame, models: tuple[str, ...]) -> dict[str, Any]:
    actual_complete = bool(np.isfinite(group["actual"]).all())
    actual_mean = float(group["actual"].mean()) if actual_complete else None
    result: dict[str, Any] = {
        "hours": len(group), "actual_hours": int(np.isfinite(group["actual"]).sum()),
        "actual_mean": actual_mean, "models": {},
    }
    for model in models:
        prediction = group[f"{model}__q50"]
        complete = bool(np.isfinite(prediction).all())
        mean = float(prediction.mean()) if complete else None
        evaluated = actual_complete and complete
        result["models"][model] = {
            "mean": mean,
            "bias": mean - actual_mean if evaluated else None,
            "daily_absolute_error": abs(mean - actual_mean) if evaluated else None,
            "hourly_mae": float((prediction - group["actual"]).abs().mean()) if evaluated else None,
            "evaluated_hours": len(group) if evaluated else 0,
        }
    return result


def _number(value: float | None) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:,.2f}".replace(",", "\u202f").replace(".", ",")


def _metrics_table(summary: dict[str, Any], models: tuple[str, ...]) -> str:
    rows = []
    for model in models:
        values = summary["models"][model]
        role = "final" if model in _REQUIRED[:2] else "reference"
        cells = [values["mean"], summary["actual_mean"], values["bias"],
                 values["daily_absolute_error"], values["hourly_mae"]]
        rows.append(f'<tr class="{role}"><td>{escape(_MODELS[model])}</td>'
                    + "".join(f'<td class="number">{_number(value)}</td>' for value in cells)
                    + f'<td class="number">{values["evaluated_hours"]}/{summary["hours"]}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Modèle</th><th>Prix moyen prévu</th>'
            '<th>Prix moyen observé</th><th>Prévu − observé</th><th>|Écart du prix moyen|</th>'
            '<th>MAE horaire</th><th>Heures évaluées</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>')


def _chart(group: pd.DataFrame, models: tuple[str, ...], timezone: str, chart_id: str) -> str:
    series = [(model, group[f"{model}__q50"].to_numpy()) for model in models]
    series.append(("actual", group["actual"].to_numpy()))
    finite = np.concatenate([values[np.isfinite(values)] for _, values in series])
    low, high = float(finite.min()), float(finite.max())
    pad = max((high - low) * .12, 1.0)
    low, high = low - pad, high + pad
    local = group["delivery_start_utc"].dt.tz_convert(timezone)
    x = lambda i: 70 + i * 870 / max(len(group) - 1, 1)
    y = lambda value: 290 - (value - low) * 250 / (high - low)
    parts = [f'<svg id="{chart_id}" viewBox="0 0 970 345" role="img" aria-label="Comparaison horaire des prévisions et observations">']
    for value in np.linspace(low, high, 5):
        parts.append(f'<line class="grid" x1="70" x2="940" y1="{y(value):.2f}" y2="{y(value):.2f}"/>'
                     f'<text x="61" y="{y(value) + 4:.2f}" text-anchor="end">{_number(value)}</text>')
    for i, timestamp in enumerate(local):
        if i % 3 == 0 or i == len(group) - 1:
            parts.append(f'<text x="{x(i):.2f}" y="312" text-anchor="middle">{timestamp.strftime("%H:%M")}</text>')
    parts.append(f'<text x="70" y="22">EUR/MWh</text><text x="940" y="337" text-anchor="end">{escape(timezone)} · heures physiques</text>')
    legend = []
    for serial, (model, values) in enumerate(series):
        label = "Prix observé" if model == "actual" else _MODELS[model]
        color = _COLORS[model]
        parts.append(f'<g data-series="{serial}" class="curve {color}">')
        segment: list[str] = []
        for i, value in enumerate(values):
            if np.isfinite(value):
                segment.append(f'{x(i):.2f},{y(value):.2f}')
            elif segment:
                parts.append(f'<polyline points="{" ".join(segment)}"/>')
                segment = []
        if segment:
            parts.append(f'<polyline points="{" ".join(segment)}"/>')
        for i, value in enumerate(values):
            if np.isfinite(value):
                tooltip = f'{label} · {local.iloc[i].strftime("%Y-%m-%d %H:%M %z")} · {_number(value)} EUR/MWh'
                parts.append(f'<circle cx="{x(i):.2f}" cy="{y(value):.2f}" r="3"><title>{escape(tooltip)}</title></circle>')
        parts.append('</g>')
        legend.append(f'<button class="legend {color}" data-chart="{chart_id}" data-series="{serial}" aria-pressed="true">'
                      f'<span class="swatch"></span>{escape(label)}</button>')
    parts.append('</svg>')
    return '<div class="legend-list">' + "".join(legend) + '</div>' + "".join(parts)


def _hourly_table(group: pd.DataFrame, models: tuple[str, ...], timezone: str) -> str:
    rows = []
    for _, row in group.iterrows():
        hour = row["delivery_start_utc"].tz_convert(timezone).strftime("%H:%M %z")
        values = [row["actual"], *(row[f"{model}__q50"] for model in models)]
        rows.append(f'<tr><td>{hour}</td>' + "".join(f'<td class="number">{_number(value)}</td>' for value in values) + '</tr>')
    return ('<details><summary>Détail horaire — survol des points ou tableau</summary><div class="scroll">'
            '<table><thead><tr><th>Heure locale / UTC offset</th><th>Observé</th>'
            + "".join(f'<th>{escape(_MODELS[model])}</th>' for model in models)
            + '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></details>')


_STYLE = """
:root{color-scheme:light;--bg:#f2f5fa;--card:#fff;--text:#17243a;--muted:#52637b;--line:#d8e1ed;--warning:#fff0ce;--highlight:#e9f2ff;--residual:#0a68c4;--kalman:#a0278e;--incumbent:#a86400;--incumbent-kalman:#188353;--raw:#797286;--actual:#19232d}
html[data-theme=dark]{color-scheme:dark;--bg:#101725;--card:#192337;--text:#e7edf7;--muted:#afbed3;--line:#35445c;--warning:#49391f;--highlight:#223956;--residual:#70baff;--kalman:#f596e5;--incumbent:#ffc067;--incumbent-kalman:#77d6a9;--raw:#b4aabe;--actual:#f5f8fc}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;gap:20px}h1{margin:0;font-size:27px}h2{font-size:23px;margin:0 0 10px}h3{font-size:17px}section{background:var(--card);padding:22px;border:1px solid var(--line);border-radius:10px;margin:20px 0}.warning{background:var(--warning)}.muted{color:var(--muted)}button{background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px 10px;cursor:pointer}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted)}.number{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.final{background:var(--highlight)}.final td:first-child{font-weight:650}svg{display:block;width:100%;min-width:540px}svg text{font:12px system-ui,sans-serif;fill:var(--muted)}.grid{stroke:var(--line)}.curve polyline{fill:none;stroke:currentColor;stroke-width:2}.curve circle{fill:currentColor}.curve.actual polyline{stroke-width:3}.curve.raw polyline{stroke-dasharray:5 4}.residual{color:var(--residual)}.kalman{color:var(--kalman)}.incumbent{color:var(--incumbent)}.incumbent-kalman{color:var(--incumbent-kalman)}.raw{color:var(--raw)}.actual{color:var(--actual)}.legend-list{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}.legend{font-size:12px;text-align:left}.swatch{display:inline-block;width:18px;border-top:3px solid currentColor;margin-right:7px;vertical-align:middle}.legend[aria-pressed=false]{opacity:.45;text-decoration:line-through}details{margin-top:16px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}footer{color:var(--muted);font-size:13px}@media(max-width:650px){main{padding:12px}section{padding:14px}header{display:block}h1{font-size:23px}}
"""
_SCRIPT = """
document.getElementById('theme-toggle').addEventListener('click',function(){
 const dark=document.documentElement.dataset.theme!=='dark';
 document.documentElement.dataset.theme=dark?'dark':'light';
 this.textContent=dark?'Mode jour':'Mode nuit';this.setAttribute('aria-pressed',String(dark));
});
document.querySelectorAll('button[data-chart]').forEach(button=>button.addEventListener('click',()=>{
 const visible=button.getAttribute('aria-pressed')!=='true';
 document.getElementById(button.dataset.chart).querySelector('g[data-series="'+button.dataset.series+'"]').style.display=visible?'':'none';
 button.setAttribute('aria-pressed',String(visible));
}));
"""


def render_retrospective_report(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    metadata: Mapping | None = None,
    timezone: str = "Europe/Paris",
) -> Path:
    """Write a new report, with no mutation of the prospective ledger or input.

    Every country must provide the same complete physical delivery day. Metrics
    are blank until all observations for that country-day are finite. Missing
    optional incumbent forecasts remain blank, never zero or a substituted model.
    This renderer validates inputs but does not certify upstream calibration.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Rapport existant : remplacement refusé ({output_path}).")
    normalised, models, day = _normalise(frame, timezone)
    sections = []
    for index, (zone, group) in enumerate(normalised.groupby("zone", sort=True)):
        summary = _summarise(group, models)
        status = ("Comparaison rétrospective évaluée sur la journée complète."
                  if summary["actual_mean"] is not None else
                  "Observations incomplètes : comparaison des prévisions uniquement ; prix moyen observé et erreurs en attente.")
        final_models = tuple(model for model in models if model != "chronos2_exogenous")
        sections.append(f'<section><h2>{escape(zone)} · {day}</h2><p>{status}</p>'
                        f'<p class="muted">Observations : {summary["actual_hours"]}/{summary["hours"]} heures physiques. '
                        'Tous les prix et écarts sont exprimés en EUR/MWh.</p>'
                        + _metrics_table(summary, final_models)
                        + '<details><summary>Référence de diagnostic : LoRA brut</summary>'
                        + _metrics_table(summary, ("chronos2_exogenous",)) + '</details>'
                        + '<div class="scroll">' + _chart(group, models, timezone, f"chart-{index}") + '</div>'
                        + _hourly_table(group, models, timezone) + '</section>')
    audit = escape(json.dumps(dict(metadata or {}), indent=2, ensure_ascii=False, default=str))
    document = (f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>Comparaison rétrospective LoRA rang 16 — {day}</title><style>{_STYLE}</style></head><body><main>'
                '<header><div><h1>RÉTROSPECTIF — LoRA rang 16</h1>'
                f'<p>Livraison du {day} · comparaison avec le run existant</p></div>'
                '<button id="theme-toggle" aria-pressed="false">Mode nuit</button></header>'
                '<section class="warning"><strong>Hors statistiques prospectives — recherche uniquement.</strong>'
                '<p>Ces prévisions ont été reconstruites après leur échéance de publication. '
                'Elles ne constituent pas des prévisions émises avant l’enchère, ni une preuve de performance en production.</p>'
                '<p>Les deux chaînes comparées sont Chronos-2 + LoRA rang 16 + correcteur résiduel, avec ou sans Kalman. '
                'Calibration : fenêtre glissante de 365 jours strictement antérieurs à la livraison ; '
                'les prix du jour comparé sont exclus de la calibration. Ces 365 jours ne sont pas un backtest indépendant de 365 jours.</p>'
                '<p>Le checkpoint neuronal a été sélectionné rétrospectivement. Ce rapport ne confère aucune qualification PIT, '
                'OOF neuronal ou éligibilité à une promotion. Les références opérationnelles absentes ne sont pas remplacées.</p></section>'
                + "".join(sections)
                + '<section><h2>Lecture des résultats</h2><p>Le prix moyen est la moyenne des médianes horaires q50 ; '
                'la MAE horaire est la moyenne des écarts absolus heure par heure. L’écart signé est prévu − observé. '
                'Toutes les comparaisons d’un pays portent sur les mêmes 23, 24 ou 25 heures physiques. '
                'Une observation manquante laisse les métriques journalières vides : aucune extrapolation ni moyenne partielle.</p>'
                '<p>Cliquez sur les légendes pour masquer une courbe ; survolez ses points pour lire les valeurs. '
                'Les heures répétées lors du changement d’heure sont distinguées par leur décalage UTC dans les infobulles et le tableau.</p>'
                f'<details><summary>Métadonnées de reconstruction</summary><pre>{audit}</pre></details></section>'
                '<footer>Diagnostic seulement · aucune activation en production · aucun ajout au journal prospectif.</footer>'
                f'</main><script>{_SCRIPT}</script></body></html>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(document)
    return output_path
