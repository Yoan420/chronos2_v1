"""Isolated, dependency-free HTML reporting for the rank-16 prospective trial.

This module does not confer production, point-in-time, or promotion evidence.
Its common evaluation support excludes calibration rows and incomplete local
delivery days, including the 23- and 25-hour daylight-saving transitions.
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
    "lora16_residual": "LoRA rang 16 + correcteur résiduel",
    "lora16_residual_kalman": "LoRA rang 16 + correcteur résiduel + Kalman",
    "chronos2_exogenous": "LoRA rang 16 brut — diagnostic",
}
_FINAL_MODELS = tuple(_MODELS)[:2]


def _normalise(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"delivery_start_utc", "zone", "actual", "prospective_eligible"}
    required.update(f"{model}__q{q}" for model in _MODELS for q in (10, 50, 90))
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Colonnes absentes du rapport prospectif : {', '.join(missing)}.")
    frame = predictions.loc[:, sorted(required)].copy()
    timestamps = [pd.Timestamp(value) for value in frame["delivery_start_utc"]]
    if any(pd.isna(value) or value.tzinfo is None for value in timestamps):
        raise ValueError("delivery_start_utc doit contenir des dates explicites avec fuseau horaire.")
    frame["delivery_start_utc"] = pd.to_datetime(timestamps, utc=True)
    if any(not isinstance(zone, str) or not zone.strip() for zone in frame["zone"]):
        raise ValueError("Chaque ligne doit identifier un pays non vide.")
    if frame.duplicated(["zone", "delivery_start_utc"]).any():
        raise ValueError("Plusieurs prévisions pour une même heure et un même pays.")
    eligible = frame["prospective_eligible"]
    if any(not isinstance(value, (bool, np.bool_)) and not pd.isna(value) for value in eligible):
        raise ValueError("prospective_eligible doit être un booléen, pas une chaîne de caractères.")
    frame["prospective_eligible"] = eligible.fillna(False).astype(bool)
    for column in ["actual", *(f"{model}__q{q}" for model in _MODELS for q in (10, 50, 90))]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if np.isinf(frame[column].to_numpy()).any():
            raise ValueError(f"Valeur infinie dans {column}.")
    for model in _MODELS:
        values = frame[[f"{model}__q{q}" for q in (10, 50, 90)]].to_numpy()
        complete = np.isfinite(values).all(axis=1)
        if (np.diff(values[complete], axis=1) < 0).any():
            raise ValueError(f"Quantiles inversés pour {model}.")
    return frame.sort_values(["zone", "delivery_start_utc"]).reset_index(drop=True)


def _summarise_days(frame: pd.DataFrame, timezone: str) -> tuple[list[dict[str, Any]], pd.Timestamp | None]:
    if frame.empty:
        return [], None
    frame = frame.assign(delivery_day=frame["delivery_start_utc"].dt.tz_convert(timezone).dt.date)
    last_day = pd.Timestamp(max(frame["delivery_day"]))
    first_day = (last_day - pd.Timedelta(days=364)).date()
    days: list[dict[str, Any]] = []
    for (zone, day), group in frame.groupby(["zone", "delivery_day"], sort=True):
        midnight = pd.Timestamp(day).tz_localize(timezone)
        next_midnight = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(timezone)
        expected = pd.date_range(midnight, next_midnight, freq="h", inclusive="left").tz_convert("UTC")
        observed_index = pd.DatetimeIndex(group["delivery_start_utc"])
        complete_hours = observed_index.equals(expected)
        actual_complete = complete_hours and bool(np.isfinite(group["actual"]).all())
        prospective = bool(group["prospective_eligible"].all())
        common_predictions = complete_hours and all(
            bool(np.isfinite(group[f"{model}__q50"]).all()) for model in _MODELS
        )
        in_window = day >= first_day
        evaluated = actual_complete and prospective and common_predictions and in_window
        if not prospective:
            status, status_code = "Rétrospectif — hors évaluation", "retrospective"
        elif not complete_hours:
            status, status_code = "Journée horaire incomplète — exclue", "incomplete"
        elif not actual_complete:
            status, status_code = "Prix observé en attente", "pending"
        elif not common_predictions:
            status, status_code = "Prévisions incomplètes — exclues", "incomplete"
        elif not in_window:
            status, status_code = "Hors fenêtre de 365 jours", "outside"
        else:
            status, status_code = "Évalué — prospectif", "evaluated"
        row: dict[str, Any] = {
            "zone": zone, "day": day.isoformat(), "expected_hours": len(expected),
            "provided_hours": len(group), "actual_mean": float(group["actual"].mean()) if actual_complete else None,
            "evaluated": evaluated, "status": status, "status_code": status_code,
            "models": {},
        }
        for model in _MODELS:
            prediction = group[f"{model}__q50"]
            model_complete = complete_hours and bool(np.isfinite(prediction).all())
            mean = float(prediction.mean()) if model_complete else None
            row["models"][model] = {
                "mean": mean,
                "hourly_absolute_error_sum": float((prediction - group["actual"]).abs().sum()) if evaluated else None,
                "daily_mean_absolute_error": abs(mean - row["actual_mean"]) if evaluated else None,
                "bias": mean - row["actual_mean"] if evaluated else None,
            }
        days.append(row)
    return days, last_day


def _number(value: float | None) -> str:
    # An unpublished observation is deliberately an empty cell, never zero.
    return "" if value is None else f"{value:,.2f}".replace(",", "\u202f").replace(".", ",")


def _metrics_rows(days: list[dict[str, Any]], models: tuple[str, ...]) -> str:
    rows: list[str] = []
    for zone in sorted({row["zone"] for row in days}):
        selected = [row for row in days if row["zone"] == zone and row["evaluated"]]
        hours = sum(row["expected_hours"] for row in selected)
        window = f"{selected[0]['day']} → {selected[-1]['day']}" if selected else "Aucun jour évalué"
        for model in models:
            hourly = sum(row["models"][model]["hourly_absolute_error_sum"] for row in selected) / hours if hours else None
            daily = float(np.mean([row["models"][model]["daily_mean_absolute_error"] for row in selected])) if selected else None
            bias = float(np.mean([row["models"][model]["bias"] for row in selected])) if selected else None
            rows.append(
                f'<tr><td>{escape(zone)}</td><td>{escape(_MODELS[model])}</td>'
                f'<td class="number">{_number(hourly)}</td><td class="number">{_number(daily)}</td>'
                f'<td class="number">{_number(bias)}</td><td class="number">{len(selected)}</td>'
                f'<td class="number">{hours}</td><td>{escape(window)}</td></tr>'
            )
    return "".join(rows) or '<tr><td colspan="8">Aucune prévision enregistrée.</td></tr>'


def _metrics_table(days: list[dict[str, Any]], models: tuple[str, ...]) -> str:
    return (
        '<div class="table-scroll"><table><thead><tr><th>Pays</th><th>Modèle</th>'
        '<th>MAE horaire</th><th>MAE du prix moyen journalier</th><th>Biais moyen journalier</th>'
        '<th>Jours</th><th>Heures</th><th>Période évaluée</th></tr></thead><tbody>'
        + _metrics_rows(days, models) + '</tbody></table></div>'
    )


def _daily_table(days: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for row in sorted(days, key=lambda value: (value["day"], value["zone"]), reverse=True):
        models = row["models"]
        errors = [models[model]["daily_mean_absolute_error"] for model in _FINAL_MODELS]
        if row["evaluated"]:
            if np.isclose(errors[0], errors[1], rtol=0, atol=1e-9):
                winner = "Écart identique"
            else:
                winner = "Résiduel" if errors[0] < errors[1] else "Résiduel + Kalman"
        else:
            winner = ""
        numeric = [_number(row["actual_mean"])]
        numeric += [_number(models[model]["mean"]) for model in _MODELS]
        numeric += [_number(error) for error in errors]
        cells = ''.join(f'<td class="number">{value}</td>' for value in numeric)
        rows.append(
            f'<tr data-zone="{escape(row["zone"], quote=True)}" data-status="{row["status_code"]}">'
            f'<td>{escape(row["day"])}</td><td>{escape(row["zone"])}</td>'
            f'<td>{row["provided_hours"]}/{row["expected_hours"]}</td>{cells}'
            f'<td>{winner}</td><td><span class="status {row["status_code"]}">{escape(row["status"])}</span></td></tr>'
        )
    return (
        '<div class="table-scroll"><table id="daily-table"><thead><tr><th>Livraison</th><th>Pays</th>'
        '<th>Heures</th><th>Observé</th><th>Résiduel</th><th>Résiduel + Kalman</th>'
        '<th>LoRA brut (diagnostic)</th><th>|Écart| résiduel</th><th>|Écart| Kalman</th>'
        '<th>Prix moyen le plus exact</th><th>Statut</th></tr></thead><tbody>'
        + (''.join(rows) or '<tr><td colspan="11">Aucune prévision enregistrée.</td></tr>')
        + '</tbody></table></div>'
    )


_STYLE = """
:root{color-scheme:light;--bg:#f4f6fa;--card:#fff;--text:#17243a;--muted:#52637b;--line:#dce3ee;--accent:#175fa9;--warn:#fff5dc;--warnline:#d39116;--ok:#e1f3e8;--pending:#fff0cb;--other:#edf0f5}
html[data-theme=dark]{color-scheme:dark;--bg:#101725;--card:#182338;--text:#e7edf7;--muted:#aebbd0;--line:#34445e;--accent:#89befa;--warn:#3b3020;--warnline:#b78337;--ok:#204537;--pending:#44371e;--other:#2e3a4e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}main{max-width:1700px;margin:auto;padding:28px}header{display:flex;gap:24px;justify-content:space-between;align-items:start}h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:0 0 10px}p{margin:8px 0}.muted{color:var(--muted)}section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;margin:20px 0}.warning{background:var(--warn);border-left:5px solid var(--warnline)}button,select{border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--text);padding:8px 12px}button{cursor:pointer;white-space:nowrap}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600;white-space:nowrap}td.number{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.status{display:inline-block;border-radius:5px;padding:3px 7px;background:var(--other);white-space:nowrap}.status.evaluated{background:var(--ok)}.status.pending{background:var(--pending)}.filters{display:flex;gap:16px;flex-wrap:wrap;margin:14px 0}.filters label{display:flex;align-items:center;gap:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer;color:var(--accent)}footer{color:var(--muted);font-size:13px}@media(max-width:650px){main{padding:14px}header{display:block}header button{margin-top:12px}section{padding:14px}h1{font-size:23px}}
"""

_SCRIPT = """
(() => {
  const toggle = document.getElementById('theme-toggle');
  toggle.addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme !== 'dark';
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    toggle.textContent = dark ? 'Mode jour' : 'Mode nuit';
    toggle.setAttribute('aria-pressed', String(dark));
  });
  const zone = document.getElementById('zone-filter');
  const status = document.getElementById('status-filter');
  const filter = () => {
    document.querySelectorAll('#daily-table tbody tr[data-zone]').forEach(row => {
      row.hidden = Boolean((zone.value && row.dataset.zone !== zone.value) ||
        (status.value && row.dataset.status !== status.value));
    });
  };
  zone.addEventListener('change', filter);
  status.addEventListener('change', filter);
})();
"""


def render_trial_report(
    predictions: pd.DataFrame,
    output_path: Path,
    *,
    timezone: str = "Europe/Paris",
    metadata: Mapping | None = None,
) -> Path:
    """Create an immutable research report; never overwrite an existing file.

    Metrics use q50 point forecasts for all three models on exactly the same
    complete, observed, prospective country-days. Their window is at most 365
    calendar days ending at the latest delivery date supplied to this call.
    Quantiles are validated but do not contribute to point-forecast metrics.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Rapport déjà présent, remplacement refusé : {output_path}")
    frame = _normalise(predictions)
    # Validate timezone even for an empty ledger.
    pd.Timestamp("2000-01-01", tz=timezone)
    days, last_day = _summarise_days(frame, timezone)
    evaluated = [row for row in days if row["evaluated"]]
    calendar_days = len({row["day"] for row in evaluated})
    pending = sum(row["status_code"] == "pending" for row in days)
    window = (
        f"{(last_day - pd.Timedelta(days=364)).date().isoformat()} → {last_day.date().isoformat()}"
        if last_day is not None else "Aucune livraison enregistrée"
    )
    zone_options = ''.join(
        f'<option value="{escape(zone, quote=True)}">{escape(zone)}</option>'
        for zone in sorted(frame["zone"].unique())
    )
    safe_metadata = escape(json.dumps(dict(metadata or {}), ensure_ascii=False, indent=2, default=str))
    document = f'''<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Essai prospectif LoRA rang 16 — résiduel et Kalman</title><style>{_STYLE}</style></head>
<body><main><header><div><h1>LoRA rang 16 · essai prospectif</h1>
<p class="muted">Chronos-2 + LoRA rang 16 + correcteur résiduel, avec ou sans Kalman.</p></div>
<button id="theme-toggle" type="button" aria-pressed="false">Mode nuit</button></header>
<section class="warning"><h2>Recherche uniquement — aucune promotion en production</h2>
<p>Le checkpoint a été choisi sur un historique déjà examiné. Cet historique sert à la calibration,
pas à un test indépendant. Seules les nouvelles prévisions marquées prospectives et les journées
entièrement observées entrent dans les métriques ci-dessous.</p>
<p>Ce rapport ne constitue ni une preuve PIT de production, ni une autorisation d’activation.</p></section>
<section><h2>Support de comparaison commun</h2>
<p><strong>{calendar_days} jours calendaires évalués</strong> · {len(evaluated)} journées-pays évaluées ·
{pending} journées-pays en attente de prix observés.</p>
<p class="muted">Fenêtre maximale : {escape(window)} ({escape(timezone)}). Au démarrage, moins de 365 jours
sont disponibles : aucun résultat historique de calibration ne complète artificiellement cette période.</p>
<p class="muted">Une journée comporte exactement 23, 24 ou 25 heures physiques selon le changement d’heure.
Les trois références partagent strictement les mêmes journées évaluées.</p></section>
<section><h2>Les deux chaînes demandées</h2><p class="muted">Erreurs et biais en EUR/MWh. Plus la MAE est faible,
plus la prévision est exacte. Un biais positif signifie une surestimation du prix moyen observé.</p>
{_metrics_table(days, _FINAL_MODELS)}</section>
<section><h2>Référence de diagnostic — LoRA brut</h2>
<p class="muted">Ce troisième résultat sert uniquement à mesurer l’apport des corrections ; ce n’est pas une troisième chaîne finale.</p>
{_metrics_table(days, ("chronos2_exogenous",))}</section>
<section><h2>Prix moyens par journée</h2><p class="muted">Prix en EUR/MWh, calculés sur toutes les heures physiques
de la journée. Une observation absente ou partielle reste vide. Les écarts ne sont affichés que pour les journées évaluées.</p>
<div class="filters"><label>Pays <select id="zone-filter"><option value="">Tous</option>{zone_options}</select></label>
<label>Statut <select id="status-filter"><option value="">Tous</option><option value="evaluated">Évalué</option>
<option value="pending">Prix en attente</option><option value="retrospective">Rétrospectif exclu</option>
<option value="incomplete">Incomplet exclu</option><option value="outside">Hors fenêtre</option></select></label></div>
{_daily_table(days)}</section>
<section><h2>Méthode de calcul</h2><p>La MAE horaire est la moyenne de |prévision q50 − observation| sur toutes
les heures évaluées. La MAE du prix moyen journalier est la moyenne, à poids égal par jour, de
|moyenne des q50 du jour − moyenne des observations du jour|. Le biais journalier conserve le signe de cet écart.</p>
<p>Une ligne rétrospective, une heure manquante, un prix manquant ou une prévision q50 manquante pour l’un des
trois modèles exclut la journée de toutes les métriques comparatives. Les lignes hors de la fenêtre de 365 jours sont également exclues.</p>
<details><summary>Métadonnées de l’essai</summary><pre>{safe_metadata}</pre></details></section>
<footer>Rapport autonome, sans ressource réseau. Les observations doivent être rattachées aux prévisions
déjà enregistrées ; la résolution d’un prix ne doit pas provoquer de nouvelle inférence historique.</footer>
</main><script>{_SCRIPT}</script></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
    return output_path
