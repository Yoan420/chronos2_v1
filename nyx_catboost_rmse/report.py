"""Self-contained, diagnostic-only reporting for the CatBoost loss ablation.

The runner owns provenance verification and the experiment namespace. This module
never loads observations, recomputes Kalman, chooses a variant, or launches training.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "actual", "chronos_q50", "mae_q50", "rmse_q50", "rmse_q10", "rmse_q90",
    "raw_correction", "applied_correction",
)
MODEL_LABELS = {
    "mae_q50": "Référence CatBoost MAE archivée (sans Kalman)",
    "rmse_q50": "Centre CatBoost RMSE",
    "chronos_q50": "Même Chronos Q50 figé",
    "storm_q50": "Storm (heures communes uniquement)",
}
DEFAULT_TIMEZONES = {
    "FR": "Europe/Paris", "DE": "Europe/Berlin", "BE": "Europe/Brussels", "NL": "Europe/Amsterdam",
}
WARNINGS = [
    "Ablation à visée diagnostique uniquement : aucune variante optimale n'est sélectionnée et aucun modèle n'est promu.",
    "La RMSE cible une moyenne conditionnelle ; le centre affiché n'est pas une médiane ni un P50 statistiquement calibré.",
    "Les bornes Q10/Q90 héritées sont décalées de la correction, sans recalibrage des intervalles de prévision.",
    "Le comparateur MAE est le CatBoost résiduel de référence archivé, et non une prévision ajustée par Kalman. Aucun Kalman n'est recalculé.",
    "Cette année historique de diagnostic a déjà été examinée ; elle ne constitue pas une confirmation hors échantillon sur une période jamais consultée.",
    "Le protocole demandé conserve Chronos, les entrées et le plafond symétrique de correction à 40 EUR/MWh, avec réentraînement quotidien sur une fenêtre glissante de 365 jours. La vérification de la provenance et de la couverture d'entraînement relève du programme d'exécution.",
    "Les tranches de pics, de prix négatifs et du 14 septembre à 19 h sont des diagnostics a posteriori, pas des critères de réglage ou de sélection de variante.",
    "Toutes les comparaisons principales utilisent les mêmes heures avec observation, MAE et RMSE finies. Storm utilise un sous-ensemble commun séparé, sans aucune imputation.",
    "La RMSE globale est calculée sur l'ensemble des erreurs au carré, et non comme une moyenne des RMSE zonales. La MAE quotidienne moyenne donne le même poids à chaque couple zone-jour observé, y compris les journées partielles.",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _expected_hours(start: date, end: date, timezone: str) -> pd.DatetimeIndex:
    begin = pd.Timestamp(start).tz_localize(timezone)
    stop = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(timezone)
    return pd.date_range(begin, stop, freq="h", inclusive="left").tz_convert("UTC")


def _prepare(
    frames: dict[str, pd.DataFrame], metadata: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = date.fromisoformat(str(metadata.get("evaluation_start", "2025-09-20")))
    end = date.fromisoformat(str(metadata.get("evaluation_end", "2026-09-19")))
    if end < start:
        raise ValueError("evaluation_end must not precede evaluation_start")
    zones = list(metadata.get("planned_zones", ["FR", "DE", "BE", "NL"]))
    if not zones or len(zones) != len(set(zones)):
        raise ValueError("planned_zones must be nonempty and unique")
    if set(frames) - set(zones):
        raise ValueError("frames contain a zone outside planned_zones")
    timezones = {**DEFAULT_TIMEZONES, **metadata.get("timezones", {})}
    parts = []
    coverage = []
    for zone in zones:
        if zone not in timezones:
            raise ValueError(f"No timezone configured for {zone}")
        expected = _expected_hours(start, end, timezones[zone])
        frame = frames.get(zone)
        if frame is None or frame.empty:
            coverage.append({"zone": zone, "timezone": timezones[zone], "expected_hours": len(expected),
                             "input_hours": 0, "period_hours": 0, "common_hours": 0,
                             "excluded_nonfinite_prediction_hours": 0, "outside_period_hours": 0,
                             "missing_expected_hours": len(expected), "unexpected_hours": 0,
                             "first_utc": None, "last_utc": None, "complete": False})
            continue
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(f"{zone}: index must be a timezone-aware UTC DatetimeIndex")
        if str(frame.index.tz).upper() != "UTC":
            raise ValueError(f"{zone}: index must use UTC")
        if frame.index.has_duplicates or frame.index.hasnans:
            raise ValueError(f"{zone}: UTC index must be unique and contain no NaT")
        if frame.columns.has_duplicates:
            raise ValueError(f"{zone}: duplicate columns")
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{zone}: missing report columns: {sorted(missing)}")
        selected = frame.loc[:, list(REQUIRED_COLUMNS) + (["storm_q50"] if "storm_q50" in frame else [])].copy()
        for column in selected:
            selected[column] = pd.to_numeric(selected[column], errors="raise").astype(float)
        if not np.isfinite(selected["actual"]).all():
            raise ValueError(f"{zone}: actual must contain only finite canonical observations")
        selected = selected.sort_index()
        if "storm_q50" not in selected:
            selected["storm_q50"] = np.nan
        local = selected.index.tz_convert(timezones[zone])
        selected["zone"] = zone
        selected["local_date"] = [value.isoformat() for value in local.date]
        selected["month"] = local.strftime("%Y-%m")
        selected["hour"] = local.hour
        in_period = (local.date >= start) & (local.date <= end)
        period = selected.loc[in_period].copy()
        finite = np.isfinite(period[["actual", "mae_q50", "rmse_q50"]]).all(axis=1)
        common = period.loc[finite].copy()
        missing_hours = expected.difference(common.index)
        unexpected = common.index.difference(expected)
        coverage.append({"zone": zone, "timezone": timezones[zone], "expected_hours": len(expected),
                         "input_hours": len(frame), "period_hours": len(period), "common_hours": len(common),
                         "excluded_nonfinite_prediction_hours": int((~finite).sum()),
                         "outside_period_hours": int((~in_period).sum()),
                         "missing_expected_hours": len(missing_hours), "unexpected_hours": len(unexpected),
                         "first_utc": common.index.min().isoformat() if len(common) else None,
                         "last_utc": common.index.max().isoformat() if len(common) else None,
                         "complete": len(missing_hours) == 0 and len(unexpected) == 0})
        parts.append(common)
    columns = list(REQUIRED_COLUMNS) + ["storm_q50", "zone", "local_date", "month", "hour"]
    panel = pd.concat(parts) if parts else pd.DataFrame(columns=columns)
    expected_count = sum(item["expected_hours"] for item in coverage)
    scope = {"evaluation_start": start.isoformat(), "evaluation_end": end.isoformat(),
             "planned_zones": zones, "reported_zones": [zone for zone in zones if zone in frames],
             "expected_hours": expected_count, "common_hours": len(panel),
             "coverage_fraction": len(panel) / expected_count,
             "annual_complete": all(item["complete"] for item in coverage), "by_zone": coverage}
    return panel, scope


def _score(panel: pd.DataFrame, column: str) -> dict[str, Any]:
    count = len(panel)
    available = count > 0 and bool(np.isfinite(panel[column]).all())
    if not available:
        return {"n_hours": count, "available": False, "mae": None, "rmse": None,
                "mean_daily_mae": None, "mae_gain": None, "rmse_gain": None,
                "mae_gain_pct": None, "rmse_gain_pct": None,
                "hours_won": 0, "hours_lost": 0, "hours_tied": 0, "hour_win_rate": None,
                "unavailable_reason": "Aucune heure commune" if not count else "Prévision non finie sur la population commune principale"}
    errors = panel[column].to_numpy(dtype=float) - panel["actual"].to_numpy(dtype=float)
    baseline = panel["mae_q50"].to_numpy(dtype=float) - panel["actual"].to_numpy(dtype=float)
    ae, baseline_ae = np.abs(errors), np.abs(baseline)
    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    baseline_mae = float(np.mean(baseline_ae))
    baseline_rmse = float(np.sqrt(np.mean(baseline ** 2)))
    daily = pd.DataFrame({"zone": panel["zone"].to_numpy(), "date": panel["local_date"].to_numpy(), "ae": ae})
    difference = baseline_ae - ae
    tied = np.isclose(difference, 0.0, rtol=0.0, atol=1e-9)
    won = int(((difference > 0) & ~tied).sum())
    lost = int(((difference < 0) & ~tied).sum())
    return {"n_hours": count, "available": True, "mae": mae, "rmse": rmse,
            "mean_daily_mae": float(daily.groupby(["zone", "date"])["ae"].mean().mean()),
            "mae_gain": baseline_mae - mae, "rmse_gain": baseline_rmse - rmse,
            "mae_gain_pct": 100 * (baseline_mae - mae) / baseline_mae if baseline_mae else None,
            "rmse_gain_pct": 100 * (baseline_rmse - rmse) / baseline_rmse if baseline_rmse else None,
            "hours_won": won, "hours_lost": lost, "hours_tied": int(tied.sum()),
            "hour_win_rate": won / count, "unavailable_reason": None}


def _summary(panel: pd.DataFrame, *, storm: bool = False) -> dict[str, Any]:
    models = ["mae_q50", "rmse_q50", "chronos_q50"] + (["storm_q50"] if storm else [])
    raw = panel["raw_correction"].to_numpy(dtype=float)
    applied = panel["applied_correction"].to_numpy(dtype=float)
    raw_finite = np.isfinite(raw)
    applied_finite = np.isfinite(applied)
    clipped = int((np.abs(raw[raw_finite]) > 40).sum())
    at_cap = int(np.isclose(np.abs(applied[applied_finite]), 40.0, atol=1e-9, rtol=0.0).sum())
    return {"n_hours": len(panel), "n_zone_days": int(panel.groupby(["zone", "local_date"]).ngroups),
            "models": {model: _score(panel, model) for model in models},
            "cap_rates": {"cap_eur_mwh": 40, "raw_finite_hours": int(raw_finite.sum()),
                          "applied_finite_hours": int(applied_finite.sum()), "raw_abs_gt_40_hours": clipped,
                          "raw_abs_gt_40_rate": clipped / int(raw_finite.sum()) if raw_finite.any() else None,
                          "applied_abs_eq_40_hours": at_cap,
                          "applied_abs_eq_40_rate": at_cap / int(applied_finite.sum()) if applied_finite.any() else None}}


def _grouped(panel: pd.DataFrame, key: str, zones: list[str]) -> list[dict[str, Any]]:
    rows = []
    for value in sorted(panel[key].unique()):
        part = panel.loc[panel[key] == value]
        rows.append({key: _json_safe(value), "zone": "ALL", **_summary(part)})
        for zone in zones:
            local = part.loc[part["zone"] == zone]
            if len(local):
                rows.append({key: _json_safe(value), "zone": zone, **_summary(local)})
    return rows


def _metrics(frames: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> dict[str, Any]:
    panel, scope = _prepare(frames, metadata)
    zones = scope["planned_zones"]
    storm_panel = panel.loc[np.isfinite(panel["storm_q50"].to_numpy(dtype=float))]
    september_day = f"{scope['evaluation_end'][:4]}-09-14"
    conditions = {
        "actual_ge_200": ("Prix observé ≥ 200 EUR/MWh", panel["actual"] >= 200),
        "actual_ge_300": ("Prix observé ≥ 300 EUR/MWh", panel["actual"] >= 300),
        "actual_le_minus_100": ("Prix observé ≤ −100 EUR/MWh", panel["actual"] <= -100),
        "september_14_19_local": (f"{september_day} à 19 h, heure locale", (panel["local_date"] == september_day) & (panel["hour"] == 19)),
    }
    slices = {}
    for key, (label, mask) in conditions.items():
        subset = panel.loc[mask]
        slices[key] = {"label": label, "ex_post_diagnostic_only": True, "pooled": _summary(subset),
                       "by_zone": {zone: _summary(subset.loc[subset["zone"] == zone]) for zone in zones}}
    complete = scope["annual_complete"]
    return _json_safe({
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE" if complete else "PARTIAL", "pending": len(panel) == 0,
        "annual_complete": complete, "diagnostic_only": True, "production_modified": False,
        "variant_selected": False, "mean_is_not_calibrated_p50": True,
        "intervals_recalibrated": False, "kalman_recomputed": False,
        "scope": scope, "model_labels": MODEL_LABELS, "metadata": metadata,
        "annual": {"label": "Période demandée entièrement couverte" if complete else "Partie observée uniquement — PAS un résultat annuel complet",
                   "pooled": _summary(panel),
                   "by_zone": {zone: _summary(panel.loc[panel["zone"] == zone]) for zone in zones}},
        "monthly": _grouped(panel, "month", zones), "hourly": _grouped(panel, "hour", zones),
        "daily": _grouped(panel, "local_date", zones), "slices": slices,
        "storm_comparison": {"matched_only": True, "filled_hours": 0, "primary_hours": len(panel),
                             "matched_hours": len(storm_panel), "pooled": _summary(storm_panel, storm=True),
                             "by_zone": {zone: _summary(storm_panel.loc[storm_panel["zone"] == zone], storm=True) for zone in zones}},
        "warnings": WARNINGS,
    })


def _number(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}".replace(",", "\u202f").replace(".", ",")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th scope=col>{escape(str(item))}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(item))}</td>" for item in row) + "</tr>" for row in rows)
    if not rows:
        body = f'<tr><td colspan="{len(headers)}">Aucune observation commune pour le moment.</td></tr>'
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _comparison_rows(groups: list[tuple[str, dict[str, Any]]]) -> list[list[Any]]:
    rows = []
    for name, group in groups:
        for model, scores in group["models"].items():
            rows.append([name, MODEL_LABELS[model], scores["n_hours"], _number(scores["mae"]), _number(scores["rmse"]),
                         _number(scores["mae_gain"]), _number(scores["rmse_gain"]),
                         _number(scores["mean_daily_mae"]),
                         _number(100 * scores["hour_win_rate"], 1) + "%" if scores["hour_win_rate"] is not None else "—"])
    return rows


def _compact_rows(groups: list[dict[str, Any]], key: str) -> list[list[Any]]:
    rows = []
    for group in groups:
        mae, rmse = group["models"]["mae_q50"], group["models"]["rmse_q50"]
        rows.append([group[key], "Toutes" if group["zone"] == "ALL" else group["zone"], group["n_hours"], _number(mae["mae"]), _number(rmse["mae"]),
                     _number(mae["rmse"]), _number(rmse["rmse"]), _number(rmse["rmse_gain"]),
                     _number(100 * rmse["hour_win_rate"], 1) + "%" if rmse["hour_win_rate"] is not None else "—"])
    return rows


def _html(payload: dict[str, Any]) -> str:
    scope = payload["scope"]
    complete = payload["annual_complete"]
    status = "PÉRIODE DEMANDÉE COMPLÈTE" if complete else "PARTIEL · PAS UN RÉSULTAT ANNUEL COMPLET"
    if payload["pending"]:
        status += " · EN ATTENTE D'OBSERVATIONS"
    annual = payload["annual"]
    groups = [("Toutes · agrégées", annual["pooled"]), *annual["by_zone"].items()]
    comparison_headers = ["Périmètre", "Modèle", "Heures communes", "MAE", "RMSE", "Gain MAE", "Gain RMSE", "MAE quotidienne moyenne", "Heures gagnées %"]
    compact_headers = ["Période", "Zone", "Heures communes", "MAE · réf. MAE", "MAE · modèle RMSE", "RMSE · réf. MAE", "RMSE · modèle RMSE", "Gain RMSE", "Heures gagnées %"]
    coverage_rows = [[item["zone"], item["timezone"], item["common_hours"], item["expected_hours"],
                      item["missing_expected_hours"], item["excluded_nonfinite_prediction_hours"],
                      "Complète" if item["complete"] else "Partielle"] for item in scope["by_zone"]]
    cap_rows = []
    for name, group in groups:
        cap = group["cap_rates"]
        cap_rows.append([name, cap["raw_finite_hours"], cap["raw_abs_gt_40_hours"],
                         _number(100 * cap["raw_abs_gt_40_rate"], 1) + "%" if cap["raw_abs_gt_40_rate"] is not None else "—",
                         cap["applied_finite_hours"], cap["applied_abs_eq_40_hours"],
                         _number(100 * cap["applied_abs_eq_40_rate"], 1) + "%" if cap["applied_abs_eq_40_rate"] is not None else "—"])
    slices = []
    for item in payload["slices"].values():
        slices.append(f'<h3>{escape(item["label"])}</h3>' + _table(comparison_headers, _comparison_rows([("Toutes · agrégées", item["pooled"]), *item["by_zone"].items()])))
    storm = payload["storm_comparison"]
    warning_items = "".join(f"<li>{escape(item)}</li>" for item in payload["warnings"])
    pooled = annual["pooled"]["models"]["rmse_q50"]
    body = f"""
<header><p class="eyebrow">NYX · ablation isolée de la fonction de perte CatBoost</p><h1>MAE → RMSE</h1>
<p class="subtitle">Chronos figé, correction résiduelle, plafond fixe à ±40 EUR/MWh.</p>
<div class="status {'complete' if complete else 'partial'}">{status}</div>
<p>Dates locales de livraison demandées : <strong>{escape(scope['evaluation_start'])} → {escape(scope['evaluation_end'])}</strong><br>
Zones : {escape(', '.join(scope['planned_zones']))} · Généré le {escape(payload['generated_at_utc'])}</p></header>
<nav><a href="#scope">Couverture</a><a href="#comparison">Comparaison</a><a href="#time">Profils temporels</a><a href="#slices">Tranches a posteriori</a><a href="#storm">Storm</a><a href="#limits">Limites</a></nav>
<section class="cards"><article><span>Heures-zones communes</span><strong>{_number(scope['common_hours'], 0)}</strong><small>sur {_number(scope['expected_hours'], 0)} attendues · {_number(100 * scope['coverage_fraction'], 1)} %</small></article>
<article><span>Centre RMSE · MAE</span><strong>{_number(pooled['mae'])}</strong><small>EUR/MWh · population commune observée</small></article>
<article><span>Centre RMSE · RMSE</span><strong>{_number(pooled['rmse'])}</strong><small>EUR/MWh · erreurs au carré agrégées</small></article>
<article><span>Gain RMSE face à la référence MAE</span><strong>{_number(pooled['rmse_gain'])}</strong><small>Positif = erreur réduite ; aucune promotion</small></article></section>
<section id="scope"><h2>01 · La couverture avant les conclusions</h2><p class="callout">{escape(annual['label'])}. Des zones, dates ou prévisions finies manquantes rendent le résultat PARTIEL. Aucun résultat annuel n'est extrapolé à partir des heures disponibles.</p>
{_table(['Zone', 'Fuseau local', 'Heures communes', 'Heures attendues', 'Heures manquantes', 'Prévisions non finies (h)', 'Couverture'], coverage_rows)}
<p>Les décomptes portent sur les heures physiques UTC ; les journées locales peuvent compter 23, 24 ou 25 heures. L'heure répétée en automne est comptée deux fois. Les lignes hors période sont exclues et documentées dans le JSON.</p></section>
<section id="comparison"><h2>02 · Comparaison à heures communes</h2><p>Les erreurs et les gains absolus sont exprimés en EUR/MWh. Gain = erreur de la référence MAE archivée − erreur du modèle : un gain positif correspond donc à une amélioration. Les heures gagnées comparent les erreurs absolues face à cette référence ; les égalités restent au dénominateur. Des valeurs Chronos manquantes rendent son indicateur indisponible, sans modifier la population commune.</p>
{_table(comparison_headers, _comparison_rows(groups))}
<h3>Diagnostic du plafond de correction</h3><p>Les taux utilisent les corrections finies de la même population principale. « Au plafond » inclut une correction exactement égale à ±40, même si la valeur brute ne dépassait pas le plafond.</p>
{_table(['Périmètre', 'Correction brute finie (h)', '|brute| > 40 (h)', 'Dépassement brut %', 'Correction appliquée finie (h)', '|appliquée| = 40 (h)', 'Au plafond %'], cap_rows)}</section>
<section id="time"><h2>03 · Profils chronologiques et par heure locale</h2><p>Ces ventilations décrivent la population commune observée ; elles ne servent pas de critères de réglage. « Toutes » agrège les heures physiques de toutes les zones.</p>
<h3>Mois dans l'ordre chronologique</h3>{_table(compact_headers, _compact_rows(payload['monthly'], 'month'))}
<details><summary>Heure locale de livraison · 00–23</summary>{_table(['Heure locale', *compact_headers[1:]], _compact_rows(payload['hourly'], 'hour'))}</details>
<details><summary>Comparaisons quotidiennes à heures communes</summary><p>La MAE quotidienne moyenne ci-dessus donne le même poids aux MAE disponibles de chaque couple zone-jour ; ce tableau permet de repérer les journées partielles et les changements d'heure.</p>{_table(['Date locale', *compact_headers[1:]], _compact_rows(payload['daily'], 'local_date'))}</details></section>
<section id="slices"><h2>04 · Tranches de tension a posteriori</h2><p class="callout">Les tranches de prix déjà observées et l'événement du 14 septembre à 19 h sont uniquement diagnostiques. Elles ne servent ni à sélectionner, ni à régler, ni à promouvoir une variante. Les tranches vides n'ont pas de score.</p>{''.join(slices)}</section>
<section id="storm"><h2>05 · Storm, sous-ensemble commun séparé</h2><p>{_number(storm['matched_hours'], 0)} heures communes avec Storm sur {_number(storm['primary_hours'], 0)} heures de la population principale. Aucune propagation de valeur, substitution ou prévision inventée. Comparez les lignes uniquement au sein de ce tableau, pas à la population plus large présentée ci-dessus.</p>
<p class="callout">Dans ce tableau aussi, les gains et le taux d'heures gagnées sont calculés face au CatBoost MAE archivé, pas face à Storm.</p>
{_table(comparison_headers, _comparison_rows([('Toutes · communes avec Storm', storm['pooled']), *storm['by_zone'].items()]))}</section>
<section id="limits"><h2>06 · Interprétation et garde-fous</h2><ul>{warning_items}</ul>
<details><summary>Métadonnées et provenance fournies par le programme d'exécution</summary><pre>{escape(json.dumps(payload['metadata'], ensure_ascii=False, indent=2, allow_nan=False))}</pre></details></section>
<footer>Rapport diagnostique autonome · aucun script externe, appel réseau ou acte de sélection de modèle.<br>Fichier de données associé : catboost_rmse_metrics.json</footer>
"""
    css = """
:root{color-scheme:light dark;--bg:#f1f5f7;--panel:#fff;--ink:#102b3b;--muted:#516678;--line:#d6e0e7;--accent:#007e87;--warn:#fff2d4;--warnink:#744800}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1500px;margin:auto;padding:34px 26px}header{padding:18px 0 24px}.eyebrow{text-transform:uppercase;letter-spacing:.17em;color:var(--accent);font-size:12px;font-weight:750}h1{font-size:clamp(44px,7vw,78px);line-height:1.05;letter-spacing:-.06em;margin:16px 0}h2{font-size:25px;letter-spacing:-.025em;margin-top:0}h3{font-size:18px;margin-top:28px}.subtitle{font-size:20px;color:var(--muted)}p{max-width:1100px}nav{display:flex;flex-wrap:wrap;gap:20px;border-block:1px solid var(--line);padding:14px 0}a{color:var(--accent)}.status{display:inline-block;font-size:13px;font-weight:800;letter-spacing:.04em;padding:8px 13px;border-radius:7px;background:var(--warn);color:var(--warnink)}.complete{background:#dcf4e9;color:#145638}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin:26px 0}.cards article,section:not(.cards){background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:23px}.cards span,.cards small{display:block;color:var(--muted);font-size:12px}.cards strong{display:block;font-size:33px;letter-spacing:-.04em;margin:8px 0}.cards article{padding:18px}section:not(.cards){margin:24px 0;scroll-margin-top:15px}.callout{border-left:4px solid var(--accent);padding:10px 15px;background:var(--bg)}.table-scroll{overflow:auto;margin:15px 0;border:1px solid var(--line);border-radius:8px;max-height:650px}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}th,td{padding:11px 13px;text-align:right;border-bottom:1px solid var(--line)}th{background:var(--bg);position:sticky;top:0;text-transform:uppercase;font-size:10px;letter-spacing:.055em;z-index:1}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}tbody tr:hover{background:var(--bg)}tbody tr:last-child td{border-bottom:0}details{border-top:1px solid var(--line);margin-top:20px;padding-top:16px}summary{cursor:pointer;color:var(--accent);font-weight:650}li{margin:10px 0}pre{overflow:auto;padding:18px;background:var(--bg);font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}footer{color:var(--muted);font-size:12px;padding:12px 0 26px}
@media(prefers-color-scheme:dark){:root{--bg:#0b151e;--panel:#112331;--ink:#e8f1f5;--muted:#9eb1c0;--line:#284152;--accent:#66d7d8;--warn:#44391e;--warnink:#ffda86}.complete{background:#163e32;color:#a4ead0}}
@media(max-width:850px){main{padding:20px 14px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}section:not(.cards){padding:18px}.cards strong{font-size:27px}}@media(max-width:440px){.cards{grid-template-columns:1fr}}
@media print{body{background:white;color:black}nav,details,footer{display:none}.table-scroll{overflow:visible;max-height:none}section{break-inside:avoid}.cards{grid-template-columns:repeat(4,1fr)}th{position:static}}
"""
    return '<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CatBoost MAE → RMSE · rapport diagnostique</title><style>' + css + '</style></head><body><main>' + body + '</main></body></html>'


def _safe_destination(directory: Path) -> Path:
    directory = Path(directory).absolute()
    if directory.resolve() != directory or directory.is_symlink():
        raise ValueError("Report directory must not contain symlinks or redirected path components")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_atomic(path: Path, content: str) -> None:
    if path.resolve() != path.absolute() or path.is_symlink():
        raise ValueError("Report destination must not be a symlink")
    if path.exists() and not path.is_file():
        raise ValueError("Report destination must be a regular file")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_report(frames: dict[str, pd.DataFrame], metadata: dict[str, Any], directory: Path) -> dict[str, Path]:
    """Write/refresh only the fixed HTML and JSON report files in ``directory``.

    Frames must contain finite verified observations on unique, aware UTC indexes.
    Nonfinite prediction hours are excluded jointly for the two loss variants.
    Empty input is supported and always emits a conspicuous PARTIAL/PENDING report.
    The caller is responsible for restricting the directory to its run namespace.
    """
    payload = _metrics(frames, metadata)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    markup = _html(payload)
    directory = _safe_destination(directory)
    paths = {"html": directory / "catboost_rmse_report.html", "metrics": directory / "catboost_rmse_metrics.json"}
    # Validate both destinations before updating either file.
    for path in paths.values():
        if path.is_symlink() or path.resolve() != path.absolute() or (path.exists() and not path.is_file()):
            raise ValueError("Report destinations must be regular files without symlink redirects")
    _write_atomic(paths["metrics"], serialized + "\n")
    _write_atomic(paths["html"], markup)
    return paths
