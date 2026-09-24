"""Clean-fuel full-chain variant using the unchanged operational HTML engine.

No inference, input synchronization, production publication, or activation is
performed here. The additional fuels must already belong to the supplied
Chronos context, residual inputs and Kalman covariates; labels cannot turn an
old residual-only experiment into the full-chain candidate.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import html
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.nuclear_cwe_reporting import (
    _attribute, _comparison, _observations, _view_frames,
)
from chronos2_hourly.nuclear_reporting import (
    _actual_values, _model_result, _report_zone_data, _timestamped,
)
from chronos2_hourly.reporting import _replace_report_labels
from chronos2_modular.common import ZoneData
from chronos2_modular.report import write_html_report
from .forecast import ENGINE
from .sources import SERIES, FORMULAS, HUBS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_FUEL_AUTONOMOUS_LABEL = "Chronos-2 + nucléaire FR + CGC/CCC + correcteur"
CLEAN_FUEL_KALMAN_LABEL = CLEAN_FUEL_AUTONOMOUS_LABEL + " + Kalman"
INCUMBENT_AUTONOMOUS_LABEL = "Chronos-2 + nucléaire FR + correcteur"
INCUMBENT_KALMAN_LABEL = INCUMBENT_AUTONOMOUS_LABEL + " + Kalman"
FUEL_ALIASES = ("cgc_fr", "cgc_de", "cgc_be", "cgc_nl", "ccc")
_LABELS = {**{f"cgc_{zone.lower()}": f"Clean Gas Cost {zone} (CO₂ inclus, EUR/MWh électrique)"
             for zone in ("FR", "DE", "BE", "NL")},
           "ccc": "Clean Coal Cost API#2 (CO₂ inclus, EUR/MWh électrique)"}
_Q = ("q10", "q50", "q90")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)


def _destination(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    allowed = (PROJECT_ROOT / "runs" / "experiments" / "nyx_clean_fuel_full_v1").resolve()
    if path != allowed and not path.is_relative_to(allowed):
        raise ValueError("Clean-fuel standard reports must remain inside runs/experiments/nyx_clean_fuel_full_v1.")
    return path


def _report_data(data: ZoneData, covariates: pd.DataFrame) -> ZoneData:
    """Keep actual known-future channel identities and fuel units in a copy."""
    copied = _report_zone_data(data, covariates)
    context = copied.model_context_covariates.copy(deep=True)
    original = data.model_context_covariates
    for alias in FUEL_ALIASES:
        channel = f"known_{alias}_oracle"
        if alias not in context or alias not in original or channel not in original or channel not in data.known_future_columns:
            raise ValueError(f"Full-chain clean-fuel report requires Chronos context/future and Kalman covariate {alias}.")
        raw = pd.to_numeric(original[alias].reindex(context.index), errors="raise").to_numpy(float)
        known = pd.to_numeric(original[channel].reindex(context.index), errors="raise").to_numpy(float)
        plotted = context[alias].to_numpy(float)
        if (not np.isfinite(raw).all() or not np.isfinite(known).all()
                or not np.allclose(raw, known, rtol=0, atol=1e-9)
                or not np.allclose(raw, plotted, rtol=0, atol=1e-9)):
            raise ValueError(f"Full-chain clean-fuel input {alias} is incomplete or its channels disagree.")
    for channel in data.known_future_columns:
        if channel not in original:
            raise ValueError(f"Declared known-future channel absent: {channel}")
        context[channel] = original[channel].reindex(context.index)
    manifest = copied.input_manifest.copy(deep=True)
    for alias in FUEL_ALIASES:
        mask = manifest.alias.astype(str).eq(alias)
        manifest.loc[mask, "source"] = _LABELS[alias]
        manifest.loc[mask, "information_type"] = "clean_fuel_cost_pre_cutoff"
        manifest.loc[mask, "unit"] = "EUR/MWh_e"
        manifest.loc[mask, "co2_already_included"] = True
    return replace(copied, input_manifest=manifest, model_context_covariates=context,
                   known_future_columns=list(data.known_future_columns))


def _fuel_attribution(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Require all new channels, after the shared forecast/SHA verification."""
    groups = raw.get("groups", [])
    for alias in FUEL_ALIASES:
        selected = [group for group in groups if group.get("key") == alias]
        legitimate = {alias, f"known_{alias}_oracle"}
        if len(selected) != 1 or any(not legitimate.intersection(selected[0].get(field, []))
                                    for field in ("context_columns", "future_columns")):
            raise ValueError("Clean-fuel attribution must be recalculated for all five fuel channels; old attribution is refused.")
    display = dict(raw)
    display["groups"] = [{**group, "label": _LABELS.get(group.get("key"), group.get("label"))}
                         for group in groups]
    if isinstance(raw.get("hourly"), pd.DataFrame):
        hourly = raw["hourly"].copy(deep=True)
        labels = hourly.variable_key.map(_LABELS)
        hourly.loc[labels.notna(), "variable_label"] = labels.dropna()
        display["hourly"] = hourly
    display.update(explained_model_label=CLEAN_FUEL_AUTONOMOUS_LABEL,
                   reported_model_label=CLEAN_FUEL_AUTONOMOUS_LABEL)
    return display


def _family_comparison(candidate: Any, incumbent: Any, *, day: pd.Timestamp,
                       timezone: str, family: str) -> dict[str, Any]:
    comparison = _comparison(candidate, incumbent, day=day, timezone=timezone)
    label = CLEAN_FUEL_KALMAN_LABEL if family == "kalman" else CLEAN_FUEL_AUTONOMOUS_LABEL
    reference = INCUMBENT_KALMAN_LABEL if family == "kalman" else INCUMBENT_AUTONOMOUS_LABEL
    return {**comparison, "candidate_model": f"clean_fuel_{family}",
            "incumbent_model": f"nuclear_{family}", "candidate_label": label, "incumbent_label": reference}


def _banner(audit: Mapping[str, Any], comparison: Mapping[str, Any], *, family: str) -> str:
    escape = lambda value: html.escape(str(value))
    selected = comparison[family]
    annual = selected["annual"]
    rows = []
    for metric, label in (("mae_eur_mwh", "MAE horaire"), ("rmse_eur_mwh", "RMSE horaire"),
                          ("daily_mean_mae_eur_mwh", "MAE des prix moyens journaliers")):
        values = (annual["incumbent"][metric], annual["candidate"][metric],
                  annual["gain_incumbent_minus_candidate"][metric])
        rows.append(f"<tr><th>{label}</th>" + "".join(f"<td>{float(value):.3f}</td>" for value in values) + "</tr>")
    attribution = ("Attribution chiffrée spécifique à cette prévision vérifiée. Dans le rapport Kalman, elle explique seulement Chronos-2 et le correcteur en amont du filtre ; le décalage Kalman est tenu fixe. Elle ne mesure donc pas l’influence totale des combustibles dans le Kalman ni des poids exacts sur son prix final."
                   if audit["variable_attribution_status"] == "verified_new_forecast_artifact"
                   else "Attribution chiffrée indisponible : elle doit être recalculée pour cette nouvelle chaîne. Aucun poids ni contribution de l’ancien modèle n’est repris.")
    return (
        '<section data-report-section="clean-fuel-methodology" data-diagnostic-only="true" data-production="false">'
        '<h2>Variante Clean Fuel Costs — chaîne complète, expérimentation séparée</h2>'
        '<p>Les coûts propres entrent dans <strong>Chronos-2, le correcteur résiduel et les covariables Kalman</strong>. '
        'Le nucléaire FR est conservé. Cette nouvelle prévision ne remplace aucun modèle opérationnel.</p>'
        '<p>CGC : coût gaz + CO₂ par MWh électrique. CCC : coût charbon API#2 + CO₂ par MWh électrique. '
        'Les indices natifs fournis contiennent déjà le CO₂ : il n’est pas ajouté une seconde fois. '
        'Les hubs et rendements exacts sont ceux des sources auditées ; aucun hub absent n’est inventé.</p>'
        '<p>Cutoff cible 08 h ; disponibilité des sources auditée en amont. Backtest : '
        f'{escape(audit["evaluation_start_day"])} au {escape(audit["evaluation_end_day"])} (365 jours avant livraison). '
        f'Statistics : {escape(selected["statistics_start_day"])} au {escape(selected["statistics_end_day"])} '
        '(365 derniers jours observés ; livraison incluse uniquement quand son prix observé est complet). '
        'Les observations non publiées restent vides. Storm est un comparateur, jamais une variable de ce modèle.</p>'
        f'<p>{escape(attribution)} Les contributions du modèle de base, du correcteur et du filtre ne sont pas des poids causaux. '
        'P10/P50/P90 sont ceux du modèle ; le rapport ne recalibre pas les intervalles.</p>'
        '<details><summary>Sources CGC/CCC, hubs et audit</summary><pre>' + escape(_json(audit)) + '</pre></details></section>'
        '<section data-report-section="clean-fuel-incumbent-comparison">'
        '<h2>Comparaison à la référence de même famille</h2><p>' + escape(selected["candidate_label"]) + ' vs '
        + escape(selected["incumbent_label"]) + '. Mêmes heures physiques et mêmes observations. '
        'Gain positif = erreur réduite ; unité EUR/MWh. Cette comparaison est indépendante des trous de Storm '
        'et ne constitue pas une validation prospective.</p><div class="table-wrap"><table><thead><tr><th>Métrique</th>'
        '<th>Référence inchangée</th><th>Clean Fuel Costs</th><th>Gain</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table></div><details><summary>Détail de la comparaison</summary><pre>'
        + escape(_json(selected)) + '</pre></details></section>'
    )


def render_clean_fuel_reports(
    result: Any, *, data: ZoneData, zone: str, delivery_day: str,
    output_directory: str | Path, incumbent: Any,
    storm_archive: str | Path | None = None,
    observed_source_audit: Mapping[str, Any] | None = None,
    attribution_directory: str | Path | None = None,
    source_audit: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Render new autonomous/Kalman reports with exact operational sections.

    ``incumbent`` must be the frozen full nuclear result, not only its Kalman
    view: each new model is compared with the incumbent of its own family.
    All validation is completed before the output directory is created.
    """
    directory = _destination(output_directory)
    zone = str(zone).strip().upper()
    if zone != str(data.zone).strip().upper():
        raise ValueError("Report zone does not match ZoneData.zone.")
    date = pd.Timestamp(delivery_day)
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError("delivery_day must be a local civil date without timezone.")
    engine = dict(_attribute(result, "audit", {}) or {})
    if engine.get("candidate_engine") != ENGINE:
        raise ValueError("Full-chain clean-fuel candidate_engine evidence is required; residual-only results cannot be relabelled.")
    day, start = date.date(), date.date() - timedelta(days=365)
    history_index = pd.date_range(pd.Timestamp(start, tz=data.timezone), pd.Timestamp(day, tz=data.timezone),
                                 freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    future_index = local_delivery_day_index(day, timezone=data.timezone).rename("delivery_start_utc")
    residual = _timestamped(_attribute(result, "residual_statistics"), name="residual_statistics")
    future = _timestamped(_attribute(result, "source_forecast"), name="source_forecast")
    history = residual.loc[(residual.index >= history_index[0]) & (residual.index <= history_index[-1])]
    incumbent_residual = _timestamped(_attribute(incumbent, "residual_statistics"), name="incumbent residual_statistics")
    incumbent_history = incumbent_residual.loc[(incumbent_residual.index >= history_index[0]) & (incumbent_residual.index <= history_index[-1])]
    incumbent_future = _timestamped(_attribute(incumbent, "source_forecast"), name="incumbent source_forecast")
    if any(not h.index.equals(history_index) for h in (history, incumbent_history)) or any(
            not f.index.equals(future_index) for f in (future, incumbent_future)):
        raise ValueError("Clean-fuel/incumbent FINAL365 and delivery must contain identical complete physical hours.")
    if not np.isfinite(_actual_values(history, name="residual_statistics")).all():
        raise ValueError("Clean-fuel historical observations must be complete.")
    kh, kf = _view_frames(_attribute(result, "kalman_view"), history_index, future_index, name="Clean-fuel Kalman")
    ih, iff = _view_frames(_attribute(incumbent, "kalman_view"), history_index, future_index, name="incumbent Kalman", allow_prefix=True)
    for label, auto, filtered in (("candidate", history, kh), ("incumbent", incumbent_history, ih)):
        if not np.allclose(_actual_values(filtered, name=label), _actual_values(auto, name=label), rtol=0, atol=1e-9):
            raise ValueError(f"{label}: Kalman/autonomous historical observations differ.")
    upstream = [f"residual_corrected__{q}" for q in _Q]
    if not set(upstream).issubset(kf) or not set(upstream).issubset(future) or not np.allclose(
            kf[upstream].to_numpy(float), future[upstream].to_numpy(float), rtol=0, atol=1e-9):
        raise ValueError("Clean-fuel Kalman upstream differs from the autonomous forecast.")
    actuals, observation_audit = _observations(residual, history, ih, future_index, data, zone, str(day), observed_source_audit)
    report_data = _report_data(data, _attribute(result, "covariates"))
    specifications = [
        ("autonomous", "residual_corrected", CLEAN_FUEL_AUTONOMOUS_LABEL, history, future),
        ("kalman", "residual_kalman", CLEAN_FUEL_KALMAN_LABEL, kh, kf),
        ("incumbent_autonomous", "residual_corrected", INCUMBENT_AUTONOMOUS_LABEL, incumbent_history, incumbent_future),
        ("incumbent_kalman", "residual_kalman", INCUMBENT_KALMAN_LABEL, ih, iff),
    ]
    prepared = {key: _model_result(history=h, forecast=f, actuals=actuals, model=model, label=label,
                                   data=report_data, output_directory=directory)
                for key, model, label, h, f in specifications}
    comparisons = {family: _family_comparison(prepared[family], prepared[f"incumbent_{family}"],
                    day=date, timezone=data.timezone, family=family) for family in ("autonomous", "kalman")}
    storm_audit: Mapping[str, Any] = {"status": "unavailable"}
    if storm_archive is not None:
        from chronos2_hourly.nuclear_report_benchmark import attach_nuclear_storm
        storm_audit = attach_nuclear_storm(prepared, Path(storm_archive), zone=zone, timezone=data.timezone)
    for item in prepared.values():
        item.statistics_scope_note = (
            "365 derniers jours observés, livraison non observée laissée vide ; heures physiques et observations "
            "communes à la variante Clean Fuel Costs et à sa référence nucléaire FR de même famille. Warm-up exclu. "
            + ("Storm officiel vérifié, uniquement aux heures communes disponibles."
               if storm_audit.get("status") == "complete" else "Storm indisponible : aucun comparateur reconstruit.")
        )
    attribution_status = "unavailable_requires_new_full_chain_attribution"
    if attribution_directory is not None:
        from chronos2_hourly.reporting import _attach_variable_attribution
        _attach_variable_attribution(prepared["autonomous"], directory=Path(attribution_directory),
                                     native_model="residual_corrected", timezone=data.timezone)
        if hasattr(prepared["autonomous"], "variable_attribution"):
            raw = _fuel_attribution(prepared["autonomous"].variable_attribution)
            prepared["autonomous"].variable_attribution = raw
            prepared["kalman"].variable_attribution = {
                **raw, "scope": "upstream_model", "is_upstream_attribution": True,
                "explained_model": "residual_corrected", "explained_model_label": CLEAN_FUEL_AUTONOMOUS_LABEL,
                "reported_model_label": CLEAN_FUEL_KALMAN_LABEL,
            }
            attribution_status = "verified_new_forecast_artifact"
    raw = _timestamped(_attribute(result, "raw_history"), name="raw_history")
    comparison = comparisons["kalman"]
    audit = {
        "schema_version": 1, "candidate_engine": ENGINE, "diagnostic_only": True,
        "production": False, "activation_performed": False, "zone": zone, "timezone": data.timezone,
        "delivery_day": str(day), "evaluation_start_day": str(start),
        "evaluation_end_day": str(day - timedelta(days=1)), "evaluation_days": 365,
        "evaluation_hours": len(history_index), "statistics_start_day": comparison["statistics_start_day"],
        "statistics_end_day": comparison["statistics_end_day"],
        "delivery_day_observed_hours": int(actuals.reindex(future_index).notna().sum()),
        "delivery_day_placeholder_hours": int(actuals.reindex(future_index).isna().sum()),
        "same_actual_hours": True, "historical_observations": observation_audit,
        "storm_used_as_input": False, "storm_hourly_comparison": dict(storm_audit),
        "variable_attribution_status": attribution_status, "co2_already_included": True,
        "gas_hubs": HUBS, "cost_formulas": FORMULAS, "source_audit": dict(source_audit or {}),
        "fuel_inputs": {alias: {"label": _LABELS[alias], "unit": "EUR/MWh_e",
                        "series": SERIES[alias],
                        "known_future_channel": f"known_{alias}_oracle"} for alias in FUEL_ALIASES},
        "warmup": {"raw_days_before_final365": len(set(raw.index[raw.index < history_index[0]].tz_convert(data.timezone).date)),
                   "residual_days_before_final365": len(set(residual.index[residual.index < history_index[0]].tz_convert(data.timezone).date)),
                   "excluded_from_report_metrics": True},
        "engine_audit": engine, "incumbent_comparisons": comparisons,
    }
    banners = {family: _banner(audit, comparisons, family=family) for family in ("autonomous", "kalman")}
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for family, label in (("autonomous", CLEAN_FUEL_AUTONOMOUS_LABEL), ("kalman", CLEAN_FUEL_KALMAN_LABEL)):
        path = directory / f"forecast_{zone.lower()}_{day}_clean_fuel_{family}.html"
        write_html_report([prepared[family]], {"report": {"title": f"{zone} — {label}", "forecast_history_hours": 168}}, path)
        _replace_report_labels(path, native_label=label, baseline_label=None)
        document = path.read_text(encoding="utf-8")
        document = re.sub(r'<h3>Comparaison au modèle prix seul</h3>\s*(?:<div class="table-wrap">)?'
                          r'<table\b.*?</table>(?:</div>)?', "", document, count=1, flags=re.DOTALL)
        document = document.replace("Prévision opérationnelle du", "Prévision expérimentale du")
        document = document.replace("<main>", "<main>" + banners[family], 1)
        path.write_text(document, encoding="utf-8")
        paths[family] = path
    for key, filename, payload in (("audit", "clean_fuel_report_audit.json", audit),
                                   ("comparison", "clean_fuel_incumbent_comparison.json", comparisons)):
        path = directory / filename
        path.write_text(_json(payload), encoding="utf-8")
        paths[key] = path
    return paths


__all__ = ["render_clean_fuel_reports", "CLEAN_FUEL_AUTONOMOUS_LABEL", "CLEAN_FUEL_KALMAN_LABEL"]
