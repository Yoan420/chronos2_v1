"""Isolated CWE nuclear reports, using the unchanged standard hourly renderer.

No fetching, training, forecast revision, activation or incumbent publication
takes place here.  Storm remains the official evaluation-only comparator;
the frozen incumbent has its own explicitly paired comparison section.
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

from .hourly_contract import local_delivery_day_index
from .nuclear_reporting import (
    _actual_values, _model_result, _report_zone_data, _require_quantiles, _timestamped,
)
from .observation_precision import validate_observation_precision
from .reporting import _replace_report_labels
from chronos2_modular.common import ZoneData
from chronos2_modular.report import write_html_report


CWE_AUTONOMOUS_LABEL = "Chronos-2 + nucléaire CWE + correcteur"
CWE_KALMAN_LABEL = CWE_AUTONOMOUS_LABEL + " + Kalman"
INCUMBENT_LABEL = "Chronos-2 + nucléaire FR + correcteur + Kalman"
_SOURCE_LABELS = {
    "fr_nuclear_generation_fcst_gw": ("FR", "Prévision de génération nucléaire", "generation_forecast"),
    "be_nuclear_available_gw": ("BE", "Prévision de puissance maximale disponible (Pmax)", "capacity_forecast"),
    "nl_nuclear_available_gw": ("NL", "Prévision de puissance maximale disponible (Pmax)", "capacity_forecast"),
}
_Q = ("q10", "q50", "q90")


def _attribute(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _json(value: Any) -> str:
    # Report metrics contain only finite Python floats or None.  Audits may
    # contain timestamps/paths, which are provenance strings, not numeric data.
    return json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)


def _view_frames(view: Any, history_index: pd.DatetimeIndex, future_index: pd.DatetimeIndex,
                 *, name: str, allow_prefix: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if view is None:
        raise ValueError(f"{name}: a frozen Kalman view is required.")
    history = _timestamped(_attribute(view, "backtest"), name=f"{name}.backtest")
    forecast = _timestamped(_attribute(view, "forecast"), name=f"{name}.forecast")
    if allow_prefix:
        history = history.loc[(history.index >= history_index[0]) & (history.index <= history_index[-1])]
    if not history.index.equals(history_index) or not forecast.index.equals(future_index):
        raise ValueError(f"{name}: physical hours differ from the common FINAL365/delivery grid.")
    for frame in (history, forecast):
        _require_quantiles(frame, "residual_kalman", name=name)
    if not np.isfinite(_actual_values(history, name=name)).all():
        raise ValueError(f"{name}: historical observations must be complete.")
    return history, forecast


def _observations(residual: pd.DataFrame, history: pd.DataFrame, incumbent_history: pd.DataFrame,
                  future_index: pd.DatetimeIndex, data: ZoneData, zone: str, day: str,
                  observed_source_audit: Mapping[str, Any] | None) -> tuple[pd.Series, dict[str, Any]]:
    old_actual = _actual_values(history, name="residual_statistics")
    target = data.target.copy(deep=True)
    target.index = pd.DatetimeIndex(pd.to_datetime(target.index, utc=True, errors="raise"))
    if target.index.hasnans or target.index.has_duplicates:
        raise ValueError("ZoneData.target contains missing or duplicate physical timestamps.")
    target = pd.to_numeric(target, errors="raise")
    if np.isinf(target.to_numpy(float)).any():
        raise ValueError("Canonical observations must not be infinite.")
    verified = None
    if observed_source_audit is not None:
        from .nuclear_reporting_refresh import verify_refreshed_observations
        verified = verify_refreshed_observations(target, observed_source_audit, zone=zone,
                                                timezone=data.timezone, delivery_day=day)
    target_history = target.reindex(history.index)
    available = np.isfinite(target_history.to_numpy(float))
    incumbent_actual = _actual_values(incumbent_history, name="incumbent")
    precision = {}
    if verified is None:
        precision["incumbent_vs_candidate"] = validate_observation_precision(
            incumbent_actual, old_actual, name="Incumbent/candidate observations")
        if available.any():
            precision["candidate_vs_canonical"] = validate_observation_precision(
                old_actual[available], target_history.to_numpy(float)[available],
                name="FINAL365 observations")
    actual = pd.Series(np.where(available, target_history, old_actual), index=history.index, name="actual")
    delivery = (pd.to_numeric(residual["actual"].reindex(future_index), errors="coerce")
                if "actual" in residual else pd.Series(np.nan, index=future_index))
    target_delivery = target.reindex(future_index)
    comparable = delivery.notna() & target_delivery.notna()
    if verified is None and not np.allclose(delivery[comparable], target_delivery[comparable], rtol=0, atol=1e-9):
        raise ValueError("Delivery-day observations disagree between source frames.")
    delivery = target_delivery.copy() if verified is not None else delivery.combine_first(target_delivery)
    finite = np.isfinite(delivery.to_numpy(float))
    if finite.any() and not finite.all():
        raise ValueError("Delivery-day observations must be complete or unavailable.")
    delivery.loc[~finite] = np.nan
    return pd.concat([actual, delivery.rename("actual")]), {
        "precision": precision, "verified_latest_canonical_reporting_observations": verified,
        "training_inputs_modified": False,
        "candidate_revised_hours": int((np.abs(actual.to_numpy(float) - old_actual) > 1e-9).sum()),
        "incumbent_revised_hours": int((np.abs(actual.to_numpy(float) - incumbent_actual) > 1e-9).sum()),
    }


def _report_data(data: ZoneData, covariates: pd.DataFrame) -> ZoneData:
    copied = _report_zone_data(data, covariates)
    manifest = copied.input_manifest.copy(deep=True)
    for alias, (zone, description, kind) in _SOURCE_LABELS.items():
        if alias not in copied.model_context_covariates:
            raise ValueError(f"CWE report: required covariate {alias} is absent.")
        mask = manifest["alias"].astype(str).eq(alias)
        manifest.loc[mask, "source_zone"] = zone
        manifest.loc[mask, "source"] = description
        manifest.loc[mask, "information_type"] = kind
        manifest.loc[mask, "unit"] = "GW"
    # The Kalman view carries raw aliases, whereas Chronos/residual inputs use
    # the separately declared known_{alias}_oracle channels.  Preserve their
    # actual identity and source values in the reporting copy; never declare a
    # raw series known in the future merely because it is plotted as an input.
    future = list(data.known_future_columns)
    context = copied.model_context_covariates.copy(deep=True)
    original_context = data.model_context_covariates
    for column in future:
        if column not in original_context:
            raise ValueError(f"CWE report: declared known-future channel {column} is absent from the source context.")
        context[column] = original_context[column].reindex(context.index)
    return replace(copied, input_manifest=manifest, known_future_columns=future,
                   model_context_covariates=context)


def _cwe_attribution(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Check real model-column identities, then relabel only display copies.

    The shared loader has already verified the original artifact/checksums and
    Shapley reconstruction. Oracle names here identify pre-cutoff known-future
    channels; accepting them does not assert provider-vintage PIT certification.
    """
    groups = raw.get("groups", [])
    for alias in _SOURCE_LABELS:
        selected = [group for group in groups if group.get("key") == alias]
        legitimate = {alias, f"known_{alias}_oracle"}
        if len(selected) != 1 or any(not legitimate.intersection(selected[0].get(field, []))
                                      for field in ("context_columns", "future_columns")):
            raise ValueError("CWE attribution must explain the three new nuclear inputs, not an old FR-only artifact.")
    display_labels = {
        "fr_nuclear_generation_fcst_gw": "Génération nucléaire prévue FR (GW)",
        "be_nuclear_available_gw": "Pmax nucléaire disponible prévue BE (GW)",
        "nl_nuclear_available_gw": "Pmax nucléaire disponible prévue NL (GW)",
    }
    display = dict(raw)
    display["groups"] = [{**group, "label": display_labels.get(group.get("key"), group.get("label"))}
                         for group in groups]
    if isinstance(raw.get("hourly"), pd.DataFrame):
        hourly = raw["hourly"].copy(deep=True)
        replacement = hourly["variable_key"].map(display_labels)
        hourly.loc[replacement.notna(), "variable_label"] = replacement.dropna()
        display["hourly"] = hourly
    display.update(explained_model_label=CWE_AUTONOMOUS_LABEL, reported_model_label=CWE_AUTONOMOUS_LABEL,
                   display_labels=display_labels)
    return display


def _comparison(candidate: Any, incumbent: Any, *, day: pd.Timestamp, timezone: str) -> dict[str, Any]:
    candidate_frame = _timestamped(candidate.statistics_candidate, name="candidate Statistics")
    baseline_frame = _timestamped(incumbent.statistics_candidate, name="incumbent Statistics")
    if not candidate_frame.index.equals(baseline_frame.index):
        raise ValueError("Candidate/incumbent Statistics physical hours differ.")
    if not np.allclose(candidate_frame.actual, baseline_frame.actual, rtol=0, atol=0, equal_nan=True):
        raise ValueError("Candidate/incumbent Statistics observed prices differ.")
    observed = candidate_frame.actual.notna()
    last_day = max(candidate_frame.index[observed].tz_convert(timezone).date)
    first_day = last_day - timedelta(days=364)
    index = pd.date_range(pd.Timestamp(first_day, tz=timezone),
                          pd.Timestamp(last_day + timedelta(days=1), tz=timezone),
                          freq="h", inclusive="left").tz_convert("UTC")
    candidate_values = candidate_frame.reindex(index)
    baseline_values = baseline_frame.reindex(index)
    if not np.isfinite(candidate_values[["actual", "q50"]].to_numpy(float)).all() or not np.isfinite(baseline_values.q50).all():
        raise ValueError("Candidate/incumbent latest365 Statistics comparison must be complete.")

    def score(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"hours": 0, "days": 0, "mae_eur_mwh": None, "rmse_eur_mwh": None,
                    "bias_eur_mwh": None, "daily_mean_mae_eur_mwh": None}
        error = frame.q50.to_numpy(float) - frame.actual.to_numpy(float)
        daily = frame[["q50", "actual"]].groupby(frame.index.tz_convert(timezone).date).mean()
        return {"hours": len(frame), "days": len(daily),
                "mae_eur_mwh": float(np.abs(error).mean()), "rmse_eur_mwh": float(np.sqrt(np.mean(error**2))),
                "bias_eur_mwh": float(error.mean()),
                "daily_mean_mae_eur_mwh": float((daily.q50 - daily.actual).abs().mean())}

    def paired(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
        c, b = score(left), score(right)
        gains = {key: (b[key] - c[key] if c[key] is not None else None)
                 for key in ("mae_eur_mwh", "rmse_eur_mwh", "daily_mean_mae_eur_mwh")}
        return {"candidate": c, "incumbent": b, "gain_incumbent_minus_candidate": gains}

    local_dates = pd.Index(index.tz_convert(timezone).date)
    june_days = [str(value) for value in sorted(set(local_dates)) if value.month == 6 and value.day in (24, 25, 26)]
    june_mask = np.array([str(value) in june_days for value in local_dates])
    return {
        "schema_version": 1, "zone": candidate.zone, "timezone": timezone,
        "candidate_model": "nuclear_cwe_kalman", "incumbent_model": "nuclear_kalman",
        "candidate_label": CWE_KALMAN_LABEL, "incumbent_label": INCUMBENT_LABEL,
        "statistics_start_day": str(first_day), "statistics_end_day": str(last_day),
        "statistics_days": 365, "delivery_day_included": last_day == day.date(),
        "same_physical_hours": True, "same_observed_prices": True,
        "mask_policy": "same_complete_candidate_incumbent_observed_hours; independent_of_Storm_availability",
        "annual": paired(candidate_values, baseline_values),
        "june_24_26": {"days": june_days, **paired(candidate_values.loc[june_mask], baseline_values.loc[june_mask])},
        "interpretation": "Positive gain means a smaller candidate error; this is not a prospective production validation.",
    }


def _banner(audit: Mapping[str, Any], comparison: Mapping[str, Any]) -> str:
    def escaped(value: Any) -> str:
        return html.escape(str(value))

    annual = comparison["annual"]
    rows = []
    for key, label in (("mae_eur_mwh", "MAE horaire"), ("rmse_eur_mwh", "RMSE horaire"),
                       ("daily_mean_mae_eur_mwh", "MAE des prix moyens journaliers")):
        rows.append("<tr><th>" + label + "</th>" + "".join(
            f"<td>{float(value):.3f}</td>" for value in (
                annual["incumbent"][key], annual["candidate"][key],
                annual["gain_incumbent_minus_candidate"][key])) + "</tr>")
    return (
        '<section data-report-section="nuclear-cwe-methodology" data-diagnostic-only="true" data-production="false">'
        '<h2>Variante nucléaire CWE — expérimentation séparée</h2>'
        '<p>Les modèles historiques et leurs rapports restent inchangés. Aucun entraînement ni '
        'promotion n’est effectué par ce rapport. Une bonne performance historique ne constitue '
        'pas une validation prospective en production.</p>'
        '<p>FR : <strong>prévision de génération nucléaire (GW)</strong>. BE et NL : '
        '<strong>prévision de puissance maximale disponible (Pmax, GW)</strong>, '
        'et non une prévision de génération effective. Ces trois séries entrent dans le modèle. '
        'DE, AT et LU : métadonnées structurelles uniquement ; aucun canal zéro et aucun '
        'remplissage de données manquantes par zéro ne sont créés par le rapport.</p>'
        '<p>Cutoff cible : 08 h. La preuve PIT dépend des vintages audités de chaque source ; '
        'elle n’est pas certifiée par le présent rapport. Les règles DST et les éventuelles '
        'diffusions de valeurs journalières sont auditées source par source. Les comparaisons '
        'conservent les heures physiques UTC et les prix observés canoniques.</p>'
        f'<p>Backtest : {escaped(audit["evaluation_start_day"])} au {escaped(audit["evaluation_end_day"])} '
        '(365 jours avant livraison, warm-up exclu). Statistics : '
        f'{escaped(comparison["statistics_start_day"])} au {escaped(comparison["statistics_end_day"])} '
        '(365 derniers jours observés). La livraison sans observation reste vide. '
        'Storm officiel reste exclusivement le comparateur d’évaluation des Statistics ; '
        'ses trous ne sont jamais interpolés.</p>'
        '<p>Les contributions Chronos-2, correcteur et Kalman décomposent le prix final, '
        'sans représenter des poids causaux. L’influence des variables et du prix passé '
        'n’est affichée que si un artefact spécifique à cette nouvelle prévision est vérifié. '
        'P10/P50/P90 sont conservés ; les déciles intermédiaires/CRPS sont des approximations d’affichage.</p>'
        '<details><summary>Sources et audit CWE</summary><pre>' + escaped(_json(audit)) + '</pre></details></section>'
        '<section data-report-section="nuclear-cwe-incumbent-comparison">'
        '<h2>Nucléaire CWE + Kalman vs modèle nucléaire FR + Kalman</h2>'
        '<p>Deux prévisions figées, exactement les mêmes heures et observations, sur la fenêtre Statistics. '
        'Ce tableau compare les deux modèles Kalman, y compris dans le rapport autonome. '
        'Gain positif = erreur réduite. Unité : EUR/MWh. La comparaison n’est pas filtrée par la disponibilité de Storm.</p>'
        '<div class="table-wrap"><table><thead><tr><th>Métrique</th><th>Nucléaire FR + Kalman</th>'
        '<th>Nucléaire CWE + Kalman</th><th>Gain</th></tr></thead><tbody>' + "".join(rows)
        + '</tbody></table></div><details><summary>Détail et diagnostic des 24–26 juin</summary><pre>'
        + escaped(_json(comparison)) + '</pre></details></section>'
    )


def render_nuclear_cwe_reports(
    result: Any, *, incumbent: Any, data: ZoneData, zone: str, delivery_day: str,
    output_directory: str | Path, source_audit: Mapping[str, Any],
    storm_archive: str | Path | None = None,
    observed_source_audit: Mapping[str, Any] | None = None,
    attribution_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Render two new reports and audits; accept a frozen incumbent Kalman view.

    ``result`` has the existing NuclearForecastResult schema. ``incumbent`` is
    either its historical counterpart or a ``backtest``/``forecast`` Kalman view.
    Both use existing residual_corrected/residual_kalman quantile prefixes.
    All validation occurs before creating report files. Source objects are never
    mutated. The caller owns isolated publication/atomic directory handling.
    """
    zone = str(zone).strip().upper()
    if zone != str(data.zone).strip().upper():
        raise ValueError("Report zone does not match ZoneData.zone.")
    date = pd.Timestamp(delivery_day)
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError("delivery_day must be a local civil date without timezone.")
    day = date.date()
    start = day - timedelta(days=365)
    history_index = pd.date_range(pd.Timestamp(start, tz=data.timezone), pd.Timestamp(day, tz=data.timezone),
                                 freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    future_index = local_delivery_day_index(day, timezone=data.timezone).rename("delivery_start_utc")
    residual = _timestamped(_attribute(result, "residual_statistics"), name="residual_statistics")
    future = _timestamped(_attribute(result, "source_forecast"), name="source_forecast")
    history = residual.loc[(residual.index >= history_index[0]) & (residual.index <= history_index[-1])]
    if not history.index.equals(history_index) or not future.index.equals(future_index):
        raise ValueError("CWE FINAL365/delivery must contain every physical hour exactly once.")
    if not np.isfinite(_actual_values(history, name="residual_statistics")).all():
        raise ValueError("CWE historical observations must be complete.")
    kh, kf = _view_frames(_attribute(result, "kalman_view"), history_index, future_index, name="CWE Kalman")
    ih, iff = _view_frames(_attribute(incumbent, "kalman_view", incumbent), history_index, future_index,
                          name="incumbent", allow_prefix=True)
    if not np.allclose(_actual_values(kh, name="CWE Kalman"), history.actual, rtol=0, atol=1e-9):
        raise ValueError("CWE Kalman/autonomous observations differ.")
    upstream = [f"residual_corrected__{q}" for q in _Q]
    if not set(upstream).issubset(kf) or not set(upstream).issubset(future) or not np.allclose(
            kf[upstream].to_numpy(float), future[upstream].to_numpy(float), rtol=0, atol=1e-9):
        raise ValueError("CWE Kalman upstream differs from the explained autonomous forecast.")
    actuals, observation_audit = _observations(residual, history, ih, future_index, data, zone,
                                             str(day), observed_source_audit)
    directory = Path(output_directory).expanduser().resolve()
    report_data = _report_data(data, _attribute(result, "covariates"))
    specifications = [
        ("autonomous", "residual_corrected", CWE_AUTONOMOUS_LABEL, history, future),
        ("kalman", "residual_kalman", CWE_KALMAN_LABEL, kh, kf),
        ("incumbent", "residual_kalman", INCUMBENT_LABEL, ih, iff),
    ]
    prepared = {key: _model_result(history=h, forecast=f, actuals=actuals, model=model, label=label,
                                   data=report_data, output_directory=directory)
                for key, model, label, h, f in specifications}
    comparison = _comparison(prepared["kalman"], prepared["incumbent"], day=date, timezone=data.timezone)
    storm_audit: Mapping[str, Any] = {"status": "unavailable"}
    if storm_archive is not None:
        from .nuclear_report_benchmark import attach_nuclear_storm
        storm_audit = attach_nuclear_storm(prepared, Path(storm_archive), zone=zone, timezone=data.timezone)
    for item in prepared.values():
        item.statistics_scope_note = (
            "365 derniers jours observés, livraison non observée laissée vide ; heures physiques et prix "
            "canoniques communs au modèle CWE et au modèle nucléaire FR figé. Warm-up exclu. "
            + ("Storm officiel vérifié, uniquement aux heures communes disponibles."
               if storm_audit.get("status") == "complete" else "Storm indisponible : aucun comparateur reconstruit.")
        )
    attribution_status = "unavailable_no_matching_artifact_supplied"
    if attribution_directory is not None:
        from .reporting import _attach_variable_attribution
        _attach_variable_attribution(prepared["autonomous"], directory=Path(attribution_directory),
                                     native_model="residual_corrected", timezone=data.timezone)
        if hasattr(prepared["autonomous"], "variable_attribution"):
            raw = _cwe_attribution(prepared["autonomous"].variable_attribution)
            prepared["autonomous"].variable_attribution = raw
            prepared["kalman"].variable_attribution = {
                **raw, "scope": "upstream_model", "is_upstream_attribution": True,
                "explained_model": "residual_corrected", "explained_model_label": CWE_AUTONOMOUS_LABEL,
                "reported_model_label": CWE_KALMAN_LABEL,
            }
            attribution_status = "verified_new_forecast_artifact"
    raw = _timestamped(_attribute(result, "raw_history"), name="raw_history")
    audit = {
        "schema_version": 1, "candidate_engine": "nuclear_cwe_forecast_v1",
        "diagnostic_only": True, "production": False, "activation_performed": False,
        "zone": zone, "timezone": data.timezone, "delivery_day": str(day),
        "evaluation_start_day": str(start), "evaluation_end_day": str(day - timedelta(days=1)),
        "evaluation_days": 365, "evaluation_hours": len(history_index),
        "statistics_start_day": comparison["statistics_start_day"],
        "statistics_end_day": comparison["statistics_end_day"],
        "delivery_day_observed_hours": int(actuals.reindex(future_index).notna().sum()),
        "delivery_day_placeholder_hours": int(actuals.reindex(future_index).isna().sum()),
        "same_actual_hours": True, "historical_observations": observation_audit,
        "storm_used_as_input": False, "storm_hourly_comparison": dict(storm_audit),
        "variable_attribution_status": attribution_status,
        "warmup": {"raw_days_before_final365": len(set(raw.index[raw.index < history_index[0]].tz_convert(data.timezone).date)),
                   "residual_days_before_final365": len(set(residual.index[residual.index < history_index[0]].tz_convert(data.timezone).date)),
                   "excluded_from_report_metrics": True},
        "dynamic_nuclear_sources": {alias: {"zone": entry[0], "description": entry[1],
                                              "information_type": entry[2], "unit": "GW"}
                                    for alias, entry in _SOURCE_LABELS.items()},
        "structural_countries": {country: {"role": "metadata_only", "model_channel_added": False,
                                           "missing_data_filled_with_zero": False} for country in ("DE", "AT", "LU")},
        "dst_policy": "per-source audited semantics; unchanged UTC physical-hour pairing; no price interpolation",
        "engine_audit": dict(_attribute(result, "audit", {}) or {}),
        "source_audit": dict(source_audit),
        "incumbent_comparison": comparison,
    }
    banner = _banner(audit, comparison)  # Validate serializability before writing.
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, label in (("autonomous", CWE_AUTONOMOUS_LABEL), ("kalman", CWE_KALMAN_LABEL)):
        path = directory / f"forecast_{zone.lower()}_{day}_nuclear_cwe_{key}.html"
        write_html_report([prepared[key]], {"report": {"title": f"{zone} — {label}", "forecast_history_hours": 168}}, path)
        _replace_report_labels(path, native_label=label, baseline_label=None)
        document = path.read_text(encoding="utf-8")
        document = re.sub(r'<h3>Comparaison au modèle prix seul</h3>\s*(?:<div class="table-wrap">)?'
                          r'<table\b.*?</table>(?:</div>)?', "", document, count=1, flags=re.DOTALL)
        document = document.replace("Prévision opérationnelle du", "Prévision expérimentale du")
        document = document.replace("<main>", "<main>" + banner, 1)
        path.write_text(document, encoding="utf-8")
        paths[key] = path
    for key, filename, payload in (("audit", "nuclear_cwe_report_audit.json", audit),
                                   ("comparison", "nuclear_cwe_incumbent_comparison.json", comparison)):
        path = directory / filename
        path.write_text(_json(payload), encoding="utf-8")
        paths[key] = path
    return paths


__all__ = ["CWE_AUTONOMOUS_LABEL", "CWE_KALMAN_LABEL", "render_nuclear_cwe_reports"]
