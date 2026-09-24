"""Isolated temperature/heatwave experiment reports; no model or source mutation.

The standard hourly renderer remains responsible for Statistics, Storm,
dark mode, the calendar, and hourly/attribution panels.  This adapter adds a
strictly forecast-defined heat-regime comparison on the same observed year.
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
from .heatwave_features import (HeatwaveFeatureConfig, heatwave_feature_aliases,
                                validate_heatwave_aggregates)
from .nuclear_cwe_reporting import _attribute, _observations, _view_frames
from .nuclear_reporting import _actual_values, _model_result, _report_zone_data, _timestamped
from .reporting import _replace_report_labels
from chronos2_modular.common import ZoneData
from chronos2_modular.report import write_html_report


COUNTRIES = ("FR", "DE", "BE", "NL", "ES")
TEMPERATURE_ALIASES = tuple(f"{country.lower()}_temperature_fcst" for country in COUNTRIES)
HEAT_FRACTION = "heat_fraction_fcst"
HEATWAVE_AUTONOMOUS_LABEL = "Chronos-2 + nucléaire FR + températures Europe + correcteur"
HEATWAVE_KALMAN_LABEL = HEATWAVE_AUTONOMOUS_LABEL + " + Kalman"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)


def _source_labels(config: HeatwaveFeatureConfig | None = None) -> dict[str, tuple[str, str, str]]:
    config = config or HeatwaveFeatureConfig()
    labels = {}
    for country in COUNTRIES:
        labels[f"{country.lower()}_temperature_fcst"] = (country, f"Indice journalier de température prévu {country}", "°C")
        labels[f"{country.lower()}_heat_excess_fcst_c"] = (country, f"Excès de chaleur prévu {country}", "°C")
        labels[f"{country.lower()}_heat_streak_fcst_days"] = (country, f"Persistance de chaleur prévue {country}", "jours")
    labels[HEAT_FRACTION] = ("Europe", f"Part des pays avec au moins {config.persistent_days} jours chauds prévus", "fraction")
    if config.include_cooling_mean:
        labels["cooling_degree_mean_fcst_c"] = ("Europe", "Degrés de refroidissement moyens prévus", "°C")
    return labels


def _feature_frame(covariates: pd.DataFrame, index: pd.DatetimeIndex, timezone: str,
                   config: HeatwaveFeatureConfig | None = None) -> pd.DataFrame:
    """Never infer hot days from actual temperatures or forecast errors."""
    source = _timestamped(covariates, name="heatwave covariates")
    config = config or HeatwaveFeatureConfig()
    required = heatwave_feature_aliases(config)
    output = pd.DataFrame(index=index)
    for alias in required:
        column = alias if alias in source else f"known_{alias}_oracle"
        if column not in source:
            raise ValueError(f"Heatwave report: missing forecast feature {alias}.")
        output[alias] = pd.to_numeric(source[column].reindex(index), errors="raise")
    if not np.isfinite(output.to_numpy(float)).all():
        raise ValueError("Heatwave comparison requires finite forecast features on every Statistics/delivery hour.")
    local_days = index.tz_convert(timezone).date
    if (output.groupby(local_days).nunique() > 1).any().any():
        raise ValueError("Temperature/heatwave inputs are daily indices, not hourly temperature forecasts.")
    streaks = output[[f"{country.lower()}_heat_streak_fcst_days" for country in COUNTRIES]]
    if ((streaks < 0) | (streaks > config.streak_clip_days) | (streaks != np.floor(streaks))).any().any():
        raise ValueError(f"Heatwave persistence must be an integer between zero and {config.streak_clip_days}.")
    excess = output[[f"{country.lower()}_heat_excess_fcst_c" for country in COUNTRIES]]
    if (excess < 0).any().any():
        raise ValueError("Forecast heat excess cannot be negative.")
    # The shared preparation stores channels as float32.  The engine's bounded
    # half-ULP validation accepts that representation without hiding a changed
    # recipe or applying a broad price/feature tolerance.
    aggregate_precision = validate_heatwave_aggregates(output, config=config)
    # Cohort membership uses the exact discrete fraction after verification,
    # never a tiny representational residue >0 on a mathematically zero day.
    # This reporting copy is separate from every displayed/source model input.
    output[HEAT_FRACTION] = (streaks >= config.persistent_days).mean(axis=1)
    output.attrs["aggregate_precision_audit"] = aggregate_precision
    return output


def _report_data(data: ZoneData, covariates: pd.DataFrame, config: HeatwaveFeatureConfig | None = None) -> ZoneData:
    copied = _report_zone_data(data, covariates)
    manifest = copied.input_manifest.copy(deep=True)
    labels = _source_labels(config)
    for alias, (country, description, unit) in labels.items():
        mask = manifest["alias"].astype(str).isin([alias, f"known_{alias}_oracle"])
        manifest.loc[mask, "source_zone"] = country
        manifest.loc[mask, "source"] = description
        manifest.loc[mask, "unit"] = unit
        manifest.loc[mask, "information_type"] = (
            "daily_temperature_forecast_index" if alias in TEMPERATURE_ALIASES else "causal_forecast_derived_index")
    context = copied.model_context_covariates.copy(deep=True)
    # Keep the original known-future identity. Plotting a raw alias never grants
    # it future availability, and a known_* channel is not a PIT certificate.
    for column in data.known_future_columns:
        if column not in data.model_context_covariates:
            raise ValueError(f"Declared known-future channel absent from source context: {column}.")
        context[column] = data.model_context_covariates[column].reindex(context.index)
    return replace(copied, input_manifest=manifest, model_context_covariates=context,
                   known_future_columns=list(data.known_future_columns))


def _attribution(raw: Mapping[str, Any], *, required: list[str], label: str,
                 config: HeatwaveFeatureConfig | None = None) -> dict[str, Any]:
    """Validate the new input identities before displaying any attribution."""
    groups = raw.get("groups", [])
    for alias in required:
        matched = [group for group in groups if group.get("key") == alias]
        legitimate = {alias, f"known_{alias}_oracle"}
        if len(matched) != 1 or any(not legitimate.intersection(matched[0].get(field, []))
                                   for field in ("context_columns", "future_columns")):
            raise ValueError(f"Heatwave attribution must explain the new input {alias}, not a previous forecast.")
    labels = {key: value[1] for key, value in _source_labels(config).items()}
    display = {**raw, "groups": [{**group, "label": labels.get(group.get("key"), group.get("label"))}
                                   for group in groups], "explained_model_label": label,
               "reported_model_label": label, "display_labels": labels}
    if isinstance(raw.get("hourly"), pd.DataFrame):
        hourly = raw["hourly"].copy(deep=True)
        replacement = hourly["variable_key"].map(labels)
        hourly.loc[replacement.notna(), "variable_label"] = replacement.dropna()
        display["hourly"] = hourly
    return display


def _score(actual: pd.Series, point: pd.Series, *, timezone: str) -> dict[str, Any]:
    if not len(actual):
        return {"hours": 0, "days": 0, **{key: None for key in
                ("mae_eur_mwh", "rmse_eur_mwh", "bias_eur_mwh", "daily_mean_mae_eur_mwh")}}
    error = point.to_numpy(float) - actual.to_numpy(float)
    daily = pd.DataFrame({"actual": actual, "point": point}).groupby(actual.index.tz_convert(timezone).date).mean()
    return {"hours": len(actual), "days": len(daily), "mae_eur_mwh": float(np.abs(error).mean()),
            "rmse_eur_mwh": float(np.sqrt(np.mean(error ** 2))), "bias_eur_mwh": float(error.mean()),
            "daily_mean_mae_eur_mwh": float((daily.point - daily.actual).abs().mean())}


def _regime_comparison(prepared: Mapping[str, Any], features: pd.DataFrame, *, timezone: str,
                       day: pd.Timestamp, incumbent_label: str, config: HeatwaveFeatureConfig) -> dict[str, Any]:
    frames = {key: _timestamped(item.statistics_candidate, name=f"{key} Statistics")
              for key, item in prepared.items()}
    reference = frames["incumbent"]
    for frame in frames.values():
        if not frame.index.equals(reference.index) or not np.allclose(frame.actual, reference.actual,
                                                                     rtol=0, atol=0, equal_nan=True):
            raise ValueError("Heatwave/Incumbent Statistics require identical physical hours and observations.")
    last_day = reference.index[reference.actual.notna()].tz_convert(timezone).date.max()
    first_day = last_day - timedelta(days=364)
    index = pd.date_range(pd.Timestamp(first_day, tz=timezone),
                          pd.Timestamp(last_day + timedelta(days=1), tz=timezone),
                          freq="h", inclusive="left").tz_convert("UTC")
    frames = {key: frame.reindex(index) for key, frame in frames.items()}
    if any(not np.isfinite(frame[["actual", "q50"]].to_numpy(float)).all() for frame in frames.values()):
        raise ValueError("The latest365 comparison requires complete candidate/incumbent observations and forecasts.")
    heat = features[HEAT_FRACTION].reindex(index)
    if not np.isfinite(heat).all():
        raise ValueError("Heatwave regime is absent on a Statistics hour.")
    dates = pd.Index(index.tz_convert(timezone).date)
    masks = {"annual": np.ones(len(index), dtype=bool), "probable_heatwave": (heat > 0).to_numpy(),
             "other_days": (heat <= 0).to_numpy(),
             "june_24_26_2026": np.array([str(value) in {"2026-06-24", "2026-06-25", "2026-06-26"} for value in dates])}
    storm = pd.Series(np.nan, index=index)
    benchmark = _attribute(prepared["kalman"], "statistics_benchmark")
    if isinstance(benchmark, pd.DataFrame):
        storm_frame = _timestamped(benchmark, name="Storm Statistics")
        storm = pd.to_numeric(storm_frame.q50.reindex(index), errors="raise")
        if np.isinf(storm).any():
            raise ValueError("Storm prices must not be infinite.")
        for item in prepared.values():
            other = _attribute(item, "statistics_benchmark")
            if not isinstance(other, pd.DataFrame) or not np.allclose(
                    _timestamped(other, name="Storm pairing").q50.reindex(index), storm,
                    rtol=0, atol=0, equal_nan=True):
                raise ValueError("Every model must use the same frozen Storm prices.")

    def gains(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
        return {key: baseline[key] - candidate[key] if candidate[key] is not None else None
                for key in ("mae_eur_mwh", "rmse_eur_mwh", "daily_mean_mae_eur_mwh")}

    variants = {}
    for variant in ("autonomous", "kalman"):
        variants[variant] = {}
        for name, mask in masks.items():
            candidate, incumbent = frames[variant].loc[mask], frames["incumbent"].loc[mask]
            native = {"candidate": _score(candidate.actual, candidate.q50, timezone=timezone),
                      "incumbent": _score(incumbent.actual, incumbent.q50, timezone=timezone)}
            native["gain_incumbent_minus_candidate"] = gains(native["candidate"], native["incumbent"])
            paired_mask = mask & np.isfinite(storm.to_numpy(float))
            observed = frames[variant].actual.loc[paired_mask]
            paired = {"candidate": _score(observed, frames[variant].q50.loc[paired_mask], timezone=timezone),
                      "incumbent": _score(observed, frames["incumbent"].q50.loc[paired_mask], timezone=timezone),
                      "storm": _score(observed, storm.loc[paired_mask], timezone=timezone)}
            paired["gain_incumbent_minus_candidate"] = gains(paired["candidate"], paired["incumbent"])
            paired["gain_storm_minus_candidate"] = gains(paired["candidate"], paired["storm"])
            variants[variant][name] = {"selected_days": sorted({str(value) for value in dates[mask]}),
                                      "full_support": native, "storm_paired": paired,
                                      "unpaired_storm_hours": int((mask & ~paired_mask).sum())}
    return {"schema_version": 1, "statistics_days": 365, "statistics_start_day": str(first_day),
            "statistics_end_day": str(last_day), "delivery_day_included": last_day == day.date(),
            "same_physical_hours": True, "same_observed_prices": True,
            "incumbent_label": incumbent_label, "variants": variants,
            "heat_partition": {"definition": f"heat_fraction_fcst > 0; at least one country with forecast heat streak >= {config.persistent_days}",
                               "persistent_days": config.persistent_days, "streak_clip_days": config.streak_clip_days,
                               "source": "pre-cutoff forecast features only", "uses_realised_temperature": False,
                               "uses_observed_price_or_error": False, "covers_every_annual_hour_once": True,
                               "probable_heatwave_days": len(set(dates[masks["probable_heatwave"]])),
                               "other_days": len(set(dates[masks["other_days"]]))},
            "storm_mask_policy": "identical finite candidate/incumbent/observed/Storm hours within each forecast-defined regime",
            "daily_mean_scope": "mean of exactly the same paired physical hours; a partial Storm day is not a full-day score",
            "june_24_26_2026_scope": "exploratory named historical episode; not a criterion for parameter selection",
            "interpretation": "Positive error gain indicates smaller error, not prospective validation or causal proof."}


def _banner(audit: Mapping[str, Any], comparison: Mapping[str, Any], *, variant: str) -> str:
    def number(value: Any) -> str:
        return "—" if value is None else f"{float(value):.3f}"

    config = audit["heatwave_feature_config"]
    labels = {"annual": "Année entière", "probable_heatwave": "Épisodes de chaleur probables",
              "other_days": "Autres journées", "june_24_26_2026": "24–26 juin 2026 (exploratoire)"}
    rows, paired_rows = [], []
    for key, label in labels.items():
        record = comparison["variants"][variant][key]
        native, paired = record["full_support"], record["storm_paired"]
        rows.append(f"<tr><th>{label}</th><td>{native['candidate']['days']}</td><td>{native['candidate']['hours']}</td>"
                    + "".join(f"<td>{number(value)}</td>" for value in
                              (native["candidate"]["mae_eur_mwh"], native["incumbent"]["mae_eur_mwh"],
                               native["gain_incumbent_minus_candidate"]["mae_eur_mwh"])) + "</tr>")
        paired_rows.append(f"<tr><th>{label}</th><td>{paired['candidate']['hours']}</td>" + "".join(
            f"<td>{number(paired[model]['mae_eur_mwh'])}</td>" for model in ("candidate", "incumbent", "storm")) + "</tr>")
    return (
        '<section data-report-section="heatwave-methodology" data-diagnostic-only="true" data-production="false">'
        '<h2>Températures Europe et épisodes de chaleur — expérimentation séparée</h2>'
        '<p>Les processus opérationnels ne sont pas modifiés. FR, DE, BE, NL et ES : '
        '<strong>indices journaliers de température prévus en °C</strong>, diffusés sur les heures physiques de la journée. '
        'Ce ne sont ni des températures horaires, ni des Tmax, ni des minima nocturnes. '
        'Ils ne permettent pas à eux seuls de certifier une canicule météorologique.</p>'
        '<p>Excès de chaleur prévu : dépassement du maximum entre un plancher par pays '
        'et un quantile historique calculé uniquement sur les jours antérieurs. '
        f'La persistance compte les journées de prévision successives au-dessus de ce seuil, plafonnée à {config["streak_clip_days"]}. '
        f'Un épisode de chaleur probable signifie qu’au moins un des cinq pays atteint {config["persistent_days"]} journées. '
        'Cette définition est fixée avant examen des erreurs ; les journées restantes sont conservées. '
        'Les paramètres réellement utilisés sont consignés dans l’audit.</p>'
        '<p>Cutoff : D−1 à 08 h civile. Les sources sont interrogées as-of ce cutoff ; '
        'le fournisseur ne restitue pas nécessairement le timestamp de publication originel. '
        'Cette traçabilité ne constitue pas une certification PIT indépendante. '
        'Aucune température réalisée ni Storm ne sert à définir les épisodes ou à entraîner le modèle.</p>'
        f'<p>Statistics : {html.escape(comparison["statistics_start_day"])} au '
        f'{html.escape(comparison["statistics_end_day"])} — 365 derniers jours observés, warm-up exclu. '
        'La livraison sans observation reste vide. Les P10/P50/P90 sont conservés ; '
        'les déciles intermédiaires et le CRPS sont des approximations d’affichage. '
        'Une performance historique ne prouve pas une amélioration prospective.</p></section>'
        '<section data-report-section="heatwave-regime-comparison"><h2>Performance annuelle et épisodes de chaleur</h2>'
        f'<p>Modèle de référence figé : {html.escape(comparison["incumbent_label"])}. '
        'MAE horaire en EUR/MWh. Gain positif = erreur du candidat plus faible. '
        'Le tableau correspond au modèle affiché dans ce rapport.</p>'
        '<div class="table-wrap"><table><thead><tr><th>Période</th><th>Jours</th><th>Heures</th>'
        '<th>Candidat</th><th>Référence</th><th>Gain</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
        '<h3>Comparaison à trois sur les seules heures communes avec Storm</h3>'
        '<p>Les trous de Storm ne sont jamais interpolés. Les trois modèles utilisent exactement les mêmes '
        'heures dans chaque ligne ; un jour partiellement couvert ne représente pas un prix moyen journalier complet.</p>'
        '<div class="table-wrap"><table><thead><tr><th>Période</th><th>Heures communes</th>'
        '<th>Candidat</th><th>Référence</th><th>Storm</th></tr></thead><tbody>' + "".join(paired_rows) + '</tbody></table></div>'
        '<p>Les 24–26 juin 2026 sont un diagnostic exploratoire explicite, non un sous-échantillon sélectionné '
        'sur les performances ni une période de calibration des seuils.</p>'
        '<details><summary>RMSE, biais, erreurs de prix moyens et audit comparatif</summary><pre>'
        + html.escape(_json(comparison)) + '</pre></details><details><summary>Sources et méthode du candidat</summary><pre>'
        + html.escape(_json(audit)) + '</pre></details></section>'
    )


def render_heatwave_reports(
    result: Any, *, incumbent: Any, data: ZoneData, zone: str, delivery_day: str,
    output_directory: str | Path, source_audit: Mapping[str, Any],
    storm_archive: str | Path | None = None, observed_source_audit: Mapping[str, Any] | None = None,
    attribution_directory: str | Path | None = None, feature_audit: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Render independent autonomous/Kalman reports from frozen engine results."""
    zone = str(zone).strip().upper()
    date = pd.Timestamp(delivery_day)
    if zone != data.zone.upper() or pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError("Report zone or civil delivery day is invalid.")
    day = date.date()
    start = day - timedelta(days=365)
    history_index = pd.date_range(pd.Timestamp(start, tz=data.timezone), pd.Timestamp(day, tz=data.timezone),
                                 freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    future_index = local_delivery_day_index(day, timezone=data.timezone).rename("delivery_start_utc")
    residual = _timestamped(_attribute(result, "residual_statistics"), name="residual Statistics")
    future = _timestamped(_attribute(result, "source_forecast"), name="source forecast")
    history = residual.loc[(residual.index >= history_index[0]) & (residual.index <= history_index[-1])]
    if not history.index.equals(history_index) or not future.index.equals(future_index):
        raise ValueError("Heatwave FINAL365/delivery must contain all physical hours exactly once.")
    if not np.isfinite(_actual_values(history, name="residual Statistics")).all():
        raise ValueError("Historical observed prices must be complete.")
    kh, kf = _view_frames(_attribute(result, "kalman_view"), history_index, future_index, name="Heatwave Kalman")
    ih, iff = _view_frames(_attribute(incumbent, "kalman_view", incumbent), history_index, future_index,
                          name="incumbent", allow_prefix=True)
    if not np.allclose(_actual_values(kh, name="Heatwave Kalman"), history.actual, rtol=0, atol=1e-9):
        raise ValueError("Heatwave autonomous/Kalman observations disagree.")
    upstream = [f"residual_corrected__q{q}" for q in (10, 50, 90)]
    if not set(upstream).issubset(kf) or not set(upstream).issubset(future) or not np.allclose(
            kf[upstream], future[upstream], rtol=0, atol=1e-9):
        raise ValueError("Kalman upstream differs from the explained autonomous forecast.")
    engine_audit = dict(_attribute(result, "audit", {}) or {})
    supplied_config = dict(feature_audit or {}).get("feature_config")
    engine_config = engine_audit.get("heatwave_feature_config")
    config = HeatwaveFeatureConfig.from_mapping(engine_config if engine_config is not None else supplied_config)
    if supplied_config is not None and HeatwaveFeatureConfig.from_mapping(supplied_config).to_dict() != config.to_dict():
        raise ValueError("The feature audit and fitted engine disagree on heatwave parameters.")
    covariates = _attribute(result, "covariates")
    features = _feature_frame(covariates, history_index.append(future_index), data.timezone, config)
    actuals, observation_audit = _observations(residual, history, ih, future_index, data, zone, str(day), observed_source_audit)
    base_variant = engine_audit.get("heatwave_base_variant", source_audit.get("baseline_kind", "nuclear_fr"))
    if base_variant not in ("nuclear_fr", "nuclear_cwe"):
        raise ValueError("Heatwave reporting requires an explicit supported nuclear base variant.")
    nuclear_label = "nucléaire CWE" if base_variant == "nuclear_cwe" else "nucléaire FR"
    autonomous_label = f"Chronos-2 + {nuclear_label} + températures Europe + correcteur"
    kalman_label = autonomous_label + " + Kalman"
    incumbent_label = str(engine_audit.get("incumbent_label", source_audit.get(
        "incumbent_label", f"Chronos-2 + {nuclear_label} + correcteur + Kalman")))
    directory = Path(output_directory).expanduser().resolve()
    report_data = _report_data(data, covariates, config)
    specifications = [("autonomous", "residual_corrected", autonomous_label, history, future),
                      ("kalman", "residual_kalman", kalman_label, kh, kf),
                      ("incumbent", "residual_kalman", incumbent_label, ih, iff)]
    prepared = {key: _model_result(history=h, forecast=f, actuals=actuals, model=model, label=label,
                                   data=report_data, output_directory=directory)
                for key, model, label, h, f in specifications}
    storm_audit = {"status": "unavailable", "used_for_prediction": False}
    if storm_archive is not None:
        from .nuclear_report_benchmark import attach_nuclear_storm
        storm_audit = attach_nuclear_storm(prepared, Path(storm_archive), zone=zone, timezone=data.timezone)
    comparison = _regime_comparison(prepared, features, timezone=data.timezone, day=date,
                                    incumbent_label=incumbent_label, config=config)
    for item in prepared.values():
        item.statistics_scope_note = (
            "365 derniers jours observés, livraison non observée laissée vide ; mêmes heures physiques et prix "
            "canoniques pour le candidat températures et la référence figée. Warm-up exclu. "
            + ("Storm officiel vérifié sur les heures communes uniquement." if storm_audit.get("status") == "complete"
               else "Storm indisponible : aucun comparateur reconstruit."))
    attribution_status = "unavailable_no_matching_artifact_supplied"
    if attribution_directory is not None:
        from .reporting import _attach_variable_attribution
        _attach_variable_attribution(prepared["autonomous"], directory=Path(attribution_directory),
                                     native_model="residual_corrected", timezone=data.timezone)
        if hasattr(prepared["autonomous"], "variable_attribution"):
            required = list(features.columns)
            raw = _attribution(prepared["autonomous"].variable_attribution, required=required,
                               label=autonomous_label, config=config)
            prepared["autonomous"].variable_attribution = raw
            prepared["kalman"].variable_attribution = {
                **raw, "scope": "upstream_model", "is_upstream_attribution": True,
                "explained_model": "residual_corrected", "explained_model_label": autonomous_label,
                "reported_model_label": kalman_label}
            attribution_status = "verified_new_forecast_artifact"
    raw_history = _timestamped(_attribute(result, "raw_history"), name="raw history")
    audit = {"schema_version": 1, "candidate_engine": "heatwave_forecast_v1", "zone": zone,
             "timezone": data.timezone, "delivery_day": str(day), "diagnostic_only": True,
             "production": False, "activation_performed": False, "operational_process_modified": False,
             "evaluation_start_day": str(start), "evaluation_end_day": str(day - timedelta(days=1)),
             "evaluation_days": 365, "evaluation_hours": len(history_index),
             "statistics_start_day": comparison["statistics_start_day"],
             "statistics_end_day": comparison["statistics_end_day"],
             "delivery_day_observed_hours": int(actuals.reindex(future_index).notna().sum()),
             "delivery_day_placeholder_hours": int(actuals.reindex(future_index).isna().sum()),
             "warmup": {"raw_days_before_final365": len(set(raw_history.index[raw_history.index < history_index[0]].tz_convert(data.timezone).date)),
                        "residual_days_before_final365": len(set(residual.index[residual.index < history_index[0]].tz_convert(data.timezone).date)),
                        "excluded_from_report_metrics": True},
             "temperature_countries": list(COUNTRIES), "temperature_semantics": "daily forecast indices broadcast to physical hours; not Tmax or nocturnal minima",
             "aggregate_precision_policy": "shared float32 half-ULP rounding bounds; heat cohort uses exact verified discrete country fraction; source values unchanged",
             "aggregate_precision_audit": features.attrs.get("aggregate_precision_audit"),
             "source_audit": dict(source_audit), "feature_audit": dict(feature_audit or {}),
             "heatwave_feature_config": config.to_dict(),
             "engine_audit": engine_audit, "historical_observations": observation_audit,
             "storm_used_as_input": False, "storm_hourly_comparison": dict(storm_audit),
             "variable_attribution_status": attribution_status, "incumbent_comparison": comparison}
    banners = {variant: _banner(audit, comparison, variant=variant) for variant in ("autonomous", "kalman")}
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant, label in (("autonomous", autonomous_label), ("kalman", kalman_label)):
        path = directory / f"forecast_{zone.lower()}_{day}_heatwave_{variant}.html"
        write_html_report([prepared[variant]], {"report": {"title": f"{zone} — {label}", "forecast_history_hours": 168}}, path)
        _replace_report_labels(path, native_label=label, baseline_label=None)
        document = path.read_text(encoding="utf-8")
        document = re.sub(r'<h3>Comparaison au modèle prix seul</h3>\s*(?:<div class="table-wrap">)?'
                          r'<table\b.*?</table>(?:</div>)?', "", document, count=1, flags=re.DOTALL)
        document = document.replace("Prévision opérationnelle du", "Prévision expérimentale du")
        document = document.replace("<main>", "<main>" + banners[variant], 1)
        path.write_text(document, encoding="utf-8")
        paths[variant] = path
    for key, filename, payload in (("audit", "heatwave_report_audit.json", audit),
                                   ("comparison", "heatwave_incumbent_comparison.json", comparison)):
        path = directory / filename
        path.write_text(_json(payload), encoding="utf-8")
        paths[key] = path
    return paths


__all__ = ["render_heatwave_reports", "HEATWAVE_AUTONOMOUS_LABEL", "HEATWAVE_KALMAN_LABEL"]
