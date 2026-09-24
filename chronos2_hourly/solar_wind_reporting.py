"""Small SolarWind adapter for the unchanged operational report engine; no fit."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .hourly_contract import local_delivery_day_index
from .nuclear_cwe_reporting import _attribute, _comparison, _observations, _view_frames
from .nuclear_reporting import _actual_values, _model_result, _report_zone_data, _timestamped
from .nuclear_forecast import NUCLEAR_ALIAS, NUCLEAR_KNOWN_COLUMN
from .solar_wind_forecast import ENGINE, GENERATION_ALIASES, GENERATION_KNOWN_COLUMNS, GENERATION_SERIES
from .reporting import _replace_report_labels
from chronos2_modular.report import write_html_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {"autonomous": "Chronos-2 + nucléaire FR + solaire CWE + éolien DE/NL + correcteur",
          "kalman": "Chronos-2 + nucléaire FR + solaire CWE + éolien DE/NL + correcteur + Kalman"}
INCUMBENT_LABELS = {"autonomous": "Chronos-2 + nucléaire FR + correcteur",
                    "kalman": "Chronos-2 + nucléaire FR + correcteur + Kalman"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)


def _destination(value):
    raw = Path(value).expanduser().absolute()
    allowed = PROJECT_ROOT.resolve() / "runs/experiments/solar_wind_v1"
    if allowed.resolve() != allowed or raw.resolve() != raw or not raw.is_relative_to(allowed):
        raise ValueError("SolarWind reports must stay in runs/experiments/solar_wind_v1 without path aliases.")
    return raw


def _inputs(result, data):
    """Verify adapter stage evidence and actual channels, not display labels."""
    audit = dict(_attribute(result, "audit", {}) or {})
    if audit.get("candidate_engine") != ENGINE or audit.get("candidate_variant") != "solar_wind_kalman":
        raise ValueError("SolarWind full-chain adapter identity is required.")
    known = set(GENERATION_KNOWN_COLUMNS)
    for key, required in (("solar_wind_chronos_context_columns", set(GENERATION_ALIASES)),
                          ("solar_wind_chronos_known_future_columns", known),
                          ("solar_wind_residual_features", known),
                          ("solar_wind_kalman_market_features", set(GENERATION_ALIASES))):
        if not required.issubset(audit.get(key, [])):
            raise ValueError(f"SolarWind adapter did not verify all six solar/wind inputs: {key}")
    if audit.get("baseline_inputs_preserved") is not True:
        raise ValueError("SolarWind adapter must preserve the nuclear FR baseline inputs.")
    for alias in GENERATION_ALIASES:
        source = audit.get("solar_wind_sources", {}).get(alias, {})
        if source != {"series": GENERATION_SERIES[alias], "unit": "GW", "semantic": "forecast_generation", "daily_broadcast": False}:
            raise ValueError(f"SolarWind requires hourly forecast generation in GW: {alias}")
    copied = _report_zone_data(data, _attribute(result, "covariates"))
    context, original = copied.model_context_covariates.copy(), data.model_context_covariates
    manifest = copied.input_manifest.copy(deep=True)
    if NUCLEAR_ALIAS not in context or NUCLEAR_KNOWN_COLUMN not in original or NUCLEAR_KNOWN_COLUMN not in data.known_future_columns:
        raise ValueError("SolarWind must retain the nuclear FR input and future channel.")
    for alias in GENERATION_ALIASES:
        channel = f"known_{alias}_oracle"
        if alias not in context or alias not in original or channel not in original or channel not in data.known_future_columns:
            raise ValueError(f"SolarWind context/future/Kalman channel missing: {alias}")
        values = np.column_stack([pd.to_numeric(source[column].reindex(context.index), errors="raise")
                                  for source, column in ((original, alias), (original, channel), (context, alias))])
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, values[:, [0]], rtol=0, atol=1e-9):
            raise ValueError(f"SolarWind forecast channels disagree or are incomplete: {alias}")
        mask = manifest.alias.astype(str).eq(alias)
        kind = "solar" if "_solar_" in alias else "wind"
        description = "solaire" if kind == "solar" else "éolienne"
        manifest.loc[mask, "source"] = f"Prévision {description} {alias[:2].upper()}"
        manifest.loc[mask, "source_zone"] = alias[:2].upper()
        manifest.loc[mask, "unit"] = "GW"
        manifest.loc[mask, "information_type"] = f"{kind}_generation_forecast"
    for channel in data.known_future_columns:
        if channel not in original:
            raise ValueError(f"Declared known-future channel missing: {channel}")
        context[channel] = original[channel].reindex(context.index)
    return replace(copied, input_manifest=manifest, model_context_covariates=context), audit


def _source_substitutions(source_audit):
    """Count actual physical substitutions once, not repeated audit references."""
    records = [source_audit]
    sources = source_audit.get("solar_wind", {})
    if not isinstance(sources, Mapping):
        raise ValueError("SolarWind source records must be a mapping.")
    records.extend(sources.values())
    unique = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("SolarWind source record must be a mapping.")
        nested = record.get("audit", {})
        if not isinstance(nested, Mapping):
            raise ValueError("SolarWind nested source audit must be a mapping.")
        for holder in (record, nested):
            substitutions = holder.get("source_substitutions", [])
            if not isinstance(substitutions, list):
                raise ValueError("SolarWind source_substitutions must be a list.")
            for entry in substitutions:
                if not isinstance(entry, Mapping) or not entry.get("alias") or not entry.get("value_time_utc"):
                    raise ValueError("SolarWind source substitution requires alias and physical hour.")
                timestamp = pd.Timestamp(entry["value_time_utc"])
                if pd.isna(timestamp) or timestamp.tzinfo is None:
                    raise ValueError("SolarWind source substitution requires an aware physical hour.")
                key = (str(entry["alias"]), timestamp.tz_convert("UTC").isoformat())
                if key in unique and unique[key] != dict(entry):
                    raise ValueError("SolarWind source substitution audits disagree for the same physical hour.")
                unique[key] = dict(entry)
    return [unique[key] for key in sorted(unique)]


def _banner(audit, comparison):
    selected_case = (
        '<p>Le cas du 22 septembre 2026 a été choisi après consultation du marché : '
        'ce test n’est pas une validation prospective indépendante.</p>'
        if audit["test_case_selected_after_market_observation"] else ""
    )
    wind_dst = (
        '<p>Attention DST éolien : politique duplicate pour les folds automnaux audités avec '
        'une seule valeur source. Cette valeur est recopiée sur les deux heures physiques UTC, '
        'y compris lorsqu’elle est non nulle. Cette convention peut biaiser les entrées et les scores ; '
        'elle ne reconstitue pas une deuxième observation indépendante. Les dates et la politique '
        'figurent dans l’audit source ; aucun autre trou ne doit être comblé par cette règle.</p>'
        if audit["source_audit"].get("wind_dst_policy") == "duplicate" else ""
    )
    substitutions = audit["source_substitutions"]
    wind_fallback = (
        '<section data-report-section="solar-wind-source-substitutions">'
        '<h3>Éolien NL : substitutions ECMWF limitées et auditées</h3>'
        f'<p>Substitutions effectives dans l’audit source : <strong>{len(substitutions)}</strong> '
        'heure(s) physique(s), comptées une seule fois malgré les références d’audit répétées. '
        'La courbe native NL est conservée, sauf les heures manquantes explicitement remplacées '
        'par ECMWF au même cutoff de requête D−1 à 08 h. La politique autorisée est limitée '
        'au 30 mars 2025 à 02:00 UTC et au 29 mars 2026 à 02:00 UTC ; le détail ci-dessous '
        'indique les substitutions réellement présentes dans ces sources, pas nécessairement '
        'dans la fenêtre annuelle évaluée.</p>'
        '<p>Il s’agit d’un remplacement de source, sans interpolation, et non d’une courbe '
        'native identique à toutes les heures. Les valeurs ECMWF en MW sont converties en GW '
        'avec le facteur 0,001. Cette expérimentation reste séparée, rétrospective et hors production ; '
        'un même as-of de requête ne certifie ni la publication originale ni la disponibilité PIT.</p>'
        '<details><summary>Heures, séries, valeurs et cutoffs des substitutions réelles</summary><pre>'
        + html.escape(_json(substitutions)) + '</pre></details></section>'
        if substitutions else ""
    )
    annual = comparison["annual"]
    rows = "".join("<tr><th>" + label + "</th>" + "".join(
        f"<td>{float(values[metric]):.3f}</td>" for values in
        (annual["incumbent"], annual["candidate"], annual["gain_incumbent_minus_candidate"])) + "</tr>"
        for metric, label in (("mae_eur_mwh", "MAE horaire"), ("rmse_eur_mwh", "RMSE horaire"),
                              ("daily_mean_mae_eur_mwh", "MAE des prix moyens journaliers")))
    return ('<section data-report-section="solar-wind-methodology" data-diagnostic-only="true" data-production="false">'
        '<h2>SolarWind — nucléaire FR + solaire CWE + éolien DE/NL</h2>'
        '<p>Chaîne complète expérimentale rétrospective as-of 08 h (D−1, heure locale), hors production. '
        'Cette exécution n’est pas une validation prospective. Le cutoff est un as-of de requête, '
        'pas une certification de la publication originale ni de la disponibilité réelle de chaque '
        'vintage à cet instant ; ces preuves doivent être auditées séparément.</p>'
        + selected_case + wind_dst + wind_fallback +
        '<p>Nucléaire FR conservé ; quatre prévisions solaires FR, DE, BE et NL et deux prévisions '
        'éoliennes DE et NL, toutes en GW dans Chronos-2, le correcteur résiduel et les covariables Kalman. '
        'Aucun entraînement ni modification de production par ce rapport ; ce n’est pas un détecteur ajouté après prévision.</p>'
        '<p>Mêmes 365 jours, heures physiques et observations pour chaque comparaison de même famille. '
        'Les observations indisponibles et les trous de Storm restent visibles ; Storm est un comparateur, jamais une entrée. '
        'Le backtest exclut la livraison ; Statistics conserve la fenêtre standard des 365 derniers jours observés.</p>'
        '<p>Les différences et contributions ne prouvent aucune causalité. P10/P50/P90 restent ceux des modèles ; '
        'les déciles intermédiaires et le CRPS sont des approximations d’affichage. Aucune promotion.</p>'
        '<h3>' + html.escape(comparison["candidate_label"] + " vs " + comparison["incumbent_label"]) + '</h3>'
        '<p>Gain positif = erreur réduite (EUR/MWh), indépendamment des trous de Storm.</p>'
        '<table><thead><tr><th>Métrique</th><th>Référence figée</th><th>SolarWind</th><th>Gain</th></tr></thead><tbody>'
        + rows + '</tbody></table><details><summary>Audit SolarWind</summary><pre>'
        + html.escape(_json(audit)) + '</pre></details></section>')


def render_solar_wind_reports(result, *, data, zone, delivery_day, output_directory,
                             incumbent, storm_archive, observed_source_audit, source_audit,
                             attribution_directory=None):
    """Render only autonomous/Kalman HTML, with frozen same-family incumbents."""
    directory = _destination(output_directory)
    zone, date = str(zone).upper(), pd.Timestamp(delivery_day)
    if zone != str(data.zone).upper() or pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError("A matching zone and a naive civil delivery date are required.")
    report_data, engine = _inputs(result, data)
    day, first = date.date(), date.date() - timedelta(days=365)
    hi = pd.date_range(pd.Timestamp(first, tz=data.timezone), pd.Timestamp(day, tz=data.timezone),
                       freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    fi = local_delivery_day_index(day, timezone=data.timezone).rename("delivery_start_utc")
    frames = {}
    for prefix, obj in (("", result), ("incumbent_", incumbent)):
        residual = _timestamped(_attribute(obj, "residual_statistics"), name=prefix + "residual_statistics")
        history = residual.loc[(residual.index >= hi[0]) & (residual.index <= hi[-1])]
        future = _timestamped(_attribute(obj, "source_forecast"), name=prefix + "source_forecast")
        if not history.index.equals(hi) or not future.index.equals(fi):
            raise ValueError("SolarWind/incumbent must share the complete FINAL365/delivery physical grid.")
        kh, kf = _view_frames(_attribute(obj, "kalman_view"), hi, fi, name=prefix + "Kalman", allow_prefix=bool(prefix))
        upstream = [f"residual_corrected__{q}" for q in ("q10", "q50", "q90")]
        if not np.allclose(_actual_values(history, name=prefix), _actual_values(kh, name=prefix), rtol=0, atol=1e-9):
            raise ValueError("Autonomous/Kalman historical observations differ.")
        for filtered, autonomous in ((kh, history), (kf, future)):
            if not set(upstream).issubset(filtered) or not set(upstream).issubset(autonomous) or not np.allclose(filtered[upstream], autonomous[upstream], rtol=0, atol=1e-9):
                raise ValueError("Kalman upstream differs from its autonomous forecast.")
        frames[prefix + "autonomous"], frames[prefix + "kalman"] = (history, future), (kh, kf)
    actuals, observation_audit = _observations(
        _timestamped(_attribute(result, "residual_statistics"), name="residual_statistics"),
        frames["autonomous"][0], frames["incumbent_autonomous"][0], fi, data, zone, str(day), observed_source_audit)
    prepared = {key: _model_result(history=h, forecast=f, actuals=actuals,
        model="residual_kalman" if key.endswith("kalman") else "residual_corrected",
        label=(INCUMBENT_LABELS if key.startswith("incumbent_") else LABELS)[key.removeprefix("incumbent_")],
        data=report_data, output_directory=directory) for key, (h, f) in frames.items()}
    comparisons = {}
    for family in LABELS:
        compared = _comparison(prepared[family], prepared["incumbent_" + family], day=date, timezone=data.timezone)
        compared.pop("june_24_26", None)  # No inherited case-specific post-hoc emphasis.
        comparisons[family] = {**compared, "candidate_model": "solar_wind_" + family,
            "incumbent_model": "nuclear_" + family, "candidate_label": LABELS[family],
            "incumbent_label": INCUMBENT_LABELS[family]}
    storm = {"status": "unavailable"}
    if storm_archive is not None:
        from .nuclear_report_benchmark import attach_nuclear_storm
        storm = attach_nuclear_storm(prepared, Path(storm_archive), zone=zone, timezone=data.timezone)
    for item in prepared.values():
        item.statistics_scope_note = "SolarWind : expérimental rétrospectif as-of 08 h, hors production, sans validation prospective ; mêmes 365 jours observés et observations ; livraison indisponible vide. Storm uniquement aux heures communes vérifiées ; trous conservés. Aucune promotion."
    attribution = "unavailable_requires_matching_solar_wind_forecast_artifact"
    if attribution_directory is not None:
        from .reporting import _attach_variable_attribution
        _attach_variable_attribution(prepared["autonomous"], directory=Path(attribution_directory), native_model="residual_corrected", timezone=data.timezone)
        raw = getattr(prepared["autonomous"], "variable_attribution", None)
        if raw is not None:
            for alias in GENERATION_ALIASES:
                groups = [g for g in raw.get("groups", []) if g.get("key") == alias]
                if len(groups) != 1 or any(not {alias, f"known_{alias}_oracle"}.intersection(groups[0].get(k, [])) for k in ("context_columns", "future_columns")):
                    raise ValueError("Attribution must cover all six solar/wind inputs of this forecast.")
            prepared["kalman"].variable_attribution = {**raw, "scope": "upstream_model", "is_upstream_attribution": True,
                "explained_model": "residual_corrected", "explained_model_label": LABELS["autonomous"], "reported_model_label": LABELS["kalman"]}
            attribution = "verified_upstream_artifact_not_total_Kalman_influence"
    audit = {"schema_version": 1, "candidate_engine": ENGINE, "candidate_variant": "solar_wind_kalman",
        "production": False, "diagnostic_only": True, "activation_performed": False, "model_fitted_by_report": False,
        "publication_mode": "experimental_retrospective_asof_08", "prospective_validation": False,
        "target_cutoff_local": "08:00", "target_cutoff_policy": "civil_D_minus_1_08:00",
        "original_publication_time_certified_by_report": False,
        "test_case_selected_after_market_observation": str(day) == "2026-09-22",
        "zone": zone, "delivery_day": str(day), "evaluation_start_day": str(first),
        "evaluation_end_day": str(day - timedelta(days=1)), "evaluation_days": 365, "evaluation_hours": len(hi),
        "historical_observations": observation_audit, "storm_hourly_comparison": storm, "storm_used_as_input": False,
        "generation_inputs": {alias: {"series": GENERATION_SERIES[alias], "unit": "GW", "kind": "generation_forecast",
            "technology": "solar" if "_solar_" in alias else "wind"} for alias in GENERATION_ALIASES},
        "variable_attribution_status": attribution, "engine_audit": engine, "source_audit": dict(source_audit or {}),
        "incumbent_comparisons": comparisons, "same_family_comparison": True}
    audit["source_substitutions"] = _source_substitutions(audit["source_audit"])
    audit["source_substitution_count"] = len(audit["source_substitutions"])
    banners = {family: _banner(audit, comparisons[family]) for family in LABELS}
    filenames = [f"forecast_{zone.lower()}_{day}_solar_wind_{family}.html" for family in LABELS]
    filenames += ["solar_wind_report_audit.json", "solar_wind_report_comparison.json"]
    if any((directory / name).resolve() != directory / name for name in filenames):
        raise ValueError("SolarWind report output files cannot be redirected.")
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for family, label in LABELS.items():
        path = directory / f"forecast_{zone.lower()}_{day}_solar_wind_{family}.html"
        if path.resolve() != path:
            raise ValueError("SolarWind report file cannot be redirected.")
        write_html_report([prepared[family]], {"report": {"title": f"{zone} — {label}", "forecast_history_hours": 168}}, path)
        _replace_report_labels(path, native_label=label, baseline_label=None)
        document = re.sub(r'<h3>Comparaison au modèle prix seul</h3>\s*(?:<div class="table-wrap">)?<table\b.*?</table>(?:</div>)?', "", path.read_text(encoding="utf-8"), count=1, flags=re.DOTALL)
        path.write_text(document.replace("Prévision opérationnelle du", "Prévision expérimentale rétrospective as-of 08 h du").replace("<main>", "<main>" + banners[family], 1), encoding="utf-8")
        paths[family] = path
    for key, payload in (("audit", audit), ("comparison", comparisons)):
        path = directory / f"solar_wind_report_{key}.json"
        if path.resolve() != path:
            raise ValueError("SolarWind report audit cannot be redirected.")
        path.write_text(_json(payload), encoding="utf-8")
        paths[key] = path
    return paths
