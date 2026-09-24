"""The operational HTML layout, with explicitly retrospective LoRA research data.

This adapter has no downloader, training, promotion or operational publisher.
It accepts already calculated predictions and never borrows incumbent metrics.
"""
from __future__ import annotations

from datetime import timedelta
import html
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_exogenous.prospective_auxiliary import RAW_MODEL, RESIDUAL_MODEL, KALMAN_MODEL, QUANTILES
from chronos2_hourly.reporting import _backtest_prediction_frame, _replace_report_labels
from chronos2_modular.common import ZoneData, ZoneRunResult
from chronos2_modular.metrics import compute_metrics, metric_breakdowns
from chronos2_modular.report import write_html_report


MODEL_LABELS = {
    RAW_MODEL: "Chronos-2 + LoRA rang 16 — référence brute de diagnostic",
    RESIDUAL_MODEL: "Chronos-2 + LoRA rang 16 + correcteur résiduel",
    KALMAN_MODEL: "Chronos-2 + LoRA rang 16 + correcteur résiduel + Kalman",
}
KNOWN_INPUT_COLUMNS = (
    "fr_residual_load_fcst", "de_residual_load_fcst", "be_residual_load_fcst",
    "nl_residual_load_fcst", "es_residual_load_fcst", "known_hour_sin",
    "known_hour_cos", "known_dow_sin", "known_dow_cos", "known_doy_sin",
    "known_doy_cos", "known_is_weekend", "local_temperature_fcst",
    "local_wind_generation_fcst", "local_solar_generation_fcst",
    "ttf_m1_eur_mwh_th", "eua_first_dec_eur_tco2", "ttf_change_1d",
    "ttf_change_5d", "eua_change_1d", "eua_change_5d", "fuel_volatility_20d",
    "ccgt_marginal_cost_eur_mwh", "flowbased_availability", "flowbased_hour_imputed",
    "flowbased_cnec_count", "flowbased_external_ram_p10_gw", "flowbased_ram_p10_gw",
    "flowbased_ram_headroom_p10_to_median_gw", "flowbased_low_ram_share",
    "flowbased_ram_to_fmax_p05", "flowbased_fr_neighbor_ptdf_spread_p90",
    "flowbased_fr_neighbor_ram_stress_p95_per_gw",
    "flowbased_core_ram_stress_p95_per_gw", "flowbased_stress_hhi",
)


class RollingResearchReportingError(ValueError):
    """A reporting input cannot support the claimed research comparison."""


def _utc_index(frame: pd.DataFrame, *, label: str) -> pd.DatetimeIndex:
    name = next((key for key in ("delivery_start_utc", "timestamp") if key in frame), None)
    values = frame[name] if name else frame.index
    if name is None and not isinstance(frame.index, pd.DatetimeIndex):
        raise RollingResearchReportingError(f"{label}: timestamp absent.")
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise RollingResearchReportingError(f"{label}: timeline non unique ou non ordonnée.")
    return index


def _complete_days(values: np.ndarray, local_days: pd.Index) -> np.ndarray:
    finite = pd.Series(np.isfinite(values), index=local_days)
    complete = finite.groupby(level=0).all()
    return np.asarray(complete.reindex(local_days), dtype=bool)


def _normalise_predictions(
    predictions: pd.DataFrame, delivery_day: str, timezone: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray, str]:
    day = pd.Timestamp(delivery_day).date()
    start_day = day - timedelta(days=364)
    expected = pd.date_range(
        pd.Timestamp(start_day, tz=timezone),
        pd.Timestamp(day + timedelta(days=1), tz=timezone), freq="h", inclusive="left",
    ).tz_convert("UTC")
    index = _utc_index(predictions, label="predictions")
    if not index.equals(expected):
        raise RollingResearchReportingError(
            "Les prédictions doivent couvrir exactement les 365 jours physiques "
            f"{start_day} → {day}, sans trou, extension ni doublon."
        )
    required = {"actual", "forecast_origin_utc", *(
        f"{model}__{quantile}" for model in MODEL_LABELS for quantile in QUANTILES
    )}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise RollingResearchReportingError(f"Prédictions incomplètes : {missing}.")
    frame = predictions.copy(deep=True).reset_index(drop=True)
    frame["delivery_start_utc"] = index
    origins = pd.DatetimeIndex(pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="raise"))
    days = pd.Index(index.tz_convert(timezone).date)
    expected_origins = pd.DatetimeIndex([
        pd.Timestamp(f"{day_value - timedelta(days=1)} 08:00", tz=timezone)
        for day_value in days
    ]).tz_convert("UTC")
    if not origins.equals(expected_origins):
        raise RollingResearchReportingError("Les origines doivent être J−1 à 08:00, heure locale.")
    frame["forecast_origin_utc"] = origins
    for model in MODEL_LABELS:
        columns = [f"{model}__{q}" for q in QUANTILES]
        values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or bool((np.diff(values, axis=1) < 0).any()):
            raise RollingResearchReportingError(f"{model}: quantiles non finis ou croisés.")
        frame[columns] = values
    actual = pd.to_numeric(frame["actual"], errors="raise").to_numpy(float)
    if np.isinf(actual).any():
        raise RollingResearchReportingError("Les prix observés ne peuvent pas être infinis.")
    complete = _complete_days(actual, days)
    if not complete.any():
        raise RollingResearchReportingError("Aucune journée observée complète à évaluer.")
    frame["actual"] = np.where(complete, actual, np.nan)
    return frame, index, complete, str(start_day)


def _zone_data(
    raw_inputs: pd.DataFrame, frame: pd.DataFrame, index: pd.DatetimeIndex,
    *, zone: str, timezone: str, delivery_day: str,
    history_target: pd.Series | None, metadata: Mapping[str, Any],
) -> ZoneData:
    input_index = _utc_index(raw_inputs, label="inputs")
    if len(index.difference(input_index)):
        raise RollingResearchReportingError("Les inputs doivent couvrir chaque heure de la fenêtre affichée.")
    missing = sorted(set(KNOWN_INPUT_COLUMNS).difference(raw_inputs.columns))
    if missing:
        raise RollingResearchReportingError(f"Les vrais inputs LoRA sont requis : {missing}.")
    covariates = raw_inputs.loc[:, KNOWN_INPUT_COLUMNS].apply(pd.to_numeric, errors="raise").copy()
    if np.isinf(covariates.to_numpy(float)).any():
        raise RollingResearchReportingError("Les inputs contiennent une valeur infinie.")
    covariates.index = input_index.tz_convert(timezone)
    day_start = pd.Timestamp(delivery_day, tz=timezone)
    if history_target is None:
        target = pd.Series(frame["actual"].to_numpy(float), index=index.tz_convert(timezone), name="target")
    else:
        target = history_target.copy(deep=True)
        if not isinstance(target.index, pd.DatetimeIndex):
            raise RollingResearchReportingError("history_target doit avoir un index temporel.")
        target.index = pd.DatetimeIndex(pd.to_datetime(target.index, utc=True)).tz_convert(timezone)
        if target.index.has_duplicates or not target.index.is_monotonic_increasing:
            raise RollingResearchReportingError("history_target contient des dates dupliquées ou désordonnées.")
        target = pd.to_numeric(target, errors="raise")
        if np.isinf(target.to_numpy(float)).any():
            raise RollingResearchReportingError("history_target contient un prix infini.")
    # The live-shaped panel displays earlier observations only, even on a retrospective run.
    target = target.loc[target.index < day_start].dropna()
    if target.empty:
        raise RollingResearchReportingError("Historique de prix antérieur à la livraison absent.")
    coverage = pd.DataFrame({"alias": KNOWN_INPUT_COLUMNS,
        "coverage_after_fill": np.isfinite(covariates.to_numpy(float)).mean(axis=0)})
    manifest = pd.DataFrame({"alias": KNOWN_INPUT_COLUMNS, "known_future": True,
        "source": "Panel LoRA fourni — aucune substitution incumbent",
        "usage": "Covariable du checkpoint rang 16"})
    return ZoneData(zone=zone, timezone=timezone, frequency="h", target=target,
        covariates=covariates, model_context_covariates=covariates.copy(),
        known_future_columns=list(KNOWN_INPUT_COLUMNS), coverage=coverage,
        input_manifest=manifest, diagnostics=dict(metadata))


def _storm_values(
    storm: pd.DataFrame | None, storm_contract: Mapping[str, Any] | None,
    *, index: pd.DatetimeIndex, actual: np.ndarray, timezone: str,
) -> tuple[pd.DataFrame | None, dict[str, Any], list[str]]:
    if storm is None:
        return None, {}, []
    if not isinstance(storm_contract, Mapping) or not storm_contract.get("report_label"):
        raise RollingResearchReportingError("Storm fourni exige un contrat explicite avec report_label.")
    if storm_contract.get("used_for_prediction") is True:
        raise RollingResearchReportingError("Storm doit être un comparateur uniquement.")
    storm_index = _utc_index(storm, label="Storm")
    if "q50" not in storm:
        raise RollingResearchReportingError("Storm: q50 absent.")
    supplied = pd.to_numeric(storm["q50"], errors="raise").to_numpy(float)
    if np.isinf(supplied).any():
        raise RollingResearchReportingError("Storm contient un prix infini.")
    values = pd.Series(supplied, index=storm_index).reindex(index).to_numpy(float)
    if "actual" in storm:
        labels = pd.Series(pd.to_numeric(storm["actual"], errors="raise").to_numpy(float), index=storm_index).reindex(index).to_numpy(float)
        compare = np.isfinite(labels) & np.isfinite(actual)
        if not np.allclose(labels[compare], actual[compare], rtol=0, atol=1e-9):
            raise RollingResearchReportingError("Les observations Storm divergent des prix du candidat.")
    days = pd.Index(index.tz_convert(timezone).date)
    available_by_day = pd.Series(np.isfinite(values), index=days).groupby(level=0).any()
    unavailable = [str(value) for value in available_by_day.index[~available_by_day]]
    result = pd.DataFrame({"timestamp": index.tz_convert(timezone), "actual": actual, "q50": values})
    return result, dict(storm_contract), unavailable


def _research_html(path: Path, *, label: str, baseline_label: str, scope: str, audit: Mapping[str, Any]) -> None:
    _replace_report_labels(path, native_label=label, baseline_label=baseline_label)
    document = path.read_text(encoding="utf-8")
    for old, new in (
        ("Prévision opérationnelle du", "Reconstitution rétrospective du"),
        ("Prévision Day-Ahead réelle", "Prévision Day-Ahead reconstruite — recherche"),
        ("prévision Day-Ahead opérationnelle", "prévision Day-Ahead reconstruite"),
    ):
        document = document.replace(old, new)
    note = (
        "Le checkpoint neuronal rang 16 est fixe et a été sélectionné rétrospectivement. "
        "Il n’est pas réentraîné chaque jour. Le préfixe antérieur de calibration recouvre "
        "la période d’entraînement/validation LoRA : il ne constitue pas une preuve OOF "
        "neuronale. Pour chacun des 365 jours évalués, le correcteur résiduel est ajusté sur "
        "les 365 jours antérieurs et le Kalman est recalibré sur les 365 jours antérieurs "
        "de prévisions corrigées causalement, sans utiliser le prix du jour à prédire. "
        "Dans le préfixe historique de calibration uniquement, la chauffe du correcteur "
        "commence par 30 jours sans correction (identité), puis utilise une fenêtre "
        "croissante jusqu’à 365 jours ; cette chauffe n’appartient pas à la fenêtre "
        "d’évaluation affichée. Ce rapport n’est ni un test prospectif "
        "indépendant, ni une validation de production ou de promotion. "
        "P10, P50 et P90 sont les quantiles calculés ; les autres déciles sont interpolés "
        "linéairement pour le rendu existant et le CRPS approximatif, sans recalibrage "
        "probabiliste. Attribution des variables : non recalculée pour ce candidat. "
        "Les diagnostics détaillés des états Kalman ne sont pas importés depuis un autre modèle."
    )
    snapshot_html = ""
    comparator_audit = audit.get("comparator_audit")
    if isinstance(comparator_audit, Mapping):
        generated = str(comparator_audit.get("snapshot_generated_text") or "non renseignée")
        extraction_values = comparator_audit.get("observation_extraction_timestamps")
        if isinstance(extraction_values, (list, tuple)):
            extracted = "; ".join(str(value) for value in extraction_values) or "non renseignées"
        else:
            extracted = str(extraction_values or "non renseignées")
        snapshot_html = (
            '<p id="research-comparator-snapshot"><strong>Comparateurs : snapshot exact du rapport publié.</strong> '
            f'Génération du rapport source : {html.escape(generated)}. '
            f'Extractions des observations : {html.escape(extracted)}. '
            'Les prix observés et Storm reproduisent ce snapshot publié, sans actualisation API '
            'lors de cette génération. Ils ne sont pas présentés comme la dernière observation '
            'actuellement disponible ; ils servent uniquement au reporting, jamais à la calibration '
            'ou aux entrées du modèle.</p>'
        )
    safe_audit = json.dumps(audit, ensure_ascii=False, allow_nan=False, default=str).replace("<", "\\u003c")
    banner = (
        '<section id="research-methodology" role="note" '
        'style="border:2px solid var(--statistics-warning-border,var(--border));background:var(--surface-soft)">'
        '<h2>Recherche rétrospective — LoRA rang 16</h2>'
        f'<p><strong>{html.escape(scope)}</strong></p><p>{html.escape(note)}</p>'
        + snapshot_html +
        '<p>Statut : diagnostic_only=true · production_pit_evidence=false · '
        'production_pipeline_evidence=false · promotion_eligible=false · neural_oof=false.</p>'
        '</section><script type="application/json" id="rolling-research-audit">'
        + safe_audit + '</script>'
    )
    if document.count("<main>") != 1:
        raise RollingResearchReportingError("Structure inattendue du renderer HTML partagé.")
    path.write_text(document.replace("<main>", "<main>" + banner, 1), encoding="utf-8")


def render_rolling_research_reports(
    predictions: pd.DataFrame, raw_inputs: pd.DataFrame, *, output_directory: Path,
    zone: str, delivery_day: str, timezone: str = "Europe/Paris",
    metadata: Mapping[str, Any] | None = None, storm: pd.DataFrame | None = None,
    storm_contract: Mapping[str, Any] | None = None, history_target: pd.Series | None = None,
) -> dict[str, Path]:
    """Render both chains using only the explicitly supplied fixed 365-day window.

    Missing observed days remain placeholders, without shifting to older dates.
    A partially missing Storm day uses identical supplied hours for both sides
    of the *paired* Statistics; missing whole days remain blank placeholders.
    Primary model metrics retain every complete observed day. No inputs or old
    reports are modified, and existing destination filenames are refused.
    """
    zone = str(zone).strip().upper()
    if zone not in {"FR", "DE", "BE", "NL"}:
        raise RollingResearchReportingError(f"Zone LoRA non supportée : {zone}.")
    frame, index, complete, start_day = _normalise_predictions(predictions, delivery_day, timezone)
    days = pd.Index(index.tz_convert(timezone).date)
    output_directory = Path(output_directory).expanduser().resolve()
    paths = {model: output_directory / f"forecast_{zone.lower()}_{delivery_day}_{model}.html"
        for model in (RESIDUAL_MODEL, KALMAN_MODEL)}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Un rapport existe déjà : choisir un nouveau répertoire de publication.")
    audit = dict(metadata or {})
    audit.update({"report_kind": "fixed_checkpoint_rolling365_retrospective_research",
        "diagnostic_only": True, "production_pit_evidence": False,
        "production_pipeline_evidence": False, "promotion_eligible": False,
        "activation_performed": False, "neural_oof": False,
        "zone": zone, "timezone": timezone, "evaluation_start_day": start_day,
        "evaluation_end_day": delivery_day, "calendar_days": 365, "physical_hours": len(index),
        "scored_days": int(len(days[complete].unique())), "scored_hours": int(complete.sum()),
        "pending_observation_days": [str(value) for value in days[~complete].unique()],
        "statistics_window_backshifted": False, "quantile_display_interpolation": "piecewise_linear_deciles",
        "variable_attribution_recomputed": False, "incumbent_predictions_used": False})
    data = _zone_data(raw_inputs, frame, index, zone=zone, timezone=timezone,
        delivery_day=delivery_day, history_target=history_target, metadata=audit)
    benchmark, contract, storm_missing = _storm_values(storm, storm_contract, index=index,
        actual=frame["actual"].to_numpy(float), timezone=timezone)
    audit["storm_unavailable_days"] = storm_missing
    audit["storm_used_as_input"] = False
    storm_missing_hours = int(benchmark["q50"].isna().sum()) if benchmark is not None else 0
    audit["storm_missing_hours"] = storm_missing_hours
    scope = (f"Fenêtre fixe de 365 jours : {start_day} → {delivery_day} ({len(index)} heures physiques). "
        f"{audit['scored_days']} journées observées complètes ; "
        f"{len(audit['pending_observation_days'])} journées en attente, sans décalage de la fenêtre.")
    if storm_missing_hours:
        scope += (f" Storm incomplet : {storm_missing_hours} heures absentes, dont "
            f"{len(storm_missing)} journées entières indisponibles. Les moyennes et métriques "
            "comparées utilisent exactement les mêmes heures disponibles des deux côtés ; "
            "les prix Storm ne sont ni interpolés ni remplacés.")
    if benchmark is None:
        scope += " Storm non fourni : comparaison Storm indisponible."
    all_rows = np.ones(len(frame), dtype=bool)
    model_frames = {model: _backtest_prediction_frame(frame, model=model,
        row_mask=all_rows, timezone=timezone, allow_missing_actual=True) for model in MODEL_LABELS}
    today = days == pd.Timestamp(delivery_day).date()
    output_directory.mkdir(parents=True, exist_ok=True)
    for model, baseline_model in ((RESIDUAL_MODEL, RAW_MODEL), (KALMAN_MODEL, RESIDUAL_MODEL)):
        native, baseline = model_frames[model], model_frames[baseline_model]
        scored_native, scored_baseline = native.loc[complete].copy(), baseline.loc[complete].copy()
        by_horizon, by_hour = metric_breakdowns(scored_native, scored_baseline, 150.0)
        result = ZoneRunResult(zone=zone, metrics_native=compute_metrics(scored_native, 150.0),
            metrics_baseline=compute_metrics(scored_baseline, 150.0),
            backtest_native=native.copy(), backtest_baseline=baseline.copy(),
            metrics_by_horizon=by_horizon, metrics_by_hour=by_hour,
            forecast_native=native.loc[today].copy(), forecast_baseline=baseline.loc[today].copy(),
            zone_data=data, output_dir=output_directory)
        result.statistics_candidate = native.copy()
        result.statistics_candidate_label = MODEL_LABELS[model]
        result.statistics_scope_note = scope
        if benchmark is not None and bool((np.isfinite(benchmark["q50"]) & complete).any()):
            pairing_complete = np.isfinite(benchmark["q50"].to_numpy(float))
            wholly_missing = np.asarray([str(value) in storm_missing for value in days])
            # Partial DST coverage is genuinely paired hour-for-hour. Whole
            # missing days retain their date as an unavailable placeholder.
            kept = pairing_complete | wholly_missing | ~complete
            result.statistics_candidate = native.loc[kept].copy()
            result.statistics_candidate.loc[wholly_missing[kept], "actual"] = np.nan
            result.statistics_benchmark = benchmark.loc[kept].copy()
            result.statistics_benchmark.loc[wholly_missing[kept], "actual"] = np.nan
            result.statistics_benchmark_label = str(contract["report_label"])
            result.statistics_benchmark_contract = contract
            if pairing_complete[today].all():
                result.forecast_benchmark = benchmark.loc[today, ["timestamp", "q50"]].copy()
                result.forecast_benchmark_label = str(contract["report_label"])
        write_html_report([result], {"report": {"title":
            f"Forecast {zone} {delivery_day} — {MODEL_LABELS[model]} — recherche rétrospective",
            "forecast_history_hours": 168}}, paths[model])
        _research_html(paths[model], label=MODEL_LABELS[model],
            baseline_label=MODEL_LABELS[baseline_model], scope=scope, audit=audit)
    return paths
