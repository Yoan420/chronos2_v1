"""Nuclear-input reports using the established hourly/Storm HTML renderer.

This adapter does no training, source fetching, or benchmark reconstruction.
An optional audited Storm snapshot supports every standard report comparison;
without one, the report explicitly leaves the comparator unavailable.
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
from chronos2_hourly.reporting import (
    _backtest_prediction_frame,
    _forecast_prediction_frame,
    _replace_report_labels,
)
from chronos2_modular.common import ZoneData, ZoneRunResult
from chronos2_modular.metrics import compute_metrics, metric_breakdowns
from chronos2_modular.report import write_html_report


NUCLEAR_MODEL_LABEL = "Chronos-2 + nucl. + correcteur"
NUCLEAR_KALMAN_LABEL = f"{NUCLEAR_MODEL_LABEL} + Kalman"
_QUANTILES = ("q10", "q50", "q90")


def _timestamped(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame.")
    result = frame.copy(deep=True)
    if "delivery_start_utc" in result:
        timestamps = result.pop("delivery_start_utc")
    elif "timestamp" in result:
        timestamps = result.pop("timestamp")
    elif isinstance(result.index, pd.DatetimeIndex):
        timestamps = result.index
    else:
        raise ValueError(f"{name}: delivery_start_utc or timestamp is required.")
    result.index = pd.DatetimeIndex(
        pd.to_datetime(timestamps, utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if result.index.hasnans or result.index.has_duplicates:
        raise ValueError(f"{name}: missing or duplicate physical timestamps.")
    return result.sort_index()


def _require_quantiles(frame: pd.DataFrame, model: str, *, name: str) -> None:
    columns = [f"{model}__{quantile}" for quantile in _QUANTILES]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing model quantiles {missing}.")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: model quantiles must be finite on every hour.")
    if (np.diff(values, axis=1) < 0).any():
        raise ValueError(f"{name}: quantile crossings are not reportable.")


def _actual_values(frame: pd.DataFrame, *, name: str) -> np.ndarray:
    if "actual" not in frame:
        raise ValueError(f"{name}: actual is required.")
    return pd.to_numeric(frame["actual"], errors="coerce").to_numpy(float)


def _report_zone_data(data: ZoneData, covariates: pd.DataFrame) -> ZoneData:
    context = _timestamped(covariates, name="covariates")
    context.index = context.index.tz_convert(data.timezone)
    context.index.name = "timestamp"
    context = context.apply(pd.to_numeric, errors="coerce")
    target = data.target.copy(deep=True)
    # Fresh canonical snapshots use UTC, whereas the shared HTML renderer
    # derives civil-day Statistics/hour bins from the target's index timezone.
    # Normalize the reporting copy, not its values or the frozen engine data.
    target.index = pd.DatetimeIndex(target.index).tz_convert(data.timezone)
    historical = context.reindex(target.index)
    coverage = pd.DataFrame(
        {
            "alias": context.columns,
            "coverage": context.notna().mean().to_numpy(),
            "coverage_after_fill": context.notna().mean().to_numpy(),
        }
    )
    manifest = data.input_manifest.copy(deep=True)
    if "alias" in manifest:
        present = set(manifest["alias"].astype(str))
        extra = [column for column in context if column not in present]
        if extra:
            manifest = pd.concat(
                [manifest, pd.DataFrame({"alias": extra, "source": "nuclear replay"})],
                ignore_index=True,
            )
        manifest = manifest.loc[manifest["alias"].astype(str).isin(context.columns)]
    else:
        manifest = pd.DataFrame({"alias": context.columns})
    return replace(
        data,
        target=target,
        covariates=historical,
        model_context_covariates=context,
        coverage=coverage,
        input_manifest=manifest,
        known_future_columns=list(data.known_future_columns),
        diagnostics={**data.diagnostics, "diagnostic_only": True, "production": False},
    )


def _model_result(
    *,
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    actuals: pd.Series,
    model: str,
    label: str,
    data: ZoneData,
    output_directory: Path,
) -> ZoneRunResult:
    history = history.copy(deep=True)
    forecast = forecast.copy(deep=True)
    _require_quantiles(history, model, name="FINAL365 history")
    _require_quantiles(forecast, model, name="delivery forecast")
    history["actual"] = actuals.reindex(history.index).to_numpy()
    forecast["actual"] = actuals.reindex(forecast.index).to_numpy()
    raw_history = history.reset_index()
    raw_statistics = pd.concat([history, forecast]).reset_index()
    native = _backtest_prediction_frame(
        raw_history,
        model=model,
        row_mask=np.ones(len(raw_history), dtype=bool),
        timezone=data.timezone,
    )
    statistics = _backtest_prediction_frame(
        raw_statistics,
        model=model,
        row_mask=np.ones(len(raw_statistics), dtype=bool),
        timezone=data.timezone,
        allow_missing_actual=True,
    )
    by_horizon, by_hour = metric_breakdowns(native, None, 150.0)
    result = ZoneRunResult(
        zone=data.zone,
        metrics_native=compute_metrics(native, 150.0),
        metrics_baseline=None,
        backtest_native=native,
        backtest_baseline=None,
        metrics_by_horizon=by_horizon,
        metrics_by_hour=by_hour,
        forecast_native=_forecast_prediction_frame(
            forecast.reset_index(), model=model, timezone=data.timezone
        ),
        forecast_baseline=None,
        zone_data=data,
        output_dir=output_directory,
    )
    result.statistics_candidate = statistics
    from chronos2_modular.forecast_explanation import attach_forecast_components
    attach_forecast_components(result, forecast.reset_index(), model=model)
    result.statistics_candidate_label = label
    result.statistics_scope_note = (
        "Diagnostic uniquement. FINAL365 commun aux variantes; les observations "
        "du jour de livraison restent vides tant qu'elles sont indisponibles. "
        "Storm indisponible: aucun comparateur vérifié n'a été fourni."
    )
    return result


def _diagnostic_banner(audit: Mapping[str, Any], *, kalman: bool) -> str:
    warmup = audit["warmup"]
    kalman_note = (
        " Le Kalman reçoit le correcteur nucléaire recalculé; son warm-up "
        "précède le FINAL365 et n'est pas inclus dans les scores affichés."
        if kalman
        else ""
    )
    storm_note = (
        "Les Statistics, prix moyens, calendrier et profils horaires comparent le modèle "
        "à un snapshot Storm officiel vérifié, uniquement sur les heures observées communes. "
        "Les jours sans Storm restent visibles, sans inventer de comparaison."
        if audit.get("storm_hourly_comparison", {}).get("status") == "complete"
        else "Storm indisponible: aucune série ou statistique Storm n'est reconstruite."
    )
    integrated = audit.get("publication_mode") == "forecast_run"
    heading = "Variante nucléaire" if integrated else "Variante nucléaire — diagnostic uniquement"
    qualification = (
        "Variante intégrée au lancement Forecast; ses sorties restent distinctes des modèles historiques. "
        "Cette publication ne constitue pas à elle seule une validation prospective en production. "
        "Aucun entraînement n'est effectué par le rapport."
        if integrated else
        "<strong>diagnostic_only=true · production=false</strong>. "
        "Ce rapport ne constitue ni une validation en production ni une preuve "
        "d'amélioration. Aucun entraînement n'est effectué par le rapport."
    )
    rendered = (
        '<section data-report-section="nuclear-diagnostic" '
        'data-diagnostic-only="true" data-production="false">'
        f"<h2>{heading}</h2><p>{qualification}</p>"
        f"<p>Scores FINAL365: {html.escape(str(audit['evaluation_start_day']))} "
        f"au {html.escape(str(audit['evaluation_end_day']))}, "
        f"{int(audit['evaluation_hours'])} heures physiques observées. "
        "Le jour de livraison n'entre pas dans ces scores de backtest.</p>"
        f"<p>Warm-up: {int(warmup['residual_days_before_final365'])} jours "
        "de sorties du correcteur avant FINAL365; "
        f"{int(warmup['raw_days_before_final365'])} jours de sorties Chronos-2 "
        f"avant FINAL365.{kalman_note}</p>"
        "<p>Statistics: 365 jours observés glissants, plus le jour de livraison "
        "s'il est encore sans observations; mêmes heures physiques et mêmes "
        "prix observés pour les deux variantes. " + storm_note + "</p>"
        "<p>DST: toute duplication de l'heure d'automne concerne uniquement "
        "la covariable nucléaire dérivée lorsque sa source l'exige; elle ne "
        "crée ni prix observé ni prévision supplémentaire. Les heures physiques "
        "distinctes restent identifiées en UTC. La traçabilité des vintages "
        "d'entrée reste soumise à l'audit source.</p>"
        "<p>P10/P50/P90 sont conservés; les déciles intermédiaires et le CRPS "
        "du moteur de rapport sont une approximation linéaire d'affichage.</p>"
        "<details><summary>Audit du rapport</summary><pre>"
        + html.escape(json.dumps(dict(audit), ensure_ascii=False, indent=2, default=str))
        + "</pre></details></section>"
    )
    if integrated:
        rendered = (
            '<details data-report-section="nuclear-methodology">'
            '<summary>Variante nucléaire — méthode et traçabilité</summary>'
            + rendered + "</details>"
        )
    reference = audit.get("actual_price_reference", {})
    label = reference.get("actual_reference_label")
    if label in ("EPEX", "ENTSO-E", "ENTSO-E + EPEX"):
        extraction = reference.get("extracted_at_utc")
        extraction_note = (
            " Extraction du snapshot : "
            + pd.Timestamp(extraction).tz_convert(audit["timezone"]).strftime("%d/%m/%Y %H:%M %Z")
            + "." if extraction else ""
        )
        provider_note = " via Saturn" if reference.get("policy") == "epex_only_v1" else ""
        rendered = (
            '<section data-report-section="actual-price-reference">'
            '<p><strong>Prix réalisés et référence des scores : '
            + html.escape(label + provider_note) + '.</strong> '
            'Les observations affichées et les métriques NYX/Storm utilisent les mêmes prix horaires '
            'du snapshot vérifié, y compris pour les dates historiques.'
            + html.escape(extraction_note) + '</p></section>' + rendered
        )
    if audit.get("delivery_day_actual_reason") == "post_auction_source_rejected":
        # This is a current-day validation failure, not a missing publication.
        # Keep it visible even when operational methodology is collapsed.
        extracted = audit.get("delivery_day_actual_extracted_at_utc")
        extraction_note = (
            '<p>Dernière extraction des sources : '
            + html.escape(pd.Timestamp(extracted).tz_convert(audit["timezone"]).strftime("%d/%m/%Y %H:%M %Z"))
            + '.</p>' if extracted else ''
        )
        rendered = (
            '<section data-report-section="observed-price-validation" '
            'data-validation-status="rejected_divergence" role="status">'
            '<p><strong>Prix réalisés du jour non validés : les sources sont en désaccord. '
            'Prévisions disponibles ; scores du jour non calculés.</strong></p>'
            + extraction_note + '</section>'
            + rendered
        )
    return rendered


def render_nuclear_reports(
    result: Any,
    *,
    data: ZoneData,
    zone: str,
    delivery_day: str,
    output_directory: str | Path,
    include_kalman: bool = True,
    report_variants: tuple[str, ...] | None = None,
    source_audit: Mapping[str, Any] | None = None,
    storm_archive: str | Path | None = None,
    attribution_directory: str | Path | None = None,
    observed_source_audit: Mapping[str, Any] | None = None,
    operational_layout: bool = False,
) -> dict[str, Path]:
    """Render isolated nuclear autonomous/Kalman reports, never incumbents.

    ``result`` is the nuclear engine result (raw_history, residual_statistics,
    source_forecast, covariates, optional kalman_view, and audit). Both reports
    score the same 365 complete civil days before delivery. Statistics also
    retain the delivery forecast without inventing an observation; when that
    day's observations exist, the standard 365-observed-day window applies.
    Neither source frames nor the supplied ZoneData are mutated.
    ``operational_layout`` only selects the integrated presentation; it does
    not promote or reseal a model. Canonical observation revisions are accepted
    only with ``observed_source_audit`` verified against their refresh artifact.
    """
    zone_key = str(zone).strip().upper()
    selected = tuple(dict.fromkeys(report_variants)) if report_variants is not None else (
        ("autonomous", "kalman") if include_kalman else ("autonomous",))
    if not selected or set(selected) - {"autonomous", "kalman"}:
        raise ValueError("Nuclear report variants must be autonomous and/or kalman.")
    if "kalman" in selected and not include_kalman:
        raise ValueError("A Kalman report requires include_kalman=True.")
    if zone_key != str(data.zone).strip().upper():
        raise ValueError("Report zone does not match ZoneData.zone.")
    day = pd.Timestamp(delivery_day).date()
    start_day = day - timedelta(days=365)
    expected_history = pd.date_range(
        pd.Timestamp(start_day, tz=data.timezone),
        pd.Timestamp(day, tz=data.timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC").rename("delivery_start_utc")
    expected_future = local_delivery_day_index(day, timezone=data.timezone).rename(
        "delivery_start_utc"
    )
    residual = _timestamped(result.residual_statistics, name="residual_statistics")
    forecast = _timestamped(result.source_forecast, name="source_forecast")
    if not forecast.index.equals(expected_future):
        raise ValueError("Nuclear forecast must cover exactly the delivery day.")
    historical = residual.loc[
        (residual.index >= expected_history[0]) & (residual.index < expected_future[0])
    ]
    if not historical.index.equals(expected_history):
        raise ValueError("Nuclear FINAL365 must contain every physical hour.")
    history_actual = _actual_values(historical, name="residual_statistics")
    if not np.isfinite(history_actual).all():
        raise ValueError("Nuclear FINAL365 observations must be complete.")
    actuals = pd.Series(history_actual, index=expected_history, name="actual")

    # Use only supplied observations, after the engine has frozen its forecast.
    # A historical rerun may have actual(D); the usual live run does not.
    delivery_actual = pd.Series(np.nan, index=expected_future, name="actual")
    if "actual" in residual:
        delivery_actual = pd.to_numeric(
            residual["actual"].reindex(expected_future), errors="coerce"
        )
    target = data.target.copy()
    target.index = pd.DatetimeIndex(pd.to_datetime(target.index, utc=True))
    if target.index.has_duplicates:
        raise ValueError("ZoneData.target contains duplicate physical timestamps.")
    refreshed_observations = None
    if observed_source_audit is not None:
        from .nuclear_reporting_refresh import verify_refreshed_observations
        refreshed_observations = verify_refreshed_observations(
            target, observed_source_audit, zone=zone_key, timezone=data.timezone,
            delivery_day=str(day),
        )
    target_history = pd.to_numeric(target.reindex(expected_history), errors="coerce")
    comparable_history = np.isfinite(target_history.to_numpy(float))
    from .observation_precision import validate_observation_precision
    try:
        actual_precision = ({
            "mode": "verified_latest_canonical_reporting_observations",
            "compared_hours": int(comparable_history.sum()),
            "changed_hours": int((np.abs(history_actual[comparable_history]
                - target_history.to_numpy(float)[comparable_history]) > 1e-9).sum()),
            "max_absolute_revision_eur_mwh": float(np.max(np.abs(
                history_actual[comparable_history] - target_history.to_numpy(float)[comparable_history]
            ))) if comparable_history.any() else 0.0,
            "training_inputs_modified": False,
            "source": refreshed_observations,
        } if refreshed_observations is not None else validate_observation_precision(
            history_actual[comparable_history],
            target_history.to_numpy(float)[comparable_history], name="FINAL365 observations",
        ) if comparable_history.any() else {
            "mode": "no_current_historical_observations", "compared_hours": 0,
            "inputs_modified": False,
        })
    except ValueError as error:
        raise ValueError("FINAL365 observations disagree with ZoneData.target.") from error
    # The replay's labels remain untouched. Only report scores use the exact
    # supplied canonical decimals, also used by the paired Storm comparison.
    actuals = pd.Series(
        np.where(comparable_history, target_history.to_numpy(float), history_actual),
        index=expected_history, name="actual",
    )
    target_actual = pd.to_numeric(target.reindex(expected_future), errors="coerce")
    comparable = delivery_actual.notna() & target_actual.notna()
    if refreshed_observations is None and not np.allclose(
        delivery_actual.loc[comparable], target_actual.loc[comparable], rtol=0, atol=1e-9
    ):
        raise ValueError("Delivery-day observations disagree between source frames.")
    delivery_actual = (target_actual.copy() if refreshed_observations is not None
                       else delivery_actual.combine_first(target_actual))
    delivery_finite = np.isfinite(delivery_actual.to_numpy(float))
    if delivery_finite.any() and not delivery_finite.all():
        raise ValueError("Delivery-day observations must be complete or unavailable.")
    delivery_actual.loc[~delivery_finite] = np.nan
    actuals = pd.concat([actuals, delivery_actual])
    delivery_actual_reason = None
    if refreshed_observations is not None and not delivery_finite.any():
        source = observed_source_audit.get("observed", {}).get("source", {})
        if source.get("current_delivery_actual_reason") == "post_auction_source_rejected":
            delivery_actual_reason = "post_auction_source_rejected"

    directory = Path(output_directory).expanduser().resolve()
    report_data = _report_zone_data(data, result.covariates)
    specifications = ([("autonomous", "residual_corrected", NUCLEAR_MODEL_LABEL, historical, forecast)]
                      if "autonomous" in selected else [])
    if "kalman" in selected:
        view = getattr(result, "kalman_view", None)
        if view is None:
            raise ValueError("include_kalman=True requires the recalculated nuclear Kalman view.")
        kalman_history = _timestamped(view.backtest, name="kalman_view.backtest")
        kalman_forecast = _timestamped(view.forecast, name="kalman_view.forecast")
        if not kalman_history.index.equals(expected_history):
            raise ValueError("Kalman and autonomous FINAL365 physical hours differ.")
        if not kalman_forecast.index.equals(expected_future):
            raise ValueError("Kalman and autonomous delivery-day physical hours differ.")
        upstream_columns = [f"residual_corrected__{q}" for q in _QUANTILES]
        if not set(upstream_columns).issubset(kalman_forecast) or not np.allclose(
            kalman_forecast[upstream_columns].to_numpy(float),
            forecast[upstream_columns].to_numpy(float), rtol=0, atol=1e-9,
        ):
            raise ValueError("Kalman upstream forecast differs from the explained nuclear autonomous forecast.")
        if not np.allclose(
            _actual_values(kalman_history, name="kalman_view.backtest"),
            history_actual,
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError("Kalman and autonomous FINAL365 observations differ.")
        specifications.append(
            ("kalman", "residual_kalman", NUCLEAR_KALMAN_LABEL, kalman_history, kalman_forecast)
        )

    # Validate and prepare every requested variant before writing any report.
    prepared = {
        key: _model_result(
            history=history,
            forecast=future,
            actuals=actuals,
            model=model,
            label=label,
            data=report_data,
            output_directory=directory,
        )
        for key, model, label, history, future in specifications
    }
    storm_audit: Mapping[str, Any] = {"status": "unavailable"}
    if storm_archive is not None:
        from .nuclear_report_benchmark import attach_nuclear_storm
        storm_audit = attach_nuclear_storm(
            prepared, Path(storm_archive), zone=zone_key, timezone=data.timezone,
        )
        if storm_audit.get("status") == "complete":
            for variant in prepared.values():
                variant.statistics_scope_note = (
                    "Statistics sur les 365 derniers jours observés; livraison non observée laissée vide. "
                    "Comparaison avec Storm officiel sur les mêmes heures physiques disponibles. "
                    "Les prix du modèle et observés restent visibles sans score comparatif quand Storm manque. "
                    "Le warm-up n'entre pas dans les scores."
                )
    if refreshed_observations is not None:
        for variant in prepared.values():
            reference_label = refreshed_observations.get("actual_reference_label")
            if reference_label in ("EPEX", "ENTSO-E", "ENTSO-E + EPEX"):
                variant.statistics_scope_note += (
                    f" Référence des prix réalisés et des scores NYX/Storm : {reference_label}; "
                    "même snapshot vérifié et mêmes prix horaires pour les deux modèles."
                )
            comparison = getattr(variant, "hourly_comparison_source", None)
            observed_index = actuals.index[np.isfinite(actuals.to_numpy(float))]
            common_index = observed_index
            storm_source: Mapping[str, Any] = {}
            allowed_dst = pd.DatetimeIndex([], tz="UTC")
            if comparison is not None:
                common_index = pd.DatetimeIndex(comparison.loc[
                    np.isfinite(comparison[["actual", "q50", "_benchmark_q50"]].to_numpy(float)).all(axis=1),
                    "_timestamp_utc",
                ])
                contract_audit = variant.statistics_benchmark_contract["materialization_audit"]
                storm_source = contract_audit["source_materialization"]["source"]
                allowed_dst = pd.DatetimeIndex(pd.to_datetime(
                    contract_audit["source_materialization"]["dst"]["native_allowed_missing_utc"], utc=True,
                ))
            complete_days = []
            for local_day in pd.Index(common_index.tz_convert(data.timezone).date).unique():
                expected_day = local_delivery_day_index(local_day, timezone=data.timezone).difference(allowed_dst)
                if len(expected_day) and not len(expected_day.difference(common_index)):
                    complete_days.append((local_day, len(expected_day)))
            latest_complete, complete_hours = complete_days[-1] if complete_days else (None, None)
            variant.statistics_freshness = {
                "timezone": data.timezone, "target_series": refreshed_observations["series"],
                "actual_extracted_at_utc": refreshed_observations["extracted_at_utc"],
                "actual_applied_end_utc": str(observed_index[-1]) if len(observed_index) else None,
                "storm_available": comparison is not None,
                "storm_extracted_at_utc": storm_source.get("extracted_at_utc"),
                "common_delivery_end_utc": str(common_index[-1]) if len(common_index) else None,
                "last_complete_common_day_local": str(latest_complete) if latest_complete is not None else None,
                "common_hours_last_day": complete_hours, "expected_common_hours_last_day": complete_hours,
            }
    if attribution_directory is not None:
        from .reporting import _attach_variable_attribution
        explained = prepared.get("autonomous")
        if explained is None:
            explained = _model_result(history=historical, forecast=forecast, actuals=actuals,
                model="residual_corrected", label=NUCLEAR_MODEL_LABEL, data=report_data,
                output_directory=directory)
        _attach_variable_attribution(
            explained, directory=Path(attribution_directory),
            native_model="residual_corrected", timezone=data.timezone,
        )
        if hasattr(explained, "variable_attribution") and "kalman" in prepared:
            prepared["kalman"].variable_attribution = {
                **explained.variable_attribution,
                "scope": "upstream_model", "is_upstream_attribution": True,
                "explained_model": "residual_corrected", "explained_model_label": NUCLEAR_MODEL_LABEL,
                "reported_model_label": NUCLEAR_KALMAN_LABEL,
            }
    raw = _timestamped(result.raw_history, name="raw_history")
    audit = {
        "schema_version": 1,
        "diagnostic_only": True,
        "production": False,
        "publication_mode": "forecast_run" if operational_layout else "diagnostic",
        "report_variants": list(selected),
        "zone": zone_key,
        "timezone": data.timezone,
        "delivery_day": str(day),
        "evaluation_start_day": str(start_day),
        "evaluation_end_day": str(day - timedelta(days=1)),
        "evaluation_days": 365,
        "evaluation_hours": len(expected_history),
        "same_actual_hours": True,
        "actual_price_reference": dict(refreshed_observations or {}),
        "historical_observation_precision": actual_precision,
        "delivery_day_observed_hours": int(delivery_finite.sum()),
        "delivery_day_placeholder_hours": int((~delivery_finite).sum()),
        "delivery_day_actual_reason": delivery_actual_reason,
        "delivery_day_actual_extracted_at_utc": (
            refreshed_observations["extracted_at_utc"] if refreshed_observations is not None else None
        ),
        "statistics_window": "latest_365_observed_local_days_plus_unobserved_delivery_placeholder",
        "storm_status": ("verified_standard_report_comparator" if storm_audit.get("status") == "complete"
                         else "unavailable_no_verified_comparator_loaded"),
        "storm_hourly_comparison": dict(storm_audit),
        "storm_used_as_input": False,
        "warmup": {
            "residual_days_before_final365": len(
                pd.Index(
                    residual.loc[residual.index < expected_history[0]]
                    .index.tz_convert(data.timezone).date
                ).unique()
            ),
            "raw_days_before_final365": len(
                pd.Index(
                    raw.loc[raw.index < expected_history[0]]
                    .index.tz_convert(data.timezone).date
                ).unique()
            ),
            "excluded_from_report_metrics": True,
        },
        "dst_policy": (
            "source-audited duplication of derived nuclear covariate only; "
            "UTC physical-hour pairing"
        ),
        "engine_audit": dict(getattr(result, "audit", {}) or {}),
        "source_audit": dict(source_audit or {}),
    }
    paths: dict[str, Path] = {}
    directory.mkdir(parents=True, exist_ok=True)
    for key, _, label, _, _ in specifications:
        path = directory / f"forecast_{zone_key.lower()}_{day}_nuclear_{key}.html"
        write_html_report(
            [prepared[key]],
            {
                "report": {
                    "title": f"{zone_key} — {label}" + ("" if operational_layout else " — diagnostic uniquement"),
                    "forecast_history_hours": 168,
                }
            },
            path,
        )
        _replace_report_labels(path, native_label=label, baseline_label=None)
        document = path.read_text(encoding="utf-8")
        # The legacy renderer always includes a price-only comparison table.
        # There is no such verified baseline here, so omit that empty section.
        document = re.sub(
            r'<h3>Comparaison au modèle prix seul</h3>\s*'
            r'(?:<div class="table-wrap">)?<table\b.*?</table>(?:</div>)?',
            "",
            document,
            count=1,
            flags=re.DOTALL,
        )
        if not operational_layout:
            document = document.replace("Prévision opérationnelle du", "Prévision diagnostique du")
            document = document.replace(
                "Prévision Day-Ahead réelle", "Prévision Day-Ahead — diagnostic"
            )
        document = document.replace(
            "<main>", "<main>" + _diagnostic_banner(audit, kalman=key == "kalman"), 1
        )
        path.write_text(document, encoding="utf-8")
        paths[key] = path
    audit_path = directory / "nuclear_report_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    paths["audit"] = audit_path
    return paths


__all__ = ["NUCLEAR_MODEL_LABEL", "NUCLEAR_KALMAN_LABEL", "render_nuclear_reports"]
