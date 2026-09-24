"""Isolated, read-only reports for solar correction of frozen nuclear Chronos."""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import html
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd

from .hourly_contract import local_delivery_day_index
from .nuclear_cwe_reporting import _attribute, _comparison, _observations, _view_frames
from .nuclear_forecast import NUCLEAR_ALIAS, _digest_frame
from .nuclear_reporting import _actual_values, _model_result, _report_zone_data, _require_quantiles, _timestamped
from .reporting import _replace_report_labels
from .solar_cwe_forecast import SOLAR_ALIASES, SOLAR_KNOWN_COLUMNS, SOLAR_SERIES
from chronos2_modular.report import write_html_report

ENGINE = "solar_correction_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "solar_residual": "Chronos nucléaire figé + correcteur solaire",
    "solar_residual_standard_kalman": "Chronos nucléaire figé + correcteur solaire + Kalman standard (A)",
    "solar_residual_solar_kalman": "Chronos nucléaire figé + correcteur solaire + Kalman solaire (B)",
    "nuclear_autonomous": "Chronos nucléaire + correcteur de référence",
    "nuclear_kalman": "Chronos nucléaire + correcteur de référence + Kalman",
}
FAMILIES = tuple(LABELS)[:3]
_Q = ("q10", "q50", "q90")


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)


def _equal(left, right, message):
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    except AssertionError as exc:
        raise ValueError(message) from exc


def _inputs(results, data):
    if set(results) != {"residual", "residual_kalman"}:
        raise ValueError("Both solar residual / solar residual+Kalman ablations are required.")
    audits = {}
    for key, result in results.items():
        audit = dict(_attribute(result, "audit", {}) or {})
        expected = {"candidate_engine": ENGINE, "candidate_variant": "solar_" + key,
            "chronos_recomputed": False, "chronos_runtime_calls": 0, "frozen_chronos_exact_match": True,
            "baseline_inputs_preserved": True, "inherited_residual_feature_builder_unchanged": True,
            "additional_raw_input_count": 4, "new_ramps_or_aggregates": False,
            "spike_classifier_added": False, "promotion_eligible": False, "production_changed": False,
            "sealed_live_contract_modified": False}
        if any(audit.get(k) != v for k, v in expected.items()):
            raise ValueError("Solar correction requires an unchanged frozen nuclear Chronos and isolated corrections.")
        channels = {"solar_chronos_context_columns": [], "solar_chronos_known_future_columns": [],
            "solar_residual_features": SOLAR_KNOWN_COLUMNS,
            "solar_kalman_market_features": SOLAR_ALIASES if key == "residual_kalman" else []}
        if any(set(audit.get(k, [])) != set(v) or k not in audit for k, v in channels.items()):
            raise ValueError("Solar stage audit dropped or reassigned solar inputs.")
        for alias in SOLAR_ALIASES:
            if audit.get("solar_sources", {}).get(alias) != {"series": SOLAR_SERIES[alias], "unit": "GW",
                    "semantic": "forecast_generation", "daily_broadcast": False}:
                raise ValueError("Four hourly solar generation forecasts in GW are required.")
        for field, digest in (("residual_statistics", "shared_upstream_statistics_sha256"),
                              ("source_forecast", "shared_upstream_forecast_sha256")):
            if audit.get(digest) != _digest_frame(_attribute(result, field)):
                raise ValueError("Solar shared upstream digest mismatch.")
        raw = _timestamped(_attribute(result, "raw_history"), name="frozen Chronos")
        future = _timestamped(_attribute(result, "source_forecast"), name="frozen delivery")
        _require_quantiles(raw.rename(columns={q: f"chronos2__{q}" for q in _Q}), "chronos2", name="frozen history")
        _require_quantiles(future, "chronos2", name="frozen delivery")
        future = future[[f"chronos2__{q}" for q in _Q]].rename(columns={f"chronos2__{q}": q for q in _Q})
        if audit.get("frozen_chronos_history_sha256") != _digest_frame(raw[list(_Q)]) or audit.get("frozen_chronos_future_sha256") != _digest_frame(future):
            raise ValueError("Frozen nuclear Chronos digest mismatch.")
        audits[key] = audit
    a, b = results["residual"], results["residual_kalman"]
    for field in ("raw_history", "residual_statistics", "source_forecast"):
        _equal(_attribute(a, field), _attribute(b, field), "A/B must share exactly the same frozen upstream.")
    copied = _report_zone_data(data, _attribute(b, "covariates"))
    context = copied.model_context_covariates.copy()
    original = _timestamped(data.model_context_covariates, name="model context")
    standard = _timestamped(_attribute(a, "covariates"), name="standard Kalman inputs")
    solar = _timestamped(_attribute(b, "covariates"), name="solar Kalman inputs")
    if NUCLEAR_ALIAS not in standard or set(SOLAR_ALIASES).intersection(standard):
        raise ValueError("Standard Kalman must retain nuclear and exclude solar covariates.")
    _equal(standard, solar.drop(columns=list(SOLAR_ALIASES), errors="ignore"), "A/B baseline Kalman inputs differ.")
    for key, covariates in (("residual", standard), ("residual_kalman", solar)):
        config = audits[key].get("kalman_covariate_config", {})
        required = set(SOLAR_ALIASES) if key == "residual_kalman" else set()
        if set(config.get("input_columns", [])) != set(covariates) or not config.get("groups") or any(set(columns).intersection(SOLAR_ALIASES) != required for columns in config["groups"].values()):
            raise ValueError("Kalman configuration and audited solar stages disagree.")
    manifest = copied.input_manifest.copy(deep=True)
    for alias, known in zip(SOLAR_ALIASES, SOLAR_KNOWN_COLUMNS):
        if alias not in solar or alias not in original or known not in original or known not in data.known_future_columns:
            raise ValueError("Required solar residual/Kalman forecast input is missing.")
        values = np.column_stack([source[column].reindex(solar.index).to_numpy(float)
            for source, column in ((solar, alias), (original, alias), (original, known))])
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, values[:, [0]], rtol=0, atol=0):
            raise ValueError("Solar report covariates must match the complete, unmodified forecasts.")
        context[known] = original[known].reindex(context.index)
        mask = manifest.alias.astype(str).eq(alias)
        manifest.loc[mask, "source"] = f"Prévision solaire {alias[:2].upper()} — correcteur et Kalman B uniquement"
        manifest.loc[mask, "unit"] = "GW"
        manifest.loc[mask, "information_type"] = "forecast_generation"
    for column in data.known_future_columns:
        if column not in original:
            raise ValueError("Declared known-future channel is missing from report inputs.")
        context[column] = original[column].reindex(context.index)
    return replace(copied, input_manifest=manifest, model_context_covariates=context), audits


def _banner(audit, comparison):
    prospective = ("Nouveaux jours préenregistrés : en attente de validation séparée ; ce rapport ne déclare aucun succès. "
        if audit["validation_protocol"] else "Nouveaux jours : préenregistrement non fourni, validation prospective indisponible. ")
    delta = audit["comparisons"]["solar_kalman_vs_standard_kalman"]["annual"]["gain_incumbent_minus_candidate"]
    delta_note = (f'<p>Gain B vs A : MAE {delta["mae_eur_mwh"]:.3f} ; RMSE {delta["rmse_eur_mwh"]:.3f} EUR/MWh.</p>'
        if comparison["candidate_model"] == FAMILIES[2] else "")
    rows = "".join("<tr><th>" + label + "</th>" + "".join(f"<td>{annual[key][metric]:.3f}</td>"
        for key in ("incumbent", "candidate", "gain_incumbent_minus_candidate")) + "</tr>"
        for annual in (comparison["annual"],) for metric, label in (("mae_eur_mwh", "MAE horaire"),
            ("rmse_eur_mwh", "RMSE horaire"), ("daily_mean_mae_eur_mwh", "MAE moyenne journalière")))
    return ('<section data-report-section="solar-correction" data-production="false">'
        '<h2>Correction solaire — expérience diagnostique</h2><p>Chronos nucléaire strictement figé : '
        'aucun solaire dans le transformer, aucun recalcul Chronos. Quatre prévisions solaires horaires '
        'FR, DE, BE, NL en GW entrent dans le correcteur résiduel ; seul B les ajoute au Kalman. '
        'A et B partagent exactement les mêmes prévisions avant Kalman.</p>'
        '<p>Historique de 365 jours : diagnostic, pas validation prospective. '
        + prospective +
        'Aucune promotion, aucun ajustement spécifique FR, aucun entraînement par le rapport. '
        'Storm est un comparateur officiel uniquement, jamais une entrée. Trous conservés. '
        'P10/P50/P90 conservés ; déciles intermédiaires et CRPS approximés pour affichage.</p><h3>'
        + html.escape(comparison["candidate_label"] + " vs " + comparison["incumbent_label"])
        + '</h3><p>Mêmes heures et mêmes observations ; gain positif = erreur réduite (EUR/MWh).</p>'
        '<table><thead><tr><th>Métrique</th><th>Référence figée</th><th>Expérience</th><th>Gain</th></tr></thead>'
        '<tbody>' + rows + '</tbody></table>' + delta_note + '<details><summary>Audit correction solaire</summary><pre>'
        + html.escape(_json(audit)) + '</pre></details></section>')


def _diagnostic_slices(candidate, reference, comparison, timezone):
    left = _timestamped(candidate.statistics_candidate, name="candidate")
    right = _timestamped(reference.statistics_candidate, name="reference")
    days = pd.Index(left.index.tz_convert(timezone).date).astype(str)
    mask = (days >= comparison["statistics_start_day"]) & (days <= comparison["statistics_end_day"])
    left, right, days = left.loc[mask], right.loc[mask], days[mask]
    errors = pd.DataFrame({"candidate": left.q50 - left.actual, "reference": right.q50 - left.actual})
    daily = errors.abs().groupby(days).mean()
    comparison["annual"]["daily_mae_win_rate"] = float((daily.candidate < daily.reference).mean())
    comparison["annual"]["daily_mae_win_rate_definition"] = "fraction of complete common days with candidate MAE < reference MAE; ties are not wins"
    slices = {}
    for threshold in (200, 300):
        selected = errors.loc[left.actual >= threshold]
        scores = {column: {"mae_eur_mwh": float(selected[column].abs().mean()) if len(selected) else None,
            "rmse_eur_mwh": float(np.sqrt((selected[column] ** 2).mean())) if len(selected) else None}
            for column in errors}
        slices[str(threshold)] = {"hours": len(selected), "candidate": scores["candidate"], "incumbent": scores["reference"]}
    comparison["high_price_slices"] = {"ex_post_only": True, "selection": "actual >= threshold EUR/MWh", "thresholds": slices}


def render_solar_correction_reports(results: Mapping, *, data, zone, delivery_day, output_directory,
                                    incumbent, storm_archive, observed_source_audit, source_audit,
                                    validation_protocol=None):
    """Three standard HTML reports, paired nuclear references and B-minus-A audit."""
    directory = Path(output_directory).expanduser().absolute()
    allowed = PROJECT_ROOT.resolve() / "runs/experiments" / ENGINE
    if allowed.resolve() != allowed or directory.resolve() != directory or not directory.is_relative_to(allowed):
        raise ValueError("Solar correction reports must stay in runs/experiments/solar_correction_v1.")
    zone, date = str(zone).upper(), pd.Timestamp(delivery_day)
    if zone != str(data.zone).upper() or pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError("A matching zone and naive civil delivery date are required.")
    report_data, engines = _inputs(results, data)
    day, first = date.date(), date.date() - timedelta(days=365)
    hi = pd.date_range(pd.Timestamp(first, tz=data.timezone), pd.Timestamp(day, tz=data.timezone),
                       freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    fi = local_delivery_day_index(day, timezone=data.timezone).rename("delivery_start_utc")
    frames, frozen = {}, {}
    for key, obj in (("solar_residual", results["residual"]), ("solar_b", results["residual_kalman"]),
                     ("nuclear_autonomous", incumbent)):
        full = _timestamped(_attribute(obj, "residual_statistics"), name=key)
        history = full.loc[(full.index >= hi[0]) & (full.index <= hi[-1])]
        future = _timestamped(_attribute(obj, "source_forecast"), name=key)
        if not history.index.equals(hi) or not future.index.equals(fi):
            raise ValueError("All models must share the complete FINAL365/delivery physical grid.")
        kh, kf = _view_frames(_attribute(obj, "kalman_view"), hi, fi, name=key, allow_prefix=True)
        for frame in (history, future):
            _require_quantiles(frame, "residual_corrected", name=key)
            _require_quantiles(frame, "chronos2", name=key)
        for filtered, upstream in ((kh, history), (kf, future)):
            cols = [f"residual_corrected__{q}" for q in _Q]
            if not set(cols).issubset(filtered) or not np.array_equal(filtered[cols].to_numpy(), upstream[cols].to_numpy()):
                raise ValueError("Kalman upstream differs from its autonomous solar/nuclear family.")
        if not np.array_equal(_actual_values(kh, name=key), _actual_values(history, name=key)):
            raise ValueError("Autonomous/Kalman observations differ.")
        kalman_key = {"solar_residual": FAMILIES[1], "solar_b": FAMILIES[2], "nuclear_autonomous": "nuclear_kalman"}[key]
        frames[key], frames[kalman_key] = (history, future), (kh, kf)
        frozen[key] = (_timestamped(_attribute(obj, "raw_history"), name=key), full, future)
    # Bare q* identify Chronos only in raw_history. In corrected frames they
    # can alias the residual forecast, which is expected to differ by candidate.
    for prefix, left, right in zip(("", "chronos2__", "chronos2__"),
                                  frozen["solar_residual"], frozen["nuclear_autonomous"]):
        cols = [f"{prefix}{q}" for q in _Q]
        if not left.index.isin(right.index).all():
            raise ValueError("Frozen nuclear Chronos grid differs.")
        _equal(left[cols], right.reindex(left.index)[cols], "Frozen nuclear Chronos quantiles differ.")
    actuals, observation_audit = _observations(frozen["solar_residual"][1], frames["solar_residual"][0],
        frames["nuclear_autonomous"][0], fi, data, zone, str(day), observed_source_audit)
    frames.pop("solar_b")
    prepared = {key: _model_result(history=h, forecast=f, actuals=actuals,
        model="residual_kalman" if key.endswith("kalman") else "residual_corrected",
        label=LABELS[key], data=report_data, output_directory=directory) for key, (h, f) in frames.items()}
    pairs = {FAMILIES[0]: "nuclear_autonomous", FAMILIES[1]: "nuclear_kalman", FAMILIES[2]: "nuclear_kalman",
             "solar_kalman_vs_standard_kalman": FAMILIES[1]}
    comparisons = {}
    for key, reference in pairs.items():
        candidate = FAMILIES[2] if key == "solar_kalman_vs_standard_kalman" else key
        compared = _comparison(prepared[candidate], prepared[reference], day=date, timezone=data.timezone)
        compared.pop("june_24_26", None)
        _diagnostic_slices(prepared[candidate], prepared[reference], compared, data.timezone)
        comparisons[key] = {**compared, "candidate_model": candidate, "incumbent_model": reference,
                           "candidate_label": LABELS[candidate], "incumbent_label": LABELS[reference]}
    storm = {"status": "unavailable"}
    if storm_archive is not None:
        from .nuclear_report_benchmark import attach_nuclear_storm
        storm = attach_nuclear_storm(prepared, Path(storm_archive), zone=zone, timezone=data.timezone)
    audit = {"schema_version": 1, "candidate_engine": ENGINE, "production": False, "diagnostic_only": True,
        "model_fitted_by_report": False, "chronos_recomputed": False, "promotion_eligible": False,
        "zone": zone, "delivery_day": str(day), "evaluation_start_day": str(first),
        "evaluation_end_day": str(day - timedelta(days=1)), "evaluation_days": 365, "evaluation_hours": len(hi),
        "historical_observations": observation_audit, "storm_hourly_comparison": storm, "storm_used_as_input": False,
        "engine_audits": engines, "source_audit": dict(source_audit or {}), "comparisons": comparisons,
        "validation_protocol": dict(validation_protocol or {}), "same_family_comparison": True,
        "prospective_validation_status": "pending_new_days_not_evaluated_by_report" if validation_protocol else "not_registered"}
    banners = {family: _banner(audit, comparisons[family]) for family in FAMILIES}
    paths = {key: directory / f"forecast_{zone.lower()}_{day}_{key}.html" for key in FAMILIES}
    paths.update({key: directory / f"solar_correction_report_{key}.json" for key in ("audit", "comparison")})
    if any(path.resolve() != path for path in paths.values()):
        raise ValueError("Solar correction output files cannot be redirected.")
    directory.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        prepared[family].statistics_scope_note = "365 jours observés communs ; diagnostic historique uniquement. Chronos nucléaire figé. Storm aux seules heures communes vérifiées ; trous conservés. Aucune promotion."
        path = paths[family]
        write_html_report([prepared[family]], {"report": {"title": f"{zone} — {LABELS[family]}", "forecast_history_hours": 168}}, path)
        _replace_report_labels(path, native_label=LABELS[family], baseline_label=None)
        document = re.sub(r'<h3>Comparaison au modèle prix seul</h3>\s*(?:<div class="table-wrap">)?<table\b.*?</table>(?:</div>)?', "", path.read_text(encoding="utf-8"), count=1, flags=re.DOTALL)
        path.write_text(document.replace("Prévision opérationnelle du", "Prévision expérimentale du").replace("<main>", "<main>" + banners[family], 1), encoding="utf-8")
    paths["audit"].write_text(_json(audit), encoding="utf-8")
    paths["comparison"].write_text(_json(comparisons), encoding="utf-8")
    return paths
