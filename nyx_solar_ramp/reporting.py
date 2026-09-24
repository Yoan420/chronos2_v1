"""Offline Solar Ramp reports from saved artefacts, without fitting or fetching.

The detailed country pages reuse the unchanged production HTML renderer and
the zonal adapter's strict 365-day/Storm archive verification.  A missing
contract is reported, never repaired by inventing rows, prices or quantiles.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ZONES = ("FR", "DE", "BE", "NL")
TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin", "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
INPUTS = ("panel.parquet", "predictions.parquet", "metrics.json", "source_audit.json", "manifest.json", "literature.json")
BASELINE = ("forecast", "q10", "q90", "actual", "benchmark_forecast", "sample", "forecast_origin_utc")
PREDICTIONS = ("candidate_forecast", "candidate_q10", "candidate_q90", "risk_probability",
               "alert_threshold", "applied_correction", "selected_weight", "gate_reason", "evaluation_phase")
CATEGORICAL_FIELDS = ("gate_reason", "evaluation_phase")
PAYLOAD_SCHEMA_VERSION = 2
SERIES = {
    "solarx_local_solar_gw": "Solaire local (GW)", "solarx_peer_solar_gw": "Solaire voisins (GW)",
    "solarx_local_solar_drop_1h_gwph": "Retrait solaire local 1 h (GW/h)",
    "solarx_peer_solar_drop_1h_gwph": "Retrait solaire voisins 1 h (GW/h)",
    "solarx_local_residual_gw": "Charge résiduelle locale (GW)",
    "solarx_peer_residual_gw": "Charge résiduelle voisins (GW)",
    "solarx_local_pressure_proxy": "Pression locale — proxy, pas une contrainte réseau observée",
}
LABEL = "NYX + Solar Ramp — candidat gouverné figé"


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(_clean(value), ensure_ascii=False, allow_nan=False, indent=indent)


def _script_json(value: Any) -> str:
    compact = json.dumps(_clean(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return (compact.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory(value: Path, root: Path) -> Path:
    project = Path(root).resolve()
    namespace = project / "runs" / "experiments" / "nyx_solar_ramp_v1"
    if namespace.resolve() != namespace:
        raise ValueError("Solar Ramp namespace must not redirect to a production path.")
    directory = Path(value).resolve()
    if not directory.is_relative_to(namespace) or directory == namespace:
        raise ValueError("Report must remain in a run below runs/experiments/nyx_solar_ramp_v1.")
    for name in INPUTS:
        path = directory / name
        if path.resolve().parent != directory or not path.is_file():
            raise ValueError(f"Missing or redirected frozen report input: {name}")
    if (directory / "reports").resolve() != directory / "reports":
        raise ValueError("Report output cannot be a redirected directory.")
    return directory


def _utc(frame: pd.DataFrame, column: str) -> None:
    values = frame[column]
    if values.isna().any() or any(pd.Timestamp(v).tzinfo is None for v in values):
        raise ValueError(f"{column}: explicit timezone-aware physical timestamps required.")
    frame[column] = pd.to_datetime(values, utc=True, errors="raise")


def _validated_frames(panel: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check unchanged baseline values and saved candidate interval arithmetic."""
    panel, predictions = panel.copy(deep=True), predictions.copy(deep=True)
    for frame in (panel, predictions):
        if frame.columns.has_duplicates or not {"zone", "timestamp_utc"}.issubset(frame):
            raise ValueError("Unique column names and zone/physical-hour identities required.")
        frame["zone"] = frame.zone.astype(str).str.upper()
        if frame.empty or not set(frame.zone).issubset(ZONES):
            raise ValueError("A non-empty FR/DE/BE/NL report panel is required.")
        _utc(frame, "timestamp_utc")
    if panel.duplicated(["zone", "timestamp_utc"]).any():
        raise ValueError("Duplicate baseline physical hour.")
    required = {"variant", "candidate_forecast", "candidate_q10", "candidate_q90"}
    if not required.issubset(predictions) or "governed" not in set(predictions.variant):
        raise ValueError("Saved governed variant and candidate P10/P50/P90 required.")
    if predictions.duplicated(["variant", "zone", "timestamp_utc"]).any():
        raise ValueError("Duplicate variant/physical-hour prediction.")
    missing = set(BASELINE) - set(panel)
    if missing:
        raise ValueError(f"Frozen baseline columns missing: {sorted(missing)}")
    keys = ["zone", "timestamp_utc"]
    panel = panel.sort_values(keys).reset_index(drop=True)
    _utc(panel, "forecast_origin_utc")
    joined = []
    preserved = [*BASELINE, *[c for c in panel if c.startswith("feature_")]]
    for variant, group in predictions.groupby("variant", sort=True):
        group = group.sort_values(keys).reset_index(drop=True)
        if not group[keys].equals(panel[keys]):
            raise ValueError(f"{variant}: all frozen baseline hours, including fallback rows, must be retained.")
        if "forecast_origin_utc" in group:
            _utc(group, "forecast_origin_utc")
        for column in preserved:
            if column not in group:
                group[column] = panel[column].to_numpy()
                continue
            try:
                pd.testing.assert_series_equal(group[column], panel[column], check_dtype=False,
                                               check_names=False, check_exact=True)
            except AssertionError as error:
                raise ValueError(f"{variant}: frozen baseline/source changed: {column}") from error
        for column in panel:
            if column not in group:
                group[column] = panel[column].to_numpy()
        for low, point, high in (("q10", "forecast", "q90"),
                                  ("candidate_q10", "candidate_forecast", "candidate_q90")):
            values = group[[low, point, high]].apply(pd.to_numeric, errors="raise").to_numpy(float)
            if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
                raise ValueError(f"{variant}: finite ordered saved P10/P50/P90 required; no interval invented.")
        if "applied_correction" in group and not np.allclose(
                group.candidate_forecast, group.forecast + group.applied_correction, rtol=1e-9, atol=1e-8):
            raise ValueError(f"{variant}: saved candidate price/correction arithmetic disagrees.")
        if {"selected_weight", "bounded_correction", "applied_correction"}.issubset(group) and not np.allclose(
                group.applied_correction, group.selected_weight * group.bounded_correction, rtol=1e-9, atol=1e-8):
            raise ValueError(f"{variant}: saved correction/weight arithmetic disagrees.")
        for column in ("risk_probability", "alert_threshold", "selected_weight"):
            if column in group:
                values = pd.to_numeric(group[column], errors="raise")
                if np.isinf(values).any() or ((values.dropna() < 0) | (values.dropna() > 1)).any():
                    raise ValueError(f"{variant}: {column} must lie in [0,1].")
        joined.append(group)
    return panel, pd.concat(joined, ignore_index=True)


def _prospective(manifest: dict) -> dict:
    raw = manifest.get("prospective", manifest.get("prospective_status", {}))
    if not isinstance(raw, dict):
        raw = {"status": str(raw)}
    return {**raw, "report_claim": "blocked_no_independently_verified_prospective_issue",
            "reasons": raw.get("reasons") or ["Aucune émission nouvelle avant publication des prix, avec contrôle frais EPEX et canonique, n’est certifiée par ces artefacts."]}


def _methodology(manifest: dict, metrics: dict) -> str:
    prospective = _prospective(manifest)
    config = manifest.get("config", {})
    minimum = html.escape(str(config.get("minimum_training_days", 120)))
    window = html.escape(str(config.get("training_window_days", 365)))
    delay = html.escape(str(config.get("label_delay_days", 2)))
    return ('<section class="notice" data-report-section="solar-ramp-methodology"><h2>Expérimental — aucune activation</h2>'
            '<p><strong>14 septembre 2026 : étude de cas connue, rétrospective.</strong> Les livraisons issues du '
            'champ « live » historique (15 septembre dans la source initiale, 18 septembre dans la référence '
            'actualisée) sont elles aussi connues : ce ne sont pas des prévisions prospectives nouvelles.</p>'
            '<p>Origine informationnelle cible J−1 08:00, heure civile locale. Les archives documentent des requêtes '
            '<em>as-of</em>, pas les horodatages de publication du fournisseur. La disponibilité historique des '
            'labels n’est pas certifiée. Aucune certification PIT stricte n’est revendiquée.</p>'
            f'<p>Apprentissage progressif : minimum {minimum} jours éligibles, fenêtre maximale {window} jours. '
            'Il ne s’agit pas d’un apprentissage sur 365 jours complets à chaque origine. Les premières heures '
            'et les inputs incomplets restent en repli NYX. Le délai conservateur appliqué aux labels '
            f'(fin de livraison + {delay} jours) est une hypothèse diagnostique, pas une publication fournisseur vérifiée.</p>'
            '<p>Année déjà examinée : exploration, sélection et fenêtre finale diagnostique ne constituent pas '
            'une validation prospective indépendante. Le gouverneur conserve la référence quand sa règle refuse '
            'une correction ; les heures de repli restent dans les scores. Aucun ajustement n’est appris par ce rapport.</p>'
            '<p>Storm est un comparateur figé ex post, jamais une feature du modèle. La pression affichée est '
            'un proxy physique, pas une mesure de congestion ou un prix marginal causal.</p>'
            '<p>P10/P50/P90 : valeurs sauvegardées, sans quantiles reconstruits dans les figures. Les rapports '
            'standards interpolent seulement les déciles intermédiaires pour le CRPS d’affichage. Les probabilités '
            'décrivent le risque, pas une attribution causale du prix.</p>'
            '<h3>Prospective : BLOQUÉE / non démontrée</h3><pre>'
            + html.escape(_json(prospective, indent=2)) + '</pre><h3>Décision sauvegardée</h3><pre>'
            + html.escape(_json(metrics.get("decision", "non fournie"), indent=2)) + '</pre></section>')


def _standard_reports(predictions: pd.DataFrame, source_audit: dict, output: Path,
                      manifest: dict, metrics: dict) -> tuple[dict[str, str], dict]:
    """Reuse exact standard renderer; unverifiable country contracts stay absent."""
    output = Path(output).absolute()
    if output.resolve() != output:
        raise ValueError("Standard report output must not be redirected.")
    from nyx_scarcity_zonal.reporting import _prepare, _result
    from chronos2_hourly.reporting import _replace_report_labels
    from chronos2_modular.report import write_html_report

    frame = predictions.loc[predictions.variant.eq("governed")].copy(deep=True)
    if "risk_probability" in frame:
        frame["spike_probability"] = frame.risk_probability
    if "alert_threshold" in frame:
        frame["probability_gate"] = frame.alert_threshold
    adapter_audit = {**source_audit,
                     "decision_policy": "saved_solar_ramp_governed_prediction_not_recomputed_by_report"}
    links, audits = {}, {}
    prepared = []
    for zone, group in frame.groupby("zone", sort=True):
        try:
            result, audit = _result(_prepare(group), source_audit=adapter_audit, directory=output, label=LABEL)
        except (ValueError, OSError, KeyError, TypeError) as error:
            audits[zone] = {"status": "unavailable", "reason": str(error), "no_missing_values_fabricated": True}
            continue
        prepared.append((zone, result, audit))
    if prepared:
        output.mkdir(parents=True, exist_ok=True)
    for zone, result, audit in prepared:
        path = output / f"forecast_{zone.lower()}_{audit['live_day']}_nyx_solar_ramp.html"
        if path.resolve() != path:
            raise ValueError("Standard report file must not be redirected.")
        write_html_report([result], {"report": {"title": f"{zone} — NYX Solar Ramp — EXPÉRIMENTAL",
                                                "forecast_history_hours": 168}}, path)
        _replace_report_labels(path, native_label=LABEL, baseline_label="NYX nucléaire + Kalman figé")
        document = path.read_text(encoding="utf-8")
        for old, new in (("Prévision opérationnelle du", "Courbe historique sauvegardée du"),
                         ("Prévision Day-Ahead réelle", "Livraison historique connue — hors scores annuels"),
                         ("prévision Day-Ahead opérationnelle", "courbe historique expérimentale"),
                         ("covariables actives", "séries affichées — sélection illustrative")):
            document = document.replace(old, new)
        document = document.replace("<main>", '<main><p><a href="../index.html">← Rapport Solar Ramp quatre pays</a></p>'
                                    + _methodology(manifest, metrics), 1)
        document = document.replace("</main>", '<section><h2>Traçabilité du rapport standard</h2><pre>'
                                    + html.escape(_json(audit, indent=2)) + '</pre></section></main>', 1)
        path.write_text(document, encoding="utf-8")
        links[zone] = "standard/" + path.name
        audits[zone] = {"status": "complete", **audit, "new_prospective_forecast": False,
                        "source_live_day_is_historical": True, "model_fitted_by_report": False,
                        "governance_evidence": "Saved governed prices, weights and gate_reason; report checks arithmetic, does not rerun the governor or certify training causality."}
    return links, audits


def _display(value: Any) -> str:
    value = _clean(value)
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return _json(value)
    return str(value)


def _table(rows: Any, columns: list[tuple[str, str]] | None = None) -> str:
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        return '<p class="muted">Données non fournies sous forme tabulaire ; voir l’audit.</p>'
    columns = columns or [(key, key) for key in dict.fromkeys(k for row in rows for k in row)]
    return ('<div class="table-wrap"><table><thead><tr>'
            + ''.join('<th>' + html.escape(label) + '</th>' for _, label in columns)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(_display(row.get(key)))
            + '</td>' for key, _ in columns) + '</tr>' for row in rows) + '</tbody></table></div>')


def _score_summary(metrics: dict) -> str:
    """Display saved candidate scores, never re-score the baseline as candidate."""
    scores = metrics.get("scores", [])
    if not isinstance(scores, list):
        return ""
    annual = [dict(row) for row in scores if isinstance(row, dict) and row.get("phase") == "all" and row.get("zone") == "ALL"]
    common = [dict(row) for row in scores if isinstance(row, dict) and row.get("phase") == "common_oos" and row.get("zone") == "ALL"]
    if not annual and not common:
        return ""
    order = {name: i for i, name in enumerate(("baseline", "control", "local", "regional", "ramps", "interactions", "governed"))}
    annual.sort(key=lambda row: (order.get(row.get("variant"), 100), str(row.get("variant"))))
    common.sort(key=lambda row: (order.get(row.get("variant"), 100), str(row.get("variant"))))
    columns = [("variant", "Variante"), ("n", "Heures prix"), ("mae", "MAE candidat"),
               ("rmse", "RMSE candidat"), ("spike_mae", "MAE pics ≥ 300"), ("pr_auc", "PR-AUC"),
               ("n_probability", "Heures probabilité"), ("probability_coverage", "Couverture probabilité")]
    result = ('<section data-report-section="solar-ramp-summary"><h2>Résultat numérique — ablations historiques</h2>'
              '<p>Scores agrégés quatre pays lus dans metrics.json. Les colonnes MAE/RMSE évaluent les '
              '<strong>prix candidats corrigés sauvegardés</strong> (candidate_forecast), pas à nouveau la référence '
              'forecast. Les pics et la PR-AUC ci-dessous concernent l’événement métier prix horaire ≥ 300 EUR/MWh.</p>')
    if annual:
        result += '<h3>Année complète : warm-up et repli inclus, livraison historique séparée exclue</h3>' + _table(annual, columns)
    if common:
        result += ('<h3>Support OOS commun : ablations de risque comparables</h3><p class="muted">'
                   'Toutes les variantes disposent d’une probabilité finie et d’un prix fini ; warm-up et livraison '
                   'historique séparée exclus. Les scores sur couvertures différentes ne démontrent pas un gain.</p>' + _table(common, columns))
    return result + '<p class="muted">Un classement rétrospectif ne vaut ni promotion, ni validation prospective. Décision détaillée ci-dessous.</p></section>'


def _physical_analysis(value: Any) -> str:
    """Render the optional, separately computed physical study without learning."""
    if not isinstance(value, dict):
        return ""
    event_columns = [("zone", "Pays"), ("delivery_start_local", "Début livraison local"),
                     ("delivery_end_local", "Fin livraison local"), ("forecast_origin_utc", "Origine information UTC"),
                     ("solar_local_gw", "Solaire local GW"), ("solar_peer_gw", "Solaire voisins GW"),
                     ("solar_drop_gwph", "Retrait local GW/h"), ("peer_solar_drop_gwph", "Retrait voisins GW/h"),
                     ("residual_gw", "Charge résiduelle GW"), ("wind_gw", "Éolien GW"),
                     ("pressure_proxy", "Pression proxy"), ("actual", "Observé"), ("baseline", "NYX"),
                     ("candidate", "Candidat"), ("risk_probability", "Probabilité"), ("alert", "Alerte")]
    counterexamples = value.get("counterexamples", {})
    if not isinstance(counterexamples, dict):
        counterexamples = {}
    matched = value.get("matched_strata", {})
    if not isinstance(matched, dict):
        matched = {}
    return ('<section id="physical-analysis" data-report-section="solar-ramp-physical-analysis">'
            '<h2>Solaire et pics — cas connu, contre-exemples et associations</h2>'
            '<p>Les tableaux suivants proviennent de l’analyse sauvegardée, sans seuil appris par le rapport. '
            'Les seuils physiques sont figés sur le warm-up initial, sans regarder les prix des périodes étudiées. '
            'Les comparaisons post-warm-up par heure, saison, charge résiduelle, vent et pression restent '
            '<strong>des associations descriptives, pas une démonstration causale</strong>.</p>'
            '<h3>14 septembre 2026 — Allemagne et Belgique, événement déjà connu</h3>'
            + _table(value.get("event_rows"), event_columns)
            + '<h3>Contre-exemples : retrait solaire marqué sans pic de prix</h3>'
            + _table(counterexamples.get("large_drop_without_spike"), event_columns)
            + '<h3>Contre-exemples : pic de prix sans retrait solaire marqué</h3>'
            + _table(counterexamples.get("spike_without_drop"), event_columns)
            + '<h3>Strates appariées — mêmes heure, saison et régimes physiques</h3>'
            + _table(matched.get("summary"))
            + '<details><summary>Toutes les strates et leurs effectifs</summary>' + _table(matched.get("rows")) + '</details>'
            + '<details><summary>Taux de pics par heure, saison et régime</summary>' + _table(value.get("regime_rates")) + '</details>'
            + '<details><summary>Seuils physiques warm-up, couverture et protocole</summary>'
            + _table(value.get("warmup_thresholds")) + '<pre>'
            + html.escape(_json({"coverage": value.get("coverage"), "protocol": value.get("protocol")}, indent=2))
            + '</pre></details></section>')


def _standard_support(metrics: dict, standards: dict) -> str:
    """Expose different score denominators; never refill a benchmark hole."""
    annual = {str(row.get("zone")): row for row in metrics.get("scores", [])
              if isinstance(row, dict) and row.get("phase") == "all" and row.get("variant") == "governed"}
    rows = []
    for zone, audit in standards.items():
        if audit.get("status") != "complete":
            continue
        total, paired = audit.get("evaluation_hours"), audit.get("paired_hours")
        rows.append({"zone": zone, "main_price_hours": annual.get(zone, {}).get("n"),
                     "evaluation_hours": total, "storm_paired_hours": paired,
                     "excluded_from_common_support": total-paired if total is not None and paired is not None else None})
    return ('<p><strong>Attention aux dénominateurs :</strong> les scores prix du rapport principal incluent '
            'toutes les heures avec observations et prévisions NYX/candidat, y compris les replis. Les comparaisons '
            'NYX–Storm des rapports standards utilisent uniquement les heures appariées avec Storm. '
            'Leurs tableaux annuels peuvent donc différer pour cette seule raison. Aucun trou Storm n’est rempli, '
            'même si une autre archive contient un prix pour cette heure.</p>'
            + _table(rows, [("zone", "Pays"), ("main_price_hours", "Scores prix principaux — heures"),
                            ("evaluation_hours", "Fenêtre standard — heures physiques"),
                            ("storm_paired_hours", "Comparaison Storm — heures appariées"),
                            ("excluded_from_common_support", "Heures hors support commun")]))


def _literature(value: Any) -> str:
    records = value if isinstance(value, list) else value.get("references", value.get("sources", [])) if isinstance(value, dict) else []
    rows = []
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        url = str(record.get("url", record.get("link", "")))
        title = str(record.get("title", record.get("name", url)))
        label = html.escape(title)
        if url.startswith(("https://", "http://")):
            label = '<a href="' + html.escape(url, quote=True) + '" rel="noopener noreferrer">' + label + '</a>'
        authors = record.get("authors", "")
        authors = "; ".join(str(author) for author in authors) if isinstance(authors, list) else str(authors or "")
        year = str(record.get("year", "") or "")
        citation = authors + (f" ({year})" if year else "")
        application = record.get("application", record.get("note", record.get("relevance", "")))
        rows.append('<li>' + label + ('<div class="muted">' + html.escape(citation) + '</div>' if citation else '')
                    + ('<p><strong>Application à ce laboratoire :</strong> ' + html.escape(str(application)) + '</p>' if application else '') + '</li>')
    return '<ul>' + ''.join(rows) + '</ul>' if rows else '<p>Références conservées dans literature.json ci-dessous.</p>'


def _payload(panel: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Lossless display encoding; repeated text and identical arrays stored once.

    Every saved numerical value is retained with Python's full float JSON
    precision.  Array references are shared only when their exact JSON bytes
    agree, so even the sign of floating zero is preserved.  Only fields read
    by the interaction JavaScript are embedded.
    """
    dictionaries = {column: sorted({str(value) for value in predictions[column].dropna()})
                    for column in CATEGORICAL_FIELDS if column in predictions}
    codes = {column: {value: index for index, value in enumerate(values)} for column, values in dictionaries.items()}
    data = {"schema_version": PAYLOAD_SCHEMA_VERSION, "zones": {}, "dictionaries": dictionaries,
            "variants": sorted(predictions.variant.astype(str).unique().tolist()), "series": SERIES}
    for zone, frame in panel.groupby("zone", sort=False):
        frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
        local = frame.timestamp_utc.dt.tz_convert(TIMEZONES[zone])
        fields = ["actual", "forecast", "q10", "q90", "benchmark_forecast", *[c for c in SERIES if c in frame]]
        zone_data = {"timestamp": frame.timestamp_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
                     "local_label": local.dt.strftime("%Y-%m-%d %H:%M %z").tolist(),
                     "day": local.dt.strftime("%Y-%m-%d").tolist(),
                     "values": {c: _clean(frame[c].tolist()) for c in fields}, "variants": {}, "arrays": []}
        # Intern exact serialized sequences, not rounded float fingerprints.
        # Baseline quantiles frequently equal candidate intervals in every hour.
        references = {_script_json(values): {"baseline": name} for name, values in zone_data["values"].items()}
        for variant, group in predictions.loc[predictions.zone.eq(zone)].groupby("variant", sort=True):
            group = group.sort_values("timestamp_utc")
            encoded = {}
            for column in PREDICTIONS:
                if column not in group:
                    continue
                values = ([codes[column][str(value)] if pd.notna(value) else -1 for value in group[column]]
                          if column in codes else _clean(group[column].tolist()))
                sequence = _script_json(values)
                if sequence not in references:
                    references[sequence] = {"array": len(zone_data["arrays"])}
                    zone_data["arrays"].append(values)
                encoded[column] = references[sequence]
            zone_data["variants"][str(variant)] = encoded
        data["zones"][zone] = zone_data
    return _clean(data)


_CSS = """
:root{color-scheme:light;--bg:#f3f6fa;--fg:#152d45;--card:#fff;--line:#dce5ef;--muted:#566b80}
body.dark{color-scheme:dark;--bg:#0c1928;--fg:#e2ebf4;--card:#14263b;--line:#31465d;--muted:#adbbca}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
header{padding:26px 4vw;background:#112d48;color:white;border-bottom:5px solid #30bfb2}header h1{margin:0;font-size:30px}
header p{margin:8px 0 0;color:#c5dbe9}.tag{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#7ce3d9}
nav{display:flex;gap:18px;flex-wrap:wrap;margin-top:18px}nav a{color:#fff}main{max-width:1780px;margin:auto;padding:24px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:24px;margin-bottom:24px}
h2{font-size:23px;margin:0 0 14px}h3{font-size:18px;margin:20px 0 10px}.muted{color:var(--muted)}
.notice{border-left:5px solid #ddaa47}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.country{padding:14px;border:1px solid var(--line);border-radius:10px;min-width:0}.country h3{margin:0 0 8px}
.plot{height:940px;width:100%}.controls{display:flex;align-items:end;flex-wrap:wrap;gap:14px;margin:15px 0}
label{display:flex;flex-direction:column;font-size:13px;gap:4px}select,button,input{font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
button{cursor:pointer}button:hover{border-color:#24978d}a{color:#168c9c}body.dark a{color:#78d7e0}.table-wrap{overflow:auto;max-height:540px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{position:sticky;top:0;background:var(--card);font-weight:650}pre{white-space:pre-wrap;word-break:break-word;font:12px/1.6 ui-monospace,monospace;max-height:460px;overflow:auto}
details{margin-top:12px}summary{cursor:pointer;font-weight:600}.links{display:flex;flex-wrap:wrap;gap:12px}.links a{border:1px solid var(--line);border-radius:8px;padding:10px 16px;text-decoration:none}
footer{padding:0 24px 30px;text-align:center;color:var(--muted)}@media(max-width:980px){.grid{grid-template-columns:1fr}main{padding:12px}section{padding:16px}.plot{height:850px}}
"""


_PAYLOAD_DECODER_JS = r"""
function decodePayload(raw){
 if(raw.schema_version!==2)throw new Error('Unsupported Solar Ramp payload schema');
 for(const z of Object.values(raw.zones)){
  for(const v of Object.values(z.variants)){
   for(const [field,reference] of Object.entries(v)){
    const values=Object.prototype.hasOwnProperty.call(reference,'baseline')?z.values[reference.baseline]:z.arrays[reference.array];
    if(!Array.isArray(values))throw new Error('Invalid Solar Ramp saved-array reference');
    v[field]=Object.prototype.hasOwnProperty.call(raw.dictionaries,field)?values.map(code=>code<0?null:raw.dictionaries[field][code]):values;
   }
  }
  delete z.arrays;
 }
 return raw;
}
"""


_JS = _PAYLOAD_DECODER_JS + r"""
const payload=decodePayload(JSON.parse(document.getElementById('solar-ramp-data').textContent));
const days=[...new Set(Object.values(payload.zones).flatMap(z=>z.day))].sort();
const daySelect=document.getElementById('delivery-day'), variantSelect=document.getElementById('variant');
for(const day of days){const o=document.createElement('option');o.value=day;o.textContent=day;daySelect.append(o);}
for(const variant of payload.variants){const o=document.createElement('option');o.value=variant;o.textContent=variant+(variant==='governed'?' — décision retenue':' — ablation / diagnostic');variantSelect.append(o);}
daySelect.value=days.includes('2026-09-14')?'2026-09-14':days[days.length-1];variantSelect.value='governed';
let syncing=false;const plotIds=[];
function selected(a,ids){return ids.map(i=>a?a[i]:null);}
function trace(z,key,label,ids,row,color,extra={}){return {x:selected(z.timestamp,ids),y:selected(z.values[key],ids),name:label,type:'scatter',mode:'lines',xaxis:row===1?'x':'x'+row,yaxis:row===1?'y':'y'+row,line:{color:color,width:2},customdata:selected(z.local_label,ids),hovertemplate:'%{customdata}<br>'+label+': %{y:.2f}<extra></extra>',connectgaps:false,...extra};}
function candidate(z,v,key,label,ids,row,color,extra={}){return trace({...z,values:v},key,label,ids,row,color,extra);}
function draw(){const day=daySelect.value,variant=variantSelect.value,dark=document.body.classList.contains('dark');
 const fg=dark?'#e2ebf4':'#152d45',bg=dark?'#14263b':'#ffffff',grid=dark?'#31465d':'#e1e8ef';
 document.getElementById('day-status').textContent=day==='2026-09-14'?'14/09 : étude connue — jamais utilisée comme preuve prospective.':'Journée de l’archive historique — pas une nouvelle émission prospective.';
 for(const [zone,z] of Object.entries(payload.zones)){const ids=z.day.map((d,i)=>d===day?i:-1).filter(i=>i>=0),v=z.variants[variant];if(!v)continue;
  const traces=[trace(z,'q10','NYX P10',ids,1,'#9eabba',{line:{width:0},showlegend:false}),trace(z,'q90','NYX P10–P90',ids,1,'#9eabba',{line:{width:0},fill:'tonexty',fillcolor:'rgba(126,149,174,.12)'}),
   candidate(z,v,'candidate_q10','Candidat P10',ids,1,'#23b4a5',{line:{width:0},showlegend:false}),candidate(z,v,'candidate_q90','Candidat P10–P90',ids,1,'#23b4a5',{line:{width:0},fill:'tonexty',fillcolor:'rgba(35,180,165,.13)'}),
   trace(z,'actual','Observé',ids,1,'#152d45',{line:{color:dark?'#fff':'#152d45',width:2.5}}),trace(z,'forecast','NYX figé',ids,1,'#4b80cf'),candidate(z,v,'candidate_forecast','Solar Ramp ('+variant+')',ids,1,'#119c8d'),trace(z,'benchmark_forecast','Storm figé — ex post',ids,1,'#b37ad6',{line:{color:'#b37ad6',dash:'dot',width:1.7}}),
   trace(z,'solarx_local_solar_gw','Solaire local',ids,2,'#d8a22a'),trace(z,'solarx_peer_solar_gw','Solaire voisins',ids,2,'#c85f2f'),
   trace(z,'solarx_local_solar_drop_1h_gwph','Retrait solaire local 1 h',ids,3,'#d8a22a'),trace(z,'solarx_peer_solar_drop_1h_gwph','Retrait solaire voisins 1 h',ids,3,'#c85f2f'),
   trace(z,'solarx_local_residual_gw','Charge résiduelle locale',ids,4,'#4b80cf'),trace(z,'solarx_peer_residual_gw','Charge résiduelle voisins',ids,4,'#7553ac'),trace(z,'solarx_local_pressure_proxy','Pression locale (proxy)',ids,5,'#bd5858'),
   candidate(z,v,'risk_probability','Probabilité sauvegardée',ids,6,'#119c8d'),candidate(z,v,'alert_threshold','Seuil alerte sauvegardé',ids,6,'#bd5858',{line:{color:'#bd5858',width:1.3,dash:'dash'}})];
  const layout={height:window.innerWidth<980?850:940,paper_bgcolor:bg,plot_bgcolor:bg,font:{color:fg,size:11},margin:{l:64,r:18,t:130,b:45},hovermode:'x unified',legend:{orientation:'h',x:0,y:1.17,font:{size:10}},uirevision:day+'-'+variant,annotations:[]};
  const titles=['Prix · EUR/MWh','Solaire · GW','Retrait solaire · GW/h','Charge résiduelle · GW','Pression · proxy','Probabilité'];
  for(let row=1;row<=6;row++){const bottom=1-row/6+.02,top=1-(row-1)/6-.025,ax=row===1?'':row;layout['xaxis'+ax]={anchor:'y'+ax,type:'date',matches:row===6?undefined:'x6',showticklabels:row===6,gridcolor:grid,tickformat:'%H:%M',title:row===6?'Heure UTC — infobulle en heure civile locale':undefined};layout['yaxis'+ax]={domain:[bottom,top],anchor:'x'+ax,gridcolor:grid,title:{text:titles[row-1],font:{size:10}},zerolinecolor:grid};if(row===6)layout['yaxis'+ax].range=[0,1];}
  const id='chart-'+zone;Plotly.react(id,traces,layout,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['select2d','lasso2d']});
  const phases=[...new Set(selected(v.evaluation_phase,ids).filter(Boolean))];document.getElementById('phase-'+zone).textContent=phases.length?'Phase sauvegardée : '+phases.join(', '):'Phase indisponible';
  const table=document.getElementById('hours-'+zone);table.replaceChildren();
  for(const i of ids){const row=document.createElement('tr');for(const value of [z.local_label[i],z.values.forecast[i],v.candidate_forecast[i],z.values.actual[i],v.risk_probability?.[i],v.applied_correction?.[i],v.selected_weight?.[i],v.gate_reason?.[i]]){const cell=document.createElement('td');cell.textContent=value==null?'—':typeof value==='number'?value.toFixed(3):String(value);row.append(cell);}table.append(row);}
 }
}
for(const zone of Object.keys(payload.zones)){const id='chart-'+zone;plotIds.push(id);}
draw();
for(const id of plotIds){document.getElementById(id).on('plotly_relayout',event=>{if(syncing)return;const key=Object.keys(event).find(k=>/^xaxis\d*\.range\[0\]$/.test(k));const reset=Object.keys(event).some(k=>/^xaxis\d*\.autorange$/.test(k)&&event[k]);if(!key&&!reset)return;let updates={};for(let row=1;row<=6;row++){const axis='xaxis'+(row===1?'':row);if(reset)updates[axis+'.autorange']=true;else{updates[axis+'.range']=[event[key],event[key.replace('[0]','[1]')]];updates[axis+'.autorange']=false;}}syncing=true;Promise.all(plotIds.filter(other=>other!==id).map(other=>Plotly.relayout(other,updates))).finally(()=>{syncing=false;});});}
daySelect.addEventListener('change',draw);variantSelect.addEventListener('change',draw);
document.getElementById('previous').addEventListener('click',()=>{daySelect.selectedIndex=Math.max(0,daySelect.selectedIndex-1);draw();});
document.getElementById('next').addEventListener('click',()=>{daySelect.selectedIndex=Math.min(days.length-1,daySelect.selectedIndex+1);draw();});
document.getElementById('case14').addEventListener('click',()=>{if(days.includes('2026-09-14')){daySelect.value='2026-09-14';draw();}});
document.getElementById('theme').addEventListener('click',()=>{document.body.classList.toggle('dark');draw();});
"""


def render_report(directory: Path, *, root: Path) -> Path:
    """Render one offline four-country index and verified standard country pages."""
    directory = _directory(directory, root)
    renderer_code_path = Path(__file__).resolve()
    renderer_code_sha256 = _digest(renderer_code_path)
    before = {name: _digest(directory / name) for name in INPUTS}
    panel, predictions = _validated_frames(pd.read_parquet(directory / "panel.parquet"),
                                           pd.read_parquet(directory / "predictions.parquet"))
    metrics, source_audit, manifest, literature = (json.loads((directory / name).read_text(encoding="utf-8"))
                                                  for name in INPUTS[2:])
    output = directory / "reports"
    for target in (output / "index.html", output / "report_audit.json", output / "standard"):
        if target.resolve() != target:
            raise ValueError("Solar Ramp report destinations must not be redirected.")
    links, standards = _standard_reports(predictions, source_audit, output / "standard", manifest, metrics)
    from plotly.offline import get_plotlyjs

    cards = []
    for zone in ZONES:
        if zone not in set(panel.zone):
            continue
        cards.append(f'<article class="country"><h3>{zone}</h3><p class="muted" id="phase-{zone}"></p>'
                     f'<div id="chart-{zone}" class="plot" aria-label="Courbes synchronisées {zone}"></div>'
                     '<details><summary>Prix et décisions horaires sauvegardés</summary><div class="table-wrap"><table><thead><tr>'
                     + ''.join('<th>' + label + '</th>' for label in ("Heure locale", "NYX", "Candidat", "Observé", "Probabilité", "Correction", "Poids", "Motif"))
                     + f'</tr></thead><tbody id="hours-{zone}"></tbody></table></div></details></article>')
    score_columns = [(key, label) for key, label in (("variant", "Variante"), ("zone", "Pays"), ("phase", "Phase"),
        ("n", "Heures"), ("mae", "MAE"), ("rmse", "RMSE"), ("bias", "Biais"), ("non_spike_mae", "MAE hors pics"),
        ("spike_mae", "MAE pics"), ("spike_underestimation", "Sous-estimation pics"), ("precision", "Précision"),
        ("recall", "Rappel"), ("pr_auc", "PR-AUC"), ("brier", "Brier"), ("log_loss", "Log-loss"),
        ("false_alerts_per_1000", "Fausses alertes / 1000 h"), ("episode_recall", "Rappel épisodes"),
        ("timing_mae_hours", "Erreur timing (h)"), ("coverage80", "Couverture P10–P90"))]
    score_columns.extend((key, key) for key in sorted({k for row in metrics.get("scores", []) for k in row if "pinball" in k}))
    report_audit = {"schema_version": 1, "input_sha256": before, "report_only": True,
                    "renderer_code_path": str(renderer_code_path), "renderer_code_sha256": renderer_code_sha256,
                    "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
                    "payload_encoding": "categorical dictionaries and exact shared arrays; no numeric rounding; all historical hours retained",
                    "model_fitted": False, "external_sources_fetched": False, "production_modified": False,
                    "activation_performed": False, "source_live_day_is_historical": True,
                    "prospective": _prospective(manifest), "standard_reports": standards,
                    "rows": len(panel), "variants": sorted(predictions.variant.unique().tolist()),
                    "price_quantiles": "saved only; no new interval created", "sources": source_audit}
    standard_links = ''.join('<a href="' + html.escape(link, quote=True) + '">' + zone + ' — rapport standard</a>' for zone, link in links.items())
    document = ('<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>NYX Solar Ramp — laboratoire quatre pays</title><style>' + _CSS + '</style></head><body>'
        '<header><div class="tag">Laboratoire isolé · NYX nucléaire + Kalman inchangé</div><h1>NYX Solar Ramp</h1>'
        '<p>Solaire, rampes et charge résiduelle · France, Allemagne, Belgique, Pays-Bas</p><nav>'
        '<a href="#summary">Décision</a><a href="#explorer">Profils synchronisés</a><a href="#metrics">Scores</a><a href="#standards">Rapports standards</a><a href="#audit">Traçabilité</a>'
        '<button id="theme" type="button">Jour / nuit</button></nav></header><main><div id="summary">'
        + _score_summary(metrics) + _methodology(manifest, metrics) + '</div><section id="explorer"><h2>Profils horaires — quatre pays, même journée</h2>'
        '<p class="muted">Zoom synchronisé entre pays et panneaux. Les trous restent vides. Les quantiles viennent des fichiers sauvegardés. '
        'Les indicateurs physiques ne constituent pas une attribution causale.</p><div class="controls">'
        '<label>Livraison<select id="delivery-day"></select></label><label>Courbe candidate<select id="variant"></select></label>'
        '<button id="previous">← Jour précédent</button><button id="next">Jour suivant →</button><button id="case14">Cas connu du 14 septembre</button></div>'
        '<p id="day-status" class="muted"></p><div class="grid">' + ''.join(cards) + '</div></section>'
        + _physical_analysis(metrics.get("physical_analysis"))
        + '<section id="metrics"><h2>Scores sauvegardés — comparaison et ablations</h2><p>Ces tables sont lues dans metrics.json, '
        'sans sélection ni recalcul du modèle par le rapport. Les phases « final_diagnostic » et « live_historical » restent rétrospectives.</p>'
        + _table(metrics.get("scores"), score_columns)
        + '<h3>Bootstrap apparié</h3>' + _table(metrics.get("paired_bootstrap"))
        + '<h3>Calibration par niveaux de risque</h3>' + _table(metrics.get("risk_bins"))
        + '<details><summary>Couverture, métriques et définitions complètes</summary><pre>' + html.escape(_json(metrics, indent=2)) + '</pre></details></section>'
        '<section id="standards"><h2>Même forme que le rapport opérationnel</h2><p>Moteur HTML inchangé : prix moyens, backtest, '
        'Statistics sur 365 jours, calendrier et profils horaires. La livraison historique séparée est exclue de ces scores. '
        'Ces pages ne sont pas des exports de production.</p>' + _standard_support(metrics, standards)
        + '<div class="links">' + standard_links + '</div>'
        + '<details><summary>Disponibilité et vérification des quatre rapports</summary><pre>' + html.escape(_json(standards, indent=2)) + '</pre></details></section>'
        '<section><h2>Références et hypothèses</h2>' + _literature(literature)
        + '<details><summary>Références originales</summary><pre>' + html.escape(_json(literature, indent=2)) + '</pre></details></section>'
        '<section id="audit"><h2>Traçabilité — sources, restrictions et artefacts</h2><p>'
        '<a href="../metrics.json">Métriques</a> · <a href="../source_audit.json">Audit sources</a> · <a href="../manifest.json">Manifeste</a> · '
        '<a href="report_audit.json">Audit du rapport</a></p><details><summary>Audit complet</summary><pre>'
        + html.escape(_json(report_audit, indent=2)) + '</pre></details></section></main>'
        '<footer>Rapport figé · aucun entraînement · aucune activation · aucune certification prospective</footer>'
        '<script>' + get_plotlyjs() + '</script><script id="solar-ramp-data" type="application/json">'
        + _script_json(_payload(panel, predictions)) + '</script><script>' + _JS + '</script></body></html>')
    if before != {name: _digest(directory / name) for name in INPUTS}:
        raise ValueError("Frozen input changed while rendering; report publication refused.")
    if _digest(renderer_code_path) != renderer_code_sha256:
        raise ValueError("Renderer code changed while rendering; report publication refused.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "index.html"
    path.write_text(document, encoding="utf-8")
    (output / "report_audit.json").write_text(_json(report_audit, indent=2), encoding="utf-8")
    return path


__all__ = ["render_report"]
