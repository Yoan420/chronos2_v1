"""Report-only adapter for issued NYX/Test2 routes; never fit or change prices.

The standard renderer is reused without modifying its production implementation.
Legacy labels and its generic attribution paragraph are adapted explicitly here.
Only supplied historical predictions are evaluated; today's prices cannot enter.
"""

from __future__ import annotations

from copy import deepcopy
import html
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_modular.common import ZoneData, ZoneRunResult
from chronos2_modular.metrics import compute_metrics, metric_breakdowns
from chronos2_modular.report import html_table, write_html_report
from .reporting import _replace_report_labels


TIMEZONES = {"DE": "Europe/Berlin", "NL": "Europe/Amsterdam",
             "BE": "Europe/Brussels", "FR": "Europe/Paris"}
QUANTILES = ("q10", "q50", "q90")
LABEL = "Hybride NYX / Test2"
BASELINE_LABEL = "NYX de référence"
_PROTECTED_DIRECTORIES = {"autonomous", "kalman", "blend", "nuclear_autonomous",
                          "nuclear_kalman", "kalman_weather", "kalman_hybrid"}
_SIGNALS = ("own_joint_deficit", "own_residual_stress", "nyx_daily_peak_gap")


def _day_index(day: Any, timezone: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(day)
    return pd.date_range(start.tz_localize(timezone),
                         (start + pd.Timedelta(days=1)).tz_localize(timezone),
                         freq="h", inclusive="left").tz_convert("UTC")


def _frame(value: pd.DataFrame, *, zone: str, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise ValueError(f"{name}: a nonempty DataFrame is required.")
    frame = value.copy(deep=True)
    time_columns = [key for key in ("delivery_start_utc", "timestamp_utc", "timestamp") if key in frame]
    source = frame[time_columns[0]] if time_columns else frame.index
    if not time_columns and not isinstance(source, pd.DatetimeIndex):
        raise ValueError(f"{name}: UTC timestamps are required.")
    index = pd.DatetimeIndex(pd.to_datetime(source, errors="raise"))
    if index.tz is None:
        raise ValueError(f"{name}: timestamps must be timezone-aware.")
    index = index.tz_convert("UTC")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{name}: invalid, duplicated or unsorted timestamps.")
    for column in time_columns[1:]:
        other = pd.DatetimeIndex(pd.to_datetime(frame[column], errors="raise"))
        if other.tz is None or not other.tz_convert("UTC").equals(index):
            raise ValueError(f"{name}: timestamp columns disagree.")
    if time_columns and isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.tz is None or not frame.index.tz_convert("UTC").equals(index):
            raise ValueError(f"{name}: timestamp index disagrees with column.")
    frame.index = index
    if "zone" in frame and not frame.zone.astype(str).eq(zone).all():
        raise ValueError(f"{name}: wrong zone.")
    if "selected_test2" not in frame or not pd.api.types.is_bool_dtype(frame.selected_test2.dtype):
        raise ValueError(f"{name}: selected_test2 must be boolean.")
    if frame.selected_test2.isna().any():
        raise ValueError(f"{name}: missing route decision.")
    arrays = {}
    for model in ("nyx", "test2", "hybrid"):
        columns = [f"{model}__{q}" for q in QUANTILES]
        if not set(columns).issubset(frame):
            raise ValueError(f"{name}: missing {model} quantiles.")
        arrays[model] = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(arrays[model]).all() or (np.diff(arrays[model], axis=1) < 0).any():
            raise ValueError(f"{name}: {model} quantiles must be finite and ordered.")
    expected = np.where(frame.selected_test2.to_numpy(bool)[:, None], arrays["test2"], arrays["nyx"])
    if not np.array_equal(arrays["hybrid"], expected):
        raise ValueError(f"{name}: hybrid quantiles must exactly copy the selected source.")
    if "selected_model" in frame:
        labels = np.where(frame.selected_test2, "Test2", "NYX")
        if not np.array_equal(frame.selected_model.astype(str).to_numpy(), labels):
            raise ValueError(f"{name}: selected_model disagrees with selected_test2.")
    return frame


def _prediction(frame: pd.DataFrame, *, model: str, timezone: str,
                historical: bool) -> pd.DataFrame:
    """Legacy deciles are interpolation only; original P10/P50/P90 stay exact."""
    raw = frame[[f"{model}__{q}" for q in QUANTILES]].to_numpy(float)
    output = pd.DataFrame({"timestamp": frame.index.tz_convert(timezone)})
    for decile in range(1, 10):
        weight = (decile - 1) / 4 if decile <= 5 else (decile - 5) / 4
        left, right = (0, 1) if decile <= 5 else (1, 2)
        output[f"q{decile * 10:02d}"] = raw[:, left] + weight * (raw[:, right] - raw[:, left])
    for column, values in zip(QUANTILES, raw.T):
        output[column] = values
    output["point"] = raw[:, 1]
    if historical:
        output["actual"] = frame.actual.to_numpy(float)
        origin_column = next((c for c in ("forecast_origin_utc", "origin_timestamp") if c in frame), None)
        if origin_column:
            origins = pd.DatetimeIndex(pd.to_datetime(frame[origin_column], errors="raise"))
            if origins.tz is None or origins.hasnans or (origins.tz_convert("UTC") >= frame.index).any():
                raise ValueError("Historical forecast origins must be aware and precede delivery.")
            output["origin_timestamp"] = origins.tz_convert(timezone)
        else:
            # Do not fabricate a D-1 issue time. Ramp-by-origin metrics stay unavailable.
            output["origin_timestamp"] = pd.Series(pd.NaT, index=output.index, dtype=f"datetime64[ns, {timezone}]")
        days = output.timestamp.dt.date
        output["horizon_step"] = output.groupby(days, sort=False).cumcount() + 1
    return output


def _prepare(history: pd.DataFrame, forecast: pd.DataFrame, *, zone: str,
             delivery_day: Any, output_path: Path, metadata: Mapping[str, Any]):
    code = str(zone).upper()
    if code not in TIMEZONES:
        raise ValueError("Only FR, DE, BE and NL are supported.")
    timezone = TIMEZONES[code]
    day = pd.Timestamp(delivery_day)
    if day.tzinfo is not None or day != day.normalize():
        raise ValueError("delivery_day must be a timezone-free civil date.")
    past = _frame(history, zone=code, name="history")
    future = _frame(forecast, zone=code, name="forecast")
    expected = _day_index(day, timezone)
    if not future.index.equals(expected):
        raise ValueError("Forecast must contain the exact complete physical delivery day.")
    if "actual" in future and future.actual.notna().any():
        raise ValueError("Future actual observations are forbidden in a live report.")
    if (past.index >= expected[0]).any():
        raise ValueError("History must strictly precede the delivery day.")
    local_days = pd.Index(past.index.tz_convert(timezone).date)
    first, last = min(local_days), max(local_days)
    expected_history = pd.date_range(pd.Timestamp(first).tz_localize(timezone),
        (pd.Timestamp(last) + pd.Timedelta(days=1)).tz_localize(timezone),
        freq="h", inclusive="left").tz_convert("UTC")
    if not past.index.equals(expected_history):
        raise ValueError("History must consist of complete contiguous civil days, including DST.")
    if "actual" not in past:
        raise ValueError("History requires actual observations.")
    actual = pd.to_numeric(past.actual, errors="raise").to_numpy(float)
    if np.isinf(actual).any() or not np.isfinite(actual).any():
        raise ValueError("History needs finite observed support; infinities are forbidden.")
    past["actual"] = actual
    supplied_rows = len(past)
    # The standard Statistics panel uses at most 365 days; all panels here share it.
    cutoff = pd.Timestamp(last) - pd.Timedelta(days=364)
    past = past.loc[np.asarray(local_days >= cutoff.date())].copy()
    first = past.index.tz_convert(timezone)[0].date()
    native_all = _prediction(past, model="hybrid", timezone=timezone, historical=True)
    baseline_all = _prediction(past, model="nyx", timezone=timezone, historical=True)
    finite = np.isfinite(past.actual.to_numpy(float))
    native, baseline = native_all.loc[finite].copy(), baseline_all.loc[finite].copy()
    by_horizon, by_hour = metric_breakdowns(native, baseline, 150.)
    target = pd.Series(past.actual.to_numpy(float), index=past.index.tz_convert(timezone), name="actual")
    combined = pd.concat([past, future], axis=0)
    signals = [name for name in _SIGNALS if name in past and name in future]
    covariates = combined[signals].apply(pd.to_numeric, errors="raise")
    covariates.index = combined.index.tz_convert(timezone)
    coverage = pd.DataFrame({"alias": signals,
        "coverage_after_fill": [float(covariates[c].notna().mean()) for c in signals]})
    manifest = pd.DataFrame([{"alias": key, "source": "artefacts fournis et figés",
        "role": "signal de routage affiché, pas attribution du prix"} for key in signals],
        columns=["alias", "source", "role"])
    data = ZoneData(code, timezone, "h", target, covariates, covariates.copy(), [], coverage, manifest,
                    {"report_only": True, "production_modified": False})
    result = ZoneRunResult(code, compute_metrics(native, 150.), compute_metrics(baseline, 150.),
        native, baseline, by_horizon, by_hour,
        _prediction(future, model="hybrid", timezone=timezone, historical=False),
        _prediction(future, model="nyx", timezone=timezone, historical=False), data, output_path.parent)
    result.statistics_candidate = native_all
    result.statistics_candidate_label = LABEL
    result.statistics_scope_note = (
        f"Historique hybride fourni du {first} au {last} : {(last-first).days+1} jours civils, "
        f"{len(past)} heures physiques, {int(finite.sum())} observations évaluables communes à NYX. "
        "Livraison du jour exclue de toutes les statistiques. Aucun gain live n’est estimé. "
        "Les origines absentes ne sont pas reconstruites."
    )
    audit = {"schema_version": 1, "zone": code, "timezone": timezone,
        "delivery_day": str(day.date()), "forecast_hours": len(future),
        "history_start_day": str(first), "history_end_day": str(last),
        "history_days": (last-first).days+1, "history_rows": len(past),
        "supplied_history_rows": supplied_rows, "paired_observed_rows": int(finite.sum()),
        "history_limit_days": 365, "future_actual_included": False,
        "live_excluded_from_statistics": True, "selected_test2_hours": int(future.selected_test2.sum()),
        "production_modified": False, "report_only": True, "price_attribution_available": False,
        "quantiles": "Original selected P10/P50/P90 copied exactly; intermediate deciles only approximate CRPS.",
        "metadata": deepcopy(dict(metadata)),
        "presentation_adaptations": ["native and baseline labels", "exploratory live scope", "routing signals label",
            "unavailable final-price attribution", "exact historical window instead of implicit 365 days"]}
    return result, future, audit


def _banner(future: pd.DataFrame, result: ZoneRunResult, audit: Mapping[str, Any]) -> str:
    metadata = audit["metadata"]
    escaped_meta = html.escape(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    rows = pd.DataFrame({"Heure locale (offset)": [t.isoformat() for t in future.index.tz_convert(result.zone_data.timezone)],
        "Route": np.where(future.selected_test2, "Test2", "NYX"),
        "NYX P50": future.nyx__q50.to_numpy(), "Test2 P50": future.test2__q50.to_numpy(),
        "Final P10": future.hybrid__q10.to_numpy(), "Final P50": future.hybrid__q50.to_numpy(),
        "Final P90": future.hybrid__q90.to_numpy(),
        "Décision": future["reason"].astype(str).to_numpy() if "reason" in future else "non fournie"})
    for key in _SIGNALS:
        if key in future:
            rows[key] = future[key].to_numpy()
    payload = json.dumps(audit, ensure_ascii=False, default=str, allow_nan=False).replace("<", "\\u003c")
    return ('<section data-report-section="nyx-live-protocol"><h2>Routage NYX / Test2 — livraison exploratoire</h2>'
        '<p>La sortie finale copie les trois quantiles de la route émise, sans mélange de médianes, '
        'sans écrêtage et sans hausse forcée. NYX est la base du routage, pas un benchmark indépendant. '
        'Test2 désigne l’expert sans les quatre maxima journaliers.</p>'
        '<p>Statut PIT : disponibilité prospective des sources non certifiée par ce rapport. '
        'Le protocole, l’as-of et le warmup déclarés ci-dessous restent des métadonnées de provenance, '
        'pas une preuve de disponibilité historique ni une validation de promotion.</p>'
        f'<p>{html.escape(result.statistics_scope_note)}</p>'
        '<p>P10/P50/P90 sont les valeurs enregistrées ; les déciles intermédiaires sont interpolés '
        'uniquement pour le CRPS approximatif du moteur standard. Aucune attribution finale '
        'Chronos/correcteur/Kalman n’est reprise.</p>'
        f'<details><summary>Protocole, as-of, warmup et références</summary><pre>{escaped_meta}</pre></details></section>'
        '<section data-report-section="nyx-routing-decisions"><h2>Décisions horaires et quantiles finaux</h2>'
        + html_table(rows) + '</section>'
        f'<script type="application/json" id="nyx-live-report-audit">{payload}</script>')


def render_live_report(history: pd.DataFrame, forecast: pd.DataFrame, *, zone: str,
                       delivery_day: Any, output_path: str | Path,
                       metadata: Mapping[str, Any]) -> Path:
    """Write one new standard HTML; inputs and incumbent exports remain untouched.

    Both frames contain NYX/Test2/hybrid ``__q10/__q50/__q90`` and a boolean
    ``selected_test2``. History requires ``actual``; forecast may only have null
    actuals. Timestamps are aware, unique and ordered (index or UTC column).
    One or more complete historical days and one exact local delivery day are
    required. No issue time is inferred when absent. Missing historical actuals
    remain missing; paired metrics exclude them. No CSV or model is written.
    """
    path = Path(output_path).expanduser().resolve()
    if path.suffix.lower() != ".html" or set(p.lower() for p in path.parts) & _PROTECTED_DIRECTORIES:
        raise ValueError("Use a separate HTML namespace, never an incumbent export directory.")
    if path.exists():
        raise FileExistsError(f"Report output already exists: {path}")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping.")
    result, future, audit = _prepare(history, forecast, zone=zone, delivery_day=delivery_day,
                                     output_path=path, metadata=metadata)
    banner = _banner(future, result, audit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nyx-report-", dir=path.parent) as temporary:
        staged = Path(temporary) / "report.html"
        write_html_report([result], {"report": {"title": f"{result.zone} {audit['delivery_day']} — {LABEL}",
                                                "forecast_history_hours": 168}}, staged)
        _replace_report_labels(staged, native_label=LABEL, baseline_label=BASELINE_LABEL)
        document = staged.read_text(encoding="utf-8")
        attribution = ('<div class="variable-attribution" data-report-section="variable-attribution" '
            'data-attribution-status="unavailable"><h3>Attribution du prix final indisponible</h3>'
            '<p>Les décisions et signaux affichés ne sont pas des contributions causales. '
            'Aucune attribution du modèle de base n’est présentée comme celle de l’hybride.</p></div>')
        pattern = (r'<div class="variable-attribution" data-report-section="variable-attribution"\s+'
                   r'data-attribution-status="unavailable">.*?</div>\s*</div>')
        document, count = re.subn(pattern, lambda _: attribution, document, count=1, flags=re.DOTALL)
        if count != 1:
            raise ValueError("Standard attribution section changed; report adaptation must be reviewed.")
        document = document.replace("Prévision opérationnelle du", "Livraison exploratoire du")
        document = document.replace("Prévision Day-Ahead réelle", "Prévision Day-Ahead — quantiles émis")
        document = document.replace("covariables actives", "signaux de routage affichés")
        document = document.replace("Variables d’entrée", "Signaux de routage fournis (pas toutes les entrées du modèle)")
        document = document.replace("365 derniers jours", f"fenêtre disponible de {audit['history_days']} jours")
        if document.count("<main>") != 1:
            raise ValueError("Standard report main section changed.")
        document = document.replace("<main>", "<main>" + banner, 1)
        # Exclusive creation protects another invocation's report as well as incumbents.
        with path.open("x", encoding="utf-8") as handle:
            handle.write(document)
    return path


__all__ = ["render_live_report"]
