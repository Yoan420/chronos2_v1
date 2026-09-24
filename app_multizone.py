#!/usr/bin/env python
"""Streamlit control room for audited multi-zone day-ahead forecasts."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd

from chronos2_hourly.app_service import (
    APP_ZONES,
    ExistingForecastArchiveError,
    ForecastProcess,
    ForecastSkip,
    MixedForecastDeliveryDaysError,
    build_statistics_view,
    inspect_zone_statuses,
    launch_zone_forecast,
    list_run_artifacts,
    load_best_statistics_history,
    load_forecast_curve,
    load_latest_forecast_comparison,
    load_statistics_history,
    read_log_tail,
)
from chronos2_hourly.consolidated_report import (
    consolidated_report_filename,
    render_consolidated_forecast_report,
)
PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "chronos2_hourly_live_zones.yaml"
LIVE_ROOT = PROJECT_ROOT / "runs" / "live"
SEALED_BENCHMARK_ROOT = PROJECT_ROOT / "runs"
LOG_ROOT = LIVE_ROOT / "_app_logs"

PERFORMANCE_FREQUENCIES = {
    "H": "Horaire",
    "D": "Journalier",
    "W": "Hebdomadaire",
    "M": "Mensuel",
}
STATISTIC_LABELS = {
    "mae": "Mean Absolute Error",
    "rmse": "Root Mean Squared Error",
    "mape": "Mean Absolute Percentage Error",
    "explained_variance": "Explained Variance",
    "r2": "R²",
    "std_error": "Standard Deviation of Error",
    "correlation": "Correlation",
}
HIGHER_IS_BETTER = {"explained_variance", "r2", "correlation"}
ZONE_NAMES = {
    "FR": "France",
    "DE": "Allemagne",
    "BE": "Belgique",
    "NL": "Pays-Bas",
    "ES": "Espagne",
}


def _streamlit() -> Any:
    """Keep Streamlit optional for CLI imports and unit tests."""

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - exercised by the CLI user
        raise SystemExit(
            "Streamlit n'est pas installé. Exécutez d'abord : "
            "python -m pip install -r requirements_app.txt"
        ) from exc
    return st


def _init_state(st: Any) -> None:
    defaults = {
        "forecast_queue": [],
        "active_forecast": None,
        "forecast_jobs": [],
        "launch_options": {},
        "results_refresh_pending": False,
        "results_refresh_completed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def _start_next(st: Any) -> None:
    if st.session_state.active_forecast is not None:
        return
    queue = st.session_state.forecast_queue
    while queue:
        zone = queue.pop(0)
        options = dict(st.session_state.launch_options)
        try:
            handle = launch_zone_forecast(
                zone=zone,
                project_root=PROJECT_ROOT,
                registry_path=REGISTRY_PATH,
                log_dir=LOG_ROOT,
                python_executable=sys.executable,
                **options,
            )
        except ExistingForecastArchiveError as exc:
            st.session_state.forecast_jobs.append(
                {
                    "zone": zone,
                    "status": "Bloqué — archive existante invalide",
                    "error": str(exc),
                }
            )
            continue
        except Exception as exc:
            st.session_state.forecast_jobs.append(
                {"zone": zone, "status": "échec au démarrage", "error": str(exc)}
            )
            return
        st.session_state.forecast_jobs.append(handle)
        if isinstance(handle, ForecastSkip):
            continue
        st.session_state.active_forecast = handle
        return


def _advance_queue(st: Any) -> None:
    active: ForecastProcess | None = st.session_state.active_forecast
    if active is not None and active.return_code is not None:
        st.session_state.active_forecast = None
    _start_next(st)


def _refresh_results_after_queue(
    st: Any,
    *,
    cached_artifacts: Any,
    cached_statistics: Any,
    cached_comparison: Any | None = None,
    cached_performance: Any | None = None,
) -> bool:
    """Refresh report views once after the last queued process finishes.

    Fragment reruns do not refresh the report section rendered by the full
    app.  The two session flags make the full-app rerun one-shot; they are
    reset only when the user starts a new queue.
    """

    state = st.session_state
    if state.active_forecast is not None or state.forecast_queue:
        return False
    if not state.results_refresh_pending or state.results_refresh_completed:
        return False
    state.results_refresh_pending = False
    state.results_refresh_completed = True
    cached_artifacts.clear()
    cached_statistics.clear()
    if cached_comparison is not None:
        cached_comparison.clear()
    if cached_performance is not None:
        cached_performance.clear()
    st.rerun(scope="app")
    return True


def _status_table(statuses: list[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Pays": status.code,
                "Lançable": "Oui" if status.launchable else "Non",
                "Activé": "Oui" if status.enabled else "Non",
                "Production ready": "Oui" if status.production_ready else "Non",
                "Garde-fou / blocage": (
                    "Tous les contrôles sont validés"
                    if status.launchable
                    else (status.blockers[0] if status.blockers else "Bundle incomplet")
                ),
            }
            for status in statuses
        ]
    )


def _forecast_chart(frame: pd.DataFrame) -> alt.LayerChart:
    """Render the P10-P90 interval and P50 without mixing in evaluation data."""

    base = alt.Chart(frame).encode(
        x=alt.X("timestamp:T", title="Heure de livraison locale"),
    )
    interval = base.mark_area(
        color="#60A5FA",
        opacity=0.18,
    ).encode(
        y=alt.Y("P10:Q", title="Prix (EUR/MWh)"),
        y2="P90:Q",
        tooltip=(
            alt.Tooltip("timestamp:T", title="Livraison", format="%d/%m %H:%M"),
            alt.Tooltip("P10:Q", title="P10", format=".2f"),
            alt.Tooltip("P50:Q", title="P50", format=".2f"),
            alt.Tooltip("P90:Q", title="P90", format=".2f"),
        ),
    )
    median = base.mark_line(
        color="#60A5FA",
        strokeWidth=2.5,
        point=True,
    ).encode(
        y=alt.Y("P50:Q", title="Prix (EUR/MWh)"),
        tooltip=(
            alt.Tooltip("timestamp:T", title="Livraison", format="%d/%m %H:%M"),
            alt.Tooltip("P50:Q", title="P50", format=".2f"),
        ),
    )
    return (interval + median).properties(height=360).interactive()


def _comparison_chart(
    frame: pd.DataFrame,
    *,
    include_intervals: bool,
    mixed_delivery_days: bool,
) -> alt.Chart | alt.LayerChart:
    """Compare only operational forecast quantiles across selected zones."""

    title = (
        "Forecasts P50 — dates de livraison différentes (comparaison explicite)"
        if mixed_delivery_days
        else "Forecasts P50 — même jour de livraison"
    )
    base = alt.Chart(frame).encode(
        x=alt.X("timestamp_utc:T", title="Heure de livraison (UTC)"),
        color=alt.Color("zone:N", title="Pays"),
        detail=("zone:N", "delivery_day:N"),
    )
    line_encodings: dict[str, Any] = {
        "y": alt.Y("P50:Q", title="Prix (EUR/MWh)"),
        "tooltip": (
            alt.Tooltip("zone:N", title="Pays"),
            alt.Tooltip("delivery_day:N", title="Jour de livraison"),
            alt.Tooltip("local_delivery:N", title="Livraison locale"),
            alt.Tooltip("timestamp_utc:T", title="Livraison UTC", format="%d/%m %H:%M"),
            alt.Tooltip("P50:Q", title="P50", format=".2f"),
        ),
    }
    if mixed_delivery_days:
        line_encodings["strokeDash"] = alt.StrokeDash(
            "delivery_day:N", title="Jour de livraison"
        )
    median = base.mark_line(strokeWidth=2.8).encode(**line_encodings)
    if not include_intervals:
        return median.properties(height=390, title=title).interactive()
    interval = base.mark_area(opacity=0.09).encode(
        y=alt.Y("P10:Q", title="Prix (EUR/MWh)"),
        y2="P90:Q",
        tooltip=(
            alt.Tooltip("zone:N", title="Pays"),
            alt.Tooltip("delivery_day:N", title="Jour de livraison"),
            alt.Tooltip("local_delivery:N", title="Livraison locale"),
            alt.Tooltip("P10:Q", title="P10", format=".2f"),
            alt.Tooltip("P50:Q", title="P50", format=".2f"),
            alt.Tooltip("P90:Q", title="P90", format=".2f"),
        ),
    )
    return (interval + median).properties(height=390, title=title).interactive()


def _filter_statistics_dataset(
    dataset: Any,
    *,
    timezone_name: str,
    start_day: date,
    end_day: date,
) -> Any:
    """Return an immutable Statistics view restricted to local civil days."""

    local_day = dataset.frame["timestamp"].dt.tz_convert(timezone_name).dt.date
    selected = dataset.frame.loc[
        (local_day >= start_day) & (local_day <= end_day)
    ].copy()
    return replace(dataset, frame=selected.reset_index(drop=True))


def _performance_series(
    dataset: Any,
    *,
    timezone_name: str,
    start_day: date,
    end_day: date,
    frequency: str,
) -> pd.DataFrame:
    """Build the outright series shown in the daily-performance workspace."""

    selected = _filter_statistics_dataset(
        dataset,
        timezone_name=timezone_name,
        start_day=start_day,
        end_day=end_day,
    ).frame.copy()
    if selected.empty:
        return selected
    local = selected["timestamp"].dt.tz_convert(timezone_name)
    selected["local_timestamp"] = local
    selected["local_label"] = local.dt.strftime("%d/%m/%Y %H:%M %z")
    if frequency == "H":
        selected["plot_timestamp"] = selected["timestamp"]
        return selected

    local_naive = local.dt.tz_localize(None)
    if frequency == "D":
        selected["period_start_local"] = local_naive.dt.normalize()
    elif frequency == "W":
        selected["period_start_local"] = local_naive.dt.to_period("W-SUN").dt.start_time
    elif frequency == "M":
        selected["period_start_local"] = local_naive.dt.to_period("M").dt.start_time
    else:
        raise ValueError(f"Fréquence inconnue : {frequency}")

    grouped = (
        selected.groupby("period_start_local", sort=True)[
            ["actual", "candidate", "benchmark"]
        ]
        .mean()
        .reset_index()
    )
    localized = pd.DatetimeIndex(grouped["period_start_local"]).tz_localize(
        timezone_name,
        ambiguous="raise",
        nonexistent="raise",
    )
    grouped["plot_timestamp"] = localized.tz_convert("UTC")
    grouped["local_label"] = grouped["period_start_local"].dt.strftime("%d/%m/%Y")
    return grouped


def _weekend_bands(
    *,
    start_day: date,
    end_day: date,
    timezone_name: str,
) -> pd.DataFrame:
    """Build exact UTC bounds for weekend shading, including DST days."""

    zone = ZoneInfo(timezone_name)
    rows: list[dict[str, datetime]] = []
    current = start_day
    while current <= end_day:
        if current.weekday() >= 5:
            start_local = datetime.combine(current, time.min, tzinfo=zone)
            end_local = datetime.combine(
                current + timedelta(days=1), time.min, tzinfo=zone
            )
            rows.append(
                {
                    "start": start_local.astimezone(ZoneInfo("UTC")),
                    "end": end_local.astimezone(ZoneInfo("UTC")),
                }
            )
        current += timedelta(days=1)
    return pd.DataFrame(rows, columns=["start", "end"])


def _performance_chart(
    frame: pd.DataFrame,
    *,
    candidate_label: str,
    benchmark_label: str | None,
    weekend_bands: pd.DataFrame,
) -> alt.LayerChart:
    """Compare realised prices, our model and Storm with trading-style cues."""

    labels = {
        "actual": "Prix réalisé",
        "candidate": candidate_label,
        "benchmark": benchmark_label or "Storm",
    }
    value_columns = ["actual", "candidate"]
    if benchmark_label and bool(frame["benchmark"].notna().any()):
        value_columns.append("benchmark")
    long = frame.melt(
        id_vars=["plot_timestamp", "local_label"],
        value_vars=value_columns,
        var_name="series_key",
        value_name="value",
    )
    long["Série"] = long["series_key"].map(labels)
    ordered_labels = [labels[key] for key in value_columns]
    colors = ["#25313A", "#159DE4", "#64748B"][: len(ordered_labels)]
    dashes = [[1, 0], [1, 0], [5, 3]][: len(ordered_labels)]

    line = (
        alt.Chart(long)
        .mark_line(strokeWidth=2.4)
        .encode(
            x=alt.X(
                "plot_timestamp:T",
                title="Période de livraison",
                axis=alt.Axis(format="%d %b", labelOverlap=True),
            ),
            y=alt.Y("value:Q", title="Prix (EUR/MWh)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Série:N",
                title=None,
                scale=alt.Scale(domain=ordered_labels, range=colors),
                sort=ordered_labels,
            ),
            strokeDash=alt.StrokeDash(
                "Série:N",
                title=None,
                scale=alt.Scale(domain=ordered_labels, range=dashes),
                sort=ordered_labels,
            ),
            tooltip=(
                alt.Tooltip("local_label:N", title="Livraison locale"),
                alt.Tooltip("Série:N", title="Série"),
                alt.Tooltip("value:Q", title="EUR/MWh", format=".2f"),
            ),
        )
    )
    zero = alt.Chart(pd.DataFrame({"value": [0.0]})).mark_rule(
        color="#475569", opacity=0.55, strokeWidth=1
    ).encode(y="value:Q")
    layers: list[alt.Chart] = []
    if not weekend_bands.empty:
        layers.append(
            alt.Chart(weekend_bands)
            .mark_rect(color="#D9EFFB", opacity=0.55)
            .encode(x="start:T", x2="end:T")
        )
    layers.extend([zero, line])
    return alt.layer(*layers).properties(height=470).interactive()


def _overall_performance(
    dataset: Any,
    *,
    timezone_name: str,
    start_day: date,
    end_day: date,
    sample: str = "daily",
) -> dict[str, Any]:
    """Compute dashboard KPIs without changing the reporting metric contract."""

    selected = _filter_statistics_dataset(
        dataset,
        timezone_name=timezone_name,
        start_day=start_day,
        end_day=end_day,
    )
    frame = selected.frame
    paired_candidate = frame.dropna(subset=["actual", "candidate"])
    benchmark_mae = np.nan
    paired_benchmark = pd.DataFrame()
    if dataset.benchmark_column is not None:
        paired_benchmark = frame.dropna(subset=["actual", "candidate", "benchmark"])
        comparison_frame = paired_benchmark
        if not comparison_frame.empty:
            benchmark_mae = float(
                np.mean(
                    np.abs(comparison_frame["benchmark"] - comparison_frame["actual"])
                )
            )
    else:
        comparison_frame = paired_candidate
    candidate_mae = (
        float(
            np.mean(
                np.abs(comparison_frame["candidate"] - comparison_frame["actual"])
            )
        )
        if not comparison_frame.empty
        else np.nan
    )
    view = build_statistics_view(
        selected,
        timezone_name=timezone_name,
        sample=sample,
    )
    mae_row = view.summary.loc[view.summary["metric"] == "mae"]
    raw_win_rate = mae_row.iloc[0]["win_rate"] if not mae_row.empty else None
    win_rate = (
        float(raw_win_rate)
        if raw_win_rate is not None and np.isfinite(float(raw_win_rate))
        else np.nan
    )
    local = frame["timestamp"].dt.tz_convert(timezone_name) if not frame.empty else None
    return {
        "candidate_mae": candidate_mae,
        "benchmark_mae": benchmark_mae,
        "advantage": benchmark_mae - candidate_mae,
        "win_rate": win_rate,
        "hours": int(len(comparison_frame)),
        "start": local.min().date() if local is not None else None,
        "end": local.max().date() if local is not None else None,
        "view": view,
        "dataset": selected,
    }


def _zone_performance_table(
    datasets: dict[str, Any],
    *,
    timezone_by_zone: dict[str, str],
    start_day: date,
    end_day: date,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for zone in APP_ZONES:
        dataset = datasets.get(zone)
        if dataset is None:
            continue
        overview = _overall_performance(
            dataset,
            timezone_name=timezone_by_zone[zone],
            start_day=start_day,
            end_day=end_day,
        )
        scope = "Partiel" if dataset.scope_note and "parti" in dataset.scope_note.lower() else "Complet"
        rows.append(
            {
                "Pays": f"{zone} · {ZONE_NAMES.get(zone, zone)}",
                "Modèle": dataset.candidate_label,
                "MAE modèle": overview["candidate_mae"],
                "MAE Storm": overview["benchmark_mae"],
                "Avantage modèle": overview["advantage"],
                "Win rate quotidien": overview["win_rate"],
                "Heures": overview["hours"],
                "Périmètre": scope,
            }
        )
    return pd.DataFrame(rows)


def _statistics_period_table(
    overview: dict[str, Any],
    *,
    metric: str,
    sample: str,
    timezone_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    view = build_statistics_view(
        overview["dataset"],
        timezone_name=timezone_name,
        sample=sample,
    )
    candidate_col = f"candidate_{metric}"
    benchmark_col = f"benchmark_{metric}"
    outcome_col = f"outcome_{metric}"
    table = view.periods.loc[
        :, ["period", "n", candidate_col, benchmark_col, outcome_col]
    ].copy()
    if metric in HIGHER_IS_BETTER:
        table["advantage"] = table[candidate_col] - table[benchmark_col]
    else:
        table["advantage"] = table[benchmark_col] - table[candidate_col]
    table[outcome_col] = table[outcome_col].map(
        {"win": "Gagné", "tie": "Égalité", "loss": "Perdu"}
    )
    table = table.rename(
        columns={
            "period": "Période",
            "n": "N",
            candidate_col: "Modèle",
            benchmark_col: "Storm",
            "advantage": "Avantage modèle",
            outcome_col: "Résultat",
        }
    )
    summary = view.summary.rename(
        columns={
            "label": "Statistic",
            "candidate": "Modèle",
            "benchmark": "Storm",
            "wins": "Wins",
            "ties": "Ties",
            "losses": "Losses",
            "comparable_periods": "Périodes comparables",
            "win_rate": "Win rate vs Storm",
        }
    ).drop(columns=["metric"])
    return table.iloc[::-1].reset_index(drop=True), summary


def _render_performance_workspace(
    st: Any,
    *,
    statuses: list[Any],
    cached_best_statistics: Any,
) -> None:
    """Render the Storm-inspired daily model-performance workspace."""

    status_by_zone = {status.code: status for status in statuses}
    zones = [zone for zone in APP_ZONES if status_by_zone[zone].launchable]
    st.subheader("Performance quotidienne des modèles", anchor=False)
    st.caption(
        "Même logique de lecture que le dashboard Storm : filtres compacts, "
        "courbe outright, puis Statistics détaillées. Storm reste chargé "
        "uniquement après gel du forecast candidat."
    )

    with st.container(border=True):
        st.markdown("**FILTERING**")
        filter_cols = st.columns([1.25, 1.05, 1.0])
        variant = filter_cols[0].segmented_control(
            "Modèle",
            options=("autonomous", "production"),
            default="autonomous",
            format_func={
                "autonomous": "Autonome",
                "production": "Production",
            }.get,
            help="Production utilise le blend MKOnline validé uniquement pour FR/NL.",
            key="performance_variant",
        ) or "autonomous"
        focus_zone = filter_cols[1].selectbox(
            "Zone",
            options=zones,
            format_func=lambda zone: f"{zone} · {ZONE_NAMES.get(zone, zone)}",
            key="performance_zone",
        )
        frequency = filter_cols[2].segmented_control(
            "Fréquence du graphe",
            options=tuple(PERFORMANCE_FREQUENCIES),
            default="H",
            format_func=lambda value: value,
            key="performance_frequency",
        ) or "H"

        datasets: dict[str, Any] = {}
        performance_artifacts: dict[str, Any] = {}
        load_errors: list[str] = []
        for zone in zones:
            try:
                artifact, dataset = cached_best_statistics(
                    str(REGISTRY_PATH),
                    REGISTRY_PATH.stat().st_mtime_ns,
                    str(PROJECT_ROOT),
                    zone,
                    variant,
                )
            except Exception as exc:
                load_errors.append(f"{zone}: {exc}")
                continue
            performance_artifacts[zone] = artifact
            datasets[zone] = dataset
        if load_errors:
            st.warning(
                "Certaines zones sont temporairement indisponibles ; les autres restent "
                "consultables. " + " · ".join(load_errors)
            )
        if not datasets:
            st.error("Aucune archive Statistics auditée n'est disponible.")
            return
        if focus_zone not in datasets:
            replacement_zone = next(zone for zone in zones if zone in datasets)
            st.warning(
                f"Aucune Statistics exploitable pour {focus_zone}; affichage de "
                f"{replacement_zone} à la place."
            )
            focus_zone = replacement_zone

        local_days = datasets[focus_zone].frame["timestamp"].dt.tz_convert(
            status_by_zone[focus_zone].timezone
        ).dt.date
        minimum_day = min(local_days)
        maximum_day = max(local_days)
        requested_day: date | None = None
        raw_requested_day = st.query_params.get("date")
        if raw_requested_day:
            try:
                requested_day = date.fromisoformat(str(raw_requested_day))
            except ValueError:
                st.warning("Le paramètre URL 'date' est invalide ; la dernière date disponible est utilisée.")
        default_end = (
            min(maximum_day, max(minimum_day, requested_day))
            if requested_day is not None
            else maximum_day
        )
        default_start = max(minimum_day, default_end - timedelta(days=69))
        period = st.date_input(
            "Période",
            value=(default_start, default_end),
            min_value=minimum_day,
            max_value=maximum_day,
            key=f"performance_period_{focus_zone}",
        )
        if not isinstance(period, (tuple, list)) or len(period) != 2:
            st.info("Sélectionnez une date de début et une date de fin.")
            return
        start_day, end_day = period
        source_label = {
            "live_day_ahead": "forecast publié",
            "pit_replay": "reconstitution causale",
        }.get(
            performance_artifacts[focus_zone].archive_kind,
            performance_artifacts[focus_zone].archive_kind,
        )
        st.caption(
            f"Données disponibles pour {focus_zone} : {minimum_day.isoformat()} → "
            f"{maximum_day.isoformat()} · source : {source_label} "
            f"· graphe {PERFORMANCE_FREQUENCIES[frequency].lower()}."
        )
        if requested_day is not None and requested_day > maximum_day:
            st.caption(
                f"La date demandée ({requested_day.isoformat()}) n'est pas encore entièrement "
                f"réalisée ; affichage arrêté au {maximum_day.isoformat()}."
            )

    focus_dataset = datasets[focus_zone]
    focus_timezone = status_by_zone[focus_zone].timezone
    overview = _overall_performance(
        focus_dataset,
        timezone_name=focus_timezone,
        start_day=start_day,
        end_day=end_day,
    )
    if not overview["hours"]:
        st.warning("La période sélectionnée ne contient aucune observation exploitable.")
        return

    with st.container(horizontal=True):
        st.metric(
            "MAE modèle",
            f"{overview['candidate_mae']:.2f} EUR/MWh",
            border=True,
        )
        st.metric(
            "MAE Storm",
            (
                f"{overview['benchmark_mae']:.2f} EUR/MWh"
                if np.isfinite(overview["benchmark_mae"])
                else "N/A"
            ),
            border=True,
        )
        st.metric(
            "Avantage modèle",
            (
                f"{overview['advantage']:+.2f} EUR/MWh"
                if np.isfinite(overview["advantage"])
                else "N/A"
            ),
            help="Valeur positive : le modèle a une MAE inférieure à Storm.",
            border=True,
        )
        st.metric(
            "Win rate quotidien",
            (
                f"{100.0 * overview['win_rate']:.1f}%"
                if np.isfinite(overview["win_rate"])
                else "N/A"
            ),
            border=True,
        )
        st.metric("Heures évaluées", f"{overview['hours']:,}".replace(",", " "), border=True)

    st.markdown("**Vue d’ensemble multi-pays**")
    zone_table = _zone_performance_table(
        datasets,
        timezone_by_zone={zone: status_by_zone[zone].timezone for zone in datasets},
        start_day=start_day,
        end_day=end_day,
    )
    st.dataframe(
        zone_table,
        hide_index=True,
        width="stretch",
        column_config={
            "MAE modèle": st.column_config.NumberColumn(format="%.2f EUR/MWh"),
            "MAE Storm": st.column_config.NumberColumn(format="%.2f EUR/MWh"),
            "Avantage modèle": st.column_config.NumberColumn(format="%+.2f EUR/MWh"),
            "Win rate quotidien": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="percent"
            ),
            "Heures": st.column_config.NumberColumn(format="%d"),
        },
    )

    with st.container(border=True):
        st.subheader("Outright graph", anchor=False)
        chart_frame = _performance_series(
            focus_dataset,
            timezone_name=focus_timezone,
            start_day=start_day,
            end_day=end_day,
            frequency=frequency,
        )
        st.altair_chart(
            _performance_chart(
                chart_frame,
                candidate_label=focus_dataset.candidate_label,
                benchmark_label=focus_dataset.benchmark_label,
                weekend_bands=_weekend_bands(
                    start_day=start_day,
                    end_day=end_day,
                    timezone_name=focus_timezone,
                ),
            ),
            width="stretch",
        )
        st.caption(
            "Zones bleutées : samedis et dimanches. Survolez les courbes pour "
            "obtenir les valeurs ; utilisez la molette et le glisser-déposer pour zoomer."
        )

    with st.container(border=True):
        st.subheader("Statistics", anchor=False)
        controls = st.columns([1.6, 1.0])
        metric = controls[0].selectbox(
            "Statistic",
            options=tuple(STATISTIC_LABELS),
            format_func=STATISTIC_LABELS.get,
            key="performance_statistic",
        )
        sample = controls[1].selectbox(
            "Sample",
            options=("daily", "weekly", "monthly"),
            format_func={
                "daily": "Daily",
                "weekly": "Weekly",
                "monthly": "Monthly",
            }.get,
            key="performance_sample",
        )
        period_table, statistics_summary = _statistics_period_table(
            overview,
            metric=metric,
            sample=sample,
            timezone_name=focus_timezone,
        )
        metric_label = STATISTIC_LABELS[metric]
        period_table = period_table.rename(
            columns={
                "Modèle": f"{focus_dataset.candidate_label} · {metric_label}",
                "Storm": f"Storm · {metric_label}",
            }
        )
        value_columns = [
            f"{focus_dataset.candidate_label} · {metric_label}",
            f"Storm · {metric_label}",
        ]
        value_style_columns = [
            column
            for column in value_columns
            if column in period_table
            and bool(pd.to_numeric(period_table[column], errors="coerce").notna().any())
        ]
        styled = period_table.style
        if value_style_columns:
            styled = styled.background_gradient(
                subset=value_style_columns,
                cmap="RdYlBu" if metric in HIGHER_IS_BETTER else "RdYlBu_r",
            )
        if bool(
            pd.to_numeric(period_table["Avantage modèle"], errors="coerce")
            .notna()
            .any()
        ):
            styled = styled.background_gradient(
                subset=["Avantage modèle"],
                cmap="RdYlBu",
            )
        st.dataframe(
            styled,
            hide_index=True,
            width="stretch",
            column_config={
                "N": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption(
            "Avantage modèle > 0 signifie que le modèle bat Storm sur la période. "
            "Les égalités restent dans le dénominateur du win rate."
        )
        all_stats = st.expander("Toutes les Statistics", on_change="rerun")
        if all_stats.open:
            with all_stats:
                st.dataframe(
                    statistics_summary,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Win rate vs Storm": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="percent"
                        )
                    },
                )
    if focus_dataset.scope_note:
        st.info(focus_dataset.scope_note, icon=":material/info:")


def main() -> None:
    st = _streamlit()
    st.set_page_config(
        page_title="Chronos-2 · Forecast Control Room",
        page_icon=":material/bolt:",
        layout="wide",
    )
    _init_state(st)

    @st.cache_data(ttl=20, show_spinner=False)
    def cached_statuses(registry: str, modified_ns: int) -> list[Any]:
        del modified_ns
        return inspect_zone_statuses(registry)

    @st.cache_data(ttl=15, show_spinner=False)
    def cached_artifacts(live_root: str, benchmark_root: str) -> list[Any]:
        return list_run_artifacts(
            live_root,
            sealed_benchmark_root=benchmark_root,
            limit=100,
        )

    @st.cache_data(ttl=30, show_spinner=False)
    def cached_statistics(path: str, modified_ns: int, variant: str) -> Any:
        del modified_ns
        return load_statistics_history(path, variant=variant)

    @st.cache_data(ttl=60, max_entries=32, show_spinner=False)
    def cached_best_statistics(
        registry: str,
        registry_modified_ns: int,
        project_root: str,
        zone: str,
        variant: str,
    ) -> Any:
        del registry_modified_ns
        zone_statuses = inspect_zone_statuses(registry, zones=(zone,))
        if len(zone_statuses) != 1:
            raise ValueError(f"Statut de zone introuvable : {zone}")
        return load_best_statistics_history(
            zone_statuses[0],
            project_root=project_root,
            variant=variant,
        )

    @st.cache_data(ttl=30, show_spinner=False)
    def cached_forecast(
        path: str,
        modified_ns: int,
        timezone_name: str,
        variant: str,
    ) -> Any:
        del modified_ns
        return load_forecast_curve(
            path,
            timezone_name=timezone_name,
            variant=variant,
        )

    @st.cache_data(ttl=15, max_entries=32, show_spinner=False)
    def cached_comparison(
        registry: str,
        registry_modified_ns: int,
        project_root: str,
        zones: tuple[str, ...],
        allow_mixed_delivery_days: bool,
        variant: str,
    ) -> Any:
        del registry_modified_ns
        comparison_statuses = inspect_zone_statuses(registry, zones=zones)
        return load_latest_forecast_comparison(
            comparison_statuses,
            project_root=project_root,
            zones=zones,
            allow_mixed_delivery_days=allow_mixed_delivery_days,
            variant=variant,
        )

    @st.cache_data(max_entries=16, show_spinner=False)
    def cached_consolidated_report(
        _comparison: Any,
        archive_fingerprints: tuple[tuple[str, str, str, str], ...],
        include_intervals: bool,
        variant: str,
    ) -> bytes:
        # ``archive_fingerprints`` is the cache key.  The comparison itself is
        # deliberately excluded from Streamlit hashing: every miss still goes
        # through the renderer's complete archive/SHA-256 revalidation.
        del archive_fingerprints, variant
        return render_consolidated_forecast_report(
            _comparison,
            include_intervals=include_intervals,
        )

    st.title(":material/bolt: Chronos-2 · Day-ahead")
    st.caption(
        "Prévisions et performance multi-pays · Storm reste un benchmark "
        "d'évaluation uniquement et n'entre jamais dans les features du modèle."
    )

    try:
        statuses = cached_statuses(
            str(REGISTRY_PATH), REGISTRY_PATH.stat().st_mtime_ns
        )
    except Exception as exc:
        st.error(f"Le registre multi-zone ne peut pas être audité : {exc}")
        st.stop()
    status_by_zone = {status.code: status for status in statuses}

    workspace = st.segmented_control(
        "Espace",
        options=("performance", "operations"),
        default="performance",
        format_func={
            "performance": "Performance des modèles",
            "operations": "Runs et rapports",
        }.get,
        label_visibility="collapsed",
        key="main_workspace",
    ) or "performance"
    if workspace == "performance":
        @st.fragment(run_every="2s")
        def poll_hidden_forecast_queue() -> None:
            if not (
                st.session_state.active_forecast
                or st.session_state.forecast_queue
                or st.session_state.results_refresh_pending
            ):
                return
            _advance_queue(st)
            _refresh_results_after_queue(
                st,
                cached_artifacts=cached_artifacts,
                cached_statistics=cached_statistics,
                cached_comparison=cached_comparison,
                cached_performance=cached_best_statistics,
            )

        poll_hidden_forecast_queue()
        _render_performance_workspace(
            st,
            statuses=statuses,
            cached_best_statistics=cached_best_statistics,
        )
        return

    st.subheader("État des pays et garde-fous", anchor=False)
    st.dataframe(
        _status_table(statuses),
        hide_index=True,
        width="stretch",
    )
    blocked = [status for status in statuses if not status.launchable]
    if blocked:
        with st.expander("Pourquoi certains pays sont-ils désactivés ?"):
            for status in blocked:
                st.markdown(f"**{status.code}**")
                for blocker in status.blockers:
                    st.write(f"- {blocker}")

    st.subheader("Lancer les forecasts", anchor=False)
    launchable = [zone for zone in APP_ZONES if status_by_zone[zone].launchable]
    left, right = st.columns([2, 1])
    with left:
        run_all = st.toggle(
            "Tous les pays disponibles",
            help="Les bundles non validés restent exclus automatiquement.",
        )
        selected = st.pills(
            "Pays",
            options=launchable,
            default=launchable[:1],
            selection_mode="multi",
            disabled=run_all,
        )
    with right:
        delivery_day = st.date_input(
            "Jour de livraison",
            value=date.today() + timedelta(days=1),
        )

    advanced = st.expander("Options avancées", on_change="rerun")
    if advanced.open:
        with advanced:
            col1, col2, col3 = st.columns(3)
            device = col1.selectbox("Device", ("auto", "cpu", "cuda"), index=0)
            workers = int(col2.number_input("Workers", 1, 64, 4, 1))
            threads = int(col3.number_input("Threads", 1, 64, 4, 1))
            local_files_only = st.toggle(
                "Modèles Hugging Face en cache local uniquement",
                value=True,
            )
    else:
        device = "auto"
        workers = 4
        threads = 4
        local_files_only = True

    chosen = launchable if run_all else selected
    busy = bool(
        st.session_state.active_forecast
        or st.session_state.forecast_queue
    )
    if st.button(
        "Lancer le forecast",
        icon=":material/play_arrow:",
        type="primary",
        disabled=busy or not chosen,
        width="stretch",
    ):
        # Sequential execution avoids competing GPU jobs and shared-cache
        # races. The queue advances from the lightweight polling fragment.
        st.session_state.forecast_queue = list(chosen)
        st.session_state.launch_options = {
            "delivery_day": delivery_day,
            "device": device,
            "threads": threads,
            "workers": workers,
            "local_files_only": local_files_only,
        }
        st.session_state.results_refresh_pending = True
        st.session_state.results_refresh_completed = False
        _start_next(st)
        st.rerun()

    @st.fragment(run_every="2s")
    def render_run_monitor() -> None:
        _advance_queue(st)
        active: ForecastProcess | None = st.session_state.active_forecast
        pending = list(st.session_state.forecast_queue)
        _refresh_results_after_queue(
            st,
            cached_artifacts=cached_artifacts,
            cached_statistics=cached_statistics,
            cached_comparison=cached_comparison,
            cached_performance=cached_best_statistics,
        )
        if active is None and not pending and not st.session_state.forecast_jobs:
            st.info("Aucun forecast lancé depuis cette session.")
            return
        if active is not None:
            st.info(
                f"{active.zone} est en cours · PID {active.process.pid}"
                + (f" · file d'attente : {', '.join(pending)}" if pending else "")
            )
            st.code(read_log_tail(active.log_path), language="text")
        elif not pending:
            st.success("La file de forecasts est terminée.")
        rows = []
        for job in st.session_state.forecast_jobs:
            if isinstance(job, ForecastProcess):
                return_code = job.return_code
                rows.append(
                    {
                        "Pays": job.zone,
                        "État": "En cours" if return_code is None else (
                            "Terminé" if return_code == 0 else "Échec"
                        ),
                        "Code retour": return_code,
                        "Log": str(job.log_path),
                    }
                )
            elif isinstance(job, ForecastSkip):
                rows.append(
                    {
                        "Pays": job.zone,
                        "État": job.status,
                        "Code retour": job.return_code,
                        "Log": str(job.archive_path),
                    }
                )
            else:
                rows.append(
                    {
                        "Pays": job.get("zone"),
                        "État": job.get("status"),
                        "Code retour": None,
                        "Log": job.get("error"),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    render_run_monitor()

    st.subheader("Comparaison des derniers forecasts", anchor=False)
    st.caption(
        "Sélectionnez de 1 à 5 pays. Le graphe contient uniquement les "
        "quantiles du forecast opérationnel audité ; aucune donnée Storm."
    )
    comparison_options = [
        zone for zone in APP_ZONES if status_by_zone[zone].launchable
    ]
    comparison_variant = st.segmented_control(
        "Version du modèle",
        options=("autonomous", "production"),
        default="autonomous",
        format_func={
            "autonomous": "Autonome · sans MKOnline",
            "production": "Production · blend validé FR/NL",
        }.get,
        help=(
            "La version autonome utilise uniquement Chronos-2 et son correcteur. "
            "La version Production ajoute le blend MKOnline uniquement pour FR/NL."
        ),
    ) or "autonomous"
    comparison_left, comparison_right = st.columns([2, 1])
    with comparison_left:
        comparison_zones = st.pills(
            "Pays à comparer",
            options=comparison_options,
            default=comparison_options,
            selection_mode="multi",
            key="comparison_zones",
        )
    with comparison_right:
        include_intervals = st.toggle(
            "Afficher les bandes P10–P90",
            value=False,
            help=(
                "Les bandes restent masquées par défaut afin de garder les "
                "courbes P50 lisibles lorsque plusieurs pays sont sélectionnés."
            ),
        )
        allow_mixed_delivery_days = st.toggle(
            "Autoriser explicitement des dates différentes",
            value=False,
            help=(
                "Désactivé par défaut : la comparaison est bloquée si les "
                "derniers forecasts ne portent pas sur le même jour."
            ),
        )

    if not comparison_zones:
        st.info("Sélectionnez au moins un pays pour afficher la comparaison.")
    else:
        comparison = None
        try:
            comparison = cached_comparison(
                str(REGISTRY_PATH),
                REGISTRY_PATH.stat().st_mtime_ns,
                str(PROJECT_ROOT),
                tuple(comparison_zones),
                allow_mixed_delivery_days,
                comparison_variant,
            )
        except MixedForecastDeliveryDaysError as exc:
            mapping = " · ".join(
                f"{zone}: {day}" for zone, day in exc.delivery_days.items()
            )
            st.warning(
                "Comparaison bloquée : les derniers jours de livraison "
                f"diffèrent ({mapping}). Activez l'autorisation explicite "
                "ci-dessus pour les afficher sans masquer cet écart."
            )
        except Exception as exc:
            st.error(f"Comparaison multi-pays refusée : {exc}")

        if comparison is not None:
            if comparison.mixed_delivery_days:
                mapping = " · ".join(
                    f"{zone}: {day}"
                    for zone, day in comparison.delivery_days.items()
                )
                st.warning(
                    "Comparaison multi-dates explicitement activée. "
                    f"Les courbes ne décrivent pas le même jour : {mapping}."
                )
            with st.container(border=True):
                comparison_metrics = st.columns(3)
                comparison_metrics[0].metric(
                    "Pays comparés", len(comparison.archives)
                )
                comparison_metrics[1].metric(
                    "Jour(s) de livraison",
                    " · ".join(sorted(set(comparison.delivery_days.values()))),
                )
                comparison_metrics[2].metric(
                    "Amplitude P50",
                    (
                        f"{comparison.frame['P50'].min():.2f} à "
                        f"{comparison.frame['P50'].max():.2f} EUR/MWh"
                    ),
                )
                st.altair_chart(
                    _comparison_chart(
                        comparison.frame,
                        include_intervals=include_intervals,
                        mixed_delivery_days=comparison.mixed_delivery_days,
                    ),
                    width="stretch",
                )
                st.caption(
                    ("Version autonome sans MKOnline. " if comparison.variant == "autonomous" else "Version de production. ")
                    + "Axe commun en UTC ; le tooltip conserve l'heure locale et "
                    "la timezone de chaque marché. Les archives, identités, dates, "
                    "timelines DST et checksums sont validés avant affichage."
                )
                try:
                    archive_fingerprints = tuple(
                        (
                            archive.zone,
                            archive.delivery_day,
                            archive.forecast_sha256,
                            archive.checksum_manifest_sha256,
                        )
                        for archive in comparison.archives
                    )
                    consolidated_html = cached_consolidated_report(
                        comparison,
                        archive_fingerprints,
                        include_intervals,
                        comparison.variant,
                    )
                except Exception as exc:
                    st.error(f"Export HTML consolidé refusé : {exc}")
                else:
                    st.download_button(
                        "Télécharger le rapport HTML consolidé",
                        icon=":material/download:",
                        data=consolidated_html,
                        file_name=consolidated_report_filename(comparison),
                        mime="text/html",
                        key="download_consolidated_forecasts",
                    )

    st.subheader("Derniers runs et rapports", anchor=False)
    artifacts = cached_artifacts(str(LIVE_ROOT), str(SEALED_BENCHMARK_ROOT))
    if not artifacts:
        st.info("Aucun run live trouvé.")
        return
    artifact = st.selectbox(
        "Artefact à consulter",
        options=artifacts,
        format_func=lambda item: item.label,
    )
    details = st.columns(3)
    details[0].metric("Pays", artifact.zone)
    details[1].metric("Type", artifact.kind)
    details[2].metric(
        "Dernière modification",
        artifact.modified_at.astimezone().strftime("%d/%m/%Y %H:%M"),
    )
    st.code(str(artifact.directory), language="text")
    artifact_variant = st.segmented_control(
        "Version consultée",
        options=("autonomous", "production"),
        default="autonomous",
        format_func={
            "autonomous": "Autonome · sans MKOnline",
            "production": "Production",
        }.get,
        key="artifact_forecast_variant",
    ) or "autonomous"
    if artifact.report_path is not None:
        with artifact.report_path.open("rb") as report_stream:
            st.download_button(
                "Télécharger le rapport archivé (production)",
                icon=":material/download:",
                data=report_stream.read(),
                file_name=artifact.report_path.name,
                mime="text/html",
            )

    if artifact.forecast_path is not None:
        st.subheader("Forecast day-ahead", anchor=False)
        try:
            forecast = cached_forecast(
                str(artifact.forecast_path),
                artifact.forecast_path.stat().st_mtime_ns,
                status_by_zone[artifact.zone].timezone,
                artifact_variant,
            ).frame
        except Exception as exc:
            st.error(f"Forecast illisible : {exc}")
        else:
            with st.container(border=True):
                kpis = st.columns(3)
                kpis[0].metric("P50 minimum", f"{forecast['P50'].min():.2f} EUR/MWh")
                kpis[1].metric("P50 moyen", f"{forecast['P50'].mean():.2f} EUR/MWh")
                kpis[2].metric("P50 maximum", f"{forecast['P50'].max():.2f} EUR/MWh")
                st.altair_chart(_forecast_chart(forecast), width="stretch")
                st.caption(
                    "La ligne représente la médiane P50 ; la zone bleue représente "
                    "l'intervalle probabiliste P10–P90. Storm n'est jamais injecté "
                    "dans ce graphique de prévision opérationnelle."
                )

    st.subheader("Statistics", anchor=False)
    if artifact.statistics_path is None:
        st.warning("Cet artefact ne contient pas statistics_history_hourly.csv.gz.")
        return
    try:
        dataset = cached_statistics(
            str(artifact.statistics_path),
            artifact.statistics_path.stat().st_mtime_ns,
            artifact_variant,
        )
    except Exception as exc:
        st.error(f"Statistics illisible : {exc}")
        return
    sample = st.segmented_control(
        "Échantillonnage",
        options=("daily", "weekly", "monthly"),
        default="daily",
        format_func={
            "daily": "Daily",
            "weekly": "Weekly",
            "monthly": "Monthly",
        }.get,
    ) or "daily"
    view = build_statistics_view(
        dataset,
        timezone_name=status_by_zone[artifact.zone].timezone,
        sample=sample,
    )
    if dataset.scope_note:
        st.info(dataset.scope_note)
    st.caption(
        f"Candidat : {dataset.candidate_label} · Benchmark : "
        f"{dataset.benchmark_label or 'non disponible'} · "
        "win rate = périodes gagnées / périodes comparables (ties incluses). "
        "La MAPE utilise |prix réalisé| au dénominateur et ignore uniquement "
        "les observations de valeur absolue ≤ 1e-9 EUR/MWh."
    )

    summary = view.summary.rename(
        columns={
            "label": "Statistic",
            "candidate": dataset.candidate_label,
            "benchmark": dataset.benchmark_label or "Storm",
            "wins": "Wins",
            "ties": "Ties",
            "losses": "Losses",
            "comparable_periods": "Périodes comparables",
            "win_rate": "Win rate vs Storm",
        }
    ).drop(columns=["metric"])
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "Win rate vs Storm": st.column_config.ProgressColumn(
                "Win rate vs Storm", min_value=0.0, max_value=1.0, format="percent"
            )
        },
    )

    latest = view.time_series.tail(24 * 14).rename(
        columns={
            "timestamp": "Livraison UTC",
            "actual": "Prix réalisé",
            "candidate": dataset.candidate_label,
            "benchmark": dataset.benchmark_label or "Storm",
        }
    )
    chart_columns = ["Prix réalisé", dataset.candidate_label]
    if dataset.benchmark_label:
        chart_columns.append(dataset.benchmark_label)
    st.line_chart(latest.set_index("Livraison UTC")[chart_columns])

    metric = st.selectbox(
        "Détail par période",
        options=list(view.summary["metric"]),
        format_func=lambda key: dict(
            zip(view.summary["metric"], view.summary["label"], strict=True)
        )[key],
    )
    period_columns = [
        "period",
        "n",
        f"candidate_{metric}",
        f"benchmark_{metric}",
        f"outcome_{metric}",
    ]
    st.dataframe(
        view.periods.loc[:, period_columns].rename(
            columns={
                "period": "Période",
                "n": "N",
                f"candidate_{metric}": dataset.candidate_label,
                f"benchmark_{metric}": dataset.benchmark_label or "Storm",
                f"outcome_{metric}": "Résultat",
            }
        ),
        hide_index=True,
        width="stretch",
    )


if __name__ == "__main__":
    main()
