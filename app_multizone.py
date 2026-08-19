#!/usr/bin/env python
"""Streamlit control room for audited multi-zone day-ahead forecasts."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Any

import altair as alt
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
    load_forecast_curve,
    load_latest_forecast_comparison,
    load_statistics_history,
    read_log_tail,
)
from chronos2_hourly.consolidated_report import (
    consolidated_report_filename,
    render_consolidated_forecast_report,
)
from chronos2_hourly.market_coupling import (
    MarketCouplingDataError,
    render_market_coupling_panel,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "chronos2_hourly_live_zones.yaml"
LIVE_ROOT = PROJECT_ROOT / "runs" / "live"
SEALED_BENCHMARK_ROOT = PROJECT_ROOT / "runs"
LOG_ROOT = LIVE_ROOT / "_app_logs"
# Conserved for a later iteration, but intentionally hidden while model
# performance and the Storm win-rate are the product priority.
ENABLE_MARKET_COUPLING = False


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
    def cached_statistics(path: str, modified_ns: int) -> Any:
        del modified_ns
        return load_statistics_history(path)

    @st.cache_data(ttl=30, show_spinner=False)
    def cached_forecast(path: str, modified_ns: int, timezone_name: str) -> Any:
        del modified_ns
        return load_forecast_curve(path, timezone_name=timezone_name)

    @st.cache_data(ttl=15, max_entries=32, show_spinner=False)
    def cached_comparison(
        registry: str,
        registry_modified_ns: int,
        project_root: str,
        zones: tuple[str, ...],
        allow_mixed_delivery_days: bool,
    ) -> Any:
        del registry_modified_ns
        comparison_statuses = inspect_zone_statuses(registry, zones=zones)
        return load_latest_forecast_comparison(
            comparison_statuses,
            project_root=project_root,
            zones=zones,
            allow_mixed_delivery_days=allow_mixed_delivery_days,
        )

    @st.cache_data(max_entries=16, show_spinner=False)
    def cached_consolidated_report(
        _comparison: Any,
        archive_fingerprints: tuple[tuple[str, str, str, str], ...],
        include_intervals: bool,
    ) -> bytes:
        # ``archive_fingerprints`` is the cache key.  The comparison itself is
        # deliberately excluded from Streamlit hashing: every miss still goes
        # through the renderer's complete archive/SHA-256 revalidation.
        del archive_fingerprints
        return render_consolidated_forecast_report(
            _comparison,
            include_intervals=include_intervals,
        )

    st.title(":material/bolt: Forecast control room")
    st.caption(
        "Prévisions day-ahead multi-pays · Storm reste un benchmark "
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
                    "Axe commun en UTC ; le tooltip conserve l'heure locale et "
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

    if ENABLE_MARKET_COUPLING:
        st.subheader("Convergence des prix prévus entre marchés", anchor=False)
        st.caption(
            "Cette vue compare les prix P50 des pays voisins. Les traits montrent "
            "l'importance des écarts de prix ; ils ne représentent pas des échanges "
            "d'électricité. Les flux physiques ne seront affichés que lorsqu'une "
            "source causale complète et vérifiée sera disponible."
        )
        try:
            render_market_coupling_panel(
                st,
                PROJECT_ROOT,
                delivery_day=None,
                border_signals=None,
                key_prefix="market_coupling",
            )
        except MarketCouplingDataError as exc:
            st.warning(f"Carte de couplage indisponible : {exc}")
        except Exception as exc:
            st.error(f"La carte de couplage ne peut pas être rendue : {exc}")

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
    if artifact.report_path is not None:
        with artifact.report_path.open("rb") as report_stream:
            st.download_button(
                "Télécharger le rapport HTML",
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
