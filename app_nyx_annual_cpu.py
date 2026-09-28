"""Separate NYX CWE annual CPU controls for the multi-zone Streamlit app."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable

from chronos2_hourly.app_service import read_log_tail
from chronos2_hourly.nyx_annual_app_service import (
    AnnualCpuProcess,
    inspect_annual_cpu_launch,
    load_annual_cpu_forecast,
    paths_for_day,
    start_annual_cpu_process,
)


def render_annual_cpu_controls(
    st: Any,
    *,
    project_root: Path,
    delivery_day: str,
    conventional_busy: bool,
    status_provider: Callable[[str, str], dict] | None = None,
) -> None:
    """Show a truthful launch decision and monitor the separate CPU process."""
    provider = status_provider or inspect_annual_cpu_launch
    report = provider(str(project_root), delivery_day)
    paths = paths_for_day(project_root, delivery_day)
    st.subheader("NYX CWE annuel CPU · France, Belgique, Pays-Bas", anchor=False)
    st.caption(
        "Réentraînement sur CPU et prévisions de prix et de probabilité de prix négatif. "
        "La mise à jour complète des sources Saturn et du bundle quotidien n'est pas encore automatisée."
    )
    with st.container(border=True):
        if report["ready"]:
            st.success("Qualification et bundle du jour valides : le calcul peut démarrer.")
        else:
            st.warning(
                "Lancement indisponible jusqu'à la qualification annuelle CPU "
                "et à la génération du bundle complet pour ce jour."
            )
        st.caption(f"Entrées attendues : {paths.bundle}")
        st.caption(f"Sorties immuables : {paths.output}")
        for blocker in report["blockers"]:
            st.write(f"- {blocker}")
        inspection = report.get("bundle_inspection") or {}
        missing = [check for check in inspection.get("checks", ())
                   if not check.get("passed")]
        if missing:
            with st.expander(f"Entrées manquantes ou invalides ({len(missing)})"):
                for check in missing:
                    st.write(f"- {check['input']} : {check.get('reason', 'échec')}")

        active: AnnualCpuProcess | None = st.session_state.annual_cpu_process
        busy = active is not None and active.return_code is None
        if st.button(
            "Lancer NYX CWE CPU · FR, BE, NL",
            key="launch_nyx_annual_cpu",
            icon=":material/play_arrow:",
            disabled=not report["ready"] or busy or conventional_busy,
            width="stretch",
        ):
            try:
                # The service repeats preflight here; the widget's disabled
                # state is only a visual guard, never authorization.
                handle = start_annual_cpu_process(
                    project_root, delivery_day, python_executable=sys.executable
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f"NYX CWE CPU n'a pas démarré : {error}")
            else:
                st.session_state.annual_cpu_process = handle
                st.rerun()

    @st.fragment(run_every="2s")
    def monitor_annual_cpu() -> None:
        handle: AnnualCpuProcess | None = st.session_state.annual_cpu_process
        if handle is not None and handle.delivery_day == delivery_day:
            code = handle.return_code
            if code is None:
                st.info(f"Réentraînement NYX CWE CPU en cours · PID {handle.process.pid}")
            elif code == 0:
                st.success("Le calcul annuel CPU s'est terminé. Vérifiez le reçu et les prévisions ci-dessous.")
            else:
                st.error(f"Le calcul NYX CWE CPU a échoué (code {code}).")
            st.code(read_log_tail(handle.log_path, max_chars=12_000), language="text")
        status_path = paths.output / "status.json"
        if status_path.is_file():
            try:
                state = json.loads(status_path.read_text(encoding="utf-8"))
                st.caption(
                    f"État enregistré : {state.get('status', 'inconnu')} · "
                    f"étape {state.get('phase', 'inconnue')}"
                )
            except (OSError, ValueError):
                st.warning("Le fichier d'état du calcul n'est pas lisible.")

    monitor_annual_cpu()

    if (paths.output / "receipt.json").is_file():
        zone = st.selectbox(
            "Prévisions annuelles CPU à consulter",
            ("FR", "BE", "NL"),
            key="annual_cpu_result_zone",
        )
        try:
            forecast = load_annual_cpu_forecast(project_root, delivery_day, zone)
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.error(f"Prévisions annuelles CPU non vérifiables : {error}")
        else:
            st.dataframe(
                forecast[["timestamp_local", "price_eur_mwh", "p_negative",
                          "is_negative_predicted"]],
                hide_index=True,
                width="stretch",
            )
