"""Produce a 365-day descriptive CNEC/RAM versus forecast-error audit."""

from __future__ import annotations

import argparse
from datetime import date
import html
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from chronos2_hourly.jao_flowbased import FLOWBASED_FEATURE_COLUMNS, sha256_file
from run_kalman_residual_experiment import source_run_for


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FLOW_STORE = (
    PROJECT_ROOT / "data" / "pit" / "jao_core_flowbased" / "flowbased_features.parquet"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "experiments" / "cnec_ram_audit_365"
TIMEZONES = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}


class CnecRamAuditError(RuntimeError):
    """Raised when an exact, paired 365-day audit cannot be built."""


def _read_flow(path: Path) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    audit_path = path.with_name("flowbased_features.audit.json")
    if not path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("parquet_sha256") != sha256_file(path):
        raise CnecRamAuditError("Checksum du store flow-based invalide.")
    frame = pd.read_parquet(path)
    required = {
        "value_time_utc",
        "flowbased_pit_eligible",
        *FLOWBASED_FEATURE_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise CnecRamAuditError(f"Store flow-based incomplet: {missing}.")
    index = pd.DatetimeIndex(pd.to_datetime(frame["value_time_utc"], utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise CnecRamAuditError("Timeline flow-based invalide.")
    frame = frame.set_index(index)
    return frame, audit


def build_audit(
    *,
    statistics: pd.DataFrame,
    flow: pd.DataFrame,
    timezone: str,
    delivery_day: date,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    required_stats = {
        "delivery_start_utc",
        "actual",
        "residual_corrected__q50",
    }
    missing = sorted(required_stats.difference(statistics.columns))
    if missing:
        raise CnecRamAuditError(f"Statistics incomplet: {missing}.")
    frame = statistics.copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame["delivery_start_utc"], utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise CnecRamAuditError("Timeline Statistics invalide.")
    frame.index = index
    local_days = pd.Index(index.tz_convert(timezone).date)
    eligible = np.asarray(local_days < delivery_day, dtype=bool)
    actual = pd.to_numeric(frame["actual"], errors="coerce")
    forecast = pd.to_numeric(frame["residual_corrected__q50"], errors="coerce")
    eligible &= actual.notna().to_numpy() & forecast.notna().to_numpy()
    available_days = list(pd.Index(local_days[eligible]).drop_duplicates())
    if len(available_days) < 365:
        raise CnecRamAuditError(
            f"Historique observe insuffisant: {len(available_days)} jours < 365."
        )
    selected_days = set(available_days[-365:])
    mask = eligible & np.asarray(pd.Index(local_days).isin(selected_days), dtype=bool)
    paired = frame.loc[mask].copy()
    expected = pd.date_range(
        pd.Timestamp(min(selected_days), tz=timezone),
        pd.Timestamp(delivery_day, tz=timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    # delivery_day may be later than the last observed day on a replay. Build
    # the exact bound from the 365 selected days instead.
    expected = pd.date_range(
        pd.Timestamp(min(selected_days), tz=timezone),
        pd.Timestamp(max(selected_days) + pd.Timedelta(days=1), tz=timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    if not paired.index.equals(expected):
        raise CnecRamAuditError("Support Statistics FINAL365 non complet.")
    flow_selected = flow.reindex(paired.index)
    if flow_selected["flowbased_pit_eligible"].isna().any():
        raise CnecRamAuditError("Heures CNEC/RAM absentes du support FINAL365.")
    if not flow_selected["flowbased_pit_eligible"].astype(bool).all():
        raise CnecRamAuditError(
            "Le rapport causal refuse une partition CNEC/RAM non admissible PIT."
        )
    for column in FLOWBASED_FEATURE_COLUMNS:
        paired[column] = pd.to_numeric(
            flow_selected[column], errors="coerce"
        ).to_numpy(dtype=float)
    if not np.isfinite(
        paired.loc[:, list(FLOWBASED_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all():
        raise CnecRamAuditError("Features CNEC/RAM non finies sur FINAL365.")
    paired["forecast_error_eur_mwh"] = (
        pd.to_numeric(paired["residual_corrected__q50"], errors="raise")
        - pd.to_numeric(paired["actual"], errors="raise")
    )
    paired["absolute_error_eur_mwh"] = paired["forecast_error_eur_mwh"].abs()
    paired["local_day"] = paired.index.tz_convert(timezone).date
    paired["local_hour"] = paired.index.tz_convert(timezone).hour
    stress_column = "flowbased_core_ram_stress_p95_per_gw"
    quartile_codes = pd.qcut(
        paired[stress_column], q=4, labels=False, duplicates="drop"
    )
    paired["stress_quartile"] = quartile_codes.map(
        lambda value: f"Q{int(value) + 1}" if pd.notna(value) else "indisponible"
    )
    daily = paired.groupby("local_day", as_index=False).agg(
        hours=("actual", "size"),
        observed_mean_eur_mwh=("actual", "mean"),
        forecast_mean_eur_mwh=("residual_corrected__q50", "mean"),
        mae_eur_mwh=("absolute_error_eur_mwh", "mean"),
        bias_eur_mwh=("forecast_error_eur_mwh", "mean"),
        ram_min_mw=("flowbased_ram_min_mw", "min"),
        ram_p10_mw=("flowbased_ram_p10_mw", "mean"),
        core_stress_p95_per_gw=(stress_column, "max"),
        cnec_count=("flowbased_cnec_count", "mean"),
    )
    conditional = (
        paired.groupby("stress_quartile", observed=True)
        .agg(
            hours=("actual", "size"),
            mae_eur_mwh=("absolute_error_eur_mwh", "mean"),
            bias_eur_mwh=("forecast_error_eur_mwh", "mean"),
            mean_ram_min_mw=("flowbased_ram_min_mw", "mean"),
            mean_stress=(stress_column, "mean"),
        )
        .reset_index()
    )
    correlations: dict[str, float | None] = {}
    for column in FLOWBASED_FEATURE_COLUMNS:
        value = float(
            paired[[column, "absolute_error_eur_mwh"]]
            .corr(method="spearman")
            .iloc[0, 1]
        )
        correlations[column] = value if np.isfinite(value) else None
    summary = {
        "schema_version": 1,
        "status": "complete",
        "role": "descriptive_network_audit_not_model_training",
        "zone": timezone,
        "days": int(daily["local_day"].nunique()),
        "hours": int(len(paired)),
        "first_day": str(daily["local_day"].min()),
        "last_day": str(daily["local_day"].max()),
        "mae_eur_mwh": float(paired["absolute_error_eur_mwh"].mean()),
        "bias_eur_mwh": float(paired["forecast_error_eur_mwh"].mean()),
        "mean_cnec_mtu_availability": float(
            paired["flowbased_cnec_mtu_availability"].mean()
        ),
        "imputed_hours": int(paired["flowbased_hour_imputed"].sum()),
        "conditional_by_stress_quartile": conditional.to_dict(orient="records"),
        "spearman_absolute_error": correlations,
        "postcoupling_labels_used_as_input": False,
        "causality_violations": 0,
    }
    return paired.reset_index(drop=True), daily, summary


def _report(
    *,
    zone: str,
    daily: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    calendar = go.Figure(
        go.Heatmap(
            x=pd.to_datetime(daily["local_day"]),
            y=["MAE"] * len(daily),
            z=[daily["mae_eur_mwh"].to_numpy()],
            colorscale="YlOrRd",
            colorbar={"title": "EUR/MWh"},
        )
    )
    calendar.update_layout(
        title="Calendrier continu de l'erreur absolue journaliere",
        height=300,
        template="plotly_white",
    )
    scatter = go.Figure(
        go.Scatter(
            x=daily["core_stress_p95_per_gw"],
            y=daily["mae_eur_mwh"],
            mode="markers",
            text=daily["local_day"].astype(str),
            marker={"color": daily["ram_min_mw"], "colorscale": "Viridis", "showscale": True},
        )
    )
    scatter.update_layout(
        title="Stress flow-based et MAE journaliere",
        xaxis_title="Stress Core P95 / GW",
        yaxis_title="MAE (EUR/MWh)",
        height=430,
        template="plotly_white",
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['stress_quartile']))}</td>"
        f"<td>{int(row['hours'])}</td>"
        f"<td>{float(row['mae_eur_mwh']):.4f}</td>"
        f"<td>{float(row['bias_eur_mwh']):+.4f}</td>"
        f"<td>{float(row['mean_ram_min_mw']):.1f}</td>"
        "</tr>"
        for row in summary["conditional_by_stress_quartile"]
    )
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Audit CNEC/RAM {zone}</title>
<style>:root{{--bg:#f5f7fb;--card:white;--text:#172033;--line:#d7deea}}body.dark{{--bg:#111827;--card:#1f2937;--text:#f3f4f6;--line:#4b5563}}body{{font-family:Segoe UI,Arial;background:var(--bg);color:var(--text);margin:0}}main{{max-width:1200px;margin:auto;padding:24px}}section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}button{{float:right}}</style></head><body><main>
<button id="theme">Mode nuit</button><h1>CNEC/RAM — audit réseau Core {zone}</h1><p>{summary['first_day']} → {summary['last_day']} · {summary['days']} jours · {summary['hours']} heures</p>
<section><h2>Lecture causale</h2><p>Les agrégats proviennent du domaine JAO initial dont <code>lastModifiedOn</code> précède le cutoff. Le backfill historique reste un vintage de recherche, pas une capture opérationnelle D-1. Les informations post-coupling restent exclues des inputs. Ce rapport mesure une association avec l'erreur ; il ne prouve pas une causalité.</p><p>Disponibilité moyenne des MTU CNEC : <strong>{float(summary['mean_cnec_mtu_availability']):.2%}</strong> · heures imputées : <strong>{int(summary['imputed_hours'])}</strong> · jours entièrement vides reconstruits : <strong>{int(summary.get('empty_initial_fallback_days') or 0)}</strong>.</p><p>TLS vérifié sur toutes les partitions : <strong>{html.escape(str(summary.get('all_partitions_tls_verified')))}</strong>.</p></section>
<section><h2>Erreur par régime de stress</h2><table><thead><tr><th>Quartile</th><th>Heures</th><th>MAE</th><th>Biais</th><th>RAM min moyen</th></tr></thead><tbody>{rows}</tbody></table></section>
<section>{calendar.to_html(full_html=False, include_plotlyjs=True)}</section><section>{scatter.to_html(full_html=False, include_plotlyjs=False)}</section>
</main><script>const b=document.body,x=document.getElementById('theme');function r(d){{if(typeof Plotly==='undefined')return;document.querySelectorAll('.js-plotly-plot').forEach(p=>Plotly.relayout(p,{{'paper_bgcolor':d?'#1f2937':'#fff','plot_bgcolor':d?'#1f2937':'#fff','font.color':d?'#f3f4f6':'#172033','xaxis.gridcolor':d?'#4b5563':'#e5e7eb','yaxis.gridcolor':d?'#4b5563':'#e5e7eb'}}));}}function a(d){{b.classList.toggle('dark',d);x.textContent=d?'Mode jour':'Mode nuit';localStorage.setItem('chronos-theme',d?'dark':'light');r(d)}}a(localStorage.getItem('chronos-theme')==='dark');x.onclick=()=>a(!b.classList.contains('dark'));</script></body></html>"""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run(
    *,
    zone: str,
    delivery_day: date,
    statistics_path: Path,
    flow_path: Path,
    output: Path,
    overwrite: bool,
) -> Path:
    experiments_root = (PROJECT_ROOT / "runs" / "experiments").resolve()
    resolved_output = output.resolve()
    if (
        resolved_output == experiments_root
        or not resolved_output.is_relative_to(experiments_root)
    ):
        raise CnecRamAuditError(
            "La sortie d'audit doit etre un sous-dossier strict de runs/experiments."
        )
    for protected in (statistics_path.resolve(), flow_path.resolve()):
        if (
            resolved_output == protected
            or resolved_output in protected.parents
            or protected in resolved_output.parents
        ):
            raise CnecRamAuditError(
                f"Sortie d'audit non disjointe d'un input: {protected}."
            )
    flow, flow_manifest = _read_flow(flow_path)
    paired, daily, summary = build_audit(
        statistics=pd.read_csv(statistics_path),
        flow=flow,
        timezone=TIMEZONES[zone],
        delivery_day=delivery_day,
    )
    summary = {
        **dict(summary),
        "all_partitions_tls_verified": flow_manifest.get(
            "all_partitions_tls_verified"
        ),
        "all_partitions_operational_pit_eligible": flow_manifest.get(
            "all_partitions_operational_pit_eligible"
        ),
        "historical_vintage_limitation": flow_manifest.get(
            "historical_vintage_limitation"
        ),
        "empty_initial_fallback_days": flow_manifest.get(
            "empty_initial_fallback_days", 0
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        paired.to_parquet(staging / "cnec_ram_evaluation_hourly.parquet", index=False)
        daily.to_csv(staging / "cnec_ram_daily_audit.csv", index=False)
        _write_json(
            staging / "cnec_ram_input_audit.json",
            {
                **summary,
                "flowbased_store": str(flow_path),
                "flowbased_store_sha256": sha256_file(flow_path),
                "flowbased_manifest": flow_manifest,
                "statistics": str(statistics_path),
                "statistics_sha256": sha256_file(statistics_path),
            },
        )
        (staging / "cnec_ram_report.html").write_text(
            _report(zone=zone, daily=daily, summary=summary), encoding="utf-8"
        )
        checksums = {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in staging.iterdir()
        }
        _write_json(staging / "artifact_checksums.json", checksums)
        previous: Path | None = None
        if output.exists():
            if not overwrite:
                raise FileExistsError(output)
            previous = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
            output.replace(previous)
        try:
            os.replace(staging, output)
        except Exception:
            if previous is not None and previous.exists() and not output.exists():
                os.replace(previous, output)
            raise
        if previous is not None:
            shutil.rmtree(previous)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit CNEC/RAM apparie sur 365 jours.")
    parser.add_argument("--zone", choices=tuple(TIMEZONES), default="FR")
    parser.add_argument("--delivery-day", type=date.fromisoformat, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--flowbased-features", type=Path, default=DEFAULT_FLOW_STORE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = (
        args.source_run.expanduser().resolve()
        if args.source_run is not None
        else source_run_for(args.zone, args.delivery_day)
    )
    statistics = (
        args.statistics.expanduser().resolve()
        if args.statistics is not None
        else source / "statistics_history_hourly.csv.gz"
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (
            DEFAULT_OUTPUT_ROOT
            / args.delivery_day.isoformat()
            / args.zone.casefold()
        ).resolve()
    )
    result = run(
        zone=args.zone,
        delivery_day=args.delivery_day,
        statistics_path=statistics,
        flow_path=args.flowbased_features.expanduser().resolve(),
        output=output,
        overwrite=bool(args.overwrite),
    )
    print(f"[CNEC/RAM] Termine: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
