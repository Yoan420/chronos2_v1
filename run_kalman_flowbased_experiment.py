"""Run the isolated governed ``kalman_flowbased`` rolling-365 POC.

The runner reuses the audited residual-corrected prefix and residual-load PIT
source already produced for ``kalman_hybrid``.  It does not modify a live
archive, ``Forecast.ps1`` or Mode All.  JAO post-coupling fields are rejected
before the model is constructed.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import date
import hashlib
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
import yaml

from chronos2_hourly.jao_flowbased import (
    FLOWBASED_FEATURE_COLUMNS,
    JaoFlowBasedError,
    sha256_file,
)
from chronos2_hourly.kalman_configuration import (
    attach_additional_kalman_sources,
    attach_kalman_upstream_history,
    load_kalman_operational_configuration,
)
from chronos2_hourly.kalman_covariates import (
    BASE_RESIDUAL_LOAD_COVARIATES,
    DerivedCovariateSpec,
    KalmanCovariateConfig,
)
from chronos2_hourly.kalman_residual import (
    KalmanResidualConfig,
    build_operational_kalman_view,
)
from run_kalman_residual_experiment import (
    _assert_sources_unchanged,
    _validate_source,
    source_run_for,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "kalman_flowbased_experimental.yaml"
DEFAULT_FLOW_STORE = (
    PROJECT_ROOT / "data" / "pit" / "jao_core_flowbased" / "flowbased_features.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs" / "experiments" / "kalman_flowbased_rolling365_v1"
)
MODEL_KEY = "residual_kalman_flowbased"
TIMEZONES = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}
POST_COUPLING_TOKENS = (
    "shadow",
    "rammcp",
    "ram_mcp",
    "active_constraint",
    "scheduled_exchange",
    "net_position",
    "netpos",
    "price_spread",
    "realised",
    "realized",
)


class KalmanFlowBasedError(RuntimeError):
    """Raised when the isolated POC contract is incomplete."""


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise KalmanFlowBasedError(f"Configuration illisible: {path}.") from exc
    if not isinstance(value, Mapping):
        raise KalmanFlowBasedError("La configuration POC doit etre un objet YAML.")
    return value


def _config_objects(
    path: Path,
) -> tuple[
    KalmanResidualConfig,
    KalmanCovariateConfig,
    Mapping[str, str],
    Mapping[str, Any],
]:
    raw = _read_yaml(path)
    if raw.get("version") != 1 or raw.get("model_key") != MODEL_KEY:
        raise KalmanFlowBasedError("Version/model_key POC flow-based invalide.")
    if raw.get("deployment_status") != "experimental_shadow":
        raise KalmanFlowBasedError("Le POC doit rester experimental_shadow.")
    parameters = raw.get("filter_parameters")
    if not isinstance(parameters, Mapping):
        raise KalmanFlowBasedError("filter_parameters doit etre un objet.")
    defaults = KalmanResidualConfig()
    allowed = {item.name for item in fields(KalmanResidualConfig)}
    unknown = sorted(set(parameters).difference(allowed))
    if unknown:
        raise KalmanFlowBasedError(f"Parametres Kalman inconnus: {unknown}.")
    values = {
        item.name: parameters.get(item.name, getattr(defaults, item.name))
        for item in fields(KalmanResidualConfig)
    }
    values["candidate_kinds"] = tuple(map(str, values["candidate_kinds"]))
    filter_config = KalmanResidualConfig(**values)
    filter_config.validate()
    market = tuple(map(str, raw.get("market_columns", ())))
    flow = tuple(map(str, raw.get("flowbased_feature_columns", ())))
    if not market or not flow or len(flow) != len(set(flow)):
        raise KalmanFlowBasedError("Listes market/flowbased invalides.")
    unknown_flow = sorted(set(flow).difference(FLOWBASED_FEATURE_COLUMNS))
    if unknown_flow:
        raise KalmanFlowBasedError(f"Features flow-based inconnues: {unknown_flow}.")
    forbidden = sorted(
        column
        for column in (*market, *flow)
        if any(token in column.casefold() for token in POST_COUPLING_TOKENS)
    )
    if forbidden:
        raise KalmanFlowBasedError(
            f"Variables post-coupling interdites dans le POC: {forbidden}."
        )
    residual_spread = "residual_load_spread"
    covariate_config = KalmanCovariateConfig(
        input_columns=(*market, *flow),
        derived=(
            DerivedCovariateSpec(
                name=residual_spread,
                kind="spread",
                sources=market,
            ),
        ),
        groups={
            "market": (*market, residual_spread),
            # Generic production names are semantic aliases in this isolated
            # POC, documented and relabelled in every report/audit.
            "fundamentals": flow,
            "market_weather": (*market, residual_spread, *flow),
        },
        history_missing_policy="complete_trailing",
        minimum_history_coverage=1.0,
        require_future_complete=True,
    )
    covariate_config.validate()
    aliases_raw = raw.get("candidate_aliases", {})
    if not isinstance(aliases_raw, Mapping):
        raise KalmanFlowBasedError("candidate_aliases doit etre un objet.")
    aliases = {str(key): str(value) for key, value in aliases_raw.items()}
    expected_aliases = {
        "linear_fundamental": "linear_flowbased",
        "linear_market_weather": "linear_market_flowbased",
    }
    if aliases != expected_aliases:
        raise KalmanFlowBasedError(
            f"candidate_aliases attendu={expected_aliases}, obtenu={aliases}."
        )
    if raw.get("evaluation_days") != 365 or raw.get("training_lookback_days") != 365:
        raise KalmanFlowBasedError("Le POC comparable exige 365+365 jours.")
    return filter_config, covariate_config, aliases, raw


def _default_base_config(delivery_day: date, zone: str) -> Path:
    return (
        PROJECT_ROOT
        / "runs"
        / "runtime"
        / "kalman_hybrid"
        / delivery_day.isoformat()
        / zone.casefold()
        / "kalman_hybrid_operational.yaml"
    )


def _load_flow_store(
    path: Path,
    *,
    configured_columns: Sequence[str],
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    audit_path = path.with_name("flowbased_features.audit.json")
    if not path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(
            f"Store/audit JAO absent: {path}; lancez materialize_jao_core_flowbased.py."
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("parquet_sha256") != sha256_file(path):
        raise KalmanFlowBasedError("Checksum du store JAO invalide.")
    if audit.get("model_input_stage") != "initialComputation":
        raise KalmanFlowBasedError("Le store JAO n'est pas initialComputation.")
    if audit.get("all_partitions_research_pit_eligible") is not True:
        raise KalmanFlowBasedError(
            "Le POC de recherche refuse un store dont lastModifiedOn depasse "
            "le cutoff. Ce gate historique ne vaut pas promotion operationnelle."
        )
    if audit.get("all_partitions_tls_verified") is not True:
        raise KalmanFlowBasedError(
            "Le POC refuse une collecte construite avec TLS non verifie. "
            "Utilisez le bundle CA du proxy d'entreprise."
        )
    frame = pd.read_parquet(path)
    required = {
        "value_time_utc",
        "flowbased_pit_eligible",
        "flowbased_publication_stage",
        *configured_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KalmanFlowBasedError(f"Store JAO incomplet: {missing}.")
    forbidden = sorted(
        column
        for column in frame.columns
        if any(token in str(column).casefold() for token in POST_COUPLING_TOKENS)
    )
    if forbidden:
        raise KalmanFlowBasedError(
            f"Colonnes post-coupling presentes dans le store features: {forbidden}."
        )
    index = pd.DatetimeIndex(pd.to_datetime(frame["value_time_utc"], utc=True))
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise KalmanFlowBasedError("Timeline du store JAO invalide.")
    if not frame["flowbased_pit_eligible"].astype(bool).all():
        raise KalmanFlowBasedError("Une heure JAO n'est pas eligible PIT.")
    allowed_stages = {"initial_computation", "initial_computation_fallback"}
    observed_stages = set(frame["flowbased_publication_stage"].astype(str).unique())
    if not observed_stages.issubset(allowed_stages):
        raise KalmanFlowBasedError(
            f"Stage JAO interdit dans les features: {sorted(observed_stages)}."
        )
    numeric = frame.loc[:, list(configured_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise KalmanFlowBasedError("Features JAO non finies.")
    numeric.index = index
    return numeric, audit


def _attach_flow_features(
    base: pd.DataFrame,
    flow: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    if "timestamp" not in base:
        raise KalmanFlowBasedError("Covariables de base sans timestamp.")
    result = base.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result["timestamp"], utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise KalmanFlowBasedError("Timeline des covariables de base invalide.")
    collisions = sorted(set(columns).intersection(result.columns))
    if collisions:
        raise KalmanFlowBasedError(f"Collision de features flow-based: {collisions}.")
    aligned = flow.reindex(index)
    for column in columns:
        result[column] = aligned[column].to_numpy(dtype=float)
    return result


def _market_only_configuration(configuration: Any) -> Any:
    """Keep only the authoritative residual-load source from hybrid runtime."""

    expected = set(BASE_RESIDUAL_LOAD_COVARIATES)
    sources = tuple(
        source
        for source in configuration.additional_sources
        if set(source.columns) == expected
    )
    if len(sources) != 1:
        raise KalmanFlowBasedError(
            "Le sidecar de base doit contenir exactement une source PIT "
            "autoritative des cinq charges residuelles."
        )
    return replace(
        configuration,
        covariate_config=KalmanCovariateConfig(),
        additional_sources=sources,
    )


def _validate_preflight_support(
    *,
    statistics: pd.DataFrame,
    source_forecast: pd.DataFrame,
    covariates: pd.DataFrame,
    configured_columns: Sequence[str],
    timezone: str,
    delivery_day: date,
) -> Mapping[str, Any]:
    history_start = delivery_day - pd.Timedelta(days=730)
    history_index = pd.date_range(
        pd.Timestamp(history_start, tz=timezone),
        pd.Timestamp(delivery_day, tz=timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    future_index = pd.date_range(
        pd.Timestamp(delivery_day, tz=timezone),
        pd.Timestamp(delivery_day + pd.Timedelta(days=1), tz=timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    statistics_index = pd.DatetimeIndex(
        pd.to_datetime(statistics["delivery_start_utc"], utc=True)
    )
    history = statistics.set_index(statistics_index).reindex(history_index)
    required_history = {
        "actual",
        "residual_corrected__q10",
        "residual_corrected__q50",
        "residual_corrected__q90",
    }
    missing_history = sorted(required_history.difference(history.columns))
    if missing_history or not np.isfinite(
        history.loc[:, sorted(required_history)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
    ).all():
        raise KalmanFlowBasedError(
            f"Preflight: historique rolling 730 jours incomplet: {missing_history}."
        )
    observed_future = pd.DatetimeIndex(
        pd.to_datetime(source_forecast["delivery_start_utc"], utc=True)
    )
    if not observed_future.equals(future_index):
        raise KalmanFlowBasedError("Preflight: jour futur incomplet.")
    covariate_index = pd.DatetimeIndex(
        pd.to_datetime(covariates["timestamp"], utc=True)
    )
    support = history_index.append(future_index)
    selected = covariates.set_index(covariate_index).reindex(support)
    required_covariates = (*BASE_RESIDUAL_LOAD_COVARIATES, *configured_columns)
    missing_covariates = sorted(set(required_covariates).difference(selected.columns))
    if missing_covariates or not np.isfinite(
        selected.loc[:, list(required_covariates)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
    ).all():
        raise KalmanFlowBasedError(
            f"Preflight: support covariables 731 jours incomplet: "
            f"{missing_covariates}."
        )
    return {
        "status": "ready",
        "history_start_utc": history_index[0].isoformat(),
        "history_end_utc": history_index[-1].isoformat(),
        "history_days": 730,
        "history_hours": int(len(history_index)),
        "future_start_utc": future_index[0].isoformat(),
        "future_end_utc": future_index[-1].isoformat(),
        "future_hours": int(len(future_index)),
        "covariate_columns": list(required_covariates),
        "finite_covariate_values": int(len(support) * len(required_covariates)),
    }


def _validate_safe_output(
    output: Path,
    *,
    protected_paths: Sequence[Path],
) -> None:
    experiments_root = (PROJECT_ROOT / "runs" / "experiments").resolve()
    resolved = output.resolve()
    if resolved == experiments_root or not resolved.is_relative_to(experiments_root):
        raise KalmanFlowBasedError(
            "La sortie POC doit etre un sous-dossier strict de runs/experiments."
        )
    for protected in protected_paths:
        candidate = protected.resolve()
        if (
            resolved == candidate
            or resolved in candidate.parents
            or candidate in resolved.parents
        ):
            raise KalmanFlowBasedError(
                f"Sortie POC non disjointe d'un input protege: {candidate}."
            )


def _metrics(backtest: pd.DataFrame, *, timezone: str) -> Mapping[str, Any]:
    actual = pd.to_numeric(backtest["actual"], errors="coerce")
    base = pd.to_numeric(backtest["residual_corrected__q50"], errors="coerce")
    challenger = pd.to_numeric(backtest[f"{MODEL_KEY}__q50"], errors="coerce")
    core_mask = actual.notna() & base.notna() & challenger.notna()
    if int(core_mask.sum()) != len(backtest):
        raise KalmanFlowBasedError(
            "Les modeles autonome/flow-based ne sont pas apparies sur FINAL365."
        )
    rows: list[dict[str, Any]] = []
    core_scope = f"FINAL365_exact_{len(backtest)}h"

    def append_metric(
        model: str,
        predicted: pd.Series,
        mask: pd.Series,
        *,
        scope: str,
    ) -> None:
        error = predicted[mask] - actual[mask]
        rows.append(
            {
                "model": model,
                "comparison_scope": scope,
                "hours": int(mask.sum()),
                "mae_eur_mwh": float(error.abs().mean()),
                "rmse_eur_mwh": float(np.sqrt(np.square(error).mean())),
                "bias_eur_mwh": float(error.mean()),
            }
        )

    append_metric("residual_corrected", base, core_mask, scope=core_scope)
    append_metric(MODEL_KEY, challenger, core_mask, scope=core_scope)
    storm_paired_hours: int | None = None
    for storm_model, column in (
        ("storm_dashboard_official", "storm_dashboard_official__q50"),
        ("storm_strict_08", "storm_strict_08__q50"),
    ):
        if column not in backtest:
            continue
        storm = pd.to_numeric(backtest[column], errors="coerce")
        storm_mask = core_mask & storm.notna()
        if not storm_mask.any():
            continue
        storm_paired_hours = int(storm_mask.sum())
        scope = f"paired_with_{storm_model}"
        append_metric("residual_corrected", base, storm_mask, scope=scope)
        append_metric(MODEL_KEY, challenger, storm_mask, scope=scope)
        append_metric(storm_model, storm, storm_mask, scope=scope)
        break
    local_days = pd.DatetimeIndex(
        pd.to_datetime(backtest["delivery_start_utc"], utc=True)
    ).tz_convert(timezone).date
    flow_error = (
        pd.to_numeric(backtest[f"{MODEL_KEY}__q50"], errors="coerce") - actual
    ).abs()
    base_error = (
        pd.to_numeric(backtest["residual_corrected__q50"], errors="coerce") - actual
    ).abs()
    daily = pd.DataFrame(
        {
            "local_day": local_days,
            "flow_mae": flow_error,
            "base_mae": base_error,
        }
    ).groupby("local_day", as_index=False).mean()
    if len(daily) != 365:
        raise KalmanFlowBasedError(
            f"Rapport POC attendu sur 365 jours, obtenu={len(daily)}."
        )
    core_rows = {
        row["model"]: row
        for row in rows
        if row["comparison_scope"] == core_scope
    }
    baseline = core_rows.get("residual_corrected")
    challenger_row = core_rows.get(MODEL_KEY)
    gain = None
    relative = None
    if baseline and challenger_row:
        gain = baseline["mae_eur_mwh"] - challenger_row["mae_eur_mwh"]
        relative = gain / baseline["mae_eur_mwh"]
    return {
        "protocol": "FINAL365_exact_rolling_D_minus_365_to_D_minus_1",
        "models": rows,
        "kalman_flowbased_gain_eur_mwh": gain,
        "kalman_flowbased_relative_gain": relative,
        "storm_paired_hours": storm_paired_hours,
        "days": 365,
        "first_day": str(daily["local_day"].min()),
        "last_day": str(daily["local_day"].max()),
        "daily": daily,
    }


def _neutralize_unavailable_flowbased_hours(
    frame: pd.DataFrame,
    *,
    flow: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Fall back to residual_corrected when no initial CNEC exists."""

    if "delivery_start_utc" not in frame:
        raise KalmanFlowBasedError("Timeline absente pour neutraliser le fallback.")
    output = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(output["delivery_start_utc"], utc=True))
    availability = pd.to_numeric(
        flow.reindex(index)["flowbased_cnec_mtu_availability"], errors="coerce"
    )
    mask = availability.eq(0.0).to_numpy(dtype=bool)
    for quantile in ("q10", "q50", "q90"):
        source = f"residual_corrected__{quantile}"
        target = f"{MODEL_KEY}__{quantile}"
        if source in output and target in output:
            output.loc[mask, target] = output.loc[mask, source].to_numpy()
    return output, int(mask.sum())


def _report_html(
    *,
    zone: str,
    delivery_day: date,
    backtest: pd.DataFrame,
    metrics: Mapping[str, Any],
    daily_audit: pd.DataFrame,
    aliases: Mapping[str, str],
    source_manifest: Mapping[str, Any],
) -> str:
    metric_rows = metrics["models"]
    table = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['model']))}</td>"
        f"<td>{html.escape(str(row['comparison_scope']))}</td>"
        f"<td>{row['hours']}</td>"
        f"<td>{row['mae_eur_mwh']:.4f}</td>"
        f"<td>{row['rmse_eur_mwh']:.4f}</td>"
        f"<td>{row['bias_eur_mwh']:+.4f}</td>"
        "</tr>"
        for row in metric_rows
    )
    daily = metrics["daily"]
    daily_figure = go.Figure()
    daily_figure.add_trace(
        go.Scatter(
            x=daily["local_day"], y=daily["base_mae"], name="Autonome + residuel"
        )
    )
    daily_figure.add_trace(
        go.Scatter(
            x=daily["local_day"], y=daily["flow_mae"], name="Kalman flow-based"
        )
    )
    daily_figure.update_layout(
        title="MAE journaliere appariee — 365 jours",
        xaxis_title="Jour de livraison",
        yaxis_title="MAE (EUR/MWh)",
        template="plotly_white",
        height=430,
    )
    selected = daily_audit.get("selected_filter_label", pd.Series(dtype=str))
    counts = selected.value_counts().sort_values(ascending=True)
    bank_figure = go.Figure(
        go.Bar(x=counts.to_numpy(), y=counts.index.astype(str), orientation="h")
    )
    bank_figure.update_layout(
        title="Candidats retenus par la gouvernance",
        xaxis_title="Jours",
        template="plotly_white",
        height=380,
    )
    gain = metrics.get("kalman_flowbased_gain_eur_mwh")
    relative = metrics.get("kalman_flowbased_relative_gain")
    gain_text = (
        f"{gain:+.4f} EUR/MWh ({relative:+.2%})"
        if gain is not None and relative is not None
        else "indisponible"
    )
    neutralized_hours = int(metrics.get("neutralized_backtest_hours", 0))
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>POC kalman_flowbased — {zone} {delivery_day}</title>
<style>
:root{{--bg:#f6f8fb;--card:#fff;--text:#172033;--muted:#5d687c;--line:#d8deea;--accent:#1261a0;--warn:#8a4b08}}
body.dark{{--bg:#111827;--card:#1f2937;--text:#f3f4f6;--muted:#c1c7d0;--line:#485467;--accent:#65b5ff;--warn:#ffd18a}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}
main{{max-width:1220px;margin:auto;padding:24px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0}}
h1,h2{{margin-top:0}} .muted{{color:var(--muted)}} .warning{{border-left:5px solid var(--warn)}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}}
button{{float:right;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--text);cursor:pointer}}
</style></head><body><main>
<button id="theme">Mode nuit</button>
<h1>POC <code>kalman_flowbased</code> — {zone}</h1>
<p class="muted">Livraison {delivery_day} · rapport {metrics['first_day']} → {metrics['last_day']} · protocole rolling exact 365/365</p>
<section class="card warning"><h2>Statut expérimental</h2><p>Le candidat est isolé du Mode All. Les features proviennent uniquement de <code>initialComputation</code> JAO, filtré <code>Presolved=true</code>, avec <code>lastModifiedOn ≤ D-1 08:00</code>. Contraintes actives, shadow prices, RAM@MCP et échanges réalisés ne sont jamais des inputs.</p><p><strong>{neutralized_hours}</strong> heure(s) sans CNEC initiale ont été neutralisées : le forecast y reste strictement égal à <code>residual_corrected</code>.</p></section>
<section class="card"><h2>Résultats comparables</h2><p>Gain MAE contre autonome + correcteur résiduel : <strong>{gain_text}</strong>. Les lignes Storm et leurs deux comparateurs utilisent exactement leur intersection horaire commune.</p><table><thead><tr><th>Modèle</th><th>Support apparié</th><th>Heures</th><th>MAE</th><th>RMSE</th><th>Biais</th></tr></thead><tbody>{table}</tbody></table></section>
<section class="card">{daily_figure.to_html(full_html=False, include_plotlyjs=True)}</section>
<section class="card">{bank_figure.to_html(full_html=False, include_plotlyjs=False)}</section>
<section class="card"><h2>Banque gouvernée</h2><p>Alias sémantiques : <code>{html.escape(json.dumps(dict(aliases), ensure_ascii=False))}</code>. L’identité reste le garde-fou implicite.</p></section>
<section class="card"><h2>Provenance JAO</h2><p>Couverture : {source_manifest.get('start_day')} → {source_manifest.get('end_day')} · {source_manifest.get('physical_hours')} heures · SHA-256 <code>{html.escape(str(source_manifest.get('parquet_sha256')))}</code>.</p><p class="muted">Limite : {html.escape(str(source_manifest.get('historical_vintage_limitation')))}</p></section>
</main><script>
const b=document.body,btn=document.getElementById('theme');
function recolor(d){{if(typeof Plotly==='undefined')return;document.querySelectorAll('.js-plotly-plot').forEach(p=>Plotly.relayout(p,{{'paper_bgcolor':d?'#1f2937':'#fff','plot_bgcolor':d?'#1f2937':'#fff','font.color':d?'#f3f4f6':'#172033','xaxis.gridcolor':d?'#485467':'#e5e7eb','yaxis.gridcolor':d?'#485467':'#e5e7eb'}}));}}
function apply(d){{b.classList.toggle('dark',d);btn.textContent=d?'Mode jour':'Mode nuit';localStorage.setItem('chronos-theme',d?'dark':'light');recolor(d);}}
apply(localStorage.getItem('chronos-theme')==='dark'); btn.onclick=()=>apply(!b.classList.contains('dark'));
</script></body></html>"""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _publish(staging: Path, output: Path, *, overwrite: bool) -> None:
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


def run_experiment(
    *,
    zone: str,
    delivery_day: date,
    source_run: Path,
    base_kalman_config: Path,
    flow_store: Path,
    experiment_config: Path,
    output: Path,
    rolling_workers: int | None,
    overwrite: bool,
    preflight_only: bool = False,
) -> Path | None:
    zone = zone.upper()
    if zone not in TIMEZONES:
        raise KalmanFlowBasedError(f"Zone non supportee: {zone}.")
    if zone != "FR":
        raise KalmanFlowBasedError(
            "Le POC v1 est volontairement limite a FR; ses variables PTDF "
            "sont centrees sur les voisins FR-DE-BE-NL."
        )
    _validate_safe_output(
        output,
        protected_paths=(
            source_run,
            base_kalman_config,
            flow_store,
            experiment_config,
        ),
    )
    timezone = TIMEZONES[zone]
    source_manifest, source_hashes = _validate_source(
        source_run, zone=zone, delivery_day=delivery_day
    )
    filter_config, covariate_config, aliases, raw_config = _config_objects(
        experiment_config
    )
    base_configuration = load_kalman_operational_configuration(
        base_kalman_config, project_root=PROJECT_ROOT
    )
    model_source_configuration = _market_only_configuration(base_configuration)
    extra_source_paths = [base_kalman_config, experiment_config, flow_store]
    extra_source_paths.extend(
        source.path for source in model_source_configuration.additional_sources
    )
    if model_source_configuration.upstream_history is not None:
        extra_source_paths.extend(
            (
                model_source_configuration.upstream_history.path,
                model_source_configuration.upstream_history.audit_path,
            )
        )
    extra_source_hashes = {
        str(path.resolve()): sha256_file(path) for path in extra_source_paths
    }
    flow_columns = tuple(raw_config["flowbased_feature_columns"])
    flow, flow_audit = _load_flow_store(
        flow_store, configured_columns=flow_columns
    )
    statistics_path = source_run / "statistics_history_hourly.csv.gz"
    forecast_path = source_run / f"forecast_hourly_{zone.casefold()}.csv"
    covariates_path = source_run / "inputs" / "model_covariates_with_future.csv.gz"
    for path in (statistics_path, forecast_path, covariates_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    statistics_raw = pd.read_csv(statistics_path)
    statistics, upstream_audit = attach_kalman_upstream_history(
        statistics_raw, model_source_configuration, timezone=timezone
    )
    source_forecast = pd.read_csv(forecast_path)
    future_index = pd.DatetimeIndex(
        pd.to_datetime(source_forecast["delivery_start_utc"], utc=True)
    )
    base_covariates, base_sources_audit = attach_additional_kalman_sources(
        pd.read_csv(covariates_path),
        model_source_configuration,
        required_future_index=future_index,
        timezone=timezone,
    )
    covariates = _attach_flow_features(
        base_covariates, flow, columns=flow_columns
    )
    preflight = _validate_preflight_support(
        statistics=statistics,
        source_forecast=source_forecast,
        covariates=covariates,
        configured_columns=flow_columns,
        timezone=timezone,
        delivery_day=delivery_day,
    )
    print(
        "[KALMAN-FLOWBASED] preflight: "
        + json.dumps(preflight, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    if preflight_only:
        return None
    configured_workers = int(raw_config.get("rolling_refit_workers", 1))
    workers = configured_workers if rolling_workers is None else int(rolling_workers)
    cache = (
        PROJECT_ROOT
        / "runs"
        / "cache"
        / "kalman_rolling"
        / zone.casefold()
        / "kalman_flowbased"
    )
    view = build_operational_kalman_view(
        statistics=statistics,
        source_forecast=source_forecast,
        covariates=covariates,
        timezone=timezone,
        delivery_day=delivery_day,
        config=filter_config,
        covariate_config=covariate_config,
        upstream_model="residual_corrected",
        output_model=MODEL_KEY,
        evaluation_days=365,
        training_lookback_days=365,
        rolling_refit_workers=workers,
        rolling_refit_cache_dir=cache,
    )
    backtest = view.backtest.copy()
    flow_report = flow.reindex(
        pd.DatetimeIndex(pd.to_datetime(backtest["delivery_start_utc"], utc=True))
    )
    for column in flow_columns:
        backtest[column] = flow_report[column].to_numpy(dtype=float)
    backtest, neutralized_backtest_hours = (
        _neutralize_unavailable_flowbased_hours(backtest, flow=flow)
    )
    statistics_output, neutralized_statistics_hours = (
        _neutralize_unavailable_flowbased_hours(view.statistics, flow=flow)
    )
    forecast_output, neutralized_forecast_hours = (
        _neutralize_unavailable_flowbased_hours(view.forecast, flow=flow)
    )
    metrics = _metrics(backtest, timezone=timezone)
    metrics = {
        **dict(metrics),
        "neutralized_backtest_hours": neutralized_backtest_hours,
        "neutralized_forecast_hours": neutralized_forecast_hours,
        "unavailable_flowbased_policy": "identity_residual_corrected",
    }
    daily = view.replay.daily_audit.copy()
    daily["selected_filter_label"] = daily["selected_filter"].map(
        lambda value: aliases.get(str(value), str(value))
    )
    state = view.replay.state_audit.copy()
    if "filter_kind" in state:
        state["filter_label"] = state["filter_kind"].map(
            lambda value: aliases.get(str(value), str(value))
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        statistics_output.to_csv(
            staging / "statistics_history_hourly.csv.gz",
            index=False,
            compression="gzip",
        )
        backtest.to_csv(
            staging / "backtest_hourly.csv.gz", index=False, compression="gzip"
        )
        forecast_output.to_csv(
            staging / f"forecast_{zone.casefold()}_{delivery_day}_kalman_flowbased.csv",
            index=False,
        )
        daily.to_csv(staging / "kalman_daily_audit.csv", index=False)
        state.to_csv(
            staging / "kalman_state_audit.csv.gz", index=False, compression="gzip"
        )
        filter_audit = {
            **dict(view.replay.audit),
            "model_key": MODEL_KEY,
            "deployment_status": "experimental_shadow",
            "semantic_candidate_aliases": dict(aliases),
            "upstream_history_audit": upstream_audit,
            "base_sources_audit": base_sources_audit,
            "flowbased_source_audit": flow_audit,
            "postcoupling_fields_used_as_input": False,
            "mode_all_modified": False,
            "unavailable_flowbased_policy": "identity_residual_corrected",
            "neutralized_backtest_hours": neutralized_backtest_hours,
            "neutralized_statistics_hours": neutralized_statistics_hours,
            "neutralized_forecast_hours": neutralized_forecast_hours,
        }
        _atomic_json(staging / "kalman_filter_audit.json", filter_audit)
        metric_payload = {key: value for key, value in metrics.items() if key != "daily"}
        _atomic_json(
            staging / "metrics_comparison_rolling365.json", metric_payload
        )
        _atomic_json(staging / "flowbased_source_audit.json", dict(flow_audit))
        report = _report_html(
            zone=zone,
            delivery_day=delivery_day,
            backtest=backtest,
            metrics=metrics,
            daily_audit=daily,
            aliases=aliases,
            source_manifest=flow_audit,
        )
        (staging / "kalman_flowbased_report.html").write_text(
            report, encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "model_key": MODEL_KEY,
            "zone": zone,
            "delivery_day": delivery_day.isoformat(),
            "source_run": str(source_run),
            "source_run_manifest": source_manifest,
            "source_hashes": {
                str(path): sha256_file(path)
                for path in (statistics_path, forecast_path, covariates_path)
            },
            "base_kalman_config": str(base_kalman_config),
            "base_kalman_config_sha256": sha256_file(base_kalman_config),
            "experiment_config": str(experiment_config),
            "experiment_config_sha256": sha256_file(experiment_config),
            "flow_store": str(flow_store),
            "flow_store_sha256": sha256_file(flow_store),
            "output_artifacts": [],
            "preflight": preflight,
            "unavailable_flowbased_policy": "identity_residual_corrected",
            "neutralized_backtest_hours": neutralized_backtest_hours,
            "neutralized_statistics_hours": neutralized_statistics_hours,
            "neutralized_forecast_hours": neutralized_forecast_hours,
        }
        for path in sorted(staging.iterdir()):
            if path.name == "artifact_checksums.json":
                continue
            manifest["output_artifacts"].append(
                {
                    "path": path.name,
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
        _atomic_json(staging / "artifact_checksums.json", manifest)
        _assert_sources_unchanged(source_hashes)
        changed_extra = [
            path
            for path, digest in extra_source_hashes.items()
            if not Path(path).is_file() or sha256_file(path) != digest
        ]
        if changed_extra:
            raise KalmanFlowBasedError(
                f"Une source du POC a change pendant le rolling: {changed_extra}."
            )
        _publish(staging, output, overwrite=overwrite)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="POC causal residual_corrected + banque Kalman flow-based."
    )
    parser.add_argument("--zone", choices=tuple(TIMEZONES), default="FR")
    parser.add_argument("--delivery-day", type=date.fromisoformat, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--base-kalman-config", type=Path)
    parser.add_argument("--flowbased-features", type=Path, default=DEFAULT_FLOW_STORE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rolling-refit-workers", type=int, choices=range(1, 9))
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Valide les 731 jours et toutes les sources sans lancer les refits.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    zone = args.zone.upper()
    source = (
        args.source_run.expanduser().resolve()
        if args.source_run is not None
        else source_run_for(zone, args.delivery_day)
    )
    base_config = (
        args.base_kalman_config.expanduser().resolve()
        if args.base_kalman_config is not None
        else _default_base_config(args.delivery_day, zone)
    )
    if not base_config.is_file():
        raise FileNotFoundError(
            f"Sidecar kalman_hybrid absent: {base_config}. Lancez d'abord un "
            "Mode All reussi ou fournissez --base-kalman-config."
        )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (
            DEFAULT_OUTPUT_ROOT
            / args.delivery_day.isoformat()
            / zone.casefold()
        ).resolve()
    )
    print(
        f"[KALMAN-FLOWBASED] {zone} {args.delivery_day}: POC rolling 365/365...",
        flush=True,
    )
    result = run_experiment(
        zone=zone,
        delivery_day=args.delivery_day,
        source_run=source,
        base_kalman_config=base_config,
        flow_store=args.flowbased_features.expanduser().resolve(),
        experiment_config=args.config.expanduser().resolve(),
        output=output,
        rolling_workers=args.rolling_refit_workers,
        overwrite=bool(args.overwrite),
        preflight_only=bool(args.preflight_only),
    )
    if result is None:
        print("[KALMAN-FLOWBASED] Preflight termine; aucun artefact ecrit.", flush=True)
    else:
        print(f"[KALMAN-FLOWBASED] Termine: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
