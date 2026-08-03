from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .common import SCRIPT_VERSION, ZoneRunResult, deep_get
from .metrics import add_comparison_fields, gain_percent


def html_table(
    frame: pd.DataFrame,
    float_format: str = "{:.4f}",
) -> str:
    if frame.empty:
        return '<p class="muted">Aucune donnée.</p>'
    columns = list(frame.columns)
    head = "".join(
        f"<th>{html.escape(str(column))}</th>"
        for column in columns
    )
    rows = []
    for record in frame.to_dict(orient="records"):
        cells = []
        for column in columns:
            value = record.get(column)
            if isinstance(value, float):
                rendered = (
                    "—"
                    if not np.isfinite(value)
                    else float_format.format(value)
                )
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def plotly_div(fig: go.Figure, include_js: bool) -> str:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=30, t=65, b=45),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    return pio.to_html(
        fig,
        include_plotlyjs="inline" if include_js else False,
        full_html=False,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
        },
    )


def figure_live_forecast(
    result: ZoneRunResult,
    history_hours: int,
) -> go.Figure:
    history = result.zone_data.target.iloc[-history_hours:]
    native = result.forecast_native
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history,
            name="Prix observé",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=native["timestamp"],
            y=native["q90"],
            name="P90",
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=native["timestamp"],
            y=native["q10"],
            name="Intervalle P10–P90",
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=native["timestamp"],
            y=native["q50"],
            name="Chronos-2 + covariables P50",
            mode="lines+markers",
        )
    )
    if result.forecast_baseline is not None:
        fig.add_trace(
            go.Scatter(
                x=result.forecast_baseline["timestamp"],
                y=result.forecast_baseline["q50"],
                name="Chronos-2 prix seul P50",
                mode="lines",
                line=dict(dash="dash"),
            )
        )
    fig.add_vline(
        x=result.zone_data.target.index[-1],
        line_dash="dot",
    )
    fig.update_layout(
        title=(
            f"{result.zone} — prévision Day-Ahead opérationnelle "
            "(24 h)"
        ),
        xaxis_title="Date de livraison",
        yaxis_title="EUR/MWh",
    )
    return fig


def figure_backtest(result: ZoneRunResult) -> go.Figure:
    native = (
        result.backtest_native
        .sort_values(["timestamp", "origin_timestamp"])
        .drop_duplicates("timestamp", keep="last")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=native["timestamp"],
            y=native["actual"],
            name="Observé",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=native["timestamp"],
            y=native["q90"],
            name="P90",
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=native["timestamp"],
            y=native["q10"],
            name="P10–P90",
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=native["timestamp"],
            y=native["q50"],
            name="Covariables P50",
            mode="lines",
        )
    )
    if result.backtest_baseline is not None:
        base = (
            result.backtest_baseline
            .sort_values(["timestamp", "origin_timestamp"])
            .drop_duplicates("timestamp", keep="last")
        )
        fig.add_trace(
            go.Scatter(
                x=base["timestamp"],
                y=base["q50"],
                name="Prix seul P50",
                mode="lines",
                line=dict(dash="dash"),
            )
        )
    fig.update_layout(
        title=f"{result.zone} — backtest glissant",
        xaxis_title="Date",
        yaxis_title="EUR/MWh",
    )
    return fig


def build_error_figure(
    zone: str,
    native: pd.DataFrame,
    baseline: pd.DataFrame | None = None,
) -> go.Figure:
    '''
    Construit le graphique temporel des erreurs du backtest.

    Erreur signée = q50 - actual
      > 0 : surestimation
      < 0 : sous-estimation
    '''
    ordered = (
        native.copy()
        .sort_values(["timestamp", "origin_timestamp"])
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    ordered["timestamp"] = pd.to_datetime(
        ordered["timestamp"],
        errors="coerce",
    )
    ordered["error"] = ordered["q50"] - ordered["actual"]
    ordered["absolute_error"] = ordered["error"].abs()
    ordered["rolling_mae_24h"] = (
        ordered["absolute_error"]
        .rolling(window=24, min_periods=1)
        .mean()
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=ordered["timestamp"],
            y=ordered["error"],
            mode="lines",
            name="Erreur Chronos-2",
            customdata=ordered[
                ["actual", "q50", "absolute_error"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Erreur : %{y:.2f} €/MWh<br>"
                "Observé : %{customdata[0]:.2f} €/MWh<br>"
                "Prévision q50 : %{customdata[1]:.2f} €/MWh<br>"
                "Erreur absolue : %{customdata[2]:.2f} €/MWh"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ordered["timestamp"],
            y=ordered["rolling_mae_24h"],
            mode="lines",
            name="MAE glissante 24 h",
            line={"dash": "dash"},
            hovertemplate=(
                "<b>%{x}</b><br>"
                "MAE glissante 24 h : %{y:.2f} €/MWh"
                "<extra></extra>"
            ),
        )
    )

    if baseline is not None and not baseline.empty:
        base = (
            baseline.copy()
            .sort_values(["timestamp", "origin_timestamp"])
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )
        base["timestamp"] = pd.to_datetime(
            base["timestamp"],
            errors="coerce",
        )
        base["error"] = base["q50"] - base["actual"]
        figure.add_trace(
            go.Scatter(
                x=base["timestamp"],
                y=base["error"],
                mode="lines",
                name="Erreur prix seul",
                line={"dash": "dot"},
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Erreur prix seul : %{y:.2f} €/MWh"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="Erreur nulle",
        annotation_position="top left",
    )
    figure.update_layout(
        title=f"{zone} — erreur du backtest glissant",
        xaxis_title="Date",
        yaxis_title="Erreur en €/MWh",
        hovermode="x unified",
        margin={"l": 55, "r": 30, "t": 65, "b": 45},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )
    figure.update_xaxes(rangeslider={"visible": True})
    return figure


def figure_horizon(result: ZoneRunResult) -> go.Figure:
    frame = result.metrics_by_horizon
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "MAE par horizon",
            "Couverture P10–P90",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=frame["horizon_step"],
            y=frame["native_mae_q50"],
            name="MAE covariables",
            mode="lines+markers",
        ),
        row=1,
        col=1,
    )
    if "baseline_mae_q50" in frame:
        fig.add_trace(
            go.Scatter(
                x=frame["horizon_step"],
                y=frame["baseline_mae_q50"],
                name="MAE prix seul",
                mode="lines+markers",
                line=dict(dash="dash"),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=frame["horizon_step"],
            y=100 * frame["native_coverage_q10_q90"],
            name="Couverture covariables",
            mode="lines+markers",
        ),
        row=1,
        col=2,
    )
    if "baseline_coverage_q10_q90" in frame:
        fig.add_trace(
            go.Scatter(
                x=frame["horizon_step"],
                y=100 * frame["baseline_coverage_q10_q90"],
                name="Couverture prix seul",
                mode="lines+markers",
                line=dict(dash="dash"),
            ),
            row=1,
            col=2,
        )
    fig.add_hline(y=80, line_dash="dot", row=1, col=2)
    fig.update_xaxes(title_text="Pas de prévision")
    fig.update_yaxes(title_text="EUR/MWh", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    fig.update_layout(
        title=f"{result.zone} — performance selon l’horizon"
    )
    return fig


def figure_hour(result: ZoneRunResult) -> go.Figure:
    frame = result.metrics_by_hour
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "MAE par heure de livraison",
            "Biais par heure",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=frame["delivery_hour"],
            y=frame["native_mae_q50"],
            name="MAE covariables",
        ),
        row=1,
        col=1,
    )
    if "baseline_mae_q50" in frame:
        fig.add_trace(
            go.Bar(
                x=frame["delivery_hour"],
                y=frame["baseline_mae_q50"],
                name="MAE prix seul",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=frame["delivery_hour"],
            y=frame["native_bias_q50"],
            name="Biais covariables",
        ),
        row=1,
        col=2,
    )
    if "baseline_bias_q50" in frame:
        fig.add_trace(
            go.Bar(
                x=frame["delivery_hour"],
                y=frame["baseline_bias_q50"],
                name="Biais prix seul",
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        title=f"{result.zone} — profil horaire des erreurs",
        barmode="group",
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Heure locale")
    fig.update_yaxes(title_text="EUR/MWh")
    return fig


def figure_scatter_residuals(result: ZoneRunResult) -> go.Figure:
    frame = result.backtest_native.copy()
    frame["error"] = frame["q50"] - frame["actual"]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Observé vs prédit",
            "Distribution des erreurs",
        ),
    )
    fig.add_trace(
        go.Scattergl(
            x=frame["actual"],
            y=frame["q50"],
            mode="markers",
            name="Observations",
            marker=dict(size=5, opacity=0.45),
        ),
        row=1,
        col=1,
    )
    lower = float(min(frame["actual"].min(), frame["q50"].min()))
    upper = float(max(frame["actual"].max(), frame["q50"].max()))
    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="Diagonale",
            line=dict(dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=frame["error"],
            nbinsx=50,
            name="Erreur P50",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Prix observé", row=1, col=1)
    fig.update_yaxes(title_text="Prix prédit P50", row=1, col=1)
    fig.update_xaxes(
        title_text="Erreur prédite − observée",
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Fréquence", row=1, col=2)
    fig.update_layout(
        title=f"{result.zone} — diagnostics des résidus",
        hovermode="closest",
    )
    return fig


def figure_inputs(
    result: ZoneRunResult,
    recent_hours: int = 24 * 30,
) -> go.Figure:
    cov = result.zone_data.covariates.iloc[-recent_hours:].copy()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Inputs normalisés récents",
            "Couverture des séries",
        ),
    )
    if not cov.empty:
        normalized = (
            (cov - cov.mean())
            / cov.std(ddof=0).replace(0, np.nan)
        )
        for column in normalized.columns:
            fig.add_trace(
                go.Scatter(
                    x=normalized.index,
                    y=normalized[column],
                    name=column,
                    mode="lines",
                ),
                row=1,
                col=1,
            )
    coverage = result.zone_data.coverage
    if not coverage.empty:
        fig.add_trace(
            go.Bar(
                x=coverage["alias"],
                y=100 * coverage["coverage_after_fill"],
                name="Couverture",
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text="Z-score", row=1, col=1)
    fig.update_yaxes(title_text="%", range=[0, 105], row=1, col=2)
    fig.update_layout(
        title=f"{result.zone} — qualité et dynamique des inputs",
        hovermode="closest",
    )
    return fig


def metric_cards(metrics: Mapping[str, Any]) -> str:
    specs = (
        ("MAE P50", "mae_q50", "EUR/MWh"),
        ("RMSE P50", "rmse_q50", "EUR/MWh"),
        ("Biais", "bias_q50", "EUR/MWh"),
        ("Corrélation", "correlation_q50", ""),
        ("CRPS approx.", "crps_quantile_approx", ""),
        ("Couverture 80 %", "coverage_q10_q90", "%"),
        ("Ramp MAE", "ramp_mae", "EUR/MWh"),
        ("Rappel extrêmes", "extreme_recall", "%"),
    )
    cards = []
    for title, key, unit in specs:
        value = metrics.get(key, math.nan)
        if unit == "%":
            rendered = (
                "—"
                if not np.isfinite(value)
                else f"{100 * value:.1f}%"
            )
        else:
            rendered = (
                "—"
                if not np.isfinite(value)
                else f"{value:.3f}"
            )
        cards.append(
            '<div class="metric-card">'
            f'<div class="metric-label">{html.escape(title)}</div>'
            f'<div class="metric-value">{rendered}</div>'
            f'<div class="metric-unit">{html.escape(unit if unit != "%" else "")}</div>'
            "</div>"
        )
    return '<div class="metric-grid">' + "".join(cards) + "</div>"


def build_global_comparison_figure(
    results: Sequence[ZoneRunResult],
) -> go.Figure:
    rows = []
    for result in results:
        rows.append(
            {
                "zone": result.zone,
                "variant": "Covariables natives",
                "mae": result.metrics_native["mae_q50"],
            }
        )
        if result.metrics_baseline is not None:
            rows.append(
                {
                    "zone": result.zone,
                    "variant": "Prix seul",
                    "mae": result.metrics_baseline["mae_q50"],
                }
            )
    frame = pd.DataFrame(rows)
    fig = go.Figure()
    for variant, block in frame.groupby("variant"):
        fig.add_trace(
            go.Bar(
                x=block["zone"],
                y=block["mae"],
                name=variant,
            )
        )
    fig.update_layout(
        title="Comparaison MAE par zone",
        xaxis_title="Zone",
        yaxis_title="MAE (EUR/MWh)",
        barmode="group",
    )
    return fig


MONTH_LABELS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _sample_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    '''
    Calcule les statistiques d'un échantillon temporel.
    La prévision centrale utilisée est q50.
    '''
    valid = frame[["actual", "q50"]].copy()
    valid["actual"] = pd.to_numeric(
        valid["actual"],
        errors="coerce",
    )
    valid["q50"] = pd.to_numeric(
        valid["q50"],
        errors="coerce",
    )
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna()

    if valid.empty:
        return {
            "mae": None,
            "rmse": None,
            "explained_variance": None,
            "r2": None,
            "std_error": None,
            "correlation": None,
            "n": 0,
        }

    actual = valid["actual"].to_numpy(dtype=float)
    predicted = valid["q50"].to_numpy(dtype=float)
    error = predicted - actual

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(np.square(error)))
    std_error = np.std(error, ddof=0)

    actual_variance = np.var(actual, ddof=0)
    error_variance = np.var(error, ddof=0)

    explained_variance = (
        1.0 - (error_variance / actual_variance)
        if actual_variance > 1e-12
        else math.nan
    )

    total_sum_squares = np.sum(
        np.square(actual - np.mean(actual))
    )
    residual_sum_squares = np.sum(np.square(error))
    r2 = (
        1.0 - (residual_sum_squares / total_sum_squares)
        if total_sum_squares > 1e-12
        else math.nan
    )

    correlation = (
        np.corrcoef(actual, predicted)[0, 1]
        if (
            len(actual) >= 2
            and np.std(actual, ddof=0) > 1e-12
            and np.std(predicted, ddof=0) > 1e-12
        )
        else math.nan
    )

    return {
        "mae": _finite_or_none(mae),
        "rmse": _finite_or_none(rmse),
        "explained_variance": _finite_or_none(
            explained_variance
        ),
        "r2": _finite_or_none(r2),
        "std_error": _finite_or_none(std_error),
        "correlation": _finite_or_none(correlation),
        "n": int(len(valid)),
    }


def _prepare_statistics_frame(
    result: ZoneRunResult,
) -> pd.DataFrame:
    frame = (
        result.backtest_native.copy()
        .sort_values(["timestamp", "origin_timestamp"])
        .drop_duplicates("timestamp", keep="last")
    )

    frame["actual"] = pd.to_numeric(
        frame["actual"],
        errors="coerce",
    )
    frame["q50"] = pd.to_numeric(
        frame["q50"],
        errors="coerce",
    )
    frame["_timestamp_local"] = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
        utc=True,
    )

    target_timezone = getattr(
        result.zone_data.target.index,
        "tz",
        None,
    )
    if target_timezone is not None:
        frame["_timestamp_local"] = (
            frame["_timestamp_local"]
            .dt.tz_convert(target_timezone)
        )

    frame = frame.dropna(
        subset=["_timestamp_local", "actual", "q50"]
    )
    frame["_timestamp_naive"] = (
        frame["_timestamp_local"]
        .dt.tz_localize(None)
    )
    return frame


def build_statistics_records(
    results: Sequence[ZoneRunResult],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for result in results:
        source = _prepare_statistics_frame(result)

        for sample in ("daily", "weekly", "monthly"):
            frame = source.copy()
            timestamp = frame["_timestamp_naive"]

            if sample == "daily":
                frame["_period_start"] = timestamp.dt.normalize()
                frame["_period_end"] = frame["_period_start"]
                frame["_sample_number"] = (
                    frame["_period_start"].dt.day
                )

            elif sample == "weekly":
                period = timestamp.dt.to_period("W-SUN")
                frame["_period_start"] = (
                    period.dt.start_time.dt.normalize()
                )
                frame["_period_end"] = (
                    period.dt.end_time.dt.normalize()
                )
                frame["_sample_number"] = (
                    frame["_period_start"]
                    .dt.isocalendar()
                    .week
                    .astype(int)
                )

            else:
                period = timestamp.dt.to_period("M")
                frame["_period_start"] = (
                    period.dt.start_time.dt.normalize()
                )
                frame["_period_end"] = (
                    period.dt.end_time.dt.normalize()
                )
                frame["_sample_number"] = (
                    frame["_period_start"].dt.month
                )

            for period_start, block in frame.groupby(
                "_period_start",
                sort=True,
            ):
                period_end = pd.Timestamp(
                    block["_period_end"].iloc[0]
                )
                sample_number = int(
                    block["_sample_number"].iloc[0]
                )
                metrics = _sample_metrics(block)

                timestamp_label = (
                    period_start.strftime("%Y-%m-%d")
                    if sample == "daily"
                    else (
                        f"{period_start:%Y-%m-%d}/"
                        f"{period_end:%Y-%m-%d}"
                    )
                )

                records.append(
                    {
                        "zone": result.zone,
                        "sample": sample,
                        "period_key": period_start.strftime(
                            "%Y-%m-%d"
                        ),
                        "period_start": period_start.strftime(
                            "%Y-%m-%d"
                        ),
                        "period_end": period_end.strftime(
                            "%Y-%m-%d"
                        ),
                        "year": int(period_start.year),
                        "month": MONTH_LABELS[
                            int(period_start.month)
                        ],
                        "sample_number": sample_number,
                        "timestamp": timestamp_label,
                        **metrics,
                    }
                )

    return records


def build_statistics_table_html(
    results: Sequence[ZoneRunResult],
) -> str:
    zones = [result.zone for result in results]
    records = build_statistics_records(results)

    payload = {
        "zones": zones,
        "records": records,
        "metrics": [
            {
                "key": "mae",
                "label": "Mean Absolute Error",
                "higher_is_better": False,
                "decimals": 2,
            },
            {
                "key": "rmse",
                "label": "Root Mean Squared Error",
                "higher_is_better": False,
                "decimals": 2,
            },
            {
                "key": "explained_variance",
                "label": "Explained Variance",
                "higher_is_better": True,
                "decimals": 3,
            },
            {
                "key": "r2",
                "label": "R²",
                "higher_is_better": True,
                "decimals": 3,
            },
            {
                "key": "std_error",
                "label": "Standard Deviation",
                "higher_is_better": False,
                "decimals": 2,
            },
            {
                "key": "correlation",
                "label": "Correlation",
                "higher_is_better": True,
                "decimals": 3,
            },
        ],
        "samples": [
            {"key": "weekly", "label": "Weekly"},
            {"key": "monthly", "label": "Monthly"},
            {"key": "daily", "label": "Daily"},
        ],
    }

    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
    ).replace("</", "<\\/")

    return f'''
    <section id="statistics" class="statistics-section">
        <div class="statistics-title-pill">Statistics</div>

        <div class="statistics-filtering-title">
            FILTERING
        </div>

        <div class="statistics-filters">
            <label class="statistics-filter">
                <span>Statistic</span>
                <select id="statistics-metric-select"></select>
            </label>

            <label class="statistics-filter">
                <span>Sample</span>
                <select id="statistics-sample-select"></select>
            </label>
        </div>

        <div
            id="statistics-table-container"
            class="statistics-table-container"
        ></div>
    </section>

    <script>
    (() => {{
        const payload = {payload_json};

        const metricSelect = document.getElementById(
            "statistics-metric-select"
        );
        const sampleSelect = document.getElementById(
            "statistics-sample-select"
        );
        const container = document.getElementById(
            "statistics-table-container"
        );

        payload.metrics.forEach((metric) => {{
            const option = document.createElement("option");
            option.value = metric.key;
            option.textContent = metric.label;
            metricSelect.appendChild(option);
        }});

        payload.samples.forEach((sample) => {{
            const option = document.createElement("option");
            option.value = sample.key;
            option.textContent = sample.label;
            sampleSelect.appendChild(option);
        }});

        metricSelect.value = "mae";
        sampleSelect.value = "weekly";

        const blue = [91, 157, 211];
        const white = [247, 248, 250];
        const red = [202, 77, 84];

        function interpolateColor(start, end, ratio) {{
            const values = start.map(
                (value, index) => Math.round(
                    value + (end[index] - value) * ratio
                )
            );
            return `rgb(${{values.join(",")}})`;
        }}

        function cellColor(
            value,
            minimum,
            maximum,
            higherIsBetter
        ) {{
            if (
                value === null
                || value === undefined
                || !Number.isFinite(value)
            ) {{
                return "transparent";
            }}

            let ratio = Math.abs(maximum - minimum) < 1e-12
                ? 0.5
                : (value - minimum) / (maximum - minimum);

            ratio = Math.max(0, Math.min(1, ratio));

            if (higherIsBetter) {{
                ratio = 1 - ratio;
            }}

            if (ratio <= 0.5) {{
                return interpolateColor(
                    blue,
                    white,
                    ratio * 2
                );
            }}

            return interpolateColor(
                white,
                red,
                (ratio - 0.5) * 2
            );
        }}

        function formatValue(value, decimals) {{
            if (
                value === null
                || value === undefined
                || !Number.isFinite(value)
            ) {{
                return "";
            }}
            return Number(value).toFixed(decimals);
        }}

        function renderStatisticsTable() {{
            const metricKey = metricSelect.value;
            const sampleKey = sampleSelect.value;

            const metric = payload.metrics.find(
                (item) => item.key === metricKey
            );

            const filtered = payload.records.filter(
                (record) => record.sample === sampleKey
            );

            const periods = new Map();

            filtered.forEach((record) => {{
                if (!periods.has(record.period_key)) {{
                    periods.set(record.period_key, {{
                        periodKey: record.period_key,
                        periodStart: record.period_start,
                        year: record.year,
                        month: record.month,
                        sampleNumber: record.sample_number,
                        timestamp: record.timestamp,
                        values: {{}},
                    }});
                }}

                periods
                    .get(record.period_key)
                    .values[record.zone] = {{
                        value: record[metricKey],
                        n: record.n,
                    }};
            }});

            const rows = Array.from(periods.values()).sort(
                (left, right) =>
                    right.periodStart.localeCompare(
                        left.periodStart
                    )
            );

            const displayedValues = [];

            rows.forEach((row) => {{
                payload.zones.forEach((zone) => {{
                    const entry = row.values[zone];
                    if (
                        entry
                        && Number.isFinite(entry.value)
                    ) {{
                        displayedValues.push(entry.value);
                    }}
                }});
            }});

            const minimum = displayedValues.length
                ? Math.min(...displayedValues)
                : 0;
            const maximum = displayedValues.length
                ? Math.max(...displayedValues)
                : 0;

            let sampleHeader = "";
            if (sampleKey === "weekly") {{
                sampleHeader = "Week";
            }} else if (sampleKey === "daily") {{
                sampleHeader = "Day";
            }}

            const headerParts = [
                "<th>Year</th>",
                "<th>Month</th>",
            ];

            if (sampleHeader) {{
                headerParts.push(
                    `<th>${{sampleHeader}}</th>`
                );
            }}

            headerParts.push("<th>Timestamp</th>");

            payload.zones.forEach((zone) => {{
                headerParts.push(
                    `<th class="statistics-zone-header">${{zone}}</th>`
                );
            }});

            const bodyRows = rows.map((row) => {{
                const cells = [
                    `<td>${{row.year}}</td>`,
                    `<td>${{row.month}}</td>`,
                ];

                if (sampleHeader) {{
                    cells.push(
                        `<td>${{row.sampleNumber}}</td>`
                    );
                }}

                cells.push(
                    `<td class="statistics-timestamp-cell">`
                    + `${{row.timestamp}}</td>`
                );

                payload.zones.forEach((zone) => {{
                    const entry = row.values[zone];
                    const value = entry ? entry.value : null;
                    const n = entry ? entry.n : 0;

                    const background = cellColor(
                        value,
                        minimum,
                        maximum,
                        metric.higher_is_better
                    );

                    const rendered = formatValue(
                        value,
                        metric.decimals
                    );

                    cells.push(
                        `<td class="statistics-value-cell"`
                        + ` style="background:${{background}}"`
                        + ` title="n=${{n}}">`
                        + `${{rendered}}</td>`
                    );
                }});

                return `<tr>${{cells.join("")}}</tr>`;
            }});

            if (!bodyRows.length) {{
                container.innerHTML = (
                    '<p class="muted">'
                    + 'Aucune donnée disponible.'
                    + '</p>'
                );
                return;
            }}

            container.innerHTML = `
                <table class="statistics-heatmap-table">
                    <thead>
                        <tr>${{headerParts.join("")}}</tr>
                    </thead>
                    <tbody>
                        ${{bodyRows.join("")}}
                    </tbody>
                </table>
            `;
        }}

        metricSelect.addEventListener(
            "change",
            renderStatisticsTable
        );
        sampleSelect.addEventListener(
            "change",
            renderStatisticsTable
        );

        renderStatisticsTable();
    }})();
    </script>
    '''


def write_html_report(
    results: Sequence[ZoneRunResult],
    config: Mapping[str, Any],
    output_path: Path,
) -> None:
    title = str(
        deep_get(
            config,
            "report.title",
            "Chronos-2 — rapport de prévision",
        )
    )
    history_hours = int(
        deep_get(config, "report.forecast_history_hours", 168)
    )
    generated = pd.Timestamp.now(tz="Europe/Paris").strftime(
        "%Y-%m-%d %H:%M %Z"
    )
    sections: list[str] = []
    include_js = True

    if len(results) > 1:
        div = plotly_div(
            build_global_comparison_figure(results),
            include_js,
        )
        include_js = False
        sections.append(
            f"<section><h2>Vue multi-zone</h2>{div}</section>"
        )

    nav_links = "".join(
        f'<a href="#{result.zone.lower()}">{result.zone}</a>'
        for result in results
    )
    nav_links += '<a href="#statistics">Statistics</a>'

    for result in results:
        comparison_rows = []
        for metric, label in (
            ("mae_q50", "MAE P50"),
            ("rmse_q50", "RMSE P50"),
            ("bias_q50", "Biais P50"),
            ("correlation_q50", "Corrélation"),
            ("crps_quantile_approx", "CRPS approx."),
            ("coverage_q10_q90", "Couverture P10–P90"),
            ("interval_width_q10_q90", "Largeur P10–P90"),
            ("ramp_mae", "Ramp MAE"),
            ("negative_recall", "Rappel prix négatifs"),
            ("extreme_recall", "Rappel prix extrêmes"),
        ):
            native_value = result.metrics_native.get(
                metric,
                math.nan,
            )
            baseline_value = (
                result.metrics_baseline.get(metric, math.nan)
                if result.metrics_baseline
                else math.nan
            )
            gain = (
                gain_percent(baseline_value, native_value)
                if metric
                not in {
                    "bias_q50",
                    "correlation_q50",
                    "coverage_q10_q90",
                    "negative_recall",
                    "extreme_recall",
                }
                else math.nan
            )
            comparison_rows.append(
                {
                    "Métrique": label,
                    "Covariables natives": native_value,
                    "Prix seul": baseline_value,
                    "Gain %": gain,
                }
            )

        comparison = pd.DataFrame(comparison_rows)
        delivery_start = result.forecast_native["timestamp"].min()
        delivery_end = result.forecast_native["timestamp"].max()

        figures = []
        for figure in (
            figure_live_forecast(result, history_hours),
            figure_backtest(result),
            build_error_figure(
                zone=result.zone,
                native=result.backtest_native,
                baseline=result.backtest_baseline,
            ),
            figure_horizon(result),
            figure_hour(result),
            figure_scatter_residuals(result),
            figure_inputs(result),
        ):
            figures.append(plotly_div(figure, include_js))
            include_js = False

        sections.append(
            f'''<section id="{result.zone.lower()}">
            <div class="zone-title"><div><h2>{result.zone}</h2>
            <p class="muted">Prévision opérationnelle du {html.escape(str(delivery_start))} au {html.escape(str(delivery_end))}</p></div>
            <span class="badge">{len(result.zone_data.covariates.columns)} covariables actives</span></div>
            {metric_cards(result.metrics_native)}
            <h3>Comparaison au modèle prix seul</h3>
            {html_table(comparison)}
            <h3>Prévision Day-Ahead réelle</h3>
            {figures[0]}
            <h3>Backtest et probabilités</h3>
            {figures[1]}
            <h3>Erreur temporelle du backtest</h3>
            <p class="muted">
                Erreur = prévision P50 − prix observé.
                Une valeur positive correspond à une surestimation,
                et une valeur négative à une sous-estimation.
            </p>
            {figures[2]}
            {figures[3]}
            <h3>Analyse des erreurs</h3>
            {figures[4]}
            {figures[5]}
            <h3>Variables d’entrée</h3>
            {figures[6]}
            {html_table(result.zone_data.input_manifest)}
            </section>'''
        )

    sections.append(
        build_statistics_table_html(results)
    )

    css = '''
    :root { --bg:#f4f6f8; --card:#ffffff; --text:#18212b; --muted:#607080; --border:#dfe5ea; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; background:var(--bg); color:var(--text); }
    header { background:#111827; color:white; padding:28px 5vw; }
    header h1 { margin:0 0 8px; font-size:30px; }
    nav { margin-top:18px; display:flex; gap:10px; flex-wrap:wrap; }
    nav a { color:white; text-decoration:none; border:1px solid rgba(255,255,255,.35); padding:7px 12px; border-radius:999px; }
    main { max-width:1500px; margin:0 auto; padding:28px 24px 60px; }
    section { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:24px; margin-bottom:26px; box-shadow:0 6px 22px rgba(20,35,55,.05); }
    h2 { font-size:26px; margin:0 0 12px; }
    h3 { margin-top:28px; border-bottom:1px solid var(--border); padding-bottom:8px; }
    .muted { color:var(--muted); }
    .zone-title { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; }
    .badge { background:#eaf0ff; color:#1746b8; padding:7px 12px; border-radius:999px; font-weight:600; }
    .metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:18px 0; }
    .metric-card { border:1px solid var(--border); border-radius:12px; padding:15px; background:#fbfcfd; }
    .metric-label { color:var(--muted); font-size:13px; }
    .metric-value { font-size:25px; font-weight:700; margin-top:5px; }
    .metric-unit { color:var(--muted); font-size:12px; }
    .table-wrap { overflow-x:auto; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { padding:9px 10px; border-bottom:1px solid var(--border); text-align:left; white-space:nowrap; }
    th { background:#f7f9fb; position:sticky; top:0; }
    footer { color:var(--muted); text-align:center; padding:20px; }

    .statistics-section {
        padding:0;
        overflow:hidden;
    }
    .statistics-title-pill {
        display:inline-block;
        background:#253448;
        color:#ffffff;
        font-size:16px;
        font-weight:700;
        padding:7px 12px;
        border-radius:0 0 12px 0;
    }
    .statistics-filtering-title {
        padding:22px 10px 4px;
        color:#637181;
        font-size:18px;
        font-weight:700;
    }
    .statistics-filters {
        display:flex;
        flex-wrap:wrap;
        gap:12px;
        padding:0 8px 18px;
        border-bottom:1px solid var(--border);
    }
    .statistics-filter {
        display:flex;
        flex-direction:column;
        gap:5px;
    }
    .statistics-filter span {
        color:var(--muted);
        font-size:12px;
        font-weight:600;
    }
    .statistics-filter select {
        width:280px;
        max-width:100%;
        min-height:46px;
        padding:9px 42px 9px 12px;
        border:1px solid var(--border);
        border-radius:10px;
        background:#ffffff;
        color:var(--text);
        font-family:inherit;
        font-size:16px;
        cursor:pointer;
    }
    .statistics-table-container {
        overflow:auto;
        max-height:720px;
    }
    .statistics-heatmap-table {
        min-width:850px;
        width:100%;
        border-collapse:separate;
        border-spacing:0;
        font-family:Consolas,"Courier New",monospace;
        font-size:14px;
    }
    .statistics-heatmap-table th,
    .statistics-heatmap-table td {
        height:45px;
        padding:9px 11px;
        border-right:1px solid #cfd5da;
        border-bottom:1px solid #cfd5da;
        text-align:center;
        white-space:nowrap;
    }
    .statistics-heatmap-table th {
        position:sticky;
        top:0;
        z-index:2;
        background:#edf0f2;
        color:#111820;
        font-weight:700;
    }
    .statistics-heatmap-table td:first-child,
    .statistics-heatmap-table th:first-child {
        border-left:1px solid #cfd5da;
    }
    .statistics-timestamp-cell {
        text-align:left !important;
        min-width:225px;
    }
    .statistics-zone-header,
    .statistics-value-cell {
        min-width:68px;
    }
    .statistics-value-cell {
        font-weight:500;
        font-variant-numeric:tabular-nums;
        transition:filter 120ms ease,transform 120ms ease;
    }
    .statistics-value-cell:hover {
        filter:brightness(.95);
        transform:scale(1.02);
    }

    @media (max-width:700px) {
        main { padding:14px; }
        section { padding:14px; }
        .statistics-section { padding:0; }
    }
    '''

    document = f'''<!doctype html>
    <html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{html.escape(title)}</title><style>{css}</style></head>
    <body><header><h1>{html.escape(title)}</h1><p>Généré le {generated} · Script {SCRIPT_VERSION}</p><nav>{nav_links}</nav></header>
    <main>{''.join(sections)}</main><footer>Chronos-2 · rapport autonome Plotly</footer></body></html>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
