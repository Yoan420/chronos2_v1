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
from .forecast_explanation import attribution_scope_html, build_forecast_components_html
from .hourly_comparison import build_hourly_comparison_html


_REPORT_THEME_BOOTSTRAP = """
<script>
(() => {
    const storageKey = "chronos2-report-theme";
    let theme = "light";
    try {
        const saved = window.localStorage.getItem(storageKey);
        if (saved === "light" || saved === "dark") {
            theme = saved;
        }
    } catch (_error) {
        // Some browsers restrict localStorage for local files.
    }
    document.documentElement.dataset.theme = theme;
})();
</script>
"""


_REPORT_THEME_CONTROLLER = r"""
<script>
(() => {
    const root = document.documentElement;
    const toggle = document.getElementById("theme-toggle");
    const storageKey = "chronos2-report-theme";

    function colorsFor(theme) {
        return theme === "dark"
            ? {
                text: "#e7edf6",
                muted: "#aab7ca",
                grid: "#314158",
                zero: "#56677f",
                surface: "#111b2e",
                hover: "#1a2940",
                border: "#41536c",
            }
            : {
                text: "#18212b",
                muted: "#607080",
                grid: "#dfe5ea",
                zero: "#b6c1cc",
                surface: "#ffffff",
                hover: "#ffffff",
                border: "#cbd5df",
            };
    }

    function themedAnnotations(plot, color) {
        const annotations = plot.layout && plot.layout.annotations;
        if (!Array.isArray(annotations)) {
            return null;
        }
        return annotations.map((annotation) => ({
            ...annotation,
            font: {...(annotation.font || {}), color},
        }));
    }

    function applyPlotlyTheme(plot, theme) {
        if (!window.Plotly || !plot || !plot.layout) {
            return;
        }
        const colors = colorsFor(theme);
        const layout = plot._fullLayout || plot.layout;
        const update = {
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            "font.color": colors.text,
            "title.font.color": colors.text,
            "legend.bgcolor": "rgba(0,0,0,0)",
            "legend.bordercolor": colors.border,
            "legend.font.color": colors.text,
            "hoverlabel.bgcolor": colors.hover,
            "hoverlabel.bordercolor": colors.border,
            "hoverlabel.font.color": colors.text,
        };

        Object.keys(layout).forEach((key) => {
            if (/^[xy]axis\d*$/.test(key)) {
                update[`${key}.color`] = colors.text;
                update[`${key}.gridcolor`] = colors.grid;
                update[`${key}.zerolinecolor`] = colors.zero;
                update[`${key}.linecolor`] = colors.border;
            }
            if (/^xaxis\d*$/.test(key)) {
                update[`${key}.rangeslider.bgcolor`] = colors.surface;
                update[`${key}.rangeslider.bordercolor`] = colors.border;
            }
            if (/^coloraxis\d*$/.test(key)) {
                update[`${key}.colorbar.tickfont.color`] = colors.text;
                update[`${key}.colorbar.title.font.color`] = colors.text;
                update[`${key}.colorbar.outlinewidth`] = 0;
            }
        });

        if (layout.scene) {
            update["scene.bgcolor"] = "rgba(0,0,0,0)";
            ["xaxis", "yaxis", "zaxis"].forEach((axis) => {
                update[`scene.${axis}.color`] = colors.text;
                update[`scene.${axis}.gridcolor`] = colors.grid;
                update[`scene.${axis}.zerolinecolor`] = colors.zero;
                update[`scene.${axis}.backgroundcolor`] = "rgba(0,0,0,0)";
            });
        }
        if (layout.polar) {
            update["polar.bgcolor"] = "rgba(0,0,0,0)";
            ["angularaxis", "radialaxis"].forEach((axis) => {
                update[`polar.${axis}.color`] = colors.text;
                update[`polar.${axis}.gridcolor`] = colors.grid;
                update[`polar.${axis}.linecolor`] = colors.border;
            });
        }

        const annotations = themedAnnotations(plot, colors.text);
        if (annotations) {
            update.annotations = annotations;
        }
        window.Plotly.relayout(plot, update);
    }

    function refreshPlots(theme) {
        const refresh = () => {
            document.querySelectorAll(".js-plotly-plot").forEach(
                (plot) => applyPlotlyTheme(plot, theme)
            );
        };
        window.requestAnimationFrame(refresh);
        window.setTimeout(refresh, 100);
    }

    function updateToggle(theme) {
        if (!toggle) {
            return;
        }
        const dark = theme === "dark";
        toggle.textContent = dark ? "☀ Mode clair" : "☾ Mode nuit";
        toggle.setAttribute("aria-pressed", String(dark));
        toggle.setAttribute(
            "aria-label",
            dark ? "Activer le mode clair" : "Activer le mode nuit"
        );
    }

    function applyTheme(theme, persist) {
        const normalized = theme === "dark" ? "dark" : "light";
        root.dataset.theme = normalized;
        updateToggle(normalized);
        if (persist) {
            try {
                window.localStorage.setItem(storageKey, normalized);
            } catch (_error) {
                // The report still works when localStorage is unavailable.
            }
        }
        refreshPlots(normalized);
        window.dispatchEvent(
            new CustomEvent("chronos2-theme-change", {
                detail: {theme: normalized},
            })
        );
    }

    if (toggle) {
        toggle.addEventListener("click", () => {
            applyTheme(root.dataset.theme === "dark" ? "light" : "dark", true);
        });
    }
    applyTheme(root.dataset.theme, false);
})();
</script>
"""


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


def build_variable_attribution_html(result: ZoneRunResult) -> str:
    """Render the audited local grouped-Shapley explanation, when available."""

    raw = getattr(result, "variable_attribution", None)
    if raw is None:
        return ('<div class="variable-attribution" data-report-section="variable-attribution" '
                'data-attribution-status="unavailable"><h3>Influence des variables et des prix passés</h3>'
                + attribution_scope_html(None) + '</div>')
    if not isinstance(raw, Mapping):
        raise TypeError("variable_attribution doit etre un mapping valide.")
    hourly = raw.get("hourly")
    audit = raw.get("audit")
    architecture = raw.get("architecture_weights")
    if (
        not isinstance(hourly, pd.DataFrame)
        or hourly.empty
        or not isinstance(audit, Mapping)
        or not isinstance(architecture, Mapping)
    ):
        raise ValueError("Attribution de variables incomplete pour le rapport.")

    variant = str(raw.get("variant", ""))
    method = str(raw.get("method", ""))
    is_upstream_attribution = bool(raw.get("is_upstream_attribution", False))
    explained_model_label = str(
        raw.get("explained_model_label", "modèle amont")
    ).strip()
    reported_model_label = str(
        raw.get("reported_model_label", "prévision Kalman")
    ).strip()
    method_labels = {
        "exact_grouped_shapley_end_to_end": (
            "Shapley groupé exact, recalcul end-to-end"
        ),
        "permutation_grouped_shapley_end_to_end": (
            "Shapley groupé par permutations, recalcul end-to-end"
        ),
    }
    method_label = method_labels.get(method, method)
    group_order = [str(group.get("key")) for group in raw.get("groups", [])]
    if not group_order:
        group_order = list(dict.fromkeys(hourly["variable_key"].astype(str)))

    summary = (
        hourly.groupby("variable_key", sort=False)
        .agg(
            variable_label=("variable_label", "first"),
            baseline_reference=("baseline_reference", "first"),
            weight_pct=("weight_pct", "first"),
            mean_contribution_eur_mwh=("contribution_eur_mwh", "mean"),
            mean_absolute_contribution_eur_mwh=(
                "absolute_contribution_eur_mwh",
                "mean",
            ),
            minimum_contribution_eur_mwh=("contribution_eur_mwh", "min"),
            maximum_contribution_eur_mwh=("contribution_eur_mwh", "max"),
        )
        .reindex(group_order)
        .dropna(subset=["variable_label"])
    )
    if summary.empty:
        raise ValueError("Résumé d'attribution vide pour le rapport.")

    ranked = summary.sort_values("weight_pct", ascending=True, kind="stable")
    bar = go.Figure(
        go.Bar(
            x=ranked["weight_pct"],
            y=ranked["variable_label"],
            orientation="h",
            marker=dict(
                color=ranked["weight_pct"],
                colorscale="Blues",
                showscale=False,
            ),
            customdata=np.column_stack(
                [
                    ranked.index.astype(str),
                    ranked["baseline_reference"].astype(str),
                    ranked["mean_absolute_contribution_eur_mwh"].to_numpy(
                        dtype=float
                    ),
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>Poids d'influence Shapley: %{x:.2f} %"
                "<br>Contribution absolue moyenne: %{customdata[2]:.2f} EUR/MWh"
                "<br>Référence: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    bar_scope = "du modèle amont" if is_upstream_attribution else "du jour"
    bar.update_layout(
        title=(
            f"{result.zone} — part d'influence Shapley sur la prévision P50 "
            f"{bar_scope}"
        ),
        xaxis_title=(
            "Poids d'influence Shapley (%) — part de la contribution "
            "absolue totale"
        ),
        yaxis_title="",
        height=max(360, 56 * len(ranked) + 150),
    )

    local_timestamps = pd.DatetimeIndex(hourly["delivery_start_local"])
    timestamp_order = pd.DatetimeIndex(local_timestamps.unique()).sort_values()
    heatmap_frame = hourly.assign(_local_timestamp=local_timestamps).pivot(
        index="variable_key",
        columns="_local_timestamp",
        values="contribution_eur_mwh",
    )
    heatmap_frame = heatmap_frame.reindex(
        index=summary.index,
        columns=timestamp_order,
    )
    heatmap_labels = summary.loc[heatmap_frame.index, "variable_label"].tolist()
    hour_labels = [
        pd.Timestamp(value).strftime("%d/%m %H:%M %Z")
        for value in heatmap_frame.columns
    ]
    heat_values = heatmap_frame.to_numpy(dtype=float)
    color_limit = float(np.nanmax(np.abs(heat_values)))
    if not np.isfinite(color_limit) or color_limit <= 0.0:
        color_limit = 1.0
    heatmap = go.Figure(
        go.Heatmap(
            x=hour_labels,
            y=heatmap_labels,
            z=heat_values,
            zmin=-color_limit,
            zmax=color_limit,
            zmid=0.0,
            colorscale="RdBu_r",
            colorbar=dict(title="EUR/MWh"),
            hovertemplate=(
                "<b>%{y}</b><br>%{x}<br>Contribution signée: "
                "%{z:.2f} EUR/MWh<extra></extra>"
            ),
        )
    )
    heatmap.update_layout(
        title=(
            f"{result.zone} — contributions additives par heure locale"
        ),
        xaxis_title=f"Heure de livraison ({audit.get('timezone', '')})",
        yaxis_title="",
        height=max(390, 62 * len(heatmap_labels) + 170),
    )

    architecture_labels = {
        "autonomous": "Modèle autonome",
        "mkonline_primary": "MKOnline primaire",
    }
    architecture_cards = []
    for key in ("autonomous", "mkonline_primary"):
        value = float(architecture.get(key, math.nan))
        rendered = "—" if not np.isfinite(value) else f"{100.0 * value:.1f}%"
        architecture_cards.append(
            '<div class="attribution-architecture-card">'
            f'<span>{html.escape(architecture_labels[key])}</span>'
            f'<strong>{html.escape(rendered)}</strong>'
            "</div>"
        )

    summary_table = summary.reset_index().rename(
        columns={
            "variable_key": "Clé",
            "variable_label": "Variable",
            "baseline_reference": "Référence contrefactuelle",
            "weight_pct": "Poids d'influence Shapley (%)",
            "mean_contribution_eur_mwh": "Contribution moyenne (EUR/MWh)",
            "mean_absolute_contribution_eur_mwh": (
                "Contribution absolue moyenne (EUR/MWh)"
            ),
            "minimum_contribution_eur_mwh": "Contribution min. (EUR/MWh)",
            "maximum_contribution_eur_mwh": "Contribution max. (EUR/MWh)",
        }
    )
    max_error = float(raw.get("max_reconstruction_error_eur_mwh", math.nan))
    max_error_text = "—" if not np.isfinite(max_error) else f"{max_error:.3g} EUR/MWh"
    bar_div = plotly_div(bar, False)
    heatmap_div = plotly_div(heatmap, False)
    reference_mean = pd.to_numeric(hourly["counterfactual_q50"], errors="raise").mean()
    explained_mean = pd.to_numeric(hourly["forecast_q50"], errors="raise").mean()
    if is_upstream_attribution:
        heading_kicker = "EXPLICATION LOCALE DU MODÈLE AMONT"
        heading_title = "Influence des variables dans la prévision amont"
        heading_text = (
            "Décomposition additive Shapley groupée de la P50 du "
            f"<strong>{html.escape(explained_model_label)}</strong>, calculée "
            "avant application du filtre Kalman. Les diagnostics Kalman "
            f"présentés séparément expliquent ensuite la correction gouvernée "
            f"appliquée pour obtenir <strong>{html.escape(reported_model_label)}</strong>. "
            "Il s'agit d'une <strong>sensibilité locale du modèle amont</strong>, "
            "pas d'un coefficient causal ni d'un effet de marché estimé."
        )
        architecture_title = "Poids d'architecture de la prévision amont"
        audit_scope = "jusqu'à la P50 amont"
        attribution_scope = "upstream-model"
    else:
        heading_kicker = "EXPLICATION LOCALE DU FORECAST"
        heading_title = "Influence des variables dans la prévision finale"
        heading_text = (
            "Décomposition additive Shapley groupée de la P50, calculée "
            "en recalculant les prédictions du pipeline sur le jour affiché, "
            "avec les paramètres appris figés. "
            "Il s'agit d'une <strong>sensibilité locale du modèle</strong>, "
            "pas d'un coefficient causal ni d'un effet de marché estimé."
        )
        architecture_title = "Poids d'architecture de la prévision finale"
        audit_scope = "jusqu'à la P50 finale"
        attribution_scope = "final-forecast"
    return f'''
    <div
        id="variable-attribution-{html.escape(str(result.zone).lower())}"
        class="variable-attribution"
        data-report-section="variable-attribution"
        data-attribution-variant="{html.escape(variant)}"
        data-attribution-scope="{attribution_scope}"
    >
        <div class="attribution-heading">
            <div>
                <div class="attribution-kicker">{heading_kicker}</div>
                <h3>{heading_title}</h3>
                <p>{heading_text}</p>
            </div>
            <span class="attribution-method-badge">{html.escape(method_label)}</span>
        </div>
        <div class="attribution-architecture">
            <div class="attribution-architecture-title">
                {architecture_title}
            </div>
            {''.join(architecture_cards)}
        </div>
        <div class="attribution-chart-grid">
            <div class="attribution-chart-card">{bar_div}</div>
            <div class="attribution-chart-card">{heatmap_div}</div>
        </div>
        <p class="attribution-audit-note">Prix moyen de référence : <strong>{reference_mean:.2f} EUR/MWh</strong>.
        P50 expliquée moyenne : <strong>{explained_mean:.2f} EUR/MWh</strong>.
        Leur différence correspond à la somme des contributions moyennes signées ci-dessous.</p>
        <h4>Tableau de synthèse</h4>
        {html_table(summary_table, float_format="{:.3f}")}
        {attribution_scope_html(raw)}
        <p class="attribution-audit-note">
            Les parts d'influence totalisent 100 % (ou 0 % si toutes les contributions sont nulles) à partir des contributions
            absolues sur toutes les heures. Les contributions signées se
            somment, avec la prévision de référence commune, jusqu'à la P50
            {audit_scope.replace("jusqu'à la P50 ", "")}. Erreur maximale de reconstruction :
            <strong>{html.escape(max_error_text)}</strong>.
            Storm n'est utilisé ni dans cette attribution, ni dans la prévision.
        </p>
    </div>
    '''


def _kalman_paired_evaluation(
    result: ZoneRunResult,
    diagnostics: Mapping[str, Any],
) -> pd.DataFrame:
    """Return the exact native/upstream point-forecast support in the audit."""

    baseline = result.backtest_baseline
    if baseline is None or baseline.empty:
        return pd.DataFrame()
    native = (
        result.backtest_native.loc[:, ["timestamp", "actual", "q50"]]
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp", keep="last")
        .rename(columns={"actual": "actual_native", "q50": "kalman_q50"})
    )
    upstream = (
        baseline.loc[:, ["timestamp", "actual", "q50"]]
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp", keep="last")
        .rename(columns={"actual": "actual_upstream", "q50": "upstream_q50"})
    )
    paired = native.merge(upstream, on="timestamp", how="inner", validate="one_to_one")
    if paired.empty:
        return paired
    numeric = paired[
        ["actual_native", "actual_upstream", "kalman_q50", "upstream_q50"]
    ].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    paired = paired.loc[finite].copy()
    if paired.empty:
        return paired
    if not np.allclose(
        paired["actual_native"].to_numpy(dtype=float),
        paired["actual_upstream"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Les observations Kalman et upstream diffèrent sur le support apparié."
        )
    paired["actual"] = paired.pop("actual_native")
    paired = paired.drop(columns="actual_upstream")
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(paired["timestamp"], utc=True, errors="raise")
    )
    timezone = str(diagnostics.get("timezone") or "Europe/Paris")
    local_days = pd.Series(timestamps.tz_convert(timezone).date, index=paired.index)
    audit = diagnostics.get("audit", {})
    if isinstance(audit, Mapping):
        start_value = audit.get("evaluation_start_day")
        end_value = audit.get("evaluation_end_day")
        if start_value not in (None, ""):
            paired = paired.loc[
                local_days >= pd.Timestamp(start_value).date()
            ].copy()
            local_days = local_days.loc[paired.index]
        if end_value not in (None, ""):
            paired = paired.loc[
                local_days <= pd.Timestamp(end_value).date()
            ].copy()
    return paired.reset_index(drop=True)


def _kalman_point_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def build_kalman_diagnostics_html(result: ZoneRunResult) -> str:
    """Render the optional governed KF/EKF/UKF rolling-window diagnostics."""

    raw = getattr(result, "kalman_diagnostics", None)
    if raw is None:
        return ""
    if not isinstance(raw, Mapping):
        raise TypeError("kalman_diagnostics doit être un mapping valide.")
    audit = raw.get("audit")
    daily = raw.get("daily")
    state = raw.get("state")
    if not isinstance(audit, Mapping):
        raise ValueError("Audit Kalman absent des diagnostics du rapport.")
    if not isinstance(daily, pd.DataFrame) or not isinstance(state, pd.DataFrame):
        raise ValueError("Artefacts Kalman tabulaires invalides.")

    paired = _kalman_paired_evaluation(result, raw)
    if paired.empty:
        raise ValueError("Support apparié Kalman/upstream vide pour le rapport.")
    actual = paired["actual"].to_numpy(dtype=float)
    kalman_metrics = _kalman_point_metrics(
        actual, paired["kalman_q50"].to_numpy(dtype=float)
    )
    upstream_metrics = _kalman_point_metrics(
        actual, paired["upstream_q50"].to_numpy(dtype=float)
    )
    timezone = str(raw.get("timezone") or "Europe/Paris")
    paired_timestamps = pd.DatetimeIndex(
        pd.to_datetime(paired["timestamp"], utc=True, errors="raise")
    )
    paired_days = pd.Index(paired_timestamps.tz_convert(timezone).date).unique()
    paired_day_count = int(len(paired_days))
    paired_hour_count = int(len(paired))
    mae_delta = upstream_metrics["mae"] - kalman_metrics["mae"]
    mae_gain = (
        100.0 * mae_delta / upstream_metrics["mae"]
        if upstream_metrics["mae"] > 0.0
        else math.nan
    )

    metric_rows = pd.DataFrame(
        [
            {
                "Métrique appariée": "MAE P50 (EUR/MWh)",
                "Kalman gouverné": kalman_metrics["mae"],
                "Correcteur résiduel amont": upstream_metrics["mae"],
                "Amélioration": mae_delta,
                "Gain (%)": mae_gain,
            },
            {
                "Métrique appariée": "RMSE P50 (EUR/MWh)",
                "Kalman gouverné": kalman_metrics["rmse"],
                "Correcteur résiduel amont": upstream_metrics["rmse"],
                "Amélioration": (
                    upstream_metrics["rmse"] - kalman_metrics["rmse"]
                ),
                "Gain (%)": (
                    100.0
                    * (upstream_metrics["rmse"] - kalman_metrics["rmse"])
                    / upstream_metrics["rmse"]
                    if upstream_metrics["rmse"] > 0.0
                    else math.nan
                ),
            },
            {
                "Métrique appariée": "Biais P50 (EUR/MWh)",
                "Kalman gouverné": kalman_metrics["bias"],
                "Correcteur résiduel amont": upstream_metrics["bias"],
                "Amélioration": (
                    abs(upstream_metrics["bias"]) - abs(kalman_metrics["bias"])
                ),
                "Gain (%)": (
                    100.0
                    * (
                        abs(upstream_metrics["bias"])
                        - abs(kalman_metrics["bias"])
                    )
                    / abs(upstream_metrics["bias"])
                    if abs(upstream_metrics["bias"]) > 0.0
                    else math.nan
                ),
            },
        ]
    )

    evaluation_start = pd.Timestamp(audit.get("evaluation_start_day")).date()
    evaluation_end = pd.Timestamp(audit.get("evaluation_end_day")).date()
    daily_evaluation = daily.copy()
    if not daily_evaluation.empty:
        daily_evaluation = daily_evaluation.loc[
            (daily_evaluation["local_day"].dt.date >= evaluation_start)
            & (daily_evaluation["local_day"].dt.date <= evaluation_end)
        ].copy()

    configured = [str(value) for value in raw.get("candidate_kinds", ())]
    canonical_order = [
        "identity",
        "linear_bias",
        "linear_harmonic",
        "linear_market",
        "linear_weather",
        "linear_renewables",
        "linear_fundamental",
        "linear_fuel",
        "linear_market_weather",
        "linear_market_weather_fuel",
        "linear_scale",
        "ekf_scale",
        "ukf_scale",
    ]
    filter_order = canonical_order + [
        value for value in configured if value not in canonical_order
    ]
    filter_labels = {
        "identity": "Garde-fou identité (aucune correction)",
        "linear_bias": "KF linéaire — biais lent",
        "linear_harmonic": "KF linéaire — profil harmonique",
        "linear_market": "KF linéaire — charges résiduelles européennes",
        "linear_weather": "KF linéaire — météo et degrés-jours",
        "linear_renewables": "KF linéaire — vent et solaire prévus",
        "linear_fundamental": "KF linéaire — fondamentaux météo-énergie",
        "linear_fuel": "KF linéaire — gaz, CO₂ et dynamique des combustibles",
        "linear_market_weather": "KF linéaire — charges résiduelles et météo",
        "linear_market_weather_fuel": (
            "KF linéaire — charges résiduelles, météo et combustibles"
        ),
        "linear_scale": "KF linéaire — biais + échelle",
        "ekf_scale": "EKF — biais + échelle bornée",
        "ukf_scale": "UKF — biais + échelle bornée",
    }
    if not daily_evaluation.empty:
        counts = daily_evaluation["selected_filter"].value_counts().to_dict()
        weights_by_filter = (
            daily_evaluation.groupby("selected_filter", sort=False)[
                "selected_weight"
            ].mean().to_dict()
        )
        selection_denominator = int(len(daily_evaluation))
    else:
        raw_counts = audit.get("selected_filter_counts", {})
        counts = dict(raw_counts) if isinstance(raw_counts, Mapping) else {}
        weights_by_filter = {}
        selection_denominator = int(sum(int(value) for value in counts.values()))
    filter_rows = []
    for kind in filter_order:
        is_configured = kind == "identity" or kind in configured
        count = int(counts.get(kind, 0)) if is_configured else 0
        frequency = (
            100.0 * count / selection_denominator
            if selection_denominator > 0 and is_configured
            else math.nan
        )
        filter_rows.append(
            {
                "Famille": filter_labels.get(kind, kind),
                "Statut": (
                    "Garde-fou actif"
                    if kind == "identity"
                    else "Candidat évalué"
                    if is_configured
                    else "Non inclus dans ce replay"
                ),
                "Jours sélectionnés": count if is_configured else "—",
                "Fréquence (%)": frequency,
                "Poids moyen appliqué (%)": (
                    100.0 * float(weights_by_filter[kind])
                    if kind in weights_by_filter
                    else math.nan
                ),
            }
        )
    filter_table = pd.DataFrame(filter_rows)

    state_table_html = ""
    if not state.empty:
        evaluation_state = state.loc[
            (state["local_day"].dt.date >= evaluation_start)
            & (state["local_day"].dt.date <= evaluation_end)
        ].copy()
        aggregations: dict[str, tuple[str, str]] = {
            "Jours audités": ("local_day", "nunique"),
        }
        if "mean_innovation" in evaluation_state:
            evaluation_state["_absolute_innovation"] = evaluation_state[
                "mean_innovation"
            ].abs()
            aggregations["Innovation absolue moyenne"] = (
                "_absolute_innovation",
                "mean",
            )
        if "innovation_clips_total" in evaluation_state:
            aggregations["Clips cumulés (fin)"] = (
                "innovation_clips_total",
                "max",
            )
        if "minimum_covariance_eigenvalue" in evaluation_state:
            aggregations["Valeur propre minimale"] = (
                "minimum_covariance_eigenvalue",
                "min",
            )
        if "covariance_repairs_total" in evaluation_state:
            aggregations["Réparations covariance (fin)"] = (
                "covariance_repairs_total",
                "max",
            )
        state_summary = (
            evaluation_state.groupby("filter_kind", sort=False)
            .agg(**aggregations)
            .reset_index()
            .rename(columns={"filter_kind": "Filtre"})
        )
        if not state_summary.empty:
            state_summary["Filtre"] = state_summary["Filtre"].map(
                lambda value: filter_labels.get(str(value), str(value))
            )
            state_table_html = (
                "<h4>Stabilité numérique des états</h4>"
                + html_table(state_summary, float_format="{:.4g}")
            )

    graph_html = ""
    if not daily_evaluation.empty:
        graph = make_subplots(specs=[[{"secondary_y": True}]])
        graph.add_trace(
            go.Bar(
                x=daily_evaluation["local_day"],
                y=daily_evaluation["applied_correction_mean"],
                name="Correction moyenne appliquée",
                marker_color="#2563eb",
                customdata=daily_evaluation[["selected_filter"]].to_numpy(),
                hovertemplate=(
                    "<b>%{x|%d/%m/%Y}</b><br>Correction: %{y:.2f} EUR/MWh"
                    "<br>Filtre: %{customdata[0]}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
        graph.add_trace(
            go.Scatter(
                x=daily_evaluation["local_day"],
                y=100.0 * daily_evaluation["selected_weight"],
                name="Poids gouverné",
                mode="lines",
                line=dict(color="#f59e0b", width=2),
                hovertemplate=(
                    "<b>%{x|%d/%m/%Y}</b><br>Poids: %{y:.1f}%<extra></extra>"
                ),
            ),
            secondary_y=True,
        )
        graph.update_yaxes(
            title_text="Correction moyenne (EUR/MWh)", secondary_y=False
        )
        graph.update_yaxes(title_text="Poids (%)", range=[0, 105], secondary_y=True)
        graph.update_xaxes(title_text="Jour local de livraison")
        graph.update_layout(
            title=f"{result.zone} — décisions quotidiennes du gouverneur Kalman"
        )
        graph_html = (
            '<div class="kalman-chart-card">'
            + plotly_div(graph, False)
            + "</div>"
        )

    warmup_days = int(audit.get("warmup_days", 0))
    audited_days = int(audit.get("evaluation_days", paired_day_count))
    audited_hours = int(audit.get("evaluation_hours", paired_hour_count))
    rolling_exact = (
        audited_days == 365
        and paired_day_count == 365
        and audited_hours == paired_hour_count
    )
    window_badge = (
        "Rolling window appariée de 365 jours"
        if rolling_exact
        else f"Fenêtre appariée de {paired_day_count} jours"
    )
    window_warning = ""
    if not rolling_exact:
        window_warning = (
            '<p class="kalman-warning"><strong>Attention :</strong> '
            "la fenêtre effectivement appariée n'est pas exactement de 365 jours "
            f"({paired_day_count} jours, {paired_hour_count} heures), alors que "
            f"l'audit déclare {audited_days} jours et {audited_hours} heures.</p>"
        )

    gain_class = "positive" if mae_delta >= 0.0 else "negative"
    gain_prefix = "+" if np.isfinite(mae_gain) and mae_gain >= 0.0 else ""
    pykalman_version = html.escape(str(audit.get("pykalman_version", "—")))
    quantile_policy = html.escape(str(audit.get("quantile_policy", "—")))
    covariates = audit.get("covariate_columns", [])
    covariate_text = (
        ", ".join(map(str, covariates))
        if isinstance(covariates, list) and covariates
        else "aucune covariable additionnelle déclarée"
    )
    raw_governance = audit.get("config", {})
    if isinstance(raw_governance, Mapping):
        governance_lookback = int(raw_governance.get("governance_lookback_days", 0) or 0)
        governance_confirmation = int(
            raw_governance.get("governance_confirmation_days", 0) or 0
        )
        governance_selection = max(
            0, governance_lookback - governance_confirmation
        )
        governance_step = float(
            raw_governance.get("governance_weight_step", math.nan)
        )
        governance_absolute_gain = float(
            raw_governance.get("minimum_gain_eur_mwh", math.nan)
        )
        governance_relative_gain = float(
            raw_governance.get("minimum_relative_gain", math.nan)
        )
    else:
        governance_lookback = governance_confirmation = governance_selection = 0
        governance_step = governance_absolute_gain = governance_relative_gain = math.nan
    if governance_lookback > 0:
        confirmation_text = (
            f"{governance_selection} j sélection + "
            f"{governance_confirmation} j confirmation"
            if governance_confirmation > 0
            else f"{governance_selection} j sélection, sans confirmation séparée"
        )
        threshold_parts = []
        if np.isfinite(governance_absolute_gain):
            threshold_parts.append(f"{governance_absolute_gain:.2f} EUR/MWh")
        if np.isfinite(governance_relative_gain):
            threshold_parts.append(f"{100.0 * governance_relative_gain:.2f} %")
        threshold_text = " et ".join(threshold_parts) or "non déclaré"
        step_text = (
            f"; pas de poids {governance_step:.2f}"
            if np.isfinite(governance_step)
            else ""
        )
        governance_text = (
            f"{governance_lookback} j ({confirmation_text}); seuils "
            f"{threshold_text}{step_text}"
        )
    else:
        governance_text = "Paramètres de gouvernance non déclarés dans l'audit"
    return f'''
    <div
        id="kalman-diagnostics-{html.escape(str(result.zone).lower())}"
        class="kalman-diagnostics"
        data-report-section="kalman-diagnostics"
        data-kalman-window-days="{paired_day_count}"
    >
        <div class="kalman-heading">
            <div>
                <div class="kalman-kicker">DIAGNOSTIC DU POST-CORRECTEUR</div>
                <h3>Filtre Kalman gouverné — comparaison causale appariée</h3>
                <p>
                    La couche est appliquée après le correcteur résiduel. Chaque
                    décision du jour est figée avant livraison puis les états ne
                    sont mis à jour qu'avec des prix déjà observés.
                </p>
            </div>
            <span class="kalman-method-badge">{html.escape(window_badge)}</span>
        </div>
        <div class="kalman-card-grid">
            <div class="kalman-card">
                <span>MAE Kalman</span>
                <strong>{kalman_metrics['mae']:.3f}</strong>
                <small>EUR/MWh</small>
            </div>
            <div class="kalman-card">
                <span>MAE correcteur amont</span>
                <strong>{upstream_metrics['mae']:.3f}</strong>
                <small>EUR/MWh</small>
            </div>
            <div class="kalman-card {gain_class}">
                <span>Gain MAE apparié</span>
                <strong>{gain_prefix}{mae_gain:.2f}%</strong>
                <small>{mae_delta:+.3f} EUR/MWh</small>
            </div>
            <div class="kalman-card">
                <span>Support évalué</span>
                <strong>{paired_day_count} j</strong>
                <small>{paired_hour_count:,} heures</small>
            </div>
        </div>
        {window_warning}
        <h4>Métriques P50 sur le même support horaire</h4>
        <p class="muted kalman-definition">
            Une amélioration positive signifie une erreur plus faible. Pour le
            biais, l'amélioration porte sur sa valeur absolue.
        </p>
        {html_table(metric_rows, float_format="{:.4f}")}
        <h4>Sélection quotidienne des familles de filtres</h4>
        {html_table(filter_table, float_format="{:.2f}")}
        {graph_html}
        {state_table_html}
        <div class="kalman-audit-grid">
            <div><strong>Warm-up strict</strong><span>{warmup_days} jours — {html.escape(str(audit.get('warmup_start_day', '—')))} au {html.escape(str(audit.get('warmup_end_day', '—')))}</span></div>
            <div><strong>Évaluation</strong><span>{html.escape(str(audit.get('evaluation_start_day', '—')))} au {html.escape(str(audit.get('evaluation_end_day', '—')))}</span></div>
            <div><strong>Moteur</strong><span>pykalman {pykalman_version}, filter_update causal</span></div>
            <div><strong>Quantiles</strong><span>{quantile_policy}</span></div>
            <div><strong>Gouvernance</strong><span>{html.escape(governance_text)}</span></div>
        </div>
        <p class="kalman-causal-note">
            <strong>Contrat causal :</strong> filtre uniquement, aucun smoother,
            aucun EM et aucune observation du jour courant avant la prévision.
            Storm est exclusivement un benchmark d'évaluation : il n'est jamais
            utilisé comme entrée, état ou signal de gouvernance. Covariables PIT
            déclarées : {html.escape(covariate_text)}.
        </p>
        <p class="kalman-warning">
            <strong>Lecture méthodologique :</strong> le warm-up disponible compte
            {warmup_days} jours. Ce replay causal mesure la performance sur la
            fenêtre affichée, mais il ne constitue pas à lui seul un jeu de
            promotion intact si les choix d'architecture ou d'hyperparamètres
            ont été effectués sur cette même période.
        </p>
    </div>
    '''


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

STATISTICS_ROLLING_DAYS = 365


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
    values = frame[["actual", "q50"]].copy()
    values["actual"] = pd.to_numeric(
        values["actual"],
        errors="coerce",
    )
    values["q50"] = pd.to_numeric(
        values["q50"],
        errors="coerce",
    )
    values = values.replace([np.inf, -np.inf], np.nan)
    predicted_values = values["q50"].dropna().to_numpy(dtype=float)
    valid = values.dropna()

    if valid.empty:
        return {
            "observed_mean_price": None,
            "mean_price": _finite_or_none(
                np.mean(predicted_values) if len(predicted_values) else None
            ),
            "mae": None,
            "rmse": None,
            "mape": None,
            "explained_variance": None,
            "r2": None,
            "std_error": None,
            "correlation": None,
            "bias": None,
            "n": 0,
        }

    actual = valid["actual"].to_numpy(dtype=float)
    predicted = valid["q50"].to_numpy(dtype=float)
    error = predicted - actual

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(np.square(error)))
    nonzero = np.abs(actual) > 1e-9
    mape = (
        100.0 * np.mean(np.abs(error[nonzero]) / np.abs(actual[nonzero]))
        if bool(nonzero.any())
        else math.nan
    )
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
        "observed_mean_price": _finite_or_none(np.mean(actual)),
        "mean_price": _finite_or_none(np.mean(predicted)),
        "mae": _finite_or_none(mae),
        "rmse": _finite_or_none(rmse),
        "mape": _finite_or_none(mape),
        "explained_variance": _finite_or_none(
            explained_variance
        ),
        "r2": _finite_or_none(r2),
        "std_error": _finite_or_none(std_error),
        "correlation": _finite_or_none(correlation),
        "bias": _finite_or_none(np.mean(error)),
        "n": int(len(valid)),
    }


def _mean_price_comparison_fields(
    candidate_metrics: Mapping[str, Any],
    benchmark_metrics: Mapping[str, Any],
    *,
    has_benchmark: bool,
) -> dict[str, Any]:
    observed = _finite_or_none(candidate_metrics.get("observed_mean_price"))
    candidate = _finite_or_none(candidate_metrics.get("mean_price"))
    benchmark = _finite_or_none(benchmark_metrics.get("mean_price"))
    candidate_distance = (
        abs(candidate - observed)
        if candidate is not None and observed is not None
        else None
    )
    benchmark_distance = (
        abs(benchmark - observed)
        if has_benchmark and benchmark is not None and observed is not None
        else None
    )
    closest = "unavailable"
    margin = None
    if candidate_distance is not None and benchmark_distance is not None:
        margin = abs(candidate_distance - benchmark_distance)
        if np.isclose(
            candidate_distance,
            benchmark_distance,
            rtol=1e-9,
            atol=1e-12,
        ):
            closest = "tie"
        elif candidate_distance < benchmark_distance:
            closest = "candidate"
        else:
            closest = "benchmark"
    return {
        "mean_price_absolute_error": _finite_or_none(candidate_distance),
        "benchmark_mean_price_absolute_error": _finite_or_none(
            benchmark_distance
        ),
        "mean_price_closest": closest,
        "mean_price_closeness_margin": _finite_or_none(margin),
    }


STATISTICS_METRICS: tuple[dict[str, Any], ...] = (
    {
        "key": "mean_price",
        "label": "Prix moyen (EUR/MWh)",
        "higher_is_better": None,
        "decimals": 2,
    },
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
        "key": "mape",
        "label": "MAPE (%)",
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
)


def _comparison_outcome(
    candidate: Any,
    benchmark: Any,
    *,
    higher_is_better: bool,
) -> str | None:
    """Classify one comparable period, keeping numerical ties separate."""

    candidate_value = _finite_or_none(candidate)
    benchmark_value = _finite_or_none(benchmark)
    if candidate_value is None or benchmark_value is None:
        return None
    if math.isclose(
        candidate_value,
        benchmark_value,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        return "tie"
    candidate_wins = (
        candidate_value > benchmark_value
        if higher_is_better
        else candidate_value < benchmark_value
    )
    return "win" if candidate_wins else "loss"


def _win_rate_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Aggregate period wins using all comparable periods as denominator."""

    counts = {"win": 0, "tie": 0, "loss": 0}
    benchmark_key = f"benchmark_{metric_key}"
    for record in records:
        outcome = _comparison_outcome(
            record.get(metric_key),
            record.get(benchmark_key),
            higher_is_better=higher_is_better,
        )
        if outcome is not None:
            counts[outcome] += 1
    comparable = sum(counts.values())
    return {
        "wins": counts["win"],
        "ties": counts["tie"],
        "losses": counts["loss"],
        "comparable_periods": comparable,
        # Ties remain in the denominator and are also reported separately.
        "win_rate": counts["win"] / comparable if comparable else None,
        "tie_rate": counts["tie"] / comparable if comparable else None,
        "loss_rate": counts["loss"] / comparable if comparable else None,
    }


def _prepare_statistics_frame(
    result: ZoneRunResult,
    prediction_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = (
        prediction_frame if prediction_frame is not None else result.backtest_native
    ).copy()
    if "timestamp" not in frame:
        raise ValueError("La source Statistics ne contient pas timestamp.")
    parsed_timestamp = pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
    if bool(parsed_timestamp.duplicated().any()):
        raise ValueError("La source Statistics contient des timestamps dupliques.")
    sort_columns = [
        column for column in ("timestamp", "origin_timestamp") if column in frame
    ]
    frame = frame.sort_values(sort_columns, kind="stable")

    frame["actual"] = pd.to_numeric(
        frame["actual"],
        errors="coerce",
    )
    frame["q50"] = pd.to_numeric(
        frame["q50"],
        errors="coerce",
    )
    frame["_timestamp_utc"] = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
        utc=True,
    )
    frame["_timestamp_local"] = frame["_timestamp_utc"]

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

    frame = frame.dropna(subset=["_timestamp_local", "q50"])
    frame["_timestamp_naive"] = (
        frame["_timestamp_local"]
        .dt.tz_localize(None)
    )
    return frame


def _latest_statistics_window(
    source: pd.DataFrame,
    *,
    days: int = STATISTICS_ROLLING_DAYS,
) -> pd.DataFrame:
    """Keep the latest local civil days used by every report statistic."""

    if source.empty:
        return source.copy()
    local_days = source["_timestamp_local"].dt.normalize()
    if "actual" in source:
        actual = pd.to_numeric(source["actual"], errors="coerce")
        complete_by_day = pd.DataFrame(
            {"local_day": local_days, "finite": np.isfinite(actual)}
        ).groupby("local_day", sort=True)["finite"].all()
        available_days = pd.Index(complete_by_day.loc[complete_by_day].index)
    else:
        available_days = pd.Index(
            local_days.dropna().drop_duplicates().sort_values()
        )
    if len(available_days) <= int(days):
        return source.copy()
    cutoff = available_days[-int(days)]
    return source.loc[local_days >= cutoff].copy()


def _statistics_source(result: ZoneRunResult) -> pd.DataFrame:
    """Align the candidate and an optional evaluation-only benchmark."""

    candidate_frame = getattr(result, "statistics_candidate", None)
    source = _prepare_statistics_frame(result, candidate_frame)
    source = _latest_statistics_window(source)
    benchmark_frame = getattr(result, "statistics_benchmark", None)
    if benchmark_frame is None:
        source["_benchmark_q50"] = np.nan
        return source

    benchmark = _prepare_statistics_frame(result, benchmark_frame)
    benchmark = benchmark.loc[
        :, ["_timestamp_utc", "actual", "q50"]
    ].rename(
        columns={
            "actual": "_benchmark_actual",
            "q50": "_benchmark_q50",
        }
    )
    source = source.merge(
        benchmark,
        on="_timestamp_utc",
        how="left",
        validate="one_to_one",
    )
    comparable_actual = source[
        ["actual", "_benchmark_actual"]
    ].dropna()
    if not comparable_actual.empty and not np.allclose(
        comparable_actual["actual"].to_numpy(dtype=float),
        comparable_actual["_benchmark_actual"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Les observations du candidat et du benchmark diffèrent."
        )
    candidate_finite = np.isfinite(
        pd.to_numeric(source["q50"], errors="coerce").to_numpy(dtype=float)
    )
    benchmark_finite = np.isfinite(
        pd.to_numeric(
            source["_benchmark_q50"], errors="coerce"
        ).to_numpy(dtype=float)
    )
    actual_finite = np.isfinite(
        pd.to_numeric(source["actual"], errors="coerce").to_numpy(dtype=float)
    )
    if not np.array_equal(
        candidate_finite[actual_finite],
        benchmark_finite[actual_finite],
    ):
        # An explicitly verified report-only snapshot may have audited DST or
        # unpublished-day holes. Keep those civil days visible without filling
        # Storm or discarding the candidate's real observations. Comparisons
        # below already use exact finite triplets; the strict default remains.
        pairing = getattr(result, "statistics_pairing_audit", None)
        valid_pairing = False
        if isinstance(pairing, Mapping):
            declared_values = pairing.get("missing_benchmark_utc", [])
            try:
                declared_timestamps = [pd.Timestamp(value) for value in declared_values]
                aware = all(value.tzinfo is not None and not pd.isna(value) for value in declared_timestamps)
                declared = pd.DatetimeIndex(pd.to_datetime(declared_timestamps, utc=True))
                missing = pd.DatetimeIndex(source.loc[~benchmark_finite, "_timestamp_utc"])
                contract = _statistics_benchmark_contract(result)
                materialization = contract.get("materialization_audit", {})
                valid_pairing = bool(
                    aware and not declared.has_duplicates and declared.is_monotonic_increasing
                    and declared.equals(missing)
                    and pairing.get("status") == "complete"
                    and pairing.get("role") == "verified_official_storm_pairing"
                    and pairing.get("zone") == result.zone
                    and pairing.get("used_for_prediction") is False
                    and pairing.get("expected_hours") == len(source)
                    and contract.get("official_dashboard_metric") is True
                    and contract.get("used_for_prediction") is False
                    and isinstance(materialization, Mapping)
                    and materialization.get("missing_benchmark_utc") == declared_values
                    and candidate_finite.all()
                )
            except (TypeError, ValueError, OverflowError):
                valid_pairing = False
        if not valid_pairing:
            raise ValueError(
                "La couverture du candidat et du benchmark diffère dans "
                "la fenêtre Statistics."
            )
    return source


def _mean_of_daily_means(
    frame: pd.DataFrame,
    *,
    value_column: str,
    timestamp_column: str,
) -> dict[str, Any]:
    """Average hourly prices within each local day, then weight days equally."""

    if frame.empty or value_column not in frame or timestamp_column not in frame:
        return {
            "mean": None,
            "days": 0,
            "hours": 0,
            "start_day": None,
            "end_day": None,
        }
    values = pd.to_numeric(frame[value_column], errors="coerce")
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    valid = pd.DataFrame({"timestamp": timestamps, "value": values}).dropna()
    valid = valid.loc[np.isfinite(valid["value"].to_numpy(dtype=float))].copy()
    if valid.empty:
        return {
            "mean": None,
            "days": 0,
            "hours": 0,
            "start_day": None,
            "end_day": None,
        }
    valid["local_day"] = valid["timestamp"].dt.date
    daily = valid.groupby("local_day", sort=True)["value"].mean()
    return {
        "mean": float(daily.mean()),
        "days": int(len(daily)),
        "hours": int(len(valid)),
        "start_day": str(daily.index[0]),
        "end_day": str(daily.index[-1]),
    }


def average_price_summary(result: ZoneRunResult) -> dict[str, Any]:
    """Build live and causal-backtest daily price averages for reporting."""

    target_timezone = getattr(result.zone_data.target.index, "tz", None)
    live = result.forecast_native.loc[:, ["timestamp", "q50"]].copy()
    live["_timestamp_utc"] = pd.to_datetime(
        live["timestamp"], errors="coerce", utc=True
    )
    live["_timestamp_local"] = live["_timestamp_utc"]
    if target_timezone is not None:
        live["_timestamp_local"] = live["_timestamp_local"].dt.tz_convert(
            target_timezone
        )

    live_benchmark_frame = getattr(result, "forecast_benchmark", None)
    live_benchmark_available = live_benchmark_frame is not None
    if live_benchmark_available:
        live_benchmark = live_benchmark_frame.loc[:, ["timestamp", "q50"]].copy()
        live_benchmark["_timestamp_utc"] = pd.to_datetime(
            live_benchmark["timestamp"], errors="coerce", utc=True
        )
        live_benchmark = live_benchmark.loc[
            :, ["_timestamp_utc", "q50"]
        ].rename(columns={"q50": "_live_benchmark_q50"})
        live = live.merge(
            live_benchmark,
            on="_timestamp_utc",
            how="left",
            validate="one_to_one",
        )
        live = live.dropna(subset=["q50", "_live_benchmark_q50"])

    live_summary = _mean_of_daily_means(
        live,
        value_column="q50",
        timestamp_column="_timestamp_local",
    )
    live_benchmark_summary = _mean_of_daily_means(
        live,
        value_column="_live_benchmark_q50",
        timestamp_column="_timestamp_local",
    )
    live_mean = live_summary["mean"]
    live_benchmark_mean = live_benchmark_summary["mean"]
    live_delta = (
        float(live_mean - live_benchmark_mean)
        if live_mean is not None and live_benchmark_mean is not None
        else None
    )

    source = _statistics_source(result)
    benchmark_available = getattr(result, "statistics_benchmark", None) is not None
    historical_columns = ["_timestamp_local", "q50"]
    if benchmark_available:
        historical_columns.append("_benchmark_q50")
    historical = source.loc[:, historical_columns].copy()
    required = ["q50"] + (["_benchmark_q50"] if benchmark_available else [])
    historical = historical.dropna(subset=required)

    candidate_summary = _mean_of_daily_means(
        historical,
        value_column="q50",
        timestamp_column="_timestamp_local",
    )
    benchmark_summary = _mean_of_daily_means(
        historical,
        value_column="_benchmark_q50",
        timestamp_column="_timestamp_local",
    )
    candidate_mean = candidate_summary["mean"]
    benchmark_mean = benchmark_summary["mean"]
    delta = (
        float(candidate_mean - benchmark_mean)
        if candidate_mean is not None and benchmark_mean is not None
        else None
    )
    benchmark_label = (
        str(_statistics_benchmark_contract(result)["report_label"])
        if benchmark_available
        else "Storm indisponible"
    )
    live_benchmark_label = str(
        getattr(result, "forecast_benchmark_label", "Storm officiel dashboard")
    )
    return {
        "live": live_summary,
        "live_benchmark": live_benchmark_summary,
        "live_benchmark_available": live_benchmark_available,
        "live_benchmark_label": live_benchmark_label,
        "delta_live_benchmark": live_delta,
        "candidate": candidate_summary,
        "benchmark": benchmark_summary,
        "benchmark_available": benchmark_available,
        "benchmark_label": benchmark_label,
        "delta_candidate_benchmark": delta,
    }


def _price_average_value(value: Any, *, signed: bool = False) -> str:
    numeric = _finite_or_none(value)
    if numeric is None:
        return "—"
    return f"{numeric:+.2f}" if signed else f"{numeric:.2f}"


def average_price_cards(result: ZoneRunResult) -> str:
    summary = average_price_summary(result)
    live = summary["live"]
    live_benchmark = summary["live_benchmark"]
    benchmark_available = bool(summary["live_benchmark_available"])
    live_period = (
        live["start_day"]
        if live["start_day"] == live["end_day"]
        else f'{live["start_day"]} → {live["end_day"]}'
    )
    benchmark_detail = (
        f'{live_period} · {live_benchmark["hours"]} h appariées'
        if benchmark_available
        else "Indisponible pour cette zone"
    )
    delta_detail = (
        "Modèle − Storm, mêmes heures du jour"
        if benchmark_available
        else "Comparaison impossible"
    )
    cards = (
        (
            "Prix moyen du jour — modèle",
            _price_average_value(live["mean"]),
            f'{live_period} · {live["hours"]} h · P50',
        ),
        (
            "Prix moyen du jour — Storm",
            _price_average_value(live_benchmark["mean"]),
            benchmark_detail,
        ),
        (
            "Écart du jour modèle − Storm",
            _price_average_value(
                summary["delta_live_benchmark"], signed=True
            ),
            delta_detail,
        ),
    )
    rendered_cards = "".join(
        '<div class="average-price-card">'
        f'<div class="average-price-label">{html.escape(label)}</div>'
        f'<div class="average-price-value">{html.escape(value)}</div>'
        '<div class="average-price-unit">EUR/MWh</div>'
        f'<div class="average-price-detail">{html.escape(detail)}</div>'
        "</div>"
        for label, value, detail in cards
    )
    pairing_note = (
        "Le modèle et Storm sont calculés sur les mêmes timestamps "
        f"appariés ({html.escape(str(summary['live_benchmark_label']))})."
        if benchmark_available
        else "Storm n’est pas disponible pour cette zone."
    )
    return f'''
    <div class="average-price-block" data-report-section="average-prices">
        <h3>Prix moyens</h3>
        <div class="average-price-grid">{rendered_cards}</div>
        <p class="average-price-method">
            Les prix du bandeau portent uniquement sur la journée civile locale
            du forecast. {pairing_note} Storm reste un comparateur de rapport :
            il n’est jamais utilisé comme entrée du modèle.
        </p>
    </div>
    '''


def _statistics_benchmark_contract(
    result: ZoneRunResult,
) -> dict[str, Any]:
    """Return explicit reporting metadata for the active benchmark.

    Older or generic callers may attach a benchmark without a contract.  That
    remains supported, but the report then says that the contract is
    unspecified instead of silently attributing dashboard semantics to it.
    """

    benchmark_label = str(
        getattr(result, "statistics_benchmark_label", "Benchmark")
    )
    raw = getattr(result, "statistics_benchmark_contract", None)
    if isinstance(raw, Mapping):
        contract = dict(raw)
    else:
        contract = {}
    contract.setdefault("id", "unspecified")
    contract.setdefault("label", benchmark_label)
    contract.setdefault("report_label", benchmark_label)
    contract.setdefault("official_dashboard_metric", None)
    contract.setdefault(
        "report_note",
        "contrat non déclaré par l’artefact source.",
    )
    return contract


def figure_storm_comparison(result: ZoneRunResult) -> go.Figure:
    """Compare the frozen candidate with its declared evaluation benchmark.

    The figure deliberately consumes the same paired source as ``Statistics``.
    It never reads ``forecast_native`` and therefore cannot expose Storm to the
    operational forecast path.
    """

    if getattr(result, "statistics_benchmark", None) is None:
        raise ValueError(
            "Le graphique Storm exige un benchmark d'évaluation apparié."
        )

    source = _statistics_source(result).dropna(
        subset=["_timestamp_local", "actual", "q50", "_benchmark_q50"]
    )
    if source.empty:
        raise ValueError(
            "Aucune observation appariée pour le graphique candidat vs "
            "benchmark."
        )

    candidate_label = str(
        getattr(result, "statistics_candidate_label", "Candidat")
    )
    benchmark_contract = _statistics_benchmark_contract(result)
    benchmark_label = str(benchmark_contract["report_label"])
    timestamps = source["_timestamp_local"]
    actual = source["actual"].to_numpy(dtype=float)
    candidate = source["q50"].to_numpy(dtype=float)
    benchmark = source["_benchmark_q50"].to_numpy(dtype=float)
    candidate_error = candidate - actual
    benchmark_error = benchmark - actual

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.62, 0.38],
        subplot_titles=(
            "Prix observé et forecasts P50",
            "Erreur absolue par forecast",
        ),
    )
    figure.add_trace(
        go.Scattergl(
            x=timestamps,
            y=actual,
            name="Observé",
            mode="lines",
            line={"color": "#263238", "width": 1.6},
            hovertemplate=(
                "<b>%{x}</b><br>Observé : %{y:.2f} EUR/MWh"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=timestamps,
            y=candidate,
            name=f"{candidate_label} P50",
            mode="lines",
            line={"color": "#1565c0", "width": 1.5},
            customdata=candidate_error,
            hovertemplate=(
                "<b>%{x}</b><br>Forecast : %{y:.2f} EUR/MWh<br>"
                "Erreur signée : %{customdata:.2f} EUR/MWh"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=timestamps,
            y=benchmark,
            name=f"{benchmark_label} P50",
            mode="lines",
            line={"color": "#ef6c00", "width": 1.5, "dash": "dash"},
            customdata=benchmark_error,
            hovertemplate=(
                "<b>%{x}</b><br>Forecast : %{y:.2f} EUR/MWh<br>"
                "Erreur signée : %{customdata:.2f} EUR/MWh"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=timestamps,
            y=np.abs(candidate_error),
            name=f"Erreur absolue {candidate_label}",
            mode="lines",
            line={"color": "#1565c0", "width": 1.3},
            hovertemplate=(
                "<b>%{x}</b><br>Erreur absolue : %{y:.2f} EUR/MWh"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=timestamps,
            y=np.abs(benchmark_error),
            name=f"Erreur absolue {benchmark_label}",
            mode="lines",
            line={"color": "#ef6c00", "width": 1.3, "dash": "dash"},
            hovertemplate=(
                "<b>%{x}</b><br>Erreur absolue : %{y:.2f} EUR/MWh"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="EUR/MWh", row=1, col=1)
    figure.update_yaxes(title_text="Erreur absolue (EUR/MWh)", row=2, col=1)
    figure.update_xaxes(
        title_text="Date de livraison",
        rangeslider={"visible": True},
        row=2,
        col=1,
    )
    figure.update_layout(
        title=(
            f"{result.zone} — comparaison historique candidat vs "
            f"{benchmark_label} "
            "(évaluation uniquement)"
        ),
        height=760,
        hovermode="x unified",
        meta={
            "evaluation_only": True,
            "source": "statistics_history_or_sealed_backtest",
        },
    )
    return figure


def build_statistics_records(
    results: Sequence[ZoneRunResult],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for result in results:
        source = _statistics_source(result)

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
                has_benchmark = bool(
                    block["_benchmark_q50"].notna().any()
                )
                observed_available = bool(
                    pd.to_numeric(block["actual"], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .notna()
                    .any()
                )
                paired = (
                    block.dropna(
                        subset=["actual", "q50", "_benchmark_q50"]
                    )
                    if has_benchmark and observed_available
                    else block
                )
                metrics = _sample_metrics(paired)
                if has_benchmark:
                    benchmark_block = paired.loc[
                        :, ["actual", "_benchmark_q50"]
                    ].rename(columns={"_benchmark_q50": "q50"})
                    benchmark_metrics = _sample_metrics(benchmark_block)
                else:
                    benchmark_metrics = {
                        key: (0 if key == "n" else None)
                        for key in metrics
                    }
                price_comparison = _mean_price_comparison_fields(
                    metrics,
                    benchmark_metrics,
                    has_benchmark=has_benchmark,
                )

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
                        **{
                            f"benchmark_{key}": value
                            for key, value in benchmark_metrics.items()
                        },
                        **price_comparison,
                    }
                )

    return records


def _statistics_comparison_payload(
    results: Sequence[ZoneRunResult],
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    zones: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for result in results:
        benchmark = getattr(result, "statistics_benchmark", None)
        candidate_label = str(
            getattr(result, "statistics_candidate_label", "Candidat")
        )
        benchmark_contract = _statistics_benchmark_contract(result)
        benchmark_label = str(benchmark_contract["report_label"])
        has_benchmark = benchmark is not None
        zones.append(
            {
                "key": result.zone,
                "candidate_label": candidate_label,
                "benchmark_label": benchmark_label,
                "benchmark_contract_id": benchmark_contract["id"],
                "benchmark_is_official_dashboard": benchmark_contract[
                    "official_dashboard_metric"
                ],
                "has_benchmark": has_benchmark,
            }
        )
        if not has_benchmark:
            continue
        for sample in ("daily", "weekly", "monthly"):
            selected = [
                record
                for record in records
                if record.get("zone") == result.zone
                and record.get("sample") == sample
            ]
            for metric in STATISTICS_METRICS:
                if metric["higher_is_better"] is None:
                    continue
                summary = _win_rate_summary(
                    selected,
                    metric_key=str(metric["key"]),
                    higher_is_better=bool(
                        metric["higher_is_better"]
                    ),
                )
                summaries.append(
                    {
                        "zone": result.zone,
                        "sample": sample,
                        "metric": metric["key"],
                        **summary,
                    }
                )
    return zones, summaries


def _overall_benchmark_summary_html(
    results: Sequence[ZoneRunResult],
) -> str:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    contract_notes: list[str] = []
    active_labels: list[str] = []
    for result in results:
        if getattr(result, "statistics_benchmark", None) is None:
            continue
        source = _statistics_source(result).dropna(
            subset=["actual", "q50", "_benchmark_q50"]
        )
        candidate = _sample_metrics(source)
        benchmark = _sample_metrics(
            source.loc[:, ["actual", "_benchmark_q50"]].rename(
                columns={"_benchmark_q50": "q50"}
            )
        )
        candidate_label = str(
            getattr(result, "statistics_candidate_label", "Candidat")
        )
        benchmark_contract = _statistics_benchmark_contract(result)
        benchmark_label = str(benchmark_contract["report_label"])
        active_labels.append(benchmark_label)
        for label, metrics in (
            (candidate_label, candidate),
            (benchmark_label, benchmark),
        ):
            rows.append(
                {
                    "Zone": result.zone,
                    "Forecast": label,
                    "Prix moyen observé (EUR/MWh)": candidate[
                        "observed_mean_price"
                    ],
                    "Prix moyen (EUR/MWh)": metrics["mean_price"],
                    "MAE": metrics["mae"],
                    "RMSE": metrics["rmse"],
                    "MAPE (%)": metrics["mape"],
                    "Biais signé": metrics["bias"],
                    "Corrélation": metrics["correlation"],
                    "Heures": metrics["n"],
                }
            )

        hourly_records = [
            {
                "absolute_error": abs(candidate_value - actual_value),
                "benchmark_absolute_error": abs(
                    benchmark_value - actual_value
                ),
            }
            for candidate_value, benchmark_value, actual_value in zip(
                source["q50"].to_numpy(dtype=float),
                source["_benchmark_q50"].to_numpy(dtype=float),
                source["actual"].to_numpy(dtype=float),
            )
        ]
        hourly = _win_rate_summary(
            hourly_records,
            metric_key="absolute_error",
            higher_is_better=False,
        )
        notes.append(
            "<strong>"
            + html.escape(result.zone)
            + " — win rate horaire sur l’erreur absolue :</strong> "
            + f"{100.0 * float(hourly['win_rate']):.2f} % "
            + f"({hourly['wins']} victoires, {hourly['ties']} égalités, "
            + f"{hourly['losses']} défaites; n={hourly['comparable_periods']})."
        )
        contract_notes.append(
            "<strong>"
            + html.escape(result.zone)
            + " — contrat actif : </strong>"
            + html.escape(str(benchmark_contract["report_label"]))
            + " — "
            + html.escape(str(benchmark_contract["report_note"]))
        )
    if not rows:
        return ""
    unique_labels = list(dict.fromkeys(active_labels))
    comparison_label = (
        unique_labels[0]
        if len(unique_labels) == 1
        else "benchmarks déclarés"
    )
    return f'''
    <div class="statistics-overall-summary">
        <h3>Résumé candidat vs {html.escape(comparison_label)} — 365 derniers jours</h3>
        {html_table(pd.DataFrame(rows))}
        <p class="statistics-definition">{'<br>'.join(notes)}</p>
        <p class="statistics-definition"><strong>Contrat du benchmark :</strong><br>
        {'<br>'.join(contract_notes)}<br>Le benchmark n’est utilisé ni comme
        variable, ni comme expert, ni pour le forecast live.</p>
        <p class="statistics-definition"><strong>Définition :</strong>
        le prix moyen est la moyenne des prix P50 horaires sur la fenêtre
        affichée; il s’agit d’un niveau de prix, sans notion de victoire.
        une heure est gagnée lorsque l’erreur absolue du candidat est
        strictement inférieure à celle du benchmark actif. Les égalités sont
        conservées séparément et restent dans le dénominateur. La MAPE utilise
        la valeur absolue du prix réalisé au dénominateur et exclut uniquement
        les observations dont |prix réalisé| ≤ 1e-9 EUR/MWh.</p>
    </div>
    '''


def build_statistics_table_html(
    results: Sequence[ZoneRunResult],
) -> str:
    records = build_statistics_records(results)
    zones, comparison_summaries = _statistics_comparison_payload(
        results, records
    )
    overall_summary = _overall_benchmark_summary_html(results)
    scope_notes = [
        str(getattr(result, "statistics_scope_note", "")).strip()
        for result in results
        if str(getattr(result, "statistics_scope_note", "")).strip()
    ]
    scope_note_html = ""
    if scope_notes:
        scope_note_html = (
            '<p class="statistics-definition"><strong>Perimetre Statistics :</strong> '
            + "<br>".join(
                html.escape(note) for note in dict.fromkeys(scope_notes)
            )
            + "</p>"
        )
    freshness_cards: list[str] = []
    price_freshness_notes: list[str] = []
    for result in results:
        freshness = getattr(result, "statistics_freshness", None)
        if not isinstance(freshness, Mapping):
            continue
        timezone = str(freshness.get("timezone") or "UTC")

        def local_timestamp(value: Any) -> str:
            if value in (None, ""):
                return "—"
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            return timestamp.tz_convert(timezone).strftime(
                "%d/%m/%Y %H:%M %Z"
            )

        observed_end = local_timestamp(
            freshness.get("actual_available_end_utc")
            or freshness.get("actual_applied_end_utc")
        )
        common_end = local_timestamp(freshness.get("common_delivery_end_utc"))
        actual_extraction = local_timestamp(
            freshness.get("actual_extracted_at_utc")
        )
        storm_available = bool(freshness.get("storm_available"))
        storm_extraction = local_timestamp(
            freshness.get("storm_extracted_at_utc")
        )
        last_day_value = freshness.get("last_complete_common_day_local")
        last_day = (
            pd.Timestamp(last_day_value).strftime("%d/%m/%Y")
            if last_day_value not in (None, "")
            else "—"
        )
        paired_hours = freshness.get("common_hours_last_day")
        expected_hours = freshness.get("expected_common_hours_last_day")
        coverage = (
            f"{int(paired_hours)}/{int(expected_hours)} h"
            if paired_hours is not None and expected_hours is not None
            else "couverture non disponible"
        )
        if storm_available:
            comparison = (
                f"Comparaison modèle / observé / Storm actualisée jusqu’au "
                f"{common_end} · dernière journée complète : {last_day} "
                f"({coverage}) · extraction observé : {actual_extraction} · "
                f"extraction Storm : {storm_extraction}."
            )
            price_refresh = (
                f"{result.zone} — actualisation automatique jusqu’au "
                f"{last_day} ({coverage}) · observé extrait le "
                f"{actual_extraction} · Storm extrait le {storm_extraction}."
            )
        else:
            comparison = (
                f"Comparaison modèle / observé actualisée jusqu’au {common_end} "
                f"· dernière journée complète : {last_day} ({coverage}) · "
                f"extraction observé : {actual_extraction}. Storm officiel "
                "indisponible pour cette zone."
            )
            price_refresh = (
                f"{result.zone} — actualisation automatique jusqu’au "
                f"{last_day} ({coverage}) · observé extrait le "
                f"{actual_extraction} · Storm officiel indisponible."
            )
        price_freshness_notes.append(
            '<span class="statistics-price-refresh-item">'
            + html.escape(price_refresh)
            + "</span>"
        )
        freshness_cards.append(
            '<div class="statistics-freshness-card">'
            f'<strong>{html.escape(result.zone)} — observé disponible jusqu’au '
            f'{html.escape(observed_end)}</strong>'
            f'<span>{html.escape(comparison)}</span>'
            "</div>"
        )
    freshness_html = (
        '<div class="statistics-freshness-grid">'
        + "".join(freshness_cards)
        + "</div>"
        if freshness_cards
        else ""
    )
    price_freshness_html = (
        '<div class="statistics-price-refresh-status" '
        'data-report-subsection="mean-price-refresh">'
        '<strong>Rafraîchissement :</strong> '
        + " ".join(price_freshness_notes)
        + "</div>"
        if price_freshness_notes
        else ""
    )

    payload = {
        "zones": zones,
        "records": records,
        "metrics": list(STATISTICS_METRICS),
        "comparison_summaries": comparison_summaries,
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
    <section id="statistics" class="statistics-section" data-report-section="statistics">
        <div class="statistics-title-pill">Statistics</div>

        {freshness_html}
        {overall_summary}
        {scope_note_html}

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
            id="statistics-win-rate-summary"
            class="statistics-win-rate-summary"
        ></div>

        <div
            id="statistics-table-container"
            class="statistics-table-container"
        ></div>

        <div
            class="statistics-price-comparison"
            data-report-subsection="mean-price-comparison"
        >
            <div class="statistics-price-header">
                <div>
                    <div class="statistics-price-kicker">
                        STATISTICS — PRIX MOYENS
                    </div>
                    <h3>Prix observé vs modèle et Storm</h3>
                    <p>
                        Les écarts comparent le niveau de prix moyen de chaque
                        forecast au prix moyen réellement observé sur la même
                        période et les mêmes heures.
                    </p>
                    <div
                        class="statistics-price-color-legend"
                        aria-label="Légende des couleurs du tableau des prix moyens"
                    >
                        <span class="observed">Prix observé</span>
                        <span class="candidate">Notre modèle</span>
                        <span class="benchmark">Storm</span>
                        <span class="distance">
                            Écart absolu : faible → élevé
                        </span>
                    </div>
                    {price_freshness_html}
                </div>
                <label class="statistics-filter">
                    <span>Sample</span>
                    <select id="statistics-price-sample-select"></select>
                </label>
            </div>
            <div
                id="statistics-price-closeness-summary"
                class="statistics-price-closeness-summary"
            ></div>
            <div
                class="statistics-price-calendar-panel"
                data-report-subsection="mean-price-calendar"
            >
                <div class="statistics-price-calendar-header">
                    <div>
                        <h4>Calendrier des écarts du modèle à l’observé</h4>
                        <p>
                            Plus une case est rouge, plus le prix moyen du
                            modèle s’écarte du prix moyen observé. Survolez une
                            case pour les valeurs, puis cliquez pour conserver
                            le détail sous le calendrier.
                        </p>
                    </div>
                    <div class="statistics-calendar-legend" aria-label="Échelle des écarts">
                        <span>Écart faible</span>
                        <span class="statistics-calendar-legend-bar"></span>
                        <span>Écart élevé</span>
                    </div>
                </div>
                <div id="statistics-price-calendar-container"></div>
                <div
                    id="statistics-price-calendar-detail"
                    class="statistics-price-calendar-detail"
                    aria-live="polite"
                ></div>
            </div>
            <div
                id="statistics-price-table-container"
                class="statistics-table-container"
            ></div>
        </div>
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
        const comparisonContainer = document.getElementById(
            "statistics-win-rate-summary"
        );
        const priceSampleSelect = document.getElementById(
            "statistics-price-sample-select"
        );
        const priceContainer = document.getElementById(
            "statistics-price-table-container"
        );
        const priceSummaryContainer = document.getElementById(
            "statistics-price-closeness-summary"
        );
        const priceCalendarContainer = document.getElementById(
            "statistics-price-calendar-container"
        );
        const priceCalendarDetail = document.getElementById(
            "statistics-price-calendar-detail"
        );
        let activePriceCalendarRows = [];

        payload.metrics.forEach((metric) => {{
            const option = document.createElement("option");
            option.value = metric.key;
            option.textContent = metric.label;
            metricSelect.appendChild(option);
        }});

        payload.samples.forEach((sample) => {{
            [sampleSelect, priceSampleSelect].forEach((select) => {{
                const option = document.createElement("option");
                option.value = sample.key;
                option.textContent = sample.label;
                select.appendChild(option);
            }});
        }});

        metricSelect.value = "mae";
        sampleSelect.value = "weekly";
        priceSampleSelect.value = "daily";

        function statisticsPalette() {{
            if (document.documentElement.dataset.theme === "dark") {{
                return {{
                    blue: [32, 78, 133],
                    neutral: [39, 50, 66],
                    orange: [148, 92, 25],
                    red: [126, 48, 61],
                    distanceLow: [20, 83, 45],
                    distanceMid: [120, 75, 20],
                    distanceHigh: [105, 40, 50],
                }};
            }}
            return {{
                blue: [91, 157, 211],
                neutral: [247, 248, 250],
                orange: [245, 158, 11],
                red: [202, 77, 84],
                distanceLow: [220, 252, 231],
                distanceMid: [254, 240, 180],
                distanceHigh: [254, 202, 202],
            }};
        }}

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

            const palette = statisticsPalette();

            if (ratio <= 0.5) {{
                return interpolateColor(
                    palette.blue,
                    palette.neutral,
                    ratio * 2
                );
            }}

            return interpolateColor(
                palette.neutral,
                palette.red,
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

        function nearlyEqual(left, right) {{
            if (!Number.isFinite(left) || !Number.isFinite(right)) {{
                return false;
            }}
            const tolerance = Math.max(
                1e-12,
                1e-9 * Math.max(Math.abs(left), Math.abs(right))
            );
            return Math.abs(left - right) <= tolerance;
        }}

        function comparisonStatus(
            candidate,
            benchmark,
            higherIsBetter,
            benchmarkLabel
        ) {{
            if (!Number.isFinite(candidate) || !Number.isFinite(benchmark)) {{
                return "non comparable";
            }}
            if (nearlyEqual(candidate, benchmark)) {{
                return typeof higherIsBetter === "boolean"
                    ? "égalité"
                    : "niveau identique";
            }}
            if (typeof higherIsBetter !== "boolean") {{
                return "écart de niveau candidat − benchmark";
            }}
            const win = higherIsBetter
                ? candidate > benchmark
                : candidate < benchmark;
            return win
                ? "victoire candidat"
                : `victoire ${{benchmarkLabel}}`;
        }}

        function deltaColor(
            delta,
            maximumAbsolute,
            higherIsBetter,
            candidate,
            benchmark
        ) {{
            if (!Number.isFinite(delta)) {{
                return "transparent";
            }}
            const palette = statisticsPalette();
            if (nearlyEqual(candidate, benchmark)) {{
                return `rgb(${{palette.neutral.join(",")}})`;
            }}
            const ratio = maximumAbsolute > 1e-12
                ? Math.min(1, Math.abs(delta) / maximumAbsolute)
                : 0;
            if (typeof higherIsBetter !== "boolean") {{
                return interpolateColor(
                    palette.neutral,
                    delta < 0 ? palette.blue : palette.red,
                    0.2 + 0.65 * ratio
                );
            }}
            const good = higherIsBetter ? delta > 0 : delta < 0;
            return interpolateColor(
                palette.neutral,
                good ? palette.blue : palette.red,
                0.2 + 0.65 * ratio
            );
        }}

        function formatPercent(value) {{
            return Number.isFinite(value)
                ? `${{(100 * value).toFixed(1)}} %`
                : "—";
        }}

        function renderComparisonSummary(metricKey, sampleKey) {{
            const summaries = payload.comparison_summaries.filter(
                (item) => item.metric === metricKey
                    && item.sample === sampleKey
            );
            if (!summaries.length) {{
                comparisonContainer.innerHTML = "";
                return;
            }}
            const sample = payload.samples.find(
                (item) => item.key === sampleKey
            );
            const cards = summaries.map((summary) => {{
                const zoneInfo = payload.zones.find(
                    (item) => item.key === summary.zone
                );
                const benchmarkLabel = zoneInfo
                    ? zoneInfo.benchmark_label
                    : "benchmark";
                return `
                    <div class="statistics-win-rate-card">
                        <div class="statistics-win-rate-heading">
                            ${{summary.zone}} — Win rate vs ${{benchmarkLabel}}
                        </div>
                        <div class="statistics-win-rate-value">
                            ${{formatPercent(summary.win_rate)}}
                        </div>
                        <div class="statistics-win-rate-detail">
                            ${{summary.wins}} victoire(s) ·
                            ${{summary.ties}} égalité(s) ·
                            ${{summary.losses}} défaite(s) ·
                            ${{summary.comparable_periods}} période(s)
                        </div>
                    </div>
                `;
            }}).join("");
            comparisonContainer.innerHTML = `
                ${{cards}}
                <p class="statistics-definition">
                    Pour la statistique et l’échantillonnage
                    <strong>${{sample ? sample.label : sampleKey}}</strong>
                    sélectionnés, le win rate vaut périodes gagnées / périodes
                    comparables. Les égalités sont affichées séparément et
                    restent dans le dénominateur. Une valeur basse gagne pour
                    MAE, RMSE, MAPE et écart-type; une valeur haute gagne pour
                    variance expliquée, R² et corrélation. Les périodes
                    hebdomadaires ou mensuelles situées aux bords de la
                    fenêtre peuvent être partielles; leur nombre d’heures
                    est affiché dans la cellule. Le prix moyen est un niveau
                    de prix et ne produit donc pas de win rate.
                </p>
            `;
        }}

        function buildPeriodRows(sampleKey, metricKey) {{
            const periods = new Map();
            payload.records
                .filter((record) => record.sample === sampleKey)
                .forEach((record) => {{
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
                            candidate: record[metricKey],
                            benchmark: record[`benchmark_${{metricKey}}`],
                            observed: record.observed_mean_price,
                            candidateDistance:
                                record.mean_price_absolute_error,
                            benchmarkDistance:
                                record.benchmark_mean_price_absolute_error,
                            closest: record.mean_price_closest,
                            closenessMargin:
                                record.mean_price_closeness_margin,
                            n: record.n,
                            benchmarkN: record.benchmark_n,
                        }};
                }});
            return Array.from(periods.values()).sort(
                (left, right) =>
                    right.periodStart.localeCompare(left.periodStart)
            );
        }}

        function renderStatisticsTable() {{
            const metricKey = metricSelect.value;
            const sampleKey = sampleSelect.value;

            const metric = payload.metrics.find(
                (item) => item.key === metricKey
            );

            const rows = buildPeriodRows(sampleKey, metricKey);

            const displayedValues = [];

            rows.forEach((row) => {{
                payload.zones.forEach((zoneInfo) => {{
                    const entry = row.values[zoneInfo.key];
                    if (
                        metricKey === "mean_price"
                        && entry
                        && Number.isFinite(entry.observed)
                    ) {{
                        displayedValues.push(entry.observed);
                    }}
                    if (
                        entry
                        && Number.isFinite(entry.candidate)
                    ) {{
                        displayedValues.push(entry.candidate);
                    }}
                    if (
                        entry
                        && zoneInfo.has_benchmark
                        && Number.isFinite(entry.benchmark)
                    ) {{
                        displayedValues.push(entry.benchmark);
                    }}
                }});
            }});

            const displayedDeltas = [];
            rows.forEach((row) => {{
                payload.zones.forEach((zoneInfo) => {{
                    const entry = row.values[zoneInfo.key];
                    if (
                        entry
                        && zoneInfo.has_benchmark
                        && Number.isFinite(entry.candidate)
                        && Number.isFinite(entry.benchmark)
                    ) {{
                        displayedDeltas.push(
                            entry.candidate - entry.benchmark
                        );
                    }}
                }});
            }});

            const minimum = displayedValues.length
                ? Math.min(...displayedValues)
                : 0;
            const maximum = displayedValues.length
                ? Math.max(...displayedValues)
                : 0;
            const maximumAbsoluteDelta = displayedDeltas.length
                ? Math.max(...displayedDeltas.map(Math.abs))
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

            payload.zones.forEach((zoneInfo) => {{
                if (metricKey === "mean_price") {{
                    headerParts.push(
                        `<th class="statistics-zone-header">`
                        + `${{zoneInfo.key}}<br>`
                        + `<small>Prix observé</small></th>`
                    );
                }}
                headerParts.push(
                    `<th class="statistics-zone-header">`
                    + `${{zoneInfo.key}}<br>`
                    + `<small>${{zoneInfo.candidate_label}}</small></th>`
                );
                if (zoneInfo.has_benchmark) {{
                    headerParts.push(
                        `<th class="statistics-zone-header">`
                        + `${{zoneInfo.key}}<br>`
                        + `<small>${{zoneInfo.benchmark_label}}</small></th>`,
                        `<th class="statistics-zone-header">`
                        + `${{zoneInfo.key}}<br>`
                        + `<small>Δ candidat − `
                        + `${{zoneInfo.benchmark_label}}</small></th>`
                    );
                }}
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

                payload.zones.forEach((zoneInfo) => {{
                    const entry = row.values[zoneInfo.key];
                    const value = entry ? entry.candidate : null;
                    const n = entry ? entry.n : 0;

                    if (metricKey === "mean_price") {{
                        const observedValue = entry
                            ? entry.observed
                            : null;
                        const observedBackground = cellColor(
                            observedValue,
                            minimum,
                            maximum,
                            false
                        );
                        cells.push(
                            `<td class="statistics-value-cell `
                            + `statistics-observed-cell"`
                            + ` style="background:${{observedBackground}}"`
                            + ` title="Prix moyen observé · n=${{n}}">`
                            + `${{formatValue(observedValue, 2)}}</td>`
                        );
                    }}

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
                    if (zoneInfo.has_benchmark) {{
                        const benchmarkValue = entry
                            ? entry.benchmark
                            : null;
                        const benchmarkN = entry
                            ? entry.benchmarkN
                            : 0;
                        const benchmarkBackground = cellColor(
                            benchmarkValue,
                            minimum,
                            maximum,
                            metric.higher_is_better
                        );
                        const benchmarkRendered = formatValue(
                            benchmarkValue,
                            metric.decimals
                        );
                        const delta = (
                            Number.isFinite(value)
                            && Number.isFinite(benchmarkValue)
                        ) ? value - benchmarkValue : null;
                        const deltaRendered = formatValue(
                            delta,
                            metric.decimals
                        );
                        const deltaBackground = deltaColor(
                            delta,
                            maximumAbsoluteDelta,
                            metric.higher_is_better,
                            value,
                            benchmarkValue
                        );
                        const status = comparisonStatus(
                            value,
                            benchmarkValue,
                            metric.higher_is_better,
                            zoneInfo.benchmark_label
                        );
                        cells.push(
                            `<td class="statistics-value-cell"`
                            + ` style="background:${{benchmarkBackground}}"`
                            + ` title="n=${{benchmarkN}}">`
                            + `${{benchmarkRendered}}</td>`,
                            `<td class="statistics-value-cell"`
                            + ` style="background:${{deltaBackground}}"`
                            + ` title="candidat − `
                            + `${{zoneInfo.benchmark_label}} · ${{status}}">`
                            + `${{deltaRendered}}</td>`
                        );
                    }}
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
            renderComparisonSummary(metricKey, sampleKey);
        }}

        function priceClosenessIndicator(entry, zoneInfo) {{
            if (
                !entry
                || !zoneInfo.has_benchmark
                || !Number.isFinite(entry.observed)
                || !Number.isFinite(entry.candidate)
                || !Number.isFinite(entry.benchmark)
            ) {{
                return {{
                    kind: "unavailable",
                    label: "Storm indisponible",
                }};
            }}
            const margin = Number.isFinite(entry.closenessMargin)
                ? ` · ${{entry.closenessMargin.toFixed(2)}} EUR/MWh`
                : "";
            if (entry.closest === "candidate") {{
                return {{
                    kind: "candidate",
                    label: `Notre modèle plus proche${{margin}}`,
                }};
            }}
            if (entry.closest === "benchmark") {{
                return {{
                    kind: "benchmark",
                    label: `Storm plus proche${{margin}}`,
                }};
            }}
            return {{kind: "tie", label: "Équidistance"}};
        }}

        function renderPriceClosenessSummary(rows) {{
            const cards = payload.zones.map((zoneInfo) => {{
                const counts = {{candidate: 0, benchmark: 0, tie: 0}};
                rows.forEach((row) => {{
                    const entry = row.values[zoneInfo.key];
                    if (entry && Object.hasOwn(counts, entry.closest)) {{
                        counts[entry.closest] += 1;
                    }}
                }});
                if (!zoneInfo.has_benchmark) {{
                    return `
                        <div class="statistics-price-summary-card">
                            <div class="statistics-win-rate-heading">
                                ${{zoneInfo.key}} — proximité à l’observé
                            </div>
                            <div class="statistics-price-summary-value">
                                Storm indisponible
                            </div>
                            <div class="statistics-win-rate-detail">
                                L’écart du modèle à l’observé reste affiché.
                            </div>
                        </div>
                    `;
                }}
                const comparable = (
                    counts.candidate + counts.benchmark + counts.tie
                );
                const candidateRate = comparable
                    ? 100 * counts.candidate / comparable
                    : 0;
                const benchmarkRate = comparable
                    ? 100 * counts.benchmark / comparable
                    : 0;
                const leader = counts.candidate === counts.benchmark
                    ? "Même nombre de périodes proches de l’observé"
                    : counts.candidate > counts.benchmark
                    ? "Notre modèle est plus proche de l’observé sur davantage de périodes"
                    : "Storm est plus proche de l’observé sur davantage de périodes";
                return `
                    <div class="statistics-price-summary-card">
                        <div class="statistics-win-rate-heading">
                            ${{zoneInfo.key}} — proximité à l’observé
                        </div>
                        <div class="statistics-price-summary-value">
                            ${{leader}}
                        </div>
                        <div class="statistics-win-rate-detail">
                            Modèle : ${{counts.candidate}}
                            (${{candidateRate.toFixed(1)}} %) ·
                            Storm : ${{counts.benchmark}}
                            (${{benchmarkRate.toFixed(1)}} %) ·
                            Équidistances : ${{counts.tie}}
                        </div>
                    </div>
                `;
            }}).join("");
            priceSummaryContainer.innerHTML = cards;
        }}

        function calendarScaleMaximum(values) {{
            const sorted = values
                .filter(Number.isFinite)
                .sort((left, right) => left - right);
            if (!sorted.length) {{
                return 0;
            }}
            const index = Math.min(
                sorted.length - 1,
                Math.floor(0.95 * (sorted.length - 1))
            );
            return sorted[index];
        }}

        function calendarErrorColor(value, scaleMaximum) {{
            if (!Number.isFinite(value)) {{
                return "transparent";
            }}
            const palette = statisticsPalette();
            const ratio = scaleMaximum > 1e-12
                ? Math.min(1, Math.max(0, value / scaleMaximum))
                : 0;
            if (ratio <= 0.55) {{
                return interpolateColor(
                    palette.distanceLow,
                    palette.distanceMid,
                    ratio / 0.55
                );
            }}
            return interpolateColor(
                palette.distanceMid,
                palette.distanceHigh,
                (ratio - 0.55) / 0.45
            );
        }}

        function calendarCellHtml(
            zoneInfo,
            row,
            entry,
            scaleMaximum,
            showValue
        ) {{
            if (!row || !entry || !Number.isFinite(entry.candidateDistance)) {{
                return '<span class="statistics-calendar-empty-cell"></span>';
            }}
            const error = entry.candidateDistance;
            const background = calendarErrorColor(error, scaleMaximum);
            const label = (
                `${{zoneInfo.key}} · ${{row.timestamp}} · `
                + `écart modèle-observé ${{error.toFixed(2)}} EUR/MWh · `
                + `observé ${{formatValue(entry.observed, 2)}} · `
                + `modèle ${{formatValue(entry.candidate, 2)}}`
            );
            return (
                `<button type="button" class="statistics-calendar-cell"`
                + ` data-zone="${{zoneInfo.key}}"`
                + ` data-period="${{row.periodKey}}"`
                + ` aria-label="${{label}}" aria-pressed="false"`
                + ` title="${{label}}"`
                + ` style="background:${{background}}">`
                + `${{showValue ? error.toFixed(0) : ""}}</button>`
            );
        }}

        function dailyCalendarHtml(zoneInfo, rows, scaleMaximum) {{
            const available = rows
                .map((row) => ({{
                    row,
                    entry: row.values[zoneInfo.key],
                }}))
                .filter((item) => item.entry);
            if (!available.length) {{
                return '<p class="muted">Aucune donnée quotidienne.</p>';
            }}
            const byDay = new Map(
                available.map((item) => [item.row.periodStart, item])
            );
            const first = new Date(
                `${{available.at(-1).row.periodStart}}T00:00:00Z`
            );
            const last = new Date(
                `${{available[0].row.periodStart}}T00:00:00Z`
            );
            const calendarStart = new Date(first);
            const firstWeekday = (calendarStart.getUTCDay() + 6) % 7;
            calendarStart.setUTCDate(calendarStart.getUTCDate() - firstWeekday);
            const weekCount = Math.floor(
                (last - calendarStart) / (7 * 24 * 60 * 60 * 1000)
            ) + 1;
            const monthNames = [
                "Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc",
            ];
            const dayNames = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"];
            const cells = ['<span class="statistics-calendar-axis-corner"></span>'];
            let priorMonth = "";
            for (let week = 0; week < weekCount; week += 1) {{
                const weekDate = new Date(calendarStart);
                weekDate.setUTCDate(weekDate.getUTCDate() + 7 * week);
                const monthKey = `${{weekDate.getUTCFullYear()}}-${{weekDate.getUTCMonth()}}`;
                const monthLabel = monthKey !== priorMonth
                    ? `${{monthNames[weekDate.getUTCMonth()]}} `
                        + `${{String(weekDate.getUTCFullYear()).slice(2)}}`
                    : "";
                priorMonth = monthKey;
                cells.push(
                    `<span class="statistics-calendar-column-label">`
                    + `${{monthLabel}}</span>`
                );
            }}
            dayNames.forEach((dayName, weekday) => {{
                cells.push(
                    `<span class="statistics-calendar-row-label">`
                    + `${{dayName}}</span>`
                );
                for (let week = 0; week < weekCount; week += 1) {{
                    const day = new Date(calendarStart);
                    day.setUTCDate(
                        day.getUTCDate() + 7 * week + weekday
                    );
                    const key = day.toISOString().slice(0, 10);
                    const item = byDay.get(key);
                    cells.push(
                        calendarCellHtml(
                            zoneInfo,
                            item ? item.row : null,
                            item ? item.entry : null,
                            scaleMaximum,
                            false
                        )
                    );
                }}
            }});
            return (
                `<div class="statistics-calendar-scroll">`
                + `<div class="statistics-calendar-grid daily"`
                + ` style="grid-template-columns:72px repeat(${{weekCount}}, minmax(14px,1fr))">`
                + `${{cells.join("")}}</div></div>`
            );
        }}

        function periodCalendarHtml(
            zoneInfo,
            rows,
            scaleMaximum,
            sampleKey
        ) {{
            const columnCount = sampleKey === "weekly" ? 53 : 12;
            const years = Array.from(
                new Set(rows.map((row) => row.year))
            ).sort((left, right) => left - right);
            const byPosition = new Map();
            rows.forEach((row) => {{
                byPosition.set(
                    `${{row.year}}-${{row.sampleNumber}}`,
                    {{row, entry: row.values[zoneInfo.key]}}
                );
            }});
            const monthNames = [
                "Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc",
            ];
            const cells = ['<span class="statistics-calendar-axis-corner">Année</span>'];
            for (let column = 1; column <= columnCount; column += 1) {{
                const label = sampleKey === "monthly"
                    ? monthNames[column - 1]
                    : (column === 1 || column % 4 === 0 || column === 53)
                    ? String(column)
                    : "";
                cells.push(
                    `<span class="statistics-calendar-column-label">`
                    + `${{label}}</span>`
                );
            }}
            years.forEach((year) => {{
                cells.push(
                    `<span class="statistics-calendar-row-label">`
                    + `${{year}}</span>`
                );
                for (let column = 1; column <= columnCount; column += 1) {{
                    const item = byPosition.get(`${{year}}-${{column}}`);
                    cells.push(
                        calendarCellHtml(
                            zoneInfo,
                            item ? item.row : null,
                            item ? item.entry : null,
                            scaleMaximum,
                            sampleKey === "monthly"
                        )
                    );
                }}
            }});
            return (
                `<div class="statistics-calendar-scroll">`
                + `<div class="statistics-calendar-grid ${{sampleKey}}"`
                + ` style="grid-template-columns:72px repeat(${{columnCount}}, minmax(18px,1fr))">`
                + `${{cells.join("")}}</div></div>`
            );
        }}

        function renderPriceCalendarDetail(zoneKey, periodKey) {{
            const row = activePriceCalendarRows.find(
                (item) => item.periodKey === periodKey
            );
            const zoneInfo = payload.zones.find(
                (item) => item.key === zoneKey
            );
            const entry = row && zoneInfo
                ? row.values[zoneInfo.key]
                : null;
            if (!row || !zoneInfo || !entry) {{
                priceCalendarDetail.innerHTML = "";
                return;
            }}
            const indicator = priceClosenessIndicator(entry, zoneInfo);
            const stormPrice = formatValue(entry.benchmark, 2) || "—";
            const stormDistance = (
                formatValue(entry.benchmarkDistance, 2) || "—"
            );
            priceCalendarDetail.innerHTML = `
                <div class="statistics-calendar-detail-title">
                    ${{zoneInfo.key}} — ${{row.timestamp}}
                </div>
                <div class="statistics-calendar-detail-grid">
                    <div><span>Prix observé</span><strong>${{formatValue(entry.observed, 2)}} EUR/MWh</strong></div>
                    <div><span>Notre modèle</span><strong>${{formatValue(entry.candidate, 2)}} EUR/MWh</strong></div>
                    <div><span>Écart modèle</span><strong>${{formatValue(entry.candidateDistance, 2)}} EUR/MWh</strong></div>
                    <div><span>Storm</span><strong>${{stormPrice}}${{stormPrice === "—" ? "" : " EUR/MWh"}}</strong></div>
                    <div><span>Écart Storm</span><strong>${{stormDistance}}${{stormDistance === "—" ? "" : " EUR/MWh"}}</strong></div>
                    <div><span>Proximité</span><strong><span class="statistics-closeness-pill ${{indicator.kind}}">${{indicator.label}}</span></strong></div>
                </div>
            `;
            priceCalendarContainer
                .querySelectorAll(".statistics-calendar-cell")
                .forEach((cell) => {{
                    const selected = (
                        cell.dataset.zone === zoneKey
                        && cell.dataset.period === periodKey
                    );
                    cell.classList.toggle("selected", selected);
                    cell.setAttribute("aria-pressed", String(selected));
                }});
        }}

        function renderPriceCalendar(rows, sampleKey) {{
            activePriceCalendarRows = rows;
            let defaultSelection = null;
            const calendars = payload.zones.map((zoneInfo) => {{
                const distances = rows
                    .map((row) => row.values[zoneInfo.key])
                    .filter(Boolean)
                    .map((entry) => entry.candidateDistance)
                    .filter(Number.isFinite);
                const scaleMaximum = calendarScaleMaximum(distances);
                const candidates = rows
                    .map((row) => ({{
                        row,
                        entry: row.values[zoneInfo.key],
                    }}))
                    .filter(
                        (item) => item.entry
                            && Number.isFinite(item.entry.candidateDistance)
                    );
                const worst = candidates.sort(
                    (left, right) =>
                        right.entry.candidateDistance
                        - left.entry.candidateDistance
                )[0];
                if (
                    worst
                    && (
                        !defaultSelection
                        || worst.entry.candidateDistance
                            > defaultSelection.entry.candidateDistance
                    )
                ) {{
                    defaultSelection = {{
                        zoneKey: zoneInfo.key,
                        periodKey: worst.row.periodKey,
                        entry: worst.entry,
                    }};
                }}
                const calendar = sampleKey === "daily"
                    ? dailyCalendarHtml(zoneInfo, rows, scaleMaximum)
                    : periodCalendarHtml(
                        zoneInfo,
                        rows,
                        scaleMaximum,
                        sampleKey
                    );
                const maximumText = worst
                    ? `Écart maximal : ${{worst.entry.candidateDistance.toFixed(2)}} EUR/MWh · ${{worst.row.timestamp}}`
                    : "Aucune période disponible";
                return `
                    <div class="statistics-calendar-zone">
                        <div class="statistics-calendar-zone-heading">
                            <strong>${{zoneInfo.key}}</strong>
                            <span>${{maximumText}}</span>
                        </div>
                        ${{calendar}}
                        <div class="statistics-calendar-scale-note">
                            Échelle de couleur plafonnée au 95e percentile :
                            ${{scaleMaximum.toFixed(2)}} EUR/MWh.
                            Les valeurs supérieures gardent la couleur maximale.
                        </div>
                    </div>
                `;
            }}).join("");
            priceCalendarContainer.innerHTML = calendars;
            if (defaultSelection) {{
                renderPriceCalendarDetail(
                    defaultSelection.zoneKey,
                    defaultSelection.periodKey
                );
            }} else {{
                priceCalendarDetail.innerHTML = "";
            }}
        }}

        function renderPriceComparisonTable() {{
            const sampleKey = priceSampleSelect.value;
            const rows = buildPeriodRows(sampleKey, "mean_price");
            const distances = [];
            rows.forEach((row) => {{
                payload.zones.forEach((zoneInfo) => {{
                    const entry = row.values[zoneInfo.key];
                    if (!entry) {{
                        return;
                    }}
                    [entry.candidateDistance, entry.benchmarkDistance]
                        .filter(Number.isFinite)
                        .forEach((value) => distances.push(value));
                }});
            }});
            const distanceScaleMaximum = calendarScaleMaximum(distances);

            let sampleHeader = "";
            if (sampleKey === "weekly") {{
                sampleHeader = "Week";
            }} else if (sampleKey === "daily") {{
                sampleHeader = "Day";
            }}
            const headerParts = ["<th>Year</th>", "<th>Month</th>"];
            if (sampleHeader) {{
                headerParts.push(`<th>${{sampleHeader}}</th>`);
            }}
            headerParts.push("<th>Timestamp</th>");
            payload.zones.forEach((zoneInfo) => {{
                headerParts.push(
                    `<th class="statistics-zone-header statistics-price-observed-header">${{zoneInfo.key}}<br>`
                    + `<small>Prix observé</small></th>`,
                    `<th class="statistics-zone-header statistics-price-model-header">${{zoneInfo.key}}<br>`
                    + `<small>Notre modèle</small></th>`,
                    `<th class="statistics-zone-header statistics-price-distance-header">${{zoneInfo.key}}<br>`
                    + `<small>|Modèle − observé|</small></th>`,
                    `<th class="statistics-zone-header statistics-price-storm-header">${{zoneInfo.key}}<br>`
                    + `<small>Storm</small></th>`,
                    `<th class="statistics-zone-header statistics-price-distance-header">${{zoneInfo.key}}<br>`
                    + `<small>|Storm − observé|</small></th>`,
                    `<th class="statistics-zone-header statistics-price-comparison-header">${{zoneInfo.key}}<br>`
                    + `<small>Forecast le plus proche</small></th>`
                );
            }});

            const bodyRows = rows.map((row) => {{
                const cells = [
                    `<td>${{row.year}}</td>`,
                    `<td>${{row.month}}</td>`,
                ];
                if (sampleHeader) {{
                    cells.push(`<td>${{row.sampleNumber}}</td>`);
                }}
                cells.push(
                    `<td class="statistics-timestamp-cell">`
                    + `${{row.timestamp}}</td>`
                );
                payload.zones.forEach((zoneInfo) => {{
                    const entry = row.values[zoneInfo.key];
                    const observed = entry ? entry.observed : null;
                    const candidate = entry ? entry.candidate : null;
                    const benchmark = entry ? entry.benchmark : null;
                    const candidateDistance = entry
                        ? entry.candidateDistance
                        : null;
                    const benchmarkDistance = entry
                        ? entry.benchmarkDistance
                        : null;
                    const indicator = priceClosenessIndicator(
                        entry,
                        zoneInfo
                    );
                    const candidateDistanceBackground = calendarErrorColor(
                        candidateDistance,
                        distanceScaleMaximum
                    );
                    const benchmarkDistanceBackground = calendarErrorColor(
                        benchmarkDistance,
                        distanceScaleMaximum
                    );
                    const observedUnavailable = Number.isFinite(observed)
                        ? ""
                        : " is-unavailable";
                    const candidateUnavailable = Number.isFinite(candidate)
                        ? ""
                        : " is-unavailable";
                    const benchmarkUnavailable = Number.isFinite(benchmark)
                        ? ""
                        : " is-unavailable";
                    const candidateBest = (
                        indicator.kind === "candidate" || indicator.kind === "tie"
                    ) ? " is-best" : "";
                    const benchmarkBest = (
                        indicator.kind === "benchmark" || indicator.kind === "tie"
                    ) ? " is-best" : "";
                    const comparisonTitle = entry
                        ? `Écart modèle=${{formatValue(candidateDistance, 2)}}; `
                            + `écart Storm=${{formatValue(benchmarkDistance, 2)}}`
                        : "Comparaison indisponible";
                    cells.push(
                        `<td class="statistics-value-cell statistics-observed-cell${{observedUnavailable}}">`
                        + `${{formatValue(observed, 2) || "—"}}</td>`,
                        `<td class="statistics-value-cell statistics-model-cell${{candidateUnavailable}}">`
                        + `${{formatValue(candidate, 2) || "—"}}</td>`,
                        `<td class="statistics-value-cell statistics-distance-cell${{candidateBest}}"`
                        + ` style="background:${{candidateDistanceBackground}}">`
                        + `${{formatValue(candidateDistance, 2) || "—"}}</td>`,
                        `<td class="statistics-value-cell statistics-storm-cell${{benchmarkUnavailable}}">`
                        + `${{formatValue(benchmark, 2) || "—"}}</td>`,
                        `<td class="statistics-value-cell statistics-distance-cell${{benchmarkBest}}"`
                        + ` style="background:${{benchmarkDistanceBackground}}">`
                        + `${{formatValue(benchmarkDistance, 2) || "—"}}</td>`,
                        `<td class="statistics-closeness-cell"`
                        + ` title="${{comparisonTitle}}">`
                        + `<span class="statistics-closeness-pill `
                        + `${{indicator.kind}}">${{indicator.label}}</span></td>`
                    );
                }});
                return `<tr>${{cells.join("")}}</tr>`;
            }});
            if (!bodyRows.length) {{
                priceContainer.innerHTML = (
                    '<p class="muted">Aucune donnée disponible.</p>'
                );
                priceSummaryContainer.innerHTML = "";
                return;
            }}
            priceContainer.innerHTML = `
                <table class="statistics-heatmap-table statistics-price-table">
                    <thead><tr>${{headerParts.join("")}}</tr></thead>
                    <tbody>${{bodyRows.join("")}}</tbody>
                </table>
            `;
            renderPriceClosenessSummary(rows);
            renderPriceCalendar(rows, sampleKey);
        }}

        priceCalendarContainer.addEventListener("click", (event) => {{
            const cell = event.target.closest(
                ".statistics-calendar-cell[data-period]"
            );
            if (!cell) {{
                return;
            }}
            renderPriceCalendarDetail(
                cell.dataset.zone,
                cell.dataset.period
            );
        }});

        metricSelect.addEventListener(
            "change",
            renderStatisticsTable
        );
        sampleSelect.addEventListener(
            "change",
            renderStatisticsTable
        );
        priceSampleSelect.addEventListener(
            "change",
            renderPriceComparisonTable
        );
        window.addEventListener(
            "chronos2-theme-change",
            () => {{
                renderStatisticsTable();
                renderPriceComparisonTable();
            }}
        );

        renderStatisticsTable();
        renderPriceComparisonTable();
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

        kalman_diagnostics_html = build_kalman_diagnostics_html(result)
        variable_attribution_html = build_variable_attribution_html(result)
        forecast_components_html = build_forecast_components_html(result)
        hourly_source = getattr(result, "hourly_comparison_source", None)
        hourly_comparison_html = build_hourly_comparison_html(
            result,
            source=(_statistics_source(result) if hourly_source is None
                    else _latest_statistics_window(hourly_source)),
            benchmark_contract=getattr(result, "hourly_comparison_contract", None)
            or _statistics_benchmark_contract(result),
        )

        storm_comparison_html = ""
        if getattr(result, "statistics_benchmark", None) is not None:
            benchmark_contract = _statistics_benchmark_contract(result)
            benchmark_label_html = html.escape(
                str(benchmark_contract["report_label"])
            )
            storm_div = plotly_div(
                figure_storm_comparison(result),
                include_js,
            )
            include_js = False
            scope_note = str(
                getattr(result, "statistics_scope_note", "")
            ).strip()
            scope_html = (
                "<br><strong>Périmètre :</strong> "
                + html.escape(scope_note)
                if scope_note
                else ""
            )
            storm_comparison_html = f'''
            <div
                class="storm-comparison"
                data-report-section="storm-comparison"
            >
                <h3>Comparaison graphique au benchmark actif — {benchmark_label_html}</h3>
                <p class="muted">
                    Les courbes utilisent exclusivement l'historique
                    d'évaluation apparié de Statistics.
                    {benchmark_label_html} reste un comparateur d'évaluation :
                    il n'entre ni dans les variables, ni dans le modèle, ni
                    dans la prévision live.
                    {scope_html}
                </p>
                {storm_div}
            </div>
            '''

        sections.append(
            f'''<section id="{result.zone.lower()}">
            <div class="zone-title"><div><h2>{result.zone}</h2>
            <p class="muted">Prévision opérationnelle du {html.escape(str(delivery_start))} au {html.escape(str(delivery_end))}</p></div>
            <span class="badge">{len(result.zone_data.covariates.columns)} covariables actives</span></div>
            {average_price_cards(result)}
            <h3>Performance du modèle</h3>
            {metric_cards(result.metrics_native)}
            <h3>Comparaison au modèle prix seul</h3>
            {html_table(comparison)}
            <h3>Prévision Day-Ahead réelle</h3>
            {figures[0]}
            <h3>Backtest et probabilités</h3>
            {figures[1]}
            {storm_comparison_html}
            {hourly_comparison_html}
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
            {kalman_diagnostics_html}
            {forecast_components_html}
            {variable_attribution_html}
            <h3>Variables d’entrée</h3>
            {figures[6]}
            {html_table(result.zone_data.input_manifest)}
            </section>'''
        )

    sections.append(
        build_statistics_table_html(results)
    )

    css = '''
    :root {
        color-scheme:light;
        --bg:#f4f6f8;
        --card:#ffffff;
        --surface-soft:#f8fafc;
        --surface-card:#fbfcfd;
        --text:#18212b;
        --muted:#607080;
        --border:#dfe5ea;
        --border-strong:#cfd5da;
        --header:#111827;
        --header-text:#ffffff;
        --accent-soft:#eaf0ff;
        --accent-text:#1746b8;
        --statistics-title:#253448;
        --statistics-title-text:#ffffff;
        --statistics-value:#174f86;
        --heatmap-text:#111820;
        --observed-accent:#7c3aed;
        --observed-cell-bg:#ede9fe;
        --observed-cell-text:#5b21b6;
        --model-cell-bg:#dbeafe;
        --model-cell-text:#1e40af;
        --storm-cell-bg:#fef3c7;
        --storm-cell-text:#92400e;
        --distance-best-border:#15803d;
        --candidate-pill-bg:#dbeafe;
        --candidate-pill-text:#1d4ed8;
        --benchmark-pill-bg:#fef3c7;
        --benchmark-pill-text:#92400e;
        --tie-pill-bg:#e5e7eb;
        --tie-pill-text:#374151;
        --unavailable-pill-bg:#f1f5f9;
        --unavailable-pill-text:#64748b;
        --calendar-low:#dcfce7;
        --calendar-mid:#fef0b4;
        --calendar-high:#fecaca;
        --shadow:0 6px 22px rgba(20,35,55,.05);
    }
    html[data-theme="dark"] {
        color-scheme:dark;
        --bg:#0b1220;
        --card:#111b2e;
        --surface-soft:#172338;
        --surface-card:#152136;
        --text:#e7edf6;
        --muted:#aab7ca;
        --border:#2b3b52;
        --border-strong:#41536c;
        --header:#080e1a;
        --header-text:#f8fafc;
        --accent-soft:#18345c;
        --accent-text:#93c5fd;
        --statistics-title:#263b59;
        --statistics-title-text:#f8fafc;
        --statistics-value:#93c5fd;
        --heatmap-text:#f4f7fb;
        --observed-accent:#c4b5fd;
        --observed-cell-bg:#3b2a59;
        --observed-cell-text:#ddd6fe;
        --model-cell-bg:#183b66;
        --model-cell-text:#bfdbfe;
        --storm-cell-bg:#553c16;
        --storm-cell-text:#fde68a;
        --distance-best-border:#4ade80;
        --candidate-pill-bg:#183b66;
        --candidate-pill-text:#bfdbfe;
        --benchmark-pill-bg:#553c16;
        --benchmark-pill-text:#fde68a;
        --tie-pill-bg:#374151;
        --tie-pill-text:#e5e7eb;
        --unavailable-pill-bg:#263449;
        --unavailable-pill-text:#aab7ca;
        --calendar-low:#14532d;
        --calendar-mid:#945c19;
        --calendar-high:#7e303d;
        --shadow:0 8px 28px rgba(0,0,0,.28);
    }
    * { box-sizing:border-box; }
    html { scroll-behavior:smooth; }
    body { margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; background:var(--bg); color:var(--text); transition:background-color 160ms ease,color 160ms ease; }
    header { background:var(--header); color:var(--header-text); padding:28px 5vw; }
    .header-top { display:flex; align-items:flex-start; justify-content:space-between; gap:24px; }
    header h1 { margin:0 0 8px; font-size:30px; }
    .theme-toggle { flex:0 0 auto; min-width:128px; border:1px solid rgba(255,255,255,.38); border-radius:999px; padding:9px 14px; background:rgba(255,255,255,.08); color:var(--header-text); font:600 14px/1.2 inherit; cursor:pointer; }
    .theme-toggle:hover { background:rgba(255,255,255,.16); border-color:rgba(255,255,255,.7); }
    .theme-toggle:focus-visible { outline:3px solid #60a5fa; outline-offset:3px; }
    nav { margin-top:18px; display:flex; gap:10px; flex-wrap:wrap; }
    nav a { color:var(--header-text); text-decoration:none; border:1px solid rgba(255,255,255,.35); padding:7px 12px; border-radius:999px; }
    main { max-width:1500px; margin:0 auto; padding:28px 24px 60px; }
    section { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:24px; margin-bottom:26px; box-shadow:var(--shadow); transition:background-color 160ms ease,border-color 160ms ease; }
    h2 { font-size:26px; margin:0 0 12px; }
    h3 { margin-top:28px; border-bottom:1px solid var(--border); padding-bottom:8px; }
    .muted { color:var(--muted); }
    .zone-title { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; }
    .badge { background:var(--accent-soft); color:var(--accent-text); padding:7px 12px; border-radius:999px; font-weight:600; }
    .average-price-block { margin:20px 0 24px; padding:18px; border:1px solid var(--border); border-radius:14px; background:var(--surface-soft); }
    .average-price-block h3 { margin:0 0 14px; border:0; padding:0; }
    .average-price-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }
    .average-price-card { min-height:142px; border:1px solid var(--border); border-top:4px solid #3b82f6; border-radius:11px; padding:14px; background:var(--card); }
    .average-price-card:nth-child(3) { border-top-color:#f59e0b; }
    .average-price-card:nth-child(4) { border-top-color:#8b5cf6; }
    .average-price-label { color:var(--muted); font-size:13px; font-weight:650; }
    .average-price-value { margin-top:7px; font-size:29px; line-height:1; font-weight:750; font-variant-numeric:tabular-nums; }
    .average-price-unit { margin-top:4px; color:var(--muted); font-size:12px; }
    .average-price-detail { margin-top:11px; color:var(--muted); font-size:12px; line-height:1.35; }
    .average-price-method { margin:13px 2px 0; color:var(--muted); font-size:12px; line-height:1.5; }
    .metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:18px 0; }
    .metric-card { border:1px solid var(--border); border-radius:12px; padding:15px; background:var(--surface-card); }
    .metric-label { color:var(--muted); font-size:13px; }
    .metric-value { font-size:25px; font-weight:700; margin-top:5px; }
    .metric-unit { color:var(--muted); font-size:12px; }
    .table-wrap { overflow-x:auto; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { padding:9px 10px; border-bottom:1px solid var(--border); text-align:left; white-space:nowrap; }
    th { background:var(--surface-soft); position:sticky; top:0; }
    footer { color:var(--muted); text-align:center; padding:20px; }

    .variable-attribution { margin:28px 0; padding:20px; border:1px solid var(--border); border-radius:14px; background:var(--surface-soft); }
    .attribution-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; }
    .attribution-heading h3 { margin:4px 0 9px; border:0; padding:0; }
    .attribution-heading p { max-width:920px; margin:0; color:var(--muted); line-height:1.55; }
    .attribution-kicker { color:var(--accent-text); font-size:12px; font-weight:800; letter-spacing:.08em; }
    .attribution-method-badge { flex:0 0 auto; max-width:290px; padding:8px 12px; border:1px solid var(--border); border-radius:999px; background:var(--accent-soft); color:var(--accent-text); font-size:12px; font-weight:700; text-align:center; }
    .attribution-architecture { display:grid; grid-template-columns:minmax(220px,1fr) repeat(2,minmax(160px,220px)); gap:10px; align-items:stretch; margin:18px 0; }
    .attribution-architecture-title { display:flex; align-items:center; color:var(--muted); font-size:13px; font-weight:700; }
    .attribution-architecture-card { display:flex; align-items:center; justify-content:space-between; gap:14px; padding:12px 14px; border:1px solid var(--border); border-radius:11px; background:var(--card); }
    .attribution-architecture-card span { color:var(--muted); font-size:13px; }
    .attribution-architecture-card strong { color:var(--statistics-value); font-size:21px; font-variant-numeric:tabular-nums; }
    .attribution-chart-grid { display:grid; grid-template-columns:minmax(0,.8fr) minmax(0,1.2fr); gap:14px; }
    .attribution-chart-card { min-width:0; overflow:hidden; border:1px solid var(--border); border-radius:12px; background:var(--card); }
    .variable-attribution h4 { margin:20px 0 9px; }
    .attribution-audit-note { margin:14px 0 0; padding:12px 14px; border-left:4px solid #2563eb; border-radius:6px; background:var(--card); color:var(--muted); font-size:12px; line-height:1.55; }

    .kalman-diagnostics { margin:28px 0; padding:20px; border:1px solid var(--border); border-radius:14px; background:var(--surface-soft); }
    .kalman-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; }
    .kalman-heading h3 { margin:4px 0 9px; border:0; padding:0; }
    .kalman-heading p { max-width:920px; margin:0; color:var(--muted); line-height:1.55; }
    .kalman-kicker { color:var(--accent-text); font-size:12px; font-weight:800; letter-spacing:.08em; }
    .kalman-method-badge { flex:0 0 auto; max-width:300px; padding:8px 12px; border:1px solid var(--border); border-radius:999px; background:var(--accent-soft); color:var(--accent-text); font-size:12px; font-weight:700; text-align:center; }
    .kalman-card-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:18px 0; }
    .kalman-card { display:flex; flex-direction:column; gap:4px; min-height:112px; padding:14px; border:1px solid var(--border); border-top:4px solid #2563eb; border-radius:11px; background:var(--card); }
    .kalman-card span { color:var(--muted); font-size:12px; font-weight:700; }
    .kalman-card strong { margin-top:3px; color:var(--text); font-size:25px; font-variant-numeric:tabular-nums; }
    .kalman-card small { color:var(--muted); }
    .kalman-card.positive { border-top-color:#16a34a; }
    .kalman-card.negative { border-top-color:#dc2626; }
    .kalman-diagnostics h4 { margin:20px 0 9px; }
    .kalman-definition { margin-top:-3px; font-size:12px; }
    .kalman-chart-card { min-width:0; margin:16px 0; overflow:hidden; border:1px solid var(--border); border-radius:12px; background:var(--card); }
    .kalman-audit-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:10px; margin:18px 0 0; }
    .kalman-audit-grid > div { display:flex; flex-direction:column; gap:5px; padding:12px 14px; border:1px solid var(--border); border-radius:10px; background:var(--card); }
    .kalman-audit-grid strong { font-size:12px; }
    .kalman-audit-grid span { color:var(--muted); font-size:12px; line-height:1.45; }
    .kalman-causal-note,.kalman-warning { margin:14px 0 0; padding:12px 14px; border-left:4px solid #2563eb; border-radius:6px; background:var(--card); color:var(--muted); font-size:12px; line-height:1.55; }
    .kalman-warning { border-left-color:#f59e0b; }

    .statistics-section {
        padding:0;
        overflow:hidden;
    }
    .statistics-title-pill {
        display:inline-block;
        background:var(--statistics-title);
        color:var(--statistics-title-text);
        font-size:16px;
        font-weight:700;
        padding:7px 12px;
        border-radius:0 0 12px 0;
    }
    .statistics-freshness-grid {
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
        gap:10px;
        margin:18px 12px 4px;
    }
    .statistics-freshness-card {
        display:flex;
        flex-direction:column;
        gap:6px;
        padding:13px 15px;
        border:1px solid var(--border);
        border-left:4px solid #0ea5e9;
        border-radius:10px;
        background:var(--surface-soft);
        font-size:12px;
        line-height:1.45;
    }
    .statistics-freshness-card strong { font-size:13px; }
    .statistics-freshness-card span { color:var(--muted); }
    .statistics-overall-summary {
        margin:18px 12px 4px;
        padding:16px;
        border:1px solid var(--border);
        border-radius:12px;
        background:var(--surface-soft);
    }
    .statistics-overall-summary h3 {
        margin:0 0 12px;
    }
    .statistics-definition {
        margin:10px 0 0;
        color:var(--muted);
        font-size:12px;
        line-height:1.5;
    }
    .statistics-filtering-title {
        padding:22px 10px 4px;
        color:var(--muted);
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
        background:var(--card);
        color:var(--text);
        font-family:inherit;
        font-size:16px;
        cursor:pointer;
    }
    .statistics-win-rate-summary {
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
        gap:10px;
        padding:14px 10px;
        border-bottom:1px solid var(--border);
        background:var(--surface-soft);
    }
    .statistics-win-rate-summary > .statistics-definition {
        grid-column:1 / -1;
    }
    .statistics-win-rate-card {
        padding:13px 15px;
        border:1px solid var(--border-strong);
        border-radius:10px;
        background:var(--card);
    }
    .statistics-win-rate-heading {
        color:var(--muted);
        font-size:12px;
        font-weight:700;
    }
    .statistics-win-rate-value {
        margin-top:3px;
        color:var(--statistics-value);
        font-size:26px;
        font-weight:750;
    }
    .statistics-win-rate-detail {
        margin-top:4px;
        color:var(--muted);
        font-size:12px;
    }
    .statistics-price-comparison {
        margin:28px 12px 18px;
        border:1px solid var(--border);
        border-radius:14px;
        overflow:hidden;
        background:var(--surface-soft);
    }
    .statistics-price-header {
        display:flex;
        align-items:flex-end;
        justify-content:space-between;
        gap:22px;
        padding:18px;
        border-bottom:1px solid var(--border);
    }
    .statistics-price-header h3 {
        margin:4px 0 6px;
        border:0;
        padding:0;
    }
    .statistics-price-header p {
        max-width:760px;
        margin:0;
        color:var(--muted);
        font-size:13px;
        line-height:1.5;
    }
    .statistics-price-color-legend {
        display:flex;
        flex-wrap:wrap;
        gap:7px;
        margin-top:11px;
    }
    .statistics-price-color-legend span {
        display:inline-flex;
        align-items:center;
        min-height:26px;
        padding:4px 9px;
        border:1px solid var(--border-strong);
        border-radius:999px;
        font-size:11px;
        font-weight:750;
    }
    .statistics-price-color-legend .observed {
        border-color:var(--observed-accent);
        background:var(--observed-cell-bg);
        color:var(--observed-cell-text);
    }
    .statistics-price-color-legend .candidate {
        border-color:var(--candidate-pill-text);
        background:var(--model-cell-bg);
        color:var(--model-cell-text);
    }
    .statistics-price-color-legend .benchmark {
        border-color:var(--benchmark-pill-text);
        background:var(--storm-cell-bg);
        color:var(--storm-cell-text);
    }
    .statistics-price-color-legend .distance {
        background:var(--card);
        color:var(--text);
    }
    .statistics-price-color-legend .distance::before {
        content:"";
        width:48px;
        height:10px;
        margin-right:7px;
        border:1px solid var(--border-strong);
        border-radius:999px;
        background:linear-gradient(
            90deg,
            var(--calendar-low),
            var(--calendar-mid),
            var(--calendar-high)
        );
    }
    .statistics-price-refresh-status {
        max-width:760px;
        margin-top:10px;
        padding:9px 11px;
        border:1px solid var(--border);
        border-radius:9px;
        background:var(--card);
        color:var(--muted);
        font-size:12px;
        line-height:1.45;
    }
    .statistics-price-refresh-item { display:block; margin-top:3px; }
    .statistics-price-kicker {
        color:var(--statistics-value);
        font-size:12px;
        font-weight:800;
        letter-spacing:.08em;
    }
    .statistics-price-closeness-summary {
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
        gap:10px;
        padding:14px;
        border-bottom:1px solid var(--border);
    }
    .statistics-price-summary-card {
        padding:13px 15px;
        border:1px solid var(--border-strong);
        border-radius:10px;
        background:var(--card);
    }
    .statistics-price-summary-value {
        margin-top:5px;
        color:var(--text);
        font-size:17px;
        font-weight:750;
    }
    .statistics-price-calendar-panel {
        margin:14px;
        border:1px solid var(--border-strong);
        border-radius:12px;
        overflow:hidden;
        background:var(--card);
    }
    .statistics-price-calendar-header {
        display:flex;
        align-items:flex-end;
        justify-content:space-between;
        gap:20px;
        padding:15px 16px;
        border-bottom:1px solid var(--border);
    }
    .statistics-price-calendar-header h4 {
        margin:0 0 5px;
        font-size:17px;
    }
    .statistics-price-calendar-header p {
        max-width:790px;
        margin:0;
        color:var(--muted);
        font-size:12px;
        line-height:1.5;
    }
    .statistics-calendar-legend {
        display:flex;
        align-items:center;
        gap:8px;
        flex:0 0 auto;
        color:var(--muted);
        font-size:11px;
        font-weight:650;
        white-space:nowrap;
    }
    .statistics-calendar-legend-bar {
        width:130px;
        height:12px;
        border:1px solid var(--border-strong);
        border-radius:999px;
        background:linear-gradient(
            90deg,
            var(--calendar-low),
            var(--calendar-mid),
            var(--calendar-high)
        );
    }
    .statistics-calendar-zone {
        padding:15px 16px;
        border-bottom:1px solid var(--border);
    }
    .statistics-calendar-zone:last-child {
        border-bottom:0;
    }
    .statistics-calendar-zone-heading {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:16px;
        margin-bottom:10px;
    }
    .statistics-calendar-zone-heading strong {
        font-size:15px;
    }
    .statistics-calendar-zone-heading span {
        color:var(--muted);
        font-size:12px;
        font-weight:650;
    }
    .statistics-calendar-scroll {
        overflow-x:auto;
        padding:3px 2px 8px;
    }
    .statistics-calendar-grid {
        display:grid;
        gap:3px;
        align-items:center;
        min-width:720px;
    }
    .statistics-calendar-grid.daily,
    .statistics-calendar-grid.weekly {
        min-width:1040px;
    }
    .statistics-calendar-axis-corner,
    .statistics-calendar-column-label,
    .statistics-calendar-row-label {
        color:var(--muted);
        font-size:10px;
        font-weight:700;
        line-height:1;
    }
    .statistics-calendar-column-label {
        min-height:18px;
        overflow:visible;
        text-align:left;
        white-space:nowrap;
    }
    .statistics-calendar-row-label {
        padding-right:7px;
        text-align:right;
    }
    .statistics-calendar-cell,
    .statistics-calendar-empty-cell {
        display:block;
        width:100%;
        min-width:12px;
        height:18px;
        border:1px solid var(--border-strong);
        border-radius:3px;
    }
    .statistics-calendar-grid.monthly .statistics-calendar-cell,
    .statistics-calendar-grid.monthly .statistics-calendar-empty-cell {
        height:34px;
    }
    .statistics-calendar-cell {
        padding:0;
        color:var(--heatmap-text);
        font:700 10px/1 Inter,Segoe UI,Arial,sans-serif;
        cursor:pointer;
        transition:transform 110ms ease,filter 110ms ease,outline-color 110ms ease;
    }
    .statistics-calendar-cell:hover,
    .statistics-calendar-cell:focus-visible {
        z-index:2;
        filter:brightness(1.12);
        outline:2px solid var(--text);
        outline-offset:1px;
        transform:scale(1.18);
    }
    .statistics-calendar-cell.selected {
        outline:3px solid var(--observed-accent);
        outline-offset:1px;
    }
    .statistics-calendar-empty-cell {
        border-color:var(--border);
        background:var(--surface-soft);
        opacity:.45;
    }
    .statistics-calendar-scale-note {
        margin-top:5px;
        color:var(--muted);
        font-size:10px;
        line-height:1.4;
    }
    .statistics-price-calendar-detail {
        padding:14px 16px 16px;
        border-top:1px solid var(--border);
        background:var(--surface-soft);
    }
    .statistics-calendar-detail-title {
        margin-bottom:10px;
        font-size:14px;
        font-weight:800;
    }
    .statistics-calendar-detail-grid {
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(155px,1fr));
        gap:8px;
    }
    .statistics-calendar-detail-grid > div {
        min-height:64px;
        padding:10px;
        border:1px solid var(--border);
        border-radius:9px;
        background:var(--card);
    }
    .statistics-calendar-detail-grid span:not(.statistics-closeness-pill) {
        display:block;
        margin-bottom:5px;
        color:var(--muted);
        font-size:10px;
        font-weight:700;
    }
    .statistics-calendar-detail-grid strong {
        font-size:13px;
        font-variant-numeric:tabular-nums;
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
        border-right:1px solid var(--border-strong);
        border-bottom:1px solid var(--border-strong);
        text-align:center;
        white-space:nowrap;
    }
    .statistics-heatmap-table th {
        position:sticky;
        top:0;
        z-index:2;
        background:var(--surface-soft);
        color:var(--text);
        font-weight:700;
    }
    .statistics-heatmap-table td:first-child,
    .statistics-heatmap-table th:first-child {
        border-left:1px solid var(--border-strong);
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
        color:var(--heatmap-text);
        font-weight:500;
        font-variant-numeric:tabular-nums;
        transition:filter 120ms ease,transform 120ms ease;
    }
    .statistics-value-cell:hover {
        filter:brightness(1.08);
        transform:scale(1.02);
    }
    .statistics-observed-cell {
        box-shadow:inset 4px 0 0 var(--observed-accent);
        background:var(--observed-cell-bg);
        color:var(--observed-cell-text);
        font-weight:750;
    }
    .statistics-model-cell {
        box-shadow:inset 4px 0 0 var(--candidate-pill-text);
        background:var(--model-cell-bg);
        color:var(--model-cell-text);
        font-weight:750;
    }
    .statistics-storm-cell {
        box-shadow:inset 4px 0 0 var(--benchmark-pill-text);
        background:var(--storm-cell-bg);
        color:var(--storm-cell-text);
        font-weight:750;
    }
    .statistics-value-cell.is-unavailable {
        box-shadow:none;
        background:var(--surface-soft);
        color:var(--muted);
        font-weight:500;
    }
    .statistics-distance-cell {
        font-weight:750;
    }
    .statistics-distance-cell.is-best {
        box-shadow:inset 0 0 0 3px var(--distance-best-border);
    }
    .statistics-price-table .statistics-price-observed-header {
        border-top:4px solid var(--observed-accent);
        background:var(--observed-cell-bg);
        color:var(--observed-cell-text);
    }
    .statistics-price-table .statistics-price-model-header {
        border-top:4px solid var(--candidate-pill-text);
        background:var(--model-cell-bg);
        color:var(--model-cell-text);
    }
    .statistics-price-table .statistics-price-storm-header {
        border-top:4px solid var(--benchmark-pill-text);
        background:var(--storm-cell-bg);
        color:var(--storm-cell-text);
    }
    .statistics-price-table .statistics-price-distance-header {
        border-top:4px solid var(--calendar-mid);
        background:var(--surface-soft);
    }
    .statistics-price-table .statistics-price-comparison-header {
        border-top:4px solid var(--distance-best-border);
        background:var(--surface-soft);
    }
    .statistics-price-table {
        min-width:1260px;
    }
    .statistics-closeness-cell {
        min-width:265px;
        padding:8px 12px !important;
    }
    .statistics-closeness-pill {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        min-height:28px;
        padding:5px 10px;
        border:1px solid transparent;
        border-radius:999px;
        font-family:Inter,Segoe UI,Arial,sans-serif;
        font-size:12px;
        font-weight:750;
    }
    .statistics-closeness-pill::before {
        content:"";
        width:7px;
        height:7px;
        margin-right:7px;
        border-radius:50%;
        background:currentColor;
    }
    .statistics-closeness-pill.candidate {
        background:var(--candidate-pill-bg);
        color:var(--candidate-pill-text);
    }
    .statistics-closeness-pill.benchmark {
        background:var(--benchmark-pill-bg);
        color:var(--benchmark-pill-text);
    }
    .statistics-closeness-pill.tie {
        background:var(--tie-pill-bg);
        color:var(--tie-pill-text);
    }
    .statistics-closeness-pill.unavailable {
        background:var(--unavailable-pill-bg);
        color:var(--unavailable-pill-text);
    }

    .js-plotly-plot .plotly .modebar { background:transparent !important; }
    .js-plotly-plot .plotly .modebar-btn path { fill:var(--muted) !important; }
    .js-plotly-plot .plotly .modebar-btn:hover path { fill:var(--text) !important; }

    @media (max-width:700px) {
        main { padding:14px; }
        section { padding:14px; }
        .statistics-section { padding:0; }
        .header-top { display:block; }
        .theme-toggle { margin-top:14px; }
        .attribution-heading { flex-direction:column; }
        .attribution-method-badge { max-width:none; }
        .attribution-architecture { grid-template-columns:1fr; }
        .attribution-chart-grid { grid-template-columns:1fr; }
        .kalman-heading { flex-direction:column; }
        .kalman-method-badge { max-width:none; }
        .statistics-price-header { align-items:stretch; flex-direction:column; }
        .statistics-price-calendar-header { align-items:stretch; flex-direction:column; }
        .statistics-calendar-legend { justify-content:space-between; }
        .statistics-calendar-zone-heading { align-items:flex-start; flex-direction:column; }
    }
    @media (prefers-reduced-motion:reduce) {
        html { scroll-behavior:auto; }
        body,section { transition:none; }
    }
    @media print {
        .theme-toggle { display:none; }
        section { box-shadow:none; break-inside:avoid; }
    }
    '''

    document = f'''<!doctype html>
    <html lang="fr" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{html.escape(title)}</title>{_REPORT_THEME_BOOTSTRAP}<style>{css}</style></head>
    <body><header><div class="header-top"><div><h1>{html.escape(title)}</h1><p>Généré le {generated} · Script {SCRIPT_VERSION}</p></div><button id="theme-toggle" class="theme-toggle" type="button" aria-pressed="false" aria-label="Activer le mode nuit">☾ Mode nuit</button></div><nav>{nav_links}</nav></header>
    <main>{''.join(sections)}</main><footer>Chronos-2 · rapport autonome Plotly</footer>{_REPORT_THEME_CONTROLLER}</body></html>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
