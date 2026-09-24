"""Standalone interactive report for auxiliary-model experiments."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _table(frame: pd.DataFrame, *, max_rows: int = 500) -> str:
    if frame.empty:
        return "<p class='muted'>Aucune donnee.</p>"
    display = frame.head(max_rows).copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.5g}")
    return display.to_html(index=False, classes="data-table", border=0, escape=True)


def _plot_html(fig: Any) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def write_report(
    output_path: str | Path,
    *,
    experiment_id: str,
    manifest: Mapping[str, Any],
    leaderboard: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
    embed_plotly: bool,
    feature_importance: pd.DataFrame | None = None,
    kalman_coefficients: pd.DataFrame | None = None,
) -> Path:
    """Write one self-contained (or CDN-backed) light/dark HTML report."""

    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    test = metrics.loc[(metrics["phase"] == "test") & (metrics["role"] == "candidate")]
    metric_fig = go.Figure()
    if not test.empty:
        metric_fig.add_bar(
            x=test["model"],
            y=test["mae"],
            name="MAE test",
            marker_color="#00a7a0",
            text=test["mae"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
        )
        baselines = metrics.loc[(metrics["phase"] == "test") & (metrics["role"] == "baseline")]
        if not baselines.empty:
            metric_fig.add_bar(
                x=baselines["model"],
                y=baselines["mae"],
                name="MAE baseline",
                marker_color="#8795a1",
                text=baselines["mae"].map(lambda value: f"{value:.3f}"),
                textposition="outside",
            )
    metric_fig.update_layout(
        title="Performance sur le test scelle",
        barmode="group",
        yaxis_title="MAE (EUR/MWh)",
        template="plotly_white",
    )

    leaderboard_fig = go.Figure()
    if not leaderboard.empty:
        for model, block in leaderboard.groupby("model", sort=True):
            leaderboard_fig.add_scatter(
                x=block["configuration_id"],
                y=block["objective_value"],
                mode="lines+markers",
                name=str(model),
                marker={"size": 8},
            )
    leaderboard_fig.update_layout(
        title="Comparaison des configurations sur validation",
        xaxis_title="Configuration",
        yaxis_title=str(manifest.get("objective", "score")),
        template="plotly_white",
    )

    daily_fig = go.Figure()
    if not daily.empty:
        for model, block in daily.loc[daily["phase"] == "test"].groupby("model", sort=True):
            daily_fig.add_scatter(
                x=block["local_day"],
                y=block["mae"],
                mode="lines",
                name=str(model),
            )
    daily_fig.update_layout(
        title="MAE journaliere du meilleur modele",
        xaxis_title="Jour local",
        yaxis_title="MAE (EUR/MWh)",
        template="plotly_white",
        hovermode="x unified",
    )

    cumulative_fig = go.Figure()
    if not predictions.empty:
        test_predictions = predictions.loc[predictions["phase"] == "test"].copy()
        for model, block in test_predictions.groupby("model", sort=True):
            block = block.sort_values("delivery_start_utc")
            delta = (block["q50"] - block["actual"]).abs() - (
                block["baseline_q50"] - block["actual"]
            ).abs()
            cumulative_fig.add_scatter(
                x=block["delivery_start_utc"],
                y=delta.cumsum(),
                mode="lines",
                name=str(model),
            )
    cumulative_fig.add_hline(y=0.0, line_dash="dash", line_color="#8795a1")
    cumulative_fig.update_layout(
        title="Delta d'erreur absolue cumulee vs baseline (negatif = mieux)",
        xaxis_title="Livraison UTC",
        yaxis_title="Somme cumulee (EUR/MWh)",
        template="plotly_white",
        hovermode="x unified",
    )

    importance_fig = go.Figure()
    if feature_importance is not None and not feature_importance.empty:
        importance = feature_importance.head(30).sort_values(
            "importance_mae_increase", ascending=True
        )
        importance_fig.add_bar(
            x=importance["importance_mae_increase"],
            y=importance["feature"],
            orientation="h",
            marker_color="#6957d5",
        )
    importance_fig.update_layout(
        title="Importance par permutation des variables du correcteur residuel",
        xaxis_title="Hausse de MAE apres permutation (EUR/MWh)",
        yaxis_title="Variable",
        template="plotly_white",
    )

    kalman_coefficient_fig = go.Figure()
    if kalman_coefficients is not None and not kalman_coefficients.empty:
        coefficients = kalman_coefficients.nlargest(
            30, "absolute_standardized_coefficient"
        ).copy()
        coefficients["label"] = (
            coefficients["filter_kind"].astype(str)
            + " / "
            + coefficients["feature"].astype(str)
        )
        coefficients = coefficients.sort_values(
            "absolute_standardized_coefficient", ascending=True
        )
        kalman_coefficient_fig.add_bar(
            x=coefficients["absolute_standardized_coefficient"],
            y=coefficients["label"],
            orientation="h",
            marker_color="#d87520",
            customdata=coefficients[["standardized_coefficient", "group"]],
            hovertemplate=(
                "%{y}<br>|coefficient standardise|=%{x:.4f}"
                "<br>coefficient signe=%{customdata[0]:.4f}"
                "<br>groupe=%{customdata[1]}<extra></extra>"
            ),
        )
    kalman_coefficient_fig.update_layout(
        title="Etats finaux des variables exogenes du Kalman",
        xaxis_title="Valeur absolue du coefficient standardise",
        yaxis_title="Filtre / variable",
        template="plotly_white",
    )
    kalman_coefficient_html = (
        "<section class='panel wide'>"
        f"{_plot_html(kalman_coefficient_fig)}"
        "<p class='muted'>Ces coefficients décrivent une association "
        "conditionnelle dans le filtre et ne démontrent aucune causalité.</p>"
        f"{_table(kalman_coefficients)}"
        "</section>"
        if kalman_coefficients is not None and not kalman_coefficients.empty
        else ""
    )

    plotly_script = (
        f"<script>{get_plotlyjs()}</script>"
        if embed_plotly
        else '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    )
    selected = leaderboard.loc[leaderboard.get("selected", False) == True]  # noqa: E712
    selected_html = _table(selected) if not selected.empty else "<p class='muted'>Aucune selection.</p>"
    html = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Auxiliary Lab — {escape(experiment_id)}</title>
{plotly_script}
<style>
:root{{--bg:#f4f7fa;--panel:#fff;--text:#17202a;--muted:#65717e;--line:#d9e0e7;--accent:#007f7a;--shadow:0 8px 28px #17202a16}}
html[data-theme="dark"]{{--bg:#111820;--panel:#1b2530;--text:#edf3f8;--muted:#a9b7c4;--line:#344454;--accent:#37d2c8;--shadow:0 8px 28px #0007}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Segoe UI,Arial,sans-serif}}
.wrap{{max-width:1500px;margin:auto;padding:22px}} header{{display:flex;justify-content:space-between;gap:16px;align-items:center}}
h1{{margin:0;font-size:28px}} h2{{margin:0 0 14px;font-size:19px}} .muted{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;margin-top:18px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow);overflow:auto}}
.wide{{grid-column:1/-1}} button{{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:9px;padding:9px 13px;cursor:pointer}}
.data-table{{border-collapse:collapse;width:100%;white-space:nowrap}} .data-table th,.data-table td{{border-bottom:1px solid var(--line);padding:8px;text-align:left}}
.data-table th{{position:sticky;top:0;background:var(--panel);color:var(--accent)}}
.facts{{display:flex;gap:18px;flex-wrap:wrap;margin-top:8px}} .fact b{{display:block;font-size:19px;color:var(--accent)}}
</style></head><body><div class="wrap">
<header><div><h1>Laboratoire des modeles auxiliaires</h1><div class="muted">{escape(experiment_id)} — entrainement, validation et test separes</div></div><button id="theme">Mode nuit</button></header>
<div class="facts">
  <div class="fact"><b>{len(leaderboard)}</b>configurations</div>
  <div class="fact"><b>{len(test)}</b>modeles evalues</div>
  <div class="fact"><b>{escape(str(manifest.get('source_run', '')))}</b>source immuable</div>
</div>
<main class="grid">
  <section class="panel">{_plot_html(metric_fig)}</section>
  <section class="panel">{_plot_html(leaderboard_fig)}</section>
  <section class="panel wide">{_plot_html(daily_fig)}</section>
  <section class="panel wide">{_plot_html(cumulative_fig)}</section>
  <section class="panel wide">{_plot_html(importance_fig)}</section>
  {kalman_coefficient_html}
  <section class="panel wide"><h2>Configurations selectionnees</h2>{selected_html}</section>
  <section class="panel wide"><h2>Metriques train / validation / test</h2>{_table(metrics)}</section>
  <section class="panel wide"><h2>Leaderboard complet</h2>{_table(leaderboard)}</section>
</main></div>
<script>
const root=document.documentElement, btn=document.getElementById('theme');
function applyTheme(dark){{root.dataset.theme=dark?'dark':'light';btn.textContent=dark?'Mode clair':'Mode nuit';localStorage.setItem('auxlab-theme',dark?'dark':'light');
document.querySelectorAll('.plotly-graph-div').forEach(el=>Plotly.relayout(el,{{paper_bgcolor:dark?'#1b2530':'#fff',plot_bgcolor:dark?'#1b2530':'#fff',font:{{color:dark?'#edf3f8':'#17202a'}},xaxis:{{gridcolor:dark?'#344454':'#e8edf2'}},yaxis:{{gridcolor:dark?'#344454':'#e8edf2'}}}}));}}
btn.onclick=()=>applyTheme(root.dataset.theme!=='dark'); applyTheme(localStorage.getItem('auxlab-theme')==='dark');
</script></body></html>"""
    output.write_text(html, encoding="utf-8")
    return output


__all__ = ["write_report"]
