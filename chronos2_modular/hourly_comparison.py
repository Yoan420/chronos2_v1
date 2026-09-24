"""Evaluation-only hourly profiles, built from the report's paired Statistics.

No input loading, model execution or independent date-window selection occurs
here. Every statistic uses the same finite (observed, model, benchmark) triples.
"""

from __future__ import annotations

import html
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def hourly_comparison_payload(
    result: Any, *, source: pd.DataFrame, benchmark_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate 00..23 local hours, retaining both physical autumn folds."""
    required = {"_timestamp_utc", "actual", "q50", "_benchmark_q50"}
    if not required.issubset(source.columns):
        raise ValueError("Hourly comparison requires the aligned Statistics source.")
    frame = source.loc[:, list(required)].copy()
    utc = pd.to_datetime(frame["_timestamp_utc"], utc=True, errors="raise")
    if utc.isna().any() or utc.duplicated().any():
        raise ValueError("Hourly comparison timestamps must be finite and unique.")
    if ((utc.dt.minute != 0) | (utc.dt.second != 0) | (utc.dt.microsecond != 0)).any():
        raise ValueError("Hourly comparison requires physical hourly timestamps.")
    if "_timestamp_local" in source:
        local = pd.to_datetime(source["_timestamp_local"], errors="raise")
        if local.dt.tz is None or not pd.DatetimeIndex(local).tz_convert("UTC").equals(pd.DatetimeIndex(utc)):
            raise ValueError("Hourly comparison local and UTC timestamps disagree.")
    else:
        timezone = getattr(getattr(getattr(result, "zone_data", None), "target", None), "index", None)
        local = utc.dt.tz_convert(getattr(timezone, "tz", None) or "UTC")
    timezone = str(local.dt.tz)
    payload: dict[str, Any] = {
        "zone": str(getattr(result, "zone", "")),
        "candidate_label": str(getattr(result, "statistics_candidate_label", "Modèle")),
        "benchmark_label": str(benchmark_contract.get("report_label") or benchmark_contract.get("label") or "Benchmark"),
        "benchmark_contract": dict(benchmark_contract),
        "timezone": timezone, "start_day": None, "end_day": None,
        "calendar_days": 0, "expected_hours": 0, "source_hours": len(frame),
        "observed_hours": 0, "paired_hours": 0, "coverage": None,
        "paired_start_day": None, "paired_end_day": None, "paired_days": 0,
        "evaluation_only": True, "records": [],
    }
    expected_counts: dict[int, int] = {}
    if not frame.empty:
        first, last = min(local.dt.date), max(local.dt.date)
        expected = pd.date_range(
            pd.Timestamp(first).tz_localize(timezone),
            (pd.Timestamp(last) + pd.Timedelta(days=1)).tz_localize(timezone),
            freq="h", inclusive="left",
        )
        expected_counts = pd.Series(expected.hour).value_counts().to_dict()
        payload.update({
            "start_day": first.isoformat(), "end_day": last.isoformat(),
            "calendar_days": (last - first).days + 1, "expected_hours": len(expected),
        })
    for column in ("actual", "q50", "_benchmark_q50"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["hour"] = local.dt.hour.to_numpy()
    frame["day"] = local.dt.date.to_numpy()
    actual_finite = np.isfinite(frame["actual"].to_numpy(dtype=float))
    finite = np.isfinite(frame[["actual", "q50", "_benchmark_q50"]].to_numpy(dtype=float)).all(axis=1)
    paired = frame.loc[finite]
    payload["observed_hours"] = int(actual_finite.sum())
    payload["paired_hours"] = len(paired)
    if not paired.empty:
        payload.update(paired_start_day=min(paired.day).isoformat(),
                       paired_end_day=max(paired.day).isoformat(),
                       paired_days=int(paired.day.nunique()))
    payload["coverage"] = len(paired) / payload["expected_hours"] if payload["expected_hours"] else None
    for hour in range(24):
        block = paired.loc[paired.hour.eq(hour)]
        count = len(block)
        record: dict[str, Any] = {
            "hour": hour, "label": f"{hour:02d}:00", "paired_hours": count,
            "paired_days": int(block.day.nunique()),
            "expected_hours": int(expected_counts.get(hour, 0)),
            "candidate_mae": None, "benchmark_mae": None, "mae_gain": None,
            "candidate_mean": None, "benchmark_mean": None, "actual_mean": None,
            "candidate_bias": None, "benchmark_bias": None,
        }
        if count:
            candidate_error = block.q50 - block.actual
            benchmark_error = block._benchmark_q50 - block.actual
            record.update({
                "candidate_mae": float(candidate_error.abs().mean()),
                "benchmark_mae": float(benchmark_error.abs().mean()),
                "candidate_bias": float(candidate_error.mean()),
                "benchmark_bias": float(benchmark_error.mean()),
                "candidate_mean": float(block.q50.mean()),
                "benchmark_mean": float(block._benchmark_q50.mean()),
                "actual_mean": float(block.actual.mean()),
            })
            record["mae_gain"] = record["benchmark_mae"] - record["candidate_mae"]
        payload["records"].append(record)
    return payload


def build_hourly_comparison_html(
    result: Any, *, source: pd.DataFrame,
    benchmark_contract: Mapping[str, Any], include_plotlyjs: bool = False,
) -> str:
    """Return a self-contained section using the host report's Plotly runtime."""
    payload = hourly_comparison_payload(result, source=source, benchmark_contract=benchmark_contract)
    candidate = html.escape(payload["candidate_label"])
    benchmark = html.escape(payload["benchmark_label"])
    heading = f"Performance par heure locale — {html.escape(payload['zone'])}"
    if not payload["paired_hours"]:
        return (
            '<section class="hourly-comparison"><h3>' + heading + '</h3>'
            '<p class="statistics-definition">Comparaison horaire indisponible : '
            'aucune heure commune avec un prix observé, le modèle et '
            + benchmark + '. Aucune valeur n’est remplacée par zéro.</p></section>'
        )
    records = payload["records"]
    hours = [row["label"] for row in records]
    custom = [[row["paired_hours"], row["expected_hours"], row["paired_days"], row["mae_gain"]] for row in records]
    figure = go.Figure()
    measures = [
        ("candidate_mae", candidate, "#3b82f6", "bar"),
        ("benchmark_mae", benchmark, "#d97706", "bar"),
        ("candidate_mean", candidate, "#3b82f6", "line"),
        ("benchmark_mean", benchmark, "#d97706", "line"),
        ("actual_mean", "Observé", "#0d9488", "line"),
        ("candidate_bias", candidate, "#3b82f6", "bar"),
        ("benchmark_bias", benchmark, "#d97706", "bar"),
    ]
    for position, (key, label, color, kind) in enumerate(measures):
        options = {
            "x": hours, "y": [row[key] for row in records], "name": label,
            "visible": position < 2, "customdata": custom,
            "hovertemplate": (
                "%{x} · " + label + "<br>%{y:.2f} EUR/MWh"
                "<br>%{customdata[0]} heures appariées / %{customdata[1]} attendues"
                "<br>%{customdata[2]} jours distincts"
                "<br>Gain MAE du modèle : %{customdata[3]:+.2f} EUR/MWh<extra></extra>"
            ),
        }
        figure.add_trace(go.Bar(**options, marker_color=color) if kind == "bar" else go.Scatter(
            **options, mode="lines+markers", line={"color": color, "width": 2.5},
            marker={"color": color, "size": 6}, connectgaps=False,
        ))
    buttons = []
    for label, visible, axis in (
        ("MAE — erreur absolue moyenne", [True, True, False, False, False, False, False], "MAE (EUR/MWh) — plus faible = meilleur"),
        ("Prix moyens — modèle / benchmark / observé", [False, False, True, True, True, False, False], "Prix moyen (EUR/MWh)"),
        ("Biais — erreur signée moyenne", [False, False, False, False, False, True, True], "Biais (EUR/MWh) — positif = surestimation"),
    ):
        buttons.append({"label": label, "method": "update", "args": [{"visible": visible}, {"yaxis.title.text": axis}]})
    figure.update_layout(
        template="plotly_white", height=480, barmode="group", hovermode="x unified",
        margin={"l": 65, "r": 20, "t": 125, "b": 60},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": f"Heure de livraison locale ({payload['timezone']})", "type": "category", "categoryorder": "array", "categoryarray": hours},
        yaxis={"title": "MAE (EUR/MWh) — plus faible = meilleur", "rangemode": "tozero"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        updatemenus=[{"type": "dropdown", "active": 0, "x": 0, "y": 1.28, "xanchor": "left", "yanchor": "top", "buttons": buttons}],
        meta={"evaluation_only": True, "source": "paired_statistics", "aggregation": "physical_hour_weighted_local_hour", "paired_hours": payload["paired_hours"]},
    )
    div_id = "hourly-comparison-" + uuid4().hex
    chart = pio.to_html(figure, full_html=False, include_plotlyjs="inline" if include_plotlyjs else False, div_id=div_id, config={"responsive": True, "displaylogo": False})
    # The host controller themes every Plotly plot; it does not theme dropdowns.
    dropdown_theme = f'''<script>(() => {{
      const graph = document.getElementById("{div_id}");
      function themeMenu() {{
        if (!graph || !window.Plotly) return;
        const dark = document.documentElement.dataset.theme === "dark";
        window.Plotly.relayout(graph, {{
          "updatemenus[0].bgcolor": dark ? "#111b2e" : "#ffffff",
          "updatemenus[0].bordercolor": dark ? "#41536c" : "#cbd5df",
          "updatemenus[0].font.color": dark ? "#e7edf6" : "#18212b"
        }});
      }}
      window.addEventListener("chronos2-theme-change", themeMenu);
      window.requestAnimationFrame(themeMenu);
    }})();</script>'''
    note = html.escape(str(benchmark_contract.get("report_note") or "Contrat du benchmark non précisé."))
    return f'''<section class="hourly-comparison" aria-label="{heading}">
      <h3>{heading}</h3>
      <p class="statistics-definition"><strong>Période évaluée : {payload['paired_start_day']} → {payload['paired_end_day']}
      · {payload['paired_days']} jours avec heures appariées.</strong><br>
      Fenêtre de Statistics, emplacements non observés inclus : {payload['start_day']} → {payload['end_day']}
      · {payload['calendar_days']} jours civils · {payload['paired_hours']}/{payload['expected_hours']} heures
      appariées ({100 * payload['coverage']:.1f} %) · {payload['observed_hours']} heures avec observation.</p>
      {chart}{dropdown_theme}
      <p class="statistics-definition">La MAE est la moyenne des erreurs absolues horaires,
      pas l’écart entre les prix moyens. Un gain MAE positif favorise {candidate}.
      Les trois courbes de prix et les deux erreurs utilisent exactement les mêmes heures.
      Les observations non publiées et toute valeur absente sont exclues, jamais remplacées par zéro.
      Chaque heure physique compte une fois : les deux occurrences de 02 h en automne sont conservées,
      l’heure inexistante au printemps n’est pas créée. Les effectifs figurent au survol.</p>
      <p class="statistics-definition">Benchmark : {benchmark}. {note}
      Évaluation uniquement : aucune donnée de benchmark ne sert à la prédiction.</p>
    </section>'''


__all__ = ["hourly_comparison_payload", "build_hourly_comparison_html"]
