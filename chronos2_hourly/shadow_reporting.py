"""Transparent reporting for prospective shadow forecasts."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def write_forecast_only_shadow_report(
    forecast_path: str | Path,
    *,
    output_path: str | Path,
    zone: str,
    delivery_day: str,
    candidate_model: str,
) -> Path:
    """Write a J+1-only report with no historical-performance claim."""

    source = Path(forecast_path).resolve()
    output = Path(output_path).resolve()
    frame = pd.read_csv(source)
    quantile_columns = {
        quantile: f"{candidate_model}__{quantile}"
        for quantile in ("q10", "q50", "q90")
    }
    missing = sorted(
        column for column in quantile_columns.values() if column not in frame
    )
    if missing:
        raise ValueError(
            f"Rapport shadow impossible; quantiles absents: {missing}."
        )
    timestamp_column = next(
        (
            column
            for column in ("delivery_start_local", "delivery_start_utc")
            if column in frame
        ),
        None,
    )
    if timestamp_column is None:
        raise ValueError(
            "Rapport shadow impossible; timeline de livraison absente."
        )
    table = pd.DataFrame(
        {
            "Heure de livraison": frame[timestamp_column].astype(str),
            "Q10 (EUR/MWh)": pd.to_numeric(
                frame[quantile_columns["q10"]], errors="raise"
            ),
            "Q50 (EUR/MWh)": pd.to_numeric(
                frame[quantile_columns["q50"]], errors="raise"
            ),
            "Q90 (EUR/MWh)": pd.to_numeric(
                frame[quantile_columns["q90"]], errors="raise"
            ),
        }
    )
    table_html = table.to_html(
        index=False,
        border=0,
        classes="forecast-table",
        justify="right",
        float_format=lambda value: f"{value:.2f}",
    )
    safe_zone = escape(str(zone))
    safe_day = escape(str(delivery_day))
    safe_model = escape(str(candidate_model))
    document = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Forecast {safe_zone} {safe_day} - shadow Chronos-2</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, Segoe UI, sans-serif; }}
    body {{ margin: 0; background: #f4f7fb; color: #172033; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero {{ background: #172033; color: white; border-radius: 16px; padding: 28px; }}
    .badge {{ display: inline-block; padding: 6px 10px; border-radius: 999px;
      background: #d8f5e5; color: #14532d; font-size: 13px; font-weight: 700; }}
    h1 {{ margin: 14px 0 8px; font-size: 30px; }}
    .card {{ margin-top: 20px; background: white; border-radius: 16px;
      padding: 22px; box-shadow: 0 8px 28px rgba(23, 32, 51, .08); }}
    .notice {{ border-left: 4px solid #2563eb; padding: 12px 16px;
      background: #eff6ff; line-height: 1.5; }}
    .meta {{ margin: 20px 0; color: #475569; }}
    .table-wrap {{ overflow-x: auto; }}
    .forecast-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .forecast-table th, .forecast-table td {{ padding: 10px 12px;
      border-bottom: 1px solid #e2e8f0; white-space: nowrap; }}
    .forecast-table th {{ background: #f8fafc; text-align: right; }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <span class="badge">SHADOW PROSPECTIF · FORECAST UNIQUEMENT</span>
    <h1>{safe_zone} · livraison {safe_day}</h1>
  </section>
  <section class="card">
    <div class="notice">
      Cette archive contient uniquement la prévision J+1 du challenger dont
      les cinq charges résiduelles futures viennent de Chronos-2.
      Aucune performance historique Saturn n’est présentée ou attribuée à ce
      challenger. L’évaluation est réalisée séparément sur les paires
      prospectives après observation des prix réels.
    </div>
    <p class="meta">Modèle de prix gelé : <strong>{safe_model}</strong> ·
      {len(table)} heures.</p>
    <div class="table-wrap">{table_html}</div>
  </section>
</main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


__all__ = ["write_forecast_only_shadow_report"]
