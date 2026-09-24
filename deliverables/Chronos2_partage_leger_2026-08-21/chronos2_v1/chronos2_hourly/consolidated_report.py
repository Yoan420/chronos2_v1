"""Standalone interactive report for audited multi-zone day-ahead forecasts.

The generated file embeds Plotly, the forecast values and the sealed Statistics
artifacts.  It therefore remains fully usable offline: no CDN, API call or
mutable benchmark is contacted when the user opens the report.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_hourly.app_service import (
    STATISTIC_DEFINITIONS,
    ForecastComparison,
    build_statistics_view,
    load_statistics_history,
)


_PALETTE: tuple[str, ...] = (
    "#2563EB",
    "#DC2626",
    "#059669",
    "#D97706",
    "#7C3AED",
    "#0891B2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Archive illisible pendant la generation du rapport: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Le fichier {path} doit contenir un objet JSON.")
    return payload


def _verify_archive_is_unchanged(
    archive_path: Path,
    *,
    expected_checksum_manifest_sha256: str,
) -> None:
    """Rehash the complete sealed archive immediately before export."""

    archive = archive_path.resolve()
    checksum_path = archive / "artifact_checksums.json"
    if _sha256(checksum_path) != expected_checksum_manifest_sha256:
        raise ValueError(f"{archive}: le manifeste de checksums a change.")
    payload = _json_object(checksum_path)
    if payload.get("algorithm") != "sha256":
        raise ValueError(f"{archive}: algorithme de checksum inattendu.")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError(f"{archive}: liste de checksums absente.")
    declared: dict[str, Mapping[str, Any]] = {}
    for item in raw_artifacts:
        if not isinstance(item, Mapping) or item.get("role") != "run_artifact":
            continue
        relative_path = Path(str(item.get("path", "")))
        if (
            not str(relative_path)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(f"{archive}: chemin de checksum non sur: {relative_path}")
        relative = relative_path.as_posix()
        if relative in declared:
            raise ValueError(f"{archive}: checksum duplique pour {relative}.")
        declared[relative] = item
    actual = {
        path.resolve().relative_to(archive).as_posix()
        for path in archive.rglob("*")
        if path.is_file() and path.name != "artifact_checksums.json"
    }
    if actual != set(declared):
        raise ValueError(f"{archive}: couverture de checksums divergente avant export.")
    for relative, declaration in declared.items():
        path = archive / relative
        try:
            expected_size = int(declaration.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{archive}: taille declaree invalide pour {relative}.") from exc
        expected_hash = str(declaration.get("sha256", "")).lower()
        if (
            path.stat().st_size != expected_size
            or len(expected_hash) != 64
            or _sha256(path) != expected_hash
        ):
            raise ValueError(f"{archive}: artefact modifie avant export: {relative}.")


def _validate_comparison_for_export(comparison: ForecastComparison) -> None:
    if not comparison.archives:
        raise ValueError("Le rapport consolide requiert au moins un forecast.")
    frame = comparison.frame
    required = {
        "timestamp_utc",
        "local_delivery",
        "zone",
        "delivery_day",
        "timezone",
        "P10",
        "P50",
        "P90",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Colonnes de comparaison absentes: {missing}")
    archive_zones = [archive.zone for archive in comparison.archives]
    if len(set(archive_zones)) != len(archive_zones):
        raise ValueError("Une zone apparait plusieurs fois dans le rapport consolide.")
    if set(frame["zone"].astype(str)) != set(archive_zones):
        raise ValueError("Les zones du tableau ne correspondent pas aux archives auditees.")
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if frame.assign(_timestamp=timestamps).duplicated(["zone", "_timestamp"]).any():
        raise ValueError("La comparaison contient des timestamps dupliques.")
    quantiles = frame[["P10", "P50", "P90"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(quantiles).all()
        or bool((quantiles[:, 0] > quantiles[:, 1]).any())
        or bool((quantiles[:, 1] > quantiles[:, 2]).any())
    ):
        raise ValueError("Les quantiles consolides sont incomplets ou croises.")

    timelines: list[pd.DatetimeIndex] = []
    for archive in comparison.archives:
        block = frame.loc[frame["zone"] == archive.zone].copy()
        if set(block["delivery_day"].astype(str)) != {archive.delivery_day}:
            raise ValueError(f"{archive.zone}: jour de livraison incoherent.")
        if set(block["timezone"].astype(str)) != {archive.timezone}:
            raise ValueError(f"{archive.zone}: timezone incoherente.")
        timeline = pd.DatetimeIndex(
            pd.to_datetime(block["timestamp_utc"], utc=True, errors="raise")
        ).sort_values()
        if timeline.has_duplicates or not timeline.is_monotonic_increasing:
            raise ValueError(f"{archive.zone}: timeline invalide.")
        timelines.append(timeline)
        if _sha256(archive.forecast_path) != archive.forecast_sha256:
            raise ValueError(f"{archive.zone}: forecast modifie avant export.")
        _verify_archive_is_unchanged(
            archive.archive_path,
            expected_checksum_manifest_sha256=archive.checksum_manifest_sha256,
        )
        manifest = _json_object(archive.archive_path / "run_manifest.json")
        expected_identity = {
            "zone": archive.zone,
            "timezone": archive.timezone,
            "delivery_day_local": archive.delivery_day,
            "run_type": "live_day_ahead",
            "forecast_status": "issued_live",
        }
        for field, expected in expected_identity.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"{archive.zone}: run_manifest.{field} ne correspond plus a l'archive."
                )
        if (
            manifest.get("storm_used_as_feature") is not False
            or manifest.get("storm_loaded_for_prediction") is not False
        ):
            raise ValueError(
                f"{archive.zone}: exclusion de Storm non prouvee avant export."
            )

    same_day = len({archive.delivery_day for archive in comparison.archives}) == 1
    aligned = same_day and all(
        timeline.equals(timelines[0]) for timeline in timelines[1:]
    )
    if same_day and (not comparison.timeline_aligned or not aligned):
        raise ValueError("Les timelines du meme jour ne sont pas alignees.")
    if not same_day and comparison.timeline_aligned:
        raise ValueError("Une comparaison multi-dates ne peut pas etre marquee alignee.")


def _svg_chart(comparison: ForecastComparison, *, include_intervals: bool) -> str:
    frame = comparison.frame.copy()
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="raise"
    )
    width, height = 1180.0, 520.0
    left, right, top, bottom = 76.0, 30.0, 48.0, 70.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = frame["timestamp_utc"].map(pd.Timestamp.timestamp).to_numpy(dtype=float)
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    y_columns = ["P10", "P90"] if include_intervals else ["P50"]
    y_values = frame[y_columns].to_numpy(dtype=float)
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    padding = max((y_max - y_min) * 0.08, 1.0)
    y_min -= padding
    y_max += padding

    def x_pos(value: pd.Timestamp) -> float:
        seconds = value.timestamp()
        return left + (seconds - x_min) / (x_max - x_min) * plot_width

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    elements: list[str] = [
        f'<svg viewBox="0 0 {int(width)} {int(height)}" role="img" '
        'aria-label="Comparaison des forecasts P50 par pays">',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="#FFFFFF" stroke="#CBD5E1"/>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y = y_pos(float(tick))
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
            f'y2="{y:.2f}" stroke="#E2E8F0"/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'class="axis-label">{tick:.1f}</text>'
        )
    tick_seconds = np.linspace(x_min, x_max, 7)
    for seconds in tick_seconds:
        x = left + (float(seconds) - x_min) / (x_max - x_min) * plot_width
        label = datetime.fromtimestamp(float(seconds), tz=timezone.utc).strftime(
            "%d/%m %H:%M"
        )
        elements.append(
            f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" '
            f'y2="{top + plot_height + 5}" stroke="#64748B"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 23}" '
            f'text-anchor="middle" class="axis-label">{escape(label)}</text>'
        )

    for position, archive in enumerate(comparison.archives):
        color = _PALETTE[position % len(_PALETTE)]
        block = frame.loc[frame["zone"] == archive.zone].sort_values(
            "timestamp_utc", kind="stable"
        )
        if include_intervals:
            lower = [
                f"{x_pos(row.timestamp_utc):.2f},{y_pos(float(row.P10)):.2f}"
                for row in block.itertuples(index=False)
            ]
            upper = [
                f"{x_pos(row.timestamp_utc):.2f},{y_pos(float(row.P90)):.2f}"
                for row in reversed(list(block.itertuples(index=False)))
            ]
            elements.append(
                f'<polygon points="{" ".join(lower + upper)}" fill="{color}" '
                'fill-opacity="0.10" stroke="none"/>'
            )
        line = [
            f"{x_pos(row.timestamp_utc):.2f},{y_pos(float(row.P50)):.2f}"
            for row in block.itertuples(index=False)
        ]
        elements.append(
            f'<polyline points="{" ".join(line)}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend_x = left + position * 160
        elements.extend(
            [
                f'<line x1="{legend_x}" y1="20" x2="{legend_x + 28}" y2="20" '
                f'stroke="{color}" stroke-width="4"/>',
                f'<text x="{legend_x + 36}" y="24" class="legend">'
                f'{escape(archive.zone)} · {escape(archive.delivery_day)}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="18" y="{top + plot_height / 2}" class="axis-title" '
            'text-anchor="middle" transform="rotate(-90 18 '
            f'{top + plot_height / 2})">Prix (EUR/MWh)</text>',
            f'<text x="{left + plot_width / 2}" y="{height - 10}" '
            'text-anchor="middle" class="axis-title">Livraison UTC</text>',
            "</svg>",
        ]
    )
    return "".join(elements)


def _json_value(value: Any) -> Any:
    """Return a compact JSON-safe scalar without non-standard NaN values."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        return timestamp.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(column): _json_value(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _statistics_payload(comparison: ForecastComparison) -> dict[str, Any]:
    """Load the immutable reporting-only histories for every selected zone."""

    payload: dict[str, Any] = {}
    for archive in comparison.archives:
        statistics_path = archive.archive_path / "statistics_history_hourly.csv.gz"
        if not statistics_path.is_file():
            payload[archive.zone] = {
                "available": False,
                "reason": "Aucun historique Statistics scelle dans cette archive.",
                "candidate_label": None,
                "benchmark_label": None,
                "scope_note": None,
                "time_series": [],
                "views": {},
            }
            continue
        dataset = load_statistics_history(
            statistics_path,
            variant=comparison.variant,
        )
        history = dataset.frame.copy()
        history["local_date"] = (
            history["timestamp"]
            .dt.tz_convert(archive.timezone)
            .dt.strftime("%Y-%m-%d")
        )
        payload[archive.zone] = {
            "available": True,
            "reason": None,
            "candidate_label": dataset.candidate_label,
            "benchmark_label": dataset.benchmark_label,
            "scope_note": dataset.scope_note,
            "history_start": str(history["local_date"].min()) if len(history) else None,
            "history_end": str(history["local_date"].max()) if len(history) else None,
            "history_hours": int(len(history)),
            "benchmark_hours": int(pd.to_numeric(history["benchmark"], errors="coerce").notna().sum()),
            "time_series": _records(history),
            "views": {},
        }
        for sample in ("daily", "weekly", "monthly"):
            view = build_statistics_view(
                dataset,
                timezone_name=archive.timezone,
                sample=sample,
            )
            payload[archive.zone]["views"][sample] = {
                "summary": _records(view.summary),
                "periods": _records(view.periods),
            }
    return payload


def _report_payload(
    comparison: ForecastComparison,
    *,
    include_intervals: bool,
) -> dict[str, Any]:
    forecast = comparison.frame.copy()
    forecast["timestamp_utc"] = pd.to_datetime(
        forecast["timestamp_utc"], utc=True, errors="raise"
    ).map(pd.Timestamp.isoformat)
    metrics = [
        {"key": key, "label": label, "higher_is_better": higher}
        for key, label, higher in STATISTIC_DEFINITIONS
    ]
    archives = [
        {
            "zone": archive.zone,
            "timezone": archive.timezone,
            "delivery_day": archive.delivery_day,
            "hours": int(len(archive.frame)),
            "archive_path": str(archive.archive_path),
            "forecast_sha256": archive.forecast_sha256,
            "checksum_manifest_sha256": archive.checksum_manifest_sha256,
        }
        for archive in comparison.archives
    ]
    return {
        "schema_version": "chronos2.consolidated.interactive.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "initial_include_intervals": bool(include_intervals),
        "timeline_aligned": bool(comparison.timeline_aligned),
        "mixed_delivery_days": bool(comparison.mixed_delivery_days),
        "forecast_variant": comparison.variant,
        "zones": [archive.zone for archive in comparison.archives],
        "metrics": metrics,
        "archives": archives,
        "forecast": _records(forecast),
        "statistics": _statistics_payload(comparison),
    }


def _safe_json(payload: Mapping[str, Any]) -> str:
    # Escaping ``</`` prevents a label/path from closing the JSON script node.
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")


def _embedded_plotly() -> str:
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Plotly est requis pour generer le rapport consolide interactif."
        ) from exc
    return str(get_plotlyjs())


def consolidated_report_filename(comparison: ForecastComparison) -> str:
    zones = "-".join(archive.zone.lower() for archive in comparison.archives)
    days = sorted({archive.delivery_day for archive in comparison.archives})
    day_token = days[0] if len(days) == 1 else f"{days[0]}_to_{days[-1]}"
    variant = (
        ""
        if comparison.variant == "production"
        else comparison.variant.replace("_", "-") + "_"
    )
    return f"chronos2_forecasts_{variant}{zones}_{day_token}.html"


def _render_static_consolidated_forecast_report(
    comparison: ForecastComparison,
    *,
    include_intervals: bool = False,
) -> bytes:
    """Build a self-contained, offline HTML comparison report."""

    _validate_comparison_for_export(comparison)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    date_map = ", ".join(
        f"{archive.zone}={archive.delivery_day}" for archive in comparison.archives
    )
    mixed_banner = (
        '<div class="warning"><strong>Attention — dates différentes.</strong> '
        f"Cette vue compare explicitement {escape(date_map)}. Les courbes ne "
        "représentent donc pas le même jour de livraison.</div>"
        if comparison.mixed_delivery_days
        else ""
    )
    archive_rows = "".join(
        "<tr>"
        f"<td>{escape(archive.zone)}</td>"
        f"<td>{escape(archive.delivery_day)}</td>"
        f"<td>{escape(archive.timezone)}</td>"
        f"<td>{len(archive.frame)}</td>"
        f"<td><code>{escape(archive.forecast_sha256)}</code></td>"
        f"<td><code>{escape(archive.checksum_manifest_sha256)}</code></td>"
        f"<td><code>{escape(str(archive.archive_path))}</code></td>"
        "</tr>"
        for archive in comparison.archives
    )
    value_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.zone))}</td>"
        f"<td>{escape(str(row.delivery_day))}</td>"
        f"<td>{escape(str(row.local_delivery))}</td>"
        f"<td>{pd.Timestamp(row.timestamp_utc).strftime('%Y-%m-%d %H:%M %z')}</td>"
        f"<td>{float(row.P10):.3f}</td>"
        f"<td>{float(row.P50):.3f}</td>"
        f"<td>{float(row.P90):.3f}</td>"
        "</tr>"
        for row in comparison.frame.sort_values(
            ["timestamp_utc", "zone"], kind="stable"
        ).itertuples(index=False)
    )
    interval_note = (
        "Les bandes transparentes représentent P10–P90."
        if include_intervals
        else "Les bandes P10–P90 sont masquées dans cette exportation."
    )
    variant_label = (
        "Autonome · sans MKOnline"
        if comparison.variant == "autonomous"
        else "Blend MKOnline validé"
        if comparison.variant == "mkonline_blend"
        else "Production actuelle"
    )
    svg = _svg_chart(comparison, include_intervals=include_intervals)
    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chronos-2 · Forecasts multi-pays</title>
<style>
:root {{ color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; color: #0F172A; }}
body {{ margin: 0; background: #F8FAFC; }}
main {{ max-width: 1320px; margin: 0 auto; padding: 32px; }}
h1 {{ margin-bottom: 6px; }} h2 {{ margin-top: 30px; }}
.muted {{ color: #475569; }}
.card {{ background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 20px; margin-top: 18px; box-shadow: 0 4px 14px #0F172A0A; }}
.warning {{ background: #FFF7ED; border: 1px solid #FDBA74; border-radius: 10px; padding: 14px; margin-top: 16px; color: #9A3412; }}
.safe {{ background: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 10px; padding: 12px; color: #065F46; }}
.chart {{ overflow-x: auto; }} svg {{ min-width: 820px; width: 100%; height: auto; }}
.axis-label {{ fill: #475569; font-size: 12px; }} .axis-title {{ fill: #334155; font-size: 13px; font-weight: 600; }} .legend {{ fill: #0F172A; font-size: 13px; font-weight: 650; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }} th, td {{ border-bottom: 1px solid #E2E8F0; text-align: left; padding: 9px; vertical-align: top; }} th {{ background: #F1F5F9; position: sticky; top: 0; }} code {{ overflow-wrap: anywhere; font-size: 11px; }}
.table-wrap {{ overflow: auto; max-height: 620px; }}
</style>
</head>
<body><main>
<h1>Forecasts day-ahead multi-pays</h1>
<p class="muted">Généré le {escape(generated)} · {len(comparison.archives)} pays · {escape(date_map)} · {escape(variant_label)}</p>
<div class="safe"><strong>Intégrité vérifiée.</strong> Toutes les archives et leurs checksums SHA-256 ont été revalidés avant l'export. Storm n'est ni une entrée de prédiction ni une série de ce rapport.</div>
{mixed_banner}
<section class="card"><h2>Comparaison P50</h2><p class="muted">{escape(interval_note)}</p><div class="chart">{svg}</div></section>
<section class="card"><h2>Provenance auditée</h2><div class="table-wrap"><table><thead><tr><th>Pays</th><th>Jour</th><th>Timezone</th><th>Heures</th><th>SHA-256 forecast</th><th>SHA-256 manifeste</th><th>Archive</th></tr></thead><tbody>{archive_rows}</tbody></table></div></section>
<section class="card"><h2>Valeurs horaires</h2><div class="table-wrap"><table><thead><tr><th>Pays</th><th>Jour</th><th>Livraison locale</th><th>Livraison UTC</th><th>P10</th><th>P50</th><th>P90</th></tr></thead><tbody>{value_rows}</tbody></table></div></section>
</main></body></html>"""
    return html.encode("utf-8")


def render_consolidated_forecast_report(
    comparison: ForecastComparison,
    *,
    include_intervals: bool = False,
) -> bytes:
    """Build the interactive, self-contained multi-country HTML dashboard."""

    _validate_comparison_for_export(comparison)
    report = _report_payload(comparison, include_intervals=include_intervals)
    data = _safe_json(report)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    date_map = ", ".join(
        f"{archive.zone}={archive.delivery_day}" for archive in comparison.archives
    )
    mixed_banner = (
        '<div class="banner warning"><strong>Dates de livraison différentes.</strong> '
        f"Comparaison explicite : {escape(date_map)}.</div>"
        if comparison.mixed_delivery_days
        else ""
    )
    variant_label = (
        "Autonome · sans MKOnline"
        if comparison.variant == "autonomous"
        else "Blend MKOnline validé"
        if comparison.variant == "mkonline_blend"
        else "Production actuelle"
    )
    template = r'''<!doctype html>
<html lang="fr" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; connect-src 'none'; object-src 'none'; base-uri 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:">
<title>Chronos-2 · Rapport consolidé interactif</title>
<style>
:root{color-scheme:light;font-family:Inter,"Segoe UI",Arial,sans-serif;--bg:#f5f7fb;--surface:#fff;--surface2:#f8fafc;--text:#0f172a;--muted:#526174;--line:#dbe3ed;--brand:#2563eb;--good:#047857;--bad:#b91c1c;--warn:#9a5b0a;--shadow:0 8px 28px rgba(15,23,42,.07)}
html[data-theme="dark"]{color-scheme:dark;--bg:#0b1220;--surface:#111b2e;--surface2:#172338;--text:#e7edf6;--muted:#aab7ca;--line:#2b3b52;--brand:#60a5fa;--good:#34d399;--bad:#fb7185;--warn:#fbbf24;--shadow:0 8px 28px rgba(0,0,0,.28)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text)}main{max-width:1540px;margin:auto;padding:26px 30px 60px}.topbar{display:flex;gap:18px;align-items:flex-start;justify-content:space-between}.topbar h1{margin:0 0 7px;font-size:clamp(25px,3vw,38px)}h2{margin:0 0 6px;font-size:21px}h3{margin:18px 0 8px}.muted{color:var(--muted)}.tiny{font-size:12px}.toolbar{position:sticky;top:0;z-index:30;display:flex;gap:8px;flex-wrap:wrap;padding:10px 0;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(8px)}button,.button,select,input{font:inherit;color:var(--text);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:8px 10px}button,.button{cursor:pointer}button:hover,.button:hover{border-color:var(--brand)}.navlink{text-decoration:none;font-size:13px}.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px;margin-top:16px;box-shadow:var(--shadow)}.banner{border-radius:10px;padding:12px 14px;margin-top:14px}.safe{background:color-mix(in srgb,var(--good) 10%,var(--surface));border:1px solid color-mix(in srgb,var(--good) 35%,var(--line));color:var(--good)}.warning{background:color-mix(in srgb,var(--warn) 12%,var(--surface));border:1px solid color-mix(in srgb,var(--warn) 38%,var(--line));color:var(--warn)}.controls{display:flex;align-items:end;gap:12px;flex-wrap:wrap;margin:13px 0}.control{display:grid;gap:5px}.control label,.label{font-size:12px;font-weight:700;color:var(--muted)}.checks{display:flex;gap:7px;flex-wrap:wrap}.check{display:flex;gap:6px;align-items:center;background:var(--surface2);border:1px solid var(--line);border-radius:999px;padding:6px 10px;font-size:13px}.check input{margin:0;padding:0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.kpi{background:var(--surface2);border:1px solid var(--line);border-radius:11px;padding:14px}.kpi .zone{font-size:20px;font-weight:800}.kpi .value{font-size:22px;font-weight:800;margin:8px 0 2px}.kpi .row{display:flex;justify-content:space-between;gap:8px;font-size:13px;margin-top:5px}.plot{width:100%;min-height:470px}.plot.small{min-height:390px}.split{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(340px,1fr);gap:16px}.table-wrap{overflow:auto;max-height:600px;border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}th{position:sticky;top:0;z-index:2;background:var(--surface2);font-size:11px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);cursor:pointer}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}tr:hover td{background:color-mix(in srgb,var(--brand) 5%,var(--surface))}#statistics-summary-table tbody tr{cursor:pointer}.positive{color:var(--good);font-weight:750}.negative{color:var(--bad);font-weight:750}.na{color:var(--muted);font-style:italic}.details-grid{display:grid;grid-template-columns:145px 1fr;gap:7px 12px}.details-grid code{overflow-wrap:anywhere;white-space:normal;font-size:11px}.foot{margin-top:24px;padding-top:15px;border-top:1px solid var(--line)}@media(max-width:900px){main{padding:18px 12px}.split{grid-template-columns:1fr}.topbar{display:block}.toolbar{position:static}.plot{min-height:400px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{.toolbar,.controls button{display:none!important}.card{break-inside:avoid;box-shadow:none}main{max-width:none}.plot{min-height:360px}}
</style>
</head>
<body><main>
<header class="topbar"><div><h1>Forecasts day-ahead · vue multi-pays</h1><div class="muted">Généré le __GENERATED__ · __DATES__ · __VARIANT__</div></div><button id="theme-toggle" type="button" aria-label="Changer le thème">◐ Thème</button></header>
<nav class="toolbar" aria-label="Sections"><a class="button navlink" href="#forecast">Forecasts</a><a class="button navlink" href="#overview">Synthèse</a><a class="button navlink" href="#history">Historique</a><a class="button navlink" href="#metrics">Statistics</a><a class="button navlink" href="#hourly-values">Valeurs</a><a class="button navlink" href="#provenance">Audit</a><button id="print-report" type="button">Imprimer / PDF</button></nav>
<div class="banner safe"><strong>Rapport autonome et audité.</strong> Les archives et leurs SHA-256 ont été revérifiés avant l'export. Plotly, les forecasts et Statistics sont embarqués : aucune requête réseau à l'ouverture. Storm est uniquement un comparateur d'évaluation.</div>
__MIXED_BANNER__
<noscript><div class="banner warning">JavaScript est nécessaire pour les graphiques interactifs.</div></noscript>

<section class="card" id="forecast"><h2>Forecast consolidé</h2><p class="muted">Survol, zoom, déplacement, sélection par légende et export PNG sont disponibles dans la barre Plotly.</p><div class="controls"><div class="control"><span class="label">Pays affichés</span><div id="forecast-zone-select" class="checks"></div></div><div class="control"><label for="forecast-start-date">Début</label><input id="forecast-start-date" type="date"></div><div class="control"><label for="forecast-end-date">Fin</label><input id="forecast-end-date" type="date"></div><label class="check"><input id="forecast-interval-toggle" type="checkbox"> P10–P90</label><label class="check"><input id="forecast-normalize" type="checkbox"> Variation depuis H00</label><button id="forecast-reset" type="button">Réinitialiser</button></div><div id="forecast-comparison-chart" class="plot" data-report-chart="forecast"></div></section>

<section class="card" id="overview"><h2>Synthèse de performance par pays</h2><p class="muted">Couverture candidat–Storm appariée lorsque le benchmark officiel est disponible.</p><div id="kpi-grid" class="grid"></div></section>

<section class="card" id="history"><h2>Explorateur historique</h2><div class="controls"><div class="control"><label for="statistics-zone-select">Pays</label><select id="statistics-zone-select"></select></div><div class="control"><label for="history-start">Début</label><input id="history-start" type="date"></div><div class="control"><label for="history-end">Fin</label><input id="history-end" type="date"></div><div class="control"><label for="history-mode">Vue</label><select id="history-mode"><option value="series">Série temporelle</option><option value="scatter">Réel vs forecast</option><option value="hour">Erreur par heure</option></select></div><button class="history-range" data-days="30" type="button">1 mois</button><button class="history-range" data-days="90" type="button">3 mois</button><button class="history-range" data-days="180" type="button">6 mois</button><button class="history-range" data-days="365" type="button">1 an</button><button id="history-full-range" type="button">Tout</button></div><div id="statistics-timeseries-chart" class="plot" data-report-chart="history"></div><p id="history-note" class="muted tiny"></p></section>

<section class="card" id="metrics"><h2>Statistics complètes</h2><p class="muted">Sept métriques pour tous les pays : MAE, RMSE, MAPE, Explained Variance, R², écart-type de l'erreur et corrélation. Chaque win rate inclut les égalités dans son dénominateur.</p><div class="controls"><div class="control"><label for="stats-sample">Granularité</label><select id="stats-sample"><option data-sample="daily" value="daily">Journalière</option><option data-sample="weekly" value="weekly">Hebdomadaire</option><option data-sample="monthly" value="monthly">Mensuelle</option></select></div><div class="control"><label for="statistics-metric-select">Métrique des graphiques</label><select id="statistics-metric-select"><option value="all">Toutes</option></select></div><button id="download-stats" type="button">Télécharger CSV</button></div><div class="split"><div><h3>Win rate face à Storm</h3><div id="winrate-plot" class="plot small" data-report-chart="win-rate"></div></div><div><h3>Évolution</h3><div class="controls"><select id="trend-zone" aria-label="Pays de la tendance"></select><select id="trend-metric" aria-label="Métrique de la tendance"></select></div><div id="statistics-period-chart" class="plot small" data-report-chart="metric-trend"></div></div></div><h3>Tableau exhaustif — toutes les métriques</h3><div class="table-wrap"><table id="statistics-summary-table"><thead><tr><th data-sort="zone">Pays</th><th data-sort="label">Métrique</th><th data-sort="candidate">Notre modèle</th><th data-sort="benchmark">Storm</th><th data-sort="delta">Écart</th><th data-sort="wins">Gagnés</th><th data-sort="ties">Égalités</th><th data-sort="losses">Perdus</th><th data-sort="comparable_periods">Périodes</th><th data-sort="win_rate">Win rate</th><th data-sort="history_hours">Heures</th><th data-sort="status">Statut</th></tr></thead><tbody></tbody></table></div><h3>Détail par période</h3><div class="table-wrap"><table id="statistics-period-table"><thead><tr><th>Période</th><th>Heures</th><th>Notre modèle</th><th>Storm</th><th>Résultat</th></tr></thead><tbody></tbody></table></div></section>

<section class="card" id="hourly-values"><h2>Valeurs horaires du forecast</h2><div class="controls"><div class="control"><label for="values-zone">Pays</label><select id="values-zone"><option value="all">Tous</option></select></div><input id="values-search" type="search" aria-label="Filtrer les valeurs" placeholder="Filtrer date / heure…"><button id="download-forecast" type="button">Télécharger CSV</button></div><div class="table-wrap"><table><thead><tr><th>Pays</th><th>Jour</th><th>Livraison locale</th><th>Livraison UTC</th><th>P10</th><th>P50</th><th>P90</th></tr></thead><tbody id="forecast-table-body"></tbody></table></div></section>
<section class="card" id="provenance"><h2>Provenance auditée</h2><div id="provenance-grid" class="grid"></div></section>
<footer class="foot muted tiny">Chronos-2 · rapport consolidé interactif · chronos2.consolidated.interactive.v1</footer>
</main>
<script>__PLOTLY__</script>
<script type="application/json" id="consolidated-report-data">__REPORT_DATA__</script>
<script>
(()=>{'use strict';
const R=JSON.parse(document.getElementById('consolidated-report-data').textContent),COLORS=['#2563eb','#dc2626','#059669','#d97706','#7c3aed','#0891b2','#db2777'],SAMPLES={daily:'Journalière',weekly:'Hebdomadaire',monthly:'Mensuelle'};
const $=id=>document.getElementById(id),color=z=>COLORS[R.zones.indexOf(z)%COLORS.length],alpha=(hex,a)=>{const n=parseInt(hex.slice(1),16);return `rgba(${n>>16},${(n>>8)&255},${n&255},${a})`},finite=v=>v!==null&&v!==undefined&&Number.isFinite(Number(v));
const fmt=(v,d=3)=>finite(v)?Number(v).toLocaleString('fr-FR',{minimumFractionDigits:d,maximumFractionDigits:d}):'N/A',pct=v=>finite(v)?`${(100*Number(v)).toLocaleString('fr-FR',{minimumFractionDigits:1,maximumFractionDigits:1})} %`:'N/A';
const cfg={responsive:true,displaylogo:false,scrollZoom:true,toImageButtonOptions:{format:'png',scale:2},locale:'fr'};
function layout(title,y){const dark=document.documentElement.dataset.theme==='dark';return{title:{text:title,x:.01,xanchor:'left',font:{size:15}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:dark?'#e7edf6':'#0f172a'},margin:{l:62,r:25,t:48,b:55},hovermode:'x unified',dragmode:'zoom',legend:{orientation:'h',y:1.11},xaxis:{gridcolor:dark?'#2b3b52':'#e5eaf1'},yaxis:{title:y,gridcolor:dark?'#2b3b52':'#e5eaf1',zerolinecolor:dark?'#43536a':'#cbd5e1'}}}
function option(s,v,l){const o=document.createElement('option');o.value=v;o.textContent=l;s.appendChild(o)}function td(v,c=''){const x=document.createElement('td');x.textContent=v;if(c)x.className=c;return x}function span(v){const x=document.createElement('span');x.textContent=v;return x}
function csvCell(v){const s=v==null?'':String(v);return /[",\n]/.test(s)?`"${s.replaceAll('"','""')}"`:s}function download(name,rows,cols){const blob=new Blob(['\ufeff'+[cols.join(','),...rows.map(r=>cols.map(c=>csvCell(r[c])).join(','))].join('\n')],{type:'text/csv;charset=utf-8'}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),500)}
R.zones.forEach(z=>{const l=document.createElement('label'),i=document.createElement('input');l.className='check';i.type='checkbox';i.value=z;i.checked=true;i.addEventListener('change',renderForecast);l.append(i,document.createTextNode(z));$('forecast-zone-select').appendChild(l);option($('values-zone'),z,z)});const AZ=R.zones.filter(z=>R.statistics[z]?.available);AZ.forEach(z=>{option($('statistics-zone-select'),z,z);option($('trend-zone'),z,z)});R.metrics.forEach(m=>{option($('statistics-metric-select'),m.key,m.label);option($('trend-metric'),m.key,m.label)});$('trend-metric').value='mae';
const forecastDays=R.forecast.map(r=>r.delivery_day).sort();$('forecast-start-date').min=forecastDays[0];$('forecast-start-date').max=forecastDays.at(-1);$('forecast-start-date').value=forecastDays[0];$('forecast-end-date').min=forecastDays[0];$('forecast-end-date').max=forecastDays.at(-1);$('forecast-end-date').value=forecastDays.at(-1);
function activeZones(){return [...document.querySelectorAll('#forecast-zone-select input:checked')].map(x=>x.value)}
function renderForecast(){const bands=$('forecast-interval-toggle').checked,norm=$('forecast-normalize').checked,start=$('forecast-start-date').value,end=$('forecast-end-date').value,tr=[];activeZones().forEach(z=>{const rows=R.forecast.filter(r=>r.zone===z&&r.delivery_day>=start&&r.delivery_day<=end),base=norm?Number(rows[0]?.P50||0):0,x=rows.map(r=>r.timestamp_utc),cd=rows.map(r=>[r.local_delivery,r.delivery_day]);if(bands){tr.push({x,y:rows.map(r=>r.P10-base),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip',legendgroup:z});tr.push({x,y:rows.map(r=>r.P90-base),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:alpha(color(z),.12),showlegend:false,legendgroup:z,hovertemplate:`${z} P10–P90<br>%{y:.2f}<extra></extra>`})}tr.push({x,y:rows.map(r=>r.P50-base),customdata:cd,mode:'lines+markers',name:`${z} P50`,legendgroup:z,line:{width:3,color:color(z)},marker:{size:5,symbol:R.zones.indexOf(z)%2?'diamond':'circle'},hovertemplate:`<b>${z}</b><br>%{customdata[0]}<br>P50 %{y:.2f} €/MWh<extra></extra>`})});const l=layout(norm?'Variation depuis la première heure':'Prix day-ahead par pays',norm?'Variation (€/MWh)':'Prix (€/MWh)');l.xaxis.rangeslider={visible:true,thickness:.08};l.legend={...l.legend,groupclick:'togglegroup'};l.uirevision='forecast';Plotly.react('forecast-comparison-chart',tr,l,cfg)}
function summary(z,s='daily'){return R.statistics[z]?.views?.[s]?.summary||[]}function metric(z,s,k){return summary(z,s).find(r=>r.metric===k)||{}}
function renderKpis(){$('kpi-grid').replaceChildren();R.zones.forEach(z=>{const s=R.statistics[z],c=document.createElement('article');c.className='kpi';const h=document.createElement('div');h.className='zone';h.style.color=color(z);h.textContent=z;c.appendChild(h);if(!s?.available){const p=document.createElement('p');p.className='na';p.textContent=s?.reason||'Statistics indisponibles';c.appendChild(p);$('kpi-grid').appendChild(c);return}const ma=metric(z,'daily','mae'),rm=metric(z,'daily','rmse'),mp=metric(z,'daily','mape'),v=document.createElement('div');v.className='value';v.textContent=`MAE ${fmt(ma.candidate,2)}`;c.appendChild(v);[['Storm',fmt(ma.benchmark,2)],['Win rate MAE',pct(ma.win_rate)],['RMSE',fmt(rm.candidate,2)],['MAPE',`${fmt(mp.candidate,1)} %`],['Historique',`${s.history_hours.toLocaleString('fr-FR')} h`]].forEach(([a,b])=>{const r=document.createElement('div');r.className='row';r.append(span(a),span(b));c.appendChild(r)});const n=document.createElement('div');n.className='muted tiny';n.style.marginTop='9px';n.textContent=s.benchmark_label||'Storm officiel indisponible';c.appendChild(n);$('kpi-grid').appendChild(c)})}
function setRange(){const s=R.statistics[$('statistics-zone-select').value];if(!s?.available)return;for(const id of ['history-start','history-end']){$(id).min=s.history_start;$(id).max=s.history_end}$('history-start').value=s.history_start;$('history-end').value=s.history_end}function historyRows(){const rows=R.statistics[$('statistics-zone-select').value]?.time_series||[],a=$('history-start').value,b=$('history-end').value;return rows.filter(r=>(!a||r.local_date>=a)&&(!b||r.local_date<=b))}
function renderHistory(){const z=$('statistics-zone-select').value,s=R.statistics[z],rows=historyRows(),mode=$('history-mode').value;if(!s)return;let tr,l;if(mode==='series'){const x=rows.map(r=>r.timestamp);tr=[{x,y:rows.map(r=>r.actual),name:'Réalisé',mode:'lines',line:{color:'#64748b',width:1.8}},{x,y:rows.map(r=>r.candidate),name:s.candidate_label,mode:'lines',line:{color:color(z),width:1.5}},{x,y:rows.map(r=>r.benchmark),name:s.benchmark_label||'Storm indisponible',mode:'lines',connectgaps:false,line:{color:'#f59e0b',width:1.3,dash:'dot'}}];l=layout(`${z} · réalisé, candidat et Storm`,'Prix (€/MWh)');l.xaxis.rangeslider={visible:true,thickness:.08}}else if(mode==='scatter'){tr=[{x:rows.map(r=>r.actual),y:rows.map(r=>r.candidate),name:s.candidate_label,mode:'markers',type:'scattergl',marker:{color:color(z),size:5,opacity:.45}},{x:rows.map(r=>r.actual),y:rows.map(r=>r.benchmark),name:s.benchmark_label||'Storm',mode:'markers',type:'scattergl',marker:{color:'#f59e0b',size:5,opacity:.4,symbol:'diamond'}}];const all=rows.flatMap(r=>[r.actual,r.candidate,r.benchmark]).filter(finite),lo=Math.min(...all),hi=Math.max(...all);tr.push({x:[lo,hi],y:[lo,hi],name:'Forecast parfait',mode:'lines',line:{dash:'dash',color:'#64748b'}});l=layout(`${z} · réel vs forecast`,'Forecast (€/MWh)');l.xaxis.title='Réalisé (€/MWh)';l.hovermode='closest'}else{const tz=R.archives.find(a=>a.zone===z).timezone,bins=Array.from({length:24},()=>({c:[],b:[]}));rows.forEach(r=>{const h=Number(new Intl.DateTimeFormat('fr-FR',{timeZone:tz,hour:'2-digit',hourCycle:'h23'}).format(new Date(r.timestamp)));if(finite(r.actual)&&finite(r.candidate))bins[h].c.push(Math.abs(r.candidate-r.actual));if(finite(r.actual)&&finite(r.benchmark))bins[h].b.push(Math.abs(r.benchmark-r.actual))});const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;tr=[{x:[...Array(24).keys()],y:bins.map(v=>mean(v.c)),name:s.candidate_label,type:'bar',marker:{color:color(z)}},{x:[...Array(24).keys()],y:bins.map(v=>mean(v.b)),name:s.benchmark_label||'Storm',type:'bar',marker:{color:'#f59e0b',pattern:{shape:'/'}}}];l=layout(`${z} · MAE par heure locale`,'MAE (€/MWh)');l.barmode='group';l.xaxis.title=`Heure locale (${tz})`}l.uirevision=`history-${z}-${mode}`;Plotly.react('statistics-timeseries-chart',tr,l,cfg);$('history-note').textContent=[s.scope_note,`Couverture Storm : ${s.benchmark_hours.toLocaleString('fr-FR')} / ${s.history_hours.toLocaleString('fr-FR')} h`].filter(Boolean).join(' · ')}
let statsRows=[],sortKey='zone',sortAsc=true;function buildStats(){const s=$('stats-sample').value;statsRows=[];R.zones.forEach(z=>summary(z,s).forEach(r=>{const x=R.statistics[z],partial=(x.scope_note||'').toLowerCase().includes('parti');statsRows.push({...r,zone:z,delta:finite(r.candidate)&&finite(r.benchmark)?r.candidate-r.benchmark:null,history_hours:x.history_hours||0,status:!x.benchmark_label?'Storm indisponible':partial?'Partiel':'Complet'})}));renderTable();renderHeat();renderTrend()}
function renderTable(){const b=$('statistics-summary-table').tBodies[0];b.replaceChildren();[...statsRows].sort((a,c)=>{const x=a[sortKey],y=c[sortKey];if(x==y)return 0;if(x==null)return 1;if(y==null)return -1;return(x>y?1:-1)*(sortAsc?1:-1)}).forEach(r=>{const tr=document.createElement('tr'),cl=finite(r.win_rate)?(r.win_rate>.5?'positive':r.win_rate<.5?'negative':''):'';tr.tabIndex=0;tr.title='Ouvrir le détail de cette métrique';[r.zone,r.label,fmt(r.candidate),fmt(r.benchmark),fmt(r.delta),String(r.wins||0),String(r.ties||0),String(r.losses||0),String(r.comparable_periods||0),pct(r.win_rate),String(r.history_hours||0),r.status].forEach((v,i)=>tr.appendChild(td(v,i===9?cl:'')));const open=()=>{if(AZ.includes(r.zone))$('trend-zone').value=r.zone;$('trend-metric').value=r.metric;renderTrend();$('statistics-period-chart').scrollIntoView({behavior:'smooth',block:'center'})};tr.addEventListener('click',open);tr.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open()}});b.appendChild(tr)})}
function renderHeat(){const s=$('stats-sample').value,f=$('statistics-metric-select').value,ms=R.metrics.filter(m=>f==='all'||m.key===f),z=R.zones.map(zone=>ms.map(m=>{const r=metric(zone,s,m.key);return finite(r.win_rate)?100*r.win_rate:null})),text=R.zones.map(zone=>ms.map(m=>{const r=metric(zone,s,m.key);return`${zone} · ${m.label}<br>${r.wins||0} W / ${r.ties||0} T / ${r.losses||0} L<br>${pct(r.win_rate)}`})),tr={type:'heatmap',x:ms.map(m=>m.label),y:R.zones,z,text,hovertemplate:'%{text}<extra></extra>',zmin:0,zmax:100,zmid:50,colorscale:[[0,'#dc2626'],[.5,'#f8fafc'],[1,'#059669']],colorbar:{title:'Win %'}};const l=layout(`Win rate ${SAMPLES[s].toLowerCase()}`,'Pays');l.margin.l=45;l.xaxis.tickangle=-20;l.hovermode='closest';l.uirevision=`heat-${s}-${f}`;Plotly.react('winrate-plot',[tr],l,cfg)}
function renderTrend(){const z=$('trend-zone').value,k=$('trend-metric').value,sample=$('stats-sample').value,s=R.statistics[z],rows=s?.views?.[sample]?.periods||[],m=R.metrics.find(x=>x.key===k),x=rows.map(r=>r.period),tr=[{x,y:rows.map(r=>r[`candidate_${k}`]),name:s?.candidate_label||'Notre modèle',mode:'lines+markers',line:{color:color(z)}},{x,y:rows.map(r=>r[`benchmark_${k}`]),name:s?.benchmark_label||'Storm',mode:'lines+markers',connectgaps:false,line:{color:'#f59e0b',dash:'dot'}}],l=layout(`${z} · ${m?.label||k} ${SAMPLES[sample].toLowerCase()}`,m?.label||k);l.uirevision=`trend-${z}-${k}-${sample}`;Plotly.react('statistics-period-chart',tr,l,cfg);const body=$('statistics-period-table').tBodies[0];body.replaceChildren();rows.forEach(r=>{const row=document.createElement('tr');[r.period,String(r.n||0),fmt(r[`candidate_${k}`]),fmt(r[`benchmark_${k}`]),r[`outcome_${k}`]||'N/A'].forEach(v=>row.appendChild(td(v)));body.appendChild(row)})}
function forecastRows(){const z=$('values-zone').value,q=$('values-search').value.trim().toLowerCase();return R.forecast.filter(r=>(z==='all'||r.zone===z)&&(!q||`${r.zone} ${r.delivery_day} ${r.local_delivery} ${r.timestamp_utc}`.toLowerCase().includes(q)))}function renderValues(){const b=$('forecast-table-body');b.replaceChildren();forecastRows().forEach(r=>{const tr=document.createElement('tr');[r.zone,r.delivery_day,r.local_delivery,r.timestamp_utc,fmt(r.P10),fmt(r.P50),fmt(r.P90)].forEach(v=>tr.appendChild(td(v)));b.appendChild(tr)})}
function renderProvenance(){$('provenance-grid').replaceChildren();R.archives.forEach(a=>{const c=document.createElement('article');c.className='kpi';const h=document.createElement('h3');h.textContent=`${a.zone} · ${a.delivery_day}`;c.appendChild(h);const g=document.createElement('div');g.className='details-grid';[['Timezone',a.timezone],['Heures',a.hours],['Forecast SHA-256',a.forecast_sha256],['Manifest SHA-256',a.checksum_manifest_sha256],['Archive',a.archive_path]].forEach(([k,v])=>{const b=document.createElement('strong'),x=document.createElement('code');b.textContent=k;x.textContent=String(v);g.append(b,x)});c.appendChild(g);$('provenance-grid').appendChild(c)})}
$('forecast-interval-toggle').checked=R.initial_include_intervals;$('forecast-interval-toggle').addEventListener('change',renderForecast);for(const id of ['forecast-start-date','forecast-end-date'])$(id).addEventListener('change',renderForecast);$('forecast-normalize').addEventListener('change',renderForecast);$('forecast-reset').addEventListener('click',()=>{document.querySelectorAll('#forecast-zone-select input').forEach(x=>x.checked=true);$('forecast-start-date').value=forecastDays[0];$('forecast-end-date').value=forecastDays.at(-1);$('forecast-normalize').checked=false;$('forecast-interval-toggle').checked=R.initial_include_intervals;renderForecast()});$('statistics-zone-select').addEventListener('change',()=>{setRange();renderHistory()});for(const id of ['history-mode','history-start','history-end'])$(id).addEventListener('change',renderHistory);$('history-full-range').addEventListener('click',()=>{setRange();renderHistory()});document.querySelectorAll('.history-range').forEach(button=>button.addEventListener('click',()=>{const end=new Date(`${R.statistics[$('statistics-zone-select').value].history_end.slice(0,10)}T00:00:00Z`),start=new Date(end);start.setUTCDate(end.getUTCDate()-Number(button.dataset.days)+1);$('history-start').value=start.toISOString().slice(0,10);$('history-end').value=end.toISOString().slice(0,10);renderHistory()}));for(const id of ['stats-sample','statistics-metric-select'])$(id).addEventListener('change',buildStats);for(const id of ['trend-zone','trend-metric'])$(id).addEventListener('change',renderTrend);document.querySelectorAll('#statistics-summary-table th').forEach(h=>h.addEventListener('click',()=>{const k=h.dataset.sort;if(sortKey===k)sortAsc=!sortAsc;else{sortKey=k;sortAsc=true}renderTable()}));$('values-zone').addEventListener('change',renderValues);$('values-search').addEventListener('input',renderValues);$('download-forecast').addEventListener('click',()=>download('forecasts_consolides.csv',forecastRows(),['zone','delivery_day','local_delivery','timestamp_utc','P10','P50','P90']));$('download-stats').addEventListener('click',()=>download(`statistics_${$('stats-sample').value}.csv`,statsRows,['zone','metric','label','candidate','benchmark','delta','wins','ties','losses','comparable_periods','win_rate','history_hours','status']));$('theme-toggle').addEventListener('click',()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark';renderForecast();if(AZ.length)renderHistory();renderHeat();renderTrend()});$('print-report').addEventListener('click',()=>window.print());
renderKpis();renderProvenance();renderValues();renderForecast();if(AZ.length){setRange();renderHistory();buildStats()}else{$('history-note').textContent='Aucun historique Statistics disponible.'}
})();
</script></body></html>'''
    html = (
        template.replace("__GENERATED__", escape(generated))
        .replace("__DATES__", escape(date_map))
        .replace("__VARIANT__", escape(variant_label))
        .replace("__MIXED_BANNER__", mixed_banner)
        .replace("__PLOTLY__", _embedded_plotly())
        .replace("__REPORT_DATA__", data)
    )
    return html.encode("utf-8")
