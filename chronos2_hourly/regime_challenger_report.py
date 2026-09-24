"""Standalone HTML reporting for one zone of the price-regime challenger.

The renderer is deliberately pure with respect to its inputs: it receives
already materialised pandas objects and an audit manifest, validates their
minimum contract, and writes one self-contained HTML file.  It performs no
network access and does not read any experiment artifact.
"""

from __future__ import annotations

from datetime import date
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd


REPORT_SCHEMA = "chronos2.regime-challenger-report.html.v1"

_HOURLY_COLUMNS = frozenset(
    {
        "delivery_start_utc",
        "variant",
        "baseline_q10",
        "baseline_q50",
        "baseline_q90",
        "challenger_q10",
        "challenger_q50",
        "challenger_q90",
        "shock_probability",
        "shock_premium",
        "regime_label",
    }
)
_METRIC_COLUMNS = frozenset(
    {
        "scope",
        "variant",
        "n_hours",
        "baseline_mae",
        "challenger_mae",
    }
)
_DAILY_COLUMNS = frozenset(
    {
        "local_date",
        "variant",
        "baseline_mae",
        "challenger_mae",
    }
)
_FEATURE_COLUMNS = frozenset({"feature", "importance"})
_PRICE_COLUMNS = (
    "baseline_q10",
    "baseline_q50",
    "baseline_q90",
    "challenger_q10",
    "challenger_q50",
    "challenger_q90",
)


class RegimeChallengerReportError(ValueError):
    """Raised when inputs cannot support an honest regime report."""


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegimeChallengerReportError(f"{name} doit être un objet.")
    return value


def _require_frame(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise RegimeChallengerReportError(f"{name} doit être un DataFrame.")
    missing = sorted(required.difference(value.columns))
    if missing:
        raise RegimeChallengerReportError(
            f"{name} incomplet; colonnes absentes: {missing}."
        )
    return value.copy(deep=True)


def _required_text(value: Any, *, name: str) -> str:
    if value is None:
        raise RegimeChallengerReportError(f"{name} doit être renseigné.")
    text = str(value).strip()
    if not text:
        raise RegimeChallengerReportError(f"{name} doit être renseigné.")
    return text


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise RegimeChallengerReportError(f"{name} doit être booléen.")
    return bool(value)


def _canonical_day(value: Any, *, name: str) -> str:
    if isinstance(value, pd.Timestamp):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    else:
        text = str(value)
        try:
            parsed = date.fromisoformat(text)
        except (TypeError, ValueError) as exc:
            raise RegimeChallengerReportError(
                f"{name} doit être une date ISO."
            ) from exc
        if parsed.isoformat() != text:
            raise RegimeChallengerReportError(
                f"{name} doit être une date ISO canonique."
            )
    return parsed.isoformat()


def _numeric_series(
    frame: pd.DataFrame,
    column: str,
    *,
    frame_name: str,
    allow_nan: bool = False,
) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise RegimeChallengerReportError(
            f"{frame_name}.{column} doit être numérique."
        ) from exc
    array = values.to_numpy(dtype=float)
    if np.isinf(array).any() or (not allow_nan and np.isnan(array).any()):
        qualifier = "finies ou manquantes" if allow_nan else "finies"
        raise RegimeChallengerReportError(
            f"{frame_name}.{column} doit contenir des valeurs {qualifier}."
        )
    return values


def _integer_series(
    frame: pd.DataFrame,
    column: str,
    *,
    frame_name: str,
) -> pd.Series:
    values = _numeric_series(frame, column, frame_name=frame_name)
    array = values.to_numpy(dtype=float)
    if not np.equal(array, np.floor(array)).all() or (array < 0).any():
        raise RegimeChallengerReportError(
            f"{frame_name}.{column} doit contenir des entiers >= 0."
        )
    return values.astype(int)


def _aware_utc_index(values: pd.Series, *, name: str) -> pd.DatetimeIndex:
    timestamps: list[pd.Timestamp] = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RegimeChallengerReportError(
                f"{name} contient un timestamp invalide."
            ) from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise RegimeChallengerReportError(
                f"{name} doit être timezone-aware."
            )
        timestamps.append(timestamp.tz_convert("UTC"))
    return pd.DatetimeIndex(timestamps, name="delivery_start_utc")


def _manifest_metadata(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str, str, str | None]:
    raw = dict(_require_mapping(manifest, name="manifest"))
    zone = _required_text(raw.get("zone"), name="manifest.zone").upper()
    timezone = _required_text(
        raw.get("timezone", "Europe/Paris"),
        name="manifest.timezone",
    )
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise RegimeChallengerReportError(
            "manifest.timezone est inconnue."
        ) from exc
    production_eligible = _strict_bool(
        raw.get("production_eligible"),
        name="manifest.production_eligible",
    )
    if production_eligible:
        raise RegimeChallengerReportError(
            "Le rapport de challenger exige production_eligible=false."
        )
    status_value = next(
        (
            raw[key]
            for key in ("forecast_status", "status", "run_type")
            if raw.get(key) not in (None, "")
        ),
        None,
    )
    status = _required_text(status_value, name="manifest.forecast_status/status")
    delivery_day: str | None = None
    for key in ("delivery_day_local", "delivery_day", "target_day"):
        if raw.get(key) not in (None, ""):
            delivery_day = _canonical_day(
                raw[key],
                name=f"manifest.{key}",
            )
            break
    return raw, zone, timezone, status, delivery_day


def _validate_hourly(
    hourly: pd.DataFrame,
    *,
    timezone: str,
    delivery_day: str | None,
) -> pd.DataFrame:
    frame = _require_frame(
        hourly,
        name="hourly",
        required=_HOURLY_COLUMNS,
    )
    if frame.empty:
        raise RegimeChallengerReportError("hourly ne peut pas être vide.")

    frame["delivery_start_utc"] = _aware_utc_index(
        frame["delivery_start_utc"],
        name="hourly.delivery_start_utc",
    )
    for column in ("variant", "regime_label"):
        frame[column] = [
            _required_text(value, name=f"hourly.{column}")
            for value in frame[column]
        ]
    for column in _PRICE_COLUMNS + ("shock_probability", "shock_premium"):
        frame[column] = _numeric_series(
            frame,
            column,
            frame_name="hourly",
        )
    probability = frame["shock_probability"].to_numpy(dtype=float)
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise RegimeChallengerReportError(
            "hourly.shock_probability doit rester dans [0, 1]."
        )
    for prefix in ("baseline", "challenger"):
        q10 = frame[f"{prefix}_q10"].to_numpy(dtype=float)
        q50 = frame[f"{prefix}_q50"].to_numpy(dtype=float)
        q90 = frame[f"{prefix}_q90"].to_numpy(dtype=float)
        if ((q10 > q50 + 1.0e-9) | (q50 > q90 + 1.0e-9)).any():
            raise RegimeChallengerReportError(
                f"hourly contient un croisement de quantiles {prefix}."
            )

    if "actual" in frame:
        frame["actual"] = _numeric_series(
            frame,
            "actual",
            frame_name="hourly",
            allow_nan=True,
        )
    else:
        frame["actual"] = np.nan

    if frame.duplicated(["variant", "delivery_start_utc"]).any():
        raise RegimeChallengerReportError(
            "hourly contient des timestamps dupliqués par variante."
        )
    frame = frame.sort_values(
        ["variant", "delivery_start_utc"],
        kind="stable",
    ).reset_index(drop=True)
    reference: tuple[pd.Timestamp, ...] | None = None
    for variant, part in frame.groupby("variant", sort=False):
        observed = tuple(part["delivery_start_utc"])
        if reference is None:
            reference = observed
        elif observed != reference:
            raise RegimeChallengerReportError(
                "hourly doit avoir exactement la même timeline pour toutes "
                f"les variantes; divergence pour {variant}."
            )

    local = frame["delivery_start_utc"].dt.tz_convert(timezone)
    frame["_local_date"] = local.dt.date.astype(str)
    frame["_local_hour"] = local.dt.hour.astype(int)
    frame["_local_label"] = local.dt.strftime("%d/%m %H:%M")
    if delivery_day is not None and delivery_day not in set(frame["_local_date"]):
        raise RegimeChallengerReportError(
            "Le jour de livraison du manifeste est absent de hourly."
        )
    return frame


def _validate_metrics(
    metrics: pd.DataFrame,
    *,
    variants: set[str],
) -> pd.DataFrame:
    frame = _require_frame(
        metrics,
        name="metrics",
        required=_METRIC_COLUMNS,
    )
    if frame.empty:
        return frame
    for column in ("scope", "variant"):
        frame[column] = [
            _required_text(value, name=f"metrics.{column}")
            for value in frame[column]
        ]
    unknown = sorted(
        value
        for value in set(frame["variant"])
        if value not in variants and value.casefold() not in {"all", "aggregate"}
    )
    if unknown:
        raise RegimeChallengerReportError(
            f"metrics contient des variantes inconnues: {unknown}."
        )
    if frame.duplicated(["scope", "variant"]).any():
        raise RegimeChallengerReportError(
            "metrics contient des doublons scope/variant."
        )
    frame["n_hours"] = _integer_series(
        frame,
        "n_hours",
        frame_name="metrics",
    )
    for column in ("baseline_mae", "challenger_mae"):
        frame[column] = _numeric_series(
            frame,
            column,
            frame_name="metrics",
            allow_nan=True,
        )
        invalid = frame[column].isna() & frame["n_hours"].gt(0)
        if invalid.any():
            raise RegimeChallengerReportError(
                f"metrics.{column} ne peut être manquant quand n_hours > 0."
            )
    if "mae_delta" not in frame:
        frame["mae_delta"] = frame["challenger_mae"] - frame["baseline_mae"]
    return frame


def _validate_daily(
    daily: pd.DataFrame,
    *,
    variants: set[str],
) -> pd.DataFrame:
    frame = _require_frame(
        daily,
        name="daily",
        required=_DAILY_COLUMNS,
    )
    if frame.empty:
        return frame
    frame["local_date"] = [
        _canonical_day(value, name="daily.local_date")
        for value in frame["local_date"]
    ]
    frame["variant"] = [
        _required_text(value, name="daily.variant")
        for value in frame["variant"]
    ]
    unknown = sorted(set(frame["variant"]).difference(variants))
    if unknown:
        raise RegimeChallengerReportError(
            f"daily contient des variantes inconnues: {unknown}."
        )
    if frame.duplicated(["local_date", "variant"]).any():
        raise RegimeChallengerReportError(
            "daily contient des doublons local_date/variant."
        )
    for column in ("baseline_mae", "challenger_mae"):
        frame[column] = _numeric_series(
            frame,
            column,
            frame_name="daily",
            allow_nan=True,
        )
    if "n_hours" in frame:
        frame["n_hours"] = _integer_series(
            frame,
            "n_hours",
            frame_name="daily",
        )
    if "mae_delta" not in frame:
        frame["mae_delta"] = frame["challenger_mae"] - frame["baseline_mae"]
    return frame.sort_values(["variant", "local_date"], kind="stable")


def _validate_feature_importance(
    feature_importance: pd.DataFrame,
    *,
    variants: set[str],
) -> pd.DataFrame:
    frame = _require_frame(
        feature_importance,
        name="feature_importance",
        required=_FEATURE_COLUMNS,
    )
    if frame.empty:
        return frame
    frame["feature"] = [
        _required_text(value, name="feature_importance.feature")
        for value in frame["feature"]
    ]
    frame["importance"] = _numeric_series(
        frame,
        "importance",
        frame_name="feature_importance",
    )
    key = ["feature"]
    if "variant" in frame:
        frame["variant"] = [
            _required_text(value, name="feature_importance.variant")
            for value in frame["variant"]
        ]
        unknown = sorted(set(frame["variant"]).difference(variants))
        if unknown:
            raise RegimeChallengerReportError(
                "feature_importance contient des variantes inconnues: "
                f"{unknown}."
            )
        key.append("variant")
    if frame.duplicated(key).any():
        raise RegimeChallengerReportError(
            "feature_importance contient des clés dupliquées."
        )
    return frame


def _plain_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "Oui" if bool(value) else "Non"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "N/A"
        if math.isinf(number):
            return "N/A"
        return f"{number:,.3f}".replace(",", " ").replace(".", ",")
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", " ")
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return "N/A"
    return str(value)


def _safe(value: Any) -> str:
    return escape(_plain_value(value), quote=True)


def _fmt_number(
    value: Any,
    *,
    digits: int = 2,
    suffix: str = "",
) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    rendered = f"{number:,.{digits}f}".replace(",", " ").replace(".", ",")
    return rendered + suffix


def _label(column: str) -> str:
    labels = {
        "scope": "Périmètre",
        "variant": "Variante",
        "n_hours": "Heures",
        "n_days": "Jours",
        "baseline_mae": "MAE baseline",
        "challenger_mae": "MAE challenger",
        "mae_delta": "Δ MAE",
        "local_date": "Jour local",
        "feature": "Feature",
        "importance": "Importance",
        "regime_label": "Régime",
        "shock_probability": "Probabilité de choc",
        "shock_premium": "Prime de choc",
    }
    return labels.get(column, column.replace("_", " ").strip().capitalize())


def _table_html(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    caption: str,
    empty_message: str = "Aucune donnée disponible.",
    table_id: str | None = None,
) -> str:
    if frame.empty:
        return f'<p class="empty">{escape(empty_message)}</p>'
    selected = list(columns) if columns is not None else list(frame.columns)
    selected = [column for column in selected if column in frame.columns]
    identifier = f' id="{escape(table_id, quote=True)}"' if table_id else ""
    header = "".join(f"<th>{escape(_label(column))}</th>" for column in selected)
    rows: list[str] = []
    for record in frame.loc[:, selected].itertuples(index=False, name=None):
        rows.append(
            "<tr>"
            + "".join(f"<td>{_safe(value)}</td>" for value in record)
            + "</tr>"
        )
    return (
        '<div class="table-wrap"><table'
        + identifier
        + f"><caption>{escape(caption)}</caption><thead><tr>{header}</tr></thead>"
        + f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _flatten_mapping(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
    depth: int = 0,
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for raw_key in sorted(value, key=lambda item: str(item)):
        key = str(raw_key)
        name = f"{prefix}.{key}" if prefix else key
        item = value[raw_key]
        if isinstance(item, Mapping) and depth < 4:
            rows.extend(
                _flatten_mapping(item, prefix=name, depth=depth + 1)
            )
        else:
            rows.append((name, item))
    return rows


def _mapping_table(
    value: Mapping[str, Any] | None,
    *,
    caption: str,
    empty_message: str,
) -> str:
    if not value:
        return f'<p class="empty">{escape(empty_message)}</p>'
    rows = _flatten_mapping(value)
    body = "".join(
        f"<tr><th>{escape(name)}</th><td>{_safe(item)}</td></tr>"
        for name, item in rows
    )
    return (
        '<div class="table-wrap"><table class="kv"><caption>'
        + escape(caption)
        + "</caption><tbody>"
        + body
        + "</tbody></table></div>"
    )


def _metric_card(
    label: str,
    value: str,
    *,
    note: str = "",
    css: str = "",
) -> str:
    return (
        f'<div class="kpi {escape(css, quote=True)}">'
        f'<div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}</div>'
        f'<div class="kpi-note">{escape(note)}</div></div>'
    )


def _scale(
    values: np.ndarray,
    *,
    low: float,
    high: float,
    start: float,
    span: float,
) -> np.ndarray:
    if high <= low:
        return np.full(len(values), start + span / 2.0)
    return start + (high - values) / (high - low) * span


def _line_points(
    x: np.ndarray,
    y: np.ndarray,
) -> str:
    return " ".join(
        f"{float(x_value):.2f},{float(y_value):.2f}"
        for x_value, y_value in zip(x, y, strict=True)
        if math.isfinite(float(y_value))
    )


def _price_curve_svg(
    frame: pd.DataFrame,
    *,
    variant: str,
    chart_index: int,
) -> str:
    part = frame.sort_values("delivery_start_utc", kind="stable")
    n = len(part)
    width = 1080.0
    height = 420.0
    left, right, top, bottom = 72.0, 26.0, 36.0, 66.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    x = (
        np.linspace(left, left + plot_width, n)
        if n > 1
        else np.asarray([left + plot_width / 2.0])
    )
    candidates = [
        part[column].to_numpy(dtype=float)
        for column in _PRICE_COLUMNS
    ]
    actual = part["actual"].to_numpy(dtype=float)
    if np.isfinite(actual).any():
        candidates.append(actual[np.isfinite(actual)])
    values = np.concatenate(candidates)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    padding = max(1.0, (high - low) * 0.08)
    low -= padding
    high += padding

    scaled = {
        column: _scale(
            part[column].to_numpy(dtype=float),
            low=low,
            high=high,
            start=top,
            span=plot_height,
        )
        for column in _PRICE_COLUMNS
    }
    actual_y = _scale(
        actual,
        low=low,
        high=high,
        start=top,
        span=plot_height,
    )
    baseline_band = np.concatenate(
        (
            np.column_stack((x, scaled["baseline_q10"])),
            np.column_stack((x[::-1], scaled["baseline_q90"][::-1])),
        )
    )
    challenger_band = np.concatenate(
        (
            np.column_stack((x, scaled["challenger_q10"])),
            np.column_stack((x[::-1], scaled["challenger_q90"][::-1])),
        )
    )
    baseline_polygon = " ".join(
        f"{row[0]:.2f},{row[1]:.2f}" for row in baseline_band
    )
    challenger_polygon = " ".join(
        f"{row[0]:.2f},{row[1]:.2f}" for row in challenger_band
    )
    grid: list[str] = []
    for ratio in np.linspace(0.0, 1.0, 5):
        y = top + ratio * plot_height
        value = high - ratio * (high - low)
        grid.append(
            f'<line class="grid-line" x1="{left:.1f}" y1="{y:.1f}" '
            f'x2="{left + plot_width:.1f}" y2="{y:.1f}"/>'
            f'<text class="axis-label" x="{left - 10:.1f}" y="{y + 4:.1f}" '
            f'text-anchor="end">{escape(_fmt_number(value, digits=0))}</text>'
        )
    tick_positions = sorted(
        set(np.linspace(0, max(n - 1, 0), min(n, 9), dtype=int).tolist())
    )
    ticks = "".join(
        f'<text class="axis-label" x="{x[position]:.1f}" '
        f'y="{height - 30:.1f}" text-anchor="middle">'
        f'{escape(str(part.iloc[position]["_local_label"]))}</text>'
        for position in tick_positions
    )
    actual_line = ""
    if np.isfinite(actual).any():
        actual_line = (
            '<polyline class="line actual" points="'
            + _line_points(x, actual_y)
            + '"/>'
        )
    title_id = f"price-chart-title-{chart_index}"
    description_id = f"price-chart-description-{chart_index}"
    return f"""
    <div class="chart-scroll">
      <svg class="chart" viewBox="0 0 {width:.0f} {height:.0f}"
        role="img" aria-labelledby="{title_id} {description_id}">
        <title id="{title_id}">Courbes de prix — {escape(variant)}</title>
        <desc id="{description_id}">Quantiles baseline et challenger, médianes
          et prix réalisé lorsqu'il est disponible.</desc>
        {''.join(grid)}
        <line class="axis-line" x1="{left:.1f}" y1="{top + plot_height:.1f}"
          x2="{left + plot_width:.1f}" y2="{top + plot_height:.1f}"/>
        <polygon class="band baseline-band" points="{baseline_polygon}"/>
        <polygon class="band challenger-band" points="{challenger_polygon}"/>
        <polyline class="line baseline" points="{_line_points(x, scaled["baseline_q50"])}"/>
        <polyline class="line challenger" points="{_line_points(x, scaled["challenger_q50"])}"/>
        {actual_line}
        {ticks}
        <text class="axis-title" x="18" y="{top + plot_height / 2:.1f}"
          transform="rotate(-90 18 {top + plot_height / 2:.1f})"
          text-anchor="middle">EUR/MWh</text>
        <g class="legend" transform="translate({left:.1f},16)">
          <line class="line actual" x1="0" y1="0" x2="25" y2="0"/>
          <text x="31" y="4">Réalisé</text>
          <line class="line baseline" x1="110" y1="0" x2="135" y2="0"/>
          <text x="141" y="4">Baseline Q50</text>
          <line class="line challenger" x1="270" y1="0" x2="295" y2="0"/>
          <text x="301" y="4">Challenger Q50</text>
        </g>
      </svg>
    </div>
    """


def _daily_svg(
    frame: pd.DataFrame,
    *,
    variant: str,
    chart_index: int,
) -> str:
    if frame.empty:
        return '<p class="empty">Aucune performance quotidienne disponible.</p>'
    part = frame.sort_values("local_date", kind="stable")
    finite = (
        part["baseline_mae"].notna()
        & part["challenger_mae"].notna()
    )
    part = part.loc[finite]
    if part.empty:
        return '<p class="empty">Aucune MAE quotidienne réalisée disponible.</p>'
    width = 1080.0
    height = 330.0
    left, right, top, bottom = 68.0, 24.0, 30.0, 64.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    n = len(part)
    x = (
        np.linspace(left, left + plot_width, n)
        if n > 1
        else np.asarray([left + plot_width / 2.0])
    )
    baseline = part["baseline_mae"].to_numpy(dtype=float)
    challenger = part["challenger_mae"].to_numpy(dtype=float)
    low = 0.0
    high = max(float(np.max(baseline)), float(np.max(challenger)), 1.0) * 1.08
    baseline_y = _scale(
        baseline,
        low=low,
        high=high,
        start=top,
        span=plot_height,
    )
    challenger_y = _scale(
        challenger,
        low=low,
        high=high,
        start=top,
        span=plot_height,
    )
    grid: list[str] = []
    for ratio in np.linspace(0.0, 1.0, 4):
        y = top + ratio * plot_height
        value = high - ratio * high
        grid.append(
            f'<line class="grid-line" x1="{left:.1f}" y1="{y:.1f}" '
            f'x2="{left + plot_width:.1f}" y2="{y:.1f}"/>'
            f'<text class="axis-label" x="{left - 9:.1f}" y="{y + 4:.1f}" '
            f'text-anchor="end">{escape(_fmt_number(value, digits=1))}</text>'
        )
    tick_positions = sorted(
        set(np.linspace(0, max(n - 1, 0), min(n, 8), dtype=int).tolist())
    )
    ticks = "".join(
        f'<text class="axis-label" x="{x[position]:.1f}" '
        f'y="{height - 30:.1f}" text-anchor="middle">'
        f'{escape(str(part.iloc[position]["local_date"]))}</text>'
        for position in tick_positions
    )
    title_id = f"daily-chart-title-{chart_index}"
    return f"""
    <div class="chart-scroll">
      <svg class="chart daily-chart" viewBox="0 0 {width:.0f} {height:.0f}"
        role="img" aria-labelledby="{title_id}">
        <title id="{title_id}">MAE quotidienne — {escape(variant)}</title>
        {''.join(grid)}
        <polyline class="line baseline" points="{_line_points(x, baseline_y)}"/>
        <polyline class="line challenger" points="{_line_points(x, challenger_y)}"/>
        {ticks}
        <text class="axis-title" x="17" y="{top + plot_height / 2:.1f}"
          transform="rotate(-90 17 {top + plot_height / 2:.1f})"
          text-anchor="middle">MAE</text>
      </svg>
    </div>
    """


def _feature_svg(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<p class="empty">Aucune importance de feature disponible.</p>'
    ordered = frame.assign(_absolute=frame["importance"].abs()).sort_values(
        "_absolute",
        ascending=False,
        kind="stable",
    ).head(20)
    width = 960.0
    row_height = 31.0
    top, bottom = 26.0, 28.0
    label_width = 245.0
    value_width = 90.0
    plot_width = width - label_width - value_width
    height = top + bottom + row_height * len(ordered)
    values = ordered["importance"].to_numpy(dtype=float)
    minimum = min(float(np.min(values)), 0.0)
    maximum = max(float(np.max(values)), 0.0)
    extent = max(abs(minimum), abs(maximum), 1.0e-12)
    zero_x = label_width + plot_width / 2.0
    half_width = plot_width / 2.0
    bars: list[str] = []
    for index, (_, row) in enumerate(ordered.iterrows()):
        value = float(row["importance"])
        y = top + index * row_height
        length = abs(value) / extent * half_width
        x = zero_x if value >= 0 else zero_x - length
        css = "positive" if value >= 0 else "negative"
        feature_label = str(row["feature"])
        if "variant" in ordered:
            feature_label += f" · {row['variant']}"
        bars.append(
            f'<text class="feature-label" x="{label_width - 10:.1f}" '
            f'y="{y + 19:.1f}" text-anchor="end">{escape(feature_label)}</text>'
            f'<rect class="feature-bar {css}" x="{x:.2f}" y="{y + 5:.1f}" '
            f'width="{length:.2f}" height="18" rx="3"/>'
            f'<text class="feature-value" x="{width - value_width + 8:.1f}" '
            f'y="{y + 19:.1f}">{escape(_fmt_number(value, digits=4))}</text>'
        )
    return f"""
    <div class="chart-scroll">
      <svg class="feature-chart" viewBox="0 0 {width:.0f} {height:.0f}"
        role="img" aria-labelledby="feature-chart-title">
        <title id="feature-chart-title">Importance des features</title>
        <line class="axis-line" x1="{zero_x:.1f}" y1="{top - 8:.1f}"
          x2="{zero_x:.1f}" y2="{height - bottom + 5:.1f}"/>
        {''.join(bars)}
      </svg>
    </div>
    """


def _scope_group(value: str) -> str:
    normalized = (
        value.casefold()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
    )
    if normalized in {"all", "overall", "global", "aggregate"} or "global" in normalized:
        return "global"
    if "solar" in normalized or "solaire" in normalized:
        return "solar"
    if any(
        token in normalized
        for token in ("tail", "spike", "shock", "choc", "extreme")
    ):
        return "tail"
    if "regime" in normalized:
        return "regime"
    return "other"


def _metric_sections(metrics: pd.DataFrame) -> str:
    groups = (
        ("global", "Métriques globales", "Aucune métrique globale fournie."),
        ("solar", "Heures solaires", "Aucune métrique solaire fournie."),
        (
            "tail",
            "Queue de distribution / spikes",
            "Aucune métrique de tail fournie.",
        ),
        ("regime", "Métriques par régime", "Aucune métrique de régime fournie."),
    )
    columns = list(metrics.columns)
    preferred = [
        "scope",
        "variant",
        "n_hours",
        "baseline_mae",
        "challenger_mae",
        "mae_delta",
    ]
    ordered = preferred + [column for column in columns if column not in preferred]
    rendered: list[str] = []
    assigned = pd.Series(False, index=metrics.index, dtype=bool)
    for key, title, empty_message in groups:
        mask = metrics["scope"].map(_scope_group).eq(key) if not metrics.empty else assigned
        assigned |= mask
        rendered.append(
            f'<article class="subcard"><h3>{escape(title)}</h3>'
            + _table_html(
                metrics.loc[mask],
                columns=ordered,
                caption=title,
                empty_message=empty_message,
            )
            + "</article>"
        )
    if not metrics.empty and (~assigned).any():
        rendered.append(
            '<article class="subcard"><h3>Autres périmètres</h3>'
            + _table_html(
                metrics.loc[~assigned],
                columns=ordered,
                caption="Autres métriques",
            )
            + "</article>"
        )
    return "".join(rendered)


def _diagnostic_cards(
    hourly: pd.DataFrame,
    *,
    diagnostic_day: str,
) -> str:
    day = hourly.loc[hourly["_local_date"].eq(diagnostic_day)]
    cards: list[str] = []
    for variant, part in day.groupby("variant", sort=False):
        probability_index = part["shock_probability"].idxmax()
        peak_probability = float(part.loc[probability_index, "shock_probability"])
        peak_hour = str(part.loc[probability_index, "_local_label"])
        premium = float(part["shock_premium"].max())
        solar = part.loc[part["_local_hour"].between(10, 16)]
        solar_peak = (
            float(solar["challenger_q50"].max()) if not solar.empty else math.nan
        )
        regime_counts = part["regime_label"].value_counts()
        dominant_regime = str(regime_counts.index[0])
        actual_available = part["actual"].notna()
        actual_peak = (
            float(part.loc[actual_available, "actual"].max())
            if actual_available.any()
            else math.nan
        )
        cards.append(
            f'<article class="variant-diagnostic"><h3>{escape(str(variant))}</h3>'
            '<div class="grid">'
            + _metric_card(
                "Pic Q50 challenger",
                _fmt_number(part["challenger_q50"].max(), suffix=" EUR/MWh"),
            )
            + _metric_card(
                "Pic solaire Q50",
                _fmt_number(solar_peak, suffix=" EUR/MWh"),
                note="10:00–16:59 locales",
            )
            + _metric_card(
                "Probabilité de choc max.",
                _fmt_number(100.0 * peak_probability, digits=1, suffix=" %"),
                note=peak_hour,
            )
            + _metric_card(
                "Prime de choc max.",
                _fmt_number(premium, suffix=" EUR/MWh"),
            )
            + _metric_card("Régime dominant", dominant_regime)
            + _metric_card(
                "Pic réalisé",
                _fmt_number(actual_peak, suffix=" EUR/MWh"),
                note=(
                    "Prix réalisé disponible"
                    if actual_available.any()
                    else "Non observé à cet instant"
                ),
            )
            + "</div></article>"
        )
    return "".join(cards)


def _calibration_payload(
    diagnostics: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for source in (diagnostics, manifest):
        for key in ("calibration_gate", "gate", "promotion_gate"):
            value = source.get(key)
            if isinstance(value, Mapping):
                return value
    calibration = diagnostics.get("calibration")
    if isinstance(calibration, Mapping):
        return calibration
    return None


def _gate_banner(gate: Mapping[str, Any] | None) -> str:
    if gate is None:
        return (
            '<div class="banner warning"><strong>Gate non renseignée.</strong> '
            "Le rapport ne conclut ni à une promotion ni à une aptitude "
            "production.</div>"
        )
    raw_status = next(
        (
            gate[key]
            for key in ("passes", "passed", "status", "decision")
            if key in gate
        ),
        None,
    )
    if isinstance(raw_status, (bool, np.bool_)):
        passes = bool(raw_status)
        status = "PASS" if passes else "FAIL"
    else:
        status = _plain_value(raw_status)
        passes = status.casefold() in {"pass", "passed", "true", "promoted"}
    css = "safe" if passes else "warning"
    return (
        f'<div class="banner {css}"><strong>Décision de calibration : '
        f"{escape(status)}.</strong> "
        "Cette décision reste expérimentale et ne modifie aucune production."
        "</div>"
    )


def _first_mapping(
    *values: Any,
) -> Mapping[str, Any] | None:
    for value in values:
        if isinstance(value, Mapping):
            return value
    return None


def _limitations_html(
    diagnostics: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    raw = diagnostics.get("limitations", manifest.get("limitations"))
    values: list[Any]
    if raw is None:
        values = []
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        values = [raw]
    defaults = [
        "Un statut shadow ne constitue pas une autorisation de production.",
        "Les flux et prix réalisés sont endogènes : ils documentent un épisode "
        "mais ne suffisent pas, seuls, à établir une causalité.",
        "Les performances de queue dépendent du faible nombre d'épisodes rares "
        "et doivent être confirmées sur des fenêtres PIT supplémentaires.",
    ]
    items = defaults + [_plain_value(value) for value in values]
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


_CSS = r"""
:root{color-scheme:light;font-family:Inter,"Segoe UI",Arial,sans-serif;
  --bg:#f4f7fb;--surface:#fff;--surface2:#f8fafc;--ink:#132033;
  --muted:#5a687c;--line:#d8e1ec;--blue:#2563eb;--orange:#ea580c;
  --green:#047857;--red:#b91c1c;--amber:#a16207;
  --shadow:0 10px 30px rgba(15,23,42,.07)}
html[data-theme="dark"]{color-scheme:dark;--bg:#0a1220;--surface:#111c2f;
  --surface2:#17243a;--ink:#e9eff8;--muted:#adbbce;--line:#2d3d55;
  --blue:#60a5fa;--orange:#fb923c;--green:#34d399;--red:#fb7185;
  --amber:#fbbf24;--shadow:0 10px 30px rgba(0,0,0,.28)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);
  color:var(--ink)}main{max-width:1500px;margin:auto;padding:28px 30px 64px}
.topbar{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}
.eyebrow{margin:0 0 8px;color:var(--blue);font-size:12px;font-weight:800;
  letter-spacing:.09em;text-transform:uppercase}.topbar h1{margin:0;font-size:clamp(28px,4vw,44px)}
.subtitle{color:var(--muted);max-width:850px;line-height:1.55}.badges{display:flex;gap:8px;
  flex-wrap:wrap;justify-content:flex-end}.badge{display:inline-block;border-radius:999px;
  padding:7px 11px;font-size:12px;font-weight:800;background:var(--surface2);
  border:1px solid var(--line)}.badge.shadow{color:var(--amber)}.badge.safe{color:var(--green)}
.toolbar{position:sticky;top:0;z-index:20;display:flex;gap:8px;flex-wrap:wrap;
  padding:11px 0;background:color-mix(in srgb,var(--bg) 92%,transparent);
  backdrop-filter:blur(8px)}button,.navlink{font:inherit;color:var(--ink);
  background:var(--surface);border:1px solid var(--line);border-radius:999px;
  padding:8px 12px;text-decoration:none;cursor:pointer;font-size:13px}
.card,.subcard,.variant-diagnostic{background:var(--surface);border:1px solid var(--line);
  border-radius:15px;box-shadow:var(--shadow)}.card{padding:22px;margin-top:18px}
.subcard,.variant-diagnostic{padding:17px;margin-top:13px}.card h2{margin:0 0 8px;
  font-size:23px}.subcard h3,.variant-diagnostic h3{margin:0 0 12px}
.section-lead{color:var(--muted);line-height:1.55;margin-top:4px}
.banner{border-radius:11px;padding:14px 16px;margin-top:14px;line-height:1.5}
.banner.warning{background:color-mix(in srgb,var(--amber) 11%,var(--surface));
  border:1px solid color-mix(in srgb,var(--amber) 38%,var(--line));color:var(--amber)}
.banner.safe{background:color-mix(in srgb,var(--green) 10%,var(--surface));
  border:1px solid color-mix(in srgb,var(--green) 36%,var(--line));color:var(--green)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));
  gap:11px}.kpi{background:var(--surface2);border:1px solid var(--line);
  border-radius:11px;padding:13px;min-width:0}.kpi-label{font-size:12px;
  color:var(--muted);font-weight:700}.kpi-value{font-size:21px;font-weight:800;
  margin:7px 0 3px;overflow-wrap:anywhere}.kpi-note{font-size:11px;color:var(--muted)}
.variant-card{border-top:1px solid var(--line);padding-top:18px;margin-top:20px}
.variant-card:first-of-type{border-top:0}.variant-card h3{margin:0 0 10px}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:12px}caption{text-align:left;
  padding:9px 10px;color:var(--muted);font-weight:700;background:var(--surface2)}
th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;
  white-space:nowrap}thead th{position:sticky;top:0;background:var(--surface2);
  z-index:1;color:var(--muted);font-size:11px;text-transform:uppercase}
th:first-child,td:first-child{text-align:left}.kv th{width:36%;text-transform:none;
  background:var(--surface2);color:var(--muted)}.kv td{text-align:left;white-space:normal;
  overflow-wrap:anywhere}.empty{color:var(--muted);font-style:italic;padding:12px 0}
.chart-scroll{overflow:auto;border:1px solid var(--line);border-radius:11px;
  background:var(--surface2)}.chart,.feature-chart{display:block;width:100%;
  min-width:820px;color:var(--ink)}.grid-line{stroke:var(--line);stroke-width:1}
.axis-line{stroke:var(--muted);stroke-width:1}.axis-label,.axis-title{fill:var(--muted);
  font-size:11px}.axis-title{font-weight:700}.band{stroke:none}
.baseline-band{fill:var(--blue);fill-opacity:.11}.challenger-band{fill:var(--orange);
  fill-opacity:.13}.line{fill:none;stroke-width:2.4;stroke-linejoin:round;
  stroke-linecap:round}.line.baseline{stroke:var(--blue)}.line.challenger{stroke:var(--orange)}
.line.actual{stroke:var(--ink);stroke-width:2}.legend text{fill:var(--ink);font-size:11px;
  font-weight:700}.feature-label,.feature-value{fill:var(--ink);font-size:11px}
.feature-bar.positive{fill:var(--blue)}.feature-bar.negative{fill:var(--orange)}
.two-columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.foot{margin-top:22px;padding-top:14px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12px}
@media(max-width:900px){main{padding:18px 12px 45px}.topbar{display:block}
  .badges{justify-content:flex-start;margin-top:14px}.toolbar{position:static}
  .two-columns{grid-template-columns:1fr}.chart,.feature-chart{min-width:760px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{.toolbar,#theme-toggle,#print-report{display:none!important}body{background:#fff}
  main{max-width:none;padding:0}.card,.subcard,.variant-diagnostic{box-shadow:none;
  break-inside:avoid}.table-wrap,.chart-scroll{overflow:visible}.chart,.feature-chart{
  min-width:0}.topbar{break-after:avoid}}
"""


_SCRIPT = r"""
(function(){
  var root=document.documentElement;
  var theme="light";
  try{
    var saved=localStorage.getItem("chronos2-report-theme");
    if(saved==="light"||saved==="dark"){theme=saved;}
  }catch(_error){}
  root.dataset.theme=theme;
  function sync(){
    var button=document.getElementById("theme-toggle");
    if(!button){return;}
    var dark=root.dataset.theme==="dark";
    button.setAttribute("aria-pressed",dark?"true":"false");
    button.setAttribute("aria-label",dark?"Activer le mode clair":"Activer le mode nuit");
    button.textContent=dark?"☀ Mode clair":"☾ Mode nuit";
  }
  document.addEventListener("DOMContentLoaded",function(){
    sync();
    var button=document.getElementById("theme-toggle");
    if(button){
      button.addEventListener("click",function(){
        root.dataset.theme=root.dataset.theme==="dark"?"light":"dark";
        try{localStorage.setItem("chronos2-report-theme",root.dataset.theme);}
        catch(_error){}
        sync();
      });
    }
    var printer=document.getElementById("print-report");
    if(printer){printer.addEventListener("click",function(){window.print();});}
  });
})();
"""


def write_regime_challenger_report(
    output_path: str | Path,
    *,
    manifest: Mapping[str, Any],
    hourly: pd.DataFrame,
    metrics: pd.DataFrame,
    daily: pd.DataFrame,
    feature_importance: pd.DataFrame,
    diagnostics: Mapping[str, Any],
) -> Path:
    """Validate inputs and write one self-contained UTF-8 HTML report."""

    manifest_value, zone, timezone, status, manifest_day = _manifest_metadata(
        manifest
    )
    diagnostics_value = dict(_require_mapping(diagnostics, name="diagnostics"))
    hourly_value = _validate_hourly(
        hourly,
        timezone=timezone,
        delivery_day=manifest_day,
    )
    variants = set(hourly_value["variant"])
    metrics_value = _validate_metrics(metrics, variants=variants)
    daily_value = _validate_daily(daily, variants=variants)
    features_value = _validate_feature_importance(
        feature_importance,
        variants=variants,
    )
    diagnostic_day = manifest_day or str(hourly_value["_local_date"].max())
    gate = _calibration_payload(diagnostics_value, manifest_value)
    pit = _first_mapping(
        diagnostics_value.get("pit"),
        diagnostics_value.get("pit_audit"),
        manifest_value.get("pit"),
        manifest_value.get("pit_audit"),
    )
    provenance = _first_mapping(
        diagnostics_value.get("provenance"),
        manifest_value.get("provenance"),
        manifest_value.get("sources"),
    )
    day_details = _first_mapping(
        diagnostics_value.get("day"),
        diagnostics_value.get("day_diagnostic"),
        diagnostics_value.get("diagnostic_du_jour"),
    )

    variant_sections: list[str] = []
    for chart_index, (variant, part) in enumerate(
        hourly_value.groupby("variant", sort=False),
        start=1,
    ):
        variant_sections.append(
            f'<article class="variant-card" id="variant-{chart_index}">'
            f"<h3>{escape(str(variant))}</h3>"
            f"{_price_curve_svg(part, variant=str(variant), chart_index=chart_index)}"
            "</article>"
        )

    daily_sections: list[str] = []
    for chart_index, variant in enumerate(sorted(variants), start=1):
        part = daily_value.loc[daily_value["variant"].eq(variant)]
        daily_sections.append(
            f'<article class="variant-card"><h3>{escape(variant)}</h3>'
            + _daily_svg(part, variant=variant, chart_index=chart_index)
            + _table_html(
                part,
                caption=f"Performance quotidienne — {variant}",
                empty_message="Aucune performance quotidienne fournie.",
            )
            + "</article>"
        )

    actual_hours = int(hourly_value["actual"].notna().sum())
    shadow_class = "shadow" if "shadow" in status.casefold() else "safe"
    status_note = (
        "Les prix réalisés sont absents ou partiels : les performances "
        "associées doivent être interprétées uniquement sur les heures observées."
        if actual_hours < len(hourly_value)
        else "Les prix réalisés sont présents dans l'objet fourni au renderer."
    )
    report = f"""<!doctype html>
<html lang="fr" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; connect-src 'none'; object-src 'none'; base-uri 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:">
  <meta name="generator" content="{REPORT_SCHEMA}">
  <title>Challenger de régime {escape(zone)} · {escape(diagnostic_day)}</title>
  <script>{_SCRIPT}</script>
  <style>{_CSS}</style>
</head>
<body><main>
  <header class="topbar" id="status">
    <div><p class="eyebrow">Chronos-2 · challenger expérimental</p>
      <h1>Prix spot {escape(zone)} · régime D / D‑1</h1>
      <p class="subtitle">Diagnostic du {escape(diagnostic_day)} · cutoff et
        provenance documentés dans l'audit. Ce rapport ne modifie aucun modèle
        ni aucune archive de production.</p></div>
    <div class="badges">
      <span class="badge {shadow_class}">{escape(status.upper())}</span>
      <span class="badge shadow">NON PRODUCTION</span>
      <span class="badge">{len(variants)} variante(s)</span>
    </div>
  </header>
  <nav class="toolbar" aria-label="Sections du rapport">
    <a class="navlink" href="#day-diagnostic">Diagnostic du jour</a>
    <a class="navlink" href="#variant-curves">Courbes</a>
    <a class="navlink" href="#metrics">Métriques</a>
    <a class="navlink" href="#calibration-gate">Gate</a>
    <a class="navlink" href="#daily-performance">Quotidien</a>
    <a class="navlink" href="#feature-importance">Features</a>
    <a class="navlink" href="#audit">Audit</a>
    <button id="theme-toggle" type="button" aria-pressed="false">☾ Mode nuit</button>
    <button id="print-report" type="button">Imprimer / PDF</button>
  </nav>
  <div class="banner warning"><strong>Shadow / non production.</strong>
    {escape(status_note)} Une gate positive ne vaut pas autorisation de
    déploiement.</div>
  <noscript><div class="banner warning">JavaScript sert uniquement au thème
    et au bouton d'impression ; toutes les données restent lisibles sans lui.</div></noscript>

  <section class="card" id="day-diagnostic">
    <h2>Diagnostic du jour</h2>
    <p class="section-lead">Lecture synthétique du niveau, de la fenêtre solaire,
      de la probabilité de choc et du régime attribué.</p>
    {_diagnostic_cards(hourly_value, diagnostic_day=diagnostic_day)}
    <article class="subcard"><h3>Diagnostic fourni par le pipeline</h3>
      {_mapping_table(day_details, caption="Diagnostic du jour",
        empty_message="Aucun diagnostic narratif structuré n'a été fourni.")}
    </article>
  </section>

  <section class="card" id="variant-curves">
    <h2>Courbes par variante</h2>
    <p class="section-lead">Les rubans représentent Q10–Q90. Les lignes
      comparent les médianes baseline et challenger au prix réalisé lorsqu'il
      est disponible.</p>
    {''.join(variant_sections)}
  </section>

  <section class="card" id="metrics">
    <h2>Métriques globales, solaires, tail et régime</h2>
    <p class="section-lead">Les colonnes statistiques additionnelles du pipeline
      sont conservées et rendues sans modifier leur valeur.</p>
    {_metric_sections(metrics_value)}
  </section>

  <section class="card" id="calibration-gate">
    <h2>Calibration et gate</h2>
    {_gate_banner(gate)}
    <article class="subcard">
      {_mapping_table(gate, caption="Critères de calibration et de gate",
        empty_message="Aucun détail de calibration n'a été fourni.")}
    </article>
  </section>

  <section class="card" id="daily-performance">
    <h2>Performance quotidienne</h2>
    <p class="section-lead">MAE appariée par jour et par variante. Les journées
      non réalisées restent explicitement absentes plutôt que reconstruites.</p>
    {''.join(daily_sections)}
  </section>

  <section class="card" id="feature-importance">
    <h2>Importance des features</h2>
    <p class="section-lead">Les vingt contributions absolues les plus fortes
      sont visualisées ; le signe original reste visible.</p>
    {_feature_svg(features_value)}
    {_table_html(features_value, caption="Importance des features",
      empty_message="Aucune importance de feature fournie.")}
  </section>

  <section class="card" id="audit">
    <h2>Audit PIT et provenance</h2>
    <p class="section-lead">Le renderer reproduit les déclarations fournies
      sans effectuer de lecture implicite des sources.</p>
    <div class="two-columns">
      <article class="subcard"><h3>Contrat PIT</h3>
        {_mapping_table(pit, caption="Contrat point-in-time",
          empty_message="Aucun audit PIT structuré n'a été fourni.")}
      </article>
      <article class="subcard"><h3>Provenance</h3>
        {_mapping_table(provenance, caption="Provenance des artefacts",
          empty_message="Aucune provenance structurée n'a été fournie.")}
      </article>
    </div>
    <details class="subcard"><summary>Manifeste complet échappé</summary>
      {_mapping_table(manifest_value, caption="Manifeste",
        empty_message="Manifeste vide.")}
    </details>
  </section>

  <section class="card" id="limitations">
    <h2>Limites d'interprétation</h2>
    {_limitations_html(diagnostics_value, manifest_value)}
  </section>
  <footer class="foot">Schéma : {REPORT_SCHEMA} · HTML autonome UTF‑8 ·
    {len(hourly_value)} lignes horaires · {actual_hours} observations réalisées.</footer>
</main></body></html>
"""

    output = Path(output_path).expanduser().resolve()
    if output.suffix.casefold() != ".html":
        raise RegimeChallengerReportError(
            "output_path doit avoir l'extension .html."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


__all__: Sequence[str] = (
    "REPORT_SCHEMA",
    "RegimeChallengerReportError",
    "write_regime_challenger_report",
)
