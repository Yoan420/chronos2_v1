"""Self-contained report for the prospective residual-load A/B comparison.

The renderer is intentionally pure: it only consumes the already paired rows,
their aggregate metrics and the comparison manifest supplied by the caller.  It
does not know where live archives, historical reports or Statistics files live.
"""

from __future__ import annotations

from datetime import date
from html import escape
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index


REPORT_SCHEMA = "chronos2.residual-load-comparison.html.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ZONE = re.compile(r"^[A-Z][A-Z0-9_-]{1,15}$")

_PAIRED_COLUMNS = frozenset(
    {
        "zone",
        "timezone",
        "delivery_day_local",
        "delivery_start_utc",
        "hours_in_local_day",
        "production_q50",
        "challenger_q50",
        "actual",
        "production_error",
        "challenger_error",
        "production_abs_error",
        "challenger_abs_error",
        "challenger_wins",
        "tie",
    }
)

_METRIC_VALUE_COLUMNS = (
    "production_mae",
    "challenger_mae",
    "mae_delta",
    "production_rmse",
    "challenger_rmse",
    "rmse_delta",
    "production_bias",
    "challenger_bias",
    "absolute_bias_delta",
    "production_smape_pct",
    "challenger_smape_pct",
    "smape_delta_pct",
    "challenger_win_rate",
    "tie_rate",
    "challenger_day_win_rate",
)

_METRIC_COLUMNS = frozenset(
    {"scope", "zone", "n_days", "n_hours", *_METRIC_VALUE_COLUMNS}
)


class ResidualLoadComparisonReportError(ValueError):
    """Raised when inputs cannot support an honest prospective report."""


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidualLoadComparisonReportError(f"{name} doit être un objet.")
    return value


def _require_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ResidualLoadComparisonReportError(f"{name} doit être un entier.")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResidualLoadComparisonReportError(
            f"{name} doit être un entier."
        ) from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResidualLoadComparisonReportError(
            f"{name} doit être un entier."
        ) from exc
    if not math.isfinite(numeric) or numeric != float(number) or number < minimum:
        raise ResidualLoadComparisonReportError(
            f"{name} doit être un entier >= {minimum}."
        )
    return number


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ResidualLoadComparisonReportError(f"{name} doit être booléen.")
    return bool(value)


def _require_day(value: Any, *, name: str) -> str:
    text = str(value)
    try:
        parsed = date.fromisoformat(text)
    except (TypeError, ValueError) as exc:
        raise ResidualLoadComparisonReportError(
            f"{name} doit être une date ISO."
        ) from exc
    if parsed.isoformat() != text:
        raise ResidualLoadComparisonReportError(
            f"{name} doit être une date ISO canonique."
        )
    return text


def _require_sha256(value: Any, *, name: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ResidualLoadComparisonReportError(
            f"{name} doit être un SHA-256 hexadécimal."
        )
    return text


def _aware_utc_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResidualLoadComparisonReportError(
            f"{name} contient un timestamp invalide."
        ) from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ResidualLoadComparisonReportError(
            f"{name} doit être timezone-aware."
        )
    return timestamp.tz_convert("UTC")


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ResidualLoadComparisonReportError(
            f"paired_hourly.{column} doit être numérique."
        ) from exc
    if not np.isfinite(values).all():
        raise ResidualLoadComparisonReportError(
            f"paired_hourly.{column} contient une valeur non finie."
        )
    return values


def _validate_paired_hourly(
    paired_hourly: pd.DataFrame,
    *,
    zones: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(paired_hourly, pd.DataFrame):
        raise ResidualLoadComparisonReportError(
            "paired_hourly doit être un DataFrame."
        )
    missing = sorted(_PAIRED_COLUMNS.difference(paired_hourly.columns))
    if missing:
        raise ResidualLoadComparisonReportError(
            f"paired_hourly incomplet; colonnes absentes: {missing}."
        )
    frame = paired_hourly.copy(deep=True)
    if frame.empty:
        return frame

    frame["zone"] = frame["zone"].map(str)
    unknown_zones = sorted(set(frame["zone"]).difference(zones))
    if unknown_zones:
        raise ResidualLoadComparisonReportError(
            f"paired_hourly contient des zones hors manifeste: {unknown_zones}."
        )
    frame["timezone"] = frame["timezone"].map(str)
    if frame["timezone"].eq("").any():
        raise ResidualLoadComparisonReportError(
            "paired_hourly.timezone contient une valeur vide."
        )
    frame["delivery_day_local"] = [
        _require_day(value, name="paired_hourly.delivery_day_local")
        for value in frame["delivery_day_local"]
    ]
    frame["delivery_start_utc"] = pd.DatetimeIndex(
        [
            _aware_utc_timestamp(value, name="paired_hourly.delivery_start_utc")
            for value in frame["delivery_start_utc"]
        ]
    )
    if frame.duplicated(["zone", "delivery_start_utc"]).any():
        raise ResidualLoadComparisonReportError(
            "paired_hourly contient des timestamps dupliqués par zone."
        )

    hours = _numeric_column(frame, "hours_in_local_day")
    if not np.equal(hours, np.floor(hours)).all():
        raise ResidualLoadComparisonReportError(
            "paired_hourly.hours_in_local_day doit être entier."
        )
    frame["hours_in_local_day"] = hours.astype(int)
    for column in (
        "production_q50",
        "challenger_q50",
        "actual",
        "production_error",
        "challenger_error",
        "production_abs_error",
        "challenger_abs_error",
    ):
        frame[column] = _numeric_column(frame, column)
    for column in ("challenger_wins", "tie"):
        if not frame[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ResidualLoadComparisonReportError(
                f"paired_hourly.{column} doit être booléen."
            )
        frame[column] = frame[column].astype(bool)

    expected_production_error = frame["production_q50"] - frame["actual"]
    expected_challenger_error = frame["challenger_q50"] - frame["actual"]
    checks = (
        ("production_error", expected_production_error),
        ("challenger_error", expected_challenger_error),
        ("production_abs_error", expected_production_error.abs()),
        ("challenger_abs_error", expected_challenger_error.abs()),
    )
    for column, expected in checks:
        if not np.allclose(
            frame[column].to_numpy(float),
            expected.to_numpy(float),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ResidualLoadComparisonReportError(
                f"paired_hourly.{column} est incohérent avec les valeurs Q50."
            )
    expected_wins = (
        frame["challenger_abs_error"] < frame["production_abs_error"]
    )
    expected_ties = (
        frame["challenger_abs_error"] == frame["production_abs_error"]
    )
    if not frame["challenger_wins"].equals(expected_wins):
        raise ResidualLoadComparisonReportError(
            "paired_hourly.challenger_wins est incohérent."
        )
    if not frame["tie"].equals(expected_ties):
        raise ResidualLoadComparisonReportError(
            "paired_hourly.tie est incohérent."
        )

    for (zone, day), part in frame.groupby(
        ["zone", "delivery_day_local"], sort=False
    ):
        timezones = part["timezone"].unique()
        if len(timezones) != 1:
            raise ResidualLoadComparisonReportError(
                f"{zone} {day}: plusieurs timezones dans une paire."
            )
        try:
            expected = local_delivery_day_index(day, timezone=str(timezones[0]))
        except Exception as exc:
            raise ResidualLoadComparisonReportError(
                f"{zone} {day}: timezone ou journée locale invalide."
            ) from exc
        observed = pd.DatetimeIndex(part["delivery_start_utc"]).sort_values()
        if not observed.equals(expected):
            raise ResidualLoadComparisonReportError(
                f"{zone} {day}: timeline différente de la journée locale "
                f"canonique ({len(expected)} h)."
            )
        if not part["hours_in_local_day"].eq(len(expected)).all():
            raise ResidualLoadComparisonReportError(
                f"{zone} {day}: hours_in_local_day incohérent."
            )

    return frame.sort_values(
        ["zone", "delivery_start_utc"], kind="stable"
    ).reset_index(drop=True)


def _computed_metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
    n_hours = len(frame)
    n_days = frame["delivery_day_local"].nunique() if n_hours else 0
    values: dict[str, float | int] = {
        "n_hours": int(n_hours),
        "n_days": int(n_days),
    }
    if not n_hours:
        return values
    actual = frame["actual"].to_numpy(float)
    production_error = frame["production_error"].to_numpy(float)
    challenger_error = frame["challenger_error"].to_numpy(float)
    production_abs = np.abs(production_error)
    challenger_abs = np.abs(challenger_error)

    def smape(forecast: np.ndarray) -> float:
        denominator = np.abs(forecast) + np.abs(actual)
        ratio = np.divide(
            2.0 * np.abs(forecast - actual),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-12,
        )
        return 100.0 * float(np.mean(ratio))

    daily = frame.assign(
        _production_abs=production_abs,
        _challenger_abs=challenger_abs,
    ).groupby(["zone", "delivery_day_local"], sort=False)[
        ["_production_abs", "_challenger_abs"]
    ].mean()
    production_mae = float(np.mean(production_abs))
    challenger_mae = float(np.mean(challenger_abs))
    production_rmse = float(np.sqrt(np.mean(np.square(production_error))))
    challenger_rmse = float(np.sqrt(np.mean(np.square(challenger_error))))
    production_bias = float(np.mean(production_error))
    challenger_bias = float(np.mean(challenger_error))
    production_smape = smape(frame["production_q50"].to_numpy(float))
    challenger_smape = smape(frame["challenger_q50"].to_numpy(float))
    values.update(
        {
            "production_mae": production_mae,
            "challenger_mae": challenger_mae,
            "mae_delta": challenger_mae - production_mae,
            "production_rmse": production_rmse,
            "challenger_rmse": challenger_rmse,
            "rmse_delta": challenger_rmse - production_rmse,
            "production_bias": production_bias,
            "challenger_bias": challenger_bias,
            "absolute_bias_delta": abs(challenger_bias)
            - abs(production_bias),
            "production_smape_pct": production_smape,
            "challenger_smape_pct": challenger_smape,
            "smape_delta_pct": challenger_smape - production_smape,
            "challenger_win_rate": float(
                np.mean(challenger_abs < production_abs)
            ),
            "tie_rate": float(np.mean(challenger_abs == production_abs)),
            "challenger_day_win_rate": float(
                np.mean(daily["_challenger_abs"] < daily["_production_abs"])
            ),
        }
    )
    return values


def _validate_metrics(
    metrics: pd.DataFrame,
    *,
    paired: pd.DataFrame,
    zones: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame):
        raise ResidualLoadComparisonReportError("metrics doit être un DataFrame.")
    missing = sorted(_METRIC_COLUMNS.difference(metrics.columns))
    if missing:
        raise ResidualLoadComparisonReportError(
            f"metrics incomplet; colonnes absentes: {missing}."
        )
    frame = metrics.copy(deep=True)
    frame["zone"] = frame["zone"].map(str)
    frame["scope"] = frame["scope"].map(str)
    if frame["zone"].duplicated().any():
        raise ResidualLoadComparisonReportError("metrics contient une zone dupliquée.")
    expected_zones = {*zones, "ALL"}
    if set(frame["zone"]) != expected_zones:
        raise ResidualLoadComparisonReportError(
            "metrics doit contenir exactement les zones du manifeste et ALL."
        )
    expected_order = [*zones, "ALL"]
    frame = frame.set_index("zone").loc[expected_order].reset_index()

    for _, row in frame.iterrows():
        zone = str(row["zone"])
        expected_scope = "aggregate" if zone == "ALL" else "zone"
        if row["scope"] != expected_scope:
            raise ResidualLoadComparisonReportError(
                f"metrics[{zone}].scope doit valoir {expected_scope}."
            )
        part = paired if zone == "ALL" else paired.loc[paired["zone"] == zone]
        expected = _computed_metric_row(part)
        for count in ("n_hours", "n_days"):
            observed_count = _require_int(
                row[count], name=f"metrics[{zone}].{count}"
            )
            if observed_count != expected[count]:
                raise ResidualLoadComparisonReportError(
                    f"metrics[{zone}].{count} est incohérent avec paired_hourly."
                )
        if not len(part):
            for column in _METRIC_VALUE_COLUMNS:
                if not pd.isna(row[column]):
                    raise ResidualLoadComparisonReportError(
                        f"metrics[{zone}].{column} doit être N/A sans paire scorée."
                    )
            continue
        for column in _METRIC_VALUE_COLUMNS:
            try:
                observed = float(row[column])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ResidualLoadComparisonReportError(
                    f"metrics[{zone}].{column} doit être numérique."
                ) from exc
            if not math.isfinite(observed) or not math.isclose(
                observed,
                float(expected[column]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ResidualLoadComparisonReportError(
                    f"metrics[{zone}].{column} est incohérent avec paired_hourly."
                )
    return frame


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    paired: pd.DataFrame,
    zones: Sequence[str],
) -> dict[str, Any]:
    payload = dict(_require_mapping(manifest, name="manifest"))
    required_identity = {
        "comparison_type": "prospective_paired_ab",
        "treatment": "residual_load_source_chronos2_vs_saturn",
        "downstream_policy": "frozen_current_production_downstream",
    }
    for key, expected in required_identity.items():
        if payload.get(key) != expected:
            raise ResidualLoadComparisonReportError(
                f"manifest.{key}={payload.get(key)!r}, attendu {expected!r}."
            )
    start = _require_day(
        payload.get("start_delivery_day_local"),
        name="manifest.start_delivery_day_local",
    )
    end = _require_day(
        payload.get("end_delivery_day_local"),
        name="manifest.end_delivery_day_local",
    )
    if start > end:
        raise ResidualLoadComparisonReportError(
            "La période du manifeste est inversée."
        )
    n_scored_hours = _require_int(
        payload.get("n_scored_hours"), name="manifest.n_scored_hours"
    )
    if n_scored_hours != len(paired):
        raise ResidualLoadComparisonReportError(
            "manifest.n_scored_hours est incohérent avec paired_hourly."
        )

    pairs_raw = payload.get("pairs")
    if not isinstance(pairs_raw, list):
        raise ResidualLoadComparisonReportError("manifest.pairs doit être une liste.")
    pairs: list[dict[str, Any]] = []
    pair_keys: set[tuple[str, str]] = set()
    scored_keys: set[tuple[str, str]] = set()
    for position, raw in enumerate(pairs_raw):
        item = dict(_require_mapping(raw, name=f"manifest.pairs[{position}]"))
        zone = str(item.get("zone"))
        if zone not in zones:
            raise ResidualLoadComparisonReportError(
                f"manifest.pairs[{position}].zone est hors périmètre."
            )
        day = _require_day(
            item.get("delivery_day_local"),
            name=f"manifest.pairs[{position}].delivery_day_local",
        )
        key = (zone, day)
        if key in pair_keys:
            raise ResidualLoadComparisonReportError(
                f"manifest.pairs contient la paire dupliquée {zone} {day}."
            )
        pair_keys.add(key)
        item["hours_in_local_day"] = _require_int(
            item.get("hours_in_local_day"),
            name=f"manifest.pairs[{position}].hours_in_local_day",
            minimum=23,
        )
        if item["hours_in_local_day"] not in (23, 24, 25):
            raise ResidualLoadComparisonReportError(
                f"manifest.pairs[{position}].hours_in_local_day est invalide."
            )
        item["realized_complete"] = _require_bool(
            item.get("realized_complete"),
            name=f"manifest.pairs[{position}].realized_complete",
        )
        item["scored"] = _require_bool(
            item.get("scored"), name=f"manifest.pairs[{position}].scored"
        )
        if item["scored"] and not item["realized_complete"]:
            raise ResidualLoadComparisonReportError(
                f"manifest.pairs[{position}] est scorée sans prix réels complets."
            )
        if item["scored"]:
            scored_keys.add(key)
        pairs.append(item)
    observed_scored_keys = {
        (str(zone), str(day))
        for zone, day in paired[["zone", "delivery_day_local"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if scored_keys != observed_scored_keys:
        raise ResidualLoadComparisonReportError(
            "Les paires scorées du manifeste diffèrent de paired_hourly."
        )
    n_discovered = _require_int(
        payload.get("n_discovered_pairs"), name="manifest.n_discovered_pairs"
    )
    n_scored = _require_int(
        payload.get("n_scored_pairs"), name="manifest.n_scored_pairs"
    )
    if n_discovered != len(pairs) or n_scored != len(scored_keys):
        raise ResidualLoadComparisonReportError(
            "Les compteurs de paires du manifeste sont incohérents."
        )

    unpaired = payload.get("unpaired_archives")
    if not isinstance(unpaired, list):
        raise ResidualLoadComparisonReportError(
            "manifest.unpaired_archives doit être une liste."
        )
    for position, raw in enumerate(unpaired):
        item = _require_mapping(
            raw, name=f"manifest.unpaired_archives[{position}]"
        )
        if str(item.get("zone")) not in zones:
            raise ResidualLoadComparisonReportError(
                f"manifest.unpaired_archives[{position}].zone est hors périmètre."
            )
        _require_day(
            item.get("delivery_day_local"),
            name=f"manifest.unpaired_archives[{position}].delivery_day_local",
        )
        if item.get("missing") not in {"production", "challenger"}:
            raise ResidualLoadComparisonReportError(
                f"manifest.unpaired_archives[{position}].missing est invalide."
            )
    actuals = payload.get("actuals")
    if not isinstance(actuals, Mapping):
        raise ResidualLoadComparisonReportError(
            "manifest.actuals doit être un objet par zone."
        )
    if set(map(str, actuals)) != set(zones):
        raise ResidualLoadComparisonReportError(
            "manifest.actuals doit couvrir exactement les zones du manifeste."
        )
    payload["pairs"] = pairs
    return payload


def _fmt(value: Any, digits: int = 2, *, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:,.{digits}f}".replace(",", " ").replace(".", ",") + suffix


def _fmt_rate(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return _fmt(100.0 * number, 1, suffix=" %")


def _delta_class(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "na"
    if not math.isfinite(number):
        return "na"
    if number < 0:
        return "good"
    if number > 0:
        return "bad"
    return "neutral"


def _metric_card(label: str, value: str, *, note: str = "", css: str = "") -> str:
    note_html = f'<div class="kpi-note">{escape(note)}</div>' if note else ""
    classes = f"kpi {css}".strip()
    return (
        f'<div class="{classes}"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}</div>{note_html}</div>'
    )


def _svg_line_chart(frame: pd.DataFrame, *, zone: str) -> str:
    if frame.empty:
        return '<p class="empty">Aucune paire réalisée à représenter.</p>'
    width, height = 1080, 390
    left, right, top, bottom = 74, 28, 32, 56
    plot_width = width - left - right
    plot_height = height - top - bottom
    series = (
        ("Prix réel", "actual", "#111827"),
        ("Production · charge résiduelle Saturn", "production_q50", "#2563eb"),
        ("Challenger · charge résiduelle Chronos-2", "challenger_q50", "#059669"),
    )
    all_values = np.concatenate(
        [frame[column].to_numpy(float) for _, column, _ in series]
    )
    low = float(np.min(all_values))
    high = float(np.max(all_values))
    span = high - low
    padding = max(span * 0.08, 1.0)
    low -= padding
    high += padding
    span = high - low
    count = len(frame)

    def x(position: int) -> float:
        return left + (plot_width * position / max(count - 1, 1))

    def y(value: float) -> float:
        return top + plot_height * (high - value) / span

    grid: list[str] = []
    for tick in range(5):
        value = low + span * tick / 4
        position = y(value)
        grid.append(
            f'<line x1="{left}" y1="{position:.2f}" x2="{width-right}" '
            'y2="{position:.2f}" class="grid-line"/>'
            f'<text x="{left-10}" y="{position+4:.2f}" text-anchor="end" '
            f'class="axis-label">{escape(_fmt(value, 0))}</text>'
        )
    paths: list[str] = []
    for label, column, color in series:
        points = " ".join(
            f"{x(position):.2f},{y(float(value)):.2f}"
            for position, value in enumerate(frame[column])
        )
        paths.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.4" stroke-linejoin="round" '
            'stroke-linecap="round" vector-effect="non-scaling-stroke"/>'
        )
    labels: list[str] = []
    label_positions = sorted({0, count // 2, count - 1})
    for position in label_positions:
        timestamp = pd.Timestamp(frame.iloc[position]["delivery_start_utc"])
        labels.append(
            f'<text x="{x(position):.2f}" y="{height-22}" text-anchor="middle" '
            f'class="axis-label">{escape(timestamp.strftime("%d/%m %Hh UTC"))}</text>'
        )
    legend: list[str] = []
    legend_x = left
    for label, _, color in series:
        legend.append(
            f'<line x1="{legend_x}" y1="17" x2="{legend_x+22}" y2="17" '
            f'stroke="{color}" stroke-width="3"/>'
            f'<text x="{legend_x+28}" y="21" class="legend-label">'
            f'{escape(label)}</text>'
        )
        legend_x += 285
    aria = escape(
        f"{zone}: prix réels et prévisions Q50 des deux traitements",
        quote=True,
    )
    return (
        '<div class="chart-scroll">'
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{aria}">{"".join(grid)}'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
        'class="axis-line"/>'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" class="axis-line"/>{"".join(paths)}'
        f'{"".join(labels)}{"".join(legend)}</svg></div>'
    )


def _daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["delivery_day_local", "production_mae", "challenger_mae"]
        )
    return (
        frame.groupby("delivery_day_local", sort=True)[
            ["production_abs_error", "challenger_abs_error"]
        ]
        .mean()
        .rename(
            columns={
                "production_abs_error": "production_mae",
                "challenger_abs_error": "challenger_mae",
            }
        )
        .reset_index()
    )


def _svg_daily_mae(frame: pd.DataFrame, *, zone: str) -> str:
    daily = _daily_frame(frame)
    if daily.empty:
        return '<p class="empty">Aucune journée réalisée à représenter.</p>'
    width = max(900, 100 + 58 * len(daily))
    height = 340
    left, right, top, bottom = 70, 24, 34, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    high = max(
        float(daily["production_mae"].max()),
        float(daily["challenger_mae"].max()),
        1.0,
    ) * 1.1
    group_width = plot_width / len(daily)
    bar_width = min(18.0, group_width * 0.32)
    grid: list[str] = []
    for tick in range(5):
        value = high * tick / 4
        y = top + plot_height * (1.0 - value / high)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" class="grid-line"/>'
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'class="axis-label">{escape(_fmt(value, 1))}</text>'
        )
    bars: list[str] = []
    label_step = max(1, math.ceil(len(daily) / 14))
    for position, row in daily.iterrows():
        center = left + group_width * (position + 0.5)
        for offset, column, color in (
            (-bar_width, "production_mae", "#2563eb"),
            (0.0, "challenger_mae", "#059669"),
        ):
            value = float(row[column])
            bar_height = plot_height * value / high
            bars.append(
                f'<rect x="{center+offset:.2f}" y="{top+plot_height-bar_height:.2f}" '
                f'width="{bar_width:.2f}" height="{bar_height:.2f}" '
                f'fill="{color}" rx="2"><title>{escape(str(row["delivery_day_local"]))}'
                f' · {escape(_fmt(value))} EUR/MWh</title></rect>'
            )
        if position % label_step == 0 or position == len(daily) - 1:
            bars.append(
                f'<text x="{center:.2f}" y="{height-43}" text-anchor="end" '
                f'transform="rotate(-35 {center:.2f} {height-43})" '
                f'class="axis-label">{escape(str(row["delivery_day_local"]))}</text>'
            )
    aria = escape(f"{zone}: MAE journalière appariée", quote=True)
    return (
        '<div class="chart-scroll">'
        f'<svg class="chart daily-chart" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{aria}">{"".join(grid)}'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
        'class="axis-line"/>'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" class="axis-line"/>{"".join(bars)}'
        f'<line x1="{left}" y1="17" x2="{left+22}" y2="17" '
        'stroke="#2563eb" stroke-width="7"/>'
        f'<text x="{left+30}" y="21" class="legend-label">Production</text>'
        f'<line x1="{left+145}" y1="17" x2="{left+167}" y2="17" '
        'stroke="#059669" stroke-width="7"/>'
        f'<text x="{left+175}" y="21" class="legend-label">Challenger</text>'
        '</svg></div>'
    )


def _metrics_table(metrics: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in metrics.iterrows():
        delta_class = _delta_class(row["mae_delta"])
        rows.append(
            "<tr>"
            f'<td>{escape(str(row["zone"]))}</td>'
            f'<td>{_require_int(row["n_days"], name="metrics.n_days")}</td>'
            f'<td>{_require_int(row["n_hours"], name="metrics.n_hours")}</td>'
            f'<td>{escape(_fmt(row["production_mae"]))}</td>'
            f'<td>{escape(_fmt(row["challenger_mae"]))}</td>'
            f'<td class="{delta_class}">{escape(_fmt(row["mae_delta"]))}</td>'
            f'<td>{escape(_fmt(row["production_rmse"]))}</td>'
            f'<td>{escape(_fmt(row["challenger_rmse"]))}</td>'
            f'<td>{escape(_fmt(row["production_bias"]))}</td>'
            f'<td>{escape(_fmt(row["challenger_bias"]))}</td>'
            f'<td>{escape(_fmt(row["production_smape_pct"], 2, suffix=" %"))}</td>'
            f'<td>{escape(_fmt(row["challenger_smape_pct"], 2, suffix=" %"))}</td>'
            f'<td>{escape(_fmt_rate(row["challenger_win_rate"]))}</td>'
            f'<td>{escape(_fmt_rate(row["tie_rate"]))}</td>'
            "</tr>"
        )
    headers = (
        "Zone",
        "Jours",
        "Heures",
        "MAE production",
        "MAE challenger",
        "Δ MAE",
        "RMSE production",
        "RMSE challenger",
        "Biais production",
        "Biais challenger",
        "sMAPE production",
        "sMAPE challenger",
        "Win rate challenger",
        "Égalités",
    )
    return (
        '<div class="table-wrap"><table id="metrics-table"><thead><tr>'
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _daily_table(frame: pd.DataFrame, *, zone: str) -> str:
    daily = _daily_frame(frame)
    rows: list[str] = []
    for _, row in daily.iterrows():
        delta = float(row["challenger_mae"] - row["production_mae"])
        outcome = "Challenger" if delta < 0 else "Production" if delta > 0 else "Égalité"
        rows.append(
            "<tr>"
            f'<td>{escape(str(row["delivery_day_local"]))}</td>'
            f'<td>{escape(_fmt(row["production_mae"]))}</td>'
            f'<td>{escape(_fmt(row["challenger_mae"]))}</td>'
            f'<td class="{_delta_class(delta)}">{escape(_fmt(delta))}</td>'
            f'<td>{escape(outcome)}</td>'
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="5" class="na">Aucune journée scorée.</td></tr>')
    return (
        '<div class="table-wrap"><table><thead><tr><th>Jour local</th>'
        '<th>MAE production</th><th>MAE challenger</th><th>Δ MAE</th>'
        f'<th>Meilleur</th></tr></thead><tbody id="daily-{escape(zone.lower(), quote=True)}">'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _hourly_table(paired: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in paired.iterrows():
        timestamp = pd.Timestamp(row["delivery_start_utc"]).isoformat()
        delta = float(row["challenger_abs_error"] - row["production_abs_error"])
        rows.append(
            "<tr>"
            f'<td>{escape(str(row["zone"]))}</td>'
            f'<td>{escape(str(row["delivery_day_local"]))}</td>'
            f'<td>{escape(timestamp)}</td>'
            f'<td>{escape(_fmt(row["actual"]))}</td>'
            f'<td>{escape(_fmt(row["production_q50"]))}</td>'
            f'<td>{escape(_fmt(row["challenger_q50"]))}</td>'
            f'<td>{escape(_fmt(row["production_abs_error"]))}</td>'
            f'<td>{escape(_fmt(row["challenger_abs_error"]))}</td>'
            f'<td class="{_delta_class(delta)}">{escape(_fmt(delta))}</td>'
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="9" class="na">Aucune paire scorée.</td></tr>')
    return (
        '<div class="table-wrap hourly"><table id="hourly-table"><thead><tr>'
        '<th>Zone</th><th>Jour local</th><th>Livraison UTC</th><th>Prix réel</th>'
        '<th>Q50 production</th><th>Q50 challenger</th><th>|Erreur| production</th>'
        '<th>|Erreur| challenger</th><th>Δ |Erreur|</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _short_hash(value: Any) -> str:
    text = str(value or "")
    return text[:16] + "…" if _SHA256.fullmatch(text) else "N/A"


def _audit_html(
    manifest: Mapping[str, Any],
    *,
    zones: Sequence[str],
    paired_sha256: str,
    metrics_sha256: str,
) -> str:
    pair_rows: list[str] = []
    for item in manifest["pairs"]:
        scored = bool(item["scored"])
        status = (
            "Scorée"
            if scored
            else "Non scorée · prix réels incomplets"
            if not item["realized_complete"]
            else "Non scorée"
        )
        status_class = "good" if scored else "warn"
        pair_rows.append(
            "<tr>"
            f'<td>{escape(str(item.get("zone", "")))}</td>'
            f'<td>{escape(str(item.get("delivery_day_local", "")))}</td>'
            f'<td>{escape(str(item.get("hours_in_local_day", "")))}</td>'
            f'<td class="{status_class}">{escape(status)}</td>'
            f'<td><code>{escape(_short_hash(item.get("production_manifest_sha256")))}</code></td>'
            f'<td><code>{escape(_short_hash(item.get("challenger_manifest_sha256")))}</code></td>'
            f'<td><code>{escape(_short_hash(item.get("downstream_source_identity_sha256")))}</code></td>'
            f'<td><code>{escape(_short_hash(item.get("protected_input_identity_sha256")))}</code></td>'
            "</tr>"
        )
    if not pair_rows:
        pair_rows.append('<tr><td colspan="8" class="na">Aucune paire découverte.</td></tr>')

    unpaired_rows: list[str] = []
    for item in manifest["unpaired_archives"]:
        missing = "challenger" if item.get("missing") == "challenger" else "production"
        unpaired_rows.append(
            "<tr>"
            f'<td>{escape(str(item.get("zone", "")))}</td>'
            f'<td>{escape(str(item.get("delivery_day_local", "")))}</td>'
            f'<td>{escape(missing)}</td>'
            "</tr>"
        )
    if not unpaired_rows:
        unpaired_rows.append('<tr><td colspan="3">Aucune archive non appariée.</td></tr>')

    actual_rows: list[str] = []
    actuals = _require_mapping(manifest["actuals"], name="manifest.actuals")
    for zone in zones:
        item = _require_mapping(actuals[zone], name=f"manifest.actuals.{zone}")
        source = str(item.get("source") or "N/A")
        sha = _short_hash(item.get("sha256"))
        hours = item.get("n_available_hours")
        hours_text = str(hours) if isinstance(hours, (int, np.integer)) else "N/A"
        actual_rows.append(
            "<tr>"
            f'<td>{escape(zone)}</td><td>{escape(source)}</td>'
            f'<td>{escape(hours_text)}</td><td><code>{escape(sha)}</code></td>'
            "</tr>"
        )

    return f"""
    <div class="grid audit-grid">
      <div class="kpi"><div class="kpi-label">paired_hourly.csv.gz</div>
        <code class="hash">{escape(paired_sha256)}</code></div>
      <div class="kpi"><div class="kpi-label">metrics.csv</div>
        <code class="hash">{escape(metrics_sha256)}</code></div>
    </div>
    <h3>Paires découvertes</h3>
    <div class="table-wrap"><table id="pair-audit-table"><thead><tr>
      <th>Zone</th><th>Jour local</th><th>Heures</th><th>Statut</th>
      <th>Manifeste production</th><th>Manifeste challenger</th>
      <th>Downstream gelé</th><th>Inputs protégés</th>
    </tr></thead><tbody>{''.join(pair_rows)}</tbody></table></div>
    <h3>Archives non appariées</h3>
    <div class="table-wrap"><table id="unpaired-audit-table"><thead><tr>
      <th>Zone</th><th>Jour local</th><th>Archive absente</th>
    </tr></thead><tbody>{''.join(unpaired_rows)}</tbody></table></div>
    <h3>Prix réels audités</h3>
    <div class="table-wrap"><table id="actual-audit-table"><thead><tr>
      <th>Zone</th><th>Source</th><th>Heures disponibles</th><th>SHA-256</th>
    </tr></thead><tbody>{''.join(actual_rows)}</tbody></table></div>
    """


def render_residual_load_comparison_report(
    paired_hourly: pd.DataFrame,
    metrics: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    paired_sha256: str,
    metrics_sha256: str,
) -> str:
    """Render the prospective paired comparison as a standalone HTML document.

    Only the objects supplied as arguments are used.  The function performs no
    file or network access and fails closed when their audit contract diverges.
    """

    manifest_object = _require_mapping(manifest, name="manifest")
    zones_raw = manifest_object.get("zones")
    if not isinstance(zones_raw, list) or not zones_raw:
        raise ResidualLoadComparisonReportError(
            "manifest.zones doit être une liste non vide."
        )
    zones = tuple(str(zone) for zone in zones_raw)
    if len(zones) != len(set(zones)) or any(
        _ZONE.fullmatch(zone) is None for zone in zones
    ):
        raise ResidualLoadComparisonReportError(
            "manifest.zones contient une zone invalide ou dupliquée."
        )
    paired_hash = _require_sha256(paired_sha256, name="paired_sha256")
    metrics_hash = _require_sha256(metrics_sha256, name="metrics_sha256")
    paired = _validate_paired_hourly(paired_hourly, zones=zones)
    validated_manifest = _validate_manifest(
        manifest_object,
        paired=paired,
        zones=zones,
    )
    validated_metrics = _validate_metrics(
        metrics,
        paired=paired,
        zones=zones,
    )

    overall = validated_metrics.loc[validated_metrics["zone"] == "ALL"].iloc[0]
    has_scores = bool(len(paired))
    if has_scores:
        score_banner = (
            '<div class="banner safe"><strong>Données réalisées disponibles.</strong> '
            f'{len(paired)} heures et {validated_manifest["n_scored_pairs"]} '
            "journées-zone complètes sont scorées.</div>"
        )
    else:
        score_banner = (
            '<div class="banner warning"><strong>Données insuffisantes.</strong> '
            "Aucune paire complète avec prix réels n’est scorée. Les métriques "
            "restent à N/A et aucun résultat de performance ne peut être conclu.</div>"
        )

    overall_cards = "".join(
        (
            _metric_card(
                "MAE production",
                _fmt(overall["production_mae"], suffix=" EUR/MWh"),
            ),
            _metric_card(
                "MAE challenger",
                _fmt(overall["challenger_mae"], suffix=" EUR/MWh"),
            ),
            _metric_card(
                "Δ MAE challenger − production",
                _fmt(overall["mae_delta"], suffix=" EUR/MWh"),
                note="Négatif : avantage challenger",
                css=_delta_class(overall["mae_delta"]),
            ),
            _metric_card(
                "Win rate horaire challenger",
                _fmt_rate(overall["challenger_win_rate"]),
                note="Les égalités restent dans le dénominateur",
            ),
            _metric_card(
                "Win rate journalier challenger",
                _fmt_rate(overall["challenger_day_win_rate"]),
            ),
            _metric_card(
                "Paires scorées",
                str(validated_manifest["n_scored_pairs"]),
                note=f'{validated_manifest["n_discovered_pairs"]} découvertes',
            ),
        )
    )

    zone_sections: list[str] = []
    nav_zones: list[str] = []
    for zone in zones:
        anchor = f"zone-{zone.lower()}"
        nav_zones.append(f'<a class="navlink" href="#{anchor}">{escape(zone)}</a>')
        part = paired.loc[paired["zone"] == zone]
        metric = validated_metrics.loc[validated_metrics["zone"] == zone].iloc[0]
        cards = "".join(
            (
                _metric_card(
                    "MAE production",
                    _fmt(metric["production_mae"], suffix=" EUR/MWh"),
                ),
                _metric_card(
                    "MAE challenger",
                    _fmt(metric["challenger_mae"], suffix=" EUR/MWh"),
                ),
                _metric_card(
                    "Δ MAE",
                    _fmt(metric["mae_delta"], suffix=" EUR/MWh"),
                    note="Négatif : avantage challenger",
                    css=_delta_class(metric["mae_delta"]),
                ),
                _metric_card(
                    "Win rate horaire",
                    _fmt_rate(metric["challenger_win_rate"]),
                ),
                _metric_card("Jours scorés", str(int(metric["n_days"]))),
                _metric_card("Heures scorées", str(int(metric["n_hours"]))),
            )
        )
        zone_sections.append(
            f"""
            <section class="card" id="{anchor}">
              <div class="section-title"><div><h2>{escape(zone)}</h2>
                <p class="muted">Paires prospectives réalisées uniquement.</p></div>
                <span class="badge">Q50 · downstream gelé</span></div>
              <div class="grid">{cards}</div>
              <h3>Prix réel et deux prévisions appariées</h3>
              {_svg_line_chart(part, zone=zone)}
              <h3>MAE journalière appariée</h3>
              {_svg_daily_mae(part, zone=zone)}
              <h3>Détail journalier</h3>
              {_daily_table(part, zone=zone)}
            </section>
            """
        )

    generated = str(validated_manifest.get("published_at_utc") or "avant publication")
    start = str(validated_manifest["start_delivery_day_local"])
    end = str(validated_manifest["end_delivery_day_local"])
    audit = _audit_html(
        validated_manifest,
        zones=zones,
        paired_sha256=paired_hash,
        metrics_sha256=metrics_hash,
    )
    css = r"""
    :root{color-scheme:light;font-family:Inter,"Segoe UI",Arial,sans-serif;
      --bg:#f5f7fb;--surface:#fff;--surface2:#f8fafc;--text:#0f172a;
      --muted:#526174;--line:#dbe3ed;--brand:#2563eb;--good:#047857;
      --bad:#b91c1c;--warn:#9a5b0a;--shadow:0 8px 28px rgba(15,23,42,.07)}
    html[data-theme="dark"]{color-scheme:dark;--bg:#0b1220;--surface:#111b2e;
      --surface2:#172338;--text:#e7edf6;--muted:#aab7ca;--line:#2b3b52;
      --brand:#60a5fa;--good:#34d399;--bad:#fb7185;--warn:#fbbf24;
      --shadow:0 8px 28px rgba(0,0,0,.28)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);
      color:var(--text)}main{max-width:1540px;margin:auto;padding:26px 30px 60px}
    .topbar{display:flex;gap:18px;align-items:flex-start;justify-content:space-between}
    .topbar h1{margin:0 0 7px;font-size:clamp(25px,3vw,38px)}h2{margin:0 0 6px;
      font-size:23px}h3{margin:24px 0 10px}.muted{color:var(--muted)}.tiny{font-size:12px}
    button,.navlink{font:inherit;color:var(--text);background:var(--surface);border:1px solid
      var(--line);border-radius:999px;padding:8px 12px}.navlink{text-decoration:none;font-size:13px}
    button{cursor:pointer}.toolbar{position:sticky;top:0;z-index:30;display:flex;gap:8px;
      flex-wrap:wrap;padding:10px 0;background:color-mix(in srgb,var(--bg) 92%,transparent);
      backdrop-filter:blur(8px)}.card{background:var(--surface);border:1px solid var(--line);
      border-radius:14px;padding:20px;margin-top:18px;box-shadow:var(--shadow)}
    .banner{border-radius:10px;padding:13px 15px;margin-top:14px;line-height:1.5}
    .safe{background:color-mix(in srgb,var(--good) 10%,var(--surface));border:1px solid
      color-mix(in srgb,var(--good) 35%,var(--line));color:var(--good)}
    .warning{background:color-mix(in srgb,var(--warn) 12%,var(--surface));border:1px solid
      color-mix(in srgb,var(--warn) 38%,var(--line));color:var(--warn)}
    .notice{border-left:4px solid var(--brand);background:var(--surface2);padding:13px 16px;
      line-height:1.55;margin-top:14px}.grid{display:grid;grid-template-columns:
      repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-top:14px}.kpi{background:var(--surface2);
      border:1px solid var(--line);border-radius:11px;padding:14px;min-width:0}
    .kpi.good .kpi-value,.good{color:var(--good);font-weight:750}.kpi.bad .kpi-value,
      .bad{color:var(--bad);font-weight:750}.warn{color:var(--warn);font-weight:750}
    .neutral{color:var(--text)}.na{color:var(--muted);font-style:italic}
    .kpi-label{font-size:12px;font-weight:700;color:var(--muted)}.kpi-value{font-size:23px;
      font-weight:800;margin:8px 0 2px}.kpi-note{font-size:12px;color:var(--muted)}
    .section-title{display:flex;align-items:flex-start;justify-content:space-between;gap:18px}
    .badge{background:color-mix(in srgb,var(--brand) 12%,var(--surface));color:var(--brand);
      padding:7px 11px;border-radius:999px;font-weight:700;font-size:12px}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}.table-wrap.hourly{
      max-height:680px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:9px 10px;
      border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}th{position:sticky;
      top:0;z-index:2;background:var(--surface2);font-size:11px;text-transform:uppercase;
      letter-spacing:.03em;color:var(--muted)}th:first-child,td:first-child{text-align:left}
    tr:hover td{background:color-mix(in srgb,var(--brand) 5%,var(--surface))}
    .chart-scroll{overflow:auto;border:1px solid var(--line);border-radius:10px;
      background:var(--surface)}.chart{display:block;width:100%;min-width:800px;min-height:330px;
      color:var(--text)}.daily-chart{width:auto}.grid-line{stroke:var(--line);stroke-width:1}
    .axis-line{stroke:var(--muted);stroke-width:1}.axis-label{fill:var(--muted);font-size:11px}
    .legend-label{fill:var(--text);font-size:11px;font-weight:700}.empty{padding:24px;
      text-align:center;color:var(--muted)}code.hash{display:block;margin-top:8px;overflow-wrap:anywhere;
      white-space:normal;font-size:11px}.audit-grid{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
    .foot{margin-top:24px;padding-top:15px;border-top:1px solid var(--line)}
    @media(max-width:900px){main{padding:18px 12px}.topbar{display:block}.toolbar{position:static}
      .section-title{display:block}.badge{display:inline-block;margin-top:8px}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
    @media print{.toolbar,#theme-toggle{display:none!important}.card{break-inside:avoid;
      box-shadow:none}main{max-width:none}.table-wrap{overflow:visible}.chart{min-width:0}}
    """
    return f"""<!doctype html>
<html lang="fr" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; connect-src 'none'; object-src 'none'; base-uri 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:">
  <title>Chronos-2 · comparaison prospective appariée</title>
  <script>(()=>{{let t='light';try{{const s=localStorage.getItem('chronos2-report-theme');if(s==='light'||s==='dark')t=s}}catch(_e){{}}document.documentElement.dataset.theme=t}})();</script>
  <style>{css}</style>
</head>
<body><main>
  <header class="topbar"><div><h1>Comparaison prospective · charge résiduelle</h1>
    <p class="muted">{escape(start)} → {escape(end)} · rapport {escape(generated)}</p></div>
    <button id="theme-toggle" type="button" aria-pressed="false" aria-label="Activer le mode nuit">☾ Mode nuit</button></header>
  <nav class="toolbar" aria-label="Sections"><a class="navlink" href="#overview">Synthèse</a>
    {''.join(nav_zones)}<a class="navlink" href="#hourly-values">Valeurs</a>
    <a class="navlink" href="#audit">Audit</a><button id="print-report" type="button">Imprimer / PDF</button></nav>
  <div class="banner warning"><strong>Interprétation encadrée.</strong>
    Comparaison prospective appariée réalisée : production avec prévisions de charge résiduelle Saturn,
    challenger avec prévisions de charge résiduelle Chronos-2. Le modèle de prix et le correcteur aval
    sont gelés et leurs entrées hors traitement sont vérifiées identiques. Il s’agit d’une mesure sous
    changement de distribution, pas d’un benchmark symétriquement réentraîné.</div>
  {score_banner}
  <noscript><div class="banner warning">JavaScript est uniquement nécessaire pour le changement de thème et l’impression.</div></noscript>
  <section class="card" id="overview"><h2>Synthèse</h2>
    <p class="notice">Seules les journées-zone possédant deux archives immuables, une timeline civile
      complète et tous les prix réels sont incluses. Une valeur Δ négative indique une erreur plus faible
      pour le challenger. Les résultats restent descriptifs.</p>
    <div class="grid">{overall_cards}</div>
    <h3>Métriques par zone</h3>{_metrics_table(validated_metrics)}
  </section>
  {''.join(zone_sections)}
  <section class="card" id="hourly-values"><h2>Valeurs horaires appariées</h2>
    <p class="muted">Erreur = Q50 − prix réel. Δ |Erreur| = challenger − production.</p>
    {_hourly_table(paired)}
  </section>
  <section class="card" id="audit"><h2>Audit de la comparaison</h2>
    <p class="muted">Les chemins locaux ne sont pas exposés. Les empreintes relient ce rapport aux
      artefacts scellés par le manifeste de publication.</p>{audit}
  </section>
  <footer class="foot muted tiny">Chronos-2 · {REPORT_SCHEMA}</footer>
</main>
<script>(()=>{{'use strict';const root=document.documentElement,b=document.getElementById('theme-toggle'),
setTheme=(theme,persist)=>{{const t=theme==='dark'?'dark':'light';root.dataset.theme=t;
b.textContent=t==='dark'?'☀ Mode clair':'☾ Mode nuit';b.setAttribute('aria-pressed',String(t==='dark'));
 b.setAttribute('aria-label',t==='dark'?'Activer le mode clair':'Activer le mode nuit');
if(persist){{try{{localStorage.setItem('chronos2-report-theme',t)}}catch(_e){{}}}}}};
setTheme(root.dataset.theme,false);b.addEventListener('click',()=>setTheme(root.dataset.theme==='dark'?'light':'dark',true));
document.getElementById('print-report').addEventListener('click',()=>window.print());}})();</script>
</body></html>"""


__all__ = [
    "REPORT_SCHEMA",
    "ResidualLoadComparisonReportError",
    "render_residual_load_comparison_report",
]
