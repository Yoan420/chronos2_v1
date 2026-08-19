"""Auditable market-coupling view for the live multi-zone forecasts.

The default signal is deliberately an *economic proxy*: the forecast price
spread between two adjacent bidding zones.  It must never be described as a
cross-border flow.  A caller may provide flow/capacity observations, but they
are promoted to verified signals only after strict border, timestamp,
provenance and point-in-time checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index


MARKET_ZONES: tuple[str, ...] = ("FR", "BE", "DE", "NL", "ES")
ZONE_COORDINATES: dict[str, tuple[float, float]] = {
    "FR": (2.21, 46.23),
    "BE": (4.47, 50.85),
    "DE": (10.45, 51.16),
    "NL": (5.29, 52.13),
    "ES": (-3.70, 40.42),
}
ZONE_TIMEZONES: dict[str, str] = {
    "FR": "Europe/Paris",
    "BE": "Europe/Brussels",
    "DE": "Europe/Berlin",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}

# Only actual bidding-zone adjacencies are rendered.  In particular, no
# synthetic FR-NL, DE-ES, BE-ES or NL-ES link is allowed.
MARKET_BORDERS: tuple[tuple[str, str], ...] = (
    ("FR", "BE"),
    ("FR", "DE"),
    ("FR", "ES"),
    ("BE", "DE"),
    ("BE", "NL"),
    ("DE", "NL"),
)

SIGNAL_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "zone_from",
    "zone_to",
    "value",
    "unit",
    "kind",
    "series",
    "as_of_utc",
    "causal_verified",
)
DIRECTION_STABILITY_THRESHOLD = 0.75
_DIRECTION_EPSILON = 1e-12


class MarketCouplingDataError(ValueError):
    """Raised when a map input cannot be represented without ambiguity."""


@dataclass(frozen=True)
class PriceForecastBundle:
    """Validated, checksum-backed forecasts on one common UTC timeline."""

    delivery_day: date
    timeline_utc: pd.DatetimeIndex
    prices: pd.DataFrame
    provenance: pd.DataFrame
    data_as_of_utc: pd.Timestamp


@dataclass(frozen=True)
class MarketCouplingView:
    """Pure-data representation consumed by PyDeck or another renderer."""

    delivery_day: date
    aggregation: Literal["hour", "day"]
    selected_time_utc: pd.Timestamp | None
    nodes: pd.DataFrame
    edges: pd.DataFrame
    provenance: pd.DataFrame
    caveat: str

    @property
    def uses_only_proxies(self) -> bool:
        return bool((self.edges["status"] == "proxy").all())


def _zone_run_directory(project_root: Path, zone: str, day: date) -> Path:
    token = day.isoformat()
    if zone == "FR":
        return project_root / "runs" / "live" / f"fr_day_ahead_{token}"
    lower = zone.lower()
    return project_root / "runs" / "live" / lower / f"{lower}_day_ahead_{token}"


def _available_live_days(project_root: Path, zone: str) -> set[date]:
    if zone == "FR":
        base = project_root / "runs" / "live"
        pattern = "fr_day_ahead_*"
    else:
        lower = zone.lower()
        base = project_root / "runs" / "live" / lower
        pattern = f"{lower}_day_ahead_*"
    if not base.exists():
        return set()

    prefix = pattern[:-1]
    available: set[date] = set()
    for candidate in base.glob(pattern):
        if not candidate.is_dir():
            continue
        token = candidate.name.removeprefix(prefix)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
            continue
        try:
            parsed_day = date.fromisoformat(token)
        except ValueError:
            continue
        forecast_name = f"forecast_hourly_{zone.lower()}.csv"
        # A directory name alone is not a publication.  Requiring the three
        # sealing files prevents an in-progress atomic publish from becoming
        # the automatically selected common day.
        if all(
            (candidate / filename).is_file()
            for filename in ("run_manifest.json", "artifact_checksums.json", forecast_name)
        ):
            available.add(parsed_day)
    return available


def discover_latest_common_price_day(
    project_root: str | Path,
    zones: Iterable[str] = MARKET_ZONES,
) -> date:
    """Return the latest day having a published run directory for every zone."""

    root = Path(project_root)
    normalized = _normalize_zones(zones)
    common: set[date] | None = None
    for zone in normalized:
        days = _available_live_days(root, zone)
        common = days if common is None else common.intersection(days)
    if not common:
        raise MarketCouplingDataError(
            "Aucun jour de prévision publié n'est commun à "
            + ", ".join(normalized)
            + "."
        )
    return max(common)


def _normalize_zones(zones: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(zone).upper() for zone in zones)
    if not normalized or len(set(normalized)) != len(normalized):
        raise MarketCouplingDataError("La liste des zones doit être unique et non vide.")
    unknown = sorted(set(normalized).difference(MARKET_ZONES))
    if unknown:
        raise MarketCouplingDataError(f"Zones non prises en charge: {', '.join(unknown)}")
    return normalized


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarketCouplingDataError(f"JSON illisible: {path}") from exc
    if not isinstance(payload, dict):
        raise MarketCouplingDataError(f"Objet JSON attendu: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_run_artifact(checksums_path: Path, artifact_path: Path) -> str:
    payload = _read_json(checksums_path)
    if payload.get("algorithm") != "sha256":
        raise MarketCouplingDataError(f"Algorithme de checksum non pris en charge: {checksums_path}")
    rows = payload.get("artifacts")
    if not isinstance(rows, list):
        raise MarketCouplingDataError(f"Liste d'artefacts absente: {checksums_path}")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("role") == "run_artifact"
        and Path(str(row.get("path", ""))).name == artifact_path.name
    ]
    if len(matches) != 1:
        raise MarketCouplingDataError(
            f"Artefact {artifact_path.name} non déclaré de façon unique dans {checksums_path}."
        )
    declared = str(matches[0].get("sha256", "")).lower()
    actual = _sha256(artifact_path)
    if not re.fullmatch(r"[0-9a-f]{64}", declared) or declared != actual:
        raise MarketCouplingDataError(f"Checksum invalide pour {artifact_path}.")
    return actual


def _as_utc_timestamp(value: Any, *, field: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise MarketCouplingDataError(f"Horodatage invalide pour {field}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise MarketCouplingDataError(f"{field} doit être timezone-aware.")
    return parsed.tz_convert("UTC")


def load_live_price_forecasts(
    project_root: str | Path,
    delivery_day: str | date | None = None,
    zones: Iterable[str] = MARKET_ZONES,
) -> PriceForecastBundle:
    """Load one atomic, common-day view of checksum-backed live forecasts.

    Storm columns are neither selected nor used as a fallback.  The q50 column
    is resolved exclusively from the candidate/native model declared by each
    run manifest.
    """

    root = Path(project_root)
    normalized = _normalize_zones(zones)
    day = (
        discover_latest_common_price_day(root, normalized)
        if delivery_day is None
        else date.fromisoformat(str(delivery_day))
        if not isinstance(delivery_day, date)
        else delivery_day
    )

    prices: dict[str, pd.Series] = {}
    provenance_rows: list[dict[str, Any]] = []
    common_index: pd.DatetimeIndex | None = None
    cutoffs: list[pd.Timestamp] = []

    for zone in normalized:
        run_dir = _zone_run_directory(root, zone, day)
        manifest_path = run_dir / "run_manifest.json"
        checksums_path = run_dir / "artifact_checksums.json"
        forecast_path = run_dir / f"forecast_hourly_{zone.lower()}.csv"
        missing = [path for path in (manifest_path, checksums_path, forecast_path) if not path.is_file()]
        if missing:
            raise MarketCouplingDataError(
                f"Publication {zone} incomplète pour {day}: "
                + ", ".join(str(path) for path in missing)
            )

        manifest = _read_json(manifest_path)
        if str(manifest.get("zone", "")).upper() != zone:
            raise MarketCouplingDataError(f"Zone incohérente dans {manifest_path}.")
        if str(manifest.get("delivery_day_local", "")) != day.isoformat():
            raise MarketCouplingDataError(f"Jour de livraison incohérent dans {manifest_path}.")
        if manifest.get("storm_used_as_feature") is not False:
            raise MarketCouplingDataError(
                f"Le manifeste {zone} ne prouve pas que Storm est exclu des prédicteurs."
            )
        timezone = str(manifest.get("timezone", ""))
        if timezone != ZONE_TIMEZONES[zone]:
            raise MarketCouplingDataError(
                f"Timezone inattendue pour {zone}: {timezone!r} (attendu {ZONE_TIMEZONES[zone]})."
            )
        model = str(manifest.get("candidate_model") or manifest.get("native_model") or "")
        if not model or "storm" in model.lower():
            raise MarketCouplingDataError(f"Modèle candidat invalide pour {zone}: {model!r}.")
        q50_column = f"{model}__q50"
        cutoff_value = manifest.get("forecast_cutoff_utc")
        if cutoff_value is None:
            cutoff_value = manifest.get("forecast_origin_utc")
        if cutoff_value is None:
            cutoff_value = manifest.get("data_as_of_utc")
        cutoff = _as_utc_timestamp(cutoff_value, field=f"forecast_origin_utc[{zone}]")
        cutoffs.append(cutoff)

        digest = _validate_run_artifact(checksums_path, forecast_path)
        frame = pd.read_csv(forecast_path)
        required = {"delivery_start_utc", q50_column}
        if not required.issubset(frame.columns):
            raise MarketCouplingDataError(
                f"Colonnes absentes dans {forecast_path}: {sorted(required.difference(frame.columns))}"
            )
        timestamp = pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="coerce")
        if timestamp.isna().any() or timestamp.duplicated().any():
            raise MarketCouplingDataError(f"Timeline UTC invalide ou dupliquée dans {forecast_path}.")
        index = pd.DatetimeIndex(timestamp)
        expected = local_delivery_day_index(day, timezone=timezone)
        if not index.equals(expected):
            raise MarketCouplingDataError(
                f"Timeline {zone} non conforme au jour local {day} ({len(expected)} heures attendues)."
            )
        values = pd.to_numeric(frame[q50_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise MarketCouplingDataError(f"Prévision q50 non finie pour {zone}.")
        if common_index is None:
            common_index = index
        elif not index.equals(common_index):
            raise MarketCouplingDataError(f"Timeline {zone} non alignée avec les autres zones.")
        prices[zone] = pd.Series(values, index=index, name=zone)
        provenance_rows.append(
            {
                "zone": zone,
                "status": "forecast_price",
                "model": model,
                "q50_column": q50_column,
                "forecast_path": str(forecast_path),
                "manifest_path": str(manifest_path),
                "sha256": digest,
                "data_as_of_utc": cutoff.isoformat(),
                "storm_used_as_feature": False,
            }
        )

    assert common_index is not None  # normalized zones cannot be empty
    if len(set(cutoffs)) != 1:
        rendered = ", ".join(sorted(value.isoformat() for value in set(cutoffs)))
        raise MarketCouplingDataError(
            "Les publications pays ne partagent pas le même cutoff causal: " + rendered
        )
    price_frame = pd.DataFrame(prices, index=common_index)
    return PriceForecastBundle(
        delivery_day=day,
        timeline_utc=common_index,
        prices=price_frame,
        provenance=pd.DataFrame(provenance_rows),
        data_as_of_utc=cutoffs[0],
    )


def _validate_border_signals(
    signals: pd.DataFrame | None,
    *,
    bundle: PriceForecastBundle,
) -> pd.DataFrame:
    if signals is None or signals.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    missing = set(SIGNAL_COLUMNS).difference(signals.columns)
    if missing:
        raise MarketCouplingDataError(
            "Schéma flux/capacité incomplet; colonnes absentes: " + ", ".join(sorted(missing))
        )
    frame = signals.loc[:, SIGNAL_COLUMNS].copy()
    frame["zone_from"] = frame["zone_from"].astype(str).str.upper()
    frame["zone_to"] = frame["zone_to"].astype(str).str.upper()
    allowed = {frozenset(border) for border in MARKET_BORDERS}
    for row in frame.itertuples(index=False):
        pair = frozenset((row.zone_from, row.zone_to))
        if (
            row.zone_from == row.zone_to
            or row.zone_from not in bundle.prices.columns
            or row.zone_to not in bundle.prices.columns
            or pair not in allowed
        ):
            raise MarketCouplingDataError(
                f"Frontière non autorisée: {row.zone_from}-{row.zone_to}."
            )
    frame["kind"] = frame["kind"].astype(str).str.lower()
    if not frame["kind"].isin(("flow", "capacity")).all():
        raise MarketCouplingDataError("kind doit valoir 'flow' ou 'capacity'.")
    if not frame["unit"].astype(str).str.upper().eq("MW").all():
        raise MarketCouplingDataError("Les flux/capacités vérifiés doivent être exprimés en MW.")
    if frame["series"].isna().any() or frame["series"].astype(str).str.strip().eq("").any():
        raise MarketCouplingDataError("Chaque signal vérifié doit déclarer sa série source.")
    if not frame["causal_verified"].map(lambda value: value is True).all():
        raise MarketCouplingDataError("Chaque signal doit porter causal_verified=True.")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise MarketCouplingDataError("Valeur flux/capacité non finie.")
    if ((frame["kind"] == "capacity") & (frame["value"] < 0)).any():
        raise MarketCouplingDataError("Une capacité directionnelle ne peut pas être négative.")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame["as_of_utc"] = pd.to_datetime(frame["as_of_utc"], utc=True, errors="coerce")
    if frame[["timestamp_utc", "as_of_utc"]].isna().any().any():
        raise MarketCouplingDataError("Horodatage flux/capacité invalide.")
    if (~frame["timestamp_utc"].isin(bundle.timeline_utc)).any():
        raise MarketCouplingDataError("Un signal est hors de la timeline de livraison publiée.")
    if (frame["as_of_utc"] > bundle.data_as_of_utc).any():
        raise MarketCouplingDataError(
            "Un signal a été révisé après le cutoff causal commun des forecasts."
        )
    duplicate_key = ["timestamp_utc", "zone_from", "zone_to", "kind"]
    if frame.duplicated(duplicate_key).any():
        raise MarketCouplingDataError("Signal frontalier ambigu: plusieurs valeurs au même instant.")
    return frame


def _spread_style(magnitude: float) -> tuple[list[int], float, str]:
    if magnitude <= 2.0:
        color, coupling = [34, 197, 94, 220], "fort"
    elif magnitude <= 10.0:
        color, coupling = [245, 158, 11, 225], "intermédiaire"
    else:
        color, coupling = [239, 68, 68, 230], "faible"
    # Keep price-proxy lines modest. Colour carries the gap class; width is a
    # secondary, monotone cue and must never resemble a physical-flow ribbon.
    width = min(6.0, 2.0 + math.sqrt(max(magnitude, 0.0)) * 0.6)
    return color, float(width), coupling


def _verified_style(kind: str, magnitude: float) -> tuple[list[int], float, str]:
    color = [37, 99, 235, 225] if kind == "flow" else [124, 58, 237, 225]
    width = float(min(6.0, 2.25 + math.log1p(max(magnitude, 0.0)) / 2.2))
    return color, width, "mesuré" if kind == "flow" else "disponible"


def _bearing_degrees(source: tuple[float, float], target: tuple[float, float]) -> float:
    source_lon, source_lat = map(math.radians, source)
    target_lon, target_lat = map(math.radians, target)
    delta_lon = target_lon - source_lon
    y = math.sin(delta_lon) * math.cos(target_lat)
    x = math.cos(source_lat) * math.sin(target_lat) - math.sin(source_lat) * math.cos(
        target_lat
    ) * math.cos(delta_lon)
    # TextLayer's zero-degree orientation is horizontal; this adjustment makes
    # the triangle point approximately along the great-circle bearing.
    return float((math.degrees(math.atan2(y, x)) + 90.0) % 360.0)


def _edge_geometry(source_zone: str, target_zone: str) -> dict[str, Any]:
    source = ZONE_COORDINATES[source_zone]
    target = ZONE_COORDINATES[target_zone]
    ratio = 0.62
    return {
        "source_position": list(source),
        "target_position": list(target),
        "arrow_position": [
            source[0] + (target[0] - source[0]) * ratio,
            source[1] + (target[1] - source[1]) * ratio,
        ],
        "arrow_angle": _bearing_degrees(source, target),
    }


def _proxy_edge(
    zone_a: str,
    zone_b: str,
    *,
    prices: pd.DataFrame,
    aggregation: Literal["hour", "day"],
    selected: pd.Timestamp | None,
) -> dict[str, Any]:
    if aggregation == "hour":
        assert selected is not None
        price_a = float(prices.at[selected, zone_a])
        price_b = float(prices.at[selected, zone_b])
        signed_spread = price_b - price_a
        mean_abs = abs(signed_spread)
        direction_stability = 1.0
    else:
        differences = prices[zone_b] - prices[zone_a]
        signed_spread = float(differences.mean())
        mean_abs = float(differences.abs().mean())
        price_a = float(prices[zone_a].mean())
        price_b = float(prices[zone_b].mean())
        if abs(signed_spread) <= _DIRECTION_EPSILON:
            direction_stability = float(
                max(
                    (differences > _DIRECTION_EPSILON).mean(),
                    (differences < -_DIRECTION_EPSILON).mean(),
                )
            )
        else:
            signed_direction = 1.0 if signed_spread > 0.0 else -1.0
            direction_stability = float(
                ((differences * signed_direction) > _DIRECTION_EPSILON).mean()
            )
    equilibrium = mean_abs <= _DIRECTION_EPSILON
    direction_is_stable = (
        aggregation == "hour"
        or equilibrium
        or direction_stability >= DIRECTION_STABILITY_THRESHOLD
    )
    if direction_is_stable and signed_spread >= 0:
        source_zone, target_zone = zone_a, zone_b
        source_price, target_price = price_a, price_b
    elif direction_is_stable:
        source_zone, target_zone = zone_b, zone_a
        source_price, target_price = price_b, price_a
    else:
        # An unstable daily aggregate has no honest economic direction.  Keep
        # the canonical border order for prices/table and state that explicitly.
        source_zone, target_zone = zone_a, zone_b
        source_price, target_price = price_a, price_b
    magnitude = mean_abs
    color, width, coupling = _spread_style(magnitude)
    if equilibrium:
        direction_label = "Équilibre de prix"
    elif not direction_is_stable:
        direction_label = (
            f"Direction instable · {direction_stability:.0%} des heures "
            "dans le sens moyen"
        )
    elif aggregation == "day":
        direction_label = (
            f"{source_zone} → {target_zone} · prix bas vers prix haut · "
            f"stabilité {direction_stability:.0%}"
        )
    else:
        direction_label = f"{source_zone} → {target_zone} · prix bas vers prix haut"
    aggregate_note = (
        f"<br/>Écart absolu moyen: {mean_abs:.2f} EUR/MWh"
        f"<br/>Spread signé moyen {zone_a}→{zone_b}: "
        f"{signed_spread:+.2f} EUR/MWh"
        f"<br/>Stabilité de direction: {direction_stability:.0%}"
        if aggregation == "day"
        else f"<br/>Écart de prix P50: {mean_abs:.2f} EUR/MWh"
    )
    tooltip = (
        f"{zone_a}-{zone_b}<br/>{direction_label}"
        f"<br/>{source_zone}: {source_price:.2f} EUR/MWh"
        f" · {target_zone}: {target_price:.2f} EUR/MWh"
        f"{aggregate_note}"
        "<br/><b>Proxy économique uniquement — aucun flux physique.</b>"
    )
    return {
        "border": f"{zone_a}-{zone_b}",
        "source_zone": source_zone,
        "target_zone": target_zone,
        "status": "proxy",
        "status_label": "Proxy de spread prévu",
        "signal_kind": "forecast_price_spread_proxy",
        "value": magnitude,
        "signed_value_canonical": signed_spread,
        "mean_abs_value": mean_abs,
        "unit": "EUR/MWh",
        "origin_price_eur_mwh": source_price,
        "destination_price_eur_mwh": target_price,
        "price_spread_eur_mwh": target_price - source_price,
        "mean_abs_price_spread_eur_mwh": mean_abs,
        "direction_label": direction_label,
        "direction_stability": direction_stability,
        "direction_is_stable": direction_is_stable,
        "is_price_equilibrium": equilibrium,
        "arrow_glyph": "",
        "series": "forecast q50 des deux zones",
        "is_reported_flow": False,
        "coupling_label": coupling,
        "color": color,
        "width": width,
        "tooltip": tooltip,
        **_edge_geometry(source_zone, target_zone),
    }


def _verified_edge(
    zone_a: str,
    zone_b: str,
    *,
    selected_rows: pd.DataFrame,
    kind: Literal["flow", "capacity"],
    node_prices: pd.Series,
    prices: pd.DataFrame,
    aggregation: Literal["hour", "day"],
) -> dict[str, Any]:
    normalized: list[float] = []
    for row in selected_rows.itertuples(index=False):
        value = float(row.value)
        normalized.append(value if row.zone_from == zone_a else -value)
    signed = float(np.mean(normalized))
    mean_abs = float(np.mean(np.abs(normalized)))
    if abs(signed) <= _DIRECTION_EPSILON:
        direction_stability = float(
            max(
                np.mean(np.asarray(normalized) > _DIRECTION_EPSILON),
                np.mean(np.asarray(normalized) < -_DIRECTION_EPSILON),
            )
        )
    else:
        signed_direction = 1.0 if signed > 0.0 else -1.0
        direction_stability = float(
            np.mean(np.asarray(normalized) * signed_direction > _DIRECTION_EPSILON)
        )
    direction_is_stable = (
        aggregation == "hour"
        or mean_abs <= _DIRECTION_EPSILON
        or direction_stability >= DIRECTION_STABILITY_THRESHOLD
    )
    if direction_is_stable and signed >= 0:
        source_zone, target_zone = zone_a, zone_b
    elif direction_is_stable:
        source_zone, target_zone = zone_b, zone_a
    else:
        source_zone, target_zone = zone_a, zone_b
    source_price = float(node_prices[source_zone])
    target_price = float(node_prices[target_zone])
    mean_abs_price_spread = float(
        (prices[target_zone] - prices[source_zone]).abs().mean()
        if aggregation == "day"
        else abs(target_price - source_price)
    )
    magnitude = abs(signed)
    color, width, coupling = _verified_style(kind, magnitude)
    series = " | ".join(sorted(set(selected_rows["series"].astype(str))))
    label = "Flux vérifié" if kind == "flow" else "Capacité vérifiée"
    if magnitude <= _DIRECTION_EPSILON:
        direction_label = f"{label} nul"
    elif not direction_is_stable:
        direction_label = (
            f"Direction {label.lower()} instable · "
            f"stabilité {direction_stability:.0%}"
        )
    else:
        direction_label = f"{source_zone} → {target_zone} · signal {label.lower()}"
    tooltip = (
        f"{zone_a}-{zone_b}<br/>{direction_label}"
        f"<br/>{label}: {magnitude:.0f} MW"
        f"<br/>Prix P50: {source_zone} {source_price:.2f} · "
        f"{target_zone} {target_price:.2f} EUR/MWh"
        f"<br/>Source: {series}<br/>Cutoff causal vérifié"
    )
    return {
        "border": f"{zone_a}-{zone_b}",
        "source_zone": source_zone,
        "target_zone": target_zone,
        "status": kind,
        "status_label": label,
        "signal_kind": kind,
        "value": magnitude,
        "signed_value_canonical": signed,
        "mean_abs_value": mean_abs,
        "unit": "MW",
        "origin_price_eur_mwh": source_price,
        "destination_price_eur_mwh": target_price,
        "price_spread_eur_mwh": target_price - source_price,
        "mean_abs_price_spread_eur_mwh": mean_abs_price_spread,
        "direction_label": direction_label,
        "direction_stability": direction_stability,
        "direction_is_stable": direction_is_stable,
        "is_price_equilibrium": mean_abs_price_spread <= _DIRECTION_EPSILON,
        "arrow_glyph": (
            "▶"
            if magnitude > _DIRECTION_EPSILON and direction_is_stable
            else ""
        ),
        "series": series,
        "is_reported_flow": kind == "flow",
        "coupling_label": coupling,
        "color": color,
        "width": width,
        "tooltip": tooltip,
        **_edge_geometry(source_zone, target_zone),
    }


def build_market_coupling_view(
    bundle: PriceForecastBundle,
    *,
    selected_time: str | pd.Timestamp | None = None,
    aggregation: Literal["hour", "day"] = "hour",
    border_signals: pd.DataFrame | None = None,
) -> MarketCouplingView:
    """Derive an hour/day map without ever inferring a physical flow.

    Verified signals take precedence per border (flow, then capacity).  A daily
    verified signal is used only if every delivery hour has exactly one causal
    observation; incomplete series fall back to the explicitly labelled proxy.
    """

    if aggregation not in ("hour", "day"):
        raise MarketCouplingDataError("aggregation doit valoir 'hour' ou 'day'.")
    if not bundle.timeline_utc.equals(bundle.prices.index):
        raise MarketCouplingDataError("La timeline du bundle et celle des prix divergent.")
    missing_zones = set(MARKET_ZONES).difference(bundle.prices.columns)
    if missing_zones:
        raise MarketCouplingDataError(
            "Prix manquants pour la carte: " + ", ".join(sorted(missing_zones))
        )
    if not np.isfinite(bundle.prices.loc[:, MARKET_ZONES].to_numpy(dtype=float)).all():
        raise MarketCouplingDataError("Prix non finis dans le bundle.")

    if aggregation == "hour":
        selected = bundle.timeline_utc[0] if selected_time is None else _as_utc_timestamp(
            selected_time, field="selected_time"
        )
        if selected not in bundle.timeline_utc:
            raise MarketCouplingDataError("L'heure choisie n'appartient pas au jour publié.")
    else:
        if selected_time is not None:
            raise MarketCouplingDataError("selected_time doit être omis pour l'agrégat journalier.")
        selected = None

    if aggregation == "hour":
        assert selected is not None
        node_prices = bundle.prices.loc[selected]
    else:
        node_prices = bundle.prices.mean(axis=0)

    signals = _validate_border_signals(border_signals, bundle=bundle)
    edges: list[dict[str, Any]] = []
    selected_signal_rows: list[pd.DataFrame] = []
    for zone_a, zone_b in MARKET_BORDERS:
        pair_mask = (
            ((signals["zone_from"] == zone_a) & (signals["zone_to"] == zone_b))
            | ((signals["zone_from"] == zone_b) & (signals["zone_to"] == zone_a))
        )
        border_rows = signals.loc[pair_mask]
        verified_edge: dict[str, Any] | None = None
        for kind in ("flow", "capacity"):
            kind_rows = border_rows.loc[border_rows["kind"] == kind]
            if aggregation == "hour":
                kind_rows = kind_rows.loc[kind_rows["timestamp_utc"] == selected]
                complete = len(kind_rows) == 1
            else:
                kind_rows = kind_rows.loc[kind_rows["timestamp_utc"].isin(bundle.timeline_utc)]
                complete = (
                    len(kind_rows) == len(bundle.timeline_utc)
                    and kind_rows["timestamp_utc"].nunique() == len(bundle.timeline_utc)
                )
            if complete:
                verified_edge = _verified_edge(
                    zone_a,
                    zone_b,
                    selected_rows=kind_rows,
                    kind=kind,  # type: ignore[arg-type]
                    node_prices=node_prices,
                    prices=bundle.prices,
                    aggregation=aggregation,
                )
                selected_signal_rows.append(kind_rows)
                break
        if verified_edge is None:
            verified_edge = _proxy_edge(
                zone_a,
                zone_b,
                prices=bundle.prices,
                aggregation=aggregation,
                selected=selected,
            )
        edges.append(verified_edge)

    nodes: list[dict[str, Any]] = []
    for zone in MARKET_ZONES:
        lon, lat = ZONE_COORDINATES[zone]
        price = float(node_prices[zone])
        nodes.append(
            {
                "zone": zone,
                "position": [lon, lat],
                "longitude": lon,
                "latitude": lat,
                "price_eur_mwh": price,
                "badge_label": f"{zone}\n{price:.1f} EUR/MWh",
                "label": f"{zone}  {price:.1f} €",
                "tooltip": f"{zone}<br/>Prix P50 prévu: {price:.2f} EUR/MWh",
                "color": [15, 23, 42, 245],
            }
        )

    provenance = bundle.provenance.copy()
    if selected_signal_rows:
        selected_signals = pd.concat(selected_signal_rows, ignore_index=True)
        signal_provenance = (
            selected_signals.loc[:, ["zone_from", "zone_to", "kind", "series", "as_of_utc"]]
            .drop_duplicates()
            .assign(status=lambda frame: frame["kind"])
        )
        provenance = pd.concat([provenance, signal_provenance], ignore_index=True, sort=False)
    caveat = (
        "Les traits sans flèche représentent un écart de prix P50 prévu. "
        "La direction prix bas vers prix haut n'est indiquée dans le tableau "
        "que lorsqu'elle est stable. Ce n'est ni un flux physique, ni un programme "
        "d'échange, ni une capacité transfrontalière."
    )
    return MarketCouplingView(
        delivery_day=bundle.delivery_day,
        aggregation=aggregation,
        selected_time_utc=selected,
        nodes=pd.DataFrame(nodes),
        edges=pd.DataFrame(edges),
        provenance=provenance,
        caveat=caveat,
    )


def build_frontier_detail_table(view: MarketCouplingView) -> pd.DataFrame:
    """Return a price-first, auditable ranking for the panel table."""

    required = {
        "border",
        "source_zone",
        "target_zone",
        "status_label",
        "value",
        "unit",
        "series",
        "origin_price_eur_mwh",
        "destination_price_eur_mwh",
        "price_spread_eur_mwh",
        "mean_abs_price_spread_eur_mwh",
        "signed_value_canonical",
        "mean_abs_value",
        "direction_label",
        "direction_stability",
        "direction_is_stable",
    }
    missing = sorted(required.difference(view.edges.columns))
    if missing:
        raise MarketCouplingDataError(
            f"Colonnes de détail frontalier absentes: {missing}"
        )
    table = view.edges.loc[:, sorted(required)].copy()
    numeric = table[
        [
            "value",
            "origin_price_eur_mwh",
            "destination_price_eur_mwh",
            "price_spread_eur_mwh",
            "mean_abs_price_spread_eur_mwh",
            "signed_value_canonical",
            "mean_abs_value",
            "direction_stability",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise MarketCouplingDataError(
            "Le détail frontalier contient des prix ou signaux non finis."
        )
    table.loc[:, numeric.columns] = numeric
    computed_spread = (
        numeric["destination_price_eur_mwh"] - numeric["origin_price_eur_mwh"]
    )
    if not np.allclose(
        computed_spread.to_numpy(dtype=float),
        numeric["price_spread_eur_mwh"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise MarketCouplingDataError(
            "Le spread affiché ne correspond pas aux deux prix frontaliers."
        )
    table = table.sort_values(
        ["mean_abs_price_spread_eur_mwh", "border"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1, dtype=int))
    return table.loc[
        :,
        [
            "rank",
            "border",
            "direction_label",
            "source_zone",
            "origin_price_eur_mwh",
            "target_zone",
            "destination_price_eur_mwh",
            "price_spread_eur_mwh",
            "mean_abs_price_spread_eur_mwh",
            "direction_stability",
            "direction_is_stable",
            "status_label",
            "value",
            "signed_value_canonical",
            "mean_abs_value",
            "unit",
            "series",
        ],
    ]


def build_market_coupling_deck(view: MarketCouplingView) -> Any:
    """Build a native PyDeck chart from a validated coupling view."""

    try:
        import pydeck as pdk
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise MarketCouplingDataError("PyDeck est requis pour afficher la carte.") from exc

    edge_records = view.edges.to_dict(orient="records")
    node_records = view.nodes.to_dict(orient="records")
    verified_direction_records = [
        record
        for record in edge_records
        if record.get("status") in {"flow", "capacity"}
        and bool(record.get("arrow_glyph"))
    ]
    layers = [
        pdk.Layer(
            "LineLayer",
            data=edge_records,
            get_source_position="source_position",
            get_target_position="target_position",
            get_color="color",
            get_width="width",
            # PyDeck treats unquoted strings as accessors.  The nested quotes
            # serialize a literal enum value instead of the invalid @@=pixels.
            width_units="'pixels'",
            width_min_pixels=1,
            width_max_pixels=6,
            pickable=True,
            auto_highlight=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=verified_direction_records,
            get_position="arrow_position",
            get_text="arrow_glyph",
            get_color="color",
            get_size=12,
            get_angle="arrow_angle",
            get_alignment_baseline="'center'",
            pickable=False,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=node_records,
            get_position="position",
            get_fill_color="color",
            get_radius=26000,
            radius_min_pixels=6,
            radius_max_pixels=10,
            stroked=True,
            get_line_color=[255, 255, 255, 235],
            line_width_min_pixels=1,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=node_records,
            get_position="position",
            get_text="badge_label",
            get_color=[15, 23, 42, 255],
            get_size=13,
            get_pixel_offset=[0, -24],
            get_alignment_baseline="'bottom'",
            get_text_anchor="'middle'",
            background=True,
            get_background_color=[255, 255, 255, 238],
            background_padding=[7, 5],
            billboard=True,
            pickable=False,
        ),
    ]
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            longitude=4.0,
            latitude=47.6,
            zoom=4.25,
            min_zoom=3.5,
            max_zoom=8,
            pitch=0,
        ),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"html": "{tooltip}", "style": {"backgroundColor": "#0f172a", "color": "white"}},
    )


def render_market_coupling_panel(
    st: Any,
    project_root: str | Path,
    *,
    delivery_day: str | date | None = None,
    border_signals: pd.DataFrame | None = None,
    key_prefix: str = "market_coupling",
) -> MarketCouplingView:
    """Render the standalone Streamlit panel and return its underlying view."""

    bundle = load_live_price_forecasts(project_root, delivery_day=delivery_day)
    control_left, control_right = st.columns(
        [1, 2], vertical_alignment="bottom"
    )
    with control_left:
        mode_label = st.segmented_control(
            "Période affichée",
            options=["Heure", "Journée"],
            default="Heure",
            key=f"{key_prefix}_aggregation",
        )
    aggregation: Literal["hour", "day"] = (
        "day" if mode_label == "Journée" else "hour"
    )
    selected: pd.Timestamp | None = None
    with control_right:
        if aggregation == "hour":
            labels = {
                timestamp: timestamp.tz_convert("Europe/Paris").strftime(
                    "%d/%m · %H:%M (%Z)"
                )
                for timestamp in bundle.timeline_utc
            }
            selected = st.selectbox(
                "Heure de livraison",
                options=list(bundle.timeline_utc),
                format_func=lambda value: labels[value],
                key=f"{key_prefix}_hour",
            )
        else:
            st.caption(
                f"Moyenne P50 sur les {len(bundle.timeline_utc)} heures du "
                f"{bundle.delivery_day.strftime('%d/%m/%Y')}."
            )
    view = build_market_coupling_view(
        bundle,
        selected_time=selected,
        aggregation=aggregation,
        border_signals=border_signals,
    )
    details = build_frontier_detail_table(view)
    st.caption(
        f"Jour de livraison : {bundle.delivery_day.strftime('%d/%m/%Y')} · "
        f"Cutoff commun : {bundle.data_as_of_utc.strftime('%d/%m/%Y %H:%M UTC')}"
    )

    lowest = view.nodes.nsmallest(1, "price_eur_mwh").iloc[0]
    highest = view.nodes.nlargest(1, "price_eur_mwh").iloc[0]
    widest = details.iloc[0]
    with st.container(horizontal=True):
        st.metric(
            f"Prix le plus bas · {lowest['zone']}",
            f"{float(lowest['price_eur_mwh']):.2f} EUR/MWh",
            border=True,
        )
        st.metric(
            f"Prix le plus haut · {highest['zone']}",
            f"{float(highest['price_eur_mwh']):.2f} EUR/MWh",
            border=True,
        )
        st.metric(
            f"Plus grand écart · {widest['border']}",
            f"{float(widest['mean_abs_price_spread_eur_mwh']):.2f} EUR/MWh",
            border=True,
        )

    with st.container(border=True):
        st.markdown("**Comment lire la carte**")
        st.caption(
            "Chaque badge indique le prix P50 prévu du pays. Une ligne sans "
            "flèche compare deux prix voisins : ce n'est pas un flux physique. "
            "Pour les proxies, la direction prix bas → prix haut apparaît "
            "uniquement dans le tableau détaillé."
        )
        legend_columns = st.columns(3)
        with legend_columns[0]:
            st.markdown(":green-badge[≤ 2 EUR/MWh] Prix très proches")
        with legend_columns[1]:
            st.markdown(":orange-badge[> 2 à 10 EUR/MWh] Écart intermédiaire")
        with legend_columns[2]:
            st.markdown(":red-badge[> 10 EUR/MWh] Écart important")
        st.caption(
            "La couleur et l'épaisseur décrivent seulement l'écart absolu de "
            "prix. Une flèche est réservée à un flux ou une capacité causalement "
            "vérifié(e), jamais à un proxy de prix."
        )

    st.pydeck_chart(build_market_coupling_deck(view), width="stretch")
    proxy_count = int((view.edges["status"] == "proxy").sum())
    if proxy_count:
        st.warning(
            f"{proxy_count}/{len(view.edges)} frontières utilisent un proxy de prix. "
            f"{view.caveat}"
        )
    else:
        st.success(
            "Toutes les frontières affichées disposent d'un signal causal vérifié."
        )

    st.markdown("**Frontières classées par écart de prix**")
    if aggregation == "day":
        st.caption(
            "Classement sur l'écart absolu moyen des heures. La direction "
            f"journalière n'est affichée que si au moins "
            f"{DIRECTION_STABILITY_THRESHOLD:.0%} des heures partagent son signe."
        )
    else:
        st.caption(
            "Classement décroissant sur |prix destination − prix origine|. "
            "Pour un proxy horaire, origine = prix le plus bas et destination "
            "= prix le plus haut."
        )
    source_title = "Zone A" if aggregation == "day" else "Origine"
    target_title = "Zone B" if aggregation == "day" else "Destination"
    st.dataframe(
        details,
        hide_index=True,
        width="stretch",
        height=280,
        column_config={
            "rank": st.column_config.NumberColumn("Rang", format="%d"),
            "border": st.column_config.TextColumn("Frontière", pinned=True),
            "direction_label": st.column_config.TextColumn("Lecture"),
            "source_zone": st.column_config.TextColumn(source_title),
            "origin_price_eur_mwh": st.column_config.NumberColumn(
                "Prix origine", format="%.2f EUR/MWh"
            ),
            "target_zone": st.column_config.TextColumn(target_title),
            "destination_price_eur_mwh": st.column_config.NumberColumn(
                "Prix destination", format="%.2f EUR/MWh"
            ),
            "price_spread_eur_mwh": st.column_config.NumberColumn(
                "Spread P50 signé", format="%+.2f EUR/MWh"
            ),
            "mean_abs_price_spread_eur_mwh": st.column_config.NumberColumn(
                "Écart prix absolu moyen", format="%.2f EUR/MWh"
            ),
            "direction_stability": st.column_config.NumberColumn(
                "Stabilité direction", format="percent"
            ),
            "direction_is_stable": None,
            "status_label": st.column_config.TextColumn("Nature du signal"),
            "value": st.column_config.NumberColumn(
                "Valeur signal", format="%.2f"
            ),
            "unit": st.column_config.TextColumn("Unité signal"),
            "signed_value_canonical": st.column_config.NumberColumn(
                "Signal signé moyen", format="%+.2f"
            ),
            "mean_abs_value": st.column_config.NumberColumn(
                "Signal absolu moyen", format="%.2f"
            ),
            "series": st.column_config.TextColumn("Source / série"),
        },
    )
    with st.expander("Provenance et règles d'audit"):
        st.dataframe(view.provenance, hide_index=True, width="stretch")
        st.caption(
            "Priorité : flux causal vérifié, puis capacité causale vérifiée, "
            "puis proxy de spread P50."
        )
    return view


__all__ = [
    "MARKET_BORDERS",
    "MARKET_ZONES",
    "MarketCouplingDataError",
    "MarketCouplingView",
    "PriceForecastBundle",
    "build_frontier_detail_table",
    "build_market_coupling_deck",
    "build_market_coupling_view",
    "discover_latest_common_price_day",
    "load_live_price_forecasts",
    "render_market_coupling_panel",
]
