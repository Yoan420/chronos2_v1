"""Causal topology context for the five supported day-ahead zones.

This is a small, clean-room adaptation of the sparse graph prior described in
PriceFM (Yu et al., arXiv:2508.04875).  It does not reproduce PriceFM's neural
architecture.  Instead, it applies the paper's binary radius mask before a
deterministic cross-sectional mean, producing a compact context suitable for
the existing hourly residual-correction workflow.

The upstream caller remains responsible for selecting and sealing the
point-in-time (PIT) residual-load vintages.  This module never resamples,
interpolates, forward-fills, or backward-fills an input.  Missing observations
stay missing and are made auditable through pool counts and coverage ratios.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np
import pandas as pd

from chronos2_hourly.features import (
    HourlyFeatureContractError,
    build_calendar_features,
    validate_utc_hourly_index,
)


TOPOLOGY_CONTEXT_SCHEMA_VERSION: Final[str] = "topology-context-v1"
TOPOLOGY_ZONES: Final[tuple[str, ...]] = ("FR", "DE", "BE", "NL", "ES")

# Induced subgraph of PriceFM Appendix Table V.  DE is the project's compact
# name for the DE-LU bidding zone.  Edges leaving the five-zone universe are
# deliberately absent: unavailable nodes must not be invented or imputed.
DIRECT_NEIGHBORS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "FR": ("DE", "BE", "ES"),
        "DE": ("FR", "BE", "NL"),
        "BE": ("FR", "DE", "NL"),
        "NL": ("DE", "BE"),
        "ES": ("FR",),
    }
)

ZONE_TIMEZONES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "FR": "Europe/Paris",
        "DE": "Europe/Berlin",
        "BE": "Europe/Brussels",
        "NL": "Europe/Amsterdam",
        "ES": "Europe/Madrid",
    }
)

POOL_FIELDS: Final[tuple[str, ...]] = (
    "local",
    "pool_mean",
    "pool_count",
    "pool_coverage",
    "local_minus_pool",
)
TOPOLOGY_SIGNALS: Final[tuple[str, ...]] = (
    "residual_load",
    "price_da_lag24h",
)
TOPOLOGY_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"topology__{signal}__{field}"
    for signal in TOPOLOGY_SIGNALS
    for field in POOL_FIELDS
)
CALENDAR_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "calendar_local_hour",
    "calendar_weekday",
    "calendar_is_weekend",
    "calendar_hour_sin",
    "calendar_hour_cos",
    "calendar_weekday_sin",
    "calendar_weekday_cos",
    "calendar_dayofyear_sin",
    "calendar_dayofyear_cos",
    "calendar_dst_fold",
    "calendar_is_dst",
    "calendar_utc_offset_hours",
)
TOPOLOGY_CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    *TOPOLOGY_FEATURE_COLUMNS,
    *CALENDAR_FEATURE_COLUMNS,
)

_FORBIDDEN_SOURCE = re.compile(r"(?i)(?:storm|mk[\s_-]*online)")


class TopologyContextError(ValueError):
    """Raised when a causal topology-context contract is violated."""


@dataclass(frozen=True)
class TopologyContextMetadata:
    """Stable, JSON-friendly description attached to every context frame."""

    schema_version: str
    target_zone: str
    radius: int
    neighbours: tuple[str, ...]
    included_zones: tuple[str, ...]
    timezone: str
    residual_load_source: str = "sealed_pit"
    price_lag_hours: int = 24
    missing_policy: str = "native_nan_no_interpolation"

    def as_dict(self) -> dict[str, object]:
        """Return metadata using lists for portable JSON serialization."""

        payload = asdict(self)
        payload["neighbours"] = list(self.neighbours)
        payload["included_zones"] = list(self.included_zones)
        return payload


def _normalise_zone(zone: str) -> str:
    if not isinstance(zone, str):
        raise TypeError("target_zone doit être une chaîne.")
    result = zone.strip().upper()
    if result not in TOPOLOGY_ZONES:
        raise TopologyContextError(
            f"Zone inconnue: {zone!r}; zones admises: {list(TOPOLOGY_ZONES)}."
        )
    return result


def zones_within_radius(target_zone: str, radius: int) -> tuple[str, ...]:
    """Return target plus direct neighbors for the supported radii 0 and 1."""

    zone = _normalise_zone(target_zone)
    if isinstance(radius, (bool, np.bool_)) or radius not in (0, 1):
        raise TopologyContextError("radius doit valoir exactement 0 ou 1.")
    if radius == 0:
        return (zone,)
    selected = {zone, *DIRECT_NEIGHBORS[zone]}
    # Target first, then the stable project-wide zone order.
    return (zone, *(item for item in TOPOLOGY_ZONES if item in selected and item != zone))


def topology_mask(target_zone: str, radius: int) -> pd.Series:
    """Return the binary PriceFM-style mask on the five-zone universe."""

    included = set(zones_within_radius(target_zone, radius))
    return pd.Series(
        [1 if zone in included else 0 for zone in TOPOLOGY_ZONES],
        index=pd.Index(TOPOLOGY_ZONES, name="zone"),
        dtype="int8",
        name="topology_mask",
    )


def _validate_input_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    required_zones: tuple[str, ...],
    expected_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} doit être un pandas.DataFrame.")
    if frame.empty:
        raise TopologyContextError(f"{name} est vide.")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise TopologyContextError(
            f"{name} contient des colonnes dupliquées: {duplicates}."
        )
    if any(not isinstance(column, str) for column in frame.columns):
        raise TypeError(f"Les colonnes de {name} doivent être des chaînes.")
    forbidden = [column for column in frame.columns if _FORBIDDEN_SOURCE.search(column)]
    if forbidden:
        raise TopologyContextError(
            f"{name} contient une source interdite Storm/MKOnline: {forbidden}."
        )
    unexpected = [column for column in frame.columns if column not in TOPOLOGY_ZONES]
    if unexpected:
        raise TopologyContextError(
            f"Colonnes de zone inconnues dans {name}: {unexpected}."
        )
    missing = [zone for zone in required_zones if zone not in frame.columns]
    if missing:
        raise TopologyContextError(
            f"Zones requises absentes de {name}: {missing}; aucun fallback implicite."
        )

    try:
        index = validate_utc_hourly_index(frame.index, name=f"{name}.index")
    except HourlyFeatureContractError as exc:
        raise TopologyContextError(str(exc)) from exc
    if expected_index is not None and not index.equals(expected_index):
        raise TopologyContextError(
            f"{name}.index doit être exactement égal à residual_load_pit.index."
        )

    invalid_types = [
        zone
        for zone in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[zone])
        or pd.api.types.is_bool_dtype(frame[zone])
    ]
    if invalid_types:
        raise TypeError(
            f"{name} doit contenir uniquement des valeurs numériques; "
            f"colonnes invalides: {invalid_types}."
        )
    result = frame.loc[:, list(required_zones)].astype(float).copy(deep=True)
    if bool(np.isinf(result.to_numpy(dtype=float, copy=False)).any()):
        raise TopologyContextError(f"{name} contient des valeurs infinies.")
    return result


def _physical_price_lag_24h(prices: pd.DataFrame) -> pd.DataFrame:
    """Lag by exactly 24 UTC hours and enforce joint day-ahead causality.

    On the 25-hour autumn day, a physical lag of 24 hours can point into the
    same local delivery day.  That value was not known for a joint day-ahead
    decision, so it is explicitly masked rather than silently substituted.
    """

    lagged = prices.shift(24)
    for zone in prices.columns:
        local = prices.index.tz_convert(ZONE_TIMEZONES[zone])
        # Keep a timezone-aware datetime dtype so comparisons with the first
        # 24 NaT sources are vectorized and well-defined (object dates would
        # attempt to compare ``None`` with ``datetime.date``).
        local_days = pd.Series(local.normalize(), index=prices.index)
        source_days = local_days.shift(24)
        strictly_earlier_day = source_days.notna() & source_days.lt(local_days)
        lagged[zone] = lagged[zone].where(strictly_earlier_day)
    return lagged


def _pool_signal(
    values: pd.DataFrame,
    *,
    target_zone: str,
    signal: str,
) -> pd.DataFrame:
    count = values.notna().sum(axis=1).astype(float)
    total = values.sum(axis=1, min_count=1)
    mean = total.div(count.where(count.gt(0.0)))
    local = values[target_zone].astype(float)
    result = pd.DataFrame(
        {
            f"topology__{signal}__local": local,
            f"topology__{signal}__pool_mean": mean,
            f"topology__{signal}__pool_count": count,
            f"topology__{signal}__pool_coverage": count / float(values.shape[1]),
            f"topology__{signal}__local_minus_pool": local - mean,
        },
        index=values.index.copy(),
    )
    return result


def build_topology_context(
    residual_load_pit: pd.DataFrame,
    day_ahead_prices: pd.DataFrame,
    *,
    target_zone: str,
    radius: int,
    timezone: str | None = None,
) -> pd.DataFrame:
    """Build the fixed 22-column causal context for one target zone.

    Parameters
    ----------
    residual_load_pit:
        Point-in-time residual-load forecasts, already selected and sealed by
        the upstream archive contract.  Columns are zone codes.
    day_ahead_prices:
        Realized day-ahead price history on the same continuous UTC timeline.
        Only a guarded physical 24-hour lag is exposed to the result.
    target_zone, radius:
        Radius is deliberately limited to the two protocol arms: local-only
        (0) and local plus directly connected zones (1).

    Notes
    -----
    Pooling ignores missing nodes only cross-sectionally at the same delivery
    timestamp.  The explicit count and coverage features reveal this choice;
    no value is ever borrowed from another timestamp.
    """

    zone = _normalise_zone(target_zone)
    included = zones_within_radius(zone, radius)
    residual = _validate_input_frame(
        residual_load_pit,
        name="residual_load_pit",
        required_zones=included,
    )
    prices = _validate_input_frame(
        day_ahead_prices,
        name="day_ahead_prices",
        required_zones=included,
        expected_index=residual.index,
    )

    selected_timezone = str(timezone or ZONE_TIMEZONES[zone])
    try:
        calendar = build_calendar_features(residual.index, timezone=selected_timezone)
    except HourlyFeatureContractError as exc:
        raise TopologyContextError(str(exc)) from exc
    calendar = calendar.loc[:, list(CALENDAR_FEATURE_COLUMNS)]

    price_lag = _physical_price_lag_24h(prices)
    residual_pool = _pool_signal(
        residual,
        target_zone=zone,
        signal="residual_load",
    )
    price_pool = _pool_signal(
        price_lag,
        target_zone=zone,
        signal="price_da_lag24h",
    )
    result = pd.concat([residual_pool, price_pool, calendar], axis=1, copy=False)
    result = result.loc[:, list(TOPOLOGY_CONTEXT_COLUMNS)]
    result.index.name = residual.index.name or "delivery_start_utc"

    neighbours = tuple(item for item in included if item != zone)
    metadata = TopologyContextMetadata(
        schema_version=TOPOLOGY_CONTEXT_SCHEMA_VERSION,
        target_zone=zone,
        radius=int(radius),
        neighbours=neighbours,
        included_zones=included,
        timezone=selected_timezone,
    )
    result.attrs["topology_context"] = metadata.as_dict()
    result.attrs["topology_missing_audit"] = {
        "residual_load_input": {
            item: int(residual[item].isna().sum()) for item in included
        },
        "price_input": {item: int(prices[item].isna().sum()) for item in included},
        "price_lag24": {
            item: int(price_lag[item].isna().sum()) for item in included
        },
        "output": {column: int(result[column].isna().sum()) for column in result},
    }
    return result


__all__ = [
    "CALENDAR_FEATURE_COLUMNS",
    "DIRECT_NEIGHBORS",
    "POOL_FIELDS",
    "TOPOLOGY_CONTEXT_COLUMNS",
    "TOPOLOGY_CONTEXT_SCHEMA_VERSION",
    "TOPOLOGY_FEATURE_COLUMNS",
    "TOPOLOGY_SIGNALS",
    "TOPOLOGY_ZONES",
    "TopologyContextError",
    "TopologyContextMetadata",
    "ZONE_TIMEZONES",
    "build_topology_context",
    "topology_mask",
    "zones_within_radius",
]
