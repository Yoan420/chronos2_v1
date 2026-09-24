"""Explicit EPEX price reference for report-only observations and scoring.

The existing Saturn transport supplies these series. This module never fetches
training targets, blends sources, fills gaps or changes a model input.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_modular.saturn import fetch_saturn_series_from_client


EPEX_REPORTING_SERIES_BY_ZONE: Mapping[str, str] = {
    "BE": "power.price.be.euromwh.h.obs.epex",
    "DE": "power.price.de.euromwh.h.obs.epex",
    "FR": "power.price.fr.euromwh.h.obs.epex",
    "NL": "power.price.nl.euromwh.h.obs.epex",
}
EPEX_REPORTING_SOURCE_KIND = "saturn_epex_latest_extraction"
EPEX_REPORTING_POLICY = "epex_only_v1"


class ReportingObservationError(ValueError):
    """The source cannot be used as an auditable hourly price reference."""


def epex_reporting_identity(zone: str, timezone: str) -> dict[str, Any]:
    """Return the fixed source identity; config cannot redirect an EPEX zone."""
    if zone not in EPEX_REPORTING_SERIES_BY_ZONE:
        raise ReportingObservationError(f"{zone}: aucune reference EPEX configuree.")
    return {
        "kind": EPEX_REPORTING_SOURCE_KIND, "policy": EPEX_REPORTING_POLICY,
        "provider": "Saturn", "origin": "EPEX", "actual_reference": "EPEX",
        "label": "EPEX", "series": EPEX_REPORTING_SERIES_BY_ZONE[zone],
        "zone": zone, "timezone": timezone, "frequency": "h", "price_unit": "EUR/MWh",
        "fallback_policy": "none", "source_mixing": False,
        "reference_role": "reporting_and_scoring_only",
        "nocache": True, "live_recomputation": True, "used_for_prediction": False,
    }


def fetch_epex_reporting_observations(
    client: Any, *, zone: str, timezone: str, expected_index: pd.DatetimeIndex,
    extracted_at_utc: pd.Timestamp,
) -> tuple[pd.Series, dict[str, Any]]:
    """Read the whole reporting horizon from one EPEX hourly series only."""
    source = epex_reporting_identity(zone, timezone)
    expected = pd.DatetimeIndex(expected_index)
    if (expected.tz is None or expected.empty or expected.hasnans
            or expected.has_duplicates or not expected.is_monotonic_increasing):
        raise ReportingObservationError("Timeline de reporting UTC unique et ordonnee requise.")
    expected = expected.tz_convert("UTC")
    if (not expected.equals(expected.floor("h"))
            or not expected.equals(pd.date_range(expected[0], expected[-1], freq="h"))):
        raise ReportingObservationError("Une grille horaire physique complete est requise.")
    extracted = pd.Timestamp(extracted_at_utc)
    if pd.isna(extracted) or extracted.tzinfo is None:
        raise ReportingObservationError("Date d'extraction UTC explicite requise.")
    extracted = extracted.tz_convert("UTC")
    requested_start = expected[0] - pd.Timedelta(hours=2)
    requested_end = expected[-1] + pd.Timedelta(hours=2)
    raw = fetch_saturn_series_from_client(
        client, source["series"], requested_start, requested_end, timezone,
        naive_timezone="UTC", nocache=True, live=True,
    )
    if not isinstance(raw, pd.Series):
        raise ReportingObservationError(f"{zone}: les observations EPEX doivent etre une Series.")
    index = pd.DatetimeIndex(raw.index)
    if (index.tz is None or index.hasnans or index.has_duplicates
            or not index.is_monotonic_increasing):
        raise ReportingObservationError(f"{zone}: timeline EPEX physique unique et ordonnee requise.")
    index = index.tz_convert("UTC")
    if not index.equals(index.floor("h")):
        raise ReportingObservationError(f"{zone}: observations EPEX hors grille horaire.")
    try:
        numeric = pd.to_numeric(raw, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ReportingObservationError(f"{zone}: prix EPEX non numerique.") from error
    if np.isinf(numeric).any():
        raise ReportingObservationError(f"{zone}: prix EPEX infini interdit.")
    values = pd.Series(numeric, index=index, name="actual")
    available = values.loc[np.isfinite(numeric)]
    source.update({
        "extracted_at_utc": str(extracted),
        "requested_from_utc": str(requested_start), "requested_to_utc": str(requested_end),
        "first_available_observation_utc": str(available.index[0]) if len(available) else None,
        "last_available_observation_utc": str(available.index[-1]) if len(available) else None,
        "selection": "latest EPEX hourly values returned by Saturn for the entire reporting horizon",
    })
    return values, source


__all__ = [
    "EPEX_REPORTING_SERIES_BY_ZONE", "EPEX_REPORTING_SOURCE_KIND", "EPEX_REPORTING_POLICY",
    "ReportingObservationError", "epex_reporting_identity", "fetch_epex_reporting_observations",
]
