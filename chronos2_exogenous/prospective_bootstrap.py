"""Read-only planning of resumable retrospective LoRA calibration.

A missing auction observation is an expected wait state, not permission to
interpolate labels or advance beyond a gap.  Each country progresses separately.
This planner neither performs inference nor changes any sealed trial artifact.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


class BootstrapPlanningError(ValueError):
    """The requested plan or supplied snapshots violate the hourly contract."""


def _civil_day(value: str) -> pd.Timestamp:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise BootstrapPlanningError("end_day doit etre une date civile YYYY-MM-DD.")
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise BootstrapPlanningError("end_day doit etre une date civile valide.") from exc


def _utc_index(values: Any, *, description: str) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(values)
    except (ValueError, TypeError) as exc:
        raise BootstrapPlanningError(f"{description}: timestamps invalides.") from exc
    # Never silently interpret naive local timestamps as UTC (notably at DST).
    if index.tz is None or index.hasnans or index.has_duplicates:
        raise BootstrapPlanningError(
            f"{description}: timestamps uniques, explicites en timezone et non nuls requis.")
    index = index.tz_convert("UTC")
    if not index.equals(index.floor("h")):
        raise BootstrapPlanningError(f"{description}: timestamps non horaires.")
    return index


def _physical_hours(day: pd.Timestamp, *, timezone: str) -> pd.DatetimeIndex:
    # Localize the two civil midnights separately: a delivery day is 23/24/25h.
    return pd.date_range(day.tz_localize(timezone),
                         (day + pd.Timedelta(days=1)).tz_localize(timezone),
                         freq="h", inclusive="left").tz_convert("UTC")


def plan_bootstrap(
    histories: Mapping[str, pd.DataFrame],
    targets: Mapping[str, pd.Series],
    end_day: str,
    *,
    timezone: str = "Europe/Paris",
) -> dict[str, Any]:
    """Plan each country's contiguous, fully observed, not-yet-sealed suffix.

    ``histories`` contains existing raw predictions, whose physical hours must
    form complete contiguous delivery days. Historical labels and prediction
    values are validated by the trial/auxiliary contracts, not by this planner.
    ``targets`` contains fresh canonical hourly observations. A suffix stops at
    its first day without all finite observations; other countries can continue.

    ``pending`` reports only that first blocking day per country, with exact UTC
    missing hours. Later days are deliberately not scheduled across the gap.
    Countries already reaching ``end_day`` need no new target snapshot.
    """
    end = _civil_day(end_day)
    try:
        _physical_hours(end, timezone=timezone)
    except (ValueError, TypeError, KeyError) as exc:
        raise BootstrapPlanningError(f"Timezone invalide: {timezone}.") from exc
    if not histories:
        raise BootstrapPlanningError("Aucun historique de pays fourni.")
    ready: dict[str, list[str]] = {}
    pending: list[dict[str, Any]] = []
    already_complete: list[str] = []
    for zone, history in histories.items():
        if not isinstance(zone, str) or not zone:
            raise BootstrapPlanningError("Identifiant de pays invalide.")
        if not isinstance(history, pd.DataFrame) or history.empty or "delivery_start_utc" not in history:
            raise BootstrapPlanningError(f"{zone}: historique horaire absent ou vide.")
        index = _utc_index(history["delivery_start_utc"], description=f"{zone}/historique")
        local = index.tz_convert(timezone)
        first, last = pd.Timestamp(local.min().date()), pd.Timestamp(local.max().date())
        expected = pd.date_range(first.tz_localize(timezone),
                                 (last + pd.Timedelta(days=1)).tz_localize(timezone),
                                 freq="h", inclusive="left").tz_convert("UTC")
        if not index.sort_values().equals(expected):
            raise BootstrapPlanningError(
                f"{zone}: historique non contigu ou journee physique incomplete; reprise refusee.")
        ready[zone] = []
        if last >= end:
            already_complete.append(zone)
            continue
        if zone not in targets or not isinstance(targets[zone], pd.Series):
            raise BootstrapPlanningError(f"{zone}: snapshot canonique des observations absent.")
        target = targets[zone]
        if target.empty:
            observed = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        else:
            target_index = _utc_index(target.index, description=f"{zone}/observations")
            try:
                observed = pd.Series(target.to_numpy(dtype=float, na_value=np.nan), index=target_index)
            except (ValueError, TypeError) as exc:
                raise BootstrapPlanningError(f"{zone}: observations non numeriques.") from exc
        for delivery in pd.date_range(last + pd.Timedelta(days=1), end, freq="D"):
            hours = _physical_hours(delivery, timezone=timezone)
            available = np.isfinite(observed.reindex(hours).to_numpy(dtype=float))
            count = int(available.sum())
            if count != len(hours):
                pending.append({
                    "zone": zone,
                    "delivery_day": delivery.date().isoformat(),
                    "observed_hours": count,
                    "expected_hours": len(hours),
                    "missing_hours_utc": [hour.isoformat() for hour in hours[~available]],
                    "reason": "observations_unavailable" if count == 0 else "observations_incomplete",
                })
                break
            ready[zone].append(delivery.date().isoformat())
    return {"ready_by_zone": ready, "pending": pending, "already_complete": already_complete}
