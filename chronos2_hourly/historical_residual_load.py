"""Causal historical replay of the five Chronos-2 residual-load inputs.

The live residual-load provider forecasts one delivery day from observations
available at a runtime cutoff.  This module applies the same contract to a
sequence of historical civil days without contacting Saturn:

* callers provide the five immutable Saturn ``.obs`` vintage frames;
* each request is made at 08:00 Europe/Paris on D-1;
* both ``snapshot_time_utc`` and ``revision_time_utc`` must be available by
  that request cutoff;
* Chronos receives exactly 2,048 consecutive hourly observations;
* full 23/24/25-hour local delivery days are preserved.

The resulting long table is deliberately self-auditing and can be converted
directly to the five PIT parquet schemas consumed by the hourly price model.
No network operation is performed unless the caller lets this module create
the default Chronos-2 pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import chronos_residual_load as live_provider
from .hourly_contract import local_delivery_day_index


DELIVERY_TIMEZONE = live_provider.DELIVERY_TIMEZONE
MODEL_ID = live_provider.MODEL_ID
MODEL_REVISION = live_provider.MODEL_REVISION
CONTEXT_LENGTH = live_provider.CONTEXT_LENGTH
MAX_INTERNAL_GAP_HOURS = live_provider.MAX_INTERNAL_GAP_HOURS
QUANTILE_LEVELS = live_provider.QUANTILE_LEVELS
COUNTRY_OBSERVED_SERIES = live_provider.COUNTRY_OBSERVED_SERIES
COUNTRY_ALIASES = live_provider.COUNTRY_ALIASES
EXPECTED_ALIASES = live_provider.EXPECTED_ALIASES

SCHEMA_VERSION = 1
CHECKPOINT_KIND = "chronos2_historical_residual_load_checkpoint"

PREDICTION_COLUMNS = (
    "delivery_day_local",
    "delivery_hours",
    "alias",
    "source_series",
    "value_time_utc",
    "snapshot_time_utc",
    "revision_time_utc",
    "request_cutoff_utc",
    "model_origin_utc",
    "context_start_utc",
    "context_end_utc",
    "context_rows",
    "maximum_selected_snapshot_time_utc",
    "maximum_selected_revision_time_utc",
    "imputed_hours",
    "bridge_horizon_hours",
    "lead_to_delivery_start_hours",
    "value",
    "q10",
    "q50",
    "q90",
)

AUDIT_COLUMNS = (
    "delivery_day_local",
    "delivery_hours",
    "alias",
    "source_series",
    "request_cutoff_utc",
    "model_origin_utc",
    "context_start_utc",
    "context_end_utc",
    "context_rows",
    "maximum_selected_snapshot_time_utc",
    "maximum_selected_revision_time_utc",
    "imputed_hours",
    "bridge_horizon_hours",
    "lead_to_delivery_start_hours",
)


class HistoricalResidualLoadError(live_provider.ResidualLoadBundleError):
    """Raised when a historical replay would violate its causal contract."""


@dataclass(frozen=True)
class HistoricalDeliveryPlan:
    """One full local delivery day and its D-1 08:00 request cutoff."""

    delivery_day_local: date
    request_cutoff_utc: pd.Timestamp
    delivery_index_utc: pd.DatetimeIndex


@dataclass(frozen=True)
class HistoricalResidualLoadReplay:
    """Predictions, causal audits and any reused/published checkpoints."""

    predictions: pd.DataFrame
    audits: pd.DataFrame
    checkpoint_paths: tuple[Path, ...] = ()

    def to_pit_frames(self) -> dict[str, pd.DataFrame]:
        return replay_to_pit_frames(self.predictions)

    def to_wide(self, *, value_column: str = "q50") -> pd.DataFrame:
        return replay_to_wide(self.predictions, value_column=value_column)


DeliveryMask = (
    Callable[[date], bool]
    | Mapping[Any, Any]
    | pd.Series
    | Iterable[Any]
    | None
)


def _as_local_date(value: Any) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise HistoricalResidualLoadError("La date de livraison est NaT.")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(DELIVERY_TIMEZONE).tz_localize(None)
    return timestamp.date()


def _as_utc(value: Any, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise HistoricalResidualLoadError(
            f"{name} n'est pas un timestamp valide: {value!r}."
        ) from exc
    if pd.isna(timestamp):
        raise HistoricalResidualLoadError(f"{name} est NaT.")
    if timestamp.tzinfo is None:
        raise HistoricalResidualLoadError(
            f"{name} doit contenir un fuseau horaire explicite."
        )
    return timestamp.tz_convert("UTC")


def _request_cutoff_utc(delivery_day: date) -> pd.Timestamp:
    previous_day = delivery_day - pd.Timedelta(days=1)
    # Localize the 08:00 wall-clock label itself.  Adding eight absolute hours
    # to local midnight would yield 09:00 after the spring transition and
    # 07:00 after the autumn transition.
    cutoff_local = (
        pd.Timestamp(previous_day) + pd.Timedelta(hours=8)
    ).tz_localize(DELIVERY_TIMEZONE)
    return cutoff_local.tz_convert("UTC")


def _hour_mask_selection(
    mask: pd.Series,
    candidate_days: Sequence[date],
) -> set[date]:
    if not isinstance(mask.index, pd.DatetimeIndex):
        values = mask.astype(bool).to_numpy()
        if len(values) != len(candidate_days):
            raise HistoricalResidualLoadError(
                "Le masque booleen doit avoir une valeur par jour candidat."
            )
        return {
            day for day, selected in zip(candidate_days, values) if selected
        }

    index = pd.DatetimeIndex(mask.index)
    if index.tz is None:
        raise HistoricalResidualLoadError(
            "Un masque horaire doit avoir un DatetimeIndex timezone-aware."
        )
    normalized = pd.Series(
        mask.astype(bool).to_numpy(),
        index=index.tz_convert("UTC"),
    )
    if normalized.index.has_duplicates:
        raise HistoricalResidualLoadError(
            "Le masque horaire contient des timestamps dupliques."
        )

    selected_days: set[date] = set()
    for day in candidate_days:
        expected = local_delivery_day_index(day, timezone=DELIVERY_TIMEZONE)
        day_mask = normalized.reindex(expected)
        present = day_mask.notna()
        if not bool(present.any()):
            continue
        if not bool(present.all()):
            raise HistoricalResidualLoadError(
                f"Le masque ne couvre qu'une partie du jour civil {day}."
            )
        if bool(day_mask.all()):
            selected_days.add(day)
        elif bool(day_mask.any()):
            raise HistoricalResidualLoadError(
                f"Le masque horaire est partiel pour le jour civil {day}."
            )
    return selected_days


def _selected_delivery_days(
    candidate_days: Sequence[date],
    delivery_mask: DeliveryMask,
) -> set[date]:
    if delivery_mask is None:
        return set(candidate_days)
    if callable(delivery_mask):
        return {day for day in candidate_days if bool(delivery_mask(day))}
    if isinstance(delivery_mask, pd.Series):
        return _hour_mask_selection(delivery_mask, candidate_days)
    if isinstance(delivery_mask, Mapping):
        normalized = {
            _as_local_date(key): bool(value)
            for key, value in delivery_mask.items()
        }
        return {day for day in candidate_days if normalized.get(day, False)}

    values = list(delivery_mask)
    if len(values) == len(candidate_days) and all(
        isinstance(value, (bool, np.bool_)) for value in values
    ):
        return {
            day for day, selected in zip(candidate_days, values) if selected
        }
    selected = {_as_local_date(value) for value in values}
    return set(candidate_days).intersection(selected)


def build_historical_delivery_plans(
    start_day: str | date | pd.Timestamp,
    end_day: str | date | pd.Timestamp,
    *,
    delivery_mask: DeliveryMask = None,
) -> list[HistoricalDeliveryPlan]:
    """Plan full local delivery days with a D-1 08:00 Europe/Paris cutoff.

    A timezone-aware boolean ``Series`` can be supplied as an hourly mask.  A
    civil day is then either selected in full or rejected in full; partial
    23/24/25-hour days are errors.  Date iterables, boolean iterables, mappings
    and callables are also accepted.
    """

    start = _as_local_date(start_day)
    end = _as_local_date(end_day)
    if end < start:
        raise HistoricalResidualLoadError(
            "end_day doit etre posterieur ou egal a start_day."
        )
    candidate_days = [
        timestamp.date()
        for timestamp in pd.date_range(start=start, end=end, freq="D")
    ]
    selected = _selected_delivery_days(candidate_days, delivery_mask)
    plans: list[HistoricalDeliveryPlan] = []
    for day in candidate_days:
        if day not in selected:
            continue
        delivery_index = local_delivery_day_index(
            day, timezone=DELIVERY_TIMEZONE
        )
        cutoff = _request_cutoff_utc(day)
        if cutoff >= delivery_index[0]:
            raise AssertionError("Le cutoff D-1 doit preceder la livraison.")
        plans.append(
            HistoricalDeliveryPlan(
                delivery_day_local=day,
                request_cutoff_utc=cutoff,
                delivery_index_utc=delivery_index,
            )
        )
    return plans


def _validate_observed_series_name(series_name: str) -> None:
    lowered = str(series_name).strip().lower()
    if ".fcst" in lowered or not lowered.endswith(".obs"):
        raise HistoricalResidualLoadError(
            f"Serie interdite pour le replay Chronos-2: {series_name!r}; "
            "seules les observations .obs sont admises."
        )


def _normalize_observed_vintages(
    frame: pd.DataFrame,
    *,
    series_name: str,
) -> pd.DataFrame:
    _validate_observed_series_name(series_name)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{series_name}: vintages doit etre un DataFrame.")
    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise HistoricalResidualLoadError(
            f"{series_name}: colonnes PIT absentes: {missing}."
        )

    result = frame.copy().reset_index(drop=True)
    if "downloaded_at_utc" not in result:
        result["downloaded_at_utc"] = pd.NaT
    for column in (
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "downloaded_at_utc",
    ):
        result[column] = pd.to_datetime(
            result[column], utc=True, errors="coerce"
        )
    if result[
        ["value_time_utc", "snapshot_time_utc", "revision_time_utc"]
    ].isna().any().any():
        raise HistoricalResidualLoadError(
            f"{series_name}: timestamp PIT obligatoire absent ou invalide."
        )
    values = pd.DatetimeIndex(result["value_time_utc"])
    if any(
        getattr(values, field).any()
        for field in ("minute", "second", "microsecond", "nanosecond")
    ):
        raise HistoricalResidualLoadError(
            f"{series_name}: value_time_utc doit etre aligne a l'heure."
        )
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result["_row_order"] = np.arange(len(result), dtype=np.int64)

    timestamp_ns = np.column_stack(
        [
            result[column].astype("int64", copy=False).to_numpy()
            for column in (
                "value_time_utc",
                "snapshot_time_utc",
                "revision_time_utc",
            )
        ]
    )
    result["_eligible_at_utc"] = pd.to_datetime(
        timestamp_ns.max(axis=1), utc=True
    )
    return result.sort_values(
        ["_eligible_at_utc", "_row_order"], kind="stable"
    ).reset_index(drop=True)


def _selection_sort_columns() -> list[str]:
    return [
        "value_time_utc",
        "revision_time_utc",
        "snapshot_time_utc",
        "downloaded_at_utc",
        "_row_order",
    ]


def _latest_eligible_asof(
    normalized: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    eligible = normalized.loc[
        normalized["value_time_utc"].le(cutoff)
        & normalized["snapshot_time_utc"].le(cutoff)
        & normalized["revision_time_utc"].le(cutoff)
    ]
    return (
        eligible.sort_values(
            _selection_sort_columns(),
            kind="stable",
            na_position="first",
        )
        .drop_duplicates("value_time_utc", keep="last")
        .sort_values("value_time_utc", kind="stable")
        .reset_index(drop=True)
    )


def _prepare_selected_context(
    selected: pd.DataFrame,
    *,
    series_name: str,
    request_cutoff_utc: pd.Timestamp,
    context_length: int,
    maximum_gap_hours: int,
    discarded_after_cutoff: int,
) -> tuple[pd.Series, dict[str, Any]]:
    if selected.empty:
        raise HistoricalResidualLoadError(
            f"{series_name}: aucune observation disponible au cutoff "
            f"{request_cutoff_utc.isoformat()}."
        )
    numeric = pd.to_numeric(selected["value"], errors="coerce").astype(float)
    finite = np.isfinite(numeric.to_numpy())
    if not bool(finite.any()):
        raise HistoricalResidualLoadError(
            f"{series_name}: aucune observation finie au cutoff."
        )
    last_observation = pd.Timestamp(
        selected.loc[finite, "value_time_utc"].iloc[-1]
    ).tz_convert("UTC")
    if last_observation > request_cutoff_utc:
        raise AssertionError("Une observation future a traverse le filtre PIT.")

    desired_index = pd.date_range(
        end=last_observation,
        periods=context_length,
        freq="h",
        tz="UTC",
    )
    within = selected.loc[
        selected["value_time_utc"].between(
            desired_index[0], desired_index[-1], inclusive="both"
        )
    ].copy()
    source = pd.Series(
        pd.to_numeric(within["value"], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(within["value_time_utc"]),
        name=series_name,
        dtype=float,
    )
    source.loc[~np.isfinite(source.to_numpy())] = np.nan
    context = source.reindex(desired_index)
    gaps = live_provider._missing_runs(context.isna())
    too_long = [gap for gap in gaps if len(gap) > maximum_gap_hours]
    if too_long:
        first = too_long[0]
        raise HistoricalResidualLoadError(
            f"{series_name}: gap de {len(first)} heures dans le contexte "
            f"({first[0].isoformat()} -> {first[-1].isoformat()}), "
            f"maximum autorise={maximum_gap_hours}."
        )
    imputed_index = pd.DatetimeIndex(
        [timestamp for gap in gaps for timestamp in gap], tz="UTC"
    )
    if len(imputed_index):
        context = context.interpolate(method="time", limit_area="inside")
    if context.isna().any():
        raise HistoricalResidualLoadError(
            f"{series_name}: couverture insuffisante pour former exactement "
            f"{context_length} heures consecutives."
        )
    if len(context) != context_length:
        raise AssertionError("Longueur de contexte Chronos divergente.")

    audit_rows = within.loc[
        within["value_time_utc"].isin(desired_index)
    ]
    maximum_snapshot = pd.Timestamp(
        audit_rows["snapshot_time_utc"].max()
    ).tz_convert("UTC")
    maximum_revision = pd.Timestamp(
        audit_rows["revision_time_utc"].max()
    ).tz_convert("UTC")
    if maximum_snapshot > request_cutoff_utc or maximum_revision > request_cutoff_utc:
        raise AssertionError("Une vintage future a traverse le filtre PIT.")
    audit = {
        "context_start_utc": context.index[0],
        "context_end_utc": context.index[-1],
        "context_rows": len(context),
        "last_observation_utc": last_observation,
        "maximum_selected_snapshot_time_utc": maximum_snapshot,
        "maximum_selected_revision_time_utc": maximum_revision,
        "discarded_rows_after_cutoff": int(discarded_after_cutoff),
        "imputed_hours": len(imputed_index),
        "imputed_timestamps_utc": tuple(imputed_index),
        "interpolation": "linear_time_internal_only",
    }
    context.name = series_name
    return context.astype(float), audit


def select_observed_context_asof(
    vintages: pd.DataFrame,
    *,
    series_name: str,
    request_cutoff_utc: Any,
    context_length: int = CONTEXT_LENGTH,
    maximum_gap_hours: int = MAX_INTERNAL_GAP_HOURS,
) -> tuple[pd.Series, dict[str, Any]]:
    """Select and regularize one exact causal context from Saturn vintages."""

    if int(context_length) <= 0:
        raise ValueError("context_length doit etre strictement positif.")
    if int(context_length) != CONTEXT_LENGTH:
        raise ValueError(
            f"Le replay de comparaison exige context_length={CONTEXT_LENGTH}."
        )
    if int(maximum_gap_hours) < 0:
        raise ValueError("maximum_gap_hours doit etre positif ou nul.")
    if int(maximum_gap_hours) != MAX_INTERNAL_GAP_HOURS:
        raise ValueError(
            "Le replay de comparaison exige maximum_gap_hours="
            f"{MAX_INTERNAL_GAP_HOURS}."
        )
    cutoff = _as_utc(request_cutoff_utc, name="request_cutoff_utc")
    normalized = _normalize_observed_vintages(
        vintages, series_name=series_name
    )
    selected = _latest_eligible_asof(normalized, cutoff)
    discarded = int(
        (
            normalized["value_time_utc"].gt(cutoff)
            | normalized["snapshot_time_utc"].gt(cutoff)
            | normalized["revision_time_utc"].gt(cutoff)
        ).sum()
    )
    return _prepare_selected_context(
        selected,
        series_name=series_name,
        request_cutoff_utc=cutoff,
        context_length=int(context_length),
        maximum_gap_hours=int(maximum_gap_hours),
        discarded_after_cutoff=discarded,
    )


class _ObservedVintageCursor:
    """Incremental latest-vintage selector for monotonically rising cutoffs."""

    def __init__(self, frame: pd.DataFrame, *, series_name: str) -> None:
        self.series_name = series_name
        self.frame = frame
        self.position = 0
        self.last_cutoff: pd.Timestamp | None = None
        self.latest: dict[pd.Timestamp, tuple[tuple[int, int, int, int], dict[str, Any]]] = {}

    @staticmethod
    def _rank(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
        downloaded = pd.Timestamp(row["downloaded_at_utc"])
        downloaded_ns = (
            np.iinfo(np.int64).min if pd.isna(downloaded) else downloaded.value
        )
        return (
            pd.Timestamp(row["revision_time_utc"]).value,
            pd.Timestamp(row["snapshot_time_utc"]).value,
            int(downloaded_ns),
            int(row["_row_order"]),
        )

    def context_at(
        self,
        cutoff: pd.Timestamp,
        *,
        context_length: int,
        maximum_gap_hours: int,
    ) -> tuple[pd.Series, dict[str, Any]]:
        if self.last_cutoff is not None and cutoff < self.last_cutoff:
            raise HistoricalResidualLoadError(
                "Les cutoffs du replay doivent etre croissants."
            )
        self.last_cutoff = cutoff
        while self.position < len(self.frame):
            row = self.frame.iloc[self.position]
            if pd.Timestamp(row["_eligible_at_utc"]) > cutoff:
                break
            payload = row.to_dict()
            value_time = pd.Timestamp(payload["value_time_utc"])
            rank = self._rank(payload)
            previous = self.latest.get(value_time)
            if previous is None or rank >= previous[0]:
                self.latest[value_time] = (rank, payload)
            self.position += 1
        if not self.latest:
            selected = self.frame.iloc[0:0].copy()
        else:
            selected = pd.DataFrame(
                [item[1] for item in self.latest.values()]
            ).sort_values("value_time_utc", kind="stable")
        return _prepare_selected_context(
            selected,
            series_name=self.series_name,
            request_cutoff_utc=cutoff,
            context_length=context_length,
            maximum_gap_hours=maximum_gap_hours,
            discarded_after_cutoff=len(self.frame) - self.position,
        )


def _vintage_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    canonical = frame.loc[
        :,
        [
            "value_time_utc",
            "snapshot_time_utc",
            "revision_time_utc",
            "value",
            "downloaded_at_utc",
            "_row_order",
        ],
    ].copy()
    hashed = pd.util.hash_pandas_object(canonical, index=False).to_numpy(
        dtype=np.uint64
    )
    digest = hashlib.sha256(hashed.tobytes()).hexdigest()
    return {
        "rows": len(canonical),
        "sha256": digest,
        "first_value_time_utc": (
            pd.Timestamp(canonical["value_time_utc"].min()).isoformat()
            if len(canonical)
            else None
        ),
        "last_value_time_utc": (
            pd.Timestamp(canonical["value_time_utc"].max()).isoformat()
            if len(canonical)
            else None
        ),
    }


def _normalize_five_sources(
    vintages_by_series: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    expected = set(COUNTRY_OBSERVED_SERIES.values())
    received = {str(name) for name in vintages_by_series}
    for name in received:
        _validate_observed_series_name(name)
    if received != expected:
        raise HistoricalResidualLoadError(
            "Le replay exige exactement les cinq series observees: "
            f"absentes={sorted(expected - received)}, "
            f"inattendues={sorted(received - expected)}."
        )
    normalized: dict[str, pd.DataFrame] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    for country, series_name in COUNTRY_OBSERVED_SERIES.items():
        alias = COUNTRY_ALIASES[country]
        frame = _normalize_observed_vintages(
            vintages_by_series[series_name], series_name=series_name
        )
        normalized[alias] = frame
        fingerprints[series_name] = _vintage_fingerprint(frame)
    return normalized, fingerprints


def _forecast_plan(
    plan: HistoricalDeliveryPlan,
    *,
    cursors: Mapping[str, _ObservedVintageCursor],
    pipeline: Any,
    batch_size: int,
    context_length: int,
    maximum_gap_hours: int,
) -> pd.DataFrame:
    contexts: dict[str, pd.Series] = {}
    context_audits: dict[str, dict[str, Any]] = {}
    for alias in EXPECTED_ALIASES:
        context, audit = cursors[alias].context_at(
            plan.request_cutoff_utc,
            context_length=context_length,
            maximum_gap_hours=maximum_gap_hours,
        )
        contexts[alias] = context.rename(alias)
        context_audits[alias] = audit

    forecasts, forecast_audits = live_provider._forecast_delivery_day(
        pipeline,
        contexts,
        plan.delivery_index_utc,
        batch_size=batch_size,
    )
    rows: list[pd.DataFrame] = []
    alias_to_series = {
        COUNTRY_ALIASES[country]: series_name
        for country, series_name in COUNTRY_OBSERVED_SERIES.items()
    }
    for alias in EXPECTED_ALIASES:
        forecast = forecasts[alias].loc[:, ["q10", "q50", "q90"]].astype(float)
        values = forecast.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise HistoricalResidualLoadError(
                f"{alias}: quantiles Chronos non finis."
            )
        if bool((forecast["q10"] > forecast["q50"]).any()) or bool(
            (forecast["q50"] > forecast["q90"]).any()
        ):
            raise HistoricalResidualLoadError(
                f"{alias}: quantiles Chronos non ordonnes q10 <= q50 <= q90."
            )
        context_audit = context_audits[alias]
        forecast_audit = forecast_audits[alias]
        model_origin = pd.Timestamp(
            context_audit["last_observation_utc"]
        ).tz_convert("UTC")
        rows.append(
            pd.DataFrame(
                {
                    "delivery_day_local": plan.delivery_day_local.isoformat(),
                    "delivery_hours": len(plan.delivery_index_utc),
                    "alias": alias,
                    "source_series": alias_to_series[alias],
                    "value_time_utc": plan.delivery_index_utc,
                    "snapshot_time_utc": plan.request_cutoff_utc,
                    "revision_time_utc": plan.request_cutoff_utc,
                    "request_cutoff_utc": plan.request_cutoff_utc,
                    "model_origin_utc": model_origin,
                    "context_start_utc": context_audit["context_start_utc"],
                    "context_end_utc": context_audit["context_end_utc"],
                    "context_rows": context_audit["context_rows"],
                    "maximum_selected_snapshot_time_utc": context_audit[
                        "maximum_selected_snapshot_time_utc"
                    ],
                    "maximum_selected_revision_time_utc": context_audit[
                        "maximum_selected_revision_time_utc"
                    ],
                    "imputed_hours": context_audit["imputed_hours"],
                    "bridge_horizon_hours": forecast_audit[
                        "bridge_horizon_hours"
                    ],
                    "lead_to_delivery_start_hours": forecast_audit[
                        "lead_to_delivery_start_hours"
                    ],
                    "value": forecast["q50"].to_numpy(dtype=float),
                    "q10": forecast["q10"].to_numpy(dtype=float),
                    "q50": forecast["q50"].to_numpy(dtype=float),
                    "q90": forecast["q90"].to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).loc[:, PREDICTION_COLUMNS]


def _normalize_replay_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "request_cutoff_utc",
        "model_origin_utc",
        "context_start_utc",
        "context_end_utc",
        "maximum_selected_snapshot_time_utc",
        "maximum_selected_revision_time_utc",
    ):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    return result


def validate_historical_replay_frame(
    frame: pd.DataFrame,
    *,
    plans: Sequence[HistoricalDeliveryPlan] | None = None,
    context_length: int = CONTEXT_LENGTH,
) -> pd.DataFrame:
    """Validate a complete replay table and return its normalized copy."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame doit etre un DataFrame pandas.")
    missing = sorted(set(PREDICTION_COLUMNS).difference(frame.columns))
    if missing:
        raise HistoricalResidualLoadError(
            f"Colonnes du replay absentes: {missing}."
        )
    result = _normalize_replay_timestamps(frame.loc[:, PREDICTION_COLUMNS])
    if result.duplicated(["alias", "value_time_utc"]).any():
        raise HistoricalResidualLoadError(
            "Le replay contient des doublons alias/value_time_utc."
        )
    if set(result["alias"].astype(str)) != set(EXPECTED_ALIASES):
        raise HistoricalResidualLoadError(
            "Le replay ne contient pas exactement les cinq aliases."
        )
    for series_name in result["source_series"].astype(str).unique():
        _validate_observed_series_name(series_name)
    for column in ("value", "q10", "q50", "q90"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result[["value", "q10", "q50", "q90"]].to_numpy()).all():
        raise HistoricalResidualLoadError(
            "Le replay contient des predictions non finies."
        )
    if bool((result["value"] != result["q50"]).any()):
        raise HistoricalResidualLoadError("value doit etre exactement q50.")
    if bool((result["q10"] > result["q50"]).any()) or bool(
        (result["q50"] > result["q90"]).any()
    ):
        raise HistoricalResidualLoadError(
            "Le replay contient des quantiles non ordonnes."
        )
    if bool((result["context_rows"] != int(context_length)).any()):
        raise HistoricalResidualLoadError(
            f"Chaque contexte doit contenir exactement {context_length} heures."
        )
    if bool(
        (result["maximum_selected_snapshot_time_utc"] > result["request_cutoff_utc"]).any()
    ) or bool(
        (result["maximum_selected_revision_time_utc"] > result["request_cutoff_utc"]).any()
    ):
        raise HistoricalResidualLoadError(
            "Une vintage posterieure au cutoff apparait dans le replay."
        )
    if bool((result["model_origin_utc"] > result["request_cutoff_utc"]).any()):
        raise HistoricalResidualLoadError(
            "Une origine modele est posterieure au cutoff demande."
        )
    if bool((result["snapshot_time_utc"] != result["request_cutoff_utc"]).any()) or bool(
        (result["revision_time_utc"] != result["request_cutoff_utc"]).any()
    ):
        raise HistoricalResidualLoadError(
            "Les lignes PIT produites doivent etre datees au cutoff demande."
        )

    expected_plans = list(plans or [])
    if not expected_plans:
        days = sorted({_as_local_date(value) for value in result["delivery_day_local"]})
        expected_plans = build_historical_delivery_plans(days[0], days[-1], delivery_mask=days)
    observed_days = {_as_local_date(value) for value in result["delivery_day_local"]}
    if observed_days != {plan.delivery_day_local for plan in expected_plans}:
        raise HistoricalResidualLoadError(
            "Les jours du replay divergent du plan de livraison."
        )
    for plan in expected_plans:
        day_rows = result.loc[
            result["delivery_day_local"].astype(str).eq(
                plan.delivery_day_local.isoformat()
            )
        ]
        for alias in EXPECTED_ALIASES:
            alias_rows = day_rows.loc[day_rows["alias"].eq(alias)].sort_values(
                "value_time_utc", kind="stable"
            )
            if not pd.DatetimeIndex(alias_rows["value_time_utc"]).equals(
                plan.delivery_index_utc
            ):
                raise HistoricalResidualLoadError(
                    f"{plan.delivery_day_local}/{alias}: jour civil incomplet."
                )
            if not alias_rows["request_cutoff_utc"].eq(
                plan.request_cutoff_utc
            ).all():
                raise HistoricalResidualLoadError(
                    f"{plan.delivery_day_local}/{alias}: cutoff divergent."
                )
            if not alias_rows["delivery_hours"].eq(
                len(plan.delivery_index_utc)
            ).all():
                raise HistoricalResidualLoadError(
                    f"{plan.delivery_day_local}/{alias}: nombre d'heures divergent."
                )
            if not bool(
                (alias_rows["model_origin_utc"] < plan.delivery_index_utc[0]).all()
            ):
                raise HistoricalResidualLoadError(
                    f"{plan.delivery_day_local}/{alias}: origine non anterieure a D."
                )
    return result.sort_values(
        ["delivery_day_local", "alias", "value_time_utc"], kind="stable"
    ).reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        pd.read_parquet(temporary, columns=["value_time_utc"]).head(1)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_identity(
    plans: Sequence[HistoricalDeliveryPlan],
    *,
    source_fingerprints: Mapping[str, Mapping[str, Any]],
    model_id: str,
    model_revision: str,
    context_length: int,
    maximum_gap_hours: int,
    inference_batch_size: int,
    execution_signature: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CHECKPOINT_KIND,
        "delivery_timezone": DELIVERY_TIMEZONE,
        "request_policy": "D-1 08:00 Europe/Paris",
        "delivery_days_local": [
            plan.delivery_day_local.isoformat() for plan in plans
        ],
        "request_cutoffs_utc": [
            plan.request_cutoff_utc.isoformat() for plan in plans
        ],
        "delivery_hours": [len(plan.delivery_index_utc) for plan in plans],
        "model_id": model_id,
        "model_revision": model_revision,
        "context_length": context_length,
        "maximum_internal_gap_hours": maximum_gap_hours,
        "inference_batch_size": inference_batch_size,
        "quantile_levels": list(QUANTILE_LEVELS),
        "cross_learning": False,
        "observed_sources": dict(source_fingerprints),
        "execution_signature": dict(execution_signature),
    }


def _identity_token(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_paths(
    root: Path,
    plans: Sequence[HistoricalDeliveryPlan],
    identity: Mapping[str, Any],
) -> tuple[Path, Path]:
    first = plans[0].delivery_day_local.isoformat()
    last = plans[-1].delivery_day_local.isoformat()
    token = _identity_token(identity)[:16]
    artifact = root / f"residual_load_{first}_{last}_{token}.parquet"
    return artifact, artifact.with_suffix(artifact.suffix + ".manifest.json")


def _load_checkpoint(
    artifact: Path,
    manifest_path: Path,
    *,
    identity: Mapping[str, Any],
    plans: Sequence[HistoricalDeliveryPlan],
    context_length: int,
) -> pd.DataFrame | None:
    if not artifact.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            return None
        if manifest.get("artifact_sha256") != _sha256(artifact):
            return None
        frame = pd.read_parquet(artifact)
        if manifest.get("rows") != len(frame):
            return None
        return validate_historical_replay_frame(
            frame, plans=plans, context_length=context_length
        )
    except (OSError, ValueError, KeyError, HistoricalResidualLoadError):
        return None


def _publish_checkpoint(
    frame: pd.DataFrame,
    artifact: Path,
    manifest_path: Path,
    *,
    identity: Mapping[str, Any],
) -> None:
    _atomic_write_parquet(frame, artifact)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CHECKPOINT_KIND,
        "identity": dict(identity),
        "artifact_path": artifact.name,
        "artifact_sha256": _sha256(artifact),
        "rows": len(frame),
    }
    _atomic_write_json(manifest, manifest_path)


def _chunks(
    plans: Sequence[HistoricalDeliveryPlan], chunk_days: int
) -> list[list[HistoricalDeliveryPlan]]:
    return [
        list(plans[start : start + chunk_days])
        for start in range(0, len(plans), chunk_days)
    ]


def _audits_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.loc[:, AUDIT_COLUMNS]
        .drop_duplicates(["delivery_day_local", "alias"], keep="first")
        .sort_values(["delivery_day_local", "alias"], kind="stable")
        .reset_index(drop=True)
    )


def replay_historical_residual_load(
    vintages_by_series: Mapping[str, pd.DataFrame],
    *,
    start_day: str | date | pd.Timestamp,
    end_day: str | date | pd.Timestamp,
    delivery_mask: DeliveryMask = None,
    pipeline: Any | None = None,
    pipeline_factory: Callable[[], Any] | None = None,
    device: str = "auto",
    local_files_only: bool = False,
    batch_size: int = 8,
    context_length: int = CONTEXT_LENGTH,
    maximum_gap_hours: int = MAX_INTERNAL_GAP_HOURS,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    execution_signature: Mapping[str, Any] | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_days: int = 7,
    resume: bool = True,
) -> HistoricalResidualLoadReplay:
    """Replay the five observed residual loads over historical delivery days.

    The provided ``pipeline`` is reused for every day.  If omitted, one
    ``pipeline_factory`` call (or one default pinned-model load) is performed
    lazily, only when at least one checkpoint still needs to be generated.
    """

    if pipeline is not None and pipeline_factory is not None:
        raise ValueError("Fournis pipeline ou pipeline_factory, pas les deux.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size doit etre strictement positif.")
    if int(context_length) <= 0:
        raise ValueError("context_length doit etre strictement positif.")
    if int(context_length) != CONTEXT_LENGTH:
        raise ValueError(
            f"Le replay de comparaison exige context_length={CONTEXT_LENGTH}."
        )
    if int(maximum_gap_hours) < 0:
        raise ValueError("maximum_gap_hours doit etre positif ou nul.")
    if int(maximum_gap_hours) != MAX_INTERNAL_GAP_HOURS:
        raise ValueError(
            "Le replay de comparaison exige maximum_gap_hours="
            f"{MAX_INTERNAL_GAP_HOURS}."
        )
    if int(checkpoint_days) <= 0:
        raise ValueError("checkpoint_days doit etre strictement positif.")

    plans = build_historical_delivery_plans(
        start_day, end_day, delivery_mask=delivery_mask
    )
    if not plans:
        empty = pd.DataFrame(columns=PREDICTION_COLUMNS)
        return HistoricalResidualLoadReplay(
            predictions=empty,
            audits=pd.DataFrame(columns=AUDIT_COLUMNS),
        )
    normalized, fingerprints = _normalize_five_sources(vintages_by_series)
    cursors = {
        alias: _ObservedVintageCursor(
            frame,
            series_name=COUNTRY_OBSERVED_SERIES[alias[:2]],
        )
        for alias, frame in normalized.items()
    }

    root = (
        Path(checkpoint_dir).expanduser().resolve()
        if checkpoint_dir is not None
        else None
    )
    checkpoint_paths: list[Path] = []
    frames: list[pd.DataFrame] = []
    loaded_pipeline = pipeline

    def get_pipeline() -> Any:
        nonlocal loaded_pipeline
        if loaded_pipeline is None:
            loaded_pipeline = (
                pipeline_factory()
                if pipeline_factory is not None
                else live_provider._load_pipeline(
                    model_id=model_id,
                    revision=model_revision,
                    device=device,
                    local_files_only=local_files_only,
                )
            )
        return loaded_pipeline

    for chunk in _chunks(plans, int(checkpoint_days)):
        identity = _checkpoint_identity(
            chunk,
            source_fingerprints=fingerprints,
            model_id=model_id,
            model_revision=model_revision,
            context_length=int(context_length),
            maximum_gap_hours=int(maximum_gap_hours),
            inference_batch_size=int(batch_size),
            execution_signature=dict(execution_signature or {}),
        )
        artifact: Path | None = None
        manifest_path: Path | None = None
        checkpoint: pd.DataFrame | None = None
        if root is not None:
            artifact, manifest_path = _checkpoint_paths(root, chunk, identity)
            if resume:
                checkpoint = _load_checkpoint(
                    artifact,
                    manifest_path,
                    identity=identity,
                    plans=chunk,
                    context_length=int(context_length),
                )
        if checkpoint is not None:
            # Advance the vintage cursors even though inference is skipped, so
            # a later missing chunk observes the correct monotonic cutoff.
            for plan in chunk:
                for alias in EXPECTED_ALIASES:
                    cursors[alias].context_at(
                        plan.request_cutoff_utc,
                        context_length=int(context_length),
                        maximum_gap_hours=int(maximum_gap_hours),
                    )
            frames.append(checkpoint)
            checkpoint_paths.append(artifact)
            continue

        generated = [
            _forecast_plan(
                plan,
                cursors=cursors,
                pipeline=get_pipeline(),
                batch_size=int(batch_size),
                context_length=int(context_length),
                maximum_gap_hours=int(maximum_gap_hours),
            )
            for plan in chunk
        ]
        chunk_frame = validate_historical_replay_frame(
            pd.concat(generated, ignore_index=True),
            plans=chunk,
            context_length=int(context_length),
        )
        if artifact is not None and manifest_path is not None:
            _publish_checkpoint(
                chunk_frame,
                artifact,
                manifest_path,
                identity=identity,
            )
            checkpoint_paths.append(artifact)
        frames.append(chunk_frame)

    predictions = validate_historical_replay_frame(
        pd.concat(frames, ignore_index=True),
        plans=plans,
        context_length=int(context_length),
    )
    return HistoricalResidualLoadReplay(
        predictions=predictions,
        audits=_audits_from_predictions(predictions),
        checkpoint_paths=tuple(checkpoint_paths),
    )


def _prediction_frame(
    replay_or_frame: HistoricalResidualLoadReplay | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(replay_or_frame, HistoricalResidualLoadReplay):
        return replay_or_frame.predictions
    if not isinstance(replay_or_frame, pd.DataFrame):
        raise TypeError("Attendu: HistoricalResidualLoadReplay ou DataFrame.")
    return replay_or_frame


def replay_to_pit_frames(
    replay_or_frame: HistoricalResidualLoadReplay | pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Convert one replay to five price-pipeline-ready PIT DataFrames."""

    frame = _prediction_frame(replay_or_frame)
    if frame.empty:
        return {
            alias: pd.DataFrame(
                columns=[
                    "value_time_utc",
                    "snapshot_time_utc",
                    "revision_time_utc",
                    "value",
                    "q10",
                    "q50",
                    "q90",
                    "request_cutoff_utc",
                    "model_origin_utc",
                    "delivery_day_local",
                    "source_series",
                ]
            )
            for alias in EXPECTED_ALIASES
        }
    normalized = validate_historical_replay_frame(frame)
    columns = [
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
        "q10",
        "q50",
        "q90",
        "request_cutoff_utc",
        "model_origin_utc",
        "delivery_day_local",
        "source_series",
    ]
    return {
        alias: normalized.loc[normalized["alias"].eq(alias), columns]
        .sort_values("value_time_utc", kind="stable")
        .reset_index(drop=True)
        for alias in EXPECTED_ALIASES
    }


def replay_to_wide(
    replay_or_frame: HistoricalResidualLoadReplay | pd.DataFrame,
    *,
    value_column: str = "q50",
) -> pd.DataFrame:
    """Pivot one replay to UTC rows and the five residual-load aliases."""

    if value_column not in {"value", "q10", "q50", "q90"}:
        raise ValueError("value_column doit etre value, q10, q50 ou q90.")
    frame = _prediction_frame(replay_or_frame)
    if frame.empty:
        return pd.DataFrame(columns=EXPECTED_ALIASES).rename_axis(
            "value_time_utc"
        )
    normalized = validate_historical_replay_frame(frame)
    wide = normalized.pivot(
        index="value_time_utc", columns="alias", values=value_column
    ).sort_index()
    return wide.reindex(columns=EXPECTED_ALIASES).rename_axis(columns=None)


__all__ = [
    "AUDIT_COLUMNS",
    "CHECKPOINT_KIND",
    "CONTEXT_LENGTH",
    "COUNTRY_ALIASES",
    "COUNTRY_OBSERVED_SERIES",
    "DELIVERY_TIMEZONE",
    "EXPECTED_ALIASES",
    "HistoricalDeliveryPlan",
    "HistoricalResidualLoadError",
    "HistoricalResidualLoadReplay",
    "MAX_INTERNAL_GAP_HOURS",
    "MODEL_ID",
    "MODEL_REVISION",
    "PREDICTION_COLUMNS",
    "build_historical_delivery_plans",
    "replay_historical_residual_load",
    "replay_to_pit_frames",
    "replay_to_wide",
    "select_observed_context_asof",
    "validate_historical_replay_frame",
]
