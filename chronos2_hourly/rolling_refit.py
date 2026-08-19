"""Pure contracts for a causal rolling residual-model refit.

This module deliberately performs no filesystem, Saturn, Storm, model, or
network access.  Callers materialise and checksum sealed OOF/PIT archive
blocks first, then pass those in-memory blocks here.  The selector returns the
exact local-delivery-day window that may be supplied to the residual
corrector.

The production contract is intentionally narrow:

* exactly ``[F - window_days, F - 1]`` local delivery days are selected;
* every selected day must match its canonical 23/24/25-hour physical grid;
* only raw Chronos quantiles from sealed OOF or immutable live/PIT archives
  are accepted;
* forecast origins must be exactly civil ``T-1 08:00`` (configurable), and
  selected covariate snapshot/revision maxima may not exceed that origin;
* Storm and MKOnline are excluded from every training schema.

The pretrained Chronos model is not fitted here.  The returned ``X``, ``y``,
``base`` and ``experts`` objects are the inputs for the daily CatBoost/HGB
residual-corrector refit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index


QUANTILES = ("q10", "q50", "q90")
EXPERT_COLUMNS = tuple(f"chronos2__{name}" for name in QUANTILES)
SOURCE_KINDS = frozenset({"sealed_oof", "pit_replay", "issued_live"})
SOURCE_PRECEDENCE = {
    "pit_replay": 0,
    "issued_live": 1,
    "sealed_oof": 2,
}
FORBIDDEN_TOKENS = ("storm", "mkonline")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class RollingRefitContractError(ValueError):
    """Raised when a rolling-fit input cannot prove the causal contract."""


@dataclass(frozen=True)
class RollingRefitPolicy:
    """Versionable operational policy for the residual-corrector fit."""

    window_days: int = 365
    origin_hour_local: int = 8
    origin_timezone: str = "Europe/Paris"

    def __post_init__(self) -> None:
        if isinstance(self.window_days, bool) or not isinstance(
            self.window_days, int
        ):
            raise RollingRefitContractError("window_days must be an integer")
        if self.window_days <= 0:
            raise RollingRefitContractError("window_days must be positive")
        if isinstance(self.origin_hour_local, bool) or not isinstance(
            self.origin_hour_local, int
        ):
            raise RollingRefitContractError(
                "origin_hour_local must be an integer"
            )
        if not 0 <= self.origin_hour_local <= 23:
            raise RollingRefitContractError(
                "origin_hour_local must be between 0 and 23"
            )
        if not str(self.origin_timezone).strip():
            raise RollingRefitContractError("origin_timezone must be explicit")
        # Force an eager timezone validation rather than discovering a typo
        # after an expensive Chronos forecast has run.
        try:
            pd.Timestamp("2026-01-01").tz_localize(self.origin_timezone)
        except Exception as exc:  # pandas exposes backend-specific exceptions
            raise RollingRefitContractError(
                f"invalid origin_timezone: {self.origin_timezone!r}"
            ) from exc


@dataclass(frozen=True)
class RollingRefitBlock:
    """One already-materialised, already-checksummed causal source block.

    ``maximum_snapshot_time_utc`` and ``maximum_revision_time_utc`` contain
    the maximum selected timestamp across all covariates for each delivery
    hour.  They are present exactly where ``pit_inputs_present`` is true.  A
    row whose declared PIT features are all missing carries ``False`` and
    genuine ``NaT`` evidence; callers must never replace that absence with a
    synthetic timestamp such as ``origin - epsilon``.  The upstream archive
    loader remains responsible for deriving the mask and maxima from its
    point-in-time materialisations.
    """

    source_kind: str
    source_id: str
    source_sha256: str
    features: pd.DataFrame
    target: pd.Series
    chronos_quantiles: pd.DataFrame
    forecast_origin_utc: pd.Series
    pit_inputs_present: pd.Series
    maximum_snapshot_time_utc: pd.Series
    maximum_revision_time_utc: pd.Series


@dataclass(frozen=True)
class RollingRefitSelection:
    """Exact aligned inputs and audit returned to a residual corrector."""

    X: pd.DataFrame
    y: pd.Series
    base: pd.DataFrame
    experts: pd.DataFrame
    forecast_origin_utc: pd.Series
    pit_inputs_present: pd.Series
    provenance: pd.DataFrame
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class _PreparedBlock:
    source_kind: str
    source_id: str
    source_sha256: str
    features: pd.DataFrame
    target: pd.Series
    chronos_quantiles: pd.DataFrame
    forecast_origin_utc: pd.Series
    pit_inputs_present: pd.Series
    maximum_snapshot_time_utc: pd.Series
    maximum_revision_time_utc: pd.Series


def _civil_day(value: date | str | pd.Timestamp, *, timezone: str) -> date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(timezone).tz_localize(None)
    if timestamp != timestamp.normalize():
        raise RollingRefitContractError(
            "forecast_delivery_day must be one civil date"
        )
    return timestamp.date()


def _require_index(index: pd.Index, *, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise RollingRefitContractError(f"{name} must use a DatetimeIndex")
    if index.tz is None:
        raise RollingRefitContractError(f"{name} index must be timezone-aware")
    if index.empty:
        raise RollingRefitContractError(f"{name} must not be empty")
    if index.has_duplicates:
        raise RollingRefitContractError(f"{name} contains duplicate timestamps")
    if not index.is_monotonic_increasing:
        raise RollingRefitContractError(f"{name} must be chronologically sorted")
    return pd.DatetimeIndex(index.tz_convert("UTC"), name="delivery_start_utc")


def _forbidden_names(values: Sequence[Any]) -> list[str]:
    return [
        str(value)
        for value in values
        if any(token in str(value).casefold() for token in FORBIDDEN_TOKENS)
    ]


def _numeric_features(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    forbidden = _forbidden_names(list(frame.columns))
    if forbidden:
        raise RollingRefitContractError(
            f"{name} contains forbidden forecast features: {forbidden}"
        )
    converted = frame.apply(pd.to_numeric, errors="coerce")
    introduced_missing = converted.isna() & ~frame.isna()
    if bool(introduced_missing.to_numpy().any()):
        raise RollingRefitContractError(f"{name} contains non-numeric values")
    values = converted.to_numpy(dtype=float)
    # Missing features are part of the frozen residual recipe and are handled
    # by its declared estimators.  Infinities, unlike NaNs, are never valid.
    if bool(np.isinf(values).any()):
        raise RollingRefitContractError(f"{name} contains infinite features")
    return converted


def _finite_series(series: pd.Series, *, name: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce").astype(float)
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise RollingRefitContractError(f"{name} contains non-finite values")
    return converted


def _chronos_quantiles(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    forbidden = _forbidden_names(list(frame.columns))
    if forbidden:
        raise RollingRefitContractError(
            f"{name} contains forbidden forecast columns: {forbidden}"
        )
    if list(frame.columns) != list(QUANTILES):
        raise RollingRefitContractError(
            f"{name} must contain exactly {list(QUANTILES)}"
        )
    result = frame.apply(pd.to_numeric, errors="coerce").astype(float)
    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RollingRefitContractError(f"{name} contains non-finite quantiles")
    if not bool(
        ((values[:, 0] <= values[:, 1]) & (values[:, 1] <= values[:, 2])).all()
    ):
        raise RollingRefitContractError(f"{name} contains crossed quantiles")
    return result


def _utc_series(
    series: pd.Series,
    *,
    name: str,
    allow_missing: bool = False,
) -> pd.Series:
    converted: list[pd.Timestamp] = []
    try:
        for value in series:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                if not allow_missing:
                    raise RollingRefitContractError(
                        f"{name} contains missing timestamps"
                    )
                converted.append(pd.NaT)
                continue
            if timestamp.tzinfo is None:
                raise RollingRefitContractError(
                    f"{name} timestamps must be timezone-aware"
                )
            converted.append(timestamp.tz_convert("UTC"))
    except RollingRefitContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise RollingRefitContractError(f"{name} contains invalid timestamps") from exc
    result = pd.Series(converted, index=series.index, name=name)
    return result


def _boolean_series(series: pd.Series, *, name: str) -> pd.Series:
    values: list[bool] = []
    for value in series:
        if pd.isna(value) or not isinstance(value, (bool, np.bool_)):
            raise RollingRefitContractError(
                f"{name} must contain explicit booleans without missing values"
            )
        values.append(bool(value))
    return pd.Series(values, index=series.index, name=name, dtype=bool)


def _prepare_block(block: RollingRefitBlock) -> _PreparedBlock:
    kind = str(block.source_kind).strip()
    if kind not in SOURCE_KINDS:
        raise RollingRefitContractError(
            f"unsupported source_kind={block.source_kind!r}; expected "
            f"one of {sorted(SOURCE_KINDS)}"
        )
    source_id = str(block.source_id).strip()
    if not source_id:
        raise RollingRefitContractError("source_id must be explicit")
    source_sha256 = str(block.source_sha256).strip().lower()
    if _SHA256.fullmatch(source_sha256) is None:
        raise RollingRefitContractError(
            f"{source_id}: source_sha256 must be one SHA-256 digest"
        )

    index = _require_index(block.features.index, name=f"{source_id}.features")
    aligned: tuple[tuple[str, pd.Series | pd.DataFrame], ...] = (
        ("target", block.target),
        ("chronos_quantiles", block.chronos_quantiles),
        ("forecast_origin_utc", block.forecast_origin_utc),
        ("pit_inputs_present", block.pit_inputs_present),
        ("maximum_snapshot_time_utc", block.maximum_snapshot_time_utc),
        ("maximum_revision_time_utc", block.maximum_revision_time_utc),
    )
    for label, value in aligned:
        observed = _require_index(value.index, name=f"{source_id}.{label}")
        if not observed.equals(index):
            raise RollingRefitContractError(
                f"{source_id}.{label} is not aligned with features"
            )

    feature_names = list(block.features.columns)
    if len(feature_names) != len(set(map(str, feature_names))):
        raise RollingRefitContractError(
            f"{source_id}.features contains duplicate column names"
        )
    series_names = [
        block.target.name,
        block.forecast_origin_utc.name,
        block.pit_inputs_present.name,
        block.maximum_snapshot_time_utc.name,
        block.maximum_revision_time_utc.name,
    ]
    forbidden = _forbidden_names([name for name in series_names if name is not None])
    if forbidden:
        raise RollingRefitContractError(
            f"{source_id} contains forbidden training fields: {forbidden}"
        )

    features = block.features.copy()
    features.index = index
    features = _numeric_features(features, name=f"{source_id}.features")
    target = block.target.copy()
    target.index = index
    target = _finite_series(target, name=f"{source_id}.target")
    target.name = "actual"
    quantiles = block.chronos_quantiles.copy()
    quantiles.index = index
    quantiles = _chronos_quantiles(
        quantiles,
        name=f"{source_id}.chronos_quantiles",
    )
    origin_input = block.forecast_origin_utc.copy()
    origin_input.index = index
    origin = _utc_series(origin_input, name="forecast_origin_utc")
    pit_input = block.pit_inputs_present.copy()
    pit_input.index = index
    pit_inputs_present = _boolean_series(
        pit_input,
        name="pit_inputs_present",
    )
    snapshot_input = block.maximum_snapshot_time_utc.copy()
    snapshot_input.index = index
    snapshot = _utc_series(
        snapshot_input,
        name="maximum_snapshot_time_utc",
        allow_missing=True,
    )
    revision_input = block.maximum_revision_time_utc.copy()
    revision_input.index = index
    revision = _utc_series(
        revision_input,
        name="maximum_revision_time_utc",
        allow_missing=True,
    )
    snapshot_present = snapshot.notna().to_numpy(dtype=bool)
    revision_present = revision.notna().to_numpy(dtype=bool)
    expected_present = pit_inputs_present.to_numpy(dtype=bool)
    if not np.array_equal(snapshot_present, expected_present):
        raise RollingRefitContractError(
            f"{source_id}.maximum_snapshot_time_utc must be present exactly "
            "where pit_inputs_present is true"
        )
    if not np.array_equal(revision_present, expected_present):
        raise RollingRefitContractError(
            f"{source_id}.maximum_revision_time_utc must be present exactly "
            "where pit_inputs_present is true"
        )
    return _PreparedBlock(
        source_kind=kind,
        source_id=source_id,
        source_sha256=source_sha256,
        features=features,
        target=target,
        chronos_quantiles=quantiles,
        forecast_origin_utc=origin,
        pit_inputs_present=pit_inputs_present,
        maximum_snapshot_time_utc=snapshot,
        maximum_revision_time_utc=revision,
    )


def _rows_equal(
    positions: Sequence[int],
    *,
    features: pd.DataFrame,
    target: pd.Series,
    quantiles: pd.DataFrame,
    origin: pd.Series,
    pit_inputs_present: pd.Series,
    snapshot: pd.Series,
    revision: pd.Series,
) -> bool:
    first = int(positions[0])
    for position in positions[1:]:
        current = int(position)
        if not np.allclose(
            features.iloc[first].to_numpy(dtype=float),
            features.iloc[current].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            return False
        if not np.isclose(
            float(target.iloc[first]),
            float(target.iloc[current]),
            rtol=0.0,
            atol=1e-12,
        ):
            return False
        if not np.allclose(
            quantiles.iloc[first].to_numpy(dtype=float),
            quantiles.iloc[current].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            return False
        if bool(pit_inputs_present.iloc[first]) != bool(
            pit_inputs_present.iloc[current]
        ):
            return False
        for series in (origin, snapshot, revision):
            left = pd.Timestamp(series.iloc[first])
            right = pd.Timestamp(series.iloc[current])
            if pd.isna(left) and pd.isna(right):
                continue
            if left != right:
                return False
    return True


def _expected_window_index(
    *,
    forecast_day: date,
    timezone: str,
    window_days: int,
) -> tuple[pd.DatetimeIndex, list[date]]:
    days = [
        timestamp.date()
        for timestamp in pd.date_range(
            pd.Timestamp(forecast_day) - pd.Timedelta(days=window_days),
            pd.Timestamp(forecast_day) - pd.Timedelta(days=1),
            freq="D",
        )
    ]
    indexes = [
        local_delivery_day_index(day, timezone=timezone) for day in days
    ]
    if not indexes:
        raise RollingRefitContractError("rolling window is empty")
    expected = indexes[0]
    for item in indexes[1:]:
        expected = expected.append(item)
    return (
        pd.DatetimeIndex(expected.tz_convert("UTC"), name="delivery_start_utc"),
        days,
    )


def _expected_origins(
    days: Sequence[date],
    *,
    timezone: str,
    policy: RollingRefitPolicy,
) -> pd.Series:
    values: list[pd.Timestamp] = []
    index_parts: list[pd.DatetimeIndex] = []
    for day in days:
        delivery = local_delivery_day_index(day, timezone=timezone)
        cutoff_naive = (
            pd.Timestamp(day)
            - pd.Timedelta(days=1)
            + pd.Timedelta(hours=policy.origin_hour_local)
        )
        cutoff = cutoff_naive.tz_localize(
            policy.origin_timezone,
            ambiguous="raise",
            nonexistent="raise",
        ).tz_convert("UTC")
        values.extend([cutoff] * len(delivery))
        index_parts.append(delivery)
    index = index_parts[0]
    for item in index_parts[1:]:
        index = index.append(item)
    return pd.Series(
        pd.DatetimeIndex(values),
        index=pd.DatetimeIndex(index.tz_convert("UTC"), name="delivery_start_utc"),
        name="expected_forecast_origin_utc",
    )


def _stable_digest(
    *,
    features: pd.DataFrame,
    target: pd.Series,
    quantiles: pd.DataFrame,
    origin: pd.Series,
    pit_inputs_present: pd.Series,
    snapshot: pd.Series,
    revision: pd.Series,
    provenance: pd.DataFrame,
) -> str:
    digest = hashlib.sha256()
    schema = {
        "features": [str(column) for column in features.columns],
        "feature_dtypes": [str(dtype) for dtype in features.dtypes],
        "target": str(target.name),
        "quantiles": [str(column) for column in quantiles.columns],
    }
    digest.update(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for value in (
        pd.util.hash_pandas_object(features, index=True),
        pd.util.hash_pandas_object(target, index=True),
        pd.util.hash_pandas_object(quantiles, index=True),
        pd.util.hash_pandas_object(origin, index=True),
        pd.util.hash_pandas_object(pit_inputs_present, index=True),
        pd.util.hash_pandas_object(snapshot, index=True),
        pd.util.hash_pandas_object(revision, index=True),
        pd.util.hash_pandas_object(provenance, index=True),
    ):
        digest.update(value.to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def select_rolling_refit_window(
    blocks: Sequence[RollingRefitBlock],
    *,
    forecast_delivery_day: date | str | pd.Timestamp,
    delivery_timezone: str,
    policy: RollingRefitPolicy | None = None,
) -> RollingRefitSelection:
    """Merge causal sources and select exact local days ``F-N`` through ``F-1``.

    Block ordering cannot change the result.  Exact duplicate rows are
    resolved with explicit precedence ``sealed_oof > issued_live >
    pit_replay``.  A sealed OOF row is the authoritative source when a
    bootstrap archive overlaps it; an actually issued archive is preferred
    to an equivalent reconstruction.  Conflicting duplicate rows fail
    closed.
    """

    active_policy = policy or RollingRefitPolicy()
    if isinstance(blocks, (str, bytes)) or not isinstance(blocks, Sequence):
        raise RollingRefitContractError("blocks must be a non-empty sequence")
    if not blocks:
        raise RollingRefitContractError("blocks must be a non-empty sequence")
    if not str(delivery_timezone).strip():
        raise RollingRefitContractError("delivery_timezone must be explicit")
    try:
        pd.Timestamp("2026-01-01").tz_localize(delivery_timezone)
    except Exception as exc:
        raise RollingRefitContractError(
            f"invalid delivery_timezone: {delivery_timezone!r}"
        ) from exc

    forecast_day = _civil_day(
        forecast_delivery_day,
        timezone=delivery_timezone,
    )
    prepared = [_prepare_block(block) for block in blocks]
    feature_schema = list(prepared[0].features.columns)
    for block in prepared[1:]:
        if list(block.features.columns) != feature_schema:
            raise RollingRefitContractError(
                "all rolling-refit blocks must share one exact feature schema"
            )

    features = pd.concat([block.features for block in prepared], axis=0)
    target = pd.concat([block.target for block in prepared], axis=0)
    quantiles = pd.concat(
        [block.chronos_quantiles for block in prepared], axis=0
    )
    origin = pd.concat([block.forecast_origin_utc for block in prepared], axis=0)
    pit_inputs_present = pd.concat(
        [block.pit_inputs_present for block in prepared], axis=0
    )
    snapshot = pd.concat(
        [block.maximum_snapshot_time_utc for block in prepared], axis=0
    )
    revision = pd.concat(
        [block.maximum_revision_time_utc for block in prepared], axis=0
    )
    provenance = pd.concat(
        [
            pd.DataFrame(
                {
                    "source_kind": block.source_kind,
                    "source_id": block.source_id,
                    "source_sha256": block.source_sha256,
                },
                index=block.features.index,
            )
            for block in prepared
        ],
        axis=0,
    )
    combined_index = pd.DatetimeIndex(features.index).tz_convert("UTC")
    groups: dict[int, list[int]] = {}
    for position, timestamp_ns in enumerate(combined_index.asi8):
        groups.setdefault(int(timestamp_ns), []).append(position)

    selected_positions: list[int] = []
    equivalent_duplicate_rows = 0
    for timestamp_ns in sorted(groups):
        positions = groups[timestamp_ns]
        if len(positions) > 1:
            if not _rows_equal(
                positions,
                features=features,
                target=target,
                quantiles=quantiles,
                origin=origin,
                pit_inputs_present=pit_inputs_present,
                snapshot=snapshot,
                revision=revision,
            ):
                timestamp = pd.Timestamp(timestamp_ns, tz="UTC")
                sources = provenance.iloc[positions]["source_id"].tolist()
                raise RollingRefitContractError(
                    f"conflicting immutable rolling-refit rows at {timestamp}: "
                    f"{sources}"
                )
            equivalent_duplicate_rows += len(positions) - 1
        chosen = min(
            positions,
            key=lambda position: (
                -SOURCE_PRECEDENCE[str(provenance.iloc[position]["source_kind"])],
                str(provenance.iloc[position]["source_id"]),
                str(provenance.iloc[position]["source_sha256"]),
            ),
        )
        selected_positions.append(chosen)

    selected_positions.sort(key=lambda position: combined_index[position].value)

    def take(value: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        result = value.iloc[selected_positions].copy()
        result.index = pd.DatetimeIndex(
            combined_index[selected_positions],
            name="delivery_start_utc",
        )
        return result

    merged_features = take(features)
    merged_target = take(target)
    merged_quantiles = take(quantiles)
    merged_origin = take(origin)
    merged_pit_inputs_present = take(pit_inputs_present)
    merged_snapshot = take(snapshot)
    merged_revision = take(revision)
    merged_provenance = take(provenance)
    assert isinstance(merged_features, pd.DataFrame)
    assert isinstance(merged_target, pd.Series)
    assert isinstance(merged_quantiles, pd.DataFrame)
    assert isinstance(merged_origin, pd.Series)
    assert isinstance(merged_pit_inputs_present, pd.Series)
    assert isinstance(merged_snapshot, pd.Series)
    assert isinstance(merged_revision, pd.Series)
    assert isinstance(merged_provenance, pd.DataFrame)

    expected, expected_days = _expected_window_index(
        forecast_day=forecast_day,
        timezone=delivery_timezone,
        window_days=active_policy.window_days,
    )
    local_dates = pd.Index(
        merged_features.index.tz_convert(delivery_timezone).date
    )
    start_day = expected_days[0]
    end_day = expected_days[-1]
    in_window = np.asarray(
        (local_dates >= start_day) & (local_dates <= end_day),
        dtype=bool,
    )
    observed = merged_features.index[in_window]
    if not observed.equals(expected):
        missing = expected.difference(observed)
        unexpected = observed.difference(expected)
        raise RollingRefitContractError(
            "rolling window must contain every canonical physical hour for "
            f"{start_day}..{end_day}: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    X = merged_features.loc[expected].copy()
    y = merged_target.loc[expected].copy()
    base = merged_quantiles.loc[expected].copy()
    selected_origin = merged_origin.loc[expected].copy()
    selected_pit_inputs_present = merged_pit_inputs_present.loc[expected].copy()
    selected_snapshot = merged_snapshot.loc[expected].copy()
    selected_revision = merged_revision.loc[expected].copy()
    selected_provenance = merged_provenance.loc[expected].copy()
    expected_origin = _expected_origins(
        expected_days,
        timezone=delivery_timezone,
        policy=active_policy,
    )
    observed_origin_ns = pd.DatetimeIndex(selected_origin).asi8
    expected_origin_ns = pd.DatetimeIndex(expected_origin).asi8
    origin_violations = int(np.count_nonzero(observed_origin_ns != expected_origin_ns))
    if origin_violations:
        raise RollingRefitContractError(
            "raw Chronos origins must equal civil T-1 "
            f"{active_policy.origin_hour_local:02d}:00 "
            f"{active_policy.origin_timezone}; violations={origin_violations}"
        )
    pit_present = selected_pit_inputs_present.to_numpy(dtype=bool)
    snapshot_ns = pd.DatetimeIndex(selected_snapshot).asi8
    revision_ns = pd.DatetimeIndex(selected_revision).asi8
    snapshot_violations = int(
        np.count_nonzero(
            pit_present & (snapshot_ns > observed_origin_ns)
        )
    )
    revision_violations = int(
        np.count_nonzero(
            pit_present & (revision_ns > observed_origin_ns)
        )
    )
    if snapshot_violations or revision_violations:
        raise RollingRefitContractError(
            "covariate snapshot/revision exceeds its own forecast origin: "
            f"snapshot={snapshot_violations}, revision={revision_violations}"
        )
    if bool((observed_origin_ns >= expected.asi8).any()):
        raise RollingRefitContractError(
            "raw Chronos origin must precede its delivery hour"
        )

    selected_local_dates = pd.Index(expected.tz_convert(delivery_timezone).date)
    for day in expected_days:
        selector = np.asarray(selected_local_dates == day, dtype=bool)
        day_sources = selected_provenance.loc[
            selector, ["source_kind", "source_id", "source_sha256"]
        ].drop_duplicates()
        if len(day_sources) != 1:
            raise RollingRefitContractError(
                f"delivery day {day} is split across multiple training sources"
            )

    experts = base.rename(
        columns={name: f"chronos2__{name}" for name in QUANTILES}
    )
    hours_per_day = pd.Series(selected_local_dates).value_counts(sort=False)
    hour_histogram = {
        str(int(hours)): int(count)
        for hours, count in hours_per_day.value_counts().sort_index().items()
    }
    source_rows = (
        selected_provenance.groupby(
            ["source_kind", "source_id", "source_sha256"],
            sort=True,
        )
        .size()
        .rename("hours")
        .reset_index()
    )
    source_audit: list[dict[str, Any]] = []
    for row in source_rows.to_dict("records"):
        selector = (
            (selected_provenance["source_kind"] == row["source_kind"])
            & (selected_provenance["source_id"] == row["source_id"])
            & (selected_provenance["source_sha256"] == row["source_sha256"])
        )
        days = pd.Index(selected_local_dates[selector.to_numpy()]).unique()
        source_audit.append(
            {
                **row,
                "days": int(len(days)),
                "first_delivery_day_local": str(days[0]),
                "last_delivery_day_local": str(days[-1]),
            }
        )
    corpus_sha256 = _stable_digest(
        features=X,
        target=y,
        quantiles=base,
        origin=selected_origin,
        pit_inputs_present=selected_pit_inputs_present,
        snapshot=selected_snapshot,
        revision=selected_revision,
        provenance=selected_provenance,
    )
    index_sha256 = hashlib.sha256(expected.asi8.tobytes()).hexdigest()
    feature_schema_sha256 = hashlib.sha256(
        json.dumps(
            [str(column) for column in X.columns],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    audit: dict[str, Any] = {
        "refit_scope": "residual_corrector_only",
        "strategy": "rolling_local_delivery_days",
        "window_days_requested": int(active_policy.window_days),
        "window_start_day_local": start_day.isoformat(),
        "window_end_day_local": end_day.isoformat(),
        "forecast_delivery_day_local": forecast_day.isoformat(),
        "training_complete_days": int(len(expected_days)),
        "training_rows": int(len(X)),
        "physical_day_hour_histogram": hour_histogram,
        "training_start_utc": str(expected[0]),
        "training_end_utc": str(expected[-1]),
        "origin_contract": (
            f"civil T-1 {active_policy.origin_hour_local:02d}:00 "
            f"{active_policy.origin_timezone}"
        ),
        "origin_violations": 0,
        "snapshot_cutoff_violations": 0,
        "revision_cutoff_violations": 0,
        "pit_input_present_hours": int(np.count_nonzero(pit_present)),
        "pit_input_missing_hours": int(np.count_nonzero(~pit_present)),
        "equivalent_duplicate_rows_resolved": equivalent_duplicate_rows,
        "source_kinds": {
            str(kind): int(count)
            for kind, count in selected_provenance["source_kind"]
            .value_counts()
            .sort_index()
            .items()
        },
        "sources": source_audit,
        "training_window_index_sha256": index_sha256,
        "feature_schema_sha256": feature_schema_sha256,
        "training_corpus_sha256": corpus_sha256,
        "causal_timestamps_in_training_corpus_hash": True,
        "storm_used_as_feature": False,
        "mkonline_used_as_feature": False,
        "chronos_pretrained_weights_refit": False,
        "residual_corrector_refit": True,
        "mkonline_blend_weights_refit": False,
    }
    selected_origin.name = "forecast_origin_utc"
    selected_pit_inputs_present.name = "pit_inputs_present"
    selected_provenance.index.name = "delivery_start_utc"
    return RollingRefitSelection(
        X=X,
        y=y,
        base=base,
        experts=experts,
        forecast_origin_utc=selected_origin,
        pit_inputs_present=selected_pit_inputs_present,
        provenance=selected_provenance,
        audit=audit,
    )


__all__ = [
    "EXPERT_COLUMNS",
    "QUANTILES",
    "RollingRefitBlock",
    "RollingRefitContractError",
    "RollingRefitPolicy",
    "RollingRefitSelection",
    "select_rolling_refit_window",
]
