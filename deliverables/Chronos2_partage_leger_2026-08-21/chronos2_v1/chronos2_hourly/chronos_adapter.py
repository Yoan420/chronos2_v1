"""DST-safe adapter for Chronos-2 hourly OOF predictions.

The legacy Chronos runner accepts one fixed ``prediction_length`` per call.
French local delivery days do not: they contain 23, 24 or 25 hourly products.
This module builds explicit delivery plans, groups them by horizon, and calls
an injected executor once per horizon.  The concrete executor factory delegates
to :func:`chronos2_modular.forecasting.run_backtest_variant` without importing
or loading Chronos at module import time.

The public OOF contract is deliberately strict:

* a unique, gap-free, hourly UTC ``delivery_start_utc`` index;
* finite ``q10``, ``q50``, ``q90`` and ``actual`` values;
* timezone-aware forecast origins strictly before delivery;
* no silently dropped, duplicated, interpolated or reordered delivery hour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .hourly_contract import local_delivery_day_index


class ChronosAdapterError(ValueError):
    """Raised when a plan or OOF frame violates the adapter contract."""


@dataclass(frozen=True)
class ChronosDeliveryPlan:
    """One forecast origin and its complete local delivery day in UTC."""

    delivery_date: date
    forecast_origin_utc: pd.Timestamp
    delivery_index_utc: pd.DatetimeIndex

    def __post_init__(self) -> None:
        origin = pd.Timestamp(self.forecast_origin_utc)
        if origin.tzinfo is None:
            raise ChronosAdapterError(
                "forecast_origin_utc must be timezone-aware."
            )
        origin = origin.tz_convert("UTC")

        index = pd.DatetimeIndex(self.delivery_index_utc)
        if index.tz is None:
            raise ChronosAdapterError(
                "delivery_index_utc must be timezone-aware."
            )
        index = index.tz_convert("UTC")
        index.name = "delivery_start_utc"
        _validate_hourly_utc_index(
            index,
            require_contiguous=True,
            expected_lengths={23, 24, 25},
            name="delivery_index_utc",
        )
        if not bool((origin < index).all()):
            raise ChronosAdapterError(
                "forecast_origin_utc must be strictly before every delivery."
            )

        object.__setattr__(self, "forecast_origin_utc", origin)
        object.__setattr__(self, "delivery_index_utc", index)

    @property
    def horizon(self) -> int:
        """Chronos prediction length required for this delivery day."""

        return len(self.delivery_index_utc)

    @property
    def delivery_start_utc(self) -> pd.Timestamp:
        return self.delivery_index_utc[0]


class ChronosGroupExecutor(Protocol):
    """Injected fixed-horizon Chronos execution interface."""

    def __call__(
        self,
        *,
        plans: Sequence[ChronosDeliveryPlan],
        horizon: int,
    ) -> pd.DataFrame:
        """Return predictions covering exactly all hours in ``plans``."""


class ChronosLiveExecutor(Protocol):
    """Injected variable-horizon Chronos live execution interface."""

    def __call__(
        self,
        *,
        plan: ChronosDeliveryPlan,
        horizon: int,
    ) -> pd.DataFrame:
        """Return future quantiles covering exactly ``plan``."""


def _parse_local_time(value: str | time) -> time:
    if isinstance(value, time):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = time.fromisoformat(value)
        except ValueError as exc:
            raise ChronosAdapterError(
                "forecast_origin_local_time must use HH:MM[:SS]."
            ) from exc
    else:
        raise TypeError("forecast_origin_local_time must be a string or time.")
    if parsed.tzinfo is not None:
        raise ChronosAdapterError(
            "forecast_origin_local_time must not carry a timezone."
        )
    return parsed


def _as_local_date(
    value: str | date | pd.Timestamp,
    *,
    timezone: str,
) -> date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert(timezone).date()
    return timestamp.date()


def _validate_hourly_utc_index(
    index: pd.DatetimeIndex,
    *,
    require_contiguous: bool,
    expected_lengths: set[int] | None,
    name: str,
) -> None:
    if index.tz is None:
        raise ChronosAdapterError(f"{name} must be timezone-aware.")
    if str(index.tz) != "UTC":
        raise ChronosAdapterError(f"{name} must use the UTC timezone.")
    if index.has_duplicates:
        raise ChronosAdapterError(f"{name} contains duplicate timestamps.")
    if not index.is_monotonic_increasing:
        raise ChronosAdapterError(f"{name} must be monotonic increasing.")
    aligned = (
        (index.minute == 0)
        & (index.second == 0)
        & (index.microsecond == 0)
        & (index.nanosecond == 0)
    )
    if not bool(np.all(aligned)):
        raise ChronosAdapterError(f"{name} is not aligned to full hours.")
    if require_contiguous and len(index) > 1:
        deltas = index[1:] - index[:-1]
        if not bool(np.all(deltas == pd.Timedelta(hours=1))):
            raise ChronosAdapterError(f"{name} contains one or more gaps.")
    if expected_lengths is not None and len(index) not in expected_lengths:
        expected = "/".join(str(value) for value in sorted(expected_lengths))
        raise ChronosAdapterError(
            f"{name} must contain {expected} hours, received {len(index)}."
        )


def build_delivery_plan(
    delivery_date: str | date | pd.Timestamp,
    *,
    forecast_origin_local_time: str | time = "08:00",
    forecast_days_before: int = 1,
    timezone: str = "Europe/Paris",
) -> ChronosDeliveryPlan:
    """Build one 23/24/25-hour plan for a French local delivery day."""

    if isinstance(forecast_days_before, bool):
        raise ChronosAdapterError("forecast_days_before must be >= 1.")
    days_before = int(forecast_days_before)
    if days_before != forecast_days_before or days_before < 1:
        raise ChronosAdapterError(
            "forecast_days_before must be a positive integer."
        )
    local_date = _as_local_date(delivery_date, timezone=timezone)
    origin_clock = _parse_local_time(forecast_origin_local_time)
    origin_date = local_date - timedelta(days=days_before)
    origin_naive = pd.Timestamp.combine(origin_date, origin_clock)
    try:
        origin_local = origin_naive.tz_localize(
            timezone,
            ambiguous="raise",
            nonexistent="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ChronosAdapterError(
            f"Invalid local forecast origin {origin_naive} in {timezone}."
        ) from exc

    return ChronosDeliveryPlan(
        delivery_date=local_date,
        forecast_origin_utc=origin_local.tz_convert("UTC"),
        delivery_index_utc=local_delivery_day_index(
            local_date,
            timezone=timezone,
        ),
    )


def generate_delivery_plans(
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    *,
    forecast_origin_local_time: str | time = "08:00",
    forecast_days_before: int = 1,
    timezone: str = "Europe/Paris",
) -> tuple[ChronosDeliveryPlan, ...]:
    """Generate inclusive, consecutive local delivery-day plans."""

    start = _as_local_date(start_date, timezone=timezone)
    end = _as_local_date(end_date, timezone=timezone)
    if end < start:
        raise ChronosAdapterError("end_date must be on or after start_date.")
    dates = pd.date_range(start, end, freq="D")
    return tuple(
        build_delivery_plan(
            value,
            forecast_origin_local_time=forecast_origin_local_time,
            forecast_days_before=forecast_days_before,
            timezone=timezone,
        )
        for value in dates
    )


def group_delivery_plans_by_horizon(
    plans: Sequence[ChronosDeliveryPlan],
) -> dict[int, tuple[ChronosDeliveryPlan, ...]]:
    """Validate non-overlapping plans and group them into 23/24/25 calls."""

    if not plans:
        raise ChronosAdapterError("At least one delivery plan is required.")
    ordered = sorted(plans, key=lambda plan: plan.delivery_start_utc)
    dates = [plan.delivery_date for plan in ordered]
    if len(dates) != len(set(dates)):
        raise ChronosAdapterError("Delivery plans contain duplicate local dates.")
    combined = ordered[0].delivery_index_utc
    if len(ordered) > 1:
        combined = combined.append(
            [plan.delivery_index_utc for plan in ordered[1:]]
        )
    combined = pd.DatetimeIndex(combined, name="delivery_start_utc")
    _validate_hourly_utc_index(
        combined,
        require_contiguous=True,
        expected_lengths=None,
        name="combined delivery plans",
    )

    grouped: dict[int, list[ChronosDeliveryPlan]] = {}
    for plan in ordered:
        grouped.setdefault(plan.horizon, []).append(plan)
    return {
        horizon: tuple(grouped[horizon])
        for horizon in sorted(grouped)
    }


def _find_column(
    frame: pd.DataFrame,
    canonical: str,
    aliases: Sequence[str],
) -> str:
    matches = [name for name in (canonical, *aliases) if name in frame.columns]
    if not matches:
        raise ChronosAdapterError(
            f"Missing Chronos OOF column '{canonical}' "
            f"(accepted aliases: {list(aliases)})."
        )
    if len(matches) > 1:
        raise ChronosAdapterError(
            f"Ambiguous columns for '{canonical}': {matches}."
        )
    return matches[0]


def _timezone_aware_utc(values: Any, *, name: str) -> pd.DatetimeIndex:
    if isinstance(values, pd.DatetimeIndex):
        raw_values = values
    else:
        raw_values = pd.Index(values)
    if len(raw_values) == 0:
        return pd.DatetimeIndex([], tz="UTC", name=name)

    # ``pd.to_datetime(..., utc=True)`` silently treats naive timestamps as
    # UTC.  Check every raw value first so a lost timezone is never hidden.
    for raw in raw_values:
        if pd.isna(raw):
            raise ChronosAdapterError(f"{name} contains missing timestamps.")
        try:
            timestamp = pd.Timestamp(raw)
        except (TypeError, ValueError) as exc:
            raise ChronosAdapterError(
                f"{name} contains an invalid timestamp: {raw!r}."
            ) from exc
        if timestamp.tzinfo is None:
            raise ChronosAdapterError(
                f"{name} contains timezone-naive timestamps."
            )
    try:
        result = pd.DatetimeIndex(pd.to_datetime(raw_values, utc=True)).as_unit(
            "ns"
        )
    except (TypeError, ValueError) as exc:
        raise ChronosAdapterError(f"Cannot parse {name} as UTC timestamps.") from exc
    result.name = name
    return result


def normalize_chronos_oof(
    frame: pd.DataFrame,
    *,
    require_quantile_order: bool = True,
) -> pd.DataFrame:
    """Normalize and validate Chronos OOF predictions.

    Accepted delivery aliases include the legacy runner's ``timestamp``;
    accepted quantile aliases include ``0.1``, ``0.5`` and ``0.9``.  The
    returned frame is sorted and indexed by canonical hourly UTC delivery.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Chronos OOF predictions must be a pandas DataFrame.")
    if frame.empty:
        raise ChronosAdapterError("Chronos OOF predictions are empty.")
    if not frame.columns.is_unique:
        raise ChronosAdapterError("Chronos OOF predictions have duplicate columns.")

    work = frame.copy()
    if "delivery_start_utc" not in work.columns:
        if isinstance(work.index, pd.DatetimeIndex):
            work.insert(0, "delivery_start_utc", work.index)
        else:
            delivery_column = _find_column(
                work,
                "delivery_start_utc",
                ("timestamp", "delivery_utc", "delivery_start"),
            )
            work = work.rename(columns={delivery_column: "delivery_start_utc"})

    origins_column = _find_column(
        work,
        "forecast_origin_utc",
        ("forecast_origin", "origin_utc", "as_of_utc"),
    )
    q10_column = _find_column(work, "q10", ("0.1", "0.10", "p10"))
    q50_column = _find_column(work, "q50", ("0.5", "0.50", "p50", "median"))
    q90_column = _find_column(work, "q90", ("0.9", "0.90", "p90"))
    actual_column = _find_column(work, "actual", ("target", "y", "observed"))

    delivery = _timezone_aware_utc(
        work["delivery_start_utc"],
        name="delivery_start_utc",
    )
    origins = _timezone_aware_utc(
        work[origins_column],
        name="forecast_origin_utc",
    )
    result = pd.DataFrame(
        {
            "forecast_origin_utc": origins,
            "q10": pd.to_numeric(work[q10_column], errors="coerce").to_numpy(),
            "q50": pd.to_numeric(work[q50_column], errors="coerce").to_numpy(),
            "q90": pd.to_numeric(work[q90_column], errors="coerce").to_numpy(),
            "actual": pd.to_numeric(work[actual_column], errors="coerce").to_numpy(),
        },
        index=delivery,
    )
    result.index.name = "delivery_start_utc"
    result = result.sort_index(kind="stable")
    _validate_hourly_utc_index(
        result.index,
        require_contiguous=True,
        expected_lengths=None,
        name="delivery_start_utc",
    )

    numeric_columns = ["q10", "q50", "q90", "actual"]
    numeric = result[numeric_columns].to_numpy(dtype=float)
    if not bool(np.isfinite(numeric).all()):
        bad = result[numeric_columns].isna().sum()
        details = ", ".join(
            f"{name}={int(count)}"
            for name, count in bad.items()
            if count
        )
        raise ChronosAdapterError(
            "Chronos OOF predictions contain missing/non-finite values"
            + (f": {details}" if details else ".")
        )
    if not bool((result["forecast_origin_utc"] < result.index).all()):
        raise ChronosAdapterError(
            "Every forecast_origin_utc must be strictly before delivery_start_utc."
        )
    if require_quantile_order:
        crossing = (result["q10"] > result["q50"]) | (
            result["q50"] > result["q90"]
        )
        if bool(crossing.any()):
            raise ChronosAdapterError(
                f"Crossing Chronos quantiles on {int(crossing.sum())} row(s)."
            )
    return result


def _delivery_column_name(frame: pd.DataFrame) -> str:
    if "delivery_start_utc" in frame.columns:
        return "delivery_start_utc"
    if isinstance(frame.index, pd.DatetimeIndex):
        return "__index__"
    return _find_column(
        frame,
        "delivery_start_utc",
        ("timestamp", "delivery_utc", "delivery_start"),
    )


def _attach_and_validate_plan_origins(
    frame: pd.DataFrame,
    plans: Sequence[ChronosDeliveryPlan],
) -> pd.DataFrame:
    """Attach trusted plan origins and require exact plan coverage."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ChronosAdapterError("A Chronos group executor returned no rows.")
    work = frame.copy()
    delivery_name = _delivery_column_name(work)
    raw_delivery = work.index if delivery_name == "__index__" else work[delivery_name]
    delivery = _timezone_aware_utc(raw_delivery, name="delivery_start_utc")
    if delivery.has_duplicates:
        raise ChronosAdapterError("A Chronos group contains duplicate deliveries.")

    origin_by_delivery: dict[pd.Timestamp, pd.Timestamp] = {}
    for plan in plans:
        for timestamp in plan.delivery_index_utc:
            if timestamp in origin_by_delivery:
                raise ChronosAdapterError("Delivery plans overlap.")
            origin_by_delivery[timestamp] = plan.forecast_origin_utc
    expected = pd.DatetimeIndex(
        sorted(origin_by_delivery),
        tz="UTC",
        name="delivery_start_utc",
    )
    actual = pd.DatetimeIndex(delivery).sort_values()
    if not actual.equals(expected):
        missing = expected.difference(actual)
        unexpected = actual.difference(expected)
        raise ChronosAdapterError(
            "Chronos group does not exactly cover its delivery plans: "
            f"missing={len(missing)}, unexpected={len(unexpected)}."
        )

    work["delivery_start_utc"] = delivery
    if delivery_name not in {"__index__", "delivery_start_utc"}:
        work = work.drop(columns=[delivery_name])
    trusted_origins = pd.DatetimeIndex(
        [origin_by_delivery[value] for value in delivery],
        tz="UTC",
    )
    origin_names = [
        name
        for name in (
            "forecast_origin_utc",
            "forecast_origin",
            "origin_utc",
            "as_of_utc",
        )
        if name in work.columns
    ]
    if len(origin_names) > 1:
        raise ChronosAdapterError(
            f"Ambiguous executor forecast-origin columns: {origin_names}."
        )
    if origin_names:
        supplied_name = origin_names[0]
        supplied = _timezone_aware_utc(
            work[supplied_name],
            name="forecast_origin_utc",
        )
        if not supplied.equals(trusted_origins):
            raise ChronosAdapterError(
                "Executor forecast origins do not match the delivery plans."
            )
        if supplied_name != "forecast_origin_utc":
            work = work.drop(columns=[supplied_name])
    work["forecast_origin_utc"] = trusted_origins
    return work.reset_index(drop=True)


def normalize_chronos_future(
    frame: pd.DataFrame,
    plan: ChronosDeliveryPlan,
    *,
    require_quantile_order: bool = True,
) -> pd.DataFrame:
    """Normalize one live Chronos forecast against an exact delivery plan.

    Unlike :func:`normalize_chronos_oof`, no observed ``actual`` is required
    or returned.  The trusted forecast origin comes from ``plan``; a supplied
    origin is accepted only when it matches.  Missing or additional delivery
    timestamps, including one of the repeated autumn hours, are rejected.
    """

    work = _attach_and_validate_plan_origins(frame, (plan,))
    q10_column = _find_column(work, "q10", ("0.1", "0.10", "p10"))
    q50_column = _find_column(
        work,
        "q50",
        ("0.5", "0.50", "p50", "median"),
    )
    q90_column = _find_column(work, "q90", ("0.9", "0.90", "p90"))
    delivery = _timezone_aware_utc(
        work["delivery_start_utc"],
        name="delivery_start_utc",
    )
    origins = _timezone_aware_utc(
        work["forecast_origin_utc"],
        name="forecast_origin_utc",
    )
    result = pd.DataFrame(
        {
            "forecast_origin_utc": origins,
            "q10": pd.to_numeric(work[q10_column], errors="coerce").to_numpy(),
            "q50": pd.to_numeric(work[q50_column], errors="coerce").to_numpy(),
            "q90": pd.to_numeric(work[q90_column], errors="coerce").to_numpy(),
        },
        index=delivery,
    ).sort_index(kind="stable")
    result.index.name = "delivery_start_utc"
    if not result.index.equals(plan.delivery_index_utc):
        raise ChronosAdapterError(
            "Chronos live forecast does not preserve the complete plan index."
        )

    numeric = result[["q10", "q50", "q90"]].to_numpy(dtype=float)
    if not bool(np.isfinite(numeric).all()):
        raise ChronosAdapterError(
            "Chronos live forecast contains missing/non-finite quantiles."
        )
    if not bool((result["forecast_origin_utc"] < result.index).all()):
        raise ChronosAdapterError(
            "Live forecast origin must be strictly before every delivery."
        )
    if require_quantile_order:
        crossing = (result["q10"] > result["q50"]) | (
            result["q50"] > result["q90"]
        )
        if bool(crossing.any()):
            raise ChronosAdapterError(
                f"Crossing Chronos quantiles on {int(crossing.sum())} row(s)."
            )
    return result


def execute_chronos_live_forecast(
    plan: ChronosDeliveryPlan,
    executor: ChronosLiveExecutor,
) -> pd.DataFrame:
    """Run one injected live executor and enforce the future contract."""

    raw = executor(plan=plan, horizon=plan.horizon)
    return normalize_chronos_future(raw, plan)


def execute_grouped_chronos_backtest(
    plans: Sequence[ChronosDeliveryPlan],
    executor: ChronosGroupExecutor,
) -> pd.DataFrame:
    """Execute one injected Chronos call per horizon and return strict OOF."""

    grouped = group_delivery_plans_by_horizon(plans)
    frames: list[pd.DataFrame] = []
    for horizon, horizon_plans in grouped.items():
        raw = executor(plans=horizon_plans, horizon=horizon)
        frames.append(
            _attach_and_validate_plan_origins(raw, horizon_plans)
        )
    return normalize_chronos_oof(pd.concat(frames, ignore_index=True))


def make_existing_forecasting_executor(
    *,
    data: Any,
    runtime: Any,
    context_length: int,
    origin_batch_size: int,
    model_batch_size: int,
    with_covariates: bool = True,
    variant: str = "hourly_oof",
    forecasting_module: Any | None = None,
) -> ChronosGroupExecutor:
    """Adapt ``chronos2_modular.forecasting.run_backtest_variant``.

    ``forecasting_module`` is injectable for unit tests.  When omitted, the
    existing repository module is imported lazily.  Origin positions are the
    first delivery timestamp of each plan, matching ``build_origin_frames``.
    """

    if forecasting_module is None:
        from chronos2_modular import forecasting as forecasting_module

    if context_length <= 0 or origin_batch_size <= 0 or model_batch_size <= 0:
        raise ValueError("Chronos lengths and batch sizes must be positive.")
    target = getattr(data, "target", None)
    if not isinstance(target, pd.Series):
        raise TypeError("data.target must be a pandas Series.")
    if not isinstance(target.index, pd.DatetimeIndex) or target.index.tz is None:
        raise ChronosAdapterError("data.target needs a timezone-aware index.")
    if target.index.has_duplicates or not target.index.is_monotonic_increasing:
        raise ChronosAdapterError("data.target index must be unique and sorted.")
    target_utc = target.index.tz_convert("UTC")

    def execute(
        *,
        plans: Sequence[ChronosDeliveryPlan],
        horizon: int,
    ) -> pd.DataFrame:
        if not plans:
            raise ChronosAdapterError("The Chronos group is empty.")
        if any(plan.horizon != horizon for plan in plans):
            raise ChronosAdapterError("Mixed horizons in one Chronos call.")

        origins: list[int] = []
        for plan in plans:
            locations = target_utc.get_indexer(plan.delivery_index_utc)
            if bool((locations < 0).any()):
                raise ChronosAdapterError(
                    f"Target does not cover delivery day {plan.delivery_date}."
                )
            expected = np.arange(locations[0], locations[0] + horizon)
            if not np.array_equal(locations, expected):
                raise ChronosAdapterError(
                    f"Target has a gap inside delivery day {plan.delivery_date}."
                )
            origins.append(int(locations[0]))

        return forecasting_module.run_backtest_variant(
            data=data,
            runtime=runtime,
            origins=origins,
            context_length=int(context_length),
            horizon=int(horizon),
            origin_batch_size=int(origin_batch_size),
            model_batch_size=int(model_batch_size),
            with_covariates=bool(with_covariates),
            variant=str(variant),
        )

    return execute


def make_existing_live_forecast_executor(
    *,
    data: Any,
    runtime: Any,
    context_length: int,
    model_batch_size: int,
    with_covariates: bool = True,
    variant: str = "hourly_live",
    forecasting_module: Any | None = None,
) -> ChronosLiveExecutor:
    """Adapt ``run_live_forecast_variant`` to a 23/24/25-hour plan.

    The returned callable is raw and injectable: use
    :func:`execute_chronos_live_forecast` to attach the plan origin and perform
    final coverage validation.  No model is imported when a test double is
    supplied through ``forecasting_module``.
    """

    if forecasting_module is None:
        from chronos2_modular import forecasting as forecasting_module

    if context_length <= 0 or model_batch_size <= 0:
        raise ValueError("Chronos lengths and batch sizes must be positive.")
    target = getattr(data, "target", None)
    if not isinstance(target, pd.Series) or target.empty:
        raise TypeError("data.target must be a non-empty pandas Series.")
    if not isinstance(target.index, pd.DatetimeIndex) or target.index.tz is None:
        raise ChronosAdapterError("data.target needs a timezone-aware index.")
    if target.index.has_duplicates or not target.index.is_monotonic_increasing:
        raise ChronosAdapterError("data.target index must be unique and sorted.")
    target_utc = target.index.tz_convert("UTC")

    def execute(
        *,
        plan: ChronosDeliveryPlan,
        horizon: int,
    ) -> pd.DataFrame:
        if horizon != plan.horizon or horizon not in {23, 24, 25}:
            raise ChronosAdapterError(
                "Live Chronos horizon must exactly match its 23/24/25-hour plan."
            )
        expected_start = target_utc[-1] + pd.Timedelta(hours=1)
        if expected_start != plan.delivery_start_utc:
            raise ChronosAdapterError(
                "data.target must end exactly one hour before the live plan."
            )
        expected_index = pd.date_range(
            expected_start,
            periods=horizon,
            freq="h",
            tz="UTC",
            name="delivery_start_utc",
        )
        if not expected_index.equals(plan.delivery_index_utc):
            raise ChronosAdapterError(
                "The live plan is not the exact hourly continuation of target."
            )
        return forecasting_module.run_live_forecast_variant(
            data=data,
            runtime=runtime,
            context_length=int(context_length),
            horizon=int(horizon),
            model_batch_size=int(model_batch_size),
            with_covariates=bool(with_covariates),
            variant=str(variant),
        )

    return execute


def run_existing_live_forecast(
    plan: ChronosDeliveryPlan,
    *,
    data: Any,
    runtime: Any,
    context_length: int,
    model_batch_size: int,
    with_covariates: bool = True,
    variant: str = "hourly_live",
    forecasting_module: Any | None = None,
) -> pd.DataFrame:
    """Call the existing live Chronos runner and return strict future output."""

    executor = make_existing_live_forecast_executor(
        data=data,
        runtime=runtime,
        context_length=context_length,
        model_batch_size=model_batch_size,
        with_covariates=with_covariates,
        variant=variant,
        forecasting_module=forecasting_module,
    )
    return execute_chronos_live_forecast(plan, executor)


def load_chronos_oof(
    path: str | Path,
    *,
    require_quantile_order: bool = True,
    read_csv_kwargs: Mapping[str, Any] | None = None,
    read_parquet_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Load and validate a CSV or Parquet Chronos OOF artifact."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix in {".csv", ".csv.gz"} or source.name.lower().endswith(".csv.gz"):
        frame = pd.read_csv(source, **dict(read_csv_kwargs or {}))
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source, **dict(read_parquet_kwargs or {}))
    else:
        raise ValueError("Chronos OOF file must be CSV, CSV.GZ, Parquet or PQ.")
    return normalize_chronos_oof(
        frame,
        require_quantile_order=require_quantile_order,
    )


__all__ = [
    "ChronosAdapterError",
    "ChronosDeliveryPlan",
    "ChronosGroupExecutor",
    "ChronosLiveExecutor",
    "build_delivery_plan",
    "execute_chronos_live_forecast",
    "execute_grouped_chronos_backtest",
    "generate_delivery_plans",
    "group_delivery_plans_by_horizon",
    "load_chronos_oof",
    "make_existing_forecasting_executor",
    "make_existing_live_forecast_executor",
    "normalize_chronos_future",
    "normalize_chronos_oof",
    "run_existing_live_forecast",
]
