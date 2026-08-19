"""Canonical hourly target and daylight-saving-time handling.

All calculations are performed on timezone-aware timestamps and the canonical
model index is UTC.  A local delivery day therefore contains 23, 24, or 25
distinct UTC instants in ``Europe/Paris``.  Target values are never
interpolated: an incomplete quarter-hour block either raises, remains ``NaN``,
or is explicitly dropped according to the caller's policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd


IncompletePolicy = Literal["raise", "nan", "drop"]
TargetInputResolution = Literal["auto", "hourly", "quarter_hour"]


class HourlyTargetContractError(ValueError):
    """Raised when a series violates the hourly-target contract."""


@dataclass(frozen=True)
class HourlyTargetValidation:
    """Result of comparing an existing hourly target with a QH-derived one."""

    is_valid: bool
    compared_hours: int
    missing_in_existing: int
    missing_in_derived: int
    incomplete_quarter_hours: int
    mismatched_hours: int
    max_abs_error: float
    mean_abs_error: float
    details: pd.DataFrame = field(repr=False, compare=False)

    def raise_if_invalid(self) -> None:
        """Raise a compact error when the validation did not pass."""

        if self.is_valid:
            return
        raise HourlyTargetContractError(
            "La cible horaire ne respecte pas le contrat: "
            f"missing_existing={self.missing_in_existing}, "
            f"missing_derived={self.missing_in_derived}, "
            f"incomplete_qh={self.incomplete_quarter_hours}, "
            f"mismatches={self.mismatched_hours}."
        )


def _as_utc_numeric_series(
    values: pd.Series,
    *,
    series_name: str,
) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{series_name} doit être une pandas.Series.")
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError(
            f"{series_name} doit avoir un pandas.DatetimeIndex."
        )
    if values.index.tz is None:
        raise HourlyTargetContractError(
            f"{series_name}: l'index doit être timezone-aware."
        )
    if values.index.has_duplicates:
        duplicates = values.index[values.index.duplicated()].unique()
        raise HourlyTargetContractError(
            f"{series_name}: timestamps dupliqués ({len(duplicates)})."
        )

    try:
        numeric = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise HourlyTargetContractError(
            f"{series_name}: les valeurs doivent être numériques."
        ) from exc

    numeric = numeric.copy()
    numeric.index = numeric.index.tz_convert("UTC")
    numeric = numeric.sort_index()
    numeric.index.name = "delivery_start_utc"
    return numeric


def _assert_quarter_hour_alignment(index: pd.DatetimeIndex) -> None:
    aligned = (
        index.minute.isin([0, 15, 30, 45])
        & (index.second == 0)
        & (index.microsecond == 0)
        & (index.nanosecond == 0)
    )
    if bool(np.all(aligned)):
        return
    examples = [str(value) for value in index[~aligned][:3]]
    raise HourlyTargetContractError(
        "quarter_hour_prices: timestamps non alignés sur 15 minutes: "
        f"{examples}."
    )


def _assert_hour_alignment(index: pd.DatetimeIndex, name: str) -> None:
    aligned = (
        (index.minute == 0)
        & (index.second == 0)
        & (index.microsecond == 0)
        & (index.nanosecond == 0)
    )
    if bool(np.all(aligned)):
        return
    examples = [str(value) for value in index[~aligned][:3]]
    raise HourlyTargetContractError(
        f"{name}: timestamps non alignés sur l'heure: {examples}."
    )


def _quarter_hour_aggregation_table(
    quarter_hour_prices: pd.Series,
) -> pd.DataFrame:
    values = _as_utc_numeric_series(
        quarter_hour_prices,
        series_name="quarter_hour_prices",
    )
    if values.empty:
        return pd.DataFrame(
            {
                "price_hourly": pd.Series(dtype=float),
                "quarter_count": pd.Series(dtype="int64"),
                "valid_quarter_count": pd.Series(dtype="int64"),
                "is_complete": pd.Series(dtype=bool),
            },
            index=pd.DatetimeIndex([], tz="UTC", name="delivery_start_utc"),
        )

    _assert_quarter_hour_alignment(values.index)
    frame = values.rename("price").to_frame()
    aggregates = frame.resample(
        "1h",
        origin="epoch",
        label="left",
        closed="left",
    )["price"].agg(["mean", "size", "count"])
    aggregates = aggregates.rename(
        columns={
            "mean": "price_hourly",
            "size": "quarter_count",
            "count": "valid_quarter_count",
        }
    )
    aggregates["quarter_count"] = aggregates["quarter_count"].astype(
        "int64"
    )
    aggregates["valid_quarter_count"] = aggregates[
        "valid_quarter_count"
    ].astype("int64")
    aggregates["is_complete"] = (
        aggregates["quarter_count"].eq(4)
        & aggregates["valid_quarter_count"].eq(4)
    )
    aggregates.loc[
        ~aggregates["is_complete"], "price_hourly"
    ] = np.nan
    aggregates.index.name = "delivery_start_utc"
    return aggregates


def aggregate_quarter_hour_prices(
    quarter_hour_prices: pd.Series,
    *,
    incomplete: IncompletePolicy = "raise",
    name: str = "price_hourly",
) -> pd.Series:
    """Aggregate four QH prices to the arithmetic hourly delivery price.

    The input may use any timezone, but it must be timezone-aware and aligned
    to ``00/15/30/45`` minutes.  Grouping is performed after conversion to UTC,
    which makes both occurrences of the autumn 02:00 local hour unambiguous.

    Parameters
    ----------
    quarter_hour_prices:
        Quarter-hour prices indexed by delivery start.
    incomplete:
        ``"raise"`` rejects any hour without four non-null quarters;
        ``"nan"`` keeps the hour with a null target; ``"drop"`` removes it.
        None of the modes interpolates or forward-fills the target.
    name:
        Name assigned to the returned series.
    """

    if incomplete not in {"raise", "nan", "drop"}:
        raise ValueError(
            "incomplete doit être 'raise', 'nan' ou 'drop'."
        )

    aggregates = _quarter_hour_aggregation_table(quarter_hour_prices)
    incomplete_mask = ~aggregates["is_complete"]
    if incomplete == "raise" and bool(incomplete_mask.any()):
        examples = [
            str(value) for value in aggregates.index[incomplete_mask][:5]
        ]
        raise HourlyTargetContractError(
            "Agrégation impossible sans interpolation: "
            f"{int(incomplete_mask.sum())} heure(s) incomplète(s), "
            f"exemples={examples}."
        )

    result = aggregates["price_hourly"].copy()
    if incomplete == "drop":
        result = result.loc[~incomplete_mask]
    result.name = name
    result.index.name = "delivery_start_utc"
    return result


def coerce_hourly_target(
    target: pd.Series,
    *,
    input_resolution: TargetInputResolution = "auto",
    incomplete: IncompletePolicy = "raise",
    name: str = "target",
) -> pd.Series:
    """Return a canonical UTC hourly target without imputing any price.

    ``quarter_hour`` applies the arithmetic mean of the four 15-minute MTUs.
    ``hourly`` only validates and normalises the index. ``auto`` accepts only
    an unambiguous all-hourly or all-quarter-hour timestamp grid; it never
    guesses from values or silently discards sub-hourly observations.
    """

    if input_resolution not in {"auto", "hourly", "quarter_hour"}:
        raise ValueError(
            "input_resolution doit être 'auto', 'hourly' ou 'quarter_hour'."
        )
    values = _as_utc_numeric_series(target, series_name="target")
    resolution = input_resolution
    if resolution == "auto":
        hourly_aligned = bool(
            np.all(
                (values.index.minute == 0)
                & (values.index.second == 0)
                & (values.index.microsecond == 0)
                & (values.index.nanosecond == 0)
            )
        )
        resolution = "hourly" if hourly_aligned else "quarter_hour"

    if resolution == "quarter_hour":
        return aggregate_quarter_hour_prices(
            values,
            incomplete=incomplete,
            name=name,
        )

    _assert_hour_alignment(values.index, "target")
    result = values.rename(name)
    result.index.name = "delivery_start_utc"
    return result


def validate_hourly_target(
    existing_hourly: pd.Series,
    quarter_hour_prices: pd.Series,
    *,
    atol: float = 1e-8,
    rtol: float = 0.0,
    raise_on_error: bool = False,
) -> HourlyTargetValidation:
    """Validate an existing hourly target against QH arithmetic means.

    The comparison is an outer join in UTC.  Missing values, incomplete QH
    blocks, and additional/missing timestamps therefore cannot disappear via
    an inner join.  No target value is imputed during validation.
    """

    if atol < 0 or rtol < 0:
        raise ValueError("atol et rtol doivent être positifs ou nuls.")

    existing = _as_utc_numeric_series(
        existing_hourly,
        series_name="existing_hourly",
    )
    _assert_hour_alignment(existing.index, "existing_hourly")
    aggregate_table = _quarter_hour_aggregation_table(quarter_hour_prices)
    derived = aggregate_table["price_hourly"]

    details = pd.concat(
        [
            existing.rename("existing_hourly"),
            derived.rename("derived_hourly"),
        ],
        axis=1,
        join="outer",
    ).sort_index()
    details = details.join(
        aggregate_table[
            ["quarter_count", "valid_quarter_count", "is_complete"]
        ],
        how="left",
    )
    details["is_complete"] = details["is_complete"].fillna(False)

    existing_present = details["existing_hourly"].notna()
    derived_present = details["derived_hourly"].notna()
    both_present = existing_present & derived_present
    details["missing_in_existing"] = derived_present & ~existing_present
    details["missing_in_derived"] = existing_present & ~derived_present
    details["abs_error"] = (
        details["existing_hourly"] - details["derived_hourly"]
    ).abs()
    tolerance = atol + rtol * details["derived_hourly"].abs()
    details["mismatch"] = both_present & details["abs_error"].gt(
        tolerance
    )

    compared = int(both_present.sum())
    missing_existing = int(details["missing_in_existing"].sum())
    missing_derived = int(details["missing_in_derived"].sum())
    incomplete_quarters = int(
        (~aggregate_table["is_complete"]).sum()
    )
    mismatches = int(details["mismatch"].sum())
    valid_errors = details.loc[both_present, "abs_error"]
    max_abs_error = (
        float(valid_errors.max()) if not valid_errors.empty else float("nan")
    )
    mean_abs_error = (
        float(valid_errors.mean()) if not valid_errors.empty else float("nan")
    )
    is_valid = (
        compared > 0
        and missing_existing == 0
        and missing_derived == 0
        and incomplete_quarters == 0
        and mismatches == 0
    )
    report = HourlyTargetValidation(
        is_valid=is_valid,
        compared_hours=compared,
        missing_in_existing=missing_existing,
        missing_in_derived=missing_derived,
        incomplete_quarter_hours=incomplete_quarters,
        mismatched_hours=mismatches,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        details=details,
    )
    if raise_on_error:
        report.raise_if_invalid()
    return report


def local_delivery_day_index(
    delivery_date: str | date | pd.Timestamp,
    *,
    timezone: str = "Europe/Paris",
) -> pd.DatetimeIndex:
    """Return the canonical UTC hourly index of one local delivery day.

    The interval is ``[local midnight, next local midnight)``.  It therefore
    contains 23 timestamps at the spring transition, 25 at the autumn
    transition, and 24 otherwise.
    """

    timestamp = pd.Timestamp(delivery_date)
    if timestamp.tzinfo is not None:
        local_date = timestamp.tz_convert(timezone).date()
    else:
        local_date = timestamp.date()

    start_local = pd.Timestamp(local_date).tz_localize(timezone)
    end_local = start_local + pd.DateOffset(days=1)
    result = pd.date_range(
        start=start_local.tz_convert("UTC"),
        end=end_local.tz_convert("UTC"),
        freq="1h",
        inclusive="left",
        name="delivery_start_utc",
    )
    if len(result) not in {23, 24, 25}:
        raise HourlyTargetContractError(
            f"Journée locale invalide: {local_date} contient {len(result)} h."
        )
    return result


def build_delivery_metadata(
    delivery_index: pd.DatetimeIndex,
    *,
    timezone: str = "Europe/Paris",
) -> pd.DataFrame:
    """Build explicit UTC/local/offset/fold metadata for hourly delivery."""

    if not isinstance(delivery_index, pd.DatetimeIndex):
        raise TypeError("delivery_index doit être un pandas.DatetimeIndex.")
    if delivery_index.tz is None:
        raise HourlyTargetContractError(
            "delivery_index doit être timezone-aware."
        )
    if delivery_index.has_duplicates:
        raise HourlyTargetContractError(
            "delivery_index contient des timestamps dupliqués."
        )

    utc_index = delivery_index.tz_convert("UTC").sort_values()
    _assert_hour_alignment(utc_index, "delivery_index")
    local_index = utc_index.tz_convert(timezone)
    local_datetimes = local_index.to_pydatetime()
    utc_offsets_minutes = [
        int(value.utcoffset().total_seconds() // 60)
        for value in local_datetimes
    ]
    folds = [int(value.fold) for value in local_datetimes]
    offset_labels = [
        f"{minutes // 60:+03d}:{abs(minutes) % 60:02d}"
        for minutes in utc_offsets_minutes
    ]

    metadata = pd.DataFrame(
        {
            "delivery_start_utc": utc_index,
            "delivery_start_local": local_index,
            "utc_offset": offset_labels,
            "utc_offset_minutes": utc_offsets_minutes,
            "fold": folds,
            "local_date": [value.date() for value in local_datetimes],
            "local_hour": [value.hour for value in local_datetimes],
        },
        index=utc_index,
    )
    metadata.index.name = "delivery_start_utc"
    metadata["delivery_hour_position"] = (
        metadata.groupby("local_date", sort=False).cumcount() + 1
    )
    metadata["hours_in_local_day"] = metadata.groupby(
        "local_date", sort=False
    )["local_hour"].transform("size")
    return metadata


def delivery_day_metadata(
    delivery_date: str | date | pd.Timestamp,
    *,
    timezone: str = "Europe/Paris",
) -> pd.DataFrame:
    """Return canonical hourly timestamps and metadata for a local day."""

    return build_delivery_metadata(
        local_delivery_day_index(delivery_date, timezone=timezone),
        timezone=timezone,
    )
