"""Hourly day-ahead forecasting contracts.

The package deliberately keeps UTC as its canonical timeline.  Local delivery
timestamps are exposed as metadata so that 23/24/25-hour Europe/Paris days are
represented without inventing, dropping, or interpolating target values.
"""

from .hourly_contract import (
    HourlyTargetContractError,
    HourlyTargetValidation,
    aggregate_quarter_hour_prices,
    build_delivery_metadata,
    coerce_hourly_target,
    delivery_day_metadata,
    local_delivery_day_index,
    validate_hourly_target,
)

__all__ = [
    "HourlyTargetContractError",
    "HourlyTargetValidation",
    "aggregate_quarter_hour_prices",
    "build_delivery_metadata",
    "coerce_hourly_target",
    "delivery_day_metadata",
    "local_delivery_day_index",
    "validate_hourly_target",
]
