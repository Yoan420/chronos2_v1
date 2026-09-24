"""Point-in-time French supply-stack features for hourly forecasting.

The module deliberately performs no resampling, imputation, forward fill or
backward fill.  Every feature at row ``t`` uses only values from row ``t`` and,
for ramps/rolling summaries, rows at or before ``t``.  The caller remains
responsible for providing forecast vintages that were available at the chosen
forecast cutoff.

All power columns must use the same unit (MW or GW).  Marginal costs are
returned in EUR/MWh_e.  TTF must be expressed in EUR/MWh_th, EUA in EUR/tCO2,
API2 in USD/t and EUR/USD as USD per EUR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd


NanPolicy = Literal["propagate", "raise"]


@dataclass(frozen=True)
class SupplyStackColumns:
    """Input-column contract for :func:`build_fr_supply_stack_features`.

    Coal can be supplied in exactly one of two forms:

    * leave ``coal_eur_mwh_th`` at ``None`` and provide API2 plus EUR/USD;
    * configure ``coal_eur_mwh_th`` and set both API2/FX fields to ``None``.
    """

    load_forecast: str = "load_forecast"
    wind_forecast: str = "wind_forecast"
    solar_forecast: str = "solar_forecast"
    nuclear_available: str = "nuclear_available"
    thermal_available: str = "thermal_available"
    hydro_firm: str = "hydro_firm"
    import_headroom: str = "import_headroom"
    export_headroom: str = "export_headroom"
    pumping_capacity: str = "pumping_capacity"
    nuclear_minimum: str = "nuclear_minimum"
    run_of_river: str = "run_of_river"
    ttf_eur_mwh_th: str = "ttf_eur_mwh_th"
    eua_eur_tco2: str = "eua_eur_tco2"
    api2_usd_tonne: str | None = "api2_usd_tonne"
    eurusd_usd_per_eur: str | None = "eurusd_usd_per_eur"
    coal_eur_mwh_th: str | None = None


@dataclass(frozen=True)
class SupplyStackParameters:
    """Technical and variable-cost assumptions used by the merit stack."""

    ccgt_efficiency: float = 0.58
    ocgt_efficiency: float = 0.39
    coal_efficiency: float = 0.40
    ccgt_emission_tco2_mwh: float = 0.36
    ocgt_emission_tco2_mwh: float = 0.55
    coal_emission_tco2_mwh: float = 0.90
    ccgt_vom_eur_mwh: float = 3.0
    ocgt_vom_eur_mwh: float = 5.0
    coal_vom_eur_mwh: float = 4.0
    coal_mwh_th_per_tonne: float = 8.141

    def validate(self) -> None:
        efficiencies = {
            "ccgt_efficiency": self.ccgt_efficiency,
            "ocgt_efficiency": self.ocgt_efficiency,
            "coal_efficiency": self.coal_efficiency,
        }
        invalid_efficiencies = [
            name
            for name, value in efficiencies.items()
            if not np.isfinite(value) or not 0.0 < value <= 1.0
        ]
        if invalid_efficiencies:
            raise ValueError(
                "Efficiencies must be finite and in ]0, 1]: "
                + ", ".join(invalid_efficiencies)
            )
        if (
            not np.isfinite(self.coal_mwh_th_per_tonne)
            or self.coal_mwh_th_per_tonne <= 0.0
        ):
            raise ValueError("coal_mwh_th_per_tonne must be positive.")


def _normalise_windows(windows: Sequence[int]) -> tuple[int, ...]:
    normalised: list[int] = []
    for raw_window in windows:
        if isinstance(raw_window, bool):
            raise ValueError("rolling_windows must contain positive integers.")
        window = int(raw_window)
        if window != raw_window or window <= 0:
            raise ValueError("rolling_windows must contain positive integers.")
        if window not in normalised:
            normalised.append(window)
    return tuple(normalised)


def _coal_input_columns(columns: SupplyStackColumns) -> tuple[str, ...]:
    direct = columns.coal_eur_mwh_th is not None
    converted = (
        columns.api2_usd_tonne is not None
        and columns.eurusd_usd_per_eur is not None
    )
    partially_converted = (
        columns.api2_usd_tonne is None
    ) != (
        columns.eurusd_usd_per_eur is None
    )
    if partially_converted or direct == converted:
        raise ValueError(
            "Configure exactly one coal-price mode: coal_eur_mwh_th, or "
            "api2_usd_tonne together with eurusd_usd_per_eur."
        )
    if direct:
        return (str(columns.coal_eur_mwh_th),)
    return (
        str(columns.api2_usd_tonne),
        str(columns.eurusd_usd_per_eur),
    )


def required_supply_stack_columns(
    columns: SupplyStackColumns | None = None,
) -> tuple[str, ...]:
    """Return the complete input contract for a column configuration."""

    configured = columns or SupplyStackColumns()
    base = (
        configured.load_forecast,
        configured.wind_forecast,
        configured.solar_forecast,
        configured.nuclear_available,
        configured.thermal_available,
        configured.hydro_firm,
        configured.import_headroom,
        configured.export_headroom,
        configured.pumping_capacity,
        configured.nuclear_minimum,
        configured.run_of_river,
        configured.ttf_eur_mwh_th,
        configured.eua_eur_tco2,
    )
    required = base + _coal_input_columns(configured)
    duplicates = sorted(
        {name for name in required if required.count(name) > 1}
    )
    if duplicates:
        raise ValueError(
            "Each supply-stack input must map to a distinct column: "
            + ", ".join(duplicates)
        )
    return required


def _numeric_inputs(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    nan_policy: NanPolicy,
) -> pd.DataFrame:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(
            "Missing French supply-stack inputs: " + ", ".join(missing)
        )

    numeric = frame.loc[:, list(required)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    malformed = frame.loc[:, list(required)].notna() & numeric.isna()
    if malformed.any().any():
        bad_columns = malformed.columns[malformed.any()].tolist()
        raise TypeError(
            "Non-numeric supply-stack values found in: "
            + ", ".join(bad_columns)
        )

    numeric = numeric.astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if nan_policy == "raise" and numeric.isna().any().any():
        counts = numeric.isna().sum()
        details = ", ".join(
            f"{name}={int(count)}"
            for name, count in counts.items()
            if count
        )
        raise ValueError("Missing/non-finite supply-stack inputs: " + details)
    return numeric


def _validate_nan_policy(nan_policy: str) -> NanPolicy:
    if nan_policy not in {"propagate", "raise"}:
        raise ValueError("nan_policy must be 'propagate' or 'raise'.")
    return nan_policy  # type: ignore[return-value]


def build_fr_supply_stack_features(
    frame: pd.DataFrame,
    *,
    columns: SupplyStackColumns | None = None,
    parameters: SupplyStackParameters | None = None,
    rolling_windows: Sequence[int] = (3, 6, 24),
    nan_policy: NanPolicy = "propagate",
) -> pd.DataFrame:
    """Build causal, hourly French merit-stack features.

    Parameters
    ----------
    frame:
        Point-in-time inputs ordered in forecast/delivery time.  The function
        never sorts the rows so that accidental reordering cannot mask a bad
        upstream data contract.
    columns:
        Mapping from economic inputs to columns in ``frame``.
    parameters:
        Efficiencies, emissions and variable O&M assumptions.
    rolling_windows:
        Trailing row windows.  ``min_periods=window`` is always used: partial
        or input-incomplete windows stay NaN rather than being imputed.
    nan_policy:
        ``"propagate"`` keeps missing inputs as missing derived features.
        ``"raise"`` fails before feature construction.  Neither mode fills.

    Returns
    -------
    pandas.DataFrame
        Feature-only frame with exactly the same index and row order as input.

    Notes
    -----
    Causality here means row causality.  It does not prove that the caller's
    input vintage existed at the operational cutoff; that must be enforced by
    the upstream point-in-time loader.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be monotonic increasing.")

    policy = _validate_nan_policy(nan_policy)
    configured = columns or SupplyStackColumns()
    assumptions = parameters or SupplyStackParameters()
    assumptions.validate()
    windows = _normalise_windows(rolling_windows)
    required = required_supply_stack_columns(configured)
    values = _numeric_inputs(frame, required, nan_policy=policy)

    load = values[configured.load_forecast]
    wind = values[configured.wind_forecast]
    solar = values[configured.solar_forecast]
    nuclear_available = values[configured.nuclear_available]
    thermal_available = values[configured.thermal_available]
    hydro_firm = values[configured.hydro_firm]
    import_headroom = values[configured.import_headroom]
    export_headroom = values[configured.export_headroom]
    pumping_capacity = values[configured.pumping_capacity]
    nuclear_minimum = values[configured.nuclear_minimum]
    run_of_river = values[configured.run_of_river]
    ttf = values[configured.ttf_eur_mwh_th]
    eua = values[configured.eua_eur_tco2]

    if configured.coal_eur_mwh_th is not None:
        coal_fuel = values[configured.coal_eur_mwh_th]
    else:
        api2 = values[str(configured.api2_usd_tonne)]
        eurusd = values[str(configured.eurusd_usd_per_eur)]
        invalid_fx = eurusd <= 0.0
        if policy == "raise" and bool(invalid_fx.fillna(False).any()):
            raise ValueError("eurusd_usd_per_eur must be strictly positive.")
        safe_fx = eurusd.where(eurusd > 0.0)
        coal_fuel = api2 / safe_fx / assumptions.coal_mwh_th_per_tonne

    result = pd.DataFrame(index=frame.index.copy())
    result["renewable_forecast"] = wind + solar + run_of_river
    result["residual_load"] = load - wind - solar
    result["firm_available_supply"] = (
        nuclear_available
        + thermal_available
        + hydro_firm
        + import_headroom
    )
    result["firm_margin"] = (
        result["firm_available_supply"] - result["residual_load"]
    )
    result["negative_pressure"] = (
        nuclear_minimum
        + run_of_river
        + wind
        + solar
        - load
        - export_headroom
        - pumping_capacity
    )
    result["coal_fuel_cost_eur_mwh_th"] = coal_fuel
    result["ccgt_marginal_cost"] = (
        ttf / assumptions.ccgt_efficiency
        + eua * assumptions.ccgt_emission_tco2_mwh
        + assumptions.ccgt_vom_eur_mwh
    )
    result["ocgt_marginal_cost"] = (
        ttf / assumptions.ocgt_efficiency
        + eua * assumptions.ocgt_emission_tco2_mwh
        + assumptions.ocgt_vom_eur_mwh
    )
    result["coal_marginal_cost"] = (
        coal_fuel / assumptions.coal_efficiency
        + eua * assumptions.coal_emission_tco2_mwh
        + assumptions.coal_vom_eur_mwh
    )
    result["gas_coal_switch_spread"] = (
        result["ccgt_marginal_cost"] - result["coal_marginal_cost"]
    )

    # One-sided pressure variables retain sign information in firm_margin and
    # negative_pressure while exposing the economically nonlinear tails.
    result["scarcity_gap"] = (-result["firm_margin"]).clip(lower=0.0)
    result["surplus_gap"] = result["negative_pressure"].clip(lower=0.0)

    ramp_sources = (
        "residual_load",
        "firm_margin",
        "negative_pressure",
        "renewable_forecast",
        "ccgt_marginal_cost",
        "coal_marginal_cost",
    )
    for name in ramp_sources:
        ramp = result[name].diff(periods=1)
        result[f"{name}_ramp_1"] = ramp
        result[f"{name}_abs_ramp_1"] = ramp.abs()

    for window in windows:
        residual = result["residual_load"].rolling(
            window=window,
            min_periods=window,
        )
        firm = result["firm_margin"].rolling(
            window=window,
            min_periods=window,
        )
        negative = result["negative_pressure"].rolling(
            window=window,
            min_periods=window,
        )
        result[f"residual_load_trailing_mean_{window}"] = residual.mean()
        result[f"residual_load_trailing_min_{window}"] = residual.min()
        result[f"residual_load_trailing_max_{window}"] = residual.max()
        result[f"residual_load_trailing_std_{window}"] = residual.std(ddof=0)
        result[f"residual_load_trailing_range_{window}"] = (
            result[f"residual_load_trailing_max_{window}"]
            - result[f"residual_load_trailing_min_{window}"]
        )
        result[f"firm_margin_trailing_mean_{window}"] = firm.mean()
        result[f"firm_margin_trailing_min_{window}"] = firm.min()
        result[f"negative_pressure_trailing_mean_{window}"] = negative.mean()
        result[f"negative_pressure_trailing_max_{window}"] = negative.max()

    # Economic interactions are deliberately unnormalised.  Scaling belongs
    # in each model pipeline and must be learned on its training fold only.
    result["residual_load_x_ccgt_cost"] = (
        result["residual_load"] * result["ccgt_marginal_cost"]
    )
    result["firm_margin_x_ccgt_cost"] = (
        result["firm_margin"] * result["ccgt_marginal_cost"]
    )
    result["scarcity_x_ocgt_cost"] = (
        result["scarcity_gap"] * result["ocgt_marginal_cost"]
    )
    result["surplus_x_export_headroom"] = (
        result["surplus_gap"] * export_headroom
    )
    result["negative_pressure_x_pumping"] = (
        result["negative_pressure"] * pumping_capacity
    )

    # Do not silently turn overflow into a valid-looking model input.
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


__all__ = [
    "NanPolicy",
    "SupplyStackColumns",
    "SupplyStackParameters",
    "build_fr_supply_stack_features",
    "required_supply_stack_columns",
]
