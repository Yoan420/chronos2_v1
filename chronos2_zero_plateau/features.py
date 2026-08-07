
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResolvedAliases:
    residual_load: str
    solar: str | None
    nuclear: str | None
    export: str | None


def resolve_alias(
    columns: Iterable[str],
    *,
    exact: list[str],
    contains_all: list[str] | None = None,
) -> str | None:
    names = [str(column) for column in columns]
    lookup = {name.lower(): name for name in names}

    for candidate in exact:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if contains_all:
        tokens = [value.lower() for value in contains_all]
        for name in names:
            lowered = name.lower()
            if all(token in lowered for token in tokens):
                return name
    return None


def resolve_plateau_aliases(columns: Iterable[str]) -> ResolvedAliases:
    residual = resolve_alias(
        columns,
        exact=[
            "fr_residual_load_fcst",
            "residual_load_gw_fcst",
            "residual_load_fcst",
        ],
        contains_all=["residual", "load", "fcst"],
    )
    if residual is None:
        raise KeyError("Aucune covariable residual load forecast détectée.")

    solar = resolve_alias(
        columns,
        exact=[
            "fr_solar_generation_fcst",
            "solar_generation_fcst",
            "solar_fcst",
            "solar_lf_fcst",
        ],
        contains_all=["solar", "fcst"],
    )
    nuclear = resolve_alias(
        columns,
        exact=[
            "fr_nuclear_generation_fcst",
            "nuclear_generation_fcst",
        ],
        contains_all=["nuclear", "fcst"],
    )
    export = resolve_alias(
        columns,
        exact=[
            "da_cap_system_asymmetry",
            "export_capacity_pressure",
            "net_export_capacity",
        ],
    )
    return ResolvedAliases(residual, solar, nuclear, export)


def _numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _robust_scale(values: pd.Series, fallback: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").abs().dropna()
    if clean.empty:
        return fallback
    return max(float(clean.quantile(0.95)), fallback)


def build_plateau_features(
    model_context: pd.DataFrame,
    structural_raw: pd.DataFrame | None,
    *,
    train_mask,
    solar_start_hour: int = 8,
    solar_end_hour: int = 19,
    valley_band_gw: float = 3.0,
) -> tuple[pd.DataFrame, ResolvedAliases, dict[str, float]]:
    frame = model_context.copy()
    aliases = resolve_plateau_aliases(frame.columns)

    residual = _numeric(frame, aliases.residual_load)
    solar = _numeric(frame, aliases.solar)
    nuclear = _numeric(frame, aliases.nuclear)
    export = _numeric(frame, aliases.export)

    index = pd.DatetimeIndex(frame.index)
    hours = index.hour.to_numpy(dtype=float)
    days = index.normalize()

    result = pd.DataFrame(index=index)
    result["zp_residual_load"] = residual
    result["zp_nuclear_fcst"] = nuclear
    result["zp_export_pressure"] = export
    result["zp_solar_fcst"] = solar
    result["zp_solar_available"] = float(aliases.solar is not None)

    # Proxy solaire purement calendaire en secours si le forecast solaire
    # n'est pas déjà présent dans le YAML.
    result["zp_solar_clock_proxy"] = np.maximum(
        0.0,
        np.sin(np.pi * (hours - 6.0) / 12.0),
    )
    result["zp_hour_sin"] = np.sin(2 * np.pi * hours / 24)
    result["zp_hour_cos"] = np.cos(2 * np.pi * hours / 24)
    result["zp_is_solar_window"] = (
        (hours >= solar_start_hour) & (hours <= solar_end_hour)
    ).astype(float)

    grouped = result["zp_residual_load"].groupby(days)
    result["zp_residual_daily_min"] = grouped.transform("min")
    result["zp_residual_daily_mean"] = grouped.transform("mean")
    result["zp_residual_valley_gap"] = (
        result["zp_residual_load"] - result["zp_residual_daily_min"]
    )

    valley_flag = (
        result["zp_residual_load"]
        <= result["zp_residual_daily_min"] + valley_band_gw
    ) & (result["zp_is_solar_window"] > 0)
    result["zp_residual_valley_flag"] = valley_flag.astype(float)
    result["zp_residual_valley_width"] = (
        result["zp_residual_valley_flag"].groupby(days).transform("sum")
    )

    # La courbe day-ahead complète étant disponible au cutoff, les pentes
    # internes à D sont des features point-in-time valides.
    result["zp_residual_slope_1h"] = (
        result["zp_residual_load"].groupby(days).diff()
    )
    result["zp_residual_forward_slope_1h"] = (
        result["zp_residual_load"].groupby(days).shift(-1)
        - result["zp_residual_load"]
    )

    structural = (
        structural_raw.reindex(index).copy()
        if structural_raw is not None
        else pd.DataFrame(index=index)
    )
    for source, target in (
        ("milp_structural_price", "zp_milp_price"),
        ("milp_spill_gw", "zp_milp_spill_gw"),
        ("milp_reserve_margin_gw", "zp_milp_reserve_margin_gw"),
        ("milp_committed_thermal_gw", "zp_milp_committed_thermal_gw"),
        ("milp_startups", "zp_milp_startups"),
    ):
        result[target] = _numeric(structural, source)

    train_mask_arr = np.asarray(train_mask, dtype=bool)
    if len(train_mask_arr) != len(result):
        raise ValueError("train_mask de taille incompatible.")

    train_frame = result.loc[train_mask_arr]
    scales = {
        "spill": _robust_scale(train_frame["zp_milp_spill_gw"], 0.25),
        "reserve": _robust_scale(
            train_frame["zp_milp_reserve_margin_gw"], 1.0
        ),
    }

    result["zp_milp_zero_pressure"] = np.exp(
        -np.abs(result["zp_milp_price"]) / 12.0
    )
    result["zp_milp_spill_pressure"] = (
        result["zp_milp_spill_gw"].clip(lower=0) / scales["spill"]
    ).clip(0, 4)
    result["zp_milp_reserve_abundance"] = (
        result["zp_milp_reserve_margin_gw"].clip(lower=0) / scales["reserve"]
    ).clip(0, 4)

    valley_pressure = np.exp(
        -result["zp_residual_valley_gap"].clip(lower=0)
        / max(valley_band_gw, 0.25)
    )
    result["zp_milp_oversupply_pressure"] = (
        0.35 * result["zp_milp_zero_pressure"]
        + 0.30 * result["zp_milp_spill_pressure"]
        + 0.20 * result["zp_milp_reserve_abundance"].clip(0, 1)
        + 0.15 * valley_pressure
    )

    result["zp_residual_slope_1h"] = result["zp_residual_slope_1h"].fillna(0)
    result["zp_residual_forward_slope_1h"] = (
        result["zp_residual_forward_slope_1h"].fillna(0)
    )

    return result.astype(np.float32), aliases, scales
