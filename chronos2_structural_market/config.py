from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class InputSpec:
    alias: str | None
    scale: float = 1.0
    default: float | None = None


@dataclass(frozen=True)
class TechnologySpec:
    name: str
    commitment: bool
    n_units: int
    unit_size_gw: float
    min_output_fraction: float
    fixed_variable_cost_eur_mwh: float
    variable_om_eur_mwh: float
    startup_cost_eur: float
    no_load_cost_eur_h: float
    ramp_up_gw_h_per_unit: float
    ramp_down_gw_h_per_unit: float
    availability_input: str | None = None
    availability_fallback_gw: float | None = None
    energy_budget_fraction: float | None = None
    fuel_input: str | None = None
    efficiency: float | None = None
    emission_factor_t_mwh: float = 0.0
    eua_input: str | None = "eua_price_eur_t"

    @property
    def installed_capacity_gw(self) -> float:
        return float(self.n_units) * float(self.unit_size_gw)


@dataclass(frozen=True)
class SolverSpec:
    mip_rel_gap: float = 0.001
    time_limit_seconds: float = 30.0
    node_limit: int | None = None
    presolve: bool = True


@dataclass(frozen=True)
class StructuralModelConfig:
    enabled: bool
    inputs: dict[str, InputSpec]
    technologies: tuple[TechnologySpec, ...]
    reserve_fraction: float
    reserve_floor_gw: float
    voll_eur_mwh: float
    spill_cost_eur_mwh: float
    reserve_shortfall_cost_eur_mwh: float
    initial_online_fraction: float
    solver: SolverSpec
    output_file: str
    diagnostics_file: str
    residual_price_alias: str
    feature_aliases: tuple[str, ...]
    impute_residual_load: bool = True
    max_residual_load_missing_hours_per_day: int = 2


DEFAULT_FEATURE_ALIASES = (
    "milp_structural_price",
    "milp_marginal_technology_code",
    "milp_reserve_margin_gw",
    "milp_committed_thermal_gw",
    "milp_online_units",
    "milp_startups",
    "milp_startup_cost_eur",
    "milp_scarcity_gw",
    "milp_ramp_shadow_eur_mwh",
    "milp_reserve_shadow_eur_mwh",
)


def _default_technologies() -> list[dict[str, Any]]:
    return [
        {
            "name": "nuclear",
            "commitment": False,
            "n_units": 1,
            "unit_size_gw": 45.0,
            "min_output_fraction": 0.0,
            "fixed_variable_cost_eur_mwh": 10.0,
            "variable_om_eur_mwh": 0.0,
            "startup_cost_eur": 0.0,
            "no_load_cost_eur_h": 0.0,
            "ramp_up_gw_h_per_unit": 6.0,
            "ramp_down_gw_h_per_unit": 6.0,
            "availability_input": "nuclear_capacity_gw",
            "availability_fallback_gw": 45.0,
        },
        {
            "name": "hydro",
            "commitment": False,
            "n_units": 1,
            "unit_size_gw": 11.0,
            "min_output_fraction": 0.0,
            "fixed_variable_cost_eur_mwh": 25.0,
            "variable_om_eur_mwh": 0.0,
            "startup_cost_eur": 0.0,
            "no_load_cost_eur_h": 0.0,
            "ramp_up_gw_h_per_unit": 5.0,
            "ramp_down_gw_h_per_unit": 5.0,
            "availability_fallback_gw": 11.0,
            "energy_budget_fraction": 0.45,
        },
        {
            "name": "coal",
            "commitment": True,
            "n_units": 4,
            "unit_size_gw": 0.60,
            "min_output_fraction": 0.40,
            "fixed_variable_cost_eur_mwh": 90.0,
            "variable_om_eur_mwh": 5.0,
            "startup_cost_eur": 35000.0,
            "no_load_cost_eur_h": 1400.0,
            "ramp_up_gw_h_per_unit": 0.25,
            "ramp_down_gw_h_per_unit": 0.25,
            "fuel_input": "coal_price_eur_mwhth",
            "efficiency": 0.38,
            "emission_factor_t_mwh": 0.90,
        },
        {
            "name": "ccgt",
            "commitment": True,
            "n_units": 20,
            "unit_size_gw": 0.60,
            "min_output_fraction": 0.35,
            "fixed_variable_cost_eur_mwh": 95.0,
            "variable_om_eur_mwh": 3.0,
            "startup_cost_eur": 18000.0,
            "no_load_cost_eur_h": 700.0,
            "ramp_up_gw_h_per_unit": 0.40,
            "ramp_down_gw_h_per_unit": 0.40,
            "fuel_input": "gas_price_eur_mwhth",
            "efficiency": 0.57,
            "emission_factor_t_mwh": 0.35,
        },
        {
            "name": "ocgt",
            "commitment": True,
            "n_units": 20,
            "unit_size_gw": 0.25,
            "min_output_fraction": 0.20,
            "fixed_variable_cost_eur_mwh": 150.0,
            "variable_om_eur_mwh": 5.0,
            "startup_cost_eur": 8000.0,
            "no_load_cost_eur_h": 250.0,
            "ramp_up_gw_h_per_unit": 0.25,
            "ramp_down_gw_h_per_unit": 0.25,
            "fuel_input": "gas_price_eur_mwhth",
            "efficiency": 0.38,
            "emission_factor_t_mwh": 0.55,
        },
        {
            "name": "oil",
            "commitment": True,
            "n_units": 8,
            "unit_size_gw": 0.25,
            "min_output_fraction": 0.20,
            "fixed_variable_cost_eur_mwh": 230.0,
            "variable_om_eur_mwh": 8.0,
            "startup_cost_eur": 12000.0,
            "no_load_cost_eur_h": 350.0,
            "ramp_up_gw_h_per_unit": 0.25,
            "ramp_down_gw_h_per_unit": 0.25,
            "efficiency": 0.35,
            "emission_factor_t_mwh": 0.75,
        },
    ]


def default_structural_model_block() -> dict[str, Any]:
    return {
        "enabled": True,
        "inputs": {
            "residual_load_gw": {
                "alias": "fr_residual_load_fcst",
                "scale": 1.0,
            },
            "nuclear_capacity_gw": {
                "alias": "fr_nuclear_generation_fcst",
                "scale": 0.001,
                "default": 45.0,
            },
            "fixed_net_exports_gw": {
                "alias": None,
                "scale": 1.0,
                "default": 0.0,
            },
            "gas_price_eur_mwhth": {
                "alias": None,
                "scale": 1.0,
                "default": None,
            },
            "coal_price_eur_mwhth": {
                "alias": None,
                "scale": 1.0,
                "default": None,
            },
            "eua_price_eur_t": {
                "alias": None,
                "scale": 1.0,
                "default": None,
            },
        },
        "reserve_fraction": 0.08,
        "reserve_floor_gw": 2.0,
        "voll_eur_mwh": 3000.0,
        "spill_cost_eur_mwh": 5.0,
        "reserve_shortfall_cost_eur_mwh": 1000.0,
        "initial_online_fraction": 0.50,
        "solver": {
            "mip_rel_gap": 0.001,
            "time_limit_seconds": 30.0,
            "node_limit": None,
            "presolve": True,
        },
        "technologies": _default_technologies(),
        "output_file": "data/derived/structural_market_features.csv.gz",
        "diagnostics_file": "data/derived/structural_market_daily_diagnostics.csv",
        "residual_price_alias": "milp_structural_price",
        "feature_aliases": list(DEFAULT_FEATURE_ALIASES),
        "residual_mode": {
            "enabled": False,
        },
        "input_repair": {
            "impute_residual_load": True,
            "max_residual_load_missing_hours_per_day": 2,
        },
    }


def _input_spec(raw: Any) -> InputSpec:
    if raw is None:
        return InputSpec(alias=None)
    if isinstance(raw, str):
        return InputSpec(alias=raw)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Input structurel invalide : {raw!r}")
    alias = raw.get("alias")
    return InputSpec(
        alias=(str(alias) if alias not in (None, "") else None),
        scale=float(raw.get("scale", 1.0)),
        default=(
            None
            if raw.get("default") in (None, "")
            else float(raw.get("default"))
        ),
    )


def _technology_spec(raw: Mapping[str, Any]) -> TechnologySpec:
    return TechnologySpec(
        name=str(raw["name"]).lower(),
        commitment=bool(raw.get("commitment", False)),
        n_units=int(raw.get("n_units", 1)),
        unit_size_gw=float(raw.get("unit_size_gw", 1.0)),
        min_output_fraction=float(raw.get("min_output_fraction", 0.0)),
        fixed_variable_cost_eur_mwh=float(
            raw.get("fixed_variable_cost_eur_mwh", 0.0)
        ),
        variable_om_eur_mwh=float(raw.get("variable_om_eur_mwh", 0.0)),
        startup_cost_eur=float(raw.get("startup_cost_eur", 0.0)),
        no_load_cost_eur_h=float(raw.get("no_load_cost_eur_h", 0.0)),
        ramp_up_gw_h_per_unit=float(
            raw.get("ramp_up_gw_h_per_unit", raw.get("unit_size_gw", 1.0))
        ),
        ramp_down_gw_h_per_unit=float(
            raw.get("ramp_down_gw_h_per_unit", raw.get("unit_size_gw", 1.0))
        ),
        availability_input=(
            str(raw["availability_input"])
            if raw.get("availability_input") not in (None, "")
            else None
        ),
        availability_fallback_gw=(
            None
            if raw.get("availability_fallback_gw") in (None, "")
            else float(raw["availability_fallback_gw"])
        ),
        energy_budget_fraction=(
            None
            if raw.get("energy_budget_fraction") in (None, "")
            else float(raw["energy_budget_fraction"])
        ),
        fuel_input=(
            str(raw["fuel_input"])
            if raw.get("fuel_input") not in (None, "")
            else None
        ),
        efficiency=(
            None
            if raw.get("efficiency") in (None, "")
            else float(raw["efficiency"])
        ),
        emission_factor_t_mwh=float(raw.get("emission_factor_t_mwh", 0.0)),
        eua_input=(
            str(raw["eua_input"])
            if raw.get("eua_input") not in (None, "")
            else None
        ),
    )


def parse_structural_model_config(config: Mapping[str, Any]) -> StructuralModelConfig:
    default = default_structural_model_block()
    raw = config.get("structural_model") or {}
    if not isinstance(raw, Mapping):
        raise TypeError("structural_model doit être un mapping YAML.")

    merged = {**default, **dict(raw)}
    inputs_raw = {**default["inputs"], **dict(raw.get("inputs") or {})}
    solver_raw = {**default["solver"], **dict(raw.get("solver") or {})}
    repair_raw = {
        **default["input_repair"],
        **dict(raw.get("input_repair") or {}),
    }
    technologies_raw = raw.get("technologies") or default["technologies"]

    technologies = tuple(_technology_spec(item) for item in technologies_raw)
    if not technologies:
        raise ValueError("Au moins une technologie doit être définie.")

    feature_aliases = tuple(
        str(value) for value in merged.get("feature_aliases", DEFAULT_FEATURE_ALIASES)
    )

    return StructuralModelConfig(
        enabled=bool(merged.get("enabled", True)),
        inputs={key: _input_spec(value) for key, value in inputs_raw.items()},
        technologies=technologies,
        reserve_fraction=float(merged.get("reserve_fraction", 0.08)),
        reserve_floor_gw=float(merged.get("reserve_floor_gw", 2.0)),
        voll_eur_mwh=float(merged.get("voll_eur_mwh", 3000.0)),
        spill_cost_eur_mwh=float(merged.get("spill_cost_eur_mwh", 5.0)),
        reserve_shortfall_cost_eur_mwh=float(
            merged.get("reserve_shortfall_cost_eur_mwh", 1000.0)
        ),
        initial_online_fraction=float(merged.get("initial_online_fraction", 0.5)),
        solver=SolverSpec(
            mip_rel_gap=float(solver_raw.get("mip_rel_gap", 0.001)),
            time_limit_seconds=float(solver_raw.get("time_limit_seconds", 30.0)),
            node_limit=(
                None
                if solver_raw.get("node_limit") in (None, "")
                else int(solver_raw["node_limit"])
            ),
            presolve=bool(solver_raw.get("presolve", True)),
        ),
        output_file=str(merged["output_file"]),
        diagnostics_file=str(merged["diagnostics_file"]),
        residual_price_alias=str(
            merged.get("residual_price_alias", "milp_structural_price")
        ),
        feature_aliases=feature_aliases,
        impute_residual_load=bool(
            repair_raw.get("impute_residual_load", True)
        ),
        max_residual_load_missing_hours_per_day=int(
            repair_raw.get(
                "max_residual_load_missing_hours_per_day",
                2,
            )
        ),
    )
