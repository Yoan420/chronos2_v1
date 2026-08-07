from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix

from .config import StructuralModelConfig, TechnologySpec


@dataclass
class Variable:
    index: int
    name: str


@dataclass
class BuiltProblem:
    c: np.ndarray
    integrality: np.ndarray
    bounds: Bounds
    A_eq: csr_matrix
    b_eq: np.ndarray
    eq_names: list[str]
    A_ub: csr_matrix
    b_ub: np.ndarray
    ub_names: list[str]
    variables: dict[str, int]
    technologies: tuple[TechnologySpec, ...]
    horizon: int
    variable_costs: dict[tuple[str, int], float]
    capacities: dict[tuple[str, int], float]
    net_demand: np.ndarray
    reserve_requirement: np.ndarray


@dataclass
class StructuralDayResult:
    features: pd.DataFrame
    dispatch: pd.DataFrame
    diagnostics: dict[str, float | int | str | bool]


class MatrixBuilder:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.c: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integrality: list[int] = []
        self.variables: dict[str, int] = {}
        self.eq_rows: list[dict[int, float]] = []
        self.eq_rhs: list[float] = []
        self.eq_names: list[str] = []
        self.ub_rows: list[dict[int, float]] = []
        self.ub_rhs: list[float] = []
        self.ub_names: list[str] = []

    def add_variable(
        self,
        name: str,
        *,
        objective: float = 0.0,
        lower: float = 0.0,
        upper: float = np.inf,
        integral: bool = False,
    ) -> int:
        if name in self.variables:
            raise KeyError(f"Variable dupliquée : {name}")
        index = len(self.names)
        self.names.append(name)
        self.variables[name] = index
        self.c.append(float(objective))
        self.lb.append(float(lower))
        self.ub.append(float(upper))
        self.integrality.append(1 if integral else 0)
        return index

    def add_eq(
        self,
        coefficients: dict[int, float],
        rhs: float,
        name: str,
    ) -> None:
        self.eq_rows.append(coefficients)
        self.eq_rhs.append(float(rhs))
        self.eq_names.append(name)

    def add_le(
        self,
        coefficients: dict[int, float],
        rhs: float,
        name: str,
    ) -> None:
        self.ub_rows.append(coefficients)
        self.ub_rhs.append(float(rhs))
        self.ub_names.append(name)

    @staticmethod
    def _matrix(rows: list[dict[int, float]], n_columns: int) -> csr_matrix:
        if not rows:
            return csr_matrix((0, n_columns), dtype=float)
        data: list[float] = []
        row_index: list[int] = []
        col_index: list[int] = []
        for row_number, coefficients in enumerate(rows):
            for column, value in coefficients.items():
                if value == 0:
                    continue
                row_index.append(row_number)
                col_index.append(column)
                data.append(float(value))
        return coo_matrix(
            (data, (row_index, col_index)),
            shape=(len(rows), n_columns),
            dtype=float,
        ).tocsr()

    def finalize(
        self,
        *,
        technologies: tuple[TechnologySpec, ...],
        horizon: int,
        variable_costs: dict[tuple[str, int], float],
        capacities: dict[tuple[str, int], float],
        net_demand: np.ndarray,
        reserve_requirement: np.ndarray,
    ) -> BuiltProblem:
        n = len(self.names)
        return BuiltProblem(
            c=np.asarray(self.c, dtype=float),
            integrality=np.asarray(self.integrality, dtype=np.uint8),
            bounds=Bounds(
                np.asarray(self.lb, dtype=float),
                np.asarray(self.ub, dtype=float),
            ),
            A_eq=self._matrix(self.eq_rows, n),
            b_eq=np.asarray(self.eq_rhs, dtype=float),
            eq_names=list(self.eq_names),
            A_ub=self._matrix(self.ub_rows, n),
            b_ub=np.asarray(self.ub_rhs, dtype=float),
            ub_names=list(self.ub_names),
            variables=dict(self.variables),
            technologies=technologies,
            horizon=horizon,
            variable_costs=variable_costs,
            capacities=capacities,
            net_demand=net_demand,
            reserve_requirement=reserve_requirement,
        )


def _finite_or_default(value: float, default: float) -> float:
    return float(value) if np.isfinite(value) else float(default)


def _capacity_for(
    technology: TechnologySpec,
    frame: pd.DataFrame,
    t: int,
) -> float:
    if technology.availability_input:
        if technology.availability_input in frame:
            value = frame.iloc[t][technology.availability_input]
            if np.isfinite(value):
                return max(0.0, float(value))
        if technology.availability_fallback_gw is not None:
            return max(0.0, technology.availability_fallback_gw)
    if technology.availability_fallback_gw is not None:
        return max(0.0, technology.availability_fallback_gw)
    return technology.installed_capacity_gw


def _variable_cost_for(
    technology: TechnologySpec,
    frame: pd.DataFrame,
    t: int,
) -> float:
    fixed = float(technology.fixed_variable_cost_eur_mwh)
    if not technology.fuel_input or technology.efficiency in (None, 0):
        return fixed + technology.variable_om_eur_mwh

    if technology.fuel_input not in frame:
        return fixed + technology.variable_om_eur_mwh

    fuel = frame.iloc[t][technology.fuel_input]
    if not np.isfinite(fuel):
        return fixed + technology.variable_om_eur_mwh

    eua = 0.0
    if technology.eua_input and technology.eua_input in frame:
        candidate = frame.iloc[t][technology.eua_input]
        if np.isfinite(candidate):
            eua = float(candidate)

    cost = (
        float(fuel) / float(technology.efficiency)
        + float(technology.emission_factor_t_mwh) * eua
        + float(technology.variable_om_eur_mwh)
    )
    return max(-500.0, min(3000.0, cost))


def build_problem(
    frame: pd.DataFrame,
    config: StructuralModelConfig,
) -> BuiltProblem:
    if frame.empty:
        raise ValueError("Le profil journalier est vide.")
    if "residual_load_gw" not in frame:
        raise KeyError("Entrée requise absente : residual_load_gw")

    horizon = len(frame)
    residual_load = pd.to_numeric(
        frame["residual_load_gw"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(residual_load).all():
        raise ValueError("residual_load_gw contient des valeurs manquantes.")

    fixed_exports = (
        pd.to_numeric(frame["fixed_net_exports_gw"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        if "fixed_net_exports_gw" in frame
        else np.zeros(horizon, dtype=float)
    )
    net_demand = residual_load + fixed_exports
    reserve_requirement = np.maximum(
        config.reserve_floor_gw,
        np.maximum(net_demand, 0.0) * config.reserve_fraction,
    )

    builder = MatrixBuilder()
    variable_costs: dict[tuple[str, int], float] = {}
    capacities: dict[tuple[str, int], float] = {}

    p: dict[tuple[str, int], int] = {}
    on: dict[tuple[str, int], int] = {}
    start: dict[tuple[str, int], int] = {}
    stop: dict[tuple[str, int], int] = {}
    shed: dict[int, int] = {}
    spill: dict[int, int] = {}
    reserve_shortfall: dict[int, int] = {}

    for technology in config.technologies:
        for t in range(horizon):
            variable_cost = _variable_cost_for(technology, frame, t)
            capacity = _capacity_for(technology, frame, t)
            variable_costs[(technology.name, t)] = variable_cost
            capacities[(technology.name, t)] = capacity
            p[(technology.name, t)] = builder.add_variable(
                f"p[{technology.name},{t}]",
                objective=variable_cost * 1000.0,
                lower=0.0,
                upper=(
                    capacity
                    if not technology.commitment
                    else technology.installed_capacity_gw
                ),
                integral=False,
            )

            if technology.commitment:
                on[(technology.name, t)] = builder.add_variable(
                    f"on[{technology.name},{t}]",
                    objective=technology.no_load_cost_eur_h,
                    lower=0.0,
                    upper=float(technology.n_units),
                    integral=True,
                )
                start[(technology.name, t)] = builder.add_variable(
                    f"start[{technology.name},{t}]",
                    objective=technology.startup_cost_eur,
                    lower=0.0,
                    upper=float(technology.n_units),
                    integral=True,
                )
                stop[(technology.name, t)] = builder.add_variable(
                    f"stop[{technology.name},{t}]",
                    objective=0.0,
                    lower=0.0,
                    upper=float(technology.n_units),
                    integral=True,
                )

    for t in range(horizon):
        shed[t] = builder.add_variable(
            f"shed[{t}]",
            objective=config.voll_eur_mwh * 1000.0,
            lower=0.0,
            upper=np.inf,
        )
        spill[t] = builder.add_variable(
            f"spill[{t}]",
            objective=config.spill_cost_eur_mwh * 1000.0,
            lower=0.0,
            upper=np.inf,
        )
        reserve_shortfall[t] = builder.add_variable(
            f"reserve_shortfall[{t}]",
            objective=config.reserve_shortfall_cost_eur_mwh * 1000.0,
            lower=0.0,
            upper=np.inf,
        )

    for t in range(horizon):
        balance: dict[int, float] = {
            p[(technology.name, t)]: 1.0
            for technology in config.technologies
        }
        balance[shed[t]] = 1.0
        balance[spill[t]] = -1.0
        builder.add_eq(balance, net_demand[t], f"balance[{t}]")

    for technology in config.technologies:
        name = technology.name
        if technology.commitment:
            initial_on = int(
                round(technology.n_units * config.initial_online_fraction)
            )
            initial_on = min(max(initial_on, 0), technology.n_units)
            initial_output = (
                initial_on
                * technology.unit_size_gw
                * max(technology.min_output_fraction, 0.50)
            )

            for t in range(horizon):
                transition = {
                    on[(name, t)]: 1.0,
                    start[(name, t)]: -1.0,
                    stop[(name, t)]: 1.0,
                }
                if t == 0:
                    builder.add_eq(
                        transition,
                        float(initial_on),
                        f"commitment_transition[{name},{t}]",
                    )
                else:
                    transition[on[(name, t - 1)]] = -1.0
                    builder.add_eq(
                        transition,
                        0.0,
                        f"commitment_transition[{name},{t}]",
                    )

                builder.add_le(
                    {
                        p[(name, t)]: 1.0,
                        on[(name, t)]: -technology.unit_size_gw,
                    },
                    0.0,
                    f"capacity_on[{name},{t}]",
                )
                builder.add_le(
                    {
                        p[(name, t)]: -1.0,
                        on[(name, t)]: (
                            technology.unit_size_gw
                            * technology.min_output_fraction
                        ),
                    },
                    0.0,
                    f"minimum_output[{name},{t}]",
                )

                if t == 0:
                    builder.add_le(
                        {
                            p[(name, t)]: 1.0,
                            start[(name, t)]: -technology.unit_size_gw,
                        },
                        initial_output
                        + technology.ramp_up_gw_h_per_unit * initial_on,
                        f"ramp_up[{name},{t}]",
                    )
                    builder.add_le(
                        {
                            p[(name, t)]: -1.0,
                            stop[(name, t)]: -technology.unit_size_gw,
                        },
                        -initial_output
                        + technology.ramp_down_gw_h_per_unit * initial_on,
                        f"ramp_down[{name},{t}]",
                    )
                else:
                    builder.add_le(
                        {
                            p[(name, t)]: 1.0,
                            p[(name, t - 1)]: -1.0,
                            on[(name, t - 1)]: (
                                -technology.ramp_up_gw_h_per_unit
                            ),
                            start[(name, t)]: -technology.unit_size_gw,
                        },
                        0.0,
                        f"ramp_up[{name},{t}]",
                    )
                    builder.add_le(
                        {
                            p[(name, t - 1)]: 1.0,
                            p[(name, t)]: -1.0,
                            on[(name, t)]: (
                                -technology.ramp_down_gw_h_per_unit
                            ),
                            stop[(name, t)]: -technology.unit_size_gw,
                        },
                        0.0,
                        f"ramp_down[{name},{t}]",
                    )
        else:
            for t in range(1, horizon):
                total_ramp_up = max(
                    technology.ramp_up_gw_h_per_unit,
                    technology.installed_capacity_gw,
                )
                total_ramp_down = max(
                    technology.ramp_down_gw_h_per_unit,
                    technology.installed_capacity_gw,
                )
                builder.add_le(
                    {
                        p[(name, t)]: 1.0,
                        p[(name, t - 1)]: -1.0,
                    },
                    total_ramp_up,
                    f"ramp_up[{name},{t}]",
                )
                builder.add_le(
                    {
                        p[(name, t - 1)]: 1.0,
                        p[(name, t)]: -1.0,
                    },
                    total_ramp_down,
                    f"ramp_down[{name},{t}]",
                )

        if technology.energy_budget_fraction is not None:
            max_energy = sum(
                capacities[(name, t)] for t in range(horizon)
            ) * float(technology.energy_budget_fraction)
            builder.add_le(
                {p[(name, t)]: 1.0 for t in range(horizon)},
                max_energy,
                f"energy_budget[{name}]",
            )

    for t in range(horizon):
        noncommitted_capacity = sum(
            capacities[(technology.name, t)]
            for technology in config.technologies
            if not technology.commitment
        )
        reserve_row: dict[int, float] = {
            reserve_shortfall[t]: -1.0
        }
        for technology in config.technologies:
            if technology.commitment:
                reserve_row[on[(technology.name, t)]] = (
                    -technology.unit_size_gw
                )
        required_committed = (
            net_demand[t]
            + reserve_requirement[t]
            - noncommitted_capacity
        )
        builder.add_le(
            reserve_row,
            -required_committed,
            f"reserve[{t}]",
        )

    return builder.finalize(
        technologies=config.technologies,
        horizon=horizon,
        variable_costs=variable_costs,
        capacities=capacities,
        net_demand=net_demand,
        reserve_requirement=reserve_requirement,
    )


def _milp_constraints(problem: BuiltProblem) -> list[LinearConstraint]:
    constraints: list[LinearConstraint] = []
    if problem.A_eq.shape[0]:
        constraints.append(
            LinearConstraint(problem.A_eq, problem.b_eq, problem.b_eq)
        )
    if problem.A_ub.shape[0]:
        constraints.append(
            LinearConstraint(
                problem.A_ub,
                np.full_like(problem.b_ub, -np.inf),
                problem.b_ub,
            )
        )
    return constraints


def solve_milp(
    problem: BuiltProblem,
    config: StructuralModelConfig,
):
    options: dict[str, float | int | bool] = {
        "disp": False,
        "presolve": config.solver.presolve,
        "time_limit": config.solver.time_limit_seconds,
        "mip_rel_gap": config.solver.mip_rel_gap,
    }
    if config.solver.node_limit is not None:
        options["node_limit"] = config.solver.node_limit

    result = milp(
        c=problem.c,
        integrality=problem.integrality,
        bounds=problem.bounds,
        constraints=_milp_constraints(problem),
        options=options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Échec MILP HiGHS : status={result.status}, message={result.message}"
        )
    return result


def solve_fixed_commitment_lp(
    problem: BuiltProblem,
    mip_solution: np.ndarray,
):
    lower = np.asarray(problem.bounds.lb, dtype=float).copy()
    upper = np.asarray(problem.bounds.ub, dtype=float).copy()
    integer_positions = np.flatnonzero(problem.integrality != 0)
    rounded = np.rint(mip_solution[integer_positions])
    lower[integer_positions] = rounded
    upper[integer_positions] = rounded

    bounds = list(zip(lower.tolist(), upper.tolist(), strict=True))
    result = linprog(
        c=problem.c,
        A_ub=problem.A_ub if problem.A_ub.shape[0] else None,
        b_ub=problem.b_ub if problem.A_ub.shape[0] else None,
        A_eq=problem.A_eq if problem.A_eq.shape[0] else None,
        b_eq=problem.b_eq if problem.A_eq.shape[0] else None,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Échec LP de pricing : status={result.status}, message={result.message}"
        )
    return result


def _value(problem: BuiltProblem, solution: np.ndarray, name: str) -> float:
    return float(solution[problem.variables[name]])


def _dual_by_name(
    names: Iterable[str],
    marginals: np.ndarray,
) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(names, marginals, strict=True)
    }


def extract_results(
    frame: pd.DataFrame,
    problem: BuiltProblem,
    mip_result,
    lp_result,
) -> StructuralDayResult:
    mip_x = np.asarray(mip_result.x, dtype=float)
    lp_x = np.asarray(lp_result.x, dtype=float)
    eq_duals = _dual_by_name(
        problem.eq_names,
        np.asarray(lp_result.eqlin.marginals, dtype=float),
    )
    ub_duals = _dual_by_name(
        problem.ub_names,
        np.asarray(lp_result.ineqlin.marginals, dtype=float),
    )

    technology_codes = {
        technology.name: float(position + 1)
        for position, technology in enumerate(problem.technologies)
    }
    records: list[dict[str, float]] = []
    dispatch_records: list[dict[str, float | str | pd.Timestamp]] = []

    for t, timestamp in enumerate(frame.index):
        production = {
            technology.name: _value(
                problem,
                lp_x,
                f"p[{technology.name},{t}]",
            )
            for technology in problem.technologies
        }
        producing = [
            technology
            for technology in problem.technologies
            if production[technology.name] > 1e-5
        ]
        marginal_technology = (
            max(
                producing,
                key=lambda technology: problem.variable_costs[
                    (technology.name, t)
                ],
            ).name
            if producing
            else "none"
        )

        committed_capacity = 0.0
        online_units = 0.0
        startups = 0.0
        startup_cost = 0.0
        available_capacity = 0.0

        for technology in problem.technologies:
            if technology.commitment:
                on_value = _value(
                    problem,
                    mip_x,
                    f"on[{technology.name},{t}]",
                )
                start_value = _value(
                    problem,
                    mip_x,
                    f"start[{technology.name},{t}]",
                )
                committed_capacity += technology.unit_size_gw * on_value
                online_units += on_value
                startups += start_value
                startup_cost += technology.startup_cost_eur * start_value
                available_capacity += technology.unit_size_gw * on_value
            else:
                available_capacity += problem.capacities[(technology.name, t)]

            dispatch_records.append(
                {
                    "timestamp": timestamp,
                    "technology": technology.name,
                    "dispatch_gw": production[technology.name],
                    "variable_cost_eur_mwh": problem.variable_costs[
                        (technology.name, t)
                    ],
                    "available_capacity_gw": problem.capacities[
                        (technology.name, t)
                    ],
                }
            )

        shed = _value(problem, lp_x, f"shed[{t}]")
        spill = _value(problem, lp_x, f"spill[{t}]")
        reserve_shortfall = _value(
            problem,
            lp_x,
            f"reserve_shortfall[{t}]",
        )
        reserve_margin = (
            available_capacity
            - problem.net_demand[t]
            - reserve_shortfall
        )
        balance_dual = eq_duals[f"balance[{t}]"] / 1000.0
        ramp_shadow = max(
            [
                abs(value) / 1000.0
                for name, value in ub_duals.items()
                if name.endswith(f",{t}]") and name.startswith("ramp_")
            ]
            or [0.0]
        )
        reserve_shadow = abs(ub_duals.get(f"reserve[{t}]", 0.0)) / 1000.0

        records.append(
            {
                "milp_structural_price": balance_dual,
                "milp_marginal_technology_code": technology_codes.get(
                    marginal_technology,
                    0.0,
                ),
                "milp_reserve_margin_gw": reserve_margin,
                "milp_committed_thermal_gw": committed_capacity,
                "milp_online_units": online_units,
                "milp_startups": startups,
                "milp_startup_cost_eur": startup_cost,
                "milp_scarcity_gw": shed,
                "milp_ramp_shadow_eur_mwh": ramp_shadow,
                "milp_reserve_shadow_eur_mwh": reserve_shadow,
                "milp_spill_gw": spill,
                "milp_reserve_shortfall_gw": reserve_shortfall,
            }
        )

    features = pd.DataFrame(records, index=frame.index, dtype=np.float64)
    dispatch = pd.DataFrame(dispatch_records)
    diagnostics = {
        "success": True,
        "mip_status": int(mip_result.status),
        "mip_message": str(mip_result.message),
        "mip_objective_eur": float(mip_result.fun),
        "lp_objective_eur": float(lp_result.fun),
        "mip_node_count": int(getattr(mip_result, "mip_node_count", 0) or 0),
        "mip_gap": float(getattr(mip_result, "mip_gap", np.nan)),
        "mip_dual_bound": float(
            getattr(mip_result, "mip_dual_bound", np.nan)
        ),
        "hours": int(problem.horizon),
    }
    return StructuralDayResult(
        features=features,
        dispatch=dispatch,
        diagnostics=diagnostics,
    )


def solve_structural_day(
    frame: pd.DataFrame,
    config: StructuralModelConfig,
) -> StructuralDayResult:
    problem = build_problem(frame, config)
    mip_result = solve_milp(problem, config)
    lp_result = solve_fixed_commitment_lp(problem, mip_result.x)
    return extract_results(frame, problem, mip_result, lp_result)
