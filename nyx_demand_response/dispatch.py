"""Explicit one-period welfare LP with qualified inputs only.

Generation plus voluntary demand reduction minus net exports equals gross
demand. There is no involuntary-shortage variable and no artificial VOLL bid.
This is a conditional dispatch calculation, not an identified market model:
block orders, commitment, ramping, storage and unknown demand curves are not
inferred. A caller cannot qualify raw initial RAM by merely assuming zero NP.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import math

import numpy as np
import pandas as pd
from scipy.optimize import linprog


class DispatchError(RuntimeError):
    """No reliable dispatch solution is available."""


class DispatchQualificationError(ValueError):
    """The physical-input or reference-frame contract is not satisfied."""


class DispatchInfeasibleError(DispatchError):
    """Supply and explicitly offered flexibility cannot satisfy the domain."""


EPSILON_MW = .001
FEASIBILITY_TOLERANCE_MW = 1e-6


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise DispatchQualificationError(f"{name}: a finite number, not a boolean, is required.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DispatchQualificationError(f"{name}: finite number required.") from exc
    if not math.isfinite(result) or positive and result <= 0:
        raise DispatchQualificationError(f"{name}: finite {'positive ' if positive else ''}number required.")
    return result


def _frame(value, names, numeric, keys, *, empty=False):
    if (not isinstance(value, pd.DataFrame) or value.columns.has_duplicates
            or set(value.columns) != set(names) or not empty and value.empty):
        raise DispatchQualificationError(f"Expected normalized columns {names}; nonempty table required unless permitted.")
    frame = value.copy(deep=True).reset_index(drop=True)
    for name in set(names).difference(numeric):
        if not frame[name].map(lambda v: isinstance(v, str) and bool(v) and v == v.strip()).all():
            raise DispatchQualificationError(f"{name}: explicit, nonempty, whitespace-free identifiers required.")
    if frame.duplicated(keys).any():
        raise DispatchQualificationError(f"Duplicate identities {keys}.")
    for name in numeric:
        frame[name] = [_number(v, name) for v in frame[name]]
    return frame


def _inputs(supply, demand, flexibility, qualification, network):
    if (not isinstance(qualification, Mapping)
            or qualification.get("physical_inputs_qualified") is not True
            or qualification.get("demand_basis") != "before_price_response"
            or qualification.get("flexibility_basis") != "additional_voluntary_reduction"
            or qualification.get("evidence_kind") not in {"synthetic_assumption", "audited_inputs"}):
        raise DispatchQualificationError("Qualified gross demand, additional voluntary flexibility and explicit evidence kind are mandatory.")
    if not isinstance(qualification.get("evidence_description"), str) or not qualification["evidence_description"].strip():
        raise DispatchQualificationError("Nonempty evidence_description is required; assumptions are not observed market curves.")
    duration = _number(qualification.get("duration_hours", 1.), "duration_hours", positive=True)
    s = _frame(supply, ["zone", "segment", "capacity_mw", "bid_eur_mwh"],
               ["capacity_mw", "bid_eur_mwh"], ["zone", "segment"], empty=True)
    d = _frame(demand, ["zone", "demand_mw"], ["demand_mw"], ["zone"])
    f = _frame(flexibility, ["zone", "segment", "capacity_mw", "reservation_eur_mwh"],
               ["capacity_mw", "reservation_eur_mwh"], ["zone", "segment"], empty=True)
    zones = d.zone.tolist()
    if (not set(s.zone).issubset(zones) or not set(f.zone).issubset(zones)
            or (s.capacity_mw < 0).any() or (f.capacity_mw < 0).any()
            or (d.demand_mw < 0).any() or (f.reservation_eur_mwh < 0).any()
            or len(s)+len(f) == 0):
        raise DispatchQualificationError("Known zones, nonnegative capacities/demand/reservations and at least one offered segment are required.")
    qualified = deepcopy(dict(qualification))
    if network is None:
        return s, d, f, None, np.zeros(len(zones)), duration, qualified
    contract = qualification.get("network")
    if (not isinstance(contract, Mapping) or contract.get("qualified") is not True
            or contract.get("basis") != "reference_net_positions"
            or not isinstance(contract.get("balanced_zones"), (tuple, list))
            or len(contract["balanced_zones"]) != len(set(contract["balanced_zones"]))
            or set(contract["balanced_zones"]) != set(zones)
            or not isinstance(contract.get("reference_evidence"), str)
            or not contract["reference_evidence"].strip()
            or not isinstance(contract.get("reference_net_positions_mw"), Mapping)
            or set(contract["reference_net_positions_mw"]) != set(zones)):
        raise DispatchQualificationError("Network requires a qualified explicit RAM/NP reference and the complete balanced zone domain.")
    ref = np.array([_number(contract["reference_net_positions_mw"][z], "reference net position") for z in zones])
    if abs(ref.sum()) > FEASIBILITY_TOLERANCE_MW:
        raise DispatchQualificationError("Reference net positions must balance to zero across all modeled zones.")
    names = ["constraint_id", "ram_mw", *["ptdf_"+z for z in zones]]
    n = _frame(network, names, names[1:], ["constraint_id"])
    return s, d, f, n, ref, duration, qualified


def validate_period_inputs(supply, demand, flexibility, *, qualification, network=None):
    """Validate normalized physical tables and qualifications, without solving."""
    _inputs(supply, demand, flexibility, qualification, network)


def _problem(s, d, f, network, reference, duration):
    zones = d.zone.tolist()
    ns, nf, nz = len(s), len(f), len(d)
    exports = network is not None
    size = ns+nf+(nz if exports else 0)
    objective = np.r_[s.bid_eur_mwh.to_numpy(float), f.reservation_eur_mwh.to_numpy(float),
                       np.zeros(nz if exports else 0)]*duration
    equality = np.zeros((nz+int(exports), size))
    inequality = np.zeros((nz+(len(network) if exports else 0), size))
    rhs = d.demand_mw.to_numpy(float)
    for i, zone in enumerate(zones):
        equality[i, :ns] = s.zone.eq(zone)
        equality[i, ns:ns+nf] = f.zone.eq(zone)
        inequality[i, ns:ns+nf] = f.zone.eq(zone)
        if exports:
            equality[i, ns+nf+i] = -1.
    if exports:
        equality[-1, ns+nf:] = 1.
        ptdf = network[["ptdf_"+z for z in zones]].to_numpy(float)
        inequality[nz:, ns+nf:] = ptdf
        rhs_ub = np.r_[rhs, network.ram_mw.to_numpy(float)+ptdf@reference]
    else:
        rhs_ub = rhs.copy()
    bounds = [(0., c) for c in np.r_[s.capacity_mw.to_numpy(float), f.capacity_mw.to_numpy(float)]]
    bounds += [(None, None)]*(nz if exports else 0)
    return dict(c=objective, A_ub=inequality, b_ub=rhs_ub, A_eq=equality,
                b_eq=np.r_[rhs, 0.] if exports else rhs.copy(), bounds=bounds)


def _solve(problem):
    result = linprog(**problem, method="highs", options={"primal_feasibility_tolerance": 1e-8,
                                                       "dual_feasibility_tolerance": 1e-8})
    if result.status == 2:
        raise DispatchInfeasibleError("Qualified supply, voluntary flexibility and network cannot satisfy demand; no artificial shortage bid was inserted.")
    if not result.success or result.x is None or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise DispatchError(f"Dispatch optimizer failed ({result.status}): {result.message}")
    return result


def solve_period(supply, demand, flexibility, *, qualification, network=None):
    """Return nodal duals, voluntary reduction, dispatch and local price brackets.

    `prices_eur_mwh` are balance duals. `demand_marginal_cost_eur_mwh`
    additionally includes the dual of reduction <= gross demand: this matters
    at full curtailment. Perturbation brackets are total-objective derivatives,
    not fitted prices or a shortage penalty. A missing bracket side is an
    infeasible/boundary perturbation, not an infinite or invented price.
    """
    s, d, f, net, reference, duration, evidence = _inputs(supply, demand, flexibility, qualification, network)
    problem = _problem(s, d, f, net, reference, duration)
    result = _solve(problem)
    zones, ns, nf, nz = d.zone.tolist(), len(s), len(f), len(d)
    generation, reduction = result.x[:ns], result.x[ns:ns+nf]
    exports = result.x[ns+nf:] if net is not None else np.zeros(nz)
    produced = np.array([generation[s.zone.eq(z)].sum() for z in zones])
    reduced = np.array([reduction[f.zone.eq(z)].sum() for z in zones])
    demand_mw = d.demand_mw.to_numpy(float)
    balance_residual = produced+reduced-exports-demand_mw
    if (np.max(np.abs(balance_residual)) > FEASIBILITY_TOLERANCE_MW
            or np.any(reduced < -FEASIBILITY_TOLERANCE_MW)
            or np.any(reduced-demand_mw > FEASIBILITY_TOLERANCE_MW)
            or np.min(result.ineqlin.residual) < -FEASIBILITY_TOLERANCE_MW):
        raise DispatchError("Dispatch failed post-solve physical feasibility checks.")
    price = result.eqlin.marginals[:nz]/duration
    demand_marginal = price+result.ineqlin.marginals[:nz]/duration
    brackets = {}
    for i, zone in enumerate(zones):
        sides = {}
        for sign, name in ((-1, "left"), (1, "right")):
            if demand_mw[i]+sign*EPSILON_MW < 0:
                sides[name] = dict(price_eur_mwh=None, status="nonnegative_demand_boundary")
                continue
            perturbed = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in problem.items()}
            perturbed["b_eq"][i] += sign*EPSILON_MW
            perturbed["b_ub"][i] += sign*EPSILON_MW
            try:
                changed = _solve(perturbed)
            except DispatchInfeasibleError:
                sides[name] = dict(price_eur_mwh=None, status="infeasible")
            else:
                marginal = (changed.fun-result.fun)/(sign*EPSILON_MW*duration)
                sides[name] = dict(price_eur_mwh=float(marginal), status="optimal")
        left, right = sides["left"]["price_eur_mwh"], sides["right"]["price_eur_mwh"]
        ambiguous = left is None or right is None or not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-4)
        brackets[zone] = {**sides, "ambiguous": bool(ambiguous), "epsilon_mw": EPSILON_MW}
    supply_rows = [{"zone": row.zone, "segment": row.segment, "generation_mw": float(generation[i]),
        "capacity_mw": float(row.capacity_mw), "unused_capacity_mw": float(row.capacity_mw-generation[i]),
        "bid_eur_mwh": float(row.bid_eur_mwh)} for i, row in enumerate(s.itertuples(index=False))]
    flexibility_rows = [{"zone": row.zone, "segment": row.segment, "reduction_mw": float(reduction[i]),
        "capacity_mw": float(row.capacity_mw), "unused_capacity_mw": float(row.capacity_mw-reduction[i]),
        "reservation_eur_mwh": float(row.reservation_eur_mwh)} for i, row in enumerate(f.itertuples(index=False))]
    network_rows = []
    if net is not None:
        change = net[["ptdf_"+z for z in zones]].to_numpy(float)@(exports-reference)
        network_rows = [{"constraint_id": row.constraint_id, "flow_change_from_reference_mw": float(change[i]),
            "ram_mw": float(row.ram_mw), "margin_mw": float(row.ram_mw-change[i]),
            "shadow_price_eur_mwh": float(-result.ineqlin.marginals[nz+i]/duration),
            "binding": bool(abs(row.ram_mw-change[i]) <= FEASIBILITY_TOLERANCE_MW)}
            for i, row in enumerate(net.itertuples(index=False))]
    return dict(status="optimal", synthetic_assumption=evidence["evidence_kind"] == "synthetic_assumption",
        qualification=evidence, duration_hours=duration, objective_eur=float(result.fun),
        prices_eur_mwh=dict(zip(zones, map(float, price))),
        demand_marginal_cost_eur_mwh=dict(zip(zones, map(float, demand_marginal))),
        price_brackets=brackets, generation=supply_rows, flexibility=flexibility_rows,
        generation_mw=dict(zip(zones, map(float, produced))),
        gross_demand_mw=dict(zip(zones, map(float, demand_mw))),
        served_demand_mw=dict(zip(zones, map(float, demand_mw-reduced))),
        voluntary_demand_reduction_mw=dict(zip(zones, map(float, reduced))),
        involuntary_shortage_mw={zone: 0. for zone in zones},
        net_exports_mw=dict(zip(zones, map(float, exports))), network=network_rows,
        balance_residual_mw=dict(zip(zones, map(float, balance_residual))),
        diagnostics=dict(network_mode="qualified_balanced_ptdf" if net is not None else "isolated_zones",
            price_duals_can_be_nonunique=any(v["ambiguous"] for v in brackets.values()),
            demand_curve_identified_from_prices=False, involuntary_shortage_variable_present=False,
            unavailable_curves_are_not_estimated=True, optimizer_message=str(result.message),
            limitations=["Conditional single-period LP; not coupled unit commitment or the SDAC clearing algorithm.",
                         "Supply and voluntary-demand bids must be supplied independently; current prices do not identify them.",
                         "Synthetic assumptions are demonstrations, not evidence of real flexible volumes or market prices."]))
