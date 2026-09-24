"""Independent physical price scenarios, with optional full-zone PTDF coupling.

This is a convex economic-dispatch approximation, not an EUPHEMIA replica or
an identifier of an observed marginal generating unit. Inputs are instantaneous
MW; the objective uses a one-hour normalization so its balance duals are in
EUR/MWh. The caller must prove point-in-time provenance and preserve civil/DST
delivery support. No Chronos forecast, electric-price lag or Storm input exists.

``must_run_mw`` denotes available low-bid supply that *can* be curtailed in
this approximation; it does not impose a physical minimum stable generation.
Negative bids and the shortage penalty must be explicitly configured.

With ``demand_basis='residual'``, the already-netted renewable forecast may
be negative. Demand and surplus are split without altering RL minus nuclear;
outputs then describe a residual-stack proxy. Full renewable curtailment and
imports during deeper renewable curtailment cannot be reconstructed from RL.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


KEYS = ["delivery_start_utc", "zone"]
REQUIRED_FEATURES = (
    "demand_mw", "must_run_mw", "ccgt_available_mw", "ocgt_available_mw",
    "ttf_eur_mwh_th", "eua_eur_tco2",
)
COAL_FEATURES = ("coal_available_mw", "coal_eur_mwh_th")
_DEFAULTS = {
    "ccgt_efficiency": 0.58, "ocgt_efficiency": 0.39, "coal_efficiency": 0.40,
    "gas_emission_tco2_mwh_th": 0.202, "coal_emission_tco2_mwh_th": 0.341,
    "ccgt_vom_eur_mwh": 3.0, "ocgt_vom_eur_mwh": 5.0, "coal_vom_eur_mwh": 4.0,
    "thermal_availability_scale": 1.0, "thermal_bid_premium_eur_mwh": 0.0,
}
_BOUNDS = {
    "ccgt_efficiency": (0.35, 0.70), "ocgt_efficiency": (0.20, 0.50),
    "coal_efficiency": (0.20, 0.55), "gas_emission_tco2_mwh_th": (0.10, 0.40),
    "coal_emission_tco2_mwh_th": (0.20, 0.60), "ccgt_vom_eur_mwh": (0.0, 100.0),
    "ocgt_vom_eur_mwh": (0.0, 100.0), "coal_vom_eur_mwh": (0.0, 100.0),
    "thermal_availability_scale": (0.5, 1.2),
    "thermal_bid_premium_eur_mwh": (-50.0, 50.0),
    "must_run_bid_eur_mwh": (-1000.0, 200.0),
    "scarcity_price_eur_mwh": (500.0, 20000.0),
}
_SEGMENTS = ("must_run", "ccgt", "ocgt", "coal", "shortage")


class MarginalCostModelError(ValueError):
    """Invalid physical, scenario, or full-zone network contract."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise MarginalCostModelError(f"{label}: a finite number is required.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MarginalCostModelError(f"{label}: a finite number is required.") from error
    if not math.isfinite(result):
        raise MarginalCostModelError(f"{label}: a finite number is required.")
    return result


def _timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    if "delivery_start_utc" not in frame:
        if isinstance(frame.index, pd.MultiIndex) and list(frame.index.names) == KEYS:
            frame = frame.reset_index()
        else:
            raise MarginalCostModelError("delivery_start_utc is required.")
    result = frame.copy(deep=True)
    values = result["delivery_start_utc"]
    if any(pd.isna(value) or pd.Timestamp(value).tzinfo is None for value in values):
        raise MarginalCostModelError("Delivery timestamps must be explicitly timezone-aware.")
    result["delivery_start_utc"] = pd.to_datetime(values, utc=True, errors="raise")
    return result


def _features(frame: pd.DataFrame, *, demand_basis: str = "gross") -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise MarginalCostModelError("A non-empty physical feature table is required.")
    result = _timestamps(frame)
    required = set(KEYS) | set(REQUIRED_FEATURES)
    missing = required.difference(result)
    if missing:
        raise MarginalCostModelError(f"Missing physical features: {sorted(missing)}")
    allowed = required | set(COAL_FEATURES) | {"forecast_origin_utc", "provenance_id"}
    extra = set(result).difference(allowed)
    if extra:
        raise MarginalCostModelError(f"Non-physical/unrecognized feature columns: {sorted(extra)}")
    result["zone"] = result["zone"].astype(str)
    if not result["zone"].str.fullmatch(r"[A-Z][A-Z0-9_-]{1,15}").all():
        raise MarginalCostModelError("Explicit uppercase zone identifiers are required.")
    if result.duplicated(KEYS).any():
        raise MarginalCostModelError("Duplicate physical delivery/zone rows.")
    present_coal = set(COAL_FEATURES).intersection(result)
    if present_coal and present_coal != set(COAL_FEATURES):
        raise MarginalCostModelError("Coal availability and coal fuel cost must be provided together.")
    for column in (*REQUIRED_FEATURES, *COAL_FEATURES):
        if column not in result:
            result[column] = 0.0
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        signed_residual = column == "demand_mw" and demand_basis == "residual"
        if not np.isfinite(values).all() or ((values < 0).any() and not signed_residual):
            raise MarginalCostModelError(f"{column}: nonnegative finite values in declared units are required.")
        result[column] = values
    result["input_demand_mw"] = result["demand_mw"]
    result["input_must_run_mw"] = result["must_run_mw"]
    result["demand_basis"] = demand_basis
    result["residual_surplus_mw"] = 0.0
    if demand_basis == "residual":
        # Preserve RL - nuclear exactly without mixing wind/solar vintages or
        # subtracting hydro twice. These are NET residual-stack quantities,
        # not a reconstruction of total generation or physically dispatched
        # renewable units. The common explicit low bid also applies to excess
        # renewables. Curtailment beyond this net surplus is not modeled.
        result["residual_surplus_mw"] = (-result["demand_mw"]).clip(lower=0.0)
        result["must_run_mw"] += result["residual_surplus_mw"]
        result["demand_mw"] = result["demand_mw"].clip(lower=0.0)
    return result.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _labels(y: pd.Series | pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if isinstance(y, pd.Series):
        if isinstance(y.index, pd.MultiIndex) and list(y.index.names) == KEYS:
            y = y.rename("actual").reset_index()
        elif isinstance(y.index, pd.DatetimeIndex) and predictions.zone.nunique() == 1:
            y = pd.DataFrame({"delivery_start_utc": y.index, "zone": predictions.zone.iloc[0],
                              "actual": y.to_numpy()})
        else:
            raise MarginalCostModelError("Labels require delivery/zone keys, not positional alignment.")
    if not isinstance(y, pd.DataFrame) or not set([*KEYS, "actual"]).issubset(y):
        raise MarginalCostModelError("Labels require delivery_start_utc, zone, actual.")
    result = _timestamps(y.loc[:, [*KEYS, "actual"]])
    result["zone"] = result["zone"].astype(str)
    result["actual"] = pd.to_numeric(result["actual"], errors="coerce")
    if result.empty or result.duplicated(KEYS).any() or not np.isfinite(result.actual).all():
        raise MarginalCostModelError("Labels must have unique keys and finite values.")
    return result


def candidate_scores(
    predictions: pd.DataFrame, y: pd.Series | pd.DataFrame, *, metric: str = "mae",
) -> pd.DataFrame:
    """Score precomputed scenarios only on explicit training-label keys.

    The caller is responsible for supplying the rolling training split; labels
    never become inference features. No scenario parameters are optimized here.
    """
    if metric not in {"mae", "rmse"}:
        raise MarginalCostModelError("Selection metric must be mae or rmse.")
    required = {*KEYS, "candidate_id", "price_eur_mwh"}
    if not required.issubset(predictions):
        raise MarginalCostModelError("Candidate predictions are missing identity/price columns.")
    frame = _timestamps(predictions)
    if frame.empty or frame.duplicated([*KEYS, "candidate_id"]).any():
        raise MarginalCostModelError("Candidate predictions must be nonempty and uniquely keyed.")
    labels = _labels(y, frame)
    records = []
    for candidate, block in frame.groupby("candidate_id", sort=False):
        paired = labels.merge(block.loc[:, [*KEYS, "price_eur_mwh"]], on=KEYS, how="left", validate="one_to_one")
        prices = pd.to_numeric(paired.price_eur_mwh, errors="coerce").to_numpy(float)
        if not np.isfinite(prices).all():
            raise MarginalCostModelError(f"Candidate {candidate} does not cover all training-label keys.")
        errors = prices - paired.actual.to_numpy(float)
        records.append({"candidate_id": str(candidate), "mae": float(np.abs(errors).mean()),
                        "rmse": float(np.sqrt(np.square(errors).mean())), "n_rows": len(paired)})
    return pd.DataFrame(records).sort_values(metric, kind="stable").reset_index(drop=True)


def select_candidate(
    predictions: pd.DataFrame, y: pd.Series | pd.DataFrame, *, metric: str = "mae",
) -> str:
    return str(candidate_scores(predictions, y, metric=metric).iloc[0].candidate_id)


class MarginalCostExpert:
    """Small fixed scenario bank with optional rolling selection by the caller."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = deepcopy(dict(config))
        self.demand_basis = self.config.get("demand_basis", "gross")
        if self.demand_basis not in {"gross", "residual"}:
            raise MarginalCostModelError("demand_basis must be explicitly gross or residual.")
        parameters = dict(self.config.get("parameters", {}))
        unknown = set(parameters).difference(_BOUNDS)
        if unknown:
            raise MarginalCostModelError(f"Unknown physical parameters: {sorted(unknown)}")
        raw_scenarios = self.config.get("scenarios")
        if not isinstance(raw_scenarios, Sequence) or isinstance(raw_scenarios, (str, bytes)) or not 1 <= len(raw_scenarios) <= 32:
            raise MarginalCostModelError("Provide an explicit bank of 1 to 32 physical scenarios.")
        self.scenarios: list[dict[str, Any]] = []
        for raw in raw_scenarios:
            if not isinstance(raw, Mapping):
                raise MarginalCostModelError("Each scenario must be a mapping.")
            extra = set(raw).difference({"id", "hypothesis", *_BOUNDS})
            if extra:
                raise MarginalCostModelError(f"Unknown scenario parameters: {sorted(extra)}")
            name, hypothesis = str(raw.get("id", "")), str(raw.get("hypothesis", "")).strip()
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or not hypothesis:
                raise MarginalCostModelError("Scenario id and explicit physical hypothesis are required.")
            values = {**_DEFAULTS, **parameters, **{key: value for key, value in raw.items() if key in _BOUNDS}}
            for key, (low, high) in _BOUNDS.items():
                if key not in values:
                    raise MarginalCostModelError(f"{key} must be explicitly configured; no invented negative or scarcity bid.")
                values[key] = _finite(values[key], key)
                if not low <= values[key] <= high:
                    raise MarginalCostModelError(f"{key} must lie in [{low}, {high}].")
            values.update(id=name, hypothesis=hypothesis)
            self.scenarios.append(values)
        if len({item["id"] for item in self.scenarios}) != len(self.scenarios):
            raise MarginalCostModelError("Scenario identifiers must be unique.")
        self.network_config = dict(self.config.get("network", {}))
        self.selected_candidate_id_: str | None = None
        self.selection_audit_: dict[str, Any] | None = None

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item["id"] for item in self.scenarios)

    def fit(
        self, features: pd.DataFrame, y: pd.Series | pd.DataFrame, *,
        network: pd.DataFrame | None = None, candidate_predictions: pd.DataFrame | None = None,
        metric: str = "mae",
    ) -> "MarginalCostExpert":
        physical = _features(features, demand_basis=self.demand_basis)
        predictions = (self.predict_candidates(features, network=network)
                       if candidate_predictions is None else candidate_predictions)
        if set(predictions.candidate_id) != set(self.candidate_ids):
            raise MarginalCostModelError("Cached candidate bank differs from the configured scenarios.")
        labels = _labels(y, predictions)
        covered = labels.merge(physical.loc[:, KEYS], on=KEYS, how="left", indicator=True, validate="one_to_one")
        if not covered["_merge"].eq("both").all():
            raise MarginalCostModelError("Training labels are outside the supplied physical feature support.")
        scores = candidate_scores(predictions, y, metric=metric)
        self.selected_candidate_id_ = str(scores.iloc[0].candidate_id)
        self.selection_audit_ = {"metric": metric, "scores": scores.to_dict("records"),
                                 "labels_used_as_features": False, "continuous_parameter_fit": False}
        return self

    def predict(self, features: pd.DataFrame, *, network: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.selected_candidate_id_ is None:
            raise MarginalCostModelError("Select a scenario with fit before calling predict.")
        frame = self.predict_candidates(features, network=network, candidate_ids=[self.selected_candidate_id_])
        return frame.reset_index(drop=True)

    def predict_candidates(
        self, features: pd.DataFrame, network: pd.DataFrame | None = None, *,
        candidate_ids: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        frame = _features(features, demand_basis=self.demand_basis)
        selected = tuple(self.candidate_ids if candidate_ids is None else candidate_ids)
        if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(self.candidate_ids):
            raise MarginalCostModelError("Unknown or duplicate candidate selection.")
        network_groups = self._network_groups(network, frame) if network is not None else None
        if network_groups is None and self.network_config.get("boundary_net_positions_mw"):
            raise MarginalCostModelError("Explicit boundary positions require a full PTDF network.")
        records = []
        for timestamp, period in frame.groupby("delivery_start_utc", sort=True):
            for scenario in self.scenarios:
                if scenario["id"] not in selected:
                    continue
                if network_groups is None:
                    for _, row in period.iterrows():
                        records.append(self._zonal(row, scenario))
                else:
                    records.extend(self._coupled(period, network_groups[timestamp], scenario))
        return pd.DataFrame(records).sort_values([*KEYS, "candidate_id"], kind="stable").reset_index(drop=True)

    def _network_groups(self, network: pd.DataFrame, features: pd.DataFrame) -> dict:
        if not isinstance(network, pd.DataFrame) or network.empty:
            raise MarginalCostModelError("An explicitly supplied PTDF network cannot be empty.")
        zones = self.network_config.get("zones")
        if not isinstance(zones, list) or not zones or len(zones) != len(set(zones)):
            raise MarginalCostModelError("network.zones must declare every PTDF bidding zone.")
        boundary = dict(self.network_config.get("boundary_net_positions_mw", {}))
        if boundary and not str(self.network_config.get("boundary_hypothesis", "")).strip():
            raise MarginalCostModelError("Fixed boundary positions require an explicit boundary_hypothesis.")
        for zone, position in boundary.items():
            if zone not in zones:
                raise MarginalCostModelError("Boundary zones must belong to the full PTDF universe.")
            _finite(position, f"boundary {zone}")
        modeled = set(features.zone.unique())
        if modeled.intersection(boundary) or modeled | set(boundary) != set(zones):
            raise MarginalCostModelError("Full PTDF coverage requires physical features or explicit boundaries for every zone.")
        frame = _timestamps(network)
        required = {"delivery_start_utc", "cnec_id", "ram_mw", *[f"ptdf_{zone}" for zone in zones]}
        if not required.issubset(frame) or {c for c in frame if c.startswith("ptdf_")} != {f"ptdf_{z}" for z in zones}:
            raise MarginalCostModelError("Missing/extra PTDF columns; never discard unmodeled bidding zones.")
        if frame.duplicated(["delivery_start_utc", "cnec_id"]).any() or frame.cnec_id.isna().any():
            raise MarginalCostModelError("CNEC identifiers must be unique within each period.")
        numerical = ["ram_mw", *[f"ptdf_{z}" for z in zones]]
        frame[numerical] = frame[numerical].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(frame[numerical].to_numpy(float)).all():
            raise MarginalCostModelError("RAM/PTDF inputs must be finite; RAM is not an NTC.")
        groups = dict(tuple(frame.groupby("delivery_start_utc", sort=True)))
        if set(groups) != set(features.delivery_start_utc.unique()):
            raise MarginalCostModelError("PTDF periods must exactly cover physical feature periods.")
        for _, group in features.groupby("delivery_start_utc"):
            if set(group.zone) != modeled:
                raise MarginalCostModelError("Every modeled zone is required for every coupled period.")
        return groups

    @staticmethod
    def _stack(row: pd.Series, scenario: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        s = scenario
        gas = float(row.ttf_eur_mwh_th) + float(row.eua_eur_tco2) * s["gas_emission_tco2_mwh_th"]
        coal = float(row.coal_eur_mwh_th) + float(row.eua_eur_tco2) * s["coal_emission_tco2_mwh_th"]
        premium = s["thermal_bid_premium_eur_mwh"]
        costs = np.array([s["must_run_bid_eur_mwh"],
            gas / s["ccgt_efficiency"] + s["ccgt_vom_eur_mwh"] + premium,
            gas / s["ocgt_efficiency"] + s["ocgt_vom_eur_mwh"] + premium,
            coal / s["coal_efficiency"] + s["coal_vom_eur_mwh"] + premium,
            s["scarcity_price_eur_mwh"]], dtype=float)
        capacities = np.array([row.must_run_mw, row.ccgt_available_mw, row.ocgt_available_mw,
                               row.coal_available_mw, row.demand_mw], dtype=float)
        capacities[1:4] *= s["thermal_availability_scale"]
        if (costs[:4][capacities[:4] > 0] >= costs[4]).any():
            raise MarginalCostModelError("Scarcity penalty must exceed all available generation bids.")
        return costs, capacities

    def _record(self, row, scenario, costs, capacities, dispatch, price, net_position, *,
                mode, reference_price, min_ram_slack=math.nan, binding_constraints=0):
        generation = float(dispatch[:4].sum())
        residual = generation + float(dispatch[4]) - float(row.demand_mw) - net_position
        if abs(residual) > 1e-5 * max(1.0, float(row.demand_mw)):
            raise MarginalCostModelError("Physical energy balance failed after dispatch.")
        partial = [i for i in range(4) if 1e-6 < dispatch[i] < capacities[i] - 1e-6
                   and np.isclose(price, costs[i], atol=1e-5, rtol=0)]
        surplus = capacities[0] - dispatch[0] > 1e-6 and np.isclose(price, costs[0], atol=1e-5, rtol=0)
        regime = ("scarcity" if dispatch[4] > 1e-6 else
                  "surplus" if surplus else
                  _SEGMENTS[partial[0]] if partial else
                  "coupled_or_capacity_boundary" if mode == "ptdf_coupled" else "capacity_boundary")
        result = {"delivery_start_utc": row.delivery_start_utc, "zone": row.zone,
                  "candidate_id": scenario["id"], "price_eur_mwh": float(price),
                  "network_mode": mode, "physical_hypothesis": scenario["hypothesis"],
                  "demand_basis": row.demand_basis,
                  "physical_scope": "residual_stack_proxy" if row.demand_basis == "residual" else "gross_stack_approximation",
                  "input_demand_mw": float(row.input_demand_mw),
                  "input_must_run_mw": float(row.input_must_run_mw),
                  "residual_surplus_mw": float(row.residual_surplus_mw),
                  "full_renewable_curtailment_modeled": False,
                  "marginal_regime_proxy": regime, "net_position_mw": float(net_position),
                  "demand_mw": float(row.demand_mw), "generation_mw": generation,
                  "shortage_mw": float(dispatch[4]),
                  "curtailment_mw": max(0.0, float(capacities[0] - dispatch[0])),
                  "thermal_margin_mw": max(0.0, float(capacities[1:4].sum() - dispatch[1:4].sum())),
                  "capacity_margin_mw": float(capacities[:4].sum() - row.demand_mw - net_position),
                  "reference_price_eur_mwh": float(reference_price),
                  "congestion_component_eur_mwh": float(price - reference_price) if mode == "ptdf_coupled" else math.nan,
                  "min_ram_slack_mw": float(min_ram_slack), "binding_constraints": int(binding_constraints),
                  "balance_error_mw": float(residual), "observed_marginal_unit_identified": False}
        for i, segment in enumerate(_SEGMENTS[:4]):
            result[f"{segment}_generation_mw"] = float(dispatch[i])
            result[f"{segment}_bid_eur_mwh"] = float(costs[i])
        return result

    def _zonal(self, row: pd.Series, scenario: Mapping[str, Any]) -> dict:
        costs, capacities = self._stack(row, scenario)
        dispatch = np.zeros(5)
        remaining = float(row.demand_mw)
        active = [i for i in np.argsort(costs, kind="stable") if capacities[i] > 0]
        price = float(costs[active[0]]) if active else float(scenario["scarcity_price_eur_mwh"])
        for i in active:
            amount = min(remaining, float(capacities[i]))
            if amount > 0:
                dispatch[i], price = amount, float(costs[i])
            remaining -= amount
            if remaining <= 1e-9:
                break
        if remaining > 1e-6:
            raise MarginalCostModelError("Zonal dispatch could not balance demand.")
        return self._record(row, scenario, costs, capacities, dispatch, price, 0.0,
                            mode="zonal_degraded", reference_price=math.nan)

    def _coupled(self, period: pd.DataFrame, constraints: pd.DataFrame, scenario: Mapping[str, Any]) -> list[dict]:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix

        rows = [row for _, row in period.iterrows()]
        n = len(rows)
        stacks = [self._stack(row, scenario) for row in rows]
        costs = np.concatenate([*(stack[0] for stack in stacks), np.zeros(n)])
        bounds = [(0.0, float(cap)) for _, caps in stacks for cap in caps] + [(None, None)] * n
        # Generation + unserved demand - net export = demand in each zone.
        er, ec, ev = [], [], []
        for i in range(n):
            for j in range(5):
                er.append(i); ec.append(i * 5 + j); ev.append(1.0)
            er.extend([i, n]); ec.extend([5 * n + i, 5 * n + i]); ev.extend([-1.0, 1.0])
        a_eq = coo_matrix((ev, (er, ec)), shape=(n + 1, 6 * n)).tocsr()
        boundary = self.network_config.get("boundary_net_positions_mw", {})
        boundary_total = sum(float(value) for value in boundary.values())
        b_eq = np.array([*(float(row.demand_mw) for row in rows), -boundary_total])
        ram = constraints.ram_mw.to_numpy(float).copy()
        for zone, position in boundary.items():
            ram -= constraints[f"ptdf_{zone}"].to_numpy(float) * float(position)
        ptdf = constraints[[f"ptdf_{row.zone}" for row in rows]].to_numpy(float)
        ir, ic = np.nonzero(ptdf)
        a_ub = coo_matrix((ptdf[ir, ic], (ir, 5 * n + ic)), shape=(len(constraints), 6 * n)).tocsr()
        solved = linprog(costs, A_ub=a_ub, b_ub=ram, A_eq=a_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")
        if not solved.success:
            raise MarginalCostModelError(f"PTDF dispatch infeasible/failed: {solved.message}")
        slack = np.asarray(solved.ineqlin.residual, dtype=float)
        if (slack < -1e-5).any() or abs(solved.x[5 * n:].sum() + boundary_total) > 1e-5:
            raise MarginalCostModelError("PTDF constraints or net-position conservation failed.")
        reference = float(solved.eqlin.marginals[n])
        return [self._record(row, scenario, *stacks[i], solved.x[5 * i:5 * i + 5],
                             float(solved.eqlin.marginals[i]), float(solved.x[5 * n + i]),
                             mode="ptdf_coupled", reference_price=reference,
                             min_ram_slack=float(slack.min()), binding_constraints=int((slack <= 1e-6).sum()))
                for i, row in enumerate(rows)]


__all__ = ["MarginalCostExpert", "MarginalCostModelError", "candidate_scores", "select_candidate",
           "REQUIRED_FEATURES", "COAL_FEATURES"]
