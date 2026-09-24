"""V2 extensible offer-stack dispatch; deliberately independent of V1 model.py.

Offers are physical bid hypotheses, not observed electricity-price forecasts.
This convex period-by-period dispatch does not reproduce EUPHEMIA's block
orders, unit commitment or actual marginal units. Research qualification is
not a certification of production PIT. A full, qualified zero-based network
domain is mandatory before using PTDF @ NP <= RAM: raw JAO RAM is NOT assumed
to have that reference. Missing periods abstain; invalid schemas fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


KEYS = ["delivery_start_utc", "zone"]
_THERMAL = {"ccgt", "ocgt", "coal", "lignite", "chp", "biomass", "oil"}
_FATAL = {"nuclear", "wind", "solar", "hydro_ror", "residual_surplus"}
_ELECTRIC_INPUT = re.compile(r"(?i)(?:^|__|_)(?:actual|target|storm|chronos|kalman|price_lag|q10|q50|q90)(?:$|__|_)")


class DispatchError(ValueError):
    """Malformed physical offer/domain contract; not a missing-data repair."""


@dataclass(frozen=True)
class OfferBook:
    demand: pd.DataFrame
    offers: pd.DataFrame
    qualification: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class DispatchResult:
    prices: pd.DataFrame
    dispatch: pd.DataFrame
    constraints: pd.DataFrame
    audit: dict[str, Any]


def _number(value: Any, name: str, *, low=-math.inf, high=math.inf) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DispatchError(f"{name}: expected a finite numeric assumption.")
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise DispatchError(f"{name}: expected a finite numeric assumption.") from error
    if not math.isfinite(result) or not low <= result <= high:
        raise DispatchError(f"{name}: expected a finite value in [{low}, {high}].")
    return result


def _keyed(frame: pd.DataFrame, *, zone=True) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise DispatchError("Expected a dataframe.")
    required = KEYS if zone else [KEYS[0]]
    if not set(required).issubset(frame):
        raise DispatchError(f"Required identity columns: {required}.")
    result = frame.copy(deep=True)
    if any(pd.isna(x) or pd.Timestamp(x).tzinfo is None for x in result[KEYS[0]]):
        raise DispatchError("Delivery timestamps must be explicitly timezone-aware.")
    result[KEYS[0]] = pd.to_datetime(result[KEYS[0]], utc=True, errors="raise")
    if zone:
        result["zone"] = result["zone"].astype(str)
        if not result.zone.str.fullmatch(r"[A-Z][A-Z0-9_-]{1,20}").all():
            raise DispatchError("Explicit uppercase zone identifiers are required.")
    return result


def _values(frame: pd.DataFrame, column: str, *, nonnegative=False) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan)
    original = frame[column]
    result = pd.to_numeric(original, errors="coerce")
    if (original.notna() & result.isna()).any() or np.isinf(result.to_numpy(float)).any():
        raise DispatchError(f"{column}: malformed/infinite numeric data.")
    values = result.to_numpy(float)
    if nonnegative and (values < 0).any():
        raise DispatchError(f"{column}: negative physical availability or fuel cost.")
    return values


def _booleans(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.zeros(len(frame), dtype=bool)
    if any(not pd.isna(x) and not isinstance(x, (bool, np.bool_)) for x in frame[column]):
        raise DispatchError(f"{column}: true/false research qualification expected.")
    return frame[column].fillna(False).to_numpy(bool)


def _units(units: Mapping[str, Any], column: str, expected: str) -> None:
    if units.get(column) != expected:
        raise DispatchError(f"{column}: explicit unit {expected!r} required.")
    if _ELECTRIC_INPUT.search(column):
        raise DispatchError(f"{column}: electricity predictions/labels are forbidden as physical inputs.")


def build_offers(features: pd.DataFrame, config: Mapping[str, Any]) -> OfferBook:
    """Build a vectorized, technology-specific offer ledger from physical inputs.

    Config keys: demand_basis, residual_netting (by zone), units,
    required_technologies and stack_scope_qualified_by_zone, and segments.
    The explicit stack-scope attestation defaults to false: complete numeric
    technology columns do not prove complete coverage of a national fleet.
    Each segment declares id,
    technology, capacity_column, optional zones/capacity_fraction and a bid.
    ``bid.kind='fuel'`` requires fuel/eua columns in thermal-energy units,
    electrical efficiency, thermal emission factor and VOM. ``assumed`` bids
    require an explicit EUR/MWh value and opportunity-cost hypothesis.
    A known zero capacity is valid without an unused fuel observation; missing
    capacity is never replaced by zero. inputs_qualified is a research flag.
    """
    frame = _keyed(features).reset_index(drop=True)
    if frame.empty or frame.duplicated(KEYS).any() or "demand_mw" not in frame:
        raise DispatchError("Unique, nonempty demand/zone identities are required.")
    basis = config.get("demand_basis")
    if basis not in {"gross", "residual"}:
        raise DispatchError("Declare demand_basis explicitly as gross or residual.")
    units = config.get("units", {})
    _units(units, "demand_mw", "MW")
    mapped_columns = {"demand_mw"}
    demand_values = _values(frame, "demand_mw", nonnegative=basis == "gross")
    demand = frame[KEYS].copy()
    demand["input_demand_mw"] = demand_values
    demand["demand_mw"] = np.maximum(demand_values, 0) if basis == "residual" else demand_values
    demand["demand_basis"] = basis
    research = _booleans(frame, "inputs_qualified") & np.isfinite(demand_values)
    consumed_complete = np.isfinite(demand_values)
    required = config.get("required_technologies", {})
    zones = list(frame.zone.unique())
    if not isinstance(required, Mapping) or set(zones).difference(required):
        raise DispatchError("Declare required_technologies separately for every modeled zone.")
    for zone in zones:
        values = required[zone]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise DispatchError(f"{zone}: declare a nonempty unique required-technology list.")
    stack_scope = config.get("stack_scope_qualified_by_zone", {})
    if not isinstance(stack_scope, Mapping) or any(
        not isinstance(value, (bool, np.bool_)) for value in stack_scope.values()
    ):
        raise DispatchError("Stack-scope qualification must be explicit booleans by zone.")
    netting = config.get("residual_netting", {})
    if basis == "residual" and (not isinstance(netting, Mapping) or set(zones).difference(netting)):
        raise DispatchError("Residual demand requires an explicit already-netted technology list per zone.")
    if basis == "residual" and any(not isinstance(netting[z], list) for z in zones):
        raise DispatchError("Residual netting technologies must be a list for every zone.")
    specifications = config.get("segments")
    if not isinstance(specifications, list) or not specifications:
        raise DispatchError("Declare the offer segments explicitly.")
    ids, fractions, offers, segment_audits = set(), {}, [], []
    represented, available = {}, {}
    for spec in specifications:
        segment, technology = str(spec.get("id", "")), str(spec.get("technology", ""))
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", segment) or segment in ids or not technology:
            raise DispatchError("Segment ids must be unique and technology explicit.")
        ids.add(segment)
        selected_zones = spec.get("zones", zones)
        if not isinstance(selected_zones, list) or len(selected_zones) != len(set(selected_zones)):
            raise DispatchError("Segment zones must be a unique list.")
        mask = frame.zone.isin(selected_zones).to_numpy()
        if basis == "residual":
            duplicate = [zone for zone in set(selected_zones).intersection(zones) if technology in netting[zone]]
            if duplicate:
                raise DispatchError(f"{technology}: already netted from residual load in {sorted(duplicate)}; double counting forbidden.")
        column = str(spec.get("capacity_column", ""))
        _units(units, column, "MW")
        mapped_columns.add(column)
        fraction = _number(spec.get("capacity_fraction", 1.0), "capacity_fraction", low=0, high=1)
        for zone in selected_zones:
            group = (zone, column)
            fractions[group] = fractions.get(group, 0.0) + fraction
            if fractions[group] > 1.0 + 1e-12:
                raise DispatchError(f"{zone}/{column}: capacity fractions exceed one; shared CHP/thermal capacity double counted.")
        capacity = _values(frame, column, nonnegative=True) * fraction
        bid = spec.get("bid", {})
        if bid.get("kind") == "fuel":
            fuel_column, eua_column = str(bid.get("fuel_column", "")), str(bid.get("eua_column", ""))
            _units(units, fuel_column, "EUR/MWh_th")
            _units(units, eua_column, "EUR/tCO2")
            mapped_columns.update([fuel_column, eua_column])
            bounds = {"ccgt": (0.35, 0.70), "ocgt": (0.20, 0.50), "coal": (0.20, 0.55),
                      "lignite": (0.20, 0.50), "chp": (0.15, 0.65)}.get(technology, (0.10, 0.80))
            efficiency = _number(bid.get("efficiency"), "electrical efficiency", low=bounds[0], high=bounds[1])
            emissions = _number(bid.get("emissions_tco2_mwh_th"), "thermal emissions", low=0, high=0.8)
            vom = _number(bid.get("vom_eur_mwh"), "VOM", low=0, high=100)
            premium = _number(bid.get("bid_premium_eur_mwh", 0), "bid premium", low=-50, high=50)
            costs = (_values(frame, fuel_column, nonnegative=True)
                     + _values(frame, eua_column, nonnegative=True) * emissions) / efficiency + vom + premium
            assumption = f"fuel plus CO2 / electrical efficiency + VOM; explicit bid premium {premium:g} EUR/MWh"
        elif bid.get("kind") == "assumed":
            assumption = str(bid.get("hypothesis", "")).strip()
            if not assumption:
                raise DispatchError(f"{segment}: hydro/CHP/bio/nuclear opportunity bid requires an explicit hypothesis.")
            costs = np.full(len(frame), _number(bid.get("value_eur_mwh"), "assumed bid", low=-1000, high=20000))
        else:
            raise DispatchError(f"{segment}: bid kind must be fuel or assumed; no electricity-price feature allowed.")
        valid = np.isfinite(capacity) & ((capacity == 0) | np.isfinite(costs))
        consumed_complete[mask] &= valid[mask]
        represented.setdefault(technology, np.zeros(len(frame), dtype=bool))
        available.setdefault(technology, np.ones(len(frame), dtype=bool))
        represented[technology] |= mask
        available[technology][mask] &= valid[mask]
        keep = mask & valid & (capacity > 0)
        block = frame.loc[keep, KEYS].copy()
        block["segment"], block["technology"] = segment, technology
        block["capacity_mw"], block["bid_eur_mwh"] = capacity[keep], costs[keep]
        fatal = spec.get("is_fatal", technology in _FATAL)
        if not isinstance(fatal, (bool, np.bool_)):
            raise DispatchError(f"{segment}: is_fatal must be an explicit boolean.")
        block["is_fatal"] = bool(fatal)
        offers.append(block)
        segment_audits.append({"segment": segment, "technology": technology, "capacity_column": column,
                               "capacity_fraction": fraction, "bid_hypothesis": assumption,
                               "missing_rows": int((mask & ~valid).sum()), "known_zero_rows": int((mask & (capacity == 0)).sum())})
    if basis == "residual":
        surplus_bid = _number(config.get("residual_surplus_bid_eur_mwh"), "residual surplus bid", low=-1000, high=200)
        surplus = np.maximum(-demand_values, 0)
        keep = np.isfinite(surplus) & (surplus > 0)
        block = frame.loc[keep, KEYS].copy()
        block["segment"], block["technology"] = "_residual_surplus", "residual_surplus"
        block["capacity_mw"], block["bid_eur_mwh"], block["is_fatal"] = surplus[keep], surplus_bid, True
        offers.append(block)
        demand["residual_surplus_mw"] = surplus
    else:
        demand["residual_surplus_mw"] = 0.0
    qualification = frame[KEYS].copy()
    qualification["inputs_qualified"] = research
    qualification["consumed_inputs_complete"] = consumed_complete
    missing_lists = [[] for _ in range(len(frame))]
    for zone in zones:
        zone_mask = frame.zone.eq(zone).to_numpy()
        for tech in required[zone]:
            complete = represented.get(tech, np.zeros(len(frame), dtype=bool)) & available.get(tech, np.zeros(len(frame), dtype=bool))
            for index in np.flatnonzero(zone_mask & ~complete):
                missing_lists[index].append(tech)
    qualification["missing_technologies"] = [",".join(value) for value in missing_lists]
    qualification["stack_scope_qualified"] = frame.zone.map(lambda zone: bool(stack_scope.get(zone, False)))
    qualification["stack_complete"] = np.array([not value for value in missing_lists]) & qualification.stack_scope_qualified.to_numpy(bool)
    qualification["physical_scope"] = "residual_stack_proxy" if basis == "residual" else "gross_stack_approximation"
    allowed = set(KEYS) | mapped_columns | {"inputs_qualified", "forecast_origin_utc", "provenance_id", "demand_basis"}
    if any(_ELECTRIC_INPUT.search(column) for column in set(frame).difference(allowed)):
        raise DispatchError("Electricity targets, lags or model forecasts are not offer-builder inputs.")
    ledger = pd.concat(offers, ignore_index=True) if offers else pd.DataFrame(columns=[*KEYS, "segment", "technology", "capacity_mw", "bid_eur_mwh", "is_fatal"])
    audit = {"schema_version": 2, "demand_basis": basis, "residual_netting": dict(netting),
             "segments": segment_audits, "qualified_rows": int(research.sum()),
             "complete_stack_rows": int(qualification.stack_complete.sum()),
             "consumed_inputs_complete_rows": int(consumed_complete.sum()),
             "stack_scope_qualified_by_zone": {zone: bool(stack_scope.get(zone, False)) for zone in zones},
             "missing_capacity_imputed": False, "electricity_price_inputs": False,
             "full_renewable_curtailment_modeled": False,
             "curtailment_scope": "declared fatal offers only; net surplus only for already-netted renewables",
             "qualification_scope": "research source, units and availability contract; not production certification",
             "production_pit_evidence": False, "production_modified": False,
             "limitations": "Independent convex periods; opportunity bids are declared hypotheses. Residual inputs cannot reconstruct curtailment beyond the net surplus."}
    return OfferBook(demand, ledger, qualification, audit)


def _qualification(demand: pd.DataFrame, qualification: pd.DataFrame | None) -> pd.DataFrame:
    result = demand[KEYS].copy()
    if qualification is not None:
        q = _keyed(qualification)
        if q.duplicated(KEYS).any():
            raise DispatchError("Duplicate qualification keys.")
        result = result.merge(q, on=KEYS, how="left", validate="one_to_one")
    result["inputs_qualified"] = _booleans(result, "inputs_qualified")
    result["stack_complete"] = _booleans(result, "stack_complete")
    # A direct offer ledger has no hidden source columns; build_offers supplies
    # this additional flag because it intentionally excludes unavailable rows.
    result["consumed_inputs_complete"] = (
        _booleans(result, "consumed_inputs_complete") if "consumed_inputs_complete" in result else True
    )
    if "physical_scope" not in result:
        result["physical_scope"] = "unspecified_stack"
    return result


def _price_record(row, q, *, candidate_id, mode, raw=math.nan, shortage=math.nan,
                  position=math.nan, generation=math.nan, capacity=math.nan, thermal_margin=math.nan,
                  curtailment=math.nan, reference=math.nan, reason=None, network_ok=True, balance=math.nan):
    eligible = bool(q.inputs_qualified and q.stack_complete and network_ok and np.isfinite(raw))
    if reason is None:
        reason = ("shortage_source_incomplete" if shortage > 1e-7 and not (q.inputs_qualified and q.stack_complete) else
                  "inputs_unqualified" if not q.inputs_qualified else "stack_incomplete" if not q.stack_complete else
                  "network_unqualified" if not network_ok else "price_indeterminate" if not np.isfinite(raw) else "qualified_proxy")
    if reason != "qualified_proxy":
        eligible = False
    raw_regime = "scarcity_proxy" if shortage > 1e-7 else "surplus_proxy" if curtailment > 1e-7 else "normal_proxy"
    return {**{key: row[key] for key in KEYS}, "candidate_id": candidate_id,
            "price_eur_mwh": float(raw) if eligible else math.nan, "raw_price_eur_mwh": float(raw),
            "eligible": eligible, "status": reason, "network_mode": mode,
            "inputs_qualified": bool(q.inputs_qualified), "stack_complete": bool(q.stack_complete),
            "consumed_inputs_complete": bool(q.consumed_inputs_complete),
            "physical_scope": q.physical_scope, "raw_regime": raw_regime if np.isfinite(raw) else "unavailable",
            "regime": raw_regime if eligible else "unavailable", "shortage_mw": float(shortage),
            "demand_mw": float(row.demand_mw), "generation_mw": float(generation),
            "net_position_mw": float(position), "capacity_margin_mw": float(capacity - row.demand_mw - position),
            "thermal_margin_mw": float(thermal_margin), "curtailment_mw": float(curtailment),
            "reference_price_eur_mwh": float(reference),
            "congestion_component_eur_mwh": float(raw - reference) if np.isfinite(reference) else math.nan,
            "balance_error_mw": float(balance), "observed_marginal_unit_identified": False}


def dispatch_periods(
    demand: pd.DataFrame, offers: pd.DataFrame, *, scarcity_price_eur_mwh: float,
    qualification: pd.DataFrame | None = None, network: pd.DataFrame | None = None,
    network_config: Mapping[str, Any] | None = None, candidate_id: str = "central",
) -> DispatchResult:
    """Dispatch independent periods and return dual prices plus the offer ledger.

    All active PTDF coordinates must be represented by modeled zones or explicit
    fixed boundary positions. Domain reference and boundary qualification are
    required *before* computing even a diagnostic coupled price. No network
    input means an explicitly isolated zonal model, never inferred NTCs.
    """
    penalty = _number(scarcity_price_eur_mwh, "explicit scarcity bid", low=500, high=20000)
    d = _keyed(demand).reset_index(drop=True)
    if d.empty or d.duplicated(KEYS).any() or "demand_mw" not in d:
        raise DispatchError("Unique nonempty demand keys and demand_mw are required.")
    d["demand_mw"] = _values(d, "demand_mw", nonnegative=True)
    o = _keyed(offers).reset_index(drop=True)
    required_offers = {"segment", "capacity_mw", "bid_eur_mwh"}
    if not required_offers.issubset(o) or o.duplicated([*KEYS, "segment"]).any():
        raise DispatchError("Offer identity/segment/capacity/bid contract is malformed.")
    extra = set(o).difference({*KEYS, *required_offers, "technology", "is_fatal"})
    if extra:
        raise DispatchError(f"Unrecognized offer inputs, never electricity forecasts: {sorted(extra)}")
    o["capacity_mw"] = _values(o, "capacity_mw", nonnegative=True)
    o["bid_eur_mwh"] = _values(o, "bid_eur_mwh")
    if ((o.capacity_mw > 0) & o.bid_eur_mwh.ge(penalty)).any():
        raise DispatchError("Scarcity penalty must exceed every available generation bid.")
    if "technology" not in o:
        o["technology"] = o.segment
    if "is_fatal" not in o:
        o["is_fatal"] = o.technology.isin(_FATAL)
    else:
        o["is_fatal"] = _booleans(o, "is_fatal")
    if not o.empty and o.merge(d[KEYS], on=KEYS, how="left", indicator=True)["_merge"].ne("both").any():
        raise DispatchError("Offers outside the demand support are forbidden.")
    q = _qualification(d, qualification).set_index(KEYS)
    cfg = dict(network_config or {})
    net = _keyed(network, zone=False) if network is not None else None
    if net is not None and (not {"cnec_id", "ram_mw"}.issubset(net) or net.duplicated([KEYS[0], "cnec_id"]).any()):
        raise DispatchError("CNEC/period identities and RAM are required.")
    if net is None:
        if cfg.get("boundary_net_positions_mw"):
            raise DispatchError("Boundary exchanges require an explicit complete network domain.")
        prices, dispatch = _zonal_batch(d, o, q, penalty, candidate_id)
        return _result(d, prices, dispatch, pd.DataFrame(), candidate_id, network_requested=False)
    net_groups = dict(tuple(net.groupby(KEYS[0], sort=True))) if net is not None else {}
    offer_groups = dict(tuple(o.groupby(KEYS[0], sort=True)))
    prices, dispatches, constraint_rows = [], [], []
    modeled = set(d.zone.unique())
    for timestamp, period in d.groupby(KEYS[0], sort=True):
        ledger = offer_groups.get(timestamp, o.iloc[:0]).copy().reset_index(drop=True)
        missing = (not np.isfinite(period.demand_mw).all()
                   or not np.isfinite(ledger[["capacity_mw", "bid_eur_mwh"]].to_numpy(float)).all()
                   or not q.loc[[(timestamp, z) for z in period.zone], "consumed_inputs_complete"].all())
        period_network = net_groups.get(timestamp)
        blocked = "missing_period_inputs" if missing and net is not None else None
        if net is not None:
            if cfg.get("domain_reference_qualified") is not True:
                blocked = "unqualified_domain_reference"
            elif cfg.get("inputs_qualified") is not True:
                blocked = "network_inputs_unqualified"
            elif period_network is None or period_network.empty:
                blocked = "network_period_missing"
            elif set(period.zone) != modeled:
                blocked = "modeled_zone_period_missing"
            elif cfg.get("boundary_net_positions_mw") and cfg.get("boundary_qualified") is not True:
                blocked = "boundary_unqualified"
        elif cfg.get("boundary_net_positions_mw"):
            raise DispatchError("Boundary exchanges require an explicit complete network domain.")
        if blocked:
            for _, row in period.iterrows():
                prices.append(_price_record(row, q.loc[(timestamp, row.zone)], candidate_id=candidate_id,
                              mode="ptdf_blocked" if net is not None else "zonal_degraded", reason=blocked, network_ok=False))
            continue
        if net is None:
            for _, row in period.iterrows():
                zone_ledger = ledger.loc[ledger.zone.eq(row.zone)].copy().reset_index(drop=True)
                zone_q = q.loc[(timestamp, row.zone)]
                if (not np.isfinite(row.demand_mw) or not zone_q.consumed_inputs_complete
                        or not np.isfinite(zone_ledger[["capacity_mw", "bid_eur_mwh"]].to_numpy(float)).all()):
                    prices.append(_price_record(row, zone_q, candidate_id=candidate_id,
                                  mode="zonal_degraded", reason="missing_period_inputs"))
                    continue
                allocation, raw, shortage = _zonal(float(row.demand_mw), zone_ledger, penalty)
                record, dispatch = _output_rows(row, zone_q, zone_ledger, allocation,
                     raw, shortage, 0.0, candidate_id=candidate_id, mode="zonal_degraded")
                prices.append(record); dispatches.append(dispatch)
        else:
            p, allocations, constraints = _coupled(period, ledger, period_network, cfg, penalty, q, candidate_id)
            prices.extend(p); dispatches.append(allocations); constraint_rows.append(constraints)
    return _result(d, pd.DataFrame(prices), pd.concat(dispatches, ignore_index=True) if dispatches else pd.DataFrame(),
                   pd.concat(constraint_rows, ignore_index=True) if constraint_rows else pd.DataFrame(),
                   candidate_id, network_requested=True)


def _result(demand, prices, dispatch, constraints, candidate_id, *, network_requested):
    price_frame = prices.sort_values(KEYS, kind="stable").reset_index(drop=True)
    audit = {"schema_version": 2, "candidate_id": candidate_id, "periods": int(demand[KEYS[0]].nunique()),
             "eligible_rows": int(price_frame.eligible.sum()), "blocked_rows": int((~price_frame.eligible).sum()),
             "network_requested": network_requested, "negative_or_scarcity_bids_invented": False,
             "shortage_from_omissions_is_not_active_scarcity": True, "electricity_price_inputs": False,
             "production_modified": False, "production_pit_evidence": False,
             "price_interpretation": "Independent convex dispatch balance dual; no observed marginal generating unit identified."}
    return DispatchResult(price_frame, dispatch, constraints, audit)


def _zonal_batch(demand, offers, qualifications, penalty, candidate_id):
    """Vectorized independent merit orders, without per-hour Pandas slicing.

    Sorting and cumulative allocation are scoped by the exact UTC/zone key.
    No calendars are resampled or filled. Incomplete keys do not contaminate
    their neighbors and never produce even a raw diagnostic price.
    """
    count = len(demand)
    d = demand.copy()
    d["__row"] = np.arange(count)
    q = qualifications.reindex(pd.MultiIndex.from_frame(d[KEYS]))
    ledger = offers.assign(__offer=np.arange(len(offers))).merge(
        d[[*KEYS, "demand_mw", "__row"]], on=KEYS, how="left", validate="many_to_one", sort=False
    )
    row_ids = ledger["__row"].to_numpy(int)
    missing_offers = np.zeros(count, dtype=bool)
    np.logical_or.at(missing_offers, row_ids, ~np.isfinite(ledger[["capacity_mw", "bid_eur_mwh"]].to_numpy(float)).all(axis=1))
    loads = d.demand_mw.to_numpy(float)
    ready = np.isfinite(loads) & q.consumed_inputs_complete.to_numpy(bool) & ~missing_offers
    active = ledger.loc[ready[row_ids]].sort_values(["__row", "bid_eur_mwh", "__offer"], kind="stable").copy()
    ids = active["__row"].to_numpy(int)
    capacities, bids = active.capacity_mw.to_numpy(float), active.bid_eur_mwh.to_numpy(float)
    before = active.groupby("__row", sort=False).capacity_mw.cumsum().to_numpy(float) - capacities
    allocation = np.minimum(capacities, np.maximum(loads[ids] - before, 0))
    unused = np.maximum(capacities - allocation, 0)
    generation = np.bincount(ids, weights=allocation, minlength=count)
    total_capacity = np.bincount(ids, weights=capacities, minlength=count)
    thermal_margin = np.bincount(ids, weights=unused * active.technology.isin(_THERMAL).to_numpy(), minlength=count)
    curtailment = np.bincount(ids, weights=unused * active.is_fatal.to_numpy(bool), minlength=count)
    shortage = np.maximum(loads - generation, 0)
    cheapest, marginal = np.full(count, np.inf), np.full(count, -np.inf)
    positive, accepted = capacities > 0, allocation > 1e-9
    np.minimum.at(cheapest, ids[positive], bids[positive])
    np.maximum.at(marginal, ids[accepted], bids[accepted])
    raw = np.where(np.isfinite(marginal), marginal, cheapest)
    raw[~np.isfinite(raw)] = np.nan
    raw[shortage > 1e-7] = penalty
    raw[~ready] = np.nan
    inputs_ok, stack_ok = q.inputs_qualified.to_numpy(bool), q.stack_complete.to_numpy(bool)
    eligible = ready & inputs_ok & stack_ok & np.isfinite(raw)
    status = np.full(count, "qualified_proxy", dtype=object)
    status[~np.isfinite(raw)] = "price_indeterminate"
    status[~stack_ok] = "stack_incomplete"
    status[~inputs_ok] = "inputs_unqualified"
    status[(shortage > 1e-7) & ~(inputs_ok & stack_ok)] = "shortage_source_incomplete"
    status[~ready] = "missing_period_inputs"
    raw_regime = np.where(shortage > 1e-7, "scarcity_proxy", np.where(curtailment > 1e-7, "surplus_proxy", "normal_proxy"))
    balance = generation + shortage - loads
    if (np.abs(balance[ready]) > np.maximum(1e-5, 1e-8 * np.abs(loads[ready]))).any():
        raise DispatchError("Dispatch violates energy conservation.")
    visible = lambda values: np.where(ready, values, np.nan)
    prices = d[KEYS].copy()
    columns = {
        "candidate_id": candidate_id, "price_eur_mwh": np.where(eligible, raw, np.nan),
        "raw_price_eur_mwh": raw, "eligible": eligible, "status": status, "network_mode": "zonal_degraded",
        "inputs_qualified": inputs_ok, "stack_complete": stack_ok,
        "consumed_inputs_complete": q.consumed_inputs_complete.to_numpy(bool),
        "physical_scope": q.physical_scope.to_numpy(),
        "raw_regime": np.where(np.isfinite(raw), raw_regime, "unavailable"),
        "regime": np.where(eligible, raw_regime, "unavailable"), "shortage_mw": visible(shortage),
        "demand_mw": loads, "generation_mw": visible(generation), "net_position_mw": visible(np.zeros(count)),
        "capacity_margin_mw": visible(total_capacity - loads), "thermal_margin_mw": visible(thermal_margin),
        "curtailment_mw": visible(curtailment), "reference_price_eur_mwh": np.nan,
        "congestion_component_eur_mwh": np.nan, "balance_error_mw": visible(balance),
        "observed_marginal_unit_identified": False,
    }
    prices = pd.concat([prices, pd.DataFrame(columns, index=prices.index)], axis=1)
    active["generation_mw"], active["unused_capacity_mw"] = allocation, unused
    active["candidate_id"], active["eligible"] = candidate_id, eligible[ids]
    dispatch = active.sort_values([KEYS[0], "__row", "__offer"], kind="stable").drop(columns=["demand_mw", "__row", "__offer"])
    return prices, dispatch.reset_index(drop=True)


def _zonal(demand: float, offers: pd.DataFrame, penalty: float) -> tuple[np.ndarray, float, float]:
    capacity = offers.capacity_mw.to_numpy(float)
    bids = offers.bid_eur_mwh.to_numpy(float)
    order = np.argsort(bids, kind="stable")
    cumulative_before = np.r_[0.0, np.cumsum(capacity[order])[:-1]] if len(order) else np.array([])
    allocations_sorted = np.minimum(capacity[order], np.maximum(demand - cumulative_before, 0))
    allocation = np.zeros(len(offers))
    allocation[order] = allocations_sorted
    shortage = max(0.0, demand - float(allocation.sum()))
    accepted = order[allocations_sorted > 1e-9]
    positive = order[capacity[order] > 0]
    raw = penalty if shortage > 1e-7 else float(bids[accepted[-1]]) if len(accepted) else float(bids[positive[0]]) if len(positive) else math.nan
    return allocation, raw, shortage


def _output_rows(row, q, ledger, allocation, price, shortage, net_position, *, candidate_id, mode,
                 reference=math.nan, network_ok=True):
    capacity = ledger.capacity_mw.to_numpy(float)
    unused = np.maximum(capacity - allocation, 0)
    thermal = ledger.technology.isin(_THERMAL).to_numpy()
    fatal = ledger.is_fatal.to_numpy(bool)
    generation = float(np.sum(allocation))
    error = generation + shortage - float(row.demand_mw) - net_position
    if abs(error) > max(1e-5, 1e-8 * abs(float(row.demand_mw))):
        raise DispatchError("Dispatch violates energy conservation.")
    record = _price_record(row, q, candidate_id=candidate_id, mode=mode, raw=price, shortage=shortage,
                          position=net_position, generation=generation, capacity=float(capacity.sum()),
                          thermal_margin=float(unused[thermal].sum()), curtailment=float(unused[fatal].sum()),
                          reference=reference, network_ok=network_ok, balance=error)
    dispatch = ledger.copy()
    dispatch["generation_mw"], dispatch["unused_capacity_mw"] = allocation, unused
    dispatch["candidate_id"], dispatch["eligible"] = candidate_id, record["eligible"]
    return record, dispatch


def _coupled(period, ledger, constraints, cfg, penalty, qualifications, candidate_id):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    zones = list(period.zone)
    full = cfg.get("zones")
    boundary = dict(cfg.get("boundary_net_positions_mw", {}))
    if not isinstance(full, list) or not full or len(full) != len(set(full)):
        raise DispatchError("The full active PTDF coordinate universe must be declared.")
    if set(zones).intersection(boundary) or set(zones) | set(boundary) != set(full):
        raise DispatchError("No missing PTDF coordinates or implicit zero-position boundaries are permitted.")
    if boundary and not str(cfg.get("boundary_hypothesis", "")).strip():
        raise DispatchError("Explicit fixed boundary hypothesis required.")
    numerical = ["ram_mw", *[f"ptdf_{zone}" for zone in full]]
    if not set(numerical).issubset(constraints) or {c for c in constraints if c.startswith("ptdf_")} != set(numerical[1:]):
        raise DispatchError("PTDF columns do not match the complete declared coordinate universe.")
    matrix = pd.DataFrame({column: _values(constraints, column) for column in numerical}, index=constraints.index)
    timestamp = period[KEYS[0]].iloc[0]
    if not np.isfinite(matrix.to_numpy(float)).all():
        blocked = [_price_record(row, qualifications.loc[(timestamp, row.zone)], candidate_id=candidate_id,
                     mode="ptdf_blocked", reason="network_period_incomplete", network_ok=False) for _, row in period.iterrows()]
        return blocked, pd.DataFrame(), pd.DataFrame()
    boundary_values = {zone: _number(value, f"boundary {zone}") for zone, value in boundary.items()}
    n, m = len(zones), len(ledger)
    lookup = {zone: i for i, zone in enumerate(zones)}
    generation_zone = ledger.zone.map(lookup).to_numpy(int)
    # Variables: all offers, unserved load per zone, net positions per zone.
    r = np.r_[generation_zone, np.arange(n), np.arange(n), np.full(n, n)]
    c = np.r_[np.arange(m), m + np.arange(n), m + n + np.arange(n), m + n + np.arange(n)]
    v = np.r_[np.ones(m + n), -np.ones(n), np.ones(n)]
    equality = coo_matrix((v, (r, c)), shape=(n + 1, m + 2 * n)).tocsr()
    rhs = np.r_[period.demand_mw.to_numpy(float), -sum(boundary_values.values())]
    ptdf = matrix[[f"ptdf_{zone}" for zone in zones]].to_numpy(float)
    nr, nc = np.nonzero(ptdf)
    inequality = coo_matrix((ptdf[nr, nc], (nr, m + n + nc)), shape=(len(matrix), m + 2 * n)).tocsr()
    ram = matrix.ram_mw.to_numpy(float).copy()
    for zone, position in boundary_values.items():
        ram -= matrix[f"ptdf_{zone}"].to_numpy(float) * position
    costs = np.r_[ledger.bid_eur_mwh.to_numpy(float), np.full(n, penalty), np.zeros(n)]
    bounds = [(0, float(cap)) for cap in ledger.capacity_mw] + [(0, float(load)) for load in period.demand_mw] + [(None, None)] * n
    solved = linprog(costs, A_eq=equality, b_eq=rhs, A_ub=inequality, b_ub=ram, bounds=bounds, method="highs")
    if not solved.success:
        blocked = [_price_record(row, qualifications.loc[(timestamp, row.zone)], candidate_id=candidate_id,
                     mode="ptdf_blocked", reason="dispatch_infeasible", network_ok=False) for _, row in period.iterrows()]
        return blocked, pd.DataFrame(), pd.DataFrame()
    positions = solved.x[m + n:]
    if abs(positions.sum() + sum(boundary_values.values())) > 1e-5 or np.min(solved.ineqlin.residual) < -1e-5:
        raise DispatchError("Coupled net-position/RAM conservation failed.")
    full_period_qualified = bool(qualifications.loc[[(timestamp, z) for z in zones], ["inputs_qualified", "stack_complete"]].to_numpy(bool).all())
    prices, dispatch = [], []
    for i, (_, row) in enumerate(period.iterrows()):
        mask = generation_zone == i
        price, detail = _output_rows(row, qualifications.loc[(timestamp, row.zone)], ledger.loc[mask], solved.x[:m][mask],
            float(solved.eqlin.marginals[i]), float(solved.x[m + i]), float(positions[i]),
            candidate_id=candidate_id, mode="ptdf_coupled", reference=float(solved.eqlin.marginals[-1]),
            network_ok=full_period_qualified)
        if not full_period_qualified and price["status"] == "network_unqualified":
            price["status"] = "coupled_stack_incomplete"
        prices.append(price); dispatch.append(detail)
    diagnostics = constraints[[KEYS[0], "cnec_id", "ram_mw"]].copy()
    diagnostics["candidate_id"] = candidate_id
    diagnostics["ptdf_expression_mw"] = diagnostics.ram_mw.to_numpy(float) - solved.ineqlin.residual
    diagnostics["remaining_ram_mw"] = solved.ineqlin.residual
    diagnostics["ram_relaxation_value_eur_mwh"] = -solved.ineqlin.marginals
    diagnostics["eligible"] = full_period_qualified
    return prices, pd.concat(dispatch, ignore_index=True), diagnostics


__all__ = ["OfferBook", "DispatchResult", "DispatchError", "build_offers", "dispatch_periods"]
