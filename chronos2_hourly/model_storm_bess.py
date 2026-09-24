"""One-cycle battery strategies from arXiv:2609.00089v1, section 6.2.

The paper specifies coupled loop bids but not a numerical clearing predicate.
We simulate acceptance on the combined loop surplus, as a price taker without
paradoxical rejection. This is explicit, not a claim to reproduce author code
or exchange clearing. Physical DST days retain every chronological hour.
"""
from __future__ import annotations

from numbers import Real
import math
import numpy as np

CAPACITY_MWH = 1.0
EFFICIENCY = 0.95
CYCLE_COST_EUR = 25.0
QUANTILE_ALPHA = 0.80
STRATEGIES = ("quantile_based", "unlimited_bid")


def _prices(values):
    if values is None:
        raise ValueError("A complete hourly price vector is required")
    values = list(values)
    if len(values) not in (23, 24, 25) or any(
        not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("Expected finite prices for one physical 23/24/25-hour day")
    return np.asarray(values, dtype=float)


def select_bess_trade(forecast):
    """Choose one earlier buy/later sell using forecasts only; stable earliest ties."""
    predicted = _prices(forecast)
    best = None
    for buy in range(len(predicted) - 1):
        for sell in range(buy + 1, len(predicted)):
            profit = float(EFFICIENCY * predicted[sell] - predicted[buy] / EFFICIENCY - CYCLE_COST_EUR)
            if best is None or profit > best["predicted_profit_eur"]:
                best = {"buy_index": buy, "sell_index": sell, "predicted_profit_eur": profit}
    return {**best, "proposed": best["predicted_profit_eur"] > 0,
            "grid_buy_mwh": CAPACITY_MWH / EFFICIENCY,
            "grid_sell_mwh": CAPACITY_MWH * EFFICIENCY}


def simulate_bess_day(forecast, observed, *, strategy="unlimited_bid", lower=None, upper=None):
    """Settle a fixed forecast-selected pair. Rejected/no-trade days contribute zero.

    QB uses native/calibrated P10/P90 at alpha=80%. Missing quantiles must be
    handled as unavailable by the caller, never replaced with point forecasts.
    """
    if strategy not in STRATEGIES:
        raise ValueError("Unknown BESS strategy")
    predicted, actual = _prices(forecast), _prices(observed)
    if predicted.shape != actual.shape:
        raise ValueError("Forecast and observed hours must match")
    lo = hi = None
    if strategy == "quantile_based":
        lo, hi = _prices(lower), _prices(upper)
        if lo.shape != predicted.shape or hi.shape != predicted.shape or (lo > predicted).any() or (hi < predicted).any():
            raise ValueError("Quantiles must be aligned and ordered P10 <= P50 <= P90")
    plan = select_bess_trade(predicted)
    record = {**plan, "strategy": strategy, "alpha": QUANTILE_ALPHA if lo is not None else None,
              "executed": False, "pnl_eur": 0.0, "operating_cost_eur": 0.0,
              "buy_limit_eur_mwh": None, "sell_limit_eur_mwh": None, "loop_surplus_eur": None}
    if not plan["proposed"]:
        record["status"] = "no_profitable_forecast_pair"
        return record
    buy, sell = plan["buy_index"], plan["sell_index"]
    accepted = True
    if lo is not None:
        buy_limit, sell_limit = float(hi[buy]), float(lo[sell])
        surplus = float((buy_limit - actual[buy]) / EFFICIENCY + EFFICIENCY * (actual[sell] - sell_limit))
        accepted = surplus >= 0
        record.update(buy_limit_eur_mwh=buy_limit, sell_limit_eur_mwh=sell_limit, loop_surplus_eur=surplus)
    record.update(executed=bool(accepted), status="executed" if accepted else "loop_rejected")
    if accepted:
        record.update(pnl_eur=float(EFFICIENCY * actual[sell] - actual[buy] / EFFICIENCY - CYCLE_COST_EUR),
                      operating_cost_eur=CYCLE_COST_EUR)
    return record


METHODOLOGY = {
    "paper": "Lipiecki & Weron, arXiv:2609.00089v1, section 6.2, pages 15–17",
    "paper_url": "https://arxiv.org/abs/2609.00089",
    "capacity_mwh": CAPACITY_MWH, "charge_efficiency": EFFICIENCY,
    "discharge_efficiency": EFFICIENCY, "round_trip_efficiency": EFFICIENCY ** 2,
    "initial_soc_mwh": 0, "final_soc_mwh": 0, "maximum_cycles_per_day": 1,
    "grid_buy_mwh": CAPACITY_MWH / EFFICIENCY, "grid_sell_mwh": CAPACITY_MWH * EFFICIENCY,
    "operating_cost_eur_per_executed_cycle": CYCLE_COST_EUR,
    "selection": "Argmax over buy < sell of eta*forecast_sell - forecast_buy/eta - 25; submit only if > 0.",
    "unlimited_bid": "Every submitted pair executes; realized net profit may be negative.",
    "quantile_based": "alpha=80%, tau=10%; buy limit=P90(buy), sell limit=P10(sell); coupled all-or-none loop.",
    "loop_acceptance": "(buy_limit-actual_buy)/eta + eta*(actual_sell-sell_limit) >= 0",
    "execution_assumption": "Price-taking combined-surplus loop simulation, no paradoxical rejections. Authors' clearing code not available/verified.",
    "loop_specification_url": "https://hupx.hu/uploads/Kereskedes/Keresked%C3%A9si%20rendszer/DAM/Smart%20block%20changes_20240510.pdf",
    "available_alpha": [QUANTILE_ALPHA], "quantile_interpolation": False,
    "dst_adaptation": "All 23/24/25 physical local-day hours preserved, unlike the paper's 24-hour normalization.",
    "point_forecast_convention": "NYX native P50; Storm published central forecast treated as the fixed median for both strategies.",
    "not_live_trading": True,
}
