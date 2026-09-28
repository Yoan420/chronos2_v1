"""Past-only Test2 scarcity policy for the annual CWE reference.

The upstream NYX/Test2/ensemble forecasts and physical signals are supplied by
the caller. This module fits no model and has no file or provider access.
"""
from __future__ import annotations

from datetime import timedelta
from itertools import product

import numpy as np
import pandas as pd


TIMEZONES = {"FR": "Europe/Paris", "BE": "Europe/Brussels",
             "DE": "Europe/Berlin", "NL": "Europe/Amsterdam"}
FEATURES = ("ensemble__q50", "nyx__q50", "test2__q50",
            "own_joint_deficit", "own_residual_stress", "nyx_daily_peak_gap",
            "spike_probability")
RULES = tuple({"nyx_min": nyx, "peak_gap_max": gap,
               "probability_min": probability, "delta_min": delta}
              for nyx, gap, probability, delta in
              product((150., 200.), (25., 50.), (.5, .8, .95), (20., 50.)))


def _grid(first, stop, zone):
    return pd.date_range(str(first), str(stop), tz=TIMEZONES[zone],
                         freq="h", inclusive="left").tz_convert("UTC")


def _checked(features):
    if not isinstance(features, pd.DataFrame) or tuple(features.columns) != FEATURES:
        raise ValueError("Exactly seven declared forecast and physical signals are required")
    index = features.index
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC"
            or not len(index) or index.hasnans or not index.is_unique
            or not index.is_monotonic_increasing or not index.equals(index.floor("h"))):
        raise ValueError("Unique ordered hourly UTC delivery timestamps required")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Finite gate inputs required; no imputation")
    if (not features.spike_probability.between(0., 1.).all()
            or not features.own_joint_deficit.between(0., 1.).all()
            or (features.nyx_daily_peak_gap < -1e-8).any()):
        raise ValueError("Probability, physical deficit or NYX peak gap is invalid")


def _mask(features, rule):
    _checked(features)
    if rule not in RULES:
        raise ValueError("Rule is outside the frozen 24-rule grid")
    return (((features.own_joint_deficit >= .5)
             | (features.own_residual_stress >= 1.))
            & (features["nyx__q50"] >= rule["nyx_min"])
            & (features.nyx_daily_peak_gap <= rule["peak_gap_max"])
            & (features.spike_probability >= rule["probability_min"])
            & (features["test2__q50"] - features["ensemble__q50"] >= rule["delta_min"])
            & (features["test2__q50"] > features["nyx__q50"])).to_numpy(dtype=bool)


def _scores(actual, point):
    error = point - actual
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error**2)))


def select_prior90_policy(past_features: pd.DataFrame, past_actual: pd.Series,
                          *, origin_day: str, zone: str) -> dict:
    """Select the frozen rule using exactly 90 complete, earlier civil days."""
    if zone not in TIMEZONES:
        raise ValueError("Unknown country")
    origin = pd.Timestamp(origin_day).date()
    _checked(past_features)
    expected = _grid(origin - timedelta(days=90), origin, zone)
    if not past_features.index.equals(expected):
        raise ValueError("Exactly 90 strictly earlier complete civil days required")
    if (not isinstance(past_actual, pd.Series)
            or not past_actual.index.equals(expected)
            or "storm" in str(past_actual.name).lower()
            or not np.isfinite(past_actual.to_numpy(dtype=float)).all()):
        raise ValueError("Aligned finite past observed prices required; Storm forbidden")

    actual = past_actual.to_numpy(dtype=float)
    base = past_features["ensemble__q50"].to_numpy(dtype=float)
    test2 = past_features["test2__q50"].to_numpy(dtype=float)
    base_error, test_error = np.abs(base - actual), np.abs(test2 - actual)
    baseline_mae, baseline_rmse = _scores(actual, base)
    days = np.asarray(expected.tz_convert(TIMEZONES[zone]).date)
    candidates = []
    for number, rule in enumerate(RULES):
        active = _mask(past_features, rule)
        point = np.where(active, test2, base)
        mae, rmse = _scores(actual, point)
        hours, count_days = int(active.sum()), len(set(days[active]))
        wins = int((test_error[active] < base_error[active]).sum())
        losses = int((test_error[active] > base_error[active]).sum())
        eligible = (hours >= 5 and count_days >= 3 and wins >= losses
                    and mae < baseline_mae and rmse <= baseline_rmse)
        candidates.append({"rule_index": number, "rule": dict(rule),
                           "active_hours": hours, "active_days": count_days,
                           "wins": wins, "losses": losses,
                           "mae": mae, "rmse": rmse, "eligible": eligible})
    eligible = [row for row in candidates if row["eligible"]]
    best = min(eligible, key=lambda row: (row["mae"], row["active_hours"],
                                          row["rule_index"])) if eligible else None
    return {"protocol": "nyx_test2_spike_gate_prior90_v1", "zone": zone,
            "origin_day": str(origin), "history_start_day": str(origin - timedelta(days=90)),
            "history_end_exclusive": str(origin), "history_hours": len(expected),
            "baseline_mae": baseline_mae, "baseline_rmse": baseline_rmse,
            "candidates": candidates,
            "selected_rule_index": best["rule_index"] if best else None,
            "selected_rule": best["rule"] if best else None}


def apply_prior90_daily(features: pd.DataFrame, policy: dict,
                        *, forecast_issued_at_utc: pd.Timestamp) -> pd.DataFrame:
    """Apply one weekly policy to one delivery day at its D-1 08h cutoff.

    `forecast_issued_at_utc` must be taken from an audited producer receipt;
    this function checks the timestamp but cannot certify the provider itself.
    """
    _checked(features)
    zone = policy.get("zone")
    if policy.get("protocol") != "nyx_test2_spike_gate_prior90_v1" or zone not in TIMEZONES:
        raise ValueError("Unknown weekly policy")
    origin = pd.Timestamp(policy["origin_day"]).date()
    days = np.asarray(features.index.tz_convert(TIMEZONES[zone]).date)
    if len(set(days)) != 1 or not origin <= days[0] < origin + timedelta(days=7):
        raise ValueError("Exactly one delivery day in the policy week required")
    day = days[0]
    if not features.index.equals(_grid(day, day + timedelta(days=1), zone)):
        raise ValueError("The complete physical delivery day is required")
    cutoff = pd.Timestamp(f"{day - timedelta(days=1)} 08:00",
                          tz=TIMEZONES[zone]).tz_convert("UTC")
    issued = pd.Timestamp(forecast_issued_at_utc)
    if issued.tzinfo is None or issued.tz_convert("UTC") > cutoff:
        raise ValueError("Forecast bundle was not available by D-1 08h local")
    gap = features.groupby(days)["nyx__q50"].transform("max") - features["nyx__q50"]
    if not np.allclose(gap.to_numpy(), features.nyx_daily_peak_gap.to_numpy(),
                       rtol=0, atol=1e-8):
        raise ValueError("NYX peak gap must use this complete day curve")
    number = policy.get("selected_rule_index")
    if number is None:
        if policy.get("selected_rule") is not None:
            raise ValueError("Fallback policy cannot contain a rule")
        selected = np.zeros(len(features), dtype=bool)
    else:
        if (type(number) is not int or not 0 <= number < len(RULES)
                or policy.get("selected_rule") != RULES[number]):
            raise ValueError("Selected rule differs from the frozen grid")
        selected = _mask(features, RULES[number])
    point = np.where(selected, features["test2__q50"], features["ensemble__q50"])
    return pd.DataFrame({"prior90_active": selected,
                         "scarcity_guarded_prior90": point}, index=features.index)
