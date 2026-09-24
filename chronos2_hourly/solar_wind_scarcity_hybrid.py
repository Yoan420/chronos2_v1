"""Isolated, interpretable selection between complete NYX and Test2 quantiles.

The rule is selected exclusively from complete preceding 90-day DE/NL panels
of saved out-of-time predictions. Applying it uses only NYX forecast levels,
NYX uncertainty, its SAME-day forecast peak, and frozen physical forecasts.
Neither realized prices nor Test2 outputs determine the current selection.

All three quantiles come intact from the chosen model. No average, clamp,
forced P50 increase, source synchronization, disk write or production mutation
is performed. Timestamp validation does not certify source-publication PIT.
The already-inspected retrospective history is exploratory, not an untouched
validation set. The caller owns source/identity pinning and label availability.
"""
from __future__ import annotations

from datetime import date, timedelta
from itertools import product

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "solar_wind_scarcity_hybrid_v1"
WINDOW_DAYS = 90
ZONES = ("DE", "NL")
TIMEZONE = "Europe/Berlin"  # Same physical civil-day boundaries as Amsterdam.
QUANTILES = ("q10", "q50", "q90")
RULES = tuple(
    {"nyx_p50_min": float(level), "peak_gap_max": float(gap), "upside_min": float(upside)}
    for level, gap, upside in product((150, 200, 250), (25, 50), (0, 50))
)
MINIMUM_MAE_GAIN = 0.02
COUNTRY_MAE_TOLERANCE = 0.05
MINIMUM_SUPPORT_HOURS = 20
MINIMUM_SUPPORT_DAYS = 5
MINIMUM_COUNTRY_SUPPORT_HOURS = 5
MINIMUM_COUNTRY_SUPPORT_DAYS = 2
NUMERICAL_TOLERANCE = 1e-12
PEAK_GAP_ATOL = 1e-8


def _origin(value) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("A finite model origin day is required")
    return timestamp.date()


def _validate_values(frame: pd.DataFrame, *, actual_required: bool) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("A nonempty frame with unique columns is required")
    if not actual_required and "actual" in frame.columns:
        raise ValueError("Current prediction inputs must not contain actual labels")
    required = {"zone", "own_joint_deficit", "own_residual_stress", "nyx_daily_peak_gap"}
    numeric = [f"{model}__{quantile}" for model in ("nyx", "test2") for quantile in QUANTILES]
    numeric += ["own_joint_deficit", "own_residual_stress", "nyx_daily_peak_gap"]
    if actual_required:
        required.update(("actual", "fit_origin"))
        numeric.append("actual")
    required.update(numeric)
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required hybrid fields: {sorted(required.difference(frame.columns))}")
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC" or index.hasnans
            or not index.equals(index.floor("h"))):
        raise ValueError("Hourly UTC DatetimeIndex required, including physical DST hours")
    zones = frame["zone"].to_numpy()
    if set(zones) != set(ZONES):
        raise ValueError("Exactly DE and NL must be represented")
    keys = pd.MultiIndex.from_arrays([index, zones], names=["timestamp_utc", "zone"])
    if not keys.is_unique:
        raise ValueError("Duplicate (timestamp_utc,zone) identity")
    values = frame[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All forecast, physical and required observed values must be finite; no imputation")
    for model in ("nyx", "test2"):
        quantiles = frame[[f"{model}__{q}" for q in QUANTILES]].to_numpy(dtype=float)
        if np.any(np.diff(quantiles, axis=1) < 0):
            raise ValueError(f"{model} quantiles must already be nondecreasing; no silent repair")
    deficit = frame["own_joint_deficit"].to_numpy(dtype=float)
    if not np.all((deficit >= 0) & (deficit <= 1)):
        raise ValueError("own_joint_deficit must lie in [0,1]")
    days = np.asarray(index.tz_convert(TIMEZONE).date)
    if actual_required:
        fit_origins = pd.DatetimeIndex(pd.to_datetime(frame["fit_origin"]))
        if fit_origins.hasnans or np.any(np.asarray(fit_origins.date) > days):
            raise ValueError("Saved Test2 model origin cannot follow its target delivery day")
    return days, zones


def _validate_complete_panel(frame: pd.DataFrame, first: date, end_exclusive: date) -> None:
    """Require exact physical-hour panels and verify peaks from NYX alone."""
    left = pd.Timestamp(first).tz_localize(TIMEZONE)
    right = pd.Timestamp(end_exclusive).tz_localize(TIMEZONE)
    expected = pd.date_range(left, right, freq="h", inclusive="left").tz_convert("UTC")
    for zone in ZONES:
        country = frame.loc[frame["zone"].to_numpy() == zone]
        if not country.index.equals(expected):
            raise ValueError(f"{zone}: exact ordered complete civil-day hourly grid required")
        days = country.index.tz_convert(TIMEZONE).date
        p50 = country["nyx__q50"].to_numpy(dtype=float)
        daily_max = pd.Series(p50).groupby(days, sort=False).transform("max").to_numpy()
        calculated_gap = daily_max - p50
        provided_gap = country["nyx_daily_peak_gap"].to_numpy(dtype=float)
        if not np.allclose(provided_gap, calculated_gap, atol=PEAK_GAP_ATOL, rtol=0):
            raise ValueError(f"{zone}: daily peak gap must use only the same-day complete NYX P50 profile")


def _validate_rule(rule: dict) -> dict:
    if not isinstance(rule, dict) or set(rule) != {"nyx_p50_min", "peak_gap_max", "upside_min"}:
        raise ValueError("Policy rule must have exactly the three preregistered parameters")
    if not any(rule == allowed for allowed in RULES):
        raise ValueError("Policy rule is outside the fixed 12-rule grid")
    return rule


def _mask(frame: pd.DataFrame, rule: dict) -> np.ndarray:
    """No Test2 prediction or observed value is read when making a decision."""
    physical = ((frame["own_joint_deficit"].to_numpy(dtype=float) >= .5)
                | (frame["own_residual_stress"].to_numpy(dtype=float) >= 1.0))
    p50 = frame["nyx__q50"].to_numpy(dtype=float)
    upside = frame["nyx__q90"].to_numpy(dtype=float) - p50
    return (physical & (p50 >= rule["nyx_p50_min"])
            & (frame["nyx_daily_peak_gap"].to_numpy(dtype=float) <= rule["peak_gap_max"])
            & (upside >= rule["upside_min"]))


def _support(mask: np.ndarray, days: np.ndarray, zones: np.ndarray) -> dict:
    return {
        "hours": int(mask.sum()), "days": len(set(days[mask])),
        "by_country": {
            zone: {"hours": int(np.count_nonzero(mask & (zones == zone))),
                   "days": len(set(days[mask & (zones == zone)]))}
            for zone in ZONES
        },
    }


def _metrics(actual: np.ndarray, prediction: np.ndarray, zones: np.ndarray) -> dict:
    result = {}
    for name in ("pooled", *ZONES):
        selected = np.ones(len(actual), dtype=bool) if name == "pooled" else zones == name
        errors = prediction[selected] - actual[selected]
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        if not np.isfinite([mae, rmse]).all():
            raise ValueError("Nonfinite score; numerical values are not safe to evaluate")
        result[name] = {"rows": int(selected.sum()), "mae": mae, "rmse": rmse}
    return result


def select_rule(past_frame: pd.DataFrame, origin_day: str | date) -> dict:
    """Select one shared rule on preceding 90 complete days, or stay with NYX.

    The frame is flat, UTC-indexed, with zone, both sets of q10/q50/q90,
    own_joint_deficit, own_residual_stress, nyx_daily_peak_gap, actual and
    fit_origin. Older rows may be present, but origin/future labels are rejected
    before filtering. Missing 90-day coverage is an error, not a baseline
    fallback that would conceal missing input data. The baseline fallback is
    used only if none of the 12 rules passes every performance/support guard.
    """
    origin = _origin(origin_day)
    all_days, _ = _validate_values(past_frame, actual_required=True)
    if np.any(all_days >= origin):
        raise ValueError("Origin/future targets and labels are forbidden in rule selection")
    first = origin - timedelta(days=WINDOW_DAYS)
    selected_window = all_days >= first
    frame = past_frame.loc[selected_window].copy()
    if frame.empty:
        raise ValueError("The preceding exact 90-day DE/NL panel is required")
    _validate_complete_panel(frame, first, origin)
    days = np.asarray(frame.index.tz_convert(TIMEZONE).date)
    zones = frame["zone"].to_numpy()
    actual = frame["actual"].to_numpy(dtype=float)
    baseline = frame["nyx__q50"].to_numpy(dtype=float)
    test2 = frame["test2__q50"].to_numpy(dtype=float)
    base_metrics = _metrics(actual, baseline, zones)
    candidates = []
    best = None
    for rule_index, rule in enumerate(RULES):
        choose = _mask(frame, rule)
        support = _support(choose, days, zones)
        prediction = np.where(choose, test2, baseline)
        metrics = _metrics(actual, prediction, zones)
        gain = base_metrics["pooled"]["mae"] - metrics["pooled"]["mae"]
        failures = []
        if support["hours"] < MINIMUM_SUPPORT_HOURS:
            failures.append("pooled_support_hours_below_20")
        if support["days"] < MINIMUM_SUPPORT_DAYS:
            failures.append("pooled_support_days_below_5")
        for zone in ZONES:
            if support["by_country"][zone]["hours"] < MINIMUM_COUNTRY_SUPPORT_HOURS:
                failures.append(f"{zone}_support_hours_below_5")
            if support["by_country"][zone]["days"] < MINIMUM_COUNTRY_SUPPORT_DAYS:
                failures.append(f"{zone}_support_days_below_2")
        if gain < MINIMUM_MAE_GAIN - NUMERICAL_TOLERANCE:
            failures.append("pooled_mae_gain_below_0.02")
        for zone in ZONES:
            if metrics[zone]["mae"] > base_metrics[zone]["mae"] + COUNTRY_MAE_TOLERANCE + NUMERICAL_TOLERANCE:
                failures.append(f"{zone}_mae_degradation_above_0.05")
            if metrics[zone]["rmse"] > base_metrics[zone]["rmse"] + NUMERICAL_TOLERANCE:
                failures.append(f"{zone}_rmse_worse_than_nyx")
        candidate = {"rule_index": rule_index, "rule": dict(rule), "support": support,
                     "metrics": metrics, "pooled_mae_gain": float(gain),
                     "eligible": not failures, "failed_guards": failures}
        candidates.append(candidate)
        if candidate["eligible"] and (best is None or metrics["pooled"]["mae"]
                                      < best["metrics"]["pooled"]["mae"] - NUMERICAL_TOLERANCE):
            best = candidate
    return {
        "protocol_version": PROTOCOL_VERSION, "origin_day": origin.isoformat(),
        "mode": "hybrid" if best is not None else "nyx",
        "rule": dict(best["rule"]) if best is not None else None,
        "selected_rule_index": best["rule_index"] if best is not None else None,
        "selection_reason": "best_eligible_pooled_mae" if best is not None else "no_rule_passed_all_guards",
        "window_days": WINDOW_DAYS, "window_start": first.isoformat(),
        "window_end_exclusive": origin.isoformat(), "rows": len(frame),
        "older_rows_ignored": int((~selected_window).sum()),
        "support": best["support"] if best is not None else _support(np.zeros(len(frame), dtype=bool), days, zones),
        "support_unit": "country-hours and distinct civil delivery days; shared rule for DE/NL",
        "baseline_metrics": base_metrics,
        "selected_metrics": best["metrics"] if best is not None else base_metrics,
        "pooled_mae_gain": float(best["pooled_mae_gain"]) if best is not None else 0.0,
        "candidate_scores": candidates,
        "tie_break": "fixed grid order: p50 150/200/250, gap 25/50, upside 0/50",
        "numerical_guard_tolerance": NUMERICAL_TOLERANCE,
        "physical_condition": "own_joint_deficit>=0.5 OR own_residual_stress>=1.0",
        "current_decision_sources": "NYX quantiles, same-day NYX P50 peak, frozen physical forecasts only",
        "quantile_combination": "copy entire NYX or Test2 quantile triplet; no averaging or clipping",
        "pit_publication_evidence_verified": False,
        "out_of_time_provenance_verified_by_module": False,
        "untouched_validation_period": False,
    }


def apply_rule(test_frame: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Copy whole forecast triplets using the already-selected, label-free rule.

    Current test data must contain complete DE/NL day profiles for one through
    seven consecutive days, starting at policy.origin_day. A column named actual
    is forbidden even if it would not be read. Test2 values are validated and
    copied if selected, but do not enter the selector.
    """
    if (not isinstance(policy, dict) or policy.get("protocol_version") != PROTOCOL_VERSION
            or policy.get("mode") not in ("nyx", "hybrid")):
        raise ValueError("A policy returned by this preregistered protocol is required")
    origin = _origin(policy.get("origin_day"))
    days, _ = _validate_values(test_frame, actual_required=False)
    first, last = min(days), max(days)
    if first != origin or last >= origin + timedelta(days=7):
        raise ValueError("Current targets must cover 1-7 full days starting at the policy origin")
    _validate_complete_panel(test_frame, first, last + timedelta(days=1))
    if policy["mode"] == "nyx":
        if policy.get("rule") is not None:
            raise ValueError("NYX-only policy cannot contain an active hybrid rule")
        choose = np.zeros(len(test_frame), dtype=bool)
        reason = np.full(len(test_frame), "nyx_fallback_no_eligible_past_rule", dtype=object)
    else:
        rule = _validate_rule(policy.get("rule"))
        choose = _mask(test_frame, rule)
        reason = np.where(choose, "test2_selected_by_nyx_and_physical_rule", "nyx_rule_conditions_not_met")
    output = test_frame.copy(deep=True)
    output["selected_test2"] = choose
    output["selected_model"] = np.where(choose, "Test2", "NYX")
    output["reason"] = reason
    for quantile in QUANTILES:
        output[f"hybrid__{quantile}"] = np.where(
            choose, test_frame[f"test2__{quantile}"].to_numpy(), test_frame[f"nyx__{quantile}"].to_numpy(),
        )
    return output


__all__ = ["PROTOCOL_VERSION", "RULES", "WINDOW_DAYS", "select_rule", "apply_rule"]
