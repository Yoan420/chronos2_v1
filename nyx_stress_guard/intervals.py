"""Causal expansion-only interval calibration, isolated from NYX production.

Scores are computed from historically issued *precalibration* bounds, never
from this calibrator's already widened bounds. Lower/upper signed scores are
q10-y and y-q90, respectively; each side uses the order statistic at
ceil((n+1)*0.9). Expansions are nonnegative and P50 is never modified.

The hierarchy is explicit: country x intervention state first, then a pool of
countries with the SAME intervention state, then unchanged bounds. This is a
chronological, group-aware calibration diagnostic inspired by conformalized
quantile regression, not an exchangeability or conditional-coverage guarantee.
The rolling forecaster and dependent electricity-price hours need prospective
coverage checks. No adaptive-alpha or post-evaluation parameter tuning is used.
"""
from __future__ import annotations

from fractions import Fraction
import numpy as np
import pandas as pd


DEFAULT_SETTINGS = {
    "window_days": 365, "tail_alpha": .1,
    "active_local_min_rows": 40, "active_local_min_days": 10,
    "inactive_local_min_rows": 120, "inactive_local_min_days": 28,
    "pooled_min_rows": 120, "pooled_min_days": 28,
    "recent_diagnostic_days": 28, "drift_min_rows": 20,
    "drift_miss_rate_change": .1,
}
TIMEZONE = "Europe/Paris"
KEYS = ("zone", "timestamp_utc", "forecast_origin_utc")
BASE_BOUNDS = ("precalibration_q10", "precalibration_q90")
SOURCE = "https://arxiv.org/abs/1905.03222"


class IntervalCalibrationError(ValueError):
    """A causal/identity/interval contract was violated."""


def _settings(settings):
    if settings is not None and (not isinstance(settings, dict) or set(settings)-set(DEFAULT_SETTINGS)):
        raise IntervalCalibrationError("Unknown interval calibration settings.")
    result = {**DEFAULT_SETTINGS, **(settings or {})}
    for name, value in result.items():
        if name in ("tail_alpha", "drift_miss_rate_change"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value < 1:
                raise IntervalCalibrationError(f"{name}: a finite probability is required.")
        elif type(value) is not int or value <= 0:
            raise IntervalCalibrationError(f"{name}: a positive integer is required.")
    if result["window_days"] != 365 or result["tail_alpha"] != .1:
        raise IntervalCalibrationError("This P10/P90 protocol requires a 365-day window and tail_alpha=0.1.")
    if result["recent_diagnostic_days"] > result["window_days"]:
        raise IntervalCalibrationError("The drift window must lie inside the calibration window.")
    return result


def _utc(values, *, name, missing=False):
    source = pd.Series(values).copy()
    if isinstance(source.dtype, pd.DatetimeTZDtype):
        result = source.dt.tz_convert("UTC")
    else:
        known = source[source.notna()]
        if any(pd.Timestamp(value).tzinfo is None for value in known):
            raise IntervalCalibrationError(f"{name}: explicit timezones are required.")
        result = pd.to_datetime(source, utc=True, errors="raise", format="mixed")
    if not missing and result.isna().any():
        raise IntervalCalibrationError(f"{name}: missing timestamps are forbidden.")
    return result


def _cutoff(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise IntervalCalibrationError("A timezone-aware forecast cutoff is required.")
    civil = stamp.tz_convert(TIMEZONE)
    wall = civil.tz_localize(None)
    if wall != wall.normalize()+pd.Timedelta(hours=8):
        raise IntervalCalibrationError("The calibration cutoff must equal civil D-1 08:00.")
    return stamp.tz_convert("UTC"), (civil.tz_localize(None).normalize()+pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _boolean(values, name):
    values = pd.Series(values)
    if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise IntervalCalibrationError(f"{name}: explicit nonmissing boolean values are required.")
    return values.to_numpy(bool)


def _prepare(frame, *, fit):
    required = {*KEYS, "candidate_forecast", "candidate_q10", "candidate_q90", "intervention_active"}
    if fit:
        required |= {"actual", "label_available_at_utc"}
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates or required-set(frame.columns):
        raise IntervalCalibrationError(f"Missing or duplicate interval columns: {sorted(required-set(frame.columns)) if isinstance(frame, pd.DataFrame) else 'not a dataframe'}.")
    data = frame.copy(deep=True).reset_index(drop=True)
    if not data.zone.map(lambda z: isinstance(z, str) and bool(z.strip())).all():
        raise IntervalCalibrationError("An explicit country is required on every row.")
    data["timestamp_utc"] = _utc(data.timestamp_utc, name="timestamp_utc")
    data["forecast_origin_utc"] = _utc(data.forecast_origin_utc, name="forecast_origin_utc")
    if data.duplicated(["zone", "timestamp_utc"]).any():
        raise IntervalCalibrationError("Duplicate zone/physical-hour identities are forbidden.")
    if not data.timestamp_utc.eq(data.timestamp_utc.dt.floor("h")).all():
        raise IntervalCalibrationError("Physical hourly delivery timestamps are required.")
    civil = data.timestamp_utc.dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
    expected = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).dt.tz_localize(TIMEZONE).dt.tz_convert("UTC")
    if not data.forecast_origin_utc.eq(expected).all():
        raise IntervalCalibrationError("Every forecast origin must equal D-1 08 h, including DST.")
    data["_delivery_day"] = civil.dt.strftime("%Y-%m-%d")
    data["intervention_active"] = _boolean(data.intervention_active, "intervention_active")
    present = [name in data for name in BASE_BOUNDS]
    if any(present) and not all(present):
        raise IntervalCalibrationError("Both precalibration bounds must be present together.")
    if not any(present):
        if "interval_calibration_status" in data:
            raise IntervalCalibrationError("Already calibrated intervals require their original precalibration bounds.")
        for target, source in zip(BASE_BOUNDS, ("candidate_q10", "candidate_q90")):
            data[target] = data[source].copy()
    numeric = ["candidate_forecast", "candidate_q10", "candidate_q90", *BASE_BOUNDS]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(float)
    if not np.isfinite(data[numeric].to_numpy()).all():
        raise IntervalCalibrationError("Finite P50 and interval bounds are required.")
    center = data.candidate_forecast
    if (not (data.candidate_q10.le(center)&center.le(data.candidate_q90)).all()
            or not (data.precalibration_q10.le(center)&center.le(data.precalibration_q90)).all()):
        raise IntervalCalibrationError("Both emitted and precalibration intervals must contain the unchanged P50.")
    if fit:
        data["actual"] = pd.to_numeric(data.actual, errors="raise").astype(float)
        if np.isinf(data.actual).any():
            raise IntervalCalibrationError("Infinite labels are forbidden.")
        data["label_available_at_utc"] = _utc(data.label_available_at_utc, name="label_available_at_utc", missing=True)
        known = data.actual.notna()
        if (known & (data.label_available_at_utc.isna() | data.label_available_at_utc.le(data.forecast_origin_utc))).any():
            raise IntervalCalibrationError("Known labels require publication strictly after their own forecast origin.")
        eligible = known.to_numpy()
        for name in ("label_eligible", "forecast_eligible", "interval_calibration_eligible"):
            if name in data:
                eligible &= _boolean(data[name], name)
        data["_label_eligible"] = eligible
        data["_lower_score"] = data.precalibration_q10-data.actual
        data["_upper_score"] = data.actual-data.precalibration_q90
    return data


def finite_sample_quantile(scores, tail_alpha=.1):
    """Left order statistic with an (n+1) correction; None means rank>n.

    Returning a finite observed maximum when the required rank is n+1 would
    silently weaken the correction. The caller instead records a fallback.
    """
    values = np.asarray(scores, float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or isinstance(tail_alpha, bool) or not np.isfinite(tail_alpha) or not 0 < tail_alpha < 1):
        raise IntervalCalibrationError("Finite nonempty scores and a valid one-sided alpha are required.")
    level = 1-Fraction(str(tail_alpha))
    numerator = (len(values)+1)*level.numerator
    rank = (numerator+level.denominator-1)//level.denominator
    if rank > len(values):
        return None, int(rank)
    return float(np.partition(values, rank-1)[rank-1]), int(rank)


def _iso(value):
    return None if pd.isna(value) else pd.Timestamp(value).isoformat()


def _drift(sample, target_day, p):
    boundary = (pd.Timestamp(target_day)-pd.Timedelta(days=p["recent_diagnostic_days"])).strftime("%Y-%m-%d")
    recent, old = sample.loc[sample._delivery_day.ge(boundary)], sample.loc[sample._delivery_day.lt(boundary)]
    enough = min(len(recent), len(old)) >= p["drift_min_rows"]
    rates = {f"{period}_{side}_miss_rate": float(group[f"_{side}_score"].gt(0).mean()) if len(group) else None
        for period, group in (("recent", recent), ("older", old)) for side in ("lower", "upper")}
    shifts = {f"{side}_miss_rate_shift_flag": bool(abs(rates[f"recent_{side}_miss_rate"]-rates[f"older_{side}_miss_rate"])
        > p["drift_miss_rate_change"]) if enough else None for side in ("lower", "upper")}
    return {"recent_rows": len(recent), "older_rows": len(old), **rates, **shifts,
        "drift_status": "descriptive_recent_vs_older" if enough else "insufficient_recent_or_older_rows"}


def _group_record(eligible, zone, active, target_day, p):
    pool = eligible.loc[eligible.intervention_active.eq(active)]
    local = pool.loc[pool.zone.eq(zone)]
    prefix = "active" if active else "inactive"
    local_days, pooled_days = int(local._delivery_day.nunique()), int(pool._delivery_day.nunique())
    local_ready = len(local) >= p[f"{prefix}_local_min_rows"] and local_days >= p[f"{prefix}_local_min_days"]
    pooled_ready = len(pool) >= p["pooled_min_rows"] and pooled_days >= p["pooled_min_days"]
    source = "country_same_intervention_state" if local_ready else "pooled_same_intervention_state" if pooled_ready else "none"
    sample = local if local_ready else pool
    record = {"zone": zone, "intervention_active": bool(active), "source": source,
        "local_rows": len(local), "local_days": local_days, "pooled_same_state_rows": len(pool),
        "pooled_same_state_days": pooled_days, "rows": len(sample), "days": int(sample._delivery_day.nunique()),
        "source_zones": sorted(sample.zone.unique().tolist()),
        "max_label_available_at_utc": _iso(sample.label_available_at_utc.max()),
        "first_delivery_day": str(sample._delivery_day.min()) if len(sample) else None,
        "last_delivery_day": str(sample._delivery_day.max()) if len(sample) else None,
        "lower_expansion_eur_mwh": 0., "upper_expansion_eur_mwh": 0.,
        "lower_score_quantile": None, "upper_score_quantile": None, "order_statistic_rank": None,
        "status": "insufficient_history_same_state" if source == "none" else "calibrated"}
    record.update(_drift(sample, target_day, p))
    if source != "none":
        lower, rank = finite_sample_quantile(sample._lower_score, p["tail_alpha"])
        upper, _ = finite_sample_quantile(sample._upper_score, p["tail_alpha"])
        record.update(order_statistic_rank=rank, lower_score_quantile=lower, upper_score_quantile=upper)
        if lower is None or upper is None:
            record["status"] = "finite_sample_rank_unavailable"
        else:
            record.update(lower_expansion_eur_mwh=max(0., lower), upper_expansion_eur_mwh=max(0., upper))
    return record


def _fit_prepared(data, cutoff, zones, p):
    cutoff, target_day = _cutoff(cutoff)
    first = (pd.Timestamp(target_day)-pd.Timedelta(days=p["window_days"])).strftime("%Y-%m-%d")
    valid = (data._delivery_day.ge(first)&data._delivery_day.lt(target_day)
        & data.forecast_origin_utc.lt(cutoff)&data._label_eligible
        & data.label_available_at_utc.notna()&data.label_available_at_utc.le(cutoff))
    eligible = data.loc[valid]
    return {"schema_version": 1, "fit_cutoff_utc": cutoff.isoformat(), "delivery_day": target_day,
        "window_start_day": first, "settings": dict(p), "zones": list(zones),
        "eligible_history_rows": len(eligible), "max_eligible_label_available_at_utc": _iso(eligible.label_available_at_utc.max()),
        "groups": {zone: {str(active).lower(): _group_record(eligible, zone, active, target_day, p)
                          for active in (False, True)} for zone in zones},
        "score_bounds": list(BASE_BOUNDS), "expansion_only": True, "p50_modified": False,
        "active_inactive_rows_ever_pooled_together": False, "conditional_coverage_guaranteed": False,
        "exchangeability_assumed_as_verified": False}


def fit_interval_state(history, forecast_origin_utc, *, zones=None, settings=None):
    """Fit a JSON-serializable state for exactly one D-1 08 h cutoff.

    Supplied history may include later rows; only already issued, published
    labels in the preceding 365 delivery days contribute. The state belongs to
    one model/decision policy; callers must not concatenate different models.
    """
    data, p = _prepare(history, fit=True), _settings(settings)
    zones = sorted(data.zone.unique()) if zones is None else list(zones)
    if not zones or len(set(zones)) != len(zones) or any(not isinstance(zone, str) or not zone.strip() for zone in zones):
        raise IntervalCalibrationError("Explicit unique target countries are required, including for empty history.")
    return _fit_prepared(data, forecast_origin_utc, zones, p)


def apply_interval_state(state, current):
    """Expand source intervals without reading current labels; P50 stays exact.

    Reapplying a state is idempotent: the preserved precalibration bounds, not
    the already widened candidate bounds, remain the transformation source.
    A stale or future state is rejected rather than reused silently.
    """
    data = _prepare(current, fit=False)
    if (state.get("schema_version") != 1 or not data.zone.isin(state["zones"]).all()
            or not data.forecast_origin_utc.eq(pd.Timestamp(state["fit_cutoff_utc"])).all()):
        raise IntervalCalibrationError("Known countries and the exact fitted forecast cutoff are required.")
    _settings(state["settings"])
    output = current.copy(deep=True)
    for name in BASE_BOUNDS:
        if name not in output:
            output[name] = data[name].to_numpy()
    lower, upper = data.precalibration_q10.to_numpy().copy(), data.precalibration_q90.to_numpy().copy()
    records = [state["groups"][zone][str(bool(active)).lower()]
        for zone, active in zip(data.zone, data.intervention_active)]
    for i, record in enumerate(records):
        additions = np.array([record["lower_expansion_eur_mwh"], record["upper_expansion_eur_mwh"]], float)
        if not np.isfinite(additions).all() or (additions < 0).any():
            raise IntervalCalibrationError("Calibration expansions must be finite and nonnegative.")
        lower[i] -= additions[0]
        upper[i] += additions[1]
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise IntervalCalibrationError("Expanded intervals overflowed finite numeric support.")
    output["candidate_q10"], output["candidate_q90"] = lower, upper
    output["interval_calibration_cutoff_utc"] = pd.Timestamp(state["fit_cutoff_utc"])
    names = ("status", "source", "rows", "days", "local_rows", "local_days", "pooled_same_state_rows", "pooled_same_state_days",
        "max_label_available_at_utc", "lower_expansion_eur_mwh", "upper_expansion_eur_mwh", "order_statistic_rank",
        "recent_rows", "older_rows", "recent_lower_miss_rate", "recent_upper_miss_rate", "older_lower_miss_rate", "older_upper_miss_rate",
        "lower_miss_rate_shift_flag", "upper_miss_rate_shift_flag", "drift_status")
    for name in names:
        values = [record[name] for record in records]
        if name in ("status", "source", "drift_status"):
            column = pd.array(values, dtype="string")
        elif name == "max_label_available_at_utc":
            column = pd.to_datetime(values, utc=True)
        elif name.endswith("_flag"):
            column = pd.array(values, dtype="boolean")
        elif name in ("rows", "days", "local_rows", "local_days", "pooled_same_state_rows", "pooled_same_state_days",
                      "recent_rows", "older_rows", "order_statistic_rank"):
            column = pd.array(values, dtype="Int64")
        else:
            column = np.asarray(values, dtype=float)
        output["interval_calibration_"+name] = column
    output["interval_calibration_source_zones"] = pd.array([",".join(record["source_zones"]) for record in records], dtype="string")
    pd.testing.assert_series_equal(output.candidate_forecast, current.candidate_forecast, check_exact=True)
    return output


def calibrate_intervals(frame, *, settings=None):
    """Chronological retrospective replay returning calibrated rows and audit.

    The original row order/index and all non-interval input columns are kept.
    Each date uses only labels already available at that date's own cutoff.
    """
    data, p = _prepare(frame, fit=True), _settings(settings)
    if data.empty:
        raise IntervalCalibrationError("A historical calibration replay requires nonempty rows.")
    zones = sorted(data.zone.unique())
    pieces, positions, states = [], [], []
    for cutoff, indices in data.groupby("forecast_origin_utc", sort=True).groups.items():
        state = _fit_prepared(data, cutoff, zones, p)
        where = np.asarray(indices, dtype=int)
        pieces.append(apply_interval_state(state, frame.iloc[where]))
        positions.append(where)
        states.append(state)
    result = pd.concat(pieces).iloc[np.argsort(np.concatenate(positions))]
    pd.testing.assert_series_equal(result.candidate_forecast, frame.candidate_forecast, check_exact=True)
    audit = {"engine": "nyx_stress_guard_intervals_v1", "settings": p, "rows": len(result), "daily_fits": len(states),
        "statuses": result.interval_calibration_status.value_counts().to_dict(),
        "sources": result.interval_calibration_source.value_counts().to_dict(),
        "source_bounds": list(BASE_BOUNDS), "scores": ["precalibration_q10 - actual", "actual - precalibration_q90"],
        "rank": "ceil((n+1)*(1-tail_alpha)), no interpolation; unavailable rank => no expansion",
        "grouping": "country x active/inactive; explicit fallback to same-state cross-country pool only",
        "labels_known_by_each_historical_cutoff_only": True, "future_or_current_labels_used": False,
        "bounds_can_only_expand": True, "p50_modified": False, "active_inactive_rows_ever_pooled_together": False,
        "conditional_coverage_guaranteed": False, "finite_sample_coverage_guaranteed_for_this_time_series": False,
        "adaptive_alpha_updates": False, "drift_flags_change_predictions": False,
        "reference": SOURCE, "latest_state": states[-1],
        "limitations": ["Temporal dependence and drifting rolling models invalidate an automatic exchangeable-split coverage claim.",
            "Country/state pooling is explicitly reported; pooled coverage need not equal a country's conditional coverage.",
            "Warmup leaves source bounds unchanged; not calibrated does not mean calibrated at nominal 80%.",
            "Expansion can improve pathwise coverage but may widen intervals and worsen interval scores.",
            "Precalibration intervals must have been issued causally by the same model/decision policy.",
            "Drift flags are descriptive recent-versus-older exceedance changes, not significance tests."]}
    return result, audit


__all__ = ["IntervalCalibrationError", "DEFAULT_SETTINGS", "finite_sample_quantile", "fit_interval_state", "apply_interval_state", "calibrate_intervals"]
