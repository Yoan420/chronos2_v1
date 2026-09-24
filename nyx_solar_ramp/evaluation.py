"""Read-only, paired evaluation of the isolated solar-ramp research experiment.

Inputs must already be point-in-time predictions. This module never certifies
vintages, learns a spike threshold from test outcomes, or authorizes promotion.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

KEYS = ["zone", "timestamp_utc", "forecast_origin_utc"]
RISK_VARIANTS = ("control", "local", "regional", "ramps", "interactions")
PHASES = ("all", "selection", "final_diagnostic", "common_oos", "live_historical")


def _finite(values: pd.Series) -> np.ndarray:
    return np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _prepare(predictions: pd.DataFrame, business_spike: float) -> pd.DataFrame:
    required = set(KEYS + ["variant", "actual", "forecast", "evaluation_phase"])
    missing = required - set(predictions)
    if missing:
        raise ValueError("Missing prediction columns: " + ", ".join(sorted(missing)))
    if predictions.columns.duplicated().any():
        raise ValueError("Duplicate prediction column names")
    frame = predictions.copy(deep=True)
    for name in ("timestamp_utc", "forecast_origin_utc"):
        parsed = pd.to_datetime(frame[name], errors="raise")
        if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise ValueError(f"{name} must be timezone aware")
        frame[name] = parsed.dt.tz_convert("UTC")
        if frame[name].isna().any():
            raise ValueError(f"Missing {name}")
    if frame["zone"].isna().any() or frame["variant"].isna().any():
        raise ValueError("Missing zone or variant")
    if frame.duplicated(["variant"] + KEYS).any():
        raise ValueError("Duplicate variant/zone/timestamp/origin")
    if frame.duplicated(["variant", "zone", "timestamp_utc"]).any():
        raise ValueError("Multiple origins per hourly target are not supported")
    if (frame.timestamp_utc != frame.timestamp_utc.dt.floor("h")).any():
        raise ValueError("Targets must be physical hourly UTC timestamps")
    numeric = ["actual", "forecast", "q10", "q90", "candidate_forecast",
               "candidate_q10", "candidate_q90", "risk_probability",
               "alert_threshold", "applied_correction", "spike_label",
               "statistical_spike_label", "ramp_label"]
    for name in numeric:
        if name not in frame:
            original = name.removeprefix("candidate_")
            frame[name] = frame[original] if name.startswith("candidate_") and original in frame else np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
    for name in ("alert", "expert_ready"):
        if name not in frame:
            frame[name] = False
        frame[name] = frame[name].fillna(False).astype(bool)
    for name in ("risk_probability", "alert_threshold"):
        if (frame[name].notna() & ~frame[name].between(0, 1)).any():
            raise ValueError(f"{name} outside [0, 1]")
    expected_spike = (frame.actual >= business_spike).astype(float).where(frame.actual.notna())
    known = frame.spike_label.notna() & frame.actual.notna()
    if not np.allclose(frame.loc[known, "spike_label"], expected_spike[known], equal_nan=True):
        raise ValueError("spike_label disagrees with actual/business_spike")
    # An unknown actual is never a negative event, even if a supplied label says so.
    frame["spike_label"] = expected_spike
    for name in ("statistical_spike_label", "ramp_label"):
        frame.loc[frame.actual.isna(), name] = np.nan
        if (frame[name].notna() & ~frame[name].isin([0, 1])).any():
            raise ValueError(f"{name} must be binary or missing")
    if "baseline" not in set(frame.variant):
        raise ValueError("A baseline variant is required")
    base = frame[frame.variant.eq("baseline")].set_index(KEYS).sort_index()
    for variant, group in frame.groupby("variant", sort=True):
        aligned = group.set_index(KEYS).sort_index()
        if not aligned.index.equals(base.index):
            raise ValueError(f"{variant}: target keys differ from baseline; coverage comparison refused")
        if not np.allclose(aligned.actual, base.actual, equal_nan=True):
            raise ValueError(f"{variant}: actual values disagree with baseline")
        if not aligned.evaluation_phase.equals(base.evaluation_phase):
            raise ValueError(f"{variant}: evaluation_phase disagrees with baseline")
    frame["delivery_day"] = (frame.timestamp_utc.dt.tz_convert("Europe/Paris")
                             .dt.tz_localize(None).dt.normalize())
    return frame.sort_values(["variant", "zone", "timestamp_utc"]).reset_index(drop=True)


def _average_precision(y: np.ndarray, probability: np.ndarray) -> float | None:
    """Step-integrated PR-AUC (average precision), with tied scores grouped."""
    positives = float(y.sum())
    if not len(y) or positives == 0:
        return None
    order = np.argsort(-probability, kind="stable")
    truth, score = y[order], probability[order]
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score) - 1]
    tp = np.cumsum(truth)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) / positives * tp / (ends + 1)))


def _risk_metrics(frame: pd.DataFrame, label: str = "spike_label") -> dict[str, Any]:
    eligible = _finite(frame[label]) & _finite(frame.risk_probability)
    part = frame.loc[eligible]
    y = part[label].to_numpy(float)
    p = part.risk_probability.to_numpy(float)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    alert_eligible = eligible & _finite(frame.alert_threshold)
    alerts = frame.loc[alert_eligible, "alert"].to_numpy(bool)
    truth = frame.loc[alert_eligible, label].to_numpy(float).astype(bool)
    tp, fp = int(np.sum(alerts & truth)), int(np.sum(alerts & ~truth))
    positives, alerted = int(truth.sum()), int(alerts.sum())
    n_known = int(_finite(frame[label]).sum())
    return {
        "n_probability": len(part), "probability_coverage": len(part) / n_known if n_known else None,
        "n_probability_spikes": int(y.sum()),
        "pr_auc": _average_precision(y, p), "pr_auc_definition": "average_precision_step_integral",
        "brier": _mean((p - y) ** 2),
        "log_loss": _mean(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "n_alert_eligible": len(alerts), "n_alerts": alerted,
        "precision": tp / alerted if alerted else None,
        "recall": tp / positives if positives else None,
        "false_alerts_per_1000": 1000 * fp / len(alerts) if len(alerts) else None,
    }


def _episodes(frame: pd.DataFrame) -> dict[str, Any]:
    """Positive episodes use physical consecutive UTC hours, never local clock labels.

    Exact recall needs an alert on any episode hour. Tolerant recall additionally
    accepts the single hour immediately before its start (not an hour afterwards).
    Timing is first matched delivery hour minus episode onset, not publication lead.
    """
    total = covered = exact = tolerant = eligible_hours = 0
    timings: list[float] = []
    for _, group in frame.groupby("zone", sort=True):
        group = group.sort_values("timestamp_utc")
        known_alert = (_finite(group.risk_probability) & _finite(group.alert_threshold)
                       & _finite(group.actual))
        eligible_hours += int(known_alert.sum())
        alert_times = set(group.loc[known_alert & group.alert.to_numpy(bool), "timestamp_utc"])
        eligible_times = set(group.loc[known_alert, "timestamp_utc"])
        spikes = group.loc[group.spike_label.eq(1), "timestamp_utc"].tolist()
        runs: list[list[pd.Timestamp]] = []
        for stamp in spikes:
            if not runs or stamp - runs[-1][-1] != pd.Timedelta(hours=1):
                runs.append([stamp])
            else:
                runs[-1].append(stamp)
        for run in runs:
            total += 1
            covered += bool(set(run) & eligible_times)
            matches = sorted(set(run) & alert_times)
            exact += bool(matches)
            early = run[0] - pd.Timedelta(hours=1)
            accepted = ([early] if early in alert_times else []) + matches
            tolerant += bool(accepted)
            if accepted:
                timings.append((accepted[0] - run[0]).total_seconds() / 3600)
    return {"n_spike_episodes": total, "n_probability_covered_episodes": covered,
            "episode_recall_exact": exact / total if total and eligible_hours else None,
            "episode_recall_early_1h": tolerant / total if total and eligible_hours else None,
            "first_alert_timing_mean_hours": _mean(np.asarray(timings)),
            "timing_mae_hours": _mean(np.abs(np.asarray(timings))),
            "first_alert_timing_median_hours": float(np.median(timings)) if timings else None,
            "n_timed_episodes": len(timings)}


def _point_metrics(frame: pd.DataFrame, prediction: str) -> dict[str, Any]:
    valid = _finite(frame.actual) & _finite(frame[prediction])
    part = frame.loc[valid]
    error = part[prediction].to_numpy(float) - part.actual.to_numpy(float)
    spike = part.spike_label.eq(1).to_numpy()
    return {"n": len(part), "mae": _mean(np.abs(error)),
            "rmse": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
            "bias": _mean(error), "n_spikes": int(spike.sum()),
            "no_spike_mae": _mean(np.abs(error[~spike])),
            "spike_mae": _mean(np.abs(error[spike])),
            "underestimation": _mean(np.maximum(-error, 0)),
            "spike_underestimation": _mean(np.maximum(-error[spike], 0))}


def _interval_metrics(frame: pd.DataFrame, prefix: str = "") -> dict[str, Any]:
    lower, upper, median = prefix + "q10", prefix + "q90", prefix + "forecast"
    finite = _finite(frame.actual) & _finite(frame[lower]) & _finite(frame[upper])
    ordered = frame[lower].le(frame[upper]).to_numpy()
    part = frame.loc[finite & ordered]
    result: dict[str, Any] = {
        "n_intervals": len(part), "n_crossed_intervals": int((finite & ~ordered).sum()),
        "interval_coverage_80": _mean(part.actual.between(part[lower], part[upper]).to_numpy(float)),
        "interval_width_80": _mean((part[upper] - part[lower]).to_numpy(float)),
    }
    for name, quantile in ((lower, .1), (median, .5), (upper, .9)):
        mask = _finite(frame.actual) & _finite(frame[name])
        residual = (frame.loc[mask, "actual"] - frame.loc[mask, name]).to_numpy(float)
        result[f"pinball_{int(quantile * 100)}"] = _mean(
            np.maximum(quantile * residual, (quantile - 1) * residual))
    return result


def _reliability(frame: pd.DataFrame, bins: int) -> list[dict[str, Any]]:
    part = frame.loc[_finite(frame.actual) & _finite(frame.risk_probability)]
    assigned = np.minimum((part.risk_probability.to_numpy(float) * bins).astype(int), bins - 1)
    output = []
    for index in range(bins):
        members = part.iloc[np.flatnonzero(assigned == index)]
        output.append({"bin": index, "lower": index / bins, "upper": (index + 1) / bins,
                       "upper_inclusive": index == bins - 1, "n": len(members),
                       "mean_probability": _mean(members.risk_probability.to_numpy(float)),
                       "event_rate": _mean(members.spike_label.to_numpy(float))})
    return output


def _block_draws(days: pd.DatetimeIndex, samples: int, block_days: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Non-circular moving blocks; never join a missing-calendar-day boundary."""
    if not len(days) or not samples:
        return np.empty((0, len(days)), dtype=int)
    cuts = np.r_[0, np.flatnonzero(np.diff(days.asi8) != pd.Timedelta(days=1).value) + 1, len(days)]
    blocks = []
    for start, stop in zip(cuts[:-1], cuts[1:]):
        width = min(block_days, stop - start)
        blocks.extend(np.arange(i, i + width) for i in range(start, stop - width + 1))
    draws = np.empty((samples, len(days)), dtype=int)
    for sample in range(samples):
        chosen: list[int] = []
        while len(chosen) < len(days):
            chosen.extend(blocks[int(rng.integers(len(blocks)))].tolist())
        draws[sample] = chosen[:len(days)]
    return draws


def _paired(frame: pd.DataFrame, samples: int, block_days: int,
            seed: int) -> list[dict[str, Any]]:
    base = frame.loc[frame.variant.eq("baseline"), KEYS + ["actual", "forecast"]]
    days = pd.DatetimeIndex(sorted(frame.delivery_day.unique()))
    draws = _block_draws(days, samples, block_days, np.random.default_rng(seed))
    result = []
    for variant, group in frame.groupby("variant", sort=True):
        if variant == "baseline":
            continue
        paired = group.merge(base.rename(columns={"forecast": "baseline_forecast", "actual": "baseline_actual"}),
                             on=KEYS, how="inner", validate="one_to_one")
        for prediction in ("candidate_forecast",):
            valid = _finite(paired.actual) & _finite(paired[prediction]) & _finite(paired.baseline_forecast)
            good = paired.loc[valid].copy()
            good["delta"] = ((good[prediction] - good.actual).abs()
                             - (good.baseline_forecast - good.actual).abs())
            for zone in ["ALL"] + sorted(frame.zone.unique().tolist()):
                part = good if zone == "ALL" else good[good.zone.eq(zone)]
                eligible = paired if zone == "ALL" else paired[paired.zone.eq(zone)]
                reference_n = int((_finite(eligible.actual) & _finite(eligible.baseline_forecast)).sum())
                row: dict[str, Any] = {"variant": variant, "zone": zone, "prediction": prediction,
                    "baseline": "baseline", "n_target": len(eligible), "n": len(part),
                    "n_baseline_eligible": reference_n,
                    "paired_coverage": len(part) / reference_n if reference_n else None,
                    "n_days": part.delivery_day.nunique(), "bootstrap_samples": samples,
                    "block_days": block_days, "seed": seed,
                    "bootstrap_status": "computed" if len(days) > block_days and samples else "insufficient_day_blocks_or_disabled",
                    "complete_point_pairing": bool(reference_n and len(part) == reference_n)}
                for metric, selected in (("mae", part), ("spike_mae", part[part.spike_label.eq(1)])):
                    daily = selected.groupby("delivery_day").delta.agg(["sum", "count"]).reindex(days, fill_value=0)
                    sums, counts = daily["sum"].to_numpy(float), daily["count"].to_numpy(float)
                    denominator = counts[draws].sum(axis=1) if len(draws) else np.array([])
                    numerator = sums[draws].sum(axis=1) if len(draws) else np.array([])
                    usable = denominator > 0
                    deltas = numerator[usable] / denominator[usable]
                    # One day is not a meaningful confidence-interval sample.
                    ci = (np.quantile(deltas, [.025, .975]) if len(deltas) and len(days) > block_days
                          and selected.delivery_day.nunique() >= 2 else [None, None])
                    row.update({f"delta_{metric}": _mean(selected.delta.to_numpy(float)),
                                f"delta_{metric}_ci95_low": ci[0], f"delta_{metric}_ci95_high": ci[1],
                                f"{metric}_n_days": selected.delivery_day.nunique(),
                                f"{metric}_bootstrap_valid_samples": len(deltas)})
                result.append(row)
    return result


def evaluate(predictions: pd.DataFrame, config: dict) -> dict[str, Any]:
    """Return JSON-safe descriptive metrics; never fit or promote a model.

    Common OOS is the intersection of finite probabilities for all five risk
    variants and finite actual/point forecasts for every variant. It excludes
    warmup. Native-coverage scores are descriptive, not an improvement claim.
    """
    seed = int(config.get("seed", 1729))
    samples = int(config.get("bootstrap_samples", 500))
    block_days = int(config.get("block_days", 7))
    bins = int(config.get("reliability_bins", 10))
    if samples < 0 or block_days < 1 or bins < 1:
        raise ValueError("bootstrap_samples must be nonnegative; block_days/bins positive")
    if predictions.empty:
        return {"status": "insufficient_data", "scores": [], "paired_bootstrap": [],
                "reliability_bins": [], "selection": {"empirical_best_candidate": None},
                "decision": {"retain_baseline": True, "promotion_allowed": False, "production_pit_verified": False}}
    frame = _prepare(predictions, float(config.get("business_spike", 300)))
    risk_variants = tuple(config.get("risk_variants", RISK_VARIANTS))
    risk_sets = []
    for variant in risk_variants:
        group = frame[frame.variant.eq(variant)]
        valid = (_finite(group.risk_probability) & _finite(group.actual)
                 & ~group.evaluation_phase.isin(["warmup", "live_historical"]).to_numpy())
        risk_sets.append(set(map(tuple, group.loc[valid, KEYS].itertuples(index=False, name=None))))
    common = set.intersection(*risk_sets) if risk_sets else set()
    for _, group in frame.groupby("variant"):
        valid = _finite(group.forecast) & _finite(group.candidate_forecast) & _finite(group.actual)
        common &= set(group.loc[valid, KEYS].itertuples(index=False, name=None))
    common_mask = np.array([key in common for key in frame[KEYS].itertuples(index=False, name=None)])
    scores, bootstrap, reliability = [], [], []
    for phase in PHASES:
        part = (frame.loc[~frame.evaluation_phase.eq("live_historical")] if phase == "all"
                else frame.loc[common_mask] if phase == "common_oos" else frame[frame.evaluation_phase.eq(phase)])
        if part.empty:
            continue
        for (variant, zone), group in list(part.groupby(["variant", "zone"], sort=True)) + [
                ((variant, "ALL"), group) for variant, group in part.groupby("variant", sort=True)]:
            identity = {"variant": variant, "zone": zone, "phase": phase}
            row = {**identity, "n_target": len(group), "n_actual": int(_finite(group.actual).sum()),
                   **_point_metrics(group, "candidate_forecast"), **_risk_metrics(group),
                   **_episodes(group), **_interval_metrics(group, "candidate_")}
            row.update({"candidate_" + key: value for key, value in _point_metrics(group, "candidate_forecast").items()})
            row.update({"candidate_" + key: value for key, value in _interval_metrics(group, "candidate_").items()})
            row.update({"baseline_" + key: value for key, value in _point_metrics(group, "forecast").items()})
            row.update({"non_spike_mae": row["no_spike_mae"], "episode_recall": row["episode_recall_exact"],
                        "coverage80": row["interval_coverage_80"]})
            row["point_coverage"] = row["n"] / row["n_actual"] if row["n_actual"] else None
            row["interval_availability"] = row["n_intervals"] / row["n_actual"] if row["n_actual"] else None
            for label in ("statistical_spike_label", "ramp_label"):
                event = group.loc[group[label].eq(1)]
                event_scores = _point_metrics(event, "candidate_forecast")
                row[label + "_regime"] = {
                    "n_label_known": int(_finite(group[label]).sum()), "n_events": len(event),
                    "n_scored": event_scores["n"], "mae": event_scores["mae"],
                    "underestimation": event_scores["underestimation"],
                    "calibration_not_applicable": "Probability targets business spike, not this label."}
            scores.append(row)
            reliability.extend({**identity, **item} for item in _reliability(group, bins))
        bootstrap.extend({"phase": phase, **row} for row in _paired(part, samples, block_days, seed))
    selection_rows = [row for row in bootstrap if row["phase"] == "selection" and row["zone"] == "ALL"
                      and row["prediction"] == "candidate_forecast" and row["complete_point_pairing"]
                      and row["delta_mae"] is not None and row["variant"] != "governed"]
    best = min(selection_rows, key=lambda row: (row["delta_mae"], row["variant"])) if selection_rows else None
    return _safe({
        "status": "complete", "scores": scores, "paired_bootstrap": bootstrap,
        "reliability_bins": reliability, "risk_bins": reliability,
        "protocol": {"seed": seed, "bootstrap_samples": samples, "block_days": block_days,
            "business_spike": float(config.get("business_spike", 300)), "risk_variants": risk_variants,
            "common_oos_n_targets": len(common), "common_oos_available": bool(common),
            "common_oos_definition": "Non-warmup historical common finite probabilities in all risk variants; finite actual and candidate point forecast in every variant. Live historical excluded.",
            "scored_prediction": "candidate_forecast / candidate_q10 / candidate_q90; baseline_* denotes uncorrected forecast. Missing feature rows retain the saved fallback.",
            "all_phase_definition": "All annual historical phases, including warmup/fallback; separate live_historical delivery excluded.",
            "bootstrap_unit": "Consecutive Europe/Paris delivery days; every hourly observation and zone retained together; identical draws across comparisons and zones.",
            "bootstrap_caveat": "Intervals condition on selected models and historical sample; no correction for model search or structural breaks.",
            "episode_definition": "Consecutive positive-spike hours in physical UTC, separately by zone; unknown or missing hours break episodes.",
            "episode_timing_definition": "First alerted delivery hour minus onset; optional one-hour-early match only; not the issuance lead time.",
            "statistical_spike_definition": "Supplied train-only q99 label, never estimated by evaluator.",
            "ramp_definition": f"Supplied observed price increase >={float(config.get('price_ramp_threshold', 100)):g} EUR/MWh per physical hour; unavailable predecessor must remain missing.",
            "history_already_seen": True, "final_diagnostic_is_virgin_test": False,
            "coverage_policy": "No gain claim from native coverage. Use common_oos or complete point pairing; report all sample counts.",
            "production_pit_verified": False},
        "selection": {"phase": "selection", "final_used_for_selection": False,
            "empirical_best_candidate": best["variant"] if best else None,
            "empirical_best_candidate_delta_mae": best["delta_mae"] if best else None,
            "empirical_best_candidate_ci95_high": best["delta_mae_ci95_high"] if best else None,
            "status": "descriptive_only" if best else "insufficient_complete_selection_data"},
        "decision": {"retain_baseline": True, "promotion_allowed": False,
            "production_pit_verified": False, "selected_production_variant": "baseline",
            "reason": "Historical diagnostic already seen, PIT not certified, and no untouched prospective confirmation. Empirical ranking is not promotion."},
    })
