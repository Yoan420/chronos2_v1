"""Independent, read-only audit of a completed sealed StressGuard replay.

No fit, parameter selection, forecast regeneration, report mutation, or production
write occurs. The only optional output is a JSON file inside this tmp directory.
Day-block bootstrap choices are fixed before reading the current replay results:
10,000 replicates, seed 1729, whole civil days jointly across countries; a fixed
seven-day circular-block sensitivity is also reported, never selected by score.
Both intervals are exploratory, not independent post-selection inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FILES = {
    "hgb_v1": "hgb_predictions.parquet",
    "p50_previous": "previous_p50.parquet",
    "empirical_previous": "previous_empirical.parquet",
    "p50_calibrated": "p50_calibrated.parquet",
    "physics_direct": "physics_direct.parquet",
    "physics_governed": "physics_governed.parquet",
}
NEW = ("p50_calibrated", "physics_direct", "physics_governed")
EVENT_DAYS = ("2026-06-24", "2026-06-25", "2026-06-26", "2026-09-14")
KEYS = ["zone", "timestamp_utc", "forecast_origin_utc"]
BOOTSTRAP_SEED = 1729
BOOTSTRAP_REPLICATES = 10000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(directory, mapping):
    assert isinstance(mapping, dict) and mapping
    for name, expected in mapping.items():
        assert sha(directory / name) == expected, str(directory / name)


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def mean(values):
    return float(np.mean(values)) if len(values) else None


def scores(actual, point, lower=None, upper=None):
    if len(actual) == 0:
        return {"hours": 0}
    error = actual - point
    result = {"hours": len(actual), "mae": mean(np.abs(error)),
              "rmse": float(np.sqrt(np.mean(error ** 2))),
              "forecast_minus_actual_bias": mean(-error),
              "observed_le_p50": mean(actual <= point)}
    if lower is not None and upper is not None:
        width = upper - lower
        result.update(
            coverage_p10_p90=mean((lower <= actual) & (actual <= upper)),
            below_p10_rate=mean(actual < lower), above_p90_rate=mean(actual > upper),
            mean_interval_width=mean(width), median_interval_width=float(np.median(width)),
            interval_width_p95=float(np.quantile(width, .95)),
            interval_score80=mean(width + 10 * np.maximum(lower - actual, 0)
                                  + 10 * np.maximum(actual - upper, 0)),
            pinball10=mean(np.maximum(.1 * (actual - lower), -.9 * (actual - lower))),
            pinball90=mean(np.maximum(.9 * (actual - upper), -.1 * (actual - upper))))
    return result


def bootstrap_indices(n_days, block_days):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_blocks = (n_days + block_days - 1) // block_days
    starts = rng.integers(0, n_days, size=(BOOTSTRAP_REPLICATES, n_blocks))
    return ((starts[..., None] + np.arange(block_days)) % n_days).reshape(
        BOOTSTRAP_REPLICATES, -1)[:, :n_days]


def bootstrap_gain(days, gain, mask, all_days, resamples):
    daily = pd.DataFrame({"day": days[mask], "gain": gain[mask]}).groupby("day").gain.agg(["sum", "count"])
    daily = daily.reindex(all_days, fill_value=0)
    totals, counts = daily["sum"].to_numpy(float), daily["count"].to_numpy(float)
    out = {}
    for block, index in resamples.items():
        denominator = counts[index].sum(axis=1)
        values = totals[index].sum(axis=1) / denominator
        out[f"block_{block}_civil_days"] = {
            "percentile_ci95": np.quantile(values, [.025, .975]).tolist(),
            "bootstrap_fraction_gain_positive": mean(values > 0),
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "interpretation": "Exploratory paired resampling; fraction positive is not a posterior probability or a multiplicity-adjusted test."}
    return out


def validate_chronology(panel, models, folds):
    trained = folds.loc[folds.status.eq("trained")].copy()
    assert len(trained)
    cutoff = pd.to_datetime(trained.fit_cutoff_utc, utc=True)
    for column in ("max_label_available_at_utc", "cdf_max_label_available_at_utc"):
        assert pd.to_datetime(trained[column], utc=True).le(cutoff).all(), column
    assert trained.model_training_end_day.lt(trained.calibration_start_day).all()
    assert trained.calibration_start_day.lt(trained.fit_day).all()
    assert trained.cdf_uses_calibration_labels.eq(False).all()
    assert trained.reference_fit_scope.eq("core_features_only").all()
    assert trained.country_specific_gate.eq(False).all()
    assert trained.probability_gate.eq(.5).all()
    assert trained.training_days.le(365).all()
    for name in NEW:
        frame = models[name]
        quantiles = frame[["candidate_q10", "candidate_forecast", "candidate_q90"]].to_numpy(float)
        assert np.isfinite(quantiles).all() and (np.diff(quantiles, axis=1) >= 0).all()
        assert frame.candidate_q10.le(frame.q10 + 1e-9).all()
        assert frame.candidate_q90.ge(frame.q90 - 1e-9).all()
        assert frame.candidate_q10.le(frame.precalibration_q10 + 1e-9).all()
        assert frame.candidate_q90.ge(frame.precalibration_q90 - 1e-9).all()
        assert frame.interval_calibration_cutoff_utc.eq(frame.forecast_origin_utc).all()
        known = pd.to_datetime(frame.interval_calibration_max_label_available_at_utc, utc=True)
        assert (known.isna() | known.le(frame.forecast_origin_utc)).all()
        np.testing.assert_allclose(frame.applied_correction, frame.selected_weight * frame.bounded_correction, rtol=0, atol=1e-9)
        np.testing.assert_allclose(frame.candidate_forecast, frame.forecast + frame.applied_correction, rtol=0, atol=1e-9)
        assert frame.intervention_active.eq(frame.applied_correction.gt(0)).all()
    for name in ("physics_direct", "physics_governed"):
        frame = models[name]
        ready = frame.expert_ready.to_numpy(bool)
        probability = frame.loc[ready, "spike_probability"].to_numpy(float)
        threshold = frame.loc[ready, "threshold_eur_mwh"].to_numpy(float)
        quantiles = frame.loc[ready, ["mixture_error_q10", "mixture_error_q50", "mixture_error_q90"]].to_numpy(float)
        assert np.isfinite(quantiles).all() and (np.diff(quantiles, axis=1) >= 0).all()
        strong = probability > .5
        assert (quantiles[strong, 1] >= threshold[strong]).all()
        assert (quantiles[~strong, 1] < threshold[~strong]).all()
        assert not (frame.applied_correction.gt(0) & ~frame.spike_probability.gt(.5)).any()
    for column in ("candidate_forecast", "applied_correction", "selected_weight"):
        pd.testing.assert_series_equal(models["p50_calibrated"][column], models["p50_previous"][column], check_exact=True)
    return {"trained_folds": len(trained), "all_metadata_and_saved_quantile_checks": "passed",
            "core_rows_min": int(trained.cdf_core_rows.min()), "core_rows_max": int(trained.cdf_core_rows.max()),
            "first_trained_day": str(trained.fit_day.min()), "last_trained_day": str(trained.fit_day.max()),
            "full_365_day_training_folds": int(trained.full_365_day_training.sum()),
            "interval_only_ablation_point_and_governor_unchanged": True,
            "caveat": "Saved metadata and source-code review establish the causal contract; no fit was repeated in this audit."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.snapshot.resolve()
    manifest = read_json(directory / "manifest.json")
    result = read_json(directory / "results_manifest.json")
    assert result["status"] == "completed", "Do not audit a partial replay."
    assert result["suite_manifest_sha256"] == sha(directory / "manifest.json")
    verify(directory, manifest["input_files"])
    verify(directory, result["result_files"])
    panel = pd.read_parquet(directory / "panel.parquet")
    assert not panel.duplicated(KEYS).any()
    models = {name: pd.read_parquet(directory / filename) for name, filename in FILES.items()}
    for name, frame in models.items():
        assert not frame.duplicated(KEYS).any(), name
        pd.testing.assert_frame_equal(frame[panel.columns], panel, check_exact=True)
    local = panel.timestamp_utc.dt.tz_convert("Europe/Paris")
    days, hours = local.dt.strftime("%Y-%m-%d").to_numpy(), local.dt.hour.to_numpy()
    config = read_json(directory / "base_config.json")
    end = config["end_day"]
    start = (pd.Timestamp(end) - pd.Timedelta(days=config["evaluation_days"] - 1)).strftime("%Y-%m-%d")
    all_days = pd.date_range(start, end).strftime("%Y-%m-%d").to_numpy()
    actual = panel.actual.to_numpy(float)
    baseline, storm = panel.forecast.to_numpy(float), panel.benchmark_forecast.to_numpy(float)
    base_lower, base_upper = panel.q10.to_numpy(float), panel.q90.to_numpy(float)
    common = panel["sample"].eq("evaluation").to_numpy() & (days >= start) & (days <= end)
    common &= np.isfinite(np.column_stack([actual, baseline, storm, base_lower, base_upper])).all(axis=1)
    for frame in models.values():
        common &= np.isfinite(frame[["candidate_forecast", "candidate_q10", "candidate_q90"]].to_numpy(float)).all(axis=1)
    assert common.any()
    resamples = {block: bootstrap_indices(len(all_days), block) for block in (1, 7)}
    out = {"snapshot": str(directory), "suite_manifest_sha256": sha(directory / "manifest.json"),
           "result_manifest_sha256": sha(directory / "results_manifest.json"),
           "start": start, "end": end, "common_hours": int(common.sum()),
           "research_only": True, "production_modified": False,
           "primary_predeclared_for_this_replay": "physics_governed",
           "independent_year_validation": False,
           "tail_definition": "Ex post: actual price >= 99th percentile within each comparison group, on exactly the same paired hours. Not a forecastable label or trading rule.",
           "bootstrap_definition": "Paired MAE gain = |actual-NYX| - |actual-candidate|. Entire civil-day clusters keep all zones/hours together; positive means improvement. Percentile bootstrap, fixed1-day and7-day blocks; previously examined year, no post-selection significance claim.",
           "causality_and_coherence": validate_chronology(panel, models, pd.read_parquet(directory / "folds.parquet")),
           "groups": {}, "preidentified_19h_cases": [], "preidentified_day_aggregates": {}}
    group_masks = {"ALL": np.ones(len(panel), dtype=bool),
                   **{z: panel.zone.eq(z).to_numpy() for z in sorted(panel.zone.unique())}}
    for group, group_mask in group_masks.items():
        mask = common & group_mask
        top_threshold = float(np.quantile(actual[mask], .99))
        top = mask & (actual >= top_threshold)
        entry = {"hours": int(mask.sum()), "days": len(set(days[mask])),
                 "nyx": scores(actual[mask], baseline[mask], base_lower[mask], base_upper[mask]),
                 "storm": scores(actual[mask], storm[mask]),
                 "top1_threshold": top_threshold, "top1_hours": int(top.sum()),
                 "top1_nyx": scores(actual[top], baseline[top], base_lower[top], base_upper[top]),
                 "top1_storm": scores(actual[top], storm[top]), "models": {}}
        for name, frame in models.items():
            point = frame.candidate_forecast.to_numpy(float)
            lower, upper = frame.candidate_q10.to_numpy(float), frame.candidate_q90.to_numpy(float)
            gain = np.abs(actual - baseline) - np.abs(actual - point)
            active = mask & frame.applied_correction.gt(0).to_numpy()
            inactive = mask & ~frame.applied_correction.gt(0).to_numpy()
            off_cases = mask & ~np.isin(days, EVENT_DAYS)
            details = scores(actual[mask], point[mask], lower[mask], upper[mask])
            details.update(
                mae_gain_vs_nyx=mean(gain[mask]), mae_gain_vs_storm=mean(np.abs(actual[mask] - storm[mask]) - np.abs(actual[mask] - point[mask])),
                signed_total_absolute_error_gain=float(gain[mask].sum()),
                exploratory_paired_day_block_gain=bootstrap_gain(days, gain, mask, all_days, resamples),
                top1=scores(actual[top], point[top], lower[top], upper[top]), top1_gain_vs_nyx=mean(gain[top]),
                active_hours=int(active.sum()), improved_active_hours=int((active & (gain > 1e-9)).sum()),
                worsened_active_hours=int((active & (gain < -1e-9)).sum()), tied_active_hours=int((active & (np.abs(gain) <= 1e-9)).sum()),
                active_nyx_already_above_actual_hours=int((active & (baseline > actual)).sum()),
                active_nyx_equal_actual_hours=int((active & (baseline == actual)).sum()),
                active_outside_residual_error50_hours=int((active & ((actual - baseline) < 50)).sum()),
                positive_residual_error50_hours=int((mask & ((actual - baseline) >= 50)).sum()),
                active_within_residual_error50_hours=int((active & ((actual - baseline) >= 50)).sum()),
                active_mae_gain_vs_nyx=mean(gain[active]),
                active_scores=scores(actual[active], point[active], lower[active], upper[active]),
                nyx_on_same_active_scores=scores(actual[active], baseline[active], base_lower[active], base_upper[active]),
                inactive_scores=scores(actual[inactive], point[inactive], lower[inactive], upper[inactive]),
                mae_gain_excluding_preidentified_days=mean(gain[off_cases]))
            if "precalibration_q10" in frame:
                prelo, prehi = frame.precalibration_q10.to_numpy(float), frame.precalibration_q90.to_numpy(float)
                details["active_before_expansion_scores"] = scores(actual[active], point[active], prelo[active], prehi[active])
                for label, subset in (("all", mask), ("active", active), ("inactive", inactive)):
                    details[f"{label}_calibration_status_counts"] = frame.loc[subset, "interval_calibration_status"].value_counts().to_dict()
                    details[f"{label}_calibration_source_counts"] = frame.loc[subset, "interval_calibration_source"].value_counts().to_dict()
                details["coverage_lost_nyx_hours"] = int((mask & (base_lower <= actual) & (actual <= base_upper)
                    & ~((lower <= actual) & (actual <= upper))).sum())
                assert details["coverage_lost_nyx_hours"] == 0
            if name.startswith("physics"):
                details["ready_hours"] = int((mask & frame.expert_ready.to_numpy(bool)).sum())
                details["strong_risk_hours"] = int((mask & frame.spike_probability.gt(.5).to_numpy()).sum())
            entry["models"][name] = details
        out["groups"][group] = entry
    cases = np.isin(days, EVENT_DAYS) & (hours == 19)
    for i in np.flatnonzero(cases):
        case = {"zone": panel.zone.iloc[i], "day": days[i], "hour_local": 19,
                "sample": panel["sample"].iloc[i], "in_annual_common_support": bool(common[i]),
                "actual": actual[i], "nyx": baseline[i], "storm": storm[i], "models": {}}
        for name, frame in models.items():
            row = frame.iloc[i]
            case["models"][name] = {"forecast": row.candidate_forecast, "uplift": row.applied_correction,
                "p": row.spike_probability, "weight": row.selected_weight, "q10": row.candidate_q10,
                "q90": row.candidate_q90, "reason": row.gate_reason,
                "mae_gain_vs_nyx": abs(actual[i] - baseline[i]) - abs(actual[i] - row.candidate_forecast)}
        out["preidentified_19h_cases"].append(case)
    for day in EVENT_DAYS:
        mask = common & (days == day)
        out["preidentified_day_aggregates"][day] = {
            "hours": int(mask.sum()), "nyx_mae": mean(np.abs(actual[mask] - baseline[mask])),
            "storm_mae": mean(np.abs(actual[mask] - storm[mask])),
            "models": {name: {"mae": mean(np.abs(actual[mask] - frame.candidate_forecast.to_numpy(float)[mask])),
                "active_hours": int((mask & frame.applied_correction.gt(0).to_numpy()).sum())} for name, frame in models.items()}}
    # Recheck hashes after reading/computing to fail if a concurrent writer touched results.
    verify(directory, manifest["input_files"])
    verify(directory, result["result_files"])
    text = json.dumps(serializable(out), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        path = args.output.resolve()
        if not path.is_relative_to(Path(__file__).resolve().parent) or path.suffix != ".json":
            raise ValueError("Only a JSON audit output inside tmp/ is permitted.")
        path.write_text(text, encoding="utf-8")
        compact = {"audit": str(path), "common_hours": out["common_hours"],
                   "checks": out["causality_and_coherence"], "groups": {}}
        for name, group in out["groups"].items():
            compact["groups"][name] = {"nyx_mae": group["nyx"]["mae"], "storm_mae": group["storm"]["mae"],
                "models": {model: {k: metrics[k] for k in ("mae", "mae_gain_vs_nyx", "active_hours",
                    "worsened_active_hours", "active_nyx_already_above_actual_hours", "coverage_p10_p90",
                    "mean_interval_width", "interval_score80", "top1_gain_vs_nyx",
                    "exploratory_paired_day_block_gain")} for model, metrics in group["models"].items()}}
        print(json.dumps(serializable(compact), indent=2, allow_nan=False))
    else:
        print(text)


if __name__ == "__main__":
    main()
