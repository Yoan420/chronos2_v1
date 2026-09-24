"""Independent read-only metrics for a completed frozen CoherentP50 replay.

No model fitting, prediction regeneration, reporting write, or tuning occurs.
JSON is printed to stdout; all inputs/results must pass their sealed hashes.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SNAPSHOT = Path(sys.argv[1])


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_group(directory, expected):
    for name, digest in expected.items():
        assert sha(directory/name).lower() == digest.lower(), str(directory/name)


def scores(observed, point, lower, upper):
    if not len(observed):
        return {"hours": 0}
    error = observed-point
    return {"hours": len(observed), "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))), "forecast_minus_actual_bias": float(-np.mean(error)),
        "observed_le_p10": float(np.mean(observed <= lower)),
        "observed_le_p50": float(np.mean(observed <= point)),
        "observed_le_p90": float(np.mean(observed <= upper)),
        "coverage_p10_p90": float(np.mean((lower <= observed)&(observed <= upper))),
        "mean_interval_width": float(np.mean(upper-lower)),
        "pinball10": float(np.mean(np.maximum(.1*(observed-lower), -.9*(observed-lower)))),
        "pinball50": float(np.mean(np.abs(error))*.5),
        "pinball90": float(np.mean(np.maximum(.9*(observed-upper), -.1*(observed-upper))))}


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    for kind in ("forest", "empirical"):
        if not (SNAPSHOT/kind/"results_manifest.json").is_file():
            print(json.dumps({"status": "pending", "variant": kind}))
            return
    manifest = read_json(SNAPSHOT/"manifest.json")
    verify_group(SNAPSHOT, manifest["input_files"])
    for kind in ("forest", "empirical"):
        result = read_json(SNAPSHOT/kind/"results_manifest.json")
        assert result["status"] == "completed"
        assert result["suite_manifest_sha256"] == sha(SNAPSHOT/"manifest.json")
        verify_group(SNAPSHOT/kind, result["result_files"])
    panel = pd.read_parquet(SNAPSHOT/"panel.parquet")
    models = {"old_fundamental_governed": pd.read_parquet(SNAPSHOT/"source_predictions.parquet"),
        "old_fundamental_25": pd.read_parquet(SNAPSHOT/"old_fundamental_25_predictions.parquet"),
        "regional_25": pd.read_parquet(SNAPSHOT/"regional_25_predictions.parquet")}
    for kind in ("forest", "empirical"):
        for policy, name in (("direct", "predictions.parquet"), ("governed", "governed_predictions.parquet")):
            models[kind+"_"+policy] = pd.read_parquet(SNAPSHOT/kind/name)
    for frame in models.values():
        pd.testing.assert_frame_equal(panel, frame[panel.columns], check_exact=True)
    observed, baseline = panel.actual.to_numpy(float), panel.forecast.to_numpy(float)
    storm = panel.benchmark_forecast.to_numpy(float)
    lower, upper = panel.q10.to_numpy(float), panel.q90.to_numpy(float)
    local = panel.timestamp_utc.dt.tz_convert("Europe/Paris")
    days, hours = local.dt.strftime("%Y-%m-%d"), local.dt.hour
    config = read_json(SNAPSHOT/"base_config.json")
    end = config["end_day"]
    start = (pd.Timestamp(end)-pd.Timedelta(days=config["evaluation_days"]-1)).strftime("%Y-%m-%d")
    common = (panel["sample"].eq("evaluation") & days.ge(start) & days.le(end)).to_numpy()
    common &= np.isfinite(np.column_stack([observed, baseline, storm, lower, upper])).all(axis=1)
    for frame in models.values():
        common &= np.isfinite(frame[["candidate_forecast", "candidate_q10", "candidate_q90"]].to_numpy(float)).all(axis=1)
    out = {"snapshot": str(SNAPSHOT), "common_hours": int(common.sum()), "start": start, "end": end,
        "top_1_percent_definition": "Ex post observed price >= within-group 99th percentile on identical common support.",
        "forecast_probability_is_frozen": True, "causality_and_support": {}, "groups": {}, "preidentified_cases": []}
    frozen = models["old_fundamental_governed"]
    for kind in ("forest", "empirical"):
        frame = models[kind+"_direct"]
        np.testing.assert_array_equal(frame.spike_probability, frozen.spike_probability)
        np.testing.assert_array_equal(frame.expert_ready, frozen.expert_ready)
        ready = frame.expert_ready.to_numpy(bool)
        q = frame.loc[ready, ["mixture_error_q10", "mixture_error_q50", "mixture_error_q90"]].to_numpy(float)
        assert (np.diff(q, axis=1) >= 0).all()
        p = frame.loc[ready, "spike_probability"].to_numpy(float)
        u = frame.loc[ready, "threshold_eur_mwh"].to_numpy(float)
        assert (q[p > .5, 1] >= u[p > .5]).all()
        assert (q[p <= .5, 1] < u[p <= .5]).all()
        folds = pd.read_parquet(SNAPSHOT/kind/"folds.parquet")
        trained = folds.loc[folds.status.eq("trained")]
        assert (pd.to_datetime(trained.cdf_max_label_available_at_utc, utc=True)
                <= pd.to_datetime(trained.fit_cutoff_utc, utc=True)).all()
        assert trained.cdf_model_training_end_day.lt(trained.calibration_start_day).all()
        assert trained.cdf_uses_calibration_labels.eq(False).all()
        assert trained.detector_retrained.eq(False).all()
        out["causality_and_support"][kind] = {"trained_folds": len(trained),
            "training_core_rows_min": int(trained.cdf_core_rows.min()),
            "training_core_rows_max": int(trained.cdf_core_rows.max()),
            "ready_common_hours": int(np.count_nonzero(ready&common)),
            "ready_common_p_above_half": int(np.count_nonzero(ready&common&frame.spike_probability.gt(.5).to_numpy())),
            "support_quantiles_causal_metadata_checks": "passed"}
    event_days = ["2026-06-24", "2026-06-25", "2026-06-26", "2026-09-14"]
    groups = {"ALL": np.ones(len(panel), bool), **{z: panel.zone.eq(z).to_numpy() for z in sorted(panel.zone.unique())}}
    for group_name, group_mask in groups.items():
        mask = common&group_mask
        top = float(np.quantile(observed[mask], .99))
        top_mask = mask&(observed >= top)
        no_preidentified = mask&~days.isin(event_days).to_numpy()
        entry = {"hours": int(mask.sum()), "baseline": scores(observed[mask], baseline[mask], lower[mask], upper[mask]),
            "storm_mae": float(np.mean(np.abs(observed[mask]-storm[mask]))),
            "top1_threshold": top, "top1_hours": int(top_mask.sum()),
            "top1_baseline_mae": float(np.mean(np.abs(observed[top_mask]-baseline[top_mask]))),
            "top1_storm_mae": float(np.mean(np.abs(observed[top_mask]-storm[top_mask]))), "models": {}}
        for name, frame in models.items():
            point = frame.candidate_forecast.to_numpy(float)
            lo, hi = frame.candidate_q10.to_numpy(float), frame.candidate_q90.to_numpy(float)
            active = mask&~np.isclose(point, baseline, rtol=0, atol=1e-9)
            gain = np.abs(observed-baseline)-np.abs(observed-point)
            daily_gain = pd.DataFrame({"day": days[mask].to_numpy(), "gain": gain[mask]}).groupby("day").gain.agg(["sum", "count"])
            mean_gain = float(gain[mask].mean())
            influence = daily_gain["sum"]-mean_gain*daily_gain["count"]
            standard_error = float(np.sqrt(len(daily_gain)/(len(daily_gain)-1)*np.square(influence).sum())/mask.sum())
            details = scores(observed[mask], point[mask], lo[mask], hi[mask])
            details.update(mae_gain_vs_nyx=mean_gain,
                exploratory_day_cluster_gain_ci95=[mean_gain-1.96*standard_error, mean_gain+1.96*standard_error],
                top1_mae=float(np.mean(np.abs(observed[top_mask]-point[top_mask]))),
                top1_mae_gain_vs_nyx=float(gain[top_mask].mean()),
                active_hours=int(active.sum()), improved_active_hours=int((active&(gain > 1e-9)).sum()),
                worsened_active_hours=int((active&(gain < -1e-9)).sum()),
                active_no_underforecast_hours=int((active&(observed <= baseline)).sum()),
                active_outside_error50_hours=int((active&((observed-baseline) < 50)).sum()),
                positive_error50_hours=int((mask&((observed-baseline) >= 50)).sum()),
                active_within_error50_hours=int((active&((observed-baseline) >= 50)).sum()),
                mae_gain_excluding_preidentified_days=float(gain[no_preidentified].mean()),
                signed_total_absolute_error_gain=float(gain[mask].sum()),
                active_scores=scores(observed[active], point[active], lo[active], hi[active]),
                nyx_on_same_active_scores=scores(observed[active], baseline[active], lower[active], upper[active]))
            entry["models"][name] = details
        out["groups"][group_name] = entry
    case_mask = days.isin(event_days)&hours.eq(19)
    for index in panel.index[case_mask]:
        row = panel.loc[index]
        case = {"zone": row.zone, "day": days.loc[index], "hour": 19,
            "actual": float(observed[index]), "nyx": float(baseline[index]), "storm": float(storm[index]),
            "old_signed_median": float(frozen.loc[index, "predicted_signed_residual_median"]),
            "p": float(frozen.loc[index, "spike_probability"]),
            "physical_gate": bool(frozen.loc[index, "physical_gate_passed"]) if pd.notna(frozen.loc[index, "physical_gate_passed"]) else None,
            "models": {}}
        for name, frame in models.items():
            values = {"point": float(frame.loc[index, "candidate_forecast"]),
                "uplift": float(frame.loc[index, "applied_correction"]),
                "reason": str(frame.loc[index, "gate_reason"])}
            if name.startswith(("forest", "empirical")):
                values.update(raw_mixture_error50=float(frame.loc[index, "mixture_error_q50"]),
                    q10=float(frame.loc[index, "candidate_q10"]), q90=float(frame.loc[index, "candidate_q90"]))
            case["models"][name] = values
        out["preidentified_cases"].append(case)
    text = json.dumps(finite_json(out), ensure_ascii=False, indent=2, allow_nan=False)
    if len(sys.argv) > 2:
        output = Path(sys.argv[2]).resolve()
        allowed = (Path(__file__).resolve().parent)
        if not output.is_relative_to(allowed) or output.suffix != ".json":
            raise ValueError("Generated audit JSON must remain inside tmp/.")
        output.write_text(text, encoding="utf-8")
        print(json.dumps({"status": "completed", "audit": str(output), "common_hours": out["common_hours"],
            "groups": {name: {"hours": group["hours"], "baseline_mae": group["baseline"]["mae"],
                "storm_mae": group["storm_mae"], "models": {key: {field: values[field] for field in
                    ("mae", "mae_gain_vs_nyx", "top1_mae", "top1_mae_gain_vs_nyx", "active_hours", "worsened_active_hours",
                     "coverage_p10_p90", "mae_gain_excluding_preidentified_days", "exploratory_day_cluster_gain_ci95")}
                    for key, values in group["models"].items()}} for name, group in out["groups"].items()}}, indent=2))
    else:
        print(text)


if __name__ == "__main__":
    main()
