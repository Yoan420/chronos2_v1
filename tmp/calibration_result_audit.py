"""Read-only independent audit of a completed calibration experiment.

No polling, fitting, mutation, report regeneration, or output files. Results
are printed once; a snapshot without its final seal is refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from nyx_congestion_calibration import policy, runner
from nyx_congestion_calibration.report import assemble_panel, common_rows, CATALOG
from nyx_fundamental_stress.features import make_fundamental_features
from nyx_physical_p50.policy import _network_contract
from nyx_scarcity import policy as base


def parsed(value):
    return json.loads(value) if isinstance(value, str) else value


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NaT or value is pd.NA or isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    directory = args.snapshot.resolve()
    if not (directory / "results_manifest.json").is_file():
        raise SystemExit("Final result seal absent: no partial-result audit performed.")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    runner.verify_result(directory, manifest)
    manifest_sha = runner.digest(directory / "manifest.json")
    result_sha = runner.digest(directory / "results_manifest.json")
    source = Path(manifest["source_dir"])
    for name in runner.COPIED_INPUTS:
        assert runner.digest(directory / name) == manifest["source_files"][name], name
    predictions = pd.read_parquet(directory / "predictions.parquet")
    folds = pd.read_parquet(directory / "folds.parquet")
    assert not folds.duplicated(["strategy", "fit_day"]).any()
    # Reconstruct admissible histories independently from sealed input rows.
    panel = pd.read_parquet(directory / "panel.parquet")
    network = pd.read_parquet(directory / "network_features.parquet")
    signals = pd.read_parquet(directory / "signals.parquet")
    augmented, physical, required, _ = make_fundamental_features(panel, variant="fundamental")
    network_names = _network_contract(panel, network)
    augmented = pd.concat([augmented, network, signals], axis=1)
    params = base._parameters({**manifest["settings"], "threads": 2,
        "calibration_days": 90, "minimum_training_days": 118,
        "feature_columns": physical + network_names + policy.SIGNALS,
        "required_feature_columns": required + policy.SIGNALS})
    history = base._prepare(augmented, params)
    summary, calibration = [], []
    trained = folds.loc[folds.status.eq("trained")].copy()
    for strategy, block in folds.groupby("strategy"):
        good = block.loc[block.status.eq("trained")]
        summary.append(dict(strategy=strategy, attempts=len(block), trained=len(good),
            first_fit=good.fit_day.min(), last_fit=good.fit_day.max(),
            fallback_reasons=block.loc[~block.status.eq("trained"), "reason"].value_counts().to_dict()))
    for day, pair in trained.groupby("fit_day"):
        assert set(pair.strategy) == {"control", "congestion"}
        control, enriched = pair.iloc[0], pair.iloc[1]
        for field in ("fit_cutoff_utc", "core_rows", "calibration_rows", "severity_rows",
                      "thresholds_eur_mwh", "partitions_sha256", "calibration_window_days",
                      "severity_exclusion_days", "max_label_available_at_utc"):
            assert control[field] == enriched[field], (day, field)
        cutoff = pd.Timestamp(control.fit_cutoff_utc)
        expected = (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
        assert cutoff == expected
        assert pd.Timestamp(control.max_label_available_at_utc) <= cutoff
        first = (pd.Timestamp(day)-pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        split = (pd.Timestamp(day)-pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        severity_end = (pd.Timestamp(day)-pd.Timedelta(days=28)).strftime("%Y-%m-%d")
        usable = history.loc[history._day.ge(first) & history._day.lt(day)
            & history._features_valid & history._label_valid & history.label_available_at_utc.le(cutoff)
            & np.isfinite(history[policy.FUEL]) & history[policy.FUEL].gt(0)]
        partitions = {"core": usable.loc[usable._day.lt(split)],
                      "calibration": usable.loc[usable._day.ge(split)],
                      "severity": usable.loc[usable._day.lt(severity_end)]}
        recorded = parsed(control.partitions_sha256)
        assert {name: policy._fingerprint(frame) for name, frame in partitions.items()} == recorded, day
        assert set(partitions["core"].index).isdisjoint(partitions["calibration"].index)
        core = partitions["core"]
        thresholds = {z: max(params["minimum_threshold_eur_mwh"],
            float(np.quantile(core.loc[core.zone.eq(z), "_error"], params["threshold_quantile"])))
            for z in sorted(core.zone.unique())}
        assert thresholds == parsed(control.thresholds_eur_mwh)
        fitted = parsed(control.calibration)
        for strategy, state in fitted.items():
            assert state["converged"] is True and state["slope"] >= 0
            calibration.append(dict(day=day, strategy=strategy, slope=state["slope"],
                intercept=state["intercept"], offsets=state["offsets"], support=state["support"]))
    assert predictions.control_expert_ready.equals(predictions.congestion_expert_ready)
    for strategy in ("control", "congestion"):
        selected = predictions.loc[predictions[strategy+"_expert_ready"].eq(True)]
        assert selected[strategy+"_expert_fit_day"].notna().all()
        fit_time = pd.to_datetime(selected[strategy+"_expert_fit_day"])-pd.Timedelta(days=1)+pd.Timedelta(hours=8)
        fit_time = fit_time.dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
        assert selected.forecast_origin_utc.ge(fit_time).all()
    _, wide, _, _ = assemble_panel(predictions, source)
    scored = common_rows(wide, end_day="2026-09-14", days=365).copy()
    scored["day"] = scored.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    scored["hour"] = scored.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    rows = []
    models = [r["id"] for r in CATALOG]
    for zone in ["ALL", *sorted(scored.zone.unique())]:
        block = scored if zone == "ALL" else scored.loc[scored.zone.eq(zone)]
        baseline_error = (block.nuclear_kalman-block.actual).abs()
        storm_error = (block.storm-block.actual).abs()
        for model in models:
            forecast = block.storm if model == "__storm__" else block[model]
            error = forecast-block.actual
            changed = (forecast-block.nuclear_kalman).abs().gt(1e-9)
            gain = baseline_error-error.abs()
            rows.append(dict(zone=zone, model=model, n=len(block), mae=error.abs().mean(),
                rmse=np.sqrt(np.square(error).mean()), win_hour_pct=100*error.abs().lt(storm_error).mean(),
                changed=int(changed.sum()), changed_days=block.loc[changed, "day"].nunique(),
                better=int((changed & gain.gt(1e-9)).sum()), worse=int((changed & gain.lt(-1e-9)).sum()),
                sum_absolute_error_gain=gain.sum()))
    case = scored.loc[scored.day.eq("2026-09-14") & scored.hour.eq(19)]
    fields = ["zone", "actual", "storm", "nuclear_kalman", *policy.MODELS]
    for strategy in ("control", "congestion"):
        fields.extend(strategy+"_"+field for field in ("expert_ready", "expert_fit_day", "raw_probability",
            "spike_probability", "threshold_eur_mwh", "raw_correction", "bounded_correction",
            "selected_weight", "calibration_status", "proposal_reason"))
    fields = [field for field in fields if field in case]
    june = scored.loc[scored.day.between("2026-06-24", "2026-06-26")]
    june_rows = []
    for zone, block in june.groupby("zone"):
        for model in ["nuclear_kalman", *policy.MODELS]:
            error = block[model]-block.actual
            changed = block[model].sub(block.nuclear_kalman).abs().gt(1e-9)
            june_rows.append(dict(zone=zone, model=model, n=len(block), mae=error.abs().mean(),
                rmse=np.sqrt(np.square(error).mean()), changed=int(changed.sum())))
    detector = []
    for strategy in ("control", "congestion"):
        block = scored.loc[scored[strategy+"_expert_ready"].eq(True)]
        y = block.actual.sub(block.nuclear_kalman).ge(block[strategy+"_threshold_eur_mwh"]).to_numpy(int)
        for kind in ("raw_probability", "spike_probability"):
            probability = block[strategy+"_"+kind].to_numpy(float)
            detector.append(dict(strategy=strategy, kind=kind, n=len(y), events=int(y.sum()),
                brier=np.square(probability-y).mean(), ap=average_precision_score(y, probability),
                average_probability=probability.mean()))
    payload = dict(snapshot=str(directory), snapshot_manifest_sha256=manifest_sha,
        results_manifest_sha256=result_sha, all_temporal_partition_checks_passed=True,
        stage1_inputs_identical_to_source=True, folds=summary, metrics=rows,
        detector=detector, september14_19h=case[fields].to_dict("records"), june24_26=june_rows,
        final_calibration=[r for r in calibration if r["day"] == "2026-09-14"],
        calibration_zero_slopes=sum(r["slope"] <= 1e-10 for r in calibration))
    pointer = directory / "latest_report.json"
    if pointer.exists():
        target = json.loads(pointer.read_text(encoding="utf8"))["report_directory"]
        report = json.loads((Path(target)/"metrics.json").read_text(encoding="utf8"))
        payload["reported_economic_all"] = [r for r in report["periods"]["365"]["economic"]["rows"] if r["zone"] == "ALL"]
    runner.verify_result(directory, manifest)
    assert runner.digest(directory/"manifest.json") == manifest_sha
    assert runner.digest(directory/"results_manifest.json") == result_sha
    print(json.dumps(safe(payload), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
