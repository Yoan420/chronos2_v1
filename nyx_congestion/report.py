"""Offline, paired diagnostics for the isolated congestion/P50 experiment.

Post-coupling labels are evaluated here, never returned as model features.
The CNEC-hour sample is separate from the price country-hour sample.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from kpi_report.economic import compute_economic_kpis
from kpi_report.metrics import compute_kpis
from nyx_physical_p50.report import _safe, _same, _sha, _typed

KEYS = ["zone", "timestamp_utc"]
MODELS = ("congestion_control_direct", "congestion_control_governed", "congestion_direct", "nyx_congestion")
COMPARATORS = ("network_fuel_direct", "nyx_physical_p50")
CATALOG = [
    {"id": "nuclear_kalman", "label": "NYX opérationnel", "kind": "production"},
    {"id": "nyx_congestion", "label": "Congestion · gouverné", "kind": "candidate"},
    {"id": "congestion_direct", "label": "Congestion · direct", "kind": "candidate"},
    {"id": "congestion_control_governed", "label": "Contrôle physique · gouverné", "kind": "control"},
    {"id": "congestion_control_direct", "label": "Contrôle physique · direct", "kind": "control"},
    {"id": "nyx_physical_p50", "label": "Réseau + CGC précédent · gouverné", "kind": "previous"},
    {"id": "network_fuel_direct", "label": "Réseau + CGC précédent · direct", "kind": "previous"},
    {"id": "__storm__", "label": "Storm", "kind": "benchmark"},
]


def _numeric(frame, name, *, probability=False, nonnegative=False):
    values = pd.to_numeric(frame[name], errors="raise").astype(float)
    if np.isinf(values).any():
        raise ValueError(f"Report {name} is infinite.")
    if probability and (values.dropna().lt(0).any() or values.dropna().gt(1).any()):
        raise ValueError(f"Report {name} must be in [0, 1].")
    if nonnegative and values.dropna().lt(0).any():
        raise ValueError(f"Report {name} must be nonnegative.")
    frame[name] = values


def assemble_panel(predictions, physical_source):
    """Verify frozen comparators and preserve the complete original panel."""
    source, seals = Path(physical_source).resolve(), {}
    def capture(path, expected=None):
        value = _sha(path)
        if expected is not None and value != expected:
            raise ValueError(f"Source checksum mismatch: {path}")
        seals[str(path)] = value
        return value
    manifest_sha = capture(source/"manifest.json")
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf8"))
    expected = manifest.get("input_files", {}).get("panel.parquet")
    if not expected:
        raise ValueError("Physical source does not seal panel.parquet.")
    capture(source/"panel.parquet", expected)
    capture(source/"results_manifest.json")
    result = json.loads((source/"results_manifest.json").read_text(encoding="utf8"))
    if result.get("status") != "completed" or result.get("suite_manifest_sha256") != manifest_sha:
        raise ValueError("Physical comparator is not sealed/completed.")
    expected = result.get("result_files", {}).get("predictions.parquet")
    if not expected:
        raise ValueError("Physical comparator does not seal predictions.parquet.")
    capture(source/"predictions.parquet", expected)
    ancestor = Path(manifest["source_dir"]).resolve()
    expected = manifest.get("source_files", {}).get("source_audit.json")
    if not expected:
        raise ValueError("Physical source does not seal ancestral source audit.")
    capture(ancestor/"source_audit.json", expected)
    source_audit = json.loads((ancestor/"source_audit.json").read_text(encoding="utf8"))
    base, wide = _typed(pd.read_parquet(source/"panel.parquet")), _typed(predictions)
    previous = _typed(pd.read_parquet(source/"predictions.parquet"))
    for other in (wide, previous):
        _same(base, other)
        for name in base.columns:
            if name not in other:
                raise ValueError(f"Missing preserved panel column {name}.")
            try:
                pd.testing.assert_series_equal(base[name], other[name], check_dtype=False,
                                               check_exact=True, check_names=False)
            except AssertionError as exc:
                raise ValueError(f"Changed preserved panel column {name}.") from exc
    if set(MODELS).difference(wide) or set(COMPARATORS).difference(previous):
        raise ValueError("Missing congestion or physical comparison models.")
    wide["nuclear_kalman"] = wide.forecast
    for model in COMPARATORS:
        if model in wide and not np.allclose(wide[model], previous[model], atol=1e-9, rtol=0, equal_nan=True):
            raise ValueError(f"Changed frozen comparator {model}.")
        wide[model] = previous[model]
    for model in (*MODELS, *COMPARATORS):
        _numeric(wide, model)
        lower, upper = f"{model}_q10", f"{model}_q90"
        if model in MODELS and (lower not in wide or upper not in wide):
            raise ValueError(f"Missing prediction intervals for {model}.")
        if lower in wide and upper in wide:
            _numeric(wide, lower)
            _numeric(wide, upper)
            valid = wide[[model, lower, upper]].notna().all(axis=1)
            if ((wide.loc[valid, lower] > wide.loc[valid, model]+1e-9)
                    | (wide.loc[valid, upper] < wide.loc[valid, model]-1e-9)).any():
                raise ValueError(f"Unordered P10/P50/P90 for {model}.")
    columns = ["actual", "storm", "sample", "forecast_origin_utc"]
    columns += [name for name in ("forecast_eligible", "benchmark_eligible") if name in wide]
    long = []
    for model in CATALOG:
        if model["id"] == "__storm__":
            continue
        part = wide[columns].copy()
        part["model_id"], part["forecast"] = model["id"], wide[model["id"]]
        long.append(part.reset_index())
    return pd.concat(long, ignore_index=True), wide.reset_index(), source_audit, seals


def _constraints(frame):
    required = {"timestamp_utc", "forecast_origin_utc", "constraint_key", "label_active",
                "label_shadow_price", "label_eligible", "label_available_at_utc",
                "activation_probability", "intensity_if_active", "expected_shadow_price",
                "climatology_probability", "expert_ready"}
    if required.difference(frame):
        raise ValueError(f"Missing constraint diagnostics: {sorted(required.difference(frame))}")
    # Do not copy hundreds of training features into the reporting working set.
    optional = {"cne_name", "contingency_name", "direction", "fit_day", "realised_fb_premium"}
    frame = frame.loc[:, [name for name in frame if name in required | optional]].copy(deep=True)
    for name in ("timestamp_utc", "forecast_origin_utc", "label_available_at_utc"):
        if (not isinstance(frame[name].dtype, pd.DatetimeTZDtype)
                and any(pd.notna(value) and pd.Timestamp(value).tzinfo is None for value in frame[name])):
            raise ValueError(f"Constraint {name} must be timezone aware.")
        frame[name] = pd.to_datetime(frame[name], utc=True)
    if frame[["timestamp_utc", "forecast_origin_utc", "constraint_key"]].isna().any().any():
        raise ValueError("Missing constraint identity.")
    if frame.duplicated(["timestamp_utc", "constraint_key"]).any():
        raise ValueError("Duplicate constraint-hour identity.")
    for name in ("label_eligible", "expert_ready", "label_active"):
        if not frame[name].dropna().isin([True, False, 0, 1]).all():
            raise ValueError(f"Constraint {name} must be boolean or missing.")
    for name in ("activation_probability", "climatology_probability"):
        _numeric(frame, name, probability=True)
    for name in ("label_shadow_price", "intensity_if_active", "expected_shadow_price"):
        _numeric(frame, name, nonnegative=True)
    valid = frame[["activation_probability", "intensity_if_active", "expected_shadow_price"]].notna().all(axis=1)
    if not np.allclose(frame.loc[valid, "expected_shadow_price"],
                       frame.loc[valid, "activation_probability"]*frame.loc[valid, "intensity_if_active"],
                       atol=1e-6, rtol=1e-6):
        raise ValueError("Expected shadow price differs from probability times conditional mean.")
    frame["hour"] = frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    return frame


def congestion_metrics(frame, *, end_day, days, evaluation_timestamps):
    """Pooled CNEC-hour scores; never multiply rows by the number of countries."""
    frame = _constraints(frame)
    end = date.fromisoformat(end_day)
    civil = frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.date
    period = frame.loc[civil.between(end-timedelta(days=days-1), end)
                       & frame.timestamp_utc.isin(evaluation_timestamps)].copy()
    eligible = (period.label_eligible.fillna(False).astype(bool)
                & period.label_available_at_utc.notna()
                & period.label_active.notna() & period.label_shadow_price.notna())
    ready = period.expert_ready.fillna(False).astype(bool)
    common = eligible & ready & period[["activation_probability", "climatology_probability",
                                        "intensity_if_active", "expected_shadow_price"]].notna().all(axis=1)
    block = period.loc[common]
    y = block.label_active.astype(int).to_numpy()
    truth = block.label_shadow_price.to_numpy(float)
    scores, reliability, pr_curves = [], [], []
    for model, column in (("congestion", "activation_probability"), ("past_prior", "climatology_probability")):
        probability = block[column].to_numpy(float)
        binary_classes = len(np.unique(y)) == 2
        brier = np.mean((probability-y)**2) if len(y) else None
        ap = float(average_precision_score(y, probability)) if binary_classes else None
        scores.append({"model_id": model, "n_rows": len(block), "n_active": int(y.sum()),
                       "prevalence": float(y.mean()) if len(y) else None,
                       "brier": brier, "average_precision": ap,
                       "average_precision_unavailable_reason": None if binary_classes else "fewer_than_two_classes"})
        bins = np.minimum((probability*10).astype(int), 9)
        for bin_id in range(10):
            take = bins == bin_id
            reliability.append({"model_id": model, "bin_lower": bin_id/10, "bin_upper": (bin_id+1)/10,
                                "n_rows": int(take.sum()),
                                "mean_probability": probability[take].mean() if take.any() else None,
                                "observed_frequency": y[take].mean() if take.any() else None})
        if binary_classes:
            precision, recall, _ = precision_recall_curve(y, probability)
            indices = np.unique(np.linspace(0, len(precision)-1, min(201, len(precision))).astype(int))
            pr_curves.append({"model_id": model, "precision": precision[indices], "recall": recall[indices]})
    active = y.astype(bool)
    intensity = {"n_rows": len(block), "n_active": int(active.sum()),
                 "global_mae_expected_shadow_price": np.abs(block.expected_shadow_price-truth).mean(),
                 "global_rmse_expected_shadow_price": np.sqrt(np.square(block.expected_shadow_price-truth).mean()),
                 "active_mae_conditional_intensity": np.abs(block.intensity_if_active.to_numpy()[active]-truth[active]).mean() if active.any() else None,
                 "active_rmse_conditional_intensity": np.sqrt(np.square(block.intensity_if_active.to_numpy()[active]-truth[active]).mean()) if active.any() else None,
                 "active_mae_expected_shadow_price": np.abs(block.expected_shadow_price.to_numpy()[active]-truth[active]).mean() if active.any() else None,
                 "zero_prediction_global_mae": np.abs(truth).mean() if len(truth) else None,
                 "zero_prediction_global_rmse": np.sqrt(np.square(truth).mean()) if len(truth) else None,
                 "observed_mean_shadow_price": truth.mean() if len(truth) else None}
    return _safe({"coverage": {"n_initial_constraint_hours": len(period), "n_label_eligible": int(eligible.sum()),
                              "n_expert_ready": int(ready.sum()), "n_common_scored": len(block),
                              "n_unknown_or_unqualified_labels": int((~eligible).sum()),
                              "n_qualified_but_not_common": int((eligible & ~common).sum()),
                              "n_distinct_constraints": int(block.constraint_key.nunique()),
                              "unit": "CNEC-hour; no country duplication"},
                  "scores": scores, "reliability": reliability, "pr_curves": pr_curves, "intensity": intensity})


def price_diagnostics(wide, *, end_day, days, zones):
    civil = wide.timestamp_utc.dt.tz_convert("Europe/Paris").dt.date
    end = date.fromisoformat(end_day)
    names = [item["id"] for item in CATALOG if item["id"] != "__storm__"]
    selected = wide.loc[civil.between(end-timedelta(days=days-1), end) & wide["sample"].eq("evaluation")].copy()
    for name in ("forecast_eligible", "benchmark_eligible"):
        if name in selected:
            selected = selected.loc[selected[name].fillna(False).astype(bool)]
    selected = selected.dropna(subset=[*names, "actual", "storm"])
    selected["hour"] = selected.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    interventions, hourly = [], []
    for zone in ["ALL", *zones]:
        block = selected if zone == "ALL" else selected.loc[selected.zone.eq(zone)]
        baseline = (block.nuclear_kalman-block.actual).abs()
        for model in [*names, "__storm__"]:
            point = block.storm if model == "__storm__" else block[model]
            error = point-block.actual
            changed, gain = (point-block.nuclear_kalman).abs().gt(1e-9), baseline-error.abs()
            tail = block.actual.ge(200)
            interventions.append({"zone": zone, "model_id": model, "n_hours": len(block),
                "changed_hours": int(changed.sum()), "better": int((changed & gain.gt(1e-9)).sum()),
                "worse": int((changed & gain.lt(-1e-9)).sum()), "tail_n": int(tail.sum()),
                "tail_mae": error.loc[tail].abs().mean(), "tail_rmse": np.sqrt((error.loc[tail]**2).mean()),
                "normal_mae": error.loc[~tail].abs().mean()})
            for hour, group in pd.DataFrame({"hour": block.hour, "ae": error.abs()}).groupby("hour"):
                hourly.append({"zone": zone, "model_id": model, "hour": int(hour), "mae": group.ae.mean()})
    return _safe({"interventions": interventions, "hourly": hourly})


def constraint_case(frame):
    """Selection only for display: predicted leaders plus realized leaders."""
    frame = _constraints(frame)
    civil = frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.date
    selected = frame.loc[civil.eq(date(2026, 9, 14))].copy()
    focus = selected.loc[selected.hour.eq(19)]
    predicted = focus.sort_values("expected_shadow_price", ascending=False, na_position="last").head(8)
    observed = focus.loc[focus.label_eligible.fillna(False).astype(bool)].sort_values(
        "label_shadow_price", ascending=False, na_position="last").head(4)
    keys = list(dict.fromkeys([*predicted.constraint_key, *observed.constraint_key]))
    columns = ["timestamp_utc", "constraint_key", "hour", "label_active", "label_shadow_price",
               "label_eligible", "activation_probability", "intensity_if_active", "expected_shadow_price",
               "expert_ready"]
    columns += [name for name in ("cne_name", "contingency_name", "direction", "fit_day") if name in frame]
    rows = selected.loc[selected.constraint_key.isin(keys), columns].sort_values(["constraint_key", "timestamp_utc"])
    return _safe({"selection": "8 largest expected shadow prices + 4 largest qualified realized shadow prices at 19h; display only, never a feature or tuned rule",
                  "constraint_keys": keys, "rows": rows.to_dict("records")})


def domain_coverage(frame, *, end_day, days, zones, evaluation_timestamps):
    """Ex-post contribution mass covered by initial universe; FR reference only.

    This audit reference is NOT the symmetric CWE reference of model features.
    Ratios are ratios of summed absolute contributions, never averaged ratios.
    """
    if frame is None or frame.empty:
        return {"available": False, "rows": [], "case": [], "reason": "No qualified zonal coverage sidecar."}
    values = ["label_absolute_contribution_eur_mwh", "initial_covered_absolute_contribution_eur_mwh",
              "outside_initial_absolute_contribution_eur_mwh"]
    flags = ["label_eligible", "label_physical_matching_complete", "initial_hour_available"]
    required = {*KEYS, *values, *flags}
    if required.difference(frame):
        raise ValueError(f"Missing domain coverage fields: {sorted(required.difference(frame))}")
    data = frame.loc[:, [name for name in frame if name in required]].copy()
    if (not isinstance(data.timestamp_utc.dtype, pd.DatetimeTZDtype)
            and any(pd.notna(v) and pd.Timestamp(v).tzinfo is None for v in data.timestamp_utc)):
        raise ValueError("Domain coverage timestamps must be timezone aware.")
    data["timestamp_utc"] = pd.to_datetime(data.timestamp_utc, utc=True)
    if data[KEYS].isna().any().any() or data.duplicated(KEYS).any() or not set(data.zone).issubset(zones):
        raise ValueError("Invalid domain coverage country/hour identity.")
    for name in values:
        _numeric(data, name, nonnegative=True)
    for name in flags:
        if not data[name].dropna().isin([True, False]).all():
            raise ValueError("Explicit domain coverage eligibility flags required.")
    data["_qualified"] = data[flags].fillna(False).astype(bool).all(axis=1) & data[values].notna().all(axis=1)
    qualified = data.loc[data._qualified]
    if not np.allclose(qualified[values[0]], qualified[values[1]]+qualified[values[2]], atol=1e-6, rtol=1e-9):
        raise ValueError("Covered and outside contributions do not sum to the total.")
    end = date.fromisoformat(end_day)
    civil = data.timestamp_utc.dt.tz_convert("Europe/Paris")
    data["hour"] = civil.dt.hour
    in_period = civil.dt.date.between(end-timedelta(days=days-1), end) & data.timestamp_utc.isin(evaluation_timestamps)
    period = data.loc[in_period]
    rows = []
    for zone in ["ALL", *zones]:
        part = period if zone == "ALL" else period.loc[period.zone.eq(zone)]
        good = part.loc[part._qualified]
        total, covered, outside = (float(good[name].sum()) if len(good) else None for name in values)
        rows.append({"zone": zone, "n_rows": len(part), "n_qualified_hours": len(good),
            "n_unqualified_hours": int((~part._qualified).sum()),
            "total_absolute_contribution_sum": total, "covered_absolute_contribution_sum": covered,
            "outside_absolute_contribution_sum": outside,
            "coverage_ratio": covered/total if total is not None and total > 0 else None})
    case = data.loc[civil.dt.date.eq(date(2026, 9, 14)) & data.hour.eq(19)
                    & data.timestamp_utc.isin(evaluation_timestamps)].copy()
    case["coverage_ratio"] = np.where(case._qualified & case[values[0]].gt(0), case[values[1]]/case[values[0]], np.nan)
    case.loc[~case._qualified, values] = np.nan
    return _safe({"available": True, "reference": "FR; post-coupling coverage audit only, NOT the model's symmetric CWE reference",
                  "aggregation": "ratio of sums of absolute hourly contributions; undefined for zero denominator",
                  "rows": rows, "case": case.drop(columns="_qualified").to_dict("records")})


def regional_metrics(frame, *, end_day, days, zones, evaluation_timestamps):
    """Separate country-hour diagnostic of all-published flow-based pressure.

    G=D-min(D) requires all four CWE labels and is invariant to a common
    reference shift. The severe-pressure target is G if G>=50, zero otherwise.
    This is not a CNEC activation, a realised electricity price, or a P50.
    """
    if frame is None or frame.empty:
        return {"available": False, "rows": [], "coverage": [], "case": [],
                "reason": "No regional prequential predictions supplied."}
    if "zone" not in frame or "realised_fb_premium" not in frame:
        raise ValueError("Regional pressure requires zone and realised_fb_premium.")
    if not set(frame.zone.dropna()).issubset({"FR", "DE", "BE", "NL"}) or frame.zone.isna().any():
        raise ValueError("Regional pressure supports exactly CWE country identities.")
    data = _constraints(frame.assign(constraint_key=frame.zone))
    data["zone"] = frame.zone.to_numpy()
    _numeric(data, "realised_fb_premium", nonnegative=True)
    known = data.label_eligible.fillna(False).astype(bool)
    if data.loc[known, ["label_active", "label_shadow_price", "realised_fb_premium", "label_available_at_utc"]].isna().any().any():
        raise ValueError("Qualified regional labels must be complete.")
    counts = data.loc[known].groupby("timestamp_utc").zone.nunique()
    if not counts.eq(4).all():
        raise ValueError("Regional G requires all four qualified CWE countries, not three.")
    complete = data.loc[known]
    if len(complete) and not np.allclose(complete.groupby("timestamp_utc").realised_fb_premium.min(), 0., atol=1e-6, rtol=0):
        raise ValueError("Regional G must be measured above the four-country minimum.")
    actual = complete.realised_fb_premium.to_numpy(float)
    active = actual >= 50.
    if (not np.array_equal(complete.label_active.astype(bool), active)
            or not np.allclose(complete.label_shadow_price, np.where(active, actual, 0.), rtol=1e-9, atol=1e-6)):
        raise ValueError("Regional severe-pressure labels must apply the fixed G>=50 threshold.")
    rows, coverage = [], []
    for zone in ["ALL", *zones]:
        part = data if zone == "ALL" else data.loc[data.zone.eq(zone)]
        metrics = congestion_metrics(part, end_day=end_day, days=days,
                                     evaluation_timestamps=evaluation_timestamps)
        coverage.append({"zone": zone, **metrics["coverage"], "unit": "country-hour of regional severe pressure"})
        for score in metrics["scores"]:
            is_model = score["model_id"] == "congestion"
            rows.append({**score, "zone": zone, "model_id": "regional" if is_model else "past_prior",
                "global_rmse_expected_pressure": metrics["intensity"]["global_rmse_expected_shadow_price"] if is_model else None,
                "active_rmse_conditional_pressure": metrics["intensity"]["active_rmse_conditional_intensity"] if is_model else None,
                "global_mae_expected_pressure": metrics["intensity"]["global_mae_expected_shadow_price"] if is_model else None,
                "zero_prediction_global_rmse": metrics["intensity"]["zero_prediction_global_rmse"]})
    civil = data.timestamp_utc.dt.tz_convert("Europe/Paris")
    case = data.loc[civil.dt.date.eq(date(2026, 9, 14)) & civil.dt.hour.eq(19)
                    & data.timestamp_utc.isin(evaluation_timestamps),
        ["zone", "timestamp_utc", "label_eligible", "expert_ready", "label_active", "realised_fb_premium",
         "activation_probability", "intensity_if_active", "expected_shadow_price"]].copy()
    case["label_active"] = case.label_active.astype("boolean")
    unknown = ~case.label_eligible.fillna(False).astype(bool)
    case.loc[unknown, "label_active"] = pd.NA
    case.loc[unknown, "realised_fb_premium"] = np.nan
    return _safe({"available": True, "threshold_eur_mwh": 50., "rows": rows, "coverage": coverage,
        "case": case.to_dict("records"), "unit": "country-hour, NOT CNEC-hour",
        "target": "G_z=D_z-min(D_FR,D_DE,D_BE,D_NL); all four qualified countries required",
        "intensity_target": "conditional mean of G given G>=50; expected severe pressure is p times this mean, not E[G] overall",
        "conditional_intensity_floor_eur_mwh": 50.,
        "common_component_limitation": "G removes a component common to all four countries; regional common price increases require fundamental/CGC signals.",
        "reference_invariance": "Adding any common reference offset to the four D values leaves G unchanged."})


def label_timing(frame, *, end_day, days, zones, evaluation_timestamps):
    """Availability audit, not an assertion about first market publication.

    Compare each label's audited availability with 08:00 Europe/Paris on its
    delivery day: the origin for the following delivery, in civil time (DST).
    No labels or availability times are changed by this diagnostic.
    """
    if frame is None or frame.empty:
        return {"available": False, "rows": [], "monthly": [], "reason": "No zonal label timing sidecar."}
    required = {*KEYS, "label_eligible", "label_available_at_utc"}
    if required.difference(frame):
        raise ValueError(f"Missing label timing fields: {sorted(required.difference(frame))}")
    data = frame.loc[:, [name for name in frame if name in required]].copy()
    for name in ("timestamp_utc", "label_available_at_utc"):
        if (not isinstance(data[name].dtype, pd.DatetimeTZDtype)
                and any(pd.notna(v) and pd.Timestamp(v).tzinfo is None for v in data[name])):
            raise ValueError("Label timing timestamps must be timezone aware.")
        data[name] = pd.to_datetime(data[name], utc=True)
    if data[KEYS].isna().any().any() or data.duplicated(KEYS).any() or not set(data.zone).issubset(zones):
        raise ValueError("Invalid label timing country/hour identity.")
    if not data.label_eligible.dropna().isin([True, False]).all():
        raise ValueError("Explicit label timing eligibility mask required.")
    civil = data.timestamp_utc.dt.tz_convert("Europe/Paris")
    civil_day = civil.dt.tz_localize(None).dt.normalize()
    next_origin = (civil_day+pd.Timedelta(hours=8)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
    data["_delay_hours"] = (data.label_available_at_utc-next_origin).dt.total_seconds()/3600.
    data["_month"] = civil.dt.strftime("%Y-%m")
    data["_known"] = data.label_eligible.fillna(False).astype(bool) & data.label_available_at_utc.notna()
    end = date.fromisoformat(end_day)
    selected = data.loc[civil.dt.date.between(end-timedelta(days=days-1), end)
                        & data.timestamp_utc.isin(evaluation_timestamps)]
    def describe(part, zone):
        known = part.loc[part._known]
        late = known.loc[known._delay_hours.gt(0)]
        return {"zone": zone, "n_rows": len(part), "n_qualified_label_hours": len(known),
            "n_unknown_label_hours": len(part)-len(known), "n_delayed_label_hours": len(late),
            "delayed_fraction": len(late)/len(known) if len(known) else None,
            "n_distinct_qualified_timestamps": int(known.timestamp_utc.nunique()),
            "n_distinct_delayed_timestamps": int(late.timestamp_utc.nunique()),
            "maximum_delay_hours": max(0., float(known._delay_hours.max())) if len(known) else None,
            "maximum_delay_days": max(0., float(known._delay_hours.max()))/24. if len(known) else None,
            "latest_audited_label_availability_utc": known.label_available_at_utc.max()}
    rows, monthly = [], []
    for zone in ["ALL", *zones]:
        part = selected if zone == "ALL" else selected.loc[selected.zone.eq(zone)]
        rows.append(describe(part, zone))
        for month, group in part.groupby("_month", sort=True):
            monthly.append({"month": month, **describe(group, zone)})
    return _safe({"available": True, "rows": rows, "monthly": monthly,
        "late_definition": "label_available_at_utc > delivery-day 08:00 Europe/Paris, the next forecast origin",
        "unit": "country-hours plus distinct physical timestamps; no multiplication hidden",
        "limitation": "Audited availability may reflect a late revision watermark rather than first publication; training still obeys that conservative timestamp."})


def stage2_fit_diagnostics(wide, audit, *, root, seals, end_day, zones):
    """Read only folds sealed to the current result; never infer fit readiness from stage 1."""
    if not audit.get("snapshot"):
        return {"available": False, "case_fits": [], "readiness": [], "reason": "No sealed snapshot supplied."}
    directory = Path(audit["snapshot"]).resolve()
    directory.relative_to(Path(root).resolve()/"runs/experiments/nyx_congestion_v1")
    manifest_path, result_path = directory/"manifest.json", directory/"results_manifest.json"
    manifest_sha = _sha(manifest_path)
    if audit.get("suite_manifest_sha256") != manifest_sha:
        raise ValueError("Stage 2 report snapshot identity mismatch.")
    result = json.loads(result_path.read_text(encoding="utf8"))
    if result.get("status") != "completed" or result.get("suite_manifest_sha256") != manifest_sha:
        raise ValueError("Stage 2 folds require completed results of the same snapshot.")
    expected = result.get("result_files", {}).get("folds.parquet")
    path = directory/"folds.parquet"
    if not expected or _sha(path) != expected:
        raise ValueError("Stage 2 folds checksum mismatch.")
    seals.update({str(manifest_path): manifest_sha, str(result_path): _sha(result_path), str(path): expected})
    folds = pd.read_parquet(path)
    required = {"strategy", "fit_day", "status", "reason"}
    if required.difference(folds) or folds.duplicated(["strategy", "fit_day"]).any():
        raise ValueError("Invalid stage 2 fold identities.")
    if not set(folds.strategy).issubset({"control", "congestion"}):
        raise ValueError("Unexpected stage 2 fold strategy.")
    folds = folds.copy()
    folds["fit_day"] = pd.to_datetime(folds.fit_day, errors="raise").dt.strftime("%Y-%m-%d")
    end = date.fromisoformat(end_day)
    start = (end-timedelta(days=364)).isoformat()
    case_fits, readiness = [], []
    local = wide.timestamp_utc.dt.tz_convert("Europe/Paris")
    history = wide.loc[wide["sample"].eq("evaluation") & local.dt.date.between(date.fromisoformat(start), end)]
    case = wide.loc[wide["sample"].eq("evaluation") & local.dt.date.eq(date(2026, 9, 14)) & local.dt.hour.eq(19)]
    for strategy in ("control", "congestion"):
        prior = folds.loc[folds.strategy.eq(strategy) & folds.fit_day.le("2026-09-14")].sort_values("fit_day")
        annual = folds.loc[folds.strategy.eq(strategy) & folds.fit_day.between(start, end_day)]
        trained = prior.loc[prior.status.eq("trained")]
        latest = prior.iloc[-1].to_dict() if len(prior) else {}
        case_fits.append({"strategy": strategy, "fit_day": latest.get("fit_day"), "status": latest.get("status"),
            "reason": latest.get("reason"), "last_trained_fit_day": trained.fit_day.max() if len(trained) else None,
            "annual_fits": len(annual), "annual_trained_fits": int(annual.status.eq("trained").sum())})
        flag = strategy+"_expert_ready"
        for zone in ["ALL", *zones]:
            part = history if zone == "ALL" else history.loc[history.zone.eq(zone)]
            focus = case if zone == "ALL" else case.loc[case.zone.eq(zone)]
            readiness.append({"strategy": strategy, "zone": zone, "annual_country_hours": len(part),
                "annual_ready_country_hours": int(part[flag].fillna(False).astype(bool).sum()) if flag in part else None,
                "case_country_hours": len(focus),
                "case_ready_country_hours": int(focus[flag].fillna(False).astype(bool).sum()) if flag in focus else None})
    return _safe({"available": True, "case_day": "2026-09-14", "case_hour_paris": 19,
        "case_fits": case_fits, "readiness": readiness,
        "meaning": "Latest attempted residual fit at or before case day, not merely latest successful fit; stage 1 pressure can be ready while stage 2 abstains."})


def build_report(predictions, physical_source: Path, destination: Path, *, root: Path,
                 audit: dict, constraint_predictions: pd.DataFrame, zonal_labels: pd.DataFrame | None = None,
                 regional_predictions: pd.DataFrame | None = None):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.relative_to(root/"runs/experiments/nyx_congestion_v1")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Report destination is not empty.")
    long, wide, source_audit, seals = assemble_panel(predictions, physical_source)
    config = source_audit["source_config"]
    delivery = date.fromisoformat(config["delivery_day"])
    end_day = config.get("end_day") or (delivery-timedelta(days=1)).isoformat()
    if (date.fromisoformat(end_day) != delivery-timedelta(days=1) or config["evaluation_days"] != 365
            or config.get("timezone") != "Europe/Paris" or config.get("cutoff_time") != "08:00"):
        raise ValueError("Report requires strict 08:00 Paris, 365 days ending before sealed delivery.")
    zones = sorted(wide.zone.unique().tolist())
    if set(zones) != set(config["zones"]):
        raise ValueError("Report source zones mismatch.")
    civil = wide.timestamp_utc.dt.tz_convert("Europe/Paris").dt.date
    scored = long.loc[long["sample"].eq("evaluation")].copy()
    evaluation_times = wide.loc[wide["sample"].eq("evaluation"), "timestamp_utc"].unique()
    constraints = _constraints(constraint_predictions)
    origins = wide[["timestamp_utc", "forecast_origin_utc"]].drop_duplicates()
    if origins.duplicated("timestamp_utc").any():
        raise ValueError("Country forecast origins disagree.")
    match = constraints.merge(origins, on="timestamp_utc", how="left", suffixes=("", "_source"), validate="many_to_one")
    if (match.forecast_origin_utc_source.isna().any()
            or not match.forecast_origin_utc.eq(match.forecast_origin_utc_source).all()):
        raise ValueError("Constraint forecast support/origin differs from price panel.")
    if regional_predictions is not None and not regional_predictions.empty:
        regional = _constraints(regional_predictions.assign(constraint_key=regional_predictions.zone))
        matched = regional.merge(origins, on="timestamp_utc", how="left", suffixes=("", "_source"), validate="many_to_one")
        if (matched.forecast_origin_utc_source.isna().any()
                or not matched.forecast_origin_utc.eq(matched.forecast_origin_utc_source).all()):
            raise ValueError("Regional forecast support/origin differs from price panel.")
    econ_config = root/"config/economic_value.yaml"
    seals[str(econ_config)] = _sha(econ_config)
    periods = {}
    for days in (365, 7):
        result = compute_kpis(scored, end_day=end_day, days=days, zones=zones)
        result.pop("daily_rows", None)
        result["economic"] = compute_economic_kpis(scored, end_day=end_day, days=days, zones=zones, config_path=econ_config)
        result.update(price_diagnostics(wide, end_day=end_day, days=days, zones=zones))
        result["congestion"] = congestion_metrics(constraints, end_day=end_day, days=days, evaluation_timestamps=evaluation_times)
        result["domain_coverage"] = domain_coverage(zonal_labels, end_day=end_day, days=days,
                                                   zones=zones, evaluation_timestamps=evaluation_times)
        result["regional"] = regional_metrics(regional_predictions, end_day=end_day, days=days,
                                               zones=zones, evaluation_timestamps=evaluation_times)
        result["label_timing"] = label_timing(zonal_labels, end_day=end_day, days=days,
                                               zones=zones, evaluation_timestamps=evaluation_times)
        periods[str(days)] = result
    names = [item["id"] for item in CATALOG if item["id"] != "__storm__"]
    case_columns = [*KEYS, "actual", "storm", *names]
    case_columns += [name for name in wide if name.startswith(("control_", "congestion_")) and name not in case_columns]
    case = wide.loc[civil.eq(date(2026, 9, 14)), case_columns].copy()
    case["hour"] = case.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    stage2_fits = stage2_fit_diagnostics(wide, audit, root=root, seals=seals, end_day=end_day, zones=zones)
    payload = _safe({"schema_version": 1, "catalog": CATALOG, "zones": zones, "periods": periods,
        "stage2_fits": stage2_fits,
        "delivery_day": delivery, "end_day": end_day, "case": case.to_dict("records"),
        "constraint_case": constraint_case(constraints), "audit": audit, "source_audit": source_audit,
        "source_sha256": seals, "generated_at_utc": datetime.now(timezone.utc),
        "production_modified": False, "activation_performed": False, "independent_validation": False,
        "forecast_pit_certified": False, "economic_reference_executable": False,
        "label_contract": {"activation": "any strictly positive shadow price above source epsilon in the qualified market time units of the hour",
            "intensity": "hourly mean shadow price including inactive market time units, conditional on hourly activation",
            "market_time_units": "one hourly MTU before 2025-10-01; four quarter-hour MTUs from 2025-10-01",
            "activation_epsilon": 1e-9,
            "expected_shadow_price": "probability times conditional mean; explanatory signal, NOT a price P50",
            "average_precision": "step-weighted average precision, not trapezoidal PR area",
            "regional_pressure": "G=D-min(D) over all four CWE countries; severe G>=50, p*E[G|G>=50] is not E[G] overall and not a P50",
            "missing_labels": "excluded, never zero-filled", "geography": "CNEC-hours pooled once across the network, independent of selected country"}})
    for path, expected in seals.items():
        if _sha(path) != expected:
            raise ValueError("Report sources changed during calculation.")
    destination.mkdir(parents=True, exist_ok=True)
    metrics, report = destination/"metrics.json", destination/"nyx_congestion_report.html"
    with metrics.open("x", encoding="utf8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    render_report(payload, report)
    for path, expected in seals.items():
        if _sha(path) != expected:
            raise ValueError("Report sources changed during publication.")
    return {"status": "completed", "report_path": str(report), "metrics_path": str(metrics),
        "report_sha256": _sha(report), "metrics_sha256": _sha(metrics), "diagnostic_only": True,
        "production_modified": False, "summary": {
            "annual_rows": [row for row in periods["365"]["rows"] if row["zone"] == "ALL"],
            "annual_economic_rows": [row for row in periods["365"]["economic"]["rows"] if row["zone"] == "ALL"],
            "congestion": periods["365"]["congestion"], "coverage": periods["365"]["coverage"]}}


def render_report(payload, destination):
    encoded = json.dumps(_safe(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for old, new in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        encoded = encoded.replace(old, new)
    with Path(destination).open("x", encoding="utf8") as handle:
        handle.write(TEMPLATE.replace("@@DATA@@", encoded))
    return Path(destination)


TEMPLATE = r'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NYX · Congestion et P50</title><style>
:root{--bg:#f1f5f9;--paper:#fff;--ink:#1b2c43;--muted:#52657b;--line:#d8e1ec;--nyx:#b87504;--storm:#037eaf;--new:#7943bd;--obs:#27384d;--good:#14724c;--bad:#b33b46;--accent:#255abb}
:root[data-theme=dark]{--bg:#111a26;--paper:#1a283a;--ink:#ebf2fc;--muted:#b1c1d6;--line:#354961;--nyx:#ffc268;--storm:#67d0fa;--new:#c4a0ff;--obs:#eef4ff;--good:#7bddaa;--bad:#ff9ea8;--accent:#91b8ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,Segoe UI,sans-serif}main{max-width:1660px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;gap:18px;align-items:start}h1{font-size:31px;margin:4px 0}h2{font-size:19px;margin:0 0 10px}h3{font-size:15px;margin:10px 0}.eyebrow{color:var(--accent);font-size:11px;letter-spacing:2px;text-transform:uppercase}.muted,small{color:var(--muted)}button,select{font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:6px;background:var(--paper);color:var(--ink);max-width:100%}label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:3px}.filters{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}.card,.notice{background:var(--paper);padding:20px;border:1px solid var(--line);border-radius:10px;margin:18px 0}.notice{border-left:4px solid var(--accent)}.scroll{overflow-x:auto}table{width:100%;white-space:nowrap;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px;border-bottom:1px solid var(--line)}th{color:var(--muted)}th:first-child,td:first-child{text-align:left}tr.primary td:first-child{font-weight:700;color:var(--accent)}.good{color:var(--good)}.bad{color:var(--bad)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chart{min-width:0}.plot{width:100%}svg{width:100%;height:auto}svg text{fill:var(--muted);font:11px system-ui}.tip{min-height:25px;color:var(--muted);font-size:12px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px}.legend span:before{content:'━';margin-right:6px}.foot{font-size:12px;color:var(--muted)}.kpis{display:flex;flex-wrap:wrap;gap:25px}.metric{min-width:160px}.metric b{font-size:22px;display:block}pre{max-height:500px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:11px}code{font-size:12px}a{color:var(--accent)}@media(max-width:850px){main{padding:14px}.grid{grid-template-columns:1fr}header{flex-direction:column}h1{font-size:24px}}
</style></head><body><main><header><div><div class="eyebrow">Laboratoire isolé · comparaison chronologique</div><h1>Prévoir la congestion, puis ajuster le P50</h1><div id="subtitle" class="muted"></div></div><button id="theme">Mode nuit</button></header>
<div class="notice"><strong>Production inchangée, aucune promotion.</strong> Deux signaux complémentaires sont prévus à partir des informations antérieures à 08 h : activation/intensité des CNEC du domaine initial, et pression régionale globale incluant les contraintes publiées hors de cet univers dans les cibles historiques. Leurs prévisions historiques hors entraînement alimentent ensuite le correcteur de distribution. Le contrôle physique n'utilise aucun de ces signaux. L'année déjà analysée reste un diagnostic, pas une validation indépendante.</div>
<div class="filters"><label>Pays · scores de prix<select id="zone"></select></label><label>Période commune<select id="period"><option value="365">365 derniers jours</option><option value="7">7 derniers jours</option></select></label><label>P50 détaillé<select id="model"></select></label></div>
<section class="card"><h2>KPI · prix et qualité des prévisions</h2><p class="muted" id="coverage"></p><div class="scroll"><table id="kpi"><thead><tr><th>Modèle</th><th>MAE €/MWh</th><th>RMSE €/MWh</th><th>Win rate heure<br>vs Storm</th><th>Win rate MAE jour<br>vs Storm</th><th>Win rate prix moyen jour<br>vs Storm</th><th>MAE prix moyen jour<br>€/MWh</th><th>Prix moyen<br>€/MWh</th></tr></thead><tbody></tbody></table></div><p class="foot">Intersection exacte des heures physiques des huit modèles, observé inclus. Replis sur NYX inclus. Jours incomplets exclus uniquement des scores journaliers ; 23/24/25 heures selon DST. Un ex æquo n'est pas une victoire. Les couleurs comparent les erreurs à NYX sans conclure à une significativité statistique.</p></section>
<section class="card"><h2>EVA · décision économique identique</h2><p id="policy" class="muted"></p><div class="scroll"><table id="eva"><thead><tr><th>Modèle</th><th>P&amp;L net simulé €</th><th>Gain vs Storm €</th><th>Gain vs NYX €</th><th>Gain vs Storm<br>€/MWh potentiel</th><th>Heures-pays communes</th></tr></thead><tbody></tbody></table></div><p class="foot">Prix de référence = day-ahead observé D−1 à la même heure civile : proxy hypothétique, non exécutable à 08 h. Références absentes ou ambiguës exclues symétriquement. Politique, seuils, allocation et coûts inchangés ; aucune annualisation. Une hausse d'un signal BUY déjà déclenché n'augmente pas automatiquement le volume.</p></section>
<section class="card"><h2>1 · Prévision de l'activation des CNEC</h2><p class="muted" id="constraint-coverage"></p><p class="foot">Ces scores portent sur le réseau entier : le filtre pays ci-dessus ne les modifie pas. Un CNEC-heure est compté une seule fois, pas quatre fois. Comparaison expert/prior sur les mêmes lignes qualifiées et prêtes. Une publication manquante, une identité ambiguë ou une heure non qualifiée reste inconnue, jamais « inactive ».</p><div class="scroll"><table id="detection"><thead><tr><th>Signal</th><th>Brier ↓</th><th>AP ↑</th><th>Fréquence active</th><th>Activations</th><th>CNEC-heures</th></tr></thead><tbody></tbody></table></div><div class="grid"><div class="chart"><h3>Fiabilité des probabilités</h3><div id="reliability" class="plot"></div><div id="reliability-tip" class="tip"></div><p class="foot">X : probabilité moyenne ; Y : fréquence observée, dix classes fixes. La diagonale décrit une calibration parfaite. Classes vides omises, effectif disponible au survol.</p></div><div class="chart"><h3>Précision–rappel</h3><div id="pr" class="plot"></div><div id="pr-tip" class="tip"></div><p class="foot">X : rappel ; Y : précision. AP = average precision pondérée par les variations de rappel, distincte de la PR-AUC trapézoïdale. Valeur absente si une seule classe observée. Courbe simplifiée pour l'affichage, score exact sur toutes les lignes.</p></div></div><div class="legend" id="detection-legend"></div></section>
<section class="card"><h2>2 · Intensité conditionnelle et signal attendu</h2><div class="kpis" id="intensity"></div><div class="scroll"><table id="intensity-rmse"><thead><tr><th>Moyenne prédite</th><th>RMSE · €/MW</th><th>Échantillon</th></tr></thead><tbody></tbody></table></div><p class="foot">L'activation horaire signifie au moins une unité de marché (MTU) dont λ dépasse 10⁻⁹. Avant le 01/10/2025 : une MTU horaire ; ensuite : quatre quarts horaires. L'intensité réalisée est la moyenne des prix duaux de l'heure, y compris les MTU inactives. L'erreur active évalue E[λ | activation] sur les seules heures réellement actives. L'erreur globale évalue p × E[λ | activation] sur toutes les heures qualifiées. Ce produit est un signal explicatif : <strong>ce n'est ni le prix électrique, ni son P50</strong>. La RMSE complète la MAE car une moyenne, contrairement à une médiane, minimise l'erreur quadratique. Le zéro constant est un contrôle trivial affiché pour éviter de confondre rareté des activations et qualité.</p></section>
<section class="card"><h2>2 bis · Pression flow-based régionale globale</h2><p class="muted" id="regional-coverage"></p><p>Ce second expert vise les épisodes de pression réseau globale, y compris les contributions issues de CNEC non représentés dans le sous-ensemble initial. Sa cible historique est <code>G_pays = D_pays − min(D_FR, D_DE, D_BE, D_NL)</code>, où D agrège toutes les contributions duales publiées qualifiées. Les <strong>quatre pays</strong> sont nécessaires : ne pas recalculer le minimum sur trois pays. Ajouter une constante commune aux quatre D ne change pas G. Cette invariance retire aussi la composante commune : <strong>G ne détecte pas à lui seul une hausse de prix simultanée identique dans les quatre pays</strong>, qui reste à expliquer par les fondamentaux et le CGC.</p><div class="scroll"><table id="regional-table"><thead><tr><th>Signal</th><th>Brier ↓</th><th>AP ↑</th><th>Fréquence G ≥50</th><th>Heures-pays</th><th>RMSE pression<br>attendue · €/MWh</th><th>RMSE intensité<br>si G ≥50 · €/MWh</th><th>RMSE zéro<br>global · €/MWh</th></tr></thead><tbody></tbody></table></div><p class="foot">Le prior historique est la fréquence CORE <strong>groupée sur les quatre pays</strong>, non une fréquence propre à chaque pays. Une partie du gain face à ce prior peut donc refléter les différences géographiques ; la comparaison des prix au contrôle apparié reste essentielle. Des scores fondés sur très peu d'événements positifs dans un pays ne constituent pas une preuve robuste.</p><p class="foot">Seuil fixé avant évaluation : G ≥50 €/MWh. Le modèle prévoit sa probabilité et E[G | G ≥50] ; cette moyenne conditionnelle est bornée inférieurement à 50 €/MWh pour respecter le support de l'événement, même si le CGC baisse. Leur produit estime la pression <em>sévère</em> attendue, pas E[G] toutes intensités confondues ; il peut être inférieur à 50 si la probabilité est faible. Les faibles pressions ont une cible d'intensité nulle dans ce diagnostic. Ce n'est pas une activation CNEC, pas un prix électrique observé, et pas un P50. Ces signaux ne sont pas ajoutés manuellement au prix.</p><h3>14 septembre · 19 h · pression régionale par pays</h3><div class="scroll"><table id="regional-case"><thead><tr><th>Pays</th><th>G réalisé<br>non tronqué · €/MWh</th><th>Événement G ≥50</th><th>Probabilité prévue</th><th>E[G | G ≥50]<br>prévue · €/MWh</th><th>Pression sévère<br>attendue · €/MWh</th><th>Expert prêt</th></tr></thead><tbody></tbody></table></div></section>
<section class="card"><h2>Le signal existe-t-il, et le correcteur peut-il l'utiliser ?</h2><p>Un expert régional peut prévoir un risque élevé alors que le correcteur aval n'est pas entraînable ou prêt. Dans ce cas, le P50 reste celui de NYX : ce n'est ni une correction manuelle oubliée, ni la preuve que le signal régional était faible. Le tableau montre le <strong>dernier fit tenté avant le 14 septembre</strong>, y compris les replis, et pas seulement le dernier fit réussi.</p><div class="scroll"><table id="stage2-fits"><thead><tr><th>Correcteur aval</th><th>Dernier fit tenté</th><th>État</th><th>Raison enregistrée</th><th>Dernier fit entraîné</th><th>Fits entraînés / tentés<br>fenêtre annuelle</th></tr></thead><tbody></tbody></table></div><div class="scroll"><table id="stage2-ready"><thead><tr><th>Correcteur aval</th><th>Heures-pays prêtes<br>fenêtre annuelle</th><th>Heures-pays évaluées<br>fenêtre annuelle</th><th>Prêtes au 14/09 à 19 h</th><th>Heures-pays du cas</th></tr></thead><tbody></tbody></table></div><p class="foot">Ces diagnostics restent sur la fenêtre annuelle et le cas du 14 septembre ; seul le pays sélectionné filtre les disponibilités. Ils distinguent apprentissage du signal, calibration du correcteur et gouvernance. Un repli récent ne signifie pas que le dernier modèle ancien est automatiquement réutilisé. Les raisons proviennent des folds du résultat courant, vérifiés par SHA.</p></section>
<section class="card"><h2>3 · Interventions du correcteur de prix</h2><div class="scroll"><table id="changes"><thead><tr><th>Modèle</th><th>Heures changées</th><th>Améliorées</th><th>Dégradées</th><th>MAE prix ≥200</th><th>RMSE prix ≥200</th><th>MAE prix &lt;200</th></tr></thead><tbody></tbody></table></div><p class="foot">Prix observé ≥200 €/MWh : segmentation descriptive après réalisation, jamais une variable de déclenchement. Le contrôle physique réapprend son détecteur et sa distribution sans signal de congestion prédit ; les variantes précédentes restent figées.</p><h3>MAE par heure civile</h3><div id="hour-plot" class="plot"></div><div id="hour-tip" class="tip"></div></section>
<section class="card"><h2>14 septembre 2026 · prix et correction à 19 h</h2><div class="scroll"><table id="case"><thead><tr><th>Pays</th><th>Observé</th><th>Storm</th><th>NYX</th><th>Réseau + CGC<br>direct</th><th>Contrôle<br>direct</th><th>Contrôle<br>gouverné</th><th>Congestion<br>direct</th><th>Congestion<br>gouverné</th></tr></thead><tbody></tbody></table></div><div id="price-legend" class="legend"></div><div id="case-charts" class="grid"></div></section>
<section class="card"><h2>14 septembre · activation prévue et réalisée</h2><p class="muted">Validation ex post des contraintes représentées dans le domaine initial, pas observation disponible au forecast.</p><label>Contrainte suivie<select id="constraint"></select></label><p id="constraint-selection" class="foot"></p><div class="grid"><div class="chart"><h3>Probabilité prévue et activation observée</h3><div id="activation-plot" class="plot"></div><div id="activation-tip" class="tip"></div></div><div class="chart"><h3>Prix dual prévu et réalisé · €/MW</h3><div id="intensity-plot" class="plot"></div><div id="intensity-tip" class="tip"></div></div></div><div id="activation-legend" class="legend"></div><p class="foot">Barres = activation effectivement constatée après coupling ; ligne = probabilité produite sans cette réalisation. Une case vide représente une valeur inconnue ou une absence d'expert prêt, pas zéro. Choix d'affichage : huit contraintes avec le signal attendu le plus élevé à 19 h, plus quatre avec l'intensité réalisée qualifiée la plus élevée. Cette sélection ex post ne change aucun entraînement ni aucune règle.</p></section>
<section class="card"><h2>Couverture du domaine · ce que l'expert ne voit pas</h2><p>Le sous-ensemble initial <em>Presolved</em> ne contient pas forcément tous les CNEC qui deviendront actifs après coupling. Le 14 septembre à 19 h, la contrainte Vigy exacte active après coupling n'était pas représentée dans le sous-ensemble initial sauvegardé ; sa paire inverse n'est pas le même CNEC. Une absence n'est pas un négatif valide. Les coûts duaux × écarts PTDF expliquent une composante des écarts entre pays, pas à eux seuls le niveau absolu du prix ni l'identité de la centrale marginale.</p><h3>Couverture des contributions absolues · période sélectionnée</h3><p id="domain-status" class="muted"></p><div class="scroll"><table id="domain-table"><thead><tr><th>Pays</th><th>Heures qualifiées</th><th>Non qualifiées</th><th>Σ absolue totale</th><th>Σ couverte</th><th>Σ hors initial</th><th>Couverture</th></tr></thead><tbody></tbody></table></div><h3>14 septembre · 19 h · contributions horaires €/MWh</h3><div class="scroll"><table id="domain-case"><thead><tr><th>Pays</th><th>Absolue totale</th><th>Couverte</th><th>Hors initial</th><th>Couverture</th></tr></thead><tbody></tbody></table></div><p class="foot">Audit ex post de |λ × (PTDF FR − PTDF pays)|, référence FR uniquement ici. Le ratio est la somme des contributions absolues couvertes divisée par la somme totale, pas la moyenne de ratios horaires. Il est indéfini pour FR (dénominateur nul), donc jamais affiché comme 0 %. Les sommes de contributions horaires ne sont ni un prix moyen, ni un P&amp;L. Les variables du modèle emploient une référence <strong>symétrique moyenne CWE</strong>, différente de cet audit. Heures sans domaine initial qualifié ou rapprochement complet exclues ; contributions externes identifiées incluses au dénominateur.</p><details><summary>Couverture, heures inconnues et contraintes actives hors univers</summary><pre id="domain-audit"></pre></details></section>
<section class="card"><h2>Disponibilité des labels · peut-on réellement apprendre à cette date ?</h2><p id="timing-summary" class="muted"></p><div class="scroll"><table id="timing"><thead><tr><th>Pays</th><th>Labels qualifiés<br>heures-pays</th><th>Après origine suivante<br>heures-pays</th><th>Part tardive</th><th>Heures physiques<br>tardives distinctes</th><th>Retard maximal<br>jours</th></tr></thead><tbody></tbody></table></div><details><summary>Détail par mois de livraison</summary><div class="scroll"><table id="timing-month"><thead><tr><th>Mois</th><th>Labels qualifiés</th><th>Tardifs</th><th>Part tardive</th><th>Retard maximal<br>jours</th></tr></thead><tbody></tbody></table></div></details><p class="foot">Un label est tardif ici si sa disponibilité auditée est postérieure à <strong>08 h Europe/Paris le jour de sa livraison</strong>, origine du forecast de livraison suivante. Le calcul respecte l'heure civile et DST. L'horodatage peut refléter une révision de l'archive, pas nécessairement la première publication au marché. L'apprentissage respecte néanmoins cet horodatage conservateur : une couverture historique complète ne signifie pas que ces labels étaient admissibles au fit d'alors. Échauffement et disponibilités tardives peuvent réduire les fenêtres de calibration utilisables et la couverture des prévisions prêtes. Cette archive n'est pas une validation prospective indépendante.</p></section>
<section class="card"><h2>Méthode et limites</h2><div class="grid"><div><p><strong>Deux étapes chronologiques.</strong> L'expert réseau apprend l'activité et une intensité normalisée par le coût gaz propre, avec calibration temporelle des probabilités. Le correcteur de prix n'apprend que sur des signaux réseau antérieurement produits hors entraînement. Les périodes d'échauffement et les abstentions restent comptées dans le résultat final.</p><p><strong>Un véritable P50.</strong> Les signaux de congestion conditionnent une distribution d'erreur à deux régimes ; son quantile 0,5 fournit la proposition. Une moyenne conditionnelle, p × amplitude ou une correction optimisant la RMSE n'est pas renommée P50. Les bornes d'intervalle et la gouvernance sont calibrées sur le passé disponible.</p><p><strong>Contrôle utile.</strong> Comparer congestion direct à contrôle direct isole la chaîne contenant le signal prédit dans le protocole fixé ; la gouvernance peut ensuite réduire ou annuler les interventions. Aucune règle par pays n'est choisie après lecture du 14 septembre.</p></div><div><p><strong>Cutoff strict de 08 h.</strong> Les variables proviennent du domaine initial et des fondamentaux connus avant le cutoff. FinalComputation et shadowPrices servent uniquement à construire les cibles historiques après publication et à auditer le résultat. Les valeurs réalisées du jour ne sont pas injectées dans sa prévision.</p><p><strong>Historique récupéré ≠ capture temps réel.</strong> Un watermark antérieur au cutoff ne certifie pas à lui seul un vintage immuable. Les prix de référence NYX/Storm et les données réseau restent soumis aux limites d'audit documentées. Les preuves de couverture des quarts d'heure et d'identité des CNEC sont requises ; la non-publication n'est jamais remplie arbitrairement.</p><p><strong>Pas de garantie de gain.</strong> Ce laboratoire doit battre le contrôle sur la même période sans multiplier les fausses corrections. Cette année a déjà guidé des hypothèses : seule une nouvelle période future gelée peut fournir une validation indépendante. Production et commandes habituelles inchangées, aucune activation automatique.</p></div></div><details><summary>Audit intégral, paramètres et sources scellées</summary><pre id="audit"></pre></details></section>
<noscript>JavaScript est nécessaire pour filtrer ce rapport. Toutes les métriques sont aussi dans metrics.json.</noscript></main><script id="congestion-data" type="application/json">@@DATA@@</script><script>
'use strict';const D=JSON.parse(document.getElementById('congestion-data').textContent),$=id=>document.getElementById(id),labels=Object.fromEntries(D.catalog.map(m=>[m.id,m.label]));Object.assign(labels,{congestion:'Expert congestion',regional:'Pression régionale sévère',past_prior:'Prior historique'});
const fmt=(v,n=2)=>Number.isFinite(v)?v.toLocaleString('fr-FR',{minimumFractionDigits:n,maximumFractionDigits:n}):'—',pct=v=>Number.isFinite(v)?fmt(v*100)+' %':'—',color=k=>getComputedStyle(document.documentElement).getPropertyValue('--'+k).trim();
function opt(id,value,label){const e=document.createElement('option');e.value=value;e.textContent=label;$(id).append(e)}opt('zone','ALL','ALL · pays agrégés');D.zones.forEach(z=>opt('zone',z,z));D.catalog.filter(m=>m.kind==='candidate'||m.kind==='control').forEach(m=>opt('model',m.id,m.label));$('model').value='nyx_congestion';
function td(tr,text,cl=''){const e=document.createElement('td');e.textContent=text;e.className=cl;tr.append(e)}function cls(v,b,lower=true){return Number.isFinite(v)&&Number.isFinite(b)&&Math.abs(v-b)>1e-9?((lower?v<b:v>b)?'good':'bad'):''}function table(id,rows,fn){const body=$(id).querySelector('tbody');body.replaceChildren();rows.forEach(r=>{const tr=document.createElement('tr');if(r.model_id==='nyx_congestion')tr.className='primary';td(tr,labels[r.model_id]||r.zone||r.model_id);fn(tr,r);body.append(tr)})}const find=(rows,z,m)=>rows.find(r=>r.zone===z&&r.model_id===m)||{};
function legend(id,series){$(id).replaceChildren();series.forEach(s=>{const e=document.createElement('span');e.textContent=s.label;e.style.color=s.color;$(id).append(e)})}
function chart(id,tip,series,{fixed=false,hour=false,aria='Graphique de diagnostic'}={}){const host=$(id);host.replaceChildren();$(tip).textContent='';const points=series.flatMap(s=>s.points.filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)));if(!points.length){host.textContent='Aucune donnée qualifiée.';return}const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 680 280');svg.setAttribute('role','img');svg.setAttribute('aria-label',aria);host.append(svg);let xmin=hour?0:Math.min(...points.map(p=>p.x)),xmax=hour?23:Math.max(...points.map(p=>p.x)),ymin=fixed?0:Math.min(0,...points.map(p=>p.y)),ymax=fixed?1:Math.max(...points.map(p=>p.y));if(fixed&&!hour){xmin=0;xmax=1}if(xmin===xmax)xmax=xmin+1;if(ymin===ymax)ymax=ymin+1;const x=v=>60+(v-xmin)*595/(xmax-xmin),y=v=>235-(v-ymin)*210/(ymax-ymin);function el(tag,attrs,text){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e}for(let i=0;i<5;i++){const v=ymin+(ymax-ymin)*i/4;el('line',{x1:60,x2:655,y1:y(v),y2:y(v),stroke:color('line')});el('text',{x:52,y:y(v)+4,'text-anchor':'end'},fmt(v,fixed?2:0));const u=xmin+(xmax-xmin)*i/4;el('text',{x:x(u),y:259,'text-anchor':'middle'},hour?Math.round(u)+' h':fmt(u,2))}series.forEach(s=>{let path='',opened=false;s.points.forEach(p=>{if(!Number.isFinite(p.y)||!Number.isFinite(p.x)){opened=false;return}if(s.bar){el('rect',{x:x(p.x)-6,y:y(p.y),width:12,height:Math.max(0,y(0)-y(p.y)),fill:s.color,opacity:.3})}else{path+=(opened?'L':'M')+x(p.x)+','+y(p.y)+' ';opened=true;el('circle',{cx:x(p.x),cy:y(p.y),r:2.5,fill:s.color})}});if(!s.bar)el('path',{d:path,fill:'none',stroke:s.color,'stroke-width':2,'stroke-dasharray':s.dashed?'5 4':'none'})});svg.addEventListener('pointermove',e=>{const rect=svg.getBoundingClientRect(),target=xmin+(((e.clientX-rect.left)*680/rect.width-60)/595)*(xmax-xmin);const message=series.filter(s=>!s.dashed).map(s=>{const valid=s.points.filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));if(!valid.length)return s.label+': —';const p=valid.reduce((a,b)=>Math.abs(a.x-target)<Math.abs(b.x-target)?a:b);return s.label+': '+(hour?fmt(p.x,0)+' h · ': 'x='+fmt(p.x,3)+' · ')+fmt(p.y,fixed?3:2)+(p.n!==undefined?' · n='+fmt(p.n,0):'')});$(tip).textContent=message.join(' | ')})}
function hourSeries(rows,specs){return specs.map(s=>({...s,points:rows.map(r=>({x:r.hour,y:r[s.id]}))}))}
const caseRows=D.constraint_case.rows||[];(D.constraint_case.constraint_keys||[]).forEach(key=>{const r=caseRows.find(r=>r.constraint_key===key)||{};opt('constraint',key,(r.cne_name||key)+(r.contingency_name?' · '+r.contingency_name:'')+(r.direction?' · '+r.direction:''))});
function renderIntensity(){const i=D.periods[$('period').value].congestion.intensity;table('intensity-rmse',[{model_id:'Signal attendu p × Eλ',value:i.global_rmse_expected_shadow_price,n:i.n_rows},{model_id:'Intensité conditionnelle Eλ',value:i.active_rmse_conditional_intensity,n:i.n_active},{model_id:'Contrôle zéro',value:i.zero_prediction_global_rmse,n:i.n_rows}],(tr,r)=>{td(tr,fmt(r.value));td(tr,fmt(r.n,0))})}
function renderDomain(){const c=D.periods[$('period').value].domain_coverage,z=$('zone').value;$('domain-status').textContent=c.available?'Même période civile, heures dont les labels et le domaine initial sont qualifiés.':(c.reason||'Couverture non disponible.');table('domain-table',(c.rows||[]).filter(r=>r.zone===z),(tr,r)=>{td(tr,fmt(r.n_qualified_hours,0));td(tr,fmt(r.n_unqualified_hours,0));['total_absolute_contribution_sum','covered_absolute_contribution_sum','outside_absolute_contribution_sum'].forEach(k=>td(tr,fmt(r[k])));td(tr,pct(r.coverage_ratio))});table('domain-case',(c.case||[]).filter(r=>z==='ALL'||r.zone===z),(tr,r)=>{['label_absolute_contribution_eur_mwh','initial_covered_absolute_contribution_eur_mwh','outside_initial_absolute_contribution_eur_mwh'].forEach(k=>td(tr,fmt(r[k])));td(tr,pct(r.coverage_ratio))})}
function renderRegional(){const r=D.periods[$('period').value].regional,z=$('zone').value,c=(r.coverage||[]).find(c=>c.zone===z)||{};$('regional-coverage').textContent=r.available?fmt(c.n_common_scored,0)+' heures-pays appariées expert/prior ; '+fmt(c.n_label_eligible,0)+' labels qualifiés et '+fmt(c.n_expert_ready,0)+' prévisions prêtes. Échantillon distinct des CNEC-heures.':(r.reason||'Expert régional non disponible.');table('regional-table',(r.rows||[]).filter(v=>v.zone===z),(tr,v)=>{td(tr,fmt(v.brier,5));td(tr,fmt(v.average_precision,4));td(tr,pct(v.prevalence));td(tr,fmt(v.n_rows,0));['global_rmse_expected_pressure','active_rmse_conditional_pressure','zero_prediction_global_rmse'].forEach(k=>td(tr,fmt(v[k])))});table('regional-case',(r.case||[]).filter(v=>z==='ALL'||v.zone===z),(tr,v)=>{td(tr,fmt(v.realised_fb_premium));td(tr,v.label_active===null?'—':v.label_active?'Oui':'Non');td(tr,v.expert_ready?pct(v.activation_probability):'—');td(tr,v.expert_ready?fmt(v.intensity_if_active):'—');td(tr,v.expert_ready?fmt(v.expected_shadow_price):'—');td(tr,v.expert_ready?'Oui':'Non')})}
function renderTiming(){const d=D.periods[$('period').value].label_timing,z=$('zone').value;$('timing-summary').textContent=d.available?'Disponibilité des versions archivées, calculée sur la période sélectionnée. Aucun label n’est avancé artificiellement dans le temps.':(d.reason||'Audit temporel non disponible.');table('timing',(d.rows||[]).filter(r=>r.zone===z),(tr,r)=>{td(tr,fmt(r.n_qualified_label_hours,0));td(tr,fmt(r.n_delayed_label_hours,0));td(tr,pct(r.delayed_fraction));td(tr,fmt(r.n_distinct_delayed_timestamps,0));td(tr,fmt(r.maximum_delay_days))});table('timing-month',(d.monthly||[]).filter(r=>r.zone===z).map(r=>({...r,model_id:r.month})),(tr,r)=>{td(tr,fmt(r.n_qualified_label_hours,0));td(tr,fmt(r.n_delayed_label_hours,0));td(tr,pct(r.delayed_fraction));td(tr,fmt(r.maximum_delay_days))})}
function renderConstraint(){const key=$('constraint').value,rows=caseRows.filter(r=>r.constraint_key===key).sort((a,b)=>a.hour-b.hour);const observed=rows.map(r=>({...r,active_plot:r.label_eligible&&r.label_active!==null?Number(r.label_active):null,actual_intensity:r.label_eligible?r.label_shadow_price:null,p_plot:r.expert_ready?r.activation_probability:null,mean_plot:r.expert_ready?r.expected_shadow_price:null}));const specs=[{id:'active_plot',label:'Activation observée · barres',color:color('obs'),bar:true},{id:'p_plot',label:'Probabilité prévue',color:color('new')}];chart('activation-plot','activation-tip',hourSeries(observed,specs),{fixed:true,hour:true,aria:'Probabilité prévue et activation observée après publication'});const intens=[{id:'actual_intensity',label:'Prix dual observé',color:color('obs')},{id:'mean_plot',label:'Prix dual attendu p × moyenne',color:color('new')}];chart('intensity-plot','intensity-tip',hourSeries(observed,intens),{hour:true,aria:'Prix duaux horaires prévus et réalisés'});legend('activation-legend',specs);$('constraint-selection').textContent=key?'Identité stricte : '+key:'Aucune contrainte disponible pour cette journée.'}
function render(){const z=$('zone').value,m=$('model').value,p=D.periods[$('period').value],rows=p.rows.filter(r=>r.zone===z),base=find(rows,z,'nuclear_kalman');$('subtitle').textContent=p.period.start_day+' → '+p.period.end_day+' · livraison '+D.delivery_day+' non évaluée · cutoff 08 h Europe/Paris';const cov=p.coverage.filter(r=>z==='ALL'||r.zone===z),sum=k=>cov.reduce((a,r)=>a+(r[k]||0),0);$('coverage').textContent=fmt(sum('n_common_hours'),0)+' / '+fmt(sum('n_expected_hours'),0)+' heures-pays communes · '+fmt(sum('n_complete_days'),0)+' jours-pays complets · observé moyen '+fmt(base.observed_mean_price_eur_mwh)+' €/MWh';table('kpi',rows,(tr,r)=>{['mae_eur_mwh','rmse_eur_mwh'].forEach(k=>td(tr,fmt(r[k]),cls(r[k],base[k])));['win_rate_hour_pct','win_rate_day_mae_pct','win_rate_day_mean_price_pct'].forEach(k=>td(tr,Number.isFinite(r[k])?fmt(r[k])+' %':'—',cls(r[k],50,false)));td(tr,fmt(r.mae_day_mean_price_eur_mwh),cls(r.mae_day_mean_price_eur_mwh,base.mae_day_mean_price_eur_mwh));td(tr,fmt(r.mean_price_eur_mwh))});const ea=p.economic.audit,eb=find(p.economic.rows,z,'nuclear_kalman');$('policy').textContent='Portefeuille alternatif '+fmt(ea.portfolio_capacity_mw,0)+' MW ; '+Object.entries(ea.zone_capacity_mw).map(([a,b])=>a+' '+fmt(b,0)+' MW').join(', ')+' fixes. Seuil |edge| > '+fmt(ea.signal_hurdle_eur_mwh)+' €/MWh ; coûts '+fmt(ea.net_cost_eur_mwh)+' €/MWh.';table('eva',p.economic.rows.filter(r=>r.zone===z),(tr,r)=>{td(tr,fmt(r.pnl_net_eur,0));td(tr,fmt(r.gain_vs_storm_eur,0),cls(r.gain_vs_storm_eur,0,false));const gain=Number.isFinite(r.pnl_net_eur)&&Number.isFinite(eb.pnl_net_eur)?r.pnl_net_eur-eb.pnl_net_eur:null;td(tr,fmt(gain,0),cls(gain,0,false));td(tr,fmt(r.gain_vs_storm_per_potential_mwh));td(tr,fmt(r.n_country_hours,0))});
const cg=p.congestion,cc=cg.coverage;$('constraint-coverage').textContent=fmt(cc.n_common_scored,0)+' / '+fmt(cc.n_initial_constraint_hours,0)+' CNEC-heures évaluées · '+fmt(cc.n_label_eligible,0)+' labels qualifiés · '+fmt(cc.n_expert_ready,0)+' expert prêt · '+fmt(cc.n_unknown_or_unqualified_labels,0)+' labels inconnus/non qualifiés · '+fmt(cc.n_qualified_but_not_common,0)+' qualifiés hors comparaison.';table('detection',cg.scores,(tr,r)=>{td(tr,fmt(r.brier,5));td(tr,fmt(r.average_precision,4));td(tr,pct(r.prevalence));td(tr,fmt(r.n_active,0));td(tr,fmt(r.n_rows,0))});const ds=[{id:'congestion',label:'Expert congestion',color:color('new')},{id:'past_prior',label:'Prior historique',color:color('storm')}];chart('reliability','reliability-tip',[{label:'Calibration idéale',color:color('muted'),dashed:true,points:[{x:0,y:0},{x:1,y:1}]},...ds.map(s=>({...s,points:cg.reliability.filter(r=>r.model_id===s.id&&r.n_rows>0).map(r=>({x:r.mean_probability,y:r.observed_frequency,n:r.n_rows}))}))],{fixed:true,aria:'Calibration des probabilités de congestion'});chart('pr','pr-tip',ds.map(s=>{const row=cg.pr_curves.find(r=>r.model_id===s.id);return {...s,points:row?row.recall.map((x,i)=>({x,y:row.precision[i]})):[]}}),{fixed:true,aria:'Courbes précision rappel de congestion'});legend('detection-legend',ds);$('intensity').replaceChildren();[['active_mae_conditional_intensity','MAE intensité · heures actives'],['global_mae_expected_shadow_price','MAE signal attendu · global'],['zero_prediction_global_mae','MAE contrôle zéro · global'],['n_active','Heures actives communes']].forEach(([key,label])=>{const e=document.createElement('div');e.className='metric';const value=document.createElement('b');value.textContent=fmt(cg.intensity[key],key==='n_active'?0:2);const caption=document.createElement('span');caption.className='muted';caption.textContent=label+(key==='n_active'?'':' · €/MW');e.append(value,caption);$('intensity').append(e)});
table('changes',p.interventions.filter(r=>r.zone===z&&r.model_id!=='__storm__'),(tr,r)=>{['changed_hours','better','worse'].forEach(k=>td(tr,fmt(r[k],0)));['tail_mae','tail_rmse','normal_mae'].forEach(k=>td(tr,fmt(r[k])))});const specs=[{id:'actual',label:'Observé',color:color('obs')},{id:'storm',label:'Storm',color:color('storm')},{id:'nuclear_kalman',label:'NYX',color:color('nyx')},{id:m,label:labels[m],color:color('new')}];chart('hour-plot','hour-tip',specs.slice(1).map(s=>({...s,points:p.hourly.filter(r=>r.zone===z&&r.model_id===(s.id==='storm'?'__storm__':s.id)).map(r=>({x:r.hour,y:r.mae}))})),{hour:true,aria:'Erreur absolue moyenne par heure de la journée'});const cases=D.case.filter(r=>z==='ALL'||r.zone===z);table('case',cases.filter(r=>r.hour===19),(tr,r)=>{['actual','storm','nuclear_kalman','network_fuel_direct','congestion_control_direct','congestion_control_governed','congestion_direct','nyx_congestion'].forEach(k=>td(tr,fmt(r[k])))});legend('price-legend',specs);$('case-charts').replaceChildren();D.zones.filter(a=>z==='ALL'||a===z).forEach(zone=>{const box=document.createElement('div');box.className='chart';const title=document.createElement('h3');title.textContent=zone;const host=document.createElement('div');host.id='case-'+zone;const tip=document.createElement('div');tip.id='tip-'+zone;tip.className='tip';box.append(title,host,tip);$('case-charts').append(box);chart(host.id,tip.id,hourSeries(cases.filter(r=>r.zone===zone).sort((a,b)=>a.hour-b.hour),specs),{hour:true,aria:'Prix horaires du 14 septembre en '+zone})});renderConstraint();$('domain-audit').textContent=JSON.stringify(D.audit.data_audit||D.audit.label_audit||D.audit.constraint_audit||D.audit,null,2);$('audit').textContent=JSON.stringify({audit:D.audit,label_contract:D.label_contract,source_sha256:D.source_sha256,production_modified:false,forecast_pit_certified:false,independent_validation:false},null,2)}
function renderStage2(){const d=D.stage2_fits||{},z=$('zone').value,model=s=>s==='control'?'Contrôle physique':'Avec signaux congestion';table('stage2-fits',(d.case_fits||[]).map(r=>({...r,model_id:model(r.strategy)})),(tr,r)=>{td(tr,r.fit_day||'—');td(tr,r.status||'—');td(tr,r.reason||'—');td(tr,r.last_trained_fit_day||'—');td(tr,fmt(r.annual_trained_fits,0)+' / '+fmt(r.annual_fits,0))});table('stage2-ready',(d.readiness||[]).filter(r=>r.zone===z).map(r=>({...r,model_id:model(r.strategy)})),(tr,r)=>{['annual_ready_country_hours','annual_country_hours','case_ready_country_hours','case_country_hours'].forEach(k=>td(tr,fmt(r[k],0)))})}
['zone','period','model'].forEach(id=>$(id).addEventListener('change',()=>{render();renderIntensity();renderDomain();renderRegional();renderTiming();renderStage2()}));$('constraint').addEventListener('change',renderConstraint);$('theme').addEventListener('click',()=>{const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';$('theme').textContent=dark?'Mode jour':'Mode nuit';render();renderIntensity();renderDomain();renderRegional();renderTiming();renderStage2()});render();renderIntensity();renderDomain();renderRegional();renderTiming();renderStage2();
</script></body></html>'''
