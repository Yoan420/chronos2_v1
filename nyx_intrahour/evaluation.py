"""Isolated point correction of NYX P50; no production or datasource imports.

Features must already be forecasts available at their delivery cutoff. This module
cannot establish vintages. Labels become usable at D-2 (Paris civil days), and
validation selects the recipe before any final-test scoring. No quantiles are made.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ZONES = ("BE", "DE", "FR", "NL")
ALPHAS = (10.0, 100.0)
SHAPE_SUFFIXES = (
    "__std_gw", "__range_gw", "__ramp_gw_per_hour",
    "__max_deviation_gw", "__min_deviation_gw",
)
CALENDAR = (
    "calendar_hour_sin", "calendar_hour_cos", "calendar_week_sin",
    "calendar_week_cos", "calendar_year_sin", "calendar_year_cos",
    "calendar_utc_offset_hours",
)
TABLES = ("predictions", "metrics", "paired_deltas", "fit_audit", "regime_checks")


def _unavailable(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "status": "insufficient_data", "reason": reason, "details": details,
        **{name: pd.DataFrame() for name in TABLES},
        "selection": {"status": "not_selected", "reason": reason},
        "decision": {"status": "insufficient_data", "encouraging": False,
                     "promotion_allowed": False, "reasons": [reason]},
    }


def _scores(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    good = np.isfinite(y) & np.isfinite(prediction)
    error = prediction[good] - y[good]
    if not len(error):
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    return {"n": len(error), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean())}


def _prepare(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp_utc", "zone", "actual", "training_actual", "nyx_q50"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {', '.join(sorted(missing))}")
    if panel.columns.duplicated().any():
        raise ValueError("Duplicate panel column names")
    frame = panel.copy(deep=True)
    try:
        timestamps = pd.to_datetime(frame.timestamp_utc)
        if timestamps.dt.tz is None:
            raise ValueError("timestamp_utc must be timezone aware")
        frame["timestamp_utc"] = timestamps.dt.tz_convert("UTC")
    except (AttributeError, TypeError) as exc:
        raise ValueError("timestamp_utc must contain timezone-aware timestamps") from exc
    if frame.timestamp_utc.isna().any():
        raise ValueError("Missing timestamp_utc")
    if (frame.timestamp_utc != frame.timestamp_utc.dt.floor("h")).any():
        raise ValueError("Evaluation targets must be physical hourly timestamps")
    if frame.zone.isna().any() or not set(frame.zone).issubset(ZONES):
        raise ValueError("Expected zones BE, DE, FR, NL only")
    if frame.duplicated(["zone", "timestamp_utc"]).any():
        raise ValueError("Duplicate zone/timestamp target; refusing repeated history")
    numeric = [c for c in frame if c in required - {"timestamp_utc", "zone"}
               or c.startswith(("feature_hourly_", "feature_intrahour_"))]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
    local = frame.timestamp_utc.dt.tz_convert("Europe/Paris")
    frame["delivery_day"] = local.dt.tz_localize(None).dt.normalize()
    frame["local_hour"] = local.dt.hour
    for name, values, period in (
        ("hour", local.dt.hour, 24), ("week", local.dt.dayofweek, 7),
        ("year", local.dt.dayofyear - 1, 365.2425),
    ):
        frame[f"calendar_{name}_sin"] = np.sin(2 * np.pi * values / period)
        frame[f"calendar_{name}_cos"] = np.cos(2 * np.pi * values / period)
    frame["calendar_utc_offset_hours"] = [t.utcoffset().total_seconds() / 3600 for t in local]
    return frame.sort_values(["timestamp_utc", "zone"]).reset_index(drop=True)


def _predict(
    frame: pd.DataFrame, stage: str, first_day: pd.Timestamp, last_day: pd.Timestamp,
    family: str, alpha: float, columns: list[str], shapes: list[str], *,
    window_days: int, refit_every_days: int, min_train_rows: int,
    fit_audit: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    wanted = frame.delivery_day.between(first_day, last_day)
    indices = np.flatnonzero(wanted)
    prediction = frame.nyx_q50.to_numpy(float).copy()
    reasons = np.full(len(frame), "", dtype=object)
    for refit_day in pd.date_range(first_day, last_day, freq=f"{refit_every_days}D"):
        last_label_day = refit_day - pd.Timedelta(days=2)
        first_label_day = last_label_day - pd.Timedelta(days=window_days - 1)
        block_end = min(last_day, refit_day + pd.Timedelta(days=refit_every_days - 1))
        train_window = frame.delivery_day.between(first_label_day, last_label_day)
        block = frame.delivery_day.between(refit_day, block_end)
        for zone in ZONES:
            train = frame.loc[train_window & frame.zone.eq(zone)]
            train = train.loc[np.isfinite(train.training_actual) & np.isfinite(train.nyx_q50)]
            target_indices = frame.index[block & frame.zone.eq(zone)].to_numpy()
            x = train[columns].to_numpy(float)
            complete_train = np.isfinite(x).all(axis=1)
            current = frame.loc[target_indices, columns].to_numpy(float)
            complete_current = np.isfinite(current).all(axis=1)
            reasons[target_indices[~complete_current]] = "missing_current_features"
            audit: dict[str, Any] = {
                "stage": stage, "family": family, "alpha": alpha, "zone": zone,
                "refit_day": refit_day.date().isoformat(),
                "max_permitted_label_day": last_label_day.date().isoformat(),
                "window_first_day": first_label_day.date().isoformat(),
                "fit_first_day": train.delivery_day.min().date().isoformat() if len(train) else None,
                "fit_last_day": train.delivery_day.max().date().isoformat() if len(train) else None,
                "fit_rows": len(train), "complete_train_rows": int(complete_train.sum()),
                "fit_unique_days": int(train.delivery_day.nunique()),
                "prediction_rows": len(target_indices),
                "complete_prediction_rows": int(complete_current.sum()),
                "feature_columns": columns, "imputation_training_only": True,
                "normalization_training_only": True, "status": "fitted",
            }
            failure = None
            if complete_train.sum() < min_train_rows:
                failure = "insufficient_complete_training_rows"
            elif not np.isfinite(x).any(axis=0).all():
                failure = "missing_training_feature_history"
            elif family == "intrahour":
                shape_train = train.loc[complete_train, shapes].to_numpy(float)
                audit["shape_has_variation"] = bool(np.any(np.ptp(shape_train, axis=0) > 1e-12))
                if not audit["shape_has_variation"]:
                    failure = "constant_intrahour_shape"
            if failure:
                reasons[target_indices[complete_current]] = failure
                audit["status"] = failure
            else:
                # No observations outside train enter imputation, scaling or labels.
                imputer = SimpleImputer(strategy="median")
                xfit = imputer.fit_transform(x)
                scaler = StandardScaler()
                zfit = scaler.fit_transform(xfit)
                target = (train.training_actual - train.nyx_q50).to_numpy(float)
                model = Ridge(alpha=alpha, solver="svd").fit(zfit, target)
                audit.update({"imputation_values": imputer.statistics_.tolist(),
                              "input_mean": scaler.mean_.tolist(),
                              "input_scale": scaler.scale_.tolist(),
                              "target_mean": float(target.mean())})
                eligible = target_indices[complete_current]
                if len(eligible):
                    # Deliberately do not impute current forecasts. Missing means NYX.
                    correction = model.predict(scaler.transform(current[complete_current]))
                    proposed = prediction[eligible] + correction
                    finite = np.isfinite(proposed)
                    prediction[eligible[finite]] = proposed[finite]
                    reasons[eligible[~finite]] = "nonfinite_model_output"
            fit_audit.append(audit)
    return prediction[indices], reasons[indices]


def _group_masks(frame: pd.DataFrame, q95: dict[str, float]):
    yield "overall", "all", np.ones(len(frame), dtype=bool)
    for zone in ZONES:
        country = frame.zone.eq(zone).to_numpy()
        yield "country", zone, country
        for hour in range(24):
            yield "country_hour", f"{zone}:{hour:02}", country & frame.local_hour.eq(hour).to_numpy()
        yield "country_negative", zone, country & frame.actual.lt(0).to_numpy()
        yield "country_train_q95", zone, country & frame.actual.gt(q95[zone]).to_numpy()
    for hour in range(24):
        yield "hour", f"{hour:02}", frame.local_hour.eq(hour).to_numpy()
    yield "negative", "actual_lt_zero", frame.actual.lt(0).to_numpy()
    thresholds = frame.zone.map(q95)
    yield "train_q95", "actual_gt_train_q95", frame.actual.gt(thresholds).to_numpy()


def _paired(
    frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray,
    days: pd.DatetimeIndex, samples: np.ndarray | None,
) -> dict[str, Any]:
    actual = frame.actual.to_numpy(float)
    valid = np.isfinite(actual) & np.isfinite(candidate) & np.isfinite(baseline)
    indices = days.get_indexer(frame.delivery_day)
    ec, eb = candidate - actual, baseline - actual
    daily = np.zeros((len(days), 5))
    for day_index in range(len(days)):
        mask = valid & (indices == day_index)
        daily[day_index] = [np.abs(ec[mask]).sum(), np.abs(eb[mask]).sum(),
                            (ec[mask] ** 2).sum(), (eb[mask] ** 2).sum(), mask.sum()]
    totals = daily.sum(axis=0)
    if not totals[4]:
        return {"n": 0}
    cmae, bmae = totals[0] / totals[4], totals[1] / totals[4]
    crmse, brmse = np.sqrt(totals[2] / totals[4]), np.sqrt(totals[3] / totals[4])
    result = {
        "n": int(totals[4]), "days": len(days), "candidate_mae": float(cmae),
        "baseline_mae": float(bmae), "candidate_rmse": float(crmse),
        "baseline_rmse": float(brmse), "mae_delta": float(cmae - bmae),
        "rmse_delta": float(crmse - brmse),
        "mae_relative_improvement": float(1 - cmae / bmae) if bmae > 0 else None,
        "mae_delta_ci_low": None, "mae_delta_ci_high": None,
        "rmse_delta_ci_low": None, "rmse_delta_ci_high": None,
        "bootstrap_block_days": 7, "bootstrap_repetitions": 0 if samples is None else len(samples),
    }
    if samples is not None:
        bootstrap = daily[samples].sum(axis=1)
        mae = (bootstrap[:, 0] - bootstrap[:, 1]) / bootstrap[:, 4]
        rmse = np.sqrt(bootstrap[:, 2] / bootstrap[:, 4]) - np.sqrt(bootstrap[:, 3] / bootstrap[:, 4])
        for metric, values in (("mae", mae), ("rmse", rmse)):
            result[f"{metric}_delta_ci_low"] = float(np.quantile(values, .025))
            result[f"{metric}_delta_ci_high"] = float(np.quantile(values, .975))
    return result


def _critical_regimes(
    test: pd.DataFrame, predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    q95: dict[str, float],
) -> list[dict[str, Any]]:
    """Prespecified observed regimes are diagnostics, not forecast-time selectors."""
    rows = []
    actual = test.actual.to_numpy(float)
    candidate = predictions["intrahour"][0]
    for zone in ZONES:
        country = test.zone.eq(zone).to_numpy()
        for regime in ("negative", "above_train_q95"):
            threshold = 0.0 if regime == "negative" else q95[zone]
            event = (actual < 0) if regime == "negative" else (actual > threshold)
            for baseline in ("nyx", "hourly_control"):
                reference = predictions[baseline][0]
                mask = country & event & np.isfinite(actual) & np.isfinite(candidate) & np.isfinite(reference)
                hours = int(mask.sum())
                distinct_days = int(test.loc[mask, "delivery_day"].nunique())
                sufficient = np.isfinite(threshold) and hours >= 30 and distinct_days >= 5
                row = {
                    "stage": "test", "family": "intrahour", "baseline": baseline,
                    "zone": zone, "regime": regime, "threshold": threshold,
                    "n": hours, "distinct_days": distinct_days,
                    "minimum_hours": 30, "minimum_distinct_days": 5,
                    "evidence": "sufficient" if sufficient else "insufficient",
                    "candidate_mae": None, "baseline_mae": None,
                    "mae_relative_degradation": None, "within_5pct": None,
                }
                if sufficient:
                    cmae = float(np.abs(candidate[mask] - actual[mask]).mean())
                    bmae = float(np.abs(reference[mask] - actual[mask]).mean())
                    row.update(candidate_mae=cmae, baseline_mae=bmae,
                               mae_relative_degradation=(cmae / bmae - 1) if bmae > 0 else None,
                               within_5pct=cmae <= bmae * 1.05)
                rows.append(row)
    return rows


def evaluate_variant(
    panel: pd.DataFrame, *, initial_train_days: int = 120, validation_days: int = 60,
    test_days: int = 60, window_days: int = 180, refit_every_days: int = 7,
    min_train_rows: int = 720, bootstrap_repetitions: int = 1000, seed: int = 20260916,
) -> dict[str, Any]:
    """Compare NYX, an hourly ridge control and a ridge with intrahour shape.

    The first consecutive civil days define the fixed split; missing days are never
    compressed away. A complete four-zone physical-hour target grid is required.
    Validation uses frozen training_actual. A one-day embargo before test makes
    the last selection label available at D-2. Reported losses use actual. Training
    can assimilate prior test labels at the fixed cutoff, never retune the recipe.
    """
    for name, value in locals().copy().items():
        if name.endswith("_days") or name in {"min_train_rows", "bootstrap_repetitions"}:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        return _unavailable("empty_panel", available_days=0)
    frame = _prepare(panel)
    day_count = initial_train_days + validation_days + 1 + test_days
    days = pd.date_range(frame.delivery_day.min(), periods=day_count, freq="D")
    available_days = pd.DatetimeIndex(frame.delivery_day.unique()).sort_values()
    if not days.isin(available_days).all():
        return _unavailable("missing_delivery_days", available_days=len(available_days),
                            required_days=day_count,
                            missing_days=[d.date().isoformat() for d in days.difference(available_days)])
    frame = frame.loc[frame.delivery_day.le(days[-1])].reset_index(drop=True)
    # Build the expected UTC grid from local midnights, preserving 23/25-hour days.
    utc_hours = pd.date_range(days[0].tz_localize("Europe/Paris"),
                              (days[-1] + pd.Timedelta(days=1)).tz_localize("Europe/Paris"),
                              freq="h", inclusive="left").tz_convert("UTC")
    expected = pd.MultiIndex.from_product([utc_hours, ZONES], names=["timestamp_utc", "zone"])
    observed = pd.MultiIndex.from_frame(frame[["timestamp_utc", "zone"]])
    if len(expected.difference(observed)) or len(observed.difference(expected)):
        return _unavailable("incomplete_target_grid", expected_rows=len(expected),
                            observed_rows=len(observed), missing_rows=len(expected.difference(observed)))
    validation_start = days[initial_train_days]
    validation_end = days[initial_train_days + validation_days - 1]
    test_start, test_end = days[initial_train_days + validation_days + 1], days[-1]
    validation = frame.delivery_day.between(validation_start, validation_end)
    testing = frame.delivery_day.between(test_start, test_end)
    if not np.isfinite(frame.loc[validation | testing, ["actual", "nyx_q50"]]).all().all():
        return _unavailable("missing_evaluation_targets_or_nyx")
    if not np.isfinite(frame.loc[validation, "training_actual"]).all():
        return _unavailable("missing_validation_frozen_labels")
    means = sorted(c for c in frame if c.startswith("feature_intrahour_") and c.endswith("__mean_gw"))
    shapes = sorted(c for c in frame if c.startswith("feature_intrahour_") and c.endswith(SHAPE_SUFFIXES))
    if not means or not shapes or not np.isfinite(frame[shapes].to_numpy(float)).any():
        return _unavailable("missing_features", intrahour_mean_columns=means, intrahour_shape_columns=shapes)
    if any(c.rsplit("__", 1)[0] + "__mean_gw" not in means for c in shapes):
        return _unavailable("missing_intrahour_mean_control")
    hourly = sorted(c for c in frame if c.startswith("feature_hourly_"))
    features = {"hourly_control": ["nyx_q50", *CALENDAR, *hourly, *means]}
    features["intrahour"] = [*features["hourly_control"], *shapes]
    audit: list[dict[str, Any]] = []
    val = frame.loc[validation].copy()
    test = frame.loc[testing].copy()
    all_validation: dict[tuple[str, float | None], tuple[np.ndarray, np.ndarray]] = {
        ("nyx", None): (val.nyx_q50.to_numpy(float), np.full(len(val), "", dtype=object))}
    candidates = [{"family": "nyx", "alpha": None,
                   **_scores(val.training_actual.to_numpy(float), val.nyx_q50.to_numpy(float))}]
    with threadpool_limits(limits=1):
        for family, columns in features.items():
            for alpha in ALPHAS:
                prediction = _predict(frame, "validation", validation_start, validation_end,
                                      family, alpha, columns, shapes, window_days=window_days,
                                      refit_every_days=refit_every_days, min_train_rows=min_train_rows,
                                      fit_audit=audit)
                all_validation[(family, alpha)] = prediction
                candidates.append({"family": family, "alpha": alpha,
                                   **_scores(val.training_actual.to_numpy(float), prediction[0])})
        selected = {family: min((c for c in candidates if c["family"] == family), key=lambda c: c["mae"])
                    for family in ("nyx", "hourly_control", "intrahour")}
        primary = min(selected.values(), key=lambda c: c["mae"])["family"]
        selection = {
            "status": "locked", "label": "training_actual", "objective": "validation pooled MAE",
            "initial_train_days": initial_train_days,
            "selection_label_embargo_days": 1,
            "validation_start": validation_start.date().isoformat(),
            "validation_end": validation_end.date().isoformat(),
            "test_start": test_start.date().isoformat(), "test_end": test_end.date().isoformat(),
            "selected_per_family": selected, "primary_family": primary,
            "validation_candidates": candidates, "feature_columns": features,
            "test_used_for_selection": False,
        }
        # This immutable identity is established before test predictions or scores.
        selection["sha256"] = hashlib.sha256(json.dumps(selection, sort_keys=True, allow_nan=False).encode()).hexdigest()
        outputs: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            "validation": {family: all_validation[(family, row["alpha"])] for family, row in selected.items()},
            "test": {"nyx": (test.nyx_q50.to_numpy(float), np.full(len(test), "", dtype=object))},
        }
        for family, columns in features.items():
            outputs["test"][family] = _predict(
                frame, "test", test_start, test_end, family, selected[family]["alpha"], columns, shapes,
                window_days=window_days, refit_every_days=refit_every_days,
                min_train_rows=min_train_rows, fit_audit=audit)
    train_for_thresholds = frame.loc[frame.delivery_day.le(validation_start - pd.Timedelta(days=2))]
    q95 = {zone: float(train_for_thresholds.loc[train_for_thresholds.zone.eq(zone), "training_actual"].quantile(.95))
           for zone in ZONES}
    prediction_tables, metric_rows = [], []
    for stage, subset in (("validation", val), ("test", test)):
        for family, (pred, reasons) in outputs[stage].items():
            table = subset[["timestamp_utc", "delivery_day", "zone", "actual", "training_actual", "nyx_q50"]].copy()
            table["stage"], table["family"] = stage, family
            table["alpha"] = pd.Series(selected[family]["alpha"], index=table.index, dtype=float)
            table["prediction"], table["fallback_reason"] = pred, reasons
            table["fallback"] = reasons != ""
            prediction_tables.append(table)
            for group, value, mask in _group_masks(subset, q95):
                metric_rows.append({"stage": stage, "family": family, "group": group, "value": value,
                                    "label": "actual", **_scores(subset.actual.to_numpy(float)[mask], pred[mask])})
    rng = np.random.default_rng(seed)
    samples = None
    if test_days >= 7:
        starts = rng.integers(0, test_days - 7 + 1, size=(bootstrap_repetitions, int(np.ceil(test_days / 7))))
        samples = (starts[:, :, None] + np.arange(7)[None, None, :]).reshape(bootstrap_repetitions, -1)[:, :test_days]
    paired_rows = []
    for family, baseline in (("hourly_control", "nyx"), ("intrahour", "nyx"), ("intrahour", "hourly_control")):
        for zone in ("all", *ZONES):
            mask = np.ones(len(test), dtype=bool) if zone == "all" else test.zone.eq(zone).to_numpy()
            paired_rows.append({"stage": "test", "family": family, "baseline": baseline, "zone": zone,
                                **_paired(test.loc[mask], outputs["test"][family][0][mask],
                                          outputs["test"][baseline][0][mask], days[-test_days:], samples)})
    contrast_checks = {}
    for baseline in ("nyx", "hourly_control"):
        rows = [r for r in paired_rows if r["family"] == "intrahour" and r["baseline"] == baseline]
        pooled = next(r for r in rows if r["zone"] == "all")
        contrast_checks[baseline] = {
            "mae_improvement_at_least_2pct": pooled["mae_relative_improvement"] is not None and pooled["mae_relative_improvement"] >= .02,
            "mae_ci_upper_below_zero": pooled["mae_delta_ci_high"] is not None and pooled["mae_delta_ci_high"] < 0,
            "rmse_within_1pct": pooled["candidate_rmse"] <= pooled["baseline_rmse"] * 1.01,
            "every_country_mae_within_5pct": all(r["candidate_mae"] <= r["baseline_mae"] * 1.05 for r in rows if r["zone"] != "all"),
        }
    regime_rows = _critical_regimes(test, outputs["test"], q95)
    regime_evidence = any(row["evidence"] == "sufficient" for row in regime_rows)
    insufficient_regimes = [{key: row[key] for key in ("zone", "regime", "n", "distinct_days")}
                            for row in regime_rows if row["baseline"] == "nyx" and row["evidence"] == "insufficient"]
    degraded_regimes = [row for row in regime_rows if row["evidence"] == "sufficient" and not row["within_5pct"]]
    chosen_audit = [a for a in audit if a["family"] == "intrahour" and a["alpha"] == selected["intrahour"]["alpha"]]
    fallback_rate = float(np.mean(outputs["test"]["intrahour"][1] != ""))
    data_checks = {
        "test_fallback_at_most_5pct": fallback_rate <= .05,
        "complete_training_rows_sufficient": all(a["complete_train_rows"] >= min_train_rows for a in chosen_audit),
        "shape_varies_in_training": all(a.get("shape_has_variation", False) for a in chosen_audit),
        "seven_day_bootstrap_available": samples is not None,
    }
    reasons = [name for name, passed in data_checks.items() if not passed]
    if primary != "intrahour":
        reasons.append("intrahour_not_selected_on_validation")
    for baseline, checks in contrast_checks.items():
        reasons.extend(f"vs_{baseline}:{name}" for name, passed in checks.items() if not passed)
    if not regime_evidence:
        reasons.append("insufficient_critical_regime_evidence")
    reasons.extend(f"vs_{row['baseline']}:{row['zone']}:{row['regime']}:mae_degradation_over_5pct"
                   for row in degraded_regimes)
    enough_data = all(data_checks.values())
    decision_status = ("insufficient_data" if not enough_data else
                       "insufficient_regime_evidence" if not regime_evidence else
                       "not_encouraging" if reasons else "encouraging")
    decision = {"status": decision_status,
                "encouraging": not reasons, "promotion_allowed": False, "reasons": reasons,
                "comparisons": contrast_checks, "data_checks": data_checks,
                "critical_regimes": {
                    "minimum_hours": 30, "minimum_distinct_days": 5,
                    "maximum_mae_degradation": .05,
                    "evidence_available": regime_evidence,
                    "coverage_complete": not insufficient_regimes,
                    "insufficient_regimes": insufficient_regimes,
                    "degraded_regimes": degraded_regimes,
                    "all_sufficient_regimes_within_5pct": not degraded_regimes,
                    "coverage_note": ("Some critical regimes lack evidence; no conclusion for those regimes."
                                      if insufficient_regimes else "All prespecified regimes meet the sample minimum."),
                },
                "test_fallback_rate": fallback_rate,
                "interpretation": "Retrospective point correction of NYX P50; no production promotion or calibrated quantiles."}
    return {"status": "complete" if enough_data else "insufficient_data",
            "reason": None if enough_data else "insufficient_usable_intrahour_features",
            "predictions": pd.concat(prediction_tables, ignore_index=True),
            "metrics": pd.DataFrame(metric_rows), "paired_deltas": pd.DataFrame(paired_rows),
            "regime_checks": pd.DataFrame(regime_rows),
            "selection": selection, "fit_audit": pd.DataFrame(audit), "decision": decision,
            "protocol": {"timezone": "Europe/Paris", "label_delay_days": 2,
                         "selection_label_embargo_days": 1,
                         "window_days": window_days, "refit_every_days": refit_every_days,
                         "bootstrap_block_days": 7, "bootstrap_seed": seed,
                         "q95_training_thresholds": q95, "quantiles_produced": False,
                         "vintages_verified_here": False}}
