"""Read-only evaluation of sealed point forecasts; never imports a model runner.

The historical test has already been inspected in earlier research. Every result
here is exploratory. Averaging four point predictions produces an hourly point
prediction, not a calibrated hourly quantile or a predictive distribution.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from io import BytesIO
import hashlib
from pathlib import Path
import re
import stat
from typing import Any

import numpy as np
import pandas as pd

ZONES = ("BE", "DE", "FR", "NL")
TIMEZONE = "Europe/Paris"
DEFAULT_START_DAY = "2026-06-18"
DEFAULT_END_DAY = "2026-09-15"


class EvaluationContractError(ValueError):
    """Forecasts violate the declared grid, population or sealed-file contract."""


def _day(value: str) -> date:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise EvaluationContractError("Dates must be YYYY-MM-DD civil dates.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise EvaluationContractError("Invalid civil date.") from exc


def _grid(start_day: str, end_day: str, frequency: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    first, last = _day(start_day), _day(end_day)
    if first > last:
        raise EvaluationContractError("start_day must not be after end_day.")
    start = pd.Timestamp(first).tz_localize(TIMEZONE)
    end = pd.Timestamp(last + timedelta(days=1)).tz_localize(TIMEZONE)
    timestamps = pd.date_range(start, end, freq=frequency, inclusive="left").tz_convert("UTC")
    days = pd.date_range(first, last, freq="D")
    return timestamps, days


def read_sealed_predictions(
    path: str | Path, expected_sha256: str, *, max_bytes: int = 64 * 1024 * 1024,
) -> pd.DataFrame:
    """Read exactly the checked CSV/parquet bytes; never refresh or rewrite them.

    The digest should come from the run's previously sealed manifest. This proves
    byte identity, not truth of source vintages or independence of the experiment.
    """
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None:
        raise EvaluationContractError("A complete SHA256 from the sealed manifest is required.")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise EvaluationContractError("max_bytes must be a positive integer.")
    path = Path(path).absolute()
    if path.suffix.lower() not in {".parquet", ".csv"}:
        raise EvaluationContractError("Only sealed parquet or CSV predictions are supported.")
    for component in (path, *path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise EvaluationContractError("Sealed inputs must not use symlinks, junctions or reparse points.")
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise EvaluationContractError("Sealed prediction file exceeds the size limit or is not a regular file.")
    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise EvaluationContractError("Sealed prediction file exceeds the size limit.")
    if hashlib.sha256(payload).hexdigest() != expected_sha256.lower():
        raise EvaluationContractError("Sealed prediction SHA256 mismatch; refusing changed input.")
    stream = BytesIO(payload)
    return pd.read_parquet(stream) if path.suffix.lower() == ".parquet" else pd.read_csv(stream)


def _canonical(
    frame: pd.DataFrame, *, columns: list[str], timestamps: pd.DatetimeIndex,
    frequency: str, label: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise EvaluationContractError(f"{label}: a DataFrame is required.")
    if frame.columns.has_duplicates:
        raise EvaluationContractError(f"{label}: duplicate column names.")
    missing = {"timestamp_utc", "zone", *columns} - set(frame.columns)
    if missing:
        raise EvaluationContractError(f"{label}: missing columns {sorted(missing)}.")
    result = frame[["timestamp_utc", "zone", *columns]].copy(deep=True)
    try:
        index = pd.DatetimeIndex(result.timestamp_utc)
    except (TypeError, ValueError) as exc:
        raise EvaluationContractError(f"{label}: valid aware timestamps are required.") from exc
    if index.tz is None or index.hasnans:
        raise EvaluationContractError(f"{label}: timezone-aware nonmissing timestamps are required.")
    index = index.tz_convert("UTC")
    if not index.equals(index.floor(frequency)):
        raise EvaluationContractError(f"{label}: timestamps are off the physical {frequency} grid.")
    result["timestamp_utc"] = index
    if result.zone.isna().any() or not set(result.zone).issubset(ZONES):
        raise EvaluationContractError(f"{label}: only BE, DE, FR and NL targets are allowed.")
    if result.duplicated(["timestamp_utc", "zone"]).any():
        raise EvaluationContractError(f"{label}: duplicate zone/timestamp targets.")
    for column in columns:
        try:
            values = pd.to_numeric(result[column], errors="raise")
            if np.iscomplexobj(values):
                raise ValueError("complex values")
            result[column] = values.astype(float)
        except (TypeError, ValueError) as exc:
            raise EvaluationContractError(f"{label}: {column} must be numeric.") from exc
        if not np.isfinite(result[column].to_numpy()).all():
            raise EvaluationContractError(f"{label}: {column} contains missing or nonfinite values; no replacement is allowed.")
    expected = pd.MultiIndex.from_product([timestamps, ZONES], names=["timestamp_utc", "zone"])
    actual = pd.MultiIndex.from_frame(result[["timestamp_utc", "zone"]])
    absent, extra = expected.difference(actual), actual.difference(expected)
    if len(absent) or len(extra):
        raise EvaluationContractError(
            f"{label}: incomplete or different target grid ({len(absent)} missing, {len(extra)} outside); "
            "all four countries must cover exactly the declared test window."
        )
    return result.set_index(["timestamp_utc", "zone"]).reindex(expected).reset_index()


def aggregate_quarterhour_predictions(
    frame: pd.DataFrame, *, start_day: str, end_day: str,
) -> pd.DataFrame:
    """Average four finite quarter points per physical hour, with no interpolation.

    Input must cover exactly the requested civil days in all four zones (92/96/100
    quarters per zone/day). Extra dates are rejected as well as gaps/duplicates.
    Other columns, including quantiles, are not aggregated or returned.
    """
    quarters, _ = _grid(start_day, end_day, "15min")
    points = _canonical(frame, columns=["prediction"], timestamps=quarters,
                        frequency="15min", label="quarter-hour predictions")
    points["timestamp_utc"] = points.timestamp_utc.dt.floor("h")
    grouped = points.groupby(["timestamp_utc", "zone"], sort=True).prediction
    if not grouped.size().eq(4).all():
        raise EvaluationContractError("Every physical hour requires exactly four finite predictions.")
    hourly = grouped.mean().rename("prediction").reset_index()
    hours, _ = _grid(start_day, end_day, "h")
    hourly = _canonical(hourly, columns=["prediction"], timestamps=hours,
                        frequency="h", label="aggregated hourly predictions")
    counts = pd.Series(1, index=quarters).groupby(quarters.tz_convert(TIMEZONE).date).sum()
    hourly.attrs["aggregation"] = "arithmetic_mean_of_four_finite_point_predictions"
    hourly.attrs["quarter_counts_per_zone_by_day"] = {str(day): int(n) for day, n in counts.items()}
    hourly.attrs["calibrated_quantiles_produced"] = False
    return hourly


def _score(actual: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if not len(actual):
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    error = prediction - actual
    return {"n": len(actual), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean())}


def _groups(frame: pd.DataFrame, q95: dict[str, float]):
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
    yield "train_q95", "actual_gt_train_q95", frame.actual.gt(frame.zone.map(q95)).to_numpy()


def _paired(
    frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray,
    days: pd.DatetimeIndex, samples: np.ndarray | None,
) -> dict[str, Any]:
    ec, eb = candidate - frame.actual.to_numpy(), baseline - frame.actual.to_numpy()
    sums = np.zeros((len(days), 5))
    day_indices = days.get_indexer(frame.delivery_day)
    for i in range(len(days)):
        mask = day_indices == i
        sums[i] = [np.abs(ec[mask]).sum(), np.abs(eb[mask]).sum(),
                   (ec[mask] ** 2).sum(), (eb[mask] ** 2).sum(), mask.sum()]
    total = sums.sum(axis=0)
    cmae, bmae = total[0] / total[4], total[1] / total[4]
    crmse, brmse = np.sqrt(total[2] / total[4]), np.sqrt(total[3] / total[4])
    row = {"n": int(total[4]), "days": len(days),
           "candidate_mae": float(cmae), "baseline_mae": float(bmae),
           "candidate_rmse": float(crmse), "baseline_rmse": float(brmse),
           "mae_delta": float(cmae - bmae), "rmse_delta": float(crmse - brmse),
           "bias_delta": float(ec.mean() - eb.mean()),
           "mae_relative_improvement": float(1 - cmae / bmae) if bmae > 0 else None,
           "mae_delta_ci_low": None, "mae_delta_ci_high": None,
           "rmse_delta_ci_low": None, "rmse_delta_ci_high": None,
           "bootstrap_block_days": 7, "bootstrap_repetitions": 0 if samples is None else len(samples)}
    if samples is not None:
        draws = sums[samples].sum(axis=1)
        mae = (draws[:, 0] - draws[:, 1]) / draws[:, 4]
        rmse = np.sqrt(draws[:, 2] / draws[:, 4]) - np.sqrt(draws[:, 3] / draws[:, 4])
        for name, values in (("mae", mae), ("rmse", rmse)):
            row[f"{name}_delta_ci_low"] = float(np.quantile(values, .025))
            row[f"{name}_delta_ci_high"] = float(np.quantile(values, .975))
    return row


def evaluate_predictions(
    baseline_hourly: pd.DataFrame, predictions: Mapping[str, pd.DataFrame], *,
    q95_thresholds: Mapping[str, float], candidate_family: str = "native_quarterhour",
    matched_control_family: str = "hourly_control", start_day: str = DEFAULT_START_DAY,
    end_day: str = DEFAULT_END_DAY, bootstrap_repetitions: int = 1000,
    seed: int = 20260916, matched_control_verified: bool = False,
) -> dict[str, Any]:
    """Evaluate predeclared hourly recipes without selecting or fitting any model.

    ``predictions`` holds hourly point frames, potentially returned by aggregation.
    The optional baseline column ``hourly_control`` supplies the named matched
    control; alternatively provide its frame in ``predictions``. The caller must
    attest identical architecture/postprocessing using matched_control_verified.
    All grids must exactly match the declared window; no intersection/drop/fill.
    q95_thresholds must have been fixed from past training data by the caller.
    """
    if not isinstance(predictions, Mapping):
        raise EvaluationContractError("predictions must map family names to hourly frames.")
    if not isinstance(q95_thresholds, Mapping) or set(q95_thresholds) != set(ZONES):
        raise EvaluationContractError("Training q95 thresholds are required for exactly BE, DE, FR and NL.")
    try:
        if any(isinstance(v, bool) for v in q95_thresholds.values()):
            raise ValueError("boolean threshold")
        q95 = {zone: float(q95_thresholds[zone]) for zone in ZONES}
    except (TypeError, ValueError) as exc:
        raise EvaluationContractError("Training q95 thresholds must be finite numbers.") from exc
    if not np.isfinite(list(q95.values())).all():
        raise EvaluationContractError("Training q95 thresholds must be finite numbers.")
    if isinstance(bootstrap_repetitions, bool) or not isinstance(bootstrap_repetitions, (int, np.integer)) or bootstrap_repetitions < 2:
        raise EvaluationContractError("bootstrap_repetitions must be an integer of at least two.")
    if not isinstance(matched_control_verified, bool):
        raise EvaluationContractError("matched_control_verified must be an explicit boolean.")
    if candidate_family in {"nyx", matched_control_family} or matched_control_family == "nyx":
        raise EvaluationContractError("Candidate, NYX and matched control must be distinct families.")
    if candidate_family not in predictions:
        raise EvaluationContractError("The explicitly declared candidate family is missing.")
    if "nyx" in predictions:
        raise EvaluationContractError("The immutable NYX reference cannot be replaced by a prediction family.")
    if any(not isinstance(name, str) or not name.strip() for name in predictions):
        raise EvaluationContractError("Prediction family names must be nonempty strings.")
    hours, days = _grid(start_day, end_day, "h")
    base_columns = ["actual", "nyx_q50"]
    embedded_control = "hourly_control" in baseline_hourly.columns
    if embedded_control:
        if matched_control_family in predictions:
            raise EvaluationContractError("Matched control was supplied twice; choose one unambiguous source.")
        base_columns.append("hourly_control")
    base = _canonical(baseline_hourly, columns=base_columns, timestamps=hours, frequency="h", label="hourly baseline")
    local = base.timestamp_utc.dt.tz_convert(TIMEZONE)
    base["delivery_day"] = local.dt.tz_localize(None).dt.normalize()
    base["local_hour"] = local.dt.hour
    forecasts = {"nyx": base.nyx_q50.to_numpy(float)}
    if embedded_control:
        forecasts[matched_control_family] = base.hourly_control.to_numpy(float)
    for name, frame in predictions.items():
        validated = _canonical(frame, columns=["prediction"], timestamps=hours, frequency="h", label=f"family {name}")
        forecasts[name] = validated.prediction.to_numpy(float)
    metric_rows, long_frames = [], []
    actual = base.actual.to_numpy(float)
    masks = list(_groups(base, q95))
    for family, values in forecasts.items():
        table = base[["timestamp_utc", "delivery_day", "zone", "actual", "nyx_q50"]].copy()
        table["family"], table["stage"], table["prediction"] = family, "test", values
        long_frames.append(table)
        for group, value, mask in masks:
            metric_rows.append({"stage": "test", "family": family, "group": group,
                                "value": value, **_score(actual[mask], values[mask])})
    samples = None
    if len(days) >= 7:
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, len(days) - 7 + 1, size=(bootstrap_repetitions, int(np.ceil(len(days) / 7))))
        samples = (starts[:, :, None] + np.arange(7)[None, None, :]).reshape(bootstrap_repetitions, -1)[:, :len(days)]
    contrasts = [(family, "nyx") for family in forecasts if family != "nyx"]
    if matched_control_family in forecasts:
        contrasts += [(family, matched_control_family) for family in forecasts if family not in {"nyx", matched_control_family}]
    paired_rows = []
    for family, reference in contrasts:
        for zone in ("all", *ZONES):
            mask = np.ones(len(base), dtype=bool) if zone == "all" else base.zone.eq(zone).to_numpy()
            paired_rows.append({"stage": "test", "family": family, "baseline": reference, "zone": zone,
                                **_paired(base.loc[mask], forecasts[family][mask], forecasts[reference][mask], days, samples)})
    checks, regime_rows, reasons = {}, [], []
    for reference in ("nyx", matched_control_family):
        if reference not in forecasts:
            reasons.append("missing_matched_control")
            checks[reference] = {"available": False}
            continue
        rows = [r for r in paired_rows if r["family"] == candidate_family and r["baseline"] == reference]
        pooled = next(r for r in rows if r["zone"] == "all")
        checks[reference] = {
            "available": True,
            "mae_improvement_at_least_2pct": pooled["mae_relative_improvement"] is not None and pooled["mae_relative_improvement"] >= .02,
            "mae_ci_upper_below_zero": pooled["mae_delta_ci_high"] is not None and pooled["mae_delta_ci_high"] < 0,
            "rmse_within_1pct": pooled["candidate_rmse"] <= pooled["baseline_rmse"] * 1.01,
            "every_country_mae_within_5pct": all(r["candidate_mae"] <= r["baseline_mae"] * 1.05 for r in rows if r["zone"] != "all"),
        }
        reasons.extend(f"vs_{reference}:{key}" for key, passed in checks[reference].items() if not passed)
        for zone in ZONES:
            country = base.zone.eq(zone).to_numpy()
            for regime in ("negative", "above_train_q95"):
                threshold = 0.0 if regime == "negative" else q95[zone]
                mask = country & ((actual < 0) if regime == "negative" else (actual > threshold))
                n, n_days = int(mask.sum()), int(base.loc[mask, "delivery_day"].nunique())
                sufficient = n >= 30 and n_days >= 5
                row = {"stage": "test", "family": candidate_family, "baseline": reference, "zone": zone,
                       "regime": regime, "threshold": threshold, "n": n, "distinct_days": n_days,
                       "evidence": "sufficient" if sufficient else "insufficient",
                       "candidate_mae": None, "baseline_mae": None,
                       "mae_relative_degradation": None, "within_5pct": None}
                if sufficient:
                    cm = float(np.abs(forecasts[candidate_family][mask] - actual[mask]).mean())
                    bm = float(np.abs(forecasts[reference][mask] - actual[mask]).mean())
                    row.update(candidate_mae=cm, baseline_mae=bm,
                               mae_relative_degradation=(cm / bm - 1) if bm > 0 else None,
                               within_5pct=cm <= bm * 1.05)
                    if not row["within_5pct"]:
                        reasons.append(f"vs_{reference}:{zone}:{regime}:mae_degradation_over_5pct")
                regime_rows.append(row)
    control_exists = matched_control_family in forecasts
    if control_exists and not matched_control_verified:
        reasons.append("matched_architecture_and_postprocessing_not_verified")
    if samples is None:
        reasons.append("insufficient_days_for_seven_day_bootstrap")
    regime_evidence = any(r["evidence"] == "sufficient" for r in regime_rows)
    if not regime_evidence:
        reasons.append("insufficient_critical_regime_evidence")
    insufficient_regimes = [{k: r[k] for k in ("zone", "regime", "n", "distinct_days")}
                            for r in regime_rows if r["baseline"] == "nyx" and r["evidence"] == "insufficient"]
    decision = {
        "status": "encouraging_exploratory" if not reasons else "not_encouraging",
        "encouraging": not reasons, "promotion_allowed": False,
        "candidate_family": candidate_family, "matched_control_family": matched_control_family,
        "matched_control_verified": matched_control_verified and control_exists,
        "comparisons": checks, "reasons": reasons,
        "critical_regimes": {"minimum_hours": 30, "minimum_distinct_days": 5,
                             "maximum_mae_degradation": .05, "evidence_available": regime_evidence,
                             "coverage_complete": not insufficient_regimes,
                             "insufficient_regimes": insufficient_regimes},
        "interpretation": "Exploratory historical point-forecast comparison; no calibrated quantile claim or production promotion.",
    }
    return {
        "status": "complete" if control_exists and samples is not None else "insufficient_data",
        "reason": "missing_matched_control" if not control_exists else ("insufficient_bootstrap_days" if samples is None else None),
        "predictions": pd.concat(long_frames, ignore_index=True), "metrics": pd.DataFrame(metric_rows),
        "paired_deltas": pd.DataFrame(paired_rows), "regime_checks": pd.DataFrame(regime_rows),
        "decision": decision,
        "protocol": {"start_day": start_day, "end_day": end_day, "days": len(days),
                     "zones": list(ZONES), "paired_hours_per_zone": len(hours),
                     "paired_points_per_family": len(base), "timezone": TIMEZONE,
                     "bootstrap_block_days": 7, "bootstrap_repetitions": bootstrap_repetitions,
                     "bootstrap_seed": seed, "exploratory_history_already_reviewed": True,
                     "automatic_model_selection": False, "calibrated_quantiles_produced": False,
                     "q95_thresholds_supplied_from_training": q95,
                     "q95_thresholds_computed_from_test": False,
                     "point_aggregation_contract": "mean of four finite points; no distribution or quantile aggregation",
                     "native_resolution_and_vintages_verified_here": False},
    }
