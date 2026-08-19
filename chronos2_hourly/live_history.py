"""Leakage-safe Statistics history for operational day-ahead forecasts.

The annual backtest remains sealed.  This module writes a separate reporting
artifact containing its declared evaluation window plus complete, realized
days sourced from immutable issued-live or explicit PIT-replay runs.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.storm_dashboard import (
    STORM_DASHBOARD_ARTIFACT,
    STORM_DASHBOARD_COLUMN,
    STORM_LEGACY_STRICT_COLUMN,
    STORM_STRICT_08_COLUMN,
    normalize_native_dashboard_series,
    storm_dashboard_series,
)
from run_mkonline_blend_hourly import _load_storm_evaluation_only


TIMEZONE = "Europe/Paris"
FORECAST_NAME = "forecast_hourly_fr.csv"
BACKTEST_NAME = "backtest_hourly_oof.csv.gz"
METRICS_NAME = "metrics_hourly.json"
STATISTICS_HISTORY_NAME = "statistics_history_hourly.csv.gz"
STATISTICS_AUDIT_NAME = "statistics_history_audit.json"
QUANTILES = ("q10", "q50", "q90")
PROVENANCE_COLUMNS = (
    "statistics_scope",
    "statistics_run_type",
    "statistics_source_run",
    "statistics_source_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _cutoff_for_day(day: date, *, timezone: str) -> pd.Timestamp:
    # Civil arithmetic must precede localization across DST transitions.
    cutoff_naive = (
        pd.Timestamp(day) - pd.DateOffset(days=1) + pd.Timedelta(hours=8)
    )
    return cutoff_naive.tz_localize(
        timezone, ambiguous="raise", nonexistent="raise"
    )


def _forecast_filename(value: str) -> str:
    name = str(value).strip()
    if not name or Path(name).name != name:
        raise ValueError("forecast_name must be a plain filename")
    return name


def _verify_forecast_checksum(
    run_dir: Path,
    *,
    forecast_name: str = FORECAST_NAME,
) -> str:
    forecast_name = _forecast_filename(forecast_name)
    forecast_path = run_dir / forecast_name
    manifest_path = run_dir / "artifact_checksums.json"
    if not forecast_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Archive incomplet: {forecast_path} / {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError(f"{manifest_path}: artifacts must be a list")
    declarations = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).name == forecast_name
        and str(item.get("role", "")) == "run_artifact"
    ]
    if len(declarations) != 1:
        raise ValueError(
            f"{manifest_path}: expected one checksum for {forecast_name}"
        )
    expected = str(declarations[0].get("sha256", "")).lower()
    observed = _sha256(forecast_path)
    if expected != observed:
        raise ValueError(f"Forecast archive checksum mismatch: {forecast_path}")
    return observed


def _run_contract(
    run_dir: Path,
    *,
    expected_run_type: str,
    zone: str,
    timezone: str,
    target_series: str,
    candidate_model: str,
    prediction_mode: str,
    forecast_name: str,
) -> tuple[date, dict[str, Any]]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    run_type = str(manifest.get("run_type", ""))
    if run_type not in {"live_day_ahead", "pit_replay"}:
        raise ValueError(f"{manifest_path}: unsupported run_type={run_type!r}")
    if run_type != expected_run_type:
        raise ValueError(
            f"{manifest_path}: run_type={run_type!r} invalid under the "
            f"{expected_run_type!r} root"
        )
    expected_zone = str(zone).strip().upper()
    expected_identity = {
        "zone": expected_zone,
        "timezone": timezone,
    }
    for key, expected in expected_identity.items():
        observed = manifest.get(key)
        if observed != expected:
            raise ValueError(
                f"{manifest_path}: {key}={observed!r} differs from "
                f"{expected!r}"
            )
    manifest_forecast_name = manifest.get("forecast_path")
    if manifest_forecast_name not in (None, forecast_name):
        raise ValueError(
            f"{manifest_path}: forecast_path={manifest_forecast_name!r} "
            f"differs from {forecast_name!r}"
        )
    modern_identity = {
        "target_series": target_series,
        "candidate_model": candidate_model,
        "prediction_mode": prediction_mode,
    }
    missing_modern_identity = [
        key for key in modern_identity if manifest.get(key) in (None, "")
    ]
    legacy_fr = expected_zone == "FR" and len(missing_modern_identity) == len(
        modern_identity
    )
    if missing_modern_identity and not legacy_fr:
        raise ValueError(
            f"{manifest_path}: incomplete modern archive identity: "
            + ", ".join(missing_modern_identity)
        )
    if not legacy_fr:
        for key, expected in modern_identity.items():
            observed = manifest.get(key)
            if observed != expected:
                raise ValueError(
                    f"{manifest_path}: {key}={observed!r} differs from "
                    f"{expected!r}"
                )
    if manifest.get("storm_used_as_feature") is not False:
        raise ValueError(
            f"{manifest_path}: storm_used_as_feature must be explicitly false"
        )
    prediction_inputs = manifest.get("prediction_inputs")
    if not isinstance(prediction_inputs, list) or any(
        "storm" in str(value).casefold() for value in prediction_inputs
    ):
        raise ValueError(
            f"{manifest_path}: prediction_inputs must be an explicit Storm-free list"
        )
    if legacy_fr and prediction_inputs != [
        "autonomous_extended_residual",
        "41551_native",
    ]:
        raise ValueError(
            f"{manifest_path}: unsupported legacy FR prediction_inputs"
        )
    day_value = manifest.get("delivery_day_local")
    if not day_value:
        raise ValueError(f"{manifest_path}: delivery_day_local is missing")
    day = pd.Timestamp(day_value).date()
    cutoff_value = manifest.get("forecast_cutoff_local")
    if not cutoff_value:
        raise ValueError(f"{manifest_path}: forecast_cutoff_local is missing")
    cutoff = pd.Timestamp(cutoff_value)
    if cutoff.tzinfo is None:
        raise ValueError(f"{manifest_path}: forecast_cutoff_local must be zoned")
    expected_cutoff = _cutoff_for_day(day, timezone=timezone)
    if cutoff.tz_convert(timezone) != expected_cutoff:
        raise ValueError(
            f"{manifest_path}: cutoff={cutoff} differs from {expected_cutoff}"
        )
    return day, manifest


def _load_forecast_day(
    run_dir: Path,
    *,
    declared_day: date,
    timezone: str,
    forecast_name: str = FORECAST_NAME,
    candidate_model: str = "mkonline_blend",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    forecast_name = _forecast_filename(forecast_name)
    checksum = _verify_forecast_checksum(
        run_dir,
        forecast_name=forecast_name,
    )
    path = run_dir / forecast_name
    frame = pd.read_csv(path)
    models = tuple(dict.fromkeys(("residual_corrected", str(candidate_model))))
    required = {
        "delivery_start_utc",
        *(f"{model}__{q}" for model in models for q in QUANTILES),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing forecast columns {missing}")
    delivery = pd.DatetimeIndex(
        pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    expected = local_delivery_day_index(declared_day, timezone=timezone)
    if not delivery.equals(expected):
        raise ValueError(f"{path}: forecast does not cover exact declared day")
    qualified_origin = f"{candidate_model}_forecast_origin_utc"
    origin_column = (
        qualified_origin if qualified_origin in frame else "forecast_origin_utc"
    )
    if origin_column not in frame:
        raise ValueError(f"{path}: forecast origin is missing")
    origin = pd.DatetimeIndex(
        pd.to_datetime(frame[origin_column], utc=True, errors="raise")
    )
    cutoff = _cutoff_for_day(declared_day, timezone=timezone).tz_convert("UTC")
    if bool((origin > cutoff).any()) or bool((origin >= delivery).any()):
        raise ValueError(f"{path}: non-causal forecast origin")
    for model in models:
        values = frame[[f"{model}__{q}" for q in QUANTILES]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite {model} quantiles")
        if bool((values[:, 0] > values[:, 1]).any()) or bool(
            (values[:, 1] > values[:, 2]).any()
        ):
            raise ValueError(f"{path}: crossed {model} quantiles")
    frame = frame.copy()
    frame.index = delivery
    return frame, {
        "delivery_day_local": declared_day.isoformat(),
        "source_run": str(run_dir),
        "forecast_path": str(path),
        "forecast_sha256": checksum,
        "hours": int(len(frame)),
        "forecast_origin_min_utc": str(origin.min()),
        "forecast_origin_max_utc": str(origin.max()),
    }


def discover_archived_forecasts(
    *,
    live_output_root: str | Path,
    replay_output_root: str | Path,
    current_delivery_day: date,
    first_history_day: date,
    timezone: str = TIMEZONE,
    forecast_name: str = FORECAST_NAME,
    candidate_model: str = "mkonline_blend",
    zone: str = "FR",
    target_series: str = "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh",
    prediction_mode: str = "mkonline_blend",
) -> tuple[dict[date, pd.DataFrame], list[dict[str, Any]]]:
    """Read explicit issued-live and PIT-replay archives only.

    The sealed benchmark forecast is intentionally excluded because it is not
    evidence of an issued operational forecast.
    """

    roots = (
        (Path(replay_output_root).expanduser().resolve(), "pit_replay", "realized_pit_replay", 0),
        (Path(live_output_root).expanduser().resolve(), "live_day_ahead", "realized_live", 1),
    )
    candidates: dict[date, tuple[pd.DataFrame, dict[str, Any], int]] = {}
    for root, run_type, scope, precedence in roots:
        if not root.is_dir():
            continue
        for run_dir in sorted(root.iterdir()):
            if (
                not run_dir.is_dir()
                or run_dir.name.startswith(".")
                or not (run_dir / "run_manifest.json").is_file()
            ):
                continue
            day, _manifest = _run_contract(
                run_dir,
                expected_run_type=run_type,
                zone=zone,
                timezone=timezone,
                target_series=target_series,
                candidate_model=candidate_model,
                prediction_mode=prediction_mode,
                forecast_name=forecast_name,
            )
            if not (first_history_day <= day < current_delivery_day):
                continue
            frame, audit = _load_forecast_day(
                run_dir,
                declared_day=day,
                timezone=timezone,
                forecast_name=forecast_name,
                candidate_model=candidate_model,
            )
            audit.update(
                {
                    "archive_kind": run_type,
                    "statistics_scope": scope,
                    "run_type": run_type,
                    "archive_root": str(root),
                }
            )
            previous = candidates.get(day)
            if previous is not None:
                models = tuple(
                    dict.fromkeys(("residual_corrected", str(candidate_model)))
                )
                columns = [
                    f"{model}__{q}" for model in models for q in QUANTILES
                ]
                if not np.allclose(
                    previous[0][columns].to_numpy(dtype=float),
                    frame[columns].to_numpy(dtype=float),
                    rtol=0.0,
                    atol=1e-9,
                ):
                    raise ValueError(
                        f"Conflicting immutable forecasts for {day}: "
                        f"{previous[1]['source_run']} vs {run_dir}"
                    )
                if precedence <= previous[2]:
                    continue
                audit["equivalent_replay_source"] = previous[1]["source_run"]
            candidates[day] = (frame, audit, precedence)
    ordered = sorted(candidates)
    return (
        {day: candidates[day][0] for day in ordered},
        [candidates[day][1] for day in ordered],
    )


def missing_statistics_archive_days(
    *,
    sealed_benchmark_run: str | Path,
    live_output_root: str | Path,
    replay_output_root: str | Path,
    current_delivery_day: date,
    timezone: str = TIMEZONE,
    forecast_name: str = FORECAST_NAME,
    candidate_model: str = "mkonline_blend",
    zone: str = "FR",
    target_series: str = "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh",
    prediction_mode: str = "mkonline_blend",
) -> list[date]:
    """Return causal archive gaps required by the next Statistics build.

    This is a local, read-only planner.  It deliberately reuses the same
    sealed-benchmark and immutable-archive validation as
    :func:`update_live_statistics_history`, so an invalid or cross-wired
    archive fails before a live candidate (or any reporting-only comparator)
    is requested.
    """

    sealed = Path(sealed_benchmark_run).expanduser().resolve()
    _benchmark, _benchmark_start, benchmark_end, _benchmark_audit = (
        _sealed_benchmark_statistics(
            sealed,
            timezone=timezone,
            candidate_model=candidate_model,
        )
    )
    first_history_day = benchmark_end + pd.Timedelta(days=1)
    last_eligible_day = current_delivery_day - pd.Timedelta(days=1)
    if first_history_day > last_eligible_day:
        return []
    forecasts, _source_audit = discover_archived_forecasts(
        live_output_root=live_output_root,
        replay_output_root=replay_output_root,
        current_delivery_day=current_delivery_day,
        first_history_day=first_history_day,
        timezone=timezone,
        forecast_name=forecast_name,
        candidate_model=candidate_model,
        zone=zone,
        target_series=target_series,
        prediction_mode=prediction_mode,
    )
    expected_days = [
        value.date()
        for value in pd.date_range(
            first_history_day,
            last_eligible_day,
            freq="D",
        )
    ]
    return [day for day in expected_days if day not in forecasts]


def _canonical_target_utc(target: pd.Series) -> pd.Series:
    if not isinstance(target, pd.Series):
        raise TypeError("canonical_target must be a pandas Series")
    index = pd.DatetimeIndex(target.index)
    if index.tz is None:
        raise ValueError("canonical_target index must be timezone-aware")
    result = pd.Series(
        pd.to_numeric(target, errors="coerce").to_numpy(dtype=float),
        index=index.tz_convert("UTC"),
        name="actual",
    ).sort_index()
    if result.index.has_duplicates:
        raise ValueError("canonical_target contains duplicate timestamps")
    return result


def _point_metric_audit(
    actual: pd.Series,
    forecast: pd.Series,
) -> dict[str, float | int | None]:
    paired = pd.concat(
        [
            pd.to_numeric(actual, errors="coerce").rename("actual"),
            pd.to_numeric(forecast, errors="coerce").rename("forecast"),
        ],
        axis=1,
    ).dropna()
    if paired.empty:
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    error = (
        paired["forecast"].to_numpy(dtype=float)
        - paired["actual"].to_numpy(dtype=float)
    )
    return {
        "n": int(len(paired)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def _sealed_benchmark_statistics(
    sealed_run: Path,
    *,
    timezone: str,
    candidate_model: str = "mkonline_blend",
) -> tuple[pd.DataFrame, date, date, dict[str, Any]]:
    backtest_path = sealed_run / BACKTEST_NAME
    metrics_path = sealed_run / METRICS_NAME
    metrics = _read_json(metrics_path)
    diagnostics = metrics.get("training_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise TypeError(f"{metrics_path}: training_diagnostics must be an object")
    start_value = diagnostics.get("evaluation_start_local_date")
    end_value = diagnostics.get("evaluation_end_local_date")
    if not start_value or not end_value:
        raise ValueError(f"{metrics_path}: evaluation window is missing")
    start_day = pd.Timestamp(start_value).date()
    end_day = pd.Timestamp(end_value).date()
    if end_day < start_day:
        raise ValueError(f"{metrics_path}: invalid evaluation window")
    raw = pd.read_csv(backtest_path)
    models = tuple(dict.fromkeys(("residual_corrected", str(candidate_model))))
    required = {
        "delivery_start_utc",
        "actual",
        *(f"{model}__{q}" for model in models for q in QUANTILES),
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{backtest_path}: missing columns {missing}")
    delivery = pd.DatetimeIndex(
        pd.to_datetime(raw["delivery_start_utc"], utc=True, errors="raise")
    )
    local_day = pd.Index(delivery.tz_convert(timezone).date)
    benchmark = raw.loc[(local_day >= start_day) & (local_day <= end_day)].copy()
    selected_index = pd.DatetimeIndex(
        pd.to_datetime(benchmark["delivery_start_utc"], utc=True, errors="raise")
    )
    expected = pd.DatetimeIndex([], tz="UTC")
    for value in pd.date_range(start_day, end_day, freq="D"):
        expected = expected.append(
            local_delivery_day_index(value.date(), timezone=timezone)
        )
    if not selected_index.equals(expected):
        raise ValueError(f"{backtest_path}: evaluation timeline is not exact")
    finite_columns = [
        "actual",
        *(f"{model}__{q}" for model in models for q in QUANTILES),
    ]
    if "storm_evaluation_only__q50" in benchmark:
        finite_columns.append("storm_evaluation_only__q50")
    # The exact native dashboard representation may intentionally omit one
    # autumn DST fold.  Its coverage is validated by the Storm audit, not by
    # the candidate's all-finite requirement below.
    values = benchmark[finite_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{backtest_path}: incomplete sealed evaluation")
    source_sha = _sha256(backtest_path)
    benchmark["statistics_scope"] = "sealed_benchmark"
    benchmark["statistics_run_type"] = "sealed_benchmark"
    benchmark["statistics_source_run"] = str(sealed_run)
    benchmark["statistics_source_sha256"] = source_sha
    return benchmark, start_day, end_day, {
        "statistics_scope": "sealed_benchmark",
        "source_run": str(sealed_run),
        "source_path": str(backtest_path),
        "source_sha256": source_sha,
        "evaluation_start_local_date": start_day.isoformat(),
        "evaluation_end_local_date": end_day.isoformat(),
        "hours": int(len(benchmark)),
        "days": int((end_day - start_day).days + 1),
        "storm_dashboard": (
            _read_json(sealed_run / STATISTICS_AUDIT_NAME).get(
                "storm_dashboard"
            )
            if (sealed_run / STATISTICS_AUDIT_NAME).is_file()
            else None
        ),
    }


def _realized_rows(
    *,
    base_columns: list[str],
    forecasts: Mapping[date, pd.DataFrame],
    forecast_audits: Mapping[date, Mapping[str, Any]],
    canonical_target: pd.Series,
    storm_pit_path: Path,
    current_delivery_day: date,
    timezone: str,
    candidate_model: str = "mkonline_blend",
    storm_strict_08_series: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not forecasts:
        return (
            pd.DataFrame(columns=[*base_columns, *PROVENANCE_COLUMNS]),
            pd.DataFrame(),
            {},
        )
    forecast = pd.concat([forecasts[day] for day in sorted(forecasts)])
    delivery = pd.DatetimeIndex(forecast.index, name="delivery_start_utc")
    if delivery.has_duplicates or not delivery.is_monotonic_increasing:
        raise ValueError("Archived forecast history has an invalid timeline")
    local_day = pd.Index(delivery.tz_convert(timezone).date)
    if bool((local_day >= current_delivery_day).any()):
        raise ValueError("Current/future delivery leaked into Statistics")
    actual = _canonical_target_utc(canonical_target).reindex(delivery)
    if not np.isfinite(actual.to_numpy(dtype=float)).all():
        raise ValueError("Canonical actuals are incomplete for Statistics")
    # Storm is attached only after all candidate forecasts are frozen.
    # The old strict-08 curve is diagnostic only. Its local vintage store may
    # lag behind the native dashboard extraction; this must never block the
    # official comparison or become a silent fill value.
    try:
        storm, storm_selected, storm_audit = _load_storm_evaluation_only(
            storm_pit_path, expected_index=delivery
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        storm = pd.Series(np.nan, index=delivery, name=STORM_LEGACY_STRICT_COLUMN)
        storm_selected = pd.DataFrame()
        storm_audit = {
            "status": "diagnostic_unavailable",
            "reason": str(exc),
            "series": storm_strict_08_series,
            "used_for_prediction": False,
        }
    payload_columns = [column for column in base_columns if column != "delivery_start_utc"]
    rows = pd.DataFrame(index=delivery, columns=payload_columns)
    candidate_columns = [
        f"{candidate_model}__{q}" for q in QUANTILES
    ]
    passthrough = [
        "forecast_origin_utc",
        f"{candidate_model}_forecast_origin_utc",
        "mkonline_primary__q50",
        "mkonline_blend_shift",
        "residual_correction",
        *(f"chronos2__{q}" for q in QUANTILES),
        *(f"residual_corrected__{q}" for q in QUANTILES),
        *candidate_columns,
    ]
    for column in passthrough:
        if column in rows and column in forecast:
            rows[column] = forecast[column].to_numpy()
    if "forecast_origin_utc" in rows:
        generic = (
            forecast["forecast_origin_utc"]
            if "forecast_origin_utc" in forecast
            else pd.Series(pd.NaT, index=forecast.index)
        )
        qualified_column = f"{candidate_model}_forecast_origin_utc"
        qualified = (
            forecast[qualified_column]
            if qualified_column in forecast
            else generic
        )
        rows["forecast_origin_utc"] = generic.where(
            generic.notna(), qualified
        ).to_numpy()
    rows["actual"] = actual.to_numpy(dtype=float)
    if STORM_LEGACY_STRICT_COLUMN in rows:
        rows[STORM_LEGACY_STRICT_COLUMN] = storm.to_numpy(dtype=float)
    if "fold_id" in rows:
        rows["fold_id"] = np.nan
    for day, audit in forecast_audits.items():
        selector = local_day == day
        rows.loc[selector, "statistics_scope"] = audit["statistics_scope"]
        rows.loc[selector, "statistics_run_type"] = audit["run_type"]
        rows.loc[selector, "statistics_source_run"] = audit["source_run"]
        rows.loc[selector, "statistics_source_sha256"] = audit["forecast_sha256"]
    rows.insert(0, "delivery_start_utc", delivery.astype(str))
    return rows.reset_index(drop=True), storm_selected, storm_audit


def update_live_statistics_history(
    *,
    staging_run_dir: str | Path,
    sealed_benchmark_run: str | Path,
    live_output_root: str | Path,
    replay_output_root: str | Path,
    current_delivery_day: date,
    canonical_target: pd.Series,
    storm_pit_path: str | Path,
    storm_dashboard_native: pd.Series | None = None,
    storm_dashboard_source: Mapping[str, Any] | None = None,
    zone: str = "FR",
    timezone: str = TIMEZONE,
    forecast_name: str = FORECAST_NAME,
    candidate_model: str = "mkonline_blend",
    storm_strict_08_series: str | None = None,
    target_series: str = "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh",
    prediction_mode: str = "mkonline_blend",
    allow_partial_prefix: bool = False,
    statistics_blocker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a strictly contiguous Statistics source and explicit audit.

    The default remains fail-closed on any archive gap.  Operational live
    recovery may explicitly request ``allow_partial_prefix`` after a causal
    auto-replay failure; in that mode only the complete civil-day prefix before
    the first gap is evaluated.  Later archives are never jumped over or used
    to fill the hole.
    """

    staging = Path(staging_run_dir).expanduser().resolve()
    staging.mkdir(parents=True, exist_ok=True)
    sealed = Path(sealed_benchmark_run).expanduser().resolve()
    benchmark, benchmark_start, benchmark_end, benchmark_audit = (
        _sealed_benchmark_statistics(
            sealed,
            timezone=timezone,
            candidate_model=candidate_model,
        )
    )
    first_history_day = benchmark_end + pd.Timedelta(days=1)
    last_eligible_day = current_delivery_day - pd.Timedelta(days=1)
    forecasts, source_audit = discover_archived_forecasts(
        live_output_root=live_output_root,
        replay_output_root=replay_output_root,
        current_delivery_day=current_delivery_day,
        first_history_day=first_history_day,
        timezone=timezone,
        forecast_name=forecast_name,
        candidate_model=candidate_model,
        zone=zone,
        target_series=target_series,
        prediction_mode=prediction_mode,
    )
    expected_days = (
        [value.date() for value in pd.date_range(first_history_day, last_eligible_day, freq="D")]
        if first_history_day <= last_eligible_day
        else []
    )
    missing_days = [day for day in expected_days if day not in forecasts]
    if missing_days and not allow_partial_prefix:
        raise ValueError(
            "Forecast archive is incomplete for Statistics; create causal "
            "PIT replay run(s): "
            + ", ".join(day.isoformat() for day in missing_days)
        )
    first_gap = missing_days[0] if missing_days else None
    evaluated_days = (
        [day for day in expected_days if day < first_gap]
        if first_gap is not None
        else expected_days
    )
    excluded_after_first_gap = [
        day for day in expected_days if first_gap is not None and day > first_gap
    ]
    selected = {day: forecasts[day] for day in evaluated_days}
    audits_by_day = {
        pd.Timestamp(item["delivery_day_local"]).date(): item
        for item in source_audit
        if pd.Timestamp(item["delivery_day_local"]).date() in selected
    }
    realized, storm_selected, storm_audit = _realized_rows(
        base_columns=[c for c in benchmark.columns if c not in PROVENANCE_COLUMNS],
        forecasts=selected,
        forecast_audits=audits_by_day,
        canonical_target=canonical_target,
        storm_pit_path=Path(storm_pit_path).expanduser().resolve(),
        current_delivery_day=current_delivery_day,
        timezone=timezone,
        candidate_model=candidate_model,
        storm_strict_08_series=storm_strict_08_series,
    )
    populated_realized = realized.dropna(axis=1, how="all")
    history = pd.concat(
        [benchmark, populated_realized], ignore_index=True, sort=False
    )
    history["_sort"] = pd.to_datetime(
        history["delivery_start_utc"], utc=True, errors="raise"
    )
    history = history.sort_values("_sort", kind="stable").drop(columns="_sort").reset_index(drop=True)
    delivery = pd.DatetimeIndex(
        pd.to_datetime(history["delivery_start_utc"], utc=True, errors="raise")
    )
    statistics_end_day = (
        evaluated_days[-1] if evaluated_days else benchmark_end
    )
    expected_full = pd.DatetimeIndex([], tz="UTC")
    for value in pd.date_range(benchmark_start, statistics_end_day, freq="D"):
        expected_full = expected_full.append(
            local_delivery_day_index(value.date(), timezone=timezone)
        )
    if delivery.has_duplicates or not delivery.equals(expected_full):
        raise ValueError("Statistics history is not a contiguous civil-day series")
    if storm_dashboard_native is not None:
        actual_history = pd.Series(
            pd.to_numeric(history["actual"], errors="coerce").to_numpy(float),
            index=delivery,
            name="actual",
        )
        dashboard = normalize_native_dashboard_series(
            storm_dashboard_native,
            zone=zone,
            expected_index=delivery,
            actual=actual_history,
            source=storm_dashboard_source,
        )
        legacy_strict = (
            history[STORM_LEGACY_STRICT_COLUMN]
            if STORM_LEGACY_STRICT_COLUMN in history
            else pd.Series(np.nan, index=history.index)
        )
        history[STORM_STRICT_08_COLUMN] = pd.to_numeric(
            legacy_strict, errors="coerce"
        )
        history[STORM_DASHBOARD_COLUMN] = dashboard.values.to_numpy(float)
    elif STORM_DASHBOARD_COLUMN in history:
        dashboard_values = pd.to_numeric(
            history[STORM_DASHBOARD_COLUMN], errors="coerce"
        )
        # Preserve the exact official comparator already sealed in the
        # benchmark.  No live-day extension is needed until a realized archive
        # is appended; in that case a fresh native extraction is mandatory.
        if len(realized) or not np.isfinite(dashboard_values.to_numpy(float)).any():
            history = history.drop(columns=[STORM_DASHBOARD_COLUMN])
            dashboard = None
        else:
            dashboard = None
            existing_dashboard_audit = benchmark_audit.get("storm_dashboard")
            if not isinstance(existing_dashboard_audit, Mapping):
                existing_statistics_audit = _read_json(
                    sealed / STATISTICS_AUDIT_NAME
                )
                existing_dashboard_audit = existing_statistics_audit.get(
                    "storm_dashboard"
                )
            if isinstance(existing_dashboard_audit, Mapping):
                audit_dashboard = dict(existing_dashboard_audit)
            else:
                audit_dashboard = None
    else:
        dashboard = None
    history_path = staging / STATISTICS_HISTORY_NAME
    history.to_csv(history_path, index=False, compression="gzip")
    if not realized.empty:
        storm_output = staging / "inputs" / "storm_evaluation_only_live_history.parquet"
        storm_output.parent.mkdir(parents=True, exist_ok=True)
        storm_selected.to_parquet(storm_output, index=False)
    if dashboard is not None:
        dashboard_output = staging / STORM_DASHBOARD_ARTIFACT
        dashboard_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "delivery_start_utc": delivery,
                STORM_DASHBOARD_COLUMN: dashboard.values.to_numpy(float),
            }
        ).to_parquet(dashboard_output, index=False)
    audit: dict[str, Any] = {
        "status": (
            "partial_contiguous_prefix" if missing_days else "complete"
        ),
        "mode": "separate_sealed_benchmark_plus_realized_archive",
        "statistics_history_path": STATISTICS_HISTORY_NAME,
        "statistics_history_sha256": _sha256(history_path),
        "statistics_audit_path": STATISTICS_AUDIT_NAME,
        "sealed_benchmark": benchmark_audit,
        "current_delivery_day_local": current_delivery_day.isoformat(),
        "eligibility_rule": "delivery_day_local < current_delivery_day_local",
        "first_required_realized_day": (
            first_history_day.isoformat() if expected_days else None
        ),
        "last_required_realized_day": (
            last_eligible_day.isoformat() if expected_days else None
        ),
        "evaluated_realized_days": [day.isoformat() for day in evaluated_days],
        "missing_realized_days": [day.isoformat() for day in missing_days],
        "excluded_after_first_gap_days": [
            day.isoformat() for day in excluded_after_first_gap
        ],
        "statistics_complete": not missing_days,
        "statistics_prefix_end_local": statistics_end_day.isoformat(),
        "n_evaluated_realized_days": len(evaluated_days),
        "n_evaluated_realized_hours": int(len(realized)),
        "n_total_statistics_hours": int(len(history)),
        "forecast_sources": source_audit,
        "canonical_actuals_complete": True,
        "storm": storm_audit,
        "storm_used_for_prediction": False,
        "historical_forecasts_rewritten": False,
        "sealed_benchmark_rewritten": False,
        "staging_backtest_rewritten": False,
        "staging_metrics_rewritten": False,
    }
    if statistics_blocker is not None:
        audit["statistics_blocker"] = dict(statistics_blocker)
    if dashboard is not None:
        dashboard_audit = dict(dashboard.audit)
        dashboard_audit["normalized_artifact_path"] = (
            STORM_DASHBOARD_ARTIFACT.as_posix()
        )
        dashboard_audit["normalized_artifact_sha256"] = _sha256(
            staging / STORM_DASHBOARD_ARTIFACT
        )
        audit.update(
            {
                "storm_primary_report_benchmark": STORM_DASHBOARD_COLUMN,
                "storm_dashboard": dashboard_audit,
                "storm_strict_08": {
                    "column": STORM_STRICT_08_COLUMN,
                    "series": storm_audit.get("series"),
                    "cutoff": storm_audit.get("cutoff"),
                    "used_for_prediction": False,
                },
            }
        )
    elif "audit_dashboard" in locals() and audit_dashboard is not None:
        audit.update(
            {
                "storm_primary_report_benchmark": STORM_DASHBOARD_COLUMN,
                "storm_dashboard": audit_dashboard,
            }
        )
    replay_days = [
        item["delivery_day_local"]
        for item in source_audit
        if item.get("statistics_scope") == "realized_pit_replay"
        and pd.Timestamp(item["delivery_day_local"]).date() in selected
    ]
    issued_days = [
        item["delivery_day_local"]
        for item in source_audit
        if item.get("statistics_scope") == "realized_live"
        and pd.Timestamp(item["delivery_day_local"]).date() in selected
    ]
    note_parts = [
        "benchmark scelle "
        f"{benchmark_start.isoformat()} au {benchmark_end.isoformat()}"
    ]
    if replay_days:
        note_parts.append(
            "replays PIT causaux (non emis en temps reel) : "
            + ", ".join(replay_days)
        )
    if issued_days:
        note_parts.append("forecasts live emis : " + ", ".join(issued_days))
    if missing_days:
        note_parts.append(
            "Statistics partielles, prefixe causal arrete au "
            f"{statistics_end_day.isoformat()}; archives manquantes : "
            + ", ".join(day.isoformat() for day in missing_days)
        )
    note_parts.append(
        "Storm est joint apres gel du candidat, uniquement pour l'evaluation"
    )
    if dashboard is not None:
        note_parts.append(
            f"benchmark principal {storm_dashboard_series(zone)} natif; "
            "heures DST absentes non interpolees"
        )
    audit["realized_pit_replay_days"] = replay_days
    audit["realized_live_days"] = issued_days
    audit["report_scope_note"] = "; ".join(note_parts) + "."
    _write_json(staging / STATISTICS_AUDIT_NAME, audit)
    return audit


__all__ = (
    "STATISTICS_AUDIT_NAME",
    "STATISTICS_HISTORY_NAME",
    "discover_archived_forecasts",
    "missing_statistics_archive_days",
    "update_live_statistics_history",
)
