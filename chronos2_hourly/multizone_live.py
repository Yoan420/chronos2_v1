"""Strict, zone-parameterised live day-ahead execution.

The sealed French runner remains untouched.  This module is the independent
runtime used by production-ready BE/DE/NL/ES bundles.  It accepts only a
fully verified :class:`~chronos2_hourly.multizone_contract.ZoneModelContract`;
there is no France default and no implicit fallback from MKOnline to the
autonomous model.

Storm is deliberately outside the candidate-building API.  The optional
day-ahead dashboard curve is fetched only after the forecast CSV has been
written to the private staging directory, and is used for reporting only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date
import hashlib
from html import escape
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.chronos_adapter import (
    ChronosDeliveryPlan,
    run_existing_live_forecast,
)
from chronos2_hourly.hourly_contract import (
    HourlyTargetContractError,
    build_delivery_metadata,
    local_delivery_day_index,
)
from chronos2_hourly.live_target_context import audit_live_target_context
from chronos2_hourly.multizone_contract import ZoneModelContract
from chronos2_hourly.shadow_reporting import (
    write_forecast_only_shadow_report,
)
from chronos2_hourly.variable_attribution import (
    remove_variable_attribution_artifacts,
    write_variable_attribution,
)
from chronos2_modular.common import (
    build_zone_configs,
    deep_get,
    load_yaml,
    set_reproducibility,
)


LOGGER = logging.getLogger("multizone_live")
SCRIPT_VERSION = "1.1.1-strict-multizone-live"
QUANTILES = ("q10", "q50", "q90")
EXPECTED_META_FEATURES = 175
LIVE_SYNC_LOOKBACK_HOURS = 48
MAX_AUTOMATIC_REPLAY_DAYS = 3
PREDICTION_MODES = frozenset({"mkonline_blend", "autonomous_only"})
RESIDUAL_LOAD_SOURCES = frozenset({"saturn", "chronos2"})
CHRONOS2_RESIDUAL_ARCHIVE_SUFFIX = "_residual_load_chronos2"
CHRONOS2_RESIDUAL_ARCHIVE_SUBDIR = Path(
    "_challengers/residual_load_chronos2"
)
ATOMIC_PUBLISH_ATTEMPTS = 5
ATOMIC_PUBLISH_RETRY_SECONDS = 0.25

BENCHMARK_REPORT_FILES = (
    "backtest_hourly_oof.csv.gz",
    "metrics_hourly.csv",
    "metrics_hourly.json",
    "feature_manifest.csv",
    "ensemble_weights.csv",
    "pit_feature_coverage_by_hour.csv",
    "pit_feature_coverage_summary.csv",
    "evaluation_by_day.csv",
    "evaluation_by_hour.csv",
    "evaluation_by_month.csv",
    "evaluation_summary.json",
    "evaluation_vs_storm_by_day.csv",
    "evaluation_vs_storm_by_hour.csv",
    "evaluation_vs_storm_by_month.csv",
    "evaluation_vs_storm_summary.json",
)


class ZoneLiveExecutionError(ValueError):
    """Raised when a verified bundle violates the live runtime contract."""


def _publish_staging_atomically(
    staging: Path,
    output: Path,
    *,
    attempts: int = ATOMIC_PUBLISH_ATTEMPTS,
    retry_seconds: float = ATOMIC_PUBLISH_RETRY_SECONDS,
) -> None:
    """Publish one immutable run, tolerating only transient Windows locks."""

    if attempts < 1:
        raise ValueError("attempts must be at least one")
    for attempt in range(1, attempts + 1):
        try:
            staging.replace(output)
            return
        except PermissionError:
            # Never turn a real archive collision into an overwrite attempt.
            if output.exists() or attempt == attempts:
                raise
            LOGGER.warning(
                "Atomic publish temporarily locked (%s/%s): %s -> %s",
                attempt,
                attempts,
                staging,
                output,
            )
            time.sleep(retry_seconds * attempt)


@dataclass(frozen=True)
class LiveSchedule:
    """One physical delivery day and its civil auction cutoff."""

    as_of_origin_local: pd.Timestamp
    delivery_day: date
    cutoff_origin_local: pd.Timestamp
    delivery_index: pd.DatetimeIndex


@dataclass(frozen=True)
class PredictionPolicy:
    """Explicit candidate mode; autonomous fallback is never inferred."""

    mode: str
    mkonline_enabled: bool
    candidate_model: str


@dataclass
class CandidateArtifacts:
    """Prediction-path result, intentionally containing no Storm data."""

    forecast: pd.DataFrame
    canonical_target: pd.Series
    fit_audit: dict[str, Any]
    input_diagnostics: dict[str, Any]
    pit_freshness: dict[str, Any]
    source_paths: dict[str, Path]
    data_config: Mapping[str, Any]
    rolling_capture_features: pd.DataFrame | None = None
    rolling_capture_chronos: pd.DataFrame | None = None
    attribution_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class LiveRuntimeOptions:
    """Runtime-only switches that cannot alter the sealed recipe."""

    device: str | None = None
    threads: int = -1
    workers: int = 8
    local_files_only: bool = False
    refresh_data: bool = True
    rolling365_capture_root: Path | None = None
    residual_load_source: str = "saturn"
    residual_load_bundle_manifest: Path | None = None


CandidateBuilder = Callable[
    [ZoneModelContract, LiveSchedule, PredictionPolicy, Mapping[str, Any], LiveRuntimeOptions, Path],
    CandidateArtifacts,
]
DashboardFetcher = Callable[
    [ZoneModelContract, LiveSchedule, Mapping[str, Any]],
    tuple[pd.Series | None, Mapping[str, Any] | None],
]
StatisticsWriter = Callable[..., Mapping[str, Any]]
ReportWriter = Callable[..., Path]
StatisticsGapPlanner = Callable[..., Sequence[date]]


@dataclass(frozen=True)
class ZoneLiveHooks:
    """Injectable seams used by tests; production defaults are below."""

    build_candidate: CandidateBuilder
    fetch_dashboard: DashboardFetcher
    update_statistics: StatisticsWriter
    write_report: ReportWriter
    plan_statistics_gaps: StatisticsGapPlanner | None = None


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ZoneLiveExecutionError(f"{name} must be an explicit mapping")
    return value


def _residual_runtime(options: LiveRuntimeOptions) -> tuple[str, Path | None]:
    source = str(options.residual_load_source or "saturn").strip().lower()
    if source not in RESIDUAL_LOAD_SOURCES:
        raise ZoneLiveExecutionError(
            "residual_load_source must be saturn or chronos2"
        )
    manifest = options.residual_load_bundle_manifest
    if source == "chronos2":
        if manifest is None:
            raise ZoneLiveExecutionError(
                "Chronos-2 residual load requires a bundle manifest"
            )
        manifest = Path(manifest).expanduser().resolve()
        if not manifest.is_file():
            raise ZoneLiveExecutionError(
                f"Chronos-2 residual-load bundle manifest is missing: {manifest}"
            )
        if options.rolling365_capture_root is not None:
            raise ZoneLiveExecutionError(
                "rolling-365 capture is reserved for the production Saturn run"
            )
    elif manifest is not None:
        raise ZoneLiveExecutionError(
            "a residual-load bundle cannot be supplied with source=saturn"
        )
    return source, manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path, date)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _reporting_error(stage: str, exc: Exception) -> dict[str, str]:
    return {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _write_degraded_html_report(
    path: Path,
    *,
    title: str,
    zone: str,
    delivery_day: date,
    forecast_name: str,
    frozen_candidate_sha256: str,
    reporting_errors: Sequence[Mapping[str, Any]],
) -> None:
    items = "".join(
        "<li><strong>"
        + escape(str(item.get("stage", "reporting")))
        + "</strong> — "
        + escape(str(item.get("error_type", "Error")))
        + ": "
        + escape(str(item.get("error", "")))
        + "</li>"
        for item in reporting_errors
    )
    document = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:3rem auto;
padding:0 1rem;color:#172033}}.ok{{background:#eaf7ee;border-left:5px solid #18864b;
padding:1rem}}.warn{{background:#fff4e5;border-left:5px solid #d97706;
padding:1rem}}code{{word-break:break-all}}</style></head><body>
<h1>{escape(title)}</h1>
<div class="ok"><strong>Le forecast {escape(zone)} du
{escape(delivery_day.isoformat())} a bien été publié.</strong><br>
Artefact : <code>{escape(forecast_name)}</code><br>
SHA-256 gelé : <code>{escape(frozen_candidate_sha256)}</code></div>
<div class="warn"><h2>Rapport dégradé</h2><p>Une étape postérieure au gel du
forecast a échoué. Cette erreur n’a pas modifié les prévisions.</p><ul>{items}</ul>
<p>Les diagnostics complets sont disponibles dans
<code>reporting_errors.json</code>.</p></div></body></html>"""
    path.write_text(document, encoding="utf-8")


def _remove_optional_statistics_artifacts(staging: Path) -> None:
    for relative in (
        "statistics_history_hourly.csv.gz",
        "statistics_history_audit.json",
        "inputs/storm_evaluation_only_live_history.parquet",
        "inputs/storm_dashboard_official_statistics.parquet",
    ):
        (staging / relative).unlink(missing_ok=True)


def _resolved_report_settings(
    contract: ZoneModelContract,
    schedule: LiveSchedule,
    live_settings: Mapping[str, Any],
) -> dict[str, Any]:
    report = _mapping(live_settings.get("report", {}), name="live.report")
    filename = str(
        report.get(
            "filename",
            f"chronos2_hourly_{contract.zone.lower()}_live_"
            "{delivery_day}.html",
        )
    ).format(delivery_day=schedule.delivery_day.isoformat())
    filename_path = Path(filename)
    if (
        filename_path.is_absolute()
        or filename_path.name != filename
        or filename_path.suffix.casefold() != ".html"
    ):
        raise ZoneLiveExecutionError(
            "live.report.filename must be one local .html filename"
        )
    title = str(
        report.get(
            "title",
            f"Forecast day-ahead {contract.zone} - livraison "
            "{delivery_day}",
        )
    ).format(delivery_day=schedule.delivery_day.isoformat())
    extreme_threshold = float(report.get("extreme_threshold", 150.0))
    history_hours = int(report.get("forecast_history_hours", 168))
    if not np.isfinite(extreme_threshold) or extreme_threshold <= 0.0:
        raise ZoneLiveExecutionError(
            "live.report.extreme_threshold must be positive and finite"
        )
    if history_hours <= 0:
        raise ZoneLiveExecutionError(
            "live.report.forecast_history_hours must be positive"
        )
    return {
        "filename": filename,
        "title": title,
        "extreme_threshold": extreme_threshold,
        "history_hours": history_hours,
    }


def _parse_as_of(
    value: str | pd.Timestamp | None,
    *,
    origin_timezone: str,
) -> pd.Timestamp:
    timestamp = (
        pd.Timestamp.now(tz=origin_timezone)
        if value in (None, "")
        else pd.Timestamp(value)
    )
    if timestamp.tzinfo is None:
        raise ZoneLiveExecutionError(
            "--data-as-of must include an explicit timezone"
        )
    return timestamp.tz_convert(origin_timezone)


def _parse_delivery_day(
    value: str | date | pd.Timestamp | None,
    *,
    delivery_timezone: str,
) -> date | None:
    if value in (None, ""):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(delivery_timezone).tz_localize(None)
    if timestamp != timestamp.normalize():
        raise ZoneLiveExecutionError(
            "--delivery-day must be a civil YYYY-MM-DD date"
        )
    return timestamp.date()


def _forecast_cutoff_for_day(
    contract: ZoneModelContract,
    delivery_day: date,
) -> pd.Timestamp:
    hour, minute = (
        int(item) for item in contract.forecast_origin_local_time.split(":")
    )
    cutoff_naive = (
        pd.Timestamp(delivery_day)
        - pd.DateOffset(days=1)
        + pd.Timedelta(hours=hour, minutes=minute)
    )
    return cutoff_naive.tz_localize(
        contract.forecast_origin_timezone,
        ambiguous="raise",
        nonexistent="raise",
    )


def _wall_clock_in_origin_timezone(
    contract: ZoneModelContract,
    wall_clock: pd.Timestamp | None,
) -> pd.Timestamp:
    now = (
        pd.Timestamp.now(tz=contract.forecast_origin_timezone)
        if wall_clock is None
        else pd.Timestamp(wall_clock)
    )
    if now.tzinfo is None:
        raise ZoneLiveExecutionError("wall_clock must be timezone-aware")
    return now.tz_convert(contract.forecast_origin_timezone)


def resolve_live_schedule(
    contract: ZoneModelContract,
    *,
    as_of: str | pd.Timestamp | None,
    delivery_day: str | date | pd.Timestamp | None,
) -> LiveSchedule:
    """Resolve J+1 at the declared origin timezone and delivery DST grid."""

    as_of_local = _parse_as_of(
        as_of,
        origin_timezone=contract.forecast_origin_timezone,
    )
    expected_day = (as_of_local.normalize() + pd.DateOffset(days=1)).date()
    requested = _parse_delivery_day(
        delivery_day,
        delivery_timezone=contract.delivery_timezone,
    )
    selected = requested or expected_day
    if selected != expected_day:
        raise ZoneLiveExecutionError(
            f"{contract.zone}: live mode accepts J+1 only; expected "
            f"{expected_day}, received {selected}"
        )
    cutoff = _forecast_cutoff_for_day(contract, selected)
    if as_of_local < cutoff:
        raise ZoneLiveExecutionError(
            f"{contract.zone}: run refused before civil cutoff {cutoff}"
        )
    delivery_index = local_delivery_day_index(
        selected,
        timezone=contract.delivery_timezone,
    )
    return LiveSchedule(
        as_of_origin_local=as_of_local,
        delivery_day=selected,
        cutoff_origin_local=cutoff,
        delivery_index=delivery_index,
    )


def load_prediction_policy(
    contract: ZoneModelContract,
    live: Mapping[str, Any],
) -> PredictionPolicy:
    """Validate the same explicit mode in live config and frozen recipe."""

    if "prediction_mode" not in live or "mkonline_enabled" not in live:
        raise ZoneLiveExecutionError(
            "live.prediction_mode and live.mkonline_enabled must be declared; "
            "an autonomous fallback is never inferred"
        )
    mode = str(live["prediction_mode"]).strip()
    if mode not in PREDICTION_MODES:
        raise ZoneLiveExecutionError(
            f"live.prediction_mode must be one of {sorted(PREDICTION_MODES)}"
        )
    if not isinstance(live["mkonline_enabled"], bool):
        raise ZoneLiveExecutionError("live.mkonline_enabled must be boolean")
    enabled = bool(live["mkonline_enabled"])
    if mode != contract.prediction_mode:
        raise ZoneLiveExecutionError("live.prediction_mode differs from contract")
    if enabled is not contract.mkonline_enabled:
        raise ZoneLiveExecutionError("live.mkonline_enabled differs from contract")
    recipe = json.loads(contract.paths.recipe_manifest.read_text(encoding="utf-8"))
    if not isinstance(recipe, Mapping):
        raise ZoneLiveExecutionError("recipe manifest must contain a JSON object")
    recipe_mode = recipe.get("prediction_mode", recipe.get("recipe_mode"))
    if recipe_mode != mode:
        raise ZoneLiveExecutionError("recipe.prediction_mode/recipe_mode mismatch")
    if recipe.get("mkonline_enabled") is not enabled:
        raise ZoneLiveExecutionError("recipe.mkonline_enabled mismatch")

    auto = contract.weights.autonomous
    mk = contract.weights.mkonline_primary
    if mode == "mkonline_blend":
        if not enabled or not (0.0 < auto < 1.0 and 0.0 < mk < 1.0):
            raise ZoneLiveExecutionError(
                "mkonline_blend requires explicit enabled=true and two positive weights"
            )
        candidate_model = "mkonline_blend"
    else:
        if enabled or auto != 1.0 or mk != 0.0:
            raise ZoneLiveExecutionError(
                "autonomous_only requires enabled=false and weights 1.0/0.0"
            )
        validation = recipe.get("autonomous_validation")
        validation_reason = (
            str(validation.get("reason", "")).strip()
            if isinstance(validation, Mapping)
            else ""
        )
        reason = str(recipe.get("autonomous_only_reason", "")).strip() or validation_reason
        if not reason:
            raise ZoneLiveExecutionError(
                "autonomous_only requires an explicit recipe reason"
            )
        candidate_model = "residual_corrected"
    return PredictionPolicy(
        mode=mode,
        mkonline_enabled=enabled,
        candidate_model=candidate_model,
    )


def _numeric_quantiles(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = [column for column in QUANTILES if column not in frame]
    if missing:
        raise ZoneLiveExecutionError(f"{name}: missing quantiles {missing}")
    result = frame.loc[:, list(QUANTILES)].apply(pd.to_numeric, errors="coerce")
    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ZoneLiveExecutionError(f"{name}: non-finite quantiles")
    if not bool(
        ((values[:, 0] <= values[:, 1]) & (values[:, 1] <= values[:, 2])).all()
    ):
        raise ZoneLiveExecutionError(f"{name}: crossed quantiles")
    return result.astype(float)


def validate_live_forecast(
    frame: pd.DataFrame,
    *,
    schedule: LiveSchedule,
    candidate_model: str,
) -> pd.DataFrame:
    """Validate exact 23/24/25 physical coverage and causal origins."""

    required = {"delivery_start_utc", "forecast_origin_utc", *QUANTILES}
    required.update(f"{candidate_model}__{q}" for q in QUANTILES)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ZoneLiveExecutionError(f"live forecast is incomplete: {missing}")
    result = frame.copy()
    delivery = pd.DatetimeIndex(
        pd.to_datetime(result["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if not delivery.equals(schedule.delivery_index):
        raise ZoneLiveExecutionError(
            "live forecast does not match the exact 23/24/25 delivery timeline"
        )
    origins = pd.DatetimeIndex(
        pd.to_datetime(result["forecast_origin_utc"], utc=True, errors="raise")
    )
    cutoff_utc = schedule.cutoff_origin_local.tz_convert("UTC")
    if bool((origins >= delivery).any()) or bool((origins > cutoff_utc).any()):
        raise ZoneLiveExecutionError("live forecast contains a non-causal origin")
    result.loc[:, list(QUANTILES)] = _numeric_quantiles(
        result,
        name="live forecast",
    )
    model_values = result.loc[
        :, [f"{candidate_model}__{q}" for q in QUANTILES]
    ].copy()
    model_values.columns = list(QUANTILES)
    validated_model = _numeric_quantiles(
        model_values,
        name=f"live forecast {candidate_model}",
    )
    if not np.allclose(
        result.loc[:, list(QUANTILES)].to_numpy(dtype=float),
        validated_model.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ZoneLiveExecutionError(
            "generic live quantiles differ from the declared candidate model"
        )
    result["delivery_start_utc"] = delivery
    result["forecast_origin_utc"] = origins
    return result


def blend_quantiles(
    autonomous: pd.DataFrame,
    primary_q50: pd.Series,
    *,
    autonomous_weight: float,
    primary_weight: float,
) -> pd.DataFrame:
    """Apply a sealed convex q50 blend and preserve interval width."""

    if not np.isclose(
        float(autonomous_weight) + float(primary_weight),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ZoneLiveExecutionError("blend weights must sum to one")
    auto = _numeric_quantiles(autonomous, name="autonomous prediction")
    mk = pd.to_numeric(primary_q50, errors="coerce").reindex(auto.index)
    if not np.isfinite(mk.to_numpy(dtype=float)).all():
        raise ZoneLiveExecutionError("primary prediction is incomplete")
    q50 = autonomous_weight * auto["q50"] + primary_weight * mk
    shift = q50 - auto["q50"]
    result = pd.DataFrame(index=auto.index)
    for quantile in QUANTILES:
        result[quantile] = auto[quantile] + shift
    result["shift"] = shift
    result["primary_q50"] = mk
    _numeric_quantiles(result, name="blended prediction")
    before = auto["q90"] - auto["q10"]
    after = result["q90"] - result["q10"]
    if not np.allclose(before, after, rtol=0.0, atol=1e-10):
        raise RuntimeError("common shift changed prediction interval width")
    return result


def _expected_primary_cutoff(
    delivery: pd.DatetimeIndex,
    *,
    delivery_timezone: str,
    origin_timezone: str,
    origin_time: str,
) -> pd.DatetimeIndex:
    local_days = pd.DatetimeIndex(delivery.tz_convert(delivery_timezone).date)
    hour, minute = (int(item) for item in origin_time.split(":"))
    civil = (
        local_days
        - pd.DateOffset(days=1)
        + pd.Timedelta(hours=hour, minutes=minute)
    )
    return civil.tz_localize(
        origin_timezone,
        ambiguous="raise",
        nonexistent="raise",
    ).tz_convert("UTC")


def load_primary_materialization(
    path: Path,
    *,
    contract: ZoneModelContract,
    schedule: LiveSchedule,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Load one exact primary curve; interpolation and formulas are forbidden."""

    frame = pd.read_parquet(path)
    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ZoneLiveExecutionError(f"{path}: missing PIT columns {missing}")
    delivery = pd.DatetimeIndex(
        pd.to_datetime(frame["value_time_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if not delivery.equals(schedule.delivery_index):
        raise ZoneLiveExecutionError(f"{path}: exact delivery timeline mismatch")
    snapshot = pd.DatetimeIndex(
        pd.to_datetime(frame["snapshot_time_utc"], utc=True, errors="raise")
    )
    revision = pd.DatetimeIndex(
        pd.to_datetime(frame["revision_time_utc"], utc=True, errors="raise")
    )
    expected_cutoff = _expected_primary_cutoff(
        delivery,
        delivery_timezone=contract.delivery_timezone,
        origin_timezone=contract.forecast_origin_timezone,
        origin_time=contract.forecast_origin_local_time,
    )
    if not snapshot.equals(expected_cutoff) or not revision.equals(expected_cutoff):
        raise ZoneLiveExecutionError(
            f"{path}: PIT marker differs from declared civil cutoff"
        )
    values = pd.Series(
        pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float),
        index=delivery,
        name="mkonline_primary__q50",
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ZoneLiveExecutionError(f"{path}: non-finite primary values")
    cutoff = pd.Series(snapshot, index=delivery, name="mkonline_cutoff_utc")
    return values, cutoff, {
        "series": contract.primary_series,
        "path": str(path),
        "sha256": _sha256(path),
        "hours": int(len(frame)),
        "coverage": 1.0,
        "cutoff": (
            f"civil D-1 {contract.forecast_origin_local_time} "
            f"{contract.forecast_origin_timezone}"
        ),
        "cutoff_violations": 0,
        "interpolation": False,
    }


def materialize_primary(
    *,
    project_root: Path,
    contract: ZoneModelContract,
    schedule: LiveSchedule,
    output: Path,
    workers: int,
) -> list[str]:
    """Materialise only the exact terminal primary declared by the contract."""

    if contract.primary_series is None:
        raise ZoneLiveExecutionError(
            "primary materialization is forbidden without a declared primary"
        )

    command = [
        sys.executable,
        str(project_root / "materialize_saturn_daily_asof.py"),
        "--series",
        str(contract.primary_series),
        "--alias",
        f"mkonline_{contract.zone.lower()}_primary",
        "--start-day",
        schedule.delivery_day.isoformat(),
        "--end-day",
        schedule.delivery_day.isoformat(),
        "--output",
        str(output),
        "--timezone",
        contract.delivery_timezone,
        "--cutoff-timezone",
        contract.forecast_origin_timezone,
        "--cutoff-time",
        contract.forecast_origin_local_time,
        "--workers",
        str(workers),
        "--hourly-on-the-hour",
    ]
    subprocess.run(command, cwd=project_root, check=True)
    return command


def _forecast_frame(
    *,
    contract: ZoneModelContract,
    schedule: LiveSchedule,
    policy: PredictionPolicy,
    chronos: pd.DataFrame,
    autonomous: pd.DataFrame,
    primary: pd.Series | None,
    primary_cutoff: pd.Series | None,
) -> pd.DataFrame:
    autonomous = _numeric_quantiles(autonomous, name="autonomous live")
    if policy.mode == "mkonline_blend":
        if primary is None or primary_cutoff is None:
            raise ZoneLiveExecutionError(
                "mkonline_blend cannot run without the declared primary"
            )
        candidate = blend_quantiles(
            autonomous,
            primary,
            autonomous_weight=contract.weights.autonomous,
            primary_weight=contract.weights.mkonline_primary,
        )
    else:
        if primary is not None or primary_cutoff is not None:
            raise ZoneLiveExecutionError(
                "autonomous_only must not read or receive a primary forecast"
            )
        candidate = autonomous.assign(shift=0.0)

    metadata = build_delivery_metadata(
        schedule.delivery_index,
        timezone=contract.delivery_timezone,
    )
    output = metadata.reset_index(drop=True)
    for quantile in QUANTILES:
        output[quantile] = candidate[quantile].to_numpy(dtype=float)
        output[f"chronos2__{quantile}"] = chronos[quantile].to_numpy(dtype=float)
        output[f"residual_corrected__{quantile}"] = autonomous[
            quantile
        ].to_numpy(dtype=float)
        output[f"{policy.candidate_model}__{quantile}"] = candidate[
            quantile
        ].to_numpy(dtype=float)
    output["price_eur_mwh"] = output["q50"]
    output["residual_correction"] = (
        autonomous["q50"].to_numpy(dtype=float)
        - chronos["q50"].to_numpy(dtype=float)
    )
    output["mkonline_blend_shift"] = candidate["shift"].to_numpy(dtype=float)
    chronos_origin = pd.DatetimeIndex(
        pd.to_datetime(chronos["forecast_origin_utc"], utc=True, errors="raise")
    )
    if primary is not None and primary_cutoff is not None:
        output["mkonline_primary__q50"] = primary.to_numpy(dtype=float)
        mk_cutoff = pd.DatetimeIndex(
            pd.to_datetime(primary_cutoff, utc=True, errors="raise")
        )
        origin = pd.DatetimeIndex(
            np.maximum(chronos_origin.asi8, mk_cutoff.asi8),
            tz="UTC",
        )
    else:
        origin = chronos_origin
    output["forecast_origin_utc"] = origin.astype(str)
    output["chronos2_forecast_origin_utc"] = chronos_origin.astype(str)
    output[f"{policy.candidate_model}_forecast_origin_utc"] = origin.astype(str)
    return validate_live_forecast(
        output,
        schedule=schedule,
        candidate_model=policy.candidate_model,
    )


def _build_dynamic_data(
    *,
    contract: ZoneModelContract,
    schedule: LiveSchedule,
    live: Mapping[str, Any],
    inputs_dir: Path,
    sync_manifest_path: Path,
    refresh_data: bool,
    residual_load_source: str = "saturn",
    residual_load_bundle_manifest: Path | None = None,
    saturn_control_archive: Path | None = None,
) -> tuple[dict[str, Any], Any, pd.DataFrame]:
    """Refresh and reconstruct exactly one declared zone's causal inputs."""

    from chronos2_modular.data import prepare_zone_data
    from chronos2_modular.saturn import sync_saturn_data
    from run_chronos2_hourly import _feature_inputs

    config = copy.deepcopy(load_yaml(contract.paths.base_config))
    data_config = config.setdefault("data", {})
    if not isinstance(data_config, dict):
        raise ZoneLiveExecutionError("base_config.data must be a mapping")
    data_config["runtime_as_of"] = schedule.cutoff_origin_local.isoformat()
    residual_load_provenance: dict[str, Any] | None = None
    if residual_load_source == "chronos2":
        if residual_load_bundle_manifest is None:
            raise ZoneLiveExecutionError(
                "Chronos-2 residual load requires a bundle manifest"
            )
        if saturn_control_archive is None:
            raise ZoneLiveExecutionError(
                "Chronos-2 residual load requires the paired sealed Saturn archive"
            )
        from chronos2_hourly.chronos_residual_load import (
            apply_residual_load_bundle,
        )

        residual_load_provenance = apply_residual_load_bundle(
            config,
            manifest_path=residual_load_bundle_manifest,
            expected_delivery_index=schedule.delivery_index,
            expected_cutoff_utc=schedule.cutoff_origin_local.tz_convert("UTC"),
            config_dir=contract.paths.base_config.parent,
            overlay_dir=inputs_dir / "residual_load_pit_overlay",
            saturn_control_archive=saturn_control_archive,
            expected_zone=contract.zone,
        )
    overrides = _mapping(
        live.get("naive_timezone_overrides", {}),
        name="live.naive_timezone_overrides",
    )
    if overrides:
        zones_config = _mapping(config.get("zones"), name="base_config.zones")
        zone_config = _mapping(
            zones_config.get(contract.zone),
            name=f"base_config.zones.{contract.zone}",
        )
        covariates = _mapping(
            zone_config.get("covariates"),
            name=f"base_config.zones.{contract.zone}.covariates",
        )
        for alias, timezone_hint in overrides.items():
            definition = covariates.get(str(alias))
            if not isinstance(definition, dict) or not bool(
                definition.get("enabled", True)
            ):
                raise ZoneLiveExecutionError(
                    f"invalid naive_timezone override for {alias}"
                )
            definition["naive_timezone"] = str(timezone_hint)
    set_reproducibility(int(deep_get(config, "model.seed", 42)))
    zones = build_zone_configs(config, [contract.zone], None, None)
    if len(zones) != 1 or zones[0].zone != contract.zone:
        raise ZoneLiveExecutionError(
            f"base config must resolve exactly zone {contract.zone}"
        )
    sync_config = copy.deepcopy(config)
    sync_data = sync_config.setdefault("data", {})
    if not isinstance(sync_data, dict):
        raise ZoneLiveExecutionError("base_config.data must be a mapping")
    sync_data["start"] = str(
        schedule.delivery_index[0]
        - pd.Timedelta(hours=LIVE_SYNC_LOOKBACK_HOURS)
    )
    manifest = (
        sync_saturn_data(
            zones,
            sync_config,
            contract.paths.base_config.parent,
            as_of=schedule.cutoff_origin_local,
        )
        if refresh_data
        else pd.DataFrame()
    )
    sync_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(sync_manifest_path, index=False)
    data = prepare_zone_data(
        zones[0],
        config,
        contract.paths.base_config.parent,
        False,
        inputs_dir,
    )
    if residual_load_provenance is not None:
        diagnostics = getattr(data, "diagnostics", None)
        if isinstance(diagnostics, dict):
            diagnostics["residual_load_provider"] = residual_load_provenance
        from chronos2_hourly.chronos_residual_load import (
            seal_challenger_zone_data_from_saturn_control,
        )

        control_provenance = seal_challenger_zone_data_from_saturn_control(
            data,
            saturn_archive_dir=saturn_control_archive,
            archive_inputs_dir=inputs_dir,
            expected_delivery_index=schedule.delivery_index,
            expected_zone=contract.zone,
        )
        residual_load_provenance["effective_input_control"] = control_provenance
    target, _history_covariates, future_covariates, all_features = (
        _feature_inputs(data, config)
    )
    target_context = audit_live_target_context(
        target,
        schedule.delivery_index,
    )
    diagnostics = getattr(data, "diagnostics", None)
    if isinstance(diagnostics, dict):
        target_diagnostics = diagnostics.setdefault("target", {})
        if isinstance(target_diagnostics, dict):
            target_diagnostics["live_context"] = target_context.as_dict()
    try:
        target_context.raise_if_not_exact(zone=contract.zone)
    except HourlyTargetContractError as exc:
        raise ZoneLiveExecutionError(str(exc)) from exc

    future_index = pd.DatetimeIndex(future_covariates.index).tz_convert("UTC")
    missing_delivery = schedule.delivery_index.difference(future_index)
    unexpected_future = future_index.difference(schedule.delivery_index)
    if len(missing_delivery) or len(unexpected_future):
        raise ZoneLiveExecutionError(
            f"{contract.zone}: future PIT covariates must cover only the "
            "declared J+1 schedule; "
            f"missing={len(missing_delivery)}, "
            f"unexpected={len(unexpected_future)}"
        )
    fresh = all_features.loc[schedule.delivery_index].copy()
    fresh.index = fresh.index.tz_convert("UTC")
    fresh.index.name = "delivery_start_utc"
    if not fresh.index.equals(schedule.delivery_index):
        raise ZoneLiveExecutionError(
            f"{contract.zone}: fresh PIT features do not cover exact J+1"
        )
    return config, data, fresh


def _validate_live_input_audit(
    *,
    contract: ZoneModelContract,
    data: Any,
    schedule: LiveSchedule,
) -> None:
    diagnostics = getattr(data, "diagnostics", {})
    covariates = diagnostics.get("covariates", {})
    expected = schedule.cutoff_origin_local.tz_convert("UTC")
    for alias in contract.required_covariates:
        audit = covariates.get(alias)
        if not isinstance(audit, Mapping):
            raise ZoneLiveExecutionError(f"missing live PIT audit for {alias}")
        observed = pd.Timestamp(audit.get("runtime_as_of_utc"))
        observed = (
            observed.tz_localize("UTC")
            if observed.tzinfo is None
            else observed.tz_convert("UTC")
        )
        if observed != expected or int(audit.get("cutoff_violations", -1)) != 0:
            raise ZoneLiveExecutionError(f"{alias}: live PIT cutoff violation")
        last_revision = pd.Timestamp(audit.get("last_selected_revision"))
        last_revision = (
            last_revision.tz_localize("UTC")
            if last_revision.tzinfo is None
            else last_revision.tz_convert("UTC")
        )
        if last_revision > expected:
            raise ZoneLiveExecutionError(
                f"{alias}: selected revision is later than the live cutoff"
            )


def _audit_future_pit_freshness(
    *,
    contract: ZoneModelContract,
    data: Any,
    schedule: LiveSchedule,
    max_revision_age_hours: float,
) -> dict[str, Any]:
    if max_revision_age_hours <= 0:
        raise ZoneLiveExecutionError("max_revision_age_hours must be positive")
    cutoff_utc = schedule.cutoff_origin_local.tz_convert("UTC")
    covariates = getattr(data, "diagnostics", {}).get("covariates", {})
    result: dict[str, Any] = {}
    for alias in contract.required_covariates:
        source_audit = covariates.get(alias)
        if not isinstance(source_audit, Mapping):
            raise ZoneLiveExecutionError(f"missing live PIT audit for {alias}")
        path = Path(str(source_audit.get("input", ""))).expanduser().resolve()
        required = {
            "value_time_utc",
            "snapshot_time_utc",
            "revision_time_utc",
            "value",
        }
        try:
            frame = pd.read_parquet(
                path,
                columns=sorted(required),
                filters=[
                    (
                        "value_time_utc",
                        ">=",
                        schedule.delivery_index[0].to_pydatetime(),
                    ),
                    (
                        "value_time_utc",
                        "<=",
                        schedule.delivery_index[-1].to_pydatetime(),
                    ),
                ],
            )
        except (TypeError, ValueError):
            frame = pd.read_parquet(path, columns=sorted(required))
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ZoneLiveExecutionError(f"{alias}: missing PIT columns {missing}")
        for column in ("value_time_utc", "snapshot_time_utc", "revision_time_utc"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
        frame = frame.loc[
            frame["value_time_utc"].between(
                schedule.delivery_index[0],
                schedule.delivery_index[-1],
                inclusive="both",
            )
            & frame["snapshot_time_utc"].le(cutoff_utc)
            & frame["revision_time_utc"].le(cutoff_utc)
        ].sort_values(
            ["value_time_utc", "snapshot_time_utc", "revision_time_utc"],
            kind="stable",
        )
        selected = frame.drop_duplicates("value_time_utc", keep="last")
        selected_index = pd.DatetimeIndex(
            selected["value_time_utc"],
            name="delivery_start_utc",
        )
        if not selected_index.equals(schedule.delivery_index):
            raise ZoneLiveExecutionError(
                f"{alias}: incomplete exact J+1 PIT coverage"
            )
        values = pd.to_numeric(selected["value"], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ZoneLiveExecutionError(f"{alias}: non-finite PIT value")
        revisions = pd.DatetimeIndex(selected["revision_time_utc"])
        ages = (cutoff_utc - revisions) / pd.Timedelta(hours=1)
        maximum_age = float(np.max(ages))
        if maximum_age > max_revision_age_hours:
            raise ZoneLiveExecutionError(
                f"{alias}: stale J+1 vintage ({maximum_age:.2f} h)"
            )
        result[alias] = {
            "path": str(path),
            "sha256": _sha256(path),
            "hours": int(len(selected)),
            "coverage": 1.0,
            "cutoff_utc": str(cutoff_utc),
            "oldest_selected_revision_utc": str(revisions.min()),
            "newest_selected_revision_utc": str(revisions.max()),
            "maximum_revision_age_hours": maximum_age,
            "maximum_allowed_revision_age_hours": float(max_revision_age_hours),
            "cutoff_violations": 0,
        }
    return result


def _train_and_predict_autonomous(
    *,
    contract: ZoneModelContract,
    fresh_future: pd.DataFrame,
    chronos_live: pd.DataFrame,
    threads: int,
) -> tuple[pd.DataFrame, dict[str, Any], Any]:
    from run_extended_residual_hourly import (
        _base_and_experts,
        _fit_inputs,
        _load_external,
        _load_features,
        _native_frame_from_source,
        _new_corrector,
        _read_timestamped,
    )

    frozen = contract.paths.frozen_autonomous_run
    expected_features = pd.read_csv(frozen / "feature_manifest.csv")[
        "feature"
    ].astype(str).tolist()
    if list(fresh_future.columns) != expected_features:
        raise ZoneLiveExecutionError(
            "live feature schema differs from the sealed autonomous model"
        )
    external = _load_external(
        frozen / "inputs" / "chronos_oof_extended.csv.gz",
        timezone=contract.delivery_timezone,
    )
    source_backtest = _read_timestamped(
        frozen / "backtest_hourly_oof.csv.gz",
        timestamp_column="delivery_start_utc",
    )
    if "fold_id" not in source_backtest:
        raise ZoneLiveExecutionError("sealed backtest is missing fold_id")
    existing = source_backtest.loc[source_backtest["fold_id"].notna()].copy()
    native = _native_frame_from_source(existing)
    _target, history_features, _future = _load_features(
        frozen,
        timezone=contract.delivery_timezone,
    )
    X, y, base, experts = _fit_inputs(external, native, history_features)
    model = _new_corrector(
        threads=threads,
        timezone=contract.delivery_timezone,
        primary_country=contract.zone,
    ).fit(X, y, base, experts)
    live_base, live_experts = _base_and_experts(chronos_live)
    prediction = model.predict(fresh_future, live_base, live_experts)
    prediction.index = fresh_future.index
    prediction.index.name = "delivery_start_utc"
    _numeric_quantiles(prediction, name="autonomous corrector")
    if len(model.feature_columns_) != EXPECTED_META_FEATURES:
        raise ZoneLiveExecutionError(
            f"unexpected residual schema: {len(model.feature_columns_)}"
        )
    training_end = native.index[-1]
    forecast_days = pd.Index(
        fresh_future.index.tz_convert(contract.delivery_timezone).date
    ).unique()
    if len(forecast_days) != 1:
        raise ZoneLiveExecutionError("live forecast must contain one local day")
    forecast_day = pd.Timestamp(forecast_days[0])
    training_end_day = pd.Timestamp(
        training_end.tz_convert(contract.delivery_timezone).date()
    )
    latest_allowed = forecast_day - pd.Timedelta(days=1)
    if training_end_day > latest_allowed:
        raise ZoneLiveExecutionError(
            "live residual fit uses a target unavailable at forecast origin"
        )
    return prediction, {
        "training_rows": int(len(X)),
        "training_start_utc": str(X.index[0]),
        "training_end_utc": str(training_end),
        "meta_features": int(len(model.feature_columns_)),
        "recipe": "blend_cat_hgb_w0.50",
        "target_availability": {
            "training_end_delivery_day_local": str(training_end_day.date()),
            "forecast_delivery_day_local": str(forecast_day.date()),
            "latest_allowed_training_delivery_day_local": str(latest_allowed.date()),
        },
        "model": model.diagnostics(),
    }, model


def build_candidate_default(
    contract: ZoneModelContract,
    schedule: LiveSchedule,
    policy: PredictionPolicy,
    live: Mapping[str, Any],
    options: LiveRuntimeOptions,
    staging: Path,
) -> CandidateArtifacts:
    """Production candidate path.  This function cannot access Storm."""

    from chronos2_modular.forecasting import load_model

    inputs_dir = staging / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    config, data, fresh_future = _build_dynamic_data(
        contract=contract,
        schedule=schedule,
        live=live,
        inputs_dir=inputs_dir,
        sync_manifest_path=staging / "saturn_sync_manifest.csv",
        refresh_data=options.refresh_data,
        residual_load_source=options.residual_load_source,
        residual_load_bundle_manifest=options.residual_load_bundle_manifest,
        saturn_control_archive=(
            contract.paths.output_root
            / f"{contract.zone.lower()}_day_ahead_{schedule.delivery_day.isoformat()}"
            if options.residual_load_source == "chronos2"
            else None
        ),
    )
    _validate_live_input_audit(
        contract=contract,
        data=data,
        schedule=schedule,
    )
    freshness = _audit_future_pit_freshness(
        contract=contract,
        data=data,
        schedule=schedule,
        max_revision_age_hours=float(live.get("max_revision_age_hours", 24.0)),
    )
    expected_origin = schedule.cutoff_origin_local.tz_convert("UTC")
    plan = ChronosDeliveryPlan(
        delivery_date=schedule.delivery_day,
        forecast_origin_utc=expected_origin,
        delivery_index_utc=schedule.delivery_index,
    )
    runtime = load_model(
        config,
        options.device,
        options.local_files_only,
    )
    context_length = int(deep_get(config, "model.context_length", 2048))
    model_batch_size = int(deep_get(config, "model.model_batch_size", 128))
    chronos = run_existing_live_forecast(
        plan,
        data=data,
        runtime=runtime,
        context_length=context_length,
        model_batch_size=model_batch_size,
        with_covariates=True,
        variant=f"hourly_live_dynamic_{contract.zone.lower()}",
    )
    chronos.reset_index().to_csv(staging / "chronos_live_hourly.csv", index=False)
    autonomous, fit_audit, corrector = _train_and_predict_autonomous(
        contract=contract,
        fresh_future=fresh_future,
        chronos_live=chronos,
        threads=options.threads,
    )
    primary: pd.Series | None = None
    cutoff: pd.Series | None = None
    source_paths: dict[str, Path] = {}
    if policy.mkonline_enabled:
        primary_path = inputs_dir / "mkonline_primary_live.parquet"
        if options.residual_load_source == "chronos2":
            from chronos2_hourly.chronos_residual_load import (
                copy_sealed_saturn_primary,
            )

            primary_control = copy_sealed_saturn_primary(
                contract.paths.output_root
                / f"{contract.zone.lower()}_day_ahead_{schedule.delivery_day.isoformat()}",
                destination=primary_path,
                expected_delivery_day=schedule.delivery_day,
                expected_zone=contract.zone,
            )
            command = [
                "sealed_saturn_control_copy",
                "inputs/mkonline_primary_live.parquet",
            ]
        else:
            primary_control = None
            command = materialize_primary(
                project_root=Path(__file__).resolve().parent.parent,
                contract=contract,
                schedule=schedule,
                output=primary_path,
                workers=options.workers,
            )
        primary, cutoff, primary_audit = load_primary_materialization(
            primary_path,
            contract=contract,
            schedule=schedule,
        )
        fit_audit["mkonline_primary"] = primary_audit
        if primary_control is not None:
            fit_audit["mkonline_primary"]["sealed_saturn_control"] = (
                primary_control
            )
        fit_audit["mkonline_primary"]["path"] = (
            "inputs/mkonline_primary_live.parquet"
        )
        command_for_manifest = list(command)
        if "--output" in command_for_manifest:
            output_index = command_for_manifest.index("--output") + 1
            command_for_manifest[output_index] = (
                "inputs/mkonline_primary_live.parquet"
            )
        fit_audit["mkonline_materialization_command"] = command_for_manifest
    forecast = _forecast_frame(
        contract=contract,
        schedule=schedule,
        policy=policy,
        chronos=chronos,
        autonomous=autonomous,
        primary=primary,
        primary_cutoff=cutoff,
    )
    return CandidateArtifacts(
        forecast=forecast,
        canonical_target=data.target,
        fit_audit=fit_audit,
        input_diagnostics=dict(data.diagnostics),
        pit_freshness=freshness,
        source_paths=source_paths,
        data_config=_mapping(config.get("data", {}), name="base_config.data"),
        rolling_capture_features=fresh_future.copy(),
        rolling_capture_chronos=chronos.copy(),
        attribution_kwargs={
            "data": data,
            "runtime": runtime,
            "fresh_future": fresh_future,
            "corrector": corrector,
            "official_autonomous": autonomous,
            "required_covariates": contract.required_covariates,
            "context_length": context_length,
            "model_batch_size": model_batch_size,
            "zone": contract.zone,
            "timezone": contract.delivery_timezone,
            "delivery_day": schedule.delivery_day.isoformat(),
            "official_blend": (
                pd.Series(
                    forecast["q50"].to_numpy(dtype=float),
                    index=schedule.delivery_index,
                )
                if policy.mkonline_enabled
                else None
            ),
            "primary": primary,
            "autonomous_weight": contract.weights.autonomous,
            "mkonline_weight": contract.weights.mkonline_primary,
        },
    )


def fetch_dashboard_default(
    contract: ZoneModelContract,
    schedule: LiveSchedule,
    data_config: Mapping[str, Any],
) -> tuple[pd.Series | None, Mapping[str, Any] | None]:
    """Fetch Storm day-ahead cache for reporting after candidate freeze."""

    if contract.storm_dashboard_series is None:
        return None, {
            "status": "native_dashboard_unavailable",
            "zone": contract.zone,
            "used_for_prediction": False,
        }
    metrics = json.loads(
        (contract.paths.sealed_benchmark_run / "metrics_hourly.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostics = _mapping(
        metrics.get("training_diagnostics", {}),
        name="training_diagnostics",
    )
    start_value = diagnostics.get("evaluation_start_local_date")
    if not start_value:
        raise ZoneLiveExecutionError("sealed benchmark evaluation start is missing")
    first = local_delivery_day_index(
        pd.Timestamp(start_value).date(),
        timezone=contract.delivery_timezone,
    )
    last = local_delivery_day_index(
        schedule.delivery_day - pd.Timedelta(days=1),
        timezone=contract.delivery_timezone,
    )
    expected = pd.date_range(start=first[0], end=last[-1], freq="h")
    # If there is no realized live/replay gap after the sealed benchmark, its
    # immutable dashboard Storm artifact already covers Statistics. Reuse it in
    # the reporting phase without a network call; the candidate is frozen by
    # the caller before this function runs.
    end_value = diagnostics.get("evaluation_end_local_date")
    if end_value and pd.Timestamp(end_value).date() == (
        schedule.delivery_day - pd.Timedelta(days=1)
    ):
        native_path = (
            contract.paths.sealed_benchmark_run
            / "inputs"
            / "storm_dashboard_official_statistics.parquet"
        )
        if native_path.is_file():
            frame = pd.read_parquet(native_path)
            required = {"delivery_start_utc", "storm_dashboard_official__q50"}
            if required.issubset(frame.columns):
                delivery = pd.DatetimeIndex(
                    pd.to_datetime(
                        frame["delivery_start_utc"], utc=True, errors="raise"
                    )
                )
                values = pd.Series(
                    pd.to_numeric(
                        frame["storm_dashboard_official__q50"], errors="coerce"
                    ).to_numpy(dtype=float),
                    index=delivery,
                )
                # Reconstruct the normalized civil curve. The missing
                # autumn fold remains absent; no interpolation is introduced.
                finite = values.notna()
                raw = pd.Series(
                    values.loc[finite].to_numpy(dtype=float),
                    index=values.loc[finite]
                    .index.tz_convert(contract.delivery_timezone)
                    .tz_localize(None),
                )
                return raw, {
                    "kind": "sealed_benchmark_dashboard_cache",
                    "path": str(native_path),
                    "sha256": _sha256(native_path),
                    "series": contract.storm_dashboard_series,
                    "requested_series": contract.storm_dashboard_series,
                    "primary_series": contract.storm_dashboard_primary_series,
                    "naive_timezone": contract.storm_dashboard_naive_timezone,
                    "used_for_prediction": False,
                }
    from chronos2_hourly.storm_dashboard import fetch_native_dashboard_snapshot
    from chronos2_modular.saturn import create_saturn_client

    client = create_saturn_client(
        str(data_config["saturn_url"]),
        str(data_config["saturn_author"]),
    )
    return fetch_native_dashboard_snapshot(
        client,
        zone=contract.zone,
        expected_index=expected,
    )


def update_statistics_default(**kwargs: Any) -> Mapping[str, Any]:
    from chronos2_hourly.live_history import update_live_statistics_history

    return update_live_statistics_history(**kwargs)


def plan_statistics_gaps_default(**kwargs: Any) -> Sequence[date]:
    from chronos2_hourly.live_history import missing_statistics_archive_days

    return missing_statistics_archive_days(**kwargs)


def write_report_default(*args: Any, **kwargs: Any) -> Path:
    from chronos2_hourly.reporting import write_hourly_html_report

    return write_hourly_html_report(*args, **kwargs)


DEFAULT_HOOKS = ZoneLiveHooks(
    build_candidate=build_candidate_default,
    fetch_dashboard=fetch_dashboard_default,
    update_statistics=update_statistics_default,
    write_report=write_report_default,
    plan_statistics_gaps=plan_statistics_gaps_default,
)


def _copy_benchmark(contract: ZoneModelContract, staging: Path) -> None:
    for filename in BENCHMARK_REPORT_FILES:
        source = contract.paths.sealed_benchmark_run / filename
        if source.is_file():
            shutil.copy2(source, staging / filename)
    # These two files are mandatory under the strict model contract.
    for filename in ("backtest_hourly_oof.csv.gz", "run_manifest.json"):
        source = contract.paths.sealed_benchmark_run / filename
        if source.is_file():
            shutil.copy2(source, staging / filename)


def _validate_benchmark_candidate(
    contract: ZoneModelContract,
    policy: PredictionPolicy,
) -> None:
    """Ensure Statistics can score the declared candidate before any run."""

    backtest = contract.paths.sealed_benchmark_run / "backtest_hourly_oof.csv.gz"
    metrics = contract.paths.sealed_benchmark_run / "metrics_hourly.json"
    if not backtest.is_file() or not metrics.is_file():
        raise ZoneLiveExecutionError(
            "sealed benchmark is missing backtest_hourly_oof.csv.gz or metrics_hourly.json"
        )
    columns = pd.read_csv(backtest, nrows=0).columns
    required = {
        "delivery_start_utc",
        "actual",
        *(f"{policy.candidate_model}__{q}" for q in QUANTILES),
    }
    missing = sorted(required.difference(columns))
    if missing:
        raise ZoneLiveExecutionError(
            f"sealed benchmark cannot score {policy.candidate_model}: {missing}"
        )
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    diagnostics = payload.get("training_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ZoneLiveExecutionError(
            "sealed benchmark metrics lack training_diagnostics"
        )
    for key in ("evaluation_start_local_date", "evaluation_end_local_date"):
        if not diagnostics.get(key):
            raise ZoneLiveExecutionError(
                f"sealed benchmark metrics lack {key}"
            )


def _write_artifact_checksums(
    staging: Path,
    *,
    output: Path,
    contract: ZoneModelContract,
    source_paths: Mapping[str, Path],
) -> None:
    entries: list[dict[str, Any]] = []
    sources: dict[str, Path] = {
        "live_config": contract.paths.live_config,
        "registry": contract.paths.registry,
        "base_config": contract.paths.base_config,
        "recipe_manifest": contract.paths.recipe_manifest,
        "frozen_autonomous_checksum_manifest": (
            contract.paths.frozen_autonomous_run / "artifact_checksums.json"
        ),
        "sealed_benchmark_checksum_manifest": (
            contract.paths.sealed_benchmark_run / "artifact_checksums.json"
        ),
        **source_paths,
    }
    if contract.paths.dependency_manifest is not None:
        sources["dependency_manifest"] = contract.paths.dependency_manifest
    for role, path in sources.items():
        if not path.is_file():
            raise ZoneLiveExecutionError(f"checksum source is missing: {path}")
        entries.append(
            {
                "path": str(path),
                "role": role,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    runtime_source = Path(__file__).resolve()
    entries.append(
        {
            "path": runtime_source.relative_to(
                runtime_source.parent.parent
            ).as_posix(),
            "role": "source_code",
            "size_bytes": int(runtime_source.stat().st_size),
            "sha256": _sha256(runtime_source),
        }
    )
    checksum_path = staging / "artifact_checksums.json"
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path != checksum_path:
            entries.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "role": "run_artifact",
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        checksum_path,
        {
            "algorithm": "sha256",
            "output_directory": str(output),
            "artifacts": entries,
        },
    )


def _validated_output(
    contract: ZoneModelContract,
    schedule: LiveSchedule,
    *,
    output_dir: str | Path | None,
    pit_replay: bool,
    residual_load_source: str = "saturn",
) -> tuple[Path, Path]:
    source = str(residual_load_source).strip().lower()
    if source not in RESIDUAL_LOAD_SOURCES:
        raise ZoneLiveExecutionError(
            "residual_load_source must be saturn or chronos2"
        )
    if pit_replay and source != "saturn":
        raise ZoneLiveExecutionError(
            "live Chronos-2 bundles cannot be reused for historical PIT replay"
        )
    root = (
        contract.paths.output_root / CHRONOS2_RESIDUAL_ARCHIVE_SUBDIR
        if source == "chronos2"
        else (
            contract.paths.output_root / "_replays"
            if pit_replay
            else contract.paths.output_root
        )
    )
    resolved_root = root.resolve()
    if source == "chronos2":
        try:
            resolved_root.relative_to(contract.paths.output_root.resolve())
        except ValueError as exc:
            raise ZoneLiveExecutionError(
                "Chronos-2 challenger root resolves outside zone output_root"
            ) from exc
    source_suffix = (
        CHRONOS2_RESIDUAL_ARCHIVE_SUFFIX if source == "chronos2" else ""
    )
    default_name = (
        f"{contract.zone.lower()}_day_ahead_{schedule.delivery_day.isoformat()}"
        f"{source_suffix}"
    )
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir not in (None, "")
        else root / default_name
    )
    if output.parent.resolve() != resolved_root:
        raise ZoneLiveExecutionError(
            f"output must remain directly under zone root {root.resolve()}"
        )
    if source == "chronos2" and output.name != default_name:
        raise ZoneLiveExecutionError(
            "Chronos-2 residual-load challengers require their canonical "
            "source-specific archive name"
        )
    if output.exists():
        raise FileExistsError(f"published live archive is immutable: {output}")
    frozen_inputs = (
        contract.paths.frozen_autonomous_run,
        contract.paths.sealed_benchmark_run,
    )
    if any(
        output == source
        or output in source.parents
        or source in output.parents
        for source in frozen_inputs
    ):
        raise ZoneLiveExecutionError("live output overlaps a frozen model input")
    return resolved_root, output.resolve()


def _run_zone_archive(
    contract: ZoneModelContract,
    *,
    live_settings: Mapping[str, Any],
    data_as_of: str | pd.Timestamp | None = None,
    delivery_day: str | date | pd.Timestamp | None = None,
    output_dir: str | Path | None = None,
    pit_replay: bool = False,
    options: LiveRuntimeOptions | None = None,
    hooks: ZoneLiveHooks = DEFAULT_HOOKS,
    wall_clock: pd.Timestamp | None = None,
    defer_reporting: bool = False,
    statistics_blocker: Mapping[str, Any] | None = None,
) -> Path:
    """Build and atomically publish exactly one archive without recursion."""

    if contract.zone == "FR":
        raise ZoneLiveExecutionError(
            "the generic runner is for independent non-FR bundles; use the "
            "sealed France runner for FR"
        )
    runtime = options or LiveRuntimeOptions(
        threads=int(live_settings.get("threads", -1)),
        workers=int(live_settings.get("workers", 8)),
    )
    residual_load_source, residual_bundle_manifest = _residual_runtime(runtime)
    if pit_replay and residual_load_source != "saturn":
        raise ZoneLiveExecutionError(
            "a live Chronos-2 residual bundle is invalid for PIT replay"
        )
    policy = load_prediction_policy(contract, live_settings)
    _validate_benchmark_candidate(contract, policy)
    schedule = resolve_live_schedule(
        contract,
        as_of=data_as_of,
        delivery_day=delivery_day,
    )
    now = _wall_clock_in_origin_timezone(contract, wall_clock)
    expected_live_day = (now.normalize() + pd.DateOffset(days=1)).date()
    if pit_replay:
        if schedule.delivery_day > now.date():
            raise ZoneLiveExecutionError("a PIT replay cannot target a future day")
        run_type = "pit_replay"
    else:
        if schedule.delivery_day != expected_live_day:
            raise ZoneLiveExecutionError(
                "historical delivery must use --pit-replay; current live day "
                f"is {expected_live_day}"
            )
        run_type = (
            "shadow_live_day_ahead"
            if residual_load_source == "chronos2"
            else "live_day_ahead"
        )
    if defer_reporting and not pit_replay:
        raise ZoneLiveExecutionError(
            "deferred reporting is reserved for automatic PIT replay archives"
        )
    if defer_reporting and statistics_blocker is not None:
        raise ZoneLiveExecutionError(
            "a bootstrap replay cannot also carry a live Statistics blocker"
        )
    # Parse all report-only configuration before the expensive candidate is
    # built.  Configuration errors therefore cannot occur after the freeze.
    report_settings = _resolved_report_settings(
        contract,
        schedule,
        live_settings,
    )
    root, output = _validated_output(
        contract,
        schedule,
        output_dir=output_dir,
        pit_replay=pit_replay,
        residual_load_source=residual_load_source,
    )
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=root))
    try:
        candidate = hooks.build_candidate(
            contract,
            schedule,
            policy,
            live_settings,
            runtime,
            staging,
        )
        archived_residual_bundle: dict[str, Any] | None = None
        if residual_bundle_manifest is not None:
            from chronos2_hourly.chronos_residual_load import (
                archive_live_residual_load_bundle,
            )

            archived_residual_bundle = archive_live_residual_load_bundle(
                residual_bundle_manifest,
                archive_inputs_dir=staging / "inputs",
                expected_delivery_day=schedule.delivery_day,
                expected_runtime_cutoff=(
                    schedule.cutoff_origin_local.tz_convert("UTC")
                ),
            )
            candidate.source_paths["residual_load_bundle_manifest"] = (
                residual_bundle_manifest
            )
        candidate.forecast = validate_live_forecast(
            candidate.forecast,
            schedule=schedule,
            candidate_model=policy.candidate_model,
        )
        forecast_path = staging / contract.forecast_filename
        candidate.forecast.to_csv(forecast_path, index=False)
        # Candidate is now frozen locally.  No comparator may enter any
        # prediction function beyond this point.
        frozen_candidate_sha256 = _sha256(forecast_path)
        attribution_status: dict[str, Any]
        if candidate.attribution_kwargs is None:
            attribution_status = {
                "status": "not_available",
                "reason": "candidate builder did not expose attribution inputs",
                "used_for_prediction": False,
            }
        else:
            try:
                attribution_audit = write_variable_attribution(
                    output_dir=staging,
                    forecast_path=forecast_path,
                    **candidate.attribution_kwargs,
                )
            except Exception as exc:
                remove_variable_attribution_artifacts(staging)
                attribution_status = {
                    "status": "failed_optional",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "used_for_prediction": False,
                    "forecast_modified": False,
                }
                LOGGER.warning(
                    "%s: variable attribution failed after candidate freeze: %s",
                    contract.zone,
                    exc,
                )
            else:
                attribution_status = {
                    "status": "complete",
                    "method": attribution_audit["method"],
                    "scenario_count": attribution_audit["scenario_count"],
                    "variants": attribution_audit["variants"],
                    "hourly_artifact": "variable_attribution_hourly.csv.gz",
                    "audit_artifact": "variable_attribution_audit.json",
                    "used_for_prediction": False,
                    "forecast_modified": False,
                }
        if _sha256(forecast_path) != frozen_candidate_sha256:
            raise ZoneLiveExecutionError(
                "variable attribution modified the frozen candidate"
            )

        if residual_load_source == "saturn":
            _copy_benchmark(contract, staging)
        source_manifest = json.loads(
            (contract.paths.sealed_benchmark_run / "run_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        target_diagnostics = candidate.input_diagnostics.get("target", {})
        target_source_value = (
            target_diagnostics.get("cache") or target_diagnostics.get("input")
            if isinstance(target_diagnostics, Mapping)
            else None
        )
        target_source_file = (
            Path(str(target_source_value)).expanduser().resolve()
            if target_source_value not in (None, "")
            else None
        )
        target_source_path = (
            str(target_source_file)
            if target_source_file is not None and target_source_file.is_file()
            else None
        )
        target_source_sha256 = (
            _sha256(target_source_file)
            if target_source_file is not None and target_source_file.is_file()
            else None
        )
        manifest = dict(source_manifest)
        manifest.update(
            {
                "script_version": SCRIPT_VERSION,
                "run_type": run_type,
                "forecast_status": (
                    "pit_reconstruction"
                    if pit_replay
                    else (
                        "shadow_challenger"
                        if residual_load_source == "chronos2"
                        else "issued_live"
                    )
                ),
                "zone": contract.zone,
                "timezone": contract.delivery_timezone,
                "forecast_origin_timezone": contract.forecast_origin_timezone,
                "forecast_origin_local_time": contract.forecast_origin_local_time,
                "target_series": contract.target_series,
                "target_source_path": target_source_path,
                "target_source_sha256": target_source_sha256,
                "primary_series": (
                    contract.primary_series if policy.mkonline_enabled else None
                ),
                "prediction_mode": policy.mode,
                "candidate_model": policy.candidate_model,
                "delivery_day_local": schedule.delivery_day.isoformat(),
                "run_started_as_of_local": str(schedule.as_of_origin_local),
                "run_started_as_of_utc": str(
                    schedule.as_of_origin_local.tz_convert("UTC")
                ),
                "execution_started_at_local": str(now),
                "execution_started_at_utc": str(now.tz_convert("UTC")),
                "forecast_cutoff_local": str(schedule.cutoff_origin_local),
                "forecast_cutoff_utc": str(
                    schedule.cutoff_origin_local.tz_convert("UTC")
                ),
                "n_forecast_hours": int(len(schedule.delivery_index)),
                "forecast_start_utc": str(schedule.delivery_index[0]),
                "forecast_end_utc": str(schedule.delivery_index[-1]),
                "prediction_inputs": [
                    "autonomous_extended_residual",
                    *([contract.primary_series] if policy.mkonline_enabled else []),
                ],
                "storm_loaded_for_prediction": False,
                "storm_used_as_feature": False,
                "automatic_statistics_bootstrap": bool(defer_reporting),
                "statistics_reporting_deferred": bool(defer_reporting),
                "annual_benchmark_recomputed": False,
                "frozen_candidate_sha256_before_comparator": frozen_candidate_sha256,
                "frozen_autonomous_manifest_sha256": (
                    contract.checksum_hashes.frozen_autonomous_checksum_manifest_sha256
                ),
                "sealed_benchmark_manifest_sha256": (
                    contract.checksum_hashes.sealed_benchmark_checksum_manifest_sha256
                ),
                "autonomous_weight": contract.weights.autonomous,
                "mkonline_weight": contract.weights.mkonline_primary,
                "live_fit": candidate.fit_audit,
                "input_diagnostics": candidate.input_diagnostics,
                "fundamental_pit_freshness": candidate.pit_freshness,
                "variable_attribution": attribution_status,
                "storm_dashboard_series": contract.storm_dashboard_series,
                "storm_strict_08_series": contract.storm_strict_08_series,
                "sha256_manifest": "artifact_checksums.json",
            }
        )
        if residual_load_source == "chronos2":
            if archived_residual_bundle is None:  # pragma: no cover - guarded above
                raise ZoneLiveExecutionError(
                    "the archived Chronos-2 residual bundle is missing"
                )
            manifest.update(
                {
                    "residual_load_source": "chronos2",
                    "residual_load_bundle_manifest_path": (
                        "inputs/"
                        + str(archived_residual_bundle["archived_manifest_path"])
                    ),
                    "residual_load_bundle_origin_manifest_path": str(
                        archived_residual_bundle["origin_manifest_path"]
                    ),
                    "residual_load_bundle_manifest_sha256": str(
                        archived_residual_bundle["archived_manifest_sha256"]
                    ),
                    "production_eligible": False,
                    "comparison_design": "prospective_paired_same_downstream",
                }
            )
        _write_json(staging / "run_manifest.json", manifest)
        summary_payload: dict[str, Any] = {
            "status": "complete",
            "run_type": run_type,
            "zone": contract.zone,
            "delivery_day_local": schedule.delivery_day,
            "hours": len(schedule.delivery_index),
            "prediction_mode": policy.mode,
            "candidate_model": policy.candidate_model,
            "weights": {
                "autonomous": contract.weights.autonomous,
                "mkonline_primary": contract.weights.mkonline_primary,
            },
            "storm_loaded_for_prediction": False,
            "variable_attribution": attribution_status,
            "forecast_path": contract.forecast_filename,
        }
        if residual_load_source == "chronos2":
            summary_payload.update(
                {
                    "forecast_status": "shadow_challenger",
                    "residual_load_source": "chronos2",
                    "production_eligible": False,
                }
            )
        _write_json(
            staging / "live_run_summary.json",
            summary_payload,
        )
        shutil.copy2(
            contract.paths.recipe_manifest,
            staging / "mkonline_blend_recipe.json",
        )
        if contract.paths.dependency_manifest is not None:
            shutil.copy2(
                contract.paths.dependency_manifest,
                staging / "mkonline_primary_dependency.json",
            )

        if residual_load_source == "chronos2":
            prospective_statistics = {
                "status": "prospective_only",
                "historical_performance_eligible": False,
                "residual_load_source": "chronos2",
                "comparison_reason": (
                    "Aucun historique causal Chronos-2 n'est substitue aux "
                    "archives Saturn de production. Le scoring se fait "
                    "uniquement sur les paires prospectives publiees."
                ),
            }
            manifest.update(
                {
                    "statistics_history": prospective_statistics,
                    "candidate_forecast_sha256_at_freeze": (
                        frozen_candidate_sha256
                    ),
                    "storm_dashboard_loaded_after_candidate_frozen_for_statistics": (
                        False
                    ),
                    "reporting_status": "forecast_only",
                    "reporting_errors": [],
                }
            )
            summary_payload.update(
                {
                    "statistics_history": prospective_statistics,
                    "storm_dashboard_loaded_for_statistics_after_candidate_frozen": (
                        False
                    ),
                    "reporting_status": "forecast_only",
                }
            )
            write_forecast_only_shadow_report(
                forecast_path,
                output_path=staging / str(report_settings["filename"]),
                zone=contract.zone,
                delivery_day=schedule.delivery_day.isoformat(),
                candidate_model=policy.candidate_model,
            )
            if _sha256(forecast_path) != frozen_candidate_sha256:
                raise ZoneLiveExecutionError(
                    "forecast-only reporting modified the frozen candidate"
                )
            _write_json(staging / "run_manifest.json", manifest)
            _write_json(
                staging / "live_run_summary.json",
                summary_payload,
            )
            _write_artifact_checksums(
                staging,
                output=output,
                contract=contract,
                source_paths=candidate.source_paths,
            )
            _publish_staging_atomically(staging, output)
            return output

        if defer_reporting:
            deferred_statistics = {
                "status": "deferred_to_following_live_run",
                "reason": "automatic_causal_archive_bootstrap",
                "delivery_day_local": schedule.delivery_day.isoformat(),
                "storm_requested": False,
                "statistics_requested": False,
                "report_requested": False,
            }
            manifest["statistics_history"] = deferred_statistics
            manifest[
                "storm_dashboard_loaded_after_candidate_frozen_for_statistics"
            ] = False
            summary_payload["status"] = "forecast_complete_statistics_deferred"
            summary_payload["statistics_history"] = deferred_statistics
            _write_json(staging / "run_manifest.json", manifest)
            _write_json(staging / "live_run_summary.json", summary_payload)
            if _sha256(forecast_path) != frozen_candidate_sha256:
                raise ZoneLiveExecutionError(
                    "bootstrap publication modified the frozen candidate"
                )
            _write_artifact_checksums(
                staging,
                output=output,
                contract=contract,
                source_paths=candidate.source_paths,
            )
            _publish_staging_atomically(staging, output)
            return output

        reporting_errors: list[dict[str, str]] = []
        try:
            dashboard, dashboard_source = hooks.fetch_dashboard(
                contract,
                schedule,
                candidate.data_config,
            )
        except Exception as exc:
            reporting_errors.append(_reporting_error("storm_dashboard", exc))
            dashboard = None
            dashboard_source = {
                "status": "reporting_error",
                "used_for_prediction": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            LOGGER.warning(
                "%s: Storm reporting failed after candidate freeze: %s",
                contract.zone,
                exc,
            )
        if _sha256(forecast_path) != frozen_candidate_sha256:
            raise ZoneLiveExecutionError(
                "report-only comparator modified the frozen candidate"
            )
        storm_pit_setting = live_settings.get("storm_pit_path")
        storm_pit_path = (
            Path(str(storm_pit_setting)).expanduser().resolve()
            if storm_pit_setting not in (None, "")
            else staging / "inputs" / "storm_strict_08_unavailable.parquet"
        )
        statistics_live_root = (
            contract.paths.output_root
            if residual_load_source == "saturn"
            else contract.paths.output_root / "_residual_load_chronos2_statistics"
        )
        statistics_kwargs: dict[str, Any] = {
            "staging_run_dir": staging,
            "sealed_benchmark_run": contract.paths.sealed_benchmark_run,
            "live_output_root": statistics_live_root,
            "replay_output_root": statistics_live_root / "_replays",
            "current_delivery_day": schedule.delivery_day,
            "canonical_target": candidate.canonical_target,
            "storm_pit_path": storm_pit_path,
            "storm_dashboard_native": dashboard,
            "storm_dashboard_source": dashboard_source,
            "zone": contract.zone,
            "timezone": contract.delivery_timezone,
            "forecast_name": contract.forecast_filename,
            "candidate_model": policy.candidate_model,
            "storm_strict_08_series": contract.storm_strict_08_series,
            "target_series": contract.target_series,
            "prediction_mode": policy.mode,
        }
        if statistics_blocker is None:
            try:
                statistics = hooks.update_statistics(**statistics_kwargs)
            except Exception as exc:
                reporting_errors.append(_reporting_error("statistics", exc))
                _remove_optional_statistics_artifacts(staging)
                statistics = {
                    "status": "blocked_reporting_error",
                    "reason": "Statistics failed after candidate freeze",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "candidate_forecast_publication": "continued",
                    "statistics_scope": "sealed_benchmark_only",
                    "storm_used_for_prediction": False,
                    "diagnostic_path": "statistics_update_blocked.json",
                }
                _write_json(
                    staging / "statistics_update_blocked.json",
                    statistics,
                )
                summary_payload["status"] = (
                    "forecast_complete_statistics_blocked"
                )
                summary_payload["statistics_history"] = statistics
                _write_json(
                    staging / "live_run_summary.json",
                    summary_payload,
                )
                LOGGER.warning(
                    "%s: Statistics reporting failed after candidate freeze: %s",
                    contract.zone,
                    exc,
                )
            else:
                statistics = dict(statistics)
        else:
            try:
                statistics = hooks.update_statistics(
                    **statistics_kwargs,
                    allow_partial_prefix=True,
                    statistics_blocker=statistics_blocker,
                )
            except Exception as exc:
                # A Statistics-only failure must not discard a valid issued
                # candidate.  Remove any half-written optional history so the
                # report cleanly falls back to the sealed benchmark.
                _remove_optional_statistics_artifacts(staging)
                statistics = {
                    **dict(statistics_blocker),
                    "status": "blocked_missing_causal_archives",
                    "partial_statistics_error_type": type(exc).__name__,
                    "partial_statistics_error": str(exc),
                    "statistics_scope": "sealed_benchmark_only",
                }
                summary_payload["status"] = (
                    "forecast_complete_statistics_blocked"
                )
            else:
                statistics = dict(statistics)
                statistics.setdefault("status", "partial_contiguous_prefix")
                summary_payload["status"] = (
                    "forecast_complete_statistics_partial"
                )
            statistics["diagnostic_path"] = "statistics_update_blocked.json"
            _write_json(
                staging / "statistics_update_blocked.json",
                statistics,
            )
            summary_payload["statistics_history"] = statistics
            _write_json(staging / "live_run_summary.json", summary_payload)
        manifest["statistics_history"] = dict(statistics)
        manifest["storm_dashboard_loaded_after_candidate_frozen_for_statistics"] = (
            dashboard is not None
        )
        _write_json(staging / "run_manifest.json", manifest)
        if _sha256(forecast_path) != frozen_candidate_sha256:
            raise ZoneLiveExecutionError(
                "Statistics/reporting modified the frozen candidate"
            )
        report_path = staging / str(report_settings["filename"])
        report_title = str(report_settings["title"])
        try:
            hooks.write_report(
                staging,
                output_path=report_path,
                title=report_title,
                native_model=policy.candidate_model,
                baseline_model=(
                    "residual_corrected"
                    if policy.mkonline_enabled
                    else "ensemble"
                ),
                zone=contract.zone,
                timezone=contract.delivery_timezone,
                extreme_threshold=float(report_settings["extreme_threshold"]),
                history_hours=int(report_settings["history_hours"]),
            )
            if not report_path.is_file():
                raise ZoneLiveExecutionError(
                    "HTML report writer did not create output"
                )
        except Exception as exc:
            reporting_errors.append(_reporting_error("html_report", exc))
            report_path.unlink(missing_ok=True)
            _write_degraded_html_report(
                report_path,
                title=report_title,
                zone=contract.zone,
                delivery_day=schedule.delivery_day,
                forecast_name=contract.forecast_filename,
                frozen_candidate_sha256=frozen_candidate_sha256,
                reporting_errors=reporting_errors,
            )
            LOGGER.warning(
                "%s: HTML reporting failed after candidate freeze; wrote "
                "degraded report: %s",
                contract.zone,
                exc,
            )
        if _sha256(forecast_path) != frozen_candidate_sha256:
            raise ZoneLiveExecutionError("report generation modified the candidate")
        if reporting_errors:
            _write_json(
                staging / "reporting_errors.json",
                {
                    "status": "degraded",
                    "candidate_forecast_sha256": frozen_candidate_sha256,
                    "forecast_modified": False,
                    "errors": reporting_errors,
                },
            )
            manifest["reporting_status"] = "degraded"
            manifest["reporting_errors"] = reporting_errors
            if summary_payload.get("status") == "complete":
                summary_payload["status"] = (
                    "forecast_complete_reporting_degraded"
                )
            summary_payload["reporting_errors"] = reporting_errors
        else:
            manifest["reporting_status"] = "complete"
            manifest["reporting_errors"] = []
        _write_json(staging / "run_manifest.json", manifest)
        _write_json(staging / "live_run_summary.json", summary_payload)
        _write_artifact_checksums(
            staging,
            output=output,
            contract=contract,
            source_paths=candidate.source_paths,
        )
        _publish_staging_atomically(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if not pit_replay and runtime.rolling365_capture_root is not None:
        # The archive is already atomically published.  Optional rolling-365
        # evidence capture is therefore unable to invalidate or rewrite the
        # official candidate.  Only raw Chronos and the supported autonomous
        # PIT/calendar subset cross this seam.
        try:
            if (
                candidate.rolling_capture_features is None
                or candidate.rolling_capture_chronos is None
            ):
                raise ZoneLiveExecutionError(
                    "candidate builder did not expose raw rolling-capture inputs"
                )
            from chronos2_hourly.rolling_capture import (
                capture_supported_issued_live_block_isolated,
            )

            capture_result = capture_supported_issued_live_block_isolated(
                capture_root=runtime.rolling365_capture_root,
                zone=contract.zone,
                delivery_day=schedule.delivery_day,
                delivery_timezone=contract.delivery_timezone,
                fresh_features=candidate.rolling_capture_features,
                chronos_live=candidate.rolling_capture_chronos,
                required_pit_aliases=contract.required_covariates,
                pit_freshness=candidate.pit_freshness,
                expected_config_sha256=_sha256(contract.paths.live_config),
                expected_base_bundle_sha256=(
                    contract.checksum_hashes.frozen_autonomous_checksum_manifest_sha256
                ),
                target_series=contract.target_series,
                target_source_path=target_source_path,
                issued_live_archive=output,
                issued_live_forecast_filename=contract.forecast_filename,
                canonical_target=candidate.canonical_target,
            )
            LOGGER.info(
                "%s: rolling-365 capture status=%s",
                contract.zone,
                capture_result.status,
            )
        except Exception as exc:
            LOGGER.warning(
                "%s: rolling-365 capture ignored after official publication: %s",
                contract.zone,
                exc,
            )
    return output


def _normalise_statistics_gap_days(
    values: Sequence[date],
    *,
    current_delivery_day: date,
    now: pd.Timestamp,
) -> list[date]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ZoneLiveExecutionError(
            "Statistics gap planner must return a sequence of civil days"
        )
    result: list[date] = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ZoneLiveExecutionError(
                f"invalid Statistics gap day: {value!r}"
            ) from exc
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(now.tz).tz_localize(None)
        if timestamp != timestamp.normalize():
            raise ZoneLiveExecutionError(
                f"Statistics gap must be a civil day: {value!r}"
            )
        day = timestamp.date()
        if day >= current_delivery_day:
            raise ZoneLiveExecutionError(
                f"Statistics planner returned current/future day {day}"
            )
        if day > now.date():
            raise ZoneLiveExecutionError(
                f"automatic PIT replay cannot target future day {day}"
            )
        result.append(day)
    if len(result) != len(set(result)):
        raise ZoneLiveExecutionError("Statistics gap planner returned duplicates")
    return sorted(result)


def run_zone_live(
    contract: ZoneModelContract,
    *,
    live_settings: Mapping[str, Any],
    data_as_of: str | pd.Timestamp | None = None,
    delivery_day: str | date | pd.Timestamp | None = None,
    output_dir: str | Path | None = None,
    pit_replay: bool = False,
    options: LiveRuntimeOptions | None = None,
    hooks: ZoneLiveHooks = DEFAULT_HOOKS,
    wall_clock: pd.Timestamp | None = None,
) -> Path:
    """Publish one live forecast, filling causal Statistics gaps first.

    Normal live runs plan all missing immutable archives before the current
    candidate is built.  Missing days are produced in chronological order by
    the non-recursive single-archive helper above, at their exact civil D-1
    cutoff.  Bootstrap archives contain the frozen forecast and provenance
    only: Storm, Statistics and HTML are evaluated once, on the current run.

    If a historical PIT replay cannot be reconstructed, the current forecast
    is still published with an explicit Statistics blocker and a benchmark-
    only report.  Invalid archive identity/planning remains fail-closed before
    any candidate is built.
    """

    if pit_replay:
        return _run_zone_archive(
            contract,
            live_settings=live_settings,
            data_as_of=data_as_of,
            delivery_day=delivery_day,
            output_dir=output_dir,
            pit_replay=True,
            options=options,
            hooks=hooks,
            wall_clock=wall_clock,
        )

    runtime = options or LiveRuntimeOptions(
        threads=int(live_settings.get("threads", -1)),
        workers=int(live_settings.get("workers", 8)),
    )
    residual_load_source, _residual_bundle_manifest = _residual_runtime(runtime)

    if contract.zone == "FR":
        raise ZoneLiveExecutionError(
            "the generic runner is for independent non-FR bundles; use the "
            "sealed France runner for FR"
        )
    policy = load_prediction_policy(contract, live_settings)
    _validate_benchmark_candidate(contract, policy)
    schedule = resolve_live_schedule(
        contract,
        as_of=data_as_of,
        delivery_day=delivery_day,
    )
    now = _wall_clock_in_origin_timezone(contract, wall_clock)
    expected_live_day = (now.normalize() + pd.DateOffset(days=1)).date()
    if schedule.delivery_day != expected_live_day:
        raise ZoneLiveExecutionError(
            "historical delivery must use --pit-replay; current live day "
            f"is {expected_live_day}"
        )
    # Reject an existing/escaping current output before any automatic replay
    # can create durable side effects.
    _validated_output(
        contract,
        schedule,
        output_dir=output_dir,
        pit_replay=False,
        residual_load_source=residual_load_source,
    )

    missing_days: list[date] = []
    if residual_load_source == "chronos2":
        # The challenger starts a new prospective evidence chain. Reusing or
        # rebuilding production/Saturn archives would make its score look
        # historical when no causal Chronos-2 residual inputs existed then.
        missing_days = []
    elif hooks.plan_statistics_gaps is not None:
        planned = hooks.plan_statistics_gaps(
            sealed_benchmark_run=contract.paths.sealed_benchmark_run,
            live_output_root=contract.paths.output_root,
            replay_output_root=contract.paths.output_root / "_replays",
            current_delivery_day=schedule.delivery_day,
            timezone=contract.delivery_timezone,
            forecast_name=contract.forecast_filename,
            candidate_model=policy.candidate_model,
            zone=contract.zone,
            target_series=contract.target_series,
            prediction_mode=policy.mode,
        )
        missing_days = _normalise_statistics_gap_days(
            planned,
            current_delivery_day=schedule.delivery_day,
            now=now,
        )
    raw_limit = live_settings.get(
        "max_automatic_replay_days",
        MAX_AUTOMATIC_REPLAY_DAYS,
    )
    if isinstance(raw_limit, bool):
        raise ZoneLiveExecutionError(
            "live.max_automatic_replay_days must be a positive integer"
        )
    try:
        replay_limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ZoneLiveExecutionError(
            "live.max_automatic_replay_days must be a positive integer"
        ) from exc
    if replay_limit <= 0:
        raise ZoneLiveExecutionError(
            "live.max_automatic_replay_days must be a positive integer"
        )
    if replay_limit > MAX_AUTOMATIC_REPLAY_DAYS:
        raise ZoneLiveExecutionError(
            "live.max_automatic_replay_days cannot exceed the hard safety "
            f"cap {MAX_AUTOMATIC_REPLAY_DAYS}"
        )
    completed_replays: list[date] = []
    statistics_blocker: dict[str, Any] | None = (
        {
            "status": "blocked_prospective_challenger",
            "reason": (
                "Chronos-2 residual-load evidence starts with this issued "
                "shadow run; production Statistics are not reused"
            ),
            "missing_realized_days": [],
            "completed_automatic_replay_days": [],
            "candidate_forecast_publication": "continued_as_shadow",
            "statistics_scope": "sealed_benchmark_only",
            "residual_load_source": "chronos2",
            "storm_used_for_prediction": False,
        }
        if residual_load_source == "chronos2"
        else None
    )
    automatic_replay_days = list(missing_days)
    expected_recent_suffix = (
        [
            value.date()
            for value in pd.date_range(
                schedule.delivery_day - pd.Timedelta(days=len(missing_days)),
                schedule.delivery_day - pd.Timedelta(days=1),
                freq="D",
            )
        ]
        if missing_days
        else []
    )
    if missing_days and missing_days != expected_recent_suffix:
        automatic_replay_days = []
        statistics_blocker = {
            "status": "blocked_missing_causal_archives",
            "reason": "archive gaps are not a recent contiguous suffix",
            "missing_realized_days": [day.isoformat() for day in missing_days],
            "completed_automatic_replay_days": [],
            "candidate_forecast_publication": "continued",
            "statistics_scope": "contiguous_prefix",
            "storm_used_for_prediction": False,
        }
    elif len(missing_days) > replay_limit:
        automatic_replay_days = []
        statistics_blocker = {
            "status": "blocked_missing_causal_archives",
            "reason": "automatic replay safety limit exceeded",
            "missing_realized_days": [day.isoformat() for day in missing_days],
            "completed_automatic_replay_days": [],
            "requested_automatic_replay_days": len(missing_days),
            "max_automatic_replay_days": replay_limit,
            "candidate_forecast_publication": "continued",
            "statistics_scope": "contiguous_prefix",
            "storm_used_for_prediction": False,
        }
    if automatic_replay_days:
        LOGGER.warning(
            "%s: automatic causal Statistics catch-up for %s",
            contract.zone,
            ", ".join(day.isoformat() for day in automatic_replay_days),
        )
    elif statistics_blocker is not None:
        LOGGER.warning(
            "%s: Statistics will use a partial causal prefix: %s",
            contract.zone,
            statistics_blocker["reason"],
        )

    for position, replay_day in enumerate(automatic_replay_days):
        replay_cutoff = _forecast_cutoff_for_day(contract, replay_day)
        if replay_cutoff > now:
            raise ZoneLiveExecutionError(
                f"automatic PIT replay cutoff is still in the future: "
                f"{replay_cutoff}"
            )
        try:
            _run_zone_archive(
                contract,
                live_settings=live_settings,
                data_as_of=replay_cutoff,
                delivery_day=replay_day,
                output_dir=None,
                pit_replay=True,
                options=runtime,
                hooks=hooks,
                wall_clock=now,
                defer_reporting=True,
            )
        except Exception as exc:
            remaining = automatic_replay_days[position:]
            statistics_blocker = {
                "status": "blocked_missing_causal_archives",
                "reason": "automatic causal PIT replay failed",
                "missing_realized_days": [day.isoformat() for day in remaining],
                "completed_automatic_replay_days": [
                    day.isoformat() for day in completed_replays
                ],
                "failed_replay_day": replay_day.isoformat(),
                "failed_replay_cutoff_local": str(replay_cutoff),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "candidate_forecast_publication": "continued",
                "statistics_scope": "contiguous_prefix",
                "storm_used_for_prediction": False,
            }
            LOGGER.warning(
                "%s: causal replay %s failed; current forecast publication "
                "continues with partial Statistics: %s",
                contract.zone,
                replay_day,
                exc,
            )
            break
        completed_replays.append(replay_day)
        LOGGER.info(
            "%s: published automatic causal replay %s",
            contract.zone,
            replay_day,
        )

    return _run_zone_archive(
        contract,
        live_settings=live_settings,
        data_as_of=data_as_of,
        delivery_day=delivery_day,
        output_dir=output_dir,
        pit_replay=False,
        options=runtime,
        hooks=hooks,
        wall_clock=now,
        statistics_blocker=statistics_blocker,
    )


__all__: Sequence[str] = (
    "BENCHMARK_REPORT_FILES",
    "CandidateArtifacts",
    "DEFAULT_HOOKS",
    "LIVE_SYNC_LOOKBACK_HOURS",
    "MAX_AUTOMATIC_REPLAY_DAYS",
    "LiveRuntimeOptions",
    "LiveSchedule",
    "PREDICTION_MODES",
    "PredictionPolicy",
    "QUANTILES",
    "SCRIPT_VERSION",
    "ZoneLiveExecutionError",
    "ZoneLiveHooks",
    "blend_quantiles",
    "build_candidate_default",
    "fetch_dashboard_default",
    "load_prediction_policy",
    "load_primary_materialization",
    "materialize_primary",
    "plan_statistics_gaps_default",
    "resolve_live_schedule",
    "run_zone_live",
    "validate_live_forecast",
)
