#!/usr/bin/env python
"""Evaluate research-only recovery paths for early residual-load features.

This utility never contacts Saturn.  It consumes already materialised, dated
as-of probes and the immutable local reference sidecars, then writes a
machine-readable manifest.  In particular it does not promote reconstructed
history to prospective/operational PIT evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = (
    ROOT
    / "runs"
    / "experiments"
    / "chronos2_exogenous_oof_history_recovery_v1"
)


class ResidualRecoveryProbeError(ValueError):
    """Raised when an input violates the bounded research probe contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ResidualRecoveryProbeError(f"Artefact absent: {resolved}")
    audit = resolved.with_name(resolved.name + ".audit.json")
    record: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
        "audit_path": str(audit) if audit.is_file() else None,
        "audit_sha256": _sha256(audit) if audit.is_file() else None,
    }
    if audit.is_file():
        payload = json.loads(audit.read_text(encoding="utf-8"))
        record["source_series"] = payload.get("series")
        record["requested_start_day"] = payload.get("requested_start_day")
        record["requested_end_day"] = payload.get("requested_end_day")
        record["cutoff_time"] = payload.get("cutoff_time")
        record["cutoff_timezone"] = payload.get("cutoff_timezone")
        record["snapshot_time_semantics"] = payload.get(
            "snapshot_time_semantics"
        )
        record["provider_revision_timestamp_available"] = payload.get(
            "provider_revision_timestamp_available"
        )
    return record


def _load_long(path: Path, *, value_column: str = "value") -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        value_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ResidualRecoveryProbeError(f"{path}: colonnes absentes {missing}.")
    selected = frame.loc[:, list(required)].copy()
    for column in ("value_time_utc", "snapshot_time_utc", "revision_time_utc"):
        selected[column] = pd.to_datetime(selected[column], utc=True, errors="raise")
    selected[value_column] = pd.to_numeric(selected[value_column], errors="coerce")
    selected = selected.sort_values("value_time_utc")
    if selected["value_time_utc"].duplicated().any():
        raise ResidualRecoveryProbeError(f"{path}: value_time_utc duplique.")
    if not np.isfinite(selected[value_column].to_numpy(dtype=float)).all():
        raise ResidualRecoveryProbeError(f"{path}: valeurs non finies.")
    return selected.set_index("value_time_utc")


def _slice_days(
    frame: pd.DataFrame,
    *,
    start_day: str,
    end_day: str,
    timezone: str,
) -> pd.DataFrame:
    local_days = frame.index.tz_convert(timezone).normalize().tz_localize(None)
    mask = (local_days >= pd.Timestamp(start_day)) & (
        local_days <= pd.Timestamp(end_day)
    )
    return frame.loc[mask].copy()


def _expected_cutoffs(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    local_days = index.tz_convert(timezone).normalize().tz_localize(None)
    return pd.DatetimeIndex(
        [
            (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8))
            .tz_localize(timezone)
            .tz_convert("UTC")
            for day in local_days
        ]
    )


def _cutoff_audit(frame: pd.DataFrame, *, timezone: str) -> dict[str, Any]:
    expected = _expected_cutoffs(pd.DatetimeIndex(frame.index), timezone)
    snapshots = pd.DatetimeIndex(frame["snapshot_time_utc"])
    revisions = pd.DatetimeIndex(frame["revision_time_utc"])
    return {
        "rows": int(len(frame)),
        "snapshot_equals_d_minus_1_0800": bool((snapshots == expected).all()),
        "revision_equals_d_minus_1_0800": bool((revisions == expected).all()),
        "information_after_cutoff_count": int(
            ((snapshots > expected) | (revisions > expected)).sum()
        ),
        "first_delivery_utc": frame.index.min().isoformat(),
        "last_delivery_utc": frame.index.max().isoformat(),
        "first_cutoff_utc": snapshots.min().isoformat(),
        "last_cutoff_utc": snapshots.max().isoformat(),
    }


def _metrics(actual: Iterable[float], forecast: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(list(actual), dtype=float)
    x = np.asarray(list(forecast), dtype=float)
    finite = np.isfinite(y) & np.isfinite(x)
    y = y[finite]
    x = x[finite]
    if not len(y):
        return {"rows": 0, "mae": None, "bias": None, "rmse": None, "corr": None}
    error = x - y
    corr = None
    if len(y) >= 2 and float(np.std(y)) > 0 and float(np.std(x)) > 0:
        corr = float(np.corrcoef(y, x)[0, 1])
    return {
        "rows": int(len(y)),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "corr": corr,
    }


def _zero_intercept_scale(actual: pd.Series, proxy: pd.Series) -> float:
    y = actual.to_numpy(dtype=float)
    x = proxy.to_numpy(dtype=float)
    active = np.isfinite(y) & np.isfinite(x) & ((np.abs(y) > 1e-6) | (np.abs(x) > 1e-6))
    denominator = float(np.dot(x[active], x[active]))
    if active.sum() < 2 or not math.isfinite(denominator) or denominator <= 0:
        raise ResidualRecoveryProbeError("Calibration solaire impossible.")
    return max(0.0, float(np.dot(x[active], y[active]) / denominator))


def _affine_parameters(actual: pd.Series, proxy: pd.Series) -> tuple[float, float]:
    y = actual.to_numpy(dtype=float)
    x = proxy.to_numpy(dtype=float)
    finite = np.isfinite(y) & np.isfinite(x)
    if finite.sum() < 3 or float(np.std(x[finite])) <= 0.0:
        raise ResidualRecoveryProbeError("Calibration affine impossible.")
    intercept, slope = np.linalg.lstsq(
        np.column_stack((np.ones(int(finite.sum())), x[finite])),
        y[finite],
        rcond=None,
    )[0]
    return float(intercept), float(slope)


def _group_metrics(
    frame: pd.DataFrame,
    *,
    group: pd.Series,
    actual: str,
    forecasts: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, positions in group.groupby(group).groups.items():
        subset = frame.loc[positions]
        record: dict[str, Any] = {"group": str(label)}
        for forecast in forecasts:
            record[forecast] = _metrics(subset[actual], subset[forecast])
        records.append(record)
    return records


def _solar_analysis(
    canonical_path: Path,
    proxy_path: Path,
    *,
    start_day: str,
    end_day: str,
    timezone: str,
) -> tuple[dict[str, Any], float, pd.DataFrame]:
    canonical = _slice_days(
        _load_long(canonical_path),
        start_day=start_day,
        end_day=end_day,
        timezone=timezone,
    )
    proxy = _slice_days(
        _load_long(proxy_path),
        start_day=start_day,
        end_day=end_day,
        timezone=timezone,
    )
    joined = pd.DataFrame(
        {
            "canonical": canonical["value"],
            "proxy_raw": proxy["value"],
        }
    ).dropna()
    expected = pd.date_range(
        pd.Timestamp(start_day).tz_localize(timezone).tz_convert("UTC"),
        (pd.Timestamp(end_day) + pd.Timedelta(days=1))
        .tz_localize(timezone)
        .tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    if not joined.index.equals(expected):
        missing = expected.difference(joined.index)
        raise ResidualRecoveryProbeError(
            f"Chevauchement solaire incomplet: {len(joined)}/{len(expected)}; "
            f"premieres heures absentes={list(missing[:5])}."
        )
    local_days = joined.index.tz_convert(timezone).normalize().tz_localize(None)
    unique_days = pd.DatetimeIndex(local_days.unique()).sort_values()
    validation_days = min(14, max(1, len(unique_days) // 4))
    split_day = unique_days[-validation_days]
    discovery = joined.loc[local_days < split_day]
    validation = joined.loc[local_days >= split_day]
    discovery_scale = _zero_intercept_scale(
        discovery["canonical"], discovery["proxy_raw"]
    )
    final_scale = _zero_intercept_scale(joined["canonical"], joined["proxy_raw"])
    joined["proxy_discovery_calibrated"] = (
        joined["proxy_raw"] * discovery_scale
    ).clip(lower=0.0)
    joined["proxy_final_calibrated"] = (
        joined["proxy_raw"] * final_scale
    ).clip(lower=0.0)
    active = (joined["canonical"].abs() > 1e-6) | (joined["proxy_raw"].abs() > 1e-6)
    local_index = joined.index.tz_convert(timezone)
    week = pd.Series(
        local_index.tz_localize(None).to_period("W-SUN").start_time.date.astype(str),
        index=joined.index,
    )
    hour = pd.Series(local_index.hour, index=joined.index)
    forecasts = ("proxy_raw", "proxy_final_calibrated")
    weekly = _group_metrics(
        joined, group=week, actual="canonical", forecasts=forecasts
    )
    hourly = _group_metrics(
        joined, group=hour, actual="canonical", forecasts=forecasts
    )
    validation_raw = _metrics(validation["canonical"], validation["proxy_raw"])
    validation_scaled = _metrics(
        validation["canonical"], validation["proxy_raw"] * discovery_scale
    )
    use_scale = bool(
        validation_scaled["mae"] is not None
        and validation_raw["mae"] is not None
        and validation_scaled["mae"] < validation_raw["mae"]
    )
    selected_scale = final_scale if use_scale else 1.0
    result = {
        "timezone": timezone,
        "calibration_start_day": start_day,
        "calibration_end_day": end_day,
        "rows": int(len(joined)),
        "complete_days": int(len(unique_days)),
        "discovery_end_day": (split_day - pd.Timedelta(days=1)).date().isoformat(),
        "validation_start_day": split_day.date().isoformat(),
        "validation_days": validation_days,
        "calibration": {
            "form": "canonical_solar_gw = max(0, scale * ecmwf_proxy_gw)",
            "intercept_gw": 0.0,
            "discovery_scale": discovery_scale,
            "final_pre_oof_scale": final_scale,
            "learning_data_policy": "only delivery days before first OOF origin",
            "governance_selected_transform": (
                "zero_intercept_scale" if use_scale else "identity_raw_proxy"
            ),
            "selected_scale_for_reconstruction": selected_scale,
            "selection_metric": "forward_validation_mae",
        },
        "all_hours": {
            "raw": _metrics(joined["canonical"], joined["proxy_raw"]),
            "final_calibrated": _metrics(
                joined["canonical"], joined["proxy_final_calibrated"]
            ),
        },
        "active_solar_hours": {
            "raw": _metrics(
                joined.loc[active, "canonical"], joined.loc[active, "proxy_raw"]
            ),
            "final_calibrated": _metrics(
                joined.loc[active, "canonical"],
                joined.loc[active, "proxy_final_calibrated"],
            ),
        },
        "forward_validation": {
            "raw": validation_raw,
            "discovery_calibrated": validation_scaled,
        },
        "weekly_stability": weekly,
        "local_hour_stability": hourly,
        "canonical_cutoff_audit": _cutoff_audit(canonical, timezone=timezone),
        "proxy_cutoff_audit": _cutoff_audit(proxy, timezone=timezone),
    }
    return result, selected_scale, joined


def _local_timezone_sidecar_audit(
    local_path: Path,
    correct_utc_path: Path,
    *,
    start_day: str,
    end_day: str,
    timezone: str,
) -> dict[str, Any]:
    local_full = _load_long(local_path)
    local = _slice_days(
        local_full, start_day=start_day, end_day=end_day, timezone=timezone
    )
    correct = _slice_days(
        _load_long(correct_utc_path),
        start_day=start_day,
        end_day=end_day,
        timezone=timezone,
    )
    joined = pd.DataFrame(
        {"correct_utc": correct["value"], "local_sidecar": local["value"]}
    ).dropna()
    shifts: list[dict[str, Any]] = []
    for shift in range(-3, 4):
        shifted = local["value"].shift(shift)
        pair = pd.concat(
            [correct["value"].rename("correct"), shifted.rename("shifted")],
            axis=1,
        ).dropna()
        metrics = _metrics(pair["correct"], pair["shifted"])
        shifts.append({"shift_hours": shift, **metrics})
    best = min(shifts, key=lambda item: float(item["mae"]))
    physical = pd.DatetimeIndex(local_full.index).tz_convert(timezone)
    offsets = pd.Series(
        [timestamp.utcoffset().total_seconds() / 3600 for timestamp in physical]
    )
    offset_counts = {
        format(float(offset), ".0f"): int(count)
        for offset, count in offsets.value_counts().sort_index().items()
    }
    transition_records: list[dict[str, Any]] = []
    offset_values = offsets.to_numpy(dtype=float)
    for position in np.flatnonzero(offset_values[1:] != offset_values[:-1]) + 1:
        transition_records.append(
            {
                "utc": local_full.index[position].isoformat(),
                "local": physical[position].isoformat(),
                "offset_before_hours": float(offset_values[position - 1]),
                "offset_after_hours": float(offset_values[position]),
            }
        )
    return {
        "path": str(local_path.resolve()),
        "audit_declared_naive_timezone": "Europe/Amsterdam",
        "expected_source_naive_timezone": "UTC",
        "overlap_metrics_without_shift": _metrics(
            joined["correct_utc"], joined["local_sidecar"]
        ),
        "integer_shift_screen": shifts,
        "best_shift_on_summer_overlap": best,
        "interpretation": (
            "During CEST, the local sidecar values are assigned two hours early; "
            "local_sidecar.shift(+2h) matches the UTC materialisation. The semantic "
            "error is one hour during CET and changes inside DST transition days."
        ),
        "full_sidecar_delivery_hours_by_local_utc_offset": offset_counts,
        "dst_transitions_inside_local_sidecar": transition_records,
        "responsible_configuration": {
            "file": str((ROOT / "materialize_saturn_kalman_weather.py").resolve()),
            "generic_solar_naive_timezone_line": 142,
            "nl_override_only_changes_wind_lines": "157-166",
            "existing_audit": str(
                local_path.with_name(local_path.name + ".audit.json").resolve()
            ),
        },
        "separate_fix_not_applied": {
            "change": (
                "Set NL solar naive_timezone=UTC in build_plan, then rebuild into a "
                "new isolated path and validate DST days before any cache migration."
            ),
            "rewrite_existing_cache": False,
        },
    }


def _hydro_analysis(
    canonical_paths: Sequence[Path],
    proxy_path: Path,
    *,
    start_day: str,
    end_day: str,
    timezone: str,
) -> tuple[dict[str, Any], tuple[float, float]]:
    canonical_frames = [_load_long(path) for path in canonical_paths]
    canonical = pd.concat(canonical_frames).sort_index()
    if canonical.index.duplicated().any():
        raise ResidualRecoveryProbeError("Segments hydro canoniques superposes.")
    canonical = _slice_days(
        canonical, start_day=start_day, end_day=end_day, timezone=timezone
    )
    proxy = _slice_days(
        _load_long(proxy_path),
        start_day=start_day,
        end_day=end_day,
        timezone=timezone,
    )
    joined = pd.DataFrame(
        {"canonical": canonical["value"], "proxy_raw": proxy["value"]}
    ).dropna()
    all_days = pd.date_range(start_day, end_day, freq="D")
    present_days = pd.DatetimeIndex(
        joined.index.tz_convert(timezone).normalize().tz_localize(None).unique()
    )
    missing_days = all_days.difference(present_days)
    local_days = joined.index.tz_convert(timezone).normalize().tz_localize(None)
    split_day = pd.Timestamp(end_day) - pd.Timedelta(days=13)
    discovery = joined.loc[local_days < split_day]
    validation = joined.loc[local_days >= split_day]
    discovery_intercept, discovery_slope = _affine_parameters(
        discovery["canonical"], discovery["proxy_raw"]
    )
    final_intercept, final_slope = _affine_parameters(
        joined["canonical"], joined["proxy_raw"]
    )
    joined["proxy_affine"] = np.maximum(
        0.0, final_intercept + final_slope * joined["proxy_raw"]
    )
    validation_affine = np.maximum(
        0.0, discovery_intercept + discovery_slope * validation["proxy_raw"]
    )
    validation_raw = _metrics(validation["canonical"], validation["proxy_raw"])
    validation_cal = _metrics(validation["canonical"], validation_affine)
    use_affine = bool(
        validation_cal["mae"] is not None
        and validation_raw["mae"] is not None
        and validation_cal["mae"] < validation_raw["mae"]
    )
    selected = (
        (final_intercept, final_slope) if use_affine else (0.0, 1.0)
    )
    week = pd.Series(
        joined.index.tz_convert(timezone)
        .tz_localize(None)
        .to_period("W-SUN")
        .start_time.date.astype(str),
        index=joined.index,
    )
    hour = pd.Series(joined.index.tz_convert(timezone).hour, index=joined.index)
    forecasts = ("proxy_raw", "proxy_affine")
    return (
        {
            "calibration_start_day": start_day,
            "calibration_end_day": end_day,
            "expected_days": int(len(all_days)),
            "complete_days": int(len(present_days)),
            "missing_days_not_imputed": [day.date().isoformat() for day in missing_days],
            "rows": int(len(joined)),
            "calibration": {
                "form": "canonical_hydro_gw=max(0, intercept+slope*gma_hydro_gw)",
                "discovery_intercept_gw": discovery_intercept,
                "discovery_slope": discovery_slope,
                "final_pre_oof_intercept_gw": final_intercept,
                "final_pre_oof_slope": final_slope,
                "governance_selected_transform": (
                    "affine_clipped_nonnegative" if use_affine else "identity_raw_proxy"
                ),
                "selected_intercept_gw": selected[0],
                "selected_slope": selected[1],
                "selection_metric": "forward_validation_mae",
            },
            "all_available_hours": {
                "raw": _metrics(joined["canonical"], joined["proxy_raw"]),
                "affine": _metrics(joined["canonical"], joined["proxy_affine"]),
            },
            "forward_validation": {
                "start_day": split_day.date().isoformat(),
                "raw": validation_raw,
                "discovery_affine": validation_cal,
            },
            "weekly_stability": _group_metrics(
                joined, group=week, actual="canonical", forecasts=forecasts
            ),
            "local_hour_stability": _group_metrics(
                joined, group=hour, actual="canonical", forecasts=forecasts
            ),
            "canonical_cutoff_audits": [
                _cutoff_audit(frame, timezone=timezone)
                for frame in canonical_frames
            ],
            "proxy_cutoff_audit": _cutoff_audit(proxy, timezone=timezone),
        },
        selected,
    )


def _aligned_value(path: Path, *, column: str = "value") -> pd.Series:
    return _load_long(path, value_column=column)[column]


def _residual_metrics(
    *,
    zone: str,
    timezone: str,
    load_path: Path,
    wind_path: Path,
    solar_path: Path,
    canonical_residual_path: Path,
    start_day: str,
    end_day: str,
    proxy_solar_path: Path | None = None,
    proxy_scale: float | None = None,
    hydro_path: Path | None = None,
) -> dict[str, Any]:
    load_frame = _slice_days(
        _load_long(load_path), start_day=start_day, end_day=end_day, timezone=timezone
    )
    wind_frame = _slice_days(
        _load_long(wind_path), start_day=start_day, end_day=end_day, timezone=timezone
    )
    solar_frame = _slice_days(
        _load_long(solar_path), start_day=start_day, end_day=end_day, timezone=timezone
    )
    canonical_frame = _slice_days(
        _load_long(canonical_residual_path, value_column=f"{zone.lower()}_residual_load_fcst"),
        start_day=start_day,
        end_day=end_day,
        timezone=timezone,
    )
    data = pd.DataFrame(
        {
            "canonical": canonical_frame[f"{zone.lower()}_residual_load_fcst"],
            "load": load_frame["value"],
            "wind": wind_frame["value"],
            "solar": solar_frame["value"],
        }
    ).dropna()
    data["derived_canonical_components"] = data["load"] - data["wind"] - data["solar"]
    hydro_frame: pd.DataFrame | None = None
    if hydro_path is not None:
        hydro_frame = _slice_days(
            _load_long(hydro_path),
            start_day=start_day,
            end_day=end_day,
            timezone=timezone,
        )
        data["hydro_ror"] = hydro_frame["value"]
        data["derived_with_hydro_ror"] = (
            data["load"] - data["wind"] - data["solar"] - data["hydro_ror"]
        )
    if proxy_solar_path is not None:
        proxy = _slice_days(
            _load_long(proxy_solar_path),
            start_day=start_day,
            end_day=end_day,
            timezone=timezone,
        )["value"]
        data["proxy_solar_raw"] = proxy
        data["derived_proxy_raw"] = data["load"] - data["wind"] - data["proxy_solar_raw"]
        if proxy_scale is not None:
            data["derived_proxy_calibrated"] = (
                data["load"] - data["wind"] - data["proxy_solar_raw"] * proxy_scale
            )
    expected = pd.date_range(
        pd.Timestamp(start_day).tz_localize(timezone).tz_convert("UTC"),
        (pd.Timestamp(end_day) + pd.Timedelta(days=1))
        .tz_localize(timezone)
        .tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    if not data.index.equals(expected):
        raise ResidualRecoveryProbeError(
            f"Reconstruction {zone} incomplete: {len(data)}/{len(expected)}."
        )
    candidates = [column for column in data if column.startswith("derived_")]
    local_days = pd.Series(
        data.index.tz_convert(timezone).date.astype(str), index=data.index
    )
    return {
        "zone": zone,
        "start_day": start_day,
        "end_day": end_day,
        "rows": int(len(data)),
        "formula_candidates": (
            ["load_fcst - wind_generation_fcst - solar_generation_fcst"]
            + (
                [
                    "load_fcst - wind_generation_fcst - solar_generation_fcst - hydro_ror_daily_fcst"
                ]
                if hydro_path is not None
                else []
            )
        ),
        "metrics": {
            candidate: _metrics(data["canonical"], data[candidate])
            for candidate in candidates
        },
        "daily_stability": _group_metrics(
            data, group=local_days, actual="canonical", forecasts=candidates
        ),
        "cutoff_audits": {
            "load": _cutoff_audit(load_frame, timezone=timezone),
            "wind": _cutoff_audit(wind_frame, timezone=timezone),
            "solar": _cutoff_audit(solar_frame, timezone=timezone),
            "canonical_residual": _cutoff_audit(canonical_frame, timezone=timezone),
            **(
                {"hydro_ror": _cutoff_audit(hydro_frame, timezone=timezone)}
                if hydro_frame is not None
                else {}
            ),
        },
    }


def _first_day_reconstruction(
    *,
    zone: str,
    timezone: str,
    load_path: Path,
    wind_path: Path,
    solar_path: Path,
    proxy_scale: float = 1.0,
    hydro_path: Path | None = None,
    hydro_intercept: float = 0.0,
    hydro_slope: float = 1.0,
) -> dict[str, Any]:
    load = _load_long(load_path)
    wind = _load_long(wind_path)
    solar = _load_long(solar_path)
    joined = pd.DataFrame(
        {"load": load["value"], "wind": wind["value"], "solar": solar["value"]}
    ).dropna()
    derived = joined["load"] - joined["wind"] - joined["solar"] * proxy_scale
    hydro: pd.DataFrame | None = None
    if hydro_path is not None:
        hydro = _load_long(hydro_path)
        joined["hydro_proxy"] = hydro["value"]
        hydro_calibrated = np.maximum(
            0.0, hydro_intercept + hydro_slope * joined["hydro_proxy"]
        )
        derived = derived - hydro_calibrated
    return {
        "zone": zone,
        "rows": int(len(joined)),
        "complete_24h": bool(len(joined) == 24 and derived.notna().all()),
        "first_delivery_utc": joined.index.min().isoformat(),
        "last_delivery_utc": joined.index.max().isoformat(),
        "derived_min_gw": float(derived.min()),
        "derived_mean_gw": float(derived.mean()),
        "derived_max_gw": float(derived.max()),
        "formula": (
            "load-wind-solar-calibrated_hydro_proxy"
            if hydro_path is not None
            else "load-wind-calibrated_solar_proxy"
        ),
        "solar_scale": float(proxy_scale),
        "hydro_proxy_transform": (
            {
                "intercept_gw": float(hydro_intercept),
                "slope": float(hydro_slope),
            }
            if hydro_path is not None
            else None
        ),
        "canonical_residual_available": False,
        "continuous_early_history_verified": False,
        "cutoff_audits": {
            "load": _cutoff_audit(load, timezone=timezone),
            "wind": _cutoff_audit(wind, timezone=timezone),
            "solar": _cutoff_audit(solar, timezone=timezone),
            **(
                {"hydro_ror_proxy": _cutoff_audit(hydro, timezone=timezone)}
                if hydro is not None
                else {}
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse locale de la recuperation residual-load pre-OOF."
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--calibration-start-day", default="2024-06-30")
    parser.add_argument("--calibration-end-day", default="2024-09-01")
    parser.add_argument("--residual-start-day", default="2024-08-30")
    parser.add_argument("--residual-end-day", default="2024-09-01")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    experiment = args.experiment_root.expanduser().resolve()
    overlap = experiment / "probe" / "overlap_pre_oof"
    first = experiment / "probe" / "exact_first_day"
    weather = root / "data" / "pit" / "kalman_weather"
    hybrid = root / "data" / "pit" / "kalman_hybrid"
    local_canonical_solar = weather / "nl_solar_generation_fcst.parquet"
    canonical_solar = (
        overlap / "nl_solar_generation_fcst_canonical_utc_pre_oof.parquet"
    )
    proxy_solar = overlap / "nl_solar_generation_fcst_ecmwf_pre_oof.parquet"

    solar, scale, _joined = _solar_analysis(
        canonical_solar,
        proxy_solar,
        start_day=args.calibration_start_day,
        end_day=args.calibration_end_day,
        timezone="Europe/Amsterdam",
    )
    local_sidecar_audit = _local_timezone_sidecar_audit(
        local_canonical_solar,
        canonical_solar,
        start_day=args.calibration_start_day,
        end_day=args.calibration_end_day,
        timezone="Europe/Amsterdam",
    )
    hydro, hydro_transform = _hydro_analysis(
        (
            overlap / "fr_hydro_ror_daily_canonical_pre_oof_seg1.parquet",
            overlap / "fr_hydro_ror_daily_canonical_pre_oof_seg2.parquet",
        ),
        overlap / "fr_hydro_ror_daily_gma_pre_oof.parquet",
        start_day=args.calibration_start_day,
        end_day=args.calibration_end_day,
        timezone="Europe/Paris",
    )
    residual_path = hybrid / "residual_load_market_features.parquet"
    fr = _residual_metrics(
        zone="FR",
        timezone="Europe/Paris",
        load_path=overlap / "fr_load_fcst_overlap.parquet",
        wind_path=weather / "fr_wind_generation_fcst.parquet",
        solar_path=weather / "fr_solar_generation_fcst.parquet",
        hydro_path=overlap / "fr_hydro_ror_daily_fcst_overlap.parquet",
        canonical_residual_path=residual_path,
        start_day=args.residual_start_day,
        end_day=args.residual_end_day,
    )
    nl = _residual_metrics(
        zone="NL",
        timezone="Europe/Amsterdam",
        load_path=overlap / "nl_load_fcst_overlap.parquet",
        wind_path=overlap / "nl_wind_generation_fcst_canonical_utc_overlap.parquet",
        solar_path=overlap / "nl_solar_generation_fcst_canonical_utc_overlap.parquet",
        proxy_solar_path=proxy_solar,
        proxy_scale=scale,
        canonical_residual_path=residual_path,
        start_day=args.residual_start_day,
        end_day=args.residual_end_day,
    )
    first_day = [
        _first_day_reconstruction(
            zone="FR",
            timezone="Europe/Paris",
            load_path=first / "fr_load_fcst.parquet",
            wind_path=first / "fr_wind_generation_fcst.parquet",
            solar_path=first / "fr_solar_generation_fcst.parquet",
            hydro_path=first / "fr_hydro_ror_fcst.parquet",
            hydro_intercept=hydro_transform[0],
            hydro_slope=hydro_transform[1],
        ),
        _first_day_reconstruction(
            zone="NL",
            timezone="Europe/Amsterdam",
            load_path=first / "nl_load_fcst.parquet",
            wind_path=first / "nl_wind_generation_fcst_canonical_utc.parquet",
            solar_path=first / "nl_solar_generation_fcst_ecmwf.parquet",
            proxy_scale=scale,
        ),
    ]
    input_paths = {
        "nl_solar_canonical_utc_pre_oof": canonical_solar,
        "nl_solar_local_sidecar_wrong_timezone_reference": local_canonical_solar,
        "nl_solar_ecmwf_pre_oof": proxy_solar,
        "fr_load_overlap": overlap / "fr_load_fcst_overlap.parquet",
        "nl_load_overlap": overlap / "nl_load_fcst_overlap.parquet",
        "fr_load_first_day": first / "fr_load_fcst.parquet",
        "nl_load_first_day": first / "nl_load_fcst.parquet",
        "fr_wind_first_day": first / "fr_wind_generation_fcst.parquet",
        "fr_solar_first_day": first / "fr_solar_generation_fcst.parquet",
        "nl_wind_first_day": first / "nl_wind_generation_fcst.parquet",
        "nl_solar_ecmwf_first_day": first / "nl_solar_generation_fcst_ecmwf.parquet",
        "nl_wind_canonical_utc_first_day": first / "nl_wind_generation_fcst_canonical_utc.parquet",
        "nl_wind_canonical_utc_overlap": overlap / "nl_wind_generation_fcst_canonical_utc_overlap.parquet",
        "nl_solar_canonical_utc_overlap": overlap / "nl_solar_generation_fcst_canonical_utc_overlap.parquet",
        "fr_hydro_gma_first_day": first / "fr_hydro_ror_fcst.parquet",
        "fr_hydro_canonical_overlap": overlap / "fr_hydro_ror_daily_fcst_overlap.parquet",
        "fr_hydro_canonical_pre_oof_segment_1": overlap / "fr_hydro_ror_daily_canonical_pre_oof_seg1.parquet",
        "fr_hydro_canonical_pre_oof_segment_2": overlap / "fr_hydro_ror_daily_canonical_pre_oof_seg2.parquet",
        "fr_hydro_gma_pre_oof": overlap / "fr_hydro_ror_daily_gma_pre_oof.parquet",
        "canonical_residual_reference": residual_path,
    }
    payload: Mapping[str, Any] = {
        "schema_version": 1,
        "purpose": "research_only_residual_history_recovery_probe",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "network_access_by_this_analysis": False,
        "mutates_lora_or_live_artifacts": False,
        "production_pit_evidence": False,
        "evidence_classification": "historical_asof_reconstruction_for_research",
        "limits": [
            "Saturn provider insertion timestamps are unavailable; revision_time is the query cutoff.",
            "The exact first required day is proven, but continuous 2023-09-04..2024-06-29 coverage is not materialised.",
            "The canonical FR/NL residual and canonical NL solar are absent at the first required cutoff.",
            "Only three pre-OOF days overlap the canonical residual sidecar; residual equivalence evidence is therefore weak.",
        ],
        "inputs": {name: _artifact_record(path) for name, path in input_paths.items()},
        "nl_solar_proxy_calibration": solar,
        "nl_local_solar_sidecar_timezone_audit": local_sidecar_audit,
        "fr_hydro_proxy_calibration": hydro,
        "residual_overlap": {"FR": fr, "NL": nl},
        "first_required_delivery_day": "2023-09-04",
        "first_day_reconstruction": first_day,
        "governance": {
            "allowed_use": "research candidate recovery only",
            "forbidden_use": "silent substitution or production_pit promotion",
            "next_gate": (
                "materialise and audit the complete missing prefix in an isolated path, "
                "then rerun strict coverage/equivalence checks"
            ),
        },
    }
    output = args.output
    if output is None:
        output = experiment / "residual_recovery_probe_manifest.json"
    elif not output.is_absolute():
        output = root / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(rendered + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Manifest: {output}")
    print(
        "NL solar proxy: "
        f"raw_MAE={solar['all_hours']['raw']['mae']:.6f} GW; "
        f"selected={solar['calibration']['governance_selected_transform']}"
    )
    print(
        "FR hydro proxy: "
        f"raw_MAE={hydro['all_available_hours']['raw']['mae']:.6f} GW; "
        f"selected={hydro['calibration']['governance_selected_transform']}; "
        f"missing_days={hydro['missing_days_not_imputed']}"
    )
    print(
        "Residual equivalence: "
        f"FR_MAE={fr['metrics']['derived_with_hydro_ror']['mae']:.12g} GW; "
        f"NL_MAE={nl['metrics']['derived_canonical_components']['mae']:.12g} GW"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ResidualRecoveryProbeError", "build_parser", "main"]
