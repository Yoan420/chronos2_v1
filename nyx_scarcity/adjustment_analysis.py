"""Fixed diagnostic overlays of already-issued out-of-sample tail proposals.

For each frozen forecast f and saved bounded tail correction c, expose
f + alpha*c for the three predeclared alpha values .25, .5 and 1. No fitting,
threshold search, governance change, best-alpha selection or interval creation
occurs here. In particular observations are used only for descriptive scoring;
they can never change the proposed prices, including the live-day proposals.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import pandas as pd

from .reporting import TIMEZONES, _prepare, _score


FRACTIONS = {"proposal25": .25, "proposal50": .5, "proposal100": 1.}
_SHARED = ("sample", "actual", "forecast", "benchmark_forecast", "forecast_origin_utc")
_COLUMNS = ("zone", "timestamp_utc", "forecast_origin_utc", "local_day", "local_hour", "local_label", "sample",
            "expert_ready", "spike_probability", "probability_gate", "raw_correction", "bounded_correction",
            "actual", "forecast", "benchmark_forecast", "candidate_forecast", "applied_correction",
            "selected_weight", "gate_reason")
_REQUIRED = set(_COLUMNS) - {"forecast_origin_utc", "local_day", "local_hour", "local_label"}


def _close(left: pd.Series, right: pd.Series) -> bool:
    return bool(np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=0, atol=1e-8, equal_nan=True))


def _validate_arithmetic(frame: pd.DataFrame, clip: float, name: str) -> None:
    for column in ("raw_correction", "bounded_correction", "applied_correction"):
        if frame[column].dropna().lt(0).any():
            raise ValueError(f"{name}: {column} must be nonnegative.")
    if frame.bounded_correction.dropna().gt(clip + 1e-8).any():
        raise ValueError(f"{name}: bounded correction exceeds the declared clip.")
    if not _close(frame.bounded_correction, frame.raw_correction.clip(lower=0, upper=clip)):
        raise ValueError(f"{name}: bounded correction is not the saved raw correction clipped at {clip}.")
    known = frame[["applied_correction", "selected_weight", "bounded_correction"]].notna().all(axis=1)
    if not _close(frame.loc[known, "applied_correction"],
                  frame.loc[known, "selected_weight"] * frame.loc[known, "bounded_correction"]):
        raise ValueError(f"{name}: governed correction is not selected_weight * bounded_correction.")
    known = frame[["candidate_forecast", "forecast", "applied_correction"]].notna().all(axis=1)
    if not _close(frame.loc[known, "candidate_forecast"],
                  frame.loc[known, "forecast"] + frame.loc[known, "applied_correction"]):
        raise ValueError(f"{name}: governed price is not NYX + applied_correction.")
    active = frame.bounded_correction.gt(0)
    valid_gate = frame.expert_ready & frame.spike_probability.gt(frame.probability_gate)
    if (active & ~valid_gate).any():
        raise ValueError(f"{name}: positive proposal without an available expert above its saved probability gate.")


def _align(predictions: Mapping[str, pd.DataFrame], clip: float) -> dict[str, pd.DataFrame]:
    if not isinstance(predictions, Mapping) or not predictions or "hgb_v1" not in predictions:
        raise ValueError("Frozen predictions including the hgb_v1 control are required.")
    frames = {}
    for name, value in predictions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Variant identifiers must be nonempty strings.")
        if not isinstance(value, pd.DataFrame) or not _REQUIRED.issubset(value.columns):
            raise ValueError(f"{name}: missing saved proposal fields; predictions must be a DataFrame.")
        frame = _prepare(value)
        if frame.empty:
            raise ValueError("Frozen predictions cannot be empty.")
        _validate_arithmetic(frame, clip, name)
        for label, fraction in FRACTIONS.items():
            frame[label] = frame.forecast + fraction * frame.bounded_correction
        if np.isinf(frame[list(FRACTIONS)].to_numpy(float)).any():
            raise ValueError(f"{name}: proposed prices overflow finite arithmetic.")
        frames[name] = frame
    base = frames["hgb_v1"]
    for name, frame in frames.items():
        if not frame[["zone", "timestamp_utc"]].equals(base[["zone", "timestamp_utc"]]):
            raise ValueError(f"{name}: physical forecast identities/shapes differ from the frozen control.")
        for column in _SHARED:
            if column in {"actual", "forecast", "benchmark_forecast"}:
                equal = _close(frame[column], base[column])
            else:
                equal = frame[column].equals(base[column])
            if not equal:
                raise ValueError(f"{name}: shared frozen source field {column} differs.")
    return frames


def _scores(frames: Mapping[str, pd.DataFrame], mask: np.ndarray) -> dict[str, Any]:
    base = frames["hgb_v1"].loc[mask]
    nyx, storm = _score(base, "forecast"), _score(base, "benchmark_forecast")
    scores: dict[str, Any] = {"nyx": nyx, "storm": storm, "variants": {}}
    for name, frame in frames.items():
        selected = frame.loc[mask]
        points = {}
        for label, column in {"governed": "candidate_forecast", **{key: key for key in FRACTIONS}}.items():
            score = _score(selected, column)
            score["gain_vs_nyx_mae_eur_mwh"] = nyx["mae_eur_mwh"] - score["mae_eur_mwh"] if len(selected) else None
            score["gain_vs_storm_mae_eur_mwh"] = storm["mae_eur_mwh"] - score["mae_eur_mwh"] if len(selected) else None
            points[label] = score
        scores["variants"][name] = points
    return scores


def _interventions(frame: pd.DataFrame, mask: np.ndarray, point: str) -> dict[str, Any]:
    active = mask & frame[point].sub(frame.forecast).gt(1e-9).to_numpy()
    selected = frame.loc[active]
    gain = (selected.forecast - selected.actual).abs() - (selected[point] - selected.actual).abs()
    improved, harmed = gain.gt(1e-9), gain.lt(-1e-9)
    count = len(selected)
    return {"active_hours": count, "improved_absolute_error_hours": int(improved.sum()),
            "harmed_absolute_error_hours": int(harmed.sum()), "tied_hours": int((~improved & ~harmed).sum()),
            "precision_beneficial": float(improved.mean()) if count else None,
            "precision_definition": "fraction of proposed interventions strictly reducing absolute NYX error; not classifier precision",
            "total_benefit_eur_mwh": float(gain.clip(lower=0).sum()),
            "total_harm_eur_mwh": float((-gain).clip(lower=0).sum()),
            "net_gain_eur_mwh": float(gain.sum()),
            "mean_correction_eur_mwh": float((selected[point] - selected.forecast).mean()) if count else None,
            "mean_absolute_error_gain_eur_mwh": float(gain.mean()) if count else None,
            "aggregation_note": "sums of hourly price errors are diagnostic, not monetary P&L or economic value"}


def _summary(frames: Mapping[str, pd.DataFrame], common: np.ndarray, population: np.ndarray) -> dict[str, Any]:
    base = frames["hgb_v1"]
    selected = common & population
    tails = {}
    for percent, quantile in ((1, .99), (5, .95)):
        threshold = float(base.actual.loc[selected].quantile(quantile)) if selected.any() else None
        tail = selected & base.actual.ge(threshold).to_numpy() if threshold is not None else selected
        tails[f"top_{percent}_percent"] = {
            "threshold_eur_mwh": threshold, "definition": "ex_post_observed_price_quantile_on_identical_common_support",
            "ties_may_increase_fraction": True, "used_for_training_or_selection": False,
            "scores": _scores(frames, tail)}
    represented = population & base.in_evaluation_window.to_numpy(bool)
    return {"represented_window_hours": int(represented.sum()), "paired_hours": int(selected.sum()),
            "unpaired_window_hours": int((represented & ~common).sum()),
            "annual": _scores(frames, selected), "tails": tails,
            "interventions": {name: {point: _interventions(frame, selected, point) for point in FRACTIONS}
                              for name, frame in frames.items()}}


def build_adjustments(predictions: Mapping[str, pd.DataFrame], *,
                      correction_clip_eur_mwh: float = 400.) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return compact long projections and common-support descriptive metrics.

    Missing numerical inputs stay missing and remove the corresponding physical
    hour from *every* model's comparison. Fallback rows remain in annual scores.
    Tail labels use observed prices only after forecasts have been projected.
    Live rows remain visible but are never included in any performance metric.
    """
    if (isinstance(correction_clip_eur_mwh, bool)
            or not isinstance(correction_clip_eur_mwh, (int, float))
            or not math.isfinite(correction_clip_eur_mwh) or correction_clip_eur_mwh <= 0):
        raise ValueError("correction_clip_eur_mwh must be a finite positive number.")
    frames = _align(predictions, float(correction_clip_eur_mwh))
    base = frames["hgb_v1"]
    common = base.in_evaluation_window.to_numpy(bool, copy=True)
    for frame in frames.values():
        common &= frame["sample"].eq("evaluation").to_numpy()
        common &= np.isfinite(frame[["actual", "forecast", "benchmark_forecast", "candidate_forecast", *FRACTIONS]].to_numpy(float)).all(axis=1)
    projected = []
    for name, frame in frames.items():
        part = frame[[*_COLUMNS, *FRACTIONS, "in_evaluation_window"]].copy()
        part.insert(0, "variant_id", name)
        part["common_support"] = common
        projected.append(part)
    windows = {}
    for zone, group in base.loc[base.in_evaluation_window].groupby("zone"):
        end = pd.Timestamp(group.local_day.max())
        start = end - pd.Timedelta(days=364)
        expected = pd.date_range(start.tz_localize(TIMEZONES[zone]), (end + pd.Timedelta(days=1)).tz_localize(TIMEZONES[zone]),
                                 freq="h", inclusive="left").tz_convert("UTC")
        windows[zone] = {"requested_start_day": str(start.date()), "end_day": str(end.date()), "requested_days": 365,
                         "represented_start_day": str(group.local_day.min()), "represented_days": int(group.local_day.nunique()),
                         "expected_hours": len(expected), "represented_hours": len(group),
                         "common_paired_hours": int(common[group.index].sum()),
                         "complete_365_common_support": int(common[group.index].sum()) == len(expected)}
    summary = {"schema_version": 1, "diagnostic_only": True, "production_activation": False,
               "operational_forecasts_modified": False, "governance_modified": False, "refit_performed": False,
               "retrospective_selection_performed": False, "best_alpha_selected": False, "quantiles_created": False,
               "fractions": dict(FRACTIONS), "variants": list(frames), "correction_clip_eur_mwh": float(correction_clip_eur_mwh),
               "formula": "proposal(alpha) = frozen NYX forecast + alpha * saved bounded_correction",
               "proposal_note": "Fixed ungoverned research overlays, not deployed forecasts; probability gate and correction are reused unchanged.",
               "pairing_policy": "identical finite actual, NYX, Storm, every governed forecast and all three proposals on evaluation physical hours",
               "baseline_fallback_rows_included": True, "live_rows_excluded": int(base["sample"].eq("live").sum()),
               "windows": windows,
               "overall": _summary(frames, common, np.ones(len(base), dtype=bool)),
               "by_zone": {zone: _summary(frames, common, base.zone.eq(zone).to_numpy()) for zone in sorted(base.zone.unique())}}
    return pd.concat(projected, ignore_index=True), summary
