"""Annual, isolated NYX/Test2 evaluation using the unchanged numerical recipes.

The 365 scored civil days are preceded by 91 Test2 OOF warmup days. Every
Test2 origin, including the first warmup origin, receives exactly 365 complete
past civil days of baseline forecasts and labels. Nothing here writes files,
fetches data, launches processes, or changes the pinned live/fast modules.

Checkpoint integrity and the causal provenance of supplied baseline forecasts
remain caller obligations. Checkpoints contain predictions, never target labels.
Publication PIT certification and production promotion are not claimed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd

from . import nyx_live_hybrid as recipe


PROTOCOL_VERSION = "nyx_annual_hybrid_v1"
EVALUATION_DAYS = 365
TRAINING_DAYS = 365
ROUTING_DAYS = 90
OOF_WARMUP_DAYS = 91  # Thirteen weekly blocks; first route has >=90 prior days.
TEST2_OOF_DAYS = EVALUATION_DAYS + OOF_WARMUP_DAYS
BASELINE_HISTORY_DAYS = TEST2_OOF_DAYS + TRAINING_DAYS
SEED = 20260923


def _grid_frame(frame, first, stop, name):
    if (not isinstance(frame.index, pd.DatetimeIndex)
            or str(frame.index.tz) != "UTC"
            or not frame.index.equals(recipe._grid(first, stop))):
        raise ValueError(f"{name}: exact complete UTC physical-hour civil-day grid required")


def _extended_baseline(frame, first, delivery, zone):
    """Reject incomplete/future input before taking the required 821-day suffix."""
    if (not isinstance(frame, pd.DataFrame) or frame.empty
            or not isinstance(frame.index, pd.DatetimeIndex)
            or str(frame.index.tz) != "UTC" or frame.index.hasnans):
        raise ValueError(f"{zone}: UTC historical baseline required")
    supplied_first = frame.index[0].tz_convert(recipe.TZ).date()
    if supplied_first > first:
        raise ValueError(f"{zone}: at least {BASELINE_HISTORY_DAYS} complete baseline days required")
    _grid_frame(frame, supplied_first, delivery, f"{zone} extended baseline")
    validated = recipe._baseline(frame, allow_actual=True)
    required = validated.loc[recipe._grid(first, delivery)].copy(deep=True)
    _grid_frame(required, first, delivery, f"{zone} required baseline")
    return required


def _training_contract(pair, origin, stop, delivery, threads, iterations, seed):
    first = origin - timedelta(days=TRAINING_DAYS)
    return {"version": PROTOCOL_VERSION, "pair": list(pair),
            "delivery_day": str(delivery), "origin_day": str(origin),
            "target_end_exclusive": str(stop), "training_start_day": str(first),
            "training_end_exclusive": str(origin), "training_complete_days": TRAINING_DAYS,
            "training_physical_hours_per_country": len(recipe._grid(first, origin)),
            "threads": threads, "iterations": iterations, "seed": seed,
            "target_actual_included": False}


def _country_first(frame, pair):
    return pd.concat([frame.loc[frame.zone == zone].sort_index() for zone in pair])


def _attach_labels(panel, history, pair):
    output = panel.copy(deep=True)
    # Both checkpoint validation and routing enforce this same country order.
    output["actual"] = np.concatenate([
        history[zone].loc[output.loc[output.zone == zone].index, "actual"].to_numpy()
        for zone in pair])
    return output


def run_pair_pipeline(history_by_zone, forecast_by_zone, covariates_by_zone, *, pair,
                      delivery_day, threads=1, iterations=120, seed=SEED,
                      load_checkpoint=None, save_checkpoint=None):
    """Return 365 evaluated hybrid days and one label-free delivery forecast.

    For delivery 2026-09-24: scoring is 2025-09-24..2026-09-23; Test2 OOF
    starts 2025-06-25 and the earliest complete training year starts 2024-06-25.
    Earlier contiguous baseline rows are accepted, but no labels at/after the
    delivery day. Covariate sources need the unchanged recipe's feature warmup
    before the first training day. Every used feature must be finite.

    load_checkpoint(stage, origin_iso) -> (panel, audit) or None, with stages
    ``test2`` and ``hybrid``. The caller must namespace/checksum checkpoints by
    this new annual plan and pair. Annual training contracts are also checked
    here, so old partial-training checkpoints cannot be silently accepted.
    """
    pair, delivery = recipe._pair(pair), recipe._day(delivery_day)
    if threads != 1 or iterations != 120 or seed != SEED:
        raise ValueError("Annual recipe requires one thread, 120 iterations and seed 20260923")
    if any(set(mapping) != set(pair) for mapping in
           (history_by_zone, forecast_by_zone, covariates_by_zone)):
        raise ValueError("Exactly the requested country pair is required throughout")
    first_hybrid = delivery - timedelta(days=EVALUATION_DAYS)
    first_test2 = first_hybrid - timedelta(days=OOF_WARMUP_DAYS)
    first_baseline = first_test2 - timedelta(days=TRAINING_DAYS)
    history, forecast = {}, {}
    for zone in pair:
        history[zone] = _extended_baseline(history_by_zone[zone], first_baseline, delivery, zone)
        forecast[zone] = recipe._baseline(forecast_by_zone[zone], allow_actual=False)
        _grid_frame(forecast[zone], delivery, delivery + timedelta(days=1), f"{zone} forecast")
    features, feature_audit = recipe.build_pair_features(covariates_by_zone, pair)
    required_features = recipe._grid(first_baseline, delivery + timedelta(days=1))
    for zone in pair:
        frame = features[zone]
        if (not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC"
                or frame.index.hasnans or frame.index.has_duplicates
                or not frame.index.is_monotonic_increasing or not frame.columns.is_unique
                or not required_features.isin(frame.index).all()):
            raise ValueError(f"{zone}: complete annual training/target feature coverage required")
        if not np.isfinite(frame.loc[required_features].to_numpy(dtype=float)).all():
            raise ValueError(f"{zone}: nonfinite annual features; supply sufficient earlier feature warmup")
    blocks = []
    origin = first_test2
    while origin < delivery:
        blocks.append((origin, min(origin + timedelta(days=7), delivery), False))
        origin += timedelta(days=7)
    blocks.append((delivery, delivery + timedelta(days=1), True))
    predictions, audits = [], []
    future_panel = None
    for origin, stop, is_forecast in blocks:
        train_grid = recipe._grid(origin - timedelta(days=TRAINING_DAYS), origin)
        target_grid = recipe._grid(origin, stop)
        training = {zone: history[zone].loc[train_grid].copy(deep=True) for zone in pair}
        current = {zone: (forecast[zone] if is_forecast else history[zone]).loc[target_grid]
                   .drop(columns="actual", errors="ignore").copy(deep=True) for zone in pair}
        for zone in pair:
            _grid_frame(training[zone], origin - timedelta(days=TRAINING_DAYS), origin,
                        f"{zone} rolling Test2 training")
        contract = _training_contract(pair, origin, stop, delivery, threads, iterations, seed)
        cached = load_checkpoint("test2", str(origin)) if load_checkpoint else None
        if cached is None:
            panel, numerical_audit = recipe.fit_test2_block(
                training, current, features, pair, origin_day=origin,
                threads=threads, iterations=iterations, seed=seed)
            audit = {"protocol_version": PROTOCOL_VERSION, "pair": list(pair),
                     "origin_day": str(origin), "annual_contract": contract,
                     "numerical_fit_audit": numerical_audit}
        else:
            panel, audit = cached
            if (audit.get("protocol_version") != PROTOCOL_VERSION
                    or audit.get("annual_contract") != contract):
                raise ValueError("Cached Test2 annual complete-training contract mismatch")
        recipe._validate_checkpoint(panel, audit, current, features, pair, origin, stop)
        if cached is None and save_checkpoint:
            save_checkpoint("test2", str(origin), panel.copy(deep=True), deepcopy(audit))
        audits.append(deepcopy(audit))
        if is_forecast:
            future_panel = panel.copy(deep=True)
        else:
            predictions.append(_attach_labels(panel, history, pair))
    all_history = _country_first(pd.concat(predictions), pair)
    recipe._checked_panel(all_history, pair, first_test2, delivery, labels=True)
    routed, policies = [], []
    future_hybrid = None
    for origin, stop, is_forecast in blocks:
        if origin < first_hybrid:
            continue
        days = np.asarray(all_history.index.tz_convert(recipe.TZ).date)
        past = all_history.loc[(days >= origin - timedelta(days=ROUTING_DAYS)) & (days < origin)].copy()
        recipe._checked_panel(past, pair, origin - timedelta(days=ROUTING_DAYS), origin, labels=True)
        target = future_panel if is_forecast else all_history
        target_days = np.asarray(target.index.tz_convert(recipe.TZ).date)
        current = target.loc[(target_days >= origin) & (target_days < stop)]\
            .drop(columns="actual", errors="ignore").copy(deep=True)
        policy = recipe.select_pair_rule(past, pair, origin)
        expected = recipe.apply_pair_rule(current, policy, pair)
        audit = {"protocol_version": PROTOCOL_VERSION, "pair": list(pair),
                 "origin_day": str(origin), "evaluation_start_day": str(first_hybrid),
                 "evaluation_end_exclusive": str(delivery), "policy": policy}
        cached = load_checkpoint("hybrid", str(origin)) if load_checkpoint else None
        if cached is None:
            output = expected
            if save_checkpoint:
                save_checkpoint("hybrid", str(origin), output.copy(deep=True), deepcopy(audit))
        else:
            output, saved_audit = cached
            if saved_audit != audit:
                raise ValueError("Cached annual routing differs from strict past-only recalculation")
            pd.testing.assert_frame_equal(output, expected, check_exact=True)
        policies.append(deepcopy(audit))
        if is_forecast:
            future_hybrid = output.copy(deep=True)
        else:
            routed.append(_attach_labels(output, history, pair))
    historical_hybrid = _country_first(pd.concat(routed), pair)
    recipe._checked_panel(historical_hybrid, pair, first_hybrid, delivery, labels=True)
    recipe._checked_panel(future_hybrid, pair, delivery, delivery + timedelta(days=1), labels=False)
    return {"historical_test2": all_history, "historical_hybrid": historical_hybrid,
            "forecast_test2": future_panel, "forecast_hybrid": future_hybrid,
            "feature_audit": feature_audit, "fit_audits": audits, "policies": policies,
            "protocol": {"version": PROTOCOL_VERSION, "pair": list(pair),
                         "delivery_day": str(delivery), "baseline_start_day": str(first_baseline),
                         "baseline_required_days": BASELINE_HISTORY_DAYS,
                         "test2_start_day": str(first_test2), "hybrid_start_day": str(first_hybrid),
                         "hybrid_end_exclusive": str(delivery),
                         "test2_historical_days": TEST2_OOF_DAYS,
                         "hybrid_historical_days": EVALUATION_DAYS,
                         "test2_training_days": TRAINING_DAYS, "complete_training_required": True,
                         "routing_past_complete_days": ROUTING_DAYS,
                         "oof_warmup_days": OOF_WARMUP_DAYS,
                         "test2_origins": len(blocks), "routing_origins": len(policies),
                         "threads": threads, "iterations": iterations, "seed": seed,
                         "recipe_DE_NL": "unchanged numerical Test2 and routing; complete rolling365 fits",
                         "extension_BE_FR": "separately trained BE<->FR with country_is_fr and own wind",
                         "production_modified": False, "pit_publication_evidence_verified": False,
                         "independent_validation": False}}


__all__ = ["PROTOCOL_VERSION", "EVALUATION_DAYS", "TRAINING_DAYS", "ROUTING_DAYS",
           "OOF_WARMUP_DAYS", "TEST2_OOF_DAYS", "BASELINE_HISTORY_DAYS", "run_pair_pipeline"]
