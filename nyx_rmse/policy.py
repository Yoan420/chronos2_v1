"""Matched-source chronological MSE replay, isolated from operational NYX.

The source detector is frozen. Two signed residual means share its exact core,
fit schedule and readiness; daily zonal governance sees only earlier issued
predictions and labels available at the current D-1 08 h origin. The original
NYX quantiles remain baseline diagnostics, never relabelled as candidate bands.
"""
from __future__ import annotations

import json
from collections import deque
import numpy as np
import pandas as pd

from nyx_scarcity import policy as base
from nyx_fundamental_stress.features import make_fundamental_features
from nyx_fundamental_stress.policy import _check_features
from .models import fit_means, predict_means, validate_options


KEYS = ["zone", "timestamp_utc", "forecast_origin_utc"]
STRATEGIES = {"residual_mse": ("residual_mse_direct", "nyx_rmse"),
              "mixture_mean": ("mixture_mean_direct", "mixture_mean_governed")}
RMSEPolicyError = base.ScarcityPolicyError


def _mapping(value):
    return json.loads(value) if isinstance(value, str) else dict(value)


def _fit_core(data, day, cutoff, parameters, zones, frozen):
    """Check membership even on a cache hit; revised labels cannot change it."""
    if pd.Timestamp(frozen["fit_cutoff_utc"]) != cutoff:
        raise RMSEPolicyError("Frozen fit cutoff differs from D-1 08 h.")
    first = max(str(data._day.min()), (pd.Timestamp(day)-pd.Timedelta(days=365)).strftime("%Y-%m-%d"))
    split = (pd.Timestamp(day)-pd.Timedelta(days=parameters["calibration_days"])).strftime("%Y-%m-%d")
    train = data.loc[data._day.ge(first) & data._day.lt(day) & data._features_valid
                     & data._label_valid & data.label_available_at_utc.le(cutoff)]
    core = train.loc[train._day.lt(split)]
    calendar_days = (pd.Timestamp(day)-pd.Timestamp(first)).days
    if (len(train) != int(frozen["training_rows"])
            or len(core) != int(frozen["signed_model_training_rows"])
            or first != frozen["training_start_day"] or split != frozen["calibration_start_day"]
            or calendar_days != int(frozen["training_days"])
            or bool(frozen["full_365_day_training"]) != (calendar_days == 365)):
        raise RMSEPolicyError("The MSE training core differs from the frozen classifier core.")
    thresholds = _mapping(frozen["thresholds_eur_mwh"])
    if set(thresholds) != set(zones):
        raise RMSEPolicyError("Frozen thresholds have different countries.")
    for zone in zones:
        errors = core.loc[core.zone.eq(zone), "_error"].to_numpy(float)
        expected = max(parameters["minimum_threshold_eur_mwh"],
                       float(np.quantile(errors, parameters["threshold_quantile"]))) if len(errors) else np.nan
        if not np.isfinite(expected) or not np.isclose(expected, thresholds[zone], rtol=0, atol=1e-10):
            raise RMSEPolicyError("Frozen event threshold no longer matches historical labels.")
    return core, thresholds


def _fit(data, matrix, day, cutoff, parameters, zones, frozen, options, load_fit, save_fit):
    record = {**frozen, "detector_retrained": False, "cache_reused": False,
        "training_scope": "exact_frozen_core_excluding_chronological_calibration"}
    if pd.Timestamp(frozen["fit_cutoff_utc"]) != cutoff:
        raise RMSEPolicyError("Frozen fit cutoff differs from D-1 08 h.")
    if frozen["status"] != "trained":
        return None, record
    core, thresholds = _fit_core(data, day, cutoff, parameters, zones, frozen)
    full_span = int(frozen["training_days"]) == 365
    # Unique hourly identities, a fixed country set and bounded civil dates were
    # already checked by _prepare. Reaching this exact expected row count thus
    # certifies that *every* prior physical hour/country is feature-eligible and
    # has a label available at this origin, not merely a 365-day calendar span.
    expected_rows = base._hours(frozen["training_start_day"], day)*len(zones)
    complete_history = full_span and int(frozen["training_rows"]) == expected_rows
    record.update(full_365_calendar_span=full_span, complete_365_eligible_history=complete_history)
    if options["require_full_training_history"] and not complete_history:
        record.update(status="fallback", reason="strict_365_day_training_prefix_unavailable")
        return None, record
    metadata = {"fit_day": day, "fit_cutoff": cutoff, "features": list(parameters["feature_columns"]),
                "zones": tuple(zones), "thresholds": thresholds, "options": options}
    state = load_fit(day) if load_fit is not None else None
    if state is not None:
        if not isinstance(state, dict) or any(state.get(key) != value for key, value in metadata.items()):
            raise RMSEPolicyError("Cached fit metadata differs from the current historical fold.")
        if state["models"]["core_rows"] != len(core):
            raise RMSEPolicyError("Cached model has a different training core.")
        record["cache_reused"] = True
    else:
        models = fit_means(matrix[core.index.to_numpy()], core._error.to_numpy(float),
                          core.zone.map(thresholds).to_numpy(float), options)
        state = {**metadata, "models": models}
    record.update(mean_core_rows=len(core), mean_normal_rows=state["models"]["normal_rows"],
        mean_spike_rows=state["models"]["spike_rows"], mean_negative_rows=state["models"]["negative_rows"],
        mean_max_label_available_at_utc=core.label_available_at_utc.max(),
        mean_training_end_day=str(core._day.max()), calibration_labels_used_for_mean_fit=False)
    state["fold_record"] = dict(record)
    if not record["cache_reused"] and save_fit is not None:
        save_fit(day, state)
    return state, record


def _eligible(records, zone, day, cutoff, lookback_days):
    if not records:
        return pd.DataFrame()
    first = (pd.Timestamp(day)-pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    past = pd.concat(records, ignore_index=True)
    usable = (past.zone.eq(zone) & past._day.ge(first) & past._day.lt(day)
              & past.expert_ready & past._label_valid & past.label_available_at_utc.le(cutoff))
    past = past.loc[usable].copy()
    if past.empty:
        return past
    complete = past.groupby("_day").size().eq(past.groupby("_day")._physical_day_hours.first())
    return past.loc[past._day.isin(complete.index[complete])]


def _govern(past, strategy, zone, day, cutoff, governance):
    """Pooled-hour MSE improvement, with a civil-day clustered standard error."""
    days = past._day.nunique() if len(past) else 0
    records = []
    best_weight, best_score = 0., 0.
    reason = "insufficient_complete_oos_days" if days < governance["minimum_days"] else "no_candidate_passes_mse_and_mae_guards"
    for weight in governance["weights"]:
        record = {"zone": zone, "delivery_day": day, "forecast_origin_utc": cutoff,
            "strategy": strategy, "weight": weight, "n_days": days, "n_hours": len(past),
            "eligible": False, "selected": False, "reason": "baseline_fallback" if weight == 0 else reason,
            "mse_gain": np.nan, "mse_gain_standard_error": np.nan, "mse_gain_lower_bound": np.nan,
            "mae_gain": np.nan, "ordinary_mae_gain": np.nan, "ordinary_hours": 0, "changed_days": 0,
            "max_label_available_at_utc": past.label_available_at_utc.max() if len(past) else pd.NaT}
        if weight and days >= governance["minimum_days"]:
            error = past._error.to_numpy(float)
            delta = weight*past[strategy+"_bounded_correction"].to_numpy(float)
            mse_gain = error**2-(error-delta)**2
            absolute_gain = np.abs(error)-np.abs(error-delta)
            ordinary = np.abs(error) < past.threshold_eur_mwh.to_numpy(float)
            changed = np.abs(delta) > 1e-12
            changed_days = past.loc[changed, "_day"].nunique()
            daily = pd.DataFrame({"day": past._day.to_numpy(), "gain": mse_gain}).groupby("day").gain.agg(["sum", "size"])
            average = float(mse_gain.mean())
            centered_sum = daily["sum"].to_numpy()-average*daily["size"].to_numpy()
            se = float(np.sqrt(days/(days-1)*np.square(centered_sum).sum())/len(past))
            lower = average-governance["uncertainty_z"]*se
            mae = float(absolute_gain.mean())
            ordinary_mae = float(absolute_gain[ordinary].mean()) if ordinary.any() else np.nan
            failures = []
            if changed_days < governance["minimum_changed_days"]:
                failures.append("insufficient_changed_days")
            if lower <= governance["minimum_mse_gain"]+1e-12:
                failures.append("mse_gain_not_robust")
            if mae < -governance["maximum_mae_degradation"]-1e-12:
                failures.append("mae_guard")
            if not ordinary.any() or ordinary_mae < -governance["maximum_ordinary_mae_degradation"]-1e-12:
                failures.append("ordinary_mae_guard")
            record.update(mse_gain=average, mse_gain_standard_error=se, mse_gain_lower_bound=lower,
                mae_gain=mae, ordinary_mae_gain=ordinary_mae, ordinary_hours=int(ordinary.sum()),
                changed_days=int(changed_days), eligible=not failures,
                reason=";".join(failures) if failures else "passed_mse_and_mae_guards")
            if not failures and lower > best_score+1e-12:
                best_score, best_weight = lower, weight
        records.append(record)
    for record in records:
        record["selected"] = record["weight"] == best_weight
    return best_weight, "governed_mse_gain_and_mae_guards" if best_weight else reason, records


def run_replay(panel, detector, source_folds, settings, options=None, *, load_fit=None, save_fit=None, progress=None):
    """Return wide forecasts in original panel order; checkpoint only pure fits."""
    options = validate_options(options)
    for callback in (load_fit, save_fit, progress):
        if callback is not None and not callable(callback):
            raise RMSEPolicyError("Fit cache and progress hooks must be callable.")
    required = {*KEYS, "expert_ready", "expert_fit_day", "spike_probability", "threshold_eur_mwh"}
    if required.difference(detector) or "fit_day" not in source_folds or source_folds.empty:
        raise RMSEPolicyError("Frozen detector predictions and non-empty folds are required.")
    pd.testing.assert_frame_equal(panel, detector[panel.columns], check_exact=True)
    augmented, features, required_features, feature_audit = make_fundamental_features(panel, variant="fundamental")
    _check_features(features)
    pd.testing.assert_frame_equal(augmented[features], detector[features], check_exact=True)
    parameters = base._parameters({**settings, "threads": options["threads"],
        "feature_columns": features, "required_feature_columns": required_features})
    data = base._prepare(augmented, parameters)
    zones = tuple(sorted(data.zone.unique()))
    matrix = base._matrix(data, features, zones)
    source = detector.set_index(KEYS, drop=False)
    source_ready = base._flag(detector, "expert_ready")
    source = source.assign(expert_ready=source_ready.to_numpy())
    frozen_folds = source_folds.set_index("fit_day", drop=False)
    if source.index.has_duplicates or frozen_folds.index.has_duplicates:
        raise RMSEPolicyError("Duplicate source detector identities or refit days.")
    blocks, folds, governance_records = [], [], []
    oos = deque()
    state, fitted_at = None, None
    day_groups = list(data.groupby("_day", sort=True))
    for position, (day, block) in enumerate(day_groups):
        current = block.copy()
        cutoff = current.forecast_origin_utc.iloc[0]
        if fitted_at is None or (pd.Timestamp(day)-pd.Timestamp(fitted_at)).days >= parameters["refit_every_days"]:
            if day not in frozen_folds.index:
                raise RMSEPolicyError("Refit schedule differs from the frozen detector.")
            if progress is not None and frozen_folds.loc[day, "status"] == "trained":
                progress(f"[NYX RMSE] verification/reprise/entrainement du fold {day}; jour={position+1}/{len(day_groups)}")
            state, fold = _fit(data, matrix, day, cutoff, parameters, zones,
                frozen_folds.loc[day].to_dict(), options, load_fit, save_fit)
            folds.append(fold)
            fitted_at = day
            if progress is not None:
                progress(f"[NYX RMSE] {day}: {fold['status']}; core={fold.get('mean_core_rows', 0)}; cache={fold['cache_reused']}; jour={position+1}/{len(day_groups)}")
        frozen = source.reindex(pd.MultiIndex.from_frame(current[KEYS])).copy()
        frozen.index = current.index
        if frozen.zone.isna().any():
            raise RMSEPolicyError("A current identity is absent from the frozen detector replay.")
        ready = current._features_valid.to_numpy(bool) & (state is not None)
        detector_ready = frozen.expert_ready.to_numpy(bool)
        if not np.array_equal(ready, detector_ready):
            strict_fallback = options["require_full_training_history"] and state is None and folds[-1]["reason"] == "strict_365_day_training_prefix_unavailable"
            if not strict_fallback or ready.any():
                raise RMSEPolicyError("MSE replay changed the frozen detector readiness coverage.")
        current["expert_ready"] = ready
        current["source_expert_ready"] = detector_ready
        current["expert_fit_day"] = state["fit_day"] if state is not None else None
        current["threshold_eur_mwh"] = frozen.threshold_eur_mwh.to_numpy()
        current["spike_probability"] = frozen.spike_probability.to_numpy()
        raw = {strategy: np.zeros(len(current), float) for strategy in STRATEGIES}
        if ready.any():
            selected = current.loc[ready]
            thresholds = selected.zone.map(state["thresholds"]).to_numpy(float)
            if (not frozen.loc[ready, "expert_fit_day"].eq(state["fit_day"]).all()
                    or not selected.forecast_origin_utc.ge(state["fit_cutoff"]).all()
                    or not np.allclose(selected.threshold_eur_mwh.to_numpy(float), thresholds, rtol=0, atol=1e-10)):
                raise RMSEPolicyError("Frozen inference state, cutoff or event thresholds differ.")
            estimated = predict_means(state["models"], matrix[selected.index.to_numpy()],
                                     selected.spike_probability.to_numpy(float), thresholds)
            for strategy in STRATEGIES:
                raw[strategy][ready] = estimated[strategy]
        for strategy, (direct, governed) in STRATEGIES.items():
            bounded = np.clip(raw[strategy], -options["correction_clip_eur_mwh"], options["correction_clip_eur_mwh"])
            current[strategy+"_raw_correction"] = raw[strategy]
            current[strategy+"_bounded_correction"] = bounded
            current[strategy+"_selected_weight"] = 0.
            current[strategy+"_applied_correction"] = 0.
            current[strategy+"_gate_reason"] = "expert_warmup_or_unavailable"
            current[direct] = current.forecast.to_numpy(float)+bounded
            current[governed] = current.forecast.to_numpy(float)
        for zone, part in current.groupby("zone", sort=True):
            past = _eligible(oos, zone, day, cutoff, options["governance"]["lookback_days"])
            for strategy, (_, governed) in STRATEGIES.items():
                weight, reason, records = _govern(past, strategy, zone, day, cutoff, options["governance"])
                governance_records.extend(records)
                indices = part.index
                weights = np.where(part.expert_ready, weight, 0.)
                correction = weights*part[strategy+"_bounded_correction"].to_numpy(float)
                current.loc[indices, strategy+"_selected_weight"] = weights
                current.loc[indices, strategy+"_applied_correction"] = correction
                current.loc[indices, strategy+"_gate_reason"] = np.where(part.expert_ready, reason, "expert_warmup_or_unavailable")
                current.loc[indices, governed] = part.forecast.to_numpy(float)+correction
        blocks.append(current)
        if current.expert_ready.any():
            columns = ["zone", "_day", "_physical_day_hours", "_label_valid", "label_available_at_utc", "_error",
                       "expert_ready", "threshold_eur_mwh", *(name+"_bounded_correction" for name in STRATEGIES)]
            oos.append(current[columns].copy())
        first = (pd.Timestamp(day)-pd.Timedelta(days=options["governance"]["lookback_days"])).strftime("%Y-%m-%d")
        while oos and str(oos[0]._day.iloc[0]) < first:
            oos.popleft()
    if set(frozen_folds.index) != {fold["fit_day"] for fold in folds}:
        raise RMSEPolicyError("Unused frozen folds differ from the replay schedule.")
    result = pd.concat(blocks).sort_values("_row").reset_index(drop=True)
    additions = result.drop(columns=[name for name in result if name in panel or name.startswith("_")])
    additions.index = panel.index
    predictions = pd.concat([panel.copy(deep=True), additions], axis=1)
    candidates = [name for pair in STRATEGIES.values() for name in pair]
    if not np.isfinite(predictions[candidates].to_numpy(float)).all():
        raise RMSEPolicyError("Non-finite RMSE candidate forecast.")
    fitted = pd.DataFrame(folds)
    trained = fitted.loc[fitted.status.eq("trained")]
    audit = {"schema_version": 1, "engine": "nyx_rmse_v1", "diagnostic_only": True,
        "production_modified": False, "activation_performed": False, "orders_placed": False,
        "detector_retrained": False, "detector_probabilities_preserved_exactly": True,
        "storm_used_for_training_or_governance": False, "economic_pnl_used": False,
        "electricity_prices_or_nyx_used_as_model_inputs": False,
        "target": "signed_actual_minus_frozen_nyx", "point_functional": "conditional_residual_mean_then_signed_cap_and_governed_shrinkage",
        "candidate_forecasts_are_p50": False, "candidate_intervals_produced": False,
        "baseline_quantiles_preserved_as_baseline_only": True, "negative_corrections_allowed": True,
        "risk_or_positive_only_gate_used": False,
        "mean_of_existing_cdf_formula": "threshold*((1-p)*forest_normal.predict(X)+p*forest_spike.predict(X))",
        "forest_support_construction_required": False, "quantile_inversion_required": False,
        "training_window_cap_days": 365, "calibration_days_excluded_from_mean_fit": parameters["calibration_days"],
        "refit_every_days": parameters["refit_every_days"], "pooled_training_zones": list(zones),
        "trained_folds": len(trained), "fallback_folds": len(fitted)-len(trained),
        "cached_folds": int(fitted.cache_reused.sum()),
        "actual_training_days_min": int(trained.training_days.min()) if len(trained) else None,
        "actual_training_days_max": int(trained.training_days.max()) if len(trained) else None,
        "all_trained_folds_have_365_days": bool(len(trained) and trained.full_365_day_training.all()),
        "all_trained_folds_have_complete_365_eligible_history": bool(len(trained) and trained.complete_365_eligible_history.all()),
        "expert_ready_rows": int(predictions.expert_ready.sum()), "source_expert_ready_rows": int(predictions.source_expert_ready.sum()),
        "decision_rows": len(predictions), "changed_forecast_rows": {name: int(predictions[name].ne(predictions.forecast).sum()) for name in candidates},
        "governance_predictions_are_prequential_oos": True, "governance_is_daily_by_zone": True,
        "governance_requires_complete_fully_expert_ready_civil_days": True,
        "governance_objective": "maximum_positive_civil_day_clustered_lower_bound_of_pooled_hour_mse_gain",
        "ordinary_hour_guard": "abs(historical_NYX_error)<historical_ex_ante_training_threshold",
        "annual_non_regression_guaranteed": False, "features": feature_audit, "config": options,
        "limitations": ["Already examined exploratory year; no independent prospective validation.",
            "Progressive source warmup, not 365 training days before every evaluated day; strict mode falls back to NYX.",
            "Chronological calibration block remains excluded to match the frozen detector training core.",
            "Historical MAE guards do not guarantee future annual or ordinary-hour non-regression.",
            "A governed/clipped conditional-mean estimate is not P50; baseline intervals do not quantify its uncertainty.",
            "Source feature availability and label timestamps inherit the existing audit limitations.",
            "Forecast quality or MSE gain does not imply an executable trading profit."]}
    return {"predictions": predictions, "folds": fitted,
            "governance": pd.DataFrame(governance_records), "audit": audit}
