"""Two predeclared physical-severity ablations on a frozen causal detector.

The fuel ablation transports the historical tail. The network ablation also
conditions its CDF on original-hour initial-domain directional exposures.
Missing/unqualified network hours fall back to NYX, never to an interpolated
capacity or post-coupling publication. Production forecasts remain inputs.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd

from nyx_scarcity import policy as base
from nyx_fundamental_stress.features import make_fundamental_features, PREFIX
from nyx_fundamental_stress.policy import _check_features
from nyx_coherent_p50.policy import _source_rows, _finalize
from nyx_rmse.policy import _fit_core
from nyx_stress_guard.intervals import calibrate_intervals
from .models import fit_physical_cdf, predict_physical_cdf, validate_options


KEYS = ["zone", "timestamp_utc", "forecast_origin_utc"]
FUEL = PREFIX+"clean_gas_cost_ccgt_proxy_eur_mwh"
STRATEGIES = {
    "fuel_transport": ("fuel_transport_direct", "fuel_transport_governed"),
    "network_fuel": ("network_fuel_direct", "nyx_physical_p50"),
}
MODELS = tuple(name for names in STRATEGIES.values() for name in names)


def _network_contract(panel, network):
    if (not isinstance(network, pd.DataFrame) or network.columns.has_duplicates
            or not network.index.equals(panel.index) or "network_eligible" not in network):
        raise ValueError("Original-index network frame and explicit eligibility required.")
    if not network.network_eligible.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("Network eligibility must be nonmissing boolean.")
    names = [n for n in network if n.startswith("feature_network_")]
    if not names or any(any(s in n.lower() for s in ("shadow", "actual", "price", "final", "label")) for n in names):
        raise ValueError("An ex-ante directional network feature allowlist is required.")
    values = network[names].astype(float)
    if np.isinf(values.to_numpy()).any() or values.loc[~network.network_eligible].notna().any().any():
        raise ValueError("Unqualified network hours must have missing features; infinities forbidden.")
    if values.loc[network.network_eligible].isna().all(axis=1).any():
        raise ValueError("Eligible network rows must have real source features.")
    return names


def _fit_pair(data, day, cutoff, p, zones, frozen, features, network_names, options, load_fit, save_fit):
    record = {**frozen, "cache_reused": False, "detector_retrained": False}
    if pd.Timestamp(frozen["fit_cutoff_utc"]) != cutoff:
        raise ValueError("Frozen cutoff differs from D-1 08 h.")
    if frozen["status"] != "trained":
        return None, record
    core, thresholds = _fit_core(data, day, cutoff, p, zones, frozen)
    metadata = {"fit_day": day, "fit_cutoff": cutoff, "features": list(features),
        "network_features": list(network_names), "thresholds": thresholds, "zones": tuple(zones), "options": options,
        "source_core_rows": len(core)}
    state = load_fit(day) if load_fit is not None else None
    if state is not None:
        if any(state.get(key) != value for key, value in metadata.items()):
            raise ValueError("Checkpoint differs from the frozen physical-severity fold.")
        record = {**state["fold_record"], "cache_reused": True}
        return state, record
    models = {}
    positive_fuel = np.isfinite(core[FUEL]) & core[FUEL].gt(0)
    for strategy in STRATEGIES:
        usable = positive_fuel.copy()
        names = list(features)
        if strategy == "network_fuel":
            usable &= core.network_eligible
            names += list(network_names)
        selected = core.loc[usable]
        u = selected.zone.map(thresholds).to_numpy(float)
        errors = selected._error.to_numpy(float)
        normal_count, spike_count = int(np.sum(errors < u)), int(np.sum(errors >= u))
        enough = (len(selected) >= 500 and selected._day.nunique() >= 30
                  and normal_count >= 100 and spike_count >= p["minimum_tail_training_rows"]
                  and set(selected.zone) == set(zones))
        record.update({strategy+"_core_rows": len(selected), strategy+"_normal_rows": normal_count,
            strategy+"_spike_rows": spike_count, strategy+"_training_days": selected._day.nunique(),
            strategy+"_max_label_available_at_utc": selected.label_available_at_utc.max(),
            strategy+"_training_end_day": str(selected._day.max()) if len(selected) else None,
            strategy+"_status": "trained" if enough else "insufficient_qualified_core",
            strategy+"_excluded_rows": len(core)-len(selected)})
        models[strategy] = (fit_physical_cdf(base._matrix(selected, names, zones), errors, u,
            selected[FUEL].to_numpy(float), selected.zone.to_numpy(), threads=options["threads"]) if enough else None)
    state = {**metadata, "models": models, "fold_record": record}
    if save_fit is not None:
        save_fit(day, state)
    return state, record


def _predict(state, current, detector, strategy):
    if (not detector.expert_ready.eq(True).all()
            or not detector.expert_fit_day.eq(state["fit_day"]).all()
            or not current.forecast_origin_utc.ge(state["fit_cutoff"]).all()):
        raise ValueError("Detector and CDF inference do not share the causal fitted state.")
    p = detector.spike_probability.to_numpy(float)
    u = detector.threshold_eur_mwh.to_numpy(float)
    if not np.allclose(u, current.zone.map(state["thresholds"]), rtol=0, atol=1e-10):
        raise ValueError("Frozen event thresholds differ.")
    cdf = state["models"][strategy]
    fuel = current[FUEL].to_numpy(float)
    ready = np.isfinite(fuel) & (fuel > 0) & (cdf is not None)
    names = list(state["features"])
    if strategy == "network_fuel":
        ready &= current.network_eligible.to_numpy(bool)
        names += state["network_features"]
    physical = detector.physical_gate_passed.eq(True).to_numpy()
    risk = p > detector.risk_probability_gate.to_numpy(float)
    evaluated = ready & physical & risk
    # For missing physical evidence the final policy is exact identity, so
    # baseline error quantiles are placeholders, not fabricated expert CDFs.
    q = np.column_stack([current.q10-current.forecast, np.zeros(len(current)), current.q90-current.forecast])
    detail = current[KEYS].copy()
    detail["physical_expert_ready"] = ready
    detail["cdf_evaluated"] = evaluated
    detail["fuel_outside_core_support"] = False
    detail["fuel_ratio_to_core_max"] = np.nan
    detail["normal_effective_sample_size"] = np.nan
    detail["spike_effective_sample_size"] = np.nan
    if evaluated.any():
        selected = current.loc[evaluated]
        q[evaluated], diagnostics = predict_physical_cdf(cdf, base._matrix(selected, names, state["zones"]),
            selected.zone.to_numpy(), p[evaluated], u[evaluated], fuel[evaluated])
        for name, values in diagnostics.items():
            detail.loc[evaluated, name] = values
    positive = q[:, 1] > 0
    raw = np.where(ready & physical & risk & positive, q[:, 1], 0.)
    for i, level in enumerate((10, 50, 90)):
        detail[f"mixture_error_q{level}"] = q[:, i]
    detail["proposal_reason"] = np.select([~ready, ~physical, ~risk, ~positive],
        ["physical_cdf_or_pre_cutoff_inputs_unavailable", "physical_stress_gate_closed",
         "frozen_risk_gate_closed", "coherent_median_not_positive"], default="physical_cdf_p50_ready")
    return p, raw, u, detail


def _bands(out):
    columns = KEYS+["actual", "label_available_at_utc", "candidate_forecast", "candidate_q10", "candidate_q90"]
    columns += [n for n in ("label_eligible", "forecast_eligible") if n in out]
    thin = out[columns].copy()
    thin["intervention_active"] = out.candidate_forecast.ne(out.forecast)
    return calibrate_intervals(thin)


def run_replay(panel, source_predictions, source_folds, network_features, settings, options=None,
               *, load_fit=None, save_fit=None, progress=None):
    options = validate_options(options)
    pd.testing.assert_frame_equal(panel, source_predictions[panel.columns], check_exact=True)
    augmented, features, required, feature_audit = make_fundamental_features(panel, variant="fundamental")
    _check_features(features)
    pd.testing.assert_frame_equal(augmented[features], source_predictions[features], check_exact=True)
    network_names = _network_contract(panel, network_features)
    if set(network_features).intersection(augmented):
        raise ValueError("Network columns must not overwrite frozen panel/features.")
    augmented = pd.concat([augmented, network_features], axis=1)
    p = base._parameters({**settings, "threads": options["threads"],
        "feature_columns": features, "required_feature_columns": required})
    source = source_predictions.set_index(KEYS, drop=False)
    folds = source_folds.set_index("fit_day", drop=False)
    if source.index.has_duplicates or folds.index.has_duplicates:
        raise ValueError("Duplicate frozen detector identities/folds.")
    outputs, records, governors, audits, interval_audits = {}, [], [], {}, {}
    # A single fold checkpoint contains both ablations. On a replay with no
    # disk cache keep only the fitting pass states until the second pass ends.
    memory = {}
    def load(day):
        return load_fit(day) if load_fit is not None else memory.get(day)
    def save(day, state):
        if save_fit is not None:
            save_fit(day, state)
        else:
            memory[day] = state
    for strategy, model_names in STRATEGIES.items():
        details, seen = [], []
        def fit(data, day, cutoff, parameters, zones):
            if day not in folds.index:
                raise ValueError("Refit schedule differs from the frozen detector.")
            seen.append(day)
            if progress:
                progress(f"{strategy}: fold {day}; source={folds.loc[day, 'status']}")
            return _fit_pair(data, day, cutoff, parameters, zones, folds.loc[day].to_dict(),
                features, network_names, options, load, save)
        def predict(state, current, parameters):
            probability, raw, threshold, detail = _predict(state, current, _source_rows(source, current), strategy)
            details.append(detail)
            return probability, raw, threshold
        result = base.run_policy(augmented, p, fit_callback=fit, predict_callback=predict)
        if set(seen) != set(folds.index):
            raise ValueError("Unused frozen folds.")
        out = result.predictions.copy()
        if not out.expert_ready.equals(source_predictions.expert_ready):
            raise ValueError("Frozen detector readiness changed.")
        if not np.allclose(out.spike_probability, source_predictions.spike_probability, rtol=0, atol=0, equal_nan=True):
            raise ValueError("Frozen detector probability changed.")
        if details:
            detail = pd.concat(details, ignore_index=True).set_index(KEYS)
            if detail.index.has_duplicates:
                raise ValueError("Duplicate expert diagnostics.")
            for column in detail:
                out[column] = detail[column].reindex(pd.MultiIndex.from_frame(out[KEYS])).to_numpy()
        else:
            for level in (10, 50, 90):
                out[f"mixture_error_q{level}"] = np.nan
            out["physical_expert_ready"] = False
            out["proposal_reason"] = "source_warmup"
        result = replace(result, predictions=out)
        for direct, model_name in zip((True, False), model_names):
            final = _finalize(result, direct=direct)
            if progress:
                progress(f"{model_name}: calibration chronologique des intervalles")
            bands, interval_audit = _bands(final.predictions)
            outputs[model_name] = final.predictions.candidate_forecast
            outputs[model_name+"_q10"] = bands.candidate_q10
            outputs[model_name+"_q90"] = bands.candidate_q90
            outputs[model_name+"_interval_status"] = bands.interval_calibration_status
            outputs[model_name+"_precalibration_q10"] = bands.precalibration_q10
            outputs[model_name+"_precalibration_q90"] = bands.precalibration_q90
            interval_audits[model_name] = interval_audit
        for column in ("physical_expert_ready", "cdf_evaluated", "raw_correction", "bounded_correction", "selected_weight",
                       "proposal_reason", "mixture_error_q10", "mixture_error_q50", "mixture_error_q90",
                       "fuel_outside_core_support", "fuel_ratio_to_core_max", "normal_effective_sample_size",
                       "spike_effective_sample_size"):
            if column in out:
                outputs[strategy+"_"+column] = out[column]
        records.append(result.folds.assign(strategy=strategy))
        governors.append(result.governance.assign(strategy=strategy))
        audits[strategy] = {**final.audit,
            "conditional_distribution": "(1-p)*F_normal(error<u)+p*F_fuel_transported_excess(error>=u)",
            "zero_atom_assumption": False, "detector_retrained": False,
            "probability_calibration": "frozen_source_probabilities_no_recalibration",
            "interval_method": "monotone_quantile_function_interpolation_then_chronological_expansion",
            "source_readiness_is_not_physical_cdf_readiness": True,
            "physical_cdf_ready_rows": int(out.physical_expert_ready.eq(True).sum())}
    outputs["spike_probability"] = source_predictions.spike_probability
    outputs["expert_ready"] = source_predictions.expert_ready
    outputs["threshold_eur_mwh"] = source_predictions.threshold_eur_mwh
    for name in network_features:
        outputs[name] = network_features[name]
    predictions = pd.concat([panel.copy(deep=True), pd.DataFrame(outputs, index=panel.index)], axis=1)
    pd.testing.assert_frame_equal(predictions[panel.columns], panel, check_exact=True)
    for model_name in MODELS:
        q = predictions[[model_name+"_q10", model_name, model_name+"_q90"]].to_numpy(float)
        if not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
            raise ValueError("Invalid physical P50/interval output.")
    audit = {"schema_version": 1, "engine": "nyx_physical_p50_v1", "production_modified": False,
        "diagnostic_only": True, "activation_performed": False, "detector_retrained": False,
        "detector_probabilities_preserved_exactly": True, "candidate_forecasts_are_p50": True,
        "electricity_prices_or_nyx_used_as_model_inputs": False, "storm_used_for_training_or_governance": False,
        "primary_model_fixed_before_evaluation": "nyx_physical_p50", "features": feature_audit,
        "network_features": network_names, "network_missing_policy": "exact_NYX_fallback_no_filling",
        "cdf_inversion_optimization": "Only compute CDF on available physical/risk-gate-open rows; closed gates keep exact NYX.",
        "raw_cdf_diagnostic_scope": "Only *_cdf_evaluated rows contain actual expert quantiles; other rows contain identity placeholders.",
        "tail_transport": "u_current+CGC_current*(error_core-u_core)/CGC_core; normal CDF unchanged",
        "point_functional": "mixture_CDF_median_then_upward_gate_cap_and_monotone_quantile_function_interpolation",
        "network_quantities_are_feasible_imports": False, "post_coupling_features_used": False,
        "training_window_cap_days": 365, "policies": audits, "interval_calibration": interval_audits,
        "changed_forecast_rows": {name: int(predictions[name].ne(predictions.forecast).sum()) for name in MODELS},
        "annual_non_regression_guaranteed": False, "evaluation_year_already_examined": True,
        "limitations": [
            "Initial RefProg-balanced directional exposures are NOT a zero-based dispatch domain or import capacity.",
            "API lastModified before cutoff is historical research evidence, not independent real-time capture certification.",
            "Frozen detector probabilities may still miss events; this experiment tests severity, not a new probability detector.",
            "Fuel scaling is a falsifiable location/scale hypothesis, not a solution of unit commitment or scarcity pricing.",
            "Progressive source warmup lacks a complete 365-day training prefix for each evaluated day.",
            "Temperature gaps, daily partial-fleet availability and the NL component consistency issue remain unresolved.",
            "The same previously inspected year is exploratory, not independent prospective validation.",
            "Upward-only correction cannot fix overpredictions; no guaranteed future MAE or interval coverage improvement."]}
    return {"predictions": predictions, "folds": pd.concat(records, ignore_index=True),
            "governance": pd.concat(governors, ignore_index=True), "audit": audit}
