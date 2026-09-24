"""Fixed-model prospective inference with causal governance/calibration updates."""
from __future__ import annotations

import json
from pathlib import Path
import uuid
import joblib
import numpy as np
import pandas as pd

from nyx_scarcity import policy as base
from . import ledger
from .features import make_stress_features
from .intervals import fit_interval_state, apply_interval_state
from .policy import predict_model, validate_output
from .runner import read_suite, verify_result, code_seals, safe_path, NAMESPACE


def freeze(directory, *, root):
    directory, config, manifest = read_suite(directory, root=root)
    verify_result(directory, manifest)
    folds = pd.read_parquet(directory/"folds.parquet")
    if folds.empty or not {"fit_day", "status"}.issubset(folds):
        raise ValueError("A nonempty completed fold audit is required before freezing a prospective model.")
    last = folds.sort_values("fit_day").iloc[-1]
    if last.status != "trained":
        raise ValueError("Latest scheduled fit must be trained; cannot resurrect an earlier fallback model.")
    model = joblib.load(directory/"latest_model.joblib")
    if model["state"]["fit_day"] != last.fit_day:
        raise ValueError("Latest scheduled fit must be trained; cannot resurrect an earlier fallback model.")
    history = pd.read_parquet(directory/"physics_governed.parquet")
    through = history.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d").max()
    candidates = {"model": directory/"latest_model.joblib", "config": directory/"config.json",
        "suite_manifest": directory/"manifest.json", "source_audit": directory/"source_audit.json",
        **{"code:"+name: Path(root)/name for name in code_seals(Path(root))}}
    stamp = ledger.now_utc().strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    destination = safe_path(root, Path(root)/NAMESPACE/"prospective"/stamp)
    result = ledger.freeze_ledger(destination, root=root, candidate_files=candidates,
        history_files={"oos": directory/"physics_governed.parquet", "panel": directory/"panel.parquet"},
        explored_through_day=through, zones=sorted(history.zone.unique()), **config["prospective"])
    from nyx_scarcity.runner import _json
    _json(safe_path(root, Path(root)/NAMESPACE/"latest_ledger.json"), {"ledger": str(result)})
    return result


def resolve_ledger(value, *, root):
    if value:
        return safe_path(root, value)
    index = safe_path(root, Path(root)/NAMESPACE/"latest_ledger.json")
    return safe_path(root, json.loads(index.read_text(encoding="utf-8"))["ledger"])


def resolved_history(ledger_directory, *, root, cutoff):
    """Use only sealed previously issued rows and subsequently captured labels."""
    _, manifest, events = ledger._read(ledger_directory, root)
    cutoff = ledger._utc(cutoff)
    ledger._verify_files(root, manifest["history_files"])
    history = pd.read_parquet(manifest["history_files"]["oos"]["path"])
    # Do not expose future labels even to downstream helpers that themselves
    # enforce a cutoff. New prospective rows start strictly after this history.
    history = history.loc[history.forecast_origin_utc.lt(cutoff)].copy()
    unavailable = history.label_available_at_utc.isna() | history.label_available_at_utc.gt(cutoff)
    history.loc[unavailable, "actual"] = np.nan
    history.loc[unavailable, "label_available_at_utc"] = pd.NaT
    if "label_eligible" in history:
        history.loc[unavailable, "label_eligible"] = False
    blocks = [history]
    for event in events:
        if (event["kind"] != "forecast" or pd.Timestamp(event["forecast_origin_utc"]) >= cutoff
                or pd.Timestamp(event["issued_at_utc"]) > cutoff):
            continue
        frame = pd.read_parquet(event["artifacts"]["predictions"]["path"])
        frame["actual"] = np.nan
        frame["label_available_at_utc"] = pd.NaT
        frame["label_available_at_utc"] = pd.to_datetime(frame.label_available_at_utc, utc=True)
        frame["label_eligible"] = False
        for observation in events:
            if (observation["kind"] != "observation_resolution" or observation["delivery_day"] != event["delivery_day"]
                    or pd.Timestamp(observation["observations_received_at_utc"]) > cutoff):
                continue
            if (observation["forecast_event_sha256"] != event["event_sha256"]
                    or pd.Timestamp(observation["observations_received_at_utc"]) <= pd.Timestamp(event["issued_at_utc"])):
                raise ValueError("Observation resolution does not refer to an earlier matching frozen issue.")
            known = pd.read_parquet(observation["artifacts"]["observations"]["path"])
            if not known.zone.eq(observation["zone"]).all() or known.timestamp_utc.duplicated().any():
                raise ValueError("Resolved country/hour identity differs from its sealed event.")
            positions = frame.zone.eq(observation["zone"])
            if not positions.any() or set(known.timestamp_utc) != set(frame.loc[positions, "timestamp_utc"]):
                raise ValueError("Resolved observations must preserve the exact issued country/hour grid.")
            values = known.set_index("timestamp_utc").actual.reindex(frame.loc[positions, "timestamp_utc"])
            if values.isna().any():
                raise ValueError("A resolved day has incomplete sealed observations.")
            frame.loc[positions, "actual"] = values.to_numpy(float)
            frame.loc[positions, "label_available_at_utc"] = pd.Timestamp(observation["observations_received_at_utc"])
            frame.loc[positions, "label_eligible"] = True
        blocks.append(frame)
    combined = pd.concat(blocks, ignore_index=True)
    if combined.duplicated(["zone", "timestamp_utc"]).any():
        raise ValueError("Prospective history overlaps the explored replay or another issue.")
    return combined


def predict_fixed(panel, manifest, *, ledger_directory, root):
    """No refit; daily decisions use only previously issued eligible evidence."""
    if "actual" in panel and panel.actual.notna().any():
        raise ValueError("A future prediction cannot receive resolved target prices.")
    ledger._verify_files(root, manifest["candidate_files"])
    # A fresh Issue process must initialise and reverify the pinned runtime
    # before joblib imports the XGBoost class stored in the local artifact.
    source = Path(manifest["candidate_files"]["suite_manifest"]["path"]).parent
    source, _, suite_manifest = read_suite(source, root=root)
    verify_result(source, suite_manifest)
    if Path(manifest["candidate_files"]["model"]["path"]).resolve() != (source/"latest_model.joblib").resolve():
        raise ValueError("Frozen candidate model differs from its sealed source snapshot.")
    model = joblib.load(manifest["candidate_files"]["model"]["path"])
    state, p = model["state"], model["settings"]
    augmented, features, required, _ = make_stress_features(panel)
    if features != p["feature_columns"] or required != p["required_feature_columns"]:
        raise ValueError("Future physical feature contract differs from the frozen model.")
    augmented["actual"] = np.nan
    augmented["label_available_at_utc"] = pd.NaT
    current = base._prepare(augmented, p)
    if current.forecast_origin_utc.nunique() != 1 or current._day.nunique() != 1:
        raise ValueError("Prospective prediction accepts exactly one common delivery day and information cutoff.")
    cutoff = current.forecast_origin_utc.iloc[0]
    day = current._day.iloc[0]
    history = resolved_history(ledger_directory, root=root, cutoff=cutoff)
    ready = current._features_valid.to_numpy(bool)
    current["spike_probability"] = np.nan
    current["raw_correction"] = 0.
    current["threshold_eur_mwh"] = np.nan
    current["expert_ready"] = ready
    current["expert_fit_day"] = state["fit_day"]
    current["physical_gate_passed"] = False
    current["strong_risk_gate"] = False
    current["proposal_reason"] = "required_physical_inputs_unavailable"
    for q in (10, 50, 90):
        current[f"mixture_error_q{q}"] = np.nan
    if ready.any():
        probability, raw, threshold, detail = predict_model(state, current.loc[ready], p)
        current.loc[ready, "spike_probability"] = probability
        current.loc[ready, "raw_correction"] = raw
        current.loc[ready, "threshold_eur_mwh"] = threshold
        for col in detail.columns.difference(["zone", "timestamp_utc", "forecast_origin_utc"]):
            if col not in current:
                current[col] = (False if pd.api.types.is_bool_dtype(detail[col])
                                else np.nan if pd.api.types.is_numeric_dtype(detail[col]) else None)
            current.loc[ready, col] = detail[col].to_numpy()
    current["bounded_correction"] = np.clip(current.raw_correction, 0., p["correction_clip_eur_mwh"])
    current["mixture_raw_p50_eur_mwh"] = current.forecast+current.mixture_error_q50
    current["probability_gate"] = .5
    current["selected_weight"] = 0.
    current["gate_reason"] = current.proposal_reason
    # Reconstruct only the fixed-weight alternatives that were available when
    # each historical raw correction was issued; no new historical prediction.
    local = history.timestamp_utc.dt.tz_convert("Europe/Paris")
    history["_day"] = local.dt.strftime("%Y-%m-%d")
    days = local.dt.tz_localize(None).dt.normalize()
    history["_physical_day_hours"] = ((days+pd.Timedelta(days=1)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
        - days.dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")).dt.total_seconds()/3600.
    history["_label_valid"] = history.actual.notna() & history.label_eligible.fillna(False) if "label_eligible" in history else history.actual.notna()
    for weight in p["candidate_weights"]:
        history[base._column(weight)] = history.forecast+weight*history.bounded_correction
    for zone, part in current.groupby("zone"):
        past = base._eligible_oos([history], zone, day, cutoff, p)
        weight, reason, _ = base._govern(past, zone, day, cutoff, p)
        active = part.index[part.expert_ready & part.raw_correction.gt(0)]
        current.loc[active, "selected_weight"] = weight
        current.loc[active, "gate_reason"] = reason
    current["applied_correction"] = current.selected_weight*current.bounded_correction
    current["candidate_forecast"] = current.forecast+current.applied_correction
    current["intervention_active"] = current.applied_correction.gt(0)
    current["candidate_q10"], current["candidate_q90"] = current.q10, current.q90
    mask = current.intervention_active.to_numpy()
    weights = current.selected_weight.to_numpy(float)
    for q in (10, 90):
        baseline = current[f"q{q}"].to_numpy(float)
        expert = current.forecast.to_numpy(float)+np.minimum(current[f"mixture_error_q{q}"].to_numpy(float), p["correction_clip_eur_mwh"])
        candidate = (1-weights)*baseline+weights*expert
        bounded = np.minimum(candidate, baseline) if q == 10 else np.maximum(candidate, baseline)
        current.loc[mask, f"candidate_q{q}"] = bounded[mask]
    current["precalibration_q10"], current["precalibration_q90"] = current.candidate_q10, current.candidate_q90
    calibration = fit_interval_state(history, cutoff, zones=state["zones"])
    current = apply_interval_state(calibration, current)
    validate_output(current)
    current = current.sort_values("_row").drop(columns=[c for c in current if c.startswith("_")])
    # Preserve the input byte/value/dtype contract, including harmless all-NaN
    # labels if the caller supplied them. Newly manufactured labels are removed.
    current = current.drop(columns=[c for c in ("actual", "label_available_at_utc") if c not in panel], errors="ignore").reset_index(drop=True)
    return pd.concat([panel.reset_index(drop=True).copy(deep=True), current.drop(columns=list(panel.columns))], axis=1)


def issue(ledger_directory, *, root, panel, input_evidence):
    from .inputs import fresh_label_check
    return ledger.issue_forecast(ledger_directory, root=root, panel=panel, input_evidence=input_evidence,
        predictor=lambda frame, manifest: predict_fixed(frame, manifest, ledger_directory=ledger_directory, root=root),
        fresh_label_check=lambda day, zones: fresh_label_check(root=root, day=day, zones=zones))
