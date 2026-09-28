"""Prospective four-country CPU HGB/Test2/prior90 reference producer.

Weekly models see only earlier labels. Each delivery-day profile is scored
separately. The prior90 selector is then fitted on genuinely out-of-sample
predictions. This producer preserves uncertified upstream status.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .nyx_annual_cpu_baseline import (
    TEXT_HASH_POLICY, copy_file, frame_digest, sha256_text, write_frame, write_json,
)
from .nyx_annual_equal_ensemble import combine_equal_ensemble
from .nyx_annual_live_preflight import SOURCE_PROTOCOL, ZONES, delivery_grid, sha256, validate_source_receipt
from .nyx_annual_nyx_quantiles_gate import validate_nyx_quantiles_source
from .nyx_annual_prior90_cpu import FEATURES, apply_prior90_daily, select_prior90_policy
from .nyx_annual_reference_pair import PARTNER, SIGNALS, build_confirmed_pair
from .nyx_annual_test2_cpu import DAILY_PEAK_COLUMNS, PAIRS, fit_test2_origin
from .nyx_historical_hgb_reference import VARIANTS, archived_columns, fit_hgb_block, predict_saved_block
from .process_lock import exclusive_process_lock


PROTOCOL = "nyx_annual_cpu_reference_v1"
WEEKLY_ANCHOR = "2025-09-24"
ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "chronos2_hourly/nyx_annual_cpu_reference_builder.py", "chronos2_hourly/nyx_historical_hgb_reference.py",
    "chronos2_hourly/nyx_annual_test2_cpu.py", "chronos2_hourly/nyx_annual_prior90_cpu.py",
    "chronos2_hourly/nyx_annual_reference_pair.py", "chronos2_hourly/nyx_annual_equal_ensemble.py",
    "chronos2_hourly/solar_wind_scarcity_regime.py",
    "chronos2_hourly/nyx_annual_cpu_baseline.py", "chronos2_hourly/nyx_local_io.py",
)


def week_origin(day, *, anchor=WEEKLY_ANCHOR):
    day, start = pd.Timestamp(day).date(), pd.Timestamp(anchor).date()
    return day - timedelta(days=(day - start).days % 7)


def _grid(first, stop):
    return pd.date_range(str(first), str(stop), tz="Europe/Paris", freq="h", inclusive="left").tz_convert("UTC")


def _test2_frame(features, baseline):
    result = features.drop(columns=list(DAILY_PEAK_COLUMNS), errors="ignore").copy()
    result["baseline_p50"] = baseline["nyx__q50"].reindex(result.index)
    if not np.isfinite(result.to_numpy(float)).all():
        raise ValueError("Incomplete finite Test2 producer features")
    return result


def _upstream(root, day, require_verified):
    path = root / "source_receipts/nyx_quantiles.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in receipt.get("artifact_sha256", {}).items():
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError("Changed CPU baseline artifact")
    if not receipt.get("artifact_sha256"):
        raise ValueError("CPU baseline receipt must bind its curves")
    try:
        validate_nyx_quantiles_source(root, day)
        verified = True
    except ValueError:
        if require_verified:
            raise
        verified = False
    return sha256(path), verified, receipt.get("provider_publication_timestamp_verified") is True


def _cached_fit(root, relative, identity, fit):
    """Reuse only locally created, byte-verified models with identical inputs."""
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = root.parent / "_cpu_reference_cache" / key
    receipt_path = directory / "receipt.json"
    path = directory / "fitted.joblib"
    if receipt_path.exists():
        record = json.loads(receipt_path.read_text(encoding="utf-8"))
        if record.get("identity") != identity or not path.is_file() or sha256(path) != record.get("model_sha256"):
            raise ValueError("Changed CPU reference model cache")
        fitted, audit = joblib.load(path)
    else:
        fitted, audit = fit()
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump((fitted, audit), path)
        write_json(receipt_path, {"identity": identity, "model_sha256": sha256(path)})
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy_file(path, destination)
    copy_file(receipt_path, destination.with_suffix(".json"))
    return fitted, audit


def build_cpu_reference_bundle(bundle, delivery_day, *, baselines, base_features,
                               augmented_features, test2_features, target_snapshots_by_day,
                               require_verified_sources=True, progress=None):
    """Build all three HGBs and both Test2 pairs, with past-only weekly OOF.

    All feature matrices must cover the baseline's extended 462-day support.
    Origin-specific price snapshots are mandatory, including policy labels.
    Saved estimators are checksum-bound; external pickle files are never read.
    """
    root = Path(bundle).resolve()
    day = pd.Timestamp(delivery_day).date()
    origin = week_origin(day)
    first_origin = week_origin(origin - timedelta(days=90))
    first_training_day = first_origin - timedelta(days=365)
    expected = _grid(first_training_day, day + timedelta(days=1))
    required_maps = (baselines, base_features, augmented_features, test2_features)
    if any(set(mapping) != set(ZONES) for mapping in required_maps):
        raise ValueError("All four countries are required by the paired reference")
    upstream_hash, verified, publication_verified = _upstream(root, delivery_day, require_verified_sources)
    for zone in ZONES:
        for label, mapping in zip(("baseline", "HGB base", "HGB augmented", "Test2"), required_maps):
            frame = mapping[zone]
            if (not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC"
                    or not frame.index.is_unique or not frame.index.is_monotonic_increasing
                    or not expected.isin(frame.index).all()):
                raise ValueError(f"{zone}: incomplete {label} past training and prior90 OOF support")
        if tuple(base_features[zone].columns) != archived_columns(zone, "hist_residual_400"):
            raise ValueError(f"{zone}: HGB base schema differs")
        if tuple(augmented_features[zone].columns) != archived_columns(zone, "augmented_hist_residual_400"):
            raise ValueError(f"{zone}: augmented HGB schema differs")
    prepared_test2 = {z: _test2_frame(test2_features[z].loc[expected], baselines[z]) for z in ZONES}
    codes = {path: sha256_text(ROOT / path) for path in CODE_FILES}
    runtime = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "catboost", "joblib")}
    forecasts = {zone: [] for zone in ZONES}
    hashes, fits = {}, []
    with exclusive_process_lock(root / ".annual_reference.lock"):
        for fit_date in pd.date_range(str(first_origin), str(origin), freq="7D").date:
            train = _grid(fit_date - timedelta(days=365), fit_date)
            first_current = _grid(fit_date, fit_date + timedelta(days=1))
            available = train.append(first_current)
            snapshots = (target_snapshots_by_day(str(fit_date)) if callable(target_snapshots_by_day)
                         else target_snapshots_by_day[str(fit_date)])
            if set(snapshots) != set(ZONES):
                raise ValueError("Four origin-specific training label snapshots required")
            hgb, test2, records = {}, {}, []
            for zone in ZONES:
                labels = snapshots[zone].reindex(available)
                labels.loc[first_current] = np.nan
                point = baselines[zone]["nyx__q50"].reindex(available)
                hgb[zone] = {}
                for variant in VARIANTS:
                    matrix = augmented_features[zone] if VARIANTS[variant][0] == "augmented" else base_features[zone]
                    def fit_one_hgb():
                        fitted, _, audit = fit_hgb_block(
                            matrix.loc[available], labels, point, zone=zone, variant=variant,
                            origin_day=str(fit_date), stop_day=str(fit_date + timedelta(days=1)),
                            initial_training_day=str(fit_date - timedelta(days=365)))
                        return replace(fitted, stop_day_exclusive=str(fit_date + timedelta(days=7))), audit
                    relative = f"reference_models/{fit_date}/{zone}_{variant}.joblib"
                    identity = {"protocol": PROTOCOL, "zone": zone, "variant": variant,
                                "origin": str(fit_date), "features": frame_digest(matrix.loc[available]),
                                "training_labels": frame_digest(labels.loc[train]),
                                "nyx": frame_digest(point), "code": codes, "runtime": runtime,
                                "code_config_hash_policy": TEXT_HASH_POLICY}
                    fitted, audit = _cached_fit(root, relative, identity, fit_one_hgb)
                    hgb[zone][variant] = fitted
                    path = root / relative
                    hashes[relative] = sha256(path)
                    hashes[str(path.with_suffix(".json").relative_to(root)).replace("\\", "/")] = sha256(path.with_suffix(".json"))
                    records.append(audit)
            for pair in PAIRS:
                matrices = [prepared_test2[z].loc[train] for z in pair]
                labels = [snapshots[z].reindex(train).to_numpy(float)
                          - baselines[z].loc[train, "nyx__q50"].to_numpy(float) for z in pair]
                train_x, train_y = pd.concat(matrices), np.concatenate(labels)
                def fit_one_test2():
                    model = fit_test2_origin(train_x, train_y,
                                             list(train.tz_convert("Europe/Paris").date) * 2,
                                             pair=pair, origin_day=str(fit_date))
                    return model, model.audit
                relative = f"reference_models/{fit_date}/Test2_{'_'.join(pair)}.joblib"
                identity = {"protocol": PROTOCOL, "pair": list(pair), "variant": "test2_120",
                            "origin": str(fit_date), "features": frame_digest(train_x),
                            "training_residual": frame_digest(pd.Series(train_y)), "code": codes, "runtime": runtime,
                            "code_config_hash_policy": TEXT_HASH_POLICY}
                fitted, _ = _cached_fit(root, relative, identity, fit_one_test2)
                test2[pair] = fitted
                path = root / relative
                hashes[relative] = sha256(path)
                hashes[str(path.with_suffix(".json").relative_to(root)).replace("\\", "/")] = sha256(path.with_suffix(".json"))
                records.append(fitted.audit)
            # A later day's physical forecast profile never enters an earlier fit.
            for score_date in pd.date_range(str(fit_date), str(min(day, fit_date + timedelta(days=6))), freq="D").date:
                index = _grid(score_date, score_date + timedelta(days=1))
                source_cutoff = pd.Timestamp(f"{score_date - timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC")
                for zone in ZONES:
                    nyx = baselines[zone].loc[index, "nyx__q50"]
                    pair = next(pair for pair in PAIRS if zone in pair)
                    prediction = test2[pair].predict_day(prepared_test2[zone].loc[index], nyx,
                                                        zone=zone, forecast_issued_at_utc=source_cutoff)
                    points = {}
                    for variant in VARIANTS:
                        matrix = augmented_features[zone] if VARIANTS[variant][0] == "augmented" else base_features[zone]
                        points[variant] = predict_saved_block(hgb[zone][variant], matrix.loc[index], nyx).point
                    ensemble = combine_equal_ensemble(
                        nyx=nyx, test2=prediction["test2__q50"],
                        price_residual=points["hist_residual_400"], price_absolute=points["hist_absolute_400"],
                        augmented_residual=points["augmented_hist_residual_400"])
                    frame = pd.DataFrame({"ensemble__q50": ensemble, "nyx__q50": nyx,
                                          "test2__q50": prediction["test2__q50"],
                                          "own_joint_deficit": prepared_test2[zone].loc[index, "own_joint_deficit"],
                                          "own_residual_stress": prepared_test2[zone].loc[index, "own_residual_stress"],
                                          "nyx_daily_peak_gap": nyx.max() - nyx,
                                          "spike_probability": prediction.spike_probability}, index=index)
                    frame = frame.loc[:, list(FEATURES)]
                    forecasts[zone].append(frame)
            relative = f"reference_runs/{fit_date}.json"
            write_json(root / relative, {"protocol": PROTOCOL, "origin_day": str(fit_date),
                                        "trained_on_cpu": True, "future_labels_used": False,
                                        "training_price_sha256": {z: frame_digest(snapshots[z].reindex(train)) for z in ZONES},
                                        "model_audits": records, "implementation_sha256": codes,
                                        "code_config_hash_policy": TEXT_HASH_POLICY})
            hashes[relative] = sha256(root / relative)
            fits.append(relative)
            if progress:
                progress({"stage": "cpu_reference", "origin_day": str(fit_date), "last_origin": str(origin)})
        _, current, cutoff = delivery_grid(delivery_day)
        policy_index = _grid(origin - timedelta(days=90), origin)
        policy_snapshots = (target_snapshots_by_day(str(origin)) if callable(target_snapshots_by_day)
                            else target_snapshots_by_day[str(origin)])
        for zone in ZONES:
            oof = pd.concat(forecasts[zone]).sort_index()
            if not oof.index.is_unique:
                raise ValueError("Duplicate reference OOF prediction")
            policy = select_prior90_policy(oof.loc[policy_index], policy_snapshots[zone].reindex(policy_index),
                                           origin_day=str(origin), zone=zone)
            active = apply_prior90_daily(oof.loc[current], policy, forecast_issued_at_utc=cutoff)
            signals = oof.loc[current, ["ensemble__q50", "nyx__q50", "test2__q50", "spike_probability"]].copy()
            signals["prior90_active"] = active.prior90_active
            signals["partner_nyx__q50"] = baselines[PARTNER[zone]].loc[current, "nyx__q50"]
            result = build_confirmed_pair(signals.loc[:, list(SIGNALS)], zone=zone, partner_zone=PARTNER[zone])
            reference = pd.DataFrame({"reference": result.scarcity_confirmed_pair,
                                      "forecast_origin_utc": cutoff}, index=current)
            for relative, frame in {f"reference/{zone}.parquet": reference,
                                     f"reference_oof/{zone}.parquet": oof,
                                     f"reference_signals/{zone}.parquet": signals.loc[:, list(SIGNALS)]}.items():
                write_frame(root / relative, frame)
                hashes[relative] = sha256(root / relative)
            relative = f"reference_policies/{zone}.json"
            write_json(root / relative, policy)
            hashes[relative] = sha256(root / relative)
        input_hashes = {}
        for family, mapping in (("base292", base_features), ("augmented334", augmented_features), ("test2", test2_features)):
            for zone in ZONES:
                relative = f"reference_inputs/{family}/{zone}.parquet"
                # Bind exactly the in-memory matrix actually consumed by this fit.
                write_frame(root / relative, mapping[zone])
                input_hashes[relative] = sha256(root / relative)
        hashes.update(input_hashes)
        receipt = {"protocol": SOURCE_PROTOCOL, "source_group": "scarcity_confirmed_pair",
                   "delivery_day": delivery_day, "state": "COMPLETE" if verified else "UNQUALIFIED",
                   "asof_state_utc": cutoff.isoformat(), "asof_cutoff_verified": verified,
                   "publication_verified": publication_verified,
                   "provider_publication_timestamp_verified": publication_verified, "training_window_complete": True,
                   "artifact_sha256": hashes,
                   "producer": {"protocol": PROTOCOL, "device": "cpu", "zones": list(ZONES),
                                "baseline_receipt_sha256": upstream_hash, "implementation_sha256": codes,
                                "code_config_hash_policy": TEXT_HASH_POLICY,
                                "weekly_anchor": WEEKLY_ANCHOR, "first_oof_origin": str(first_origin),
                                "current_origin": str(origin), "training_days": 365, "prior_days": 90,
                                "future_labels_used": False, "storm_used_as_input": False,
                                "weekly_run_receipts": fits}}
        write_json(root / "source_receipts/scarcity_confirmed_pair.json", receipt)
        return receipt


def build_from_bundle(bundle, delivery_day, **kwargs):
    from .nyx_annual_saturn_source import load_target_snapshots
    root = Path(bundle).resolve()
    read = lambda family: {z: pd.read_parquet(root / f"reference_inputs/{family}/{z}.parquet") for z in ZONES}
    return build_cpu_reference_bundle(
        root, delivery_day, baselines={z: pd.read_parquet(root / f"baseline_history/{z}.parquet") for z in ZONES},
        base_features=read("base292"), augmented_features=read("augmented334"), test2_features=read("test2"),
        target_snapshots_by_day=load_target_snapshots(root), **kwargs)


def validate_cpu_reference_source(bundle, delivery_day):
    """Bind all fitted weekly models to their own source snapshots and policies."""
    from .nyx_annual_saturn_source import load_target_snapshots
    root = Path(bundle).resolve()
    path = root / "source_receipts/scarcity_confirmed_pair.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    _, current, cutoff = delivery_grid(delivery_day)
    validate_source_receipt(receipt, group="scarcity_confirmed_pair", day=delivery_day, bundle=root, cutoff=cutoff)
    upstream_hash, _, publication_verified = _upstream(root, delivery_day, True)
    day = pd.Timestamp(delivery_day).date()
    origin = week_origin(day)
    first_origin = week_origin(origin - timedelta(days=90))
    expected = _grid(first_origin - timedelta(days=365), day + timedelta(days=1))
    codes = {name: sha256_text(ROOT / name) for name in CODE_FILES}
    runtime = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "catboost", "joblib")}
    origins = [str(d) for d in pd.date_range(str(first_origin), str(origin), freq="7D").date]
    producer = receipt.get("producer", {})
    expected_producer = {"protocol": PROTOCOL, "device": "cpu", "zones": list(ZONES),
                         "baseline_receipt_sha256": upstream_hash, "implementation_sha256": codes,
                         "code_config_hash_policy": TEXT_HASH_POLICY,
                         "weekly_anchor": WEEKLY_ANCHOR, "first_oof_origin": str(first_origin),
                         "current_origin": str(origin), "training_days": 365, "prior_days": 90,
                         "future_labels_used": False, "storm_used_as_input": False,
                         "weekly_run_receipts": [f"reference_runs/{d}.json" for d in origins]}
    if producer != expected_producer:
        raise ValueError("CPU reference producer protocol, baseline, code or chronology differs")
    def bound_frame(relative):
        if relative not in receipt["artifact_sha256"]:
            raise ValueError(f"Unbound reference input or result: {relative}")
        return pd.read_parquet(root / relative)
    baselines = {z: pd.read_parquet(root / f"baseline_history/{z}.parquet") for z in ZONES}
    base = {z: bound_frame(f"reference_inputs/base292/{z}.parquet") for z in ZONES}
    augmented = {z: bound_frame(f"reference_inputs/augmented334/{z}.parquet") for z in ZONES}
    test2 = {z: _test2_frame(bound_frame(f"reference_inputs/test2/{z}.parquet").loc[expected], baselines[z]) for z in ZONES}
    load = load_target_snapshots(root)
    checked = 0
    def model_receipt(relative, identity):
        nonlocal checked
        sidecar = str(Path(relative).with_suffix(".json")).replace("\\", "/")
        if relative not in receipt["artifact_sha256"] or sidecar not in receipt["artifact_sha256"]:
            raise ValueError("Missing model and input-identity evidence")
        record = json.loads((root / sidecar).read_text())
        if record.get("identity") != identity or record.get("model_sha256") != receipt["artifact_sha256"][relative]:
            raise ValueError("CPU reference weekly model input snapshot, code or runtime differs")
        checked += 1
    for date in origins:
        fit_date = pd.Timestamp(date).date()
        train = _grid(fit_date - timedelta(days=365), fit_date)
        first_current = _grid(fit_date, fit_date + timedelta(days=1))
        available = train.append(first_current)
        snapshots = load(date)
        run_path = f"reference_runs/{date}.json"
        if run_path not in receipt["artifact_sha256"]:
            raise ValueError("Unbound CPU weekly fit audit")
        run = json.loads((root / run_path).read_text())
        if (run.get("protocol") != PROTOCOL or run.get("origin_day") != date
                or run.get("trained_on_cpu") is not True or run.get("future_labels_used") is not False
                or run.get("implementation_sha256") != codes or len(run.get("model_audits", [])) != 14
                or run.get("code_config_hash_policy") != TEXT_HASH_POLICY
                or run.get("training_price_sha256") != {z: frame_digest(snapshots[z].reindex(train)) for z in ZONES}):
            raise ValueError("Weekly CPU reference fit evidence changed")
        for zone in ZONES:
            point = baselines[zone]["nyx__q50"].reindex(available)
            labels = snapshots[zone].reindex(available)
            labels.loc[first_current] = np.nan
            for variant in VARIANTS:
                matrix = augmented[zone] if VARIANTS[variant][0] == "augmented" else base[zone]
                if tuple(matrix.columns) != archived_columns(zone, variant):
                    raise ValueError("Reference HGB trained feature schema differs")
                identity = {"protocol": PROTOCOL, "zone": zone, "variant": variant,
                            "origin": date, "features": frame_digest(matrix.loc[available]),
                            "training_labels": frame_digest(labels.loc[train]),
                            "nyx": frame_digest(point), "code": codes, "runtime": runtime,
                            "code_config_hash_policy": TEXT_HASH_POLICY}
                model_receipt(f"reference_models/{date}/{zone}_{variant}.joblib", identity)
        for pair in PAIRS:
            x = pd.concat([test2[z].loc[train] for z in pair])
            y = np.concatenate([snapshots[z].reindex(train).to_numpy(float)
                                - baselines[z].loc[train, "nyx__q50"].to_numpy(float) for z in pair])
            identity = {"protocol": PROTOCOL, "pair": list(pair), "variant": "test2_120",
                        "origin": date, "features": frame_digest(x),
                        "training_residual": frame_digest(pd.Series(y)), "code": codes, "runtime": runtime,
                        "code_config_hash_policy": TEXT_HASH_POLICY}
            model_receipt(f"reference_models/{date}/Test2_{'_'.join(pair)}.joblib", identity)
    policy_index = _grid(origin - timedelta(days=90), origin)
    labels = load(str(origin))
    for zone in ZONES:
        oof = bound_frame(f"reference_oof/{zone}.parquet")
        if not oof.index.equals(_grid(first_origin, day + timedelta(days=1))):
            raise ValueError("Prior90 OOF sequence incomplete")
        policy = select_prior90_policy(oof.loc[policy_index], labels[zone].reindex(policy_index),
                                       origin_day=str(origin), zone=zone)
        relative = f"reference_policies/{zone}.json"
        if relative not in receipt["artifact_sha256"] or json.loads((root / relative).read_text()) != policy:
            raise ValueError("Prior90 policy differs from the own-origin OOF and observed history")
        active = apply_prior90_daily(oof.loc[current], policy, forecast_issued_at_utc=cutoff)
        signals = bound_frame(f"reference_signals/{zone}.parquet")
        expected_signals = oof.loc[current, ["ensemble__q50", "nyx__q50", "test2__q50", "spike_probability"]].copy()
        expected_signals["prior90_active"] = active.prior90_active
        expected_signals["partner_nyx__q50"] = baselines[PARTNER[zone]].loc[current, "nyx__q50"]
        pd.testing.assert_frame_equal(signals, expected_signals.loc[:, list(SIGNALS)], check_freq=False)
        result = build_confirmed_pair(signals, zone=zone, partner_zone=PARTNER[zone])
        reference = bound_frame(f"reference/{zone}.parquet")
        if not reference.index.equals(current) or not np.array_equal(reference.reference.to_numpy(), result.scarcity_confirmed_pair.to_numpy()):
            raise ValueError("Published reference differs from the fitted prior90 paired formula")
    return {"protocol": PROTOCOL, "receipt_sha256": sha256(path), "weekly_origins_verified": len(origins),
            "cpu_model_artifacts_verified": checked, "prior90_recomputed": True,
            "provider_publication_timestamp_verified": publication_verified}


__all__ = ["build_cpu_reference_bundle", "build_from_bundle", "week_origin", "PROTOCOL", "validate_cpu_reference_source"]
