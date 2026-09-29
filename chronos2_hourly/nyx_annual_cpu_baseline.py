"""Daily CPU NYX producer with declared price vintages and resumable forecasts.

Chronos is the pinned pretrained model, followed by the existing interaction40
corrector and governed rolling Kalman. Each target day is scored without its
observed price. Provider qualification is inherited, never manufactured here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from chronos2_modular.common import CALENDAR_COLUMNS
from chronos2_modular.data import calendar_frame
from .chronos_adapter import build_delivery_plan
from .features import build_history_future_feature_matrix, validate_utc_hourly_index
from . import nuclear_forecast as residual
from . import kalman_residual as kalman
from .nyx_live_baseline import make_residual_factory
from .nyx_live_hybrid import build_pair_interaction
from .nyx_local_io import publish_bytes, publish_verified_immutable_copy, replace_retry
from .nyx_annual_live_preflight import SOURCE_PROTOCOL, ZONES, delivery_grid, sha256, validate_source_receipt
from .nyx_annual_nyx_quantiles_gate import CHRONOS_REVISION, EXPECTED_RECIPE, PRODUCER_PROTOCOL, validate_curve
from .process_lock import exclusive_process_lock
from .solar_wind_forecast import INPUT_ALIASES, solar_wind_kalman_covariate_config


ROOT = Path(__file__).resolve().parents[1]
RAW_ALIASES = (*INPUT_ALIASES, "fr_wind_generation_fcst", "be_wind_generation_fcst")
QUANTILES = ("q10", "q50", "q90")
NYX_QUANTILES = tuple(f"nyx__{q}" for q in QUANTILES)
TEXT_HASH_POLICY = "utf8_crlf_to_lf_v1"
CODE_FILES = (
    "chronos2_hourly/nyx_annual_cpu_baseline.py", "chronos2_hourly/nyx_live_baseline.py",
    "chronos2_hourly/nyx_live_hybrid.py", "chronos2_hourly/nuclear_forecast.py",
    "chronos2_hourly/kalman_residual.py", "chronos2_hourly/models/residual_corrector.py",
    "chronos2_hourly/solar_wind_interaction_features.py", "chronos2_hourly/solar_wind_scarcity_regime.py",
    "chronos2_hourly/kalman_covariates.py", "chronos2_hourly/features.py",
    "chronos2_hourly/nyx_local_io.py",
)


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode("utf-8")


def sha256_text(path: Path) -> str:
    """Hash code/config text across Git checkouts; preserve every other byte.

    This policy never applies to source data, receipts or fitted model files.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def chronos_identity_digest(model: dict) -> str:
    """Validate and fingerprint the pinned CPU weights/config inventory."""
    pinned = {"model_id": "amazon/chronos-2", "revision": CHRONOS_REVISION,
              "device": "cpu", "dtype": "torch.float32"}
    if not isinstance(model, dict) or set(model) != {*pinned, "files"} or any(model[k] != v for k, v in pinned.items()):
        raise ValueError("Pinned CPU Chronos model identity differs")
    files = model["files"]
    if not isinstance(files, dict) or "config.json" not in files or not any(
            isinstance(name, str) and name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("Chronos identity requires bound configuration and weights")
    for name, digest in files.items():
        if (not isinstance(name, str) or not name or PureWindowsPath(name).drive
                or PureWindowsPath(name).is_absolute() or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name.replace("\\", "/")).parts
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Invalid Chronos model artifact identity")
    return hashlib.sha256(_json(model)).hexdigest()


def frame_digest(frame) -> str:
    names = list(frame.columns) if isinstance(frame, pd.DataFrame) else [frame.name]
    return hashlib.sha256(_json(names) + pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def write_json(path: Path, value) -> None:
    publish_bytes(path, _json(value) + b"\n")


def copy_file(source: Path, destination: Path) -> None:
    publish_verified_immutable_copy(source, destination, sha256(source))


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        clean = frame.copy(deep=False)
        clean.attrs = {}
        clean.to_parquet(temporary)
        replace_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _grid(first, stop):
    if pd.Timestamp(first) > pd.Timestamp(stop):
        raise ValueError("CPU baseline dates reversed")
    if pd.Timestamp(first) == pd.Timestamp(stop):
        # pandas can include the endpoint when start==end despite left-only.
        # The first warmup day has no earlier fitted residual observations.
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(str(first), str(stop), tz="Europe/Paris", freq="h", inclusive="left").tz_convert("UTC")


def load_cpu_chronos(*, allow_download=True, cache_dir=None):
    """Install the exact snapshot if needed, and load it in CPU float32."""
    import torch
    from chronos import Chronos2Pipeline
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download("amazon/chronos-2", revision=CHRONOS_REVISION,
                                     cache_dir=cache_dir, local_files_only=not allow_download)).resolve()
    files = sorted(p for p in snapshot.rglob("*") if p.is_file()
                   and (p.suffix in (".safetensors", ".bin") or p.name.endswith("config.json")
                        or p.name.endswith(".index.json")))
    if not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError("The pinned Chronos snapshot has no model weights")
    identity = {"model_id": "amazon/chronos-2", "revision": CHRONOS_REVISION,
                "device": "cpu", "dtype": "torch.float32",
                "files": {str(p.relative_to(snapshot)): sha256(p) for p in files}}
    pipeline = Chronos2Pipeline.from_pretrained(str(snapshot), device_map="cpu", dtype=torch.float32,
                                               local_files_only=True)
    pipeline._nyx_annual_cpu_identity = identity
    return pipeline


def canonical_covariates(raw: pd.DataFrame, *, zone: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Names and calendar exactly match the annual twelve-channel recipe."""
    if zone not in ZONES or not set(INPUT_ALIASES).issubset(raw):
        raise ValueError("Four-country zone and all twelve Chronos inputs required")
    validate_utc_hourly_index(raw.index, name="annual raw covariates")
    core = raw.loc[:, list(INPUT_ALIASES)].astype(float)
    if np.isinf(core.to_numpy()).any():
        raise ValueError("Infinite annual source covariate")
    calendar = calendar_frame(raw.index.tz_convert(residual.ZONE_TIMEZONES[zone]))
    calendar.index = raw.index
    known = core.rename(columns=lambda name: f"known_{name}_oracle")
    future = pd.concat([known, calendar], axis=1)
    return pd.concat([core, future], axis=1), future


def predict_chronos_day(*, target: pd.Series, covariates: pd.DataFrame, zone: str,
                       delivery_day: str, pipeline, context_length=2048) -> pd.DataFrame:
    """A real future-day score: no current/future target is read or required."""
    plan = build_delivery_plan(delivery_day, timezone=residual.ZONE_TIMEZONES[zone])
    validate_utc_hourly_index(target.index, name="annual target")
    history = pd.date_range(end=plan.delivery_start_utc - pd.Timedelta(hours=1),
                            periods=context_length, freq="h", tz="UTC")
    labels = target.reindex(history).to_numpy(dtype=float)
    if not np.isfinite(labels).all():
        raise ValueError(f"{zone}/{delivery_day}: incomplete Chronos past target context")
    # The price context predates the first collected physical forecast profile
    # during bootstrap. Keep those physical channels missing, while calendar
    # channels remain known exactly as in the annual Chronos recipe.
    expanded = covariates.reindex(covariates.index.union(history))
    context, future = canonical_covariates(expanded, zone=zone)
    known = future.reindex(plan.delivery_index_utc)
    if not np.isfinite(known.to_numpy(dtype=float)).all():
        raise ValueError(f"{zone}/{delivery_day}: missing delivery forecast covariates")
    past = context.reindex(history)
    inputs = [{"target": labels.astype(np.float32),
               "past_covariates": {c: past[c].to_numpy(dtype=np.float32) for c in past},
               "future_covariates": {c: known[c].to_numpy(dtype=np.float32) for c in known}}]
    predictions, _ = pipeline.predict_quantiles(
        inputs, prediction_length=len(known), quantile_levels=[.1, .5, .9],
        batch_size=max(128, len(context.columns) + 1), context_length=context_length, cross_learning=False)
    values = predictions[0]
    if hasattr(values, "detach"):
        values = values.detach().cpu().float().numpy()
    values = np.asarray(values)
    if values.shape != (1, len(known), 3) or not np.isfinite(values).all() or (np.diff(values, axis=2) < 0).any():
        raise ValueError("Invalid native Chronos quantiles")
    output = pd.DataFrame(values[0].astype(np.float64), index=known.index, columns=QUANTILES)
    output["forecast_origin_utc"] = plan.forecast_origin_utc
    return output


def _configuration(zone, threads):
    suffix = "residual_v1" if zone == "FR" else "residual_candidate_v1"
    path = ROOT / f"chronos2_hourly_{zone.lower()}_{suffix}.yaml"
    configuration = {"hourly": deepcopy(yaml.safe_load(path.read_text(encoding="utf-8"))["hourly"])}
    recipe = configuration["hourly"]["residual_correction"]
    for key, expected in EXPECTED_RECIPE["residual"].items():
        if key in recipe and recipe[key] != expected:
            raise ValueError(f"Pinned CPU residual recipe changed: {key}")
    if recipe.get("backend") != "catboost" or recipe.get("iterations") != 700:
        raise ValueError("The annual interaction40 CPU corrector is required")
    recipe["thread_count"] = threads
    kalman_path = ROOT / "config/kalman_operational.yaml"
    kalman_configuration = yaml.safe_load(kalman_path.read_text(encoding="utf-8"))
    if kalman_configuration["training_lookback_days"] != 365:
        raise ValueError("Annual Kalman requires 365 days")
    configuration["nuclear_experiment"] = {"filter_parameters": kalman_configuration["filter_parameters"]}
    return configuration, {str(path.relative_to(ROOT)): sha256_text(path),
                           str(kalman_path.relative_to(ROOT)): sha256_text(kalman_path)}


def predict_residual_day(*, raw_history, raw_future, covariates, zone, delivery_day,
                         configuration, interaction):
    """Refit the existing CPU corrector on this origin's past price snapshot."""
    day = pd.Timestamp(delivery_day).date()
    factory = make_residual_factory(configuration, interaction)
    if raw_history.empty:
        result = raw_future.copy()
        for q in QUANTILES:
            result[f"chronos2__{q}"] = result[q]
            result[f"residual_corrected__{q}"] = result[q]
        result["residual_correction"] = 0.
        return result, {"delivery_day": str(day), "generation_source": "identity_chronos_cold_start",
                        "training_rows": 0, "future_labels_used": False}
    _, known = canonical_covariates(covariates, zone=zone)
    options = configuration["hourly"]["feature_engineering"]
    features = build_history_future_feature_matrix(
        raw_history.actual, known.reindex(raw_history.index), known.reindex(raw_future.index),
        price_lags=options.get("target_lags", (24, 48, 168)),
        rolling_windows=options.get("target_rolling_windows", (24, 168)), timezone=residual.ZONE_TIMEZONES[zone])
    from .models.residual_corrector import ResidualMetaFeatureBuilder
    prototype = factory()
    builder = prototype.feature_builder or ResidualMetaFeatureBuilder(**prototype.feature_builder_options)
    features = features.loc[:, [c for c in features if not builder._is_excluded(c)]]
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite retained residual features")
    _, forecast, audit = residual.causal_residual_replay(
        raw_history=raw_history, raw_future=raw_future, features=features,
        timezone=residual.ZONE_TIMEZONES[zone], delivery_day=day,
        residual_factory=factory, output_start_day=day)
    forecast = forecast.set_index("delivery_start_utc")
    return forecast, {**audit.iloc[-1].to_dict(), "future_labels_used": False}


def predict_kalman_day(*, history, future, covariates, zone, delivery_day, configuration):
    """Use the numerical rolling-day engine without requiring an observed target."""
    config, _ = residual._kalman_filter_configuration(configuration)
    cov_config = solar_wind_kalman_covariate_config()
    timezone = residual.ZONE_TIMEZONES[zone]
    day = pd.Timestamp(delivery_day).date()
    history_days = tuple(pd.Index(history.index.tz_convert(timezone).date).unique())
    if not 1 <= len(history_days) <= 365 or history_days[-1] != day - timedelta(days=1):
        raise ValueError("Kalman requires complete earlier civil days ending D-1")
    past, columns, _ = kalman._normalise_input(
        history.rename_axis("delivery_start_utc").reset_index(), upstream_model="residual_corrected",
        timezone=timezone, covariates=covariates.loc[history.index, list(INPUT_ALIASES)]
        .rename_axis("timestamp").reset_index(), covariate_config=cov_config)
    current = kalman._normalise_future_input(
        future.drop(columns="actual", errors="ignore").rename_axis("delivery_start_utc").reset_index(),
        upstream_model="residual_corrected", timezone=timezone,
        covariates=covariates.loc[future.index, list(INPUT_ALIASES)].rename_axis("timestamp").reset_index(),
        covariate_columns=columns, covariate_config=cov_config)
    candidate_columns = {kind: kalman._candidate_market_feature_columns(
        kind, covariate_config=cov_config, covariate_columns=columns) for kind in config.candidate_kinds}
    fitted = kalman._fit_rolling_target_day(
        training_frame=past, training_days=history_days, target_block=current, target_day=day,
        timezone=timezone, upstream_model="residual_corrected", output_model="nyx", config=config,
        covariate_columns=columns, candidate_feature_columns=candidate_columns)
    return fitted["predictions"].loc[:, list(NYX_QUANTILES)], fitted["daily_audit"]


def _sources(bundle: Path, day: str, require_verified: bool):
    _, _, cutoff = delivery_grid(day)
    records = {}
    verified = True
    publication_verified = True
    for group in ("saturn", "auction_prices"):
        path = bundle / f"source_receipts/{group}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        # Integrity is mandatory even for an explicitly unqualified CPU replay.
        for relative, expected in record.get("artifact_sha256", {}).items():
            artifact = (bundle / relative).resolve()
            if not artifact.is_relative_to(bundle) or not artifact.is_file() or sha256(artifact) != expected:
                raise ValueError(f"Changed or unsafe {group} artifact: {relative}")
        if not record.get("artifact_sha256"):
            raise ValueError(f"{group}: no bound source artifacts")
        try:
            validate_source_receipt(record, group=group, day=day, bundle=bundle, cutoff=cutoff)
        except ValueError:
            verified = False
            if require_verified:
                raise
        records[group] = sha256(path)
        publication_verified = publication_verified and record.get("provider_publication_timestamp_verified") is True
    return records, verified, publication_verified


def _advance_state(previous, files):
    """Bind all preceding numerical stages, including transitive Kalman inputs."""
    return hashlib.sha256(_json({"previous": previous, "files": files})).hexdigest()


INITIAL_STATE_SHA256 = hashlib.sha256(b"nyx-annual-cpu-state-v1").hexdigest()


def build_cpu_baseline_bundle(bundle, delivery_day, *, targets: Mapping[str, pd.Series],
                              covariates: pd.DataFrame,
                              target_snapshots_by_day: Callable | Mapping,
                              first_output_day=None, threads=2, pipeline=None,
                              allow_model_download=True, require_verified_sources=True,
                              checkpoint_cache=None, progress=None) -> dict:
    """Create the four baseline curves plus their CPU/source/checkpoint lineage.

    ``target_snapshots_by_day(day)`` returns four histories ending before that
    internal day. The Saturn receipt identifies either original daily vintages
    or reconstruction using prices available at this outer fit's cutoff.
    Reconstructed inner curves are training inputs, not historical forecasts
    certified at their inner origins. The public 365-day curves are
    accompanied by at least 469 days: 462 for weekly prior90 training and seven
    earlier baseline days for the lagged own-error feature columns.
    """
    bundle = Path(bundle).resolve()
    day = pd.Timestamp(delivery_day).date()
    output_first = pd.Timestamp(first_output_day).date() if first_output_day else day - timedelta(days=469)
    if output_first > day - timedelta(days=469) or not isinstance(threads, int) or threads < 1:
        raise ValueError("At least 469 output days and a positive CPU thread count required")
    # The source collector pins its first date. Keeping it fixed makes daily
    # append/resume numerically identical to the initial chronological replay.
    raw_first = covariates.index[0].tz_convert("Europe/Paris").date()
    if raw_first > output_first - timedelta(days=365):
        raise ValueError("At least 365 raw warmup days before the requested baseline required")
    first = raw_first + timedelta(days=365)
    needed = _grid(raw_first, day + timedelta(days=1))
    if set(targets) != set(ZONES) or not set(RAW_ALIASES).issubset(covariates):
        raise ValueError("Four targets and the fourteen raw annual covariates required")
    validate_utc_hourly_index(covariates.index, name="annual covariates")
    if not needed.isin(covariates.index).all() or not np.isfinite(covariates.loc[needed, list(RAW_ALIASES)].to_numpy(float)).all():
        raise ValueError("Missing raw forecast history for the CPU baseline warmup")
    source_hashes, verified, publication_verified = _sources(bundle, delivery_day, require_verified_sources)
    from .nyx_annual_saturn_source import target_history_contract
    target_contract = target_history_contract(bundle)
    # Numeric hashes already bind every price/covariate actually consumed.
    # Bind the policy too, while allowing identical computations to be reused
    # at a later outer cutoff; that cutoff is bound by this run's source graph.
    target_identity = ({"target_history_policy": target_contract["target_history_policy"]}
                       if target_contract else {})
    source_path = bundle / "source_artifacts/saturn/covariates.parquet"
    if not source_path.is_file() or frame_digest(pd.read_parquet(source_path)) != frame_digest(covariates):
        raise ValueError("In-memory covariates differ from the bound Saturn source")
    import torch
    torch.set_num_threads(threads)
    if pipeline is None:
        pipeline = load_cpu_chronos(allow_download=allow_model_download)
    model_identity = getattr(pipeline, "_nyx_annual_cpu_identity", None)
    model_hash = chronos_identity_digest(model_identity)
    runtime = {name: importlib.metadata.version(name) for name in
               ("torch", "chronos-forecasting", "numpy", "pandas", "catboost", "pykalman", "scipy", "holidays")}
    code = {relative: sha256_text(ROOT / relative) for relative in CODE_FILES}
    configurations = {z: _configuration(z, threads) for z in ZONES}
    # Interactions depend only on previously frozen forecast profiles.
    interactions = {z: build_pair_interaction(covariates.loc[needed], z, residual.ZONE_TIMEZONES[z])[0] for z in ZONES}
    raw_parts, corrected_parts, baseline_parts = ({z: [] for z in ZONES} for _ in range(3))
    prior_state = {z: INITIAL_STATE_SHA256 for z in ZONES}
    all_runs = {z: {stage: [] for stage in ("chronos", "residual", "kalman")} for z in ZONES}
    with exclusive_process_lock(bundle / ".annual_baseline.lock"):
        for date in pd.date_range(str(raw_first), str(day), freq="D").date:
            snapshots = (target_snapshots_by_day(str(date)) if callable(target_snapshots_by_day)
                         else target_snapshots_by_day[str(date)])
            if set(snapshots) != set(ZONES):
                raise ValueError(f"{date}: all four origin-specific price snapshots required")
            plan = build_delivery_plan(date, timezone="Europe/Paris")
            past_index = _grid(max(raw_first, date - timedelta(days=365)), date)
            for zone in ZONES:
                labels = snapshots[zone]
                validate_utc_hourly_index(labels.index, name=f"{zone}/{date} price snapshot")
                # Even if the supplier exposes target-day prices, they are discarded.
                labels = labels.loc[labels.index < plan.delivery_start_utc].copy()
                if len(past_index) and not np.isfinite(labels.reindex(past_index).to_numpy(float)).all():
                    raise ValueError(f"{zone}/{date}: incomplete as-of training prices")
                configuration, config_hash = configurations[zone]
                identity = {"protocol": PRODUCER_PROTOCOL, "zone": zone, "day": str(date),
                            **target_identity,
                            "prior_state_sha256": prior_state[zone],
                            "model_sha256": model_hash, "recipe": EXPECTED_RECIPE, "code": code,
                            "runtime": runtime, "config": config_hash, "cpu_threads": threads,
                            "code_config_hash_policy": TEXT_HASH_POLICY,
                            "prices_sha256": frame_digest(labels),
                            "covariates_sha256": frame_digest(covariates.loc[covariates.index <= plan.delivery_index_utc[-1]]),
                            "raw_first_day": str(raw_first)}
                root = bundle / "baseline_checkpoints" / zone / str(date)
                cache_base = Path(checkpoint_cache) if checkpoint_cache else bundle.parent / "_cpu_baseline_cache"
                cache_root = cache_base / zone / str(date) / hashlib.sha256(_json(identity)).hexdigest()
                receipt_path = root / "receipt.json"
                if not receipt_path.exists() and (cache_root / "receipt.json").exists():
                    saved = json.loads((cache_root / "receipt.json").read_text())
                    if saved.get("identity") != identity:
                        raise ValueError("Content-addressed baseline cache identity changed")
                    root.mkdir(parents=True, exist_ok=True)
                    for name, expected_digest in saved["files"].items():
                        if Path(name).name != name or sha256(cache_root / name) != expected_digest:
                            raise ValueError("Changed persistent baseline cache")
                        copy_file(cache_root / name, root / name)
                    copy_file(cache_root / "receipt.json", receipt_path)
                cached = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
                if cached is not None:
                    if cached.get("identity") != identity:
                        raise ValueError(f"{zone}/{date}: CPU checkpoint inputs or implementation changed")
                    for name, digest in cached["files"].items():
                        if Path(name).name != name or sha256(root / name) != digest:
                            raise ValueError("Changed baseline checkpoint")
                    raw = pd.read_parquet(root / "chronos.parquet")
                    corrected = pd.read_parquet(root / "residual.parquet")
                    result = pd.read_parquet(root / "baseline.parquet") if date >= first else None
                    audits = cached["audits"]
                else:
                    raw = predict_chronos_day(target=labels, covariates=covariates, zone=zone,
                                              delivery_day=str(date), pipeline=pipeline)
                    history = (pd.concat(raw_parts[zone]).loc[past_index].copy() if raw_parts[zone]
                               else raw.iloc[:0].copy())
                    history["actual"] = labels.reindex(history.index)
                    corrected, residual_audit = predict_residual_day(
                        raw_history=history, raw_future=raw, covariates=covariates, zone=zone,
                        delivery_day=str(date), configuration=configuration, interaction=interactions[zone])
                    audits = {"chronos": {"day": str(date), "context_length": 2048,
                                          "context_end_utc": (plan.delivery_start_utc - pd.Timedelta(hours=1)).isoformat(),
                                          "future_labels_used": False}, "residual": residual_audit}
                    result = None
                    if date >= first:
                        corrected_history = pd.concat(corrected_parts[zone]).loc[past_index].copy()
                        corrected_history["actual"] = labels.reindex(past_index)
                        result, audits["kalman"] = predict_kalman_day(
                            history=corrected_history, future=corrected, covariates=covariates,
                            zone=zone, delivery_day=str(date), configuration=configuration)
                        result["forecast_origin_utc"] = plan.forecast_origin_utc
                    frames = {"chronos.parquet": raw, "residual.parquet": corrected}
                    if result is not None:
                        frames["baseline.parquet"] = result
                    for name, frame in frames.items():
                        write_frame(root / name, frame)
                    cached = {"identity": identity, "audits": audits,
                              "files": {name: sha256(root / name) for name in frames}}
                    write_json(receipt_path, cached)
                    cache_root.mkdir(parents=True, exist_ok=True)
                    for name in cached["files"]:
                        copy_file(root / name, cache_root / name)
                    copy_file(receipt_path, cache_root / "receipt.json")
                prior_state[zone] = _advance_state(prior_state[zone], cached["files"])
                raw_parts[zone].append(raw)
                corrected_parts[zone].append(corrected)
                if result is not None:
                    baseline_parts[zone].append(result)
                relative = str(receipt_path.relative_to(bundle)).replace("\\", "/")
                for stage in audits:
                    all_runs[zone][stage].append({"day": str(date), "receipt": relative,
                                                "sha256": sha256(receipt_path)})
            if progress:
                progress({"stage": "cpu_baseline", "delivery_day": str(date), "last_day": str(day),
                          "completed_days": (date - raw_first).days + 1, "total_days": (day - raw_first).days + 1})
        full, current, cutoff = delivery_grid(delivery_day)
        artifact_hashes, zones = {}, {}
        for zone in ZONES:
            baseline = pd.concat(baseline_parts[zone]).sort_index().loc[_grid(output_first, day + timedelta(days=1))]
            baseline["actual"] = targets[zone].reindex(baseline.index)
            baseline.loc[current, "actual"] = np.nan
            if not np.isfinite(baseline.loc[baseline.index < current[0], "actual"].to_numpy(float)).all():
                raise ValueError("Missing final-cutoff historical labels in baseline output")
            curve = baseline.loc[full, [*NYX_QUANTILES, "actual", "forecast_origin_utc"]]
            validate_curve(curve, delivery_day=delivery_day, zone=zone)
            curve_path, audit_path = f"baseline/{zone}.parquet", f"baseline_audits/{zone}.json"
            frames = {curve_path: curve, f"baseline_history/{zone}.parquet": baseline,
                      f"sources/nyx_quantiles/{zone}.parquet": baseline.rename(columns={f"nyx__{q}": q for q in QUANTILES})}
            for relative, frame in frames.items():
                write_frame(bundle / relative, frame)
                artifact_hashes[relative] = sha256(bundle / relative)
            upstream = {}
            for stage in ("chronos", "residual", "kalman"):
                relative = f"baseline_runs/{zone}/{stage}.json"
                record = {"zone": zone, "delivery_day": delivery_day, "stage": stage, "device": "cpu",
                          **target_contract,
                          "source_asof_cutoff_utc": cutoff.isoformat(), "complete": True,
                          "recipe": EXPECTED_RECIPE[stage], "source_receipts_sha256": source_hashes,
                          "implementation_sha256": code, "runtime": runtime, "daily_runs": all_runs[zone][stage],
                          "code_config_hash_policy": TEXT_HASH_POLICY,
                          "cpu_threads": threads, "future_labels_used": False,
                          "provider_publication_verified": publication_verified}
                write_json(bundle / relative, record)
                artifact_hashes[relative] = sha256(bundle / relative)
                upstream[stage] = {"path": relative, "sha256": artifact_hashes[relative]}
                for checkpoint in all_runs[zone][stage]:
                    artifact_hashes[checkpoint["receipt"]] = checkpoint["sha256"]
                    cp = json.loads((bundle / checkpoint["receipt"]).read_text())
                    directory = Path(checkpoint["receipt"]).parent
                    for name, digest in cp["files"].items():
                        artifact_hashes[str(directory / name).replace("\\", "/")] = digest
            audit = {"protocol": PRODUCER_PROTOCOL, "zone": zone, "delivery_day": delivery_day,
                     **target_contract,
                     "recipe": EXPECTED_RECIPE, "source_asof_cutoff_utc": cutoff.isoformat(),
                     "cpu_retrained": True, "reused_archived_gpu_predictions": False,
                     "cpu_threads": threads,
                     "chronos_model_sha256": model_hash, "chronos_model_identity": model_identity,
                     "upstream_receipts": upstream, "provider_publication_verified": publication_verified,
                     "historical_label_policy": ("current-fit reconstruction; labels strictly before each internal origin"
                                                 if target_contract else "origin-specific supplier price snapshot"),
                     "first_output_day": str(output_first), "first_fitted_day": str(first),
                     "cold_start_before_first_fitted_day": 365}
            write_json(bundle / audit_path, audit)
            artifact_hashes[audit_path] = sha256(bundle / audit_path)
            zones[zone] = {"curve": curve_path, "audit": audit_path}
        receipt = {"protocol": SOURCE_PROTOCOL, "source_group": "nyx_quantiles", "delivery_day": delivery_day,
                   "state": "COMPLETE" if verified else "UNQUALIFIED", "asof_state_utc": cutoff.isoformat(),
                   "asof_cutoff_verified": verified, "publication_verified": publication_verified,
                   "provider_publication_timestamp_verified": publication_verified,
                   "training_window_complete": True, "artifact_sha256": artifact_hashes,
                   "source_receipts_sha256": source_hashes,
                   "producer": {"protocol": PRODUCER_PROTOCOL, "recipe": EXPECTED_RECIPE, "zones": zones,
                                **target_contract,
                                "code_config_hash_policy": TEXT_HASH_POLICY}}
        write_json(bundle / "source_receipts/nyx_quantiles.json", receipt)
        return receipt


def build_from_bundle(bundle, delivery_day, **kwargs):
    """Consume bound Saturn inputs under their declared price-history policy."""
    from .nyx_annual_saturn_source import load_target_snapshots
    root = Path(bundle).resolve()
    loader = load_target_snapshots(root)
    return build_cpu_baseline_bundle(
        root, delivery_day, targets=loader(delivery_day),
        covariates=pd.read_parquet(root / "source_artifacts/saturn/covariates.parquet"),
        target_snapshots_by_day=loader, **kwargs)


def validate_cpu_baseline_evidence(bundle, delivery_day):
    """Verify the daily source/CPU-run graph in addition to the public gate.

    This is reproducibility evidence, not an independent provider attestation.
    In particular a Saturn revision-date query does not become a certified
    original publication timestamp by passing this validator.
    """
    from .nyx_annual_nyx_quantiles_gate import validate_nyx_quantiles_source
    from .nyx_annual_saturn_source import load_target_snapshots, target_history_contract
    root = Path(bundle).resolve()
    summary = validate_nyx_quantiles_source(root, delivery_day)
    source_hashes, _, publication_verified = _sources(root, delivery_day, True)
    receipt = json.loads((root / "source_receipts/nyx_quantiles.json").read_text(encoding="utf-8"))
    target_contract = target_history_contract(root)
    target_identity = ({"target_history_policy": target_contract["target_history_policy"]}
                       if target_contract else {})
    contract_keys = ("target_history_policy", "target_revision_utc",
                     "target_origin_snapshot_verified", "target_future_labels_used")
    def contract_matches(record):
        return {key: record[key] for key in contract_keys if key in record} == target_contract
    if (receipt.get("source_receipts_sha256") != source_hashes
            or not contract_matches(receipt.get("producer", {}))
            or receipt.get("producer", {}).get("code_config_hash_policy") != TEXT_HASH_POLICY):
        raise ValueError("CPU baseline source receipt graph changed")
    codes = {relative: sha256_text(ROOT / relative) for relative in CODE_FILES}
    runtime = {name: importlib.metadata.version(name) for name in
               ("torch", "chronos-forecasting", "numpy", "pandas", "catboost", "pykalman", "scipy", "holidays")}
    covariates = pd.read_parquet(root / "source_artifacts/saturn/covariates.parquet")
    first_raw = covariates.index[0].tz_convert("Europe/Paris").date()
    day = pd.Timestamp(delivery_day).date()
    first = first_raw + timedelta(days=365)
    if first > day - timedelta(days=469):
        raise ValueError("CPU baseline reference warmup is insufficient")
    load = load_target_snapshots(root)
    all_days = [str(d) for d in pd.date_range(str(first_raw), str(day), freq="D").date]
    baseline_days = [str(d) for d in pd.date_range(str(first), str(day), freq="D").date]
    runs = {}
    shared_model, shared_model_hash = None, None
    for zone in ZONES:
        audit = json.loads((root / f"baseline_audits/{zone}.json").read_text())
        model = audit.get("chronos_model_identity", {})
        model_hash = chronos_identity_digest(model)
        if shared_model is None:
            shared_model, shared_model_hash = model, model_hash
        elif model != shared_model or model_hash != shared_model_hash:
            raise ValueError("The four CPU baselines must use identical pinned Chronos weights")
        if (audit.get("chronos_model_sha256") != model_hash
                or not contract_matches(audit)
                or audit.get("historical_label_policy") != (
                    "current-fit reconstruction; labels strictly before each internal origin"
                    if target_contract else "origin-specific supplier price snapshot")
                or audit.get("first_fitted_day") != str(first)
                or not first <= pd.Timestamp(audit.get("first_output_day")).date() <= day - timedelta(days=469)
                or type(audit.get("cpu_threads")) is not int or audit["cpu_threads"] < 1):
            raise ValueError(f"{zone}: pinned CPU Chronos identity differs")
        for stage in ("chronos", "residual", "kalman"):
            run = json.loads((root / f"baseline_runs/{zone}/{stage}.json").read_text())
            entries = run.get("daily_runs", [])
            if (run.get("implementation_sha256") != codes or run.get("runtime") != runtime
                    or not contract_matches(run)
                    or run.get("code_config_hash_policy") != TEXT_HASH_POLICY
                    or run.get("source_receipts_sha256") != source_hashes
                    or run.get("cpu_threads") != audit["cpu_threads"]
                    or run.get("recipe") != EXPECTED_RECIPE[stage]
                    or run.get("future_labels_used") is not False
                    or [entry.get("day") for entry in entries] != (baseline_days if stage == "kalman" else all_days)):
                raise ValueError(f"{zone}: {stage} daily CPU run evidence differs")
            runs[(zone, stage)] = entries
    checked = 0
    assembled = {z: [] for z in ZONES}
    prior_state = {z: INITIAL_STATE_SHA256 for z in ZONES}
    for date in all_days:
        plan = build_delivery_plan(date, timezone="Europe/Paris")
        snapshots = load(date)
        expected_covariates = frame_digest(covariates.loc[covariates.index <= plan.delivery_index_utc[-1]])
        for zone in ZONES:
            relative = f"baseline_checkpoints/{zone}/{date}/receipt.json"
            item = json.loads((root / relative).read_text())
            identity = item.get("identity", {})
            audit = json.loads((root / f"baseline_audits/{zone}.json").read_text())
            _, config_hash = _configuration(zone, 1)  # Thread count is not a numerical config-file change.
            labels = snapshots[zone].loc[snapshots[zone].index < plan.delivery_start_utc]
            expected_identity = {"protocol": PRODUCER_PROTOCOL, "zone": zone, "day": date,
                                 **target_identity,
                                 "prior_state_sha256": prior_state[zone],
                                 "model_sha256": audit["chronos_model_sha256"], "recipe": EXPECTED_RECIPE,
                                 "code": codes, "runtime": runtime, "config": config_hash,
                                 "cpu_threads": audit["cpu_threads"],
                                 "code_config_hash_policy": TEXT_HASH_POLICY,
                                 "prices_sha256": frame_digest(labels), "covariates_sha256": expected_covariates,
                                 "raw_first_day": str(first_raw)}
            if identity != expected_identity:
                raise ValueError(f"{zone}/{date}: source snapshot or CPU checkpoint identity differs")
            required = {"chronos.parquet", "residual.parquet"}
            if pd.Timestamp(date).date() >= first:
                required.add("baseline.parquet")
            if set(item.get("files", {})) != required:
                raise ValueError("CPU daily checkpoint stage inventory differs")
            for name, digest in item["files"].items():
                artifact = str(Path(relative).parent / name).replace("\\", "/")
                if receipt["artifact_sha256"].get(artifact) != digest:
                    raise ValueError("Unbound CPU daily checkpoint payload")
                frame = pd.read_parquet(root / artifact)
                if "actual" in frame or not frame.index.equals(plan.delivery_index_utc):
                    raise ValueError("CPU daily prediction must be label-free on the exact target grid")
                names = NYX_QUANTILES if name == "baseline.parquet" else QUANTILES
                values = frame.loc[:, list(names)].to_numpy(float)
                if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
                    raise ValueError("Invalid checkpoint quantiles")
                if not pd.to_datetime(frame.forecast_origin_utc, utc=True).eq(plan.forecast_origin_utc).all():
                    raise ValueError("Checkpoint daily cutoff changed")
                if name == "baseline.parquet":
                    assembled[zone].append(frame)
            if (item["audits"]["chronos"].get("future_labels_used") is not False
                    or item["audits"]["residual"].get("future_labels_used") is not False):
                raise ValueError("Missing label-free daily fitting evidence")
            if "kalman" in item["audits"] and (item["audits"]["kalman"].get("training_window_days") != 365
                    or item["audits"]["kalman"].get("target_observations_assimilated") != 0):
                raise ValueError("Kalman daily 365-day fitting evidence differs")
            for stage in item["audits"]:
                match = next((x for x in runs[(zone, stage)] if x["day"] == date), None)
                if match != {"day": date, "receipt": relative, "sha256": sha256(root / relative)}:
                    raise ValueError("CPU run receipt does not bind its daily evidence")
            prior_state[zone] = _advance_state(prior_state[zone], item["files"])
            checked += 1
    final_labels = load(delivery_day)
    full, current, _ = delivery_grid(delivery_day)
    for zone in ZONES:
        audit = json.loads((root / f"baseline_audits/{zone}.json").read_text())
        index = _grid(pd.Timestamp(audit["first_output_day"]).date(), day + timedelta(days=1))
        expected = pd.concat(assembled[zone]).loc[index].copy()
        expected["actual"] = final_labels[zone].reindex(index)
        expected.loc[current, "actual"] = np.nan
        for relative, wanted in (
            (f"baseline_history/{zone}.parquet", expected),
            (f"baseline/{zone}.parquet", expected.loc[full, [*NYX_QUANTILES, "actual", "forecast_origin_utc"]]),
            (f"sources/nyx_quantiles/{zone}.parquet", expected.rename(columns={f"nyx__{q}": q for q in QUANTILES})),
        ):
            if relative not in receipt["artifact_sha256"]:
                raise ValueError("Unbound assembled CPU baseline curve")
            pd.testing.assert_frame_equal(pd.read_parquet(root / relative), wanted, check_freq=False)
    return {**summary, "cpu_daily_checkpoints_verified": checked,
            **target_contract,
            "source_snapshots_bound": True, "runtime_verified": True,
            "chronos_model_identity": shared_model, "chronos_model_sha256": shared_model_hash,
            "provider_publication_timestamp_verified": publication_verified}


__all__ = ["RAW_ALIASES", "load_cpu_chronos", "predict_chronos_day", "predict_residual_day",
           "predict_kalman_day", "build_cpu_baseline_bundle", "build_from_bundle",
           "validate_cpu_baseline_evidence", "chronos_identity_digest"]
