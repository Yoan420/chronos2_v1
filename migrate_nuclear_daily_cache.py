"""Explicit, verified import of completed legacy nuclear runs into daily caches.

Never edits a frozen source or a model implementation. Import is staged and
replayed with fitting disabled before publication. A real residual fit and
Chronos inference validate the legacy numerical recipe on its delivery day.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta, datetime
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch

# Validation uses the already pinned local checkpoint; never query the hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly import nuclear_forecast as engine
from chronos2_hourly import kalman_residual as kalman
from chronos2_hourly.chronos_adapter import generate_delivery_plans, run_existing_live_forecast
from chronos2_hourly.nuclear_daily_cache import NuclearDailyChronosCache
from chronos2_hourly.nuclear_incremental import prepare_incremental_settings
from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
from chronos2_hourly.nuclear_residual_cache import ResidualDayCache
from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
from chronos2_modular.common import build_zone_configs, set_reproducibility
from run_chronos2_hourly import _feature_inputs, _residual_corrector_factory


LEGACY_ENGINE = "bda946d3cd5fbfcf14ad3c06b93071aa7e8ae1535a717f5f9c9e1a8829e91db9"
Q = ["q10", "q50", "q90"]


def indexed(frame):
    return frame.set_index("delivery_start_utc") if "delivery_start_utc" in frame else frame.copy()


def canonical_raw(result):
    """Use the exact float32 values consumed by the archived corrector.

    Some legacy raw_history bundles were refreshed from a decimal CSV, while
    their original exact neural outputs remained in residual_statistics.
    """
    stats, future = indexed(result.residual_statistics), indexed(result.source_forecast)
    raw = stats[["chronos2__" + q for q in Q] + ["actual", "forecast_origin_utc"]].rename(
        columns={"chronos2__" + q: q for q in Q})
    raw_future = future[["chronos2__" + q for q in Q] + ["forecast_origin_utc"]].rename(
        columns={"chronos2__" + q: q for q in Q})
    for q in Q + ["actual"]:
        if raw[q].dtype != np.dtype("float32"):
            raise ValueError("Exact original float32 training inputs are required")
        np.testing.assert_array_equal(result.raw_history[q].astype("float32"), raw[q])
    pd.testing.assert_index_equal(result.raw_history.index, raw.index)
    return raw, raw_future


def fail_fit(*args, **kwargs):
    raise AssertionError("Cache verification attempted to fit a model")


def verify_kalman(result, config, directory):
    filter_config, _ = engine._kalman_filter_configuration(config)
    with patch.object(kalman, "_fit_rolling_target_day", fail_fit), \
         patch.object(kalman, "_fit_rolling_target_day_chunk", fail_fit), \
         patch.object(kalman, "_write_cached_rolling_fit", fail_fit):
        view = kalman.build_operational_kalman_view(
            statistics=result.residual_statistics.copy(deep=True),
            source_forecast=result.source_forecast.copy(deep=True),
            covariates=result.covariates.copy(deep=True),
            timezone=engine.ZONE_TIMEZONES[result.audit["zone"]],
            delivery_day=result.audit["delivery_day"], config=filter_config,
            covariate_config=engine.nuclear_kalman_covariate_config(),
            upstream_model="residual_corrected", output_model="residual_kalman",
            training_lookback_days=365, rolling_refit_workers=1,
            rolling_refit_cache_dir=directory)
    return view.replay.audit["rolling_refit_cache"]


def publish_epoch(staged: Path, destination: Path, backup_root: Path):
    """Publish verified files, preserving any occupied incompatible epoch."""
    record = json.loads((staged / "epoch.json").read_text())
    if destination.exists():
        existing = json.loads((destination / "epoch.json").read_text())
        if existing != record:
            if list(destination.rglob("*")) != [destination / "epoch.json"]:
                raise ValueError("An occupied epoch has a different anchor; cannot replace it")
            backup_root.mkdir(parents=True, exist_ok=True)
            backup = backup_root / (destination.name + "_empty_epoch")
            destination.rename(backup)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        staged.rename(destination)
    else:
        # Repeating a successful import is allowed only for byte-identical files.
        for source in staged.rglob("*"):
            if not source.is_file():
                continue
            target = destination / source.relative_to(staged)
            if target.exists() and engine._file_sha256(target) != engine._file_sha256(source):
                raise ValueError(f"Existing cache entry differs: {target}")
        for source in staged.rglob("*"):
            if not source.is_file():
                continue
            target = destination / source.relative_to(staged)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # A hard link publishes a complete same-volume file without ever
            # truncating an entry another process may already be reading.
            try:
                os.link(source, target)
            except FileExistsError:
                if engine._file_sha256(target) != engine._file_sha256(source):
                    raise ValueError(f"Concurrent cache entry differs: {target}")


def migrate(source: Path, cache_root: Path, work: Path, *, device="auto", threads=4):
    source, cache_root, work = source.resolve(), cache_root.resolve(), work.resolve()
    project = Path(__file__).resolve().parent
    if any(path == project or not path.is_relative_to(project)
           for path in (source, cache_root, work)):
        raise ValueError("Migration paths must remain inside the project workspace")
    result = load_nuclear_result_bundle(workdir=source)
    if result.audit["source_hashes"]["engine_file_sha256"] != LEGACY_ENGINE:
        raise ValueError("This importer only supports the explicitly reviewed legacy engine")
    config = yaml.safe_load((source / "resolved_config.yaml").read_text(encoding="utf-8"))
    zone, day = result.audit["zone"], pd.Timestamp(result.audit["delivery_day"]).date()
    timezone = engine.ZONE_TIMEZONES[zone]
    work.mkdir(parents=True, exist_ok=False)
    spec = build_zone_configs(config, [zone], None, None)[0]
    data = prepare_nuclear_zone_data(spec, config, source, work / "prepared")
    target, _, _, features = _feature_inputs(data, config)
    hashes = {"target": engine._digest_frame(target),
              "model_context_covariates": engine._digest_frame(data.model_context_covariates),
              "covariates": engine._digest_frame(data.covariates),
              "residual_features": engine._digest_frame(features),
              "known_future_columns": engine._digest_json(list(data.known_future_columns))}
    for key, digest in hashes.items():
        if digest != result.audit["source_hashes"][key]:
            raise ValueError(f"Reconstructed source inputs differ: {key}")
    options = config["hourly"]["residual_correction"]
    options["thread_count"] = threads
    if options != result.audit["residual_recipe"]:
        raise ValueError("Archived residual recipe differs from import recipe")
    raw, future = canonical_raw(result)
    np.testing.assert_array_equal(target.reindex(raw.index).to_numpy(dtype="float32"), raw.actual)
    factory, base_model = _residual_corrector_factory(config, timezone=timezone)
    if base_model != "chronos2" or factory is None:
        raise ValueError("Unexpected residual factory")
    plans = generate_delivery_plans(day - timedelta(days=730), day,
                                    timezone=timezone, forecast_origin_local_time="08:00")
    set_reproducibility(config["model"]["seed"])
    import torch
    torch.set_num_threads(threads)
    from chronos2_modular.forecasting import load_model
    print(f"[{zone}] source verified; checking one real Chronos/residual delivery", flush=True)
    runtime = load_model(config, device, True)
    fresh = run_existing_live_forecast(plans[-1], data=data, runtime=runtime,
        context_length=config["model"]["context_length"],
        model_batch_size=config["model"]["model_batch_size"], with_covariates=True,
        variant="nuclear_legacy_migration_validation")
    np.testing.assert_array_equal(fresh[Q], future[Q])
    del runtime
    training = raw.index[raw.index.tz_convert(timezone).date >= day - timedelta(days=365)]
    fitted = factory()
    fitted.fit(features.loc[training], raw.actual.loc[training], raw.loc[training, Q],
               raw.loc[training, Q].rename(columns={q: "chronos2__" + q for q in Q}))
    fresh_residual = fitted.predict(features.loc[future.index], future[Q],
                       future[Q].rename(columns={q: "chronos2__" + q for q in Q}))
    saved_future = indexed(result.source_forecast)
    np.testing.assert_array_equal(fresh_residual[Q], saved_future[["residual_corrected__" + q for q in Q]])
    staged_config = deepcopy(config)
    staged_config.setdefault("nuclear_experiment", {}).update(
        mode="incremental", incremental_cache_dir=str(work / "cache"))
    namespace, anchor, _ = prepare_incremental_settings(staged_config, day)
    neural = NuclearDailyChronosCache(namespace / "chronos", data=data, config=config,
        context_length=config["model"]["context_length"], device=device,
        model_batch_size=config["model"]["model_batch_size"],
        origin_batch_size=config["model"]["origin_batch_size"], execution_signature={"threads": threads})
    prototype = factory()
    from chronos2_hourly.models.residual_corrector import ResidualMetaFeatureBuilder
    builder = getattr(prototype, "feature_builder", None) or ResidualMetaFeatureBuilder(
        **getattr(prototype, "feature_builder_options", {}))
    selected = features.loc[:, [c for c in features if not builder._is_excluded(c)]]
    residual = ResidualDayCache(namespace / "residual", {
        "recipe": options, "timezone": timezone, "factory": engine._factory_cache_identity(None),
        "engine_sha256": engine._file_sha256(Path(engine.__file__)),
        "input_schema": [(str(c), str(t)) for c, t in features.dtypes.items()],
    }, features=selected, raw=raw, timezone=timezone)
    records = result.residual_daily_audit.set_index("delivery_day")
    stats = indexed(result.residual_statistics)
    print(f"[{zone}] exact numerical checks passed; staging daily entries", flush=True)
    for plan in plans:
        is_future = plan.delivery_date == day
        base = future if is_future else raw.loc[plan.delivery_index_utc]
        neural.store(plan, base)
        restored = neural.load(plan, historical=not is_future)
        if restored is None:
            raise ValueError("Chronos entry could not be reread")
        np.testing.assert_array_equal(restored[Q], base[Q])
        audit = records.loc[plan.delivery_date.isoformat()]
        if audit.generation_source != "daily_prequential_refit":
            continue
        days = raw.index.tz_convert(timezone).date
        train = raw.index[(days >= plan.delivery_date - timedelta(days=365)) & (days < plan.delivery_date)]
        if len(train) != audit.training_rows or str(train[0].tz_convert(timezone).date()) != audit.fit_start_day:
            raise ValueError("Archived residual training window mismatch")
        predictions = (saved_future if is_future else stats).loc[plan.delivery_index_utc,
                       ["residual_corrected__" + q for q in Q]].rename(
                       columns={"residual_corrected__" + q: q for q in Q})
        if not residual.store(plan.delivery_date, train, plan.delivery_index_utc, base[Q],
                              predictions, list(audit.residual_feature_columns)):
            raise ValueError("Residual cache write failed")
    # End-to-end cached replay must reproduce every archived corrected value.
    prototype.fit = fail_fit
    replay_stats, replay_future, _ = engine.causal_residual_replay(
        raw_history=raw, raw_future=future, features=features, timezone=timezone,
        delivery_day=day, residual_factory=lambda: prototype, daily_cache=residual)
    for left, right in ((indexed(replay_stats), stats), (indexed(replay_future), saved_future)):
        cols = ["residual_corrected__" + q for q in Q]
        pd.testing.assert_frame_equal(left[cols], right[cols], check_exact=True)
    if residual.misses:
        raise ValueError("Imported residual replay has cache misses")
    old_kalman = Path(result.kalman_view.replay.audit["rolling_refit_cache"]["directory"])
    if not old_kalman.resolve().is_relative_to(source):
        raise ValueError("Kalman cache escapes its source workspace")
    shutil.copytree(old_kalman, namespace / "kalman_rolling")
    kalman_audit = verify_kalman(result, config, namespace / "kalman_rolling")
    destination = cache_root / "epochs" / namespace.name
    publish_epoch(namespace, destination, work / "backups")
    receipt = {"zone": zone, "source": str(source), "source_day": str(day),
        "source_manifest_sha256": engine._file_sha256(source / "report_only/frozen_result/manifest.json"),
        "source_engine_sha256": LEGACY_ENGINE, "destination": str(destination),
        "history_anchor_day": str(anchor), "reconstructed_source_hashes": hashes,
        "chronos_entries": neural.audit["writes"], "residual_entries": residual.writes,
        "kalman_entries": kalman_audit["history_hits"] + kalman_audit["future_hits"],
        "exact_live_chronos_check": True, "exact_live_residual_check": True,
        "exact_cached_residual_replay": True, "kalman_fits_during_verification": 0}
    (work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    provenance = cache_root / "imports"
    provenance.mkdir(parents=True, exist_ok=True)
    pending_receipt = provenance / (str(day) + ".pending.json")
    pending_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(pending_receipt, provenance / (str(day) + ".json"))
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-day", required=True)
    parser.add_argument("--zones", nargs="+", required=True, choices=["FR", "NL"])
    parser.add_argument("--root", type=Path, default=Path("runs/experiments/nuclear_forecast_v1"))
    parser.add_argument("--work", type=Path, default=Path("runs/tmp") / ("ncm_" + datetime.now().strftime("%m%d_%H%M%S")))
    args = parser.parse_args()
    for zone in args.zones:
        migrate(args.root / args.source_day / zone.lower() / "civil_pit_v2",
                args.root / "_daily_cache" / zone.lower() / "civil_pit_v2", args.work / zone.lower())
