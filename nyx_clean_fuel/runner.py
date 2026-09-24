"""Replay only the residual correction on frozen nuclear Chronos predictions.

All writes stay in this experiment. Existing forecasts, recipes, models and
published reports are read-only. Fitting never consumes Storm or report labels.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock
from .features import build_features, PREFIX

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = ROOT / "runs/experiments/nyx_clean_fuel_v1"
Q = ("q10", "q50", "q90")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, default=str).encode()


def safe(path):
    path = Path(path).absolute()
    if path != path.resolve() or not path.is_relative_to(NAMESPACE.resolve()):
        raise ValueError(f"Writes must stay under {NAMESPACE}; aliases forbidden.")
    return path


def write_json(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_bytes(canonical(value))
    pending.replace(path)


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    required = {"schema_version", "output_root", "source_root", "zones", "delivery_day",
                "threads", "workers", "include_kalman", "fuel_sources", "diagnostic_only", "production_modified"}
    if not isinstance(cfg, dict) or set(cfg) != required or cfg["schema_version"] != 1:
        raise ValueError("Unknown or incomplete clean fuel configuration.")
    if (cfg["diagnostic_only"] is not True or cfg["production_modified"] is not False
            or type(cfg["include_kalman"]) is not bool):
        raise ValueError("Clean Fuel is an isolated diagnostic, never a production activation.")
    if (ROOT / cfg["output_root"]).resolve() != NAMESPACE.resolve():
        raise ValueError("The experiment output namespace is fixed.")
    validate_options(cfg)
    return cfg


def validate_options(cfg):
    day = pd.Timestamp(cfg["delivery_day"])
    if (not isinstance(cfg["delivery_day"], str) or day.tzinfo is not None or day != day.normalize()
            or pd.isna(day) or day.strftime("%Y-%m-%d") != cfg["delivery_day"]):
        raise ValueError("delivery_day must be a civil ISO date.")
    if (not isinstance(cfg["zones"], list) or not cfg["zones"]
            or len(set(cfg["zones"])) != len(cfg["zones"])
            or set(cfg["zones"]) - {"FR", "DE", "BE", "NL"}):
        raise ValueError("Unique zones FR/DE/BE/NL are required.")
    for name, maximum in (("threads", 32), ("workers", 2)):
        if type(cfg[name]) is not int or not 1 <= cfg[name] <= maximum:
            raise ValueError(f"Invalid {name}.")
    from .sources import _settings
    _settings({"sources": cfg["fuel_sources"]})


def source_dir(cfg, zone):
    base = (ROOT / cfg["source_root"]).resolve()
    if base != (ROOT / "runs/experiments/nuclear_forecast_v1").resolve():
        raise ValueError("Only the frozen nuclear baseline is supported by this adapter.")
    return base / cfg["delivery_day"] / zone.lower() / "civil_pit_v2"


def source_audit(cfg):
    """Cheap read-only preflight; no model imports, downloads or writes."""
    starts = []
    zones = {}
    for zone in cfg["zones"]:
        work = source_dir(cfg, zone)
        path = work / "report_only/frozen_result/manifest.json"
        if not path.is_file():
            raise ValueError(f"{zone}: frozen nuclear result absent for {cfg['delivery_day']}.")
        audits = json.loads((path.parent / "audits.json").read_text())
        record = audits["result"]
        starts.append(record["raw_history_start_day"])
        zones[zone] = {"source": str(work), "source_manifest_sha256": sha(path),
                       "raw_start": record["raw_history_start_day"], "chronos_recalculated": False}
    return {"start_day": min(starts), "end_day": cfg["delivery_day"], "zones": zones,
            "production_modified": False}


def code_identity():
    files = list((ROOT / "nyx_clean_fuel").glob("*.py"))
    files = [p for p in files if p.name != "report.py"]
    files += [ROOT / name for name in (
        "run_chronos2_hourly.py", "chronos2_hourly/nuclear_forecast.py",
        "chronos2_hourly/models/residual_corrector.py", "chronos2_hourly/features.py",
        "chronos2_hourly/models/blended_residual_corrector.py", "chronos2_hourly/models/catboost_hourly.py",
        "chronos2_hourly/nuclear_residual_cache.py", "chronos2_hourly/kalman_residual.py",
        "chronos2_hourly/kalman_covariates.py", "chronos2_hourly/nuclear_preparation.py",
        "chronos2_hourly/nuclear_run_archive.py")]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}


def indexed(frame):
    result = frame.copy()
    if "delivery_start_utc" in result:
        result = result.set_index("delivery_start_utc")
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, utc=True), name="delivery_start_utc")
    if not result.index.is_unique or not result.index.is_monotonic_increasing:
        raise ValueError("Unsorted or duplicate forecast hours.")
    return result


def load_raw_history(work, result):
    """Restore the original warm-up prefix without a neural rerun.

    The source CSV serializes float32. Casting back must reproduce every
    overlapping frozen parquet value EXACTLY; otherwise fail closed.
    """
    checkpoint = work / "checkpoints/nuclear_chronos_oof.csv.gz"
    manifest = json.loads(checkpoint.with_name(checkpoint.name + ".manifest.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("output_sha256") != sha(checkpoint)
            or manifest.get("source_hashes") != result.audit["source_hashes"]):
        raise ValueError("The raw Chronos checkpoint no longer matches the frozen source.")
    raw = indexed(pd.read_csv(checkpoint))
    raw.forecast_origin_utc = pd.to_datetime(raw.forecast_origin_utc, utc=True)
    archived = indexed(result.raw_history)
    for col in (*Q, "actual"):
        raw[col] = raw[col].astype(archived[col].dtype)
        if not np.array_equal(raw.loc[archived.index, col].to_numpy(), archived[col].to_numpy()):
            raise ValueError(f"Raw Chronos checkpoint/immutable parquet mismatch for {col}.")
    return raw.loc[:, ["forecast_origin_utc", *Q, "actual"]]


def reporting_inputs(work, zone, timezone):
    """Current observations are ONLY for scoring, not residual training labels."""
    from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
    from chronos2_hourly.nuclear_reporting_refresh import verify_refreshed_observations
    receipt = json.loads((work / "run_result.json").read_text())
    audit = receipt.get("reporting_sources", {})
    if audit.get("status") != "complete":
        raise ValueError("Frozen report-only observations and Storm must be available; run the usual report refresh first.")
    observed_path = Path(audit["observed"]["artifact_path"])
    if sha(observed_path) != audit["observed"]["artifact_sha256"]:
        raise ValueError("Reporting observations checksum mismatch.")
    frame = pd.read_parquet(observed_path)
    if isinstance(frame.index, pd.DatetimeIndex):
        observed = frame.iloc[:, 0]
    else:
        observed = frame.set_index("timestamp").iloc[:, 0]
    observed.index = pd.to_datetime(observed.index, utc=True)
    verify_refreshed_observations(observed, audit, zone=zone, timezone=timezone,
                                  delivery_day=receipt["audit"]["delivery_day"])
    storm = _load_verified_snapshot(Path(audit["snapshot_directory"]), zone=zone, timezone=timezone)
    if storm is None:
        raise ValueError("Verified Storm reporting snapshot absent.")
    return observed.astype(float), storm[0], audit


def prepare_zone(cfg, zone, bank_path):
    from .sources import load_bank
    from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
    from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
    from chronos2_hourly.nuclear_forecast import _digest_frame
    from chronos2_modular.common import build_zone_configs
    from run_chronos2_hourly import _feature_inputs
    work = source_dir(cfg, zone)
    source = load_nuclear_result_bundle(workdir=work)
    bank, fuel_audit = load_bank(bank_path)
    resolved = yaml.safe_load((work / "resolved_config.yaml").read_text(encoding="utf-8-sig"))
    identity = {"config": cfg, "zone": zone, "source_manifest": sha(work / "report_only/frozen_result/manifest.json"),
                "fuel_sha256": sha(bank_path), "fuel_audit_sha256": sha(str(bank_path) + ".audit.json"),
                "code": code_identity(), "python": sys.version,
                "runtime": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "catboost", "pyarrow")}}
    key = hashlib.sha256(canonical(identity)).hexdigest()[:24]
    directory = safe(NAMESPACE / cfg["delivery_day"] / zone.lower() / key)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Existing clean fuel snapshot has a different identity.")
        verify_files(directory, manifest["files"])
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(directory / "prepare.lock"):
        LOG.info("[CleanFuel/%s] Prepare frozen Chronos and features (no neural execution)", zone)
        spec = build_zone_configs(resolved, [zone], None, None)[0]
        data = prepare_nuclear_zone_data(spec, resolved, work, directory / "prepared")
        _, _, _, features = _feature_inputs(data, resolved)
        if _digest_frame(features) != source.audit["source_hashes"]["residual_features"]:
            raise ValueError("Reconstructed residual features differ from the frozen operational inputs.")
        raw = load_raw_history(work, source)
        future = indexed(source.source_forecast)[["forecast_origin_utc", *["chronos2__" + q for q in Q]]]
        future = future.rename(columns={"chronos2__" + q: q for q in Q})
        base = pd.concat([raw.loc[:, Q], future.loc[:, Q]])
        extras = build_features(bank, base.index, zone=zone, timezone=spec.timezone, base=base)
        X = features.loc[base.index].join(extras, validate="one_to_one")
        observed, storm, reporting_audit = reporting_inputs(work, zone, spec.timezone)
        labels = pd.DataFrame({"actual": observed.reindex(base.index), "storm_q50": storm.reindex(base.index)})
        frames = {"features": X, "raw": raw, "future": future, "labels": labels,
                  "reference_residual": indexed(source.residual_statistics),
                  "reference_forecast": indexed(source.source_forecast),
                  "reference_kalman": indexed(source.kalman_view.backtest),
                  "reference_kalman_forecast": indexed(source.kalman_view.forecast),
                  "covariates": indexed(source.covariates)}
        for name, frame in frames.items():
            frame.to_parquet(directory / (name + ".parquet"))
        recipe = deepcopy(source.audit["residual_recipe"])
        recipe["thread_count"] = cfg["threads"]
        prepared = {"timezone": spec.timezone, "residual_recipe": recipe,
                    "kalman_parameters": source.audit["kalman_filter_parameters"],
                    "fuel_audit": fuel_audit, "reporting_audit": reporting_audit,
                    "added_features": list(extras), "source_audit": source.audit}
        write_json(directory / "settings.json", prepared)
        files = {name + ".parquet": sha(directory / (name + ".parquet")) for name in frames}
        files["settings.json"] = sha(directory / "settings.json")
        write_json(manifest_path, {"identity": identity, "files": files})
        write_json(directory / "status.json", {"status": "prepared", "production_modified": False})
    return directory


def verify_files(directory, files):
    if not isinstance(files, dict) or not files:
        raise ValueError("A nonempty artifact manifest is required.")
    for name, checksum in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Artifact filenames must be local basenames.")
        path = safe(directory / name)
        if not path.is_file() or sha(path) != checksum:
            raise ValueError(f"Missing or modified experiment artifact: {path}.")


def verify_results(directory, manifest):
    result = json.loads((directory / "results.json").read_text())
    expected = {"candidate_residual.parquet", "candidate_forecast.parquet", "residual_fit_audit.parquet"}
    if manifest["identity"]["config"]["include_kalman"]:
        expected.update({"candidate_kalman.parquet", "candidate_kalman_forecast.parquet", "kalman_audit.json"})
    if (result.get("status") != "complete" or set(result.get("files", {})) != expected
            or result.get("manifest_sha256") != sha(directory / "manifest.json")):
        raise ValueError("Result/input manifest mismatch or incomplete result inventory.")
    verify_files(directory, result["files"])
    return result


class ProgressCache:
    """Existing per-day cache + visible progress; no model pickle deserialization."""
    def __init__(self, cache, directory):
        self.cache, self.directory = cache, directory
        self.n, self.started = 0, time.monotonic()

    def load(self, *args):
        self.n += 1
        if self.n == 1 or self.n % 10 == 0:
            elapsed = time.monotonic() - self.started
            state = {"status": "running", "phase": "residual", "delivery_day": str(args[0]),
                     "days_visited": self.n, "elapsed_seconds": round(elapsed),
                     "cache_hits": self.cache.hits, "production_modified": False}
            write_json(self.directory / "status.json", state)
            LOG.info("[CleanFuel] residual %s, %s days, %.1f min, cache hits=%s", args[0], self.n, elapsed / 60, self.cache.hits)
        return self.cache.load(*args)

    def store(self, *args):
        return self.cache.store(*args)


def evaluate(directory):
    from chronos2_hourly.nuclear_forecast import causal_residual_replay, nuclear_kalman_covariate_config
    from chronos2_hourly.nuclear_residual_cache import ResidualDayCache
    from chronos2_hourly.kalman_residual import build_operational_kalman_view, KalmanResidualConfig
    from run_chronos2_hourly import _residual_corrector_factory
    directory = safe(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_files(directory, manifest["files"])
    if manifest["identity"]["code"] != code_identity():
        raise ValueError("Code changed since Prepare; create a new experiment snapshot.")
    cfg, zone = manifest["identity"]["config"], manifest["identity"]["zone"]
    with exclusive_process_lock(directory / "run.lock"):
        if (directory / "results.json").exists():
            verify_results(directory, manifest)
            LOG.info("[CleanFuel/%s] Complete result reused; zero fit", zone)
            return directory
        settings = json.loads((directory / "settings.json").read_text())
        read = lambda name: pd.read_parquet(directory / (name + ".parquet"))
        X, raw, future = read("features"), read("raw"), read("future")
        day, timezone = pd.Timestamp(cfg["delivery_day"]).date(), settings["timezone"]
        factory, _ = _residual_corrector_factory({"hourly": {"residual_correction": settings["residual_recipe"]}}, timezone=timezone)
        prototype = factory()
        from chronos2_hourly.models.residual_corrector import ResidualMetaFeatureBuilder
        builder = ResidualMetaFeatureBuilder(**prototype.feature_builder_options)
        if any(builder._is_excluded(name) for name in settings["added_features"]):
            raise ValueError("The configured residual builder excludes a clean fuel feature.")
        used = X.loc[:, [col for col in X if not builder._is_excluded(col)]]
        contract = {"recipe": settings["residual_recipe"], "code": manifest["identity"]["code"]}
        # Share only content-addressed fits across dated experiments. Adding a
        # delivery day must not repeat unchanged historical fits; the key still
        # includes exact causal features, labels, base predictions and recipe.
        cache = ResidualDayCache(safe(NAMESPACE / "_daily_cache" / zone.lower() / "residual"),
                                 contract, features=used, raw=raw, timezone=timezone)
        statistics, forecast, daily = causal_residual_replay(
            raw_history=raw, raw_future=future, features=X, timezone=timezone, delivery_day=day,
            residual_factory=factory, output_start_day=day - timedelta(days=730),
            daily_cache=ProgressCache(cache, directory))
        fitted = daily.loc[daily.generation_source.eq("daily_prequential_refit"), "residual_feature_columns"]
        if not all(set(settings["added_features"]).issubset(set(cols)) for cols in fitted):
            raise ValueError("The fitted correction silently dropped clean fuel features.")
        outputs = {"candidate_residual": statistics, "candidate_forecast": forecast, "residual_fit_audit": daily}
        if cfg["include_kalman"]:
            LOG.info("[CleanFuel/%s] Kalman rolling365 on candidate-specific corrected history", zone)
            write_json(directory / "status.json", {"status": "running", "phase": "kalman", "production_modified": False})
            params = settings["kalman_parameters"]
            params["candidate_kinds"] = tuple(params["candidate_kinds"])
            view = build_operational_kalman_view(
                statistics=statistics, source_forecast=forecast, covariates=read("covariates"),
                timezone=timezone, delivery_day=day, config=KalmanResidualConfig(**params),
                covariate_config=nuclear_kalman_covariate_config(), upstream_model="residual_corrected",
                output_model="residual_kalman", training_lookback_days=365,
                rolling_refit_workers=cfg["workers"],
                rolling_refit_cache_dir=safe(NAMESPACE / "_daily_cache" / zone.lower() / "kalman"))
            outputs.update(candidate_kalman=view.backtest, candidate_kalman_forecast=view.forecast)
            write_json(directory / "kalman_audit.json", view.replay.audit)
        for name, frame in outputs.items():
            frame.to_parquet(directory / (name + ".parquet"))
        files = {name + ".parquet": sha(directory / (name + ".parquet")) for name in outputs}
        if cfg["include_kalman"]:
            files["kalman_audit.json"] = sha(directory / "kalman_audit.json")
        write_json(directory / "results.json", {"status": "complete", "files": files,
                    "manifest_sha256": sha(directory / "manifest.json"), "production_modified": False,
                    "cache_hits": cache.hits, "cache_misses": cache.misses})
        write_json(directory / "status.json", {"status": "complete", "production_modified": False})
    return directory


def panel_for_report(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_files(directory, manifest["files"])
    verify_results(directory, manifest)
    cfg, zone = manifest["identity"]["config"], manifest["identity"]["zone"]
    day = pd.Timestamp(cfg["delivery_day"])
    read = lambda name: indexed(pd.read_parquet(directory / (name + ".parquet")))
    labels, frames = read("labels"), []
    for model, hist, future, prefix in (
        ("nuclear_autonomous", "reference_residual", "reference_forecast", "residual_corrected__"),
        ("clean_fuel_autonomous", "candidate_residual", "candidate_forecast", "residual_corrected__"),
        ("nuclear_kalman", "reference_kalman", "reference_kalman_forecast", "residual_kalman__"),
        ("clean_fuel_kalman", "candidate_kalman", "candidate_kalman_forecast", "residual_kalman__"),
    ):
        if model == "clean_fuel_kalman" and not cfg["include_kalman"]:
            continue
        for phase, name in (("history", hist), ("future", future)):
            source = read(name)
            local = source.index.tz_convert("Europe/Paris").tz_localize(None).normalize()
            source = source.loc[local >= day - pd.Timedelta(days=365)]
            frame = source.loc[:, [prefix + q for q in Q]].rename(columns={prefix + q: q for q in Q})
            frame = frame.join(labels)
            if phase == "history" and not np.isfinite(frame.actual).all():
                raise ValueError("Reporting labels do not cover every evaluated hour.")
            frame["phase"], frame["model"], frame["zone"] = phase, model, zone
            frames.append(frame.reset_index())
    return pd.concat(frames, ignore_index=True)


def report(directories, path):
    from .report import render_report
    panel = pd.concat([panel_for_report(d) for d in directories], ignore_index=True)
    audit = {"diagnostic_only": True, "production_modified": False, "activation_performed": False,
             "baseline_by_model": {"clean_fuel_autonomous": "nuclear_autonomous", "clean_fuel_kalman": "nuclear_kalman"},
             "cutoff": "D-1 08:00 Europe/Paris", "training_window_days": 365,
             "evaluation_days": 365, "daily_refit": True, "source_directories": list(map(str, directories)),
             "chronos_recomputed": False, "quantile_intervals_recalibrated": False,
             "cost_formulas": {"CGC": "2 × gaz DA du hub + 0.368 × EUA (EUR/MWh électrique)",
                               "CCC": "2.63 × (API2 USD/t / EURUSD / 6.9776 + 0.34 × EUA)"},
             "gas_hubs": {"FR": "PEG", "DE": "THE", "BE": "ZEE", "NL": "TTF"},
             "co2_already_included": True,
             "note": "CGC/CCC enter the existing residual learner. Carbon is already included. No enforced price floor. Historical labels are frozen; latest observations are scoring-only. The first warmup year is diagnostic, not full nested training. Saturn provider insertion times and historical formula versions are not independently attested. Retrospective diagnostic, no production promotion."}
    return render_report(panel, safe(path), audit)
