"""Frozen NYX+CGC/CCC experiment: neural replay, correction, Kalman and reports.

No monkeypatches or changes to the nuclear/operational pipelines. Five native
cost series are exposed in exactly the same PIT input mechanism as nuclear FR.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock
from run_nuclear_forecast import sha256, write_json as _write_json, run_progress, run_forecast_with_storage_retry
from run_nuclear_cwe_forecast import _copy_checked, _freeze_reporting, _verify_reporting
from .sources import SERIES, load_bank, materialize
from . import runner as residual_lab

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/experiments/nyx_clean_fuel_full_v1"


def safe(path):
    path = Path(path).absolute()
    if path != path.resolve() or not path.is_relative_to(OUTPUT.resolve()):
        raise ValueError("Full Clean Fuel writes must stay inside its isolated namespace.")
    return path


def write_json(path, value):
    _write_json(safe(path), value)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_settings(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    fields = {"schema_version", "delivery_day", "zones", "output_root", "source_root", "fuel_sources",
              "include_attribution", "diagnostic_only", "production_modified"}
    if not isinstance(cfg, dict) or set(cfg) != fields or cfg["schema_version"] != 1:
        raise ValueError("Invalid full Clean Fuel configuration.")
    if (cfg["diagnostic_only"] is not True or cfg["production_modified"] is not False
            or type(cfg["include_attribution"]) is not bool
            or (ROOT / cfg["output_root"]).resolve() != OUTPUT.resolve()):
        raise ValueError("An isolated diagnostic configuration is mandatory.")
    validate_settings(cfg)
    return cfg


def validate_settings(cfg):
    residual_lab.validate_options({**cfg, "threads": 4, "workers": 2})
    residual_lab.source_dir(cfg, cfg["zones"][0])


def input_audit(cfg):
    audit = residual_lab.source_audit(cfg)
    for record in audit["zones"].values():
        record.pop("chronos_recalculated", None)
        record["new_chronos_replay_required"] = True
    return audit


def fuel_bank(cfg, *, workers):
    audit = input_audit(cfg)
    return materialize({"sources": cfg["fuel_sources"]}, audit["start_day"], audit["end_day"],
        residual_lab.NAMESPACE / "inputs" / cfg["delivery_day"], workers=workers)


def write_pit_costs(bank_path, directory):
    """Persist a D-1 information set on delivery hours; never claim future closes.

    value_time_utc is the DELIVERY hour. Source trading dates/ages remain in
    the immutable original daily bank audit. revision_time means query-asof,
    not an independently attested provider publication timestamp.
    """
    bank, audit = load_bank(bank_path)
    directory = safe(directory)
    directory.mkdir(parents=True, exist_ok=True)
    grid = pd.date_range(pd.Timestamp(bank.delivery_day.iloc[0], tz="Europe/Paris"),
        pd.Timestamp(bank.delivery_day.iloc[-1], tz="Europe/Paris") + pd.DateOffset(days=1),
        freq="h", inclusive="left").tz_convert("UTC")
    dates = grid.tz_convert("Europe/Paris").strftime("%Y-%m-%d")
    values = bank.set_index("delivery_day").loc[dates]
    shared = {"value_time_utc": grid,
              "snapshot_time_utc": pd.to_datetime(values.cutoff_time_utc.to_numpy(), utc=True),
              "revision_time_utc": pd.to_datetime(values.cutoff_time_utc.to_numpy(), utc=True)}
    paths = {}
    for alias in SERIES:
        path = directory / (alias + ".parquet")
        frame = pd.DataFrame({**shared, "value": values[alias].to_numpy(float)})
        if path.exists():
            try:
                pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)
            except AssertionError as exc:
                raise ValueError(
                    f"Frozen clean-fuel PIT values/schema changed for {alias}: {path}. "
                    "The existing artifact was not replaced."
                ) from exc
        else:
            frame.to_parquet(path, index=False)
        paths[alias] = path
    return paths, audit


def identity(cfg, zone, bank_path):
    from .forecast import clean_fuel_input_protocol
    baseline = residual_lab.source_dir(cfg, zone)
    return {"zone": zone, "delivery_day": cfg["delivery_day"], "input_protocol": clean_fuel_input_protocol(),
            "settings": cfg, "runner_sha256": sha256(Path(__file__)),
            "fuel_bank_sha256": sha256(Path(bank_path)), "fuel_audit_sha256": sha256(Path(str(bank_path) + ".audit.json")),
            "baseline_frozen_result_sha256": sha256(baseline / "report_only/frozen_result/manifest.json"),
            "baseline_inputs_sha256": sha256(baseline / "input_snapshot.json"),
            "baseline_receipt_sha256": sha256(baseline / "run_result.json")}


def workdir_for(cfg, zone, bank_path):
    key = hashlib.sha256(json.dumps(identity(cfg, zone, bank_path), sort_keys=True).encode()).hexdigest()[:16]
    return safe(OUTPUT / cfg["delivery_day"] / zone.lower() / key)


def verify_snapshot(work, expected=None):
    work = safe(work)
    manifest = read(work / "input_snapshot.json")
    if expected is not None and manifest["identity"] != expected:
        raise ValueError("Full Clean Fuel snapshot identity changed; a new snapshot is required.")
    if any(not isinstance(manifest.get(key), list) or not manifest[key] for key in ("files", "reference_files")):
        raise ValueError("Nonempty input and reference inventories required.")
    seen, inputs = set(), set()
    for item in manifest["files"] + manifest["reference_files"]:
        path = safe(item["snapshot"])
        if str(path) in seen or not path.is_relative_to(work) or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {path}")
        seen.add(str(path))
    for item in manifest["files"]:
        path = safe(item["snapshot"])
        if not path.is_relative_to(work / "snapshot"):
            raise ValueError("Model inputs must be in the sealed snapshot directory.")
        inputs.add(str(path))
    if sha256(work / "resolved_config.yaml") != manifest["resolved_config_sha256"]:
        raise ValueError("Frozen clean fuel configuration changed.")
    config = yaml.safe_load((work / "resolved_config.yaml").read_text(encoding="utf-8"))
    zone = manifest["identity"]["zone"]
    required = [config["zones"][zone]["target"]["file"], *config["data"]["pit_files"].values()]
    if config.get("nuclear_experiment", {}).get("residual_bank_audit"):
        required.append(config["nuclear_experiment"]["residual_bank_audit"])
    if any(str(Path(p).resolve()) not in inputs for p in required):
        raise ValueError("A configured model input is not checksum-pinned in the snapshot.")
    return config


def prepare(cfg, zone, bank_path):
    from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
    from chronos2_hourly.nuclear_incremental import prepare_incremental_settings
    from .forecast import clean_fuel_input_protocol
    work, baseline = workdir_for(cfg, zone, bank_path), residual_lab.source_dir(cfg, zone)
    expected = identity(cfg, zone, bank_path)
    with exclusive_process_lock(work / "prepare.lock"):
        if (work / "input_snapshot.json").exists():
            verify_snapshot(work, expected)
            return work
        incumbent = load_nuclear_result_bundle(workdir=baseline)
        prior = read(baseline / "input_snapshot.json")
        receipt_hash = sha256(baseline / "run_result.json")
        receipt = read(baseline / "run_result.json")
        config = yaml.safe_load((baseline / "resolved_config.yaml").read_text(encoding="utf-8"))
        mapping = {"target": config["zones"][zone]["target"]["file"], **config["data"]["pit_files"]}
        aliases_by_path = {str(Path(p).resolve()): alias for alias, p in mapping.items()}
        files, paths = [], {}
        for item in prior["files"]:
            source = Path(item["snapshot"]).resolve()
            if not source.is_relative_to(baseline / "snapshot"):
                raise ValueError("Baseline input escapes its frozen snapshot.")
            record = _copy_checked(source, safe(work / "snapshot" / source.name), expected=item["sha256"])
            files.append(record)
            if str(source) in aliases_by_path:
                paths[aliases_by_path[str(source)]] = record["snapshot"]
        if set(paths) != set(mapping):
            raise ValueError("Incomplete frozen nuclear input mapping.")
        for path in (Path(bank_path), Path(str(bank_path) + ".audit.json")):
            files.append(_copy_checked(path, safe(work / "snapshot" / path.name)))
        costs, cost_audit = write_pit_costs(work / "snapshot/bank.parquet", work / "snapshot")
        for alias, path in costs.items():
            paths[alias] = str(path)
            files.append({"snapshot": str(path), "snapshot_path": str(path), "sha256": sha256(path)})
        spec = config["zones"][zone]
        spec["target"]["file"] = paths["target"]
        for alias, series in SERIES.items():
            spec["covariates"][alias] = {
                "enabled": True, "source": "pit_parquet", "series": series,
                "description": "Indice clean fuel natif, CO2 inclus, connu avant 08h",
                "fill_method": "none", "fill_limit": 0, "minimum_coverage": 0.01,
                "future": {"known_future": True, "strategies": ["oracle"]},
                "unit": "EUR/MWh_e", "semantic": "native_clean_fuel_cost", "daily_broadcast": True,
                "carbon_included": True, "information_type": "prior_market_close_asof_cutoff",
            }
        for alias, covar in spec["covariates"].items():
            if covar.get("enabled", True):
                covar["pit_file"] = paths[alias]
                if "file" in covar:
                    covar["file"] = paths[alias]
        config["data"].update(pit_files={k: paths[k] for k, v in spec["covariates"].items() if v.get("enabled", True)},
            pit_vintage_dir=str(work / "snapshot"), cache_dir=str(work / "snapshot/cache"))
        config["output"]["directory"] = str(work)
        config["report"]["title"] = f"NYX nucleaire + CGC/CCC {zone} — candidat independant"
        columns = config["hourly"]["feature_engineering"].get("covariate_columns")
        if columns is not None:
            columns.extend(f"known_{a}_oracle" for a in SERIES if f"known_{a}_oracle" not in columns)
        experiment = config["nuclear_experiment"]
        for key in ("history_anchor_day", "raw_history_start_day", "incremental_namespace"):
            experiment.pop(key, None)
        experiment.update(input_protocol=clean_fuel_input_protocol(), mode="incremental",
            incremental_cache_dir=str(OUTPUT / "_daily_cache" / zone.lower()),
            candidate_model="clean_fuel_full", production_modified=False)
        if experiment.get("residual_bank_audit"):
            experiment["residual_bank_audit"] = str(work / "snapshot" / Path(experiment["residual_bank_audit"]).name)
        anchor = incumbent.audit["history_anchor_day"]
        prepare_incremental_settings(config, (pd.Timestamp(anchor) + pd.Timedelta(days=730)).date())
        _, actual_anchor, _ = prepare_incremental_settings(config, pd.Timestamp(cfg["delivery_day"]).date())
        if str(actual_anchor) != anchor:
            raise ValueError("Candidate and incumbent cold-start anchors differ.")
        _, refs = _freeze_reporting(baseline, work, receipt)
        reference = work / "reference"
        for name, frame in (("incumbent_backtest", incumbent.kalman_view.backtest),
                            ("incumbent_forecast", incumbent.kalman_view.forecast),
                            ("incumbent_residual", incumbent.residual_statistics),
                            ("incumbent_source_forecast", incumbent.source_forecast)):
            path = reference / (name + ".parquet")
            frame.to_parquet(path)
            refs.append({"snapshot_path": str(path), "sha256": sha256(path)})
        for ref in refs:
            ref["snapshot"] = ref.get("snapshot", ref["snapshot_path"])
        write_json(work / "source_audit.json", {"clean_fuel": cost_audit, "production_pit_evidence": False,
                   "upstream_neural_replay_required": True, "baseline": str(baseline),
                   "same_history_anchor": anchor, "same_model_hyperparameters": True})
        refs.append({"snapshot": str(work / "source_audit.json"), "sha256": sha256(work / "source_audit.json")})
        resolved = work / "resolved_config.yaml"
        resolved.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if sha256(baseline / "run_result.json") != receipt_hash:
            raise ValueError("Baseline reporting changed while snapshotting.")
        write_json(work / "input_snapshot.json", {"schema_version": 1, "identity": expected,
            "files": files, "reference_files": refs, "resolved_config_sha256": sha256(resolved),
            "production_modified": False})
        verify_snapshot(work, expected)
        _verify_reporting(work, zone=zone, day=cfg["delivery_day"], timezone=spec["timezone"])
    return work


def run_zone(cfg, zone, work, *, action, device, threads, workers, skip_attribution=False):
    from chronos2_modular.common import build_zone_configs
    from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
    from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle, save_nuclear_result_bundle
    from .forecast import run_clean_fuel_forecast
    from .standard_report import render_clean_fuel_reports
    day = cfg["delivery_day"]
    config = verify_snapshot(work)
    with exclusive_process_lock(work / "run.lock"), run_progress(work, zone, pd.Timestamp(day), action) as progress:
        progress("prepare_inputs")
        spec = build_zone_configs(config, [zone], None, None)[0]
        target, reporting_audit = _verify_reporting(work, zone=zone, day=day, timezone=spec.timezone)
        data = prepare_nuclear_zone_data(spec, config, work, work / "prepared")
        if action == "prepare":
            return {"status": "prepared", "workdir": str(work)}
        reuse = (work / "report_only/frozen_result").exists()
        if action == "report" or reuse:
            result = load_nuclear_result_bundle(workdir=work)
        else:
            progress("forecast", clean_fuel_in_chronos=True, clean_fuel_in_residual=True, clean_fuel_in_kalman=True)
            print(f"[CleanFuel/{zone}] nouveau replay Chronos CGC/CCC + correcteur + Kalman; caches isoles.", flush=True)
            result = run_forecast_with_storage_retry(run_clean_fuel_forecast, config=config, data=data,
                zone=zone, delivery_day=day, workdir=work, device=device, threads=threads, workers=workers)
            save_nuclear_result_bundle(result, workdir=work)
        attribution_directory = None
        attribution = {"status": "unavailable", "reason": "no_matching_new_model_attribution"}
        if cfg["include_attribution"] and not skip_attribution:
            try:
                if action == "report":
                    prior = read(work / "run_result.json")
                    attribution = prior.get("variable_attribution", attribution)
                    if attribution.get("status") == "complete":
                        attribution_directory = Path(attribution["directory"])
                else:
                    progress("attribution")
                    from chronos2_hourly.nuclear_attribution import prepare_nuclear_attribution
                    attribution_directory = prepare_nuclear_attribution(result, data, config, work,
                                                                          device=device, threads=threads)
                    attribution = {"status": "complete", "directory": str(attribution_directory)}
            except Exception as error:
                attribution = {"status": "failed", "error": str(error)}
                print(f"[CleanFuel/{zone}] attribution unavailable: {error}", flush=True)
        progress("render_reports")
        incumbent = SimpleNamespace(
            residual_statistics=pd.read_parquet(work / "reference/incumbent_residual.parquet"),
            source_forecast=pd.read_parquet(work / "reference/incumbent_source_forecast.parquet"),
            kalman_view=SimpleNamespace(backtest=pd.read_parquet(work / "reference/incumbent_backtest.parquet"),
                                       forecast=pd.read_parquet(work / "reference/incumbent_forecast.parquet")))
        reports = render_clean_fuel_reports(result, incumbent=incumbent, data=replace(data, target=target),
            zone=zone, delivery_day=day, output_directory=work / "reports",
            source_audit=read(work / "source_audit.json"), storm_archive=work / "reference/reporting",
            observed_source_audit=reporting_audit, attribution_directory=attribution_directory)
        receipt = {"status": "complete", "zone": zone, "delivery_day": day, "reports": reports,
                   "variable_attribution": attribution, "result_reused": reuse,
                   "production_modified": False, "activation_performed": False, "audit": result.audit}
        write_json(work / "run_result.json", receipt)
        return receipt
