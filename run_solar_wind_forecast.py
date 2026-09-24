"""NYX solar CWE + DE/NL wind; isolated full-chain replay, never production."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import html
import json
import logging
import os
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock
from run_nuclear_forecast import sha256, write_json, run_progress, run_forecast_with_storage_retry
from run_nuclear_cwe_forecast import _copy_checked, _freeze_reporting, _verify_reporting, _day

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs/experiments/solar_wind_v1"
SOURCE = ROOT / "data/pit/solar_wind_v1"
BASELINE = ROOT / "runs/experiments/nuclear_forecast_v1"
ZONES = ("DE", "NL")
SCIENTIFIC_FILES = (
    "chronos2_hourly/solar_wind_forecast.py", "chronos2_hourly/solar_wind_sources.py",
    "chronos2_hourly/nuclear_forecast.py", "chronos2_hourly/nuclear_incremental.py",
    "chronos2_hourly/nuclear_preparation.py", "chronos2_hourly/chronos_adapter.py",
    "chronos2_hourly/models/residual_corrector.py", "chronos2_hourly/features.py",
    "chronos2_hourly/kalman_residual.py", "chronos2_hourly/kalman_covariates.py",
    "chronos2_modular/forecasting.py", "chronos2_modular/data.py", "run_chronos2_hourly.py",
)


def scientific_identity():
    """Pin scientific implementations even when a frozen result is reused."""
    code_root = Path(__file__).resolve().parent
    versions = {}
    for name in ("numpy", "pandas", "catboost", "torch", "chronos-forecasting"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return {"files": {name: sha256(code_root/name) for name in SCIENTIFIC_FILES}, "dependencies": versions}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe(path):
    path = Path(path).absolute()
    if OUTPUT.resolve() != OUTPUT or path != path.resolve() or not path.is_relative_to(OUTPUT) or path == OUTPUT:
        raise ValueError("Solar writes must stay in a real child of runs/experiments/solar_wind_v1.")
    return path


def load_settings(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    fields = {"schema_version", "delivery_day", "zones", "output_root", "source_root", "include_attribution", "wind_dst_policy", "wind_gap_policy",
              "diagnostic_only", "production_modified"}
    if not isinstance(cfg, dict) or set(cfg) != fields or type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("Invalid SolarWind configuration.")
    if (cfg["diagnostic_only"] is not True or cfg["production_modified"] is not False
            or cfg["wind_dst_policy"] not in {"raise", "duplicate"}
            or cfg["wind_gap_policy"] not in {None, "nl_ecmwf_spring_2025_2026"}
            or type(cfg["include_attribution"]) is not bool or (ROOT/cfg["output_root"]).resolve() != OUTPUT
            or (ROOT/cfg["source_root"]).resolve() != SOURCE):
        raise ValueError("Isolated source/output namespaces and diagnostic mode are mandatory.")
    validate_settings(cfg)
    return cfg


def validate_settings(cfg):
    _day(cfg["delivery_day"])
    zones = cfg["zones"]
    if not isinstance(zones, list) or not zones or len(zones) != len(set(zones)) or set(zones)-set(ZONES):
        raise ValueError("Choose unique target countries DE and/or NL.")


def baseline_for(cfg, zone):
    if zone not in ZONES:
        raise ValueError("Invalid zone.")
    return BASELINE / _day(cfg["delivery_day"]) / zone.lower() / "civil_pit_v2"


def audit_inputs(cfg):
    """No fitting or network requests; all requested countries checked first."""
    records = {}
    for zone in cfg["zones"]:
        work = baseline_for(cfg, zone)
        manifest = work/"report_only/frozen_result/manifest.json"
        if not manifest.is_file():
            raise ValueError(f"{zone}: completed nuclear baseline missing for {cfg['delivery_day']}. Choose an existing delivery.")
        result = read(manifest.parent/"audits.json")["result"]
        records[zone] = {"baseline": str(work), "manifest_sha256": sha256(manifest),
                         "raw_start_day": result["raw_history_start_day"], "history_anchor_day": result["history_anchor_day"]}
    return {"zones": records, "start_day": min(r["raw_start_day"] for r in records.values()),
            "end_day": cfg["delivery_day"], "production_modified": False}


def identity(cfg, zone, sources):
    from chronos2_hourly.solar_wind_forecast import solar_wind_input_protocol
    baseline = baseline_for(cfg, zone)
    for alias, record in sources.items():
        for field, path in (("sha256", Path(record["path"])),
                            ("audit_sha256", Path(str(record["path"])+".audit.json"))):
            if not record.get(field) or sha256(path) != record[field]:
                raise ValueError(f"{alias}: audited source changed before snapshot preparation.")
    return {"zone": zone, "delivery_day": cfg["delivery_day"], "settings": cfg,
        "scientific_identity": scientific_identity(),
        "input_protocol": solar_wind_input_protocol(), "runner_sha256": sha256(Path(__file__)),
        "sources": {alias: {"parquet": sha256(Path(rec["path"])),
                    "audit": sha256(Path(str(rec["path"])+".audit.json"))} for alias, rec in sources.items()},
        "baseline_inputs": sha256(baseline/"input_snapshot.json"),
        "baseline_result": sha256(baseline/"report_only/frozen_result/manifest.json"),
        "baseline_receipt": sha256(baseline/"run_result.json")}


def verify_snapshot(work, *, zone=None, delivery_day=None):
    from chronos2_hourly.solar_wind_forecast import solar_wind_input_protocol
    work = safe(work)
    manifest = read(work/"input_snapshot.json")
    signature = manifest.get("identity", {})
    saved_zone, saved_day = signature.get("zone"), signature.get("delivery_day")
    if (saved_zone not in ZONES or not isinstance(saved_day, str)
            or work.parent != OUTPUT/_day(saved_day)/saved_zone.lower()
            or (zone is not None and saved_zone != zone)
            or (delivery_day is not None and saved_day != delivery_day)
            or signature.get("input_protocol") != solar_wind_input_protocol()
            or signature.get("runner_sha256") != sha256(Path(__file__))
            or signature.get("scientific_identity") != scientific_identity()):
        raise ValueError("Solar snapshot zone, day or input protocol mismatch.")
    if any(not isinstance(manifest.get(key), list) or not manifest[key] for key in ("files", "reference_files")):
        raise ValueError("Incomplete solar snapshot manifest.")
    seen, inputs = set(), set()
    for item in manifest["files"] + manifest["reference_files"]:
        path = safe(item["snapshot_path"])
        if path in seen or not path.is_relative_to(work) or sha256(path) != item["sha256"]:
            raise ValueError(f"Solar snapshot changed: {path}")
        seen.add(path)
    for item in manifest["files"]:
        path = Path(item["snapshot_path"])
        if not path.is_relative_to(work/"snapshot"):
            raise ValueError("Model input outside sealed snapshot.")
        inputs.add(path)
    resolved = work/"resolved_config.yaml"
    if sha256(resolved) != manifest["resolved_config_sha256"]:
        raise ValueError("Solar resolved config changed.")
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if config["nuclear_experiment"].get("input_protocol") != signature["input_protocol"]:
        raise ValueError("Solar recipe input protocol mismatch.")
    required = [config["zones"][saved_zone]["target"]["file"], *config["data"]["pit_files"].values()]
    extra = config["nuclear_experiment"].get("residual_bank_audit")
    if extra:
        required.append(extra)
    if any(Path(p).resolve() not in inputs for p in required):
        raise ValueError("A solar model input is not sealed.")
    return config


def prepare(cfg, zone, sources):
    from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
    from chronos2_hourly.nuclear_incremental import prepare_incremental_settings
    from chronos2_hourly.solar_wind_forecast import GENERATION_SERIES, solar_wind_input_protocol
    if set(sources) != set(GENERATION_SERIES):
        raise ValueError("Exactly four CWE solar and two DE/NL wind forecasts are required.")
    signature = identity(cfg, zone, sources)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    work = safe(OUTPUT/cfg["delivery_day"]/zone.lower()/key)
    baseline = baseline_for(cfg, zone)
    with exclusive_process_lock(work/"prepare.lock"):
        if (work/"input_snapshot.json").exists():
            if read(work/"input_snapshot.json")["identity"] != signature:
                raise ValueError("Solar snapshot identity mismatch.")
            verify_snapshot(work)
            return work
        incumbent = load_nuclear_result_bundle(workdir=baseline)
        prior, receipt = read(baseline/"input_snapshot.json"), read(baseline/"run_result.json")
        if sha256(baseline/"resolved_config.yaml") != prior["resolved_config_sha256"]:
            raise ValueError("Baseline recipe changed.")
        config = yaml.safe_load((baseline/"resolved_config.yaml").read_text(encoding="utf-8"))
        mapping = {"target": config["zones"][zone]["target"]["file"], **config["data"]["pit_files"]}
        aliases = {str(Path(p).resolve()): alias for alias, p in mapping.items()}
        files, paths = [], {}
        for item in prior["files"]:
            source = Path(item["snapshot"]).resolve()
            if not source.is_relative_to(baseline/"snapshot"):
                raise ValueError("Baseline input outside its frozen snapshot.")
            record = _copy_checked(source, safe(work/"snapshot"/source.name), expected=item["sha256"])
            files.append(record)
            if str(source) in aliases:
                paths[aliases[str(source)]] = record["snapshot_path"]
        if set(paths) != set(mapping) or set(mapping).intersection(GENERATION_SERIES):
            raise ValueError("Incomplete baseline inputs or duplicate pre-existing solar inputs.")
        for alias, record in sources.items():
            source = Path(record["path"])
            for source_file in (source, Path(str(source)+".audit.json")):
                expected = record["sha256" if source_file == source else "audit_sha256"]
                copied = _copy_checked(source_file, safe(work/"snapshot"/source_file.name), expected=expected)
                files.append(copied)
                if source_file == source:
                    paths[alias] = copied["snapshot_path"]
        spec = config["zones"][zone]
        spec["target"]["file"] = paths["target"]
        for alias, series in GENERATION_SERIES.items():
            spec["covariates"][alias] = {"enabled": True, "source": "pit_parquet", "series": series,
                "description": f"{alias[:2].upper()} generation {'eolienne' if '_wind_' in alias else 'solaire'} prevue horaire (GW)",
                "include_base_context": True, "fill_method": "none", "fill_limit": 0, "minimum_coverage": .01,
                "future": {"known_future": True, "strategies": ["oracle"]},
                "unit": "GW", "semantic": "forecast_generation", "daily_broadcast": False}
        for alias, covar in spec["covariates"].items():
            if covar.get("enabled", True):
                covar["pit_file"] = paths[alias]
                if "file" in covar:
                    covar["file"] = paths[alias]
        config["data"].update(pit_files={a: paths[a] for a, v in spec["covariates"].items() if v.get("enabled", True)},
            pit_vintage_dir=str(work/"snapshot"), cache_dir=str(work/"snapshot/cache"))
        config["output"]["directory"] = str(work)
        config["report"]["title"] = f"NYX nucleaire FR + solaire CWE + eolien DE/NL — {zone}"
        columns = config["hourly"]["feature_engineering"].get("covariate_columns")
        if columns is not None:
            columns.extend(f"known_{a}_oracle" for a in GENERATION_SERIES if f"known_{a}_oracle" not in columns)
        experiment = config["nuclear_experiment"]
        for field in ("history_anchor_day", "raw_history_start_day", "incremental_namespace"):
            experiment.pop(field, None)
        experiment.update(input_protocol=solar_wind_input_protocol(), mode="incremental",
            incremental_cache_dir=str(OUTPUT/"_daily_cache"/zone.lower()), candidate_model="solar_wind", production_modified=False)
        if experiment.get("residual_bank_audit"):
            experiment["residual_bank_audit"] = str(work/"snapshot"/Path(experiment["residual_bank_audit"]).name)
        anchor = incumbent.audit["history_anchor_day"]
        prepare_incremental_settings(config, (pd.Timestamp(anchor)+pd.Timedelta(days=730)).date())
        _, selected_anchor, _ = prepare_incremental_settings(config, pd.Timestamp(cfg["delivery_day"]).date())
        if str(selected_anchor) != anchor:
            raise ValueError("Baseline and solar replay cold-start dates differ.")
        _, references = _freeze_reporting(baseline, work, receipt)
        for name, frame in (("incumbent_backtest", incumbent.kalman_view.backtest),
                            ("incumbent_forecast", incumbent.kalman_view.forecast),
                            ("incumbent_residual", incumbent.residual_statistics),
                            ("incumbent_source_forecast", incumbent.source_forecast)):
            path = safe(work/"reference"/(name+".parquet"))
            frame.to_parquet(path)
            references.append({"snapshot_path": str(path), "sha256": sha256(path)})
        audit = {"solar_wind": sources, "wind_dst_policy": cfg["wind_dst_policy"], "wind_gap_policy": cfg["wind_gap_policy"],
            "baseline": str(baseline), "same_history_anchor": anchor,
            "same_model_hyperparameters": True, "added_inputs_only": list(GENERATION_SERIES),
            "no_custom_ramp_or_spike_expert": True,
            "prospective_validation": False, "retrospective_asof_reconstruction": True,
            "test_case_selected_after_market_observation": cfg["delivery_day"] == "2026-09-22", "renewables_already_implicit_in_residual_load": True,
            "production_pit_evidence": False, "production_modified": False,
            "cutoff": "D-1 08:00 civil; historical query-asof, not certified original publication"}
        write_json(safe(work/"source_audit.json"), audit)
        references.append({"snapshot_path": str(work/"source_audit.json"), "sha256": sha256(work/"source_audit.json")})
        resolved = safe(work/"resolved_config.yaml")
        resolved.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if identity(cfg, zone, sources) != signature:
            raise ValueError("Sources or baseline changed during solar snapshotting.")
        write_json(safe(work/"input_snapshot.json"), {"schema_version": 1, "identity": signature,
            "files": files, "reference_files": references, "resolved_config_sha256": sha256(resolved)})
        verify_snapshot(work)
        _verify_reporting(work, zone=zone, day=cfg["delivery_day"], timezone=spec["timezone"])
    return work


def run_zone(cfg, zone, work, args):
    from chronos2_modular.common import build_zone_configs
    from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
    from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle, save_nuclear_result_bundle
    from chronos2_hourly.solar_wind_forecast import run_solar_wind_forecast
    from chronos2_hourly.solar_wind_reporting import render_solar_wind_reports
    config = verify_snapshot(work, zone=zone, delivery_day=cfg["delivery_day"])
    with exclusive_process_lock(work/"run.lock"), run_progress(work, zone, pd.Timestamp(cfg["delivery_day"]), args.action) as progress:
        progress("prepare_inputs")
        spec = build_zone_configs(config, [zone], None, None)[0]
        actual, reporting = _verify_reporting(work, zone=zone, day=cfg["delivery_day"], timezone=spec.timezone)
        data = prepare_nuclear_zone_data(spec, config, work, work/"prepared")
        if args.action == "prepare":
            return {"status": "prepared", "workdir": str(work)}
        reuse = (work/"report_only/frozen_result").is_dir()
        if args.action == "report" or reuse:
            result = load_nuclear_result_bundle(workdir=work)
        else:
            progress("forecast", solar_wind_in_chronos=True, solar_wind_in_residual=True, solar_wind_in_kalman=True)
            print(f"[SolarWind/{zone}] Nouveau replay Chronos + residuel + Kalman, avec quatre courbes solaires et deux courbes eoliennes DE/NL.", flush=True)
            result = run_forecast_with_storage_retry(run_solar_wind_forecast, config=config, data=data, zone=zone,
                delivery_day=cfg["delivery_day"], workdir=work, device=args.device, threads=args.threads, workers=args.workers)
            save_nuclear_result_bundle(result, workdir=work)
        attribution = None
        if cfg["include_attribution"] and args.action != "report" and not args.skip_attribution:
            from chronos2_hourly.nuclear_attribution import prepare_nuclear_attribution
            progress("attribution")
            attribution = prepare_nuclear_attribution(result, data, config, work, device=args.device, threads=args.threads)
        progress("render_reports")
        incumbent = SimpleNamespace(residual_statistics=pd.read_parquet(work/"reference/incumbent_residual.parquet"),
            source_forecast=pd.read_parquet(work/"reference/incumbent_source_forecast.parquet"),
            kalman_view=SimpleNamespace(backtest=pd.read_parquet(work/"reference/incumbent_backtest.parquet"),
                                       forecast=pd.read_parquet(work/"reference/incumbent_forecast.parquet")))
        reports = render_solar_wind_reports(result, data=replace(data, target=actual), zone=zone,
            delivery_day=cfg["delivery_day"], output_directory=work/"reports", incumbent=incumbent,
            storm_archive=work/"reference/reporting", observed_source_audit=reporting,
            source_audit=read(work/"source_audit.json"), attribution_directory=attribution)
        receipt = {"status": "complete", "reports": reports, "result_reused": reuse,
                   "production_modified": False, "activation_performed": False, "audit": result.audit}
        write_json(safe(work/"run_result.json"), receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config/solar_wind.yaml")
    parser.add_argument("--action", choices=("audit", "sync", "prepare", "run", "report", "status"), default="run")
    parser.add_argument("--delivery-day")
    parser.add_argument("--zones", nargs="+", choices=ZONES)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--skip-attribution", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 32:
        parser.error("threads must be between 1 and 32")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_settings(args.config)
    for field in ("delivery_day", "zones"):
        if getattr(args, field) is not None:
            cfg[field] = getattr(args, field)
    validate_settings(cfg)
    pointer = safe(OUTPUT/cfg["delivery_day"]/("latest_"+"_".join(cfg["zones"])+".json"))
    if args.action == "status":
        receipt = read(pointer) if pointer.exists() else {"status": "not_prepared"}
        states = {}
        for zone, raw in receipt.get("workdirs", {}).items():
            path = safe(raw)
            if zone not in cfg["zones"] or path.parent != OUTPUT/cfg["delivery_day"]/zone.lower():
                raise ValueError("Solar status pointer zone/date mismatch.")
            state_path = path/"run_status.json"
            states[zone] = read(state_path) if state_path.exists() else {"status": "prepared"}
        receipt["states"] = states
        print(json.dumps(receipt, indent=2, default=str))
        return 0
    if args.action == "audit":
        print(json.dumps(audit_inputs(cfg), indent=2))
        return 0
    # One mutating batch per delivery; Status/Audit remain read-only.
    with exclusive_process_lock(safe(OUTPUT/"_batch_locks"/(cfg["delivery_day"]+".lock"))):
        return execute(cfg, args, pointer)


def execute(cfg, args, pointer):
    if args.action == "report":
        workdirs = {zone: safe(path) for zone, path in read(pointer)["workdirs"].items()}
        if set(workdirs) != set(cfg["zones"]):
            raise ValueError("Report requires a completed run for exactly these countries.")
    else:
        audit = audit_inputs(cfg)
        from chronos2_hourly.solar_wind_sources import ensure_solar_wind_sources
        if args.action != "sync":
            write_json(pointer, {"status": "syncing_sources", "pid": os.getpid(), "zones": cfg["zones"],
                "workdirs": {}, "production_modified": False, "started_at_utc": str(pd.Timestamp.now(tz="UTC"))})
        try:
            sources = ensure_solar_wind_sources(output_root=SOURCE, start_day=audit["start_day"], end_day=audit["end_day"],
                                           workers=args.workers, sync=True, wind_dst_policy=cfg["wind_dst_policy"],
                                           wind_gap_policy=cfg["wind_gap_policy"])
        except Exception as exc:
            if args.action != "sync":
                write_json(pointer, {"status": "failed", "stage": "sync_sources", "workdirs": {},
                    "error": str(exc), "production_modified": False})
            raise
        if args.action == "sync":
            print(json.dumps(sources, indent=2, default=str))
            return 0
        write_json(pointer, {"status": "preparing", "pid": os.getpid(), "zones": cfg["zones"],
            "workdirs": {}, "production_modified": False})
        try:
            workdirs = {zone: prepare(cfg, zone, sources) for zone in cfg["zones"]}
        except Exception as exc:
            write_json(pointer, {"status": "failed", "stage": "prepare", "workdirs": {},
                "error": str(exc), "production_modified": False})
            raise
        write_json(pointer, {"status": "prepared", "workdirs": workdirs, "production_modified": False})
    results = {}
    for zone, work in workdirs.items():
        write_json(pointer, {"status": "running", "active_zone": zone, "workdirs": workdirs,
                             "results": results, "production_modified": False})
        try:
            results[zone] = run_zone(cfg, zone, work, args)
        except Exception as exc:
            write_json(pointer, {"status": "failed", "failed_zone": zone, "workdirs": workdirs,
                                 "results": results, "error": str(exc), "production_modified": False})
            raise
    write_json(pointer, {"status": "prepared" if args.action == "prepare" else "complete", "workdirs": workdirs,
                         "results": results, "production_modified": False})
    if args.action == "prepare":
        print(json.dumps({"status": "prepared", "workdirs": workdirs, "production_modified": False}, default=str, indent=2))
    if args.action != "prepare":
        index = safe(pointer.parent/("solar_wind_"+"_".join(cfg["zones"])+"_index.html"))
        links = []
        for zone, result in results.items():
            for model, raw in result["reports"].items():
                path = Path(raw)
                if path.suffix == ".html":
                    href = os.path.relpath(path, index.parent).replace("\\", "/")
                    links.append(f'<li><a href="{html.escape(href, quote=True)}">{html.escape(zone+" — "+model)}</a></li>')
        index.write_text('<!doctype html><meta charset="utf-8"><title>NYX SolarWind DE/NL</title>'
            '<style>body{font:18px system-ui;max-width:900px;margin:4rem auto;background:#111827;color:#e5e7eb}'
            'a{color:#7dd3fc}li{margin:1rem}</style><h1>NYX nucléaire FR + solaire CWE + éolien DE/NL</h1>'
            '<p>Quatre courbes solaires CWE et deux courbes éoliennes DE/NL dans Chronos-2, le correcteur et Kalman. Reconstitution expérimentale as-of J-1 08 h, non prospective ; production inchangée.</p><ul>'
            +''.join(links)+'</ul>', encoding="utf-8")
        print(json.dumps({"index": str(index), "reports": {z: r["reports"] for z, r in results.items()},
                          "production_modified": False}, default=str, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        logging.exception("[SolarWind] %s — production unchanged", exc)
        raise SystemExit(1)
