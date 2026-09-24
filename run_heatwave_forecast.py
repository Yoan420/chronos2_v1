"""Isolated five-country forecast-temperature challenger; operational runs are read-only."""
from __future__ import annotations

import os
# Some PEFT/Transformers versions probe adapter_config even when the base
# loader receives local_files_only. This CLI uses frozen local weights only.
# Set these before importing Hugging Face, without touching the parent shell.
if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import yaml

from run_nuclear_forecast import sha256, write_json, run_progress, run_forecast_with_storage_retry
from chronos2_hourly.process_lock import exclusive_process_lock
from run_nuclear_cwe_forecast import _read, _day, _inside, _copy_checked, _verify_reporting

ROOT = Path(__file__).resolve().parent


def load_settings(path: Path) -> dict:
    from chronos2_hourly.heatwave_features import HeatwaveFeatureConfig
    path = path.resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or settings.get("schema_version") != 1:
        raise ValueError("Configuration Heatwave schema_version=1 requise.")
    project = (path.parent / settings.get("project_root", "..")).resolve()
    if project != ROOT:
        raise ValueError("Le projet doit etre celui du launcher courant.")
    settings.update(config_path=str(path), config_sha256=sha256(path), project_root=project)
    for key, default in (("output_root", "runs/experiments/heatwave_v1"),
                         ("source_root", "data/pit/heatwave/saturn_daily_v1"),
                         ("baseline_root", "runs/experiments/nuclear_forecast_v1")):
        settings[key] = (project / Path(settings.get(key, default))).resolve()
    for key, allowed in (("output_root", project / "runs/experiments/heatwave_v1"),
                         ("source_root", project / "data/pit/heatwave")):
        if settings[key] != allowed.resolve():
            _inside(settings[key], allowed)
    _inside(settings["baseline_root"], project / "runs/experiments")
    if settings["baseline_root"] == settings["output_root"] or settings["output_root"].is_relative_to(settings["baseline_root"]):
        raise ValueError("La reference et le candidat doivent rester dans des espaces distincts.")
    kind = settings.setdefault("baseline_kind", "nuclear_fr")
    if kind not in {"nuclear_fr", "nuclear_cwe"}:
        raise ValueError("baseline_kind doit etre nuclear_fr ou nuclear_cwe.")
    settings["features"] = HeatwaveFeatureConfig.from_mapping(settings.get("features")).to_dict()
    settings["delivery_day"] = _day(settings["delivery_day"])
    settings["source_start_day"] = _day(settings.get("source_start_day", "2024-06-30"))
    if settings["source_start_day"] > settings["delivery_day"]:
        raise ValueError("source_start_day doit preceder la livraison.")
    return settings

def workdir_for(settings: dict, day: str, zone: str) -> Path:
    if zone not in {"FR", "DE", "BE", "NL", "ES"}:
        raise ValueError(f"Zone non supportee : {zone}")
    return _inside(settings["output_root"] / _day(day) / zone.lower(), settings["output_root"])


def baseline_for(settings: dict, day: str, zone: str) -> Path:
    path = settings["baseline_root"] / _day(day) / zone.lower()
    if settings.get("baseline_kind", "nuclear_fr") == "nuclear_fr":
        path /= "civil_pit_v2"
    return _inside(path, settings["baseline_root"])


def _protocol(settings: dict) -> str:
    from chronos2_hourly.heatwave_forecast import heatwave_input_protocol
    return heatwave_input_protocol(settings.get("features"))


def _freeze_reporting(baseline: Path, workdir: Path, receipt: dict) -> tuple[dict, list]:
    """Relocate frozen reporting, including already relocated CWE snapshots."""
    audit = deepcopy(receipt["reporting_sources"])
    original = _inside(Path(audit["snapshot_directory"]), baseline)
    if audit.get("status") != "complete" or not audit.get("storm_dashboard"):
        raise ValueError("Observations et Storm geles complets requis.")
    destination = workdir / "reference/reporting"
    # Never collide with the audit chain copied by an earlier experiment.
    renamed_audit = "parent_statistics_history_audit.json"
    while (original / renamed_audit).exists():
        renamed_audit = "parent_" + renamed_audit
    records = []
    for source in sorted(original.rglob("*")):
        if source.is_file():
            _inside(source, original)
            relative = source.relative_to(original)
            if relative == Path("statistics_history_audit.json"):
                relative = Path(renamed_audit)
            records.append(_copy_checked(source, destination / relative))
    audit["snapshot_directory"] = str(destination.resolve())
    audit["observed"]["artifact_path"] = str(_inside(destination / audit["observed"]["relative_artifact_path"], destination))
    path = destination / "statistics_history_audit.json"
    write_json(path, audit)
    records.append({"snapshot_path": str(path), "sha256": sha256(path),
                    "transformation": "local_path_relocation_only"})
    return audit, records


def _verify_snapshot(workdir: Path, settings: dict, day: str, zone: str) -> dict:
    manifest = _read(workdir / "input_snapshot.json")
    expected = {"zone": zone, "delivery_day": day, "config_sha256": settings["config_sha256"],
                "input_protocol": _protocol(settings), "orchestrator_sha256": sha256(Path(__file__))}
    if any(manifest["identity"].get(k) != v for k, v in expected.items()):
        raise ValueError("Snapshot Heatwave incompatible : recette/code/date changes. Utiliser un nouvel output_root Heatwave.")
    for item in manifest["files"] + manifest["reference_files"]:
        path = _inside(Path(item["snapshot_path"]), workdir)
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Snapshot Heatwave modifie : {path}")
    if sha256(workdir / "resolved_config.yaml") != manifest["resolved_config_sha256"]:
        raise ValueError("Configuration gelee Heatwave modifiee.")
    return yaml.safe_load((workdir / "resolved_config.yaml").read_text(encoding="utf-8"))




def prepare_snapshot(settings: dict, day: str, zone: str, sources: dict) -> dict:
    from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
    from chronos2_hourly.nuclear_incremental import prepare_incremental_settings
    from chronos2_hourly.heatwave_features import (
        HeatwaveFeatureConfig, build_heatwave_features, feature_metadata,
        heatwave_feature_aliases, temperature_aliases)
    from chronos2_hourly.heatwave_sources import audit_temperature_store
    from chronos2_hourly.nuclear_sources import NUCLEAR_ALIAS, audit_nuclear_store

    workdir = workdir_for(settings, day, zone)
    if (workdir / "input_snapshot.json").exists():
        return _verify_snapshot(workdir, settings, day, zone)
    baseline = baseline_for(settings, day, zone)
    receipt_path = baseline / "run_result.json"
    receipt_sha = sha256(receipt_path)
    receipt = _read(receipt_path)
    # Legacy nuclear receipts omit status; the frozen bundle validator below
    # verifies completeness and integrity independently of this cosmetic field.
    if receipt.get("status") not in (None, "complete"):
        raise ValueError("Un run de reference termine et valide est requis.")
    incumbent = load_nuclear_result_bundle(workdir=baseline)
    original_manifest_sha = sha256(baseline / "input_snapshot.json")
    prior = _read(baseline / "input_snapshot.json")
    if sha256(baseline / "resolved_config.yaml") != prior["resolved_config_sha256"]:
        raise ValueError("Configuration incumbent modifiee apres validation du resultat.")
    config = yaml.safe_load((baseline / "resolved_config.yaml").read_text(encoding="utf-8"))
    if sha256(baseline / "resolved_config.yaml") != prior["resolved_config_sha256"]:
        raise ValueError("Configuration incumbent modifiee pendant sa lecture.")
    if incumbent.audit["zone"] != zone or incumbent.audit["delivery_day"] != day:
        raise ValueError("Comparateur incoherent avec la livraison demandee.")
    if config["model"]["model_id"] != "amazon/chronos-2":
        raise ValueError("Cette experience requiert la meme base Chronos-2 que nuclear_kalman.")
    anchor = _day(incumbent.audit["history_anchor_day"])
    features = HeatwaveFeatureConfig.from_mapping(settings.get("features"))
    aliases = heatwave_feature_aliases(features)
    files, paths = [], {}
    configured_paths = {"target": config["zones"][zone]["target"]["file"], **config["data"]["pit_files"]}
    for item in prior["files"]:
        source = _inside(Path(item["snapshot"]), baseline / "snapshot")
        record = _copy_checked(source, workdir / "snapshot" / source.name, expected=item["sha256"])
        files.append(record)
        for alias, original in configured_paths.items():
            if Path(original).resolve() == source:
                paths[alias] = record["snapshot_path"]
    if set(paths) != set(configured_paths):
        raise ValueError("Snapshot incumbent incomplet pour les sources configurees.")
    if set(aliases).intersection(paths):
        raise ValueError("Le comparateur contient deja ces temperatures; un test incremental distinct est requis.")

    raw_sources, source_records = {}, {}
    for alias in temperature_aliases(features):
        source = Path(sources[alias]["path"])
        country = alias[:2].upper()
        first = settings.get("source_start_day", "2024-06-30")
        audit = audit_temperature_store(source, country, first, day)
        record = _copy_checked(source, workdir / "snapshot" / "temperature_sources" / source.name)
        record["alias"] = alias + "_raw_source"
        files.append(record)
        sidecar = source.with_name(source.name + ".audit.json")
        if not sidecar.is_file():
            raise ValueError(f"Audit source temperature absent : {sidecar}")
        files.append(_copy_checked(sidecar, Path(record["snapshot_path"] + ".audit.json")))
        raw_sources[alias] = pd.read_parquet(record["snapshot_path"])
        source_records[country] = {"path": record["snapshot_path"],
                                  "specification": sources[alias]["specification"], "audit": audit}
    # One shared 08:00 Paris weather contract; output zones retain their own
    # timezone. These five zones share the same physical civil cutoff.
    wide, feature_audit = build_heatwave_features(raw_sources, start_day=anchor, end_day=day,
                                                 timezone="Europe/Paris", config=features)
    # One canonical wide vintage file; each SeriesSpec selects its exact alias.
    feature_path = workdir / "snapshot/heatwave_features.parquet"
    if feature_path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(feature_path), wide)
    else:
        wide.to_parquet(feature_path, index=False)
    files.append({"source": "derived_from_pinned_temperature_forecasts", "snapshot": str(feature_path),
                  "snapshot_path": str(feature_path), "sha256": sha256(feature_path)})
    paths.update({alias: str(feature_path) for alias in aliases})

    spec = config["zones"][zone]
    spec["target"]["file"] = paths["target"]
    metadata = feature_metadata(features)
    covars = spec["covariates"]
    for alias in aliases:
        is_raw = alias in temperature_aliases(features)
        covars[alias] = {
            "enabled": True, "source": "pit_parquet",
            "series": sources[alias]["specification"]["series"] if is_raw else f"derived.heatwave.{alias}",
            "description": metadata[alias]["semantic"], "unit": metadata[alias]["unit"],
            "fill_method": "none", "fill_limit": 0, "minimum_coverage": 0.01,
            "value_col": alias, "future": {"known_future": True, "strategies": ["oracle"]},
            "information_type": "daily_forecast_temperature_index" if is_raw else "derived_causal_forecast_signal",
            "daily_broadcast": True,
        }
    for alias, covar in covars.items():
        if covar.get("enabled", True):
            covar["pit_file"] = paths[alias]
            if "file" in covar:
                covar["file"] = paths[alias]
    config["data"].update(pit_files={k: paths[k] for k,v in covars.items() if v.get("enabled", True)},
                          pit_vintage_dir=str(workdir / "snapshot"), cache_dir=str(workdir / "snapshot/cache"))
    config["output"]["directory"] = str(workdir)
    config["report"]["title"] = f"Chronos-2 nucleaire + temperatures / chaleur persistante {zone}"
    columns = config["hourly"]["feature_engineering"].get("covariate_columns")
    if columns is not None:
        columns.extend(f"known_{a}_oracle" for a in aliases if f"known_{a}_oracle" not in columns)
    experiment = config["nuclear_experiment"]
    for key in ("history_anchor_day", "raw_history_start_day", "incremental_namespace"):
        experiment.pop(key, None)
    experiment.update(input_protocol=_protocol(settings), mode="incremental",
        incremental_cache_dir=str(settings["output_root"] / "_daily_cache" / zone.lower()),
        candidate_model="heatwave", production_modified=False, heatwave_features=features.to_dict(),
        heatwave_base_variant=settings.get("baseline_kind", "nuclear_fr"))
    if experiment.get("residual_bank_audit"):
        experiment["residual_bank_audit"] = str(workdir / "snapshot" / Path(experiment["residual_bank_audit"]).name)
    prepare_incremental_settings(config, (pd.Timestamp(anchor) + pd.Timedelta(days=730)).date())
    _, new_anchor, _ = prepare_incremental_settings(config, day)
    if str(new_anchor) != anchor:
        raise ValueError("Ancre de calibration differente du modele de reference.")

    if "reporting_sources" not in receipt:
        receipt["reporting_sources"] = _read(baseline / "reference/provenance.json")["reporting_sources"]
    reporting_audit, reference_files = _freeze_reporting(baseline, workdir, receipt)
    _verify_reporting(workdir, zone=zone, day=day, timezone=spec["timezone"])
    reference = workdir / "reference"
    for name, frame in (("incumbent_backtest", incumbent.kalman_view.backtest),
                        ("incumbent_forecast", incumbent.kalman_view.forecast)):
        path = reference / f"{name}.parquet"
        frame.to_parquet(path, index=True)
        reference_files.append({"snapshot_path": str(path), "sha256": sha256(path)})
    provenance = {"baseline_directory": str(baseline), "baseline_run_result_sha256": receipt_sha,
        "baseline_snapshot_sha256": original_manifest_sha,
        "baseline_frozen_result_manifest_sha256": sha256(baseline / "report_only/frozen_result/manifest.json"),
        "baseline_audit": incumbent.audit, "reporting_sources": reporting_audit,
        "same_hyperparameters": True, "same_history_anchor": anchor,
        "production_modified": False, "promotion_performed": False}
    fr_audit = audit_nuclear_store(Path(paths[NUCLEAR_ALIAS]), anchor, day)
    if fr_audit.get("blockers"):
        raise ValueError(f"Nucleaire FR incomplet : {fr_audit['blockers']}")
    source_audit = {"temperatures": source_records, "nuclear_fr": fr_audit,
        "baseline_kind": settings.get("baseline_kind", "nuclear_fr"),
        "incumbent_label": "Nuclear CWE Kalman" if settings.get("baseline_kind") == "nuclear_cwe" else "Nuclear Kalman",
        "forecast_semantics": "Daily national temperature FORECAST indices, Celsius; not Tmax, not observed weather",
        "cutoff": "D-1 08:00 Europe/Paris", "production_pit_evidence": False,
        "publication_evidence": "Saturn as-of queries; provider publication timestamps unavailable"}
    for name, payload in (("reference/provenance.json", provenance), ("source_audit.json", source_audit),
                          ("feature_audit.json", feature_audit)):
        path = workdir / name
        write_json(path, payload)
        reference_files.append({"snapshot_path": str(path), "sha256": sha256(path)})
    resolved = workdir / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if sha256(receipt_path) != receipt_sha or sha256(baseline / "input_snapshot.json") != original_manifest_sha:
        raise ValueError("Le rapport de reference a change pendant sa copie; preparation arretee.")
    write_json(workdir / "input_snapshot.json", {
        "schema_version": 1, "identity": {"zone": zone, "delivery_day": day,
            "config_sha256": settings["config_sha256"], "input_protocol": _protocol(settings),
            "orchestrator_sha256": sha256(Path(__file__))},
        "files": files, "reference_files": reference_files, "resolved_config_sha256": sha256(resolved),
        "production_modified": False})
    return _verify_snapshot(workdir, settings, day, zone)

def run_zone(settings: dict, day: str, zone: str, args) -> dict:
    from chronos2_modular.common import build_zone_configs
    from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
    from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle, save_nuclear_result_bundle
    from chronos2_hourly.heatwave_forecast import run_heatwave_forecast
    from chronos2_hourly.heatwave_reporting import render_heatwave_reports
    workdir = workdir_for(settings, day, zone)
    config = _verify_snapshot(workdir, settings, day, zone)
    with run_progress(workdir, zone, pd.Timestamp(day), args.action) as progress:
        progress("prepare_inputs")
        spec = build_zone_configs(config, [zone], None, None)[0]
        target, source = _verify_reporting(workdir, zone=zone, day=day, timezone=spec.timezone)
        data = prepare_nuclear_zone_data(spec, config, workdir, workdir / "prepared")
        if args.action == "prepare":
            from chronos2_hourly.heatwave_forecast import _prepare_inputs
            # Exercise native Europe/Paris DST indices, not a simplified UTC fixture.
            _prepare_inputs(config, data, zone, pd.Timestamp(day).date())
            return {"status": "prepared", "zone": zone, "workdir": str(workdir), "production_modified": False}
        reused = (workdir / "report_only/frozen_result").is_dir()
        if args.action == "report" or reused:
            progress("load_frozen_result")
            result = load_nuclear_result_bundle(workdir=workdir)
        else:
            progress("forecast", history_anchor=config["nuclear_experiment"]["history_anchor_day"])
            print(f"[Heatwave/{zone}] nouveau replay Chronos + correcteur + Kalman; checkpoints isoles reutilisables.", flush=True)
            result = run_forecast_with_storage_retry(run_heatwave_forecast, config=config, data=data,
                zone=zone, delivery_day=day, workdir=workdir, device=args.device,
                threads=args.threads, workers=args.workers)
            save_nuclear_result_bundle(result, workdir=workdir)
        attribution_directory = None
        attribution = {"status": "unavailable", "reason": "no_matching_attribution"}
        if not args.skip_attribution and settings.get("include_attribution", True):
            if args.action == "report":
                prior = _read(workdir / "run_result.json") if (workdir / "run_result.json").exists() else {}
                attribution = prior.get("variable_attribution", attribution)
                if attribution.get("status") == "complete":
                    attribution_directory = Path(attribution["directory"])
            else:
                try:
                    progress("attribution")
                    from chronos2_hourly.nuclear_attribution import prepare_nuclear_attribution
                    attribution_directory = prepare_nuclear_attribution(result, data, config, workdir,
                                                                          device=args.device, threads=args.threads)
                    attribution = {"status": "complete", "directory": str(attribution_directory)}
                except Exception as error:
                    attribution = {"status": "failed", "error": str(error)}
                    print(f"[Heatwave/{zone}] attribution indisponible : {error}", flush=True)
        progress("render_reports")
        reference = workdir / "reference"
        incumbent = SimpleNamespace(backtest=pd.read_parquet(reference / "incumbent_backtest.parquet"),
                                    forecast=pd.read_parquet(reference / "incumbent_forecast.parquet"))
        reports = render_heatwave_reports(result, incumbent=incumbent, data=replace(data, target=target),
            zone=zone, delivery_day=day, output_directory=workdir / "reports",
            source_audit=_read(workdir / "source_audit.json"), feature_audit=_read(workdir / "feature_audit.json"),
            storm_archive=reference / "reporting",
            observed_source_audit=source, attribution_directory=attribution_directory)
        receipt = {"status": "complete", "zone": zone, "delivery_day": day, "reports": reports,
                   "audit": result.audit, "variable_attribution": attribution, "forecast_result_reused": reused,
                   "production_modified": False, "promotion_performed": False}
        write_json(workdir / "run_result.json", receipt)
        return receipt


def read_status(workdir: Path, zone: str) -> dict:
    """Read-only status: a stale status file is not proof of a running model."""
    path = workdir / "run_status.json"
    progress = _read(path) if path.exists() else None
    state = {"zone": zone, "workdir": str(workdir), "progress": progress,
             "reports_available": (workdir / "run_result.json").is_file()}
    if progress and progress.get("status") == "running":
        import psutil
        try:
            process = psutil.Process(int(progress["pid"]))
            started = pd.Timestamp(progress["started_at_utc"]).timestamp()
            same = (process.create_time() <= started + 1 and
                    any(Path(arg).name == Path(__file__).name for arg in process.cmdline()))
            state["process_active"] = same
            if not same:
                state["warning"] = "Statut ancien; ce PID ne correspond plus au lancement."
        except psutil.NoSuchProcess:
            state.update(process_active=False, warning="Processus termine sans mise a jour du statut.")
        except (psutil.AccessDenied, OSError):
            state.update(process_active=None, warning="Activite du processus non verifiable.")
    resolved = workdir / "resolved_config.yaml"
    if resolved.is_file():
        config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        namespace = Path(config["nuclear_experiment"]["incremental_namespace"])
        experiment_root = (ROOT / "runs/experiments/heatwave_v1").resolve()
        _inside(namespace, experiment_root)
        state["cache_files_not_validation_proof"] = {
            "chronos_daily": len(list((namespace / "chronos" / zone.lower()).glob("*/*.json"))),
            "residual_daily": len(list((namespace / "residual").glob("*.json"))),
            "kalman_daily": len(list((namespace / "kalman_rolling").rglob("*.pickle"))),
        }
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["run", "prepare", "sync", "report", "status"], default="run")
    parser.add_argument("--config", type=Path, default=ROOT / "config/heatwave.yaml")
    parser.add_argument("--delivery-day")
    parser.add_argument("--zones", nargs="+", choices=["FR", "DE", "BE", "NL", "ES"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-attribution", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 64 or not 1 <= args.workers <= 16:
        parser.error("threads 1..64 et workers 1..16 requis")
    settings = load_settings(args.config)
    day = _day(args.delivery_day or settings["delivery_day"])
    zones = list(dict.fromkeys(args.zones or settings.get("zones", ["FR"])))
    if not zones:
        raise ValueError("Au moins une zone requise.")
    if args.action == "status":
        for zone in zones:
            workdir = workdir_for(settings, day, zone)
            print(json.dumps(read_status(workdir, zone), ensure_ascii=False))
        return 0
    with exclusive_process_lock(settings["output_root"] / ".heatwave.lock"):
        if args.action == "sync":
            from chronos2_hourly.heatwave_sources import materialize_temperature_sources
            sources = materialize_temperature_sources(settings["source_root"], settings["source_start_day"],
                                                      day, workers=min(args.workers, 2))
            print(json.dumps(sources, indent=2, default=str))
            return 0
        # Preflight every requested zone before any expensive model calculation.
        starts, needs = [], []
        for zone in zones:
            workdir = workdir_for(settings, day, zone)
            if (workdir / "input_snapshot.json").exists():
                _verify_snapshot(workdir, settings, day, zone)
            elif args.action == "report":
                raise ValueError(f"{zone} : lancer Prepare / Run avant Report.")
            else:
                from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
                base = load_nuclear_result_bundle(workdir=baseline_for(settings, day, zone))
                starts.append(base.audit["history_anchor_day"])
                needs.append(zone)
        if needs:
            from chronos2_hourly.heatwave_sources import materialize_temperature_sources
            start = settings["source_start_day"]
            if starts and start > min(starts):
                raise ValueError("Le prefixe temperature doit commencer avant la calibration de reference.")
            sources = materialize_temperature_sources(settings["source_root"], start, day, workers=min(args.workers, 2))
            for zone in needs:
                prepare_snapshot(settings, day, zone, sources)
        for zone in zones:
            result = run_zone(settings, day, zone, args)
            print(json.dumps({k: result[k] for k in ("status", "zone", "reports", "workdir", "production_modified")
                              if k in result}, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"[Heatwave] ECHEC : {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
