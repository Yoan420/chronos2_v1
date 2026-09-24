"""Frozen, independent CWE nuclear comparison. No operational cache or export writes."""
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
import shutil
import sys
from types import SimpleNamespace

import pandas as pd
import yaml

from run_nuclear_forecast import sha256, write_json, run_progress, run_forecast_with_storage_retry
from chronos2_hourly.process_lock import exclusive_process_lock

ROOT = Path(__file__).resolve().parent
ALIASES = ("be_nuclear_available_gw", "nl_nuclear_available_gw")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _day(value: str) -> str:
    import re
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("DeliveryDay doit etre une date YYYY-MM-DD.")
    return pd.Timestamp(value).date().isoformat()


def _inside(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Chemin hors du perimetre isole autorise : {path}")
    return path


def load_settings(path: Path) -> dict:
    path = path.resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or settings.get("schema_version") != 1:
        raise ValueError("Configuration Nuclear CWE schema_version=1 requise.")
    project = (path.parent / settings.get("project_root", "..")).resolve()
    if project != ROOT:
        raise ValueError("Le projet doit etre celui du launcher courant.")
    settings.update(config_path=str(path), config_sha256=sha256(path), project_root=project)
    for key, default in (("output_root", "runs/experiments/nuclear_cwe_v1"),
                         ("source_root", "data/pit/nuclear_cwe"),
                         ("baseline_root", "runs/experiments/nuclear_forecast_v1")):
        value = Path(settings.get(key, default))
        settings[key] = (project / value).resolve()
    # A distinct namespace is mandatory even for a custom configuration.
    allowed = project / "runs/experiments/nuclear_cwe_v1"
    if settings["output_root"] != allowed.resolve():
        _inside(settings["output_root"], allowed)
    allowed = project / "data/pit/nuclear_cwe"
    if settings["source_root"] != allowed.resolve():
        _inside(settings["source_root"], allowed)
    _inside(settings["baseline_root"], project / "runs/experiments")
    protocol = str(settings.get("baseline_protocol", "civil_pit_v2"))
    if protocol != "civil_pit_v2":
        raise ValueError("Le comparateur doit respecter le protocole civil_pit_v2.")
    settings["delivery_day"] = _day(settings["delivery_day"])
    return settings


def workdir_for(settings: dict, day: str, zone: str) -> Path:
    if zone not in {"FR", "DE", "BE", "NL"}:
        raise ValueError(f"Zone non supportee : {zone}")
    return _inside(settings["output_root"] / _day(day) / zone.lower(), settings["output_root"])


def baseline_for(settings: dict, day: str, zone: str) -> Path:
    return _inside(settings["baseline_root"] / _day(day) / zone.lower() /
                   settings.get("baseline_protocol", "civil_pit_v2"), settings["baseline_root"])


def _copy_checked(source: Path, destination: Path, *, expected: str | None = None) -> dict:
    source, destination = source.resolve(), destination.resolve()
    digest = sha256(source)
    if expected is not None and digest != expected:
        raise ValueError(f"SHA source divergent : {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError(f"Copie isolee deja presente mais differente : {destination}")
    else:
        # Only a new destination can be created. Incomplete copies never validate.
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
    if sha256(source) != digest or sha256(destination) != digest:
        raise ValueError(f"Source modifiee pendant sa copie : {source}")
    return {"source_path": str(source), "snapshot_path": str(destination),
            "source": str(source), "snapshot": str(destination), "sha256": digest}


def _verify_snapshot(workdir: Path, settings: dict, day: str, zone: str) -> dict:
    from chronos2_hourly.nuclear_cwe_forecast import nuclear_cwe_input_protocol
    manifest = _read(workdir / "input_snapshot.json")
    expected = {"zone": zone, "delivery_day": day, "config_sha256": settings["config_sha256"],
                "input_protocol": nuclear_cwe_input_protocol()}
    if any(manifest["identity"].get(k) != v for k, v in expected.items()):
        raise ValueError("Snapshot CWE incompatible : recette/code/date changes. Utiliser un nouvel output_root CWE.")
    for item in manifest["files"] + manifest["reference_files"]:
        path = _inside(Path(item["snapshot_path"]), workdir)
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Snapshot CWE modifie : {path}")
    if sha256(workdir / "resolved_config.yaml") != manifest["resolved_config_sha256"]:
        raise ValueError("Configuration gelee CWE modifiee.")
    return yaml.safe_load((workdir / "resolved_config.yaml").read_text(encoding="utf-8"))


def _freeze_reporting(baseline: Path, workdir: Path, receipt: dict) -> tuple[dict, list]:
    audit = deepcopy(receipt["reporting_sources"])
    original = Path(audit["snapshot_directory"]).resolve()
    _inside(original, baseline)
    if audit.get("status") != "complete" or not audit.get("storm_dashboard"):
        raise ValueError("Comparaison exacte impossible sans observations et Storm geles du rapport precedent.")
    copied = []
    dest = workdir / "reference/reporting"
    for source in sorted(original.rglob("*")):
        if source.is_file():
            _inside(source, original)
            relative = source.relative_to(original)
            if relative == Path("statistics_history_audit.json"):
                relative = Path("original_statistics_history_audit.json")
            copied.append(_copy_checked(source, dest / relative))
    audit["snapshot_directory"] = str(dest.resolve())
    audit["observed"]["artifact_path"] = str((dest / audit["observed"]["relative_artifact_path"]).resolve())
    # The extraction and numerical bytes do not change: only local artifact paths do.
    original_audit = dest / "statistics_history_audit.json"
    write_json(original_audit, audit)
    copied.append({"snapshot_path": str(original_audit.resolve()), "sha256": sha256(original_audit),
                   "transformation": "local_path_relocation_only"})
    return audit, copied


def _verify_reporting(workdir: Path, *, zone: str, day: str, timezone: str) -> tuple[pd.Series, dict]:
    from chronos2_hourly.nuclear_reporting_refresh import verify_refreshed_observations
    from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
    directory = workdir / "reference/reporting"
    audit = _read(directory / "statistics_history_audit.json")
    path = _inside(Path(audit["observed"]["artifact_path"]), directory)
    observed = pd.read_parquet(path)
    target = pd.Series(observed.actual.to_numpy(), index=pd.DatetimeIndex(observed.timestamp), name="actual")
    verify_refreshed_observations(target, audit, zone=zone, timezone=timezone, delivery_day=day)
    if _load_verified_snapshot(directory, zone=zone, timezone=timezone) is None:
        raise ValueError("Snapshot Storm verifie absent; comparaison refusee avant calcul.")
    return target, audit


def prepare_snapshot(settings: dict, day: str, zone: str, sources: dict) -> dict:
    from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
    from chronos2_hourly.nuclear_incremental import prepare_incremental_settings
    from chronos2_hourly.nuclear_cwe_forecast import nuclear_cwe_input_protocol
    from chronos2_hourly.nuclear_sources import NUCLEAR_ALIAS, audit_nuclear_store

    workdir = workdir_for(settings, day, zone)
    if (workdir / "input_snapshot.json").exists():
        return _verify_snapshot(workdir, settings, day, zone)
    baseline = baseline_for(settings, day, zone)
    receipt_path = baseline / "run_result.json"
    receipt_sha = sha256(receipt_path)
    receipt = _read(receipt_path)
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
    files, paths = [], {}
    configured_paths = {"target": config["zones"][zone]["target"]["file"], **config["data"]["pit_files"]}
    source_aliases = {str(Path(path).resolve()): alias for alias, path in configured_paths.items()}
    for item in prior["files"]:
        source = _inside(Path(item["snapshot"]), baseline / "snapshot")
        record = _copy_checked(source, workdir / "snapshot" / source.name, expected=item["sha256"])
        files.append(record)
        alias = source_aliases.get(str(source))
        if alias is not None:
            record["alias"] = alias
            paths[alias] = record["snapshot_path"]
    if set(paths) != set(configured_paths):
        raise ValueError("Snapshot incumbent incomplet pour les sources configurees.")
    for alias in ALIASES:
        source = Path(sources[alias]["path"])
        record = _copy_checked(source, workdir / "snapshot" / f"{alias}.parquet")
        record["alias"] = alias
        paths[alias] = record["snapshot_path"]
        files.append(record)
        sidecar = source.with_suffix(source.suffix + ".audit.json")
        if sidecar.exists():
            record = _copy_checked(sidecar, workdir / "snapshot" / f"{alias}.parquet.audit.json")
            record["alias"] = alias + "_source_audit"
            files.append(record)
    spec = config["zones"][zone]
    spec["target"]["file"] = paths["target"]
    covars = spec["covariates"]
    for alias in ALIASES:
        country = alias[:2]
        covars[alias] = {
            "enabled": True, "source": "pit_parquet",
            "series": f"power.nrjscan.{country}.3mv.availability.pmax.type.nuclear.gw",
            "description": f"{country.upper()} capacite nucleaire disponible prevue (Pmax GW), pas generation",
            "fill_method": "none", "fill_limit": 0, "minimum_coverage": 0.01,
            "future": {"known_future": True, "strategies": ["oracle"]},
            "information_type": "capacity_forecast", "unit": "GW", "daily_broadcast": True,
        }
    for alias, covar in covars.items():
        if covar.get("enabled", True):
            covar["pit_file"] = paths[alias]
            if "file" in covar:
                covar["file"] = paths[alias]
    config["data"].update(pit_files={k: paths[k] for k, v in covars.items() if v.get("enabled", True)},
                          pit_vintage_dir=str(workdir / "snapshot"), cache_dir=str(workdir / "snapshot/cache"))
    config["output"]["directory"] = str(workdir)
    config["report"]["title"] = f"Chronos-2 nucleaire CWE {zone} — candidat independant"
    columns = config["hourly"]["feature_engineering"].get("covariate_columns")
    if columns is not None:
        for alias in ALIASES:
            known = f"known_{alias}_oracle"
            if known not in columns:
                columns.append(known)
    experiment = config["nuclear_experiment"]
    for key in ("history_anchor_day", "raw_history_start_day", "incremental_namespace"):
        experiment.pop(key, None)
    experiment.update(input_protocol=nuclear_cwe_input_protocol(), mode="incremental",
                      incremental_cache_dir=str(settings["output_root"] / "_daily_cache" / zone.lower()),
                      candidate_model="nuclear_cwe", production_modified=False)
    if experiment.get("residual_bank_audit"):
        experiment["residual_bank_audit"] = str(workdir / "snapshot" / Path(experiment["residual_bank_audit"]).name)
    # Initialize a NEW semantic epoch at the incumbent cold-start date, not at today.
    prepare_incremental_settings(config, (pd.Timestamp(anchor) + pd.Timedelta(days=730)).date())
    _, new_anchor, _ = prepare_incremental_settings(config, day)
    if str(new_anchor) != anchor:
        raise ValueError("Ancre de calibration differente du modele de reference.")
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
    if receipt.get("exports", {}).get("kalman"):
        path = Path(receipt["exports"]["kalman"])
        provenance["reference_html"] = {"path": str(path), "sha256": sha256(path)}
    write_json(reference / "provenance.json", provenance)
    reference_files.append({"snapshot_path": str(reference / "provenance.json"),
                            "sha256": sha256(reference / "provenance.json")})
    fr_audit = audit_nuclear_store(Path(paths[NUCLEAR_ALIAS]), anchor, day)
    if fr_audit.get("blockers"):
        raise ValueError(f"Nucleaire FR incomplet : {fr_audit['blockers']}")
    source_audit = {"FR": fr_audit, **{a[:2].upper(): sources[a] for a in ALIASES},
                    "forecast_semantics": "FR generation; BE/NL available Pmax capacity, daily broadcast",
                    "cutoff": "D-1 08:00 Europe/Paris", "production_pit_evidence": False,
                    "structural_zero_no_model_channel": ["DE", "AT", "LU"]}
    write_json(workdir / "source_audit.json", source_audit)
    reference_files.append({"snapshot_path": str(workdir / "source_audit.json"),
                            "sha256": sha256(workdir / "source_audit.json")})
    resolved = workdir / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if sha256(receipt_path) != receipt_sha or sha256(baseline / "input_snapshot.json") != original_manifest_sha:
        raise ValueError("Le rapport de reference a change pendant sa copie; preparation arretee.")
    write_json(workdir / "input_snapshot.json", {
        "schema_version": 1, "identity": {"zone": zone, "delivery_day": day,
            "config_sha256": settings["config_sha256"], "input_protocol": nuclear_cwe_input_protocol()},
        "files": files, "reference_files": reference_files, "resolved_config_sha256": sha256(resolved),
        "production_modified": False,
    })
    return _verify_snapshot(workdir, settings, day, zone)


def run_zone(settings: dict, day: str, zone: str, args) -> dict:
    from chronos2_modular.common import build_zone_configs
    from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
    from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle, save_nuclear_result_bundle
    from chronos2_hourly.nuclear_cwe_forecast import run_nuclear_cwe_forecast
    from chronos2_hourly.nuclear_cwe_kalman import build_nuclear_cwe_kalman_view
    from chronos2_hourly.nuclear_cwe_reporting import render_nuclear_cwe_reports
    workdir = workdir_for(settings, day, zone)
    config = _verify_snapshot(workdir, settings, day, zone)
    with run_progress(workdir, zone, pd.Timestamp(day), args.action) as progress:
        progress("prepare_inputs")
        spec = build_zone_configs(config, [zone], None, None)[0]
        target, source = _verify_reporting(workdir, zone=zone, day=day, timezone=spec.timezone)
        data = prepare_nuclear_zone_data(spec, config, workdir, workdir / "prepared")
        if args.action == "prepare":
            return {"status": "prepared", "zone": zone, "workdir": str(workdir), "production_modified": False}
        reused = (workdir / "report_only/frozen_result").is_dir()
        if args.action == "report" or reused:
            progress("load_frozen_result")
            result = load_nuclear_result_bundle(workdir=workdir)
        else:
            progress("forecast", history_anchor=config["nuclear_experiment"]["history_anchor_day"])
            print(f"[CWE/{zone}] nouveau replay Chronos + correcteur + Kalman; checkpoints isoles reutilisables.", flush=True)
            result = run_forecast_with_storage_retry(run_nuclear_cwe_forecast, config=config, data=data,
                zone=zone, delivery_day=day, workdir=workdir, device=args.device,
                threads=args.threads, workers=args.workers, kalman_builder=build_nuclear_cwe_kalman_view)
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
                    print(f"[CWE/{zone}] attribution indisponible : {error}", flush=True)
        progress("render_reports")
        reference = workdir / "reference"
        incumbent = SimpleNamespace(backtest=pd.read_parquet(reference / "incumbent_backtest.parquet"),
                                    forecast=pd.read_parquet(reference / "incumbent_forecast.parquet"))
        reports = render_nuclear_cwe_reports(result, incumbent=incumbent, data=replace(data, target=target),
            zone=zone, delivery_day=day, output_directory=workdir / "reports",
            source_audit=_read(workdir / "source_audit.json"), storm_archive=reference / "reporting",
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
        experiment_root = (ROOT / "runs/experiments/nuclear_cwe_v1").resolve()
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
    parser.add_argument("--config", type=Path, default=ROOT / "config/nuclear_cwe.yaml")
    parser.add_argument("--delivery-day")
    parser.add_argument("--zones", nargs="+", choices=["FR", "DE", "BE", "NL"])
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
    with exclusive_process_lock(settings["output_root"] / ".nuclear_cwe.lock"):
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
        if needs or args.action == "sync":
            from chronos2_hourly.nuclear_cwe_sources import materialize_cwe_sources
            start = min(starts) if starts else (pd.Timestamp(day) - pd.Timedelta(days=730)).date().isoformat()
            sources = materialize_cwe_sources(settings["source_root"], start, day, workers=min(args.workers, 2))
            if args.action == "sync":
                print(json.dumps(sources, indent=2, default=str))
                return 0
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
        print(f"[CWE] ECHEC : {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
