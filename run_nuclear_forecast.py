"""Opt-in nuclear runs and standard Storm exports; incumbent archives stay intact."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout
import yaml

from chronos2_hourly.nuclear_sources import (
    NUCLEAR_ALIAS, NUCLEAR_SERIES, audit_nuclear_store, build_materialize_command,
)
from chronos2_modular.common import build_zone_configs, resolve_path
from chronos2_modular.saturn import cache_path_for_series, resolve_pit_path, is_pit_spec
from chronos2_hourly.process_lock import exclusive_process_lock as exclusive_lock

ROOT = Path(__file__).resolve().parent
RESIDUAL_ALIASES = tuple(f"{zone}_residual_load_fcst" for zone in ("fr", "de", "be", "nl", "es"))
INPUT_PROTOCOL = "civil_pit_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from chronos2_hourly.nuclear_exports import _replace_with_retry
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + "-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, default=str, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def run_progress(workdir: Path, zone: str, day: pd.Timestamp, stage: str):
    """Track progress without replacing a model error or a publication receipt.

    This auxiliary status is not evidence that exports were published. The
    caller still must write run_result.json and emit its completion receipt;
    the launcher separately verifies the exported files and their checksums.
    """
    record = {"zone": zone, "delivery_day": str(day.date()), "stage": stage,
              "pid": os.getpid(), "started_at_utc": str(pd.Timestamp.now(tz="UTC"))}

    def persist():
        try:
            write_json(workdir / "run_status.json", record)
        except Exception as status_error:
            # A locked/full status destination must neither mask the original
            # exception nor turn already-published, verified exports into a
            # failed run. Never apply this policy to run_result.json itself.
            try:
                print(f"[Nuclear/{zone}] AVERTISSEMENT : statut {record.get('phase')} "
                      f"non enregistre ({type(status_error).__name__}: {status_error}). "
                      "run_status.json peut etre absent ou ancien ; "
                      "seul le recu de publication confirme la fin du run.",
                      file=sys.stderr, flush=True)
            except OSError:
                pass

    def update(phase: str, **details):
        record.update(phase=phase, status="running", updated_at_utc=str(pd.Timestamp.now(tz="UTC")), **details)
        persist()
    update("starting")
    try:
        yield update
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error),
                      updated_at_utc=str(pd.Timestamp.now(tz="UTC")))
        persist()
        raise
    else:
        record.update(status="complete", updated_at_utc=str(pd.Timestamp.now(tz="UTC")))
        persist()


def refresh_reporting_sources(*args):
    """Retry transport failures only; never conceal invalid observations."""
    from chronos2_hourly.nuclear_reporting_refresh import refresh_nuclear_reporting_sources
    for attempt in range(1, 4):
        try:
            return refresh_nuclear_reporting_sources(*args)
        except (ConnectionError, TimeoutError, RuntimeError,
                RequestsConnectionError, RequestsTimeout) as error:
            if attempt == 3:
                raise
            print(f"[Nuclear] collecte reporting interrompue ({type(error).__name__}); "
                  f"nouvel essai {attempt + 1}/3 dans {2 * attempt} s.", flush=True)
            time.sleep(2 * attempt)


def _retryable_result_publication(error: PermissionError, workdir: Path) -> dict | None:
    """Recognize only a failed atomic publication of this run's result cache."""
    if getattr(error, "winerror", None) not in {5, 32, 33}:
        return None
    from chronos2_hourly.nuclear_forecast import _save_result_cache, _digest_json
    trace = error.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code is _save_result_cache.__code__:
            directory = trace.tb_frame.f_locals.get("directory")
            staging = trace.tb_frame.f_locals.get("staging")
            if not isinstance(directory, Path) or not isinstance(staging, Path):
                return None
            target, source = directory.resolve(), staging.resolve()
            cache_root = workdir.resolve() / "cache" / "nuclear_result"
            if (target.parent.parent != cache_root or target.name not in {"raw_future", "residual"}
                    or re.fullmatch(r"[0-9a-f]{64}", target.parent.name) is None
                    or source.parent != target.parent
                    or not source.name.startswith("." + target.name + "-")
                    or os.path.lexists(target)):
                return None
            if (not error.filename or not error.filename2
                    or Path(error.filename).resolve() != source
                    or Path(error.filename2).resolve() != target):
                return None
            context = trace.tb_frame.f_locals
            frames, identity, files = (context.get(key) for key in ("frames", "identity", "files"))
            expected = {"raw_future"} if target.name == "raw_future" else {"statistics", "forecast", "daily_audit"}
            if (not isinstance(frames, dict) or set(frames) != expected
                    or not all(isinstance(frame, pd.DataFrame) for frame in frames.values())
                    or not isinstance(identity, dict) or not isinstance(files, dict)
                    or set(files) != {name + ".parquet" for name in expected}
                    or not all(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                               for digest in files.values())):
                return None
            identity = deepcopy(identity)
            if target.parent.name != _digest_json(identity):
                return None
            return {"target": target, "frames": dict(frames), "identity": identity, "files": dict(files)}
        trace = trace.tb_next
    return None


def _recover_result_publication(context: dict) -> None:
    """Republish the exact payload retained by the failed writer's traceback.

    The immutable numerical engine cleans its failed staging directory. Its
    completed frames and checksums remain in the exception frame, so recover
    those bytes through the resilient I/O layer without editing the engine or
    invalidating its daily cache identities. Any serialization difference fails
    closed; no model, source refresh or prediction change happens here.
    """
    from chronos2_hourly.atomic_directory import AtomicDirectoryStaging
    from chronos2_hourly.nuclear_forecast import _load_result_cache
    target = context["target"]
    with AtomicDirectoryStaging(target.parent, prefix="." + target.name + "-recovery-") as stage:
        for name, frame in context["frames"].items():
            filename = name + ".parquet"
            path = stage.path / filename
            frame.to_parquet(path, index=True)
            if sha256(path) != context["files"][filename]:
                raise ValueError(f"Cache recupere divergent du resultat deja calcule : {filename}.")
        (stage.path / "cache_manifest.json").write_text(json.dumps(
            {"schema_version": 1, "identity": context["identity"], "files": context["files"]},
            ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        stage.publish(target)
    _load_result_cache(target, context["identity"])


def _retryable_kalman_worker_startup(error: PermissionError) -> bool:
    """Recognize a denied Windows pipe before loky has submitted Kalman fits."""
    if getattr(error, "winerror", None) != 5:
        return False
    from multiprocessing.connection import Pipe
    from joblib import Parallel
    from joblib._parallel_backends import LokyBackend
    from chronos2_hourly.kalman_residual import _replay_rolling_kalman_overlay

    required = {Pipe.__code__, Parallel._initialize_backend.__code__,
                LokyBackend.configure.__code__, _replay_rolling_kalman_overlay.__code__}
    trace = error.__traceback__
    observed = set()
    while trace is not None:
        observed.add(trace.tb_frame.f_code)
        trace = trace.tb_next
    return required.issubset(observed)


def run_forecast_with_storage_retry(forecast_function, **kwargs):
    """Resume cached work after a Windows publication or Kalman IPC failure.

    Keep the numerical engine and its cache identity unchanged. That engine
    saves each completed day before publishing the assembled result directory.
    Recover the already-computed payload, then resume from that sealed cache.
    A denied loky startup falls back once to sequential Kalman with the same
    numerical recipe. Input/training errors and full-mode runs are not retried.
    """
    incremental = kwargs["config"].get("nuclear_experiment", {}).get("mode") == "incremental"
    publication_retries = 0
    while True:
        try:
            return forecast_function(**kwargs)
        except PermissionError as error:
            if (incremental and kwargs.get("workers", 1) > 1
                    and _retryable_kalman_worker_startup(error)):
                kwargs = dict(kwargs, workers=1)
                print(f"[Nuclear/{kwargs.get('zone', '?')}] Windows refuse le canal de communication "
                      "des workers Kalman. Reprise avec un seul worker depuis les caches ; "
                      "parametres du modele et nombre de threads inchanges.", flush=True)
                continue
            context = (_retryable_result_publication(error, Path(kwargs["workdir"]))
                       if incremental else None)
            if context is None or publication_retries == 2:
                # The experiment CLIs print only str(error). Keep the failing
                # operation visible even for Windows errors with no filename.
                import traceback
                traceback.print_exception(error)
                raise
            print(f"[Nuclear/{kwargs.get('zone', '?')}] sauvegarde Windows temporairement bloquee : {context['target']}. "
                  "Recuperation des fichiers deja calcules avec verification de leurs SHA.", flush=True)
            _recover_result_publication(context)
            publication_retries += 1
            print(f"[Nuclear/{kwargs.get('zone', '?')}] Reprise depuis les caches quotidiens verifies "
                  f"et le resultat recupere, essai {publication_retries + 1}/3.", flush=True)


def load_settings(path: Path) -> dict:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if settings.get("schema_version") != 1:
        raise ValueError("schema_version nucleaire doit valoir 1.")
    project = resolve_path(settings.get("project_root", ".."), path.parent)
    settings["project_root"] = project
    settings.setdefault("residual_bank", "data/pit/nuclear_forecast/residual_load_market_features.parquet")
    settings.setdefault("residual_bank_seed", "data/pit/kalman_hybrid/residual_load_market_features.parquet")
    for key in ("output_root", "nuclear_store", "lora_activation_config", "kalman_config",
                "residual_bank", "residual_bank_seed"):
        settings[key] = resolve_path(settings[key], project)
    # Keep the opt-in strictly outside production even after path resolution.
    for key, parent in (("output_root", project / "runs/experiments"),
                        ("nuclear_store", project / "data/pit/nuclear_forecast"),
                        ("residual_bank", project / "data/pit/nuclear_forecast")):
        if parent.resolve() not in settings[key].parents:
            raise ValueError(f"{key} doit rester sous {parent}.")
    if settings["residual_bank"].name != "residual_load_market_features.parquet":
        raise ValueError("Le residual_bank doit se nommer residual_load_market_features.parquet.")
    if settings["residual_bank"] in (settings["residual_bank_seed"], settings["nuclear_store"]):
        raise ValueError("Le cache residuel isole doit etre distinct du seed et de la source nucleaire.")
    if settings.get("incomplete_dst_policy") not in {"raise", "duplicate"}:
        raise ValueError("Politique DST attendue : raise ou duplicate.")
    if not 1 <= int(settings.get("sync_chunk_days", 31)) <= 31:
        raise ValueError("sync_chunk_days doit etre entre 1 et 31.")
    settings.setdefault("computation_mode", "incremental")
    if settings["computation_mode"] not in {"incremental", "full"}:
        raise ValueError("computation_mode doit valoir incremental ou full.")
    return settings


def delivery_date(value: str | None) -> pd.Timestamp:
    now = pd.Timestamp.now(tz="Europe/Paris")
    day = pd.Timestamp(value) if value else now.tz_localize(None).normalize() + pd.Timedelta(days=1)
    if day.tzinfo is not None or pd.isna(day) or day != day.normalize():
        raise ValueError("delivery-day doit etre une date civile YYYY-MM-DD.")
    cutoff = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris")
    if cutoff > now:
        raise ValueError("Le cutoff D-1 08:00 Europe/Paris n'est pas encore atteint.")
    return day


def source_bounds(day: pd.Timestamp) -> tuple[str, str]:
    return (day - pd.Timedelta(days=730)).date().isoformat(), day.date().isoformat()


def required_source_bounds(settings: dict, day: pd.Timestamp) -> tuple[str, str]:
    start, end = source_bounds(day)
    if settings.get("computation_mode", "incremental") == "incremental":
        from chronos2_hourly.nuclear_incremental import retained_source_start
        retained = retained_source_start(settings["output_root"] / "_daily_cache", day.date())
        if retained < (day - pd.Timedelta(days=730)).date():
            start = min(start, retained.isoformat())
    return start, end


def compact_audit(audit: dict) -> dict:
    summary = {key: value for key, value in audit.items()
               if key not in {"missing_hours", "missing_days", "coverage_by_day"}}
    summary["missing_days_sample"] = audit.get("missing_days", [])[:5]
    return summary


def sync_source(settings: dict, day: pd.Timestamp, workers: int) -> dict:
    """Resume only verified isolated source chunks, never the old timezone cache."""
    path = settings["nuclear_store"]
    start, end = required_source_bounds(settings, day)
    with exclusive_lock(path.with_suffix(".sync.lock")):
        if path.exists():
            metadata = json.loads(path.with_name(path.name + ".audit.json").read_text(encoding="utf-8"))
            prior = audit_nuclear_store(path, metadata["start_day"], metadata["end_day"])
            if not prior["complete"]:
                raise ValueError("Cache nucleaire existant non valide; choisir un nouveau nuclear_store. "
                                 + " | ".join(prior["blockers"][:3]))
        audit = audit_nuclear_store(path, start, end)
        missing = set(audit["missing_days"])
        days = pd.date_range(start, end, freq="D")
        pending = [item for item in days if item.date().isoformat() in missing]
        # A missing initial artifact can report its blocker before listing hours.
        if not path.exists():
            pending = list(days)
        while pending:
            chunk = [pending.pop(0)]
            while (pending and len(chunk) < int(settings.get("sync_chunk_days", 31))
                   and pending[0] == chunk[-1] + pd.Timedelta(days=1)):
                chunk.append(pending.pop(0))
            command = build_materialize_command(
                chunk[0], chunk[-1], path, sys.executable, min(workers, 32),
                settings["incomplete_dst_policy"],
            )
            if path.exists():
                command.append("--merge-existing")
            print(f"[Nuclear] collecte {chunk[0].date()} -> {chunk[-1].date()}", flush=True)
            subprocess.run(command, check=True, cwd=ROOT)
        audit = audit_nuclear_store(path, start, end)
        if not audit["complete"]:
            raise ValueError("Collecte incomplete : " + " | ".join(audit["blockers"][:5]))
        return audit


def zone_inputs(settings: dict, zone: str) -> tuple[dict, Path, dict[str, Path]]:
    project = settings["project_root"]
    source_path = resolve_path(settings["zone_configs"][zone], project)
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    zone_spec = build_zone_configs(config, [zone], None, None)[0]
    source_project = resolve_path(config["data"].get("project_root", "."), source_path.parent)
    cache = resolve_path(config["data"]["cache_dir"], source_project)
    target = (resolve_path(zone_spec.target.file, source_path.parent) if zone_spec.target.file
              else cache_path_for_series(cache, zone, zone_spec.target))
    inputs = {"target": target}
    for alias, spec in zone_spec.covariates.items():
        if not spec.enabled:
            continue
        if not is_pit_spec(spec, config) or spec.file:
            raise ValueError(f"{zone}/{alias}: ce challenger exige une covariable PIT explicite.")
        inputs[alias] = (settings["residual_bank"] if alias in RESIDUAL_ALIASES
                         else resolve_pit_path(spec, config, source_path.parent))
    if NUCLEAR_ALIAS in inputs:
        raise ValueError("La configuration de base contient deja l'alias nucleaire challenger.")
    inputs[NUCLEAR_ALIAS] = settings["nuclear_store"]
    return config, source_path, inputs


def check_lora_inactive(settings: dict, zones: list[str]) -> None:
    activation = yaml.safe_load(settings["lora_activation_config"].read_text(encoding="utf-8"))
    for zone in zones:
        enabled = activation.get("zones", {}).get(zone, {}).get("enabled_modes", [])
        if set(enabled or []) & {"autonomous", "kalman"}:
            raise ValueError(f"{zone}: LoRA active. Le challenger nucleaire est Chronos-2 sans LoRA; "
                             "une nouvelle calibration LoRA avec ce schema serait necessaire.")


def validate_target_cache(path: Path) -> None:
    """The legacy generic file reader must never reinterpret naive UTC labels."""
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if frame.empty or not {"timestamp", "value"}.issubset(frame):
        raise ValueError(f"Cache cible attendu avec timestamp/value : {path}")
    if any(pd.isna(value) or pd.Timestamp(value).tzinfo is None for value in frame.timestamp):
        raise ValueError(f"Cache cible exige des timestamps timezone-aware explicites : {path}")
    index = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True, errors="raise"))
    if index.has_duplicates or index.hasnans or not index.is_monotonic_increasing:
        raise ValueError(f"Cache cible : timeline physique unique et croissante requise : {path}")
    if not np.isfinite(pd.to_numeric(frame.value, errors="coerce").to_numpy(dtype=float)).all():
        raise ValueError(f"Cache cible : prix finis requis : {path}")


def snapshot_config(settings: dict, zone: str, day: pd.Timestamp, workdir: Path,
                    *, target_override: Path | None = None, computation_mode: str | None = None) -> dict:
    """Pin input bytes once; retries use the same sealed research inputs."""
    config, config_path, inputs = zone_inputs(settings, zone)
    if target_override is not None:
        inputs["target"] = target_override
    kalman_settings = yaml.safe_load(settings["kalman_config"].read_text(encoding="utf-8"))
    filter_parameters = kalman_settings.get("filter_parameters", {})
    if int(kalman_settings.get("training_lookback_days", 365)) != 365:
        raise ValueError("Le challenger exige une calibration Kalman de 365 jours.")
    mode = computation_mode or settings.get("computation_mode", "incremental")
    if mode not in {"incremental", "full"}:
        raise ValueError("computation_mode doit valoir incremental ou full.")
    identity = {"base_config_sha256": sha256(config_path), "delivery_day": str(day.date()),
                "kalman_filter_parameters": filter_parameters,
                "zone": zone, "nuclear_alias": NUCLEAR_ALIAS, "schema_version": 1,
                "input_protocol": INPUT_PROTOCOL, "computation_mode": mode}
    manifest_path = workdir / "input_snapshot.json"
    resolved_path = workdir / "resolved_config.yaml"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if computation_mode is None and "computation_mode" in manifest["identity"]:
            identity["computation_mode"] = manifest["identity"]["computation_mode"]
        # An old sealed delivery retains its original full-replay semantics.
        # An explicit request to reinterpret it as incremental is refused.
        if "computation_mode" not in manifest["identity"] and computation_mode in (None, "full"):
            identity.pop("computation_mode")
        if manifest["identity"] != identity:
            raise ValueError("La configuration a change : utiliser un nouveau output_root experimental.")
        for item in manifest["files"]:
            pinned = Path(item["snapshot"])
            if not pinned.is_file() or sha256(pinned) != item["sha256"]:
                raise ValueError(f"Snapshot modifie : {pinned}")
        if sha256(resolved_path) != manifest["resolved_config_sha256"]:
            raise ValueError("Configuration experimentale figee modifiee.")
        return yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    snapshot = workdir / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    pinned_inputs, files = {}, []
    for alias, source in inputs.items():
        destination = snapshot / (alias + "".join(source.suffixes))
        before = sha256(source)
        shutil.copy2(source, destination)
        if sha256(destination) != before or sha256(source) != before:
            raise ValueError(f"Source modifiee pendant la copie : {source}. Relancer apres la synchronisation.")
        pinned_inputs[alias] = destination
        files.append({"source": str(source), "snapshot": str(destination), "sha256": before})
    source_audit = settings["nuclear_store"].with_name(settings["nuclear_store"].name + ".audit.json")
    audit_destination = pinned_inputs[NUCLEAR_ALIAS].with_name(pinned_inputs[NUCLEAR_ALIAS].name + ".audit.json")
    shutil.copy2(source_audit, audit_destination)
    copied_audit = audit_nuclear_store(pinned_inputs[NUCLEAR_ALIAS], *required_source_bounds(settings, day))
    if not copied_audit["complete"]:
        raise ValueError("Audit du snapshot nucleaire refuse : " + " | ".join(copied_audit["blockers"][:3]))
    files.append({"source": str(source_audit), "snapshot": str(audit_destination), "sha256": sha256(audit_destination)})
    residual_audit_destination = None
    if all(inputs.get(alias) == settings["residual_bank"] for alias in RESIDUAL_ALIASES):
        from chronos2_hourly.nuclear_residual_inputs import audit_residual_bank
        residual_source_audit = settings["residual_bank"].with_name(settings["residual_bank"].name + ".audit.json")
        residual_audit_destination = pinned_inputs[RESIDUAL_ALIASES[0]].with_name(
            pinned_inputs[RESIDUAL_ALIASES[0]].name + ".audit.json")
        shutil.copy2(residual_source_audit, residual_audit_destination)
        residual_audit = audit_residual_bank(pinned_inputs[RESIDUAL_ALIASES[0]], *required_source_bounds(settings, day))
        if not residual_audit["complete"]:
            raise ValueError("Snapshot residuel refuse : " + " | ".join(residual_audit["blockers"]))
        files.append({"source": str(residual_source_audit), "snapshot": str(residual_audit_destination),
                      "sha256": sha256(residual_audit_destination)})
    validate_target_cache(pinned_inputs["target"])
    config = deepcopy(config)
    config["zones"] = {zone: config["zones"][zone]}
    raw_zone = config["zones"][zone]
    raw_zone["target"].update(file=str(pinned_inputs["target"]), source="file",
                              timestamp_col="timestamp", value_col="value", naive_timezone="UTC")
    covariates = {alias: item for alias, item in raw_zone["covariates"].items()
                  if item.get("enabled", True)}
    covariates[NUCLEAR_ALIAS] = {
        "enabled": True, "source": "pit_parquet", "series": NUCLEAR_SERIES,
        "fill_method": "none", "fill_limit": 0, "minimum_coverage": 0.01,
        "description": "Production nucleaire FR prevue en GW, as-of D-1 08:00 Paris",
        "future": {"known_future": True, "strategies": ["oracle"]},
    }
    for alias, item in covariates.items():
        item.update(source="pit_parquet", pit_file=str(pinned_inputs[alias]))
    raw_zone["covariates"] = covariates
    cutoff = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris")
    config["data"].update(project_root=str(settings["project_root"]), source="cache",
                          cache_dir=str(snapshot / "cache"), pit_vintage_dir=str(snapshot),
                          pit_files={key: str(value) for key, value in pinned_inputs.items() if key != "target"},
                          runtime_as_of=cutoff.isoformat(), target_end_policy="current_day_end")
    config["backtest"]["windows"] = 730
    features = config.setdefault("hourly", {}).setdefault("feature_engineering", {})
    if features.get("covariate_columns"):
        features["covariate_columns"] = list(dict.fromkeys([
            *features["covariate_columns"], f"known_{NUCLEAR_ALIAS}_oracle",
        ]))
    config.setdefault("output", {})["directory"] = str(workdir)
    config["model"]["local_files_only"] = True
    daily_cache = (settings["output_root"] / "_daily_cache" / zone.lower() / INPUT_PROTOCOL).resolve()
    if not daily_cache.is_relative_to(settings["output_root"].resolve()):
        raise ValueError("Le cache quotidien sort de output_root.")
    config["nuclear_experiment"] = {"filter_parameters": filter_parameters, "mode": mode,
                                     "incremental_cache_dir": str(daily_cache)}
    config["nuclear_experiment"]["input_protocol"] = INPUT_PROTOCOL
    if residual_audit_destination:
        config["nuclear_experiment"]["residual_bank_audit"] = str(residual_audit_destination)
    config["model"]["revision"] = resolve_local_model_revision(config)
    from chronos2_hourly.nuclear_incremental import prepare_incremental_settings
    prepare_incremental_settings(config, day.date())
    resolved_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(manifest_path, {"identity": identity, "files": files,
                              "resolved_config_sha256": sha256(resolved_path)})
    return config


def resolve_local_model_revision(config: dict) -> str:
    """Pin the already downloaded HF commit; never request a model download."""
    from huggingface_hub import hf_hub_download
    local = Path(hf_hub_download(repo_id=config["model"]["model_id"], filename="config.json",
                                revision=config["model"].get("revision", "main"), local_files_only=True))
    if local.parent.parent.name != "snapshots":
        raise ValueError("Impossible d'identifier le commit du modele Chronos local.")
    return local.parent.name


def experiment_workdir(settings: dict, day: pd.Timestamp, zone: str) -> Path:
    # Do not rewrite the v1 input seal after a failed run. The corrected
    # protocol has its own snapshot/cache namespace; old artifacts stay intact.
    result = (settings["output_root"] / str(day.date()) / zone.lower() / INPUT_PROTOCOL).resolve()
    if settings["output_root"] not in result.parents:
        raise ValueError("Le dossier de run experimental sort de output_root.")
    return result


def ensure_residual_inputs(settings: dict, day: pd.Timestamp, workers: int) -> dict:
    from chronos2_hourly.nuclear_residual_inputs import ensure_residual_bank
    start, end = required_source_bounds(settings, day)
    return ensure_residual_bank(path=settings["residual_bank"], seed_path=settings["residual_bank_seed"],
                                start_day=start, end_day=end, workers=workers, allow_sync=True)


def local_storm_archive(settings: dict, zone: str, day: pd.Timestamp) -> Path | None:
    """Resolve the normal run's output root from its live config, never assume FR."""
    project = settings["project_root"]
    registry_path = project / "chronos2_hourly_live_zones.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    entry = registry["zones"][zone]
    live_path = resolve_path(entry["live_config"], registry_path.parent)
    live = yaml.safe_load(live_path.read_text(encoding="utf-8"))["live"]
    root = resolve_path(live["output_root"], live_path.parent)
    return root / f"{zone.lower()}_day_ahead_{day.date()}"


def refreshed_target_snapshot(settings: dict, zone: str, day: pd.Timestamp,
                              workdir: Path, *, client=None) -> Path:
    """Refresh training from its own canonical source, independently of report labels."""
    from chronos2_modular.saturn import create_saturn_client, fetch_saturn_series_from_client

    if (workdir / "input_snapshot.json").exists():
        raise ValueError("Un snapshot deja fige ne peut pas actualiser ses prix d'entrainement.")
    config, _, inputs = zone_inputs(settings, zone)
    validate_target_cache(inputs["target"])
    old = pd.read_parquet(inputs["target"]) if inputs["target"].suffix == ".parquet" else pd.read_csv(inputs["target"])
    prices = pd.Series(old.value.to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(old.timestamp, utc=True)))
    spec = build_zone_configs(config, [zone], None, None)[0]
    timezone = spec.timezone
    end = day.tz_localize(timezone).tz_convert("UTC")
    start = (day - pd.Timedelta(days=365)).tz_localize(timezone).tz_convert("UTC")
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    data_config = config.get("data", {})
    if client is None:
        client = create_saturn_client(data_config.get("saturn_url"),
                                      os.getenv("SATURN_AUTHOR") or data_config.get("saturn_author"))
    extracted = pd.Timestamp.now(tz="UTC")
    # The reporting source can differ from the model's target. Never pass EPEX
    # report observations into this function or substitute them for a missing target.
    observed = fetch_saturn_series_from_client(
        client, spec.target.series, start - pd.Timedelta(hours=2), end - pd.Timedelta(hours=1),
        timezone, naive_timezone="UTC", nocache=True, live=True)
    index = pd.DatetimeIndex(observed.index)
    if (index.tz is None or index.hasnans or index.has_duplicates
            or not index.is_monotonic_increasing):
        raise ValueError(f"{zone}: timeline des prix d'entrainement invalide.")
    index = index.tz_convert("UTC")
    if not index.equals(index.floor("h")):
        raise ValueError(f"{zone}: prix d'entrainement hors grille horaire.")
    latest = pd.Series(pd.to_numeric(observed, errors="raise").to_numpy(float),
                       index=index).reindex(expected)
    if not np.isfinite(latest.to_numpy(float)).all():
        raise ValueError(f"{zone}: historique canonique d'entrainement incomplet; aucun prix de reporting substitue.")
    prices = latest.combine_first(prices).sort_index().loc[lambda values: values.index < end]
    target = workdir / "report_only" / "target_for_new_snapshot.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": prices.index, "value": prices.to_numpy(float)}).to_parquet(target, index=False)
    write_json(target.with_suffix(".audit.json"), {
        "kind": "canonical_training_target_refresh", "series": spec.target.series,
        "zone": zone, "timezone": timezone, "delivery_day": str(day.date()),
        "extracted_at_utc": str(extracted), "used_for_prediction": True,
        "report_observations_used": False, "delivery_day_observations_used": False,
        "refreshed_history_hours": len(expected), "artifact_sha256": sha256(target),
    })
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/nuclear_forecast.yaml")
    parser.add_argument("--stage", type=str.lower, choices=("audit", "sync", "prepare", "run", "report"), default="run")
    parser.add_argument("--zones", nargs="+", choices=("FR", "DE", "BE", "NL", "ES"), default=["FR"])
    parser.add_argument("--delivery-day")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--computation-mode", choices=("incremental", "full"), default=None,
                        help="Calcul quotidien incremental ou replay complet de reference (nouveau snapshot).")
    parser.add_argument("--skip-observed-sync", action="store_true",
                        help="Explicit offline report mode: use local observed and audited Storm caches.")
    parser.add_argument("--skip-source-sync", action="store_true",
                        help="Sources communes deja synchronisees par le batch; leurs audits restent obligatoires.")
    parser.add_argument("--report-variants", choices=("both", "kalman", "autonomous"), default="both")
    parser.add_argument("--skip-attribution", action="store_true",
                        help="Ne pas calculer les explications contrefactuelles du rapport.")
    args = parser.parse_args(argv)
    report_variants = ("autonomous", "kalman") if args.report_variants == "both" else (args.report_variants,)
    if not 1 <= args.threads <= 128 or not 1 <= args.workers <= 128:
        parser.error("threads/workers doivent etre entre 1 et 128")
    settings, day = load_settings(args.config.resolve()), delivery_date(args.delivery_day)
    if args.computation_mode is not None:
        settings["computation_mode"] = args.computation_mode
    if args.stage != "report":
        check_lora_inactive(settings, args.zones)
    if args.stage == "sync":
        nuclear = sync_source(settings, day, args.workers)
        residual = ensure_residual_inputs(settings, day, args.workers)
        print(json.dumps({"nuclear": compact_audit(nuclear), "residual_load": compact_audit(residual)},
                         indent=2, ensure_ascii=False))
        return 0
    if args.stage == "run" and not args.skip_source_sync:
        # Missing suffix only; sync_source validates and preserves the prefix.
        print("[Nuclear] verification / synchronisation des sources nucleaires...", flush=True)
        sync_source(settings, day, args.workers)
    audit = ({"blockers": []} if args.stage == "report" else
             audit_nuclear_store(settings["nuclear_store"], *required_source_bounds(settings, day)))
    blockers = list(audit["blockers"])
    from chronos2_hourly.nuclear_residual_inputs import audit_residual_bank
    if args.stage in ("run", "prepare") and not blockers and not args.skip_source_sync:
        # The existing validated prefix is copied, never modified. Only the
        # missing suffix is queried as-of, before snapshotting or loading Chronos.
        ensure_residual_inputs(settings, day, args.workers)
    residual_audit = ({"blockers": []} if args.stage == "report" else
                      audit_residual_bank(settings["residual_bank"], *required_source_bounds(settings, day)))
    blockers.extend(residual_audit["blockers"])
    for zone in ([] if args.stage == "report" else args.zones):
        _, _, inputs = zone_inputs(settings, zone)
        blockers.extend(f"{zone}: input absent : {path}" for path in inputs.values() if not path.is_file())
    if args.stage == "audit":
        print(json.dumps({"delivery_day": str(day.date()), "nuclear_source": compact_audit(audit),
                          "residual_load_source": compact_audit(residual_audit),
                          "blockers": blockers, "ready": not blockers,
                          "production_modified": False}, indent=2, ensure_ascii=False))
        return 0 if not blockers else 2
    if blockers:
        raise ValueError("Preflight nucleaire refuse : " + " | ".join(blockers[:5])
                         + ". Lancer -WithNuclear -NuclearStage Sync puis verifier les caches du run Both habituel.")
    # Heavy model imports happen only after the read-only preflight.
    from chronos2_modular.data import read_series_file
    from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
    for zone in dict.fromkeys(args.zones):
        workdir = experiment_workdir(settings, day, zone)
        with exclusive_lock(workdir / "run.lock"), run_progress(workdir, zone, day, args.stage) as progress:
            if args.stage == "report" and not (workdir / "input_snapshot.json").is_file():
                raise ValueError(f"{zone}: aucun run fige pour Report; lancer NuclearStage Run d'abord.")
            observed_source_audit = None
            fresh_observed = None
            storm_archive = None
            target_override = None
            if args.stage in ("run", "report") and not args.skip_observed_sync:
                progress("reporting_sources")
                base, _, _ = zone_inputs(settings, zone)
                base_spec = build_zone_configs(base, [zone], None, None)[0]
                print(f"[Nuclear/{zone}] actualisation des observations et de Storm pour les rapports...", flush=True)
                fresh_observed, storm_archive, observed_source_audit = refresh_reporting_sources(
                    base, zone, base_spec.timezone, str(day.date()), workdir / "report_only" / "sources")
                if not (workdir / "input_snapshot.json").exists():
                    target_override = refreshed_target_snapshot(settings, zone, day, workdir)
            progress("prepare_inputs")
            config = snapshot_config(settings, zone, day, workdir, target_override=target_override,
                                     computation_mode=args.computation_mode)
            spec = build_zone_configs(config, [zone], None, None)[0]
            data = prepare_nuclear_zone_data(spec, config, workdir, workdir / "prepared")
            if args.stage == "prepare":
                print(json.dumps({"zone": zone, "status": "prepared", "delivery_day": str(day.date()),
                    "history_days": 730, "covariates": len(spec.covariates), "workdir": str(workdir),
                    "forecast_started": False, "production_modified": False}), flush=True)
                continue
            from chronos2_hourly.nuclear_forecast import run_nuclear_forecast
            from chronos2_hourly.nuclear_reporting import render_nuclear_reports
            from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle, save_nuclear_result_bundle
            reuse_result = args.stage == "report" or (workdir / "report_only/frozen_result").exists()
            if reuse_result:
                progress("load_frozen_result")
                print(f"[Nuclear/{zone}] lecture du resultat fige, aucun entrainement ni replay.", flush=True)
                result = load_nuclear_result_bundle(workdir=workdir)
            else:
                progress("forecast", computation_mode=config.get("nuclear_experiment", {}).get("mode", "full"))
                mode = config.get("nuclear_experiment", {}).get("mode", "full")
                message = ("calcul quotidien incremental; reutilisation des journees valides"
                           if mode == "incremental" else "replay complet de reference sur 730 jours")
                print(f"[Nuclear/{zone}] livraison {day.date()} : {message}.", flush=True)
                result = run_forecast_with_storage_retry(run_nuclear_forecast,
                    config=config, data=data, zone=zone, delivery_day=str(day.date()),
                    workdir=workdir, device=args.device, threads=args.threads, workers=args.workers)
                save_nuclear_result_bundle(result, workdir=workdir)
            progress("forecast_saved", model_result_saved=True, forecast_result_reused=reuse_result)
            if reuse_result:
                # Frozen audits describe the original computation, not this launch.
                print(f"[Nuclear/{zone}] Prevision integralement reutilisee : "
                      "0 nouveau calcul Chronos, correcteur ou Kalman.", flush=True)
            else:
                cache_audit = result.kalman_view.replay.audit.get("rolling_refit_cache", {})
                print(f"[Nuclear/{zone}] Caches : Chronos futur={result.audit.get('raw_future_cache_hit')}; "
                      f"correcteur={result.audit.get('residual_result_cache_hit')}; "
                      f"Kalman historique hits={cache_audit.get('history_hits', 0)}, "
                      f"futur hits={cache_audit.get('future_hits', 0)}.", flush=True)
                if result.audit.get("computation_mode") == "incremental":
                    chronos_days = result.audit.get("daily_chronos_cache", {})
                    residual_days = result.audit.get("daily_residual_cache", {})
                    print(f"[Nuclear/{zone}] Reutilisation quotidienne : "
                          f"Chronos={chronos_days.get('history_hits', 0) + chronos_days.get('future_hits', 0)} jour(s), "
                          f"correcteur={residual_days.get('hits', 0)} jour(s); "
                          f"nouveaux calculs Chronos={chronos_days.get('history_misses', 0) + chronos_days.get('future_misses', 0)}, "
                          f"correcteur={residual_days.get('misses', 0)}, "
                          f"Kalman={cache_audit.get('history_fitted_days', 0) + cache_audit.get('future_fitted_days', 0)}.",
                          flush=True)
            # Explain only the issued day. This has an independent report cache
            # and never invalidates or repeats the 730-day nuclear replay.
            from chronos2_hourly.nuclear_attribution import prepare_nuclear_attribution
            attribution_directory = None
            attribution_status = {"status": "unavailable"}
            try:
                if args.skip_attribution:
                    attribution_status = {"status": "skipped", "reason": "explicit_skip_attribution"}
                    print(f"[Nuclear/{zone}] explications detaillees desactivees pour ce rapport.", flush=True)
                elif args.stage == "report":
                    progress("load_attribution")
                    prior = json.loads((workdir / "run_result.json").read_text(encoding="utf-8"))
                    existing = prior.get("variable_attribution", {})
                    if existing.get("status") == "complete":
                        attribution_directory = Path(existing["directory"])
                    else:
                        raise ValueError("Attribution non disponible; Report ne lance aucun calcul modele.")
                else:
                    progress("attribution")
                    print(f"[Nuclear/{zone}] explication variables + prix passés du jour (cache dédié)...", flush=True)
                    attribution_directory = prepare_nuclear_attribution(
                        result=result, data=data, config=config, workdir=workdir,
                        device=args.device, threads=args.threads,
                    )
                if not args.skip_attribution:
                    attribution_status = {"status": "complete", "directory": str(attribution_directory)}
            except Exception as error:
                # Failure to explain is not a failure of the already-frozen
                # model. The renderer must show unavailable, never fake weights.
                attribution_status = {"status": "failed", "error": str(error)}
                print(f"[Nuclear/{zone}] attribution indisponible : {error}", flush=True)
            # Labels may be refreshed for display, never fed into an already-issued prediction.
            _, _, current_inputs = zone_inputs(settings, zone)
            if fresh_observed is None:
                observed_digest = sha256(current_inputs["target"])
                validate_target_cache(current_inputs["target"])
                observed = read_series_file(current_inputs["target"], spec.target, spec.timezone)
                if sha256(current_inputs["target"]) != observed_digest:
                    raise ValueError("Le cache des observations a change pendant la lecture; relancer la publication.")
            else:
                observed, observed_digest = fresh_observed, observed_source_audit["observed"]["artifact_sha256"]
            observed = observed.loc[observed.index < (day + pd.Timedelta(days=1)).tz_localize(spec.timezone)]
            if args.skip_observed_sync:
                storm_archive = local_storm_archive(settings, zone, day)
                print(f"[Nuclear/{zone}] mode local explicite : observations / Storm non actualises par API.", flush=True)
            pinned_audit = audit_nuclear_store(Path(config["data"]["pit_files"][NUCLEAR_ALIAS]), *source_bounds(day))
            residual_audit_path = config["nuclear_experiment"].get("residual_bank_audit")
            if residual_audit_path:
                pinned_audit["residual_load_source_audit"] = json.loads(Path(residual_audit_path).read_text(encoding="utf-8"))
            progress("render_reports")
            reports = render_nuclear_reports(result, data=replace(data, target=observed), zone=zone,
                delivery_day=str(day.date()), output_directory=workdir / "reports", source_audit=pinned_audit,
                attribution_directory=attribution_directory,
                storm_archive=storm_archive, observed_source_audit=observed_source_audit,
                operational_layout=True, report_variants=report_variants)
            from chronos2_hourly.nuclear_exports import publish_nuclear_exports
            progress("publish_exports")
            exports = publish_nuclear_exports(result, reports, project_root=settings["project_root"],
                zone=zone, delivery_day=str(day.date()), timezone=spec.timezone, report_variants=report_variants)
            write_json(workdir / "run_result.json", {"reports": reports, "audit": result.audit,
                        "execution": {"forecast_result_reused": reuse_result},
                        "observed_labels_sha256": observed_digest, "production_modified": False,
                        "reporting_sources": observed_source_audit or {"status": "explicit_local_cache"},
                        "variable_attribution": attribution_status, "exports": exports})
            print(json.dumps({"zone": zone, "status": "complete", "reports": reports,
                              "exports": exports}, default=str, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[Nuclear] ECHEC : {error}", file=sys.stderr)
        raise SystemExit(2)
