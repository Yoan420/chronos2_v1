"""Isolated, sealed congestion-to-P50 replay; production is read-only."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shutil
import uuid

import joblib
import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock
from nyx_coherent_p50 import runner as source_runner
from nyx_physical_p50 import runner as physical_runner
from nyx_rmse.runner import runtime_identity, _verify_files
from nyx_scarcity.runner import digest, _json, _parquet
from nyx_scarcity.variants_runner import _audit_parquet_frame

LOGGER = logging.getLogger(__name__)
NAMESPACE = Path("runs/experiments/nyx_congestion_v1")
INPUTS = {"config.json", "panel.parquet", "network_features.parquet", "network_audit.json",
          "constraints.parquet", "zonal_labels.parquet", "constraint_audit.json"}
ACTIVATION_FILES = {"constraint_predictions.parquet", "activation_folds.parquet", "signals.parquet", "activation_audit.json",
                    "regional_predictions.parquet", "regional_folds.parquet", "regional_audit.json"}
RESULTS = {"predictions.parquet", "folds.parquet", "governance.parquet", "model_audit.json"}
MODELS = ("congestion_control_direct", "congestion_control_governed", "congestion_direct", "nyx_congestion")


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]


def safe_path(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    namespace = root/NAMESPACE
    raw = Path(value)
    path = (raw if raw.is_absolute() else root/raw).absolute()
    if path != path.resolve() or namespace != namespace.resolve() or not path.is_relative_to(namespace):
        raise ValueError(f"Sorties exclusivement dans {namespace}; chemins de production et alias interdits.")
    return path


def validate_config(config: dict) -> None:
    keys = {"schema_version", "source_suite", "output_root", "options", "diagnostic_only",
            "production_modified", "activation_performed"}
    if (not isinstance(config, dict) or set(config) != keys
            or type(config["schema_version"]) is not int or config["schema_version"] != 1):
        raise ValueError("Configuration Congestion schema 1 complete requise; champs inconnus refuses.")
    if any(not isinstance(config[k], str) or not config[k].strip() for k in ("source_suite", "output_root")):
        raise ValueError("Source et dossier de sortie explicites requis.")
    if Path(config['output_root']).as_posix() != NAMESPACE.as_posix():
        raise ValueError('Le namespace de ce laboratoire est fixe.')
    if (config["diagnostic_only"] is not True or config["production_modified"] is not False
            or config["activation_performed"] is not False):
        raise ValueError("Laboratoire exclusivement diagnostique, sans activation ni modification de production.")
    options = config["options"]
    if (not isinstance(options, dict) or set(options) != {"threads"}
            or type(options["threads"]) is not int or options["threads"] not in (1, 2)):
        raise ValueError("options accepte uniquement threads=1 ou 2; aucune optimisation sur la journee cible.")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def training_code(root: Path) -> dict:
    names = [p.relative_to(root).as_posix() for p in sorted((root/"nyx_congestion").glob("*.py")) if p.name != "report.py"]
    names += ["run_nyx_congestion.py", "NyxCongestion.ps1"]
    return {**physical_runner.training_code(root), **{n: digest(root/n) for n in names}}


def collect(config: dict, *, root: Path, start_day: str, end_day: str):
    from .data import collect_labels
    validate_config(config)
    before = source_runner.protected_state(root)
    result = collect_labels(start_day, end_day, output_root=safe_path(root, root/NAMESPACE/"raw/labels"),
        workers=2, progress=lambda m: LOGGER.info("[Congestion] %s", m))
    path = safe_path(root, root/NAMESPACE/"collections"/(_stamp()+".json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    _json(path, result)
    if before != source_runner.protected_state(root):
        raise ValueError("Production changed during collection.")
    return {"status": result["status"], "completed_days": len(result["days"]), "failed_days": len(result["failures"]),
            "audit_path": str(path), "production_modified": False}


def prepare(config: dict, *, root: Path) -> Path:
    from .data import build_constraint_panel
    validate_config(config)
    output = safe_path(root, config["output_root"])
    with exclusive_process_lock(safe_path(root, output/"prepare.lock")):
        before = source_runner.protected_state(root)
        source, _, source_manifest = physical_runner.read_snapshot(root/config["source_suite"], root=root)
        physical_runner.verify_result(source, source_manifest)
        identity = {n: digest(source/n) for n in {*physical_runner.INPUTS, *physical_runner.RESULTS,
                                                 "manifest.json", "results_manifest.json"}}
        panel = pd.read_parquet(source/"panel.parquet")
        network = pd.read_parquet(source/"network_features.parquet")
        network_audit = json.loads((source/"network_audit.json").read_text(encoding="utf8"))
        constraints, zonal, audit = build_constraint_panel(panel, network, network_audit,
            label_root=output/"raw/labels", progress=lambda m: LOGGER.info("[Congestion] %s", m))
        if constraints.empty or not constraints.label_eligible.any():
            raise ValueError("Aucun label de congestion qualifie. Lancer Collect puis Prepare.")
        directory = safe_path(root, output/"snapshots"/_stamp())
        directory.mkdir(parents=True, exist_ok=False)
        _json(directory/"config.json", config)
        for name in ("panel.parquet", "network_features.parquet", "network_audit.json"):
            shutil.copyfile(source/name, directory/name)
            if digest(directory/name) != identity[name]:
                raise ValueError("Changed copied input: "+name)
        _parquet(directory/"constraints.parquet", constraints)
        _parquet(directory/"zonal_labels.parquet", zonal)
        _json(directory/"constraint_audit.json", audit)
        _verify_files(source, identity, set(identity))
        if before != source_runner.protected_state(root):
            raise ValueError("Production changed during Prepare.")
        manifest = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
            config=config, settings=source_manifest["settings"], source_dir=str(source), source_files=identity,
            input_files={n: digest(directory/n) for n in sorted(INPUTS)}, training_code=training_code(root),
            runtime=runtime_identity(), protected_files=before, diagnostic_only=True,
            production_modified=False, activation_performed=False)
        _json(directory/"manifest.json", manifest)
        _json(directory/"status.json", {"status":"prepared", "snapshot":str(directory)})
        _json(output/"latest_prepared.json", {"snapshot":str(directory), "manifest_sha256":digest(directory/"manifest.json")})
        return directory


def read_snapshot(directory: Path, *, root: Path, check_code=True):
    directory = safe_path(root, directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    _verify_files(directory, manifest.get("input_files"), INPUTS)
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    validate_config(config)
    if config != manifest.get("config") or directory.parent != safe_path(root, config["output_root"])/"snapshots":
        raise ValueError("Snapshot/config divergent.")
    source = Path(manifest["source_dir"])
    if source != (root/config["source_suite"]).resolve():
        raise ValueError("Source/config divergent.")
    _verify_files(source, manifest["source_files"], set(manifest["source_files"]))
    if json.loads((source/"manifest.json").read_text(encoding="utf-8")).get("settings") != manifest.get("settings"):
        raise ValueError("Parametres de la source historique divergents.")
    if check_code and (manifest["training_code"] != training_code(root) or manifest["runtime"] != runtime_identity()):
        raise ValueError("Code/runtime modifie depuis Prepare. Creer un nouveau snapshot; ne pas resceller l'ancien.")
    evidence = json.loads((directory/'constraint_audit.json').read_text(encoding='utf8'))
    for path, expected in evidence.get('source_files', {}).items():
        if digest(Path(path)) != expected:
            raise ValueError('Original source evidence changed: '+str(path))
    return directory, config, manifest


class FitCache:
    """A fold checkpoint contains both variants, sealed to this exact snapshot."""
    def __init__(self, directory: Path, identity: str, *, root: Path):
        self.directory = safe_path(root, directory/"fits")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity, self.root = identity, root
        self.saved = self.reused = 0

    def _paths(self, day):
        if (not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day)
                or pd.Timestamp(day).strftime("%Y-%m-%d") != day):
            raise ValueError("Date de checkpoint invalide.")
        return tuple(safe_path(self.root, self.directory/(day+suffix)) for suffix in (".json", ".joblib"))

    def load(self, day):
        seal, path = self._paths(day)
        if not seal.exists():
            return None
        record = json.loads(seal.read_text(encoding="utf-8"))
        if (record.get("fit_day") != day or record.get("suite_manifest_sha256") != self.identity
                or not path.is_file() or record.get("model_sha256") != digest(path)):
            raise ValueError(f"Checkpoint invalide: {day}; chargement refuse.")
        state = joblib.load(path)
        if not isinstance(state, dict) or state.get("fit_day") != day:
            raise ValueError("Date de l'etat charge divergente.")
        self.reused += 1
        return state

    def save(self, day, state):
        seal, path = self._paths(day)
        if not isinstance(state, dict) or state.get("fit_day") != day or seal.exists():
            raise ValueError("Checkpoint deja scelle ou date divergente.")
        temporary = safe_path(self.root, path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp"))
        try:
            joblib.dump(state, temporary, compress=3)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        _json(seal, {"fit_day": day, "suite_manifest_sha256": self.identity, "model_sha256": digest(path)})
        self.saved += 1


def validate_predictions(panel: pd.DataFrame, predictions: pd.DataFrame):
    pd.testing.assert_frame_equal(panel.reset_index(drop=True), predictions[panel.columns].reset_index(drop=True), check_exact=True)
    valid = np.isfinite(panel.forecast.to_numpy(float))
    for model in MODELS:
        values = predictions[model].to_numpy(float)
        if not np.array_equal(np.isfinite(values), valid) or np.isinf(values).any():
            raise ValueError(f"Support de prediction divergent: {model}.")
        for lower_name, upper_name in ((model+"_q10", model+"_q90"), (model+"_p10", model+"_p90")):
            if lower_name in predictions or upper_name in predictions:
                if lower_name not in predictions or upper_name not in predictions:
                    raise ValueError("Une distribution publiee exige ses deux bornes.")
                lo, hi = predictions[lower_name].to_numpy(float), predictions[upper_name].to_numpy(float)
                if (not np.array_equal(np.isfinite(lo), valid) or not np.array_equal(np.isfinite(hi), valid)
                        or (lo[valid] > values[valid]).any() or (values[valid] > hi[valid]).any()):
                    raise ValueError(f"Quantiles non finis ou croises: {model}.")


def verify_result(directory: Path, manifest: dict):
    result = json.loads((directory/"results_manifest.json").read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("suite_manifest_sha256") != digest(directory/"manifest.json"):
        raise ValueError("Resultats non lies au snapshot courant.")
    _verify_files(directory, result.get("result_files"), RESULTS)
    if result.get('activation_manifest_sha256') != digest(directory/'activation_manifest.json'):
        raise ValueError('Residual results and activation diagnostics belong to different replays.')
    return result


def verify_activation(directory):
    manifest = json.loads((directory/"activation_manifest.json").read_text(encoding="utf8"))
    if manifest.get("suite_manifest_sha256") != digest(directory/"manifest.json"):
        raise ValueError("Activation predictions linked to another snapshot.")
    _verify_files(directory, manifest.get("files"), ACTIVATION_FILES)


def evaluate(directory: Path, *, root: Path) -> Path:
    from .activation import run_activation, aggregate_signals
    from .regional import run_regional, REGIONAL_SIGNALS
    from .policy import run_replay, SIGNALS
    directory, config, manifest = read_snapshot(directory, root=root)
    with exclusive_process_lock(safe_path(root, directory/"run.lock")):
        if (directory/"results_manifest.json").exists():
            verify_activation(directory)
            verify_result(directory, manifest)
            LOGGER.info("[Congestion] Replay completed and verified; zero refits.")
            return directory
        before = source_runner.protected_state(root)
        identity = digest(directory/"manifest.json")
        activation_cache = FitCache(directory/"activation", identity, root=root)
        regional_cache = FitCache(directory/"regional", identity, root=root)
        residual_cache = FitCache(directory/"residual", identity, root=root)
        def progress(message):
            LOGGER.info("[Congestion] %s", message)
            _json(directory/"status.json", dict(status="running", progress=message,
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
                activation_fits_saved=activation_cache.saved, activation_fits_reused=activation_cache.reused,
                regional_fits_saved=regional_cache.saved, regional_fits_reused=regional_cache.reused,
                residual_fits_saved=residual_cache.saved, residual_fits_reused=residual_cache.reused))
        try:
            panel = pd.read_parquet(directory/"panel.parquet")
            if not (directory/"activation_manifest.json").exists():
                predictions, folds, audit = run_activation(panel, pd.read_parquet(directory/"constraints.parquet"),
                    threads=config["options"]["threads"], load_fit=activation_cache.load,
                    save_fit=activation_cache.save, progress=progress)
                if not predictions.expert_ready.any():
                    raise ValueError("No trained congestion predictions; insufficient chronological qualified labels.")
                signals = aggregate_signals(panel, predictions)
                regional_predictions, regional_folds, regional_signals, regional_audit = run_regional(
                    panel, pd.read_parquet(directory/'network_features.parquet'),
                    pd.read_parquet(directory/'zonal_labels.parquet'), manifest['settings'],
                    threads=config['options']['threads'], load_fit=regional_cache.load,
                    save_fit=regional_cache.save, progress=progress)
                signals['congestion_ready'] &= regional_signals.regional_ready
                for name in REGIONAL_SIGNALS:
                    signals[name] = regional_signals[name]
                signals.loc[~signals.congestion_ready, SIGNALS] = np.nan
                _parquet(directory/'regional_predictions.parquet', regional_predictions)
                _parquet(directory/'regional_folds.parquet', _audit_parquet_frame(regional_folds))
                _json(directory/'regional_audit.json', regional_audit)
                _parquet(directory/"constraint_predictions.parquet", predictions)
                _parquet(directory/"activation_folds.parquet", _audit_parquet_frame(folds))
                _parquet(directory/"signals.parquet", signals)
                _json(directory/"activation_audit.json", audit)
                read_snapshot(directory, root=root)
                _json(directory/"activation_manifest.json", dict(suite_manifest_sha256=identity,
                      files={n: digest(directory/n) for n in sorted(ACTIVATION_FILES)}))
            verify_activation(directory)
            progress("Activation OOF completed; fitting paired residual control and congestion models.")
            result = run_replay(panel, pd.read_parquet(directory/"network_features.parquet"),
                pd.read_parquet(directory/"signals.parquet"), manifest["settings"],
                threads=config["options"]["threads"], load_fit=residual_cache.load,
                save_fit=residual_cache.save, progress=progress)
            validate_predictions(panel, result["predictions"])
            if not residual_cache.saved and not residual_cache.reused:
                raise ValueError("Insufficient OOF history to train residual candidates; activation diagnostics retained.")
            for name in ("predictions", "folds", "governance"):
                _parquet(directory/(name+".parquet"), result[name] if name=="predictions" else _audit_parquet_frame(result[name]))
            _json(directory/"model_audit.json", {**result["audit"],
                "data":json.loads((directory/"constraint_audit.json").read_text(encoding="utf8")),
                "activation":json.loads((directory/"activation_audit.json").read_text(encoding="utf8")),
                "regional_audit":json.loads((directory/'regional_audit.json').read_text(encoding='utf8')),
                "fits_saved_this_run":activation_cache.saved+regional_cache.saved+residual_cache.saved,
                "fits_reused_this_run":activation_cache.reused+regional_cache.reused+residual_cache.reused})
            read_snapshot(directory, root=root)
            if before != source_runner.protected_state(root) or digest(directory/"manifest.json") != identity:
                raise ValueError("Concurrent changes; publication refused.")
            _json(directory/"results_manifest.json", dict(status="completed", suite_manifest_sha256=identity,
                  activation_manifest_sha256=digest(directory/'activation_manifest.json'),
                  result_files={n:digest(directory/n) for n in sorted(RESULTS)}))
            _json(directory/"status.json", dict(status="evaluated", snapshot=str(directory)))
            return directory
        except BaseException as exc:
            _json(directory/"status.json", dict(status="failed", error=str(exc), snapshot=str(directory)))
            raise


def report(directory: Path, *, root: Path) -> Path:
    from .report import build_report
    directory, config, manifest = read_snapshot(directory, root=root)
    verify_result(directory, manifest)
    verify_activation(directory)
    with exclusive_process_lock(safe_path(root, directory/"report.lock")):
        before = source_runner.protected_state(root)
        names = [p.relative_to(root).as_posix() for package in ("kpi_report", "economic_value") for p in sorted((root/package).glob("*.py"))]
        names += ["nyx_congestion/report.py", "nyx_physical_p50/report.py", "config/economic_value.yaml"]
        code = {n: digest(root/n) for n in names}
        destination = safe_path(root, directory/"reports"/_stamp())
        audit = json.loads((directory/"model_audit.json").read_text(encoding="utf-8"))
        audit['data_audit'] = audit.pop('data')
        audit['activation_audit'] = audit.pop('activation')
        result = build_report(pd.read_parquet(directory/"predictions.parquet"), Path(manifest["source_dir"]), destination,
                              root=root, constraint_predictions=pd.read_parquet(directory/"constraint_predictions.parquet"),
                              zonal_labels=pd.read_parquet(directory/'zonal_labels.parquet'),
                              regional_predictions=pd.read_parquet(directory/'regional_predictions.parquet'),
                              audit={**audit, "snapshot": str(directory), "suite_manifest_sha256": digest(directory/"manifest.json")})
        read_snapshot(directory, root=root)
        verify_result(directory, manifest)
        if before != source_runner.protected_state(root) or code != {n: digest(root/n) for n in names}:
            raise ValueError("Code, politique economique ou production modifies pendant le rapport; publication refusee.")
        files = {p.relative_to(destination).as_posix(): digest(p) for p in destination.rglob("*") if p.is_file()}
        if not any(n.endswith(".html") for n in files):
            raise ValueError("Rapport HTML manquant.")
        _json(destination/"report_manifest.json", {"status": "completed", "suite_manifest_sha256": digest(directory/"manifest.json"),
              "results_manifest_sha256": digest(directory/"results_manifest.json"), "reporting_code": code,
              "files": files, "result": result, "production_modified": False})
        pointer = {"snapshot": str(directory), "report_directory": str(destination), "manifest_sha256": digest(directory/"manifest.json"),
                   "report_manifest_sha256": digest(destination/"report_manifest.json")}
        _json(directory/"latest_report.json", pointer)
        _json(safe_path(root, config["output_root"])/"latest.json", pointer)
        _json(directory/"status.json", {"status": "completed", **pointer})
        return destination


def resolve_snapshot(config: dict, *, root: Path, value=None, create=False):
    validate_config(config)
    if value:
        directory = safe_path(root, value)
        _, frozen, _ = read_snapshot(directory, root=root)
        if frozen != config:
            raise ValueError("Configuration differente du snapshot explicite.")
        return directory
    pointer = safe_path(root, config["output_root"])/"latest_prepared.json"
    if not pointer.exists():
        if create:
            return prepare(config, root=root)
        raise ValueError("Aucun snapshot prepare. Lancer -Action Prepare ou -Action Run.")
    data = json.loads(pointer.read_text(encoding="utf-8"))
    directory = safe_path(root, data["snapshot"])
    if data.get("manifest_sha256") != digest(directory/"manifest.json"):
        raise ValueError("Pointeur du snapshot divergent.")
    _, frozen, _ = read_snapshot(directory, root=root)
    if frozen != config:
        raise ValueError("Configuration modifiee: lancer Prepare pour une nouvelle experience.")
    return directory


def status(directory: Path, *, root: Path):
    directory, _, manifest = read_snapshot(directory, root=root)
    result = json.loads((directory/"status.json").read_text(encoding="utf-8"))
    if (directory/"results_manifest.json").exists():
        verify_result(directory, manifest)
        verify_activation(directory)
        result["results_verified"] = True
    if (directory/"latest_report.json").exists():
        pointer = json.loads((directory/"latest_report.json").read_text(encoding="utf-8"))
        target = safe_path(root, pointer["report_directory"])
        if (pointer.get("snapshot") != str(directory) or target.parent != directory/"reports"
                or pointer.get("manifest_sha256") != digest(directory/"manifest.json")
                or digest(target/"report_manifest.json") != pointer.get("report_manifest_sha256")):
            raise ValueError("Pointeur/SHA du rapport divergent.")
        sealed = json.loads((target/"report_manifest.json").read_text(encoding="utf-8"))
        if (sealed.get("status") != "completed" or sealed.get("suite_manifest_sha256") != digest(directory/"manifest.json")
                or sealed.get("results_manifest_sha256") != digest(directory/"results_manifest.json")):
            raise ValueError("Rapport non lie aux resultats courants.")
        _verify_files(target, sealed["files"], set(sealed["files"]))
        result.update(report_directory=str(target), report_verified=True)
    return {**result, "snapshot": str(directory), "models": list(MODELS), "production_modified": False}
