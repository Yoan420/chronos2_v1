"""Sealed calibration-only replay; existing forecasts and stage one are read-only."""
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
from nyx_coherent_p50 import runner as protected_runner
from nyx_congestion import runner as source_runner
from nyx_rmse.runner import runtime_identity, _verify_files
from nyx_scarcity.runner import digest, _json, _parquet
from nyx_scarcity.variants_runner import _audit_parquet_frame

LOGGER = logging.getLogger(__name__)
NAMESPACE = Path("runs/experiments/nyx_congestion_calibration_v1")
COPIED_INPUTS = {"panel.parquet", "network_features.parquet", "signals.parquet"}
INPUTS = {"config.json", *COPIED_INPUTS}
RESULTS = {"predictions.parquet", "folds.parquet", "governance.parquet", "model_audit.json"}
SOURCE_FILES = {*source_runner.INPUTS, *source_runner.ACTIVATION_FILES, *source_runner.RESULTS,
                "manifest.json", "activation_manifest.json", "results_manifest.json"}
MODELS = ("calibrated_control_direct", "calibrated_control_governed",
          "congestion_calibrated_direct", "nyx_congestion_calibrated")


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]


def safe_path(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    namespace = root/NAMESPACE
    value = Path(value)
    path = (value if value.is_absolute() else root/value).absolute()
    if path != path.resolve() or namespace != namespace.resolve() or not path.is_relative_to(namespace):
        raise ValueError("Sorties exclusivement dans nyx_congestion_calibration_v1; alias et production interdits.")
    return path


def validate_config(config):
    expected = {"schema_version", "source_suite", "output_root", "options", "diagnostic_only",
                "production_modified", "activation_performed"}
    if (not isinstance(config, dict) or set(config) != expected
            or type(config["schema_version"]) is not int or config["schema_version"] != 1):
        raise ValueError("Configuration calibration schema 1 complete requise, sans champs inconnus.")
    if any(not isinstance(config[k], str) or not config[k].strip() for k in ("source_suite", "output_root")):
        raise ValueError("Source et sortie explicites requises.")
    if Path(config["output_root"]).as_posix() != NAMESPACE.as_posix():
        raise ValueError("Le namespace calibration est fixe et distinct des experiences anterieures.")
    if (config["diagnostic_only"] is not True or config["production_modified"] is not False
            or config["activation_performed"] is not False):
        raise ValueError("Laboratoire uniquement diagnostique; modification et activation interdites.")
    options = config["options"]
    if (not isinstance(options, dict) or set(options) != {"threads"}
            or type(options["threads"]) is not int or options["threads"] not in (1, 2)):
        raise ValueError("Seul options.threads=1 ou 2 est modifiable; politique de calibration fixe.")


def load_config(path: Path):
    result = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    validate_config(result)
    return result


def training_code(root: Path):
    names = [p.relative_to(root).as_posix() for p in sorted((root/"nyx_congestion_calibration").glob("*.py"))
             if p.name != "report.py"]
    names += ["run_nyx_congestion_calibration.py", "NyxCongestionCalibration.ps1"]
    return {**source_runner.training_code(root), **{n: digest(root/n) for n in names}}


def inspect_source(config, *, root):
    source, _, manifest = source_runner.read_snapshot(root/Path(config["source_suite"]), root=root)
    source_runner.verify_activation(source)
    source_runner.verify_result(source, manifest)
    return source, manifest, {n: digest(source/n) for n in sorted(SOURCE_FILES)}


def prepare(config, *, root: Path):
    validate_config(config)
    output = safe_path(root, config["output_root"])
    with exclusive_process_lock(safe_path(root, output/"prepare.lock")):
        before = protected_runner.protected_state(root)
        source, source_manifest, identity = inspect_source(config, root=root)
        directory = safe_path(root, output/"snapshots"/_stamp())
        directory.mkdir(parents=True, exist_ok=False)
        _json(directory/"config.json", config)
        for name in COPIED_INPUTS:
            shutil.copyfile(source/name, safe_path(root, directory/name))
            if digest(directory/name) != identity[name]:
                raise ValueError("Copie differente de la source scellee : "+name)
        _verify_files(source, identity, SOURCE_FILES)
        if before != protected_runner.protected_state(root):
            raise ValueError("Production modifiee pendant Prepare; publication refusee.")
        manifest = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
            config=config, settings=source_manifest["settings"], source_dir=str(source), source_files=identity,
            input_files={n:digest(directory/n) for n in sorted(INPUTS)}, training_code=training_code(root),
            runtime=runtime_identity(), protected_files=before, stage1_retrained=False,
            diagnostic_only=True, production_modified=False, activation_performed=False)
        _json(directory/"manifest.json", manifest)
        _json(directory/"status.json", {"status":"prepared", "snapshot":str(directory), "stage1_retrained":False})
        _json(safe_path(root, output/"latest_prepared.json"),
              {"snapshot":str(directory), "manifest_sha256":digest(directory/"manifest.json")})
        return directory


def read_snapshot(directory: Path, *, root: Path, check_code=True):
    directory = safe_path(root, directory)
    manifest = json.loads(safe_path(root, directory/"manifest.json").read_text(encoding="utf8"))
    _verify_files(directory, manifest.get("input_files"), INPUTS)
    config = json.loads((directory/"config.json").read_text(encoding="utf8"))
    validate_config(config)
    if (config != manifest.get("config") or directory.parent != safe_path(root, config["output_root"])/"snapshots"
            or manifest.get("schema_version") != 1 or manifest.get("stage1_retrained") is not False
            or manifest.get("diagnostic_only") is not True or manifest.get("production_modified") is not False
            or manifest.get("activation_performed") is not False):
        raise ValueError("Snapshot/config/contrat diagnostique divergent.")
    source = source_runner.safe_path(root, Path(manifest["source_dir"]))
    if source != (root/Path(config["source_suite"])).resolve():
        raise ValueError("Source/config divergent.")
    _verify_files(source, manifest.get("source_files"), SOURCE_FILES)
    if any(manifest["input_files"][n] != manifest["source_files"][n] for n in COPIED_INPUTS):
        raise ValueError("Les entrees et signaux stage1 doivent rester strictement identiques a la source.")
    source_manifest = json.loads((source/"manifest.json").read_text(encoding="utf8"))
    if source_manifest.get("settings") != manifest.get("settings"):
        raise ValueError("Parametres de base divergents.")
    source_runner.verify_activation(source)
    source_runner.verify_result(source, source_manifest)
    if check_code and (manifest.get("training_code") != training_code(root) or manifest.get("runtime") != runtime_identity()):
        raise ValueError("Code/runtime modifie depuis Prepare; creer un nouveau snapshot, sans resceller l'ancien.")
    return directory, config, manifest


class FitCache:
    """Private verified checkpoint, never writes to the original stage-one cache."""
    def __init__(self, directory, identity, *, root):
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
        record = json.loads(seal.read_text(encoding="utf8"))
        if (record.get("fit_day") != day or record.get("suite_manifest_sha256") != self.identity
                or not path.is_file() or record.get("model_sha256") != digest(path)):
            raise ValueError("Checkpoint invalide; chargement refuse.")
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
        _json(seal, {"fit_day":day, "suite_manifest_sha256":self.identity, "model_sha256":digest(path)})
        self.saved += 1


def validate_predictions(panel, predictions):
    pd.testing.assert_frame_equal(panel.reset_index(drop=True), predictions[panel.columns].reset_index(drop=True), check_exact=True)
    support = np.isfinite(panel.forecast.to_numpy(float))
    for name in MODELS:
        values = predictions[name].to_numpy(float)
        if not np.array_equal(np.isfinite(values), support) or np.isinf(values).any():
            raise ValueError("Support de prediction divergent : "+name)
        for suffixes in (("_q10", "_q90"), ("_p10", "_p90")):
            lo_name, hi_name = (name+s for s in suffixes)
            if lo_name not in predictions and hi_name not in predictions:
                continue
            if lo_name not in predictions or hi_name not in predictions:
                raise ValueError("Deux bornes de distribution requises.")
            lo, hi = predictions[lo_name].to_numpy(float), predictions[hi_name].to_numpy(float)
            if (not np.array_equal(np.isfinite(lo), support) or not np.array_equal(np.isfinite(hi), support)
                    or np.isinf(lo).any() or np.isinf(hi).any()
                    or (lo[support] > values[support]).any() or (values[support] > hi[support]).any()):
                raise ValueError("Quantiles non finis ou croises : "+name)


def verify_result(directory, manifest):
    result = json.loads((directory/"results_manifest.json").read_text(encoding="utf8"))
    if (result.get("status") != "completed" or result.get("suite_manifest_sha256") != digest(directory/"manifest.json")
            or result.get("source_activation_manifest_sha256") != manifest["source_files"]["activation_manifest.json"]):
        raise ValueError("Resultats non lies au snapshot et au stage1 figes.")
    _verify_files(directory, result.get("result_files"), RESULTS)
    return result


def evaluate(directory: Path, *, root: Path):
    from .policy import run_replay
    directory, config, manifest = read_snapshot(directory, root=root)
    with exclusive_process_lock(safe_path(root, directory/"run.lock")):
        if (directory/"results_manifest.json").exists():
            verify_result(directory, manifest)
            LOGGER.info("[Calibration] Replay termine et verifie; aucun nouvel entrainement.")
            return directory
        before = protected_runner.protected_state(root)
        identity = digest(directory/"manifest.json")
        cache = FitCache(directory/"residual", identity, root=root)
        def progress(message):
            LOGGER.info("[Calibration] %s", message)
            _json(directory/"status.json", dict(status="running", progress=message,
                updated_at_utc=datetime.now(timezone.utc).isoformat(), stage1_retrained=False,
                residual_fits_saved=cache.saved, residual_fits_reused=cache.reused))
        try:
            panel = pd.read_parquet(directory/"panel.parquet")
            progress("Stage1 fige reutilise; ajustement du correcteur et de sa calibration uniquement.")
            result = run_replay(panel, pd.read_parquet(directory/"network_features.parquet"),
                pd.read_parquet(directory/"signals.parquet"), manifest["settings"],
                threads=config["options"]["threads"], load_fit=cache.load, save_fit=cache.save, progress=progress)
            validate_predictions(panel, result["predictions"])
            if not cache.saved and not cache.reused:
                raise ValueError("Historique OOF insuffisant : aucun correcteur calibre entraine.")
            for name in ("predictions", "folds", "governance"):
                _parquet(safe_path(root, directory/(name+".parquet")), result[name] if name == "predictions" else _audit_parquet_frame(result[name]))
            _json(directory/"model_audit.json", {**result["audit"], "stage1_retrained":False,
                "source_snapshot":manifest["source_dir"],
                "source_stage1_manifest_sha256":manifest["source_files"]["activation_manifest.json"],
                "fits_saved_this_run":cache.saved, "fits_reused_this_run":cache.reused})
            read_snapshot(directory, root=root)
            if before != protected_runner.protected_state(root) or identity != digest(directory/"manifest.json"):
                raise ValueError("Modifications concurrentes; publication refusee.")
            _json(directory/"results_manifest.json", dict(status="completed", suite_manifest_sha256=identity,
                source_activation_manifest_sha256=manifest["source_files"]["activation_manifest.json"],
                result_files={n:digest(directory/n) for n in sorted(RESULTS)}))
            _json(directory/"status.json", dict(status="evaluated", snapshot=str(directory), stage1_retrained=False))
            return directory
        except BaseException as exc:
            _json(directory/"status.json", dict(status="failed", error=str(exc), snapshot=str(directory)))
            raise


def report(directory: Path, *, root: Path):
    from .report import build_report
    directory, config, manifest = read_snapshot(directory, root=root)
    verify_result(directory, manifest)
    with exclusive_process_lock(safe_path(root, directory/"report.lock")):
        before = protected_runner.protected_state(root)
        names = [p.relative_to(root).as_posix() for package in ("kpi_report", "economic_value") for p in sorted((root/package).glob("*.py"))]
        names += ["nyx_congestion_calibration/report.py", "nyx_congestion/report.py", "nyx_physical_p50/report.py", "config/economic_value.yaml"]
        code = {n:digest(root/n) for n in names}
        destination = safe_path(root, directory/"reports"/_stamp())
        audit = json.loads((directory/"model_audit.json").read_text(encoding="utf8"))
        result = build_report(pd.read_parquet(directory/"predictions.parquet"), Path(manifest["source_dir"]), destination,
            root=root, audit={**audit, "snapshot":str(directory), "suite_manifest_sha256":digest(directory/"manifest.json")})
        read_snapshot(directory, root=root)
        verify_result(directory, manifest)
        if before != protected_runner.protected_state(root) or code != {n:digest(root/n) for n in names}:
            raise ValueError("Code du rapport ou production modifies; publication refusee.")
        files = {p.relative_to(destination).as_posix():digest(safe_path(root, p)) for p in destination.rglob("*") if p.is_file()}
        if not any(n.endswith(".html") for n in files):
            raise ValueError("Rapport HTML manquant.")
        _json(destination/"report_manifest.json", dict(status="completed", suite_manifest_sha256=digest(directory/"manifest.json"),
            results_manifest_sha256=digest(directory/"results_manifest.json"), reporting_code=code,
            files=files, result=result, production_modified=False))
        pointer = dict(snapshot=str(directory), report_directory=str(destination), manifest_sha256=digest(directory/"manifest.json"),
            report_manifest_sha256=digest(destination/"report_manifest.json"))
        _json(directory/"latest_report.json", pointer)
        _json(safe_path(root, config["output_root"])/"latest.json", pointer)
        _json(directory/"status.json", {"status":"completed", **pointer})
        return destination


def resolve_snapshot(config, *, root: Path, value=None, create=False):
    validate_config(config)
    if value:
        directory = safe_path(root, value)
    else:
        pointer = safe_path(root, config["output_root"])/"latest_prepared.json"
        if not pointer.exists():
            if create:
                return prepare(config, root=root)
            raise ValueError("Aucun snapshot prepare. Lancer Prepare ou Run.")
        record = json.loads(pointer.read_text(encoding="utf8"))
        directory = safe_path(root, record["snapshot"])
        if record.get("manifest_sha256") != digest(directory/"manifest.json"):
            raise ValueError("Pointeur du snapshot divergent.")
    _, frozen, _ = read_snapshot(directory, root=root)
    if frozen != config:
        raise ValueError("Configuration differente du snapshot; creer une nouvelle experience avec Prepare.")
    return directory


def status(directory: Path, *, root: Path):
    directory, _, manifest = read_snapshot(directory, root=root)
    result = json.loads(safe_path(root, directory/"status.json").read_text(encoding="utf8"))
    if (directory/"results_manifest.json").exists():
        verify_result(directory, manifest)
        result["results_verified"] = True
    if (directory/"latest_report.json").exists():
        pointer = json.loads(safe_path(root, directory/"latest_report.json").read_text(encoding="utf8"))
        target = safe_path(root, pointer["report_directory"])
        if (pointer.get("snapshot") != str(directory) or target.parent != directory/"reports"
                or pointer.get("manifest_sha256") != digest(directory/"manifest.json")
                or pointer.get("report_manifest_sha256") != digest(target/"report_manifest.json")):
            raise ValueError("Pointeur/SHA du rapport divergent.")
        sealed = json.loads((target/"report_manifest.json").read_text(encoding="utf8"))
        if (sealed.get("status") != "completed" or sealed.get("suite_manifest_sha256") != digest(directory/"manifest.json")
                or sealed.get("results_manifest_sha256") != digest(directory/"results_manifest.json")):
            raise ValueError("Rapport non lie aux resultats courants.")
        for name in sealed.get("files", {}):
            if safe_path(root, target/name).parent == target.parent or not (target/name).resolve().is_relative_to(target):
                raise ValueError("Fichier de rapport hors destination.")
        _verify_files(target, sealed.get("files"), set(sealed.get("files", {})))
        result.update(report_directory=str(target), report_verified=True)
    return {**result, "snapshot":str(directory), "models":list(MODELS), "stage1_retrained":False, "production_modified":False}


def dry_run(config, *, root: Path):
    validate_config(config)
    return dict(status="dry_run", source=str(source_runner.safe_path(root, root/Path(config["source_suite"]))),
        output_root=str(safe_path(root, config["output_root"])), models=list(MODELS),
        stage1_retrained=False, api_calls=False, writes=False, production_modified=False)
