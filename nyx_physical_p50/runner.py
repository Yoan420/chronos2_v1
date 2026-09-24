"""Isolated, sealed replay of physical P50 candidates; production is read-only."""
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
from nyx_rmse.runner import inspect_source, runtime_identity, _verify_files
from nyx_scarcity.runner import digest, _json, _parquet
from nyx_scarcity.variants_runner import _audit_parquet_frame

LOGGER = logging.getLogger(__name__)
NAMESPACE = Path("runs/experiments/nyx_physical_p50_v1")
INPUTS = {"config.json", "panel.parquet", "source_predictions.parquet", "source_folds.parquet",
          "network_features.parquet", "network_audit.json"}
RESULTS = {"predictions.parquet", "folds.parquet", "governance.parquet", "model_audit.json"}
MODELS = ("fuel_transport_direct", "fuel_transport_governed", "network_fuel_direct", "nyx_physical_p50")


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
        raise ValueError("Configuration Physical P50 schema 1 complete requise; champs inconnus refuses.")
    if any(not isinstance(config[k], str) or not config[k].strip() for k in ("source_suite", "output_root")):
        raise ValueError("Source et dossier de sortie explicites requis.")
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
    names = [p.relative_to(root).as_posix() for p in sorted((root/"nyx_physical_p50").glob("*.py"))
             if p.name != "report.py"]
    names += ["nyx_rmse/runner.py", "nyx_rmse/policy.py", "nyx_stress_guard/intervals.py",
              "marginal_cost_expert/network.py", "chronos2_hourly/jao_flowbased.py",
              "run_nyx_physical_p50.py", "NyxPhysical.ps1"]
    return {**source_runner.code_seals(root), **{name: digest(root/name) for name in names}}


def _network_inputs(panel: pd.DataFrame, *, root: Path, config: dict):
    from .network import prepare_network_features
    return prepare_network_features(panel, raw_roots=[root/"data/pit/jao_core_flowbased",
        root/"data/pit/marginal_cost_expert_v2/network", root/NAMESPACE])


def collect(config: dict, *, root: Path, start_day: str, end_day: str, ca_bundle=None) -> dict:
    """Explicit bounded collection only; Run/Prepare never fetch external data."""
    from . import network
    validate_config(config)
    for day in (start_day, end_day):
        if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("Collect exige --start-day et --end-day au format AAAA-MM-JJ.")
        if pd.Timestamp(day).strftime("%Y-%m-%d") != day:
            raise ValueError("Date de collecte invalide.")
    days = (pd.Timestamp(end_day)-pd.Timestamp(start_day)).days+1
    if not 1 <= days <= 30:
        raise ValueError("Collect est limite a une plage explicite de 1 a 30 jours.")
    output = safe_path(root, config["output_root"])
    with exclusive_process_lock(safe_path(root, output/"collect.lock")):
        before = source_runner.protected_state(root)
        factory = None
        if ca_bundle:
            import ssl
            from chronos2_hourly.jao_flowbased import JaoCoreClient
            bundle = Path(ca_bundle)
            if not bundle.is_file():
                raise ValueError(f"Certificat introuvable: {bundle}")
            context = ssl.create_default_context(cafile=str(bundle))
            factory = lambda: JaoCoreClient(verify=context, timeout_seconds=45., maximum_retries=2,
                                            request_interval_seconds=.65, page_size=40000)
        result = network.capture_network_days(pd.date_range(start_day, end_day, freq="D").strftime("%Y-%m-%d").tolist(),
            output_root=safe_path(root, root/NAMESPACE/"raw"), client_factory=factory)
        if before != source_runner.protected_state(root):
            raise ValueError("La production a change pendant Collect; ne pas utiliser la collecte sans verification.")
        audit_path = safe_path(root, output/"collections"/(_stamp()+".json"))
        _json(audit_path, result)
        return {"status": result["status"], "new_days": result["new_days"], "reused_days": result["reused_days"],
                "output_root": result["output_root"], "audit_path": str(audit_path),
                "audit_sha256": digest(audit_path), "production_modified": False}


def prepare(config: dict, *, root: Path) -> Path:
    validate_config(config)
    output = safe_path(root, config["output_root"])
    with exclusive_process_lock(safe_path(root, output/"prepare.lock")):
        before = source_runner.protected_state(root)
        source, source_manifest, identity = inspect_source(root/config["source_suite"], root=root)
        panel = pd.read_parquet(source/"panel.parquet")
        network_features, network_audit = _network_inputs(panel, root=root, config=config)
        directory = safe_path(root, output/"snapshots"/_stamp())
        directory.mkdir(parents=True, exist_ok=False)
        _json(directory/"config.json", config)
        for name in ("panel.parquet", "source_predictions.parquet", "source_folds.parquet"):
            shutil.copyfile(source/name, directory/name)
            if digest(directory/name) != identity[name]:
                raise ValueError(f"Copie d'entree divergente: {name}.")
        _parquet(directory/"network_features.parquet", network_features)
        _json(directory/"network_audit.json", network_audit)
        _verify_files(source, identity, set(identity))
        if before != source_runner.protected_state(root):
            raise ValueError("La production a change pendant Prepare; snapshot non publie.")
        manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": config, "settings": source_manifest["settings"], "source_dir": str(source),
                    "source_files": identity, "input_files": {n: digest(directory/n) for n in sorted(INPUTS)},
                    "training_code": training_code(root), "runtime": runtime_identity(), "protected_files": before,
                    "diagnostic_only": True, "production_modified": False, "activation_performed": False}
        _json(directory/"manifest.json", manifest)
        _json(directory/"status.json", {"status": "prepared", "snapshot": str(directory)})
        _json(output/"latest_prepared.json", {"snapshot": str(directory), "manifest_sha256": digest(directory/"manifest.json")})
        LOGGER.info("[Physical P50] Snapshot prepare: %s", directory)
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
    return result


def evaluate(directory: Path, *, root: Path) -> Path:
    from .policy import run_replay
    directory, config, manifest = read_snapshot(directory, root=root)
    with exclusive_process_lock(safe_path(root, directory/"run.lock")):
        if (directory/"results_manifest.json").exists():
            verify_result(directory, manifest)
            LOGGER.info("[Physical P50] Replay scelle deja termine: aucun fit.")
            return directory
        before = source_runner.protected_state(root)
        identity = digest(directory/"manifest.json")
        cache = FitCache(directory, identity, root=root)
        try:
            def progress(message):
                LOGGER.info("[Physical P50] %s", message)
                _json(directory/"status.json", {"status": "running", "progress": message,
                      "updated_at_utc": datetime.now(timezone.utc).isoformat(), "fits_saved": cache.saved, "fits_reused": cache.reused})
            progress("Entrees locales figees; aucun run Chronos ni appel API.")
            panel = pd.read_parquet(directory/"panel.parquet")
            result = run_replay(panel, pd.read_parquet(directory/"source_predictions.parquet"),
                                pd.read_parquet(directory/"source_folds.parquet"),
                                pd.read_parquet(directory/"network_features.parquet"), manifest["settings"], config["options"],
                                load_fit=cache.load, save_fit=cache.save, progress=progress)
            validate_predictions(panel, result["predictions"])
            if not cache.saved and not cache.reused:
                raise ValueError("Aucun fold entraine: historique insuffisant; aucun candidat publie.")
            for name in ("predictions", "folds", "governance"):
                frame = result[name] if name == "predictions" else _audit_parquet_frame(result[name])
                _parquet(directory/(name+".parquet"), frame)
            _json(directory/"model_audit.json", {**result["audit"], "network_audit": json.loads((directory/"network_audit.json").read_text(encoding="utf-8")),
                  "fits_saved_this_run": cache.saved, "fits_reused_this_run": cache.reused,
                  "diagnostic_only": True, "production_modified": False, "activation_performed": False,
                  "evaluation_year_already_examined": True, "annual_non_regression_guaranteed": False})
            read_snapshot(directory, root=root)
            if before != source_runner.protected_state(root) or identity != digest(directory/"manifest.json"):
                raise ValueError("Modification concurrente: publication refusee.")
            _json(directory/"results_manifest.json", {"status": "completed", "suite_manifest_sha256": identity,
                  "result_files": {n: digest(directory/n) for n in sorted(RESULTS)}})
            _json(directory/"status.json", {"status": "evaluated", "snapshot": str(directory)})
            return directory
        except BaseException as exc:
            _json(directory/"status.json", {"status": "failed", "error": str(exc), "fits_saved": cache.saved, "fits_reused": cache.reused})
            raise


def report(directory: Path, *, root: Path) -> Path:
    from .report import build_report
    directory, config, manifest = read_snapshot(directory, root=root)
    verify_result(directory, manifest)
    with exclusive_process_lock(safe_path(root, directory/"report.lock")):
        before = source_runner.protected_state(root)
        names = [p.relative_to(root).as_posix() for package in ("kpi_report", "economic_value") for p in sorted((root/package).glob("*.py"))]
        names += ["nyx_physical_p50/report.py", "config/economic_value.yaml"]
        code = {n: digest(root/n) for n in names}
        destination = safe_path(root, directory/"reports"/_stamp())
        audit = json.loads((directory/"model_audit.json").read_text(encoding="utf-8"))
        result = build_report(pd.read_parquet(directory/"predictions.parquet"), Path(manifest["source_dir"]), destination,
                              root=root, audit={**audit, "snapshot": str(directory), "suite_manifest_sha256": digest(directory/"manifest.json")})
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
