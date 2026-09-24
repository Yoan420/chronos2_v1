"""Frozen local inputs, per-fit checkpoints and isolated comparison reports."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
from pathlib import Path
import re
import shutil
import sys
import uuid

import joblib
import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock
from nyx_coherent_p50 import runner as source_runner
from nyx_scarcity.runner import digest, _json, _parquet
from nyx_scarcity.variants_runner import _audit_parquet_frame

LOGGER = logging.getLogger(__name__)
NAMESPACE = Path("runs/experiments/nyx_rmse_v1")
INPUTS = {"config.json", "panel.parquet", "source_predictions.parquet", "source_folds.parquet"}
RESULTS = {"predictions.parquet", "folds.parquet", "governance.parquet", "model_audit.json"}
MODELS = ("residual_mse_direct", "nyx_rmse", "mixture_mean_direct", "mixture_mean_governed")


def safe_path(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    namespace = root / NAMESPACE
    raw = Path(value)
    path = (raw if raw.is_absolute() else root / raw).absolute()
    if path != path.resolve() or namespace != namespace.resolve() or not path.is_relative_to(namespace):
        raise ValueError(f"Sorties exclusivement dans {namespace}; alias et chemins de production interdits.")
    return path


def validate_config(config: dict) -> None:
    from .policy import validate_options
    expected = {"schema_version", "source_suite", "output_root", "options", "diagnostic_only",
                "production_modified", "activation_performed"}
    if not isinstance(config, dict) or set(config) != expected or type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Configuration NYX RMSE schema 1 complete requise; champs inconnus refuses.")
    if any(not isinstance(config[k], str) or not config[k].strip() for k in ("source_suite", "output_root")):
        raise ValueError("source_suite et output_root explicites requis.")
    if config["diagnostic_only"] is not True or config["production_modified"] is not False or config["activation_performed"] is not False:
        raise ValueError("Laboratoire exclusivement diagnostique, sans activation.")
    validate_options(config["options"])


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def training_code(root: Path) -> dict:
    # Rendering may evolve without invalidating expensive numerical checkpoints.
    names = ["nyx_rmse/__init__.py", "nyx_rmse/runner.py", "nyx_rmse/models.py", "nyx_rmse/policy.py",
             "run_nyx_rmse.py", "NyxRMSE.ps1"]
    return {**source_runner.code_seals(root), **{n: digest(root/n) for n in names}}


def runtime_identity() -> dict:
    return {"python": sys.version, "executable": sys.executable,
            "versions": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "pyarrow", "joblib", "threadpoolctl")}}


def _verify_files(directory: Path, values: dict, expected: set[str]) -> None:
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError("Manifeste de fichiers incomplet.")
    for name, checksum in values.items():
        path = directory/name
        if path != path.resolve() or not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Fichier absent, alias ou SHA divergent: {path}")


def inspect_source(path: Path, *, root: Path) -> tuple[Path, dict, dict]:
    source, _, manifest = source_runner.read_suite(path, root=root)
    for name in ("forest", "empirical"):
        source_runner.verify_result(source, name, manifest)
    comparison = json.loads((source/"comparison_manifest.json").read_text(encoding="utf-8"))
    if (comparison.get("status") != "completed"
            or comparison.get("suite_manifest_sha256") != digest(source/"manifest.json")
            or comparison.get("comparison_sha256") != digest(source/"comparison.json")
            or comparison.get("variant_manifests") != {n: digest(source/n/"results_manifest.json") for n in ("forest", "empirical")}):
        raise ValueError("La comparaison source n'est pas scellee ou complete.")
    # Recreating the same conditional CDF requires its original implementation.
    for name in ("nyx_coherent_p50/distribution.py", "nyx_coherent_p50/policy.py",
                 "nyx_fundamental_stress/features.py", "nyx_scarcity/policy.py"):
        if manifest["code_sha256"].get(name) != digest(root/name):
            raise ValueError(f"Code historique divergent: {name}; pas de reconstruction silencieuse.")
    for package in ("numpy", "pandas", "scikit-learn"):
        if manifest.get("runtime", {}).get("versions", {}).get(package) != version(package):
            raise ValueError(f"Runtime historique divergent: {package}; CDF non reproductible a l'identique.")
    names = set(source_runner.INPUTS) | {"manifest.json", "comparison.json", "comparison_manifest.json"}
    for variant in ("forest", "empirical"):
        names.update(f"{variant}/{name}" for name in source_runner.RESULTS | {"results_manifest.json"})
    return source, manifest, {n: digest(source/n) for n in sorted(names)}


def prepare(config: dict, *, root: Path) -> Path:
    validate_config(config)
    output = safe_path(root, config["output_root"])
    with exclusive_process_lock(safe_path(root, output/"prepare.lock")):
        before = source_runner.protected_state(root)
        source, source_manifest, identity = inspect_source(root/config["source_suite"], root=root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
        directory = safe_path(root, output/"snapshots"/stamp)
        directory.mkdir(parents=True, exist_ok=False)
        _json(directory/"config.json", config)
        for name in sorted(INPUTS - {"config.json"}):
            shutil.copyfile(source/name, directory/name)
            if digest(directory/name) != identity[name]:
                raise ValueError(f"Copie d'entree divergente: {name}.")
        _verify_files(source, identity, set(identity))
        if before != source_runner.protected_state(root):
            raise ValueError("La production a change pendant Prepare; snapshot non publie.")
        manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": config, "settings": source_manifest["settings"], "source_dir": str(source),
                    "source_files": identity, "input_files": {n: digest(directory/n) for n in sorted(INPUTS)},
                    "training_code": training_code(root), "runtime": runtime_identity(), "protected_files": before,
                    "diagnostic_only": True, "activation_performed": False, "production_modified": False}
        _json(directory/"manifest.json", manifest)
        _json(directory/"status.json", {"status": "prepared", "snapshot": str(directory)})
        _json(output/"latest_prepared.json", {"snapshot": str(directory), "manifest_sha256": digest(directory/"manifest.json")})
        LOGGER.info("[NYX RMSE] Snapshot prepare: %s", directory)
        return directory


def read_snapshot(directory: Path, *, root: Path, check_code: bool = True):
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
    source_manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("settings") != manifest.get("settings"):
        raise ValueError("Parametres historiques divergents.")
    _verify_files(source, manifest["source_files"], set(manifest["source_files"]))
    if check_code and (manifest["training_code"] != training_code(root) or manifest["runtime"] != runtime_identity()):
        raise ValueError("Code/runtime modifie depuis Prepare. Creer un nouveau snapshot, ne pas resceller l'ancien.")
    return directory, config, manifest


class FitCache:
    """Only deserialize locally created, hash-verified states bound to this run."""
    def __init__(self, directory: Path, identity: str, *, root: Path):
        self.directory = safe_path(root, directory/"fits")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity, self.root = identity, root
        self.reused = self.saved = 0

    def _paths(self, day):
        if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) or pd.Timestamp(day).strftime("%Y-%m-%d") != day:
            raise ValueError("Date de checkpoint invalide.")
        return tuple(safe_path(self.root, self.directory/(day+suffix)) for suffix in (".json", ".joblib"))

    def load(self, day):
        seal, path = self._paths(day)
        if not seal.exists():
            # A crash before publishing the seal leaves an uncommitted file.
            return None
        record = json.loads(seal.read_text(encoding="utf-8"))
        if (record.get("fit_day") != day or record.get("suite_manifest_sha256") != self.identity
                or not path.is_file() or record.get("model_sha256") != digest(path)):
            raise ValueError(f"Checkpoint invalide: {day}; aucun chargement non verifie.")
        state = joblib.load(path)
        if state.get("fit_day") != day:
            raise ValueError("Etat charge avec une date de fit divergente.")
        self.reused += 1
        return state

    def save(self, day, state):
        seal, path = self._paths(day)
        if state.get("fit_day") != day or seal.exists():
            raise ValueError("Checkpoint deja scelle ou date divergente.")
        temporary = safe_path(self.root, path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp"))
        try:
            joblib.dump(state, temporary, compress=3)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        _json(seal, {"fit_day": day, "suite_manifest_sha256": self.identity, "model_sha256": digest(path)})
        self.saved += 1


def validate_predictions(panel: pd.DataFrame, predictions: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(panel.reset_index(drop=True), predictions[panel.columns].reset_index(drop=True), check_exact=True)
    valid = np.isfinite(panel.forecast.to_numpy(float))
    for name in MODELS:
        values = predictions[name].to_numpy(float)
        if len(values) != len(panel) or not np.array_equal(np.isfinite(values), valid):
            raise ValueError(f"Support de prediction divergent: {name}")
        if np.isinf(values).any():
            raise ValueError("Prediction infinie.")
    # These are mean-targeted point estimates, NOT quantile medians.
    # Do not clip them inside baseline P10/P90 or manufacture new intervals.
    if any(n in predictions for n in ("candidate_q10", "candidate_q90", "candidate_p50")):
        raise ValueError("Un correcteur MSE ne fabrique pas une distribution calibree.")


def verify_result(directory: Path, manifest: dict) -> dict:
    result = json.loads((directory/"results_manifest.json").read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("suite_manifest_sha256") != digest(directory/"manifest.json"):
        raise ValueError("Resultats non lies a ce snapshot.")
    _verify_files(directory, result.get("result_files"), RESULTS)
    return result


def evaluate(directory: Path, *, root: Path) -> Path:
    from .policy import run_replay
    directory, config, manifest = read_snapshot(directory, root=root)
    with exclusive_process_lock(safe_path(root, directory/"run.lock")):
        if (directory/"results_manifest.json").exists():
            verify_result(directory, manifest)
            LOGGER.info("[NYX RMSE] Backtest valide deja termine: aucun fit.")
            return directory
        before = source_runner.protected_state(root)
        identity = digest(directory/"manifest.json")
        cache = FitCache(directory, identity, root=root)
        try:
            def progress(message):
                LOGGER.info("[NYX RMSE] %s", message)
                _json(directory/"status.json", {"status": "running", "stage": "rolling_replay", "progress": message,
                      "updated_at_utc": datetime.now(timezone.utc).isoformat(), "fits_saved": cache.saved, "fits_reused": cache.reused})
            progress("Chargement des entrees figees; aucun run Chronos ni appel API.")
            panel = pd.read_parquet(directory/"panel.parquet")
            result = run_replay(panel, pd.read_parquet(directory/"source_predictions.parquet"),
                                pd.read_parquet(directory/"source_folds.parquet"), manifest["settings"], config["options"],
                                load_fit=cache.load, save_fit=cache.save, progress=progress)
            validate_predictions(panel, result["predictions"])
            if not cache.saved and not cache.reused:
                raise ValueError("Aucun fold entraine: historique insuffisant. Aucun resultat candidat publie.")
            for name in ("predictions", "folds", "governance"):
                frame = result[name] if name == "predictions" else _audit_parquet_frame(result[name])
                _parquet(directory/(name+".parquet"), frame)
            _json(directory/"model_audit.json", {**result["audit"], "fits_saved_this_run": cache.saved,
                  "fits_reused_this_run": cache.reused, "diagnostic_only": True, "activation_performed": False,
                  "production_modified": False, "annual_non_regression_guaranteed": False,
                  "evaluation_year_already_examined": True})
            read_snapshot(directory, root=root)
            if before != source_runner.protected_state(root) or identity != digest(directory/"manifest.json"):
                raise ValueError("Modification concurrente: resultats non publies.")
            _json(directory/"results_manifest.json", {"status": "completed", "suite_manifest_sha256": identity,
                  "result_files": {n: digest(directory/n) for n in sorted(RESULTS)}})
            _json(directory/"status.json", {"status": "evaluated", "snapshot": str(directory)})
            return directory
        except BaseException as exc:
            _json(directory/"status.json", {"status": "failed", "error": str(exc),
                  "fits_saved": cache.saved, "fits_reused": cache.reused})
            raise


def report(directory: Path, *, root: Path) -> Path:
    from .report import build_report
    directory, config, manifest = read_snapshot(directory, root=root)
    verify_result(directory, manifest)
    with exclusive_process_lock(safe_path(root, directory/"report.lock")):
        before = source_runner.protected_state(root)
        names = [p.relative_to(root).as_posix() for package in ("kpi_report", "economic_value") for p in sorted((root/package).glob("*.py"))]
        names += ["nyx_rmse/report.py", "config/economic_value.yaml"]
        code = {n: digest(root/n) for n in names}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
        destination = safe_path(root, directory/"reports"/stamp)
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
        pointer = {"snapshot": str(directory), "report_directory": str(destination),
                   "manifest_sha256": digest(directory/"manifest.json"), "report_manifest_sha256": digest(destination/"report_manifest.json")}
        _json(directory/"latest_report.json", pointer)
        _json(safe_path(root, config["output_root"])/"latest.json", pointer)
        _json(directory/"status.json", {"status": "completed", **pointer})
        return destination


def resolve_snapshot(config: dict, *, root: Path, value: str | None = None, create=False) -> Path:
    if value:
        directory = safe_path(root, value)
        _, frozen, _ = read_snapshot(directory, root=root)
        if frozen != config:
            raise ValueError("Configuration modifiee: --run-directory exige la recette figee de ce snapshot.")
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
        raise ValueError("Configuration modifiee: lancer -Action Prepare pour une nouvelle experience.")
    return directory


def status(directory: Path, *, root: Path) -> dict:
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
                or digest(target/"report_manifest.json") != pointer["report_manifest_sha256"]):
            raise ValueError("SHA du rapport divergent.")
        report_manifest = json.loads((target/"report_manifest.json").read_text(encoding="utf-8"))
        if (report_manifest.get("status") != "completed"
                or report_manifest.get("suite_manifest_sha256") != digest(directory/"manifest.json")
                or report_manifest.get("results_manifest_sha256") != digest(directory/"results_manifest.json")):
            raise ValueError("Rapport non lie aux resultats du snapshot courant.")
        _verify_files(target, report_manifest["files"], set(report_manifest["files"]))
        result.update(report_directory=str(target), report_verified=True)
    return {**result, "snapshot": str(directory), "models": list(MODELS), "production_modified": False}
