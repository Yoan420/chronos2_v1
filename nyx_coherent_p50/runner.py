"""Sealed, resumable CoherentP50 experiments; no operational activation path."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import uuid

import joblib
import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock
from nyx_scarcity import runner as base, variants_runner as old
from nyx_fundamental_stress import runner as fundamental

LOGGER = logging.getLogger(__name__)
VARIANTS = ("forest", "empirical")
NAMESPACE = base.NAMESPACE / "coherent_p50"
INPUTS = {"config.json", "base_config.json", "source_audit.json", "panel.parquet",
          "source_predictions.parquet", "source_folds.parquet", "hgb_predictions.parquet",
          "regional_25_predictions.parquet", "old_fundamental_25_predictions.parquet"}
RESULTS = {"predictions.parquet", "governed_predictions.parquet", "folds.parquet", "governance.parquet",
           "model_audit.json", "latest_model.joblib"}


def run_coherent_policy(*args, **kwargs):
    # Import only when fitting; Status/Report never initialise or fit a model.
    from .policy import run_coherent_policy as execute
    return execute(*args, **kwargs)


def safe_path(root: Path, value: str | Path) -> Path:
    path = base.safe_output(root, value)
    namespace = root.resolve()/NAMESPACE
    if namespace.resolve() != namespace or not path.is_relative_to(namespace):
        raise ValueError("CoherentP50 writes must remain inside its private experiment namespace.")
    return path


def validate_config(config: dict) -> None:
    expected = {"schema_version", "source_suite", "output_root", "variants", "primary_variant",
                "max_parallel", "diagnostic_only", "production_modified", "activation_performed"}
    if (not isinstance(config, dict) or set(config) != expected
            or type(config["schema_version"]) is not int or config["schema_version"] != 1):
        raise ValueError("Complete CoherentP50 schema 1 required; unknown fields rejected.")
    if config["variants"] != list(VARIANTS) or config["primary_variant"] != "forest":
        raise ValueError("Predeclared forest/empirical variants and DIRECT forest primary are required.")
    if type(config["max_parallel"]) is not int or config["max_parallel"] not in (1, 2):
        raise ValueError("One or two workers only.")
    if any(not isinstance(config[k], str) or not config[k].strip() for k in ("source_suite", "output_root")):
        raise ValueError("Explicit source/output paths required.")
    if config["diagnostic_only"] is not True or config["production_modified"] is not False or config["activation_performed"] is not False:
        raise ValueError("Operational modification, activation and promotion are forbidden.")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def protected_state(root: Path) -> dict:
    return fundamental.protected_state(root)


def code_seals(root: Path) -> dict:
    names = [p.relative_to(root).as_posix() for p in sorted((root/"nyx_coherent_p50").glob("*.py"))]
    names += ["run_nyx_coherent_p50.py", "CoherentP50.ps1"]
    return {**fundamental.code_seals(root), **{n: base.digest(root/n) for n in names}}


def runtime_identity(root: Path) -> dict:
    return fundamental.runtime_identity(root)


def _source(root: Path, value: str | Path):
    directory, config, manifest = fundamental.read_suite(root/Path(value), root=root)
    comparison = json.loads((directory/"comparison_manifest.json").read_text(encoding="utf-8"))
    if (comparison.get("status") != "completed"
            or comparison.get("suite_manifest_sha256") != base.digest(directory/"manifest.json")
            or comparison.get("comparison_sha256") != base.digest(directory/"comparison.json")):
        raise ValueError("A completed, sealed fundamental comparison is required.")
    for name in config["variants"]:
        fundamental.verify_result(directory, name, manifest)
    if comparison.get("variant_manifests") != {n: base.digest(directory/n/"results_manifest.json") for n in config["variants"]}:
        raise ValueError("Fundamental comparison is not bound to all verified results.")
    identity = {"manifest": base.digest(directory/"manifest.json"),
                "comparison_manifest": base.digest(directory/"comparison_manifest.json"),
                "results": comparison["variant_manifests"]}
    return directory, config, manifest, identity


def prepare(config: dict, *, root: Path) -> Path:
    validate_config(config)
    output = safe_path(root, config["output_root"])
    before = protected_state(root)
    with exclusive_process_lock(output/"prepare.lock"):
        source, _, source_manifest, source_identity = _source(root, config["source_suite"])
        settings = source_manifest["settings"]
        if (type(settings.get("threads")) is not int or not 1 <= settings["threads"] <= 2
                or settings["threads"]*config["max_parallel"] > 4):
            raise ValueError("At most four total CPU threads; one or two per worker.")
        baseline = json.loads((source/"base_config.json").read_text(encoding="utf-8"))
        audit = json.loads((source/"source_audit.json").read_text(encoding="utf-8"))
        if not {"source_data_audit", "source_config", "source_suite_path"}.issubset(audit):
            raise ValueError("Original Storm/data provenance is required in the source audit.")
        audit.update(coherent_source_suite=str(source), coherent_source_identity=source_identity,
            source_settings=settings, primary_variant="forest", decision_policy="direct",
            classifier_refitted=False, frozen_detector_reused=True, conditional_cdfs_refitted=True,
            diagnostic_only=True, production_modified=False, activation_performed=False)
        paths = {"panel": source/"panel.parquet", "source_predictions": source/"fundamental/predictions.parquet",
            "source_folds": source/"fundamental/folds.parquet", "hgb_predictions": source/"hgb_predictions.parquet",
            "regional_25_predictions": source/"regional_25_predictions.parquet",
            "old_fundamental_25_predictions": source/"fundamental/proposals_25.parquet"}
        frames = {name: pd.read_parquet(path) for name, path in paths.items()}
        # Verify again after the read; do not publish a mixed-version snapshot.
        if _source(root, source)[3] != source_identity or before != protected_state(root):
            raise ValueError("Protected or source files changed during preparation.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
        directory = safe_path(root, output/"snapshots"/stamp)
        directory.mkdir(parents=True, exist_ok=False)
        for name, value in (("config", config), ("base_config", baseline), ("source_audit", audit)):
            base._json(directory/f"{name}.json", value)
        for name, frame in frames.items():
            base._parquet(directory/f"{name}.parquet", frame)
        manifest = {"schema_version": 1, "config": config, "settings": settings,
            "input_files": {n: base.digest(directory/n) for n in sorted(INPUTS)},
            "code_sha256": code_seals(root), "runtime": runtime_identity(root), "protected_files": before,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True,
            "production_modified": False, "activation_performed": False}
        base._json(directory/"manifest.json", manifest)
        base._json(directory/"status.json", {"status": "prepared", "snapshot": str(directory)})
        base._json(output/"latest_prepared.json", {"snapshot": str(directory)})
        LOGGER.info("[CoherentP50] Prepared %s; %d unchanged rows; detector frozen.", directory, len(frames["panel"]))
        return directory


def read_suite(directory: Path, *, root: Path):
    directory = safe_path(root, directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    base._verify(directory, manifest.get("input_files"), INPUTS)
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    validate_config(config)
    if config != manifest.get("config") or not directory.is_relative_to(safe_path(root, config["output_root"])/"snapshots"):
        raise ValueError("CoherentP50 manifest/config identity mismatch.")
    audit = json.loads((directory/"source_audit.json").read_text(encoding="utf-8"))
    if manifest.get("settings") != audit.get("source_settings"):
        raise ValueError("Frozen model settings have changed.")
    return directory, config, manifest


def _verify_code(manifest: dict, root: Path) -> None:
    if manifest["code_sha256"] != code_seals(root) or manifest["runtime"] != runtime_identity(root):
        raise ValueError("Code/runtime changed since Prepare. Prepare a NEW snapshot; no stale partial fit is reused.")


def verify_result(directory: Path, name: str, manifest: dict) -> dict:
    if name not in manifest["config"]["variants"]:
        raise ValueError("Unregistered CoherentP50 model.")
    result = json.loads((directory/name/"results_manifest.json").read_text(encoding="utf-8"))
    if (result.get("status") != "completed" or result.get("variant") != name
            or result.get("suite_manifest_sha256") != base.digest(directory/"manifest.json")):
        raise ValueError("CoherentP50 results not bound to this frozen suite.")
    base._verify(directory/name, result.get("result_files"), RESULTS)
    return result


def load_model(directory: Path, name: str, *, root: Path) -> dict:
    """Only this lab's trusted, hash-verified local model can be deserialised."""
    directory, _, manifest = read_suite(directory, root=root)
    verify_result(directory, name, manifest)
    _verify_code(manifest, root)
    return joblib.load(directory/name/"latest_model.joblib")


def run_worker(directory: Path, name: str, *, root: Path) -> dict:
    directory, config, manifest = read_suite(directory, root=root)
    if name not in config["variants"]:
        raise ValueError("Unregistered worker.")
    destination = safe_path(root, directory/name)
    with exclusive_process_lock(destination/"worker.lock"):
        if (destination/"results_manifest.json").is_file():
            verify_result(directory, name, manifest)
            return {"status": "reused", "variant": name}
        _verify_code(manifest, root)
        before, identity = protected_state(root), base.digest(directory/"manifest.json")
        latest = {}
        def remember(state, parameters):
            latest.update(state=state, parameters=parameters)
        base._json(destination/"status.json", {"status": "running", "variant": name})
        try:
            panel = pd.read_parquet(directory/"panel.parquet")
            results = run_coherent_policy(panel, pd.read_parquet(directory/"source_predictions.parquet"),
                pd.read_parquet(directory/"source_folds.parquet"), manifest["settings"], name, on_last_fit=remember)
            if not isinstance(results, dict) or set(results) != {"direct", "governed"}:
                raise ValueError("Coherent policy must return exactly direct and governed results.")
            direct, governed = results["direct"], results["governed"]
            baseline = json.loads((directory/"base_config.json").read_text(encoding="utf-8"))
            for result in (direct, governed):
                base.validate_predictions(panel, result.predictions, baseline)
            if not latest:
                raise ValueError("No fitted CDF state: a viable experiment requires a trained fold.")
            for key, frame in (("predictions", direct.predictions), ("governed_predictions", governed.predictions),
                               ("folds", direct.folds), ("governance", governed.governance)):
                base._parquet(destination/f"{key}.parquet", old._audit_parquet_frame(frame) if key in ("folds", "governance") else frame)
            base._json(destination/"model_audit.json", {"direct": direct.audit, "governed": governed.audit})
            temporary = destination/(".model_"+uuid.uuid4().hex+".joblib")
            try:
                joblib.dump({**latest, "variant": name, "suite_manifest_sha256": identity,
                    "diagnostic_only": True, "activation_performed": False}, temporary, compress=3)
                temporary.replace(destination/"latest_model.joblib")
            finally:
                temporary.unlink(missing_ok=True)
            _verify_code(manifest, root)
            read_suite(directory, root=root)
            if before != protected_state(root) or identity != base.digest(directory/"manifest.json"):
                raise ValueError("Concurrent protected/source change; results will not be published.")
            result = {"status": "completed", "variant": name, "suite_manifest_sha256": identity,
                "result_files": {n: base.digest(destination/n) for n in sorted(RESULTS)},
                "diagnostic_only": True, "production_modified": False, "activation_performed": False}
            base._json(destination/"results_manifest.json", result)
            base._json(destination/"status.json", {"status": "completed", "variant": name})
            return result
        except BaseException as exc:
            base._json(destination/"status.json", {"status": "failed", "error": str(exc)})
            raise


def collect(directory: Path, config: dict, manifest: dict):
    predictions = {name: pd.read_parquet(directory/path) for name, path in {
        "hgb_v1": "hgb_predictions.parquet", "regional_25": "regional_25_predictions.parquet",
        "fundamental_old": "source_predictions.parquet", "fundamental_old_25": "old_fundamental_25_predictions.parquet"}.items()}
    audits = {}
    for name in config["variants"]:
        verify_result(directory, name, manifest)
        predictions[name] = pd.read_parquet(directory/name/"predictions.parquet")
        predictions[name+"_governed"] = pd.read_parquet(directory/name/"governed_predictions.parquet")
        audits[name] = json.loads((directory/name/"model_audit.json").read_text(encoding="utf-8"))
    return predictions, audits


def run_suite(directory: Path, *, root: Path) -> Path:
    directory, config, manifest = read_suite(directory, root=root)
    with exclusive_process_lock(directory/"suite.lock"):
        if (directory/"comparison_manifest.json").is_file():
            LOGGER.info("[CoherentP50] Completed predictions reused without fitting.")
            return report(directory, root=root)
        _verify_code(manifest, root)
        pending, completed, active = [], [], {}
        for name in config["variants"]:
            if (directory/name/"results_manifest.json").is_file():
                verify_result(directory, name, manifest)
                completed.append(name)
            else:
                pending.append(name)
        try:
            while pending or active:
                while pending and len(active) < config["max_parallel"]:
                    name = pending.pop(0)
                    destination = safe_path(root, directory/name)
                    destination.mkdir(parents=True, exist_ok=True)
                    log = (destination/"worker.log").open("a", encoding="utf-8")
                    try:
                        proc = subprocess.Popen([sys.executable, str(root/"run_nyx_coherent_p50.py"), "--action", "worker",
                            "--run-directory", str(directory), "--variant", name], cwd=root, shell=False,
                            stdout=log, stderr=subprocess.STDOUT)
                    except BaseException:
                        log.close()
                        raise
                    active[name] = (proc, log, time.monotonic())
                    LOGGER.info("[CoherentP50] Started %s; log=%s", name, destination/"worker.log")
                base._json(directory/"status.json", {"status": "running", "snapshot": str(directory),
                    "completed": completed, "active": list(active), "pending": pending})
                for name, (proc, log, started) in list(active.items()):
                    rc = proc.poll()
                    if rc is None and time.monotonic()-started > 7200:
                        raise TimeoutError(f"{name}: two-hour worker limit exceeded.")
                    if rc is not None:
                        log.close()
                        del active[name]
                        if rc:
                            raise ValueError(f"{name} failed ({rc}); inspect {directory/name/'worker.log'}. Completed workers remain reusable.")
                        verify_result(directory, name, manifest)
                        completed.append(name)
                        LOGGER.info("[CoherentP50] Completed %s (%d/2)", name, len(completed))
                if pending or active:
                    time.sleep(.5)
            from nyx_scarcity.variant_reporting import build_comparison
            base._json(directory/"status.json", {"status": "running", "stage": "comparison", "snapshot": str(directory),
                "completed": completed, "active": [], "pending": []})
            predictions, _ = collect(directory, config, manifest)
            comparison = build_comparison(predictions)
            comparison.update(primary_variant="forest", primary_decision_policy="direct",
                primary_selected_before_new_backtest=True, strict_governor_enforced_in_primary=False,
                annual_non_regression_guaranteed=False, prospective_validation_completed=False,
                exploratory_year_already_examined=True, classifier_refitted=False, frozen_detector_reused=True,
                price_features_used=False, final_distribution_probabilities_are_not_the_detector_probabilities=True)
            comparison["model_labels"].update({"regional_25": "Expert régional précédent — 25 %",
                "fundamental_old": "Expert fondamental précédent — gouverné", "fundamental_old_25": "Expert fondamental précédent — 25 %",
                "forest": "P50 cohérent — forêt DIRECTE (principal)", "forest_governed": "P50 cohérent — forêt gouvernée",
                "empirical": "P50 cohérent — CDF empirique DIRECTE", "empirical_governed": "P50 cohérent — CDF empirique gouvernée"})
            base._json(directory/"comparison.json", comparison)
            base._json(directory/"comparison_manifest.json", {"status": "completed",
                "suite_manifest_sha256": base.digest(directory/"manifest.json"),
                "comparison_sha256": base.digest(directory/"comparison.json"),
                "variant_manifests": {n: base.digest(directory/n/"results_manifest.json") for n in config["variants"]}})
            return report(directory, root=root)
        except BaseException as exc:
            for proc, log, _ in active.values():
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                log.close()
            base._json(directory/"status.json", {"status": "failed", "snapshot": str(directory), "error": str(exc), "completed": completed})
            raise


def _report(directory: Path, *, root: Path) -> Path:
    from nyx_scarcity.variant_reporting import render_comparison
    from .reporting import render_p50_reports
    directory, config, manifest = read_suite(directory, root=root)
    result = json.loads((directory/"comparison_manifest.json").read_text(encoding="utf-8"))
    if (result.get("status") != "completed" or result.get("suite_manifest_sha256") != base.digest(directory/"manifest.json")
            or result.get("comparison_sha256") != base.digest(directory/"comparison.json")
            or result.get("variant_manifests") != {n: base.digest(directory/n/"results_manifest.json") for n in config["variants"]}):
        raise ValueError("Completed CoherentP50 comparison checksum mismatch.")
    before = protected_state(root)
    predictions, audits = collect(directory, config, manifest)
    comparison = json.loads((directory/"comparison.json").read_text(encoding="utf-8"))
    source = json.loads((directory/"source_audit.json").read_text(encoding="utf-8"))
    with exclusive_process_lock(directory/"report.lock"):
        base._json(directory/"status.json", {"status": "running", "stage": "reporting", "snapshot": str(directory),
            "completed": list(config["variants"]), "active": [], "pending": []})
        comparison_path = safe_path(root, directory/"coherent_p50_comparison.html")
        render_comparison(predictions, comparison, {**manifest, "models": audits, "data": source["source_data_audit"],
            "diagnostic_only": True, "production_modified": False}, comparison_path)
        document = comparison_path.read_text(encoding="utf-8")
        document = document.replace("NYX — comparaison contrôlée des experts XGB", "NYX — comparaison des distributions P50 cohérentes")
        document = document.replace("NYX — variantes XGB et seuils causaux", "NYX — P50 cohérent et probabilités figées")
        document = document.replace("P.summary.variants.includes('xgb_unweighted_fixed')?'xgb_unweighted_fixed':'hgb_v1'",
                                    "P.summary.primary_variant||'forest'")
        document = document.replace("<main>", '<main><section><h2>P50 cohérent — laboratoire isolé</h2>'
            '<p>Candidat principal prédéclaré : forêt DIRECTE. La variante gouvernée est évaluée séparément. '
            'Les probabilités du détecteur sont figées ; les distributions conditionnelles sont apprises sur le passé. '
            'Les probabilités de la distribution finale ne sont pas celles du détecteur. Aucun prix électrique, '
            'forecast NYX ou Storm comme variable explicative des CDF. Année déjà examinée : expérience '
            'exploratoire, sans garantie de non-régression ni promotion automatique.</p>'
            '<p><a href="reports/index.html">Rapports forêt DIRECTE</a> · '
            '<a href="reports_governed/index.html">Rapports forêt gouvernée</a></p></section>', 1)
        comparison_path.write_text(document, encoding="utf-8")
        paths = {}
        for key, folder, decision in (("forest", "reports", "direct"), ("forest_governed", "reports_governed", "governed")):
            rendered = render_p50_reports(predictions[key], source_audit={**source, "model_audit": audits["forest"],
                "decision_policy": decision, "strict_governor_enforced": decision == "governed"},
                output_directory=safe_path(root, directory/folder), comparison=comparison, decision_policy=decision,
                model_name="coherent_p50" if decision == "direct" else "coherent_p50_governed")
            for name, path in rendered.items():
                path = safe_path(root, path)
                if not path.is_relative_to(directory/folder) or not path.is_file():
                    raise ValueError("Reporter returned an artifact outside its assigned snapshot directory.")
                paths[folder+"_"+name] = path
        if before != protected_state(root):
            raise ValueError("Protected operational files changed concurrently; publication is not certified.")
        status = {"status": "completed", "snapshot": str(directory), "report": str(paths["reports_index"]),
            "comparison_report": str(comparison_path), "production_modified": False, "activation_performed": False,
            "diagnostic_only": True, "reports_sha256": {str(p.relative_to(directory)): base.digest(p) for p in [*paths.values(), comparison_path]}}
        base._json(directory/"status.json", status)
        base._json(safe_path(root, config["output_root"])/"latest.json", status)
        LOGGER.info("[CoherentP50] Reports complete: %s", paths["reports_index"])
        return paths["reports_index"]


def report(directory: Path, *, root: Path) -> Path:
    directory, _, _ = read_suite(directory, root=root)
    try:
        return _report(directory, root=root)
    except BaseException as exc:
        base._json(directory/"status.json", {"status": "failed", "stage": "reporting", "snapshot": str(directory),
            "error": str(exc), "predictions_retained": True, "production_modified": False})
        raise


def resolve_latest(config: dict, *, root: Path, completed: bool) -> Path:
    validate_config(config)
    pointer = safe_path(root, config["output_root"])/("latest.json" if completed else "latest_prepared.json")
    if not pointer.is_file():
        raise ValueError("No CoherentP50 experiment yet; use Run or Prepare first.")
    return safe_path(root, json.loads(pointer.read_text(encoding="utf-8"))["snapshot"])
