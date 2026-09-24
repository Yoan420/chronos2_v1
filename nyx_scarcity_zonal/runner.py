"""Frozen, resumable zonal experiments. No operational activation path exists."""
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
from nyx_scarcity.variant_runtime import runtime_seals
from .policy import VARIANTS, fixed_conservative_forecast, run_zonal_policy

LOGGER = logging.getLogger(__name__)
NAMESPACE = base.NAMESPACE / "zonal"
INPUTS = {"config.json", "base_config.json", "source_audit.json", "panel.parquet",
          "hgb_predictions.parquet", "regional_predictions.parquet", "regional_25_predictions.parquet", "regional_audit.json"}
RESULTS = {"predictions.parquet", "strict_predictions.parquet", "folds.parquet", "governance.parquet",
           "model_audit.json", "latest_model.joblib"}


def safe_path(root: Path, value: str | Path) -> Path:
    path = base.safe_output(root, value)
    namespace = root.resolve() / NAMESPACE
    if namespace.resolve() != namespace or not path.is_relative_to(namespace):
        raise ValueError("All zonal writes must remain in the private zonal experiment namespace.")
    return path


def validate_config(config: dict) -> None:
    expected = {"schema_version", "source_suite", "output_root", "variants", "primary_variant",
                "max_parallel", "offset_penalty", "offset_bound", "fixed_alpha",
                "diagnostic_only", "production_modified", "activation_performed"}
    if (not isinstance(config, dict) or set(config) != expected
            or type(config["schema_version"]) is not int or config["schema_version"] != 1):
        raise ValueError("Complete zonal schema 1 required; unknown settings are rejected.")
    if (config["variants"] != list(VARIANTS) or config["primary_variant"] != "zonal_hiercal"
            or config["fixed_alpha"] != .25):
        raise ValueError("Use the three predeclared ablations and primary zonal_hiercal at fixed 25%.")
    for key, value in (("offset_penalty", 1.), ("offset_bound", 3.)):
        if type(config[key]) not in (int, float) or config[key] != value:
            raise ValueError(f"{key} is predeclared at {value}; do not optimize on the evaluation year.")
    if type(config["max_parallel"]) is not int or config["max_parallel"] not in (1, 2):
        raise ValueError("One or two workers only.")
    if any(not isinstance(config[k], str) or not config[k].strip() for k in ("source_suite", "output_root")):
        raise ValueError("Explicit source/output paths required.")
    if config["diagnostic_only"] is not True or config["production_modified"] is not False or config["activation_performed"] is not False:
        raise ValueError("Production modification and automatic promotion are forbidden.")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def protected_state(root: Path) -> dict:
    names = ("chronos2_modular/report.py", "chronos2_hourly/reporting.py",
             "chronos2_hourly/nuclear_report_benchmark.py", "chronos2_hourly/nuclear_reporting.py")
    return {**base.protected_state(root), **{n: base.digest(root/n) for n in names}}


def code_seals(root: Path) -> dict:
    names = [p.relative_to(root).as_posix() for p in sorted((root/"nyx_scarcity_zonal").glob("*.py"))]
    names += ["run_nyx_scarcity_zonal.py", "ScarcityZonal.ps1", "run_nyx_scarcity_adjustments.py"]
    return {**old.code_seals(), **protected_state(root), **{n: base.digest(root/n) for n in names}}


def prepare(config: dict, *, root: Path) -> Path:
    from run_nyx_scarcity_adjustments import read_source
    from nyx_scarcity.policy import PolicyResult
    validate_config(config)
    output = safe_path(root, config["output_root"])
    source = old.safe_path(root, config["source_suite"])
    before = protected_state(root)
    with exclusive_process_lock(output/"prepare.lock"):
        predictions, source_audit = read_source(source, root=root)
        _, _, original = old.read_suite(source, root=root)
        panel = pd.read_parquet(source/"panel.parquet")
        baseline = json.loads((source/"base_config.json").read_text(encoding="utf-8"))
        data_audit = json.loads((source/"data_audit.json").read_text(encoding="utf-8"))
        settings = original["settings"]
        if settings["threads"] > 2 or settings["threads"]*config["max_parallel"] > 4:
            raise ValueError("At most four total CPU threads; two per worker.")
        regional_audit = json.loads((source/"xgb_unweighted_fixed/model_audit.json").read_text(encoding="utf-8"))
        regional_25 = fixed_conservative_forecast(PolicyResult(predictions["xgb_unweighted_fixed"],
            pd.DataFrame(), pd.DataFrame(), regional_audit)).predictions
        audit = {"source_suite_path": str(source), "source_config": baseline, "source_data_audit": data_audit,
                 "source_comparison": source_audit, "new_training_performed": True,
                 "diagnostic_only": True, "production_modified": False, "activation_performed": False}
        _, checked = read_source(source, root=root)
        if checked != source_audit or before != protected_state(root):
            raise ValueError("Sources or protected files changed during preparation.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
        directory = safe_path(root, output/"snapshots"/stamp)
        directory.mkdir(parents=True, exist_ok=False)
        for name, value in (("config", config), ("base_config", baseline), ("source_audit", audit),
                            ("regional_audit", regional_audit)):
            base._json(directory/f"{name}.json", value)
        for name, value in (("panel", panel), ("hgb_predictions", predictions["hgb_v1"]),
                            ("regional_predictions", predictions["xgb_unweighted_fixed"]),
                            ("regional_25_predictions", regional_25)):
            base._parquet(directory/f"{name}.parquet", value)
        manifest = {"schema_version": 1, "config": config, "settings": settings,
                    "input_files": {n: base.digest(directory/n) for n in sorted(INPUTS)},
                    "code_sha256": code_seals(root), "runtime": runtime_seals(root),
                    "protected_files": before, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "diagnostic_only": True, "activation_performed": False}
        base._json(directory/"manifest.json", manifest)
        base._json(directory/"status.json", {"status": "prepared", "snapshot": str(directory)})
        base._json(output/"latest_prepared.json", {"snapshot": str(directory)})
        LOGGER.info("[Zonal] Prepared %s; %d unchanged rows.", directory, len(panel))
        return directory


def read_suite(directory: Path, *, root: Path):
    directory = safe_path(root, directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    base._verify(directory, manifest.get("input_files"), INPUTS)
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    validate_config(config)
    if config != manifest.get("config") or not directory.is_relative_to(safe_path(root, config["output_root"])/"snapshots"):
        raise ValueError("Zonal manifest/config identity mismatch.")
    audit = json.loads((directory/"source_audit.json").read_text(encoding="utf-8"))
    if manifest.get("settings") != audit["source_comparison"]["source_settings"]:
        raise ValueError("Frozen model settings have changed.")
    return directory, config, manifest


def _verify_code(manifest: dict, root: Path) -> None:
    if manifest["code_sha256"] != code_seals(root) or manifest["runtime"] != runtime_seals(root):
        raise ValueError("Code/runtime changed since Prepare. Prepare a NEW snapshot; no stale partial fit is reused.")


def verify_result(directory: Path, name: str, manifest: dict) -> dict:
    if name not in manifest["config"]["variants"]:
        raise ValueError("Unregistered zonal model.")
    result = json.loads((directory/name/"results_manifest.json").read_text(encoding="utf-8"))
    if (result.get("status") != "completed" or result.get("variant") != name
            or result.get("suite_manifest_sha256") != base.digest(directory/"manifest.json")):
        raise ValueError("Zonal results not bound to this frozen suite.")
    base._verify(directory/name, result.get("result_files"), RESULTS)
    return result


def load_model(directory: Path, name: str, *, root: Path) -> dict:
    """Only load this lab's trusted, hash-verified local pickle; never a user upload."""
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
        before = protected_state(root)
        identity = base.digest(directory/"manifest.json")
        latest = {}
        def remember(state, parameters):
            latest.update(state=state, parameters=parameters)
        base._json(destination/"status.json", {"status": "running", "variant": name})
        try:
            panel = pd.read_parquet(directory/"panel.parquet")
            strict = run_zonal_policy(panel, manifest["settings"], name,
                offset_penalty=config["offset_penalty"], offset_bound=config["offset_bound"], on_last_fit=remember)
            fixed = fixed_conservative_forecast(strict, alpha=config["fixed_alpha"])
            baseline = json.loads((directory/"base_config.json").read_text(encoding="utf-8"))
            for result in (strict, fixed):
                base.validate_predictions(panel, result.predictions, baseline)
            if not latest:
                raise ValueError("No trained fold: a viable experiment requires a fitted model.")
            for key, frame in (("predictions", fixed.predictions), ("strict_predictions", strict.predictions),
                               ("folds", strict.folds), ("governance", strict.governance)):
                base._parquet(destination/f"{key}.parquet", old._audit_parquet_frame(frame) if key in ("folds", "governance") else frame)
            base._json(destination/"model_audit.json", {"fixed": fixed.audit, "strict": strict.audit})
            tmp = destination/(".model_"+uuid.uuid4().hex+".joblib")
            try:
                joblib.dump({**latest, "variant": name, "suite_manifest_sha256": identity,
                             "diagnostic_only": True, "activation_performed": False}, tmp, compress=3)
                tmp.replace(destination/"latest_model.joblib")
            finally:
                tmp.unlink(missing_ok=True)
            _verify_code(manifest, root)
            read_suite(directory, root=root)
            if before != protected_state(root) or identity != base.digest(directory/"manifest.json"):
                raise ValueError("Concurrent protected/source change; results will not be published.")
            result = {"status": "completed", "variant": name, "suite_manifest_sha256": identity,
                      "result_files": {n: base.digest(destination/n) for n in sorted(RESULTS)},
                      "diagnostic_only": True, "activation_performed": False}
            base._json(destination/"results_manifest.json", result)
            base._json(destination/"status.json", {"status": "completed", "variant": name})
            return result
        except BaseException as exc:
            base._json(destination/"status.json", {"status": "failed", "error": str(exc)})
            raise


def collect(directory: Path, config: dict, manifest: dict):
    predictions = {"hgb_v1": pd.read_parquet(directory/"hgb_predictions.parquet")}
    predictions["regional_25"] = pd.read_parquet(directory/"regional_25_predictions.parquet")
    audits = {}
    for name in config["variants"]:
        verify_result(directory, name, manifest)
        predictions[name] = pd.read_parquet(directory/name/"predictions.parquet")
        predictions[name+"_strict"] = pd.read_parquet(directory/name/"strict_predictions.parquet")
        audits[name] = json.loads((directory/name/"model_audit.json").read_text(encoding="utf-8"))
    return predictions, audits


def run_suite(directory: Path, *, root: Path) -> Path:
    directory, config, manifest = read_suite(directory, root=root)
    with exclusive_process_lock(directory/"suite.lock"):
        if (directory/"comparison_manifest.json").is_file():
            LOGGER.info("[Zonal] Completed predictions reused: no fitting.")
            return report(directory, root=root)
        _verify_code(manifest, root)
        pending, completed, active = [], [], {}
        for name in config["variants"]:
            if (directory/name/"results_manifest.json").is_file():
                verify_result(directory, name, manifest); completed.append(name)
            else:
                pending.append(name)
        try:
            while pending or active:
                while pending and len(active) < config["max_parallel"]:
                    name = pending.pop(0); destination = safe_path(root, directory/name)
                    destination.mkdir(parents=True, exist_ok=True)
                    log = (destination/"worker.log").open("a", encoding="utf-8")
                    try:
                        proc = subprocess.Popen([sys.executable, str(root/"run_nyx_scarcity_zonal.py"), "--action", "worker",
                            "--run-directory", str(directory), "--variant", name], cwd=root, shell=False,
                            stdout=log, stderr=subprocess.STDOUT)
                    except BaseException:
                        log.close(); raise
                    active[name] = (proc, log, time.monotonic())
                    LOGGER.info("[Zonal] Started %s; log=%s", name, destination/"worker.log")
                base._json(directory/"status.json", {"status": "running", "snapshot": str(directory),
                    "completed": completed, "active": list(active), "pending": pending})
                for name, (proc, log, started) in list(active.items()):
                    rc = proc.poll()
                    if rc is None and time.monotonic()-started > 7200:
                        raise TimeoutError(f"{name}: two-hour worker limit exceeded.")
                    if rc is not None:
                        log.close(); del active[name]
                        if rc:
                            raise ValueError(f"{name} failed ({rc}); inspect {directory/name/'worker.log'}. Completed workers are reusable.")
                        verify_result(directory, name, manifest); completed.append(name)
                        LOGGER.info("[Zonal] Completed %s (%d/3)", name, len(completed))
                if pending or active:
                    time.sleep(.5)
            from nyx_scarcity.variant_reporting import build_comparison
            predictions, _ = collect(directory, config, manifest)
            comparison = build_comparison(predictions)
            comparison.update(primary_variant=config["primary_variant"], fixed_alpha=.25,
                primary_selected_before_new_backtest=True, strict_governor_enforced_in_primary=False,
                annual_non_regression_guaranteed=False, prospective_validation_completed=False)
            comparison["model_labels"].update({"regional_25": "Expert régional précédent — 25 %",
                "regional_hiercal": "Régional + calibration zonale — 25 %", "zonal_context": "Contexte zonal — 25 %",
                "zonal_hiercal": "NYX zonal + calibration — 25 % (candidat principal)",
                **{n+"_strict": n+" — gouverneur strict" for n in config["variants"]}})
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
                        proc.kill(); proc.wait(timeout=5)
                log.close()
            base._json(directory/"status.json", {"status": "failed", "snapshot": str(directory), "error": str(exc), "completed": completed})
            raise


def report(directory: Path, *, root: Path) -> Path:
    from nyx_scarcity.variant_reporting import render_comparison
    from .reporting import render_zonal_reports
    directory, config, manifest = read_suite(directory, root=root)
    result = json.loads((directory/"comparison_manifest.json").read_text(encoding="utf-8"))
    if (result.get("status") != "completed" or result.get("suite_manifest_sha256") != base.digest(directory/"manifest.json")
            or result.get("comparison_sha256") != base.digest(directory/"comparison.json")
            or result.get("variant_manifests") != {n: base.digest(directory/n/"results_manifest.json") for n in config["variants"]}):
        raise ValueError("Completed zonal comparison checksum mismatch.")
    before = protected_state(root)
    predictions, audits = collect(directory, config, manifest)
    comparison = json.loads((directory/"comparison.json").read_text(encoding="utf-8"))
    source = json.loads((directory/"source_audit.json").read_text(encoding="utf-8"))
    with exclusive_process_lock(directory/"report.lock"):
        render_comparison(predictions, comparison, {**manifest, "models": audits,
            "data": source["source_data_audit"], "diagnostic_only": True, "production_modified": False},
            directory/"zonal_comparison.html")
        comparison_path = directory/"zonal_comparison.html"
        document = comparison_path.read_text(encoding="utf-8")
        document = document.replace("P.summary.variants.includes('xgb_unweighted_fixed')?'xgb_unweighted_fixed':'hgb_v1'",
                                    "P.summary.primary_variant||'hgb_v1'")
        document = document.replace("<main>", '<main><section><h2>Expérience zonale — correction fixe 25 %</h2>'
            '<p>Le candidat principal est zonal_hiercal, déclaré avant ce backtest. Les autres variantes sont des ablations, '
            'pas des gagnants choisis pays par pays. Le gouverneur strict est affiché séparément : il n’est PAS appliqué '
            'aux propositions fixes à 25 %. Cette année déjà examinée n’est pas un test prospectif ; aucune activation.</p>'
            '<p><a href="reports/index.html">Rapports pays au format opérationnel</a></p></section>', 1)
        comparison_path.write_text(document, encoding="utf-8")
        paths = render_zonal_reports(predictions[config["primary_variant"]], source_audit={**source,
            "model_audit": audits[config["primary_variant"]], "decision_policy": "fixed_25_percent_experimental",
            "strict_governor_enforced": False}, output_directory=safe_path(root, directory/"reports"),
            model_name="nyx_zonal_25", comparison=comparison)
        if before != protected_state(root):
            raise ValueError("Protected operational files changed concurrently; publication is not certified.")
        status = {"status": "completed", "snapshot": str(directory), "report": str(paths["index"]),
                  "comparison_report": str(directory/"zonal_comparison.html"), "production_modified": False,
                  "activation_performed": False, "diagnostic_only": True,
                  "reports_sha256": {str(p.relative_to(directory)): base.digest(p) for p in paths.values()}}
        base._json(directory/"status.json", status)
        base._json(safe_path(root, config["output_root"])/"latest.json", status)
        LOGGER.info("[Zonal] Reports complete: %s", paths["index"])
        return paths["index"]


def resolve_latest(config: dict, *, root: Path, completed: bool) -> Path:
    pointer = safe_path(root, config["output_root"])/("latest.json" if completed else "latest_prepared.json")
    if not pointer.is_file():
        raise ValueError("No zonal experiment yet; use Run or Prepare first.")
    return safe_path(root, json.loads(pointer.read_text(encoding="utf-8"))["snapshot"])
