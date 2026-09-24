"""Frozen, resumable fundamental experiments. No operational activation path exists."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
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
from nyx_scarcity_zonal import runner as zonal
from nyx_scarcity_zonal.policy import fixed_conservative_forecast
from .policy import run_fundamental_policy
VARIANTS = ("fundamental", "calendar")

LOGGER = logging.getLogger(__name__)
NAMESPACE = base.NAMESPACE / "fundamental"
INPUTS = {"config.json", "base_config.json", "source_audit.json", "panel.parquet",
          "hgb_predictions.parquet", "regional_25_predictions.parquet"}
RESULTS = {"predictions.parquet", "proposals_25.parquet", "folds.parquet", "governance.parquet",
           "model_audit.json", "latest_model.joblib"}


def safe_path(root: Path, value: str | Path) -> Path:
    path = base.safe_output(root, value)
    namespace = root.resolve() / NAMESPACE
    if namespace.resolve() != namespace or not path.is_relative_to(namespace):
        raise ValueError("All fundamental writes must remain in the private fundamental experiment namespace.")
    return path


def validate_config(config: dict) -> None:
    expected = {"schema_version", "source_suite", "output_root", "variants", "primary_variant",
                "max_parallel", "fixed_alpha", "diagnostic_only", "production_modified", "activation_performed"}
    if not isinstance(config, dict) or set(config) != expected or type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Complete fundamental schema 1 required; unknown fields rejected.")
    if config["variants"] != list(VARIANTS) or config["primary_variant"] != "fundamental" or config["fixed_alpha"] != .25:
        raise ValueError("Predeclared fundamental/calendar models, governed primary, fixed25 diagnostic required.")
    if type(config["max_parallel"]) is not int or config["max_parallel"] not in (1, 2):
        raise ValueError("One or two workers only.")
    if any(not isinstance(config[k], str) or not config[k].strip() for k in ("source_suite", "output_root")):
        raise ValueError("Explicit source/output paths required.")
    if config["diagnostic_only"] is not True or config["production_modified"] is not False or config["activation_performed"] is not False:
        raise ValueError("Production modification and promotion are forbidden.")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def protected_state(root: Path) -> dict:
    names = ("chronos2_modular/report.py", "chronos2_hourly/reporting.py",
             "chronos2_hourly/nuclear_report_benchmark.py", "chronos2_hourly/nuclear_reporting.py")
    return {**base.protected_state(root), **{n: base.digest(root/n) for n in names}}


def code_seals(root: Path) -> dict:
    names = [p.relative_to(root).as_posix() for p in sorted((root/"nyx_fundamental_stress").glob("*.py"))]
    names += ["run_nyx_fundamental_stress.py", "FundamentalStress.ps1"]
    return {**zonal.code_seals(root), **{n: base.digest(root/n) for n in names}}


def runtime_identity(root: Path) -> dict:
    return {"xgboost": runtime_seals(root), "versions": {
        name: version(name) for name in ("numpy", "pandas", "scikit-learn", "pyarrow", "joblib")}}


def prepare(config: dict, *, root: Path) -> Path:
    validate_config(config)
    output = safe_path(root, config["output_root"])
    source, source_config, source_manifest = zonal.read_suite(root/config["source_suite"], root=root)
    source_identity = base.digest(source/"manifest.json")
    before = protected_state(root)
    with exclusive_process_lock(output/"prepare.lock"):
        comparison = json.loads((source/"comparison_manifest.json").read_text(encoding="utf-8"))
        if (comparison.get("status") != "completed"
                or comparison.get("suite_manifest_sha256") != base.digest(source/"manifest.json")
                or comparison.get("comparison_sha256") != base.digest(source/"comparison.json")):
            raise ValueError("A completed, sealed source comparison is required.")
        for name in source_config["variants"]:
            zonal.verify_result(source, name, source_manifest)
        if comparison.get("variant_manifests") != {n: base.digest(source/n/"results_manifest.json") for n in source_config["variants"]}:
            raise ValueError("Source comparison is not bound to its verified results.")
        panel = pd.read_parquet(source/"panel.parquet")
        settings = source_manifest["settings"]
        if settings["threads"] > 2 or settings["threads"]*config["max_parallel"] > 4:
            raise ValueError("At most four total CPU threads; two per worker.")
        baseline = json.loads((source/"base_config.json").read_text(encoding="utf-8"))
        audit = json.loads((source/"source_audit.json").read_text(encoding="utf-8"))
        audit.update(fundamental_source_suite=str(source), fundamental_source_manifest_sha256=base.digest(source/"manifest.json"),
                     source_settings=settings, primary_variant="fundamental", decision_policy="governed",
                     new_training_performed=True, diagnostic_only=True, production_modified=False, activation_performed=False)
        frames = {n: pd.read_parquet(source/(n+".parquet")) for n in ("hgb_predictions", "regional_25_predictions")}
        zonal.read_suite(source, root=root)
        if before != protected_state(root) or source_identity != base.digest(source/"manifest.json"):
            raise ValueError("Protected files changed during preparation.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
        directory = safe_path(root, output/"snapshots"/stamp)
        directory.mkdir(parents=True, exist_ok=False)
        for name, value in (("config", config), ("base_config", baseline), ("source_audit", audit)):
            base._json(directory/f"{name}.json", value)
        for name, value in {"panel": panel, **frames}.items():
            base._parquet(directory/f"{name}.parquet", value)
        manifest = {"schema_version": 1, "config": config, "settings": settings,
                    "input_files": {n: base.digest(directory/n) for n in sorted(INPUTS)},
                    "code_sha256": code_seals(root), "runtime": runtime_identity(root),
                    "protected_files": before, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "diagnostic_only": True, "activation_performed": False}
        base._json(directory/"manifest.json", manifest)
        base._json(directory/"status.json", {"status": "prepared", "snapshot": str(directory)})
        base._json(output/"latest_prepared.json", {"snapshot": str(directory)})
        LOGGER.info("[Fundamental] Prepared %s; %d unchanged rows.", directory, len(panel))
        return directory


def read_suite(directory: Path, *, root: Path):
    directory = safe_path(root, directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    base._verify(directory, manifest.get("input_files"), INPUTS)
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    validate_config(config)
    if config != manifest.get("config") or not directory.is_relative_to(safe_path(root, config["output_root"])/"snapshots"):
        raise ValueError("Fundamental manifest/config identity mismatch.")
    audit = json.loads((directory/"source_audit.json").read_text(encoding="utf-8"))
    if manifest.get("settings") != audit["source_settings"]:
        raise ValueError("Frozen model settings have changed.")
    return directory, config, manifest


def _verify_code(manifest: dict, root: Path) -> None:
    if manifest["code_sha256"] != code_seals(root) or manifest["runtime"] != runtime_identity(root):
        raise ValueError("Code/runtime changed since Prepare. Prepare a NEW snapshot; no stale partial fit is reused.")


def verify_result(directory: Path, name: str, manifest: dict) -> dict:
    if name not in manifest["config"]["variants"]:
        raise ValueError("Unregistered fundamental model.")
    result = json.loads((directory/name/"results_manifest.json").read_text(encoding="utf-8"))
    if (result.get("status") != "completed" or result.get("variant") != name
            or result.get("suite_manifest_sha256") != base.digest(directory/"manifest.json")):
        raise ValueError("Fundamental results not bound to this frozen suite.")
    base._verify(directory/name, result.get("result_files"), RESULTS)
    return result


def load_model(directory: Path, name: str, *, root: Path) -> dict:
    """Only load this lab's trusted, hash-verified local pickle; never a user upload."""
    directory, _, manifest = read_suite(directory, root=root)
    verify_result(directory, name, manifest)
    _verify_code(manifest, root)
    return joblib.load(directory/name/"latest_model.joblib")


def fixed_diagnostic(strict):
    """Reuse the frozen interval/weight protocol, retaining physical non-action reasons."""
    from nyx_scarcity.policy import PolicyResult
    fixed = fixed_conservative_forecast(strict, alpha=.25)
    out = fixed.predictions.copy()
    inactive = out.expert_ready & out["bounded_correction"].le(0)
    out.loc[inactive, "gate_reason"] = out.loc[inactive, "proposal_reason"]
    active = out["applied_correction"].gt(0)
    out.loc[active, "gate_reason"] = str(strict.audit["variant"])+"_proposal_fixed25_diagnostic"
    audit = dict(fixed.audit)
    audit.update(decision_policy="fixed25_diagnostic", fixed_alpha=.25, strict_governor_enforced=False,
                 strict_governor_enforced_in_this_point_forecast=False,
                 annual_non_regression_guaranteed=False)
    audit["point_functional"] = str(audit.get("point_functional", "")).replace("then_existing_governor", "then_fixed_quarter_weight")
    return PolicyResult(out, fixed.folds, fixed.governance, audit)


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
            strict = run_fundamental_policy(panel, manifest["settings"], name, on_last_fit=remember)
            fixed = fixed_diagnostic(strict)
            baseline = json.loads((directory/"base_config.json").read_text(encoding="utf-8"))
            for result in (strict, fixed):
                base.validate_predictions(panel, result.predictions, baseline)
            if not latest:
                raise ValueError("No trained fold: a viable experiment requires a fitted model.")
            for key, frame in (("predictions", strict.predictions), ("proposals_25", fixed.predictions),
                               ("folds", strict.folds), ("governance", strict.governance)):
                base._parquet(destination/f"{key}.parquet", old._audit_parquet_frame(frame) if key in ("folds", "governance") else frame)
            base._json(destination/"model_audit.json", {"fixed25": fixed.audit, "governed": strict.audit})
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
        predictions[name+"_25"] = pd.read_parquet(directory/name/"proposals_25.parquet")
        audits[name] = json.loads((directory/name/"model_audit.json").read_text(encoding="utf-8"))
    return predictions, audits


def run_suite(directory: Path, *, root: Path) -> Path:
    directory, config, manifest = read_suite(directory, root=root)
    with exclusive_process_lock(directory/"suite.lock"):
        if (directory/"comparison_manifest.json").is_file():
            LOGGER.info("[Fundamental] Completed predictions reused: no fitting.")
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
                        proc = subprocess.Popen([sys.executable, str(root/"run_nyx_fundamental_stress.py"), "--action", "worker",
                            "--run-directory", str(directory), "--variant", name], cwd=root, shell=False,
                            stdout=log, stderr=subprocess.STDOUT)
                    except BaseException:
                        log.close(); raise
                    active[name] = (proc, log, time.monotonic())
                    LOGGER.info("[Fundamental] Started %s; log=%s", name, destination/"worker.log")
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
                        LOGGER.info("[Fundamental] Completed %s (%d/2)", name, len(completed))
                if pending or active:
                    time.sleep(.5)
            from nyx_scarcity.variant_reporting import build_comparison
            base._json(directory/"status.json", {"status": "running", "stage": "comparison", "snapshot": str(directory),
                       "completed": completed, "active": [], "pending": []})
            predictions, _ = collect(directory, config, manifest)
            comparison = build_comparison(predictions)
            comparison.update(primary_variant="fundamental", fixed_alpha=.25,
                primary_selected_before_new_backtest=True, strict_governor_enforced_in_primary=True,
                annual_non_regression_guaranteed=False, prospective_validation_completed=False,
                exploratory_year_already_examined=True, price_features_used=False)
            comparison["model_labels"].update({"regional_25": "Expert régional précédent — 25 %",
                "fundamental": "Expert fondamental — gouverné (principal)",
                "fundamental_25": "Expert fondamental — 25 % (diagnostic)",
                "calendar": "Calendrier seul — gouverné",
                "calendar_25": "Calendrier seul — 25 % (diagnostic)"})
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


def _report(directory: Path, *, root: Path) -> Path:
    from nyx_scarcity.variant_reporting import render_comparison
    from .reporting import render_fundamental_reports
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
        base._json(directory/"status.json", {"status": "running", "stage": "reporting", "snapshot": str(directory),
                   "completed": list(config["variants"]), "active": [], "pending": []})
        render_comparison(predictions, comparison, {**manifest, "models": audits,
            "data": source["source_data_audit"], "diagnostic_only": True, "production_modified": False},
            directory/"fundamental_comparison.html")
        comparison_path = directory/"fundamental_comparison.html"
        document = comparison_path.read_text(encoding="utf-8")
        document = document.replace("NYX — comparaison contrôlée des experts XGB", "NYX — expert fondamental et contrôle calendrier")
        document = document.replace("NYX — variantes XGB et seuils causaux", "NYX — stress fondamental et correction gouvernée")
        document = document.replace("P.summary.variants.includes('xgb_unweighted_fixed')?'xgb_unweighted_fixed':'hgb_v1'",
                                    "P.summary.primary_variant||'hgb_v1'")
        document = document.replace("<main>", '<main><section><h2>Expert fondamental — laboratoire isolé</h2>'
            '<p>Candidat principal : fundamental avec gouverneur strict. Les propositions fixes à 25 % sont des '
            'diagnostics distincts, sans garantie de non-régression. Le témoin calendrier n’utilise aucun input physique '
            'ni gate physique. Les modèles appris n’utilisent aucun prix électrique, forecast NYX ou Storm en entrée. '
            'Cette année déjà examinée reste exploratoire : aucune promotion.</p>'
            '<p><a href="reports/index.html">Rapports gouvernés</a> · '
            '<a href="reports_fixed25/index.html">Rapports diagnostic à 25 %</a></p></section>', 1)
        comparison_path.write_text(document, encoding="utf-8")
        paths = {}
        for key, folder, policy in (("fundamental", "reports", "governed"), ("fundamental_25", "reports_fixed25", "fixed25")):
            rendered = render_fundamental_reports(predictions[key], source_audit={**source,
                "model_audit": audits["fundamental"], "decision_policy": policy,
                "strict_governor_enforced": policy == "governed"}, output_directory=safe_path(root, directory/folder),
                model_name="fundamental_stress" if policy == "governed" else "fundamental_stress_25",
                comparison=comparison, decision_policy=policy)
            paths.update({folder+"_"+k: p for k, p in rendered.items()})
        if before != protected_state(root):
            raise ValueError("Protected operational files changed concurrently; publication is not certified.")
        status = {"status": "completed", "snapshot": str(directory), "report": str(paths["reports_index"]),
                  "comparison_report": str(directory/"fundamental_comparison.html"), "production_modified": False,
                  "activation_performed": False, "diagnostic_only": True,
                  "reports_sha256": {str(p.relative_to(directory)): base.digest(p) for p in paths.values()}}
        base._json(directory/"status.json", status)
        base._json(safe_path(root, config["output_root"])/"latest.json", status)
        LOGGER.info("[Fundamental] Reports complete: %s", paths["reports_index"])
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
    pointer = safe_path(root, config["output_root"])/("latest.json" if completed else "latest_prepared.json")
    if not pointer.is_file():
        raise ValueError("No zonal experiment yet; use Run or Prepare first.")
    return safe_path(root, json.loads(pointer.read_text(encoding="utf-8"))["snapshot"])
