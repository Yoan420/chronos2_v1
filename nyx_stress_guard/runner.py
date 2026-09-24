"""Private, sealed research snapshots. Nothing writes to Forecast.ps1 or exports."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import uuid

import joblib
import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock
from nyx_scarcity import runner as base, variants_runner as old
from nyx_coherent_p50 import runner as previous

LOGGER = logging.getLogger(__name__)
NAMESPACE = base.NAMESPACE/"stress_guard"
INPUTS = {"config.json", "base_config.json", "source_audit.json", "panel.parquet",
          "previous_p50.parquet", "previous_empirical.parquet", "hgb_predictions.parquet"}
OUTPUTS = {"physics_direct.parquet", "physics_governed.parquet", "p50_calibrated.parquet",
           "folds.parquet", "governance.parquet", "model_audit.json", "latest_model.joblib"}


def safe_path(root, value):
    root = Path(root).resolve()
    raw = Path(value)
    if ".." in raw.parts:
        raise ValueError("Parent traversal is forbidden.")
    path = base.safe_output(root, raw)
    namespace = root/NAMESPACE
    if namespace.resolve() != namespace or path == namespace or not path.is_relative_to(namespace):
        raise ValueError("StressGuard outputs must stay inside the isolated stress_guard namespace.")
    return path


def code_seals(root):
    names = [p.relative_to(root).as_posix() for p in sorted((root/"nyx_stress_guard").glob("*.py"))]
    names += ["run_nyx_stress_guard.py", "StressGuard.ps1"]
    return {**previous.code_seals(root), **{n: base.digest(root/n) for n in names}}


def validate_config(config):
    keys = {"schema_version", "source_suite", "output_root", "primary_variant", "event_probability_gate",
            "amplitude_kind", "intervals", "prospective", "diagnostic_only", "production_modified", "activation_performed"}
    if not isinstance(config, dict) or set(config) != keys or type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Complete StressGuard schema1 required, unknown fields forbidden.")
    if (config["primary_variant"] != "physics_governed" or config["event_probability_gate"] != .5
            or isinstance(config["event_probability_gate"], bool) or config["amplitude_kind"] != "empirical"):
        raise ValueError("Predeclared governed physics / p>.5 / empirical amplitude recipe required.")
    if not isinstance(config["intervals"], dict) or config["intervals"]:
        raise ValueError("This preregistered experiment uses fixed interval defaults; no posthoc tuning grid.")
    prospective = config["prospective"]
    if (not isinstance(prospective, dict) or set(prospective) != {"issue_policy", "deadline_local"}
            or prospective["issue_policy"] not in {"pre_observation_asof08", "strict_08_issue"}
            or prospective["deadline_local"] != "11:45"
            or config["diagnostic_only"] is not True or config["production_modified"] is not False
            or config["activation_performed"] is not False):
        raise ValueError("Explicit research-only prospective policy required; activation forbidden.")
    if any(not isinstance(config[k], str) or not config[k].strip() for k in ("source_suite", "output_root")):
        raise ValueError("Explicit source/output paths required.")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def prepare(config, *, root):
    root = Path(root).resolve()
    validate_config(config)
    output = root/Path(config["output_root"])
    safe_path(root, output/"snapshots")
    before = previous.protected_state(root)
    source, _, source_manifest = previous.read_suite(root/Path(config["source_suite"]), root=root)
    source_status = json.loads((source/"status.json").read_text(encoding="utf-8"))
    if source_status.get("status") != "completed":
        raise ValueError("A completed original P50 experiment is required.")
    for kind in ("forest", "empirical"):
        previous.verify_result(source, kind, source_manifest)
    identity = {"manifest": base.digest(source/"manifest.json"),
        "forest_result": base.digest(source/"forest/results_manifest.json"),
        "empirical_result": base.digest(source/"empirical/results_manifest.json")}
    baseline = json.loads((source/"base_config.json").read_text(encoding="utf-8"))
    audit = json.loads((source/"source_audit.json").read_text(encoding="utf-8"))
    settings = source_manifest["settings"]
    if audit.get("source_settings") != settings:
        raise ValueError("Source model settings differ from the sealed source audit.")
    if settings["threads"] > 2 or settings["correction_clip_eur_mwh"] != 400.:
        raise ValueError("At most two threads and fixed400 EUR/MWh cap required.")
    frames = {"panel": pd.read_parquet(source/"panel.parquet"),
        "previous_p50": pd.read_parquet(source/"forest/predictions.parquet"),
        "previous_empirical": pd.read_parquet(source/"empirical/predictions.parquet"),
        "hgb_predictions": pd.read_parquet(source/"hgb_predictions.parquet")}
    for key in ("previous_p50", "previous_empirical"):
        pd.testing.assert_frame_equal(frames[key][frames["panel"].columns], frames["panel"], check_exact=True)
    for kind in ("forest", "empirical"):
        previous.verify_result(source, kind, source_manifest)
    _, _, checked_manifest = previous.read_suite(source, root=root)
    checked_identity = {"manifest": base.digest(source/"manifest.json"),
        "forest_result": base.digest(source/"forest/results_manifest.json"),
        "empirical_result": base.digest(source/"empirical/results_manifest.json")}
    if checked_manifest != source_manifest or checked_identity != identity:
        raise ValueError("Source identity changed while preparing the frozen input copy.")
    if before != previous.protected_state(root):
        raise ValueError("Protected files changed during preparation.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    directory = safe_path(root, output/"snapshots"/stamp)
    with exclusive_process_lock(safe_path(root, output/"prepare.lock")):
        directory.mkdir(parents=True, exist_ok=False)
        audit.update(stress_guard_source=str(source), stress_guard_source_identity=identity,
            primary_variant="physics_governed", classifier_refitted=True, independent_validation=False,
            diagnostic_only=True, production_modified=False, activation_performed=False)
        for name, value in (("config", config), ("base_config", baseline), ("source_audit", audit)):
            base._json(directory/f"{name}.json", value)
        for name, frame in frames.items():
            base._parquet(directory/f"{name}.parquet", frame)
        manifest = {"schema_version": 1, "config": config, "settings": settings,
            "input_files": {n: base.digest(directory/n) for n in sorted(INPUTS)},
            "code_sha256": code_seals(root), "runtime": previous.runtime_identity(root),
            "protected_files": before, "source_identity": identity,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_declared_before_new_replay": True, "exploratory_year_already_examined": True,
            "prospective_validation_completed": False, "production_modified": False, "activation_performed": False}
        base._json(directory/"manifest.json", manifest)
        base._json(directory/"status.json", {"status": "prepared", "snapshot": str(directory)})
        base._json(safe_path(root, output/"latest_prepared.json"), {"snapshot": str(directory)})
    LOGGER.info("[StressGuard] Prepared %s; immutable inputs=%d rows", directory, len(frames["panel"]))
    return directory


def read_suite(directory, *, root):
    root = Path(root).resolve()
    directory = safe_path(root, directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    base._verify(directory, manifest.get("input_files"), INPUTS)
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    audit = json.loads((directory/"source_audit.json").read_text(encoding="utf-8"))
    validate_config(config)
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("production_modified") is not False or manifest.get("activation_performed") is not False
            or manifest.get("settings") != audit.get("source_settings")
            or manifest.get("source_identity") != audit.get("stress_guard_source_identity")):
        raise ValueError("Sealed settings, source identity or research-only manifest flags differ from their frozen input audit.")
    if config != manifest.get("config") or not directory.is_relative_to(root/Path(config["output_root"])/"snapshots"):
        raise ValueError("Config/manifest identity mismatch.")
    if manifest.get("code_sha256") != code_seals(root):
        raise ValueError("Sealed StressGuard source changed; prepare a new snapshot, never repair the old hashes.")
    if manifest.get("runtime") != previous.runtime_identity(root):
        raise ValueError("StressGuard runtime changed.")
    if manifest.get("protected_files") != previous.protected_state(root):
        raise ValueError("Protected operational files changed since preparation.")
    return directory, config, manifest


def verify_result(directory, manifest):
    saved = json.loads((directory/"results_manifest.json").read_text(encoding="utf-8"))
    if saved.get("status") != "completed" or saved.get("suite_manifest_sha256") != base.digest(directory/"manifest.json"):
        raise ValueError("Incomplete or mismatched StressGuard result.")
    base._verify(directory, saved.get("result_files"), OUTPUTS)
    return saved


def load_model(directory, *, root):
    directory, _, manifest = read_suite(directory, root=root)
    verify_result(directory, manifest)
    # Only a locally generated, sealed artifact in this namespace is deserialised.
    return joblib.load(directory/"latest_model.joblib")


def run_suite(directory, *, root):
    from .policy import run_stress_policy, recalibrate_previous
    directory, config, manifest = read_suite(directory, root=root)
    with exclusive_process_lock(directory/"run.lock"):
        if (directory/"results_manifest.json").exists():
            verify_result(directory, manifest)
            LOGGER.info("[StressGuard] Completed replay verified; reuse without refitting: %s", directory)
            return report(directory, root=root)
        base._json(directory/"status.json", {"status": "running", "stage": "backtest", "snapshot": str(directory)})
        try:
            panel = pd.read_parquet(directory/"panel.parquet")
            states = []
            def capture(state, settings):
                states[:] = [{"state": state, "settings": settings}]
            result = run_stress_policy(panel, manifest["settings"], interval_settings=config["intervals"], on_last_fit=capture)
            if not states:
                raise ValueError("No trained model exists; no prospective artifact can be frozen.")
            LOGGER.info("[StressGuard] Saving direct/governed predictions; calibrating the sealed previous P50 interval-only control")
            for name in ("direct", "governed"):
                base._parquet(directory/f"physics_{name}.parquet", result[name].predictions)
            calibrated, cal_audit = recalibrate_previous(pd.read_parquet(directory/"previous_p50.parquet"), interval_settings=config["intervals"])
            LOGGER.info("[StressGuard] Control calibration completed; saving and verifying result artifacts")
            base._parquet(directory/"p50_calibrated.parquet", calibrated)
            base._parquet(directory/"folds.parquet", old._audit_parquet_frame(result["governed"].folds))
            base._parquet(directory/"governance.parquet", old._audit_parquet_frame(result["governed"].governance))
            base._json(directory/"model_audit.json", {"direct": result["direct"].audit,
                "governed": result["governed"].audit, "p50_calibrated": cal_audit})
            joblib.dump(states[0], directory/"latest_model.joblib", compress=3)
            read_suite(directory, root=root)
            base._json(directory/"results_manifest.json", {"status": "completed",
                "suite_manifest_sha256": base.digest(directory/"manifest.json"),
                "result_files": {n: base.digest(directory/n) for n in sorted(OUTPUTS)}})
            LOGGER.info("[StressGuard] Replay results sealed; generating HTML reports")
            return report(directory, root=root)
        except BaseException as exc:
            base._json(directory/"status.json", {"status": "failed", "snapshot": str(directory), "error": str(exc)})
            raise


def collect(directory):
    return {"hgb_v1": pd.read_parquet(directory/"hgb_predictions.parquet"),
        "p50_previous": pd.read_parquet(directory/"previous_p50.parquet"),
        "empirical_previous": pd.read_parquet(directory/"previous_empirical.parquet"),
        **{name: pd.read_parquet(directory/f"{name}.parquet") for name in ("p50_calibrated", "physics_direct", "physics_governed")}}


def report(directory, *, root):
    from .reporting import render_reports
    directory, _, manifest = read_suite(directory, root=root)
    verify_result(directory, manifest)
    with exclusive_process_lock(directory/"report.lock"):
        base._json(directory/"status.json", {"status": "running", "stage": "reporting", "snapshot": str(directory)})
        try:
            LOGGER.info("[StressGuard] Rendering reports in %s", directory)
            paths = render_reports(collect(directory), source_audit=json.loads((directory/"source_audit.json").read_text(encoding="utf-8")),
                model_audit=json.loads((directory/"model_audit.json").read_text(encoding="utf-8")), output_directory=directory, root=root)
            read_suite(directory, root=root)
            for path in paths.values():
                if not safe_path(root, path).is_relative_to(directory) or not path.is_file():
                    raise ValueError("A reporter returned an artifact outside its private snapshot.")
            status = {"status": "completed", "snapshot": str(directory), "report": str(paths["index"]),
                "report_files": {str(p.relative_to(directory)): base.digest(p) for p in paths.values()},
                "diagnostic_only": True, "production_modified": False, "activation_performed": False,
                "prospective_validation_completed": False}
            base._json(directory/"status.json", status)
            base._json(safe_path(root, root/NAMESPACE/"latest.json"), status)
            LOGGER.info("[StressGuard] Completed: %s", paths["index"])
            return paths["index"]
        except BaseException as exc:
            base._json(directory/"status.json", {"status": "failed", "stage": "reporting", "error": str(exc)})
            raise


def resolve_latest(config, *, root, completed=False):
    name = "latest.json" if completed else "latest_prepared.json"
    index = safe_path(root, Path(root)/config["output_root"]/name)
    return safe_path(root, json.loads(index.read_text(encoding="utf-8"))["snapshot"])
