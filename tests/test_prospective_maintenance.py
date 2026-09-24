from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from chronos2_exogenous import prospective_maintenance as maintenance


def _entry(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


@pytest.fixture
def trial(tmp_path, monkeypatch):
    output = tmp_path / "runs/experiments/trial"
    directory = output / maintenance.REVISION_DIRECTORY
    directory.mkdir(parents=True)
    runner = tmp_path / maintenance.RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_bytes(b"# original runner\n")
    before = _entry(runner)
    monkeypatch.setattr(maintenance, "ORIGINAL_RUNNER_SHA256", before["sha256"])
    (directory / "prospective_trial.before.py").write_bytes(runner.read_bytes())
    other = runner.parent / "numerical_model.py"
    other.write_bytes(b"# unchanged scientific code\n")
    for name in maintenance.ADDED_CODE:
        (tmp_path / name).write_bytes(b"# new maintenance code\n")
    config = {"output_root": str(output), "project_root": str(tmp_path)}
    manifest = {"config": config, "created_at_utc": "2026-01-01T00:00:00+00:00",
                "code_files": [before, _entry(other)]}
    _write_json(output / "trial_manifest.json", manifest)
    return config, manifest, runner, other


def _register(trial):
    config, manifest, runner, other = trial
    runner.write_bytes(b"# repaired bootstrap runner\n")
    return maintenance.register_bootstrap_resume_revision(config)


def test_original_code_is_verified_without_revision(trial):
    config, manifest, _, _ = trial
    assert maintenance.verify_trial_code(manifest, Path(config["output_root"])) is None
    assert not (Path(config["output_root"]) / maintenance.REVISION_DIRECTORY / "revision.json").exists()


def test_unregistered_changed_runner_is_rejected(trial):
    config, manifest, runner, _ = trial
    runner.write_bytes(b"changed")
    with pytest.raises(maintenance.ProspectiveMaintenanceError, match="Empreinte"):
        maintenance.verify_trial_code(manifest, Path(config["output_root"]))


def test_register_is_append_only_and_idempotent(trial):
    config, manifest, _, _ = trial
    output = Path(config["output_root"])
    before = (output / "trial_manifest.json").read_bytes()
    first = _register(trial)
    revision_path = Path(first["revision_path"])
    revision_bytes = revision_path.read_bytes()
    second = maintenance.register_bootstrap_resume_revision(config)
    assert first["status"] == "registered"
    assert second["status"] == "existing"
    assert revision_path.read_bytes() == revision_bytes
    assert (output / "trial_manifest.json").read_bytes() == before
    assert maintenance.verify_trial_code(manifest, output) == pd.Timestamp(first["revision"]["created_at_utc"])
    assert first["revision"]["original_runner"] == manifest["code_files"][0]
    for key, expected in maintenance.FLAGS.items():
        assert first["revision"][key] is expected


@pytest.mark.parametrize("mode", ["wrong_backup", "wrong_manifest_original", "other_code_changed"])
def test_register_rejects_unapproved_changes(trial, mode):
    config, manifest, runner, other = trial
    output = Path(config["output_root"])
    if mode == "wrong_backup":
        (output / maintenance.REVISION_DIRECTORY / "prospective_trial.before.py").write_bytes(b"wrong")
    elif mode == "wrong_manifest_original":
        manifest["code_files"][0]["sha256"] = "0" * 64
        _write_json(output / "trial_manifest.json", manifest)
    else:
        other.write_bytes(b"changed numerical model")
    with pytest.raises(maintenance.ProspectiveMaintenanceError):
        _register(trial)
    assert not (output / maintenance.REVISION_DIRECTORY / "revision.json").exists()


@pytest.mark.parametrize("part", ["after_runner", "added_0", "added_1", "backup", "other_code", "trial_manifest"])
def test_registered_revision_does_not_accept_later_changes(trial, part):
    config, manifest, runner, other = trial
    registered = _register(trial)
    output = Path(config["output_root"])
    paths = {"after_runner": runner, "added_0": Path(config["project_root"]) / maintenance.ADDED_CODE[0],
             "added_1": Path(config["project_root"]) / maintenance.ADDED_CODE[1],
             "backup": output / maintenance.REVISION_DIRECTORY / "prospective_trial.before.py",
             "other_code": other, "trial_manifest": output / "trial_manifest.json"}
    path = paths[part]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(maintenance.ProspectiveMaintenanceError):
        maintenance.verify_trial_code(manifest, output)


@pytest.mark.parametrize("mode", ["prospective", "ambiguous", "partial"])
def test_registration_refuses_existing_prospective_or_ambiguous_days(trial, mode):
    config, _, _, _ = trial
    daily_dir = Path(config["output_root"]) / "days/2026-09-08/FR"
    daily_dir.mkdir(parents=True)
    if mode == "partial":
        (daily_dir / "predictions.csv.gz").write_bytes(b"partial publication")
    else:
        payload = {"prospective_eligible": True, "role": "prospective_two_chain_forecast"} if mode == "prospective" else {}
        _write_json(daily_dir / "manifest.json", payload)
    with pytest.raises(maintenance.ProspectiveMaintenanceError, match="prospective"):
        _register(trial)


def test_retrospective_days_do_not_prevent_registration(trial):
    config, _, _, _ = trial
    daily_dir = Path(config["output_root"]) / "days/2026-09-08/FR"
    daily_dir.mkdir(parents=True)
    path = daily_dir / "manifest.json"
    _write_json(path, {"prospective_eligible": False, "role": "retrospective_calibration_only"})
    before = path.read_bytes()
    _register(trial)
    assert path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["outside_runner", "different_inside_runner", "wrong_added_path",
                                    "original_hash", "after_hash", "before_time", "naive_time", "future_time",
                                    "flag", "reason", "extra_file"])
def test_revision_strict_whitelist_and_provenance(trial, tmp_path, mutation):
    config, manifest, _, _ = trial
    registered = _register(trial)
    revision = deepcopy(registered["revision"])
    if mutation == "outside_runner":
        revision["after_runner"]["path"] = str(tmp_path.parent / "outside.py")
    elif mutation == "different_inside_runner":
        revision["after_runner"] = revision["added_code_files"][0]
    elif mutation == "wrong_added_path":
        revision["added_code_files"][0] = revision["after_runner"]
    elif mutation == "original_hash":
        revision["original_runner"]["sha256"] = "0" * 64
    elif mutation == "after_hash":
        revision["after_runner"]["sha256"] = "0" * 64
    elif mutation == "before_time":
        revision["created_at_utc"] = "2025-01-01T00:00:00Z"
    elif mutation == "naive_time":
        revision["created_at_utc"] = "2026-01-01T00:00:00"
    elif mutation == "future_time":
        revision["created_at_utc"] = "2099-01-01T00:00:00Z"
    elif mutation == "flag":
        revision["promotion_eligible"] = True
    elif mutation == "reason":
        revision["reason"] = "arbitrary code override"
    elif mutation == "extra_file":
        revision["added_code_files"].append(revision["after_runner"])
    _write_json(Path(registered["revision_path"]), revision)
    with pytest.raises(maintenance.ProspectiveMaintenanceError):
        maintenance.verify_trial_code(manifest, Path(config["output_root"]))


def test_future_prospective_entries_do_not_invalidate_prior_revision(trial):
    config, manifest, _, _ = trial
    registered = _register(trial)
    output = Path(config["output_root"])
    daily_dir = output / "days/2026-09-09/FR"
    daily_dir.mkdir(parents=True)
    _write_json(daily_dir / "manifest.json", {"prospective_eligible": True})
    assert maintenance.verify_trial_code(manifest, output) == pd.Timestamp(registered["revision"]["created_at_utc"])
    assert maintenance.register_bootstrap_resume_revision(config)["status"] == "existing"


def test_supplied_manifest_cannot_differ_from_sealed_document(trial):
    config, manifest, _, _ = trial
    manifest = deepcopy(manifest)
    manifest["config"]["invented"] = True
    with pytest.raises(maintenance.ProspectiveMaintenanceError, match="different"):
        maintenance.verify_trial_code(manifest, Path(config["output_root"]))
