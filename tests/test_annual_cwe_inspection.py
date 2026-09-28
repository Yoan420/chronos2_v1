from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from experiment_console.annual_cwe_inspection import MANIFEST_PATH, inspect_annual_cwe


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_artifact(root: Path, spec: dict, content: bytes) -> None:
    path = root / spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    spec["sha256"] = hashlib.sha256(content).hexdigest()


def _fixture_root(tmp_path: Path, *, with_artifacts: bool) -> Path:
    manifest = json.loads((REPO_ROOT / MANIFEST_PATH).read_text(encoding="utf-8"))
    # pytest's Windows temp root may already be long; the archived report path
    # and checkpoint paths can exceed MAX_PATH. Keep real paths in the shipped
    # manifest, and shorten every synthetic path in this isolated fixture.
    for name, spec in manifest["evidence"].items():
        spec["path"] = f"e/{name}.json"
    for name, spec in manifest["source_feature_matrices"].items():
        spec["path"] = f"f/{name}.parquet"
    for zone, spec in manifest["historical_references"].items():
        spec["path"] = f"r/{zone}.parquet"
    for name in ("price_overview", "price_kpi", "negative_overview"):
        manifest["reports"][name]["path"] = f"test_reports/{name}.html"
    for zone, kinds in manifest["reports"]["countries"].items():
        for kind, spec in kinds.items():
            spec["path"] = f"test_reports/{zone}_{kind}.html"
    for name, checkpoint in manifest["checkpoints"].items():
        checkpoint["attempt_directory"] = "attempt-test"
        checkpoint["receipt"]["path"] = f"c/{name}/receipt.json"
        checkpoint["model"]["path"] = f"c/{name}/attempt-test/model.cbm"
        if "calibration" in checkpoint:
            checkpoint["calibration"]["path"] = f"c/{name}/attempt-test/model.json"
    if with_artifacts:
        for name, spec in manifest["evidence"].items():
            _write_artifact(tmp_path, spec, f"evidence:{name}".encode())
        for name, spec in manifest["source_feature_matrices"].items():
            _write_artifact(tmp_path, spec, f"feature:{name}".encode())
        for name, spec in manifest["historical_references"].items():
            _write_artifact(tmp_path, spec, f"reference:{name}".encode())
        for name in ("price_overview", "price_kpi", "negative_overview"):
            _write_artifact(tmp_path, manifest["reports"][name], f"report:{name}".encode())
        for zone, kinds in manifest["reports"]["countries"].items():
            for kind, spec in kinds.items():
                _write_artifact(tmp_path, spec, f"report:{zone}:{kind}".encode())
        for name, checkpoint in manifest["checkpoints"].items():
            _write_artifact(tmp_path, checkpoint["model"], f"model:{name}".encode())
            files = {"model.cbm": checkpoint["model"]["sha256"]}
            if "calibration" in checkpoint:
                _write_artifact(tmp_path, checkpoint["calibration"], f"calibration:{name}".encode())
                files["model.json"] = checkpoint["calibration"]["sha256"]
            receipt = {
                "state": "COMPLETE",
                "attempt_directory": checkpoint["attempt_directory"],
                "files": files,
            }
            _write_artifact(tmp_path, checkpoint["receipt"], json.dumps(receipt).encode())
    path = tmp_path / MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_clean_clone_reports_missing_sources_and_never_enables_forecast(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, with_artifacts=False)
    result = inspect_annual_cwe(root, ("FR", "BE"))
    assert result["manifest_valid"]
    assert result["forecast_ready"] is False
    assert result["evidence"]["session_summary"]["status"] == "missing"
    assert result["source_feature_matrices"]["pooled_FR"]["status"] == "missing"
    assert result["checkpoints"]["pooled_residual"]["model"]["status"] == "missing"
    assert result["countries"]["FR"]["archived_files_verified"] is False
    assert result["reports"]["countries"]["FR"]["price"]["status"] == "missing"
    assert set(result["countries"]) == {"FR", "BE"}
    assert not (root / "runs").exists()


def test_complete_historical_archive_is_still_inspection_only(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, with_artifacts=True)
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    result = inspect_annual_cwe(root)
    after = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    assert before == after
    assert all(item["archived_files_verified"] for item in result["countries"].values())
    assert all(item["reports_verified"] for item in result["countries"].values())
    assert result["forecast_ready"] is False
    assert all(item["forecast_ready"] is False for item in result["countries"].values())
    assert "storm_goal_not_met" in {item["code"] for item in result["countries"]["DE"]["blockers"]}
    assert "storm_goal_not_met" not in {item["code"] for item in result["countries"]["FR"]["blockers"]}


def test_hash_mismatch_and_preflight_exit_code(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, with_artifacts=True)
    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    feature_path = root / manifest["source_feature_matrices"]["compact_FR"]["path"]
    feature_path.write_bytes(b"changed source matrix")
    result = inspect_annual_cwe(root, ("FR",))
    assert result["source_feature_matrices"]["compact_FR"]["status"] == "sha256_mismatch"
    assert "source_feature_matrices_unverified" in {item["code"] for item in result["countries"]["FR"]["blockers"]}
    run = subprocess.run(
        [sys.executable, str(REPO_ROOT / "inspect_nyx_annual_cwe.py"), "--root", str(root), "--country", "FR", "--preflight"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 2
    assert json.loads(run.stdout)["forecast_ready"] is False


def test_manifest_cannot_point_outside_checkout(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, with_artifacts=False)
    path = root / MANIFEST_PATH
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["source_feature_matrices"]["pooled_FR"]["path"] = "../outside.parquet"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = inspect_annual_cwe(root, ("FR",))
    assert result["source_feature_matrices"]["pooled_FR"]["status"] == "invalid_manifest"
    assert result["forecast_ready"] is False
