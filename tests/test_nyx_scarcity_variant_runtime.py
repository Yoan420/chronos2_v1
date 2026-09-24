"""Private installer lifecycle tests; no package download or shared mutation."""
from pathlib import Path
import subprocess
import sys

import pytest

from chronos2_hourly.process_lock import exclusive_process_lock
from nyx_scarcity import variant_runtime as runtime


@pytest.fixture
def root(tmp_path):
    at = tmp_path / "project"
    (at / "config").mkdir(parents=True)
    (at / "config/nyx_scarcity_requirements.txt").write_text("pinned test recipe\n")
    return at


def _package(target, *, version="3.2.0"):
    package = target / "xgboost"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"__version__ = {version!r}\n", encoding="utf-8")


def _stages(root):
    target = runtime.target_path(root)
    return list(target.parent.glob(f".{target.name}.stage_*"))


def _fake_success(monkeypatch, root, calls):
    def fake(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs == {"cwd": root.resolve(), "shell": False, "check": True}
        if command[1:4] == ["-m", "pip", "install"]:
            stage = Path(command[command.index("--target") + 1])
            assert stage != runtime.target_path(root)
            _package(stage)
        else:
            assert command[1:3] == ["-I", "-c"]
            assert (Path(command[-2]) / "xgboost/__init__.py").is_file()
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runtime.subprocess, "run", fake)


def test_install_publishes_validated_sibling_and_never_imports_stage(root, monkeypatch):
    calls = []
    _fake_success(monkeypatch, root, calls)
    monkeypatch.setattr(runtime, "ensure_runtime", lambda *a, **k: pytest.fail("Parent imported stage"))
    before_path = list(sys.path)
    before_module = sys.modules.get("xgboost")
    target = runtime.install_runtime(root)
    assert target == runtime.target_path(root)
    assert (target / "xgboost/__init__.py").is_file()
    assert len(calls) == 2 and not _stages(root)
    pip = calls[0][0]
    assert "--no-deps" in pip and "--require-hashes" in pip and "--only-binary=:all:" in pip
    assert pip[pip.index("-r") + 1] == str(root / "config/nyx_scarcity_requirements.txt")
    assert sys.path == before_path and sys.modules.get("xgboost") is before_module


def test_valid_existing_install_is_validated_and_not_overwritten(root, monkeypatch):
    target = runtime.target_path(root)
    _package(target)
    original = (target / "xgboost/__init__.py").read_bytes()
    calls = []
    _fake_success(monkeypatch, root, calls)
    assert runtime.install_runtime(root) == target
    assert len(calls) == 1 and calls[0][0][1:3] == ["-I", "-c"]
    assert (target / "xgboost/__init__.py").read_bytes() == original
    assert not _stages(root)


def test_incomplete_existing_install_preserved_with_recovery_instructions(root, monkeypatch):
    target = runtime.target_path(root)
    target.mkdir(parents=True)
    (target / "partial.txt").write_text("keep")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess"))
    with pytest.raises(ValueError, match="move it aside manually"):
        runtime.install_runtime(root)
    assert (target / "partial.txt").read_text() == "keep"
    assert not _stages(root)


@pytest.mark.parametrize("failure_stage", ["pip", "validation"])
def test_failed_stage_retained_and_retry_uses_new_stage(root, monkeypatch, failure_stage):
    def fail(command, **kwargs):
        if command[1:4] == ["-m", "pip", "install"]:
            _package(Path(command[command.index("--target") + 1]))
            if failure_stage == "validation":
                return subprocess.CompletedProcess(command, 0)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(runtime.subprocess, "run", fail)
    with pytest.raises(ValueError, match="stage preserved.*Retry Install"):
        runtime.install_runtime(root)
    failed = _stages(root)
    assert len(failed) == 1 and (failed[0] / "xgboost/__init__.py").is_file()
    assert not runtime.target_path(root).exists()
    calls = []
    _fake_success(monkeypatch, root, calls)
    runtime.install_runtime(root)
    assert _stages(root) == failed
    assert Path(calls[0][0][calls[0][0].index("--target") + 1]) != failed[0]


def test_interruption_retains_stage_and_releases_installer_lock(root, monkeypatch):
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        runtime.install_runtime(root)
    assert len(_stages(root)) == 1 and not runtime.target_path(root).exists()
    calls = []
    _fake_success(monkeypatch, root, calls)
    assert runtime.install_runtime(root).is_dir()


def test_concurrent_installer_refused_without_starting_pip(root, monkeypatch):
    target = runtime.target_path(root)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: pytest.fail("Second installer started"))
    with exclusive_process_lock(target.parent / f".{target.name}.install.lock"):
        with pytest.raises(ValueError, match="Verrou"):
            runtime.install_runtime(root)
    assert not target.exists() and not _stages(root)


def test_destination_created_during_install_is_never_replaced(root, monkeypatch):
    target = runtime.target_path(root)
    calls = []
    _fake_success(monkeypatch, root, calls)
    def validate(stage, **kwargs):
        target.mkdir()
        (target / "other_install.txt").write_text("keep")
    monkeypatch.setattr(runtime, "_validate_private_install", validate)
    with pytest.raises(ValueError, match="stage preserved"):
        runtime.install_runtime(root)
    assert (target / "other_install.txt").read_text() == "keep"
    assert len(_stages(root)) == 1


def test_missing_pinned_recipe_stops_before_creating_stage(root, monkeypatch):
    (root / "config/nyx_scarcity_requirements.txt").unlink()
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected download"))
    with pytest.raises(ValueError, match="requirements missing"):
        runtime.install_runtime(root)
    assert not _stages(root) and not runtime.target_path(root).exists()


def test_private_validation_uses_real_fresh_process_without_parent_import(root):
    target = runtime.target_path(root)
    _package(target)
    before = sys.modules.get("xgboost")
    path = list(sys.path)
    runtime._validate_private_install(target, root=root)
    assert sys.modules.get("xgboost") is before and sys.path == path


def test_private_validation_rejects_wrong_version(root):
    target = runtime.target_path(root)
    _package(target, version="0.0.0")
    with pytest.raises(subprocess.CalledProcessError):
        runtime._validate_private_install(target, root=root)


def test_private_validation_rejects_module_outside_target(root):
    target = runtime.target_path(root)
    _package(target)
    with (target / "xgboost/__init__.py").open("a", encoding="utf-8") as stream:
        stream.write(f"__file__ = {str(root / 'outside.py')!r}\n")
    with pytest.raises(subprocess.CalledProcessError):
        runtime._validate_private_install(target, root=root)
