"""Offline transfer fixtures contain no corporate client or corporate code."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from importlib.metadata import Distribution
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

import prepare_nyx_saturn as installer


def distribution(root, name, version="0.5", requires=(), tag="py3-none-any", extra=None):
    normalized = name.replace("-", "_")
    info = root / f"{normalized}-{version}.dist-info"
    info.mkdir(parents=True)
    files = {
        f"{normalized}/__init__.py": b"FIXTURE = True\n",
        f"{info.name}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {value}\n" for value in requires) + "\n"
        ).encode(),
        f"{info.name}/WHEEL": f"Wheel-Version: 1.0\nGenerator: nyx-test-fixture\nRoot-Is-Purelib: true\nTag: {tag}\n".encode(),
        f"{info.name}/direct_url.json": b'{"url":"https://private.invalid/local-only"}',
    }
    if name == "tshistory_lite":
        files[f"{normalized}/__init__.py"] = (
            "import nyx_saturn_fixture_dep\n"
            "class Client:\n"
            "    def __init__(self, *args, **kwargs): raise AssertionError('No connection during probe')\n"
            "    def get(self, *args, **kwargs): raise AssertionError('No network')\n"
            "    def block_staircase(self, *args, **kwargs): raise AssertionError('No network')\n"
        ).encode()
    files.update(extra or {})
    rows = []
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        rows.append([name, installer._digest(value), str(len(value))])
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows + [[f"{info.name}/RECORD", "", ""]])
    (info / "RECORD").write_text(output.getvalue(), encoding="utf-8")
    return Distribution.at(info)


def runtime():
    result = installer.runtime_info()
    result["installed"] = {}
    return result


def hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}


def test_real_offline_transfer_and_install_keeps_source_unchanged(tmp_path):
    source = tmp_path / "source"
    dep = distribution(source, "nyx_saturn_fixture_dep", "1.2")
    client = distribution(source, "tshistory_lite", requires=["nyx_saturn_fixture_dep>=1", 'ignored_fixture; python_version < "3.0"'])
    original = hashes(source)
    wheelhouse = tmp_path / "wheels"
    by_name = {"tshistory_lite": client, "nyx_saturn_fixture_dep": dep}
    manifest = installer.export_installed("tshistory_lite==0.5", wheelhouse, runtime(), distribution=by_name.__getitem__)
    assert {item["name"] for item in manifest["packages"]} == set(by_name)
    assert hashes(source) == original
    for path in wheelhouse.glob("*.whl"):
        with zipfile.ZipFile(path) as archive:
            assert not any("direct_url" in name for name in archive.namelist())
    environment = tmp_path / "target"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True, capture_output=True)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse), "tshistory_lite==0.5"], check=True, capture_output=True)
    probe = subprocess.run([str(python), "-B", installer.__file__, "probe"], check=True, capture_output=True, text=True)
    assert json.loads(probe.stdout)["ok"] is True
    assert hashes(source) == original


def test_tampered_source_record_rejected(tmp_path):
    dist = distribution(tmp_path, "fixture")
    (tmp_path / "fixture/__init__.py").write_text("CHANGED = True\n")
    with pytest.raises(ValueError, match="RECORD altere"):
        installer.repack_distribution(dist, tmp_path / "wheels", runtime())


@pytest.mark.parametrize("tag", ["cp310-cp310-win_amd64", "cp311-cp311-manylinux_2_17_aarch64"])
def test_compiled_source_wheel_must_match_target(tmp_path, tag):
    dist = distribution(tmp_path, "fixture", tag=tag)
    with pytest.raises(ValueError, match="incompatible"):
        installer.repack_distribution(dist, tmp_path / "wheels", runtime())


def test_rejects_source_python310_and_public_version_conflict(tmp_path, monkeypatch):
    target = runtime()
    monkeypatch.setattr(installer.sys, "version_info", (3, 10, 9))
    with pytest.raises(ValueError, match="Python 3.11"):
        installer.export_installed("fixture", tmp_path / "wheels", target)
    monkeypatch.undo()
    dist = distribution(tmp_path, "fixture", requires=["numpy<2"])
    target["installed"] = {"numpy": "2.4.6"}
    with pytest.raises(ValueError, match="versions CPU figees"):
        installer.export_installed("fixture", tmp_path / "wheels", target, distribution=lambda _: dist)


def test_rejects_editable_path_injection(tmp_path):
    dist = distribution(tmp_path, "fixture", extra={"editable.pth": b"/outside/source\n"})
    with pytest.raises(ValueError, match=".pth"):
        installer.repack_distribution(dist, tmp_path / "wheels", runtime())


def test_existing_compatible_public_dependency_not_exported(tmp_path):
    dist = distribution(tmp_path, "fixture", requires=["numpy>=1"])
    target = runtime()
    target["installed"] = {"numpy": "2.4.6"}
    result = installer.export_installed("fixture", tmp_path / "wheels", target, distribution=lambda name: dist if name == "fixture" else pytest.fail("Public dependency must stay in target"))
    assert [item["name"] for item in result["packages"]] == ["fixture"]
