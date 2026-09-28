"""Transfer an authorized installed Saturn client into the isolated NYX environment.

The source interpreter is read-only. Repacked wheels stay in .venv-annual, which
is ignored by Git. No Saturn request is made during installation or validation.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import zipfile

try:
    from packaging.requirements import Requirement
    from packaging.tags import parse_tag, sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.tags import parse_tag, sys_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename


def runtime_info():
    return {
        "python": list(sys.version_info[:2]),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "platform": sys.platform,
        "tags": [str(tag) for tag in sys_tags()],
        "installed": {
            canonicalize_name(dist.metadata["Name"]): dist.version
            for dist in metadata.distributions() if dist.metadata["Name"]
        },
    }


def probe_client(requirement="tshistory_lite==0.5"):
    result = {"ok": False, "python": list(sys.version_info[:2]), "executable": sys.executable}
    try:
        required = Requirement(requirement)
        if canonicalize_name(required.name) != "tshistory-lite" or required.url:
            raise ValueError("La dependance doit nommer tshistory_lite avec une version.")
        result["version"] = metadata.version(required.name)
        if not required.specifier.contains(result["version"]):
            raise ValueError("Version du client Saturn incompatible avec la version demandee.")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            client = getattr(importlib.import_module("tshistory_lite"), "Client", None)
        if not callable(client) or not all(callable(getattr(client, name, None)) for name in ("get", "block_staircase")):
            raise ValueError("Client Saturn doit exposer Client.get et Client.block_staircase.")
        if sys.version_info[:2] != (3, 11):
            raise ValueError("Le client source doit fonctionner avec Python 3.11.")
        result["ok"] = True
    except Exception as exc:
        # Do not echo import exception text: an internal client can include URLs.
        result["error_type"] = type(exc).__name__
    return result


def validate_wheel(path, runtime):
    path = Path(path)
    _, _, _, wheel_tags = parse_wheel_filename(path.name)
    compatible = set(runtime["tags"])
    if not any(str(tag) in compatible for tag in wheel_tags):
        raise ValueError(f"Wheel incompatible avec Python 3.11 cible: {path.name}")
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name for name in members):
            raise ValueError("Chemin non autorise dans le wheel.")
        wheels = [name for name in members if name.endswith(".dist-info/WHEEL")]
        if len(wheels) != 1:
            raise ValueError("Metadonnees WHEEL manquantes ou ambigues.")
        embedded = [line[5:].strip() for line in archive.read(wheels[0]).decode().splitlines() if line.startswith("Tag: ")]
        if not embedded or not any(str(tag) in compatible for value in embedded for tag in parse_tag(value)):
            raise ValueError("Tags internes du wheel incompatibles avec le poste cible.")
    return path


def _digest(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def repack_distribution(dist, wheelhouse, runtime):
    """Reconstruct a standards-compliant wheel from its recorded installed files."""
    files = dist.files
    if not files:
        raise ValueError("Distribution sans RECORD; fournir le wheel Saturn autorise.")
    root = Path(dist.locate_file("")).resolve()
    wheel_entries = [entry for entry in files if str(entry).replace("\\", "/").endswith(".dist-info/WHEEL")]
    if len(wheel_entries) != 1:
        raise ValueError("Installation editable/conda sans WHEEL: fournir le wheel autorise.")
    wheel_entry = wheel_entries[0]
    wheel_text = Path(dist.locate_file(wheel_entry)).read_text(encoding="utf-8")
    tags = [line[5:].strip() for line in wheel_text.splitlines() if line.startswith("Tag: ")]
    matches = sorted(str(tag) for value in tags for tag in parse_tag(value) if str(tag) in runtime["tags"])
    if not matches:
        raise ValueError(f"Distribution source incompatible avec le poste cible: {dist.metadata['Name']}")
    name = re.sub(r"[-_.]+", "_", dist.metadata["Name"])
    version = dist.version.replace("-", "_")
    destination = Path(wheelhouse) / f"{name}-{version}-{matches[0]}.whl"
    info_dir = PurePosixPath(str(wheel_entry).replace("\\", "/")).parent
    record_name = str(info_dir / "RECORD")
    scripts = {entry.name for entry in dist.entry_points if entry.group in {"console_scripts", "gui_scripts"}}
    records = []
    content = {}
    for entry in files:
        member = str(entry).replace("\\", "/")
        member_path = PurePosixPath(member)
        if member_path.suffix == ".pyc" or "__pycache__" in member_path.parts:
            continue
        if member_path.name in {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json", "RECORD.jws", "RECORD.p7s"} and member_path.parent == info_dir:
            continue
        if member_path.suffix == ".pth":
            raise ValueError("Installation avec fichier .pth: utiliser le wheel autorise.")
        file_path = Path(dist.locate_file(entry)).resolve()
        if not file_path.is_relative_to(root) or ".." in member_path.parts or member_path.is_absolute():
            # pip-generated executable launchers are recreated from entry_points.
            basename = member_path.name
            launcher = basename.removesuffix(".exe").removesuffix("-script.py").removesuffix(".exe.manifest")
            if file_path.parent.name.lower() in {"scripts", "bin"} and launcher in scripts:
                continue
            raise ValueError(f"Fichier hors site-packages pour {name}; utiliser le wheel autorise.")
        data = file_path.read_bytes()
        if entry.hash:
            actual = base64.urlsafe_b64encode(hashlib.new(entry.hash.mode, data).digest()).rstrip(b"=").decode()
            if actual != entry.hash.value:
                raise ValueError(f"RECORD altere pour {name}: {member}")
        elif member_path.name not in {"METADATA", "WHEEL"}:
            raise ValueError(f"Fichier sans empreinte pour {name}: {member}")
        content[member] = data
        records.append([member, _digest(data), str(len(data))])
    if str(info_dir / "METADATA") not in content:
        raise ValueError("METADATA manquant dans la distribution source.")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(records + [[record_name, "", ""]])
    content[record_name] = output.getvalue().encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, data in sorted(content.items()):
            archive.writestr(member, data)
    os.replace(temporary, destination)
    validate_wheel(destination, runtime)
    return destination


def export_installed(requirement, wheelhouse, runtime, *, distribution=metadata.distribution):
    if list(sys.version_info[:2]) != [3, 11] or runtime["python"] != [3, 11]:
        raise ValueError("Le transfert exige Python 3.11 dans les deux environnements.")
    source = runtime_info()
    if any(source[key] != runtime[key] for key in ("implementation", "machine", "platform")):
        raise ValueError("Le transfert exige la meme plateforme et architecture Python.")
    pending = [(Requirement(requirement), True)]
    visited = set()
    exported = {}
    while pending:
        required, is_root = pending.pop()
        if required.url:
            raise ValueError("Dependance URL directe: utiliser un index ou wheel autorise.")
        key = canonicalize_name(required.name)
        target_version = runtime["installed"].get(key)
        if target_version and not is_root and not required.specifier.contains(target_version):
            raise ValueError(f"Dependance {required.name} incompatible avec les versions CPU figees.")
        if not is_root and target_version and not required.extras:
            continue
        dist = distribution(required.name)
        if not required.specifier.contains(dist.version):
            raise ValueError(f"Dependance absente/incompatible dans l'environnement source: {required.name}")
        visit = (key, tuple(sorted(required.extras)))
        if visit in visited:
            continue
        visited.add(visit)
        if key not in exported and (is_root or not target_version):
            path = repack_distribution(dist, wheelhouse, runtime)
            exported[key] = {"name": dist.metadata["Name"], "version": dist.version, "wheel": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for value in dist.requires or []:
            dependency = Requirement(value)
            if not dependency.marker or any(dependency.marker.evaluate({"extra": extra}) for extra in (set(required.extras) | {""})):
                pending.append((dependency, False))
    result = {"protocol": "nyx_local_saturn_transfer_v1", "packages": sorted(exported.values(), key=lambda item: item["name"])}
    Path(wheelhouse, "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "runtime", "constraints", "export", "check-wheel"))
    parser.add_argument("--requirement", default="tshistory_lite==0.5")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-runtime", type=Path)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "probe":
            result = probe_client(args.requirement)
            print(json.dumps(result))
            return 0
        if args.action == "runtime":
            args.output.write_text(json.dumps(runtime_info(), indent=2), encoding="utf-8")
        elif args.action == "constraints":
            args.output.write_text("\n".join(f"{name}=={version}" for name, version in sorted(runtime_info()["installed"].items()) if name != "tshistory-lite") + "\n", encoding="utf-8")
        elif args.action == "check-wheel":
            validate_wheel(args.wheel, runtime_info())
        else:
            if not probe_client(args.requirement)["ok"]:
                raise ValueError("Client Saturn Python 3.11 source absent ou incompatible (Client.get/block_staircase requis).")
            result = export_installed(args.requirement, args.output, json.loads(args.target_runtime.read_text(encoding="utf-8")))
            print(json.dumps(result))
        return 0
    except Exception as exc:
        print(f"Preparation Saturn echouee ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
