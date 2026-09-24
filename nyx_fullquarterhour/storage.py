"""Confine artifacts and verify explicit stage identities on resume."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

from nyx_intrahour.runner import clean, write_json as _write_json

NAMESPACE = Path("runs/experiments/nyx_fullquarterhour_v1")


def safe_path(root: Path, path: Path) -> Path:
    root = root.resolve()
    namespace = root / NAMESPACE
    path = path if path.is_absolute() else root / path
    if namespace.resolve() != namespace or path.resolve() != path.absolute() or not path.resolve().is_relative_to(namespace):
        raise ValueError("Full-chain artifacts must remain in their isolated namespace without links.")
    for current in (path, *path.parents):
        if current == root:
            break
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
            raise ValueError("Artifact path traverses a reparse point.")
    return path


def write_json(path: Path, value, *, root: Path):
    safe_path(root, path)
    safe_path(root, path.with_name(path.name + ".tmp"))
    _write_json(path, value)


def identity(value) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, allow_nan=False).encode()).hexdigest()
