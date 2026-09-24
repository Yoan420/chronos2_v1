"""Private XGBoost dependency; never install into the operational environment."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import subprocess
import sys
import tempfile

from chronos2_hourly.process_lock import exclusive_process_lock

VERSION = "3.2.0"
RELATIVE_TARGET = Path("runs/experiments/nyx_scarcity_v1/runtime/xgboost_3_2_0")
LOGGER = logging.getLogger(__name__)


def target_path(root: Path | None = None) -> Path:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    target = root / RELATIVE_TARGET
    if target.resolve() != target:
        raise ValueError("Private XGBoost target must not redirect through a symlink/junction.")
    return target


def ensure_runtime(root: Path | None = None):
    target = target_path(root)
    if not (target / "xgboost/__init__.py").is_file():
        raise ValueError("Private XGBoost missing. Run ScarcityVariants.ps1 -Action Install first.")
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    import xgboost
    if xgboost.__version__ != VERSION or not Path(xgboost.__file__).resolve().is_relative_to(target):
        raise ValueError("Expected the pinned private XGBoost 3.2.0, not a shared environment package.")
    return xgboost


def runtime_seals(root: Path | None = None) -> dict:
    module = ensure_runtime(root)
    target = target_path(root)
    names = sorted(p for p in (target / "xgboost").rglob("*")
                   if p.is_file() and p.suffix in {".py", ".dll", ".json"})
    hashes = {}
    for p in names:
        with p.open("rb") as stream:
            hashes[p.relative_to(target).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"xgboost_version": module.__version__, "target": str(target), "files": hashes}


def _validate_private_install(target: Path, *, root: Path) -> None:
    """Import in a fresh process, never pin a staging DLL/module in this process."""
    if not (target / "xgboost/__init__.py").is_file():
        raise ValueError(f"Private XGBoost package is incomplete: {target}")
    script = (
        "import sys; from pathlib import Path; "
        "target=Path(sys.argv[1]).resolve(); sys.path.insert(0,str(target)); "
        "import xgboost; "
        "assert xgboost.__version__ == sys.argv[2], 'Incorrect XGBoost version'; "
        "assert Path(xgboost.__file__).resolve().is_relative_to(target), "
        "'XGBoost was imported outside the private target'"
    )
    subprocess.run([sys.executable, "-I", "-c", script, str(target), VERSION],
                   cwd=root, shell=False, check=True)


def install_runtime(root: Path) -> Path:
    """Publish a validated sibling stage atomically; never overwrite an install.

    Failed stages are deliberately retained for inspection. A retry uses a new
    stage, so an interrupted pip invocation cannot poison the public target.
    The process-held lock excludes concurrent installers and recovers crashes.
    """
    root = root.resolve()
    target = target_path(root)
    with exclusive_process_lock(target.parent / f".{target.name}.install.lock"):
        # Re-resolve after acquiring the lock, rejecting redirected namespaces.
        target = target_path(root)
        if target.exists():
            try:
                _validate_private_install(target, root=root)
            except (ValueError, subprocess.CalledProcessError) as exc:
                raise ValueError(
                    f"Existing private XGBoost installation is invalid and was preserved: {target}. "
                    "Inspect it and move it aside manually before retrying Install; "
                    "no existing directory is overwritten automatically."
                ) from exc
            return target
        requirement = root / "config/nyx_scarcity_requirements.txt"
        if not requirement.is_file():
            raise ValueError(f"Pinned XGBoost requirements missing: {requirement}")
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage_", dir=target.parent))
        try:
            command = [sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:",
                       "--require-hashes", "--disable-pip-version-check", "--timeout", "45", "--retries", "2",
                       "--target", str(stage), "-r", str(requirement)]
            subprocess.run(command, cwd=root, shell=False, check=True)
            _validate_private_install(stage, root=root)
            if target_path(root) != target or stage.resolve() != stage or target.exists():
                raise ValueError("Private runtime destination changed during installation; refusing publication.")
            # On Windows rename cannot replace an existing destination. The
            # shared installer lock also prevents concurrent normal publishers.
            stage.rename(target)
        except BaseException as exc:
            LOGGER.error("Private installation not published. Stage preserved: %s. "
                         "Inspect this directory; retry Install to use a fresh stage.", stage)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ValueError(
                f"Private XGBoost installation failed; stage preserved at {stage}. "
                "Retry Install to start a fresh stage; existing installations were not replaced."
            ) from exc
    return target
