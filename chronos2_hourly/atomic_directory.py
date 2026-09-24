"""Publish complete sibling directories without overwriting an existing one."""
from __future__ import annotations

import errno
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


class AtomicDirectoryPublishError(OSError):
    """The completed staging directory remains available for a later publication."""


def _rename_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        # Windows rename fails atomically if the destination already exists.
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        # POSIX rename can overwrite an empty directory. Linux's exclusive
        # rename closes that race rather than relying on a preflight exists().
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise OSError(errno.ENOTSUP, "Exclusive directory rename is unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))
        return
    raise OSError(errno.ENOTSUP, "Exclusive directory publication is unsupported on this platform")


def publish_directory_no_replace(source: str | Path, destination: str | Path, *,
                                 attempts: int = 7, retry_seconds: float = 0.25) -> Path:
    """Retry only Windows sharing/access locks, with at most ten seconds of waits.

    Each attempt renames the same completed staging directory. Existing
    destinations, including empty directories and links, are never replaced.
    On failure this function leaves all source bytes in place.
    """
    if (isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 8
            or isinstance(retry_seconds, bool) or not isinstance(retry_seconds, (int, float))
            or not math.isfinite(retry_seconds) or retry_seconds < 0
            or retry_seconds * attempts * (attempts - 1) / 2 > 10):
        raise ValueError("Invalid bounded directory-publication retry settings")
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if source == destination or source.parent.resolve() != destination.parent.resolve():
        raise ValueError("Atomic directory publication requires distinct sibling paths")
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"Completed staging directory is missing or redirected: {source}")
    for attempt in range(1, attempts + 1):
        if os.path.lexists(destination):
            raise FileExistsError(errno.EEXIST, "Refusing to replace an existing directory", str(destination))
        try:
            _rename_no_replace(source, destination)
            return destination
        except OSError as error:
            if os.path.lexists(destination):
                raise FileExistsError(errno.EEXIST, "Destination appeared during publication", str(destination)) from error
            if getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            if attempt == attempts:
                raise AtomicDirectoryPublishError(
                    f"Publication du dossier bloquee apres {attempts} essais : {destination}. "
                    f"Le dossier temporaire complet est conserve dans {source}. "
                    "Aucune destination existante n'a ete remplacee."
                ) from error
            time.sleep(retry_seconds * attempt)
    raise AssertionError("Unreachable directory-publication state")


class AtomicDirectoryStaging:
    """Clean incomplete construction; retain a completed stage if publishing fails."""

    def __init__(self, parent: str | Path, *, prefix: str):
        self.parent = Path(parent).resolve()
        self.prefix = prefix
        self.path: Path | None = None
        self.preserve = False

    def __enter__(self):
        self.parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix, dir=self.parent))
        return self

    def publish(self, destination: str | Path) -> Path:
        if self.path is None:
            raise RuntimeError("Staging context has not been entered")
        self.preserve = True
        result = publish_directory_no_replace(self.path, destination)
        self.preserve = False
        return result

    def __exit__(self, *_exc):
        if self.path is None or self.preserve or not self.path.exists():
            return
        if self.path.resolve().parent != self.parent:
            raise OSError("Staging cleanup path escapes its directory")
        try:
            shutil.rmtree(self.path)
        except OSError as error:
            # Cleanup must not hide the original construction error.
            try:
                print(f"AVERTISSEMENT : nettoyage temporaire incomplet dans {self.path}: {error}",
                      file=sys.stderr, flush=True)
            except OSError:
                pass


__all__ = ["AtomicDirectoryPublishError", "AtomicDirectoryStaging", "publish_directory_no_replace"]
