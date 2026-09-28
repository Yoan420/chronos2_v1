"""Atomic publication with bounded retries for Windows sharing violations."""
import errno
import hashlib
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4


def replace_retry(source, target, *, timeout=10.):
    source, target = Path(source), Path(target)
    deadline = time.monotonic() + timeout
    delay = .01
    while True:
        try:
            source.replace(target)
            return
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, .25)


def promote_directory_retry(source, target, *, timeout=10.):
    """Publish a sealed directory, refusing existing destinations on every try.

    Callers hold their publication lock and validate both workspace paths.
    Windows rename itself also refuses a destination created concurrently.
    """
    source, target = Path(source), Path(target)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("A real sealed source directory is required")
    deadline, delay = time.monotonic() + timeout, .01
    while True:
        if os.path.lexists(target):
            raise FileExistsError(f"Publication already exists: {target}")
        try:
            source.rename(target)
            return
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, .25)


def publish_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(value)
        replace_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish_verified_immutable_copy(source, target, digest):
    """Publish one immutable artifact, sharing storage on a compatible volume.

    Both paths are checked against the supplied binary digest. Existing files
    are never overwritten. Cross-volume copies and filesystems without hard
    links keep the same portable paths and verification contract. Callers must
    only use sealed artifacts: later writes through either link affect both.
    """
    source, target = Path(source), Path(target)
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Expected a lowercase SHA-256 digest")
    if _file_sha256(source) != digest:
        raise ValueError(f"Immutable source differs: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)

    def verify_target():
        if _file_sha256(target) != digest or _file_sha256(source) != digest:
            raise ValueError(f"Immutable copied artifact differs: {target}")

    if target.exists():
        verify_target()
        return "existing"
    try:
        os.link(source, target)
    except FileExistsError:
        verify_target()
        return "existing"
    except OSError as error:
        unsupported = {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EMLINK,
                       errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)}
        if error.errno not in unsupported and getattr(error, "winerror", None) not in (1, 17, 50, 1142):
            raise
    else:
        verify_target()
        return "linked"

    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if _file_sha256(temporary) != digest or _file_sha256(source) != digest:
            raise ValueError(f"Immutable source changed during copying: {source}")
        try:
            if os.name == "nt":
                # Unlike POSIX rename, Windows rename refuses an existing target.
                deadline = time.monotonic() + 10.
                while True:
                    try:
                        os.rename(temporary, target)
                        break
                    except PermissionError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(.05)
            else:
                # The temporary is on the target volume, so this also handles
                # an EXDEV failure of the original source-to-target link.
                try:
                    os.link(temporary, target)
                except OSError as error:
                    if isinstance(error, FileExistsError) or error.errno not in unsupported:
                        raise
                    # Filesystems without any hard-link support still get an
                    # exclusive creation. No receipt is published until verified.
                    with temporary.open("rb") as incoming, target.open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                        outgoing.flush()
                        os.fsync(outgoing.fileno())
        except FileExistsError:
            verify_target()
            return "existing"
        verify_target()
        return "copied"
    finally:
        temporary.unlink(missing_ok=True)
