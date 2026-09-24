"""OS-held locks with conservative recovery of legacy JSON sentinels.

The OS releases the sidecar guard after a crash.  The JSON sentinel remains
visible to older runners, and can only be reclaimed after proving that its
owner is gone.  Corrupt, foreign-host and inaccessible ownership stays locked.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import socket
import tempfile
from typing import Iterator
import uuid

from filelock import FileLock, Timeout
import psutil


LOGGER = logging.getLogger(__name__)


def _blocked(path: Path, reason: str) -> ValueError:
    return ValueError(f"Verrou present : {path}. {reason}")


def _owner_is_proven_gone(path: Path, raw: bytes) -> bool:
    try:
        owner = json.loads(raw)
        if not isinstance(owner, dict):
            raise ValueError("objet JSON requis")
        pid = owner.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("PID invalide")
        hostname = owner.get("hostname")
        if hostname is not None and (
            not isinstance(hostname, str)
            or hostname.casefold() != socket.gethostname().casefold()
        ):
            raise ValueError("proprietaire sur un autre hote ou hote invalide")
        recorded_start = owner.get("process_create_time")
        if recorded_start is not None and (
            isinstance(recorded_start, bool)
            or not isinstance(recorded_start, (int, float))
            or not math.isfinite(recorded_start)
            or recorded_start <= 0
        ):
            raise ValueError("identite de processus invalide")
        created = owner.get("created_at")
        created_timestamp = None
        if created is not None:
            stamp = datetime.fromisoformat(created)
            if stamp.tzinfo is None:
                raise ValueError("date du verrou sans fuseau horaire")
            created_timestamp = stamp.timestamp()
    except (ValueError, TypeError, OverflowError, UnicodeDecodeError) as exc:
        raise _blocked(path, "Identite illisible ou invalide ; verrou conserve.") from exc

    try:
        actual_start = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return True
    except (psutil.AccessDenied, OSError) as exc:
        raise _blocked(path, "Impossible de verifier le proprietaire ; verrou conserve.") from exc
    # An older or equal process start never proves reuse.  In particular, a
    # live legacy {pid}-only sentinel is always respected.  Legacy wall-clock
    # timestamps get a one-second margin for platform timestamp precision.
    if recorded_start is not None and actual_start > recorded_start + 0.001:
        return True
    if recorded_start is None and created_timestamp is not None:
        return actual_start > created_timestamp + 1.0
    return False


def _publish_owner(path: Path, owner: dict) -> None:
    """Expose a complete JSON record with create-if-absent semantics.

    A hard link is atomic on the same filesystem and cannot replace a sentinel
    that a legacy runner creates concurrently.  A crash while writing the
    private temporary file therefore cannot leave an unreadable live lock.
    """
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(owner, stream, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise _blocked(path, "Un autre processus vient de prendre ce verrou.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _remove_own_sentinel(path: Path, token: str) -> None:
    try:
        owner = json.loads(path.read_bytes())
        if isinstance(owner, dict) and owner.get("owner_token") == token:
            path.unlink()
        else:
            LOGGER.warning("Verrou conserve car son proprietaire a change : %s", path)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, UnicodeDecodeError):
        LOGGER.warning("Verrou conserve car son identite ne peut plus etre verifiee : %s", path)


@contextmanager
def exclusive_process_lock(path: str | Path) -> Iterator[None]:
    """Acquire immediately or fail closed; recover only a demonstrably dead owner."""
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = FileLock(str(path) + ".guard", timeout=0)
    try:
        guard.acquire()
    except Timeout as exc:
        raise _blocked(path, "Un processus detient le verrou systeme.") from exc
    except OSError as exc:
        raise _blocked(path, "Le verrou systeme est inaccessible.") from exc
    try:
        try:
            previous = path.read_bytes()
        except FileNotFoundError:
            previous = None
        except OSError as exc:
            raise _blocked(path, "Identite inaccessible ; verrou conserve.") from exc
        if previous is not None:
            if not _owner_is_proven_gone(path, previous):
                raise _blocked(path, "Le processus proprietaire est encore actif.")
            if path.read_bytes() != previous:
                raise _blocked(path, "Le proprietaire a change pendant la verification.")
            path.unlink()
            LOGGER.warning("Verrou abandonne repris apres verification du processus : %s", path)

        token = uuid.uuid4().hex
        owner = {
            "schema_version": 1,
            "pid": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "process_create_time": psutil.Process().create_time(),
            "hostname": socket.gethostname(),
            "owner_token": token,
        }
        _publish_owner(path, owner)
        try:
            yield
        finally:
            _remove_own_sentinel(path, token)
    finally:
        guard.release()


__all__ = ["exclusive_process_lock"]
