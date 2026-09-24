"""Read-only process and experiment observations; never imports scientific code."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from datetime import datetime, timezone
from typing import Any

import psutil

from experiment_console.security import redact, redact_text
from .estimates import Estimator


MAX_JSON = 256 * 1024
MAX_LOG = 32 * 1024
CREATION_TOLERANCE = 0.05
KNOWN_DAYS = {"solar_wind_interaction_v1": "2026-09-22", "solar_wind_corrector_interaction_v1": "2026-09-22",
              "solar_wind_corrector_parallel_v1": "2026-09-22", "solar_wind_corrector_reuse_v1": "2026-09-22"}
TITLES = {"solar_wind_interaction_v1": "SolarWind — interaction Kalman", "solar_wind_v1": "SolarWind",
          "solar_wind_corrector_interaction_v1": "SolarWind — correcteur × plafond +80",
          "solar_wind_corrector_parallel_v1": "SolarWind — correcteur × plafond +80 · parallèle",
          "solar_wind_corrector_reuse_v1": "SolarWind — correcteur × plafond +80 · réemploi vérifié"}
INTERACTION_RESULTS = {
    "backtest.parquet", "baseline_controls.json", "experiment.json", "feature_audit.json",
    "forecast.parquet", "interaction_covariates.parquet", "metrics.json", "replay_audit.json", "report.html",
}
PARALLEL_CORRECTOR_RESULTS = {
    "experiment.json", "baseline_controls.json", "interaction.parquet", "feature_audit.json",
    "corrector_audit.json", "metrics.json", "report.html", "migration_audit.json",
    *(variant + "/" + name for variant in ("interaction_40", "cap_80", "interaction_80")
      for name in ("upstream_history.parquet", "upstream_forecast.parquet", "backtest.parquet",
                   "forecast.parquet", "replay_audit.json")),
}
REUSE_CORRECTOR_RESULTS = PARALLEL_CORRECTOR_RESULTS | {'reconstruction_audit.json'}


def _iso(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp, timezone.utc).isoformat()


def _timestamp(value: Any) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _key(process: dict) -> tuple[int, float]:
    return process["pid"], process["created"]


def _log_redact(text: str) -> str:
    text = redact_text(text)
    # Logs sometimes echo CLI strings, whereas shared redact() expects argv lists.
    def replace(match):
        masked = redact([match["flag"], match["value"]])
        return match["flag"] + match["space"] + str(masked[1])
    return re.sub(r'(?P<flag>--?[A-Za-z][A-Za-z0-9_-]*)(?P<space>[ \t]+)(?P<value>"[^"\r\n]*"|\'[^\'\r\n]*\'|[^\s]+)', replace, text)


class Collector:
    """Snapshot registered launches and project process trees without writing files."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.runs = self.root / "runs"
        self._lock = threading.RLock()
        self._artifacts: dict[str, tuple[Path, Path]] = {}
        self._seen: dict[str, dict[int, float]] = {}
        self._cpu_previous: dict[tuple[int, float], float] = {}
        self._sample_time: float | None = None
        self._estimator = Estimator(self.root)
        self._sample_launches = None
        self._tail_cache = {}

    def _safe(self, path: str | Path, *, within: Path | None = None) -> Path | None:
        """Reject escapes and all redirected components, even internal symlinks."""
        try:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            candidate = Path(os.path.abspath(candidate))
            boundary = within or self.runs
            candidate.relative_to(boundary)
            candidate.relative_to(self.runs)
            # Check every component on every call, without resolving all ancestor
            # chains repeatedly. lstat never follows the component's redirection.
            parts = [self.runs]
            for segment in candidate.relative_to(self.runs).parts:
                parts.append(parts[-1] / segment)
            for part in parts:
                try:
                    info = part.lstat()
                except FileNotFoundError:
                    continue
                if (stat.S_ISLNK(info.st_mode)
                        or getattr(info, "st_file_attributes", 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
                    return None
            # Keep a final independent resolution check (including nonexistent
            # leaves) and do not cache trusted paths across observations.
            if candidate.resolve() != candidate:
                return None
            return candidate
        except (OSError, ValueError, RuntimeError):
            return None

    def _json(self, path: Path) -> dict | None:
        safe = self._safe(path)
        try:
            if safe is None or not safe.is_file() or safe.stat().st_size > MAX_JSON:
                return None
            with safe.open("rb") as handle:
                raw = handle.read(MAX_JSON + 1)
            if len(raw) > MAX_JSON:
                return None
            data = json.loads(raw.decode("utf-8-sig"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError, UnicodeError):
            return None

    def _tail(self, value: Any) -> tuple[str, float | None]:
        safe = self._safe(value) if isinstance(value, (str, Path)) else None
        try:
            if safe is None or not safe.is_file():
                return "", None
            info = safe.stat()
            signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            cached = self._tail_cache.get(safe)
            if cached is not None and cached[0] == signature:
                return cached[1]
            with safe.open("rb") as handle:
                start = max(0, info.st_size - MAX_LOG)
                handle.seek(start)
                raw = handle.read(MAX_LOG)
            # A partial first line may start in a secret value: discard it.
            if start:
                raw = raw.partition(b"\n")[2]
                # A bounded tail can start inside a multiline private key.
                ending = re.search(rb"-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----", raw)
                beginning = re.search(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----", raw)
                if ending and (not beginning or beginning.start() > ending.start()):
                    raw = b"[MASQUE]" + raw[ending.end():]
            text = _log_redact(raw.decode("utf-8", errors="replace"))
            result = ("\n".join(text.splitlines()[-120:]), info.st_mtime)
            after = safe.stat()
            after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            if signature == after_signature:
                if len(self._tail_cache) >= 128:
                    self._tail_cache.clear()
                self._tail_cache[safe] = (signature, result)
            return result
        except OSError:
            return "", None

    def _processes(self) -> tuple[dict[int, dict], list[str]]:
        processes, warnings = {}, []
        try:
            # On Windows, Process.ppid() enumerates the entire parent map each
            # time. Use the same bulk adapter that psutil.children() uses once,
            # not once for each of the hundreds of unrelated OS processes.
            parent_reader = getattr(psutil, "_ppid_map", None)
            parents = None
            if callable(parent_reader):
                try:
                    parents = parent_reader()
                except (psutil.Error, OSError):
                    pass
            inventory = {}
            attributes = ["pid", "name"] if parents is not None else ["pid", "name", "ppid"]
            for process in psutil.process_iter(attributes, ad_value=None):
                try:
                    info = process.info
                    inventory[info["pid"]] = {
                        "name": info.get("name") or "",
                        "ppid": parents.get(info["pid"]) if parents is not None else info.get("ppid"),
                    }
                except (psutil.Error, OSError, ValueError):
                    continue
            selected = {pid for pid, info in inventory.items() if info["name"].lower().startswith("python")}
            # Keep already identified orphaned non-Python descendants too.
            selected.update(pid for seen in self._seen.values() for pid in seen if pid in inventory)
            # Recorded identities also cover orphan helpers after the monitor
            # itself has restarted (before its in-memory history is populated).
            launches = self._sample_launches if self._sample_launches is not None else self._launches()
            for _, metadata in launches:
                recorded = [
                    (metadata.get("python_pid"), metadata.get("python_created_utc")),
                    (metadata.get("venv_wrapper_pid"), metadata.get("venv_wrapper_created_utc")),
                ]
                workers = metadata.get("verified_workers", [])
                if isinstance(workers, list):
                    recorded.extend((worker.get("pid"), worker.get("created_utc")) for worker in workers if isinstance(worker, dict))
                for pid, created in recorded:
                    try:
                        if _timestamp(created) is not None and int(pid) in inventory:
                            selected.add(int(pid))
                    except (TypeError, ValueError):
                        pass
            while True:
                added = {pid for pid, info in inventory.items() if pid not in selected and info["ppid"] in selected}
                if not added:
                    break
                selected.update(added)
            details = ["create_time", "cmdline", "cpu_times", "memory_info", "name"]
            for pid in selected:
                try:
                    # New Process objects avoid process_iter's cached identity.
                    # Recheck after reading details to reject PID reuse races.
                    info = psutil.Process(pid).as_dict(details, ad_value=None)
                    created = info.get("create_time")
                    if created is None or abs(psutil.Process(pid).create_time() - created) > CREATION_TOLERANCE:
                        continue
                    times = info.get("cpu_times")
                    memory = info.get("memory_info")
                    processes[pid] = {
                        "pid": pid, "ppid": inventory[pid]["ppid"], "created": float(created),
                        "argv": info.get("cmdline") or [], "name": info.get("name") or "",
                        "cpu": (times.user + times.system) if times else None,
                        "memory": memory.rss if memory else None,
                    }
                except (psutil.Error, OSError, ValueError):
                    continue
        except (psutil.Error, OSError) as exc:
            warnings.append("Inventaire des processus indisponible : " + type(exc).__name__)
        return processes, warnings

    @staticmethod
    def _descendants(seeds: set[int], processes: dict[int, dict]) -> set[int]:
        found = set(seeds)
        while True:
            added = {
                pid for pid, row in processes.items()
                if pid not in found and row["ppid"] in found
                and row["created"] + CREATION_TOLERANCE >= processes[row["ppid"]]["created"]
            }
            if not added:
                return found
            found.update(added)

    @staticmethod
    def _matches(processes: dict[int, dict], pid: Any, created: Any) -> int | None:
        try:
            row = processes.get(int(pid))
            stamp = _timestamp(created)
            if row and stamp is not None and abs(row["created"] - stamp) <= CREATION_TOLERANCE:
                return row["pid"]
        except (ValueError, TypeError):
            pass
        return None

    def _launches(self) -> list[tuple[Path, dict]]:
        found = []
        experiments = self._safe(self.runs / "experiments")
        if experiments is None or not experiments.is_dir():
            return found
        try:
            for engine in sorted(experiments.iterdir()):
                logs = self._safe(engine / "launcher_logs")
                if logs is None or not logs.is_dir():
                    continue
                for path in sorted(logs.glob("*.launch.json")):
                    metadata = self._json(path)
                    if metadata is not None:
                        found.append((path, metadata))
        except OSError:
            pass
        found.sort(key=lambda item: _timestamp(item[1].get("started_utc") or item[1].get("started_at_utc")) or 0)
        return found[-20:]

    def _report(self, path: Path, directory: Path) -> str | None:
        safe = self._safe(path, within=directory)
        if safe is None or safe.suffix.lower() != ".html" or not safe.is_file():
            return None
        artifact_id = hashlib.sha256(str(safe).encode()).hexdigest()[:32]
        self._artifacts[artifact_id] = (safe, directory)
        return "/artifact?id=" + artifact_id

    def artifact(self, artifact_id: str) -> Path | None:
        with self._lock:
            item = self._artifacts.get(artifact_id)
            if item is None:
                return None
            safe = self._safe(item[0], within=item[1])
            return safe if safe is not None and safe.is_file() and safe.suffix.lower() == ".html" else None

    def _zones(self, engine: str, metadata: dict) -> tuple[list[dict], list[str]]:
        zones, warnings = [], []
        identities = metadata.get("run_identities", {})
        if not isinstance(identities, dict) or not identities:
            return [], ["Métadonnées anciennes : identités des résultats non enregistrées ; fin non attribuée à ce lancement."]
        day = metadata.get("delivery_day")
        arguments = metadata.get("arguments", [])
        if day is None and isinstance(arguments, list):
            for index, argument in enumerate(arguments):
                if argument == "--delivery-day" and index + 1 < len(arguments):
                    day = arguments[index + 1]
                elif isinstance(argument, str) and argument.startswith("--delivery-day="):
                    day = argument.split("=", 1)[1]
        day = day or KNOWN_DAYS.get(engine)
        if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            return [], ["Date de livraison non enregistrée : résultats non associés."]
        for zone, identity in identities.items():
            if not isinstance(zone, str) or not re.fullmatch(r"[A-Z]{2}", zone) or not isinstance(identity, str) or not re.fullmatch(r"[a-fA-F0-9]{16,64}", identity):
                warnings.append("Identité de zone invalide dans les métadonnées.")
                continue
            directory = self.runs / "experiments" / engine / day / zone.lower() / identity
            state = self._json(directory / "status.json") or {}
            status = str(state.get("status", "unknown")).lower()
            if state and (state.get("identity") != identity or state.get("zone") != zone):
                warnings.append(zone + " : identité du statut incohérente.")
                state, status = {}, "unknown"
            if status not in {"running", "complete", "failed"}:
                status = "unknown"
            if status == "complete":
                receipt = self._json(directory / "completion.json") or {}
                inventory = receipt.get("files", {})
                valid = isinstance(inventory, dict) and bool(inventory) and receipt.get("identity") == identity
                if engine == "solar_wind_interaction_v1":
                    valid = valid and set(inventory) == INTERACTION_RESULTS
                elif engine in ("solar_wind_corrector_parallel_v1", "solar_wind_corrector_reuse_v1"):
                    expected = REUSE_CORRECTOR_RESULTS if engine == "solar_wind_corrector_reuse_v1" else PARALLEL_CORRECTOR_RESULTS
                    valid = (valid and set(inventory) == expected
                             and receipt.get("annual_complete") is True
                             and state.get("evaluation_days") == 365
                             and state.get("evaluation_hours") == 8760
                             and state.get("future_forecast_hours") == 24)
                if valid:
                    for name, digest in inventory.items():
                        candidate = self._safe(directory / str(name), within=directory)
                        if candidate is None or not candidate.is_file() or not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
                            valid = False
                            break
                if state.get("annual_complete") is not True or not valid:
                    warnings.append(zone + " : fin annoncée mais reçu ou fichiers incomplets/incohérents.")
                    status = "unknown"
            error = state.get("error")
            if error:
                warnings.append(zone + " : " + redact_text(str(error)))
            # This is an exact counter within one stage, not a whole-job ETA
            # or percentage: CatBoost fits and Kalman replays have different costs.
            phase_progress = None
            progress = state.get("progress")
            if isinstance(progress, dict):
                completed, total = progress.get("completed"), progress.get("total")
                if (type(completed) is int and type(total) is int
                        and 0 <= completed <= total and 0 < total <= 1000000):
                    phase_progress = {"completed": completed, "total": total,
                        "unit": redact_text(str(progress.get("unit", "unités")))[:80]}
            zones.append({
                "zone": zone, "status": status, "phase": state.get("phase", "unknown"),
                "updated_at": state.get("updated_utc"), "report_url": self._report(directory / "report.html", directory),
                "phase_progress": phase_progress,
            })
        return zones, warnings

    @staticmethod
    def _role(row: dict, owner: int | None, wrapper: int | None) -> str:
        command = " ".join(row["argv"]).lower()
        if row["pid"] == wrapper:
            return "wrapper"
        if "resource_tracker" in command:
            return "resource_tracker"
        if "loky" in command or "multiprocessing" in command:
            return "worker"
        return "main" if row["pid"] == owner else "child"

    def _process_rows(self, ids: set[int], all_processes: dict[int, dict], cpu: dict, owner=None, wrapper=None) -> list[dict]:
        return [{
            "pid": pid, "ppid": row["ppid"], "created_at": _iso(row["created"]),
            "role": self._role(row, owner, wrapper), "cpu_percent": cpu.get(_key(row)),
            "memory_mb": round(row["memory"] / 1024 ** 2, 1) if row["memory"] is not None else None,
            "command": " ".join(redact(row["argv"])),
        } for pid in sorted(ids) if (row := all_processes.get(pid)) is not None]

    @staticmethod
    def _resources(rows: list[dict]) -> tuple[float | None, float | None]:
        cpu = [row["cpu_percent"] for row in rows]
        memory = [row["memory_mb"] for row in rows]
        return (
            round(min(100, sum(cpu)), 2) if cpu and all(v is not None for v in cpu) else None,
            round(sum(memory), 1) if memory and all(v is not None for v in memory) else None,
        )

    def _registered(self, path: Path, meta: dict, processes: dict[int, dict], cpu: dict, now: float) -> tuple[dict, set[int]]:
        engine = path.parent.parent.name
        job_id = hashlib.sha256(str(path).encode()).hexdigest()[:24]
        owner = self._matches(processes, meta.get("python_pid"), meta.get("python_created_utc"))
        wrapper = self._matches(processes, meta.get("venv_wrapper_pid"), meta.get("venv_wrapper_created_utc"))
        seeds = {pid for pid in (owner, wrapper) if pid is not None}
        verified = meta.get("verified_workers", [])
        if isinstance(verified, list):
            for worker in verified:
                if isinstance(worker, dict):
                    match = self._matches(processes, worker.get("pid"), worker.get("created_utc"))
                    if match is not None:
                        seeds.add(match)
        for pid, created in self._seen.get(job_id, {}).items():
            if pid in processes and abs(processes[pid]["created"] - created) <= CREATION_TOLERANCE:
                seeds.add(pid)
        ids = self._descendants(seeds, processes)
        self._seen[job_id] = {pid: processes[pid]["created"] for pid in ids}
        zones, warnings = self._zones(engine, meta)
        identities = meta.get("run_identities", {})
        complete = bool(zones) and isinstance(identities, dict) and len(zones) == len(identities) and all(row["status"] == "complete" for row in zones)
        failed = any(row["status"] == "failed" for row in zones)
        if complete:
            status = "complete"
            if ids:
                warnings.append("Résultats terminés ; processus encore présents pendant leur sortie.")
        elif failed:
            status = "failed"
        elif ids:
            status = "running"
        elif meta.get("python_created_utc"):
            status = "absent"
            warnings.append("Processus identifié absent ; cela ne prouve ni un échec ni une autorisation de reprise.")
        else:
            status = "unknown"
            warnings.append("PID historique sans date de création : présence non attribuable de façon sûre.")
        if ids and owner is None:
            warnings.append("Processus principal absent ; descendants identifiés encore présents.")
        rows = self._process_rows(ids, processes, cpu, owner, wrapper)
        cpu_total, memory_total = self._resources(rows)
        # The UI only shows active processes. Do not repeatedly read/redact
        # historical logs for launches that are not displayed.
        stdout, stdout_time = self._tail(meta.get("stdout")) if ids else ("", None)
        stderr, stderr_time = self._tail(meta.get("stderr")) if ids else ("", None)
        log_time = max((stamp for stamp in (stdout_time, stderr_time) if stamp is not None), default=None)
        started = meta.get("started_utc") or meta.get("started_at_utc")
        started_stamp = _timestamp(started)
        end_stamp = max((_timestamp(zone.get("updated_at")) or 0 for zone in zones), default=0) if complete else now
        elapsed = max(0, round(end_stamp - started_stamp)) if started_stamp is not None and end_stamp else None
        done = sum(zone["status"] == "complete" for zone in zones)
        active = next((zone for zone in zones if zone["status"] == "running"), None)
        if complete:
            phase, label = "complete", "Tous les pays terminés ; inventaire présent (SHA non revérifiés en continu)."
        elif active:
            phase = active["zone"] + " · " + active["phase"]
            label = f"{done}/{len(zones)} pays terminés — progression du lot indéterminée"
            if engine == "solar_wind_interaction_v1" and active["phase"] == "kalman_replay":
                label += " (366 recalibrations ; caches publiés après les lots)"
            if active.get("phase_progress"):
                counter = active["phase_progress"]
                label = (f"{active['zone']} · étape : {counter['completed']}/{counter['total']} {counter['unit']}"
                         f" — {done}/{len(zones)} pays terminés")
        else:
            phase, label = "unknown", "Progression non disponible"
        planning_estimate = None
        estimate = meta.get("planning_estimate")
        if isinstance(estimate, dict):
            duration = estimate.get("total_seconds")
            if (type(duration) in (int, float) and math.isfinite(duration)
                    and 30 <= duration <= 365 * 86400 and isinstance(estimate.get("basis"), str)):
                planning_estimate = {"total_seconds": float(duration),
                    "basis": redact_text(estimate["basis"])[:1500]}
        return {
            "id": job_id, "title": TITLES.get(engine, engine), "subtitle": path.stem.replace(".launch", ""),
            "kind": "registered", "status": status, "started_at": started,
            "elapsed_seconds": elapsed,
            "phase": phase, "progress": {"completed": done, "total": len(zones) or None, "percent": 100 if complete else None, "label": label},
            "planning_estimate": planning_estimate,
            "cpu_percent": cpu_total, "memory_mb": memory_total, "processes": rows, "zones": zones,
            "stdout_tail": stdout, "stderr_tail": stderr, "logs_updated_at": _iso(log_time) if log_time is not None else None,
            "warnings": warnings,
        }, ids

    def _project_script(self, row: dict) -> str | None:
        if not row["name"].lower().startswith("python"):
            return None
        for argument in row["argv"][1:]:
            candidate = Path(argument.strip('"'))
            if not re.fullmatch(r"run_[A-Za-z0-9_]+\.py", candidate.name):
                continue
            if candidate.is_absolute() and candidate.parent.resolve() == self.root:
                return candidate.name
            # A relative script requires a verified process cwd; no guesses.
            if not candidate.is_absolute() and len(candidate.parts) == 1:
                try:
                    process = psutil.Process(row["pid"])
                    if abs(process.create_time() - row["created"]) <= CREATION_TOLERANCE and Path(process.cwd()).resolve() == self.root:
                        return candidate.name
                except (psutil.Error, OSError):
                    pass
        return None

    def snapshot(self) -> dict:
        with self._lock:
            now, monotonic = time.time(), time.monotonic()
            launches = self._launches()
            self._sample_launches = launches
            try:
                processes, warnings = self._processes()
            finally:
                self._sample_launches = None
            delta = monotonic - self._sample_time if self._sample_time is not None else None
            cores = max(1, psutil.cpu_count() or 1)
            cpu = {}
            for row in processes.values():
                previous = self._cpu_previous.get(_key(row))
                if delta is not None and delta > 0 and row["cpu"] is not None and previous is not None:
                    cpu[_key(row)] = round(max(0, row["cpu"] - previous) / delta / cores * 100, 2)
            self._cpu_previous = {_key(row): row["cpu"] for row in processes.values() if row["cpu"] is not None}
            self._sample_time = monotonic
            self._artifacts = {}
            jobs, claimed = [], set()
            for path, metadata in launches:
                job, ids = self._registered(path, metadata, processes, cpu, now)
                jobs.append(job)
                claimed.update(ids)
            candidates = {pid: script for pid, row in processes.items() if pid not in claimed and (script := self._project_script(row))}
            for pid in sorted(candidates, key=lambda item: processes[item]["created"]):
                if pid in claimed:
                    continue
                ancestors, parent = set(), processes[pid]["ppid"]
                while parent in processes and parent not in ancestors:
                    ancestors.add(parent)
                    parent = processes[parent]["ppid"]
                if ancestors.intersection(candidates):
                    continue
                ids = self._descendants({pid}, processes) - claimed
                claimed.update(ids)
                rows = self._process_rows(ids, processes, cpu, pid)
                cpu_total, memory_total = self._resources(rows)
                jobs.append({
                    "id": f"detected-{pid}-{processes[pid]['created']}", "title": candidates[pid],
                    "subtitle": "Autre calcul du projet détecté — origine non attribuée", "kind": "detected", "status": "running",
                    "started_at": _iso(processes[pid]["created"]), "elapsed_seconds": max(0, round(now - processes[pid]["created"])),
                    "phase": "Processus actif", "progress": {"completed": None, "total": None, "percent": None, "label": "Progression non instrumentée"},
                    "cpu_percent": cpu_total, "memory_mb": memory_total, "processes": rows, "zones": [],
                    "stdout_tail": "", "stderr_tail": "", "logs_updated_at": None, "warnings": [],
                })
            jobs.sort(key=lambda job: (job["status"] != "running", -( _timestamp(job.get("started_at")) or 0)))
            for job in jobs:
                job["active"] = bool(job["processes"])
                job["eta"] = self._estimator.estimate(job, now) if job["active"] else None
            return redact({
                "app": "nyx-process-monitor", "updated_at": _iso(now), "project_root": str(self.root),
                "refresh_seconds": 1, "jobs": jobs, "warnings": warnings,
            })
