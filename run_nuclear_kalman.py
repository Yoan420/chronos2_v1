"""Run the nuclear Kalman pipeline, then assemble one CWE Model / Storm report."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence
import uuid
import webbrowser

import psutil

from run_model_storm_report import delivery_date, resolve_project_path


PROJECT_ROOT = Path(__file__).resolve().parent
ZONES = ("BE", "DE", "FR", "NL")
INTERRUPTED_CODES = {130, -2, 3221225786, -1073741510}


@dataclass(frozen=True)
class RunStep:
    name: str
    zone: str | None
    command: tuple[str, ...]


@dataclass(frozen=True)
class StepOutcome:
    returncode: int
    completion: dict[str, Any] | None = None
    error: str | None = None
    child_pid: int | None = None
    child_status: str | None = None


class ChildCleanupError(RuntimeError):
    """The batch must stop if an owned child could not be stopped reliably."""


def _console_print(value: str, *, end: str = "\n", file=None, flush: bool = True) -> None:
    """Preserve logging even when a redirected Windows console only accepts cp1252."""
    stream = sys.stdout if file is None else file
    text = str(value) + end
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        # Only the display is escaped. The UTF-8 log always retains original bars,
        # names and model output, before attempting any console forwarding.
        stream.write(text.encode(encoding, errors="backslashreplace").decode(encoding))
    if flush:
        stream.flush()


def build_nuclear_kalman_plan(
    *, project_root: Path, python_executable: str, zones: Sequence[str], delivery_day: str,
    nuclear_config: Path, nuclear_root: Path, report_output: Path,
    device: str = "auto", threads: int = 4, workers: int = 4,
    with_attribution: bool = False, skip_observed_sync: bool = False,
) -> tuple[RunStep, ...]:
    """Pin the date and all paths once; no ordinary model is launched."""
    root = Path(project_root).resolve()
    selected = tuple(dict.fromkeys(zones))
    if not selected or any(zone not in ZONES for zone in selected):
        raise ValueError("Selection de pays invalide (BE, DE, FR, NL).")
    day = delivery_date(delivery_day)
    if device not in {"auto", "cpu", "cuda"} or not 1 <= threads <= 128 or not 1 <= workers <= 128:
        raise ValueError("Device ou threads/workers invalides.")
    config = resolve_project_path(Path(nuclear_config), project_root=root)
    source = (python_executable, "-u", str(root / "run_nuclear_forecast.py"),
              "--config", str(config), "--delivery-day", day, "--workers", str(workers))
    steps = [RunStep("sources", None, (*source, "--stage", "Sync", "--zones", *selected))]
    for zone in selected:
        command = (*source, "--stage", "Run", "--zones", zone, "--device", device,
                   "--threads", str(threads), "--skip-source-sync", "--report-variants", "kalman")
        if not with_attribution:
            command += ("--skip-attribution",)
        if skip_observed_sync:
            command += ("--skip-observed-sync",)
        steps.append(RunStep("nuclear_kalman", zone, command))
    report_command = (python_executable, "-u", str(root / "run_model_storm_report.py"), "--delivery-day", day)
    if skip_observed_sync:
        report_command += ("--skip-vps-sync",)
    report_command += ("--nuclear-root", str(resolve_project_path(Path(nuclear_root), project_root=root)),
                       "--output", str(resolve_project_path(Path(report_output), project_root=root)))
    steps.append(RunStep("CWE_Model_Storm", None, report_command))
    return tuple(steps)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Readers see either the previous complete status or the next one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            name = stream.name
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        from chronos2_hourly.nuclear_exports import _replace_with_retry
        _replace_with_retry(Path(name), path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def _stop_child_tree(process: subprocess.Popen, identity: psutil.Process | None) -> None:
    """Stop only this child's process tree, including the Windows venv interpreter.

    Process objects retain creation-time identities, so psutil's signals cannot
    accidentally target unrelated programs if Windows reuses one of their PIDs.
    """
    descendants = []
    if identity is not None:
        try:
            descendants = identity.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    owned = ([identity] if identity is not None else []) + list(reversed(descendants))
    for child in owned:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, survivors = psutil.wait_procs(owned, timeout=5)
    for child in survivors:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(survivors, timeout=5)
    if remaining:
        raise ChildCleanupError("Processus enfants encore actifs : " + ", ".join(str(child.pid) for child in remaining))
    process.wait(timeout=5)


def _run_logged(step: RunStep, *, project_root: Path, log_path: Path) -> StepOutcome:
    """Tee unbuffered child output to a persistent log; retain its completion receipt."""
    command_text = "Commande (argv, shell=False): " + json.dumps(step.command, ensure_ascii=False)
    completion = None
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(command_text + "\n")
        log.flush()
        _console_print(command_text)
        try:
            process = subprocess.Popen(list(step.command), cwd=project_root, shell=False,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1, env=env)
        except OSError as error:
            log.write(f"Lancement impossible : {error}\n")
            return StepOutcome(1, error=str(error))
        child_identity = None
        try:
            try:
                # Capture identity while the owned child is alive, before any interrupt.
                child_identity = psutil.Process(process.pid)
                child_identity.create_time()
            except psutil.NoSuchProcess:
                pass
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                _console_print(line, end="")
                if step.zone and line.lstrip().startswith("{"):
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict) and value.get("zone") == step.zone:
                            completion = value
                    except json.JSONDecodeError:
                        pass
            return StepOutcome(process.wait(), completion=completion,
                               child_pid=process.pid, child_status="exited")
        except BaseException as error:
            cleanup_error = None
            try:
                _stop_child_tree(process, child_identity)
            except (psutil.Error, OSError, subprocess.TimeoutExpired, ChildCleanupError) as failure:
                cleanup_error = failure
            detail = f"Suivi du processus {process.pid} interrompu : {type(error).__name__}: {error}. "
            detail += (f"Arret incomplet : {cleanup_error}" if cleanup_error is not None
                       else "Arbre enfant arrete et processus termine.")
            try:
                log.write(detail + "\n")
                log.flush()
            except OSError:
                pass
            if cleanup_error is not None:
                raise ChildCleanupError(detail) from error
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return StepOutcome(1, error=detail, child_pid=process.pid,
                               child_status="stopped_after_supervision_failure")
        finally:
            if process.stdout is not None:
                process.stdout.close()


def _verified_zone_report(outcome: StepOutcome, *, root: Path, zone: str, day: str) -> Path:
    """A stale file alone cannot turn a failed or incomplete child into success."""
    receipt = outcome.completion or {}
    if receipt.get("zone") != zone or receipt.get("status") != "complete":
        raise ValueError(f"{zone}: le processus n'a pas confirme la publication de ce lancement.")
    exports = receipt.get("exports", {})
    destination = root / "runs/exports" / day / zone.lower()
    expected = destination / "nuclear_kalman" / f"forecast_{zone.lower()}_{day}_nuclear_kalman.html"
    if Path(exports.get("kalman", "")).resolve() != expected.resolve():
        raise ValueError(f"{zone}: chemin du rapport nuclear_kalman inattendu.")
    manifest_path = destination / "current_nuclear_batch_manifest.json"
    if Path(exports.get("manifest", "")).resolve() != manifest_path.resolve():
        raise ValueError(f"{zone}: manifeste de publication inattendu.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("zone") != zone or manifest.get("delivery_day") != day:
        raise ValueError(f"{zone}: identite du manifeste incorrecte.")
    records = [record for record in manifest.get("exports", [])
               if record.get("variant") == "nuclear_kalman"]
    if len(records) != 1 or not records[0].get("files"):
        raise ValueError(f"{zone}: export nuclear_kalman absent du manifeste.")
    verified_paths = set()
    for item in records[0]["files"]:
        path = (destination / item["path"]).resolve()
        if not path.is_relative_to((destination / "nuclear_kalman").resolve()):
            raise ValueError(f"{zone}: chemin hors de l'export nuclear_kalman.")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{zone}: export absent ou vide : {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
            raise ValueError(f"{zone}: export modifie depuis la publication : {path}")
        verified_paths.add(path)
    if expected.resolve() not in verified_paths or not any(path.suffix == ".csv" for path in verified_paths):
        raise ValueError(f"{zone}: rapport HTML ou prevision CSV non certifie.")
    return expected


def execute_nuclear_kalman_plan(
    plan: Sequence[RunStep], *, project_root: Path, delivery_day: str,
    output: Path, log_directory: Path, run_id: str,
) -> int:
    """Continue after a country failure, preserve failures, assemble available results."""
    root = Path(project_root).resolve()
    status_path = log_directory / "status.json"
    latest_path = log_directory.parent / "latest_status.json"
    status: dict[str, Any] = {"schema_version": 1, "run_id": run_id, "delivery_day": delivery_day,
        "status": "running", "started_at_utc": _utc_now(), "finished_at_utc": None,
        "zones": [step.zone for step in plan if step.zone], "output": str(output),
        "status_file": str(status_path), "steps": []}
    for index, step in enumerate(plan):
        status["steps"].append({"name": step.name, "zone": step.zone, "status": "pending",
            "command": list(step.command), "log": str(log_directory / f"{index:02d}_{step.zone or step.name}.log")})

    def save() -> None:
        _write_json(status_path, status)
        _write_json(latest_path, status)

    save()
    _console_print(f"Statut du batch : {status_path}", flush=True)
    interrupted = False
    try:
        for index, step in enumerate(plan):
            item = status["steps"][index]
            if step.zone and status["steps"][0]["status"] != "complete":
                item.update(status="skipped", reason="La synchronisation commune a echoue.")
                save()
                continue
            _console_print(f"\n[Nuclear Kalman {index + 1}/{len(plan)}] {step.zone or 'CWE'} | {step.name}", flush=True)
            item.update(status="running", started_at_utc=_utc_now())
            save()
            try:
                outcome = _run_logged(step, project_root=root, log_path=Path(item["log"]))
                item["returncode"] = outcome.returncode
                if outcome.child_pid is not None:
                    item["child_pid"] = outcome.child_pid
                    item["child_status"] = outcome.child_status
                if outcome.returncode in INTERRUPTED_CODES:
                    raise KeyboardInterrupt
                if outcome.returncode != 0:
                    raise ValueError(outcome.error or f"Le processus s'est termine avec le code {outcome.returncode}.")
                if step.zone:
                    item["report"] = str(_verified_zone_report(outcome, root=root, zone=step.zone, day=delivery_day))
                elif step.name == "CWE_Model_Storm":
                    staged_report = Path(step.command[step.command.index("--output") + 1])
                    if not staged_report.is_file() or staged_report.stat().st_size == 0:
                        raise ValueError("Le processus n'a pas produit de nouveau rapport CWE Model / Storm.")
                    # The batch-specific path cannot point to yesterday's or a prior run's HTML.
                    from chronos2_hourly.nuclear_exports import _replace_with_retry
                    _replace_with_retry(staged_report, output)
                    item["report"] = str(output)
                item["status"] = "complete"
            except ChildCleanupError as error:
                item.update(status="failed", error=str(error), child_status="cleanup_failed")
                for pending in status["steps"][index + 1:]:
                    pending.update(status="skipped", reason="Arret de l'enfant non confirme; batch interrompu.")
                _console_print(f"[Nuclear Kalman] ARRET DU BATCH : {error}")
                break
            except (ValueError, OSError, KeyError, TypeError) as error:
                item.update(status="failed", error=str(error))
                _console_print(f"[Nuclear Kalman] ECHEC {step.zone or step.name} : {error}", flush=True)
            finally:
                item["finished_at_utc"] = _utc_now()
                save()
    except KeyboardInterrupt:
        interrupted = True
        for item in status["steps"]:
            if item["status"] in {"pending", "running"}:
                item["status"] = "interrupted" if item["status"] == "running" else "skipped"
                item["reason"] = "Interruption utilisateur."
    finally:
        success = all(item["status"] == "complete" for item in status["steps"])
        status["updated_zones"] = [item["zone"] for item in status["steps"]
                                   if item["zone"] and item["status"] == "complete"]
        status["failed_zones"] = [item["zone"] for item in status["steps"]
                                  if item["zone"] and item["status"] != "complete"]
        status.update(status="interrupted" if interrupted else "complete" if success else "failed",
                      finished_at_utc=_utc_now())
        save()
        # Remove only this launcher's own temporary artifact, never the previous public report.
        final_step = plan[-1]
        staged_report = Path(final_step.command[final_step.command.index("--output") + 1])
        staged_report.unlink(missing_ok=True)
    _console_print("\nResultat Nuclear Kalman", flush=True)
    for item in status["steps"]:
        _console_print(f"{item['zone'] or item['name']} : {item['status']}" +
              (f" | {item['error']}" if item.get("error") else ""), flush=True)
    if status["steps"][-1]["status"] == "complete":
        _console_print(f"Rapport des resultats disponibles : {output}", flush=True)
    _console_print("Rapports pays actualises : " + (", ".join(status["updated_zones"]) or "aucun"), flush=True)
    if status["failed_zones"]:
        _console_print("Pays non actualises : " + ", ".join(status["failed_zones"]) +
              ". Le HTML groupe utilise leurs resultats deja disponibles, s'ils existent.", flush=True)
    _console_print(f"Journal et statut du batch : {status_path}", flush=True)
    return 130 if interrupted else 0 if success else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones", nargs="+", choices=ZONES, default=list(ZONES))
    parser.add_argument("--delivery-day")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nuclear-config", type=Path, default=Path("config/nuclear_forecast.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-attribution", action="store_true")
    parser.add_argument("--skip-observed-sync", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = PROJECT_ROOT.resolve()
        day = delivery_date(args.delivery_day)
        zones = tuple(dict.fromkeys(args.zones))
        config = resolve_project_path(args.nuclear_config, project_root=root)
        output = resolve_project_path(args.output or Path("runs/reports/model_storm") /
                                      f"CWE_Model_Storm_{day}.html", project_root=root)
        if output.suffix.lower() != ".html":
            raise ValueError("Le rapport de sortie doit porter l'extension .html.")
        from run_nuclear_forecast import load_settings, check_lora_inactive, exclusive_lock
        settings = load_settings(config)
        if settings["project_root"] != root:
            raise ValueError("Le lanceur et la configuration nucleaire doivent partager la meme racine de projet.")
        for name in ("run_nuclear_forecast.py", "run_model_storm_report.py"):
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
        if set(zones).difference(settings.get("zone_configs", {})):
            raise ValueError("Un pays selectionne est absent de la configuration nucleaire.")
        check_lora_inactive(settings, list(zones))
        for zone in zones:
            zone_path = resolve_project_path(Path(settings["zone_configs"][zone]), project_root=root)
            if not zone_path.is_file():
                raise FileNotFoundError(zone_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
        staged_output = output.with_name(f".{output.stem}.{run_id}.html")
        plan = build_nuclear_kalman_plan(project_root=root, python_executable=sys.executable,
            zones=zones, delivery_day=day, nuclear_config=config, nuclear_root=settings["output_root"],
            report_output=staged_output, device=args.device, threads=args.threads, workers=args.workers,
            with_attribution=args.with_attribution, skip_observed_sync=args.skip_observed_sync)
        _console_print(f"Livraison : {day}\nPays : {', '.join(zones)}\nVariante : nuclear_kalman", flush=True)
        _console_print("Reutilisation des caches verifies; une synchronisation commune, puis les pays successifs.", flush=True)
        if args.dry_run:
            for step in plan:
                _console_print("Commande (argv, shell=False): " + json.dumps(step.command, ensure_ascii=False), flush=True)
            _console_print(f"Rapport final : {output}\nDryRun : aucun pipeline lance ni fichier cree.", flush=True)
            return 0
        # All selections for this delivery share the lock. A second launcher must not
        # race source synchronization, zone publication, or the final report.
        with exclusive_lock(settings["output_root"] / "_batch_locks" / f"nuclear_kalman_{day}.lock"):
            log_directory = root / "runs/logs/nuclear_kalman" / day / run_id
            output.parent.mkdir(parents=True, exist_ok=True)
            code = execute_nuclear_kalman_plan(plan, project_root=root, delivery_day=day,
                output=output, log_directory=log_directory, run_id=run_id)
        if code == 0 and not args.no_open:
            webbrowser.open(output.as_uri())
        return code
    except (ValueError, OSError, ImportError) as error:
        _console_print(f"[Nuclear Kalman] ECHEC : {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
