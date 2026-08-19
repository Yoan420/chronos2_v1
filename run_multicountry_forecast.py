#!/usr/bin/env python
"""Sequential multi-country launcher for immutable day-ahead forecasts.

The launcher deliberately reuses the same audited dispatcher and archive
validator as the Streamlit application.  It never calls a zone runner through
a shell, never overwrites an existing archive, and treats a missing/degraded
country HTML report as a batch failure even when the forecast itself was
successfully published.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Sequence
from zoneinfo import ZoneInfo

from chronos2_hourly.app_service import (
    APP_ZONES,
    ForecastProcess,
    ForecastSkip,
    ZoneStatus,
    build_dispatch_command,
    inspect_zone_statuses,
    launch_zone_forecast,
    read_log_tail,
    validate_existing_forecast_archive,
)
from chronos2_hourly.zone_live import canonical_zone


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = PROJECT_ROOT / "chronos2_hourly_live_zones.yaml"
DEFAULT_LOG_DIR = PROJECT_ROOT / "runs" / "live" / "_launcher_logs"
LOCAL_TIMEZONE = ZoneInfo("Europe/Paris")


class DetailedReportError(RuntimeError):
    """Raised when a published forecast lacks its complete detailed HTML."""


@dataclass(frozen=True)
class BatchZoneResult:
    zone: str
    delivery_day: str
    state: str
    return_code: int
    message: str
    archive_path: Path | None = None
    report_path: Path | None = None
    log_path: Path | None = None
    command: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.state in {"success", "skipped", "dry_run"}


@dataclass(frozen=True)
class BatchForecastResult:
    delivery_day: str
    zones: tuple[str, ...]
    results: tuple[BatchZoneResult, ...]

    @property
    def ok(self) -> bool:
        return len(self.results) == len(self.zones) and all(
            result.ok for result in self.results
        )


def normalize_zones(values: Sequence[str]) -> tuple[str, ...]:
    """Return supported canonical zones, rejecting duplicates and aliases."""

    if isinstance(values, (str, bytes)):
        values = [str(values)]
    result: list[str] = []
    for raw in values:
        text = str(raw).strip().upper()
        if not text:
            raise ValueError("La liste des pays contient une valeur vide.")
        try:
            code = canonical_zone(text)
        except Exception as exc:
            raise ValueError(f"Pays inconnu ou non supporte: {raw!r}.") from exc
        if code not in APP_ZONES:
            raise ValueError(
                f"{code} n'est pas disponible dans le launcher; "
                f"choix autorises: {', '.join(APP_ZONES)}."
            )
        if code in result:
            raise ValueError(f"Pays duplique dans la selection: {code}.")
        result.append(code)
    if not result:
        raise ValueError("Selectionnez au moins un pays.")
    return tuple(result)


def normalise_delivery_day(value: str | date | None) -> str:
    if value in (None, ""):
        return (datetime.now(LOCAL_TIMEZONE).date() + timedelta(days=1)).isoformat()
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ValueError("DeliveryDay doit respecter YYYY-MM-DD.") from exc


def find_detailed_html_report(archive: str | Path) -> Path:
    """Return the unique detailed HTML, rejecting any degraded fallback."""

    directory = Path(archive).expanduser().resolve()
    if not directory.is_dir():
        raise DetailedReportError(f"Archive publiee absente: {directory}")
    if (directory / "reporting_errors.json").exists():
        raise DetailedReportError(
            f"Le forecast est publie mais son reporting est degrade: {directory}"
        )
    for filename in ("run_manifest.json", "live_run_summary.json"):
        path = directory / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DetailedReportError(
                f"Impossible d'auditer le rapport via {path}."
            ) from exc
        if not isinstance(payload, dict):
            raise DetailedReportError(f"{path} doit contenir un objet JSON.")
        if payload.get("reporting_status") == "degraded" or "reporting_degraded" in str(
            payload.get("status", "")
        ):
            raise DetailedReportError(
                f"Le forecast est publie mais son rapport detaille a echoue: {directory}"
            )
    reports = sorted(path for path in directory.glob("*.html") if path.is_file())
    if len(reports) != 1:
        raise DetailedReportError(
            f"Une archive doit contenir exactement un rapport HTML; "
            f"trouves={len(reports)} dans {directory}."
        )
    report = reports[0]
    if report.stat().st_size <= 0:
        raise DetailedReportError(f"Le rapport HTML est vide: {report}")
    return report


def _status_map(
    registry_path: Path,
    zones: tuple[str, ...],
) -> dict[str, ZoneStatus]:
    statuses = inspect_zone_statuses(registry_path, zones=zones)
    return {status.code: status for status in statuses}


def _dry_run_result(
    *,
    status: ZoneStatus,
    zone: str,
    delivery_day: str,
    project_root: Path,
    registry_path: Path,
    python_executable: Path,
    device: str,
    threads: int,
    workers: int,
    local_files_only: bool,
) -> BatchZoneResult:
    command = tuple(
        build_dispatch_command(
            status,
            project_root=project_root,
            registry_path=registry_path,
            python_executable=python_executable,
            delivery_day=delivery_day,
            device=device,
            threads=threads,
            workers=workers,
            local_files_only=local_files_only,
        )
    )
    return BatchZoneResult(
        zone=zone,
        delivery_day=delivery_day,
        state="dry_run",
        return_code=0,
        message="Commande validee; aucune execution demandee.",
        command=command,
    )


def run_forecast_batch(
    *,
    zones: Sequence[str],
    delivery_day: str | date | None,
    project_root: str | Path = PROJECT_ROOT,
    registry_path: str | Path = DEFAULT_REGISTRY,
    python_executable: str | Path = sys.executable,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    device: str = "auto",
    threads: int = 4,
    workers: int = 4,
    local_files_only: bool = True,
    stop_on_error: bool = False,
    dry_run: bool = False,
) -> BatchForecastResult:
    """Run selected countries sequentially and audit every detailed report."""

    selected = normalize_zones(zones)
    delivery = normalise_delivery_day(delivery_day)
    root = Path(project_root).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    executable = Path(python_executable).expanduser().resolve()
    logs = Path(log_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Projet introuvable: {root}")
    if not registry.is_file():
        raise FileNotFoundError(f"Registre introuvable: {registry}")
    if not executable.is_file():
        raise FileNotFoundError(f"Python introuvable: {executable}")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device doit etre auto, cpu ou cuda.")
    if int(threads) < 1 or int(workers) < 1:
        raise ValueError("threads et workers doivent etre superieurs ou egaux a 1.")

    statuses = _status_map(registry, selected)
    results: list[BatchZoneResult] = []
    for zone in selected:
        status = statuses.get(zone)
        if status is None or not status.launchable:
            blockers = status.blockers if status is not None else ("audit absent",)
            results.append(
                BatchZoneResult(
                    zone=zone,
                    delivery_day=delivery,
                    state="failed",
                    return_code=2,
                    message="Preflight refuse: " + "; ".join(blockers),
                )
            )
            if stop_on_error:
                break
            continue
        if dry_run:
            results.append(
                _dry_run_result(
                    status=status,
                    zone=zone,
                    delivery_day=delivery,
                    project_root=root,
                    registry_path=registry,
                    python_executable=executable,
                    device=device,
                    threads=int(threads),
                    workers=int(workers),
                    local_files_only=local_files_only,
                )
            )
            continue

        print(f"\n[{zone}] lancement pour la livraison {delivery}...", flush=True)
        launched: ForecastProcess | ForecastSkip | None = None
        process_return_code: int | None = None
        try:
            launched = launch_zone_forecast(
                zone=zone,
                project_root=root,
                registry_path=registry,
                log_dir=logs,
                python_executable=executable,
                delivery_day=delivery,
                device=device,
                threads=int(threads),
                workers=int(workers),
                local_files_only=local_files_only,
            )
            if isinstance(launched, ForecastSkip):
                return_code = 0
                state = "skipped"
                message = "Archive deja publiee, validee et reutilisee."
                log_path = None
            else:
                print(f"[{zone}] processus demarre; log: {launched.log_path}", flush=True)
                return_code = int(launched.process.wait())
                process_return_code = return_code
                state = "success" if return_code == 0 else "failed"
                message = (
                    "Forecast et rapport publies."
                    if return_code == 0
                    else "Le runner a retourne un code non nul."
                )
                log_path = launched.log_path
            if return_code != 0:
                if log_path is not None:
                    tail = read_log_tail(log_path, max_chars=5_000).strip()
                    if tail:
                        message += "\n" + tail
                raise RuntimeError(message)
            archive = validate_existing_forecast_archive(
                status,
                project_root=root,
                delivery_day=delivery,
            )
            if archive is None:
                raise RuntimeError(
                    "Le runner a reussi mais aucune archive immutable n'a ete trouvee."
                )
            report = find_detailed_html_report(archive)
            summary_payload = json.loads(
                (archive / "live_run_summary.json").read_text(encoding="utf-8")
            )
            summary_status = str(summary_payload.get("status", "complete"))
            if summary_status != "complete":
                message += f" Statut de la couche Statistics: {summary_status}."
            results.append(
                BatchZoneResult(
                    zone=zone,
                    delivery_day=delivery,
                    state=state,
                    return_code=0,
                    message=message,
                    archive_path=archive,
                    report_path=report,
                    log_path=log_path,
                    command=(
                        tuple(launched.command)
                        if isinstance(launched, ForecastProcess)
                        else ()
                    ),
                )
            )
            print(f"[{zone}] rapport HTML: {report}", flush=True)
        except Exception as exc:
            log_path = (
                launched.log_path if isinstance(launched, ForecastProcess) else None
            )
            results.append(
                BatchZoneResult(
                    zone=zone,
                    delivery_day=delivery,
                    state="failed",
                    return_code=(
                        process_return_code
                        if process_return_code not in (None, 0)
                        else 1
                    ),
                    message=str(exc),
                    log_path=log_path,
                    command=(
                        tuple(launched.command)
                        if isinstance(launched, ForecastProcess)
                        else ()
                    ),
                )
            )
            print(f"[{zone}] ECHEC: {exc}", flush=True)
            if stop_on_error:
                break

    return BatchForecastResult(
        delivery_day=delivery,
        zones=selected,
        results=tuple(results),
    )


def print_batch_summary(batch: BatchForecastResult) -> None:
    print("\nResultat du batch")
    print("=" * 88)
    print(f"Livraison: {batch.delivery_day}")
    for result in batch.results:
        label = {
            "success": "SUCCES",
            "skipped": "DEJA PUBLIE",
            "dry_run": "DRY-RUN",
            "failed": "ECHEC",
        }.get(result.state, result.state.upper())
        print(f"{result.zone:>2} | {label:<12} | {result.message.splitlines()[0]}")
        if result.report_path is not None:
            print(f"   Rapport: {result.report_path}")
        if result.log_path is not None:
            print(f"   Log:     {result.log_path}")
        if result.state == "dry_run" and result.command:
            print("   argv:    " + json.dumps(result.command, ensure_ascii=False))
    print("=" * 88)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lance sequentiellement les forecasts day-ahead et exige un "
            "rapport HTML detaille par pays."
        )
    )
    parser.add_argument("--zones", nargs="+", default=list(APP_ZONES))
    parser.add_argument("--delivery-day", default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Autorise le telechargement Hugging Face si le modele local est absent.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        batch = run_forecast_batch(
            zones=args.zones,
            delivery_day=args.delivery_day,
            project_root=PROJECT_ROOT,
            registry_path=args.registry,
            python_executable=args.python_executable,
            log_dir=DEFAULT_LOG_DIR,
            device=args.device,
            threads=args.threads,
            workers=args.workers,
            local_files_only=not args.allow_model_download,
            stop_on_error=args.stop_on_error,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Erreur launcher: {exc}", file=sys.stderr)
        return 2
    print_batch_summary(batch)
    return 0 if batch.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
