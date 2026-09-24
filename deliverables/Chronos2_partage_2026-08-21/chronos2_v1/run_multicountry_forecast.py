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
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence
from zoneinfo import ZoneInfo

import pandas as pd

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
from chronos2_hourly.reporting import write_hourly_html_report


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = PROJECT_ROOT / "chronos2_hourly_live_zones.yaml"
DEFAULT_LOG_DIR = PROJECT_ROOT / "runs" / "live" / "_launcher_logs"
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "runs" / "exports"
LOCAL_TIMEZONE = ZoneInfo("Europe/Paris")
FORECAST_MODES = ("production", "autonomous", "blend", "both")
BLEND_ZONES = frozenset({"FR", "NL"})


@dataclass(frozen=True)
class ForecastExport:
    """A user-facing view derived from an immutable production archive."""

    variant: str
    source_model: str
    csv_path: Path
    report_path: Path
    source_archive: Path


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
    exports: tuple[ForecastExport, ...] = ()

    @property
    def ok(self) -> bool:
        return self.state in {"success", "skipped", "dry_run"}


@dataclass(frozen=True)
class BatchForecastResult:
    delivery_day: str
    zones: tuple[str, ...]
    results: tuple[BatchZoneResult, ...]
    mode: str = "production"

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


def normalise_forecast_mode(value: str | None) -> str:
    """Return the canonical execution view requested by the user."""

    mode = str(value or "production").strip().lower()
    if mode not in FORECAST_MODES:
        raise ValueError(
            "Mode doit etre Production, Autonomous, Blend ou Both."
        )
    return mode


def requested_export_variants(mode: str, zone: str) -> tuple[str, ...]:
    """Resolve export variants without ever changing the sealed recipe."""

    canonical_mode = normalise_forecast_mode(mode)
    canonical = canonical_zone(zone)
    if canonical_mode == "production":
        return ()
    if canonical_mode == "autonomous":
        return ("autonomous",)
    if canonical_mode == "blend":
        if canonical not in BLEND_ZONES:
            raise ValueError(
                f"Le mode Blend n'est disponible que pour FR et NL; "
                f"{canonical} ne possede pas de blend MKOnline promu."
            )
        return ("blend",)
    if canonical in BLEND_ZONES:
        return ("autonomous", "blend")
    return ("autonomous",)


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


@dataclass(frozen=True)
class _ExportSpec:
    zone: str
    timezone: str
    delivery_day: str
    variant: str
    source_model: str
    baseline_model: str
    archive: Path
    csv_path: Path
    report_path: Path


_REPORT_INPUT_FILES = (
    "aligned_inputs.csv.gz",
    "model_covariates_with_future.csv.gz",
    "input_coverage.csv",
    "input_manifest.csv",
)
_REPORT_OPTIONAL_FILES = (
    "metrics_hourly.json",
    "statistics_history_hourly.csv.gz",
    "statistics_history_audit.json",
    "run_manifest.json",
)


def _forecast_path(archive: Path, zone: str) -> Path:
    path = archive / f"forecast_hourly_{zone.lower()}.csv"
    if not path.is_file() and zone == "FR":
        path = archive / "forecast_hourly_fr.csv"
    return path


def _model_for_variant(variant: str) -> tuple[str, str]:
    if variant == "autonomous":
        # Chronos-2 is the autonomous upstream baseline and is available in
        # the same frozen run.  It does not introduce an MKOnline dependency.
        return "residual_corrected", "chronos2"
    if variant == "blend":
        return "mkonline_blend", "residual_corrected"
    raise ValueError(f"Variante d'export inconnue: {variant!r}.")


def _build_export_specs(
    *,
    zone_results: Sequence[BatchZoneResult],
    statuses: dict[str, ZoneStatus],
    mode: str,
    delivery_day: str,
    export_root: Path,
) -> tuple[_ExportSpec, ...]:
    specs: list[_ExportSpec] = []
    for result in zone_results:
        if result.archive_path is None:
            raise ValueError(f"{result.zone}: archive source absente.")
        archive = result.archive_path.resolve()
        for variant in requested_export_variants(mode, result.zone):
            source_model, baseline_model = _model_for_variant(variant)
            directory = export_root / delivery_day / result.zone.lower() / variant
            stem = f"forecast_{result.zone.lower()}_{delivery_day}_{variant}"
            specs.append(
                _ExportSpec(
                    zone=result.zone,
                    timezone=statuses[result.zone].timezone,
                    delivery_day=delivery_day,
                    variant=variant,
                    source_model=source_model,
                    baseline_model=baseline_model,
                    archive=archive,
                    csv_path=(directory / f"{stem}.csv").resolve(),
                    report_path=(directory / f"{stem}.html").resolve(),
                )
            )
    return tuple(specs)


def _required_model_columns(model: str) -> set[str]:
    return {f"{model}__q10", f"{model}__q50", f"{model}__q90"}


def _read_csv_columns(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return set(pd.read_csv(path, nrows=0).columns.astype(str))


def _validate_export_spec(spec: _ExportSpec) -> None:
    """Fail closed before creating anything in the export directory."""

    forecast_path = _forecast_path(spec.archive, spec.zone)
    backtest_path = spec.archive / "backtest_hourly_oof.csv.gz"
    forecast_columns = _read_csv_columns(forecast_path)
    backtest_columns = _read_csv_columns(backtest_path)
    required_forecast = {
        "delivery_start_utc",
        *_required_model_columns(spec.source_model),
        *_required_model_columns(spec.baseline_model),
    }
    required_backtest = {
        "delivery_start_utc",
        "actual",
        *_required_model_columns(spec.source_model),
        *_required_model_columns(spec.baseline_model),
    }
    missing_forecast = sorted(required_forecast - forecast_columns)
    missing_backtest = sorted(required_backtest - backtest_columns)
    if missing_forecast or missing_backtest:
        details: list[str] = []
        if missing_forecast:
            details.append("forecast=" + ", ".join(missing_forecast))
        if missing_backtest:
            details.append("backtest=" + ", ".join(missing_backtest))
        raise ValueError(
            f"{spec.zone}/{spec.variant}: colonnes auditees absentes ("
            + "; ".join(details)
            + "). Aucun export n'a ete cree."
        )
    statistics_path = spec.archive / "statistics_history_hourly.csv.gz"
    if statistics_path.is_file():
        statistics_columns = _read_csv_columns(statistics_path)
        required_statistics = {
            "delivery_start_utc",
            "actual",
            *_required_model_columns(spec.source_model),
        }
        missing_statistics = sorted(required_statistics - statistics_columns)
        if missing_statistics:
            raise ValueError(
                f"{spec.zone}/{spec.variant}: Statistics ne contient pas "
                f"{', '.join(missing_statistics)}. Aucun export n'a ete cree."
            )
    for filename in _REPORT_INPUT_FILES:
        path = spec.archive / "inputs" / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"{spec.zone}/{spec.variant}: artefact de reporting absent: {path}"
            )


def _copy_reporting_view(spec: _ExportSpec, destination: Path) -> None:
    """Create a disposable report view, never a child of the live archive."""

    destination.mkdir(parents=True, exist_ok=False)
    inputs = destination / "inputs"
    inputs.mkdir()
    shutil.copy2(
        spec.archive / "backtest_hourly_oof.csv.gz",
        destination / "backtest_hourly_oof.csv.gz",
    )
    shutil.copy2(
        _forecast_path(spec.archive, spec.zone),
        destination / f"forecast_hourly_{spec.zone.lower()}.csv",
    )
    for filename in _REPORT_INPUT_FILES:
        shutil.copy2(spec.archive / "inputs" / filename, inputs / filename)
    for filename in _REPORT_OPTIONAL_FILES:
        source = spec.archive / filename
        if source.is_file():
            shutil.copy2(source, destination / filename)


def _write_forecast_csv(spec: _ExportSpec, output_path: Path) -> None:
    raw = pd.read_csv(_forecast_path(spec.archive, spec.zone))
    model_columns = {
        quantile: f"{spec.source_model}__{quantile}"
        for quantile in ("q10", "q50", "q90")
    }
    timing_columns = [
        column
        for column in (
            "delivery_start_utc",
            "delivery_start_local",
            "utc_offset",
            "utc_offset_minutes",
            "fold",
            "local_date",
            "local_hour",
            "delivery_hour_position",
            "hours_in_local_day",
        )
        if column in raw
    ]
    export = raw.loc[:, timing_columns].copy()
    export.insert(0, "zone", spec.zone)
    export.insert(1, "forecast_variant", spec.variant)
    export.insert(2, "source_model", spec.source_model)
    export.insert(3, "uses_mkonline", spec.variant == "blend")
    for quantile, source in model_columns.items():
        export[quantile] = pd.to_numeric(raw[source], errors="raise")
    export["price_eur_mwh"] = export["q50"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_path, index=False)


def _publish_exports(specs: Sequence[_ExportSpec]) -> dict[str, tuple[ForecastExport, ...]]:
    """Build the complete batch in temp space, then publish its files."""

    if not specs:
        return {}
    for spec in specs:
        _validate_export_spec(spec)

    staged: list[tuple[_ExportSpec, Path, Path]] = []
    with tempfile.TemporaryDirectory(prefix="chronos2_forecast_exports_") as raw_tmp:
        temporary_root = Path(raw_tmp).resolve()
        for index, spec in enumerate(specs):
            work = temporary_root / f"view_{index}_{spec.zone.lower()}_{spec.variant}"
            _copy_reporting_view(spec, work)
            generated = temporary_root / "generated" / str(index)
            csv_staged = generated / spec.csv_path.name
            report_staged = generated / spec.report_path.name
            _write_forecast_csv(spec, csv_staged)
            write_hourly_html_report(
                work,
                output_path=report_staged,
                title=(
                    f"Forecast {spec.zone} {spec.delivery_day} - "
                    + (
                        "autonome sans MKOnline"
                        if spec.variant == "autonomous"
                        else "blend MKOnline"
                    )
                ),
                native_model=spec.source_model,
                baseline_model=spec.baseline_model,
                zone=spec.zone,
                timezone=spec.timezone,
            )
            if not csv_staged.is_file() or csv_staged.stat().st_size <= 0:
                raise RuntimeError(f"Export CSV vide: {spec.zone}/{spec.variant}.")
            if not report_staged.is_file() or report_staged.stat().st_size <= 0:
                raise RuntimeError(f"Rapport HTML vide: {spec.zone}/{spec.variant}.")
            staged.append((spec, csv_staged, report_staged))

        # No runs/live path is touched: only derived, reproducible files are
        # atomically replaced after the whole export batch has succeeded.
        for spec, csv_staged, report_staged in staged:
            spec.csv_path.parent.mkdir(parents=True, exist_ok=True)
            spec.report_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(csv_staged, spec.csv_path)
            os.replace(report_staged, spec.report_path)

    by_zone: dict[str, list[ForecastExport]] = {}
    for spec in specs:
        by_zone.setdefault(spec.zone, []).append(
            ForecastExport(
                variant=spec.variant,
                source_model=spec.source_model,
                csv_path=spec.csv_path,
                report_path=spec.report_path,
                source_archive=spec.archive,
            )
        )
    return {zone: tuple(exports) for zone, exports in by_zone.items()}


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
    mode: str = "production",
    export_root: str | Path = DEFAULT_EXPORT_ROOT,
) -> BatchForecastResult:
    """Run selected countries sequentially and audit every detailed report."""

    selected = normalize_zones(zones)
    delivery = normalise_delivery_day(delivery_day)
    selected_mode = normalise_forecast_mode(mode)
    # This validates the complete selection before inspecting, launching or
    # creating any output.  Both deliberately falls back to autonomous for
    # countries without a promoted MKOnline blend.
    for zone in selected:
        requested_export_variants(selected_mode, zone)
    root = Path(project_root).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    executable = Path(python_executable).expanduser().resolve()
    logs = Path(log_dir).expanduser().resolve()
    exports_root = Path(export_root).expanduser().resolve()
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
    live_root = (root / "runs" / "live").resolve()
    if exports_root == live_root or live_root in exports_root.parents:
        raise ValueError(
            "Le dossier d'exports doit rester hors de runs/live afin de "
            "preserver les archives immuables."
        )

    statuses = _status_map(registry, selected)
    preflight_failures: dict[str, str] = {}
    for zone in selected:
        status = statuses.get(zone)
        if status is None:
            preflight_failures[zone] = "audit absent"
        elif not status.launchable:
            preflight_failures[zone] = "; ".join(status.blockers)
    if preflight_failures:
        # Batch-level preflight: one invalid country prevents every runner
        # from starting, so selection errors cannot leave a partial batch.
        results = tuple(
            BatchZoneResult(
                zone=zone,
                delivery_day=delivery,
                state="failed",
                return_code=2,
                message=(
                    "Preflight refuse: " + preflight_failures[zone]
                    if zone in preflight_failures
                    else "Batch annule avant lancement: un autre pays a echoue au preflight."
                ),
            )
            for zone in selected
        )
        return BatchForecastResult(
            delivery_day=delivery,
            zones=selected,
            results=results,
            mode=selected_mode,
        )

    if not dry_run and selected_mode != "production":
        source_failures: dict[str, str] = {}
        for zone in selected:
            try:
                existing = validate_existing_forecast_archive(
                    statuses[zone],
                    project_root=root,
                    delivery_day=delivery,
                )
                if existing is None:
                    continue
                find_detailed_html_report(existing)
                provisional = BatchZoneResult(
                    zone=zone,
                    delivery_day=delivery,
                    state="skipped",
                    return_code=0,
                    message="Archive existante pre-validee.",
                    archive_path=existing,
                )
                existing_specs = _build_export_specs(
                    zone_results=(provisional,),
                    statuses=statuses,
                    mode=selected_mode,
                    delivery_day=delivery,
                    export_root=exports_root,
                )
                for spec in existing_specs:
                    _validate_export_spec(spec)
            except Exception as exc:
                source_failures[zone] = str(exc)
        if source_failures:
            results = tuple(
                BatchZoneResult(
                    zone=zone,
                    delivery_day=delivery,
                    state="failed",
                    return_code=2,
                    message=(
                        "Preflight export refuse: " + source_failures[zone]
                        if zone in source_failures
                        else "Batch annule avant lancement: une archive existante "
                        "est incompatible avec le mode demande."
                    ),
                )
                for zone in selected
            )
            return BatchForecastResult(
                delivery_day=delivery,
                zones=selected,
                results=results,
                mode=selected_mode,
            )

    results: list[BatchZoneResult] = []
    for zone in selected:
        status = statuses[zone]
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

    if not dry_run and selected_mode != "production":
        all_runs_ready = len(results) == len(selected) and all(
            result.ok and result.archive_path is not None for result in results
        )
        if all_runs_ready:
            try:
                specs = _build_export_specs(
                    zone_results=results,
                    statuses=statuses,
                    mode=selected_mode,
                    delivery_day=delivery,
                    export_root=exports_root,
                )
                published = _publish_exports(specs)
                results = [
                    replace(
                        result,
                        report_path=(
                            published[result.zone][0].report_path
                            if published.get(result.zone)
                            else result.report_path
                        ),
                        exports=published.get(result.zone, ()),
                        message=(
                            result.message
                            + f" {len(published.get(result.zone, ()))} export(s) "
                            + "publie(s) hors de l'archive live."
                        ),
                    )
                    for result in results
                ]
            except Exception as exc:
                # Export is an explicit part of these modes.  A failure must
                # be visible even though the sealed production run succeeded.
                results = [
                    replace(
                        result,
                        state="failed",
                        return_code=1,
                        message=(
                            result.message
                            + " Echec du batch d'exports: "
                            + str(exc)
                        ),
                        exports=(),
                    )
                    for result in results
                ]

    return BatchForecastResult(
        delivery_day=delivery,
        zones=selected,
        results=tuple(results),
        mode=selected_mode,
    )


def print_batch_summary(batch: BatchForecastResult) -> None:
    print("\nResultat du batch")
    print("=" * 88)
    print(f"Livraison: {batch.delivery_day}")
    print(f"Mode:      {batch.mode}")
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
        for forecast_export in result.exports:
            print(
                f"   Export {forecast_export.variant}: "
                f"{forecast_export.csv_path}"
            )
            print(f"   Rapport {forecast_export.variant}: {forecast_export.report_path}")
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
    parser.add_argument(
        "--mode",
        default="Production",
        help=(
            "Production conserve le run audite; Autonomous exporte la variante "
            "sans MKOnline; Blend est limite a FR/NL; Both exporte les deux "
            "pour FR/NL et l'autonome pour les autres pays."
        ),
    )
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
            mode=args.mode,
        )
    except Exception as exc:
        print(f"Erreur launcher: {exc}", file=sys.stderr)
        return 2
    print_batch_summary(batch)
    return 0 if batch.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
