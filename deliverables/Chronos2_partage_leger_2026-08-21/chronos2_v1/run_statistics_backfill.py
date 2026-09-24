#!/usr/bin/env python
"""Rebuild missing causal PIT forecast archives, zone by zone.

The utility never edits an issued-live archive.  It discovers the gaps needed
by the latest immutable live forecast, creates each historical forecast at its
exact D-1 08:00 civil cutoff, validates the published replay, and finally
checks that the Statistics history is contiguous.  An optional final live run
creates new detailed HTML reports containing the completed Statistics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
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
    load_latest_forecast_comparison,
    read_log_tail,
    validate_existing_forecast_archive,
)
from chronos2_hourly.live_history import missing_statistics_archive_days
from chronos2_hourly.zone_live import canonical_zone, load_zone_registry
from chronos2_modular.common import load_yaml
from run_multicountry_forecast import (
    normalize_zones,
    print_batch_summary,
    run_forecast_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = PROJECT_ROOT / "chronos2_hourly_live_zones.yaml"
DEFAULT_LOG_DIR = PROJECT_ROOT / "runs" / "live" / "_backfill_logs"
LOCAL_TIMEZONE = ZoneInfo("Europe/Paris")


@dataclass(frozen=True)
class ZoneHistoryContract:
    zone: str
    timezone: str
    forecast_origin_timezone: str
    forecast_origin_local_time: str
    target_series: str
    prediction_mode: str
    candidate_model: str
    forecast_name: str
    output_root: Path
    sealed_benchmark_run: Path


@dataclass(frozen=True)
class ReplayResult:
    zone: str
    delivery_day: date
    cutoff_local: str
    state: str
    return_code: int
    message: str
    archive_path: Path | None = None
    log_path: Path | None = None
    command: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.state in {"success", "skipped", "dry_run"}


@dataclass(frozen=True)
class BackfillResult:
    source_delivery_days: Mapping[str, date]
    planned_days: Mapping[str, tuple[date, ...]]
    replay_results: tuple[ReplayResult, ...]
    remaining_days: Mapping[str, tuple[date, ...]]

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.replay_results) and all(
            not values for values in self.remaining_days.values()
        )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} doit etre un mapping.")
    return value


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _status_map(
    registry_path: Path,
    zones: tuple[str, ...],
) -> dict[str, ZoneStatus]:
    statuses = inspect_zone_statuses(registry_path, zones=zones)
    return {canonical_zone(status.code): status for status in statuses}


def load_zone_history_contract(
    status: ZoneStatus,
    *,
    registry_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> ZoneHistoryContract:
    """Resolve the exact live-history identity without a permissive fallback."""

    root = Path(project_root).expanduser().resolve()
    registry_file = Path(registry_path).expanduser().resolve()
    registry, _registry_dir = load_zone_registry(registry_file)
    zones = _mapping(registry.get("zones"), name="registry.zones")
    code = canonical_zone(status.code)
    zone_entry = _mapping(zones.get(code), name=f"registry.zones.{code}")
    if status.live_config is None:
        raise ValueError(f"{code}: configuration live absente.")
    config_path = Path(status.live_config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    live = _mapping(config.get("live"), name=f"{code}.live")

    prediction_mode = str(
        live.get("prediction_mode", zone_entry.get("prediction_mode", ""))
    ).strip()
    if prediction_mode not in {"mkonline_blend", "autonomous_only"}:
        raise ValueError(f"{code}: prediction_mode invalide: {prediction_mode!r}.")
    candidate_model = (
        "mkonline_blend"
        if prediction_mode == "mkonline_blend"
        else "residual_corrected"
    )
    output_root = _resolve(live["output_root"], base=config_path.parent)
    benchmark = _resolve(
        live["sealed_benchmark_run"], base=config_path.parent
    )
    try:
        output_root.relative_to(root)
        benchmark.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{code}: les chemins doivent rester dans le projet.") from exc

    target_series = str(zone_entry.get("target_series", "")).strip()
    origin_timezone = str(
        zone_entry.get("forecast_origin_timezone", "")
    ).strip()
    origin_clock = str(
        zone_entry.get("forecast_origin_local_time", "")
    ).strip()
    if not target_series or not origin_timezone or not origin_clock:
        raise ValueError(f"{code}: identite historique incomplete dans le registre.")
    forecast_name = str(
        live.get("forecast_filename")
        or f"forecast_hourly_{code.lower()}.csv"
    ).strip()
    if Path(forecast_name).name != forecast_name:
        raise ValueError(f"{code}: forecast_filename invalide.")
    return ZoneHistoryContract(
        zone=code,
        timezone=status.timezone,
        forecast_origin_timezone=origin_timezone,
        forecast_origin_local_time=origin_clock,
        target_series=target_series,
        prediction_mode=prediction_mode,
        candidate_model=candidate_model,
        forecast_name=forecast_name,
        output_root=output_root,
        sealed_benchmark_run=benchmark,
    )


def replay_cutoff_local(
    delivery_day: date,
    *,
    timezone: str,
    local_time: str = "08:00",
) -> pd.Timestamp:
    """Return the exact civil D-1 cutoff, rejecting ambiguous/nonexistent time."""

    try:
        parsed_time = time.fromisoformat(str(local_time))
    except ValueError as exc:
        raise ValueError(f"Heure de cutoff invalide: {local_time!r}.") from exc
    naive = pd.Timestamp(
        datetime.combine(delivery_day - timedelta(days=1), parsed_time)
    )
    return naive.tz_localize(timezone, ambiguous="raise", nonexistent="raise")


def _latest_issued_day(
    status: ZoneStatus,
    *,
    project_root: Path,
) -> date:
    comparison = load_latest_forecast_comparison(
        [status],
        project_root=project_root,
        zones=[status.code],
    )
    if len(comparison.archives) != 1:
        raise RuntimeError(f"{status.code}: dernier forecast live introuvable.")
    return date.fromisoformat(comparison.archives[0].delivery_day)


def plan_zone_backfill(
    contract: ZoneHistoryContract,
    *,
    source_delivery_day: date,
) -> tuple[date, ...]:
    days = missing_statistics_archive_days(
        sealed_benchmark_run=contract.sealed_benchmark_run,
        live_output_root=contract.output_root,
        replay_output_root=contract.output_root / "_replays",
        current_delivery_day=source_delivery_day,
        timezone=contract.timezone,
        forecast_name=contract.forecast_name,
        candidate_model=contract.candidate_model,
        zone=contract.zone,
        target_series=contract.target_series,
        prediction_mode=contract.prediction_mode,
    )
    ordered = tuple(sorted(days))
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"{contract.zone}: jours de replay dupliques.")
    if any(day >= source_delivery_day for day in ordered):
        raise ValueError(f"{contract.zone}: replay courant/futur refuse.")
    return ordered


def run_statistics_backfill(
    *,
    zones: Sequence[str],
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
) -> BackfillResult:
    """Plan, execute and re-audit every missing causal replay sequentially."""

    selected = normalize_zones(zones)
    root = Path(project_root).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    executable = Path(python_executable).expanduser().resolve()
    logs = Path(log_dir).expanduser().resolve()
    statuses = _status_map(registry, selected)

    # Preflight and planning for every selected zone happen before the first
    # durable replay, so a malformed bundle cannot yield a half-planned batch.
    contracts: dict[str, ZoneHistoryContract] = {}
    source_days: dict[str, date] = {}
    planned: dict[str, tuple[date, ...]] = {}
    for zone in selected:
        status = statuses.get(zone)
        if status is None or not status.launchable:
            detail = "; ".join(status.blockers) if status else "audit absent"
            raise RuntimeError(f"{zone}: preflight refuse: {detail}")
        contract = load_zone_history_contract(
            status,
            registry_path=registry,
            project_root=root,
        )
        source_day = _latest_issued_day(status, project_root=root)
        contracts[zone] = contract
        source_days[zone] = source_day
        planned[zone] = plan_zone_backfill(
            contract,
            source_delivery_day=source_day,
        )

    results: list[ReplayResult] = []
    stop = False
    for zone in selected:
        if stop:
            break
        status = statuses[zone]
        contract = contracts[zone]
        for replay_day in planned[zone]:
            cutoff = replay_cutoff_local(
                replay_day,
                timezone=contract.forecast_origin_timezone,
                local_time=contract.forecast_origin_local_time,
            )
            command = tuple(
                build_dispatch_command(
                    status,
                    project_root=root,
                    registry_path=registry,
                    python_executable=executable,
                    delivery_day=replay_day,
                    data_as_of=cutoff.isoformat(),
                    device=device,
                    threads=int(threads),
                    workers=int(workers),
                    local_files_only=local_files_only,
                    pit_replay=True,
                )
            )
            if dry_run:
                results.append(
                    ReplayResult(
                        zone=zone,
                        delivery_day=replay_day,
                        cutoff_local=cutoff.isoformat(),
                        state="dry_run",
                        return_code=0,
                        message="Commande causale validee; aucune execution.",
                        command=command,
                    )
                )
                continue

            print(
                f"\n[{zone}] replay {replay_day.isoformat()} "
                f"(cutoff {cutoff.isoformat()})...",
                flush=True,
            )
            launched: ForecastProcess | ForecastSkip | None = None
            process_code: int | None = None
            try:
                launched = launch_zone_forecast(
                    zone=zone,
                    project_root=root,
                    registry_path=registry,
                    log_dir=logs,
                    python_executable=executable,
                    delivery_day=replay_day,
                    data_as_of=cutoff.isoformat(),
                    device=device,
                    threads=int(threads),
                    workers=int(workers),
                    local_files_only=local_files_only,
                    pit_replay=True,
                )
                if isinstance(launched, ForecastSkip):
                    state = "skipped"
                    message = "Replay deja publie, valide et reutilise."
                    log_path = None
                else:
                    print(f"[{zone}] log: {launched.log_path}", flush=True)
                    process_code = int(launched.process.wait())
                    log_path = launched.log_path
                    if process_code != 0:
                        tail = read_log_tail(log_path, max_chars=8_000).strip()
                        raise RuntimeError(
                            f"runner code={process_code}"
                            + (("\n" + tail) if tail else "")
                        )
                    state = "success"
                    message = "Replay causal publie et valide."
                archive = validate_existing_forecast_archive(
                    status,
                    project_root=root,
                    delivery_day=replay_day,
                    pit_replay=True,
                )
                if archive is None:
                    raise RuntimeError("Replay absent apres succes du runner.")
                results.append(
                    ReplayResult(
                        zone=zone,
                        delivery_day=replay_day,
                        cutoff_local=cutoff.isoformat(),
                        state=state,
                        return_code=0,
                        message=message,
                        archive_path=archive,
                        log_path=log_path,
                        command=(
                            tuple(launched.command)
                            if isinstance(launched, ForecastProcess)
                            else command
                        ),
                    )
                )
            except Exception as exc:
                log_path = (
                    launched.log_path
                    if isinstance(launched, ForecastProcess)
                    else None
                )
                results.append(
                    ReplayResult(
                        zone=zone,
                        delivery_day=replay_day,
                        cutoff_local=cutoff.isoformat(),
                        state="failed",
                        return_code=(
                            process_code if process_code not in (None, 0) else 1
                        ),
                        message=str(exc),
                        log_path=log_path,
                        command=command,
                    )
                )
                print(f"[{zone}] ECHEC replay {replay_day}: {exc}", flush=True)
                # Statistics require a strictly contiguous prefix.  Once one
                # replay day fails, later days from the same zone cannot repair
                # that prefix and would only repeat expensive network/model
                # work.  Continue with the next zone unless the caller asked
                # for a global stop.
                if stop_on_error:
                    stop = True
                break

    remaining: dict[str, tuple[date, ...]] = {}
    if dry_run:
        remaining = dict(planned)
    else:
        for zone in selected:
            remaining[zone] = plan_zone_backfill(
                contracts[zone],
                source_delivery_day=source_days[zone],
            )
    return BackfillResult(
        source_delivery_days=source_days,
        planned_days=planned,
        replay_results=tuple(results),
        remaining_days=remaining,
    )


def print_backfill_summary(result: BackfillResult) -> None:
    print("\nRattrapage Statistics")
    print("=" * 92)
    for zone, days in result.planned_days.items():
        text = ", ".join(day.isoformat() for day in days) or "aucun"
        print(
            f"{zone}: source live {result.source_delivery_days[zone]} | "
            f"replays planifies: {text}"
        )
    for item in result.replay_results:
        print(
            f"{item.zone} {item.delivery_day} | {item.state.upper():<8} | "
            f"{item.message.splitlines()[0]}"
        )
        if item.archive_path is not None:
            print(f"   Archive: {item.archive_path}")
        if item.log_path is not None:
            print(f"   Log:     {item.log_path}")
        if item.state == "dry_run":
            print("   argv:    " + json.dumps(item.command, ensure_ascii=False))
    for zone, days in result.remaining_days.items():
        if days:
            print(
                f"{zone}: RESTE A RECONSTRUIRE: "
                + ", ".join(day.isoformat() for day in days)
            )
    print("=" * 92)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruit les archives PIT manquantes puis, en option, lance "
            "le prochain forecast pour publier des Statistics completes."
        )
    )
    parser.add_argument("--zones", nargs="+", default=list(APP_ZONES))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--then-run-live", action="store_true")
    parser.add_argument("--live-delivery-day", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selected = normalize_zones(args.zones)
        # Freeze the follow-up live date before a potentially long replay
        # batch so crossing midnight cannot silently change the requested day.
        follow_up_delivery_day = args.live_delivery_day
        if args.then_run_live and not follow_up_delivery_day:
            follow_up_delivery_day = (
                datetime.now(LOCAL_TIMEZONE).date() + timedelta(days=1)
            ).isoformat()
        result = run_statistics_backfill(
            zones=selected,
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
        print_backfill_summary(result)
        if args.dry_run:
            return 0
        if not result.ok:
            return 1
        if args.then_run_live:
            live_batch = run_forecast_batch(
                zones=selected,
                delivery_day=follow_up_delivery_day,
                project_root=PROJECT_ROOT,
                registry_path=args.registry,
                python_executable=args.python_executable,
                device=args.device,
                threads=args.threads,
                workers=args.workers,
                local_files_only=not args.allow_model_download,
                stop_on_error=args.stop_on_error,
            )
            print_batch_summary(live_batch)
            return 0 if live_batch.ok else 1
        print(
            "Les replays sont complets. Lancez ensuite le forecast live suivant "
            "pour regenerer les rapports HTML avec toutes les Statistics."
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
