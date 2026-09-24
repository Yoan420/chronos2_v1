"""Testable services for the multi-zone forecast control room.

This module intentionally has no Streamlit dependency.  The UI can therefore
be imported and tested in the forecasting environment before Streamlit is
installed.  Storm data is read only from post-forecast Statistics artifacts;
it is never forwarded to a runner or exposed as a model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.zone_live import (
    ZoneBundleError,
    audit_zone_live_bundle,
    canonical_zone,
    load_zone_registry,
    strict_contract_preflight,
)
from chronos2_modular.common import load_yaml


APP_ZONES: tuple[str, ...] = ("FR", "DE", "BE", "NL", "ES")
_REQUESTS_PROXY_ENV_NAMES: tuple[str, ...] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
STATISTIC_DEFINITIONS: tuple[tuple[str, str, bool], ...] = (
    ("mae", "MAE", False),
    ("rmse", "RMSE", False),
    ("mape", "MAPE (%)", False),
    ("explained_variance", "Explained Variance", True),
    ("r2", "R²", True),
    ("std_error", "Standard Deviation", False),
    ("correlation", "Correlation", True),
)
MAPE_DENOMINATOR_EPSILON = 1e-9
SAMPLE_FREQUENCIES: Mapping[str, str] = {
    "daily": "D",
    "weekly": "W-SUN",
    "monthly": "M",
}
CANDIDATE_COLUMN_PRIORITY: tuple[str, ...] = (
    "mkonline_blend__q50",
    "residual_corrected__q50",
    "ensemble__q50",
    "chronos2__q50",
)
FORECAST_VARIANTS: tuple[str, ...] = (
    "production",
    "autonomous",
    "mkonline_blend",
)
BENCHMARK_COLUMN_PRIORITY: tuple[str, ...] = (
    # The native dashboard snapshot is the reporting benchmark when its
    # sidecar audit explicitly activates it.
    "storm_dashboard_official__q50",
    "storm_evaluation_only__q50",
)


@dataclass(frozen=True)
class ZoneStatus:
    code: str
    timezone: str
    enabled: bool
    production_ready: bool
    ready: bool
    runner: Path | None
    live_config: Path | None
    checks: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def launchable(self) -> bool:
        return self.enabled and self.production_ready and self.ready


@dataclass(frozen=True)
class RunArtifact:
    zone: str
    kind: str
    directory: Path
    modified_at: datetime
    report_path: Path | None
    statistics_path: Path | None
    forecast_path: Path | None

    @property
    def label(self) -> str:
        timestamp = self.modified_at.astimezone().strftime("%Y-%m-%d %H:%M")
        return f"{self.zone} · {self.kind} · {timestamp} · {self.directory.name}"


@dataclass(frozen=True)
class StatisticsDataset:
    path: Path
    frame: pd.DataFrame
    candidate_column: str
    candidate_label: str
    benchmark_column: str | None
    benchmark_label: str | None
    scope_note: str | None


@dataclass(frozen=True)
class PerformanceArtifact:
    """One fully audited Statistics history selected for a zone."""

    zone: str
    timezone: str
    archive_kind: str
    delivery_day: str
    archive_path: Path
    statistics_path: Path
    audit_path: Path
    statistics_prefix_end_local: str
    n_total_statistics_hours: int
    statistics_complete: bool
    missing_realized_days: tuple[str, ...]
    statistics_sha256: str
    modified_at: datetime


@dataclass(frozen=True)
class ForecastDataset:
    path: Path
    frame: pd.DataFrame


@dataclass(frozen=True)
class ForecastArchiveDataset:
    """One fully audited issued-live forecast used by the comparison view."""

    zone: str
    timezone: str
    delivery_day: str
    archive_path: Path
    forecast_path: Path
    forecast_sha256: str
    checksum_manifest_sha256: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class ForecastComparison:
    """Long-form view of one or more immutable day-ahead forecasts."""

    archives: tuple[ForecastArchiveDataset, ...]
    frame: pd.DataFrame
    timeline_aligned: bool
    variant: str = "production"

    @property
    def zones(self) -> tuple[str, ...]:
        return tuple(archive.zone for archive in self.archives)

    @property
    def delivery_days(self) -> dict[str, str]:
        return {
            archive.zone: archive.delivery_day for archive in self.archives
        }

    @property
    def mixed_delivery_days(self) -> bool:
        return len(set(self.delivery_days.values())) > 1


@dataclass(frozen=True)
class StatisticsView:
    sample: str
    periods: pd.DataFrame
    summary: pd.DataFrame
    time_series: pd.DataFrame


@dataclass
class ForecastProcess:
    zone: str
    command: tuple[str, ...]
    log_path: Path
    started_at: datetime
    process: subprocess.Popen[Any]

    @property
    def return_code(self) -> int | None:
        return self.process.poll()

    @property
    def running(self) -> bool:
        return self.return_code is None


class ExistingForecastArchiveError(ValueError):
    """An immutable output exists but fails the idempotent archive contract."""


class MixedForecastDeliveryDaysError(ExistingForecastArchiveError):
    """Selected zones do not share the same latest delivery day."""

    def __init__(self, delivery_days: Mapping[str, str]) -> None:
        self.delivery_days = dict(delivery_days)
        detail = ", ".join(
            f"{zone}={day}" for zone, day in self.delivery_days.items()
        )
        super().__init__(
            "Les derniers forecasts ne portent pas sur le meme jour de "
            f"livraison ({detail}). Une autorisation explicite est requise."
        )


@dataclass(frozen=True)
class ForecastSkip:
    """Synthetic successful queue result for an already published forecast."""

    zone: str
    delivery_day: str
    archive_path: Path
    status: str = "Déjà publié — ignoré"
    return_code: int = 0

    @property
    def running(self) -> bool:
        return False


def inspect_zone_statuses(
    registry_path: str | Path,
    *,
    zones: Sequence[str] = APP_ZONES,
) -> list[ZoneStatus]:
    """Read the registry and run the same fail-closed audit as the dispatcher."""

    registry, registry_dir = load_zone_registry(registry_path)
    statuses: list[ZoneStatus] = []
    for requested in zones:
        code = canonical_zone(requested)
        try:
            audit = audit_zone_live_bundle(
                registry,
                zone=code,
                registry_dir=registry_dir,
            )
            audit = strict_contract_preflight(
                audit,
                registry_path=registry_path,
            )
            statuses.append(
                ZoneStatus(
                    code=code,
                    timezone=audit.timezone,
                    enabled=audit.enabled,
                    production_ready=audit.production_ready,
                    ready=audit.ready,
                    runner=audit.runner,
                    live_config=audit.live_config,
                    checks=audit.checks,
                    blockers=audit.blockers,
                )
            )
        except Exception as exc:  # keep one malformed zone from hiding others
            raw = registry.get("zones", {}).get(code, {})
            statuses.append(
                ZoneStatus(
                    code=code,
                    timezone=str(raw.get("delivery_timezone", "")),
                    enabled=bool(raw.get("enabled", False)),
                    production_ready=bool(raw.get("production_ready", False)),
                    ready=False,
                    runner=None,
                    live_config=None,
                    checks=(),
                    blockers=(f"audit impossible: {exc}",),
                )
            )
    return statuses


def _iso_date(value: str | date | None, *, name: str) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} doit respecter YYYY-MM-DD") from exc


def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExistingForecastArchiveError(
            f"Archive existante invalide: {path} est absent ou illisible."
        ) from exc
    if not isinstance(payload, dict):
        raise ExistingForecastArchiveError(
            f"Archive existante invalide: {path} doit contenir un objet JSON."
        )
    return payload


def validate_existing_forecast_archive(
    status: ZoneStatus,
    *,
    project_root: str | Path,
    delivery_day: str | date,
    pit_replay: bool = False,
) -> Path | None:
    """Return a fully verified immutable archive, or ``None`` if absent.

    Only files sealed as ``role=run_artifact`` are rehashed.  Absolute source
    references describe provenance outside the immutable archive and are not
    part of this idempotency decision.
    """

    code = canonical_zone(status.code)
    delivery_text = _iso_date(delivery_day, name="delivery_day")
    if delivery_text is None:  # defensive: the public type requires a value
        raise ValueError("delivery_day est obligatoire pour l'audit idempotent")
    delivery = date.fromisoformat(delivery_text)
    root = Path(project_root).expanduser().resolve()
    if status.live_config is None:
        raise ExistingForecastArchiveError(
            f"{code}: configuration live absente; archive impossible à auditer."
        )
    config_path = Path(status.live_config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    try:
        config = load_yaml(config_path)
    except Exception as exc:
        raise ExistingForecastArchiveError(
            f"{code}: configuration live illisible: {config_path}"
        ) from exc
    live = config.get("live") if isinstance(config, Mapping) else None
    if not isinstance(live, Mapping):
        raise ExistingForecastArchiveError(
            f"{code}: la configuration live doit contenir un mapping 'live'."
        )
    raw_output_root = live.get("output_root")
    if raw_output_root in (None, ""):
        raise ExistingForecastArchiveError(
            f"{code}: live.output_root est absent."
        )
    output_root = Path(str(raw_output_root)).expanduser()
    if not output_root.is_absolute():
        output_root = config_path.parent / output_root
    output_root = output_root.resolve()
    try:
        output_root.relative_to(root)
    except ValueError as exc:
        raise ExistingForecastArchiveError(
            f"{code}: live.output_root doit rester dans le projet: {output_root}"
        ) from exc
    archive_root = output_root / "_replays" if pit_replay else output_root
    archive = (
        archive_root / f"{code.lower()}_day_ahead_{delivery_text}"
    ).resolve()
    if archive.parent != archive_root.resolve():
        raise ExistingForecastArchiveError(
            f"{code}: chemin d'archive inattendu: {archive}"
        )
    if not archive.exists():
        return None
    if not archive.is_dir():
        raise ExistingForecastArchiveError(
            f"{code}: le chemin publié existe mais n'est pas un dossier: {archive}"
        )

    forecast_name = str(
        live.get("forecast_filename") or f"forecast_hourly_{code.lower()}.csv"
    ).strip()
    if not forecast_name or Path(forecast_name).name != forecast_name:
        raise ExistingForecastArchiveError(
            f"{code}: nom de forecast invalide dans la configuration live."
        )
    manifest = _archive_json(archive / "run_manifest.json")
    summary = _archive_json(archive / "live_run_summary.json")
    checksum_manifest = _archive_json(archive / "artifact_checksums.json")
    expected_run_type = "pit_replay" if pit_replay else "live_day_ahead"
    expected_forecast_status = "pit_reconstruction" if pit_replay else "issued_live"
    expected_identity = {
        "zone": code,
        "timezone": status.timezone,
        "delivery_day_local": delivery_text,
        "run_type": expected_run_type,
        "forecast_status": expected_forecast_status,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ExistingForecastArchiveError(
                f"{archive}: run_manifest.{key}={manifest.get(key)!r}, "
                f"attendu {expected!r}."
            )
    if manifest.get("forecast_path") not in (None, forecast_name):
        raise ExistingForecastArchiveError(
            f"{archive}: run_manifest.forecast_path ne correspond pas à {forecast_name}."
        )
    if manifest.get("sha256_manifest") != "artifact_checksums.json":
        raise ExistingForecastArchiveError(
            f"{archive}: manifeste SHA-256 non déclaré."
        )
    summary_status = str(summary.get("status", ""))
    if summary_status != "complete" and not summary_status.startswith(
        "forecast_complete_"
    ):
        raise ExistingForecastArchiveError(
            f"{archive}: live_run_summary.status={summary_status!r} n'est pas complet."
        )
    summary_identity = {
        "delivery_day_local": delivery_text,
        "run_type": expected_run_type,
        "forecast_path": forecast_name,
    }
    for key, expected in summary_identity.items():
        if str(summary.get(key, "")) != expected:
            raise ExistingForecastArchiveError(
                f"{archive}: live_run_summary.{key}={summary.get(key)!r}, "
                f"attendu {expected!r}."
            )
    if summary.get("zone") not in (None, code):
        raise ExistingForecastArchiveError(
            f"{archive}: live_run_summary.zone ne correspond pas à {code}."
        )

    if checksum_manifest.get("algorithm") != "sha256":
        raise ExistingForecastArchiveError(
            f"{archive}: artifact_checksums.algorithm doit être sha256."
        )
    declared_output = checksum_manifest.get("output_directory")
    if declared_output and Path(str(declared_output)).expanduser().resolve() != archive:
        raise ExistingForecastArchiveError(
            f"{archive}: artifact_checksums.output_directory ne correspond pas."
        )
    artifacts = checksum_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ExistingForecastArchiveError(
            f"{archive}: artifact_checksums.artifacts doit être une liste."
        )
    declarations: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping) or item.get("role") != "run_artifact":
            continue
        raw_path = Path(str(item.get("path", "")))
        if (
            not str(raw_path)
            or raw_path.is_absolute()
            or ".." in raw_path.parts
        ):
            raise ExistingForecastArchiveError(
                f"{archive}: chemin run_artifact non sûr: {raw_path}"
            )
        relative = raw_path.as_posix()
        if relative in declarations:
            raise ExistingForecastArchiveError(
                f"{archive}: checksum dupliqué pour {relative}."
            )
        declarations[relative] = item
    required = {
        "run_manifest.json",
        "live_run_summary.json",
        forecast_name,
    }
    missing_declarations = sorted(required.difference(declarations))
    if missing_declarations:
        raise ExistingForecastArchiveError(
            f"{archive}: checksums obligatoires absents: {missing_declarations}."
        )
    actual_files: set[str] = set()
    for path in archive.rglob("*"):
        if not path.is_file() or path.name == "artifact_checksums.json":
            continue
        try:
            relative = path.resolve().relative_to(archive).as_posix()
        except ValueError as exc:
            raise ExistingForecastArchiveError(
                f"{archive}: fichier publié hors archive: {path}"
            ) from exc
        actual_files.add(relative)
    if actual_files != set(declarations):
        undeclared = sorted(actual_files.difference(declarations))
        absent = sorted(set(declarations).difference(actual_files))
        raise ExistingForecastArchiveError(
            f"{archive}: couverture checksum incomplète; "
            f"non déclarés={undeclared}, absents={absent}."
        )
    for relative, item in declarations.items():
        path = archive / relative
        try:
            expected_size = int(item.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise ExistingForecastArchiveError(
                f"{archive}: taille invalide pour {relative}."
            ) from exc
        if path.stat().st_size != expected_size:
            raise ExistingForecastArchiveError(
                f"{archive}: taille divergente pour {relative}."
            )
        expected_sha = str(item.get("sha256", "")).lower()
        if len(expected_sha) != 64 or _archive_sha256(path) != expected_sha:
            raise ExistingForecastArchiveError(
                f"{archive}: checksum divergent pour {relative}."
            )

    forecast_path = archive / forecast_name
    try:
        forecast = pd.read_csv(forecast_path)
        delivery_index = pd.DatetimeIndex(
            pd.to_datetime(forecast["delivery_start_utc"], utc=True, errors="raise")
        )
        quantiles = forecast[["q10", "q50", "q90"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
    except (KeyError, OSError, ValueError) as exc:
        raise ExistingForecastArchiveError(
            f"{archive}: forecast CSV incomplet ou illisible."
        ) from exc
    expected_index = local_delivery_day_index(delivery, timezone=status.timezone)
    if not delivery_index.equals(expected_index):
        raise ExistingForecastArchiveError(
            f"{archive}: forecast ne couvre pas exactement {delivery_text}."
        )
    if int(summary.get("hours", -1)) != len(expected_index):
        raise ExistingForecastArchiveError(
            f"{archive}: nombre d'heures publié incohérent."
        )
    if (
        not np.isfinite(quantiles).all()
        or bool((quantiles[:, 0] > quantiles[:, 1]).any())
        or bool((quantiles[:, 1] > quantiles[:, 2]).any())
    ):
        raise ExistingForecastArchiveError(
            f"{archive}: quantiles incomplets ou croisés."
        )
    return archive


def _live_output_contract(
    status: ZoneStatus,
    *,
    project_root: str | Path,
) -> tuple[Path, str]:
    """Resolve the canonical live root and forecast name without guessing."""

    code = canonical_zone(status.code)
    root = Path(project_root).expanduser().resolve()
    if status.live_config is None:
        raise ExistingForecastArchiveError(
            f"{code}: configuration live absente; comparaison impossible."
        )
    config_path = Path(status.live_config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    try:
        config = load_yaml(config_path)
    except Exception as exc:
        raise ExistingForecastArchiveError(
            f"{code}: configuration live illisible: {config_path}"
        ) from exc
    live = config.get("live") if isinstance(config, Mapping) else None
    if not isinstance(live, Mapping):
        raise ExistingForecastArchiveError(
            f"{code}: la configuration live doit contenir un mapping 'live'."
        )
    raw_output_root = live.get("output_root")
    if raw_output_root in (None, ""):
        raise ExistingForecastArchiveError(f"{code}: live.output_root est absent.")
    output_root = Path(str(raw_output_root)).expanduser()
    if not output_root.is_absolute():
        output_root = config_path.parent / output_root
    output_root = output_root.resolve()
    try:
        output_root.relative_to(root)
    except ValueError as exc:
        raise ExistingForecastArchiveError(
            f"{code}: live.output_root doit rester dans le projet: {output_root}"
        ) from exc
    forecast_name = str(
        live.get("forecast_filename") or f"forecast_hourly_{code.lower()}.csv"
    ).strip()
    if not forecast_name or Path(forecast_name).name != forecast_name:
        raise ExistingForecastArchiveError(
            f"{code}: nom de forecast invalide dans la configuration live."
        )
    return output_root, forecast_name


def _latest_issued_delivery_day(
    status: ZoneStatus,
    *,
    project_root: str | Path,
) -> str:
    """Return the newest canonical issued-live day, never a replay/report."""

    code = canonical_zone(status.code)
    output_root, _forecast_name = _live_output_contract(
        status,
        project_root=project_root,
    )
    if not output_root.is_dir():
        raise ExistingForecastArchiveError(
            f"{code}: aucun dossier live publie dans {output_root}."
        )
    pattern = re.compile(
        rf"^{re.escape(code.lower())}_day_ahead_(\d{{4}}-\d{{2}}-\d{{2}})$"
    )
    candidates: list[date] = []
    for path in output_root.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        try:
            candidates.append(date.fromisoformat(match.group(1)))
        except ValueError:
            continue
    if not candidates:
        raise ExistingForecastArchiveError(
            f"{code}: aucun forecast live day-ahead canonique n'est publie."
        )
    return max(candidates).isoformat()


def _require_prediction_without_storm(
    manifest: Mapping[str, Any],
    *,
    archive: Path,
) -> None:
    """Fail closed unless the frozen candidate explicitly excludes Storm."""

    for field in ("storm_used_as_feature", "storm_loaded_for_prediction"):
        if manifest.get(field) is not False:
            raise ExistingForecastArchiveError(
                f"{archive}: run_manifest.{field} doit etre explicitement false."
            )
    if (
        "storm_used_for_prediction" in manifest
        and manifest.get("storm_used_for_prediction") is not False
    ):
        raise ExistingForecastArchiveError(
            f"{archive}: run_manifest.storm_used_for_prediction doit etre false."
        )
    inputs = manifest.get("prediction_inputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or any(not isinstance(item, str) or not item.strip() for item in inputs)
    ):
        raise ExistingForecastArchiveError(
            f"{archive}: prediction_inputs doit etre une liste explicite non vide."
        )
    if any("storm" in item.casefold() for item in inputs):
        raise ExistingForecastArchiveError(
            f"{archive}: Storm apparait dans les prediction_inputs."
        )


def load_latest_forecast_comparison(
    statuses: Sequence[ZoneStatus],
    *,
    project_root: str | Path,
    zones: Sequence[str],
    allow_mixed_delivery_days: bool = False,
    variant: str = "production",
) -> ForecastComparison:
    """Load the newest issued-live forecast for every requested zone.

    Every archive is first checked by :func:`validate_existing_forecast_archive`,
    including its complete SHA-256 coverage and exact DST-aware local delivery
    timeline.  For a common delivery date the UTC timelines must also match
    exactly across zones.  Different latest dates are refused unless the caller
    explicitly opts in; the returned frame still keeps a delivery-day column so
    the UI and report cannot hide that difference.
    """

    selected_variant = _normalize_forecast_variant(variant)
    requested = tuple(canonical_zone(zone) for zone in zones)
    if not requested:
        raise ValueError("Selectionnez au moins un pays pour la comparaison.")
    if len(set(requested)) != len(requested):
        raise ValueError("La selection de pays contient des doublons.")
    normalized_statuses = [canonical_zone(status.code) for status in statuses]
    if len(set(normalized_statuses)) != len(normalized_statuses):
        raise ExistingForecastArchiveError(
            "La liste de statuts contient une zone dupliquee."
        )
    status_by_zone = dict(zip(normalized_statuses, statuses, strict=True))
    missing_statuses = sorted(set(requested).difference(status_by_zone))
    if missing_statuses:
        raise ExistingForecastArchiveError(
            f"Statut de zone absent pour: {', '.join(missing_statuses)}."
        )
    blocked_statuses = [
        zone for zone in requested if not status_by_zone[zone].launchable
    ]
    if blocked_statuses:
        raise ExistingForecastArchiveError(
            "Comparaison refusee: contrat live non validable pour "
            + ", ".join(blocked_statuses)
            + "."
        )

    days = {
        zone: _latest_issued_delivery_day(
            status_by_zone[zone], project_root=project_root
        )
        for zone in requested
    }
    if len(set(days.values())) > 1 and not allow_mixed_delivery_days:
        raise MixedForecastDeliveryDaysError(days)

    archives: list[ForecastArchiveDataset] = []
    long_frames: list[pd.DataFrame] = []
    utc_timelines: dict[str, pd.DatetimeIndex] = {}
    for zone in requested:
        status = status_by_zone[zone]
        archive = validate_existing_forecast_archive(
            status,
            project_root=project_root,
            delivery_day=days[zone],
            pit_replay=False,
        )
        if archive is None:  # discovery and validation must be atomic in intent
            raise ExistingForecastArchiveError(
                f"{zone}: le dernier forecast publie a disparu pendant l'audit."
            )
        manifest = _archive_json(archive / "run_manifest.json")
        _require_prediction_without_storm(manifest, archive=archive)
        _output_root, forecast_name = _live_output_contract(
            status,
            project_root=project_root,
        )
        forecast_path = archive / forecast_name
        dataset = load_forecast_curve(
            forecast_path,
            timezone_name=status.timezone,
            variant=selected_variant,
        )
        utc = pd.DatetimeIndex(dataset.frame["timestamp"]).tz_convert("UTC")
        expected = local_delivery_day_index(
            date.fromisoformat(days[zone]),
            timezone=status.timezone,
        )
        if not utc.equals(expected):
            raise ExistingForecastArchiveError(
                f"{archive}: timeline transformee divergente apres chargement."
            )
        utc_timelines[zone] = utc
        local_labels = pd.DatetimeIndex(dataset.frame["timestamp"]).map(
            lambda value: value.strftime("%Y-%m-%d %H:%M %z")
        )
        long_frames.append(
            pd.DataFrame(
                {
                    "timestamp_utc": utc,
                    "local_delivery": local_labels,
                    "zone": zone,
                    "delivery_day": days[zone],
                    "timezone": status.timezone,
                    "P10": dataset.frame["P10"].to_numpy(dtype=float),
                    "P50": dataset.frame["P50"].to_numpy(dtype=float),
                    "P90": dataset.frame["P90"].to_numpy(dtype=float),
                }
            )
        )
        archives.append(
            ForecastArchiveDataset(
                zone=zone,
                timezone=status.timezone,
                delivery_day=days[zone],
                archive_path=archive,
                forecast_path=forecast_path,
                forecast_sha256=_archive_sha256(forecast_path),
                checksum_manifest_sha256=_archive_sha256(
                    archive / "artifact_checksums.json"
                ),
                frame=dataset.frame,
            )
        )

    same_day = len(set(days.values())) == 1
    timeline_aligned = same_day
    if same_day:
        reference_zone = requested[0]
        reference = utc_timelines[reference_zone]
        for zone in requested[1:]:
            if not utc_timelines[zone].equals(reference):
                raise ExistingForecastArchiveError(
                    "Les timelines UTC des forecasts ne sont pas identiques "
                    f"pour le meme jour ({reference_zone} vs {zone})."
                )
    # Refuse a stale snapshot if a newer directory appeared while the selected
    # archives were being hashed.  The next UI rerun will load the new latest
    # set instead of silently presenting a no-longer-current comparison.
    for zone in requested:
        latest_after_audit = _latest_issued_delivery_day(
            status_by_zone[zone], project_root=project_root
        )
        if latest_after_audit != days[zone]:
            raise ExistingForecastArchiveError(
                f"{zone}: un forecast plus recent ({latest_after_audit}) est "
                "apparu pendant l'audit; rechargez la comparaison."
            )
    comparison_frame = pd.concat(long_frames, ignore_index=True)
    comparison_frame = comparison_frame.sort_values(
        ["timestamp_utc", "zone"], kind="stable"
    ).reset_index(drop=True)
    return ForecastComparison(
        archives=tuple(archives),
        frame=comparison_frame,
        timeline_aligned=timeline_aligned,
        variant=selected_variant,
    )


def build_dispatch_command(
    status: ZoneStatus,
    *,
    project_root: str | Path,
    registry_path: str | Path,
    python_executable: str | Path | None = None,
    delivery_day: str | date | None = None,
    data_as_of: str | None = None,
    device: str | None = None,
    threads: int | None = None,
    workers: int | None = None,
    local_files_only: bool = False,
    pit_replay: bool = False,
) -> list[str]:
    """Build a dispatcher argv list; no shell parsing is ever involved."""

    if not status.launchable:
        detail = "; ".join(status.blockers) or "bundle non prêt"
        raise ZoneBundleError(f"{status.code}: lancement refusé: {detail}")
    code = canonical_zone(status.code)
    root = Path(project_root).expanduser().resolve()
    dispatcher = (root / "run_mkonline_live_zone.py").resolve()
    if not dispatcher.is_file():
        raise FileNotFoundError(f"Dispatcher absent: {dispatcher}")
    registry = Path(registry_path).expanduser().resolve()
    if not registry.is_file():
        raise FileNotFoundError(f"Registre absent: {registry}")
    executable = Path(python_executable or sys.executable).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Interpréteur Python absent: {executable}")
    if device not in (None, "auto", "cpu", "cuda"):
        raise ValueError("device doit être auto, cpu ou cuda")
    for name, value in (("threads", threads), ("workers", workers)):
        if value is not None and int(value) < 1:
            raise ValueError(f"{name} doit être supérieur ou égal à 1")

    command = [
        str(executable),
        str(dispatcher),
        "--registry",
        str(registry),
        "--zone",
        code,
    ]
    delivery = _iso_date(delivery_day, name="delivery_day")
    if delivery:
        command.extend(("--delivery-day", delivery))
    if data_as_of:
        # This is passed as one argv item.  It is not interpreted by a shell.
        command.extend(("--data-as-of", str(data_as_of).strip()))
    if device:
        command.extend(("--device", device))
    if threads is not None:
        command.extend(("--threads", str(int(threads))))
    if workers is not None:
        command.extend(("--workers", str(int(workers))))
    if local_files_only:
        command.append("--local-files-only")
    if pit_replay:
        command.append("--pit-replay")
    return command


def start_dispatch_process(
    command: Sequence[str],
    *,
    zone: str,
    project_root: str | Path,
    log_dir: str | Path,
) -> ForecastProcess:
    """Start one forecast in the background and redirect all output to a log."""

    code = canonical_zone(zone)
    if not command or any(not isinstance(item, str) for item in command):
        raise ValueError("La commande doit être une liste d'arguments non vide.")
    root = Path(project_root).expanduser().resolve()
    logs = Path(log_dir).expanduser().resolve()
    try:
        logs.relative_to(root)
    except ValueError as exc:
        raise ValueError("Le dossier de logs doit rester dans le projet.") from exc
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = logs / f"{code.lower()}_{stamp}_{uuid4().hex[:8]}.log"
    creationflags = 0
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    blocked_proxy_names = _local_blackhole_proxy_names()
    child_environment: dict[str, str] | None = None
    if blocked_proxy_names:
        if _is_codex_sandbox_environment():
            raise RuntimeError(
                "L'application appartient encore au sandbox Codex et herite du "
                "proxy 127.0.0.1:9 "
                f"({', '.join(blocked_proxy_names)}). Quittez completement Codex, "
                "puis utilisez launch_forecast_app.cmd depuis l'Explorateur Windows."
            )
        child_environment = dict(os.environ)
        for name in blocked_proxy_names:
            child_environment.pop(name, None)
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        stream.write(
            "Commande (argv, shell=False):\n"
            + json.dumps(list(command), ensure_ascii=False)
            + "\n\n"
        )
        stream.flush()
        process = subprocess.Popen(
            list(command),
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            shell=False,
            creationflags=creationflags,
            env=child_environment,
        )
    return ForecastProcess(
        zone=code,
        command=tuple(command),
        log_path=log_path,
        started_at=datetime.now(timezone.utc),
        process=process,
    )


def _is_local_blackhole_proxy(value: str | None) -> bool:
    """Return true only for the loopback port-9 proxy injected by a sandbox."""

    if not value or not value.strip():
        return False
    candidate = value.strip()
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    try:
        parsed = urlsplit(candidate)
        return parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port == 9
    except ValueError:
        return False


def _local_blackhole_proxy_names(
    source: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Identify the known sandbox proxy without changing the environment."""

    environment = dict(os.environ if source is None else source)
    blocked: list[str] = []
    for name in _REQUESTS_PROXY_ENV_NAMES:
        if _is_local_blackhole_proxy(environment.get(name)):
            blocked.append(name)
    return tuple(blocked)


def _is_codex_sandbox_environment(source: Mapping[str, str] | None = None) -> bool:
    environment = os.environ if source is None else source
    return environment.get("CODEX_SANDBOX_NETWORK_DISABLED") == "1" or bool(
        environment.get("CODEX_THREAD_ID")
    )


def launch_zone_forecast(
    *,
    zone: str,
    project_root: str | Path,
    registry_path: str | Path,
    log_dir: str | Path,
    python_executable: str | Path | None = None,
    delivery_day: str | date | None = None,
    data_as_of: str | None = None,
    device: str | None = None,
    threads: int | None = None,
    workers: int | None = None,
    local_files_only: bool = False,
    pit_replay: bool = False,
) -> ForecastProcess | ForecastSkip:
    """Re-audit immediately before dispatch, then launch the zone safely."""

    code = canonical_zone(zone)
    status = inspect_zone_statuses(registry_path, zones=(code,))[0]
    command = build_dispatch_command(
        status,
        project_root=project_root,
        registry_path=registry_path,
        python_executable=python_executable,
        delivery_day=delivery_day,
        data_as_of=data_as_of,
        device=device,
        threads=threads,
        workers=workers,
        local_files_only=local_files_only,
        pit_replay=pit_replay,
    )
    delivery = _iso_date(delivery_day, name="delivery_day")
    if delivery is not None:
        existing = validate_existing_forecast_archive(
            status,
            project_root=project_root,
            delivery_day=delivery,
            pit_replay=pit_replay,
        )
        if existing is not None:
            return ForecastSkip(
                zone=code,
                delivery_day=delivery,
                archive_path=existing,
            )
    return start_dispatch_process(
        command,
        zone=code,
        project_root=project_root,
        log_dir=log_dir,
    )


def read_log_tail(path: str | Path, *, max_chars: int = 30_000) -> str:
    log_path = Path(path)
    if not log_path.is_file():
        return ""
    with log_path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - max_chars), os.SEEK_SET)
        text = stream.read().decode("utf-8", errors="replace")
    return ("…\n" if size > max_chars else "") + text


def _infer_zone(directory: Path) -> str | None:
    for json_name in ("live_run_summary.json", "run_manifest.json"):
        path = directory / json_name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_zone = payload.get("zone")
        if raw_zone:
            try:
                return canonical_zone(str(raw_zone))
            except ZoneBundleError:
                pass
    match = re.search(
        r"(?:^|[_-])(fr|de|be|nl|es)(?:[_-]|$)",
        directory.name,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


def list_run_artifacts(
    live_root: str | Path,
    *,
    zones: Iterable[str] = APP_ZONES,
    limit: int = 50,
    sealed_benchmark_root: str | Path | None = None,
) -> list[RunArtifact]:
    """List live/report directories and explicit sealed zone benchmarks.

    ``sealed_benchmark_root`` is opt-in and only admits the canonical
    ``chronos2_hourly_<zone>_sealed_benchmark_v1`` directories.  It never
    scans arbitrary training runs.
    """

    root = Path(live_root).expanduser().resolve()
    if not root.is_dir():
        return []
    allowed = {canonical_zone(zone) for zone in zones}
    directories: set[Path] = set()
    for pattern in (
        "statistics_history_hourly.csv.gz",
        "live_run_summary.json",
        "*.html",
    ):
        directories.update(path.parent for path in root.rglob(pattern))
    if sealed_benchmark_root is not None:
        benchmark_root = Path(sealed_benchmark_root).expanduser().resolve()
        for zone in allowed:
            candidate = (
                benchmark_root
                / f"chronos2_hourly_{zone.lower()}_sealed_benchmark_v1"
            )
            if candidate.is_dir():
                directories.add(candidate)
    artifacts: list[RunArtifact] = []
    for directory in directories:
        zone = _infer_zone(directory)
        if zone not in allowed:
            continue
        reports = sorted(
            directory.glob("*.html"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        statistics = directory / "statistics_history_hourly.csv.gz"
        forecasts = sorted(directory.glob(f"forecast_hourly_{zone.lower()}*.csv"))
        relevant = [
            path
            for path in (
                reports[0] if reports else None,
                statistics if statistics.is_file() else None,
                forecasts[0] if forecasts else None,
                directory / "live_run_summary.json",
            )
            if path is not None and path.is_file()
        ]
        if not relevant:
            continue
        modified = max(path.stat().st_mtime for path in relevant)
        parts_lower = {part.lower() for part in directory.parts}
        kind = (
            "benchmark scellé"
            if directory.name.endswith("_sealed_benchmark_v1")
            else "rapport"
            if "_reports" in parts_lower
            else "replay"
            if "_replays" in parts_lower
            else "run live"
        )
        artifacts.append(
            RunArtifact(
                zone=zone,
                kind=kind,
                directory=directory,
                modified_at=datetime.fromtimestamp(modified, tz=timezone.utc),
                report_path=reports[0] if reports else None,
                statistics_path=statistics if statistics.is_file() else None,
                forecast_path=forecasts[0] if forecasts else None,
            )
        )
    return sorted(
        artifacts,
        key=lambda artifact: artifact.modified_at,
        reverse=True,
    )[: max(0, int(limit))]


def _first_existing_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def load_statistics_history(
    path: str | Path,
    *,
    variant: str = "production",
) -> StatisticsDataset:
    """Load a reporting-only Statistics artifact with explicit Storm semantics."""

    statistics_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(statistics_path, compression="infer")
    required = {"delivery_start_utc", "actual"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Colonnes Statistics absentes: {missing}")
    delivery = pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="raise")
    if bool(delivery.duplicated().any()):
        raise ValueError("La timeline Statistics contient des doublons.")
    selected_variant = _normalize_forecast_variant(variant)
    candidate_priority = {
        "production": CANDIDATE_COLUMN_PRIORITY,
        "autonomous": ("residual_corrected__q50",),
        "mkonline_blend": ("mkonline_blend__q50",),
    }[selected_variant]
    candidate = _first_existing_column(frame, candidate_priority)
    if candidate is None:
        raise ValueError(
            f"Variante {selected_variant} absente de l'historique Statistics."
        )

    audit_path = statistics_path.parent / "statistics_history_audit.json"
    audit: dict[str, Any] = {}
    if audit_path.is_file():
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            audit = payload
    official = "storm_dashboard_official__q50"
    official_active = audit.get("storm_primary_report_benchmark") == official
    if official in frame.columns and not official_active:
        # Do not silently label an unaudited column as the official dashboard.
        benchmark_candidates = tuple(
            name for name in BENCHMARK_COLUMN_PRIORITY if name != official
        )
    else:
        benchmark_candidates = BENCHMARK_COLUMN_PRIORITY
    benchmark = _first_existing_column(frame, benchmark_candidates)
    candidate_label = {
        "mkonline_blend__q50": "Chronos-2 + MKOnline",
        "residual_corrected__q50": "Chronos-2 corrigé",
        "ensemble__q50": "Chronos-2 ensemble",
        "chronos2__q50": "Chronos-2",
    }.get(candidate, candidate.removesuffix("__q50"))
    benchmark_label = None
    if benchmark == official:
        benchmark_label = "Storm officiel dashboard"
    elif benchmark:
        benchmark_label = "Storm D-1 disponible à 08:00"

    selected = pd.DataFrame(
        {
            "timestamp": delivery,
            "actual": pd.to_numeric(frame["actual"], errors="coerce"),
            "candidate": pd.to_numeric(frame[candidate], errors="coerce"),
        }
    )
    if benchmark:
        selected["benchmark"] = pd.to_numeric(frame[benchmark], errors="coerce")
    else:
        selected["benchmark"] = np.nan
    selected = selected.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return StatisticsDataset(
        path=statistics_path,
        frame=selected,
        candidate_column=candidate,
        candidate_label=candidate_label,
        benchmark_column=benchmark,
        benchmark_label=benchmark_label,
        scope_note=str(audit.get("report_scope_note") or "").strip() or None,
    )


@dataclass(frozen=True)
class _PerformanceCandidate:
    archive_path: Path
    pit_replay: bool
    delivery_day: str
    prefix_end_local: str
    max_timestamp_utc: pd.Timestamp
    row_count: int


def _performance_candidate(
    archive_path: Path,
    *,
    pit_replay: bool,
    delivery_day: str,
    timezone_name: str,
) -> _PerformanceCandidate:
    statistics_path = archive_path / "statistics_history_hourly.csv.gz"
    audit_path = archive_path / "statistics_history_audit.json"
    audit = _archive_json(audit_path)
    try:
        timeline_frame = pd.read_csv(
            statistics_path,
            compression="infer",
            usecols=["delivery_start_utc"],
        )
        delivery = pd.DatetimeIndex(
            pd.to_datetime(
                timeline_frame["delivery_start_utc"],
                utc=True,
                errors="raise",
            )
        )
    except (KeyError, OSError, ValueError) as exc:
        raise ExistingForecastArchiveError(
            f"{archive_path}: historique Statistics illisible."
        ) from exc
    if delivery.empty:
        raise ExistingForecastArchiveError(
            f"{archive_path}: historique Statistics vide."
        )
    max_timestamp_utc = pd.Timestamp(delivery.max())
    observed_prefix = max_timestamp_utc.tz_convert(timezone_name).date().isoformat()
    raw_prefix = audit.get("statistics_prefix_end_local")
    if raw_prefix in (None, ""):
        prefix_end_local = observed_prefix
    else:
        try:
            prefix_end_local = date.fromisoformat(str(raw_prefix)).isoformat()
        except ValueError as exc:
            raise ExistingForecastArchiveError(
                f"{archive_path}: statistics_prefix_end_local invalide."
            ) from exc
    return _PerformanceCandidate(
        archive_path=archive_path,
        pit_replay=pit_replay,
        delivery_day=delivery_day,
        prefix_end_local=prefix_end_local,
        max_timestamp_utc=max_timestamp_utc,
        row_count=len(timeline_frame),
    )


def _canonical_performance_candidates(
    status: ZoneStatus,
    *,
    project_root: str | Path,
) -> tuple[_PerformanceCandidate, ...]:
    code = canonical_zone(status.code)
    output_root, _forecast_name = _live_output_contract(
        status,
        project_root=project_root,
    )
    if not output_root.is_dir():
        raise ExistingForecastArchiveError(
            f"{code}: aucun dossier live canonique n'est publie."
        )
    pattern = re.compile(
        rf"^{re.escape(code.lower())}_day_ahead_(\d{{4}}-\d{{2}}-\d{{2}})$"
    )
    candidates: list[_PerformanceCandidate] = []
    for archive_root, pit_replay in (
        (output_root, False),
        (output_root / "_replays", True),
    ):
        if not archive_root.is_dir():
            continue
        try:
            archive_root.resolve().relative_to(output_root)
        except ValueError as exc:
            raise ExistingForecastArchiveError(
                f"{code}: racine d'archive Statistics hors du projet live."
            ) from exc
        for raw_archive in archive_root.iterdir():
            match = pattern.fullmatch(raw_archive.name)
            if match is None or not raw_archive.is_dir():
                continue
            try:
                delivery_day = date.fromisoformat(match.group(1)).isoformat()
            except ValueError:
                continue
            archive_path = raw_archive.resolve()
            if archive_path.parent != archive_root.resolve():
                raise ExistingForecastArchiveError(
                    f"{code}: archive Statistics hors de sa racine canonique."
                )
            statistics_path = archive_path / "statistics_history_hourly.csv.gz"
            audit_path = archive_path / "statistics_history_audit.json"
            if not statistics_path.is_file() and not audit_path.is_file():
                continue
            if not statistics_path.is_file() or not audit_path.is_file():
                raise ExistingForecastArchiveError(
                    f"{archive_path}: paire Statistics/audit incomplete."
                )
            candidates.append(
                _performance_candidate(
                    archive_path,
                    pit_replay=pit_replay,
                    delivery_day=delivery_day,
                    timezone_name=status.timezone,
                )
            )
    if not candidates:
        raise ExistingForecastArchiveError(
            f"{code}: aucun historique Statistics audite n'est publie."
        )
    return tuple(candidates)


def _validated_performance_artifact(
    status: ZoneStatus,
    candidate: _PerformanceCandidate,
    *,
    project_root: str | Path,
    variant: str,
) -> tuple[PerformanceArtifact, StatisticsDataset]:
    code = canonical_zone(status.code)
    validated_archive = validate_existing_forecast_archive(
        status,
        project_root=project_root,
        delivery_day=candidate.delivery_day,
        pit_replay=candidate.pit_replay,
    )
    if validated_archive is None or validated_archive != candidate.archive_path:
        raise ExistingForecastArchiveError(
            f"{candidate.archive_path}: archive Statistics introuvable apres selection."
        )
    statistics_path = validated_archive / "statistics_history_hourly.csv.gz"
    audit_path = validated_archive / "statistics_history_audit.json"
    audit = _archive_json(audit_path)
    if audit.get("statistics_history_path") != statistics_path.name:
        raise ExistingForecastArchiveError(
            f"{validated_archive}: statistics_history_path non canonique."
        )
    if audit.get("statistics_audit_path") != audit_path.name:
        raise ExistingForecastArchiveError(
            f"{validated_archive}: statistics_audit_path non canonique."
        )
    current_delivery_day = audit.get("current_delivery_day_local")
    if current_delivery_day not in (None, candidate.delivery_day):
        raise ExistingForecastArchiveError(
            f"{validated_archive}: current_delivery_day_local incoherent."
        )
    expected_sha = str(audit.get("statistics_history_sha256") or "").lower()
    actual_sha = _archive_sha256(statistics_path)
    if (
        len(expected_sha) != 64
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
        or expected_sha != actual_sha
    ):
        raise ExistingForecastArchiveError(
            f"{validated_archive}: statistics_history_sha256 divergent."
        )
    try:
        dataset = load_statistics_history(statistics_path, variant=variant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ExistingForecastArchiveError(
            f"{validated_archive}: historique Statistics invalide pour {variant}."
        ) from exc
    if len(dataset.frame) != candidate.row_count:
        raise ExistingForecastArchiveError(
            f"{validated_archive}: nombre de lignes Statistics instable."
        )
    raw_total = audit.get("n_total_statistics_hours")
    if isinstance(raw_total, bool):
        raise ExistingForecastArchiveError(
            f"{validated_archive}: n_total_statistics_hours invalide."
        )
    try:
        audited_total = int(raw_total)
    except (TypeError, ValueError) as exc:
        raise ExistingForecastArchiveError(
            f"{validated_archive}: n_total_statistics_hours invalide."
        ) from exc
    if audited_total != len(dataset.frame):
        raise ExistingForecastArchiveError(
            f"{validated_archive}: n_total_statistics_hours divergent."
        )
    observed_max = pd.Timestamp(dataset.frame["timestamp"].max())
    observed_prefix = observed_max.tz_convert(status.timezone).date().isoformat()
    raw_prefix = audit.get("statistics_prefix_end_local")
    if raw_prefix not in (None, ""):
        try:
            audited_prefix = date.fromisoformat(str(raw_prefix)).isoformat()
        except ValueError as exc:
            raise ExistingForecastArchiveError(
                f"{validated_archive}: statistics_prefix_end_local invalide."
            ) from exc
        if audited_prefix != observed_prefix:
            raise ExistingForecastArchiveError(
                f"{validated_archive}: statistics_prefix_end_local divergent."
            )
    elif candidate.prefix_end_local != observed_prefix:
        raise ExistingForecastArchiveError(
            f"{validated_archive}: couverture Statistics instable."
        )
    raw_complete = audit.get("statistics_complete", False)
    if not isinstance(raw_complete, bool):
        raise ExistingForecastArchiveError(
            f"{validated_archive}: statistics_complete doit etre booleen."
        )
    raw_missing = audit.get("missing_realized_days", [])
    if not isinstance(raw_missing, list):
        raise ExistingForecastArchiveError(
            f"{validated_archive}: missing_realized_days doit etre une liste."
        )
    missing_realized_days: list[str] = []
    for raw_day in raw_missing:
        try:
            missing_realized_days.append(date.fromisoformat(str(raw_day)).isoformat())
        except ValueError as exc:
            raise ExistingForecastArchiveError(
                f"{validated_archive}: missing_realized_days contient une date invalide."
            ) from exc
    modified_at = datetime.fromtimestamp(
        max(statistics_path.stat().st_mtime, audit_path.stat().st_mtime),
        tz=timezone.utc,
    )
    artifact = PerformanceArtifact(
        zone=code,
        timezone=status.timezone,
        archive_kind="pit_replay" if candidate.pit_replay else "live_day_ahead",
        delivery_day=candidate.delivery_day,
        archive_path=validated_archive,
        statistics_path=statistics_path,
        audit_path=audit_path,
        statistics_prefix_end_local=observed_prefix,
        n_total_statistics_hours=len(dataset.frame),
        statistics_complete=raw_complete,
        missing_realized_days=tuple(missing_realized_days),
        statistics_sha256=actual_sha,
        modified_at=modified_at,
    )
    return artifact, dataset


def load_best_statistics_history(
    status: ZoneStatus,
    *,
    project_root: str | Path,
    variant: str = "production",
) -> tuple[PerformanceArtifact, StatisticsDataset]:
    """Load the freshest checksum-sealed Statistics history for one zone.

    Freshness is based only on scored coverage, never on filesystem mtime.
    The highest-ranked archive is validated fail-closed; a corrupt fresher
    artifact is not silently replaced with an older history.
    """

    selected_variant = _normalize_forecast_variant(variant)
    candidates = _canonical_performance_candidates(
        status,
        project_root=project_root,
    )
    selected = max(
        candidates,
        key=lambda item: (
            date.fromisoformat(item.prefix_end_local),
            item.max_timestamp_utc.value,
            item.row_count,
            date.fromisoformat(item.delivery_day),
            not item.pit_replay,
        ),
    )
    return _validated_performance_artifact(
        status,
        selected,
        project_root=project_root,
        variant=selected_variant,
    )


def _normalize_forecast_variant(value: str) -> str:
    variant = str(value).strip().lower()
    aliases = {
        "production": "production",
        "autonomous": "autonomous",
        "autonome": "autonomous",
        "mkonline_blend": "mkonline_blend",
        "blend": "mkonline_blend",
    }
    try:
        return aliases[variant]
    except KeyError as exc:
        raise ValueError(
            "variant doit etre production, autonomous ou mkonline_blend."
        ) from exc


def load_forecast_curve(
    path: str | Path,
    *,
    timezone_name: str,
    variant: str = "production",
) -> ForecastDataset:
    """Load one immutable forecast for display, with strict quantile guards."""

    forecast_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(forecast_path)
    selected_variant = _normalize_forecast_variant(variant)
    quantile_columns = {
        "production": ("q10", "q50", "q90"),
        "autonomous": (
            "residual_corrected__q10",
            "residual_corrected__q50",
            "residual_corrected__q90",
        ),
        "mkonline_blend": (
            "mkonline_blend__q10",
            "mkonline_blend__q50",
            "mkonline_blend__q90",
        ),
    }[selected_variant]
    required = {"delivery_start_utc", *quantile_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Variante {selected_variant} indisponible; colonnes absentes: {missing}"
        )
    delivery = pd.DatetimeIndex(
        pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="raise")
    )
    if delivery.has_duplicates or not delivery.is_monotonic_increasing:
        raise ValueError("La timeline forecast doit être unique et triée.")
    quantiles = frame.loc[:, list(quantile_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    values = quantiles.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Le forecast contient des quantiles non finis.")
    if bool((values[:, 0] > values[:, 1]).any()) or bool(
        (values[:, 1] > values[:, 2]).any()
    ):
        raise ValueError("Le forecast contient des quantiles croisés.")
    if "forecast_origin_utc" in frame:
        origin = pd.DatetimeIndex(
            pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="raise")
        )
        if len(origin) != len(delivery) or bool((origin >= delivery).any()):
            raise ValueError("L'origine du forecast n'est pas causale.")
    display = pd.DataFrame(
        {
            "timestamp": delivery.tz_convert(timezone_name),
            "P10": values[:, 0],
            "P50": values[:, 1],
            "P90": values[:, 2],
        }
    )
    return ForecastDataset(path=forecast_path, frame=display)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float | None]:
    valid = np.isfinite(actual) & np.isfinite(forecast)
    actual = actual[valid]
    forecast = forecast[valid]
    if not len(actual):
        return {key: None for key, _label, _higher in STATISTIC_DEFINITIONS}
    error = forecast - actual
    actual_variance = float(np.var(actual, ddof=0))
    error_variance = float(np.var(error, ddof=0))
    total = float(np.sum(np.square(actual - np.mean(actual))))
    residual = float(np.sum(np.square(error)))
    correlation = (
        float(np.corrcoef(actual, forecast)[0, 1])
        if len(actual) >= 2
        and float(np.std(actual, ddof=0)) > 1e-12
        and float(np.std(forecast, ddof=0)) > 1e-12
        else math.nan
    )
    nonzero = np.abs(actual) > MAPE_DENOMINATOR_EPSILON
    mape = (
        100.0 * float(np.mean(np.abs(error[nonzero]) / np.abs(actual[nonzero])))
        if bool(nonzero.any())
        else math.nan
    )
    return {
        "mae": _finite(np.mean(np.abs(error))),
        "rmse": _finite(np.sqrt(np.mean(np.square(error)))),
        "mape": _finite(mape),
        "explained_variance": _finite(
            1.0 - error_variance / actual_variance
            if actual_variance > 1e-12
            else math.nan
        ),
        "r2": _finite(1.0 - residual / total if total > 1e-12 else math.nan),
        "std_error": _finite(np.std(error, ddof=0)),
        "correlation": _finite(correlation),
    }


def _outcome(candidate: Any, benchmark: Any, *, higher_is_better: bool) -> str | None:
    left = _finite(candidate)
    right = _finite(benchmark)
    if left is None or right is None:
        return None
    if math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12):
        return "tie"
    won = left > right if higher_is_better else left < right
    return "win" if won else "loss"


def build_statistics_view(
    dataset: StatisticsDataset,
    *,
    timezone_name: str,
    sample: str = "daily",
) -> StatisticsView:
    """Compute the seven report Statistics and a per-metric win rate vs Storm."""

    if sample not in SAMPLE_FREQUENCIES:
        raise ValueError(f"Échantillonnage inconnu: {sample}")
    work = dataset.frame.copy()
    work["local_timestamp"] = work["timestamp"].dt.tz_convert(timezone_name)
    # Periods intentionally use local civil time so DST delivery days remain
    # 23/24/25-hour market days.
    local_naive = work["local_timestamp"].dt.tz_localize(None)
    if sample == "daily":
        work["period"] = local_naive.dt.strftime("%Y-%m-%d")
    else:
        work["period"] = local_naive.dt.to_period(SAMPLE_FREQUENCIES[sample]).astype(str)

    period_rows: list[dict[str, Any]] = []
    has_benchmark = dataset.benchmark_column is not None
    for period, block in work.groupby("period", sort=True):
        columns = ["actual", "candidate"] + (["benchmark"] if has_benchmark else [])
        paired = block.dropna(subset=columns)
        candidate_metrics = _metrics(
            paired["actual"].to_numpy(dtype=float),
            paired["candidate"].to_numpy(dtype=float),
        )
        benchmark_metrics = (
            _metrics(
                paired["actual"].to_numpy(dtype=float),
                paired["benchmark"].to_numpy(dtype=float),
            )
            if has_benchmark
            else {key: None for key, _label, _higher in STATISTIC_DEFINITIONS}
        )
        row: dict[str, Any] = {"period": str(period), "n": int(len(paired))}
        for key, _label, higher in STATISTIC_DEFINITIONS:
            row[f"candidate_{key}"] = candidate_metrics[key]
            row[f"benchmark_{key}"] = benchmark_metrics[key]
            row[f"outcome_{key}"] = _outcome(
                candidate_metrics[key],
                benchmark_metrics[key],
                higher_is_better=higher,
            )
        period_rows.append(row)
    periods = pd.DataFrame(period_rows)

    paired_all = work.dropna(
        subset=["actual", "candidate"] + (["benchmark"] if has_benchmark else [])
    )
    candidate_overall = _metrics(
        paired_all["actual"].to_numpy(dtype=float),
        paired_all["candidate"].to_numpy(dtype=float),
    )
    benchmark_overall = (
        _metrics(
            paired_all["actual"].to_numpy(dtype=float),
            paired_all["benchmark"].to_numpy(dtype=float),
        )
        if has_benchmark
        else {key: None for key, _label, _higher in STATISTIC_DEFINITIONS}
    )
    summary_rows: list[dict[str, Any]] = []
    for key, label, _higher in STATISTIC_DEFINITIONS:
        outcomes = periods.get(f"outcome_{key}", pd.Series(dtype=object)).dropna()
        wins = int((outcomes == "win").sum())
        ties = int((outcomes == "tie").sum())
        losses = int((outcomes == "loss").sum())
        comparable = wins + ties + losses
        summary_rows.append(
            {
                "metric": key,
                "label": label,
                "candidate": candidate_overall[key],
                "benchmark": benchmark_overall[key],
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "comparable_periods": comparable,
                "win_rate": wins / comparable if comparable else None,
            }
        )
    time_series = work.loc[
        :, ["timestamp", "actual", "candidate", "benchmark"]
    ].reset_index(drop=True)
    return StatisticsView(
        sample=sample,
        periods=periods,
        summary=pd.DataFrame(summary_rows),
        time_series=time_series,
    )
