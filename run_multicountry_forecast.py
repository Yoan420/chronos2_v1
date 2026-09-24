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
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from chronos2_hourly.app_service import (
    APP_ZONES,
    ForecastProcess,
    ForecastSkip,
    ZoneStatus,
    build_dispatch_command,
    inspect_zone_statuses,
    launch_zone_forecast,
    normalize_residual_load_source,
    read_log_tail,
    validate_existing_forecast_archive,
)
from chronos2_hourly.zone_live import canonical_zone
from chronos2_hourly.reporting import write_hourly_html_report
from chronos2_hourly.live_history import update_live_statistics_history
from chronos2_hourly.storm_dashboard import (
    STORM_DASHBOARD_COLUMN,
    STORM_DASHBOARD_LIVE_FORECAST_ARTIFACT,
    STORM_DASHBOARD_LIVE_FORECAST_AUDIT,
    build_dashboard_comparator,
    fetch_native_dashboard_snapshot,
    normalize_native_dashboard_series,
    statistics_contract,
    storm_dashboard_series,
)
from chronos2_modular.common import ZoneData, deep_get, load_yaml, set_reproducibility
from chronos2_modular.saturn import (
    create_saturn_client,
    fetch_saturn_series_from_client,
)
from chronos2_exogenous.activation_contract import (
    ActivationResolution,
    ExogenousActivationContract,
    load_activation_contract,
)
from chronos2_exogenous.production import (
    BASE_MODEL as LORA_BASE_MODEL,
    OUTPUT_MODEL as LORA_AUTONOMOUS_MODEL,
    ExogenousRunResult,
    ExogenousProductionError,
    load_promoted_shadow_history,
    run_registered_candidate,
    validate_registered_candidate_run,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = PROJECT_ROOT / "chronos2_hourly_live_zones.yaml"
DEFAULT_LOG_DIR = PROJECT_ROOT / "runs" / "live" / "_launcher_logs"
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "runs" / "exports"
DEFAULT_KALMAN_CONFIG = PROJECT_ROOT / "config" / "kalman_operational.yaml"
DEFAULT_LORA_ACTIVATION_CONFIG = (
    PROJECT_ROOT / "config" / "chronos2_exogenous_activation_v1.yaml"
)
DEFAULT_KALMAN_WEATHER_CONFIG = (
    PROJECT_ROOT / "config" / "kalman_weather_operational.yaml"
)
DEFAULT_KALMAN_HYBRID_CONFIG = (
    PROJECT_ROOT / "config" / "kalman_hybrid_operational.yaml"
)
DEFAULT_RESIDUAL_LOAD_BUNDLE_ROOT = (
    PROJECT_ROOT / "runs" / "experiments" / "chronos2_residual_load" / "upstream"
)
LOCAL_TIMEZONE = ZoneInfo("Europe/Paris")
FORECAST_MODES = ("production", "autonomous", "blend", "both", "all")
BLEND_ZONES = frozenset({"FR", "NL"})
KALMAN_MODEL = "residual_kalman"
KALMAN_WEATHER_MODEL = "residual_kalman_weather"
KALMAN_HYBRID_MODEL = "residual_kalman_hybrid"
KALMAN_UPSTREAM_MODEL = "residual_corrected"
KALMAN_VARIANTS: Mapping[str, str] = {
    "kalman": KALMAN_MODEL,
    "kalman_weather": KALMAN_WEATHER_MODEL,
    "kalman_hybrid": KALMAN_HYBRID_MODEL,
}
KALMAN_EXACT_ROLLING_VARIANTS = frozenset(KALMAN_VARIANTS)
KALMAN_WEATHER_HISTORY_START_DAY = date(2024, 6, 30)

# Report-only post-auction observations.  The ENTSO-E targets remain the
# canonical training/evaluation source.  These EPEX series may only fill a
# missing observation hours after their preceding overlap and the canonical
# hours around internal gaps have been proven identical. They are never exposed to the
# forecast runner or to a filter state before the forecast is frozen.
POST_AUCTION_OBSERVED_SERIES_BY_ZONE: dict[str, str] = {
    "FR": "power.price.fr.euromwh.h.obs.epex",
    "DE": "power.price.de.euromwh.h.obs.epex",
    "BE": "power.price.be.euromwh.h.obs.epex",
    "NL": "power.price.nl.euromwh.h.obs.epex",
}
POST_AUCTION_VALIDATION_DAYS = 7
POST_AUCTION_MINIMUM_PAIRED_HOURS = 7 * 24
POST_AUCTION_EQUIVALENCE_ATOL = 1e-9


class PostAuctionObservationDivergenceError(ValueError):
    """Two fresh report-only observation sources disagree on finite hours."""

    def __init__(self, *, zone, timestamp, canonical, post_auction, maximum,
                 divergent_hours, compared_hours, canonical_series,
                 post_auction_series, extracted_at_utc):
        self.diagnostic = {
            "zone": zone, "timestamp_utc": str(timestamp),
            "canonical_value_eur_mwh": float(canonical),
            "post_auction_value_eur_mwh": float(post_auction),
            "maximum_absolute_difference_eur_mwh": float(maximum),
            "divergent_hours": int(divergent_hours), "compared_hours": int(compared_hours),
            "canonical_series": canonical_series, "post_auction_series": post_auction_series,
            "extracted_at_utc": str(extracted_at_utc),
        }
        super().__init__(
            f"{zone}: la source post-enchere diverge de la cible canonique "
            f"(ecart max={maximum:.12g} EUR/MWh; heure UTC={timestamp}; "
            f"canonique={canonical:.12g}, post-enchere={post_auction:.12g} EUR/MWh; "
            f"{divergent_hours}/{compared_hours} heures divergentes)."
        )


@dataclass(frozen=True)
class ForecastExport:
    """A user-facing view derived from an immutable production archive."""

    variant: str
    source_model: str
    csv_path: Path
    report_path: Path
    source_archive: Path
    residual_load_source: str = "saturn"
    statistics_end_local: str | None = None
    kalman_audit_path: Path | None = None
    lora_audit_path: Path | None = None


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
    residual_load_source: str = "saturn"
    residual_load_bundle_manifest: Path | None = None

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


def _residual_load_cutoff(delivery_day: str) -> pd.Timestamp:
    cutoff_naive = (
        pd.Timestamp(date.fromisoformat(delivery_day))
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=8)
    )
    return cutoff_naive.tz_localize(
        LOCAL_TIMEZONE,
        ambiguous="raise",
        nonexistent="raise",
    )


def _residual_load_provider_settings(project_root: Path) -> tuple[str, str]:
    config_path = project_root / "chronos2_hourly_fr_residual_v1.yaml"
    config = load_yaml(config_path)
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise TypeError(f"{config_path}: data doit etre un mapping.")
    saturn_url = str(data.get("saturn_url") or "").strip()
    saturn_author = str(
        os.getenv("SATURN_AUTHOR") or data.get("saturn_author") or ""
    ).strip()
    if not saturn_url or not saturn_author:
        raise ValueError(
            f"{config_path}: acces Saturn observe incomplet pour le provider Chronos-2."
        )
    return saturn_url, saturn_author


def _build_residual_load_bundle_once(
    *,
    project_root: Path,
    delivery_day: str,
    device: str,
    local_files_only: bool,
    dry_run: bool,
) -> Path:
    from chronos2_hourly.chronos_residual_load import (
        build_live_residual_load_bundle,
        planned_live_residual_load_manifest_path,
    )

    cutoff = _residual_load_cutoff(delivery_day)
    output_root = (
        project_root
        / "runs"
        / "experiments"
        / "chronos2_residual_load"
        / "upstream"
    ).resolve()
    if dry_run:
        return planned_live_residual_load_manifest_path(
            delivery_day=delivery_day,
            runtime_cutoff=cutoff,
            output_root=output_root,
        )
    saturn_url, saturn_author = _residual_load_provider_settings(project_root)
    print(
        "\n[UPSTREAM] Forecast Chronos-2 des 5 charges residuelles "
        f"au cutoff {cutoff.isoformat()}...",
        flush=True,
    )
    manifest = build_live_residual_load_bundle(
        delivery_day=delivery_day,
        runtime_cutoff=cutoff,
        output_root=output_root,
        saturn_url=saturn_url,
        saturn_author=saturn_author,
        device=device,
        local_files_only=local_files_only,
        batch_size=8,
    )
    print(f"[UPSTREAM] Bundle audite: {manifest}", flush=True)
    return manifest.resolve()


def normalise_forecast_mode(value: str | None) -> str:
    """Return the canonical execution view requested by the user."""

    mode = str(value or "production").strip().lower()
    if mode not in FORECAST_MODES:
        raise ValueError(
            "Mode doit etre Production, Autonomous, Blend, Both ou All."
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
    if canonical_mode in {"both", "all"}:
        return ("autonomous", "kalman")
    raise AssertionError(f"Mode valide non resolu: {canonical_mode}.")


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


def _report_or_deferred_export(
    archive: str | Path,
    *,
    mode: str,
    residual_load_source: str,
) -> Path | None:
    """Allow derived Saturn exports to repair a degraded archived report.

    The immutable live archive is never rewritten.  Non-production modes build
    their detailed HTML from the sealed CSV/Statistics artifacts in temporary
    space, so a report-only failure must not discard an otherwise valid price
    forecast.  Production mode and Chronos-2 shadow reports remain strict.
    """

    try:
        return find_detailed_html_report(archive)
    except DetailedReportError:
        if mode != "production" and residual_load_source == "saturn":
            return None
        raise


@dataclass(frozen=True)
class _ExportSpec:
    zone: str
    timezone: str
    delivery_day: str
    variant: str
    source_model: str
    baseline_model: str
    archive: Path
    history_archive: Path
    csv_path: Path
    report_path: Path
    history_contract: "_ExportHistoryContract | None"
    batch_mode: str = "unknown"
    residual_load_source: str = "saturn"
    live_config: Path | None = None
    registry_path: Path | None = None
    device: str = "auto"
    threads: int = 4
    local_files_only: bool = True
    kalman_config: Path | None = None
    lora_resolution: ActivationResolution | None = None
    lora_candidate: "_LoRACandidateRuntime | None" = None
    project_root: Path = PROJECT_ROOT


@dataclass(frozen=True)
class _LoRACandidateRuntime:
    """One verified inference shared by autonomous and Kalman exports."""

    resolution: ActivationResolution
    run: ExogenousRunResult
    history: pd.DataFrame
    forecast: pd.DataFrame
    history_start_utc: str
    history_end_utc: str
    issued_days_reused: tuple[str, ...]
    shadow_days_reused: tuple[str, ...] = ()
    shadow_predictions_sha256: str | None = None
    shadow_manifest_sha256: str | None = None
    shadow_issued_history_sha256: str | None = None


@dataclass(frozen=True)
class _ExportHistoryContract:
    sealed_benchmark_run: Path
    live_output_root: Path
    forecast_name: str
    candidate_model: str
    target_series: str
    prediction_mode: str
    storm_pit_path: Path | None
    storm_dashboard_enabled: bool
    saturn_url: str | None
    saturn_author: str | None


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
_REPORT_ATTRIBUTION_FILES = (
    "variable_attribution_hourly.csv.gz",
    "variable_attribution_audit.json",
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
    if variant == "kalman":
        return KALMAN_MODEL, KALMAN_UPSTREAM_MODEL
    if variant == "kalman_weather":
        return KALMAN_WEATHER_MODEL, KALMAN_UPSTREAM_MODEL
    if variant == "kalman_hybrid":
        return KALMAN_HYBRID_MODEL, KALMAN_UPSTREAM_MODEL
    raise ValueError(f"Variante d'export inconnue: {variant!r}.")


def _resolve_config_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _paired_saturn_history_archive(
    *,
    status: ZoneStatus,
    delivery_day: str,
) -> Path:
    """Return the checksum-sealed Saturn control used for rich reporting."""

    if status.live_config is None:
        raise ValueError(
            f"{status.code}: configuration live absente pour retrouver le "
            "controle Saturn."
        )
    config_path = Path(status.live_config).expanduser().resolve()
    config = load_yaml(config_path)
    live = config.get("live")
    if not isinstance(live, Mapping):
        raise TypeError(f"{config_path}: live doit etre un mapping.")
    output_root = _resolve_config_path(live["output_root"], base=config_path.parent)
    archive = (
        output_root
        / f"{status.code.lower()}_day_ahead_{delivery_day}"
    ).resolve()
    from chronos2_hourly.chronos_residual_load import (
        validate_sealed_saturn_control_archive,
    )

    validate_sealed_saturn_control_archive(
        archive,
        expected_delivery_day=delivery_day,
        expected_zone=status.code,
    )
    return archive


def _load_export_history_contract(
    *,
    status: ZoneStatus,
    archive: Path,
) -> _ExportHistoryContract | None:
    """Resolve the immutable history inputs used by a disposable export.

    Minimal/synthetic test archives may not carry a live configuration; in
    that case the exporter keeps its legacy copy-only behavior.  Real zone
    archives always resolve this strict contract.
    """

    if status.live_config is None:
        return None
    config_path = Path(status.live_config).expanduser().resolve()
    if not config_path.is_file():
        return None
    config = load_yaml(config_path)
    live = config.get("live")
    if not isinstance(live, dict):
        raise TypeError(f"{config_path}: live doit etre un mapping.")
    manifest_path = archive / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"{manifest_path}: objet JSON attendu.")
    prediction_mode = str(manifest.get("prediction_mode", "")).strip()
    candidate_model = str(manifest.get("candidate_model", "")).strip()
    target_series = str(manifest.get("target_series", "")).strip()
    if prediction_mode not in {"mkonline_blend", "autonomous_only"}:
        raise ValueError(
            f"{status.code}: prediction_mode historique invalide."
        )
    expected_candidate = (
        "mkonline_blend"
        if prediction_mode == "mkonline_blend"
        else "residual_corrected"
    )
    if candidate_model != expected_candidate or not target_series:
        raise ValueError(f"{status.code}: identite Statistics incomplete.")
    base = config_path.parent
    benchmark = _resolve_config_path(live["sealed_benchmark_run"], base=base)
    output_root = _resolve_config_path(live["output_root"], base=base)
    forecast_name = _forecast_path(archive, status.code).name
    storm_value = live.get("storm_pit_path")
    storm_path = (
        _resolve_config_path(storm_value, base=base)
        if storm_value not in (None, "")
        else None
    )
    dashboard_value = live.get("storm_dashboard_series")
    try:
        expected_dashboard = storm_dashboard_series(status.code)
    except ValueError:
        expected_dashboard = None
    dashboard_enabled = expected_dashboard is not None
    if dashboard_enabled and (
        dashboard_value not in (None, "")
        and str(dashboard_value).strip() != expected_dashboard
    ):
        raise ValueError(
            f"{status.code}: serie Storm dashboard non verifiee."
        )
    # Every report refreshes its observed target from Saturn, including zones
    # without an official Storm contract such as ES.
    base_config_path = _resolve_config_path(live["base_config"], base=base)
    base_config = load_yaml(base_config_path)
    data = base_config.get("data")
    if not isinstance(data, dict):
        raise TypeError(f"{base_config_path}: data doit etre un mapping.")
    saturn_url = str(data.get("saturn_url", "")).strip()
    saturn_author = str(data.get("saturn_author", "")).strip()
    if not saturn_url or not saturn_author:
        raise ValueError(
            f"{status.code}: acces Saturn absent pour actualiser les observations."
        )
    return _ExportHistoryContract(
        sealed_benchmark_run=benchmark,
        live_output_root=output_root,
        forecast_name=forecast_name,
        candidate_model=candidate_model,
        target_series=target_series,
        prediction_mode=prediction_mode,
        storm_pit_path=storm_path,
        storm_dashboard_enabled=dashboard_enabled,
        saturn_url=saturn_url,
        saturn_author=saturn_author,
    )


def _resolve_lora_routes(
    contract: ExogenousActivationContract,
    *,
    zones: Sequence[str],
    mode: str,
) -> dict[tuple[str, str], ActivationResolution]:
    """Resolve only autonomous/Kalman exports; blend is never rewritten."""

    routes: dict[tuple[str, str], ActivationResolution] = {}
    for zone in zones:
        for variant in requested_export_variants(mode, zone):
            if variant in {"autonomous", "kalman"}:
                routes[(zone, variant)] = contract.resolve(
                    zone=zone,
                    mode=variant,
                )
    return routes


def _lora_run_directory(
    resolution: ActivationResolution,
    *,
    delivery_day: str,
) -> Path:
    bundle = resolution.bundle
    if not resolution.lora_enabled or bundle is None:
        raise ValueError("Un run LoRA requiert un bundle promu actif.")
    return (
        resolution.runtime_output_root
        / resolution.zone.lower()
        / delivery_day
        / bundle.bundle_manifest_sha256[:16]
    ).resolve()


def _normalise_lora_predictions(
    frame: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    required = {
        "delivery_start_utc",
        *(f"{LORA_AUTONOMOUS_MODEL}__{quantile}" for quantile in ("q10", "q50", "q90")),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label}: predictions LoRA finales absentes: {missing}.")
    result = frame.copy()
    result.index = pd.DatetimeIndex(
        pd.to_datetime(result.pop("delivery_start_utc"), utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    selected = [
        f"{LORA_AUTONOMOUS_MODEL}__{quantile}"
        for quantile in ("q10", "q50", "q90")
    ]
    for optional in (
        *(f"{LORA_BASE_MODEL}__{quantile}" for quantile in ("q10", "q50", "q90")),
        "forecast_origin_utc",
        "residual_shift_eur_mwh",
    ):
        if optional in result:
            selected.append(optional)
    result = result.loc[:, selected]
    numeric = result.loc[:, [column for column in selected if column != "forecast_origin_utc"]]
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: predictions LoRA non finies.")
    result.loc[:, numeric.columns] = numeric
    if result.index.has_duplicates:
        duplicate = result.loc[result.index.duplicated(keep=False)]
        conflicts = []
        for timestamp, block in duplicate.groupby(level=0, sort=False):
            values = block.loc[:, numeric.columns].to_numpy(dtype=float)
            if not np.allclose(values, values[:1], rtol=0.0, atol=1e-9):
                conflicts.append(str(timestamp))
        if conflicts:
            raise ValueError(
                f"{label}: predictions LoRA divergentes en doublon: {conflicts[:3]}."
            )
        result = result.loc[~result.index.duplicated(keep="last")]
    result = result.sort_index()
    quantiles = result.loc[:, selected[:3]].to_numpy(dtype=float)
    if bool(((quantiles[:, 0] > quantiles[:, 1]) | (quantiles[:, 1] > quantiles[:, 2])).any()):
        raise ValueError(f"{label}: quantiles LoRA croises.")
    return result


def _prepare_one_lora_candidate(
    resolution: ActivationResolution,
    *,
    delivery_day: str,
    project_root: Path,
    registry_path: Path,
    python_executable: Path,
    device: str,
) -> _LoRACandidateRuntime:
    """Build/reuse one promoted final pipeline, then prove FINAL365 coverage."""

    bundle = resolution.bundle
    manifest_path = resolution.live_source_manifest_path
    if bundle is None or manifest_path is None:
        raise ValueError(
            f"{resolution.zone}: activation LoRA incomplete au preflight."
        )
    expected_manifest_sha = resolution.live_source_manifest_sha256
    if expected_manifest_sha is None:
        raise ValueError(
            f"{resolution.zone}: SHA du manifest de sources live LoRA absent."
        )
    actual_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            f"{resolution.zone}: le manifest de sources live LoRA a change "
            "apres le chargement du contrat; preflight refuse."
        )
    output = _lora_run_directory(resolution, delivery_day=delivery_day)
    if output.is_dir():
        run = validate_registered_candidate_run(
            registry_path=registry_path,
            alias=bundle.alias,
            output_directory=output,
            delivery_day=delivery_day,
            expected_zone=resolution.zone,
        )
    else:
        panel_builder = project_root / "run_chronos2_exogenous_panel.py"
        if not panel_builder.is_file():
            raise FileNotFoundError(
                f"Builder du panel LoRA introuvable: {panel_builder}."
            )
        schema = json.loads(bundle.schema_path.read_text(encoding="utf-8"))
        experiment = json.loads(
            bundle.experiment_manifest_path.read_text(encoding="utf-8")
        )
        context_length = int(schema.get("context_length", 0))
        pack = str(experiment.get("panel_pack", "")).strip()
        timezone_name = str(schema.get("timezone", "")).strip()
        if context_length < 1 or not pack or not timezone_name:
            raise ValueError(
                f"{resolution.zone}: schema de panel du bundle promu incomplet."
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"chronos2_lora_panel_{resolution.zone.lower()}_"
        ) as raw_tmp:
            panel_path = Path(raw_tmp) / "live_panel.parquet"
            command = [
                str(python_executable),
                str(panel_builder),
                "--mode",
                "shadow",
                "--zones",
                resolution.zone,
                "--pack",
                pack,
                "--layout",
                "per_zone",
                "--end-day",
                delivery_day,
                "--context-length",
                str(context_length),
                "--timezone",
                timezone_name,
                "--project-root",
                str(project_root),
                "--output",
                str(panel_path),
                "--live-source-manifest",
                str(manifest_path),
                "--require-complete-context",
            ]
            completed = subprocess.run(
                command,
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(
                    f"{resolution.zone}: construction du panel live LoRA en echec"
                    + (f": {details[-3000:]}" if details else ".")
                )
            audit_path = panel_path.with_suffix(panel_path.suffix + ".audit.json")
            try:
                run_registered_candidate(
                    registry_path=registry_path,
                    alias=bundle.alias,
                    live_panel_path=panel_path,
                    live_panel_audit_path=audit_path,
                    delivery_day=delivery_day,
                    output_directory=output,
                    device_map=device,
                )
            except ExogenousProductionError:
                # A concurrent invocation may have won the immutable publish
                # race.  Reuse it only through the complete sealed validator.
                if not output.is_dir():
                    raise
        run = validate_registered_candidate_run(
            registry_path=registry_path,
            alias=bundle.alias,
            output_directory=output,
            delivery_day=delivery_day,
            expected_zone=resolution.zone,
        )
    return _assemble_lora_runtime(
        resolution,
        run=run,
        delivery_day=delivery_day,
        registry_path=registry_path,
    )


def _assemble_lora_runtime(
    resolution: ActivationResolution,
    *,
    run: ExogenousRunResult,
    delivery_day: str,
    registry_path: Path,
) -> _LoRACandidateRuntime:
    """Join rolling, final shadow, then later immutable issued forecasts."""

    bundle = resolution.bundle
    if bundle is None:
        raise ValueError(f"{resolution.zone}: bundle LoRA absent.")
    schema = json.loads(bundle.schema_path.read_text(encoding="utf-8"))
    timezone_name = str(schema.get("timezone", "")).strip()
    requested = date.fromisoformat(delivery_day)
    expected_history = pd.date_range(
        pd.Timestamp(requested - timedelta(days=365), tz=timezone_name),
        pd.Timestamp(requested, tz=timezone_name),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    candidate_columns = [
        f"{LORA_AUTONOMOUS_MODEL}__{quantile}"
        for quantile in ("q10", "q50", "q90")
    ]
    base = _normalise_lora_predictions(
        pd.read_csv(run.backtest_path),
        label=f"{resolution.zone}/rolling promu",
    ).loc[:, candidate_columns]
    pieces = [base]
    available = base.index.intersection(expected_history)
    shadow_days_reused: list[str] = []
    shadow_predictions_sha256: str | None = None
    shadow_manifest_sha256: str | None = None
    shadow_issued_history_sha256: str | None = None
    if len(expected_history.difference(available)):
        shadow_artifacts = load_promoted_shadow_history(bundle)
        shadow = _normalise_lora_predictions(
            shadow_artifacts.predictions,
            label=f"{resolution.zone}/shadow promu",
        ).loc[:, candidate_columns]
        overlapping = (
            shadow.index.intersection(base.index).intersection(expected_history)
        )
        if len(overlapping) and not np.allclose(
            base.reindex(overlapping).to_numpy(dtype=float),
            shadow.reindex(overlapping).to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{resolution.zone}: predictions rolling/shadow divergentes."
            )
        shadow_fill = shadow.index.intersection(
            expected_history.difference(available)
        )
        if len(shadow_fill):
            pieces.append(shadow.reindex(shadow_fill))
            shadow_days_reused = sorted(
                {
                    day.isoformat()
                    for day in shadow_fill.tz_convert(timezone_name).date
                }
            )
            available = available.union(shadow_fill)
        shadow_predictions_sha256 = shadow_artifacts.predictions_sha256
        shadow_manifest_sha256 = shadow_artifacts.manifest_sha256
        shadow_issued_history_sha256 = shadow_artifacts.issued_history_sha256
    missing_days = sorted(
        set(expected_history.difference(available).tz_convert(timezone_name).date)
    )
    reused_days: list[str] = []
    absent_days: list[str] = []
    for missing_day in missing_days:
        day_text = missing_day.isoformat()
        prior_output = _lora_run_directory(resolution, delivery_day=day_text)
        if not prior_output.is_dir():
            absent_days.append(day_text)
            continue
        prior = validate_registered_candidate_run(
            registry_path=registry_path,
            alias=bundle.alias,
            output_directory=prior_output,
            delivery_day=day_text,
            expected_zone=resolution.zone,
        )
        issued = _normalise_lora_predictions(
            pd.read_csv(prior.forecast_path),
            label=f"{resolution.zone}/forecast emis {day_text}",
        ).loc[:, candidate_columns]
        pieces.append(issued)
        reused_days.append(day_text)
    if absent_days:
        preview = ", ".join(absent_days[:8])
        suffix = "..." if len(absent_days) > 8 else ""
        raise ValueError(
            f"{resolution.zone}: historique {LORA_AUTONOMOUS_MODEL} FINAL365 "
            f"incomplet; runs emis absents pour {preview}{suffix}."
        )
    combined = pd.concat(pieces, axis=0).sort_index()
    if combined.index.has_duplicates:
        duplicates = combined.loc[combined.index.duplicated(keep=False)]
        conflicts = []
        for timestamp, block in duplicates.groupby(level=0, sort=False):
            values = block.to_numpy(dtype=float)
            if not np.allclose(values, values[:1], rtol=0.0, atol=1e-9):
                conflicts.append(str(timestamp))
        if conflicts:
            raise ValueError(
                f"{resolution.zone}: historique LoRA divergent: {conflicts[:3]}."
            )
        combined = combined.loc[~combined.index.duplicated(keep="last")]
    history = combined.reindex(expected_history)
    if not np.isfinite(history.to_numpy(dtype=float)).all():
        raise ValueError(
            f"{resolution.zone}: historique {LORA_AUTONOMOUS_MODEL} FINAL365 troue."
        )
    future = _normalise_lora_predictions(
        pd.read_csv(run.forecast_path),
        label=f"{resolution.zone}/forecast LoRA {delivery_day}",
    )
    expected_future = pd.date_range(
        pd.Timestamp(requested, tz=timezone_name),
        pd.Timestamp(requested + timedelta(days=1), tz=timezone_name),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    if not future.index.equals(expected_future):
        raise ValueError(
            f"{resolution.zone}: forecast LoRA hors de la journee {delivery_day}."
        )
    return _LoRACandidateRuntime(
        resolution=resolution,
        run=run,
        history=history,
        forecast=future,
        history_start_utc=expected_history[0].isoformat(),
        history_end_utc=expected_history[-1].isoformat(),
        issued_days_reused=tuple(reused_days),
        shadow_days_reused=tuple(shadow_days_reused),
        shadow_predictions_sha256=shadow_predictions_sha256,
        shadow_manifest_sha256=shadow_manifest_sha256,
        shadow_issued_history_sha256=shadow_issued_history_sha256,
    )


def _prepare_lora_candidates(
    routes: Mapping[tuple[str, str], ActivationResolution],
    *,
    contract: ExogenousActivationContract,
    delivery_day: str,
    project_root: Path,
    python_executable: Path,
    device: str,
) -> dict[str, _LoRACandidateRuntime]:
    active: dict[str, ActivationResolution] = {}
    for (zone, _variant), resolution in routes.items():
        if not resolution.lora_enabled:
            continue
        previous = active.get(zone)
        if previous is not None and previous.bundle != resolution.bundle:
            raise ValueError(f"{zone}: routes LoRA autonomous/Kalman divergentes.")
        active[zone] = resolution
    prepared: dict[str, _LoRACandidateRuntime] = {}
    for zone, resolution in active.items():
        print(
            f"[{zone}] preflight LoRA: pipeline final promu "
            f"{LORA_AUTONOMOUS_MODEL}...",
            flush=True,
        )
        prepared[zone] = _prepare_one_lora_candidate(
            resolution,
            delivery_day=delivery_day,
            project_root=project_root,
            registry_path=contract.registry_path,
            python_executable=python_executable,
            device=device,
        )
    return prepared


def _build_export_specs(
    *,
    zone_results: Sequence[BatchZoneResult],
    statuses: dict[str, ZoneStatus],
    mode: str,
    delivery_day: str,
    export_root: Path,
    residual_load_source: str = "saturn",
    registry_path: str | Path = DEFAULT_REGISTRY,
    device: str = "auto",
    threads: int = 4,
    local_files_only: bool = True,
    kalman_config: str | Path = DEFAULT_KALMAN_CONFIG,
    kalman_weather_configs: Mapping[str, str | Path] | None = None,
    kalman_hybrid_configs: Mapping[str, str | Path] | None = None,
    lora_routes: Mapping[tuple[str, str], ActivationResolution] | None = None,
    lora_candidates: Mapping[str, _LoRACandidateRuntime] | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[_ExportSpec, ...]:
    specs: list[_ExportSpec] = []
    for result in zone_results:
        if result.archive_path is None:
            raise ValueError(f"{result.zone}: archive source absente.")
        archive = result.archive_path.resolve()
        status = statuses[result.zone]
        history_archive = (
            _paired_saturn_history_archive(
                status=status,
                delivery_day=delivery_day,
            )
            if residual_load_source == "chronos2"
            else archive
        )
        history_contract = _load_export_history_contract(
            status=status,
            archive=history_archive,
        )
        for variant in requested_export_variants(mode, result.zone):
            source_model, baseline_model = _model_for_variant(variant)
            lora_candidate: _LoRACandidateRuntime | None = None
            lora_resolution = (lora_routes or {}).get((result.zone, variant))
            if lora_resolution is not None and lora_resolution.lora_enabled:
                lora_candidate = (lora_candidates or {}).get(result.zone)
                if lora_candidate is None:
                    raise ValueError(
                        f"{result.zone}/{variant}: run LoRA preflight absent."
                    )
                if variant == "autonomous":
                    source_model = lora_resolution.selected_model
                    baseline_model = lora_resolution.fallback_model
                elif variant == "kalman":
                    source_model = lora_resolution.selected_model
                    if lora_resolution.kalman_upstream_model is None:
                        raise ValueError(
                            f"{result.zone}/kalman: upstream LoRA non resolu."
                        )
                    baseline_model = lora_resolution.kalman_upstream_model
                else:
                    raise ValueError(
                        f"{result.zone}/{variant}: activation LoRA interdite."
                    )
            selected_kalman_config: Path | None = None
            if variant == "kalman":
                selected_kalman_config = Path(kalman_config).expanduser().resolve()
            elif variant == "kalman_weather":
                configured = (kalman_weather_configs or {}).get(result.zone)
                if configured is None:
                    raise ValueError(
                        f"{result.zone}/kalman_weather: sidecar runtime absent."
                    )
                selected_kalman_config = Path(configured).expanduser().resolve()
            elif variant == "kalman_hybrid":
                configured = (kalman_hybrid_configs or {}).get(result.zone)
                if configured is None:
                    raise ValueError(
                        f"{result.zone}/kalman_hybrid: sidecar runtime absent."
                    )
                selected_kalman_config = Path(configured).expanduser().resolve()
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
                    history_archive=history_archive,
                    csv_path=(directory / f"{stem}.csv").resolve(),
                    report_path=(directory / f"{stem}.html").resolve(),
                    history_contract=history_contract,
                    batch_mode=mode,
                    residual_load_source=residual_load_source,
                    live_config=(
                        Path(status.live_config).expanduser().resolve()
                        if status.live_config is not None
                        else None
                    ),
                    registry_path=Path(registry_path).expanduser().resolve(),
                    device=device,
                    threads=int(threads),
                    local_files_only=bool(local_files_only),
                    kalman_config=selected_kalman_config,
                    lora_resolution=(
                        lora_resolution
                        if lora_resolution is not None
                        and lora_resolution.lora_enabled
                        else None
                    ),
                    lora_candidate=lora_candidate,
                    project_root=Path(project_root).expanduser().resolve(),
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

    # residual_kalman is derived later, inside the disposable reporting view.
    # Its immutable source only needs to expose the audited autonomous
    # upstream; requiring Kalman columns here would make every valid live
    # archive fail before the causal overlay can be built.
    if spec.lora_candidate is not None:
        resolution = spec.lora_resolution
        if resolution is None:
            raise ValueError(
                f"{spec.zone}/{spec.variant}: resolution LoRA absente."
            )
        incumbent_upstream = (
            resolution.fallback_upstream_model
            if spec.variant == "kalman"
            else resolution.fallback_model
        )
        if incumbent_upstream is None:
            raise ValueError(
                f"{spec.zone}/{spec.variant}: fallback LoRA non resolu."
            )
        source_models = (incumbent_upstream,)
    else:
        source_models = (
            (spec.baseline_model,)
            if spec.variant in KALMAN_VARIANTS
            else (spec.source_model, spec.baseline_model)
        )
    forecast_path = _forecast_path(spec.archive, spec.zone)
    forecast_columns = _read_csv_columns(forecast_path)
    required_forecast = {
        "delivery_start_utc",
        *(
            column
            for model in source_models
            for column in _required_model_columns(model)
        ),
    }
    missing_forecast = sorted(required_forecast - forecast_columns)
    if missing_forecast:
        raise ValueError(
            f"{spec.zone}/{spec.variant}: colonnes auditees absentes ("
            + "forecast="
            + ", ".join(missing_forecast)
            + "). Aucun export n'a ete cree."
        )

    backtest_path = spec.history_archive / "backtest_hourly_oof.csv.gz"
    backtest_columns = _read_csv_columns(backtest_path)
    required_backtest = {
        "delivery_start_utc",
        "actual",
        *(
            column
            for model in source_models
            for column in _required_model_columns(model)
        ),
    }
    missing_backtest = sorted(required_backtest - backtest_columns)
    if missing_backtest:
        raise ValueError(
            f"{spec.zone}/{spec.variant}: colonnes auditees absentes ("
            + "backtest="
            + ", ".join(missing_backtest)
            + "). Aucun export n'a ete cree."
        )
    statistics_path = spec.history_archive / "statistics_history_hourly.csv.gz"
    if statistics_path.is_file():
        statistics_columns = _read_csv_columns(statistics_path)
        required_statistics = {
            "delivery_start_utc",
            "actual",
            *(
                column
                for model in source_models
                for column in _required_model_columns(model)
            ),
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
    if spec.residual_load_source == "chronos2":
        for filename in (
            "statistics_history_hourly.csv.gz",
            "statistics_history_audit.json",
        ):
            path = spec.history_archive / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"{spec.zone}/{spec.variant}: historique Saturn scelle "
                    f"absent pour le rapport canonique: {path}"
                )


def _canonical_target_from_archive(spec: _ExportSpec) -> pd.Series:
    path = spec.archive / "inputs" / "aligned_inputs.csv.gz"
    raw = pd.read_csv(path, usecols=["timestamp", "target"])
    index = pd.DatetimeIndex(
        pd.to_datetime(raw["timestamp"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    values = pd.to_numeric(raw["target"], errors="coerce")
    target = pd.Series(values.to_numpy(dtype=float), index=index, name="actual")
    target = target.loc[target.notna()]
    if target.empty or target.index.has_duplicates or not target.index.is_monotonic_increasing:
        raise ValueError(f"{spec.zone}: cible canonique d'export invalide.")
    return target


def _fetch_latest_observed_snapshot(
    client: object,
    *,
    spec: _ExportSpec,
    expected_index: pd.DatetimeIndex,
    extracted_at_utc: pd.Timestamp | None = None,
    allow_pending_current_day: bool = False,
) -> tuple[pd.Series, dict[str, object]]:
    """Read latest actuals; optional D-only rejection never admits unverified prices."""

    if not isinstance(allow_pending_current_day, bool):
        raise TypeError("allow_pending_current_day must be an explicit boolean.")
    expected = pd.DatetimeIndex(expected_index)
    if expected.tz is None:
        raise ValueError("La timeline d'observation doit etre timezone-aware.")
    expected = expected.tz_convert("UTC").sort_values()
    if expected.empty or expected.has_duplicates:
        raise ValueError("La timeline d'observation est vide ou dupliquee.")
    extracted = (
        pd.Timestamp.now(tz="UTC")
        if extracted_at_utc is None
        else pd.Timestamp(extracted_at_utc)
    )
    if extracted.tzinfo is None:
        raise ValueError("extracted_at_utc doit etre timezone-aware.")
    extracted = extracted.tz_convert("UTC")
    requested_start = expected[0] - pd.Timedelta(hours=2)
    requested_end = expected[-1] + pd.Timedelta(hours=2)
    values = fetch_saturn_series_from_client(
        client,
        spec.history_contract.target_series,
        requested_start,
        requested_end,
        spec.timezone,
        naive_timezone="UTC",
        nocache=True,
        live=True,
    )
    index = pd.DatetimeIndex(values.index)
    if index.tz is None:
        raise ValueError(f"{spec.zone}: observations Saturn sans timezone.")
    normalized = pd.Series(
        pd.to_numeric(values, errors="coerce").to_numpy(dtype=float),
        index=index.tz_convert("UTC"),
        name="actual",
    ).sort_index()
    if normalized.index.has_duplicates:
        raise ValueError(f"{spec.zone}: observations Saturn dupliquees.")
    available = normalized.loc[np.isfinite(normalized.to_numpy(dtype=float))]
    if available.empty:
        raise RuntimeError(f"{spec.zone}: aucune observation Saturn recente.")
    source: dict[str, object] = {
        "kind": "saturn_target_latest_extraction",
        "series": spec.history_contract.target_series,
        "zone": spec.zone,
        "extracted_at_utc": str(extracted),
        "requested_from_utc": str(requested_start),
        "requested_to_utc": str(requested_end),
        "first_available_observation_utc": str(available.index[0]),
        "last_available_observation_utc": str(available.index[-1]),
        "selection": "latest values returned by Saturn at extraction time",
        "nocache": True,
        "live_recomputation": True,
        "used_for_prediction": False,
    }

    current_day = date.fromisoformat(spec.delivery_day)
    current_selector = pd.Index(expected.tz_convert(spec.timezone).date) == current_day
    current_delivery = expected[current_selector]
    expected_values = pd.to_numeric(
        normalized.reindex(expected), errors="coerce"
    ).to_numpy(dtype=float)
    canonical_missing = ~np.isfinite(expected_values)
    missing_delivery = expected[canonical_missing]
    fallback_series = POST_AUCTION_OBSERVED_SERIES_BY_ZONE.get(spec.zone)
    if len(missing_delivery) and fallback_series is not None:
        first_missing = pd.Timestamp(missing_delivery[0])
        missing_suffix = pd.date_range(
            first_missing,
            expected[-1],
            freq="h",
            tz="UTC",
        )
        suffix_only = missing_delivery.equals(missing_suffix)
        validation_start = first_missing - pd.Timedelta(
            days=POST_AUCTION_VALIDATION_DAYS
        )
        validation_index = pd.date_range(
            validation_start,
            first_missing - pd.Timedelta(hours=1),
            freq="h",
            tz="UTC",
        )
        if not suffix_only:
            # A short requested history must not weaken the overlap guard.
            # Retrieve only an absent prefix, with the same latest/report-only
            # options; these observations never enter a prediction state.
            prefix_end = normalized.index.min() - pd.Timedelta(hours=1)
            if validation_start <= prefix_end:
                prefix_raw = fetch_saturn_series_from_client(
                    client, spec.history_contract.target_series,
                    validation_start, prefix_end, spec.timezone,
                    naive_timezone="UTC", nocache=True, live=True,
                )
                prefix_index = pd.DatetimeIndex(prefix_raw.index)
                if prefix_index.tz is None or prefix_index.has_duplicates:
                    raise ValueError(f"{spec.zone}: recouvrement canonique invalide.")
                prefix = pd.Series(
                    pd.to_numeric(prefix_raw, errors="coerce").to_numpy(dtype=float),
                    index=prefix_index.tz_convert("UTC"), name="actual",
                )
                prefix = prefix.loc[(prefix.index >= validation_start) & (prefix.index <= prefix_end)]
                normalized = pd.concat([prefix, normalized]).sort_index()
                source["canonical_validation_prefix"] = {
                    "requested_from_utc": str(validation_start),
                    "requested_to_utc": str(prefix_end),
                    "nocache": True, "live_recomputation": True,
                    "used_for_prediction": False,
                }
            known_after_gap = normalized.loc[
                (normalized.index >= first_missing)
                & (normalized.index <= expected[-1])
                & np.isfinite(normalized.to_numpy(dtype=float))
            ].index
            validation_index = validation_index.union(known_after_gap)
        fallback_raw = fetch_saturn_series_from_client(
            client,
            fallback_series,
            validation_start,
            expected[-1] + pd.Timedelta(hours=2),
            spec.timezone,
            naive_timezone="UTC",
            nocache=True,
            live=True,
        )
        fallback_index = pd.DatetimeIndex(fallback_raw.index)
        if fallback_index.tz is None:
            raise ValueError(
                f"{spec.zone}: observations post-enchere Saturn sans timezone."
            )
        fallback = pd.Series(
            pd.to_numeric(fallback_raw, errors="coerce").to_numpy(dtype=float),
            index=fallback_index.tz_convert("UTC"),
            name="post_auction_actual",
        ).sort_index()
        if fallback.index.has_duplicates:
            raise ValueError(
                f"{spec.zone}: observations post-enchere Saturn dupliquees."
            )

        fallback_missing = pd.to_numeric(
            fallback.reindex(missing_delivery), errors="coerce"
        ).to_numpy(dtype=float)
        fillable = np.isfinite(fallback_missing)
        unapplied_fallback = {
            "status": "not_applied_no_available_observations",
            "series": fallback_series,
            "role": (
                "reporting_only_missing_canonical_suffix" if suffix_only
                else "reporting_only_missing_canonical_hours"
            ),
            "used_for_prediction": False,
            "requested_from_utc": str(validation_start),
            "requested_to_utc": str(expected[-1] + pd.Timedelta(hours=2)),
            "validation_status": "not_required_no_values_applied",
            "validation_days": POST_AUCTION_VALIDATION_DAYS,
            "validation_minimum_paired_hours": POST_AUCTION_MINIMUM_PAIRED_HOURS,
            "validation_paired_hours": 0,
            "validation_maximum_absolute_difference_eur_mwh": None,
            "validation_tolerance_eur_mwh": POST_AUCTION_EQUIVALENCE_ATOL,
            "missing_canonical_suffix_hours": int((missing_delivery > available.index[-1]).sum()),
            "missing_canonical_internal_hours": int((missing_delivery <= available.index[-1]).sum()),
            "missing_canonical_hours": int(len(missing_delivery)),
            "first_missing_canonical_utc": str(missing_delivery[0]),
            "last_missing_canonical_utc": str(missing_delivery[-1]),
            "missing_canonical_current_hours": int(current_selector[canonical_missing].sum()),
            "applied_suffix_hours": 0,
            "applied_internal_hours": 0,
            "applied_current_hours": 0,
            "applied_hours": 0,
            "applied_value_times_utc": [],
        }
        if not fillable.any():
            # Before the auction, both sources can legitimately lack all of D.
            # An unused fallback does not need to prove price equivalence. Keep
            # every canonical value (and gap); callers still enforce historical
            # coverage and mask incomplete delivery days. No substitute is used.
            source["post_auction_fallback"] = unapplied_fallback
            return normalized, source

        paired = pd.concat(
            [
                normalized.reindex(validation_index).rename("canonical"),
                fallback.reindex(validation_index).rename("post_auction"),
            ],
            axis=1,
        )
        paired = paired.loc[np.isfinite(paired.to_numpy(dtype=float)).all(axis=1)]
        if len(paired) != len(validation_index):
            raise ValueError(
                f"{spec.zone}: recouvrement insuffisant entre la cible "
                "canonique et la source post-enchere "
                f"({len(paired)} h != {len(validation_index)} h requises)."
            )
        absolute_difference = (
            paired["canonical"] - paired["post_auction"]
        ).abs()
        maximum_difference = float(absolute_difference.max())
        if maximum_difference > POST_AUCTION_EQUIVALENCE_ATOL:
            worst_hour = absolute_difference.idxmax()
            error = PostAuctionObservationDivergenceError(
                zone=spec.zone, timestamp=worst_hour,
                canonical=float(paired.loc[worst_hour, "canonical"]),
                post_auction=float(paired.loc[worst_hour, "post_auction"]),
                maximum=maximum_difference,
                divergent_hours=int((absolute_difference > POST_AUCTION_EQUIVALENCE_ATOL).sum()),
                compared_hours=len(paired), canonical_series=spec.history_contract.target_series,
                post_auction_series=fallback_series, extracted_at_utc=extracted,
            )
            if allow_pending_current_day and missing_delivery.difference(current_delivery).empty:
                # NYX may publish a forecast while D's observations are pending.
                # Reject the optional complement, not the complete historical
                # canonical series. No discrepant price enters labels or scores.
                # Any missing hour outside D retains the strict failure below.
                source["post_auction_fallback"] = {
                    **unapplied_fallback,
                    "status": "rejected_divergent_optional_current_day",
                    "validation_status": "rejected_divergence",
                    "validation_paired_hours": int(len(paired)),
                    "validation_maximum_absolute_difference_eur_mwh": maximum_difference,
                    "candidate_available_hours": int(fillable.sum()),
                    "rejection_diagnostic": error.diagnostic,
                }
                source["current_delivery_actual_reason"] = "post_auction_source_rejected"
                return normalized, source
            raise error

        if fillable.any():
            supplement = pd.Series(
                fallback_missing[fillable],
                index=missing_delivery[fillable],
                name="actual",
            )
            # Saturn may omit the not-yet-propagated hours entirely instead of
            # returning explicit NaNs. ``combine_first`` safely handles both
            # representations without mutating any canonical finite value.
            normalized = normalized.where(np.isfinite(normalized.to_numpy(dtype=float)))
            normalized = normalized.combine_first(supplement).sort_index()
        source["post_auction_fallback"] = {
            "status": (
                ("complete_missing_suffix" if suffix_only else "complete_missing_hours")
                if np.isfinite(
                    pd.to_numeric(
                        normalized.reindex(expected), errors="coerce"
                    ).to_numpy(dtype=float)
                ).all()
                else ("incomplete_missing_suffix" if suffix_only else "incomplete_missing_hours")
            ),
            "series": fallback_series,
            "role": (
                "reporting_only_missing_canonical_suffix" if suffix_only
                else "reporting_only_missing_canonical_hours"
            ),
            "used_for_prediction": False,
            "requested_from_utc": str(validation_start),
            "requested_to_utc": str(expected[-1] + pd.Timedelta(hours=2)),
            "validation_days": POST_AUCTION_VALIDATION_DAYS,
            "validation_minimum_paired_hours": (
                POST_AUCTION_MINIMUM_PAIRED_HOURS
            ),
            "validation_paired_hours": int(len(paired)),
            "validation_maximum_absolute_difference_eur_mwh": maximum_difference,
            "validation_tolerance_eur_mwh": POST_AUCTION_EQUIVALENCE_ATOL,
            "missing_canonical_suffix_hours": int((missing_delivery > available.index[-1]).sum()),
            "missing_canonical_internal_hours": int((missing_delivery <= available.index[-1]).sum()),
            "missing_canonical_hours": int(len(missing_delivery)),
            "first_missing_canonical_utc": str(missing_delivery[0]),
            "last_missing_canonical_utc": str(missing_delivery[-1]),
            "applied_suffix_hours": int((missing_delivery[fillable] > available.index[-1]).sum()),
            "applied_internal_hours": int((missing_delivery[fillable] <= available.index[-1]).sum()),
            "applied_hours": int(fillable.sum()),
            "applied_value_times_utc": [value.isoformat() for value in missing_delivery[fillable]],
            "missing_canonical_current_hours": int(
                current_selector[canonical_missing].sum()
            ),
            "applied_current_hours": int(
                (
                    pd.Index(
                        missing_delivery[fillable].tz_convert(spec.timezone).date
                    )
                    == current_day
                ).sum()
            ),
        }
        available = normalized.loc[
            np.isfinite(normalized.to_numpy(dtype=float))
        ]
        source["last_available_observation_utc"] = str(available.index[-1])
        source["selection"] = (
            "latest canonical values returned by Saturn; missing observation "
            "hours may be completed from a recent-equivalent post-auction "
            "EPEX observation series"
        )
    return normalized, source


def _latest_statistics_day(
    spec: _ExportSpec,
    *,
    observed: pd.Series,
    forecast_delivery: pd.DatetimeIndex,
) -> date:
    """Include the current forecast day only once every actual is available."""

    current_day = date.fromisoformat(spec.delivery_day)
    current_values = pd.to_numeric(
        observed.reindex(forecast_delivery), errors="coerce"
    ).to_numpy(dtype=float)
    if len(current_values) == len(forecast_delivery) and np.isfinite(
        current_values
    ).all():
        return current_day
    return current_day - timedelta(days=1)


def _remove_unsupported_storm_from_statistics(directory: Path) -> None:
    """Remove Storm only for a zone without a verified native contract."""

    history_path = directory / "statistics_history_hourly.csv.gz"
    audit_path = directory / "statistics_history_audit.json"
    history = pd.read_csv(history_path)
    storm_columns = [
        column for column in history.columns if "storm" in column.casefold()
    ]
    if storm_columns:
        history = history.drop(columns=storm_columns)
        history.to_csv(history_path, index=False, compression="gzip")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise TypeError(f"{audit_path}: objet JSON attendu.")
    audit.pop("storm_primary_report_benchmark", None)
    audit.pop("storm_dashboard", None)
    audit["storm_refresh_status"] = (
        "unsupported_for_zone"
    )
    import hashlib

    digest = hashlib.sha256(history_path.read_bytes()).hexdigest()
    audit["statistics_history_sha256"] = digest
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _refresh_statistics_view(
    spec: _ExportSpec,
    destination: Path,
    *,
    actual_snapshot_cache: dict[
        tuple[str, str, str, str], tuple[pd.Series, Mapping[str, object]]
    ] | None = None,
    storm_snapshot_cache: dict[
        tuple[str, str, str], tuple[pd.Series, Mapping[str, object]]
    ] | None = None,
) -> None:
    contract = spec.history_contract
    if contract is None:
        for filename in (
            "statistics_history_hourly.csv.gz",
            "statistics_history_audit.json",
        ):
            source = spec.history_archive / filename
            if source.is_file():
                shutil.copy2(source, destination / filename)
        return
    unavailable_storm = destination / "inputs" / "storm_refresh_unavailable.parquet"
    storm_path = (
        contract.storm_pit_path
        if contract.storm_pit_path is not None
        and contract.storm_pit_path.is_file()
        else unavailable_storm
    )
    update_kwargs = {
        "staging_run_dir": destination,
        "sealed_benchmark_run": contract.sealed_benchmark_run,
        "live_output_root": contract.live_output_root,
        "replay_output_root": contract.live_output_root / "_replays",
        "current_delivery_day": date.fromisoformat(spec.delivery_day),
        "canonical_target": _canonical_target_from_archive(spec),
        "storm_pit_path": storm_path,
        "storm_dashboard_native": None,
        "zone": spec.zone,
        "timezone": spec.timezone,
        "forecast_name": contract.forecast_name,
        "candidate_model": contract.candidate_model,
        "target_series": contract.target_series,
        "prediction_mode": contract.prediction_mode,
    }
    audit = update_live_statistics_history(**update_kwargs)
    if contract.saturn_url is None or contract.saturn_author is None:
        raise RuntimeError(f"{spec.zone}: contrat Saturn observations incomplet.")
    delivery, _actual = statistics_contract(destination)
    forecast_delivery = _forecast_delivery_index(spec)
    extraction_delivery = delivery.union(forecast_delivery).sort_values()
    client = create_saturn_client(
        contract.saturn_url,
        os.getenv("SATURN_AUTHOR") or contract.saturn_author,
    )
    actual_key = (
        spec.zone,
        contract.target_series,
        str(extraction_delivery[0]),
        str(extraction_delivery[-1]),
    )
    actual_cache = (
        actual_snapshot_cache if actual_snapshot_cache is not None else {}
    )
    actual_snapshot = actual_cache.get(actual_key)
    if actual_snapshot is None:
        actual_values, actual_source = _fetch_latest_observed_snapshot(
            client,
            spec=spec,
            expected_index=extraction_delivery,
        )
        actual_snapshot = (actual_values, actual_source)
        actual_cache[actual_key] = actual_snapshot
    latest_complete_actual_day = _latest_statistics_day(
        spec,
        observed=actual_snapshot[0],
        forecast_delivery=forecast_delivery,
    )
    current_day = date.fromisoformat(spec.delivery_day)
    current_actual_complete = latest_complete_actual_day == current_day
    canonical_for_statistics = actual_snapshot[0].copy()
    current_raw = pd.to_numeric(
        canonical_for_statistics.reindex(forecast_delivery), errors="coerce"
    ).to_numpy(dtype=float)
    current_available_hours = int(np.isfinite(current_raw).sum())
    if not current_actual_complete:
        # A partially propagated auction curve must never create a misleading
        # partial daily average. Keep the full delivery-day support but mask
        # every observed value until all 23/24/25 physical hours are present.
        canonical_for_statistics = canonical_for_statistics.reindex(
            canonical_for_statistics.index.union(forecast_delivery)
        )
        canonical_for_statistics.loc[forecast_delivery] = np.nan
    actual_source = {
        **dict(actual_snapshot[1]),
        "current_delivery_day_local": spec.delivery_day,
        "current_delivery_expected_hours": int(len(forecast_delivery)),
        "current_delivery_raw_available_hours": current_available_hours,
        "current_delivery_actual_status": (
            "complete" if current_actual_complete else "pending_placeholder"
        ),
        "partial_daily_average_forbidden": True,
        "used_for_prediction": False,
    }
    # The forecast day is always present in reporting Statistics. Actuals are
    # all finite when published, otherwise all NaN by the rule above.
    statistics_through_day = current_day
    refreshed_kwargs = {
        **update_kwargs,
        "canonical_target": canonical_for_statistics,
        "canonical_target_source": actual_source,
        "statistics_through_day": statistics_through_day,
        "allow_current_day_placeholder": True,
    }
    if contract.storm_dashboard_enabled:
        cache_key = (
            spec.zone,
            str(extraction_delivery[0]),
            str(extraction_delivery[-1]),
        )
        cache = storm_snapshot_cache if storm_snapshot_cache is not None else {}
        snapshot = cache.get(cache_key)
        if snapshot is None:
            native, source = fetch_native_dashboard_snapshot(
                client,
                zone=spec.zone,
                expected_index=extraction_delivery,
            )
            snapshot = (native, dict(source))
            cache[cache_key] = snapshot
        audit = update_live_statistics_history(
            **{
                **refreshed_kwargs,
                "storm_dashboard_native": snapshot[0],
                "storm_dashboard_source": snapshot[1],
            }
        )
        _write_storm_live_forecast(
            destination,
            spec=spec,
            raw_snapshot=snapshot[0],
            source=snapshot[1],
            delivery=forecast_delivery,
        )
    else:
        audit = update_live_statistics_history(**refreshed_kwargs)
        _remove_unsupported_storm_from_statistics(destination)
        audit = json.loads(
            (destination / "statistics_history_audit.json").read_text(
                encoding="utf-8"
            )
        )
    if (
        audit.get("status") != "complete"
        or audit.get("statistics_complete") is not True
        or audit.get("statistics_prefix_end_local")
        != statistics_through_day.isoformat()
        or (audit.get("canonical_actuals") or {}).get("source", {}).get(
            "extracted_at_utc"
        )
        is None
    ):
        raise RuntimeError(
            f"{spec.zone}: Statistics rafraichies incompletes: {audit}."
        )


def _mark_chronos2_reference_statistics(spec: _ExportSpec, directory: Path) -> None:
    """Make the hybrid report's historical scope impossible to misread."""

    if spec.residual_load_source != "chronos2":
        return
    audit_path = directory / "statistics_history_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"{spec.zone}: audit Statistics Saturn absent: {audit_path}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise TypeError(f"{audit_path}: objet JSON attendu.")
    challenger_note = (
        "La prevision J+1 affichee utilise residual_load Chronos-2. "
        "Les panneaux historiques et Statistics proviennent du controle "
        "Saturn scelle du meme pays et du meme jour; ils decrivent le modele "
        "de prix aval gele et ne constituent pas une performance historique "
        "du challenger. Sa performance est mesuree uniquement par la "
        "comparaison prospective appariee apres observation des prix reels."
    )
    prior_note = str(audit.get("report_scope_note") or "").strip()
    audit["report_scope_note"] = (
        f"{prior_note} {challenger_note}".strip()
        if challenger_note not in prior_note
        else prior_note
    )
    audit["residual_load_challenger_reporting_context"] = {
        "forecast_source": "chronos2",
        "historical_metrics_source": "paired_sealed_saturn_control",
        "saturn_control_archive": str(spec.history_archive),
        "challenger_archive": str(spec.archive),
        "historical_metrics_are_challenger_performance": False,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _forecast_delivery_index(spec: _ExportSpec) -> pd.DatetimeIndex:
    forecast = pd.read_csv(
        _forecast_path(spec.archive, spec.zone),
        usecols=["delivery_start_utc"],
    )
    delivery = pd.DatetimeIndex(
        pd.to_datetime(
            forecast["delivery_start_utc"], utc=True, errors="raise"
        ),
        name="delivery_start_utc",
    )
    if delivery.empty or delivery.has_duplicates or not delivery.is_monotonic_increasing:
        raise ValueError(f"{spec.zone}: timeline forecast invalide pour Storm live.")
    local_days = pd.Index(delivery.tz_convert(spec.timezone).date).unique()
    if list(map(str, local_days)) != [spec.delivery_day]:
        raise ValueError(
            f"{spec.zone}: le forecast Storm live ne correspond pas à "
            f"la livraison {spec.delivery_day}."
        )
    return delivery


def _write_storm_live_forecast(
    destination: Path,
    *,
    spec: _ExportSpec,
    raw_snapshot: pd.Series,
    source: Mapping[str, object],
    delivery: pd.DatetimeIndex,
) -> None:
    actual_placeholder = pd.Series(
        0.0,
        index=delivery,
        name="report_only_placeholder",
    )
    comparator = normalize_native_dashboard_series(
        raw_snapshot,
        zone=spec.zone,
        expected_index=delivery,
        actual=actual_placeholder,
        source=source,
    )
    values = comparator.values.reindex(delivery)
    available = values.notna()
    if not bool(available.any()):
        raise RuntimeError(
            f"{spec.zone}: Storm officiel indisponible pour le jour de livraison."
        )
    artifact = destination / STORM_DASHBOARD_LIVE_FORECAST_ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            STORM_DASHBOARD_COLUMN: values.to_numpy(dtype=float),
        }
    ).to_parquet(artifact, index=False)
    audit = {
        "status": "complete",
        "role": "report_only_live_comparator",
        "zone": spec.zone,
        "delivery_day": spec.delivery_day,
        "expected_hours": int(len(delivery)),
        "available_hours": int(available.sum()),
        "missing_hours": int((~available).sum()),
        "used_for_prediction": False,
        "source": dict(source),
        "normalization": {
            "naive_timezone": spec.timezone,
            "missing_utc": [str(value) for value in delivery[~available]],
        },
    }
    audit_path = destination / STORM_DASHBOARD_LIVE_FORECAST_AUDIT
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _archived_zone_data(
    archive: Path,
    *,
    zone: str,
    timezone: str,
    frequency: str,
) -> ZoneData:
    """Rebuild exactly the immutable ZoneData frames saved by a live run."""

    inputs = archive / "inputs"
    aligned = pd.read_csv(inputs / "aligned_inputs.csv.gz")
    model = pd.read_csv(inputs / "model_covariates_with_future.csv.gz")
    for name, frame in (("aligned", aligned), ("model_covariates", model)):
        if "timestamp" not in frame:
            raise ValueError(f"{zone}: timestamp absent de {name}.")
        index = pd.DatetimeIndex(
            pd.to_datetime(frame.pop("timestamp"), utc=True, errors="raise")
        ).tz_convert(timezone)
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError(f"{zone}: timeline archive invalide dans {name}.")
        frame.index = index
        for column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "target" not in aligned:
        raise ValueError(f"{zone}: cible absente des inputs archives.")
    target = aligned.pop("target").dropna().astype(float).rename("target")
    if target.empty or not target.index.is_monotonic_increasing:
        raise ValueError(f"{zone}: cible archive invalide.")
    manifest = json.loads(
        (archive / "run_manifest.json").read_text(encoding="utf-8")
    )
    diagnostics = manifest.get("input_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    coverage_path = inputs / "input_coverage.csv"
    input_manifest_path = inputs / "input_manifest.csv"
    return ZoneData(
        zone=zone,
        timezone=timezone,
        frequency=frequency,
        target=target,
        covariates=aligned,
        model_context_covariates=model,
        known_future_columns=[
            str(column) for column in model if str(column).startswith("known_")
        ],
        coverage=(
            pd.read_csv(coverage_path)
            if coverage_path.is_file()
            else pd.DataFrame()
        ),
        input_manifest=(
            pd.read_csv(input_manifest_path)
            if input_manifest_path.is_file()
            else pd.DataFrame()
        ),
        diagnostics=diagnostics,
    )


def _archived_chronos_prediction(archive: Path) -> pd.DataFrame:
    path = archive / "chronos_live_hourly.csv"
    frame = pd.read_csv(path)
    required = {"delivery_start_utc", "q10", "q50", "q90"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Chronos archive incomplet: {missing}.")
    index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("delivery_start_utc"), utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Timeline Chronos archive invalide.")
    frame.index = index
    return frame


def _official_archive_quantiles(
    forecast: pd.DataFrame,
    *,
    prefix: str,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    columns = {
        quantile: f"{prefix}__{quantile}"
        for quantile in ("q10", "q50", "q90")
    }
    missing = sorted(set(columns.values()).difference(forecast.columns))
    if missing:
        raise ValueError(f"Quantiles archives absents pour {prefix}: {missing}.")
    return pd.DataFrame(
        {
            quantile: pd.to_numeric(forecast[column], errors="raise").to_numpy(
                dtype=float
            )
            for quantile, column in columns.items()
        },
        index=index,
    )


def _materialize_archived_variable_attribution(
    spec: _ExportSpec,
    destination: Path,
    *,
    registry_path: Path,
    device: str,
    threads: int,
    local_files_only: bool,
) -> Path:
    """Explain an older sealed forecast inside disposable export space."""

    if spec.live_config is None or not spec.live_config.is_file():
        raise FileNotFoundError(
            f"{spec.zone}: configuration live absente pour reconstruire l'attribution."
        )
    live_config = load_yaml(spec.live_config)
    live = live_config.get("live")
    if not isinstance(live, Mapping):
        raise TypeError(f"{spec.live_config}: live doit etre un mapping.")
    base_config_path = _resolve_config_path(
        live["base_config"], base=spec.live_config.parent
    )
    config = load_yaml(base_config_path)
    manifest = json.loads(
        (spec.archive / "run_manifest.json").read_text(encoding="utf-8")
    )
    cutoff = str(manifest.get("forecast_cutoff_local") or "").strip()
    data_config = config.setdefault("data", {})
    if not isinstance(data_config, dict):
        raise TypeError(f"{base_config_path}: data doit etre un mapping.")
    if cutoff:
        data_config["runtime_as_of"] = cutoff
    set_reproducibility(int(deep_get(config, "model.seed", 42)))
    data = _archived_zone_data(
        spec.archive,
        zone=spec.zone,
        timezone=spec.timezone,
        frequency=str(deep_get(config, "data.frequency", "h")),
    )
    from run_chronos2_hourly import _feature_inputs

    _target, _history, future_covariates, all_features = _feature_inputs(
        data, config
    )
    fresh_future = all_features.loc[future_covariates.index].copy()
    fresh_future.index = fresh_future.index.tz_convert("UTC")
    fresh_future.index.name = "delivery_start_utc"
    chronos = _archived_chronos_prediction(spec.archive)
    if not fresh_future.index.equals(chronos.index):
        raise ValueError(
            f"{spec.zone}: les features archivees divergent de l'horizon Chronos."
        )
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from chronos2_modular.forecasting import load_model

    from chronos2_hourly.report_attribution_cache import local_attribution_model_config
    runtime = load_model(
        local_attribution_model_config(config) if local_files_only else config,
        device, local_files_only,
    )
    context_length = int(deep_get(config, "model.context_length", 2048))
    model_batch_size = int(deep_get(config, "model.model_batch_size", 128))
    if spec.zone == "FR":
        from run_mkonline_live_hourly import _train_and_predict_extended

        frozen = _resolve_config_path(
            live["frozen_autonomous_run"], base=spec.live_config.parent
        )
        _prediction, _fit_audit, corrector = _train_and_predict_extended(
            frozen_source=frozen,
            fresh_future=fresh_future,
            chronos_live=chronos,
            threads=int(threads),
        )
        required_covariates = tuple(map(str, data.covariates.columns))
    else:
        from chronos2_hourly.multizone_contract import load_zone_model_contract
        from chronos2_hourly.multizone_live import _train_and_predict_autonomous

        contract = load_zone_model_contract(spec.live_config, registry_path)
        _prediction, _fit_audit, corrector = _train_and_predict_autonomous(
            contract=contract,
            fresh_future=fresh_future,
            chronos_live=chronos,
            threads=int(threads),
        )
        required_covariates = contract.required_covariates
    forecast_path = _forecast_path(spec.archive, spec.zone)
    forecast = pd.read_csv(forecast_path)
    forecast_index = pd.DatetimeIndex(
        pd.to_datetime(forecast["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    official_autonomous = _official_archive_quantiles(
        forecast,
        prefix="residual_corrected",
        index=forecast_index,
    )
    prediction_mode = str(manifest.get("prediction_mode", "")).strip()
    official_blend: pd.Series | None = None
    primary: pd.Series | None = None
    autonomous_weight = float(manifest.get("autonomous_weight", 1.0))
    mkonline_weight = float(manifest.get("mkonline_weight", 0.0))
    if prediction_mode == "mkonline_blend":
        required = {"mkonline_blend__q50", "mkonline_primary__q50"}
        missing = sorted(required.difference(forecast.columns))
        if missing:
            raise ValueError(
                f"{spec.zone}: blend archive incomplet pour attribution: {missing}."
            )
        official_blend = pd.Series(
            pd.to_numeric(
                forecast["mkonline_blend__q50"], errors="raise"
            ).to_numpy(dtype=float),
            index=forecast_index,
        )
        primary = pd.Series(
            pd.to_numeric(
                forecast["mkonline_primary__q50"], errors="raise"
            ).to_numpy(dtype=float),
            index=forecast_index,
        )
    from chronos2_hourly.variable_attribution import write_variable_attribution

    destination.mkdir(parents=True, exist_ok=True)
    write_variable_attribution(
        output_dir=destination,
        forecast_path=forecast_path,
        data=data,
        runtime=runtime,
        fresh_future=fresh_future,
        corrector=corrector,
        official_autonomous=official_autonomous,
        required_covariates=required_covariates,
        context_length=context_length,
        model_batch_size=model_batch_size,
        zone=spec.zone,
        timezone=spec.timezone,
        delivery_day=spec.delivery_day,
        official_blend=official_blend,
        primary=primary,
        autonomous_weight=autonomous_weight,
        mkonline_weight=mkonline_weight,
    )
    return destination


def _copy_reporting_view(
    spec: _ExportSpec,
    destination: Path,
    *,
    attribution_source_dir: Path | None = None,
    actual_snapshot_cache: dict[
        tuple[str, str, str, str], tuple[pd.Series, Mapping[str, object]]
    ] | None = None,
    storm_snapshot_cache: dict[
        tuple[str, str, str], tuple[pd.Series, Mapping[str, object]]
    ] | None = None,
) -> None:
    """Create a disposable report view, never a child of the live archive."""

    destination.mkdir(parents=True, exist_ok=False)
    inputs = destination / "inputs"
    inputs.mkdir()
    shutil.copy2(
        spec.history_archive / "backtest_hourly_oof.csv.gz",
        destination / "backtest_hourly_oof.csv.gz",
    )
    shutil.copy2(
        _forecast_path(spec.archive, spec.zone),
        destination / f"forecast_hourly_{spec.zone.lower()}.csv",
    )
    for filename in _REPORT_INPUT_FILES:
        shutil.copy2(spec.archive / "inputs" / filename, inputs / filename)
    for filename in _REPORT_OPTIONAL_FILES:
        if filename in {
            "statistics_history_hourly.csv.gz",
            "statistics_history_audit.json",
        }:
            continue
        source_root = (
            spec.archive
            if filename == "run_manifest.json"
            else spec.history_archive
        )
        source = source_root / filename
        if source.is_file():
            shutil.copy2(source, destination / filename)
    if spec.lora_candidate is None and (
        spec.variant not in KALMAN_VARIANTS or (spec.archive / "artifact_checksums.json").is_file()
    ):
        attribution_root = attribution_source_dir or spec.archive
        attribution_sources = [
            attribution_root / filename for filename in _REPORT_ATTRIBUTION_FILES
        ]
        attribution_present = [source.is_file() for source in attribution_sources]
        if any(attribution_present) and not all(attribution_present):
            raise FileNotFoundError(
                f"{spec.zone}: paire d'attribution incomplete dans {attribution_root}."
            )
        for source in attribution_sources:
            if source.is_file():
                # Attribution describes the forecast being exported.  In
                # challenger mode ``history_archive`` is a different Saturn
                # control, so these two files must always come from ``archive``.
                shutil.copy2(source, destination / source.name)
        if spec.variant in KALMAN_VARIANTS and all(attribution_present):
            # The attribution is upstream of Kalman. The reader verifies this
            # original forecast seal and the preserved residual_corrected P50.
            shutil.copy2(spec.archive / "artifact_checksums.json", destination / "artifact_checksums.json")
    _refresh_statistics_view(
        spec,
        destination,
        actual_snapshot_cache=actual_snapshot_cache,
        storm_snapshot_cache=storm_snapshot_cache,
    )
    _mark_chronos2_reference_statistics(spec, destination)


def _materialize_lora_reporting_view(
    spec: _ExportSpec,
    destination: Path,
) -> None:
    """Overlay one promoted final LoRA pipeline in disposable report space."""

    runtime = spec.lora_candidate
    if runtime is None:
        return
    if spec.variant not in {"autonomous", "kalman"}:
        raise ValueError(
            f"{spec.zone}/{spec.variant}: overlay LoRA interdit sur cette variante."
        )
    resolution = spec.lora_resolution
    if resolution is None:
        raise ValueError(f"{spec.zone}/{spec.variant}: resolution LoRA absente.")
    bundle = resolution.bundle
    if bundle is None or runtime.run.zone != spec.zone:
        raise ValueError(f"{spec.zone}: run LoRA promu incoherent.")
    incumbent = (
        resolution.fallback_upstream_model
        if spec.variant == "kalman"
        else resolution.fallback_model
    )
    if incumbent is None:
        raise ValueError(f"{spec.zone}: fallback LoRA absent.")
    candidate_columns = [
        f"{LORA_AUTONOMOUS_MODEL}__{quantile}"
        for quantile in ("q10", "q50", "q90")
    ]
    incumbent_columns = [
        f"{incumbent}__{quantile}" for quantile in ("q10", "q50", "q90")
    ]
    current_predictions = runtime.forecast.loc[:, candidate_columns]
    full_predictions = pd.concat(
        [runtime.history.loc[:, candidate_columns], current_predictions],
        axis=0,
    )
    if full_predictions.index.has_duplicates:
        raise ValueError(f"{spec.zone}: timeline LoRA history/future chevauchante.")

    fallback_rows_by_file: dict[str, int] = {}
    for filename in (
        "backtest_hourly_oof.csv.gz",
        "statistics_history_hourly.csv.gz",
    ):
        path = destination / filename
        if not path.is_file():
            if filename.startswith("statistics"):
                continue
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        index = pd.DatetimeIndex(
            pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="raise")
        )
        if index.has_duplicates:
            raise ValueError(f"{spec.zone}/{filename}: timeline dupliquee.")
        missing_incumbent = sorted(set(incumbent_columns).difference(frame.columns))
        if missing_incumbent:
            raise ValueError(
                f"{spec.zone}/{filename}: fallback incumbent absent: {missing_incumbent}."
            )
        aligned = full_predictions.reindex(index)
        fallback_mask = aligned[candidate_columns[0]].isna().to_numpy()
        fallback_rows_by_file[filename] = int(fallback_mask.sum())
        for candidate_column, incumbent_column in zip(
            candidate_columns, incumbent_columns, strict=True
        ):
            candidate = pd.to_numeric(
                aligned[candidate_column], errors="coerce"
            ).to_numpy(dtype=float)
            incumbent_values = pd.to_numeric(
                frame[incumbent_column], errors="coerce"
            ).to_numpy(dtype=float)
            frame[candidate_column] = np.where(
                np.isfinite(candidate), candidate, incumbent_values
            )
        # This report-local signal is the incremental final-pipeline delta
        # versus the incumbent autonomous model.  It is the only coherent
        # residual signal available for the downstream Kalman warm-up.
        frame["residual_correction"] = (
            pd.to_numeric(frame[f"{LORA_AUTONOMOUS_MODEL}__q50"], errors="raise")
            - pd.to_numeric(frame[f"{incumbent}__q50"], errors="raise")
        )
        frame.to_csv(
            path,
            index=False,
            compression="gzip" if path.name.endswith(".gz") else None,
        )

    forecast_path = _forecast_path(destination, spec.zone)
    forecast = pd.read_csv(forecast_path)
    forecast_index = pd.DatetimeIndex(
        pd.to_datetime(forecast["delivery_start_utc"], utc=True, errors="raise")
    )
    aligned_future = runtime.forecast.reindex(forecast_index)
    if not aligned_future.index.equals(forecast_index):
        raise ValueError(f"{spec.zone}: horizon LoRA non aligne au forecast live.")
    for column in (
        *candidate_columns,
        *(f"{LORA_BASE_MODEL}__{quantile}" for quantile in ("q10", "q50", "q90")),
    ):
        if column not in aligned_future:
            raise ValueError(f"{spec.zone}: colonne LoRA live absente: {column}.")
        values = pd.to_numeric(aligned_future[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{spec.zone}: colonne LoRA live incomplete: {column}.")
        forecast[column] = values
    forecast["residual_correction"] = (
        forecast[f"{LORA_AUTONOMOUS_MODEL}__q50"]
        - pd.to_numeric(forecast[f"{incumbent}__q50"], errors="raise")
    )
    if "forecast_origin_utc" in aligned_future:
        forecast[f"{LORA_AUTONOMOUS_MODEL}_forecast_origin_utc"] = (
            aligned_future["forecast_origin_utc"].astype(str).to_numpy()
        )
    forecast.to_csv(forecast_path, index=False)

    audit = {
        "status": "complete",
        "overlay_kind": "promoted_exogenous_residual_corrected_final_pipeline",
        "raw_lora_checkpoint_used_as_report_model": False,
        "final_pipeline_model": LORA_AUTONOMOUS_MODEL,
        "raw_adapter_model": LORA_BASE_MODEL,
        "zone": spec.zone,
        "delivery_day": spec.delivery_day,
        "variant": spec.variant,
        "source_archive": str(spec.archive),
        "runs_live_modified": False,
        "promotion_alias": bundle.alias,
        "promotion_candidate_id": bundle.candidate_id,
        "promotion_bundle_manifest_sha256": bundle.bundle_manifest_sha256,
        "promotion_bundle_checksums_sha256": bundle.artifact_checksums_sha256,
        "candidate_run_directory": str(runtime.run.output_directory),
        "candidate_run_checksums_sha256": hashlib.sha256(
            (runtime.run.output_directory / "artifact_checksums.json").read_bytes()
        ).hexdigest(),
        "rolling365_start_utc": runtime.history_start_utc,
        "rolling365_end_utc": runtime.history_end_utc,
        "rolling365_hours": int(len(runtime.history)),
        "issued_days_reused": list(runtime.issued_days_reused),
        "promoted_shadow_days_reused": list(runtime.shadow_days_reused),
        "promoted_shadow_predictions_sha256": (
            runtime.shadow_predictions_sha256
        ),
        "promoted_shadow_manifest_sha256": runtime.shadow_manifest_sha256,
        "promoted_shadow_issued_history_sha256": (
            runtime.shadow_issued_history_sha256
        ),
        "warmup_fallback_model": incumbent,
        "warmup_fallback_is_outside_scored_final365": True,
        "fallback_rows_by_file": fallback_rows_by_file,
        "residual_signal": (
            f"{LORA_AUTONOMOUS_MODEL}__q50 - {incumbent}__q50"
        ),
        "storm_used_as_input": False,
        "mkonline_used_as_input": False,
    }
    lora_audit_path = destination / "exogenous_lora_overlay_audit.json"
    _write_export_json(lora_audit_path, audit)
    statistics_audit_path = destination / "statistics_history_audit.json"
    if statistics_audit_path.is_file():
        statistics_audit = json.loads(
            statistics_audit_path.read_text(encoding="utf-8")
        )
        statistics_audit["lora_overlay"] = dict(audit)
        statistics_path = destination / "statistics_history_hourly.csv.gz"
        statistics_audit["statistics_history_sha256"] = hashlib.sha256(
            statistics_path.read_bytes()
        ).hexdigest()
        _write_export_json(statistics_audit_path, statistics_audit)
    manifest_path = destination / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "derived_autonomous_model": LORA_AUTONOMOUS_MODEL,
                "lora_overlay": dict(audit),
                "production_changed": False,
            }
        )
        _write_export_json(manifest_path, manifest)


def _write_export_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _kalman_csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["object"]):
        result[column] = result[column].map(
            lambda value: (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            )
        )
    return result


def _rebase_storm_dashboard_audit_for_kalman(
    history_audit: dict[str, object],
    statistics: pd.DataFrame,
    *,
    timezone: str,
) -> None:
    """Align the Storm/DST audit with the causal Statistics prefix at D-1."""

    dashboard_column = history_audit.get("storm_primary_report_benchmark")
    if not isinstance(dashboard_column, str) or not dashboard_column:
        return
    if dashboard_column not in statistics:
        raise ValueError(
            "Le benchmark Storm declare est absent de la vue Kalman."
        )
    existing = history_audit.get("storm_dashboard")
    if not isinstance(existing, Mapping):
        raise ValueError("L'audit Storm dashboard est absent de la vue Kalman.")
    if "actual" not in statistics:
        raise ValueError("La vue Kalman ne contient pas les observations.")

    delivery = pd.DatetimeIndex(
        pd.to_datetime(
            statistics["delivery_start_utc"], utc=True, errors="raise"
        )
    )
    if delivery.has_duplicates or not delivery.is_monotonic_increasing:
        raise ValueError("Timeline Statistics invalide pour rebaser Storm.")
    dashboard = pd.Series(
        pd.to_numeric(statistics[dashboard_column], errors="coerce").to_numpy(
            dtype=float
        ),
        index=delivery,
        name=dashboard_column,
    )
    actual = pd.Series(
        pd.to_numeric(statistics["actual"], errors="coerce").to_numpy(
            dtype=float
        ),
        index=delivery,
        name="actual",
    )
    dashboard_missing = delivery[
        ~np.isfinite(dashboard.to_numpy(dtype=float))
    ]
    actual_missing = delivery[
        ~np.isfinite(actual.to_numpy(dtype=float))
    ]
    allowed_missing_actual: pd.DatetimeIndex | None = None
    if len(actual_missing):
        canonical_actuals = history_audit.get("canonical_actuals")
        placeholder_enabled = (
            isinstance(canonical_actuals, Mapping)
            and canonical_actuals.get("current_delivery_placeholder") is True
        )
        raw_placeholder_day = (
            canonical_actuals.get("current_delivery_day_local")
            if isinstance(canonical_actuals, Mapping)
            else None
        )
        if not placeholder_enabled or raw_placeholder_day in (None, ""):
            raise ValueError(
                "La vue Kalman contient des observations manquantes hors "
                "placeholder courant audite."
            )
        placeholder_day = date.fromisoformat(str(raw_placeholder_day))
        local_days = pd.Index(delivery.tz_convert(timezone).date)
        expected_placeholder = delivery[local_days == placeholder_day]
        if not actual_missing.equals(expected_placeholder):
            raise ValueError(
                "Le placeholder courant de la vue Kalman doit etre une "
                "journee physique entierement vide."
            )
        allowed_missing_actual = actual_missing
    source = existing.get("source")
    rebased = build_dashboard_comparator(
        dashboard,
        expected_index=delivery,
        actual=actual,
        source=(dict(source) if isinstance(source, Mapping) else {}),
        timezone=timezone,
        minimum_coverage=0.0,
        maximum_missing_hours=len(dashboard_missing),
        allowed_missing_actual_index=allowed_missing_actual,
    ).audit

    existing_dst = existing.get("dst")
    raw_allowed = (
        existing_dst.get("native_allowed_missing_utc", [])
        if isinstance(existing_dst, Mapping)
        else []
    )
    if not isinstance(raw_allowed, list):
        raise ValueError("Audit Storm DST: native_allowed_missing_utc invalide.")
    declared_allowed = pd.DatetimeIndex(
        pd.to_datetime(raw_allowed, utc=True, errors="raise")
    )
    allowed_missing = delivery[delivery.isin(declared_allowed)]
    rebased_dst = dict(rebased["dst"])
    rebased_dst.update(
        {
            "native_allowed_missing_hours": int(len(allowed_missing)),
            "native_allowed_missing_utc": [
                str(value) for value in allowed_missing
            ],
            "native_actual_missing_matches_allowed": dashboard_missing.equals(
                allowed_missing
            ),
        }
    )
    merged = dict(existing)
    merged.update(rebased)
    merged["dst"] = rebased_dst
    history_audit["storm_dashboard"] = merged


def _validate_materialized_kalman_view(
    spec: _ExportSpec,
    view: object,
) -> None:
    """Fail closed before a Kalman-derived file can leave temp space."""

    replay = getattr(view, "replay")
    audit = dict(replay.audit)
    required_audit = {
        "status": "complete",
        "model_key": spec.source_model,
        "upstream_model": spec.baseline_model,
        "filter_only": True,
        "smoother_used": False,
        "em_used": False,
        "storm_used_as_input": False,
    }
    for key, expected in required_audit.items():
        if audit.get(key) != expected:
            raise ValueError(
                f"{spec.zone}/{spec.variant}: audit {key}={audit.get(key)!r}, "
                f"attendu={expected!r}."
            )
    if int(audit.get("causality_violations", -1)) != 0:
        raise ValueError(
            f"{spec.zone}/{spec.variant}: violation de causalite detectee."
        )
    if int(audit.get("quantile_crossings", -1)) != 0:
        raise ValueError(
            f"{spec.zone}/{spec.variant}: croisement de quantiles detecte."
        )
    if spec.variant in KALMAN_EXACT_ROLLING_VARIANTS and (
        audit.get("training_policy") != "fixed_length_rolling_local_days"
        or int(audit.get("training_lookback_days", -1)) != 365
        or int(audit.get("warmup_days", -1)) != 365
    ):
        raise ValueError(
            f"{spec.zone}/{spec.variant}: le refit D-365 exact n'est pas audite."
        )

    requested_day = date.fromisoformat(spec.delivery_day)
    evaluation_start = getattr(view, "evaluation_start_day")
    evaluation_end = getattr(view, "evaluation_end_day")
    expected_evaluation = pd.date_range(
        pd.Timestamp(evaluation_start, tz=spec.timezone),
        pd.Timestamp(requested_day, tz=spec.timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    if (
        evaluation_end != requested_day - timedelta(days=1)
        or (evaluation_end - evaluation_start).days + 1 != 365
        or not getattr(view, "evaluation_index").equals(expected_evaluation)
        or int(audit.get("evaluation_days", -1)) != 365
        or int(audit.get("evaluation_hours", -1)) != len(expected_evaluation)
        or int(audit.get("warmup_days", -1)) < 14
        or int(audit.get("future_observations_assimilated", -1)) != 0
    ):
        raise ValueError(
            f"{spec.zone}/{spec.variant}: la fenetre d'evaluation n'est pas FINAL365."
        )

    expected_future = pd.date_range(
        pd.Timestamp(requested_day, tz=spec.timezone),
        pd.Timestamp(requested_day + timedelta(days=1), tz=spec.timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    if not getattr(view, "future_index").equals(expected_future):
        raise ValueError(
            f"{spec.zone}/{spec.variant}: horizon futur local invalide."
        )

    for name in ("statistics", "backtest", "forecast"):
        frame = getattr(view, name)
        columns = [
            f"{spec.source_model}__{quantile}"
            for quantile in ("q10", "q50", "q90")
        ]
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(
                f"{spec.zone}/{spec.variant}: colonnes absentes de {name}: {missing}."
            )
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        matrix = values.to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"{spec.zone}/{spec.variant}: valeurs non finies dans {name}."
            )
        crossed = (matrix[:, 0] > matrix[:, 1]) | (
            matrix[:, 1] > matrix[:, 2]
        )
        if bool(crossed.any()):
            raise ValueError(
                f"{spec.zone}/{spec.variant}: quantiles croises dans {name}."
            )
    if len(getattr(view, "backtest")) != len(getattr(view, "evaluation_index")):
        raise ValueError(
            f"{spec.zone}/{spec.variant}: backtest FINAL365 incomplet."
        )


def _preflight_kalman_reporting_view(
    spec: _ExportSpec,
    destination: Path,
) -> None:
    """Check every physical training/evaluation hour before any Kalman fit."""

    if spec.variant not in KALMAN_VARIANTS:
        return
    from chronos2_hourly.kalman_configuration import (
        attach_kalman_upstream_history,
        load_kalman_operational_configuration,
    )
    from chronos2_hourly.kalman_residual import (
        validate_operational_kalman_history,
    )

    if spec.kalman_config is None:
        raise ValueError(f"{spec.zone}/{spec.variant}: configuration sidecar absente.")
    try:
        configuration = load_kalman_operational_configuration(
            spec.kalman_config,
            project_root=spec.project_root,
            zone=spec.zone,
            upstream_model=spec.baseline_model,
        )
        if configuration.training_lookback_days != 365:
            raise ValueError("Le refit operationnel exige training_lookback_days: 365.")
        statistics, _ = attach_kalman_upstream_history(
            pd.read_csv(destination / "statistics_history_hourly.csv.gz"),
            configuration,
            timezone=spec.timezone,
        )
        validate_operational_kalman_history(
            statistics=statistics,
            timezone=spec.timezone,
            delivery_day=spec.delivery_day,
            upstream_model=spec.baseline_model,
            evaluation_days=365,
            training_lookback_days=365,
        )
    except ValueError as exc:
        raise ValueError(
            f"{spec.zone}/{spec.variant}: preflight historique Kalman refuse: {exc}"
        ) from exc
    print(
        f"[KALMAN-PREFLIGHT] {spec.zone}/{spec.variant}: "
        "365 jours de calibration + 365 jours evalues verifies.",
        flush=True,
    )


def _materialize_kalman_reporting_view(
    spec: _ExportSpec,
    destination: Path,
) -> None:
    """Derive Kalman from refreshed Statistics in disposable export space."""

    if spec.variant not in KALMAN_VARIANTS:
        return
    # Modes without a Kalman export remain independent from pykalman.
    from chronos2_hourly.kalman_residual import (
        build_operational_kalman_view,
    )
    from chronos2_hourly.kalman_configuration import (
        attach_additional_kalman_sources,
        attach_kalman_upstream_history,
        load_kalman_operational_configuration,
    )

    if spec.kalman_config is None:
        raise ValueError(
            f"{spec.zone}/{spec.variant}: configuration sidecar absente."
        )
    operational_config = load_kalman_operational_configuration(
        spec.kalman_config,
        project_root=spec.project_root,
        zone=spec.zone,
        upstream_model=spec.baseline_model,
    )

    statistics_path = destination / "statistics_history_hourly.csv.gz"
    forecast_path = _forecast_path(destination, spec.zone)
    covariates_path = (
        destination / "inputs" / "model_covariates_with_future.csv.gz"
    )
    requested_day = date.fromisoformat(spec.delivery_day)
    required_future_index = pd.date_range(
        pd.Timestamp(requested_day, tz=spec.timezone),
        pd.Timestamp(requested_day + timedelta(days=1), tz=spec.timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    covariates, sidecar_input_audit = attach_additional_kalman_sources(
        pd.read_csv(covariates_path),
        operational_config,
        required_future_index=required_future_index,
        timezone=spec.timezone,
    )
    statistics, upstream_input_audit = attach_kalman_upstream_history(
        pd.read_csv(statistics_path),
        operational_config,
        timezone=spec.timezone,
    )
    view = build_operational_kalman_view(
        statistics=statistics,
        source_forecast=pd.read_csv(forecast_path),
        covariates=covariates,
        timezone=spec.timezone,
        delivery_day=spec.delivery_day,
        config=operational_config.filter_config,
        covariate_config=operational_config.covariate_config,
        upstream_model=spec.baseline_model,
        output_model=spec.source_model,
        training_lookback_days=operational_config.training_lookback_days,
        rolling_refit_workers=operational_config.rolling_refit_workers,
        rolling_refit_cache_dir=(
            spec.project_root
            / "runs"
            / "cache"
            / "kalman_rolling"
            / spec.zone.lower()
            / spec.variant
        ),
    )
    cache_audit = view.replay.audit.get("rolling_refit_cache", {})
    if isinstance(cache_audit, Mapping) and cache_audit.get("enabled"):
        print(
            f"[KALMAN-CACHE] {spec.zone}/{spec.variant}: "
            f"historique hits={cache_audit.get('history_hits', 0)}, "
            f"refits={cache_audit.get('history_fitted_days', 0)}; "
            f"futur hits={cache_audit.get('future_hits', 0)}, "
            f"refits={cache_audit.get('future_fitted_days', 0)}",
            flush=True,
        )
    _validate_materialized_kalman_view(spec, view)

    view.statistics.to_csv(statistics_path, index=False, compression="gzip")
    view.backtest.to_csv(
        destination / "backtest_hourly_oof.csv.gz",
        index=False,
        compression="gzip",
    )
    view.forecast.to_csv(forecast_path, index=False)
    _kalman_csv_safe(view.replay.daily_audit).to_csv(
        destination / "kalman_daily_audit.csv",
        index=False,
    )
    _kalman_csv_safe(view.replay.state_audit).to_csv(
        destination / "kalman_state_audit.csv.gz",
        index=False,
        compression="gzip",
    )

    kalman_audit = {
        **dict(view.replay.audit),
        "publication_status": "derived_live_export",
        "production_changed": False,
        "source_archive": str(spec.archive),
        "future_delivery_day": spec.delivery_day,
        "future_hours": int(len(view.forecast)),
        "future_audit": dict(view.replay.future_audit),
        "rolling365_hours": int(len(view.backtest)),
        "storm_used_as_input": False,
        "mkonline_used_as_input": False,
        "operational_configuration": operational_config.audit_dict(),
        "upstream_history": dict(upstream_input_audit),
    }
    filter_audit_path = destination / "kalman_filter_audit.json"
    _write_export_json(
        filter_audit_path,
        kalman_audit,
    )
    _write_export_json(
        destination / "kalman_operational_sidecar_audit.json",
        {
            **dict(sidecar_input_audit),
            "status": "complete",
            "zone": spec.zone,
            "delivery_day": spec.delivery_day,
            "source_archive": str(spec.archive),
            "production_changed": False,
            "sealed_live_contract_modified": False,
            "filter_audit_path": "kalman_filter_audit.json",
            "filter_audit_sha256": hashlib.sha256(
                filter_audit_path.read_bytes()
            ).hexdigest(),
            "replay_audit": dict(view.replay.audit),
            "future_audit": dict(view.replay.future_audit),
            "upstream_history": dict(upstream_input_audit),
        },
    )

    evaluation_end = view.evaluation_end_day.isoformat()
    statistics_end_day = max(
        pd.DatetimeIndex(
            pd.to_datetime(
                view.statistics["delivery_start_utc"], utc=True, errors="raise"
            )
        ).tz_convert(spec.timezone).date
    )
    statistics_end = statistics_end_day.isoformat()
    history_audit_path = destination / "statistics_history_audit.json"
    history_audit = json.loads(history_audit_path.read_text(encoding="utf-8"))
    if not isinstance(history_audit, dict):
        raise TypeError(f"{history_audit_path}: objet JSON attendu.")
    _rebase_storm_dashboard_audit_for_kalman(
        history_audit,
        view.statistics,
        timezone=spec.timezone,
    )
    for key in (
        "evaluated_realized_days",
        "missing_realized_days",
        "excluded_after_first_gap_days",
    ):
        raw_days = history_audit.get(key)
        if isinstance(raw_days, list):
            history_audit[key] = [
                str(value)
                for value in raw_days
                if date.fromisoformat(str(value)) <= statistics_end_day
            ]
    statistics_delivery = pd.DatetimeIndex(
        pd.to_datetime(
            view.statistics["delivery_start_utc"], utc=True, errors="raise"
        )
    )
    evaluated_days = set(history_audit.get("evaluated_realized_days") or [])
    statistics_local_days = pd.Index(
        statistics_delivery.tz_convert(spec.timezone).date
    )
    history_audit.update(
        {
            "status": "complete",
            "statistics_history_path": "statistics_history_hourly.csv.gz",
            "statistics_history_sha256": hashlib.sha256(
                statistics_path.read_bytes()
            ).hexdigest(),
            "statistics_through_day_local": statistics_end,
            "statistics_prefix_end_local": statistics_end,
            "last_required_realized_day": statistics_end,
            "statistics_complete": True,
            "n_evaluated_realized_days": len(evaluated_days),
            "n_evaluated_realized_hours": int(
                pd.Index(statistics_local_days.astype(str)).isin(evaluated_days).sum()
            ),
            "n_total_statistics_hours": int(len(view.statistics)),
            "kalman_overlay": {
                "status": "complete",
                "model": spec.source_model,
                "source_candidate": spec.baseline_model,
                "scope": "last_365_complete_local_delivery_days",
                "candidate_frozen_before_storm_attachment": True,
                "storm_used_as_input": False,
                "mkonline_used_as_input": False,
                "audit_path": "kalman_filter_audit.json",
            },
        }
    )
    _write_export_json(history_audit_path, history_audit)

    metrics_path = destination / "metrics_hourly.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise TypeError(f"{metrics_path}: objet JSON attendu.")
    training = metrics.get("training_diagnostics")
    if not isinstance(training, dict):
        training = {}
    training.update(
        {
            "evaluation_start_local_date": view.evaluation_start_day.isoformat(),
            "evaluation_end_local_date": evaluation_end,
            "evaluation_days": 365,
            "evaluation_hours": int(len(view.backtest)),
            "kalman_overlay": dict(view.replay.audit),
        }
    )
    metrics["training_diagnostics"] = training
    _write_export_json(metrics_path, metrics)

    manifest_path = destination / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"{manifest_path}: objet JSON attendu.")
    manifest.update(
        {
            "derived_export_model": spec.source_model,
            "native_model": spec.source_model,
            "baseline_model": spec.baseline_model,
            "n_evaluation_days": 365,
            "n_evaluation_hours": int(len(view.backtest)),
            "evaluation_start_local_date": view.evaluation_start_day.isoformat(),
            "evaluation_end_local_date": evaluation_end,
            "production_changed": False,
            "storm_used_for_prediction": False,
            "storm_used_as_input": False,
            "mkonline_used_as_input": False,
        }
    )
    _write_export_json(manifest_path, manifest)


def _statistics_payload_from_html(report_path: Path) -> Mapping[str, object]:
    """Return the Statistics payload embedded in a generated HTML report."""

    rendered = report_path.read_text(encoding="utf-8")
    marker = "const payload = "
    offset = 0
    decoder = json.JSONDecoder()
    while True:
        position = rendered.find(marker, offset)
        if position < 0:
            break
        payload_start = position + len(marker)
        try:
            payload, consumed = decoder.raw_decode(rendered[payload_start:])
        except json.JSONDecodeError:
            offset = payload_start
            continue
        if (
            isinstance(payload, Mapping)
            and isinstance(payload.get("records"), list)
            and isinstance(payload.get("metrics"), list)
        ):
            return payload
        offset = payload_start + consumed
    raise ValueError(
        f"{report_path}: payload JSON de la section Statistics introuvable."
    )


def _validate_statistics_price_report(
    spec: _ExportSpec,
    *,
    reporting_view: Path,
    report_path: Path,
) -> None:
    """Fail closed when the HTML mean-price table is stale or inconsistent.

    The report is a derived artifact and is rebuilt on every launcher call.
    This contract verifies that its complete daily rolling window was actually
    computed from the report-only Statistics snapshot refreshed during that
    same call, including the official Storm pairing when available.
    """

    history_path = reporting_view / "statistics_history_hourly.csv.gz"
    audit_path = reporting_view / "statistics_history_audit.json"
    if not history_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(
            f"{spec.zone}: artefacts Statistics rafraichis absents du rapport."
        )
    history = pd.read_csv(history_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, Mapping):
        raise TypeError(f"{audit_path}: objet JSON attendu.")

    candidate_column = f"{spec.source_model}__q50"
    required_columns = {
        "delivery_start_utc",
        "actual",
        candidate_column,
    }
    missing = sorted(required_columns.difference(history.columns))
    if missing:
        raise ValueError(
            f"{spec.zone}: source Prix moyens incomplete: {', '.join(missing)}."
        )
    benchmark_column = audit.get("storm_primary_report_benchmark")
    benchmark_active = (
        isinstance(benchmark_column, str)
        and benchmark_column in history.columns
    )
    numeric_columns = ["actual", candidate_column]
    if benchmark_active:
        numeric_columns.append(str(benchmark_column))

    delivery = pd.DatetimeIndex(
        pd.to_datetime(
            history["delivery_start_utc"], utc=True, errors="raise"
        )
    )
    if delivery.has_duplicates or not delivery.is_monotonic_increasing:
        raise ValueError(f"{spec.zone}: timeline Statistics invalide.")
    numeric = history.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    local_days = pd.Index(delivery.tz_convert(spec.timezone).date)
    ordered_days = list(local_days.drop_duplicates())
    complete_observed_days = [
        local_day
        for local_day in ordered_days
        if np.isfinite(
            numeric.loc[np.asarray(local_days == local_day), "actual"].to_numpy(
                dtype=float
            )
        ).all()
    ]
    if not complete_observed_days:
        raise ValueError(f"{spec.zone}: aucun prix moyen Statistics calculable.")
    rolling_days = complete_observed_days[-365:]
    cutoff_day = rolling_days[0]
    displayed_days = [day for day in ordered_days if day >= cutoff_day]
    requested_day = date.fromisoformat(
        str(getattr(spec, "delivery_day", displayed_days[-1].isoformat()))
    )
    expected: dict[date, dict[str, float | None]] = {}
    for local_day in displayed_days:
        selector = np.asarray(local_days == local_day, dtype=bool)
        block = numeric.loc[selector]
        actual_values = block["actual"].to_numpy(dtype=float)
        candidate_values = block[candidate_column].to_numpy(dtype=float)
        actual_complete = np.isfinite(actual_values).all()
        actual_empty = not np.isfinite(actual_values).any()
        if not actual_complete and not (
            local_day == requested_day and actual_empty
        ):
            raise ValueError(
                f"{spec.zone}/{local_day}: observations Statistics partielles; "
                "une journee doit etre complete ou vide."
            )
        if not np.isfinite(candidate_values).all():
            raise ValueError(
                f"{spec.zone}/{local_day}: forecast candidat Statistics incomplet."
            )
        if actual_complete:
            paired_mask = np.isfinite(actual_values) & np.isfinite(candidate_values)
            if benchmark_active:
                benchmark_values = block[str(benchmark_column)].to_numpy(dtype=float)
                paired_mask &= np.isfinite(benchmark_values)
            expected_values: dict[str, float | None] = {
                "observed_mean_price": float(np.mean(actual_values[paired_mask])),
                "mean_price": float(np.mean(candidate_values[paired_mask])),
                "benchmark_mean_price": (
                    float(np.mean(benchmark_values[paired_mask]))
                    if benchmark_active
                    else None
                ),
            }
        else:
            benchmark_values = (
                block[str(benchmark_column)].to_numpy(dtype=float)
                if benchmark_active
                else np.asarray([], dtype=float)
            )
            finite_benchmark = benchmark_values[np.isfinite(benchmark_values)]
            expected_values = {
                "observed_mean_price": None,
                "mean_price": float(np.mean(candidate_values)),
                "benchmark_mean_price": (
                    float(np.mean(finite_benchmark))
                    if len(finite_benchmark)
                    else None
                ),
            }
        expected[local_day] = expected_values

    payload = _statistics_payload_from_html(report_path)
    records = payload.get("records")
    assert isinstance(records, list)
    report_daily: dict[str, Mapping[str, object]] = {}
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            continue
        if raw_record.get("zone") != spec.zone or raw_record.get("sample") != "daily":
            continue
        period = str(raw_record.get("period_start") or "")
        if period in report_daily:
            raise ValueError(
                f"{spec.zone}: jour duplique dans Statistics — Prix moyens: {period}."
            )
        report_daily[period] = raw_record

    expected_days = [value.isoformat() for value in expected]
    if list(report_daily) != expected_days:
        raise ValueError(
            f"{spec.zone}: fenetre Prix moyens HTML perimee ou incomplete "
            f"(attendu {expected_days[0]} -> {expected_days[-1]}, "
            f"obtenu {next(iter(report_daily), 'aucun')} -> "
            f"{next(reversed(report_daily), 'aucun')})."
        )

    for local_day, values in expected.items():
        record = report_daily[local_day.isoformat()]
        for report_key in (
            "observed_mean_price",
            "mean_price",
            "benchmark_mean_price",
        ):
            rendered_value = record.get(report_key)
            expected_value = values[report_key]
            matches = (
                rendered_value is None
                if expected_value is None
                else rendered_value is not None
                and np.isclose(
                    float(rendered_value),
                    float(expected_value),
                    rtol=0.0,
                    atol=1e-9,
                )
            )
            if not matches:
                raise ValueError(
                    f"{spec.zone}/{local_day}: {report_key} HTML divergent "
                    "de la source Statistics rafraichie."
                )

    extracted_at = (
        (audit.get("canonical_actuals") or {}).get("source", {}).get(
            "extracted_at_utc"
        )
        if isinstance(audit.get("canonical_actuals"), Mapping)
        else None
    )
    if extracted_at is None or "mean-price-refresh" not in report_path.read_text(
        encoding="utf-8"
    ):
        raise ValueError(
            f"{spec.zone}: preuve de rafraichissement Prix moyens absente du HTML."
        )


def _write_forecast_csv(
    spec: _ExportSpec,
    output_path: Path,
    *,
    source_directory: Path | None = None,
) -> None:
    raw = pd.read_csv(
        _forecast_path(source_directory or spec.archive, spec.zone)
    )
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
    next_column = 3
    if spec.residual_load_source == "chronos2":
        export.insert(next_column, "residual_load_source", "chronos2")
        next_column += 1
    export.insert(next_column, "uses_mkonline", spec.variant == "blend")
    for quantile, source in model_columns.items():
        export[quantile] = pd.to_numeric(raw[source], errors="raise")
    export["price_eur_mwh"] = export["q50"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_path, index=False)


def _read_statistics_start_local(
    reporting_view: Path, *, timezone: str
) -> str | None:
    """Return the first physical day represented in the Statistics view."""

    history_path = reporting_view / "statistics_history_hourly.csv.gz"
    if not history_path.is_file():
        return None
    delivery = pd.to_datetime(
        pd.read_csv(history_path, usecols=["delivery_start_utc"])[
            "delivery_start_utc"
        ],
        utc=True,
        errors="raise",
    )
    if delivery.empty:
        return None
    return delivery.dt.tz_convert(timezone).dt.date.min().isoformat()


def _current_batch_directory(specs: Sequence[_ExportSpec]) -> Path:
    """Resolve and validate the common ``<export-root>/<delivery-day>`` path."""

    roots: set[Path] = set()
    delivery_days = {spec.delivery_day for spec in specs}
    modes = {spec.batch_mode for spec in specs}
    identities: set[tuple[str, str]] = set()
    for spec in specs:
        if spec.csv_path.parent != spec.report_path.parent:
            raise ValueError(
                f"{spec.zone}/{spec.variant}: CSV et HTML hors du meme export."
            )
        variant_directory = spec.csv_path.parent
        zone_directory = variant_directory.parent
        batch_directory = zone_directory.parent
        if variant_directory.name != spec.variant:
            raise ValueError(
                f"{spec.zone}/{spec.variant}: dossier de variante incoherent."
            )
        if zone_directory.name.casefold() != spec.zone.casefold():
            raise ValueError(
                f"{spec.zone}/{spec.variant}: dossier de zone incoherent."
            )
        if batch_directory.name != spec.delivery_day:
            raise ValueError(
                f"{spec.zone}/{spec.variant}: dossier de livraison incoherent."
            )
        identity = (spec.zone, spec.variant)
        if identity in identities:
            raise ValueError(f"Export duplique dans le batch: {identity}.")
        identities.add(identity)
        roots.add(batch_directory.resolve())
    if len(delivery_days) != 1 or len(modes) != 1 or len(roots) != 1:
        raise ValueError(
            "Le manifeste courant exige une livraison, un mode et une racine uniques."
        )
    return roots.pop()


def _relative_manifest_path(path: Path, *, batch_directory: Path) -> str:
    try:
        return path.resolve().relative_to(batch_directory).as_posix()
    except ValueError as exc:
        raise ValueError(f"Fichier d'export hors du batch courant: {path}.") from exc


def _build_current_batch_manifest(
    staged: Sequence[
        tuple[_ExportSpec, Path, Path, tuple[Path, ...], str | None, str | None]
    ],
    *,
    batch_directory: Path,
) -> dict[str, object]:
    """Describe exactly the outputs staged by this invocation."""

    specs = [item[0] for item in staged]
    zones = list(dict.fromkeys(spec.zone for spec in specs))
    exports: list[dict[str, object]] = []
    for (
        spec,
        csv_staged,
        report_staged,
        audits_staged,
        statistics_start_local,
        statistics_end_local,
    ) in staged:
        destination = spec.report_path.parent
        exports.append(
            {
                "zone": spec.zone,
                "variant": spec.variant,
                "source_model": spec.source_model,
                "statistics_start_local": statistics_start_local,
                "statistics_end_local": statistics_end_local,
                "csv": {
                    "path": _relative_manifest_path(
                        spec.csv_path, batch_directory=batch_directory
                    ),
                    "sha256": _sha256_file(csv_staged),
                },
                "html": {
                    "path": _relative_manifest_path(
                        spec.report_path, batch_directory=batch_directory
                    ),
                    "sha256": _sha256_file(report_staged),
                },
                "audits": [
                    {
                        "name": audit.name,
                        "path": _relative_manifest_path(
                            destination / audit.name,
                            batch_directory=batch_directory,
                        ),
                        "sha256": _sha256_file(audit),
                    }
                    for audit in audits_staged
                ],
            }
        )
    return {
        "schema_version": 1,
        "delivery_day": specs[0].delivery_day,
        "mode": specs[0].batch_mode,
        "zones": zones,
        "exports": exports,
    }


def _publish_current_batch_manifest(
    payload: Mapping[str, object],
    *,
    batch_directory: Path,
) -> Path:
    """Atomically replace the current-batch pointer after all files exist."""

    manifest_path = batch_directory / "current_batch_manifest.json"
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".current_batch_manifest.",
        suffix=".tmp",
        dir=batch_directory,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest_path


def _publish_exports(
    specs: Sequence[_ExportSpec],
) -> dict[str, tuple[ForecastExport, ...]]:
    """Build the complete batch in temp space, then publish its files."""

    if not specs:
        return {}
    for spec in specs:
        _validate_export_spec(spec)

    staged: list[
        tuple[
            _ExportSpec,
            Path,
            Path,
            tuple[Path, ...],
            str | None,
            str | None,
        ]
    ] = []
    batch_directory = _current_batch_directory(specs)
    actual_snapshot_cache: dict[
        tuple[str, str, str, str], tuple[pd.Series, Mapping[str, object]]
    ] = {}
    storm_snapshot_cache: dict[
        tuple[str, str, str], tuple[pd.Series, Mapping[str, object]]
    ] = {}
    with tempfile.TemporaryDirectory(prefix="chronos2_forecast_exports_") as raw_tmp:
        temporary_root = Path(raw_tmp).resolve()
        attribution_cache: dict[Path, Path | None] = {}
        prepared: list[tuple[_ExportSpec, Path, Path]] = []
        for index, spec in enumerate(specs):
            attribution_source: Path | None = None
            if (
                spec.lora_candidate is None
            ):
                attribution_source = attribution_cache.get(spec.archive)
            if (
                spec.lora_candidate is None
                and spec.archive not in attribution_cache
            ):
                archive_sources = [
                    spec.archive / filename
                    for filename in _REPORT_ATTRIBUTION_FILES
                ]
                archive_present = [source.is_file() for source in archive_sources]
                if any(archive_present) and not all(archive_present):
                    raise FileNotFoundError(
                        f"{spec.zone}: paire d'attribution incomplete dans "
                        f"{spec.archive}."
                    )
                from chronos2_hourly.report_attribution_cache import cached_attribution, has_historical_prices, report_attribution_sources
                can_reconstruct = (spec.live_config is not None
                                   and (spec.archive / "chronos_live_hourly.csv").is_file())
                if all(archive_present) and (has_historical_prices(spec.archive) or not can_reconstruct):
                    attribution_source = spec.archive
                elif can_reconstruct:
                    print(
                        f"[{spec.zone}] attribution variables + prix passés (cache report-only)...",
                        flush=True,
                    )
                    attribution_source = cached_attribution(
                        root=Path(__file__).resolve().parent / "runs/cache/report_attribution" / spec.zone.lower(),
                        sources=report_attribution_sources(
                            archive=spec.archive, forecast_path=_forecast_path(spec.archive, spec.zone),
                            live_config=spec.live_config, registry_path=(spec.registry_path or DEFAULT_REGISTRY),
                            project_root=Path(__file__).resolve().parent),
                        materialize=lambda destination: _materialize_archived_variable_attribution(
                            spec, destination, registry_path=(spec.registry_path or DEFAULT_REGISTRY),
                            device=spec.device, threads=spec.threads, local_files_only=spec.local_files_only),
                    )
                else:
                    attribution_source = None
                attribution_cache[spec.archive] = attribution_source
            generated = temporary_root / "generated" / str(index)
            csv_staged = generated / spec.csv_path.name
            report_staged = generated / spec.report_path.name
            work = (
                temporary_root
                / f"view_{index}_{spec.zone.lower()}_{spec.variant}"
            )
            _copy_reporting_view(
                spec,
                work,
                attribution_source_dir=attribution_source,
                actual_snapshot_cache=actual_snapshot_cache,
                storm_snapshot_cache=storm_snapshot_cache,
            )
            _materialize_lora_reporting_view(spec, work)
            prepared.append((spec, work, generated))

        # Refresh all countries first: a missing prefix/hour in the last
        # country must not be discovered after hundreds of earlier refits.
        for spec, work, _generated in prepared:
            _preflight_kalman_reporting_view(spec, work)

        for spec, work, generated in prepared:
            csv_staged = generated / spec.csv_path.name
            report_staged = generated / spec.report_path.name
            _materialize_kalman_reporting_view(spec, work)
            statistics_audit_path = work / "statistics_history_audit.json"
            statistics_start_local = _read_statistics_start_local(
                work,
                timezone=spec.timezone,
            )
            statistics_end_local: str | None = (
                date.fromisoformat(spec.delivery_day) - timedelta(days=1)
            ).isoformat()
            if statistics_audit_path.is_file():
                statistics_audit = json.loads(
                    statistics_audit_path.read_text(encoding="utf-8")
                )
                if isinstance(statistics_audit, Mapping):
                    raw_statistics_end = (
                        statistics_audit.get("statistics_through_day_local")
                        or statistics_audit.get("statistics_prefix_end_local")
                    )
                    if raw_statistics_end not in (None, ""):
                        statistics_end_local = str(raw_statistics_end)
            _write_forecast_csv(
                spec,
                csv_staged,
                source_directory=work,
            )
            kalman_artifacts_staged: tuple[Path, ...] = ()
            if spec.variant in KALMAN_VARIANTS:
                kalman_artifact_names = (
                    "kalman_filter_audit.json",
                    "kalman_operational_sidecar_audit.json",
                )
                missing_kalman_artifacts = [
                    name
                    for name in kalman_artifact_names
                    if not (work / name).is_file()
                ]
                if missing_kalman_artifacts:
                    raise FileNotFoundError(
                        f"{spec.zone}/{spec.variant}: audits non materialises: "
                        f"{missing_kalman_artifacts}."
                    )
                kalman_artifacts_staged = tuple(
                    generated / name for name in kalman_artifact_names
                )
                for staged_artifact in kalman_artifacts_staged:
                    shutil.copy2(work / staged_artifact.name, staged_artifact)
            if spec.lora_candidate is not None:
                lora_artifact = generated / "exogenous_lora_overlay_audit.json"
                shutil.copy2(work / lora_artifact.name, lora_artifact)
                kalman_artifacts_staged = (
                    *kalman_artifacts_staged,
                    lora_artifact,
                )
            source_suffix = (
                " - challenger residual_load Chronos-2"
                if spec.residual_load_source == "chronos2"
                else ""
            )
            report_labels = {
                "autonomous": "autonome + correcteur residuel",
                "blend": "Chronos-2 + MKOnline",
                "kalman": (
                    "autonome + correcteur residuel + Kalman gouverne"
                ),
                "kalman_weather": (
                    "autonome + correcteur residuel + Kalman meteo gouverne"
                ),
                "kalman_hybrid": (
                    "autonome + correcteur residuel + banque Kalman "
                    "marche-meteo-combustibles gouvernee"
                ),
            }
            if spec.lora_candidate is not None:
                report_labels["autonomous"] = (
                    "Chronos-2 + LoRA exogene + correcteur residuel"
                )
                report_labels["kalman"] = (
                    "Chronos-2 + LoRA exogene + correcteur residuel + "
                    "Kalman gouverne"
                )
            write_hourly_html_report(
                work,
                output_path=report_staged,
                title=(
                    f"Forecast {spec.zone} {spec.delivery_day} - "
                    + report_labels[spec.variant]
                    + source_suffix
                ),
                native_model=spec.source_model,
                baseline_model=spec.baseline_model,
                zone=spec.zone,
                timezone=spec.timezone,
            )
            _validate_statistics_price_report(
                spec,
                reporting_view=work,
                report_path=report_staged,
            )
            if not csv_staged.is_file() or csv_staged.stat().st_size <= 0:
                raise RuntimeError(f"Export CSV vide: {spec.zone}/{spec.variant}.")
            if not report_staged.is_file() or report_staged.stat().st_size <= 0:
                raise RuntimeError(f"Rapport HTML vide: {spec.zone}/{spec.variant}.")
            staged.append(
                (
                    spec,
                    csv_staged,
                    report_staged,
                    kalman_artifacts_staged,
                    statistics_start_local,
                    statistics_end_local,
                )
            )

        statistics_ends_by_zone: dict[str, set[str]] = {}
        for (
            spec,
            _csv,
            _report,
            _audit,
            _statistics_start_local,
            statistics_end_local,
        ) in staged:
            if statistics_end_local not in (None, ""):
                statistics_ends_by_zone.setdefault(spec.zone, set()).add(
                    str(statistics_end_local)
                )
        inconsistent_zones = {
            zone: sorted(ends)
            for zone, ends in statistics_ends_by_zone.items()
            if len(ends) != 1
        }
        if inconsistent_zones:
            details = "; ".join(
                f"{zone}={','.join(ends)}"
                for zone, ends in sorted(inconsistent_zones.items())
            )
            raise RuntimeError(
                "Les variantes exportees n'utilisent pas la meme borne "
                f"Statistics ({details})."
            )
        manifest_payload = _build_current_batch_manifest(
            staged,
            batch_directory=batch_directory,
        )

        # No runs/live path is touched: only derived, reproducible files are
        # atomically replaced after the whole export batch has succeeded.
        for (
            spec,
            csv_staged,
            report_staged,
            kalman_artifacts_staged,
            _statistics_start_local,
            _statistics_end_local,
        ) in staged:
            spec.csv_path.parent.mkdir(parents=True, exist_ok=True)
            spec.report_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(csv_staged, spec.csv_path)
            os.replace(report_staged, spec.report_path)
            for staged_artifact in kalman_artifacts_staged:
                os.replace(
                    staged_artifact,
                    spec.report_path.parent / staged_artifact.name,
                )
        for record in manifest_payload["exports"]:
            if not isinstance(record, Mapping):
                raise TypeError("Entree de manifeste export invalide.")
            for key in ("csv", "html"):
                file_record = record.get(key)
                if not isinstance(file_record, Mapping):
                    raise TypeError(f"Manifeste export: entree {key} invalide.")
                published_path = batch_directory / str(file_record["path"])
                if _sha256_file(published_path) != str(file_record["sha256"]):
                    raise RuntimeError(
                        f"Controle SHA256 post-publication refuse: {published_path}."
                    )
            audits = record.get("audits", [])
            if not isinstance(audits, list):
                raise TypeError("Manifeste export: liste audits invalide.")
            for audit_record in audits:
                if not isinstance(audit_record, Mapping):
                    raise TypeError("Manifeste export: audit invalide.")
                published_path = batch_directory / str(audit_record["path"])
                if _sha256_file(published_path) != str(audit_record["sha256"]):
                    raise RuntimeError(
                        f"Controle SHA256 post-publication refuse: {published_path}."
                    )
        _publish_current_batch_manifest(
            manifest_payload,
            batch_directory=batch_directory,
        )

    statistics_end_by_export = {
        (spec.zone, spec.variant): statistics_end_local
        for (
            spec,
            _csv,
            _report,
            _audit,
            _statistics_start_local,
            statistics_end_local,
        ) in staged
    }
    by_zone: dict[str, list[ForecastExport]] = {}
    for spec in specs:
        by_zone.setdefault(spec.zone, []).append(
            ForecastExport(
                variant=spec.variant,
                source_model=spec.source_model,
                csv_path=spec.csv_path,
                report_path=spec.report_path,
                source_archive=spec.archive,
                residual_load_source=spec.residual_load_source,
                statistics_end_local=statistics_end_by_export.get(
                    (spec.zone, spec.variant)
                ),
                kalman_audit_path=(
                    spec.report_path.parent
                    / "kalman_operational_sidecar_audit.json"
                    if spec.variant in KALMAN_VARIANTS
                    else None
                ),
                lora_audit_path=(
                    spec.report_path.parent / "exogenous_lora_overlay_audit.json"
                    if spec.lora_candidate is not None
                    else None
                ),
            )
        )
    return {zone: tuple(exports) for zone, exports in by_zone.items()}


def _statistics_completion_message(
    exports: Sequence[ForecastExport],
) -> str:
    ends = {
        str(export.statistics_end_local)
        for export in exports
        if export.statistics_end_local not in (None, "")
    }
    if not ends:
        return ""
    if len(ends) != 1:
        raise RuntimeError(
            "Les variantes exportees n'utilisent pas la meme borne Statistics."
        )
    return " Statistics des rapports completes jusqu'au " + ends.pop() + "."


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
    residual_load_source: str = "saturn",
    residual_load_bundle_manifest: Path | None = None,
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
            residual_load_source=residual_load_source,
            residual_load_bundle_manifest=residual_load_bundle_manifest,
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


def _complete_statistics_archives(
    *,
    zones: tuple[str, ...],
    project_root: Path,
    registry_path: Path,
    python_executable: Path,
    device: str,
    threads: int,
    workers: int,
    local_files_only: bool,
) -> None:
    """Explicitly rebuild every causal gap before Both-mode exports."""

    launcher = project_root / "run_statistics_backfill.py"
    if not launcher.is_file():
        # Lightweight isolated fixtures intentionally omit the real backfill
        # launcher. Production installations always include it.
        return
    command = [
        str(python_executable),
        str(launcher),
        "--zones",
        *zones,
        "--registry",
        str(registry_path),
        "--python-executable",
        str(python_executable),
        "--device",
        device,
        "--threads",
        str(int(threads)),
        "--workers",
        str(int(workers)),
    ]
    if not local_files_only:
        command.append("--allow-model-download")
    print(
        "\nMise a jour causale des archives Backtest/Statistics...",
        flush=True,
    )
    completed = subprocess.run(command, cwd=project_root, check=False)
    if int(completed.returncode) != 0:
        raise RuntimeError(
            "Le rattrapage causal des Backtests/Statistics a echoue "
            f"(code={completed.returncode})."
        )


def _prepare_kalman_weather_runtime_configs(
    *,
    zones: tuple[str, ...],
    delivery_day: str,
    project_root: Path,
    python_executable: Path,
    template_path: Path,
    threads: int,
    workers: int,
) -> dict[str, Path]:
    """Refresh PIT weather and render checksum-pinned per-zone sidecars."""

    from chronos2_hourly.kalman_configuration import (
        KALMAN_WEATHER_ZONES,
        render_kalman_weather_operational_configuration,
    )

    materializer = project_root / "materialize_saturn_kalman_weather.py"
    if not materializer.is_file():
        raise FileNotFoundError(
            f"Materializer Kalman meteo introuvable: {materializer}."
        )
    missing_upstream: list[str] = []
    weather_data_root = project_root / "data" / "pit" / "kalman_weather"
    for zone in zones:
        lower = zone.casefold()
        for suffix in (
            "residual_corrected_prequential.csv.gz",
            "residual_corrected_prequential.audit.json",
        ):
            path = weather_data_root / f"{lower}_{suffix}"
            if not path.is_file():
                missing_upstream.append(str(path))
    if missing_upstream:
        raise FileNotFoundError(
            "Prefixes prequentiels residual_corrected absents pour le Kalman "
            "meteo: " + ", ".join(missing_upstream)
        )

    requested_day = date.fromisoformat(delivery_day)
    now_utc = pd.Timestamp.now(tz="UTC")
    future_cutoffs: list[str] = []
    for zone in zones:
        timezone = KALMAN_WEATHER_ZONES[zone]
        cutoff = (
            pd.Timestamp(requested_day - timedelta(days=1))
            + pd.Timedelta(hours=8)
        ).tz_localize(timezone, ambiguous="raise", nonexistent="raise").tz_convert(
            "UTC"
        )
        if cutoff > now_utc:
            future_cutoffs.append(f"{zone}={cutoff.isoformat()}")
    if future_cutoffs:
        raise RuntimeError(
            "Kalman meteo refuse: le cutoff causal D-1 08:00 n'est pas encore "
            "atteint pour " + ", ".join(future_cutoffs) + "."
        )
    required_start = requested_day - timedelta(days=730)
    start_day = min(KALMAN_WEATHER_HISTORY_START_DAY, required_start)
    series_workers = min(max(1, int(workers)), 4, len(zones) * 3)
    day_workers = min(max(1, int(threads)), 8, max(1, 32 // series_workers))
    command = [
        str(python_executable),
        str(materializer),
        "--start-day",
        start_day.isoformat(),
        "--end-day",
        delivery_day,
        "--zones",
        *zones,
        "--output-dir",
        str(weather_data_root),
        "--series-workers",
        str(series_workers),
        "--day-workers",
        str(day_workers),
    ]
    print(
        "\n[KALMAN-WEATHER] Actualisation causale Saturn "
        f"{start_day.isoformat()} -> {delivery_day} pour {', '.join(zones)}...",
        flush=True,
    )
    completed = subprocess.run(command, cwd=project_root, check=False)
    if int(completed.returncode) != 0:
        raise RuntimeError(
            "La materialisation Saturn du Kalman meteo a echoue "
            f"(code={completed.returncode})."
        )

    runtime_root = (
        project_root
        / "runs"
        / "runtime"
        / "kalman_weather"
        / delivery_day
    )
    output: dict[str, Path] = {}
    for zone in zones:
        sidecar = runtime_root / zone.casefold() / "kalman_weather_operational.yaml"
        configuration = render_kalman_weather_operational_configuration(
            template_path,
            zone=zone,
            delivery_day=delivery_day,
            output_path=sidecar,
            project_root=project_root,
        )
        output[zone] = configuration.path
        print(
            f"[KALMAN-WEATHER] {zone} sidecar audite: {configuration.path}",
            flush=True,
        )
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_kalman_hybrid_sources(
    sources: Sequence[Path],
    *,
    runtime_root: Path,
) -> Path:
    """Create one content-addressed immutable source bundle for a sidecar."""

    records: list[dict[str, object]] = []
    names: set[str] = set()
    for raw_source in sources:
        source = Path(raw_source).expanduser().resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Source runtime Kalman hybride absente ou vide: {source}."
            )
        if source.name in names:
            raise ValueError(
                f"Nom de source runtime Kalman hybride duplique: {source.name}."
            )
        names.add(source.name)
        records.append(
            {
                "name": source.name,
                "source": str(source),
                "sha256": _sha256_file(source),
                "size_bytes": int(source.stat().st_size),
            }
        )
    records.sort(key=lambda item: str(item["name"]))
    bundle_sha = hashlib.sha256(
        json.dumps(records, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    parent = (runtime_root / "sources").resolve()
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / bundle_sha

    def validate_snapshot() -> None:
        manifest_path = destination / "source_bundle_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Snapshot Kalman hybride incomplet: {destination}."
            ) from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("bundle_sha256") != bundle_sha
            or manifest.get("files") != records
        ):
            raise ValueError(
                f"Manifest de snapshot Kalman hybride incompatible: {destination}."
            )
        for record in records:
            target = destination / str(record["name"])
            if (
                not target.is_file()
                or int(target.stat().st_size) != int(record["size_bytes"])
                or _sha256_file(target) != record["sha256"]
            ):
                raise ValueError(
                    f"Snapshot Kalman hybride altere: {target}."
                )

    if destination.is_dir():
        validate_snapshot()
        return destination

    temporary = Path(
        tempfile.mkdtemp(prefix=".hybrid-sources-", dir=parent)
    ).resolve()
    try:
        for record in records:
            source = Path(str(record["source"]))
            target = temporary / str(record["name"])
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        (temporary / "source_bundle_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "bundle_sha256": bundle_sha,
                    "files": records,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            os.replace(temporary, destination)
        except OSError:
            if not destination.is_dir():
                raise
        validate_snapshot()
    finally:
        # The temporary directory is flat and is always created immediately
        # below the validated runtime source parent.
        if temporary.exists() and temporary.parent == parent:
            for child in temporary.iterdir():
                if child.is_file():
                    child.unlink()
            temporary.rmdir()
    return destination


def _prepare_kalman_hybrid_runtime_configs(
    *,
    zones: tuple[str, ...],
    delivery_day: str,
    project_root: Path,
    python_executable: Path,
    template_path: Path,
    threads: int,
    workers: int,
) -> dict[str, Path]:
    """Refresh common residual-load/fuel inputs and pin hybrid sidecars.

    Weather materializations and the causal prequential upstream prefixes are
    deliberately reused from the immediately preceding Kalman-weather
    preparation.  The materializer writes common PIT tables for the five
    residual-load forecasts and for TTF/EUA; the complete input set is then
    snapshotted under one content hash before each rendered sidecar pins every
    referenced source checksum.
    """

    from chronos2_hourly.kalman_configuration import (
        KALMAN_WEATHER_ZONES,
        render_kalman_weather_operational_configuration,
    )

    materializer = project_root / "materialize_saturn_kalman_fuel.py"
    if not materializer.is_file():
        raise FileNotFoundError(
            f"Materializer Kalman hybride introuvable: {materializer}."
        )

    requested_day = date.fromisoformat(delivery_day)
    now_utc = pd.Timestamp.now(tz="UTC")
    future_cutoffs: list[str] = []
    for zone in zones:
        timezone = KALMAN_WEATHER_ZONES[zone]
        cutoff = (
            pd.Timestamp(requested_day - timedelta(days=1))
            + pd.Timedelta(hours=8)
        ).tz_localize(timezone, ambiguous="raise", nonexistent="raise").tz_convert(
            "UTC"
        )
        if cutoff > now_utc:
            future_cutoffs.append(f"{zone}={cutoff.isoformat()}")
    if future_cutoffs:
        raise RuntimeError(
            "Kalman hybride refuse: le cutoff causal D-1 08:00 n'est pas "
            "encore atteint pour " + ", ".join(future_cutoffs) + "."
        )

    weather_root = project_root / "data" / "pit" / "kalman_weather"
    weather_paths = [
        weather_root / f"{zone.casefold()}_{kind}_fcst.parquet"
        for zone in zones
        for kind in ("temperature", "wind_generation", "solar_generation")
    ]
    weather_audit_paths = [
        path.with_name(path.name + ".audit.json") for path in weather_paths
    ]
    upstream_paths = [
        weather_root
        / f"{zone.casefold()}_residual_corrected_prequential.csv.gz"
        for zone in zones
    ]
    upstream_audit_paths = [
        weather_root
        / f"{zone.casefold()}_residual_corrected_prequential.audit.json"
        for zone in zones
    ]
    missing_weather = [
        str(path)
        for path in (
            *weather_paths,
            *weather_audit_paths,
            *upstream_paths,
            *upstream_audit_paths,
        )
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing_weather:
        raise FileNotFoundError(
            "Le Kalman hybride exige les vintages meteo et prefixes "
            "prequentiels deja actualises: "
            + ", ".join(missing_weather)
        )

    required_start = requested_day - timedelta(days=730)
    start_day = min(KALMAN_WEATHER_HISTORY_START_DAY, required_start)
    output_root = project_root / "data" / "pit" / "kalman_hybrid"
    # The residual bank always contains five countries.  Spread the bounded
    # source budget over all five first, otherwise the fifth series would wait
    # for an entire second historical wave when the default worker count is 4.
    source_worker_budget = min(
        32,
        max(1, int(workers)) * max(1, int(threads)),
    )
    series_workers = min(5, source_worker_budget)
    day_workers = min(8, max(1, source_worker_budget // series_workers))
    command = [
        str(python_executable),
        str(materializer),
        "--start-day",
        start_day.isoformat(),
        "--residual-start-day",
        required_start.isoformat(),
        "--end-day",
        delivery_day,
        "--output-dir",
        str(output_root),
        "--series-workers",
        str(series_workers),
        "--day-workers",
        str(day_workers),
    ]
    print(
        "\n[KALMAN-HYBRID] Actualisation causale charges residuelles/gaz/CO2 "
        f"{start_day.isoformat()} -> {delivery_day}...",
        flush=True,
    )
    completed = subprocess.run(command, cwd=project_root, check=False)
    if int(completed.returncode) != 0:
        raise RuntimeError(
            "La materialisation gaz/CO2 du Kalman hybride a echoue "
            f"(code={completed.returncode})."
        )
    fuel_path = output_root / "market_fuel_features.parquet"
    fuel_audit_path = fuel_path.with_name(fuel_path.name + ".audit.json")
    if (
        not fuel_path.is_file()
        or fuel_path.stat().st_size <= 0
        or not fuel_audit_path.is_file()
        or fuel_audit_path.stat().st_size <= 0
    ):
        raise FileNotFoundError(
            "Bundle PIT gaz/CO2 du Kalman hybride absent ou vide: "
            f"{fuel_path}, {fuel_audit_path}."
        )
    try:
        fuel_audit = json.loads(fuel_audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Audit PIT gaz/CO2 illisible: {fuel_audit_path}."
        ) from exc
    fuel_sha256 = hashlib.sha256(fuel_path.read_bytes()).hexdigest()
    if (
        not isinstance(fuel_audit, dict)
        or fuel_audit.get("sha256") != fuel_sha256
        or fuel_audit.get("information_type")
        != "market_observation_known_before_cutoff"
        or fuel_audit.get("causality_violations") != 0
        or str(fuel_audit.get("end_day", "")) < delivery_day
    ):
        raise ValueError(
            "Audit PIT gaz/CO2 incompatible, non causal ou checksum invalide: "
            f"{fuel_audit_path}."
        )

    residual_path = output_root / "residual_load_market_features.parquet"
    residual_audit_path = residual_path.with_name(
        residual_path.name + ".audit.json"
    )
    if (
        not residual_path.is_file()
        or residual_path.stat().st_size <= 0
        or not residual_audit_path.is_file()
        or residual_audit_path.stat().st_size <= 0
    ):
        raise FileNotFoundError(
            "Bundle PIT de charge residuelle du Kalman hybride absent ou "
            f"vide: {residual_path}, {residual_audit_path}."
        )
    try:
        residual_audit = json.loads(
            residual_audit_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Audit PIT de charge residuelle illisible: {residual_audit_path}."
        ) from exc
    if (
        not isinstance(residual_audit, dict)
        or residual_audit.get("sha256") != _sha256_file(residual_path)
        or residual_audit.get("causality_violations") != 0
        or str(residual_audit.get("end_day", "")) < delivery_day
    ):
        raise ValueError(
            "Audit PIT de charge residuelle incompatible, non causal ou "
            f"checksum invalide: {residual_audit_path}."
        )

    runtime_root = (
        project_root / "runs" / "runtime" / "kalman_hybrid" / delivery_day
    )
    immutable_source_root = _snapshot_kalman_hybrid_sources(
        (
            fuel_path,
            fuel_audit_path,
            residual_path,
            residual_audit_path,
            *weather_paths,
            *weather_audit_paths,
            *upstream_paths,
            *upstream_audit_paths,
        ),
        runtime_root=runtime_root,
    )
    output: dict[str, Path] = {}
    for zone in zones:
        sidecar = runtime_root / zone.casefold() / "kalman_hybrid_operational.yaml"
        configuration = render_kalman_weather_operational_configuration(
            template_path,
            zone=zone,
            delivery_day=delivery_day,
            output_path=sidecar,
            project_root=project_root,
            runtime_source_root=immutable_source_root,
        )
        output[zone] = configuration.path
        print(
            f"[KALMAN-HYBRID] {zone} sidecar audite: {configuration.path}",
            flush=True,
        )
    return output


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
    residual_load_source: str = "saturn",
    kalman_config: str | Path = DEFAULT_KALMAN_CONFIG,
    lora_activation_config: str | Path = DEFAULT_LORA_ACTIVATION_CONFIG,
) -> BatchForecastResult:
    """Run selected countries sequentially and audit every detailed report."""

    selected = normalize_zones(zones)
    delivery = normalise_delivery_day(delivery_day)
    selected_mode = normalise_forecast_mode(mode)
    selected_residual_source = normalize_residual_load_source(
        residual_load_source
    )
    if selected_mode in {"both", "all"} and selected_residual_source != "saturn":
        raise ValueError(
            f"Le mode {selected_mode.title()} exige ResidualLoadSource Saturn: son replay Kalman "
            "doit utiliser le meme historique causal que le forecast autonome."
        )
    # This validates the complete selection before inspecting, launching or
    # creating any output. Both exports autonomous + standard Kalman in every
    # country, using incumbent routes when LoRA is explicitly inactive. All
    # requires promoted, active LoRA routes for the same two variants below.
    for zone in selected:
        requested_export_variants(selected_mode, zone)
    root = Path(project_root).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    executable = Path(python_executable).expanduser().resolve()
    logs = Path(log_dir).expanduser().resolve()
    exports_root = Path(export_root).expanduser().resolve()
    raw_kalman_config = Path(kalman_config).expanduser()
    resolved_kalman_config = (
        raw_kalman_config
        if raw_kalman_config.is_absolute()
        else root / raw_kalman_config
    ).resolve()
    raw_lora_activation = Path(lora_activation_config).expanduser()
    resolved_lora_activation = (
        raw_lora_activation
        if raw_lora_activation.is_absolute()
        else root / raw_lora_activation
    ).resolve()
    if selected_residual_source == "chronos2":
        exports_root = (
            exports_root / "residual_load_chronos2"
        ).resolve()
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
    # Production and Blend never consume the autonomous LoRA branch.  Do not
    # make those independent modes depend on an activation file (or on a
    # bundle registered for another country) that they cannot use.
    lora_requested = any(
        variant in {"autonomous", "kalman"}
        for zone in selected
        for variant in requested_export_variants(selected_mode, zone)
    )
    lora_contract = (
        load_activation_contract(resolved_lora_activation)
        if lora_requested
        else None
    )
    lora_routes = (
        _resolve_lora_routes(
            lora_contract,
            zones=selected,
            mode=selected_mode,
        )
        if lora_contract is not None
        else {}
    )
    active_lora_routes = {
        key: value for key, value in lora_routes.items() if value.lora_enabled
    }
    if (
        active_lora_routes
        and lora_contract is not None
        and lora_contract.project_root != root
    ):
        raise ValueError(
            "Le project_root du contrat LoRA actif differe du projet lance."
        )
    if active_lora_routes and selected_residual_source != "saturn":
        raise ValueError(
            "LoRA operationnel exige ResidualLoadSource Saturn; le challenger "
            "de charge residuelle reste un protocole separe."
        )
    if selected_mode == "all":
        missing_lora = [
            f"{zone}/{variant}"
            for zone in selected
            for variant in ("autonomous", "kalman")
            if not lora_routes[(zone, variant)].lora_enabled
        ]
        if missing_lora:
            raise ValueError(
                "Mode All refuse: aucun fallback incumbent n'est autorise pour "
                "les chaines LoRA. Bundle final promu non active pour: "
                + ", ".join(missing_lora)
                + "."
            )
    if selected_mode in {"both", "all"}:
        # Validate the isolated sidecar before any country runner starts.  It
        # never alters the sealed live-model configuration.
        from chronos2_hourly.kalman_configuration import (
            load_kalman_operational_configuration,
        )

        for zone in selected:
            route = lora_routes.get((zone, "kalman"))
            upstream_model = (
                route.kalman_upstream_model
                if route is not None and route.lora_enabled
                else KALMAN_UPSTREAM_MODEL
            )
            load_kalman_operational_configuration(
                resolved_kalman_config,
                project_root=root,
                zone=zone,
                upstream_model=upstream_model,
            )
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
            residual_load_source=selected_residual_source,
        )

    try:
        lora_candidates = (
            _prepare_lora_candidates(
                lora_routes,
                contract=lora_contract,
                delivery_day=delivery,
                project_root=root,
                python_executable=executable,
                device=device,
            )
            if active_lora_routes and not dry_run
            else {}
        )
    except Exception as exc:
        results = tuple(
            BatchZoneResult(
                zone=zone,
                delivery_day=delivery,
                state="failed",
                return_code=2,
                message=(
                    "Preflight LoRA refuse avant tout run live: " + str(exc)
                ),
            )
            for zone in selected
        )
        return BatchForecastResult(
            delivery_day=delivery,
            zones=selected,
            results=results,
            mode=selected_mode,
            residual_load_source=selected_residual_source,
        )

    # ``kalman_weather`` and ``kalman_hybrid`` remain available to their
    # dedicated experiment runners, but are deliberately no longer prepared
    # by the operational Both/All paths. Both permits the incumbent chains;
    # All remains limited to promoted LoRA chains. Neither requires the two
    # large experimental daily materialisations.
    kalman_weather_configs: dict[str, Path] | None = None
    kalman_hybrid_configs: dict[str, Path] | None = None

    residual_bundle_manifest: Path | None = None
    if selected_residual_source == "chronos2":
        residual_bundle_manifest = _build_residual_load_bundle_once(
            project_root=root,
            delivery_day=delivery,
            device=device,
            local_files_only=local_files_only,
            dry_run=dry_run,
        )

    if not dry_run and selected_mode != "production":
        source_failures: dict[str, str] = {}
        for zone in selected:
            try:
                existing = validate_existing_forecast_archive(
                    statuses[zone],
                    project_root=root,
                    delivery_day=delivery,
                    residual_load_source=selected_residual_source,
                    residual_load_bundle_manifest=residual_bundle_manifest,
                )
                if existing is None:
                    continue
                _report_or_deferred_export(
                    existing,
                    mode=selected_mode,
                    residual_load_source=selected_residual_source,
                )
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
                    residual_load_source=selected_residual_source,
                    registry_path=registry,
                    device=device,
                    threads=int(threads),
                    local_files_only=local_files_only,
                    kalman_config=resolved_kalman_config,
                    kalman_weather_configs=kalman_weather_configs,
                    kalman_hybrid_configs=kalman_hybrid_configs,
                    lora_routes=lora_routes,
                    lora_candidates=lora_candidates,
                    project_root=root,
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
                residual_load_source=selected_residual_source,
                residual_load_bundle_manifest=residual_bundle_manifest,
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
                    residual_load_source=selected_residual_source,
                    residual_load_bundle_manifest=residual_bundle_manifest,
                )
            )
            continue

        print(f"\n[{zone}] lancement pour la livraison {delivery}...", flush=True)
        launched: ForecastProcess | ForecastSkip | None = None
        process_return_code: int | None = None
        archive_after_runner: Path | None = None
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
                residual_load_source=selected_residual_source,
                residual_load_bundle_manifest=residual_bundle_manifest,
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
                # Two Forecast.ps1 invocations can pass the pre-dispatch
                # existence check before either immutable directory is
                # published.  On Windows the process losing that publication
                # race exits with WinError 5 because directory replacement is
                # deliberately forbidden.  Reuse the winner only after the
                # complete archive identity/checksum audit; every genuine
                # runner failure still follows the normal error path.
                try:
                    archive_after_runner = validate_existing_forecast_archive(
                        status,
                        project_root=root,
                        delivery_day=delivery,
                        residual_load_source=selected_residual_source,
                        residual_load_bundle_manifest=residual_bundle_manifest,
                    )
                except Exception as archive_exc:
                    message += (
                        " L'archive apparue apres l'echec est invalide: "
                        f"{archive_exc}"
                    )
                if archive_after_runner is None:
                    if log_path is not None:
                        tail = read_log_tail(log_path, max_chars=5_000).strip()
                        if tail:
                            message += "\n" + tail
                    raise RuntimeError(message)
                original_return_code = return_code
                return_code = 0
                process_return_code = 0
                state = "skipped"
                message = (
                    "Une execution concurrente a publie la meme archive "
                    f"pendant ce calcul (runner={original_return_code}); "
                    "archive integralement validee et reutilisee."
                )
            archive = archive_after_runner or validate_existing_forecast_archive(
                status,
                project_root=root,
                delivery_day=delivery,
                residual_load_source=selected_residual_source,
                residual_load_bundle_manifest=residual_bundle_manifest,
            )
            if archive is None:
                raise RuntimeError(
                    "Le runner a reussi mais aucune archive immutable n'a ete trouvee."
                )
            report = _report_or_deferred_export(
                archive,
                mode=selected_mode,
                residual_load_source=selected_residual_source,
            )
            if report is None:
                message += (
                    " Rapport archive degrade; reconstruction detaillee "
                    "planifiee dans les exports, sans modifier l'archive live."
                )
            summary_payload = json.loads(
                (archive / "live_run_summary.json").read_text(encoding="utf-8")
            )
            summary_status = str(summary_payload.get("status", "complete"))
            if summary_status != "complete" and selected_mode not in {"both", "all"}:
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
            if report is not None:
                print(f"[{zone}] rapport HTML: {report}", flush=True)
            else:
                print(
                    f"[{zone}] rapport HTML detaille a reconstruire dans les exports.",
                    flush=True,
                )
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
                if (
                    selected_mode in {"both", "all"}
                    and selected_residual_source == "saturn"
                ):
                    _complete_statistics_archives(
                        zones=selected,
                        project_root=root,
                        registry_path=registry,
                        python_executable=executable,
                        device=device,
                        threads=int(threads),
                        workers=int(workers),
                        local_files_only=local_files_only,
                    )
                specs = _build_export_specs(
                    zone_results=results,
                    statuses=statuses,
                    mode=selected_mode,
                    delivery_day=delivery,
                    export_root=exports_root,
                    residual_load_source=selected_residual_source,
                    registry_path=registry,
                    device=device,
                    threads=int(threads),
                    local_files_only=local_files_only,
                    kalman_config=resolved_kalman_config,
                    kalman_weather_configs=kalman_weather_configs,
                    kalman_hybrid_configs=kalman_hybrid_configs,
                    lora_routes=lora_routes,
                    lora_candidates=lora_candidates,
                    project_root=root,
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
                            + (
                                _statistics_completion_message(
                                    published.get(result.zone, ())
                                )
                                if (
                                    selected_mode in {"both", "all"}
                                    and selected_residual_source == "saturn"
                                )
                                else ""
                            )
                            + (
                                " Challenger prospectif: forecast J+1 "
                                "Chronos-2; historique et Statistics du "
                                "controle Saturn scelle, affiches uniquement "
                                "comme reference aval."
                                if selected_residual_source == "chronos2"
                                else ""
                            )
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
        residual_load_source=selected_residual_source,
        residual_load_bundle_manifest=residual_bundle_manifest,
    )


def print_batch_summary(batch: BatchForecastResult) -> None:
    print("\nResultat du batch")
    print("=" * 88)
    print(f"Livraison: {batch.delivery_day}")
    print(f"Mode:      {batch.mode}")
    if batch.residual_load_source == "chronos2":
        print("Charge:    chronos2")
        if batch.residual_load_bundle_manifest is not None:
            print(f"Bundle:    {batch.residual_load_bundle_manifest}")
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
            if forecast_export.kalman_audit_path is not None:
                print(
                    "   Audit Kalman: "
                    f"{forecast_export.kalman_audit_path}"
                )
            if forecast_export.lora_audit_path is not None:
                print(
                    "   Audit LoRA:   "
                    f"{forecast_export.lora_audit_path}"
                )
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
        "--residual-load-source",
        choices=("saturn", "chronos2"),
        default="saturn",
        help=(
            "saturn conserve le run de production; chronos2 publie un "
            "challenger isole avec forecasts amont de charge residuelle."
        ),
    )
    parser.add_argument(
        "--mode",
        default="Production",
        help=(
            "Production conserve le run audite; Autonomous exporte la variante "
            "sans MKOnline; Blend est limite a FR/NL; Both exporte l'autonome "
            "corrige et sa variante Kalman standard pour chaque pays, sans "
            "exiger LoRA lorsque son activation est desactivee; All exige et "
            "exporte exactement les deux chaines promues LoRA: autonome "
            "corrigee et autonome corrigee + Kalman standard."
        ),
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument(
        "--kalman-config",
        default=str(DEFAULT_KALMAN_CONFIG),
        help=(
            "Sidecar YAML autonome du filtre Kalman; utilise seulement par "
            "les modes Both et All et jamais par le contrat live scelle."
        ),
    )
    parser.add_argument(
        "--lora-activation-config",
        default=str(DEFAULT_LORA_ACTIVATION_CONFIG),
        help=(
            "Contrat fail-closed des bundles LoRA promus et des sources "
            "prospectives. En mode All, autonomous et kalman doivent etre "
            "actifs pour chaque pays."
        ),
    )
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
    # Windows PowerShell may expose a legacy code page even when runner logs
    # contain UTF-8 replacement characters.  Reporting an upstream failure
    # must never abort the remaining countries because of console encoding.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
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
            residual_load_source=args.residual_load_source,
            kalman_config=args.kalman_config,
            lora_activation_config=args.lora_activation_config,
        )
    except Exception as exc:
        print(f"Erreur launcher: {exc}", file=sys.stderr)
        return 2
    print_batch_summary(batch)
    return 0 if batch.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
