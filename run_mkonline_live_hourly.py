#!/usr/bin/env python
"""Produce one leakage-safe FR day-ahead forecast with the frozen model.

This operational runner deliberately keeps the validated annual benchmark
immutable.  For one delivery day it refreshes the target and PIT covariates,
runs Chronos-2 once, refits the frozen EXT223 + OOF730 residual recipe, and
combines the result with the terminal MKOnline primary series ``41551_native``.

Storm is never read by the prediction path. After the candidate CSV has been
frozen, the Storm day-ahead dashboard cache is fetched independently for
Statistics and the comparison chart. It is never passed to Chronos, the
residual corrector, the MKOnline blend, or the live forecast output.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import date
import hashlib
from html import escape
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd

from chronos2_hourly.chronos_adapter import (
    build_delivery_plan,
    run_existing_live_forecast,
)
from chronos2_hourly.hourly_contract import (
    build_delivery_metadata,
    local_delivery_day_index,
)
from chronos2_hourly.live_target_context import audit_live_target_context
from chronos2_hourly.live_history import (
    missing_statistics_archive_days,
    update_live_statistics_history,
)
from chronos2_hourly.reporting import write_hourly_html_report
from chronos2_hourly.shadow_reporting import (
    write_forecast_only_shadow_report,
)
from chronos2_hourly.storm_dashboard import fetch_native_dashboard_snapshot
from chronos2_hourly.variable_attribution import (
    remove_variable_attribution_artifacts,
    write_variable_attribution,
)
from chronos2_modular.common import (
    build_zone_configs,
    deep_get,
    load_yaml,
    resolve_path,
    set_reproducibility,
)
from chronos2_modular.data import prepare_zone_data
from chronos2_modular.forecasting import load_model
from chronos2_modular.saturn import create_saturn_client, sync_saturn_data
from run_chronos2_hourly import _feature_inputs
from run_extended_residual_hourly import (
    _base_and_experts,
    _fit_inputs,
    _load_external,
    _load_features,
    _native_frame_from_source,
    _new_corrector,
    _read_timestamped,
)
from run_mkonline_blend_hourly import (
    EXPECTED_DEPENDENCY_SHA256,
    EXPECTED_RECIPE_SHA256,
    EXPECTED_SOURCE_CHECKSUM_MANIFEST_SHA256,
    PRIMARY_SERIES,
    QUANTILES,
    STORM_COMPARATOR_PATH,
    WEIGHT_AUTONOMOUS,
    WEIGHT_MK,
    _blend_quantiles,
    _load_primary,
    _load_recipe,
    _materialize,
    _validate_da_target_availability,
    _verify_source_run,
)


LOGGER = logging.getLogger("mkonline_live_hourly")
TIMEZONE = "Europe/Paris"
ATOMIC_PUBLISH_ATTEMPTS = 5
ATOMIC_PUBLISH_RETRY_SECONDS = 0.25
SCRIPT_VERSION = "1.0.0-dynamic-day-ahead-live"
DEFAULT_CONFIG = "chronos2_hourly_fr_mkonline_live_v1.yaml"
DEFAULT_BASE_CONFIG = "chronos2_hourly_fr_residual_v1.yaml"
LIVE_SYNC_LOOKBACK_HOURS = 48
MAX_AUTOMATIC_REPLAY_DAYS = 3
CHRONOS2_RESIDUAL_ARCHIVE_SUBDIR = Path(
    "_challengers/residual_load_chronos2"
)
EXPECTED_BENCHMARK_CHECKSUM_MANIFEST_SHA256 = (
    "e5af979b842ffcdB717be6e0d5eed190a12166c5d00456c4c6f1543c672f24dd".lower()
)
BENCHMARK_REPORT_FILES = (
    "backtest_hourly_oof.csv.gz",
    "metrics_hourly.csv",
    "metrics_hourly.json",
    "feature_manifest.csv",
    "ensemble_weights.csv",
    "pit_feature_coverage_by_hour.csv",
    "pit_feature_coverage_summary.csv",
    "evaluation_by_day.csv",
    "evaluation_by_hour.csv",
    "evaluation_by_month.csv",
    "evaluation_summary.json",
    "evaluation_vs_storm_by_day.csv",
    "evaluation_vs_storm_by_hour.csv",
    "evaluation_vs_storm_by_month.csv",
    "evaluation_vs_storm_summary.json",
    "run_manifest.json",
)


@dataclass(frozen=True)
class LiveSchedule:
    """Resolved operational date contract in civil Europe/Paris time."""

    as_of_local: pd.Timestamp
    delivery_day: date
    cutoff_local: pd.Timestamp
    delivery_index: pd.DatetimeIndex


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path, date)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _reporting_error(stage: str, exc: Exception) -> dict[str, str]:
    return {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _remove_optional_statistics_artifacts(staging: Path) -> None:
    """Remove only report-layer artifacts after a post-freeze failure."""

    for relative in (
        "statistics_history_hourly.csv.gz",
        "statistics_history_audit.json",
        "inputs/storm_evaluation_only_live_history.parquet",
        "inputs/storm_dashboard_official_statistics.parquet",
    ):
        (staging / relative).unlink(missing_ok=True)


def _write_degraded_html_report(
    path: Path,
    *,
    title: str,
    delivery_day: date,
    frozen_candidate_sha256: str,
    reporting_errors: Sequence[Mapping[str, Any]],
) -> None:
    """Write a self-contained notice without changing the frozen forecast."""

    items = "".join(
        "<li><strong>"
        + escape(str(item.get("stage", "reporting")))
        + "</strong> - "
        + escape(str(item.get("error_type", "Error")))
        + ": "
        + escape(str(item.get("error", "")))
        + "</li>"
        for item in reporting_errors
    )
    document = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:3rem auto;
padding:0 1rem;color:#172033}}.ok{{background:#eaf7ee;border-left:5px solid #18864b;
padding:1rem}}.warn{{background:#fff4e5;border-left:5px solid #d97706;
padding:1rem}}code{{word-break:break-all}}</style></head><body>
<h1>{escape(title)}</h1>
<div class="ok"><strong>Le forecast FR du {delivery_day.isoformat()} a bien ete
publie.</strong><br>Artefact : <code>forecast_hourly_fr.csv</code><br>
SHA-256 gele : <code>{escape(frozen_candidate_sha256)}</code></div>
<div class="warn"><h2>Rapport degrade</h2><p>Une etape posterieure au gel du
forecast a echoue. Cette erreur n'a pas modifie les previsions.</p><ul>{items}</ul>
<p>Les diagnostics complets sont disponibles dans
<code>reporting_errors.json</code>.</p></div></body></html>"""
    path.write_text(document, encoding="utf-8")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} doit etre un mapping.")
    return value


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _parse_as_of(value: str | pd.Timestamp | None) -> pd.Timestamp:
    timestamp = (
        pd.Timestamp.now(tz=TIMEZONE)
        if value in (None, "")
        else pd.Timestamp(value)
    )
    if timestamp.tzinfo is None:
        raise ValueError("--data-as-of doit inclure un fuseau horaire explicite.")
    return timestamp.tz_convert(TIMEZONE)


def _parse_delivery_day(value: str | date | pd.Timestamp | None) -> date | None:
    if value in (None, ""):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(TIMEZONE).tz_localize(None)
    if timestamp != timestamp.normalize():
        raise ValueError("--delivery-day doit etre une date civile YYYY-MM-DD.")
    return timestamp.date()


def _resolve_schedule(
    as_of: str | pd.Timestamp | None,
    delivery_day: str | date | pd.Timestamp | None,
) -> LiveSchedule:
    """Resolve tomorrow's French delivery day and enforce its 08:00 gate."""

    as_of_local = _parse_as_of(as_of)
    expected_delivery_day = (as_of_local.normalize() + pd.DateOffset(days=1)).date()
    requested_delivery_day = _parse_delivery_day(delivery_day)
    selected_day = requested_delivery_day or expected_delivery_day
    if selected_day != expected_delivery_day:
        raise ValueError(
            "Le mode live accepte uniquement J+1: "
            f"as-of={as_of_local}, attendu={expected_delivery_day}, "
            f"recu={selected_day}."
        )
    cutoff_naive = pd.Timestamp(selected_day) - pd.Timedelta(days=1) + pd.Timedelta(
        hours=8
    )
    cutoff_local = cutoff_naive.tz_localize(
        TIMEZONE,
        ambiguous="raise",
        nonexistent="raise",
    )
    if as_of_local < cutoff_local:
        raise ValueError(
            "Le run day-ahead est refuse avant le cutoff civil D-1 08:00 "
            f"Europe/Paris ({cutoff_local})."
        )
    delivery_index = local_delivery_day_index(
        selected_day,
        timezone=TIMEZONE,
    )
    return LiveSchedule(
        as_of_local=as_of_local,
        delivery_day=selected_day,
        cutoff_local=cutoff_local,
        delivery_index=delivery_index,
    )


def _archive_output_root(
    configured_root: Path,
    *,
    pit_replay: bool,
    residual_load_source: str,
) -> Path:
    """Keep production/replay roots stable and shadow archives quarantined."""

    if residual_load_source == "chronos2":
        root = (configured_root / CHRONOS2_RESIDUAL_ARCHIVE_SUBDIR).resolve()
        try:
            root.relative_to(configured_root.resolve())
        except ValueError as exc:
            raise ValueError(
                "La racine challenger Chronos-2 sort de output_root."
            ) from exc
        return root
    return configured_root / "_replays" if pit_replay else configured_root


def _validate_live_forecast(
    frame: pd.DataFrame,
    schedule: LiveSchedule,
) -> pd.DataFrame:
    required = {"delivery_start_utc", "forecast_origin_utc", *QUANTILES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Forecast live incomplet: {missing}.")
    result = frame.copy()
    delivery = pd.DatetimeIndex(
        pd.to_datetime(result["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if delivery.has_duplicates or not delivery.is_monotonic_increasing:
        raise ValueError("Forecast live duplique ou non trie.")
    if not delivery.equals(schedule.delivery_index):
        raise ValueError("Forecast live: timeline physique 23/24/25 non exacte.")
    origins = pd.DatetimeIndex(
        pd.to_datetime(result["forecast_origin_utc"], utc=True, errors="raise")
    )
    if not bool((origins < delivery).all()):
        raise ValueError("Forecast live: origine non causale.")
    latest_allowed_origin = schedule.cutoff_local.tz_convert("UTC")
    if not bool((origins <= latest_allowed_origin).all()):
        raise ValueError(
            "Forecast live: origine posterieure au cutoff civil D-1 08:00."
        )
    numeric = result.loc[:, list(QUANTILES)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Forecast live: quantile manquant ou non fini.")
    if not bool(((values[:, 0] <= values[:, 1]) & (values[:, 1] <= values[:, 2])).all()):
        raise ValueError("Forecast live: quantiles croises.")
    result.loc[:, list(QUANTILES)] = numeric
    result["delivery_start_utc"] = delivery
    result["forecast_origin_utc"] = origins
    return result


def _load_storm_dashboard_statistics_snapshot(
    *,
    benchmark_run: Path,
    current_delivery_day: date,
    data_config: Mapping[str, Any],
    zone: str = "FR",
    timezone: str = TIMEZONE,
) -> tuple[pd.Series, dict[str, Any]]:
    """Fetch the report-only Storm dashboard snapshot on Statistics."""

    metrics = json.loads(
        (benchmark_run / "metrics_hourly.json").read_text(encoding="utf-8")
    )
    diagnostics = _mapping(
        metrics.get("training_diagnostics", {}),
        name="training_diagnostics",
    )
    start_value = diagnostics.get("evaluation_start_local_date")
    if not start_value:
        raise ValueError("sealed benchmark evaluation start is missing")
    first_day = pd.Timestamp(start_value).date()
    last_day = local_delivery_day_index(
        current_delivery_day - pd.Timedelta(days=1),
        timezone=timezone,
    )
    first_day_index = local_delivery_day_index(first_day, timezone=timezone)
    expected = pd.date_range(start=first_day_index[0], end=last_day[-1], freq="h")
    client = create_saturn_client(
        str(data_config["saturn_url"]),
        str(data_config["saturn_author"]),
    )
    return fetch_native_dashboard_snapshot(
        client,
        zone=zone,
        expected_index=expected,
    )


def _checksum_entry(
    manifest: Mapping[str, Any],
    *,
    path: str,
    role: str,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in manifest.get("artifacts", [])
        if item.get("path") == path and item.get("role") == role
    ]
    if len(matches) != 1:
        raise ValueError(f"Entree checksum absente ou ambigue: {role}/{path}.")
    return matches[0]


def _verify_frozen_training_source(source_run: Path) -> dict[str, str]:
    """Verify every frozen source actually consumed by the live fit."""

    manifest_path = source_run / "artifact_checksums.json"
    if _sha256(manifest_path) != EXPECTED_SOURCE_CHECKSUM_MANIFEST_SHA256:
        raise ValueError("Manifest checksum du run autonome scelle invalide.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "backtest_hourly_oof.csv.gz": "run_artifact",
        "feature_manifest.csv": "run_artifact",
        "inputs/aligned_inputs.csv.gz": "materialized_input",
        "inputs/model_covariates_with_future.csv.gz": "materialized_input",
        "inputs/chronos_oof_extended.csv.gz": "materialized_input",
    }
    verified: dict[str, str] = {}
    for relative, role in required.items():
        item = _checksum_entry(manifest, path=relative, role=role)
        source = source_run / relative
        observed = _sha256(source)
        expected = str(item["sha256"]).lower()
        if observed != expected:
            raise ValueError(f"Artefact scelle modifie: {relative}.")
        verified[relative] = observed
    return verified


def _verify_benchmark(benchmark_run: Path) -> dict[str, str]:
    manifest_path = benchmark_run / "artifact_checksums.json"
    if _sha256(manifest_path) != EXPECTED_BENCHMARK_CHECKSUM_MANIFEST_SHA256:
        raise ValueError("Manifest checksum du benchmark MKOnline scelle invalide.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {name: "run_artifact" for name in BENCHMARK_REPORT_FILES}
    verified: dict[str, str] = {}
    for relative, role in required.items():
        item = _checksum_entry(manifest, path=relative, role=role)
        source = benchmark_run / relative
        observed = _sha256(source)
        if observed != str(item["sha256"]).lower():
            raise ValueError(f"Benchmark scelle modifie: {relative}.")
        verified[relative] = observed
    return verified


def _build_dynamic_data(
    *,
    base_config_path: Path,
    schedule: LiveSchedule,
    inputs_dir: Path,
    sync_manifest_path: Path,
    naive_timezone_overrides: Mapping[str, Any] | None = None,
    refresh_data: bool = True,
    residual_load_source: str = "saturn",
    residual_load_bundle_manifest: Path | None = None,
    saturn_control_archive: Path | None = None,
) -> tuple[dict[str, Any], Any, pd.DataFrame]:
    config = copy.deepcopy(load_yaml(base_config_path))
    data_config = config.setdefault("data", {})
    if not isinstance(data_config, dict):
        raise TypeError("data doit etre un mapping YAML.")
    # The forecast origin is the civil D-1 08:00 auction cutoff.  The actual
    # wall-clock launch may be later, but neither the latest target cache nor
    # any PIT vintage is allowed to observe that later time.
    input_cutoff = schedule.cutoff_local
    data_config["runtime_as_of"] = input_cutoff.isoformat()
    residual_load_provenance: dict[str, Any] | None = None
    if residual_load_source == "chronos2":
        if residual_load_bundle_manifest is None:
            raise ValueError(
                "Le manifeste du bundle de charge residuelle Chronos-2 est absent."
            )
        if saturn_control_archive is None:
            raise ValueError(
                "L'archive Saturn scellee pairee est obligatoire avec Chronos-2."
            )
        from chronos2_hourly.chronos_residual_load import (
            apply_residual_load_bundle,
        )

        residual_load_provenance = apply_residual_load_bundle(
            config,
            manifest_path=residual_load_bundle_manifest,
            expected_delivery_index=schedule.delivery_index,
            expected_cutoff_utc=schedule.cutoff_local.tz_convert("UTC"),
            config_dir=base_config_path.parent,
            overlay_dir=inputs_dir / "residual_load_pit_overlay",
            saturn_control_archive=saturn_control_archive,
            expected_zone="FR",
        )
    overrides = dict(naive_timezone_overrides or {})
    if overrides:
        zones_config = _mapping(config.get("zones"), name="zones")
        fr_config = _mapping(zones_config.get("FR"), name="zones.FR")
        covariates = _mapping(
            fr_config.get("covariates"),
            name="zones.FR.covariates",
        )
        for alias, timezone_hint in overrides.items():
            raw = covariates.get(str(alias))
            if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
                raise ValueError(
                    f"Override naive_timezone invalide pour {alias}: "
                    "covariable active introuvable."
                )
            raw["naive_timezone"] = str(timezone_hint)
    set_reproducibility(int(deep_get(config, "model.seed", 42)))
    zones = build_zone_configs(config, ["FR"], None, None)
    if len(zones) != 1 or zones[0].zone != "FR":
        raise ValueError("Le runner live attend exactement FR.")
    config_dir = base_config_path.parent
    # The persistent target/PIT stores already contain and retain the long
    # history used by Chronos and the sealed residual fit.  An incremental live
    # refresh only needs the current delivery context and J+1.  Passing the
    # four-year training start to Saturn is both wasteful and unsafe: a provider
    # may revise an unrelated, structurally incomplete legacy DST day and make
    # the whole J+1 refresh fail before that old value is ever used.  Keep this
    # D-2 bound on a sync-only copy; data preparation below still opens the
    # complete persistent caches and verifies the required Chronos context.
    sync_config = copy.deepcopy(config)
    sync_data_config = sync_config.setdefault("data", {})
    if not isinstance(sync_data_config, dict):
        raise TypeError("data doit etre un mapping YAML.")
    sync_value_start = schedule.delivery_index[0] - pd.Timedelta(
        hours=LIVE_SYNC_LOOKBACK_HOURS
    )
    sync_data_config["start"] = sync_value_start.isoformat()

    manifest = (
        sync_saturn_data(
            zones,
            sync_config,
            config_dir,
            as_of=input_cutoff,
        )
        if refresh_data
        else pd.DataFrame()
    )
    sync_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(sync_manifest_path, index=False)
    data = prepare_zone_data(
        zones[0],
        config,
        config_dir,
        False,
        inputs_dir,
    )
    if residual_load_provenance is not None:
        diagnostics = getattr(data, "diagnostics", None)
        if isinstance(diagnostics, dict):
            diagnostics["residual_load_provider"] = residual_load_provenance
        from chronos2_hourly.chronos_residual_load import (
            seal_challenger_zone_data_from_saturn_control,
        )

        control_provenance = seal_challenger_zone_data_from_saturn_control(
            data,
            saturn_archive_dir=saturn_control_archive,
            archive_inputs_dir=inputs_dir,
            expected_delivery_index=schedule.delivery_index,
            expected_zone="FR",
        )
        residual_load_provenance["effective_input_control"] = control_provenance
    target, _history_covariates, future_covariates, all_features = _feature_inputs(
        data,
        config,
    )
    target_context = audit_live_target_context(
        target,
        schedule.delivery_index,
    )
    diagnostics = getattr(data, "diagnostics", None)
    if isinstance(diagnostics, dict):
        target_diagnostics = diagnostics.setdefault("target", {})
        if isinstance(target_diagnostics, dict):
            target_diagnostics["live_context"] = target_context.as_dict()
    target_context.raise_if_not_exact(zone="FR")

    future_index = pd.DatetimeIndex(future_covariates.index).tz_convert("UTC")
    missing_delivery = schedule.delivery_index.difference(future_index)
    unexpected_future = future_index.difference(schedule.delivery_index)
    if len(missing_delivery) or len(unexpected_future):
        raise ValueError(
            "Les covariables PIT futures doivent couvrir uniquement le planning "
            "J+1 declare: "
            f"missing={len(missing_delivery)}, "
            f"unexpected={len(unexpected_future)}."
        )
    fresh_future = all_features.loc[schedule.delivery_index].copy()
    fresh_future.index = fresh_future.index.tz_convert("UTC")
    fresh_future.index.name = "delivery_start_utc"
    if not fresh_future.index.equals(schedule.delivery_index):
        raise ValueError("Les features PIT fraiches ne couvrent pas exactement J+1.")
    return config, data, fresh_future


def _train_and_predict_extended(
    *,
    frozen_source: Path,
    fresh_future: pd.DataFrame,
    chronos_live: pd.DataFrame,
    threads: int,
) -> tuple[pd.DataFrame, dict[str, Any], Any]:
    expected_features = pd.read_csv(frozen_source / "feature_manifest.csv")[
        "feature"
    ].astype(str).tolist()
    if list(fresh_future.columns) != expected_features:
        raise ValueError("Le schema live differe des features figees du modele.")
    external = _load_external(
        frozen_source / "inputs" / "chronos_oof_extended.csv.gz"
    )
    source_backtest = _read_timestamped(
        frozen_source / "backtest_hourly_oof.csv.gz",
        timestamp_column="delivery_start_utc",
    )
    if "fold_id" not in source_backtest:
        raise ValueError("Le backtest scelle ne contient pas fold_id.")
    existing = source_backtest.loc[source_backtest["fold_id"].notna()].copy()
    existing_native = _native_frame_from_source(existing)
    _target, history_features, _frozen_future = _load_features(frozen_source)
    X, y, base, experts = _fit_inputs(
        external,
        existing_native,
        history_features,
    )
    model = _new_corrector(threads=threads).fit(X, y, base, experts)
    live_base, live_experts = _base_and_experts(chronos_live)
    prediction = model.predict(
        fresh_future,
        live_base,
        live_experts,
    )
    prediction.index = schedule_index = fresh_future.index
    prediction.index.name = "delivery_start_utc"
    values = prediction.loc[:, list(QUANTILES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Le correcteur extended produit des valeurs invalides.")
    if len(model.feature_columns_) != 175:
        raise RuntimeError(
            f"Schema meta-features inattendu: {len(model.feature_columns_)} != 175."
        )
    training_end = existing_native.index[-1]
    availability = _validate_da_target_availability(
        training_end_utc=training_end,
        forecast_index=schedule_index,
    )
    return prediction, {
        "training_rows": int(len(X)),
        "training_start_utc": str(X.index[0]),
        "training_end_utc": str(training_end),
        "meta_features": int(len(model.feature_columns_)),
        "recipe": "blend_cat_hgb_w0.50",
        "target_availability": availability,
        "model": model.diagnostics(),
    }, model


def _validate_live_input_audit(
    *,
    data: Any,
    schedule: LiveSchedule,
) -> None:
    expected_runtime = schedule.cutoff_local.tz_convert("UTC")
    diagnostics = getattr(data, "diagnostics", {})
    covariates = diagnostics.get("covariates", {})
    required_aliases = (
        "fr_residual_load_fcst",
        "de_residual_load_fcst",
        "be_residual_load_fcst",
        "nl_residual_load_fcst",
        "es_residual_load_fcst",
    )
    for alias in required_aliases:
        audit = covariates.get(alias)
        if not isinstance(audit, Mapping):
            raise ValueError(f"Audit PIT live absent pour {alias}.")
        observed_runtime = pd.Timestamp(audit.get("runtime_as_of_utc"))
        if observed_runtime.tzinfo is None:
            observed_runtime = observed_runtime.tz_localize("UTC")
        else:
            observed_runtime = observed_runtime.tz_convert("UTC")
        if observed_runtime != expected_runtime:
            raise ValueError(
                f"{alias}: plafond PIT {observed_runtime} != {expected_runtime}."
            )
        if int(audit.get("cutoff_violations", -1)) != 0:
            raise ValueError(f"{alias}: violation PIT live detectee.")
        last_revision = pd.Timestamp(audit.get("last_selected_revision"))
        if last_revision.tzinfo is None:
            last_revision = last_revision.tz_localize("UTC")
        else:
            last_revision = last_revision.tz_convert("UTC")
        if last_revision > expected_runtime:
            raise ValueError(f"{alias}: revision posterieure au cutoff live.")


def _audit_future_pit_freshness(
    *,
    data: Any,
    schedule: LiveSchedule,
    max_revision_age_hours: float,
) -> dict[str, Any]:
    if max_revision_age_hours <= 0:
        raise ValueError("max_revision_age_hours doit etre strictement positif.")
    cutoff_utc = schedule.cutoff_local.tz_convert("UTC")
    diagnostics = getattr(data, "diagnostics", {})
    covariates = diagnostics.get("covariates", {})
    required_aliases = (
        "fr_residual_load_fcst",
        "de_residual_load_fcst",
        "be_residual_load_fcst",
        "nl_residual_load_fcst",
        "es_residual_load_fcst",
    )
    audits: dict[str, Any] = {}
    for alias in required_aliases:
        source_audit = covariates.get(alias)
        if not isinstance(source_audit, Mapping):
            raise ValueError(f"Audit PIT live absent pour {alias}.")
        path = Path(str(source_audit.get("input", ""))).expanduser().resolve()
        required = {
            "value_time_utc",
            "snapshot_time_utc",
            "revision_time_utc",
            "value",
        }
        try:
            frame = pd.read_parquet(
                path,
                columns=sorted(required),
                filters=[
                    (
                        "value_time_utc",
                        ">=",
                        schedule.delivery_index[0].to_pydatetime(),
                    ),
                    (
                        "value_time_utc",
                        "<=",
                        schedule.delivery_index[-1].to_pydatetime(),
                    ),
                ],
            )
        except (TypeError, ValueError):
            frame = pd.read_parquet(path, columns=sorted(required))
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{alias}: colonnes PIT absentes {missing}.")
        for column in (
            "value_time_utc",
            "snapshot_time_utc",
            "revision_time_utc",
        ):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
        frame = frame.loc[
            frame["value_time_utc"].between(
                schedule.delivery_index[0],
                schedule.delivery_index[-1],
                inclusive="both",
            )
            & frame["snapshot_time_utc"].le(cutoff_utc)
            & frame["revision_time_utc"].le(cutoff_utc)
        ].sort_values(
            ["value_time_utc", "snapshot_time_utc", "revision_time_utc"],
            kind="stable",
        )
        selected = frame.drop_duplicates("value_time_utc", keep="last")
        selected_index = pd.DatetimeIndex(
            selected["value_time_utc"],
            name="delivery_start_utc",
        )
        if not selected_index.equals(schedule.delivery_index):
            raise ValueError(
                f"{alias}: couverture PIT J+1 incomplete "
                f"({len(selected_index)}/{len(schedule.delivery_index)})."
            )
        values = pd.to_numeric(selected["value"], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{alias}: valeur PIT J+1 non finie.")
        revisions = pd.DatetimeIndex(selected["revision_time_utc"])
        ages = (cutoff_utc - revisions) / pd.Timedelta(hours=1)
        maximum_age = float(np.max(ages))
        if maximum_age > float(max_revision_age_hours):
            raise ValueError(
                f"{alias}: vintage J+1 trop ancien au cutoff "
                f"(age max={maximum_age:.2f} h > {max_revision_age_hours:.2f} h)."
            )
        audits[alias] = {
            "path": str(path),
            "sha256": _sha256(path),
            "hours": int(len(selected)),
            "coverage": 1.0,
            "cutoff_utc": str(cutoff_utc),
            "oldest_selected_revision_utc": str(revisions.min()),
            "newest_selected_revision_utc": str(revisions.max()),
            "maximum_revision_age_hours": maximum_age,
            "median_revision_age_hours": float(np.median(ages)),
            "maximum_allowed_revision_age_hours": float(max_revision_age_hours),
            "cutoff_violations": 0,
        }
    return audits


def _forecast_frame(
    *,
    schedule: LiveSchedule,
    chronos_live: pd.DataFrame,
    extended_live: pd.DataFrame,
    mkonline: pd.Series,
    cutoff: pd.Series,
) -> pd.DataFrame:
    blended = _blend_quantiles(extended_live, mkonline)
    metadata = build_delivery_metadata(schedule.delivery_index, timezone=TIMEZONE)
    output = metadata.reset_index(drop=True)
    for quantile in QUANTILES:
        output[quantile] = blended[quantile].to_numpy(dtype=float)
        output[f"chronos2__{quantile}"] = chronos_live[quantile].to_numpy(dtype=float)
        output[f"residual_corrected__{quantile}"] = extended_live[
            quantile
        ].to_numpy(dtype=float)
        output[f"mkonline_blend__{quantile}"] = blended[quantile].to_numpy(
            dtype=float
        )
    output["price_eur_mwh"] = output["q50"]
    output["residual_correction"] = (
        extended_live["q50"].to_numpy(dtype=float)
        - chronos_live["q50"].to_numpy(dtype=float)
    )
    output["mkonline_primary__q50"] = mkonline.to_numpy(dtype=float)
    output["mkonline_blend_shift"] = blended["shift"].to_numpy(dtype=float)
    chronos_origin = pd.DatetimeIndex(
        pd.to_datetime(
            chronos_live["forecast_origin_utc"],
            utc=True,
            errors="raise",
        )
    )
    mk_cutoff = pd.DatetimeIndex(pd.to_datetime(cutoff, utc=True, errors="raise"))
    candidate_origin = pd.DatetimeIndex(
        np.maximum(chronos_origin.asi8, mk_cutoff.asi8),
        tz="UTC",
    )
    # The unqualified origin belongs to the user-facing final q10/q50/q90 and
    # must therefore represent every input of the blend, not Chronos alone.
    output["forecast_origin_utc"] = candidate_origin.astype(str)
    output["chronos2_forecast_origin_utc"] = chronos_origin.astype(str)
    output["mkonline_blend_forecast_origin_utc"] = candidate_origin.astype(str)
    return _validate_live_forecast(output, schedule)


def _copy_benchmark_for_report(benchmark_run: Path, staging: Path) -> None:
    for filename in BENCHMARK_REPORT_FILES:
        source = benchmark_run / filename
        if source.is_file():
            shutil.copy2(source, staging / filename)


def _update_live_metrics(
    path: Path,
    *,
    schedule: LiveSchedule,
    fit_audit: Mapping[str, Any],
    run_type: str = "live_day_ahead",
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = payload.setdefault("training_diagnostics", {})
    diagnostics["live_run_type"] = run_type
    diagnostics["live_data_as_of_local"] = str(schedule.as_of_local)
    diagnostics["live_input_cutoff_local"] = str(schedule.cutoff_local)
    diagnostics["live_fit"] = dict(fit_audit)
    blend = diagnostics.get("mkonline_blend")
    if isinstance(blend, dict):
        blend["target_availability"] = fit_audit["target_availability"]
    payload["forecast_diagnostics"] = {
        "n_forecast_hours": int(len(schedule.delivery_index)),
        "forecast_start_utc": str(schedule.delivery_index[0]),
        "forecast_end_utc": str(schedule.delivery_index[-1]),
        "delivery_day_local": schedule.delivery_day.isoformat(),
        "forecast_origin_utc": str(schedule.cutoff_local.tz_convert("UTC")),
        "model": "mkonline_blend",
        "annual_metrics_source": "sealed_benchmark_not_recomputed",
        "storm_loaded_for_prediction": False,
        "storm_dashboard_loaded_after_candidate_frozen_for_statistics": True,
    }
    _write_json(path, payload)


def _write_checksums(
    staging: Path,
    *,
    output: Path,
    source_paths: Mapping[str, Path],
) -> None:
    entries: list[dict[str, Any]] = []
    for role, path in source_paths.items():
        entries.append(
            {
                "path": str(path),
                "role": role,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    project_root = Path(__file__).resolve().parent
    for path in (
        Path(__file__).resolve(),
        project_root / "run_extended_residual_hourly.py",
        project_root / "run_mkonline_blend_hourly.py",
        project_root / "materialize_saturn_daily_asof.py",
        project_root / "chronos2_hourly" / "reporting.py",
        project_root / "chronos2_hourly" / "live_history.py",
        project_root / "chronos2_hourly" / "storm_dashboard.py",
        project_root / "chronos2_modular" / "report.py",
    ):
        entries.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "role": "source_code",
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    checksum_path = staging / "artifact_checksums.json"
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path != checksum_path:
            entries.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "role": "run_artifact",
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        checksum_path,
        {
            "algorithm": "sha256",
            "output_directory": str(output),
            "artifacts": entries,
        },
    )


def _publish(
    staging: Path,
    output: Path,
    *,
    attempts: int = ATOMIC_PUBLISH_ATTEMPTS,
    retry_seconds: float = ATOMIC_PUBLISH_RETRY_SECONDS,
) -> None:
    """Publish one immutable archive, tolerating only transient Windows locks."""

    if output.exists():
        raise FileExistsError(f"Le run publie est immuable: {output}.")
    if attempts < 1:
        raise ValueError("attempts doit etre superieur ou egal a un.")
    for attempt in range(1, attempts + 1):
        try:
            staging.replace(output)
            return
        except PermissionError:
            # A concurrent publication is an immutable archive collision, not
            # a transient lock and must never become an overwrite attempt.
            if output.exists() or attempt == attempts:
                raise
            LOGGER.warning(
                "Publication atomique temporairement verrouillee (%s/%s): %s -> %s",
                attempt,
                attempts,
                staging,
                output,
            )
            time.sleep(retry_seconds * attempt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast day-ahead FR dynamique: Chronos EXT + MKOnline primaire."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-as-of", default=None)
    parser.add_argument("--delivery-day", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--residual-load-source",
        choices=("saturn", "chronos2"),
        default="saturn",
    )
    parser.add_argument("--residual-load-bundle-manifest", default=None)
    parser.add_argument(
        "--rolling365-capture-root",
        default=None,
        help=(
            "Active uniquement la capture prospective causale du challenger "
            "rolling-365 dans une racine separee. Le forecast officiel reste "
            "inchangé et est publie avant toute capture."
        ),
    )
    parser.add_argument(
        "--pit-replay",
        action="store_true",
        help=(
            "Reconstruit causalement une ancienne livraison au cutoff historique; "
            "le resultat est etiquete replay PIT et jamais live emis."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    project_root = Path(__file__).resolve().parent
    residual_load_source = str(args.residual_load_source).strip().lower()
    residual_load_bundle_manifest = (
        Path(args.residual_load_bundle_manifest).expanduser().resolve()
        if args.residual_load_bundle_manifest
        else None
    )
    if residual_load_source == "chronos2":
        if residual_load_bundle_manifest is None:
            raise ValueError(
                "--residual-load-bundle-manifest est obligatoire avec chronos2."
            )
        if not residual_load_bundle_manifest.is_file():
            raise FileNotFoundError(residual_load_bundle_manifest)
        try:
            residual_load_bundle_manifest.relative_to(project_root)
        except ValueError as exc:
            raise ValueError(
                "Le bundle Chronos-2 doit rester dans le projet."
            ) from exc
        if args.pit_replay:
            raise ValueError(
                "Un bundle live Chronos-2 ne peut pas servir a un replay PIT."
            )
        if args.rolling365_capture_root:
            raise ValueError(
                "La capture rolling-365 est reservee au run Saturn de production."
            )
    elif residual_load_bundle_manifest is not None:
        raise ValueError(
            "Un manifeste Chronos-2 ne peut pas etre fourni avec source=saturn."
        )
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    settings = _mapping(config.get("live"), name="live")
    config_dir = config_path.parent
    base_config_path = _resolve(
        settings.get("base_config", DEFAULT_BASE_CONFIG),
        base=config_dir,
    )
    frozen_source = _resolve(settings["frozen_autonomous_run"], base=config_dir)
    benchmark_run = _resolve(settings["sealed_benchmark_run"], base=config_dir)
    recipe_path = _resolve(settings["recipe_manifest"], base=config_dir)
    dependency_path = _resolve(settings["dependency_manifest"], base=config_dir)
    naive_timezone_overrides = _mapping(
        settings.get("naive_timezone_overrides", {}),
        name="live.naive_timezone_overrides",
    )
    schedule = _resolve_schedule(args.data_as_of, args.delivery_day)
    wall_clock_local = pd.Timestamp.now(tz=TIMEZONE)
    expected_live_day = (
        wall_clock_local.normalize() + pd.DateOffset(days=1)
    ).date()
    if args.pit_replay:
        if schedule.delivery_day > wall_clock_local.date():
            raise ValueError(
                "Un replay PIT ne peut pas viser une livraison future."
            )
        run_type = "pit_replay"
    else:
        if schedule.delivery_day != expected_live_day:
            raise ValueError(
                "Une date historique doit etre lancee avec --pit-replay; "
                f"le live courant attend {expected_live_day}."
            )
        run_type = (
            "shadow_live_day_ahead"
            if residual_load_source == "chronos2"
            else "live_day_ahead"
        )
    configured_root = _resolve(
        settings.get("output_root", "runs/live"),
        base=config_dir,
    )
    saturn_control_archive = (
        configured_root
        / f"fr_day_ahead_{schedule.delivery_day.isoformat()}"
        if residual_load_source == "chronos2"
        else None
    )
    default_output_root = _archive_output_root(
        configured_root,
        pit_replay=args.pit_replay,
        residual_load_source=residual_load_source,
    )
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_root
        / (
            f"fr_day_ahead_{schedule.delivery_day.isoformat()}"
            + (
                "_residual_load_chronos2"
                if residual_load_source == "chronos2"
                else ""
            )
        )
    )
    expected_parent = default_output_root.resolve()
    if residual_load_source == "chronos2":
        try:
            expected_parent.relative_to(configured_root.resolve())
        except ValueError as exc:
            raise ValueError(
                "La racine challenger Chronos-2 sort de output_root."
            ) from exc
    if output.parent.resolve() != expected_parent:
        raise ValueError(
            "--output-dir doit rester dans la racine correspondant au type "
            f"de run: {expected_parent}."
        )
    if residual_load_source == "chronos2":
        expected_shadow_name = (
            f"fr_day_ahead_{schedule.delivery_day.isoformat()}"
            "_residual_load_chronos2"
        )
        if output.name != expected_shadow_name:
            raise ValueError(
                "Le challenger Chronos-2 exige son nom d'archive isole canonique."
            )
    if output.exists():
        raise FileExistsError(
            "Une archive live/replay publiee est immuable; choisissez une "
            f"nouvelle date au lieu de la remplacer: {output}."
        )
    threads = int(args.threads if args.threads is not None else settings.get("threads", -1))
    workers = int(args.workers if args.workers is not None else settings.get("workers", 8))
    if output in {frozen_source, benchmark_run}:
        raise ValueError("Le dossier live doit etre distinct des runs scelles.")

    recipe = _load_recipe(recipe_path, dependency_path, project_root)
    _verify_source_run(frozen_source, recipe)
    frozen_hashes = _verify_frozen_training_source(frozen_source)
    benchmark_hashes = _verify_benchmark(benchmark_run)
    LOGGER.warning("%s", recipe["external_expert"]["commercial_entitlement_status"])

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    rolling_capture_result: Any | None = None
    try:
        inputs_dir = staging / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        dynamic_config, data, fresh_future = _build_dynamic_data(
            base_config_path=base_config_path,
            schedule=schedule,
            inputs_dir=inputs_dir,
            sync_manifest_path=staging / "saturn_sync_manifest.csv",
            naive_timezone_overrides=naive_timezone_overrides,
            residual_load_source=residual_load_source,
            residual_load_bundle_manifest=residual_load_bundle_manifest,
            saturn_control_archive=saturn_control_archive,
        )
        _validate_live_input_audit(data=data, schedule=schedule)
        pit_freshness = _audit_future_pit_freshness(
            data=data,
            schedule=schedule,
            max_revision_age_hours=float(
                settings.get("max_revision_age_hours", 24.0)
            ),
        )
        plan = build_delivery_plan(
            schedule.delivery_day,
            forecast_origin_local_time="08:00",
            timezone=TIMEZONE,
        )
        if not plan.delivery_index_utc.equals(schedule.delivery_index):
            raise RuntimeError("Le plan Chronos differe du planning live.")
        runtime = load_model(
            dynamic_config,
            args.device,
            bool(args.local_files_only),
        )
        context_length = int(
            deep_get(dynamic_config, "model.context_length", 2048)
        )
        model_batch_size = int(
            deep_get(dynamic_config, "model.model_batch_size", 128)
        )
        chronos_live = run_existing_live_forecast(
            plan,
            data=data,
            runtime=runtime,
            context_length=context_length,
            model_batch_size=model_batch_size,
            with_covariates=True,
            variant="hourly_live_dynamic",
        )
        chronos_live.reset_index().to_csv(
            staging / "chronos_live_hourly.csv",
            index=False,
        )
        fit_result = _train_and_predict_extended(
            frozen_source=frozen_source,
            fresh_future=fresh_future,
            chronos_live=chronos_live,
            threads=threads,
        )
        extended_live, fit_audit = fit_result[:2]
        corrector = fit_result[2] if len(fit_result) > 2 else None
        primary_path = inputs_dir / "mkonline_primary_live.parquet"
        if residual_load_source == "chronos2":
            from chronos2_hourly.chronos_residual_load import (
                copy_sealed_saturn_primary,
            )

            primary_control = copy_sealed_saturn_primary(
                saturn_control_archive,
                destination=primary_path,
                expected_delivery_day=schedule.delivery_day,
                expected_zone="FR",
            )
            command = [
                "sealed_saturn_control_copy",
                "inputs/mkonline_primary_live.parquet",
            ]
        else:
            primary_control = None
            command = _materialize(
                project_root=project_root,
                start_day=schedule.delivery_day.isoformat(),
                end_day=schedule.delivery_day.isoformat(),
                output=primary_path,
                workers=workers,
            )
        mkonline, cutoff, primary_audit = _load_primary(
            primary_path,
            expected_index=schedule.delivery_index,
            expected_histogram={len(schedule.delivery_index): 1},
        )
        # Staging is atomically renamed at publication.  Persist paths relative
        # to the run instead of leaking the ephemeral ``.tmp-*`` directory into
        # the audit manifest.
        primary_audit = dict(primary_audit)
        primary_audit["path"] = "inputs/mkonline_primary_live.parquet"
        if primary_control is not None:
            primary_audit["sealed_saturn_control"] = primary_control
        command_for_manifest = list(command)
        if "--output" in command_for_manifest:
            output_flag = command_for_manifest.index("--output")
            if output_flag + 1 >= len(command_for_manifest):
                raise ValueError("Commande MKOnline invalide: --output sans valeur.")
            command_for_manifest[output_flag + 1] = (
                "inputs/mkonline_primary_live.parquet"
            )
        forecast = _forecast_frame(
            schedule=schedule,
            chronos_live=chronos_live,
            extended_live=extended_live,
            mkonline=mkonline,
            cutoff=cutoff,
        )
        forecast_path = staging / "forecast_hourly_fr.csv"
        forecast.to_csv(forecast_path, index=False)
        frozen_candidate_sha256 = _sha256(forecast_path)
        attribution_status: dict[str, Any]
        if corrector is None:
            attribution_status = {
                "status": "not_available",
                "reason": "residual corrector was not exposed by the fit",
                "used_for_prediction": False,
            }
        else:
            try:
                attribution_audit = write_variable_attribution(
                    output_dir=staging,
                    forecast_path=forecast_path,
                    data=data,
                    runtime=runtime,
                    fresh_future=fresh_future,
                    corrector=corrector,
                    official_autonomous=extended_live,
                    required_covariates=tuple(
                        str(column) for column in data.covariates.columns
                    ),
                    context_length=context_length,
                    model_batch_size=model_batch_size,
                    zone="FR",
                    timezone=TIMEZONE,
                    delivery_day=schedule.delivery_day.isoformat(),
                    official_blend=pd.Series(
                        forecast["q50"].to_numpy(dtype=float),
                        index=schedule.delivery_index,
                    ),
                    primary=mkonline,
                    autonomous_weight=WEIGHT_AUTONOMOUS,
                    mkonline_weight=WEIGHT_MK,
                )
            except Exception as exc:
                remove_variable_attribution_artifacts(staging)
                attribution_status = {
                    "status": "failed_optional",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "used_for_prediction": False,
                    "forecast_modified": False,
                }
                LOGGER.warning(
                    "L'attribution des variables a echoue apres gel du forecast: %s",
                    exc,
                )
            else:
                attribution_status = {
                    "status": "complete",
                    "method": attribution_audit["method"],
                    "scenario_count": attribution_audit["scenario_count"],
                    "variants": attribution_audit["variants"],
                    "hourly_artifact": "variable_attribution_hourly.csv.gz",
                    "audit_artifact": "variable_attribution_audit.json",
                    "used_for_prediction": False,
                    "forecast_modified": False,
                }
        if _sha256(forecast_path) != frozen_candidate_sha256:
            raise RuntimeError(
                "L'attribution des variables a modifie le forecast fige."
            )
        archived_residual_bundle: dict[str, Any] | None = None
        if residual_load_bundle_manifest is not None:
            from chronos2_hourly.chronos_residual_load import (
                archive_live_residual_load_bundle,
            )

            archived_residual_bundle = archive_live_residual_load_bundle(
                residual_load_bundle_manifest,
                archive_inputs_dir=inputs_dir,
                expected_delivery_day=schedule.delivery_day,
                expected_runtime_cutoff=schedule.cutoff_local.tz_convert("UTC"),
            )

        if residual_load_source == "saturn":
            _copy_benchmark_for_report(benchmark_run, staging)
            _update_live_metrics(
                staging / "metrics_hourly.json",
                schedule=schedule,
                fit_audit=fit_audit,
                run_type=run_type,
            )
        source_manifest = json.loads(
            (benchmark_run / "run_manifest.json").read_text(encoding="utf-8")
        )
        target_series = source_manifest.get("target_series")
        dynamic_zones = dynamic_config.get("zones")
        if isinstance(dynamic_zones, Mapping):
            dynamic_fr = dynamic_zones.get("FR")
            if isinstance(dynamic_fr, Mapping):
                dynamic_target = dynamic_fr.get("target")
                if isinstance(dynamic_target, Mapping) and dynamic_target.get(
                    "series"
                ):
                    target_series = str(dynamic_target["series"])
        target_diagnostics = data.diagnostics.get("target", {})
        target_source_value = (
            target_diagnostics.get("cache")
            or target_diagnostics.get("input")
            if isinstance(target_diagnostics, Mapping)
            else None
        )
        target_source_file = (
            Path(str(target_source_value)).expanduser().resolve()
            if target_source_value not in (None, "")
            else None
        )
        target_source_path = (
            str(target_source_file)
            if target_source_file is not None and target_source_file.is_file()
            else None
        )
        target_source_sha256 = (
            _sha256(target_source_file)
            if target_source_file is not None and target_source_file.is_file()
            else None
        )
        run_manifest = dict(source_manifest)
        run_manifest.update(
            {
                "script_version": SCRIPT_VERSION,
                "config": str(config_path),
                "base_config": str(base_config_path),
                "run_type": run_type,
                "forecast_status": (
                    "pit_reconstruction"
                    if run_type == "pit_replay"
                    else (
                        "shadow_challenger"
                        if residual_load_source == "chronos2"
                        else "issued_live"
                    )
                ),
                "zone": "FR",
                "target_series": target_series,
                "candidate_model": "mkonline_blend",
                "prediction_mode": "mkonline_blend",
                "target_source_path": target_source_path,
                "target_source_sha256": target_source_sha256,
                "timezone": TIMEZONE,
                "delivery_day_local": schedule.delivery_day.isoformat(),
                "run_started_as_of_local": str(schedule.as_of_local),
                "run_started_as_of_utc": str(schedule.as_of_local.tz_convert("UTC")),
                "execution_started_at_local": str(wall_clock_local),
                "execution_started_at_utc": str(
                    wall_clock_local.tz_convert("UTC")
                ),
                "issued_at_utc": (
                    None
                    if run_type == "pit_replay"
                    else str(pd.Timestamp.now(tz="UTC"))
                ),
                "replayed_at_utc": (
                    str(pd.Timestamp.now(tz="UTC"))
                    if run_type == "pit_replay"
                    else None
                ),
                "data_as_of_local": str(schedule.cutoff_local),
                "data_as_of_utc": str(schedule.cutoff_local.tz_convert("UTC")),
                "forecast_cutoff_local": str(schedule.cutoff_local),
                "forecast_cutoff_utc": str(schedule.cutoff_local.tz_convert("UTC")),
                "saturn_sync_value_start_utc": str(
                    schedule.delivery_index[0]
                    - pd.Timedelta(hours=LIVE_SYNC_LOOKBACK_HOURS)
                ),
                "n_forecast_hours": int(len(schedule.delivery_index)),
                "n_training_hours": int(len(data.target)),
                "forecast_start_utc": str(schedule.delivery_index[0]),
                "forecast_end_utc": str(schedule.delivery_index[-1]),
                "prediction_inputs": ["autonomous_extended_residual", PRIMARY_SERIES],
                "storm_loaded_for_prediction": False,
                "storm_dashboard_loaded_after_candidate_frozen_for_statistics": True,
                "storm_used_as_feature": False,
                "annual_benchmark_recomputed": False,
                "sealed_benchmark_evaluation": {
                    "evaluation_start_local_date": source_manifest.get(
                        "evaluation_start_local_date"
                    ),
                    "evaluation_end_local_date": source_manifest.get(
                        "evaluation_end_local_date"
                    ),
                    "n_evaluation_hours": source_manifest.get(
                        "n_evaluation_hours"
                    ),
                    "n_evaluation_days": source_manifest.get(
                        "n_evaluation_days"
                    ),
                },
                "frozen_autonomous_manifest_sha256": EXPECTED_SOURCE_CHECKSUM_MANIFEST_SHA256,
                "sealed_benchmark_manifest_sha256": EXPECTED_BENCHMARK_CHECKSUM_MANIFEST_SHA256,
                "mkonline_weight": WEIGHT_MK,
                "autonomous_weight": WEIGHT_AUTONOMOUS,
                "live_fit": fit_audit,
                "target_availability": fit_audit["target_availability"],
                "input_diagnostics": data.diagnostics,
                "naive_timezone_overrides": dict(naive_timezone_overrides),
                "mkonline_pit": primary_audit,
                "fundamental_pit_freshness": pit_freshness,
                "variable_attribution": attribution_status,
                "mkonline_materialization_command": command_for_manifest,
                "commercial_entitlement_status": recipe["external_expert"][
                    "commercial_entitlement_status"
                ],
                "sha256_manifest": "artifact_checksums.json",
            }
        )
        if residual_load_source == "chronos2":
            if archived_residual_bundle is None:  # pragma: no cover - guarded above
                raise RuntimeError(
                    "Le bundle Chronos-2 archive est absent."
                )
            run_manifest.update(
                {
                    "residual_load_source": "chronos2",
                    "residual_load_bundle_manifest_path": (
                        "inputs/"
                        + str(archived_residual_bundle["archived_manifest_path"])
                    ),
                    "residual_load_bundle_origin_manifest_path": str(
                        archived_residual_bundle["origin_manifest_path"]
                    ),
                    "residual_load_bundle_manifest_sha256": str(
                        archived_residual_bundle["archived_manifest_sha256"]
                    ),
                    "production_eligible": False,
                    "comparison_design": "prospective_paired_same_downstream",
                }
            )
        _write_json(staging / "run_manifest.json", run_manifest)
        live_summary_payload: dict[str, Any] = {
            "status": "complete",
            "run_type": run_type,
            "forecast_status": (
                "pit_reconstruction"
                if run_type == "pit_replay"
                else (
                    "shadow_challenger"
                    if residual_load_source == "chronos2"
                    else "issued_live"
                )
            ),
            "delivery_day_local": schedule.delivery_day,
            "hours": len(schedule.delivery_index),
            "as_of_local": schedule.as_of_local,
            "cutoff_local": schedule.cutoff_local,
            "models": {
                "chronos2": "amazon/chronos-2",
                "autonomous": "EXT223 residual blend CatBoost/HistGBR",
                "external_primary": PRIMARY_SERIES,
                "final": "frozen autonomous + MKOnline L1 blend",
            },
            "weights": {
                "autonomous": WEIGHT_AUTONOMOUS,
                "mkonline_primary": WEIGHT_MK,
            },
            "storm_loaded_for_prediction": False,
            "storm_dashboard_loaded_for_statistics_after_candidate_frozen": True,
            "fundamental_pit_freshness": pit_freshness,
            "variable_attribution": attribution_status,
            "forecast_path": "forecast_hourly_fr.csv",
        }
        if residual_load_source == "chronos2":
            live_summary_payload.update(
                {
                    "zone": "FR",
                    "residual_load_source": "chronos2",
                    "production_eligible": False,
                }
            )
        _write_json(
            staging / "live_run_summary.json",
            live_summary_payload,
        )
        shutil.copy2(recipe_path, staging / "mkonline_blend_recipe.json")
        shutil.copy2(dependency_path, staging / "mkonline_primary_dependency.json")

        checksum_source_paths = {
            "live_config": config_path,
            "base_config": base_config_path,
            "frozen_recipe": recipe_path,
            "dependency_manifest": dependency_path,
            "frozen_autonomous_checksum_manifest": frozen_source
            / "artifact_checksums.json",
            "sealed_benchmark_checksum_manifest": benchmark_run
            / "artifact_checksums.json",
            **(
                {
                    "residual_load_bundle_manifest": (
                        residual_load_bundle_manifest
                    )
                }
                if residual_load_bundle_manifest is not None
                else {}
            ),
        }
        if residual_load_source == "chronos2":
            prospective_statistics = {
                "status": "prospective_only",
                "historical_performance_eligible": False,
                "residual_load_source": "chronos2",
                "comparison_reason": (
                    "Aucun historique causal Chronos-2 n'est substitue aux "
                    "archives Saturn de production. Le scoring se fait "
                    "uniquement sur les paires prospectives publiees."
                ),
            }
            run_manifest.update(
                {
                    "statistics_history": prospective_statistics,
                    "candidate_forecast_sha256_at_freeze": (
                        frozen_candidate_sha256
                    ),
                    "storm_dashboard_loaded_after_candidate_frozen_for_statistics": (
                        False
                    ),
                    "reporting_status": "forecast_only",
                    "reporting_errors": [],
                }
            )
            live_summary_payload.update(
                {
                    "statistics_history": prospective_statistics,
                    "storm_dashboard_loaded_for_statistics_after_candidate_frozen": (
                        False
                    ),
                    "reporting_status": "forecast_only",
                }
            )
            report = _mapping(config.get("report", {}), name="report")
            report_name = str(
                report.get(
                    "filename",
                    f"fr_day_ahead_{schedule.delivery_day}.html",
                )
            ).format(delivery_day=schedule.delivery_day.isoformat())
            write_forecast_only_shadow_report(
                forecast_path,
                output_path=staging / report_name,
                zone="FR",
                delivery_day=schedule.delivery_day.isoformat(),
                candidate_model="mkonline_blend",
            )
            if _sha256(forecast_path) != frozen_candidate_sha256:
                raise ValueError(
                    "Forecast-only reporting modified the frozen candidate."
                )
            _write_json(staging / "run_manifest.json", run_manifest)
            _write_json(
                staging / "live_run_summary.json",
                live_summary_payload,
            )
            _write_checksums(
                staging,
                output=output,
                source_paths=checksum_source_paths,
            )
            _publish(staging, output)
            print(f"Run live : {output}")
            print(
                "Livraison : "
                f"{schedule.delivery_day} ({len(schedule.delivery_index)} heures)"
            )
            return 0

        # Everything below is report-only.  The candidate is already frozen;
        # a Statistics, Storm or HTML failure must never erase or alter it.
        reporting_errors: list[dict[str, str]] = []
        data_config = _mapping(
            dynamic_config.get("data", {}), name="data"
        )
        try:
            dashboard_raw, dashboard_source = (
                _load_storm_dashboard_statistics_snapshot(
                    benchmark_run=benchmark_run,
                    current_delivery_day=schedule.delivery_day,
                    data_config=data_config,
                    zone="FR",
                    timezone=TIMEZONE,
                )
            )
        except Exception as exc:
            dashboard_raw = None
            dashboard_source = {
                "status": "reporting_error",
                "used_for_prediction": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            reporting_errors.append(_reporting_error("storm_dashboard", exc))
            LOGGER.warning(
                "Storm reporting failed after candidate freeze: %s", exc
            )
        if _sha256(forecast_path) != frozen_candidate_sha256:
            raise ValueError("Storm reporting modified the frozen candidate.")

        statistics_live_root = (
            configured_root
            if residual_load_source == "saturn"
            else configured_root / "_residual_load_chronos2_statistics"
        )
        history_identity = {
            "sealed_benchmark_run": benchmark_run,
            "live_output_root": statistics_live_root,
            "replay_output_root": statistics_live_root / "_replays",
            "current_delivery_day": schedule.delivery_day,
            "timezone": TIMEZONE,
            "forecast_name": "forecast_hourly_fr.csv",
            "candidate_model": "mkonline_blend",
            "zone": "FR",
            "target_series": str(target_series),
            "prediction_mode": "mkonline_blend",
        }
        statistics_kwargs = {
            "staging_run_dir": staging,
            **history_identity,
            "canonical_target": data.target,
            "storm_pit_path": (project_root / STORM_COMPARATOR_PATH).resolve(),
            "storm_dashboard_native": dashboard_raw,
            "storm_dashboard_source": dashboard_source,
        }
        statistics_diagnostic: dict[str, Any] | None = None
        missing_days: list[date] = []
        try:
            statistics_history = update_live_statistics_history(
                **statistics_kwargs
            )
        except Exception as strict_statistics_error:
            try:
                missing_days = missing_statistics_archive_days(
                    **history_identity
                )
            except Exception as planner_error:
                reporting_errors.append(
                    _reporting_error("statistics_planner", planner_error)
                )
                _remove_optional_statistics_artifacts(staging)
                statistics_history = {
                    "status": "blocked_reporting_error",
                    "reason": (
                        "Statistics archive planning failed after candidate "
                        "freeze"
                    ),
                    "error_type": type(planner_error).__name__,
                    "error": str(planner_error),
                    "initial_statistics_error_type": type(
                        strict_statistics_error
                    ).__name__,
                    "initial_statistics_error": str(strict_statistics_error),
                    "candidate_forecast_publication": "continued",
                    "statistics_scope": "sealed_benchmark_only",
                    "storm_used_for_prediction": False,
                }
                statistics_diagnostic = dict(statistics_history)
            else:
                if missing_days:
                    statistics_blocker = {
                        "status": "blocked_missing_causal_archives",
                        "reason": (
                            "automatic replay safety limit exceeded"
                            if len(missing_days) > MAX_AUTOMATIC_REPLAY_DAYS
                            else "automatic causal PIT replay is not enabled "
                            "for the sealed FR runner"
                        ),
                        "missing_realized_days": [
                            day.isoformat() for day in missing_days
                        ],
                        "completed_automatic_replay_days": [],
                        "requested_automatic_replay_days": len(missing_days),
                        "max_automatic_replay_days": (
                            MAX_AUTOMATIC_REPLAY_DAYS
                        ),
                        "candidate_forecast_publication": "continued",
                        "statistics_scope": "contiguous_prefix",
                        "storm_used_for_prediction": False,
                    }
                    try:
                        statistics_history = update_live_statistics_history(
                            **statistics_kwargs,
                            allow_partial_prefix=True,
                            statistics_blocker=statistics_blocker,
                        )
                    except Exception as partial_statistics_error:
                        reporting_errors.append(
                            _reporting_error(
                                "statistics", partial_statistics_error
                            )
                        )
                        _remove_optional_statistics_artifacts(staging)
                        statistics_history = {
                            **statistics_blocker,
                            "status": "blocked_reporting_error",
                            "reason": (
                                "Partial Statistics failed after candidate "
                                "freeze"
                            ),
                            "error_type": type(
                                partial_statistics_error
                            ).__name__,
                            "error": str(partial_statistics_error),
                        }
                    else:
                        statistics_history = dict(statistics_history)
                        statistics_history.setdefault(
                            "status", "partial_contiguous_prefix"
                        )
                        # Keep the operational publication contract visible at
                        # the audit root as well as under statistics_blocker.
                        for key, value in statistics_blocker.items():
                            statistics_history.setdefault(key, value)
                        LOGGER.warning(
                            "Statistics partielles: prefixe causal conserve; "
                            "archives manquantes=%s",
                            ", ".join(
                                day.isoformat() for day in missing_days
                            ),
                        )
                    statistics_diagnostic = dict(statistics_history)
                else:
                    reporting_errors.append(
                        _reporting_error(
                            "statistics", strict_statistics_error
                        )
                    )
                    _remove_optional_statistics_artifacts(staging)
                    statistics_history = {
                        "status": "blocked_reporting_error",
                        "reason": "Statistics failed after candidate freeze",
                        "error_type": type(
                            strict_statistics_error
                        ).__name__,
                        "error": str(strict_statistics_error),
                        "candidate_forecast_publication": "continued",
                        "statistics_scope": "sealed_benchmark_only",
                        "storm_used_for_prediction": False,
                    }
                    statistics_diagnostic = dict(statistics_history)
        else:
            statistics_history = dict(statistics_history)
        if residual_load_source == "chronos2":
            statistics_history = dict(statistics_history)
            statistics_history.update(
                {
                    "comparison_status": "prospective_only",
                    "historical_performance_eligible": False,
                    "residual_load_source": "chronos2",
                    "comparison_reason": (
                        "Aucun historique causal Chronos-2 n'est substitue "
                        "aux archives Saturn de production. Le scoring se fait "
                        "uniquement sur les paires prospectives publiees."
                    ),
                }
            )
        if statistics_diagnostic is not None:
            statistics_history["diagnostic_path"] = (
                "statistics_update_blocked.json"
            )
            statistics_diagnostic = dict(statistics_history)
            _write_json(
                staging / "statistics_update_blocked.json",
                statistics_diagnostic,
            )

        run_manifest["statistics_history"] = statistics_history
        run_manifest["candidate_forecast_sha256_at_freeze"] = (
            frozen_candidate_sha256
        )
        run_manifest[
            "storm_dashboard_loaded_after_candidate_frozen_for_statistics"
        ] = dashboard_raw is not None
        _write_json(staging / "run_manifest.json", run_manifest)
        summary_path = staging / "live_run_summary.json"
        live_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        live_summary["statistics_history"] = statistics_history
        live_summary[
            "storm_dashboard_loaded_for_statistics_after_candidate_frozen"
        ] = dashboard_raw is not None
        if statistics_history.get("status") == "partial_contiguous_prefix":
            live_summary["status"] = "forecast_complete_statistics_partial"
        elif str(statistics_history.get("status", "")).startswith("blocked"):
            live_summary["status"] = "forecast_complete_statistics_blocked"
        _write_json(summary_path, live_summary)
        if _sha256(forecast_path) != frozen_candidate_sha256:
            raise ValueError("Statistics modified the frozen candidate.")

        report = _mapping(config.get("report", {}), name="report")
        report_name = str(
            report.get("filename", f"fr_day_ahead_{schedule.delivery_day}.html")
        ).format(delivery_day=schedule.delivery_day.isoformat())
        report_path = staging / report_name
        report_title = str(
            report.get(
                "title",
                "Forecast day-ahead FR {delivery_day}",
            )
        ).format(delivery_day=schedule.delivery_day.isoformat())
        try:
            write_hourly_html_report(
                staging,
                output_path=report_path,
                title=report_title,
                native_model="mkonline_blend",
                baseline_model="residual_corrected",
                zone="FR",
                timezone=TIMEZONE,
                extreme_threshold=float(report.get("extreme_threshold", 150.0)),
                history_hours=int(report.get("forecast_history_hours", 168)),
            )
            if not report_path.is_file():
                raise ValueError("Le rapport HTML detaille n'a pas ete cree.")
        except Exception as exc:
            reporting_errors.append(_reporting_error("html_report", exc))
            report_path.unlink(missing_ok=True)
            _write_degraded_html_report(
                report_path,
                title=report_title,
                delivery_day=schedule.delivery_day,
                frozen_candidate_sha256=frozen_candidate_sha256,
                reporting_errors=reporting_errors,
            )
            LOGGER.warning(
                "HTML reporting failed after candidate freeze; wrote a "
                "degraded report: %s",
                exc,
            )
        if _sha256(forecast_path) != frozen_candidate_sha256:
            raise ValueError("HTML reporting modified the frozen candidate.")

        if reporting_errors:
            _write_json(
                staging / "reporting_errors.json",
                {
                    "status": "degraded",
                    "candidate_forecast_sha256": frozen_candidate_sha256,
                    "forecast_modified": False,
                    "errors": reporting_errors,
                },
            )
            run_manifest["reporting_status"] = "degraded"
            run_manifest["reporting_errors"] = reporting_errors
            if live_summary.get("status") == "complete":
                live_summary["status"] = (
                    "forecast_complete_reporting_degraded"
                )
            live_summary["reporting_errors"] = reporting_errors
        else:
            run_manifest["reporting_status"] = "complete"
            run_manifest["reporting_errors"] = []
        _write_json(staging / "run_manifest.json", run_manifest)
        _write_json(summary_path, live_summary)
        _write_checksums(
            staging,
            output=output,
            source_paths=checksum_source_paths,
        )
        _publish(staging, output)
        if (
            args.rolling365_capture_root
            and not args.pit_replay
            and residual_load_source == "saturn"
        ):
            # Instrumentation only: the official archive has already been
            # atomically published.  This call receives raw Chronos and
            # autonomous PIT features, never Storm, MKOnline or the final
            # production curve.  Ordinary capture failures are non-fatal.
            try:
                from chronos2_hourly.rolling_capture import (
                    capture_supported_issued_live_block_isolated,
                )

                rolling_capture_result = (
                    capture_supported_issued_live_block_isolated(
                        capture_root=args.rolling365_capture_root,
                        zone="FR",
                        delivery_day=schedule.delivery_day,
                        delivery_timezone=TIMEZONE,
                        fresh_features=fresh_future,
                        chronos_live=chronos_live,
                        required_pit_aliases=(
                            "fr_residual_load_fcst",
                            "de_residual_load_fcst",
                            "be_residual_load_fcst",
                            "nl_residual_load_fcst",
                            "es_residual_load_fcst",
                        ),
                        pit_freshness=pit_freshness,
                        expected_config_sha256=_sha256(config_path),
                        expected_base_bundle_sha256=(
                            EXPECTED_SOURCE_CHECKSUM_MANIFEST_SHA256
                        ),
                        target_series=str(run_manifest["target_series"]),
                        target_source_path=target_source_path,
                        issued_live_archive=output,
                        issued_live_forecast_filename="forecast_hourly_fr.csv",
                        canonical_target=data.target,
                    )
                )
                LOGGER.info(
                    "Capture rolling-365 FR: %s",
                    rolling_capture_result.status,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Capture rolling-365 FR ignoree apres publication: %s",
                    exc,
                )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Run live : {output}")
    print(f"Livraison : {schedule.delivery_day} ({len(schedule.delivery_index)} heures)")
    print(f"Forecast CSV : {output / 'forecast_hourly_fr.csv'}")
    print(f"Rapport HTML : {output / report_name}")
    if rolling_capture_result is not None:
        print(f"Capture rolling-365 : {rolling_capture_result.status}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Execution interrompue.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("Echec du run day-ahead live: %s", exc)
        raise SystemExit(1)
