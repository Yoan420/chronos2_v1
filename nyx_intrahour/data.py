"""Read verified NYX archives and sealed native-quarter-hour forecast sources.

No market collection, imputation, operational mutation or model loading occurs
here. The input source contract records retrospective versus prospective timing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ZONES = ("BE", "DE", "FR", "NL")
HOURLY_ALIASES = (
    "fr_residual_load_fcst", "de_residual_load_fcst", "be_residual_load_fcst",
    "nl_residual_load_fcst", "es_residual_load_fcst", "fr_nuclear_generation_fcst_gw",
)


class DataUnavailable(ValueError):
    """A required real input is absent; no fabricated experiment can replace it."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_day(root: Path, value: str = "latest") -> str:
    directory = root / "runs/experiments/nuclear_forecast_v1"
    candidates = ([value] if value != "latest" else
                  sorted((p.name for p in directory.iterdir()
                          if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)), reverse=True)
                  if directory.exists() else [])
    for day in candidates:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("Delivery day must be ISO YYYY-MM-DD or latest.")
        if all((directory/day/z.lower()/"civil_pit_v2/report_only/frozen_result/manifest.json").is_file()
               for z in ZONES):
            return day
    raise DataUnavailable("Aucun ensemble NYX figé complet BE/DE/FR/NL n'est disponible.")


def load_baseline(root: Path, delivery_day: str = "latest") -> tuple[pd.DataFrame, dict]:
    from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
    from chronos2_hourly.model_storm_data import _latest_source, _series
    from chronos2_hourly.nuclear_reporting_refresh import verify_refreshed_observations

    day = baseline_day(root, delivery_day)
    frames, identities = [], []
    for zone in ZONES:
        work = root/"runs/experiments/nuclear_forecast_v1"/day/zone.lower()/"civil_pit_v2"
        bundle = work/"report_only/frozen_result"
        before = {str(p): digest(p) for p in sorted(bundle.iterdir()) if p.is_file()}
        result = load_nuclear_result_bundle(workdir=work)
        if result.audit["zone"] != zone or result.audit["delivery_day"] != day:
            raise ValueError("Baseline country/delivery identity mismatch.")
        replay = result.kalman_view.replay.audit
        if (replay.get("causality_violations") != 0
                or replay.get("target_actuals_assimilated_before_forecast") != 0):
            raise ValueError("Baseline causal replay checks did not pass.")
        backtest = result.kalman_view.backtest
        index = pd.DatetimeIndex(pd.to_datetime(backtest.delivery_start_utc, utc=True))
        if index.has_duplicates or not index.equals(index.floor("h")):
            raise ValueError("Unique physical hourly NYX targets required.")
        frozen = backtest.actual.to_numpy(float)
        frame = pd.DataFrame({"timestamp_utc": index, "zone": zone,
                              "actual": frozen, "training_actual": frozen,
                              "nyx_q50": backtest["residual_kalman__q50"].to_numpy(float)})
        label_source = "frozen_actual"
        source_identity = {}
        # Refreshes are already archived files, never a new request to a provider.
        selected = _latest_source(work)
        if selected is not None:
            _, audit_path, source_audit = selected
            audit_bytes = audit_path.read_bytes()
            audit_sha = hashlib.sha256(audit_bytes).hexdigest()
            source_audit = json.loads(audit_bytes)
            if (not isinstance(source_audit, dict) or source_audit.get("status") != "complete"
                    or source_audit.get("zone") != zone or source_audit.get("delivery_day_local") != day
                    or source_audit.get("used_for_prediction") is not False):
                raise ValueError("Observed source audit does not match the baseline identity.")
            observed_path = audit_path.parent/"inputs/observed_latest.parquet"
            observed_sha = digest(observed_path)
            if observed_sha != source_audit.get("observed", {}).get("artifact_sha256"):
                raise ValueError("Observed source checksum mismatch.")
            observed = _series(pd.read_parquet(observed_path), "actual")
            tz = {"BE": "Europe/Brussels", "DE": "Europe/Berlin", "FR": "Europe/Paris", "NL": "Europe/Amsterdam"}[zone]
            verify_refreshed_observations(observed, source_audit, zone=zone, timezone=tz, delivery_day=day)
            values = observed.reindex(index).to_numpy(float)
            if not np.isfinite(values).all():
                raise DataUnavailable(f"{zone}: observations vérifiées incomplètes sur le backtest.")
            frame["actual"] = values
            label_source = "verified_latest_observed"
            if digest(audit_path) != audit_sha or digest(observed_path) != observed_sha:
                raise ValueError("Observed source changed while reading.")
            source_identity = {str(audit_path): audit_sha, str(observed_path): observed_sha}
        cov = result.covariates.copy()
        cov.index = pd.DatetimeIndex(pd.to_datetime(cov.timestamp, utc=True))
        if cov.index.has_duplicates or not set(HOURLY_ALIASES).issubset(cov):
            raise ValueError("Baseline fundamental schema/timeline mismatch.")
        for alias in HOURLY_ALIASES:
            frame[f"feature_hourly_{alias}"] = cov[alias].reindex(index).to_numpy(float)
        required = ["actual", "training_actual", "nyx_q50"]+[f"feature_hourly_{a}" for a in HOURLY_ALIASES]
        if not np.isfinite(frame[required].to_numpy()).all():
            raise DataUnavailable(f"{zone}: baseline ou fondamentaux horaires incomplets.")
        if before != {p: digest(Path(p)) for p in before}:
            raise ValueError("Baseline files changed while reading.")
        identities.append({"zone": zone, "delivery_day": day, "files": before,
                           "observations": source_identity, "evaluation_label_source": label_source})
        frames.append(frame)
    common = frames[0].timestamp_utc
    if any(not f.timestamp_utc.equals(common) for f in frames[1:]):
        raise ValueError("The four baseline backtests must have exactly the same physical hours.")
    panel = pd.concat(frames, ignore_index=True).sort_values(["timestamp_utc", "zone"]).reset_index(drop=True)
    return panel, {"delivery_day": day, "identities": identities,
                   "baseline": "nuclear_forecast_v1/civil_pit_v2/residual_kalman__q50",
                   "production_pit_evidence": False, "label_vintage_warning":
                   "Frozen retrospective labels are not proof of availability at historical origins."}


def read_native_sources(manifest_path: Path) -> tuple[pd.DataFrame, list[dict], dict]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise DataUnavailable("Archive de prévisions natives à 15 minutes absente : "+str(manifest_path))
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema_version") != 1
            or manifest.get("artifact_type") != "nyx_intrahour_forecast_vintages"):
        raise ValueError("Native-quarter-hour forecast manifest schema 1 required.")
    filename = manifest.get("data_file")
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".parquet"):
        raise ValueError("Native source data_file must name a sibling parquet, without traversal.")
    path = (manifest_path.parent/filename).resolve()
    if path.parent != manifest_path.parent or not path.is_file():
        raise ValueError("Native source parquet must remain beside its manifest.")
    if digest(path) != manifest.get("data_sha256"):
        raise ValueError("Native source parquet checksum mismatch.")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DataUnavailable("Aucune série fondamentale native à 15 minutes n'a été validée.")
    if manifest.get("temporal_evidence") not in {"retrospective_asof", "archived_at_issue"}:
        raise ValueError("Declare retrospective_asof or archived_at_issue temporal evidence.")
    if type(manifest.get("provider_revision_timestamp_available")) is not bool:
        raise ValueError("Provider revision timestamp availability must be explicit.")
    if manifest["temporal_evidence"] == "archived_at_issue" and not manifest.get("receipt_evidence"):
        raise ValueError("Archived-at-issue inputs require an explicit receipt evidence reference.")
    frame = pd.read_parquet(path)
    if digest(path) != manifest["data_sha256"] or digest(manifest_path) != manifest_sha:
        raise ValueError("Native source manifest or parquet changed while reading.")
    return frame, sources, {**manifest, "manifest_path": str(manifest_path),
                            "manifest_sha256": manifest_sha, "data_path": str(path),
                            "production_pit_evidence": False}


def join_features(baseline: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    if (not {"timestamp_utc", "zone"}.issubset(baseline)
            or baseline[["timestamp_utc", "zone"]].isna().any().any()
            or baseline.duplicated(["timestamp_utc", "zone"]).any()):
        raise ValueError("Baseline requires unique physical country/hour keys.")
    if hourly.index.name != "timestamp_utc" or hourly.index.has_duplicates:
        raise ValueError("Hourly features require a unique timestamp_utc index.")
    collision = set(baseline.columns).intersection(hourly.columns)
    if collision:
        raise ValueError(f"Features cannot overwrite baseline columns: {sorted(collision)}")
    panel = baseline.merge(hourly.reset_index(), on="timestamp_utc", how="left", validate="many_to_one")
    if len(panel) != len(baseline):
        raise ValueError("Feature join changed the baseline evaluation population.")
    return panel
