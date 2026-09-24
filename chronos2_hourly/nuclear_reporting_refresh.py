"""Fresh, report-only actuals and official Storm snapshots for nuclear reports.

No forecast, calibration cache, input snapshot or live archive is modified.
Failures are explicit: using an older archive is a separate caller decision.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.atomic_directory import AtomicDirectoryStaging
from chronos2_hourly.storm_dashboard import (
    STORM_DASHBOARD_COLUMN, build_dashboard_comparator,
    fetch_native_dashboard_snapshot, storm_dashboard_series,
)
from chronos2_modular.saturn import create_saturn_client
from chronos2_hourly.reporting_observations import (
    EPEX_REPORTING_POLICY, EPEX_REPORTING_SERIES_BY_ZONE, EPEX_REPORTING_SOURCE_KIND,
    ReportingObservationError, epex_reporting_identity, fetch_epex_reporting_observations,
)


class NuclearReportingRefreshError(ValueError):
    """The latest report-only extraction does not meet the physical contract."""


_STORM_ARTIFACT = "inputs/storm_dashboard_official_statistics.parquet"
_OBSERVED_ARTIFACT = "inputs/observed_latest.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _support(delivery_day: str, timezone: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    day = pd.Timestamp(delivery_day)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise NuclearReportingRefreshError("delivery_day doit etre une date civile sans fuseau.")
    index = pd.date_range((day-pd.Timedelta(days=365)).tz_localize(timezone),
                          (day+pd.Timedelta(days=1)).tz_localize(timezone),
                          freq="h", inclusive="left").tz_convert("UTC")
    return index, local_delivery_day_index(day.date(), timezone=timezone)


def _utc_series(values: pd.Series, name: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise NuclearReportingRefreshError(f"{name}: Series requise.")
    index = pd.DatetimeIndex(values.index)
    if index.tz is None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise NuclearReportingRefreshError(f"{name}: timeline UTC physique unique et ordonnee requise.")
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise NuclearReportingRefreshError(f"{name}: prix infini interdit.")
    return pd.Series(numeric, index=index.tz_convert("UTC"), name=name)


def _fetch_observed(client: Any, **kwargs):
    # Keep the normal export's strict admission policy. D may remain unscored
    # after a rejected complement; missing historical prices never may.
    # Import lazily: report-only module import never starts a model or a run.
    from run_multicountry_forecast import (
        PostAuctionObservationDivergenceError, _fetch_latest_observed_snapshot,
    )
    failures = []
    retry_delays = (2, 4)
    for attempt in range(1, 4):
        try:
            # Retry transient source revisions first. On the final reading only,
            # complete history permits D to remain unscored if EPEX is rejected.
            options = {**kwargs, "allow_pending_current_day": True} if attempt == 3 else kwargs
            values, source = _fetch_latest_observed_snapshot(client, **options)
        except PostAuctionObservationDivergenceError as error:
            failures.append({"attempt": attempt, **error.diagnostic})
            if attempt == 3:
                raise
            delay = retry_delays[attempt - 1]
            print(f"[Nuclear/{kwargs['spec'].zone}] observations recentes en desaccord; "
                  f"nouvelle lecture des deux sources {attempt + 1}/3 dans {delay} s.", flush=True)
            time.sleep(delay)
            kwargs = {**kwargs, "extracted_at_utc": pd.Timestamp.now(tz="UTC")}
            continue
        fallback = source.get("post_auction_fallback", {})
        rejected = fallback.get("status") == "rejected_divergent_optional_current_day"
        if rejected:
            failures.append({"attempt": attempt, **fallback["rejection_diagnostic"]})
            print(f"[Nuclear/{kwargs['spec'].zone}] prix realises du jour non valides : "
                  "sources en desaccord. Complement rejete ; previsions conservees, "
                  "scores du jour non calcules.", flush=True)
        source = {**source, "observation_pair_read": {
            "policy": "reread_latest_pair_on_divergence_only", "attempts": attempt,
            "maximum_attempts": 3, "divergence_failures_count": len(failures),
            "retry_delays_seconds": list(retry_delays[:len(failures)]),
            "failures": failures, "used_for_prediction": False,
            "optional_current_day_fallback_rejected": rejected,
        }}
        return values, source


def _fetch_epex_observed(client: Any, *, spec: Any, expected_index: pd.DatetimeIndex,
                         extracted_at_utc: pd.Timestamp):
    try:
        return fetch_epex_reporting_observations(
            client, zone=spec.zone, timezone=spec.timezone, expected_index=expected_index,
            extracted_at_utc=extracted_at_utc)
    except ReportingObservationError as error:
        raise NuclearReportingRefreshError(str(error)) from error


def refresh_nuclear_reporting_sources(
    config: dict, zone: str, timezone: str, delivery_day: str,
    output_directory: Path, client: Any = None,
) -> tuple[pd.Series, Path, dict[str, Any]]:
    """Fetch latest actuals/Storm and publish one isolated, checksummed snapshot.

    The retained support is D-365 through D. A not-yet-complete auction day is
    entirely masked for actuals, never averaged partially. Historical Storm
    holes may only be the unrepresented first autumn 02:00 physical fold.
    Current-day Storm missing values remain missing and have a separate audit.
    """
    zone = str(zone).upper()
    # ES has a canonical observation source but no verified official Storm
    # comparator. Do not invent a Storm series or block its ordinary reports.
    series = None if zone == "ES" else storm_dashboard_series(zone)
    expected, current = _support(delivery_day, timezone)
    target = config.get("zones", {}).get(zone, {}).get("target", {}).get("series")
    if not isinstance(target, str) or not target:
        raise NuclearReportingRefreshError(f"{zone}: serie cible canonique absente de la configuration.")
    data_config = config.get("data", {})
    root = Path(output_directory).resolve()
    # New snapshots only. Never repurpose the project, model cache or sealed archive.
    if root == Path(root.anchor) or root == Path(__file__).resolve().parents[1]:
        raise NuclearReportingRefreshError("Un sous-dossier de reporting isole est requis.")
    if any((root / marker).exists() for marker in
           ("artifact_checksums.json", "input_snapshot.json", "resolved_config.yaml")):
        raise NuclearReportingRefreshError("Le dossier de sortie est un artefact gele, pas un dossier de reporting.")
    if client is None:
        client = create_saturn_client(data_config.get("saturn_url"),
                                      os.getenv("SATURN_AUTHOR") or data_config.get("saturn_author"))
    extracted = pd.Timestamp.now(tz="UTC")
    spec = SimpleNamespace(zone=zone, timezone=timezone, delivery_day=str(pd.Timestamp(delivery_day).date()),
                           history_contract=SimpleNamespace(target_series=target))
    epex_reference = zone in EPEX_REPORTING_SERIES_BY_ZONE
    observed_series = EPEX_REPORTING_SERIES_BY_ZONE[zone] if epex_reference else target
    reader = _fetch_epex_observed if epex_reference else _fetch_observed
    observed_raw, observed_source = reader(
        client, spec=spec, expected_index=expected, extracted_at_utc=extracted)
    # A coherence retry reads a new pair. Its successful extraction timestamp
    # must name the report snapshot and the subsequent Storm extraction audit.
    extracted = pd.Timestamp(observed_source["extracted_at_utc"])
    observed = _utc_series(observed_raw, "actual").reindex(expected)
    if observed.loc[expected.difference(current)].isna().any():
        raise NuclearReportingRefreshError(f"{zone}: observations historiques recentes incompletes; aucun cache ancien substitue.")
    raw_available = int(observed.reindex(current).notna().sum())
    current_complete = raw_available == len(current)
    if not current_complete:
        observed.loc[current] = np.nan
    observed_source = {
        **deepcopy(observed_source), "current_delivery_day_local": spec.delivery_day,
        "current_delivery_expected_hours": len(current),
        "current_delivery_raw_available_hours": raw_available,
        "current_delivery_actual_status": "complete" if current_complete else "pending_placeholder",
        "partial_daily_average_forbidden": True, "used_for_prediction": False,
    }
    required_actual_source = (
        epex_reporting_identity(zone, timezone) if epex_reference else
        {"kind": "saturn_target_latest_extraction", "series": target,
         "zone": zone, "nocache": True, "live_recomputation": True, "used_for_prediction": False})
    if any(observed_source.get(key) != value for key, value in required_actual_source.items()):
        raise NuclearReportingRefreshError("Provenance de la reference d'observation recente divergente.")
    if epex_reference and "post_auction_fallback" in observed_source:
        raise NuclearReportingRefreshError("Une reference EPEX unique ne peut contenir de complement.")
    comparator = None
    if series is not None:
        raw_storm, storm_source = fetch_native_dashboard_snapshot(
            client, zone=zone, expected_index=expected, extracted_at_utc=extracted)
        storm = _utc_series(raw_storm, STORM_DASHBOARD_COLUMN).reindex(expected)
        for key, value in {"kind": "saturn_storm_day_ahead_cache_with_native_gap_fallback",
                           "series": series, "primary_series": series, "zone": zone,
                           "used_for_prediction": False}.items():
            if storm_source.get(key) != value:
                raise NuclearReportingRefreshError(f"Source officielle Storm divergente: {key}.")
        if storm_source.get("fallback_series") not in (None, f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm"):
            raise NuclearReportingRefreshError("Fallback Storm non officiel.")
        missing = expected[storm.isna()]
        pending_missing = missing.intersection(current)
        historical_missing = missing.difference(current)
        local_naive = expected.tz_convert(timezone).tz_localize(None)
        standard_fold = local_naive.tz_localize(timezone, ambiguous=False, nonexistent="NaT").tz_convert("UTC")
        allowed_dst = expected[standard_fold != expected]
        if len(historical_missing.difference(allowed_dst)):
            raise NuclearReportingRefreshError("Storm officiel incomplet hors trou DST historique; aucun remplissage ni cache ancien substitue.")
        storm_source = {
            **deepcopy(storm_source), "series_kind": "frozen_day_ahead_cache",
            "fallback_policy": "native_only_where_day_ahead_cache_is_missing", "cache_precedence": True,
        }
        comparator = build_dashboard_comparator(
            storm, expected_index=expected, actual=observed, source=storm_source,
            timezone=timezone, minimum_coverage=0, maximum_missing_hours=len(missing),
            allowed_missing_actual_index=current if not current_complete else pd.DatetimeIndex([], tz="UTC"),
        )
        comparator.audit["dst"].update({
            "native_allowed_missing_hours": len(historical_missing),
            "native_allowed_missing_utc": [str(value) for value in historical_missing],
            "native_actual_missing_matches_allowed": True,
        })
        comparator.audit["delivery_placeholder"] = {
            "delivery_day_local": spec.delivery_day, "timezone": timezone,
            "allowed_missing_utc": [str(value) for value in pending_missing],
            "allowed_missing_hours": len(pending_missing),
            "status": "pending_placeholder" if len(pending_missing) else "complete",
            "used_for_prediction": False,
        }
    root.mkdir(parents=True, exist_ok=True)
    final = root / (extracted.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8])
    with AtomicDirectoryStaging(root, prefix=".refresh_") as publication:
        staging = publication.path
        (staging / "inputs").mkdir()
        if comparator is not None:
            pd.DataFrame({"delivery_start_utc": expected, STORM_DASHBOARD_COLUMN: comparator.values.to_numpy(float)}).to_parquet(
                staging / _STORM_ARTIFACT, index=False)
            comparator.audit.update(normalized_artifact_path=_STORM_ARTIFACT,
                                    normalized_artifact_sha256=_sha256(staging / _STORM_ARTIFACT))
        pd.DataFrame({"timestamp": expected, "actual": observed.to_numpy(float)}).to_parquet(
            staging / _OBSERVED_ARTIFACT, index=False)
        observed_audit = {
            "artifact_path": str(final / _OBSERVED_ARTIFACT),
            "relative_artifact_path": _OBSERVED_ARTIFACT,
            "artifact_sha256": _sha256(staging / _OBSERVED_ARTIFACT),
            "series": observed_series, "zone": zone, "timezone": timezone,
            "delivery_day_local": spec.delivery_day, "source": observed_source,
        }
        audit = {
            "schema_version": 2 if epex_reference else 1,
            "status": "complete", "mode": "latest_report_only_refresh",
            "snapshot_directory": str(final), "extracted_at_utc": str(extracted),
            "zone": zone, "timezone": timezone, "delivery_day_local": spec.delivery_day,
            "used_for_prediction": False, "frozen_inputs_modified": False,
            "sealed_archives_modified": False, "old_snapshot_fallback": False,
            "observed": observed_audit,
            "canonical_actuals": {"source": observed_source},
        }
        if epex_reference:
            audit.update(observation_policy=EPEX_REPORTING_POLICY, training_target_series=target)
        if comparator is not None:
            audit.update(storm_primary_report_benchmark=STORM_DASHBOARD_COLUMN,
                         storm_dashboard=comparator.audit)
        else:
            audit["storm_status"] = "unavailable_zone_not_supported"
            audit["storm_unavailable"] = {
                "zone": zone, "reason": "no_verified_official_dashboard_series",
                "used_for_prediction": False, "comparator_fabricated": False,
            }
        (staging / "statistics_history_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        publication.publish(final)
    return observed, final, audit


def verify_refreshed_observations(
    values: pd.Series, audit: Mapping[str, Any], *, zone: str, timezone: str, delivery_day: str,
) -> dict[str, Any]:
    """Verify exact latest values before permitting label revisions for scoring.

    Older fit-context values may be present in ``values``; they are not part
    of this report snapshot and cannot justify a calibration or model change.
    """
    day = str(pd.Timestamp(delivery_day).date())
    for key, value in {"status": "complete", "mode": "latest_report_only_refresh",
                       "zone": zone, "timezone": timezone, "delivery_day_local": day,
                       "used_for_prediction": False, "frozen_inputs_modified": False,
                       "sealed_archives_modified": False, "old_snapshot_fallback": False}.items():
        if audit.get(key) != value:
            raise NuclearReportingRefreshError(f"Audit observations recentes invalide: {key}.")
    evidence = audit.get("observed", {})
    root = Path(audit.get("snapshot_directory", "")).resolve()
    path = Path(evidence.get("artifact_path", "")).resolve()
    if path != root / _OBSERVED_ARTIFACT or evidence.get("relative_artifact_path") != _OBSERVED_ARTIFACT:
        raise NuclearReportingRefreshError("Le snapshot d'observations ne correspond pas au chemin audite.")
    if not path.is_relative_to(root) or not path.is_file() or _sha256(path) != evidence.get("artifact_sha256"):
        raise NuclearReportingRefreshError("Checksum du snapshot d'observations divergent.")
    source = evidence.get("source", {})
    if not isinstance(evidence.get("series"), str) or not evidence["series"]:
        raise NuclearReportingRefreshError("Source canonique absente.")
    for key, value in {"zone": zone, "timezone": timezone, "delivery_day_local": day}.items():
        if evidence.get(key) != value:
            raise NuclearReportingRefreshError(f"Observation recente: {key} divergent.")
    if source.get("kind") == EPEX_REPORTING_SOURCE_KIND:
        try:
            required_identity = epex_reporting_identity(zone, timezone)
        except ReportingObservationError as error:
            raise NuclearReportingRefreshError(str(error)) from error
        if (audit.get("schema_version") != 2 or audit.get("observation_policy") != EPEX_REPORTING_POLICY
                or evidence["series"] != required_identity["series"]
                or "post_auction_fallback" in source
                or audit.get("canonical_actuals", {}).get("source") != source):
            raise NuclearReportingRefreshError("Identite de la reference EPEX divergente.")
        reference = {"actual_reference": "EPEX", "actual_reference_label": "EPEX",
                     "source_kind": EPEX_REPORTING_SOURCE_KIND, "policy": EPEX_REPORTING_POLICY}
    else:
        # Old, sealed reports retain their real identity and remain verifiable.
        # An EPEX series must never be relabelled as a legacy canonical source.
        if (audit.get("schema_version") != 1 or audit.get("observation_policy") is not None
                or evidence["series"] in EPEX_REPORTING_SERIES_BY_ZONE.values()
                or source.get("policy") is not None
                or source.get("actual_reference") == "EPEX"):
            raise NuclearReportingRefreshError("Identite de la reference historique divergente.")
        required_identity = {"kind": "saturn_target_latest_extraction", "series": evidence["series"],
                             "zone": zone, "nocache": True, "live_recomputation": True,
                             "used_for_prediction": False}
        fallback = source.get("post_auction_fallback", {})
        mixed = isinstance(fallback, Mapping) and int(fallback.get("applied_hours", 0)) > 0
        reference = {"actual_reference": "ENTSO-E", "actual_reference_label": "ENTSO-E + EPEX" if mixed else "ENTSO-E",
                     "source_kind": "saturn_target_latest_extraction",
                     "policy": "legacy_canonical_with_validated_fallback"}
    for key, value in {**required_identity, "partial_daily_average_forbidden": True,
                       "current_delivery_day_local": day}.items():
        if source.get(key) != value:
            raise NuclearReportingRefreshError(f"Provenance observation recente: {key} divergent.")
    extracted = pd.Timestamp(source.get("extracted_at_utc"))
    if pd.isna(extracted) or extracted.tzinfo is None:
        raise NuclearReportingRefreshError("Date d'extraction de l'observation absente.")
    frame = pd.read_parquet(path)
    if list(frame.columns) != ["timestamp", "actual"]:
        raise NuclearReportingRefreshError("Schema du snapshot d'observations divergent.")
    expected, current = _support(day, timezone)
    frozen = _utc_series(pd.Series(frame.actual.to_numpy(), index=pd.DatetimeIndex(frame.timestamp)), "actual")
    if not frozen.index.equals(expected):
        raise NuclearReportingRefreshError("Support du snapshot d'observations divergent.")
    if frozen.loc[expected.difference(current)].isna().any():
        raise NuclearReportingRefreshError("Observations historiques incompletes.")
    future_count = int(frozen.loc[current].notna().sum())
    status = "complete" if future_count == len(current) else "pending_placeholder"
    if (future_count not in (0, len(current)) or source.get("current_delivery_actual_status") != status
            or source.get("current_delivery_expected_hours") != len(current)):
        raise NuclearReportingRefreshError("Statut d'observation du jour de livraison divergent.")
    actual = _utc_series(values, "actual")
    if len(expected.difference(actual.index)) or not np.array_equal(
            actual.reindex(expected).to_numpy(float), frozen.to_numpy(float), equal_nan=True):
        raise NuclearReportingRefreshError("Les observations du rapport divergent du snapshot recent audite.")
    if _sha256(path) != evidence["artifact_sha256"]:
        raise NuclearReportingRefreshError("Le snapshot d'observations a change pendant sa verification.")
    return {"status": "verified", "mode": "latest_report_only_refresh", "compared_hours": len(expected),
            **reference,
            "series": evidence["series"], "extracted_at_utc": str(extracted),
            "artifact_sha256": evidence["artifact_sha256"], "artifact_path": str(path),
            "used_for_prediction": False, "calibration_inputs_modified": False}


__all__ = ["NuclearReportingRefreshError", "refresh_nuclear_reporting_sources", "verify_refreshed_observations"]
