"""Read-only, audit-checked Storm comparisons for nuclear reports.

Only reporting attributes are attached. Candidate predictions, observations,
native scores and their complete 365-day support are never rewritten.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.reporting import STORM_DASHBOARD_CONTRACT_ID, storm_benchmark_contracts
from chronos2_hourly.storm_dashboard import STORM_DASHBOARD_COLUMN, build_dashboard_comparator, storm_dashboard_series


class NuclearBenchmarkError(ValueError):
    """A present comparator has invalid provenance or physical pairing."""


_ARTIFACT = "inputs/storm_dashboard_official_statistics.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: Any, label: str) -> pd.DatetimeIndex:
    parsed = [pd.Timestamp(value) for value in values]
    if any(pd.isna(value) or value.tzinfo is None for value in parsed):
        raise NuclearBenchmarkError(f"{label}: timestamps avec fuseau explicite requis.")
    index = pd.DatetimeIndex(pd.to_datetime(parsed, utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise NuclearBenchmarkError(f"{label}: timeline physique non unique ou desordonnee.")
    return index


def _load_verified_snapshot(archive: Path, *, zone: str, timezone: str):
    root = Path(archive).resolve()
    path = root / _ARTIFACT
    audit_path = root / "statistics_history_audit.json"
    if not path.is_file() and not audit_path.is_file():
        return None
    if not audit_path.is_file():
        raise NuclearBenchmarkError("Storm present sans statistics_history_audit.json.")
    try:
        history_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise NuclearBenchmarkError("Audit Statistics/Storm illisible.") from exc
    if not isinstance(history_audit, Mapping):
        raise NuclearBenchmarkError("Audit Statistics/Storm doit etre un objet.")
    declared = history_audit.get("storm_primary_report_benchmark")
    source_audit = history_audit.get("storm_dashboard")
    if not path.is_file() and declared is None and source_audit is None:
        return None
    if not path.is_file() or declared != STORM_DASHBOARD_COLUMN or not isinstance(source_audit, Mapping):
        raise NuclearBenchmarkError("Paire Storm officiel/contrat/audit incomplete ou divergente.")
    if path.resolve().parent != (root / "inputs").resolve() or not path.resolve().is_relative_to(root):
        raise NuclearBenchmarkError("Le snapshot Storm doit rester dans son archive.")
    expected_series = storm_dashboard_series(zone)
    if history_audit.get("status") != "complete":
        raise NuclearBenchmarkError("Audit Statistics incomplet.")
    required = {
        "role": "evaluation_only_dashboard_comparator", "series": expected_series,
        "column": STORM_DASHBOARD_COLUMN, "used_for_prediction": False,
        "used_for_live_forecast": False, "normalized_artifact_path": _ARTIFACT,
    }
    for key, value in required.items():
        if source_audit.get(key) != value:
            raise NuclearBenchmarkError(f"Contrat Storm invalide: {key}.")
    source = source_audit.get("source")
    if not isinstance(source, Mapping):
        raise NuclearBenchmarkError("Provenance du cache officiel Storm absente.")
    source_required = {
        "kind": "saturn_storm_day_ahead_cache_with_native_gap_fallback",
        "zone": zone, "series": expected_series, "primary_series": expected_series,
        "series_kind": "frozen_day_ahead_cache", "used_for_prediction": False,
        "fallback_policy": "native_only_where_day_ahead_cache_is_missing", "cache_precedence": True,
    }
    for key, value in source_required.items():
        if source.get(key) != value:
            raise NuclearBenchmarkError(f"Source Storm officiel non verifiee: {key}.")
    if source.get("fallback_series") not in (None, f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm"):
        raise NuclearBenchmarkError("Source fallback Storm non officielle; le strict-08 ne peut pas la remplacer.")
    _utc([source.get("extracted_at_utc")], "Storm/extracted_at_utc")
    digest = _sha256(path)
    if source_audit.get("normalized_artifact_sha256") != digest:
        raise NuclearBenchmarkError("Checksum du snapshot Storm divergent.")
    frame = pd.read_parquet(path)
    if not {"delivery_start_utc", STORM_DASHBOARD_COLUMN}.issubset(frame.columns):
        raise NuclearBenchmarkError("Colonnes du snapshot officiel Storm absentes.")
    index = _utc(frame.delivery_start_utc, "Storm")
    if index.empty or not index.equals(index.floor("h")):
        raise NuclearBenchmarkError("Storm exige une grille physique horaire non vide.")
    values = pd.Series(pd.to_numeric(frame[STORM_DASHBOARD_COLUMN], errors="raise").to_numpy(float), index=index)
    if np.isinf(values).any():
        raise NuclearBenchmarkError("Le snapshot Storm contient un prix infini.")
    period = source_audit.get("period")
    if not isinstance(period, Mapping) or period.get("timezone") != timezone:
        raise NuclearBenchmarkError("Periode ou fuseau du snapshot Storm invalide.")
    start = _utc([period.get("start_utc")], "Storm/start")[0]
    end = _utc([period.get("end_utc")], "Storm/end")[0]
    expected = pd.date_range(start, end, freq="h")
    if not index.equals(expected):
        raise NuclearBenchmarkError("La grille Storm ne correspond pas exactement a sa periode auditee.")
    if (start != local_delivery_day_index(start.tz_convert(timezone).date(), timezone=timezone)[0]
            or end != local_delivery_day_index(end.tz_convert(timezone).date(), timezone=timezone)[-1]):
        raise NuclearBenchmarkError("La periode du cache Storm doit contenir des jours physiques complets.")
    missing = index[values.isna()]
    for key, value in {"expected_hours": len(index), "available_hours": len(index)-len(missing), "missing_hours": len(missing)}.items():
        if source_audit.get(key) != value:
            raise NuclearBenchmarkError(f"Couverture Storm/audit divergente: {key}.")
    dst = source_audit.get("dst")
    if not isinstance(dst, Mapping) or dst.get("interpolation") is not False or dst.get("strict_08_fallback") is not False:
        raise NuclearBenchmarkError("Audit DST Storm absent ou autorisant un remplissage interdit.")
    allowed = _utc(dst.get("native_allowed_missing_utc", []), "Storm/DST")
    placeholder = source_audit.get("delivery_placeholder")
    placeholder_missing = pd.DatetimeIndex([], tz="UTC")
    if placeholder is not None:
        if not isinstance(placeholder, Mapping) or placeholder.get("used_for_prediction") is not False:
            raise NuclearBenchmarkError("Audit du jour Storm non publie invalide.")
        placeholder_day = pd.Timestamp(placeholder.get("delivery_day_local")).date()
        if (placeholder.get("timezone") != timezone or placeholder_day != end.tz_convert(timezone).date()
                or placeholder.get("status") not in ("pending_placeholder", "complete")):
            raise NuclearBenchmarkError("Le placeholder Storm doit designer le dernier jour physique du snapshot.")
        placeholder_missing = _utc(placeholder.get("allowed_missing_utc", []), "Storm/placeholder")
        if (placeholder.get("allowed_missing_hours") != len(placeholder_missing)
                or any(value.tz_convert(timezone).date() != placeholder_day for value in placeholder_missing)
                or (placeholder.get("status") == "complete") != (len(placeholder_missing) == 0)
                or len(placeholder_missing.intersection(allowed))):
            raise NuclearBenchmarkError("Les trous du placeholder Storm ne correspondent pas a leur audit.")
    all_allowed = allowed.union(placeholder_missing).sort_values()
    if (dst.get("native_allowed_missing_hours") != len(allowed)
            or dst.get("native_actual_missing_matches_allowed") is not True or not missing.equals(all_allowed)):
        raise NuclearBenchmarkError("Les trous Storm ne correspondent pas exactement a l'audit DST.")
    for timestamp in allowed:
        local = timestamp.tz_convert(timezone)
        physical = local_delivery_day_index(local.date(), timezone=timezone)
        if len(physical) != 25 or local.hour != 2:
            raise NuclearBenchmarkError("Un trou Storm declare DST ne correspond pas a une heure d'automne.")
    if _sha256(path) != digest:
        raise NuclearBenchmarkError("Le snapshot Storm a change pendant sa lecture.")
    return values, deepcopy(dict(source_audit)), {
        "archive": str(root), "artifact_path": str(path), "artifact_sha256": digest,
        "audit_path": str(audit_path), "audit_sha256": _sha256(audit_path),
    }


def _comparison_support(result: Any, *, timezone: str) -> pd.DataFrame:
    source = getattr(result, "statistics_candidate", None)
    if not isinstance(source, pd.DataFrame) or not {"timestamp", "actual", "q50"}.issubset(source.columns):
        raise NuclearBenchmarkError("Le resultat nucleaire doit fournir Statistics candidat avec timestamp/actual/q50.")
    future = result.forecast_native
    future_index = _utc(future["timestamp"], "forecast candidat")
    days = pd.Index(future_index.tz_convert(timezone).date).unique()
    if len(days) != 1 or not future_index.equals(local_delivery_day_index(days[0], timezone=timezone)):
        raise NuclearBenchmarkError("Le forecast candidat doit couvrir exactement un jour physique.")
    delivery_day = pd.Timestamp(days[0])
    index = _utc(source.timestamp, "Statistics candidat")
    actual = pd.to_numeric(source.actual, errors="raise").to_numpy(float)
    point = pd.to_numeric(source.q50, errors="raise").to_numpy(float)
    if np.isinf(actual).any() or not np.isfinite(point).all():
        raise NuclearBenchmarkError("Le candidat contient des prix non finis invalides.")
    support = pd.DataFrame({"actual": actual, "q50": point}, index=index)
    future_point = pd.to_numeric(future["q50"], errors="raise").to_numpy(float)
    if not np.allclose(support.q50.reindex(future_index), future_point, rtol=0, atol=1e-9):
        raise NuclearBenchmarkError("Les prix prevus du jour different entre Statistics et forecast candidat.")
    delivery_actual = support.actual.reindex(future_index)
    if delivery_actual.notna().any() and not delivery_actual.notna().all():
        raise NuclearBenchmarkError("Le jour de livraison doit etre observe entierement ou rester vide.")
    start_day = delivery_day - pd.Timedelta(days=364 if delivery_actual.notna().all() else 365)
    expected = pd.date_range(start_day.tz_localize(timezone), (delivery_day + pd.Timedelta(days=1)).tz_localize(timezone),
                             freq="h", inclusive="left").tz_convert("UTC")
    if len(expected.difference(index)):
        raise NuclearBenchmarkError("Statistics candidat ne couvre pas les 365 jours physiques requis.")
    support = support.reindex(expected)
    historical = expected < future_index[0]
    if not np.isfinite(support.actual.to_numpy(float)[historical]).all():
        raise NuclearBenchmarkError("Les observations historiques du candidat sont incompletes.")
    support["_timestamp_utc"] = expected
    support["_timestamp_local"] = expected.tz_convert(timezone)
    return support.reset_index(drop=True)


def attach_nuclear_storm(
    prepared: Mapping[str, Any], archive: Path, zone: str, timezone: str,
) -> dict[str, Any]:
    """Attach the standard Statistics, mean-price/calendar and hourly hooks.

    All candidate timestamps are retained. Storm remains NaN for its declared
    DST hole and outside the archive's period, including unpublished delivery.
    Every comparison uses finite candidate/actual/Storm triplets. Fully missing
    Storm days remain visible with standalone observed/candidate prices and an
    empty comparator. Native model scores and all candidate frames are intact.
    """
    zone = str(zone).upper()
    loaded = _load_verified_snapshot(Path(archive), zone=zone, timezone=timezone)
    if loaded is None:
        return {"status": "unavailable", "reason": "no_verified_local_official_storm_snapshot",
                "used_for_prediction": False, "native_statistics_modified": False}
    if not prepared:
        raise NuclearBenchmarkError("Aucun resultat nucleaire a apparier.")
    values, source_audit, provenance = loaded
    pending = []
    reference = None
    by_variant = {}
    for key, result in prepared.items():
        if str(result.zone).upper() != zone:
            raise NuclearBenchmarkError("La zone du candidat differe de celle du benchmark.")
        comparison = _comparison_support(result, timezone=timezone)
        expected = pd.DatetimeIndex(comparison._timestamp_utc)
        placeholder = source_audit.get("delivery_placeholder")
        if placeholder is not None:
            delivery_day = pd.DatetimeIndex(result.forecast_native.timestamp).tz_convert(timezone)[0].date()
            if pd.Timestamp(placeholder["delivery_day_local"]).date() != delivery_day:
                raise NuclearBenchmarkError("Le placeholder Storm differe du jour prevu par le candidat.")
        actual = pd.Series(comparison.actual.to_numpy(float), index=expected)
        if reference is not None:
            reference_index, reference_actual = reference
            if not expected.equals(reference_index) or not np.allclose(actual.to_numpy(float), reference_actual, rtol=0, atol=1e-9, equal_nan=True):
                raise NuclearBenchmarkError("Les variantes nucleaires ne partagent pas les memes heures et observations.")
        reference = (expected, actual.to_numpy(float))
        aligned = values.reindex(expected)
        missing = expected[aligned.isna()]
        missing_actual = expected[actual.isna()]
        comparator = build_dashboard_comparator(
            values, expected_index=expected, actual=actual, source=source_audit["source"],
            timezone=timezone, minimum_coverage=0, maximum_missing_hours=len(missing),
            allowed_missing_actual_index=missing_actual,
        )
        comparison["_benchmark_q50"] = comparator.values.to_numpy(float)
        matched = np.isfinite(comparison[["actual", "q50", "_benchmark_q50"]].to_numpy(float)).all(axis=1)
        observed = actual.notna().to_numpy()
        local_days = expected.tz_convert(timezone).date
        daily = pd.DataFrame({"matched": matched.astype(int), "hours": 1}, index=local_days).groupby(level=0).sum()
        declared_dst = pd.DatetimeIndex(pd.to_datetime(source_audit["dst"]["native_allowed_missing_utc"], utc=True))
        audit = {
            **comparator.audit, "source_materialization": source_audit, "snapshot": provenance,
            "variant": str(key), "expected_comparison_hours": len(expected), "matched_hours": int(matched.sum()),
            "observed_hours": int(observed.sum()), "unmatched_observed_hours": int((observed & ~matched).sum()),
            "forecast_placeholder_hours": len(missing_actual), "storm_missing_hours": len(missing),
            "storm_missing_dst_hours": len(missing.intersection(declared_dst)),
            "storm_missing_outside_archive_hours": len(expected.difference(values.index)),
            "fully_paired_days": int(daily.matched.eq(daily.hours).sum()),
            "partially_paired_days": int(((daily.matched > 0) & (daily.matched < daily.hours)).sum()),
            "full_observed_coverage": bool(matched[observed].all()),
            "native_statistics_modified": False, "observations_source": "supplied_nuclear_result_only",
            "comparison_scope": "latest_365_observed_civil_days_plus_unobserved_delivery_placeholder",
            "missing_benchmark_utc": [value.isoformat() for value in missing],
        }
        contract = deepcopy(storm_benchmark_contracts(zone, timezone=timezone)[STORM_DASHBOARD_CONTRACT_ID])
        contract.update(
            status="loaded_from_explicit_audited_artifact", used_for_prediction=False,
            report_note=(f"Comparaisons sur {audit['matched_hours']} heures physiques appariees; "
                         f"{audit['unmatched_observed_hours']} heures observees sans Storm. "
                         "Les prix du modele et observes restent visibles quand Storm est absent, sans score comparatif. "
                         "Aucun remplissage des trous DST ni du jour de livraison absent du cache. "
                         "Source: cache officiel audite et date d'extraction conservee."),
            materialization_audit=audit,
        )
        pending.append((result, comparison, contract))
        by_variant[str(key)] = audit
    # Do not expose a partially attached set if a later variant fails checks.
    for result, comparison, contract in pending:
        result.hourly_comparison_source = comparison
        result.hourly_comparison_contract = contract
        benchmark = pd.DataFrame({
            "timestamp": comparison._timestamp_local,
            "actual": comparison.actual,
            "q50": comparison._benchmark_q50,
        })
        result.statistics_benchmark = benchmark
        result.statistics_benchmark_label = contract["report_label"]
        result.statistics_benchmark_contract = contract
        result.statistics_benchmark_contract_catalog = storm_benchmark_contracts(zone, timezone=timezone)
        result.statistics_pairing_audit = {
            "status": "complete", "role": "verified_official_storm_pairing",
            "zone": zone, "used_for_prediction": False,
            "expected_hours": len(comparison),
            "missing_benchmark_utc": contract["materialization_audit"]["missing_benchmark_utc"],
        }
        forecast_index = _utc(result.forecast_native.timestamp, "forecast candidat")
        future_values = values.reindex(forecast_index)
        if future_values.notna().any():
            result.forecast_benchmark = pd.DataFrame({
                "timestamp": result.forecast_native.timestamp.to_numpy(),
                "q50": future_values.to_numpy(float),
            })
            result.forecast_benchmark_label = contract["report_label"]
            result.forecast_benchmark_audit = {
                "status": "complete", "used_for_prediction": False, "snapshot": provenance,
                "expected_hours": len(forecast_index), "available_hours": int(future_values.notna().sum()),
                "missing_hours": int(future_values.isna().sum()),
                "daily_means_scope": "exact_paired_physical_hours",
            }
        result.statistics_scope_note = contract["report_note"]
    return {"status": "complete", "used_for_prediction": False, "native_statistics_modified": False,
            "statistics_comparison_attached": True, "snapshot": provenance, "variants": by_variant}


__all__ = ["NuclearBenchmarkError", "attach_nuclear_storm"]
