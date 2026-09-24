"""Read existing, verified nuclear forecasts and report sources for CWE.

No model, source refresh, cache mutation or prior-delivery fallback is performed.
Missing and rejected sources are kept explicit in the returned report payload.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
from chronos2_hourly.nuclear_reporting_refresh import verify_refreshed_observations
from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
from chronos2_hourly.storm_dashboard import STORM_DASHBOARD_COLUMN


CWE_ZONES = (
    ("BE", "Belgium", "Europe/Brussels"),
    ("DE", "Germany", "Europe/Berlin"),
    ("FR", "France", "Europe/Paris"),
    ("NL", "The Netherlands", "Europe/Amsterdam"),
)
MODEL_COLUMN = "residual_kalman__q50"
_MODEL_COLUMNS = {"model_p10": "residual_kalman__q10", "model": MODEL_COLUMN,
                  "model_p90": "residual_kalman__q90"}
_MODEL_QUANTILE_COLUMNS = {"p10": _MODEL_COLUMNS["model_p10"], "p50": MODEL_COLUMN,
                           "p90": _MODEL_COLUMNS["model_p90"]}
_SOURCE_ERRORS = (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError)
_ROLLING_DAYS = 366  # 365 historical civil days plus the delivery placeholder.
_DASHBOARD_CACHE = "storm_dashboard_cache"
_ROLLING_BASE_SOURCES = ("observed", "storm", "model")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _day(value: str) -> str:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
        raise ValueError("delivery_day must be a local calendar date (YYYY-MM-DD).")
    return str(parsed.date())


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame or frame.columns.has_duplicates:
        raise ValueError(f"Required unique price column absent: {column}.")
    key = next((name for name in ("delivery_start_utc", "timestamp") if name in frame), None)
    raw_index = frame[key] if key else frame.index
    parsed = [pd.Timestamp(value) for value in raw_index]
    if any(pd.isna(value) or value.tzinfo is None for value in parsed):
        raise ValueError("Explicit timezone-aware hourly timestamps are required.")
    index = pd.DatetimeIndex(pd.to_datetime(parsed, utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing or not index.equals(index.floor("h")):
        raise ValueError("Hourly timestamps must be unique and ordered.")
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    if np.isinf(values).any():
        raise ValueError("Infinite prices are invalid.")
    return pd.Series(values, index=index)


def _source_state(values: pd.Series, expected: pd.DatetimeIndex, **metadata: Any) -> dict[str, Any]:
    count = int(values.reindex(expected).notna().sum())
    return {**metadata, "status": "complete" if count == len(expected) else "partial" if count else "unavailable",
            "available_hours": count, "expected_hours": len(expected)}


def _failed(exc: Exception) -> dict[str, Any]:
    return {"status": "invalid", "error": str(exc)}


def _rolling_source(rolling: dict[str, Any] | None, key: str, values: pd.Series | None,
                    source: dict[str, Any]) -> None:
    if rolling is None:
        return
    expected = rolling["index"]
    rolling["values"][key] = values.reindex(expected) if values is not None else None
    if source.get("status") == "invalid" or values is None:
        rolling["sources"][key] = dict(source)
    else:
        metadata = {name: value for name, value in source.items()
                    if name not in {"status", "available_hours", "expected_hours"}}
        rolling["sources"][key] = _source_state(values, expected, **metadata)


def _cache_only_history(storm: pd.Series, source_audit: dict[str, Any], *, timezone: str):
    """Recover cache-only values only when the audited fallback mask is provable.

    Existing snapshots store the merged comparator, fallback day inventory and
    total fallback hour count, not a raw cache. A day inventory identifies exact
    instants only if its total physical-hour capacity equals that count. Partial
    fallback days therefore fail closed instead of silently removing good cache
    prices or presenting native values as frozen day-ahead cache prices.
    """
    source = source_audit.get("source")
    if not isinstance(source, dict):
        raise ValueError("Cache-only Storm requires an audited source mapping.")
    required = {"cache_precedence": True,
                "fallback_policy": "native_only_where_day_ahead_cache_is_missing"}
    for key, value in required.items():
        if source.get(key) != value:
            raise ValueError(f"Cache-only Storm source contract differs: {key}.")
    counts = {}
    for key in ("fallback_used_hours", "cache_available_hours", "cache_missing_hours"):
        value = source.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Cache-only Storm requires a nonnegative audited integer: {key}.")
        counts[key] = value
    days = source.get("fallback_used_local_days")
    if (not isinstance(days, list) or any(not isinstance(day, str) for day in days)
            or len(days) != len(set(days))):
        raise ValueError("Cache-only Storm requires unique audited fallback civil days.")
    masked = pd.DatetimeIndex([], tz="UTC")
    for day in days:
        if _day(day) != day:
            raise ValueError("Cache-only Storm fallback dates must use YYYY-MM-DD.")
        physical = local_delivery_day_index(day, timezone=timezone)
        if len(physical.difference(storm.index)):
            raise ValueError("Cache-only Storm fallback day falls outside its audited snapshot.")
        masked = masked.union(physical)
    if len(masked) != counts["fallback_used_hours"]:
        raise ValueError("Cache-only Storm exact fallback hours are not provable from partial-day metadata.")
    if not storm.reindex(masked).notna().all():
        raise ValueError("Cache-only Storm audited fallback hours contain unavailable merged values.")
    cache = storm.copy()
    cache.loc[masked] = float("nan")
    available = int(cache.notna().sum())
    if (available != counts["cache_available_hours"]
            or len(cache) - available != counts["cache_missing_hours"]):
        raise ValueError("Cache-only Storm availability differs from its audited source counts.")
    return cache, {
        "kind": "audited_storm_day_ahead_cache_only",
        "series": source.get("series"),
        "method": "audited_complete_fallback_days",
        "cache_precedence": True,
        "native_fallback_removed": True,
        "fallback_used_local_days": list(days),
        "removed_fallback_hours": len(masked),
        "audited_cache_available_hours": available,
        "audited_cache_missing_hours": len(cache) - available,
        "exact_cache_hour_mask_verified": True,
        "official_dashboard_formula_verified": False,
        "raw_official_hourly_export_verified": False,
        "used_for_prediction": False,
    }


def _model_history(result: Any, directory: Path, expected: pd.DatetimeIndex,
                   source_day: str, day: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use only verified forecast columns; never the archive's training labels."""
    selected = []
    artifacts = []
    for name, frame in (("kalman_backtest.parquet", result.kalman_view.backtest),
                        ("kalman_forecast.parquet", result.kalman_view.forecast)):
        values = pd.DataFrame({key: _series(frame, column) for key, column in _MODEL_COLUMNS.items()})
        values = values.loc[(values.index >= expected[0]) & (values.index <= expected[-1])]
        if values.empty:
            continue
        quantiles = values.to_numpy(float)
        if not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
            raise ValueError("Historical model quantiles must be finite and satisfy P10 <= P50 <= P90.")
        selected.append(values)
        artifact = directory / name
        artifacts.append({"artifact_path": str(artifact), "artifact_sha256": _sha256(artifact)})
    values = pd.concat(selected) if selected else pd.DataFrame(columns=list(_MODEL_COLUMNS), index=expected[:0])
    if values.index.has_duplicates or not values.index.is_monotonic_increasing:
        raise ValueError("Backtest and forecast must form a unique, ordered physical timeline.")
    return values, {
        "kind": "nuclear_kalman_verified_backtest_and_forecast", "source_delivery_day": source_day,
        "delivery_day": day, "column": MODEL_COLUMN, "artifacts": artifacts,
        "quantile_columns": dict(_MODEL_QUANTILE_COLUMNS),
        "manifest_path": str(directory / "manifest.json"),
        "manifest_sha256": _sha256(directory / "manifest.json"),
        "used_for_prediction": False, "observations_from_backtest": False,
    }


def _rolling_model(rolling: dict[str, Any] | None, values: pd.DataFrame | None,
                   source: dict[str, Any]) -> None:
    """Preserve only native verified quantiles, with one common model provenance."""
    _rolling_source(rolling, "model", values["model"] if values is not None else None, source)
    if rolling is not None:
        for key in ("model_p10", "model_p90"):
            rolling["values"][key] = values[key].reindex(rolling["index"]) if values is not None else None


def _load_model(workspace: Path, zone: str, day: str, expected: pd.DatetimeIndex,
                *, history_from_delivery: str | None = None, rolling: dict[str, Any] | None = None):
    directory = workspace / "report_only" / "frozen_result"
    if not directory.exists():
        source = {"status": "unavailable", "reason": "No completed nuclear forecast for this delivery day."}
        _rolling_model(rolling, None, source)
        return None, source
    try:
        result = load_nuclear_result_bundle(workdir=workspace)
        source_day = history_from_delivery or day
        if result.audit.get("zone") != zone or result.audit.get("delivery_day") != source_day:
            raise ValueError("Frozen nuclear result differs from the requested zone or delivery day.")
        frame = result.kalman_view.backtest if history_from_delivery else result.kalman_view.forecast
        values = pd.DataFrame({key: _series(frame, column)
                               for key, column in _MODEL_COLUMNS.items()})
        if history_from_delivery:
            values = values.loc[(values.index >= expected[0]) & (values.index <= expected[-1])]
        quantiles = values.to_numpy(float)
        if not values.index.equals(expected) or not np.isfinite(quantiles).all():
            raise ValueError("Model quantiles must cover the complete physical delivery day with finite prices.")
        if (np.diff(quantiles, axis=1) < 0).any():
            raise ValueError("Model quantiles must satisfy P10 <= P50 <= P90 for every hour.")
        artifact = directory / ("kalman_backtest.parquet" if history_from_delivery else "kalman_forecast.parquet")
        manifest = directory / "manifest.json"
        if rolling is not None:
            try:
                history, history_source = _model_history(result, directory, rolling["index"], source_day, day)
                _rolling_model(rolling, history, history_source)
            except _SOURCE_ERRORS as exc:
                _rolling_model(rolling, None, _failed(exc))
        return values, _source_state(values["model"], expected,
                                     kind="nuclear_kalman_historical_replay" if history_from_delivery else "nuclear_kalman",
                                     source_delivery_day=source_day, delivery_day=day, column=MODEL_COLUMN,
                                     quantile_columns=dict(_MODEL_QUANTILE_COLUMNS),
                                     artifact_path=str(artifact), artifact_sha256=_sha256(artifact),
                                     manifest_path=str(manifest), manifest_sha256=_sha256(manifest))
    except _SOURCE_ERRORS as exc:
        _rolling_model(rolling, None, _failed(exc))
        return None, _failed(exc)


def _latest_source(workspace: Path):
    """Select the newest published extraction; invalid newer data never falls back."""
    root = workspace / "report_only" / "sources"
    candidates = []
    for path in root.glob("*/statistics_history_audit.json"):
        if path.parent.name.startswith("."):
            continue
        audit = None
        try:
            audit = json.loads(path.read_text(encoding="utf-8"))
            extracted = pd.Timestamp(audit["extracted_at_utc"])
            if pd.isna(extracted) or extracted.tzinfo is None:
                raise ValueError("Invalid extraction time.")
            rank = extracted.value
        except _SOURCE_ERRORS:
            # An unreadable newest publication must not quietly reveal old prices.
            rank = path.stat().st_mtime_ns
        candidates.append((rank, path, audit))
    return max(candidates, key=lambda item: (item[0], str(item[1]))) if candidates else None


def _load_comparators(workspace: Path, zone: str, timezone: str, day: str, expected: pd.DatetimeIndex,
                      *, rolling: dict[str, Any] | None = None):
    empty = {"status": "unavailable", "reason": "No existing report-only source snapshot for this delivery day."}
    sources = {"storm": dict(empty), "observed": dict(empty)}
    values = {"storm": None, "observed": None}
    _rolling_source(rolling, _DASHBOARD_CACHE, None, dict(empty))
    for key in values:
        _rolling_source(rolling, key, None, sources[key])
    try:
        latest = _latest_source(workspace)
        if latest is None:
            return values, sources
        _, audit_path, audit = latest
        if not isinstance(audit, dict):
            raise ValueError("Unreadable report-only source audit.")
        root = audit_path.parent.resolve()
        for key, wanted in {"status": "complete", "mode": "latest_report_only_refresh", "zone": zone,
                            "timezone": timezone, "delivery_day_local": day, "used_for_prediction": False}.items():
            if audit.get(key) != wanted:
                raise ValueError(f"Report-only source identity differs: {key}.")
        if Path(audit.get("snapshot_directory", "")).resolve() != root:
            raise ValueError("Report-only source directory differs from its audit.")
        audit_digest = _sha256(audit_path)
        common = {"extracted_at_utc": audit["extracted_at_utc"], "audit_path": str(audit_path),
                  "audit_sha256": audit_digest}
    except _SOURCE_ERRORS as exc:
        for key in values:
            _rolling_source(rolling, key, None, _failed(exc))
        _rolling_source(rolling, _DASHBOARD_CACHE, None, _failed(exc))
        return values, {"storm": _failed(exc), "observed": _failed(exc)}
    try:
        loaded = _load_verified_snapshot(root, zone=zone, timezone=timezone)
        if loaded is not None:
            storm, storm_audit, provenance = loaded
            metadata = {**common, **{key: value for key, value in provenance.items() if key not in common}}
            # The verified Storm contract contains a point forecast only. Do
            # not invent an interval from NYX quantiles or realized prices.
            metadata.update(quantile_columns={"p50": STORM_DASHBOARD_COLUMN},
                            quantile_interval_available=False)
            _rolling_source(rolling, "storm", storm, metadata)
            if rolling is not None:
                try:
                    cache, cache_metadata = _cache_only_history(storm, storm_audit, timezone=timezone)
                    _rolling_source(rolling, _DASHBOARD_CACHE, cache, {**metadata, **cache_metadata})
                except _SOURCE_ERRORS as exc:
                    _rolling_source(rolling, _DASHBOARD_CACHE, None, {**metadata, **_failed(exc),
                                    "exact_cache_hour_mask_verified": False,
                                    "official_dashboard_formula_verified": False})
            if len(expected.difference(storm.index)):
                # An audited historical snapshot may omit the unpublished
                # delivery altogether. Its past values remain usable, but
                # this must never manufacture a comparator for today's chart.
                sources["storm"] = _failed(ValueError("Storm snapshot does not cover the requested delivery day."))
            else:
                values["storm"] = storm.reindex(expected)
                sources["storm"] = _source_state(values["storm"], expected, **metadata)
    except _SOURCE_ERRORS as exc:
        sources["storm"] = _failed(exc)
        _rolling_source(rolling, "storm", None, sources["storm"])
        _rolling_source(rolling, _DASHBOARD_CACHE, None, _failed(exc))
    try:
        path = root / "inputs" / "observed_latest.parquet"
        if not path.resolve().is_relative_to(root) or _sha256(path) != audit.get("observed", {}).get("artifact_sha256"):
            raise ValueError("Observed source checksum or path differs from its audit.")
        observed = _series(pd.read_parquet(path), "actual")
        provenance = verify_refreshed_observations(observed, audit, zone=zone, timezone=timezone, delivery_day=day)
        values["observed"] = observed.reindex(expected)
        sources["observed"] = _source_state(values["observed"], expected, **common, **{
            key: value for key, value in provenance.items() if key not in common and key != "status"})
        _rolling_source(rolling, "observed", observed, sources["observed"])
        observed_source = audit["observed"].get("source", {})
        if provenance.get("policy") == "legacy_canonical_with_validated_fallback":
            # A verified legacy snapshot can mix canonical ENTSO-E prices and
            # a validated EPEX supplement. Attribute only the hours displayed;
            # the snapshot's delivery day need not be this historical report's.
            fallback = observed_source.get("post_auction_fallback", {})
            applied = fallback.get("applied_value_times_utc")
            if isinstance(applied, list):
                applied_index = pd.DatetimeIndex(pd.to_datetime(applied, utc=True, errors="raise"))
                count = len(expected.intersection(applied_index))
                sources["observed"]["actual_reference_label"] = (
                    "EPEX" if count == len(expected) else "ENTSO-E + EPEX" if count else "ENTSO-E"
                )
                sources["observed"]["displayed_epex_hours"] = count
                if rolling is not None:
                    history_count = len(rolling["index"].intersection(applied_index))
                    rolling["sources"]["observed"].update(
                        actual_reference_label=("EPEX" if history_count == len(rolling["index"])
                                                else "ENTSO-E + EPEX" if history_count else "ENTSO-E"),
                        displayed_epex_hours=history_count,
                    )
        # A later source snapshot may also serve a historical CWE report. Its
        # current-day rejection must never label valid earlier observations.
        if (all(stamp.tz_convert(timezone).date().isoformat() == day for stamp in expected)
                and not values["observed"].notna().any()
                and observed_source.get("current_delivery_actual_reason") == "post_auction_source_rejected"):
            sources["observed"].update(
                current_delivery_actual_reason="post_auction_source_rejected",
                validation_status="rejected_divergence",
            )
    except _SOURCE_ERRORS as exc:
        values["observed"] = None
        sources["observed"] = _failed(exc)
        _rolling_source(rolling, "observed", None, sources["observed"])
    try:
        if _sha256(audit_path) != audit_digest:
            raise ValueError("Source audit changed during reading.")
    except _SOURCE_ERRORS as exc:
        for key in values:
            _rolling_source(rolling, key, None, _failed(exc))
        _rolling_source(rolling, _DASHBOARD_CACHE, None, _failed(exc))
        return {"storm": None, "observed": None}, {key: _failed(exc) for key in values}
    return values, sources


def _rolling_payload(rolling: dict[str, Any], *, day: str, timezone: str) -> dict[str, Any]:
    expected = rolling["index"]
    aligned = {key: value.reindex(expected) if value is not None else pd.Series(np.nan, index=expected)
               for key, value in rolling["values"].items()}
    coverage = {key: int(value.notna().sum()) for key, value in aligned.items()}
    # The optional cache-only extraction must not change the established state
    # of the ordinary model / merged Storm / observed history or day charts.
    base_coverage = [coverage[key] for key in _ROLLING_BASE_SOURCES]
    invalid = any(rolling["sources"].get(key, {}).get("status") == "invalid" for key in _ROLLING_BASE_SOURCES)
    status = ("complete" if all(count == len(expected) for count in base_coverage) else
              "partial" if any(base_coverage) else "invalid" if invalid else "unavailable")
    columns = {key: [float(value) if pd.notna(value) else None for value in series.to_numpy()]
               for key, series in aligned.items()} if any(coverage.values()) else {}
    return {
        "schema_version": 2, "start_day": str(expected[0].tz_convert(timezone).date()),
        "end_day": day, "window_days": _ROLLING_DAYS, "timezone": timezone,
        "expected_hours": len(expected), "coverage": coverage, "status": status,
        "sources": rolling["sources"],
        "rows": [{"timestamp_utc": stamp.isoformat(),
                  **{key: values[number] for key, values in columns.items()}}
                 for number, stamp in enumerate(expected)] if columns else [],
    }


def load_model_storm_payload(project_root: Path, delivery_day: str, nuclear_root: Path | None = None,
                            *, history_from_delivery: str | None = None) -> dict[str, Any]:
    """Assemble one delivery day from canonical ``civil_pit_v2`` completed runs.

    Every CWE zone is retained even when unavailable. The nuclear root defaults
    to ``runs/experiments/nuclear_forecast_v1`` under the project; a relative
    override is also resolved under the project. Returned values are JSON-safe.
    Explicit historical mode selects the requested hours from a later run's
    verified backtest and source snapshots; it never starts or reconstructs a model.
    """
    project = Path(project_root).expanduser().resolve()
    root = Path(nuclear_root) if nuclear_root is not None else Path("runs/experiments/nuclear_forecast_v1")
    root = (project / root).resolve() if not root.is_absolute() else root.resolve()
    day = _day(delivery_day)
    source_day = _day(history_from_delivery) if history_from_delivery is not None else day
    if history_from_delivery is not None and source_day <= day:
        raise ValueError("The historical source delivery must be later than the report delivery.")
    zones = []
    for zone, name, timezone in CWE_ZONES:
        expected = local_delivery_day_index(day, timezone=timezone)
        history_start = pd.Timestamp(day) - pd.Timedelta(days=_ROLLING_DAYS - 1)
        history_end = pd.Timestamp(day) + pd.Timedelta(days=1)
        rolling = {"index": pd.date_range(history_start.tz_localize(timezone), history_end.tz_localize(timezone),
                                          freq="h", inclusive="left").tz_convert("UTC"),
                   "values": {key: None for key in (*_ROLLING_BASE_SOURCES, _DASHBOARD_CACHE,
                                                    "model_p10", "model_p90")}, "sources": {}}
        workspace = root / source_day / zone.lower() / "civil_pit_v2"
        model, model_source = _load_model(workspace, zone, day, expected,
                                        history_from_delivery=source_day if history_from_delivery is not None else None,
                                        rolling=rolling)
        prices, sources = _load_comparators(workspace, zone, timezone, source_day, expected, rolling=rolling)
        if history_from_delivery is not None:
            for source in (*sources.values(), *rolling["sources"].values()):
                source.update(source_delivery_day=source_day, delivery_day=day)
        prices["model"], sources["model"] = (model["model"] if model is not None else None), model_source
        aligned = {key: value.reindex(expected) if value is not None else pd.Series(np.nan, index=expected)
                   for key, value in prices.items()}
        coverage = {key: int(value.notna().sum()) for key, value in aligned.items()}
        # The envelope belongs to Model, not to two additional forecast sources.
        for key in ("model_p10", "model_p90"):
            aligned[key] = model[key] if model is not None else pd.Series(np.nan, index=expected)
        invalid = [key for key, source in sources.items() if source["status"] == "invalid"]
        complete = all(count == len(expected) for count in coverage.values())
        status = "complete" if complete else "partial" if any(coverage.values()) else "invalid" if invalid else "unavailable"
        messages = []
        if coverage["model"] == 0:
            messages.append("Model unavailable" if not invalid or "model" not in invalid else "Model source rejected")
        if coverage["storm"] < len(expected):
            messages.append("Storm unavailable" if coverage["storm"] == 0 else "Storm incomplete")
        if coverage["observed"] == 0:
            messages.append(
                "Realized prices are not validated: sources disagree. Forecasts remain available; today's scores are not calculated."
                if sources["observed"].get("current_delivery_actual_reason") == "post_auction_source_rejected"
                else "Observed prices unavailable"
            )
        if invalid:
            messages.append("Invalid source: " + ", ".join(invalid))
        rows = [{"timestamp_utc": stamp.isoformat(), "local_label": stamp.tz_convert(timezone).strftime("%H:%M"),
                 **{key: float(value.iloc[number]) if pd.notna(value.iloc[number]) else None
                    for key, value in aligned.items()}}
                for number, stamp in enumerate(expected)]
        zones.append({"zone": zone, "name": name, "timezone": timezone, "status": status,
                      "message": "; ".join(messages), "rows": rows, "sources": sources,
                      "coverage": coverage, "expected_hours": len(expected),
                      "rolling_history": _rolling_payload(rolling, day=day, timezone=timezone)})
    return {"delivery_day": day,
            **({"history_from_delivery": source_day} if history_from_delivery is not None else {}),
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "has_data": any(any(zone["coverage"].values()) for zone in zones), "zones": zones}


__all__ = ["CWE_ZONES", "MODEL_COLUMN", "load_model_storm_payload"]
