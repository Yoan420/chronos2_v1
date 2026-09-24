"""Fresh, report-only Saturn VPS history for the CWE Model/Storm report.

The public entry point returns a deep copy of the supplied payload.  It never
uses an older snapshot as a fallback and writes evidence only below
``runs/reports/model_storm/vps_snapshots``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import date, timedelta
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd

from chronos2_hourly.atomic_directory import AtomicDirectoryStaging
from chronos2_modular.common import load_yaml
from chronos2_modular.saturn import create_saturn_client


_CONFIG_NAME = "chronos2_hourly_fr_residual_v1.yaml"
_SNAPSHOT_RELATIVE_ROOT = Path("runs/reports/model_storm/vps_snapshots")
_RAW_NAME = "vps_hourly_raw.parquet"
_AUDIT_NAME = "vps_audit.json"
_CHECKSUM_NAME = "artifact_checksums.json"
_MAX_WORKERS = 2
_REQUEST_TIMEOUT_SECONDS = (8, 60)
_SUPPORTED_ZONES = {"BE", "DE", "FR", "NL"}
_ZONE_TIMEZONES = {
    "BE": "Europe/Brussels",
    "DE": "Europe/Berlin",
    "FR": "Europe/Paris",
    "NL": "Europe/Amsterdam",
}


class _InvalidVpsSource(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_snapshot(directory: Path, *, raw_sha: str, audit_sha: str) -> None:
    for name, expected in ((_RAW_NAME, raw_sha), (_AUDIT_NAME, audit_sha)):
        if _sha256(directory / name) != expected:
            raise ValueError("Published VPS snapshot failed checksum verification.")


def _utc_text(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _delivery_day(payload: Mapping[str, Any]) -> date:
    raw = payload.get("delivery_day")
    try:
        value = pd.Timestamp(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("delivery_day must be a local calendar date (YYYY-MM-DD).") from exc
    if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
        raise ValueError("delivery_day must be a local calendar date (YYYY-MM-DD).")
    return value.date()


def _window(day: date, timezone: str) -> pd.DatetimeIndex:
    try:
        start = pd.Timestamp(day - timedelta(days=364), tz=timezone)
        end = pd.Timestamp(day + timedelta(days=1), tz=timezone)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid report timezone for VPS history: {timezone!r}.") from exc
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def _series_name(zone: str) -> str:
    return f"power.vps.{zone.lower()}.euromwh.h.da.pnl.storm"


def _safe_failure(category: str) -> dict[str, str]:
    messages = {
        "client_unavailable": "Fresh Saturn VPS extraction could not be started.",
        "timeout": "Fresh Saturn VPS extraction timed out.",
        "network_error": "Fresh Saturn VPS extraction failed at the source boundary.",
        "source_error": "Fresh Saturn VPS extraction was unavailable.",
        "tls_error": "The Saturn TLS connection failed; certificate verification was not disabled.",
    }
    return {"error_type": category, "error": messages[category]}


def _failure_category(error: Exception) -> str:
    name = type(error).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "ssl" in name or "tls" in name:
        return "tls_error"
    if any(token in name for token in ("connection", "request", "http", "network")):
        return "network_error"
    return "source_error"


def _base_source(
    *, zone: str, timezone: str, expected: pd.DatetimeIndex,
    extracted_at_utc: pd.Timestamp,
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "series": _series_name(zone),
        "zone": zone,
        "timezone": timezone,
        "extracted_at_utc": _utc_text(extracted_at_utc),
        "last_available_at_utc": None,
        "window_start_utc": _utc_text(expected[0]),
        "window_end_utc": _utc_text(expected[-1]),
        "window_days": 365,
        "expected_physical_hours": int(len(expected)),
        "returned_physical_hours": 0,
        "available_hours": 0,
        "nan_hours": 0,
        "missing_physical_hours": int(len(expected)),
        "nocache": True,
        "live": False,
        "keep_nans": True,
        "fill_policy": "none",
        "used_for_prediction": False,
    }


def _normalize(raw: Any, expected: pd.DatetimeIndex) -> pd.Series:
    if not isinstance(raw, pd.Series):
        raise _InvalidVpsSource("series_required")
    if len(raw) == 0:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name="pnl")

    try:
        index = pd.DatetimeIndex(raw.index)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _InvalidVpsSource("timestamp_parse_failed") from exc
    if index.hasnans:
        raise _InvalidVpsSource("timestamp_missing")
    if index.tz is None:
        raise _InvalidVpsSource("timezone_required")
    index = index.tz_convert("UTC")

    # The server query is bounded, but an out-of-window row is not evidence for
    # this report.  Exclude it before validating values or duplicate support.
    inside = (index >= expected[0]) & (index <= expected[-1])
    index = index[inside]
    selected = raw.iloc[np.flatnonzero(inside)]
    if len(index) == 0:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name="pnl")
    if index.has_duplicates:
        raise _InvalidVpsSource("duplicate_physical_hour")
    if not index.is_monotonic_increasing:
        raise _InvalidVpsSource("timestamps_not_sorted")
    if not index.equals(index.floor("h")):
        raise _InvalidVpsSource("timestamps_not_hourly")
    if selected.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise _InvalidVpsSource("pnl_not_numeric")
    try:
        values = pd.to_numeric(selected, errors="raise").to_numpy(
            dtype=float, na_value=np.nan,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _InvalidVpsSource("pnl_not_numeric") from exc
    if np.isinf(values).any():
        raise _InvalidVpsSource("infinite_pnl")
    return pd.Series(values, index=index, name="pnl", dtype=float)


def _fetch_one(
    client: Any, *, zone: str, timezone: str, expected: pd.DatetimeIndex,
    extracted_at_utc: pd.Timestamp,
) -> tuple[pd.Series | None, dict[str, Any]]:
    source = _base_source(
        zone=zone, timezone=timezone, expected=expected,
        extracted_at_utc=extracted_at_utc,
    )
    try:
        raw = client.get(
            source["series"],
            from_value_date=expected[0],
            to_value_date=expected[-1],
            nocache=True,
            live=False,
            _keep_nans=True,
        )
    except Exception as error:  # source failures must not block other zones
        source.update(_safe_failure(_failure_category(error)))
        return None, source
    # tshistory returns None for a missing series and some HTTP failures can be
    # returned as response-like objects instead of being raised by the SDK.
    if raw is None or (
        not isinstance(raw, pd.Series) and hasattr(raw, "status_code")
    ):
        source.update(_safe_failure("source_error"))
        return None, source
    try:
        values = _normalize(raw, expected)
    except _InvalidVpsSource as error:
        source.update({
            "status": "invalid",
            "error_type": "invalid_source_data",
            "error_code": error.code,
            "error": "Fresh Saturn VPS data violated the physical-hour contract.",
        })
        return None, source

    if values.empty:
        source.update(_safe_failure("source_error"))
        return values, source
    available = values.notna()
    missing = expected.difference(values.index)
    complete = values.index.equals(expected) and bool(available.all())
    source.update({
        "status": "complete" if complete else "partial",
        "last_available_at_utc": _utc_text(values.index[available][-1]) if available.any() else None,
        "returned_physical_hours": int(len(values)),
        "available_hours": int(available.sum()),
        "nan_hours": int(values.isna().sum()),
        "missing_physical_hours": int(len(missing)),
        "coverage": float(available.sum() / len(expected)),
    })
    return values, source


def _owned_client_settings(project_root: Path) -> tuple[str, str]:
    config = load_yaml(project_root / _CONFIG_NAME)
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Configuration data mapping is unavailable.")
    url = data.get("saturn_url")
    author = os.getenv("SATURN_AUTHOR") or data.get("saturn_author")
    if not isinstance(url, str) or not isinstance(author, str):
        raise ValueError("Saturn report-only client configuration is unavailable.")
    return url, author


def _create_owned_client(settings: tuple[str, str]) -> Any:
    client = create_saturn_client(*settings)
    session = getattr(client, "session", None)
    if session is None or not callable(getattr(session, "request", None)):
        raise RuntimeError("Saturn client does not expose a configurable request deadline.")
    session.request = partial(session.request, timeout=_REQUEST_TIMEOUT_SECONDS)
    return client


def _empty_raw_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "zone": pd.Series(dtype="string"),
        "series": pd.Series(dtype="string"),
        "timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "pnl": pd.Series(dtype="float64"),
    })


def _publish_snapshot(
    *, project_root: Path, delivery_day: date, extracted_at_utc: pd.Timestamp,
    results: Mapping[str, tuple[pd.Series | None, dict[str, Any]]],
    configuration_sha256: str | None,
    endpoint_sha256: str | None,
) -> dict[str, Any]:
    parent = (project_root / _SNAPSHOT_RELATIVE_ROOT).resolve()
    expected_parent = project_root.joinpath(*_SNAPSHOT_RELATIVE_ROOT.parts).resolve()
    if parent != expected_parent or not parent.is_relative_to(project_root):
        raise ValueError("VPS snapshot directory escapes the project root.")
    unique = extracted_at_utc.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    final = parent / unique
    pieces = []
    for zone in sorted(results):
        values, source = results[zone]
        if values is None or values.empty:
            continue
        pieces.append(pd.DataFrame({
            "zone": zone,
            "series": source["series"],
            "timestamp_utc": values.index,
            "pnl": values.to_numpy(dtype=float),
        }))
    raw = pd.concat(pieces, ignore_index=True) if pieces else _empty_raw_frame()
    raw = raw[["zone", "series", "timestamp_utc", "pnl"]]

    with AtomicDirectoryStaging(parent, prefix=".vps_refresh_") as publication:
        staging = publication.path
        assert staging is not None
        raw_path = staging / _RAW_NAME
        raw.to_parquet(raw_path, index=False)
        raw_sha = _sha256(raw_path)
        audit = {
            "schema_version": 1,
            "mode": "fresh_report_only_vps_extraction",
            "delivery_day_local": delivery_day.isoformat(),
            "window_days": 365,
            "extracted_at_utc": _utc_text(extracted_at_utc),
            "snapshot_id": unique,
            "snapshot_directory": (_SNAPSHOT_RELATIVE_ROOT / unique).as_posix(),
            "configuration_sha256": configuration_sha256,
            "saturn_endpoint_sha256": endpoint_sha256,
            "query": {
                "nocache": True, "live": False, "keep_nans": True,
                "maximum_concurrency": _MAX_WORKERS,
                "owned_client_request_timeout_seconds": list(_REQUEST_TIMEOUT_SECONDS),
            },
            "raw_artifact": {
                "path": _RAW_NAME, "sha256": raw_sha, "rows": int(len(raw)),
                "columns": list(raw.columns),
            },
            "zones": {
                zone: deepcopy(results[zone][1]) for zone in sorted(results)
            },
            "no_stale_fallback": True,
            "missing_values_filled": False,
            "used_for_prediction": False,
            "forecast_or_model_data_modified": False,
        }
        audit_path = staging / _AUDIT_NAME
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        audit_sha = _sha256(audit_path)
        checksums = {
            "algorithm": "sha256",
            "artifacts": [
                {"path": _RAW_NAME, "sha256": raw_sha, "size_bytes": raw_path.stat().st_size},
                {"path": _AUDIT_NAME, "sha256": audit_sha, "size_bytes": audit_path.stat().st_size},
            ],
        }
        checksum_path = staging / _CHECKSUM_NAME
        checksum_path.write_text(
            json.dumps(checksums, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        checksum_sha = _sha256(checksum_path)
        publication.publish(final)
    _verify_snapshot(final, raw_sha=raw_sha, audit_sha=audit_sha)
    relative = _SNAPSHOT_RELATIVE_ROOT / unique
    return {
        "status": "published",
        "snapshot_id": unique,
        "directory": relative.as_posix(),
        "raw_artifact_path": (relative / _RAW_NAME).as_posix(),
        "raw_artifact_sha256": raw_sha,
        "audit_path": (relative / _AUDIT_NAME).as_posix(),
        "audit_sha256": audit_sha,
        "checksum_manifest_path": (relative / _CHECKSUM_NAME).as_posix(),
        "checksum_manifest_sha256": checksum_sha,
        "used_for_prediction": False,
    }


def refresh_vps_payload(
    payload: Mapping[str, Any], project_root: str | Path, *, client: Any = None,
) -> dict[str, Any]:
    """Return a deep-copied payload enriched with one fresh VPS extraction.

    Each zone is independent: a network/source failure produces an empty
    ``rows`` list and an ``unavailable`` source, never a stale-data fallback.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping.")
    enriched = deepcopy(dict(payload))
    zones = enriched.get("zones")
    if not isinstance(zones, list):
        raise ValueError("payload.zones must be a list.")
    day = _delivery_day(enriched)
    root = Path(project_root).resolve()
    if root == Path(root.anchor) or not root.is_dir():
        raise ValueError("project_root must be an existing project directory.")
    extracted = pd.Timestamp.now(tz="UTC")

    specifications: list[tuple[str, str, pd.DatetimeIndex]] = []
    seen: set[str] = set()
    for item in zones:
        if not isinstance(item, dict):
            raise ValueError("Every payload zone must be a mapping.")
        zone = str(item.get("zone", "")).upper()
        timezone = item.get("timezone")
        if (zone not in _SUPPORTED_ZONES or zone in seen
                or timezone != _ZONE_TIMEZONES.get(zone)):
            raise ValueError(
                "CWE VPS zones must be unique BE/DE/FR/NL entries with their canonical timezone."
            )
        seen.add(zone)
        specifications.append((zone, timezone, _window(day, timezone)))

    results: dict[str, tuple[pd.Series | None, dict[str, Any]]] = {}
    owned_clients: list[Any] = []
    configuration_sha = endpoint_sha = None
    try:
        if client is not None:
            # An injected test/caller client has no declared thread-safety
            # contract. Keep it sequential while preserving the <=2 bound.
            for zone, timezone, expected in specifications:
                results[zone] = _fetch_one(
                    client, zone=zone, timezone=timezone, expected=expected,
                    extracted_at_utc=extracted,
                )
        else:
            try:
                settings = _owned_client_settings(root)
                configuration_sha = _sha256(root / _CONFIG_NAME)
                endpoint_sha = hashlib.sha256(settings[0].encode("utf-8")).hexdigest()
            except Exception:
                settings = None
            if settings is None:
                for zone, timezone, expected in specifications:
                    source = _base_source(
                        zone=zone, timezone=timezone, expected=expected,
                        extracted_at_utc=extracted,
                    )
                    source.update(_safe_failure("client_unavailable"))
                    results[zone] = (None, source)
            else:
                local = threading.local()
                clients_lock = threading.Lock()

                def fetch_with_thread_client(
                    *, zone: str, timezone: str, expected: pd.DatetimeIndex,
                ) -> tuple[pd.Series | None, dict[str, Any]]:
                    if not hasattr(local, "client"):
                        local.client = _create_owned_client(settings)
                        with clients_lock:
                            owned_clients.append(local.client)
                    return _fetch_one(
                        local.client, zone=zone, timezone=timezone,
                        expected=expected, extracted_at_utc=extracted,
                    )

                with ThreadPoolExecutor(
                    max_workers=min(_MAX_WORKERS, max(1, len(specifications)))
                ) as executor:
                    pending = {
                        executor.submit(
                            fetch_with_thread_client, zone=zone,
                            timezone=timezone, expected=expected,
                        ): zone
                        for zone, timezone, expected in specifications
                    }
                    for future in as_completed(pending):
                        zone = pending[future]
                        try:
                            results[zone] = future.result()
                        except Exception as error:
                            timezone, expected = next(
                                (tz, idx) for code, tz, idx in specifications if code == zone
                            )
                            source = _base_source(
                                zone=zone, timezone=timezone, expected=expected,
                                extracted_at_utc=extracted,
                            )
                            source.update(_safe_failure(_failure_category(error)))
                            results[zone] = (None, source)
    finally:
        for connector in owned_clients:
            session = getattr(connector, "session", None)
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # Source-session cleanup cannot invalidate a fully fetched,
                    # immutable report snapshot.
                    pass

    snapshot = _publish_snapshot(
        project_root=root, delivery_day=day, extracted_at_utc=extracted,
        results=results, configuration_sha256=configuration_sha, endpoint_sha256=endpoint_sha,
    )
    for item in zones:
        zone = str(item["zone"]).upper()
        values, original_source = results[zone]
        source = deepcopy(original_source)
        source.update({
            "snapshot_id": snapshot["snapshot_id"],
            "raw_artifact_sha256": snapshot["raw_artifact_sha256"],
            "audit_sha256": snapshot["audit_sha256"],
        })
        rows = [] if values is None else [
            {"timestamp_utc": _utc_text(stamp),
             "pnl": None if pd.isna(value) else float(value)}
            for stamp, value in values.items()
        ]
        item["vps_history"] = {"source": source, "rows": rows}
    enriched["vps_snapshot"] = snapshot
    return enriched


__all__ = ["refresh_vps_payload"]
