#!/usr/bin/env python
"""Materialise causal France weather forecasts from Open-Meteo Previous Runs.

The day-ahead decision contract in this project is D-1 08:00 Europe/Paris.
Consequently this utility deliberately requests only fixed-lead ``previous_dayN``
series with N >= 2.  ``previous_day1`` is not causal for the late hours of D:
its 24-hour reference time can be later than D-1 08:00.

The output is an exact hourly UTC table.  Local delivery days retain their
physical 23/24/25 hours around DST changes.  Missing API timestamps or values
are fatal: this program never resamples, interpolates, forward-fills, or
back-fills weather data.

Open-Meteo's Previous Runs API exposes a fixed lead, not the identifier of an
individual model run.  The manifest therefore records both the explicit model
request and the per-value fixed-lead reference time, while stating that an
individual run id is unavailable.  If an exact run id is required, use a
separately audited Single Runs materialisation instead.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

import numpy as np
import pandas as pd


API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DOCUMENTATION_URL = "https://open-meteo.com/en/docs/previous-runs-api"
LICENSE_URL = "https://open-meteo.com/en/license"
TERMS_URL = "https://open-meteo.com/en/terms"
DEFAULT_MODEL = "ecmwf_ifs025"
DEFAULT_TIMEZONE = "Europe/Paris"
DEFAULT_CUTOFF = "08:00"
DEFAULT_LEAD_DAYS = 2


@dataclass(frozen=True)
class Site:
    slug: str
    label: str
    latitude: float
    longitude: float


# A deliberately small, geographically spread panel.  No fitted or
# holdout-derived weights are embedded: aggregate columns are simple means and
# dispersions, and every point value is retained for downstream calibration.
FRANCE_PANEL: tuple[Site, ...] = (
    Site("lille", "Lille", 50.6292, 3.0573),
    Site("paris", "Paris", 48.8566, 2.3522),
    Site("rennes", "Rennes", 48.1173, -1.6778),
    Site("strasbourg", "Strasbourg", 48.5734, 7.7521),
    Site("lyon", "Lyon", 45.7640, 4.8357),
    Site("bordeaux", "Bordeaux", 44.8378, -0.5792),
    Site("toulouse", "Toulouse", 43.6047, 1.4442),
    Site("marseille", "Marseille", 43.2965, 5.3698),
)


@dataclass(frozen=True)
class WeatherVariable:
    api_name: str
    output_stem: str
    expected_unit: str
    description: str


WEATHER_VARIABLES: tuple[WeatherVariable, ...] = (
    WeatherVariable(
        "temperature_2m",
        "openmeteo_temperature_2m_c",
        "°C",
        "Air temperature at 2 metres.",
    ),
    WeatherVariable(
        "wind_speed_100m",
        "openmeteo_wind_speed_100m_kmh",
        "km/h",
        "Wind speed at 100 metres.",
    ),
    WeatherVariable(
        "shortwave_radiation",
        "openmeteo_shortwave_radiation_wm2",
        "W/m²",
        "Global horizontal shortwave radiation, preceding-hour mean.",
    ),
)


class WeatherMaterialisationError(RuntimeError):
    """Raised when the causal or exact-timeline contract cannot be proven."""


def _parse_day(value: str | date | pd.Timestamp, *, name: str) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if day.tzinfo is not None:
        raise ValueError(f"{name} must be a naive local calendar date")
    return day.normalize()


def _parse_clock(value: str) -> pd.Timedelta:
    parts = str(value).strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError("cutoff-time must use HH:MM or HH:MM:SS")
    try:
        hour, minute = int(parts[0]), int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ValueError("cutoff-time must use numeric HH:MM[:SS]") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError("cutoff-time is outside the civil-day clock")
    return pd.Timedelta(hours=hour, minutes=minute, seconds=second)


def physical_delivery_index(
    start_day: str | date | pd.Timestamp,
    end_day: str | date | pd.Timestamp,
    *,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DatetimeIndex:
    """Return the exact UTC hours of inclusive local delivery days."""

    start = _parse_day(start_day, name="start-day")
    end = _parse_day(end_day, name="end-day")
    if end < start:
        raise ValueError("end-day must be on or after start-day")
    local_start = start.tz_localize(timezone)
    local_stop = (end + pd.Timedelta(days=1)).tz_localize(timezone)
    result = pd.date_range(
        local_start.tz_convert("UTC"),
        local_stop.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise WeatherMaterialisationError("delivery UTC timeline is not unique/sorted")
    if len(result) > 1 and not bool(
        np.all((result[1:] - result[:-1]) == pd.Timedelta(hours=1))
    ):
        raise WeatherMaterialisationError("delivery UTC timeline is not contiguous hourly")
    return result


def civil_cutoff(
    delivery_day: str | date | pd.Timestamp,
    *,
    timezone: str = DEFAULT_TIMEZONE,
    cutoff_time: str = DEFAULT_CUTOFF,
) -> pd.Timestamp:
    """Build D-1 cutoff in civil time before timezone localisation."""

    day = _parse_day(delivery_day, name="delivery-day")
    naive = day - pd.Timedelta(days=1) + _parse_clock(cutoff_time)
    return naive.tz_localize(timezone)


def timing_contract(
    index: pd.DatetimeIndex,
    *,
    lead_days: int = DEFAULT_LEAD_DAYS,
    timezone: str = DEFAULT_TIMEZONE,
    cutoff_time: str = DEFAULT_CUTOFF,
) -> pd.DataFrame:
    """Return fixed-lead references/cutoffs and prove strict causality."""

    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise TypeError("index must be an explicitly timezone-aware DatetimeIndex")
    utc_index = index.tz_convert("UTC")
    if not isinstance(lead_days, int) or isinstance(lead_days, bool) or lead_days < 1:
        raise ValueError("lead-days must be a positive integer")
    local_days = utc_index.tz_convert(timezone).normalize().tz_localize(None)
    cutoff_values = pd.DatetimeIndex(
        [
            civil_cutoff(day, timezone=timezone, cutoff_time=cutoff_time)
            .tz_convert("UTC")
            for day in local_days
        ]
    )
    reference = utc_index - pd.Timedelta(days=lead_days)
    causal = reference < cutoff_values
    if not bool(np.all(causal)):
        positions = np.flatnonzero(~causal)[:5]
        examples = [
            {
                "value_time_utc": utc_index[position].isoformat(),
                "fixed_lead_reference_time_utc": reference[position].isoformat(),
                "cutoff_time_utc": cutoff_values[position].isoformat(),
            }
            for position in positions
        ]
        raise WeatherMaterialisationError(
            "fixed lead is not strictly earlier than D-1 civil cutoff; "
            f"examples={examples}"
        )
    return pd.DataFrame(
        {
            "value_time_utc": utc_index,
            "weather_fixed_lead_reference_time_utc": reference,
            "weather_cutoff_time_utc": cutoff_values,
        }
    )


def _api_variable(variable: WeatherVariable, lead_days: int) -> str:
    return f"{variable.api_name}_previous_day{lead_days}"


def _request_params(
    *,
    sites: Sequence[Site],
    utc_start_date: date,
    utc_end_date: date,
    model: str,
    lead_days: int,
    api_key: str | None,
) -> dict[str, str]:
    params = {
        "latitude": ",".join(f"{site.latitude:.6f}" for site in sites),
        "longitude": ",".join(f"{site.longitude:.6f}" for site in sites),
        "start_date": utc_start_date.isoformat(),
        "end_date": utc_end_date.isoformat(),
        "timezone": "UTC",
        "timeformat": "unixtime",
        "models": model,
        "hourly": ",".join(
            _api_variable(variable, lead_days) for variable in WEATHER_VARIABLES
        ),
    }
    if api_key:
        params["apikey"] = api_key
    return params


def _public_request_params(params: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in params.items() if key != "apikey"}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ssl_context() -> ssl.SSLContext:
    # requests honours REQUESTS_CA_BUNDLE; urllib does not, so make the
    # corporate-CA behaviour explicit while avoiding a third-party dependency.
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if cafile:
        return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def fetch_response_bytes(
    api_url: str,
    params: Mapping[str, str],
    *,
    timeout_seconds: float,
    retries: int,
) -> bytes:
    """Fetch one response without ever logging a supplied API key."""

    url = f"{api_url}?{urlencode(dict(params))}"
    secret = params.get("apikey")

    def redact(value: object) -> str:
        text = str(value)
        return text.replace(secret, "<redacted>") if secret else text

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = Request(url, headers={"User-Agent": "chronos2-weather-materializer/1"})
        try:
            with urlopen(
                request,
                timeout=float(timeout_seconds),
                context=_ssl_context(),
            ) as response:
                status = int(getattr(response, "status", 200))
                body = response.read()
            if status != 200:
                raise WeatherMaterialisationError(f"Open-Meteo HTTP status {status}")
            return body
        except HTTPError as exc:
            try:
                detail = exc.read(500).decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            last_error = WeatherMaterialisationError(
                f"Open-Meteo HTTP {exc.code}: {redact(detail)}"
            )
        except (URLError, TimeoutError, OSError, ssl.SSLError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 8))
    raise WeatherMaterialisationError(
        f"Open-Meteo request failed after {retries} attempt(s): "
        f"{type(last_error).__name__}: {redact(last_error)}"
    ) from last_error


def _ordered_location_payloads(payload: Any, sites: Sequence[Site]) -> list[Mapping[str, Any]]:
    if len(sites) == 1 and isinstance(payload, Mapping):
        return [payload]
    if not isinstance(payload, list) or len(payload) != len(sites):
        raise WeatherMaterialisationError(
            f"API returned {type(payload).__name__} for {len(sites)} locations"
        )
    if all(isinstance(item, Mapping) and "location_id" in item for item in payload):
        by_id = {int(item["location_id"]): item for item in payload}
        if set(by_id) == set(range(len(sites))):
            return [by_id[position] for position in range(len(sites))]
    if not all(isinstance(item, Mapping) for item in payload):
        raise WeatherMaterialisationError("API location payload is not a mapping")
    return list(payload)


def parse_weather_payload(
    raw_payload: bytes | str | Any,
    *,
    sites: Sequence[Site],
    expected_index: pd.DatetimeIndex,
    lead_days: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Parse exact API hours; missing timestamps/values fail, never fill."""

    if isinstance(raw_payload, bytes):
        payload = json.loads(raw_payload.decode("utf-8"))
    elif isinstance(raw_payload, str):
        payload = json.loads(raw_payload)
    else:
        payload = raw_payload
    if isinstance(payload, Mapping) and payload.get("error"):
        raise WeatherMaterialisationError(
            f"Open-Meteo error: {payload.get('reason', 'unknown reason')}"
        )
    locations = _ordered_location_payloads(payload, sites)
    columns: dict[str, pd.Series] = {}
    grids: list[dict[str, Any]] = []
    expected = expected_index.tz_convert("UTC")
    for position, (site, location) in enumerate(zip(sites, locations, strict=True)):
        hourly = location.get("hourly")
        if not isinstance(hourly, Mapping) or "time" not in hourly:
            raise WeatherMaterialisationError(f"{site.slug}: missing hourly.time")
        time_values = hourly["time"]
        parsed_index = pd.to_datetime(time_values, unit="s", utc=True, errors="raise")
        if parsed_index.has_duplicates or not parsed_index.is_monotonic_increasing:
            raise WeatherMaterialisationError(
                f"{site.slug}: API timestamps are duplicated or unsorted"
            )
        missing_times = expected.difference(parsed_index)
        if len(missing_times):
            raise WeatherMaterialisationError(
                f"{site.slug}: {len(missing_times)} exact UTC hour(s) missing; "
                f"first={missing_times[0]}"
            )
        for variable in WEATHER_VARIABLES:
            api_column = _api_variable(variable, lead_days)
            values = hourly.get(api_column)
            if not isinstance(values, list) or len(values) != len(parsed_index):
                raise WeatherMaterialisationError(
                    f"{site.slug}: invalid/missing array {api_column}"
                )
            numeric = pd.Series(
                pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float),
                index=parsed_index,
                name=f"{variable.output_stem}__{site.slug}",
            ).loc[expected]
            if not bool(np.isfinite(numeric.to_numpy(dtype=float)).all()):
                bad = numeric.index[~np.isfinite(numeric.to_numpy(dtype=float))]
                raise WeatherMaterialisationError(
                    f"{site.slug}: non-finite {api_column} at {bad[0]}"
                )
            columns[numeric.name] = numeric
        grids.append(
            {
                "site_position": position,
                "site_slug": site.slug,
                "requested_latitude": site.latitude,
                "requested_longitude": site.longitude,
                "grid_latitude": location.get("latitude"),
                "grid_longitude": location.get("longitude"),
                "grid_elevation_m": location.get("elevation"),
                "timezone": location.get("timezone"),
                "utc_offset_seconds": location.get("utc_offset_seconds"),
                "generationtime_ms": location.get("generationtime_ms"),
                "hourly_units_as_returned": location.get("hourly_units"),
            }
        )
    frame = pd.DataFrame(columns, index=expected)
    for variable in WEATHER_VARIABLES:
        point_columns = [
            f"{variable.output_stem}__{site.slug}" for site in sites
        ]
        values = frame.loc[:, point_columns]
        frame[f"{variable.output_stem}__panel_mean"] = values.mean(axis=1)
        frame[f"{variable.output_stem}__panel_std"] = values.std(axis=1, ddof=0)
        frame[f"{variable.output_stem}__panel_min"] = values.min(axis=1)
        frame[f"{variable.output_stem}__panel_max"] = values.max(axis=1)
    if frame.isna().any().any():
        raise WeatherMaterialisationError("parsed weather frame contains NaN")
    return frame, grids


def _local_day_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int,
) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(days=1)


def _atomic_write_bytes(path: Path, content: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    reloaded = pd.read_parquet(temporary)
    if len(reloaded) != len(frame) or list(reloaded.columns) != list(frame.columns):
        raise WeatherMaterialisationError("Parquet round-trip validation failed")
    temporary.replace(path)


def materialize(
    *,
    start_day: str | date | pd.Timestamp,
    end_day: str | date | pd.Timestamp,
    output: Path,
    manifest_path: Path,
    manifest_sha_path: Path,
    raw_dir: Path | None,
    sites: Sequence[Site] = FRANCE_PANEL,
    timezone: str = DEFAULT_TIMEZONE,
    cutoff_time: str = DEFAULT_CUTOFF,
    lead_days: int = DEFAULT_LEAD_DAYS,
    model: str = DEFAULT_MODEL,
    api_url: str = API_URL,
    api_key: str | None = None,
    chunk_days: int = 14,
    timeout_seconds: float = 90.0,
    retries: int = 3,
    overwrite: bool = False,
) -> dict[str, Any]:
    start = _parse_day(start_day, name="start-day")
    end = _parse_day(end_day, name="end-day")
    if end < start:
        raise ValueError("end-day must be on or after start-day")
    if lead_days < 2:
        # This strong guard is intentional even though timing_contract would
        # find the individual offending hours.
        raise WeatherMaterialisationError(
            "lead-days=1 is unsafe for a D-1 08:00 day-ahead cutoff; use >= 2"
        )
    if not sites:
        raise ValueError("at least one weather site is required")
    if len({site.slug for site in sites}) != len(sites):
        raise ValueError("weather site slugs must be unique")
    if chunk_days < 1:
        raise ValueError("chunk-days must be positive")
    if retries < 1:
        raise ValueError("retries must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    if not str(model).strip():
        raise ValueError("model must be non-empty")

    expected_all = physical_delivery_index(start, end, timezone=timezone)
    timing = timing_contract(
        expected_all,
        lead_days=lead_days,
        timezone=timezone,
        cutoff_time=cutoff_time,
    ).set_index("value_time_utc")
    feature_chunks: list[pd.DataFrame] = []
    request_manifest: list[dict[str, Any]] = []
    downloaded_at = pd.Timestamp.now(tz="UTC")

    for number, (chunk_start, chunk_end) in enumerate(
        _local_day_chunks(start, end, chunk_days), start=1
    ):
        chunk_index = physical_delivery_index(
            chunk_start, chunk_end, timezone=timezone
        )
        params = _request_params(
            sites=sites,
            utc_start_date=chunk_index.min().date(),
            utc_end_date=chunk_index.max().date(),
            model=model,
            lead_days=lead_days,
            api_key=api_key,
        )
        public_params = _public_request_params(params)
        raw = fetch_response_bytes(
            api_url,
            params,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        response_sha = hashlib.sha256(raw).hexdigest()
        weather, grids = parse_weather_payload(
            raw,
            sites=sites,
            expected_index=chunk_index,
            lead_days=lead_days,
        )
        feature_chunks.append(weather)
        raw_relative: str | None = None
        if raw_dir is not None:
            raw_path = raw_dir / (
                f"chunk_{number:04d}_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.json"
            )
            _atomic_write_bytes(raw_path, raw, overwrite=overwrite)
            raw_relative = str(raw_path.resolve())
        request_manifest.append(
            {
                "chunk_number": number,
                "local_start_day": str(chunk_start.date()),
                "local_end_day": str(chunk_end.date()),
                "request_parameters_without_api_key": public_params,
                "request_fingerprint_sha256": _canonical_sha256(
                    {"api_url": api_url, "params": public_params}
                ),
                "response_sha256": response_sha,
                "response_bytes": len(raw),
                "raw_response_path": raw_relative,
                "resolved_grids": grids,
            }
        )
        print(
            f"[openmeteo] chunk {number}: {chunk_start.date()}..{chunk_end.date()} "
            f"rows={len(weather)} sha256={response_sha[:12]}",
            flush=True,
        )

    features = pd.concat(feature_chunks, axis=0)
    if features.index.has_duplicates or not features.index.equals(expected_all):
        raise WeatherMaterialisationError(
            "concatenated feature timeline differs from exact delivery timeline"
        )
    # pandas drops the index name when concatenating a named and unnamed
    # DatetimeIndex.  Restore the canonical schema explicitly; never let the
    # delivery timestamp silently become a generic ``index`` column.
    features.index.name = "value_time_utc"
    output_frame = pd.concat([timing, features], axis=1)
    output_frame.index.name = "value_time_utc"
    output_frame = output_frame.reset_index()
    _atomic_write_parquet(output, output_frame, overwrite=overwrite)
    output_sha = _file_sha256(output)

    local_counts = pd.Series(
        1,
        index=expected_all.tz_convert(timezone),
    ).groupby(lambda timestamp: timestamp.date()).sum()
    causal_margin_hours = (
        timing["weather_cutoff_time_utc"]
        - timing["weather_fixed_lead_reference_time_utc"]
    ) / pd.Timedelta(hours=1)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": downloaded_at.isoformat(),
        "provider": "Open-Meteo",
        "product": "Previous Model Runs API",
        "api_url": api_url,
        "documentation_url": DOCUMENTATION_URL,
        "output_sha256": output_sha,
        "license": {
            "data_license": "CC BY 4.0 (attribution required)",
            "license_url": LICENSE_URL,
            "terms_url": TERMS_URL,
            "free_endpoint_notice": (
                "The public free API is limited to non-commercial evaluation; "
                "confirm an appropriate Open-Meteo commercial plan or self-hosted "
                "rights before production use."
            ),
        },
        "causality_contract": {
            "delivery_timezone": timezone,
            "decision_cutoff": f"D-1 {cutoff_time} civil time",
            "model_requested": model,
            "fixed_lead_days": lead_days,
            "fixed_lead_hours": lead_days * 24,
            "api_suffix": f"_previous_day{lead_days}",
            "reference_time_rule": (
                f"weather_fixed_lead_reference_time_utc = value_time_utc - "
                f"{lead_days * 24} hours"
            ),
            "strictly_before_cutoff_verified_for_every_row": True,
            "minimum_reference_to_cutoff_margin_hours": float(
                causal_margin_hours.min()
            ),
            "maximum_reference_to_cutoff_margin_hours": float(
                causal_margin_hours.max()
            ),
            "individual_run_id_exposed_by_api": False,
            "run_identity_note": (
                "Previous Runs supplies a fixed-lead series but no individual run "
                "identifier. The explicit model and lead are pinned in each request."
            ),
            "storm_used": False,
            "actual_weather_used": False,
            "target_or_price_used": False,
        },
        "causality": {
            "classification": "strict_fixed_lead_forecast",
            "strict_pit_eligible": True,
            "available_by_d_minus_1_08_europe_paris": True,
            "cutoff_violations": 0,
            "target_or_price_used": False,
            "storm_used": False,
        },
        "timeline_contract": {
            "local_start_day": str(start.date()),
            "local_end_day": str(end.date()),
            "rows": len(output_frame),
            "utc_start": expected_all.min().isoformat(),
            "utc_end": expected_all.max().isoformat(),
            "hourly_utc_contiguous": True,
            "interpolation": "none",
            "resampling": "none",
            "fill": "none",
            "physical_local_day_hour_counts": {
                str(int(key)): int((local_counts == key).sum())
                for key in sorted(local_counts.unique())
            },
        },
        "panel": [asdict(site) for site in sites],
        "panel_aggregation": (
            "Unweighted mean/std/min/max; all site values retained. No weights were "
            "fitted on calibration or holdout data."
        ),
        "variables": [
            {
                **asdict(variable),
                "api_requested_name": _api_variable(variable, lead_days),
            }
            for variable in WEATHER_VARIABLES
        ],
        "api_key_supplied": api_key is not None,
        "raw_responses_preserved": raw_dir is not None,
        "requests": request_manifest,
        "output": {
            "path": str(output.resolve()),
            "format": "parquet",
            "sha256": output_sha,
            "rows": len(output_frame),
            "columns": list(output_frame.columns),
        },
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(manifest_path, manifest_bytes, overwrite=overwrite)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    sha_content = f"{manifest_sha}  {manifest_path.name}\n".encode("ascii")
    _atomic_write_bytes(manifest_sha_path, sha_content, overwrite=overwrite)
    manifest["manifest_sha256"] = manifest_sha
    manifest["manifest_sha256_path"] = str(manifest_sha_path.resolve())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-day", required=True, help="Inclusive local day YYYY-MM-DD")
    parser.add_argument("--end-day", required=True, help="Inclusive local day YYYY-MM-DD")
    parser.add_argument("--output", required=True, help="Output Parquet path")
    parser.add_argument("--manifest", help="Default: <output stem>.manifest.json")
    parser.add_argument("--manifest-sha", help="Default: <manifest>.sha256")
    parser.add_argument("--raw-dir", help="Optional directory retaining exact JSON responses")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--cutoff-time", default=DEFAULT_CUTOFF)
    parser.add_argument("--lead-days", type=int, choices=range(2, 8), default=DEFAULT_LEAD_DAYS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument(
        "--api-key-env",
        help="Environment variable holding an API key; its value is never manifested",
    )
    parser.add_argument("--chunk-days", type=int, default=14)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-large-range",
        action="store_true",
        help="Required when requesting more than 31 local days",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print request/row estimates without network or file writes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = _parse_day(args.start_day, name="start-day")
    end = _parse_day(args.end_day, name="end-day")
    local_days = int((end - start) / pd.Timedelta(days=1)) + 1
    if local_days > 31 and not args.allow_large_range:
        raise SystemExit(
            f"Refusing {local_days} days without --allow-large-range. "
            "Use --dry-run first and confirm API licence/capacity."
        )
    expected_rows = len(
        physical_delivery_index(start, end, timezone=args.timezone)
    )
    request_count = math.ceil(local_days / max(1, int(args.chunk_days)))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "network_calls": 0,
                    "planned_api_requests": request_count,
                    "local_days": local_days,
                    "expected_physical_hours": expected_rows,
                    "sites": len(FRANCE_PANEL),
                    "variables": len(WEATHER_VARIABLES),
                    "model": args.model,
                    "lead_days": args.lead_days,
                    "cutoff": f"D-1 {args.cutoff_time} {args.timezone}",
                },
                indent=2,
            )
        )
        return 0

    output = Path(args.output).expanduser().resolve()
    manifest = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else output.with_suffix(".manifest.json")
    )
    manifest_sha = (
        Path(args.manifest_sha).expanduser().resolve()
        if args.manifest_sha
        else manifest.with_suffix(manifest.suffix + ".sha256")
    )
    raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else None
    api_key: str | None = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"Environment variable {args.api_key_env!r} is empty/missing")
    result = materialize(
        start_day=start,
        end_day=end,
        output=output,
        manifest_path=manifest,
        manifest_sha_path=manifest_sha,
        raw_dir=raw_dir,
        timezone=args.timezone,
        cutoff_time=args.cutoff_time,
        lead_days=args.lead_days,
        model=args.model,
        api_url=args.api_url,
        api_key=api_key,
        chunk_days=args.chunk_days,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        overwrite=args.overwrite,
    )
    print(
        f"{output} | rows={result['output']['rows']} | "
        f"sha256={result['output']['sha256']}",
        flush=True,
    )
    print(
        f"{manifest} | sha256={result['manifest_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
