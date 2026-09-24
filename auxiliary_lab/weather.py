"""Causal weather-forecast materialisation for auxiliary-model experiments.

The materialiser deliberately retrieves one *issued* ECMWF IFS run for every
delivery day instead of a stitched historical-weather series.  For a civil
delivery day ``D`` it uses the IFS run initialised at ``D-2 18:00 UTC`` and
keeps exactly the physical hours of ``D`` in ``Europe/Paris``.  Consequently,
spring/autumn DST days contain 23/25 rows without interpolation.

This module is independent from the operational Chronos/Kalman pipeline.  Its
Parquet output is a wide, hourly PIT source that can subsequently be declared
as an ``additional_source`` by the auxiliary lab.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import time
from typing import Any, Mapping, Sequence
import uuid

import httpx
import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
DEFAULT_ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"
DEFAULT_MODEL = "ecmwf_ifs"
DEFAULT_API_KEY_ENV = "OPEN_METEO_API_KEY"
DEFAULT_TIMEZONE = "Europe/Paris"
DEFAULT_CUTOFF_TIME = "08:00"
DEFAULT_FORECAST_HOURS = 72

HOURLY_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_100m",
    "wind_direction_100m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
    "surface_pressure",
)


@dataclass(frozen=True)
class WeatherPoint:
    name: str
    latitude: float
    longitude: float


# Stable, transparent spatial samples.  They cover several consumption and
# renewable-generation regions per bidding zone while keeping each API request
# compact (all points of one zone are requested together).
ZONE_POINTS: Mapping[str, tuple[WeatherPoint, ...]] = {
    "FR": (
        WeatherPoint("lille", 50.6292, 3.0573),
        WeatherPoint("paris", 48.8566, 2.3522),
        WeatherPoint("nantes", 47.2184, -1.5536),
        WeatherPoint("lyon", 45.7640, 4.8357),
        WeatherPoint("toulouse", 43.6047, 1.4442),
    ),
    "DE": (
        WeatherPoint("hamburg", 53.5511, 9.9937),
        WeatherPoint("berlin", 52.5200, 13.4050),
        WeatherPoint("cologne", 50.9375, 6.9603),
        WeatherPoint("frankfurt", 50.1109, 8.6821),
        WeatherPoint("munich", 48.1351, 11.5820),
    ),
    "BE": (
        WeatherPoint("brussels", 50.8503, 4.3517),
        WeatherPoint("antwerp", 51.2194, 4.4025),
        WeatherPoint("ghent", 51.0543, 3.7174),
        WeatherPoint("liege", 50.6326, 5.5797),
    ),
    "NL": (
        WeatherPoint("amsterdam", 52.3676, 4.9041),
        WeatherPoint("rotterdam", 51.9244, 4.4777),
        WeatherPoint("eindhoven", 51.4416, 5.4697),
        WeatherPoint("groningen", 53.2194, 6.5665),
    ),
    "ES": (
        WeatherPoint("madrid", 40.4168, -3.7038),
        WeatherPoint("barcelona", 41.3874, 2.1686),
        WeatherPoint("bilbao", 43.2630, -2.9350),
        WeatherPoint("valencia", 39.4699, -0.3763),
        WeatherPoint("seville", 37.3891, -5.9845),
    ),
}


class WeatherMaterializationError(ValueError):
    """Raised when an issued run violates the PIT or completeness contract."""


@dataclass(frozen=True)
class ParsedZoneDay:
    frame: pd.DataFrame
    hourly_units: Mapping[str, str]
    returned_grid_points: tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class FetchRecord:
    delivery_day: str
    zone: str
    run_init_utc: str
    cutoff_utc: str
    request_sha256: str
    response_sha256: str
    cache_path: str
    from_cache: bool
    row_count: int
    hourly_units: Mapping[str, str]
    returned_grid_points: tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class MaterializationResult:
    output_path: Path
    manifest_path: Path
    row_count: int
    day_count: int
    response_count: int
    dataset_sha256: str


def _naive_day(value: str | pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if day.tzinfo is not None:
        raise WeatherMaterializationError(
            "Les jours de livraison doivent etre des dates civiles sans fuseau."
        )
    if day != day.normalize():
        raise WeatherMaterializationError(
            f"Jour de livraison non normalise: {value!r}."
        )
    return day


def delivery_utc_index(
    delivery_day: str | pd.Timestamp,
    *,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DatetimeIndex:
    """Return the exact physical hours of one civil delivery day."""

    day = _naive_day(delivery_day)
    local_start = day.tz_localize(timezone)
    local_end = (day + pd.Timedelta(days=1)).tz_localize(timezone)
    index = pd.date_range(
        local_start.tz_convert("UTC"),
        local_end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
        name="delivery_start_utc",
    )
    if len(index) not in {23, 24, 25}:
        raise WeatherMaterializationError(
            f"{day.date()}: jour civil inattendu de {len(index)} heures."
        )
    return index


def issue_times_for_delivery(
    delivery_day: str | pd.Timestamp,
    *,
    timezone: str = DEFAULT_TIMEZONE,
    cutoff_time: str = DEFAULT_CUTOFF_TIME,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the fixed D-2 18Z run and the D-1 civil auction cutoff."""

    day = _naive_day(delivery_day)
    run_init_utc = (day - pd.Timedelta(days=2) + pd.Timedelta(hours=18)).tz_localize(
        "UTC"
    )
    try:
        cutoff_delta = pd.Timedelta(
            cutoff_time + ":00" if cutoff_time.count(":") == 1 else cutoff_time
        )
    except ValueError as exc:
        raise WeatherMaterializationError(
            f"Heure de cutoff invalide: {cutoff_time!r}."
        ) from exc
    if cutoff_delta < pd.Timedelta(0) or cutoff_delta >= pd.Timedelta(days=1):
        raise WeatherMaterializationError(
            "L'heure de cutoff doit appartenir a la journee civile D-1."
        )
    cutoff_local = (day - pd.Timedelta(days=1) + cutoff_delta).tz_localize(timezone)
    cutoff_utc = cutoff_local.tz_convert("UTC")
    validate_causality(run_init_utc, cutoff_utc)
    return run_init_utc, cutoff_utc


def validate_causality(run_init_utc: Any, cutoff_utc: Any) -> None:
    run = pd.Timestamp(run_init_utc)
    cutoff = pd.Timestamp(cutoff_utc)
    if run.tzinfo is None or cutoff.tzinfo is None:
        raise WeatherMaterializationError(
            "run_init_utc et cutoff_utc doivent etre timezone-aware."
        )
    run = run.tz_convert("UTC")
    cutoff = cutoff.tz_convert("UTC")
    if run >= cutoff:
        raise WeatherMaterializationError(
            f"Run non causal: initialisation={run.isoformat()} >= "
            f"cutoff={cutoff.isoformat()}."
        )


def _normalise_payload(payload: Any, *, expected_points: int) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping) and bool(payload.get("error")):
        raise WeatherMaterializationError(
            f"Erreur Open-Meteo: {payload.get('reason', payload)}"
        )
    responses: list[Any]
    if isinstance(payload, Mapping):
        responses = [payload]
    elif isinstance(payload, list):
        responses = payload
    else:
        raise WeatherMaterializationError(
            "La reponse Open-Meteo doit etre un objet ou une liste d'objets."
        )
    if len(responses) != expected_points:
        raise WeatherMaterializationError(
            f"Open-Meteo a retourne {len(responses)} point(s), attendu "
            f"{expected_points}."
        )
    if any(not isinstance(item, Mapping) for item in responses):
        raise WeatherMaterializationError("Reponse multi-points Open-Meteo invalide.")
    return responses  # type: ignore[return-value]


def _point_frame(
    response: Mapping[str, Any],
    *,
    expected_index: pd.DatetimeIndex,
    variables: Sequence[str],
) -> tuple[pd.DataFrame, Mapping[str, str], Mapping[str, float]]:
    hourly = response.get("hourly")
    if not isinstance(hourly, Mapping) or "time" not in hourly:
        raise WeatherMaterializationError("Bloc hourly/time absent de la reponse.")
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(hourly["time"], utc=True, errors="raise"),
            name="delivery_start_utc",
        )
    except (TypeError, ValueError) as exc:
        raise WeatherMaterializationError("Timeline Open-Meteo invalide.") from exc
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise WeatherMaterializationError(
            "Timeline Open-Meteo dupliquee ou non croissante."
        )
    if bool((index.minute != 0).any()) or bool((index.second != 0).any()):
        raise WeatherMaterializationError(
            "Timeline Open-Meteo non alignee sur les heures pleines."
        )
    missing_variables = [variable for variable in variables if variable not in hourly]
    if missing_variables:
        raise WeatherMaterializationError(
            f"Variables hourly absentes: {missing_variables}."
        )
    columns: dict[str, pd.Series] = {}
    for variable in variables:
        values = hourly[variable]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise WeatherMaterializationError(
                f"Variable hourly invalide: {variable}."
            )
        if len(values) != len(index):
            raise WeatherMaterializationError(
                f"{variable}: {len(values)} valeurs pour {len(index)} timestamps."
            )
        columns[variable] = pd.to_numeric(
            pd.Series(values, index=index), errors="coerce"
        )
    frame = pd.DataFrame(columns, index=index)
    missing_hours = expected_index.difference(frame.index)
    if len(missing_hours):
        preview = ", ".join(str(item) for item in missing_hours[:3])
        raise WeatherMaterializationError(
            f"{len(missing_hours)} heure(s) de livraison absente(s): {preview}."
        )
    selected = frame.loc[expected_index]
    invalid = ~np.isfinite(selected.to_numpy(dtype=float))
    if bool(invalid.any()):
        positions = np.argwhere(invalid)
        row, column = positions[0]
        raise WeatherMaterializationError(
            "Valeur meteo manquante/non finie sans fill autorise: "
            f"{selected.columns[int(column)]} a {selected.index[int(row)]}."
        )
    raw_units = response.get("hourly_units", {})
    units = {
        variable: str(raw_units.get(variable, ""))
        for variable in variables
        if isinstance(raw_units, Mapping)
    }
    try:
        grid = {
            "latitude": float(response["latitude"]),
            "longitude": float(response["longitude"]),
            "elevation": float(response.get("elevation", math.nan)),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherMaterializationError(
            "Coordonnees de grille absentes ou invalides dans la reponse."
        ) from exc
    return selected.astype(float), units, grid


def _circular_mean_std_degrees(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(np.mod(values, 360.0))
    sin_mean = np.mean(np.sin(radians), axis=1)
    cos_mean = np.mean(np.cos(radians), axis=1)
    mean = np.mod(np.rad2deg(np.arctan2(sin_mean, cos_mean)), 360.0)
    resultant = np.clip(np.hypot(sin_mean, cos_mean), 1e-12, 1.0)
    std = np.minimum(np.rad2deg(np.sqrt(-2.0 * np.log(resultant))), 180.0)
    return mean, std


def parse_zone_day_payload(
    payload: Any,
    *,
    zone: str,
    points: Sequence[WeatherPoint],
    delivery_day: str | pd.Timestamp,
    run_init_utc: Any,
    cutoff_utc: Any,
    timezone: str = DEFAULT_TIMEZONE,
    variables: Sequence[str] = HOURLY_VARIABLES,
) -> ParsedZoneDay:
    """Parse, strictly slice and spatially aggregate one zone/day response."""

    zone = str(zone).upper()
    if not points:
        raise WeatherMaterializationError(f"{zone}: aucun point meteo configure.")
    validate_causality(run_init_utc, cutoff_utc)
    run = pd.Timestamp(run_init_utc).tz_convert("UTC")
    cutoff = pd.Timestamp(cutoff_utc).tz_convert("UTC")
    expected = delivery_utc_index(delivery_day, timezone=timezone)
    responses = _normalise_payload(payload, expected_points=len(points))

    point_frames: list[pd.DataFrame] = []
    units_reference: Mapping[str, str] | None = None
    grid_points: list[Mapping[str, float]] = []
    for response in responses:
        point_frame, units, grid = _point_frame(
            response,
            expected_index=expected,
            variables=variables,
        )
        if units_reference is None:
            units_reference = units
        elif dict(units) != dict(units_reference):
            raise WeatherMaterializationError(
                f"{zone}: unites hourly incoherentes entre les points."
            )
        point_frames.append(point_frame)
        grid_points.append(grid)

    output = pd.DataFrame(
        {
            "delivery_start_utc": expected,
            "run_init_utc": pd.DatetimeIndex([run] * len(expected)),
            "cutoff_utc": pd.DatetimeIndex([cutoff] * len(expected)),
        }
    )
    lead = (expected - run) / pd.Timedelta(hours=1)
    if not np.allclose(lead, np.rint(lead)):
        raise WeatherMaterializationError("Lead hours non entiers.")
    if bool((lead < 0).any()):
        raise WeatherMaterializationError("Le run ne couvre pas la livraison future.")
    output["lead_hours"] = np.rint(lead).astype(int)

    prefix = zone.casefold()
    for variable in variables:
        matrix = np.column_stack(
            [frame[variable].to_numpy(dtype=float) for frame in point_frames]
        )
        if variable.startswith("wind_direction_"):
            mean, std = _circular_mean_std_degrees(matrix)
        else:
            mean = np.mean(matrix, axis=1)
            std = np.std(matrix, axis=1, ddof=0)
        output[f"{prefix}_{variable}_mean"] = mean
        output[f"{prefix}_{variable}_std"] = std

    feature_columns = output.columns[4:]
    if bool((~np.isfinite(output.loc[:, feature_columns].to_numpy(dtype=float))).any()):
        raise WeatherMaterializationError(
            f"{zone}: agregation spatiale non finie."
        )
    return ParsedZoneDay(
        frame=output,
        hourly_units=dict(units_reference or {}),
        returned_grid_points=tuple(grid_points),
    )


def build_request_parameters(
    *,
    points: Sequence[WeatherPoint],
    run_init_utc: Any,
    forecast_hours: int = DEFAULT_FORECAST_HOURS,
    model: str = DEFAULT_MODEL,
    variables: Sequence[str] = HOURLY_VARIABLES,
) -> dict[str, str | int]:
    run = pd.Timestamp(run_init_utc)
    if run.tzinfo is None:
        raise WeatherMaterializationError("run_init_utc doit etre timezone-aware.")
    run = run.tz_convert("UTC")
    if not points:
        raise WeatherMaterializationError("Au moins un point meteo est requis.")
    if int(forecast_hours) <= 0:
        raise WeatherMaterializationError("forecast_hours doit etre strictement positif.")
    return {
        "latitude": ",".join(f"{point.latitude:.6f}" for point in points),
        "longitude": ",".join(f"{point.longitude:.6f}" for point in points),
        "hourly": ",".join(variables),
        "models": str(model),
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "forecast_hours": int(forecast_hours),
        "timezone": "GMT",
        "timeformat": "iso8601",
        "wind_speed_unit": "ms",
    }


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bytes_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ssl_verify_context() -> ssl.SSLContext | bool:
    """Honor the corporate CA variables commonly configured on Windows.

    HTTPX natively recognises ``SSL_CERT_FILE`` but not every environment maps
    the Requests-compatible ``REQUESTS_CA_BUNDLE`` variable.  Resolving both
    explicitly avoids disabling TLS verification behind a corporate proxy.
    """

    ca_value = os.environ.get("SSL_CERT_FILE") or os.environ.get(
        "REQUESTS_CA_BUNDLE"
    )
    if not ca_value:
        return True
    ca_path = Path(ca_value).expanduser()
    if not ca_path.is_file():
        raise WeatherMaterializationError(
            f"Bundle CA configure mais introuvable: {ca_path}."
        )
    try:
        return ssl.create_default_context(cafile=str(ca_path))
    except (OSError, ssl.SSLError) as exc:
        raise WeatherMaterializationError(
            f"Bundle CA illisible/invalide: {ca_path}."
        ) from exc


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _request_raw(
    client: httpx.Client,
    *,
    endpoint: str,
    parameters: Mapping[str, str | int],
    api_key: str | None,
    retries: int,
) -> bytes:
    request_parameters = dict(parameters)
    if api_key:
        request_parameters["apikey"] = api_key
    last_error: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            response = client.get(endpoint, params=request_parameters)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < int(retries):
                time.sleep(min(2 ** (attempt - 1), 8))
    raise WeatherMaterializationError(
        f"Echec Open-Meteo apres {retries} tentative(s): {last_error}"
    ) from last_error


def _fetch_zone_day(
    *,
    client: httpx.Client,
    endpoint: str,
    cache_dir: Path,
    force_refresh: bool,
    api_key: str | None,
    retries: int,
    delivery_day: pd.Timestamp,
    zone: str,
    timezone: str,
    cutoff_time: str,
    forecast_hours: int,
    model: str,
    variables: Sequence[str],
) -> tuple[pd.DataFrame, FetchRecord]:
    points = ZONE_POINTS[zone]
    run, cutoff = issue_times_for_delivery(
        delivery_day,
        timezone=timezone,
        cutoff_time=cutoff_time,
    )
    parameters = build_request_parameters(
        points=points,
        run_init_utc=run,
        forecast_hours=forecast_hours,
        model=model,
        variables=variables,
    )
    request_identity = {
        "endpoint": endpoint,
        "parameters": parameters,
        "zone": zone,
        "points": [point.__dict__ for point in points],
    }
    request_sha256 = _canonical_hash(request_identity)
    run_label = run.strftime("%Y%m%dT%H%MZ")
    cache_path = (
        cache_dir
        / run.strftime("%Y")
        / run.strftime("%m")
        / f"{run_label}_{zone.casefold()}_{request_sha256[:16]}.json"
    )
    from_cache = cache_path.is_file() and not force_refresh
    if from_cache:
        raw = cache_path.read_bytes()
    else:
        raw = _request_raw(
            client,
            endpoint=endpoint,
            parameters=parameters,
            api_key=api_key,
            retries=retries,
        )
        # Validate JSON before committing it as a resumable raw response.
        try:
            json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WeatherMaterializationError(
                f"{zone} {delivery_day.date()}: reponse JSON invalide."
            ) from exc
        _atomic_bytes(cache_path, raw)
    response_sha256 = _bytes_sha256(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeatherMaterializationError(
            f"Cache JSON invalide: {cache_path}."
        ) from exc
    parsed = parse_zone_day_payload(
        payload,
        zone=zone,
        points=points,
        delivery_day=delivery_day,
        run_init_utc=run,
        cutoff_utc=cutoff,
        timezone=timezone,
        variables=variables,
    )
    record = FetchRecord(
        delivery_day=str(delivery_day.date()),
        zone=zone,
        run_init_utc=run.isoformat(),
        cutoff_utc=cutoff.isoformat(),
        request_sha256=request_sha256,
        response_sha256=response_sha256,
        cache_path=str(cache_path.resolve()),
        from_cache=from_cache,
        row_count=len(parsed.frame),
        hourly_units=dict(parsed.hourly_units),
        returned_grid_points=parsed.returned_grid_points,
    )
    return parsed.frame, record


def _combine_zone_frames(
    by_day: Mapping[pd.Timestamp, Mapping[str, pd.DataFrame]],
    *,
    zones: Sequence[str],
) -> pd.DataFrame:
    metadata = ["delivery_start_utc", "run_init_utc", "cutoff_utc", "lead_hours"]
    day_frames: list[pd.DataFrame] = []
    for day in sorted(by_day):
        zone_frames = by_day[day]
        missing = [zone for zone in zones if zone not in zone_frames]
        if missing:
            raise WeatherMaterializationError(
                f"{day.date()}: zones absentes apres collecte: {missing}."
            )
        reference = zone_frames[zones[0]].loc[:, metadata].reset_index(drop=True)
        combined = reference.copy()
        for zone in zones:
            frame = zone_frames[zone].reset_index(drop=True)
            if not frame.loc[:, metadata].equals(reference):
                raise WeatherMaterializationError(
                    f"{day.date()}: timeline/provenance incoherente pour {zone}."
                )
            feature_columns = [column for column in frame if column not in metadata]
            overlap = sorted(set(feature_columns).intersection(combined.columns))
            if overlap:
                raise WeatherMaterializationError(
                    f"Colonnes meteo dupliquees: {overlap}."
                )
            combined = pd.concat(
                [combined, frame.loc[:, feature_columns]], axis=1, copy=False
            )
        day_frames.append(combined)
    output = pd.concat(day_frames, ignore_index=True)
    index = pd.DatetimeIndex(output["delivery_start_utc"])
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise WeatherMaterializationError(
            "Timeline finale dupliquee ou non croissante."
        )
    return output


def materialize_open_meteo_weather(
    *,
    start_day: str | pd.Timestamp,
    end_day: str | pd.Timestamp,
    output_path: str | Path,
    zones: Sequence[str] = ("FR", "DE", "BE", "NL", "ES"),
    manifest_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    endpoint: str | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    require_api_key: bool = False,
    timezone: str = DEFAULT_TIMEZONE,
    cutoff_time: str = DEFAULT_CUTOFF_TIME,
    forecast_hours: int = DEFAULT_FORECAST_HOURS,
    model: str = DEFAULT_MODEL,
    variables: Sequence[str] = HOURLY_VARIABLES,
    workers: int = 4,
    retries: int = 3,
    timeout_seconds: float = 60.0,
    force_refresh: bool = False,
) -> MaterializationResult:
    """Download/cache issued runs and atomically build a strict PIT Parquet."""

    start = _naive_day(start_day)
    end = _naive_day(end_day)
    if end < start:
        raise WeatherMaterializationError("end_day doit etre >= start_day.")
    selected_zones = tuple(dict.fromkeys(str(zone).upper() for zone in zones))
    if not selected_zones:
        raise WeatherMaterializationError("Au moins une zone est requise.")
    unknown = sorted(set(selected_zones).difference(ZONE_POINTS))
    if unknown:
        raise WeatherMaterializationError(f"Zones non supportees: {unknown}.")
    if int(workers) <= 0 or int(retries) <= 0:
        raise WeatherMaterializationError("workers/retries doivent etre positifs.")
    output = Path(output_path).expanduser().resolve()
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    cache = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else output.parent / "open_meteo_single_runs_cache"
    )
    selected_endpoint = (
        endpoint
        or os.environ.get("OPEN_METEO_SINGLE_RUNS_ENDPOINT")
        or DEFAULT_ENDPOINT
    ).strip()
    if not selected_endpoint:
        raise WeatherMaterializationError("Endpoint Open-Meteo vide.")
    api_key = os.environ.get(api_key_env)
    if require_api_key and not api_key:
        raise WeatherMaterializationError(
            f"Cle API absente de la variable d'environnement {api_key_env}."
        )

    days = tuple(pd.date_range(start, end, freq="D"))
    by_day: dict[pd.Timestamp, dict[str, pd.DataFrame]] = {
        day: {} for day in days
    }
    records: list[FetchRecord] = []
    tasks: dict[Future[tuple[pd.DataFrame, FetchRecord]], tuple[pd.Timestamp, str]] = {}
    limits = httpx.Limits(
        max_connections=max(2, int(workers)),
        max_keepalive_connections=max(2, int(workers)),
    )
    with httpx.Client(
        timeout=float(timeout_seconds),
        limits=limits,
        verify=_ssl_verify_context(),
        follow_redirects=True,
    ) as client:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            for day in days:
                for zone in selected_zones:
                    future = pool.submit(
                        _fetch_zone_day,
                        client=client,
                        endpoint=selected_endpoint,
                        cache_dir=cache,
                        force_refresh=force_refresh,
                        api_key=api_key,
                        retries=int(retries),
                        delivery_day=day,
                        zone=zone,
                        timezone=timezone,
                        cutoff_time=cutoff_time,
                        forecast_hours=int(forecast_hours),
                        model=model,
                        variables=tuple(variables),
                    )
                    tasks[future] = (day, zone)
            total = len(tasks)
            for number, future in enumerate(as_completed(tasks), start=1):
                day, zone = tasks[future]
                try:
                    frame, record = future.result()
                except Exception as exc:
                    for pending in tasks:
                        pending.cancel()
                    raise WeatherMaterializationError(
                        f"{zone} {day.date()}: materialisation impossible: {exc}"
                    ) from exc
                by_day[day][zone] = frame
                records.append(record)
                if number == 1 or number % 25 == 0 or number == total:
                    cached = sum(record.from_cache for record in records)
                    print(
                        f"[Open-Meteo] {number}/{total} zone-jours | "
                        f"cache={cached} reseau={number - cached}",
                        flush=True,
                    )

    dataset = _combine_zone_frames(by_day, zones=selected_zones)
    expected_rows = sum(len(delivery_utc_index(day, timezone=timezone)) for day in days)
    if len(dataset) != expected_rows:
        raise WeatherMaterializationError(
            f"Dataset final incomplet: {len(dataset)} lignes, attendu {expected_rows}."
        )
    feature_columns = [
        column
        for column in dataset.columns
        if column
        not in {"delivery_start_utc", "run_init_utc", "cutoff_utc", "lead_hours"}
    ]
    if bool((~np.isfinite(dataset[feature_columns].to_numpy(dtype=float))).any()):
        raise WeatherMaterializationError("Dataset final contenant des valeurs non finies.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.parquet")
    dataset.to_parquet(temporary, index=False)
    # Read-back catches a truncated/unsupported Parquet before publication.
    verification = pd.read_parquet(temporary)
    if len(verification) != len(dataset) or list(verification) != list(dataset):
        temporary.unlink(missing_ok=True)
        raise WeatherMaterializationError("Verification Parquet en echec.")
    temporary.replace(output)
    dataset_sha256 = file_sha256(output)

    sorted_records = sorted(records, key=lambda item: (item.delivery_day, item.zone))
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": {
            "provider": "Open-Meteo Single Runs API",
            "endpoint": selected_endpoint,
            "model": model,
            "api_key_env": api_key_env,
            "api_key_used": bool(api_key),
        },
        "causal_contract": {
            "run_policy": "delivery day D uses ECMWF IFS D-2 18:00 UTC",
            "cutoff_policy": f"D-1 {cutoff_time} {timezone}",
            "strict_rule": "run_init_utc < cutoff_utc",
            "fill_or_interpolation": "forbidden",
            "forecast_hours": int(forecast_hours),
        },
        "coverage": {
            "start_day": str(start.date()),
            "end_day": str(end.date()),
            "day_count": len(days),
            "row_count": len(dataset),
            "timezone": timezone,
            "zones": list(selected_zones),
        },
        "variables": list(variables),
        "feature_columns": feature_columns,
        "zone_points": {
            zone: [point.__dict__ for point in ZONE_POINTS[zone]]
            for zone in selected_zones
        },
        "dataset": {
            "path": str(output),
            "sha256": dataset_sha256,
            "size_bytes": output.stat().st_size,
        },
        "responses": [record.__dict__ for record in sorted_records],
    }
    _atomic_bytes(
        manifest,
        (json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    return MaterializationResult(
        output_path=output,
        manifest_path=manifest,
        row_count=len(dataset),
        day_count=len(days),
        response_count=len(records),
        dataset_sha256=dataset_sha256,
    )


__all__ = [
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_CUTOFF_TIME",
    "DEFAULT_ENDPOINT",
    "DEFAULT_FORECAST_HOURS",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEZONE",
    "HOURLY_VARIABLES",
    "MaterializationResult",
    "ParsedZoneDay",
    "WeatherMaterializationError",
    "WeatherPoint",
    "ZONE_POINTS",
    "build_request_parameters",
    "delivery_utc_index",
    "file_sha256",
    "issue_times_for_delivery",
    "materialize_open_meteo_weather",
    "parse_zone_day_payload",
    "validate_causality",
]
