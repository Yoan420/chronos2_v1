"""Experimental, publication-audited NOAA GFS weather archives.

Only original 00Z D-1 forecasts are read.  NOAA's object publication time is
historical archive evidence, not proof of a locally emitted forecast.  Outputs
are deliberately new weather aliases and cannot certify production promotion.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
import ssl
import time
from typing import Any, Mapping, Sequence
import uuid

import httpx
import numpy as np
import pandas as pd

from auxiliary_lab.weather import WeatherPoint, ZONE_POINTS, delivery_utc_index

SCHEMA_VERSION = 1
BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
MAX_INDEX_BYTES = 128 * 1024
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_OBJECT_BYTES = 1024 * 1024 * 1024
DEFAULT_ZONES = ("FR", "DE", "BE", "NL")


class GfsError(ValueError):
    """A source, publication, decoding or completeness contract failed."""


@dataclass(frozen=True)
class FieldSpec:
    index_name: str
    index_level: str
    category: int
    number: int
    level_type: str
    level: int
    units: tuple[str, ...]
    average: bool = False


FIELDS = {
    "temperature_2m": FieldSpec("TMP", "2 m above ground", 0, 0, "heightAboveGround", 2, ("K",)),
    "u100": FieldSpec("UGRD", "100 m above ground", 2, 2, "heightAboveGround", 100, ("m s**-1", "m s-1")),
    "v100": FieldSpec("VGRD", "100 m above ground", 2, 3, "heightAboveGround", 100, ("m s**-1", "m s-1")),
    "solar": FieldSpec("DSWRF", "surface", 4, 7, "surface", 0, ("W m**-2", "W m-2"), True),
}


@dataclass(frozen=True)
class FieldSlice:
    name: str
    start: int
    end: int  # inclusive HTTP byte offset
    start_step: int
    end_step: int
    index_line: str


@dataclass(frozen=True)
class DecodedField:
    values: np.ndarray
    start_step: int
    end_step: int
    metadata: Mapping[str, Any]


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_bounded(path: Path, limit: int) -> bytes:
    if path.stat().st_size > limit:
        raise GfsError(f"Cached file exceeds its bound: {path}")
    with path.open("rb") as source:
        value = source.read(limit + 1)
    if len(value) > limit:
        raise GfsError(f"Cached file grew beyond its bound: {path}")
    return value


def _unlink_owned(path: Path, identity: os.stat_result) -> None:
    """Never remove a destination that another process has replaced."""
    try:
        current = path.stat()
        if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            path.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def _output_reservations(output: Path, manifest: Path):
    destinations = (output, manifest)
    lock_paths = sorted((p.with_name(p.name + ".publish.lock") for p in destinations), key=lambda p: str(p).casefold())
    if output == manifest or any(p in destinations for p in lock_paths):
        raise GfsError("Output and manifest reservations collide.")
    owned = []
    try:
        for lock in lock_paths:
            lock.parent.mkdir(parents=True, exist_ok=True)
            try:
                with lock.open("xb") as stream:
                    owned.append((lock, os.fstat(stream.fileno())))
                    stream.write(f"pid={os.getpid()} reservation={uuid.uuid4().hex}\n".encode("ascii"))
            except FileExistsError as exc:
                raise GfsError(f"Another materialisation reserved this destination: {lock}") from exc
        if any(p.exists() for p in destinations):
            raise GfsError("Final outputs already exist; choose a new output path.")
        yield
    finally:
        for lock, identity in reversed(owned):
            _unlink_owned(lock, identity)


def _reserve_destinations(function):
    @wraps(function)
    def reserved(**kwargs):
        output = Path(kwargs["output_path"]).resolve()
        manifest = Path(kwargs["manifest_path"]).resolve() if kwargs.get("manifest_path") else output.with_suffix(".manifest.json")
        with _output_reservations(output, manifest):
            return function(**kwargs)
    return reserved


def _publish_pair(data_stage: Path, manifest_stage: Path, output: Path, manifest: Path) -> None:
    """Publish complete staged bytes, without replacing either destination."""
    created = []
    try:
        for staged, destination in ((data_stage, output), (manifest_stage, manifest)):
            identity = staged.stat()
            # link() is atomic and fails if destination already exists. Staging
            # lives beside its destination; after unlink, final files have nlink=1.
            os.link(staged, destination)
            created.append((destination, identity))
    except BaseException:
        for destination, identity in reversed(created):
            _unlink_owned(destination, identity)
        raise


def issue_times(delivery_day: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = pd.Timestamp(delivery_day)
    if day.tzinfo is not None or day != day.normalize():
        raise GfsError("delivery_day must be a naive civil date.")
    previous = day - pd.Timedelta(days=1)
    return previous.tz_localize("UTC"), (previous + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")


def object_url(run_init: pd.Timestamp, forecast_hour: int) -> str:
    run = pd.Timestamp(run_init)
    if run.tzinfo is None or run.utcoffset() != pd.Timedelta(0) or run != run.normalize():
        raise GfsError("Only the original 00 UTC cycle is supported.")
    if isinstance(forecast_hour, bool) or not isinstance(forecast_hour, int) or not 1 <= forecast_hour <= 120:
        raise GfsError("Forecast hour must be an integer within [1, 120].")
    return f"{BASE_URL}/gfs.{run:%Y%m%d}/00/atmos/gfs.t00z.pgrb2.0p25.f{forecast_hour:03d}"


def parse_index(text: str, run_init: pd.Timestamp, forecast_hour: int, content_length: int) -> dict[str, FieldSlice]:
    """Resolve exact messages; no substring matches or whole-file fallback."""
    if not 0 < content_length <= MAX_OBJECT_BYTES:
        raise GfsError("GRIB object length outside the permitted bound.")
    rows = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) < 7 or not parts[0].isdigit() or not parts[1].isdigit():
            raise GfsError("Malformed GFS index record.")
        if parts[2] != f"d={run_init:%Y%m%d%H}":
            raise GfsError("GFS index initialisation differs from the requested cycle.")
        rows.append((int(parts[1]), parts, line))
    offsets = [row[0] for row in rows]
    if not offsets or offsets[0] != 0 or any(b <= a for a, b in zip(offsets, offsets[1:])) or offsets[-1] >= content_length:
        raise GfsError("GFS index byte offsets are invalid or ambiguous.")
    result = {}
    for i, (start, parts, line) in enumerate(rows):
        for name, spec in FIELDS.items():
            if parts[3:5] != [spec.index_name, spec.index_level]:
                continue
            descriptor = parts[5]
            pattern = r"(\d+)-(\d+) hour ave fcst" if spec.average else r"(\d+) hour fcst"
            match = re.fullmatch(pattern, descriptor)
            if match is None:
                raise GfsError(f"Unexpected forecast descriptor for {name}: {descriptor}")
            begin, end = (int(match[1]), int(match[2])) if spec.average else (forecast_hour, int(match[1]))
            if end != forecast_hour or begin > end or (spec.average and begin == end):
                raise GfsError(f"Wrong forecast interval for {name}.")
            finish = (offsets[i + 1] if i + 1 < len(offsets) else content_length) - 1
            if name in result or not 16 <= finish - start + 1 <= MAX_MESSAGE_BYTES:
                raise GfsError(f"Duplicate or oversized message: {name}.")
            result[name] = FieldSlice(name, start, finish, begin, end, line)
    if set(result) != set(FIELDS):
        raise GfsError(f"Required GFS fields missing: {sorted(set(FIELDS) - set(result))}")
    return result


def hourly_radiation(previous: DecodedField, current: DecodedField, tolerance: float = 0.5) -> tuple[np.ndarray, int]:
    """Convert cumulative interval means to the physical [h,h+1) hour."""
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise GfsError("Radiation packing tolerance must be between 0 and 1 W/m2.")
    if current.end_step != previous.end_step + 1 or current.values.shape != previous.values.shape:
        raise GfsError("Solar endpoints must have consecutive hours and identical points.")
    if current.start_step == previous.start_step:
        values = current.values * (current.end_step - current.start_step) - previous.values * (previous.end_step - previous.start_step)
    elif current.start_step == previous.end_step and current.end_step - current.start_step == 1:
        values = current.values.copy()
    else:
        raise GfsError("Solar accumulation reset cannot be resolved to exactly one hour.")
    if not np.isfinite(values).all() or np.any(values < -tolerance) or np.any(values > 1600):
        raise GfsError("Deaveraged solar radiation is non-finite or outside physical bounds.")
    clipped = int(np.count_nonzero(values < 0))
    return np.maximum(values, 0), clipped


def decode_message(raw: bytes, name: str, run_init: pd.Timestamp, forecast_hour: int, points: Sequence[WeatherPoint], radiation_tolerance: float = 0.5) -> DecodedField:
    """Decode just one pinned GRIB message and 18 spatial samples, lazily."""
    if raw[:4] != b"GRIB" or len(raw) < 20 or raw[7] != 2 or int.from_bytes(raw[8:16], "big") != len(raw) or raw[-4:] != b"7777":
        raise GfsError("Invalid or concatenated GRIB2 message.")
    try:
        import eccodes
    except ImportError as exc:
        raise GfsError("ecCodes is required; use --dependency-directory for the isolated decoder installation.") from exc
    spec = FIELDS[name]
    if not math.isfinite(radiation_tolerance) or not 0 <= radiation_tolerance <= 1:
        raise GfsError("Invalid solar packing tolerance.")
    handle = eccodes.codes_new_from_message(raw)
    try:
        keys = ("edition", "centre", "discipline", "parameterCategory", "parameterNumber", "typeOfLevel", "level", "units", "stepType", "stepUnits", "startStep", "endStep", "dataDate", "dataTime", "validityDate", "validityTime", "gridType", "Ni", "Nj", "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees", "missingValue")
        meta = {key: eccodes.codes_get(handle, key) for key in keys}
        expected = {"edition": 2, "discipline": 0, "parameterCategory": spec.category, "typeOfLevel": spec.level_type, "level": spec.level, "stepType": "avg" if spec.average else "instant", "stepUnits": 1, "dataDate": int(run_init.strftime("%Y%m%d")), "dataTime": 0, "endStep": forecast_hour, "gridType": "regular_ll", "Ni": 1440, "Nj": 721, "iDirectionIncrementInDegrees": 0.25, "jDirectionIncrementInDegrees": 0.25}
        for key, value in expected.items():
            if meta[key] != value:
                raise GfsError(f"{name}: decoded {key}={meta[key]!r}, expected {value!r}.")
        if meta["centre"] not in {"kwbc", 7} or meta["units"] not in spec.units:
            raise GfsError(f"Wrong NOAA centre or physical units for {name}: {meta}")
        # Both DSWRF identifiers are explicitly defined by NCEP table 4.2-0-4:
        # WMO parameter 7 and NOAA local parameter 192. Other radiation fields
        # (net, clear-sky, direct-beam, upward) are never interchangeable here.
        if meta["parameterNumber"] not in ((7, 192) if name == "solar" else (spec.number,)):
            raise GfsError(f"Wrong GRIB physical parameter for {name}.")
        valid = run_init + pd.Timedelta(hours=forecast_hour)
        if meta["validityDate"] != int(valid.strftime("%Y%m%d")) or meta["validityTime"] != int(valid.strftime("%H%M")):
            raise GfsError("GRIB validity time differs from its requested endpoint.")
        if not 0 <= meta["startStep"] <= forecast_hour or (not spec.average and meta["startStep"] != forecast_hour):
            raise GfsError("Invalid GRIB startStep.")
        nearest = eccodes.codes_grib_find_nearest_multiple(handle, False, [p.latitude for p in points], [p.longitude % 360 for p in points])
        if len(nearest) != len(points) or any(float(p["distance"]) > 40 for p in nearest):
            raise GfsError("Unexpected nearest grid point location.")
        values = np.array([p["value"] for p in nearest], dtype=float)
        if not np.isfinite(values).all() or np.any(values == float(meta["missingValue"])):
            raise GfsError("Missing or non-finite decoded weather values.")
        low, high = (150, 350) if name == "temperature_2m" else ((-200, 200) if name in {"u100", "v100"} else (-radiation_tolerance, 1600))
        if np.any(values < low) or np.any(values > high):
            raise GfsError(f"{name}: decoded values outside broad physical bounds [{low}, {high}].")
        meta["grid_points"] = [{k: float(p[k]) for k in ("lat", "lon", "distance")} for p in nearest]
        meta["eccodes_version"] = str(eccodes.codes_get_api_version())
        return DecodedField(values, int(meta["startStep"]), int(meta["endStep"]), meta)
    finally:
        eccodes.codes_release(handle)


def _grid_signature(field: DecodedField) -> tuple[tuple[float, float], ...]:
    points = field.metadata.get("grid_points", ())
    signature = tuple((float(p["lat"]), float(p["lon"]) % 360) for p in points)
    if not signature or len(signature) != len(field.values) or not np.isfinite(signature).all():
        raise GfsError("Decoded grid-point identity is missing or invalid.")
    return signature


def _publication(headers: Mapping[str, str], cutoff: pd.Timestamp) -> str:
    try:
        value = pd.Timestamp(headers["last-modified"])
        if value.tzinfo is None:
            raise ValueError("naive timestamp")
        value = value.tz_convert("UTC")
    except (KeyError, ValueError) as exc:
        raise GfsError("A timezone-aware Last-Modified is required.") from exc
    if value > cutoff:
        raise GfsError(f"NOAA object was published after cutoff: {value} > {cutoff}.")
    return value.isoformat()


class GfsClient:
    def __init__(self, cache_dir: str | Path, *, timeout_seconds: float = 45, retries: int = 2, ca_bundle: str | None = None, client: httpx.Client | None = None):
        if not 1 <= timeout_seconds <= 120 or not 0 <= retries <= 4:
            raise GfsError("Timeout/retries outside bounded limits.")
        self.cache_dir = Path(cache_dir)
        self.retries = retries
        self.owns_client = client is None
        self.client = client or httpx.Client(verify=ssl.create_default_context(cafile=ca_bundle), timeout=timeout_seconds, follow_redirects=False, headers={"Accept-Encoding": "identity", "User-Agent": "chronos2-noaa-weather-research/1"})

    def close(self) -> None:
        if self.owns_client:
            self.client.close()

    def _request(self, method: str, url: str, *, bound: int, headers: Mapping[str, str] | None = None, expected: int = 200) -> tuple[bytes, dict[str, str]]:
        if not url.startswith(BASE_URL + "/gfs."):
            raise GfsError("Only NOAA's fixed HTTPS bucket is permitted.")
        for attempt in range(self.retries + 1):
            try:
                with self.client.stream(method, url, headers=headers) as response:
                    response.raise_for_status()
                    if response.status_code != expected or response.headers.get("content-encoding", "identity") != "identity":
                        raise GfsError(f"Unexpected NOAA response status/encoding: {response.status_code}.")
                    if method != "HEAD" and int(response.headers.get("content-length", "0")) > bound:
                        raise GfsError("Response Content-Length exceeds the permitted download bound.")
                    data = bytearray()
                    if method != "HEAD":
                        for part in response.iter_bytes(chunk_size=65536):
                            if len(data) + len(part) > bound:
                                raise GfsError("Response exceeds its download bound; stopped streaming.")
                            data.extend(part)
                    return bytes(data), dict(response.headers)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in {429, 500, 502, 503, 504}:
                    raise GfsError(f"NOAA request failed: {exc.response.status_code} {url}") from exc
                if attempt == self.retries:
                    raise GfsError(f"NOAA request failed after {attempt + 1} attempts: {url}") from exc
                time.sleep(min(2 ** attempt, 4))
        raise AssertionError("unreachable")

    def fetch_endpoint(self, run_init: pd.Timestamp, cutoff: pd.Timestamp, forecast_hour: int, fields: Sequence[str] = tuple(FIELDS)) -> tuple[dict[str, bytes], dict[str, Any]]:
        url = object_url(run_init, forecast_hour)
        expected_cutoff = (run_init.tz_localize(None) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
        if pd.Timestamp(cutoff) != expected_cutoff:
            raise GfsError("The cutoff must be 08:00 Europe/Paris on the fixed run's civil date.")
        if not fields or any(name not in FIELDS for name in fields) or len(set(fields)) != len(fields):
            raise GfsError("Invalid requested fields.")
        folder = self.cache_dir / run_init.strftime("%Y%m%d00") / f"f{forecast_hour:03d}"
        manifest_path = folder / "source.json"
        if manifest_path.exists():
            manifest_raw = _read_bounded(manifest_path, 128 * 1024)
            meta = json.loads(manifest_raw)
            if meta.get("schema_version") != SCHEMA_VERSION or meta.get("url") != url or meta.get("run_init_utc") != run_init.isoformat() or meta.get("forecast_hour") != forecast_hour or meta.get("index_url") != url + ".idx":
                raise GfsError(f"Cache identity mismatch: {manifest_path}")
            index_raw = _read_bounded(folder / "index.idx", MAX_INDEX_BYTES)
            if sha256(index_raw) != meta["index_sha256"]:
                raise GfsError("Cached index SHA mismatch.")
            publications = [_publication(meta["index_headers"], cutoff), _publication(meta["object_headers"], cutoff)]
            etag = meta["object_headers"].get("etag", "")
            if not etag.startswith('"') or not etag.endswith('"') or meta["object_headers"].get("accept-ranges") != "bytes":
                raise GfsError("Cached source lacks a strong ETag or byte-range contract.")
            slices = parse_index(index_raw.decode("ascii"), run_init, forecast_hour, int(meta["object_headers"]["content-length"]))
            if set(fields).issubset(meta["messages"]):
                data = {}
                for name in fields:
                    item = meta["messages"][name]
                    raw = _read_bounded(folder / f"{name}.grib2", MAX_MESSAGE_BYTES)
                    expected_range = f"bytes {slices[name].start}-{slices[name].end}/{meta['object_headers']['content-length']}"
                    if sha256(raw) != item["sha256"] or len(raw) != slices[name].end - slices[name].start + 1 or item["headers"].get("content-range") != expected_range or item["headers"].get("etag") != meta["object_headers"].get("etag"):
                        raise GfsError(f"Cached message SHA or byte range mismatch: {name}")
                    if (item["start"], item["end"], item["start_step"], item["end_step"], item["index_line"]) != (slices[name].start, slices[name].end, slices[name].start_step, slices[name].end_step, slices[name].index_line):
                        raise GfsError("Cached interval metadata differs from the archived index.")
                    publications.append(_publication(item["headers"], cutoff))
                    data[name] = raw
                if min(pd.Timestamp(p) for p in publications) < run_init:
                    raise GfsError("Cached publication predates the model initialisation.")
                return data, {**meta, "publication_max_utc": max(publications), "cache_manifest": str(manifest_path.resolve()), "cache_manifest_sha256": sha256(manifest_raw), "from_cache": True}
        index_raw, index_headers = self._request("GET", url + ".idx", bound=MAX_INDEX_BYTES)
        _, object_headers = self._request("HEAD", url, bound=0)
        published = [_publication(index_headers, cutoff), _publication(object_headers, cutoff)]
        etag = object_headers.get("etag", "")
        if not etag.startswith('"') or not etag.endswith('"') or object_headers.get("accept-ranges") != "bytes":
            raise GfsError("A strong ETag and HTTP byte-range support are required.")
        slices = parse_index(index_raw.decode("ascii"), run_init, forecast_hour, int(object_headers["content-length"]))
        meta = {"schema_version": SCHEMA_VERSION, "url": url, "run_init_utc": run_init.isoformat(), "forecast_hour": forecast_hour, "index_url": url + ".idx", "index_sha256": sha256(index_raw), "index_headers": index_headers, "object_headers": object_headers, "messages": {}}
        result = {}
        for name in fields:
            item = slices[name]
            bound = item.end - item.start + 1
            raw, response_headers = self._request("GET", url, bound=bound, headers={"Range": f"bytes={item.start}-{item.end}", "If-Match": etag}, expected=206)
            expected_range = f"bytes {item.start}-{item.end}/{object_headers['content-length']}"
            if response_headers.get("content-range") != expected_range or response_headers.get("etag") != etag or len(raw) != bound:
                raise GfsError("GRIB response differs from the pinned ETag or exact byte range.")
            published.append(_publication(response_headers, cutoff))
            result[name] = raw
            meta["messages"][name] = {"sha256": sha256(raw), "bytes": len(raw), "start": item.start, "end": item.end, "start_step": item.start_step, "end_step": item.end_step, "index_line": item.index_line, "headers": response_headers}
        if min(pd.Timestamp(p) for p in published) < run_init:
            raise GfsError("NOAA publication predates the model initialisation.")
        meta["publication_max_utc"] = max(published)
        meta["retrieved_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        _write(folder / "index.idx", index_raw)
        for name, raw in result.items():
            _write(folder / f"{name}.grib2", raw)
        _write(manifest_path, _json_bytes(meta))
        return result, {**meta, "cache_manifest": str(manifest_path.resolve()), "cache_manifest_sha256": sha256(manifest_path.read_bytes()), "from_cache": False}


@_reserve_destinations
def materialize_noaa_gfs_weather(*, start_day: str, end_day: str, output_path: str | Path, zones: Sequence[str] = DEFAULT_ZONES, cache_dir: str | Path | None = None, manifest_path: str | Path | None = None, workers: int = 2, timeout_seconds: float = 45, retries: int = 2, ca_bundle: str | None = None, radiation_tolerance: float = 0.5) -> dict[str, Any]:
    if not 1 <= workers <= 4 or not 0 <= radiation_tolerance <= 1:
        raise GfsError("Workers must be 1..4 and radiation tolerance 0..1 W/m2.")
    zones = tuple(z.upper() for z in zones)
    if not zones or len(set(zones)) != len(zones) or any(z not in DEFAULT_ZONES for z in zones):
        raise GfsError("Select unique zones from FR, DE, BE, NL.")
    issue_times(start_day)
    issue_times(end_day)
    days = pd.date_range(start_day, end_day, freq="D")
    if not 1 <= len(days) <= 1096:
        raise GfsError("Select between 1 and 1096 delivery days.")
    output = Path(output_path).resolve()
    manifest = Path(manifest_path).resolve() if manifest_path else output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists() or output == manifest:
        raise GfsError("Final outputs already exist or collide; choose a new output path.")
    points = tuple(point for zone in zones for point in ZONE_POINTS[zone])
    point_slices = {}
    cursor = 0
    for zone in zones:
        point_slices[zone] = slice(cursor, cursor + len(ZONE_POINTS[zone]))
        cursor += len(ZONE_POINTS[zone])
    client = GfsClient(cache_dir or output.parent / "raw_cache", timeout_seconds=timeout_seconds, retries=retries, ca_bundle=ca_bundle)
    frames, records = [], []
    clipped_total = 0
    reference_grid = None
    try:
        for day in days:
            run, cutoff = issue_times(day)
            index = delivery_utc_index(day)
            first = int((index[0] - run) / pd.Timedelta(hours=1))
            hours = list(range(first, first + len(index) + 1))
            def endpoint(hour: int) -> tuple[int, dict[str, bytes], dict[str, Any]]:
                names = ("solar",) if hour == hours[-1] else tuple(FIELDS)
                raw, record = client.fetch_endpoint(run, cutoff, hour, names)
                return hour, raw, record
            endpoints = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # Only HTTP/cache I/O is parallel. This Windows ecCodes build
                # must be called sequentially from the coordinating thread.
                for hour, raw, record in pool.map(endpoint, hours):
                    decoded = {name: decode_message(value, name, run, hour, points, radiation_tolerance) for name, value in raw.items()}
                    for name, value in decoded.items():
                        archived = record["messages"][name]
                        if (value.start_step, value.end_step) != (archived["start_step"], archived["end_step"]):
                            raise GfsError("Index and decoded GRIB intervals disagree.")
                        signature = _grid_signature(value)
                        if reference_grid is None:
                            reference_grid = signature
                        elif signature != reference_grid:
                            raise GfsError("Weather variables or forecast endpoints use different ordered grid points.")
                    record["decoded"] = {name: value.metadata for name, value in decoded.items()}
                    endpoints.append((hour, decoded, record))
            decoded_by_hour = {hour: value for hour, value, _ in endpoints}
            publication = max(record["publication_max_utc"] for _, _, record in endpoints)
            data: dict[str, Any] = {"delivery_start_utc": index, "delivery_day": day.strftime("%Y-%m-%d"), "run_init_utc": run, "cutoff_utc": cutoff, "publication_max_utc": pd.Timestamp(publication)}
            for zone in zones:
                for variable in ("temperature_2m_c", "wind_speed_100m_ms", "shortwave_radiation_wm2"):
                    data[f"{zone.lower()}_gfs_{variable}"] = []
            for hour in hours[:-1]:
                current = decoded_by_hour[hour]
                solar, clipped = hourly_radiation(current["solar"], decoded_by_hour[hour + 1]["solar"], radiation_tolerance)
                clipped_total += clipped
                temperature = current["temperature_2m"].values - 273.15
                wind = np.hypot(current["u100"].values, current["v100"].values)
                for zone, positions in point_slices.items():
                    for name, values in (("temperature_2m_c", temperature), ("wind_speed_100m_ms", wind), ("shortwave_radiation_wm2", solar)):
                        data[f"{zone.lower()}_gfs_{name}"].append(float(values[positions].mean()))
            frame = pd.DataFrame(data)
            if len(frame) != len(index) or not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
                raise GfsError("Output day is incomplete or contains non-finite features.")
            frames.append(frame)
            records.extend(record for _, _, record in endpoints)
            print(f"[GFS] {day:%Y-%m-%d}: {len(index)} physical hours; {len(hours)} forecast endpoints; publication={publication}", flush=True)
    finally:
        client.close()
    dataset = pd.concat(frames, ignore_index=True)
    for column in ("delivery_start_utc", "run_init_utc", "cutoff_utc", "publication_max_utc"):
        dataset[column] = pd.to_datetime(dataset[column], utc=True).astype("datetime64[ns, UTC]")
    if dataset["delivery_start_utc"].duplicated().any():
        raise GfsError("Duplicate physical delivery hours.")
    output.parent.mkdir(parents=True, exist_ok=True)
    successful_download_bytes = sum(item["bytes"] for record in records if not record["from_cache"] for item in record["messages"].values())
    temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
    manifest_temporary = manifest.with_name(manifest.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        dataset.to_parquet(temporary, index=False)
        pd.testing.assert_frame_equal(dataset, pd.read_parquet(temporary))
        dataset_sha = sha256(temporary.read_bytes())
        result = {"schema_version": SCHEMA_VERSION, "source": "NOAA GFS 0.25 degree original operational 00Z D-1", "source_license": "NOAA NODD public use with attribution", "source_documentation": "https://registry.opendata.aws/noaa-gfs-bdp-pds/", "evidence_kind": "historical_archive_publication", "local_prospective_capture": False, "production_pit_evidence": False, "production_pipeline_evidence": False, "promotion_eligible": False, "output_path": str(output), "dataset_sha256": dataset_sha, "start_day": start_day, "end_day": end_day, "row_count": len(dataset), "day_count": len(days), "zones": list(zones), "timezone": "Europe/Paris", "cutoff_policy": "D-1 08:00 Europe/Paris", "run_policy": "D-1 00:00 UTC; no cycle substitution", "feature_policy": "instant temperature and wind at hour start; deaveraged solar mean over [hour,hour+1); arithmetic mean of fixed nearest spatial samples", "points": {z: [vars(p) for p in ZONE_POINTS[z]] for z in zones}, "feature_units": {"temperature_2m_c": "degC", "wind_speed_100m_ms": "m/s", "shortwave_radiation_wm2": "W/m2"}, "radiation_negative_tolerance_wm2": radiation_tolerance, "radiation_clipped_point_hours": clipped_total, "forecast_endpoint_count": len(records), "sources": records}
        # Match the existing Chronos feature-bank sidecar contract, without
        # claiming that archive publication is a prospective local capture.
        result["output_sha256"] = dataset_sha
        result["cutoff_time"] = "08:00"
        result["cutoff_timezone"] = "Europe/Paris"
        result["successful_message_download_bytes"] = successful_download_bytes
        result["cached_endpoint_count"] = sum(bool(record["from_cache"]) for record in records)
        result["materializer_source_sha256"] = sha256(Path(__file__).read_bytes())
        manifest_payload = _json_bytes(result)
        with manifest_temporary.open("xb") as target:
            target.write(manifest_payload)
        if manifest_temporary.read_bytes() != manifest_payload:
            raise GfsError("Staged manifest roundtrip differs from the serialized result.")
        _publish_pair(temporary, manifest_temporary, output, manifest)
    finally:
        for path in (temporary, manifest_temporary):
            if path.exists():
                path.unlink()
    return {"output_path": str(output), "manifest_path": str(manifest), "dataset_sha256": dataset_sha, "row_count": len(dataset), "day_count": len(days), "radiation_clipped_point_hours": clipped_total, "successful_message_download_bytes": successful_download_bytes}
