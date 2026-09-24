"""Audited native day-ahead quarter-hour observations, isolated from NYX.

Labels are read at their latest available vintage. This is not a historical
publication archive or proof of availability at a forecast origin.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import numpy as np
import pandas as pd
import requests

ZONES = ("BE", "DE", "FR", "NL")
SATURN_URL = "https://saturn-energyscan.gem.myengie.com//api"
PRIMARY_IDS = {"BE": "60451", "DE": "60452", "FR": "60454", "NL": "60453"}
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "data/pit/nyx_quarterhour"
SOURCES = [{"zone": z, "series": PRIMARY_IDS[z], "kind": "primary", "catalog_source": "power",
            "unit": "EUR/MWh", "timezone": "UTC", "native_resolution_minutes": 15,
            "interpolation": "none", "is_forecast": False,
            "market": "day_ahead", "published_alias": f"power.price.{z.lower()}.euromwh.qh.obs.epex",
            "native_resolution_evidence": "Primary UTC-aware series with null formula, audited 2026-09-16; direct raw quarter-hour grid across autumn DST, without resample or ffill."}
           for z in ZONES]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain_path(path: Path) -> Path:
    """Reject every existing redirected component, including Windows junctions."""
    path = Path(os.path.abspath(path))
    if path.resolve() != path:
        raise ValueError("A source/archive path must not follow a symlink or junction.")
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlinks, junctions and reparse points are not permitted.")
    return path


def safe_output(path: Path) -> Path:
    namespace = plain_path(ARCHIVE_ROOT)
    path = plain_path(path)
    if not path.is_relative_to(namespace) or path == namespace:
        raise ValueError("Use a dedicated child of data/pit/nyx_quarterhour.")
    return path


def civil_day(value: str) -> date:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("An ISO calendar date YYYY-MM-DD is required.")
    return date.fromisoformat(value)


def quarter_grid(start_day: str, end_day: str) -> pd.DatetimeIndex:
    first, last = civil_day(start_day), civil_day(end_day)
    if first > last or (last-first).days > 731:
        raise ValueError("Use an ordered window of at most 732 civil days.")
    begin = pd.Timestamp(first).tz_localize("Europe/Paris").tz_convert("UTC")
    finish = pd.Timestamp(last+timedelta(days=1)).tz_localize("Europe/Paris").tz_convert("UTC")
    return pd.date_range(begin, finish, freq="15min", inclusive="left", name="timestamp_utc")


class ReadOnlySession(requests.Session):
    def request(self, method, url, *args, **kwargs):
        if method.upper() != "GET":
            raise ValueError("Only read-only GET requests are allowed.")
        kwargs.setdefault("timeout", (10, 120))
        return super().request(method, url, *args, **kwargs)


def make_client():
    from tshistory_lite import Client
    client = Client(SATURN_URL, author="nyx-quarterhour-readonly")
    client.session.close()
    client.session = ReadOnlySession()
    return client


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({"timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                         "zone": pd.Series(dtype="str"), "actual_15m": pd.Series(dtype="float64")})


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) != {"timestamp_utc", "zone", "actual_15m"} or frame.columns.has_duplicates:
        raise ValueError("Native price columns must be timestamp_utc, zone, actual_15m.")
    frame = frame.copy()
    index = pd.DatetimeIndex(frame.timestamp_utc)
    if index.tz is None or index.hasnans:
        raise ValueError("Timezone-aware, finite physical timestamps are required.")
    index = index.tz_convert("UTC")
    if not index.equals(index.floor("15min")):
        raise ValueError("Prices must lie on the exact 15-minute grid; no rounding.")
    frame["timestamp_utc"] = index
    if not frame.zone.isin(ZONES).all() or frame.duplicated(["zone", "timestamp_utc"]).any():
        raise ValueError("Countries and unique country/timestamp identities are required.")
    numeric = pd.to_numeric(frame.actual_15m, errors="raise")
    if np.iscomplexobj(numeric) or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Prices must be finite real values; do not impute missing labels.")
    frame["actual_15m"] = numeric.astype(float)
    return frame.sort_values(["timestamp_utc", "zone"]).reset_index(drop=True)


def fetch_range(client: Any, zone: str, start_day: str, end_day: str) -> tuple[pd.DataFrame, dict]:
    if zone not in PRIMARY_IDS:
        raise ValueError("Only BE, DE, FR and NL primary source contracts are supported.")
    if civil_day(start_day) < date(2025, 10, 1):
        raise ValueError("The audited native day-ahead quarter-hour window starts 2025-10-01.")
    grid = quarter_grid(start_day, end_day)
    audit = {"zone": zone, "series": PRIMARY_IDS[zone], "start_day": start_day, "end_day": end_day,
             "query_from_utc": grid[0].isoformat(), "query_to_utc": grid[-1].isoformat(),
             "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
             "price_vintage": "latest_observations", "provider_revision_timestamp_available": False,
             "expected_quarters": len(grid), "expected_hours": len(grid)//4, "interpolation": "none"}
    try:
        raw = client.get(PRIMARY_IDS[zone], from_value_date=grid[0], to_value_date=grid[-1], _keep_nans=True)
        if raw is None or len(raw) == 0:
            return empty_frame(), {**audit, "status": "empty", "returned_quarters": 0}
        if not isinstance(raw, pd.Series):
            raise ValueError("Expected a raw primary series.")
        index = pd.DatetimeIndex(raw.index)
        if index.tz is None or index.hasnans or index.has_duplicates:
            raise ValueError("Primary prices require unique timezone-aware timestamps.")
        index = index.tz_convert("UTC")
        if not index.equals(index.floor("15min")):
            raise ValueError("Off-grid native labels cannot be repaired.")
        numeric = pd.to_numeric(raw, errors="raise")
        if np.iscomplexobj(numeric):
            raise ValueError("Complex prices are not allowed.")
        series = pd.Series(numeric.to_numpy(float), index=index).sort_index()
        outside = int((~series.index.isin(grid)).sum())
        series = series.loc[series.index.isin(grid)]
        if len(series) and set(series.index.minute) == {0}:
            raise ValueError("The source returns hourly prices, not native quarters.")
        missing_nonfinite = int((~np.isfinite(series.to_numpy())).sum())
        # Exclude nonfinite labels explicitly; no old revision or interpolated value is substituted.
        series = series.loc[np.isfinite(series.to_numpy())]
        frame = validate_frame(pd.DataFrame({"timestamp_utc": series.index, "zone": zone,
                                             "actual_15m": series.to_numpy()}))
        groups = series.groupby(series.index.floor("h"))
        complete = int(groups.size().eq(4).sum())
        varying = int(sum(len(g) == 4 and g.nunique() > 1 for _, g in groups))
        by_day = series.groupby(series.index.tz_convert("Europe/Paris").strftime("%Y-%m-%d")).size().to_dict()
        expected_days = pd.Series(1, index=grid).groupby(grid.tz_convert("Europe/Paris").strftime("%Y-%m-%d")).size().to_dict()
        days = [{"day": day, "expected_quarters": int(n), "actual_quarters": int(by_day.get(day, 0))}
                for day, n in expected_days.items()]
        return frame, {**audit, "status": "complete" if len(frame) == len(grid) else "partial",
                       "returned_quarters": len(frame), "missing_quarters": len(grid)-len(frame),
                       "nonfinite_quarters_excluded": missing_nonfinite, "outside_quarters_excluded": outside,
                       "complete_hours": complete, "varying_hours": varying, "days": days}
    except Exception as exc:
        # Never serialize response text or credential-bearing proxy URLs.
        return empty_frame(), {**audit, "status": "error", "error_type": type(exc).__name__, "returned_quarters": 0}


def hourly_means(frame: pd.DataFrame) -> pd.DataFrame:
    frame = validate_frame(frame)
    frame["timestamp_utc"] = frame.timestamp_utc.dt.floor("h")
    grouped = frame.groupby(["timestamp_utc", "zone"], sort=True).actual_15m.agg(["size", "mean"])
    return grouped.loc[grouped["size"].eq(4), ["mean"]].rename(columns={"mean": "actual_hourly_from_15m"}).reset_index()


def read_native_prices(manifest_path: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = plain_path(Path(manifest_path))
    if manifest_path.stat().st_size > 1024*1024:
        raise ValueError("Source manifest exceeds the bounded schema size.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "nyx_quarterhour_price_observations"
            or manifest.get("timezone") != "UTC" or manifest.get("unit") != "EUR/MWh"
            or manifest.get("resolution_minutes") != 15 or manifest.get("interpolation") != "none"
            or manifest.get("price_vintage") != "latest_observations" or manifest.get("status") not in {"complete", "partial"}
            or manifest.get("provider_revision_timestamp_available") is not False):
        raise ValueError("Invalid native-quarter-hour observed-price manifest contract.")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 4 or {s.get("zone") for s in sources if isinstance(s, dict)} != set(ZONES):
        raise ValueError("Four country source declarations are required.")
    for source in sources:
        if (source.get("series") != PRIMARY_IDS[source["zone"]] or source.get("native_resolution_minutes") != 15
                or source.get("interpolation") != "none" or source.get("kind") != "primary"
                or source.get("unit") != "EUR/MWh" or source.get("timezone") != "UTC"
                or source.get("is_forecast") is not False):
            raise ValueError("Source declaration differs from the audited primary contract.")
    name = manifest.get("data_file")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".parquet"):
        raise ValueError("data_file must name a sibling parquet without traversal.")
    path = plain_path(manifest_path.parent/name)
    if path.parent != manifest_path.parent or path.stat().st_size > 256*1024*1024:
        raise ValueError("Source data must be a bounded sibling parquet.")
    if digest(path) != manifest.get("data_sha256"):
        raise ValueError("Native source parquet checksum mismatch.")
    frame = validate_frame(pd.read_parquet(path))
    grid = quarter_grid(manifest["start_day"], manifest["end_day"])
    if not frame.timestamp_utc.isin(grid).all():
        raise ValueError("Source contains observations outside its declared window.")
    if manifest["status"] == "complete" and any(
            not pd.DatetimeIndex(frame.loc[frame.zone.eq(z), "timestamp_utc"]).equals(grid) for z in ZONES):
        raise ValueError("A complete archive requires every physical quarter in all four countries.")
    return frame, {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": digest(manifest_path),
                   "data_path": str(path), "production_pit_evidence": False,
                   "vintage_warning": "Latest observed labels, not historical publication vintages. Retrospective research only."}
