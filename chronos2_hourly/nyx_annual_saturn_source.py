"""Daily as-of Saturn inputs for the annual CPU pipeline.

Each forecast profile and target history is queried at its own civil cutoff.
The immutable cache is resumable and does not read production residual banks.
Saturn revision_date evidence is recorded separately from provider publication
evidence; downloading an old state never certifies the provider's publication.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np
import pandas as pd

from chronos2_modular.saturn import create_saturn_client, fetch_saturn_series_from_client
from .nyx_annual_live_preflight import ROOT, SOURCE_PROTOCOL, sha256, validate_source_receipt
from .nyx_local_io import publish_verified_immutable_copy
from .process_lock import exclusive_process_lock

PROTOCOL = "nyx_annual_saturn_daily_v1"
ZONES = ("FR", "DE", "BE", "NL")
TZ = {"FR": "Europe/Paris", "DE": "Europe/Berlin", "BE": "Europe/Brussels",
      "NL": "Europe/Amsterdam", "ES": "Europe/Madrid"}
CORE_ALIASES = tuple(f"{z}_residual_load_fcst" for z in ("fr", "de", "be", "nl", "es")) + (
    "fr_nuclear_generation_fcst_gw",
) + tuple(f"{z}_solar_generation_fcst" for z in ("fr", "de", "be", "nl")) + (
    "de_wind_generation_fcst", "nl_wind_generation_fcst")
ALIASES = CORE_ALIASES + ("fr_wind_generation_fcst", "be_wind_generation_fcst")
DEFAULT_CACHE = ROOT / "data/pit/nyx_annual_saturn"
WARMUP_DAYS = 834
TARGET_CONTEXT_HOURS = 479 * 24 + 2  # 469 baseline/reference days + price lags and DST.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def cutoff(day: str) -> pd.Timestamp:
    d = date.fromisoformat(day)
    require(d.isoformat() == day, "Expected YYYY-MM-DD")
    return (pd.Timestamp(d - timedelta(days=1)) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")


def grid(day: str) -> pd.DatetimeIndex:
    d = date.fromisoformat(day)
    return pd.date_range(pd.Timestamp(d, tz="Europe/Paris"),
                         pd.Timestamp(d + timedelta(days=1), tz="Europe/Paris"),
                         freq="h", inclusive="left").tz_convert("UTC").rename("timestamp_utc")


def specs() -> dict:
    from materialize_saturn_kalman_fuel import RESIDUAL_LOAD_SERIES
    result = {s.alias: {"series": s.series, "naive_timezone": s.naive_timezone,
                       "dst_policy": s.incomplete_dst_policy, "unit": "GW"}
              for s in RESIDUAL_LOAD_SERIES}
    result["fr_nuclear_generation_fcst_gw"] = {
        "series": "power.fr.generation.nuclear.gw.fcst", "naive_timezone": TZ["FR"],
        "dst_policy": "duplicate", "unit": "GW"}
    for zone in ZONES:
        for kind in ("solar", "wind"):
            alias = f"{zone.lower()}_{kind}_generation_fcst"
            result[alias] = {"series": f"power.{zone.lower()}.generation.{kind}.hourly.gw.fcst",
                             "naive_timezone": TZ[zone], "unit": "GW",
                             "dst_policy": "duplicate_zero_only" if alias == "nl_solar_generation_fcst" else "duplicate"}
    return {a: result[a] for a in ALIASES}


def _encoded(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _immutable(path: Path, raw: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == raw, f"Immutable Saturn artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as f:
        temp = Path(f.name)
        f.write(raw)
    try:
        require(not path.exists(), f"Concurrent Saturn artifact: {path}")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _parquet(path: Path, frame: pd.DataFrame):
    import io
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    _immutable(path, buffer.getvalue())


def _numeric(series, expected, label):
    require(isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None
            and series.index.is_unique, f"{label}: ambiguous source timestamps")
    series = series.copy()
    series.index = series.index.tz_convert("UTC")
    values = pd.to_numeric(series.reindex(expected), errors="coerce").to_numpy(float)
    require(np.isfinite(values).all(), f"{label}: {int((~np.isfinite(values)).sum())} missing/nonfinite hours at its cutoff")
    return values


def verify_day(directory: Path, day: str, *, expected_specs: dict | None = None):
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    require(receipt.get("protocol") == PROTOCOL and receipt.get("delivery_day") == day
            and receipt.get("forecast_origin_utc") == cutoff(day).isoformat()
            and receipt.get("state") == "COMPLETE"
            and receipt.get("series") == (expected_specs or specs()), "Saturn daily cache contract changed")
    require(set(receipt.get("artifact_sha256", {})) == {"covariates.parquet", "prices.parquet"}, "Saturn artifact inventory differs")
    for name, digest in receipt["artifact_sha256"].items():
        require(sha256(directory / name) == digest, f"Saturn day {day}: modified {name}")
    cov = pd.read_parquet(directory / "covariates.parquet")
    prices = pd.read_parquet(directory / "prices.parquet")
    expected = grid(day)
    history = pd.date_range(expected[0] - pd.Timedelta(hours=TARGET_CONTEXT_HOURS),
                            expected[0], freq="h", inclusive="left", name="timestamp_utc")
    require(cov.index.equals(expected) and tuple(cov.columns) == ALIASES
            and np.isfinite(cov.to_numpy(float)).all(), "Invalid Saturn daily forecast grid")
    require(prices.index.equals(history) and tuple(prices.columns) == ZONES
            and np.isfinite(prices.to_numpy(float)).all(), "Invalid Saturn daily target history")
    require(pd.Timestamp(receipt["retrieved_at_utc"]).tz_convert("UTC") >= cutoff(day),
            "Saturn cutoff had not occurred at retrieval")
    return cov, prices, receipt


def capture_day(client, day: str, cache: Path = DEFAULT_CACHE, *, now_utc=None):
    from run_nyx_annual_auction_prices_source import SERIES
    directory = Path(cache) / day
    contract = specs()
    if (directory / "receipt.json").exists():
        return verify_day(directory, day, expected_specs=contract)[2]
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    require(now.tzinfo is not None and now >= cutoff(day), "Saturn D-1 08:00 cutoff not reached")
    expected = grid(day)
    values, repair_evidence = {}, {}
    for alias, spec in contract.items():
        series = fetch_saturn_series_from_client(client, spec["series"],
            expected[0] - pd.Timedelta(hours=8), expected[-1] + pd.Timedelta(hours=8), "UTC",
            revision_date=cutoff(day), naive_timezone=spec["naive_timezone"],
            incomplete_dst_policy=spec["dst_policy"], nocache=True)
        values[alias] = _numeric(series, expected, alias)
        # Saturn's normalizer explicitly records duplicated autumn folds.
        repair_evidence[alias] = {"policy": spec["dst_policy"],
                                  "normalizer_attrs": json.loads(json.dumps(series.attrs, default=str))}
    cov = pd.DataFrame(values, index=expected)
    history = pd.date_range(expected[0] - pd.Timedelta(hours=TARGET_CONTEXT_HOURS),
                            expected[0], freq="h", inclusive="left", name="timestamp_utc")
    targets = {}
    for zone in ZONES:
        series = fetch_saturn_series_from_client(client, SERIES[zone], history[0], history[-1], "UTC",
            revision_date=cutoff(day), naive_timezone="UTC", incomplete_dst_policy="raise", nocache=True)
        targets[zone] = _numeric(series, history, f"{zone} target")
    prices = pd.DataFrame(targets, index=history)
    receipt = {"protocol": PROTOCOL, "delivery_day": day, "state": "COMPLETE",
        "forecast_origin_utc": cutoff(day).isoformat(), "series": contract,
        "target_series": dict(SERIES), "retrieved_at_utc": (pd.Timestamp.now(tz="UTC") if now_utc is None else now).isoformat(),
        "publication_verified": False, "provider_publication_timestamp_verified": False,
        "availability_basis": "Saturn revision_date at each delivery D-1 08:00",
        "target_future_labels_used": False, "dst_evidence": repair_evidence,
        "collector_code_sha256": sha256(Path(__file__))}
    for name, frame in (("covariates.parquet", cov), ("prices.parquet", prices)):
        _parquet(directory / name, frame)
    receipt["artifact_sha256"] = {n: sha256(directory / n) for n in ("covariates.parquet", "prices.parquet")}
    _immutable(directory / "receipt.json", _encoded(receipt))
    verify_day(directory, day)
    return receipt


def sync(delivery_day: str, *, first_day: str | None = None, cache: Path = DEFAULT_CACHE,
         workers: int = 2, client_factory: Callable | None = None):
    from run_nyx_annual_auction_prices_source import load_plan
    require(type(workers) is int and 1 <= workers <= 8, "Use 1-8 source workers")
    last = date.fromisoformat(delivery_day)
    first = source_start(cache, delivery_day, first_day)
    require(first <= last, "Source dates reversed")
    require(pd.Timestamp.now(tz="UTC") >= cutoff(delivery_day), "Latest source cutoff has not occurred")
    plan = load_plan()
    if client_factory is None:
        def client_factory():
            from functools import partial
            client = create_saturn_client(plan["saturn_url"], os.getenv("SATURN_AUTHOR") or plan["saturn_author"])
            client.session.request = partial(client.session.request, timeout=60)
            return client
    days = [x.date().isoformat() for x in pd.date_range(first, last, freq="D")]
    def one(day):
        if (Path(cache) / day / "receipt.json").exists():
            return verify_day(Path(cache) / day, day)[2]
        client = client_factory()
        try:
            return capture_day(client, day, cache)
        finally:
            session = getattr(client, "session", None)
            if session is not None:
                session.close()
    results = {}
    with exclusive_process_lock(Path(cache) / "sync.lock"):
        _immutable(Path(cache) / "plan.json", _encoded({"protocol": PROTOCOL, "first_delivery_day": first.isoformat()}))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(one, d): d for d in days}
            for future in as_completed(pending):
                d = pending[future]
                try:
                    results[d] = future.result()
                except Exception:
                    for remaining in pending:
                        remaining.cancel()
                    raise
                if len(results) % 10 == 0 or len(results) == len(days):
                    print(f"Saturn: {len(results)}/{len(days)} daily states verified", flush=True)
    return results


def source_start(cache: Path, delivery_day: str, first_day: str | None = None):
    """Keep the bootstrap origin stable as the live cache grows day by day."""
    last = date.fromisoformat(delivery_day)
    plan_path = Path(cache) / "plan.json"
    requested = date.fromisoformat(first_day) if first_day else last - timedelta(days=WARMUP_DAYS)
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        require(plan.get("protocol") == PROTOCOL, "Saturn source cache plan changed")
        first = date.fromisoformat(plan["first_delivery_day"])
        require(first <= requested, "Requested deeper history requires a separate source cache")
        return first
    return requested


def publish(bundle: Path, delivery_day: str, *, first_day: str | None = None,
            cache: Path = DEFAULT_CACHE):
    last = date.fromisoformat(delivery_day)
    first = source_start(cache, delivery_day, first_day)
    require(first <= last - timedelta(days=365), "Saturn training history too short")
    frames, daily, hashes = [], {}, {}
    bundle = Path(bundle).resolve()
    for stamp in pd.date_range(first, last, freq="D"):
        day = stamp.date().isoformat()
        directory = Path(cache) / day
        cov, _, receipt = verify_day(directory, day)
        frames.append(cov)
        for name in ("covariates.parquet", "prices.parquet", "receipt.json"):
            relative = f"source_artifacts/saturn/days/{day}/{name}"
            digest = sha256(directory / name)
            publish_verified_immutable_copy(directory / name, bundle / relative, digest)
            hashes[relative] = digest
        daily[day] = {"forecast_origin_utc": receipt["forecast_origin_utc"],
                      "covariates_sha256": receipt["artifact_sha256"]["covariates.parquet"],
                      "prices_sha256": receipt["artifact_sha256"]["prices.parquet"],
                      "publication_verified": False,
                      "receipt_path": f"source_artifacts/saturn/days/{day}/receipt.json"}
    relative = "source_artifacts/saturn/covariates.parquet"
    combined = pd.concat(frames)
    _parquet(bundle / relative, combined)
    hashes[relative] = sha256(bundle / relative)
    origins = pd.DataFrame({"forecast_origin_utc": [cutoff(str(d)) for d in combined.index.tz_convert("Europe/Paris").date]}, index=combined.index)
    origin_path = "source_artifacts/saturn/origins.parquet"
    _parquet(bundle / origin_path, origins)
    hashes[origin_path] = sha256(bundle / origin_path)
    receipt = {"protocol": SOURCE_PROTOCOL, "source_group": "saturn", "delivery_day": delivery_day,
        "state": "COMPLETE", "asof_cutoff_verified": True, "training_window_complete": True,
        "asof_state_utc": cutoff(delivery_day).isoformat(), "daily_vintages": daily,
        "first_delivery_day": first.isoformat(), "last_delivery_day": delivery_day,
        "availability_basis": "Each profile and price history queried with its own Saturn revision_date",
        "provider_publication_timestamp_verified": False,
        "historical_publication_vintages_certified": False,
        "publication_vintage_limit": "Query-as-of is not certified original provider insertion time",
        "artifact_sha256": hashes, "series": specs(), "collector_code_sha256": sha256(Path(__file__))}
    validate_source_receipt(receipt, group="saturn", day=delivery_day, bundle=bundle, cutoff=cutoff(delivery_day))
    path = bundle / "source_receipts/saturn.json"
    _immutable(path, _encoded(receipt))
    return path


def load_target_snapshots(bundle: Path):
    bundle = Path(bundle)
    receipt = json.loads((bundle / "source_receipts/saturn.json").read_text(encoding="utf-8"))
    def load(day):
        day = str(pd.Timestamp(day).date())
        expected = receipt["daily_vintages"][day]["prices_sha256"]
        path = bundle / "source_artifacts/saturn/days" / day / "prices.parquet"
        require(sha256(path) == expected, f"Changed historical price snapshot: {day}")
        frame = pd.read_parquet(path)
        return {zone: frame[zone].rename("target") for zone in ZONES}
    return load
