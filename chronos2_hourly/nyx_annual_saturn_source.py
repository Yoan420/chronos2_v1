"""Daily as-of Saturn inputs for the annual CPU pipeline.

Forecast profiles retain their historical civil cutoffs. Training prices for
an outer forecast use the canonical snapshot at that outer forecast's cutoff;
inner reconstruction windows never receive prices for their delivery day.
Legacy packets with per-origin price snapshots remain readable.
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

from experiment_console.security import redact_text

from chronos2_modular.saturn import create_saturn_client, fetch_saturn_series_from_client
from .nyx_annual_live_preflight import ROOT, SOURCE_PROTOCOL, sha256, validate_source_receipt
from .nyx_local_io import publish_verified_immutable_copy
from .process_lock import exclusive_process_lock

PROTOCOL = "nyx_annual_saturn_daily_v1"
PROFILE_PROTOCOL = "nyx_annual_saturn_profiles_daily_v2"
PROFILE_NORMALIZATION_POLICY = "historical_saturn_profiles_with_audited_nl_spring_v1"
TARGET_PROTOCOL = "nyx_annual_saturn_current_fit_targets_v1"
TARGET_HISTORY_POLICY = "current_fit_origin_reconstruction_v1"
TARGET_HISTORY_FIELDS = ("target_history_policy", "target_revision_utc",
                         "target_origin_snapshot_verified", "target_future_labels_used")
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


class SaturnSourceError(RuntimeError):
    """Context for one failed day; the original exception remains its cause."""

    def __init__(self, day, phase, error, *, alias=None, series=None):
        self.day, self.phase, self.alias, self.series = day, phase, alias, series
        context = f"Saturn {day} / {phase}"
        if alias:
            context += f" / {alias}"
        if series:
            context += f" ({series})"
        super().__init__(f"{context}: {type(error).__name__}: {redact_text(str(error))}")


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
    missing = ~np.isfinite(values)
    first = ", ".join(stamp.isoformat() for stamp in expected[missing][:5])
    require(not missing.any(), f"{label}: {int(missing.sum())} missing/nonfinite hours at its cutoff; first UTC: {first}")
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
    """Legacy producer retained for older strict daily packets."""
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
        try:
            series = fetch_saturn_series_from_client(client, spec["series"],
                expected[0] - pd.Timedelta(hours=8), expected[-1] + pd.Timedelta(hours=8), "UTC",
                revision_date=cutoff(day), naive_timezone=spec["naive_timezone"],
                incomplete_dst_policy=spec["dst_policy"], nocache=True)
            values[alias] = _numeric(series, expected, alias)
        except Exception as error:
            raise SaturnSourceError(day, "forecast_profile", error, alias=alias, series=spec["series"]) from error
        # Saturn's normalizer explicitly records duplicated autumn folds.
        repair_evidence[alias] = {"policy": spec["dst_policy"],
                                  "normalizer_attrs": json.loads(json.dumps(series.attrs, default=str))}
    cov = pd.DataFrame(values, index=expected)
    history = pd.date_range(expected[0] - pd.Timedelta(hours=TARGET_CONTEXT_HOURS),
                            expected[0], freq="h", inclusive="left", name="timestamp_utc")
    targets = {}
    for zone in ZONES:
        try:
            series = fetch_saturn_series_from_client(client, SERIES[zone], history[0], history[-1], "UTC",
                revision_date=cutoff(day), naive_timezone="UTC", incomplete_dst_policy="raise", nocache=True)
            targets[zone] = _numeric(series, history, f"{zone} target")
        except Exception as error:
            raise SaturnSourceError(day, "target_history", error, alias=f"{zone}_target", series=SERIES[zone]) from error
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


def _target_contract(day):
    return {"target_history_policy": TARGET_HISTORY_POLICY,
            "target_revision_utc": cutoff(day).isoformat(),
            "target_origin_snapshot_verified": False, "target_future_labels_used": False}


def _check_target_contract(receipt, day):
    expected = _target_contract(day)
    require(all(receipt.get(key) == value for key, value in expected.items())
            and receipt.get("target_origin_snapshot_verified") is False
            and receipt.get("target_future_labels_used") is False,
            "Saturn current-fit target policy or outer revision differs")
    return expected


def target_grid(first_profile_day, delivery_day):
    require(date.fromisoformat(first_profile_day) <= date.fromisoformat(delivery_day), "Saturn profile dates reversed")
    first = grid(first_profile_day)[0] - pd.Timedelta(hours=TARGET_CONTEXT_HOURS)
    end = grid(delivery_day)[0]
    require(first < end, "Saturn target snapshot dates reversed")
    return pd.date_range(first, end, freq="h", inclusive="left", name="timestamp_utc")


def _nl_spring_profile(series, day, expected):
    from materialize_saturn_kalman_fuel import RESIDUAL_LOAD_SERIES, _repair_nl_spring_dst_hour
    spec = next(item for item in RESIDUAL_LOAD_SERIES if item.alias == "nl_residual_load_fcst")
    selected = series.loc[(series.index >= expected[0]) & (series.index <= expected[-1])].copy()
    selected.index = selected.index.tz_convert(spec.delivery_timezone)
    repaired, stamps = _repair_nl_spring_dst_hour(selected, spec=spec, day=pd.Timestamp(day), expected=expected)
    ledger = []
    values = repaired.copy()
    values.index = values.index.tz_convert("UTC")
    for stamp in stamps:
        donors = [stamp - pd.Timedelta(hours=1), stamp + pd.Timedelta(hours=1)]
        ledger.append({"value_time_utc": stamp.isoformat(),
                       "value_time_local": stamp.tz_convert(spec.delivery_timezone).isoformat(),
                       "donor_hours_utc": [donor.isoformat() for donor in donors],
                       "donor_values_gw": [float(values.loc[donor]) for donor in donors],
                       "repaired_value_gw": float(values.loc[stamp]),
                       "forecast_origin_utc": cutoff(day).isoformat()})
    return repaired, ledger


def _verify_profile_normalization(frame, receipt, day):
    from materialize_saturn_kalman_fuel import NL_SPRING_DST_REPAIR_POLICY
    require(receipt.get("profile_normalization_policy") == PROFILE_NORMALIZATION_POLICY,
            "Saturn profile normalization policy differs")
    evidence = receipt.get("spring_dst_repair", {})
    require(evidence.get("alias") == "nl_residual_load_fcst"
            and evidence.get("policy") == NL_SPRING_DST_REPAIR_POLICY
            and isinstance(evidence.get("entries"), list), "Saturn NL spring repair evidence missing")
    entries = evidence["entries"]
    if not entries:
        return
    expected = grid(day)
    require(len(expected) == 23 and len(entries) == 2, "Saturn NL spring repair only permits two hours on a 23-hour day")
    stamps = pd.DatetimeIndex([pd.Timestamp(item["value_time_utc"]) for item in entries])
    require(stamps.tz is not None and str(stamps.tz) == "UTC"
            and all(stamp.date() == date.fromisoformat(day) for stamp in stamps.tz_convert("Europe/Amsterdam"))
            and tuple(stamps.tz_convert("Europe/Amsterdam").strftime("%H:%M")) == ("04:00", "06:00"),
            "Saturn NL spring repair hours differ")
    raw = frame["nl_residual_load_fcst"].drop(stamps)
    reconstructed, ledger = _nl_spring_profile(raw, day, expected)
    require(entries == ledger, "Saturn NL spring repair ledger differs from same-vintage donors")
    require(np.array_equal(_numeric(reconstructed, expected, "nl_residual_load_fcst"),
                           frame["nl_residual_load_fcst"].to_numpy(float)),
            "Saturn NL spring repaired values differ from historical normalization")


def verify_profile_day(directory: Path, day: str):
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    require(receipt.get("protocol") == PROFILE_PROTOCOL and receipt.get("delivery_day") == day
            and receipt.get("forecast_origin_utc") == cutoff(day).isoformat()
            and receipt.get("state") == "COMPLETE" and receipt.get("series") == specs(),
            "Saturn profile cache contract changed")
    require(set(receipt.get("artifact_sha256", {})) == {"covariates.parquet"},
            "Saturn profile artifact inventory differs")
    require(sha256(directory / "covariates.parquet") == receipt["artifact_sha256"]["covariates.parquet"],
            f"Saturn profile {day}: modified covariates.parquet")
    frame = pd.read_parquet(directory / "covariates.parquet")
    require(frame.index.equals(grid(day)) and tuple(frame.columns) == ALIASES
            and np.isfinite(frame.to_numpy(float)).all(), "Invalid Saturn daily forecast grid")
    _verify_profile_normalization(frame, receipt, day)
    require(pd.Timestamp(receipt["retrieved_at_utc"]).tz_convert("UTC") >= cutoff(day),
            "Saturn profile cutoff had not occurred at retrieval")
    return frame, receipt


def capture_profile_day(client, day: str, cache: Path = DEFAULT_CACHE, *, now_utc=None):
    directory = Path(cache) / "profiles_v2" / day
    if (directory / "receipt.json").is_file():
        return verify_profile_day(directory, day)[1]
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    require(now.tzinfo is not None and now >= cutoff(day), "Saturn D-1 08:00 cutoff not reached")
    from materialize_saturn_kalman_fuel import NL_SPRING_DST_REPAIR_POLICY
    expected, contract, values, evidence, spring_ledger = grid(day), specs(), {}, {}, []
    for alias, spec in contract.items():
        try:
            series = fetch_saturn_series_from_client(client, spec["series"],
                expected[0] - pd.Timedelta(hours=8), expected[-1] + pd.Timedelta(hours=8), "UTC",
                revision_date=cutoff(day), naive_timezone=spec["naive_timezone"],
                incomplete_dst_policy=spec["dst_policy"], nocache=True)
            selected = series
            if alias == "nl_residual_load_fcst":
                selected, spring_ledger = _nl_spring_profile(series, day, expected)
            values[alias] = _numeric(selected, expected, alias)
        except Exception as error:
            raise SaturnSourceError(day, "forecast_profile", error, alias=alias, series=spec["series"]) from error
        evidence[alias] = {"policy": spec["dst_policy"],
                           "normalizer_attrs": json.loads(json.dumps(series.attrs, default=str))}
    _parquet(directory / "covariates.parquet", pd.DataFrame(values, index=expected))
    receipt = {"protocol": PROFILE_PROTOCOL, "delivery_day": day, "state": "COMPLETE",
        "forecast_origin_utc": cutoff(day).isoformat(), "series": contract,
        "retrieved_at_utc": (pd.Timestamp.now(tz="UTC") if now_utc is None else now).isoformat(),
        "publication_verified": False, "provider_publication_timestamp_verified": False,
        "availability_basis": "Forecast profiles queried at their own delivery D-1 08:00 revision_date",
        "profile_normalization_policy": PROFILE_NORMALIZATION_POLICY,
        "spring_dst_repair": {"alias": "nl_residual_load_fcst", "policy": NL_SPRING_DST_REPAIR_POLICY,
                              "entries": spring_ledger},
        "dst_evidence": evidence, "collector_code_sha256": sha256(Path(__file__)),
        "artifact_sha256": {"covariates.parquet": sha256(directory / "covariates.parquet")}}
    _immutable(directory / "receipt.json", _encoded(receipt))
    verify_profile_day(directory, day)
    return receipt


def verify_target_snapshot(directory: Path, delivery_day: str, first_profile_day: str):
    from run_nyx_annual_auction_prices_source import SERIES
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    _check_target_contract(receipt, delivery_day)
    require(receipt.get("protocol") == TARGET_PROTOCOL and receipt.get("state") == "COMPLETE"
            and receipt.get("delivery_day") == delivery_day
            and receipt.get("first_profile_day") == first_profile_day and receipt.get("series") == dict(SERIES),
            "Saturn target snapshot contract changed")
    require(set(receipt.get("artifact_sha256", {})) == {"prices.parquet"}
            and sha256(directory / "prices.parquet") == receipt["artifact_sha256"]["prices.parquet"],
            "Saturn current-fit prices changed")
    frame = pd.read_parquet(directory / "prices.parquet")
    expected = target_grid(first_profile_day, delivery_day)
    require(frame.index.equals(expected) and tuple(frame.columns) == ZONES
            and np.isfinite(frame.to_numpy(float)).all(), "Invalid Saturn current-fit target grid")
    require(receipt.get("first_hour_utc") == expected[0].isoformat()
            and receipt.get("last_hour_utc") == expected[-1].isoformat()
            and receipt.get("hours_per_zone") == len(expected), "Saturn target snapshot coverage differs")
    require(pd.Timestamp(receipt["retrieved_at_utc"]).tz_convert("UTC") >= cutoff(delivery_day),
            "Saturn target revision had not occurred at retrieval")
    return frame, receipt


def capture_target_snapshot(client, delivery_day: str, first_profile_day: str,
                            cache: Path = DEFAULT_CACHE, *, now_utc=None):
    from run_nyx_annual_auction_prices_source import SERIES
    directory = Path(cache) / "targets_current_fit_v1" / delivery_day
    if (directory / "receipt.json").is_file():
        return verify_target_snapshot(directory, delivery_day, first_profile_day)[1]
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    require(now.tzinfo is not None and now >= cutoff(delivery_day), "Saturn outer D-1 08:00 cutoff not reached")
    expected, values = target_grid(first_profile_day, delivery_day), {}
    for zone in ZONES:
        try:
            series = fetch_saturn_series_from_client(client, SERIES[zone], expected[0], expected[-1], "UTC",
                revision_date=cutoff(delivery_day), naive_timezone="UTC", incomplete_dst_policy="raise", nocache=True)
            values[zone] = _numeric(series, expected, f"{zone} target")
        except Exception as error:
            raise SaturnSourceError(delivery_day, "current_fit_targets", error,
                                    alias=f"{zone}_target", series=SERIES[zone]) from error
    _parquet(directory / "prices.parquet", pd.DataFrame(values, index=expected))
    receipt = {"protocol": TARGET_PROTOCOL, "state": "COMPLETE", "delivery_day": delivery_day,
        "first_profile_day": first_profile_day, "series": dict(SERIES), **_target_contract(delivery_day),
        "first_hour_utc": expected[0].isoformat(), "last_hour_utc": expected[-1].isoformat(),
        "hours_per_zone": len(expected),
        "retrieved_at_utc": (pd.Timestamp.now(tz="UTC") if now_utc is None else now).isoformat(),
        "provider_publication_timestamp_verified": False,
        "availability_basis": "Canonical targets queried at the outer forecast cutoff; historical inner origins are reconstructed",
        "collector_code_sha256": sha256(Path(__file__)),
        "artifact_sha256": {"prices.parquet": sha256(directory / "prices.parquet")}}
    _immutable(directory / "receipt.json", _encoded(receipt))
    verify_target_snapshot(directory, delivery_day, first_profile_day)
    return receipt


def sync(delivery_day: str, *, first_day: str | None = None, cache: Path = DEFAULT_CACHE,
         workers: int = 2, client_factory: Callable | None = None):
    from run_nyx_annual_auction_prices_source import load_plan
    require(type(workers) is int and 1 <= workers <= 8, "Use 1-8 source workers")
    last = date.fromisoformat(delivery_day)
    profiles = Path(cache) / "profiles_v2"
    first = source_start(profiles, delivery_day, first_day, plan_protocol=PROFILE_PROTOCOL)
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
        if (profiles / day / "receipt.json").exists():
            return verify_profile_day(profiles / day, day)[1]
        client = client_factory()
        try:
            return capture_profile_day(client, day, cache)
        finally:
            session = getattr(client, "session", None)
            if session is not None:
                session.close()
    results = {}
    with exclusive_process_lock(profiles / "sync.lock"):
        _immutable(profiles / "plan.json", _encoded({"protocol": PROFILE_PROTOCOL, "first_delivery_day": first.isoformat()}))
        print(f"Saturn: canonical training prices at outer cutoff {cutoff(delivery_day).isoformat()}", flush=True)
        client = None
        try:
            target_directory = Path(cache) / "targets_current_fit_v1" / delivery_day
            if (target_directory / "receipt.json").is_file():
                verify_target_snapshot(target_directory, delivery_day, first.isoformat())
            else:
                client = client_factory()
                capture_target_snapshot(client, delivery_day, first.isoformat(), cache)
        except Exception as error:
            if isinstance(error, SaturnSourceError):
                raise
            raise SaturnSourceError(delivery_day, "current_fit_targets", error) from error
        finally:
            session = getattr(client, "session", None)
            if session is not None:
                session.close()
        print(f"Saturn: checking {len(days)} archived profile days ({first} to {last})", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(one, d): d for d in days}
            for future in as_completed(pending):
                d = pending[future]
                try:
                    results[d] = future.result()
                except Exception as error:
                    for remaining in pending:
                        remaining.cancel()
                    if isinstance(error, SaturnSourceError):
                        raise
                    raise SaturnSourceError(d, "daily_capture", error) from error
                if len(results) % 10 == 0 or len(results) == len(days):
                    print(f"Saturn: {len(results)}/{len(days)} daily states verified", flush=True)
    return results


def source_start(cache: Path, delivery_day: str, first_day: str | None = None, *, plan_protocol=PROTOCOL):
    """Keep the bootstrap origin stable as the live cache grows day by day."""
    last = date.fromisoformat(delivery_day)
    plan_path = Path(cache) / "plan.json"
    requested = date.fromisoformat(first_day) if first_day else last - timedelta(days=WARMUP_DAYS)
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        require(plan.get("protocol") == plan_protocol, "Saturn source cache plan changed")
        first = date.fromisoformat(plan["first_delivery_day"])
        require(first <= requested, "Requested deeper history requires a separate source cache")
        return first
    return requested


def publish(bundle: Path, delivery_day: str, *, first_day: str | None = None,
            cache: Path = DEFAULT_CACHE):
    last = date.fromisoformat(delivery_day)
    profiles = Path(cache) / "profiles_v2"
    first = source_start(profiles, delivery_day, first_day, plan_protocol=PROFILE_PROTOCOL)
    require(first <= last - timedelta(days=365), "Saturn training history too short")
    frames, daily, hashes = [], {}, {}
    bundle = Path(bundle).resolve()
    for stamp in pd.date_range(first, last, freq="D"):
        day = stamp.date().isoformat()
        directory = profiles / day
        cov, receipt = verify_profile_day(directory, day)
        frames.append(cov)
        for name in ("covariates.parquet", "receipt.json"):
            relative = f"source_artifacts/saturn/days/{day}/{name}"
            digest = sha256(directory / name)
            publish_verified_immutable_copy(directory / name, bundle / relative, digest)
            hashes[relative] = digest
        daily[day] = {"forecast_origin_utc": receipt["forecast_origin_utc"],
                      "covariates_sha256": receipt["artifact_sha256"]["covariates.parquet"],
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
    target_directory = Path(cache) / "targets_current_fit_v1" / delivery_day
    _, target_receipt = verify_target_snapshot(target_directory, delivery_day, first.isoformat())
    for name in ("prices.parquet", "receipt.json"):
        relative = f"source_artifacts/saturn/targets/{name}"
        digest = sha256(target_directory / name)
        publish_verified_immutable_copy(target_directory / name, bundle / relative, digest)
        hashes[relative] = digest
    target_snapshot = {"receipt_path": "source_artifacts/saturn/targets/receipt.json",
        "prices_path": "source_artifacts/saturn/targets/prices.parquet",
        "prices_sha256": target_receipt["artifact_sha256"]["prices.parquet"],
        "first_profile_day": first.isoformat(), "target_revision_utc": cutoff(delivery_day).isoformat()}
    receipt = {"protocol": SOURCE_PROTOCOL, "source_group": "saturn", "delivery_day": delivery_day,
        "state": "COMPLETE", "asof_cutoff_verified": True, "training_window_complete": True,
        "asof_state_utc": cutoff(delivery_day).isoformat(), "daily_vintages": daily,
        "first_delivery_day": first.isoformat(), "last_delivery_day": delivery_day,
        "availability_basis": "Each forecast profile uses its historical revision_date; training prices use the outer forecast revision_date",
        **_target_contract(delivery_day), "targets_snapshot": target_snapshot,
        "provider_publication_timestamp_verified": False,
        "historical_publication_vintages_certified": False,
        "publication_vintage_limit": "Query-as-of is not certified original provider insertion time",
        "artifact_sha256": hashes, "series": specs(), "collector_code_sha256": sha256(Path(__file__))}
    verify_current_fit_source(bundle, delivery_day, receipt)
    validate_source_receipt(receipt, group="saturn", day=delivery_day, bundle=bundle, cutoff=cutoff(delivery_day))
    path = bundle / "source_receipts/saturn.json"
    _immutable(path, _encoded(receipt))
    return path


def _bound(bundle, receipt, relative):
    path = (Path(bundle) / relative).resolve()
    require(path.is_relative_to(Path(bundle).resolve()) and path.is_file(), "Saturn source artifact missing or outside bundle")
    require(sha256(path) == receipt.get("artifact_sha256", {}).get(relative), f"Saturn source artifact changed: {relative}")
    return path


def _current_targets(bundle, receipt):
    day = receipt["delivery_day"]
    require(receipt.get("protocol") == SOURCE_PROTOCOL and receipt.get("source_group") == "saturn"
            and receipt.get("state") == "COMPLETE", "Saturn source receipt invalid")
    contract = _check_target_contract(receipt, day)
    declaration = receipt.get("targets_snapshot", {})
    price_path = "source_artifacts/saturn/targets/prices.parquet"
    receipt_path = "source_artifacts/saturn/targets/receipt.json"
    require(declaration.get("prices_path") == price_path and declaration.get("receipt_path") == receipt_path
            and declaration.get("first_profile_day") == receipt.get("first_delivery_day")
            and declaration.get("target_revision_utc") == contract["target_revision_utc"],
            "Saturn target snapshot declaration differs")
    _bound(bundle, receipt, receipt_path)
    target_path = _bound(bundle, receipt, price_path)
    frame, raw = verify_target_snapshot(target_path.parent, day, receipt["first_delivery_day"])
    require(declaration.get("prices_sha256") == raw["artifact_sha256"]["prices.parquet"],
            "Saturn target snapshot checksum declaration differs")
    return frame, contract


def target_history_contract(bundle):
    receipt = json.loads((Path(bundle) / "source_receipts/saturn.json").read_text(encoding="utf-8"))
    if receipt.get("target_history_policy") is None:
        require(not any(key in receipt for key in TARGET_HISTORY_FIELDS), "Incomplete Saturn target policy metadata")
        return {}
    return _current_targets(bundle, receipt)[1]


def verify_current_fit_source(bundle, day, receipt):
    """Independently bind raw profiles and the outer target snapshot to this run."""
    bundle = Path(bundle)
    require(receipt.get("delivery_day") == day and receipt.get("last_delivery_day") == day,
            "Saturn current-fit outer delivery day differs")
    prices, contract = _current_targets(bundle, receipt)
    first, last = date.fromisoformat(receipt["first_delivery_day"]), date.fromisoformat(day)
    days = [stamp.date().isoformat() for stamp in pd.date_range(first, last, freq="D")]
    require(set(receipt.get("daily_vintages", {})) == set(days) and first <= last - timedelta(days=365)
            and receipt.get("series") == specs(), "Saturn profile range or series differs")
    frames = []
    expected_artifacts = {"source_artifacts/saturn/covariates.parquet", "source_artifacts/saturn/origins.parquet",
                          "source_artifacts/saturn/targets/prices.parquet", "source_artifacts/saturn/targets/receipt.json"}
    for origin in days:
        prefix = f"source_artifacts/saturn/days/{origin}"
        for name in ("receipt.json", "covariates.parquet"):
            _bound(bundle, receipt, f"{prefix}/{name}")
            expected_artifacts.add(f"{prefix}/{name}")
        covariates, raw = verify_profile_day(bundle / prefix, origin)
        declared = receipt["daily_vintages"][origin]
        require(declared.get("forecast_origin_utc") == raw["forecast_origin_utc"]
                and declared.get("covariates_sha256") == raw["artifact_sha256"]["covariates.parquet"]
                and declared.get("receipt_path") == f"{prefix}/receipt.json"
                and "prices_sha256" not in declared, f"Saturn profile declaration differs: {origin}")
        frames.append(covariates)
    require(set(receipt.get("artifact_sha256", {})) == expected_artifacts, "Saturn current-fit source inventory differs")
    combined = pd.concat(frames)
    stored = pd.read_parquet(_bound(bundle, receipt, "source_artifacts/saturn/covariates.parquet"))
    require(stored.equals(combined), "Saturn combined forecast profiles differ from raw profiles")
    origins = pd.DataFrame({"forecast_origin_utc": [cutoff(str(d))
        for d in combined.index.tz_convert("Europe/Paris").date]}, index=combined.index)
    stored = pd.read_parquet(_bound(bundle, receipt, "source_artifacts/saturn/origins.parquet"))
    require(stored.equals(origins), "Saturn combined origin grid differs from raw profiles")
    latest = prices.loc[target_grid(day, day)]
    return combined, latest, {"daily_states_verified": len(days), "provider_first_publication_certified": False,
                              "asof_cutoff_verified": True, **contract}


def load_target_snapshots(bundle: Path):
    bundle = Path(bundle)
    receipt = json.loads((bundle / "source_receipts/saturn.json").read_text(encoding="utf-8"))
    if receipt.get("target_history_policy") is not None:
        frame, contract = _current_targets(bundle, receipt)
        def current_load(day):
            day = str(pd.Timestamp(day).date())
            require(receipt["first_delivery_day"] <= day <= receipt["delivery_day"]
                    and day in receipt["daily_vintages"], "Inner reconstruction day outside Saturn source range")
            _bound(bundle, receipt, "source_artifacts/saturn/targets/prices.parquet")
            selected = frame.loc[target_grid(day, day)]
            require(selected.index[-1] < grid(day)[0] and np.isfinite(selected.to_numpy(float)).all(),
                    "Inner reconstruction contains unavailable or future target labels")
            result = {zone: selected[zone].rename("target") for zone in ZONES}
            for series in result.values():
                series.attrs.update(contract)
            return result
        return current_load
    def load(day):
        day = str(pd.Timestamp(day).date())
        expected = receipt["daily_vintages"][day]["prices_sha256"]
        path = bundle / "source_artifacts/saturn/days" / day / "prices.parquet"
        require(sha256(path) == expected, f"Changed historical price snapshot: {day}")
        frame = pd.read_parquet(path)
        return {zone: frame[zone].rename("target") for zone in ZONES}
    return load
