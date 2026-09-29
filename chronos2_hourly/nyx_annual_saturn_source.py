"""Daily as-of Saturn inputs for the annual CPU pipeline.

Forecast series use their historical civil cutoffs where available. After
bounded retries, an unavailable historical series may use the outer fit cutoff;
the live delivery day keeps its own cutoff. Logical origins and real profile
revisions are recorded separately. Canonical training prices use the outer
cutoff; inner windows never receive prices for their delivery day.
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
import threading
import time
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
PROFILE_RECOVERY_PROTOCOL = "nyx_annual_saturn_profiles_current_fit_v1"
PROFILE_SERIES_PROTOCOL = "nyx_annual_saturn_profiles_per_series_v2"
PROFILE_ALIAS_PROTOCOL = "nyx_annual_saturn_profile_alias_v2"
PROFILE_NORMALIZATION_POLICY = "historical_saturn_profiles_with_audited_nl_spring_v1"
WHOLE_PROFILE_HISTORY_POLICY = "own_origin_with_current_fit_recovery_v1"
PROFILE_HISTORY_POLICY = "own_origin_with_per_series_recovery_v2"
PROFILE_HISTORY_POLICIES = (WHOLE_PROFILE_HISTORY_POLICY, PROFILE_HISTORY_POLICY)
PROFILE_HISTORY_FIELDS = ("profile_history_policy", "profile_revision_ceiling_utc",
                          "profile_origin_snapshot_verified")
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
_ARCHIVE_CACHE_LOCK = threading.Lock()


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


def _nl_spring_profile(series, day, expected, *, query_revision=None):
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
                       "forecast_origin_utc": (cutoff(day) if query_revision is None else pd.Timestamp(query_revision)).isoformat()})
    return repaired, ledger


def _verify_profile_normalization(frame, receipt, day, *, query_revision=None):
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
    reconstructed, ledger = _nl_spring_profile(raw, day, expected, query_revision=query_revision)
    require(entries == ledger, "Saturn NL spring repair ledger differs from same-vintage donors")
    require(np.array_equal(_numeric(reconstructed, expected, "nl_residual_load_fcst"),
                           frame["nl_residual_load_fcst"].to_numpy(float)),
            "Saturn NL spring repaired values differ from historical normalization")


def _wind_audit(substitutions):
    from .nyx_annual_wind_source import POLICY
    return {"source_substitutions": substitutions, "source_substitution_count": len(substitutions),
            "wind_gap_policy": POLICY if substitutions else None}


def _verify_profile_wind(frame, receipt, day, revision):
    from .nyx_annual_wind_source import verify_nl_wind_substitutions
    substitutions = receipt.get("source_substitutions", [])
    require(isinstance(substitutions, list), "Saturn wind substitution audit differs")
    expected = _wind_audit(substitutions)
    if any(key in receipt for key in expected):
        require(type(receipt.get("source_substitution_count")) is int
                and all(key in receipt and receipt[key] == value for key, value in expected.items()),
                "Saturn wind substitution audit differs")
    verify_nl_wind_substitutions(frame["nl_wind_generation_fcst"], substitutions, day, revision)


def _profile_contract(day, policy=PROFILE_HISTORY_POLICY):
    return {"profile_history_policy": policy,
            "profile_revision_ceiling_utc": cutoff(day).isoformat(),
            "profile_origin_snapshot_verified": False}


def _check_profile_contract(receipt, day):
    if receipt.get("profile_history_policy") is None:
        require(not any(key in receipt for key in PROFILE_HISTORY_FIELDS)
                and "logical_forecast_origins" not in receipt
                and "source_artifacts/saturn/profile_revisions.parquet" not in receipt.get("artifact_sha256", {}),
                "Incomplete or downgraded Saturn profile history metadata")
        return {}
    require(receipt.get("profile_history_policy") in PROFILE_HISTORY_POLICIES,
            "Unsupported Saturn profile history policy")
    expected = _profile_contract(day, receipt["profile_history_policy"])
    require(all(receipt.get(key) == value for key, value in expected.items())
            and receipt.get("profile_origin_snapshot_verified") is False
            and receipt.get("logical_forecast_origins") is True,
            "Saturn profile history policy or revision ceiling differs")
    return expected


def _profile_revision(receipt, day, outer_day=None):
    if receipt.get("protocol") == PROFILE_PROTOCOL:
        require(receipt.get("forecast_origin_utc") == cutoff(day).isoformat()
                and receipt.get("profile_revision_utc", cutoff(day).isoformat()) == cutoff(day).isoformat()
                and receipt.get("profile_origin_snapshot_verified", True) is True
                and "outer_delivery_day" not in receipt, "Saturn strict profile origin changed")
        return cutoff(day)
    require(receipt.get("protocol") == PROFILE_RECOVERY_PROTOCOL and outer_day is not None
            and date.fromisoformat(day) < date.fromisoformat(outer_day),
            "Saturn current-fit profile recovery is forbidden for the live delivery day")
    revision = cutoff(outer_day)
    require(receipt.get("outer_delivery_day") == outer_day
            and receipt.get("forecast_origin_utc") == revision.isoformat()
            and receipt.get("profile_revision_utc") == revision.isoformat()
            and receipt.get("logical_forecast_origin_utc") == cutoff(day).isoformat()
            and receipt.get("profile_history_policy") == WHOLE_PROFILE_HISTORY_POLICY
            and receipt.get("profile_revision_ceiling_utc") == revision.isoformat()
            and receipt.get("profile_origin_snapshot_verified") is False,
            "Saturn recovered profile revision or logical origin changed")
    return revision


def _profile_revisions(receipt, day, outer_day=None):
    if receipt.get("protocol") != PROFILE_SERIES_PROTOCOL:
        return dict.fromkeys(ALIASES, _profile_revision(receipt, day, outer_day))
    require(outer_day is not None and day <= outer_day
            and receipt.get("outer_delivery_day") == outer_day
            and receipt.get("logical_forecast_origin_utc") == cutoff(day).isoformat()
            and "forecast_origin_utc" not in receipt and "profile_revision_utc" not in receipt,
            "Saturn per-series logical origin or outer delivery differs")
    require(all(receipt.get(k) == v for k, v in _profile_contract(outer_day).items())
            and receipt.get("profile_origin_snapshot_verified") is False,
            "Saturn per-series profile policy differs")
    aliases = receipt.get("alias_revisions_utc", {})
    origins = receipt.get("alias_origins_verified", {})
    sources = receipt.get("alias_sources", {})
    require(set(aliases) == set(ALIASES) and set(origins) == set(ALIASES) and set(sources) == set(ALIASES),
            "Saturn per-series revision aliases differ")
    revisions = {a: pd.Timestamp(aliases[a]) for a in ALIASES}
    for alias, revision in revisions.items():
        method = sources[alias]
        require(revision.tzinfo is not None and revision <= cutoff(outer_day)
                and ((method == "own_origin" and revision == cutoff(day))
                     or (method == "outer_revision" and day < outer_day and revision == cutoff(outer_day))
                     or (method == "repository_vintage" and alias == "nl_residual_load_fcst" and day < outer_day)),
                "Saturn per-series revision or recovery source differs")
    require(all(origins[a] is (sources[a] == "own_origin") for a in ALIASES),
            "Saturn per-series historical origin claim differs")
    require(receipt.get("profile_revision_max_utc") == max(revisions.values()).isoformat(),
            "Saturn per-series maximum revision differs")
    return revisions


def verify_profile_day(directory: Path, day: str, *, outer_day=None):
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    revisions = _profile_revisions(receipt, day, outer_day)
    require(receipt.get("delivery_day") == day
            and receipt.get("state") == "COMPLETE" and receipt.get("series") == specs(),
            "Saturn profile cache contract changed")
    artifacts = {"covariates.parquet"}
    if receipt.get("alias_sources", {}).get("nl_residual_load_fcst") == "repository_vintage":
        artifacts.add("nl_repository_vintages.parquet")
    require(set(receipt.get("artifact_sha256", {})) == artifacts,
            "Saturn profile artifact inventory differs")
    for name in artifacts:
        require(sha256(directory / name) == receipt["artifact_sha256"][name],
                f"Saturn profile {day}: modified {name}")
    frame = pd.read_parquet(directory / "covariates.parquet")
    require(frame.index.equals(grid(day)) and tuple(frame.columns) == ALIASES
            and np.isfinite(frame.to_numpy(float)).all(), "Invalid Saturn daily forecast grid")
    if receipt["protocol"] == PROFILE_SERIES_PROTOCOL:
        evidence = receipt.get("alias_evidence", {})
        require(set(evidence) == set(ALIASES), "Saturn per-series evidence aliases differ")
        for alias in ALIASES:
            _verify_alias_evidence(evidence[alias], frame[alias].to_numpy(float), day, outer_day, alias, directory)
            require(evidence[alias]["revision_utc"] == revisions[alias].isoformat()
                    and evidence[alias]["source"] == receipt["alias_sources"][alias],
                    "Saturn per-series evidence revision differs")
        require(receipt.get("dst_evidence") == {a: evidence[a]["dst_evidence"] for a in ALIASES}
                and receipt.get("spring_dst_repair") == evidence["nl_residual_load_fcst"]["spring_dst_repair"]
                and receipt.get("source_substitutions", []) == evidence["nl_wind_generation_fcst"].get("source_substitutions", []),
                "Saturn per-series normalization evidence differs")
    _verify_profile_normalization(frame, receipt, day, query_revision=revisions["nl_residual_load_fcst"])
    _verify_profile_wind(frame, receipt, day, revisions["nl_wind_generation_fcst"])
    require(pd.Timestamp(receipt["retrieved_at_utc"]).tz_convert("UTC") >= max(revisions.values()),
            "Saturn profile cutoff had not occurred at retrieval")
    return frame, receipt


def capture_profile_day(client, day: str, cache: Path = DEFAULT_CACHE, *, now_utc=None,
                        outer_day=None, own_origin_failure=None):
    if outer_day is not None:
        require(date.fromisoformat(day) < date.fromisoformat(outer_day),
                "Saturn current-fit profile recovery is forbidden for the live delivery day")
    directory = (Path(cache) / "profiles_current_fit_v1" / outer_day / day if outer_day is not None
                 else Path(cache) / "profiles_v2" / day)
    revision = cutoff(outer_day or day)
    if (directory / "receipt.json").is_file():
        return verify_profile_day(directory, day, outer_day=outer_day)[1]
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    require(now.tzinfo is not None and now >= revision, "Saturn D-1 08:00 cutoff not reached")
    from materialize_saturn_kalman_fuel import NL_SPRING_DST_REPAIR_POLICY
    expected, contract, values, evidence, spring_ledger, substitutions = grid(day), specs(), {}, {}, [], []
    for alias, spec in contract.items():
        try:
            series = fetch_saturn_series_from_client(client, spec["series"],
                expected[0] - pd.Timedelta(hours=8), expected[-1] + pd.Timedelta(hours=8), "UTC",
                revision_date=revision, naive_timezone=spec["naive_timezone"],
                incomplete_dst_policy=spec["dst_policy"], nocache=True)
            selected = series
            if alias == "nl_residual_load_fcst":
                selected, spring_ledger = _nl_spring_profile(series, day, expected, query_revision=revision)
            elif alias == "nl_wind_generation_fcst":
                from .nyx_annual_wind_source import normalize_nl_wind_profile
                selected, substitutions = normalize_nl_wind_profile(client, series, day, expected, revision)
            values[alias] = _numeric(selected, expected, alias)
        except Exception as error:
            phase = "current_fit_profile_recovery" if outer_day is not None else "forecast_profile"
            raise SaturnSourceError(day, phase, error, alias=alias, series=spec["series"]) from error
        evidence[alias] = {"policy": spec["dst_policy"],
                           "normalizer_attrs": json.loads(json.dumps(series.attrs, default=str))}
    _parquet(directory / "covariates.parquet", pd.DataFrame(values, index=expected))
    receipt = {"protocol": PROFILE_RECOVERY_PROTOCOL if outer_day is not None else PROFILE_PROTOCOL,
        "delivery_day": day, "state": "COMPLETE", "forecast_origin_utc": revision.isoformat(), "series": contract,
        "retrieved_at_utc": (pd.Timestamp.now(tz="UTC") if now_utc is None else now).isoformat(),
        "publication_verified": False, "provider_publication_timestamp_verified": False,
        "availability_basis": "Forecast profiles queried at their own delivery D-1 08:00 revision_date",
        "profile_normalization_policy": PROFILE_NORMALIZATION_POLICY,
        "spring_dst_repair": {"alias": "nl_residual_load_fcst", "policy": NL_SPRING_DST_REPAIR_POLICY,
                              "entries": spring_ledger},
        "dst_evidence": evidence, **_wind_audit(substitutions), "collector_code_sha256": sha256(Path(__file__)),
        "artifact_sha256": {"covariates.parquet": sha256(directory / "covariates.parquet")}}
    if outer_day is not None:
        receipt.update(**_profile_contract(outer_day, WHOLE_PROFILE_HISTORY_POLICY), outer_delivery_day=outer_day,
            profile_revision_utc=revision.isoformat(), logical_forecast_origin_utc=cutoff(day).isoformat(),
            own_origin_failure=redact_text(str(own_origin_failure or "Historical origin unavailable after bounded retries")),
            availability_basis="Complete historical forecast profile reconstructed at the outer fit cutoff; not an original historical origin snapshot")
    _immutable(directory / "receipt.json", _encoded(receipt))
    verify_profile_day(directory, day, outer_day=outer_day)
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


def _profile_cache_directory(cache, day, outer_day):
    strict = Path(cache) / "profiles_v2" / day
    if (strict / "receipt.json").is_file():
        # A damaged strict cache must never be hidden by a recovery packet.
        verify_profile_day(strict, day)
        return strict
    recovery = Path(cache) / "profiles_per_series_v2" / outer_day / day
    if (recovery / "receipt.json").is_file():
        verify_profile_day(recovery, day, outer_day=outer_day)
        return recovery
    return None


def _values_digest(values, day):
    digest = hashlib.sha256(grid(day).asi8.astype("<i8").tobytes())
    digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def _spring_evidence(entries=()):
    from materialize_saturn_kalman_fuel import NL_SPRING_DST_REPAIR_POLICY
    return {"alias": "nl_residual_load_fcst", "policy": NL_SPRING_DST_REPAIR_POLICY,
            "entries": list(entries)}


def _verify_alias_evidence(evidence, values, day, outer_day, alias, directory):
    require(evidence.get("protocol") == PROFILE_ALIAS_PROTOCOL
            and evidence.get("delivery_day") == day and evidence.get("outer_delivery_day") == outer_day
            and evidence.get("alias") == alias and evidence.get("series") == specs()[alias]
            and evidence.get("logical_forecast_origin_utc") == cutoff(day).isoformat(),
            "Saturn partial series contract differs")
    revision = pd.Timestamp(evidence["revision_utc"])
    method = evidence.get("source")
    require(revision.tzinfo is not None and revision <= cutoff(outer_day)
            and ((method == "own_origin" and revision == cutoff(day))
                 or (method == "outer_revision" and day < outer_day and revision == cutoff(outer_day))
                 or (method == "repository_vintage" and alias == "nl_residual_load_fcst" and day < outer_day)),
            "Saturn partial series revision or source differs")
    require(evidence.get("origin_snapshot_verified") is (method == "own_origin")
            and pd.Timestamp(evidence["retrieved_at_utc"]).tz_convert("UTC") >= revision,
            "Saturn partial series origin or retrieval differs")
    require(np.asarray(values).shape == (len(grid(day)),) and np.isfinite(values).all()
            and evidence.get("values_sha256") == _values_digest(values, day),
            "Saturn partial series values changed or incomplete")
    require(evidence.get("dst_evidence", {}).get("policy") == specs()[alias]["dst_policy"],
            "Saturn partial series normalization policy differs")
    if method != "own_origin":
        require(isinstance(evidence.get("own_origin_failure"), str) and bool(evidence["own_origin_failure"]),
                "Saturn series recovery lacks own-origin failure evidence")
    if alias == "nl_residual_load_fcst":
        frame = pd.DataFrame({alias: values}, index=grid(day))
        _verify_profile_normalization(frame, {"profile_normalization_policy": PROFILE_NORMALIZATION_POLICY,
            "spring_dst_repair": evidence.get("spring_dst_repair")}, day, query_revision=revision)
    else:
        require(evidence.get("spring_dst_repair") == _spring_evidence(),
                "Saturn repair evidence present for another alias")
    if alias == "nl_wind_generation_fcst":
        from .nyx_annual_wind_source import verify_nl_wind_substitutions
        verify_nl_wind_substitutions(pd.Series(values, index=grid(day), name=alias),
                                    evidence.get("source_substitutions", []), day, revision)
    else:
        require(evidence.get("source_substitutions", []) == [], "Wind substitution recorded on another alias")
    if method == "repository_vintage":
        from .nyx_annual_saturn_archive import verify_nl_profile
        require(bool(evidence.get("outer_origin_failure")), "Saturn repository recovery lacks outer failure evidence")
        archive = evidence.get("repository_evidence", {})
        require(evidence["revision_utc"] == archive.get("snapshot_time_utc"),
                "Saturn repository snapshot revision differs")
        verify_nl_profile(pd.Series(values, index=grid(day), name=alias), archive, day, outer_day,
                          archive_path=Path(directory) / "nl_repository_vintages.parquet")


def _save_alias(directory, alias, values, evidence, day, outer_day):
    _verify_alias_evidence(evidence, values, day, outer_day, alias, directory)
    _immutable(directory / "series" / f"{alias}.json",
               _encoded({"values": np.asarray(values, dtype=float).tolist(), "evidence": evidence}))
    return np.asarray(values, dtype=float), evidence


def _alias_evidence(alias, day, outer_day, revision, values, *, method, dst, spring=(),
                    retrieved=None, **extra):
    return {"protocol": PROFILE_ALIAS_PROTOCOL, "delivery_day": day, "outer_delivery_day": outer_day,
        "alias": alias, "series": specs()[alias], "source": method,
        "revision_utc": pd.Timestamp(revision).isoformat(), "logical_forecast_origin_utc": cutoff(day).isoformat(),
        "origin_snapshot_verified": method == "own_origin", "values_sha256": _values_digest(values, day),
        "retrieved_at_utc": (pd.Timestamp.now(tz="UTC") if retrieved is None else pd.Timestamp(retrieved)).isoformat(),
        "dst_evidence": dst, "spring_dst_repair": _spring_evidence(spring), **extra}


def _fetch_profile_alias(day, outer_day, alias, revision, client_factory):
    expected, spec = grid(day), specs()[alias]
    phase = "forecast_profile" if revision == cutoff(day) else "current_fit_profile_recovery"
    for attempt in range(1, 4):
        client = None
        try:
            client = client_factory()
            series = fetch_saturn_series_from_client(client, spec["series"],
                expected[0] - pd.Timedelta(hours=8), expected[-1] + pd.Timedelta(hours=8), "UTC",
                revision_date=revision, naive_timezone=spec["naive_timezone"],
                incomplete_dst_policy=spec["dst_policy"], nocache=True)
            selected, ledger, substitutions = series, [], []
            if alias == "nl_residual_load_fcst":
                selected, ledger = _nl_spring_profile(series, day, expected, query_revision=revision)
            elif alias == "nl_wind_generation_fcst":
                from .nyx_annual_wind_source import normalize_nl_wind_profile
                selected, substitutions = normalize_nl_wind_profile(client, series, day, expected, revision)
                if substitutions:
                    print(json.dumps({"source": "saturn", "event": "wind_substitution", "profile_day": day,
                        "alias": alias, "value_time_utc": substitutions[0]["value_time_utc"],
                        "revision_utc": revision.isoformat(), "native_series": spec["series"],
                        "component_series": substitutions[0]["fallback_series"]}), flush=True)
            values = _numeric(selected, expected, alias)
            return values, _alias_evidence(alias, day, outer_day, revision, values,
                method="own_origin" if revision == cutoff(day) else "outer_revision", spring=ledger,
                source_substitutions=substitutions,
                dst={"policy": spec["dst_policy"], "normalizer_attrs": json.loads(json.dumps(series.attrs, default=str))})
        except Exception as error:
            print(json.dumps({"source": "saturn", "event": "profile_retry", "profile_day": day,
                "alias": alias, "attempt": attempt, "max_attempts": 3, "revision_utc": revision.isoformat(),
                "error": redact_text(str(error))}, ensure_ascii=False), flush=True)
            if attempt == 3:
                detail = RuntimeError(f"3 attempts at revision {revision.isoformat()} failed: {redact_text(str(error))}")
                raise SaturnSourceError(day, phase, detail, alias=alias, series=spec["series"]) from error
        finally:
            session = getattr(client, "session", None)
            if session is not None:
                try:
                    session.close()
                except Exception as error:
                    print(f"Saturn session close: {redact_text(str(error))}", flush=True)
        time.sleep(2 ** (attempt - 1))


def _seal_own_profile(day, cache, frame, evidence):
    """Keep complete historical origins reusable for every subsequent outer fit."""
    directory = Path(cache) / "profiles_v2" / day
    _parquet(directory / "covariates.parquet", frame)
    receipt = {"protocol": PROFILE_PROTOCOL, "delivery_day": day, "state": "COMPLETE",
        "forecast_origin_utc": cutoff(day).isoformat(), "series": specs(),
        "retrieved_at_utc": max(pd.Timestamp(e["retrieved_at_utc"]) for e in evidence.values()).isoformat(),
        "publication_verified": False, "provider_publication_timestamp_verified": False,
        "availability_basis": "Forecast profiles queried at their own delivery D-1 08:00 revision_date",
        "profile_normalization_policy": PROFILE_NORMALIZATION_POLICY,
        "spring_dst_repair": evidence["nl_residual_load_fcst"]["spring_dst_repair"],
        **_wind_audit(evidence["nl_wind_generation_fcst"].get("source_substitutions", [])),
        "dst_evidence": {a: evidence[a]["dst_evidence"] for a in ALIASES},
        "collector_code_sha256": sha256(Path(__file__)),
        "artifact_sha256": {"covariates.parquet": sha256(directory / "covariates.parquet")}}
    _immutable(directory / "receipt.json", _encoded(receipt))
    verify_profile_day(directory, day)
    return receipt


def _repository_archive_path(cache):
    from .nyx_annual_saturn_archive import DEFAULT_ARCHIVE, PINNED_ARCHIVE_SHA256
    path = Path(cache) / "repository_vintages" / f"{PINNED_ARCHIVE_SHA256}.parquet"
    # The source sync owns the process lock; this lock also serializes its workers.
    with _ARCHIVE_CACHE_LOCK:
        if not path.is_file():
            try:
                raw = DEFAULT_ARCHIVE.read_bytes()
            except OSError as error:
                raise ValueError("Pinned NL repository archive is unavailable") from error
            require(hashlib.sha256(raw).hexdigest() == PINNED_ARCHIVE_SHA256,
                    "Pinned NL repository archive changed")
            _immutable(path, raw)  # A byte copy: the tracked archive may later be refreshed.
        require(sha256(path) == PINNED_ARCHIVE_SHA256, "Pinned NL repository cache changed")
    return path


def _sync_profile_day(day, delivery_day, cache, client_factory):
    existing = _profile_cache_directory(cache, day, delivery_day)
    if existing is not None:
        return verify_profile_day(existing, day, outer_day=delivery_day)[1]
    directory = Path(cache) / "profiles_per_series_v2" / delivery_day / day
    require(day <= delivery_day and pd.Timestamp.now(tz="UTC") >= cutoff(delivery_day),
            "Saturn profile or outer cutoff is in the future")
    values, evidence, missing = {}, {}, {}
    # Complete each own-origin request independently: one absent FR series must
    # not prevent us from asking Saturn for the valid historical NL series.
    for alias in ALIASES:
        partial = directory / "series" / f"{alias}.json"
        if partial.is_file():
            item = json.loads(partial.read_text(encoding="utf-8"))
            value, proof = np.asarray(item["values"], dtype=float), item["evidence"]
            _verify_alias_evidence(proof, value, day, delivery_day, alias, directory)
            values[alias], evidence[alias] = value, proof
            continue
        try:
            value, proof = _fetch_profile_alias(day, delivery_day, alias, cutoff(day), client_factory)
        except SaturnSourceError as error:
            missing[alias] = error
            continue
        values[alias], evidence[alias] = _save_alias(directory, alias, value, proof, day, delivery_day)
    legacy = Path(cache) / "profiles_current_fit_v1" / delivery_day / day
    old = verify_profile_day(legacy, day, outer_day=delivery_day) if missing and (legacy / "receipt.json").is_file() else None
    failures = []
    for alias, error in missing.items():
        if day >= delivery_day:
            failures.append(error)
            continue
        print(json.dumps({"source": "saturn", "event": "profile_current_fit_recovery", "profile_day": day,
            "alias": alias, "outer_delivery_day": delivery_day, "logical_origin_utc": cutoff(day).isoformat(),
            "actual_revision_utc": cutoff(delivery_day).isoformat(), "reason": redact_text(str(error))}), flush=True)
        try:
            if old is None:
                value, proof = _fetch_profile_alias(day, delivery_day, alias, cutoff(delivery_day), client_factory)
            else:
                frame, raw = old
                value = frame[alias].to_numpy(float)
                proof = _alias_evidence(alias, day, delivery_day, cutoff(delivery_day), value,
                    method="outer_revision", dst=raw["dst_evidence"][alias], retrieved=raw["retrieved_at_utc"],
                    spring=raw["spring_dst_repair"]["entries"] if alias == "nl_residual_load_fcst" else (),
                    source_substitutions=raw.get("source_substitutions", []) if alias == "nl_wind_generation_fcst" else [],
                    reused_whole_profile_receipt_sha256=sha256(legacy / "receipt.json"))
            proof["own_origin_failure"] = redact_text(str(error))
        except SaturnSourceError as outer_error:
            if alias != "nl_residual_load_fcst":
                failures.append(outer_error)
                continue
            from .nyx_annual_saturn_archive import recover_nl_profile
            try:
                archive_path = _repository_archive_path(cache)
                series, archive = recover_nl_profile(day, delivery_day, archive_path=archive_path)
            except ValueError as archive_error:
                combined = RuntimeError(f"{redact_text(str(outer_error))}; repository vintage unavailable: {redact_text(str(archive_error))}")
                failures.append(SaturnSourceError(day, "current_fit_profile_recovery", combined,
                                                 alias=alias, series=specs()[alias]["series"]))
                continue
            publish_verified_immutable_copy(archive_path, directory / "nl_repository_vintages.parquet", archive["archive_sha256"])
            value = _numeric(series, grid(day), alias)
            proof = _alias_evidence(alias, day, delivery_day, pd.Timestamp(archive["snapshot_time_utc"]), value,
                method="repository_vintage", dst={"policy": specs()[alias]["dst_policy"], "normalizer_attrs": {}},
                repository_evidence=archive, own_origin_failure=redact_text(str(error)),
                outer_origin_failure=redact_text(str(outer_error)))
            print(json.dumps({"source": "saturn", "event": "profile_repository_recovery", "profile_day": day,
                "alias": alias, "actual_revision_utc": archive["snapshot_time_utc"],
                "outer_cutoff_utc": cutoff(delivery_day).isoformat()}), flush=True)
        values[alias], evidence[alias] = _save_alias(directory, alias, value, proof, day, delivery_day)
    if failures:
        first = failures[0]
        detail = RuntimeError(f"{len(failures)} series unavailable: {', '.join(e.alias for e in failures)}; {redact_text(str(first))}")
        raise SaturnSourceError(day, first.phase, detail, alias=first.alias, series=first.series) from first
    frame = pd.DataFrame({a: values[a] for a in ALIASES}, index=grid(day))
    if all(e["source"] == "own_origin" for e in evidence.values()):
        return _seal_own_profile(day, cache, frame, evidence)
    _parquet(directory / "covariates.parquet", frame)
    revisions = {a: evidence[a]["revision_utc"] for a in ALIASES}
    artifacts = {"covariates.parquet": sha256(directory / "covariates.parquet")}
    if evidence["nl_residual_load_fcst"]["source"] == "repository_vintage":
        artifacts["nl_repository_vintages.parquet"] = sha256(directory / "nl_repository_vintages.parquet")
    receipt = {"protocol": PROFILE_SERIES_PROTOCOL, "delivery_day": day, "outer_delivery_day": delivery_day,
        "state": "COMPLETE", "series": specs(), **_profile_contract(delivery_day),
        "logical_forecast_origin_utc": cutoff(day).isoformat(), "alias_revisions_utc": revisions,
        "alias_sources": {a: evidence[a]["source"] for a in ALIASES},
        "alias_origins_verified": {a: evidence[a]["origin_snapshot_verified"] for a in ALIASES},
        "profile_revision_max_utc": max(pd.Timestamp(r) for r in revisions.values()).isoformat(),
        "alias_evidence": evidence, "retrieved_at_utc": max(pd.Timestamp(e["retrieved_at_utc"]) for e in evidence.values()).isoformat(),
        "publication_verified": False, "provider_publication_timestamp_verified": False,
        "availability_basis": "Complete series selected independently at own origin, outer fit cutoff, or audited NL repository vintage",
        "profile_normalization_policy": PROFILE_NORMALIZATION_POLICY,
        "spring_dst_repair": evidence["nl_residual_load_fcst"]["spring_dst_repair"],
        **_wind_audit(evidence["nl_wind_generation_fcst"].get("source_substitutions", [])),
        "dst_evidence": {a: evidence[a]["dst_evidence"] for a in ALIASES},
        "collector_code_sha256": sha256(Path(__file__)), "artifact_sha256": artifacts}
    _immutable(directory / "receipt.json", _encoded(receipt))
    verify_profile_day(directory, day, outer_day=delivery_day)
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
        return _sync_profile_day(day, delivery_day, cache, client_factory)
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
    bundle = Path(bundle).resolve()
    path = bundle / "source_receipts/saturn.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("target_history_policy") is not None:
            verify_current_fit_source(bundle, delivery_day, existing)
        else:
            from .nyx_annual_source_validation import _saturn
            _saturn(bundle, delivery_day, existing)
        validate_source_receipt(existing, group="saturn", day=delivery_day, bundle=bundle, cutoff=cutoff(delivery_day))
        return path
    last = date.fromisoformat(delivery_day)
    profiles = Path(cache) / "profiles_v2"
    first = source_start(profiles, delivery_day, first_day, plan_protocol=PROFILE_PROTOCOL)
    require(first <= last - timedelta(days=365), "Saturn training history too short")
    frames, daily, hashes, revision_frames = [], {}, {}, []
    for stamp in pd.date_range(first, last, freq="D"):
        day = stamp.date().isoformat()
        directory = _profile_cache_directory(cache, day, delivery_day)
        require(directory is not None, f"Saturn profile cache missing for {day} at outer delivery {delivery_day}")
        cov, receipt = verify_profile_day(directory, day, outer_day=delivery_day)
        actual_revisions = _profile_revisions(receipt, day, delivery_day)
        frames.append(cov)
        revision_frames.append(pd.DataFrame({a: actual_revisions[a] for a in ALIASES}, index=cov.index))
        for name in (*receipt["artifact_sha256"], "receipt.json"):
            relative = f"source_artifacts/saturn/days/{day}/{name}"
            digest = sha256(directory / name)
            publish_verified_immutable_copy(directory / name, bundle / relative, digest)
            hashes[relative] = digest
        sources = (receipt["alias_sources"] if receipt["protocol"] == PROFILE_SERIES_PROTOCOL
                   else dict.fromkeys(ALIASES, "own_origin"))
        daily[day] = {"logical_forecast_origin_utc": cutoff(day).isoformat(),
                      "alias_revisions_utc": {a: r.isoformat() for a, r in actual_revisions.items()},
                      "alias_sources": sources,
                      "alias_origins_verified": {a: sources[a] == "own_origin" for a in ALIASES},
                      "profile_revision_max_utc": max(actual_revisions.values()).isoformat(),
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
    revision_path = "source_artifacts/saturn/profile_revisions.parquet"
    _parquet(bundle / revision_path, pd.concat(revision_frames))
    hashes[revision_path] = sha256(bundle / revision_path)
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
        "availability_basis": "Each historical series uses its own revision_date, otherwise the outer fit revision_date or an audited NL repository vintage; live delivery remains at its own origin",
        **_target_contract(delivery_day), "targets_snapshot": target_snapshot,
        **_profile_contract(delivery_day), "logical_forecast_origins": True,
        "provider_publication_timestamp_verified": False,
        "historical_publication_vintages_certified": False,
        "publication_vintage_limit": "Query-as-of is not certified original provider insertion time",
        "artifact_sha256": hashes, "series": specs(), "collector_code_sha256": sha256(Path(__file__))}
    verify_current_fit_source(bundle, delivery_day, receipt)
    validate_source_receipt(receipt, group="saturn", day=delivery_day, bundle=bundle, cutoff=cutoff(delivery_day))
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


def profile_history_contract(bundle):
    bundle = Path(bundle)
    receipt = json.loads((bundle / "source_receipts/saturn.json").read_text(encoding="utf-8"))
    contract = _check_profile_contract(receipt, receipt.get("delivery_day"))
    if not contract:
        require(not (bundle / "source_artifacts/saturn/profile_revisions.parquet").exists(),
                "Downgraded Saturn profile revision evidence")
        return {}
    verify_current_fit_source(bundle, receipt["delivery_day"], receipt)
    return contract


def verify_current_fit_source(bundle, day, receipt):
    """Independently bind raw profiles and the outer target snapshot to this run."""
    bundle = Path(bundle)
    require(receipt.get("delivery_day") == day and receipt.get("last_delivery_day") == day,
            "Saturn current-fit outer delivery day differs")
    prices, contract = _current_targets(bundle, receipt)
    profile_contract = _check_profile_contract(receipt, day)
    first, last = date.fromisoformat(receipt["first_delivery_day"]), date.fromisoformat(day)
    days = [stamp.date().isoformat() for stamp in pd.date_range(first, last, freq="D")]
    require(set(receipt.get("daily_vintages", {})) == set(days) and first <= last - timedelta(days=365)
            and receipt.get("series") == specs(), "Saturn profile range or series differs")
    frames = []
    expected_artifacts = {"source_artifacts/saturn/covariates.parquet", "source_artifacts/saturn/origins.parquet",
                          "source_artifacts/saturn/targets/prices.parquet", "source_artifacts/saturn/targets/receipt.json"}
    revision_frames = []
    for origin in days:
        prefix = f"source_artifacts/saturn/days/{origin}"
        for name in ("receipt.json", "covariates.parquet"):
            _bound(bundle, receipt, f"{prefix}/{name}")
            expected_artifacts.add(f"{prefix}/{name}")
        covariates, raw = verify_profile_day(bundle / prefix, origin, outer_day=day if profile_contract else None)
        actual_revisions = _profile_revisions(raw, origin, day if profile_contract else None)
        for name in raw["artifact_sha256"]:
            _bound(bundle, receipt, f"{prefix}/{name}")
            expected_artifacts.add(f"{prefix}/{name}")
        declared = receipt["daily_vintages"][origin]
        require(declared.get("covariates_sha256") == raw["artifact_sha256"]["covariates.parquet"]
                and declared.get("receipt_path") == f"{prefix}/receipt.json"
                and "prices_sha256" not in declared, f"Saturn profile declaration differs: {origin}")
        if profile_contract.get("profile_history_policy") == PROFILE_HISTORY_POLICY:
            require(raw["protocol"] in (PROFILE_PROTOCOL, PROFILE_SERIES_PROTOCOL),
                    "Whole-day profile recovery cannot bypass per-series own-origin requests")
            sources = (raw["alias_sources"] if raw["protocol"] == PROFILE_SERIES_PROTOCOL
                       else dict.fromkeys(ALIASES, "own_origin"))
            require(declared.get("logical_forecast_origin_utc") == cutoff(origin).isoformat()
                    and declared.get("alias_revisions_utc") == {a: r.isoformat() for a, r in actual_revisions.items()}
                    and declared.get("alias_sources") == sources
                    and declared.get("alias_origins_verified") == {a: sources[a] == "own_origin" for a in ALIASES}
                    and declared.get("profile_revision_max_utc") == max(actual_revisions.values()).isoformat()
                    and "forecast_origin_utc" not in declared and "profile_revision_utc" not in declared,
                    f"Saturn per-series revision declaration differs: {origin}")
        elif profile_contract:
            require(raw["protocol"] != PROFILE_SERIES_PROTOCOL,
                    "Per-series profile recovery cannot use the old whole-day policy")
            revision = _profile_revision(raw, origin, day)
            require(declared.get("logical_forecast_origin_utc") == cutoff(origin).isoformat()
                    and declared.get("forecast_origin_utc") == raw["forecast_origin_utc"]
                    and declared.get("profile_revision_utc") == revision.isoformat()
                    and declared.get("profile_origin_snapshot_verified") is (raw["protocol"] == PROFILE_PROTOCOL)
                    and revision <= cutoff(day), f"Saturn actual profile revision declaration differs: {origin}")
        else:
            require(raw["protocol"] == PROFILE_PROTOCOL, "Unlabelled historical profile recovery")
            require(declared.get("forecast_origin_utc") == raw["forecast_origin_utc"],
                    "Saturn strict profile declaration differs")
        frames.append(covariates)
        revision_frames.append(pd.DataFrame({a: actual_revisions[a] for a in ALIASES}, index=covariates.index))
    revision_path = "source_artifacts/saturn/profile_revisions.parquet"
    if profile_contract:
        expected_artifacts.add(revision_path)
    else:
        require(not (bundle / revision_path).exists(), "Downgraded Saturn profile revision evidence")
    require(set(receipt.get("artifact_sha256", {})) == expected_artifacts, "Saturn current-fit source inventory differs")
    combined = pd.concat(frames)
    stored = pd.read_parquet(_bound(bundle, receipt, "source_artifacts/saturn/covariates.parquet"))
    require(stored.equals(combined), "Saturn combined forecast profiles differ from raw profiles")
    origins = pd.DataFrame({"forecast_origin_utc": [cutoff(str(d))
        for d in combined.index.tz_convert("Europe/Paris").date]}, index=combined.index)
    stored = pd.read_parquet(_bound(bundle, receipt, "source_artifacts/saturn/origins.parquet"))
    require(stored.equals(origins), "Saturn combined origin grid differs from raw profiles")
    if profile_contract:
        actual = pd.concat(revision_frames)
        stored = pd.read_parquet(_bound(bundle, receipt, revision_path))
        require(stored.equals(actual), "Saturn actual profile revisions differ from raw receipts")
    latest = prices.loc[target_grid(day, day)]
    return combined, latest, {"daily_states_verified": len(days), "provider_first_publication_certified": False,
                              "asof_cutoff_verified": True, **contract, **profile_contract}


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
