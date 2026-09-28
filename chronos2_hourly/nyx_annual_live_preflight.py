"""Structural input gate for the annual CWE CPU recipes on a future day.

The ordered feature schema is code-only metadata copied from the audited 2026-09-23
plans. This module never reconstructs a feature, downloads a source, trains a model,
or interprets a historical matrix as a live forecast. A producer must place a dated
bundle with fresh matrices, prices, references, and source receipts before this gate
can pass. Passing this gate alone does not qualify a model for production.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "config/nyx_annual_cpu_ordered_features.json"
PROTOCOL = "nyx_annual_cpu_live_inputs_v1"
SOURCE_PROTOCOL = "nyx_annual_cpu_live_source_receipt_v1"
ZONES = ("FR", "DE", "BE", "NL")
FAMILIES = {"fr_residual_1000": 449, "cwe_residual_2000": 123,
            "cwe_absolute_2000": 503}
SOURCE_GROUPS = ("saturn", "auction_prices", "nyx_quantiles",
                 "scarcity_confirmed_pair", "jao_initial", "public_hydro",
                 "fuel", "thermal_capacity", "lagged_exchange")
SCHEMA_COLUMNS_SHA256 = "daefca0c17ee7a425ec12ccfe5e03bb25699f36960464a678770582387806e3b"
SOURCE_PLAN_SHA256 = {
    "fr_residual_1000": "c548cc6069724349d6ed1b6cb4d2bfc6c6fd567709acec62e415f64ca4fce7f1",
    "cwe_residual_2000": "5b5d94ae39e9386b665797054d3a052c05d84e1e85eb653a57edd53c1af36357",
    "cwe_absolute_2000": "5b5d94ae39e9386b665797054d3a052c05d84e1e85eb653a57edd53c1af36357",
}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_schema(path: Path = SCHEMA) -> dict:
    schema = json.loads(path.read_text(encoding="utf-8"))
    _require(schema.get("protocol") == "nyx_annual_cpu_ordered_feature_schema_v1"
             and schema.get("zones") == list(ZONES)
             and set(schema.get("families", {})) == set(FAMILIES)
             and schema.get("source_plan_sha256") == SOURCE_PLAN_SHA256,
             "Annual CPU schema identity differs")
    for family, width in FAMILIES.items():
        _require(set(schema["families"][family].get("columns", {})) == set(ZONES),
                 f"{family}: four country schemas required")
        for zone in ZONES:
            columns = schema["families"][family]["columns"][zone]
            _require(isinstance(columns, list) and len(columns) == len(set(columns)) == width
                     and all(isinstance(name, str) and name and not any(
                         forbidden in name.lower() for forbidden in
                         ("storm", "actual", "observed", "target")) for name in columns),
                     f"{family}/{zone}: ordered column contract differs")
    raw = json.dumps({family: schema["families"][family]["columns"]
                      for family in sorted(FAMILIES)}, sort_keys=True,
                     separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    _require(hashlib.sha256(raw).hexdigest() == SCHEMA_COLUMNS_SHA256,
             "Annual CPU ordered columns changed")
    return schema


def delivery_grid(delivery_day: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.Timestamp]:
    day = date.fromisoformat(delivery_day)
    _require(day.isoformat() == delivery_day, "Delivery day must be YYYY-MM-DD")
    start = pd.Timestamp(day - timedelta(days=365), tz="Europe/Paris")
    stop = pd.Timestamp(day + timedelta(days=1), tz="Europe/Paris")
    full = pd.date_range(start, stop, freq="h", inclusive="left").tz_convert("UTC")
    current = full[full.tz_convert("Europe/Paris").date == day]
    cutoff = pd.Timestamp(f"{day - timedelta(days=1)} 08:00",
                          tz="Europe/Paris").tz_convert("UTC")
    _require(len(current) in (23, 24, 25), "Invalid physical delivery-day grid")
    return full, current, cutoff


def _utc_index(frame: pd.DataFrame, expected: pd.DatetimeIndex, label: str) -> None:
    index = frame.index
    _require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
             and index.is_unique and index.is_monotonic_increasing
             and not index.hasnans and index.equals(expected),
             f"{label}: exact trailing 365-day plus delivery UTC grid required")


def validate_feature_frame(frame: pd.DataFrame, columns: list[str],
                           expected: pd.DatetimeIndex, label: str) -> None:
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique,
             f"{label}: DataFrame with unique columns required")
    _utc_index(frame, expected, label)
    _require(list(frame.columns) == columns, f"{label}: ordered annual feature schema differs")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: nonnumeric feature") from error
    _require(not np.isinf(values).any(), f"{label}: infinite feature")
    for name in columns:
        if not name.endswith("__available"):
            continue
        base = name.removesuffix("__available")
        flag = frame[name].to_numpy(dtype=float)
        if name == "extra_jao__available":
            # JAO exposes one shared flag for 27 descriptors (four of them in
            # the compact projection), not a value column named extra_jao.
            jao = [column for column in columns if column.startswith("extra_jao_")
                   and column != name]
            _require(bool(jao), f"{label}: JAO shared flag has no descriptors")
            actual = frame[jao].notna().all(axis=1).to_numpy(dtype=float)
        else:
            _require(base in frame, f"{label}: availability flag without value: {name}")
            actual = frame[base].notna().to_numpy(dtype=float)
        _require(np.isin(flag, (0., 1.)).all() and np.array_equal(flag, actual),
                 f"{label}: availability flag/value mismatch: {name}")


def _utc_stamps(values: pd.Series, label: str) -> pd.DatetimeIndex:
    try:
        stamps = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid UTC origin timestamps") from error
    _require(not stamps.hasnans, f"{label}: missing forecast origin")
    return stamps


def validate_baseline(frame: pd.DataFrame, full: pd.DatetimeIndex,
                      current: pd.DatetimeIndex, cutoff: pd.Timestamp, label: str) -> None:
    _utc_index(frame, full, label)
    _require({"actual", "nyx__q50", "forecast_origin_utc"} <= set(frame),
             f"{label}: actual, NYX q50, and forecast origin required")
    past = frame.loc[~frame.index.isin(current), "actual"].to_numpy(dtype=float)
    future = frame.loc[current, "actual"].to_numpy(dtype=float)
    q50 = frame["nyx__q50"].to_numpy(dtype=float)
    _require(np.isfinite(past).all() and np.isnan(future).all()
             and np.isfinite(q50).all(),
             f"{label}: complete observed training prices, future labels absent, finite NYX q50 required")
    origins = _utc_stamps(frame.loc[current, "forecast_origin_utc"], label)
    _require((origins <= cutoff).all(), f"{label}: NYX q50 origin after D-1 08:00 cutoff")


def validate_reference(frame: pd.DataFrame, current: pd.DatetimeIndex,
                       cutoff: pd.Timestamp, label: str) -> None:
    _utc_index(frame, current, label)
    _require({"reference", "forecast_origin_utc"} <= set(frame),
             f"{label}: selected scarcity reference and origin required")
    _require(np.isfinite(frame["reference"].to_numpy(dtype=float)).all(),
             f"{label}: nonfinite selected scarcity reference")
    origins = _utc_stamps(frame["forecast_origin_utc"], label)
    _require((origins <= cutoff).all(), f"{label}: reference origin after D-1 08:00 cutoff")


def _inside(bundle: Path, relative: str) -> Path:
    path = (bundle / relative).resolve()
    _require(path.is_relative_to(bundle.resolve()), "Source artifact escapes live bundle")
    return path


def validate_source_receipt(receipt: dict, *, group: str, day: str,
                            bundle: Path, cutoff: pd.Timestamp) -> None:
    _require(receipt.get("protocol") == SOURCE_PROTOCOL
             and receipt.get("source_group") == group
             and receipt.get("delivery_day") == day
             and receipt.get("state") == "COMPLETE"
             and receipt.get("asof_cutoff_verified") is True
             and receipt.get("training_window_complete") is True,
             f"{group}: prospective source receipt incomplete")
    stamps = receipt.get("latest_available_utc")
    _require(isinstance(stamps, str), f"{group}: latest availability time required")
    available = pd.Timestamp(stamps)
    _require(available.tzinfo is not None and available.tz_convert("UTC") <= cutoff,
             f"{group}: source availability exceeds D-1 08:00 cutoff")
    hashes = receipt.get("artifact_sha256")
    _require(isinstance(hashes, dict) and bool(hashes),
             f"{group}: bound source artifacts required")
    for relative, expected in hashes.items():
        _require(isinstance(relative, str) and isinstance(expected, str)
                 and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected),
                 f"{group}: malformed source checksum")
        path = _inside(bundle, relative)
        _require(path.is_file() and sha256(path) == expected,
                 f"{group}: source artifact missing or changed: {relative}")


def inspect_bundle(bundle: Path, delivery_day: str,
                   schema_path: Path = SCHEMA) -> dict:
    """Inspect a future bundle; return all failures without changing files."""
    full, current, cutoff = delivery_grid(delivery_day)
    schema = load_schema(schema_path)
    bundle = bundle.resolve()
    checks = []

    def check(name: str, action) -> None:
        try:
            action()
            checks.append({"input": name, "passed": True})
        except Exception as error:
            checks.append({"input": name, "passed": False, "reason": str(error)})

    def read_parquet(relative: str) -> pd.DataFrame:
        path = _inside(bundle, relative)
        _require(path.is_file(), f"Missing live input: {relative}")
        return pd.read_parquet(path)

    for group in SOURCE_GROUPS:
        relative = f"source_receipts/{group}.json"

        def source_action(group=group, relative=relative):
            path = _inside(bundle, relative)
            _require(path.is_file(), f"Missing live source receipt: {relative}")
            validate_source_receipt(json.loads(path.read_text(encoding="utf-8")),
                                    group=group, day=delivery_day, bundle=bundle,
                                    cutoff=cutoff)

        check(f"source/{group}", source_action)

    for family in FAMILIES:
        for zone in ZONES:
            relative = f"features/{family}/{zone}.parquet"
            columns = schema["families"][family]["columns"][zone]

            def feature_action(relative=relative, columns=columns):
                validate_feature_frame(read_parquet(relative), columns, full, relative)

            check(f"feature/{family}/{zone}", feature_action)

    for zone in ZONES:
        relative = f"baseline/{zone}.parquet"

        def baseline_action(relative=relative):
            validate_baseline(read_parquet(relative), full, current, cutoff, relative)

        check(f"baseline/{zone}", baseline_action)
    for zone in ("FR", "BE", "NL"):
        relative = f"reference/{zone}.parquet"

        def reference_action(relative=relative):
            validate_reference(read_parquet(relative), current, cutoff, relative)

        check(f"reference/{zone}", reference_action)
    return {"protocol": PROTOCOL, "delivery_day": delivery_day,
            "bundle": str(bundle), "input_bundle_valid": all(c["passed"] for c in checks),
            "checks": checks,
            "scope": "Structural inputs only; no retraining, model score, or live rollout certification"}
