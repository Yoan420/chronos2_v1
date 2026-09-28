"""Capture the thirteen CWE thermal Pmax inputs from Saturn for a live day.

Every daily value is a Saturn state queried at that delivery day's civil D-1
08:00 Europe/Paris cutoff. The resulting 36 columns are a partial provider
fleet descriptor. Missing states stay NaN with availability zero. This source
receipt does not certify provider publication times or complete annual inputs.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from chronos2_hourly.nyx_annual_live_preflight import (
    ROOT, SOURCE_PROTOCOL, delivery_grid, load_schema, sha256,
    validate_source_receipt,
)
from chronos2_hourly.process_lock import exclusive_process_lock
from chronos2_modular.saturn import create_saturn_client
from marginal_cost_expert.sources import (
    _daily_values, _one_day, capacity_catalog, civil_cutoff,
)
from run_nyx_annual_auction_prices_source import load_plan as load_saturn_plan


ZONES = ("FR", "DE", "BE", "NL")
SOURCES = tuple(sorted((
    "fr_ccgt", "fr_gt", "de_ccgt", "de_gt", "de_coal", "de_lignite",
    "be_ccgt", "be_gt", "be_nuclear", "nl_ccgt", "nl_gt", "nl_coal",
    "nl_nuclear",
)))
TECHNOLOGIES = ("ccgt", "gt", "coal", "lignite", "nuclear")
COLUMNS = tuple(
    column for source in SOURCES
    for column in (f"thermal__{source}_pmax_gw",
                   f"thermal__{source}_pmax_gw__available")
) + tuple(
    column for technology in TECHNOLOGIES
    for column in (f"thermal__own_{technology}_pmax_gw",
                   f"thermal__own_{technology}_pmax_gw__available")
)
REQUEST = {
    "revision_freq": {"days": 1}, "revision_time": {"hour": 8},
    "revision_tz": "Europe/Paris", "maturity_offset": {"days": 1},
    "maturity_time": {"hour": 0},
}
DEFAULT_CACHE = ROOT / "data/pit/nyx_annual_thermal"
CACHE_PROTOCOL = "nyx_annual_thermal_daily_state_v1"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def plan(root: Path = ROOT) -> dict:
    """Resolve only the historical thirteen technology contracts."""
    catalogue = capacity_catalog(list(ZONES))
    aliases = {source: f"{source}_available_gw" for source in SOURCES}
    require(all(alias in catalogue for alias in aliases.values()),
            "Thirteen Saturn capacity aliases are required")
    specs = {source: catalogue[alias] for source, alias in aliases.items()}
    for source, spec in specs.items():
        require(spec["role"] == "capacity_forecast" and spec["unit"] == "GW"
                and spec["series"].startswith("power.nrjscan.")
                and ".availability.pmax." in spec["series"]
                and spec["zone"].lower() == source.split("_")[0],
                f"{source}: Saturn technology source changed")
    config = load_saturn_plan(root)
    return {"specs": specs, "aliases": aliases,
            "saturn_url": config["saturn_url"],
            "saturn_author": config["saturn_author"],
            "config_sha256": config["config_sha256"]}


def grids(delivery_day: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.Timestamp]:
    full, _, cutoff = delivery_grid(delivery_day)
    day = date.fromisoformat(delivery_day)
    days = pd.date_range(day - timedelta(days=365), day, freq="D")
    require(len(days) == 366, "Expected 365 training days plus delivery")
    return days, full, cutoff


def _cache_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _independent_state(client, source: str, spec: dict, day: pd.Timestamp,
                       *, cache: Path | None, identity_base: dict,
                       retrieved_at_utc: pd.Timestamp):
    """Retain the independent as-of query; never invent a state from staircase."""
    identity = {**identity_base, "source": source, "series_spec": spec,
                "delivery_day": day.date().isoformat(),
                "asof_query_utc": civil_cutoff(day).isoformat()}
    path = None if cache is None else cache / source / f"{day.date().isoformat()}.json"
    if path is not None and path.exists():
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            body = envelope["state"]
            require(envelope.get("sha256") == hashlib.sha256(_cache_bytes(body)).hexdigest(),
                    f"{source}/{day.date()}: thermal cache checksum changed")
            require(body.get("protocol") == CACHE_PROTOCOL and body.get("identity") == identity,
                    f"{source}/{day.date()}: thermal cache contract changed")
            stamp = pd.Timestamp(body["retrieved_at_utc"])
            require(stamp.tzinfo is not None and stamp >= civil_cutoff(day),
                    f"{source}/{day.date()}: cached query predates its as-of state")
            value = body["value"]
            require(value is None or (type(value) in (int, float) and np.isfinite(value) and value >= 0.),
                    f"{source}/{day.date()}: invalid cached capacity")
            return value
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"{source}/{day.date()}: malformed immutable thermal cache") from error
    value = _one_day(client, spec["series"], day)
    require(value is None or (np.isfinite(value) and value >= 0),
            f"{source}/{day.date()}: invalid independently queried capacity")
    if path is not None:
        body = {"protocol": CACHE_PROTOCOL, "identity": identity,
                "value": None if value is None else float(value),
                "retrieved_at_utc": retrieved_at_utc.isoformat()}
        content = _cache_bytes({"state": body, "sha256": hashlib.sha256(_cache_bytes(body)).hexdigest()}) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="thermal_state.", suffix=".tmp", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        try:
            # collect holds the shared cache lock. Publication is atomic, so an
            # interrupted download never becomes an apparently complete cache.
            require(not path.exists(), f"Concurrent thermal cache state appeared: {path}")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return value


def collect(client: Any, delivery_day: str, *,
            now_utc: pd.Timestamp | None = None,
            cache: Path | None = None) -> tuple[dict[str, pd.Series], dict]:
    """Capture daily states, optionally resuming their immutable local cache.

    Every run still reads the supplier's complete block staircase and compares
    it with all independent per-day states. A new day requires only thirteen
    new individual queries. Legacy callers retain uncached behavior.
    """
    if cache is None:
        return _collect(client, delivery_day, now_utc=now_utc, cache=None)
    cache = Path(cache).resolve()
    with exclusive_process_lock(cache / "thermal_state_cache.lock"):
        return _collect(client, delivery_day, now_utc=now_utc, cache=cache)


def _collect(client: Any, delivery_day: str, *,
             now_utc: pd.Timestamp | None, cache: Path | None) -> tuple[dict[str, pd.Series], dict]:
    """Read the complete dynamic daily span without substituting absent states."""
    days, _, cutoff = grids(delivery_day)
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    require(now.tzinfo is not None and now.tz_convert("UTC") >= cutoff,
            "D-1 08:00 delivery cutoff has not occurred")
    source_plan = plan()
    from marginal_cost_expert import sources as source_reader
    identity_base = {"collector_code_sha256": sha256(Path(__file__)),
        "reader_code_sha256": sha256(Path(source_reader.__file__)),
        "saturn_endpoint_sha256": hashlib.sha256(source_plan["saturn_url"].encode()).hexdigest(),
        "staircase_request": REQUEST}
    values: dict[str, pd.Series] = {}
    state_checks: dict[str, list[dict]] = {}
    for source, spec in source_plan["specs"].items():
        raw = client.block_staircase(
            spec["series"], from_value_date=days[0],
            to_value_date=days[-1], **REQUEST,
        )
        bulk = _daily_values(raw).reindex(days)
        numeric = bulk.to_numpy(dtype=float)
        require(not np.isinf(numeric).any()
                and not ((np.isfinite(numeric)) & (numeric < 0)).any(),
                f"{source}: invalid Saturn Pmax value")
        evidence = []
        # The staircase call is an efficient extraction, but a COMPLETE receipt
        # requires an independent revision_date state for *every* civil day.
        # One verified daily Pmax is then broadcast to its 23/24/25 UTC hours.
        for day in days:
            state = _independent_state(client, source, spec, day, cache=cache,
                                       identity_base=identity_base, retrieved_at_utc=now.tz_convert("UTC"))
            block = bulk.loc[day]
            require((state is None and not np.isfinite(block)) or
                    (state is not None and np.isfinite(block) and
                     np.isclose(state, float(block), rtol=0, atol=1e-9)),
                    f"{source}/{day.date()}: block staircase disagrees with D-1 state")
            evidence.append({"day": day.date().isoformat(),
                             "value": state,
                             "asof_query_utc": civil_cutoff(day).isoformat()})
        values[source] = pd.Series(numeric, index=days.copy(), name="pmax_gw")
        state_checks[source] = evidence
    return values, {"cutoff_utc": cutoff.isoformat(),
                    "requested_first_day": days[0].date().isoformat(),
                    "requested_last_day": days[-1].date().isoformat(),
                    "requested_days_per_series": len(days),
                    "independent_daily_state_checks": state_checks,
                    "states_verified_per_series": len(days)}


def source_hourly(daily: pd.Series, delivery_day: str) -> pd.DataFrame:
    days, full, cutoff = grids(delivery_day)
    require(daily.index.equals(days) and daily.name == "pmax_gw",
            "Daily Pmax index or name differs")
    values = daily.to_numpy(dtype=float)
    require(not np.isinf(values).any() and
            not ((np.isfinite(values)) & (values < 0)).any(),
            "Invalid Pmax levels")
    local_days = full.tz_convert("Europe/Paris").tz_localize(None).normalize()
    positions = days.get_indexer(local_days)
    require((positions >= 0).all(), "Thermal UTC grid exceeds civil days")
    origins = pd.DatetimeIndex([civil_cutoff(day) for day in days])
    asof = origins.take(positions)
    require((asof <= cutoff).all(), "Thermal state query after delivery cutoff")
    frame = pd.DataFrame({"pmax_gw": values[positions],
                          "asof_query_utc": asof},
                         index=full.copy().rename("timestamp_utc"))
    return frame


def build_features(sources: dict[str, pd.DataFrame], delivery_day: str) -> dict[str, pd.DataFrame]:
    """Recreate only the 36 thermal descriptors in their audited order."""
    _, full, cutoff = grids(delivery_day)
    require(set(sources) == set(SOURCES), "Exactly thirteen Pmax sources required")
    values = {}
    for source in SOURCES:
        frame = sources[source]
        require(frame.index.equals(full.rename("timestamp_utc"))
                and list(frame.columns) == ["pmax_gw", "asof_query_utc"],
                f"{source}: hourly source schema or grid differs")
        numeric = frame["pmax_gw"].to_numpy(dtype=float)
        require(not np.isinf(numeric).any() and
                not ((np.isfinite(numeric)) & (numeric < 0)).any(),
                f"{source}: invalid source level")
        origins = pd.DatetimeIndex(frame["asof_query_utc"])
        require(str(origins.tz) == "UTC" and (origins <= cutoff).all(),
                f"{source}: future query state")
        values[source] = numeric
    outputs = {}
    for zone in ZONES:
        columns: dict[str, np.ndarray] = {}
        for source in SOURCES:
            name = f"thermal__{source}_pmax_gw"
            columns[name] = values[source]
            columns[name + "__available"] = np.isfinite(values[source]).astype(float)
        for technology in TECHNOLOGIES:
            source = f"{zone.lower()}_{technology}"
            name = f"thermal__own_{technology}_pmax_gw"
            own = values[source] if source in values else np.full(len(full), np.nan)
            columns[name] = own
            columns[name + "__available"] = np.isfinite(own).astype(float)
        outputs[zone] = pd.DataFrame(columns, index=full.copy().rename("timestamp_utc"))
        require(tuple(outputs[zone].columns) == COLUMNS,
                f"{zone}: thermal feature order differs")
    return outputs


def _write_immutable(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(pd.read_parquet(path).equals(frame),
                f"Existing thermal artifact differs: {path}")
        return sha256(path)
    with tempfile.NamedTemporaryFile(prefix=path.stem + ".", suffix=".tmp.parquet",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary)
        require(pd.read_parquet(temporary).equals(frame),
                f"Thermal artifact round trip differs: {path}")
        require(not path.exists(), f"Concurrent thermal artifact appeared: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(path)


def publish(bundle: Path, delivery_day: str, daily: dict[str, pd.Series],
            evidence: dict, source_plan: dict, *,
            now_utc: pd.Timestamp | None = None) -> Path:
    """Freeze 13 checked states and 4 thermal projections in a source receipt."""
    bundle = bundle.resolve()
    days, full, cutoff = grids(delivery_day)
    require(set(daily) == set(SOURCES)
            and evidence["cutoff_utc"] == cutoff.isoformat()
            and evidence["requested_first_day"] == days[0].date().isoformat()
            and evidence["requested_last_day"] == days[-1].date().isoformat()
            and evidence["states_verified_per_series"] == len(days),
            "Thermal capture identity differs")
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    require(now.tzinfo is not None and now.tz_convert("UTC") >= cutoff,
            "D-1 08:00 delivery cutoff has not occurred")
    require(source_plan["specs"] == plan()["specs"],
            "Saturn Pmax source plan changed")
    checks = evidence.get("independent_daily_state_checks", {})
    require(set(checks) == set(SOURCES),
            "Independent D-1 daily state evidence missing")
    for source in SOURCES:
        require(daily[source].index.equals(days) and
                len(checks[source]) == len(days),
                f"{source}: 366 daily as-of states required")
        for day, value, checked in zip(days, daily[source].to_numpy(dtype=float),
                                       checks[source], strict=True):
            same_value = (checked.get("value") is None and np.isnan(value)) or (
                checked.get("value") is not None and np.isfinite(value) and
                np.isclose(float(checked["value"]), value, rtol=0, atol=1e-9))
            require(checked.get("day") == day.date().isoformat() and
                    checked.get("asof_query_utc") == civil_cutoff(day).isoformat()
                    and same_value,
                    f"{source}/{day.date()}: D-1 state evidence differs")
    hourly = {source: source_hourly(daily[source], delivery_day)
              for source in SOURCES}
    features = build_features(hourly, delivery_day)
    schema = load_schema()
    for zone in ZONES:
        expected = [name for name in schema["families"]["cwe_absolute_2000"]["columns"][zone]
                    if name.startswith("thermal__")]
        require(list(features[zone].columns) == expected,
                f"{zone}: thermal schema differs from CWE 503")
    hashes = {}
    for source, frame in hourly.items():
        relative = f"source_artifacts/thermal_capacity/{source}.parquet"
        hashes[relative] = _write_immutable(frame, bundle / relative)
    for zone, frame in features.items():
        relative = f"source_artifacts/thermal_capacity/features_{zone}.parquet"
        hashes[relative] = _write_immutable(frame, bundle / relative)
    missing = {source: [day.date().isoformat() for day in days[daily[source].isna()]]
               for source in SOURCES}
    receipt = {
        "protocol": SOURCE_PROTOCOL, "source_group": "thermal_capacity",
        "delivery_day": delivery_day, "state": "COMPLETE",
        "asof_cutoff_verified": True, "training_window_complete": True,
        "asof_state_utc": cutoff.isoformat(),
        "availability_basis": "Saturn daily block_staircase at each civil D-1 08:00; independent get probes, including missing days",
        "provider_publication_timestamp_verified": False,
        "provider_fleet_coverage_certified": False,
        "capacity_semantics": "Daily provider-fleet Pmax forecast, GW, broadcast to 23/24/25 physical hours",
        "missing_values_filled": False,
        "missing_days_by_series": missing,
        "series": {source: spec["series"] for source, spec in source_plan["specs"].items()},
        "config_sha256": source_plan["config_sha256"],
        "collector_code_sha256": sha256(Path(__file__)),
        "saturn_endpoint_sha256": hashlib.sha256(source_plan["saturn_url"].encode()).hexdigest(),
        "retrieved_at_utc": now.tz_convert("UTC").isoformat(),
        "artifact_sha256": hashes,
        "rows_per_source_and_zone": len(full),
        "feature_columns_per_zone": list(COLUMNS),
        **evidence, "model_inputs_complete": False,
    }
    validate_source_receipt(receipt, group="thermal_capacity", day=delivery_day,
                            bundle=bundle, cutoff=cutoff)
    destination = bundle / "source_receipts/thermal_capacity.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        receipt["retrieved_at_utc"] = previous.get("retrieved_at_utc")
        require(previous == receipt, "Existing thermal receipt differs")
        return destination
    with tempfile.NamedTemporaryFile(prefix="thermal_capacity.", suffix=".tmp.json",
                                     dir=destination.parent, delete=False,
                                     mode="w", encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    try:
        require(not destination.exists(), "Concurrent thermal receipt appeared")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="Immutable per-series, per-day Saturn state cache")
    args = parser.parse_args(argv)
    day = date.fromisoformat(args.delivery_day)
    require(day.isoformat() == args.delivery_day, "Delivery day must be YYYY-MM-DD")
    bundle = args.bundle or ROOT / "runs/live/nyx_annual_cpu" / args.delivery_day
    source_plan = plan()
    client = create_saturn_client(source_plan["saturn_url"],
                                  os.getenv("SATURN_AUTHOR") or source_plan["saturn_author"])
    with exclusive_process_lock(bundle / "thermal_capacity_live.lock"):
        daily, evidence = collect(client, args.delivery_day, cache=args.cache)
        receipt = publish(bundle, args.delivery_day, daily, evidence, source_plan)
    print(json.dumps({"state": "COMPLETE", "source_group": "thermal_capacity",
                      "delivery_day": args.delivery_day, "receipt": str(receipt),
                      "model_inputs_complete": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
