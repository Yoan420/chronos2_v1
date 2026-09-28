"""Capture four realized EPEX price histories for annual NYX CPU features.

Saturn is queried at the delivery D-1 08:00 Europe/Paris state.  The output
contains the preceding 365 training days plus seven days needed by D-7 lag
features.  It contains no target price for the delivery day.  A successful
capture supplies only the auction_prices source group, not a complete feature
bundle or a forecast.
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
import yaml

from chronos2_hourly.nyx_annual_live_preflight import (
    ROOT, SOURCE_PROTOCOL, delivery_grid, sha256, validate_source_receipt,
)
from chronos2_hourly.process_lock import exclusive_process_lock
from chronos2_modular.saturn import (
    create_saturn_client, fetch_saturn_series_from_client,
)


ZONE_CONFIGS = {
    "FR": "chronos2_hourly_fr_residual_v1.yaml",
    "DE": "chronos2_hourly_de_residual_candidate_v1.yaml",
    "BE": "chronos2_hourly_be_residual_candidate_v1.yaml",
    "NL": "chronos2_hourly_nl_residual_candidate_v1.yaml",
}
SERIES = {
    "FR": "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh",
    "DE": "power.price.da.de_lu.bzn.hourly.entsoe.utc.cdh.eurmwh",
    "BE": "power.price.da.be.bzn.hourly.entsoe.utc.cdh.eurmwh",
    "NL": "power.price.da.nl.bzn.hourly.entsoe.utc.cdh.eurmwh",
}
LAG_DAYS = 7
TRAIN_DAYS = 365


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _day(value: str) -> date:
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, "Delivery day must be YYYY-MM-DD")
    return parsed


def price_grid(delivery_day: str) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
    """Return 372 complete Paris civil days, ending before delivery D."""
    day = _day(delivery_day)
    _, current, cutoff = delivery_grid(delivery_day)
    start = pd.Timestamp(day - timedelta(days=TRAIN_DAYS + LAG_DAYS),
                         tz="Europe/Paris").tz_convert("UTC")
    expected = pd.date_range(start, current[0], freq="h", inclusive="left")
    require(len(expected) >= 24 * (TRAIN_DAYS + LAG_DAYS) - 1
            and expected[-1] < current[0], "Auction history grid differs")
    return expected, cutoff


def load_plan(root: Path = ROOT) -> dict:
    """Use the four checked-in production target contracts, with no fallback."""
    plan: dict[str, Any] = {"series": {}, "config_sha256": {}}
    endpoints, authors = set(), set()
    for zone, filename in ZONE_CONFIGS.items():
        path = root / filename
        require(path.is_file(), f"Missing tracked target config: {filename}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        data, zone_config = config["data"], config["zones"][zone]
        target = zone_config["target"]
        require(target["series"] == SERIES[zone]
                and target.get("naive_timezone") == "UTC",
                f"{zone}: canonical UTC EPEX target contract changed")
        endpoints.add(str(data["saturn_url"]))
        authors.add(str(data["saturn_author"]))
        plan["series"][zone] = SERIES[zone]
        plan["config_sha256"][filename] = sha256(path)
    require(len(endpoints) == len(authors) == 1,
            "Four target configs require one common Saturn connection")
    plan["saturn_url"] = endpoints.pop()
    plan["saturn_author"] = authors.pop()
    return plan


def collect(client: Any, delivery_day: str, *,
            now_utc: pd.Timestamp | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Read one as-of Saturn state and reject missing or future target hours."""
    expected, cutoff = price_grid(delivery_day)
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    require(now.tzinfo is not None and now.tz_convert("UTC") >= cutoff,
            "D-1 08:00 delivery cutoff has not occurred")
    values: dict[str, pd.DataFrame] = {}
    for zone, series_name in SERIES.items():
        fetched = fetch_saturn_series_from_client(
            client, series_name, expected[0], expected[-1], "UTC",
            revision_date=cutoff, naive_timezone="UTC",
            incomplete_dst_policy="raise", nocache=True,
        )
        require(isinstance(fetched.index, pd.DatetimeIndex)
                and str(fetched.index.tz) == "UTC" and fetched.index.is_unique,
                f"{zone}: Saturn target UTC timeline invalid")
        selected = fetched.reindex(expected)
        numeric = pd.to_numeric(selected, errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(numeric)
        require(not missing.any(),
                f"{zone}: {int(missing.sum())} auction price hours missing at D-1 08; "
                f"first={expected[np.flatnonzero(missing)[0]] if missing.any() else 'none'}")
        values[zone] = pd.DataFrame({"price_eur_mwh": numeric},
                                    index=expected.copy().rename("timestamp_utc"))
    return values, {"cutoff_utc": cutoff.isoformat(), "first_hour_utc": expected[0].isoformat(),
                    "last_hour_utc": expected[-1].isoformat(),
                    "hours_per_zone": len(expected), "training_days": TRAIN_DAYS,
                    "lag_warmup_days": LAG_DAYS}


def _write_immutable_parquet(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(pd.read_parquet(path).equals(frame),
                f"Existing auction artifact differs: {path}")
        return sha256(path)
    with tempfile.NamedTemporaryFile(prefix=path.stem + ".", suffix=".tmp.parquet",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary)
        require(pd.read_parquet(temporary).equals(frame),
                f"Auction artifact round trip differs: {path}")
        if path.exists():
            require(pd.read_parquet(path).equals(frame),
                    f"Concurrent auction artifact differs: {path}")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(path)


def publish(bundle: Path, delivery_day: str, values: dict[str, pd.DataFrame],
            evidence: dict, plan: dict, *, now_utc: pd.Timestamp | None = None) -> Path:
    """Bind all four complete histories in an immutable dated bundle."""
    bundle = bundle.resolve()
    expected, cutoff = price_grid(delivery_day)
    require(set(values) == set(SERIES) and evidence["cutoff_utc"] == cutoff.isoformat(),
            "Auction capture identity differs")
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    require(now.tzinfo is not None and now.tz_convert("UTC") >= cutoff,
            "D-1 08:00 delivery cutoff has not occurred")
    hashes = {}
    for zone in SERIES:
        frame = values[zone]
        require(frame.index.equals(expected.rename("timestamp_utc"))
                and list(frame.columns) == ["price_eur_mwh"]
                and np.isfinite(frame.price_eur_mwh.to_numpy(dtype=float)).all(),
                f"{zone}: incomplete auction history")
        relative = f"source_artifacts/auction_prices/{zone}.parquet"
        hashes[relative] = _write_immutable_parquet(frame, bundle / relative)
    receipt = {
        "protocol": SOURCE_PROTOCOL, "source_group": "auction_prices",
        "delivery_day": delivery_day, "state": "COMPLETE",
        "asof_cutoff_verified": True, "training_window_complete": True,
        "asof_state_utc": cutoff.isoformat(),
        "availability_basis": "Saturn revision_date query as-of D-1 08:00",
        "provider_publication_timestamp_verified": False,
        "publication_vintage_limit": "Saturn as-of state does not certify original provider insertion times",
        "retrieved_at_utc": now.tz_convert("UTC").isoformat(),
        "series": dict(SERIES),
        "config_sha256": plan["config_sha256"],
        "saturn_endpoint_sha256": hashlib.sha256(plan["saturn_url"].encode()).hexdigest(),
        "artifact_sha256": hashes,
        **evidence, "model_inputs_complete": False,
    }
    validate_source_receipt(receipt, group="auction_prices", day=delivery_day,
                            bundle=bundle, cutoff=cutoff)
    destination = bundle / "source_receipts/auction_prices.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        # Re-running the same as-of capture is allowed; the retrieval clock is
        # operational metadata, while every source hour and hash must match.
        receipt["retrieved_at_utc"] = previous.get("retrieved_at_utc")
        require(previous == receipt, "Existing auction receipt differs")
        return destination
    with tempfile.NamedTemporaryFile(prefix="auction_prices.", suffix=".tmp.json",
                                     dir=destination.parent, delete=False, mode="w",
                                     encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    try:
        require(not destination.exists(), "Concurrent auction receipt appeared")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(argv)
    day = _day(args.delivery_day)
    bundle = args.bundle or ROOT / "runs/live/nyx_annual_cpu" / day.isoformat()
    expected, cutoff = price_grid(args.delivery_day)
    require(pd.Timestamp.now(tz="UTC") >= cutoff,
            "D-1 08:00 delivery cutoff has not occurred")
    plan = load_plan()
    client = create_saturn_client(plan["saturn_url"],
                                  os.getenv("SATURN_AUTHOR") or plan["saturn_author"])
    with exclusive_process_lock(bundle / "auction_prices_live.lock"):
        values, evidence = collect(client, args.delivery_day)
        receipt = publish(bundle, args.delivery_day, values, evidence, plan)
    print(json.dumps({"state": "COMPLETE", "source_group": "auction_prices",
                      "delivery_day": args.delivery_day, "receipt": str(receipt),
                      "hours_per_zone": len(expected), "model_inputs_complete": False},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
