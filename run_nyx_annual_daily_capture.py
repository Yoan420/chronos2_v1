"""Capture tomorrow's annual CWE JAO, hydro and lagged-exchange PIT sources.

Schedule this command once per day at 07:00 Europe/Paris.  The command only
captures immutable source observations.  It does not assemble a 366-day
training window, qualify a model, or enable an annual forecast.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path

import pandas as pd

from chronos2_hourly import jao_flowbased as jao
from chronos2_hourly.jao_flowbased import JaoCoreClient
from chronos2_hourly.nyx_annual_jao_source import (
    capture_day, exclusive_cache_lock, verify_daily_capture,
)
from chronos2_hourly.process_lock import exclusive_process_lock
from materialize_jao_core_flowbased import _tls_configuration
import run_nyx_annual_exchange_source as exchange_source
import run_nyx_annual_hydro_source as hydro_source
from run_nyx_annual_jao_source import DEFAULT_CACHE as DEFAULT_JAO_CACHE


PARIS = "Europe/Paris"
DEFAULT_LOG = (Path(__file__).resolve().parent / "runs/live/nyx_annual_cpu"
               / "daily_capture_log.jsonl")


def _clock(now_utc: pd.Timestamp | None) -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if pd.isna(now) or now.tzinfo is None:
        raise ValueError("Capture clock must include a UTC offset")
    return now.tz_convert("UTC")


def target_day(now_utc: pd.Timestamp, requested: date | None = None) -> date:
    """Allow only tomorrow's delivery during today's initial publication window."""
    today = now_utc.tz_convert(PARIS).date()
    day = today + timedelta(days=1)
    if requested is not None and requested != day:
        raise ValueError(f"Only tomorrow's delivery day {day} can be captured today")
    start = jao.expected_initial_publication_utc(day)
    cutoff = jao.expected_cutoff_utc(day)
    if not start <= now_utc < cutoff:
        raise ValueError("Capture requires the D-1 01:15–08:00 Europe/Paris window")
    return day


def capture_daily(*, now_utc: pd.Timestamp | None = None,
                  requested_day: date | None = None,
                  jao_cache: Path = DEFAULT_JAO_CACHE,
                  exchange_archive: Path = exchange_source.DEFAULT_ARCHIVE,
                  hydro_archive: Path = hydro_source.DEFAULT_ARCHIVE,
                  ca_bundle: Path | None = None) -> dict:
    """Run both existing collectors independently, keeping all their safeguards."""
    now = _clock(now_utc)
    day = target_day(now, requested_day)
    jao_cache = Path(jao_cache).resolve()
    exchange_archive = Path(exchange_archive).resolve()
    sources: dict[str, dict] = {}

    try:
        verify, trust_source = _tls_configuration(
            argparse.Namespace(insecure=False, ca_bundle=ca_bundle))
        with exclusive_cache_lock(jao_cache):
            with JaoCoreClient(verify=verify) as client:
                result = capture_day(day=day, cache_root=jao_cache,
                                     client=client, now_utc=None,
                                     tls_trust_source=trust_source)
        sources["jao_initial"] = {
            "state": "COMPLETE", "retrieved_at_utc": result["retrieved_at_utc"],
            "cache_root": str(jao_cache),
        }
    except Exception as error:
        sources["jao_initial"] = {"state": "ERROR", "error": str(error)}

    try:
        with exclusive_process_lock(exchange_archive / "exchange_source.lock"):
            receipt = exchange_source.capture(day.isoformat(), exchange_archive,
                                              now_utc=None)
        sources["lagged_exchange"] = {
            "state": "COMPLETE", "capture_receipt": str(receipt),
        }
    except Exception as error:
        sources["lagged_exchange"] = {"state": "ERROR", "error": str(error)}

    try:
        hydro_archive = Path(hydro_archive).resolve()
        with exclusive_process_lock(hydro_archive / "hydro_source.lock"):
            receipt = hydro_source.capture(day.isoformat(), hydro_archive, now_utc=None)
        sources["public_hydro"] = {"state": "COMPLETE", "capture_receipt": str(receipt)}
    except Exception as error:
        sources["public_hydro"] = {"state": "ERROR", "error": str(error)}

    return {
        "action": "capture",
        "delivery_day": day.isoformat(),
        "started_at_utc": now.isoformat(),
        "state": ("COMPLETE" if all(item["state"] == "COMPLETE"
                                    for item in sources.values()) else "ERROR"),
        "sources": sources,
        "model_inputs_complete": False,
        "forecast_enabled": False,
    }


def verify_daily(day: date, *, jao_cache: Path = DEFAULT_JAO_CACHE,
                 exchange_archive: Path = exchange_source.DEFAULT_ARCHIVE,
                 hydro_archive: Path = hydro_source.DEFAULT_ARCHIVE) -> dict:
    """Read and verify both daily PIT archives without contacting providers."""
    jao_cache = Path(jao_cache).resolve()
    exchange_archive = Path(exchange_archive).resolve()
    sources: dict[str, dict] = {}
    try:
        with exclusive_cache_lock(jao_cache):
            result = verify_daily_capture(jao_cache, day)
        sources["jao_initial"] = {
            "state": "COMPLETE", "retrieved_at_utc": result["retrieved_at_utc"],
            "cache_root": str(jao_cache),
        }
    except Exception as error:
        sources["jao_initial"] = {"state": "ERROR", "error": str(error)}
    try:
        with exclusive_process_lock(exchange_archive / "exchange_source.lock"):
            _, receipt = exchange_source._verify_capture(
                exchange_archive / day.isoformat(), day.isoformat())
        sources["lagged_exchange"] = {
            "state": "COMPLETE", "retrieved_at_utc": receipt["retrieved_at_utc"],
            "capture_receipt": str(exchange_archive / day.isoformat() / "capture.json"),
        }
    except Exception as error:
        sources["lagged_exchange"] = {"state": "ERROR", "error": str(error)}
    try:
        with exclusive_process_lock(Path(hydro_archive) / "hydro_source.lock"):
            _, receipt = hydro_source.verify_capture(Path(hydro_archive) / day.isoformat(), day.isoformat())
        sources["public_hydro"] = {"state": "COMPLETE", "retrieved_at_utc": receipt["retrieved_at_utc"]}
    except Exception as error:
        sources["public_hydro"] = {"state": "ERROR", "error": str(error)}
    return {
        "action": "verify", "delivery_day": day.isoformat(),
        "checked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "state": ("COMPLETE" if all(item["state"] == "COMPLETE"
                                    for item in sources.values()) else "ERROR"),
        "sources": sources, "model_inputs_complete": False,
        "forecast_enabled": False,
    }


def _append_log(path: Path, result: dict) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with exclusive_process_lock(path.with_name(path.name + ".lock")):
        with path.open("ab") as output:
            output.write(line)
            output.flush()
            os.fsync(output.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", type=date.fromisoformat,
                        help="Optional safety check: must equal tomorrow in Europe/Paris")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify an archived day without network access")
    parser.add_argument("--jao-cache", type=Path, default=DEFAULT_JAO_CACHE)
    parser.add_argument("--exchange-archive", type=Path,
                        default=exchange_source.DEFAULT_ARCHIVE)
    parser.add_argument("--hydro-archive", type=Path, default=hydro_source.DEFAULT_ARCHIVE)
    parser.add_argument("--ca-bundle", type=Path,
                        help="Trusted PEM bundle for an enterprise JAO TLS proxy")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG,
                        help="Local JSON-lines result log (not committed to Git)")
    args = parser.parse_args(argv)
    if args.verify_only and args.delivery_day is None:
        parser.error("--verify-only requires --delivery-day")
    try:
        if args.verify_only:
            result = verify_daily(args.delivery_day, jao_cache=args.jao_cache,
                                  exchange_archive=args.exchange_archive, hydro_archive=args.hydro_archive)
        else:
            result = capture_daily(requested_day=args.delivery_day,
                                   jao_cache=args.jao_cache,
                                   exchange_archive=args.exchange_archive,
                                   hydro_archive=args.hydro_archive,
                                   ca_bundle=args.ca_bundle)
    except Exception as error:
        started = pd.Timestamp.now(tz="UTC")
        result = {
            "action": "verify" if args.verify_only else "capture",
            "delivery_day": (args.delivery_day.isoformat() if args.delivery_day
                             else (started.tz_convert(PARIS).date()
                                   + timedelta(days=1)).isoformat()),
            "started_at_utc": started.isoformat(),
            "state": "ERROR", "error": str(error),
            "model_inputs_complete": False, "forecast_enabled": False,
        }
    _append_log(args.log_file, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["state"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
