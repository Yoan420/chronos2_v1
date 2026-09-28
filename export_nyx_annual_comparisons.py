"""Export verified official EPEX/Storm observations for CPU evaluation only.

These files are never inputs to a model. The evaluator opens their values only
after all predictions have been sealed, and rechecks this source evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path, PureWindowsPath, PurePosixPath

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.nuclear_reporting_refresh import refresh_nuclear_reporting_sources, verify_refreshed_observations
from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
from chronos2_hourly.reporting_observations import EPEX_REPORTING_SOURCE_KIND, EPEX_REPORTING_POLICY
from chronos2_hourly.nyx_annual_live_preflight import ROOT, sha256
from chronos2_hourly.nyx_annual_saturn_source import _immutable, _encoded, _parquet
from chronos2_hourly.process_lock import exclusive_process_lock
from run_nyx_annual_auction_prices_source import ZONE_CONFIGS

PROTOCOL = "nyx_annual_official_comparisons_v1"
TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin", "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
ARTIFACTS = ("statistics_history_audit.json", "inputs/observed_latest.parquet", "inputs/storm_dashboard_official_statistics.parquet")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def evaluation_grid(first, stop):
    start, end = date.fromisoformat(first), date.fromisoformat(stop)
    require(0 < (end - start).days <= 365, "Evaluation dates must cover 1-365 civil days")
    return pd.date_range(pd.Timestamp(start, tz="Europe/Paris"), pd.Timestamp(end, tz="Europe/Paris"),
                         freq="h", inclusive="left").tz_convert("UTC").rename("timestamp_utc")


def _inside(root, relative):
    require(isinstance(relative, str) and relative and not PureWindowsPath(relative).drive
            and not PureWindowsPath(relative).is_absolute() and not PurePosixPath(relative).is_absolute(),
            "Absolute comparison path refused")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Comparison path escapes archive")
    return path


def read_snapshot(directory, zone, stop):
    directory = Path(directory).resolve()
    original = json.loads((directory / ARTIFACTS[0]).read_text(encoding="utf-8"))
    audit = deepcopy(original)
    old_root = original.get("snapshot_directory", "")
    require(isinstance(old_root, str), "Original comparison snapshot directory must be a string")
    path_type = PureWindowsPath if "\\" in old_root or ":" in old_root else PurePosixPath
    require(isinstance(old_root, str) and path_type(old_root).is_absolute(),
            "Original comparison snapshot directory must be absolute")
    require(path_type(original["observed"]["artifact_path"]) == path_type(old_root) / ARTIFACTS[1],
            "Original comparison observation path differs")
    audit["snapshot_directory"] = str(directory)
    audit["observed"]["artifact_path"] = str(directory / ARTIFACTS[1])
    raw = pd.read_parquet(directory / ARTIFACTS[1])
    actual = pd.Series(raw.actual.to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(raw.timestamp, utc=True)), name="actual")
    observation = verify_refreshed_observations(actual, audit, zone=zone, timezone=TIMEZONES[zone], delivery_day=stop)
    # The generic verifier deliberately retains legacy ENTSO-E report support.
    # This annual qualification protocol requires the single EPEX reference.
    require(observation.get("actual_reference") == "EPEX"
            and observation.get("source_kind") == EPEX_REPORTING_SOURCE_KIND
            and observation.get("policy") == EPEX_REPORTING_POLICY,
            f"{zone}: annual comparisons require official EPEX-only observations")
    loaded = _load_verified_snapshot(directory, zone=zone, timezone=TIMEZONES[zone])
    require(loaded is not None, f"{zone}: official Storm snapshot absent")
    return actual, loaded[0]


def _country(root, zone, record, first, stop):
    snapshot = _inside(root, record["snapshot"])
    require(set(record["source_sha256"]) == set(ARTIFACTS), "Comparison source inventory incomplete")
    for name, digest in record["source_sha256"].items():
        require(sha256(snapshot / name) == digest, f"{zone}: source comparison modified")
    # This preserves the producing checkout's configuration fingerprint. Its
    # exact bytes may differ after a portable copy (e.g. Git CRLF conversion);
    # the authoritative EPEX/Storm identities are checked in the snapshot.
    config_hash = record.get("target_config_sha256")
    require(isinstance(config_hash, str) and len(config_hash) == 64
            and all(c in "0123456789abcdef" for c in config_hash),
            f"{zone}: comparison configuration fingerprint missing")
    expected = evaluation_grid(first, stop)
    actual, storm = read_snapshot(snapshot, zone, stop)
    computed = pd.DataFrame({"actual": actual.reindex(expected), "storm": storm.reindex(expected)}, index=expected)
    require(np.isfinite(computed.actual.to_numpy(float)).all() and computed.storm.isna().sum() <= 1
            and not np.isinf(computed.storm.to_numpy(float)).any(), "Incomplete official comparison")
    return computed


def validate_comparisons(directory, first, stop):
    """Read labels and verify official provenance; call only during scoring."""
    root = Path(directory).resolve()
    receipt = json.loads((root / "comparisons_receipt.json").read_text(encoding="utf-8"))
    require(receipt.get("protocol") == PROTOCOL and receipt.get("first_day") == first
            and receipt.get("stop_day_exclusive") == stop and receipt.get("used_for_prediction") is False
            and set(receipt.get("countries", {})) == set(TIMEZONES), "Official comparison receipt identity differs")
    evaluation_grid(first, stop)
    for zone, record in receipt["countries"].items():
        computed = _country(root, zone, record, first, stop)
        progress = _inside(root, f"source_checkpoints/{zone}.json")
        require(progress.is_file() and sha256(progress) == record.get("source_checkpoint_sha256"),
                f"{zone}: source checkpoint missing or changed")
        checkpoint = json.loads(progress.read_text(encoding="utf-8"))
        require(checkpoint.get("protocol") == PROTOCOL and checkpoint.get("zone") == zone
                and checkpoint.get("first_day") == first and checkpoint.get("stop_day_exclusive") == stop
                and checkpoint.get("country") == {key: value for key, value in record.items()
                                                   if key not in ("source_checkpoint_sha256", "output_sha256")},
                "Comparison source checkpoint identity differs")
        output = root / f"{zone}.parquet"
        require(sha256(output) == record["output_sha256"], f"{zone}: comparison output modified")
        saved = pd.read_parquet(output)
        pd.testing.assert_frame_equal(saved, computed, check_exact=True, check_freq=False)
    return receipt


def export(directory, first, stop):
    root = Path(directory).resolve()
    expected = evaluation_grid(first, stop)
    require(expected[-1] < pd.Timestamp.now(tz="UTC"), "Cannot evaluate unobserved future delivery hours")
    if (root / "comparisons_receipt.json").exists():
        return validate_comparisons(root, first, stop)
    receipt = {"protocol": PROTOCOL, "first_day": first, "stop_day_exclusive": stop,
               "used_for_prediction": False, "countries": {}, "collector_code_sha256": sha256(Path(__file__))}
    for zone, filename in ZONE_CONFIGS.items():
        config_path = ROOT / filename
        checkpoint_path = root / f"source_checkpoints/{zone}.json"
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            require(checkpoint.get("protocol") == PROTOCOL and checkpoint.get("zone") == zone
                    and checkpoint.get("first_day") == first and checkpoint.get("stop_day_exclusive") == stop,
                    "Partial export dates or country differ; use a new directory")
            record = checkpoint["country"]
        else:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            _, snapshot, _ = refresh_nuclear_reporting_sources(config, zone, TIMEZONES[zone], stop, root / "snapshots" / zone)
            record = {"snapshot": snapshot.relative_to(root).as_posix(),
                      "source_sha256": {name: sha256(snapshot / name) for name in ARTIFACTS},
                      "target_config_sha256": sha256(config_path)}
            # Pin the verified source before publishing its derived values, so
            # an interrupted country can resume without fetching revised data.
            _country(root, zone, record, first, stop)
            _immutable(checkpoint_path, _encoded({"protocol": PROTOCOL, "zone": zone,
                "first_day": first, "stop_day_exclusive": stop, "country": record}))
        frame = _country(root, zone, record, first, stop)
        _parquet(root / f"{zone}.parquet", frame)
        receipt["countries"][zone] = {**record, "source_checkpoint_sha256": sha256(checkpoint_path),
            "output_sha256": sha256(root / f"{zone}.parquet")}
    _immutable(root / "comparisons_receipt.json", _encoded(receipt))
    validate_comparisons(root, first, stop)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-day", required=True)
    parser.add_argument("--stop-day-exclusive", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    with exclusive_process_lock(args.output.parent / (args.output.name + ".lock")):
        receipt = (validate_comparisons if args.verify_only else export)(args.output, args.first_day, args.stop_day_exclusive)
    print(json.dumps({"state": "COMPLETE", "countries": list(receipt["countries"]), "used_for_prediction": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
