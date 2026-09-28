"""Capture and assemble pre-cutoff French hydro inputs for annual NYX CPU."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests

from chronos2_hourly import nyx_fr_hydro_lagged_features as hydro
from chronos2_hourly.nyx_annual_live_preflight import ROOT, SOURCE_PROTOCOL, delivery_grid, sha256, validate_source_receipt
from chronos2_hourly.process_lock import exclusive_process_lock
from run_nyx_annual_exchange_source import _atomic_immutable, _parquet_bytes, _response, encoded, require

PROTOCOL = "nyx_annual_public_hydro_capture_v1"
DEFAULT_ARCHIVE = ROOT / "data/pit/nyx_annual_hydro_captures"
URL = "https://api.energy-charts.info/v2/public_power"
LEVELS = tuple("extra_hydro_" + name for name in hydro.FEATURES)
COLUMNS = LEVELS + tuple(name + "__available" for name in LEVELS)
NAMES = ("FR.json", "features.parquet", "capture.json")


def source_window(day):
    _, current, cutoff = delivery_grid(day)
    first = current[0] - pd.Timedelta(hours=48 + 167)
    stop = current[-1] - pd.Timedelta(hours=47)
    return current, first, stop, cutoff


def feature_frame(payload, current):
    hourly = hydro.public_power_hourly([payload])
    frame, _ = hydro.build_fr_hydro_features(hourly, current)
    frame = frame.rename(columns=lambda name: "extra_hydro_" + name)
    for name in LEVELS:
        frame[name + "__available"] = np.isfinite(frame[name].to_numpy(float)).astype(float)
    frame.index.name = "timestamp_utc"
    require(tuple(frame.columns) == COLUMNS, "Hydro feature schema differs")
    return frame


def verify_capture(directory, day):
    directory = Path(directory)
    current, first, stop, cutoff = source_window(day)
    receipt = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    require(receipt.get("protocol") == PROTOCOL and receipt.get("state") == "COMPLETE"
            and receipt.get("delivery_day") == day and receipt.get("cutoff_utc") == cutoff.isoformat()
            and receipt.get("source_window_utc") == [first.isoformat(), stop.isoformat()], "Hydro capture contract differs")
    hashes = receipt.get("artifact_sha256", {})
    require(set(hashes) == set(NAMES[:2]), "Hydro capture inventory differs")
    for name, digest in hashes.items():
        require(sha256(directory / name) == digest, f"Hydro artifact changed: {name}")
    retrieved = pd.Timestamp(receipt["retrieved_at_utc"])
    require(retrieved.tzinfo is not None and retrieved <= cutoff, "Hydro captured after cutoff")
    payload, _ = _response((directory / "FR.json").read_bytes(), "FR", first, stop, cutoff, retrieved)
    computed = feature_frame(payload, current)
    stored = pd.read_parquet(directory / "features.parquet")
    pd.testing.assert_frame_equal(stored, computed, check_exact=True, check_freq=False)
    return stored, receipt


def capture(day, archive=DEFAULT_ARCHIVE, *, session=None, now_utc=None):
    current, first, stop, cutoff = source_window(day)
    directory = Path(archive).resolve() / day
    if (directory / "capture.json").exists():
        verify_capture(directory, day)
        return directory / "capture.json"
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    require(now.tzinfo is not None and stop <= now < cutoff, "Hydro capture requires lagged observations before D-1 08:00")
    own_session = session is None
    session = requests.Session() if own_session else session
    try:
        response = session.get(URL, params={"country": "fr", "start": first.isoformat(),
            "end": (stop - pd.Timedelta(minutes=1)).isoformat()}, timeout=(10, 45), allow_redirects=False)
        require(response.status_code == 200, f"Hydro HTTP {response.status_code}")
        retrieved = pd.Timestamp.now(tz="UTC") if now_utc is None else now
        obj, provider = _response(response.content, "FR", first, stop, cutoff, retrieved)
        frame = feature_frame(obj, current)
        bodies = {"FR.json": response.content, "features.parquet": _parquet_bytes(frame)}
        receipt = {"protocol": PROTOCOL, "delivery_day": day, "state": "COMPLETE",
            "cutoff_utc": cutoff.isoformat(), "retrieved_at_utc": retrieved.isoformat(),
            "source_window_utc": [first.isoformat(), stop.isoformat()], "provider_response": provider,
            "artifact_sha256": {n: hashlib.sha256(raw).hexdigest() for n, raw in bodies.items()},
            "capture_code_sha256": sha256(Path(__file__)), "feature_code_sha256": sha256(Path(hydro.__file__)),
            "origin_snapshot_capture_verified": True, "provider_first_publication_timestamp_verified": False}
        for name, raw in bodies.items():
            _atomic_immutable(directory / name, raw)
        path = directory / "capture.json"
        _atomic_immutable(path, encoded(receipt))
        verify_capture(directory, day)
        return path
    finally:
        if own_session:
            session.close()


def assemble(day, archive=DEFAULT_ARCHIVE, bundle=None):
    full, _, cutoff = delivery_grid(day)
    last = date.fromisoformat(day)
    days = [(last - timedelta(days=n)).isoformat() for n in range(365, -1, -1)]
    archive = Path(archive).resolve()
    frames, receipts = [], {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for d in days:
            frame, receipt = verify_capture(archive / d, d)
            frames.append(frame)
            receipts[d] = sha256(archive / d / "capture.json")
            for name in NAMES:
                info = zipfile.ZipInfo(f"{d}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                output.writestr(info, (archive / d / name).read_bytes())
    combined = pd.concat(frames)
    require(combined.index.equals(full.rename("timestamp_utc")), "Hydro annual capture timeline incomplete")
    bundle = Path(bundle or ROOT / "runs/live/nyx_annual_cpu" / day).resolve()
    artifacts = {"source_artifacts/public_hydro/features.parquet": _parquet_bytes(combined),
                 "source_artifacts/public_hydro/pit_captures.zip": buffer.getvalue()}
    hashes = {n: hashlib.sha256(raw).hexdigest() for n, raw in artifacts.items()}
    for name, raw in artifacts.items():
        _atomic_immutable(bundle / name, raw)
    receipt = {"protocol": SOURCE_PROTOCOL, "source_group": "public_hydro", "delivery_day": day,
        "state": "COMPLETE", "asof_cutoff_verified": True, "training_window_complete": True,
        "asof_state_utc": cutoff.isoformat(), "origin_snapshot_capture_verified": True,
        "provider_first_publication_timestamp_verified": False,
        "availability_basis": "Immutable daily responses actually observed before each D-1 08:00 cutoff",
        "capture_receipt_sha256": receipts, "captures": len(days), "artifact_sha256": hashes,
        "collector_code_sha256": sha256(Path(__file__)), "feature_code_sha256": sha256(Path(hydro.__file__))}
    validate_source_receipt(receipt, group="public_hydro", day=day, bundle=bundle, cutoff=cutoff)
    path = bundle / "source_receipts/public_hydro.json"
    _atomic_immutable(path, encoded(receipt))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("capture", "assemble", "verify"), required=True)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(argv)
    with exclusive_process_lock(args.archive / "hydro_source.lock"):
        if args.action == "capture":
            path = capture(args.delivery_day, args.archive)
        elif args.action == "assemble":
            path = assemble(args.delivery_day, args.archive, args.bundle)
        else:
            verify_capture(args.archive / args.delivery_day, args.delivery_day)
            path = args.archive / args.delivery_day / "capture.json"
    print(json.dumps({"state": "COMPLETE", "source_group": "public_hydro", "receipt": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
