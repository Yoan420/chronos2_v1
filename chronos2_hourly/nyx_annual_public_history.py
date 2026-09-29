"""Bootstrap revised public observations for a current annual CPU fit.

The training snapshot has its real retrieval time. It never claims to be an
old forecast vintage. Delivery features always come from the separately
verified daily capture made before the delivery cutoff.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from chronos2_hourly import nyx_fr_hydro_lagged_features as hydro
from chronos2_hourly import nyx_lagged_exchange_features as exchange
from chronos2_hourly.nyx_annual_live_preflight import ROOT, SOURCE_PROTOCOL, delivery_grid, sha256
from chronos2_hourly.process_lock import exclusive_process_lock

POLICY = "current_fit_snapshot_v1"
PROTOCOL = "nyx_annual_public_history_v1"
PARTITION_PROTOCOL = "nyx_annual_public_history_partition_v1"
DEFAULT_HISTORY_ARCHIVE = ROOT / "data/pit/nyx_annual_public_history"
GROUPS = ("public_hydro", "lagged_exchange")
MIN_REQUEST_INTERVAL_SECONDS = 2.0
_LAST_NETWORK_REQUEST = 0.0
URLS = {"FR": "https://api.energy-charts.info/v2/public_power",
        "DE": "https://api.energy-charts.info/v2/cbpf"}


def _helpers():
    # Lazy imports avoid a CLI/module import cycle.
    import run_nyx_annual_exchange_source as capture
    return capture


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _utc(value, name):
    return _helpers()._utc(value, name)


def _encoded(value):
    return _helpers().encoded(value)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name, raw in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            output.writestr(info, raw)
    return buffer.getvalue()


def _unzip(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "Duplicate history ZIP members")
        _require(all(not name.startswith(("/", "\\")) and ".." not in Path(name).parts
                     and "\\" not in name for name in names), "Unsafe history ZIP member")
        return {name: archive.read(name) for name in names}


def _windows(first, stop):
    """Non-overlapping UTC month partitions, including the two edge fragments."""
    while first < stop:
        following = pd.Timestamp(year=first.year, month=first.month, day=1, tz="UTC") + pd.offsets.MonthBegin(1)
        end = min(following, stop)
        yield first, end
        first = end


def _training_window(group, training):
    warmup = 167 if group == "public_hydro" else 0
    return training[0] - pd.Timedelta(hours=48 + warmup), training[-1] - pd.Timedelta(hours=47)


def _specs(group, training):
    first, stop = _training_window(group, training)
    zones = ("FR",) if group == "public_hydro" else ("DE", "FR")
    return [(zone, begin, end) for zone in zones for begin, end in _windows(first, stop)]


def _key(zone, first, stop):
    return zone + "/" + first.strftime("%Y%m%dT%H%M%SZ") + "_" + stop.strftime("%Y%m%dT%H%M%SZ")


def _verify_partition(entries, zone, first, stop):
    _require(set(entries) == {"response.json", "retrieval.json"}, "History partition inventory differs")
    receipt = json.loads(entries["retrieval.json"])
    _require(receipt.get("protocol") == PARTITION_PROTOCOL and receipt.get("state") == "COMPLETE"
             and receipt.get("url") == URLS[zone] and receipt.get("country") == zone
             and receipt.get("source_window_utc") == [first.isoformat(), stop.isoformat()]
             and receipt.get("response_sha256") == _digest(entries["response.json"]),
             "History partition identity or bytes differ")
    retrieved = _utc(receipt.get("retrieved_at_utc"), "Historical response retrieval")
    _require(stop <= retrieved, "Training observations extend beyond actual retrieval")
    # The provider generation clock is recorded, not interpreted as a
    # publication time or substituted for the actual local retrieval clock.
    payload, metadata = _helpers()._response(entries["response.json"], zone, first, stop, retrieved, retrieved,
                                             require_generated_before_retrieval=False)
    _require(receipt.get("provider_response") == metadata, "History response provenance differs")
    return payload, receipt


def _partition(zone, first, stop, archive, session, now_utc):
    """One atomic cache object binds exact response bytes and the real clock."""
    path = archive / (_key(zone, first, stop) + ".zip")
    if path.exists():
        entries = _unzip(path.read_bytes())
        _verify_partition(entries, zone, first, stop)
        return entries
    global _LAST_NETWORK_REQUEST
    response = None
    for attempt in range(3):
        delay = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _LAST_NETWORK_REQUEST)
        if delay > 0:
            time.sleep(delay)
        _LAST_NETWORK_REQUEST = time.monotonic()
        try:
            response = session.get(URLS[zone], params=_helpers()._params(zone, first, stop),
                                   timeout=(10, 45), allow_redirects=False)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
            continue
        if response.status_code == 200:
            break
        if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
            raise ValueError(f"{zone}: history download HTTP {response.status_code} for {first.date()}")
        # The public API may throttle bursts. Respect bounded Retry-After
        # seconds and give rate limits time to clear before the next attempt.
        try:
            retry_after = float(getattr(response, "headers", {}).get("Retry-After", 0))
        except (TypeError, ValueError):
            retry_after = 0
        time.sleep(min(30, max(15 * (attempt + 1) if response.status_code == 429 else attempt + 1,
                               retry_after)))
    _require(response is not None and response.status_code == 200, "No historical response")
    retrieved = pd.Timestamp.now(tz="UTC") if now_utc is None else _utc(now_utc, "History clock")
    _require(stop <= retrieved, "Training observations extend beyond actual retrieval")
    _, metadata = _helpers()._response(response.content, zone, first, stop, retrieved, retrieved,
                                      require_generated_before_retrieval=False)
    entries = {"response.json": response.content, "retrieval.json": _encoded({
        "protocol": PARTITION_PROTOCOL, "state": "COMPLETE", "country": zone, "url": URLS[zone],
        "source_window_utc": [first.isoformat(), stop.isoformat()], "retrieved_at_utc": retrieved.isoformat(),
        "response_sha256": _digest(response.content), "provider_response": metadata,
        "historical_publication_vintage_verified": False})}
    _verify_partition(entries, zone, first, stop)
    _helpers()._atomic_immutable(path, _zip(entries))
    return entries


def _build_training(group, payloads, training):
    if group == "public_hydro":
        hourly = hydro.public_power_hourly(payloads["FR"])
        frame, _ = hydro.build_fr_hydro_features(hourly, training)
        frame = frame.rename(columns=lambda name: "extra_hydro_" + name)
        for name in tuple(frame):
            frame[name + "__available"] = np.isfinite(frame[name].to_numpy(float)).astype(float)
    else:
        hourly = exchange.exchange_hourly(payloads["DE"], payloads["FR"])
        country, _ = exchange.build_features(hourly, training)
        frame = country["FR"]
    frame.index.name = "timestamp_utc"
    # Explicit provider nulls remain NaN; an entirely absent series is not a
    # usable source and must not silently produce a year of unavailable flags.
    levels = [name for name in frame if not name.endswith("__available")]
    _require(all(np.isfinite(frame[name].to_numpy(float)).any() for name in levels),
             "Historical source has no usable observations for a required feature")
    return frame


def _capture_module(group):
    if group == "public_hydro":
        import run_nyx_annual_hydro_source as module
        return module, module.NAMES, module.verify_capture
    module = _helpers()
    return module, module.CAPTURE_NAMES, module._verify_capture


def _verify_delivery(group, day, entries):
    _, names, verify = _capture_module(group)
    _require(set(entries) == set(names), "Delivery capture inventory differs")
    # Reuse the strict original raw-to-feature verifier; ZIP paths never get
    # extracted from untrusted names.
    with tempfile.TemporaryDirectory(prefix="nyx-public-delivery-") as temporary:
        directory = Path(temporary)
        for name in names:
            (directory / name).write_bytes(entries[name])
        return verify(directory, day)


def _reconstruct(group, day, manifest, entries):
    full, current, cutoff = delivery_grid(day)
    training = full[~full.isin(current)]
    _require(manifest.get("protocol") == PROTOCOL and manifest.get("history_policy") == POLICY
             and manifest.get("source_group") == group and manifest.get("delivery_day") == day
             and manifest.get("training_target_window_utc") == [training[0].isoformat(), training[-1].isoformat()]
             and manifest.get("cutoff_utc") == cutoff.isoformat(), "History manifest identity differs")
    payloads, times, expected_entries, expected_partitions = {"FR": [], "DE": []}, [], set(), []
    for zone, first, stop in _specs(group, training):
        prefix = "training/" + _key(zone, first, stop) + "/"
        names = [prefix + "response.json", prefix + "retrieval.json"]
        _require(all(name in entries for name in names), "Missing historical partition")
        part = {name[len(prefix):]: entries[name] for name in names}
        payload, receipt = _verify_partition(part, zone, first, stop)
        payloads[zone].append(payload)
        times.append(_utc(receipt["retrieved_at_utc"], "Training retrieval"))
        expected_entries.update(names)
        expected_partitions.append({"key": _key(zone, first, stop),
                                    "retrieval_sha256": _digest(part["retrieval.json"])})
    _require(manifest.get("partitions") == expected_partitions, "Historical partition manifest differs")
    _, names, _ = _capture_module(group)
    delivery_entries = {name: entries["delivery/" + name] for name in names if "delivery/" + name in entries}
    expected_entries.update("delivery/" + name for name in names)
    _require(set(entries) == expected_entries, "History ZIP inventory differs")
    delivery, capture = _verify_delivery(group, day, delivery_entries)
    _require(manifest.get("delivery_capture_sha256") == _digest(delivery_entries["capture.json"]),
             "Delivery capture manifest differs")
    trained = _build_training(group, payloads, training)
    combined = pd.concat([trained, delivery])
    _require(combined.index.equals(full.rename("timestamp_utc")), "Public annual feature timeline incomplete")
    maximum = max(times)
    return combined, {"asof_state_utc": capture["retrieved_at_utc"],
        "training_snapshot_max_retrieved_at_utc": maximum.isoformat(),
        "asof_cutoff_verified": bool(maximum <= cutoff), "training_window_complete": True,
        "delivery_snapshot_pre_cutoff_verified": True, "origin_snapshot_capture_verified": False,
        "training_hours": len(training), "target_hours": len(current),
        "missing_training_hours_by_feature": {name: int(trained[name].isna().sum())
             for name in trained if not name.endswith("__available")}}


def verify_bundle_history(bundle, day, receipt):
    """Independently reconstruct every feature and time assertion from raw bytes."""
    bundle = Path(bundle).resolve()
    group = receipt.get("source_group")
    _require(group in GROUPS and receipt.get("protocol") == SOURCE_PROTOCOL
             and receipt.get("state") == "COMPLETE" and receipt.get("delivery_day") == day
             and receipt.get("history_policy") == POLICY, "Public history source receipt differs")
    prefix = "source_artifacts/" + group + "/"
    names = {prefix + name for name in ("features.parquet", "history_manifest.json", "history_captures.zip")}
    hashes = receipt.get("artifact_sha256", {})
    _require(set(hashes) == names, "Public history source artifact inventory differs")
    for name in sorted(names):
        _require((bundle / name).is_file() and sha256(bundle / name) == hashes[name],
                 f"Public history artifact bytes changed: {name}")
    manifest = json.loads((bundle / (prefix + "history_manifest.json")).read_text(encoding="utf-8"))
    entries = _unzip((bundle / (prefix + "history_captures.zip")).read_bytes())
    computed, facts = _reconstruct(group, day, manifest, entries)
    stored = pd.read_parquet(bundle / (prefix + "features.parquet"))
    pd.testing.assert_frame_equal(stored, computed, check_exact=True, check_freq=False)
    for name, value in facts.items():
        _require(receipt.get(name) == value, f"Public history source assertion differs: {name}")
    _require(receipt.get("provider_first_publication_timestamp_verified") is False
             and receipt.get("provider_revision_vintage_verified") is False,
             "Revised historical observations cannot certify old publication vintages")
    return facts


def publish_history(group, day, bundle, *, archive=None, history_archive=None, session=None, now_utc=None):
    """Build annual training history while retaining the strict delivery capture.

    A post-cutoff training snapshot can be prepared for diagnosis/re-evaluation;
    its false asof_cutoff_verified assertion must prevent forecast qualification.
    """
    _require(group in GROUPS, "Unknown public history source group")
    bundle = Path(bundle).resolve()
    destination = bundle / "source_receipts" / (group + ".json")
    if destination.exists():
        receipt = json.loads(destination.read_text(encoding="utf-8"))
        verify_bundle_history(bundle, day, receipt)
        return destination
    module, names, verify = _capture_module(group)
    archive = Path(archive or module.DEFAULT_ARCHIVE).resolve()
    delivery_dir = archive / day
    # Check the only irreplaceable input before doing any network work.
    verify(delivery_dir, day)
    entries = {"delivery/" + name: (delivery_dir / name).read_bytes() for name in names}
    full, current, cutoff = delivery_grid(day)
    training = full[~full.isin(current)]
    manifest = {"protocol": PROTOCOL, "history_policy": POLICY, "source_group": group,
        "delivery_day": day, "cutoff_utc": cutoff.isoformat(),
        "training_target_window_utc": [training[0].isoformat(), training[-1].isoformat()],
        "delivery_capture_sha256": _digest(entries["delivery/capture.json"]), "partitions": []}
    history_archive = Path(history_archive or DEFAULT_HISTORY_ARCHIVE).resolve()
    own_session = session is None
    session = requests.Session() if own_session else session
    try:
        with exclusive_process_lock(history_archive / "public_history.lock"):
            for zone, first, stop in _specs(group, training):
                part = _partition(zone, first, stop, history_archive, session, now_utc)
                key = _key(zone, first, stop)
                entries.update({"training/" + key + "/" + name: raw for name, raw in part.items()})
                manifest["partitions"].append({"key": key, "retrieval_sha256": _digest(part["retrieval.json"])})
    finally:
        if own_session:
            session.close()
    combined, facts = _reconstruct(group, day, manifest, entries)
    prefix = "source_artifacts/" + group + "/"
    bodies = {prefix + "features.parquet": _helpers()._parquet_bytes(combined),
              prefix + "history_captures.zip": _zip(entries),
              prefix + "history_manifest.json": _encoded(manifest)}
    receipt = {"protocol": SOURCE_PROTOCOL, "source_group": group, "delivery_day": day,
        "state": "COMPLETE", "history_policy": POLICY, **facts,
        "provider_first_publication_timestamp_verified": False, "provider_revision_vintage_verified": False,
        "availability_basis": "Training uses revised observations at their recorded actual retrieval; delivery uses the pre-cutoff daily capture",
        "artifact_sha256": {name: _digest(raw) for name, raw in bodies.items()},
        "collector_code_sha256": sha256(Path(__file__)),
        "feature_code_sha256": sha256(Path(hydro.__file__ if group == "public_hydro" else exchange.__file__))}
    for name, raw in bodies.items():
        _helpers()._atomic_immutable(bundle / name, raw)
    verify_bundle_history(bundle, day, receipt)
    _helpers()._atomic_immutable(destination, _encoded(receipt))
    return destination
