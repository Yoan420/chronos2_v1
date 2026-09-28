"""Collect and bind the live Saturn TTF/EUA inputs for annual CWE features.

This wraps the existing audited Kalman fuel materializer. It extends one local,
ignored as-of cache, verifies its full causal hourly contract, and makes an
immutable dated copy plus a source receipt for the annual CPU input preflight.
It produces only the eight fuel values and six timestamps, never a price
forecast or a completed 449/123/503-column feature matrix.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from chronos2_hourly.nyx_annual_live_preflight import (
    ROOT, SOURCE_PROTOCOL, delivery_grid, sha256, validate_source_receipt,
)
from chronos2_hourly.process_lock import exclusive_process_lock
import materialize_saturn_kalman_fuel as fuel


DEFAULT_HISTORY_START = "2025-09-24"
DEFAULT_CACHE = ROOT / "data/pit/nyx_annual_cpu_live_fuel"
SOURCE_NAMES = (fuel.OUTPUT_NAME, fuel.OUTPUT_NAME + fuel.AUDIT_SUFFIX)
ASSUMPTIONS = {"ccgt_efficiency": 0.58,
               "ccgt_emission_tco2_mwh": 0.36,
               "ccgt_vom_eur_mwh": 3.0}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def parse_day(value: str) -> date:
    day = date.fromisoformat(value)
    require(day.isoformat() == value, "Date attendue au format YYYY-MM-DD")
    return day


def verify_fuel_artifact(cache: Path, *, delivery_day: str,
                         history_start_day: str,
                         now_utc: pd.Timestamp | None = None) -> dict:
    """Recheck the existing materializer artifact without a Saturn request."""
    day = parse_day(delivery_day)
    history_start = parse_day(history_start_day)
    full, current, cutoff = delivery_grid(delivery_day)
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    require(now.tzinfo is not None and now.tz_convert("UTC") >= cutoff,
            "La coupure civile D-1 08:00 n'est pas encore atteinte")
    require(history_start <= day - timedelta(days=365),
            "L'historique Saturn ne couvre pas les 365 jours d'entraînement")
    output = cache / fuel.OUTPUT_NAME
    audit_path = cache / SOURCE_NAMES[1]
    require(output.is_file() and audit_path.is_file(),
            "Cache fuel Saturn ou audit absent ; lancer la collecte")
    frame, audit, artifact_end = fuel._load_existing(
        output, audit_path, requested_start=pd.Timestamp(history_start),
        **ASSUMPTIONS)
    require(artifact_end.date() == day,
            "Cache fuel daté d'un autre jour ; aucune lecture de jours futurs")
    index = pd.DatetimeIndex(pd.to_datetime(frame["value_time_utc"], utc=True))
    indexed = frame.set_index(index)
    selected = indexed.loc[full]
    require(selected.index.equals(full),
            "Fuel Saturn : grille complète entraînement et livraison manquante")
    snapshot = pd.DatetimeIndex(pd.to_datetime(selected.loc[current, "snapshot_time_utc"], utc=True))
    revision = pd.DatetimeIndex(pd.to_datetime(selected.loc[current, "revision_time_utc"], utc=True))
    require((snapshot == cutoff).all() and (revision == cutoff).all(),
            "Fuel Saturn : D-1 08:00 snapshot/revision différent")
    latest_sources = []
    for name in fuel.SOURCE_TIME_COLUMNS:
        stamps = pd.DatetimeIndex(pd.to_datetime(selected.loc[current, name], utc=True))
        require(not stamps.hasnans and (stamps <= cutoff).all(),
                f"Fuel Saturn : source {name} absente ou postérieure à la coupure")
        latest_sources.append(stamps.max())
    require(np.isfinite(selected[list(fuel.FEATURE_COLUMNS)].to_numpy(dtype=float)).all(),
            "Fuel Saturn : prix combustible ou coût CCGT absent")
    return {"output": output, "audit_path": audit_path,
            "source_sha256": sha256(output), "audit_sha256": sha256(audit_path),
            "latest_source_value_time_utc": max(latest_sources).isoformat(),
            "cutoff_utc": cutoff.isoformat(), "rows_in_training_and_delivery": len(full),
            "audit": audit}


def _copy_immutable(source: Path, target: Path, expected: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    require(sha256(source) == expected,
            f"Source modifiée avant la copie : {source.name}")
    if target.exists():
        require(sha256(target) == expected,
                f"Lot source existant différent : {target.name}")
        return
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        require(sha256(temporary) == expected and sha256(source) == expected,
                f"Source modifiée pendant la copie : {source.name}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def publish_fuel_receipt(bundle: Path, *, delivery_day: str,
                         evidence: dict) -> Path:
    """Freeze the verified source and its provenance under one dated bundle."""
    bundle = bundle.resolve()
    _, _, cutoff = delivery_grid(delivery_day)
    require(evidence["cutoff_utc"] == cutoff.isoformat()
            and evidence["audit"]["artifact_type"] == "kalman_market_fuel_features_pit_hourly",
            "Evidence fuel datée ou typée différemment")
    locations = {
        "source_artifacts/fuel/" + fuel.OUTPUT_NAME:
            (Path(evidence["output"]), evidence["source_sha256"]),
        "source_artifacts/fuel/" + fuel.OUTPUT_NAME + fuel.AUDIT_SUFFIX:
            (Path(evidence["audit_path"]), evidence["audit_sha256"]),
    }
    for relative, (source, digest) in locations.items():
        _copy_immutable(source, bundle / relative, digest)
    receipt = {
        "protocol": SOURCE_PROTOCOL, "source_group": "fuel",
        "delivery_day": delivery_day, "state": "COMPLETE",
        "asof_cutoff_verified": True, "training_window_complete": True,
        # This is the Saturn state queried AS OF the cutoff. It is not the
        # provider's publication timestamp (which Saturn does not expose).
        "asof_state_utc": evidence["cutoff_utc"],
        "availability_basis": "Saturn revision_date query as-of D-1 08:00",
        "latest_source_value_time_utc": evidence["latest_source_value_time_utc"],
        "cutoff_utc": evidence["cutoff_utc"],
        "rows_in_training_and_delivery": evidence["rows_in_training_and_delivery"],
        "artifact_sha256": {relative: digest for relative, (_, digest) in locations.items()},
        "materializer_code_sha256": sha256(Path(fuel.__file__)),
        "materializer_artifact_type": evidence["audit"]["artifact_type"],
        "provider_revision_timestamp_available": False,
        "provider_publication_timestamp_verified": False,
        "publication_vintage_limit": (
            "Saturn queried with revision_date at D-1 08:00; provider insertion "
            "timestamps are not exposed by this source."),
        "model_inputs_complete": False,
    }
    validate_source_receipt(receipt, group="fuel", day=delivery_day,
                            bundle=bundle, cutoff=cutoff)
    raw = (json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    destination = bundle / "source_receipts/fuel.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(json.loads(destination.read_text(encoding="utf-8")) == receipt,
                "Reçu fuel existant différent")
        return destination
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--history-start-day", default=DEFAULT_HISTORY_START,
                        help="Début fixe de l'artefact Saturn réutilisé et prolongé")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--verify-only", action="store_true",
                        help="Vérifier un cache existant sans requête Saturn")
    args = parser.parse_args(argv)
    day = parse_day(args.delivery_day)
    start = parse_day(args.history_start_day)
    cache = args.cache_dir.resolve()
    bundle = args.bundle or ROOT / "runs/live/nyx_annual_cpu" / day.isoformat()
    # The check precedes Saturn requests as well as receipt publication.
    _, _, cutoff = delivery_grid(args.delivery_day)
    require(pd.Timestamp.now(tz="UTC") >= cutoff,
            "La coupure civile D-1 08:00 n'est pas encore atteinte")
    require(start <= day - timedelta(days=365),
            "L'historique demandé ne couvre pas 365 jours")
    with exclusive_process_lock(cache / "fuel_live.lock"):
        if not args.verify_only:
            result = fuel.main([
                "--start-day", start.isoformat(), "--end-day", day.isoformat(),
                "--output-dir", str(cache), "--skip-residual-load",
            ])
            require(result == 0, "La collecte Saturn combustible a échoué")
        evidence = verify_fuel_artifact(cache, delivery_day=day.isoformat(),
                                        history_start_day=start.isoformat())
        receipt_path = publish_fuel_receipt(bundle, delivery_day=day.isoformat(),
                                            evidence=evidence)
    print(json.dumps({"state": "COMPLETE", "source": "fuel",
                      "delivery_day": day.isoformat(), "receipt": str(receipt_path),
                      "model_inputs_complete": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
