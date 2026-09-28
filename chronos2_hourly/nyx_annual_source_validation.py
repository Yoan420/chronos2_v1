"""Recheck portable annual source packets against their raw daily evidence.

This verifies recorded as-of states and recomputes derived features. It does
not turn supplier query timestamps into certified first-publication times.
No network access, activation flag, source receipt or model result is written.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

import numpy as np
import pandas as pd

from . import nyx_annual_live_preflight as gate
from .nyx_annual_cpu_bundle_builder import source_graph
from .nyx_local_extra_features import cutoff_times, normalize_fuel, FUEL_TIME_COLUMNS

PROTOCOL = "nyx_annual_raw_source_validation_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _receipt(bundle, group):
    return json.loads((bundle / f"source_receipts/{group}.json").read_text(encoding="utf-8"))


def _bound(bundle, receipt, relative):
    require(relative in receipt["artifact_sha256"], f"Raw source evidence is not bound: {relative}")
    path = (bundle / relative).resolve()
    require(path.is_relative_to(bundle.resolve()) and path.is_file()
            and gate.sha256(path) == receipt["artifact_sha256"][relative],
            f"Raw source evidence changed: {relative}")
    return path


def _exact(left, right, label):
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True, check_freq=False)
    except AssertionError as error:
        raise ValueError(f"{label}: stored values differ from raw reconstruction") from error


def _saturn(bundle, day, receipt):
    from . import nyx_annual_saturn_source as source
    first = date.fromisoformat(receipt["first_delivery_day"])
    last = date.fromisoformat(day)
    days = [stamp.date().isoformat() for stamp in pd.date_range(first, last, freq="D")]
    require(set(receipt.get("daily_vintages", {})) == set(days)
            and first <= last - timedelta(days=365)
            and receipt.get("series") == source.specs(), "Saturn daily vintage range or series differs")
    frames, latest_prices = [], None
    for origin in days:
        prefix = f"source_artifacts/saturn/days/{origin}"
        for name in ("receipt.json", "covariates.parquet", "prices.parquet"):
            _bound(bundle, receipt, f"{prefix}/{name}")
        covariates, prices, daily = source.verify_day(bundle / prefix, origin)
        declared = receipt["daily_vintages"][origin]
        require(declared.get("forecast_origin_utc") == daily["forecast_origin_utc"]
                and declared.get("covariates_sha256") == daily["artifact_sha256"]["covariates.parquet"]
                and declared.get("prices_sha256") == daily["artifact_sha256"]["prices.parquet"],
                f"Saturn daily vintage declaration differs: {origin}")
        frames.append(covariates)
        latest_prices = prices
    combined = pd.concat(frames)
    stored = pd.read_parquet(_bound(bundle, receipt, "source_artifacts/saturn/covariates.parquet"))
    _exact(stored, combined, "Saturn forecast profiles")
    origins = pd.DataFrame({"forecast_origin_utc": [source.cutoff(str(d))
        for d in combined.index.tz_convert("Europe/Paris").date]}, index=combined.index)
    stored = pd.read_parquet(_bound(bundle, receipt, "source_artifacts/saturn/origins.parquet"))
    _exact(stored, origins, "Saturn per-day origins")
    return {"daily_states_verified": len(days), "provider_first_publication_certified": False}, latest_prices


def _auction(bundle, day, receipt, latest_prices):
    from run_nyx_annual_auction_prices_source import SERIES
    _, _, cutoff = gate.delivery_grid(day)
    require(receipt.get("series") == SERIES and receipt.get("cutoff_utc") == cutoff.isoformat(),
            "Auction source series or cutoff differs")
    count = None
    for zone in gate.ZONES:
        frame = pd.read_parquet(_bound(bundle, receipt, f"source_artifacts/auction_prices/{zone}.parquet"))
        expected = pd.date_range(pd.Timestamp(receipt["first_hour_utc"]),
                                 pd.Timestamp(receipt["last_hour_utc"]), freq="h")
        require(frame.index.equals(expected) and list(frame) == ["price_eur_mwh"]
                and np.isfinite(frame.to_numpy(float)).all()
                and frame.index[-1] < pd.Timestamp(day, tz="Europe/Paris").tz_convert("UTC"),
                f"{zone}: incomplete auction history or delivery labels present")
        common = latest_prices.index.intersection(frame.index)
        full, current, _ = gate.delivery_grid(day)
        require(full[~full.isin(current)].isin(common).all(), "Auction/Saturn training histories do not overlap")
        require(np.array_equal(frame.loc[common, "price_eur_mwh"].to_numpy(float),
                               latest_prices.loc[common, zone].to_numpy(float)),
                f"{zone}: auction prices disagree with current Saturn revision-date state")
        count = len(frame)
    return {"hours_per_country_verified": count, "provider_first_publication_certified": False}


def _jao(bundle, day, receipt, full):
    from . import nyx_annual_jao_source as source
    prefix = source.SOURCE_SUBDIR
    ledger = json.loads(_bound(bundle, receipt, f"{prefix}/{source.LEDGER_NAME}").read_text(encoding="utf-8"))
    root = bundle / prefix / "captures"
    days = sorted(set(full.tz_convert("Europe/Paris").date))
    verified = []
    for origin in days:
        for path in source._paths(root, origin):
            _bound(bundle, receipt, path.relative_to(bundle).as_posix())
        verified.append(source.verify_daily_capture(root, origin))
    require(ledger.get("protocol") == source.LEDGER_PROTOCOL
            and ledger.get("delivery_day") == day and ledger.get("missing_days") == []
            and ledger.get("captures") == verified
            and ledger.get("captured_days") == len(days), "JAO raw captures differ from source ledger")
    computed, audit = source.build_strict_history_features(root, full)
    stored = pd.read_parquet(_bound(bundle, receipt, f"{prefix}/{source.FEATURE_NAME}"))
    _exact(stored, computed, "JAO descriptors")
    require(audit.get("historical_fallback") is False and audit.get("imputation") is False,
            "JAO reconstruction changed its causal policy")
    return {"daily_raw_captures_recomputed": len(days),
            "actual_capture_before_own_cutoff_verified": True,
            "provider_first_publication_certified": False}


def _extract_capture_zip(path, target, days, names):
    """Read only the expected flat day/member layout, never arbitrary ZIP paths."""
    expected = {f"{day}/{name}" for day in days for name in names}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == len(expected) and {item.filename for item in entries} == expected,
                "Capture ZIP inventory differs (missing, duplicate or unexpected member)")
        for item in entries:
            relative = PurePosixPath(item.filename)
            require(not relative.is_absolute() and ".." not in relative.parts
                    and "\\" not in item.filename and not item.is_dir(), "Unsafe capture ZIP member")
            destination = target.joinpath(*relative.parts)
            require(destination.resolve().is_relative_to(target.resolve()), "Capture ZIP escapes destination")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as reader, destination.open("xb") as writer:
                while block := reader.read(1024 * 1024):
                    writer.write(block)


def _captured_features(bundle, day, receipt, full, *, group):
    if group == "public_hydro":
        import run_nyx_annual_hydro_source as source
        verify, names = source.verify_capture, source.NAMES
    else:
        import run_nyx_annual_exchange_source as source
        verify, names = source._verify_capture, source.CAPTURE_NAMES
    days = [str(value) for value in sorted(set(full.tz_convert("Europe/Paris").date))]
    require(set(receipt.get("capture_receipt_sha256", {})) == set(days),
            f"{group}: incomplete daily capture checksums")
    prefix = f"source_artifacts/{group}"
    archive = _bound(bundle, receipt, f"{prefix}/pit_captures.zip")
    frames = []
    with tempfile.TemporaryDirectory(prefix="nyx_source_validation_") as name:
        target = Path(name)
        _extract_capture_zip(archive, target, days, names)
        for origin in days:
            frame, _ = verify(target / origin, origin)
            require(gate.sha256(target / origin / "capture.json") == receipt["capture_receipt_sha256"][origin],
                    f"{group}/{origin}: daily capture receipt differs")
            frames.append(frame)
    combined = pd.concat(frames)
    require(combined.index.equals(full), f"{group}: capture timeline differs")
    stored = pd.read_parquet(_bound(bundle, receipt, f"{prefix}/features.parquet"))
    _exact(stored, combined, f"{group} captured features")
    return {"daily_raw_captures_recomputed": len(days),
            "actual_capture_before_own_cutoff_verified": True,
            "provider_first_publication_certified": False}


def _thermal(bundle, day, receipt):
    import run_nyx_annual_thermal_source as source
    days, full, _ = source.grids(day)
    expected_series = {name: spec["series"] for name, spec in source.plan()["specs"].items()}
    require(receipt.get("series") == expected_series, "Thermal Saturn series differs")
    checks = receipt.get("independent_daily_state_checks", {})
    require(set(checks) == set(source.SOURCES), "Thermal daily independent states absent")
    sources, missing = {}, {}
    expected_origins = cutoff_times(full, "Europe/Paris")
    for name in source.SOURCES:
        frame = pd.read_parquet(_bound(bundle, receipt, f"source_artifacts/thermal_capacity/{name}.parquet"))
        require(frame.index.equals(full) and list(frame) == ["pmax_gw", "asof_query_utc"]
                and pd.DatetimeIndex(frame.asof_query_utc).equals(pd.DatetimeIndex(expected_origins)),
                f"{name}: thermal origin differs from its own daily cutoff")
        require(len(checks[name]) == len(days), f"{name}: thermal daily state count differs")
        absent = []
        daily_values = frame.pmax_gw.groupby(frame.index.tz_convert("Europe/Paris").date)
        unique_by_day = {d: values.unique() for d, values in daily_values}
        for stamp, check in zip(days, checks[name], strict=True):
            d = stamp.date().isoformat()
            unique = unique_by_day[stamp.date()]
            require(len(unique) == 1, f"{name}/{d}: capacity is not a daily value")
            value = unique[0]
            matches = ((check.get("value") is None and pd.isna(value)) or
                       (check.get("value") is not None and np.isfinite(value)
                        and np.isclose(float(check["value"]), value, rtol=0, atol=1e-9)))
            require(check.get("day") == d and check.get("asof_query_utc") == source.civil_cutoff(stamp).isoformat()
                    and matches, f"{name}/{d}: source differs from independently queried as-of state")
            if pd.isna(value):
                absent.append(d)
        sources[name] = frame
        missing[name] = absent
    require(receipt.get("missing_days_by_series") == missing, "Thermal missingness declaration differs")
    computed = source.build_features(sources, day)
    for zone, frame in computed.items():
        stored = pd.read_parquet(_bound(bundle, receipt, f"source_artifacts/thermal_capacity/features_{zone}.parquet"))
        _exact(stored, frame, f"{zone} thermal features")
    return {"source_daily_states_verified": len(days) * len(source.SOURCES),
            "provider_first_publication_certified": False}


def _fuel(bundle, day, receipt):
    import run_nyx_annual_fuel_source as source
    cache = bundle / "source_artifacts/fuel"
    path = _bound(bundle, receipt, "source_artifacts/fuel/" + source.fuel.OUTPUT_NAME)
    audit_path = _bound(bundle, receipt, "source_artifacts/fuel/" + source.fuel.OUTPUT_NAME + source.fuel.AUDIT_SUFFIX)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source.verify_fuel_artifact(cache, delivery_day=day, history_start_day=audit["start_day"])
    frame = normalize_fuel(pd.read_parquet(path))
    limits = cutoff_times(frame.index, "Europe/Paris")
    for name in FUEL_TIME_COLUMNS[1:]:
        stamps = frame[name]
        require(stamps.notna().all() and (stamps <= limits).all(),
                f"Fuel {name}: source later than its own daily cutoff")
    return {"hourly_source_cutoffs_verified": len(frame), "provider_first_publication_certified": False}


def validate_source_packet(bundle: Path, delivery_day: str) -> dict:
    """Validate all seven raw source groups; derived baseline/reference gates remain separate."""
    bundle = Path(bundle).resolve()
    before = source_graph(bundle, delivery_day)
    full, _, _ = gate.delivery_grid(delivery_day)
    receipts = {group: _receipt(bundle, group) for group in gate.SOURCE_GROUPS}
    checks = {}
    checks["saturn"], prices = _saturn(bundle, delivery_day, receipts["saturn"])
    checks["auction_prices"] = _auction(bundle, delivery_day, receipts["auction_prices"], prices)
    checks["jao_initial"] = _jao(bundle, delivery_day, receipts["jao_initial"], full)
    for group in ("public_hydro", "lagged_exchange"):
        checks[group] = _captured_features(bundle, delivery_day, receipts[group], full, group=group)
    checks["thermal_capacity"] = _thermal(bundle, delivery_day, receipts["thermal_capacity"])
    checks["fuel"] = _fuel(bundle, delivery_day, receipts["fuel"])
    require(source_graph(bundle, delivery_day) == before, "Source packet changed during validation")
    return {"protocol": PROTOCOL, "delivery_day": delivery_day, "passed": True,
            "source_snapshot_asof_verified": True, "supplier_first_publication_certified": False,
            "source_receipts_sha256": before[0], "checks": checks}
