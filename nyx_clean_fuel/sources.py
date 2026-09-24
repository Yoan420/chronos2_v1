"""Isolated native Saturn CGC/CCC daily bank, selected independently at D-1 08h.

Native clean costs already contain CO2 and efficiency: never add either again.
Daily labels are trading dates, not publication times. We additionally exclude
the cutoff's entire civil date, including any close returned early by Saturn.
Formula definitions are pinned, but historical formula versioning and provider
insertion timestamps are not attested: this is research evidence, not promotion.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://saturn-energyscan.gem.myengie.com//api"
TIMEZONE = "Europe/Paris"
SCHEMA_VERSION = 1
HUBS = {"fr": "peg", "de": "the", "be": "zee", "nl": "ttf"}
SERIES = {f"cgc_{z}": f"power.{z}.price.everyday.cgc.da.index.eurmwh" for z in HUBS}
SERIES["ccc"] = "ccc.price.mid.api2.everyday.month.1.ice.eurmwh"
FORMULAS = {
    f"cgc_{z}": '(add (* 2 (series "gas.price.everyday.' + hub + '.da.eurmwh' + ('.new' if z == 'de' else '') + '")) (* 0.368 (series "carbon.eu.price.everyday.eua.1stdec.eurt")))'
    for z, hub in HUBS.items()
}
FORMULAS["ccc"] = '(* 2.63 (add (/ (div (series "coal.api2.price.everyday.month.1.ice.usdt") (series "forex.nrjscan.everyday.eurusd.close")) 6.9776) (* 0.34 (series "carbon.eu.price.everyday.eua.ice.1st.dec"))))'
UNITS = {alias: "EUR/MWh_e" for alias in SERIES}
DEFAULTS = {"maximum_age_hours": 176.0, "request_timeout_seconds": 45.0, "retries": 2}
LOCAL = threading.local()


class CleanFuelSourceError(ValueError):
    """The clean cost source contract cannot be satisfied without fabrication."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _day(value: Any) -> str:
    if not isinstance(value, str):
        raise CleanFuelSourceError("Delivery days must be YYYY-MM-DD strings.")
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise CleanFuelSourceError("Invalid delivery date.") from exc
    if pd.isna(day) or day.tz is not None or day.strftime("%Y-%m-%d") != value:
        raise CleanFuelSourceError("Delivery days must be exact YYYY-MM-DD strings.")
    return value


def civil_cutoff(day: str) -> pd.Timestamp:
    previous = pd.Timestamp(_day(day)) - pd.Timedelta(days=1)
    return previous.replace(hour=8).tz_localize(TIMEZONE).tz_convert("UTC")


def _days(start: str, end: str) -> list[str]:
    _day(start)
    _day(end)
    if end < start or (pd.Timestamp(end) - pd.Timestamp(start)).days > 4000:
        raise CleanFuelSourceError("Clean-fuel collection requires 1 to 4001 consecutive days.")
    return [x.strftime("%Y-%m-%d") for x in pd.date_range(start, end, freq="D")]


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config.get("sources", {})
    if not isinstance(source, Mapping) or set(source).difference(DEFAULTS):
        raise CleanFuelSourceError("Unknown clean-fuel sources setting.")
    settings = {**DEFAULTS, **source}
    for name in ("maximum_age_hours", "request_timeout_seconds"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= (176 if name == "maximum_age_hours" else 120):
            raise CleanFuelSourceError(f"Invalid {name}.")
        settings[name] = float(value)
    retries = settings["retries"]
    if isinstance(retries, bool) or not isinstance(retries, int) or not 1 <= retries <= 3:
        raise CleanFuelSourceError("retries must be an integer from 1 to 3.")
    return settings


def _contract(maximum_age_hours: float) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "series": SERIES, "formulas": FORMULAS,
            "units": UNITS, "maximum_age_hours": maximum_age_hours,
            "cutoff_timezone": TIMEZONE, "cutoff_time": "08:00", "cutoff_day_offset": -1,
            "same_day_closes_excluded": True, "source_naive_timezone": TIMEZONE,
            "selection": "latest finite prior-civil-date value in independently queried asof state"}


def _output(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    project = ROOT.resolve()
    allowed = [(ROOT / "data/pit/nyx_clean_fuel").resolve(),
               (ROOT / "runs/experiments/nyx_clean_fuel_v1/inputs").resolve()]
    if any(project not in base.parents for base in allowed):
        raise CleanFuelSourceError("Isolated output namespaces cannot redirect outside the project.")
    if not any(resolved == base or base in resolved.parents for base in allowed):
        raise CleanFuelSourceError("Source output must remain in the isolated nyx_clean_fuel input namespace.")
    return resolved


def _atomic_bytes(path: Path, raw: bytes) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _locked(directory: Path):
    path = directory / ".materialize.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CleanFuelSourceError(f"Another clean-fuel collection owns {path}; do not remove an active lock.") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes({"pid": os.getpid(), "created_at_utc": datetime.now(timezone.utc).isoformat()}))
        yield
    finally:
        path.unlink()


def _client(timeout: float):
    if getattr(LOCAL, "timeout", None) != timeout:
        from chronos2_modular.saturn import create_saturn_client
        client = create_saturn_client(URL, "BQ6757")
        client.session.request = partial(client.session.request, timeout=timeout)
        LOCAL.client, LOCAL.timeout = client, timeout
    return LOCAL.client


def _retry(operation, retries: int):
    for attempt in range(retries):
        try:
            return operation()
        except (requests.Timeout, requests.ConnectionError) as exc:
            # TLS/certificate failures are not transient and must never trigger
            # insecure fallback. Validation, HTTP4xx and malformed data fail fast.
            if isinstance(exc, requests.exceptions.SSLError) or attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def _source_evidence(settings: dict[str, Any]) -> dict[str, Any]:
    client = _client(settings["request_timeout_seconds"])
    evidence = {}
    for alias, name in SERIES.items():
        response = _retry(lambda: client.session.get(URL + "/series/formula", params={"name": name}), settings["retries"])
        response.raise_for_status()
        formula = response.json()
        if formula != FORMULAS[alias]:
            raise CleanFuelSourceError(f"{alias}: native formula changed; review units/hub/CO2 before reuse.")
        metadata = _retry(lambda: client.metadata(name, all=True), settings["retries"])
        if not isinstance(metadata, dict) or metadata.get("tzaware") is not False:
            raise CleanFuelSourceError(f"{alias}: expected naive daily Saturn trading-date labels.")
        evidence[alias] = {"series": name, "formula": formula, "formula_sha256": _sha(formula.encode("utf8")),
                           "unit": UNITS[alias], "tzaware": False}
    return evidence


def select_previous_close(raw: Any, cutoff: pd.Timestamp, *, maximum_age_hours: float = 176) -> tuple[float, pd.Timestamp, float]:
    if not isinstance(raw, pd.Series) or not isinstance(raw.index, pd.DatetimeIndex):
        raise CleanFuelSourceError("Saturn must return a dated numeric Series.")
    if raw.index.hasnans or raw.index.has_duplicates:
        raise CleanFuelSourceError("Missing or duplicated source timestamps.")
    try:
        values = pd.to_numeric(raw, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise CleanFuelSourceError("Malformed source values.") from exc
    if np.isinf(values).any():
        raise CleanFuelSourceError("Infinite source values.")
    index = raw.index
    if index.tz is None:
        index = index.tz_localize(TIMEZONE, ambiguous="raise", nonexistent="raise")
    index = index.tz_convert("UTC")
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tz is None:
        raise CleanFuelSourceError("A timezone-aware cutoff is required.")
    local = cutoff.tz_convert(TIMEZONE)
    values.index = index
    # A last available previous close is explicit bounded state selection, not
    # filling future deliveries or substituting the latest revised series.
    selected = values[(index < local.normalize().tz_convert("UTC")) & (index <= cutoff)].dropna().sort_index()
    if selected.empty:
        raise CleanFuelSourceError("No finite previous close known before the cutoff's civil day.")
    stamp, value = selected.index[-1], float(selected.iloc[-1])
    age = float((cutoff - stamp).total_seconds() / 3600)
    if not math.isfinite(maximum_age_hours) or maximum_age_hours <= 0 or age > maximum_age_hours:
        raise CleanFuelSourceError("Native clean cost exceeds the maximum source age.")
    return value, stamp, age


def _one_day(day: str, settings: dict[str, Any]) -> dict[str, Any]:
    cutoff = civil_cutoff(day)
    client = _client(settings["request_timeout_seconds"])
    row = {"delivery_day": day, "cutoff_time_utc": cutoff.isoformat()}
    for alias, name in SERIES.items():
        raw = _retry(lambda: client.get(name, from_value_date=cutoff - pd.Timedelta(days=11),
                                       to_value_date=cutoff, revision_date=cutoff), settings["retries"])
        value, stamp, age = select_previous_close(raw, cutoff, maximum_age_hours=settings["maximum_age_hours"])
        row.update({alias: value, alias + "__value_time_utc": stamp.isoformat(), alias + "__age_hours": age})
    return row


def _columns() -> list[str]:
    return ["delivery_day", "cutoff_time_utc"] + [name for alias in SERIES for name in (alias, alias + "__value_time_utc", alias + "__age_hours")]


def _validate_frame(frame: pd.DataFrame, *, maximum_age_hours: float, expected_days: list[str]) -> pd.DataFrame:
    if list(frame.columns) != _columns() or frame.empty or frame.isna().any().any():
        raise CleanFuelSourceError("Incomplete or unexpected daily bank schema.")
    if frame.delivery_day.tolist() != expected_days:
        raise CleanFuelSourceError("The bank must contain exactly the ordered consecutive requested days.")
    result = frame.copy()
    try:
        if any(pd.Timestamp(stamp).tz is None for stamp in frame.cutoff_time_utc):
            raise CleanFuelSourceError("Persisted cutoff timestamps must be timezone-aware.")
        cutoff = pd.DatetimeIndex(pd.to_datetime(frame.cutoff_time_utc, utc=True, errors="raise"))
        expected_cutoff = pd.DatetimeIndex([civil_cutoff(day) for day in expected_days])
        if not cutoff.equals(expected_cutoff):
            raise CleanFuelSourceError("Every bank day must have its exact D-1 civil 08h cutoff.")
        result["cutoff_time_utc"] = cutoff
        for alias in SERIES:
            raw_stamp = frame[alias + "__value_time_utc"]
            # pd.to_datetime(...,utc=True) alone would silently accept UTC-naive
            # strings. Persisted source timestamps must carry explicit offsets.
            if any(pd.Timestamp(stamp).tz is None for stamp in raw_stamp):
                raise CleanFuelSourceError("Persisted source timestamps must be timezone-aware.")
            stamp = pd.DatetimeIndex(pd.to_datetime(raw_stamp, utc=True, errors="raise"))
            values = pd.to_numeric(frame[alias], errors="raise").to_numpy(dtype=float)
            ages = pd.to_numeric(frame[alias + "__age_hours"], errors="raise").to_numpy(dtype=float)
            correct_ages = (cutoff - stamp).total_seconds().to_numpy() / 3600
            if not np.isfinite(values).all() or not np.isfinite(ages).all() or not np.allclose(ages, correct_ages, atol=1e-10, rtol=0):
                raise CleanFuelSourceError("Malformed values or inconsistent source ages.")
            if (ages < 0).any() or (ages > maximum_age_hours).any() or (stamp >= cutoff.tz_convert(TIMEZONE).normalize().tz_convert("UTC")).any():
                raise CleanFuelSourceError("Source values violate previous-close timing or age limits.")
            result[alias] = values
            result[alias + "__value_time_utc"] = stamp
            result[alias + "__age_hours"] = ages
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CleanFuelSourceError):
            raise
        raise CleanFuelSourceError(f"Invalid daily bank timestamps/numeric fields: {exc}") from exc
    return result


def load_bank(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    raw = path.read_bytes()
    audit_path = Path(str(path) + ".audit.json")
    audit_raw = audit_path.read_bytes()
    audit = json.loads(audit_raw)
    if not isinstance(audit, dict) or audit.get("sha256") != _sha(raw):
        raise CleanFuelSourceError("Clean-fuel bank checksum mismatch.")
    maximum_age = audit.get("maximum_age_hours")
    _settings({"sources": {"maximum_age_hours": maximum_age}})
    contract = _contract(maximum_age)
    if any(audit.get(key) != value for key, value in contract.items()) or audit.get("contract_sha256") != _sha(_json_bytes(contract)):
        raise CleanFuelSourceError("Clean-fuel bank provenance/units/selection contract mismatch.")
    for flag in ("provider_revision_timestamp_available", "historical_formula_version_attested", "production_pit_evidence", "promotion_eligible"):
        if audit.get(flag) is not False:
            raise CleanFuelSourceError("Research provenance limitations must remain explicit.")
    evidence = audit.get("source_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(SERIES):
        raise CleanFuelSourceError("Missing native source formula evidence.")
    for alias in SERIES:
        spec = evidence[alias]
        if spec != {"series": SERIES[alias], "formula": FORMULAS[alias], "formula_sha256": _sha(FORMULAS[alias].encode("utf8")), "unit": UNITS[alias], "tzaware": False}:
            raise CleanFuelSourceError("Native source evidence differs from the audited registry.")
    days = _days(audit.get("start_day"), audit.get("end_day"))
    if audit.get("days") != len(days):
        raise CleanFuelSourceError("Source-bank day count mismatch.")
    frame = pd.read_parquet(io.BytesIO(raw))
    result = _validate_frame(frame, maximum_age_hours=maximum_age, expected_days=days)
    if path.read_bytes() != raw or audit_path.read_bytes() != audit_raw:
        raise CleanFuelSourceError("Source bank changed concurrently during reading.")
    return result, audit


def _seed_path(path: Path, parent: Path) -> Path:
    """A seed is an ordinary sibling directory/file, never a link or junction."""
    if path.resolve() != path.absolute() or path.parent.resolve() != parent:
        raise CleanFuelSourceError("Sibling clean-fuel seeds cannot follow symlinks or junctions.")
    return path


def _reuse_sibling_banks(directory: Path, days: list[str], rows: dict[str, dict],
                        fingerprint: str, maximum_age: float) -> dict[str, Any]:
    """Copy only validated overlapping daily rows; completed banks stay untouched.

    Selection is deterministic (largest remaining overlap, then latest end day,
    then directory name). An incompatible contract is explicitly skipped. Any
    selected candidate must pass the full load_bank validation; corruption never
    becomes a silent reason to download replacements or choose a different seed.
    """
    result: dict[str, Any] = {"strategy": "verified_immutable_sibling_banks", "copied_days": 0,
                              "sources": [], "skipped": []}
    namespaces = {(ROOT / "data/pit/nyx_clean_fuel").resolve(),
                  (ROOT / "runs/experiments/nyx_clean_fuel_v1/inputs").resolve()}
    # Do not scan arbitrary ancestors if an advanced caller chose a deeper
    # subdirectory. Normal operational inputs are direct YYYY-MM-DD children.
    if directory.parent not in namespaces or set(days).issubset(rows):
        return result
    candidates = []
    for sibling in sorted(directory.parent.iterdir(), key=lambda item: item.name):
        if sibling == directory or re.fullmatch(r"\d{4}-\d{2}-\d{2}", sibling.name) is None:
            continue
        _day(sibling.name)
        _seed_path(sibling, directory.parent)
        if not sibling.is_dir():
            continue
        path = sibling / "bank.parquet"
        audit_path = Path(str(path) + ".audit.json")
        for item in (path, audit_path):
            _seed_path(item, sibling)
        if (sibling / ".materialize.lock").exists():
            result["skipped"].append({"path": str(path.relative_to(ROOT)), "reason": "collection_active"})
            continue
        if not path.is_file():
            continue  # unfinished sibling; checkpoints are not trusted seeds
        try:
            header = json.loads(audit_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise CleanFuelSourceError(f"Sibling bank has unreadable provenance: {path}") from exc
        if not isinstance(header, dict):
            raise CleanFuelSourceError(f"Sibling bank has malformed provenance: {path}")
        if header.get("contract_sha256") != fingerprint:
            result["skipped"].append({"path": str(path.relative_to(ROOT)), "reason": "different_source_contract",
                                       "contract_sha256": header.get("contract_sha256")})
            continue
        covered = set(_days(header.get("start_day"), header.get("end_day")))
        if not covered.intersection(days):
            result["skipped"].append({"path": str(path.relative_to(ROOT)), "reason": "no_requested_overlap"})
            continue
        candidates.append((path, header, covered))
    while candidates:
        missing = set(days).difference(rows)
        candidates.sort(key=lambda item: (len(missing.intersection(item[2])), item[1]["end_day"], item[0].parent.name), reverse=True)
        path, header, covered = candidates.pop(0)
        selected_days = sorted(missing.intersection(covered))
        if not selected_days:
            result["skipped"].append({"path": str(path.relative_to(ROOT)), "reason": "requested_days_already_available"})
            continue
        audit_path = Path(str(path) + ".audit.json")
        audit_sha = _sha(audit_path.read_bytes())
        try:
            seed, audit = load_bank(path)
        except (OSError, ValueError) as exc:
            raise CleanFuelSourceError(f"Selected sibling seed is invalid; refusing replacement: {path}: {exc}") from exc
        if audit["contract_sha256"] != fingerprint or audit["start_day"] != header["start_day"] or audit["end_day"] != header["end_day"] or _sha(audit_path.read_bytes()) != audit_sha:
            raise CleanFuelSourceError("Selected sibling seed changed during selection.")
        indexed = seed.set_index("delivery_day", drop=False)
        # Existing checkpoints and a seed describe the same as-of state. A
        # conflict signals revisions/contract drift, not a licence to mix them.
        for day in sorted(set(rows).intersection(covered)):
            expected = indexed.loc[[day]].reset_index(drop=True)
            actual = _validate_frame(pd.DataFrame([rows[day]], columns=_columns()), maximum_age_hours=maximum_age, expected_days=[day])
            if not actual.equals(expected):
                raise CleanFuelSourceError(f"{day}: existing checkpoint/seed values conflict.")
        for day in selected_days:
            rows[day] = {key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                         for key, value in indexed.loc[day].to_dict().items()}
        result["sources"].append({"path": str(path.relative_to(ROOT)), "sha256": audit["sha256"],
                                    "audit_sha256": audit_sha, "contract_sha256": fingerprint,
                                    "start_day": audit["start_day"], "end_day": audit["end_day"],
                                    "copied_days": selected_days, "copied_count": len(selected_days)})
        result["copied_days"] += len(selected_days)
    return result


def _verify_seed_sources(reuse: dict[str, Any]) -> None:
    for source in reuse["sources"]:
        path = ROOT / source["path"]
        _seed_path(path.parent, path.parent.parent.resolve())
        _seed_path(path, path.parent.resolve())
        audit = Path(str(path) + ".audit.json")
        _seed_path(audit, path.parent.resolve())
        if _sha(path.read_bytes()) != source["sha256"] or _sha(audit.read_bytes()) != source["audit_sha256"]:
            raise CleanFuelSourceError("A reused immutable sibling bank changed during collection.")


def materialize(config: Mapping[str, Any], start_day: str, end_day: str,
                output_dir: str | Path, workers: int = 2) -> Path:
    settings = _settings(config)
    days = _days(start_day, end_day)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 2:
        raise CleanFuelSourceError("At most two source workers are allowed.")
    directory = _output(output_dir)
    contract = _contract(settings["maximum_age_hours"])
    fingerprint = _sha(_json_bytes(contract))
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "bank.parquet"
    with _locked(directory):
        if output.exists():
            _, audit = load_bank(output)
            if audit["contract_sha256"] != fingerprint:
                raise CleanFuelSourceError("Existing bank uses different source settings; use a new input directory.")
            if audit["start_day"] == start_day and audit["end_day"] == end_day:
                return output
            raise CleanFuelSourceError("A completed bank is immutable; choose a new input directory for new dates.")
        checkpoint_dir = directory / "daily"
        checkpoint_dir.mkdir(exist_ok=True)
        if checkpoint_dir.resolve().parent != directory:
            raise CleanFuelSourceError("Daily checkpoints cannot redirect outside their input directory.")
        evidence = _source_evidence(settings)
        rows = {}
        for day in days:
            checkpoint = checkpoint_dir / (day + ".json")
            if not checkpoint.exists():
                continue
            record = json.loads(checkpoint.read_text(encoding="utf8"))
            if not isinstance(record, dict) or record.get("contract_sha256") != fingerprint or record.get("row_sha256") != _sha(_json_bytes(record.get("row"))):
                raise CleanFuelSourceError(f"{day}: checkpoint contract/checksum mismatch.")
            _validate_frame(pd.DataFrame([record["row"]], columns=_columns()), maximum_age_hours=settings["maximum_age_hours"], expected_days=[day])
            if set(record["row"]) != set(_columns()):
                raise CleanFuelSourceError("Unexpected checkpoint fields.")
            rows[day] = record["row"]
        reuse = _reuse_sibling_banks(directory, days, rows, fingerprint, settings["maximum_age_hours"])
        missing = [day for day in days if day not in rows]
        started = time.monotonic()
        print(f"[CleanFuel] {len(rows)}/{len(days)} valid days ({reuse['copied_days']} reused from immutable banks); {len(missing)} days to collect.", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one_day, day, settings): day for day in missing}
            try:
                for number, future in enumerate(as_completed(futures), 1):
                    day, row = futures[future], future.result()
                    _validate_frame(pd.DataFrame([row], columns=_columns()), maximum_age_hours=settings["maximum_age_hours"], expected_days=[day])
                    record = {"contract_sha256": fingerprint, "row": row, "row_sha256": _sha(_json_bytes(row))}
                    _atomic_bytes(checkpoint_dir / (day + ".json"), _json_bytes(record))
                    rows[day] = row
                    if number == 1 or number % 25 == 0 or number == len(missing):
                        print(f"[CleanFuel] {len(rows)}/{len(days)} days; elapsed {(time.monotonic()-started)/60:.1f} min.", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if _source_evidence(settings) != evidence:
            raise CleanFuelSourceError("Native source definitions changed during collection.")
        _verify_seed_sources(reuse)
        frame = _validate_frame(pd.DataFrame([rows[day] for day in days], columns=_columns()), maximum_age_hours=settings["maximum_age_hours"], expected_days=days)
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        raw = buffer.getvalue()
        audit = {**contract, "contract_sha256": fingerprint, "sha256": _sha(raw), "source_evidence": evidence,
                 "start_day": start_day, "end_day": end_day, "days": len(days),
                 "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                 "seed_reuse": reuse,
                 "provider_revision_timestamp_available": False, "historical_formula_version_attested": False,
                 "production_pit_evidence": False, "promotion_eligible": False,
                 "revision_semantics": "Saturn state queried at exact cutoff, not independently attested provider insertion time",
                 "native_clean_costs_include_co2_and_efficiency": True,
                 "gas_hubs": {k.upper(): v.upper() for k, v in HUBS.items()},
                 "ccc_contract": "API2 M1; 6.9776 MWh_th/t; inverse efficiency 2.63; CO2 0.34 t/MWh_th"}
        # Audit is staged first: an interrupted write leaves no apparently valid
        # data file; all readers require both matching artifacts.
        _atomic_bytes(Path(str(output) + ".audit.json"), _json_bytes(audit))
        _atomic_bytes(output, raw)
        load_bank(output)
        return output
