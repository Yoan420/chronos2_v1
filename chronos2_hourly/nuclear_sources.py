"""Isolated, fail-closed contract for Saturn nuclear generation forecasts.

This is an as-of-query audit, not evidence of provider publication timestamps.
The local-civil source has a singleton autumn 02:00 in the historical probes;
``duplicate`` is an explicit covariate convention, never an independently
observed second forecast. No function in this module downloads or writes data.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER = ROOT / "materialize_saturn_daily_asof.py"
NUCLEAR_SERIES = "power.fr.generation.nuclear.gw.fcst"
NUCLEAR_ALIAS = "fr_nuclear_generation_fcst_gw"
NUCLEAR_TIMEZONE = "Europe/Paris"
DEFAULT_NUCLEAR_PATH = (
    ROOT / "data" / "pit" / "nuclear_forecast" / f"{NUCLEAR_ALIAS}.parquet"
)
_COLUMNS = (
    "value_time_utc", "snapshot_time_utc", "revision_time_utc", "value",
    "downloaded_at_utc",
)
_DUPLICATE_FILL = (
    "none_except_duplicate_missing_autumn_fold_if_source_is_civil_naive"
)
_REVISION_SEMANTICS = (
    "query_asof_cutoff; provider insertion timestamp is not returned by "
    "Client.get(revision_date=...)"
)


def _day(value: Any, name: str) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(f"{name} must be a naive local calendar date.")
    return day


def _days(start_day: Any, end_day: Any) -> pd.DatetimeIndex:
    start, end = _day(start_day, "start_day"), _day(end_day, "end_day")
    if start > end:
        raise ValueError("start_day must not be after inclusive end_day.")
    return pd.date_range(start, end, freq="D")


def _physical_hours(day: pd.Timestamp) -> pd.DatetimeIndex:
    start = day.tz_localize(NUCLEAR_TIMEZONE).tz_convert("UTC")
    end = (day + pd.Timedelta(days=1)).tz_localize(NUCLEAR_TIMEZONE).tz_convert("UTC")
    return pd.date_range(start, end, freq="h", inclusive="left")


def build_materialize_command(
    start_day: Any,
    end_day: Any,
    output: str | Path,
    python_executable: str | Path,
    workers: int,
    incomplete_dst_policy: str = "duplicate",
) -> list[str]:
    """Build argv for an isolated rebuild; both civil dates are inclusive.

    ``duplicate`` permits the materializer's disclosed singleton-autumn-fold
    convention. ``raise`` refuses it. Neither allows arbitrary missing hours.
    The helper intentionally never supplies ``--merge-existing``.
    """
    days = _days(start_day, end_day)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer between 1 and 32.")
    if incomplete_dst_policy not in {"duplicate", "raise"}:
        raise ValueError("incomplete_dst_policy must be 'duplicate' or 'raise'.")
    if not str(python_executable).strip():
        raise ValueError("python_executable must not be empty.")
    destination = Path(output).expanduser().resolve()
    legacy_root = (ROOT / "data" / "pit" / "vintages").resolve()
    if destination == legacy_root or legacy_root in destination.parents:
        raise ValueError("Do not overwrite the shared legacy vintage store; use nuclear_forecast.")
    return [
        str(python_executable), str(MATERIALIZER),
        "--series", NUCLEAR_SERIES, "--alias", NUCLEAR_ALIAS,
        "--start-day", days[0].date().isoformat(),
        "--end-day", days[-1].date().isoformat(),
        "--output", str(destination),
        "--timezone", NUCLEAR_TIMEZONE,
        "--naive-timezone", NUCLEAR_TIMEZONE,
        "--cutoff-timezone", NUCLEAR_TIMEZONE, "--cutoff-time", "08:00",
        "--value-scale", "1", "--incomplete-dst-policy", incomplete_dst_policy,
        "--request-padding-hours", "8", "--workers", str(workers),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_blockers(payload: dict[str, Any]) -> list[str]:
    expected = {
        "schema_version": 1,
        "series": NUCLEAR_SERIES,
        "alias": NUCLEAR_ALIAS,
        "timezone": NUCLEAR_TIMEZONE,
        "naive_timezone": NUCLEAR_TIMEZONE,
        "cutoff_timezone": NUCLEAR_TIMEZONE,
        "cutoff_time": "08:00",
        "daily_broadcast": False,
        "value_scale": 1.0,
        "causal_contract": "Saturn state queried as-of D-1 civil cutoff",
        "snapshot_time_semantics": "query_asof_cutoff",
        "revision_time_semantics": _REVISION_SEMANTICS,
        "provider_revision_timestamp_available": False,
    }
    blockers = []
    for key, value in expected.items():
        actual = payload.get(key)
        valid = actual == value
        if isinstance(value, bool):
            valid = actual is value
        if key in {"schema_version", "value_scale"} and isinstance(actual, bool):
            valid = False
        if not valid:
            blockers.append(f"Source audit {key} must be {value!r}; rematerialize the isolated GW source.")
    if payload.get("unit", "GW") != "GW":
        blockers.append("Source audit unit must be GW; REMIT MW is not this source.")
    policy = payload.get("incomplete_dst_policy")
    if not isinstance(policy, str) or policy not in {"duplicate", "raise"}:
        blockers.append("Source audit must declare incomplete_dst_policy 'duplicate' or 'raise'.")
    fill = _DUPLICATE_FILL if policy == "duplicate" else "none"
    if payload.get("fill_or_interpolation") != fill:
        blockers.append("Source audit fill_or_interpolation does not match its explicit DST policy.")
    return blockers


def audit_nuclear_store(
    path: str | Path, start_day: Any, end_day: Any,
) -> dict[str, Any]:
    """Audit bytes, source contract, causal selection and finite physical hours.

    Returns JSON-safe actionable blockers rather than raising for an unusable
    artifact. Invalid caller date arguments raise ValueError. ``complete`` is
    a data-readiness result, not production or provider-publication PIT proof.
    Latest admissible rows are selected before testing finite values: an older
    finite vintage cannot silently conceal a missing/nonfinite latest value.
    """
    days = _days(start_day, end_day)
    store = Path(path).expanduser().resolve()
    sidecar = store.with_name(store.name + ".audit.json")
    expected = _physical_hours(days[0])
    for day in days[1:]:
        expected = expected.append(_physical_hours(day))
    result: dict[str, Any] = {
        "schema_version": 1, "path": str(store), "source_audit_path": str(sidecar),
        "series": NUCLEAR_SERIES, "alias": NUCLEAR_ALIAS, "unit": "GW",
        "start_day": days[0].date().isoformat(), "end_day": days[-1].date().isoformat(),
        "days": len(days), "expected_hours": len(expected), "covered_hours": 0,
        "complete": False, "source_provenance_valid": False, "timing_valid": False,
        "provider_revision_timestamp_available": False, "production_pit_evidence": False,
        "pit_evidence_level": "query_asof_cutoff_only",
        "blockers": [], "warnings": [
            "Snapshot/revision times identify the Saturn as-of query, not provider publication; "
            "this audit alone does not establish production PIT evidence."
        ],
    }
    blockers = result["blockers"]
    payload: dict[str, Any] = {}
    try:
        raw_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            raise ValueError("sidecar root must be an object")
        payload = raw_payload
    except (OSError, UnicodeError, ValueError):
        blockers.append("Missing or invalid .parquet.audit.json; rebuild with the nuclear source command.")
    if payload:
        blockers.extend(_source_blockers(payload))
    frame = pd.DataFrame(columns=list(_COLUMNS))
    try:
        actual_sha256 = _sha256(store)
        result["sha256"] = actual_sha256
        if payload.get("sha256") != actual_sha256:
            blockers.append("Source audit SHA256 does not match the parquet; rebuild or recover its matching audit.")
        frame = pd.read_parquet(store)
    except (OSError, ValueError, ImportError):
        blockers.append("Nuclear parquet is missing or unreadable; materialize the isolated source first.")
    result["source_provenance_valid"] = bool(payload) and not blockers
    result["rows"] = len(frame)
    missing_columns = sorted(set(_COLUMNS).difference(frame.columns))
    if missing_columns:
        blockers.append(f"Nuclear parquet lacks PIT columns: {', '.join(missing_columns)}.")
    chosen = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    timing_blockers: list[str] = []
    if not missing_columns and not frame.empty:
        frame = frame.copy()
        for column in (name for name in _COLUMNS if name != "value"):
            # An implicit conversion of naive timestamps could conceal the old
            # local-civil-as-UTC mistake. Require an explicit timezone first.
            try:
                index = pd.DatetimeIndex(frame[column])
                aware = index.tz is not None
            except (ValueError, TypeError):
                aware = False
            if not aware:
                timing_blockers.append(f"{column} must carry an explicit UTC-aware timezone.")
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
            if frame[column].isna().any():
                timing_blockers.append(f"{column} contains invalid or missing timestamps.")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        valid_times = frame.loc[frame[list(_COLUMNS[:3])].notna().all(axis=1)].copy()
        value_times = valid_times["value_time_utc"]
        if (value_times != value_times.dt.floor("h")).any():
            timing_blockers.append("Delivery timestamps must identify exact physical hourly products.")
        if (valid_times["snapshot_time_utc"] != valid_times["revision_time_utc"]).any():
            timing_blockers.append("As-of-cutoff snapshot and revision timestamps must agree.")
        if (valid_times["downloaded_at_utc"] < valid_times["snapshot_time_utc"]).any():
            timing_blockers.append("An as-of query cutoff cannot postdate its recorded download time.")
        local_days = value_times.dt.tz_convert(NUCLEAR_TIMEZONE).dt.tz_localize(None).dt.normalize()
        cutoffs = (
            local_days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
        ).dt.tz_localize(NUCLEAR_TIMEZONE).dt.tz_convert("UTC")
        admissible = (
            valid_times["snapshot_time_utc"].le(cutoffs)
            & valid_times["revision_time_utc"].le(cutoffs)
        )
        result["post_cutoff_rows_excluded"] = int((~admissible).sum())
        eligible = valid_times.loc[admissible].copy()
        identity = ["value_time_utc", "snapshot_time_utc", "revision_time_utc"]
        if eligible.duplicated(identity, keep=False).any():
            timing_blockers.append("Duplicate PIT row identities make selection ambiguous; rematerialize without overlap conflicts.")
        eligible = eligible.sort_values(["snapshot_time_utc", "revision_time_utc"]).drop_duplicates(
            "value_time_utc", keep="last",
        )
        chosen = eligible.set_index("value_time_utc")["value"].reindex(expected)
        if payload.get("rows") != len(frame):
            blockers.append("Source audit row count disagrees with the parquet.")
        if not local_days.empty:
            actual_span = {
                "start_day": local_days.min().date().isoformat(),
                "end_day": local_days.max().date().isoformat(),
                "days": int(local_days.nunique()),
            }
            for key, actual in actual_span.items():
                if payload.get(key) != actual:
                    blockers.append(f"Source audit {key} disagrees with the parquet delivery span.")
    else:
        timing_blockers.append("Nuclear store has no auditable PIT rows.")
    result["source_provenance_valid"] = bool(payload) and not blockers
    blockers.extend(timing_blockers)
    result["timing_valid"] = not timing_blockers
    selected = chosen.reindex(expected)
    finite = selected.map(lambda value: pd.notna(value) and math.isfinite(float(value)))
    missing = expected[~finite.to_numpy(dtype=bool)]
    result["covered_hours"] = int(finite.sum())
    result["missing_hour_count"] = len(missing)
    result["missing_hours"] = [value.isoformat() for value in missing]
    result["missing_days"] = sorted(set(missing.tz_convert(NUCLEAR_TIMEZONE).strftime("%Y-%m-%d")))
    coverage = []
    autumn_days = []
    for day in days:
        physical = _physical_hours(day)
        day_missing = physical.intersection(missing)
        coverage.append({
            "day": day.date().isoformat(), "expected_hours": len(physical),
            "covered_hours": len(physical) - len(day_missing), "missing_hours": len(day_missing),
        })
        if len(physical) == 25:
            civil = physical.tz_convert(NUCLEAR_TIMEZONE).tz_localize(None)
            folds = physical[civil.duplicated(keep=False)]
            autumn_days.append({
                "day": day.date().isoformat(),
                "ambiguous_physical_hours_utc": [value.isoformat() for value in folds],
            })
    result["coverage_by_day"] = coverage
    raw_policy = payload.get("incomplete_dst_policy")
    policy = raw_policy if isinstance(raw_policy, str) else None
    result["synthetic_dst_disclosure"] = {
        "policy": policy,
        "convention": "duplicate_with_audit" if policy == "duplicate" else "no_synthetic_fold_allowed",
        "potentially_synthetic_days": autumn_days if policy == "duplicate" else [],
        "potential_added_fold_hours": len(autumn_days) if policy == "duplicate" else 0,
        "exact_repair_count_available": False,
        "independent_second_fold_evidence": False,
        "explanation": (
            "The generic materializer records the duplicate policy but not per-row repairs. "
            "Every requested autumn fold is conservatively disclosed as potentially synthesized; "
            "the singleton civil 02:00 may be repeated into both physical UTC hours. "
            "This is a forecast covariate assumption, not independent second-fold evidence."
            if policy == "duplicate" else
            "The strict source policy does not permit synthesis; provider publication evidence remains unavailable."
        ),
    }
    if len(missing):
        blockers.append(f"Missing {len(missing)} finite admissible physical hours across {len(result['missing_days'])} days; materialize those days.")
    result["complete"] = not blockers and not len(missing)
    return result


__all__ = [
    "NUCLEAR_SERIES", "NUCLEAR_ALIAS", "NUCLEAR_TIMEZONE", "DEFAULT_NUCLEAR_PATH",
    "build_materialize_command", "audit_nuclear_store",
]
