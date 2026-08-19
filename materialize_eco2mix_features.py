#!/usr/bin/env python
"""Materialize isolated, lagged Eco2Mix features from the public ODRE API.

``eco2mix-national-cons-def`` is a revision-latest consolidated/definitive
dataset, not a publication-vintage archive.  In particular, the dataset does
not provide a timestamp proving that ``prevision_j1`` for delivery day D was
available at a D-1 08:00 cutoff.  This materializer therefore exports only
complete civil-day profiles shifted by D-2 or D-7, including historical J-1
load forecasts, and labels every output as ``exploratory_non_pit``.

No DST interpolation is performed.  Source and target hours are matched by
``(local civil day, local hour, fold)``.  An hour that cannot be matched stays
NaN and is accompanied by an explicit missing/alignment flag.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from chronos2_hourly.hourly_contract import (
    build_delivery_metadata,
    local_delivery_day_index,
)


DATASET_ID = "eco2mix-national-cons-def"
API_URL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    f"{DATASET_ID}/records"
)
EXPORT_CSV_URL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    f"{DATASET_ID}/exports/csv"
)
TIMEZONE = "Europe/Paris"
CLASSIFICATION = "exploratory_non_pit"
PROFILE_LAGS_DAYS = (2, 7)
MAX_API_PAGE_SIZE = 100

SOURCE_FIELD_ALIASES = {
    "prevision_j1": "load_fcst_j1",
    "consommation": "consommation",
    "nucleaire": "nucleaire",
    "eolien": "eolien",
    "solaire": "solaire",
    "hydraulique": "hydraulique",
    "pompage": "pompage",
    "ech_physiques": "ech_physiques",
}
RAW_FIELDS = ("date_heure", *SOURCE_FIELD_ALIASES)

# The consolidated dataset exposes forecasts at quarter-hour resolution and
# realised mix/consumption values at half-hour resolution.  A partial hour is
# deliberately converted to NaN rather than averaged silently.
MIN_POINTS_PER_HOUR = {
    "prevision_j1": 4,
    **{field: 2 for field in SOURCE_FIELD_ALIASES if field != "prevision_j1"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_day(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None or timestamp != timestamp.normalize():
        raise ValueError(f"Expected an unzoned civil date, got {value!r}")
    return timestamp


def _validate_days(start_day: str, end_day: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = _parse_day(start_day)
    end = _parse_day(end_day)
    if start > end:
        raise ValueError(f"start_day {start_day} is after end_day {end_day}")
    return start, end


def _expected_index(start_day: str, end_day: str) -> pd.DatetimeIndex:
    start, end = _validate_days(start_day, end_day)
    pieces = [
        local_delivery_day_index(day, timezone=TIMEZONE)
        for day in pd.date_range(start, end, freq="D")
    ]
    result = pieces[0].append(pieces[1:])
    result.name = "delivery_start_utc"
    return result


def _source_window_utc(
    start_day: str,
    end_day: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the minimal half-open UTC range needed for D-2/D-7 profiles."""

    start, end = _validate_days(start_day, end_day)
    first_source_day = start - pd.Timedelta(days=max(PROFILE_LAGS_DAYS))
    last_source_day = end - pd.Timedelta(days=min(PROFILE_LAGS_DAYS))
    start_local = first_source_day.tz_localize(TIMEZONE)
    end_local_exclusive = (last_source_day + pd.Timedelta(days=1)).tz_localize(
        TIMEZONE
    )
    return start_local.tz_convert("UTC"), end_local_exclusive.tz_convert("UTC")


def _format_api_timestamp(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _configured_ca_bundle() -> tuple[str | None, bool | str]:
    """Resolve the CA bundle from requests-compatible environment settings."""

    for variable in (
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
    ):
        value = os.environ.get(variable)
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{variable} points to a missing CA bundle: {path}"
            )
        return variable, str(path.resolve())
    return None, True


def _build_session() -> tuple[requests.Session, str | None]:
    ca_variable, verify = _configured_ca_bundle()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    # requests uses REQUESTS_CA_BUNDLE/CURL_CA_BUNDLE only when trust_env is
    # enabled.  ``verify`` is also set explicitly so SSL_CERT_FILE works in
    # managed enterprise environments.
    session.trust_env = True
    session.verify = verify
    session.headers.update(
        {"User-Agent": "chronos2-eco2mix-exploratory/1.0"}
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session, ca_variable


def _utc_chunks(
    start_utc: pd.Timestamp,
    end_utc_exclusive: pd.Timestamp,
    *,
    chunk_days: int,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    if chunk_days < 1 or chunk_days > 31:
        raise ValueError("chunk_days must be between 1 and 31")
    cursor = start_utc
    while cursor < end_utc_exclusive:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end_utc_exclusive)
        yield cursor, chunk_end
        cursor = chunk_end


def _fetch(
    start_day: str,
    end_day: str,
    *,
    page_size: int,
    chunk_days: int = 7,
    transport: str = "records",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch only the bounded source window required by the requested lags."""

    if page_size < 1 or page_size > MAX_API_PAGE_SIZE:
        raise ValueError(
            f"page_size must be between 1 and {MAX_API_PAGE_SIZE}"
        )
    if transport not in {"records", "csv"}:
        raise ValueError("transport must be 'records' or 'csv'")
    start_utc, end_utc_exclusive = _source_window_utc(start_day, end_day)
    owned_session = session is None
    ca_variable: str | None = None
    if session is None:
        session, ca_variable = _build_session()
    # Keep the resolved variable available for audit without exposing its path.
    setattr(session, "_chronos2_ca_env_variable", ca_variable)

    records: list[dict[str, Any]] = []
    try:
        for chunk_start, chunk_end in _utc_chunks(
            start_utc, end_utc_exclusive, chunk_days=chunk_days
        ):
            where = (
                f"date_heure >= '{_format_api_timestamp(chunk_start)}' and "
                f"date_heure < '{_format_api_timestamp(chunk_end)}'"
            )
            if transport == "csv":
                response = session.get(
                    EXPORT_CSV_URL,
                    params={
                        "select": ",".join(RAW_FIELDS),
                        "where": where,
                        "order_by": "date_heure",
                        "use_labels": "false",
                        "delimiter": ",",
                    },
                    timeout=(15, 120),
                )
                response.raise_for_status()
                batch_frame = pd.read_csv(io.StringIO(response.text))
                if not batch_frame.empty:
                    records.extend(batch_frame.to_dict(orient="records"))
                continue
            offset = 0
            while True:
                response = session.get(
                    API_URL,
                    params={
                        "select": ",".join(RAW_FIELDS),
                        "where": where,
                        "order_by": "date_heure",
                        "limit": page_size,
                        "offset": offset,
                    },
                    timeout=(15, 90),
                )
                response.raise_for_status()
                payload = response.json()
                batch = payload.get("results")
                if not isinstance(batch, list):
                    raise RuntimeError("Eco2Mix response has no results list")
                total = int(payload.get("total_count", offset + len(batch)))
                if not batch:
                    if offset < total:
                        raise RuntimeError(
                            "Eco2Mix pagination stopped before total_count"
                        )
                    break
                records.extend(batch)
                offset += len(batch)
                if offset >= total:
                    break
    finally:
        if owned_session:
            session.close()

    if not records:
        raise RuntimeError("Eco2Mix returned no records")
    frame = pd.DataFrame.from_records(records)
    if "date_heure" not in frame:
        raise RuntimeError("Eco2Mix response has no date_heure column")
    duplicate_mask = frame["date_heure"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_rows = frame.loc[duplicate_mask, list(RAW_FIELDS)].copy()
        for column in RAW_FIELDS[1:]:
            duplicate_rows[column] = pd.to_numeric(
                duplicate_rows[column], errors="coerce"
            )
        conflicts = []
        for timestamp, group in duplicate_rows.groupby(
            "date_heure", sort=False
        ):
            reference = group.iloc[0]
            values = group.loc[:, list(RAW_FIELDS[1:])].to_numpy(dtype=float)
            expected = np.broadcast_to(
                reference.loc[list(RAW_FIELDS[1:])].to_numpy(dtype=float),
                values.shape,
            )
            if not np.allclose(
                values, expected, rtol=0.0, atol=0.0, equal_nan=True
            ):
                conflicts.append(str(timestamp))
        if conflicts:
            # The consolidated source contains an ambiguous repeated hour at
            # some DST transitions.  Never select one conflicting value:
            # quarantine every row for those timestamps.  Downstream exact
            # local-hour/fold mapping then emits NaN plus its missing/shape
            # flags, which is safer than silently choosing or averaging.
            frame = frame.loc[~frame["date_heure"].isin(conflicts)].copy()
        frame = frame.drop_duplicates("date_heure", keep="first")
        frame.attrs["identical_duplicate_rows_removed"] = int(
            sum(
                max(0, len(group) - 1)
                for timestamp, group in duplicate_rows.groupby(
                    "date_heure", sort=False
                )
                if str(timestamp) not in conflicts
            )
        )
        frame.attrs["conflicting_duplicate_timestamps_quarantined"] = int(
            len(conflicts)
        )
        frame.attrs["conflicting_duplicate_rows_quarantined"] = int(
            duplicate_rows["date_heure"].astype(str).isin(conflicts).sum()
        )
    else:
        frame.attrs["identical_duplicate_rows_removed"] = 0
        frame.attrs["conflicting_duplicate_timestamps_quarantined"] = 0
        frame.attrs["conflicting_duplicate_rows_quarantined"] = 0
    return frame.reset_index(drop=True)


def _hourly(raw: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(RAW_FIELDS).difference(raw.columns))
    if missing_columns:
        raise ValueError(f"Eco2Mix fields missing from response: {missing_columns}")

    frame = raw.loc[:, list(RAW_FIELDS)].copy()
    frame["date_heure"] = pd.to_datetime(
        frame["date_heure"], utc=True, errors="coerce"
    )
    if frame["date_heure"].isna().any():
        raise ValueError("Eco2Mix contains invalid date_heure values")
    if frame["date_heure"].duplicated().any():
        raise ValueError("duplicate Eco2Mix timestamps")
    invalid_quarters = (
        frame["date_heure"].dt.second.ne(0)
        | frame["date_heure"].dt.microsecond.ne(0)
        | frame["date_heure"].dt.minute.mod(15).ne(0)
    )
    if invalid_quarters.any():
        raise ValueError("Eco2Mix timestamps are not quarter-hour aligned")

    for column in SOURCE_FIELD_ALIASES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.set_index("date_heure").sort_index()
    hour_bucket = frame.index.floor("h")
    values = frame.loc[:, list(SOURCE_FIELD_ALIASES)]
    hourly = values.groupby(hour_bucket, sort=True).mean()
    counts = values.groupby(hour_bucket, sort=True).count()
    for column, minimum in MIN_POINTS_PER_HOUR.items():
        hourly[column] = hourly[column].where(counts[column].ge(minimum))
    hourly.index = pd.DatetimeIndex(hourly.index, name="delivery_start_utc")
    return hourly


def _local_day_profile(
    hourly: pd.DataFrame,
    *,
    source_column: str,
    output_index: pd.DatetimeIndex,
    lag_days: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Map a lagged civil-day profile by exact local hour and fold."""

    if source_column not in hourly:
        raise KeyError(source_column)
    if lag_days not in PROFILE_LAGS_DAYS:
        raise ValueError(f"Unsupported Eco2Mix profile lag: D-{lag_days}")

    target_meta = build_delivery_metadata(output_index, timezone=TIMEZONE)
    source_meta = build_delivery_metadata(hourly.index, timezone=TIMEZONE)
    source_rows = pd.DataFrame(
        {
            "local_date": source_meta["local_date"].to_numpy(),
            "local_hour": source_meta["local_hour"].to_numpy(),
            "fold": source_meta["fold"].to_numpy(),
            "value": hourly[source_column].to_numpy(dtype=float),
        }
    )
    key_columns = ["local_date", "local_hour", "fold"]
    if source_rows.duplicated(key_columns).any():
        raise ValueError("ambiguous Eco2Mix local day/hour/fold keys")

    source_lookup = {
        (row.local_date, int(row.local_hour), int(row.fold)): float(row.value)
        for row in source_rows.itertuples(index=False)
    }
    source_key_sets: dict[Any, set[tuple[int, int]]] = {}
    for local_date, group in source_rows.groupby("local_date", sort=False):
        source_key_sets[local_date] = set(
            zip(group["local_hour"].astype(int), group["fold"].astype(int))
        )

    values = pd.Series(np.nan, index=output_index, dtype=float)
    missing_flag = pd.Series(1, index=output_index, dtype="uint8")
    shape_mismatch_flag = pd.Series(0, index=output_index, dtype="uint8")

    for target_day, target_group in target_meta.groupby("local_date", sort=False):
        source_day = target_day - timedelta(days=lag_days)
        target_keys = set(
            zip(
                target_group["local_hour"].astype(int),
                target_group["fold"].astype(int),
            )
        )
        if target_keys != source_key_sets.get(source_day, set()):
            shape_mismatch_flag.loc[target_group.index] = np.uint8(1)

        for timestamp, row in target_group.iterrows():
            key = (source_day, int(row["local_hour"]), int(row["fold"]))
            value = source_lookup.get(key, np.nan)
            if np.isfinite(value):
                values.at[timestamp] = value
                missing_flag.at[timestamp] = np.uint8(0)

    values.index.name = "delivery_start_utc"
    missing_flag.index.name = "delivery_start_utc"
    shape_mismatch_flag.index.name = "delivery_start_utc"
    return values, missing_flag, shape_mismatch_flag


def build_features(
    raw: pd.DataFrame,
    *,
    start_day: str,
    end_day: str,
) -> pd.DataFrame:
    """Build lag-only, explicitly non-PIT Eco2Mix feature profiles."""

    hourly = _hourly(raw)
    index = _expected_index(start_day, end_day)
    result = pd.DataFrame(index=index)
    value_columns: list[str] = []
    shape_flags: dict[int, pd.Series] = {}

    for source, alias in SOURCE_FIELD_ALIASES.items():
        for lag in PROFILE_LAGS_DAYS:
            feature = f"eco2mix_{alias}_profile_d{lag}"
            values, missing, shape_mismatch = _local_day_profile(
                hourly,
                source_column=source,
                output_index=index,
                lag_days=lag,
            )
            result[feature] = values
            result[f"{feature}_missing_flag"] = missing
            value_columns.append(feature)
            shape_flags[lag] = shape_mismatch

    for lag in PROFILE_LAGS_DAYS:
        derived = {
            f"eco2mix_renewable_profile_d{lag}": (
                result[f"eco2mix_eolien_profile_d{lag}"]
                + result[f"eco2mix_solaire_profile_d{lag}"]
            ),
            f"eco2mix_net_load_profile_d{lag}": (
                result[f"eco2mix_consommation_profile_d{lag}"]
                - result[f"eco2mix_eolien_profile_d{lag}"]
                - result[f"eco2mix_solaire_profile_d{lag}"]
            ),
            f"eco2mix_dispatchable_profile_d{lag}": (
                result[f"eco2mix_nucleaire_profile_d{lag}"]
                + result[f"eco2mix_hydraulique_profile_d{lag}"]
            ),
        }
        for feature, values in derived.items():
            result[feature] = values
            result[f"{feature}_missing_flag"] = values.isna().astype("uint8")
            value_columns.append(feature)
        result[f"eco2mix_day_shape_mismatch_d{lag}_flag"] = shape_flags[lag]

    result["eco2mix_feature_missing_count"] = (
        result[value_columns].isna().sum(axis=1).astype("int16")
    )
    result["eco2mix_any_missing_flag"] = (
        result["eco2mix_feature_missing_count"].gt(0).astype("uint8")
    )
    result["eco2mix_source_revision_latest_flag"] = np.uint8(1)
    result["eco2mix_exploratory_non_pit_flag"] = np.uint8(1)
    result.attrs.update(
        {
            "data_classification": CLASSIFICATION,
            "source_revision_policy": "revision_latest",
            "profile_lags_days": list(PROFILE_LAGS_DAYS),
            "same_day_prevision_j1_exported": False,
        }
    )
    return result


def _write_atomic_parquet(frame: pd.DataFrame, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        frame.reset_index().to_parquet(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-day", required=True)
    parser.add_argument("--end-day", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--fetch-chunk-days", type=int, default=7)
    parser.add_argument(
        "--transport",
        choices=("csv", "records"),
        default="csv",
        help="CSV export avoids the 100-row pagination of the records API.",
    )
    parser.add_argument(
        "--max-output-days",
        type=int,
        default=93,
        help="Safety cap; increase explicitly when materialising in batches.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    start, end = _validate_days(args.start_day, args.end_day)
    output_days = int((end - start).days) + 1
    if output_days > args.max_output_days:
        parser.error(
            f"requested {output_days} output days exceeds --max-output-days "
            f"{args.max_output_days}"
        )
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; use --overwrite")
    if manifest_path.exists() and not args.overwrite:
        parser.error(f"manifest already exists: {manifest_path}; use --overwrite")

    ca_variable, _ = _configured_ca_bundle()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    raw = _fetch(
        args.start_day,
        args.end_day,
        page_size=args.page_size,
        chunk_days=args.fetch_chunk_days,
        transport=args.transport,
    )
    features = build_features(raw, start_day=args.start_day, end_day=args.end_day)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic_parquet(features, args.output)

    query_start, query_end = _source_window_utc(args.start_day, args.end_day)
    manifest = {
        "source": f"ODRE/RTE {DATASET_ID}",
        "api": API_URL,
        "retrieved_at_utc": retrieved_at,
        "requests_ca_environment": ca_variable or "system_default",
        "requested_delivery_days": {
            "start": args.start_day,
            "end": args.end_day,
            "count": output_days,
        },
        "source_query_utc": {
            "start": _format_api_timestamp(query_start),
            "end_exclusive": _format_api_timestamp(query_end),
            "raw_rows": int(len(raw)),
            "identical_duplicate_rows_removed": int(
                raw.attrs.get("identical_duplicate_rows_removed", 0)
            ),
            "conflicting_duplicate_timestamps_quarantined": int(
                raw.attrs.get(
                    "conflicting_duplicate_timestamps_quarantined", 0
                )
            ),
            "conflicting_duplicate_rows_quarantined": int(
                raw.attrs.get("conflicting_duplicate_rows_quarantined", 0)
            ),
            "transport": args.transport,
        },
        "hours": int(len(features)),
        "columns": list(features.columns),
        "coverage": {
            column: float(features[column].notna().mean())
            for column in features
        },
        "causality": {
            "classification": CLASSIFICATION,
            "strict_pit_eligible": False,
            "source_revision_policy": "revision_latest",
            "publication_timestamp_available": False,
            "same_day_prevision_j1_exported": False,
            "allowed_profile_lags_days": list(PROFILE_LAGS_DAYS),
            "dst_policy": (
                "exact local civil day/hour/fold mapping; no interpolation; "
                "NaN plus flags when no exact match exists"
            ),
            "reason": (
                "The consolidated dataset is not a vintage archive; all "
                "features remain exploratory even after D-2/D-7 shifting."
            ),
            "target_or_price_used": False,
            "production_pipeline_modified": False,
        },
        "license": {
            "name": "Licence Ouverte 2.0",
            "dataset": (
                "https://odre.opendatasoft.com/explore/dataset/"
                f"{DATASET_ID}/"
            ),
        },
        "output_sha256": _sha256(args.output),
    }
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
