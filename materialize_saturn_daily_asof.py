#!/usr/bin/env python
"""Materialise causal day-ahead Saturn curves at one cutoff per delivery day.

This is a compact alternative to downloading every historical revision.  For
each Europe/Paris delivery day D it asks Saturn for the state of the series at
D-1 08:00 local, then stores only D's physical 23/24/25 hours.  The resulting
Parquet follows the project's PIT schema; ``snapshot_time_utc`` and
``revision_time_utc`` denote the explicit as-of cutoff used for the query,
not Saturn's provider-side insertion timestamp.  This distinction is recorded
explicitly in the sidecar audit.

No realised value and no external price forecast is read by this utility.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
import uuid

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronos2_modular.saturn import (  # noqa: E402
    VINTAGE_COLUMNS,
    create_saturn_client,
    fetch_saturn_series_from_client,
    normalize_vintage_frame,
)


LOCAL = threading.local()

NL_SOLAR_ZERO_DUPLICATION_CONTRACT = {
    "alias": "nl_solar_generation_fcst",
    "series": "power.nl.generation.solar.hourly.gw.fcst",
    "timezone": "Europe/Amsterdam",
    "cutoff_timezone": "Europe/Amsterdam",
    "naive_timezone": "Europe/Amsterdam",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--start-day", required=True)
    parser.add_argument("--end-day", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument(
        "--cutoff-timezone",
        default=None,
        help=(
            "Fuseau civil de l'origine D-1; par defaut identique a --timezone. "
            "Permet de separer le fuseau de livraison du fuseau d'emission."
        ),
    )
    parser.add_argument("--cutoff-time", default="08:00")
    parser.add_argument("--naive-timezone", default="UTC")
    parser.add_argument("--saturn-url", default="https://saturn-energyscan.gem.myengie.com//api")
    parser.add_argument("--author", default="BQ6757")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--value-scale",
        type=float,
        default=1.0,
        help="Facteur multiplicatif deterministe applique aux valeurs (defaut 1).",
    )
    parser.add_argument("--allow-incomplete-days", action="store_true")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help=(
            "Fusionne le nouveau segment avec le Parquet existant, en "
            "dedupliquant le contrat PIT. Utile pour une mise a jour incrementale."
        ),
    )
    parser.add_argument(
        "--incomplete-dst-policy",
        choices=("raise", "duplicate", "duplicate_zero_only"),
        default="raise",
        help=(
            "Politique appliquee par le normaliseur Saturn aux series civiles "
            "sans second fold d'automne. 'duplicate' est reserve aux "
            "covariables connues dans le futur, jamais a la cible; "
            "'duplicate_zero_only' refuse tout singleton non nul/non fini."
        ),
    )
    parser.add_argument(
        "--daily-broadcast",
        action="store_true",
        help=(
            "Exige exactement une valeur quotidienne finie pour D et la "
            "repete sur les 23/24/25 heures physiques de livraison."
        ),
    )
    parser.add_argument(
        "--hourly-on-the-hour",
        action="store_true",
        help=(
            "Pour une source infra-horaire, conserve explicitement les "
            "observations HH:00 avant le controle 23/24/25 heures. Cette "
            "option reproduit la reindexation horaire exacte du pipeline et "
            "n'effectue aucune moyenne ni interpolation."
        ),
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--request-padding-hours",
        type=int,
        default=8,
        help=(
            "Marge symetrique demandee a Saturn autour du jour civil. "
            "La sortie reste strictement reindexee sur les 23/24/25 heures "
            "physiques de livraison (defaut: 8)."
        ),
    )
    return parser.parse_args()


def _client(args: argparse.Namespace):
    client = getattr(LOCAL, "client", None)
    if client is None:
        client = create_saturn_client(args.saturn_url, args.author)
        client.session.request = partial(
            client.session.request,
            timeout=float(args.request_timeout_seconds),
        )
        LOCAL.client = client
    return client


def _civil_cutoff(
    day: pd.Timestamp,
    *,
    timezone: str,
    cutoff_time: str,
) -> pd.Timestamp:
    cutoff_clock = pd.Timedelta(
        cutoff_time + ":00" if cutoff_time.count(":") == 1 else cutoff_time
    )
    return (day - pd.Timedelta(days=1) + cutoff_clock).tz_localize(timezone)


def _physical_utc_index(
    day: pd.Timestamp,
    *,
    timezone: str,
) -> pd.DatetimeIndex:
    local_start = day.tz_localize(timezone)
    local_end = (day + pd.Timedelta(days=1)).tz_localize(timezone)
    return pd.date_range(
        local_start.tz_convert("UTC"),
        local_end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )


def _validate_dst_repair_scope(args: argparse.Namespace) -> None:
    """Keep every synthetic DST fold inside its explicitly audited scope."""

    policy = str(args.incomplete_dst_policy).strip().casefold()
    alias = str(getattr(args, "alias", "")).strip().casefold()
    series = str(getattr(args, "series", "")).strip().casefold()
    if policy != "raise" and (
        alias == "target"
        or alias.startswith("target_")
        or ".price.da." in series
    ):
        raise ValueError(
            "La duplication DST est interdite pour une cible/prix day-ahead."
        )
    if policy != "duplicate_zero_only":
        return

    actual = {
        "alias": str(getattr(args, "alias", "")),
        "series": str(getattr(args, "series", "")),
        "timezone": str(getattr(args, "timezone", "")),
        "cutoff_timezone": str(
            getattr(args, "cutoff_timezone", None)
            or getattr(args, "timezone", "")
        ),
        "naive_timezone": str(getattr(args, "naive_timezone", "")),
    }
    mismatches = {
        key: (actual[key], expected)
        for key, expected in NL_SOLAR_ZERO_DUPLICATION_CONTRACT.items()
        if actual[key] != expected
    }
    daily_broadcast = bool(getattr(args, "daily_broadcast", False))
    if mismatches or daily_broadcast:
        details = ", ".join(
            f"{key}={value!r} (attendu={expected!r})"
            for key, (value, expected) in sorted(mismatches.items())
        )
        if daily_broadcast:
            details = ", ".join(
                item
                for item in (details, "daily_broadcast=True (attendu=False)")
                if item
            )
        raise ValueError(
            "incomplete_dst_policy=duplicate_zero_only est reserve au "
            f"solaire NL local-civil audite; {details}."
        )


def _select_physical_delivery_day(
    series: pd.Series,
    *,
    day: pd.Timestamp,
    timezone: str,
    allow_incomplete: bool = False,
) -> pd.Series:
    """Select exactly one civil delivery day's physical UTC products.

    The surrounding Saturn request is deliberately wider than one day.  This
    function is the fail-closed boundary: margins may supply boundary points,
    but no timestamp is shifted, duplicated or interpolated to manufacture a
    missing physical product.
    """

    index = pd.DatetimeIndex(series.index)
    if index.tz is None:
        raise RuntimeError(
            f"{day.date()}: index Saturn non localise apres normalisation"
        )
    physical = series.copy()
    physical.index = index.tz_convert("UTC")
    physical = physical.sort_index()
    if physical.index.has_duplicates:
        duplicates = physical.index[physical.index.duplicated()].unique()
        raise RuntimeError(
            f"{day.date()}: produits physiques UTC dupliques: "
            f"{[str(value) for value in duplicates[:4]]}"
        )

    expected = _physical_utc_index(day, timezone=timezone)
    selected = physical.reindex(expected)
    numeric = pd.to_numeric(selected, errors="coerce")
    finite = numeric.map(lambda value: math.isfinite(float(value)))
    if not bool(finite.all()):
        missing = expected[~finite.to_numpy()]
        if not allow_incomplete:
            raise RuntimeError(
                f"{day.date()}: {len(missing)} heure(s) physique(s) "
                "absente(s) ou non finie(s): "
                f"{[str(value) for value in missing[:6]]}"
            )
        selected = selected.loc[finite]
    return selected.astype(float)


def _broadcast_daily_value(
    selected: pd.Series,
    *,
    day: pd.Timestamp,
    timezone: str,
) -> pd.Series:
    """Broadcast one causal daily forecast over D's physical UTC hours."""

    numeric = pd.to_numeric(selected, errors="coerce")
    if len(numeric) != 1 or not math.isfinite(float(numeric.iloc[0])):
        raise RuntimeError(
            f"{day.date()}: daily-broadcast exige exactement une valeur finie; "
            f"obtenu={len(numeric)}"
        )
    return pd.Series(
        float(numeric.iloc[0]),
        index=_physical_utc_index(day, timezone=timezone),
        dtype=float,
        name=selected.name,
    )


def _one_day(day: pd.Timestamp, args: argparse.Namespace) -> pd.DataFrame:
    _validate_dst_repair_scope(args)
    local_start = day.tz_localize(args.timezone)
    local_end = (day + pd.Timedelta(days=1)).tz_localize(args.timezone)
    # Build D-1 08:00 as a civil timestamp before localisation. Adding eight
    # absolute hours after localising midnight shifts the clock on the day of
    # a DST transition (07:00 after the autumn switch, 09:00 after spring).
    cutoff_local = _civil_cutoff(
        day,
        timezone=args.cutoff_timezone or args.timezone,
        cutoff_time=args.cutoff_time,
    )
    cutoff_utc = cutoff_local.tz_convert("UTC")
    last_error: Exception | None = None
    for attempt in range(1, int(args.retries) + 1):
        try:
            # Wide UTC/local margins avoid losing the first/last physical hour
            # when a Saturn source is UTC-naive.
            series = fetch_saturn_series_from_client(
                _client(args),
                args.series,
                local_start,
                local_end,
                args.timezone,
                revision_date=cutoff_utc,
                naive_timezone=args.naive_timezone,
                incomplete_dst_policy=args.incomplete_dst_policy,
                request_padding_hours=args.request_padding_hours,
            )
            dst_repairs = list(series.attrs.get("dst_repairs", []))
            if args.daily_broadcast:
                selected = series.loc[
                    (series.index >= local_start) & (series.index < local_end)
                ]
                selected = _broadcast_daily_value(
                    selected,
                    day=day,
                    timezone=args.timezone,
                )
            else:
                selected = series
            if not math.isfinite(float(args.value_scale)):
                raise RuntimeError("value-scale doit etre fini")
            if args.hourly_on_the_hour:
                selected = selected.loc[
                    (selected.index.minute == 0)
                    & (selected.index.second == 0)
                    & (selected.index.microsecond == 0)
                ]
            selected = _select_physical_delivery_day(
                selected,
                day=day,
                timezone=args.timezone,
                allow_incomplete=args.allow_incomplete_days,
            )
            selected = selected * float(args.value_scale)
            frame = pd.DataFrame(
                {
                    "value_time_utc": selected.index.tz_convert("UTC"),
                    "snapshot_time_utc": cutoff_utc,
                    "revision_time_utc": cutoff_utc,
                    "value": selected.to_numpy(dtype=float),
                    "downloaded_at_utc": pd.Timestamp.now(tz="UTC"),
                }
            )
            frame.attrs["dst_repairs"] = dst_repairs
            return frame
        except Exception as exc:  # network retry with the exact same cutoff
            last_error = exc
            if attempt < int(args.retries):
                time.sleep(float(2 ** (attempt - 1)))
                LOCAL.client = None
    raise RuntimeError(f"{day.date()}: Saturn as-of failed: {last_error}") from last_error


def main() -> int:
    args = parse_args()
    _validate_dst_repair_scope(args)
    if (
        not math.isfinite(args.request_timeout_seconds)
        or args.request_timeout_seconds <= 0
    ):
        raise ValueError("request-timeout-seconds doit etre strictement positif.")
    if not 0 <= int(args.request_padding_hours) <= 48:
        raise ValueError("request-padding-hours doit etre compris entre 0 et 48.")
    start = pd.Timestamp(args.start_day).normalize()
    end = pd.Timestamp(args.end_day).normalize()
    if start.tzinfo is not None or end.tzinfo is not None:
        raise ValueError("start-day/end-day must be naive local calendar dates")
    days = list(pd.date_range(start, end, freq="D"))
    frames: dict[pd.Timestamp, pd.DataFrame] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_to_day = {pool.submit(_one_day, day, args): day for day in days}
        for number, future in enumerate(as_completed(future_to_day), start=1):
            day = future_to_day[future]
            try:
                frames[day] = future.result()
            except Exception as exc:
                failures.append(str(exc))
            if number == 1 or number % 25 == 0 or number == len(days):
                print(
                    f"[{args.alias}] {number}/{len(days)} jours | "
                    f"ok={len(frames)} fail={len(failures)}",
                    flush=True,
                )
    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(f"{len(failures)} jour(s) en echec:\n{preview}")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    segments = [frames[day] for day in sorted(frames)]
    dst_repairs = (
        [
            dict(repair)
            for day in sorted(frames)
            for repair in frames[day].attrs.get("dst_repairs", [])
        ]
        if args.incomplete_dst_policy == "duplicate_zero_only"
        else []
    )
    if args.merge_existing and output.is_file():
        segments.append(pd.read_parquet(output))
        previous_audit = output.with_name(output.name + ".audit.json")
        if previous_audit.is_file():
            try:
                previous_payload = json.loads(
                    previous_audit.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                previous_payload = {}
            previous_repairs = (
                previous_payload.get("dst_zero_duplicate_repairs", [])
                if args.incomplete_dst_policy == "duplicate_zero_only"
                else []
            )
            if isinstance(previous_repairs, list):
                dst_repairs.extend(
                    dict(repair)
                    for repair in previous_repairs
                    if isinstance(repair, dict)
                )
    repairs_by_identity = {
        json.dumps(repair, sort_keys=True, allow_nan=False): repair
        for repair in dst_repairs
    }
    dst_repairs = [repairs_by_identity[key] for key in sorted(repairs_by_identity)]
    merged = normalize_vintage_frame(pd.concat(segments, ignore_index=True))
    delivery_local_days = pd.DatetimeIndex(
        pd.to_datetime(merged["value_time_utc"], utc=True)
    ).tz_convert(args.timezone).normalize()
    materialized_start = delivery_local_days.min().date().isoformat()
    materialized_end = delivery_local_days.max().date().isoformat()
    materialized_days = int(delivery_local_days.nunique())
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.parquet")
    merged.loc[:, list(VINTAGE_COLUMNS)].to_parquet(temporary, index=False)
    pd.read_parquet(temporary, columns=["value_time_utc"]).head(1)
    temporary.replace(output)
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    audit_path = output.with_name(output.name + ".audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "series": args.series,
                "alias": args.alias,
                "timezone": args.timezone,
                "cutoff_timezone": args.cutoff_timezone or args.timezone,
                "cutoff_time": args.cutoff_time,
                "naive_timezone": args.naive_timezone,
                "incomplete_dst_policy": args.incomplete_dst_policy,
                "daily_broadcast": bool(args.daily_broadcast),
                "value_scale": float(args.value_scale),
                "request_padding_hours": int(args.request_padding_hours),
                "dst_zero_duplicate_repair_count": len(dst_repairs),
                "dst_zero_duplicate_repairs": dst_repairs,
                # For an incremental merge these fields describe the complete
                # artifact, while requested_* records only this invocation.
                "start_day": materialized_start,
                "end_day": materialized_end,
                "days": materialized_days,
                "requested_start_day": start.date().isoformat(),
                "requested_end_day": end.date().isoformat(),
                "requested_days": len(days),
                "rows": len(merged),
                "first_delivery_utc": merged["value_time_utc"].min().isoformat(),
                "last_delivery_utc": merged["value_time_utc"].max().isoformat(),
                "sha256": digest.hexdigest(),
                "causal_contract": "Saturn state queried as-of D-1 civil cutoff",
                "snapshot_time_semantics": "query_asof_cutoff",
                "revision_time_semantics": (
                    "query_asof_cutoff; provider insertion timestamp is not "
                    "returned by Client.get(revision_date=...)"
                ),
                "provider_revision_timestamp_available": False,
                "fill_or_interpolation": (
                    "daily_value_broadcast_to_physical_hours"
                    if args.daily_broadcast
                    else (
                        "none_except_duplicate_missing_autumn_fold_if_source_"
                        "is_civil_naive"
                        if args.incomplete_dst_policy == "duplicate"
                        else (
                            "none_except_verified_zero_duplicate_for_singleton_"
                            "autumn_fold"
                            if args.incomplete_dst_policy
                            == "duplicate_zero_only"
                            else "none"
                        )
                    )
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{output} | rows={len(merged)} | days={materialized_days} | "
        f"delivery={merged['value_time_utc'].min()}..{merged['value_time_utc'].max()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_broadcast_daily_value",
    "_civil_cutoff",
    "_physical_utc_index",
    "_select_physical_delivery_day",
    "_validate_dst_repair_scope",
    "NL_SOLAR_ZERO_DUPLICATION_CONTRACT",
    "main",
    "parse_args",
]
