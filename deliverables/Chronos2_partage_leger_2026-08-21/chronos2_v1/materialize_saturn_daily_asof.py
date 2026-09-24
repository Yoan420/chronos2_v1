#!/usr/bin/env python
"""Materialise causal day-ahead Saturn curves at one cutoff per delivery day.

This is a compact alternative to downloading every historical revision.  For
each Europe/Paris delivery day D it asks Saturn for the state of the series at
D-1 08:00 local, then stores only D's physical 23/24/25 hours.  The resulting
Parquet follows the project's PIT schema; ``snapshot_time_utc`` and
``revision_time_utc`` denote the explicit as-of cutoff used for the query.

No realised value and no external price forecast is read by this utility.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    parser.add_argument("--allow-incomplete-days", action="store_true")
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
    return parser.parse_args()


def _client(args: argparse.Namespace):
    client = getattr(LOCAL, "client", None)
    if client is None:
        client = create_saturn_client(args.saturn_url, args.author)
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


def _one_day(day: pd.Timestamp, args: argparse.Namespace) -> pd.DataFrame:
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
                local_start - pd.Timedelta(hours=8),
                local_end + pd.Timedelta(hours=8),
                args.timezone,
                revision_date=cutoff_utc,
                naive_timezone=args.naive_timezone,
            )
            selected = series.loc[(series.index >= local_start) & (series.index < local_end)]
            if args.hourly_on_the_hour:
                selected = selected.loc[
                    (selected.index.minute == 0)
                    & (selected.index.second == 0)
                    & (selected.index.microsecond == 0)
                ]
            expected = int((local_end.tz_convert("UTC") - local_start.tz_convert("UTC")) / pd.Timedelta(hours=1))
            if not args.allow_incomplete_days and len(selected) != expected:
                raise RuntimeError(
                    f"{day.date()}: {len(selected)} heures, attendu {expected}"
                )
            return pd.DataFrame(
                {
                    "value_time_utc": selected.index.tz_convert("UTC"),
                    "snapshot_time_utc": cutoff_utc,
                    "revision_time_utc": cutoff_utc,
                    "value": selected.to_numpy(dtype=float),
                    "downloaded_at_utc": pd.Timestamp.now(tz="UTC"),
                }
            )
        except Exception as exc:  # network retry with the exact same cutoff
            last_error = exc
            if attempt < int(args.retries):
                time.sleep(float(2 ** (attempt - 1)))
                LOCAL.client = None
    raise RuntimeError(f"{day.date()}: Saturn as-of failed: {last_error}") from last_error


def main() -> int:
    args = parse_args()
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
    merged = normalize_vintage_frame(
        pd.concat([frames[day] for day in sorted(frames)], ignore_index=True)
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.parquet")
    merged.loc[:, list(VINTAGE_COLUMNS)].to_parquet(temporary, index=False)
    pd.read_parquet(temporary, columns=["value_time_utc"]).head(1)
    temporary.replace(output)
    print(
        f"{output} | rows={len(merged)} | days={len(days)} | "
        f"delivery={merged['value_time_utc'].min()}..{merged['value_time_utc'].max()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
