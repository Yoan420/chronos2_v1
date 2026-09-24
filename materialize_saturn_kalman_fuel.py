#!/usr/bin/env python
"""Materialise causal residual-load and TTF/EUA Kalman inputs.

For every delivery day ``D``, the utility queries the historical Saturn state
at the civil cutoff ``D-1 08:00 Europe/Paris``.  It retains the latest finite
daily market observation whose value timestamp is not later than that cutoff,
derives slow fuel features on the daily timeline, and broadcasts the result to
the physical 23/24/25 hours of ``D``.

The utility also queries the five national residual-load forecasts independently
at the same day-ahead cutoff and requires their exact 23/24/25-hour physical
grids.  It writes two common European PIT sidecars; neither contains a realised
power price, Storm forecast or MKOnline forecast.  Incremental updates validate
each complete existing artifact and its audit before appending only an exact
missing suffix.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Sequence
import uuid

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronos2_modular.saturn import (  # noqa: E402
    coerce_saturn_to_series,
    create_saturn_client,
    fetch_saturn_series_from_client,
)


TIMEZONE = "Europe/Paris"
CUTOFF_TIME = "08:00"
OUTPUT_NAME = "market_fuel_features.parquet"
RESIDUAL_LOAD_OUTPUT_NAME = "residual_load_market_features.parquet"
RESIDUAL_LOAD_SCHEMA_VERSION = 2
AUDIT_SUFFIX = ".audit.json"
WARMUP_DAYS = 20
MAX_TOTAL_WORKERS = 32
SATURN_URL = "https://saturn-energyscan.gem.myengie.com//api"
SATURN_AUTHOR = "BQ6757"
VINTAGE_ROOT = ROOT / "data" / "pit" / "vintages"

RESIDUAL_LOAD_ALIASES: tuple[str, ...] = (
    "fr_residual_load_fcst",
    "de_residual_load_fcst",
    "be_residual_load_fcst",
    "nl_residual_load_fcst",
    "es_residual_load_fcst",
)
RESIDUAL_LOAD_RAW_COLUMNS: tuple[str, ...] = (
    "value_time_utc",
    "snapshot_time_utc",
    "revision_time_utc",
    "value",
    "downloaded_at_utc",
)
RESIDUAL_LOAD_OUTPUT_COLUMNS: tuple[str, ...] = (
    "value_time_utc",
    "snapshot_time_utc",
    "revision_time_utc",
    "cutoff_time_utc",
    *RESIDUAL_LOAD_ALIASES,
)
NL_SPRING_DST_REPAIR_ALIAS = "nl_residual_load_fcst"
NL_SPRING_DST_REPAIR_LOCAL_TIMES = ("04:00", "06:00")
NL_SPRING_DST_REPAIR_POLICY = (
    "nl_residual_load_fcst uniquement: grille contractuellement naive UTC "
    "d'un jour de passage a l'heure d'ete; si 04:00 et 06:00 locales sont "
    "exactement les deux seules heures physiques manquantes, moyenne lineaire "
    "des voisins UTC immediats issus du meme etat Saturn as-of; toute autre "
    "signature echoue"
)
SATURN_RESIDUAL_FILL_POLICY = (
    "none_except_duplicate_missing_autumn_fold_if_source_is_civil_naive_and_"
    "the_strictly_audited_nl_spring_04_and_06_linear_repairs; exact physical-hour "
    "coverage required"
)


@dataclass(frozen=True)
class ResidualLoadSeries:
    alias: str
    zone: str
    series: str
    delivery_timezone: str
    naive_timezone: str
    incomplete_dst_policy: str = "duplicate"


RESIDUAL_LOAD_SERIES: tuple[ResidualLoadSeries, ...] = (
    ResidualLoadSeries(
        alias="fr_residual_load_fcst",
        zone="FR",
        series="power.fr.residual.load.hourly.gw.fcst",
        delivery_timezone="Europe/Paris",
        naive_timezone="Europe/Paris",
    ),
    ResidualLoadSeries(
        alias="de_residual_load_fcst",
        zone="DE",
        series="power.de.residual.load.hourly.gw.fcst",
        delivery_timezone="Europe/Berlin",
        naive_timezone="Europe/Berlin",
    ),
    ResidualLoadSeries(
        alias="be_residual_load_fcst",
        zone="BE",
        series="power.be.residual.load.hourly.gw.fcst",
        delivery_timezone="Europe/Brussels",
        naive_timezone="Europe/Brussels",
    ),
    ResidualLoadSeries(
        alias="nl_residual_load_fcst",
        zone="NL",
        series="power.nl.residual.load.hourly.gw.fcst",
        delivery_timezone="Europe/Amsterdam",
        naive_timezone="UTC",
    ),
    ResidualLoadSeries(
        alias="es_residual_load_fcst",
        zone="ES",
        series="power.es.residual.load.hourly.gw.fcst",
        delivery_timezone="Europe/Madrid",
        naive_timezone="Europe/Madrid",
    ),
)


@dataclass(frozen=True)
class MarketSeries:
    alias: str
    series: str
    unit: str
    lookback_days: int = 120


MARKET_SERIES: tuple[MarketSeries, ...] = (
    MarketSeries(
        alias="ttf_m1_eur_mwh_th",
        series="gas.ttf.price.everyday.month.1.ice.eurmwh",
        unit="EUR/MWh_th",
    ),
    MarketSeries(
        alias="eua_first_dec_eur_tco2",
        series="carbon.eu.price.everyday.eua.ice.1st.dec",
        unit="EUR/tCO2",
    ),
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "ttf_m1_eur_mwh_th",
    "eua_first_dec_eur_tco2",
    "ttf_change_1d",
    "ttf_change_5d",
    "eua_change_1d",
    "eua_change_5d",
    "fuel_volatility_20d",
    "ccgt_marginal_cost_eur_mwh",
)
SOURCE_TIME_COLUMNS: tuple[str, ...] = (
    "ttf_source_value_time_utc",
    "eua_source_value_time_utc",
    "market_source_value_time_utc",
)
OUTPUT_COLUMNS: tuple[str, ...] = (
    "value_time_utc",
    "snapshot_time_utc",
    "revision_time_utc",
    *SOURCE_TIME_COLUMNS,
    *FEATURE_COLUMNS,
)


LOCAL = threading.local()


class FuelMaterializationError(RuntimeError):
    """Raised when the causal or immutable artifact contract is violated."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialise TTF/EUA et leurs features causales au cutoff "
            "D-1 08:00 pour les challengers Kalman."
        )
    )
    parser.add_argument("--start-day", required=True)
    parser.add_argument(
        "--residual-start-day",
        default=None,
        help=(
            "Premier jour du sidecar charge residuelle. Par defaut, reprend "
            "--start-day; permet de conserver un historique fuel plus large."
        ),
    )
    parser.add_argument("--end-day", required=True)
    parser.add_argument(
        "--output-dir",
        default="data/pit/kalman_hybrid",
    )
    parser.add_argument("--series-workers", type=int, default=2)
    parser.add_argument("--day-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--ccgt-efficiency", type=float, default=0.58)
    parser.add_argument(
        "--ccgt-emission-tco2-mwh",
        type=float,
        default=0.36,
    )
    parser.add_argument("--ccgt-vom-eur-mwh", type=float, default=3.0)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Reconstruit l'artefact au lieu de reutiliser son prefixe valide.",
    )
    parser.add_argument(
        "--skip-residual-load",
        action="store_true",
        help=(
            "Ne materialise pas le sidecar PIT commun des cinq forecasts de "
            "charge residuelle (reserve aux tests fuel isoles)."
        ),
    )
    parser.add_argument(
        "--residual-load-source",
        choices=("saturn", "local-vintages"),
        default="saturn",
        help=(
            "Source du sidecar charge residuelle: Saturn as-of (operationnel, "
            "defaut) ou caches de vintages locaux (diagnostic strict)."
        ),
    )
    args = parser.parse_args(argv)
    if args.residual_start_day is None:
        args.residual_start_day = args.start_day
    return args


def _normalise_day(value: str | pd.Timestamp, *, name: str) -> pd.Timestamp:
    try:
        day = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} invalide: {value!r}.") from exc
    if day.tzinfo is not None:
        raise ValueError(f"{name} doit etre une date civile sans fuseau.")
    return day


def _validate_parameters(args: argparse.Namespace) -> None:
    if args.series_workers < 1 or args.day_workers < 1:
        raise ValueError("series-workers et day-workers doivent etre >= 1.")
    if args.series_workers * args.day_workers > MAX_TOTAL_WORKERS:
        raise ValueError(
            "Parallelisme refuse: series-workers * day-workers doit etre "
            f"<= {MAX_TOTAL_WORKERS}."
        )
    if args.retries < 1:
        raise ValueError("retries doit etre >= 1.")
    if (
        not math.isfinite(float(args.request_timeout_seconds))
        or float(args.request_timeout_seconds) <= 0.0
    ):
        raise ValueError("request-timeout-seconds doit etre strictement positif.")
    if (
        not math.isfinite(float(args.ccgt_efficiency))
        or not 0.0 < float(args.ccgt_efficiency) <= 1.0
    ):
        raise ValueError("ccgt-efficiency doit appartenir a ]0, 1].")
    for name in ("ccgt_emission_tco2_mwh", "ccgt_vom_eur_mwh"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} doit etre fini et positif ou nul.")


def _civil_cutoff(day: pd.Timestamp) -> pd.Timestamp:
    """Return D-1 08:00 as civil time, including exact DST transitions."""

    naive = day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    return naive.tz_localize(TIMEZONE)


def _physical_utc_index(day: pd.Timestamp) -> pd.DatetimeIndex:
    local_start = day.tz_localize(TIMEZONE)
    local_end = (day + pd.Timedelta(days=1)).tz_localize(TIMEZONE)
    return pd.date_range(
        local_start.tz_convert("UTC"),
        local_end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )


def _expected_hourly_index(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> pd.DatetimeIndex:
    start = start_day.tz_localize(TIMEZONE).tz_convert("UTC")
    end = (end_day + pd.Timedelta(days=1)).tz_localize(TIMEZONE).tz_convert(
        "UTC"
    )
    return pd.date_range(start, end, freq="h", inclusive="left")


def _client(timeout_seconds: float):
    client = getattr(LOCAL, "client", None)
    configured_timeout = getattr(LOCAL, "timeout_seconds", None)
    if client is None or configured_timeout != float(timeout_seconds):
        client = create_saturn_client(SATURN_URL, SATURN_AUTHOR)
        client.session.request = partial(
            client.session.request,
            timeout=float(timeout_seconds),
        )
        LOCAL.client = client
        LOCAL.timeout_seconds = float(timeout_seconds)
    return client


def _normalise_market_series(raw: object, *, series: str) -> pd.Series:
    try:
        coerced = coerce_saturn_to_series(raw, series)
    except ValueError as exc:
        raise FuelMaterializationError(str(exc)) from exc
    numeric = pd.to_numeric(coerced, errors="coerce")
    parsed = pd.DatetimeIndex(pd.to_datetime(numeric.index, errors="coerce"))
    valid_time = ~parsed.isna()
    numeric = numeric.iloc[np.flatnonzero(valid_time)].astype(float)
    parsed = parsed[valid_time]
    if parsed.tz is None:
        parsed = parsed.tz_localize(
            TIMEZONE,
            ambiguous="raise",
            nonexistent="raise",
        )
    else:
        parsed = parsed.tz_convert(TIMEZONE)
    numeric.index = parsed
    if numeric.index.has_duplicates:
        raise FuelMaterializationError(
            f"{series}: timestamps de valeur dupliques."
        )
    return numeric.replace([np.inf, -np.inf], np.nan).dropna().sort_index()


def _query_market_value(
    client: Any,
    spec: MarketSeries,
    day: pd.Timestamp,
) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    cutoff_local = _civil_cutoff(day)
    cutoff_utc = cutoff_local.tz_convert("UTC")
    raw = client.get(
        spec.series,
        from_value_date=cutoff_local - pd.Timedelta(days=spec.lookback_days),
        to_value_date=cutoff_local,
        revision_date=cutoff_utc,
    )
    values = _normalise_market_series(raw, series=spec.series)
    selected = values.loc[values.index <= cutoff_local]
    if selected.empty:
        raise FuelMaterializationError(
            f"{spec.alias}: aucune valeur finie connue au cutoff {cutoff_local}."
        )
    source_time = pd.Timestamp(selected.index[-1]).tz_convert("UTC")
    value = float(selected.iloc[-1])
    if not math.isfinite(value):  # pragma: no cover - normalized above.
        raise FuelMaterializationError(f"{spec.alias}: valeur non finie.")
    if source_time > cutoff_utc:  # fail closed even after the explicit filter.
        raise FuelMaterializationError(
            f"{spec.alias}: valeur source posterieure au cutoff."
        )
    return value, source_time, cutoff_utc


def _query_one_day(
    spec: MarketSeries,
    day: pd.Timestamp,
    *,
    retries: int,
    timeout_seconds: float,
) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _query_market_value(_client(timeout_seconds), spec, day)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(float(2 ** (attempt - 1)))
                LOCAL.client = None
                LOCAL.timeout_seconds = None
    raise FuelMaterializationError(
        f"{day.date()} {spec.alias}: {last_error}"
    ) from last_error


def _fetch_series_days(
    spec: MarketSeries,
    days: Sequence[pd.Timestamp],
    *,
    day_workers: int,
    retries: int,
    timeout_seconds: float,
) -> pd.DataFrame:
    rows: dict[pd.Timestamp, tuple[float, pd.Timestamp, pd.Timestamp]] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=day_workers) as pool:
        jobs = {
            pool.submit(
                _query_one_day,
                spec,
                day,
                retries=retries,
                timeout_seconds=timeout_seconds,
            ): day
            for day in days
        }
        for number, future in enumerate(as_completed(jobs), start=1):
            day = jobs[future]
            try:
                rows[day] = future.result()
            except Exception as exc:
                failures.append(str(exc))
            if number == 1 or number % 50 == 0 or number == len(days):
                print(
                    f"[{spec.alias}] {number}/{len(days)} jours | "
                    f"ok={len(rows)} fail={len(failures)}",
                    flush=True,
                )
    if failures:
        raise FuelMaterializationError(
            f"{spec.alias}: {len(failures)} jour(s) en echec:\n"
            + "\n".join(failures[:20])
        )
    ordered = sorted(rows)
    return pd.DataFrame(
        {
            "delivery_day_local": ordered,
            spec.alias: [rows[day][0] for day in ordered],
            f"{spec.alias}__source_value_time_utc": [
                rows[day][1] for day in ordered
            ],
            f"{spec.alias}__cutoff_utc": [rows[day][2] for day in ordered],
        }
    ).set_index("delivery_day_local")


def _fetch_daily_market(
    days: Sequence[pd.Timestamp],
    *,
    series_workers: int,
    day_workers: int,
    retries: int,
    timeout_seconds: float,
) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(
        max_workers=min(series_workers, len(MARKET_SERIES))
    ) as pool:
        jobs = {
            pool.submit(
                _fetch_series_days,
                spec,
                days,
                day_workers=day_workers,
                retries=retries,
                timeout_seconds=timeout_seconds,
            ): spec
            for spec in MARKET_SERIES
        }
        for future in as_completed(jobs):
            spec = jobs[future]
            frames[spec.alias] = future.result()
    result: pd.DataFrame | None = None
    for spec in MARKET_SERIES:
        current = frames[spec.alias]
        result = current if result is None else result.join(
            current,
            how="inner",
            validate="one_to_one",
        )
    if result is None or len(result) != len(days):  # pragma: no cover.
        raise FuelMaterializationError("Jointure quotidienne TTF/EUA incomplete.")
    ttf_cutoff = pd.DatetimeIndex(
        pd.to_datetime(
            result["ttf_m1_eur_mwh_th__cutoff_utc"],
            utc=True,
            errors="raise",
        )
    )
    eua_cutoff = pd.DatetimeIndex(
        pd.to_datetime(
            result["eua_first_dec_eur_tco2__cutoff_utc"],
            utc=True,
            errors="raise",
        )
    )
    if not ttf_cutoff.equals(eua_cutoff):
        raise FuelMaterializationError("Cutoffs TTF/EUA incoherents.")
    result["cutoff_time_utc"] = ttf_cutoff
    return result.sort_index()


def _derive_daily_features(
    raw: pd.DataFrame,
    *,
    ccgt_efficiency: float,
    ccgt_emission_tco2_mwh: float,
    ccgt_vom_eur_mwh: float,
) -> pd.DataFrame:
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.has_duplicates:
        raise FuelMaterializationError(
            "La timeline quotidienne TTF/EUA doit etre unique."
        )
    output = pd.DataFrame(index=raw.index.copy())
    ttf = pd.to_numeric(raw["ttf_m1_eur_mwh_th"], errors="coerce").astype(float)
    eua = pd.to_numeric(
        raw["eua_first_dec_eur_tco2"], errors="coerce"
    ).astype(float)
    output["cutoff_time_utc"] = pd.to_datetime(
        raw["cutoff_time_utc"], utc=True, errors="raise"
    )
    output["ttf_source_value_time_utc"] = pd.to_datetime(
        raw["ttf_m1_eur_mwh_th__source_value_time_utc"],
        utc=True,
        errors="raise",
    )
    output["eua_source_value_time_utc"] = pd.to_datetime(
        raw["eua_first_dec_eur_tco2__source_value_time_utc"],
        utc=True,
        errors="raise",
    )
    output["market_source_value_time_utc"] = pd.concat(
        [
            output["ttf_source_value_time_utc"],
            output["eua_source_value_time_utc"],
        ],
        axis=1,
    ).max(axis=1)
    output["ttf_m1_eur_mwh_th"] = ttf
    output["eua_first_dec_eur_tco2"] = eua
    output["ttf_change_1d"] = ttf.diff(1)
    output["ttf_change_5d"] = ttf.diff(5)
    output["eua_change_1d"] = eua.diff(1)
    output["eua_change_5d"] = eua.diff(5)
    output["ccgt_marginal_cost_eur_mwh"] = (
        ttf / float(ccgt_efficiency)
        + eua * float(ccgt_emission_tco2_mwh)
        + float(ccgt_vom_eur_mwh)
    )
    output["fuel_volatility_20d"] = (
        output["ccgt_marginal_cost_eur_mwh"]
        .diff(1)
        .rolling(WARMUP_DAYS, min_periods=WARMUP_DAYS)
        .std(ddof=0)
    )
    return output


def _broadcast_daily_features(
    daily: pd.DataFrame,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> pd.DataFrame:
    selected = daily.loc[start_day:end_day]
    expected_days = pd.date_range(start_day, end_day, freq="D")
    if not selected.index.equals(expected_days):
        raise FuelMaterializationError(
            "Les features quotidiennes ne couvrent pas exactement la demande."
        )
    if not np.isfinite(
        selected.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all():
        counts = selected.loc[:, list(FEATURE_COLUMNS)].isna().sum()
        details = ", ".join(
            f"{column}={int(count)}"
            for column, count in counts.items()
            if count
        )
        raise FuelMaterializationError(
            "Warm-up insuffisant ou features fuel non finies: " + details
        )
    frames: list[pd.DataFrame] = []
    for day, row in selected.iterrows():
        index = _physical_utc_index(pd.Timestamp(day))
        block = pd.DataFrame(index=np.arange(len(index)))
        block["value_time_utc"] = index
        block["snapshot_time_utc"] = row["cutoff_time_utc"]
        block["revision_time_utc"] = row["cutoff_time_utc"]
        for column in SOURCE_TIME_COLUMNS:
            block[column] = row[column]
        for column in FEATURE_COLUMNS:
            block[column] = float(row[column])
        frames.append(block)
    result = pd.concat(frames, ignore_index=True)
    return result.loc[:, list(OUTPUT_COLUMNS)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cutoffs_for_value_times(
    value_time: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Map physical delivery hours to their exact civil D-1 08:00 cutoff."""

    if value_time.tz is None:
        raise FuelMaterializationError(
            "La timeline de livraison residuelle doit etre fusee."
        )
    local_days = value_time.tz_convert(TIMEZONE).normalize().tz_localize(None)
    unique_days, inverse = np.unique(local_days.to_numpy(), return_inverse=True)
    unique_cutoffs = pd.DatetimeIndex(
        [
            _civil_cutoff(pd.Timestamp(day)).tz_convert("UTC")
            for day in unique_days
        ]
    )
    return unique_cutoffs.take(inverse)


def _query_residual_saturn_day(
    spec: ResidualLoadSeries,
    day: pd.Timestamp,
    *,
    retries: int,
    timeout_seconds: float,
) -> pd.DataFrame:
    """Fetch one complete residual-load forecast day at its causal cutoff."""

    local_start = day.tz_localize(spec.delivery_timezone)
    local_end = (day + pd.Timedelta(days=1)).tz_localize(
        spec.delivery_timezone
    )
    cutoff_utc = _civil_cutoff(day).tz_convert("UTC")
    expected = pd.date_range(
        local_start.tz_convert("UTC"),
        local_end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    common_expected = _physical_utc_index(day)
    if not expected.equals(common_expected):
        raise FuelMaterializationError(
            f"{spec.alias}: calendrier physique incompatible avec {TIMEZONE}."
        )

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            series = fetch_saturn_series_from_client(
                _client(timeout_seconds),
                spec.series,
                local_start - pd.Timedelta(hours=8),
                local_end + pd.Timedelta(hours=8),
                spec.delivery_timezone,
                revision_date=cutoff_utc,
                naive_timezone=spec.naive_timezone,
                incomplete_dst_policy=spec.incomplete_dst_policy,
            )
            selected = series.loc[
                (series.index >= local_start) & (series.index < local_end)
            ].sort_index()
            selected_index = pd.DatetimeIndex(selected.index).tz_convert("UTC")
            repaired_utc: tuple[pd.Timestamp, ...] = ()
            if selected_index.has_duplicates or not selected_index.equals(expected):
                selected, repaired_utc = _repair_nl_spring_dst_hour(
                    selected,
                    spec=spec,
                    day=day,
                    expected=expected,
                )
                selected_index = pd.DatetimeIndex(selected.index).tz_convert(
                    "UTC"
                )
            if selected_index.has_duplicates or not selected_index.equals(expected):
                missing = expected.difference(selected_index)
                extra = selected_index.difference(expected)
                raise FuelMaterializationError(
                    f"{spec.alias} {day.date()}: couverture Saturn incomplete "
                    f"ou desordonnee; obtenu={len(selected_index)}, "
                    f"attendu={len(expected)}, missing={len(missing)}, "
                    f"extra={len(extra)}."
                )
            values = pd.to_numeric(selected, errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(values).all():
                raise FuelMaterializationError(
                    f"{spec.alias} {day.date()}: valeur Saturn non finie."
                )
            return pd.DataFrame(
                {
                    spec.alias: values,
                    f"{spec.alias}__snapshot_time_utc": cutoff_utc,
                    f"{spec.alias}__revision_time_utc": cutoff_utc,
                    f"{spec.alias}__spring_dst_repair": (
                        selected_index.isin(repaired_utc)
                        if repaired_utc
                        else np.zeros(len(expected), dtype=bool)
                    ),
                },
                index=expected,
            )
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(float(2 ** (attempt - 1)))
                LOCAL.client = None
                LOCAL.timeout_seconds = None
    raise FuelMaterializationError(
        f"{spec.alias} {day.date()}: Saturn as-of en echec: {last_error}"
    ) from last_error


def _repair_nl_spring_dst_hour(
    selected: pd.Series,
    *,
    spec: ResidualLoadSeries,
    day: pd.Timestamp,
    expected: pd.DatetimeIndex,
) -> tuple[pd.Series, tuple[pd.Timestamp, ...]]:
    """Repair the two documented NL spring-DST source omissions.

    Saturn's NL residual-load formula is contractually UTC-naive.  On the
    spring switch days inspected in the historical archive its UTC labels 02
    and 04 are absent, yielding the two missing physical products 04:00 and
    06:00 CEST.  They are reconstructed only when they are the exact missing
    set, from their immediate UTC neighbours in the same as-of forecast.  No
    realised or later-revision information is used.
    """

    if spec.alias != NL_SPRING_DST_REPAIR_ALIAS or len(expected) != 23:
        return selected, ()
    selected_index = pd.DatetimeIndex(selected.index)
    if selected_index.tz is None or selected_index.has_duplicates:
        return selected, ()
    selected_utc = selected_index.tz_convert("UTC")
    missing = expected.difference(selected_utc)
    extra = selected_utc.difference(expected)
    if len(missing) != 2 or len(extra) != 0:
        return selected, ()

    missing_utc = tuple(pd.Timestamp(value) for value in missing)
    missing_local = tuple(
        value.tz_convert(spec.delivery_timezone) for value in missing_utc
    )
    expected_day = pd.Timestamp(day).date()
    if (
        any(value.date() != expected_day for value in missing_local)
        or tuple(value.strftime("%H:%M") for value in missing_local)
        != NL_SPRING_DST_REPAIR_LOCAL_TIMES
    ):
        return selected, ()

    by_utc = pd.Series(
        pd.to_numeric(selected, errors="coerce").to_numpy(dtype=float),
        index=selected_utc,
        dtype=float,
    )
    repaired_values: list[float] = []
    for value in missing_utc:
        donors = [
            value - pd.Timedelta(hours=1),
            value + pd.Timedelta(hours=1),
        ]
        if any(donor not in by_utc.index for donor in donors):
            return selected, ()
        neighbours = by_utc.loc[donors].to_numpy(dtype=float)
        if not np.isfinite(neighbours).all():
            return selected, ()
        repaired_values.append(float(neighbours.mean()))

    repaired = pd.concat(
        [
            selected,
            pd.Series(
                repaired_values,
                index=pd.DatetimeIndex(missing_local),
                name=selected.name,
                dtype=float,
            ),
        ]
    ).sort_index()
    repaired_index = pd.DatetimeIndex(repaired.index).tz_convert("UTC")
    if repaired_index.has_duplicates or not repaired_index.equals(expected):
        return selected, ()
    return repaired, missing_utc


def _fetch_residual_saturn_series(
    spec: ResidualLoadSeries,
    days: Sequence[pd.Timestamp],
    *,
    day_workers: int,
    retries: int,
    timeout_seconds: float,
) -> pd.DataFrame:
    frames: dict[pd.Timestamp, pd.DataFrame] = {}
    failures: dict[pd.Timestamp, str] = {}
    with ThreadPoolExecutor(max_workers=day_workers) as pool:
        jobs = {
            pool.submit(
                _query_residual_saturn_day,
                spec,
                day,
                retries=retries,
                timeout_seconds=timeout_seconds,
            ): day
            for day in days
        }
        for number, future in enumerate(as_completed(jobs), start=1):
            day = jobs[future]
            try:
                frames[day] = future.result()
            except Exception as exc:
                failures[day] = str(exc)
            if number == 1 or number % 50 == 0 or number == len(days):
                print(
                    f"[{spec.alias}/saturn] {number}/{len(days)} jours | "
                    f"ok={len(frames)} fail={len(failures)}",
                    flush=True,
                )

    # A long backfill can encounter a small number of transient proxy/server
    # failures even after the per-request retries.  Retry only those days in a
    # bounded sequential recovery pass.  A fresh thread-local client is used
    # for every failed day; coverage remains fail-closed after this pass.
    if failures:
        failed_days = sorted(failures)
        print(
            f"[{spec.alias}/saturn] rattrapage sequentiel de "
            f"{len(failed_days)} jour(s): "
            + ", ".join(day.date().isoformat() for day in failed_days),
            flush=True,
        )
        final_failures: dict[pd.Timestamp, str] = {}
        for number, day in enumerate(failed_days, start=1):
            LOCAL.client = None
            LOCAL.timeout_seconds = None
            try:
                frames[day] = _query_residual_saturn_day(
                    spec,
                    day,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                final_failures[day] = str(exc)
            print(
                f"[{spec.alias}/saturn/recovery] {number}/{len(failed_days)} "
                f"| ok={number - len(final_failures)} "
                f"fail={len(final_failures)}",
                flush=True,
            )
        failures = final_failures
    if failures:
        details = "\n".join(
            f"{day.date().isoformat()}: {message}"
            for day, message in sorted(failures.items())
        )
        raise FuelMaterializationError(
            f"{spec.alias}: {len(failures)} jour(s) Saturn en echec:\n"
            + details
        )
    merged = pd.concat([frames[day] for day in sorted(frames)])
    expected = _expected_hourly_index(days[0], days[-1])
    if not merged.index.equals(expected):
        raise FuelMaterializationError(
            f"{spec.alias}: timeline Saturn multi-jours incomplete."
        )
    return merged


def _fetch_residual_saturn_bank(
    days: Sequence[pd.Timestamp],
    *,
    series_workers: int,
    day_workers: int,
    retries: int,
    timeout_seconds: float,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(
        max_workers=min(series_workers, len(RESIDUAL_LOAD_SERIES))
    ) as pool:
        jobs = {
            pool.submit(
                _fetch_residual_saturn_series,
                spec,
                days,
                day_workers=day_workers,
                retries=retries,
                timeout_seconds=timeout_seconds,
            ): spec
            for spec in RESIDUAL_LOAD_SERIES
        }
        for future in as_completed(jobs):
            spec = jobs[future]
            frames[spec.alias] = future.result()
    if set(frames) != set(RESIDUAL_LOAD_ALIASES):
        raise FuelMaterializationError(
            "Banque Saturn de charge residuelle incomplete."
        )
    return frames


def _read_residual_load_vintage(
    path: Path,
    *,
    alias: str,
    expected_index: pd.DatetimeIndex,
    expected_cutoffs: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Select the latest causally eligible vintage for every physical hour."""

    if not path.is_file():
        raise FuelMaterializationError(
            f"{alias}: vintage local introuvable: {path}."
        )
    try:
        raw = pd.read_parquet(path, columns=list(RESIDUAL_LOAD_RAW_COLUMNS))
    except Exception as exc:
        raise FuelMaterializationError(
            f"{alias}: lecture du vintage local impossible: {exc}"
        ) from exc
    if tuple(raw.columns) != RESIDUAL_LOAD_RAW_COLUMNS:
        raise FuelMaterializationError(
            f"{alias}: schema vintage invalide: {list(raw.columns)}."
        )
    frame = raw.copy()
    time_columns = (
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "downloaded_at_utc",
    )
    for column in time_columns:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if bool(frame[column].isna().any()):
            raise FuelMaterializationError(
                f"{alias}: timestamp absent ou invalide dans {column}."
            )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    cutoff_by_hour = pd.Series(
        expected_cutoffs,
        index=expected_index,
        name="cutoff_time_utc",
    )
    frame["cutoff_time_utc"] = frame["value_time_utc"].map(cutoff_by_hour)
    frame = frame.loc[frame["cutoff_time_utc"].notna()].copy()
    frame = frame.loc[
        (frame["snapshot_time_utc"] <= frame["cutoff_time_utc"])
        & (frame["revision_time_utc"] <= frame["cutoff_time_utc"])
    ].copy()
    if frame.empty:
        raise FuelMaterializationError(
            f"{alias}: aucune vintage causale dans la periode demandee."
        )

    selection_keys = [
        "value_time_utc",
        "revision_time_utc",
        "snapshot_time_utc",
        "downloaded_at_utc",
    ]
    conflicting = frame.loc[
        frame.duplicated(selection_keys, keep=False),
        [*selection_keys, "value"],
    ]
    if not conflicting.empty:
        conflicts = conflicting.groupby(selection_keys, dropna=False)[
            "value"
        ].nunique(dropna=False)
        if bool((conflicts > 1).any()):
            raise FuelMaterializationError(
                f"{alias}: valeurs contradictoires pour une meme vintage."
            )
    selected = (
        frame.sort_values(selection_keys, kind="mergesort")
        .drop_duplicates("value_time_utc", keep="last")
        .set_index("value_time_utc")
        .reindex(expected_index)
    )
    missing = selected["revision_time_utc"].isna()
    if bool(missing.any()):
        missing_hours = expected_index[missing.to_numpy()]
        preview = ", ".join(timestamp.isoformat() for timestamp in missing_hours[:5])
        raise FuelMaterializationError(
            f"{alias}: {int(missing.sum())} heure(s) sans vintage causale; "
            f"premieres={preview}."
        )
    values = pd.to_numeric(selected["value"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise FuelMaterializationError(
            f"{alias}: la vintage selectionnee contient une valeur non finie."
        )
    return pd.DataFrame(
        {
            alias: values,
            f"{alias}__snapshot_time_utc": pd.DatetimeIndex(
                selected["snapshot_time_utc"]
            ),
            f"{alias}__revision_time_utc": pd.DatetimeIndex(
                selected["revision_time_utc"]
            ),
        },
        index=expected_index,
    )


def _validate_residual_load_market_frame(
    frame: pd.DataFrame,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> None:
    if tuple(frame.columns) != RESIDUAL_LOAD_OUTPUT_COLUMNS:
        raise FuelMaterializationError(
            f"Schema du sidecar charge residuelle invalide: {list(frame.columns)}."
        )
    value_time = pd.DatetimeIndex(
        pd.to_datetime(frame["value_time_utc"], utc=True, errors="raise")
    )
    expected = _expected_hourly_index(start_day, end_day)
    if not value_time.equals(expected):
        raise FuelMaterializationError(
            "Le sidecar charge residuelle ne couvre pas exactement les heures "
            "physiques demandees."
        )
    cutoff = pd.DatetimeIndex(
        pd.to_datetime(frame["cutoff_time_utc"], utc=True, errors="raise")
    )
    expected_cutoff = _cutoffs_for_value_times(expected)
    if not cutoff.equals(expected_cutoff):
        raise FuelMaterializationError(
            "Le cutoff du sidecar charge residuelle n'est pas D-1 08:00 civil."
        )
    for column in ("snapshot_time_utc", "revision_time_utc"):
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(frame[column], utc=True, errors="raise")
        )
        if bool((timestamps > cutoff).any()):
            raise FuelMaterializationError(
                f"{column}: violation de causalite dans le sidecar residuel."
            )
    numeric = frame.loc[:, list(RESIDUAL_LOAD_ALIASES)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise FuelMaterializationError(
            "Valeur non finie dans le sidecar charge residuelle."
        )


def _validate_spring_dst_repair_audit(
    audit: dict[str, Any],
    *,
    artifact_start: pd.Timestamp,
    artifact_end: pd.Timestamp,
) -> None:
    """Validate the dynamic NL spring repair ledger before cache reuse."""

    counts = audit.get("spring_dst_repair_count")
    details = audit.get("spring_dst_repair_details")
    aliases = set(RESIDUAL_LOAD_ALIASES)
    if not isinstance(counts, dict) or set(counts) != aliases:
        raise FuelMaterializationError(
            "Audit des reparations DST incomplet: compteurs invalides."
        )
    if not isinstance(details, dict) or set(details) != aliases:
        raise FuelMaterializationError(
            "Audit des reparations DST incomplet: details invalides."
        )

    for alias in RESIDUAL_LOAD_ALIASES:
        entries = details.get(alias)
        if not isinstance(entries, list) or any(
            not isinstance(value, dict) for value in entries
        ):
            raise FuelMaterializationError(
                f"Audit des reparations DST invalide pour {alias}."
            )
        if counts.get(alias) != len(entries):
            raise FuelMaterializationError(
                f"Audit des reparations DST incoherent pour {alias}."
            )
        if alias != NL_SPRING_DST_REPAIR_ALIAS and entries:
            raise FuelMaterializationError(
                f"Reparation DST interdite pour {alias}."
            )
        required = {
            "value_time_utc",
            "value_time_local",
            "donor_value_times_utc",
            "donor_value_times_local",
            "method",
            "same_asof_forecast",
        }
        if any(set(entry) != required for entry in entries):
            raise FuelMaterializationError(
                f"Schema du ledger des reparations DST invalide pour {alias}."
            )
        parsed = pd.DatetimeIndex(
            pd.to_datetime(
                [entry["value_time_utc"] for entry in entries],
                utc=True,
                errors="raise",
            )
        )
        if parsed.has_duplicates or not parsed.is_monotonic_increasing:
            raise FuelMaterializationError(
                f"Ledger des reparations DST non unique ou desordonne pour {alias}."
            )
        repaired_times_by_day: dict[str, list[str]] = {}
        for timestamp, entry in zip(parsed, entries):
            local = pd.Timestamp(timestamp).tz_convert("Europe/Amsterdam")
            local_day = pd.Timestamp(local.date())
            repaired_times_by_day.setdefault(
                local_day.date().isoformat(), []
            ).append(local.strftime("%H:%M"))
            donors = pd.DatetimeIndex(
                pd.to_datetime(
                    entry["donor_value_times_utc"],
                    utc=True,
                    errors="raise",
                )
            )
            expected_donors = pd.DatetimeIndex(
                [
                    timestamp - pd.Timedelta(hours=1),
                    timestamp + pd.Timedelta(hours=1),
                ]
            )
            if (
                local_day < artifact_start
                or local_day > artifact_end
                or local.strftime("%H:%M")
                not in NL_SPRING_DST_REPAIR_LOCAL_TIMES
                or len(_physical_utc_index(local_day)) != 23
                or entry["value_time_local"] != local.isoformat()
                or not donors.equals(expected_donors)
                or entry["donor_value_times_local"]
                != [
                    value.tz_convert("Europe/Amsterdam").isoformat()
                    for value in expected_donors
                ]
                or entry["method"] != "linear_mean"
                or entry["same_asof_forecast"] is not True
            ):
                raise FuelMaterializationError(
                    "Ledger des reparations DST contient un instant non autorise: "
                    f"{timestamp}."
                )
        if any(
            tuple(times) != NL_SPRING_DST_REPAIR_LOCAL_TIMES
            for times in repaired_times_by_day.values()
        ):
            raise FuelMaterializationError(
                "Ledger des reparations DST NL incomplet pour un jour de "
                "passage a l'heure d'ete."
            )


def _load_existing_residual_load_market(
    output: Path,
    audit_path: Path,
    *,
    requested_start: pd.Timestamp,
    source_mode: str,
    vintage_root: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.Timestamp]:
    """Validate an existing residual sidecar before reuse or suffix append."""

    if output.is_file() != audit_path.is_file():
        raise FuelMaterializationError(
            "Sidecar residuel incomplet: Parquet et audit doivent exister ensemble."
        )
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FuelMaterializationError(
            f"Audit du sidecar residuel illisible: {exc}"
        ) from exc
    expected_fields: dict[str, Any] = {
        "schema_version": RESIDUAL_LOAD_SCHEMA_VERSION,
        "artifact_type": "kalman_residual_load_market_features_pit_hourly",
        "timezone": TIMEZONE,
        "cutoff_timezone": TIMEZONE,
        "cutoff_time": CUTOFF_TIME,
        "source_mode": source_mode,
        "information_type": "day_ahead_forecasts_known_before_cutoff",
        "columns": list(RESIDUAL_LOAD_OUTPUT_COLUMNS),
        "aliases": list(RESIDUAL_LOAD_ALIASES),
        "snapshot_time_semantics": (
            "maximum selected source snapshot per hour"
        ),
        "revision_time_semantics": (
            "maximum selected source revision per hour"
        ),
        "fill_or_interpolation": (
            SATURN_RESIDUAL_FILL_POLICY
            if source_mode == "saturn"
            else "none; exact physical-hour coverage required"
        ),
        "causality_violations": 0,
    }
    expected_selection_rule = (
        "Saturn state queried independently for each series and delivery "
        "day with revision_date=D-1 08:00 Europe/Paris; exact physical "
        "23/24/25-hour delivery grid required"
        if source_mode == "saturn"
        else (
            "for each physical delivery hour, snapshot_time_utc<=cutoff and "
            "revision_time_utc<=cutoff, then latest lexicographic "
            "revision/snapshot/download order"
        )
    )
    expected_fields["selection_rule"] = expected_selection_rule
    for key, expected in expected_fields.items():
        if audit.get(key) != expected:
            raise FuelMaterializationError(
                f"Audit du sidecar residuel incompatible: {key}="
                f"{audit.get(key)!r}, attendu={expected!r}."
            )
    actual_sha = _sha256(output)
    if (
        audit.get("sha256") != actual_sha
        or audit.get("output_sha256") != actual_sha
    ):
        raise FuelMaterializationError(
            "Checksum Parquet/audit du sidecar residuel different."
        )
    artifact_start = _normalise_day(
        audit.get("start_day"), name="audit residuel.start_day"
    )
    artifact_end = _normalise_day(
        audit.get("end_day"), name="audit residuel.end_day"
    )
    if artifact_start > requested_start or artifact_end < artifact_start:
        raise FuelMaterializationError(
            "Le sidecar residuel existant commence apres le debut demande "
            f"({artifact_start.date().isoformat()} > "
            f"{requested_start.date().isoformat()}); extension de prefixe "
            "refusee."
        )
    frame = pd.read_parquet(output)
    _validate_residual_load_market_frame(
        frame,
        start_day=artifact_start,
        end_day=artifact_end,
    )
    expected_days = int((artifact_end - artifact_start).days + 1)
    if audit.get("days") != expected_days or audit.get("rows") != len(frame):
        raise FuelMaterializationError(
            "Couverture Parquet/audit du sidecar residuel incoherente."
        )
    first_delivery = pd.Timestamp(frame["value_time_utc"].iloc[0]).isoformat()
    last_delivery = pd.Timestamp(frame["value_time_utc"].iloc[-1]).isoformat()
    if (
        audit.get("first_delivery_utc") != first_delivery
        or audit.get("last_delivery_utc") != last_delivery
    ):
        raise FuelMaterializationError(
            "Bornes Parquet/audit du sidecar residuel incoherentes."
        )

    if source_mode == "saturn":
        saturn_fields: dict[str, Any] = {
            "series": {spec.alias: spec.series for spec in RESIDUAL_LOAD_SERIES},
            "delivery_timezones": {
                spec.alias: spec.delivery_timezone
                for spec in RESIDUAL_LOAD_SERIES
            },
            "naive_timezones": {
                spec.alias: spec.naive_timezone for spec in RESIDUAL_LOAD_SERIES
            },
            "incomplete_dst_policy": {
                spec.alias: spec.incomplete_dst_policy
                for spec in RESIDUAL_LOAD_SERIES
            },
            "spring_dst_repair_policy": {
                spec.alias: (
                    NL_SPRING_DST_REPAIR_POLICY
                    if spec.alias == NL_SPRING_DST_REPAIR_ALIAS
                    else "none"
                )
                for spec in RESIDUAL_LOAD_SERIES
            },
            "revision_query": (
                "revision_date=cutoff_time_utc for every delivery day"
            ),
            "provider_revision_timestamp_available": False,
        }
        for key, expected in saturn_fields.items():
            if audit.get(key) != expected:
                raise FuelMaterializationError(
                    f"Provenance Saturn incompatible dans l'audit: {key}."
                )
        _validate_spring_dst_repair_audit(
            audit,
            artifact_start=artifact_start,
            artifact_end=artifact_end,
        )
    else:
        source_root = VINTAGE_ROOT if vintage_root is None else Path(vintage_root)
        paths = {
            alias: source_root / f"{alias}.parquet"
            for alias in RESIDUAL_LOAD_ALIASES
        }
        if any(not path.is_file() for path in paths.values()):
            raise FuelMaterializationError(
                "Un vintage local du sidecar residuel reutilise est absent."
            )
        expected_paths = {
            alias: str(path.resolve()) for alias, path in paths.items()
        }
        expected_checksums = {
            alias: _sha256(path) for alias, path in paths.items()
        }
        if (
            audit.get("raw_input_files") != expected_paths
            or audit.get("raw_input_sha256") != expected_checksums
        ):
            raise FuelMaterializationError(
                "Les vintages locaux ont change depuis la materialisation "
                "du sidecar residuel."
            )
    return frame, audit, artifact_end


def _materialize_residual_load_market_features(
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    output_dir: Path,
    source_mode: str = "saturn",
    series_workers: int = 2,
    day_workers: int = 4,
    retries: int = 3,
    timeout_seconds: float = 60.0,
    vintage_root: Path | None = None,
    force_rebuild: bool = False,
) -> tuple[Path, Path]:
    """Rebuild the five-zone residual-load PIT sidecar atomically."""

    mode = str(source_mode).strip().lower()
    if mode not in {"saturn", "local-vintages"}:
        raise ValueError(
            "source_mode residuel doit etre saturn ou local-vintages."
        )
    if series_workers < 1 or day_workers < 1:
        raise ValueError("series_workers et day_workers doivent etre >= 1.")
    if series_workers * day_workers > MAX_TOTAL_WORKERS:
        raise ValueError(
            "Parallelisme residuel refuse: series_workers * day_workers doit "
            f"etre <= {MAX_TOTAL_WORKERS}."
        )
    if retries < 1:
        raise ValueError("retries residuel doit etre >= 1.")
    if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds residuel doit etre strictement positif.")

    output = Path(output_dir) / RESIDUAL_LOAD_OUTPUT_NAME
    audit_path = output.with_name(output.name + AUDIT_SUFFIX)
    existing: pd.DataFrame | None = None
    existing_audit: dict[str, Any] | None = None
    artifact_start = start_day
    materialize_start = start_day
    if not force_rebuild and (output.exists() or audit_path.exists()):
        existing, existing_audit, existing_end = _load_existing_residual_load_market(
            output,
            audit_path,
            requested_start=start_day,
            source_mode=mode,
            vintage_root=vintage_root,
        )
        artifact_start = _normalise_day(
            existing_audit.get("start_day"),
            name="audit residuel.start_day",
        )
        if existing_end >= end_day:
            print(
                f"{output} | REUSE | couverture residuelle valide jusqu'au "
                f"{existing_end.date().isoformat()}",
                flush=True,
            )
            return output, audit_path
        materialize_start = existing_end + pd.Timedelta(days=1)

    expected = _expected_hourly_index(materialize_start, end_day)
    cutoffs = _cutoffs_for_value_times(expected)
    sources: dict[str, Path] = {}
    checksums_after: dict[str, str] = {}
    if mode == "saturn":
        days = list(pd.date_range(materialize_start, end_day, freq="D"))
        selected = _fetch_residual_saturn_bank(
            days,
            series_workers=int(series_workers),
            day_workers=int(day_workers),
            retries=int(retries),
            timeout_seconds=float(timeout_seconds),
        )
        selection_rule = (
            "Saturn state queried independently for each series and delivery "
            "day with revision_date=D-1 08:00 Europe/Paris; exact physical "
            "23/24/25-hour delivery grid required"
        )
    else:
        source_root = VINTAGE_ROOT if vintage_root is None else Path(vintage_root)
        sources = {
            alias: source_root / f"{alias}.parquet"
            for alias in RESIDUAL_LOAD_ALIASES
        }
        for alias, path in sources.items():
            if not path.is_file():
                raise FuelMaterializationError(
                    f"{alias}: vintage local introuvable: {path}."
                )
        checksums_before = {
            alias: _sha256(path) for alias, path in sources.items()
        }
        selected = {
            alias: _read_residual_load_vintage(
                sources[alias],
                alias=alias,
                expected_index=expected,
                expected_cutoffs=cutoffs,
            )
            for alias in RESIDUAL_LOAD_ALIASES
        }
        checksums_after = {
            alias: _sha256(path) for alias, path in sources.items()
        }
        if checksums_after != checksums_before:
            raise FuelMaterializationError(
                "Un vintage de charge residuelle a change pendant sa lecture."
            )
        selection_rule = (
            "for each physical delivery hour, snapshot_time_utc<=cutoff and "
            "revision_time_utc<=cutoff, then latest lexicographic "
            "revision/snapshot/download order"
        )

    spring_dst_repairs: dict[str, list[dict[str, Any]]] = {
        alias: [] for alias in RESIDUAL_LOAD_ALIASES
    }
    if mode == "saturn":
        if existing_audit is not None:
            prior = existing_audit.get("spring_dst_repair_details", {})
            spring_dst_repairs = {
                alias: list(prior.get(alias, []))
                for alias in RESIDUAL_LOAD_ALIASES
            }
        for alias in RESIDUAL_LOAD_ALIASES:
            marker = f"{alias}__spring_dst_repair"
            if marker not in selected[alias].columns:
                raise FuelMaterializationError(
                    f"{alias}: indicateur interne de reparation DST absent."
                )
            repaired_index = selected[alias].index[
                selected[alias][marker].astype(bool).to_numpy()
            ]
            for timestamp in repaired_index:
                repaired_utc = pd.Timestamp(timestamp).tz_convert("UTC")
                donors_utc = [
                    repaired_utc - pd.Timedelta(hours=1),
                    repaired_utc + pd.Timedelta(hours=1),
                ]
                spring_dst_repairs[alias].append(
                    {
                        "value_time_utc": repaired_utc.isoformat(),
                        "value_time_local": repaired_utc.tz_convert(
                            "Europe/Amsterdam"
                        ).isoformat(),
                        "donor_value_times_utc": [
                            value.isoformat() for value in donors_utc
                        ],
                        "donor_value_times_local": [
                            value.tz_convert("Europe/Amsterdam").isoformat()
                            for value in donors_utc
                        ],
                        "method": "linear_mean",
                        "same_asof_forecast": True,
                    }
                )
            by_timestamp = {
                entry["value_time_utc"]: entry
                for entry in spring_dst_repairs[alias]
            }
            spring_dst_repairs[alias] = [
                by_timestamp[key] for key in sorted(by_timestamp)
            ]

    snapshot_ns = np.column_stack(
        [
            pd.DatetimeIndex(
                selected[alias][f"{alias}__snapshot_time_utc"]
            ).asi8
            for alias in RESIDUAL_LOAD_ALIASES
        ]
    )
    revision_ns = np.column_stack(
        [
            pd.DatetimeIndex(
                selected[alias][f"{alias}__revision_time_utc"]
            ).asi8
            for alias in RESIDUAL_LOAD_ALIASES
        ]
    )
    suffix = pd.DataFrame(
        {
            "value_time_utc": expected,
            "snapshot_time_utc": pd.to_datetime(
                snapshot_ns.max(axis=1), utc=True
            ),
            "revision_time_utc": pd.to_datetime(
                revision_ns.max(axis=1), utc=True
            ),
            "cutoff_time_utc": cutoffs,
            **{
                alias: selected[alias][alias].to_numpy(dtype=float)
                for alias in RESIDUAL_LOAD_ALIASES
            },
        }
    ).loc[:, list(RESIDUAL_LOAD_OUTPUT_COLUMNS)]
    frame = (
        suffix
        if existing is None
        else pd.concat([existing, suffix], ignore_index=True)
    ).loc[:, list(RESIDUAL_LOAD_OUTPUT_COLUMNS)]
    _validate_residual_load_market_frame(
        frame,
        start_day=artifact_start,
        end_day=end_day,
    )

    payload: dict[str, Any] = {
        "schema_version": RESIDUAL_LOAD_SCHEMA_VERSION,
        "artifact_type": "kalman_residual_load_market_features_pit_hourly",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "timezone": TIMEZONE,
        "cutoff_timezone": TIMEZONE,
        "cutoff_time": CUTOFF_TIME,
        "source_mode": mode,
        "information_type": "day_ahead_forecasts_known_before_cutoff",
        "selection_rule": selection_rule,
        "columns": list(RESIDUAL_LOAD_OUTPUT_COLUMNS),
        "aliases": list(RESIDUAL_LOAD_ALIASES),
        "start_day": artifact_start.date().isoformat(),
        "end_day": end_day.date().isoformat(),
        "days": int((end_day - artifact_start).days + 1),
        "requested_start_day": start_day.date().isoformat(),
        "rows": int(len(frame)),
        "first_delivery_utc": pd.Timestamp(
            frame["value_time_utc"].iloc[0]
        ).isoformat(),
        "last_delivery_utc": pd.Timestamp(
            frame["value_time_utc"].iloc[-1]
        ).isoformat(),
        "snapshot_time_semantics": "maximum selected source snapshot per hour",
        "revision_time_semantics": "maximum selected source revision per hour",
        "fill_or_interpolation": (
            SATURN_RESIDUAL_FILL_POLICY
            if mode == "saturn"
            else "none; exact physical-hour coverage required"
        ),
        "causality_violations": 0,
    }
    if mode == "saturn":
        payload.update(
            {
                "series": {
                    spec.alias: spec.series for spec in RESIDUAL_LOAD_SERIES
                },
                "delivery_timezones": {
                    spec.alias: spec.delivery_timezone
                    for spec in RESIDUAL_LOAD_SERIES
                },
                "naive_timezones": {
                    spec.alias: spec.naive_timezone
                    for spec in RESIDUAL_LOAD_SERIES
                },
                "incomplete_dst_policy": {
                    spec.alias: spec.incomplete_dst_policy
                    for spec in RESIDUAL_LOAD_SERIES
                },
                "spring_dst_repair_policy": {
                    spec.alias: (
                        NL_SPRING_DST_REPAIR_POLICY
                        if spec.alias == NL_SPRING_DST_REPAIR_ALIAS
                        else "none"
                    )
                    for spec in RESIDUAL_LOAD_SERIES
                },
                "spring_dst_repair_count": {
                    alias: len(spring_dst_repairs[alias])
                    for alias in RESIDUAL_LOAD_ALIASES
                },
                "spring_dst_repair_details": spring_dst_repairs,
                "revision_query": (
                    "revision_date=cutoff_time_utc for every delivery day"
                ),
                "provider_revision_timestamp_available": False,
            }
        )
    else:
        payload.update(
            {
                "raw_input_files": {
                    alias: str(path.resolve()) for alias, path in sources.items()
                },
                "raw_input_sha256": checksums_after,
            }
        )
    _write_bundle(frame, output, audit_path, payload)
    return output, audit_path


def _recover_daily(frame: pd.DataFrame) -> pd.DataFrame:
    value_time = pd.DatetimeIndex(
        pd.to_datetime(frame["value_time_utc"], utc=True, errors="raise")
    )
    local_days = value_time.tz_convert(TIMEZONE).normalize().tz_localize(None)
    work = frame.copy()
    work["delivery_day_local"] = local_days
    daily = work.groupby("delivery_day_local", sort=True).first()
    daily.index = pd.DatetimeIndex(daily.index)
    return daily


def _validate_hourly_frame(
    frame: pd.DataFrame,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    ccgt_efficiency: float,
    ccgt_emission_tco2_mwh: float,
    ccgt_vom_eur_mwh: float,
) -> None:
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise FuelMaterializationError(
            f"Schema fuel invalide: {list(frame.columns)}."
        )
    index = pd.DatetimeIndex(
        pd.to_datetime(frame["value_time_utc"], utc=True, errors="raise")
    )
    expected = _expected_hourly_index(start_day, end_day)
    if not index.equals(expected):
        raise FuelMaterializationError(
            "La timeline fuel ne couvre pas exactement les heures physiques."
        )
    snapshot = pd.DatetimeIndex(
        pd.to_datetime(frame["snapshot_time_utc"], utc=True, errors="raise")
    )
    revision = pd.DatetimeIndex(
        pd.to_datetime(frame["revision_time_utc"], utc=True, errors="raise")
    )
    local_days = index.tz_convert(TIMEZONE).normalize().tz_localize(None)
    expected_cutoffs = pd.DatetimeIndex(
        [_civil_cutoff(day).tz_convert("UTC") for day in local_days]
    )
    if not snapshot.equals(expected_cutoffs) or not revision.equals(
        expected_cutoffs
    ):
        raise FuelMaterializationError(
            "snapshot/revision ne correspondent pas au cutoff civil D-1 08:00."
        )
    for column in SOURCE_TIME_COLUMNS:
        source_time = pd.DatetimeIndex(
            pd.to_datetime(frame[column], utc=True, errors="raise")
        )
        if bool((source_time > expected_cutoffs).any()):
            raise FuelMaterializationError(
                f"{column}: timestamp source posterieur au cutoff."
            )
    numeric = frame.loc[:, list(FEATURE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise FuelMaterializationError("Features fuel non finies dans l'artefact.")
    expected_ccgt = (
        numeric["ttf_m1_eur_mwh_th"] / float(ccgt_efficiency)
        + numeric["eua_first_dec_eur_tco2"]
        * float(ccgt_emission_tco2_mwh)
        + float(ccgt_vom_eur_mwh)
    )
    if not np.allclose(
        numeric["ccgt_marginal_cost_eur_mwh"],
        expected_ccgt,
        rtol=0.0,
        atol=1e-10,
    ):
        raise FuelMaterializationError("Formule CCGT incoherente dans l'artefact.")
    work = frame.copy()
    work["delivery_day_local"] = local_days
    invariant = [*SOURCE_TIME_COLUMNS, *FEATURE_COLUMNS]
    counts = work.groupby("delivery_day_local")[invariant].nunique(dropna=False)
    if bool((counts > 1).any().any()):
        raise FuelMaterializationError(
            "Les features fuel ne sont pas constantes dans une journee."
        )


def _load_existing(
    output: Path,
    audit_path: Path,
    *,
    requested_start: pd.Timestamp,
    ccgt_efficiency: float,
    ccgt_emission_tco2_mwh: float,
    ccgt_vom_eur_mwh: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.Timestamp]:
    if output.is_file() != audit_path.is_file():
        raise FuelMaterializationError(
            "Artefact fuel incomplet: Parquet et audit doivent exister ensemble."
        )
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FuelMaterializationError(f"Audit fuel illisible: {exc}") from exc
    expected_series = {spec.alias: spec.series for spec in MARKET_SERIES}
    expected_units = {spec.alias: spec.unit for spec in MARKET_SERIES}
    expected_fields: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "kalman_market_fuel_features_pit_hourly",
        "series": expected_series,
        "units": expected_units,
        "timezone": TIMEZONE,
        "cutoff_timezone": TIMEZONE,
        "cutoff_time": CUTOFF_TIME,
        "warmup_days": WARMUP_DAYS,
        "columns": list(OUTPUT_COLUMNS),
        "information_type": "market_observation_known_before_cutoff",
        "snapshot_time_semantics": "query_asof_cutoff",
        "revision_time_semantics": (
            "query_asof_cutoff; provider insertion timestamp unavailable"
        ),
        "provider_revision_timestamp_available": False,
        "causality_violations": 0,
    }
    for key, expected in expected_fields.items():
        if audit.get(key) != expected:
            raise FuelMaterializationError(
                f"Audit fuel incompatible: {key}={audit.get(key)!r}, "
                f"attendu={expected!r}."
            )
    assumptions = audit.get("ccgt_assumptions")
    expected_assumptions = {
        "efficiency": float(ccgt_efficiency),
        "emission_tco2_mwh": float(ccgt_emission_tco2_mwh),
        "vom_eur_mwh": float(ccgt_vom_eur_mwh),
    }
    if assumptions != expected_assumptions:
        raise FuelMaterializationError(
            "Les hypotheses CCGT demandees different de l'artefact existant."
        )
    actual_sha = _sha256(output)
    if audit.get("sha256") != actual_sha:
        raise FuelMaterializationError("Checksum Parquet/audit fuel different.")
    artifact_start = _normalise_day(audit.get("start_day"), name="audit.start_day")
    artifact_end = _normalise_day(audit.get("end_day"), name="audit.end_day")
    if artifact_start != requested_start:
        raise FuelMaterializationError(
            "start-day doit rester identique au debut de l'artefact fuel "
            f"({artifact_start.date().isoformat()})."
        )
    frame = pd.read_parquet(output)
    _validate_hourly_frame(
        frame,
        start_day=artifact_start,
        end_day=artifact_end,
        ccgt_efficiency=ccgt_efficiency,
        ccgt_emission_tco2_mwh=ccgt_emission_tco2_mwh,
        ccgt_vom_eur_mwh=ccgt_vom_eur_mwh,
    )
    if audit.get("rows") != len(frame):
        raise FuelMaterializationError("Nombre de lignes incoherent dans l'audit.")
    expected_days = int((artifact_end - artifact_start).days + 1)
    if audit.get("days") != expected_days:
        raise FuelMaterializationError("Nombre de jours incoherent dans l'audit.")
    if audit.get("first_delivery_utc") != pd.Timestamp(
        frame["value_time_utc"].iloc[0]
    ).isoformat():
        raise FuelMaterializationError("Premiere livraison incoherente dans l'audit.")
    if audit.get("last_delivery_utc") != pd.Timestamp(
        frame["value_time_utc"].iloc[-1]
    ).isoformat():
        raise FuelMaterializationError("Derniere livraison incoherente dans l'audit.")
    return frame, audit, artifact_end


def _compare_overlap(existing: pd.DataFrame, queried: pd.DataFrame) -> None:
    existing_daily = _recover_daily(existing)
    overlap = existing_daily.index.intersection(queried.index)
    if overlap.empty:
        return
    pairs = (
        ("ttf_m1_eur_mwh_th", "ttf_m1_eur_mwh_th"),
        ("eua_first_dec_eur_tco2", "eua_first_dec_eur_tco2"),
    )
    for existing_column, queried_column in pairs:
        left = pd.to_numeric(
            existing_daily.loc[overlap, existing_column], errors="raise"
        ).to_numpy(dtype=float)
        right = pd.to_numeric(
            queried.loc[overlap, queried_column], errors="raise"
        ).to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-12):
            raise FuelMaterializationError(
                f"Le prefixe Saturn a change pour {existing_column}; "
                "extension incrementale refusee."
            )
    time_pairs = (
        (
            "ttf_source_value_time_utc",
            "ttf_m1_eur_mwh_th__source_value_time_utc",
        ),
        (
            "eua_source_value_time_utc",
            "eua_first_dec_eur_tco2__source_value_time_utc",
        ),
    )
    for existing_column, queried_column in time_pairs:
        left = pd.DatetimeIndex(
            pd.to_datetime(
                existing_daily.loc[overlap, existing_column],
                utc=True,
                errors="raise",
            )
        )
        right = pd.DatetimeIndex(
            pd.to_datetime(
                queried.loc[overlap, queried_column],
                utc=True,
                errors="raise",
            )
        )
        if not left.equals(right):
            raise FuelMaterializationError(
                f"Le timestamp source du prefixe a change pour {existing_column}; "
                "extension incrementale refusee."
            )


def _audit_payload(
    frame: pd.DataFrame,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    query_start: pd.Timestamp,
    ccgt_efficiency: float,
    ccgt_emission_tco2_mwh: float,
    ccgt_vom_eur_mwh: float,
) -> dict[str, Any]:
    daily = _recover_daily(frame)
    cutoff = pd.DatetimeIndex(
        pd.to_datetime(daily["snapshot_time_utc"], utc=True, errors="raise")
    )
    source_age: dict[str, float] = {}
    for alias, column in (
        ("ttf_m1_eur_mwh_th", "ttf_source_value_time_utc"),
        ("eua_first_dec_eur_tco2", "eua_source_value_time_utc"),
    ):
        source_time = pd.DatetimeIndex(
            pd.to_datetime(daily[column], utc=True, errors="raise")
        )
        source_age[alias] = float(
            ((cutoff - source_time) / pd.Timedelta(hours=1)).max()
        )
    return {
        "schema_version": 1,
        "artifact_type": "kalman_market_fuel_features_pit_hourly",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "series": {spec.alias: spec.series for spec in MARKET_SERIES},
        "units": {spec.alias: spec.unit for spec in MARKET_SERIES},
        "timezone": TIMEZONE,
        "cutoff_timezone": TIMEZONE,
        "cutoff_time": CUTOFF_TIME,
        "selection_rule": (
            "latest finite value timestamp <= cutoff in Saturn state queried "
            "with revision_date=cutoff"
        ),
        "snapshot_time_semantics": "query_asof_cutoff",
        "revision_time_semantics": (
            "query_asof_cutoff; provider insertion timestamp unavailable"
        ),
        "provider_revision_timestamp_available": False,
        "information_type": "market_observation_known_before_cutoff",
        "fill_or_interpolation": (
            "no interpolation; one independently selected last-known value per "
            "delivery day, then deterministic broadcast to physical hours"
        ),
        "warmup_days": WARMUP_DAYS,
        "feature_definitions": {
            "ttf_change_1d": "TTF(D)-TTF(D-1), EUR/MWh_th",
            "ttf_change_5d": "TTF(D)-TTF(D-5), EUR/MWh_th",
            "eua_change_1d": "EUA(D)-EUA(D-1), EUR/tCO2",
            "eua_change_5d": "EUA(D)-EUA(D-5), EUR/tCO2",
            "fuel_volatility_20d": (
                "population standard deviation over 20 daily changes of the "
                "derived CCGT marginal cost, EUR/MWh_e"
            ),
            "ccgt_marginal_cost_eur_mwh": (
                "TTF/efficiency + EUA*emission_factor + variable_O&M"
            ),
        },
        "ccgt_assumptions": {
            "efficiency": float(ccgt_efficiency),
            "emission_tco2_mwh": float(ccgt_emission_tco2_mwh),
            "vom_eur_mwh": float(ccgt_vom_eur_mwh),
        },
        "columns": list(OUTPUT_COLUMNS),
        "start_day": start_day.date().isoformat(),
        "end_day": end_day.date().isoformat(),
        "days": int((end_day - start_day).days + 1),
        "requested_start_day": requested_start.date().isoformat(),
        "requested_end_day": requested_end.date().isoformat(),
        "query_start_day_including_warmup": query_start.date().isoformat(),
        "rows": int(len(frame)),
        "first_delivery_utc": pd.Timestamp(
            frame["value_time_utc"].iloc[0]
        ).isoformat(),
        "last_delivery_utc": pd.Timestamp(
            frame["value_time_utc"].iloc[-1]
        ).isoformat(),
        "maximum_source_age_hours": source_age,
        "causality_violations": 0,
    }


def _write_bundle(
    frame: pd.DataFrame,
    output: Path,
    audit_path: Path,
    payload: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_output = output.with_name(f".{output.name}.{token}.tmp.parquet")
    temporary_audit = audit_path.with_name(f".{audit_path.name}.{token}.tmp")
    try:
        frame.to_parquet(temporary_output, index=False)
        verified = pd.read_parquet(temporary_output)
        if not verified.equals(frame.reset_index(drop=True)):
            raise FuelMaterializationError(
                "Verification de lecture du Parquet fuel en echec."
            )
        payload = dict(payload)
        output_sha256 = _sha256(temporary_output)
        payload["sha256"] = output_sha256
        payload["output_sha256"] = output_sha256
        temporary_audit.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_output, output)
        os.replace(temporary_audit, audit_path)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()
        if temporary_audit.exists():
            temporary_audit.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_parameters(args)
    requested_start = _normalise_day(args.start_day, name="start-day")
    residual_requested_start = _normalise_day(
        args.residual_start_day,
        name="residual-start-day",
    )
    requested_end = _normalise_day(args.end_day, name="end-day")
    if requested_end < requested_start:
        raise ValueError("end-day doit etre >= start-day.")
    if requested_end < residual_requested_start:
        raise ValueError("end-day doit etre >= residual-start-day.")
    output_dir = Path(args.output_dir).expanduser()
    output_dir = (
        output_dir if output_dir.is_absolute() else ROOT / output_dir
    ).resolve()
    output = output_dir / OUTPUT_NAME
    audit_path = output.with_name(output.name + AUDIT_SUFFIX)

    existing: pd.DataFrame | None = None
    artifact_start = requested_start
    materialize_start = requested_start
    if not args.force_rebuild and output.is_file() and audit_path.is_file():
        existing, _, existing_end = _load_existing(
            output,
            audit_path,
            requested_start=requested_start,
            ccgt_efficiency=float(args.ccgt_efficiency),
            ccgt_emission_tco2_mwh=float(args.ccgt_emission_tco2_mwh),
            ccgt_vom_eur_mwh=float(args.ccgt_vom_eur_mwh),
        )
        if existing_end >= requested_end:
            print(
                f"{output} | REUSE | couverture valide jusqu'au "
                f"{existing_end.date().isoformat()}",
                flush=True,
            )
            if not args.skip_residual_load:
                residual_output, _ = _materialize_residual_load_market_features(
                    start_day=residual_requested_start,
                    end_day=requested_end,
                    output_dir=output_dir,
                    source_mode=str(args.residual_load_source),
                    series_workers=int(args.series_workers),
                    day_workers=int(args.day_workers),
                    retries=int(args.retries),
                    timeout_seconds=float(args.request_timeout_seconds),
                    force_rebuild=bool(args.force_rebuild),
                )
                print(
                    f"{residual_output} | READY | "
                    f"delivery={residual_requested_start.date().isoformat()}.."
                    f"{requested_end.date().isoformat()}",
                    flush=True,
                )
            return 0
        materialize_start = existing_end + pd.Timedelta(days=1)
    elif not args.force_rebuild and (output.exists() or audit_path.exists()):
        raise FuelMaterializationError(
            "Artefact fuel incomplet: utilisez --force-rebuild apres audit manuel."
        )

    query_start = materialize_start - pd.Timedelta(days=WARMUP_DAYS)
    query_days = list(pd.date_range(query_start, requested_end, freq="D"))
    raw_daily = _fetch_daily_market(
        query_days,
        series_workers=int(args.series_workers),
        day_workers=int(args.day_workers),
        retries=int(args.retries),
        timeout_seconds=float(args.request_timeout_seconds),
    )
    if existing is not None:
        _compare_overlap(existing, raw_daily)
    daily = _derive_daily_features(
        raw_daily,
        ccgt_efficiency=float(args.ccgt_efficiency),
        ccgt_emission_tco2_mwh=float(args.ccgt_emission_tco2_mwh),
        ccgt_vom_eur_mwh=float(args.ccgt_vom_eur_mwh),
    )
    suffix = _broadcast_daily_features(
        daily,
        start_day=materialize_start,
        end_day=requested_end,
    )
    frame = (
        suffix
        if existing is None
        else pd.concat([existing, suffix], ignore_index=True)
    ).loc[:, list(OUTPUT_COLUMNS)]
    _validate_hourly_frame(
        frame,
        start_day=artifact_start,
        end_day=requested_end,
        ccgt_efficiency=float(args.ccgt_efficiency),
        ccgt_emission_tco2_mwh=float(args.ccgt_emission_tco2_mwh),
        ccgt_vom_eur_mwh=float(args.ccgt_vom_eur_mwh),
    )
    payload = _audit_payload(
        frame,
        start_day=artifact_start,
        end_day=requested_end,
        requested_start=requested_start,
        requested_end=requested_end,
        query_start=query_start,
        ccgt_efficiency=float(args.ccgt_efficiency),
        ccgt_emission_tco2_mwh=float(args.ccgt_emission_tco2_mwh),
        ccgt_vom_eur_mwh=float(args.ccgt_vom_eur_mwh),
    )
    _write_bundle(frame, output, audit_path, payload)
    print(
        f"{output} | rows={len(frame)} | days={payload['days']} | "
        f"delivery={payload['first_delivery_utc']}.."
        f"{payload['last_delivery_utc']}",
        flush=True,
    )
    if not args.skip_residual_load:
        residual_output, _ = _materialize_residual_load_market_features(
            start_day=residual_requested_start,
            end_day=requested_end,
            output_dir=output_dir,
            source_mode=str(args.residual_load_source),
            series_workers=int(args.series_workers),
            day_workers=int(args.day_workers),
            retries=int(args.retries),
            timeout_seconds=float(args.request_timeout_seconds),
            force_rebuild=bool(args.force_rebuild),
        )
        print(
            f"{residual_output} | READY | "
            f"delivery={residual_requested_start.date().isoformat()}.."
            f"{requested_end.date().isoformat()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_SUFFIX",
    "FEATURE_COLUMNS",
    "FuelMaterializationError",
    "MARKET_SERIES",
    "OUTPUT_COLUMNS",
    "OUTPUT_NAME",
    "RESIDUAL_LOAD_ALIASES",
    "RESIDUAL_LOAD_OUTPUT_COLUMNS",
    "RESIDUAL_LOAD_OUTPUT_NAME",
    "RESIDUAL_LOAD_SERIES",
    "SOURCE_TIME_COLUMNS",
    "TIMEZONE",
    "WARMUP_DAYS",
    "_broadcast_daily_features",
    "_civil_cutoff",
    "_derive_daily_features",
    "_materialize_residual_load_market_features",
    "_physical_utc_index",
    "_query_market_value",
    "main",
    "parse_args",
]
