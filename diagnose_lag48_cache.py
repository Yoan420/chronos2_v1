#!/usr/bin/env python3
"""Diagnostic en lecture seule des covariables observationnelles en lag48.

Le script ne contacte pas Saturn, ne modifie aucun cache et n'affiche aucune
valeur de marché. Il reproduit seulement le contrat temporel du pipeline pour
identifier si la couverture future nulle vient :

* d'un cache absent ou produit avec une autre identité de configuration ;
* d'un cache trop ancien, auquel cas une actualisation est à tester ;
* de la latence de publication de la série source ;
* d'un décalage d'index/heure ;
* de trous ponctuels que la politique de remplissage ne couvre pas.

Utilisation depuis la racine du dépôt Windows :

    python .\\diagnose_lag48_cache.py --config .\\chronos2_hourly_fr.yaml

Pour reproduire exactement un ancien run :

    python .\\diagnose_lag48_cache.py --config .\\chronos2_hourly_fr.yaml ^
        --as-of 2026-08-11T10:16:00+02:00
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


DEFAULT_ALIASES = (
    "fr_nuclear_generation_obs",
    "fr_net_exports",
)


@dataclass(frozen=True)
class CacheSnapshot:
    path: Path
    raw_rows: int
    raw_kind: str
    raw_min: str | None
    raw_max: str | None
    normalized: pd.Series
    normalized_duplicates: int
    normalized_non_hourly: int
    mtime_utc: pd.Timestamp


@dataclass(frozen=True)
class DiagnosticSpec:
    alias: str
    series: str | None
    source: str
    enabled: bool
    fill_method: str
    fill_limit: int
    future_strategies: tuple[str, ...]
    naive_timezone: str | None
    incomplete_dst_policy: str


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostique les couvertures lag48 sans réseau ni écriture."
        )
    )
    parser.add_argument(
        "--config",
        default="chronos2_hourly_fr.yaml",
        help="Chemin du YAML horaire (défaut: chronos2_hourly_fr.yaml).",
    )
    parser.add_argument(
        "--zone",
        default="FR",
        help="Zone à diagnostiquer (défaut: FR).",
    )
    parser.add_argument(
        "--aliases",
        nargs="+",
        default=list(DEFAULT_ALIASES),
        help="Alias YAML à diagnostiquer.",
    )
    parser.add_argument(
        "--lag-hours",
        type=int,
        default=48,
        help="Lag physique en heures (défaut: 48).",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help=(
            "Horodatage du run, avec offset recommandé. Sinon le runtime_as_of "
            "du YAML ou l'heure courante est utilisé."
        ),
    )
    parser.add_argument(
        "--fresh-cache-hours",
        type=float,
        default=24.0,
        help=(
            "Cache considéré récemment actualisé sous ce seuil "
            "(défaut: 24 h)."
        ),
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="N'afficher que les heures sources encore manquantes.",
    )
    return parser.parse_args()


def _deep_get(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _parse_spec(
    alias: str,
    raw: dict[str, Any],
    *,
    default_fill_limit: int,
) -> DiagnosticSpec:
    future_raw = raw.get("future", raw.get("future_strategies", []))
    if isinstance(future_raw, dict):
        strategies = future_raw.get("strategies", [])
    else:
        strategies = future_raw
    if isinstance(strategies, str):
        strategies = [strategies]
    return DiagnosticSpec(
        alias=alias,
        series=(str(raw["series"]) if raw.get("series") else None),
        source=str(raw.get("source", "auto")).lower(),
        enabled=bool(raw.get("enabled", True)),
        fill_method=str(raw.get("fill_method", "ffill")).lower(),
        fill_limit=int(raw.get("fill_limit", default_fill_limit)),
        future_strategies=tuple(str(value).lower() for value in strategies or []),
        naive_timezone=(
            str(raw["naive_timezone"]).strip()
            if raw.get("naive_timezone") not in (None, "")
            else None
        ),
        incomplete_dst_policy=str(
            raw.get("incomplete_dst_policy", "raise")
        ).strip().lower(),
    )


def _sanitize_filename(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value.strip("_") or "series"


def _cache_path_for_series(
    cache_root: Path,
    zone: str,
    spec: DiagnosticSpec,
) -> Path:
    identity = spec.series or spec.alias
    if spec.naive_timezone:
        identity = f"{identity}|naive_timezone={spec.naive_timezone}"
    if spec.incomplete_dst_policy != "raise":
        identity = (
            f"{identity}|incomplete_dst_policy="
            f"{spec.incomplete_dst_policy}"
        )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    return (
        cache_root
        / zone.lower()
        / f"{_sanitize_filename(spec.alias)}__{digest}.csv.gz"
    )


def _parse_timestamp_series(
    values: pd.Series,
    timezone: str,
) -> pd.DatetimeIndex:
    sample = values.dropna().astype(str).head(300)
    aware = bool(
        not sample.empty
        and sample.str.contains(
            r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True
        ).mean()
        >= 0.5
    )
    if aware:
        return pd.DatetimeIndex(
            pd.to_datetime(values, errors="coerce", utc=True)
        ).tz_convert(timezone)

    parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if parsed.tz is not None:
        return parsed.tz_convert(timezone)
    try:
        return parsed.tz_localize(
            timezone,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    except Exception:
        return parsed.tz_localize(
            timezone,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )


def _runtime_as_of(timezone: str, config: dict[str, Any]) -> pd.Timestamp:
    raw = _deep_get(config, "data.runtime_as_of")
    if raw in (None, ""):
        return pd.Timestamp.now(tz=timezone)
    timestamp = pd.Timestamp(raw)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(timezone)


def _target_cutoff(timezone: str, config: dict[str, Any]) -> pd.Timestamp:
    policy = str(
        _deep_get(config, "data.target_end_policy", "current_day_end")
    ).strip().lower()
    if policy == "configured":
        configured_end = _deep_get(config, "data.end")
        if not configured_end:
            raise ValueError("target_end_policy=configured exige data.end")
        cutoff = pd.Timestamp(configured_end)
        return (
            cutoff.tz_localize(timezone)
            if cutoff.tzinfo is None
            else cutoff.tz_convert(timezone)
        )
    if policy == "now":
        return _runtime_as_of(timezone, config).floor(
            str(_deep_get(config, "data.frequency", "h"))
        )
    if policy != "current_day_end":
        raise ValueError(
            "Le diagnostic requiert target_end_policy=current_day_end, "
            "configured ou now."
        )
    frequency = str(_deep_get(config, "data.frequency", "h"))
    offset = pd.tseries.frequencies.to_offset(frequency)
    today_start = _runtime_as_of(timezone, config).normalize()
    return today_start + pd.DateOffset(days=1) - offset


def _local_delivery_day_index(
    delivery_date: str | date | pd.Timestamp,
    *,
    timezone: str,
) -> pd.DatetimeIndex:
    timestamp = pd.Timestamp(delivery_date)
    local_date = (
        timestamp.tz_convert(timezone).date()
        if timestamp.tzinfo is not None
        else timestamp.date()
    )
    start_local = pd.Timestamp(local_date).tz_localize(timezone)
    end_local = start_local + pd.DateOffset(days=1)
    result = pd.date_range(
        start=start_local.tz_convert("UTC"),
        end=end_local.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    if len(result) not in {23, 24, 25}:
        raise ValueError(
            f"Journée locale invalide : {local_date} contient {len(result)} h."
        )
    return result


def _format_timestamp(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _format_hours(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} h"


def _hours(delta: pd.Timedelta) -> float:
    return float(delta.total_seconds() / 3600.0)


def _cache_timestamp_column(columns: Iterable[Any]) -> str:
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): str(column)
        for column in columns
    }
    for alias in (
        "timestamp",
        "datetime",
        "date_time",
        "date",
        "time",
        "index",
        "unnamed: 0",
    ):
        match = normalized.get(re.sub(r"[^a-z0-9]", "", alias.lower()))
        if match is not None:
            return match
    raise KeyError("colonne temporelle non identifiée dans le cache")


def _cache_value_column(columns: Iterable[Any], timestamp_col: str) -> str:
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): str(column)
        for column in columns
        if str(column) != timestamp_col
    }
    for alias in ("value", "observation", "actual", "target", "price"):
        match = normalized.get(re.sub(r"[^a-z0-9]", "", alias.lower()))
        if match is not None:
            return match
    remaining = [str(column) for column in columns if str(column) != timestamp_col]
    if len(remaining) == 1:
        return remaining[0]
    raise KeyError("colonne de valeur non identifiée dans le cache")


def _read_cache_snapshot(
    path: Path,
    *,
    timezone: str,
    parse_timestamp_series: Any,
) -> CacheSnapshot:
    header = pd.read_csv(path, nrows=0)
    timestamp_col = _cache_timestamp_column(header.columns)
    value_col = _cache_value_column(header.columns, timestamp_col)
    frame = pd.read_csv(
        path,
        usecols=[timestamp_col, value_col],
        low_memory=False,
    )

    raw_timestamps = frame[timestamp_col]
    raw_strings = raw_timestamps.dropna().astype(str)
    aware_ratio = (
        float(
            raw_strings.str.contains(
                r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True
            ).mean()
        )
        if not raw_strings.empty
        else 0.0
    )
    raw_kind = "aware" if aware_ratio >= 0.5 else "naive"
    raw_parsed = pd.to_datetime(
        raw_timestamps,
        errors="coerce",
        utc=(raw_kind == "aware"),
    )

    index = parse_timestamp_series(raw_timestamps, timezone)
    values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy()
    valid_index = ~index.isna()
    normalized = pd.Series(values[valid_index], index=index[valid_index])
    normalized = normalized.sort_index()
    duplicate_count = int(normalized.index.duplicated().sum())
    if duplicate_count:
        normalized = normalized.groupby(level=0).mean()

    if len(normalized.index) >= 2:
        deltas = normalized.index.tz_convert("UTC").to_series().diff().dropna()
        normalized_non_hourly = int((deltas != pd.Timedelta(hours=1)).sum())
    else:
        normalized_non_hourly = 0

    mtime_utc = pd.Timestamp(
        os.path.getmtime(path), unit="s", tz="UTC"
    )
    return CacheSnapshot(
        path=path,
        raw_rows=int(len(frame)),
        raw_kind=raw_kind,
        raw_min=_format_timestamp(raw_parsed.min()),
        raw_max=_format_timestamp(raw_parsed.max()),
        normalized=normalized,
        normalized_duplicates=duplicate_count,
        normalized_non_hourly=normalized_non_hourly,
        mtime_utc=mtime_utc,
    )


def _live_future_index(
    *,
    timezone: str,
    config: dict[str, Any],
) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    cutoff = _target_cutoff(timezone, config)
    frequency = str(_deep_get(config, "data.frequency", "h"))
    offset = pd.tseries.frequencies.to_offset(frequency)
    dynamic = bool(
        _deep_get(config, "data.dynamic_delivery_day_horizon", False)
    )
    if dynamic:
        next_delivery_date = (cutoff + offset).tz_convert(timezone).date()
        future = _local_delivery_day_index(
            next_delivery_date, timezone=timezone
        ).tz_convert(timezone)
    else:
        horizon = int(_deep_get(config, "model.horizon", 24))
        future = pd.date_range(
            start=cutoff + offset,
            periods=horizon,
            freq=frequency,
            tz=timezone,
        )
    return cutoff, future


def _filled_source_values(
    series: pd.Series,
    required: pd.DatetimeIndex,
    *,
    fill_method: str,
    fill_limit: int,
) -> pd.Series:
    if required.empty:
        return pd.Series(dtype=float, index=required)

    history = max(int(fill_limit), 0)
    start = required.min() - pd.Timedelta(hours=history)
    analysis_index = pd.date_range(
        start=start,
        end=required.max(),
        freq="h",
        tz=required.tz,
    )
    aligned = series.reindex(analysis_index)
    if fill_method == "ffill" and history > 0:
        aligned = aligned.ffill(limit=history)
    elif fill_method == "zero":
        aligned = aligned.fillna(0.0)
    return aligned.reindex(required)


def _shift_candidates(
    series: pd.Series,
    required: pd.DatetimeIndex,
) -> list[tuple[int, float]]:
    candidates: list[tuple[int, float]] = []
    for shift_hours in (-24, -2, -1, 1, 2, 24):
        shifted = required + pd.Timedelta(hours=shift_hours)
        candidates.append(
            (shift_hours, float(series.reindex(shifted).notna().mean()))
        )
    return sorted(candidates, key=lambda item: item[1], reverse=True)


def _diagnose_cause(
    *,
    snapshot: CacheSnapshot,
    required: pd.DatetimeIndex,
    exact_coverage: float,
    filled_coverage: float,
    runtime_as_of: pd.Timestamp,
    fresh_cache_hours: float,
) -> tuple[str, str]:
    series = snapshot.normalized
    if series.empty:
        return "CACHE_EMPTY", "Le cache ne contient aucun timestamp exploitable."
    if filled_coverage >= 1.0:
        return (
            "COVERAGE_OK",
            "Le cache couvre tout le lag demandé après la politique de remplissage.",
        )

    maximum = series.index.max().tz_convert(required.tz)
    cache_age = _hours(runtime_as_of.tz_convert("UTC") - snapshot.mtime_utc)
    recent_cache = cache_age <= fresh_cache_hours
    if maximum < required.min():
        if recent_cache:
            return (
                "SOURCE_PUBLICATION_LATENCY_PROBABLE",
                "Le cache a été écrit récemment, mais la dernière observation "
                "précède toute la fenêtre requise. Un simple refetch risque de "
                "reproduire le même manque.",
            )
        return (
            "REFETCH_NEEDED_TO_DISCRIMINATE",
            "Le cache est ancien et s'arrête avant toute la fenêtre requise. "
            "Actualiser cette seule série, puis relancer le diagnostic : si son "
            "maximum ne bouge pas assez, la cause est la latence source.",
        )

    if maximum < required.max():
        if recent_cache:
            return (
                "SOURCE_PUBLICATION_LATENCY_PARTIAL",
                "Le cache récent ne publie qu'une partie de la fenêtre source.",
            )
        return (
            "PARTIAL_CACHE_REFETCH_NEEDED",
            "Le cache ancien s'arrête au milieu de la fenêtre source.",
        )

    best_shift, best_coverage = _shift_candidates(series, required)[0]
    if exact_coverage < 0.5 and best_coverage >= 0.9:
        sign = "+" if best_shift > 0 else ""
        return (
            "INDEX_OR_TIMEZONE_SHIFT",
            f"La couverture devient {100 * best_coverage:.1f}% avec un décalage "
            f"fixe de {sign}{best_shift} h : vérifier naive_timezone et l'unité "
            "temporelle de la série.",
        )

    if snapshot.normalized_non_hourly:
        return (
            "INDEX_CADENCE_MISMATCH",
            f"{snapshot.normalized_non_hourly} pas temporels normalisés ne valent "
            "pas exactement une heure.",
        )

    return (
        "INTERNAL_GAPS_OR_NULL_VALUES",
        "Le cache atteint la fin requise, mais conserve des trous ou valeurs "
        "nulles que fill_limit ne couvre pas.",
    )


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _print_hour_table(
    *,
    future: pd.DatetimeIndex,
    required: pd.DatetimeIndex,
    exact: pd.Series,
    filled: pd.Series,
    missing_only: bool,
) -> None:
    print(
        "position | future_local | future_utc | source_lag_local | "
        "source_lag_utc | exact | apres_fill"
    )
    for position, (future_ts, source_ts) in enumerate(
        zip(future, required), start=1
    ):
        exact_ok = bool(pd.notna(exact.loc[source_ts]))
        filled_ok = bool(pd.notna(filled.loc[source_ts]))
        if missing_only and filled_ok:
            continue
        print(
            f"{position:02d} | {future_ts.isoformat()} | "
            f"{future_ts.tz_convert('UTC').isoformat()} | "
            f"{source_ts.isoformat()} | "
            f"{source_ts.tz_convert('UTC').isoformat()} | "
            f"{'OK' if exact_ok else 'MISS'} | "
            f"{'OK' if filled_ok else 'MISS'}"
        )


def main() -> int:
    args = _arguments()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration introuvable : {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Le YAML doit contenir un mapping à la racine.")
    if args.as_of:
        data_config = config.setdefault("data", {})
        if not isinstance(data_config, dict):
            raise TypeError("data doit être un mapping YAML.")
        data_config["runtime_as_of"] = args.as_of

    zones_raw = config.get("zones", {})
    if not isinstance(zones_raw, dict):
        raise TypeError("zones doit être un mapping YAML.")
    zone_raw = zones_raw.get(args.zone.upper())
    if not isinstance(zone_raw, dict):
        raise KeyError(f"Zone absente du YAML : {args.zone.upper()}")
    timezone = str(zone_raw.get("timezone", "Europe/Paris"))
    covariates_raw = zone_raw.get("covariates", {})
    if not isinstance(covariates_raw, dict):
        raise TypeError("zones.<zone>.covariates doit être un mapping YAML.")
    default_fill = int(_deep_get(config, "data.default_fill_limit", 3))
    covariates = {
        str(alias): _parse_spec(
            str(alias),
            raw,
            default_fill_limit=default_fill,
        )
        for alias, raw in covariates_raw.items()
        if isinstance(raw, dict)
    }
    runtime_as_of = _runtime_as_of(timezone, config)
    cutoff, future = _live_future_index(
        timezone=timezone,
        config=config,
    )
    required = future - pd.Timedelta(hours=args.lag_hours)

    configured_root = _resolve_path(
        _deep_get(config, "data.project_root", "."), config_path.parent
    )
    cache_root = _resolve_path(
        _deep_get(config, "data.cache_dir", "data/chronos2_modular"),
        configured_root,
    )

    print("=== CONTRAT LIVE ===")
    print(f"config={config_path}")
    print(f"project_root={configured_root}")
    print(f"runtime_as_of={runtime_as_of.isoformat()}")
    print(f"target_context_end={cutoff.isoformat()}")
    print(
        f"forecast={future[0].isoformat()} -> {future[-1].isoformat()} "
        f"({len(future)} heures physiques)"
    )
    print(
        f"sources_lag{args.lag_hours}={required[0].isoformat()} -> "
        f"{required[-1].isoformat()} ({len(required)} timestamps)"
    )

    for alias in args.aliases:
        print(f"\n=== {alias} ===")
        spec = covariates.get(alias)
        if spec is None:
            print("cause=ALIAS_NOT_CONFIGURED")
            print("detail=L'alias n'existe pas sous zones.<zone>.covariates.")
            continue
        if not spec.enabled:
            print("cause=ALIAS_DISABLED")
            print("detail=L'alias est désactivé dans le YAML.")
            continue

        expected_path = _cache_path_for_series(
            cache_root, args.zone.upper(), spec
        )
        candidates = sorted(
            expected_path.parent.glob(
                f"{_sanitize_filename(spec.alias)}__*.csv.gz"
            )
        )
        print(f"series={spec.series}")
        print(f"source={spec.source}")
        print(f"configured_strategies={','.join(spec.future_strategies)}")
        print(f"expected_cache={_display_path(expected_path, configured_root)}")
        print(f"matching_cache_files={len(candidates)}")

        if not expected_path.exists():
            for candidate in candidates:
                print(
                    "alternate_cache="
                    f"{_display_path(candidate, configured_root)}"
                )
            if candidates:
                print("cause=CACHE_IDENTITY_MISMATCH")
                print(
                    "detail=Un cache du même alias existe, mais pas celui de "
                    "l'identité YAML courante (series/naive_timezone/"
                    "incomplete_dst_policy). Le pipeline ne le lit pas."
                )
            else:
                print("cause=CACHE_ABSENT")
                print("detail=Aucun cache local ne correspond à cet alias.")
            continue

        snapshot = _read_cache_snapshot(
            expected_path,
            timezone=timezone,
            parse_timestamp_series=_parse_timestamp_series,
        )
        series = snapshot.normalized
        normalized_first = series.index.min() if not series.empty else None
        normalized_last = series.index.max() if not series.empty else None
        exact = series.reindex(required)
        filled = _filled_source_values(
            series,
            required,
            fill_method=spec.fill_method,
            fill_limit=spec.fill_limit,
        )
        exact_coverage = float(exact.notna().mean())
        filled_coverage = float(filled.notna().mean())
        cache_age = _hours(
            runtime_as_of.tz_convert("UTC") - snapshot.mtime_utc
        )
        observation_age = (
            _hours(
                runtime_as_of.tz_convert("UTC")
                - normalized_last.tz_convert("UTC")
            )
            if normalized_last is not None
            else None
        )
        required_end_deficit = (
            max(
                0.0,
                _hours(
                    required[-1].tz_convert("UTC")
                    - normalized_last.tz_convert("UTC")
                ),
            )
            if normalized_last is not None
            else None
        )

        print(f"cache_mtime_utc={snapshot.mtime_utc.isoformat()}")
        print(f"cache_age_at_run={_format_hours(cache_age)}")
        print(f"raw_rows={snapshot.raw_rows}")
        print(f"raw_timestamp_kind={snapshot.raw_kind}")
        print(f"raw_timestamp_min={snapshot.raw_min}")
        print(f"raw_timestamp_max={snapshot.raw_max}")
        print(f"normalized_timestamp_min={_format_timestamp(normalized_first)}")
        print(f"normalized_timestamp_max={_format_timestamp(normalized_last)}")
        print(f"normalized_duplicates={snapshot.normalized_duplicates}")
        print(f"normalized_non_hourly_steps={snapshot.normalized_non_hourly}")
        print(f"observation_age_at_run={_format_hours(observation_age)}")
        print(
            "hours_missing_at_required_end="
            f"{_format_hours(required_end_deficit)}"
        )
        print(
            f"coverage_exact={int(exact.notna().sum())}/{len(required)} "
            f"({100 * exact_coverage:.1f}%)"
        )
        print(
            f"coverage_after_{spec.fill_method}_limit_{spec.fill_limit}="
            f"{int(filled.notna().sum())}/{len(required)} "
            f"({100 * filled_coverage:.1f}%)"
        )

        cause, detail = _diagnose_cause(
            snapshot=snapshot,
            required=required,
            exact_coverage=exact_coverage,
            filled_coverage=filled_coverage,
            runtime_as_of=runtime_as_of,
            fresh_cache_hours=args.fresh_cache_hours,
        )
        print(f"cause={cause}")
        print(f"detail={detail}")
        _print_hour_table(
            future=future,
            required=required,
            exact=exact,
            filled=filled,
            missing_only=args.missing_only,
        )

    print(
        "\nNOTE: le diagnostic est entièrement local et ne peut pas prouver "
        "la latence Saturn si le cache est ancien. Dans ce cas il indique "
        "REFETCH_NEEDED_TO_DISCRIMINATE, puis la comparaison après "
        "actualisation tranche entre cache périmé et source retardée."
    )
    print(
        "NOTE DST: la borne 2025-10-27 affichée par l'ancien diagnostic DST "
        "venait de sa fenêtre d'analyse limitée aux transitions 2022-2025 ; "
        "elle ne prouve pas que la série Saturn ou le cache courant s'arrête "
        "en 2025. Seul normalized_timestamp_max ci-dessus mesure ce cache."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
