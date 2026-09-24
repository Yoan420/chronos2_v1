#!/usr/bin/env python
"""Materialise the causal Saturn weather bundle used by the Kalman sidecar.

The command deliberately delegates every curve to
``materialize_saturn_daily_asof.py``.  It only owns the multi-zone plan,
bounded orchestration, resumability checks and the configuration catalogue.
No realised weather value is queried or joined here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
MATERIALIZER = ROOT / "materialize_saturn_daily_asof.py"
CATALOG_JSON = "saturn_kalman_weather_catalog.json"
CATALOG_YAML = "saturn_kalman_weather_catalog.yaml"
CATALOG_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = "data/pit/kalman_weather"
MAX_TOTAL_WORKERS = 32

ZONE_TIMEZONES: dict[str, str] = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}


@dataclass(frozen=True)
class SeriesPlan:
    zone: str
    kind: str
    alias: str
    series: str
    timezone: str
    naive_timezone: str
    output: Path
    daily_broadcast: bool = False
    value_scale: float = 1.0
    unit: str = ""

    @property
    def audit_path(self) -> Path:
        return self.output.with_name(self.output.name + ".audit.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialise les previsions Saturn vent/solaire/temperature, "
            "strictement as-of D-1 08:00, pour le Kalman."
        )
    )
    parser.add_argument("--start-day", required=True)
    parser.add_argument("--end-day", required=True)
    parser.add_argument(
        "--zones",
        nargs="+",
        default=list(ZONE_TIMEZONES),
        help="Zones separees par des virgules ou des espaces (FR,DE,BE,NL,ES).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Repertoire de sortie des caches PIT meteo. Le cache historique "
            "reste reutilisable seulement apres validation complete de sa "
            "semantique temporelle et de son checksum."
        ),
    )
    parser.add_argument("--series-workers", type=int, default=2)
    parser.add_argument("--day-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _normalise_day(value: str, *, name: str) -> pd.Timestamp:
    try:
        day = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} invalide: {value!r}.") from exc
    if day.tzinfo is not None:
        raise ValueError(f"{name} doit etre une date civile sans fuseau.")
    return day


def normalise_zones(values: Sequence[str]) -> tuple[str, ...]:
    zones: list[str] = []
    for raw in values:
        zones.extend(item.strip().upper() for item in str(raw).split(","))
    zones = [zone for zone in zones if zone]
    unknown = sorted(set(zones).difference(ZONE_TIMEZONES))
    if unknown:
        raise ValueError(
            f"Zones non supportees: {unknown}; attendu={list(ZONE_TIMEZONES)}."
        )
    if not zones:
        raise ValueError("--zones ne doit pas etre vide.")
    return tuple(dict.fromkeys(zones))


def validate_parallelism(series_workers: int, day_workers: int) -> None:
    if series_workers < 1 or day_workers < 1:
        raise ValueError("--series-workers et --day-workers doivent etre >= 1.")
    if series_workers * day_workers > MAX_TOTAL_WORKERS:
        raise ValueError(
            "Parallelisme refuse: series-workers * day-workers doit etre "
            f"<= {MAX_TOTAL_WORKERS}."
        )


def build_plan(zones: Sequence[str], output_dir: Path) -> tuple[SeriesPlan, ...]:
    """Return the immutable three-series Saturn contract for each zone."""

    output_root = Path(output_dir).expanduser().resolve()
    plans: list[SeriesPlan] = []
    for zone in normalise_zones(zones):
        lower = zone.casefold()
        timezone = ZONE_TIMEZONES[zone]
        definitions = [
            {
                "kind": "wind",
                "alias": f"{lower}_wind_generation_fcst",
                "series": f"power.{lower}.generation.wind.hourly.gw.fcst",
                "naive_timezone": timezone,
                "daily_broadcast": False,
                "value_scale": 1.0,
                "unit": "GW",
            },
            {
                "kind": "solar",
                "alias": f"{lower}_solar_generation_fcst",
                "series": f"power.{lower}.generation.solar.hourly.gw.fcst",
                "naive_timezone": timezone,
                "daily_broadcast": False,
                "value_scale": 1.0,
                "unit": "GW",
            },
            {
                "kind": "temperature_2m",
                "alias": f"{lower}_temperature_fcst",
                "series": f"meteo.nrjscan.{lower}.t_2m.index.fcst.d",
                "naive_timezone": timezone,
                "daily_broadcast": True,
                "value_scale": 1.0,
                "unit": "degC",
            },
        ]
        # The canonical NL hourly curve is a UTC-aware primary in MW.  The
        # generic formula has no sufficiently deep causal vintage history.
        if zone == "NL":
            definitions[0].update(
                {
                    "series": (
                        "power.nl.prod.total.wind.mw.ecmwf_avg."
                        "pointconnect.6h.cache"
                    ),
                    "naive_timezone": "UTC",
                    "value_scale": 0.001,
                }
            )
            # The canonical NL solar formula explicitly applies
            # ``naive(..., "CET")`` to its UTC-aware ECMWF component before
            # blending it with the Meteologica primary.  Its returned labels
            # are therefore local-civil Europe/Amsterdam, not UTC-naive.
        for definition in definitions:
            alias = str(definition["alias"])
            plans.append(
                SeriesPlan(
                    zone=zone,
                    kind=str(definition["kind"]),
                    alias=alias,
                    series=str(definition["series"]),
                    timezone=timezone,
                    naive_timezone=str(definition["naive_timezone"]),
                    # Aliases already contain the zone, so a flat directory is
                    # unambiguous and can be referenced directly by the lab
                    # configuration generated below.
                    output=output_root / f"{alias}.parquet",
                    daily_broadcast=bool(definition["daily_broadcast"]),
                    value_scale=float(definition["value_scale"]),
                    unit=str(definition["unit"]),
                )
            )
    return tuple(plans)


def build_command(
    plan: SeriesPlan,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    day_workers: int,
    merge_existing: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(MATERIALIZER),
        "--series",
        plan.series,
        "--alias",
        plan.alias,
        "--start-day",
        start_day.date().isoformat(),
        "--end-day",
        end_day.date().isoformat(),
        "--output",
        str(plan.output),
        "--timezone",
        plan.timezone,
        "--cutoff-timezone",
        plan.timezone,
        "--cutoff-time",
        "08:00",
        "--naive-timezone",
        plan.naive_timezone,
        "--workers",
        str(day_workers),
        "--incomplete-dst-policy",
        (
            "duplicate_zero_only"
            if plan.alias == "nl_solar_generation_fcst"
            else "duplicate"
        ),
        "--request-padding-hours",
        "8",
        "--value-scale",
        format(plan.value_scale, ".12g"),
    ]
    if plan.daily_broadcast:
        command.append("--daily-broadcast")
    if merge_existing:
        command.append("--merge-existing")
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_index(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    *,
    timezone: str,
) -> pd.DatetimeIndex:
    start_utc = start_day.tz_localize(timezone).tz_convert("UTC")
    end_utc = (end_day + pd.Timedelta(days=1)).tz_localize(timezone).tz_convert(
        "UTC"
    )
    return pd.date_range(start_utc, end_utc, freq="h", inclusive="left")


def reusable_output(
    plan: SeriesPlan,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> tuple[bool, str]:
    """Validate the complete semantic contract before skipping one series."""

    if not plan.output.is_file() or not plan.audit_path.is_file():
        return False, "parquet ou audit absent"
    try:
        audit = json.loads(plan.audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"audit illisible: {exc}"
    try:
        artifact_start = _normalise_day(
            str(audit["start_day"]), name="audit.start_day"
        )
        artifact_end = _normalise_day(
            str(audit["end_day"]), name="audit.end_day"
        )
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"bornes audit invalides: {exc}"
    if artifact_start != start_day:
        return False, (
            f"audit.start_day={artifact_start.date().isoformat()!r}, "
            f"attendu={start_day.date().isoformat()!r}"
        )
    if artifact_end < end_day:
        return False, (
            f"audit.end_day={artifact_end.date().isoformat()!r}, "
            f"couverture requise jusqu'au {end_day.date().isoformat()!r}"
        )
    expected_audit: dict[str, Any] = {
        "series": plan.series,
        "alias": plan.alias,
        "timezone": plan.timezone,
        "cutoff_timezone": plan.timezone,
        "cutoff_time": "08:00",
        "naive_timezone": plan.naive_timezone,
        "daily_broadcast": plan.daily_broadcast,
        "value_scale": plan.value_scale,
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            return False, f"audit.{key}={audit.get(key)!r}, attendu={expected!r}"
    dst_policy = audit.get("incomplete_dst_policy")
    expected_dst_policy = (
        "duplicate_zero_only"
        if plan.alias == "nl_solar_generation_fcst"
        else "duplicate"
    )
    if plan.alias == "nl_solar_generation_fcst":
        # The immutable legacy bank used the former covariate-only duplicate
        # policy.  It may remain reusable only when every repeated autumn fold
        # is exactly zero; all new downloads use the narrower zero-only rule.
        if dst_policy not in {"duplicate_zero_only", "duplicate"}:
            return False, (
                f"audit.incomplete_dst_policy={dst_policy!r}, "
                "attendu='duplicate_zero_only' "
                "(ou legacy 'duplicate' verifie)"
            )
    elif dst_policy != expected_dst_policy:
        return False, (
            f"audit.incomplete_dst_policy={dst_policy!r}, "
            f"attendu={expected_dst_policy!r}"
        )
    try:
        actual_sha = _sha256(plan.output)
    except OSError as exc:
        return False, f"parquet illisible: {exc}"
    if audit.get("sha256") != actual_sha:
        return False, "checksum parquet/audit different"
    try:
        frame = pd.read_parquet(
            plan.output,
            columns=[
                "value_time_utc",
                "snapshot_time_utc",
                "revision_time_utc",
                "value",
            ],
        )
        raw_index = frame["value_time_utc"]
        actual_index = pd.DatetimeIndex(pd.to_datetime(raw_index, utc=True))
    except Exception as exc:
        return False, f"timeline parquet illisible: {exc}"
    expected_index = _expected_index(
        artifact_start,
        artifact_end,
        timezone=plan.timezone,
    )
    if not actual_index.equals(expected_index):
        return False, "timeline physique non exactement couverte"
    values = pd.to_numeric(frame["value"], errors="coerce")
    if not values.map(lambda value: math.isfinite(float(value))).all():
        return False, "valeurs non finies dans le parquet"
    local_days = actual_index.tz_convert(plan.timezone).normalize().tz_localize(
        None
    )
    expected_cutoffs = pd.DatetimeIndex(
        [
            (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8))
            .tz_localize(plan.timezone)
            .tz_convert("UTC")
            for day in local_days
        ]
    )
    try:
        snapshots = pd.DatetimeIndex(
            pd.to_datetime(frame["snapshot_time_utc"], utc=True, errors="raise")
        )
        revisions = pd.DatetimeIndex(
            pd.to_datetime(frame["revision_time_utc"], utc=True, errors="raise")
        )
    except (TypeError, ValueError) as exc:
        return False, f"cutoffs PIT illisibles: {exc}"
    if len(snapshots) != len(expected_cutoffs) or not bool(
        (snapshots == expected_cutoffs).all()
    ):
        return False, "snapshot_time_utc ne correspond pas a D-1 08:00 civil"
    if len(revisions) != len(expected_cutoffs) or not bool(
        (revisions == expected_cutoffs).all()
    ):
        return False, "revision_time_utc ne correspond pas a D-1 08:00 civil"
    if plan.alias == "nl_solar_generation_fcst":
        local_wall = actual_index.tz_convert(plan.timezone).tz_localize(None)
        repeated = local_wall.duplicated(keep=False)
        if bool(repeated.any()):
            repeated_values = values.loc[repeated].to_numpy(dtype=float)
            if not bool((abs(repeated_values) <= 1e-12).all()):
                return False, (
                    "legacy NL solaire: fold automnal duplique non nul; "
                    "preuve physique insuffisante"
                )
        if dst_policy == "duplicate_zero_only":
            repair_rows = audit.get("dst_zero_duplicate_repairs")
            repair_count = audit.get("dst_zero_duplicate_repair_count")
            if not isinstance(repair_rows, list) or repair_count != len(repair_rows):
                return False, "audit des folds zero-only absent ou incoherent"
            seen_repairs: set[tuple[str, ...]] = set()
            for number, repair in enumerate(repair_rows, start=1):
                if not isinstance(repair, dict):
                    return False, f"audit fold zero-only #{number} non structure"
                try:
                    local_timestamp = pd.Timestamp(repair["local_timestamp"])
                    duplicated_value = float(repair["duplicated_value"])
                    declared_hours = pd.DatetimeIndex(
                        pd.to_datetime(
                            repair["physical_hours_utc"],
                            utc=True,
                            errors="raise",
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    return False, f"audit fold zero-only #{number} illisible: {exc}"
                if repair.get("policy") != "duplicate_zero_only":
                    return False, f"audit fold zero-only #{number}: policy invalide"
                if (
                    local_timestamp.tzinfo is not None
                    or not math.isfinite(duplicated_value)
                    or abs(duplicated_value) > 1e-12
                ):
                    return False, f"audit fold zero-only #{number}: valeur invalide"
                try:
                    first = local_timestamp.tz_localize(
                        plan.timezone, ambiguous=True, nonexistent="raise"
                    ).tz_convert("UTC")
                    second = local_timestamp.tz_localize(
                        plan.timezone, ambiguous=False, nonexistent="raise"
                    ).tz_convert("UTC")
                except (TypeError, ValueError) as exc:
                    return False, f"audit fold zero-only #{number}: heure invalide: {exc}"
                expected_hours = pd.DatetimeIndex(sorted((first, second)))
                if first == second or not declared_hours.equals(expected_hours):
                    return False, (
                        f"audit fold zero-only #{number}: folds physiques invalides"
                    )
                identity = tuple(item.isoformat() for item in expected_hours)
                if identity in seen_repairs:
                    return False, f"audit fold zero-only #{number}: doublon d'audit"
                seen_repairs.add(identity)
                positions = actual_index.isin(expected_hours)
                if int(positions.sum()) != 2 or not bool(
                    (values.loc[positions].abs() <= 1e-12).all()
                ):
                    return False, (
                        f"audit fold zero-only #{number}: valeurs parquet non prouvees"
                    )
    artifact_days = int((artifact_end - artifact_start).days + 1)
    if audit.get("days") != artifact_days:
        return False, "nombre de jours audit incoherent"
    if audit.get("rows") != len(expected_index):
        return False, "nombre de lignes audit incoherent"
    if audit.get("first_delivery_utc") != expected_index[0].isoformat():
        return False, "premiere heure audit incoherente"
    if audit.get("last_delivery_utc") != expected_index[-1].isoformat():
        return False, "derniere heure audit incoherente"
    coverage = (
        "plage exacte"
        if artifact_end == end_day
        else f"prefixe demande couvert jusqu'au {artifact_end.date().isoformat()}"
    )
    return True, f"{coverage}, configuration, timeline et checksum valides"


def reject_in_place_semantic_rewrite(plans: Sequence[SeriesPlan]) -> None:
    """Refuse to overwrite a cache whose timestamp interpretation changed.

    Historical artefacts are immutable evidence.  In particular, changing
    ``naive_timezone`` rewrites the physical meaning of every row even when
    the alias and Saturn series are unchanged.  Such a change must be audited
    and materialised into a fresh isolated directory.
    """

    conflicts: list[str] = []
    for plan in plans:
        if not plan.output.exists() and not plan.audit_path.exists():
            continue
        if not plan.output.is_file() or not plan.audit_path.is_file():
            conflicts.append(f"{plan.alias}: parquet/sidecar partiel")
            continue
        try:
            audit = json.loads(plan.audit_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            conflicts.append(f"{plan.alias}: sidecar illisible ({exc})")
            continue
        if (
            audit.get("series") == plan.series
            and audit.get("alias") == plan.alias
            and audit.get("naive_timezone") != plan.naive_timezone
        ):
            conflicts.append(
                f"{plan.alias}: naive_timezone historique="
                f"{audit.get('naive_timezone')!r}, corrige="
                f"{plan.naive_timezone!r}"
            )
    if conflicts:
        raise RuntimeError(
            "Reecriture semantique in-place refusee. Les caches existants "
            "restent des preuves immuables; choisissez un --output-dir neuf "
            "apres un audit explicite. Conflits: "
            + "; ".join(conflicts)
        )


def incremental_extension_start(
    plan: SeriesPlan,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> tuple[pd.Timestamp | None, str]:
    """Return the first missing suffix day when an existing prefix is exact."""

    if not plan.output.is_file() or not plan.audit_path.is_file():
        return None, "prefixe absent"
    try:
        audit = json.loads(plan.audit_path.read_text(encoding="utf-8"))
        existing_end = _normalise_day(
            str(audit["end_day"]), name="audit.end_day"
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
        return None, f"prefixe illisible: {exc}"
    if existing_end < start_day or existing_end >= end_day:
        return None, "prefixe non prolongeable"
    reusable, reason = reusable_output(
        plan,
        start_day=start_day,
        end_day=existing_end,
    )
    if not reusable:
        return None, f"prefixe invalide: {reason}"
    return existing_end + pd.Timedelta(days=1), "prefixe exact; extension incrementale"


def _config_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _catalog_entry(plan: SeriesPlan) -> dict[str, Any]:
    audit = json.loads(plan.audit_path.read_text(encoding="utf-8"))
    return {
        "zone": plan.zone,
        "kind": plan.kind,
        "alias": plan.alias,
        "series": plan.series,
        "unit": plan.unit,
        "value_scale": plan.value_scale,
        "timezone": plan.timezone,
        "naive_timezone": plan.naive_timezone,
        "source_timestamp_basis": (
            "utc_naive"
            if plan.naive_timezone.upper() == "UTC"
            else "local_civil_naive"
        ),
        "path": _config_path(plan.output),
        "absolute_path": str(plan.output.resolve()),
        "audit_path": str(plan.audit_path.resolve()),
        "audit_sha256": _sha256(plan.audit_path),
        "sha256": audit["sha256"],
        "rows": audit["rows"],
        "start_day": audit["start_day"],
        "end_day": audit["end_day"],
        "timestamp_column": "value_time_utc",
        "value_column": "value",
        "origin_column": "snapshot_time_utc",
        "revision_column": "revision_time_utc",
        "cutoff_column": "snapshot_time_utc",
        "cutoff_time": "08:00",
        "cutoff_policy": "Saturn as-of D-1 08:00 civil",
        "snapshot_time_semantics": audit.get(
            "snapshot_time_semantics", "query_asof_cutoff"
        ),
        "revision_time_semantics": audit.get(
            "revision_time_semantics",
            "query_asof_cutoff; provider insertion timestamp unavailable",
        ),
        "provider_revision_timestamp_available": bool(
            audit.get("provider_revision_timestamp_available", False)
        ),
        "incomplete_dst_policy": audit.get("incomplete_dst_policy", "raise"),
        "request_padding_hours": audit.get("request_padding_hours"),
        "dst_zero_duplicate_repair_count": audit.get(
            "dst_zero_duplicate_repair_count"
        ),
        "dst_zero_duplicate_repairs": audit.get(
            "dst_zero_duplicate_repairs"
        ),
        "fill_or_interpolation": audit.get("fill_or_interpolation"),
    }


def build_catalog(
    plans: Sequence[SeriesPlan],
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    series_workers: int,
    day_workers: int,
) -> dict[str, Any]:
    entries = [_catalog_entry(plan) for plan in plans]
    artifact_start_day = min(entry["start_day"] for entry in entries)
    artifact_end_day = max(entry["end_day"] for entry in entries)
    lab_sources: list[dict[str, Any]] = []
    live_sources: dict[str, dict[str, Any]] = {}
    for entry in entries:
        common = {
            "path": entry["path"],
            "timestamp_column": entry["timestamp_column"],
            "columns": {entry["alias"]: entry["value_column"]},
            "origin_column": entry["origin_column"],
            "revision_column": entry["revision_column"],
            "cutoff_column": entry["cutoff_column"],
            "cutoff_time": entry["cutoff_time"],
        }
        lab_sources.append(dict(common))
        live_sources[entry["alias"]] = {
            **common,
            "information_type": "day_ahead_forecast",
            "cutoff_policy": entry["cutoff_policy"],
            "sha256": entry["sha256"],
        }
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "provider": "Saturn",
        "causal_contract": "latest series state queried as-of D-1 08:00 civil",
        "provenance_semantics": {
            "snapshot_time_utc": "query_asof_cutoff",
            "revision_time_utc": (
                "query_asof_cutoff; not provider insertion timestamp"
            ),
        },
        "actual_weather_used": False,
        # The files can legitimately be strict supersets of a historical
        # request.  Keep the physical artifact bounds authoritative and expose
        # the requested interval separately so the catalog never understates
        # the data that its checksums actually identify.
        "start_day": artifact_start_day,
        "end_day": artifact_end_day,
        "requested_start_day": start_day.date().isoformat(),
        "requested_end_day": end_day.date().isoformat(),
        "zones": list(dict.fromkeys(entry["zone"] for entry in entries)),
        "parallelism": {
            "series_workers": series_workers,
            "day_workers": day_workers,
            "maximum_concurrent_day_requests": series_workers * day_workers,
        },
        "columns": {
            "delivery": "value_time_utc",
            "value": "value",
            "origin": "snapshot_time_utc",
            "revision": "revision_time_utc",
            "cutoff": "snapshot_time_utc",
        },
        "series": entries,
        "additional_sources_snippets": {
            "auxiliary_lab": {"additional_sources": lab_sources},
            "operational_live": {"additional_sources": live_sources},
        },
    }


def write_catalog(catalog: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / CATALOG_JSON
    yaml_path = output / CATALOG_YAML
    json_path.write_text(
        json.dumps(
            catalog,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    yaml_path.write_text(
        yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return json_path, yaml_path


def _run_one(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True, cwd=ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start_day = _normalise_day(args.start_day, name="--start-day")
    end_day = _normalise_day(args.end_day, name="--end-day")
    if end_day < start_day:
        raise ValueError("--end-day doit etre posterieur ou egal a --start-day.")
    validate_parallelism(args.series_workers, args.day_workers)
    output_dir = Path(args.output_dir).expanduser().resolve()
    plans = build_plan(args.zones, output_dir)
    if not args.dry_run:
        reject_in_place_semantic_rewrite(plans)
    pending: list[tuple[SeriesPlan, list[str]]] = []
    for plan in plans:
        reusable, reason = reusable_output(
            plan,
            start_day=start_day,
            end_day=end_day,
        )
        status = "SKIP" if reusable else "BUILD"
        extension_start: pd.Timestamp | None = None
        if not reusable:
            extension_start, extension_reason = incremental_extension_start(
                plan,
                start_day=start_day,
                end_day=end_day,
            )
            if extension_start is not None:
                status = "EXTEND"
                reason = extension_reason
        print(f"[{plan.zone}] {plan.alias} | {status} | {reason}", flush=True)
        if not reusable:
            pending.append(
                (
                    plan,
                    build_command(
                        plan,
                        start_day=extension_start or start_day,
                        end_day=end_day,
                        day_workers=args.day_workers,
                        merge_existing=extension_start is not None,
                    ),
                )
            )
    if args.dry_run:
        for _, command in pending:
            print(json.dumps(command, ensure_ascii=False), flush=True)
        print(
            f"DRY-RUN | series={len(plans)} | a_construire={len(pending)}",
            flush=True,
        )
        return 0

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(args.series_workers, len(pending) or 1)) as pool:
        future_to_plan = {
            pool.submit(_run_one, command): plan for plan, command in pending
        }
        for future in as_completed(future_to_plan):
            plan = future_to_plan[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{plan.alias}: {exc}")
    if failures:
        raise RuntimeError(
            "Echec de materialisation Saturn:\n" + "\n".join(failures)
        )

    for plan in plans:
        valid, reason = reusable_output(
            plan,
            start_day=start_day,
            end_day=end_day,
        )
        if not valid:
            raise RuntimeError(f"Sortie finale invalide {plan.alias}: {reason}.")
    catalog = build_catalog(
        plans,
        start_day=start_day,
        end_day=end_day,
        series_workers=args.series_workers,
        day_workers=args.day_workers,
    )
    json_path, yaml_path = write_catalog(catalog, output_dir)
    print(f"Catalogue JSON: {json_path}", flush=True)
    print(f"Catalogue YAML: {yaml_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CATALOG_JSON",
    "CATALOG_YAML",
    "DEFAULT_OUTPUT_DIR",
    "MATERIALIZER",
    "MAX_TOTAL_WORKERS",
    "SeriesPlan",
    "ZONE_TIMEZONES",
    "build_catalog",
    "build_command",
    "build_plan",
    "incremental_extension_start",
    "main",
    "normalise_zones",
    "parse_args",
    "reusable_output",
    "reject_in_place_semantic_rewrite",
    "validate_parallelism",
    "write_catalog",
]
