#!/usr/bin/env python
"""Synchronise les fondamentaux horaires FR candidats avec leurs vintages PIT.

Les fichiers produits conservent ``value_time_utc`` et
``revision_time_utc``. Le pipeline peut ainsi sélectionner, pour chaque jour
de livraison, uniquement la dernière prévision publiée avant D-1 08:00.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

import pandas as pd

from chronos2_modular.saturn import (
    create_saturn_client,
    fetch_saturn_series_from_client,
    normalize_vintage_frame,
    read_vintage_store,
    sync_vintage_series,
)


SATURN_HOURLY_CANDIDATES: dict[str, str] = {
    "fr_load_fcst": "power.fr.load.hourly.gw.fcst",
    "fr_wind_generation_fcst": "power.fr.generation.wind.hourly.gw.fcst",
    "fr_solar_generation_fcst": "power.fr.generation.solar.hourly.gw.fcst",
    "fr_hydro_ror_generation_fcst": (
        "power.fr.gma.generation.hydro.ror.hourly.gw.fcst"
    ),
    "fr_nuclear_generation_fcst_long": "power.fr.generation.nuclear.gw.fcst",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge les vintages PIT de fondamentaux FR depuis Saturn."
    )
    parser.add_argument(
        "--saturn-url",
        default="https://saturn-energyscan.gem.myengie.com//api",
    )
    parser.add_argument("--author", default="BQ6757")
    parser.add_argument("--output-dir", default="data/pit/vintages")
    parser.add_argument(
        "--parts-tag",
        default="",
        help=(
            "Suffixe optionnel du dossier de morceaux; permet plusieurs "
            "workers paralleles sans collision de manifest ni de fichiers."
        ),
    )
    parser.add_argument("--revision-start", default="2024-07-01T00:00:00Z")
    parser.add_argument("--revision-end", default=None)
    parser.add_argument("--value-start", default="2024-07-01T00:00:00Z")
    parser.add_argument("--value-end", default=None)
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--chunk-days", type=int, default=90)
    parser.add_argument(
        "--persist-days",
        type=int,
        default=30,
        help=(
            "Nombre de jours de revisions traites avant chaque ecriture "
            "atomique du parquet. Une valeur <= 0 conserve le mode "
            "monolithique."
        ),
    )
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help=(
            "Ecrit chaque segment dans un petit parquet independant, puis "
            "compacte une seule fois a la fin."
        ),
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help=(
            "En mode partitionne, conserve les morceaux sans construire "
            "le parquet final. Utile pour paralleliser les telechargements."
        ),
    )
    parser.add_argument(
        "--local-value-window-days",
        type=int,
        default=0,
        help=(
            "Si >0, borne les dates de livraison de chaque segment a "
            "[debut_revision-2j, fin_revision+N jours] au lieu de relire "
            "toute la plage globale."
        ),
    )
    parser.add_argument(
        "--daily-cutoff",
        action="store_true",
        help=(
            "Telecharge directement un snapshot a D-1 08:00 pour chaque "
            "jour de livraison, au lieu de l'historique complet."
        ),
    )
    parser.add_argument(
        "--aliases",
        nargs="+",
        choices=tuple(SATURN_HOURLY_CANDIDATES),
        default=None,
        help="Sous-ensemble de séries à synchroniser.",
    )
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def _utc(value: str | None, *, default: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value) if value is not None else pd.Timestamp(default)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _compact_vintage_parts(parts_dir: Path, path: Path) -> int:
    """Fusionne atomiquement les morceaux deja dedupliques d'une serie."""
    parts = sorted(parts_dir.glob("*.parquet"))
    if not parts:
        return 0
    frames = [pd.read_parquet(part) for part in parts]
    if path.exists():
        frames.append(read_vintage_store(path))
    merged = normalize_vintage_frame(pd.concat(frames, ignore_index=True))
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    merged.to_parquet(temporary, index=False)
    pd.read_parquet(temporary, columns=["value_time_utc"]).head(1)
    temporary.replace(path)
    return int(len(merged))


def _sync_daily_cutoffs(
    client: object,
    *,
    alias: str,
    series: str,
    output_dir: Path,
    parts_tag: str,
    value_start: pd.Timestamp,
    value_end: pd.Timestamp,
    timezone: str,
) -> dict[str, object]:
    first_day = value_start.tz_convert(timezone).normalize()
    last_day = value_end.tz_convert(timezone).normalize()
    daily_parts_dir = output_dir / ".daily" / (
        alias + (f"__{parts_tag}" if parts_tag else "")
    )
    daily_parts_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failures: list[str] = []

    for delivery_day in pd.date_range(
        first_day,
        last_day,
        freq="D",
        tz=timezone,
    ):
        part = daily_parts_dir / f"{delivery_day.strftime('%Y%m%d')}.parquet"
        if part.exists():
            skipped += 1
            continue
        cutoff_local = (
            delivery_day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
        )
        series_at_cutoff = None
        error = None
        for attempt in range(3):
            try:
                # Les bornes larges neutralisent toute ambiguite UTC/local;
                # le filtrage final conserve exactement le jour civil D.
                series_at_cutoff = fetch_saturn_series_from_client(
                    client,
                    series,
                    delivery_day - pd.Timedelta(hours=8),
                    delivery_day + pd.DateOffset(days=1, hours=8),
                    timezone,
                    revision_date=cutoff_local.tz_convert("UTC"),
                    naive_timezone="UTC",
                )
                break
            except Exception as exc:  # reseau Saturn: retries bornes
                error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        if series_at_cutoff is None:
            failures.append(f"{delivery_day.date()}: {error}")
            print(
                f"[Saturn/PIT] {alias}: echec {delivery_day.date()}: {error}",
                flush=True,
            )
            continue
        selected = series_at_cutoff.loc[
            (series_at_cutoff.index >= delivery_day)
            & (series_at_cutoff.index < delivery_day + pd.DateOffset(days=1))
        ]
        if selected.empty:
            failures.append(f"{delivery_day.date()}: snapshot vide")
            continue
        cutoff_utc = cutoff_local.tz_convert("UTC")
        frame = pd.DataFrame(
            {
                "value_time_utc": selected.index.tz_convert("UTC"),
                "snapshot_time_utc": cutoff_utc,
                "revision_time_utc": cutoff_utc,
                "value": selected.to_numpy(dtype=float),
                "downloaded_at_utc": pd.Timestamp.now(tz="UTC"),
            }
        )
        frame = normalize_vintage_frame(frame)
        temporary = part.with_name(f".{part.name}.{uuid.uuid4().hex}.tmp.parquet")
        frame.to_parquet(temporary, index=False)
        temporary.replace(part)
        downloaded += int(len(frame))
        if downloaded % (24 * 30) < 25:
            print(
                f"[Saturn/PIT] {alias}: cutoff {delivery_day.date()}, "
                f"{downloaded} lignes",
                flush=True,
            )

    return {
        "zone": "FR",
        "alias": alias,
        "series": series,
        "kind": "daily_cutoff_snapshots",
        "path": str(daily_parts_dir),
        "status": "updated" if downloaded else "up_to_date",
        "rows_before": skipped,
        "rows_downloaded": downloaded,
        "rows_after": skipped + downloaded,
        "first_revision_utc": None,
        "last_revision_utc": None,
        "sync_as_of_utc": str(pd.Timestamp.now(tz="UTC")),
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    now = pd.Timestamp.now(tz="UTC")
    revision_end = _utc(args.revision_end, default=now)
    value_end = _utc(
        args.value_end,
        default=revision_end + pd.Timedelta(days=3),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    client = create_saturn_client(args.saturn_url, args.author)

    results = []
    selected = args.aliases or list(SATURN_HOURLY_CANDIDATES)
    for alias in selected:
        series = SATURN_HOURLY_CANDIDATES[alias]
        path = output_dir / f"{alias}.parquet"
        parts_alias = alias + (f"__{args.parts_tag}" if args.parts_tag else "")
        parts_dir = output_dir / ".parts" / parts_alias
        if args.partitioned:
            parts_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Saturn/PIT] {alias} <- {series}", flush=True)
        requested_start = _utc(args.revision_start, default=revision_end)
        value_start = _utc(args.value_start, default=revision_end)

        if args.daily_cutoff:
            result_dict = _sync_daily_cutoffs(
                client,
                alias=alias,
                series=series,
                output_dir=output_dir,
                parts_tag=args.parts_tag,
                value_start=value_start,
                value_end=value_end,
                timezone=args.timezone,
            )
            results.append(result_dict)
            continue

        # Reprendre pres de la derniere revision deja durablement ecrite.
        # Cela evite de relire tous les segments anterieurs apres une
        # interruption, tout en gardant deux jours de recouvrement pour les
        # revisions tardives.
        segment_start = requested_start
        if path.exists() and not args.full and not args.partitioned:
            stored_revisions = pd.read_parquet(
                path,
                columns=["revision_time_utc"],
            )["revision_time_utc"]
            last_stored = pd.to_datetime(
                stored_revisions,
                errors="coerce",
                utc=True,
            ).max()
            if pd.notna(last_stored):
                segment_start = max(
                    requested_start,
                    last_stored - pd.Timedelta(days=2),
                )

        persist_days = int(args.persist_days)
        persist_delta = pd.Timedelta(
            days=max(1, persist_days if persist_days > 0 else 10**6)
        )
        segment_results = []
        first_segment = True
        while segment_start <= revision_end:
            segment_end = min(segment_start + persist_delta, revision_end)
            segment_path = path
            if args.partitioned:
                segment_path = parts_dir / (
                    f"{segment_start.strftime('%Y%m%dT%H%M%SZ')}__"
                    f"{segment_end.strftime('%Y%m%dT%H%M%SZ')}.parquet"
                )
                if segment_path.exists() and not args.full:
                    print(
                        f"[Saturn/PIT] {alias}: segment deja present, saute",
                        flush=True,
                    )
                    if segment_end >= revision_end:
                        break
                    segment_start = segment_end
                    first_segment = False
                    continue
            print(
                f"[Saturn/PIT] {alias}: segment revisions "
                f"{segment_start} -> {segment_end}",
                flush=True,
            )
            try:
                segment_value_start = value_start
                segment_value_end = value_end
                if args.local_value_window_days > 0:
                    segment_value_start = max(
                        value_start,
                        segment_start - pd.Timedelta(days=2),
                    )
                    segment_value_end = min(
                        value_end,
                        segment_end
                        + pd.Timedelta(
                            days=int(args.local_value_window_days)
                        ),
                    )
                segment_result = sync_vintage_series(
                    client,
                    zone="FR",
                    alias=alias,
                    series_name=series,
                    path=segment_path,
                    revision_start=segment_start,
                    revision_end=segment_end,
                    value_start=segment_value_start,
                    value_end=segment_value_end,
                    timezone=args.timezone,
                    naive_timezone="UTC",
                    chunk_days=min(
                        max(1, int(args.chunk_days)),
                        max(1, persist_days)
                        if persist_days > 0
                        else int(args.chunk_days),
                    ),
                    full=bool(
                        args.partitioned or (args.full and first_segment)
                    ),
                )
            except RuntimeError as exc:
                if "Aucune revision Saturn recue" not in str(exc) and (
                    "Aucune r\u00e9vision Saturn re\u00e7ue" not in str(exc)
                ):
                    raise
                print(
                    f"[Saturn/PIT] {alias}: segment vide, passe",
                    flush=True,
                )
                first_segment = False
                if segment_end >= revision_end:
                    break
                segment_start = segment_end
                continue
            segment_results.append(segment_result)
            print(
                f"[Saturn/PIT] {alias}: segment "
                f"{segment_result.status}, {segment_result.rows_after} lignes",
                flush=True,
            )
            first_segment = False
            if segment_end >= revision_end:
                break
            # Les bornes Saturn sont inclusives. Le chevauchement exact du
            # point frontiere est volontaire et sera deduplique a l'ecriture.
            segment_start = segment_end

        if not segment_results:
            raise RuntimeError(
                f"Aucun segment a synchroniser pour {alias}: "
                f"{segment_start} > {revision_end}."
            )
        first_result = segment_results[0]
        result = segment_results[-1]
        result_dict = asdict(result)
        if args.partitioned:
            result_dict["path"] = str(path)
            if not args.no_compact:
                result_dict["rows_after"] = _compact_vintage_parts(
                    parts_dir,
                    path,
                )
        result_dict["rows_before"] = first_result.rows_before
        result_dict["rows_downloaded"] = sum(
            item.rows_downloaded for item in segment_results
        )
        result_dict["status"] = (
            "updated"
            if any(item.status == "updated" for item in segment_results)
            else "up_to_date"
        )
        results.append(result_dict)
        print(
            f"[Saturn/PIT] {alias}: {result.status}, "
            f"{result.rows_after} lignes",
            flush=True,
        )

    manifest = output_dir / "hourly_candidate_sync_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "generated_at_utc": str(now),
                "revision_end_utc": str(revision_end),
                "value_end_utc": str(value_end),
                "series": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Manifest : {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
