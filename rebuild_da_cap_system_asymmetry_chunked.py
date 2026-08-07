#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

import build_da_cap_system_asymmetry as base
from chronos2_modular.common import load_yaml

PIT_COLUMNS = [
    "value_time_utc",
    "snapshot_time_utc",
    "revision_time_utc",
    "value",
]


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def parquet_delivery_bounds(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table(columns=["value_time_utc"])
    values = pd.to_datetime(
        table.column("value_time_utc").to_pandas(),
        errors="coerce",
        utc=True,
    ).dropna()
    del table
    gc.collect()
    if values.empty:
        raise ValueError(f"Aucune livraison valide dans {path}")
    return _utc(values.min()), _utc(values.max())


def read_window(
    path: Path,
    alias: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> pd.DataFrame:
    dataset = ds.dataset(str(path), format="parquet")
    field_type = dataset.schema.field("value_time_utc").type
    predicate = (
        (ds.field("value_time_utc") >= pa.scalar(start_utc.to_pydatetime(), type=field_type))
        & (ds.field("value_time_utc") < pa.scalar(end_utc.to_pydatetime(), type=field_type))
    )
    table = dataset.to_table(columns=PIT_COLUMNS, filter=predicate)
    frame = table.to_pandas()
    del table
    gc.collect()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "value_time_utc",
                "event_time_utc",
                "alias",
                "component_value_mw",
            ]
        )
    event_time = pd.concat(
        [
            pd.to_datetime(frame["snapshot_time_utc"], errors="coerce", utc=True),
            pd.to_datetime(frame["revision_time_utc"], errors="coerce", utc=True),
        ],
        axis=1,
    ).max(axis=1)
    events = pd.DataFrame(
        {
            "value_time_utc": pd.to_datetime(
                frame["value_time_utc"], errors="coerce", utc=True
            ),
            "event_time_utc": event_time,
            "alias": alias,
            "component_value_mw": pd.to_numeric(frame["value"], errors="coerce"),
        }
    ).dropna(
        subset=[
            "value_time_utc",
            "event_time_utc",
            "component_value_mw",
        ]
    )
    del frame
    gc.collect()
    return (
        events.sort_values(["value_time_utc", "event_time_utc"])
        .drop_duplicates(
            ["value_time_utc", "event_time_utc", "alias"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def iter_chunks(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    chunk_days: int,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = start_utc.floor("D")
    stop = end_utc.ceil("D") + pd.Timedelta(days=1)
    while cursor < stop:
        next_cursor = min(cursor + pd.Timedelta(days=chunk_days), stop)
        yield cursor, next_cursor
        cursor = next_cursor


def build_chunked(
    raw_dir: Path,
    output_path: Path,
    *,
    output_unit: str,
    chunk_days: int,
) -> pd.DataFrame:
    stores = base.component_store_paths(raw_dir)
    missing = [str(path) for path in stores.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Stores PIT manquants :\n" + "\n".join(missing))

    bounds = {alias: parquet_delivery_bounds(path) for alias, path in stores.items()}
    common_start = max(value[0] for value in bounds.values())
    common_end = min(value[1] for value in bounds.values())
    if common_start > common_end:
        raise ValueError("Les cinq séries n'ont aucune période commune.")

    chunks = list(iter_chunks(common_start, common_end, chunk_days))
    derived_parts: list[pd.DataFrame] = []

    print(f"Période commune : {common_start} -> {common_end}")
    print(f"Découpage       : {len(chunks)} blocs de {chunk_days} jours maximum")

    for number, (start_utc, end_utc) in enumerate(chunks, start=1):
        print(
            f"[{number:03d}/{len(chunks):03d}] "
            f"{start_utc.date()} -> {(end_utc - pd.Timedelta(seconds=1)).date()}",
            flush=True,
        )
        component_frames = {}
        incomplete = False
        for alias, path in stores.items():
            events = read_window(path, alias, start_utc, end_utc)
            if events.empty:
                print(f"  bloc ignoré : {alias} est vide", flush=True)
                incomplete = True
                break
            component_frames[alias] = events

        if incomplete:
            del component_frames
            gc.collect()
            continue

        try:
            part = base.derive_asymmetry_vintages(
                component_frames,
                output_unit=output_unit,
            )
        except ValueError as exc:
            print(f"  bloc ignoré : {exc}", flush=True)
        else:
            if not part.empty:
                derived_parts.append(part)
                print(f"  {len(part):,} révisions dérivées", flush=True)

        del component_frames
        gc.collect()

    if not derived_parts:
        raise ValueError("Aucune asymétrie n'a pu être reconstruite.")

    result = base._normalize_output_vintage(
        pd.concat(derived_parts, ignore_index=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    metadata = {
        "alias": base.DERIVED_ALIAS,
        "unit": output_unit.upper(),
        "chunk_days": chunk_days,
        "formula": (
            "ES->FR + IT_NORTH->FR + UK->FR "
            "- FR->IT_NORTH - FR->UK"
        ),
        "rows": int(len(result)),
        "delivery_hours": int(result["value_time_utc"].nunique()),
        "first_delivery_utc": str(result["value_time_utc"].min()),
        "last_delivery_utc": str(result["value_time_utc"].max()),
        "first_revision_utc": str(result["revision_time_utc"].min()),
        "last_revision_utc": str(result["revision_time_utc"].max()),
    }
    output_path.with_name(
        f"{output_path.stem}_chunked_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruit da_cap_system_asymmetry par blocs sans charger "
            "les cinq Parquet complets en mémoire."
        )
    )
    parser.add_argument("--config", default="chronos2_selected_core.yaml")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--unit", choices=("GW", "MW"), default="GW")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_days < 1:
        raise ValueError("--chunk-days doit être >= 1")

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    default_raw, default_output = base.resolve_default_paths(
        config,
        config_path.parent,
    )
    raw_dir = (
        Path(args.raw_dir).expanduser().resolve()
        if args.raw_dir
        else default_raw
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output
    )

    result = build_chunked(
        raw_dir,
        output_path,
        output_unit=args.unit,
        chunk_days=args.chunk_days,
    )

    print("\nSérie dérivée construite :")
    print(f"  fichier    : {output_path}")
    print(f"  lignes     : {len(result):,}")
    print(f"  livraisons : {result['value_time_utc'].nunique():,}")
    print(
        "  période    : "
        f"{result['value_time_utc'].min()} -> "
        f"{result['value_time_utc'].max()}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
