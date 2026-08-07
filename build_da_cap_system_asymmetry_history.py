#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transforme le store dérivé PIT en historique horaire final "
            "utilisable avec cutoff_persistence."
        )
    )
    parser.add_argument(
        "--input",
        default="data/pit/vintages/da_cap_system_asymmetry.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/derived/da_cap_system_asymmetry_history.csv.gz",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(source)

    frame = pd.read_parquet(
        source,
        columns=[
            "value_time_utc",
            "snapshot_time_utc",
            "revision_time_utc",
            "value",
        ],
    )

    for column in (
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
    ):
        frame[column] = pd.to_datetime(
            frame[column], errors="coerce", utc=True
        )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    final = (
        frame.dropna(subset=["value_time_utc", "value"])
        .sort_values(
            [
                "value_time_utc",
                "snapshot_time_utc",
                "revision_time_utc",
            ]
        )
        .drop_duplicates("value_time_utc", keep="last")
        .loc[:, ["value_time_utc", "value"]]
        .rename(columns={"value_time_utc": "timestamp"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(destination, index=False, compression="gzip")

    print(f"Historique créé : {destination}")
    print(f"Lignes          : {len(final):,}")
    print(
        "Période         : "
        f"{final['timestamp'].min()} -> {final['timestamp'].max()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
