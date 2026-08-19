#!/usr/bin/env python
"""Inspect raw Saturn target timestamps and discover FR price candidates."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import tshistory_lite
import yaml


FALL_DAYS = (
    "2022-10-30",
    "2023-10-29",
    "2024-10-27",
    "2025-10-26",
)


def _flatten_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    names: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and "." in key:
                names.append(key)
            names.extend(_flatten_names(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            names.extend(_flatten_names(child))
    return names


def _raw_get(client: Any, name: str, start: pd.Timestamp, end: pd.Timestamp):
    as_of = pd.Timestamp.now(tz="UTC")
    attempts = (
        {"from_value_date": start, "to_value_date": end},
        {"from_value": start, "to_value": end},
        {"start": start, "end": end},
    )
    errors: list[str] = []
    for date_kwargs in attempts:
        try:
            raw = client.get(name, **date_kwargs, revision_date=as_of)
            if raw is not None and len(raw):
                return raw, date_kwargs
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("Saturn ne renvoie aucune ligne. " + " | ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="chronos2_hourly_fr.yaml")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    zone = config["zones"]["FR"]
    name = str(zone["target"]["series"])
    timezone = str(zone.get("timezone", "Europe/Paris"))
    author = os.getenv("SATURN_AUTHOR") or str(data["saturn_author"])
    client = tshistory_lite.Client(uri=str(data["saturn_url"]), author=author)

    start = pd.Timestamp("2022-10-29 00:00", tz=timezone)
    end = pd.Timestamp("2025-10-27 23:00", tz=timezone)
    raw, api_kwargs = _raw_get(client, name, start, end)
    if isinstance(raw, pd.DataFrame) and raw.shape[1] == 1:
        raw = raw.iloc[:, 0]

    rows: list[dict[str, Any]] = []
    for position, original in enumerate(raw.index):
        parsed = pd.Timestamp(original)
        local = parsed if parsed.tzinfo is None else parsed.tz_convert(timezone)
        wall = local.strftime("%Y-%m-%d %H:%M:%S")
        if wall[:10] not in FALL_DAYS:
            continue
        value = raw.iloc[position]
        if isinstance(value, pd.Series):
            value = value.to_dict()
        rows.append(
            {
                "position": position,
                "original_timestamp": repr(original),
                "original_timezone": repr(parsed.tzinfo),
                "utc_offset": (
                    None if parsed.tzinfo is None else str(parsed.utcoffset())
                ),
                "fold": getattr(parsed.to_pydatetime(), "fold", None),
                "local_wall_timestamp": wall,
                "value": repr(value),
            }
        )

    report = pd.DataFrame(
        rows,
        columns=(
            "position",
            "original_timestamp",
            "original_timezone",
            "utc_offset",
            "fold",
            "local_wall_timestamp",
            "value",
        ),
    )
    raw_path = output_dir / "saturn_target_dst_raw.csv"
    report.to_csv(raw_path, index=False)

    lines = [
        f"series={name}",
        f"api_kwargs={api_kwargs!r}",
        f"raw_type={type(raw).__name__}",
        f"raw_rows={len(raw)}",
        f"index_type={type(raw.index).__name__}",
        f"index_dtype={getattr(raw.index, 'dtype', None)!r}",
        f"index_timezone={getattr(raw.index, 'tz', None)!r}",
    ]
    for day in FALL_DAYS:
        block = report.loc[
            report["local_wall_timestamp"].str.startswith(day, na=False)
        ]
        counts = Counter(block["local_wall_timestamp"])
        repeated = {stamp: count for stamp, count in counts.items() if count > 1}
        lines.append(f"\n{day}: rows={len(block)}, repeated={repeated}")
        lines.append(block.to_string(index=False))

    text_path = output_dir / "saturn_target_dst_raw.txt"
    text_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    catalog_method = getattr(client, "catalog", None)
    if catalog_method is not None:
        catalog = None
        for kwargs in ({}, {"allsources": True}, {"allsources": False}):
            try:
                catalog = catalog_method(**kwargs)
                if catalog is not None:
                    break
            except Exception:
                continue
        if catalog is not None:
            names = sorted(set(_flatten_names(catalog)))
            candidates = [
                candidate
                for candidate in names
                if "fr" in candidate.lower() and "price" in candidate.lower()
            ]
            candidate_path = output_dir / "saturn_fr_price_candidates.csv"
            pd.DataFrame({"series": candidates}).to_csv(
                candidate_path,
                index=False,
            )
            print(f"\nCatalogue prix FR: {len(candidates)} -> {candidate_path}")

    print(f"\nDiagnostic brut: {raw_path}")
    print(f"Rapport texte: {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
