#!/usr/bin/env python
"""Read-only DST diagnosis for a Saturn nuclear-generation series."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import tshistory_lite
import yaml


LOCAL_TZ = "Europe/Paris"
TRANSITIONS = {
    "2022-03-27": "spring",
    "2022-10-30": "fall",
    "2023-03-26": "spring",
    "2023-10-29": "fall",
    "2024-03-31": "spring",
    "2024-10-27": "fall",
    "2025-03-30": "spring",
    "2025-10-26": "fall",
}


def _raw_get(client: Any, name: str) -> pd.Series:
    start = pd.Timestamp("2022-03-26 00:00", tz=LOCAL_TZ)
    end = pd.Timestamp("2025-10-27 23:00", tz=LOCAL_TZ)
    revision = pd.Timestamp.now(tz="UTC")
    errors: list[str] = []
    for date_kwargs in (
        {"from_value_date": start, "to_value_date": end},
        {"from_value": start, "to_value": end},
        {"start": start, "end": end},
    ):
        try:
            raw = client.get(name, revision_date=revision, **date_kwargs)
            if raw is None or len(raw) == 0:
                continue
            if isinstance(raw, pd.DataFrame):
                if raw.shape[1] != 1:
                    raise ValueError(f"payload DataFrame ambigu: {list(raw.columns)}")
                raw = raw.iloc[:, 0]
            return pd.Series(raw)
        except Exception as exc:  # diagnostics: retain every client signature error
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("aucune donnée Saturn: " + " | ".join(errors))


def _flatten_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, child in value.items():
            if isinstance(key, str) and "." in key:
                result.append(key)
            result.extend(_flatten_names(child))
        return result
    if isinstance(value, pd.DataFrame):
        return [str(item) for item in value.to_numpy().ravel() if isinstance(item, str)]
    if isinstance(value, (pd.Index, pd.Series)):
        return [str(item) for item in value if isinstance(item, str)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for child in value:
            result.extend(_flatten_names(child))
        return result
    return []


def _catalog(client: Any) -> list[str]:
    method = getattr(client, "catalog", None)
    if method is None:
        return []
    for kwargs in ({}, {"allsources": True}, {"allsources": False}):
        try:
            catalog = method(**kwargs)
            if catalog is not None:
                names = set(_flatten_names(catalog))
                return sorted(
                    name
                    for name in names
                    if "nuclear" in name.lower()
                    and (
                        ".fr." in f".{name.lower()}."
                        or name.lower().startswith("power.fr.")
                    )
                    and ("entsoe" in name.lower() or "utc" in name.lower())
                )
        except Exception:
            continue
    return []


def _transition_table(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(LOCAL_TZ) if index.tz is not None else index
    rows = []
    for day, season in TRANSITIONS.items():
        block = local[local.strftime("%Y-%m-%d") == day]
        wall = block.tz_localize(None) if block.tz is not None else block
        rows.append(
            {
                "day": day,
                "transition": season,
                "rows": len(block),
                "hour_02": int((wall.hour == 2).sum()),
                "duplicate_wall_labels": int(wall.duplicated(keep=False).sum()),
            }
        )
    return pd.DataFrame(rows)


def _classification(index: pd.DatetimeIndex, table: pd.DataFrame, name: str) -> str:
    # Ignore transition dates outside the actual history returned by Saturn.
    # At least one observed spring and autumn transition is still required to
    # distinguish a genuine local clock from a regular UTC-naive grid.
    spring = table.loc[
        (table["transition"] == "spring") & (table["rows"] > 0)
    ]
    fall = table.loc[
        (table["transition"] == "fall") & (table["rows"] > 0)
    ]
    if spring.empty or fall.empty:
        return "HISTORIQUE_INSUFFISANT_POUR_CLASSER_DST"
    physical = bool(
        (spring["rows"] == 23).all()
        and (spring["hour_02"] == 0).all()
        and (fall["rows"] == 25).all()
        and (fall["hour_02"] == 2).all()
    )
    if index.tz is not None:
        return "TIMELINE_TZ_AWARE_PHYSIQUE" if physical else "TIMELINE_TZ_AWARE_INCOMPLETE"
    if physical:
        return "LOCAL_NAIF_DST_23_25"

    spring_23 = bool((spring["rows"] == 23).all() and (spring["hour_02"] == 0).all())
    fall_24 = bool((fall["rows"] == 24).all() and (fall["hour_02"] == 1).all())
    if spring_23 and fall_24:
        return "LOCAL_NAIF_23_24_PERTE_DU_SECOND_FOLD_AUTOMNE"

    regular_24 = bool((table["rows"] == 24).all() and (table["hour_02"] == 1).all())
    if regular_24:
        # Timestamps alone cannot prove which clock was used.  An explicit
        # UTC series name/metadata is required to resolve this case safely.
        if ".utc." in f".{name.lower()}.":
            return "UTC_NAIF_CONTINU_EXPLICITE_DANS_LE_NOM"
        return "NAIF_REGULIER_24_AMBIGU_UTC_OU_LOCAL_COMPRESSE"
    return "FORME_DST_INCOMPLETE_OU_IRREGULIERE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="chronos2_hourly_fr.yaml")
    parser.add_argument(
        "--series",
        default="power.fr.generation.nuclear.entsoe.hourly.gw.obs",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    data = config["data"]
    author = os.getenv("SATURN_AUTHOR") or str(data["saturn_author"])
    client = tshistory_lite.Client(uri=str(data["saturn_url"]), author=author)

    raw = _raw_get(client, args.series)
    index = pd.DatetimeIndex(pd.to_datetime(raw.index, errors="coerce"))
    index = index[~index.isna()].sort_values()
    table = _transition_table(index)

    print(f"series = {args.series}")
    print(f"rows = {len(index)}")
    print(f"first = {index.min() if len(index) else None}")
    print(f"last = {index.max() if len(index) else None}")
    print(f"index_timezone = {index.tz}")
    print(f"duplicates_raw = {int(index.duplicated(keep=False).sum())}")
    if index.tz is None and len(index) > 1:
        deltas = index.to_series().diff().dropna()
        print(f"zero_steps = {int((deltas == pd.Timedelta(0)).sum())}")
        print(f"steps_gt_1h = {int((deltas > pd.Timedelta(hours=1)).sum())}")
    print(table.to_string(index=False))
    print(f"classification = {_classification(index, table, args.series)}")

    print("\n=== CANDIDATS NUCLEAIRE FR ENTSO-E / UTC ===")
    candidates = _catalog(client)
    if candidates:
        for candidate in sorted(candidates, key=lambda value: ("utc" not in value.lower(), value)):
            print(candidate)
    else:
        print("Aucun candidat trouvé (ou méthode catalog indisponible).")

    print(
        "\nNOTE: NAIF_REGULIER_24 est indécidable avec les timestamps seuls. "
        "Ne pas forcer naive_timezone: UTC sans série/metadata explicitement UTC."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
