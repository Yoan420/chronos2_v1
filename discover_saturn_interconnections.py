from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from chronos2_modular.saturn import create_saturn_client


KEYWORDS = (
    "capacity",
    "ntc",
    "atc",
    "ram",
    "transfer",
    "interconnector",
    "crossborder",
    "cross-border",
)


def flatten_names(value: Any) -> list[str]:
    names: list[str] = []

    if isinstance(value, str):
        return [value]

    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and "." in key:
                names.append(key)
            names.extend(flatten_names(child))
        return names

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes)
    ):
        for child in value:
            names.extend(flatten_names(child))

    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="chronos2_m1_calendar.yaml",
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data = config.get("data", {})
    client = create_saturn_client(
        str(data.get("saturn_url", "")),
        str(data.get("saturn_author", "")),
    )

    catalog_method = getattr(client, "catalog", None)
    if catalog_method is None:
        raise RuntimeError(
            "Cette version de tshistory_lite n'expose pas catalog()."
        )

    attempts = (
        {},
        {"allsources": True},
        {"allsources": False},
    )
    catalog = None
    errors: list[str] = []
    for kwargs in attempts:
        try:
            catalog = catalog_method(**kwargs)
            if catalog is not None:
                break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    if catalog is None:
        raise RuntimeError(
            "Impossible de lire le catalogue Saturn. "
            + " | ".join(errors)
        )

    names = sorted(set(flatten_names(catalog)))
    candidates = [
        name
        for name in names
        if "fr" in name.lower()
        and any(keyword in name.lower() for keyword in KEYWORDS)
    ]

    output = Path("saturn_interconnection_candidates.csv")
    pd.DataFrame({"series": candidates}).to_csv(
        output, index=False
    )
    print(f"{len(candidates)} candidats écrits dans {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
