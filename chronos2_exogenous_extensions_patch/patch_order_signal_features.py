#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path


def main() -> int:
    path = Path("chronos2_order_signals/features.py")
    text = path.read_text(encoding="utf-8")
    needle = (
        '    excluded_prefixes = (\n'
        '        "known_hour_",\n'
        '        "known_dow_",\n'
        '        "known_doy_",\n'
        '        "known_is_weekend",\n'
        '    )'
    )
    replacement = (
        '    excluded_prefixes = (\n'
        '        "known_hour_",\n'
        '        "known_dow_",\n'
        '        "known_doy_",\n'
        '        "known_is_weekend",\n'
        '        "known_cal_",\n'
        '    )'
    )
    if replacement in text:
        print("features.py déjà corrigé.")
        return 0
    if needle not in text:
        raise RuntimeError(
            "Bloc excluded_prefixes non reconnu dans features.py."
        )
    path.write_text(
        text.replace(needle, replacement),
        encoding="utf-8",
        newline="\n",
    )
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
