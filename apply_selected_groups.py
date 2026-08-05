#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"YAML invalide : {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="chronos2_inputs_extended_exogenous.yaml",
    )
    parser.add_argument("--groups", default="feature_groups.yaml")
    parser.add_argument(
        "--selected",
        required=True,
        help="YAML produit par summarize_feature_selection.py.",
    )
    parser.add_argument(
        "--level",
        choices=("family", "component"),
        default="family",
    )
    parser.add_argument(
        "--output",
        default="chronos2_inputs_selected.yaml",
    )
    args = parser.parse_args()

    config = load_yaml(Path(args.config).expanduser().resolve())
    spec = load_yaml(Path(args.groups).expanduser().resolve())
    selected_payload = load_yaml(Path(args.selected).expanduser().resolve())
    selected_subjects = [
        str(x) for x in selected_payload.get("recommended_subjects", [])
    ]

    components = spec.get("components", {}) or {}
    if args.level == "component":
        selected_components = selected_subjects
    else:
        families = spec.get("families", {}) or {}
        selected_components = []
        for family in selected_subjects:
            if family not in families:
                raise KeyError(f"Famille inconnue : {family}")
            for component in families[family].get("components", []):
                if component not in selected_components:
                    selected_components.append(component)

    config["feature_selection"] = {
        "enabled": True,
        "selected_groups": selected_components,
        "group_definitions": components,
    }
    config.setdefault("output", {})["directory"] = "runs/chronos2_selected"
    config.setdefault("report", {})["filename"] = "chronos2_selected.html"
    config["report"]["title"] = "Chronos-2 — variables sélectionnées"

    output = Path(args.output).expanduser().resolve()
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
