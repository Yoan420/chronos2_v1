#!/usr/bin/env python
"""CLI for atomic per-zone copies of a trained multi-zone LoRA artefact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chronos2_exogenous.zone_artifacts import prepare_zone_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copie physiquement un artefact LoRA d'entrainement vierge vers "
            "des runs isoles zones/<ZONE>/artifact, sans modifier la source."
        )
    )
    parser.add_argument(
        "--source-run-directory",
        required=True,
        type=Path,
        help="Artefact LoRA termine, verifie et encore non evalue.",
    )
    parser.add_argument(
        "--zones",
        nargs="+",
        required=True,
        help="Zones explicites presentes dans le panel d'entrainement.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Racine avant zones/<ZONE>/artifact; par defaut le parent de "
            "l'artefact source."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_zone_artifacts(
        args.source_run_directory,
        zones=args.zones,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "source_directory": str(result.source_directory),
                "output_root": str(result.output_root),
                "source_tree_sha256": result.source_tree_sha256,
                "artifacts": [
                    {
                        "zone": artifact.zone,
                        "path": str(artifact.path),
                        "manifest": str(artifact.manifest_path),
                        "created": artifact.created,
                    }
                    for artifact in result.artifacts
                ],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
