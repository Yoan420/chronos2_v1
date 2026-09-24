#!/usr/bin/env python
"""CLI for fail-closed recovery of an evaluated LoRA training snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chronos2_exogenous.training_snapshot_recovery import (
    recover_training_snapshot_and_prepare_zones,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authentifie un backtest LoRA brut deja publie, reconstruit un "
            "snapshot training-only avec filiation SHA-256, puis prepare des "
            "copies physiques isolees par zone. Aucun overwrite n'est permis."
        )
    )
    parser.add_argument("--source-run-directory", type=Path, required=True)
    parser.add_argument(
        "--config-reference",
        type=Path,
        required=True,
        help=(
            "Reference YAML semantiquement identique au training. Elle est "
            "archivee comme reference post-hoc, jamais presentee comme les "
            "octets originaux si le trainer ne les avait pas sauvegardes."
        ),
    )
    parser.add_argument("--zones", nargs="+", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Racine du snapshot et de zones/<ZONE>/artifact. Par defaut: "
            "<parent-source>/recovered_training_snapshot."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = recover_training_snapshot_and_prepare_zones(
        args.source_run_directory,
        config_reference=args.config_reference,
        zones=args.zones,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "source_directory": str(result.source_directory),
                "source_tree_sha256": result.source_tree_sha256,
                "snapshot_directory": str(result.snapshot_directory),
                "recovery_manifest": str(result.recovery_manifest_path),
                "output_root": str(result.output_root),
                "snapshot_created": result.created,
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
