#!/usr/bin/env python
"""Publish one offline, sealed DE/BE/NL/ES hourly benchmark."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from chronos2_hourly.zone_benchmark import publish_zone_benchmark


LOGGER = logging.getLogger("zone_benchmark_hourly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publie un benchmark zonal scellé depuis des artefacts déjà "
            "matérialisés. Aucun appel Saturn et aucun fallback implicite."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    output = publish_zone_benchmark(
        args.config,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    manifest = output / "run_manifest.json"
    LOGGER.info("Benchmark zonal scellé : %s", output)
    print(f"Run horaire : {output}")
    print(f"Manifest : {manifest}")
    reports = sorted(Path(output).glob("*.html"))
    if reports:
        print(f"Rapport HTML : {reports[0]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("Zone benchmark runner failed: %s", exc)
        raise SystemExit(1)
