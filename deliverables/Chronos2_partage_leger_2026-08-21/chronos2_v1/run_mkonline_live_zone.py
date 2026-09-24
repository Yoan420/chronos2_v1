#!/usr/bin/env python
"""Preflight and dispatch a zone-specific day-ahead live bundle.

This script performs no Saturn access during preflight.  It never substitutes
France when the requested zone is incomplete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from chronos2_hourly.zone_live import (
    audit_zone_live_bundle,
    build_zone_runner_command,
    load_zone_registry,
    strict_contract_preflight,
)


DEFAULT_REGISTRY = "chronos2_hourly_live_zones.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatcher live day-ahead multi-zone avec preflight local."
    )
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--zone", required=True, help="FR, BE, DE, ES, NL, GB/UK ou IT")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--data-as-of", default=None)
    parser.add_argument("--delivery-day", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--pit-replay", action="store_true")
    parser.add_argument(
        "--no-rolling365-capture",
        action="store_true",
        help=(
            "Desactive explicitement la capture prospective rolling-365. "
            "Par defaut, chaque vrai run live publie d'abord le forecast "
            "officiel puis alimente runs/rolling365_shadow."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry, registry_dir = load_zone_registry(args.registry)
    audit = audit_zone_live_bundle(
        registry,
        zone=args.zone,
        registry_dir=registry_dir,
    )
    audit = strict_contract_preflight(
        audit,
        registry_path=Path(args.registry).expanduser().resolve(),
    )
    print(json.dumps(audit.as_dict(), ensure_ascii=False, indent=2))
    audit.require_ready()
    if args.preflight_only:
        return 0
    command = build_zone_runner_command(
        audit,
        data_as_of=args.data_as_of,
        delivery_day=args.delivery_day,
        output_dir=args.output_dir,
        device=args.device,
        threads=args.threads,
        workers=args.workers,
        local_files_only=args.local_files_only,
        pit_replay=args.pit_replay,
    )
    runner = Path(command[1]).resolve()
    if runner.name == "run_mkonline_live_model.py":
        command.extend(["--registry", str(Path(args.registry).expanduser().resolve())])
    if not args.no_rolling365_capture and not args.pit_replay:
        command.extend(
            [
                "--rolling365-capture-root",
                str((registry_dir / "runs" / "rolling365_shadow").resolve()),
            ]
        )
    if Path(command[1]).resolve() == Path(__file__).resolve():
        raise RuntimeError("Le registre ne peut pas dispatcher vers lui-meme.")
    subprocess.run(command, check=True, cwd=registry_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
