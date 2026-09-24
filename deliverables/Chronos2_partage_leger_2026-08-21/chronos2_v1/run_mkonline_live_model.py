#!/usr/bin/env python
"""Run one strict non-FR live bundle through the multizone engine."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from chronos2_hourly.multizone_contract import load_zone_model_contract
from chronos2_hourly.multizone_live import LiveRuntimeOptions, run_zone_live
from chronos2_modular.common import load_yaml


DEFAULT_REGISTRY = "chronos2_hourly_live_zones.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict generic day-ahead runner for BE/DE/NL/ES bundles."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--data-as-of", default=None)
    parser.add_argument("--delivery-day", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--pit-replay", action="store_true")
    parser.add_argument(
        "--rolling365-capture-root",
        default=None,
        help=(
            "Racine separee de capture causale prospective rolling-365. "
            "Desactivee par defaut et executee seulement apres publication "
            "du forecast officiel."
        ),
    )
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
    config_path = Path(args.config).expanduser().resolve()
    registry_path = Path(args.registry).expanduser().resolve()
    contract = load_zone_model_contract(
        config_path,
        registry_path,
        strict=True,
    )
    config = load_yaml(config_path)
    live = config.get("live")
    if not isinstance(live, dict):
        raise TypeError("live must be a mapping")
    report = config.get("report")
    if report is not None:
        if not isinstance(report, dict):
            raise TypeError("report must be a mapping")
        live = dict(live)
        live["report"] = dict(report)
    options = LiveRuntimeOptions(
        device=args.device,
        threads=int(args.threads if args.threads is not None else live.get("threads", -1)),
        workers=int(args.workers if args.workers is not None else live.get("workers", 8)),
        local_files_only=bool(args.local_files_only),
        rolling365_capture_root=(
            Path(args.rolling365_capture_root).expanduser().resolve()
            if args.rolling365_capture_root
            else None
        ),
    )
    output = run_zone_live(
        contract,
        live_settings=live,
        data_as_of=args.data_as_of,
        delivery_day=args.delivery_day,
        output_dir=args.output_dir,
        pit_replay=bool(args.pit_replay),
        options=options,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
