#!/usr/bin/env python
"""Audit, without network access, the strict 365+365+365 OOF history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chronos2_exogenous.history_recovery import audit_local_history
from run_chronos2_exogenous_panel import _canonical_target_path


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audite localement l'historique requis par le correcteur LoRA OOF."
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--zones", nargs="+", default=["FR", "DE", "BE", "NL"])
    parser.add_argument("--pack", default="full")
    parser.add_argument("--end-day", required=True)
    parser.add_argument("--training-days", type=int, default=365)
    parser.add_argument("--oof-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=365)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    zones = tuple(dict.fromkeys(str(zone).strip().upper() for zone in args.zones))
    target_paths = {zone: _canonical_target_path(root, zone)[0] for zone in zones}
    payload = audit_local_history(
        root,
        target_paths=target_paths,
        end_day=args.end_day,
        zones=zones,
        pack=args.pack,
        training_days=args.training_days,
        oof_days=args.oof_days,
        holdout_days=args.holdout_days,
        context_length=args.context_length,
        timezone=args.timezone,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.expanduser()
        if not output.is_absolute():
            output = root / output
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)
        print(f"Audit: {output}")
    print(rendered)
    return 0 if payload["research_horizon_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
