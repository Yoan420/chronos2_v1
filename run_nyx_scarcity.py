"""Standalone NYX scarcity challenger; never launches or activates Forecast.ps1."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["run", "audit", "refresh", "prepare", "backtest", "report", "status"], default="run")
    parser.add_argument("--config", type=Path, default=Path("config/nyx_scarcity.yaml"))
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--zones", nargs="+")
    parser.add_argument("--delivery-day")
    parser.add_argument("--end-day")
    parser.add_argument("--refresh-sources", action="store_true", help="Refresh known missing source suffixes in private copies only.")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from nyx_scarcity import runner
    try:
        if args.action in {"backtest", "report", "status"} and any(v is not None for v in (args.zones, args.delivery_day, args.end_day)):
            raise ValueError("Backtest/Report/Status use their frozen recipe: no country/date overrides.")
        if args.run_directory is not None and args.action in {"run", "audit", "refresh", "prepare"}:
            raise ValueError("RunDirectory is reserved for Backtest, Report and Status.")
        if args.refresh_sources and args.action not in {"run", "prepare"}:
            raise ValueError("RefreshSources is reserved for Run and Prepare; Audit remains read-only.")
        if args.run_directory:
            snapshot, config, _, _, _ = runner.read_snapshot(args.run_directory, root=root)
        else:
            path = args.config if args.config.is_absolute() else root / args.config
            zones = [z.strip().upper() for item in args.zones for z in item.split(",")] if args.zones else None
            config = runner.load_config(path, zones=zones, delivery_day=args.delivery_day, end_day=args.end_day)
        if args.refresh_sources or args.action == "refresh":
            from nyx_scarcity.refresh import refresh_sources
            from nyx_scarcity.fuel_refresh import refresh_fuels
            refreshed = refresh_sources(config, root=root)
            if not refreshed["required_sources_complete"]:
                raise ValueError(f"Source refresh incomplete. Private diagnostic: {refreshed['audit_path']}. "
                                 "No training or operational update performed. Use the saved configuration without RefreshSources to explicitly test fallback.")
            physical_refresh = refreshed
            refreshed = refresh_fuels(physical_refresh["config"], root=root)
            if not refreshed["required_sources_complete"]:
                raise ValueError(f"Fuel refresh incomplete. Private diagnostic: {refreshed['audit_path']}. "
                                 "No training or operational update performed. Use the saved configuration without RefreshSources to explicitly test fallback.")
            config = refreshed["config"]
        if args.action == "refresh":
            result = {k: refreshed[k] for k in ("status", "saved_config", "source_dir", "audit_path", "required_sources_complete")}
            result["physical_audit_path"] = physical_refresh["audit_path"]
            result["physical_sources"] = {k: v["status"] for k, v in physical_refresh.get("sources", {}).items()}
            result["fuel_sources"] = list(refreshed.get("sources", {}))
        elif args.action == "audit":
            panel, audit = runner.audit_inputs(config, root=root)
            result = {"status": "audit_read_only", "rows": len(panel), "audit": audit}
        elif args.action in {"run", "prepare"}:
            snapshot = runner.prepare(config, root=root)
            path = runner.evaluate(snapshot, root=root) if args.action == "run" else None
            result = {"snapshot": str(snapshot), "report": str(path) if path else None}
        else:
            snapshot = runner.resolve_snapshot(root, config, args.run_directory, prepared=args.action != "report")
            if args.action == "status":
                result = json.loads((snapshot / "status.json").read_text(encoding="utf-8"))
            else:
                path = runner.evaluate(snapshot, root=root) if args.action == "backtest" else runner.report(snapshot, root=root)
                result = {"snapshot": str(snapshot), "report": str(path)}
        print(json.dumps({**result, "diagnostic_only": True, "production_modified": False, "activation_performed": False},
                         ensure_ascii=False, indent=2, allow_nan=False, default=str))
        return 0
    except KeyboardInterrupt:
        logging.error("[Scarcity] Interrompu. Backtest permet une reprise deterministe du snapshot incomplet.")
        return 130
    except (ValueError, OSError, KeyError, TypeError) as exc:
        logging.error("[Scarcity] %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
