"""Separate StressGuard replay and prospective validation; no production writes."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["run", "prepare", "backtest", "report", "status", "freeze", "capture", "issue", "evaluate"], default="run")
    parser.add_argument("--config", type=Path, default=ROOT/"config/nyx_stress_guard.yaml")
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--ledger-directory", type=Path)
    parser.add_argument("--input-directory", type=Path)
    parser.add_argument("--delivery-day")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        import pandas as pd
        from nyx_stress_guard import runner, ledger, prospective
        if args.run_directory and args.action in ("run", "prepare"):
            raise ValueError("Run prepares a new snapshot. Use Backtest to resume an existing one.")
        if args.action in ("run", "prepare"):
            directory = runner.prepare(runner.load_config(args.config), root=ROOT)
            result = {"snapshot": str(directory)}
            if args.action == "run":
                result["report"] = str(runner.run_suite(directory, root=ROOT))
        elif args.action in ("backtest", "report", "freeze") or (args.action == "status" and not args.ledger_directory):
            directory = args.run_directory or runner.resolve_latest(runner.load_config(args.config), root=ROOT,
                completed=args.action in ("report", "freeze"))
            if args.action == "backtest":
                result = {"report": str(runner.run_suite(directory, root=ROOT))}
            elif args.action == "report":
                result = {"report": str(runner.report(directory, root=ROOT))}
            elif args.action == "freeze":
                frozen = prospective.freeze(directory, root=ROOT)
                result = {"ledger": str(frozen), **ledger.ledger_status(frozen, root=ROOT)}
            else:
                directory, _, _ = runner.read_suite(directory, root=ROOT)
                result = json.loads((directory/"status.json").read_text(encoding="utf-8"))
        else:
            target = prospective.resolve_ledger(args.ledger_directory, root=ROOT)
            if args.action == "status":
                result = ledger.ledger_status(target, root=ROOT)
            elif args.action in ("capture", "issue"):
                from nyx_stress_guard.inputs import capture_inputs
                _, manifest, _ = ledger._read(target, ROOT)
                ledger._verify_files(ROOT, manifest["candidate_files"])
                ledger._verify_files(ROOT, manifest["history_files"])
                day = args.delivery_day or (ledger.now_utc().tz_convert("Europe/Paris").date()+pd.Timedelta(days=1)).isoformat()
                # Reject late/seen deliveries before any remote call or source copy.
                if (pd.Timestamp(day).date().isoformat() != day or day < manifest["start_day"]
                        or day > (pd.Timestamp(manifest["start_day"])+pd.Timedelta(days=364)).date().isoformat()):
                    raise ValueError("Delivery is already explored or outside the frozen 365-day prospective window.")
                ledger._check_window(day, manifest, ledger.now_utc())
                if args.input_directory is not None:
                    capture = runner.safe_path(ROOT, args.input_directory)
                    panel = pd.read_parquet(capture/"panel.parquet")
                    evidence = json.loads((capture/"input_evidence.json").read_text(encoding="utf-8"))
                    actual_days = panel.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
                    if not actual_days.eq(day).all():
                        raise ValueError("Saved input delivery differs from --delivery-day.")
                else:
                    source = Path(manifest["candidate_files"]["suite_manifest"]["path"]).parent
                    stamp = ledger.now_utc().strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
                    capture = runner.safe_path(ROOT, ROOT/runner.NAMESPACE/"captures"/stamp)
                    panel, evidence = capture_inputs(root=ROOT, snapshot_dir=source, delivery_day=day, output_directory=capture)
                checked_panel, checked_day = ledger._frame(panel, manifest)
                ledger._input_evidence(ROOT, evidence, checked_day, ledger.now_utc(), panel=checked_panel, manifest=manifest)
                ledger._check_window(day, manifest, ledger.now_utc())
                result = {"input_directory": str(capture), "rows": len(panel)}
                if args.action == "issue":
                    result.update(prospective.issue(target, root=ROOT, panel=panel, input_evidence=evidence))
            else:
                from nyx_stress_guard.inputs import collect_observations
                _, manifest, events = ledger._read(target, ROOT)
                emitted = {e["delivery_day"] for e in events if e["kind"] == "forecast"}
                resolved = {(e["delivery_day"], e["zone"]) for e in events if e["kind"] == "observation_resolution"}
                pending = sorted(d for d in emitted if any((d,z) not in resolved for z in manifest["zones"]))
                if args.delivery_day:
                    if args.delivery_day not in emitted:
                        raise ValueError("No prior issued forecast exists for the requested delivery.")
                    pending = [args.delivery_day]
                if pending:
                    first = pd.Timestamp(pending[0])
                    selected = [d for d in pending if pd.Timestamp(d) <= first+pd.Timedelta(days=30)]
                    observations, evidence = collect_observations(root=ROOT, days=selected, zones=manifest["zones"])
                    result = ledger.resolve_observations(target, root=ROOT, observations=observations, observation_evidence=evidence)
                    result["prospective_status"] = ledger.ledger_status(target, root=ROOT)
                else:
                    result = {"status": "no_unresolved_issued_forecasts", **ledger.ledger_status(target, root=ROOT)}
        print(json.dumps({**result, "diagnostic_only": True, "production_modified": False,
                          "activation_performed": False}, indent=2, default=str))
        return 0
    except Exception:
        logging.exception("StressGuard stopped safely. No operational forecast was modified.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
