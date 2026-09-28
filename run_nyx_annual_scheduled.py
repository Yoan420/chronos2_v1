"""Silent Windows scheduled entry point with persistent per-attempt logs."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent


def launch(action, day=None):
    if action not in ("capture", "forecast"):
        raise ValueError("Expected capture or forecast")
    day = day or (datetime.now(ZoneInfo("Europe/Paris")).date() + timedelta(days=1)).isoformat()
    folder = ROOT / "runs/logs/nyx_annual_cpu" / day
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    log = folder / f"{action}_{stamp}_{uuid4().hex[:8]}.log"
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":
        python = python.with_name("python.exe")
    with log.open("w", encoding="utf-8") as stream:
        command = [str(python), "-u", str(ROOT / "run_nyx_annual_pipeline.py"),
                   "--action", action, "--delivery-day", day]
        stream.write(json.dumps({"action": action, "delivery_day": day, "started_at_utc": stamp}) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        stream.write(json.dumps({"return_code": result.returncode}) + "\n")
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("capture", "forecast"), required=True)
    parser.add_argument("--delivery-day")
    args = parser.parse_args(argv)
    return launch(args.action, args.delivery_day)


if __name__ == "__main__":
    raise SystemExit(main())
