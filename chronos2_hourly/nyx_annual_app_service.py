"""App-facing, fail-closed launch of the annual FR/BE/NL CPU consumer.

The source adapters write to ``runs/live/nyx_annual_cpu/<delivery_day>``.
This service only inspects that bundle and starts the audited consumer. It
never downloads Saturn data or creates missing model inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from chronos2_hourly.nyx_annual_cpu_live import PROTOCOL, preflight
from chronos2_hourly.nyx_annual_live_preflight import delivery_grid, sha256


@dataclass(frozen=True)
class AnnualCpuPaths:
    bundle: Path
    output: Path
    logs: Path


@dataclass
class AnnualCpuProcess:
    delivery_day: str
    output: Path
    log_path: Path
    command: tuple[str, ...]
    process: subprocess.Popen[Any]

    @property
    def return_code(self) -> int | None:
        return self.process.poll()


def paths_for_day(project_root: str | Path, delivery_day: str) -> AnnualCpuPaths:
    """Use one fixed local layout shared with the annual source adapters."""
    delivery_grid(delivery_day)
    root = Path(project_root).resolve()
    return AnnualCpuPaths(
        bundle=root / "runs" / "live" / "nyx_annual_cpu" / delivery_day,
        output=root / "runs" / "nyx_annual_cpu_live" / delivery_day,
        logs=root / "runs" / "nyx_annual_cpu_live" / "_app_logs",
    )


def inspect_annual_cpu_launch(project_root: str | Path, delivery_day: str) -> dict:
    """Show exactly the consumer's current read-only launch decision."""
    paths = paths_for_day(project_root, delivery_day)
    return preflight(paths.bundle, delivery_day, paths.output)


def start_annual_cpu_process(
    project_root: str | Path,
    delivery_day: str,
    *,
    python_executable: str | Path = sys.executable,
) -> AnnualCpuProcess:
    """Recheck all gates immediately before spawning one immutable run."""
    root = Path(project_root).resolve()
    paths = paths_for_day(root, delivery_day)
    runner = root / "run_nyx_annual_cpu_live.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Consommateur NYX CWE CPU absent : {runner}")
    decision = preflight(paths.bundle, delivery_day, paths.output)
    if not decision["ready"]:
        raise ValueError("NYX CWE CPU indisponible : " + "; ".join(decision["blockers"]))
    executable = Path(python_executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Interpréteur Python absent : {executable}")
    command = (
        str(executable), str(runner), "--bundle", str(paths.bundle),
        "--delivery-day", delivery_day, "--output", str(paths.output),
    )
    paths.logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = paths.logs / f"nyx_cwe_cpu_{delivery_day}_{stamp}_{uuid4().hex[:8]}.log"
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        stream.write("Commande (argv, shell=False):\n")
        stream.write(json.dumps(command, ensure_ascii=False) + "\n\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            shell=False,
            creationflags=creationflags,
        )
    return AnnualCpuProcess(delivery_day, paths.output, log_path, command, process)


def load_annual_cpu_forecast(
    project_root: str | Path, delivery_day: str, zone: str
) -> pd.DataFrame:
    """Display only a complete receipt and its unchanged country CSV."""
    if zone not in ("FR", "BE", "NL"):
        raise ValueError("Le pays NYX CWE CPU doit être FR, BE ou NL")
    paths = paths_for_day(project_root, delivery_day)
    receipt_path = paths.output / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("protocol") != PROTOCOL
            or receipt.get("status") != "COMPLETE"
            or receipt.get("delivery_day") != delivery_day
            or set(receipt.get("countries", {})) != {"FR", "BE", "NL"}):
        raise ValueError("Le reçu du forecast annuel CPU est incomplet")
    details = receipt["countries"][zone]
    path = Path(details["csv"]).resolve()
    expected_dir = (paths.output / "zones" / zone).resolve()
    if (path.parent != expected_dir or not path.is_file()
            or sha256(path) != details.get("csv_sha256")):
        raise ValueError(f"{zone} : forecast CSV absent, déplacé ou modifié")
    frame = pd.read_csv(path)
    required = {"timestamp_utc", "timestamp_local", "price_eur_mwh",
                "p_negative", "is_negative_predicted"}
    if not required <= set(frame):
        raise ValueError(f"{zone} : colonnes de forecast manquantes")
    _, current, _ = delivery_grid(delivery_day)
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
    price = pd.to_numeric(frame["price_eur_mwh"], errors="coerce").to_numpy(float)
    probability = pd.to_numeric(frame["p_negative"], errors="coerce").to_numpy(float)
    if (not stamps.equals(current) or len(frame) != details.get("hours")
            or not np.isfinite(price).all() or not np.isfinite(probability).all()
            or not ((0 <= probability) & (probability <= 1)).all()):
        raise ValueError(f"{zone} : contenu de forecast invalide")
    return frame
