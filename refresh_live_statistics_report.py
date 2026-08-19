#!/usr/bin/env python
"""Create an immutable report snapshot with the latest realized Statistics.

This utility never reruns or rewrites a published forecast.  It verifies and
copies one immutable live run into ``runs/live/_reports``, builds the separate
Statistics history from the sealed benchmark plus prior issued/replay runs,
then regenerates the HTML and a fresh checksum manifest.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

import pandas as pd

from chronos2_hourly.live_history import update_live_statistics_history
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.reporting import write_hourly_html_report
from chronos2_hourly.storm_dashboard import fetch_native_dashboard_snapshot
from chronos2_modular.common import load_yaml
from chronos2_modular.saturn import create_saturn_client
from run_mkonline_blend_hourly import STORM_COMPARATOR_PATH


TIMEZONE = "Europe/Paris"
DEFAULT_CONFIG = "chronos2_hourly_fr_mkonline_live_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: objet JSON attendu.")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _verify_published_run(run_dir: Path) -> dict[str, str]:
    checksum_path = run_dir / "artifact_checksums.json"
    manifest = _read_json(checksum_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError(f"{checksum_path}: artifacts doit etre une liste.")
    verified: dict[str, str] = {}
    for item in artifacts:
        if not isinstance(item, dict) or item.get("role") != "run_artifact":
            continue
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Chemin run_artifact non relatif: {relative}.")
        source = run_dir / relative
        observed = _sha256(source)
        if observed != str(item.get("sha256", "")).lower():
            raise ValueError(f"Archive publiee modifiee: {relative}.")
        verified[relative.as_posix()] = observed
    if "forecast_hourly_fr.csv" not in verified:
        raise ValueError("Checksum du forecast publie absent.")
    return verified


def _canonical_target(run_dir: Path) -> pd.Series:
    frame = pd.read_csv(
        run_dir / "inputs" / "aligned_inputs.csv.gz",
        usecols=["timestamp", "target"],
    )
    index = pd.DatetimeIndex(
        pd.to_datetime(frame["timestamp"], utc=True, errors="raise"),
        name="timestamp",
    )
    target = pd.Series(
        pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype=float),
        index=index,
        name="target",
    )
    if target.index.has_duplicates:
        raise ValueError("Cible canonique du run source dupliquee.")
    return target


def _load_native_storm_snapshot(
    *,
    config: dict[str, Any],
    config_dir: Path,
    benchmark_run: Path,
    current_delivery_day: date,
) -> tuple[pd.Series, dict[str, Any]]:
    """Fetch the exact native FR dashboard curve for a report-only snapshot."""

    base_config_path = _resolve(
        config["live"]["base_config"], base=config_dir
    )
    base_config = load_yaml(base_config_path)
    data = base_config.get("data")
    if not isinstance(data, dict):
        raise TypeError("base data config must be a mapping")
    metrics = _read_json(benchmark_run / "metrics_hourly.json")
    diagnostics = metrics.get("training_diagnostics")
    if not isinstance(diagnostics, dict):
        raise TypeError("benchmark training_diagnostics must be a mapping")
    start_day = pd.Timestamp(diagnostics["evaluation_start_local_date"]).date()
    expected = pd.date_range(
        local_delivery_day_index(start_day, timezone=TIMEZONE)[0],
        local_delivery_day_index(
            current_delivery_day - pd.Timedelta(days=1), timezone=TIMEZONE
        )[-1],
        freq="h",
    )
    client = create_saturn_client(
        str(data["saturn_url"]), str(data["saturn_author"])
    )
    return fetch_native_dashboard_snapshot(
        client, zone="FR", expected_index=expected
    )


def _write_checksums(directory: Path, *, source_run: Path) -> None:
    checksum_path = directory / "artifact_checksums.json"
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path != checksum_path:
            entries.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "role": "report_snapshot_artifact",
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        checksum_path,
        {
            "algorithm": "sha256",
            "run_type": "live_statistics_report_snapshot",
            "source_run": str(source_run),
            "source_checksum_manifest_sha256": _sha256(
                source_run / "artifact_checksums.json"
            ),
            "artifacts": entries,
        },
    )


def _publish(staging: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(
            f"Le snapshot existe deja et reste immuable: {output}."
        )
    staging.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerer un HTML live avec Statistics realisees, sans rerun."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    settings = config.get("live")
    if not isinstance(settings, dict):
        raise TypeError("live doit etre un mapping YAML.")
    source = Path(args.source_run).expanduser().resolve()
    source_manifest = _read_json(source / "run_manifest.json")
    if source_manifest.get("run_type") != "live_day_ahead":
        raise ValueError("Le snapshot de rapport exige un run live_day_ahead.")
    delivery_day = pd.Timestamp(source_manifest["delivery_day_local"]).date()
    _verify_published_run(source)
    config_dir = config_path.parent
    live_root = _resolve(settings.get("output_root", "runs/live"), base=config_dir)
    benchmark = _resolve(settings["sealed_benchmark_run"], base=config_dir)
    default_output = (
        live_root
        / "_reports"
        / f"fr_day_ahead_{delivery_day.isoformat()}_statistics"
    )
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output
    )
    if output.parent.resolve() != (live_root / "_reports").resolve():
        raise ValueError("Le snapshot doit rester sous runs/live/_reports.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        for old_html in staging.glob("*.html"):
            old_html.unlink()
        dashboard_raw, dashboard_source = _load_native_storm_snapshot(
            config=config,
            config_dir=config_dir,
            benchmark_run=benchmark,
            current_delivery_day=delivery_day,
        )
        statistics_audit = update_live_statistics_history(
            staging_run_dir=staging,
            sealed_benchmark_run=benchmark,
            live_output_root=live_root,
            replay_output_root=live_root / "_replays",
            current_delivery_day=delivery_day,
            canonical_target=_canonical_target(source),
            storm_pit_path=(project_root / STORM_COMPARATOR_PATH).resolve(),
            storm_dashboard_native=dashboard_raw,
            storm_dashboard_source=dashboard_source,
            zone="FR",
            timezone=TIMEZONE,
        )
        derived_manifest = dict(source_manifest)
        derived_manifest.update(
            {
                "run_type": "live_statistics_report_snapshot",
                "source_run": str(source),
                "source_run_manifest_sha256": _sha256(
                    source / "run_manifest.json"
                ),
                "source_forecast_sha256": _sha256(
                    source / "forecast_hourly_fr.csv"
                ),
                "snapshot_created_at_utc": str(pd.Timestamp.now(tz="UTC")),
                "statistics_history": statistics_audit,
                "forecast_recomputed": False,
                "forecast_rewritten": False,
            }
        )
        _write_json(staging / "run_manifest.json", derived_manifest)
        report_cfg = config.get("report", {})
        if not isinstance(report_cfg, dict):
            raise TypeError("report doit etre un mapping YAML.")
        report_path = staging / (
            f"chronos2_hourly_fr_mkonline_live_{delivery_day.isoformat()}_statistics.html"
        )
        write_hourly_html_report(
            staging,
            output_path=report_path,
            title=(
                "Forecast day-ahead FR - livraison "
                f"{delivery_day.isoformat()} - Statistics actualisees"
            ),
            native_model="mkonline_blend",
            baseline_model="residual_corrected",
            zone="FR",
            timezone=TIMEZONE,
            extreme_threshold=float(report_cfg.get("extreme_threshold", 150.0)),
            history_hours=int(report_cfg.get("forecast_history_hours", 168)),
        )
        _write_checksums(staging, source_run=source)
        _publish(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Snapshot rapport : {output}")
    print(f"Rapport HTML : {report_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
