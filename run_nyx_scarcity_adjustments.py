"""Price-adjustment proposals from frozen OOS experts, never an operational run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import uuid

import pandas as pd

from chronos2_hourly.process_lock import exclusive_process_lock
from nyx_scarcity import runner as base, variants_runner as variants

ROOT = Path(__file__).resolve().parent
NAMESPACE = variants.NAMESPACE / "adjustments"
DATA_FILES = {"adjustments.parquet", "metrics.json", "source_audit.json"}


def output_path(value: Path | str, *, root: Path = ROOT) -> Path:
    path = variants.safe_path(root, value)
    namespace = root.resolve() / NAMESPACE
    if not path.is_relative_to(namespace) or namespace.resolve() != namespace:
        raise ValueError("Adjustment reports must stay in their separate experiment namespace.")
    return path


def read_source(source: Path, *, root: Path = ROOT):
    directory, config, manifest = variants.read_suite(source, root=root)
    result = json.loads((directory / "comparison_manifest.json").read_text(encoding="utf-8"))
    hashes = {name: base.digest(directory / name) for name in ("manifest.json", "comparison_manifest.json", "comparison.json")}
    if (result.get("status") != "completed" or result.get("suite_manifest_sha256") != hashes["manifest.json"]
            or result.get("comparison_sha256") != hashes["comparison.json"]):
        raise ValueError("Only a completed and checksum-verified variant comparison can supply proposals.")
    expected = {v["id"]: base.digest(directory / v["id"] / "results_manifest.json") for v in config["variants"]}
    if result.get("variant_manifests") != expected:
        raise ValueError("Source variant result manifests differ from the completed comparison.")
    predictions = {"hgb_v1": pd.read_parquet(directory / "control_predictions.parquet")}
    for item in config["variants"]:
        name = item["id"]
        variants._verify_variant(directory, name, manifest)
        predictions[name] = pd.read_parquet(directory / name / "predictions.parquet")
    if hashes != {name: base.digest(directory / name) for name in hashes}:
        raise ValueError("Source comparison changed during the read.")
    audit = {"source_suite": str(directory), "source_hashes": hashes, "variant_manifests": expected,
             "source_settings": manifest["settings"], "diagnostic_only": True,
             "production_modified": False, "activation_performed": False,
             "existing_oos_amplitude_reused": True, "new_training_performed": False,
             "fixed_amplitude_fractions": [0.25, 0.5, 1.0], "fraction_selected_by_backtest": False}
    return predictions, audit


def _code_seals() -> dict:
    names = ["run_nyx_scarcity_adjustments.py", "ScarcityAdjustments.ps1",
             "nyx_scarcity/adjustment_analysis.py", "nyx_scarcity/adjustment_reporting.py"]
    return {name: base.digest(ROOT / name) for name in names}


def run(source: Path, *, root: Path = ROOT) -> Path:
    from nyx_scarcity.adjustment_analysis import build_adjustments
    destination = output_path(root / NAMESPACE, root=root)
    before = base.protected_state(root)
    with exclusive_process_lock(destination / "run.lock"):
        predictions, audit = read_source(source, root=root)
        rows, summary = build_adjustments(predictions,
            correction_clip_eur_mwh=audit["source_settings"]["correction_clip_eur_mwh"])
        # Re-verify the source after analysis before creating a published snapshot.
        _, checked_audit = read_source(source, root=root)
        if audit != checked_audit or before != base.protected_state(root):
            raise ValueError("Sources or protected operational files changed during analysis.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        directory = output_path(destination / "snapshots" / stamp, root=root)
        directory.mkdir(parents=True, exist_ok=False)
        base._parquet(directory / "adjustments.parquet", rows)
        base._json(directory / "metrics.json", summary)
        base._json(directory / "source_audit.json", audit)
        base._json(directory / "manifest.json", {
            "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_files": {name: base.digest(directory / name) for name in sorted(DATA_FILES)},
            "code_sha256": _code_seals(), "protected_files": before,
            "diagnostic_only": True, "production_modified": False, "activation_performed": False})
        return report(directory, root=root)


def read_snapshot(directory: Path, *, root: Path = ROOT):
    directory = output_path(directory, root=root)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("diagnostic_only") is not True or manifest.get("activation_performed") is not False:
        raise ValueError("Invalid research-only adjustment manifest.")
    base._verify(directory, manifest.get("input_files"), DATA_FILES)
    return directory, manifest


def report(directory: Path, *, root: Path = ROOT) -> Path:
    from nyx_scarcity.adjustment_reporting import render_adjustments
    directory, manifest = read_snapshot(directory, root=root)
    before = base.protected_state(root)
    with exclusive_process_lock(directory / "report.lock"):
        final = directory / "nyx_scarcity_adjustments.html"
        temporary = directory / (".report_" + uuid.uuid4().hex + ".html")
        try:
            render_adjustments(pd.read_parquet(directory / "adjustments.parquet"),
                json.loads((directory / "metrics.json").read_text(encoding="utf-8")),
                {**manifest, "source": json.loads((directory / "source_audit.json").read_text(encoding="utf-8"))},
                temporary)
            if before != base.protected_state(root):
                raise ValueError("Operational files changed concurrently; report publication stopped.")
            temporary.replace(final)
        finally:
            temporary.unlink(missing_ok=True)
        result = {"status": "completed", "snapshot": str(directory), "report": str(final),
                  "diagnostic_only": True, "activation_performed": False}
        base._json(directory / "status.json", result)
        base._json(output_path(root / NAMESPACE, root=root) / "latest.json", result)
        return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["run", "report", "status"], default="run")
    parser.add_argument("--source-suite", type=Path)
    parser.add_argument("--run-directory", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        if args.source_suite and args.action != "run":
            raise ValueError("SourceSuite is reserved for Run; completed adjustment reports are frozen.")
        if args.run_directory and args.action == "run":
            raise ValueError("RunDirectory is reserved for Report/Status.")
        if args.action == "run":
            if args.source_suite:
                source = args.source_suite
            else:
                pointer = ROOT / variants.NAMESPACE / "latest.json"
                if not pointer.is_file():
                    raise ValueError("First complete ScarcityVariants.ps1 -Action Run.")
                source = Path(json.loads(pointer.read_text(encoding="utf-8"))["snapshot"])
            result = {"report": str(run(source))}
        else:
            if args.run_directory:
                directory = args.run_directory
            else:
                pointer = output_path(ROOT / NAMESPACE) / "latest.json"
                directory = Path(json.loads(pointer.read_text(encoding="utf-8"))["snapshot"])
            directory, _ = read_snapshot(directory)
            result = (json.loads((directory / "status.json").read_text(encoding="utf-8"))
                      if args.action == "status" else {"report": str(report(directory))})
        print(json.dumps({**result, "diagnostic_only": True, "new_training_performed": False,
                          "production_modified": False, "activation_performed": False}, indent=2))
        return 0
    except Exception:
        logging.exception("Adjustment diagnostic failed; operational forecasts were not modified.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
