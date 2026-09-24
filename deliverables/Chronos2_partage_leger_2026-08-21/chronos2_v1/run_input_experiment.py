#!/usr/bin/env python
"""Audit or run isolated hourly input experiments.

Production configurations and ``runs/live`` are read-only.  Every experiment
is resolved below ``runs/experiments/<experiment_id>`` and uses the standard
hourly runner, so adding a numeric PIT series does not require editing model
code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import yaml

from chronos2_hourly.experiment_contract import (
    SUPPORTED_ZONES,
    ExperimentContract,
    ResolvedExperimentConfig,
    load_experiment_contract,
    resolve_experiment_config,
    validate_experiment_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = PROJECT_ROOT / "config" / "experiment.yaml"
HOURLY_RUNNER = PROJECT_ROOT / "run_chronos2_hourly.py"


@dataclass(frozen=True)
class ExperimentPlan:
    contract: ExperimentContract
    resolved: tuple[ResolvedExperimentConfig, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_zones(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        values = [str(values)]
    zones: list[str] = []
    for value in values:
        zone = str(value).strip().upper()
        if zone not in SUPPORTED_ZONES:
            raise ValueError(
                f"Pays non supporte: {value!r}; choix={', '.join(SUPPORTED_ZONES)}"
            )
        if zone in zones:
            raise ValueError(f"Pays duplique: {zone}")
        zones.append(zone)
    if not zones:
        raise ValueError("Selectionnez au moins un pays.")
    return tuple(zones)


def build_experiment_plan(
    experiment_path: str | Path,
    *,
    zones: Sequence[str],
    project_root: str | Path = PROJECT_ROOT,
) -> ExperimentPlan:
    """Validate every requested zone before creating any output."""

    root = Path(project_root).expanduser().resolve()
    contract = load_experiment_contract(experiment_path, project_root=root)
    validate_experiment_inputs(contract)
    selected = normalize_zones(zones)
    resolved = tuple(
        resolve_experiment_config(contract, zone=zone, validate_inputs=False)
        for zone in selected
    )
    collisions = [str(item.output_directory) for item in resolved if item.output_directory.exists()]
    if collisions:
        raise FileExistsError(
            "Refus d'ecraser un resultat d'experience existant: "
            + ", ".join(collisions)
        )
    return ExperimentPlan(contract=contract, resolved=resolved)


def build_runner_command(
    item: ResolvedExperimentConfig,
    *,
    resolved_config_path: Path,
    python_executable: str | Path,
    local_files_only: bool,
    device: str,
) -> tuple[str, ...]:
    command = [
        str(Path(python_executable).expanduser().resolve()),
        str(HOURLY_RUNNER),
        "--config",
        str(resolved_config_path),
        "--output-dir",
        str(item.output_directory),
        "--device",
        device,
    ]
    if local_files_only:
        command.append("--local-files-only")
    return tuple(command)


def _write_request(item: ResolvedExperimentConfig, contract: ExperimentContract) -> Path:
    output = item.output_directory
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "resolved_experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(item.config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    inputs = []
    for source in contract.inputs_for_zone(item.zone):
        inputs.append(
            {
                "alias": source.alias,
                "series": source.series,
                "pit_file": str(source.pit_file),
                "pit_file_sha256": _sha256(source.pit_file),
                "minimum_coverage": source.minimum_coverage,
            }
        )
    manifest = {
        "schema_version": 1,
        "run_type": "offline_input_experiment",
        "experiment_id": contract.experiment_id,
        "zone": item.zone,
        "production_changed": False,
        "production_config": str(item.production_config),
        "production_config_sha256": _sha256(item.production_config),
        "experiment_contract": str(contract.source_path),
        "experiment_contract_sha256": _sha256(contract.source_path),
        "resolved_config": config_path.name,
        "resolved_config_sha256": _sha256(config_path),
        "enabled_inputs": inputs,
        "storm_used_as_input": False,
        "mkonline_used_as_input": False,
        "output_directory": str(output),
    }
    (output / "experiment_request.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def execute_experiment_plan(
    plan: ExperimentPlan,
    *,
    python_executable: str | Path = sys.executable,
    local_files_only: bool = True,
    device: str = "auto",
) -> tuple[Path, ...]:
    """Run zones sequentially; an experiment can never publish to live roots."""

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device doit etre auto, cpu ou cuda")
    executable = Path(python_executable).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Python introuvable: {executable}")
    if not HOURLY_RUNNER.is_file():
        raise FileNotFoundError(f"Runner introuvable: {HOURLY_RUNNER}")

    outputs: list[Path] = []
    for item in plan.resolved:
        config_path = _write_request(item, plan.contract)
        command = build_runner_command(
            item,
            resolved_config_path=config_path,
            python_executable=executable,
            local_files_only=local_files_only,
            device=device,
        )
        print(f"[{item.zone}] experience {plan.contract.experiment_id}")
        print("Commande (argv, shell=False): " + json.dumps(command, ensure_ascii=False))
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        outputs.append(item.output_directory)
    return tuple(outputs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audite ou lance un test isole de nouvelles series PIT."
    )
    parser.add_argument("--experiment", default=str(DEFAULT_EXPERIMENT))
    parser.add_argument("--zones", nargs="+", default=list(SUPPORTED_ZONES))
    parser.add_argument("--run", action="store_true", help="Lance le backtest apres audit.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_experiment_plan(args.experiment, zones=args.zones)
        for item in plan.resolved:
            aliases = ", ".join(item.enabled_aliases) or "aucune"
            print(f"[{item.zone}] OK | inputs={aliases} | sortie={item.output_directory}")
        if not args.run:
            print("Audit termine. Ajoutez --run pour lancer les backtests sequentiels.")
            return 0
        execute_experiment_plan(
            plan,
            python_executable=args.python_executable,
            local_files_only=not args.allow_model_download,
            device=args.device,
        )
        return 0
    except Exception as exc:
        print(f"Erreur experience: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
