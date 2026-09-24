"""One command for ordinary Both and both nuclear variants; no model changes."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
ZONES = ("FR", "DE", "BE", "NL", "ES")


@dataclass(frozen=True)
class RunStep:
    name: str
    zone: str | None
    command: tuple[str, ...]


def build_complete_plan(
    *, project_root: Path, python_executable: str, zones: Sequence[str], delivery_day: str,
    device: str = "auto", threads: int = 4, workers: int = 4,
    nuclear_config: Path | None = None, stop_on_error: bool = False,
) -> tuple[RunStep, ...]:
    """Pin all child arguments, including the delivery date, before starting."""
    root = Path(project_root).resolve()
    selected = tuple(dict.fromkeys(zones))
    if not selected or any(zone not in ZONES for zone in selected):
        raise ValueError("Selection de pays invalide.")
    if device not in {"auto", "cpu", "cuda"} or not 1 <= threads <= 128 or not 1 <= workers <= 128:
        raise ValueError("Device ou threads/workers invalides.")
    config = Path(nuclear_config) if nuclear_config else root / "config/nuclear_forecast.yaml"
    config = (config if config.is_absolute() else root / config).resolve()
    shared = ("--delivery-day", delivery_day, "--device", device,
              "--threads", str(threads), "--workers", str(workers))
    ordinary = (python_executable, str(root / "run_multicountry_forecast.py"),
                "--zones", *selected, "--mode", "Both", *shared)
    if stop_on_error:
        ordinary += ("--stop-on-error",)
    steps = [RunStep("autonomous + kalman", None, ordinary)]
    for zone in selected:
        steps.append(RunStep("nuclear_autonomous + nuclear_kalman", zone,
            (python_executable, str(root / "run_nuclear_forecast.py"), "--config", str(config),
             "--stage", "Run", "--zones", zone, *shared)))
    return tuple(steps)


def validate_complete_inputs(*, project_root: Path, zones: Sequence[str], nuclear_config: Path) -> None:
    """Cheap read-only checks before either pipeline is allowed to write."""
    from run_nuclear_forecast import load_settings, check_lora_inactive

    root = Path(project_root).resolve()
    for name in ("run_multicountry_forecast.py", "run_nuclear_forecast.py"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    settings = load_settings(Path(nuclear_config).resolve())
    if settings["project_root"] != root:
        raise ValueError("Complete exige la meme racine de projet pour les deux pipelines.")
    missing = set(zones).difference(settings.get("zone_configs", {}))
    if missing:
        raise ValueError(f"Pays absents de la configuration nucleaire: {sorted(missing)}")
    for key in ("kalman_config", "lora_activation_config"):
        if not settings[key].is_file():
            raise FileNotFoundError(settings[key])
    for zone in zones:
        path = Path(settings["zone_configs"][zone])
        if not (path if path.is_absolute() else root / path).is_file():
            raise FileNotFoundError(f"Configuration de base nucleaire absente: {zone}/{path}")
    # Complete routes to ordinary Both, not All (which requires active LoRA).
    # Do not start half a batch if the isolated nuclear schema is incompatible.
    check_lora_inactive(settings, list(zones))


def execute_complete_plan(
    plan: Sequence[RunStep], *, project_root: Path, stop_on_error: bool = False,
) -> int:
    """Run sequentially with inherited logs and preserve failures across steps."""
    outcomes: list[tuple[RunStep, int]] = []
    for index, step in enumerate(plan, start=1):
        label = f"{step.zone or 'Batch pays'} | {step.name}"
        print(f"\n[Complete {index}/{len(plan)}] {label}", flush=True)
        print("Commande (argv, shell=False): " + json.dumps(step.command, ensure_ascii=False), flush=True)
        try:
            completed = subprocess.run(list(step.command), check=False, cwd=project_root, shell=False)
            code = completed.returncode
        except KeyboardInterrupt:
            print("[Complete] Interrompu; aucune etape suivante ne sera lancee.", flush=True)
            return 130
        except OSError as error:
            print(f"[Complete] lancement impossible: {error}", flush=True)
            code = 1
        if code in {130, -2, 3221225786, -1073741510}:
            print("[Complete] Processus interrompu; aucune etape suivante ne sera lancee.", flush=True)
            return 130
        outcomes.append((step, code))
        if code != 0 and stop_on_error:
            break
    print("\nResultat du lancement Complete", flush=True)
    for step, code in outcomes:
        print(f"{step.zone or 'Batch pays'} | {step.name} | "
              + ("OK" if code == 0 else f"ECHEC (code {code})"), flush=True)
    for step in plan[len(outcomes):]:
        print(f"{step.zone or 'Batch pays'} | {step.name} | NON LANCE", flush=True)
    return 0 if outcomes and all(code == 0 for _, code in outcomes) and len(outcomes) == len(plan) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones", nargs="+", choices=ZONES, default=list(ZONES))
    parser.add_argument("--delivery-day")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nuclear-config", type=Path, default=PROJECT_ROOT / "config/nuclear_forecast.yaml")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        from run_nuclear_forecast import delivery_date
        # One civil Paris date for all subprocesses, even if a long batch
        # crosses midnight or a daylight-saving transition.
        day = str(delivery_date(args.delivery_day).date())
        zones = tuple(dict.fromkeys(args.zones))
        nuclear_config = (args.nuclear_config if args.nuclear_config.is_absolute()
                          else PROJECT_ROOT / args.nuclear_config).resolve()
        plan = build_complete_plan(project_root=PROJECT_ROOT, python_executable=sys.executable,
            zones=zones, delivery_day=day, device=args.device, threads=args.threads,
            workers=args.workers, nuclear_config=nuclear_config, stop_on_error=args.stop_on_error)
        validate_complete_inputs(project_root=PROJECT_ROOT, zones=zones, nuclear_config=nuclear_config)
        print(f"Livraison: {day}\nMode: Complete\nPays: {', '.join(zones)}", flush=True)
        print("4 variantes par pays: autonomous, kalman, nuclear_autonomous, nuclear_kalman.", flush=True)
        print("Execution successive; une nouvelle livraison nucleaire conserve son replay de 730 jours.", flush=True)
        if args.dry_run:
            for step in plan:
                print("Commande (argv, shell=False): " + json.dumps(step.command, ensure_ascii=False), flush=True)
            print("DryRun: aucun pipeline lance.", flush=True)
            return 0
        return execute_complete_plan(plan, project_root=PROJECT_ROOT, stop_on_error=args.stop_on_error)
    except (ValueError, FileNotFoundError) as error:
        print(f"[Complete] Preflight refuse: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
