"""Replay the complete NYX architecture at native quarter-hour cadence."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/nyx_fullquarterhour.yaml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage", choices=("all", "raw", "postprocess"), default="all")
    args = parser.parse_args(argv)
    from nyx_fullquarterhour.raw import load_config
    from nyx_fullquarterhour.runner import run
    directory = run(load_config(args.config), root=ROOT, resume=args.resume, stage=args.stage)
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    print(json.dumps({**status, "directory": str(directory)}, indent=2))
    return 0 if status["status"] in {"complete", "raw_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
