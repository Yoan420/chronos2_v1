"""Reproducible local native-quarter-hour Chronos research runner."""
from pathlib import Path
import argparse
import json

ROOT=Path(__file__).resolve().parent


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/nyx_quarterhour.yaml")
    parser.add_argument("--resume",type=Path)
    args=parser.parse_args(argv)
    from nyx_quarterhour.runner import load_config,run
    directory=run(load_config(args.config),root=ROOT,resume=args.resume)
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"status":summary["status"],"reason":summary.get("reason"),"directory":str(directory),
                      "report":str(directory/"report.html"),"activation_performed":False},indent=2))
    return 0 if summary["status"]=="complete" else 2


if __name__=="__main__":
    raise SystemExit(main())
