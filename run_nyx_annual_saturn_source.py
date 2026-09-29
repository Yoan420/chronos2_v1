"""Download/resume Saturn inputs at each historical and current NYX cutoff."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from chronos2_hourly.nyx_annual_saturn_source import DEFAULT_CACHE, ROOT, sync, publish
from experiment_console.security import redact, redact_text


def failure_report(error, *, delivery_day, phase, cache, bundle):
    """A bounded exception chain with context from this collector attempt."""
    chain, seen = [], set()
    current = error
    while current is not None and id(current) not in seen and len(chain) < 6:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": redact_text(str(current))})
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return redact({"state": "ERROR", "source_group": "saturn", "source": "saturn",
                   "delivery_day": delivery_day, "failed_day": getattr(error, "day", None),
                   "phase": getattr(error, "phase", phase), "alias": getattr(error, "alias", None),
                   "series": getattr(error, "series", None), "cache": str(cache), "bundle": str(bundle),
                   "error": f"{type(error).__name__}: {redact_text(str(error))}",
                   "exception_chain": chain})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--first-day")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args(argv)
    bundle = args.bundle or ROOT / "runs/live/nyx_annual_cpu" / args.delivery_day
    phase = "sync"
    try:
        if not args.assemble_only:
            sync(args.delivery_day, first_day=args.first_day, cache=args.cache, workers=args.workers)
        phase = "publish"
        receipt = publish(bundle, args.delivery_day, first_day=args.first_day, cache=args.cache)
    except Exception as error:
        print(json.dumps(failure_report(error, delivery_day=args.delivery_day, phase=phase,
                         cache=args.cache, bundle=bundle), ensure_ascii=False), flush=True)
        return 1
    print(json.dumps({"state": "COMPLETE", "source_group": "saturn", "receipt": str(receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
