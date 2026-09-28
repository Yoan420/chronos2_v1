"""Capture today's JAO Initial Computation for a prospective annual CWE bundle.

Run between D-1 01:15 and D-1 08:00 Europe/Paris for delivery day D.
Historical research downloads cannot be imported as pre-cutoff captures.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import pandas as pd

from chronos2_hourly.jao_flowbased import JaoCoreClient
from chronos2_hourly.nyx_annual_jao_source import (
    capture_day,
    exclusive_cache_lock,
    publish_jao_receipt,
    verify_daily_capture,
)
from materialize_jao_core_flowbased import _tls_configuration


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "data/pit/nyx_annual_jao_initial_live"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", required=True,
                        help="Civil delivery day YYYY-MM-DD Europe/Paris")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--ca-bundle", type=Path,
                        help="Trusted PEM bundle for an enterprise TLS proxy")
    parser.add_argument("--verify-only", action="store_true",
                        help="Use an existing pre-cutoff capture; never call JAO")
    args = parser.parse_args(argv)
    day = date.fromisoformat(args.delivery_day)
    if day.isoformat() != args.delivery_day:
        raise ValueError("Delivery day must be YYYY-MM-DD")
    cache = args.cache_root.resolve()
    bundle = (args.bundle or ROOT / "runs/live/nyx_annual_cpu" / day.isoformat()).resolve()
    if bundle == cache or bundle.is_relative_to(cache):
        raise ValueError("JAO bundle must be outside the immutable source cache")
    with exclusive_cache_lock(cache):
        if args.verify_only:
            verify_daily_capture(cache, day)
        else:
            verify, tls_source = _tls_configuration(
                argparse.Namespace(insecure=False, ca_bundle=args.ca_bundle))
            with JaoCoreClient(verify=verify) as client:
                capture_day(day=day, cache_root=cache, client=client,
                            now_utc=pd.Timestamp.now(tz="UTC"),
                            tls_trust_source=tls_source)
        result = publish_jao_receipt(day=day, cache_root=cache, bundle=bundle)
    print(json.dumps({"source": "jao_initial", "delivery_day": day.isoformat(),
                      **result}, ensure_ascii=False))
    return 0 if result["state"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
