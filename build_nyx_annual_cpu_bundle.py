"""Build the 12 annual CWE CPU feature matrices, then seal completed inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chronos2_hourly.nyx_annual_cpu_bundle_builder import materialize_features, seal_bundle
from chronos2_hourly.process_lock import exclusive_process_lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--action", choices=("features", "seal"), default="features")
    args = parser.parse_args(argv)
    with exclusive_process_lock(args.bundle / "bundle_build.lock"):
        if args.action == "features":
            built = materialize_features(args.bundle, args.delivery_day)
            result = {"state": "FEATURES_BUILT", "delivery_day": args.delivery_day,
                "matrices": {family: {zone: list(frame.shape) for zone, frame in countries.items()}
                             for family, countries in built.features.items()}}
        else:
            seal_bundle(args.bundle, args.delivery_day)
            result = {"state": "BUNDLE_SEALED", "delivery_day": args.delivery_day}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
