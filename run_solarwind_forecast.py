"""Opt-in NYX SolarWind interaction ±40 forecast; never publishes production."""
from __future__ import annotations

import argparse
import logging
import os


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery-day', required=True)
    parser.add_argument('--zones', nargs='+', choices=('DE', 'NL'), default=['DE', 'NL'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--skip-source-sync', action='store_true',
                        help='Use only previously captured inputs; all coverage audits still apply.')
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 32 or len(args.zones) != 2 or set(args.zones) != {'DE', 'NL'}:
        parser.error('1-32 threads and both DE and NL countries required')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from chronos2_hourly.solarwind_live import run
    run(args.delivery_day, zones=args.zones, threads=args.threads, workers=args.workers,
        device=args.device, sync=not args.skip_source_sync)
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception('SolarWind interaction ±40 failed; the usual forecast remains unchanged')
        raise SystemExit(1)
