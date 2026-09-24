"""Isolated, offline Kalman-only ablation on sealed SolarWind predictions."""
from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('validate', 'run'), default='validate')
    parser.add_argument('--zones', nargs='+', choices=('DE', 'NL'), default=['DE', 'NL'])
    parser.add_argument('--threads', type=int, choices=(1, 2), default=2)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    from chronos2_hourly.solar_wind_interaction import run
    run(action=args.action, zones=args.zones, workers=args.workers, threads=args.threads)


if __name__ == '__main__':
    main()
