"""Offline DE/NL ablation: CatBoost interaction and asymmetric correction caps."""
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
    if len(args.zones) != len(set(args.zones)):
        parser.error('Each zone must appear exactly once.')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    from chronos2_hourly.solar_wind_corrector_interaction import run
    run(action=args.action, zones=args.zones, threads=args.threads, workers=args.workers)


if __name__ == '__main__':
    main()
