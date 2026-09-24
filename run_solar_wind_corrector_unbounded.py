"""Unbounded residual-corrector ablation after both sealed DE/NL reuse runs."""
from __future__ import annotations

import argparse
import json
import math
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('validate', 'run'), default='validate')
    parser.add_argument('--zones', nargs='+', choices=('DE', 'NL'), default=['DE', 'NL'])
    parser.add_argument('--threads', type=int, choices=(2,), default=2)
    parser.add_argument('--workers', type=int, choices=(2, 3, 4), default=4)
    parser.add_argument('--min-free-memory-gb', type=float, default=3.0)
    args = parser.parse_args()
    if len(args.zones) != len(set(args.zones)):
        parser.error('Each zone must appear exactly once.')
    if not math.isfinite(args.min_free_memory_gb) or not 2 <= args.min_free_memory_gb <= 8:
        parser.error('The free-memory reserve must be between 2 and 8 GiB.')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    from chronos2_hourly.solar_wind_corrector_unbounded import PendingPredecessor, run
    try:
        result = run(action=args.action, zones=args.zones, threads=args.threads,
                     workers=args.workers, min_free_memory_gb=args.min_free_memory_gb)
    except PendingPredecessor as exc:
        print(json.dumps({'status': 'PENDING_PREDECESSOR', 'reason': str(exc),
                          'started': False, 'production_modified': False}), flush=True)
        return 75
    if args.action == 'validate':
        print(json.dumps({'status': 'VALIDATED', 'identities': result}, default=str), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
