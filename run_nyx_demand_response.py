"""Independent flexible-demand diagnostic; never modifies Forecast.ps1."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from nyx_demand_response import runner
from nyx_demand_response.inputs import validate_bundle
from nyx_demand_response.dispatch import DispatchError

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','run','report','status','validate','demo','dryrun'),default='run')
    parser.add_argument('--config',type=Path,default=ROOT/'config/nyx_demand_response.yaml')
    parser.add_argument('--run-directory')
    args = parser.parse_args(argv)
    if args.run_directory and args.action in ('prepare','demo','validate','dryrun'):
        parser.error('RunDirectory applies to Run/Report/Status only.')
    try:
        config = runner.load_config(args.config)
        if args.action == 'dryrun':
            result = dict(status='dry_run',writes=False,api_calls=False,production_modified=False,
                source=config['source_suite'],bundle=config['bundle_path'],output=config['output_root'])
        elif args.action == 'demo':
            result = dict(status='synthetic_demonstration_not_forecast',production_modified=False,examples=runner.demos())
        elif args.action == 'validate':
            if config['bundle_path'] is None:
                result = dict(status='not_ready',reason='No documented pre-08 scenario bundle supplied.',promotion_eligible=False)
            else:
                values = validate_bundle(json.loads((ROOT/config['bundle_path']).read_text(encoding='utf8')))
                result = dict(status='contract_valid',scenario_periods=len(values),promotion_eligible=False,
                    warning='Schema validation is not independent evidence of source quality or forecasting skill.')
        elif args.action == 'prepare':
            result = dict(status='prepared',snapshot=str(runner.prepare(config,ROOT)))
        else:
            directory = runner.resolve(config,ROOT,args.run_directory,create=args.action=='run')
            if args.action == 'run':
                runner.evaluate(directory,ROOT)
            result = runner.report(directory,ROOT) if args.action in ('run','report') else runner.status(directory,ROOT)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,default=str),flush=True)
        return 0
    except (ValueError,OSError,KeyError,AssertionError,DispatchError) as exc:
        print('Demand Response : '+str(exc),flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
