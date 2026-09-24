"""Bounded read-only Saturn diagnosis. Only output is a new tmp directory."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from chronos2_modular.saturn import create_saturn_client, normalize_saturn_series


def main():
    logging.disable(logging.CRITICAL)
    out = ROOT / 'tmp' / ('nyx_observation_diagnostic_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(exist_ok=False)
    nuclear = yaml.safe_load((ROOT / 'config/nuclear_forecast.yaml').read_text(encoding='utf-8'))
    start, end = pd.Timestamp('2026-08-31T00:00:00Z'), pd.Timestamp('2026-09-12T23:00:00Z')
    expected = pd.date_range(start, pd.Timestamp('2026-09-12T21:00:00Z'), freq='h')
    report = {'diagnostic_only': True, 'scientific_inputs_modified': False,
              'requested_start_utc': str(start), 'requested_end_utc': str(end), 'zones': {}}
    for zone in ['DE', 'FR']:
        config = yaml.safe_load((ROOT / nuclear['zone_configs'][zone]).read_text(encoding='utf-8'))
        data = config['data']
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            client = create_saturn_client(data['saturn_url'], data['saturn_author'])
        original_request = client.session.request

        def bounded_request(method, url, **kwargs):
            kwargs.setdefault('timeout', (10, 35))
            return original_request(method, url, **kwargs)

        client.session.request = bounded_request
        canonical_name = config['zones'][zone]['target']['series']
        names = {'canonical': canonical_name, 'post_auction': f'power.price.{zone.lower()}.euromwh.h.obs.epex'}
        modes = {'latest': None,
                 'at_failed_step_end': pd.Timestamp('2026-09-11T13:29:45Z' if zone == 'DE' else '2026-09-11T13:30:11Z'),
                 'at_previous_valid_snapshot': pd.Timestamp('2026-09-11T13:06:03Z' if zone == 'DE' else '2026-09-11T13:06:33Z')}
        result = {'series': names, 'insertions': {}, 'queries': {}, 'comparisons': {}}
        curves = {}
        for kind, name in names.items():
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    dates = client.insertion_dates(name, from_insertion_date=pd.Timestamp('2026-09-11T13:00:00Z'),
                                                   to_insertion_date=pd.Timestamp('2026-09-11T13:35:00Z'),
                                                   from_value_date=start, to_value_date=end, nocache=True)
                result['insertions'][kind] = [str(value) for value in dates] if isinstance(dates, list) else None
            except Exception as error:
                result['insertions'][kind] = {'error_type': type(error).__name__}
            for label, revision in modes.items():
                key = label + '_' + kind
                query = {'revision_date': str(revision) if revision is not None else None,
                         'extracted_at_utc': datetime.now(timezone.utc).isoformat(), 'nocache': True, 'live': True}
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        raw = client.get(name, revision_date=revision, from_value_date=start, to_value_date=end,
                                         nocache=True, live=True, _keep_nans=True)
                        values = normalize_saturn_series(raw, name, 'Europe/Paris', naive_timezone='UTC')
                    values.index = values.index.tz_convert('UTC')
                    values = values.loc[(values.index >= start) & (values.index <= end)]
                    curves[key] = values
                    raw_index = pd.DatetimeIndex(raw.index)
                    cadence = pd.Series(raw_index.asi8).diff().dropna().div(1e9).value_counts().sort_index()
                    query.update(rows=len(values), missing=int(values.isna().sum()),
                                 raw_index_timezone=str(raw_index.tz),
                                 raw_cadence_seconds={str(k): int(v) for k, v in cadence.items()},
                                 raw_non_hourly_labels=int(((raw_index.minute != 0) | (raw_index.second != 0)).sum()))
                    pd.DataFrame({'timestamp': values.index, 'value': values.to_numpy(float)}).to_parquet(out / f'{zone}_{key}.parquet', index=False)
                except Exception as error:
                    query['error_type'] = type(error).__name__
                result['queries'][key] = query
                print(json.dumps({'zone': zone, 'query': key, **query}), flush=True)
        for label in modes:
            canonical, alternate = curves.get(label + '_canonical'), curves.get(label + '_post_auction')
            if canonical is None or alternate is None:
                continue
            paired = pd.concat([canonical.rename('canonical'), alternate.rename('post_auction')], axis=1).reindex(expected)
            finite = np.isfinite(paired.to_numpy(float)).all(axis=1)
            missing = paired.index[~np.isfinite(paired['canonical'].to_numpy(float))]
            first_missing = missing[0] if len(missing) else None
            validation = (pd.date_range(first_missing-pd.Timedelta(days=7), first_missing-pd.Timedelta(hours=1), freq='h')
                          .union(paired.index[(paired.index >= first_missing) & paired['canonical'].notna()])) if first_missing is not None else expected
            checked = paired.reindex(validation).dropna()
            checked['absolute_difference'] = (checked.canonical-checked.post_auction).abs()
            changed = checked.loc[checked.absolute_difference > 1e-9].sort_values('absolute_difference', ascending=False)
            checked.reset_index(names='timestamp').to_parquet(out / f'{zone}_{label}_comparison.parquet', index=False)
            result['comparisons'][label] = {
                'paired_hours': int(finite.sum()), 'missing_canonical_hours': len(missing),
                'first_missing_utc': str(first_missing) if first_missing is not None else None,
                'missing_canonical_utc': [str(value) for value in missing],
                'validation_hours': len(validation), 'validation_paired_hours': len(checked),
                'divergent_hours': len(changed),
                'maximum_absolute_difference': float(checked.absolute_difference.max()) if len(checked) else None,
                'largest_pairs': [{**record, 'timestamp': str(record['timestamp'])} for record in changed.head(12).reset_index(names='timestamp').to_dict('records')],
            }
        report['zones'][zone] = result
        (out / 'diagnosis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output_directory': str(out), 'comparisons': {zone: value['comparisons'] for zone, value in report['zones'].items()}}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
