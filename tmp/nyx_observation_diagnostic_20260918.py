"""Read recent source coverage without changing model inputs or published data."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import numpy as np
import pandas as pd
import run_nuclear_forecast as runner
from chronos2_modular.saturn import create_saturn_client
from run_multicountry_forecast import fetch_saturn_series_from_client, POST_AUCTION_OBSERVED_SERIES_BY_ZONE

out = ROOT / 'tmp' / ('nyx_observation_diagnostic_' + (sys.argv[1] if len(sys.argv) > 1 else '20260918'))
out.mkdir(exist_ok=True)
settings = runner.load_settings(ROOT / 'config/nuclear_forecast.yaml')
start = pd.Timestamp('2026-09-10', tz='Europe/Paris').tz_convert('UTC')
end = pd.Timestamp('2026-09-20', tz='Europe/Paris').tz_convert('UTC')
expected = pd.date_range(start, end, freq='h', inclusive='left')
current = expected[expected.tz_convert('Europe/Paris').date == pd.Timestamp('2026-09-19').date()]
summaries = []
for zone in ('BE', 'DE', 'FR', 'NL'):
    cfg, _, _ = runner.zone_inputs(settings, zone)
    client = create_saturn_client(cfg['data']['saturn_url'], cfg['data']['saturn_author'])
    target = cfg['zones'][zone]['target']['series']
    curves = {}
    for key, source in [('canonical', target), ('post_auction', POST_AUCTION_OBSERVED_SERIES_BY_ZONE[zone])]:
        values = fetch_saturn_series_from_client(client, source, start, end, 'Europe/Paris', naive_timezone='UTC', nocache=True, live=True)
        curves[key] = values.reindex(expected)
    frame = pd.DataFrame(curves)
    frame.to_parquet(out / f'{zone}.parquet')
    missing = frame.canonical.isna()
    fillable = missing & frame.post_auction.notna()
    paired = frame.dropna()
    delta = (paired.canonical - paired.post_auction).abs()
    record = {'zone': zone, 'extracted_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
              'canonical_source': target, 'post_auction_source': POST_AUCTION_OBSERVED_SERIES_BY_ZONE[zone],
              'canonical_last': str(frame.canonical.last_valid_index()),
              'post_auction_last': str(frame.post_auction.last_valid_index()),
              'missing_canonical_hours': int(missing.sum()), 'fillable_hours': int(fillable.sum()),
              'missing_historical_hours': int(frame.loc[expected.difference(current), 'canonical'].isna().sum()),
              'current_canonical_hours': int(frame.loc[current, 'canonical'].notna().sum()),
              'current_post_auction_hours': int(frame.loc[current, 'post_auction'].notna().sum()),
              'paired_hours': len(paired), 'max_difference': float(delta.max()),
              'divergent_hours': int((delta > 1e-9).sum()),
              'divergent_days': sorted({str(d) for d in delta.loc[delta > 1e-9].index.tz_convert('Europe/Paris').date})}
    summaries.append(record)
    print(json.dumps(record), flush=True)
(out / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
