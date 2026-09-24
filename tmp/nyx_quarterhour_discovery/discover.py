"""Small read-only Saturn structural/observation probe; no hourly synthesis."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from nyx_intrahour.saturn_sources import make_client

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['metadata','probe'])
    p.add_argument('--series',action='append',required=True)
    p.add_argument('--start-day',default='2025-10-25')
    p.add_argument('--end-day',default='2025-10-27')
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('native_discovery',ROOT/'tmp/nyx_intrahour_discovery/discover.py')
    discovery=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(discovery)
    discovery.HERE=HERE
    client=make_client()
    try:
        if a.mode=='metadata':
            discovery.metadata(client.session,a.series)
            return
        findings=[]
        for name in a.series:
            entry={'series':name,'start_day':a.start_day,'end_day':a.end_day,'vintage':'latest_observations'}
            begin=pd.Timestamp(a.start_day,tz='Europe/Paris').tz_convert('UTC')
            end=(pd.Timestamp(a.end_day)+pd.Timedelta(days=1)).tz_localize('Europe/Paris').tz_convert('UTC')
            try:
                raw=client.get(name,from_value_date=begin-pd.Timedelta(hours=3),to_value_date=end+pd.Timedelta(hours=3),_keep_nans=True)
                if raw is None or len(raw)==0:
                    entry['raw_rows']=0
                else:
                    ix=pd.DatetimeIndex(raw.index)
                    entry.update(raw_rows=len(raw),raw_timezone=str(ix.tz),raw_start=str(ix.min()),raw_end=str(ix.max()),step_seconds=ix.to_series().diff().dt.total_seconds().value_counts().to_dict(),minutes=sorted(set(ix.minute)),nulls=int(raw.isna().sum()),duplicates=int(ix.duplicated().sum()))
                    safe_name=name.replace('.','_')
                    raw.rename('actual_15m').to_csv(HERE/(safe_name+'_'+a.start_day+'.csv'))
            except Exception as exc:
                entry['error_type']=type(exc).__name__
            print(json.dumps(entry,default=str),flush=True)
            findings.append(entry)
        discovery.write_json(HERE/('probe_'+a.start_day+'.json'),findings)
    finally:
        client.session.close()

if __name__=='__main__':
    main()
