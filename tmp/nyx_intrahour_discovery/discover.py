"""Read-only, bounded Saturn discovery. Writes only beside this script."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
import requests
from tshistory_lite import Client

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = 'https://saturn-energyscan.gem.myengie.com//api'
CATALOG = ROOT / 'tmp/saturn_demand_deep_catalog.json'
SELECTED = [
    *(f'power.stp.wind_production.{z}.mw.qh.fcst.cet.meteologica' for z in ('be','de','fr','nl')),
    'power.stp.da_solar_production.be.mw.qh.fcst.elia',
    'power.stp.da_total_load.be.mw.qh.fcst.cet.elia',
    'power.stp.da_total_load.nl.mw.qh.fcst.cet.entsoe',
    'power.fr.demand.peak.gw.fcst.quarter.hourly',
]

class ReadOnlySession(requests.Session):
    def request(self, method, url, *args, **kwargs):
        if method.upper() != 'GET':
            raise ValueError('Discovery permits GET only')
        kwargs.setdefault('timeout', (10,45))
        return super().request(method,url,*args,**kwargs)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf8')

def safe_error(exc):
    # Never persist response bodies, proxy URLs, or credentials in errors.
    return type(exc).__name__

def catalog():
    doc=json.loads(CATALOG.read_text(encoding='utf8'))
    return {v[0]: {'kind':v[1], 'source':src} for src, vals in doc.items() for v in vals}

def metadata(session, names):
    known=catalog()
    findings=[]
    for name in names:
        if name not in known:
            raise ValueError('Series absent from cached catalog')
        entry={'series':name, **known[name]}
        for endpoint in ('metadata','formula'):
            try:
                r=session.get(BASE+'/series/'+endpoint,params={'name':name, 'all':1} if endpoint=='metadata' else {'name':name})
                entry[endpoint+'_http_status']=r.status_code
                if r.status_code==200:
                    payload=r.json()
                    # Metadata only approved structural fields; formula is public series algebra.
                    if endpoint=='metadata' and isinstance(payload,dict):
                        payload={k:v for k,v in payload.items() if k.lower() in {'tzaware','index_type','value_type','index_dtype','value_dtype','frequency','freq','unit','units','timezone','source','description'}}
                    entry[endpoint]=payload
            except Exception as exc:
                entry[endpoint+'_error']=safe_error(exc)
        findings.append(entry)
        print(json.dumps(entry,ensure_ascii=True), flush=True)
    label=hashlib.sha256(json.dumps(names).encode()).hexdigest()[:10]
    write_json(HERE/f'metadata_{label}.json',findings)

def probe(client, names, days):
    results=[]
    rows=[]
    for name in names:
        if name not in catalog():
            raise ValueError('Series absent from cached catalog')
        for day in days:
            start=pd.Timestamp(day,tz='Europe/Paris')
            end=start+pd.DateOffset(days=1)
            cutoff=(start-pd.DateOffset(days=1))+pd.Timedelta(hours=8)
            item={'series':name,'day':day,'cutoff_utc':cutoff.tz_convert('UTC').isoformat()}
            try:
                raw=client.get(name, revision_date=cutoff.tz_convert('UTC'),from_value_date=start-pd.Timedelta(hours=3),to_value_date=end+pd.Timedelta(hours=3),_keep_nans=True)
                item['raw_count']=0 if raw is None else len(raw)
                if raw is not None and len(raw):
                    ix=pd.DatetimeIndex(raw.index)
                    item.update(raw_timezone=str(ix.tz),raw_min=str(ix.min()),raw_max=str(ix.max()),raw_minutes=sorted(set(ix.minute)),raw_step_seconds=ix.to_series().diff().dt.total_seconds().value_counts().head(8).to_dict(),nan_count=int(raw.isna().sum()),duplicate_count=int(ix.duplicated().sum()))
                    if ix.tz is None:
                        # No guessed civil timezone for ambiguous DST: preserve raw evidence only.
                        crop=(ix>=start.tz_localize(None)) & (ix<end.tz_localize(None))
                    else:
                        crop=(ix>=start) & (ix<end)
                    cut=raw[crop]
                    item['delivery_count']=len(cut)
                    item['delivery_distinct_values']=int(cut.nunique())
                    item['complete_quarters_expected']=int((end.tz_convert('UTC')-start.tz_convert('UTC')).total_seconds()/900)
                    if len(cut):
                        vals=cut.to_numpy()
                        item['within_hour_nonconstant_count']=int(sum(len(set(vals[i:i+4]))>1 for i in range(0,len(vals),4)))
                        rows.extend({'series':name,'day':day,'cutoff_time_utc':item['cutoff_utc'],'raw_timestamp':str(t),'value':None if pd.isna(v) else float(v)} for t,v in cut.items())
            except Exception as exc:
                item['error']=safe_error(exc)
            results.append(item)
            print(json.dumps(item,default=str),flush=True)
    label=hashlib.sha256(json.dumps([names,days]).encode()).hexdigest()[:10]
    write_json(HERE/f'probe_{label}.json',results)
    pd.DataFrame(rows).to_csv(HERE/f'raw_{label}.csv',index=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['metadata','probe'])
    p.add_argument('--series',action='append')
    p.add_argument('--day',action='append')
    a=p.parse_args()
    session=ReadOnlySession()
    client=Client(BASE,author='nyx-intrahour-readonly')
    client.session.close()
    client.session=session
    try:
        if a.mode=='metadata':
            metadata(session,a.series or SELECTED)
        else:
            probe(client,a.series or SELECTED,a.day or ['2026-09-15'])
    finally:
        session.close()

if __name__=='__main__':
    main()
