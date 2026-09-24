"""Bounded read-only source qualification; no market or production writes."""
from __future__ import annotations
from functools import partial
import json
from pathlib import Path
import re
import time
import pandas as pd
import requests
from tshistory_lite import Client

BASE = "https://saturn-energyscan.gem.myengie.com//api"
ROOT = Path(__file__).resolve().parents[1]
NAMES = [f"power.stp.DA.{z}.volume.cleared.mwh.h.epex_sftp" for z in ("de","be","fr","nl")]+[
    "power.stp.da_volume.de.mw.qh.obs.epex", "power.stp.da_price.de.eurmwh.qh.obs.epex",
    "power.price.base.bid.de.euromwh.d.obs.algotrading",
    "power.price.base.ask.de.euromwh.d.obs.algotrading",
    "power.eu.bidvolume.hupx.auction.mwh",
    "power.nrjscan.eu.bidvolume.hupx.auction.mwh",
    "power.volume.ro.da.qh.mw.buy.opcom.obs.utc",
    "power.price.de.clearing-price.da.pnl",
]


def scrub(value):
    if isinstance(value,dict):
        return {k:("[redacted]" if re.search(r"password|token|secret|authorization|credential|cookie",k,re.I) else scrub(v)) for k,v in value.items()}
    if isinstance(value,list):
        return [scrub(x) for x in value]
    return value


def main():
    records=[]
    with requests.Session() as session:
        for name in NAMES:
            row={"name":name}
            for kind,params in [("metadata",{"name":name,"all":1}), ("interval",{"name":name,"type":"interval"})]:
                response=session.get(BASE+"/series/metadata",params=params,timeout=30)
                row[kind]={"http_status":response.status_code,"payload":scrub(response.json()) if response.status_code==200 else None}
                time.sleep(.1)
            if "nrjscan.eu.bidvolume" in name or "power.stp.da_" in name:
                response=session.get(BASE+"/series/formula",params={"name":name},timeout=30)
                row["formula"]={"http_status":response.status_code,"payload":scrub(response.json()) if response.status_code==200 else None}
            records.append(row)
            print(json.dumps(row,ensure_ascii=True),flush=True)
    client=Client(BASE,author="BQ6757")
    client.session.request=partial(client.session.request,timeout=30)
    try:
        for name in NAMES[:4]+NAMES[4:5]+NAMES[6:7]+NAMES[8:9]:
            for asof in (None,pd.Timestamp("2026-09-13T06:00:00Z")):
                values=client.get(name,from_value_date=pd.Timestamp("2026-09-14"),to_value_date=pd.Timestamp("2026-09-14T23:59:59"),revision_date=asof)
                row={"name":name,"sample_delivery_day":"2026-09-14","asof":str(asof),"shape":None}
                if isinstance(values,pd.Series):
                    row.update(shape=list(values.shape),dtype=str(values.dtype),index_type=type(values.index).__name__,index_tz=str(getattr(values.index,'tz',None)),first=str(values.index.min()) if len(values) else None,last=str(values.index.max()) if len(values) else None,sample=[{"time":str(t),"value":float(v)} for t,v in values.iloc[:3].items()])
                records.append(row);print(json.dumps(row,ensure_ascii=True),flush=True)
                time.sleep(.1)
    finally:
        client.session.close()
    destination=ROOT/"tmp"/"saturn_demand_deep_direct_metadata.json"
    with destination.open("x",encoding="utf8") as out:
        json.dump(records,out,indent=2,ensure_ascii=True)


if __name__=="__main__":
    main()
