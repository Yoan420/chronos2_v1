"""Bounded read-only catalogue-ID metadata inventory, isolated diagnostic output."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import threading
import time
import requests

DIRECTORY = Path(__file__).resolve().parent
BASE = "https://saturn-energyscan.gem.myengie.com//api"
IDENTITIES = DIRECTORY / "saturn_demand_deep_power_stp_ids.json"
PROGRESS = DIRECTORY / "saturn_demand_deep_power_stp_metadata.json"
PATTERN = re.compile(r"^(\d+)(?:_.*)?$")
TERMS = re.compile(r"supply|demand|curve|bid|ask|auction|aggregat|effac|curtail|sensitiv|elastic|epex|nord.?pool|destruct|ration|merit|order.?book|achat|vente|offre",re.I)
LOCAL = threading.local()
SESSIONS = []
SESSION_LOCK = threading.Lock()


def session_for_thread():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        with SESSION_LOCK:
            SESSIONS.append(LOCAL.session)
    return LOCAL.session


def now():
    return datetime.now(timezone.utc).isoformat()


def select():
    catalog = json.loads((DIRECTORY / "saturn_demand_deep_catalog.json").read_text(encoding="utf8"))
    global_names = {n for rows in catalog.values() for n, _ in rows}
    buckets = {}
    for source in ("power","stp"):
        for name,kind in catalog[source]:
            match = PATTERN.fullmatch(name)
            if match:
                buckets.setdefault(match[1],[]).append(dict(source=source,name=name,kind=kind))
    selected = [dict(base_id=base,name=base if base in global_names else rows[0]["name"],catalog_entries=rows)
                for base,rows in sorted(buckets.items(),key=lambda item:int(item[0]))]
    with IDENTITIES.open("w",encoding="utf8") as stream:
        json.dump(dict(selected_at_utc=now(),regex=PATTERN.pattern,sources=["power","stp"],count=len(selected),records=selected),stream,indent=2)
    print("SELECTED",len(selected),"identities",str(IDENTITIES),flush=True)
    print("EXCLUSION_BASE_IDS",json.dumps([r["base_id"] for r in selected]),flush=True)


def safe(value):
    if isinstance(value,dict):
        return {k:("[redacted]" if re.search(r"password|token|secret|authorization|credential|cookie",k,re.I) else safe(v)) for k,v in value.items()}
    if isinstance(value,list):
        return [safe(x) for x in value]
    return value


def probe(record):
    start=time.monotonic()
    try:
        response=session_for_thread().get(BASE+"/series/metadata",params={"name":record["name"],"all":1},timeout=(8,20))
        metadata=safe(response.json()) if response.status_code==200 else None
        output={**record,"retrieved_at_utc":now(),"http_status":response.status_code,"metadata":metadata}
        if metadata:
            hay=" ".join(str(v) for k,v in metadata.items() if any(s in k.lower() for s in ("label","description","source","provider","country","unit","market")))
            output["matched_terms"]=sorted(set(m.group().lower() for m in TERMS.finditer(hay)))
        else:
            output["matched_terms"]=[]
    except requests.RequestException as exc:
        output={**record,"retrieved_at_utc":now(),"http_status":None,"error_type":type(exc).__name__,"metadata":None,"matched_terms":[]}
    output["elapsed_seconds"]=round(time.monotonic()-start,3)
    time.sleep(.1)
    return output


def run():
    selected=json.loads(IDENTITIES.read_text(encoding="utf8"))["records"]
    existing=json.loads(PROGRESS.read_text(encoding="utf8")) if PROGRESS.exists() else dict(started_at_utc=now(),records=[])
    records={r["base_id"]:r for r in existing["records"]}
    todo=[r for r in selected if r["base_id"] not in records]
    print("PROBE_START",len(todo),"cached",len(records),flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending={executor.submit(probe,r):r for r in todo}
        for future in as_completed(pending):
            row=future.result();records[row["base_id"]]=row
            state=dict(started_at_utc=existing["started_at_utc"],updated_at_utc=now(),selected=len(selected),completed=len(records),records=list(records.values()))
            temp=PROGRESS.with_suffix(".json.tmp")
            with temp.open("w",encoding="utf8") as out:json.dump(state,out,indent=2,ensure_ascii=True)
            temp.replace(PROGRESS)
            if row["matched_terms"] or row["http_status"]!=200:
                meta=row["metadata"] or {}
                fields={k:v for k,v in meta.items() if any(s in k.lower() for s in ("label","description","source","provider","country","unit","market"))}
                print("CANDIDATE",json.dumps(dict(id=row["base_id"],http=row["http_status"],terms=row["matched_terms"],fields=fields),ensure_ascii=True),flush=True)
            if len(records)%25==0 or len(records)==len(selected):print("PROGRESS",len(records),"/",len(selected),flush=True)
    for session in SESSIONS:
        session.close()
    print("DONE",str(PROGRESS),flush=True)


if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="select":select()
    else:run()
