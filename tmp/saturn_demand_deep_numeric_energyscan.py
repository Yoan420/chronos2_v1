"""Bounded read-only metadata search over otherwise opaque Saturn series."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import re
import sys
import threading
from datetime import datetime, timezone
import requests

BASE='https://saturn-energyscan.gem.myengie.com//api'
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'tmp/saturn_demand_deep_numeric_energyscan.jsonl'
catalog=json.loads((ROOT/'tmp/saturn_demand_deep_catalog.json').read_text(encoding='utf8'))
pattern=re.compile(r'^(\d+)(?:_|$)')
other={pattern.match(row[0]).group(1) for src in ('power','stp') for row in catalog[src] if pattern.match(row[0])}
groups={}
for row in catalog['energyscan']:
    match=pattern.match(row[0])
    if match and match.group(1) not in other:
        groups.setdefault(match.group(1),[]).append(row[0])
selected={min(names,key=lambda n:(len(n),n)):names for names in groups.values()}
done={}
if DEST.exists():
    for line in DEST.read_text(encoding='utf8').splitlines():
        item=json.loads(line)
        if item.get('status')==200:
            done[item['name']]=item
print(json.dumps(dict(selected_id_families=len(selected),covered_named_contracts=sum(map(len,selected.values())),existing=len(done),
                     excluded_power_stp_id_families=len(other))),flush=True)
if '--plan' in sys.argv:
    print(json.dumps(dict(bare_ids=sum(n.isdigit() for n in selected),contract_examples=[n for n in selected if not n.isdigit()][:25])),flush=True)
    raise SystemExit(0)
local=threading.local()
def fetch(name):
    if not hasattr(local,'session'):
        local.session=requests.Session()
    try:
        response=local.session.get(BASE+'/series/metadata',params={'name':name,'all':1},timeout=(10,25))
        return dict(name=name,contracts=selected[name],status=response.status_code,
            checked_at_utc=datetime.now(timezone.utc).isoformat(),
            metadata=response.json() if response.status_code==200 else None)
    except requests.RequestException as exc:
        return dict(name=name,status='transport_error',error_type=type(exc).__name__)

count=0
with DEST.open('a',encoding='utf8') as out, ThreadPoolExecutor(max_workers=2) as pool:
    futures={pool.submit(fetch,name):name for name in selected if name not in done}
    for future in as_completed(futures):
        item=future.result()
        out.write(json.dumps(item,ensure_ascii=False)+'\n');out.flush()
        count+=1
        if count%20==0 or count==len(futures):
            print(json.dumps(dict(processed=count,total=len(futures))),flush=True)
items=[json.loads(line) for line in DEST.read_text(encoding='utf8').splitlines()]
terms=re.compile(r'curve|demand|supply|bid|ask|auction|effac|destruct|elastic|epex|nord.?pool|merit|order|curtail|load',re.I)
for item in items:
    if terms.search(json.dumps(item.get('metadata'),ensure_ascii=False)):
        print(json.dumps(item,ensure_ascii=True),flush=True)
print(json.dumps(dict(completed=len(items),status_counts={str(code):sum(i['status']==code for i in items) for code in set(i['status'] for i in items)})),flush=True)
