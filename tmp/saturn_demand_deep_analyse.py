"""Local analysis of read-only Saturn discovery artifacts, no API calls."""
import json
import re
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parent
catalog=json.loads((ROOT/'saturn_demand_deep_catalog.json').read_text(encoding='utf8'))
terms=re.compile(r'break.?even|smelt|alumin|steel|paper|switch|parity|merit|curtail|price.?sens|elastic|willing|afford|load.?shed|scarcity',re.I)
named=[dict(name=r[0],source=src) for src,rows in catalog.items() for r in rows if terms.search(r[0]) and r[0].lower().startswith('power.')]
print('NAMED_POWER',json.dumps(named,ensure_ascii=True),flush=True)
allmeta=[]
for file in ROOT.glob('saturn_demand_deep_numeric_energyscan.jsonl'):
    for line in file.read_text(encoding='utf8').splitlines():
        try: allmeta.append(json.loads(line))
        except json.JSONDecodeError: pass  # A concurrently flushed final line can be incomplete.
out=[]
for item in allmeta:
    m=item.get('metadata') or {}
    if not isinstance(m,dict): continue
    market=m.get('mercure:market','')
    text=' '.join(str(m.get(k,'')) for k in ('mercure:label','mercure:description','mercure:source','mercure:provider'))
    if str(market).upper() in ('POWER','ELECTRICITY') or re.search(r'EPEX|NORD.?POOL|ELEC|ELECTRI|PWR|EEX_POWER',text,re.I):
        out.append(dict(name=item['name'],status=item['status'],
            **{k.replace('mercure:',''):m.get(k) for k in ('mercure:market','mercure:label','mercure:description','mercure:unit','mercure:country','mercure:source','mercure:discontinued')}))
print('META_STATUS',dict(Counter(str(i['status']) for i in allmeta)))
direct=re.compile(r'bid|ask|buy|sell|curve|effac|destruct|elastic|curtail|interrupt|merit|price.?sens',re.I)
print('POWER_META_COUNT',len(out))
hits=[r for r in out if direct.search(str(r['label'])+' '+str(r['description']))]
print('POWER_DIRECT_HITS',len(hits),json.dumps(hits,ensure_ascii=True))
(ROOT/'saturn_demand_deep_numeric_energyscan_power_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
