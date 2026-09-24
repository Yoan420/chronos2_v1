"""Read-only Saturn endpoint discovery; no series or metadata writes."""
import json
import re
import sys
from pathlib import Path
import requests

BASE = 'https://saturn-energyscan.gem.myengie.com'
session = requests.Session()
rows = []
for suffix in (sys.argv[1:] or ('/', '/api', '/api/swagger.json', '/api/series/find')):
    response = session.get(BASE+suffix, timeout=(10,40))
    row = dict(path=suffix, status=response.status_code, content_type=response.headers.get('Content-Type'), bytes=len(response.content))
    if 'html' in row['content_type']:
        row['links'] = re.findall(r'(?:src|href)=[\"\']([^\"\']+)', response.text)[:45]
        row['title'] = re.findall(r'<title>(.*?)</title>',response.text,re.S)[:1]
        row['route_hints'] = [line.strip()[:240] for line in response.text.splitlines() if any(s in line.lower() for s in ('url:', 'fetch(', 'action=', '.get(', 'ajax', 'window.location', 'iframe', 'tsview'))][:40]
    elif 'json' in row['content_type']:
        value = response.json()
        row['keys'] = list(value)[:30] if isinstance(value,dict) else []
        if suffix.endswith('swagger.json') and isinstance(value,dict):
            row['all_paths'] = list(value.get('paths',{}))
            row['paths'] = {k:v for k,v in value.get('paths',{}).items() if any(s in k for s in ('find','search','metadata','catalog'))}
        if isinstance(value,dict) and 'errors' in value:
            row['errors'] = value['errors']
    elif 'javascript' in row['content_type']:
        strings = re.findall(r'[\"\']([^\"\'\n]{1,160})[\"\']',response.text)
        row['route_hints'] = sorted(set(s for s in strings if any(t in s.lower() for t in ('catalog','metadata','search','series','query','find'))))[:100]
        row['catalog_context'] = [response.text[max(0,m.start()-180):m.end()+180] for m in re.finditer(r'[\"\']catalog[\"\']',response.text)][:5]
    rows.append(row)
    print(json.dumps(row,ensure_ascii=True),flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
