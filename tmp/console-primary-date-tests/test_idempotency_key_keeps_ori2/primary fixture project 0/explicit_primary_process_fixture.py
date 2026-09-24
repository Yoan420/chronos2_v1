import json, sys
from pathlib import Path
print('INFO explicit dated primary process fixture', flush=True)
output = Path(sys.argv[1]); output.mkdir(parents=True, exist_ok=True)
(output / 'fixture-result.json').write_text(json.dumps({'fixture': True, 'delivery_day': sys.argv[3]}))
raise SystemExit(7 if sys.argv[2] == 'failure' else 0)
