import sys,time
from pathlib import Path
print("[Nuclear] ECHEC : Saturn ProxyError: HTTPConnectionPool(host='127.0.0.1', port=9): Unable to connect to proxy", flush=True)
print('[Nuclear Kalman] ECHEC sources : returncode=2', flush=True)
print('Journal et statut du batch : runs/logs/nuclear_kalman/explicit_fixture', flush=True)
output=Path(sys.argv[1]); output.mkdir(parents=True,exist_ok=True)
(output/'fixture-ready').write_text('Explicit test fixture only')
if sys.argv[2]=='cancel': time.sleep(20)
raise SystemExit(0 if sys.argv[2]=='success' else 2)
