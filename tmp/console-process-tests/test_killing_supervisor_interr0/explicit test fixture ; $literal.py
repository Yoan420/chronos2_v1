
"""Explicit test fixture: bounded local process, no scientific computation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

mode, output, release = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
with (output / "launches.txt").open("a", encoding="utf-8") as stream:
    stream.write("launched\n")
(output / "fixture.pid").write_text(str(os.getpid()), encoding="utf-8")
print("INFO explicit test fixture started", flush=True)
if mode == "secrets":
    print("api_key=synthetic-secret-value", flush=True)
    print("-----BEGIN PRIVATE KEY-----", flush=True)
    print("SYNTHETIC_PRIVATE_KEY_BODY_NO_REAL_SECRET", flush=True)
    print("-----END PRIVATE KEY-----", flush=True)
if mode == "child":
    code = """
import os, sys, time
from pathlib import Path
output = Path(sys.argv[1])
(output / 'child.pid').write_text(str(os.getpid()), encoding='utf-8')
while True:
    (output / 'heartbeat.txt').write_text(str(time.time()), encoding='utf-8')
    time.sleep(0.08)
"""
    subprocess.Popen([sys.executable, "-u", "-c", code, str(output)])
if mode in {"slow", "child"}:
    deadline = time.monotonic() + 45
    while not Path(release).exists():
        if time.monotonic() >= deadline:
            print("ERROR fixture release timeout", flush=True)
            sys.exit(19)
        time.sleep(0.04)
if mode == "fail":
    print("ERROR deliberate fixture failure", flush=True)
    sys.exit(7)
(output / "result.json").write_text(json.dumps({"fixture": True, "value": 42}), encoding="utf-8")
print("INFO explicit test fixture completed", flush=True)
