"""Refresh only the rolling HTML presentation, using already embedded metrics."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chronos2_hourly.model_storm_rolling_view import CSS, render_rolling_section

block = re.compile(r'<style>\s*#rolling-performance\{.*?<script id="rolling-performance-data" type="application/json">(.*?)</script><script>.*?</script>', re.S)
payload_pattern = re.compile(r'<script id="rolling-performance-data" type="application/json">(.*?)</script>', re.S)
backup_root = ROOT / "tmp/cwe_presentation_20260921"
backup_root.mkdir(exist_ok=True)
records = []
for day in ("2026-09-19", "2026-09-22"):
    path = ROOT / "runs/reports/model_storm" / f"CWE_Model_Storm_{day}.html"
    original = path.read_bytes()
    source = original.decode("utf-8")
    matches = list(block.finditer(source))
    assert len(matches) == 1, (path, len(matches))
    match = matches[0]
    data = json.loads(match.group(1))
    expected = deepcopy(data)
    expected["scope_notes"]["dashboard_cache"] = ""
    # Old reports can contain earlier wording unrelated to this request.
    # Keep all of it intact and replace only the requested presentation pieces.
    section = match.group(0)
    title_pattern = r'<h2 class="rolling-title" id="rolling-performance-title">.*?</h2>'
    title = re.search(title_pattern, render_rolling_section(data), re.S).group(0)
    section, count = re.subn(title_pattern, lambda _: title, section, count=1, flags=re.S)
    assert count == 1
    new_title_css = "\n".join(line for line in CSS.splitlines()
                               if line.startswith("#rolling-performance>.rolling-title")
                               or line.startswith("#rolling-performance .rolling-construction"))
    section, count = re.subn(r'#rolling-performance>\.rolling-title\{[^}]*\}',
                            lambda _: new_title_css, section, count=1)
    assert count == 1
    section, count = re.subn(r'<p class="rolling-scope-note"[^>]*>.*?</p>',
                            '<p class="rolling-scope-note" hidden></p>', section, count=1, flags=re.S)
    assert count == 1
    encoded = json.dumps(expected, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    section = payload_pattern.sub(lambda _: '<script id="rolling-performance-data" type="application/json">'
                                  + encoded + '</script>', section, count=1)
    old_js = "if(note)note.textContent=scope==='dashboard_cache'?data.scope_notes.dashboard_cache:data.scope_notes.internal_completed;"
    new_js = "if(note){note.textContent=data.scope_notes?.[scope]||'';note.hidden=!note.textContent;}"
    assert section.count(old_js) == 1
    section = section.replace(old_js, new_js)
    updated_data = json.loads(payload_pattern.search(section).group(1))
    assert updated_data == expected, "Only the requested scope note may change in the data."
    updated = source[:match.start()] + section + source[match.end():]
    assert "Dashboard scope: prix Storm .da.cache" not in updated
    assert 'aria-label="En cours de construction"' in updated
    assert '<p class="rolling-scope-note" hidden></p>' in updated
    assert "note.hidden=!note.textContent" in updated
    backup = backup_root / path.name
    if not backup.exists():
        backup.write_bytes(original)
    temporary = path.with_suffix(".presentation.tmp")
    temporary.write_bytes(updated.encode("utf-8"))
    temporary.replace(path)
    records.append({"report": str(path), "backup": str(backup),
                    "metrics_unchanged": True, "outside_rolling_unchanged": True,
                    "before_sha256": hashlib.sha256(original).hexdigest(),
                    "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(backup_root / "verification.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
print(json.dumps(records, indent=2))
