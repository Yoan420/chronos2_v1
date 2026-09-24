"""HTTP acceptance tests using isolated files and a disabled process scheduler.

No scientific CLI is executed. Script placeholders only exercise the allowlist.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import yaml

from experiment_console.adapters import SCRIPTS
from experiment_console.manager import Manager
from experiment_console.server import ConsoleHTTPServer
from experiment_console.store import Store


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api(tmp_path):
    project = tmp_path / "isolated project"
    project.mkdir()
    for filename in SCRIPTS.values():
        (project / filename).write_text("# HTTP fixture: this script is never executed.\n", encoding="utf-8")
    source = PROJECT_ROOT / "chronos2_hourly_fr_residual_v1.yaml"
    (project / source.name).write_bytes(source.read_bytes())
    state = project / "runs" / ".experiment_console"
    manager = Manager(project, state, Path(sys.executable).resolve(), start_scheduler=False)
    server = ConsoleHTTPServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    with httpx.Client(base_url=base_url, trust_env=False, timeout=10) as client:
        yield SimpleNamespace(project=project, manager=manager, server=server, client=client,
                              headers={"X-Console-Token": server.token, "Origin": base_url})
    server.shutdown()
    server.server_close()
    manager.close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def post(api, path, body):
    return api.client.post(path, json=body, headers=api.headers)


def archive(api, name="fixture archive", **files):
    path = api.project / "runs" / "history" / name
    path.mkdir(parents=True)
    (path / "run_manifest.json").write_text(json.dumps({"run_type": "fixture", "model_id": "fixture-model"}), encoding="utf-8")
    for filename, content in files.items():
        target = path / filename
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return path


def import_via_api(api, root):
    response = post(api, "/api/import", {"root": str(root)})
    assert response.status_code == 202, response.text
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = api.client.get("/api/import-status").json()
        if not state["running"]:
            return state
        time.sleep(.01)
    pytest.fail("The small fixture import did not finish in five seconds.")


def test_bootstrap_exposes_usable_session_nonce_and_local_config(api):
    response = api.client.get("/api/bootstrap")
    assert response.status_code == 200
    data = response.json()
    assert data["token"] == api.server.token
    assert data["python_executable"] == str(Path(sys.executable).resolve())
    assert data["max_concurrency"] == 1
    assert response.headers["cache-control"] == "no-store"
    assert "same-origin" in response.headers["cross-origin-resource-policy"]


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example"},
    {"Origin": "https://evil.example"},
    {"Origin": "null"},
])
def test_cross_site_reads_are_rejected(api, headers):
    response = api.client.get("/api/bootstrap", headers=headers)
    assert response.status_code == 403
    assert api.server.token not in response.text


def test_mutation_requires_nonce_allowed_origin_and_json(api):
    request = {"adapter_id": "model_storm_report", "parameters": {"delivery_day": "2026-09-11"}}
    assert api.client.post("/api/preview", json=request).status_code == 403
    assert api.client.post("/api/preview", json=request, headers={"X-Console-Token": "wrong"}).status_code == 403
    assert api.client.post("/api/preview", json=request, headers={**api.headers, "Origin": "https://evil.example"}).status_code == 403
    assert api.client.post("/api/preview", content=json.dumps(request), headers=api.headers).status_code == 400
    assert post(api, "/api/preview", []).status_code == 400
    assert post(api, "/api/preview", request).status_code == 200
    assert not api.manager.list_runs()


def test_preview_launch_idempotency_and_queue_cancellation(api):
    response = post(api, "/api/preview", {"adapter_id": "model_storm_report", "name": "HTTP fixture", "parameters": {"delivery_day": "2026-09-11"}})
    assert response.status_code == 200, response.text
    plan = response.json()
    assert not Path(plan["run_dir"]).exists()
    body = {"plan_id": plan["id"], "idempotency_key": "http-double-click"}
    first = post(api, "/api/launch", body)
    second = post(api, "/api/launch", body)
    assert first.status_code == second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert first.json()["status"] == "queued"
    assert len(api.manager.list_runs()) == 1
    assert Path(first.json()["config_path"]).is_file()
    assert post(api, f"/api/runs/{plan['id']}/cancel", {}).status_code == 400
    cancelled = post(api, f"/api/runs/{plan['id']}/cancel", {"confirmed": True})
    assert cancelled.json()["status"] == "cancelled"


def test_effective_defaults_are_retained_for_future_duplication(api):
    response = post(api, "/api/preview", {"adapter_id": "model_storm_report", "name": "Default date"})
    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan["request"]["parameters"]["delivery_day"] == plan["config"]["parameters"]["delivery_day"]
    assert plan["request"]["config_id"] == "local-results"
    assert plan["request"]["name"] == "Default date"


def test_managed_duplicate_preserves_saved_configuration_without_launch(api):
    preview = post(api, "/api/preview", {"adapter_id": "hourly_forecast", "name": "Stored configuration"})
    assert preview.status_code == 200, preview.text
    original = post(api, "/api/launch", {"plan_id": preview.json()["id"], "idempotency_key": "original"}).json()
    source = api.project / "chronos2_hourly_fr_residual_v1.yaml"
    updated = yaml.safe_load(source.read_text(encoding="utf-8"))
    updated["hourly"]["catboost"]["iterations"] += 1
    source.write_text(yaml.safe_dump(updated), encoding="utf-8")
    duplicate = post(api, f"/api/runs/{original['id']}/duplicate", {})
    assert duplicate.status_code == 200, duplicate.text
    draft = duplicate.json()["request"]
    assert draft["duplicate_of"] == original["id"]
    prepared = post(api, "/api/preview", draft)
    assert prepared.status_code == 200, prepared.text
    new_plan = prepared.json()
    assert new_plan["config"]["hourly"] == original["config"]["hourly"]
    assert new_plan["config"]["data"] == original["config"]["data"]
    assert new_plan["output_dir"] != original["output_dir"]
    assert new_plan["config_path"] != original["config_path"]
    assert not Path(new_plan["run_dir"]).exists()
    assert len(api.manager.list_runs()) == 1
    for key in ("snapshot_config", "_snapshot_config", "command"):
        attempted = post(api, "/api/preview", {"adapter_id": "hourly_forecast", key: {"evil": "value"}})
        assert attempted.status_code == 400, (key, attempted.text)
    assert post(api, "/api/preview", {"adapter_id": "hourly_forecast", "duplicate_of": "missing-run"}).status_code == 400


def test_import_is_persistent_idempotent_and_missing_files_remain_safe(api):
    good = archive(api, **{"metrics_hourly.json": json.dumps({"metrics": [{"model": "fixture", "mae": 1.5}]})})
    broken = archive(api, "incomplete", **{"metrics_hourly.json": "{incomplete json"})
    unrecognized = api.project / "runs" / "history" / "unknown format"
    unrecognized.mkdir()
    (unrecognized / "other.txt").write_text("Not a supported run.", encoding="utf-8")
    before = {path: path.read_bytes() for path in (api.project / "runs" / "history").rglob("*") if path.is_file()}
    first = import_via_api(api, good.parent)
    assert first["error"] is None
    assert first["result"]["added"] == 2
    assert first["result"]["warnings"]
    ids = {run["output_dir"]: run["id"] for run in api.manager.list_runs()}
    second = import_via_api(api, good.parent)
    assert second["result"]["added"] == 0
    assert second["result"]["updated"] == 2
    assert {run["output_dir"]: run["id"] for run in api.manager.list_runs()} == ids
    assert {path: path.read_bytes() for path in before} == before
    persisted = Store(api.manager.store.path).list()
    assert {run["id"] for run in persisted} == set(ids.values())
    malformed = api.client.get(f"/api/runs/{ids[str(broken)]}").json()
    assert any("incomplet" in warning for warning in malformed["warnings"])
    assert malformed["status"] == "unknown"
    assert malformed["started_at"] is None
    (good / "metrics_hourly.json").unlink()
    assert api.client.get(f"/api/runs/{ids[str(good)]}/artifact", params={"path": "metrics_hourly.json"}).status_code == 404
    assert api.client.get(f"/api/runs/{ids[str(good)]}").status_code == 200
    missing = import_via_api(api, good.parent / "missing-folder")
    assert missing["result"]["added"] == 0
    assert missing["result"]["warnings"]


def test_external_artifacts_cannot_traverse_paths_or_expose_excluded_inputs(api):
    path = archive(api, **{"report.html": "<html><script>window.parent.document.body.innerHTML='changed';</script>Report</html>"})
    (path / "inputs").mkdir()
    (path / "inputs" / "hidden.txt").write_text("private-source", encoding="utf-8")
    (path.parent / "outside.txt").write_text("private-outside", encoding="utf-8")
    api.manager.import_history(path)
    run = api.manager.list_runs()[0]
    endpoint = f"/api/runs/{run['id']}/artifact"
    for relative in ("../outside.txt", "..\\outside.txt", str(path.parent / "outside.txt"), "inputs/hidden.txt"):
        response = api.client.get(endpoint, params={"path": relative})
        assert response.status_code == 404, response.text
        assert "private-source" not in response.text and "private-outside" not in response.text
    report = api.client.get(endpoint, params={"path": "report.html"})
    assert report.status_code == 200
    policy = report.headers["content-security-policy"]
    assert "sandbox" in policy and "allow-same-origin" not in policy
    assert post(api, f"/api/runs/{run['id']}/cancel", {"confirmed": True}).status_code == 400


@pytest.mark.parametrize("filename,content,secret", [
    ("settings.json", json.dumps({"nested": {"api_key": "json-private-value"}}), "json-private-value"),
    ("settings.yaml", "password: |\n  yaml-private-value\npublic: visible\n", "yaml-private-value"),
    ("settings.csv", "api_key,public\ncsv-private-value,visible\n", "csv-private-value"),
    ("settings.csv.gz", gzip.compress(b"api_key,public\ngzip-private-value,visible\n"), "gzip-private-value"),
])
def test_secret_bearing_configuration_exports_are_sanitized(api, filename, content, secret):
    path = archive(api, **{filename: content})
    before = (path / filename).read_bytes()
    api.manager.import_history(path)
    run = api.manager.list_runs()[0]
    response = api.client.get(f"/api/runs/{run['id']}/artifact", params={"path": filename, "download": "1"})
    assert response.status_code == 200, response.text
    decoded = gzip.decompress(response.content).decode() if filename.endswith(".gz") else response.text
    assert secret not in decoded
    assert "MASQUÉ" in decoded
    assert (path / filename).read_bytes() == before


def test_logs_and_annotations_are_redacted_without_mutating_history(api, monkeypatch):
    monkeypatch.setenv("CONSOLE_TEST_API_KEY", "environment-private-value")
    raw_log = "INFO password=log-private-value\nWARNING environment-private-value\n"
    path = archive(api, **{"run.log": raw_log})
    api.manager.import_history(path)
    run = api.manager.list_runs()[0]
    endpoint = f"/api/runs/{run['id']}"
    for suffix in ("/logs", "/logs?download=1"):
        response = api.client.get(endpoint + suffix)
        assert response.status_code == 200
        assert "log-private-value" not in response.text
        assert "environment-private-value" not in response.text
    note = post(api, endpoint + "/annotate", {"note": "token=annotation-private-value", "tags": ["reviewed", "reviewed"]})
    assert note.status_code == 200
    assert "annotation-private-value" not in note.text
    assert note.json()["tags"] == ["reviewed"]
    assert (path / "run.log").read_text(encoding="utf-8") == raw_log
    disallowed_import = import_via_api(api, api.project.parent)
    assert disallowed_import["error"]
    assert len(api.manager.list_runs()) == 1


@pytest.mark.parametrize('complete', [True, False])
def test_live_log_tail_never_starts_inside_a_private_key(api, complete):
    raw = ('INFO before key\n-----BEGIN PRIVATE KEY-----\n'
           + 'private-key-continuation\n' * 30000
           + ('-----END PRIVATE KEY-----\nINFO after key\n' if complete else ''))
    path = archive(api, **{'run.log': raw})
    api.manager.import_history(path)
    run = api.manager.list_runs()[0]
    response = api.client.get(f"/api/runs/{run['id']}/logs")
    assert response.status_code == 200
    assert 'private-key-continuation' not in response.text
    assert 'MASQUÉ' in response.json()['text']
    if complete:
        assert 'INFO after key' in response.json()['text']


@pytest.mark.parametrize('status_file,terminal_status', [('status.json', 'succeeded'), ('run_status.json', 'unknown')])
def test_external_run_list_refreshes_status_and_phase_without_artifact_scan(api, monkeypatch, status_file, terminal_status):
    status = {'status': 'running', 'zone': 'FR', 'stage': 'run', 'phase': 'preparation',
              'started_at_utc': '2026-09-11T08:00:00+00:00'}
    path = archive(api, **{status_file: json.dumps(status)})
    api.manager.import_history(path)
    run_id = api.manager.list_runs()[0]['id']
    assert api.manager.get(run_id)['source'] == 'external'

    def no_artifact_scan(*_args, **_kwargs):
        raise AssertionError('Polling external status must not enumerate the archive artifacts.')

    # Import already captured the files. Dashboard polling should refresh only
    # live metadata, even when a historical directory contains many artifacts.
    with monkeypatch.context() as patch:
        patch.setattr('experiment_console.artifacts.list_artifacts', no_artifact_scan)
        first = api.client.get('/api/runs')
        assert first.status_code == 200, first.text
        row = next(item for item in first.json()['runs'] if item['id'] == run_id)
        assert row['reported_status'] == 'running'
        assert row['status'] == 'unknown'
        assert row['activity'] == 'FR · run · preparation'

        status['phase'] = 'residual fit'
        (path / status_file).write_text(json.dumps(status), encoding='utf-8')
        api.server.external_refreshed = float('-inf')
        changed = api.client.get('/api/runs')
        assert changed.status_code == 200, changed.text
        row = next(item for item in changed.json()['runs'] if item['id'] == run_id)
        assert row['activity'] == 'FR · run · residual fit'

        status.update(status='completed', phase='final publication', finished_at_utc='2026-09-11T08:00:03+00:00', return_code=0)
        (path / status_file).write_text(json.dumps(status), encoding='utf-8')
        api.server.external_refreshed = float('-inf')
        finished = api.client.get('/api/runs')
        assert finished.status_code == 200, finished.text
        row = next(item for item in finished.json()['runs'] if item['id'] == run_id)
        assert row['reported_status'] == 'completed'
        assert row['status'] == terminal_status
        assert row['activity'] == 'FR · run · final publication'
        assert row['duration_seconds'] == 3
        assert row['finished_at'] == status['finished_at_utc']
    assert api.manager.get(run_id)['return_code'] == 0
    assert len(api.manager.list_runs()) == 1


def test_external_logs_discover_new_files_after_initial_import(api):
    path = archive(api, **{'initial.log': 'INFO initial log\n',
                          'status.json': json.dumps({'status': 'running'})})
    api.manager.import_history(path)
    run = api.manager.list_runs()[0]
    endpoint = f"/api/runs/{run['id']}/logs"
    assert 'initial log' in api.client.get(endpoint).json()['text']
    (path / 'subsequent.log').write_text('WARNING log created after import\n', encoding='utf-8')
    updated = api.client.get(endpoint)
    assert updated.status_code == 200
    assert 'initial log' in updated.json()['text']
    assert 'log created after import' in updated.json()['text']
    assert updated.json()['available'] is True


def test_successful_managed_report_has_no_historical_unknown_completion_warning(api):
    preview = post(api, '/api/preview', {'adapter_id': 'model_storm_report', 'name': 'Completion fixture',
                                       'parameters': {'delivery_day': '2026-09-11'}}).json()
    run = post(api, '/api/launch', {'plan_id': preview['id'], 'idempotency_key': 'completed-report'}).json()
    output = Path(run['output_dir'])
    output.mkdir()
    (output / 'report.html').write_text('<html>Explicit HTTP fixture report</html>', encoding='utf-8')
    # The process-manager tests verify actual success transitions. This isolated
    # HTTP fixture exercises presentation of a persisted supervisor receipt.
    api.manager.store.update(run['id'], status='succeeded', return_code=0,
                             started_at='2026-09-11T08:00:00+00:00',
                             finished_at='2026-09-11T08:00:03+00:00', duration_seconds=3,
                             activity='Report written')
    response = api.client.get(f"/api/runs/{run['id']}")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['source'] == 'managed'
    assert result['status'] == 'succeeded' and result['return_code'] == 0
    assert result['activity'] == 'Report written'
    assert any(artifact['name'] == 'report.html' for artifact in result['artifacts'])
    assert not any('Fin d\'exécution non attestée' in warning or 'Format de run non reconnu' in warning
                   for warning in result['warnings'])
    assert result['metrics'] == {}


def test_html_export_preserves_exact_plotly_and_sanitizes_data_scripts(api):
    from plotly.offline import get_plotlyjs
    from experiment_console.server import _ScriptSpans
    vendor = get_plotlyjs()
    data = {'api_key': 'json-report-private-value', 'label': '</script><div>public label</div>'}
    embedded = json.dumps(data).replace('<', '\\u003c')
    init = 'window.chartReady = true;'
    document = (f'<!doctype html><html><body><p>password=markup-private-value</p><script>{vendor}</script>'
                f'<script type="application/json" id="payload">{embedded}</script>'
                '<script id="plotly-trusted">window.config = {api_key: "script-private-value"};</script>'
                f'<script>{init}</script></body></html>')
    path = archive(api, **{'report.html': document})
    original_bytes = (path / 'report.html').read_bytes()
    original_text = original_bytes.decode('utf-8')
    original_vendor = _ScriptSpans(original_text).finish()[0]
    vendor_on_disk = original_text[original_vendor[1]:original_vendor[2]]
    api.manager.import_history(path)
    run = api.manager.list_runs()[0]
    response = api.client.get(f"/api/runs/{run['id']}/artifact", params={'path': 'report.html'})
    assert response.status_code == 200, response.text[:500]
    exported = response.text
    scripts = _ScriptSpans(exported).finish()
    assert len(scripts) == 3
    assert exported[scripts[0][1]:scripts[0][2]] == vendor_on_disk
    assert vendor_on_disk.replace('\r\n', '\n') == vendor
    assert exported[scripts[2][1]:scripts[2][2]] == init
    safe_data = exported[scripts[1][1]:scripts[1][2]]
    assert '</script' not in safe_data
    assert json.loads(safe_data)['label'] == data['label']
    assert json.loads(safe_data)['api_key'] == '[MASQUÉ]'
    assert 'json-report-private-value' not in exported
    assert 'script-private-value' not in exported
    assert 'markup-private-value' not in exported
    assert 'a désactivé 1 script' in exported
    assert (path / 'report.html').read_bytes() == original_bytes


def test_plotly_tag_name_never_bypasses_redaction_for_modified_vendor(api):
    from plotly.offline import get_plotlyjs
    from experiment_console.server import _ScriptSpans
    vendor = get_plotlyjs()
    document = ('<html><body><script id="plotly" data-trusted="true">' + vendor
                + '\nwindow.api_key="appended-private-value";</script><p>Report text</p></body></html>')
    path = archive(api, **{'report.html': document})
    api.manager.import_history(path)
    run = api.manager.list_runs()[0]
    response = api.client.get(f"/api/runs/{run['id']}/artifact", params={'path': 'report.html'})
    assert response.status_code == 200, response.text[:500]
    assert not _ScriptSpans(response.text).finish()
    assert 'appended-private-value' not in response.text
    assert 'a désactivé 1 script' in response.text
    assert 'Report text' in response.text
