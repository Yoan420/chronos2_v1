import json
from pathlib import Path
import sys

from chronos2_hourly import nyx_annual_pipeline as pipeline


def test_collectors_detailed_failure_survives_in_pipeline_status(tmp_path, monkeypatch):
    root = tmp_path / "NYX espace"
    root.mkdir()
    script = root / "run_nyx_annual_daily_capture.py"
    result = {"sources": {"jao_initial": {"state": "COMPLETE"},
        "public_hydro": {"state": "ERROR", "error": "Hydro captured after cutoff"},
        "lagged_exchange": {"state": "ERROR", "error": "No capture for 2026-09-30"}}}
    script.write_text("import sys\nprint(" + repr(json.dumps(result)) + ", flush=True)\nsys.exit(1)\n")
    monkeypatch.setattr(pipeline, "ROOT", root)
    state = pipeline.run("2026-09-30", action="bootstrap", output=root / "result")
    saved = json.loads((root / "result.pipeline.json").read_text(encoding="utf-8"))
    assert state == saved
    assert saved["state"] == "BLOCKED"
    message = saved["stages"][0]["error"]
    assert "public_hydro: Hydro captured after cutoff" in message
    assert "lagged_exchange: No capture for 2026-09-30" in message
    assert "see preceding source error" not in message


def test_native_stderr_warning_does_not_fail_a_successful_source(capsys):
    code, reason = pipeline._execute_source([sys.executable, "-c",
        "import sys; print('données reçues'); print('warning only', file=sys.stderr)"])
    assert code == 0 and reason is None
    console = capsys.readouterr().out
    assert "données reçues" in console and "warning only" in console


def test_recovered_profile_retry_logs_do_not_fail_successful_source(capsys, tmp_path, monkeypatch):
    retry = {"source": "saturn", "event": "profile_retry", "profile_day": "2024-08-16",
             "error": "historical profile empty"}
    complete = {"state": "COMPLETE", "source_group": "saturn",
                "profile_history_policy": "own_origin_with_current_fit_recovery_v1"}
    script = "print(" + repr(json.dumps(retry)) + "); print(" + repr(json.dumps(complete)) + ")"
    (tmp_path / "run_nyx_annual_daily_capture.py").write_text(script, encoding="utf-8")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "source_commands", lambda *a: [])
    state = pipeline.run("2026-09-30", action="capture", output=tmp_path / "result")
    assert state["state"] == "CAPTURED"
    assert state["stages"] == [{"name": "capture_public_sources", "state": "COMPLETE"}]
    output = capsys.readouterr().out
    assert "profile_retry" in output and "COMPLETE" in output


def test_source_traceback_is_preserved_but_secret_is_redacted(capsys):
    code, reason = pipeline._execute_source([sys.executable, "-c",
        "raise ValueError('network failed api_key=example-secret')"])
    assert code != 0
    assert "network failed" in reason and "example-secret" not in reason
    assert "example-secret" not in capsys.readouterr().out


def test_structured_failure_redacts_before_truncation():
    text = json.dumps({"sources": {"saturn": {"error": "password=example-secret"}}})
    assert "example-secret" not in pipeline._source_failure(text)
    assert pipeline._source_failure(json.dumps({"state": "COMPLETE"})) is None
    assert pipeline._source_failure("not JSON") is None


def test_sensitive_nested_errors_are_redacted_in_stream_and_summary():
    nested = json.dumps({"sources": {"saturn": {"error": "password=example-secret"}}})
    assert "example-secret" not in pipeline._safe_source_line(nested)
    assert "example-secret" not in pipeline._safe_source_line("ValueError: password=example-secret\n")
