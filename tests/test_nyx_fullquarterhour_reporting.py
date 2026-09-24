from nyx_fullquarterhour import reporting, runner
import pytest


def test_partial_regime_coverage_visible_even_if_aggregate_gate_passes(tmp_path):
    report = reporting.render_report(tmp_path, {"status": "complete", "decision": {
        "encouraging": True, "critical_regimes": {"coverage_complete": False}}}, {})
    body = report.read_text(encoding="utf-8")
    assert "Régimes extrêmes : couverture partielle" in body
    assert body.index("couverture partielle") < body.index("<details>")


def test_failed_run_has_no_success_banner_or_score_table(tmp_path):
    report = reporting.render_report(tmp_path, {"status": "failed", "reason": "Missing quarter", "decision": {"encouraging": False}}, {})
    body = report.read_text(encoding="utf-8")
    assert "Comparaison incomplète" in body and "Missing quarter" in body
    assert "<table" not in body


def test_postprocess_status_retries_transient_windows_lock(monkeypatch, tmp_path):
    calls = []
    def attempt(path, value, *, root):
        calls.append((path, value, root))
        if len(calls) < 3:
            raise PermissionError("Synthetic Windows sharing violation")
    monkeypatch.setattr(runner, "_write_json", attempt)
    monkeypatch.setattr(runner.time, "sleep", lambda delay: None)
    runner.write_json(tmp_path / "status.json", {"state": "test"}, root=tmp_path)
    assert len(calls) == 3


def test_postprocess_status_does_not_hide_persistent_io_failure(monkeypatch, tmp_path):
    calls = []
    def attempt(*args, **kwargs):
        calls.append(1)
        raise PermissionError("Synthetic persistent refusal")
    monkeypatch.setattr(runner, "_write_json", attempt)
    monkeypatch.setattr(runner.time, "sleep", lambda delay: None)
    with pytest.raises(PermissionError):
        runner.write_json(tmp_path / "status.json", {}, root=tmp_path)
    assert len(calls) == 6
