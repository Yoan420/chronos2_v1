from types import SimpleNamespace
import pytest
import run_nyx_annual_scheduled as m


@pytest.mark.parametrize("code", [0, 2])
def test_scheduled_action_preserves_logs_and_failure_exit(monkeypatch, tmp_path, code):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m.sys, "executable", str(tmp_path / "Scripts/pythonw.exe"))
    def invoke(command, **kwargs):
        assert command == [str(tmp_path / "Scripts/python.exe"), "-u", str(tmp_path / "run_nyx_annual_pipeline.py"),
                           "--action", "forecast", "--delivery-day", "2026-09-30"]
        assert kwargs["shell"] is False and kwargs["stderr"] == m.subprocess.STDOUT
        kwargs["stdout"].write("qualified forecast outcome\n")
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr(m.subprocess, "run", invoke)
    assert m.launch("forecast", "2026-09-30") == code
    logs = list((tmp_path / "runs/logs/nyx_annual_cpu/2026-09-30").glob("forecast_*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "qualified forecast outcome" in text and f'"return_code": {code}' in text
