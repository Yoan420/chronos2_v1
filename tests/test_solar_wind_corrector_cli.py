"""The new runner is opt-in and applies CPU limits before scientific imports."""
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'run_solar_wind_corrector_interaction.py'


def test_default_is_readonly_and_cpu_environment_precedes_engine(monkeypatch):
    module = ModuleType('chronos2_hourly.solar_wind_corrector_interaction')
    calls = []
    def run(**kwargs):
        assert all(os.environ[name] == '2' for name in (
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
        calls.append(kwargs)
    module.run = run
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT)])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '9')
    runpy.run_path(str(SCRIPT), run_name='__main__')
    assert calls == [{'action': 'validate', 'zones': ['DE', 'NL'], 'threads': 2, 'workers': 2}]


@pytest.mark.parametrize('arguments', [
    ['--threads', '4'], ['--workers', '4'], ['--zones', 'FR'],
    ['--zones', 'DE', 'DE'], ['--delivery-day', '2026-09-23'],
])
def test_unapproved_settings_are_rejected_before_engine_import(monkeypatch, arguments):
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), *arguments])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(SCRIPT), run_name='__main__')
    assert result.value.code == 2
