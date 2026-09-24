import os
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(params=['parallel', 'reuse'])
def launcher(request):
    kind = request.param
    return (Path(__file__).resolve().parents[1] / f'run_solar_wind_corrector_{kind}.py',
            f'chronos2_hourly.solar_wind_corrector_{kind}')


def test_readonly_default_fixes_scientific_thread_count_before_import(monkeypatch, launcher):
    script, module_name = launcher
    module = ModuleType(module_name)
    calls = []
    def run(**kwargs):
        assert all(os.environ[name] == '2' for name in (
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
        calls.append(kwargs)
    module.run = run
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, 'argv', [str(script)])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '16')
    runpy.run_path(str(script), run_name='__main__')
    assert calls == [dict(action='validate', zones=['DE', 'NL'], threads=2,
                         workers=4, min_free_memory_gb=3.0)]


@pytest.mark.parametrize('arguments', [
    ['--threads', '4'], ['--workers', '1'], ['--workers', '16'], ['--zones', 'FR'],
    ['--zones', 'DE', 'DE'], ['--delivery-day', '2026-09-23'],
    ['--min-free-memory-gb', '0'], ['--min-free-memory-gb', '9'],
    ['--min-free-memory-gb', 'nan'], ['--min-free-memory-gb', 'inf'],
])
def test_invalid_settings_rejected_before_scientific_import(monkeypatch, arguments, launcher):
    script, _ = launcher
    monkeypatch.setattr(sys, 'argv', [str(script), *arguments])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script), run_name='__main__')
    assert result.value.code == 2
