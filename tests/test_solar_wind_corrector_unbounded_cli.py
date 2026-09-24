import os
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'run_solar_wind_corrector_unbounded.py'
MODULE = 'chronos2_hourly.solar_wind_corrector_unbounded'


class PendingPredecessor(RuntimeError):
    pass


def install(monkeypatch, runner):
    module = ModuleType(MODULE)
    module.PendingPredecessor = PendingPredecessor
    module.run = runner
    monkeypatch.setitem(sys.modules, MODULE, module)
    return module


def test_readonly_default_and_thread_environment(monkeypatch, capsys):
    calls = []
    def run(**kwargs):
        assert all(os.environ[name] == '2' for name in (
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
        calls.append(kwargs)
        return {'DE': {'run_id': 'de'}, 'NL': {'run_id': 'nl'}}
    install(monkeypatch, run)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT)])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '16')
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(SCRIPT), run_name='__main__')
    assert result.value.code == 0
    assert calls == [dict(action='validate', zones=['DE', 'NL'], threads=2,
                         workers=4, min_free_memory_gb=3.0)]
    assert 'VALIDATED' in capsys.readouterr().out


@pytest.mark.parametrize('action', ['validate', 'run'])
def test_expected_pending_is_explicit_non_success(monkeypatch, capsys, action):
    def run(**kwargs):
        raise PendingPredecessor('NL not complete')
    install(monkeypatch, run)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--action', action])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(SCRIPT), run_name='__main__')
    assert result.value.code == 75
    output = capsys.readouterr().out
    assert 'PENDING_PREDECESSOR' in output and '"started": false' in output


def test_validation_errors_are_not_hidden_as_pending(monkeypatch):
    def run(**kwargs):
        raise ValueError('source checksum mismatch')
    install(monkeypatch, run)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--action', 'run'])
    with pytest.raises(ValueError, match='source checksum'):
        runpy.run_path(str(SCRIPT), run_name='__main__')


@pytest.mark.parametrize('arguments', [
    ['--threads', '4'], ['--workers', '1'], ['--workers', '16'], ['--zones', 'FR'],
    ['--zones', 'DE', 'DE'], ['--delivery-day', '2026-09-23'],
    ['--min-free-memory-gb', '0'], ['--min-free-memory-gb', '9'],
    ['--min-free-memory-gb', 'nan'], ['--min-free-memory-gb', 'inf'],
])
def test_invalid_arguments_never_import_scientific_runner(monkeypatch, arguments):
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), *arguments])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(SCRIPT), run_name='__main__')
    assert result.value.code == 2
    assert MODULE not in sys.modules
