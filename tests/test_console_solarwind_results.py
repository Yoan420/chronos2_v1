import hashlib
import json

import pytest

from experiment_console.solarwind_results import build_solarwind_results, read_solarwind_artifact


def publish(root):
    folder = root / 'runs/solarwind_interaction40/2026-09-30'
    folder.mkdir(parents=True)
    files = {'index.html': b'<html>SolarWind</html>', 'forecast_de.csv': b'timestamp,q50\na,1\n',
             'forecast_nl.csv': b'timestamp,q50\na,2\n'}
    for name, content in files.items():
        (folder / name).write_bytes(content)
    manifest = {'model': 'solarwind_interaction40', 'delivery_day': '2026-09-30', 'status': 'COMPLETE',
                'files': {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    return folder


def test_complete_results_are_separate_from_nuclear(tmp_path):
    publish(tmp_path)
    (tmp_path / 'runs/exports').mkdir()
    results = build_solarwind_results(tmp_path)
    assert [day['date'] for day in results['days']] == ['2026-09-30']
    day = results['days'][0]
    assert [zone['zone'] for zone in day['zones']] == ['DE', 'NL']
    assert read_solarwind_artifact(tmp_path, day['report']['path'])[1] == b'<html>SolarWind</html>'
    assert list((tmp_path / 'runs/exports').iterdir()) == []


def test_modified_or_partial_report_is_not_served(tmp_path):
    folder = publish(tmp_path)
    (folder / 'forecast_de.csv').write_bytes(b'modified')
    assert not build_solarwind_results(tmp_path)['days']
    with pytest.raises(ValueError, match='modifiée'):
        read_solarwind_artifact(tmp_path, 'solarwind_interaction40/2026-09-30/index.html')
    (folder / 'manifest.json').unlink()
    assert not build_solarwind_results(tmp_path)['days']


@pytest.mark.parametrize('path', ['../config.yaml', 'reports/model_storm/CWE_Model_Storm_2026-09-30.html',
                                 'solarwind_interaction40/2026-02-30/index.html',
                                 'solarwind_interaction40/2026-09-30/manifest.json'])
def test_only_allowlisted_publications_can_be_read(tmp_path, path):
    with pytest.raises(ValueError):
        read_solarwind_artifact(tmp_path, path)
