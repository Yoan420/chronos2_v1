"""Primary progress uses synthetic console/status files, never a scientific run."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiment_console.primary_progress import MAX_CONSOLE_BYTES, MAX_STATUS_BYTES, read_primary_progress


@pytest.fixture
def primary_batch(tmp_path):
    now = datetime.now(timezone.utc)
    day = (now.date() + timedelta(days=1)).isoformat()
    batch_id = (now - timedelta(seconds=4)).strftime('%Y%m%dT%H%M%S_%fZ') + '_deadbeef'
    run_dir = tmp_path / 'runs' / '.experiment_console' / 'executions' / ('a' * 32)
    run_dir.mkdir(parents=True)
    status_path = tmp_path / 'runs' / 'logs' / 'nuclear_kalman' / day / batch_id / 'status.json'
    status_path.parent.mkdir(parents=True)
    run = {'id': run_dir.name, 'run_dir': str(run_dir), 'source': 'managed',
           'adapter_id': 'primary_nuclear_kalman', 'delivery_day': day, 'status': 'running',
           'started_at': (now - timedelta(seconds=5)).isoformat(), 'finished_at': None}
    batch = {'run_id': batch_id, 'delivery_day': day, 'status_file': str(status_path),
             'status': 'running', 'started_at_utc': (now - timedelta(seconds=4)).isoformat(),
             'finished_at_utc': None, 'steps': [
                 {'name': 'sources', 'zone': None, 'status': 'complete'},
                 *[{'name': 'nuclear_kalman', 'zone': zone, 'status': 'running' if zone == 'BE' else 'pending'}
                   for zone in ['BE', 'DE', 'FR', 'NL']],
                 {'name': 'CWE_Model_Storm', 'zone': None, 'status': 'pending'}]}

    def save():
        status_path.write_text(json.dumps(batch), encoding='utf-8')

    save()
    console = run_dir / 'console.log'
    console.write_text(f'Statut du batch : {status_path}\n', encoding='utf-8')
    return tmp_path, run, batch, status_path, console, save


def test_progress_counts_real_completed_steps_and_labels_running_country(primary_batch):
    root, run, batch, _, _, save = primary_batch
    progress = read_primary_progress(root, run)
    assert (progress['completed'], progress['total'], progress['percent']) == (1, 6, 17)
    assert progress['source'] == 'batch_status' and progress['indeterminate'] is False
    assert progress['phase'] == 'Belgique · Prévision et publication'
    batch['steps'][1]['status'] = 'complete'
    batch['steps'][2]['status'] = 'running'
    save()
    progress = read_primary_progress(root, run)
    assert (progress['completed'], progress['percent']) == (2, 33)
    assert progress['phase'] == 'Allemagne · Prévision et publication'


def test_processed_failures_and_skips_reach_100_without_claiming_success(primary_batch):
    root, run, batch, _, _, save = primary_batch
    run['status'] = 'failed'
    run['finished_at'] = datetime.now(timezone.utc).isoformat()
    batch['status'] = 'failed'
    for step, status in zip(batch['steps'], ['failed', 'skipped', 'skipped', 'skipped', 'skipped', 'complete']):
        step['status'] = status
    save()
    progress = read_primary_progress(root, run)
    assert progress['percent'] == 100 and progress['completed'] == 6
    assert progress['batch_status'] == 'failed' and 'échecs' in progress['phase']
    assert progress['steps'][0]['status'] == 'failed'


def test_old_final_announcement_links_only_its_exact_batch(primary_batch):
    root, run, _, status_path, console, _ = primary_batch
    console.write_text(f'Journal et statut du batch : {status_path}\n', encoding='utf-8')
    assert read_primary_progress(root, run)['source'] == 'batch_status'


@pytest.mark.parametrize('status,indeterminate', [('queued', False), ('running', True), ('failed', False), ('unknown', False)])
def test_missing_manifest_has_no_percentage_and_never_uses_latest_status(primary_batch, status, indeterminate):
    root, run, batch, status_path, console, _ = primary_batch
    run['status'] = status
    console.write_text('No batch announcement\n', encoding='utf-8')
    (status_path.parent.parent / 'latest_status.json').write_text(json.dumps(batch), encoding='utf-8')
    progress = read_primary_progress(root, run)
    assert progress['percent'] is None and progress['total'] is None and progress['steps'] == []
    assert progress['indeterminate'] is indeterminate


@pytest.mark.parametrize('field,value', [
    ('run_id', 'another-run'), ('delivery_day', '2020-01-01'),
    ('started_at_utc', '2000-01-01T00:00:00+00:00'),
    ('started_at_utc', '2099-01-01T00:00:00+00:00'),
    ('started_at_utc', '2026-09-11T10:00:00'),
    ('status_file', 'status.json'), ('steps', [{'name': 'unknown', 'status': 'complete'}]),
    ('status', 'complete'), ('finished_at_utc', '2099-01-01T00:00:00+00:00'),
])
def test_unrelated_or_malformed_batch_is_not_associated(primary_batch, field, value):
    root, run, batch, _, _, save = primary_batch
    batch[field] = value
    save()
    progress = read_primary_progress(root, run)
    assert progress['source'] == 'process' and progress['percent'] is None
    assert 'warning' in progress


def test_duplicate_same_announcement_is_allowed_but_two_batches_are_ambiguous(primary_batch):
    root, run, _, path, console, _ = primary_batch
    console.write_text(f'Statut du batch : {path}\nJournal et statut du batch : {path}\n', encoding='utf-8')
    assert read_primary_progress(root, run)['source'] == 'batch_status'
    with console.open('a', encoding='utf-8') as stream:
        stream.write(f'Statut du batch : {path.parent.parent / "different" / "status.json"}\n')
    assert read_primary_progress(root, run)['percent'] is None


def test_console_read_is_bounded_and_retains_early_and_legacy_tail_announcements(primary_batch):
    root, run, _, path, console, _ = primary_batch
    console.write_text(f'Statut du batch : {path}\n' + 'synthetic noise\n' * MAX_CONSOLE_BYTES,
                       encoding='utf-8')
    assert read_primary_progress(root, run)['source'] == 'batch_status'
    console.write_text('synthetic noise\n' * MAX_CONSOLE_BYTES + f'Journal et statut du batch : {path}\n',
                       encoding='utf-8')
    assert read_primary_progress(root, run)['source'] == 'batch_status'


def test_oversized_status_and_outside_paths_are_rejected(primary_batch):
    root, run, _, path, console, _ = primary_batch
    path.write_bytes(b' ' * (MAX_STATUS_BYTES + 1))
    assert read_primary_progress(root, run)['percent'] is None
    console.write_text(f'Statut du batch : {root.parent / "outside-status.json"}\n', encoding='utf-8')
    assert read_primary_progress(root, run)['percent'] is None


def test_console_must_belong_to_this_run_directory(primary_batch):
    root, run, _, _, _, _ = primary_batch
    run['id'] = 'other-run'
    assert read_primary_progress(root, run)['percent'] is None


def test_dead_process_does_not_present_stale_batch_as_current_activity(primary_batch):
    root, run, _, _, _, _ = primary_batch
    run['status'] = 'interrupted'
    progress = read_primary_progress(root, run)
    assert progress['phase'] == 'Superviseur interrompu'
    assert progress['percent'] == 17 and progress['indeterminate'] is False
    assert 'warning' in progress


def test_global_steps_without_optional_zone_key_do_not_break_status_endpoint(primary_batch):
    root, run, batch, _, _, save = primary_batch
    batch['steps'][0].pop('zone')
    batch['steps'][-1].pop('zone')
    save()
    progress = read_primary_progress(root, run)
    assert progress['source'] == 'batch_status'
    assert progress['steps'][0]['zone'] is None


@pytest.mark.parametrize('link_parent', [False, True])
def test_links_in_status_path_are_rejected(primary_batch, link_parent):
    root, run, _, path, _, _ = primary_batch
    original = path.parent if link_parent else path
    target = original.with_name(original.name + '-actual')
    original.rename(target)
    try:
        original.symlink_to(target, target_is_directory=link_parent)
    except OSError:
        pytest.skip('Creating symbolic links is unavailable for this Windows account')
    assert read_primary_progress(root, run)['percent'] is None


@pytest.mark.parametrize('location', ['status', 'status_parent', 'console_parent'])
def test_windows_reparse_attributes_block_reads_without_symlink_privileges(primary_batch, monkeypatch, location):
    root, run, _, path, console, _ = primary_batch
    blocked = {'status': path, 'status_parent': path.parent, 'console_parent': console.parent}[location]
    original_lstat = Path.lstat

    def flagged_lstat(candidate, *args, **kwargs):
        info = original_lstat(candidate, *args, **kwargs)
        if candidate == blocked:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=1024)
        return info

    monkeypatch.setattr(Path, 'lstat', flagged_lstat)
    assert read_primary_progress(root, run)['percent'] is None
