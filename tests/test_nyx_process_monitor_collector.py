"""Collector tests use only temporary experiment trees and fake process snapshots."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from nyx_process_monitor.collector import Collector, INTERACTION_RESULTS, MAX_LOG


STAMP = "2026-09-22T07:40:51.314404Z"
CREATED = datetime.fromisoformat(STAMP.replace("Z", "+00:00")).timestamp()
IDENTITY = "bfad212f23034144"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def process(pid=10, ppid=1, created=CREATED, argv=None, cpu=2):
    return {"pid": pid, "ppid": ppid, "created": created, "argv": argv or ["python.exe", "run_solar_wind_interaction.py"], "name": "python.exe", "cpu": cpu, "memory": 1024**2}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    base = root / "runs/experiments/solar_wind_interaction_v1"
    launch = base / "launcher_logs/test.launch.json"
    logs = base / "launcher_logs/test.stdout.log"
    meta = {"python_pid": 10, "python_created_utc": STAMP, "started_utc": STAMP,
            "run_identities": {"DE": IDENTITY}, "stdout": str(logs)}
    write_json(launch, meta)
    logs.write_text("phase started\n", encoding="utf-8")
    directory = base / "2026-09-22/de" / IDENTITY
    write_json(directory / "status.json", {"zone": "DE", "identity": IDENTITY, "status": "RUNNING", "phase": "kalman_replay", "updated_utc": STAMP})
    collector = Collector(root)
    monkeypatch.setattr(collector, "_processes", lambda: ({10: process()}, []))
    monkeypatch.setattr(collector, "_project_script", lambda row: None)
    return collector, root, launch, directory, meta


def test_running_has_indeterminate_percent_not_cache_zero(setup):
    collector, *_ = setup
    snapshot = collector.snapshot()
    assert snapshot["app"] == "nyx-process-monitor"
    job = snapshot["jobs"][0]
    assert job["status"] == "running"
    assert job["active"] is True
    assert job["eta"]["remaining_seconds"] > 0
    assert snapshot["refresh_seconds"] == 1
    assert job["progress"]["percent"] is None
    assert "366" in job["progress"]["label"]
    assert job["cpu_percent"] is None
    assert job["memory_mb"] == 1
    assert job["stdout_tail"] == "phase started"


def test_pid_reuse_is_not_running(setup, monkeypatch):
    collector, *_ = setup
    monkeypatch.setattr(collector, "_processes", lambda: ({10: process(created=CREATED + 5)}, []))
    job = collector.snapshot()["jobs"][0]
    assert job["status"] == "absent"
    assert job["active"] is False
    assert job["eta"] is None


def test_orphan_worker_remains_running_and_memoized(setup, monkeypatch):
    collector, _, launch, _, meta = setup
    meta["verified_workers"] = [{"pid": 20, "created_utc": STAMP}]
    write_json(launch, meta)
    worker = process(20, 10, argv=["python.exe", "loky", "--multiprocessing-fork"])
    monkeypatch.setattr(collector, "_processes", lambda: ({20: worker, 30: process(30, 20)}, []))
    job = collector.snapshot()["jobs"][0]
    assert job["status"] == "running"
    assert len(job["processes"]) == 2
    assert "descendants" in " ".join(job["warnings"])
    monkeypatch.setattr(collector, "_processes", lambda: ({30: process(30, 999)}, []))
    assert collector.snapshot()["jobs"][0]["status"] == "running"
    monkeypatch.setattr(collector, "_processes", lambda: ({30: process(30, 999, CREATED+100)}, []))
    assert collector.snapshot()["jobs"][0]["status"] == "absent"


def test_cpu_normalized_to_machine_and_descendants_counted(setup, monkeypatch):
    collector, *_ = setup
    monkeypatch.setattr("nyx_process_monitor.collector.psutil.cpu_count", lambda: 4)
    samples = iter([100.0, 105.0])
    monkeypatch.setattr("nyx_process_monitor.collector.time.monotonic", lambda: next(samples))
    monkeypatch.setattr(collector, "_processes", lambda: ({10: process(cpu=0), 20: process(20, 10, cpu=0)}, []))
    collector.snapshot()
    monkeypatch.setattr(collector, "_processes", lambda: ({10: process(cpu=1), 20: process(20, 10, cpu=4)}, []))
    job = collector.snapshot()["jobs"][0]
    assert job["cpu_percent"] == 25
    assert job["memory_mb"] == 2


def complete(directory):
    for name in INTERACTION_RESULTS:
        (directory / name).write_text("synthetic result", encoding="utf-8")
    write_json(directory / "completion.json", {"identity": IDENTITY, "files": {name: "a"*64 for name in INTERACTION_RESULTS}})
    write_json(directory / "status.json", {"identity": IDENTITY, "zone": "DE", "status": "COMPLETE", "phase": "complete", "annual_complete": True})


def test_complete_requires_inventory_and_all_files(setup, monkeypatch):
    collector, _, _, directory, _ = setup
    complete(directory)
    monkeypatch.setattr(collector, "_processes", lambda: ({}, []))
    job = collector.snapshot()["jobs"][0]
    assert job["status"] == "complete"
    assert job["progress"]["percent"] == 100
    url = job["zones"][0]["report_url"]
    assert collector.artifact(url.split("=")[1]) == directory / "report.html"
    assert collector.artifact(str(directory / "report.html")) is None
    (directory / "forecast.parquet").unlink()
    job = collector.snapshot()["jobs"][0]
    assert job["status"] != "complete"
    assert "incomplets" in " ".join(job["warnings"])


def test_error_secrets_and_command_flags_redacted(setup, monkeypatch):
    collector, _, _, directory, _ = setup
    write_json(directory / "status.json", {"identity": IDENTITY, "zone": "DE", "status": "FAILED", "phase": "blocked", "error": "password=synthetic-secret"})
    monkeypatch.setattr(collector, "_processes", lambda: ({10: process(argv=["python.exe", "script", "--api-key", "synthetic-secret"])}, []))
    job = collector.snapshot()["jobs"][0]
    assert job["status"] == "failed"
    assert "synthetic-secret" not in json.dumps(job)


def test_tail_bounded_and_external_paths_refused(setup, tmp_path):
    collector, root, launch, directory, meta = setup
    path = Path(meta["stdout"])
    path.write_text(("padding\n" * MAX_LOG) + "password=synthetic-secret\n", encoding="utf-8")
    tail, _ = collector._tail(path)
    assert len(tail.splitlines()) <= 120
    assert len(tail.encode()) <= MAX_LOG
    assert "synthetic-secret" not in tail
    external = tmp_path / "secret.log"
    external.write_text("external", encoding="utf-8")
    assert collector._tail(external) == ("", None)
    assert collector._safe(root / "runs/../outside") is None


@pytest.mark.parametrize("internal", [True, False])
def test_redirected_paths_refused_even_inside_runs(setup, tmp_path, internal):
    collector, root, _, directory, _ = setup
    target = root / "runs/target" if internal else tmp_path / "external"
    target.mkdir()
    (target / "report.html").write_text("secret", encoding="utf-8")
    link = root / "runs/link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit symlink creation")
    assert collector._safe(link / "report.html") is None
    assert collector._report(link / "report.html", link) is None


def test_status_identity_and_receipt_escape_fail_closed(setup):
    collector, _, _, directory, _ = setup
    complete(directory)
    write_json(directory / "status.json", {"identity": "different", "zone": "DE", "status": "COMPLETE", "annual_complete": True})
    job = collector.snapshot()["jobs"][0]
    assert job["zones"][0]["status"] == "unknown"
    assert "incohérente" in " ".join(job["warnings"])
    complete(directory)
    receipt = json.loads((directory / "completion.json").read_text())
    receipt["files"]["../../outside"] = "a" * 64
    write_json(directory / "completion.json", receipt)
    assert collector.snapshot()["jobs"][0]["status"] != "complete"


def test_legacy_pid_without_creation_never_attached(setup):
    collector, _, launch, _, meta = setup
    meta.pop("python_created_utc")
    meta.pop("run_identities")
    write_json(launch, meta)
    job = collector.snapshot()["jobs"][0]
    assert job["status"] == "unknown"
    assert job["processes"] == []


def test_read_only_snapshot_does_not_change_files(setup):
    collector, root, *_ = setup
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    collector.snapshot()
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_explicit_delivery_argument_and_completed_duration(setup, monkeypatch):
    collector, root, launch, directory, meta = setup
    meta["arguments"] = ["--delivery-day", "2026-09-23"]
    write_json(launch, meta)
    new_directory = root / "runs/experiments/solar_wind_interaction_v1/2026-09-23/de" / IDENTITY
    new_directory.mkdir(parents=True)
    complete(new_directory)
    write_json(new_directory / "status.json", {"identity": IDENTITY, "zone": "DE", "status": "COMPLETE", "phase": "complete", "annual_complete": True, "updated_utc": "2026-09-22T08:40:51.314404Z"})
    monkeypatch.setattr(collector, "_processes", lambda: ({}, []))
    first = collector.snapshot()["jobs"][0]
    second = collector.snapshot()["jobs"][0]
    assert first["status"] == "complete"
    assert first["elapsed_seconds"] == second["elapsed_seconds"] == 3600
    assert collector.artifact(first["zones"][0]["report_url"].split("=")[1]) == new_directory / "report.html"


def test_detected_wrapper_real_and_child_group_into_one_batch(setup, monkeypatch):
    collector, root, *_ = setup
    processes = {
        50: process(50, 1, argv=["python.exe", str(root / "run_nuclear_kalman.py")]),
        51: process(51, 50, argv=["python.exe", str(root / "run_nuclear_kalman.py")]),
        52: process(52, 51, argv=["python.exe", str(root / "run_nuclear_forecast.py")]),
        53: process(53, 52, argv=["python.exe", "loky"]),
    }
    monkeypatch.setattr(collector, "_processes", lambda: (processes, []))
    monkeypatch.setattr(collector, "_project_script", lambda row: Path(row["argv"][1]).name if row["pid"] != 53 else None)
    jobs = [job for job in collector.snapshot()["jobs"] if job["kind"] == "detected"]
    assert len(jobs) == 1
    assert {p["pid"] for p in jobs[0]["processes"]} == {50, 51, 52, 53}


def test_invalid_extra_zone_cannot_count_as_complete(setup):
    collector, _, launch, directory, meta = setup
    complete(directory)
    meta["run_identities"]["NL"] = "../escape"
    write_json(launch, meta)
    assert collector.snapshot()["jobs"][0]["status"] != "complete"


def test_tail_masks_cli_and_mid_private_key(setup):
    collector, _, _, _, meta = setup
    path = Path(meta["stdout"])
    path.write_text("before\n--api-key synthetic-value --epochs 3\n", encoding="utf-8")
    tail, _ = collector._tail(path)
    assert "synthetic-value" not in tail
    assert "--epochs 3" in tail
    path.write_text("-----BEGIN PRIVATE KEY-----\n" + "SYNTHETIC-BODY\n" * MAX_LOG + "-----END PRIVATE KEY-----\nafter\n", encoding="utf-8")
    tail, _ = collector._tail(path)
    assert "SYNTHETIC-BODY" not in tail
    assert "after" in tail


def test_artifact_rechecks_after_report_disappears(setup):
    collector, _, _, directory, _ = setup
    complete(directory)
    url = collector.snapshot()["jobs"][0]["zones"][0]["report_url"]
    (directory / "report.html").unlink()
    assert collector.artifact(url.split("=")[1]) is None


def test_resolved_redirection_refused_without_host_symlink_privilege(setup, monkeypatch):
    collector, root, *_ = setup
    target = root / "runs/actual"
    alias = root / "runs/alias"
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == alias / "report.html":
            return target / "report.html"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    assert collector._safe(alias / "report.html") is None


def test_process_inventory_uses_one_parent_scan_and_only_relevant_details(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import psutil
    collector = Collector(tmp_path)
    collector._seen["previous-job"] = {40: CREATED}
    names = {10: "python.exe", 20: "conhost.exe", 30: "unrelated.exe", 40: "helper.exe", 50: "recorded-helper.exe"}
    parents = {10: 1, 20: 10, 30: 1, 40: 999, 50: 999}
    parent_calls, detail_calls = [], []

    def parent_map():
        parent_calls.append(True)
        return parents

    def iterate(attrs, ad_value=None):
        assert attrs == ["pid", "name"]
        return [SimpleNamespace(info={"pid": pid, "name": name}) for pid, name in names.items()]

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def as_dict(self, attrs, ad_value=None):
            assert "ppid" not in attrs
            detail_calls.append(self.pid)
            return {"create_time": CREATED, "cmdline": [names[self.pid]], "name": names[self.pid],
                    "cpu_times": SimpleNamespace(user=1, system=2), "memory_info": SimpleNamespace(rss=1024)}

        def create_time(self):
            return CREATED

    monkeypatch.setattr(psutil, "_ppid_map", parent_map)
    monkeypatch.setattr(psutil, "process_iter", iterate)
    monkeypatch.setattr(psutil, "Process", FakeProcess)
    monkeypatch.setattr(collector, "_launches", lambda: [(Path("metadata"), {"verified_workers": [{"pid": 50, "created_utc": STAMP}]})])
    rows, warnings = collector._processes()
    assert warnings == []
    assert len(parent_calls) == 1
    assert set(detail_calls) == {10, 20, 40, 50}
    assert set(rows) == {10, 20, 40, 50}
    assert rows[20]["ppid"] == 10
    assert rows[40]["created"] == CREATED


def test_process_detail_identity_race_is_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import psutil
    collector = Collector(tmp_path)
    monkeypatch.setattr(psutil, "_ppid_map", lambda: {10: 1})
    monkeypatch.setattr(psutil, "process_iter", lambda attrs, ad_value=None: [SimpleNamespace(info={"pid": 10, "name": "python.exe"})])

    class ReusedProcess:
        def __init__(self, pid):
            self.pid = pid

        def as_dict(self, attrs, ad_value=None):
            return {"create_time": CREATED}

        def create_time(self):
            return CREATED + 1

    monkeypatch.setattr(psutil, "Process", ReusedProcess)
    rows, warnings = collector._processes()
    assert rows == {}
    assert warnings == []
