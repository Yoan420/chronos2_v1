"""Synthetic isolated launcher checks: no provider, fitting or production writes."""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import run_solar_correction as runner
from chronos2_hourly import nuclear_run_archive, solar_correction_protocol as protocol


def blocked(*args, **kwargs):
    pytest.fail("Unexpected model execution, source synchronization or production write")


def hashes(root):
    return {p.relative_to(root).as_posix(): runner.sha256(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "runs/experiments/solar_correction_v1")
    monkeypatch.setattr(runner, "SOURCE", tmp_path / "data/pit/solar_cwe")
    monkeypatch.setattr(runner, "BASELINE", tmp_path / "runs/experiments/nuclear_forecast_v1")
    cfg = {"schema_version": 1, "delivery_day": "2026-09-19", "zones": list(runner.ZONES),
           "output_root": "runs/experiments/solar_correction_v1", "source_root": "data/pit/solar_cwe",
           "historical_end": "2026-09-19", "diagnostic_only": True, "production_modified": False}
    path = tmp_path / "config/solar_correction.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg, path


@pytest.fixture
def snapshot(sandbox):
    cfg, _ = sandbox
    work = runner.OUTPUT / cfg["delivery_day"] / "fr" / "test-identity"
    input_path = work / "snapshot/target.csv"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("time,actual\n2026-09-17T00:00:00Z,1\n", encoding="utf-8")
    reference = work / "reference/frozen.json"
    reference.parent.mkdir()
    reference.write_text("{}", encoding="utf-8")
    config = {"zones": {"FR": {"target": {"file": str(input_path)}, "timezone": "Europe/Paris"}},
              "data": {"pit_files": {}}, "model": {}, "hourly": {}}
    resolved = work / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = {"identity": {"zone": "FR", "delivery_day": cfg["delivery_day"], "protocol_sha256": "p" * 64},
                "files": [{"snapshot_path": str(input_path), "sha256": runner.sha256(input_path)}],
                "reference_files": [{"snapshot_path": str(reference), "sha256": runner.sha256(reference)}],
                "resolved_config_sha256": runner.sha256(resolved)}
    runner.write_json(work / "input_snapshot.json", manifest)
    runner.write_json(work / "source_audit.json", {"chronos_recomputed": False})
    return work, config, manifest


def test_fixed_four_country_config_and_namespace(sandbox):
    cfg, path = sandbox
    assert runner.settings(path) == cfg
    assert runner.safe(runner.OUTPUT / "valid") == runner.OUTPUT / "valid"
    for unsafe in (runner.OUTPUT, runner.ROOT / "runs/exports/new", runner.OUTPUT / "../escape"):
        with pytest.raises(ValueError, match="namespace"):
            runner.safe(unsafe)
    assert runner.baseline("2026-09-19", "FR") == runner.BASELINE / "2026-09-19/fr/civil_pit_v2"
    with pytest.raises(ValueError):
        runner.baseline("2026-09-19", "ES")


@pytest.mark.parametrize("field,value", [("schema_version", True), ("zones", ["FR"]),
    ("zones", ["FR", "DE", "BE", "BE"]), ("zones", ["FR", "DE", "BE", "ES"]),
    ("historical_end", "2026-09-20"), ("output_root", "runs/exports"),
    ("source_root", "data/pit/nuclear_cwe"), ("diagnostic_only", False),
    ("production_modified", True), ("delivery_day", "../2026-09-19")])
def test_configuration_rejects_scope_changes(sandbox, field, value):
    cfg, path = sandbox
    cfg[field] = value
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        runner.settings(path)


def test_snapshot_validation_is_readonly_and_binds_zone_day(snapshot):
    work, config, _ = snapshot
    before = hashes(work)
    assert runner.verify_snapshot(work, zone="FR", day="2026-09-19") == config
    assert hashes(work) == before
    for zone, day in (("DE", "2026-09-19"), ("FR", "2026-09-20")):
        with pytest.raises(ValueError, match="zone/day"):
            runner.verify_snapshot(work, zone=zone, day=day)


@pytest.mark.parametrize("corruption", ["input", "reference", "config", "empty_files", "empty_references", "duplicate", "unsealed_target"])
def test_snapshot_rejects_changed_or_unsealed_inputs(snapshot, corruption):
    work, config, manifest = snapshot
    if corruption in ("input", "reference", "config"):
        target = {"input": work / "snapshot/target.csv", "reference": work / "reference/frozen.json",
                  "config": work / "resolved_config.yaml"}[corruption]
        target.write_bytes(target.read_bytes() + b"\nmodified")
    else:
        if corruption == "empty_files":
            manifest["files"] = []
        elif corruption == "empty_references":
            manifest["reference_files"] = []
        elif corruption == "duplicate":
            manifest["reference_files"].append(manifest["files"][0])
        else:
            config["zones"]["FR"]["target"]["file"] = str(work / "snapshot/not-sealed.csv")
            (work / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            manifest["resolved_config_sha256"] = runner.sha256(work / "resolved_config.yaml")
        runner.write_json(work / "input_snapshot.json", manifest)
    with pytest.raises(ValueError):
        runner.verify_snapshot(work, zone="FR", day="2026-09-19")


@pytest.fixture
def small_result(snapshot, monkeypatch):
    work, _, _ = snapshot
    # Semantic archive validators have dedicated 731-day tests. Keep these IO
    # and orchestration fixtures tiny without bypassing any runner checksum.
    monkeypatch.setattr(nuclear_run_archive, "_validate", lambda *args: None)
    index = pd.date_range("2026-09-16T22:00:00Z", periods=48, freq="h", name="delivery_start_utc")
    raw = pd.DataFrame({name: np.resize(np.asarray([1.123456789, 2.987654321, 3.14159265, 4.6756789], dtype="float32"), 48)
                        for name in ("q10", "q50", "q90", "actual")}, index=index)
    raw["forecast_origin_utc"] = (index.tz_convert("Europe/Paris").normalize()
                                 - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_convert("UTC")
    frames = {name: raw.copy() for name in nuclear_run_archive._FRAMES}
    audit = {"zone": "FR", "delivery_day": "2026-09-19", "raw_history_start_day": "2026-09-17"}
    view = SimpleNamespace(backtest=frames["kalman_backtest"], forecast=frames["kalman_forecast"],
                           replay=SimpleNamespace(audit={}))
    result = SimpleNamespace(**{name: frames[name] for name in nuclear_run_archive._FRAMES[:5]},
                             kalman_view=view, audit=audit)
    return work, raw, result


def frozen_reference(work, raw, result):
    directory = work / "reference/baseline"
    directory.mkdir(parents=True)
    frames, audits = nuclear_run_archive._collect(result)
    frames["raw_history"] = raw.iloc[-2:]
    for name, frame in frames.items():
        frame.to_parquet(directory / (name + ".parquet"))
    runner.write_json(directory / "audits.json", audits)
    checkpoint = directory / "nuclear_chronos_oof.csv.gz"
    raw.reset_index().to_csv(checkpoint, index=False)
    receipt = {"status": "complete", "zone": "FR", "output_sha256": runner.sha256(checkpoint),
               "timezone": "Europe/Paris", "n_delivery_days": 2, "completed_days": 2,
               "n_delivery_hours": len(raw), "completed_hours": len(raw),
               "first_delivery_utc": str(raw.index[0]), "last_delivery_utc": str(raw.index[-1]),
               "first_delivery_day_local": "2026-09-17", "last_delivery_day_local": "2026-09-18"}
    runner.write_json(Path(str(checkpoint) + ".manifest.json"), receipt)
    return checkpoint, receipt


def test_full_history_recovery_preserves_exact_float32_overlap(small_result):
    work, raw, result = small_result
    frozen_reference(work, raw, result)
    before = hashes(work)
    loaded = runner.load_incumbent(work, full_history=True)
    pd.testing.assert_frame_equal(loaded.raw_history, raw, check_exact=True, check_freq=False)
    assert len(runner.load_incumbent(work).raw_history) == 2
    assert hashes(work) == before


@pytest.mark.parametrize("corruption", ["overlap", "zone", "last_day", "sha", "truncate", "truncate_prefix"])
def test_recovery_rejects_wrong_checkpoint_and_changed_overlap(small_result, corruption):
    work, raw, result = small_result
    checkpoint, receipt = frozen_reference(work, raw, result)
    if corruption in ("overlap", "truncate", "truncate_prefix"):
        changed = raw.copy()
        if corruption == "overlap":
            changed.loc[changed.index[-1], "q50"] += np.float32(1)
        elif corruption == "truncate":
            changed = changed.iloc[:-1]
        else:
            changed = changed.iloc[1:]
        changed.reset_index().to_csv(checkpoint, index=False)
        receipt["output_sha256"] = runner.sha256(checkpoint)
    else:
        key, value = {"zone": ("zone", "DE"), "last_day": ("last_delivery_day_local", "2026-09-19"),
                      "sha": ("output_sha256", "0" * 64)}[corruption]
        receipt[key] = value
    runner.write_json(Path(str(checkpoint) + ".manifest.json"), receipt)
    with pytest.raises(ValueError):
        runner.load_incumbent(work, full_history=True)


def test_archive_requires_two_arms_and_reloads_identical_bytes(small_result):
    work, raw, result = small_result
    with pytest.raises(ValueError, match="Both"):
        runner.archive_results(work, {"residual": result})
    assert not (work / "results").exists()
    loaded, record = runner.archive_results(work, {variant: result for variant in runner.VARIANTS})
    assert set(loaded) == set(runner.VARIANTS)
    assert pd.Timestamp(record["sealed_at_utc"]).tzinfo is not None
    before = hashes(work)
    again, second = runner.archive_results(work)
    assert second == record and hashes(work) == before
    pd.testing.assert_frame_equal(again["residual"].raw_history, raw, check_exact=True, check_freq=False)
    with pytest.raises(ValueError, match="overwrite"):
        runner.archive_results(work, {variant: result for variant in runner.VARIANTS})
    assert hashes(work) == before
    path = work / "results/residual/kalman_forecast.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        runner.archive_results(work)


def test_report_reuses_both_archives_without_model_or_sync(sandbox, small_result, monkeypatch):
    import chronos2_modular.common as common
    from chronos2_hourly import nuclear_preparation, solar_correction_forecast, solar_correction_reporting, solar_cwe_sources

    cfg, _ = sandbox
    work, _, result = small_result
    runner.archive_results(work, {variant: result for variant in runner.VARIANTS})
    before = hashes(work / "results")
    monkeypatch.setattr(solar_correction_forecast, "run_solar_correction_forecast", blocked)
    monkeypatch.setattr(solar_cwe_sources, "ensure_solar_sources", blocked)
    monkeypatch.setattr(common, "build_zone_configs", lambda *args: [SimpleNamespace(timezone="Europe/Paris")])

    @dataclass
    class Data:
        target: pd.Series

    actual = pd.Series([10.], index=pd.date_range("2026-09-18", periods=1, tz="UTC"))
    monkeypatch.setattr(nuclear_preparation, "prepare_nuclear_zone_data", lambda *args: Data(pd.Series(dtype=float)))
    monkeypatch.setattr(runner, "_verify_reporting", lambda *args, **kwargs: (actual, {"frozen": True}))
    monkeypatch.setattr(runner, "load_incumbent", lambda *args, **kwargs: result)
    calls = []

    def render(results, **kwargs):
        assert set(results) == set(runner.VARIANTS)
        pd.testing.assert_series_equal(kwargs["data"].target, actual)
        calls.append(kwargs)
        return {"solar_residual": str(work / "reports/report.html")}

    monkeypatch.setattr(solar_correction_reporting, "render_solar_correction_reports", render)

    @contextmanager
    def quiet(*args, **kwargs):
        yield lambda *args, **kwargs: None

    monkeypatch.setattr(runner, "run_progress", quiet)
    monkeypatch.setattr(runner, "exclusive_process_lock", quiet)
    output = runner.run_zone(cfg, "FR", work, SimpleNamespace(action="report", threads=1, workers=1),
                             {"protocol_sha256": "p" * 64})
    assert output["status"] == "complete" and output["chronos_recomputed"] is False
    assert len(calls) == 1 and hashes(work / "results") == before


def test_main_status_is_readonly_even_without_preparation(sandbox, monkeypatch, capsys):
    cfg, path = sandbox
    for name in ("preregister", "prepare", "run_zone", "baseline_signature", "write_json", "evaluate"):
        monkeypatch.setattr(runner, name, blocked)
    before = hashes(runner.ROOT)
    assert runner.main(["--config", str(path), "--action", "status"]) == 0
    assert '"status": "not_prepared"' in capsys.readouterr().out
    assert hashes(runner.ROOT) == before


def test_main_status_reads_only_requested_day_pointer(sandbox, snapshot, monkeypatch, capsys):
    cfg, path = sandbox
    work, _, _ = snapshot
    pointer = runner.OUTPUT / cfg["delivery_day"] / "latest_FR_DE_BE_NL.json"
    runner.write_json(pointer, {"status": "running", "workdirs": {"FR": str(work)}})
    runner.write_json(work / "run_status.json", {"status": "running", "stage": "residual_then_two_kalman"})
    before = hashes(runner.ROOT)
    for name in ("preregister", "prepare", "run_zone", "baseline_signature", "write_json", "evaluate"):
        monkeypatch.setattr(runner, name, blocked)
    assert runner.main(["--config", str(path), "--action", "status"]) == 0
    assert '"stage": "residual_then_two_kalman"' in capsys.readouterr().out
    assert hashes(runner.ROOT) == before


def test_main_status_rejects_out_of_scope_workdir(sandbox, snapshot, monkeypatch):
    cfg, path = sandbox
    work, _, _ = snapshot
    pointer = runner.OUTPUT / cfg["delivery_day"] / "latest_FR_DE_BE_NL.json"
    runner.write_json(pointer, {"status": "running", "workdirs": {"DE": str(work)}})
    monkeypatch.setattr(runner, "write_json", blocked)
    with pytest.raises(ValueError, match="scope"):
        runner.main(["--config", str(path), "--action", "status"])


class StopBeforePreparation(RuntimeError):
    """End a synthetic queue test before any forecast preparation is possible."""


@pytest.fixture
def queue_sandbox(sandbox, snapshot, monkeypatch):
    cfg, path = sandbox
    work, _, _ = snapshot
    pointer = runner.OUTPUT / cfg["delivery_day"] / "latest_FR_DE_BE_NL.json"
    prior = {"status": "prepared", "workdirs": {"FR": str(work)},
             "results": {"FR": {"status": "prepared"}}, "production_modified": False}
    runner.write_json(pointer, prior)

    def reached_preflight(*args, **kwargs):
        raise StopBeforePreparation("Reached unchanged baseline preflight")

    monkeypatch.setattr(runner, "baseline_signature", reached_preflight)
    for name in ("preregister", "prepare", "run_zone"):
        monkeypatch.setattr(runner, name, blocked)
    return path, pointer, prior


def test_after_solar_pid_accepts_matching_command_and_waits_before_preflight(queue_sandbox, monkeypatch, capsys):
    import psutil
    import time
    path, pointer, prior = queue_sandbox
    running = iter([True, False])
    process_calls, sleeps = [], []
    process = SimpleNamespace(
        cmdline=lambda: ["python.exe", str(runner.ROOT / "run_solar_cwe_forecast.py").upper(), "--action", "run"],
        is_running=lambda: next(running), status=lambda: psutil.STATUS_RUNNING)

    def get_process(pid):
        process_calls.append(pid)
        return process

    def fake_sleep(seconds):
        queued = runner.read(pointer)
        assert queued["status"] == "waiting_existing_solar_run"
        assert queued["workdirs"] == prior["workdirs"]
        sleeps.append(seconds)

    monkeypatch.setattr(psutil, "Process", get_process)
    monkeypatch.setattr(time, "sleep", fake_sleep)
    with pytest.raises(StopBeforePreparation, match="preflight"):
        runner.main(["--config", str(path), "--action", "run", "--after-solar-pid", "321"])
    queued = runner.read(pointer)
    assert process_calls == [321] and sleeps == [15]
    assert queued["status"] == "waiting_existing_solar_run"
    assert queued["waiting_pid"] == 321
    assert queued["queue_pid"] == runner.os.getpid()
    assert pd.Timestamp(queued["queued_at_utc"]).tzinfo is not None
    assert queued["production_modified"] is False
    assert queued["results"] == prior["results"]
    assert "aucun processus interrompu" in capsys.readouterr().out


@pytest.mark.parametrize("option_present", [False, True])
def test_after_solar_pid_absent_or_already_finished_does_not_wait(queue_sandbox, monkeypatch, option_present):
    import psutil
    import time
    path, pointer, _ = queue_sandbox
    before = hashes(runner.ROOT)
    calls = []

    def absent_process(pid):
        calls.append(pid)
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", absent_process if option_present else blocked)
    monkeypatch.setattr(time, "sleep", blocked)
    args = ["--config", str(path), "--action", "run"]
    if option_present:
        args.extend(["--after-solar-pid", "321"])
    with pytest.raises(StopBeforePreparation):
        runner.main(args)
    assert calls == ([321] if option_present else [])
    assert runner.read(pointer)["status"] == "prepared"
    assert hashes(runner.ROOT) == before


def test_after_solar_pid_refuses_another_process_without_writes(queue_sandbox, monkeypatch):
    import psutil
    import time
    path, _, _ = queue_sandbox
    before = hashes(runner.ROOT)
    process = SimpleNamespace(cmdline=lambda: ["python.exe", str(runner.ROOT / "run_nuclear_forecast.py")],
                              is_running=blocked, status=blocked)
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    monkeypatch.setattr(time, "sleep", blocked)
    with pytest.raises(ValueError, match="Only an existing SolarCWE"):
        runner.main(["--config", str(path), "--action", "run", "--after-solar-pid", "321"])
    assert hashes(runner.ROOT) == before


def test_after_solar_pid_queued_state_survives_wait_and_status_reads(queue_sandbox, monkeypatch, capsys):
    import psutil
    import time
    path, pointer, prior = queue_sandbox
    process = SimpleNamespace(cmdline=lambda: ["python.exe", str(runner.ROOT / "run_solar_cwe_forecast.py")],
                              is_running=lambda: True, status=lambda: psutil.STATUS_RUNNING)

    def suspend_synthetic_wait(seconds):
        raise StopBeforePreparation("Synthetic waiting checkpoint")

    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    monkeypatch.setattr(time, "sleep", suspend_synthetic_wait)
    with pytest.raises(StopBeforePreparation, match="waiting checkpoint"):
        runner.main(["--config", str(path), "--action", "run", "--after-solar-pid", "321"])
    queued = runner.read(pointer)
    assert queued["status"] == "waiting_existing_solar_run"
    assert queued["workdirs"] == prior["workdirs"] and queued["results"] == prior["results"]
    before = hashes(runner.ROOT)
    monkeypatch.setattr(runner, "write_json", blocked)
    monkeypatch.setattr(psutil, "Process", blocked)
    assert runner.main(["--config", str(path), "--action", "status"]) == 0
    assert hashes(runner.ROOT) == before
    assert '"status": "waiting_existing_solar_run"' in capsys.readouterr().out


@pytest.fixture
def registered(sandbox, monkeypatch):
    monkeypatch.setattr(protocol, "_utc_now", lambda: datetime(2026, 9, 20, 19, tzinfo=timezone.utc))
    frozen = protocol.create_or_load_protocol(runner.OUTPUT, recipe_contract={"synthetic": True})
    monkeypatch.setattr(protocol, "_utc_now", lambda: datetime(2027, 4, 1, tzinfo=timezone.utc))
    return frozen


def test_evaluate_without_new_days_never_fetches_or_fits(sandbox, registered, monkeypatch, capsys):
    from chronos2_hourly import nuclear_reporting_refresh, solar_correction_forecast, solar_cwe_sources
    monkeypatch.setattr(nuclear_reporting_refresh, "refresh_nuclear_reporting_sources", blocked)
    monkeypatch.setattr(solar_correction_forecast, "run_solar_correction_forecast", blocked)
    monkeypatch.setattr(solar_cwe_sources, "ensure_solar_sources", blocked)
    assert runner.evaluate(sandbox[0]) == 0
    evaluation = next((runner.OUTPUT / "validation").glob("*/evaluation.json"))
    result = runner.read(evaluation)
    assert result["records"] == []
    assert result["validation"]["prospective_complete_days"] == 0
    assert result["validation"]["promotion_allowed"] is False
    capsys.readouterr()


def test_evaluate_skips_unpublished_failed_receipts(sandbox, registered, monkeypatch, capsys):
    from chronos2_hourly import nuclear_reporting_refresh
    work = runner.OUTPUT / "2026-09-22/fr/failed"
    runner.write_json(work / "run_result.json", {"status": "failed"})
    monkeypatch.setattr(runner, "verify_snapshot", blocked)
    monkeypatch.setattr(runner, "archive_results", blocked)
    monkeypatch.setattr(nuclear_reporting_refresh, "refresh_nuclear_reporting_sources", blocked)
    assert runner.evaluate(sandbox[0]) == 0
    result = runner.read(next((runner.OUTPUT / "validation").glob("*/evaluation.json")))
    assert result["records"] == []
    capsys.readouterr()


@pytest.mark.parametrize("incomplete", [False, True])
@pytest.mark.parametrize("day,hours", [("2026-10-25", 25), ("2027-03-28", 23)])
def test_evaluate_full_dst_day_uses_only_sealed_forecasts(sandbox, registered, monkeypatch, capsys, incomplete, day, hours):
    from chronos2_hourly import nuclear_reporting_refresh, nuclear_report_benchmark, solar_correction_forecast, solar_cwe_sources
    from chronos2_hourly.hourly_contract import local_delivery_day_index
    index = local_delivery_day_index(pd.Timestamp(day).date(), timezone="Europe/Paris")
    assert len(index) == hours
    actual = pd.Series(50., index=index)
    actual.iloc[2] = 350.
    monkeypatch.setattr(nuclear_report_benchmark, "_load_verified_snapshot", lambda *args, **kwargs: (actual + 4., {}))
    calls = []
    for zone in runner.ZONES:
        work = runner.OUTPUT / day / zone.lower() / "sealed"
        runner.write_json(work / "run_result.json", {"status": "complete"})
        runner.write_json(work / "input_snapshot.json", {"identity": {"protocol_sha256": registered["protocol_sha256"]}})
    monkeypatch.setattr(solar_correction_forecast, "run_solar_correction_forecast", blocked)
    monkeypatch.setattr(solar_cwe_sources, "ensure_solar_sources", blocked)
    monkeypatch.setattr(runner, "verify_snapshot", lambda work, *, zone, day:
                        {"zones": {zone: {"timezone": "Europe/Paris"}}})

    def forecast(offset):
        return pd.DataFrame({"delivery_start_utc": index, "residual_kalman__q50": actual.to_numpy() + offset,
                             "chronos2__q50": actual.to_numpy() + 4.,
                             "residual_corrected__q50": actual.to_numpy() + offset})

    def archive(work):
        frames = {variant: SimpleNamespace(kalman_view=SimpleNamespace(forecast=forecast(i + 1)),
                  source_forecast=forecast(i + 1)) for i, variant in enumerate(runner.VARIANTS)}
        return frames, {"sealed_at_utc": (protocol.delivery_cutoff(day) - pd.Timedelta(minutes=1)).isoformat(),
                        "files": {variant: {"kalman_forecast.parquet": "a" * 64} for variant in runner.VARIANTS}}

    monkeypatch.setattr(runner, "archive_results", archive)
    monkeypatch.setattr(runner, "load_incumbent", lambda *args, **kwargs:
                        SimpleNamespace(kalman_view=SimpleNamespace(forecast=forecast(3)), source_forecast=forecast(3)))

    def refresh(config, zone, timezone, delivery_day, output_directory):
        calls.append(zone)
        observed = actual.copy()
        if incomplete and zone == "NL":
            observed.iloc[-1] = np.nan
        return observed, output_directory, {"synthetic": True, "used_for_prediction": False}

    monkeypatch.setattr(nuclear_reporting_refresh, "refresh_nuclear_reporting_sources", refresh)
    assert runner.evaluate(sandbox[0]) == 0
    evaluation = next((runner.OUTPUT / "validation").glob("*/evaluation.json"))
    result = runner.read(evaluation)
    assert set(calls) == set(runner.ZONES)
    assert len(result["records"]) == 8
    assert {row["hours"] for row in result["records"]} == {hours}
    assert result["validation"]["prospective_complete_days"] == (0 if incomplete else 1)
    assert result["activation_performed"] is False
    assert result["validation"]["promotion_allowed"] is False
    for row in result["records"]:
        if incomplete and row["zone"] == "NL":
            assert row["complete"] is False and row["mae_eur_mwh"] is None
        else:
            assert row["complete"] is True and row["mae_eur_mwh"] in (1., 2.)
            assert row["day_mae_win_vs_baseline"] is True
            assert row["spike_slices_ex_post"]["200"]["hours"] == 1
            assert row["spike_slices_ex_post"]["300"]["hours"] == 1
    if not incomplete:
        assert len(result["summary"]["prospective"]) == 8
        for aggregate in result["summary"]["prospective"].values():
            assert aggregate["days"] == 1 and aggregate["hours"] == hours
            assert aggregate["mae_eur_mwh"] == aggregate["rmse_eur_mwh"]
            assert aggregate["day_mae_win_rate_vs_baseline"] == 1.
    capsys.readouterr()
