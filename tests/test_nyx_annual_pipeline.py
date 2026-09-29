import json
from pathlib import Path
import pytest
from chronos2_hourly import nyx_annual_pipeline as m


@pytest.mark.parametrize("missing", ["jao", "hydro", "exchange"])
def test_missing_archive_checks_all_public_archives_but_never_starts_saturn(monkeypatch, tmp_path, missing):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    calls = []
    def runner(script, args):
        calls.append(script)
        return 2 if missing in script else 0
    output = tmp_path / "output"
    result = m.run("2026-09-28", action="prepare", bundle=tmp_path / "bundle", output=output, runner=runner)
    assert calls == ["run_nyx_annual_daily_capture.py", "run_nyx_annual_jao_source.py", "run_nyx_annual_hydro_source.py",
                     "run_nyx_annual_exchange_source.py"]
    assert result["state"] == "BLOCKED" and result["forecast_published"] is False
    assert sum(stage["state"] == "ERROR" for stage in result["stages"]) == 1
    assert not output.exists()
    saved = json.loads((tmp_path / "output.pipeline.json").read_text())
    assert len(saved["stages"]) == 4


def test_production_refuses_before_any_network_without_qualification(monkeypatch, tmp_path):
    from chronos2_hourly import nyx_annual_cpu_live
    monkeypatch.setattr(m, "ROOT", tmp_path)
    def unavailable():
        raise ValueError("full input chain not qualified")
    monkeypatch.setattr(nyx_annual_cpu_live, "verify_activation", unavailable)
    result = m.run("2026-09-28", action="forecast", output=tmp_path / "out",
                   runner=lambda *args: pytest.fail("Unqualified forecast contacted sources"))
    assert result["state"] == "BLOCKED"
    assert result["stages"][0]["name"] == "qualification"


def test_prepare_calls_every_producer_in_order_and_does_not_publish_forecast(monkeypatch, tmp_path):
    from chronos2_hourly import nyx_annual_cpu_baseline as baseline
    from chronos2_hourly import nyx_annual_cpu_bundle_builder as builder
    from chronos2_hourly import nyx_annual_cpu_reference_builder as reference
    monkeypatch.setattr(m, "ROOT", tmp_path)
    produced = []
    monkeypatch.setattr(baseline, "build_from_bundle", lambda *a, **k: produced.append("baseline"))
    monkeypatch.setattr(builder, "materialize_features", lambda *a, **k: produced.append("features"))
    monkeypatch.setattr(reference, "build_from_bundle", lambda *a, **k: produced.append("reference"))
    monkeypatch.setattr(builder, "seal_bundle", lambda *a, **k: produced.append("seal"))
    calls = []
    result = m.run("2026-09-28", action="prepare", bundle=tmp_path / "bundle", output=tmp_path / "out",
                   runner=lambda script, args: calls.append(script) or 0)
    assert calls == ["run_nyx_annual_daily_capture.py", "run_nyx_annual_jao_source.py", "run_nyx_annual_hydro_source.py",
                     "run_nyx_annual_exchange_source.py", "run_nyx_annual_saturn_source.py",
                     "run_nyx_annual_auction_prices_source.py", "run_nyx_annual_fuel_source.py",
                     "run_nyx_annual_thermal_source.py"]
    assert produced == ["baseline", "features", "reference", "seal"]
    assert result["state"] == "PREPARED" and result["forecast_published"] is False


def test_capture_uses_real_collector_and_tomorrow_safety_check(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    calls = []
    result = m.run("2026-09-30", action="capture", output=tmp_path / "out",
                   runner=lambda script, args: calls.append((script, args)) or 2)
    assert calls == [("run_nyx_annual_daily_capture.py", ["--delivery-day", "2026-09-30"])]
    assert result["state"] == "BLOCKED" and not result["forecast_published"]


def test_evaluation_source_caches_are_separate_from_daily_caches(tmp_path):
    commands = dict((group, args) for group, _, args in m.source_commands(
        "2026-09-28", tmp_path / "bundle", tmp_path / "eval_cache"))
    assert commands["saturn"][-2:] == ["--cache", str(tmp_path / "eval_cache/saturn")]
    assert commands["thermal_capacity"][-2:] == ["--cache", str(tmp_path / "eval_cache/thermal")]
    assert commands["fuel"][-2:] == ["--cache-dir", str(tmp_path / "eval_cache/fuel")]


@pytest.mark.parametrize("valid", [True, False])
def test_sealed_day_reuse_requires_revalidation_and_never_rewinds_newer_caches(monkeypatch, tmp_path, valid):
    from chronos2_hourly.nyx_annual_live_preflight import MATERIALIZATION_PATH
    bundle = tmp_path / "bundle"
    manifest = bundle / MATERIALIZATION_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    calls = []
    def validate(path, day, *, require_asof=True):
        assert require_asof is False
        calls.append((path, day))
        if not valid:
            raise ValueError("Source artifact changed")
    monkeypatch.setattr(m, "validate_prepared_bundle", validate)
    result = m.run("2026-09-28", action="prepare", bundle=bundle, output=tmp_path / "out",
                   runner=lambda *args: pytest.fail("Sealed day touched mutable caches"))
    assert calls == [(bundle, "2026-09-28")]
    assert result["state"] == ("PREPARED" if valid else "BLOCKED")
    assert not result["forecast_published"]


def test_missing_delivery_capture_stops_before_history_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    calls = []
    result = m.run("2026-09-30", action="bootstrap", bundle=tmp_path / "bundle", output=tmp_path / "out",
        runner=lambda script, args: calls.append((script, args)) or 1)
    assert calls == [("run_nyx_annual_daily_capture.py", ["--verify-only", "--delivery-day", "2026-09-30"])]
    assert result["state"] == "BLOCKED"
    assert "08 h Paris" in result["remediation"]


def test_bootstrap_downloads_history_but_never_starts_saturn_or_models(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    calls = []
    bundle = tmp_path / "bundle"
    receipt_dir = bundle / "source_receipts"
    receipt_dir.mkdir(parents=True)
    for group in m.ARCHIVE_GROUPS:
        (receipt_dir / f"{group}.json").write_text(json.dumps({"asof_cutoff_verified": False}))
    result = m.run("2026-09-30", action="bootstrap", bundle=bundle, output=tmp_path / "out",
        runner=lambda script, args: calls.append((script, args)) or 0)
    assert [script for script, _ in calls] == ["run_nyx_annual_daily_capture.py", "run_nyx_annual_jao_source.py",
        "run_nyx_annual_hydro_source.py", "run_nyx_annual_exchange_source.py"]
    assert "--bootstrap-history" in calls[1][1]
    assert calls[2][1][-2:] == ["--action", "bootstrap"]
    assert result["state"] == "BOOTSTRAPPED" and result["qualification_required"] is True
    assert result["public_history_before_forecast_cutoff"] is False
    assert result["forecast_published"] is False


def test_morning_capture_refreshes_training_history_before_prepare(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    calls = []
    result = m.run("2026-09-30", action="capture", bundle=tmp_path / "bundle", output=tmp_path / "out",
        runner=lambda script, args: calls.append(script) or 0)
    assert calls == ["run_nyx_annual_daily_capture.py", "run_nyx_annual_jao_source.py",
                     "run_nyx_annual_hydro_source.py", "run_nyx_annual_exchange_source.py"]
    assert result["state"] == "CAPTURED" and not result["forecast_published"]


def test_new_late_training_snapshot_never_reaches_production_cpu(monkeypatch, tmp_path):
    from chronos2_hourly import nyx_annual_cpu_live as live
    from test_nyx_annual_history_policy import receipt
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(live, "verify_activation", lambda: {})
    bundle = tmp_path / "bundle"
    folder = bundle / "source_receipts"
    folder.mkdir(parents=True)
    for group in m.ARCHIVE_GROUPS:
        record, _ = receipt(bundle, group=group)
        (folder / f"{group}.json").write_text(json.dumps(record))
    calls = []
    result = m.run("2026-09-30", action="forecast", bundle=bundle, output=tmp_path / "out",
        runner=lambda script, args: calls.append(script) or 0)
    assert result["state"] == "BLOCKED" and not result["forecast_published"]
    assert result["stages"][-1]["name"] == "training_history_cutoff"
    assert "history downloaded after" in result["stages"][-1]["error"]
    assert all("saturn" not in script for script in calls)
