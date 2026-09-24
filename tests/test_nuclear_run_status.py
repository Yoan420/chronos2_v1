import hashlib
import json

import pandas as pd
import pytest
from requests import exceptions as request_errors

import run_nuclear_forecast as runner
from chronos2_hourly import nuclear_reporting_refresh as refresh


def test_failed_publication_retains_phase_and_saved_result_status(tmp_path):
    with pytest.raises(PermissionError, match="locked report"):
        with runner.run_progress(tmp_path, "BE", pd.Timestamp("2026-09-11"), "run") as progress:
            progress("forecast_saved", model_result_saved=True)
            progress("publish_exports")
            raise PermissionError("locked report")
    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["phase"] == "publish_exports"
    assert status["model_result_saved"] is True
    assert status["error_type"] == "PermissionError"
    assert not list(tmp_path.glob("*.tmp"))


def test_interrupted_status_is_not_marked_complete(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with runner.run_progress(tmp_path, "DE", pd.Timestamp("2026-09-11"), "run"):
            raise KeyboardInterrupt
    assert json.loads((tmp_path / "run_status.json").read_text())["status"] == "interrupted"


def test_reporting_transport_retries_without_model_execution(monkeypatch):
    attempts, sleeps = [], []
    expected = object()
    def request(*args):
        attempts.append(args)
        if len(attempts) < 3:
            raise RuntimeError("Saturn transport unavailable")
        return expected
    monkeypatch.setattr(refresh, "refresh_nuclear_reporting_sources", request)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    assert runner.refresh_reporting_sources("config", "FR") is expected
    assert attempts == [("config", "FR")] * 3
    assert sleeps == [2, 4]


def test_invalid_reporting_values_are_never_retried_or_accepted(monkeypatch):
    def invalid(*args):
        raise refresh.NuclearReportingRefreshError("Historical observations missing")
    monkeypatch.setattr(refresh, "refresh_nuclear_reporting_sources", invalid)
    monkeypatch.setattr(runner.time, "sleep", lambda _: pytest.fail("Invalid data is not a transport failure"))
    with pytest.raises(ValueError, match="Historical observations missing"):
        runner.refresh_reporting_sources("config", "FR")


@pytest.mark.parametrize("original", [RuntimeError("model failed"), KeyboardInterrupt()])
def test_status_write_failure_preserves_original_exception(tmp_path, monkeypatch, capsys, original):
    write = runner.write_json

    def unavailable_status(path, payload):
        if path.name == "run_status.json" and payload["status"] in {"failed", "interrupted"}:
            raise PermissionError("status locked")
        return write(path, payload)

    monkeypatch.setattr(runner, "write_json", unavailable_status)
    with pytest.raises(type(original)) as captured:
        with runner.run_progress(tmp_path, "BE", pd.Timestamp("2026-09-11"), "run"):
            raise original
    assert captured.value is original
    warning = capsys.readouterr().err
    assert "AVERTISSEMENT" in warning and "status locked" in warning
    assert "absent ou ancien" in warning


@pytest.mark.parametrize("all_updates_fail", [False, True])
def test_auxiliary_status_failure_does_not_invalidate_verified_completion(
    tmp_path, monkeypatch, capsys, all_updates_fail,
):
    import run_nuclear_kalman as launcher

    zone, day = "BE", "2026-09-11"
    export_root = tmp_path / "runs/exports" / day / "be"
    group = export_root / "nuclear_kalman"
    group.mkdir(parents=True)
    html = group / f"forecast_be_{day}_nuclear_kalman.html"
    csv = html.with_suffix(".csv")
    html.write_text("<html>Published Kalman report</html>")
    csv.write_text("q50\n50\n")
    manifest = export_root / "current_nuclear_batch_manifest.json"
    runner.write_json(manifest, {"zone": zone, "delivery_day": day, "exports": [{
        "variant": "nuclear_kalman", "files": [
            {"path": str(path.relative_to(export_root)),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (html, csv)
        ],
    }]})
    receipt = {"zone": zone, "status": "complete", "reports": {"kalman": str(html)},
               "exports": {"kalman": str(html), "manifest": str(manifest)}}
    work = tmp_path / "work"
    write = runner.write_json

    def unavailable_status(path, payload):
        if path.name == "run_status.json" and (all_updates_fail or payload["status"] == "complete"):
            raise PermissionError("status locked")
        return write(path, payload)

    monkeypatch.setattr(runner, "write_json", unavailable_status)
    with runner.run_progress(work, zone, pd.Timestamp(day), "run") as progress:
        progress("publish_exports", model_result_saved=True)
        runner.write_json(work / "run_result.json", receipt)
        print(json.dumps(receipt), flush=True)
    captured = capsys.readouterr()
    assert "AVERTISSEMENT" in captured.err
    received = json.loads(captured.out)
    assert json.loads((work / "run_result.json").read_text()) == received
    outcome = launcher.StepOutcome(0, completion=received)
    assert launcher._verified_zone_report(outcome, root=tmp_path, zone=zone, day=day) == html
    # The launcher still requires the receipt and rejects an altered export.
    with pytest.raises(ValueError, match="confirme"):
        launcher._verified_zone_report(launcher.StepOutcome(0), root=tmp_path, zone=zone, day=day)
    csv.write_text("changed\n")
    with pytest.raises(ValueError, match="modifie"):
        launcher._verified_zone_report(outcome, root=tmp_path, zone=zone, day=day)


def test_required_result_write_failure_still_fails_without_completion(tmp_path, monkeypatch, capsys):
    original = PermissionError("run_result locked")

    def unavailable(path, payload):
        if path.name == "run_result.json":
            raise original
        raise OSError("status destination unavailable")

    monkeypatch.setattr(runner, "write_json", unavailable)
    with pytest.raises(PermissionError) as captured:
        with runner.run_progress(tmp_path, "FR", pd.Timestamp("2026-09-11"), "run"):
            runner.write_json(tmp_path / "run_result.json", {"exports": {}})
            print(json.dumps({"zone": "FR", "status": "complete"}))
    assert captured.value is original
    output = capsys.readouterr()
    assert output.out == ""
    assert "AVERTISSEMENT" in output.err
    assert not (tmp_path / "run_result.json").exists()


@pytest.mark.parametrize("error_type", [
    request_errors.ConnectionError, request_errors.Timeout,
    request_errors.ConnectTimeout, request_errors.ReadTimeout,
])
def test_real_requests_transport_errors_are_retried(monkeypatch, error_type):
    attempts, sleeps = [], []
    expected = object()

    def request(*args):
        attempts.append(args)
        if len(attempts) < 3:
            raise error_type("temporary Storm transport failure")
        return expected

    monkeypatch.setattr(refresh, "refresh_nuclear_reporting_sources", request)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    assert runner.refresh_reporting_sources("config", "FR") is expected
    assert len(attempts) == 3
    assert sleeps == [2, 4]


def test_requests_transport_failure_after_retries_propagates_original(monkeypatch):
    original = request_errors.ReadTimeout("Storm timed out")
    attempts, sleeps = [], []

    def request(*args):
        attempts.append(args)
        raise original

    monkeypatch.setattr(refresh, "refresh_nuclear_reporting_sources", request)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with pytest.raises(request_errors.ReadTimeout) as captured:
        runner.refresh_reporting_sources("config", "BE")
    assert captured.value is original
    assert len(attempts) == 3 and sleeps == [2, 4]


@pytest.mark.parametrize("error_type", [request_errors.HTTPError, request_errors.RequestException])
def test_http_or_generic_request_error_is_not_blindly_retried(monkeypatch, error_type):
    original = error_type("invalid or rejected request")

    def request(*args):
        raise original

    monkeypatch.setattr(refresh, "refresh_nuclear_reporting_sources", request)
    monkeypatch.setattr(runner.time, "sleep", lambda _: pytest.fail("not a selected transport error"))
    with pytest.raises(error_type) as captured:
        runner.refresh_reporting_sources("config", "NL")
    assert captured.value is original
