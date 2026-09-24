"""Canonical primary-result browsing with isolated, explicitly synthetic files.

No database, scientific pipeline, live run, or user report is modified.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import httpx
import pytest

from experiment_console.primary_results import build_primary_results, read_primary_artifact
from experiment_console.server import ConsoleHTTPServer


DAY = "2026-09-11"


def export(project, day=DAY, zone="fr", *, html=True, csv=True):
    zone_root = project / "runs" / "exports" / day / zone
    group = zone_root / "nuclear_kalman"
    group.mkdir(parents=True, exist_ok=True)
    stem = f"forecast_{zone}_{day}_nuclear_kalman"
    report = group / f"{stem}.html"
    values = group / f"{stem}.csv"
    if html:
        report.write_text("<!doctype html><html><body><p>Explicit primary report fixture</p></body></html>", encoding="utf-8")
    if csv:
        values.write_text(f"delivery_day,q50\n{day},42\n", encoding="utf-8")
    return SimpleNamespace(project=project, day=day, zone=zone.upper(), zone_root=zone_root,
                           group=group, report=report, csv=values,
                           manifest=zone_root / "current_nuclear_batch_manifest.json")


def write_manifest(item, *, audit=False):
    files = [path for path in (item.report, item.csv) if path.exists()]
    if audit:
        audit_path = item.group / "nuclear_report_audit.json"
        audit_path.write_text('{"explicit_test_fixture": true}', encoding="utf-8")
        files.append(audit_path)
    payload = {"schema_version": 1, "delivery_day": item.day, "zone": item.zone,
               "mode": "nuclear_kalman", "exports": [{"variant": "nuclear_kalman", "zone": item.zone,
               "source_model": "residual_kalman", "files": [
                   {"path": path.relative_to(item.zone_root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]}]}
    item.manifest.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def relative(item, path):
    return path.relative_to(item.project / "runs").as_posix()


def zone_row(result, day=DAY, zone="FR"):
    row = next(row for row in result["days"] if row["date"] == day)
    return next(item for item in row["zones"] if item["zone"] == zone)


def snapshot(project):
    return {path.relative_to(project).as_posix(): path.read_bytes()
            for path in project.rglob("*") if path.is_file()}


def test_missing_runs_root_has_an_understandable_empty_result_and_creates_nothing(tmp_path):
    result = build_primary_results(tmp_path)
    assert result["root"] == "runs" and result["model_name"] == "NYX"
    assert result["days"] == []
    assert list(tmp_path.iterdir()) == []


def test_only_canonical_reports_are_grouped_by_day_with_latest_first_and_csv_only_visible(tmp_path):
    fr = export(tmp_path)
    export(tmp_path, zone="be")
    nl = export(tmp_path, day="2026-09-12", zone="nl", html=False)
    cwe = tmp_path / "runs" / "reports" / "model_storm" / "CWE_Model_Storm_2026-09-13.html"
    cwe.parent.mkdir(parents=True)
    cwe.write_text("<html>Explicit CWE fixture</html>", encoding="utf-8")
    for name in [
        "reports/model_storm/experimental/CWE_Model_Storm_2099-01-01.html",
        "reports/model_storm/FR_Model_Storm_2099-01-02.html",
        "exports/2099-01-03/fr/autonomous/forecast_fr_2099-01-03_autonomous.html",
        "exports/2099-01-04/fr/nuclear_kalman/experimental.html",
        "exports/not-a-date/fr/nuclear_kalman/forecast_fr_not-a-date_nuclear_kalman.html",
    ]:
        path = tmp_path / "runs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Incidental test fixture must not be surfaced", encoding="utf-8")
    before = snapshot(tmp_path)
    result = build_primary_results(tmp_path)
    assert [row["date"] for row in result["days"]] == ["2026-09-13", "2026-09-12", DAY]
    assert result["days"][0]["cwe"]["path"] == cwe.relative_to(tmp_path / "runs").as_posix()
    assert result["days"][0]["zones"] == [], "a CWE-only day does not imply zone exports exist"
    france = zone_row(result)
    assert france["label"] == "France"
    assert france["report"]["path"] == relative(fr, fr.report)
    assert france["publication_status"] == "unverified"
    netherlands = zone_row(result, "2026-09-12", "NL")
    assert netherlands["report"] is None
    assert netherlands["csv"]["path"] == relative(nl, nl.csv)
    assert netherlands["publication_status"] == "incomplete"
    assert netherlands["warnings"]
    assert datetime.fromisoformat(france["report"]["updated_at"].replace("Z", "+00:00")).tzinfo
    assert france["report"]["size"] == fr.report.stat().st_size
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("audit", [False, True])
def test_verified_publication_requires_matching_identity_and_content_hashes(tmp_path, audit):
    item = export(tmp_path)
    write_manifest(item, audit=audit)
    before = snapshot(tmp_path)
    first = build_primary_results(tmp_path)
    second = build_primary_results(tmp_path)
    assert zone_row(first)["publication_status"] == "verified"
    assert second == first
    assert snapshot(tmp_path) == before
    assert not list(tmp_path.rglob("*.sqlite*"))


@pytest.mark.parametrize("forgery", [
    "wrong_day", "wrong_zone", "lowercase_zone", "wrong_record_zone", "duplicate_variant",
    "wrong_variant", "missing_html_record", "missing_csv_record", "wrong_hash", "wrong_audit_hash",
    "forbidden_declared_file", "malformed_json",
])
def test_forged_or_incomplete_manifest_never_confers_verified_status(tmp_path, forgery):
    item = export(tmp_path)
    manifest = write_manifest(item, audit=forgery == "wrong_audit_hash")
    record = manifest["exports"][0]
    if forgery == "wrong_day":
        manifest["delivery_day"] = "2026-09-10"
    elif forgery == "wrong_zone":
        manifest["zone"] = "DE"
    elif forgery == "lowercase_zone":
        manifest["zone"] = "fr"
    elif forgery == "wrong_record_zone":
        record["zone"] = "DE"
    elif forgery == "duplicate_variant":
        manifest["exports"].append(deepcopy(record))
    elif forgery == "wrong_variant":
        record["variant"] = "nuclear_autonomous"
    elif forgery == "missing_html_record":
        record["files"] = [file for file in record["files"] if not file["path"].endswith(".html")]
    elif forgery == "missing_csv_record":
        record["files"] = [file for file in record["files"] if not file["path"].endswith(".csv")]
    elif forgery == "wrong_hash":
        record["files"][0]["sha256"] = "0" * 64
    elif forgery == "wrong_audit_hash":
        next(file for file in record["files"] if file["path"].endswith(".json"))["sha256"] = "0" * 64
    elif forgery == "forbidden_declared_file":
        record["files"].append({"path": "../../private.txt", "sha256": "0" * 64})
    item.manifest.write_text("{broken manifest" if forgery == "malformed_json" else json.dumps(manifest), encoding="utf-8")
    result = build_primary_results(tmp_path)
    row = zone_row(result)
    assert row["publication_status"] == "incomplete"
    assert row["warnings"]
    # A failed publication check must not hide existing files from human review.
    assert row["report"] and row["csv"]
    path, body = read_primary_artifact(tmp_path, relative(item, item.report))
    assert path == item.report and body == item.report.read_bytes()


def test_disappearance_after_listing_is_rechecked_before_read_and_flagged_in_next_listing(tmp_path):
    item = export(tmp_path)
    write_manifest(item)
    assert zone_row(build_primary_results(tmp_path))["publication_status"] == "verified"
    item.report.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        read_primary_artifact(tmp_path, relative(item, item.report))
    row = zone_row(build_primary_results(tmp_path))
    assert row["report"] is None and row["csv"]
    assert row["publication_status"] == "incomplete"


@pytest.mark.parametrize("path", [
    "../config/experiment_console.json", "/etc/passwd", "C:/private.txt",
    "reports\\model_storm\\CWE_Model_Storm_2026-09-11.html",
    "reports/model_storm/../model_storm/CWE_Model_Storm_2026-09-11.html",
    "reports/model_storm/CWE_Model_Storm_2026-02-30.html",
    "exports/2026-09-11/fr/nuclear_autonomous/forecast_fr_2026-09-11_nuclear_autonomous.html",
    "exports/2026-09-11/fr/nuclear_kalman/nuclear_report_audit.json",
    "exports/2026-09-11/fr/nuclear_kalman/private.html",
])
def test_direct_reads_reject_noncanonical_or_traversal_paths(tmp_path, path):
    export(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        read_primary_artifact(tmp_path, path)


def test_artifact_read_obeys_the_size_limit(tmp_path):
    item = export(tmp_path)
    with pytest.raises(ValueError):
        read_primary_artifact(tmp_path, relative(item, item.report), max_bytes=16)


def test_directory_replaced_with_junction_or_symlink_after_listing_is_rejected(tmp_path):
    item = export(tmp_path, csv=False)
    assert zone_row(build_primary_results(tmp_path))["report"]
    outside = tmp_path / "outside canonical runs"
    outside.mkdir()
    (outside / item.report.name).write_text("Must never be served through a filesystem link", encoding="utf-8")
    item.report.unlink()
    item.group.rmdir()
    if os.name == "nt":
        created = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(item.group), str(outside)],
                                 capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        assert created.returncode == 0, created.stderr
    else:
        item.group.symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises((ValueError, FileNotFoundError)):
            read_primary_artifact(tmp_path, relative(item, item.report))
    finally:
        # Remove only the test link itself, never recursively traverse its target.
        item.group.rmdir() if os.name == "nt" else item.group.unlink()
    assert (outside / item.report.name).read_text() == "Must never be served through a filesystem link"


@pytest.fixture
def primary_api(tmp_path):
    # This manager deliberately has no store/list_runs/registry. These endpoints
    # must remain file readers independent of the execution database.
    manager = SimpleNamespace(project_root=tmp_path)
    server = ConsoleHTTPServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    with httpx.Client(base_url=base_url, trust_env=False, timeout=10) as client:
        yield SimpleNamespace(project=tmp_path, server=server, client=client)
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)
    assert not thread.is_alive()


def test_http_primary_results_and_artifacts_redact_secrets_without_a_database_or_source_writes(primary_api):
    item = export(primary_api.project)
    item.report.write_text("<html><body><p>Public report</p><p>api_key=SYNTHETIC_PRIMARY_HTML_SECRET</p></body></html>", encoding="utf-8")
    item.csv.write_text("delivery_day,q50,api_key,password,access_token\n2026-09-11,42,SYNTHETIC_PRIMARY_API_SECRET,SYNTHETIC_PRIMARY_PASSWORD_SECRET,SYNTHETIC_PRIMARY_TOKEN_SECRET\n", encoding="utf-8")
    write_manifest(item)
    before = snapshot(primary_api.project)
    listing = primary_api.client.get("/api/primary-results")
    assert listing.status_code == 200, listing.text
    assert zone_row(listing.json())["publication_status"] == "verified"
    for path, suffix in [(item.report, "html"), (item.csv, "csv")]:
        response = primary_api.client.get("/api/primary-artifact", params={"path": relative(item, path), "download": "1"})
        assert response.status_code == 200, response.text
        assert "SYNTHETIC_PRIMARY_" not in response.text
        assert "attachment" in response.headers["content-disposition"]
        if suffix == "html":
            assert "Public report" in response.text
            assert "sandbox" in response.headers["content-security-policy"]
            assert "default-src 'none'" in response.headers["content-security-policy"]
        else:
            assert "2026-09-11,42" in response.text
    assert snapshot(primary_api.project) == before
    assert not list(primary_api.project.rglob("*.sqlite*"))


@pytest.mark.parametrize("headers", [{"Host": "evil.example"}, {"Origin": "https://evil.example"}])
def test_primary_http_endpoints_preserve_local_origin_boundary(primary_api, headers):
    item = export(primary_api.project)
    assert primary_api.client.get("/api/primary-results", headers=headers).status_code == 403
    assert primary_api.client.get("/api/primary-artifact", params={"path": relative(item, item.report)}, headers=headers).status_code == 403


def test_http_artifact_endpoint_cannot_be_used_as_an_arbitrary_file_reader(primary_api):
    private = primary_api.project / "private.txt"
    private.write_text("SYNTHETIC_PRIVATE_OUTSIDE_RUNS", encoding="utf-8")
    response = primary_api.client.get("/api/primary-artifact", params={"path": "../private.txt"})
    assert response.status_code in {400, 403, 404}
    assert "SYNTHETIC_PRIVATE_OUTSIDE_RUNS" not in response.text
