"""Read-only integration checks against the available frozen baseline schema.

All copies and candidate snapshots are created in pytest's temporary directory.
These tests never fetch data, launch a forecast or modify the real baseline.
"""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import run_nuclear_cwe_forecast as runner
from chronos2_hourly.nuclear_cwe_sources import CWE_CAPACITY_SOURCES, audit_cwe_source
from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
from chronos2_hourly.nuclear_reporting_refresh import verify_refreshed_observations
from chronos2_hourly.nuclear_run_archive import _source_contract


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "runs/experiments/nuclear_forecast_v1/2026-09-11/fr/civil_pit_v2"
DAY = "2026-09-11"


@pytest.fixture
def existing_baseline():
    if not (BASELINE / "run_result.json").is_file():
        pytest.skip("The optional real frozen FR baseline fixture is unavailable.")
    receipt = json.loads((BASELINE / "run_result.json").read_text(encoding="utf-8"))
    prior = json.loads((BASELINE / "input_snapshot.json").read_text(encoding="utf-8"))
    controls = [BASELINE / name for name in ("run_result.json", "input_snapshot.json", "resolved_config.yaml")]
    files = controls + [Path(item["snapshot"]) for item in prior["files"]]
    files += [p for p in Path(receipt["reporting_sources"]["snapshot_directory"]).rglob("*") if p.is_file()]
    before = {p: runner.sha256(p) for p in files}
    yield receipt, prior
    assert {p: runner.sha256(p) for p in files} == before, "A read-only comparison modified the incumbent."


def test_real_reporting_snapshot_relocation_preserves_observed_and_storm_contracts(existing_baseline, tmp_path):
    receipt, _ = existing_baseline
    audit, copies = runner._freeze_reporting(BASELINE, tmp_path / "candidate", receipt)
    observed = pd.read_parquet(audit["observed"]["artifact_path"])
    values = pd.Series(observed.actual.to_numpy(), index=pd.DatetimeIndex(observed.timestamp), name="actual")
    checked = verify_refreshed_observations(values, audit, zone="FR", timezone="Europe/Paris", delivery_day=DAY)
    assert checked["artifact_sha256"] == receipt["reporting_sources"]["observed"]["artifact_sha256"]
    assert checked["used_for_prediction"] is False
    assert audit["extracted_at_utc"] == receipt["reporting_sources"]["extracted_at_utc"]
    original_storm = _load_verified_snapshot(Path(receipt["reporting_sources"]["snapshot_directory"]), zone="FR", timezone="Europe/Paris")
    relocated_storm = _load_verified_snapshot(Path(audit["snapshot_directory"]), zone="FR", timezone="Europe/Paris")
    pd.testing.assert_series_equal(original_storm[0], relocated_storm[0])
    assert original_storm[2]["artifact_sha256"] == relocated_storm[2]["artifact_sha256"]
    assert all(Path(record["snapshot_path"]).is_relative_to(tmp_path / "candidate") for record in copies)


@pytest.mark.parametrize("artifact", ["observed_latest.parquet", "storm_dashboard_official_statistics.parquet"])
def test_reporting_preflight_rejects_corrupted_frozen_prices(existing_baseline, tmp_path, artifact):
    receipt, _ = existing_baseline
    workspace = tmp_path / "candidate"
    runner._freeze_reporting(BASELINE, workspace, receipt)
    runner._verify_reporting(workspace, zone="FR", day=DAY, timezone="Europe/Paris")
    path = workspace / "reference/reporting/inputs" / artifact
    frame = pd.read_parquet(path)
    price_column = frame.select_dtypes(include="number").columns[0]
    frame.loc[frame.index[0], price_column] += 1.0
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        runner._verify_reporting(workspace, zone="FR", day=DAY, timezone="Europe/Paris")


def test_real_clone_retains_training_labels_and_hyperparameters(existing_baseline, tmp_path):
    _, prior = existing_baseline
    sources = {}
    for alias, specification in CWE_CAPACITY_SOURCES.items():
        source = ROOT / f"data/pit/nuclear_cwe/{alias}/bundles/2024-09-09_2026-09-11/{alias}.parquet"
        if not source.is_file():
            pytest.skip("The optional completed CWE source bundles are unavailable.")
        sources[alias] = {"path": str(source), "specification": specification,
                          "audit": audit_cwe_source(source, specification, "2024-09-09", DAY)}
    settings = {"output_root": tmp_path / "candidate", "baseline_root": BASELINE.parents[2],
                "baseline_protocol": "civil_pit_v2", "config_sha256": "isolated-test-recipe"}
    original = yaml.safe_load((BASELINE / "resolved_config.yaml").read_text(encoding="utf-8"))
    config = runner.prepare_snapshot(settings, DAY, "FR", sources)
    workspace = runner.workdir_for(settings, DAY, "FR")
    assert config["model"] == original["model"]
    assert config["backtest"] == original["backtest"]
    assert config["nuclear_experiment"]["filter_parameters"] == original["nuclear_experiment"]["filter_parameters"]
    assert config["nuclear_experiment"]["history_anchor_day"] == original["nuclear_experiment"]["history_anchor_day"]
    assert config["data"]["runtime_as_of"] == original["data"]["runtime_as_of"]
    assert Path(config["nuclear_experiment"]["incremental_namespace"]).is_relative_to(settings["output_root"])
    # The frozen-result archive reader must accept the new input manifest format.
    source_contract = _source_contract(workspace)
    assert source_contract["snapshot_identity"]["zone"] == "FR"
    original_sources = {"target": original["zones"]["FR"]["target"]["file"], **original["data"]["pit_files"]}
    candidate_sources = {"target": config["zones"]["FR"]["target"]["file"], **config["data"]["pit_files"]}
    for alias, path in original_sources.items():
        assert runner.sha256(Path(candidate_sources[alias])) == runner.sha256(Path(path))
        assert Path(candidate_sources[alias]).is_relative_to(workspace / "snapshot")
    target = pd.read_parquet(config["zones"]["FR"]["target"]["file"])
    target_times = pd.DatetimeIndex(pd.to_datetime(target.timestamp, utc=True))
    assert target_times.max().tz_convert("Europe/Paris").date() < pd.Timestamp(DAY).date()
    reporting = json.loads((workspace / "reference/reporting/statistics_history_audit.json").read_text(encoding="utf-8"))
    assert Path(reporting["observed"]["artifact_path"]).is_relative_to(workspace / "reference")
    assert reporting["observed"]["artifact_path"] != config["zones"]["FR"]["target"]["file"]
    assert set(candidate_sources)-set(original_sources) == set(CWE_CAPACITY_SOURCES)
    for alias in CWE_CAPACITY_SOURCES:
        spec = config["zones"]["FR"]["covariates"][alias]
        assert spec["information_type"] == "capacity_forecast"
        assert spec["daily_broadcast"] is True and spec["fill_method"] == "none"
    # Re-entry must verify/reuse, not reinterpret dates or refresh labels.
    again = runner.prepare_snapshot(settings, DAY, "FR", {})
    assert again == config


def test_original_sources_remain_equal_to_cwe_frozen_seed_copies():
    found = 0
    for alias, specification in CWE_CAPACITY_SOURCES.items():
        seed = ROOT / f"data/pit/nuclear_cwe/{alias}/seed"
        provenance = seed / "seed_origin.json"
        if not provenance.is_file():
            continue
        found += 1
        evidence = json.loads(provenance.read_text(encoding="utf-8"))
        original = ROOT / specification["seed_relative_path"]
        assert runner.sha256(original) == evidence["source_sha256"]
        assert runner.sha256(original.with_name(original.name + ".audit.json")) == evidence["source_audit_sha256"]
        assert runner.sha256(seed / original.name) == evidence["source_sha256"]
    if not found:
        pytest.skip("The optional real frozen CWE seeds are unavailable.")
