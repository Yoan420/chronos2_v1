"""Orchestration guards; no actual neural inference or production writes."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous import rolling_research as research
from chronos2_exogenous import prospective_trial as trial
from chronos2_exogenous import lora_finetune
from chronos2_exogenous import rolling_research_prefix
from test_prospective_trial import _raw


def test_window_is_exact_365_plus365_calendar_days():
    assert research._bounds("2026-09-08") == ("2024-09-09", "2025-09-09", "2026-09-08")


@pytest.mark.parametrize("day", ["2026-09-08T01:00", "2026-09-08T00:00Z", "NaT"])
def test_non_civil_day_is_rejected(day):
    with pytest.raises(trial.ProspectiveTrialError):
        research._bounds(day)


@pytest.fixture
def prefix_case(tmp_path, monkeypatch):
    panel = tmp_path / "panel.parquet"
    rows = []
    for day in ("2025-09-01", "2025-09-02"):
        raw = _raw(day, 1)
        rows.append(pd.DataFrame({"timestamp": raw.delivery_start_utc,
            "delivery_day": day, "item_id": "FR", "phase": "horizon", "target": 777.0}))
    pd.concat(rows).to_parquet(panel)
    config = {"project_root": str(tmp_path), "candidate_config": str(tmp_path / "candidate.yaml"),
              "checkpoint_sha256": "a" * 64}
    manifest = {"panel": trial._file(panel), "schema": {"known_future_covariates": []},
                "zones": {"FR": {"bundle": str(tmp_path / "bundle")}}, "code_files": []}
    monkeypatch.setattr(lora_finetune, "load_config", lambda *_: SimpleNamespace())
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda *_: None))
    monkeypatch.setattr(lora_finetune, "load_checkpoint", lambda *a, **kw: calls.append("load") or object())
    def predict(group, *_):
        day = group.delivery_day.iloc[0]
        calls.append(day)
        raw = _raw(day, 1)
        raw["actual"] = np.nan
        return raw, {"horizon_target_used": False}
    monkeypatch.setattr(rolling_research_prefix, "predict_prefix_group", predict)
    return config, manifest, calls


def test_prefix_resume_reuses_sealed_days_and_attaches_labels_after_inference(prefix_case):
    config, manifest, calls = prefix_case
    first, audit = research.reconstruct_prefix(config, manifest, zone="FR", start_day="2025-09-01", max_new_days=1)
    assert first.actual.eq(777).all()
    assert audit["complete"] is False
    assert calls == ["load", "2025-09-01"]
    second, audit = research.reconstruct_prefix(config, manifest, zone="FR", start_day="2025-09-01")
    assert audit["complete"] is True
    assert calls == ["load", "2025-09-01", "load", "2025-09-02"]
    again, _ = research.reconstruct_prefix(config, manifest, zone="FR", start_day="2025-09-01")
    pd.testing.assert_frame_equal(second, again)
    assert len(calls) == 4
    assert audit["neural_oof"] is False and audit["neural_in_sample"] is True


def test_tampered_cached_prefix_is_rejected_before_loading_model(prefix_case):
    config, manifest, calls = prefix_case
    research.reconstruct_prefix(config, manifest, zone="FR", start_day="2025-09-01", max_new_days=1)
    record_path = next((Path(config["project_root"]) / "runs").rglob("manifest.json"))
    raw_path = Path(json.loads(record_path.read_text())["raw"]["path"])
    raw_path.write_bytes(b"tampered")
    with pytest.raises(trial.ProspectiveTrialError):
        research.reconstruct_prefix(config, manifest, zone="FR", start_day="2025-09-01")
    assert calls == ["load", "2025-09-01"]


def test_changed_checkpoint_has_distinct_cache_without_overwrite(prefix_case):
    config, manifest, calls = prefix_case
    research.reconstruct_prefix(config, manifest, zone="FR", start_day="2025-09-01", max_new_days=1)
    original = {p: p.read_bytes() for p in Path(config["project_root"]).rglob("raw.csv.gz")}
    research.reconstruct_prefix({**config, "checkpoint_sha256": "different"}, manifest, zone="FR", start_day="2025-09-01", max_new_days=1)
    assert all(p.read_bytes() == content for p, content in original.items())
    assert len(list(Path(config["project_root"]).rglob("raw.csv.gz"))) == 2
    assert len(calls) == 4


def test_original_panel_sha_verified_before_inference(prefix_case):
    config, manifest, calls = prefix_case
    Path(manifest["panel"]["path"]).write_bytes(b"changed")
    with pytest.raises(trial.ProspectiveTrialError):
        research.reconstruct_prefix(config, manifest, zone="FR", start_day="2025-09-01")
    assert calls == []


@pytest.mark.parametrize("kwargs", [{"threads": 0}, {"threads": True}, {"workers": 9}, {"stage": "promote"}, {"max_new_prefix_days": 0}])
def test_invalid_settings_fail_before_verifying_or_loading_models(kwargs):
    with pytest.raises(research.RollingResearchError):
        research.run_rolling_report({}, delivery_day="2026-09-08", zones=["FR"], **kwargs)


def test_actual_cli_fullreport_routes_to_isolated_workflow(monkeypatch):
    import run_lora_rank16_trial as cli
    monkeypatch.setattr(cli, "load_trial_config", lambda *_: {"test": True})
    calls = []
    monkeypatch.setattr(research, "run_rolling_report", lambda config, **kw: calls.append((config, kw)) or {"status": "ok"})
    monkeypatch.setattr(cli, "execute_trial", lambda *_a, **_kw: pytest.fail("No prospective execution"))
    monkeypatch.setattr(sys, "argv", ["runner", "--action", "fullreport", "--delivery-day", "2026-09-08", "--zones", "FR"])
    assert cli.main() == 0
    assert calls[0][1]["zones"] == ["FR"]
    assert calls[0][1]["delivery_day"] == "2026-09-08"


def test_compute_lock_prevents_duplicate_and_releases(tmp_path):
    with research._compute_lock(tmp_path):
        with pytest.raises(research.RollingResearchError, match="deja en cours"):
            with research._compute_lock(tmp_path):
                pytest.fail("Concurrent lock acquired")
    with research._compute_lock(tmp_path):
        pass
