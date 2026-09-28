"""Test chronological separation, evidence tamper rejection and activation gates."""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_cpu_full_chain as chain
from chronos2_hourly import nyx_annual_cpu_live as live
from chronos2_hourly.nyx_annual_cpu_baseline import chronos_identity_digest
from chronos2_hourly.nyx_annual_nyx_quantiles_gate import CHRONOS_REVISION


def chronos_pin(weight_hash="b"*64):
    model = {"model_id": "amazon/chronos-2", "revision": CHRONOS_REVISION,
        "device": "cpu", "dtype": "torch.float32",
        "files": {"config.json": "a"*64, "model.safetensors": weight_hash}}
    return {"chronos_model_identity": model, "chronos_model_sha256": chronos_identity_digest(model)}


def test_windows_checkout_line_endings_only_normalized_for_code(tmp_path):
    unix, windows = tmp_path / "unix.py", tmp_path / "windows.py"
    unix.write_bytes(b"value = 1\nsecond = 2\n")
    windows.write_bytes(b"value = 1\r\nsecond = 2\r\n")
    assert chain._code_sha256(unix) == chain._code_sha256(windows)
    assert chain.sha256(unix) != chain.sha256(windows)
    windows.write_bytes(b"value = 2\r\nsecond = 2\r\n")
    assert chain._code_sha256(unix) != chain._code_sha256(windows)


def test_annual_receipt_requires_cpu_chain_and_de_exception_is_only_performance(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "_code", lambda root: {"source": "a"*64})
    monkeypatch.setattr(chain, "_versions", lambda: {"runtime": "fixed"})
    days = chain._days("2025-09-24", "2026-09-24")
    receipt = {"protocol": chain.PROTOCOL, "qualified": True,
        **chronos_pin(),
        "full_input_chain_qualified": True, "negative_replay_verified": True,
        "price_expert_replay_qualified": True, "first_delivery_day": days[0],
        "last_delivery_day": days[-1], "days_evaluated": 365, "physical_hours": 8760,
        "price_experts": list(live.PRICE_EXPERTS), "compositions": live.COMPOSITIONS,
        "negative_model_protocol": live.NEGATIVE_PROTOCOL,
        "price_threads": 8, "negative_threads": 2, "frequency": "daily_retraining",
        "de_exception": chain.DE_EXCEPTION, "predictions_sealed_before_scoring": True,
        "source_snapshots_asof_verified": True,
        "code_sha256": chain._code(tmp_path), "code_hash_policy": chain.CODE_HASH_POLICY,
        "runtime_versions": chain._versions(),
        "evidence_sha256": {"evaluation_plan": "a"*64,
            "official_comparisons_receipt": "d"*64,
            "prediction_receipts": {d: "b"*64 for d in days},
            "official_comparisons": {z: "c"*64 for z in live.COUNTRIES}},
        "price_country_metrics": {z: {"hours": 8759, "storm_common_hours": 8759,
            "rmse": 10 if z != "DE" else 99, "storm_rmse": 11,
            "strict_win_rate": .51 if z != "DE" else .1} for z in live.COUNTRIES},
        "negative_country_metrics": {z: {"hours": 8760, "brier": .02} for z in live.COUNTRIES}}
    chain.verify_receipt(receipt, root=tmp_path)
    receipt["chronos_model_sha256"] = "c"*64
    with pytest.raises(ValueError, match="Chronos weight identity checksum"):
        chain.verify_receipt(receipt, root=tmp_path)
    receipt.update(chronos_pin())
    receipt["price_country_metrics"]["FR"]["rmse"] = 12
    with pytest.raises(ValueError, match="FR: CPU price does not qualify"):
        chain.verify_receipt(receipt, root=tmp_path)
    receipt["price_country_metrics"]["FR"]["rmse"] = 10
    receipt["negative_country_metrics"]["DE"]["hours"] = 0
    with pytest.raises(ValueError, match="DE: negative-probability"):
        chain.verify_receipt(receipt, root=tmp_path)
    receipt["negative_country_metrics"]["DE"]["hours"] = 8760
    receipt["full_input_chain_qualified"] = False
    with pytest.raises(ValueError, match="Full-chain qualification scope"):
        chain.verify_receipt(receipt, root=tmp_path)


def _small_plan(tmp_path, monkeypatch, first="2026-10-25"):
    monkeypatch.setattr(chain, "_code", lambda root: {"source": "a"*64})
    monkeypatch.setattr(chain, "_versions", lambda: {"runtime": "fixed"})
    monkeypatch.setattr(chain, "inspect_bundle", lambda *_: {"input_bundle_valid": True})
    monkeypatch.setattr(chain, "validate_nyx_quantiles_source", lambda *_: {})
    monkeypatch.setattr(chain, "_source_packet", lambda *_: {"baseline": chronos_pin()})
    monkeypatch.setattr(chain, "_official_comparisons", lambda *_: {})
    monkeypatch.setattr(live, "bundle_hashes", lambda bundle: {"source": "d"*64})
    observations = tmp_path / "observations"
    observations.mkdir()
    (observations / "comparisons_receipt.json").write_text('{"synthetic": true}')
    index = chain.delivery_grid(first)[1]
    for zone in live.COUNTRIES:
        pd.DataFrame({"actual": np.linspace(-5., 5., len(index)),
                      "storm": np.full(len(index), 50.)}, index=index).to_parquet(observations/f"{zone}.parquet")
    output = tmp_path / "evaluation"
    stop = (pd.Timestamp(first).date() + timedelta(days=1)).isoformat()
    plan = chain.prepare_plan(root=tmp_path, bundles=tmp_path/"bundles",
        comparisons=observations, output=output, first=first, stop=stop)
    return output, plan, observations


def test_evaluation_refuses_a_different_chronos_model_after_plan(tmp_path, monkeypatch):
    output, plan, observations = _small_plan(tmp_path, monkeypatch)
    assert chain._chronos_pin(plan) == chronos_pin()
    monkeypatch.setattr(chain, "_source_packet", lambda *_: {"baseline": chronos_pin("c"*64)})
    monkeypatch.setattr(live, "execute_models", lambda *_: pytest.fail("Different model reached fitting"))
    with pytest.raises(ValueError, match="Chronos weights differ"):
        chain.predict_plan(root=tmp_path, output=output)


def test_current_labels_never_opened_before_all_predictions_sealed(tmp_path, monkeypatch):
    output, plan, observations = _small_plan(tmp_path, monkeypatch)
    pd_read = pd.read_parquet
    comparison_reads = []
    def read(path, *args, **kwargs):
        if Path(path).parent == observations:
            comparison_reads.append(path)
        return pd_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", read)
    monkeypatch.setattr(chain, "_official_comparisons",
                        lambda *_: pytest.fail("Official labels accessed before prediction sealing"))
    with pytest.raises(FileNotFoundError):
        chain.score_plan(root=tmp_path, output=output)
    assert comparison_reads == []


def test_short_dst_day_evaluation_scores_four_countries_but_cannot_activate(tmp_path, monkeypatch):
    output, plan, observations = _small_plan(tmp_path, monkeypatch)
    day = plan["delivery_days"][0]
    index = chain.delivery_grid(day)[1]
    assert len(index) == 25
    monkeypatch.setattr(live, "load_bundle", lambda *_: object())
    def fit(data, delivery_day, model_dir):
        model_dir.mkdir(parents=True)
        price_audits, negative_audits = {}, {}
        for family in live.PRICE_EXPERTS:
            path = model_dir/f"{family}.cbm"
            path.write_bytes(b"test price model")
            price_audits[family] = {"training_labels_before_origin": True,
                "training_days": 365,
                "Storm_used_as_input": False,
                "tree_count": 1000 if family == "fr_residual_1000" else 2000,
                "model": {"sha256": chain.sha256(path)}}
        frames = {}
        for zone in live.COUNTRIES:
            path = model_dir/f"negative_{zone}.cbm"
            path.write_bytes(b"test negative model")
            negative_audits[zone] = {"forecast_labels_used": False, "storm_used": False,
                "tree_count": 120, "models_fitted": 1, "model": {"sha256": chain.sha256(path)}}
            frames[zone] = pd.DataFrame({"price_eur_mwh": np.zeros(len(index)),
                "p_negative": np.full(len(index), .5)}, index=index)
        return frames, {"price_experts": price_audits, "negative_countries": negative_audits,
                        "compositions": live.COMPOSITIONS}
    legacy = output / "predictions" / day / "models"
    legacy.mkdir(parents=True)
    (legacy / "unfinished.cbm").write_bytes(b"preserve interrupted model")
    def interrupted_fit(data, delivery_day, model_dir):
        model_dir.mkdir()
        (model_dir / "partial.cbm").write_bytes(b"partially completed model")
        raise RuntimeError("Simulated interruption")
    monkeypatch.setattr(live, "execute_models", interrupted_fit)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        chain.predict_plan(root=tmp_path, output=output)
    assert not (output / "predictions" / day).exists()
    assert len(list((output / "attempts" / day).glob("*/models/*.cbm"))) == 2
    monkeypatch.setattr(live, "execute_models", fit)
    status = chain.predict_plan(root=tmp_path, output=output)
    assert status["state"] == "PREDICTIONS_SEALED"
    # Repeated prediction command validates the immutable result, no new fit.
    monkeypatch.setattr(live, "execute_models", lambda *_: pytest.fail("Unexpected refit"))
    assert chain.predict_plan(root=tmp_path, output=output) == status
    score = chain.qualify(root=tmp_path, output=output)
    assert score["qualified"] is False and score["full_input_chain_qualified"] is False
    assert score["physical_hours"] == 25
    assert set(score["price_country_metrics"]) == set(live.COUNTRIES)
    with pytest.raises(ValueError, match="Full-chain qualification scope"):
        chain.qualify(root=tmp_path, output=output, activate=True)
    assert not (tmp_path/"config"/live.MANIFEST.name).exists()
    (output/"predictions"/day/"DE.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="sealed predictions changed"):
        chain.score_plan(root=tmp_path, output=output)


def test_changed_official_comparison_rejected(tmp_path, monkeypatch):
    output, plan, observations = _small_plan(tmp_path, monkeypatch)
    day = plan["delivery_days"][0]
    folder = output/"predictions"/day
    folder.mkdir(parents=True)
    (folder/"receipt.json").write_text("{}")
    monkeypatch.setattr(chain, "_prediction_receipt", lambda *_: {})
    for z in live.COUNTRIES:
        pd.DataFrame({"price_eur_mwh": np.zeros(25), "p_negative": np.full(25,.5)},
            index=chain.delivery_grid(day)[1]).to_parquet(folder/f"{z}.parquet")
    (observations/"FR.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="official comparison changed"):
        chain.score_plan(root=tmp_path, output=output)
