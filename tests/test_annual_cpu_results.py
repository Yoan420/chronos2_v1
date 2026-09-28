from __future__ import annotations

import json
import hashlib
from pathlib import Path

from experiment_console.annual_cpu_results import RESULTS_PATH, RECEIPT_PATH, DE_RESULTS_PATH, inspect_annual_cpu_results


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture_root(tmp_path: Path) -> Path:
    for relative in (RESULTS_PATH, RECEIPT_PATH):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / relative).read_bytes())
    return tmp_path


def test_pinned_cpu_replay_is_read_only_and_keeps_future_forecast_locked(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    expected_rmse = {"FR": 18.596481350684698, "BE": 21.202686840287583, "NL": 20.56667435723672}
    expected_precision = {"FR": 0.8086021505376344, "BE": 0.7305194805194806, "NL": 0.786697247706422}
    expected_recall = {"FR": 0.6482758620689655, "BE": 0.6859756097560976, "NL": 0.7903225806451613}
    expected_brier = {"FR": 0.023431041496344274, "BE": 0.013765561088228854, "NL": 0.014937370846933024}
    for country in ("FR", "BE", "NL"):
        result = inspect_annual_cpu_results(root, country)
        assert result["available"] is True
        assert result["price"]["rmse"] == expected_rmse[country]
        assert isinstance(result["price"]["composition"], str)
        assert result["negative"]["precision"] == expected_precision[country]
        assert result["negative"]["recall"] == expected_recall[country]
        assert result["negative"]["brier"] == expected_brier[country]
        assert result["price_expert_replay_qualified"] is True
        assert result["full_input_chain_qualified"] is False
        assert result["qualified"] is False
        assert result["forecast_ready"] is False
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_cpu_replay_fails_closed_if_receipt_or_diagnostics_change(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    receipt = root / RECEIPT_PATH
    receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
    changed = inspect_annual_cpu_results(root, "FR")
    assert changed["available"] is False
    assert changed["forecast_ready"] is False

    receipt.write_bytes((REPO_ROOT / RECEIPT_PATH).read_bytes())
    summary = root / RESULTS_PATH
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["negative_diagnostics"]["FR"]["precision"] = 1.1
    summary.write_text(json.dumps(data), encoding="utf-8")
    invalid = inspect_annual_cpu_results(root, "FR")
    assert invalid["available"] is False
    assert invalid["forecast_ready"] is False


def test_cpu_replay_missing_files_remains_unavailable(tmp_path: Path) -> None:
    result = inspect_annual_cpu_results(tmp_path, "FR")
    assert result["available"] is False
    assert result["forecast_ready"] is False


def test_cpu_replay_does_not_assume_a_new_qualification_state(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    receipt_path = root / RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["qualified"] = True
    receipt_bytes = json.dumps(receipt).encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    summary_path = root / RESULTS_PATH
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["qualification_receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = inspect_annual_cpu_results(root, "FR")
    assert result["available"] is False
    assert result["forecast_ready"] is False


def test_new_production_qualification_does_not_destroy_original_fr_be_nl_scores(tmp_path):
    root = _fixture_root(tmp_path)
    canonical = root / "config/nyx_annual_cpu_qualification_receipt.json"
    canonical.write_text('{"protocol":"new full-chain qualification"}')
    assert inspect_annual_cpu_results(root, "FR")["price"]["rmse"] == 18.596481350684698


def test_de_accepts_only_pinned_complete_replay_and_keeps_actual_failed_score(tmp_path):
    root = _fixture_root(tmp_path)
    receipt = {"identity": "nyx_de_cpu_annual_results_20260929_v1", "country": "DE",
        "replay_date": "2026-09-29", "composition": "boosting_mean_disagreement20",
        "first_delivery_day": "2025-09-24", "last_delivery_day": "2026-09-23",
        "price_fits": 106, "negative_fits": 53, "frozen_legacy_baseline_and_reference": True,
        "negative_replay_verified": True, "full_input_chain_qualified": False,
        "qualified": False, "performance_exception_authorized": True,
        "replay_receipts_sha256": {"price": "a"*64, "negative": "b"*64},
        "output_sha256": {"price": "c"*64, "negative": "d"*64},
        "price": {"hours": 8759, "storm_common_hours": 8759, "strict_wins": 5000,
            "strict_win_rate": 5000/8759, "rmse": 21., "storm_rmse": 20.},
        "negative": {"hours": 8760, "brier": .02, "precision": .8, "recall": .8}}
    path = root / DE_RESULTS_PATH
    path.write_text(json.dumps(receipt))
    summary = json.loads((root/RESULTS_PATH).read_text())
    summary["de_cpu_results"] = {"path": DE_RESULTS_PATH.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root/RESULTS_PATH).write_text(json.dumps(summary))
    result = inspect_annual_cpu_results(root, "DE")
    assert result["available"] is True
    assert result["forecast_ready"] is False and result["full_input_chain_qualified"] is False
    assert result["price"]["rmse"] == 21.
    assert result["price_expert_replay_qualified"] is False
    assert result["performance_exception_authorized"] is True
    path.write_text(path.read_text()+" ")
    assert inspect_annual_cpu_results(root,"DE")["available"] is False


def test_real_de_cpu_result_is_portable_and_preserves_failed_storm_criterion(tmp_path):
    root = _fixture_root(tmp_path)
    (root / DE_RESULTS_PATH).write_bytes((REPO_ROOT / DE_RESULTS_PATH).read_bytes())
    result = inspect_annual_cpu_results(root, "DE")
    assert result["available"] is True
    assert result["forecast_ready"] is False and result["full_input_chain_qualified"] is False
    assert result["performance_exception_authorized"] is True
    assert result["price_expert_replay_qualified"] is False
    assert result["price"]["rmse"] == 21.23112573005414
    assert result["price"]["storm_rmse"] == 20.306583759838535
    assert result["price"]["strict_wins"] == 4775
    assert result["price"]["strict_win_rate"] == 4775/8759
    assert result["negative"]["brier"] == 0.013845741445702215
    assert result["negative"]["precision"] == 0.841897233201581
    assert result["negative"]["recall"] == 0.8239845261121856
