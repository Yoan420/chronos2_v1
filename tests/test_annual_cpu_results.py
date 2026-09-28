from __future__ import annotations

import json
import hashlib
from pathlib import Path

from experiment_console.annual_cpu_results import RESULTS_PATH, RECEIPT_PATH, inspect_annual_cpu_results


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
