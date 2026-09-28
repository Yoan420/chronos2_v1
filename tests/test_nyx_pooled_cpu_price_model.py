"""Contract and executable checks for the independently named CPU expert."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_pooled_cpu_price_model as cpu


def synthetic_inputs():
    index = cpu.grid("2025-03-01", "2025-07-03")
    hours = np.arange(len(index), dtype=float)
    features, actual, nyx = {}, {}, {}
    for number, zone in enumerate(cpu.ZONES):
        features[zone] = pd.DataFrame({
            "known_load": 18. + np.sin(hours / 24.),
            "calendar_hour": hours % 24,
            f"own_{zone}": number + hours % 7,
            f"own_{zone}__available": np.ones(len(index)),
        }, index=index)
        actual[zone] = pd.Series(50. + number * 3 + np.cos(hours / 24.),
                                 index=index, name="actual_price")
        nyx[zone] = pd.Series(45. + number * 3 + np.sin(hours / 18.),
                              index=index, name="nyx_price")
    return features, actual, nyx


KW = dict(origin_day="2025-06-25", stop_day="2025-07-02",
          initial_training_day="2025-03-01")

# Frozen with the audited GPU input builder on this synthetic DST-spanning
# fixture. No GPU module or historical artifact is needed on a clean clone.
HISTORICAL_HASHES = {
    "training_features_sha256": "6b157ef1f159fe43642d037692be32060bf85bd91f3c175a55d1da5f64bdfc84",
    "prediction_features_sha256": "49c271fe301496df8c411fb58da7917dc651f64ee301cfd5bca936636b1eaa59",
    "prediction_base_sha256": "77f771f92937ed9dbd7d37ec9d9cdb62cf6603e458496420a73528bc759453ac",
}
HISTORICAL_TARGET_HASHES = {
    "residual": "dd49c39cc50ade06adc37c47960b47a0f38e95c38837b433060c23c08f462ab3",
    "absolute": "83e96390c9708e38571a29e5f41dd902deee737543c21e30c0713abaa8c3917e",
}
HISTORICAL_COLUMNS = [
    "calendar_hour", "known_load", "own_BE", "own_BE__available",
    "own_DE", "own_DE__available", "own_FR", "own_FR__available",
    "own_NL", "own_NL__available", "country_FR", "country_DE",
    "country_BE", "country_NL",
]


@pytest.mark.parametrize("iterations", [1000, 2000])
@pytest.mark.parametrize("mode", ["residual", "absolute"])
def test_historical_input_target_base_parity(iterations, mode):
    features, actual, nyx = synthetic_inputs()
    config = cpu.PooledConfig(target_mode=mode, iterations=iterations)
    train, target, current, base, audit = cpu.build_inputs(
        features, actual, nyx, config=config, **KW)
    assert audit["training_rows"] == 11132 and audit["prediction_rows"] == 672
    assert list(train) == list(current) == HISTORICAL_COLUMNS
    assert train.index.names == current.index.names == ["timestamp_utc", "zone"]
    assert train.index.get_level_values("zone")[:8].tolist() == list(cpu.ZONES) * 2
    for key, expected in HISTORICAL_HASHES.items():
        assert audit[key] == expected
    assert audit["training_target_sha256"] == HISTORICAL_TARGET_HASHES[mode]
    first = cpu.grid(KW["initial_training_day"], KW["origin_day"])
    for zone in cpu.ZONES:
        expected = actual[zone].loc[first]
        if mode == "residual":
            expected = expected - nyx[zone].loc[first]
        np.testing.assert_array_equal(target.xs(zone, level="zone").to_numpy(),
                                      expected.to_numpy())
        assert current.xs(zone, level="zone")[f"own_{zone}__available"].eq(1.).all()
        for other in set(cpu.ZONES) - {zone}:
            assert current.xs(zone, level="zone")[f"own_{other}__available"].eq(0.).all()
    assert base.index.equals(current.index)
    assert audit["parameters"]["task_type"] == "CPU"
    assert audit["GPU_historical_scores_reproduced"] is False


def test_cpu_parameters_and_prospective_history_contract():
    params = cpu.parameters(cpu.PooledConfig(iterations=2000), thread_count=2)
    assert params["task_type"] == "CPU" and params["iterations"] == 2000
    assert params["depth"] == 7 and params["learning_rate"] == .035
    assert params["l2_leaf_reg"] == 10. and params["border_count"] == 254
    assert "devices" not in params and "gpu_ram_part" not in params
    features, actual, nyx = synthetic_inputs()
    with pytest.raises(ValueError, match="chronological training window"):
        cpu.build_inputs(features, actual, nyx, origin_day="2025-06-25",
                         stop_day="2025-07-02", initial_training_day="2025-05-01")


def test_cpu_fit_saved_inference_and_future_label_independence(tmp_path):
    features, actual, nyx = synthetic_inputs()
    config = cpu.PooledConfig(target_mode="residual", iterations=4)
    path = tmp_path / "expert.cbm"
    points, audit = cpu.fit_pooled_block(features, actual, nyx, config=config,
                                         model_path=path, **KW)
    assert audit["tree_count"] == 4 and audit["model"]["saved"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit["model_sha256"] == digest
    assert audit["fitted_parameters"]["task_type"] == "CPU"
    changed = {zone: series.copy() for zone, series in actual.items()}
    for zone in cpu.ZONES:
        changed[zone].loc[cpu.grid(KW["origin_day"], KW["stop_day"])] = np.nan
    replay, inference = cpu.predict_saved_block(path, features, changed, nyx,
                                                 config=config,
                                                 expected_model_sha256=digest, **KW)
    assert inference["models_fitted"] == 0 and inference["inference_task_type"] == "CPU"
    for zone in cpu.ZONES:
        pd.testing.assert_frame_equal(replay[zone], points[zone], check_exact=True)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        cpu.predict_saved_block(path, features, actual, nyx, config=config,
                                expected_model_sha256="0" * 64, **KW)
    json.dumps(audit, allow_nan=False)


def test_real_historical_gpu_checkpoint_infers_on_cpu_when_locally_available():
    root = Path(__file__).resolve().parents[1]
    exp = root / "runs/experiments/nyx_improvement_to20260923"
    checkpoint = exp / "rmse_boosting_2000_gpu_v1/residual/checkpoints/2026-09-23/receipt.json"
    feature_root = exp / "feature_sets/pooled_jao_refresh_v1/compact"
    baseline_root = root / "runs/experiments/nyx_local_365_to20260923/baseline"
    if not checkpoint.is_file() or not all((feature_root / f"features_{zone}.parquet").is_file()
                                              and (baseline_root / f"{zone}.parquet").is_file()
                                              for zone in cpu.ZONES):
        pytest.skip("Local historical research artifacts are absent from this checkout")
    receipt = json.loads(checkpoint.read_text(encoding="utf-8"))
    model_path = checkpoint.parent / receipt["attempt_directory"] / "model.cbm"
    features = {zone: pd.read_parquet(feature_root / f"features_{zone}.parquet")
                for zone in cpu.ZONES}
    baselines = {zone: pd.read_parquet(baseline_root / f"{zone}.parquet")
                 for zone in cpu.ZONES}
    actual = {zone: baseline.actual for zone, baseline in baselines.items()}
    nyx = {zone: baseline.nyx__q50 for zone, baseline in baselines.items()}
    points, audit = cpu.predict_saved_block(
        model_path, features, actual, nyx, origin_day="2026-09-23",
        stop_day="2026-09-24", config=cpu.PooledConfig(iterations=2000),
        expected_model_sha256=receipt["files"]["model.cbm"])
    assert audit["models_fitted"] == 0 and audit["tree_count"] == 2000
    for zone in cpu.ZONES:
        expected = pd.read_parquet(exp / "rmse_boosting_2000_gpu_v1/residual"
                                   / f"{zone}_annual.parquet").loc[points[zone].index]
        np.testing.assert_allclose(points[zone].point.to_numpy(),
                                   expected.point.to_numpy(), rtol=1e-12, atol=1e-10)
