from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_timer_s1_comparison as runner


TIMEZONE = "Europe/Paris"
MODEL_ID = "bytedance-research/Timer-S1"
REVISION = "8911430cc7f32add5c8913afe12e3b05742f5bb2"
INFERENCE_CONTRACT = {
    "revin": True,
    "use_cache": False,
    "torch_dtype": "bfloat16",
    "model_kwargs": {"use_cache": False},
    "quantile_indices": {"q10": 0, "q50": 4, "q90": 8},
}


def _protocol() -> SimpleNamespace:
    index = pd.date_range(
        "2023-12-01",
        periods=96,
        freq="h",
        tz="UTC",
        name="delivery_start_utc",
    )
    target = pd.Series(
        np.linspace(40.0, 80.0, len(index)),
        index=index,
        name="target",
    )
    return SimpleNamespace(
        target=target,
        feature_manifest_sha256="f" * 64,
        residual_recipe_sha256="r" * 64,
    )


def _valid_sidecar(artifact: Path, protocol: SimpleNamespace) -> dict[str, object]:
    return {
        "model_name": "timer_s1",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "context_length": 2048,
        "forecast_mode": "strict_native_target_only",
        "native_covariates": [],
        "source_feature_manifest_sha256": protocol.feature_manifest_sha256,
        "source_target_sha256": runner.sha256_target_series(protocol.target),
        "source_residual_recipe_sha256": protocol.residual_recipe_sha256,
        "inference_contract": INFERENCE_CONTRACT,
        "artifact_sha256": runner.sha256_file(artifact),
    }


def _write_sidecar(artifact: Path, sidecar: dict[str, object]) -> Path:
    path = runner._sidecar_path(artifact)
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    return path


def _require_sidecar(artifact: Path, protocol: SimpleNamespace) -> dict[str, object]:
    return runner._require_generation_sidecar(
        artifact,
        model_name="timer_s1",
        model_id=MODEL_ID,
        revision=REVISION,
        context_length=2048,
        inference_contract=INFERENCE_CONTRACT,
        protocol=protocol,
    )


def test_cli_requires_one_explicit_action() -> None:
    with pytest.raises(SystemExit) as exc_info:
        runner.parse_args([])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("context_length", 1024),
        ("extended_days", 222),
        ("calibration_days", 364),
        ("evaluation_days", 366),
    ],
)
def test_frozen_protocol_rejects_context_or_split_changes(
    tmp_path: Path,
    setting: str,
    value: int,
) -> None:
    args = runner.parse_args(["--plan-only"])
    comparison = {
        "context_length": 2048,
        "extended_days": 223,
        "calibration_days": 365,
        "evaluation_days": 365,
    }
    comparison[setting] = value
    config = {"project_root": ".", "comparison": comparison}

    with pytest.raises(ValueError, match="protocole de comparaison est gele"):
        runner._resolved_settings(tmp_path / "config.yaml", config, args)


def test_generation_sidecar_accepts_exact_artifact_target_and_features(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    artifact = tmp_path / "timer_oof.csv.gz"
    artifact.write_bytes(b"synthetic-timer-oof")
    expected = _valid_sidecar(artifact, protocol)
    _write_sidecar(artifact, expected)

    observed = _require_sidecar(artifact, protocol)

    assert observed == expected


def test_generation_sidecar_normalizes_legacy_pandas_microsecond_hash(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    artifact = tmp_path / "timer_oof.csv.gz"
    artifact.write_bytes(b"synthetic-timer-oof")
    sidecar = _valid_sidecar(artifact, protocol)
    legacy_hash = runner.legacy_target_hashes_by_timestamp_unit(
        protocol.target
    )["us"]
    sidecar["source_target_sha256"] = legacy_hash
    _write_sidecar(artifact, sidecar)

    observed = _require_sidecar(artifact, protocol)

    assert observed["source_target_sha256"] == runner.sha256_target_series(
        protocol.target
    )
    assert observed["source_target_sha256_recorded"] == legacy_hash
    assert observed["source_target_hash_compatibility"] == {
        "status": "accepted_legacy_datetime_unit",
        "legacy_datetime_unit": "us",
        "canonical_datetime_unit": "ns",
    }


def test_generation_sidecar_is_mandatory(tmp_path: Path) -> None:
    protocol = _protocol()
    artifact = tmp_path / "timer_oof.csv.gz"
    artifact.write_bytes(b"synthetic-timer-oof")

    with pytest.raises(ValueError, match="sidecar de provenance absent"):
        _require_sidecar(artifact, protocol)


@pytest.mark.parametrize(
    ("tampering", "message"),
    [
        ("artifact_sha", "SHA-256 different"),
        ("missing_artifact_sha", "artifact_sha256 absent ou invalide"),
        ("target", "source_target_sha256"),
        ("feature_manifest", "source_feature_manifest_sha256"),
    ],
)
def test_generation_sidecar_rejects_tampered_provenance(
    tmp_path: Path,
    tampering: str,
    message: str,
) -> None:
    protocol = _protocol()
    artifact = tmp_path / "timer_oof.csv.gz"
    artifact.write_bytes(b"synthetic-timer-oof")
    sidecar = _valid_sidecar(artifact, protocol)
    if tampering == "artifact_sha":
        sidecar["artifact_sha256"] = "0" * 64
    elif tampering == "missing_artifact_sha":
        del sidecar["artifact_sha256"]
    elif tampering == "target":
        sidecar["source_target_sha256"] = "0" * 64
    elif tampering == "feature_manifest":
        sidecar["source_feature_manifest_sha256"] = "0" * 64
    else:  # pragma: no cover - parameter list is closed
        raise AssertionError(tampering)
    _write_sidecar(artifact, sidecar)

    with pytest.raises(ValueError, match=message):
        _require_sidecar(artifact, protocol)


def test_post_publication_sensitivity_excludes_earlier_days() -> None:
    index = pd.date_range(
        "2026-04-09",
        "2026-04-12",
        inclusive="left",
        freq="h",
        tz=TIMEZONE,
    ).tz_convert("UTC")
    index.name = "delivery_start_utc"
    local_days = pd.Index(index.tz_convert(TIMEZONE).date)
    post_publication = np.asarray(
        local_days >= runner.POST_PUBLICATION_START_DAY,
        dtype=bool,
    )
    predictions = pd.DataFrame(
        {
            "actual": np.zeros(len(index)),
            # Large pre-publication errors prove the secondary sensitivity
            # window does not leak observations from 2026-04-09.
            "baseline__q50": np.where(post_publication, 2.0, 200.0),
            "candidate__q50": np.where(post_publication, 1.0, 100.0),
        },
        index=index,
    )

    result = runner._post_publication_sensitivity(
        predictions,
        timezone=TIMEZONE,
        pairs={"candidate_minus_baseline": ("baseline", "candidate")},
        bootstrap_samples=50,
        seed=7,
    )

    metrics = {item["model"]: item for item in result["metrics"]}
    assert result["role"] == "secondary_sensitivity_not_primary_metric"
    assert result["first_included_local_day"] == "2026-04-10"
    assert result["last_included_local_day"] == "2026-04-11"
    assert metrics["baseline"] == {
        "model": "baseline",
        "n_hours": 48,
        "n_days": 2,
        "mae_q50": 2.0,
    }
    assert metrics["candidate"] == {
        "model": "candidate",
        "n_hours": 48,
        "n_days": 2,
        "mae_q50": 1.0,
    }
    paired = result["paired_daily_mae"]["candidate_minus_baseline"]
    assert paired["n_delivery_days"] == 2
    assert paired["delta_mae"] == -1.0
    assert paired["ci95_delta_mae"] == [-1.0, -1.0]
