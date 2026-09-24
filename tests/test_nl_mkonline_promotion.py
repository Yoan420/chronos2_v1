from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chronos2_hourly.live_history import (
    discover_archived_forecasts,
    missing_statistics_archive_days,
)
from chronos2_hourly.multizone_contract import load_zone_model_contract
from chronos2_hourly.multizone_live import load_prediction_policy
from chronos2_modular.common import load_yaml


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "runs/chronos2_hourly_nl_mkonline_blend_v1"
LEGACY_BENCHMARK = ROOT / "runs/chronos2_hourly_nl_sealed_benchmark_v1"
LIVE_CONFIG = ROOT / "chronos2_hourly_nl_mkonline_live_v1.yaml"
REGISTRY = ROOT / "chronos2_hourly_live_zones.yaml"
NEW_ROOT = ROOT / "runs/live/nl_mkonline_v1"
LEGACY_D12 = ROOT / "runs/live/nl/_replays/nl_day_ahead_2026-08-12"
NEW_D12 = NEW_ROOT / "_replays/nl_day_ahead_2026-08-12"

WEIGHT_MK = 0.4677256033079484
WEIGHT_AUTO = 0.5322743966920516


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_nl_selection_and_sealed_benchmark_are_checksum_consistent() -> None:
    recipe_path = ROOT / "mkonline_nl_blend_recipe_v1.json"
    recipe = _json(recipe_path)
    assert _sha256(recipe_path) == (
        "de691832431b15acacc5336a6ae2587a677c899cc7c7e1ac87b81e986c4fc6f3"
    )
    assert recipe["status"] == "frozen_before_final_opening"
    assert recipe["recipe_mode"] == "mkonline_blend"
    assert recipe["final_target_used_for_weight_or_hyperparameters"] is False
    assert recipe["weights"] == {
        "mkonline_primary": WEIGHT_MK,
        "autonomous": WEIGHT_AUTO,
    }
    selection = recipe["selection_protocol"]
    assert isinstance(selection, dict)
    assert selection["final_loaded"] is False
    assert selection["final_target_used_for_weight_or_hyperparameters"] is False
    for block, path_key, hash_key in (
        ("B1", "b1_gate_artifact", "b1_gate_artifact_sha256"),
        ("B2", "b2_veto_artifact", "b2_veto_artifact_sha256"),
    ):
        path = ROOT / str(selection[path_key])
        assert _sha256(path) == selection[hash_key]
        gate = _json(path)
        assert gate[block]["passes"] is True
        assert gate["protocol"]["final_loaded"] is False

    checksum_path = BENCHMARK / "artifact_checksums.json"
    assert _sha256(checksum_path) == (
        "a99c38a6f023f91c2e5e6c6cd2a3f18d956071236cfa9ce217aba1ed5a6eebbb"
    )
    checksum = _json(checksum_path)
    for item in checksum["artifacts"]:
        if item.get("role") != "run_artifact":
            continue
        relative = Path(str(item["path"]))
        assert not relative.is_absolute() and ".." not in relative.parts
        artifact = BENCHMARK / relative
        assert artifact.is_file()
        assert _sha256(artifact) == item["sha256"]

    manifest = _json(BENCHMARK / "run_manifest.json")
    assert manifest["zone"] == "NL"
    assert manifest["timezone"] == "Europe/Amsterdam"
    assert manifest["recipe_mode"] == "mkonline_blend"
    assert manifest["native_model"] == "mkonline_blend"
    assert manifest["prediction_inputs"] == [
        "autonomous_extended_residual",
        "41554_native",
    ]
    assert manifest["storm_used_as_feature"] is False
    assert manifest["storm_evaluation_only_loaded_after_candidate_frozen"] is True

    statistics = pd.read_csv(BENCHMARK / "statistics_history_hourly.csv.gz")
    delivery = pd.DatetimeIndex(
        pd.to_datetime(statistics["delivery_start_utc"], utc=True, errors="raise")
    )
    assert len(statistics) == 8760 and delivery.is_unique
    origin = pd.DatetimeIndex(
        pd.to_datetime(
            statistics["mkonline_blend_forecast_origin_utc"],
            utc=True,
            errors="raise",
        )
    )
    assert bool((origin < delivery).all()) and origin.nunique() == 365
    actual = pd.to_numeric(statistics["actual"], errors="raise")
    autonomous = pd.to_numeric(
        statistics["residual_corrected__q50"], errors="raise"
    )
    blend = pd.to_numeric(statistics["mkonline_blend__q50"], errors="raise")
    storm = pd.to_numeric(
        statistics["storm_dashboard_official__q50"], errors="coerce"
    )
    assert np.isclose(np.abs(autonomous - actual).mean(), 11.174835316072068)
    assert np.isclose(np.abs(blend - actual).mean(), 10.647611177392772)

    daily = pd.DataFrame(
        {
            "day": delivery.tz_convert("Europe/Amsterdam").date,
            "candidate_error": np.abs(blend - actual),
            "storm_error": np.abs(storm - actual),
            "storm_finite": np.isfinite(storm),
        }
    ).groupby("day", sort=True).agg(
        candidate_mae=("candidate_error", "mean"),
        storm_mae=("storm_error", "mean"),
        storm_finite=("storm_finite", "all"),
    )
    complete = daily.loc[daily["storm_finite"]]
    assert len(complete) == 364
    assert int((complete["candidate_mae"] < complete["storm_mae"]).sum()) == 193
    assert daily.index[~daily["storm_finite"]].tolist() == [date(2025, 10, 26)]


def test_nl_live_contract_promotes_primary_and_uses_versioned_output_root() -> None:
    contract = load_zone_model_contract(LIVE_CONFIG, REGISTRY, strict=True)
    config = load_yaml(LIVE_CONFIG)
    live = config["live"]
    policy = load_prediction_policy(contract, live)
    assert contract.zone == "NL"
    assert policy.mode == "mkonline_blend"
    assert policy.candidate_model == "mkonline_blend"
    assert contract.primary_series == "41554_native"
    assert contract.weights.mkonline_primary == WEIGHT_MK
    assert contract.weights.autonomous == WEIGHT_AUTO
    assert contract.paths.output_root == NEW_ROOT.resolve()
    assert contract.paths.output_root != (ROOT / "runs/live/nl").resolve()
    assert contract.paths.sealed_benchmark_run == BENCHMARK.resolve()


def test_nl_migration_replays_cover_the_initial_suffix() -> None:
    forecasts, audits = discover_archived_forecasts(
        live_output_root=NEW_ROOT,
        replay_output_root=NEW_ROOT / "_replays",
        current_delivery_day=date(2026, 8, 13),
        first_history_day=date(2026, 8, 12),
        timezone="Europe/Amsterdam",
        forecast_name="forecast_hourly_nl.csv",
        candidate_model="mkonline_blend",
        zone="NL",
        target_series="power.price.da.nl.bzn.hourly.entsoe.utc.cdh.eurmwh",
        prediction_mode="mkonline_blend",
    )
    assert list(forecasts) == [date(2026, 8, 12)]
    assert len(audits) == 1 and audits[0]["archive_kind"] == "pit_replay"

    manifest = _json(NEW_D12 / "run_manifest.json")
    assert manifest["prediction_mode"] == "mkonline_blend"
    assert manifest["candidate_model"] == "mkonline_blend"
    assert manifest["prediction_inputs"] == [
        "autonomous_extended_residual",
        "41554_native",
    ]
    assert manifest["storm_used_as_feature"] is False
    assert manifest["storm_loaded_for_prediction"] is False
    assert manifest["statistics_history"]["status"] == (
        "deferred_to_following_live_run"
    )
    assert manifest["live_fit"]["training_or_refit_performed"] is False
    assert manifest["live_fit"]["network_used"] is False

    missing = missing_statistics_archive_days(
        sealed_benchmark_run=BENCHMARK,
        live_output_root=NEW_ROOT,
        replay_output_root=NEW_ROOT / "_replays",
        current_delivery_day=date(2026, 8, 16),
        timezone="Europe/Amsterdam",
        forecast_name="forecast_hourly_nl.csv",
        candidate_model="mkonline_blend",
        zone="NL",
        target_series="power.price.da.nl.bzn.hourly.entsoe.utc.cdh.eurmwh",
        prediction_mode="mkonline_blend",
    )
    assert missing == []


def test_nl_autonomous_benchmark_and_replay_remain_immutable() -> None:
    assert _sha256(LEGACY_BENCHMARK / "artifact_checksums.json") == (
        "5d38ec3bd7a5ea48ac230e1790f6360be16122c967004b25a011a5cca928b7af"
    )
    assert _sha256(LEGACY_D12 / "forecast_hourly_nl.csv") == (
        "04f64b57477e1c4ca532355be7c80d3cadc0e015f683e82f9b878d1ef790923b"
    )
    assert _sha256(LEGACY_D12 / "artifact_checksums.json") == (
        "37ff11121a666bc7c4b0662382538c545bd9dfeae940afcf74fd83645bf9e16a"
    )
