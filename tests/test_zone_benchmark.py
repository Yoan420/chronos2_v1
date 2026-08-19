from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.zone_benchmark import (
    BenchmarkIdentity,
    BenchmarkContractError,
    attach_native_storm_for_statistics,
    build_frozen_candidate,
    validate_dependency_manifest,
    validate_recipe_manifest,
)


ZONE = "DE"
TIMEZONE = "Europe/Berlin"
SOURCE_CHECKSUM = "a" * 64


def _identity() -> BenchmarkIdentity:
    return BenchmarkIdentity(
        zone=ZONE,
        timezone=TIMEZONE,
        target_series="power.price.da.de_lu.bzn.hourly.entsoe.utc.cdh.eurmwh",
        primary_series="41550_native",
        storm_dashboard_series="power.price.de.euromwh.h.fcst.3mv.storm",
        storm_dashboard_primary_series="41376_native",
        storm_dashboard_naive_timezone=TIMEZONE,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    final_index = pd.date_range(
        "2025-08-10 22:00:00Z", periods=3, freq="h", tz="UTC",
        name="delivery_start_utc",
    )
    live_index = pd.date_range(
        "2025-08-11 22:00:00Z", periods=3, freq="h", tz="UTC",
        name="delivery_start_utc",
    )

    def frame(index: pd.DatetimeIndex, offset: float, *, actual: bool) -> pd.DataFrame:
        q50 = np.asarray([10.0, 20.0, 30.0]) + offset
        data: dict[str, Any] = {
            "residual_corrected__q10": q50 - 5.0,
            "residual_corrected__q50": q50,
            "residual_corrected__q90": q50 + 5.0,
        }
        if actual:
            data["actual"] = q50 + np.asarray([1.0, -1.0, 2.0])
        return pd.DataFrame(data, index=index)

    return frame(final_index, 0.0, actual=True), frame(
        live_index, 100.0, actual=False
    )


def _gate_files(
    root: Path,
    *,
    weight: float = 0.25,
    b1_passes: bool = True,
    b2_passes: bool = True,
) -> tuple[Path, Path]:
    b1 = _write_json(
        root / "gate_b1.json",
        {
            "protocol": {"phase": "b1", "final_loaded": False},
            "learned_mkonline_weight": weight,
            "B1": {"passes": b1_passes},
        },
    )
    b2 = _write_json(
        root / "gate_b2.json",
        {
            "protocol": {
                "phase": "b2",
                "final_loaded": False,
                "frozen_mkonline_weight": weight,
            },
            "B2": {"passes": b2_passes},
        },
    )
    return b1, b2


def _base_recipe(root: Path, *, mode: str, weight: float) -> dict[str, Any]:
    source_run = root / "autonomous_run"
    source_run.mkdir(exist_ok=True)
    recipe: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen_before_final_opening",
        "recipe_mode": mode,
        "zone": ZONE,
        "timezone": TIMEZONE,
        "source_autonomous_run": str(source_run),
        "source_autonomous_checksum_manifest_sha256": SOURCE_CHECKSUM,
        "final_target_used_for_weight_or_hyperparameters": False,
        "weights": {
            "autonomous": 1.0 - weight,
            "mkonline_primary": weight,
        },
    }
    if mode == "mkonline_blend":
        b1, b2 = _gate_files(root, weight=weight)
        recipe.update(
            {
                "external_expert": {
                    "series": "41550_native",
                    "storm_used_as_feature": False,
                    "interpolation_allowed": False,
                },
                "selection_protocol": {
                    "final_loaded": False,
                    "b1_gate_artifact": str(b1),
                    "b1_gate_artifact_sha256": _sha256(b1),
                    "b2_veto_artifact": str(b2),
                    "b2_veto_artifact_sha256": _sha256(b2),
                },
            }
        )
    elif mode == "autonomous_only":
        recipe["autonomous_validation"] = {
            "approved": True,
            "final_loaded": False,
            "reason": "The external expert did not pass the frozen B1/B2 protocol.",
        }
    return recipe


def _validate(root: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    return validate_recipe_manifest(
        recipe,
        zone=ZONE,
        timezone=TIMEZONE,
        source_run=root / "autonomous_run",
        source_checksum_sha256=SOURCE_CHECKSUM,
        project_root=root,
    )


def test_validate_and_build_autonomous_recipe_is_an_identity(tmp_path: Path) -> None:
    recipe = _validate(
        tmp_path, _base_recipe(tmp_path, mode="autonomous_only", weight=0.0)
    )
    source_backtest, source_forecast = _source_frames()

    backtest, forecast, audit = build_frozen_candidate(
        source_backtest,
        source_forecast,
        recipe,
        source_backtest.index,
        source_forecast.index,
    )

    for quantile in ("q10", "q50", "q90"):
        np.testing.assert_allclose(
            backtest[f"residual_corrected__{quantile}"],
            source_backtest[f"residual_corrected__{quantile}"],
        )
        np.testing.assert_allclose(
            forecast[quantile],
            source_forecast[f"residual_corrected__{quantile}"],
        )
    assert not any(column.startswith("mkonline_") for column in backtest)
    assert not any(column.startswith("mkonline_") for column in forecast)
    assert backtest["actual"].equals(source_backtest["actual"])
    assert audit["recipe_mode"] == "autonomous_only"
    assert audit["native_model"] == "residual_corrected"
    assert audit["weights"] == {
        "autonomous": 1.0,
        "mkonline_primary": 0.0,
    }
    assert audit["fallback_used"] is False


def test_build_mkonline_blend_applies_one_common_quantile_shift(
    tmp_path: Path,
) -> None:
    recipe = _validate(
        tmp_path, _base_recipe(tmp_path, mode="mkonline_blend", weight=0.25)
    )
    source_backtest, source_forecast = _source_frames()
    primary_final = pd.Series(
        [18.0, 12.0, 46.0], index=source_backtest.index, name="primary"
    )
    primary_live = pd.Series(
        [118.0, 112.0, 146.0], index=source_forecast.index, name="primary"
    )

    backtest, forecast, audit = build_frozen_candidate(
        source_backtest,
        source_forecast,
        recipe,
        source_backtest.index,
        source_forecast.index,
        primary_final=primary_final,
        primary_live=primary_live,
    )

    expected_final_q50 = 0.75 * np.asarray([10.0, 20.0, 30.0]) + 0.25 * np.asarray(
        [18.0, 12.0, 46.0]
    )
    expected_live_q50 = 0.75 * np.asarray([110.0, 120.0, 130.0]) + 0.25 * np.asarray(
        [118.0, 112.0, 146.0]
    )
    np.testing.assert_allclose(backtest["mkonline_blend__q50"], expected_final_q50)
    np.testing.assert_allclose(forecast["mkonline_blend__q50"], expected_live_q50)
    np.testing.assert_allclose(
        backtest["mkonline_blend__q90"] - backtest["mkonline_blend__q10"],
        source_backtest["residual_corrected__q90"]
        - source_backtest["residual_corrected__q10"],
    )
    assert bool(
        (
            backtest["mkonline_blend__q10"]
            <= backtest["mkonline_blend__q50"]
        ).all()
    )
    assert bool(
        (
            backtest["mkonline_blend__q50"]
            <= backtest["mkonline_blend__q90"]
        ).all()
    )
    np.testing.assert_allclose(backtest["mkonline_primary__q50"], primary_final)
    assert audit["recipe_mode"] == "mkonline_blend"
    assert audit["native_model"] == "mkonline_blend"
    assert audit["fallback_used"] is False


@pytest.mark.parametrize("missing", ["final", "live"])
def test_blend_requires_both_pit_primary_windows(
    tmp_path: Path, missing: str
) -> None:
    recipe = _validate(
        tmp_path, _base_recipe(tmp_path, mode="mkonline_blend", weight=0.25)
    )
    source_backtest, source_forecast = _source_frames()
    kwargs = {
        "primary_final": pd.Series(1.0, index=source_backtest.index),
        "primary_live": pd.Series(1.0, index=source_forecast.index),
    }
    kwargs[f"primary_{missing}"] = None

    with pytest.raises(BenchmarkContractError, match="primary|PIT"):
        build_frozen_candidate(
            source_backtest,
            source_forecast,
            recipe,
            source_backtest.index,
            source_forecast.index,
            **kwargs,
        )


def test_blend_never_falls_back_on_missing_or_nonfinite_primary(
    tmp_path: Path,
) -> None:
    recipe = _validate(
        tmp_path, _base_recipe(tmp_path, mode="mkonline_blend", weight=0.25)
    )
    source_backtest, source_forecast = _source_frames()
    primary_final = pd.Series(1.0, index=source_backtest.index)
    primary_live = pd.Series(1.0, index=source_forecast.index)
    primary_live.iloc[1] = np.nan

    with pytest.raises(BenchmarkContractError, match="finite|missing|primary"):
        build_frozen_candidate(
            source_backtest,
            source_forecast,
            recipe,
            source_backtest.index,
            source_forecast.index,
            primary_final=primary_final,
            primary_live=primary_live,
        )

    misaligned = pd.Series(
        1.0, index=source_forecast.index + pd.Timedelta(hours=1)
    )
    with pytest.raises(BenchmarkContractError, match="timeline|index|primary"):
        build_frozen_candidate(
            source_backtest,
            source_forecast,
            recipe,
            source_backtest.index,
            source_forecast.index,
            primary_final=primary_final,
            primary_live=misaligned,
        )


def test_autonomous_recipe_rejects_primary_or_external_expert(tmp_path: Path) -> None:
    raw = _base_recipe(tmp_path, mode="autonomous_only", weight=0.0)
    raw["external_expert"] = {
        "series": "41550_native",
        "storm_used_as_feature": False,
        "interpolation_allowed": False,
    }
    with pytest.raises(BenchmarkContractError, match="external|primary|autonomous"):
        _validate(tmp_path, raw)

    recipe = _validate(
        tmp_path, _base_recipe(tmp_path, mode="autonomous_only", weight=0.0)
    )
    source_backtest, source_forecast = _source_frames()
    with pytest.raises(BenchmarkContractError, match="primary|autonomous"):
        build_frozen_candidate(
            source_backtest,
            source_forecast,
            recipe,
            source_backtest.index,
            source_forecast.index,
            primary_final=pd.Series(1.0, index=source_backtest.index),
            primary_live=pd.Series(1.0, index=source_forecast.index),
        )


@pytest.mark.parametrize(
    ("mode", "weights"),
    [
        ("mkonline_blend", {"autonomous": 0.8, "mkonline_primary": 0.3}),
        ("mkonline_blend", {"autonomous": 1.1, "mkonline_primary": -0.1}),
        ("autonomous_only", {"autonomous": 0.9, "mkonline_primary": 0.1}),
    ],
)
def test_recipe_weights_are_strictly_convex_and_mode_consistent(
    tmp_path: Path, mode: str, weights: dict[str, float]
) -> None:
    recipe = _base_recipe(
        tmp_path,
        mode=mode,
        weight=float(weights["mkonline_primary"]),
    )
    recipe["weights"] = weights
    with pytest.raises(BenchmarkContractError, match="weight|sum|autonomous"):
        _validate(tmp_path, recipe)


@pytest.mark.parametrize("bad_mode", [None, "fallback", "storm_blend"])
def test_recipe_mode_is_explicit_and_closed(bad_mode: object, tmp_path: Path) -> None:
    recipe = _base_recipe(tmp_path, mode="autonomous_only", weight=0.0)
    if bad_mode is None:
        recipe.pop("recipe_mode")
    else:
        recipe["recipe_mode"] = bad_mode
    with pytest.raises(BenchmarkContractError, match="recipe_mode|mode"):
        _validate(tmp_path, recipe)


@pytest.mark.parametrize("gate", ["b1", "b2"])
def test_blend_requires_b1_and_b2_to_have_passed(
    tmp_path: Path, gate: str
) -> None:
    recipe = _base_recipe(tmp_path, mode="mkonline_blend", weight=0.25)
    selection = recipe["selection_protocol"]
    path_key = "b1_gate_artifact" if gate == "b1" else "b2_veto_artifact"
    hash_key = (
        "b1_gate_artifact_sha256"
        if gate == "b1"
        else "b2_veto_artifact_sha256"
    )
    gate_path = Path(selection[path_key])
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    payload[gate.upper()]["passes"] = False
    _write_json(gate_path, payload)
    selection[hash_key] = _sha256(gate_path)

    with pytest.raises(BenchmarkContractError, match=rf"(?i){gate}|pass"):
        _validate(tmp_path, recipe)


def test_blend_checks_gate_hash_and_frozen_weight(tmp_path: Path) -> None:
    recipe = _base_recipe(tmp_path, mode="mkonline_blend", weight=0.25)
    recipe["selection_protocol"]["b1_gate_artifact_sha256"] = "0" * 64
    with pytest.raises(BenchmarkContractError, match="hash|sha256|checksum"):
        _validate(tmp_path, recipe)

    recipe = _base_recipe(tmp_path, mode="mkonline_blend", weight=0.25)
    b2_path = Path(recipe["selection_protocol"]["b2_veto_artifact"])
    payload = json.loads(b2_path.read_text(encoding="utf-8"))
    payload["protocol"]["frozen_mkonline_weight"] = 0.30
    _write_json(b2_path, payload)
    recipe["selection_protocol"]["b2_veto_artifact_sha256"] = _sha256(b2_path)
    with pytest.raises(BenchmarkContractError, match="weight|frozen"):
        _validate(tmp_path, recipe)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"status": "draft"}, "status|frozen"),
        ({"final_target_used_for_weight_or_hyperparameters": True}, "final"),
        ({"zone": "FR"}, "zone"),
        ({"timezone": "Europe/Paris"}, "timezone"),
        ({"source_autonomous_checksum_manifest_sha256": "b" * 64}, "checksum|sha256"),
    ],
)
def test_manifest_identity_and_final_sealing_are_fail_closed(
    tmp_path: Path, mutation: dict[str, Any], message: str
) -> None:
    recipe = _base_recipe(tmp_path, mode="autonomous_only", weight=0.0)
    recipe.update(mutation)
    with pytest.raises(BenchmarkContractError, match=message):
        _validate(tmp_path, recipe)


def test_attach_native_storm_normalizes_exact_de_civil_timeline() -> None:
    index = pd.date_range(
        "2025-01-01 00:00:00Z", periods=4, freq="h", tz="UTC",
        name="delivery_start_utc",
    )
    statistics = pd.DataFrame(
        {
            "actual": [50.0, 51.0, 52.0, 53.0],
            "mkonline_blend__q50": [49.0, 50.0, 54.0, 52.0],
        },
        index=index,
    )
    civil_index = index.tz_convert(TIMEZONE).tz_localize(None)
    native = pd.Series([48.0, 52.0, 53.0, 55.0], index=civil_index)
    source = {
        "series": "power.price.de.euromwh.h.fcst.3mv.storm",
        "primary_series": "41376_native",
    }

    frame, audit, comparator = attach_native_storm_for_statistics(
        statistics,
        native,
        zone=ZONE,
        timezone=TIMEZONE,
        source=source,
    )

    np.testing.assert_allclose(
        frame["storm_dashboard_official__q50"], native.to_numpy(float)
    )
    assert comparator.values.index.equals(index)
    assert audit["zone"] == ZONE
    assert audit["storm_used_for_prediction"] is False
    assert comparator.audit["source"]["primary_series"] == "41376_native"


def test_native_storm_is_refused_for_es() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    statistics = pd.DataFrame({"actual": [1.0, 2.0]}, index=index)
    native = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(index).tz_localize(None))

    with pytest.raises(BenchmarkContractError, match="ES|verified|native"):
        attach_native_storm_for_statistics(
            statistics,
            native,
            zone="ES",
            timezone="Europe/Madrid",
        )


@pytest.mark.parametrize("failure", ["aware", "missing"])
def test_native_storm_requires_an_exact_timezone_naive_civil_curve(
    failure: str,
) -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    statistics = pd.DataFrame({"actual": [1.0, 2.0, 3.0]}, index=index)
    civil = index.tz_convert(TIMEZONE).tz_localize(None)
    native = pd.Series([3.0, 2.0, 1.0], index=civil)
    if failure == "aware":
        native.index = native.index.tz_localize(TIMEZONE)
    else:
        native = native.iloc[:-1]

    with pytest.raises(BenchmarkContractError, match="naive|timeline|coverage|missing"):
        attach_native_storm_for_statistics(
            statistics,
            native,
            zone=ZONE,
            timezone=TIMEZONE,
        )


def test_dependency_manifest_is_direct_mkonline_and_storm_free() -> None:
    payload = {
        "schema_version": 1,
        "zone": ZONE,
        "timezone": TIMEZONE,
        "terminal_series": "41550_native",
        "terminal_type": "primary",
        "terminal_formula": None,
        "storm_token_found": False,
        "dependency_gate_passed": True,
        "terminal_metadata": {
            "mercure:provider": "MKONLINE",
            "mercure:source": "WATTSIGHT",
        },
    }
    assert validate_dependency_manifest(payload, identity=_identity()) == payload
    for patch, match in (
        ({"terminal_series": "41551_native"}, "primary series"),
        ({"terminal_type": "formula"}, "primary series"),
        ({"storm_token_found": True}, "Storm-free"),
        ({"dependency_gate_passed": False}, "gate"),
    ):
        invalid = {**payload, **patch}
        with pytest.raises(BenchmarkContractError, match=match):
            validate_dependency_manifest(invalid, identity=_identity())


def test_es_monthly_dependency_cannot_be_promoted_as_validated_blend() -> None:
    identity = BenchmarkIdentity(
        zone="ES",
        timezone="Europe/Madrid",
        target_series="power.price.da.es.bzn.hourly.entsoe.utc.cdh.eurmwh",
        primary_series="58307_native",
        storm_dashboard_series=None,
        storm_dashboard_primary_series=None,
        storm_dashboard_naive_timezone=None,
    )
    dependency = {
        "schema_version": 1,
        "zone": "ES",
        "timezone": "Europe/Madrid",
        "status": "experimental_monthly_fallback",
        "terminal_series": "58307_native",
        "terminal_type": "primary",
        "terminal_formula": None,
        "storm_token_found": False,
        "dependency_gate_passed": True,
        "terminal_metadata": {
            "mercure:provider": "MKONLINE",
            "mercure:source": "WATTSIGHT",
        },
    }
    with pytest.raises(BenchmarkContractError, match="ES monthly|fallback"):
        validate_dependency_manifest(dependency, identity=identity)
