from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import run_topology_experiment as runner

from chronos2_hourly.reporting import (
    STORM_DASHBOARD_CONTRACT_ID,
    build_hourly_zone_result,
)

from run_topology_experiment import (
    DEFAULT_SCALE_GRID,
    DEFAULT_SPLIT_DAYS,
    FROZEN_MODEL_PARAMETERS,
    CandidateSeal,
    FormalStageData,
    RecipeSeal,
    TopologyExperimentError,
    _current_archives_after_candidate_seals,
    _display_stage_from_opened,
    _forecast_origins_for_delivery,
    _load_storm_after_candidate,
    _sha256,
    _storm_evaluation_metrics,
    atomic_experiment_publish,
    build_protocol_splits,
    evaluate_gate,
    load_comparator_after_seal,
    paired_segment_metrics,
    run_blend_protocol,
    run_development_protocol,
    run_formal_autonomous_protocol,
    seal_candidate_predictions,
)


def _annual_index() -> pd.DatetimeIndex:
    start = pd.Timestamp("2025-08-12", tz="Europe/Paris").tz_convert("UTC")
    stop = pd.Timestamp("2026-08-12", tz="Europe/Paris").tz_convert("UTC")
    return pd.date_range(
        start,
        stop,
        freq="h",
        inclusive="left",
        name="delivery_start_utc",
    )


def _splits():
    return build_protocol_splits(
        _annual_index(),
        timezone_name="Europe/Paris",
        split_days=DEFAULT_SPLIT_DAYS,
    )


def _base(index: pd.DatetimeIndex, q50: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"q10": q50 - 1.0, "q50": q50, "q90": q50 + 1.0},
        index=index,
    )


def _context(index: pd.DatetimeIndex, radius: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        0.0,
        index=index,
        columns=[f"feature_{number}" for number in range(22)],
    )
    frame.attrs["topology_context"] = {
        "target_zone": "FR",
        "radius": radius,
        "neighbours": [] if radius == 0 else ["DE", "BE", "ES"],
    }
    return frame


class _FakeCorrector:
    fit_lengths: list[int] = []

    def __init__(self, scale: float, *, raw_by_radius: dict[int, float]) -> None:
        self.scale = float(scale)
        self.raw_by_radius = raw_by_radius
        self.radius = 0

    def fit(self, X, y, base_predictions):
        self.radius = int(X.attrs["topology_context"]["radius"])
        type(self).fit_lengths.append(len(X))
        return self

    def predict_correction(self, X, base_predictions):
        raw = pd.Series(
            self.raw_by_radius[self.radius],
            index=X.index,
            name="raw",
        )
        applied = raw * self.scale
        applied.attrs["raw_correction"] = raw
        return applied

    def predict(self, X, base_predictions):
        shift = np.clip(
            self.raw_by_radius[self.radius] * self.scale,
            -10.0,
            10.0,
        )
        return base_predictions.loc[:, ["q10", "q50", "q90"]] + shift

    def hyperparameter_sha256(self):
        return hashlib.sha256(f"fake:{self.scale}".encode()).hexdigest()

    def audit_metadata(self):
        return {
            "fake": True,
            "scale": self.scale,
            "radius": self.radius,
            "fit_rows": self.fit_lengths[-1],
        }


def _preformal(splits, *, a_actual: float = 0.0, development_actual: float = 0.0):
    index = splits.indices["seed"].append(splits.indices["a"])
    index = index.append(splits.indices["development"])
    actual = pd.Series(0.0, index=index)
    actual.loc[splits.indices["a"]] = a_actual
    actual.loc[splits.indices["development"]] = development_actual
    base = _base(index, 1.0)
    contexts = {radius: _context(index, radius) for radius in (0, 1)}
    return index, actual, base, contexts


def _recipe_seal(tmp_path: Path, development) -> RecipeSeal:
    path = tmp_path / "topology_recipe_seal.json"
    path.write_text("{}", encoding="utf-8")
    return RecipeSeal(
        path=path,
        sha256=_sha256(path),
        selected_radius=development.selected_radius,
        selected_scale=development.selected_scale,
        config_sha256="a" * 64,
    )


def test_governed_split_is_dst_safe_and_has_new_formal_halves() -> None:
    splits = _splits()

    assert {name: len(days) for name, days in splits.days.items()} == {
        "seed": 120,
        "a": 65,
        "development": 60,
        "b1": 30,
        "b2": 30,
        "final": 60,
    }
    assert splits.all_index.equals(_annual_index())
    assert len(splits.indices["b1"]) in {719, 720, 721}
    assert len(splits.indices["b2"]) in {719, 720, 721}
    assert len(splits.indices["final"]) in {1439, 1440, 1441}


def test_scale_and_radius_are_selected_on_a_not_development() -> None:
    splits = _splits()
    _, actual, base, contexts = _preformal(
        splits,
        a_actual=0.0,
        development_actual=1.0,  # identity wins development, but cannot retune A
    )

    result = run_development_protocol(
        zone="FR",
        actual=actual,
        base_predictions=base,
        contexts=contexts,
        splits=splits,
        model_parameters=FROZEN_MODEL_PARAMETERS,
        scale_grid=DEFAULT_SCALE_GRID,
        bootstrap_samples=200,
        corrector_factory=lambda scale: _FakeCorrector(
            scale,
            raw_by_radius={0: -1.0, 1: -0.4},
        ),
    )

    assert result.selected_arm == "radius0_scale1"
    assert result.selected_radius == 0
    assert result.selected_scale == 1.0
    assert result.a_arm_metrics["radius0_scale1"]["mae"] == pytest.approx(0.0)
    assert result.a_arm_metrics["radius0_scale0"]["mae"] == pytest.approx(
        result.a_arm_metrics["identity"]["mae"]
    )
    assert result.a_arm_metrics["radius1_scale0"]["mae"] == pytest.approx(
        result.a_arm_metrics["identity"]["mae"]
    )
    assert result.development_metrics.gain_eur_mwh < 0.0


def test_forecast_origin_is_previous_local_day_0800_across_dst() -> None:
    index = pd.date_range(
        "2025-10-25T20:00:00Z",
        "2025-10-27T04:00:00Z",
        freq="h",
        name="delivery_start_utc",
    )

    origins = _forecast_origins_for_delivery(
        index,
        timezone_name="Europe/Paris",
    ).tz_convert("Europe/Paris")
    delivery = index.tz_convert("Europe/Paris")

    assert set(origins.hour) == {8}
    assert all(
        origin.date() == delivery_hour.date() - pd.Timedelta(days=1)
        for origin, delivery_hour in zip(origins, delivery)
    )


def test_identity_winner_spends_no_formal_gate_and_cannot_promote(tmp_path: Path) -> None:
    splits = _splits()
    _, actual, base, contexts = _preformal(
        splits,
        a_actual=1.0,  # base q50 is exact; every non-zero correction is worse
        development_actual=1.0,
    )
    development = run_development_protocol(
        zone="FR",
        actual=actual,
        base_predictions=base,
        contexts=contexts,
        splits=splits,
        model_parameters=FROZEN_MODEL_PARAMETERS,
        bootstrap_samples=100,
        corrector_factory=lambda scale: _FakeCorrector(
            scale,
            raw_by_radius={0: -1.0, 1: -0.5},
        ),
    )
    calls: list[str] = []

    result = run_formal_autonomous_protocol(
        development=development,
        recipe_seal=_recipe_seal(tmp_path, development),
        stage_loader=lambda stage: calls.append(stage),  # type: ignore[arg-type,return-value]
        splits=splits,
        model_parameters=FROZEN_MODEL_PARAMETERS,
        bootstrap_samples=100,
        corrector_factory=lambda scale: _FakeCorrector(
            scale,
            raw_by_radius={0: -1.0, 1: -0.5},
        ),
    )

    assert development.selected_arm == "identity"
    assert calls == []
    assert result.gates == {}
    assert result.opened_stages == ("a", "development")
    assert result.promoted is False


def test_recipe_seal_precedes_b1_and_development_is_in_post_freeze_fit(
    tmp_path: Path,
) -> None:
    _FakeCorrector.fit_lengths = []
    splits = _splits()
    _, actual, base, contexts = _preformal(splits)
    development = run_development_protocol(
        zone="FR",
        actual=actual,
        base_predictions=base,
        contexts=contexts,
        splits=splits,
        model_parameters=FROZEN_MODEL_PARAMETERS,
        bootstrap_samples=100,
        corrector_factory=lambda scale: _FakeCorrector(
            scale,
            raw_by_radius={0: -1.0, 1: -0.5},
        ),
    )
    seal = _recipe_seal(tmp_path, development)
    events: list[str] = []

    def load(stage: str) -> FormalStageData:
        assert seal.path.is_file()
        assert _sha256(seal.path) == seal.sha256
        events.append(stage)
        index = splits.indices[stage]
        return FormalStageData(
            actual=pd.Series(0.0, index=index),
            base_predictions=_base(index, 1.0),
            context=_context(index, development.selected_radius),
        )

    result = run_formal_autonomous_protocol(
        development=development,
        recipe_seal=seal,
        stage_loader=load,
        splits=splits,
        model_parameters=FROZEN_MODEL_PARAMETERS,
        bootstrap_samples=200,
        corrector_factory=lambda scale: _FakeCorrector(
            scale,
            raw_by_radius={0: -1.0, 1: -0.5},
        ),
    )

    assert events == ["b1", "b2", "final"]
    expected_preformal_rows = sum(
        len(splits.indices[name]) for name in ("seed", "a", "development")
    )
    formal_fit_lengths = [
        audit["fit_rows"]
        for name, audit in result.model_audits.items()
        if name.startswith("selected_refit_to_b") or name.endswith("_final")
    ]
    assert formal_fit_lengths[0] == expected_preformal_rows
    assert formal_fit_lengths[1] == expected_preformal_rows + len(splits.indices["b1"])
    assert formal_fit_lengths[2] == (
        expected_preformal_rows
        + len(splits.indices["b1"])
        + len(splits.indices["b2"])
    )
    assert "development" not in result.gates
    assert result.promoted is True


def test_formal_gate_requires_gain_halves_and_paired_bootstrap() -> None:
    index = pd.date_range("2026-01-01", periods=30 * 24, freq="h", tz="UTC")
    actual = pd.Series(0.0, index=index)
    baseline = pd.Series(1.0, index=index)
    candidate = pd.Series(0.8, index=index)
    metrics = paired_segment_metrics(
        actual,
        candidate,
        baseline,
        timezone_name="UTC",
        bootstrap_samples=500,
    )

    gate = evaluate_gate(metrics, minimum_gain=0.05)

    assert gate.passes is True
    broken = replace(metrics, second_half_gain_eur_mwh=-0.01)
    failed = evaluate_gate(broken, minimum_gain=0.05)
    assert failed.passes is False
    assert "second_half_positive" not in [
        name for name, passed in failed.checks.items() if passed
    ]


def test_non_blend_zone_is_rejected_before_any_loader() -> None:
    calls: list[str] = []
    # The zone guard is the first line that can decide eligibility; none of the
    # remaining objects need to be valid because loaders must stay untouched.
    result = run_blend_protocol(
        zone="DE",
        autonomous=object(),  # type: ignore[arg-type]
        actual=pd.Series(dtype=float),
        mkonline_q50_loader=lambda: calls.append("mk"),  # type: ignore[return-value]
        production_autonomous_loader=lambda: calls.append("prod"),  # type: ignore[return-value]
        splits=object(),  # type: ignore[arg-type]
        previous_weight_mkonline=0.5,
    )
    assert result is None
    assert calls == []


def test_storm_loader_runs_only_after_candidate_file_and_hash(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "delivery_start_utc": ["2026-01-01T00:00:00Z"],
            "topology__q50": [42.0],
        }
    )
    events: list[str] = []
    seal = seal_candidate_predictions(
        tmp_path,
        frame,
        zone="FR",
        experiment_id="pricefm_topology_v1",
    )

    def loader():
        assert seal.prediction_path.is_file()
        assert _sha256(seal.prediction_path) == seal.prediction_sha256
        events.append("storm")
        return pd.Series([40.0])

    result = load_comparator_after_seal(seal, loader)

    assert events == ["storm"]
    assert isinstance(result, pd.Series)
    seal.prediction_path.write_bytes(b"changed")
    with pytest.raises(TopologyExperimentError, match="change"):
        load_comparator_after_seal(seal, loader)


def test_storm_metrics_are_paired_daily_and_never_a_gate_input() -> None:
    index = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "delivery_start_utc": index,
            "stage": ["final"] * len(index),
            "actual": np.zeros(len(index)),
            "topology_autonomous__q50": np.ones(len(index)),
            "residual_corrected__q50": np.full(len(index), 2.0),
        }
    )
    storm = pd.Series(3.0, index=index, name="storm_dashboard_official__q50")

    metrics = _storm_evaluation_metrics(
        frame,
        storm,
        stage="final",
        candidate_model="topology_autonomous",
        baseline_model="residual_corrected",
        timezone_name="UTC",
    )

    assert metrics["candidate_mae"] == pytest.approx(1.0)
    assert metrics["baseline_mae"] == pytest.approx(2.0)
    assert metrics["storm_mae"] == pytest.approx(3.0)
    assert metrics["candidate_vs_storm_daily_win_rate"] == pytest.approx(1.0)
    assert metrics["n_paired_days"] == 2
    assert metrics["used_for_gate"] is False
    assert metrics["used_for_promotion"] is False


def test_native_storm_loader_accepts_only_the_audited_autumn_dst_gap(
    tmp_path: Path,
) -> None:
    index = pd.DatetimeIndex(
        ["2025-10-26T00:00:00Z", "2025-10-26T01:00:00Z"],
        name="delivery_start_utc",
    )
    archive = tmp_path / "archive"
    inputs = archive / "inputs"
    inputs.mkdir(parents=True)
    storm_path = inputs / "storm_dashboard_official_statistics.parquet"
    pd.DataFrame(
        {
            "delivery_start_utc": index,
            "storm_dashboard_official__q50": [np.nan, 40.0],
        }
    ).to_parquet(storm_path, index=False)
    (archive / "statistics_history_audit.json").write_text(
        json.dumps(
            {
                "storm_primary_report_benchmark": (
                    "storm_dashboard_official__q50"
                ),
                "storm_dashboard": {
                    "column": "storm_dashboard_official__q50",
                    "normalized_artifact_sha256": _sha256(storm_path),
                    "dst": {
                        "interpolation": False,
                        "strict_08_fallback": False,
                        "native_allowed_missing_utc": [index[0].isoformat()],
                        "native_actual_missing_matches_allowed": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    seal = seal_candidate_predictions(
        tmp_path / "sealed",
        pd.DataFrame({"delivery_start_utc": index}),
        zone="FR",
        experiment_id="pricefm_topology_v1",
    )

    storm, audit = _load_storm_after_candidate(
        seal=seal,
        zone="FR",
        archive=archive,
        expected_index=index,
    )

    assert storm is not None
    assert storm.isna().sum() == 1
    assert audit["native_dashboard"] is True
    assert audit["native_allowed_missing_hours"] == 1
    assert audit["native_actual_missing_matches_allowed"] is True


def test_current_archive_loader_explicitly_requests_autonomous_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = tmp_path / "candidate.csv.gz"
    manifest = tmp_path / "candidate.json"
    prediction.write_bytes(b"candidate")
    manifest.write_bytes(b"manifest")
    seal = CandidateSeal(
        prediction_path=prediction,
        manifest_path=manifest,
        prediction_sha256=_sha256(prediction),
        manifest_sha256=_sha256(manifest),
    )
    observed: dict[str, object] = {}

    def fake_statuses(_registry, *, zones):
        observed["zones"] = tuple(zones)
        return "audited-statuses"

    def fake_comparison(statuses, **kwargs):
        observed["statuses"] = statuses
        observed.update(kwargs)
        return "autonomous-current"

    monkeypatch.setattr(
        "chronos2_hourly.app_service.inspect_zone_statuses",
        fake_statuses,
    )
    monkeypatch.setattr(
        "chronos2_hourly.app_service.load_latest_forecast_comparison",
        fake_comparison,
    )
    config = SimpleNamespace(
        zones=("FR",),
        contract=SimpleNamespace(project_root=tmp_path),
    )

    result = _current_archives_after_candidate_seals(  # type: ignore[arg-type]
        config,
        {"FR": seal},
    )

    assert result == "autonomous-current"
    assert observed["variant"] == "autonomous"
    assert observed["allow_mixed_delivery_days"] is False


def test_atomic_publish_is_scoped_and_cleans_failed_staging(tmp_path: Path) -> None:
    project = tmp_path / "project"
    output = project / "runs" / "experiments" / "test_topology"

    published = atomic_experiment_publish(
        output,
        project_root=project,
        writer=lambda staging: (staging / "ok.txt").write_text(
            "ok", encoding="utf-8"
        ),
    )

    assert (published / "ok.txt").read_text(encoding="utf-8") == "ok"
    with pytest.raises(TopologyExperimentError, match="sous-dossier"):
        atomic_experiment_publish(
            project / "runs" / "live" / "bad",
            project_root=project,
            writer=lambda staging: None,
        )


@pytest.mark.parametrize(
    ("opened_stages", "expected"),
    [
        pytest.param(("a", "development", "b1"), "b1", id="autonomous-b1"),
        pytest.param(
            ("a", "development", "b1", "b2", "final"),
            "final",
            id="autonomous-final",
        ),
        pytest.param(
            ("a", "b1", "b2", "final"),
            "final",
            id="blend-final",
        ),
    ],
)
def test_display_stage_is_the_last_sequentially_opened_stage(
    opened_stages: tuple[str, ...],
    expected: str,
) -> None:
    assert _display_stage_from_opened(opened_stages) == expected


def test_current_context_uses_future_covariates_beyond_aligned_history(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "live_archive"
    inputs = archive / "inputs"
    inputs.mkdir(parents=True)
    forecast_index = pd.date_range(
        "2026-08-22T00:00:00Z",
        periods=24,
        freq="h",
        name="delivery_start_utc",
    )
    extended = pd.date_range(
        forecast_index[0] - pd.Timedelta(hours=24),
        forecast_index[-1],
        freq="h",
        name="delivery_start_utc",
    )
    history = extended[:24]
    residual_columns = {
        zone: f"{zone.lower()}_residual_load_fcst"
        for zone in runner.SUPPORTED_ZONES
    }

    # This mirrors the real live contract: aligned inputs stop one hour before
    # D+1, while model covariates contain the forecast horizon.
    aligned = pd.DataFrame({"timestamp": history})
    future = pd.DataFrame({"timestamp": extended})
    for offset, column in enumerate(residual_columns.values(), start=1):
        aligned[column] = np.arange(len(history), dtype=float) + offset
        future[column] = np.arange(len(extended), dtype=float) + offset
    aligned.to_csv(
        inputs / "aligned_inputs.csv.gz",
        index=False,
        compression="gzip",
    )
    future.to_csv(
        inputs / "model_covariates_with_future.csv.gz",
        index=False,
        compression="gzip",
    )

    sources: dict[str, SimpleNamespace] = {}
    for offset, zone in enumerate(runner.SUPPORTED_ZONES, start=1):
        target_cache = tmp_path / f"target_{zone.lower()}.csv.gz"
        pd.DataFrame(
            {
                "timestamp": history,
                "value": np.arange(len(history), dtype=float) + 100.0 + offset,
            }
        ).to_csv(target_cache, index=False, compression="gzip")
        sources[zone] = SimpleNamespace(
            target_cache=target_cache,
            timezone={
                "FR": "Europe/Paris",
                "DE": "Europe/Berlin",
                "BE": "Europe/Brussels",
                "NL": "Europe/Amsterdam",
                "ES": "Europe/Madrid",
            }[zone],
        )
    contract = SimpleNamespace(
        residual_load_columns=residual_columns,
        sources=sources,
    )

    context = runner._load_current_context(  # type: ignore[arg-type]
        contract=contract,
        zone="FR",
        archive=archive,
        forecast_index=forecast_index,
        radius=0,
    )

    assert context.index.equals(forecast_index)
    assert len(context) == 24
    assert np.allclose(
        context["topology__residual_load__local"].to_numpy(dtype=float),
        future.loc[24:, residual_columns["FR"]].to_numpy(dtype=float),
    )
    assert np.allclose(
        context["topology__price_da_lag24h__local"].to_numpy(dtype=float),
        np.arange(len(history), dtype=float) + 101.0,
    )


def test_report_keeps_rows_when_native_storm_has_a_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporting = tmp_path / "current_archive"
    inputs = reporting / "inputs"
    inputs.mkdir(parents=True)
    index = pd.DatetimeIndex(
        ["2025-10-26T00:00:00Z", "2025-10-26T01:00:00Z"]
    )
    forecast_index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    pd.DataFrame(
        {"timestamp": index, "target": [40.0, 41.0], "known": [1.0, 1.0]}
    ).to_csv(inputs / "aligned_inputs.csv.gz", index=False, compression="gzip")
    pd.DataFrame(
        {
            "timestamp": index.append(forecast_index),
            "known": [1.0] * 4,
        }
    ).to_csv(
        inputs / "model_covariates_with_future.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        {"alias": ["known"], "coverage": [1.0], "coverage_after_fill": [1.0]}
    ).to_csv(inputs / "input_coverage.csv", index=False)
    pd.DataFrame(
        {"alias": ["known"], "known_future": [True]}
    ).to_csv(inputs / "input_manifest.csv", index=False)
    candidate = pd.DataFrame(
        {
            "delivery_start_utc": index,
            "stage": ["b1", "b1"],
            "actual": [40.0, 41.0],
            "residual_corrected__q10": [38.0, 39.0],
            "residual_corrected__q50": [40.0, 41.0],
            "residual_corrected__q90": [42.0, 43.0],
            "topology_autonomous__q10": [38.0, 39.0],
            "topology_autonomous__q50": [40.0, 41.0],
            "topology_autonomous__q90": [42.0, 43.0],
        }
    )
    storm = pd.Series(
        [np.nan, 39.5],
        index=index,
        name="storm_dashboard_official__q50",
    )

    def fake_report(run_dir, **_kwargs):
        path = Path(run_dir) / "reports" / "topology_fr_autonomous.html"
        path.parent.mkdir(parents=True)
        path.write_text("<html></html>", encoding="utf-8")
        return path

    monkeypatch.setattr(runner, "write_topology_html_report", fake_report)
    run_dir = tmp_path / "autonomous"
    runner._write_variant_run(
        directory=run_dir,
        zone="FR",
        variant="autonomous",
        candidate_frame=candidate,
        current_forecast=pd.DataFrame({"delivery_start_utc": index}),
        evaluation={
            "timezone": "Europe/Paris",
            "config_sha256": "a" * 64,
            "protocol_frozen_at_utc": "2026-08-21T00:00:00+00:00",
            "storm_evaluation": {
                "native_allowed_missing_utc": [index[0].isoformat()],
            },
            "variants": {
                "autonomous": {
                    "unopened_holdouts_spared": ["b2", "final"],
                }
            },
        },
        source_directory=tmp_path / "sealed_source",
        reporting_inputs_directory=reporting,
        published_directory=tmp_path / "published" / "autonomous",
        display_stage="b1",
        storm=storm,
        project_root=tmp_path,
    )

    written = pd.read_csv(run_dir / "backtest_hourly_oof.csv.gz")
    assert len(written) == 2
    assert written["storm_dashboard_official__q50"].notna().sum() == 1
    checksum_payload = (run_dir / "artifact_checksums.json").read_text(
        encoding="utf-8"
    )
    assert "reports/topology_fr_autonomous.html" in checksum_payload

    statistics = pd.read_csv(run_dir / "statistics_history_hourly.csv.gz")
    assert len(statistics) == 2
    statistics_audit = (run_dir / "statistics_history_audit.json").read_text(
        encoding="utf-8"
    )
    assert "holdout decisionnel ouvert: b1" in statistics_audit
    assert "n'est pas l'historique live Statistics" in statistics_audit

    # Replace the intentionally minimal forecast with the real quantile schema,
    # then exercise the standard reporting loader (no report-writer mock).
    pd.DataFrame(
        {
            "delivery_start_utc": forecast_index,
            "residual_corrected__q10": [38.0, 39.0],
            "residual_corrected__q50": [40.0, 41.0],
            "residual_corrected__q90": [42.0, 43.0],
            "topology_autonomous__q10": [38.0, 39.0],
            "topology_autonomous__q50": [40.0, 41.0],
            "topology_autonomous__q90": [42.0, 43.0],
        }
    ).to_csv(run_dir / "forecast_hourly_fr.csv", index=False)
    result = build_hourly_zone_result(
        run_dir,
        native_model="topology_autonomous",
        baseline_model="residual_corrected",
        zone="FR",
        timezone="Europe/Paris",
    )
    assert len(result.statistics_benchmark) == 1
    assert result.statistics_benchmark_contract["id"] == (
        STORM_DASHBOARD_CONTRACT_ID
    )


def test_main_audit_allows_an_already_published_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_load(
        path,
        *,
        zones,
        allow_existing_output=False,
        verify_hashes=True,
    ):
        observed["path"] = path
        observed["zones"] = tuple(zones)
        observed["allow_existing_output"] = allow_existing_output
        observed["verify_hashes"] = verify_hashes
        return SimpleNamespace()

    monkeypatch.setattr(runner, "load_experiment_config", fake_load)
    monkeypatch.setattr(
        runner,
        "audit_experiment_sources",
        lambda _config: {
            "FR": {
                "pit_coverage": {"FR": 1.0},
                "forecast_origin_violations": 0,
                "price_lag_masked_same_day_hours": 0,
            }
        },
    )

    assert runner.main(["--config", "published.yaml", "--zones", "FR", "--audit-only"]) == 0
    assert observed == {
        "path": "published.yaml",
        "zones": ("FR",),
        "allow_existing_output": True,
        "verify_hashes": True,
    }
    assert "aucun fichier d'experience n'a ete cree" in capsys.readouterr().out


def test_daily_mode_policy_includes_production_and_refuses_non_blend_zone() -> None:
    assert runner._daily_variants_for_zone("production", "FR") == (
        "mkonline_blend",
    )
    assert runner._daily_variants_for_zone("production", "BE") == (
        "autonomous",
    )
    assert runner._daily_variants_for_zone("autonomous", "NL") == (
        "autonomous",
    )
    assert runner._daily_variants_for_zone("both", "FR") == (
        "autonomous",
        "mkonline_blend",
    )
    assert runner._daily_variants_for_zone("both", "DE") == ("autonomous",)
    with pytest.raises(TopologyExperimentError, match="reserve a FR/NL"):
        runner._daily_variants_for_zone("blend", "BE")


def test_daily_autonomous_policy_is_identity_except_for_be() -> None:
    index = pd.date_range("2026-08-22", periods=2, freq="h", tz="UTC")
    base = _base(index, q50=40.0)

    class ModelMustNotRun:
        def predict(self, *_args, **_kwargs):
            raise AssertionError("identity zones must not call the model")

    for zone in ("FR", "DE", "NL", "ES"):
        actual = runner._apply_daily_autonomous_policy(
            zone=zone,
            base=base,
            model=ModelMustNotRun(),  # type: ignore[arg-type]
        )
        assert np.array_equal(actual.to_numpy(), base.to_numpy())

    class BeModel:
        calls = 0

        def predict(self, _context, current_base):
            self.calls += 1
            return current_base + 2.0

    model = BeModel()
    corrected = runner._apply_daily_autonomous_policy(
        zone="BE",
        base=base,
        model=model,  # type: ignore[arg-type]
        context=pd.DataFrame(index=index),
    )
    assert model.calls == 1
    assert np.array_equal(corrected.to_numpy(), base.to_numpy() + 2.0)


def test_daily_blend_forecast_is_exact_passthrough(tmp_path: Path) -> None:
    index = pd.date_range("2026-08-22", periods=2, freq="h", tz="UTC")
    origins = pd.DatetimeIndex(
        [pd.Timestamp("2026-08-21T06:00:00Z")] * 2,
        name="forecast_origin_utc",
    )
    blend = _base(index, q50=50.0)
    forecast = runner._daily_forecast_frame(
        index=index,
        origins=origins,
        baseline=blend,
        candidate=blend.copy(),
        variant="mkonline_blend",
    )
    for quantile in ("q10", "q50", "q90"):
        assert np.array_equal(
            forecast[f"mkonline_blend__{quantile}"].to_numpy(),
            forecast[f"topology_mkonline_blend__{quantile}"].to_numpy(),
        )

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    source_report = archive_dir / "current_live_report.html"
    source_report.write_text("<html>production blend</html>", encoding="utf-8")
    forecast_source = archive_dir / "forecast_hourly_fr.csv"
    forecast_source.write_text("sealed", encoding="utf-8")
    checksum_source = archive_dir / "artifact_checksums.json"
    checksum_source.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": source_report.name,
                        "sha256": _sha256(source_report),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    archive = runner.DailyArchive(
        zone="FR",
        timezone="Europe/Paris",
        delivery_day="2026-08-22",
        directory=archive_dir,
        forecast_path=forecast_source,
        artifact_manifest_sha256=_sha256(checksum_source),
    )
    output = tmp_path / "daily_blend"
    published = tmp_path / "published" / "fr" / "mkonline_blend"
    report = runner._write_daily_passthrough_blend(
        directory=output,
        published_directory=published,
        zone="FR",
        forecast=forecast,
        archive=archive,
        application={"fixed_calibration": True},
    )
    written = pd.read_csv(output / "forecast_hourly_fr.csv")
    assert np.array_equal(
        written["mkonline_blend__q50"].to_numpy(),
        written["topology_mkonline_blend__q50"].to_numpy(),
    )
    assert report.read_bytes() == source_report.read_bytes()
    proof = json.loads((output / "daily_application.json").read_text("utf-8"))
    assert proof["weights_recomputed"] is False
    assert proof["candidate_equals_production_blend"] is True
    runner._validate_artifact_checksums(output)
    checksum_manifest = json.loads(
        (output / "artifact_checksums.json").read_text("utf-8")
    )
    assert Path(checksum_manifest["output_directory"]) == published


def test_daily_archive_members_are_rehashed_immediately_before_publish(
    tmp_path: Path,
) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    forecast = archive_dir / "forecast_hourly_fr.csv"
    forecast.write_text("original", encoding="utf-8")
    manifest = archive_dir / "artifact_checksums.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": forecast.name,
                        "sha256": _sha256(forecast),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    archive = runner.DailyArchive(
        zone="FR",
        timezone="Europe/Paris",
        delivery_day="2026-08-22",
        directory=archive_dir,
        forecast_path=forecast,
        artifact_manifest_sha256=_sha256(manifest),
    )
    expected = _sha256(forecast)
    runner._revalidate_consumed_archive_members(
        {"FR": archive},
        {forecast: expected},
    )
    forecast.write_text("mutated after read", encoding="utf-8")
    with pytest.raises(TopologyExperimentError, match="SHA archive divergent"):
        runner._revalidate_consumed_archive_members(
            {"FR": archive},
            {forecast: expected},
        )


def test_calibration_trust_anchor_fails_before_artifact_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    calibration = project / "runs" / "experiments" / "calibration"
    calibration.mkdir(parents=True)
    (calibration / "artifact_checksums.json").write_text("{}", encoding="utf-8")
    config = SimpleNamespace(contract=SimpleNamespace(project_root=project))
    monkeypatch.setattr(
        runner,
        "_validate_artifact_checksums",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("must fail on pinned manifest before artifact reads")
        ),
    )
    with pytest.raises(TopologyExperimentError, match="Trust anchor calibration"):
        runner.audit_published_calibration(  # type: ignore[arg-type]
            config,
            calibration,
            expected_manifest_sha256="0" * 64,
        )


def test_operational_bundle_trust_anchor_fails_before_joblib_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    bundle = project / "runs" / "experiments" / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "artifact_checksums.json").write_text("{}", encoding="utf-8")
    config = SimpleNamespace(contract=SimpleNamespace(project_root=project))
    monkeypatch.setattr(
        runner.joblib,
        "load",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("pickle must not be opened before trust anchor")
        ),
    )
    with pytest.raises(TopologyExperimentError, match="Trust anchor du bundle"):
        runner.load_operational_bundle(  # type: ignore[arg-type]
            config,
            bundle,
            expected_manifest_sha256="0" * 64,
        )


def test_exact_daily_archive_loader_uses_requested_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    archive = project / "runs" / "live" / "fr_day_ahead_2026-08-22"
    archive.mkdir(parents=True)
    (archive / "forecast_hourly_fr.csv").write_text("sealed", encoding="utf-8")
    (archive / "artifact_checksums.json").write_text("{}", encoding="utf-8")
    observed: dict[str, object] = {}
    status = SimpleNamespace(
        code="FR",
        timezone="Europe/Paris",
        launchable=True,
    )

    monkeypatch.setattr(
        "chronos2_hourly.app_service.inspect_zone_statuses",
        lambda _registry, *, zones: [status],
    )

    def fake_validate(_status, *, project_root, delivery_day):
        observed["project_root"] = Path(project_root)
        observed["delivery_day"] = delivery_day
        return archive

    monkeypatch.setattr(
        "chronos2_hourly.app_service.validate_existing_forecast_archive",
        fake_validate,
    )
    result = runner._load_exact_daily_archives(
        project_root=project,
        zones=("FR",),
        delivery_day="2026-08-22",
    )
    assert result["FR"].directory == archive
    assert observed["delivery_day"] == "2026-08-22"


@pytest.mark.parametrize(
    ("delivery_day", "expected_hours"),
    [("2026-03-29", 23), ("2026-10-25", 25)],
)
def test_daily_archive_quantiles_preserve_dst_day_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery_day: str,
    expected_hours: int,
) -> None:
    index = runner.local_delivery_day_index(
        delivery_day,
        timezone="Europe/Paris",
    )
    forecast_path = tmp_path / "forecast_hourly_fr.csv"
    pd.DataFrame(
        {
            "forecast_origin_utc": [index[0] - pd.Timedelta(hours=18)] * len(index)
        }
    ).to_csv(forecast_path, index=False)
    (tmp_path / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": forecast_path.name,
                        "sha256": _sha256(forecast_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    archive = runner.DailyArchive(
        zone="FR",
        timezone="Europe/Paris",
        delivery_day=delivery_day,
        directory=tmp_path,
        forecast_path=forecast_path,
        artifact_manifest_sha256="a" * 64,
    )
    frame = pd.DataFrame(
        {
            "timestamp": index.tz_convert("Europe/Paris"),
            "P10": np.arange(len(index), dtype=float),
            "P50": np.arange(len(index), dtype=float) + 1.0,
            "P90": np.arange(len(index), dtype=float) + 2.0,
        }
    )
    monkeypatch.setattr(
        "chronos2_hourly.app_service.load_forecast_curve",
        lambda *_args, **_kwargs: SimpleNamespace(frame=frame),
    )
    quantiles, _origins = runner._read_daily_archive_quantiles(
        archive,
        variant="autonomous",
    )
    assert len(quantiles) == expected_hours


def test_daily_existing_output_revalidates_bundle_metadata_and_live_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    output_root = project / "runs" / "experiments" / "daily"
    destination = output_root / "2026-08-22" / "production"
    destination.mkdir(parents=True)
    bundle_sha = "a" * 64
    archive_dir = project / "runs" / "live" / "fr"
    archive_dir.mkdir(parents=True)
    forecast = archive_dir / "forecast_hourly_fr.csv"
    forecast.write_text("sealed forecast", encoding="utf-8")
    archive_manifest = archive_dir / "artifact_checksums.json"
    archive_manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {"path": forecast.name, "sha256": _sha256(forecast)}
                ]
            }
        ),
        encoding="utf-8",
    )
    archive = runner.DailyArchive(
        zone="FR",
        timezone="Europe/Paris",
        delivery_day="2026-08-22",
        directory=archive_dir,
        forecast_path=forecast,
        artifact_manifest_sha256=_sha256(archive_manifest),
    )
    runner._write_json(
        destination / "daily_manifest.json",
        {
            "delivery_day": "2026-08-22",
            "mode": "production",
            "zones": ["FR"],
            "operational_bundle_artifact_manifest_sha256": bundle_sha,
            "source_archives": {
                "FR": {
                    "directory": str(archive_dir),
                    "artifact_manifest_sha256": _sha256(archive_manifest),
                    "forecast_sha256": _sha256(forecast),
                }
            },
            "complete": True,
        },
    )
    (destination / "index.html").write_text("complete", encoding="utf-8")
    runner._write_artifact_checksums(destination)
    bundle_loads: list[bool] = []

    def fake_bundle_load(*_args, **kwargs):
        bundle_loads.append(bool(kwargs["load_model"]))
        return SimpleNamespace()

    monkeypatch.setattr(runner, "load_operational_bundle", fake_bundle_load)
    monkeypatch.setattr(
        runner,
        "_load_exact_daily_archives",
        lambda **_kwargs: {"FR": archive},
    )
    config = SimpleNamespace(contract=SimpleNamespace(project_root=project))
    result = runner.apply_daily_topology(  # type: ignore[arg-type]
        config,
        operational_dir=project / "runs" / "experiments" / "bundle",
        operational_manifest_sha256=bundle_sha,
        delivery_day="2026-08-22",
        zones=("FR",),
        mode="production",
        output_root=output_root,
    )
    assert result == destination
    assert bundle_loads == [False]

    forecast.write_text("source changed", encoding="utf-8")
    with pytest.raises(TopologyExperimentError, match="SHA archive divergent"):
        runner.apply_daily_topology(  # type: ignore[arg-type]
            config,
            operational_dir=project / "runs" / "experiments" / "bundle",
            operational_manifest_sha256=bundle_sha,
            delivery_day="2026-08-22",
            zones=("FR",),
            mode="production",
            output_root=output_root,
        )
    forecast.write_text("sealed forecast", encoding="utf-8")

    (destination / "index.html").write_text("tampered", encoding="utf-8")
    with pytest.raises(TopologyExperimentError, match="Artefact modifie"):
        runner.apply_daily_topology(  # type: ignore[arg-type]
            config,
            operational_dir=project / "runs" / "experiments" / "bundle",
            operational_manifest_sha256=bundle_sha,
            delivery_day="2026-08-22",
            zones=("FR",),
            mode="production",
            output_root=output_root,
        )


def test_apply_blend_rejects_non_fr_nl_before_any_bundle_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    monkeypatch.setattr(
        runner,
        "load_operational_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid blend selection must fail before I/O")
        ),
    )
    config = SimpleNamespace(contract=SimpleNamespace(project_root=project))
    with pytest.raises(TopologyExperimentError, match="reserve a FR/NL"):
        runner.apply_daily_topology(  # type: ignore[arg-type]
            config,
            operational_dir=project / "runs" / "experiments" / "bundle",
            operational_manifest_sha256="a" * 64,
            delivery_day="2026-08-22",
            zones=("BE",),
            mode="blend",
            output_root=project / "runs" / "experiments" / "daily",
        )


def test_apply_blend_uses_metadata_only_bundle_without_be_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    bundle_dir = project / "runs" / "experiments" / "bundle"
    bundle_dir.mkdir(parents=True)
    bundle_manifest = bundle_dir / "artifact_checksums.json"
    bundle_manifest.write_text("sealed", encoding="utf-8")
    bundle_sha = _sha256(bundle_manifest)
    archive_dir = project / "runs" / "live" / "fr"
    archive_dir.mkdir(parents=True)
    forecast_path = archive_dir / "forecast_hourly_fr.csv"
    forecast_path.write_text("forecast", encoding="utf-8")
    report_path = archive_dir / "live.html"
    report_path.write_text("report", encoding="utf-8")
    live_manifest = archive_dir / "artifact_checksums.json"
    live_manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {"path": forecast_path.name, "sha256": _sha256(forecast_path)},
                    {"path": report_path.name, "sha256": _sha256(report_path)},
                ]
            }
        ),
        encoding="utf-8",
    )
    archive = runner.DailyArchive(
        zone="FR",
        timezone="Europe/Paris",
        delivery_day="2026-08-22",
        directory=archive_dir,
        forecast_path=forecast_path,
        artifact_manifest_sha256=_sha256(live_manifest),
    )
    bundle = SimpleNamespace(
        directory=bundle_dir,
        artifact_manifest_sha256=bundle_sha,
        manifest={
            "calibration_artifact_manifest_sha256": "c" * 64,
            "model": {
                "sha256": "m" * 64,
                "hyperparameter_sha256": "h" * 64,
            },
        },
        model=None,
    )
    load_model_flags: list[bool] = []

    def fake_bundle_loader(*_args, **kwargs):
        load_model_flags.append(bool(kwargs["load_model"]))
        return bundle

    monkeypatch.setattr(runner, "load_operational_bundle", fake_bundle_loader)
    monkeypatch.setattr(
        runner,
        "_load_exact_daily_archives",
        lambda **_kwargs: {"FR": archive},
    )
    index = pd.date_range("2026-08-22", periods=2, freq="h", tz="UTC")
    origins = pd.DatetimeIndex([index[0] - pd.Timedelta(hours=18)] * len(index))
    monkeypatch.setattr(
        runner,
        "_read_daily_archive_quantiles",
        lambda *_args, **_kwargs: (_base(index, q50=50.0), origins),
    )

    def fake_writer(**kwargs):
        directory = Path(kwargs["directory"])
        directory.mkdir(parents=True)
        report = directory / "reports" / "report.html"
        report.parent.mkdir()
        report.write_text("blend", encoding="utf-8")
        return report

    monkeypatch.setattr(runner, "_write_daily_passthrough_blend", fake_writer)
    real_validate = runner._validate_artifact_checksums
    monkeypatch.setattr(
        runner,
        "_validate_artifact_checksums",
        lambda path: (
            bundle_sha if Path(path) == bundle_dir else real_validate(Path(path))
        ),
    )
    config = SimpleNamespace(
        contract=SimpleNamespace(project_root=project),
    )
    output = runner.apply_daily_topology(  # type: ignore[arg-type]
        config,
        operational_dir=bundle_dir,
        operational_manifest_sha256=bundle_sha,
        delivery_day="2026-08-22",
        zones=("FR",),
        mode="blend",
        output_root=project / "runs" / "experiments" / "daily",
    )
    assert load_model_flags == [False]
    daily_manifest = json.loads((output / "daily_manifest.json").read_text("utf-8"))
    assert daily_manifest["be_model_sha256"] == "m" * 64
    assert daily_manifest["be_model_loaded"] is False
    assert daily_manifest["be_model_predicted"] is False


def test_atomic_publish_cleans_late_failure_and_never_overwrites_collision(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    destination = project / "runs" / "experiments" / "late_failure"

    def failing_writer(staging: Path) -> None:
        (staging / "partial.html").write_text("partial", encoding="utf-8")
        raise RuntimeError("renderer failed")

    with pytest.raises(RuntimeError, match="renderer failed"):
        atomic_experiment_publish(
            destination,
            project_root=project,
            writer=failing_writer,
        )
    assert not destination.exists()
    assert not list(destination.parent.glob(".late_failure.staging-*"))

    def permission_writer(staging: Path) -> None:
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise PermissionError("transient report lock")

    with pytest.raises(PermissionError, match="transient report lock"):
        atomic_experiment_publish(
            destination,
            project_root=project,
            writer=permission_writer,
        )
    assert not destination.exists()
    assert not list(destination.parent.glob(".late_failure.staging-*"))

    def colliding_writer(staging: Path) -> None:
        (staging / "complete.txt").write_text("new", encoding="utf-8")
        destination.mkdir()
        (destination / "owner.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Collision"):
        atomic_experiment_publish(
            destination,
            project_root=project,
            writer=colliding_writer,
        )
    assert (destination / "owner.txt").read_text("utf-8") == "existing"
    assert not (destination / "complete.txt").exists()
    assert not list(destination.parent.glob(".late_failure.staging-*"))


def test_prepare_existing_bundle_requires_anchor_and_never_refits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    destination = project / "runs" / "experiments" / "bundle"
    destination.mkdir(parents=True)
    calibration = SimpleNamespace(artifact_manifest_sha256="c" * 64)
    config = SimpleNamespace(contract=SimpleNamespace(project_root=project))
    monkeypatch.setattr(runner, "audit_published_calibration", lambda *_a, **_k: calibration)
    with pytest.raises(TopologyExperimentError, match="fournissez son SHA"):
        runner.prepare_operational_bundle(  # type: ignore[arg-type]
            config,
            calibration_dir="ignored",
            output_dir=destination,
        )

    existing = SimpleNamespace(
        manifest={"calibration_artifact_manifest_sha256": "c" * 64}
    )
    monkeypatch.setattr(runner, "load_operational_bundle", lambda *_a, **_k: existing)
    monkeypatch.setattr(
        runner,
        "_fit_fixed_operational_be_model",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("valid existing bundle must not refit")
        ),
    )
    assert runner.prepare_operational_bundle(  # type: ignore[arg-type]
        config,
        calibration_dir="ignored",
        output_dir=destination,
        expected_existing_bundle_sha256="b" * 64,
    ) == destination


def test_prepare_v1_rejects_calibration_anchor_rotation_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        contract=SimpleNamespace(project_root=tmp_path / "project")
    )
    monkeypatch.setattr(
        runner,
        "audit_published_calibration",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("v1 anchor mismatch must fail before calibration I/O")
        ),
    )
    with pytest.raises(TopologyExperimentError, match="ancre de calibration v1"):
        runner.prepare_operational_bundle(  # type: ignore[arg-type]
            config,
            calibration_dir="ignored",
            expected_calibration_manifest_sha256="d" * 64,
        )


def test_prepare_new_operational_bundle_is_atomic_and_checksum_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    calibration_dir = project / "runs" / "experiments" / "calibration"
    calibration_dir.mkdir(parents=True)
    evaluations: dict[str, dict[str, object]] = {}
    recipes: dict[str, dict[str, object]] = {}
    candidate_seals: dict[str, CandidateSeal] = {}
    for zone in runner.SUPPORTED_ZONES:
        source = calibration_dir / zone.lower()
        autonomous = source / "autonomous"
        autonomous.mkdir(parents=True)
        candidate = source / "candidate_predictions.csv.gz"
        pd.DataFrame({"delivery_start_utc": ["2026-01-01T00:00:00Z"]}).to_csv(
            candidate,
            index=False,
            compression="gzip",
        )
        seal = source / "candidate_prediction_seal.json"
        seal.write_text("{}", encoding="utf-8")
        recipe = source / "topology_recipe_seal.json"
        recipe.write_text("{}", encoding="utf-8")
        evaluation = autonomous / "topology_evaluation.json"
        evaluation.write_text("{}", encoding="utf-8")
        evaluations[zone] = {}
        recipes[zone] = {
            "selected_radius": 1 if zone == "BE" else 0,
            "selected_scale": 0.25 if zone == "BE" else 0.0,
        }
        candidate_seals[zone] = CandidateSeal(
            prediction_path=candidate,
            manifest_path=seal,
            prediction_sha256=_sha256(candidate),
            manifest_sha256=_sha256(seal),
        )
    calibration = runner.PublishedCalibration(
        directory=calibration_dir,
        artifact_manifest_sha256="c" * 64,
        experiment_manifest={},
        evaluations=evaluations,
        recipes=recipes,
        candidate_seals=candidate_seals,
    )
    config_path = project / "config.yaml"
    config_path.write_text("fixed", encoding="utf-8")
    model = _FakeCorrector(0.25, raw_by_radius={0: 0.0, 1: 1.0})
    monkeypatch.setattr(
        runner,
        "audit_published_calibration",
        lambda *_args, **_kwargs: calibration,
    )
    monkeypatch.setattr(runner, "audit_experiment_sources", lambda _config: {})
    monkeypatch.setattr(
        runner,
        "_fit_fixed_operational_be_model",
        lambda *_args, **_kwargs: (
            model,
            {
                "training_window_policy": "fixed_published_365_not_rolling",
                "rolling365_enabled": False,
            },
        ),
    )
    config = SimpleNamespace(
        contract=SimpleNamespace(project_root=project),
        source_path=config_path,
        config_sha256=_sha256(config_path),
        experiment_id="test_topology",
    )
    output = project / "runs" / "experiments" / "operational"
    result = runner.prepare_operational_bundle(  # type: ignore[arg-type]
        config,
        calibration_dir=calibration_dir,
        output_dir=output,
    )
    assert result == output
    assert (output / "be_topology_model.joblib").is_file()
    manifest = json.loads((output / "bundle_manifest.json").read_text("utf-8"))
    assert manifest["rolling365_enabled"] is False
    assert manifest["promotion_policy"]["BE"] == "topology_promoted"
    assert manifest["blend_policy"]["FR"] == (
        "production_mkonline_blend_passthrough"
    )
    runner._validate_artifact_checksums(output)
    assert not list(output.parent.glob(".operational.staging-*"))


def test_operational_cli_is_mutually_exclusive_and_apply_requires_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--audit-only", "--apply-daily"])
    monkeypatch.setattr(
        runner,
        "load_experiment_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("CLI validation must happen before config I/O")
        ),
    )
    assert runner.main(["--apply-daily", "--delivery-day", "2026-08-22"]) == 1


def test_cli_blend_without_zones_defaults_to_fr_nl_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    config = SimpleNamespace()

    def fake_load(_path, *, zones, allow_existing_output, verify_hashes):
        observed["config_zones"] = tuple(zones)
        assert allow_existing_output is True
        assert verify_hashes is False
        return config

    def fake_apply(_config, **kwargs):
        observed["apply_zones"] = tuple(kwargs["zones"])
        observed["mode"] = kwargs["mode"]
        output = tmp_path / "runs" / "experiments" / "daily"
        output.mkdir(parents=True)
        return output

    monkeypatch.setattr(runner, "load_experiment_config", fake_load)
    monkeypatch.setattr(runner, "apply_daily_topology", fake_apply)
    assert runner.main(
        [
            "--apply-daily",
            "--delivery-day",
            "2026-08-22",
            "--mode",
            "blend",
            "--operational-manifest-sha256",
            "a" * 64,
        ]
    ) == 0
    assert observed == {
        "config_zones": runner.SUPPORTED_ZONES,
        "apply_zones": ("FR", "NL"),
        "mode": "blend",
    }


def test_apply_daily_both_enforces_policy_without_refit_or_blend_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    bundle_dir = project / "runs" / "experiments" / "bundle"
    bundle_dir.mkdir(parents=True)
    bundle_checksum = bundle_dir / "artifact_checksums.json"
    bundle_checksum.write_text("sealed", encoding="utf-8")
    bundle_sha = _sha256(bundle_checksum)
    index = pd.date_range("2026-08-22", periods=2, freq="h", tz="UTC")
    origins = pd.DatetimeIndex(
        [pd.Timestamp("2026-08-21T06:00:00Z")] * len(index)
    )

    archive_by_zone: dict[str, runner.DailyArchive] = {}
    seals: dict[str, CandidateSeal] = {}
    evaluations: dict[str, dict[str, object]] = {}
    for zone in runner.SUPPORTED_ZONES:
        archive_dir = project / "runs" / "live" / zone.lower()
        inputs_dir = archive_dir / "inputs"
        inputs_dir.mkdir(parents=True)
        forecast_path = archive_dir / f"forecast_hourly_{zone.lower()}.csv"
        forecast_path.write_text("immutable", encoding="utf-8")
        reporting_inputs = []
        for name in runner.REPORTING_INPUT_FILENAMES:
            input_path = inputs_dir / name
            input_path.write_text(f"{zone}-{name}", encoding="utf-8")
            reporting_inputs.append(input_path)
        artifact_paths = [forecast_path, *reporting_inputs]
        if zone in runner.BLEND_ZONES:
            report_path = archive_dir / f"report_{zone.lower()}.html"
            report_path.write_text("production blend", encoding="utf-8")
            artifact_paths.append(report_path)
        checksum_path = archive_dir / "artifact_checksums.json"
        checksum_path.write_text(
            json.dumps(
                {
                    "artifacts": [
                        {
                            "path": path.relative_to(archive_dir).as_posix(),
                            "sha256": _sha256(path),
                        }
                        for path in artifact_paths
                    ]
                }
            ),
            encoding="utf-8",
        )
        archive_by_zone[zone] = runner.DailyArchive(
            zone=zone,
            timezone=runner.ZONE_TIMEZONES[zone],
            delivery_day="2026-08-22",
            directory=archive_dir,
            forecast_path=forecast_path,
            artifact_manifest_sha256=_sha256(checksum_path),
        )
        candidate = bundle_dir / f"{zone.lower()}_candidate.csv.gz"
        pd.DataFrame({"delivery_start_utc": index}).to_csv(
            candidate,
            index=False,
            compression="gzip",
        )
        seal_manifest = bundle_dir / f"{zone.lower()}_seal.json"
        seal_manifest.write_text("{}", encoding="utf-8")
        seals[zone] = CandidateSeal(
            prediction_path=candidate,
            manifest_path=seal_manifest,
            prediction_sha256=_sha256(candidate),
            manifest_sha256=_sha256(seal_manifest),
        )
        evaluations[zone] = {
            "variants": {
                "autonomous": {
                    "opened_stages": ["a", "development", "b1"],
                }
            }
        }

    class FixedBeModel:
        calls = 0

        def predict(self, _context, base):
            self.calls += 1
            return base + 3.0

        def hyperparameter_sha256(self):
            return "h" * 64

    model = FixedBeModel()
    bundle = SimpleNamespace(
        directory=bundle_dir,
        artifact_manifest_sha256=bundle_sha,
        manifest={
            "calibration_artifact_manifest_sha256": "c" * 64,
            "model": {
                "sha256": "m" * 64,
                "selected_radius": 1,
                "hyperparameter_sha256": "h" * 64,
            },
        },
        model=model,
        evaluations=evaluations,
        candidate_seals=seals,
    )
    contract = SimpleNamespace(
        project_root=project,
        sources={
            zone: SimpleNamespace(
                run_directory=project / "sealed" / zone.lower(),
                timezone=runner.ZONE_TIMEZONES[zone],
            )
            for zone in runner.SUPPORTED_ZONES
        },
    )
    config = SimpleNamespace(contract=contract)
    monkeypatch.setattr(runner, "load_operational_bundle", lambda *_a, **_k: bundle)
    monkeypatch.setattr(
        runner,
        "_load_exact_daily_archives",
        lambda **_kwargs: archive_by_zone,
    )

    def fake_quantiles(archive, *, variant):
        level = 100.0 if variant == "mkonline_blend" else 10.0
        return _base(index, q50=level), origins

    monkeypatch.setattr(runner, "_read_daily_archive_quantiles", fake_quantiles)
    monkeypatch.setattr(
        runner,
        "_load_operational_be_context",
        lambda **_kwargs: (
            pd.DataFrame(index=index),
            {
                "be_future_covariates_sha256": _sha256(
                    archive_by_zone["BE"].directory
                    / "inputs"
                    / "model_covariates_with_future.csv.gz"
                ),
                **{
                    f"{zone.lower()}_aligned_inputs_sha256": _sha256(
                        archive_by_zone[zone].directory
                        / "inputs"
                        / "aligned_inputs.csv.gz"
                    )
                    for zone in runner.SUPPORTED_ZONES
                },
            },
        ),
    )
    monkeypatch.setattr(
        runner,
        "_load_storm_after_candidate",
        lambda **_kwargs: (
            pd.Series([10.0, 10.0], index=index, name="storm_dashboard_official__q50"),
            {},
        ),
    )
    monkeypatch.setattr(
        runner,
        "_storm_evaluation_metrics",
        lambda *_args, **kwargs: {"stage": kwargs["stage"]},
    )
    observed: dict[tuple[str, str], pd.DataFrame] = {}

    def fake_autonomous_writer(**kwargs):
        directory = Path(kwargs["directory"])
        directory.mkdir(parents=True)
        report = directory / "reports" / "report.html"
        report.parent.mkdir()
        report.write_text("autonomous", encoding="utf-8")
        observed[(kwargs["zone"], kwargs["variant"])] = kwargs[
            "current_forecast"
        ].copy()
        return SimpleNamespace(path=report)

    def fake_blend_writer(**kwargs):
        directory = Path(kwargs["directory"])
        directory.mkdir(parents=True)
        report = directory / "reports" / "report.html"
        report.parent.mkdir()
        report.write_text("blend", encoding="utf-8")
        observed[(kwargs["zone"], "mkonline_blend")] = kwargs["forecast"].copy()
        return report

    monkeypatch.setattr(runner, "_write_variant_run", fake_autonomous_writer)
    monkeypatch.setattr(runner, "_write_daily_passthrough_blend", fake_blend_writer)
    revalidated_members: set[Path] = set()
    real_revalidate = runner._revalidate_consumed_archive_members

    def capture_revalidation(archives, consumed_members):
        revalidated_members.update(Path(path).resolve() for path in consumed_members)
        return real_revalidate(archives, consumed_members)

    monkeypatch.setattr(
        runner,
        "_revalidate_consumed_archive_members",
        capture_revalidation,
    )
    real_validate = runner._validate_artifact_checksums
    monkeypatch.setattr(
        runner,
        "_validate_artifact_checksums",
        lambda path: bundle_sha if Path(path) == bundle_dir else real_validate(Path(path)),
    )
    for forbidden in (
        "_fit_fixed_operational_be_model",
        "_blend_quantiles",
        "run_blend_protocol",
        "select_l1_blend_weight",
    ):
        monkeypatch.setattr(
            runner,
            forbidden,
            lambda *_a, _name=forbidden, **_k: (_ for _ in ()).throw(
                AssertionError(f"daily apply called forbidden {_name}")
            ),
        )

    live_root = project / "runs" / "live"
    live_before = {
        path.relative_to(live_root).as_posix(): _sha256(path)
        for path in live_root.rglob("*")
        if path.is_file()
    }
    output = runner.apply_daily_topology(  # type: ignore[arg-type]
        config,
        operational_dir=bundle_dir,
        operational_manifest_sha256=bundle_sha,
        delivery_day="2026-08-22",
        zones=runner.SUPPORTED_ZONES,
        mode="both",
        output_root=project / "runs" / "experiments" / "daily",
    )
    assert output == (
        project / "runs" / "experiments" / "daily" / "2026-08-22" / "both"
    )
    assert len(observed) == 7
    assert model.calls == 1
    for zone in ("FR", "DE", "NL", "ES"):
        frame = observed[(zone, "autonomous")]
        assert np.array_equal(
            frame["residual_corrected__q50"].to_numpy(),
            frame["topology_autonomous__q50"].to_numpy(),
        )
    be_frame = observed[("BE", "autonomous")]
    assert np.array_equal(
        be_frame["topology_autonomous__q50"].to_numpy(),
        be_frame["residual_corrected__q50"].to_numpy() + 3.0,
    )
    for zone in ("FR", "NL"):
        frame = observed[(zone, "mkonline_blend")]
        assert np.array_equal(
            frame["mkonline_blend__q50"].to_numpy(),
            frame["topology_mkonline_blend__q50"].to_numpy(),
        )
    for zone in runner.SUPPORTED_ZONES:
        for name in runner.REPORTING_INPUT_FILENAMES:
            assert (
                archive_by_zone[zone].directory / "inputs" / name
            ).resolve() in revalidated_members
    manifest = json.loads((output / "daily_manifest.json").read_text("utf-8"))
    assert manifest["rolling365_enabled"] is False
    assert manifest["model_refitted"] is False
    assert manifest["be_model_sha256"] == "m" * 64
    assert manifest["be_model_loaded"] is True
    assert manifest["be_model_predicted"] is True
    runner._validate_artifact_checksums(output)
    live_after = {
        path.relative_to(live_root).as_posix(): _sha256(path)
        for path in live_root.rglob("*")
        if path.is_file()
    }
    assert live_after == live_before


def _annual_candidate_frame(
    splits,
    *,
    opened_stages: tuple[str, ...],
    base_q50: float = 1.0,
    candidate_q50: float = 0.0,
) -> pd.DataFrame:
    pieces = []
    for stage in opened_stages:
        index = splits.indices[stage]
        pieces.append(
            pd.DataFrame(
                {
                    "delivery_start_utc": index,
                    "forecast_origin_utc": _forecast_origins_for_delivery(
                        index, timezone_name=splits.timezone
                    ),
                    "stage": stage,
                    "actual": np.zeros(len(index)),
                    "residual_corrected__q10": np.full(len(index), base_q50 - 0.5),
                    "residual_corrected__q50": np.full(len(index), base_q50),
                    "residual_corrected__q90": np.full(len(index), base_q50 + 0.5),
                    "topology_autonomous__q10": np.full(
                        len(index), candidate_q50 - 0.5
                    ),
                    "topology_autonomous__q50": np.full(len(index), candidate_q50),
                    "topology_autonomous__q90": np.full(
                        len(index), candidate_q50 + 0.5
                    ),
                }
            )
        )
    result = pd.concat(pieces, ignore_index=True)
    result.index = pd.DatetimeIndex(
        pd.to_datetime(result["delivery_start_utc"], utc=True),
        name="delivery_start_utc",
    )
    return result


def _annual_evaluation(
    *, opened_stages: tuple[str, ...], gate_passes: dict[str, bool]
) -> dict[str, object]:
    return {
        "variants": {
            "autonomous": {
                "opened_stages": list(opened_stages),
                "gates": {
                    stage: {"passes": passes}
                    for stage, passes in gate_passes.items()
                },
            }
        },
        "storm_evaluation": {"available": False},
    }


def _annual_calibration(
    tmp_path: Path,
    *,
    zone: str,
    evaluation: dict[str, object],
) -> runner.PublishedCalibration:
    candidate = tmp_path / f"{zone.lower()}_candidate.csv.gz"
    candidate.write_bytes(b"sealed")
    seal = tmp_path / f"{zone.lower()}_candidate_seal.json"
    seal.write_text("{}", encoding="utf-8")
    return runner.PublishedCalibration(
        directory=tmp_path,
        artifact_manifest_sha256="c" * 64,
        experiment_manifest={},
        evaluations={zone: evaluation},
        recipes={},
        candidate_seals={
            zone: CandidateSeal(
                prediction_path=candidate,
                manifest_path=seal,
                prediction_sha256=_sha256(candidate),
                manifest_sha256=_sha256(seal),
            )
        },
    )


def test_annual_policy_waits_for_b1_and_excludes_nonformal_stages(
    tmp_path: Path,
) -> None:
    splits = _splits()
    calibration = _annual_calibration(
        tmp_path,
        zone="BE",
        evaluation=_annual_evaluation(
            opened_stages=("a", "development", "b1", "b2", "final"),
            gate_passes={"b1": True, "b2": True, "final": True},
        ),
    )
    directory = tmp_path / "be"
    directory.mkdir()
    policy = runner._write_annual_policy_seal(
        directory,
        config=SimpleNamespace(experiment_id="annual_test", config_sha256="a" * 64),
        calibration=calibration,
        zone="BE",
        splits=splits,
    )
    assert policy.governed_actions == {
        "seed": "identity",
        "a": "identity",
        "development": "identity",
        "b1": "identity",
        "b2": "sealed_topology_candidate",
        "final": "sealed_topology_candidate",
    }
    assert policy.formal_shadow_actions == {
        "seed": "identity",
        "a": "identity",
        "development": "identity",
        "b1": "sealed_topology_candidate",
        "b2": "sealed_topology_candidate",
        "final": "sealed_topology_candidate",
    }
    payload = json.loads(policy.path.read_text(encoding="utf-8"))
    assert payload["full_year_outcomes_opened_before_policy_seal"] is False
    assert payload["selection_A_excluded_from_strategies"] is True
    assert payload["development_excluded_from_strategies"] is True
    assert payload["out_of_sample_scope"] == "mixed_sequential_governed"
    for flag in ("no_fit", "no_predict", "no_refit", "no_new_prediction"):
        assert payload[flag] is True


def test_annual_strategy_copies_only_authorized_sealed_predictions(
    tmp_path: Path,
) -> None:
    splits = _splits()
    index = splits.all_index
    calibration = _annual_calibration(
        tmp_path,
        zone="BE",
        evaluation=_annual_evaluation(
            opened_stages=("a", "development", "b1", "b2", "final"),
            gate_passes={"b1": True, "b2": True, "final": True},
        ),
    )
    directory = tmp_path / "be"
    directory.mkdir()
    policy = runner._write_annual_policy_seal(
        directory,
        config=SimpleNamespace(experiment_id="annual_test", config_sha256="a" * 64),
        calibration=calibration,
        zone="BE",
        splits=splits,
    )
    source = runner.AnnualSourceData(
        actual=pd.Series(0.0, index=index),
        base_predictions=pd.DataFrame(
            {"q10": 0.5, "q50": 1.0, "q90": 1.5}, index=index
        ),
        forecast_origins=_forecast_origins_for_delivery(
            index, timezone_name=splits.timezone
        ),
        source_sha256="s" * 64,
    )
    candidate = _annual_candidate_frame(
        splits,
        opened_stages=("a", "development", "b1", "b2", "final"),
    )
    frame = runner._build_annual_strategy_frame(
        zone="BE",
        splits=splits,
        source=source,
        candidate=candidate,
        policy=policy,
    )
    stage = frame["protocol_stage"]
    governed_active = stage.isin(["b2", "final"])
    shadow_active = stage.isin(["b1", "b2", "final"])
    assert int(frame["governed_topology_active"].sum()) == 2160
    assert int(frame["formal_shadow_topology_active"].sum()) == 2880
    assert np.array_equal(
        frame.loc[~governed_active, "sequential_governed_strategy__q50"],
        frame.loc[~governed_active, "residual_corrected__q50"],
    )
    assert np.array_equal(
        frame.loc[governed_active, "sequential_governed_strategy__q50"],
        frame.loc[governed_active, "topology_opened_candidate__q50"],
    )
    assert np.array_equal(
        frame.loc[shadow_active, "causal_formal_shadow_strategy__q50"],
        frame.loc[shadow_active, "topology_opened_candidate__q50"],
    )
    assert not bool(
        frame.loc[stage.isin(["a", "development"]), "candidate_formal_oos"].any()
    )
    metrics = runner._annual_strategy_metrics(
        frame,
        model="sequential_governed_strategy",
        active_column="governed_topology_active",
        timezone_name=splits.timezone,
        bootstrap_samples=100,
        bootstrap_seed=120,
    )
    assert metrics["active_hours"] == 2160
    assert metrics["active_days"] == 90
    assert metrics["full_period_metrics"]["gain_eur_mwh"] == pytest.approx(
        2160 / 8760
    )
    assert metrics["active_only_metrics"]["gain_eur_mwh"] == pytest.approx(1.0)
    outcomes = metrics["daily_outcomes"]["all_days"]
    assert (outcomes["wins"], outcomes["ties"], outcomes["losses"]) == (90, 275, 0)


def test_report_365_seals_all_policies_before_outcomes_and_never_calls_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "runs" / "experiments").mkdir(parents=True)
    splits = build_protocol_splits(
        _annual_index(),
        timezone_name="Europe/Brussels",
        split_days=DEFAULT_SPLIT_DAYS,
    )
    index = splits.all_index
    calibration = _annual_calibration(
        tmp_path,
        zone="BE",
        evaluation=_annual_evaluation(
            opened_stages=("a", "development", "b1", "b2", "final"),
            gate_passes={"b1": True, "b2": True, "final": True},
        ),
    )
    source = runner.AnnualSourceData(
        actual=pd.Series(0.0, index=index),
        base_predictions=pd.DataFrame(
            {"q10": 0.5, "q50": 1.0, "q90": 1.5}, index=index
        ),
        forecast_origins=_forecast_origins_for_delivery(
            index, timezone_name=splits.timezone
        ),
        source_sha256="d" * 64,
    )
    candidate = _annual_candidate_frame(
        splits,
        opened_stages=("a", "development", "b1", "b2", "final"),
    )
    config = SimpleNamespace(
        contract=SimpleNamespace(project_root=project),
        zones=("BE",),
        experiment_id="annual_test",
        config_sha256="a" * 64,
        bootstrap_samples=50,
        bootstrap_seed=120,
    )
    monkeypatch.setattr(runner, "audit_published_calibration", lambda *_a, **_k: calibration)
    monkeypatch.setattr(runner, "audit_experiment_sources", lambda _c: {"BE": {}})
    monkeypatch.setattr(runner, "_splits_for_contract", lambda *_a, **_k: splits)
    monkeypatch.setattr(
        runner.TopologyResidualCorrector,
        "fit",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("fit interdit")),
    )
    monkeypatch.setattr(
        runner.TopologyResidualCorrector,
        "predict",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("predict interdit")),
    )
    policies_written = 0
    real_policy_writer = runner._write_annual_policy_seal

    def capture_policy(*args, **kwargs):
        nonlocal policies_written
        result = real_policy_writer(*args, **kwargs)
        policies_written += 1
        return result

    def guarded_source(*_args, **_kwargs):
        assert policies_written == 1
        return source

    monkeypatch.setattr(runner, "_write_annual_policy_seal", capture_policy)
    monkeypatch.setattr(runner, "_load_annual_source_data", guarded_source)
    monkeypatch.setattr(runner, "_load_annual_opened_candidate", lambda *_a, **_k: candidate)
    monkeypatch.setattr(
        runner,
        "_load_annual_storm_reference",
        lambda **_k: (None, {"available": False, "reason": "test"}),
    )
    output = runner.report_topology_annual_365(
        config,
        calibration_dir=tmp_path,
        output_dir=project / "runs" / "experiments" / "annual",
        expected_calibration_manifest_sha256="c" * 64,
    )
    metrics = json.loads((output / "be" / "annual_metrics.json").read_text("utf-8"))
    assert metrics["period"]["n_hours"] == 8760
    assert metrics["strategies"]["sequential_governed_strategy"]["active_hours"] == 2160
    assert metrics["strategies"]["causal_formal_shadow_strategy"]["active_hours"] == 2880
    assert metrics["pure_topology_annual"]["available"] is False
    assert metrics["no_fit"] is True and metrics["no_predict"] is True
    report = output / "be" / "reports" / "topology_annual_be.html"
    index_report = output / "reports" / "topology_annual_365_index.html"
    assert report.is_file() and report.stat().st_size > 0
    assert index_report.is_file() and index_report.stat().st_size > 0
    record = runner.load_topology_annual_evaluation(
        output / "be",
        project_root=project,
    )
    assert record.zone == "BE" and record.n_hours == 8760 and record.n_days == 365
    manifest = json.loads((output / "annual_manifest.json").read_text("utf-8"))
    assert manifest["annual_index_path"] == "reports/topology_annual_365_index.html"
    assert manifest["zone_artifacts"]["BE"]["annual_report_sha256"] == runner._sha256(
        report
    )
    runner._validate_artifact_checksums(output / "be")
    runner._validate_artifact_checksums(output)
    with pytest.raises(FileExistsError):
        runner.report_topology_annual_365(
            config,
            calibration_dir=tmp_path,
            output_dir=output,
            expected_calibration_manifest_sha256="c" * 64,
        )


def test_report_365_cli_is_mutually_exclusive_and_has_safe_default() -> None:
    args = runner.parse_args(["--report-365"])
    assert args.report_365 is True
    assert Path(args.annual_output_dir) == runner.DEFAULT_ANNUAL_OUTPUT_DIRECTORY
    with pytest.raises(SystemExit):
        runner.parse_args(["--report-365", "--apply-daily"])


def test_rolling365_cli_is_mutually_exclusive_and_has_safe_defaults() -> None:
    args = runner.parse_args(["--rolling365-backtest"])
    assert args.rolling365_backtest is True
    assert args.rolling365_workers == 5
    assert Path(args.rolling365_output_dir) == (
        runner.DEFAULT_ROLLING365_OUTPUT_DIRECTORY
    )
    with pytest.raises(SystemExit):
        runner.parse_args(["--rolling365-backtest", "--report-365"])


def test_rolling365_refits_each_day_on_exact_previous_365_local_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    combined = runner._rolling365_combined_index(splits)
    evaluation = splits.all_index
    assert len(pd.Index(combined.tz_convert(splits.timezone).date).unique()) == 730
    assert len(evaluation) == 8760

    ensemble = pd.DataFrame(
        {"q10": 0.0, "q50": 1.0, "q90": 2.0}, index=combined
    )
    current = pd.DataFrame(
        {"q10": 0.5, "q50": 1.5, "q90": 2.5}, index=evaluation
    )
    contexts = {
        radius: pd.DataFrame({"feature": 0.0}, index=combined)
        for radius in (0, 1)
    }
    data = runner.Rolling365ZoneData(
        actual=pd.Series(1.0, index=combined),
        ensemble_base=ensemble,
        current_autonomous=current,
        forecast_origins=_forecast_origins_for_delivery(
            combined, timezone_name=splits.timezone
        ),
        contexts=contexts,
        evaluation_index=evaluation,
        source_sha256="s" * 64,
    )
    fit_windows: list[pd.DatetimeIndex] = []

    class FakeCorrector:
        def hyperparameter_sha256(self) -> str:
            return "h" * 64

        def fit(self, X, y, base_predictions):
            fit_windows.append(X.index)
            return self

        def predict(self, X, base_predictions):
            result = base_predictions.copy() + 1.0
            result.attrs["topology_correction"] = pd.Series(1.0, index=X.index)
            return result

        def audit_metadata(self):
            return {
                "n_training_rows": len(fit_windows[-1]),
                "n_dropped_target_rows": 0,
                "fit_all_missing_feature_rows": 0,
            }

    config = SimpleNamespace(
        contract=object(),
        config_sha256="c" * 64,
        model_parameters={},
    )
    monkeypatch.setattr(runner, "_splits_for_contract", lambda *_a, **_k: splits)
    monkeypatch.setattr(
        runner, "_load_rolling365_zone_data", lambda *_a, **_k: data
    )
    monkeypatch.setattr(runner, "_new_corrector", lambda *_a, **_k: FakeCorrector())
    result = runner._fit_rolling365_zone(
        config,
        zone="FR",
        recipe={
            "selected_radius": 0,
            "selected_scale": 0.25,
            "selected_arm": "radius0_scale0.25",
            "config_sha256": "c" * 64,
            "model_parameters": {},
            "model_hyperparameters_sha256": "h" * 64,
        },
        recipe_sha256="r" * 64,
    )
    assert len(fit_windows) == 365
    assert len(result.refits) == 365
    assert len(result.predictions) == 8760
    for window in fit_windows:
        assert len(pd.Index(window.tz_convert(splits.timezone).date).unique()) == 365
    assert set(result.refits["training_physical_hours"]) <= {8759, 8760, 8761}
    assert np.array_equal(
        result.predictions["topology_rolling365__q50"].to_numpy(),
        np.full(8760, 2.0),
    )
