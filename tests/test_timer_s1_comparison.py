from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.timer_s1_comparison as comparison_module
from chronos2_hourly.timer_s1_comparison import (
    QUANTILES,
    SourceProtocol,
    TimerS1ComparisonError,
    compare_target_only_native,
    compare_target_only_same_downstream_features,
    paired_daily_mae_bootstrap,
    sha256_target_series,
    validate_timer_oof,
)


TIMEZONE = "Europe/Paris"


def _delivery_index(start_day: str, end_day_exclusive: str) -> pd.DatetimeIndex:
    return pd.date_range(
        start_day,
        end_day_exclusive,
        inclusive="left",
        freq="h",
        tz=TIMEZONE,
    ).tz_convert("UTC").rename("delivery_start_utc")


def _forecast_origins(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    origins = []
    for timestamp in index:
        delivery_day = timestamp.tz_convert(TIMEZONE).date()
        origin = (
            pd.Timestamp(delivery_day, tz=TIMEZONE)
            - pd.Timedelta(days=1)
            + pd.Timedelta(hours=8)
        )
        origins.append(origin.tz_convert("UTC"))
    return pd.DatetimeIndex(origins)


def test_target_hash_is_stable_across_pandas_datetime_units() -> None:
    index_ns = pd.date_range(
        "2024-01-01",
        periods=48,
        freq="h",
        tz="UTC",
    ).as_unit("ns")
    values = np.linspace(-50.0, 125.0, len(index_ns))
    target_ns = pd.Series(values, index=index_ns)
    target_us = pd.Series(values, index=index_ns.as_unit("us"))

    assert sha256_target_series(target_ns) == sha256_target_series(target_us)


def _quantile_frame(
    index: pd.DatetimeIndex,
    actual: pd.Series,
    *,
    median_bias: float,
    half_width: float = 3.0,
) -> pd.DataFrame:
    truth = actual.loc[index].to_numpy(dtype=float)
    median = truth + median_bias
    return pd.DataFrame(
        {
            "q10": median - half_width,
            "q50": median,
            "q90": median + half_width,
            "forecast_origin_utc": _forecast_origins(index),
            "actual": truth,
        },
        index=index,
    )


def _provenance(protocol: SourceProtocol, model_name: str) -> dict[str, object]:
    contract = (
        {
            "revin": True,
            "use_cache": False,
            "quantile_indices": {"q10": 0, "q50": 4, "q90": 8},
        }
        if model_name == "timer_s1"
        else {
            "cross_learning": False,
            "quantile_levels": [0.1, 0.5, 0.9],
        }
    )
    return {
        "model_name": model_name,
        "forecast_mode": "strict_native_target_only",
        "context_length": 2048,
        "source_feature_manifest_sha256": protocol.feature_manifest_sha256,
        "source_target_sha256": sha256_target_series(protocol.target),
        "native_covariates": [],
        "inference_contract": contract,
    }


@pytest.fixture
def protocol_and_oof() -> tuple[SourceProtocol, pd.DataFrame, pd.DataFrame]:
    # One extended day, one calibration day, then two sealed evaluation days.
    # Using local-day construction keeps the fixture faithful to the production
    # protocol while leaving every expected score analytically simple.
    full_index = _delivery_index("2024-01-02", "2024-01-06")
    target = pd.Series(
        np.arange(len(full_index), dtype=float),
        index=full_index,
        name="actual",
    )
    extended_index = _delivery_index("2024-01-02", "2024-01-03")
    main_index = _delivery_index("2024-01-03", "2024-01-06")
    calibration_index = _delivery_index("2024-01-03", "2024-01-04")
    evaluation_index = _delivery_index("2024-01-04", "2024-01-06")

    chronos_full = _quantile_frame(full_index, target, median_bias=2.0)
    timer_full = _quantile_frame(full_index, target, median_bias=1.0)
    chronos_extended = chronos_full.loc[extended_index].copy()
    chronos_main = chronos_full.loc[main_index].copy()

    protocol = SourceProtocol(
        source_run=Path("synthetic-source-run"),
        timezone=TIMEZONE,
        target=target,
        history_features=pd.DataFrame({"feature": 0.0}, index=full_index),
        future_features=pd.DataFrame({"feature": 0.0}, index=full_index),
        chronos_extended=chronos_extended,
        chronos_main=chronos_main,
        current_backtest=pd.DataFrame(index=evaluation_index),
        calibration_index=calibration_index,
        evaluation_index=evaluation_index,
        feature_names=("feature",),
        feature_manifest_sha256="synthetic-sha256",
        expected_meta_features=1,
    )
    return protocol, timer_full, chronos_full


def test_validate_timer_oof_accepts_only_exact_frozen_protocol(
    protocol_and_oof: tuple[SourceProtocol, pd.DataFrame, pd.DataFrame],
) -> None:
    protocol, timer, _ = protocol_and_oof

    validated = validate_timer_oof(timer, protocol)

    pd.testing.assert_frame_equal(validated, timer)
    assert validated is not timer
    assert validated.index.equals(protocol.full_oof_index)
    expected_origins = pd.concat(
        [
            protocol.chronos_extended["forecast_origin_utc"],
            protocol.chronos_main["forecast_origin_utc"],
        ]
    )
    assert pd.DatetimeIndex(validated["forecast_origin_utc"]).equals(
        pd.DatetimeIndex(expected_origins)
    )


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing_row", "does not exactly cover EXT\\+CAL\\+FINAL"),
        ("wrong_origin", "origins differ"),
        ("wrong_actual", "actual values differ"),
    ],
)
def test_validate_timer_oof_rejects_index_origin_and_target_drift(
    protocol_and_oof: tuple[SourceProtocol, pd.DataFrame, pd.DataFrame],
    defect: str,
    message: str,
) -> None:
    protocol, timer, _ = protocol_and_oof
    malformed = timer.copy()
    if defect == "missing_row":
        malformed = malformed.iloc[1:]
    elif defect == "wrong_origin":
        # It remains causal, so this specifically exercises exact origin parity.
        malformed.iloc[0, malformed.columns.get_loc("forecast_origin_utc")] -= (
            pd.Timedelta(hours=1)
        )
    else:
        malformed.iloc[0, malformed.columns.get_loc("actual")] += 0.01

    with pytest.raises(TimerS1ComparisonError, match=message):
        validate_timer_oof(malformed, protocol)


def test_paired_daily_bootstrap_equal_weights_dst_days_and_is_reproducible() -> None:
    # These two delivery days contain 24 and 25 physical hours. The expected
    # point estimate is nevertheless the equal-weight mean of the two day MAEs.
    index = _delivery_index("2024-10-26", "2024-10-28")
    local_days = pd.Index(index.tz_convert(TIMEZONE).date)
    first_day = local_days[0]
    actual = pd.Series(0.0, index=index)
    baseline = pd.Series(
        np.where(local_days == first_day, 1.0, 3.0),
        index=index,
    )
    candidate = pd.Series(
        np.where(local_days == first_day, 0.0, 2.0),
        index=index,
    )

    first = paired_daily_mae_bootstrap(
        actual,
        baseline,
        candidate,
        timezone=TIMEZONE,
        samples=251,
        seed=7,
    )
    second = paired_daily_mae_bootstrap(
        actual,
        baseline,
        candidate,
        timezone=TIMEZONE,
        samples=251,
        seed=7,
    )

    assert first == second
    assert first["n_delivery_days"] == 2
    assert first["baseline_mae"] == pytest.approx(2.0)
    assert first["candidate_mae"] == pytest.approx(1.0)
    assert first["delta_mae"] == pytest.approx(-1.0)
    assert first["relative_improvement"] == pytest.approx(0.5)
    assert first["ci95_delta_mae"] == pytest.approx([-1.0, -1.0])
    assert first["estimator"] == "equal_weight_daily_mae"
    assert first["bootstrap_fraction_delta_below_zero"] == pytest.approx(1.0)


def test_paired_daily_bootstrap_rejects_unaligned_inputs() -> None:
    index = _delivery_index("2024-01-02", "2024-01-03")
    actual = pd.Series(0.0, index=index)
    baseline = pd.Series(1.0, index=index)
    candidate = pd.Series(0.0, index=index.shift(1, freq="h"))

    with pytest.raises(TimerS1ComparisonError, match="not aligned"):
        paired_daily_mae_bootstrap(
            actual,
            baseline,
            candidate,
            timezone=TIMEZONE,
            samples=10,
        )


def test_compare_target_only_native_scores_sealed_window_and_quantiles(
    protocol_and_oof: tuple[SourceProtocol, pd.DataFrame, pd.DataFrame],
) -> None:
    protocol, timer, chronos = protocol_and_oof
    outside_evaluation = ~chronos.index.isin(protocol.evaluation_index)
    # Extreme errors outside the sealed evaluation window must not leak into
    # the reported benchmark.
    chronos.loc[outside_evaluation, list(QUANTILES)] += 1_000.0
    timer.loc[outside_evaluation, list(QUANTILES)] += 1_000.0

    metrics, paired = compare_target_only_native(
        protocol,
        chronos_target_only_oof=chronos,
        timer_oof=timer,
        chronos_provenance=_provenance(protocol, "chronos2_target_only"),
        timer_provenance=_provenance(protocol, "timer_s1"),
        bootstrap_samples=127,
        seed=11,
    )

    assert metrics["model"].tolist() == [
        "chronos2_target_only",
        "timer_s1_target_only",
    ]
    assert metrics["stage"].eq("strict_native_target_only").all()
    assert metrics["native_information_parity"].eq(True).all()
    assert metrics["n_scored"].eq(len(protocol.evaluation_index)).all()

    chronos_row = metrics.set_index("model").loc["chronos2_target_only"]
    assert chronos_row["mae"] == pytest.approx(2.0)
    assert chronos_row["rmse"] == pytest.approx(2.0)
    assert chronos_row["bias"] == pytest.approx(2.0)
    assert chronos_row["coverage_80"] == pytest.approx(1.0)
    assert chronos_row["mean_width_80"] == pytest.approx(6.0)
    assert chronos_row["pinball_q10"] == pytest.approx(0.1)
    assert chronos_row["pinball_q50"] == pytest.approx(1.0)
    assert chronos_row["pinball_q90"] == pytest.approx(0.5)
    assert chronos_row["mean_pinball_q10_q50_q90"] == pytest.approx(1.6 / 3.0)

    timer_row = metrics.set_index("model").loc["timer_s1_target_only"]
    assert timer_row["mae"] == pytest.approx(1.0)
    assert timer_row["rmse"] == pytest.approx(1.0)
    assert timer_row["bias"] == pytest.approx(1.0)
    assert timer_row["coverage_80"] == pytest.approx(1.0)
    assert timer_row["mean_width_80"] == pytest.approx(6.0)
    assert timer_row["pinball_q10"] == pytest.approx(0.2)
    assert timer_row["pinball_q50"] == pytest.approx(0.5)
    assert timer_row["pinball_q90"] == pytest.approx(0.4)
    assert timer_row["mean_pinball_q10_q50_q90"] == pytest.approx(1.1 / 3.0)

    assert paired["comparison_claim"] == (
        "strict_native_target_only_information_parity"
    )
    assert paired["baseline_mae"] == pytest.approx(2.0)
    assert paired["candidate_mae"] == pytest.approx(1.0)
    assert paired["delta_mae"] == pytest.approx(-1.0)
    assert paired["bootstrap_fraction_delta_below_zero"] == pytest.approx(1.0)


def test_compare_target_only_native_rejects_chronos_origin_drift(
    protocol_and_oof: tuple[SourceProtocol, pd.DataFrame, pd.DataFrame],
) -> None:
    protocol, timer, chronos = protocol_and_oof
    chronos = chronos.copy()
    chronos.iloc[0, chronos.columns.get_loc("forecast_origin_utc")] -= (
        pd.Timedelta(hours=1)
    )

    with pytest.raises(TimerS1ComparisonError, match="origins differ"):
        compare_target_only_native(
            protocol,
            chronos_target_only_oof=chronos,
            timer_oof=timer,
            chronos_provenance=_provenance(protocol, "chronos2_target_only"),
            timer_provenance=_provenance(protocol, "timer_s1"),
            bootstrap_samples=10,
        )


def test_compare_target_only_native_rejects_unverified_feature_mode(
    protocol_and_oof: tuple[SourceProtocol, pd.DataFrame, pd.DataFrame],
) -> None:
    protocol, timer, chronos = protocol_and_oof
    chronos_provenance = _provenance(protocol, "chronos2_target_only")
    chronos_provenance["native_covariates"] = ["smuggled_covariate"]

    with pytest.raises(TimerS1ComparisonError, match="native_covariates"):
        compare_target_only_native(
            protocol,
            chronos_target_only_oof=chronos,
            timer_oof=timer,
            chronos_provenance=chronos_provenance,
            timer_provenance=_provenance(protocol, "timer_s1"),
            bootstrap_samples=10,
        )


def test_compare_target_only_same_downstream_features_reports_four_fair_rows(
    protocol_and_oof: tuple[SourceProtocol, pd.DataFrame, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, timer, chronos = protocol_and_oof
    meta_names = tuple(f"meta_feature_{position:03d}" for position in range(175))
    pd.DataFrame({"feature": meta_names}).to_csv(
        tmp_path / "residual_feature_importance.csv",
        index=False,
    )
    protocol = replace(
        protocol,
        source_run=tmp_path,
        expected_meta_features=len(meta_names),
    )
    fit_calls: list[dict[str, object]] = []

    class FakeCorrector:
        def __init__(self, label: str) -> None:
            self.label = label
            self.components_ = {
                "cat_v1": SimpleNamespace(feature_columns_=meta_names)
            }

        def diagnostics(self) -> dict[str, object]:
            return {"label": self.label, "meta_feature_count": len(meta_names)}

    def fake_fit_frozen_corrector(
        received_protocol: SourceProtocol,
        *,
        extended: pd.DataFrame,
        main: pd.DataFrame,
        threads: int,
    ) -> tuple[pd.DataFrame, FakeCorrector]:
        assert received_protocol is protocol
        assert extended.index.equals(protocol.chronos_extended.index)
        assert main.index.equals(protocol.chronos_main.index)
        evaluation = main.loc[protocol.evaluation_index, QUANTILES].copy()
        native_bias = float(
            (
                evaluation["q50"]
                - protocol.target.loc[protocol.evaluation_index]
            ).iloc[0]
        )
        label = "timer_s1" if native_bias == pytest.approx(1.0) else "chronos2"
        # The identical synthetic downstream recipe removes one unit from each
        # quantile for both models; only the incoming backbone forecast differs.
        corrected = evaluation - 1.0
        fit_calls.append(
            {
                "label": label,
                "threads": threads,
                "extended": extended.copy(),
                "main": main.copy(),
            }
        )
        return corrected, FakeCorrector(label)

    monkeypatch.setattr(
        comparison_module,
        "_fit_frozen_corrector",
        fake_fit_frozen_corrector,
    )

    result = compare_target_only_same_downstream_features(
        protocol,
        chronos_target_only_oof=chronos,
        timer_oof=timer,
        chronos_provenance=_provenance(protocol, "chronos2_target_only"),
        timer_provenance=_provenance(protocol, "timer_s1"),
        threads=3,
        bootstrap_samples=101,
        seed=19,
    )

    assert [call["label"] for call in fit_calls] == ["timer_s1", "chronos2"]
    assert [call["threads"] for call in fit_calls] == [3, 3]
    assert result.metrics["model"].tolist() == [
        "chronos2_target_only_native",
        "timer_s1_target_only_native",
        "chronos2_target_only_same_corrector",
        "timer_s1_target_only_same_corrector",
    ]
    assert result.metrics["native_information_parity"].eq(True).all()
    assert result.metrics.set_index("model")["mae"].to_dict() == pytest.approx(
        {
            "chronos2_target_only_native": 2.0,
            "timer_s1_target_only_native": 1.0,
            "chronos2_target_only_same_corrector": 1.0,
            "timer_s1_target_only_same_corrector": 0.0,
        }
    )

    assert set(result.paired_tests) == {
        "strict_native_target_only",
        "strict_target_only_same_downstream_features",
    }
    for paired in result.paired_tests.values():
        assert paired["delta_mae"] == pytest.approx(-1.0)
        assert paired["bootstrap_fraction_delta_below_zero"] == pytest.approx(1.0)

    assert result.feature_parity["native_backbone_information_parity"] is True
    assert result.feature_parity["downstream_feature_manifest_exact"] is True
    assert result.feature_parity["residual_meta_schema_exact"] is True
    assert result.feature_parity["residual_meta_feature_count"] == 175
    assert (
        result.feature_parity["both_downstream_correctors_refitted_by_same_function"]
        is True
    )
    assert result.corrector_diagnostics == {
        "timer_s1": {"label": "timer_s1", "meta_feature_count": 175},
        "chronos2_target_only": {
            "label": "chronos2",
            "meta_feature_count": 175,
        },
    }
