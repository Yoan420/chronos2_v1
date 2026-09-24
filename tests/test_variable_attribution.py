from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.variable_attribution import (
    VARIABLE_ATTRIBUTION_AUDIT,
    VARIABLE_ATTRIBUTION_HOURLY,
    PAST_PRICE_GROUP_KEY,
    MAX_ATTRIBUTION_SCENARIOS,
    VariableAttributionError,
    _causal_profile,
    _coalition_masks,
    _shapley_values,
    _target_price_derivatives,
    build_variable_groups,
    remove_variable_attribution_artifacts,
    write_variable_attribution,
)
from chronos2_modular.common import ZoneData


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _InteractionPipeline:
    def predict_df(
        self,
        context_df: pd.DataFrame,
        *,
        future_df: pd.DataFrame,
        prediction_length: int,
        **_kwargs: object,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for item_id in context_df["item_id"].drop_duplicates():
            block = future_df.loc[future_df["item_id"].eq(item_id)].copy()
            assert len(block) == prediction_length
            a = block["known_a_oracle"].to_numpy(dtype=float)
            b = block["known_b_oracle"].to_numpy(dtype=float)
            q50 = 10.0 + 2.0 * a + b + 3.0 * a * b
            for timestamp, value in zip(block["timestamp"], q50, strict=True):
                rows.append(
                    {
                        "item_id": item_id,
                        "timestamp": timestamp,
                        "target_name": "target",
                        "predictions": value,
                        "0.1": value - 5.0,
                        "0.5": value,
                        "0.9": value + 5.0,
                    }
                )
        return pd.DataFrame(rows)


class _IdentityCorrector:
    def predict(
        self,
        _fresh: pd.DataFrame,
        base: pd.DataFrame,
        _experts: pd.DataFrame,
    ) -> pd.DataFrame:
        return base.copy()


def _fixture() -> tuple[ZoneData, pd.DataFrame, pd.DataFrame]:
    future_index = pd.date_range(
        "2026-08-12T22:00:00Z", periods=24, freq="h", tz="UTC"
    )
    history_index = pd.date_range(
        end=future_index[0] - pd.Timedelta(hours=1),
        periods=24 * 60,
        freq="h",
        tz="UTC",
    )
    all_index = history_index.append(future_index)
    target = pd.Series(50.0, index=history_index, name="target")
    covariates = pd.DataFrame(
        {
            "a": np.r_[np.ones(len(history_index)), np.full(24, 3.0)],
            "b": np.r_[np.full(len(history_index), 2.0), np.full(24, 4.0)],
        },
        index=all_index,
    )
    model_context = covariates.copy()
    model_context["known_a_oracle"] = model_context["a"]
    model_context["known_b_oracle"] = model_context["b"]
    data = ZoneData(
        zone="FR",
        timezone="Europe/Paris",
        frequency="h",
        target=target,
        covariates=covariates,
        model_context_covariates=model_context,
        known_future_columns=["known_a_oracle", "known_b_oracle"],
        coverage=pd.DataFrame(),
        input_manifest=pd.DataFrame(),
        diagnostics={},
    )
    fresh = model_context.loc[
        future_index, ["known_a_oracle", "known_b_oracle"]
    ].copy()
    q50 = np.full(24, 56.0)
    autonomous = pd.DataFrame(
        {"q10": q50 - 5.0, "q50": q50, "q90": q50 + 5.0},
        index=future_index,
    )
    return data, fresh, autonomous


def test_grouped_shapley_explains_autonomous_and_mkonline_blend(
    tmp_path: Path,
) -> None:
    data, fresh, autonomous = _fixture()
    forecast_path = tmp_path / "forecast_hourly_fr.csv"
    primary = pd.Series(100.0, index=fresh.index)
    official_blend = 0.6 * autonomous["q50"] + 0.4 * primary
    pd.DataFrame(
        {
            "delivery_start_utc": fresh.index,
            "q10": official_blend - 5.0,
            "q50": official_blend,
            "q90": official_blend + 5.0,
        }
    ).to_csv(forecast_path, index=False)
    frozen_hash = _sha256(forecast_path)

    audit = write_variable_attribution(
        output_dir=tmp_path,
        forecast_path=forecast_path,
        data=data,
        runtime=SimpleNamespace(pipeline=_InteractionPipeline()),
        fresh_future=fresh,
        corrector=_IdentityCorrector(),
        official_autonomous=autonomous,
        required_covariates=("a", "b"),
        context_length=168,
        model_batch_size=8,
        zone="FR",
        timezone="Europe/Paris",
        delivery_day="2026-08-13",
        official_blend=official_blend,
        primary=primary,
        autonomous_weight=0.6,
        mkonline_weight=0.4,
        reproduction_tolerance_eur_mwh=1e-6,
    )

    assert _sha256(forecast_path) == frozen_hash
    assert audit["method"] == "exact_grouped_shapley_end_to_end"
    assert audit["scenario_count"] == 8
    assert audit["past_prices_included"] is True
    assert audit["used_for_prediction"] is False
    assert audit["storm_used"] is False
    assert audit["architecture_weights"]["mkonline_blend"] == {
        "autonomous": 0.6,
        "mkonline_primary": 0.4,
    }

    hourly = pd.read_csv(tmp_path / VARIABLE_ATTRIBUTION_HOURLY)
    autonomous_rows = hourly.loc[hourly["variant"].eq("autonomous")]
    blend_rows = hourly.loc[hourly["variant"].eq("mkonline_blend")]
    expected_autonomous = {"a": 22.0, "b": 14.0}
    for key, expected in expected_autonomous.items():
        auto = autonomous_rows.loc[autonomous_rows["variable_key"].eq(key)]
        blend = blend_rows.loc[blend_rows["variable_key"].eq(key)]
        assert np.allclose(auto["contribution_eur_mwh"], expected)
        assert np.allclose(blend["contribution_eur_mwh"], 0.6 * expected)
    for block in (autonomous_rows, blend_rows):
        reconstructed = block.groupby("delivery_start_utc").agg(
            forecast=("forecast_q50", "first"),
            baseline=("counterfactual_q50", "first"),
            contribution=("contribution_eur_mwh", "sum"),
        )
        assert np.allclose(
            reconstructed["forecast"],
            reconstructed["baseline"] + reconstructed["contribution"],
        )
        weights = block.groupby("variable_key")["weight_pct"].first()
        assert weights.sum() == pytest.approx(100.0)

    persisted_audit = json.loads(
        (tmp_path / VARIABLE_ATTRIBUTION_AUDIT).read_text(encoding="utf-8")
    )
    assert persisted_audit["forecast_sha256"] == frozen_hash
    assert persisted_audit["max_reconstruction_error_eur_mwh"] <= 1e-6


def test_variable_groups_follow_new_configured_series_and_reject_comparators() -> None:
    groups = build_variable_groups(
        required_covariates=("fr_load_fcst", "new_hydro_fcst"),
        context_columns=(
            "fr_load_fcst",
            "known_fr_load_fcst_oracle",
            "new_hydro_fcst",
            "known_new_hydro_fcst_oracle",
        ),
        future_columns=(
            "calendar_hour",
            "known_fr_load_fcst_oracle",
            "known_new_hydro_fcst_oracle",
        ),
    )
    assert [group.key for group in groups] == ["fr_load_fcst", "new_hydro_fcst", PAST_PRICE_GROUP_KEY]
    assert groups[1].future_columns == ("known_new_hydro_fcst_oracle",)
    assert groups[-1].target_context is True
    assert groups[-1].context_columns == ("target",)
    assert groups[-1].future_columns == ()

    with pytest.raises(VariableAttributionError, match="comparator"):
        build_variable_groups(
            required_covariates=("storm_dashboard",),
            context_columns=("storm_dashboard",),
            future_columns=(),
        )


def test_permutation_shapley_is_deterministic_and_additive() -> None:
    masks, orders, method = _coalition_masks(
        7, seed=17, approximate_permutations=20
    )
    repeated_masks, repeated_orders, _ = _coalition_masks(
        7, seed=17, approximate_permutations=20
    )
    assert (masks, orders) == (repeated_masks, repeated_orders)
    assert method == "permutation_grouped_shapley_end_to_end"
    values = {
        mask: np.asarray(
            [sum((index + 1) for index in range(7) if mask & (1 << index))],
            dtype=float,
        )
        for mask in masks
    }
    contributions = _shapley_values(values, n_groups=7, orders=orders)
    assert np.allclose(contributions[:, 0], np.arange(1.0, 8.0))
    assert contributions[:, 0].sum() == pytest.approx(values[(1 << 7) - 1][0])


def test_optional_artifact_cleanup_is_scoped(tmp_path: Path) -> None:
    keep = tmp_path / "forecast_hourly_fr.csv"
    keep.write_text("forecast", encoding="utf-8")
    (tmp_path / VARIABLE_ATTRIBUTION_HOURLY).write_text("partial", encoding="utf-8")
    (tmp_path / VARIABLE_ATTRIBUTION_AUDIT).write_text("{}", encoding="utf-8")

    remove_variable_attribution_artifacts(tmp_path)

    assert keep.is_file()
    assert not (tmp_path / VARIABLE_ATTRIBUTION_HOURLY).exists()
    assert not (tmp_path / VARIABLE_ATTRIBUTION_AUDIT).exists()


class _PastPricePipeline(_InteractionPipeline):
    """Frozen toy model exposing an observable historical-target dependence."""
    target_weight = 0.2

    def predict_df(self, context_df, *, future_df, prediction_length, **kwargs):
        assert "target" in context_df
        assert "target" not in future_df
        self.last_context = context_df.copy(deep=True)
        self.last_future = future_df.copy(deep=True)
        result = super().predict_df(
            context_df, future_df=future_df, prediction_length=prediction_length, **kwargs,
        )
        for item_id in result.item_id.unique():
            target_last = context_df.loc[context_df.item_id.eq(item_id), "target"].iloc[-1]
            for column in ("predictions", "0.1", "0.5", "0.9"):
                result.loc[result.item_id.eq(item_id), column] += self.target_weight * target_last
        return result


class _FrozenPriceCorrector:
    lag_weight = 0.1
    rolling_weight = 0.05

    def __init__(self):
        self.frames = []

    def fit(self, *args, **kwargs):
        raise AssertionError("Attribution must never refit the frozen corrector")

    def predict(self, fresh, base, experts):
        self.frames.append(fresh.copy(deep=True))
        shift = self.lag_weight * fresh.price_lag_24h + self.rolling_weight * fresh.price_rolling_mean_24h
        return base.add(shift, axis=0)


def _past_price_fixture():
    data, fresh, _ = _fixture()
    data.target.iloc[-48:] = 80.0
    columns = ("price_lag_24h", "price_rolling_mean_24h")
    derived = _target_price_derivatives(
        data.target, output_index=data.model_context_covariates.index,
        columns=columns, timezone="Europe/Paris",
    )
    data.model_context_covariates[list(columns)] = derived
    fresh[list(columns)] = derived.loc[fresh.index]
    # Calendar is present but is not attributed or changed by coalitions.
    fresh["calendar_dayofyear_sin"] = 0.75
    official = pd.DataFrame({"q10": 79.0, "q50": 84.0, "q90": 89.0}, index=fresh.index)
    return data, fresh, official


def _write_past_price_test(tmp_path, *, data, fresh, official, pipeline, corrector, **options):
    path = tmp_path / "forecast_hourly_fr.csv"
    published = official.copy()
    published.index.name = "delivery_start_utc"
    published.to_csv(path)
    return write_variable_attribution(
        output_dir=tmp_path, forecast_path=path, data=data,
        runtime=SimpleNamespace(pipeline=pipeline), fresh_future=fresh,
        corrector=corrector, official_autonomous=official,
        required_covariates=("a", "b"), context_length=168, model_batch_size=8,
        zone="FR", timezone="Europe/Paris", delivery_day="2026-08-13",
        reproduction_tolerance_eur_mwh=1e-6, **options,
    )


def test_past_target_group_changes_chronos_context_and_consistent_price_derivatives(tmp_path):
    data, fresh, official = _past_price_fixture()
    target_before = data.target.copy(deep=True)
    covariates_before = data.covariates.copy(deep=True)
    context_before = data.model_context_covariates.copy(deep=True)
    fresh_before = fresh.copy(deep=True)
    pipeline, corrector = _PastPricePipeline(), _FrozenPriceCorrector()
    audit = _write_past_price_test(
        tmp_path, data=data, fresh=fresh, official=official,
        pipeline=pipeline, corrector=corrector,
    )
    groups = {group["key"]: group for group in audit["groups"]}
    assert groups[PAST_PRICE_GROUP_KEY]["target_context"] is True
    assert groups[PAST_PRICE_GROUP_KEY]["future_columns"] == ["price_lag_24h", "price_rolling_mean_24h"]
    assert "calendar" not in groups
    assert audit["baseline"]["fixed_background_inputs"] == ["calendar"]
    assert audit["baseline"]["past_price_baseline_uses_future_targets"] is False
    assert audit["parameters_frozen"]["refitting_performed"] is False
    assert audit["kalman_scope"]["included"] is False
    assert pipeline.target_weight == 0.2
    assert corrector.lag_weight == 0.1 and corrector.rolling_weight == 0.05
    assert all(frame.calendar_dayofyear_sin.eq(0.75).all() for frame in corrector.frames)
    assert {float(frame.price_lag_24h.iloc[0]) for frame in corrector.frames} == {50.0, 80.0}
    assert {float(frame.price_rolling_mean_24h.iloc[0]) for frame in corrector.frames} == {50.0, 80.0}
    # Masks 3/7 retain both physical inputs and differ only in past prices.
    ids = pipeline.last_context.item_id.unique()
    without_prices = next(item for item in ids if item.endswith("_003"))
    full = next(item for item in ids if item.endswith("_007"))
    baseline_context = pipeline.last_context.loc[pipeline.last_context.item_id.eq(without_prices)]
    full_context = pipeline.last_context.loc[pipeline.last_context.item_id.eq(full)]
    assert baseline_context.target.iloc[-1] == 50.0
    assert full_context.target.iloc[-1] == 80.0
    assert baseline_context.price_lag_24h.iloc[-1] == 50.0
    assert full_context.price_lag_24h.iloc[-1] == 80.0
    pd.testing.assert_frame_equal(
        pipeline.last_future.loc[pipeline.last_future.item_id.eq(without_prices)].drop(columns="item_id").reset_index(drop=True),
        pipeline.last_future.loc[pipeline.last_future.item_id.eq(full)].drop(columns="item_id").reset_index(drop=True),
    )
    rows = pd.read_csv(tmp_path / VARIABLE_ATTRIBUTION_HOURLY)
    assert np.allclose(rows.loc[rows.variable_key.eq(PAST_PRICE_GROUP_KEY), "contribution_eur_mwh"], 10.5)
    reconstruction = rows.groupby("delivery_start_utc").agg(
        baseline=("counterfactual_q50", "first"), contribution=("contribution_eur_mwh", "sum"),
    )
    assert np.allclose(reconstruction.baseline + reconstruction.contribution, 84.0)
    assert audit["max_base_prediction_error_eur_mwh"] <= 1e-6
    pd.testing.assert_series_equal(data.target, target_before)
    pd.testing.assert_frame_equal(data.covariates, covariates_before)
    pd.testing.assert_frame_equal(data.model_context_covariates, context_before)
    pd.testing.assert_frame_equal(fresh, fresh_before)
    # Schema v1 and existing labels/columns remain consumable without a report
    # or reporting module change, despite the additional group/audit fields.
    from chronos2_hourly.reporting import _attach_variable_attribution
    report_result = SimpleNamespace(zone="FR", forecast_native=pd.DataFrame({
        "timestamp": fresh.index, "q50": official.q50.to_numpy(),
    }))
    _attach_variable_attribution(report_result, directory=tmp_path,
                                 native_model="residual_corrected", timezone="Europe/Paris")
    assert PAST_PRICE_GROUP_KEY in {group["key"] for group in report_result.variable_attribution["groups"]}


def test_future_target_observations_cannot_influence_the_past_price_reference(tmp_path):
    data, fresh, official = _past_price_fixture()
    end = data.target.index[-1]
    baseline = _causal_profile(data.target, historical_end=end, output_index=data.target.index, timezone="Europe/Paris")
    poisoned = pd.concat([data.target, pd.Series(1e9, index=fresh.index)])
    baseline_poisoned = _causal_profile(poisoned, historical_end=end, output_index=data.target.index, timezone="Europe/Paris")
    np.testing.assert_array_equal(baseline, baseline_poisoned)
    data.target = poisoned
    with pytest.raises(VariableAttributionError, match="end exactly one hour before delivery"):
        _write_past_price_test(tmp_path, data=data, fresh=fresh, official=official,
                               pipeline=_PastPricePipeline(), corrector=_FrozenPriceCorrector())
    assert not (tmp_path / VARIABLE_ATTRIBUTION_HOURLY).exists()


def test_full_coalition_must_still_reproduce_published_p50(tmp_path):
    data, fresh, official = _past_price_fixture()
    official["q50"] += 1.0
    with pytest.raises(VariableAttributionError, match="does not reproduce frozen forecast"):
        _write_past_price_test(tmp_path, data=data, fresh=fresh, official=official,
                               pipeline=_PastPricePipeline(), corrector=_FrozenPriceCorrector())
    assert not (tmp_path / VARIABLE_ATTRIBUTION_HOURLY).exists()


def test_physical_only_legacy_scope_remains_available(tmp_path):
    data, fresh, official = _past_price_fixture()
    audit = _write_past_price_test(tmp_path, data=data, fresh=fresh, official=official,
                                  pipeline=_PastPricePipeline(), corrector=_FrozenPriceCorrector(),
                                  include_past_prices=False)
    assert audit["past_prices_included"] is False
    assert audit["scenario_count"] == 4
    assert audit["baseline"]["fixed_background_inputs"] == ["historical_target_price", "calendar"]


def test_expanded_attribution_cost_is_bounded_and_unknown_price_derivatives_fail_closed():
    masks, orders, method = _coalition_masks(32, seed=42, approximate_permutations=128)
    assert len(masks) <= MAX_ATTRIBUTION_SCENARIOS
    assert len(orders) == 8
    assert method == "permutation_grouped_shapley_end_to_end"
    with pytest.raises(VariableAttributionError, match="unsupported target-derived"):
        build_variable_groups(required_covariates=("a",), context_columns=("a",),
                              future_columns=("price_rolling_unknown_24h",))
    with pytest.raises(VariableAttributionError, match="context-only"):
        build_variable_groups(required_covariates=("a",), context_columns=("a",), future_columns=("target",))
    with pytest.raises(VariableAttributionError, match="calendar stays fixed"):
        build_variable_groups(required_covariates=("calendar_hour",), context_columns=("calendar_hour",), future_columns=())


def test_past_price_derivatives_keep_production_dst_mask_and_use_no_same_day_label():
    future = pd.date_range("2026-10-25", "2026-10-26", tz="Europe/Paris",
                           freq="h", inclusive="left").tz_convert("UTC")
    assert len(future) == 25
    history = pd.date_range(end=future[0] - pd.Timedelta(hours=1), periods=56 * 24,
                            tz="UTC", freq="h")
    target = pd.Series(50.0, index=history)
    result = _target_price_derivatives(
        target, output_index=future, timezone="Europe/Paris",
        columns=("price_lag_24h", "price_rolling_mean_24h", "price_rolling_std_24h"),
    )
    assert result.price_lag_24h.iloc[:24].eq(50.0).all()
    assert pd.isna(result.price_lag_24h.iloc[24])
    assert result.price_rolling_mean_24h.eq(50.0).all()
    assert result.price_rolling_std_24h.eq(0.0).all()
    assert "target" not in result
