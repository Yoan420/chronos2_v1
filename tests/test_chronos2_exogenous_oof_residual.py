from __future__ import annotations

from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

import chronos2_exogenous.oof_residual as oof_residual
from chronos2_exogenous.oof_residual import (
    ExogenousOofResidualError,
    _SharedFoldCheckpointCache,
    _assert_plan_matches_final_split,
    _authenticate_completed_calibration_manifest,
    _cache_fold_identity,
    _claim_path,
    _commit_completed_calibration_manifest,
    _fit_fold,
    _fold_manifest_core,
    _materialize_fold_from_shared_cache,
    _prepare_shared_fold_cache,
    _process_observation,
    _publish_fold_to_shared_cache,
    _reuse_corrector_if_exact,
    _shared_fold_fit_claim,
    _validate_cached_fold,
    build_oof_plan,
)
from chronos2_exogenous.production import (
    ExogenousProductionError,
    fit_oof_residual_corrector,
)


TIMEZONE = "Europe/Paris"


def _origins(first: str, count: int) -> pd.DatetimeIndex:
    days = pd.date_range(first, periods=count, freq="D")
    return pd.DatetimeIndex(
        [pd.Timestamp(f"{day:%Y-%m-%d} 08:00", tz=TIMEZONE) for day in days]
    ).tz_convert("UTC")


def _range(values: pd.DatetimeIndex) -> dict[str, object]:
    return {
        "count": len(values),
        "first_utc": values[0].isoformat(),
        "last_utc": values[-1].isoformat(),
    }


def test_plan_is_365_warmup_365_oof_365_closed_holdout() -> None:
    origins = _origins("2023-01-01", 1095)
    plan = build_oof_plan(
        origins,
        timezone_name=TIMEZONE,
        cutoff_local_time="08:00",
        validation_days=30,
        block_days=30,
    )

    assert plan.required_origins == 1095
    assert len(plan.calibration_origins) == 365
    assert len(plan.holdout_origins) == 365
    assert len(plan.folds) == 13
    assert len(plan.folds[-1].prediction_origins) == 5
    assert plan.calibration_origins[-1] < plan.holdout_origins[0]
    for fold in plan.folds:
        assert len(fold.fit_origins) == 365
        assert len(fold.train_origins) == 335
        assert len(fold.validation_origins) == 30
        assert fold.fit_origins[-1] < fold.prediction_origins[0]
        assert fold.train_origins + fold.validation_origins == fold.fit_origins


def test_plan_reports_exact_missing_history_before_any_neural_fit() -> None:
    with pytest.raises(
        ExogenousOofResidualError,
        match=r"730 origines disponibles, 1095 requises.*il manque 365 jours",
    ):
        build_oof_plan(
            _origins("2024-01-01", 730),
            timezone_name=TIMEZONE,
            cutoff_local_time="08:00",
            validation_days=30,
        )


def test_final_checkpoint_split_must_match_the_oof_plan() -> None:
    plan = build_oof_plan(
        _origins("2023-01-01", 1095),
        timezone_name=TIMEZONE,
        cutoff_local_time="08:00",
        validation_days=30,
    )
    manifest = {
        "splits": {
            "train": _range(pd.DatetimeIndex(plan.calibration_origins[:-30])),
            "validation": _range(pd.DatetimeIndex(plan.calibration_origins[-30:])),
            "evaluation_holdout": _range(pd.DatetimeIndex(plan.holdout_origins)),
        }
    }
    _assert_plan_matches_final_split(plan, manifest=manifest, validation_days=30)

    manifest["splits"]["evaluation_holdout"]["last_utc"] = "2020-01-01T00:00:00+00:00"
    with pytest.raises(ExogenousOofResidualError, match="evaluation_holdout"):
        _assert_plan_matches_final_split(plan, manifest=manifest, validation_days=30)


def _oof_v2(tmp_path: Path) -> tuple[Path, Path]:
    all_origins = _origins("2023-01-01", 1095)
    plan = build_oof_plan(
        all_origins,
        timezone_name=TIMEZONE,
        cutoff_local_time="08:00",
        validation_days=30,
        block_days=30,
    )
    parts: list[pd.DataFrame] = []
    fold_payloads: list[dict[str, object]] = []
    checkpoint_contract: list[dict[str, object]] = []
    for fold in plan.folds:
        for origin in fold.prediction_origins:
            local_origin = origin.tz_convert(TIMEZONE)
            delivery_day = local_origin.date() + timedelta(days=1)
            start = pd.Timestamp(delivery_day, tz=TIMEZONE).tz_convert("UTC")
            end = pd.Timestamp(delivery_day + timedelta(days=1), tz=TIMEZONE).tz_convert(
                "UTC"
            )
            hours = pd.date_range(start, end, freq="h", inclusive="left")
            parts.append(
                pd.DataFrame(
                    {
                        "delivery_start_utc": hours,
                        "forecast_origin_utc": origin,
                        "actual": 12.0,
                        "candidate_q50": 10.0,
                    }
                )
            )
        checkpoint_sha = hashlib.sha256(f"fold-{fold.index}".encode()).hexdigest()
        predictions_sha = hashlib.sha256(
            f"predictions-{fold.index}".encode()
        ).hexdigest()
        fold_payloads.append(
            {
                "fold_index": fold.index,
                "fit_origins": _range(pd.DatetimeIndex(fold.fit_origins)),
                "train_origins": _range(pd.DatetimeIndex(fold.train_origins)),
                "validation_origins": _range(
                    pd.DatetimeIndex(fold.validation_origins)
                ),
                "prediction_origins": _range(
                    pd.DatetimeIndex(fold.prediction_origins)
                ),
                "checkpoint_sha256": checkpoint_sha,
                "predictions_sha256": predictions_sha,
                "refit_uses_only_strictly_prior_days": True,
            }
        )
        checkpoint_contract.append(
            {"fold_index": fold.index, "checkpoint_sha256": checkpoint_sha}
        )
    source = tmp_path / "oof.csv.gz"
    pd.concat(parts, ignore_index=True).to_csv(source, index=False, compression="gzip")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint_set_sha = hashlib.sha256(
        json.dumps(
            checkpoint_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    first_delivery = (
        plan.calibration_origins[0].tz_convert(TIMEZONE).date() + timedelta(days=1)
    )
    last_delivery = (
        plan.calibration_origins[-1].tz_convert(TIMEZONE).date() + timedelta(days=1)
    )
    holdout_start = (
        plan.holdout_origins[0].tz_convert(TIMEZONE).date() + timedelta(days=1)
    )
    audit = tmp_path / "oof.csv.gz.audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "purpose": "chronos2_exogenous_blocked_prequential_oof",
                "fit_protocol": "blocked_prequential_oof_rolling365",
                "candidate_model": "chronos2_exogenous",
                "candidate_checkpoint_sha256": "f" * 64,
                "candidate_checkpoint_role": (
                    "deployment_identity_anchor_not_oof_predictor"
                ),
                "deployment_checkpoint_used_for_oof": False,
                "fold_checkpoints_are_origin_specific": True,
                "fold_checkpoint_set_sha256": checkpoint_set_sha,
                "fold_candidate_recipe_sha256": "e" * 64,
                "fold_count": len(fold_payloads),
                "folds": fold_payloads,
                "training_days": 365,
                "fold_lookback_days": 365,
                "training_start_day": str(first_delivery),
                "training_end_day": str(last_delivery),
                "holdout_start_day": str(holdout_start),
                "predictions_sha256": digest,
                "refit_uses_only_strictly_prior_days": True,
                "same_day_actual_excluded_from_fit": True,
                "future_actuals_used_as_features": False,
                "holdout_used_for_fit": False,
                "selection_frozen_before_oof": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return source, audit


def test_v2_fold_sidecar_fits_corrector_and_keeps_dst_hours(tmp_path: Path) -> None:
    source, audit = _oof_v2(tmp_path)
    output = fit_oof_residual_corrector(
        oof_predictions_path=source,
        oof_audit_path=audit,
        holdout_start_day=json.loads(audit.read_text())["holdout_start_day"],
        output_path=tmp_path / "corrector.json",
        feature_columns=["intercept"],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["oof_sidecar_schema_version"] == 2
    assert len(payload["oof_fold_checkpoint_set_sha256"]) == 64
    assert payload["coefficients"] == pytest.approx([2.0])


def test_v2_sidecar_rejects_fold_that_touches_prediction_block(tmp_path: Path) -> None:
    source, audit = _oof_v2(tmp_path)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["folds"][0]["fit_origins"]["last_utc"] = payload["folds"][0][
        "prediction_origins"
    ]["first_utc"]
    audit.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExogenousProductionError, match="plage fit_origins non consecutive"):
        fit_oof_residual_corrector(
            oof_predictions_path=source,
            oof_audit_path=audit,
            holdout_start_day=payload["holdout_start_day"],
            output_path=tmp_path / "forbidden.json",
        )


def _small_shared_cache(
    tmp_path: Path,
    *,
    checkpoint_anchor: str = "a" * 64,
    calibration_panel_sha: str = "c" * 64,
    learning_rate: float = 1e-5,
    origin_start: str = "2026-01-01",
) -> tuple[_SharedFoldCheckpointCache, object, dict[str, object], str]:
    origins = _origins(origin_start, 6)
    plan = build_oof_plan(
        origins,
        timezone_name=TIMEZONE,
        cutoff_local_time="08:00",
        validation_days=1,
        block_days=1,
        training_days=2,
        oof_days=2,
        holdout_days=2,
    )
    recipe: dict[str, object] = {
        "schema_version": 1,
        "training": {
            "learning_rate": learning_rate,
            "num_steps": 500,
            "batch_size": 64,
            "seed": 42,
            "lora_config": {"r": 16},
        },
    }
    recipe_sha = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "checkpoint_sha256": checkpoint_anchor,
        "panel_sha256": "b" * 64,
        "splits": {
            "train": _range(pd.DatetimeIndex(plan.calibration_origins[:-1])),
            "validation": _range(pd.DatetimeIndex(plan.calibration_origins[-1:])),
            "evaluation_holdout": _range(pd.DatetimeIndex(plan.holdout_origins)),
        },
    }
    panel_audit = {
        "upstream_panel_audit": {
            "panel_sha256": calibration_panel_sha,
            "panel_audit_sha256": "d" * 64,
        }
    }
    cache = _prepare_shared_fold_cache(
        requested_root=tmp_path / "shared",
        config=SimpleNamespace(project_root=tmp_path),
        run_directory=tmp_path / "zone" / "artifact",
        manifest=manifest,
        plan=plan,
        recipe=recipe,
        recipe_sha256=recipe_sha,
        panel_audit=panel_audit,
        block_days=1,
    )
    assert cache is not None
    return cache, plan.folds[0], recipe, recipe_sha


def _fake_checkpoint(path: Path, content: bytes = b"adapter") -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter.safetensors").write_bytes(content)
    return path


def test_shared_fold_cache_copies_only_a_sealed_checkpoint(tmp_path: Path) -> None:
    cache, fold, recipe, recipe_sha = _small_shared_cache(tmp_path)
    source = _fake_checkpoint(tmp_path / "source_checkpoint")

    seal = _publish_fold_to_shared_cache(
        cache=cache,
        fold=fold,
        timezone_name=TIMEZONE,
        checkpoint=source,
    )
    cached = _validate_cached_fold(cache=cache, fold=fold, timezone_name=TIMEZONE)
    assert cached is not None
    cached_checkpoint, validated_seal = cached
    assert validated_seal == seal
    assert {path.name for path in cached_checkpoint.parent.iterdir()} == {
        "checkpoint",
        "fold_seal.json",
    }
    assert not (cached_checkpoint.parent / "oof_predictions.csv.gz").exists()
    assert not (cached_checkpoint.parent / "residual_corrector.json").exists()

    destination = tmp_path / "zone_fr" / "fold_001"
    destination.parent.mkdir()
    materialized = _materialize_fold_from_shared_cache(
        cache=cache,
        fold=fold,
        fold_directory=destination,
        timezone_name=TIMEZONE,
        recipe=recipe,
        recipe_sha256=recipe_sha,
    )
    assert materialized is not None
    assert materialized["predictions_sha256"] is None
    assert materialized["checkpoint_materialization"]["predictions_shared"] is False
    assert materialized["checkpoint_materialization"]["corrector_shared"] is False
    assert (destination / "checkpoint" / "adapter.safetensors").read_bytes() == b"adapter"
    assert not (destination / "oof_predictions.csv.gz").exists()
    assert not (destination / "residual_corrector.json").exists()
    assert not (destination / "checkpoint" / "adapter.safetensors").samefile(
        cached_checkpoint / "adapter.safetensors"
    )


def test_shared_fold_cache_refuses_contract_or_checkpoint_tampering(
    tmp_path: Path,
) -> None:
    cache, fold, _recipe, _recipe_sha = _small_shared_cache(tmp_path)
    source = _fake_checkpoint(tmp_path / "source_checkpoint")
    _publish_fold_to_shared_cache(
        cache=cache,
        fold=fold,
        timezone_name=TIMEZONE,
        checkpoint=source,
    )
    cached = _validate_cached_fold(cache=cache, fold=fold, timezone_name=TIMEZONE)
    assert cached is not None
    (cached[0] / "adapter.safetensors").write_bytes(b"tampered")
    with pytest.raises(ExogenousOofResidualError, match="Sceau du fold cache"):
        _validate_cached_fold(cache=cache, fold=fold, timezone_name=TIMEZONE)

    contract_path = cache.contract_directory / "cache_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["deployment_checkpoint_sha256"] = "e" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ExogenousOofResidualError, match="contrat du cache OOF"):
        _prepare_shared_fold_cache(
            requested_root=cache.root,
            config=SimpleNamespace(project_root=tmp_path),
            run_directory=tmp_path / "zone" / "artifact",
            manifest={
                "checkpoint_sha256": "a" * 64,
                "panel_sha256": "b" * 64,
                "splits": cache.contract["split_contract"][
                    "final_checkpoint_splits"
                ],
            },
            plan=build_oof_plan(
                _origins("2026-01-01", 6),
                timezone_name=TIMEZONE,
                cutoff_local_time="08:00",
                validation_days=1,
                block_days=1,
                training_days=2,
                oof_days=2,
                holdout_days=2,
            ),
            recipe=cache.contract["fold_candidate_recipe"],
            recipe_sha256=cache.contract["fold_candidate_recipe_sha256"],
            panel_audit={
                "upstream_panel_audit": {
                    "panel_sha256": "c" * 64,
                    "panel_audit_sha256": "d" * 64,
                }
            },
            block_days=1,
        )


def test_shared_cache_identity_binds_anchor_panel_split_and_hyperparameters(
    tmp_path: Path,
) -> None:
    reference = _small_shared_cache(tmp_path / "reference")[0]
    changed_anchor = _small_shared_cache(
        tmp_path / "anchor", checkpoint_anchor="f" * 64
    )[0]
    changed_panel = _small_shared_cache(
        tmp_path / "panel", calibration_panel_sha="e" * 64
    )[0]
    changed_hyperparameter = _small_shared_cache(
        tmp_path / "hyper", learning_rate=2e-5
    )[0]
    changed_split = _small_shared_cache(
        tmp_path / "split", origin_start="2026-02-01"
    )[0]
    assert len(
        {
            reference.contract_sha256,
            changed_anchor.contract_sha256,
            changed_panel.contract_sha256,
            changed_hyperparameter.contract_sha256,
            changed_split.contract_sha256,
        }
    ) == 5


def test_fit_fold_reuses_shared_checkpoint_without_refitting(tmp_path: Path) -> None:
    cache, fold, recipe, recipe_sha = _small_shared_cache(tmp_path)
    source = _fake_checkpoint(tmp_path / "source_checkpoint")
    _publish_fold_to_shared_cache(
        cache=cache,
        fold=fold,
        timezone_name=TIMEZONE,
        checkpoint=source,
    )
    destination = tmp_path / "zone_be" / "fold"
    destination.parent.mkdir()

    def forbidden_loader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Le fold scelle ne doit pas etre reentraine.")

    result = _fit_fold(
        fold=fold,
        fold_directory=destination,
        panel=pd.DataFrame(),
        config=SimpleNamespace(timezone=TIMEZONE),
        model_source=tmp_path / "unused",
        recipe=recipe,
        recipe_sha256=recipe_sha,
        pipeline_loader=forbidden_loader,
        shared_cache=cache,
    )
    assert result["checkpoint_materialization"]["mode"] == (
        "physical_copy_from_sealed_shared_cache"
    )
    assert (destination / "checkpoint" / "adapter.safetensors").is_file()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completed_calibration_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    output = tmp_path / "calibration"
    output.mkdir()
    predictions = output / "oof_predictions_365.csv.gz"
    audit = output / "oof_predictions_365.csv.gz.audit.json"
    corrector = output / "residual_corrector.json"
    predictions.write_bytes(b"sealed-oof")
    audit.write_text('{"sealed": true}', encoding="utf-8")
    corrector.write_text(
        json.dumps(
            {
                "coefficients": [1.25],
                "oof_training_predictions_sha256": _sha256(predictions),
                "oof_training_audit_sha256": _sha256(audit),
            }
        ),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "purpose": "chronos2_exogenous_lora_residual_calibration",
        "status": "complete",
        "predictions_relative_path": predictions.name,
        "predictions_sha256": _sha256(predictions),
        "oof_audit_relative_path": audit.name,
        "oof_audit_sha256": _sha256(audit),
        "corrector_relative_path": corrector.name,
        "corrector_sha256": _sha256(corrector),
        "completed_at_utc": "2026-09-04T12:00:00+00:00",
    }
    return output, manifest


def test_completed_corrector_resume_authenticates_and_does_not_reseal(
    tmp_path: Path,
) -> None:
    output, manifest = _completed_calibration_fixture(tmp_path)
    corrector = output / "residual_corrector.json"
    before = corrector.read_bytes()

    _authenticate_completed_calibration_manifest(output, manifest)
    assert _reuse_corrector_if_exact(
        corrector,
        predictions_path=output / "oof_predictions_365.csv.gz",
        audit_path=output / "oof_predictions_365.csv.gz.audit.json",
        completed_manifest=manifest,
    )
    assert corrector.read_bytes() == before
    assert manifest["corrector_sha256"] == _sha256(corrector)

    manifest_path = output / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    sealed_bytes = manifest_path.read_bytes()
    proposed = {**manifest, "completed_at_utc": "2099-01-01T00:00:00+00:00"}
    _commit_completed_calibration_manifest(
        manifest_path,
        proposed,
        existing_manifest=manifest,
    )
    assert manifest_path.read_bytes() == sealed_bytes


def test_completed_corrector_tampering_is_refused_without_resealing(
    tmp_path: Path,
) -> None:
    output, manifest = _completed_calibration_fixture(tmp_path)
    manifest_before = json.dumps(manifest, sort_keys=True)
    corrector = output / "residual_corrector.json"
    payload = json.loads(corrector.read_text(encoding="utf-8"))
    payload["coefficients"] = [99.0]
    corrector.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExogenousOofResidualError, match="re-scellage silencieux refuse"):
        _authenticate_completed_calibration_manifest(output, manifest)
    assert json.dumps(manifest, sort_keys=True) == manifest_before
    assert manifest["corrector_sha256"] != _sha256(corrector)


def test_unsealed_existing_corrector_is_not_adopted_on_resume(tmp_path: Path) -> None:
    output, _manifest = _completed_calibration_fixture(tmp_path)
    with pytest.raises(ExogenousOofResidualError, match="sans manifeste complete"):
        _reuse_corrector_if_exact(
            output / "residual_corrector.json",
            predictions_path=output / "oof_predictions_365.csv.gz",
            audit_path=output / "oof_predictions_365.csv.gz.audit.json",
            completed_manifest=None,
        )


def test_live_fold_claim_wait_is_bounded_and_fail_closed(tmp_path: Path) -> None:
    cache, fold, _recipe, _recipe_sha = _small_shared_cache(tmp_path)
    with _shared_fold_fit_claim(
        cache=cache,
        fold=fold,
        timezone_name=TIMEZONE,
        wait_seconds=1.0,
        poll_seconds=0.01,
    ) as owner:
        assert owner is True
        with pytest.raises(ExogenousOofResidualError, match="Attente bornee"):
            with _shared_fold_fit_claim(
                cache=cache,
                fold=fold,
                timezone_name=TIMEZONE,
                wait_seconds=0.02,
                poll_seconds=0.005,
            ):
                pass


def test_dead_process_fold_claim_is_recovered_safely(tmp_path: Path) -> None:
    cache, fold, _recipe, _recipe_sha = _small_shared_cache(tmp_path)
    path = _claim_path(cache, fold, TIMEZONE)
    identity_sha, _identity = _cache_fold_identity(cache, fold)
    now = time.time()
    dead_process_id = 2_147_483_647
    assert _process_observation(dead_process_id)[0] is False
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "chronos2_exogenous_shared_oof_fold_claim",
                "contract_sha256": cache.contract_sha256,
                "fold_identity_sha256": identity_sha,
                "token": "dead-owner-token",
                "host": socket.gethostname(),
                "process_id": dead_process_id,
                "process_start_identity": "dead-process-start",
                "created_at_unix": now - 60.0,
                "lease_expires_at_unix": now + 3600.0,
            }
        ),
        encoding="utf-8",
    )

    with _shared_fold_fit_claim(
        cache=cache,
        fold=fold,
        timezone_name=TIMEZONE,
        wait_seconds=0.2,
        poll_seconds=0.005,
    ) as owner:
        assert owner is True
    assert not path.exists()


def test_two_zone_workers_fit_a_shared_fold_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, fold, recipe, recipe_sha = _small_shared_cache(tmp_path)
    counter = 0
    counter_lock = threading.Lock()

    def fake_local_fit(**kwargs: object) -> dict[str, object]:
        nonlocal counter
        with counter_lock:
            counter += 1
        # Keep the claim long enough for the second worker to observe it.
        time.sleep(0.15)
        destination = Path(kwargs["fold_directory"])
        checkpoint = _fake_checkpoint(destination / "checkpoint")
        payload: dict[str, object] = {
            **_fold_manifest_core(
                fold,
                recipe=recipe,
                recipe_sha256=recipe_sha,
                checkpoint_sha256=oof_residual.sha256_directory(checkpoint),
            ),
            "checkpoint_materialization": {
                "mode": "locally_fitted",
                "predictions_shared": False,
                "corrector_shared": False,
            },
            "created_at_utc": "2026-09-04T12:00:00+00:00",
        }
        (destination / "fold_manifest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    monkeypatch.setattr(oof_residual, "_fit_fold_locally", fake_local_fit)
    config = SimpleNamespace(timezone=TIMEZONE)

    def worker(zone: str) -> dict[str, object]:
        destination = tmp_path / zone / "fold"
        destination.parent.mkdir()
        return _fit_fold(
            fold=fold,
            fold_directory=destination,
            panel=pd.DataFrame(),
            config=config,
            model_source=tmp_path / "unused",
            recipe=recipe,
            recipe_sha256=recipe_sha,
            pipeline_loader=lambda *_args, **_kwargs: None,
            shared_cache=cache,
            shared_cache_wait_seconds=2.0,
            shared_cache_poll_seconds=0.01,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker, ("FR", "DE")))

    assert counter == 1
    assert sorted(
        result["checkpoint_materialization"]["mode"] for result in results
    ) == ["locally_fitted", "physical_copy_from_sealed_shared_cache"]
    assert (tmp_path / "FR" / "fold" / "checkpoint").is_dir()
    assert (tmp_path / "DE" / "fold" / "checkpoint").is_dir()
    assert not _claim_path(cache, fold, TIMEZONE).exists()
