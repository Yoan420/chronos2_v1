from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import chronos2_exogenous.zone_artifacts as zone_artifacts
from chronos2_exogenous.governance import (
    ExogenousGovernanceError,
    REQUIRED_EXPERIMENT_FLAGS,
    validate_experiment_manifest,
)
from chronos2_exogenous.lora_finetune import (
    ExogenousFineTuneError,
    sha256_directory,
    verify_bundle,
)
from chronos2_exogenous.oof_residual import (
    ExogenousOofResidualError,
    _zone_source_cache_identity,
)
from chronos2_exogenous.zone_artifacts import (
    ZONE_COPY_MANIFEST_NAME,
    ZONE_PROVENANCE_KEY,
    ZoneArtifactPreparationError,
    prepare_zone_artifacts,
)
from run_chronos2_exogenous_prepare_zones import main as prepare_zones_main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _source_bundle(
    tmp_path: Path,
    *,
    name: str = "artifact",
    evaluation_role: str = "primary_predeclared",
) -> Path:
    source = tmp_path / "experiment" / name
    checkpoint = source / "checkpoint"
    checkpoint.mkdir(parents=True)
    _write_json(checkpoint / "adapter_config.json", {"peft_type": "LORA"})
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    retained = source / "checkpoint-500"
    retained.mkdir()
    _write_json(retained / "trainer_state.json", {"global_step": 500})
    (retained / "training_args.bin").write_bytes(b"training-args")
    schema = source / "schema.json"
    _write_json(
        schema,
        {
            "format_version": 1,
            "target_columns": ["target"],
            "timezone": "Europe/Paris",
            "cutoff_local_time": "08:00",
        },
    )
    panel_sha = "a" * 64
    panel_audit = tmp_path / "inputs" / "training_panel.parquet.audit.json"
    _write_json(
        panel_audit,
        {
            "schema_version": 1,
            "layout": "per_zone",
            "zones": ["FR", "DE", "BE", "NL"],
            "panel_sha256": panel_sha,
        },
    )
    manifest = {
        "format_version": 1,
        "experiment_id": "multi-zone-lora-v1",
        "evaluation_role": evaluation_role,
        "model_id": "amazon/chronos-2",
        "finetune_mode": "lora",
        "training_window_days": 365,
        "evaluation_days": 365,
        "cutoff_local_time": "08:00",
        "candidate_frozen_before_evaluation": True,
        "feature_selection_frozen_before_evaluation": True,
        "actual_future_used_as_input": False,
        "storm_used_for_input": False,
        "storm_used_for_selection": False,
        "mkonline_used_for_input": False,
        "pit_audit_passed": True,
        "production_pit_evidence": evaluation_role == "primary_predeclared",
        "production_pipeline_evidence": False,
        "checkpoint_sha256": sha256_directory(checkpoint),
        "schema_sha256": _sha256(schema),
        "panel_sha256": panel_sha,
        "panel_audit_sha256": _sha256(panel_audit),
        "panel_audit_summary": {
            "audit_path": str(panel_audit.resolve()),
            "zones": ["FR", "DE", "BE", "NL"],
        },
        "pit_audit": {"items": ["BE", "DE", "FR", "NL"]},
    }
    _write_json(source / "experiment_manifest.json", manifest)
    verify_bundle(source)
    return source


def _source_bytes(source: Path) -> dict[str, bytes]:
    return {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


def test_prepare_zone_artifacts_copies_complete_bundle_and_binds_each_zone(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path)
    before = _source_bytes(source)

    result = prepare_zone_artifacts(source, zones=["fr", "DE"])

    assert result.source_directory == source.resolve()
    assert [item.zone for item in result.artifacts] == ["FR", "DE"]
    assert all(item.created for item in result.artifacts)
    assert _source_bytes(source) == before
    for item in result.artifacts:
        assert item.path == source.parent / "zones" / item.zone / "artifact"
        assert (item.path / "checkpoint-500" / "trainer_state.json").is_file()
        clone = verify_bundle(item.path)
        assert clone["zone"] == item.zone
        provenance = clone[ZONE_PROVENANCE_KEY]
        assert provenance["zone"] == item.zone
        assert provenance["source_experiment_manifest_sha256"] == _sha256(
            source / "experiment_manifest.json"
        )
        seal = json.loads(
            (item.path / ZONE_COPY_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert seal["shared_hardlinks_with_source"] is False
        assert seal["clone_tree_entries"] > 0
        for relative in before:
            assert not os.path.samefile(source / relative, item.path / relative)


def test_prepared_zones_infer_one_authenticated_shared_oof_cache(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path)
    prepared = prepare_zone_artifacts(source, zones=["FR", "DE"])

    resolved = [
        _zone_source_cache_identity(item.path, verify_bundle(item.path))
        for item in prepared.artifacts
    ]
    assert all(value is not None for value in resolved)
    assert {value[0] for value in resolved if value is not None} == {
        source.parent / "shared_oof_fold_checkpoints"
    }
    assert {
        value[1]["source_experiment_manifest_sha256"]
        for value in resolved
        if value is not None
    } == {_sha256(source / "experiment_manifest.json")}

    seal_path = prepared.artifacts[0].path / ZONE_COPY_MANIFEST_NAME
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["source_tree_sha256"] = "f" * 64
    _write_json(seal_path, seal)
    with pytest.raises(ExogenousOofResidualError, match="Sceau PrepareZones divergent"):
        _zone_source_cache_identity(
            prepared.artifacts[0].path,
            verify_bundle(prepared.artifacts[0].path),
        )


def test_prepare_zone_artifacts_is_idempotent_only_while_seal_is_exact(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path)
    first = prepare_zone_artifacts(source, zones=["FR"])
    second = prepare_zone_artifacts(source, zones=["FR"])
    assert first.artifacts[0].created is True
    assert second.artifacts[0].created is False

    destination = first.artifacts[0].path
    (destination / "unexpected.txt").write_text("mutation", encoding="utf-8")
    with pytest.raises(ZoneArtifactPreparationError, match="modifiee ou completee"):
        prepare_zone_artifacts(source, zones=["FR"])


def test_prepare_zone_artifacts_rejects_hardlinks_shared_between_zone_clones(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path)
    prepared = prepare_zone_artifacts(source, zones=["FR", "DE"])
    destinations = {artifact.zone: artifact.path for artifact in prepared.artifacts}
    de_schema = destinations["DE"] / "schema.json"
    de_schema.unlink()
    de_schema.hardlink_to(destinations["FR"] / "schema.json")

    with pytest.raises(ZoneArtifactPreparationError, match="Hardlink.*zones"):
        prepare_zone_artifacts(source, zones=["FR", "DE"])


def test_prepare_zone_artifacts_rejects_zone_absent_from_panel_before_writing(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path)

    with pytest.raises(ZoneArtifactPreparationError, match="ES.*absentes|absentes.*ES"):
        prepare_zone_artifacts(source, zones=["FR", "ES"])

    assert not (source.parent / "zones").exists()


def test_prepare_zone_artifacts_rejects_active_temporary_directory(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path, name=".artifact.tmp-training")

    with pytest.raises(ZoneArtifactPreparationError, match="temporaire/en cours"):
        prepare_zone_artifacts(source, zones=["FR"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"zone": "FR"}, "deja liee"),
        ({"production_pipeline_evidence": True}, "avant preuve finale"),
        (
            {"evaluation_evidence": {"relative_path": "evaluation_predictions.csv.gz"}},
            "n'est plus un artefact d'entrainement vierge",
        ),
    ],
)
def test_prepare_zone_artifacts_rejects_non_pristine_source(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    source = _source_bundle(tmp_path)
    manifest_path = source / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(mutation)
    _write_json(manifest_path, manifest)

    with pytest.raises(ZoneArtifactPreparationError, match=message):
        prepare_zone_artifacts(source, zones=["FR"])


def test_prepare_zone_artifacts_rolls_back_all_new_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_bundle(tmp_path)
    real_commit = zone_artifacts._commit_staging
    calls = 0

    def fail_second(staging: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated commit failure")
        real_commit(staging, destination)

    monkeypatch.setattr(zone_artifacts, "_commit_staging", fail_second)
    with pytest.raises(OSError, match="simulated commit failure"):
        prepare_zone_artifacts(source, zones=["FR", "DE"])

    assert not (source.parent / "zones" / "FR" / "artifact").exists()
    assert not (source.parent / "zones" / "DE" / "artifact").exists()
    assert not list((source.parent / "zones").rglob("*.prepare-*"))


def test_prepare_zone_artifacts_cli_reports_created_and_reused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    output_root = tmp_path / "published"

    assert (
        prepare_zones_main(
            [
                "--source-run-directory",
                str(source),
                "--zones",
                "FR",
                "DE",
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert [item["created"] for item in first["artifacts"]] == [True, True]
    assert Path(first["artifacts"][0]["path"]) == output_root / "zones" / "FR" / "artifact"

    assert (
        prepare_zones_main(
            [
                "--source-run-directory",
                str(source),
                "--zones",
                "FR",
                "DE",
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)
    assert [item["created"] for item in second["artifacts"]] == [False, False]


@pytest.mark.parametrize("explicit_diagnostic_flags", [False, True])
def test_diagnostic_four_zone_copies_keep_source_contract_and_shared_cache_identity(
    tmp_path: Path, explicit_diagnostic_flags: bool
) -> None:
    source = _source_bundle(tmp_path, evaluation_role="diagnostic_only")
    manifest_path = source / "experiment_manifest.json"
    original = verify_bundle(source)
    if explicit_diagnostic_flags:
        original.update(
            diagnostic_only=True, promotion_eligible=False, activation_performed=False
        )
        _write_json(manifest_path, original)
    source_before = _source_bytes(source)
    manifest_sha = _sha256(manifest_path)

    prepared = prepare_zone_artifacts(source, zones=["FR", "DE", "BE", "NL"])

    assert [item.zone for item in prepared.artifacts] == ["FR", "DE", "BE", "NL"]
    assert all(item.created for item in prepared.artifacts)
    cache_identities = []
    clone_snapshots = {}
    for item in prepared.artifacts:
        clone = verify_bundle(item.path)
        assert clone["evaluation_role"] == "diagnostic_only"
        assert clone["production_pit_evidence"] is False
        assert clone["production_pipeline_evidence"] is False
        assert clone["checkpoint_sha256"] == original["checkpoint_sha256"]
        assert clone["schema_sha256"] == original["schema_sha256"]
        assert clone[ZONE_PROVENANCE_KEY]["source_experiment_manifest_sha256"] == manifest_sha
        restored = dict(clone)
        assert restored.pop("zone") == item.zone
        restored.pop(ZONE_PROVENANCE_KEY)
        assert restored == original
        for relative, contents in source_before.items():
            assert not os.path.samefile(source / relative, item.path / relative)
            if relative != "experiment_manifest.json":
                assert (item.path / relative).read_bytes() == contents
        with pytest.raises(ExogenousGovernanceError, match="diagnostic_only"):
            validate_experiment_manifest(clone, zone=item.zone)
        cache_identities.append(_zone_source_cache_identity(item.path, clone))
        clone_snapshots[item.zone] = _source_bytes(item.path)

    assert all(identity is not None for identity in cache_identities)
    assert all(identity == cache_identities[0] for identity in cache_identities)
    cache_root, identity = cache_identities[0]
    assert cache_root == source.parent / "shared_oof_fold_checkpoints"
    assert not cache_root.exists()
    assert identity["source_experiment_manifest_sha256"] == manifest_sha
    reused = prepare_zone_artifacts(source, zones=["FR", "DE", "BE", "NL"])
    assert all(not item.created for item in reused.artifacts)
    assert all(_source_bytes(item.path) == clone_snapshots[item.zone] for item in reused.artifacts)
    assert _source_bytes(source) == source_before
    assert _sha256(manifest_path) == manifest_sha


def test_primary_copy_still_calls_the_original_governance_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_bundle(tmp_path)
    validated_roles = []

    def record_validation(manifest, *, zone):
        validated_roles.append(manifest["evaluation_role"])
        return validate_experiment_manifest(manifest, zone=zone)

    monkeypatch.setattr(zone_artifacts, "validate_experiment_manifest", record_validation)
    prepare_zone_artifacts(source, zones=["FR"])
    assert len(validated_roles) >= 3
    assert set(validated_roles) == {"primary_predeclared"}


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        (key, not expected if isinstance(expected, bool) else "invalid")
        for key, expected in REQUIRED_EXPERIMENT_FLAGS.items()
    ]
    + [
        (key, int(expected))
        for key, expected in REQUIRED_EXPERIMENT_FLAGS.items()
        if isinstance(expected, bool)
    ],
)
def test_diagnostic_copy_keeps_every_causal_and_freeze_check(
    tmp_path: Path, key: str, invalid: object
) -> None:
    source = _source_bundle(tmp_path, evaluation_role="diagnostic_only")
    manifest = verify_bundle(source)
    manifest[key] = invalid
    _write_json(source / "experiment_manifest.json", manifest)
    before = _source_bytes(source)

    with pytest.raises(ZoneArtifactPreparationError, match="causales/freeze"):
        prepare_zone_artifacts(source, zones=["FR", "DE", "BE", "NL"])

    assert not (source.parent / "zones").exists()
    assert _source_bytes(source) == before


@pytest.mark.parametrize(
    ("key", "invalid"),
    [(key, invalid) for key in ("production_pit_evidence", "production_pipeline_evidence")
     for invalid in (None, "false", 0, 1)]
    + [
        ("production_pipeline_evidence", True),
        ("diagnostic_only", False),
        ("diagnostic_only", 1),
        ("promotion_eligible", True),
        ("promotion_eligible", 0),
        ("activation_performed", True),
        ("activation_performed", 0),
    ],
)
def test_diagnostic_copy_rejects_inconsistent_production_flags_before_publication(
    tmp_path: Path, key: str, invalid: object
) -> None:
    source = _source_bundle(tmp_path, evaluation_role="diagnostic_only")
    manifest = verify_bundle(source)
    manifest[key] = invalid
    _write_json(source / "experiment_manifest.json", manifest)
    before = _source_bytes(source)

    with pytest.raises(ZoneArtifactPreparationError):
        prepare_zone_artifacts(source, zones=["FR", "DE", "BE", "NL"])

    assert not (source.parent / "zones").exists()
    assert _source_bytes(source) == before


@pytest.mark.parametrize("role", [None, "", "diagnostic", "shadow", 1])
def test_copy_refuses_unknown_evaluation_roles_without_publication(
    tmp_path: Path, role: object
) -> None:
    source = _source_bundle(tmp_path)
    manifest = verify_bundle(source)
    if role is None:
        manifest.pop("evaluation_role")
    else:
        manifest["evaluation_role"] = role
    _write_json(source / "experiment_manifest.json", manifest)
    before = _source_bytes(source)

    with pytest.raises(ZoneArtifactPreparationError, match="evaluation_role"):
        prepare_zone_artifacts(source, zones=["FR", "DE"])

    assert not (source.parent / "zones").exists()
    assert _source_bytes(source) == before


@pytest.mark.parametrize("model_id", [None, "", "  ", 1])
def test_diagnostic_copy_refuses_invalid_model_ids(tmp_path: Path, model_id: object) -> None:
    source = _source_bundle(tmp_path, evaluation_role="diagnostic_only")
    manifest = verify_bundle(source)
    manifest["model_id"] = model_id
    _write_json(source / "experiment_manifest.json", manifest)

    with pytest.raises(ZoneArtifactPreparationError, match="model_id"):
        prepare_zone_artifacts(source, zones=["FR"])

    assert not (source.parent / "zones").exists()


@pytest.mark.parametrize("relative", ["schema.json", "checkpoint/adapter_model.safetensors"])
def test_diagnostic_copy_verifies_source_checksums_before_any_publication(
    tmp_path: Path, relative: str
) -> None:
    source = _source_bundle(tmp_path, evaluation_role="diagnostic_only")
    path = source / relative
    path.write_bytes(path.read_bytes() + b" ")
    before = _source_bytes(source)

    with pytest.raises(ExogenousFineTuneError):
        prepare_zone_artifacts(source, zones=["FR", "DE", "BE", "NL"])

    assert not (source.parent / "zones").exists()
    assert _source_bytes(source) == before


def test_diagnostic_copy_does_not_overwrite_modified_clone_or_publish_missing_zones(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path, evaluation_role="diagnostic_only")
    prepared = prepare_zone_artifacts(source, zones=["FR"])
    destination = prepared.artifacts[0].path
    (destination / "unexpected.txt").write_text("mutation", encoding="utf-8")
    clone_before = _source_bytes(destination)
    source_before = _source_bytes(source)

    with pytest.raises(ZoneArtifactPreparationError, match="modifiee ou completee"):
        prepare_zone_artifacts(source, zones=["DE", "FR", "BE", "NL"])

    assert _source_bytes(destination) == clone_before
    assert _source_bytes(source) == source_before
    assert not (source.parent / "zones" / "DE").exists()
    assert not (source.parent / "zones" / "BE").exists()
    assert not (source.parent / "zones" / "NL").exists()


@pytest.mark.parametrize("mutation", ["missing_zone", "sidecar_sha"])
def test_diagnostic_copy_preserves_panel_inventory_and_sidecar_audit_checks(
    tmp_path: Path, mutation: str
) -> None:
    source = _source_bundle(tmp_path, evaluation_role="diagnostic_only")
    if mutation == "sidecar_sha":
        manifest = verify_bundle(source)
        audit_path = Path(manifest["panel_audit_summary"]["audit_path"])
        audit_path.write_bytes(audit_path.read_bytes() + b" ")
        zones = ["FR", "DE"]
    else:
        zones = ["FR", "ES"]
    source_before = _source_bytes(source)

    with pytest.raises(ZoneArtifactPreparationError):
        prepare_zone_artifacts(source, zones=zones)

    assert not (source.parent / "zones").exists()
    assert _source_bytes(source) == source_before


def test_diagnostic_prepare_zones_cli_copies_four_zones_and_reuses_exactly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path, evaluation_role="diagnostic_only")
    source_before = _source_bytes(source)
    args = ["--source-run-directory", str(source), "--zones", "FR", "DE", "BE", "NL"]

    for created in (True, False):
        assert prepare_zones_main(args) == 0
        result = json.loads(capsys.readouterr().out)
        assert [item["zone"] for item in result["artifacts"]] == ["FR", "DE", "BE", "NL"]
        assert all(item["created"] is created for item in result["artifacts"])
        for item in result["artifacts"]:
            clone = verify_bundle(Path(item["path"]))
            with pytest.raises(ExogenousGovernanceError, match="diagnostic_only"):
                validate_experiment_manifest(clone, zone=item["zone"])
    assert _source_bytes(source) == source_before
