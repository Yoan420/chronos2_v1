from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronos2_hourly import report_attribution_cache as module


def test_offline_snapshot_config_does_not_probe_adapter_on_hub(tmp_path, monkeypatch):
    import huggingface_hub
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    calls = []
    def cached_only(**kwargs):
        calls.append(kwargs)
        assert kwargs["local_files_only"] is True
        return str(snapshot / "config.json")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cached_only)
    config = {"model": {"model_id": "amazon/chronos-2", "revision": "a" * 40}}
    result = module.local_attribution_model_config(config)
    assert result["model"]["model_id"] == str(snapshot.resolve())
    assert "revision" not in result["model"]
    assert config["model"]["revision"] == "a" * 40
    assert calls[0]["revision"] == "a" * 40
    assert module.local_attribution_model_config(result) == result
    assert len(calls) == 1


def _sources_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "project"
    archive = root / "runs/live"
    frozen = root / "runs/frozen"
    config = root / "config"
    for path in (archive / "inputs", frozen / "inputs", config, root / "algorithms"):
        path.mkdir(parents=True)
    monkeypatch.setattr(module, "_ALGORITHM_SOURCES", ("algorithms/current.py",))
    (root / "algorithms/current.py").write_text("recipe = 1")
    (root / "algorithms/sealed.py").write_text("sealed_recipe = 1")
    live_config = config / "live.yaml"
    live_config.write_text(
        "live:\n  base_config: base.yaml\n  frozen_autonomous_run: ../runs/frozen\n"
        "  recipe_manifest: recipe.json\n  dependency_manifest: dependencies.json\n"
    )
    base_config = config / "base.yaml"
    base_config.write_text("model:\n  model_id: amazon/chronos-2\n  revision: '" + "a" * 40 + "'\n")
    for name in ("recipe.json", "dependencies.json", "registry.yaml"):
        (config / name).write_text("{}")
    for name in module._REPORT_INPUTS:
        (archive / "inputs" / name).write_text("archived data")
    for name in module._REFIT_INPUTS:
        (frozen / name).write_text("frozen training data")
    for name in ("forecast_hourly_fr.csv", "run_manifest.json", "chronos_live_hourly.csv"):
        (archive / name).write_text("frozen forecast")
    # Manifest-only dependencies must also enter the identity. A declared
    # report or model-weight binary must not trigger a multi-GB hash/read.
    (frozen / "extra_input.csv").write_text("other sealed data")
    (frozen / "artifact_checksums.json").write_text(json.dumps({
        "algorithm": "sha256", "artifacts": [
            {"path": "extra_input.csv", "role": "materialized_input"},
            {"path": "algorithms/sealed.py", "role": "source_code"},
            {"path": "unused_weights.safetensors", "role": "run_artifact"},
            {"path": "unused_report.html", "role": "run_artifact"},
        ],
    }))
    options = dict(archive=archive, forecast_path=archive / "forecast_hourly_fr.csv",
                   live_config=live_config, registry_path=config / "registry.yaml",
                   project_root=root)
    return options, root, archive, frozen, config


def _materializer(calls):
    def materialize(directory):
        calls.append(directory)
        (directory / module.FILES[0]).write_bytes(b"hourly attribution")
        (directory / module.FILES[1]).write_text(json.dumps({
            "groups": [{"key": "historical_target_price"}],
        }))
    return materialize


def test_sources_resolve_configs_run_data_and_project_code_without_weights(tmp_path, monkeypatch):
    options, root, archive, frozen, config = _sources_fixture(tmp_path, monkeypatch)
    sources = module.report_attribution_sources(**options)
    assert config / "base.yaml" in sources
    assert config / "registry.yaml" in sources
    assert config / "recipe.json" in sources
    assert config / "dependencies.json" in sources
    assert archive / "chronos_live_hourly.csv" in sources
    assert frozen / "inputs/chronos_oof_extended.csv.gz" in sources
    assert frozen / "extra_input.csv" in sources
    assert root / "algorithms/sealed.py" in sources
    assert not any(path.suffix in {".safetensors", ".html"} for path in sources)
    assert sources == tuple(sorted(set(sources)))


@pytest.mark.parametrize("changed", ["base", "frozen_input", "manifest_input", "algorithm", "chronos", "registry"])
def test_changed_real_dependency_invalidates_cache(tmp_path, monkeypatch, changed):
    options, root, archive, frozen, config = _sources_fixture(tmp_path, monkeypatch)
    calls = []
    cache_options = dict(root=tmp_path / "cache", materialize=_materializer(calls))
    first = module.cached_attribution(sources=module.report_attribution_sources(**options), **cache_options)
    assert module.cached_attribution(sources=module.report_attribution_sources(**options), **cache_options) == first
    assert len(calls) == 1
    affected = {
        "base": config / "base.yaml", "frozen_input": frozen / "inputs/aligned_inputs.csv.gz",
        "manifest_input": frozen / "extra_input.csv", "algorithm": root / "algorithms/sealed.py",
        "chronos": archive / "chronos_live_hourly.csv", "registry": config / "registry.yaml",
    }[changed]
    affected.write_text(affected.read_text() + "\n# changed\n")
    second = module.cached_attribution(sources=module.report_attribution_sources(**options), **cache_options)
    assert second != first and len(calls) == 2


def test_manifest_code_cannot_escape_project(tmp_path, monkeypatch):
    options, _root, _archive, frozen, _config = _sources_fixture(tmp_path, monkeypatch)
    (frozen / "artifact_checksums.json").write_text(json.dumps({
        "algorithm": "sha256", "artifacts": [{"path": "../escape.py", "role": "source_code"}],
    }))
    with pytest.raises(ValueError, match="non relatif"):
        module.report_attribution_sources(**options)


def test_missing_declared_dependency_is_not_silently_omitted(tmp_path, monkeypatch):
    options, _root, _archive, frozen, _config = _sources_fixture(tmp_path, monkeypatch)
    (frozen / "extra_input.csv").unlink()
    with pytest.raises(FileNotFoundError, match="extra_input"):
        module.report_attribution_sources(**options)
    with pytest.raises(FileNotFoundError, match="missing"):
        module.cached_attribution(root=tmp_path / "cache", sources=[tmp_path / "missing"],
                                  materialize=lambda destination: pytest.fail("must not materialize"))


def test_deleted_legacy_nonexecuted_script_does_not_block_reporting(tmp_path, monkeypatch):
    options, root, _archive, _frozen, _config = _sources_fixture(tmp_path, monkeypatch)
    (root / "algorithms/sealed.py").unlink()
    assert root / "algorithms/current.py" in module.report_attribution_sources(**options)
    (root / "algorithms/current.py").unlink()
    with pytest.raises(FileNotFoundError, match="current.py"):
        module.report_attribution_sources(**options)


def test_floating_hf_revision_tracks_small_commit_ref(tmp_path, monkeypatch):
    options, _root, _archive, _frozen, config = _sources_fixture(tmp_path, monkeypatch)
    (config / "base.yaml").write_text("model:\n  model_id: amazon/chronos-2\n")
    cache = tmp_path / "hub"
    reference = cache / "models--amazon--chronos-2/refs/main"
    reference.parent.mkdir(parents=True)
    reference.write_text("a" * 40)
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    sources = module.report_attribution_sources(**options)
    assert reference in sources
    calls = []
    cache_options = dict(root=tmp_path / "cache", materialize=_materializer(calls))
    first = module.cached_attribution(sources=sources, **cache_options)
    reference.write_text("b" * 40)
    assert module.cached_attribution(sources=module.report_attribution_sources(**options), **cache_options) != first
