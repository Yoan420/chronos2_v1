"""Architecture explanation checks: read-only configuration, no model execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import yaml

import experiment_console.architecture as architecture


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_real_project_graph_keeps_reference_data_out_of_the_forecast_chain():
    result = architecture.build_architecture(PROJECT_ROOT)
    nodes = {node["id"] for node in result["nodes"]}
    assert len(nodes) == len(result["nodes"]) == 8
    assert len(result["edges"]) == 9
    assert all(edge["source"] in nodes and edge["target"] in nodes for edge in result["edges"])
    references = [edge for edge in result["edges"] if edge["source"] == "reference"]
    assert [(edge["target"], edge["kind"]) for edge in references] == [("comparison", "reference")]
    chain = [(edge["source"], edge["target"]) for edge in result["edges"] if edge["kind"] == "main"]
    assert chain == [("sources", "pit"), ("pit", "transformer"), ("transformer", "residual"),
                     ("residual", "kalman"), ("kalman", "forecast"), ("forecast", "comparison")]
    assert result["facts"]["storm_used_as_input"] is False


def test_verified_transformer_explanation_uses_real_operations_without_invented_dimensions(tmp_path, monkeypatch):
    # Only a source-text fixture is needed: the architecture reader must never
    # execute a model module to explain the operations it has recognized.
    package = tmp_path / "fixture_chronos"
    (package / "chronos2").mkdir(parents=True)
    (package / "chronos2" / "model.py").write_text(
        "class Chronos2EncoderBlock\nTimeSelfAttention(config)\nGroupSelfAttention(config)\n"
        "FeedForward(config)\nself.input_patch_embedding\nself.output_patch_embedding\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(architecture.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(package / "__init__.py")))
    result = architecture.build_architecture(PROJECT_ROOT)
    assert result["inner_transformer_source"]["verified"] is True
    assert [step["id"] for step in result["inner_transformer"]] == [
        "patches", "time_attention", "group_attention", "feed_forward", "quantiles",
    ]
    text = json.dumps(result["inner_transformer"], ensure_ascii=False).lower()
    assert "attention temporelle" in text and "attention de groupe" in text and "quantiles" in text
    assert "décodeur" not in text and "decoder" not in text
    assert not re.search(r"\b\d+\s+(?:couches|blocs|layers|têtes|heads)\b", text)


def test_missing_configs_and_unavailable_model_package_report_unknowns(tmp_path, monkeypatch):
    monkeypatch.setattr(architecture.importlib.util, "find_spec", lambda name: None)
    result = architecture.build_architecture(tmp_path)
    facts = result["facts"]
    assert facts["model_id"] is None and facts["context_hours"] is None
    assert facts["residual_backend"] is None and facts["computation_mode"] is None
    assert facts["lora_active"] is None
    assert all(value is None for value in facts["context_hours_by_zone"].values())
    assert result["warnings"]
    assert result["inner_transformer"] == []
    assert result["inner_transformer_source"]["verified"] is False
    assert any(badge["value"] == "Inconnu" for badge in result["badges"])


def test_secret_values_from_yaml_are_never_part_of_architecture_output(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    marker = "SYNTHETIC_SECRET_DO_NOT_EXPOSE"
    settings = {
        "password": marker + "_PASSWORD", "nuclear_store": marker + "_PRIVATE_PATH",
        "computation_mode": marker + "_MODE", "lora_activation_config": "config/activation.yaml",
        "kalman_config": "config/kalman.yaml", "zone_configs": {zone: "config/zone.yaml" for zone in architecture.ZONES},
    }
    zone = {
        "model": {"model_id": marker + "_MODEL", "context_length": marker + "_CONTEXT", "api_key": marker + "_KEY"},
        "hourly": {"residual_correction": {"backend": marker + "_BACKEND", "iterations": marker + "_TREES", "depth": marker + "_DEPTH"}},
    }
    activation = {"token": marker + "_TOKEN", "zones": {zone: {"enabled_modes": [marker + "_ADAPTER"]} for zone in architecture.ZONES}}
    kalman = {"filter_parameters": {"governance_lookback_days": marker + "_GOVERNANCE"}, "private_key": marker + "_PEM"}
    for name, contents in {"nuclear_forecast.yaml": settings, "zone.yaml": zone,
                           "activation.yaml": activation, "kalman.yaml": kalman}.items():
        (config / name).write_text(yaml.safe_dump(contents), encoding="utf-8")
    before = {path: path.read_bytes() for path in config.iterdir()}
    result = architecture.build_architecture(tmp_path)
    assert marker not in json.dumps(result, ensure_ascii=False)
    assert result["facts"]["model_id"] is None
    assert result["facts"]["context_hours"] is None
    assert result["facts"]["residual_trees"] is None
    assert result["facts"]["kalman_governance_days"] is None
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_architecture_in_clean_subprocess_does_not_import_scientific_model_runtime():
    code = """
import json, sys
from experiment_console.architecture import build_architecture
result = build_architecture(sys.argv[1])
print(json.dumps({'nodes': len(result['nodes']), 'torch_imported': 'torch' in sys.modules,
                  'chronos_imported': 'chronos' in sys.modules}))
"""
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    process = subprocess.run([sys.executable, "-c", code, str(PROJECT_ROOT)], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=30, check=True, **options)
    result = json.loads(process.stdout)
    assert result == {"nodes": 8, "torch_imported": False, "chronos_imported": False}
