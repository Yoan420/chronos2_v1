"""Resume the DE annual CPU price replay without repeating the FR expert.

The original pooled fits discarded DE predictions and did not save models.
Consequently 106 fits are necessary. All four predicted country curves and CBM
models are retained here so extending the audit again never requires a refit.
This retrospective score does not qualify the complete live input chain.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_nyx_selected_cpu_historical as archived
from chronos2_hourly.nyx_pooled_cpu_price_model import fit_pooled_block


ROLES = ("cwe_residual_2000", "cwe_absolute_2000")


def replay(output: Path, *, roles: tuple[str, ...] = ROLES, threads: int = 8) -> dict:
    archived.require(set(roles) <= set(ROLES) and roles, "Unknown DE expert role")
    actual, nyx, baselines = archived._load_baselines(archived._plan(ROLES[0]))
    reference_path = archived.source_path(archived.ARCHIVE /
        "rmse_exchange_composition_v1/oof/DE.parquet")
    reference_frame = pd.read_parquet(reference_path)
    reference = reference_frame["reference"].loc[archived.ANNUAL_GRID]
    comparison_path = archived.source_path(archived.BASELINE / "inputs/DE_comparison.parquet")
    comparison = pd.read_parquet(comparison_path).loc[archived.ANNUAL_GRID]
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for role in roles:
        features, feature_evidence = archived._load_features(role)
        config = archived.ROLE_CONFIGS[role]
        manifest = {"protocol": "nyx_de_cpu_annual_role_v1", "role": role,
            "config": asdict(config), "threads": threads,
            "model_code_sha256": archived.sha256(Path(archived.cpu_model.__file__)),
            "runner_code_sha256": archived.sha256(Path(__file__)),
            "baselines": baselines, "features": feature_evidence,
            "origins": archived.annual_origins(),
            "reference_sha256": archived.sha256(reference_path),
            "comparison_sha256": archived.sha256(comparison_path)}
        folder = output / role
        folder.mkdir(exist_ok=True)
        manifest_path = folder / "manifest.json"
        encoded = archived._canonical_json(manifest)
        if manifest_path.exists():
            archived.require(archived._canonical_json(json.loads(manifest_path.read_text()))
                == encoded, "DE replay inputs or implementation changed")
        else:
            archived._atomic_json(manifest_path, manifest)
        pieces, audits = [], []
        for position, (origin, stop) in enumerate(archived.annual_origins(), 1):
            checkpoint = folder / f"{origin}.json"
            model = folder / f"{origin}.cbm"
            points_path = folder / f"{origin}.parquet"
            if checkpoint.exists():
                audit = json.loads(checkpoint.read_text())
                archived.require(audit["manifest_sha256"] == archived.sha256(manifest_path)
                    and audit["points_sha256"] == archived.sha256(points_path)
                    and audit["model"]["sha256"] == archived.sha256(model),
                    "DE checkpoint or model was modified")
                frame = pd.read_parquet(points_path)
            else:
                # Unsealed interrupted output is never reused.
                if model.exists():
                    model.unlink()
                predictions, audit = fit_pooled_block(features, actual, nyx,
                    origin_day=origin, stop_day=stop, config=config,
                    model_path=model, thread_count=threads,
                    initial_training_day=archived.FIRST_TRAINING_DAY)
                frame = pd.DataFrame({z: predictions[z].point for z in archived.ZONES})
                audit["points_sha256"] = archived._atomic_parquet(points_path, frame)
                audit["manifest_sha256"] = archived.sha256(manifest_path)
                archived._atomic_json(checkpoint, audit)
            archived.require(frame.index.equals(archived.grid(origin, stop))
                and list(frame.columns) == list(archived.ZONES)
                and np.isfinite(frame.to_numpy()).all(), "DE checkpoint grid invalid")
            pieces.append(frame.DE)
            audits.append(audit)
            print(json.dumps({"role": role, "completed": position, "total": 53,
                "origin": origin}), flush=True)
        series = pd.concat(pieces)
        archived._atomic_parquet(folder / "DE.parquet", series.to_frame("point"))
        archived._atomic_json(folder / "receipt.json", {"complete": True,
            "fits": 53, "points_sha256": archived.sha256(folder / "DE.parquet"),
            "manifest_sha256": archived.sha256(manifest_path), "audits": audits})
    if all((output / role / "receipt.json").is_file() for role in ROLES):
        for role in ROLES:
            folder = output / role
            receipt = json.loads((folder / "receipt.json").read_text())
            archived.require(receipt["complete"] and receipt["fits"] == 53
                and receipt["points_sha256"] == archived.sha256(folder / "DE.parquet"),
                "DE expert aggregate invalid")
            results[role] = pd.read_parquet(folder / "DE.parquet").point
        mean = (results[ROLES[0]] + results[ROLES[1]]) / 2.
        selected = mean.where((mean-reference).abs() >= 20., reference).rename("point")
        digest = archived._atomic_parquet(output / "DE.parquet", selected.to_frame())
        receipt = {"protocol": "nyx_de_cpu_annual_selected_v1", "complete": True,
            "country": "DE", "composition": "boosting_mean_disagreement20",
            "score": archived.score_selected(selected, comparison),
            "expert_receipts_sha256": {r: archived.sha256(output/r/"receipt.json") for r in ROLES},
            "selected_points_sha256": digest, "full_input_chain_qualified": False,
            "performance_exception_authorized": True, "total_cpu_fits": 106}
        archived._atomic_json(output / "receipt.json", receipt)
        return receipt
    return {"complete": False, "completed_roles": list(roles)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(replay(args.output, roles=(args.role,) if args.role else ROLES,
                            threads=args.threads), allow_nan=False))
