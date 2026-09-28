"""Read-only inventory of the archived annual CWE price and negative-price models.

The archived 2026-09-23 models are research artifacts. This module intentionally
does not offer an execution path: a complete historical archive is not a live
feature pipeline, a CPU-equivalent price trainer, or an independent validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


MANIFEST_PATH = Path("config/nyx_annual_cwe_historical.json")
COUNTRIES = ("FR", "DE", "BE", "NL")


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path is empty")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if posix.is_absolute() or windows.drive or ".." in posix.parts or "\\" in relative:
        raise ValueError(f"artifact path must be repository-relative: {relative!r}")
    target = (root / Path(*posix.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"artifact path leaves the repository: {relative!r}")
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_file(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    relative = spec.get("path")
    result: dict[str, Any] = {"path": relative, "status": "invalid_manifest"}
    expected = spec.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        result["detail"] = "missing or invalid pinned SHA-256"
        return result
    try:
        path = _safe_path(root, relative)
    except (TypeError, ValueError) as exc:
        result["detail"] = str(exc)
        return result
    if not path.exists():
        result["status"] = "missing"
        return result
    if not path.is_file():
        result["status"] = "not_a_file"
        return result
    size = path.stat().st_size
    result["size_bytes"] = size
    if size == 0:
        result["status"] = "empty"
        return result
    actual = _sha256(path)
    result["status"] = "verified" if actual == expected else "sha256_mismatch"
    if actual != expected:
        result["actual_sha256"] = actual
    return result


def _inspect_checkpoint(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    result = {key: _inspect_file(root, spec[key]) for key in ("receipt", "model", "calibration") if key in spec}
    receipt_state = result.get("receipt", {}).get("status")
    if receipt_state == "verified":
        try:
            receipt = json.loads(_safe_path(root, spec["receipt"]["path"]).read_text(encoding="utf-8"))
            names_match = (
                receipt.get("state") == "COMPLETE"
                and receipt.get("attempt_directory") == spec.get("attempt_directory")
                and receipt.get("files", {}).get("model.cbm") == spec["model"]["sha256"]
                and (
                    "calibration" not in spec
                    or receipt.get("files", {}).get("model.json") == spec["calibration"]["sha256"]
                )
            )
            result["receipt_contract"] = "verified" if names_match else "mismatch"
        except (OSError, ValueError, TypeError, KeyError):
            result["receipt_contract"] = "unreadable"
    else:
        result["receipt_contract"] = "unavailable"
    result["verified"] = result["receipt_contract"] == "verified" and all(
        item["status"] == "verified" for item in result.values() if isinstance(item, dict)
    )
    return result


def _invalid_result(message: str) -> dict[str, Any]:
    return {
        "operation": "inspect_annual_cwe",
        "manifest_valid": False,
        "forecast_ready": False,
        "forecast_enabled": False,
        "countries": {},
        "blockers": [{"code": "invalid_manifest", "message": message}],
    }


def inspect_annual_cwe(root: Path, countries: tuple[str, ...] = COUNTRIES) -> dict[str, Any]:
    """Inspect only local archived files; never sync, fit, predict, or write."""

    root = root.resolve()
    unknown = sorted(set(countries) - set(COUNTRIES))
    if unknown:
        return _invalid_result(f"unsupported countries: {', '.join(unknown)}")
    try:
        manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _invalid_result(f"cannot load {MANIFEST_PATH.as_posix()}: {exc}")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("identity") != "nyx_annual_cwe_2026-09-23_historical_inspection_v1"
        or manifest.get("forecast_enabled") is not False
        or set(manifest.get("countries", {})) != set(COUNTRIES)
    ):
        return _invalid_result("historical manifest schema or inspection-only contract differs")

    try:
        selected = tuple(dict.fromkeys(countries))
        evidence = {key: _inspect_file(root, value) for key, value in manifest["evidence"].items()}
        feature_keys = tuple(dict.fromkeys(
            key
            for zone in selected
            for key in (
                *manifest["countries"][zone]["price"]["source_feature_keys"],
                manifest["countries"][zone]["negative"]["source_feature_key"],
            )
        ))
        checkpoint_keys = tuple(dict.fromkeys(
            key
            for zone in selected
            for key in (
                *manifest["countries"][zone]["price"]["checkpoint_keys"],
                manifest["countries"][zone]["negative"]["checkpoint_key"],
            )
        ))
        features = {key: _inspect_file(root, manifest["source_feature_matrices"][key]) for key in feature_keys}
        checkpoints = {key: _inspect_checkpoint(root, manifest["checkpoints"][key]) for key in checkpoint_keys}
        references = {zone: _inspect_file(root, manifest["historical_references"][zone]) for zone in selected}
        reports = {
            key: _inspect_file(root, manifest["reports"][key])
            for key in ("price_overview", "price_kpi", "negative_overview")
        }
        reports["countries"] = {
            zone: {
                kind: _inspect_file(root, manifest["reports"]["countries"][zone][kind])
                for kind in ("price", "negative")
            }
            for zone in selected
        }
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        return _invalid_result(f"historical manifest is incomplete: {exc}")

    evidence_ok = all(item["status"] == "verified" for item in evidence.values())
    output_countries: dict[str, Any] = {}
    for zone in selected:
        selection = manifest["countries"][zone]
        needed_features = tuple(dict.fromkeys((
            *selection["price"]["source_feature_keys"],
            selection["negative"]["source_feature_key"],
        )))
        needed_checkpoints = tuple(dict.fromkeys((
            *selection["price"]["checkpoint_keys"],
            selection["negative"]["checkpoint_key"],
        )))
        missing_features = [key for key in needed_features if features[key]["status"] != "verified"]
        missing_checkpoints = [key for key in needed_checkpoints if not checkpoints[key]["verified"]]
        blockers = [
            {"code": "inspection_only", "message": "No live annual CWE forecast runner is registered."},
            {"code": "gpu_price_training", "message": "The exact price recipes were trained on a GPU; CPU retraining equivalence is unproven."},
            {"code": "live_features_unavailable", "message": "Historical feature matrices and checkpoints do not supply future Saturn/market features."},
            {"code": "independent_validation_missing", "message": "The annual comparison was retrospective, without an independent validation period."},
        ]
        if not evidence_ok:
            blockers.append({"code": "historical_evidence_unverified", "items": [key for key, item in evidence.items() if item["status"] != "verified"]})
        if missing_features:
            blockers.append({"code": "source_feature_matrices_unverified", "items": missing_features})
        if missing_checkpoints:
            blockers.append({"code": "historical_checkpoints_unverified", "items": missing_checkpoints})
        if references[zone]["status"] != "verified":
            blockers.append({"code": "historical_reference_unverified", "items": [zone]})
        if selection["price"]["score"]["goal_satisfied"] is False:
            blockers.append({"code": "storm_goal_not_met", "message": f"{zone} did not satisfy the selected historical Storm criterion."})
        output_countries[zone] = {
            "price": selection["price"],
            "negative": selection["negative"],
            "archived_files_verified": evidence_ok and not missing_features and not missing_checkpoints and references[zone]["status"] == "verified",
            "reports_verified": all(item["status"] == "verified" for item in reports["countries"][zone].values()),
            "forecast_ready": False,
            "blockers": blockers,
        }

    return {
        "operation": "inspect_annual_cwe",
        "manifest": MANIFEST_PATH.as_posix(),
        "manifest_valid": True,
        "origin_day": manifest["origin_day"],
        "evaluation_period": manifest["evaluation_period"],
        "forecast_enabled": False,
        "forecast_ready": False,
        "historical_evaluation": manifest["historical_evaluation"],
        "architectures": manifest["architectures"],
        "evidence": evidence,
        "source_feature_matrices": features,
        "historical_references": references,
        "checkpoints": checkpoints,
        "reports": reports,
        "countries": output_countries,
    }
