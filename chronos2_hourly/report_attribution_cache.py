"""Cache report-only explanations outside immutable live archives."""
from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from chronos2_hourly.atomic_directory import AtomicDirectoryStaging, publish_directory_no_replace


FILES = ("variable_attribution_hourly.csv.gz", "variable_attribution_audit.json")


def local_attribution_model_config(config: Mapping) -> dict:
    """Point inference at the cached snapshot, avoiding PEFT's optional HEAD.

    Some transformer versions probe adapter_config.json on the Hub even when
    local_files_only=True. A real local directory makes that probe filesystem-
    only. Resolve the requested commit locally; never change the model recipe.
    """
    from huggingface_hub import hf_hub_download

    result = copy.deepcopy(dict(config))
    model = result.setdefault("model", {})
    model_id = str(model.get("model_id", "amazon/chronos-2"))
    if Path(model_id).is_dir():
        snapshot = Path(model_id).resolve()
    else:
        cached_config = hf_hub_download(
            repo_id=model_id, filename="config.json", revision=model.get("revision") or "main",
            local_files_only=True,
        )
        snapshot = Path(cached_config).parent.resolve()
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Snapshot Chronos local incomplet : {snapshot}")
    model["model_id"] = str(snapshot)
    model["local_files_only"] = True
    model.pop("revision", None)
    return result

_REPORT_INPUTS = (
    "aligned_inputs.csv.gz", "model_covariates_with_future.csv.gz",
    "input_coverage.csv", "input_manifest.csv",
)
_REFIT_INPUTS = (
    "feature_manifest.csv", "backtest_hourly_oof.csv.gz", "run_manifest.json",
    "inputs/chronos_oof_extended.csv.gz", "inputs/aligned_inputs.csv.gz",
    "inputs/model_covariates_with_future.csv.gz",
)
_ALGORITHM_SOURCES = (
    "run_multicountry_forecast.py", "run_chronos2_hourly.py",
    "run_mkonline_live_hourly.py", "run_extended_residual_hourly.py",
    "chronos2_hourly/report_attribution_cache.py",
    "chronos2_hourly/variable_attribution.py", "chronos2_hourly/chronos_adapter.py",
    "chronos2_hourly/features.py", "chronos2_hourly/fundamental_features.py",
    "chronos2_hourly/hourly_contract.py", "chronos2_hourly/multizone_contract.py",
    "chronos2_hourly/multizone_live.py", "chronos2_hourly/models/residual_corrector.py",
    "chronos2_hourly/models/blended_residual_corrector.py",
    "chronos2_hourly/models/base.py", "chronos2_hourly/models/calibration.py",
    "chronos2_hourly/models/catboost_hourly.py", "chronos2_hourly/models/ensemble.py",
    "chronos2_hourly/models/lear.py",
    "chronos2_modular/common.py", "chronos2_modular/forecasting.py",
)


def _config_path(value: object, *, parent: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Source d'attribution {name} invalide.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def _manifest_sources(manifest: Path, *, project_root: Path) -> set[Path]:
    """Resolve current materialized bytes, not just their old sealed digest.

    Source code is project-relative; data artifacts are run-relative. Original
    upstream inputs and reports are not consumed by this report-only refit.
    Weight binaries are deliberately not traversed or hashed.
    """
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("algorithm") != "sha256" or not isinstance(payload.get("artifacts"), list):
        raise ValueError(f"Manifeste des sources d'attribution invalide : {manifest}")
    result = {manifest.resolve()}
    for item in payload["artifacts"]:
        if not isinstance(item, Mapping):
            raise ValueError(f"Entree de manifeste invalide : {manifest}")
        role = item.get("role")
        if role not in {"source_code", "run_artifact", "materialized_input"}:
            continue
        raw = str(item.get("path") or "").replace("\\", "/")
        relative = Path(raw)
        if not raw or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Chemin de source d'attribution non relatif : {raw!r}")
        root = project_root if role == "source_code" else manifest.parent
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError(f"Source d'attribution hors racine : {raw!r}")
        if role == "source_code" and not resolved.is_file() and raw not in _ALGORITHM_SOURCES:
            # Old live seals also list reporting/benchmark scripts since removed
            # from the project. They are not executed by this attribution. The
            # current algorithm dependencies above remain mandatory, even if
            # absent from a legacy seal. Never rewrite the old seal to fix this.
            continue
        if role == "source_code" or raw.lower().endswith((".csv", ".csv.gz", ".parquet", ".json", ".yaml", ".yml")):
            result.add(resolved)
    return result


def report_attribution_sources(*, archive: Path, forecast_path: Path,
                               live_config: Path, registry_path: Path,
                               project_root: Path) -> tuple[Path, ...]:
    """Complete file identity of the archived-day explanation/refit.

    Config paths follow the live runner's resolution rules. Frozen-manifest
    declarations supplement explicit refit inputs so legacy manifests cannot
    accidentally omit a file that the reconstruction actually reads. Existing
    seals are only read, never compared against changed report code or rewritten.
    """
    from chronos2_modular.common import load_yaml

    root = Path(project_root).resolve()
    archive = Path(archive).resolve()
    live_config = Path(live_config).resolve()
    live_payload = load_yaml(live_config)
    live = live_payload.get("live")
    if not isinstance(live, Mapping):
        raise ValueError(f"Configuration live d'attribution invalide : {live_config}")
    base = _config_path(live.get("base_config"), parent=live_config.parent, name="base_config")
    frozen = _config_path(live.get("frozen_autonomous_run"), parent=live_config.parent, name="frozen_autonomous_run")
    sources = {
        live_config, base, Path(registry_path).resolve(), Path(forecast_path).resolve(),
        archive / "run_manifest.json", archive / "chronos_live_hourly.csv",
        *[archive / "inputs" / name for name in _REPORT_INPUTS],
        *[frozen / name for name in _REFIT_INPUTS],
        *[root / name for name in _ALGORITHM_SOURCES],
    }
    for name in ("recipe_manifest", "dependency_manifest"):
        if live.get(name) is not None:
            sources.add(_config_path(live[name], parent=live_config.parent, name=name))
    # A pre-attribution live archive may have no checksum manifest. The frozen
    # training source must have one, and all declared consumed files must exist.
    sources.update(_manifest_sources(frozen / "artifact_checksums.json", project_root=root))
    live_seal = archive / "artifact_checksums.json"
    if live_seal.is_file():
        sources.update(_manifest_sources(live_seal, project_root=root))
    # A floating HF revision is represented by its tiny local commit ref. Do not
    # hash multi-GB weights; an explicit commit revision is already in base YAML.
    config = load_yaml(base)
    model = config.get("model", {})
    if isinstance(model, Mapping):
        model_id = str(model.get("model_id", "amazon/chronos-2"))
        revision = str(model.get("revision") or "main")
        if not (len(revision) == 40 and all(char in "0123456789abcdefABCDEF" for char in revision)):
            from huggingface_hub.constants import HF_HUB_CACHE

            cache_root = Path(HF_HUB_CACHE).resolve()
            reference = (cache_root / ("models--" + model_id.replace("/", "--")) / "refs" / revision).resolve()
            if reference.is_relative_to(cache_root) and reference.is_file():
                sources.add(reference)
    missing = sorted(str(path) for path in sources if not path.is_file())
    if missing:
        raise FileNotFoundError("Sources d'attribution absentes : " + ", ".join(missing))
    return tuple(sorted(path.resolve() for path in sources))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_historical_prices(directory: Path) -> bool:
    path = directory / FILES[1]
    if not path.is_file():
        return False
    audit = json.loads(path.read_text(encoding="utf-8"))
    return any(group.get("key") == "historical_target_price" for group in audit.get("groups", []))


def _validated_attribution(directory: Path, identity: Mapping[str, str]) -> Path:
    manifest = directory / "report_cache_manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Cache d'attribution incomplet : {directory}")
    sealed = json.loads(manifest.read_text(encoding="utf-8"))
    artifacts = sealed.get("artifacts") if isinstance(sealed, dict) else None
    if (not isinstance(artifacts, dict) or not set(FILES).issubset(artifacts)
            or sealed.get("sources") != identity):
        raise ValueError(f"Cache d'attribution divergent : {directory}")
    for name, digest in artifacts.items():
        relative = Path(name)
        path = directory / relative
        if (relative.is_absolute() or ".." in relative.parts
                or not path.resolve().is_relative_to(directory.resolve())
                or not path.is_file() or sha256(path) != digest):
            raise ValueError(f"Cache d'attribution divergent : {directory}")
    if not has_historical_prices(directory):
        raise ValueError(f"Cache d'attribution divergent : {directory}")
    return directory


def cached_attribution(*, root: Path, sources: Sequence[Path],
                       materialize: Callable[[Path], object]) -> Path:
    """Reuse only an exact identity and intact complete payload; never reseal a run."""
    paths = sorted(set(path.resolve() for path in sources))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Sources d'attribution absentes : " + ", ".join(missing))
    identity = {str(path): sha256(path) for path in paths}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(root).resolve()
    destination = root / key
    if destination.exists():
        return _validated_attribution(destination, identity)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f".attribution-{key}-"
    # A previous publication may have exhausted its Windows lock retries. Its
    # sealed payload is reusable only with the same sources and intact bytes.
    for staged in sorted(root.glob(prefix + "*")):
        if not staged.is_dir() or staged.is_symlink():
            continue
        try:
            _validated_attribution(staged, identity)
        except (ValueError, FileNotFoundError):
            # Incomplete construction or a damaged stage never counts as a hit.
            continue
        try:
            publish_directory_no_replace(staged, destination)
        except FileExistsError:
            pass  # Validate the concurrent publisher's result below.
        except (ValueError, FileNotFoundError):
            # Another caller may have published this very stage meanwhile.
            if staged.exists() or not destination.exists():
                raise
        return _validated_attribution(destination, identity)
    if destination.exists():
        return _validated_attribution(destination, identity)
    with AtomicDirectoryStaging(root, prefix=prefix) as publication:
        staged = publication.path
        materialize(staged)
        if not has_historical_prices(staged):
            raise ValueError("Le nouveau cache doit expliquer les prix passés.")
        if any(not Path(path).is_file() or sha256(Path(path)) != digest
               for path, digest in identity.items()):
            raise ValueError("Les sources ont changé pendant l'attribution.")
        # Keep companion files (including the frozen forecast copy) under the
        # same integrity seal when resuming a completed publication.
        artifacts = {path.relative_to(staged).as_posix(): sha256(path)
                     for path in sorted(staged.rglob("*")) if path.is_file()}
        if not set(FILES).issubset(artifacts):
            raise ValueError("Le nouveau cache d'attribution est incomplet.")
        payload = {"sources": identity, "artifacts": artifacts}
        (staged / "report_cache_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # A concurrent publisher may have won the same identity. Never overwrite.
        try:
            _validated_attribution(staged, identity)
            publication.publish(destination)
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            if not isinstance(error, FileExistsError) and (staged.exists() or not destination.exists()):
                raise
            result = _validated_attribution(destination, identity)
            publication.preserve = False
            return result
    return _validated_attribution(destination, identity)
