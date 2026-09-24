"""Allowlisted adapters for existing CLIs; no computation or shell lives here.

Only output paths and local-only I/O policy are changed in scientific YAMLs.
The original CLI remains responsible for its scientific validation and results.
"""
from __future__ import annotations

import copy
import csv
from datetime import date, datetime, timedelta
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml


REPORT_MODELS = ["auto", "residual_corrected", "ensemble", "chronos2", "catboost", "lear", "mkonline_blend"]
SCRIPTS = {
    "model_storm_report": "run_model_storm_report.py",
    "hourly_report": "generate_hourly_html_report.py",
    "hourly_evaluation": "evaluate_hourly_backtest.py",
    "hourly_forecast": "run_chronos2_hourly.py",
}


def _field(name: str, label: str, kind: str = "text", default: Any = "", **extra: Any) -> dict:
    return {"name": name, "label": label, "type": kind, "default": default, **extra}


def _read_yaml(path: Path) -> dict:
    if path.stat().st_size > 2_000_000:
        raise ValueError("Configuration trop volumineuse (maximum 2 Mo).")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("La configuration YAML doit être un objet.")
    return value


def _no_secrets(value: Any) -> None:
    # This rejects credentials, rather than silently redacting executable settings.
    from .security import has_secrets
    if has_secrets(value):
        raise ValueError("La configuration contient un secret. Utilisez les variables d’environnement du pipeline.")


def _absolute(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def validate_primary_delivery_day(value):
    """Accept the PS1 calendar contract without inventing a date-range policy."""
    if not isinstance(value, str):
        raise ValueError('La date de livraison doit être une date valide au format YYYY-MM-DD.')
    try:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError()
    except ValueError as exc:
        raise ValueError('La date de livraison doit être une date valide au format YYYY-MM-DD.') from exc
    return value


class AdapterRegistry:
    def __init__(self, project_root: str | Path, python_executable: str | Path):
        self.project_root = Path(project_root).resolve()
        executable = Path(python_executable).expanduser()
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError("Configurez le chemin absolu d’un interpréteur Python existant.")
        self.python_executable = executable.resolve()
        self._primary_defaults_lock = threading.Lock()
        self._primary_defaults_cache = None

    def _primary_command(self, delivery_day=None):
        if delivery_day is not None:
            delivery_day = validate_primary_delivery_day(delivery_day)
        if os.name != 'nt':
            raise ValueError('Le lancement NYX utilise NuclearKalman.ps1 sous Windows.')
        shell = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
        script = self.project_root / 'NuclearKalman.ps1'
        if not shell.is_absolute() or not shell.is_file() or not script.is_file():
            raise ValueError('Le lanceur NuclearKalman.ps1 ou Windows PowerShell est introuvable.')
        if not script.resolve().is_relative_to(self.project_root):
            raise ValueError('Le lanceur NYX doit rester dans ce dépôt.')
        # The script keeps its scientific defaults. Only an explicit user date
        # is pinned; the legacy empty request still means tomorrow at execution.
        command = [str(shell.resolve()), '-NoProfile', '-File', str(script), '-NoOpen']
        if delivery_day is not None:
            command += ['-DeliveryDay', delivery_day]
        return command

    def primary_defaults(self, delivery_day=None):
        command = self._primary_command(delivery_day)
        source = (self.project_root / 'NuclearKalman.ps1').stat()
        key = (source.st_mtime_ns, source.st_size, datetime.now(ZoneInfo('Europe/Paris')).date(), delivery_day)
        with self._primary_defaults_lock:
            cached = self._primary_defaults_cache
            if cached and cached[0] == key and time.monotonic() - cached[1] < 30:
                return copy.deepcopy(cached[2])
            result = self._resolve_primary_defaults(command, delivery_day)
            self._primary_defaults_cache = (key, time.monotonic(), result)
            return copy.deepcopy(result)

    def _resolve_primary_defaults(self, command, selected_delivery_day=None):
        try:
            result = subprocess.run(command + ['-DryRun'], cwd=self.project_root,
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace', timeout=20,
                                    shell=False, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
            prefix = 'Commande (argv, shell=False): '
            lines = [line[len(prefix):] for line in result.stdout.splitlines() if line.startswith(prefix)]
            argv = json.loads(lines[-1])
            if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
                raise ValueError()
            if Path(argv[0]).resolve() != self.python_executable:
                raise ValueError('Le Python par défaut du lanceur doit correspondre au Python configuré dans NYX.')
            if Path(argv[1]).resolve() != self.project_root / 'run_nuclear_kalman.py':
                raise ValueError()
            def argument(name):
                return argv[argv.index(name) + 1]
            countries = argv[argv.index('--zones') + 1:argv.index('--delivery-day')]
            delivery_day = argument('--delivery-day')
            if date.fromisoformat(delivery_day).isoformat() != delivery_day or countries != ['BE', 'DE', 'FR', 'NL']:
                raise ValueError()
            if selected_delivery_day is not None and delivery_day != selected_delivery_day:
                raise ValueError('Le lanceur n’a pas conservé la date de livraison sélectionnée.')
            config_path = self._input_path(argument('--nuclear-config'))
            output = Path(argument('--output')).resolve()
            expected_output = self.project_root / f'runs/reports/model_storm/CWE_Model_Storm_{delivery_day}.html'
            if output != expected_output or '--no-open' not in argv:
                raise ValueError()
            values = {'delivery_day': delivery_day, 'countries': countries,
                      'device': argument('--device'), 'threads': int(argument('--threads')),
                      'workers': int(argument('--workers')),
                      'sync_enabled': '--skip-observed-sync' not in argv,
                      'attribution_enabled': '--with-attribution' in argv,
                      'nuclear_config': str(config_path), 'output': str(output),
                      'command': command, 'resolved_command': argv,
                      'delivery_day_resolved_at_execution': selected_delivery_day is None, 'no_open': True}
            if (values['device'], values['threads'], values['workers'], values['sync_enabled'], values['attribution_enabled']) != ('auto', 4, 4, True, False):
                raise ValueError('Les paramètres par défaut de NuclearKalman.ps1 ont changé. Vérifiez le lanceur avant exécution.')
            _no_secrets(values)
            return values
        except (OSError, subprocess.SubprocessError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Impossible de vérifier les paramètres par défaut de NuclearKalman.ps1 (-DryRun).') from exc
        except ValueError as exc:
            if not str(exc):
                raise ValueError('Les paramètres par défaut de NuclearKalman.ps1 ne correspondent plus au contrat NYX.') from exc
            raise

    def _primary_sources(self, defaults):
        nuclear_path = Path(defaults['nuclear_config'])
        nuclear = _read_yaml(nuclear_path)
        _no_secrets(nuclear)
        scientific_root = _absolute(nuclear.get('project_root', '..'), nuclear_path.parent)
        if scientific_root != self.project_root:
            raise ValueError('La configuration nucléaire doit référencer ce dépôt.')
        sources = [self.project_root / 'NuclearKalman.ps1', self.project_root / 'run_nuclear_kalman.py', nuclear_path]
        for key in ('kalman_config', 'lora_activation_config'):
            if nuclear.get(key):
                sources.append(self._input_path(nuclear[key]))
        for zone in defaults['countries']:
            sources.append(self._input_path(nuclear['zone_configs'][zone]))
        result = {}
        for source in sources:
            if source.stat().st_size > 2_000_000:
                raise ValueError('Source de lancement trop volumineuse pour son snapshot.')
            content = source.read_bytes()
            if source.suffix.lower() in {'.yaml', '.yml'}:
                _no_secrets(_read_yaml(source))
            else:
                _no_secrets(content.decode('utf-8-sig'))
            result[source.relative_to(self.project_root).as_posix()] = content
        return result

    def prepare_primary_run(self, run_directory, write=False, *, delivery_day=None):
        root = Path(run_directory).resolve()
        if not root.is_relative_to(self.project_root) or root == self.project_root:
            raise ValueError('Les métadonnées de lancement doivent rester sous ce dépôt.')
        snapshot = root / 'config.yaml'
        if snapshot.exists():
            raise FileExistsError('Ce lancement possède déjà un snapshot.')
        defaults = self.primary_defaults(delivery_day)
        sources = self._primary_sources(defaults)
        hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
        config = {'adapter_id': 'primary_nuclear_kalman', 'model': 'NYX', 'defaults_at_request': defaults,
                  'selected_delivery_day': delivery_day,
                  'source_hashes': hashes, 'snapshots_for_audit_only': True,
                  'publication_root': str(self.project_root / 'runs')}
        prepared = {'adapter_id': 'primary_nuclear_kalman', 'type': 'primary_forecast', 'model': 'NYX',
                    'command': defaults['command'], 'cwd': str(self.project_root),
                    'config': config, 'config_path': str(snapshot), 'delivery_day': defaults['delivery_day'],
                    'output_dir': str(self.project_root / 'runs'),
                    'resource_keys': ['scientific-cache', 'nyx-primary-pipeline', 'canonical-publications'],
                    'warnings': ['Exécute NuclearKalman.ps1 avec ses paramètres scientifiques par défaut, synchronisation comprise.',
                                 '-NoOpen conserve la consultation des rapports dans NYX.',
                                 ('La livraison sélectionnée est conservée même si le lancement attend en file.' if delivery_day is not None else
                                  'La livraison est calculée au démarrage du script ; une attente au-delà de minuit peut la décaler.'),
                                 'Les protections natives du pipeline restent actives ; un lancement extérieur sur une autre date peut utiliser les mêmes ressources.'],
                    'request': {'adapter_id': 'primary_nuclear_kalman', **({'delivery_day': delivery_day} if delivery_day is not None else {})}}
        _no_secrets(prepared)
        if write:
            root.mkdir(parents=True, exist_ok=True)
            with snapshot.open('x', encoding='utf-8') as stream:
                yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
            for name, content in sources.items():
                target = root / 'source_snapshots' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(content)
        return prepared

    def validate_primary_sources(self, run):
        selected = run.get('request', {}).get('delivery_day')
        if selected is not None:
            selected = validate_primary_delivery_day(selected)
            if run.get('delivery_day') != selected:
                raise ValueError('La date du lancement ne correspond plus à la sélection enregistrée.')
        if run.get('config', {}).get('selected_delivery_day') != selected:
            raise ValueError('La sélection de livraison ne correspond plus au snapshot enregistré.')
        if run['command'] != self._primary_command(selected):
            raise ValueError('Le lanceur NYX a changé depuis la demande. Préparez un nouveau lancement.')
        for relative, expected in run['config']['source_hashes'].items():
            source = self._input_path(relative)
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError('Une source de NuclearKalman.ps1 a changé pendant l’attente. Relancez depuis NYX.')

    def _configs(self) -> list[dict]:
        results = []
        # Only the actual standalone hourly contracts are offered. Live orchestration
        # YAMLs have different output semantics and are deliberately excluded.
        for path in sorted(self.project_root.glob("chronos2_hourly_*.yaml")):
            try:
                raw = _read_yaml(path)
                if not all(isinstance(raw.get(key), dict) for key in ("model", "data", "hourly", "zones")):
                    continue
                _no_secrets(raw)
                zones = [key for key, value in raw["zones"].items() if isinstance(value, dict) and value.get("enabled", True)]
                if len(zones) != 1:
                    continue
                results.append({"id": path.name, "label": f"{zones[0]} · {path.stem}",
                                "model": raw.get("report", {}).get("native_model", "ensemble"), "zone": zones[0]})
            except (OSError, ValueError, yaml.YAMLError):
                continue
        return results

    def _default_source(self) -> str:
        for folder in sorted((self.project_root / "runs").glob("chronos2_hourly*")):
            if (folder / "backtest_hourly_oof.csv.gz").is_file():
                return folder.relative_to(self.project_root).as_posix()
        return ""

    def catalog(self) -> list[dict]:
        configs = self._configs()
        source = self._default_source()
        local = [{"id": "local-results", "label": "Résultats locaux existants"}]
        tomorrow = (datetime.now(ZoneInfo("Europe/Paris")).date() + timedelta(days=1)).isoformat()
        common = [_field("timezone", "Fuseau horaire", default="Europe/Paris", advanced=True)]
        result = [
            {"id": "model_storm_report", "label": "Rapport Model / Storm", "type": "report",
             "models": ["Model / Storm"], "configs": local,
             "description": "Assemble les prévisions et prix locaux vérifiés, puis actualise uniquement la métrique VPS Saturn du rapport; aucun modèle n'est relancé.",
             "parameters": [_field("delivery_day", "Date de livraison", "date", tomorrow)]},
            {"id": "hourly_report", "label": "Rapport horaire", "type": "report", "models": REPORT_MODELS,
             "configs": local, "description": "Génère le rapport HTML du pipeline depuis un dossier existant.",
             "parameters": [_field("source_run", "Dossier du run source", default=source),
                            _field("zone", "Pays", "select", "FR", options=["FR", "BE", "DE", "NL", "ES"]),
                            _field("title", "Titre du rapport", default=""),
                            _field("baseline_model", "Modèle de référence", "select", "auto", options=REPORT_MODELS, advanced=True),
                            _field("history_hours", "Heures d’historique affichées", "integer", 168, min=1, max=8760, advanced=True), *common]},
            {"id": "hourly_evaluation", "label": "Évaluation horaire appariée", "type": "evaluation",
             "models": ["paired_day_bootstrap"], "configs": local,
             "description": "Réutilise l’évaluation existante et son bootstrap par jour complet; aucune prévision recalculée.",
             "parameters": [_field("backtest_file", "Fichier de backtest", default=f"{source}/backtest_hourly_oof.csv.gz" if source else ""),
                            _field("baseline", "Colonne de référence", default="ensemble__q50"),
                            _field("candidate", "Colonne candidate", default="residual_corrected__q50"),
                            _field("actual", "Colonne observée", default="actual", advanced=True),
                            _field("bootstrap_samples", "Échantillons bootstrap", "integer", 20000, min=1, max=20000, advanced=True),
                            _field("seed", "Graine", "integer", 42, min=0, max=2147483647, advanced=True), *common]},
            {"id": "hourly_forecast", "label": "Prévision / backtest horaire", "type": "forecast",
             "models": sorted({item["model"] for item in configs}), "configs": configs,
             "description": "Exécute Chronos-2, LEAR, CatBoost et le correcteur de la configuration. Calcul potentiellement long; caches locaux obligatoires.",
             "parameters": [_field("device", "Calcul", "select", "auto", options=["auto", "cpu", "cuda"]),
                            _field("log_level", "Niveau de journalisation", "select", "INFO", options=["DEBUG", "INFO", "WARNING", "ERROR"], advanced=True)]},
        ]
        return [entry for entry in result if (self.project_root / SCRIPTS[entry["id"]]).is_file() and entry["models"]]

    def _parameters(self, schema: list[dict], supplied: Any) -> dict:
        if not isinstance(supplied, dict):
            raise ValueError("Les paramètres doivent être un objet.")
        unknown = set(supplied) - {field["name"] for field in schema}
        if unknown:
            raise ValueError("Paramètres non autorisés : " + ", ".join(sorted(unknown)))
        result = {}
        for field in schema:
            name = field["name"]
            value = supplied.get(name, field["default"])
            if field["type"] == "integer":
                if isinstance(value, bool) or not isinstance(value, int) or not field["min"] <= value <= field["max"]:
                    raise ValueError(f"{field['label']} doit être un entier entre {field['min']} et {field['max']}.")
            else:
                if not isinstance(value, str) or len(value) > 2048 or "\x00" in value:
                    raise ValueError(f"Valeur invalide pour {field['label']}.")
                value = value.strip()
                if field["type"] == "select" and value not in field["options"]:
                    raise ValueError(f"Choix invalide pour {field['label']}.")
                if field["type"] == "date":
                    try:
                        if date.fromisoformat(value).isoformat() != value:
                            raise ValueError()
                    except ValueError as error:
                        raise ValueError("La livraison doit être au format YYYY-MM-DD.") from error
            result[name] = value
        if "timezone" in result:
            try:
                ZoneInfo(result["timezone"])
            except (ValueError, ZoneInfoNotFoundError) as error:
                raise ValueError("Fuseau horaire inconnu.") from error
        return result

    def _input_path(self, value: str, *, directory: bool = False) -> Path:
        if not value:
            raise ValueError("Sélectionnez un fichier ou dossier source existant.")
        path = _absolute(value, self.project_root)
        if not path.is_relative_to(self.project_root):
            raise ValueError("Les sources de lancement doivent rester dans le dépôt (les imports externes restent consultables).")
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f"Source introuvable : {path}")
        return path

    @staticmethod
    def _backtest_columns(path: Path) -> list[str]:
        if path.name.endswith((".csv", ".csv.gz")):
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
                return next(csv.reader(stream), [])
        if path.suffix.lower() in {".parquet", ".pq"}:
            try:
                import pyarrow.parquet as parquet
                return parquet.read_schema(path).names
            except ImportError as error:
                raise ValueError("Installez pyarrow dans l’environnement de la console pour lire le schéma Parquet.") from error
        raise ValueError("Un fichier CSV, CSV.GZ ou Parquet est requis.")

    def prepare(self, request: dict, run_directory: str | Path, write: bool = False,
                *, snapshot_config: dict | None = None) -> dict:
        """Prepare a command. snapshot_config is trusted backend storage only.

        HTTP callers must never supply snapshot_config directly. The manager can
        use it after looking up an existing managed run for exact duplication.
        """
        if not isinstance(request, dict):
            raise ValueError("Requête invalide.")
        adapter = next((item for item in self.catalog() if item["id"] == request.get("adapter_id")), None)
        frozen_selection = None
        if snapshot_config is not None:
            if request.get("adapter_id") != "hourly_forecast" or not isinstance(snapshot_config, dict):
                raise ValueError("Une copie scientifique est réservée aux prévisions horaires enregistrées.")
            _no_secrets(snapshot_config)
            if not all(isinstance(snapshot_config.get(key), dict) for key in ("model", "data", "hourly", "zones")):
                raise ValueError("La configuration enregistrée ne respecte pas le contrat horaire.")
            zones = [key for key, value in snapshot_config["zones"].items()
                     if isinstance(value, dict) and value.get("enabled", True)]
            if len(zones) != 1:
                raise ValueError("La copie doit contenir exactement un pays activé.")
            frozen_selection = {"id": request.get("config_id") or "managed-snapshot", "zone": zones[0],
                                "model": snapshot_config.get("report", {}).get("native_model", "ensemble")}
            if frozen_selection["model"] not in REPORT_MODELS[1:]:
                raise ValueError("Modèle horaire enregistré inconnu.")
            if not (self.project_root / SCRIPTS["hourly_forecast"]).is_file():
                raise ValueError("Le CLI horaire n’est plus disponible.")
            adapter = {"id": "hourly_forecast", "type": "forecast", "configs": [frozen_selection],
                       "models": [frozen_selection["model"]], "parameters": [
                           _field("device", "Calcul", "select", "auto", options=["auto", "cpu", "cuda"]),
                           _field("log_level", "Niveau de journalisation", "select", "INFO", options=["DEBUG", "INFO", "WARNING", "ERROR"])]}
        if adapter is None:
            raise ValueError("Type de run inconnu ou indisponible.")
        config_id = request.get("config_id") or adapter["configs"][0]["id"]
        selected = next((item for item in adapter["configs"] if item["id"] == config_id), None)
        if selected is None:
            raise ValueError("Configuration absente du catalogue.")
        model = request.get("model") or adapter["models"][0]
        if model not in adapter["models"]:
            raise ValueError("Modèle absent du catalogue.")
        params = self._parameters(adapter["parameters"], request.get("parameters", {}))
        _no_secrets(params)
        root = Path(run_directory).resolve()
        if not root.is_relative_to(self.project_root) or root == self.project_root:
            raise ValueError("Le nouveau dossier de run doit rester sous le dépôt.")
        output = root / "outputs"
        snapshot = root / "config.yaml"
        if snapshot.exists() or output.exists():
            raise FileExistsError("Ce run possède déjà une configuration ou des sorties; créez un nouvel identifiant.")
        aid = adapter["id"]
        command = [str(self.python_executable), "-u", str(self.project_root / SCRIPTS[aid])]
        warnings, resources = [], []
        effective = {"adapter_id": aid, "model": model, "parameters": params, "output_directory": str(output)}
        source_config = None
        if aid == "model_storm_report":
            command += ["--delivery-day", params["delivery_day"], "--output", str(output / "model_storm.html")]
            warnings.append("Les données absentes restent indisponibles; une livraison sans résultat local fait échouer le rapport.")
        elif aid == "hourly_report":
            source = self._input_path(params["source_run"], directory=True)
            table = source / "backtest_hourly_oof.csv.gz"
            if not table.is_file():
                raise ValueError("Le dossier source ne contient pas backtest_hourly_oof.csv.gz.")
            columns = self._backtest_columns(table)
            for selected_model in (model, params["baseline_model"]):
                if selected_model != "auto" and selected_model + "__q50" not in columns:
                    raise ValueError(f"Le backtest ne contient pas {selected_model}__q50.")
            command += [str(source), "--output", str(output / "report.html"), "--zone", params["zone"],
                        "--timezone", params["timezone"], "--history-hours", str(params["history_hours"])]
            if params["title"]:
                command += ["--title", params["title"]]
            if model != "auto":
                command += ["--native-model", model]
            if params["baseline_model"] != "auto":
                command += ["--baseline-model", params["baseline_model"]]
            effective["source_run"] = str(source)
        elif aid == "hourly_evaluation":
            source = self._input_path(params["backtest_file"])
            columns = self._backtest_columns(source)
            for column in ["delivery_start_utc", params["actual"], params["baseline"], params["candidate"]]:
                if column not in columns:
                    raise ValueError(f"Colonne absente du backtest : {column}")
            if params["baseline"] == params["candidate"]:
                raise ValueError("Choisissez deux colonnes de modèles différentes.")
            command += [str(source), "--output-dir", str(output)]
            for name in ("baseline", "candidate", "actual", "timezone", "bootstrap_samples", "seed"):
                command += ["--" + name.replace("_", "-"), str(params[name])]
            effective["backtest_file"] = str(source)
            warnings.append("Le CLI vérifiera les jours complets et contigus avant de calculer les métriques.")
        else:
            source_config = self.project_root / config_id if snapshot_config is None else None
            effective = copy.deepcopy(_read_yaml(source_config) if source_config else snapshot_config)
            _no_secrets(effective)
            if model != selected["model"]:
                raise ValueError("Le modèle doit correspondre à la configuration sélectionnée.")
            data = effective["data"]
            original_base = source_config.parent if source_config else self.project_root
            data["project_root"] = str(_absolute(data.get("project_root", "."), original_base))
            data["pit_vintage_dir"] = str(_absolute(data.get("pit_vintage_dir", "."), original_base))
            data["source"] = "cache"
            for zone in effective["zones"].values():
                specs = [zone.get("target", {}), *zone.get("covariates", {}).values()]
                for spec in specs:
                    if spec.get("file"):
                        spec["file"] = str(_absolute(spec["file"], original_base))
                    if spec.get("source", "auto") in {"auto", "saturn", "cache"}:
                        spec["source"] = "cache"
                    elif spec.get("source") != "pit_parquet" and not spec.get("file"):
                        raise ValueError("Le lancement exige des sources PIT, fichiers ou caches locaux.")
            effective.setdefault("output", {})["directory"] = str(output)
            effective.setdefault("report", {})["filename"] = str(output / "report.html")
            effective["model"]["local_files_only"] = True
            effective["model"]["device"] = params["device"]
            # Reuse the pipeline's zone/series contract without training or fetching.
            from chronos2_modular.common import build_zone_configs
            build_zone_configs(effective, [selected["zone"]], None, None)
            command += ["--config", str(snapshot), "--zone", selected["zone"], "--output-dir", str(output),
                        "--device", params["device"], "--local-files-only", "--log-level", params["log_level"]]
            resources = ["scientific-cache"]
            warnings += ["Calcul potentiellement long : backtest et prévision selon les paramètres scientifiques conservés.",
                         "Lecture des caches locaux uniquement; aucune actualisation Saturn. Un cache absent ou incomplet provoque un échec.",
                         "Évitez les modifications externes des mêmes caches pendant ce run (les processus PowerShell externes ne sont pas contrôlés)."]
        _no_secrets(effective)
        result = {"adapter_id": aid, "type": adapter["type"], "model": model, "command": command,
                  "cwd": str(self.project_root), "config": effective, "config_path": str(snapshot),
                  "output_dir": str(output), "warnings": warnings, "resource_keys": resources,
                  "request": {"adapter_id": aid, "config_id": config_id, "model": model, "parameters": params}}
        if source_config:
            result["source_config"] = str(source_config)
            result["source_config_sha256"] = hashlib.sha256(source_config.read_bytes()).hexdigest()
        if write:
            root.mkdir(parents=True, exist_ok=True)
            with snapshot.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(effective, stream, allow_unicode=True, sort_keys=False)
        return result
