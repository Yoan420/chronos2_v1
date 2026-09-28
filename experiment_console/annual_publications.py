"""Read sealed annual forecasts from scheduled or command-line launches."""
from datetime import date
import hashlib
import json
from pathlib import Path

COUNTRIES = ("FR", "DE", "NL", "BE")
PROTOCOL = "nyx_annual_cpu_prospective_consumer_v1"


def _directory(root, day):
    if date.fromisoformat(day).isoformat() != day:
        raise ValueError("Date de livraison invalide.")
    project = Path(root).resolve()
    base = (project / "runs/nyx_annual_cpu_live").resolve()
    if not base.is_relative_to(project):
        raise ValueError("Dossier de publication hors du projet.")
    path = base / day
    if path.is_symlink() or path.resolve().parent != base:
        raise ValueError("Chemin de publication non autorisé.")
    return path


def _receipt(directory, day):
    path = directory / "receipt.json"
    if path.is_symlink() or path.stat().st_size > 8_000_000:
        raise ValueError("Reçu de publication non autorisé.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("protocol") != PROTOCOL or value.get("status") != "COMPLETE"
            or value.get("delivery_day") != day or set(value.get("countries", {})) != set(COUNTRIES)
            or value.get("future_labels_used") is not False or value.get("Storm_used_as_model_input") is not False
            or not isinstance(value.get("qualification_sha256"), str) or len(value["qualification_sha256"]) != 64):
        raise ValueError("Publication annuelle incomplète ou reçu invalide.")
    return value


def _file(directory, day, country, kind, record, limit):
    if country not in COUNTRIES or kind not in ("html", "csv"):
        raise ValueError("Pays ou format non autorisé.")
    target = directory / "zones" / country / f"forecast_{country.lower()}_{day}_nyx_annual_cpu.{kind}"
    if target.is_symlink() or not target.resolve().is_relative_to(directory.resolve()) or target.stat().st_size > limit:
        raise ValueError("Fichier de publication non autorisé.")
    body = target.read_bytes()
    if len(body) > limit or hashlib.sha256(body).hexdigest() != record.get(kind + "_sha256"):
        raise ValueError("La publication annuelle a été modifiée.")
    return target, body


def read_annual_artifact(root, day, country, kind, limit=64_000_000):
    directory = _directory(root, day)
    receipt = _receipt(directory, day)
    if country not in COUNTRIES:
        raise ValueError("Pays annuel invalide.")
    return _file(directory, day, country, kind, receipt["countries"][country], limit)


def annual_publications(root):
    base = Path(root) / "runs/nyx_annual_cpu_live"
    days, warnings = [], []
    if not base.is_dir():
        return {"days": days, "warnings": warnings}
    for directory in sorted(base.iterdir(), reverse=True):
        if len(days) >= 30:
            break
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        try:
            day = directory.name
            directory = _directory(root, day)
            if not (directory / "receipt.json").exists():
                continue
            receipt = _receipt(directory, day)
            countries = []
            for zone in COUNTRIES:
                record = receipt["countries"][zone]
                _file(directory, day, zone, "csv", record, 64_000_000)
                has_report = "html_sha256" in record
                if has_report:
                    _file(directory, day, zone, "html", record, 64_000_000)
                countries.append({"zone": zone, "hours": record["hours"], "report_available": has_report})
            days.append({"delivery_day": day, "countries": countries,
                         "de_price_performance_exception_used": receipt.get("de_price_performance_exception_used", False)})
        except (OSError, ValueError, KeyError, TypeError):
            warnings.append(f"Publication {directory.name} incomplète ou modifiée : consulter le journal du lancement.")
    return {"days": days, "warnings": warnings[:10]}
