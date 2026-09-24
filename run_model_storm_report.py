"""Assemble verified local forecasts and completed history; no collection or forecast is run."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from typing import Sequence
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parent


def delivery_date(value: str | None, *, now: datetime | None = None) -> str:
    """Use tomorrow in Paris, independently of the workstation's local timezone."""
    if value is not None:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("La livraison doit etre une date valide au format YYYY-MM-DD.") from error
        if parsed.isoformat() != value:
            raise ValueError("La livraison doit etre au format YYYY-MM-DD.")
        return value
    current = now if now is not None else datetime.now(ZoneInfo("Europe/Paris"))
    if current.tzinfo is None:
        raise ValueError("L'horloge doit inclure son fuseau horaire.")
    return (current.astimezone(ZoneInfo("Europe/Paris")).date() + timedelta(days=1)).isoformat()


def resolve_project_path(value: Path, *, project_root: Path) -> Path:
    return (value if value.is_absolute() else project_root / value).resolve()


def _without_vps_refresh(payload: dict) -> dict:
    """Remove legacy VPS results without importing clients or reading caches."""
    result = deepcopy(payload)
    result.pop("vps_snapshot", None)
    for zone in result.get("zones", []):
        zone.pop("vps_history", None)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", help="Date de livraison YYYY-MM-DD (demain a Paris par defaut).")
    parser.add_argument("--output", type=Path, help="Chemin du rapport HTML unique.")
    parser.add_argument("--nuclear-root", type=Path, help="Racine locale des resultats nucleaires.")
    parser.add_argument("--history-from-delivery", help="Extraire ce jour du backtest verifie d'un run ulterieur YYYY-MM-DD, sans calcul.")
    parser.add_argument("--skip-vps-sync", action="store_true",
                        help="Option obsolete acceptee pour compatibilite, sans effet : le rapport est toujours local.")
    args = parser.parse_args(argv)
    try:
        day = delivery_date(args.delivery_day)
        root = PROJECT_ROOT.resolve()
        output = resolve_project_path(
            args.output or Path("runs/reports/model_storm") / f"CWE_Model_Storm_{day}.html",
            project_root=root,
        )
        if output.suffix.lower() != ".html":
            raise ValueError("Le rapport de sortie doit porter l'extension .html.")
        nuclear_root = (resolve_project_path(args.nuclear_root, project_root=root)
                        if args.nuclear_root is not None else None)

        # Both strategy simulations use verified completed local price history.
        # No collector or forecast launcher is imported or run.
        from chronos2_hourly.model_storm_data import load_model_storm_payload
        from chronos2_hourly.model_storm_report import render_model_storm_report

        history_options = ({"history_from_delivery": delivery_date(args.history_from_delivery)}
                           if args.history_from_delivery is not None else {})
        print("[Model / Storm] Lecture des resultats et historiques locaux...", flush=True)
        payload = load_model_storm_payload(root, day, nuclear_root=nuclear_root, **history_options)
        if not payload.get("has_data", False):
            raise ValueError(f"Aucun resultat local exploitable pour la livraison {day}.")
        payload = _without_vps_refresh(payload)
        print("[Model / Storm] Calcul des tableaux rolling et rendu HTML...", flush=True)
        result = render_model_storm_report(payload, output)
        print(f"Rapport Model / Storm : {result}", flush=True)
        return 0
    except (ValueError, OSError, ImportError) as error:
        print(f"[Model / Storm] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
