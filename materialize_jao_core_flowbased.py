"""Reconstruct an auditable JAO Core CNEC/RAM feature history.

The default support is 731 delivery days: 365 days used to train every
rolling origin, the 365 days displayed in the comparable report, and the live
delivery day. Pass ``--training-days 0 --future-days 0`` when only the
descriptive 365-day CNEC/RAM audit is required.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import json
import os
from pathlib import Path
import ssl
import threading
from typing import Any, Mapping, Sequence

import certifi
import pandas as pd

from chronos2_hourly.jao_flowbased import (
    FLOWBASED_FEATURE_COLUMNS,
    FLOWBASED_SCHEMA_VERSION,
    JaoCoreClient,
    JaoFetchResult,
    JaoFlowBasedError,
    assemble_flowbased_feature_store,
    build_causal_empty_day_fallback,
    build_hourly_flowbased_features,
    build_windows_trust_context,
    normalise_initial_computation,
    sha256_file,
    write_daily_flowbased_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "pit" / "jao_core_flowbased"


def _load_ca_bundle(path: Path, *, source: str) -> tuple[ssl.SSLContext, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Bundle CA introuvable ({source}): {resolved}. Fournissez un vrai "
            "fichier PEM ou retirez --ca-bundle pour utiliser la detection "
            "automatique."
        )
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        context.load_verify_locations(cafile=str(resolved))
    except (OSError, ssl.SSLError) as exc:
        raise JaoFlowBasedError(
            f"Bundle CA invalide ou illisible ({source}): {resolved}."
        ) from exc
    return context, f"{source}:{resolved}"


def _tls_configuration(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool | ssl.SSLContext, str]:
    """Resolve a verified TLS source without ever falling back to insecure."""

    if args.insecure:
        return False, "disabled_by_explicit_option"
    if args.ca_bundle is not None:
        return _load_ca_bundle(args.ca_bundle, source="explicit_ca_bundle")

    environment = os.environ if environ is None else environ
    for variable in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        configured = str(environment.get(variable, "")).strip()
        if configured:
            return _load_ca_bundle(
                Path(configured), source=f"environment_{variable}"
            )

    if os.name == "nt":
        context, roots = build_windows_trust_context()
        return context, f"windows_root_store:{roots}_roots"
    return ssl.create_default_context(cafile=certifi.where()), "certifi_default"


def _resolve_window(args: argparse.Namespace) -> tuple[date, date]:
    end = args.end_day
    if args.start_day is not None:
        start = args.start_day
    else:
        support_days = (
            int(args.evaluation_days)
            + int(args.training_days)
            + int(args.future_days)
        )
        if support_days < 1:
            raise ValueError("evaluation_days + training_days doit etre positif.")
        start = end - timedelta(days=support_days - 1)
    if start > end:
        raise ValueError("--start-day doit etre anterieur a --end-day.")
    return start, end


def _existing_partition(
    root: Path,
    day: date,
) -> dict[str, Any] | None:
    token = day.isoformat()
    raw_path = root / "raw" / "initialComputation" / f"{token}.json.gz"
    audit_path = root / "raw" / "initialComputation" / f"{token}.audit.json"
    normalised_path = root / "normalised" / f"{token}.parquet"
    feature_path = root / "daily_features" / f"{token}.parquet"
    paths = (raw_path, audit_path, normalised_path, feature_path)
    if not any(path.exists() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise JaoFlowBasedError(f"Partition JAO partielle pour {token}.")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise JaoFlowBasedError(f"Audit JAO illisible pour {token}.") from exc
    checks = {
        raw_path: audit.get("raw_gzip_sha256"),
        normalised_path: audit.get("normalised_sha256"),
        feature_path: audit.get("features_sha256"),
    }
    for path, expected in checks.items():
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise JaoFlowBasedError(f"Checksum JAO invalide: {path}.")
    if audit.get("schema_version") != FLOWBASED_SCHEMA_VERSION or tuple(
        audit.get("feature_columns", ())
    ) != FLOWBASED_FEATURE_COLUMNS:
        raise JaoFlowBasedError(
            f"Partition JAO {token} issue d'un ancien schema; "
            "relancer avec --overwrite."
        )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collecte JAO Core initialComputation/Presolved, archive les CNEC/RAM "
            "bruts et produit les agregats horaires du POC kalman_flowbased."
        )
    )
    parser.add_argument("--end-day", type=date.fromisoformat, required=True)
    parser.add_argument("--start-day", type=date.fromisoformat)
    parser.add_argument(
        "--evaluation-days",
        type=int,
        default=365,
        help="Jours affiches dans le rapport comparable (defaut: 365).",
    )
    parser.add_argument(
        "--training-days",
        type=int,
        default=365,
        help=(
            "Warm-up de chaque origine rolling; 365 implique 730 jours de "
            "support au total. Utiliser 0 pour un audit descriptif seul."
        ),
    )
    parser.add_argument(
        "--future-days",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "Ajoute le jour live a la suite des 365+365 jours historiques "
            "(defaut: 1). Mettre 0 pour l'audit descriptif seul."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--page-size", type=int, default=40000)
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 2),
        default=2,
        help="Deux lectures paralleles maximum; le cache reste journalier.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--maximum-retries", type=int, default=4)
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=0.65,
        help="Protection de la limite publique JAO (environ 100 requetes/min).",
    )
    parser.add_argument(
        "--ca-bundle",
        type=Path,
        help="Bundle CA PEM du proxy d'entreprise, si necessaire.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Desactive TLS uniquement si le proxy local ne fournit aucun CA.",
    )
    parser.add_argument(
        "--allow-non-pit",
        action="store_true",
        help=(
            "Assemble aussi les partitions dont JAO lastModifiedOn depasse le "
            "cutoff. Elles restent interdites au POC Kalman strict."
        ),
    )
    parser.add_argument(
        "--require-operational-pit",
        action="store_true",
        help=(
            "Exige une capture locale effectuee avant D-1 08:00. Ce mode est "
            "destine aux futures captures live, pas au backfill historique."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.evaluation_days < 1 or args.training_days < 0:
        raise ValueError("evaluation_days >= 1 et training_days >= 0 sont requis.")
    if args.insecure and args.ca_bundle is not None:
        raise ValueError("--insecure et --ca-bundle sont mutuellement exclusifs.")
    if args.allow_non_pit and args.require_operational_pit:
        raise ValueError(
            "--allow-non-pit et --require-operational-pit sont incompatibles."
        )
    verify, tls_trust_source = _tls_configuration(args)
    start, end = _resolve_window(args)
    days = [item.date() for item in pd.date_range(start, end, freq="D")]
    output_root = args.output_root.expanduser().resolve()
    print(
        f"[JAO] support {start} -> {end}: {len(days)} jours; "
        f"evaluation={args.evaluation_days}, training={args.training_days}, "
        f"future={args.future_days}.",
        flush=True,
    )
    print(
        "[JAO] Entree modele: initialComputation + Presolved=true. "
        "Les publications post-coupling restent hors features.",
        flush=True,
    )
    if args.insecure:
        print(
            "[JAO] AVERTISSEMENT: verification TLS desactivee; le manifeste le "
            "signalera et cette collecte ne doit pas etre promue telle quelle.",
            flush=True,
        )
    else:
        print(f"[JAO] TLS verifie via {tls_trust_source}.", flush=True)
    if args.dry_run:
        print(
            f"[JAO] DRY-RUN: environ {len(days)} requetes si chaque jour tient "
            f"dans une page de {args.page_size} lignes."
        )
        return 0
    reused = 0
    downloaded = 0
    pit_violations = 0
    pending: list[date] = []
    repair_days: set[date] = set()
    for position, day in enumerate(days, start=1):
        existing = None if args.overwrite else _existing_partition(output_root, day)
        if existing is None:
            pending.append(day)
            continue
        if not args.allow_non_pit and existing.get("pit_eligible") is not True:
            pending.append(day)
            repair_days.add(day)
            print(
                f"[JAO] scan {position}/{len(days)} {day}: cache non PIT; "
                "reparation causale planifiee.",
                flush=True,
            )
            continue
        reused += 1
        pit_violations += int(existing.get("pit_eligible") is not True)
        if position == 1 or position == len(days) or position % 25 == 0:
            print(
                f"[JAO] scan {position}/{len(days)} {day}: cache valide.",
                flush=True,
            )

    thread_state = threading.local()
    clients: list[JaoCoreClient] = []
    clients_lock = threading.Lock()

    def client_for_thread() -> JaoCoreClient:
        client = getattr(thread_state, "jao_client", None)
        if client is None:
            client = JaoCoreClient(
                page_size=args.page_size,
                timeout_seconds=args.timeout_seconds,
                maximum_retries=args.maximum_retries,
                # Keep the aggregate request rate below the public limit.
                request_interval_seconds=(
                    args.request_interval_seconds * int(args.workers)
                ),
                verify=verify,
            )
            thread_state.jao_client = client
            with clients_lock:
                clients.append(client)
        return client

    def materialize_day(
        day: date,
    ) -> tuple[
        date,
        int,
        int,
        Mapping[str, Any],
        JaoFetchResult | None,
        pd.DataFrame | None,
    ]:
        fetch = client_for_thread().fetch_initial_day(day)
        normalised, audit = normalise_initial_computation(
            fetch, delivery_day=day
        )
        if normalised.empty or (
            not args.allow_non_pit and audit.get("pit_eligible") is not True
        ):
            return day, 0, 0, audit, fetch, normalised
        features = build_hourly_flowbased_features(
            normalised, daily_audit=audit
        )
        write_daily_flowbased_bundle(
            output_root,
            delivery_day=day,
            fetch=fetch,
            normalised=normalised,
            features=features,
            audit=audit,
            overwrite=bool(args.overwrite or day in repair_days),
            tls_verification=not args.insecure,
            tls_trust_source=tls_trust_source,
        )
        return day, len(normalised), len(features), audit, None, None

    executor = ThreadPoolExecutor(
        max_workers=int(args.workers), thread_name_prefix="jao-flowbased"
    )
    futures = {executor.submit(materialize_day, day): day for day in pending}
    fallback_publications: dict[
        date, tuple[JaoFetchResult, pd.DataFrame, Mapping[str, Any]]
    ] = {}
    try:
        for completed, future in enumerate(as_completed(futures), start=1):
            failed_day = futures[future]
            try:
                day, rows, hours, audit, empty_fetch, empty_normalised = (
                    future.result()
                )
            except Exception as exc:
                raise JaoFlowBasedError(
                    f"Echec de la partition JAO {failed_day}: {exc}"
                ) from exc
            downloaded += 1
            if empty_fetch is not None and empty_normalised is not None:
                fallback_publications[day] = (
                    empty_fetch,
                    empty_normalised,
                    audit,
                )
                fallback_reason = (
                    "publication initiale vide"
                    if empty_normalised.empty
                    else f"publication non PIT ({audit.get('pit_status')})"
                )
                print(
                    f"[JAO] download {completed}/{len(pending)} {day}: "
                    f"{fallback_reason}; fallback causal differe.",
                    flush=True,
                )
                continue
            pit_violations += int(audit.get("pit_eligible") is not True)
            print(
                f"[JAO] download {completed}/{len(pending)} {day}: "
                f"rows={rows}, heures={hours}, "
                f"PIT={audit.get('pit_status')}.",
                flush=True,
            )
    except Exception:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        for client in clients:
            client.close()
    if fallback_publications:
        seed_downloaded = 0
        with JaoCoreClient(
            page_size=args.page_size,
            timeout_seconds=args.timeout_seconds,
            maximum_retries=args.maximum_retries,
            request_interval_seconds=args.request_interval_seconds,
            verify=verify,
        ) as seed_client:

            def previous_admissible_partition(
                target_day: date,
            ) -> tuple[date, Mapping[str, Any]]:
                nonlocal seed_downloaded
                for lag in range(1, 32):
                    candidate = target_day - timedelta(days=lag)
                    existing = (
                        None
                        if args.overwrite
                        else _existing_partition(output_root, candidate)
                    )
                    if existing is not None:
                        source_tls_allowed = bool(
                            args.insecure
                            or existing.get("tls_verification") is True
                        )
                        if (
                            existing.get("pit_eligible") is True
                            and source_tls_allowed
                        ):
                            return candidate, existing
                        continue
                    fetch = seed_client.fetch_initial_day(candidate)
                    normalised, audit = normalise_initial_computation(
                        fetch, delivery_day=candidate
                    )
                    seed_downloaded += 1
                    if normalised.empty or audit.get("pit_eligible") is not True:
                        continue
                    features = build_hourly_flowbased_features(
                        normalised, daily_audit=audit
                    )
                    metadata = write_daily_flowbased_bundle(
                        output_root,
                        delivery_day=candidate,
                        fetch=fetch,
                        normalised=normalised,
                        features=features,
                        audit=audit,
                        overwrite=bool(args.overwrite),
                        tls_verification=not args.insecure,
                        tls_trust_source=tls_trust_source,
                    )
                    print(
                        f"[JAO] seed causal ajoute: {candidate}.", flush=True
                    )
                    return candidate, metadata
                raise JaoFlowBasedError(
                    f"Aucune publication initiale admissible trouvee dans les "
                    f"31 jours precedant {target_day}."
                )

            for day in sorted(fallback_publications):
                fetch, normalised, raw_audit = fallback_publications[day]
                previous_day, previous_audit = previous_admissible_partition(day)
                previous_path = (
                    output_root
                    / "daily_features"
                    / f"{previous_day.isoformat()}.parquet"
                )
                previous_features = pd.read_parquet(previous_path)
                fallback_features, fallback_audit = (
                    build_causal_empty_day_fallback(
                        previous_features,
                        daily_audit=raw_audit,
                        previous_audit=previous_audit,
                        previous_day=previous_day,
                    )
                )
                fallback_audit = {
                    **dict(fallback_audit),
                    "fallback_source_features_sha256": previous_audit.get(
                        "features_sha256"
                    ),
                    "fallback_source_tls_verification": bool(
                        previous_audit.get("tls_verification") is True
                    ),
                }
                fallback_tls_verification = bool(
                    not args.insecure
                    and previous_audit.get("tls_verification") is True
                )
                write_daily_flowbased_bundle(
                    output_root,
                    delivery_day=day,
                    fetch=fetch,
                    normalised=normalised,
                    features=fallback_features,
                    audit=fallback_audit,
                    overwrite=bool(args.overwrite or day in repair_days),
                    tls_verification=fallback_tls_verification,
                    tls_trust_source=tls_trust_source,
                )
                pit_violations += int(
                    fallback_audit.get("pit_eligible") is not True
                )
                print(
                    f"[JAO] fallback {day} <- {previous_day}: "
                    f"{len(fallback_features)} heures, disponibilite=0.",
                    flush=True,
                )
        if seed_downloaded:
            print(
                f"[JAO] seeds historiques telecharges={seed_downloaded}.",
                flush=True,
            )
    store, manifest = assemble_flowbased_feature_store(
        output_root,
        start_day=start,
        end_day=end,
        require_research_pit=not args.allow_non_pit,
        require_operational_pit=bool(args.require_operational_pit),
        overwrite=bool(args.overwrite or repair_days),
    )
    print(
        f"[JAO] Termine: telecharges={downloaded}, reutilises={reused}, "
        f"violations_PIT={pit_violations}.",
        flush=True,
    )
    print(f"[JAO] Features: {store}", flush=True)
    print(f"[JAO] SHA-256: {manifest['parquet_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
