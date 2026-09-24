"""Explicitly uncertain ETAs from monitor observations, never scientific inputs.

An ETA is a planning estimate, not measured percent complete or a time bound.
The estimator does not mutate files, query models, inspect caches, or stop jobs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import time
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _timestamp(value: Any) -> float | None:
    if isinstance(value, datetime):
        return value.timestamp() if value.tzinfo is not None else None
    number = _number(value)
    if number is not None:
        return number
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, OverflowError, OSError):
        return None


def _iso(timestamp: float) -> str:
    try:
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return datetime.max.replace(tzinfo=timezone.utc).isoformat()


def _minutes(seconds: float) -> str:
    return f"{max(1, round(seconds / 60))} min"


class Estimator:
    """Estimate remaining wall time using only already observed monitor fields.

    ``now`` accepts Unix seconds, an aware datetime or an ISO timestamp.
    ``progress.exact is True`` or ``progress.kind == 'exact'`` is required to
    extrapolate a work counter. Country counts alone are never timed progress.
    Unknown jobs always receive a numerical but explicitly uncalibrated planning
    assumption while processes exist, as requested by the user.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._anchors: dict[tuple[str, str, str], dict] = {}

    def _anchored(self, job: dict, now: float, proposal: dict) -> dict:
        """Keep an assumed deadline fixed between explicit revision milestones."""
        key = (str(job.get("id", "")), str(job.get("started_at", "")), proposal["method"])
        anchor = self._anchors.get(key)
        revision = 0
        if anchor is not None and now >= anchor["created_at"]:
            if now < anchor["deadline"]:
                remaining = max(0.1, anchor["deadline"] - now)
                basis = anchor["basis"] + " Horizon fixé au relevé de " + _iso(anchor["created_at"]) + "; compte à rebours révisable."
                method = proposal["method"] + ("_revised" if anchor["revision"] else "")
                result = self._result(now, remaining, proposal["confidence"], basis, method)
                # Keep the original uncertainty horizons too: polling alone is
                # not new evidence that should collapse uncertainty near the ETA.
                result["low_seconds"] = round(max(0, anchor["low_deadline"] - now), 1)
                result["high_seconds"] = round(max(remaining, anchor["high_deadline"] - now), 1)
                return result
            revision = anchor["revision"] + 1
        basis = proposal["basis"]
        if revision:
            basis += f" L’horizon précédent est dépassé : révision explicite n°{revision}, sans preuve d’avancement supplémentaire."
        anchor = {"created_at": now, "deadline": now + proposal["remaining_seconds"],
                  "low_deadline": now + proposal["low_seconds"], "high_deadline": now + proposal["high_seconds"],
                  "basis": basis, "revision": revision}
        self._anchors[key] = anchor
        if len(self._anchors) > 256:
            oldest = min(self._anchors, key=lambda item: self._anchors[item]["created_at"])
            self._anchors.pop(oldest, None)
        basis += " Horizon fixé au relevé de " + _iso(now) + "; compte à rebours révisable."
        method = proposal["method"] + ("_revised" if revision else "")
        return self._result(now, proposal["remaining_seconds"], proposal["confidence"], basis, method)

    @staticmethod
    def _result(now: float, remaining: float | None, confidence: str, basis: str,
                method: str, *, low_factor: float = 0.25, high_factor: float = 4.0) -> dict:
        if remaining is None:
            low = high = end = None
        else:
            # Finite, nonnegative public numbers even for malformed old metadata.
            remaining = round(max(0.0, min(remaining, 10 * 365 * 24 * 3600)), 1)
            low, high = round(remaining * low_factor, 1), round(remaining * high_factor, 1)
            end = _iso(now + remaining)
        return {
            "remaining_seconds": remaining, "low_seconds": low, "high_seconds": high,
            "estimated_end_at": end, "confidence": confidence, "basis": basis,
            "method": method, "as_of": _iso(now),
        }

    @staticmethod
    def _elapsed(job: dict, now: float) -> float:
        started = _timestamp(job.get("started_at"))
        if started is not None and 0 <= started <= now:
            return now - started
        elapsed = _number(job.get("elapsed_seconds"))
        return max(0.0, elapsed) if elapsed is not None else 0.0

    def _exact_counter(self, job: dict, now: float, elapsed: float) -> dict | None:
        progress = job.get("progress")
        if not isinstance(progress, dict) or not (progress.get("exact") is True or progress.get("kind") == "exact"):
            return None
        completed, total = _number(progress.get("completed")), _number(progress.get("total"))
        if completed is None or total is None or not (0 < completed < total) or elapsed <= 0:
            return None
        remaining = max(30.0, elapsed * (total - completed) / completed)
        substantial = completed >= 3 and completed / total >= 0.1
        return self._result(
            now, remaining, "medium" if substantial else "low",
            f"Extrapolation du compteur exact ({completed:g}/{total:g}) au rythme moyen observé. "
            "Le coût des étapes restantes et la charge CPU peuvent varier ; fourchette indicative, non garantie.",
            "exact_work_counter", low_factor=0.5 if substantial else 0.25, high_factor=2 if substantial else 4,
        )

    def _interaction_reference(self, job: dict, now: float) -> dict | None:
        title = str(job.get("title", "")).lower()
        engine = job.get("_engine", job.get("engine"))
        if engine != "solar_wind_interaction_v1" and not ("solarwind" in title and "interaction" in title):
            return None
        zones = job.get("zones")
        started = _timestamp(job.get("started_at"))
        if not isinstance(zones, list) or started is None or started > now:
            return None
        zones = [zone for zone in zones if isinstance(zone, dict)]
        completed = []
        active = []
        for zone in zones:
            stamp = _timestamp(zone.get("updated_at"))
            state = str(zone.get("status", "")).lower()
            if stamp is None or not (started < stamp <= now):
                continue
            if state == "complete":
                completed.append((stamp, str(zone.get("zone", "?"))))
            elif state == "running" and zone.get("phase") == "kalman_replay":
                active.append((stamp, str(zone.get("zone", "?"))))
        # This adapter applies only to the known sequential DE/NL protocol.
        if len(active) != 1 or not completed:
            return None
        completed.sort()
        active_start, active_zone = active[0]
        if completed[-1][0] > active_start:
            return None
        durations, previous = [], started
        for finished, _ in completed:
            duration = finished - previous
            if duration <= 0:
                return None
            durations.append(duration)
            previous = finished
        reference = statistics.median(durations)
        if reference < 30:
            return None
        zone_elapsed = now - active_start
        unfinished = sum(str(zone.get("status", "")).lower() != "complete" for zone in zones)
        later_zones = max(0, unfinished - 1)
        reference_zones = ", ".join(zone for _, zone in completed)
        common = (
            f"Référence observée {reference_zones} : {_minutes(reference)} par pays, du lancement à la fin "
            f"(contrôles et rapport inclus). {active_zone} est en replay depuis {_minutes(zone_elapsed)}. "
            "Comparaison entre pays du même test, pas mesure des recalibrations terminées. "
        )
        if zone_elapsed < reference:
            remaining = max(0.1, reference - zone_elapsed) + later_zones * reference
            result = self._result(
                now, remaining, "low", common +
                "Le contrôle initial est inclus dans la référence et peut allonger cette estimation. "
                "La charge CPU et le pays peuvent modifier la durée ; fourchette indicative, non garantie.",
                "observed_sequential_zone_reference", low_factor=0.5, high_factor=2.5,
            )
            result["low_seconds"] = round(max(0, 0.5 * reference - zone_elapsed) + 0.5 * later_zones * reference, 1)
            result["high_seconds"] = round(max(remaining, 2.5 * reference - zone_elapsed) + 2.5 * later_zones * reference, 1)
            return result
        overrun = zone_elapsed - reference
        # Once the reference is exceeded, zero would falsely suggest completion.
        # This revised margin is an explicit convention, not observed work left.
        remaining = max(300.0, reference * 0.25, overrun * 0.5) + later_zones * reference
        proposal = self._result(
            now, remaining, "very_low", common +
            f"Durée de référence dépassée de {_minutes(overrun)} : estimation révisée. "
            "Marge conventionnelle restante = maximum de 5 min, 25 % de la référence et 50 % du dépassement "
            "(plus les pays suivants). Aucun compteur interne ne permet de mesurer le travail restant. "
            "La fin peut encore être plus tardive ; fourchette non garantie.",
            "observed_zone_reference_overrun",
        )
        return self._anchored(job, now, proposal)

    def estimate(self, job: dict, now: Any = None) -> dict:
        stamp = _timestamp(now)
        if stamp is None or stamp < 0 or stamp > 253402000000:
            stamp = time.time()
        if not isinstance(job, dict):
            job = {}
        processes = job.get("processes")
        active = isinstance(processes, (list, tuple)) and bool(processes)
        if not active:
            if str(job.get("status", "")).lower() == "complete":
                return self._result(stamp, 0, "medium", "Calcul déclaré terminé ; aucun processus actif observé.", "complete")
            return self._result(stamp, None, "very_low", "Aucun processus actif observé : heure de fin non estimable.", "inactive")
        if str(job.get("status", "")).lower() == "complete":
            return self._anchored(job, stamp, self._result(
                stamp, 120, "very_low",
                "Résultats déclarés terminés ; processus encore présents. HYPOTHÈSE NON CALIBRÉE de sortie/nettoyage : 2 min. "
                "Ce délai n’est pas mesuré et ne garantit pas leur arrêt ; l’horizon sera révisé s’ils restent présents.",
                "process_finalization_assumption",
            ))
        elapsed = self._elapsed(job, stamp)
        reference = self._interaction_reference(job, stamp)
        if reference is not None:
            return reference
        counter = self._exact_counter(job, stamp, elapsed)
        if counter is not None:
            return counter
        benchmark = job.get("planning_estimate")
        if isinstance(benchmark, dict):
            duration = _number(benchmark.get("total_seconds"))
            basis = benchmark.get("basis")
            if duration is not None and 30 <= duration <= 365 * 86400 and isinstance(basis, str) and basis:
                explanation = ("EXTRAPOLATION DES CONTRÔLES PRÉALABLES : " + basis
                    + " Estimation de planification, pas une progression mesurée ni une garantie de fin.")
                if elapsed < duration:
                    remaining = duration - elapsed
                    result = self._result(stamp, remaining, "low", explanation,
                        "preflight_benchmark", low_factor=0.5, high_factor=2.0)
                    result["low_seconds"] = round(max(0, duration * .5 - elapsed), 1)
                    result["high_seconds"] = round(max(remaining, duration * 2 - elapsed), 1)
                    return result
                return self._anchored(job, stamp, self._result(
                    stamp, max(300, duration * .25), "very_low",
                    explanation + " L’échéance initiale est dépassée ; prolongation hypothétique explicite.",
                    "preflight_benchmark_overrun", low_factor=.25, high_factor=4,
                ))
        total = max(3600.0, 2.0 * elapsed)
        remaining = max(300.0, total - elapsed)
        proposal = self._result(
            stamp, remaining, "very_low",
            "HYPOTHÈSE NON CALIBRÉE : aucun historique comparable ni compteur de travail exact disponible. "
            "Convention de planification : durée totale = maximum de 1 h et deux fois le temps écoulé ; "
            "reste = maximum de 5 min et durée totale moins temps écoulé. "
            "La fourchette de 0,25 à 4 fois ce reste est indicative, pas une borne garantie. "
            "Cette estimation peut se décaler et ne démontre pas que le calcul finira à cette heure.",
            "uncalibrated_planning_assumption",
        )
        return self._anchored(job, stamp, proposal)
