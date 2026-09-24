"""Read-only accounting of forecast layers; never estimate missing weights."""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd


def attach_forecast_components(result: Any, forecast: pd.DataFrame, *, model: str) -> None:
    """Attach issued prices, not fitted coefficients or causal contributions.

    Only the known incumbent chain is supported. LoRA and other recipes must
    provide their own upstream identities rather than borrow Chronos columns.
    """
    if model not in {"residual_corrected", "residual_kalman", "residual_kalman_weather",
                     "residual_kalman_hybrid", "residual_kalman_flowbased"}:
        return
    columns = ["chronos2__q50", "residual_corrected__q50", f"{model}__q50"]
    if "delivery_start_utc" not in forecast or not set(columns).issubset(forecast):
        return
    index = pd.DatetimeIndex(pd.to_datetime(forecast["delivery_start_utc"], utc=True))
    expected = pd.DatetimeIndex(pd.to_datetime(result.forecast_native["timestamp"], utc=True))
    if index.has_duplicates or not index.equals(expected):
        raise ValueError("Timeline des composantes différente du forecast affiché.")
    values = forecast.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not np.allclose(
        values[:, 2], result.forecast_native["q50"].to_numpy(float), rtol=0, atol=1e-9
    ):
        raise ValueError("Les composantes ne reconstruisent pas le forecast affiché.")
    result.forecast_components = pd.DataFrame({
        "chronos_q50": values[:, 0],
        "residual_correction": values[:, 1] - values[:, 0],
        "kalman_correction": values[:, 2] - values[:, 1],
        "final_q50": values[:, 2],
    }, index=index)
    result.forecast_components_has_kalman = model != "residual_corrected"


def build_forecast_components_html(result: Any) -> str:
    frame = getattr(result, "forecast_components", None)
    if frame is None:
        return ""
    labels = [("chronos_q50", "Chronos-2 — prix de départ"),
              ("residual_correction", "Correction résiduelle ajoutée")]
    if getattr(result, "forecast_components_has_kalman", False):
        labels.append(("kalman_correction", "Correction Kalman ajoutée"))
    labels.append(("final_q50", "Prix final du modèle"))
    rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{frame[key].mean():.2f}</td></tr>"
        for key, label in labels
    )
    return (
        '<div data-report-section="forecast-components"><h3>Du modèle de base au prix final</h3>'
        '<p class="muted">Moyennes sur les heures physiques de la livraison affichée : '
        'prix Chronos-2 + correction résiduelle + éventuelle correction Kalman = prix final. '
        'Ces montants proviennent du forecast émis, sans réentraînement. Ce sont des '
        'ajustements en EUR/MWh, pas des poids de variables.</p><div class="table-wrap">'
        '<table><thead><tr><th>Étape</th><th>Moyenne (EUR/MWh)</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div></div>'
    )


def attribution_scope_html(raw: Any) -> str:
    """Always explain the role of target history, even for legacy artifacts."""
    hourly = raw.get("hourly") if isinstance(raw, dict) else None
    has_prices = isinstance(hourly, pd.DataFrame) and "historical_target_price" in set(
        hourly["variable_key"].astype(str)
    )
    if raw is None:
        status = (
            "Attribution chiffrée indisponible pour cet artefact : aucun poids n'est inventé. "
            "Les nouveaux runs compatibles calculent cette explication après gel du forecast."
        )
    elif has_prices:
        status = (
            "Le groupe « Prix passés » mesure l'influence de l'historique cible fourni à "
            "Chronos-2, et son effet transmis au correcteur. Il n'inclut ni les prix futurs "
            "observés, ni l'effet d'un réentraînement sur d'autres prix."
        )
    else:
        status = (
            "Cet ancien artefact attribue les variables physiques uniquement. Le poids "
            "des prix passés n'y a pas été calculé : ils étaient maintenus fixes, ce qui "
            "ne signifie pas une influence nulle."
        )
    return (
        '<div class="attribution-audit-note" data-report-section="attribution-methodology">'
        '<h4>Comment lire les influences — variables et prix passés</h4>'
        '<p>Les prix passés sont bien une entrée : la série cible historique constitue le '
        'contexte de Chronos-2, même si elle ne figure pas dans la liste des covariables. '
        'Les prévisions fondamentales complètent ce contexte.</p>'
        f'<p>{status}</p>'
        '<p>Une influence Shapley est une sensibilité locale à une référence historique '
        '(médiane par heure de la semaine sur 56 jours), avec les paramètres appris figés. '
        'La référence utilise uniquement le contexte historique fourni au modèle. Une '
        'contribution positive relève le prix par rapport à cette référence ; une contribution '
        'négative le baisse. Le calendrier reste fixe. Les dépendances entre variables et '
        'le choix de référence influencent le résultat : ce ne sont ni des coefficients '
        'constants du réseau ni des effets économiques causaux.</p>'
        '<p>Part relative (%) = somme des contributions absolues de la variable sur les '
        'heures du jour / somme des contributions absolues de tous les groupes expliqués. '
        'Ce pourcentage ne mesure pas une part du prix ; il vaut zéro si toutes les '
        'contributions sont nulles. Pour Kalman, l’attribution amont et l’ajustement final '
        'sont distingués : les états internes du filtre ne sont pas attribués par variable.</p></div>'
    )
