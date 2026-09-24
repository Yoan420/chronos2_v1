# SolarWind : interaction Kalman seule, DE/NL

Test autorisé le 22 septembre 2026 après analyse des rapports SolarWind.
Expérience isolée `solar_wind_interaction_v1`, sans modification de production,
des configurations existantes, des données sources ou des bundles précédents.
Il s'agit d'une hypothèse post-hoc, pas d'une validation prospective.

## Hypothèse unique, fixée avant calcul

Une faible production éolienne ET solaire peut apporter une information
supplémentaire au Kalman lorsque la charge résiduelle prévue est élevée.
Une seule covariable par pays est ajoutée au groupe `market` (candidat
`linear_market`). Les autres candidats, les règles de gouvernance, leurs
fenêtres et leurs plafonds sont conservés. Le coefficient et son signe sont
appris : aucune majoration arbitraire des prix n'est appliquée.

Pour chaque jour civil D, avec les prévisions du même pays :

```
lowW = clip(1 - W / q75_W, 0, 1)
lowS = clip(1 - S / q75_positive_S, 0, 1)
highRL = clip((RL - q50_RL) / (q90_RL - q50_RL), 0, 1)
score = lowW * lowS * highRL
```

Les quantiles de normalisation sont calculés uniquement sur les jours
antérieurs disponibles dans [D-365,D), jamais sur D ni sur FINAL365 entier.
Le q75 du vent conserve les vrais zéros ; le q75 solaire ne prend que les
valeurs positives, évitant une échelle nocturne nulle. La nuit peut donc
activer `lowS`, mais pas le score sans faible vent ET tension résiduelle.
Il n'y a ni nouvelle soustraction du renouvelable à RL, ni nouvelle source.
Le score est un indicateur borné, pas une probabilité calibrée de spike.

Les 14 premiers jours de l'historique de calibration ont un score nul,
explicitement audité ; aucune production n'est imputée. Après ce warm-up,
une échelle invalide bloque le calcul. Les dates DST gardent leurs 23/25
heures physiques. Les seuils ne seront pas sélectionnés sur les résultats.

## Références immuables

- DE : `solar_wind_v1/2026-09-22/de/45ec8314c37a2fe5`.
- NL : `solar_wind_v1/2026-09-22/nl/b1da388bb4df6c63`.

Les prévisions Chronos et CatBoost sont réutilisées bit pour bit depuis les
bundles scellés : pas de fit CatBoost, d'inférence Chronos ni de synchronisation.
Le chargeur valide leurs fichiers, snapshots, empreintes et identités.
Les deux fichiers scientifiques Kalman doivent correspondre aux SHA épinglés.
La version pykalman et tous les paramètres de replay sont conservés.

Avant le replay modifié, le lanceur reproduit trois jours de contrôle
(premier, milieu, dernier du backtest) et le forecast de livraison. Il compare
Q10/Q50/Q90, corrections, poids et filtre choisi, à 1e-9 près. Ce contrôle
échantillonné n'est pas présenté comme une reproduction annuelle intégrale.

Le correcteur CatBoost reste plafonné à ±40 €/MWh ; Kalman reste plafonné
à ±20 avant pondération. Changer ces plafonds nécessiterait un autre test.

## Exécution et résultats

```
python -B -u run_solar_wind_interaction.py --action validate --zones DE NL --threads 2 --workers 2
python -B -u run_solar_wind_interaction.py --action run --zones DE NL --threads 2 --workers 2
```

Utiliser le Python du venv `pricefm311`. Le lancement réel s'effectue en
arrière-plan, fenêtre cachée, priorité BelowNormal, avec logs distincts.
DE puis NL sont traités séquentiellement, deux workers maximum par replay.
Un verrou système interdit les batches simultanés. Le calcul Kalman annuel
peut encore être long : 366 recalibrations par pays. Les caches historiques
ne sont publiés par le moteur qu'après retour de tous ses chunks.

Les résultats sont dans `runs/experiments/solar_wind_interaction_v1/2026-09-22/`,
avec une nouvelle identité par zone fondée sur sources, code et paramètres.
Ils comprennent les audits de feature et Kalman, contrôles de référence,
prévisions/backtest Parquet, métriques JSON, rapport HTML et reçu SHA256.
Une reprise utilise uniquement les caches du nouveau test et refuse une
identité divergente ; ne pas modifier son code pendant une exécution.

Comparaison sur les 365 jours scellés du 22/09/2025 au 21/09/2026 : MAE,
RMSE, biais, métriques mensuelles, erreurs sur prix observés ≥200/300 et
vrais/faux pics aux mêmes seuils. Le forecast du 22/09/2026 est séparé,
sans observation injectée. La fenêtre des rapports antérieurs incluant
le 22/09/2026 est différente : ne pas comparer leurs totaux directement.

Limites PIT héritées : requêtes rétrospectives as-of J−1 08h ne certifiant
pas la publication originale, deux substitutions NL documentées et warm-up
diagnostique de la référence. Aucun résultat ne déclenche de promotion.
