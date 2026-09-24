# SolarWind : interaction dans le correcteur et plafond asymétrique

Protocole fixé le 22 septembre 2026, après accord explicite de l'utilisateur
pour lancer les trois variantes DE/NL. Expérience isolée
`solar_wind_corrector_interaction_v1` ; aucune promotion ni modification de
production, des configurations originales, des sources ou des résultats scellés.
Cette hypothèse fait suite à l'analyse des résultats : elle est post-hoc,
et non une validation prospective.

## Plan factoriel fixé avant calcul

| Variante | Interaction supplémentaire dans CatBoost | Correction CatBoost autorisée |
| --- | --- | --- |
| Référence SolarWind scellée | Non | [−40, +40] €/MWh |
| Interaction seule | Oui | [−40, +40] €/MWh |
| Plafond seul | Non | [−40, +80] €/MWh |
| Interaction et plafond | Oui | [−40, +80] €/MWh |

Le relèvement est uniquement une autorisation : aucune hausse n'est imposée,
aucun signe n'est forcé et aucune règle manuelle de prix n'est ajoutée.
La perte MAE, la recette CatBoost, la fenêtre d'entraînement et les autres
hyperparamètres restent ceux de la référence. Les deux variantes avec
interaction partagent les mêmes corrections brutes, puis appliquent leur
plafond respectif ; le plafond n'est pas une nouvelle cible d'entraînement.
Les limites de −40 et +80 ne seront pas choisies à nouveau sur les résultats.

## Interaction causale, uniquement dans le correcteur

Le test reprend sans modification `build_interaction`, déjà utilisé dans
l'expérience Kalman seule, à partir des prévisions du même pays :

```
lowW = clip(1 - W / q75_W, 0, 1)
lowS = clip(1 - S / q75_positive_S, 0, 1)
highRL = clip((RL - q50_RL) / (q90_RL - q50_RL), 0, 1)
score = lowW * lowS * highRL
```

Les quantiles sont estimés sur les seuls jours antérieurs disponibles dans
[D−365, D), jamais D ni les 365 jours d'évaluation pris ensemble. Le q75 du
vent conserve les vrais zéros ; celui du solaire utilise les valeurs positives.
Les 14 premiers jours d'historique ont un score nul, explicitement audité.
Après ce warm-up, des échelles invalides bloquent le calcul. Les heures
physiques des jours DST de 23/25 heures sont conservées. Aucune production
n'est imputée ; aucune nouvelle source ni nouvelle soustraction du renouvelable
à la charge résiduelle n'est introduite. Le score n'est pas une probabilité
calibrée de spike et n'utilise pas le prix réalisé.

Le score est strictement identique à celui du premier test sur son support
scellé de 730 jours historiques plus le jour de forecast. Certains jours
Chronos de calibration précèdent ce support (12 jours pour DE) : leur
interaction est explicitement fixée à zéro, faute de calibration du score
sur ce préfixe. Cette initialisation diagnostique est auditée séparément du
warm-up de 14 jours. Elle n'impute ni vent, ni solaire, ni charge résiduelle,
et n'étend pas rétrospectivement l'échantillon de normalisation.

L'interaction supplémentaire entre uniquement dans la matrice du correcteur
CatBoost. Elle n'entre ni dans Chronos ni dans les covariables Kalman. Les
prévisions Chronos sont figées. Kalman est rejoué sur chaque nouvel amont
corrigé, avec ses covariables, candidats, paramètres, règles de gouvernance
et plafonds de référence inchangés. Ses coefficients et choix appris peuvent
donc changer en réponse aux résidus, sans changer sa recette.

## Références et garanties de calcul

- DE : `solar_wind_v1/2026-09-22/de/45ec8314c37a2fe5`.
- NL : `solar_wind_v1/2026-09-22/nl/b1da388bb4df6c63`.

Le chargeur doit vérifier les identités, inventaires scellés, snapshots et
empreintes des sources et fichiers scientifiques. Il ne synchronise pas de
source. La nouvelle identité inclut les références, paramètres et code du test.
Une reprise ne peut réutiliser que des caches de cette même identité ; aucun
résultat ni checkpoint de référence n'est écrasé ou supprimé.

Le correcteur est réentraîné chronologiquement, uniquement sur les labels
autorisés par la recette originale avant le jour à prévoir. On ne soumet pas
le prix réel de livraison au modèle. Les quantiles Chronos reçoivent le même
décalage résiduel : la correction doit conserver leur largeur et leur ordre.
Les contrôles de référence doivent réussir avant l'évaluation des variantes.
Le correcteur est contrôlé au premier jour historiquement entraîné, puis au
premier, milieu et dernier jour d'évaluation et au forecast de livraison
(dates dédoublonnées). Kalman est contrôlé au premier, milieu et dernier
jour d'évaluation, ainsi qu'au forecast. La tolérance est de 1e−9. La portée
exacte est publiée dans les audits ; ces contrôles initiaux échantillonnés
ne constituent pas une reproduction annuelle intégrale. Pendant le replay,
la correction sans interaction replafonnée à ±40 est à nouveau comparée
chaque jour à la référence, avant de produire les variantes.

## Période et mesures

Comparaisons appariées sur les 365 jours du **22/09/2025 au 21/09/2026**,
soit 8 760 heures par pays. Le forecast de livraison du **22/09/2026** est
séparé, sans observation injectée. Ne pas comparer directement les agrégats
avec un ancien rapport utilisant une autre fenêtre de 365 jours.

Pour chaque variante et la référence : MAE, RMSE et biais globaux, résultats
mensuels, erreurs conditionnelles aux prix réalisés ≥200 et ≥300 €/MWh,
vrais pics, faux pics, pics manqués, précision et rappel aux mêmes seuils.
Les dépassements bruts et la fréquence d'activation des plafonds doivent être
audités. Une amélioration sur les pics sera confrontée à son coût éventuel
sur les autres heures et aux faux pics, pas seulement au score global.

## Exécution isolée

```
python -B -u run_solar_wind_corrector_interaction.py --action validate --zones DE NL --threads 2 --workers 2
python -B -u run_solar_wind_corrector_interaction.py --action run --zones DE NL --threads 2 --workers 2
```

Utiliser le Python du venv `pricefm311`. `validate` est en lecture seule :
pas de fit, pas de source synchronisée, pas d'écriture de sortie ni de verrou
d'exécution. Le lancement réel utilise une fenêtre cachée, priorité BelowNormal,
stdout/stderr distincts et uniques. Un batch au plus ; pays traités
séquentiellement et limites CPU de deux threads/deux workers conservées.

Les sorties appartiennent exclusivement à
`runs/experiments/solar_wind_corrector_interaction_v1/2026-09-22/` :
statuts, identité, audits, checkpoints, backtests et forecasts, métriques,
rapport HTML et inventaire scellé. La fin exige tous les pays, toutes les
variantes, 365 jours évalués, les forecasts et les rapports attendus ; un
processus absent ou un cache présent ne suffit pas.

## Limites d'interprétation

Le test hérite des limites PIT de la référence : requêtes rétrospectives as-of
J−1 08h sans preuve certifiée de publication originale, deux substitutions
NL documentées et warm-up diagnostique. Les contrôles causaux des traitements
ne transforment pas cet historique en archive de publication certifiée.
Les dates analysées ont contribué à formuler l'hypothèse ; une éventuelle
amélioration restera exploratoire et devra être confirmée hors échantillon.
Aucun résultat n'autorise automatiquement une mise en production.
