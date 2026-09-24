# SolarWind : prime de rareté renouvelable apprise, DE/NL

Protocole fixé le 22 septembre 2026 avant l'évaluation de cette nouvelle
variante. L'hypothèse est issue de l'analyse rétrospective des expériences
précédentes : ce document ne constitue pas une préinscription antérieure à
l'observation du cas SolarWind. Il ne contient aucun résultat de la nouvelle
variante. La précision de l'utilisateur sur l'amplitude est intégrée avant
tout lancement de cette expérience : la variante principale est sans plafond
de prime, et une prime plafonnée à +40 reste un contrôle dérivé du même fit.
Les bases, fenêtres, régularisation et deux règles de sortie ci-dessous ne
doivent pas être ajustées après lecture des résultats.

L'expérience est isolée dans un nouveau namespace
`solar_wind_scarcity_premium_v1`. Elle ajoute une prime apprise aux prévisions
finales scellées de SolarWind, après Chronos, CatBoost et Kalman. Elle ne
réentraîne aucun de ces trois modèles, ne rejoue pas Kalman et ne modifie ni
la production, ni les sources, ni les configurations ou expériences
existantes. Les expériences de correcteur éventuellement en cours ne sont
pas utilisées comme entrées.

## Hypothèse et référence

NYX peut déjà identifier les heures de hausse tout en sous-estimant leur
amplitude. L'hypothèse porte donc sur la taille de l'erreur restante lorsque
les prévisions de vent et de solaire sont simultanément faibles,
particulièrement si la charge résiduelle prévue est élevée. Le supplément
apprend cette erreur résiduelle ; il n'est pas présenté comme un nouveau
détecteur indispensable des heures de hausse et ne réapplique pas
mécaniquement au prix un effet déjà capté par la chaîne.

Les références immuables sont les bundles finaux de :

- DE : `solar_wind_v1/2026-09-22/de/45ec8314c37a2fe5` ;
- NL : `solar_wind_v1/2026-09-22/nl/b1da388bb4df6c63`.

Leurs identités et empreintes doivent être vérifiées avant utilisation. Les
prévisions de référence sont préquentielles : pour chaque heure historique,
la prédiction utilisée pour construire l'erreur est celle produite sans le
prix réalisé de cette heure. Les prédictions et observations scellées restent
inchangées. Aucune série de prix de remplacement ni observation supplémentaire
ne doit être injectée pour cette comparaison.

## Trois bases fixées

Les seules variables fondamentales lues sont les prévisions de génération
éolienne `W`, de génération solaire `S` et de charge résiduelle `RL` du même
pays, en GW, selon le contrat d'origine civil J−1 à 08 h. Les générations
réalisées ne sont jamais utilisées comme variables explicatives.

Pour chaque jour civil D, les quantiles de normalisation utilisent uniquement
les jours de prévisions antérieurs disponibles dans `[D−365,D)`. D est exclu,
ainsi que tout jour ultérieur. Les prix ne participent pas à la normalisation.
Le quantile 75 du vent inclut ses vrais zéros ; celui du solaire utilise les
valeurs strictement positives pour éviter une échelle nocturne nulle.

```
lowW = clip(1 − W / q75_W, 0, 1)
lowS = clip(1 − S / q75_positive_S, 0, 1)
u = lowW × lowS
g = clip((RL − q50_RL) / (q90_RL − q50_RL), 0, 2)

b1 = u
b2 = u × g
b3 = u × max(g − 1, 0)
```

La première base ne dépend pas d'un dépassement de la médiane de charge
résiduelle. La deuxième représente l'association avec une demande résiduelle
élevée. La troisième ajoute une pente au-dessus du quantile 90 ; la borne 2
limite l'extrapolation. La charge résiduelle n'est pas recalculée en lui
soustrayant une seconde fois le vent ou le solaire.

Les bornes, quantiles et trois bases sont fixés, sans sélection de seuil sur
l'évaluation. Le contrat de normalisation conserve le démarrage documenté à
score nul pendant les 14 premiers jours d'historique disponible, puis exige
des dénominateurs strictement positifs et finis. Ce démarrage ne remplace pas
des valeurs de génération manquantes par des zéros. Les trous, doublons ou
valeurs invalides doivent être signalés, sans imputation silencieuse.

## Apprentissage causal de la prime

Pour chaque D, les erreurs d'apprentissage sont :

```
r_t = prix réalisé_t − Q50 final NYX scellé_t
```

Seuls les jours civils complets strictement antérieurs à D, dans une fenêtre
maximale de 365 jours, sont admissibles. Chaque ligne conserve ses propres
bases calculées avec l'historique disponible à sa date ; les bases du passé
ne sont pas renormalisées avec les données de D. Aucun label de D ou d'un
jour ultérieur ne peut intervenir dans le fit, le choix de ses paramètres ou
la prédiction de D.

Un minimum de **90 jours complets antérieurs** est exigé. Les 90 premiers
jours du backtest scellé servent au démarrage et ne constituent pas des jours
de test de la prime apprise. Avant cette disponibilité, la prime est nulle
et le statut de calibration doit être explicite.

Sur les n heures admissibles, les trois coefficients sont ajustés avec une
régression de moindres carrés non négatifs et une pénalisation ridge :

```
minimiser (1/n) × somme_t (r_t − b_t · alpha)²
          + 0,05 × somme_j alpha_j²
sous la contrainte alpha_j >= 0 pour j = 1, 2, 3
```

Il n'y a pas d'ordonnée à l'origine. Les trois coefficients peuvent rester
nuls. La régularisation est fixée à **0,05**, sans recherche d'hyperparamètres
sur les jours évalués. L'objectif est **L2** : il accorde davantage de poids
aux grosses erreurs et ne garantit donc pas une amélioration de la MAE.
La cible est l'erreur de la chaîne finale, et non le prix lui-même ou une
erreur d'ajustement en échantillon d'un modèle réentraîné.

La prime horaire principale, sans plafond arbitraire d'amplitude, est :

```
prime_t = max(b_t · alpha, 0)           # €/MWh, sans plafond supérieur
Q10 candidat_t = Q10 NYX_t + prime_t
Q50 candidat_t = Q50 NYX_t + prime_t
Q90 candidat_t = Q90 NYX_t + prime_t
```

Le fit porte sur la prime linéaire brute. Une seconde sortie de contrôle
utilise `prime_controle_t = min(prime_t, 40)`, puis le même décalage des trois
quantiles. Ce contrôle ne réalise **aucun fit supplémentaire** et emploie
exactement les mêmes coefficients quotidiens, bases et heures que la
variante principale. La comparaison isole donc l'effet du plafonnement de
la prime, sans nouvelle sélection de modèle ou d'hyperparamètres.

Les plafonds internes de CatBoost et de Kalman restent inchangés : la prime
est ajoutée à leur sortie finale scellée. La borne 2 de `g` limite les
variables de tension, mais n'impose aucun plafond en €/MWh à l'amplitude
apprise de la variante principale. Pour chaque sortie, le même décalage
préserve l'ordre des quantiles et la largeur de leur intervalle, mais ne
prouve pas leur calibration probabiliste après correction.

## Comparaison prévue

Les deux pays sont évalués séparément sur les mêmes heures que leur propre
référence. Le backtest final scellé couvre 365 jours du 22/09/2025 au
21/09/2026. Après les 90 jours de calibration initiale, la comparaison de la
prime apprise porte sur les **275 jours suivants**, avec des recalibrations
quotidiennes strictement causales. Les journées civiles de changement
d'heure conservent leurs 23 ou 25 heures physiques.

La livraison du 22/09/2026 est présentée séparément, sans prix réalisé utilisé
pour cette prévision ni mélangé aux métriques historiques.

Comparer la référence, la prime principale sans plafond et son contrôle +40
sur les erreurs appariées MAE, RMSE et biais, les résultats mensuels, la
fréquence et l'amplitude des primes, ainsi que les situations de faible
renouvelable avec demande résiduelle faible ou élevée. L'analyse
des hauts prix doit distinguer les pics effectivement captés des faux
positifs : relever une prévision n'est pas à lui seul une amélioration.
Les métriques sur 275 jours ne sont pas directement comparables aux totaux
sur 365 jours des précédents rapports.

## Interprétation et limites

La contrainte de signe impose une **prime non négative et non décroissante
avec les bases de rareté**, à coefficients et autres entrées fixés. Elle ne
rend pas la prévision totale monotone en toutes circonstances et ne signifie
pas « aucun vent et aucun solaire implique toujours un prix élevé ». La
demande peut être faible et les importations, l'hydraulique, le nucléaire ou
d'autres moyens peuvent compenser. L'absence de solaire la nuit est normale.
La première base peut donc produire des hausses inutiles ; les résultats
dans ce régime doivent rester visibles.

Apprendre sur les erreurs de la référence évite un double comptage
mécanique. Les variables restent néanmoins corrélées, et RL contient déjà
l'effet du renouvelable : les coefficients sont des associations prédictives,
pas une estimation causale séparée de l'effet du vent ou du solaire.
Une prime positive ne peut pas corriger une surestimation existante. Elle
peut améliorer le RMSE tout en dégradant le biais ou la MAE. Le fit L2 est
sensible aux grosses erreurs : quelques épisodes extrêmes peuvent augmenter
fortement les coefficients et provoquer des hausses excessives sur d'autres
heures. La pénalisation 0,05 réduit cette sensibilité sans supprimer le
risque. Sans plafond de sortie, les amplitudes maximales et les faux positifs
doivent donc rester visibles. À l'inverse, le contrôle +40 limite par
construction sa capacité à rattraper de très grandes sous-estimations.

Les limites des données de référence sont héritées : requêtes rétrospectives
« as-of J−1 08 h » sans certification de la publication originale, traitement
documenté des heures de changement d'heure et deux substitutions historiques
NL par des prévisions ECMWF au même cutoff. Vérifier les horodatages et les
empreintes ne lève pas ces limites de provenance PIT. Le démarrage de la
chaîne de référence conserve également sa provenance diagnostique.

Cette nouvelle hypothèse a été choisie après observation du cas et des tests
précédents. Le calcul chronologique interdit les labels futurs dans chaque
fit, mais ne transforme pas l'étude en validation prospective indépendante.
Aucun résultat de cette expérience ne déclenche de promotion automatique en
production. Une décision ultérieure doit reposer sur la robustesse des gains
et une période nouvelle, sans réajuster ce protocole sur ses résultats.
