# NYX : solaire régional, rampes et risque de prix extrêmes

Note de recherche isolée, 19 septembre 2026. Aucune de ces références ne démontre
« rampes solaires CWE → spikes NYX » ni un gain opérationnel à 08h J−1. Les résultats
publiés ne sont pas transposés quantitativement à cette expérience.

## Six références primaires vérifiées

1. **Marc Gürtler et Thomas Paulsen (2018)**, *The effect of wind and solar power
   forecasts on day-ahead and intraday electricity prices in Germany*, Energy
   Economics 75, 150–162. DOI : [10.1016/j.eneco.2018.07.006](https://doi.org/10.1016/j.eneco.2018.07.006).
   [Article éditeur](https://www.sciencedirect.com/science/article/pii/S0140988318302512).
   Prévisions renouvelables, demande résiduelle, technologie marginale et coûts de
   ramping sont étudiés ensemble sur des prix horaires allemands de 2010 à 2016.
   Les besoins des heures suivantes entrent dans le mécanisme étudié. Cela motive
   des rampes signées et des interactions, sans établir leur causalité dans NYX.
   Les effets intraday et erreurs de prévision réalisées ne constituent pas des
   variables disponibles à 08h J−1.

2. **Tuomas Rintamäki, Afzal S. Siddiqui et Ahti Salo (2017)**, *Does renewable
   energy generation decrease the volatility of electricity prices? An analysis
   of Denmark and Germany*, Energy Economics 62, 270–282. DOI :
   [10.1016/j.eneco.2016.12.019](https://doi.org/10.1016/j.eneco.2016.12.019).
   [Dépôt institutionnel des auteurs](https://research.aalto.fi/en/publications/does-renewable-energy-generation-decrease-the-volatility-of-elect/).
   Contrepoint : le solaire réduit la volatilité quotidienne allemande dans leur
   échantillon, tandis que les renouvelables augmentent la volatilité hebdomadaire.
   Volatilité agrégée et spikes horaires sont des objets différents ; le signe
   dépend du marché, du profil de production et de l'horizon.

3. **Joanna Janczura et Rafał Weron (2010)**, *An empirical comparison of alternate
   regime-switching models for electricity spot prices*, Energy Economics 32,
   1059–1073. DOI : [10.1016/j.eneco.2010.05.008](https://doi.org/10.1016/j.eneco.2010.05.008).
   [PDF des auteurs](https://alfa.im.pwr.edu.pl/~hugo/publ/JJanczuraRWeron10_EE.pdf).
   La distinction entre régime normal, spikes positifs et chutes, avec transitions
   et saisonnalité variables, fournit un cadre descriptif pertinent. La qualité
   d'ajustement, principalement sur prix moyens journaliers, ne prouve pas un gain
   prédictif horaire par des variables solaires.

4. **Jesus Lago, Grzegorz Marcjasz, Bart De Schutter et Rafał Weron (2021)**,
   *Forecasting day-ahead electricity prices: A review of state-of-the-art
   algorithms, best practices and an open-access benchmark*, Applied Energy 293,
   116983. DOI : [10.1016/j.apenergy.2021.116983](https://doi.org/10.1016/j.apenergy.2021.116983).
   [PDF institutionnel](https://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/21_011.pdf).
   Référence pour des contrôles simples solides, une période chronologique longue,
   la séparation validation/test et les tests sur pertes appariées. Le qualificatif
   « day-ahead » ne certifie pas à lui seul la disponibilité des données à 08h.

5. **Tilmann Gneiting et Adrian E. Raftery (2007)**, *Strictly Proper Scoring Rules,
   Prediction, and Estimation*, Journal of the American Statistical Association
   102, 359–378. DOI : [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).
   [PDF des auteurs](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
   Fondement des scores probabilistes propres : Brier, log-loss et scores de
   distributions. Motive l'évaluation de la calibration en plus du classement des
   risques. Il s'agit d'un cadre statistique général, pas d'un résultat énergétique.

6. **Dimitris N. Politis et Joseph P. Romano (1994)**, *The Stationary Bootstrap*,
   Journal of the American Statistical Association 89, 1303–1313. DOI :
   [10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870).
   [Article éditeur](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870).
   Justifie la famille des rééchantillonnages par blocs face à la dépendance
   temporelle. Le bootstrap de cette expérience est un moving-block non circulaire
   à longueur fixe, pas l'algorithme stationnaire à longueurs géométriques du papier.
   Aucun bootstrap ne corrige automatiquement une rupture ou la recherche de modèles.

## Hypothèse testable et limites physiques

Une baisse solaire peut accroître la demande résiduelle, mais sa tension dépend
aussi de la charge, du vent, des disponibilités, du stockage, de la flexibilité et
des imports. Une rampe synchronisée entre pays peut réduire la compensation
géographique. Inversement, une congestion peut rendre un agrégat CWE trompeur.
Ce sont des hypothèses mécanistiques à tester, pas des résultats établis ici.

La comparaison doit distinguer niveau solaire, rampe solaire, niveau et rampe de
demande résiduelle. La rampe de la somme régionale n'est pas la somme des rampes
absolues : des variations opposées se compensent. Le régional hors zone cible
évite de présenter deux fois le même solaire. Si le contrôle contient déjà
`charge − vent − solaire`, l'ajout du solaire mesure une représentation différente,
et non son introduction initiale.

## Protocole de recherche

1. Geler l'information à 08h J−1 : éditions historiques publiées avant ce cutoff,
   calendrier et fuseaux explicites. Sans preuve d'édition disponible, le résultat
   reste rétrospectif non certifié point-in-time. Aucun réalisé solaire, aucune
   révision ultérieure ni erreur de prévision future ne doit servir de feature.
2. Comparer à budget de réglage constant : baseline NYX ; calendrier et fondamentaux
   de contrôle ; solaire local ; solaire régional ; rampes ; interactions limitées
   avec demande résiduelle, disponibilité et contraintes d'échange.
3. Évaluer séparément le risque de spike positif (seuil métier fixé avant test),
   les extrêmes statistiques (q99 calculé sur l'apprentissage seulement) et la hausse
   horaire du prix. Les prix négatifs constituent une autre cible, pas le même régime.
4. Calibrer les probabilités sur des prédictions chronologiques hors échantillon.
   Mesurer Brier, log-loss, PR-AUC, fiabilité, rappel et fausses alertes ; publier
   effectifs et couverture. Le risque peut informer une alerte sans modifier P50.
5. Une correction P50 éventuelle vise la médiane conditionnelle du résidu, sur tous
   les cas hors échantillon, pas uniquement les spikes réalisés. `probabilité ×
   amplitude` n'est pas en général une correction de médiane. Amplitude, activation
   et règle de repli sont gelées avant l'évaluation suivante.
6. Comparer les pertes sur les mêmes cibles. Rééchantillonner des blocs de jours
   consécutifs, toutes heures et toutes zones d'un même jour restant ensemble.
   Le découpage de 365 jours déjà examinés n'engendre pas un test vierge : la phase
   finale est explicitement `final_diagnostic`. Une confirmation prospective gelée
   reste nécessaire avant toute promotion.

## Conventions de l'évaluateur

`evaluate(predictions, config)` est sans écriture et ne certifie pas les vintages.
Les scores natifs sont descriptifs. `common_oos` exige l'intersection des
probabilités finies des cinq modèles de risque et les prévisions ponctuelles
finies de tous les variants, hors warmup et livraison `live_historical`. Cette
dernière est également exclue de `all` et reçoit sa propre phase. Les scores
ponctuels et intervalles portent sur `candidate_forecast/candidate_q10/candidate_q90` ;
les champs `baseline_*` conservent la référence non corrigée. Les clés diffèrent : l'évaluation refuse
la comparaison. Les actuals manquants ne deviennent jamais des labels négatifs.

Le biais vaut prévision moins réalisé ; la sous-estimation vaut
`max(réalisé − prévision, 0)`. La PR-AUC est l'average precision en escalier, avec
ex aequo groupés. Les probabilités nulles et égales à un sont bornées à 1e−12 pour
le calcul numérique du log-loss. La couverture nominale de `[q10,q90]` est 80 % ;
des bornes croisées sont comptées et exclues de cette couverture.

Un épisode réunit les heures positives consécutives en UTC physique, par zone ;
un trou ou un réalisé manquant le coupe. Le rappel exact nécessite une alerte
dans l'épisode. Le rappel tolérant accepte en plus l'heure immédiatement avant le
début, jamais une heure après. Le timing est l'heure de livraison alertée moins
l'heure de début, pas le délai entre émission day-ahead et livraison.

Les IC bootstrap à 95 % portent sur les différences de MAE et MAE des spikes,
candidat moins baseline, avec mêmes tirages de jours pour chaque zone et variant.
Ils conditionnent sur le modèle retenu et cet historique. Une seule journée
d'événements ne donne pas d'IC de MAE-spike exploitable. L'estimation native d'un
gain sur une couverture plus faible ne permet aucune sélection.

La sélection descriptive du meilleur candidat lit uniquement `selection`, pas
`final_diagnostic`. La décision rendue reste **conserver baseline ; promotion
interdite ; PIT production non certifié**, même devant un gain empirique. Un gain
probabiliste robuste sans amélioration MAE justifierait un signal de risque, pas
une correction P50. Un gain d'ablation est prédictif, pas une preuve causale.
