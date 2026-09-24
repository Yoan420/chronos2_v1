# NYX — sélection horaire interaction ±40 / Test 2

Essai isolé autorisé le 23 septembre 2026. Aucun changement de production,
données, configurations ou expériences précédentes. NYX désigne ici exactement
la référence interaction40 scellée, après Kalman, pas un nouveau modèle courant.
Test 2 est `regime_hour_local` de l'expérience ablation `397682c4250baa54`.
Ses trois prix finaux incluent déjà la référence : on ne les additionne pas.

## Sélecteur fixé avant calcul des résultats hybrides

Douze règles communes DE/NL, produit cartésien de trois conditions :

- P50 NYX au moins 150, 200 ou 250 EUR/MWh ;
- écart au maximum P50 NYX du même jour au plus 25 ou 50 EUR/MWh ;
- P90 NYX moins P50 NYX au moins 0 ou 50 EUR/MWh.

Chaque règle exige aussi `own_joint_deficit >= 0.5` OU
`own_residual_stress >= 1.0`, issus des prévisions physiques gelées.
Le maximum journalier est celui de la courbe NYX prévue du jour entier connu
à l'origine, jamais celui des observations, de Test 2 ou du lendemain.
Ni la probabilité de Test 2, ni sa correction ne conditionnent la bascule.
Les seuils constituent une petite famille exploratoire, pas des paramètres
établis ou ajustés pour obtenir une bascule particulière le 22 septembre.

À chaque origine hebdomadaire héritée, choisir sur les 90 jours civils
strictement antérieurs la règle minimisant la MAE DE/NL combinée, à condition :

- de gagner au moins 0,02 EUR/MWh de MAE combinée contre NYX ;
- de ne pas augmenter la MAE d'un pays de plus de 0,05 EUR/MWh ;
- de ne pas augmenter la RMSE de chacun des deux pays ;
- de sélectionner au moins 20 heures sur 5 dates distinctes au total,
  dont au moins 5 heures sur 2 dates distinctes par pays.

Les ex aequo suivent l'ordre fixe de la grille. Sans candidat admissible,
conserver NYX partout. Toutes les règles et leurs rejets sont audités ; ces
gardes passées ne garantissent pas l'absence de dégradation future. Des données
incomplètes/incohérentes font échouer la validation, sans imputation silencieuse.

La règle est ensuite figée pour le bloc suivant. Elle sélectionne ensemble
P10/P50/P90 de NYX ou de Test 2, bit à bit, sans moyenne, prime supplémentaire,
plafond ou hausse forcée. Test 2 peut donc aussi abaisser une prévision.
Ce choix conserve l'ordre des quantiles, pas nécessairement leur calibration.

## Périodes et limites

Les prévisions hors-échantillon Test 2 commencent le 22 mars 2026. Le premier
réentraînement hebdomadaire avec 90 jours antérieurs est le 21 juin 2026 :
évaluation appariée du 21 juin au 21 septembre, 93 jours / 2 232 heures par pays.
14 blocs historiques, puis une origine distincte le 22 septembre (24 h/pays).
Les comparateurs NYX et Test 2 partout sont recalculés sur ces mêmes heures.
Les 90 jours de réglage incluent les changements d'heure selon Europe/Berlin.

Les erreurs passées servent au choix de la règle ; jamais les prix du bloc
évalué. Les labels du 22 septembre sont réservés au rapport, après scellement
de toutes les décisions. Les périodes ayant déjà été examinées et les archives
PIT n'étant pas certifiées, cet essai est rétrospectif exploratoire, pas une
validation indépendante/prospective, et n'autorise aucune promotion automatique.

Rapporter MAE, RMSE, biais, pinball, couverture ; mois ; heures sélectionnées
et non sélectionnées ; prix >=200/300 ; faible renouvelable sans pic ; bascules
utiles/défavorables ; faux positifs et pics manqués, pour les trois méthodes.
Ne pas interpréter l'alerte binaire comme une probabilité calibrée de spike.

## Exécution et reprise

Aucun nouvel entraînement de NYX, Chronos, Kalman ou Test 2. Seulement un petit
sélecteur sur les archives figées. Python pricefm311, -B -u, un processus et
un thread, priorité BelowNormal. Attendre avant chaque bloc si mémoire libre
<3,5 GiB ; réserve cible 3 GiB, non garantie par l'OS. Ne tuer aucun processus.
Validation en lecture seule, empreintes sources/code/dépendances, identité
explicite obligatoire pour lancer. Verrou système, reçus SHA par bloc, refus des
checkpoints partiels non scellés. Reprise de la même identité seulement.
Sorties uniquement dans runs/experiments/solar_wind_scarcity_hybrid_v1/2026-09-22.
Une fin exige 15 blocs, grilles exactes, sélection bit à bit et 52 fichiers
scellés (53 avec completion.json), rapport et index présents.
