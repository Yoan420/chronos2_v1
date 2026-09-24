# NYX — trois tests isolés, autorisés le 23 septembre 2026

Sources : interaction40 DE/NL et expérience de régime de rareté
`696eae68b4a16a10`, scellés et conservés sans modification. Aucune source nouvelle,
aucun recalcul Chronos/Kalman, aucun changement de production ou des anciennes
expériences. Le 22 septembre est un diagnostic déjà examiné, jamais un test
prospectif indépendant. Les réserves PIT historiques restent entières.

## Trois variantes, trois changements séparés

1. `direct_quantile` : un correcteur CatBoost MultiQuantile sur tous les résidus
   contre le P50 final interaction40, sans classificateur ni partition à +50.
   Les mêmes variables et paramètres que v1 sont employés, quantiles 10/50/90.
2. `regime_hour_local` : recette complète v1 inchangée, sauf retrait des QUATRE
   colonnes `own/other_daily_peak_residual_stress` et
   `own/other_daily_peak_deficit_stress`. Les rampes et anticipations intra-D
   restent présentes. C'est une ablation du signal horaire, pas une réparation
   de fuite. Les variables relatives au pic seraient un autre test ultérieur.
3. `regime_calibration_oof90` : seules les probabilités v1 archivées sont
   recalibrées sur les 90 jours civils immédiatement antérieurs. Régression
   logistique non pondérée, C=0.1, logit(p_v1) et indicateur NL : pente commune,
   ajustement pays régularisé. Les probabilités v1 sont parfois brutes et parfois
   déjà Platt selon le support des anciens blocs ; la variante porte sur le
   recalibrage de ce pipeline tel qu'il était, pas sur un unique logit brut.
   Repli identité si moins de 20 événements >50 poolés, moins de 5 par pays,
   moins de 32 non-événements ou pente ajustée non positive. Aucun risque inversé
   silencieusement. Les experts conditionnels sont reconstruits exactement ;
   leur mélange avec les probabilités v1 DOIT reproduire les quantiles scellés
   (tolérance absolue1e-7, relative1e-8) avant de changer les probabilités.

Toutes les variantes gardent 120 arbres, profondeur4, taux0.05, L2=3, seed20260923,
un thread. Les distributions sont apprises sur les seuls résidus antérieurs à
chaque origine, au maximum365jours ; pas de plafond ±40 ni hausse forcée. Les
trois quantiles résiduels sont ajoutés chacun au même P50 interaction40. Ordre des
quantiles vérifié/remis en ordre ; cela ne garantit pas leur calibration.

## Comparaison équitable

Les premières probabilités v1 hors-échantillon datent du 21 décembre2025.
Le premier bloc hebdomadaire v1 disposant de90jours est le22mars2026. Donc tous
les résultats sont comparés sur **22mars–21septembre2026 :184jours,4415heures/pays**,
27origines historiques. Une28eorigine produit les24heures du22septembre séparément.
Trois tests ×28origines =84lots sauvegardés. Références : interaction40 ET régimesv1,
réévalués sur exactement ces mêmes heures ; aucun mélange de fenêtres différentes.

Le recalibrage ne reçoit que des probabilités effectivement prédites hors des
échantillons d'apprentissage correspondants, avec contrôle pays/horodatage/origine.
Les observations du22septembre ne sont lues pour le rapport qu'après sauvegarde de
toutes les prévisions. Aucun réglage des paramètres à partir des résultats.

Scores fixés : MAE,RMSE,biais,pinball10/50/90,couverture,fréquences sous quantiles,
spikes≥200/300 (détections,faussesalertes,manqués), heures renouvelablesfaibles sans
spike, mois, et Brier du régime quand la variante fournit une probabilité. Aucune
probabilité de classification n'est inventée pour le correcteur quantile direct.
Ces comparaisons sont exploratoires, non une sélection définitive de production.

## Exécution

Pythonpricefm311 `-B -u`, un processus, un thread, BelowNormal. Trois tests
séquencés par bloc pour préserver la prévision de production. Avant chaque fit,
mémoire disponible≥3.5GiB (réserve visée3GiB, pas une réservationOS), sinon attente.
Validation en lecture seule, empreintes sources/code/dépendances, identité imposée
au lancement. Verrou système, checkpoints SHA, refus d'une identité modifiée ou
d'un checkpoint partiel non scellé. Aucun autre processus tué ou modifié.

Sorties exclusivement sous `runs/experiments/solar_wind_scarcity_ablation_v1/2026-09-22`.
Chaque lot possède un reçu et un audit. Le rapport final contient les cinq méthodes
sur les deux pays ; sa complétion exige l'inventaire exact, les SHA et les grilles.
