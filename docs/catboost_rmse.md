# CatBoost RMSE — expérience séparée

Cette expérience ne modifie ni `Forecast.ps1`, ni `-Mode Both/Complete`, ni le
correcteur MAE opérationnel. Elle ne lance pas Chronos, LoRA ou Kalman et
n'actualise aucune donnée réseau. Ce n'est pas l'ancien expert `NyxRMSE.ps1` :
ici, on remplace **la fonction de perte du correcteur CatBoost lui-même**.

## Protocole fixé avant les résultats

- FR, DE, BE, NL ; mêmes prévisions nucléaires et mêmes variables que la référence.
- Score sur le 20 septembre 2025 au 19 septembre 2026 inclus, 365 jours civils.
- Chaque prévision D est issue d'un réentraînement sur D−365 à D−1.
- 700 arbres, profondeur 6, learning rate 0,03, L2 15, seed 42, correction ±40 €/MWh.
- Seuls `loss_function` et `eval_metric` deviennent `RMSE`. Certains paramètres
  internes par défaut de CatBoost dépendent de l'objectif ; ils ne sont pas
  présentés comme identiques à ceux de MAE.
- Les observations du 19 septembre sont ajoutées à l'évaluation **après** la
  prédiction, jamais à son apprentissage. Storm est exclusivement un comparateur.
- Un réentraînement MAE de contrôle doit reproduire la première journée archivée
  à 0,0001 €/MWh près, avec le même schéma de variables, avant le test annuel.
- Les corrections brutes et plafonnées sont sauvegardées pour chaque heure.

L'année a déjà été examinée : c'est un diagnostic historique, pas une validation
indépendante. Les sources historiques sont les mêmes reconstructions figées que
la référence ; leur utilisation ne crée pas une preuve supplémentaire de
publication en temps réel à 08 h. Pas de choix automatique de règle par pays,
pas d'optimisation de plafond sur cette année, pas de promotion.

## Commandes PowerShell

Depuis n'importe quel dossier :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\CatBoostRMSE.ps1' -Action Run
```

La même commande reprend les journées terminées et vérifiées après une
interruption. Deux threads, un pays à la fois, priorité Windows inférieure à la
normale. Le programme n'arrête aucun autre calcul. Ne pas changer `-Threads`
pendant une expérience déjà commencée : ce réglage fait partie de son identité.

```powershell
& 'C:\Users\BQ6757\chronos2_v1\CatBoostRMSE.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\CatBoostRMSE.ps1' -Action Report
```

`Prepare` fige les données sans entraîner. `Smoke` vérifie la reproduction MAE
et calcule une journée RMSE par pays. `Run -MaxDays 7` limite le calcul aux sept
premiers jours chronologiques ; le rapport reste explicitement partiel. Ces
journées sont réutilisables ensuite par `Run` sans limite. `-DryRun` affiche la
commande sans rien lancer.

## Résultats

Répertoire : `runs/experiments/catboost_rmse_v1/2026-09-19`.

- `reports/catboost_rmse_report.html` : comparaison MAE/RMSE/Storm, populations
  appariées, sous-périodes et extrêmes. Mise à jour après la première journée,
  tous les 30 jours calculés, puis à la fin de chaque pays.
- `<pays>/inputs` : données préparées immuables, recettes, provenance et empreintes.
- `<pays>/checkpoints/<empreinte>/<jour>` : quantiles déplacés, correction brute,
  correction appliquée, dates d'entraînement et schéma de variables.
- `<pays>/status.json` : état et dernière journée traitée.

Une empreinte invalide ou un changement de recette bloque la reprise ; le
programme ne contourne pas le contrôle et ne détruit pas le checkpoint.

## Interprétation

Le modèle RMSE vise une moyenne conditionnelle, pas une médiane calibrée.
L'emplacement historique `q50` représente ici le **centre RMSE expérimental**.
Les bornes q10/q90 sont décalées de la même correction mais ne sont pas
recalibrées : leur couverture probabiliste n'est pas garantie. Le rapport garde
la MAE en plus de la RMSE et distingue les populations Storm incomplètes.

Le Kalman n'est pas simulé avec son ancienne correction figée. Si le correcteur
RMSE est intéressant, la prochaine expérience sera un véritable replay Kalman
sur ses nouvelles sorties, suivi d'une validation sur des journées inédites.

Référence sur les objectifs : [documentation CatBoost](https://catboost.ai/docs/en/concepts/loss-functions-regression).
