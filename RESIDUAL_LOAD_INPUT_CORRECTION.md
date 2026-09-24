# Correcteur causal des inputs de charge résiduelle

## Objectif

Ce challenger corrige les cinq séries `*_residual_load_fcst` avant leur
utilisation par le modèle de prix. Pour chaque pays, la cible supervisée est :

```text
charge_residuelle_observee - charge_residuelle_prevue
```

La recette reprend l'architecture du correcteur de prix actuel : un
`CatBoostRegressor` et un `HistGradientBoostingRegressor`, combinés à poids
50/50. Les bornes de correction sont toutefois exprimées en GW et sont donc
configurées séparément du clip de 40 EUR/MWh du correcteur de prix.

## Contrat causal

- Les forecasts bruts sont les vintages PIT disponibles avant D-1 08:00
  Europe/Paris.
- Les observations et leurs révisions doivent également être disponibles au
  cutoff du fit.
- Pour prédire une journée D, le dernier label de charge résiduelle autorisé
  appartient à D-2. La journée D-1 n'est pas considérée complète à 08:00.
- Les features de mémoire d'erreur utilisent uniquement des lags d'au moins
  48 heures.
- Les journées civiles de 23, 24 ou 25 heures sont prédites en bloc et leur
  timeline UTC physique est conservée.
- Les corrections historiques sont préquentielles : aucun modèle entraîné sur
  une observation ne peut corriger cette même observation dans un backtest.
- Les heures de forecast absentes restent `NaN`, conformément au
  `fill_method: none` des recettes prix ; aucune valeur brute n'est inventée.
- Si une série observée n'a pas encore 48 couples causaux disponibles, son
  forecast reste brut avec une correction nulle. Ce cold start est audité par
  heure et le modèle s'active automatiquement dès que le seuil est atteint.

## Isolation de la production

Les recettes live actuelles sont scellées avec leurs inputs, leurs OOF et leurs
checksums. Remplacer seulement les valeurs J+1 créerait un décalage entre le
fit historique et le live. Le correcteur est donc d'abord matérialisé sous
`runs/experiments/residual_load_input_correction_v1` et ne modifie ni
`data/pit`, ni `runs/live`, ni les YAML de production.

La commande habituelle reste inchangée :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Both
```

## Exécution du challenger

Le plan est sans écriture et indique les sources manquantes :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' .\run_residual_load_input_correction.py --stage plan
```

Les vintages observées doivent d'abord être présentes localement. Si elles
manquent, le plan affiche cette commande :

```powershell
& '.\Forecast.ps1' -Action ResidualCompare -ResidualComparisonStage SyncObserved -Countries FR,DE,BE,NL,ES
```

Le build et le rapport s'exécutent ensuite avec :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' .\run_residual_load_input_correction.py --stage all --threads 4
```

La configuration par défaut effectue un refit quotidien, soit 365 refits sur
FINAL365. La mémoire d'erreur est ainsi remise à jour à chaque origine, au prix
d'un calcul sensiblement plus long. Les sorties corrigées, audits PIT,
checksums et métriques sont confinés sous
`runs/experiments/residual_load_input_correction_v1`. Le rapport compare MAE,
biais et MAE des rampes, globalement et sur les 10 % de rampes observées les
plus fortes. Il sépare également les périodes `fitted` des périodes
`cold_start_raw_passthrough`.

## Étapes de validation avant promotion

1. Générer les cinq trajectoires corrigées préquentielles.
2. Comparer MAE, biais et MAE des rampes des charges résiduelles, par pays et
   par régime.
3. Recalculer les OOF Chronos prix avec ces inputs corrigés.
4. Réentraîner le correcteur prix CatBoost/HistGradientBoosting sur les mêmes
   folds et les mêmes masques que le contrôle.
5. Recalculer les blends FR/NL et comparer la MAE prix sur FINAL365.
6. Seulement après validation, sceller de nouveaux bundles v2 et mettre à jour
   leurs manifests/checksums. Le flux `ResidualLoadSource=Chronos2` doit garder
   un correcteur distinct : ses erreurs ne sont pas interchangeables avec
   celles du provider Saturn.
