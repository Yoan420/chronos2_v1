# Comparaison charge résiduelle Saturn vs Chronos-2

Deux usages coexistent dans le dépôt. Ils répondent à des questions différentes
et leurs résultats ne doivent pas être mélangés :

- `ResidualCompare` est le benchmark historique complet. Les deux branches sont
  recalculées de bout en bout et peuvent être comparées sur `FINAL365`.
- `Run -ResidualLoadSource Chronos2` est le challenger quotidien hybride. Il est
  utile pour une campagne prospective, mais son historique Saturn ne mesure pas
  la performance historique du challenger.

## Benchmark historique complet

### Question mesurée

Le contrôle consomme les forecasts historiques `residual_load` de Saturn. Le
challenger les remplace par un replay Chronos-2 construit à partir des cinq
séries observées ENTSO-E :

```text
power.<pays>.residual.load.entsoe.hourly.gw.obs
```

Pour chaque origine, les observations sont lues causalement au cutoff D-1
08:00 heure locale. Le modèle `amazon/chronos-2` est épinglé à la révision
`29ec3766d36d6f73f0696f85560a422f50e8498c`, avec un contexte de 2048 heures.

Le protocole impose une comparaison symétrique :

- même cible prix scellée, mêmes timestamps, mêmes folds et mêmes seeds ;
- mêmes features et mêmes modèles aval ;
- même masque historique de disponibilité Saturn, y compris ses heures
  absentes ;
- seules les cinq colonnes `*_residual_load_fcst` et leurs cinq miroirs
  `known_*_oracle` peuvent différer ;
- recalcul des OOF Chronos-2 prix, des modèles aval, du correcteur résiduel et de
  l'ensemble pour les deux branches ;
- pour FR et NL, rejeu symétrique du primaire et des poids MKOnline gelés, sans
  les réentraîner ;
- aucun `statistics_history` ni aucune métrique historique du contrôle Saturn
  n'est réutilisé.

Les contrôles de parité et les checksums interrompent le benchmark si une autre
feature, la cible, un fold ou un masque diverge entre les branches.

### Fenêtres gelées

| Fenêtre | Dates de livraison locales incluses | Rôle |
|---|---|---|
| `EXT223` | 2024-01-02 au 2024-08-11 | Ajustement étendu du correcteur |
| `OOF730` | 2024-08-12 au 2026-08-11 | Backtest rolling-origin commun |
| `FINAL365` | 2025-08-12 au 2026-08-11 | Score affiché pour la comparaison |

Le replay amont calcule aussi le 2026-08-12 pour produire la journée de forecast
du rapport. Cette journée n'entre pas dans le score `FINAL365`.

### Lancement

Le lancement opérationnel autonome/Kalman avec la source Saturn reste distinct
du benchmark historique des sources :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Both
```

Afficher et vérifier le protocole sans calculer :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action ResidualCompare -ResidualComparisonStage Plan -Countries FR,DE,BE,NL,ES
```

Lancer le benchmark complet :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action ResidualCompare -ResidualComparisonStage All -Countries FR,DE,BE,NL,ES -Device cuda
```

Ajouter `-AllowModelDownload` au premier lancement si la révision épinglée de
Chronos-2 n'est pas déjà présente dans le cache local.

Le benchmark peut également être exécuté étape par étape, dans cet ordre :

1. `SyncObserved` synchronise les vintages des cinq séries `.obs`.
2. `ResidualReplay` produit les forecasts historiques de charge résiduelle.
3. `PriceReplay` recalcule les OOF prix Saturn et Chronos-2.
4. `Downstream` recalcule les modèles aval et le correcteur pour les deux
   branches.
5. `Blend` rejoue les blends gelés de FR et NL.
6. `Report` vérifie la paire, recalcule les métriques et publie les rapports.

Par exemple, pour reprendre à partir du replay prix :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action ResidualCompare -ResidualComparisonStage PriceReplay -Countries FR,DE,BE,NL,ES -Device cuda
```

La reprise des checkpoints est activée par défaut. Relancer la même étape après
une interruption reprend les blocs déjà validés. Les options utiles sont :

- `-SkipObservedSync` avec `All` lorsque les stores observés complets existent
  déjà ;
- `-NoResume` pour désactiver la reprise des replays ;
- `-OverwriteComparison` pour republier des sorties aval ou des rapports déjà
  présents après vérification explicite ;
- `-DryRun` pour afficher la commande Python exacte sans l'exécuter.

Pour forcer un nouveau calcul intégral après une première exécution, combiner
`-NoResume` et `-OverwriteComparison`. Les artefacts précédents ne sont alors
remplacés qu'après publication atomique réussie :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action ResidualCompare -ResidualComparisonStage All -Countries FR,DE,BE,NL,ES -Device cuda -NoResume -OverwriteComparison
```

### Rapports et audit

Les artefacts détaillés et les deux runs recalculés sont conservés sous :

```text
runs/experiments/residual_load_chronos2_historical_v1/
```

Les rapports utilisateur sont publiés sous la même forme que les exports
standards :

```text
runs/exports/residual_load_chronos2/2026-08-12/<zone>/<variante>/
  forecast_<zone>_2026-08-12_<variante>.html
  forecast_<zone>_2026-08-12_<variante>.csv
```

Le HTML est produit par le renderer horaire standard. Il contient les panneaux
et métriques recalculés des deux branches sur `FINAL365`, y compris le résumé
apparié et les win rates de la section Statistics ; il ne reprend aucun fichier
Statistics Saturn. Chaque dossier utilisateur contient exactement le couple
HTML/CSV des exports standards. Les variantes `autonomous` sont produites pour
les cinq pays, et `blend` uniquement pour FR et NL. Les audits détaillés et le
résumé global restent sous
`runs/experiments/residual_load_chronos2_historical_v1/comparison`.

Le protocole scellé dispose d'une cible réalisée jusqu'au 2026-08-11 et d'un
horizon live exact pour le 2026-08-12. Il est donc volontairement exporté dans
le dossier `2026-08-12`. Les anciens rapports hybrides déjà présents sous
`runs/exports/residual_load_chronos2/2026-08-26` restent immuables et ne sont
pas des résultats du recalcul historique complet.

## Challenger quotidien hybride — usage prospectif uniquement

Le challenger quotidien conserve la chaîne de prix actuelle et remplace
uniquement les cinq valeurs futures `*_residual_load_fcst` de J+1 par des
prévisions Chronos-2 :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Autonomous -ResidualLoadSource Chronos2
```

`Run -Mode Both` et `Run -Mode All` incluent le Kalman standard et exigent
désormais la source Saturn ; cette expérience quotidienne utilise donc la
seule vue autonome.

L'archive Saturn publiée du même pays et du même jour est son contrôle
obligatoire. Son historique aligné et son contexte modèle sont réutilisés ;
seules les dix colonnes de traitement de J+1 sont remplacées. Pour FR et NL, le
primaire MKOnline est également repris depuis l'archive Saturn.

Les archives shadow restent isolées de la production :

```text
runs/live/<zone>/_challengers/residual_load_chronos2/
  <zone>_day_ahead_YYYY-MM-DD_residual_load_chronos2
```

Elles déclarent `production_eligible=false` et ne participent ni au rattrapage
Statistics ni au rolling-365 de production. Le forecast J+1 de leur export est
bien celui du challenger, mais les panneaux historiques viennent de l'archive
Saturn scellée. Ces rapports hybrides ne sont donc pas valables pour conclure à
une amélioration historique de Chronos-2.

### Comparaison prospective appariée

Après accumulation de jours disposant des deux archives immuables et des prix
réalisés complets, lancer par exemple :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' .\run_residual_load_source_comparison.py --zones FR DE BE NL ES --start 2026-08-27 --end 2026-09-30 --registry .\chronos2_hourly_live_zones.yaml --output-dir .\runs\experiments\residual_load_source_comparison_2026-08-27_2026-09-30
```

Ce comparateur prospectif ne score que les jours appariés avec 23/24/25 heures
réalisées. Il reste pertinent pour mesurer les performances futures après mise
en place du challenger, mais il ne remplace pas le benchmark historique
`ResidualCompare`.

Les paires du 2026-08-26 ne constituent pas un panel cinq pays comparable : FR
et NL ont consommé deux états différents du PIT NL avant le contrôle scellé.
Elles restent immuables ; la première campagne cinq pays fiable commence le
2026-08-27.
