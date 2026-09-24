# Note méthodologique — correcteur résiduel + Kalman gouverné

## Objectif

La couche Kalman intervient **après** le correcteur résiduel et **avant** un éventuel blend MKOnline :

```text
Chronos-2 → correcteur résiduel → Kalman causal → prévision autonome
                                                    ↓
                                      blend MKOnline éventuel
```

Elle ne remplace ni Chronos-2 ni le correcteur résiduel. Elle suit lentement les biais encore présents dans leur prévision finale et s'efface automatiquement quand son historique récent ne montre pas de gain.

## Variables retenues et extension fondamentale

La configuration opérationnelle par défaut utilise uniquement des informations
disponibles au moment de la prévision :

- niveau P50 du correcteur résiduel ;
- largeur P10–P90, comme mesure d'incertitude du modèle amont ;
- amplitude de la correction résiduelle ;
- heure cyclique (sinus et cosinus) ;
- prévisions de charge résiduelle de FR, DE, BE, NL et ES ;
- moyenne et dispersion de ces cinq fondamentaux.

Storm, MKOnline, les observations futures et toutes les colonnes `*_oracle` sont exclus par une liste blanche explicite. Un essai avec davantage de variables calendaires (jour de semaine, saison et week-end) n'a apporté qu'un gain marginal en France et a dégradé DE, BE, NL et ES ; ces variables n'ont donc pas été conservées.

Le moteur accepte désormais trois familles fondamentales supplémentaires,
configurables sans modifier le code métier :

- `linear_weather` : température prévue, vent météorologique et rayonnement ;
- `linear_renewables` : production éolienne et solaire prévue, ainsi que leurs
  rampes intrajournalières ;
- `linear_fundamental` : combinaison contrôlée de météo, renouvelables,
  charge, disponibilité nucléaire et charges résiduelles.

Chaque famille reçoit exclusivement les colonnes de son groupe. Les variables
sont normalisées séparément sur le préfixe d'entraînement ; aucune moyenne ou
dispersion n'est calculée entre des unités incompatibles. Les transformations
autorisées sont déterministes : moyenne ou étendue de séries de même unité,
différence explicite, rampes intrajournalières et degrés-jours de chauffage ou
de refroidissement. Une rampe est remise à zéro au début de chaque journée
locale physique, y compris lors des journées DST de 23 ou 25 heures.

Les nouvelles familles sont volontairement **opt-in**. Une variable activée
doit être une prévision historisée au cutoff day-ahead, couvrir le train, la
validation, les 365 jours de test et la journée future selon les seuils du
contrat. Il n'y a ni interpolation, ni backfill, ni remplacement silencieux
par zéro. Le run échoue avec la liste des colonnes incomplètes si ce contrat
n'est pas satisfait. Cela évite qu'une météo observée ou une série révisée a
posteriori améliore artificiellement le backtest.

## Famille de filtres et gouvernance

Les bras suivants peuvent être comparés causalement chaque jour :

1. identité — aucune correction ;
2. KF de biais lent ;
3. KF biais + profil horaire harmonique ;
4. KF linéaire de marché utilisant les variables ci-dessus ;
5. KF linéaire biais + réévaluation bornée de la correction résiduelle ;
6. UKF additif biais + échelle logistique bornée dans `[0,5 ; 1,5]`.

Les trois bras fondamentaux décrits ci-dessus s'ajoutent à cette liste quand
leurs groupes sont activés. Leur état contient un intercept, deux composantes
horaires, trois descripteurs du modèle amont (niveau, largeur P10–P90 et shift
résiduel), puis les variables standardisées de leur groupe. La dimension est
limitée à 64 features par candidat.

Un EKF manuel a également été implémenté et testé. Il n'est pas activé dans le challenger par défaut : l'observation biais + échelle admet déjà une formulation linéaire exacte, `pykalman` ne fournit pas d'EKF natif et le Jacobien logistique devient peu informatif près des bornes. L'UKF reste, lui, présent dans la compétition pour vérifier empiriquement qu'une non-linéarité bornée apporte ou non une valeur supplémentaire.

La gouvernance mesure la MAE de chaque bras sur les 60 derniers jours disponibles, avec au moins 14 jours d'historique. Elle choisit aussi une amplitude entre 0 et 1 par pas de 0,05. Une correction n'est appliquée que si le gain passé dépasse à la fois 0,05 EUR/MWh et 0,5 % de la MAE de référence. Le bras identité protège donc automatiquement contre une correction dégradante.

Le ratio bruit de processus / bruit d'observation du KF est fixé à `0,001`. Les innovations sont tronquées à trois écarts-types et le déplacement final à ±20 EUR/MWh afin de limiter l'effet des pointes de prix.

## Causalité et probabilités

Pour une journée de livraison D, l'état est figé avant de produire les 23, 24 ou 25 heures de D. Les valeurs de D ne sont assimilées qu'après cette prévision ; elles ne peuvent modifier que D+1 et les journées suivantes. La prévision future n'assimile aucune observation.

Le même déplacement additif est appliqué à P10, P50 et P90. Cette règle préserve exactement l'ordre des quantiles ainsi que la largeur de l'intervalle. Aucun `smooth()`, aucun EM et aucune interpolation DST ne sont utilisés ; l'inférence en ligne repose sur `filter_update`.

## Protocole rolling 365 jours

- warm-up causal : 12/08/2025 au 27/08/2025, soit 16 jours ;
- évaluation : 28/08/2025 au 27/08/2026 ;
- support : 365 jours civils complets, exactement 8 760 heures physiques par zone ;
- comparaison : mêmes observations et mêmes timestamps pour le correcteur résiduel et Kalman ;
- Storm : benchmark d'évaluation seulement, sur son intersection finie.

Les 381 jours sont conservés dans l'artefact Statistics pour le warm-up et l'audit Storm, mais les cartes et métriques du rapport sont limitées de façon stricte aux 365 jours audités.

## Résultats

| Zone | MAE résiduelle | MAE Kalman | Gain MAE | RMSE résiduelle | RMSE Kalman | Biais résiduel | Biais Kalman |
|---|---:|---:|---:|---:|---:|---:|---:|
| FR | 12,604 | 12,407 | 1,56 % | 20,205 | 19,839 | -1,143 | -0,518 |
| DE | 11,861 | 11,643 | 1,84 % | 23,149 | 22,755 | -1,321 | -0,980 |
| BE | 12,449 | 11,950 | 4,01 % | 26,504 | 25,821 | -3,264 | -1,955 |
| NL | 11,789 | 11,475 | 2,66 % | 23,830 | 23,381 | -2,705 | -1,719 |
| ES | 10,130 | 10,035 | 0,93 % | 14,543 | 14,450 | -0,239 | -0,193 |

Le KF de marché est le bras le plus fréquemment retenu. L'UKF n'est sélectionné sur aucune zone dans ce replay, ce qui confirme que la non-linéarité supplémentaire n'apporte pas de gain démontré ici.

La couverture P10–P90 progresse également dans les cinq zones — FR 74,46 % → 75,02 %, DE 74,25 % → 74,73 %, BE 73,54 % → 74,74 %, NL 76,27 % → 76,94 % et ES 75,08 % → 75,63 % — alors que la largeur moyenne des intervalles reste strictement inchangée par construction.

## Limite et statut

Le replay est causal, mais il ne constitue pas un holdout de promotion totalement indépendant : seuls 16 jours de prévision **déjà corrigée résiduellement** existent avant la fenêtre finale de 365 jours. Les choix d'architecture ont été examinés sur cette même période. Le résultat doit donc rester un challenger expérimental et être confirmé en forward-test, ou après reconstruction d'environ 730 jours de prévisions résiduelles réellement OOF, avant remplacement silencieux du modèle autonome de production.

Les archives live sources ne sont pas modifiées. Chaque bundle expérimental contient les prévisions amont et Kalman, les états, innovations, décisions quotidiennes, métriques, manifestes et checksums nécessaires à une reproduction complète.

Au 28 août 2026, les fichiers locaux météo, vent, solaire, charge et nucléaire
audités ne couvrent pas encore simultanément le warm-up, FINAL365 et J+1. Les
résultats chiffrés ci-dessus restent donc ceux du contrat de marché par défaut ;
ils ne doivent pas être présentés comme les résultats de l'extension météo.
L'infrastructure est prête, mais l'activation fiable exige d'abord une
matérialisation PIT complète. Cette absence provoque désormais un refus
explicite, et non une dégradation silencieuse du modèle.

## Paramétrage et promotion

- `config/auxiliary_lab.yaml` décrit les groupes, transformations, sources
  externes et grilles d'ablation utilisés par `FineTune.ps1` ;
- `config/kalman_operational.yaml` est le sidecar lu par `Forecast.ps1 -Mode
  All` ;
- les colonnes externes sont jointes uniquement dans la vue Kalman jetable :
  elles ne changent ni le modèle Chronos-2, ni son manifeste de features, ni
  ses archives live scellées ;
- les rapports et artefacts enregistrent le SHA-256 de la configuration, les
  sources, la couverture, les scalers et les features exactes de chaque bras.

La bonne séquence est : matérialiser les vintages PIT, lancer les ablations du
laboratoire, comparer validation puis test FINAL365, reporter la configuration
retenue dans le sidecar opérationnel, et seulement ensuite lancer `Forecast.ps1
-Mode All`.

## Exécution

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  'C:\Users\BQ6757\chronos2_v1\run_kalman_residual_experiment.py' `
  --delivery-day 2026-08-28 --zones FR DE BE NL ES --overwrite
```

Références d'implémentation : [pykalman](https://github.com/pykalman/pykalman/tree/main) et [documentation officielle](https://pykalman.readthedocs.io/en/latest/index.html).
