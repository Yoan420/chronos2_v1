# Forecast day-ahead multi-pays

## NYX — Prévisions et résultats locaux

NYX lance la prévision complète avec `NuclearKalman.ps1` et présente les rapports
CWE et les résultats par pays publiés dans `runs`. L'interface française regroupe
ces résultats par livraison. Les calculs continuent après fermeture de la fenêtre.

Ouvrir **NYX** depuis son icône sur le Bureau ou dans le menu Démarrer.
Le serveur démarre automatiquement, sans fenêtre PowerShell.

Installation, lancement et périmètre des résultats :
[guide de la console](README_EXPERIMENT_CONSOLE.md).
L'application de prévision existante reste accessible avec `Forecast.ps1 -Action App`.

Ce dépôt contient la chaîne nécessaire aux forecasts actuels pour la France,
l’Allemagne, la Belgique, les Pays-Bas et l’Espagne.

## Production dédiée Kalman nucléaire

Calculer les prévisions `nuclear_kalman` pour BE, DE, FR et NL, publier leur
rapport HTML individuel et ouvrir le rapport groupé `CWE_Model_Storm` :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\NuclearKalman.ps1'
```

La date de livraison est fixée au lendemain à Paris pour tout le lancement,
à partir du cutoff de 08 h (heure de Paris).
Utiliser `-Countries FR,BE` pour choisir les pays, `-DeliveryDay 2026-09-11`
pour imposer une date et `-NoOpen` pour ne pas ouvrir le rapport final.

Ce lanceur synchronise les sources communes une fois, conserve les caches
quotidiens vérifiés et ne lance pas les modèles classiques. Chronos et le
correcteur résiduel restent les étapes nécessaires du modèle Kalman nucléaire.
Les explications contrefactuelles coûteuses sont désactivées par défaut ;
`-WithAttribution` les réactive. Aucun paramètre de prévision n'est modifié.

Les rapports individuels sont publiés dans
`runs/exports/YYYY-MM-DD/<pays>/nuclear_kalman/` et le rapport groupé dans
`runs/reports/model_storm/CWE_Model_Storm_YYYY-MM-DD.html`.
Les journaux et le statut du lancement restent dans
`runs/logs/nuclear_kalman/YYYY-MM-DD/`, avec un `latest_status.json`.
Chaque pays dispose également d'un `run_status.json` dans son dossier expérimental.

Un échec reste signalé même si les autres pays et le rapport groupé aboutissent.
Après correction de la cause, relancer la même commande reprend les résultats
déjà sauvegardés. Un verrou empêche deux calculs simultanés pour la même livraison
et le même pays ; les verrous abandonnés par un processus arrêté sont récupérés
uniquement après vérification de son identité. Les révisions réelles des données
entraînent le recalcul des journées concernées.

## Rapport unique Model / Storm

Assembler et ouvrir un seul HTML avec les résultats déjà disponibles :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\ModelStorm.ps1'
```

La livraison est celle du lendemain à Paris. Utiliser `-DeliveryDay 2026-09-10`
pour choisir une date, ou `-NoOpen` pour générer le fichier sans l’ouvrir.
Le rapport est enregistré sous
`runs/reports/model_storm/CWE_Model_Storm_YYYY-MM-DD.html` et fonctionne hors ligne.

Il reprend les cartes BE/DE/FR/NL, les graphiques horaires et leurs erreurs,
avec uniquement **Model** (Kalman nucléaire), **Storm** et le prix **Observed**.
Les onglets **GRAPH**, **TABLE** et **INFO** donnent accès aux courbes, aux
valeurs horaires et aux définitions. L’Allemagne reste affichée si son résultat
manque et sera intégrée automatiquement dès qu’un résultat valide sera disponible.

Ce lanceur ne calcule pas de prévision. Il lit les archives nucléaires et les
derniers snapshots locaux vérifiés des prix Storm et des observations pour la
livraison choisie. Il actualise uniquement la métrique de reporting VPS de
Saturn, sans modifier les sources des modèles. Une donnée absente reste indiquée
comme indisponible ; les moyennes quotidiennes exigent toutes les heures du jour.

Pour une génération entièrement hors ligne, ajouter `-SkipVpsSync` : la colonne
VPS est alors indisponible, sans réutilisation silencieuse d'un ancien cache.
Le HTML produit reste consultable hors ligne dans tous les cas.
`NuclearKalman.ps1 -SkipObservedSync` désactive également cette collecte VPS
pendant l'assemblage du rapport groupé.

Chaque génération inclut aussi **Rolling Models Performance** : tableaux BE/DE/FR/NL,
périodes 7/30/60/90 jours et **365 jours (1 an)**, fréquences **60-MIN** et **DAY**, tri par colonne et
meilleures valeurs en vert. La vue par défaut **DASHBOARD · CACHE ONLY** utilise
uniquement les heures prouvées du cache Storm `.da.cache`, sans complément natif.
Les prix horaires sont arrondis au centime avant les erreurs et les moyennes
journalières. MAE, biais et RMSE utilisent les paires finies. Le hit rate compte
les erreurs absolues ≤ 5 et conserve la fenêtre entière au dénominateur ; le R² utilise
la variance de toutes les observations. DAY calcule d'abord les moyennes
disponibles par jour civil. La fenêtre se termine à la date de livraison du
rapport. Une absence de Model ne change jamais les scores Storm ; le classement
est désactivé si leurs couvertures diffèrent. Une provenance cache insuffisante
rend cette vue indisponible, sans substitution silencieuse par la courbe native.

Ces conventions sont rapprochées des tableaux officiels 7/30/60/90 jours du
18 septembre 2026. Elles ne garantissent pas une identité future si le dashboard
change ses formules, son périmètre ou sa version des données. Le lanceur lit les
derniers snapshots locaux vérifiés des prix ; il ne lit pas le site du dashboard.
La fenêtre **365 jours** est une extension des mêmes règles, non proposée par le
dashboard officiel. Le P&L VPS Storm est collecté séparément depuis les séries
Saturn `power.vps.<pays>.euromwh.h.da.pnl.storm`, avec une extraction auditable
propre au rapport. Une collecte indisponible ne remplace pas ces données par un
ancien cache ou par la simulation locale. Cette métrique n'est jamais une entrée
des modèles ; les hypothèses de la simulation locale et la provenance Saturn
restent distinctes.

Dans cette vue, **Daily P&L (VPS)** affiche Storm publié et **≈ NYX VPS estimé**.
Les flux disponibles sont divisés par tous les jours calendaires de la fenêtre,
comme le dashboard. NYX est masqué si toutes les journées publiées pour Storm ne
sont pas intégralement calculables ; aucune comparaison sur des sous-ensembles
différents, aucun classement vert officiel/estimé. La reconstruction NYX utilise
1 MW / 4 MWh, rendement charge 0,85 et décharge 1, stock initial 2 MWh/final 1 MWh,
vente de 1 MWh à la première heure. Planning optimisé seulement sur la prévision,
puis valorisé aux observations avec arrondi au centime par heure. Première
prévision ≤ 0 : règle non validée, donc pas d'estimation. Les jours DST sont une
extension physique non réconciliée. **Le prélèvement de 1 MWh de stock initial
n'est pas autofinancé : ce n'est pas un profit net de trading.** Couverture,
extraction et limites sont indiquées dans chaque rapport.

La vue **INTERNAL · COMPLETED HISTORY** conserve le benchmark précédent :
complément natif audité, mêmes heures Model/Storm, DAY sur journées physiques
complètes et fenêtre terminée au dernier jour observé complet. Sa simulation
économique reste distincte des métriques officielles.

Dans la vue interne, **Daily P&L (sim.)** est un diagnostic d'arbitrage de stockage 1 MW / 4 MWh,
rendement aller-retour 85 %, stock initial/final nul, sans charge et décharge
simultanées. Le planning est choisi uniquement avec la prévision, puis valorisé
aux prix observés. Les frais et la dégradation sont exclus. Ce n'est ni un P&L
réalisé, ni l'EVA utilisant le prix de la veille, ni une reproduction certifiée
de la stratégie du dashboard externe. Le mode DAY garde cette simulation horaire.
La simulation locale ne relance aucun modèle et ne requiert aucune API. Seule
la collecte VPS utilise Saturn, sauf si `-SkipVpsSync` est demandé.

## Un seul point d’entrée

Ouvrir l’application :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action App
```

Lancer les cinq pays sans MKOnline :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Autonomous
```

Produire l'autonome et le Kalman standard pour chacun des cinq pays :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Both
```

Après promotion et activation des bundles LoRA, produire les deux chaînes
opérationnelles LoRA :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode All
```

Pour `-Action Run`, les modes sont :

- `Production` : recette officielle actuelle ;
- `Autonomous` : chaîne autonome sans MKOnline ; elle utilise
  Chronos-2 + LoRA + correcteur résiduel lorsqu'un bundle promu est
  explicitement activé pour ce mode, sinon la recette autonome incumbent ;
- `Blend` : blend MKOnline validé, uniquement pour FR et NL ;
- `Both` : exactement deux rapports par pays, `autonomous` puis `kalman` ;
  le Kalman standard utilise la même base autonome, incumbent tant que LoRA
  est inactif, ou LoRA promu et explicitement activé. Aucun blend MKOnline
  n'est exporté et aucune activation LoRA n'est effectuée automatiquement ;
- `All` : exactement deux rapports par pays : `autonomous` =
  Chronos-2 + LoRA + correcteur résiduel, puis `kalman` = la même sortie
  autonome suivie du Kalman standard.

`All` n'inclut ni le blend MKOnline, ni `kalman_weather`, ni
`kalman_hybrid`. Le blend reste disponible avec `Blend` pour FR et NL. Les deux
Kalman météo/hybride restent reproductibles dans le laboratoire
d'expérimentation et leurs anciens rapports sont conservés, mais ils ne sont
plus matérialisés par le batch opérationnel.

`Run -Mode Both` et `Run -Mode All` exigent `-ResidualLoadSource Saturn` afin
de conserver un historique Kalman causal homogène. Pour tester la charge
résiduelle Chronos-2, utiliser `-Mode Autonomous -ResidualLoadSource Chronos2`.
Ces règles ne changent pas le sens de `Both` dans les actions expérimentales
`RegimeChallenger` et `Topology`, qui conservent leurs vues autonome/blend.

Le raccord LoRA est volontairement **fail-closed**. `All` refuse de démarrer
si chaque pays demandé ne possède pas un bundle final réellement promu,
enregistré, épinglé par checksum et adossé à des captures PIT/shadow valides.
La configuration livrée est donc inactive tant que ces preuves n'existent pas :
aucun fallback silencieux vers l'ancien autonome n'est autorisé dans `All`.
Un bundle actif devenu invalide arrête également le preflight. Le contrat et
le chemin de promotion sont détaillés dans
[CHRONOS2_EXOGENOUS_PRODUCTION.md](CHRONOS2_EXOGENOUS_PRODUCTION.md).

La preuve LoRA suit trois étapes distinctes. `Backtest` compare d'abord les
sorties **brutes** de Chronos-2 et de Chronos-2 + LoRA sur le holdout fermé.
Avant ce premier backtest, `PrepareZones` copie physiquement le checkpoint
multi-pays terminé vers un artefact isolé par zone. Les preuves, correcteurs et
décisions FR/DE/BE/NL ne peuvent ainsi jamais s'écraser :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action PrepareZones -Zones FR,DE,BE,NL -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v2\artifact'
```

Les sorties sont créées sous
`runs/experiments/chronos2_exogenous_lora_poc_v2/zones/<ZONE>/artifact`.
L'action refuse un entraînement encore temporaire, un artefact déjà évalué et
une zone absente du panel ; le panel actuel ne contient donc pas ES.
`CalibrationPanel` puis `CalibrateResidual` produisent ensuite 365 jours de
prédictions OOF préquentielles et ajustent le correcteur résiduel sans utiliser
ce holdout. Enfin, `FinalBacktest` applique ce correcteur gelé au challenger et
le compare à l'incumbent `residual_corrected` sur exactement les mêmes 365
jours locaux et leur grille physique exacte (8 759 à 8 761 heures selon les
frontières DST). Le rapport, les métriques et la preuve
scellée sont écrits sous `<bundle>/final_pipeline/`.

Seule cette évaluation finale appariée peut faire passer
`production_pipeline_evidence` à `true`. Elle ne change jamais
`production_pit_evidence`, ne promeut pas le bundle et ne l'active pas. La
gouvernance doit recevoir
`final_pipeline/final_pipeline_predictions.csv.gz`, jamais le CSV brut de
`Backtest`.

Pour un futur cycle réellement prospectif, le dernier prix du holdout peut ne
pas être publié au moment où checkpoint et correcteur doivent être gelés.
Les actions `EpochEarliest`, `EpochPlan`, `EpochFreeze`, `ResolveHoldout` et
`EpochFinalizeCheck` implémentent ce pré-engagement en deux phases sans ouvrir
le holdout ni relâcher la gouvernance. L'opt-in reste désactivé par défaut et
exige des preuves PIT opérationnelles vraies. Le calendrier, le template
`config/chronos2_exogenous_lora_prospective.example.yaml` et les commandes sont
détaillés dans [CHRONOS2_EXOGENOUS_PRODUCTION.md](CHRONOS2_EXOGENOUS_PRODUCTION.md).

Les entraînements LoRA de rang 8 et de rang 16 restent deux candidats
distincts. Le rang 8 déjà produit est conservé comme référence ; le rang 16 en
cours n'est jamais promu automatiquement. Ils doivent être comparés sur la
même fenêtre et le même incumbent après correcteur résiduel, puis gouvernés
séparément. La recette exacte du rang 8 est figée dans
`config/chronos2_exogenous_lora_rank8_reference.yaml`; la configuration
principale `chronos2_exogenous_lora_poc.yaml` reste celle du rang 16.

L'action `Exogenous.ps1 -Action CompareCandidates` automatise cette comparaison
appariée et écrit un JSON, un rapport HTML et le détail journalier dans un
dossier de sortie explicite. Deux preuves finales corrigées sont le mode de
référence. Deux backtests LoRA bruts peuvent aussi être comparés lorsque les
preuves OOF ne sont pas encore disponibles, mais le résultat est alors marqué
recherche uniquement et inéligible à la promotion. Le comparateur n'enregistre,
ne promeut et n'active jamais un modèle.

Deux preuves finales ne deviennent sélectionnables que si leur preuve PIT est
vraie et si la chaîne LoRA brut → correcteur OOF → prédiction finale, ainsi que
l'incumbent `residual_corrected`, restent intégralement vérifiables par leurs
empreintes. Sinon le rapport conserve seulement une préférence de recherche et
interdit toute décision `select_rank*`.

Les refits Kalman en fenêtre glissante utilisent un cache persistant sous
`runs/cache/kalman_rolling`. Le premier lancement remplit ce cache ; à données
et configuration inchangées, le lancement quotidien suivant ne réentraîne que
la nouvelle journée. Toute modification des données, paramètres, dépendances
ou du code invalide automatiquement les entrées concernées. Cette mécanique
est commune aux runs opérationnels, backtests, expériences et au laboratoire
`FineTune/Retrain`.

Le Kalman standard exige 365 jours de calibration avant les 365 jours évalués.
`config/kalman_operational.yaml` raccorde, via `upstream_history_by_zone`, les
préfixes résiduels préquentiels existants aux Statistics récentes. Les fichiers
et audits sont épinglés par SHA ; les prévisions déjà émises restent prioritaires
et tout conflit sur le chevauchement bloque l'export. Ces préfixes ne sont pas
utilisés pour une chaîne LoRA, qui doit fournir son propre historique.
La couverture des 730 jours (y compris les heures des changements d'heure) est
vérifiée pour tous les pays avant le premier refit Kalman. Les archives live ne
sont pas modifiées : après un échec d'export, relancer la même commande avec la
même `-DeliveryDay` réutilise les forecasts publiés et reconstruit les rapports.

À chaque `Run -Mode All`, les rapports reconstruisent également la journée de
livraison dans `Statistics`. Si les 23/24/25 prix observés du jour sont tous
publiés, ils sont appliqués immédiatement et les métriques sont recalculées.
Si la courbe n'est pas encore complète, la journée et les forecasts restent
affichés mais le prix observé demeure vide ; aucune moyenne observée partielle
n'est calculée. Le prochain lancement remplace automatiquement cet emplacement
par la courbe officielle complète dès qu'elle est disponible.

Les archives officielles restent immuables dans `runs/live`. Les vues demandées
par le launcher sont écrites dans `runs/exports` avec un CSV et un rapport HTML
détaillé. Une demande `Blend` sur DE, BE ou ES est refusée avant tout calcul.

## Challenger de changement de régime (shadow)

Le challenger spécialisé dans les chocs haussiers des heures solaires se lance
séparément. Pour cette action, `Both` conserve les vues autonome/blend
historiques, sans ajouter de Kalman :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action RegimeChallenger -Countries FR,DE,BE,NL,ES -Mode Both
```

Par défaut, il relit les archives officielles déjà émises et n'écrit jamais dans
`runs/live`. Il apprend une probabilité et une prime de choc à partir des cinq
charges résiduelles disponibles au cutoff J-1 08:00. Les variations J/J-1,
rampes, profils solaires, écarts entre zones, prix de base et désaccords entre
marchés alimentent un gate causal. Les labels d'entraînement s'arrêtent à J-2 et
les statistiques sont produites par replay préquentiel expanding-window.

Les résultats sont exclusivement écrits sous
`runs/challengers/price_regime_shock_v1/<jour>` : synthèse HTML multi-pays,
rapport détaillé par pays, forecasts CSV, métriques globales/solaires/spikes,
calibration du gate, importance des variables, audit PIT et checksums. Le
manifeste impose `production_eligible=false`, `automatic_promotion=false` et
`writes_runs_live_by_challenger=false`.

Pour reconstruire un rapport sur des archives déjà émises, sans relancer les
forecasts officiels :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action RegimeChallenger -Countries FR,DE,BE,NL,ES -Mode Both -DeliveryDay 2026-08-25 -OverwriteChallenger
```

Pour une exécution en une commande qui lance d'abord le batch officiel (lequel
peut publier dans `runs/live`), ajouter explicitement `-RunForecastsFirst`.

## Comparer Saturn et Chronos-2 sur la charge résiduelle

Le benchmark historique complet est une action séparée : il laisse les runs de
production ci-dessus inchangés et recalcule symétriquement les branches Saturn
et Chronos-2 avec les mêmes cibles, folds, features et modèles aval.

Vérifier d'abord le protocole et le volume de calcul :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action ResidualCompare -ResidualComparisonStage Plan -Countries FR,DE,BE,NL,ES
```

Lancer ensuite toutes les étapes, avec reprise automatique des checkpoints :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action ResidualCompare -ResidualComparisonStage All -Countries FR,DE,BE,NL,ES -Device cuda
```

Le score de comparaison est recalculé sur `FINAL365` ; aucun historique
Statistics du run Saturn n'est recopié. Les rapports utilisent le même renderer
HTML et la même arborescence `date/pays/variante` que les exports standards,
sous `runs/exports/residual_load_chronos2`.

Le protocole scellé s'arrête au 2026-08-11 : ses nouveaux exports sont donc
rangés sous `2026-08-12`. Les anciens dossiers hybrides `2026-08-26` sont
conservés, mais ne doivent pas être utilisés comme résultat du benchmark
historique complet.

Le mode quotidien `-Action Run ... -ResidualLoadSource Chronos2` reste un test
prospectif hybride. Ses panneaux historiques viennent du run Saturn et ne
constituent donc pas une comparaison historique des deux sources.

Le protocole, le détail des fenêtres, les étapes et les règles de reprise sont
documentés dans
[CHRONOS2_RESIDUAL_LOAD_CHALLENGER.md](CHRONOS2_RESIDUAL_LOAD_CHALLENGER.md).

## Tester une nouvelle série d’entrée

Modifier uniquement [config/experiment.yaml](config/experiment.yaml) :

- `enabled` active ou désactive la série du test ;
- `zones` choisit les pays ;
- `series` contient l’identifiant Saturn ;
- `pit_file` désigne le Parquet historique point-in-time ;
- `columns` décrit ses quatre colonnes horodatées.

Auditer le fichier sans lancer de modèle :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Experiment -Countries FR,DE,BE,NL,ES -ExperimentConfig 'config\experiment.yaml'
```

Lancer ensuite les backtests isolés :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Experiment -Countries FR,DE,BE,NL,ES -ExperimentConfig 'config\experiment.yaml' -RunExperiment
```

Les expériences vont exclusivement dans `runs/experiments`. Elles ne peuvent
ni modifier `runs/live`, ni utiliser Storm ou MKOnline comme input, ni être
promues automatiquement. Les nouvelles séries doivent être numériques,
horaires, historisées par révision et disponibles au cutoff causal.

## Fine-tuner les modèles auxiliaires

Le laboratoire séparé entraîne et compare le correcteur résiduel, calibre le
blend MKOnline et recherche les paramètres KF/EKF/UKF sans appeler le pipeline
live :

```powershell
& '.\FineTune.ps1' -Action Validate -Config 'config\auxiliary_lab.yaml'
& '.\FineTune.ps1' -Action Train -Config 'config\auxiliary_lab.yaml'
```

Les hyperparamètres, données, métriques, horizons et partitions chronologiques
sont déclarés dans le YAML. Les modèles, recettes, leaderboards et rapports
HTML interactifs sont publiés uniquement sous `runs/experiments/auxiliary_lab`.
Les refits glissants réutilisent le même cache sûr : relancer `Retrain` avec le
même jeu de données ne recalcule pas les journées déjà identiques.
Le guide complet est disponible dans
[AUXILIARY_MODEL_LAB.md](AUXILIARY_MODEL_LAB.md).

## Autres actions utiles

Compléter les journées manquantes des Statistics :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Backfill -Countries FR,DE,BE,NL,ES
```

Vérifier une commande sans l’exécuter : ajouter `-DryRun`.

L’application peut aussi être ouverte par double-clic sur
`launch_forecast_app.cmd`.

## Règles d’intégrité

- Les pays sont exécutés séquentiellement.
- Les contrats et checksums sont contrôlés avant lancement.
- Les forecasts existants valides sont réutilisés ; aucune archive n’est
  écrasée.
- Storm est chargé seulement après gel du candidat, pour les Statistics.
- Les trous, données futures, interpolations et quantiles croisés sont refusés.
