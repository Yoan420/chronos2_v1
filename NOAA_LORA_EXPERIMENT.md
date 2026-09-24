# Candidat LoRA météo publique — expérience isolée v1

## Pause demandée par l'utilisateur — 08/09/2026 au matin

**Ne pas reprendre ni relancer NOAA sans nouvelle demande de l'utilisateur.**
L'utilisateur a demandé de libérer le calcul pour des travaux urgents. Le
suivi Codex `suite-lora-rang-16` est désormais **PAUSED**. Le processus de
calibration a été suspendu, pas terminé : premier fold achevé, deuxième fold
en cours. Aucun checkpoint, résultat terminé ou fichier de production supprimé.

Identités Windows revérifiées avant suspension avec `psutil` :

- Lanceur **49196**, création Unix **1788827885.3488054**.
- Worker **58144**, création Unix **1788827885.7384465**, enfant de 49196.
- Commande : `run_chronos2_exogenous_calibrate_residual.py`, configuration
  `chronos2_exogenous_noaa_lora_v1.yaml`, zone FR et cache `runs/noaa_oof_v1`.

Les deux états sont `stopped` (suspendus), avec **zéro seconde CPU consommée
sur le contrôle de trois secondes**. Le worker n'a aucun processus enfant de
calcul ; le conhost du lanceur et les processus Forecast de l'utilisateur
n'ont pas été suspendus. Les journaux restent ceux du lancement à 02:38.

Cette pause conserve l'état modèle/optimiseur **en mémoire**, pas dans un
nouveau checkpoint durable ; elle ne libère pas toutes les allocations RAM.
Ne pas terminer ces processus ni redémarrer Windows pour conserver exactement
l'étape courante. Après autorisation de reprise, revérifier PID, création et
commande, puis appeler `resume()` une seule fois pour le worker et le lanceur
s'ils sont toujours suspendus. Ne pas rappeler `suspend()` sur un processus
déjà suspendu. Si les processus ont disparu, vérifier les artefacts terminés
et le staging avant toute reprise sur disque, sans suppression automatique.
Réactiver le suivi uniquement sur demande de l'utilisateur.

Autorisé le 7 septembre 2026. Ce candidat ne remplace ni les rangs 8/16
existants, ni les configurations opérationnelles. Le schéma est nouveau :
température à 2 m (°C), vent à 100 m (m/s), rayonnement horaire (W/m²), sept
variables calendaires et contexte de prix canonique. Pas d'alias Saturn en GW,
pas de prix Storm en entrée, pas de réanalyse ou météo observée future.

Configuration : `config/chronos2_exogenous_noaa_lora_v1.yaml`.
Rang 16, alpha 32, dropout 0,05, 500 étapes, batch 64, learning rate 1e-5,
seed 42, même révision Chronos-2 que les références. Ces choix sont figés
avant les nouveaux résultats. Les colonnes qualité de la banque restent
auditées mais ne sont pas utilisées par le modèle.

## Périodes et méthode

| Segment | Livraisons | Usage |
|---|---|---|
| Amorçage | 04/09/2023–02/09/2024 | Historique antérieur aux premiers folds |
| Calibration OOF | 03/09/2024–02/09/2025 | 365 jours de prévisions hors échantillon |
| Comparaison finale | 03/09/2025–02/09/2026 | Les mêmes 365 jours que les références |
| Support météo | 08/06/2023–02/09/2026 | Segments ci-dessus et marge du contexte 2 048 h |

Le fit principal emploie les 365 jours antérieurs au holdout : 335 pour
l'entraînement et 30 pour la validation. Le correcteur emploie de vraies
prédictions OOF : 13 folds préquentiels de 30 jours (dernier de 5 jours),
avec fenêtres d'apprentissage de 365 jours. Les checkpoints des folds sont
partagés entre pays, pas réentraînés quatre fois. Chaque pays conserve ses
prédictions et son correcteur. Les jours DST gardent 23/25 heures en évaluation
et sont exclus du fit à horizon fixe 24 h, comme dans les références.

Le budget comprend donc 14 entraînements neuronaux à 500 étapes : un principal
et 13 folds partagés, pas 13 folds par pays. Les durées de collecte ne couvrent
pas ces entraînements ; l'ensemble peut prendre plusieurs jours. L'estimation
sera affinée sur le premier entraînement avec ce nouveau schéma d'entrées.

Le choix d'explorer ces entrées intervient après analyse des anciens résultats.
La comparaison est donc **diagnostic_only** : elle n'est pas une confirmation
sur un jeu de test jamais consulté. Les indicateurs de promotion restent faux.
Les caches de prix couvrent les 28 328 heures nécessaires sans trou, mais ne
contiennent pas les timestamps de publication de leurs versions historiques.
Leur identité canonique et le masquage des labels futurs ne remplacent pas
cette preuve. NOAA conserve séparément publication d'archive et capture locale.

## État de départ vérifié

- Rangs 8/16 et leurs rapports terminés : ne pas les recalculer.
- Quatre samples NOAA réels validés, dont deux DST ; trois sont dans la fenêtre.
- Dépendances ecCodes installées uniquement dans le répertoire d'expérience.
- Préflight : 1 183 journées météo, environ 106 Go de budget brut conservateur,
  et 20 GiB de marge disque obligatoire ; environ 217 Go libres au départ.
- Le pilote est limité à sept journées nouvelles avant reprise complète.
- Aucun entraînement NOAA terminé et aucun résultat de performance NOAA à ce stade.

Collecte complète lancée le **07/09/2026 à 10:22:48 Europe/Paris** après
réussite du pilote : 10 partitions déjà validées, dont sept nouvelles et trois
seeds. Processus de lancement Windows 37488 (le lanceur du venv peut avoir un
processus Python enfant ; vérifier l'arbre réel, pas ce numéro seul).
Journaux :
`runs/experiments/chronos2_exogenous_public_weather_v1/history/logs/backfill_20260907_102248.out.log`
et le fichier `.err.log` correspondant. Quatre workers HTTP ; décodage natif
toujours séquentiel. Estimation initiale du pilote à deux workers : environ
11 h pour la collecte ; premiers jours à quatre workers : environ 8 h.
Ces estimations ne comprennent ni l'entraînement ni la calibration OOF.
L'automatisation existante `suite-lora-rang-16` a été mise à jour pour ce
candidat et réactivée à une cadence de 30 minutes, sans doublon.

Vérification du 07/09/2026 à 10:35 Europe/Paris : collecte active,
43/1 183 partitions validées, environ 7 h restantes estimées à ce rythme.
Les 88 tests ciblés des sources, panels et backfill ont réussi, ainsi que
les 7 tests de l'évaluateur OOF de recherche (103,70 s). L'option explicite
de recalcul des références est présente dans la CLI et testée. La configuration
d'activation opérationnelle conserve son SHA initial ; aucun modèle n'a été activé.

### Incident et point de reprise du 07/09/2026 à 12 h

La collecte initiale s'est arrêtée le 07/09 à 11:31:51 Europe/Paris sur
`httpx.RemoteProtocolError: Server disconnected without sending a response`.
Les processus NOAA 37488/19452 sont terminés ; le journal consigne l'échec et
le verrou de backfill a été libéré normalement. Aucun fichier n'a été supprimé.
Le préflight de reprise, exécuté sans réseau à 12 h, a revérifié les SHA,
contrats et heures des **173 partitions** existantes : 1 010 journées restent
à télécharger, la première étant le 26/11/2023. Le disque dispose d'environ
202 Go libres pour un budget brut restant de 90,9 Go plus la marge de 20 GiB.

**Ne pas relancer NOAA tant que le batch Forecast de l'utilisateur travaille.**
Le batch `Run -Mode Both` FR/DE/BE/NL, livraison 08/09/2026, a été lancé à
10:59:30 (lanceur 21344, enfant 2524) ; un descendant calculait BE lors de
la vérification à 12 h. Vérifier l'arbre courant et ses commandes, pas seulement
ces PID historiques. Une fois ce batch réellement terminé, reprendre le même
historique et le même cache avec la commande ci-dessous, sous de nouveaux
noms de journaux. Ne pas relancer les 173 partitions validées.

Le wrapper de reprise dispose désormais de `--day-retries 2` (valeur par
défaut, borne 0 à 4) : seuls les `RemoteProtocolError` directs avant
publication sont repris, en réutilisant le cache brut, après une puis deux
secondes. Chaque tentative est journalisée. Toute trace de publication,
erreur de certificat, erreur HTTP ou anomalie de contrat reste bloquante.
Le code scientifique `auxiliary_lab/noaa_gfs.py` n'a pas changé ; son SHA
reste celui du contrat des partitions déjà téléchargées. Les 28 tests du
wrapper corrigé ont réussi. La collecte n'a pas été relancée à ce stade.

### Reprise confirmée le 07/09/2026 à 13:32

Le batch Forecast précédent et ses processus Python de projet n'étaient plus
actifs lors des vérifications avant relance. Cela ne constitue pas une
attestation de réussite de ses exports ; aucun résultat Forecast n'a été modifié.
La collecte NOAA a repris à **13:31:49 Europe/Paris**, en arrière-plan caché,
avec le lanceur **13276**, quatre workers HTTP et `--day-retries 2`.
Les journaux courants sont désormais :
`history/logs/backfill_resume_20260907_133149.out.log` et
`history/logs/backfill_resume_20260907_133149.err.log`, sous la même expérience
météo publique. Les anciens journaux sont conservés.

Le nouveau processus a revérifié les 173 partitions existantes puis publié
le 26/11/2023 à 13:32:25 : **174/1 183 journées** confirmées, sans erreur dans
le nouveau journal stderr. Utiliser ces nouveaux handles/journaux pour le
suivi, en revérifiant toujours les processus réels avant toute autre relance.

## Exécution et reprise

### Collecte terminée et préparation des panels — 07/09/2026, 20:35 Paris

La collecte complète a publié **1 183 journées / 28 392 heures physiques**,
du 08/06/2023 au 02/09/2026. Le processus de collecte est terminé et son
journal stderr est vide. `validate_aggregate` a revérifié, sans réseau, les
1 183 partitions, leurs SHA, l'égalité avec l'agrégat et les contrats ; sortie 0.
SHA de `history/aggregate/noaa_gfs_weather.parquet` :
`341ee7e428e9567ade1748ad8e4a18ad31c292d0fe9015d21a0bd7553c867e86`.
Ne pas relancer le backfill terminé.

Après vérification de l'absence d'autre calcul de projet, la construction du
panel **training** a été lancée à 20:35:12, en arrière-plan caché, lanceur
**9880**. Journaux sous `runs/experiments/chronos2_exogenous_noaa_lora_v1/logs` :
`panel_training_20260907_203512.out.log` et `.err.log`. Revérifier son processus
réel avant toute reprise ; ces références ne prouvent pas qu'il travaille encore.
Le panel calibration, sa validation complète et l'entraînement neuronal restent
à exécuter après vérification des étapes précédentes. Les commandes ci-dessous
restent le protocole de référence ; aucun ancien checkpoint n'a été modifié.

À 20:37, le panel training est terminé : **6 050 240 lignes**, 730 origines
par pays, fichiers Parquet et sidecar publiés et SHA revérifié. Les processus
de construction training sont terminés. La construction du panel calibration
a ensuite démarré à **20:37:40**, lanceur **45300**, journaux
`panel_calibration_20260907_203740.out.log` et `.err.log` dans le même dossier
`logs`. Aucun entraînement neuronal n'a encore été lancé à ce point.
Le contrôle matériel confirme que `auto` choisira le CPU : ni CUDA ni XPU
n'est disponible dans cet environnement.

### Panels validés et entraînement principal lancé — 07/09/2026, 21:08 Paris

Les deux constructeurs sont terminés, sans erreur. `validate_panel` a ensuite
été exécuté intégralement et séquentiellement sur les deux panels (sorties 0) :

- Training : 6 050 240 lignes, découpage 335 apprentissage / 30 validation /
  365 holdout. SHA :
  `b901d539757fef2aff50400d8d75d742b86ea3798236a799107939657897888e`.
- Calibration : 9 075 360 lignes, 1 095 origines. Le plan OOF vérifié contient
  13 folds de 365 jours de fit chacun (335 + 30), douze blocs de 30 jours
  prédits puis un bloc de 5. SHA :
  `3a6758e8515ee8eeee967abc469a2621d67f39d4fc485298e14142245407b92a`.

Les deux panels réservent les mêmes livraisons holdout, du 03/09/2025 au
02/09/2026. La valeur 730 de `training_window_days` a servi **uniquement en
mémoire pour valider le panel long**, comme dans le validateur OOF ; le YAML
et la recette du réseau restent à 365 jours. Toutes les preuves de production
restent fausses. Aucun autre calcul lourd n'était actif avant lancement.

L'entraînement principal a été lancé à **21:07:59**, sans `--overwrite`,
avec la configuration figée rang 16 / 500 étapes / batch 64. Lanceur **59924** ;
journaux dans `runs/experiments/chronos2_exogenous_noaa_lora_v1/logs` :
`train_main_20260907_210759.out.log` et `.err.log`.
Processus Python effectif confirmé : **60420**, enfant de 59924, même démarrage
à 21:07:59. Le chargement des poids est terminé et la boucle affiche `0/500`.
L'avertissement `XPU device count is zero` est cohérent avec le CPU attendu ;
ce n'est pas à lui seul un échec du fit.
Ne pas le relancer tant que son processus réel ou son enfant travaille.
Le bundle `artifact` n'est considéré terminé qu'après publication et vérification
de son manifeste et de ses SHA. Après cette étape : `PrepareZones`, puis
calibrations OOF et évaluations selon les commandes ci-dessous.

### Entraînement terminé, zones vérifiées, OOF lancé — 08/09/2026, 02:38 Paris

L'entraînement principal a publié son bundle à **02:14:53 Paris** après
500 étapes et environ **5 h 05** de boucle fit/validation. Les anciens processus
59924/60420 sont terminés. `verify_bundle` a revérifié les empreintes :
checkpoint SHA **`5192e592eb8b0393bea7077c2423235ea46946bc46b6f4884dcdb1c8f2a7fa58`**.
La sélection sur validation retient l'étape **500**, perte 0,2116607726.
Ce chiffre est une perte de validation, pas une MAE de backtest ni une preuve
de supériorité sur les anciens modèles. Les hyperparamètres restent inchangés.

`PrepareZones` s'est terminé pour FR/DE/BE/NL. Une seconde vérification a
confirmé les quatre copies physiques indépendantes, leurs SHA identiques et
l'intégrité de la source, sans créer de nouvelle copie. Empreinte de l'arbre
source : `4318e036e7ec2bde22b3f30d40f77de900bd8c8e89c60329c0707efcd71a16ae`.
Journaux : `logs/prepare_zones_20260908_023653.out.log` et `.err.log` (vide).
Les **76 tests** OOF et PrepareZones ont réussi ; le plan réel à 13 folds a
également été vérifié contre le split du bundle. Une revue indépendante
confirme que chaque fold repart de Chronos-2 initial, pas du LoRA principal.

Après contrôle d'absence d'autre calcul et de la marge disque, la calibration
FR a démarré à **02:38:05 Paris**, en arrière-plan caché, lanceur **49196**.
Processus Python enfant confirmé : **58144**, même commande et même démarrage.
Journaux sous `runs/experiments/chronos2_exogenous_noaa_lora_v1/logs` :
`calibration_oof_fr_20260908_023805.out.log` et `.err.log`.
Elle utilise le panel calibration et son audit, les sorties `residual_calibration/fr`
et le cache partagé court **`runs/noaa_oof_v1`**, exactement comme ci-dessous.
Revérifier le processus enfant réel et les journaux avant toute reprise.
Ne lancer ni un autre pays ni un backtest tant que ce calcul travaille.

À cadence comparable au fit principal, les 13 entraînements peuvent demander
environ **66 heures**, hors inférences et validations initiales ; c'est une
estimation, à affiner sur le premier fold. DE/BE/NL réutiliseront les checkpoints,
pas les prédictions FR. Aucun ancien artefact ni configuration opérationnelle
n'a été modifié ; l'empreinte de la configuration d'activation reste
`62405d4b22be279a9b6554127ab21f8cd5d424a7c9f24a115c8d97b88ea01e04`.
Le rôle diagnostic et les preuves de production fausses sont conservés.

Toutes les commandes suivantes sont exécutées depuis
`C:\Users\BQ6757\chronos2_v1`, avec le Python
`C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe`.
Les exemples utilisent `$NoaaPython` uniquement comme raccourci :

```powershell
$NoaaPython = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
```

Ne lancer qu'un calcul lourd à la fois. Vérifier le processus réel, son
identité, les journaux et les SHA avant toute reprise. Un fichier lock n'est
pas une preuve de processus actif. Ne jamais écraser un bundle terminé ni
supprimer automatiquement un résultat ou une réservation.

### 1. Collecte météo

```powershell
& $NoaaPython -X faulthandler -u run_noaa_gfs_backfill.py `
  --start-day 2023-06-08 --end-day 2026-09-02 `
  --output-root runs/experiments/chronos2_exogenous_public_weather_v1/history `
  --cache-dir runs/experiments/chronos2_exogenous_public_weather_v1/raw_cache `
  --seed-directory runs/experiments/chronos2_exogenous_public_weather_v1/samples `
  --dependency-directory runs/experiments/chronos2_exogenous_public_weather_v1/deps `
  --workers 4
```

Ajouter `--max-new-days 7` pour le pilote, ou `--dry-run` pour le préflight.
Le journal est `history/journal.jsonl`, l'inventaire `history/inventory.json`.
L'agrégat `history/aggregate/noaa_gfs_weather.parquet` et son manifeste ne sont
publiés qu'après la couverture intégrale. `validate_aggregate` du même script
revérifie l'ensemble sans réseau. Une journée absente ou une publication après
cutoff arrête le batch : ne pas inventer ni interpoler une journée.

### 2. Panels et validation

```powershell
& $NoaaPython run_chronos2_noaa_panel.py --mode calibration --plan
& $NoaaPython run_chronos2_noaa_panel.py --mode training
& $NoaaPython run_chronos2_noaa_panel.py --mode calibration
& $NoaaPython run_chronos2_exogenous_finetune.py validate `
  --config config/chronos2_exogenous_noaa_lora_v1.yaml
```

Les panels se trouvent dans `runs/experiments/chronos2_exogenous_noaa_lora_v1/inputs`.
Les fichiers existants sont refusés : vérifier leur sidecar avant de sauter
une étape déjà achevée. Ne pas déplacer la période pour contourner une erreur.

### 3. Entraînement principal et séparation par pays

```powershell
& $NoaaPython -u run_chronos2_exogenous_finetune.py train `
  --config config/chronos2_exogenous_noaa_lora_v1.yaml
& '.\Exogenous.ps1' -Action PrepareZones `
  -Config config/chronos2_exogenous_noaa_lora_v1.yaml `
  -Zones FR,DE,BE,NL `
  -ZoneArtifactsRoot runs/experiments/chronos2_exogenous_noaa_lora_v1
```

Le bundle principal est `runs/experiments/chronos2_exogenous_noaa_lora_v1/artifact`.
Après vérification, les bundles pays sont dans `zones/FR/artifact`, etc.
Ne jamais utiliser `-Overwrite` pour relancer un entraînement fini.

`PrepareZones` est une copie de recherche, pas une décision de gouvernance.
Il conserve le rôle `diagnostic_only`, les preuves de production et tous les
contrôles d'intégrité du bundle source. Les copies restent donc refusées par
la gouvernance de production ; leur création n'autorise ni shadow formel ni
promotion. Le précontrôle du 07/09 a identifié et corrigé la confusion entre
ces deux validations avant le premier entraînement NOAA.

### 4. Correcteurs OOF, puis backtests pays

Répéter séquentiellement pour FR, DE, BE et NL (exemple FR ci-dessous) :

```powershell
& $NoaaPython -u run_chronos2_exogenous_calibrate_residual.py `
  --config config/chronos2_exogenous_noaa_lora_v1.yaml `
  --run-directory runs/experiments/chronos2_exogenous_noaa_lora_v1/zones/FR/artifact `
  --panel runs/experiments/chronos2_exogenous_noaa_lora_v1/inputs/residual_calibration_panel.parquet `
  --panel-audit runs/experiments/chronos2_exogenous_noaa_lora_v1/inputs/residual_calibration_panel.parquet.audit.json `
  --output-directory runs/experiments/chronos2_exogenous_noaa_lora_v1/residual_calibration/fr `
  --shared-checkpoint-cache-directory runs/noaa_oof_v1 `
  --item-id FR --block-days 30 --batch-size 64 --device-map auto
& $NoaaPython -u run_chronos2_exogenous_evaluate.py `
  --config config/chronos2_exogenous_noaa_lora_v1.yaml `
  --run-directory runs/experiments/chronos2_exogenous_noaa_lora_v1/zones/FR/artifact `
  --item-id FR --device-map auto --mode backtest
```

La première calibration entraîne les folds communs ; les suivantes doivent
réutiliser le cache vérifié. Ne pas remplacer l'OOF par le correcteur
expérimental validation30. Ne pas ajuster les hyperparamètres sur le holdout.
Le chemin court `runs/noaa_oof_v1` est réservé à cette expérience et doit être
fourni pour les quatre pays et chaque reprise. Il réduit le plus long chemin
temporaire identifié du cache partagé de 253 à 193 caractères sous Windows.
Il ne change ni les clés de contrat, ni les checkpoints, ni les contrôles SHA ;
aucun cache existant n'a été déplacé. Les sorties et correcteurs par pays
restent dans le répertoire d'expérience indiqué ci-dessus.

### 5. Rapports et décision

L'évaluateur dédié `run_chronos2_exogenous_research_oof_evaluate.py` produit
le rapport corrigé en conservant le statut recherche. Consulter son `--help`
pour les chemins exacts des artefacts OOF et les références facultatives.
Il doit comparer NOAA brut, NOAA corrigé OOF et les références brutes r8/r16
sur les mêmes heures et les mêmes observations, en indiquant que les entrées
diffèrent. Présenter MAE horaire, erreur des prix moyens journaliers et couverture.

Les observations des anciens rapports DE/BE/NL du 02/09/2026 diffèrent de
quelques millièmes d'EUR/MWh du cache canonique actuel (maximum observé :
0,0025 en DE/BE et 0,005 en NL). Pour une comparaison homogène, l'option
explicite `--reference-actual-policy canonical_recompute` recalcule les scores
des **prévisions déjà figées** contre les observations canoniques du panel
NOAA. Elle ne modifie aucun ancien fichier ; le rapport doit conserver les
anciennes observations et quantifier les écarts. Sans cette option, le
pairage reste strict et ces différences sont refusées. Ce changement des
observations d'évaluation ne doit pas être confondu avec un nouveau forecast.

Exemple FR, à exécuter seulement après les artefacts nécessaires :

```powershell
& $NoaaPython run_chronos2_exogenous_research_oof_evaluate.py `
  --config config/chronos2_exogenous_noaa_lora_v1.yaml `
  --run-directory runs/experiments/chronos2_exogenous_noaa_lora_v1/zones/FR/artifact `
  --calibration-directory runs/experiments/chronos2_exogenous_noaa_lora_v1/residual_calibration/fr `
  --calibration-panel runs/experiments/chronos2_exogenous_noaa_lora_v1/inputs/residual_calibration_panel.parquet `
  --calibration-panel-audit runs/experiments/chronos2_exogenous_noaa_lora_v1/inputs/residual_calibration_panel.parquet.audit.json `
  --output-directory runs/experiments/chronos2_exogenous_noaa_lora_v1/reports/fr `
  --zone FR --reference-actual-policy canonical_recompute `
  --reference-run 'Rang8=runs/experiments/chronos2_exogenous_lora_poc_v1/artifact' `
  --reference-run 'Rang16=runs/experiments/chronos2_exogenous_lora_poc_v2/zones/FR/artifact'
```

Pour DE/BE/NL, la référence rang 8 se situe dans
`runs/experiments/chronos2_exogenous_lora_poc_v1_recovered/zones/<ZONE>/artifact`.
Le rang 16 utilise toujours `chronos2_exogenous_lora_poc_v2/zones/<ZONE>/artifact`.

Ne pas exécuter `FinalBacktest`, `Shadow`, `Govern` ou `Promote` pour transformer
artificiellement ce candidat diagnostic en production. Si les résultats sont
prometteurs, préparer une qualification prospective distincte et les preuves
historiques requises pour revue, puis demander cette revue avant activation.
Les rapports et la note méthodologique restent dus même si le candidat perd.

## Surveillance pendant l'absence

L'automatisation de cette tâche reprend uniquement la première étape manquante,
à partir des processus et artefacts vérifiés. Elle ne relance pas les anciens
backtests et ne reste pas dans une boucle de constats inchangés. Elle est
silencieuse quand un calcul avance normalement, signale les étapes terminées
ou les erreurs utiles, et se met en pause après les rapports ou lorsqu'une
décision humaine est indispensable. L'ordinateur et l'application doivent
rester ouverts pour le suivi local.
