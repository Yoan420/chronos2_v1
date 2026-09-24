# Contrat de mise en production conditionnelle — Chronos-2 exogène

Le raccord du challenger exogène est présent dans `Forecast.ps1`, mais il est
conditionné par un contrat d'activation strict. Après promotion et activation,
`-Mode All` publie exactement deux chaînes par pays :

- `autonomous` : Chronos-2 + LoRA + correcteur résiduel ;
- `kalman` : la même sortie autonome suivie du Kalman standard.

Avec `-Action Run`, `Both` produit également `autonomous` et `kalman`, sur
l'incumbent tant que LoRA est inactif ou sur LoRA promu et activé. Il ne
déclenche aucune promotion. Le blend MKOnline reste séparé dans `Blend`
pour FR et NL. `kalman_weather` et
`kalman_hybrid` restent des challengers expérimentaux et ne font pas partie de
`All`. Le raccord ne modifie ni les recettes ni les archives live scellées.

À ce jour, aucun bundle ne possède encore l'ensemble des preuves PIT et shadow
nécessaires : la configuration d'activation livrée reste donc inactive. En
mode `All`, cette absence provoque un refus avant lancement ; elle ne déclenche
jamais un fallback silencieux vers l'ancien autonome.

## États et gates

Un bundle rolling365 portant la décision `shadow` ne peut jamais être
enregistré. La commande `register` exige toutes les preuves suivantes :

- `evaluation_role=primary_predeclared` dans le manifeste d'expérience ; les
  ablations `diagnostic_only` ne peuvent entrer dans la gouvernance ;
- décision scellée `promote` ;
- gates rolling365 et live shadow passées ;
- `production_pit_evidence=true` et `production_pipeline_evidence=true` dans
  la décision et dans le manifeste d'expérience ;
- checkpoint, schéma, holdout et manifeste live shadow intègres ;
- snapshot Chronos-2 local identique octet par octet à celui du fine-tuning ;
- runtime `per_zone` v1 explicitement déclaré ;
- correcteur résiduel final appris sur des prédictions OOF préquentielles et
  absent du holdout de promotion.

Le POC déclare volontairement `production_pit_evidence=false` pour les
historiques Saturn/JAO reconstruits. Le `Backtest` brut conserve également
`production_pipeline_evidence=false` : seule l'action `FinalBacktest` peut
faire passer cette seconde preuve à `true`, après comparaison appariée de la
sortie finale avec correcteur face à l'incumbent. Tant que la preuve PIT reste
fausse, le bundle peut rester en shadow mais il échoue normalement au
preflight production.

## Isolation des candidats et des zones

Un checkpoint peut apprendre conjointement sur FR, DE, BE et NL, mais toutes
les étapes qui suivent sont strictement mono-zone. Avant le premier `Backtest`,
le bundle d'entraînement terminé et encore vierge doit être forké :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' `
  -Action PrepareZones `
  -Zones FR,DE,BE,NL `
  -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v2\artifact'
```

Le lanceur crée des copies physiques complètes sous
`<parent-du-bundle>/zones/<ZONE>/artifact`. Il vérifie le bundle, le sidecar du
panel et ses SHA, refuse liens et hardlinks, prépare toutes les zones en
staging puis annule l'ensemble en cas d'échec. La source n'est jamais modifiée.
Une relance n'est acceptée que tant que la copie est restée identique ; après
un backtest, il faut continuer directement avec le dossier de la zone.

Le panel actuel contient FR/DE/BE/NL, pas ES. Une activation `All` incluant ES
reste donc impossible sans entraîner un candidat contenant ES. De même, les
rangs LoRA 8 et 16 sont deux candidats distincts : chacun conserve son propre
répertoire, son propre backtest final et sa propre décision de gouvernance.
Le rang 16 ne remplace jamais automatiquement le rang 8. La recette du rang 8
est figée dans `config/chronos2_exogenous_lora_rank8_reference.yaml`, afin que
ses folds OOF ne soient jamais recalculés avec les hyperparamètres du rang 16.

## Correcteur résiduel obligatoire

Le bundle de production doit contenir
`artifacts/residual_corrector/residual_corrector.json`. Le format v1 est un
correcteur linéaire sûr, sans désérialisation `pickle` :

```json
{
  "schema_version": 1,
  "model_kind": "linear_shift_v1",
  "base_model": "chronos2_exogenous",
  "output_model": "exogenous_residual_corrected",
  "fit_protocol": "blocked_prequential_oof_rolling365",
  "training_days": 365,
  "selection_frozen_before_holdout": true,
  "holdout_used_for_fit": false,
  "future_actuals_used_as_features": false,
  "oof_training_predictions_sha256": "<sha256>",
  "oof_training_audit_sha256": "<sha256>",
  "candidate_checkpoint_sha256": "<sha256>",
  "feature_columns": ["intercept", "local_hour_sin", "local_hour_cos"],
  "feature_means": [0.0, 0.0, 0.0],
  "feature_scales": [1.0, 0.707, 0.707],
  "coefficients": [0.0, 0.0, 0.0],
  "maximum_absolute_shift_eur_mwh": 20.0
}
```

Son SHA-256 doit être déclaré dans `experiment_manifest.json` sous
`residual_corrector_sha256`. Le même manifeste doit porter :

```json
{
  "candidate_output_stage": "exogenous_residual_corrected",
  "production_runtime": {
    "schema_version": 1,
    "layout": "per_zone",
    "cross_learning": false,
    "target": "target"
  }
}
```

La gate évalue donc la sortie finale corrigée, pas la sortie LoRA brute. Les
coefficients peuvent consommer les variables futures du schéma ou les
variables calendaires sûres documentées par le runtime. Toute colonne absente,
valeur non finie ou correction supérieure au clip fait échouer le run.

Le fit reproductible refuse tout chevauchement avec le holdout et exige 365
jours physiques consécutifs de prédictions OOF, avec une origine exactement à
D-1 08:00 locale (y compris les jours DST). Il exige aussi un sidecar JSON
SHA-lié au fichier de prédictions. La version 2 du sidecar prouve que le
checkpoint final n'a **pas** été rejoué sur ses propres jours de fit : un
adaptateur LoRA distinct est entraîné pour chaque bloc, sur les 365 origines
strictement antérieures au premier jour du bloc. Les 30 dernières origines de
ce lookback restent la validation LoRA ; les 335 autres servent au gradient.
Les actuals des blocs déjà émis peuvent entrer dans les refits suivants, ce qui
est autorisé par le protocole préquentiel ; l'actual du bloc courant et tout le
holdout final restent exclus.

Le contrat complet exige donc 1 095 origines : 365 jours d'amorçage, 365 jours
OOF pour le correcteur, puis 365 jours de holdout final fermé. Le sidecar scelle
l'identité du checkpoint final comme ancre de déploiement, mais lie chaque
plage OOF au SHA du checkpoint fold-specific qui l'a réellement produite :

```json
{
  "schema_version": 2,
  "purpose": "chronos2_exogenous_blocked_prequential_oof",
  "fit_protocol": "blocked_prequential_oof_rolling365",
  "predictions_sha256": "<sha256 du CSV OOF>",
  "candidate_checkpoint_sha256": "<sha256 du checkpoint final, ancre>",
  "candidate_checkpoint_role": "deployment_identity_anchor_not_oof_predictor",
  "deployment_checkpoint_used_for_oof": false,
  "fold_checkpoints_are_origin_specific": true,
  "fold_checkpoint_set_sha256": "<sha256 de la liste des checkpoints folds>",
  "fold_candidate_recipe_sha256": "<sha256 schema+snapshot+hyperparametres>",
  "fold_lookback_days": 365,
  "folds": [
    {
      "fold_index": 1,
      "fit_origins": {"count": 365, "first_utc": "...", "last_utc": "..."},
      "prediction_origins": {"count": 30, "first_utc": "...", "last_utc": "..."},
      "checkpoint_sha256": "<sha256 du checkpoint fold 1>",
      "predictions_sha256": "<sha256 des predictions fold 1>",
      "refit_uses_only_strictly_prior_days": true
    }
  ],
  "training_start_day": "2025-08-01",
  "training_end_day": "2026-07-31",
  "holdout_start_day": "2026-08-01",
  "refit_uses_only_strictly_prior_days": true,
  "same_day_actual_excluded_from_fit": true,
  "selection_frozen_before_oof": true
}
```

Le lanceur dédié reconstruit d'abord le panel long, puis reprend automatiquement
les folds déjà terminés. `-BlockDays 30` produit 13 folds (12 blocs de 30 jours
et un bloc de 5 jours). Aucun calcul n'est lancé par `-DryRun` :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' `
  -Action CalibrationPanel -Zones FR,DE,BE,NL -Pack full `
  -EndDay 2026-09-02 -Overwrite

& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' `
  -Action CalibrateResidual -Zones FR -Pack full `
  -BlockDays 30 -Device auto `
  -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact'
```

Les sorties sont écrites sous
`<bundle>/residual_calibration/fr/` : prédictions OOF, sidecar v2,
`residual_corrector.json`, manifeste global et caches immuables par fold. Une
interruption peut être relancée avec la même commande. `-Overwrite` est
volontairement interdit pour cette action ; une nouvelle recette doit utiliser
un autre `-ResidualCalibrationDirectory`.

Pour les copies créées par `PrepareZones`, les checkpoints neuronaux des folds
sont mutualisés automatiquement dans
`<artefact-source-parent>/shared_oof_fold_checkpoints/`. Cette mutualisation est
sûre parce que le fit de chaque fold porte sur le même panel multi-zone complet.
Le namespace de cache lie exactement le SHA du checkpoint de déploiement, le
SHA du panel et de son audit, le split complet, les plages des folds, le snapshot
de base et tous les hyperparamètres LoRA. Chaque checkpoint est publié
atomiquement avec un sceau SHA-256, puis copié physiquement dans la zone ; tout
écart arrête la calibration. Les prédictions OOF et le correcteur résiduel ne
sont jamais partagés et restent propres à FR, DE, BE ou NL. Une racine explicite
peut être fournie avec `-OofCheckpointCacheDirectory`.

Un claim atomique par fold garantit qu'un seul processus effectue le fit : les
autres attendent de façon bornée puis chargent la publication scellée. Sous
Windows, le claim lie le PID à l'heure de création réelle du processus afin de
détecter sans ambiguïté un PID réutilisé. Après un crash local, un claim dont le
processus n'existe plus est mis en quarantaine puis retiré ; un propriétaire
vivant, distant ou non vérifiable n'est jamais évincé et conduit à un arrêt
fail-closed au terme de l'attente. Enfin, un `residual_corrector.json` existant
n'est repris que si un `calibration_manifest.json` déjà `complete` authentifie
son SHA et ceux des prédictions/audits. Le manifeste complet n'est jamais
réécrit lors d'une reprise.

Cette action ne modifie ni `production_pit_evidence`, ni
`production_pipeline_evidence`, ne produit aucune décision `promote` et ne
branche rien automatiquement dans le run live.

## Évaluation finale appariée rolling-365

Le `Backtest` initial mesure le LoRA **brut** sur le holdout fermé. Il ne doit
pas être utilisé directement pour promouvoir la chaîne opérationnelle. Après
la calibration OOF, l'action suivante applique le correcteur résiduel gelé au
même holdout puis l'apparie au vrai modèle autonome incumbent
`residual_corrected` :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' `
  -Action FinalBacktest `
  -Zones FR `
  -Pack full `
  -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact'
```

Par défaut, le lanceur lit le correcteur et son audit sous
`<bundle>/residual_calibration/fr/`, puis sélectionne le fichier incumbent
`statistics_history_hourly.csv.gz` le plus récent de la zone. Utiliser
`-ResidualCalibrationDirectory` ou `-IncumbentStatistics` pour fournir des
chemins explicites.

L'évaluation refuse toute fenêtre non appariée : elle exige exactement 365
jours locaux et leur nombre exact d'heures physiques (8 759, 8 760 ou 8 761
selon les frontières DST), conserve les journées DST de 23 et 25 heures,
vérifie les origines D-1 08:00, les prix observés et les SHA-256 du
checkpoint, du correcteur et de son audit OOF. Elle compare alors :

- baseline : autonome incumbent + son correcteur résiduel ;
- challenger : Chronos-2 + LoRA + correcteur résiduel OOF gelé.

Les sorties atomiques sont écrites sous `<bundle>/final_pipeline/` :

- `final_pipeline_predictions.csv.gz`, preuve rolling-365 à transmettre à
  `Govern` ;
- `final_pipeline_metrics.json` et `final_pipeline_report.html` ;
- `final_pipeline_audit.json` et `final_pipeline_manifest.json`.

Une exécution réussie met à jour le manifeste d'expérience avec
`candidate_output_stage=exogenous_residual_corrected` et
`production_pipeline_evidence=true`. La valeur de
`production_pit_evidence` est un invariant : `FinalBacktest` la conserve
strictement. Cette action ne décide donc ni la promotion ni l'activation ;
`Govern` doit encore évaluer la preuve finale et, plus tard, les 30 jours de
shadow prospectif.

## Comparaison reproductible du rang 8 et du rang 16

`CompareCandidates` traite le rang 8 comme incumbent et le rang 16 comme
challenger. Le comparateur ne relance aucune inférence et ne modifie aucun des
deux artefacts. Il exige la même zone, les mêmes 365 jours physiques, les mêmes
heures, origines, prix observés et quantiles de l'incumbent d'origine. Les
empreintes des manifestes, prédictions et métriques sont vérifiées avant de
recalculer les métriques appariées et tous les seuils de la politique de
gouvernance :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' `
  -Action CompareCandidates `
  -Zones FR `
  -Rank8Candidate 'C:\chemin\rang8\FR\artifact\final_pipeline\final_pipeline_manifest.json' `
  -Rank16Candidate 'C:\chemin\rang16\FR\artifact\final_pipeline\final_pipeline_manifest.json' `
  -ComparisonOutput 'runs\experiments\chronos2_exogenous\candidate_comparisons\FR\final' `
  -Overwrite
```

Un run, son dossier `final_pipeline`, son manifeste, son fichier de prédictions
ou son fichier de métriques peuvent être fournis. Si un run contient déjà une
preuve finale, celle-ci est choisie par défaut. Les sorties atomiques sont
`rank8_vs_rank16_comparison.json`, `rank8_vs_rank16_report.html` et le détail
journalier `rank8_vs_rank16_daily.csv`. La décision indique seulement quel
candidat franchit les seuils face au rang 8 ; elle conserve toujours
`promotion_performed=false` et `activation_performed=false`.

Même avec deux dossiers `final_pipeline`, une sélection opérationnelle n'est
émise que si **les deux** preuves portent `production_pit_evidence=true` et si
leur chaîne complète est encore reproductible. Le comparateur recalcule le
contrat causal/freeze, vérifie les détails concordants des manifestes et de
l'audit final, le runtime `per_zone`, puis recharge les sources SHA-liées. Il
réapplique le correcteur au LoRA brut, contrôle son sidecar OOF strictement
pré-holdout et apparie à nouveau l'incumbent `residual_corrected`. Une source
absente ou modifiée, un audit factice, une preuve PIT fausse ou une divergence
de quantiles force `candidate_selection_eligible=false` et une décision
`research_preference_rank8|rank16`. Aucun `select_rank*` ne peut alors être
émis. `promotion_eligible` reste `false` dans tous les cas : la promotion relève
exclusivement de `Govern` après le shadow prospectif.

Lorsque le correcteur OOF et le `FinalBacktest` ne sont pas encore disponibles,
la même action accepte deux preuves **brutes** `evaluation_manifest.json` ou
`evaluation_predictions.csv.gz`. Elle vérifie en plus que
`input_contract_sha256` est identique. Ce résultat est explicitement étiqueté
`evidence_stage=raw_lora`, `candidate_selection_eligible=false` et
`promotion_eligible=false` : il donne une préférence de recherche rang 8/rang
16, mais ne peut jamais être présenté comme une preuve de la chaîne
opérationnelle corrigée. Mélanger une preuve brute et une preuve finale est
refusé.

## Enregistrement inactif

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  'C:\Users\BQ6757\chronos2_v1\run_chronos2_exogenous_production.py' `
  register `
  --bundle 'C:\chemin\bundle-promu-fr' `
  --alias 'fr-exogenous-v1' `
  --zone FR
```

L'enregistrement sauvegarde le chemin et les empreintes du bundle dans
`runs/experiments/chronos2_exogenous/production_registry.json`. Il conserve
toujours `enabled_by_default=false`. Toute mutation ultérieure du bundle est
détectée lors du chargement.

Le contrôle peut être rejoué sans charger Chronos-2 :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  'C:\Users\BQ6757\chronos2_v1\run_chronos2_exogenous_production.py' `
  preflight --alias 'fr-exogenous-v1' --zone FR
```

## Exécution dédiée et activation explicitement opt-in

Le matérialiseur amont doit produire un panel ne contenant qu'une origine
D-1 08:00 et la journée demandée : contexte cible observé, cible future vide,
prévisions exogènes futures complètes. Son audit doit fournir le SHA-256 du
panel, `production_ready=true` et une preuve PIT vraie pour chaque zone.

Une interface dédiée accepte maintenant un manifeste JSON de captures live :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  'C:\Users\BQ6757\chronos2_v1\run_chronos2_exogenous_panel.py' `
  --mode shadow `
  --zones FR `
  --pack full `
  --end-day 2026-09-04 `
  --live-source-manifest `
    'C:\Users\BQ6757\chronos2_v1\config\chronos2_exogenous_live_sources.json' `
  --output 'C:\chemin\live_panel.parquet'
```

Le contrat est illustré dans
`config/chronos2_exogenous_live_sources.example.json`. Cet exemple décrit le
pack `full` FR et reprend exactement les identités de sources utilisées au
fit. Les noms, familles et colonnes d'un manifeste live doivent correspondre
au manifeste d'entraînement ; ajouter ou renommer une source est refusé.
Chaque source doit déjà être matérialisée prospectivement et accompagnée d'un
sidecar indépendant lié aux octets Parquet par SHA-256. Le sidecar doit déclarer
`production_evidence_kind=prospective_capture`, zéro violation causale et zéro
violation de capture opérationnelle. Le loader ne crée aucune preuve et refuse
explicitement `versioned_revision_history`; les backfills actuels conservent
donc leur classification recherche. L'option est réservée au mode `shadow` et
force automatiquement les contrôles de complétude et de preuve production. Le
contrat actuel exige aussi les 2 048 heures de contexte : il faut donc environ
86 jours de captures prospectives complètes avant le premier shadow strict.

Après avoir copié l'exemple vers un manifeste réel et renseigné les chemins
des captures, la boucle quotidienne est disponible dans le lanceur principal
du POC :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' `
  -Action Shadow `
  -Zones FR `
  -Pack full `
  -EndDay 2026-09-04 `
  -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact' `
  -LiveSourceManifest `
    'config\chronos2_exogenous_live_sources.json'
```

La première exécution, avant publication du prix, scelle les quantiles dans le
journal append-only. Relancer la même commande après publication joint les
actuals sans refaire l'inférence. Le journal et son manifeste v3 restent des
artefacts d'audit bruts. La commande publie séparément
`shadow_final/shadow_final_evidence.csv.gz` (journées observées utilisées pour
le scoring), `shadow_final_issued_history.csv.gz` (toutes les émissions finales,
actual nullable, utilisé pour le bootstrap causal) et
`shadow_final_manifest.json` v4. Le manifeste lie le correcteur OOF,
l'incumbent `residual_corrected`, le journal, le schéma et le FinalBacktest ; le
bundle de promotion embarque tout ce dossier de provenance.

Après 30 journées complètes, exécuter `Exogenous.ps1 -Action Govern` avec la
preuve finale rolling-365 produite par `FinalBacktest` et uniquement la preuve
shadow finale v4. Le CSV brut de `Backtest`,
`shadow_observed_evidence.csv.gz` et `shadow_manifest.json` v3 ne sont pas des
preuves de pipeline final admissibles. Seul un bundle retournant
`decision=promote` peut ensuite être enregistré par la commande `register`
ci-dessus.

La fenêtre finale et le shadow doivent former une seule chronologie : le
premier jour du shadow observé et de l'historique émis doit être exactement le
lendemain du dernier jour rolling-365. Le panel effectivement utilisé par le
rang 16 ferme son holdout sur la livraison du 02/09/2026. Comme ce candidat
n'était pas encore gelé et émis prospectivement pour le 03/09, commencer son
shadow plus tard laisserait un trou ; il est donc **non activable en l'état**,
même après 30 jours. Il faut reconstruire un candidat avec un holdout actualisé
jusqu'à la veille du shadow, ou constituer un bridge causal prospectif scellé
et auditable.

## Nouveau cycle prospectif two-phase (sans rétrodatation)

Le bridge rétrospectif reste interdit. Pour un nouveau candidat, le lanceur
sépare désormais deux événements :

1. **Phase A, avant la première origine shadow** : gel immuable du checkpoint,
   du correcteur OOF, de leurs SHA, du holdout exact de 365 jours et du premier
   jour shadow. La target peut être `NaN` uniquement sur les 23/24/25 heures de
   l'horizon de la toute dernière origine holdout. Elle reste obligatoire et
   finie dans tous les contextes, dans train, validation et OOF. Les covariables
   passées et connues-futures ne bénéficient d'aucune tolérance.
2. **Phase B, après publication du prix** : un second panel remplit uniquement
   les cellules `NaN` pré-déclarées. Chaque autre ligne et chaque autre valeur
   doivent être strictement identiques. Le panel gelé, le panel résolu et leurs
   deux sidecars restent séparés et SHA-liés. Le `Backtest`, puis le
   `FinalBacktest`, peuvent alors être finalisés sans changer le candidat déjà
   émis en shadow.

Ce mécanisme ne modifie aucun seuil de gouvernance, ne fabrique aucune preuve
PIT et ne promeut rien. `production_pit_evidence=false`, côté entraînement **ou**
côté OOF, interdit toujours `EpochFreeze`. Le fichier d'epoch porte toujours
`promotion_eligible=false`.

L'opt-in est désactivé par défaut. Une **nouvelle** configuration (ne pas
modifier celle d'un artefact existant) doit déclarer :

```powershell
Copy-Item `
  'config\chronos2_exogenous_lora_prospective.example.yaml' `
  'config\chronos2_exogenous_lora_prospective.yaml'
```

Remplacer ensuite toutes les occurrences `RENAME_ME`, puis conserver :

```yaml
data:
  allow_unresolved_final_evaluation_day: true
  production_pit_evidence: true
```

Il faut aussi utiliser de nouveaux `experiment_id`, `data.panel_path` et
`output.directory`. La valeur `production_pit_evidence: true` n'est acceptée
que si les sidecars de toutes les sources prouvent réellement leurs captures
opérationnelles prospectives. Les backfills de recherche actuels restent
classés `false`.

### Dates concrètes au 04/09/2026 après 08:00 Europe/Paris

La première frontière encore théoriquement ouverte est :

- holdout livraison : **06/09/2025 au 05/09/2026**, 365 jours contigus ;
- gel phase A strictement avant **05/09/2026 08:00 Europe/Paris**
  (`06:00Z`) ;
- première livraison shadow : **06/09/2026** ;
- 30e livraison shadow : **05/10/2026** ;
- première gouvernance possible : **06/10/2026**, seulement après résolution
  de toutes les observations et validation des gates existants.

Si le gel n'est pas terminé avant le 05/09 à 08:00, cette frontière est perdue.
La suivante devient : holdout **07/09/2025–06/09/2026**, gel avant
**06/09/2026 08:00**, premier shadow **07/09/2026**, 30e jour **06/10/2026**,
gouvernance au plus tôt **07/10/2026**. `EpochEarliest` recalcule cette frontière
à partir de l'horloge réelle :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action EpochEarliest
```

Ces dates ne sont réalisables que si les captures PIT opérationnelles exigées
existent déjà et si entraînement + OOF finissent avant le deadline. L'artefact
rang 16 actuel porte `production_pit_evidence=false` et ne satisfait donc pas
cette condition.

### Phase A — commandes d'exemple, à ne lancer que sur un nouveau candidat

```powershell
# Les deux panels pré-déclarent uniquement le dernier horizon target nullable.
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action Panel `
  -Config 'config\chronos2_exogenous_lora_prospective.yaml' `
  -Zones FR,DE,BE,NL -Pack full -EndDay 2026-09-05 `
  -ProspectiveFreeze -Overwrite

& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action CalibrationPanel `
  -Config 'config\chronos2_exogenous_lora_prospective.yaml' `
  -Zones FR,DE,BE,NL -Pack full -EndDay 2026-09-05 `
  -ProspectiveFreeze -Overwrite

& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action Validate `
  -Config 'config\chronos2_exogenous_lora_prospective.yaml'
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action Train `
  -Config 'config\chronos2_exogenous_lora_prospective.yaml'

# PrepareZones et CalibrateResidual restent inchangés. Pour chaque zone, le
# correcteur doit être entièrement terminé avant l'origine du premier shadow.
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action EpochPlan `
  -Config 'config\chronos2_exogenous_lora_prospective.yaml' -Zones FR `
  -RunDirectory 'runs\experiments\nouveau\zones\FR\artifact' `
  -ResidualCalibrationDirectory 'runs\experiments\nouveau\zones\FR\residual' `
  -FirstShadowDay 2026-09-06

& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action EpochFreeze `
  -Config 'config\chronos2_exogenous_lora_prospective.yaml' -Zones FR `
  -RunDirectory 'runs\experiments\nouveau\zones\FR\artifact' `
  -ResidualCalibrationDirectory 'runs\experiments\nouveau\zones\FR\residual' `
  -ShadowEpochDirectory 'runs\experiments\nouveau\zones\FR\shadow_epoch' `
  -FirstShadowDay 2026-09-06

& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action Shadow `
  -Config 'config\chronos2_exogenous_lora_prospective.yaml' -Zones FR `
  -RunDirectory 'runs\experiments\nouveau\zones\FR\artifact' `
  -ResidualCalibrationDirectory 'runs\experiments\nouveau\zones\FR\residual' `
  -ShadowEpochDirectory 'runs\experiments\nouveau\zones\FR\shadow_epoch' `
  -EndDay 2026-09-06 -Pack full `
  -LiveSourceManifest 'config\chronos2_exogenous_live_sources.json'
```

`EpochPlan` est read-only. `EpochFreeze` est atomique, sans overwrite et refuse
si l'heure limite est passée, si le premier shadow n'est pas le lendemain exact
du holdout, si un SHA diverge ou si l'une des deux preuves PIT vaut `false`.
Avant chaque livraison prospective, `Shadow` rejoue aussi un préflight lié à
l'epoch : sans journal, seule la première journée pré-engagée est admise ; avec
un journal, une journée déjà publiée peut être rejouée de façon idempotente et
la seule nouvelle journée admise est le lendemain exact. La grille locale
23/24/25 heures et l'origine D-1 de chaque journée déjà publiée sont revérifiées.

### Phase B — résolution tardive et preuve finale

Après disponibilité de toute la target du 05/09, reconstruire le même panel
vers **un autre chemin**. Le panel gelé ne doit jamais être écrasé :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action ResolveHoldout `
  -Config 'config\chronos2_exogenous_lora_prospective.yaml' `
  -Zones FR,DE,BE,NL -Pack full -EndDay 2026-09-05 `
  -ResolvedEvaluationPanel 'runs\experiments\nouveau\inputs\training_panel_resolved.parquet'

& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action Backtest `
  -Config 'config\chronos2_exogenous_lora_prospective.yaml' -Zones FR `
  -RunDirectory 'runs\experiments\nouveau\zones\FR\artifact' `
  -ResolvedEvaluationPanel 'runs\experiments\nouveau\inputs\training_panel_resolved.parquet'

& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action FinalBacktest `
  -Config 'config\chronos2_exogenous_lora_prospective.yaml' -Zones FR `
  -RunDirectory 'runs\experiments\nouveau\zones\FR\artifact' `
  -ResidualCalibrationDirectory 'runs\experiments\nouveau\zones\FR\residual'

& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action EpochFinalizeCheck `
  -Config 'config\chronos2_exogenous_lora_prospective.yaml' -Zones FR `
  -RunDirectory 'runs\experiments\nouveau\zones\FR\artifact' `
  -ShadowEpochDirectory 'runs\experiments\nouveau\zones\FR\shadow_epoch'
```

Le `Shadow` brut peut donc commencer avant le `FinalBacktest`. Tant que celui-ci
n'existe pas, le lanceur conserve le journal append-only et affiche une
finalisation différée. Il ne crée la preuve shadow finale qu'après scellement du
`FinalBacktest`; `EpochFinalizeCheck` vérifie alors que checkpoint, correcteur,
OOF, fenêtre, panel non résolu, panel résolu et leurs sidecars correspondent
exactement au pré-engagement.

Le lanceur refuse explicitement que `-ResolvedEvaluationPanel` désigne le
panel gelé. Pour une configuration two-phase, `Govern` exige également
`-ShadowEpochDirectory` et rejoue `EpochFinalizeCheck` avant toute décision.
Le CLI Python de gouvernance applique le même verrou via
`--shadow-epoch-directory` : appeler directement le runner ne permet donc pas
de contourner le pré-engagement.

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  'C:\Users\BQ6757\chronos2_v1\run_chronos2_exogenous_production.py' `
  run `
  --alias 'fr-exogenous-v1' `
  --delivery-day 2026-09-04 `
  --live-panel 'C:\chemin\live_panel.parquet' `
  --live-panel-audit 'C:\chemin\live_panel.parquet.audit.json' `
  --output-directory 'C:\chemin\sortie-immutable'
```

La sortie contient :

- `forecast_hourly_fr.csv`, avec `chronos2_exogenous__q10/q50/q90` et
  `exogenous_residual_corrected__q10/q50/q90` ;
- `backtest_hourly_oof.csv.gz`, directement dérivé du holdout promu ;
- le schéma et le correcteur utilisés ;
- `run_manifest.json` et `artifact_checksums.json`.

Le launcher principal ne consomme cette sortie qu'après enregistrement du
bundle promu puis activation explicite dans
`config/chronos2_exogenous_activation_v1.yaml`. Pour chaque zone, le contrat
épingle l'alias du registre, l'identité du candidat, les SHA-256 du bundle et
des artefacts, ainsi que le manifeste de sources live prospectives et son
SHA-256. Les deux entrées `autonomous` et `kalman` doivent être activées pour
que la zone puisse participer à `-Mode All`.

Le preflight recharge le bundle depuis le registre et vérifie toutes ces
identités avant tout calcul. Une activation incomplète, une mutation du
bundle, un manifeste live absent ou divergent, ou une preuve de promotion
insuffisante arrête le batch. La politique est `validation_failure_policy:
error` : un mode déclaré actif ne peut jamais retomber sur l'incumbent. La
configuration ne doit être renseignée qu'à partir des valeurs d'un bundle
réellement promu ; modifier manuellement les champs ne fabrique aucune preuve.

## Limites avant activation réelle

1. Le matérialiseur quotidien du panel doit encore être industrialisé et
   audité pour toutes les sources météo, combustibles et JAO. L'interface de
   manifeste ne remplace pas ce service externe de capture prospective.
2. Le format runtime v1 est volontairement `per_zone`. Le modèle CWE joint
   devra recevoir une gate portefeuille supplémentaire avant extension.
3. Les jours 23/25 heures sont acceptés à l'inférence mais doivent faire
   l'objet d'un test réel avec le checkpoint final.
4. Les poids MKOnline existants ont été calibrés sur l'ancien modèle autonome.
   Ils ne doivent pas être réutilisés : un `blend_exogenous` nécessitera son
   propre backtest et sa propre promotion.
5. Le raccord technique à `Forecast.ps1` existe, mais il doit rester inactif
   jusqu'à la production d'un bundle réellement promu et d'un panel live
   automatisé. Son activation est explicite et donne à `-Mode All` sa
   sémantique stricte à deux sorties. `Both` permet ces deux vues avec
   l'incumbent quand LoRA est inactif ; `Blend` conserve le MKOnline historique.
