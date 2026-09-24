# Laboratoire de fine-tuning des modèles auxiliaires

Ce laboratoire est séparé du forecast live. Il lit une archive existante,
écrit uniquement sous `runs/experiments/auxiliary_lab` et ne modifie ni
`Forecast.ps1`, ni les YAML de production, ni `runs/live`.

Les trois composants opérationnels sont couverts :

- le correcteur résiduel, seul ou en blend CatBoost/HGB ;
- le poids du blend MKOnline (le modèle fournisseur n'est pas entraînable ici) ;
- les paramètres et familles KF/EKF/UKF de la couche Kalman.

## Démarrage rapide

Copier `config/auxiliary_lab.yaml`, puis changer au minimum
`experiment_id`, `source_run` et `output_directory`.

```powershell
& '.\FineTune.ps1' -Action List
& '.\FineTune.ps1' -Action Validate -Config 'config\auxiliary_lab.yaml'
& '.\FineTune.ps1' -Action Train -Config 'config\auxiliary_lab.yaml'
& '.\FineTune.ps1' -Action Retrain -Config 'config\auxiliary_lab.yaml'
```

Pour construire les prévisions météo PIT utilisées par Kalman :

```powershell
& '.\FineTune.ps1' -Action Weather `
  -StartDay '2024-06-30' -EndDay '2026-08-29' `
  -Countries FR,DE,BE,NL,ES
```

Cette commande matérialise trois séries Saturn par zone (température 2 m,
production éolienne et production solaire) sous
`data/pit/kalman_weather`. `-Output`, `-SeriesWorkers`, `-DayWorkers`
et `-DryRun` permettent respectivement de changer la destination, régler le
parallélisme et inspecter le plan sans requête. Le produit des deux nombres de
workers est plafonné à 32. Un lancement identique reprend les fichiers déjà
complets seulement après contrôle de la plage, de la timeline physique, du
contrat et du SHA-256.

L'expérience FR prête à l'emploi se lance ensuite ainsi :

```powershell
& '.\FineTune.ps1' -Action Validate `
  -Config 'config\auxiliary_lab_kalman_weather.yaml'

& '.\FineTune.ps1' -Action Retrain `
  -Config 'config\auxiliary_lab_kalman_weather.yaml'
```

Elle compare deux valeurs de `q_over_r` sur 60 jours de validation, elles-mêmes
rejouées en rolling 365 exact, puis n'ouvre les 365 jours de test que pour le
gagnant. `rolling_refit_workers: 4` parallélise les origines indépendantes sans
changer le modèle ; les valeurs autorisées vont de 1 à 8.

Pour ne lancer qu'une partie :

```powershell
& '.\FineTune.ps1' -Action Train `
  -Config 'config\auxiliary_lab.yaml' `
  -Models 'residual_corrector,kalman'
```

Équivalents Python :

```powershell
python -m auxiliary_lab list-models
python -m auxiliary_lab validate --config config/auxiliary_lab.yaml
python -m auxiliary_lab train --config config/auxiliary_lab.yaml
```

## Protocole

Les journées sont toujours découpées sur le fuseau de livraison et validées
sur leur timeline physique : 23, 24 ou 25 heures aux changements DST.

1. **Train** : apprentissage des modèles ou calibration initiale du poids.
2. **Validation** : choix de la configuration selon `objective`.
3. **Test** : ouverture uniquement pour la configuration sélectionnée.
4. **Modèle full** : réentraînement sur toutes les observations après gel des
   métriques ; il sert exclusivement à l'inférence future.

Le champ `horizon_hours` vaut normalement `null` pour évaluer le jour complet.
Une valeur entière limite les métriques aux premières positions physiques de
chaque journée.

L'archive live FR actuelle contient environ 381 jours appariés pour MKOnline,
contre plus de 730 jours OOF Chronos pour le correcteur. Le chargeur conserve
l'OOF scellé comme historique d'entraînement et superpose les Statistics
publiées les plus récentes. Lorsque `residual_corrector` et `kalman` sont
lancés ensemble, le Kalman reçoit automatiquement un historique résiduel
causal produit par l'expérience. Le correcteur sélectionné est réentraîné par
blocs (28 jours par défaut) avec uniquement les labels des jours strictement
antérieurs, sur une fenêtre glissante de 365 jours. Tant que le nombre minimal
de labels n'est pas atteint, q10/q50/q90 restent identiques à Chronos. Les
prévisions `residual_corrected` déjà publiées et complètes sont réutilisées sur
validation/test, journée physique entière par journée physique entière. Les
artefacts `kalman_upstream_prequential_audit.csv` et `.json` documentent chaque
refit, sa borne maximale de label et les overlays publiés.

Le sous-bloc strict `prequential_bridge` du correcteur permet de modifier
`refit_cadence_days`, `training_lookback_days` et `cold_start_policy` (seule la
politique sûre `identity` est acceptée).
`history_prefix_path` peut en plus pointer vers un préfixe OOF Chronos strict
(`delivery_start_utc`, `forecast_origin_utc`, `q10/q50/q90`, `actual`). Le
chargeur exige l'origine contractuelle D-1 08:00, vérifie les conflits sur le
chevauchement et inscrit le SHA-256 du fichier dans les manifests.

Le YAML fourni réserve donc directement les 365 derniers jours au test de
`residual_corrector` + `kalman`. MKOnline doit rester désactivé, sauf si une
archive de calibration couvrant aussi train et validation avant ces 365 jours
est disponible.

Pour Kalman, `training_lookback_days: 365` active un vrai protocole
walk-forward : pour chaque journée de livraison `D`, le filtre est recréé et
entraîné uniquement sur les journées physiques `D-365` à `D-1`. Ni le prix
observé de `D`, ni une météo publiée après le cutoff ne peuvent entrer dans ce
fit. Il faut donc matérialiser au moins 365 jours avant la première journée à
évaluer, en plus de la validation et des 365 jours de test. Le fichier météo
FR fourni utilise 60 jours de validation et 365 jours de test, du 29 août 2025
au 28 août 2026 ; le préfixe OOF porte l'historique Chronos total à 970 jours
et les vintages Saturn du 30 juin 2024 au 29 août 2026 couvrent exactement le
support rolling requis avant la validation.
La sélection des hyperparamètres applique le même contrat rolling que le test,
et le test scellé est rejoué séparément après la sélection.

Storm n'est jamais chargé comme variable d'entraînement ou de sélection.
MKOnline est interdit comme entrée du correcteur autonome et de Kalman ; il
n'apparaît que dans son adapter de blend dédié.

## Paramétrage

- `fixed_parameters` contient les paramètres communs.
- `parameter_grid` contient des listes et génère un produit cartésien.
- Pour le correcteur composite, `components` configure CatBoost/HGB et
  `weight_candidates` énumère les poids à comparer. Une grille spécifique à
  un composant utilise `components.<nom>.<paramètre>`, par exemple
  `components.cat_v1.learning_rate: [0.025, 0.03, 0.04]`.
- Pour MKOnline, le poids est appris sur train avec le pas `weight_step` ; les
  valeurs de clip sont sélectionnées sur validation.
- Pour Kalman, tous les champs de `KalmanResidualConfig` sont disponibles, y
  compris `candidate_kinds` pour activer KF, EKF, UKF et les familles
  fondamentales décrites ci-dessous.

Pour tester une nouvelle série dans le correcteur résiduel, la matérialiser
d'abord dans une archive expérimentale via le contrat PIT existant, puis
utiliser cette archive dans `source_run`. Le correcteur récupère automatiquement
toutes les colonnes de `inputs/aligned_inputs.csv.gz` sauf `target`.

### Météo et fondamentaux du Kalman

Le bloc facultatif `models.kalman.covariates` décrit les entrées brutes, leurs
transformations déterministes et quatre groupes aux unités séparées :
`market`, `weather`, `renewables`, `fundamentals`. Les candidats associés sont
respectivement `linear_market`, `linear_weather`, `linear_renewables` et
`linear_fundamental`. Un candidat ne peut lire que son groupe : température,
GW et rayonnement ne sont donc jamais agrégés dans une moyenne sans unité.

Les nouvelles tables locales sont déclarées dans `additional_sources`. Les
chemins doivent rester sous la racine du projet et viser un fichier
CSV/CSV.GZ/Parquet existant. La direction du mapping est toujours
`alias_du_modele: colonne_du_fichier` :

```yaml
kalman:
  upstream_model: residual_corrected
  training_lookback_days: 365
  covariates:
    input_columns:
      [fr_residual_load_fcst, de_residual_load_fcst,
       be_residual_load_fcst, nl_residual_load_fcst,
       es_residual_load_fcst, fr_temperature_fcst,
       fr_wind_generation_fcst]
    groups:
      market: [fr_residual_load_fcst, de_residual_load_fcst,
               be_residual_load_fcst, nl_residual_load_fcst,
               es_residual_load_fcst]
      weather: [fr_temperature_fcst, fr_heating_degree]
      renewables: [fr_wind_generation_fcst, fr_wind_ramp]
      fundamentals: [fr_temperature_fcst, fr_wind_generation_fcst]
    derived:
      fr_heating_degree:
        kind: heating_degree
        source: fr_temperature_fcst
        threshold: 15.0
      fr_wind_ramp:
        kind: ramp
        source: fr_wind_generation_fcst
        periods: 1
    history_missing_policy: neutral
    minimum_history_coverage: 0.95
    require_future_complete: true
  additional_sources:
    - path: data/pit/kalman_weather_fr.csv.gz
      timestamp_column: delivery_start_utc
      origin_column: run_init_utc
      revision_column: revision_time_utc
      cutoff_column: cutoff_utc
      cutoff_time: "08:00"
      columns:
        fr_temperature_fcst: temperature_c
        fr_wind_generation_fcst: wind_generation_gw
```

`origin_column` est obligatoire : elle désigne l'heure d'émission de la
prévision. `revision_column`, lorsqu'elle existe, porte la dernière révision
connue ; `cutoff_column` doit répéter le cutoff civil déclaré pour chaque
heure de livraison. Le chargeur exige `origin <= cutoff`,
`revision <= cutoff` et un cutoff exactement égal à `D-1 cutoff_time` dans le
fuseau de la zone. Une violation arrête l'entraînement au lieu de neutraliser
la valeur.

Chaque timestamp externe doit porter explicitement son fuseau ou son offset.
Les sources sont alignées par timestamp UTC avec une jointure gauche : le
laboratoire ne supprime aucune heure et n'effectue ni remplissage, ni
interpolation. `minimum_history_coverage` est contrôlé séparément sur train,
validation et test. `require_future_complete: true` refuse l'inférence dès
qu'une feature future manque. La transformation `ramp` repart à zéro au début
de chaque journée locale physique, y compris les journées DST de 23/25 heures.

Avec `history_missing_policy: neutral`, les rares trous historiques restant
admis par le seuil sont neutralisés après standardisation robuste par le
filtre. `complete_trailing` exige à la place un suffixe historique entièrement
fini. Les noms contenant `actual`, `target`, `observed`, `Storm`, `MKOnline` ou
`oracle` sont rejetés pour prévenir les fuites et l'usage d'un concurrent.

### Séries Saturn retenues

Le bundle `Weather` utilise exclusivement des prévisions, jamais des mesures
météo réalisées :

| Variable | Série Saturn FR/DE/BE/NL/ES | Granularité | Traitement |
| --- | --- | --- | --- |
| Température 2 m | `meteo.nrjscan.<zone>.t_2m.index.fcst.d` | quotidienne | valeur du jour répétée sur les 23/24/25 heures physiques |
| Éolien | `power.<zone>.generation.wind.hourly.gw.fcst` | horaire | aucun fill/interpolation |
| Solaire | `power.<zone>.generation.solar.hourly.gw.fcst` | horaire | aucun fill/interpolation |

Pour NL uniquement, l'éolien utilise
`power.nl.prod.total.wind.mw.ecmwf_avg.pointconnect.6h.cache`, converti de MW
en GW (`x 0,001`), car sa timeline PIT couvre correctement les journées DST.
Chaque journée `D` est demandée à Saturn **as-of D-1 08:00 dans le fuseau de
la zone**. Le catalogue JSON/YAML généré dans le répertoire de sortie contient
les chemins, colonnes de provenance, empreintes et snippets
`additional_sources` prêts à copier dans le laboratoire ou la configuration
opérationnelle.

Dans les Parquet, `snapshot_time_utc` et `revision_time_utc` représentent le
cutoff de la requête `Client.get(revision_date=...)`. Saturn ne renvoie pas
l'horodatage d'insertion fournisseur dans cette réponse compacte : le sidecar
d'audit l'indique explicitement. La garantie causale porte donc sur l'état de
la série demandé as-of au cutoff, avec checksum du résultat matérialisé.

### Fallback ECMWF via Open-Meteo Single Runs

Si une série Saturn manque réellement, le fallback expérimental sait
matérialiser un run ECMWF IFS déterministe et reproductible. Pour une livraison
`D`, il fixe le run à `D-2 18:00 UTC`, donc strictement avant le cutoff
`D-1 08:00` local :

```powershell
$env:OPEN_METEO_API_KEY = '<cle-professionnelle>'
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  '.\materialize_open_meteo_weather.py' `
  --start-day 2024-07-01 --end-day 2026-08-29 `
  --zones FR,DE,BE,NL,ES `
  --output 'data\pit\kalman_weather_open_meteo.parquet' `
  --cache-dir 'data\pit\open_meteo_raw_cache' `
  --require-api-key
```

La table contient `delivery_start_utc`, `run_init_utc`, `cutoff_utc` et le
lead time, ainsi que les agrégats multi-points de température, humidité,
vent 100 m, direction du vent, rayonnement, nébulosité, précipitations et
pression. Les réponses JSON brutes et un manifest avec SHA-256 sont conservés.
Il faut utiliser l'API **Single Runs**, et non un historique reconstruit à
partir des premières heures de plusieurs runs. En contexte commercial,
utiliser un abonnement/endpoint Open-Meteo autorisé et une clé ; la clé reste
dans l'environnement et n'est jamais écrite dans les logs ni le manifest.
Voir la documentation officielle [Single Runs](https://open-meteo.com/en/docs/single-runs-api),
les [offres API](https://open-meteo.com/en/pricing) et la
[licence](https://open-meteo.com/en/license) avant activation.

## Artefacts

Chaque expérience produit :

```text
run_manifest.json
resolved_config.json
input_config.yaml
dataset_manifest.json
splits.json
leaderboard.csv
predictions.csv.gz
metrics.csv
metrics_by_day.csv
report.html
checksums.json
models/
  residual_corrector/model_evaluation.joblib + model_full.joblib
  residual_corrector/kalman_upstream_history.csv.gz
  mkonline_blend/model.json
  kalman/model.json + audits d'état
  kalman/kalman_covariate_coverage.csv
  kalman/kalman_final_coefficients.csv
```

Le `model.json`, `dataset_manifest.json` et le leaderboard conservent le
contrat de covariables, les nombres de features par candidat, la couverture
par phase et les SHA-256 des sources externes utilisées à l'entraînement.
Le rapport trace aussi la valeur absolue des coefficients exogènes finaux
standardisés. Ils décrivent une association conditionnelle apprise par le
filtre et ne doivent pas être interprétés comme un effet causal.

Le rapport contient un mode nuit, le leaderboard, les résultats validation et
test, les erreurs journalières et le delta cumulé face au modèle amont.

## Activation dans Forecast.ps1

Le laboratoire ne promeut jamais automatiquement une expérience. Après avoir
gelé la configuration gagnante sur validation et vérifié son test FINAL365 :

1. recopier les `filter_parameters`, `covariates`, groupes et transformations
   retenus dans `config/kalman_operational.yaml` ;
2. ajouter chaque fichier PIT dans `additional_sources` avec la même convention
   `alias_du_modele: colonne_du_fichier` ;
3. calculer son empreinte avant activation :

```powershell
(Get-FileHash `
  'data\pit\kalman_weather_fr.csv.gz' `
  -Algorithm SHA256).Hash.ToLowerInvariant()
```

4. renseigner `information_type: day_ahead_forecast`, une `cutoff_policy`
   explicite (par exemple `latest vintage available before D-1 08:00
   Europe/Paris`) et le SHA-256 ;
5. ajouter les candidats concernés à `filter_parameters.candidate_kinds`, puis
   lancer :

```powershell
& '.\Forecast.ps1' -Action Run `
  -Countries FR,DE,BE,NL,ES -Mode All `
  -KalmanConfig 'config\kalman_operational.yaml'
```

Le préflight est exécuté avant les runners. Une clé inconnue, une empreinte
différente, une colonne concurrente/réalisée, un timestamp naïf, une couverture
future incomplète ou une transformation invalide arrête le batch. Les fichiers
externes sont joints uniquement à la copie jetable de l'export Kalman ; le
contrat live scellé de Chronos-2 et `runs/live` restent inchangés. Les exports
Kalman publient `kalman_filter_audit.json` et
`kalman_operational_sidecar_audit.json` à côté du CSV et du rapport HTML.

## Évaluation et comparaison sans réentraînement

```powershell
& '.\FineTune.ps1' -Action Evaluate `
  -RunDirectory 'runs\experiments\auxiliary_lab\auxiliary_models_fr_20260828_v1'

& '.\FineTune.ps1' -Action Compare `
  -Runs @('runs\experiments\auxiliary_lab\essai_a', 'runs\experiments\auxiliary_lab\essai_b') `
  -Output 'runs\experiments\auxiliary_lab\comparaison_a_b'
```

## Chargement et inférence

```powershell
& '.\FineTune.ps1' -Action Predict `
  -Artifact 'runs\experiments\auxiliary_lab\essai_a\models\residual_corrector' `
  -SourceRun 'runs\live\fr_day_ahead_2026-08-29' `
  -Output 'runs\experiments\auxiliary_lab\predictions\fr_2026-08-29.csv'
```

Le correcteur joblib n'est chargé qu'après vérification de son SHA-256. Les
artefacts MKOnline et Kalman sont des recettes JSON ; Kalman rejoue son état
causalement à partir de l'historique fourni.

## Extension

La logique commune (splits, métriques, artefacts et rapport) se trouve dans
`auxiliary_lab`. Un futur adapter peut réutiliser ce contrat pour LEAR,
CatBoost horaire, topology, Foundation-MoE ou le correcteur des inputs de
charge résiduelle sans ajouter de logique au pipeline live.
