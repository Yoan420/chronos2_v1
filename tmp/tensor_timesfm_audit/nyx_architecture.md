# Audit de la chaîne NYX effectivement utilisée

Audit local du 15 septembre 2026, effectué sans réseau, collecte, entraînement, lancement scientifique ni modification de production. Cette note distingue le parcours utilisé par l'application de sa qualification scientifique. Elle ne conclut pas à un gain prédictif : les mesures d'erreurs et l'audit Tensor-TimesFM font l'objet des autres volets.

## 1. Référence à conserver : NYX complet, pas Chronos-2 seul

**Fait vérifié.** Le bouton NYX utilise `NuclearKalman.ps1`, qui exécute successivement les pays **BE, DE, FR, NL** et publie `nuclear_kalman`. Le modèle composite est :

```text
Prix historiques de chaque pays + calendriers
5 prévisions de charge résiduelle (FR, DE, BE, NL, ES), en GW
Prévision de production nucléaire FR, en GW
       │ sélection aux cutoffs civils + snapshots immuables
       ▼
Chronos-2 local, non affiné, contexte 2 048 heures
       │ q10 / q50 / q90
       ▼
Correcteur résiduel CatBoost, refit quotidien sur D−365 … D−1
       │ residual_corrected
       ▼
Kalman gouverné, refit sur D−365 … D−1
       │ residual_kalman = sortie NYX
       ▼
Exports horaires par pays + rapport CWE Model / Storm

Storm officiel + observations réactualisées ──► comparaison/reporting
```

La présence de répertoires `experiments`, de mots « challenger » dans les commentaires ou de `production=false` dans les audits **ne signifie pas que l'interface lance un autre modèle**. Le parcours nucléaire est bien celui de NYX aujourd'hui. Ces marqueurs signifient qu'il ne modifie pas les anciennes archives opérationnelles et qu'une publication ne vaut pas validation prospective. Le rapport le précise explicitement : [nuclear_reporting.py, `_diagnostic_banner`](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_reporting.py:177).

### Traçage de l'interface jusqu'au calcul

| Frontière | Comportement vérifié | Référence |
|---|---|---|
| Interface | Date de lancement séparée de la date de consultation ; envoie `{delivery_day}` et une clé d'idempotence | [app.js](C:/Users/BQ6757/chronos2_v1/experiment_console/static/app.js:108) |
| HTTP | `POST /api/primary-run` n'accepte que la date optionnelle | [server.py](C:/Users/BQ6757/chronos2_v1/experiment_console/server.py:453) |
| Manager | Enregistrement durable, réutilisation d'un lancement identique actif, conflit si date/mode diffère ; date explicite conservée au dispatch | [manager.py, `launch_primary_run`](C:/Users/BQ6757/chronos2_v1/experiment_console/manager.py:214), [`_tick`](C:/Users/BQ6757/chronos2_v1/experiment_console/manager.py:310) |
| Adaptateur | Commande fixe PowerShell `-NoProfile -File NuclearKalman.ps1 -NoOpen`, et `-DeliveryDay` si date explicite ; valeurs scientifiques dérivées du vrai `-DryRun` | [adapters.py, `_primary_command`](C:/Users/BQ6757/chronos2_v1/experiment_console/adapters.py:81) |
| Worker | Processus supervisé avec `shell=False` | [worker.py](C:/Users/BQ6757/chronos2_v1/experiment_console/worker.py:39) |
| PS1 | Quatre pays, demain à Paris par défaut, device auto, threads/workers 4, configuration nucléaire ; attribution désactivée, synchronisation activée | [NuclearKalman.ps1](C:/Users/BQ6757/chronos2_v1/NuclearKalman.ps1:21) |
| Batch | Une synchronisation commune, puis `run_nuclear_forecast.py --stage Run --report-variants kalman` par pays, puis rapport CWE | [run_nuclear_kalman.py, `build_nuclear_kalman_plan`](C:/Users/BQ6757/chronos2_v1/run_nuclear_kalman.py:63) |
| Calcul ou réutilisation | Un résultat déjà figé est rechargé ; sinon calcul puis gel. Une actualisation de rapport ne réentraîne pas ce résultat | [run_nuclear_forecast.py](C:/Users/BQ6757/chronos2_v1/run_nuclear_forecast.py:634) |
| Publication | `nuclear_kalman` correspond exactement à `result.kalman_view.forecast`, vérification des heures et quantiles, manifeste avec SHA | [nuclear_exports.py, `publish_nuclear_exports`](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_exports.py:130) |

**Conséquence pour l'expérience.** Comparer Tensor-TimesFM ou un complément au `residual_kalman` du même snapshot, et conserver Chronos seul puis `residual_corrected` comme ablations. Les noms identiques `residual_kalman` d'autres chemins historiques ne suffisent pas à établir une même référence : il faut aussi l'engine `nuclear_forecast_v1`, le protocole `civil_pit_v2`, la zone, la livraison et l'identité des sources.

## 2. Modèles et entrées réellement consommées

### Chronos-2

- Modèle `amazon/chronos-2`, contexte **2 048 heures physiques**, poids locaux, pas de fine-tuning dans ce parcours. Le code refuse une autre base et les colonnes contenant Storm/MKOnline : [nuclear_forecast.py, `run_nuclear_forecast`](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_forecast.py:408).
- Chargement par `Chronos2Pipeline.from_pretrained` : [forecasting.py, `load_model`](C:/Users/BQ6757/chronos2_v1/chronos2_modular/forecasting.py:27). Les appels effectifs utilisent `predict_df(..., target='target', prediction_length=horizon, cross_learning=False)` : [forecasting.py](C:/Users/BQ6757/chronos2_v1/chronos2_modular/forecasting.py:397), [prévision future](C:/Users/BQ6757/chronos2_v1/chronos2_modular/forecasting.py:504).
- Il s'agit de quatre cibles pays traitées séparément, pas d'un apprentissage conjoint des quatre prix dans le batch. Le partage vient des covariables communes et des poids de fondation. `cross_learning=False` ne signifie pas absence d'attention aux covariables à l'intérieur d'une tâche Chronos.
- Les configurations ordinaires ont une ancienne covariable REMIT nucléaire désactivée ; le snapshot nucléaire remplace ce schéma par **`fr_nuclear_generation_fcst_gw`**, série `power.fr.generation.nuclear.gw.fcst`, unité GW, activée dans le contexte et le futur. Ce n'est ni la génération observée, ni la disponibilité Pmax, ni la série REMIT MW : [run_nuclear_forecast.py, `snapshot_config`](C:/Users/BQ6757/chronos2_v1/run_nuclear_forecast.py:461), [nuclear_forecast.py, `_covariates`](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_forecast.py:245).
- Les cinq charges résiduelles FR/DE/BE/NL/ES sont disponibles dans chaque pays ; ES est ici une covariable, **pas un cinquième pays lancé depuis NYX**. La configuration et le moteur permettent ES ailleurs, mais le PS1/API primaire sont limités aux quatre pays CWE.

### Résiduel CatBoost

Recette commune : **700 arbres, profondeur 6, learning rate 0,03, L2 15, minimum 720 lignes, seed 42**, correction plafonnée à **±40 EUR/MWh** : [configuration FR](C:/Users/BQ6757/chronos2_v1/chronos2_hourly_fr_residual_v1.yaml:79) ; mêmes valeurs dans les trois configurations `*_residual_candidate_v1.yaml`.

Le correcteur apprend **`actual − Chronos_q50`**, à partir des variables autorisées et des quantiles Chronos. Calendrier riche et profils journaliers sont présents ; les variables de prix historiques et jour-de-l'année sont exclues de cette recette résiduelle. Chronos lui-même garde bien son contexte de prix historiques. Voir [ResidualCorrector.fit](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/models/residual_corrector.py:1115), [ResidualMetaFeatureBuilder](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/models/residual_corrector.py:341).

Chaque livraison est réajustée sur les **365 jours civils strictement antérieurs**, sans reprendre un ancien préfixe résiduel d'un autre modèle. Les premières journées trop courtes restent explicitement l'identité Chronos : [causal_residual_replay](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_forecast.py:272).

### Kalman gouverné

Il reçoit l'amont `residual_corrected`, les cinq charges résiduelles, leur moyenne/dispersion et le nucléaire français. La branche nucléaire impose couverture historique complète et futur complet, même si les paramètres génériques de la configuration autorisent ailleurs une politique neutre : [nuclear_kalman_covariate_config](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_forecast.py:200).

Les candidats actifs sont `linear_bias`, `linear_harmonic`, `linear_market`, `linear_scale`, `ukf_scale`. Il n'y a ni smoother ni EM. Le refit sélectionne D−365 … D−1 ; l'état est figé pour la livraison, dont les observations ne sont pas assimilées : [kalman_residual.py](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/kalman_residual.py:1), [`_rolling_training_frame`](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/kalman_residual.py:1439), [`_fit_rolling_target_day`](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/kalman_residual.py:1474).

Le gouverneur compare la MAE passée sur au plus **60 jours**, minimum **14**, recherche un poids par pas de **0,05**, et exige un gain supérieur à `max(0,05 EUR/MWh, 0,5 % de la MAE de base)` ; sinon identité. La correction brute est plafonnée à **±20 EUR/MWh** : [`_governance_choice`](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/kalman_residual.py:1287), [application](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/kalman_residual.py:1583), [configuration](C:/Users/BQ6757/chronos2_v1/config/kalman_operational.yaml:51).

**Conséquence probabiliste vérifiée.** CatBoost et Kalman déplacent tous deux q10/q50/q90 par le même montant. Ils préservent ordre et largeur des intervalles ; ils ne recalibrent pas leur couverture. Une sortie P10–P90 ne prouve donc pas une couverture empirique de 80 %. Voir [apply_residual_correction](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/models/residual_corrector.py:286) et [application Kalman](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/kalman_residual.py:1614). Les limites ±40 puis ±20 peuvent être pertinentes face à un très grand raté de Chronos ; leur responsabilité doit être mesurée, pas supposée.

## 3. Contrat temporel : contrôle causal réel, preuve PIT limitée

| Élément | Fait vérifié | Portée / limite |
|---|---|---|
| Horizon | Un jour local complet, fréquence horaire, 23/24/25 heures selon DST | `horizon:24` dans le YAML est nominal ; les plans passent le nombre physique réel au modèle |
| Origine | **D−1 à 08:00 heure de Paris** ; le lancement refuse une livraison dont le cutoff n'est pas encore atteint | Date civile puis localisation, pas soustraction naïve de 24 heures UTC |
| Covariables | Dernière ligne dont snapshot **et** révision sont ≤ `min(cutoff de sa livraison, runtime_as_of)` | La dernière valeur manquante ne peut être masquée par une ancienne valeur valide |
| Futur | Six covariables de forecast intégralement finies sur la livraison | Une stratégie appelée `oracle` représente ici une **prévision connue au cutoff**, pas une réalisation future |
| Cible historique | Cache horaire canonique complet, contexte jusqu'à la fin de D−1, jamais prix de D pour prévoir D | Historique révisé au moment du gel ; aucune preuve de publication initiale de chaque prix dans ce chemin |
| Révisions fournisseur | `provider_revision_timestamp_available=false`, `production_pit_evidence=false` | Les horodatages représentent l'interrogation Saturn « as-of », pas les dates d'insertion retournées par le fournisseur |

Références : [run_nuclear_forecast.delivery_date](C:/Users/BQ6757/chronos2_v1/run_nuclear_forecast.py:277), [chronos_adapter.build_delivery_plan](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/chronos_adapter.py:172), [hourly_contract.local_delivery_day_index](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/hourly_contract.py:358), [nuclear_preparation._civil_cutoffs / _strict_selection](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_preparation.py:51), [contrôle de complétude](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_preparation.py:119), [limite de preuve de publication](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_sources.py:145).

**Nuance importante sur les observations.** Au premier lancement d'une livraison non figée, les observations récemment extraites réactualisent la copie de contexte historique antérieure à D (`latest.combine_first(prices)`), puis cette copie est gelée. Un résultat déjà figé garde ses anciennes entrées, même si les rapports reçoivent des observations révisées : [refreshed_target_snapshot](C:/Users/BQ6757/chronos2_v1/run_nuclear_forecast.py:543), [appel conditionnel](C:/Users/BQ6757/chronos2_v1/run_nuclear_forecast.py:634). Les prix day-ahead de D−1 peuvent être connus avant leur livraison physique ; leur utilisation jusqu'à la fin de D−1 n'est donc pas, à elle seule, une fuite. En revanche, un historique révisé extrait aujourd'hui ne démontre pas ses valeurs exactes disponibles à chaque ancien cutoff.

**DST : deux couches distinctes.** Les cibles et sorties conservent les heures physiques UTC, offsets et `fold` ; aucune moyenne 24 heures forcée. Les sources fondamentales ont des conventions documentées : duplication possible du singleton automnal 02:00 ; le nucléaire ne fournit pas le nombre exact de réparations ni une preuve indépendante de la seconde occurrence. La banque de charge résiduelle admet en plus une réparation linéaire extrêmement ciblée des heures locales 04:00 et 06:00 du printemps NL, avec registre vérifié. Le sidecar courant indique **4 réparations NL** et zéro pour les quatre autres charges ; ce n'est pas une politique générale d'imputation. Références : [nuclear_sources.py](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_sources.py:285), [validateur du registre](C:/Users/BQ6757/chronos2_v1/materialize_saturn_kalman_fuel.py:1085), [sidecar courant](C:/Users/BQ6757/chronos2_v1/data/pit/nuclear_forecast/residual_load_market_features.parquet.audit.json:70).

## 4. Fenêtres, données présentes et qualification des résultats

Le moteur matérialise **730 jours** d'historique brut/corrigé et **365 jours de backtest Kalman** avant D. Les 365 premiers jours servent de warm-up diagnostique, pas de jeu de validation imbriqué complet. Le mode incrémental garde un ancrage de warm-up fixe, peut conserver jusqu'à **1 095 jours bruts**, et réutilise uniquement les jours dont l'identité demeure valide. Il ne raccourcit pas les refits scientifiques : [nuclear_forecast.py, constantes](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_forecast.py:43), [préparation incrémentale](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_forecast.py:497), [appel Kalman et audit](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_forecast.py:640).

Les métadonnées des résultats figés **15 et 16 septembre 2026** ont été lues pour les quatre pays. Elles confirment toutes : engine nucléaire, mode incrémental, LoRA/Storm/MKOnline absents des entrées, refit résiduel quotidien et lookback Kalman 365. Pour le **16**, la fenêtre évaluée est **2025-09-16 à 2026-09-15**. Le checkpoint figé est `amazon/chronos-2` révision **`29ec3766d36d6f73f0696f85560a422f50e8498c`**. Exemple vérifiable : [resolved_config.yaml FR](C:/Users/BQ6757/chronos2_v1/runs/experiments/nuclear_forecast_v1/2026-09-16/fr/civil_pit_v2/resolved_config.yaml:11), [audits.json FR, objet result et kalman_replay](C:/Users/BQ6757/chronos2_v1/runs/experiments/nuclear_forecast_v1/2026-09-16/fr/civil_pit_v2/report_only/frozen_result/audits.json:1).

Les audits de préparation du 16 annoncent **17 712 heures par entrée en FR/NL** et **17 688 en BE/DE**, six entrées, sans trou ; différence due aux ancrages de cache FR/NL au 2024-09-09 et BE/DE au 2024-09-10. Il faut comparer sur une fenêtre commune, pas déduire le nombre d'exemples à partir de fichiers d'âges différents. Exemple : [pit_selection_audit FR](C:/Users/BQ6757/chronos2_v1/runs/experiments/nuclear_forecast_v1/2026-09-16/fr/civil_pit_v2/prepared/pit_selection_audit.json:5). Cette note a contrôlé les métadonnées ; elle ne prétend pas avoir revérifié ici tous les octets des huit bundles.

**Les fenêtres de présentation diffèrent.** FINAL365 exclut D ; la section Statistics utilise les 365 derniers jours observés, avec D vide tant que son observation est absente. Les rapports actualisés peuvent employer des prix révisés vérifiés et les comparateurs Storm uniquement sur les heures communes. Ne pas extraire les chiffres d'une table HTML en présumant que sa fenêtre est celle de `kalman_backtest.parquet` : [nuclear_reporting.render_nuclear_reports](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_reporting.py:254), [contrat Statistics](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_reporting.py:487).

Un fichier CWE existe parfois après un échec de pays : le batch assemble les résultats disponibles mais garde le statut failed. L'existence HTML n'est donc pas la preuve d'un run entièrement réussi : [execute_nuclear_kalman_plan](C:/Users/BQ6757/chronos2_v1/run_nuclear_kalman.py:248). Les rapports historiques 13/14 extraits du backtest d'un run ultérieur sont signalés `history_from_delivery` ; ils ne constituent pas de nouvelles prévisions historiquement émises : [model_storm_data._load_model](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/model_storm_data.py:76).

## 5. Variantes à ne pas confondre

| Élément | Statut observé | Conséquence |
|---|---|---|
| NYX principal | NuclearKalman, `nuclear_forecast_v1`, `civil_pit_v2`, `nuclear_kalman` | Référence principale complète |
| `nuclear_autonomous` | Même amont nucléaire, sortie avant filtre `residual_corrected` | Ablation, pas NYX final |
| LoRA exogène | `enabled_modes: []` pour tous les pays ; le préflight nucléaire refuse une activation | Aucun poids LoRA dans NYX actuel |
| `nuclear_cwe_v1` | Challenger isolé ; ajoute disponibilité nucléaire Pmax BE/NL en GW, diffusée du quotidien vers l'horaire | Ce n'est ni le rapport CWE, ni le signal nucléaire FR actuel |
| `nuclear_kalman_extreme_v1` | Complément expérimental distinct ; minimum historique 90 jours, gouvernance et expert propres | Ne pas mélanger ses erreurs avec celles du NYX final standard |
| Modèles ordinaires/anciens, blends MKOnline | Autres commandes, configurations et exports | Le bouton primaire ne les lance pas |

Références : [activation LoRA](C:/Users/BQ6757/chronos2_v1/config/chronos2_exogenous_activation_v1.yaml:23), [check_lora_inactive](C:/Users/BQ6757/chronos2_v1/run_nuclear_forecast.py:369), [nuclear_cwe_forecast.py](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_cwe_forecast.py:1), [config/nuclear_cwe.yaml](C:/Users/BQ6757/chronos2_v1/config/nuclear_cwe.yaml:1), [config/nuclear_kalman_extreme.yaml](C:/Users/BQ6757/chronos2_v1/config/nuclear_kalman_extreme.yaml:1).

## 6. Points d'intégration et expériences à privilégier

Les éléments suivants sont des **propositions**, pas des caractéristiques de Tensor-TimesFM ni des gains mesurés.

1. **Commencer hors ligne après NYX final.** Lire les bundles avec [load_nuclear_result_bundle](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_run_archive.py:203). Aligner quatre colonnes d'erreurs `actual − residual_kalman__q50` sur les mêmes timestamps UTC, snapshots et dates ; stocker les identités, erreurs, covariables et masques dans un nouveau répertoire de recherche. Les résidus des sorties brutes/résiduelles peuvent servir aux ablations, jamais à fabriquer un historique NYX final avant son existence.
2. **Tester une correction multivariée simple avant un décodeur neuronal.** Références minimales : identité, biais régularisé par heure/pays, régression ridge des résidus passés avec les six fondamentaux connus, PCA de rang faible ajustée sur le passé puis prévision des facteurs. Avec seulement quatre séries de prix, la réduction inter-pays n'est pas nécessairement un problème de grande dimension. Un tenseur jours × pays × heure crée au plus 96 cellules nominales par jour ; le nombre de cellules n'augmente pas le nombre de jours indépendants disponibles.
3. **Ne pas perdre DST en remodelant le tenseur.** Un premier panneau UTC `[heure physique, pays]` avec heure locale, offset et fold explicites est plus sûr. Une version `[jour, pays, heure_locale, fold]` nécessite masque de présence structurelle et distingue absence de slot de valeur manquante ; les journées ne doivent pas être comprimées ou complétées artificiellement à 24 observations. Calculer les pertes uniquement sur les heures réellement présentes, pas sur des zéros de remplissage.
4. **Séparer unités et disponibilité.** Prix/résidus en EUR/MWh, fondamentaux en GW, calendriers à part ; normalisation ajustée sur le train uniquement, conservant signe et événements extrêmes. Les séries de demande/éolien/solaire individuelles, flux, marges, combustibles ou contraintes réseau ne sont pas démontrées comme entrées futures complètes par cet audit. NYX utilise les charges résiduelles du fournisseur ; un module fondamental supplémentaire doit d'abord obtenir ses vintages et sa complétude au cutoff, sans substituer les réalisations futures.
5. **Adaptation Chronos des facteurs : nouveau protocole d'expérience.** Les frontières d'exécution injectables existent dans [nuclear_forecast.run_nuclear_forecast](C:/Users/BQ6757/chronos2_v1/chronos2_hourly/nuclear_forecast.py:408), mais le moteur valide explicitement `amazon/chronos-2`, le schéma nucléaire et les horizons. Un nouveau latent ne doit pas être déguisé en la même recette ni réutiliser son cache. Préférer un exécuteur concurrent isolé qui réutilise la construction des plans, la sélection des sources et les vérifications horaires.
6. **Éviter une intervention unique au milieu de la chaîne sans refit.** Modifier l'amont Chronos ou les features du résiduel change les distributions d'erreurs vues par CatBoost et Kalman. Une telle adaptation doit refaire causalement les couches dépendantes ; ajouter un complément final permet d'abord une expérience peu coûteuse sans cela. Si le complément ajoute seulement un décalage, les intervalles hérités restent non recalibrés ; calibrer séparément sur erreurs strictement antérieures et rapporter couverture/largeur/WIS ou pinball.
7. **Facteurs stables et test final.** Figer le décodeur à l'intérieur de chaque bloc de validation ; si réentraînement, résoudre permutation/signe/rotation avec un alignement calculé uniquement sur le chevauchement passé. Ne pas concaténer sans contrôle des coordonnées latentes apprises avec des bases différentes. Une période finale séparée et une étape prospective sont indispensables pour décider, vu les prix révisés et le caractère rétrospectif des preuves as-of.

**Hypothèses à réfuter.** (a) Les erreurs finales comportent un facteur commun encore prévisible malgré le Kalman marché ; mesurer dépendance puis gains hors échantillon. (b) Une compression ne dégrade pas les pics, prix négatifs et spreads entre pays ; mesurer MAE/RMSE/biais et scores probabilistes dans ces régimes séparément. (c) Une amélioration sur résidus n'est pas simplement une moyenne qui réduit quelques erreurs ordinaires au détriment des extrêmes. Une corrélation contemporaine des erreurs ne prouve pas leur prévisibilité au cutoff.

## 7. Incertitudes et reproductibilité

- Pas de preuve ici que les révisions fournisseur étaient capturées en temps réel aux anciens cutoffs ; l'audit le nie explicitement. Les observations de backtest peuvent être plus récentes que leur date de livraison.
- Il existe 365 jours de sorties finales par bundle, pas 730 jours de sorties Kalman finales. Un autre modèle ne doit pas recevoir les 365 jours évalués pour apprendre factorisation ou combinaison avant d'être évalué sur ces mêmes jours.
- Réentraînements des nouveaux composants, nombre de jours réellement nouveaux en cache, et disponibilité GPU doivent être mesurés. Le build Torch installé est `2.13.0+xpu`, mais `resolve_device('auto')` dans ce code sélectionne uniquement CUDA ou CPU : [common.py](C:/Users/BQ6757/chronos2_v1/chronos2_modular/common.py:380). Aucun GPU Intel n'est sélectionné automatiquement par ce chemin.
- Versions de l'environnement configuré lues via métadonnées : chronos-forecasting 2.3.1, torch 2.13.0+xpu, transformers 5.14.1, catboost 1.2.10, pykalman 0.11.2, pandas 2.3.3. Ce sont les versions installées lors de l'audit, pas une preuve qu'elles étaient identiques pour chaque archive historique.
- HEAD local : `900d8fc36a5f7d44e91f895786e49e4ed7614086`. Le dépôt comporte du travail local ; le commit seul ne décrit pas le code exécuté. Empreintes SHA256 lors de cette lecture :

```text
run_nuclear_forecast.py                    8f92b592e347435325ab38ebd489baa8c4a9b06a9b14d2ec3942666331338ea0
chronos2_hourly/nuclear_forecast.py         04d64754ba13a08ca5e1f6a22695a4a43ab934f4b01d13991269f4ad98ac54c4
chronos2_hourly/kalman_residual.py          62566e49ce50e491b6070f237ffe5a3c8dbb0e2c745208482c3414df5a58abf8
chronos2_hourly/models/residual_corrector.py 5c47036f2ac621215038b8b6451f8fa85d1de486fb356447cc2b90c3ce4a19f9
config/nuclear_forecast.yaml              e7f47b29a84df0220d7fcc44fa544b78d456592d63bedff41818253bf572d851
```

La lecture a porté sur le code, les configurations publiques pertinentes et les champs d'audit nécessaires. Aucun secret ni contenu d'authentification n'est reproduit.
