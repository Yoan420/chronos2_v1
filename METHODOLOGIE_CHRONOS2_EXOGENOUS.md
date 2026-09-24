# Chronos-2 exogene — methode du POC et chemin de production

## Objectif

Le challenger `chronos2_exogenous_lora` apprend a utiliser les fondamentaux
**dans Chronos-2**, avant le correcteur residuel. Son entraînement, son
backtest et sa gouvernance restent isolés du pipeline live. Un raccord
opérationnel fail-closed est toutefois prêt dans `Forecast.ps1` : après
promotion et activation explicites, `-Mode All` publie exactement
`autonomous` = Chronos-2 + LoRA + correcteur résiduel et `kalman` = la même
base suivie du Kalman standard. Pour `-Action Run`, `Both` produit aussi ces
deux vues, avec l'incumbent si LoRA est inactif ou LoRA promu et activé.
Le blend MKOnline reste dans `Blend`, uniquement pour FR et NL ;
`kalman_weather` et `kalman_hybrid` restent expérimentaux hors de `All`.

Tant qu'aucun bundle final ne fournit les preuves PIT, rolling-365 et shadow
requises, la configuration reste inactive et `-Mode All` refuse le lancement.
Le simple fait d'entraîner un checkpoint ou d'obtenir une bonne métrique ne
remplace jamais une promotion gouvernée.

Cette architecture reprend l'idee commune a [ChronosX (AISTATS 2025)](https://proceedings.mlr.press/v258/arango25a.html) et [TimeXer (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/0113ef4642264adc2e6924a3cbbdf532-Paper-Conference.pdf) : faire interagir la serie cible avec des variables exogenes en amont, au lieu de demander a un filtre aval de recuperer tout le signal manquant. Le POC utilise toutefois l'interface native de covariables de Chronos-2 et un adaptateur LoRA ; il ne pretend pas reproduire exactement l'architecture de ces articles.

## Architecture

```text
sources PIT -> banque exogene -> panel par origine D-1 08:00
                                   |          |
                                   |          +-> holdout final 365 jours, gele
                                   +-> fit LoRA 365 jours
                                                |
                         Chronos-2 + adaptateur exogene gele
                                                |
                 backtest apparie au baseline -> gates -> shadow -> production
```

La separation des consommateurs est explicite : une famille peut etre envoyee a `chronos`, `residual` ou `kalman`. Le POC route par defaut les nouvelles variables uniquement vers Chronos-2. Cela evite de compter deux fois le meme signal dans le correcteur residuel ou dans un Kalman.

Tous les packs incluent aussi le meme calendrier deterministe connu a l'avance : sinus/cosinus de l'heure, du jour de semaine et du jour de l'annee, ainsi que l'indicateur de week-end. Ces variables ne dependent d'aucune publication externe, suivent le calendrier local et couvrent correctement les changements DST.

## Variables et ablations

La banque accepte cinq packs reproductibles :

| Pack | Variables |
|---|---|
| `residual_only` | charge residuelle prevue FR, DE, BE, NL, ES |
| `residual_weather` | precedent + temperature, production eolienne et solaire prevues de la zone |
| `residual_fuel` | charge residuelle + TTF, EUA, variations 1/5 jours, volatilite et cout marginal CCGT |
| `residual_flowbased` | charge residuelle + resume CNEC/RAM CORE |
| `full` | charge residuelle + meteo + combustibles/CO2 + CNEC/RAM |

Le resume flow-based conserve notamment disponibilite de la publication, nombre de CNEC, quantiles de RAM, faible RAM, ratio RAM/Fmax, dispersion PTDF entre la France et ses voisins, stress RAM et concentration du stress. Des indicateurs de disponibilite, age, couverture et imputation accompagnent les valeurs ; une absence n'est jamais masquee par une interpolation silencieuse. Les heures CNEC manquantes dans un horizon connu sont neutralisees par une valeur causale explicite et un drapeau d'imputation, de sorte que les futurs connus restent finis. Les heures de contexte anterieures au debut de la couverture JAO peuvent rester `NaN` ; Chronos-2 les traite comme valeurs historiques manquantes et elles ne sont pas soumises au controle `require_complete_known_future`.

Les ablations doivent comparer au minimum :

1. Chronos-2 actuel, sans nouvel adaptateur ;
2. `residual_only`, qui controle l'effet du fine-tuning seul ;
3. `residual_weather` ;
4. `residual_fuel` ;
5. `residual_flowbased` ;
6. `full`.

Le rôle de chaque expérience est pré-déclaré dans son YAML et recopié dans
les manifestes :

- `full` est l'unique `primary_predeclared`. Son holdout de 365 jours est le
  test final utilisable par la gouvernance ;
- les quatre packs `residual_*` sont `diagnostic_only`. Ils quantifient les
  familles de signal, mais une gouvernance ou une promotion les refuse
  explicitement.

Cette séparation interdit de choisir après coup l'ablation ayant obtenu le
meilleur score sur le holdout final. Changer de candidat primaire exige une
nouvelle campagne pré-déclarée et un nouveau holdout encore fermé.

Chaque pack possede son YAML et son repertoire de sortie propres. `Exogenous.ps1 -Pack <nom>` selectionne automatiquement le YAML correspondant lorsque `-Config` n'est pas fourni. Un `-Config` explicite reste prioritaire. La liste `data.known_future_covariates` de chaque YAML correspond exactement aux colonnes du pack, calendrier deterministe inclus.

## Contrat point-in-time

Pour une livraison civile `D`, toutes les variables sont vues telles qu'elles etaient disponibles a l'origine **D-1 08:00 Europe/Paris**. Chaque ligne porte `feature_available_at_utc` et le validateur impose :

```text
feature_available_at_utc <= origin_timestamp
```

Storm, MKOnline, prix futurs observes et toute variable nommee comme un oracle sont interdits dans les inputs et dans la selection du challenger. Les prix observes sont charges depuis le cache cible canonique exact de chaque zone. Son chemin `data/cache/<zone>/target__<identite>.csv.gz` est recalcule a partir du `SeriesSpec` du `base_config` reference par le contrat live ; le materialiseur ne choisit jamais le fichier le plus recent parmi plusieurs series. Un override vers une autre identite est refuse. Les prix servent uniquement de cible passee dans le contexte et, apres emission du forecast, de verite terrain pour l'evaluation.

Le panel et ses sources sont scelles par SHA-256. Le manifeste enregistre le modele/revision, les splits, le schema, les hyperparametres, le checkpoint et les preuves PIT. Il enregistre aussi `base_model_snapshot_sha256`, l'empreinte du snapshot Chronos-2 local exact utilise pour le fit. Au rechargement, cette empreinte est recalculee : un adaptateur ne peut donc pas etre execute avec un autre snapshot de base portant simplement le meme nom.

## Entrainement LoRA et split 365 + 365

Le protocole contient deux periodes consecutives :

- **365 jours de fit**, dont les 30 derniers jours sont reserves a la validation interne ;
- **365 jours de holdout final**, jamais utilises pour le fit, le choix des variables ou des hyperparametres.

Chaque exemple correspond a une vraie origine day-ahead. Le contexte contient par defaut 2 048 heures physiques ; l'horizon est le jour civil suivant. LoRA adapte une petite partie des poids d'attention et de sortie de Chronos-2, tout en conservant le socle gele. `peft` est obligatoire : si la dependance manque ou si aucun `adapter_config.json` n'est produit, l'entrainement echoue au lieu de basculer silencieusement vers un fine-tuning complet.

La reserve finale doit rester la meme pour toutes les ablations. Toute recherche d'hyperparametres se fait exclusivement dans la fenetre de fit/validation.

### Calibration OOF du correcteur résiduel LoRA

Le correcteur résiduel final ne peut pas être entraîné sur des prédictions du
checkpoint LoRA final calculées sur les jours qui ont servi à ce checkpoint :
ce ne serait pas de l'OOF. La calibration séparée ajoute donc 365 jours
d'amorçage en amont du split existant. Sur chacun des blocs de 30 jours de la
fenêtre de calibration, elle réentraîne la même architecture LoRA avec le même
snapshot, schéma et les mêmes hyperparamètres, en utilisant seulement les 365
origines strictement antérieures. Elle prédit ensuite le bloc, scelle le
checkpoint et les prédictions, puis avance dans le temps.

```text
365 jours amorçage -> 365 jours LoRA OOF -> 365 jours holdout final fermé
       |                    |
       +-- lookback glissant+--> fit du correcteur résiduel
```

Les jours de 23/25 heures restent dans les prédictions OOF ; seuls les exemples
de fit LoRA à horizon fixe excluent ces jours, conformément au trainer actuel.
Le cache est atomique et reprenable fold par fold. Le pipeline échoue avant de
charger Chronos-2 si le panel ne fournit pas les 1 095 origines nécessaires et
indique le nombre exact de jours manquants. Cette preuve technique ne requalifie
pas les backfills Saturn/JAO en données PIT de production et n'autorise aucune
promotion.

### Temps de calcul CPU mesure

Sur le poste actuel, le smoke reel donne environ **67 secondes par step LoRA**, auxquels s'ajoutent environ **5 minutes de validation**. La configuration standard a 500 steps represente donc environ **9 a 10 heures d'entrainement CPU**, hors backtest rolling-365. Cette valeur est un ordre de grandeur mesure, pas un engagement de duree : charge machine, taille du pack, contexte et device peuvent la faire varier. Pour verifier la chaine avant un run long, utiliser `-DryRun`, puis une configuration smoke distincte ; ne pas reduire silencieusement `num_steps` dans le YAML gouverne.

## DST

Les jours civils europeens contiennent 23, 24 ou 25 heures. Le panel conserve exactement ces heures physiques :

- le trainer LoRA a longueur fixe utilise seulement les jours de 24 heures pour le fit et la validation ;
- le backtest final conserve les **365 jours**, y compris les horizons de 23 et 25 heures ;
- aucune duplication, suppression ou interpolation d'heure DST n'est autorisee ;
- les metriques globales sont horaires, et les controles journaliers regroupent les heures selon `Europe/Paris`.

## Evaluation et gates

Le CSV rolling-365 apparie contient, pour chaque heure, l'origine, le prix observe, les quantiles `q10/q50/q90` du baseline et ceux du challenger. La decision n'est pas fondee sur la seule MAE globale. La politique verifie aussi :

- gain MAE absolu et relatif ;
- taux de jours gagnes et gain positif dans les deux moities chronologiques ;
- borne basse d'un bootstrap apparie par blocs de sept jours ;
- non-degradation excessive aux heures de pointe, dans les extremes et sur le prix moyen journalier ;
- pinball loss et couverture de l'intervalle `q10-q90` ;
- couverture exacte de 365 jours et origine D-1 08:00.

Une gate rolling-365 reussie autorise seulement le **shadow**. La promotion exige ensuite 30 jours d'emissions live scellees, produites avec exactement le meme checkpoint et le meme schema. Le bundle de gouvernance recopie les preuves, calcule les checksums et indique toujours `activation_performed=false` : l'activation du contrat live reste une operation separee et explicite.

Une seconde gate, `production_pipeline_evidence`, distingue le POC fondation du forecast operationnel final. `Backtest` compare Chronos-2 brut et son LoRA brut avec les memes inputs et laisse donc cette preuve a `false`, sans option YAML pour la forcer. Apres la calibration residuelle OOF, `FinalBacktest` applique le correcteur gele au holdout LoRA, apparie exactement les memes 365 jours avec le vrai pipeline autonome incumbent `residual_corrected`, puis scelle le rapport et la preuve finale. Seule cette action peut fixer `production_pipeline_evidence=true` ; elle ne change jamais `production_pit_evidence` et ne vaut ni promotion ni activation.

## Limite actuelle des preuves historiques Saturn et JAO

Les backfills historiques Saturn et JAO sont admissibles pour la recherche causale, mais ne constituent pas a eux seuls une preuve de capture operationnelle prospective. Pour JAO, les publications selectionnees ont bien un `lastModifiedOn` anterieur au cutoff ; l'audit indique toutefois `flowbased_operational_pit_eligible=false` sur l'historique reconstruit. Les series Saturn historiques restent elles aussi classees **recherche** tant qu'une capture archivee avant cutoff n'a pas ete observee et scellee prospectivement.

Consequence : tous les packs actuels peuvent etre evalues et executes en shadow, mais **la gouvernance bloque leur promotion reelle**, meme si leurs metriques sont bonnes. La levee de ce verrou necessite une banque prospective de captures Saturn et, selon le pack, JAO, archivees avant D-1 08:00, avec la couverture requise sur la fenetre live gouvernee. Ce verrou ne doit pas etre contourne en modifiant simplement le booleen du YAML.

## Utilisation

Le point d'entrée de l'entraînement, du backtest, du shadow et de la
gouvernance reste `Exogenous.ps1`. Ces actions n'appellent jamais
`Forecast.ps1` et ne peuvent pas activer un modèle. Seul un bundle ayant passé
toutes les gates, enregistré dans le registre puis épinglé explicitement dans
`config/chronos2_exogenous_activation_v1.yaml` peut être consommé par le
launcher opérationnel.

Dans le contrat livré, toutes les zones sont inactives. Le mode `All` de
`Forecast.ps1` refuse alors le batch au preflight, sans fallback incumbent.
Une fois un bundle valide activé pour `autonomous` et `kalman`, `All` produit
uniquement ces deux rapports. Le blend reste une demande séparée via `Blend`
pour FR et NL ; `Both` est réservé aux vues autonome/Kalman de `Run`.

Commande minimale pour verifier le POC fondation brut sans calcul :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action POC -Zones FR -Pack full -EndDay 2026-09-02 -DryRun
```

Commande minimale pour lancer effectivement ce POC fondation :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action POC -Zones FR -Pack full -EndDay 2026-09-02
```

`POC` inclut l'étape de gouvernance et est donc réservé au pack `full`
pré-déclaré. Les packs `residual_only`, `residual_weather`, `residual_fuel` et
`residual_flowbased` peuvent être matérialisés, validés, entraînés et
backtestés séparément pour le diagnostic, mais `Govern` échoue volontairement
pour eux. Cette action composée s'arrête à l'évaluation LoRA brute : elle ne
remplace ni la calibration OOF du correcteur ni `FinalBacktest` et ne peut pas
produire à elle seule un bundle promouvable.

Verifier le panel et le contrat sans charger le modele :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action Panel -Zones FR -EndDay 2026-09-02 -Pack full
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action Validate -Pack full
```

Entrainer, isoler physiquement les zones, puis backtester le challenger FR :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action Train -Pack full
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action PrepareZones -Zones FR,DE,BE,NL -Pack full
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action Backtest -Zones FR -Pack full -Device auto -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact'
```

`PrepareZones` doit être lancé sur l'artefact terminé avant toute évaluation.
Il copie l'intégralité du checkpoint sans lien partagé et ajoute seulement la
zone et la provenance de la copie. Les rangs 8 et 16 restent dans deux racines
distinctes et sont comparés comme deux candidats ; aucun rang ne remplace
l'autre automatiquement. La recette historique du rang 8 est figée dans
`config/chronos2_exogenous_lora_rank8_reference.yaml`; il faut la passer avec
`-Config` pour ses folds OOF. Le panel courant ne permet que FR/DE/BE/NL, pas
ES.

Construire le support long puis calibrer le correcteur LoRA sans toucher au
holdout final :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action CalibrationPanel -Zones FR,DE,BE,NL -Pack full -EndDay 2026-09-02 -Overwrite
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action CalibrateResidual -Zones FR -Pack full -BlockDays 30 -Device auto -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact'
```

Appliquer le correcteur gele et comparer le pipeline final avec l'incumbent sur
la meme fenetre physique rolling-365 :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' `
  -Action FinalBacktest `
  -Zones FR `
  -Pack full `
  -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact'
```

Le lanceur decouvre par defaut les Statistics incumbent les plus recentes de
la zone ; `-IncumbentStatistics` permet d'epingler un fichier precis. Les
artefacts de calibration sont pris dans
`<bundle>/residual_calibration/fr/`, sauf surcharge avec
`-ResidualCalibrationDirectory`. Le controle exige 365 jours locaux et leur
grille physique exacte (8 759, 8 760 ou 8 761 heures selon les frontières
DST), les journees DST intactes et une correspondance exacte des
livraisons, origines et actuals. Il publie sous `<bundle>/final_pipeline/` le
rapport HTML, les metriques, l'audit, le manifeste et
`final_pipeline_predictions.csv.gz`.

Lorsque les actions sont lancees separement, repeter le meme `-Pack` a chaque etape afin de viser le meme YAML et le meme bundle.

Après un premier `Govern` réussi sur la preuve finale rolling-365, émettre le
dernier jour disponible en shadow. Le journal est append-only : `-Overwrite`
y est volontairement interdit.

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' -Action Shadow -Zones FR -EndDay 2026-09-04 -Device auto -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact'
```

`-EndDay` designe le jour de livraison D+1. Le lanceur construit d'abord un panel prospectif dont la cible horizon est absente, puis scelle le forecast. La premiere emission doit avoir lieu dans la fenetre prospective **D-1 08:00 a 12:00 Europe/Paris**. Elle est refusee avant 08:00, apres 12:00 ou si le prix observe est deja disponible : un historique reconstruit apres coup ne peut donc pas se faire passer pour du shadow live.

La premiere emission produit le journal brut scelle
`shadow_predictions.csv.gz`. Tant qu'aucun prix observe n'est disponible, le
finaliseur retourne `pending_observation` sans publier une fausse evidence
vide, mais il verifie deja que le bundle, le `FinalBacktest`, le correcteur OOF
et leurs SHA ont ete geles avant le shadow.

Lorsque les prix deviennent disponibles, relancer exactement le meme
`-EndDay`. Le journal brut joint alors les actuals aux quantiles deja scelles
et conserve, pour audit, `shadow_observed_evidence.csv.gz` et le manifeste v3
`shadow_manifest.json`. Ces deux fichiers **bruts** ne sont jamais recevables
par `Govern`. La meme commande construit automatiquement sous
`shadow_final/` :

- `shadow_final_evidence.csv.gz`, preuve observee utilisee uniquement pour le
  scoring, avec baseline `residual_corrected` incumbent et candidat
  `exogenous_residual_corrected` ;
- `shadow_final_issued_history.csv.gz`, toutes les emissions finales deja
  scellees, y compris la derniere journee dont l'actual peut encore etre vide ;
  ce fichier sert au bootstrap causal du premier run actif ;
- `shadow_final_manifest.json`, manifeste v4 qui lie par SHA le journal brut,
  l'incumbent apparie, le correcteur OOF, le schema, le FinalBacktest et les
  deux CSV finaux.

Le dossier de provenance complet est recopie et checksumme dans le bundle de
promotion. Le validateur recalcule lui-meme les quantiles LoRA corriges : une
preuve brute, arbitrairement modifiee ou corrigee deux fois est refusee.

Pour reconstruire des sorties existantes, ajouter `-Overwrite`. Utiliser d'abord `-DryRun` pour afficher les tableaux d'arguments exacts sans lancer de calcul.

Recalculer la gouvernance a partir de la preuve **finale** rolling-365 :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' `
  -Action Govern `
  -Zones FR `
  -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact' `
  -Predictions 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact\final_pipeline\final_pipeline_predictions.csv.gz'
```

`Govern` exige toujours `-Predictions`, qui doit pointer vers la preuve du
pipeline final et non vers `evaluation_predictions.csv.gz` du backtest brut.
Il infere sous `RunDirectory` le checkpoint, `schema.json` et
`experiment_manifest.json`. Lorsque le correcteur et l'audit OOF se trouvent
dans le dossier de calibration attendu, le lanceur les ajoute automatiquement
aux artefacts gouvernes. Avec `-ShadowPredictions`, il accepte uniquement
`shadow_final/shadow_final_evidence.csv.gz` et exige le manifeste v4
`shadow_final_manifest.json` dans le meme dossier. `-Device`
surcharge le device du backtest et du shadow ; le device d'entrainement reste
defini par `model.device_map` dans le YAML afin que le manifeste de fit demeure
reproductible.

Sans `-ShadowPredictions`, cette commande valide la gate finale rolling-365 et
ne peut rendre qu'une décision de shadow. Après 30 journées observées, relancer
la même gouvernance avec l'évidence scellée :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Exogenous.ps1' `
  -Action Govern `
  -Zones FR `
  -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact' `
  -Predictions 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact\final_pipeline\final_pipeline_predictions.csv.gz' `
  -ShadowPredictions 'runs\experiments\chronos2_exogenous_lora_poc_v2\zones\FR\artifact\shadow_final\shadow_final_evidence.csv.gz'
```

La continuite temporelle est une gate dure : le premier jour local du shadow
observe **et** de l'historique emis doit etre exactement le lendemain du
dernier jour du holdout final rolling-365. Un trou, un chevauchement ou un
backfill retrospectif bloque la promotion. Le panel réellement chargé par le
rang 16 ferme son holdout sur la livraison du 02/09/2026. Comme ce candidat
n'était pas encore gelé et émis prospectivement pour le 03/09, un shadow
commencé plus tard laisserait un trou : ce POC reste non activable même après
30 jours. Une candidature de production doit reconstruire le holdout jusqu'à
la veille du démarrage shadow, ou fournir un bridge causal prospectif
explicitement scellé.

Chaque évaluation finale et chaque gouvernance porte une zone par invocation.
`Panel` peut construire un panel commun avec `-Zones FR,DE,BE,NL`, mais les
résultats et les gates restent séparés par zone pour éviter de masquer une
dégradation locale.

## Chemin shadow vers production

Le contrat technique conditionnel (correcteur résiduel OOF, registre inactif,
preflight et activation opt-in) est détaillé dans
[`CHRONOS2_EXOGENOUS_PRODUCTION.md`](CHRONOS2_EXOGENOUS_PRODUCTION.md). Le
raccord à `Forecast.ps1 -Mode All` existe, mais reste inutilisable tant que le
bundle promu et les captures live prospectives ne satisfont pas toutes les
preuves du contrat.

1. Geler `full` comme candidat primaire avant d'ouvrir son holdout final de 365 jours ; exécuter les ablations uniquement comme diagnostics non gouvernables.
2. Produire les prédictions OOF préquentielles et calibrer le correcteur sans toucher au holdout final.
3. Exécuter `FinalBacktest` ; conserver sa preuve appariée rolling-365 et geler checkpoint, schéma et correcteur.
4. Lancer ce bundle en sidecar shadow, sans influence sur les forecasts publies.
5. Commencer le shadow le lendemain exact du holdout final, sceller chaque forecast avant de joindre le prix observe et accumuler 30 jours complets sans discontinuite.
6. Rejouer `Govern` avec la preuve finale rolling-365, l'evidence shadow et son manifeste.
7. Exiger `decision=promote`, `production_pit_evidence=true` **et** `production_pipeline_evidence=true`.
8. Effectuer une revue manuelle par zone, spreads CWE, DST, donnees manquantes et latence.
9. Seulement après autorisation explicite, enregistrer puis épingler le bundle
   par zone dans le contrat d'activation pour `autonomous` et `kalman` ;
   conserver le rollback incumbent hors de la sémantique fail-closed de
   `Mode All`.

Le rollback conserve le contrat et le checkpoint autonomes actuels. Une indisponibilite d'une covariable, un checksum invalide, une preuve PIT insuffisante ou une latence hors budget doit faire echouer le challenger ferme et laisser le pipeline incumbent intact.
