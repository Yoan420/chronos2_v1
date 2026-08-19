# Pipeline horaire France — Chronos-2, LEAR, CatBoost et supply stack

Cette extension produit **un prix pour chaque produit horaire français**. Elle
ne transforme pas le modèle en modèle 15 minutes : les données 15 minutes
peuvent enrichir les fondamentaux, mais la cible, l'entraînement, le backtest,
la MAE et l'export restent horaires.

Cette documentation décrit le contrat de données `2.3.5`.

Sur un changement d'heure Europe/Paris, « chaque heure » signifie toutes les
heures physiques du marché : 23 lignes au printemps, 24 un jour normal et 25
en automne. Les deux heures locales `02:00` d'automne sont distinguées par leur
timestamp UTC, leur offset et leur champ `fold`.

## Installation

```bash
python -m pip install -r requirements_hourly.txt
```

CatBoost est chargé paresseusement par le code, mais le fichier de dépendances
l'installe afin que `backend: catboost` soit réellement exécuté. Pour un test
CPU léger, `backend: sklearn` utilise un gradient boosting quantile sans
modifier le contrat de sortie.

Le YAML conserve `model.local_files_only: true`. Si `amazon/chronos-2` n'est
pas déjà dans le cache local, passer temporairement cette valeur à `false` pour
le premier téléchargement.

## Exécution

```bash
python run_chronos2_hourly.py \
  --config chronos2_hourly_fr.yaml \
  --refresh-data
```

Pour rejouer exactement un run point-in-time :

```bash
python run_chronos2_hourly.py \
  --config chronos2_hourly_fr.yaml \
  --data-as-of 2026-08-10T08:00:00+02:00
```

Les appels Chronos peuvent être coûteux. Des artefacts déjà calculés peuvent
être réutilisés sans relâcher les validations :

```bash
python run_chronos2_hourly.py \
  --config chronos2_hourly_fr.yaml \
  --chronos-oof-file runs/reference/chronos_oof.csv.gz \
  --chronos-live-file runs/reference/chronos_live.csv
```

L'OOF doit contenir `delivery_start_utc`, `forecast_origin_utc`, `q10`, `q50`,
`q90` et `actual`. Le live ne requiert pas `actual`. Dans les deux cas, les
timestamps doivent être timezone-aware, les quantiles finis et ordonnés, et
l'origine strictement antérieure à la livraison.

## Contrat de cible

- Timeline canonique : UTC, pas horaire, triée, unique et continue.
- Cible Saturn de référence :
  `power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh`, déclarée avec
  `naive_timezone: UTC`. Sa grille physique, vérifiée en UTC, contient 23
  produits au printemps, 24 un jour normal et 25 en automne ; les deux heures
  locales `02:00` d'automne correspondent à deux timestamps UTC distincts.
- L'ancienne cible `power.price.everyday.fr.hourly.eurmwh` est interdite pour
  ce pipeline : elle livre 24 positions civiles même les jours d'automne et
  ne contient qu'un seul `02:00`. La seconde heure et son prix ne peuvent pas
  être reconstruits sans inventer une observation, ce qui invaliderait la MAE.
- Toute autre source Saturn naïve doit déclarer son fuseau réel à la frontière
  de chargement. Une source fichier locale doit fournir des offsets explicites
  aux changements d'heure.
- Aucun `interpolate`, `ffill` ou `bfill` sur le prix cible.
- Avec une source 15 minutes, quatre MTU valides sont obligatoires et le prix
  horaire est leur moyenne arithmétique.
- Une heure incomplète bloque le run par défaut.
- La métrique est calculée uniquement contre la cible horaire observée.

Pour passer à une série 15 minutes, modifier la série cible et :

```yaml
data:
  target_input_resolution: quarter_hour
  quarter_hour_incomplete: raise
  frequency: h
```

Le résultat demeure une prévision horaire.

Après une mise à jour depuis une version antérieure à `2.3.3`, ou après le
remplacement de l'ancienne série cible, reconstruire une fois entièrement le
cache de la cible. `--full-data-refresh` ne doit pas être utilisé pour cette
correction : il retéléchargerait aussi plusieurs années de vintages PIT.

```bash
python run_chronos2_hourly.py \
  --config chronos2_hourly_fr.yaml \
  --full-target-refresh \
  --skip-pit-refresh
```

Retirer `--skip-pit-refresh` pour actualiser les PIT de manière incrémentale
pendant la même exécution.

Après cette reconstruction unique, l'actualisation opérationnelle redevient
incrémentale :

```bash
python run_chronos2_hourly.py \
  --config chronos2_hourly_fr.yaml \
  --refresh-data
```

Ajouter `--skip-pit-refresh` à cette commande seulement si les vintages PIT
ont déjà été actualisés séparément. Le champ `naive_timezone: UTC` reste
nécessaire : il empêche qu'une tranche incrémentale courte, ne traversant aucun
changement d'heure, soit interprétée à tort comme une grille Europe/Paris.

## Causalité et PIT

Pour une livraison D, le chargeur PIT conserve uniquement un snapshot dont
**la disponibilité et la révision** sont antérieures au cutoff D−1 configuré.
Le `runtime_as_of` borne également ce cutoff : un run lancé avant 08:00 ne peut
pas lire une révision qui paraîtra plus tard le même matin.

La couverture est bloquante à deux niveaux : couverture historique minimale
de chaque série et couverture à 100 % de toutes les features du jour futur.
Une feature absente ne peut donc pas être silencieusement retirée.

Les observations historiques sont autorisées lorsqu'elles étaient disponibles
au cutoff. La version `2.3.5` active deux proxys conservateurs à 96 heures
physiques :

- `fr_nuclear_generation_obs` via `lag96` ;
- `fr_net_exports` via `lag96`.

Dans les deux cas, `include_base_context: false` empêche Chronos de recevoir la
série observée brute. Les `lag24`, `lag48` et `lag72` ne sont pas construits :
pour le forecast du 12 août 2026, les dernières observations Saturn valides
étaient respectivement le 9 août à 05:00 et 06:00. Un `lag72` aurait exigé
toute la journée du 9 août et laissé une partie du futur sans valeur. Le
`lag96` reporte le jour à prévoir sur une plage entièrement disponible.

`lag96` signifie exactement un `shift(96)` sur la grille horaire physique
continue, et non une sélection adaptative de la dernière observation. Il
équivaut à D−4 hors transition DST. Aucun fallback silencieux vers un lag
plus court, une persistance ou un remplissage illimité n'est appliqué. La
couverture future reste exigée à 100 % ; si les 96 heures ne suffisent plus,
le run s'arrête et le contrat doit être réévalué explicitement.

La série nucléaire agrégée
`power.fr.generation.nuclear.entsoe.hourly.gw.obs` utilise des timestamps naïfs
en heure civile française. Elle contient correctement 23 positions au passage
à l'heure d'été, mais seulement 24 positions au passage à l'heure d'hiver : le
second `02:00` n'est pas fourni. Le YAML déclare donc explicitement :

```yaml
naive_timezone: Europe/Paris
incomplete_dst_policy: duplicate
```

Cette politique duplique la valeur du seul `02:00` sur les deux folds
d'automne. L'imputation porte sur une heure physique par an et uniquement sur
une covariable. Chaque réparation doit être journalisée par le chargeur. Elle
est également déclarée pour `fr_net_exports` afin que le chargement reste
robuste si cette source présente le même défaut de grille.

La duplication n'agit que lorsque le normaliseur identifie un unique label
local ambigu lors du passage à l'heure d'hiver. Si la source contient déjà la
vraie paire des deux folds, ou si elle fournit une timeline UTC continue, les
valeurs et timestamps restent inchangés. Les exports ne déclarent donc pas de
`naive_timezone` arbitraire : la détection existante conserve leur sémantique.

`incomplete_dst_policy: duplicate` est strictement interdit pour la cible prix
et pour toute série évaluée par la MAE : deux produits horaires distincts
peuvent avoir deux prix très différents, et inventer le second invaliderait le
backtest. La cible conserve donc une timeline UTC complète sans aucune
imputation DST.

Pour une variable observée plus réactive, ou si le second fold a un sens
économique distinct, il faut préférer une série UTC complète.

À terme, deux sources plus précises restent préférables aux proxys `lag96` :

1. une série scheduled/forecast réellement versionnée et connue avant le
   cutoff ;
2. une série observée versionnée permettant de reproduire exactement son état
   au cutoff historique.

Les labels d'order/block ne lisent plus le premier prix du jour suivant. Ils
restent des labels ex post et ne doivent entrer dans un run que sous forme de
prévisions OOF matérialisées PIT.

## Features day-ahead

Les prix de toutes les heures de D sont prédits simultanément. Une simple
causalité « heure t−1 » serait donc insuffisante : le prix réalisé de 00:00 D
n'est pas disponible lorsque 23:00 D est prévu.

Le feature builder applique le contrat plus strict suivant :

- un lag est masqué si sa source appartient au même jour local D ;
- le `lag24` de la 25e heure automnale est donc `NaN` ;
- les statistiques rolling sont calculées à la fin de D−1 puis figées sur
  toutes les heures de D ;
- modifier n'importe quel prix réalisé de D ne peut changer aucune feature de
  D.

Les calendriers, fondamentaux PIT et courbes prévisionnelles de D restent
variables par heure, puisqu'ils sont connus avant l'enchère.

## LEAR, CatBoost et ensemble

- **LEAR** : ElasticNet robuste avec modèles par heure locale et fallback
  global.
- **CatBoost** : trois modèles quantiles (`q10/q50/q90`), global par défaut ;
  une spécialisation par heure peut être activée via
  `min_samples_per_hour`.
- **Chronos-2** : expert zero-shot existant, exécuté par groupes d'horizon
  23/24/25.
- **Ensemble** : poids non négatifs dont la somme vaut 1, optimisés sur la MAE.

Les prédictions des experts supervisés sont expanding-window OOF, avec folds
alignés sur des journées locales entières et purge temporelle. Les poids sont
appris sur un bloc OOF antérieur. Le dernier bloc de 365 jours est un holdout
intact : il ne sert ni au choix des poids ni à l'entraînement de ses propres
prévisions. C'est uniquement sur ce bloc que la MAE publiée est calculée.

## Supply stack France

Le bloc `chronos2_hourly/fundamental_features.py` construit notamment :

- charge résiduelle et rampes ;
- offre ferme disponible et marge ferme ;
- pression de prix négatif ;
- coûts marginaux CCGT, OCGT et charbon à partir de TTF, EUA, API2 et EUR/USD ;
- spread de switching gaz/charbon ;
- gaps scarcity/surplus et interactions non linéaires.

Les puissances doivent partager la même unité (MW ou GW). Les combustibles et
le CO2 suivent les unités documentées dans le module. Aucune série n'est
remplie ou resamplée par ce bloc.

Le YAML livre le mapping complet, mais garde `enabled: false` tant que les 15
inputs standardisés ne sont pas reliés à de vrais vintages PIT. Après ajout de
ces alias dans `zones.FR.covariates`, activer :

```yaml
hourly:
  supply_stack:
    enabled: true
    nan_policy: raise
```

Le premier input manquant bloque alors l'exécution, ce qui évite de comparer
une version « supply stack » qui n'aurait en réalité pas reçu le stack.

## Fichiers produits

- `forecast_hourly_fr.csv` : q10/q50/q90, prix central, experts et métadonnées
  UTC/locales pour chaque heure physique ;
- `backtest_hourly_oof.csv.gz` : actuals, experts, ensemble, folds et origines ;
- `metrics_hourly.csv/json` : MAE du holdout et couvertures ;
- `ensemble_weights.csv` : poids convexes appris hors holdout ;
- `feature_manifest.csv` : schéma et couverture historique ;
- `run_manifest.json` : configuration, dates, horizon DST et diagnostics PIT.

## Forecast live day-ahead et Statistics cumulees

Apres 08:00 Europe/Paris, la commande operationnelle sans argument de date
produit automatiquement la livraison de J+1 :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  'C:\Users\BQ6757\chronos2_v1\run_mkonline_live_hourly.py' `
  --config 'C:\Users\BQ6757\chronos2_v1\chronos2_hourly_fr_mkonline_live_v1.yaml'
```

Chaque run est immuable et publie son forecast sous
`runs/live/fr_day_ahead_YYYY-MM-DD`. Le rapport conserve les KPI du benchmark
annuel scelle, mais sa table **Statistics** lit une source separee :
`statistics_history_hourly.csv.gz`. Celle-ci ajoute uniquement les jours
anterieurs complets (actuals canoniques + Storm selectionne au cutoff civil
J-1 08:00 apres gel du candidat).

Un jour historique manquant peut etre reconstruit explicitement, sans etre
presente comme un forecast live emis :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  'C:\Users\BQ6757\chronos2_v1\run_mkonline_live_hourly.py' `
  --config 'C:\Users\BQ6757\chronos2_v1\chronos2_hourly_fr_mkonline_live_v1.yaml' `
  --data-as-of '2026-08-12T08:00:00+02:00' `
  --delivery-day '2026-08-13' `
  --pit-replay
```

Ces reconstructions sont rangees sous `runs/live/_replays` et libellees
`replays PIT causaux (non emis en temps reel)` dans le HTML. Pour actualiser
un ancien rapport sans toucher a son forecast publie, utiliser
`refresh_live_statistics_report.py`; le resultat immuable est place sous
`runs/live/_reports`.

## Tests

```bash
python -m unittest \
  tests.test_hourly_contract \
  tests.test_hourly_features \
  tests.test_fundamental_features \
  tests.test_hourly_models \
  tests.test_chronos_adapter \
  tests.test_oof_pipeline \
  tests.test_hourly_runner \
  tests.test_dynamic_delivery_data \
  tests.test_target_dst_regression \
  tests.test_causality_guards \
  test_saturn_pit -v
```

La suite couvre explicitement les journées 23/25 heures, l'agrégation QH,
l'absence de prix futurs dans les features, les revisions post-cutoff, la
couverture bloquante et l'alignement OOF.
