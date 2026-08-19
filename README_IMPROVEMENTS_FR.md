# Amélioration du modèle Day-Ahead France

## Résultat obtenu

Le meilleur point de départ **comparable et causal** est le pipeline horaire
canonique `chronos2_hourly_fr_baseline_v235.yaml`, et non le run legacy qui
affiche 12,710. Sur le holdout final du 12 août 2025 au 11 août 2026 :

| Modèle | MAE (€/MWh) | Heures | Couverture |
|---|---:|---:|---:|
| Chronos-2 | 12,8650 | 8 760 | 100 % |
| Ensemble horaire | 12,7696 | 8 760 | 100 % |
| **Chronos-2 + correcteur résiduel v1** | **12,2835** | **8 760** | **100 %** |

Le gain apparié face à l'ensemble est de **0,4861 €/MWh (3,81 %)**. Un
bootstrap de 20 000 rééchantillonnages de journées locales complètes donne un
IC95 de **[-0,7230 ; -0,2458] €/MWh** et une probabilité empirique de gain de
99,99 %.

Ce résultat n'atteint pas encore 10. Il ne faut pas présenter 10 comme acquis
avec les données actuelles : l'amélioration restante nécessite surtout de
nouvelles informations PIT de stack de production et une meilleure gestion
des régimes extrêmes.

## Pourquoi le run legacy à 12,710 n'est pas le socle

`selected_calendar_fr_neighbours_order_signals_structural_covariates` est le
plus bas chiffre historique, mais il utilise l'ancienne cible, interpole quatre
valeurs et exclut les deux journées DST irrégulières. Il ne couvre donc pas le
même ensemble de 8 760 heures. Son fichier structurel amont a aussi été
réécrit après le run sans checksum.

Le protocole retenu ici impose :

- cible UTC `power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh` ;
- zéro interpolation de la cible ;
- 365 journées locales consécutives, y compris DST 23/25 heures ;
- exactement 8 760 prédictions et 100 % de couverture ;
- cutoff de toutes les courbes PIT à D-1 08:00 Europe/Paris.

## Implémentations ajoutées

### Correcteur résiduel causal

`chronos2_hourly/models/residual_corrector.py` apprend :

```text
résidu = prix_réalisé - Chronos_q50
prévision_corrigée_qXX = Chronos_qXX + résidu_prédit
```

La même correction est ajoutée à q10, q50 et q90 : l'ordre et la largeur des
quantiles sont donc conservés. Les variables comprennent :

- courbes de charge résiduelle PIT FR, DE, BE, NL et ES ;
- profils journaliers, amplitudes et rampes 1 h/2 h ;
- écarts France–voisins ;
- quantiles, largeur d'intervalle et désaccord des experts ;
- calendrier français et voisin, jours fériés, ponts et DST ;
- variables déterministes d'heure et de jour de semaine.

Les lags/rollings du prix observé et les variables de jour de l'année sont
exclus par défaut du correcteur. Le backend principal est CatBoost, avec un
fallback scikit-learn testable.

Le pipeline entraîne deux instances séparées :

1. le modèle d'évaluation apprend uniquement sur les 365 premiers jours OOF et
   prédit l'année finale scellée ;
2. une fois les métriques figées, le modèle live apprend sur les 730 jours OOF
   pour prévoir strictement après la fin de l'historique.

### Garde-fous et diagnostics

- `pit_feature_coverage_by_hour.csv` et
  `pit_feature_coverage_summary.csv` exposent les trous par heure locale ;
- `artifact_checksums.json` contient les SHA256 de la configuration, du code,
  des entrées matérialisées et des résultats ;
- `evaluate_hourly_backtest.py` produit métriques mensuelles/horaires et
  bootstrap apparié par journées physiques complètes ;
- la réparation des quantiles conserve désormais q50 au lieu de trier et de
  pouvoir remplacer silencieusement la médiane dédiée.

## Exécution reproductible

Depuis la racine du projet :

```powershell
C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe -m pip install -r requirements_hourly.txt

$env:OMP_NUM_THREADS='1'
$env:LOKY_MAX_CPU_COUNT='1'
C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe run_chronos2_hourly.py `
  --config chronos2_hourly_fr_residual_v1.yaml `
  --chronos-oof-file runs\chronos2_hourly_fr_baseline_v235\chronos_oof_hourly.csv.gz `
  --chronos-live-file runs\chronos2_hourly_fr_baseline_v235\chronos_live_hourly.csv
```

Sans artefacts Chronos pré-calculés, omettre les deux options
`--chronos-*-file`; le runner appellera le runtime Chronos configuré.

Évaluation indépendante des prédictions exportées :

```powershell
C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe evaluate_hourly_backtest.py `
  runs\chronos2_hourly_fr_residual_v1\backtest_hourly_oof.csv.gz `
  --bootstrap-samples 20000
```

Le rapport HTML autonome (même gabarit Plotly que les rapports historiques)
est généré automatiquement par la configuration résiduelle. Il peut aussi être
recréé sans réentraîner les modèles :

```powershell
C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe generate_hourly_html_report.py `
  runs\chronos2_hourly_fr_residual_v1 `
  --output runs\chronos2_hourly_fr_residual_v1\chronos2_hourly_fr_residual_v1.html `
  --native-model residual_corrected `
  --baseline-model ensemble
```

Les résultats principaux se trouvent dans :

- `runs/chronos2_hourly_fr_residual_v1/metrics_hourly.csv` ;
- `runs/chronos2_hourly_fr_residual_v1/evaluation_summary.json` ;
- `runs/chronos2_hourly_fr_residual_v1/evaluation_by_month.csv` ;
- `runs/chronos2_hourly_fr_residual_v1/residual_feature_importance.csv` ;
- `runs/chronos2_hourly_fr_residual_v1/chronos2_hourly_fr_residual_v1.html` ;
- `runs/chronos2_hourly_fr_residual_v1/artifact_checksums.json`.

## Ce qui bloque encore l'objectif de 10

### 1. Compléter les courbes PIT de fin de journée

Sur le holdout, la charge résiduelle FR est disponible à 93,71 % au total,
mais seulement à **3,56 % à 23 h** (45,75 % à 22 h lors de l'audit détaillé
de la source). Les 551 heures FR manquantes sont associées à une MAE bien plus
élevée. Le headroom arithmétique maximal est d'environ 0,24 €/MWh.

Action : étendre l'horizon de la dernière vintage publiée avant le cutoff, ou
utiliser une source/vintage de secours elle aussi antérieure au cutoff. Ne pas
utiliser une observation ex post ni une interpolation de cible.

### 2. Activer un vrai supply stack PIT

Le projet contient déjà `chronos2_supply_stack_fr_inputs.example.yaml` et le
feature builder correspondant. Les entrées actuellement manquantes sont les
plus susceptibles d'expliquer les mauvais jours de printemps/été :

- load, éolien et solaire séparés ;
- disponibilités nucléaire et thermique ;
- hydro, pompage et run-of-river ;
- capacités d'import/export ;
- TTF, EUA, API2 et EUR/USD connus au cutoff.

Le bloc `hourly.supply_stack` doit rester désactivé tant que ces fichiers PIT
ne sont pas fournis et audités.

### 3. Valider les régimes sans réutiliser le holdout

Le correcteur améliore 11 mois sur 13 mais dégrade novembre et décembre. La
suite logique est un gate soft entraîné OOF entre correction nulle et
correction résiduelle, puis des experts dédiés oversupply/near-zero/spike. Les
seuils doivent être sélectionnés sur une période antérieure, jamais sur cette
même année finale.

### 4. Passer à une évaluation préquentielle

Après le benchmark gelé, ajouter un second rapport opérationnel avec
réentraînement hebdomadaire ou mensuel du correcteur sur les seuls labels déjà
publiés. Tester des fenêtres de 365, 730 et 1 095 jours. Conserver en parallèle
le holdout gelé pour comparer les versions.

### 5. Réserver un nouveau test réellement vierge

L'année 2025-08/2026-08 a maintenant servi à plusieurs diagnostics. Elle reste
une validation annuelle comparable, mais plus un test totalement vierge pour
de nouvelles décisions. Après stabilisation des données et hyperparamètres,
geler le code et mesurer une période ultérieure jamais inspectée.
# Mise a jour - run scelle MKOnline primaire

Le meilleur run annuel publie est desormais
`chronos2_hourly_fr_mkonline_blend_v1`. Il combine le modele autonome etendu
avec la serie primaire MKOnline/WattSight `41551_native`. Le poids MKOnline
(`0.5022390717075804`) a ete appris par validation croisee temporelle avant la
periode finale, puis a passe deux gates chronologiques B1 et B2. Il n'a pas
ete retouche apres ouverture du test annuel.

Sur les 365 jours locaux du 12 aout 2025 au 11 aout 2026 :

| Modele | MAE (EUR/MWh) | Heures | Couverture |
|---|---:|---:|---:|
| Modele autonome etendu | 12.1654 | 8 760 | 100 % |
| Storm legacy, evaluation seulement | 14.2080 | 8 760 | 100 % |
| **Blend autonome + MKOnline primaire** | **11.2311** | **8 760** | **100 %** |

Le gain est de `0.9343 EUR/MWh` face au modele autonome (7.68 %, IC95
bootstrap journalier `[-1.3193; -0.5558]`) et de `2.9768 EUR/MWh` face a Storm
(20.95 %, IC95 `[-3.6615; -2.3202]`). Le candidat bat Storm sur les 13 mois
civils partiels/complets et les 24 heures locales. Storm est charge apres le
gel de la prediction et sert exclusivement de comparateur : il n'entre ni dans
les variables, ni dans le blend, ni dans le forecast live.

Commande PowerShell reproductible (elle genere aussi le rapport HTML) :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  .\run_mkonline_blend_hourly.py `
  --config .\chronos2_hourly_fr_mkonline_blend_v1.yaml `
  --overwrite
```

Artefacts principaux :

- `runs/chronos2_hourly_fr_mkonline_blend_v1/chronos2_hourly_fr_mkonline_blend_v1.html` ;
- `runs/chronos2_hourly_fr_mkonline_blend_v1/evaluation_summary.json` ;
- `runs/chronos2_hourly_fr_mkonline_blend_v1/evaluation_vs_storm_summary.json` ;
- `runs/chronos2_hourly_fr_mkonline_blend_v1/artifact_checksums.json`.

La cible ideale sous 11 n'est pas atteinte sur ce holdout : l'ecart restant
est de `0.2311 EUR/MWh`. Aucun ajustement post-hoc du poids n'est autorise sur
cette annee finale. Les droits d'utilisation et de remplacement du flux
MKOnline/WattSight doivent etre confirmes avant mise en production.
