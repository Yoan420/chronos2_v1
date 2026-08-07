# Chronos-2 — Merit Order structurel MILP + Branch-and-Cut

Cette extension ajoute un modèle structurel agrégé au pipeline Chronos-2 sans modifier les fichiers cœur du projet.

## Architecture

1. Les fondamentaux point-in-time déjà matérialisés par Chronos-2 sont chargés.
2. Un MILP journalier agrégé décide le nombre d'unités thermiques en ligne, les démarrages et le dispatch.
3. Le MILP est résolu par `scipy.optimize.milp`, qui utilise le solveur MIP HiGHS.
4. Les variables entières optimales sont fixées.
5. Le problème est résolu une seconde fois comme LP avec `scipy.optimize.linprog(method="highs")`.
6. Le dual de la contrainte d'équilibre donne `milp_structural_price`.
7. Les états structurels sont transmis à Chronos-2 comme covariables futures.
8. Un second runner prévoit le résidu `prix réel - prix structurel`, puis réadditionne le prix structurel aux points et quantiles prévus.

## Fichiers principaux

- `chronos2_structural_market/config.py` : paramètres du marché et technologies.
- `chronos2_structural_market/model.py` : MILP, LP de pricing, extraction des états.
- `chronos2_structural_market/features.py` : résolution jour par jour et exports.
- `chronos2_structural_market/residual.py` : transformation de cible et retour à l'espace des prix.
- `install_structural_market.py` : création des deux YAML.
- `build_structural_market_features.py` : backfill des variables structurelles.
- `run_chronos2_structural_covariates.py` : Chronos avec covariables MILP.
- `run_chronos2_structural_residual.py` : prix structurel + résidu Chronos.
- `run_structural_branch_cut_pipeline.ps1` : orchestration complète.
- `compare_structural_models.py` : comparaison C0/C1/C3.

## Variables extraites

Le fichier `data/derived/structural_market_features.csv.gz` contient :

- `milp_structural_price`
- `milp_marginal_technology_code`
- `milp_reserve_margin_gw`
- `milp_committed_thermal_gw`
- `milp_online_units`
- `milp_startups`
- `milp_startup_cost_eur`
- `milp_scarcity_gw`
- `milp_ramp_shadow_eur_mwh`
- `milp_reserve_shadow_eur_mwh`

Deux diagnostics supplémentaires sont exportés dans le fichier, mais ne sont pas ajoutés par défaut à Chronos :

- `milp_spill_gw`
- `milp_reserve_shortfall_gw`

## Installation

Extraire le ZIP à la racine de :

```text
C:\Users\BQ6757\chronos2_v1
```

Les fichiers du dossier `chronos2_structural_market` doivent rester dans ce sous-dossier.

Vérification des dépendances :

```powershell
python -m pip install -r .\requirements_structural.txt
```

## Smoke test recommandé

Le smoke test limite le backfill aux 14 derniers jours :

```powershell
.\run_structural_branch_cut_pipeline.ps1 `
    -BaseConfig "chronos2_selected_core.yaml" `
    -Mode both `
    -MaxDays 14 `
    -FailFast
```

Il vérifie :

- la disponibilité de SciPy/HiGHS ;
- les tests unitaires ;
- la résolution MILP ;
- le LP de pricing ;
- le chargement des covariables dans Chronos ;
- la reconstruction des prix du mode résiduel.

Avec `-MaxDays 14`, les runs Chronos longs peuvent manquer de couverture historique. Pour vérifier seulement la partie optimisation, exécuter directement :

```powershell
python .\install_structural_market.py `
    --config .\chronos2_selected_core.yaml

python .\build_structural_market_features.py `
    --config .\chronos2_selected_core_structural_covariates.yaml `
    --zone FR `
    --max-days 14 `
    --fail-fast
```

## Backfill complet et runs C1/C3

```powershell
.\run_structural_branch_cut_pipeline.ps1 `
    -BaseConfig "chronos2_selected_core.yaml" `
    -Mode both `
    -RefreshData
```

L'actualisation Saturn est effectuée avant la construction du MILP. Les runs Chronos réutilisent ensuite exactement les mêmes caches et vintages.

Pour une reconstruction Saturn complète :

```powershell
.\run_structural_branch_cut_pipeline.ps1 `
    -BaseConfig "chronos2_selected_core.yaml" `
    -Mode both `
    -FullDataRefresh
```

Pour rejouer un cutoff historique :

```powershell
.\run_structural_branch_cut_pipeline.ps1 `
    -BaseConfig "chronos2_selected_core.yaml" `
    -Mode both `
    -DataAsOf "2026-08-06T08:00:00+02:00"
```

## Configurations produites

### C1 — covariables structurelles

```text
chronos2_selected_core_structural_covariates.yaml
```

Le prix reste la cible de Chronos et les dix états MILP sont des covariables.

### C3 — prix structurel + résidu

```text
chronos2_selected_core_structural_residual.yaml
```

La cible interne devient :

```text
résidu = prix observé - milp_structural_price
```

Après la prévision :

```text
prix final = prix structurel + résidu prévu
```

La même constante est ajoutée à tous les quantiles. L'ordre des quantiles est donc conservé.

## Entrées structurelles par défaut

```yaml
structural_model:
  inputs:
    residual_load_gw:
      alias: fr_residual_load_fcst
      scale: 1.0
    nuclear_capacity_gw:
      alias: fr_nuclear_generation_fcst
      scale: 0.001
      default: 45.0
    fixed_net_exports_gw:
      alias: null
      default: 0.0
```

Le forecast nucléaire est supposé être en MW et converti en GW.

## Coûts dynamiques des combustibles

La version initiale fonctionne avec des coûts variables fixes. Pour activer les coûts dynamiques, ajouter d'abord les séries correspondantes comme covariables Chronos, puis renseigner leurs alias :

```yaml
structural_model:
  inputs:
    gas_price_eur_mwhth:
      alias: ttf_eur_mwhth
      scale: 1.0
    coal_price_eur_mwhth:
      alias: api2_eur_mwhth
      scale: 1.0
    eua_price_eur_t:
      alias: eua_eur_t
      scale: 1.0
```

Les unités attendues sont impératives :

- gaz et charbon : EUR/MWh thermique ;
- EUA : EUR/tCO2 ;
- capacités et charge : GW.

Le coût est alors calculé par :

```text
fuel / efficiency + emission_factor × EUA + variable_OM
```

Si une série combustible est absente, le modèle utilise `fixed_variable_cost_eur_mwh`.

## Technologies agrégées initiales

- nuclear
- hydro
- coal
- ccgt
- ocgt
- oil

Les paramètres sont volontairement configurables dans les deux YAML : nombre d'unités agrégées, puissance unitaire, minimum technique, coûts, rampes et budget hydro journalier.

## Exports

```text
data/derived/structural_market_features.csv.gz
data/derived/structural_market_features_dispatch.csv.gz
data/derived/structural_market_daily_diagnostics.csv
data/derived/structural_market_metadata.json
```

Le diagnostic journalier contient notamment :

- statut MILP ;
- objectif MILP ;
- objectif du LP fixé ;
- nombre de nœuds Branch-and-Cut ;
- MIP gap ;
- borne duale.

## Comparaison des modèles

Le pipeline tente de comparer :

- C0 : configuration de base ;
- C1 : covariables structurelles ;
- C3 : prix structurel + résidu.

Le fichier produit est :

```text
runs/structural_model_comparison.csv
```

Le baseline C0 doit avoir été exécuté auparavant avec son propre `output.directory`.

## Limites méthodologiques

Cette première version est un proxy structurel, pas une reproduction d'EUPHEMIA :

- technologies agrégées ;
- pas de carnet d'ordres complet ;
- pas de block orders explicites ;
- pas de flow-based domain ;
- coûts de démarrage et rampes à calibrer ;
- hydro représenté par un budget énergétique simple ;
- capacités étrangères non modélisées explicitement.

Avant toute conclusion, calibrer les paramètres uniquement sur une période d'entraînement et comparer les modèles sur les mêmes origines de backtest.
