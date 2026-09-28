# Variables des trois modèles annuels CWE : filiation et contrôle

Le manifeste `config/nyx_annual_cwe_historical.json` désigne les **vraies
matrices utilisées par les modèles retenus**. Les 449 variables du modèle FR
proviennent de `pooled_fundamentals_v1`. Les 503 variables des modèles BE/NL
proviennent de `pooled_jao_refresh_v1` ; leurs 123 variables compactes sont
une projection ordonnée de cette matrice 503 rafraîchie.

## Construction historique par pays

| Étape | Colonnes | Ajout précis | Entrées qui doivent devenir prospectives |
|---|---:|---|---|
| `run_nyx_price_experiment.py` | 292 | Variables de prix retardés, NYX et covariables de marché | Historique EPEX FR/DE/BE/NL, prévisions NYX et covariables Saturn à chaque coupure civile D−1 08 h. La recette de recherche est datée et lit des archives locales. |
| `prepare_nyx_extra_features.py` | 334 | 21 valeurs voisinage/saison/combustible et 21 indicateurs de disponibilité | Quantiles NYX q10/q50/q90 et origines des quatre pays ; TTF/EUA et coûts marginaux. Seul le collecteur fuel dédié fournit actuellement une partie de ce groupe. |
| `run_nyx_pooled_feature_materialization.py` | 449 | 9 calendaires, 16 hydro FR, 28 JAO initial, 62 profils prévus | Hydro public, JAO `initialComputation` et profils de prévision des quatre pays. Le script historique exige 17 880 heures figées et des SHA d'archives locales. |
| `run_nyx_canonical_feature_materialization.py` | 457 | 8 variables Test2 propres au pays | Vent, solaire et charge résiduelle prévues avec 14 jours d'échauffement et origines vérifiées. |
| `run_nyx_thermal_feature_materialization.py` | 493 | **36 colonnes** : 13 capacités Pmax, 5 alias propres au pays, chacune avec indicateur de disponibilité | Treize séries de capacité prévisionnelle. Les caches historiques déclarent eux-mêmes que leur preuve PIT de production est absente. |
| `run_nyx_exchange_feature_materialization.py` | 503 | **10 colonnes** : 5 échanges retardés de 48 heures et 5 indicateurs | Échanges DE/FR issus d'Energy Charts ; lag physique et coupure nominale, sans heure fournisseur de publication certifiée. |
| `run_nyx_jao_late_refresh_features.py` | 503 | Remplace les **28 colonnes JAO** de l'ancien 503 pour les heures admissibles du 4 au 23 septembre 2026 | Captures JAO initiales avec watermark et normalisation des contraintes. Il ne rajoute aucune colonne. |
| Projection compacte | 123 | Sélection ordonnée de 123 colonnes du 503 rafraîchi | Aucune collecte ou formule supplémentaire. |

Les **54 colonnes présentes dans le 503 et absentes du 449** sont donc les
8 variables Test2, les 36 capacités thermiques et les 10 échanges retardés.
Leurs formules de calcul se trouvent dans les modules de recherche correspondants,
mais les producteurs historiques sont bornés à des chemins et dates 2024–2026.
Les copier sur le poste de travail ne les rendrait pas capables de produire le
jour suivant.

## Résultat de parité sur les archives locales

`python run_nyx_annual_feature_projection.py --action historical-parity`
vérifie les SHA du manifeste, l'ordre, les types, les heures UTC, les NaN et
les valeurs des douze matrices. Le 123 est une projection exacte du 503 pour
FR, DE, BE et NL. Les 421 colonnes communes **hors JAO** du 449 et du 503
sont exactes. En revanche, sur chacun des quatre pays, 271 heures × 28
colonnes JAO, soit **7 588 cellules**, diffèrent entre le 449 original et le
503 rafraîchi. Une projection 503→449 aurait donc changé l'entraînement du
modèle FR ; elle est interdite par le nouvel outil.

Après production réelle des 449 et 503 matrices d'un nouveau jour, la commande
suivante peut seulement construire la projection 123 et vérifier les 421
colonnes communes :

```powershell
python run_nyx_annual_feature_projection.py --action project-live --bundle runs/live/nyx_annual_cpu/AAAA-MM-JJ --delivery-day AAAA-MM-JJ
```

Elle attend quatre fichiers `features/fr_residual_1000/{FR,DE,BE,NL}.parquet`
et quatre fichiers `features/cwe_absolute_2000/{FR,DE,BE,NL}.parquet` au
format du manifeste ordonné. Elle refuse une matrice absente, une colonne
manquante/réordonnée, une disponibilité incohérente, une heure manquante ou
une différence dans les 421 variables communes hors JAO. Elle écrit seulement
les quatre `features/cwe_residual_2000/*.parquet`. Elle ne produit ni le 449,
ni le 503, ni les références prix, ni une preuve de publication fournisseur.

## Blocages sur un clone du dépôt

Les archives `runs/experiments` sont ignorées par Git ; les matrices 449 et
503 ne seront donc pas sur le poste professionnel. Les collecteurs suivis
apportent actuellement les prix EPEX passés et le fuel Saturn, mais pas les
quantiles NYX des quatre pays, les covariables complètes, l'hydro, les treize
capacités, les échanges ni les captures JAO requises par ces variables.
La source `pooled_jao_refresh_v1` est un correctif de la tranche 2026-09-04
à 2026-09-23, pas un producteur continu. Pour entraîner les modèles sur une
fenêtre glissante de 365 jours, il faut des recettes prospectives pour les
449 et 503 variables, des historiques compatibles et un audit chronologique
de leurs coupures. Ce contrôle de projection ne qualifie pas les scores de
prix, les probabilités négatives ou la référence `scarcity_confirmed_pair`.
