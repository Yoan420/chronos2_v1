# Entrées prospectives du modèle annuel CWE sur un clone NYX

Le replay CPU des rapports du 23 septembre lit des matrices scellées sous
`runs/experiments/nyx_improvement_to20260923`. Ces fichiers sont ignorés par
Git. Leur présence sur le poste d'étude ne démontre pas qu'un clone sur le
poste de travail peut calculer les mêmes variables pour une nouvelle date.
La commande `python preflight_nyx_annual_live.py --bundle <dossier> --delivery-day
AAAA-MM-JJ` donne l'inventaire exact de chaque entrée manquante.

## Ce que le code suivi sait déjà produire

| Groupe | Producteur présent | Portée réelle |
|---|---|---|
| Prix EPEX réalisés FR, DE, BE, NL | `run_nyx_annual_auction_prices_source.py`, utilisant les quatre séries cibles canoniques déclarées dans `chronos2_hourly_{fr,de,be,nl}_residual*.yaml` et `chronos2_modular/saturn.py` | Capture Saturn à l'état D−1 08 h de 365 jours de labels plus 7 jours de lags. Écrit quatre Parquet sans prix de la journée prévue et un reçu `auction_prices` vérifié. Un accès Saturn réel sur le poste professionnel reste à tester. |
| Gaz TTF et EUA | `run_nyx_annual_fuel_source.py` avec `materialize_saturn_kalman_fuel.py` | Huit valeurs et six horodatages par heure, reçu `fuel`. Ne produit aucune matrice du modèle. |
| Résiduelle et nucléaire Saturn | `chronos2_hourly/nyx_regional_cpu_sources.py` avec `run_nuclear_forecast.py`, `chronos2_hourly/nuclear_residual_inputs.py`, `chronos2_modular/saturn.py` | Synchronise des banques ciblées pour le challenger régional CPU. Il ne calcule ni les 449/503 variables CWE ni la référence annuelle. Ses fichiers `residual_bank` et nucléaires sont créés localement, pas livrés dans Git. |
| JAO, hydro, thermique, échanges | `materialize_jao_core_flowbased.py` est suivi ; `collect_nyx_public_hydro.py`, `collect_nyx_energy_charts_de_cbpf.py` et les constructeurs de recherche sont uniquement présents comme fichiers non suivis sur le poste d'étude | Aucun producteur complet de ces variables n'est livré par le clone. Les producteurs historiques exigent aussi des archives scellées à dates fixes sous `data/pit` et `runs/experiments`. |

Commande de capture des prix après la coupure D−1 08 h locale :

```powershell
python run_nyx_annual_auction_prices_source.py --delivery-day 2026-09-29 --bundle runs/live/nyx_annual_cpu/2026-09-29
```

Cette commande ne lance pas de prévision. Le reçu porte les SHA-256 des quatre
Parquet et des configurations cibles. L'état `revision_date` demandé à Saturn
n'est pas une attestation de l'heure de publication du fournisseur. Si une
heure manque, la capture échoue sans reçu complet.

## Chemins absents d'un clone propre

Le manifeste suivi `config/nyx_annual_cwe_historical.json` pointe vers douze
matrices historiques locales, à raison d'un fichier par pays dans chacune de
ces familles :

- `runs/experiments/nyx_improvement_to20260923/feature_sets/pooled_fundamentals_v1/features_{FR,DE,BE,NL}.parquet` (449 colonnes) ;
- `runs/experiments/nyx_improvement_to20260923/feature_sets/pooled_jao_refresh_v1/features_{FR,DE,BE,NL}.parquet` (503 colonnes) ;
- `runs/experiments/nyx_improvement_to20260923/feature_sets/pooled_jao_refresh_v1/compact/features_{FR,DE,BE,NL}.parquet` (123 colonnes).

Ses références historiques sont dans
`runs/experiments/nyx_improvement_to20260923/test2_confirmed_gate_v2/annual/FR.parquet`
et `runs/experiments/nyx_improvement_to20260923/rmse_exchange_composition_v1/annual/{BE,NL}.parquet`.
Leurs valeurs ne couvrent pas une livraison future. Les sources brutes que
chargent les constructeurs de recherche manquent elles aussi :
`data/pit/nyx_rmse_external_20260925/energy_charts_fr/`,
`data/pit/nyx_rmse_external_20260925/jao_initial_filtered/`,
`data/pit/nyx_rmse_external_20260926/energy_charts_de_cbpf/` et
`data/pit/nyx_rmse_external_20260926/jao_late_initial/`. La baseline locale
utilise des caches `data/cache/{fr,de,be,nl}/` et des replays sous
`runs/experiments/nyx_local_365_to20260923/`. Copier ces répertoires de
recherche ne les actualiserait pas.

## Chaîne encore nécessaire pour obtenir un vrai bundle

1. Alimenter tous les jours, à chaque coupure D−1 08 h, les covariables
   Saturn des quatre pays, JAO initial, hydro public, capacité thermique,
   échanges passés, fuel et les prix EPEX. Les neuf groupes doivent remettre
   des reçus qui lient leurs artefacts. Aujourd'hui seuls `fuel` et
   `auction_prices` ont un adaptateur dédié à ce contrat.
2. Refaire la baseline NYX `q50` pour FR, DE, BE et NL en CPU avec ses données
   disponibles à cette coupure, conserver 365 jours de points et les origines.
   `run_nyx_local_365.py` n'est pas suivi par Git ; même localement, il ne
   contacte pas Saturn et lit les caches historiques. Le flux suivi
   `NuclearKalman.ps1` est un autre challenger : ses
   sorties ne sont pas automatiquement les quatre baselines de ce contrat.
3. Prolonger les constructeurs de recherche pour fournir exactement les
   colonnes ordonnées du fichier
   `config/nyx_annual_cpu_ordered_features.json` : 449 (`pooled_fundamentals_v1`),
   503 (`pooled_jao_refresh_v1`) puis projection exacte 123. Les scripts
   `run_nyx_pooled_feature_materialization.py`,
   `run_nyx_exchange_feature_materialization.py` et
   `run_nyx_jao_late_refresh_features.py` ne sont pas suivis par Git. Ils
   sont aussi bornés à des archives et dates 2024–2026, avec chemins de
   recherche figés. Ils ne prolongeraient donc pas les matrices au lendemain
   du 23 septembre même si on les copiait tels quels.
4. Produire les quatre entrées du sélecteur de référence en plus des points
   NYX : trois HGB, Test2, et la décision `prior90_active` construite à partir
   de 90 jours OOF antérieurs. Les formules CPU autonomes sont suivies, mais
   leurs matrices et leurs historiques opérationnels manquent. Voir
   `docs/nyx_annual_reference_dependencies_20260928.md`.
5. Évaluer en chronologie la **chaîne complète** sur CPU et qualifier ses
   scores avant d'activer `forecast_enabled` dans
   `config/nyx_annual_cwe_historical.json`. L'évaluation CPU des experts avec
   référence historique figée ne valide pas à elle seule ce producteur futur.

Le consommateur `run_nyx_annual_cpu_live.py` refuse donc de publier tant que
le bundle n'a pas passé son précontrôle et que la qualification CPU n'est pas
scellée. Aucun des adaptateurs ci-dessus ne remplace une entrée absente par
zéro, par un ancien cache ou par le modèle nucléaire.
