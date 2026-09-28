# Hydro et capacité thermique pour les entrées CWE prospectives

## Capacité thermique Saturn

`run_nyx_annual_thermal_source.py` peut être lancé sur un clone qui contient les
fichiers suivis du dépôt et dispose de l'accès Saturn utilisé par NYX :

```powershell
python run_nyx_annual_thermal_source.py --delivery-day 2026-09-29 --bundle runs/live/nyx_annual_cpu/2026-09-29
```

La date est dynamique. Le collecteur prend les 365 jours civils d'entraînement
et le jour de livraison. Pour chacune des 13 séries Pmax retenues, il compare
la réponse `block_staircase` à **366 requêtes individuelles** avec
`revision_date` égal à la coupure civile D−1 08 h propre à chaque jour. Une
réponse réseau en erreur, une divergence, une valeur négative ou une origine
incorrecte arrête la publication. Les jours sans valeur dans les deux réponses
restent `NaN`, avec masque `0` ; un zéro fourni garde son masque `1`.

Le lot contient treize Parquet horaires de capacité, les quatre blocs de 36
colonnes ordonnés comme la matrice CWE 503, et
`source_receipts/thermal_capacity.json`. Ce reçu lie les 17 Parquet par SHA-256,
les noms des séries, les 366 états par série, les jours manquants et le code de
collecte. `training_window_complete` atteste la vérification des **requêtes**
sur tous les jours, pas une capacité finie pour chaque jour. La capacité reste
une prévision Pmax journalière du périmètre fournisseur, diffusée sur les
23/24/25 heures physiques ; la couverture nationale et les heures de
publication du fournisseur ne sont pas certifiées.

Sur un clone vide, la première collecte réalise 4 758 requêtes individuelles
plus treize extractions de contrôle. Sa durée et l'authentification Saturn sur
le poste professionnel n'ont pas été mesurées. Les états sont conservés dans
`data/pit/nyx_annual_thermal`, par série et journée, avec leur coupure, leur contrat
et leurs empreintes de code. La relance d'un même jour réutilise les états
vérifiés ; le lendemain demande seulement **13 nouvelles requêtes individuelles**.
Les treize extractions `block_staircase` restent effectuées à chaque passage et
comparées à tous les états de la fenêtre. Une altération du cache ou une révision
incompatible bloque la collecte. `--cache <dossier>` permet de choisir son
emplacement. Les quatre blocs thermiques
ne sont qu'une partie des matrices de 449/503 variables exigées par NYX ; ce
collecteur ne lance aucune prévision.

## Hydro public français

Le collecteur historique `collect_nyx_public_hydro.py` est un fichier de
recherche local. Il lit l'API Energy-Charts par mois, avec
des dates et un répertoire de sortie fixés jusqu'au 24 septembre 2026. Les
réponses locales portent `generated_at` au moment de la collecte du 25
septembre, après les origines de l'évaluation annuelle. Le constructeur
`nyx_fr_hydro_lagged_features.py` est désormais livré pour réutiliser exactement
les mêmes formules sur les captures prospectives. Son retard de 48
heures prouve que l'intervalle physique utilisé précède l'origine ; il ne
prouve pas que la valeur révisée était publiée à cette origine.

Le collecteur prospectif livré est `run_nyx_annual_hydro_source.py`. Son action
`capture` archive la réponse brute avant la coupure D−1 08 h, ainsi que les huit
valeurs calculées et leurs indicateurs. Son action `assemble` exige les captures
des 366 jours de la fenêtre, les revalide, puis lie leurs archives au bundle.
Le validateur de chaîne peut extraire ces archives et refaire les calculs sans
le cache du poste personnel.

La présence du collecteur ne crée pas les captures passées. Sans historique
admissible, l'assemblage reste bloqué. Recopier les réponses rétrospectives de
recherche ne prouve pas leur disponibilité à l'époque.
