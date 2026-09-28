# Évaluation complète et activation CPU FR / DE / BE / NL

Le rejeu des experts sur les anciennes matrices GPU reste une **évaluation
conditionnelle**. Il ne peut pas activer la production. Le nouvel évaluateur
consomme les sorties des nouveaux producteurs CPU, pour chaque journée,
avec les preuves des données disponibles à son propre cutoff.

## Données attendues

- Un bundle complet par date dans `runs/evaluation_inputs/YYYY-MM-DD/` :
  toutes les sources et leurs preuves, les quatre baselines CPU, les douze
  matrices de variables et les quatre références `scarcity_confirmed_pair`.
- Quatre fichiers `runs/evaluation_comparisons/FR.parquet`, `DE.parquet`,
  `BE.parquet`, `NL.parquet`, indexés par heure physique UTC. Colonnes
  `actual` (prix officiel observé) et `storm`. La grille doit correspondre
  exactement à la période évaluée. Une seule valeur Storm manquante est tolérée.
  `comparisons_receipt.json` doit attester les exports officiels et leurs
  snapshots source. L'exporteur `export_nyx_annual_comparisons.py` le produit ;
  quatre fichiers Parquet ajoutés manuellement ne suffisent pas.

Les fichiers de comparaison ne sont jamais fournis aux modèles. Ils sont
empreintés lors du gel du plan, puis lus pour calculer les scores seulement
après le scellement de toutes les prévisions.

## Commandes PowerShell

Depuis la racine du dépôt, avec l'environnement CPU installé :

```powershell
$python = (Resolve-Path .\.venv-annual\Scripts\python.exe).Path
& $python .\export_nyx_annual_comparisons.py --output runs/evaluation_comparisons --first-day 2025-09-24 --stop-day-exclusive 2026-09-24
& $python .\evaluate_nyx_annual_cpu_full_chain.py plan --output runs/evaluations/annual_cpu_complete --bundles runs/evaluation_inputs --comparisons runs/evaluation_comparisons --first-day 2025-09-24 --stop-day-exclusive 2026-09-24
& $python .\evaluate_nyx_annual_cpu_full_chain.py predict --output runs/evaluations/annual_cpu_complete
& $python .\evaluate_nyx_annual_cpu_full_chain.py score --output runs/evaluations/annual_cpu_complete
& $python .\qualify_nyx_annual_cpu.py --full-chain-evaluation runs/evaluations/annual_cpu_complete --preflight
& $python .\qualify_nyx_annual_cpu.py --full-chain-evaluation runs/evaluations/annual_cpu_complete --activate
```

Ces dates illustrent la période des rapports historiques. Elles ne rendent pas
les archives de recherche certifiées : la première commande refuse les bundles
absents ou sans preuves adéquates. Une autre période de **365 jours consécutifs**
est acceptée, à condition de disposer des données et comparaisons correspondantes.
Les requêtes Saturn à une date passée sont distinguées des archives prouvant la
première publication fournisseur.

L'action `predict` entraîne chaque jour les trois experts prix et quatre
classifieurs, conformément au consommateur de production. Les jours déjà
scellés sont vérifiés et réutilisés. `--max-days 1` permet de commencer par une
seule journée ; une évaluation incomplète ne peut pas activer la production.
Une interruption conserve les fichiers partiels dans `attempts/`. Relancer la
même commande recalcule seulement le jour interrompu et les jours suivants ;
aucun jour n'est publié avant son scellement complet.

## Critères d'activation

- 365 journées consécutives, avec toutes les heures UTC, y compris les journées
  de 23 et 25 heures.
- Données brutes, transformations, entraînements des baselines et références,
  sorties, paramètres et versions de bibliothèques vérifiés.
- FR, BE et NL : RMSE strictement inférieur à Storm et victoire stricte sur plus
  de 50 % des heures communes.
- DE : mêmes contrôles de données et de calculs ; l'écart de performance contre
  Storm est accepté explicitement par l'utilisateur et reste visible dans les
  métriques et les reçus des prévisions.
- Probabilités négatives : scores réellement recalculés sur toutes les heures,
  bornes et couvertures vérifiées pour les quatre pays.

`--activate` recalcule la qualification à partir des preuves. Il conserve
l'ancien reçu, installe le nouveau reçu et son empreinte, puis active le
manifeste. Modifier seulement `forecast_enabled` ne permet pas d'activer le
modèle. Un changement ultérieur de code, de paramètres ou de versions rend la
qualification invalide.
Les empreintes de code normalisent uniquement les fins de ligne LF/CRLF pour
permettre le même checkout sous Windows. Les données, modèles, prévisions et
reçus restent vérifiés par leurs empreintes binaires exactes.

## Rejeu DE conditionnel indépendant

Les 159 anciens fits CPU n'avaient conservé aucune prédiction DE ni modèle CBM.
Le complément DE exige donc deux experts × 53 origines, soit 106 fits :

```powershell
& $python .\run_nyx_de_cpu_historical.py --output runs/experiments/nyx_de_cpu_20260929 --threads 8
& $python .\run_nyx_negative_annual_replay.py --countries DE --output runs/experiments/nyx_negative_cpu_DE_20260929
```

Ce complément conserve maintenant les modèles et les sorties des quatre pays.
Il n'écrase pas les anciens scores FR/BE/NL et ne qualifie pas à lui seul les
nouvelles baselines ou la collecte prospective.
