# Baseline NYX annuelle : dépendances CPU prospectives

La référence `scarcity_confirmed_pair` des rapports annuels exige les quatre
baselines NYX `q50` de FR, DE, BE et NL. NL utilise DE comme partenaire ;
FR et BE s'utilisent mutuellement. Le replay des **experts prix** sur CPU lit
ces courbes historiques déjà calculées. Il ne réentraîne pas cette baseline.

## Chaîne ayant créé les courbes du 23 septembre

`run_nyx_local_365.py` prépare des fichiers locaux `data/pit`, des prix
`data/cache`, puis appelle `nyx_local_chronos.py` et
`nyx_local_baseline.py`. Ces trois modules et le runner ne sont pas suivis
dans le dépôt GitHub. Leur préparation indique explicitement qu'elle
n'interroge pas Saturn. Le run historique utilisait Chronos-2 sur GPU,
un correcteur CatBoost GPU Bayesian de 700 arbres, puis un Kalman gouverné
sur 365 jours. Les quelques suffixes manquants des archives météo ont été
estimés et audités dans ce replay rétrospectif ; ils ne sont pas une source
Saturn prospective.

Le moteur Chronos local accepte `device="cpu"` et utilise alors
`torch.float32`. Cette possibilité ne suffit pas au clone : le code exige
un snapshot local de `amazon/chronos-2` à la révision
`29ec3766d36d6f73f0696f85560a422f50e8498c` avec
`local_files_only=True`. Aucun poids n'est suivi dans Git. Le correcteur
CPU est une recette numérique distincte de la variante GPU historique.
Le replay Chronos exige en outre les prix réalisés de **toutes** les heures
livrées pour valider sa sortie, et `nyx_local_baseline._prepare` exige un
`raw_history.actual` fini même sur la dernière journée avant de retirer ce
champ du fit. Ces fonctions ne peuvent donc pas être appelées telles quelles
pour une journée future dont le prix n'est pas encore connu.

Le code suivi `nyx_live_baseline.py` exécute un flux isolé SolarWind/Test2
à partir d'une archive nucléaire déjà capturée. Il ne synchronise pas les
douze courbes Chronos de la baseline annuelle et n'émet pas les quatre
`baseline/{FR,DE,BE,NL}.parquet` du bundle CWE. Le flux suivi
`NuclearKalman.ps1` est également un challenger nucléaire différent.

## Contrat vérifiable du producteur futur

`inspect_nyx_annual_nyx_quantiles.py` vérifie un éventuel producteur CPU :

```powershell
python inspect_nyx_annual_nyx_quantiles.py --bundle runs/live/nyx_annual_cpu/2026-09-29 --delivery-day 2026-09-29
```

Il attend dans le bundle `source_receipts/nyx_quantiles.json`, quatre
`baseline/<pays>.parquet`, quatre `baseline_audits/<pays>.json` et douze
reçus `baseline_runs/<pays>/{chronos,residual,kalman}.json`. Le reçu principal
doit lier tous ces fichiers par
SHA-256, annoncer le protocole `nyx_annual_cpu_nyx_quantiles_v1` et la
recette CPU exacte déclarée dans
`chronos2_hourly/nyx_annual_nyx_quantiles_gate.py`. Chaque audit pays
doit lier les reçus des runs Chronos, correcteur et Kalman, dire que la baseline a
été réentraînée sur CPU et qu'aucune prédiction GPU archivée n'a été
réutilisée. Chaque courbe doit couvrir exactement les 365 jours passés et
la journée future en heures UTC, conserver `q10 ≤ q50 ≤ q90`, afficher les
prix passés et aucun prix futur, et porter pour **chaque jour local** une
origine exactement D−1 à 08 h Europe/Paris. Les transitions été/hiver sont
testées.

Ce contrôle vérifie les déclarations et l'intégrité des artefacts ; il ne
prouve pas l'heure de publication effective de Saturn ni la qualité des
scores. Son absence doit maintenir le lancement annuel bloqué. Un reçu
générique `nyx_quantiles` ou une simple colonne `q50` ne prouve pas que le
producteur est celui des rapports.

## Travail nécessaire avant l'activation

1. Porter le producteur historique dans Git, supprimer ses dates et chemins
   d'archives figés, fournir les quatre prix et les douze séries de
   covariables depuis des captures Saturn à l'état D−1 08 h. Chaque
   journée doit garder son reçu source ; une valeur publiée plus tard ne
   peut pas être injectée dans un ancien point OOF.
2. Installer le snapshot Chronos-2 et ses dépendances sur le poste CPU,
   puis faire produire `q10/q50/q90` prospectifs et une histoire de 365
   jours pour les quatre pays, avec reçus Chronos/correcteur/Kalman liés
   aux fichiers du bundle. Il faut un chemin de score sans étiquette future
   pour Chronos et la baseline, plus la vérification que la journée future
   n'a pas participé aux fits. Le correcteur CPU et son Kalman doivent être
   réentraînés dans l'ordre du temps.
3. Brancher le validateur de ce document sur le précontrôle et le
   consommateur annuels, puis reconstruire les HGB, Test2, `prior90` et la
   référence sur ces nouveaux q50. Le simple replay prix CPU avec les
   q50 et la référence GPU historiques ne qualifie **pas** cette chaîne.
4. Refaire une évaluation chronologique **de bout en bout sur CPU**,
   comparer Storm sur les mêmes heures et sceller un reçu de qualification
   de la chaîne entière avant de changer `forecast_enabled`.
