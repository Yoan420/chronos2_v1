# SolarCorrection — conserver Chronos nucléaire, isoler l'apport du solaire

Laboratoire séparé de `Forecast.ps1` et de SolarCWE. Aucun modèle opérationnel,
aucun artefact nucléaire ni aucun résultat existant n'est remplacé.

## Expériences fixées avant les nouveaux résultats

Les quatre prévisions horaires de génération solaire FR/DE/BE/NL alimentent
le correcteur résiduel, mais **jamais le transformer**. Les prévisions Chronos
nucléaires sont chargées depuis leurs archives : quantiles, précision numérique,
dates et origines sont vérifiés à l'identique. Aucun modèle neuronal n'est chargé.

Un seul replay du correcteur est calculé pour chaque pays. Ses mêmes sorties
alimentent deux expériences :

- **A — `solar_residual_standard_kalman`** : solaire dans le correcteur ; Kalman
  utilise uniquement ses entrées nucléaires/charges résiduelles habituelles.
- **B — `solar_residual_solar_kalman`** : mêmes prévisions corrigées ; les quatre
  prévisions solaires entrent également dans Kalman.

Un troisième HTML, `solar_residual`, présente le correcteur autonome commun.
Chaque modèle est comparé à sa référence nucléaire de même famille et à Storm.
La différence B−A isole l'effet de l'ajout solaire au filtre, pas celui d'un
nouvel entraînement du correcteur. Les recettes et la gouvernance restent celles
de la référence ; aucune variante n'est choisie pays par pays après lecture des scores.

## Lancement et reprise

```powershell
& 'C:\Users\BQ6757\chronos2_v1\SolarCorrection.ps1' -Action Run -Zones FR,DE,BE,NL
& 'C:\Users\BQ6757\chronos2_v1\SolarCorrection.ps1' -Action Status
```

La livraison par défaut est le 19/09/2026. C'est un **diagnostic historique**,
sur la même année déjà consultée : ce n'est pas une nouvelle validation.
Les caches journaliers permettent de reprendre la même commande après interruption.
Sur une livraison suivante avec les mêmes entrées historiques, les corrections
compatibles sont réutilisées ; seul le nouveau jour doit être entraîné.

Si un calcul SolarCWE du même projet utilise encore le CPU, `Run -AfterSolarPid <PID>`
attend la fin de ce processus avant de démarrer. Le PID et son programme sont vérifiés ;
aucun processus n'est interrompu. `Status` affiche alors `waiting_existing_solar_run`.
Cette attente ponctuelle n'est pas un suivi automatique des livraisons suivantes.

`Prepare` fige les entrées sans entraîner ; `Audit` vérifie la présence des
références ; `Report` recharge les sorties figées sans entraîner ni accéder au
réseau. Les résultats sont sous `runs/experiments/solar_correction_v1`.

## Nouvelles journées : validation séparée et honnête

Le premier `Prepare`/`Run` crée un protocole immuable, horodaté par l'horloge réelle,
avec les recettes et les empreintes du code scientifique. Les mêmes deux candidats
sont suivis sur **FR, DE, BE et NL**. Une modification de recette interrompt la
comparaison : elle exige une nouvelle expérience, pas le remplacement du protocole.

Pour chaque nouvelle livraison, produire d'abord le forecast nucléaire avec le
processus habituel, puis lancer par exemple :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\SolarCorrection.ps1' -Action Run -DeliveryDay 2026-09-22
```

Cette commande exige l'archive nucléaire complète correspondante et ne relance
pas Chronos elle-même. Les sources solaires sont complétées dans leur cache
expérimental si nécessaire. Il faut ensuite attendre les observations :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\SolarCorrection.ps1' -Action Evaluate
```

`Evaluate` **ne recalcule aucune prévision**. Elle vérifie les prévisions scellées,
actualise par lecture Saturn les observations et Storm dans un nouvel espace de
reporting du laboratoire, puis produit un JSON de scores et de couverture sous
`validation`. Sans nouveau forecast scellé, elle ne contacte pas Saturn et renvoie
une validation en attente.

Trois ensembles ne sont jamais mélangés :

1. Historique déjà consulté, jusqu'au 19/09/2026 : diagnostic seulement.
2. Jours nouveaux mais prévisions reconstituées après le cutoff : rétrospectif.
3. Prévisions effectivement scellées après le gel de la recette et **avant
   J−1 08 h, heure civile** : suivi prospectif. Une simple origine nominale «08 h»
   dans un fichier ne suffit pas. Le fuseau/DST, la couverture physique complète
   et l'empreinte des fichiers sont vérifiés.

Pour un gel le 20 septembre au soir, la première livraison théoriquement
prospective est le **22 septembre**, sous réserve de finir le calcul avant le
21 septembre à 08 h. La livraison du 21 ne peut pas être requalifiée après coup.
Les prix de livraison d'un jour ne sont jamais injectés dans sa prédiction.

Les scores portent sur l'intersection des journées complètes des quatre pays et
des deux candidats. Le seuil de **30 journées prospectives communes** ouvre un
bilan descriptif, pas une promotion automatique ni une preuve de supériorité.
MAE, RMSE et proportion de journées gagnées en MAE sont fixées à l'avance. Les
segments de prix observés ≥200/≥300 €/MWh sont des diagnostics ex post, pas des
régimes connus au moment de la prévision. Un jour incomplet n'est pas rempli.

Les horodatages de requête Saturn as-of restent distincts d'une certification
des publications d'origine. Les empreintes locales protègent contre les changements
accidentels ; elles ne constituent pas une notarisation externe.
