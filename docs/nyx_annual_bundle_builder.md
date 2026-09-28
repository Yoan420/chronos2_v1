# Matrices annuelles CPU et vérification des sources

Le constructeur `chronos2_hourly/nyx_annual_cpu_bundle_builder.py` matérialise
les douze matrices FR/DE/BE/NL : 449 variables, 503 variables, puis la projection
ordonnée de 123 variables. Il assemble séparément les deux branches JAO. Une
matrice historique de 449 colonnes n'est jamais tirée de la matrice 503 rafraîchie.

Les calculs purs historiques sont réutilisés sans date ni répertoire de recherche
dans le chemin prospectif : prix retardés, NYX, covariables Saturn, voisinage/fuel,
calendrier, hydro, profils, Test2, thermique et échanges. Un manque prévu par une
recette reste `NaN` avec son indicateur de disponibilité ; aucune donnée absente
n'est remplacée arbitrairement par zéro.

## Ordre d'exécution

1. Les collecteurs remplissent `source_artifacts` et lient leurs fichiers aux
   reçus `source_receipts`. Les origines Saturn et les historiques de prix de
   chaque journée sont conservés dans le bundle.
2. Le producteur NYX CPU écrit les quatre quantiles historiques et prospectifs
   dans `sources/nyx_quantiles`, ainsi que les quatre fichiers `baseline`.
3. Le constructeur écrit les douze matrices et les entrées de référence :

   ```powershell
   python .\build_nyx_annual_cpu_bundle.py --bundle runs/live/nyx_annual_cpu/2026-09-30 --delivery-day 2026-09-30 --action features
   ```

4. Le producteur de référence consomme
   `reference_inputs/{base292,augmented334,test2}/{FR,DE,BE,NL}.parquet`, entraîne
   les composants CPU et écrit `reference` avec son reçu.
5. Le bundle complet est scellé :

   ```powershell
   python .\build_nyx_annual_cpu_bundle.py --bundle runs/live/nyx_annual_cpu/2026-09-30 --delivery-day 2026-09-30 --action seal
   ```

Ces commandes ne collectent aucune source et n'activent aucun modèle. Elles
échouent si leurs entrées liées aux reçus manquent. Les fichiers existants différents
sont refusés. Le sceau lie tous les reçus, artefacts, transformations et résultats.

## Contrats importants

- Les prix utilisés pour les variables retardées sont ceux disponibles à
  l'origine propre à chaque journée. Une révision du prix historique provoque
  le recalcul de la journée concernée depuis son snapshot Saturn.
- La normalisation Test2 utilise les jours précédents de covariables prévues.
  Les heures de production restent distinctes lors des transitions de fuseau.
- Les quantiles NYX commencent au moins 469 jours avant la livraison. Les
  entrées 292/334/Test2 couvrent les 462 jours nécessaires aux fenêtres HGB et
  aux 90 jours OOF, avec sept jours de quantiles supplémentaires pour les erreurs
  retardées. Les covariables commencent au moins 834 jours avant la livraison.
- La journée prévue n'a aucun prix réalisé. La présence d'un tel prix dans les
  entrées du constructeur est une erreur.
- Les quantiles et covariables portent leurs origines explicites. Le constructeur
  ne déduit pas une preuve de disponibilité de la date de livraison.

## Validation des vrais artefacts

`validate_source_packet(bundle, delivery_day)` dans
`chronos2_hourly/nyx_annual_source_validation.py` relit les sources brutes et
recalcule leurs sorties :

- états Saturn quotidiens, profils assemblés et origines ;
- cohérence des prix avec le snapshot Saturn courant ;
- toutes les captures JAO initiales et les 28 descripteurs ;
- captures hydro/échanges archivées dans les ZIP, avec recalcul exact ;
- treize capacités thermiques, états D−1 propres à chaque journée et 36 variables ;
- artefact fuel et chronologie des sources pour chaque heure.

Un bundle JAO complet contient désormais les quatre fichiers de chacun des
366 jours dans `source_artifacts/jao_initial/captures`. Le lecteur peut donc
reconstruire l'historique sans le cache original du poste personnel.

Le validateur distingue la vérification des états « as of » et des captures
réellement faites avant la coupure de l'heure de première publication du
fournisseur. Cette dernière n'est pas certifiée par ces données. La qualification
statistique de la chaîne CPU demeure un contrôle séparé.

## Vérifications réalisées

Le test de parité sur archives locales compare exactement les douze matrices
de 17 880 heures aux matrices des rapports historiques, types inclus. Il est
ignoré sur un clone sans ces archives. Les autres tests couvrent une livraison
future sans label, la journée de 25 heures, les branches JAO distinctes, les
révisions de prix, les cutoffs, la reconstruction depuis les archives portables
et le rejet de fichiers dont les valeurs ont été modifiées même après rehashage.
