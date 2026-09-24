# SolarWind : réutilisation vérifiée de la correction baseline

Optimisation autorisée le 22 septembre 2026, après le lancement parallèle.
Le protocole [correcteur](solar_wind_corrector_interaction.md) et son
[ordonnancement parallèle](solar_wind_corrector_parallel.md) restent inchangés :
DE/NL, Chronos figé, interaction exclusivement dans CatBoost, recette MAE,
fenêtres quotidiennes [D−365,D), plafonds [−40,+40] et [−40,+80], Kalman rejoué
sans nouvelle covariable. L'année d'évaluation est le 22/09/2025–21/09/2026 ;
le forecast du 22/09/2026 est séparé. Aucun apprentissage supplémentaire de
paramètres, synchronisation des sources ou changement de production.

## Pourquoi une partie du calcul peut être évitée

La baseline applique le même décalage aux trois quantiles :

`quantile corrigé = quantile Chronos + clip(correction brute, −40, +40)`.

Avec l'échelle de correction baseline égale à un, un décalage strictement
intérieur aux deux bornes identifie la correction brute. Le coordinateur peut
donc reconstruire `q50 corrigé − q50 Chronos`, puis vérifier que ce même
décalage reproduit également q10 et q90, à une tolérance absolue de 1e−9.
Cette opération ne constitue pas un nouvel ajustement CatBoost.
La soustraction en virgule flottante n'est pas présentée comme une
reproduction bit à bit des valeurs brutes internes à l'ancien modèle :
l'équivalence numérique et l'erreur de reconstruction sont auditées.

Une marge de sécurité de 1e−6 €/MWh exclut la proximité des deux bornes :
toutes les heures du jour doivent respecter `−40+1e−6 < correction < 40−1e−6`.
Si une seule heure touche une borne ou sa marge, l'inversion est ambiguë :
par exemple, une correction observée de +40 peut provenir de +40, +60 ou
+100. Le jour doit alors être recalculé avec la recette originale. Cette
précaution est indispensable pour la variante dont le plafond monte à +80.

La correction avec interaction continue d'être ajustée quotidiennement pour
chaque jour non déjà scellé. Réutiliser la baseline ne remplace jamais ce fit,
ne copie pas un modèle d'un autre jour et ne raccourcit pas sa fenêtre.

## Contrôles et traçabilité

La réutilisation exige les mêmes entrées Chronos et artefacts baseline
scellés, la recette originale, les grilles horaires UTC exactes et la
provenance quotidienne. Des valeurs infinies, trous, doublons, quantiles
incompatibles ou une provenance invalide ne sont pas réparés silencieusement.
Un démarrage à froid ne peut être interprété comme un modèle ajusté : sa
provenance et la correction nulle attendue doivent rester explicites.

L'audit distingue les valeurs reconstruites de la baseline, les ajustements
de secours et les prédictions déjà calculées puis importées. Les données
réalisées du jour prédit ne servent pas à reconstruire une correction.
Les vérifications des origines temporelles sont conservées ; elles ne
certifient pas rétrospectivement la publication des sources.

Le nouveau namespace est
`runs/experiments/solar_wind_corrector_reuse_v1/2026-09-22/`.
Son identité explicite inclut celle du protocole d'origine, les empreintes de
son nouveau coordinateur et de son lanceur, les paramètres d'exécution et
la règle de réutilisation. Les moteurs et sorties séquentiels/parallèles
précédents restent immuables.

Les checkpoints parallèles autorisés sont prioritaires, puis les checkpoints
séquentiels. Tout import conserve les corrections brutes et leur audit,
vérifie identité, jour, grille, checksum, recette et provenance d'origine,
et enregistre les empreintes des fichiers donneurs. Il ne modifie ni ne
supprime les fichiers sources. Un fichier non scellé ou altéré n'est pas une
preuve de calcul réutilisable.
La chaîne séquentielle imbriquée est revérifiée lorsqu'un checkpoint
parallèle provient déjà d'un import. Cette migration est limitée aux donneurs
restés au stade correcteur ; elle refuse un donneur ayant commencé Kalman.
Pour les jours reconstruits de la variante `cap_80`, les quantiles baseline
scellés sont conservés directement, bit à bit : la soustraction suivie d'une
addition ne doit pas introduire un arrondi inutile dans l'amont de Kalman.

## Exécution, arrêt et fin

Les deux threads par fit, les deux à quatre workers, la réserve mémoire et
l'écriture atomique exclusivement par le parent sont conservés. Les résultats
peuvent arriver dans le désordre ; leur assemblage est chronologique. Les
verrous des trois namespaces empêchent de faire travailler simultanément les
anciens et le nouveau batch. Une demande d'arrêt cesse les nouvelles
soumissions et draine les tâches engagées, sans effacer les checkpoints.
Le remplacement atomique d'un fichier tolère les erreurs Windows temporaires
5, 32 et 33, avec huit tentatives au maximum et moins de trois secondes
d'attente cumulée. Les autres erreurs sont propagées immédiatement. En cas
d'échec persistant, ni l'ancien fichier ni les checkpoints déjà scellés ne
sont supprimés ; le fichier temporaire reste disponible pour le diagnostic.

Une fin exige les 365 jours, 8 760 heures évaluées, 24 heures de forecast,
toutes les variantes, les rapports et **24 artefacts scellés** : les
23 du coordinateur parallèle plus `reconstruction_audit.json`.
Un état de processus, un nombre de fichiers ou une année partielle ne suffit
pas. L'estimation de durée reste incertaine et dépend du nombre de jours
réellement réutilisables et du coût du rejeu Kalman.

```
python -B -u run_solar_wind_corrector_reuse.py --action validate --zones DE NL --threads 2 --workers 4
python -B -u run_solar_wind_corrector_reuse.py --action run --zones DE NL --threads 2 --workers 4
```

`validate` doit rester en lecture seule, sans fit ni création de sorties.
Le lancement utilise le Python `pricefm311`, une fenêtre cachée, une priorité
Normal et des journaux distincts. Avant la bascule, les contrôles comparatifs
et l'arrêt vérifié du batch précédent restent nécessaires.

Les limites d'interprétation restent celles du test d'origine : hypothèse
post-hoc, publication PIT non certifiée, substitutions NL, warm-up et préfixe
avant calibration hérités. Cette optimisation ne transforme pas l'étude en
validation prospective et n'autorise aucune promotion automatique.
