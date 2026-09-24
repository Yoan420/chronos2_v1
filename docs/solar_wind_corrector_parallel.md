# SolarWind : migration parallèle du même test correcteur

Accélération autorisée le 22 septembre 2026. Le protocole scientifique reste
celui de [l'expérience correcteur](solar_wind_corrector_interaction.md) :
DE/NL, Chronos figé, interaction uniquement dans CatBoost, bornes [−40,+40]
ou [−40,+80], perte MAE et apprentissages quotidiens sur [D−365,D).
Kalman conserve ses covariables et sa recette ; il est rejoué sur les nouveaux
amonts. Les dates d'évaluation restent du 22/09/2025 au 21/09/2026, avec le
forecast du 22/09/2026 séparé. Aucune recherche de paramètres supplémentaire,
nouvelle source, synchronisation, promotion ou modification de production.

## Ce qui change : l'ordonnancement uniquement

Le coordinateur `solar_wind_corrector_parallel_v1` distribue des jours
indépendants à un nombre borné de processus. Chaque ajustement CatBoost garde
**deux threads**, exactement comme le calcul séquentiel. Le pool accepte de
**deux à quatre workers**, sans lancer simultanément un second batch du même
test. Le budget maximal est donc de quatre ajustements et huit threads de
calcul ; le nombre de jobs soumis reste borné. Le débit commence à deux tâches
simultanées et augmente seulement après des résultats et si la mémoire garde
une marge suffisante. Passer sous le seuil de mémoire disponible arrête les
nouvelles soumissions, draine les tâches engagées puis met le batch en pause,
sans tuer les tâches ni reprendre automatiquement.

Les workers calculent et renvoient leurs résultats. Seul le parent écrit les
checkpoints, audits et résultats, de façon atomique. Les jours terminent
éventuellement dans le désordre ; l'assemblage des séries est toujours
chronologique, sans trou, doublon ni changement de la grille physique UTC.
Cette parallélisation ne raccourcit pas les fenêtres d'entraînement et ne
réutilise pas le modèle d'un jour postérieur pour un jour antérieur.

## Identité distincte et migration vérifiée

Les résultats séquentiels restent immuables sous
`runs/experiments/solar_wind_corrector_interaction_v1/2026-09-22/`.
Le nouveau calcul écrit exclusivement sous
`runs/experiments/solar_wind_corrector_parallel_v1/2026-09-22/`.

La nouvelle identité contient l'identité complète `Prepared` de l'ancien
test, les empreintes du coordinateur et de son lanceur, et la signature
d'exécution parallèle. Cette nouvelle identité est déclarée explicitement :
il ne s'agit pas de présenter un nouveau lanceur comme une reprise invisible
de l'ancien batch. Les fichiers scientifiques d'origine restent inchangés.

L'import d'un checkpoint séquentiel exige le chemin source autorisé, son
identité exacte, le jour de livraison, la grille horaire attendue, des valeurs
finies et l'intégrité SHA256. Les corrections brutes originales et avec
interaction sont conservées sans changement ; les plafonds s'appliquent
ensuite de la même façon. Le nouveau checkpoint conserve une provenance
d'import et l'empreinte du fichier source dans `migration_audit.json`.
Un checkpoint mal identifié, altéré, incomplet ou provenant d'un autre dossier
est refusé. Aucun checkpoint source n'est modifié, déplacé ou supprimé.

La bascule exige d'abord l'arrêt vérifié du batch séquentiel et de ses
descendants. Les unités déjà scellées sont importables ; une unité interrompue
avant son scellement est recalculée. Les journaux précédents sont conservés.
Les validations et contrôles du nouveau coordinateur précèdent son lancement.

## Arrêt, reprise et preuve de fin

Une demande d'arrêt gracieux cesse les nouvelles soumissions, laisse les
tâches déjà engagées se terminer et scelle leurs checkpoints. Elle ne marque
pas le calcul terminé. Une erreur conserve les unités déjà validées et les
diagnostics ; elle n'autorise ni contournement des validations ni suppression
des checkpoints. La reprise utilise les mêmes identités et vérifie les reçus.

Une fin par pays exige toutes les variantes, les 365 jours/8 760 heures et
les 24 heures de forecast, les audits et rapports attendus : **23 artefacts
scellés**, soit les 22 artefacts du calcul correcteur plus l'audit de migration.
Les indices, prix réalisés, sorties Chronos et largeurs/ordre des quantiles
sont contrôlés. Aucun compteur de processus ou de fichiers ne suffit à prouver
la fin. Le moniteur affiche les processus actifs, les étapes réelles et une
ETA explicitement incertaine.

## Commandes et limites d'interprétation

```
python -B -u run_solar_wind_corrector_parallel.py --action validate --zones DE NL --threads 2 --workers 4
python -B -u run_solar_wind_corrector_parallel.py --action run --zones DE NL --threads 2 --workers 4
```

Utiliser le Python `pricefm311`, lancement en arrière-plan, fenêtre cachée,
priorité **Normal** et journaux stdout/stderr uniques ; jamais High ni Realtime.
Le lancement accéléré utilise quatre workers, deux threads chacun et un seuil
de mémoire disponible de 3 GiB. `validate` est en
lecture seule, sans entraînement ni création des sorties. L'égalité des
calculs séquentiels et parallèles doit être contrôlée avant la migration ;
un contrôle échantillonné ne certifie pas à lui seul tous les jours futurs.

Les limites PIT et statistiques ne changent pas : publication d'origine non
certifiée, substitutions NL et warm-up hérités, initialisation du préfixe
avant calibration explicitement nulle, hypothèse post-hoc issue des rapports.
Les résultats restent exploratoires et ne déclenchent aucune promotion.
