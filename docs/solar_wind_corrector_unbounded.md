# SolarWind : correcteur résiduel sans plafonds

Test autorisé le 22 septembre 2026, **à lancer seulement après la fin
vérifiée des calculs reuse DE et NL**. Il s'agit d'une nouvelle expérience
isolée, pas d'une modification des calculs actifs ni d'une promotion.

## Changement scientifique limité

Deux variantes utilisent les corrections brutes CatBoost déjà calculées :
la recette baseline sans interaction et la recette avec interaction dans
le correcteur uniquement. Pour chacune, le nouveau décalage est appliqué
tel quel à q10, q50 et q90, sans borne inférieure ou supérieure. Une correction
de +120 €/MWh reste +120 et une correction de −90 €/MWh reste −90.
L'échelle reste un ; les trois quantiles reçoivent le même décalage.
Les valeurs non finies, grilles incompatibles et croisements de quantiles
ne sont pas corrigés silencieusement : ils bloquent le calcul.

L'absence de plafond concerne **uniquement le correcteur résiduel CatBoost**.
La recette, les covariables, fenêtres et protections du Kalman original ne
changent pas. Kalman est rejoué sur les deux nouveaux amonts ; aucun plafond
ou garde-fou Kalman n'est retiré. L'interaction n'entre toujours pas dans ses
covariables.

Chronos, les entrées et les résultats CatBoost sont figés. Il n'y a aucun
nouveau fit CatBoost ni nouvelle recherche de paramètres. La différence
testée est le retrait des bornes de correction, y compris de la borne
négative : ce n'est donc pas seulement un test de plafond positif supérieur
à +80 €/MWh. Les corrections extrêmes et les erreurs sur les pics doivent
être examinées, sans présumer que le déplafonnement améliore les résultats.

## Préalable impératif et sources scellées

Les deux donneurs sous `solar_wind_corrector_reuse_v1/2026-09-22` sont fixés :

- DE : `48d7c14fd8bc2e4f` ;
- NL : `8f246d6bbd54b511`.

Les deux pays doivent présenter une fin annuelle complète, une identité
exacte, 365 jours et 8 760 heures évaluées, 24 heures de forecast, ainsi que
les **24 artefacts attendus avec leurs SHA256 exacts**. La fin de DE seul
n'autorise pas à commencer le nouveau calcul. Un rapport HTML présent,
un statut isolé ou un nombre de fichiers ne suffit pas.

Tant que ce préalable manque, la vérification reste en lecture seule :
aucun worker, fit, verrou d'exécution ou dossier de résultats nouveau.
Une altération d'identité, d'empreinte, de grille ou de contrat bloque le
test ; elle ne justifie pas de redémarrer sous une autre identité.

Les corrections brutes des deux recettes sont importées pour tout le
calendrier requis, y compris l'historique de préparation, les 365 jours
évalués et le forecast séparé : **731 checkpoints journaliers par pays**.
Chaque checkpoint doit conserver sa grille,
son jour, ses corrections, ses audits et sa chaîne de provenance. Les
jours dont la baseline a été reconstruite strictement à l'intérieur des
bornes gardent explicitement cette preuve ; ils ne sont pas présentés comme
de nouveaux fits ni comme une reproduction brute bit à bit de CatBoost.
Les fichiers donneurs ne sont ni modifiés, ni déplacés, ni supprimés.
Les deux audits de recette, les compteurs de dépassement et la provenance
quotidienne sont comparés à `corrector_audit.json`, lui-même scellé parmi
les 24 artefacts. Cette cohérence est une preuve de provenance, pas une
inversion des valeurs saturées : une sortie plafonnée à +80 ne permet pas
de distinguer une correction brute de +100 d'une correction brute de +120.
Les valeurs brutes extrêmes restent fondées sur les checkpoints quotidiens
et leurs reçus, puis sont épinglées dans l'identité du nouveau calcul.

## Nouveau calcul, reprise et fin

Le namespace distinct est
`runs/experiments/solar_wind_corrector_unbounded_v1/2026-09-22/`.
L'identité contient les sources scellées, leurs empreintes, le nouveau
coordinateur/lanceur et la signature d'exécution. Les autres moteurs actifs
restent inchangés.

Les workers ne publient pas directement de fichiers ; le parent assemble
chronologiquement et scelle les résultats. Le rejeu Kalman conserve les
365 jours passés disponibles, sans prix réalisé du jour prédit ou du
forecast futur. Les jours de changement d'heure gardent leurs grilles UTC
physiques, sans fusion de doublons locaux.

Une reprise vérifie les reçus journaliers Kalman, les SHA256, les indices,
la causalité et le forecast terminal avant de réutiliser le résultat.
Une interruption transitoire ne doit pas faire perdre les journées déjà
scellées. Une pause volontaire ne doit jamais être annulée automatiquement.
Une mémoire insuffisante cesse les nouvelles soumissions et draine les
tâches engagées ; elle n'autorise pas à tuer les workers pour forcer une
reprise. Les verrous interdisent les batchs concurrents.

La fin exige, pour **les deux variantes et les deux pays**, les 365 jours
du 22/09/2025 au 21/09/2026, les 8 760 heures évaluées et les 24 heures du
22/09/2026 traitées séparément, ainsi que tous les audits, rapports et
artefacts du manifeste final. Les compteurs d'étape et l'ETA ne constituent
pas une preuve d'achèvement.
Le manifeste de cette nouvelle expérience contient **17 artefacts** :
l'identité, l'audit des sources, l'audit des corrections sans bornes, les
métriques, le rapport, les deux séries baseline scellées, puis cinq fichiers
par variante (amonts historique/futur, résultats historique/futur, audit
Kalman). Les 24 artefacts des donneurs et les 17 du nouveau résultat ne sont
pas interchangeables.

L'étude reste rétrospective et post-hoc. Les limites héritées ne changent
pas : publication d'origine PIT non certifiée, substitutions NL, warm-up
et préfixe avant calibration. Aucune modification de production, aucune
certification prospective et aucune promotion automatique.
