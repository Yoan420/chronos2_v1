# Proposition de révision

Conserver le titre et la spécialisation du mémoire sur les séries temporelles. Remplacer les textes ci-dessous, maintenir les chapitres 2 à 5 et leurs références, puis actualiser le sommaire. La contribution reste une synthèse critique accompagnée de quatre protocoles proposés. Aucun résultat expérimental original n’est revendiqué.

# Résumé

Ce mémoire analyse les Transformers et les modèles de fondation pour la prévision des séries temporelles. Il relie les choix de représentation, les mécanismes d’attention, le préentraînement et la construction des sorties aux conditions du transfert vers de nouvelles séries. L’étude compare des architectures spécialisées et plusieurs familles préentraînées, puis examine la contamination des évaluations, les changements de distribution, l’adaptation et la dépendance entre horizons. La littérature conduit à considérer conjointement l’architecture, les informations accessibles et le protocole d’évaluation : une bonne précision moyenne ne suffit pas à établir une généralisation robuste. Quatre hypothèses expérimentales portent sur la prévision prospective, les covariables, l’adaptation limitée et les événements multi-horizons. Le travail constitue une revue critique et une proposition méthodologique ; les résultats cités proviennent des publications et aucune des expériences proposées n’a été réalisée dans ce mémoire.

Mots-clés : séries temporelles ; Transformers ; modèles de fondation ; préentraînement ; généralisation ; prévision probabiliste ; évaluation.

# Abstract

This dissertation analyses Transformers and foundation models for time series forecasting. It connects input representations, attention mechanisms, pretraining and output construction to the conditions under which models transfer to new time series. The review compares specialised architectures and pretrained model families, then examines evaluation contamination, distribution shifts, adaptation and dependence across forecast horizons. The literature motivates a joint assessment of architecture, available information and evaluation design: strong average accuracy alone does not establish robust generalisation. Four experimental hypotheses address prospective forecasting, covariate quality, limited adaptation and multi-horizon events. This work is a critical literature review and a methodological proposal. All reported results originate from cited publications; none of the proposed experiments has been conducted as part of this dissertation.

# 1 Introduction

## 1.1 Sujet et problématique

Prévoir une série temporelle demande d’extraire des régularités d’un historique sans supposer qu’elles resteront inchangées. Une saisonnalité peut évoluer, une variable devenir indisponible et une relation entre mesures disparaître. Les modèles de fondation cherchent à réutiliser des connaissances acquises sur de nombreuses séries, parfois sans mise à jour de leurs paramètres sur la tâche cible. Cette ambition de transfert motive leur étude, mais ne constitue pas une garantie de généralisation.

Le Transformer fournit un mécanisme flexible de mise en relation des observations. Son comportement dépend toutefois de ce que représentent les tokens, des interactions autorisées et de l’objectif d’apprentissage. Des valeurs discrétisées, des segments continus et des variables entières ne décrivent pas le même problème au réseau. De même, prédire des quantiles par horizon et produire des trajectoires conjointes répond à des besoins différents.

La problématique est donc la suivante : **quels choix de représentation et de préentraînement permettent un transfert utile vers de nouvelles séries, et quels protocoles permettent de l’établir ?** Elle articule l’analyse des mécanismes à celle des preuves empiriques. Un gain peut provenir d’une meilleure architecture, mais aussi d’informations supplémentaires, d’une proximité avec le corpus appris ou d’un budget plus élevé.

L’évaluation doit tenir compte de la dépendance temporelle et des informations disponibles à chaque origine de prévision. Elle doit aussi distinguer la réutilisation de connaissances sur des archives du transfert vers des observations réellement futures. Ces distinctions guideront l’analyse des résultats publiés et la formulation des expériences proposées.

## 1.2 Démarche et contribution

La revue sélectionne des méthodes illustrant des choix distincts de représentation, d’architecture et d’apprentissage. Pour chaque famille, elle suit les entrées, les calculs, l’objectif et les sorties. Les chapitres 2 et 3 exposent les mécanismes et leurs adaptations temporelles ; le chapitre 4 étudie le préentraînement ; le chapitre 5 analyse les limites des évaluations. Le chapitre 6 transforme ces limites en hypothèses réfutables.

La contribution réside dans cette grille de comparaison et dans les contrôles expérimentaux proposés. Les articles d’origine sont privilégiés ; les rapports techniques sont distingués des publications évaluées. Le corpus couvre les travaux consultés jusqu’au 9 septembre 2026. La sélection est centrée sur la prévision temporelle et ne constitue pas une revue systématique exhaustive. Ce travail théorique et bibliographique ne présente aucune nouvelle architecture validée ni aucun résultat expérimental original.

# 6 Pistes de recherche et expériences proposées

## 6.1 Cadre commun

Les quatre hypothèses suivantes restent à tester. Avant l’accès au test, on fixerait les versions et empreintes des modèles, les partitions, les pertes principales, les budgets et les règles de décision. Les transformations seraient ajustées exclusivement sur les données admissibles. Les résultats seraient présentés par tâche puis agrégés à poids égal, avec analyses par domaine et régime.

Des intervalles à 95 % reposeraient sur un rééchantillonnage apparié de groupes de séries et de blocs temporels ; leur longueur serait choisie sur le développement. Des corrections pour les comparaisons confirmatoires multiples seraient prévues. Les marges proposées ci-dessous devraient être justifiées avant le test. Un intervalle recouvrant une limite serait non concluant : l’absence de réfutation ne vaut pas confirmation.

## 6.2 H1 L’avantage zero-shot persiste après le gel du modèle

On comparerait un modèle préentraîné gelé à une référence statistique sélectionnée puis figée sur le développement. Les observations seraient collectées après le gel, dans au moins trois domaines et à plusieurs fréquences. Chaque prévision serait horodatée avant réception de sa cible. Les modèles partageraient séries, dates, horizons et informations accessibles.

Pour chaque tâche, le gain serait g = 1 − Lmodèle/Lréférence, avec traitement séparé des pertes de référence nulles. Le critère principal serait le gain prospectif moyen. H1 serait soutenue si sa borne inférieure était positive, réfutée dans ce périmètre si sa borne supérieure était négative ou nulle, et non concluante sinon. La collecte suivrait toute la durée prévue. La comparaison avec des archives renseignerait la stabilité, sans attribuer automatiquement une baisse à la contamination.

## 6.3 H2 Les covariables apportent un gain malgré une qualité imparfaite

Un même modèle serait évalué avec la cible seule, des covariables correctes, retardées ou permutées par blocs. Une variable aléatoire servirait de contrôle négatif. Le conditionnement natif serait distingué d’une correction résiduelle apprise avant le test. Toute covariable future devrait être effectivement connue à l’origine ; sa présence dans un historique complet ne suffirait pas.

Les critères proposés seraient un gain d’au moins 2 % avec les bonnes variables et une dégradation d’au plus 5 % sous perturbation, relativement à la cible seule. H2 serait soutenue si les intervalles respectaient simultanément ces deux exigences, et réfutée si la borne supérieure du gain restait sous 2 % ou la borne inférieure de la dégradation dépassait 5 %. Un gain avec les contrôles négatifs imposerait d’examiner redondance, régularisation et fuite d’information.

## 6.4 H3 Une adaptation limitée réduit le coût sans perte excessive

À modèle initial et données identiques, on comparerait gel des paramètres, LoRA, correction résiduelle et ajustement complet. Une architecture compacte spécialisée compléterait les références. Les régimes seraient présentés successivement, avec retours à des régimes anciens. Après chaque adaptation, l’évaluation porterait sur la nouvelle tâche et sur un ensemble fixe de tâches antérieures.

On mesurerait temps total, mémoire maximale et, si possible, énergie, en incluant les recherches d’hyperparamètres. Les comparaisons à données identiques et à calcul identique seraient séparées. La non-infériorité proposée autoriserait au plus 2 % de perte supplémentaire face à l’ajustement complet ; l’oubli serait limité à 5 % relativement à l’état précédent. Les pertes nulles seraient traitées séparément. H3 exigerait que les bornes supérieures restent sous les deux marges et qu’un avantage de coût soit établi. Elle serait réfutée si une borne inférieure dépassait sa marge ou si le coût était clairement supérieur ou égal. Le nombre de paramètres entraînables ne remplacerait pas la mesure.

## 6.5 H4 La dépendance améliore la prévision des événements multi-horizons

On comparerait indépendance, copule estimée sur les résidus historiques et dépendance d’un générateur natif. Pour isoler le couplage, les scénarios seraient transformés en rangs puis reconstruits sur une grille commune de quantiles par horizon. Les sorties natives non recalées seraient évaluées séparément. Une simulation à marges fixées vérifierait le protocole avant les données publiques.

Trois événements seraient définis avant le test : dépassement au moins une fois, dépassement persistant et somme supérieure à un seuil. Le critère principal serait la diminution absolue moyenne du score de Brier face à l’indépendance, à poids égal entre événements. Un score multivarié et des diagnostics de calibration compléteraient l’analyse. H4 serait soutenue par un gain dont la borne inférieure est positive, réfutée si sa borne supérieure est négative ou nulle, ou si une dégradation après changement de régime est établie. Le nombre de scénarios serait identique ; temps et stockage seraient mesurés.

## 6.6 Portée du programme

H1 à H4 évalueraient les limites discutées dans la revue : généralisation temporelle, qualité des covariables, compromis de l’adaptation et dépendance des prévisions. Le programme serait progressif : reproduction des comparaisons, vérification des contrôles, puis tests prospectifs. Les conclusions resteraient limitées aux modèles, tâches et régimes étudiés. Des résultats négatifs préciseraient eux aussi les conditions dans lesquelles une méthode plus simple reste préférable.

# 7 Conclusion

Cette revue montre que l’analyse des Transformers temporels doit relier les représentations aux informations accessibles et aux sorties attendues. Le choix des tokens organise les interactions ; le préentraînement détermine les régularités rencontrées ; le protocole précise ce que mesure réellement un score. Une bonne performance ne suffit donc pas à identifier le mécanisme du transfert.

Les limites étudiées conduisent à privilégier des comparaisons contrôlées : données réellement futures, covariables disponibles à l’origine, budgets explicités et adaptation évaluée avec ses effets sur les tâches anciennes. Pour les prévisions probabilistes, les marges et la dépendance doivent être distinguées lorsque la décision porte sur plusieurs horizons.

Les quatre hypothèses proposées constituent un programme à réaliser. Elles visent à déterminer si les gains persistent sur des observations futures, résistent à une dégradation des covariables et restent utiles après adaptation. Elles interrogent également la valeur des dépendances entre horizons pour des événements définis avant le test. Leurs critères distinguent un résultat favorable, une réfutation dans le périmètre étudié et une expérience non concluante.

La contribution de ce mémoire est une synthèse critique du fonctionnement des modèles de fondation temporels et une démarche expérimentale explicite pour en étudier les limites. Les résultats cités appartiennent aux travaux examinés ; les protocoles proposés restent à mettre en œuvre. Cette distinction permet de formuler des questions de recherche précises sans présenter les possibilités suggérées par la littérature comme des résultats acquis.

# Références à ajouter
