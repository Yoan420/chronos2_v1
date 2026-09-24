# NYX complet à 15 minutes : protocole expérimental

Premiers résultats mesurés : [analyse provisoire du 17 septembre 2026](research/NYX_FULLQUARTERHOUR_2026-09-17.md). La chaîne native est complète ; le témoin complet horaire reste en cours de reprise au moment de cet instantané.

La demande porte sur toute l'architecture opérationnelle : **Chronos-2 → correcteur résiduel CatBoost → Kalman gouverné**, chaque étape calculée au pas de 15 minutes. La prévision ponctuelle horaire est ensuite la moyenne arithmétique de quatre points. Les résultats de chaque étape sont conservés pour mesurer ce qu'ajoutent les corrections.

Ce protocole est fixé avant les nouveaux scores. Les fichiers opérationnels, leur configuration, les caches de production et les rapports publiés ne sont pas modifiés.

## Comparaisons

- **NYX actuel**, référence complète archivée pour la livraison du 16/09/2026.
- **Chaîne NYX horaire sur historique commun**, recalculée avec les mêmes dates d'apprentissage que le candidat.
- **Chaîne NYX native à 15 minutes**, avec apprentissage CatBoost et assimilation Kalman des vrais résidus de chaque quart.
- Ablations Chronos seul et Chronos + CatBoost, aux deux fréquences.

Les sept familles sont évaluées sur les mêmes heures, pour BE, DE, FR et NL, du **18/06/2026 au 15/09/2026** (90 jours). Ce test reste exploratoire : cette période a été consultée dans les expériences précédentes. Aucun réglage n'est choisi selon ses résultats.

## Historique effectivement disponible

Les archives natives couvrent le 01/10/2025 au 15/09/2026, soit 350 journées. Le contexte Chronos de 2 048 heures impose de commencer ses prévisions préquentielles le **26/12/2025**, avec 8 192 quarts de contexte. Le premier jour testé dispose donc de **174 jours** de prévisions et de résidus antérieurs ; cette durée atteint 263 jours au dernier jour de test.

Le plafond d'apprentissage de NYX reste de 365 jours civils, mais ne peut être atteint avec cette archive. **Il s'agit du port de toute l'architecture avec un historique plus court, pas d'une reproduction de 365 jours natifs inexistants.** Le témoin horaire a la même restriction. La référence NYX actuelle conserve son historique plus long ; sa différence avec le témoin permet de contextualiser cette restriction, sans lui attribuer exclusivement tout l'écart : le témoin est aussi recalculé sur les observations rétrospectives.

Le début préquentiel conserve une période de démarrage : avant 30 journées complètes et 720 heures physiques (2 880 quarts), CatBoost laisse explicitement la sortie Chronos inchangée. Ces jours sont audités et entrent dans l'historique de Kalman, comme les démarrages déclarés du moteur existant. Aucun démarrage ne subsiste dans la période évaluée.

## Fidélité des composants

### Chronos-2

Poids locaux gelés `amazon/chronos-2`, sans téléchargement ni fine-tuning. Contexte horaire de 2 048 points et contexte natif de 8 192 points ; horizon civil 23/24/25 heures ou 92/96/100 quarts. Inférence CPU float32, huit threads, batch matériel 64, seed 42, sans apprentissage entre pays (`cross_learning=False`).

Le schéma opérationnel est reproduit : **19 covariables historiques et 13 futures**. Les six fondamentaux figurent sous leurs alias historiques et sous leurs noms de prévisions connues ; sept calendriers complètent les entrées. Les historiques et futurs horaires reconstruits ont été comparés exactement, en float32, aux fichiers préparés des quatre pays du bundle 16/09. Les six alias historiques sont numériquement identiques à leurs versions `known_*_oracle` dans ces archives. `oracle` est ici le nom interne existant d'une prévision fondamentale archivée, pas un prix futur réalisé.

Les fondamentaux disponibles restent horaires et sont constants dans chaque heure. Les prix sont de vrais quarts natifs ; les calendriers utilisent une heure fractionnaire. Les trois quantiles Chronos sont recalculés et conservés pour **toutes** les dates d'apprentissage et de test : l'ancien essai avait seulement sauvegardé q50 et ne peut servir de substitut.

### CatBoost

La recette de chaque pays provient de l'audit archivé de NYX : perte MAE, 700 arbres, profondeur 6, taux 0,03, L2 de 15, seed 42, ordre temporel conservé et correction commune plafonnée à ±40 €/MWh. Chaque jour D, un nouveau modèle apprend uniquement sur les jours antérieurs, dans la limite de D−365 à D−1. Aucun modèle horaire n'est repris au quart d'heure.

Les familles de variables opérationnelles sont conservées : fondamentaux, calendriers et jours fériés, quantiles et dispersions de Chronos, profils journaliers et interactions. Les 16 retards/statistiques de prix exclus par NYX ne sont jamais nécessaires à la reconstruction. Les colonnes annuelles sont exclues comme dans NYX. Les rampes d'une et deux heures portent sur quatre et huit quarts ; les profils d'une journée n'utilisent que les prévisions connues de cette journée.

### Kalman

Les cinq candidats opérationnels sont conservés : biais, harmonique, marché, échelle linéaire et échelle UKF. Six fondamentaux et agrégats de charge résiduelle suivent la configuration archivée. Chaque D, les filtres sont réinitialisés et rejoués sur les jours antérieurs disponibles, plafonnés à 365. Variance d'observation et normalisations sont estimées uniquement sur cette fenêtre passée.

La transition d'état et les paramètres Q/persistance restent **quotidiens** : les diviser par quatre serait une modification injustifiée du mécanisme actuel. Les observations sont assimilées à la fréquence native, après gel de la prévision de leur journée. Le jour D n'est jamais assimilé avant sa prévision. La gouvernance reste sur 60 journées, au moins 14, avec un plafond de correction à ±20 €/MWh.

Le recalibrage de variance et des normalisations sur la fenêtre antérieure à D reproduit la production. Les prévisions internes reconstruites pour la gouvernance ne constituent donc pas une validation imbriquée intégrale ; aucune observation de D n'entre néanmoins dans ce calcul.

## Temporalité, scores et décision

Les prix day-ahead de D−1 jusqu'à la fin de D−1 sont admis selon la convention du moteur NYX au cutoff civil D−1 à 08:00. Aucune cible D n'entre dans Chronos, CatBoost ou Kalman. L'archive de prix est rétrospective et ne certifie pas le vintage de publication de chaque observation ; les limites des vintages fondamentaux sont celles du bundle existant.

La cible horaire commune est la moyenne des quatre prix natifs. MAE, RMSE et biais sont calculés globalement, par pays, heure et régime. Incertitude appariée : 1 000 bootstrap de blocs communs de sept jours, seed 20260916. Les q95 des régimes élevés sont les seuils figés sur les labels NYX avant le 14/03/2026, comme dans l'essai précédent.

Critères de poursuite : gain MAE d'au moins 2 % contre NYX actuel **et** le témoin complet horaire, IC du delta MAE entièrement négatif, RMSE au plus 1 % moins bonne, MAE de chaque pays au plus 5 % moins bonne ; les régimes négatifs et élevés suffisamment renseignés ne doivent pas se dégrader de plus de 5 %. Aucun score ne déclenche une promotion automatique.

Les corrections translatent ensemble q10/q50/q90 sans changer leur ordre. Leur moyenne horaire n'est pas présentée comme un nouvel intervalle calibré : les comparaisons principales portent sur les prévisions ponctuelles.

## Exécution et reprise

```powershell
& 'C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe' run_nyx_fullquarterhour.py
```

Le CLI accepte `--stage raw`, `--stage postprocess` et `--resume <dossier>`. Les données, recettes, versions logicielles, poids et code sont figés par empreintes. Chaque journée produit un checkpoint vérifié. Les deux phases sont scellées séparément : le portage des corrections peut être vérifié pendant le calcul des bases, mais sa recette est verrouillée avant tout apprentissage sur les données du marché.

Sorties exclusivement sous `runs/experiments/nyx_fullquarterhour_v1`. Tout quart manquant, doublon, sortie non finie, fuite de cible ou checkpoint altéré fait échouer la phase ; aucune substitution silencieuse par NYX n'est admise.

## Vérifications avant calcul réel

Les tests vérifient la parité horaire avec la production, les calendriers et rampes, les jours DST, la causalité par perturbation des valeurs futures, la durée physique minimale, les plafonds, la reconstruction des entrées et l'intégrité des reprises. Des benchmarks synthétiques ont mesuré environ 53 secondes pour quatre origines Chronos natives, 13 secondes pour un CatBoost de 180 jours et 27 secondes pour un Kalman de 180 jours. Ces mesures servent au dimensionnement informatique et ne sont pas des résultats prédictifs.
