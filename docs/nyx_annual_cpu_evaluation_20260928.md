# Réentraînement CPU annuel CWE — 28 septembre 2026

## Périmètre

La période évaluée va du 24 septembre 2025 au 23 septembre 2026. Les modèles
de prix sont réentraînés à chaque origine hebdomadaire sur CPU : 53 origines
pour chacun des trois experts, soit 159 fits. Les prix sont comparés à Storm
sur les 8 759 heures communes avec prix EPEX observé. Les règles de composition
par pays sont celles des rapports annuels du 23 septembre, fixées avant ce
replay : FR choisit le résiduel 1 000 arbres si son désaccord avec la référence
atteint 20 €/MWh ; BE applique la même règle à la moyenne des deux experts
2 000 arbres ; NL utilise cette moyenne sur toutes les heures.

**Portée du test :** les experts de prix sont entraînés sur CPU, mais leurs
matrices de variables, la baseline NYX `q50` et la référence de rareté
proviennent des archives historiques. Ce test ne mesure donc pas une chaîne
de prévision future entièrement reproductible depuis le dépôt. Il réutilise
la même année que les rapports historiques ; ce n'est pas une validation sur
une année indépendante. Les scores CPU portent une nouvelle identité et ne
remplacent pas les scores des modèles GPU d'origine.

## Prix horaires : résultat annuel

La porte annoncée avant le replay exige pour **chaque** pays une RMSE CPU
strictement inférieure à Storm et plus de 50 % de victoires horaires strictes
sur les mêmes 8 759 heures. Les 159 checkpoints sont présents ; le runner et
le validateur de qualification ont recalculé les scores suivants :

| Pays | RMSE CPU | RMSE Storm | RMSE GPU historique | Heures gagnées CPU | Verdict CPU |
|---|---:|---:|---:|---:|---|
| FR | 18,5965 | 18,8476 | 18,4738 | 4 382 / 8 759 (50,03 %) | Passe |
| BE | 21,2027 | 23,6096 | 21,1624 | 4 383 / 8 759 (50,04 %) | Passe |
| NL | 20,5667 | 20,7973 | 20,5532 | 4 535 / 8 759 (51,78 %) | Passe |

Le reçu de prix local est
`runs/experiments/nyx_selected_cwe_cpu_20260928/receipt.json` ; sa somme
SHA-256 est `4239966eb24359def2595809e43bc101769537260a467f05ab36d370c41e031a`.
Le reçu de qualification suivi dans Git,
`config/nyx_annual_cpu_qualification_receipt.json`, relie les 159 checkpoints,
les sorties horaires, les métriques négatives et les versions du code. Il
atteste `price_expert_replay_qualified: true` et
`negative_replay_verified: true`, mais conserve
`full_input_chain_qualified: false` et `qualified: false`.

## Probabilités de prix négatifs

Le classifieur CatBoost et sa calibration étaient déjà CPU dans la recette
historique. Le replay a réexécuté ses 53 fits par pays et reproduit, à la
précision binaire des séries, les probabilités et alertes archivées sur les
8 760 heures de chaque pays. Au seuil d'alerte de 50 % :

| Pays | Brier | Précision | Rappel | AP |
|---|---:|---:|---:|---:|
| FR | 0,023431 | 80,86 % | 64,83 % | 0,835199 |
| BE | 0,013766 | 73,05 % | 68,60 % | 0,810867 |
| NL | 0,014937 | 78,67 % | 79,03 % | 0,863513 |

Le reçu local est `runs/nyx_negative_annual_replay/fr_be_nl_20260928/receipt.json`.
Ces scores de probabilité ne valident pas à eux seuls les prix ni les données
futures.

## Conditions avant activation dans NYX

Le manifeste `config/nyx_annual_cwe_historical.json` conserve
`forecast_enabled: false`. Le consommateur annuel n'actualise pas Saturn :
il exige un bundle journalier produit et vérifié avant son lancement. La
baseline Chronos/Kalman et la référence de rareté de la recette historique
n'ont pas encore de producteur prospectif CPU qualifié dans le dépôt.

Le cache JAO de recherche ne prouve aucune capture réelle avant les coupures
des 731 partitions examinées. Les sources d'échanges retardés demandent elles
aussi des captures quotidiennes antérieures. Un clone neuf ne peut pas
recréer ces versions passées après coup. Les autres sources, les matrices
ordonnées et leurs liens cryptographiques restent à produire. Enfin, le client
Saturn dépend de `tshistory_lite`, absent de l'index Python configuré sur ce
poste, et l'accès au serveur interne n'a pas pu être testé depuis ce poste.

La branche de travail est
[`codex/nyx-regional-rmse-production`](https://github.com/Yoan420/chronos2_v1/tree/codex/nyx-regional-rmse-production).
Elle est séparée de `main`. Le panneau NYX doit présenter ces conditions et
refuser le lancement annuel tant qu'elles ne sont pas remplies.
