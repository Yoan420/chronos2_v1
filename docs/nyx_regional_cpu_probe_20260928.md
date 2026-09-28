# NYX régional CPU — contrôle exploratoire du 28 septembre 2026

La nouvelle recette à 77 variables et 250 arbres CatBoost CPU a été rejouée
hors ligne sur des archives locales de prix et de comparaison Storm, ainsi que
sur les banques Saturn PIT présentes dans le dépôt. Ce contrôle utilisait 53
réentraînements hebdomadaires. La recette désormais codée confirme chaque jour
avec un nouveau fit ; ses scores devront être mesurés sur le poste relié à
Saturn. Les nombres ci-dessous **ne qualifient pas** le lancement.

La sélection s'étend du 24 septembre 2025 au 5 mai 2026. La confirmation
exploratoire couvre les 3 384 heures du 6 mai au 23 septembre 2026. Les prix
sont comparés à Storm sur les mêmes heures et à la même observation.

| Pays | Prix choisi sur sélection | RMSE CPU confirmation | RMSE Storm | Heures gagnées | Taux gagné |
|---|---|---:|---:|---:|---:|
| FR | `blend50` | 27,816 | 19,101 | 1 078 | 31,86 % |
| DE | `absolute` | 31,875 | 21,951 | 874 | 25,83 % |
| BE | `blend50` | 36,385 | 31,548 | 960 | 28,37 % |
| NL | `blend50` | 32,967 | 24,455 | 956 | 28,25 % |

Pour `P(prix < 0)`, la sortie brute obtient le meilleur Brier sur la sélection
dans les quatre pays. Son Brier de confirmation reste meilleur que la fréquence
historique utilisée comme témoin, mais la sortie calibrée fait mieux sur cette
période en DE, BE et NL. Le choix n'a pas été ajusté après confirmation.

| Pays | Brier brut sélection | Brier brut confirmation | Brier calibré confirmation | Fréquence historique confirmation |
|---|---:|---:|---:|---:|
| FR | 0,017700 | 0,037025 | 0,036751 | 0,089297 |
| DE | 0,010252 | 0,032879 | 0,025303 | 0,075872 |
| BE | 0,009304 | 0,035512 | 0,026227 | 0,045488 |
| NL | 0,013081 | 0,032580 | 0,025364 | 0,067503 |

Le prix CPU échoue nettement aux critères RMSE et taux de victoires contre
Storm pour FR, BE et NL. La configuration reste `pending_backtest` et le bouton
de prévision régionale reste verrouillé. Les meilleurs scores GPU historiques
sont conservés dans `docs/nyx_workstation_deployment_contract_20260928.md` ;
ils ne sont pas attribuables à cette recette CPU.

Un contrôle ponctuel de 1 000 et 2 000 arbres sur la seule livraison du 6 mai
2026 a amélioré BE mais dégradé ou peu changé FR et NL. Une journée ne permet
pas de modifier la recette ni de déduire ses scores annuels ; la configuration
à 250 arbres reste celle du présent candidat.

Provenance locale non incluse dans Git :
`runs/experiments/nyx_local_365_to20260923` et
`.cache/nyx_regional_cpu_validation_20260928/results`. Les banques Saturn
utilisées sont sous `data/pit/nuclear_forecast/`. Aucune synchronisation Saturn
live ni validation du poste professionnel n'a eu lieu sur cette machine.
