# NYX sur le poste de travail — contrat de déploiement non satisfait

Le raccourci NYX ouvre `NYX.pyw` puis `experiment_console`. Son lancement actuel
exécute `NuclearKalman.ps1`, synchronise les sources Saturn nécessaires à ce
modèle, puis publie ses rapports. Ce chemin ne produit pas les entrées des
experts RMSE récents ni les probabilités de prix négatif.

## Sélection historique demandée

La sélection retient les meilleurs résultats annuels connus au 27 septembre
2026, calculés sur 8 759 heures communes avec Storm :

| Pays | Point retenu | RMSE | RMSE Storm | Victoires |
|---|---|---:|---:|---:|
| FR | `residual__disagreement20__w1p0` | 18,473773 | 18,847597 | 4 384 |
| DE | `boosting_2000_mean_disagreement20` | 21,142357 | 20,306584 | 4 774 |
| BE | `boosting_2000_mean_disagreement20` | 21,162364 | 23,609583 | 4 385 |
| NL | `boosting_2000_mean_all` | 20,553187 | 20,797265 | 4 519 |

DE ne bat pas Storm en RMSE. Les choix sont rétrospectifs. Le sélecteur pur
`chronos2_hourly/nyx_regional_live_selection.py` reproduit bit à bit les
points historiques enregistrés : 10 944 points FR et 8 760 points DE/BE/NL.
Il ne produit pas les experts en entrée.

La colonne `reference` du sélecteur vient de `scarcity_confirmed_pair` sur
l'année évaluée. Elle dépend des prévisions Test2, d'une probabilité de pic et
d'une politique utilisant les erreurs des 90 jours antérieurs. Les quantiles
`nuclear_kalman` que publie la console NYX ne sont pas cette référence. Aucun
de ces fichiers de référence sous `runs/experiments` n'est suivi par Git.

Le détecteur de prix négatifs est un modèle CPU séparé. Il exige 123 variables
compactes ordonnées, 365 jours de prix passés et la fenêtre de livraison.
Il n'a actuellement ni collecte continue ni branchement dans la console NYX.
Ses scores historiques Brier sur 8 760 heures par pays sont : FR 0,023431,
DE 0,013846, BE 0,013766 et NL 0,014937. Ces scores concernent la
probabilité, pas l'erreur de la prévision de prix.

## Entrées et tests nécessaires avant activation

1. Produire pour chaque livraison les 123 variables compactes et les 503
   variables complètes, avec leurs 365 jours d'historique, leurs colonnes dans
   le même ordre, leurs indicateurs de disponibilité et leurs coupures D−1 08 h.
   Le prix NYX q50 historique et courant, la référence de composition,
   JAO, hydraulique, échanges, thermique, combustible et prévisions voisines
   doivent conserver leur provenance. Une valeur absente reste absente : la
   remplacer silencieusement par zéro changerait le modèle.
2. Fournir un entraînement compatible avec le poste sans GPU et évaluer cette
   **nouvelle** recette chronologiquement. Un premier fit CPU de 1 000 arbres
   a réussi ; la simple compatibilité technique n'établit pas les scores GPU.
3. Produire et auditer les prix et probabilités négatives sur une nouvelle
   livraison, avec des reçus, hachages, horaires UTC et sorties par pays.
4. Ajouter un lanceur Windows avec précontrôle, synchronisation Saturn et des
   autres sources requises, puis adapter `experiment_console/adapters.py`,
   `manager.py`, `primary_progress.py`, `primary_results.py` et le frontend.
   Les publications nucléaires actuelles doivent garder leur identité.
5. Tester le chemin complet dans un environnement équivalent au poste de
   travail avant de retirer d'anciennes options ou du code partagé.

Le dépôt GitHub public ne suit aucun fichier sous
`runs/experiments/nyx_improvement_to20260923`. Les modèles CBM, matrices,
références et reçus sont seulement locaux. La liste ordonnée des 123 colonnes
est elle aussi dans `.cache`, ignoré par Git. Les collecteurs JAO, hydro,
échanges et thermique de recherche sont bornés à des dates passées. Le sync
Saturn nucléaire ne remplace pas leurs producteurs prospectifs. Ne pas
supprimer ces archives locales : Git ne peut pas les restaurer.

L'arbre de travail comporte également des modifications suivies et plusieurs
centaines de fichiers non suivis d'autres travaux. Aucun `git clean` global ni
suppression du code historique n'est sûr avant un inventaire des dépendances et
une sauvegarde vérifiée.
