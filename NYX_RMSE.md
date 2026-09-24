# NYX RMSE — laboratoire indépendant

Ce candidat complète les prévisions **nuclear_kalman** archivées. Il ne remplace aucun modèle et n'appelle jamais `Forecast.ps1`. Les commandes opérationnelles restent inchangées.

## Utilisation

```powershell
& 'C:\Users\BQ6757\chronos2_v1\NyxRMSE.ps1' -Action Run
```

Le premier `Run` prépare les données locales, exécute le backtest puis génère le rapport HTML. Les suivants réutilisent les calculs scellés et produisent un nouveau rapport. Les fits hebdomadaires terminés sont sauvegardés individuellement : après interruption, relancer la même commande reprend sans réentraîner les folds valides.

```powershell
# Lire l'état et vérifier les fichiers
& '.\NyxRMSE.ps1' -Action Status
# Régénérer seulement le rapport
& '.\NyxRMSE.ps1' -Action Report
# Après modification de config/nyx_rmse.yaml : nouvelle expérience isolée
& '.\NyxRMSE.ps1' -Action Prepare
& '.\NyxRMSE.ps1' -Action Run
```

`-Action Backtest` est un alias de `Run`. `-RunDirectory` permet de désigner explicitement un snapshot. `-Config` sélectionne une autre recette ; `-PythonExecutable` un environnement compatible. `-DryRun` affiche les arguments sans action. Les chemins sont indépendants du répertoire courant ; les exemples relatifs supposent d'être dans le projet.

Les sorties restent exclusivement dans `runs/experiments/nyx_rmse_v1/`. `latest.json` désigne le dernier rapport terminé, `latest_prepared.json` la dernière expérience préparée. Une interruption pendant un rapport n'efface pas les résultats calculés. Un checkpoint corrompu ou une modification de code, de runtime ou de source provoque un refus explicite : ne pas modifier ses SHA manuellement.

## Méthode

La cible est l'erreur **observé − NYX**. Un apprentissage à perte quadratique estime son espérance conditionnelle ; minimiser la MSE minimise aussi la RMSE sur le même échantillon. Contrairement à une correction limitée aux pics haussiers, ce résidu peut être positif ou négatif.

Deux méthodes sont comparées, chacune directe et gouvernée :

| Identifiant | Rôle |
|---|---|
| `nyx_rmse` | Candidat principal : résidu MSE, poids gouverné par pays |
| `residual_mse_direct` | Même régression, correction directe diagnostique |
| `mixture_mean_governed` | Espérance des deux régimes conditionnels, poids gouverné |
| `mixture_mean_direct` | Même espérance, correction directe diagnostique |

Le régresseur MSE utilise le gradient boosting histogramme, avec 80 itérations par défaut. L'alternative conserve les paramètres des forêts conditionnelles et les probabilités du détecteur précédent : espérance = seuil × [(1−p) × moyenne régime normal + p × moyenne régime pic]. L'espérance des feuilles est calculée directement par `forest.predict`, sans reconstruire toute la CDF ni inverser ses quantiles à chaque heure. Cette variante supprime aussi les anciennes portes d'intervention exclusivement haussières : la comparaison à l'ancien P50 n'isole donc pas seulement le choix moyenne/médiane.

Les fondamentaux, pays, cutoffs, folds entraînés et disponibilités du détecteur source sont conservés. Les forêts et la régression ne voient pas les prix électriques en variables explicatives ; le prix intervient dans la base NYX et dans la cible résiduelle. Aucun nouveau téléchargement ou calcul neuronal n'est requis.

La gouvernance teste les poids fixes 0, 0,25, 0,5 et 1 sur les propositions réellement émises dans les 90 jours précédents, avec uniquement les labels alors disponibles. Elle demande un gain de MSE après une pénalisation d'incertitude calculée par jour et refuse les dégradations de MAE sur l'ensemble et sur les heures ordinaires. Le poids zéro conserve NYX. Une correction est plafonnée symétriquement à ±400 EUR/MWh avant pondération. Ces contrôles historiques ne garantissent pas la non-régression future.

## Paramètres modifiables

Tout se règle dans `config/nyx_rmse.yaml` : complexité et régularisation du régresseur, nombre de threads (maximum deux), plafond, fenêtre et seuils de gouvernance. La forêt de contrôle conserve délibérément les paramètres historiques. Aucun tuning automatique sur l'année évaluée ni sélection a posteriori d'une règle différente par pays.

## Comparabilité et limites

- L'évaluation porte sur les mêmes 365 jours civils que la source figée. Le jour prospectif suivant reste séparé, même si son observé est déjà disponible. Les métriques utilisent les heures communes, avec gestion explicite des journées de 23/25 heures.
- **365 jours évalués ne signifie pas 365 jours d'apprentissage avant chaque prévision.** La source ne fournit pas une année de chauffe supplémentaire. Les premières périodes retombent sur NYX, exactement comme l'expérience source. Le cœur d'entraînement exclut en outre les 28 derniers jours réservés au détecteur. `require_full_training_history: true` refuse de publier un candidat si aucun fold n'a l'historique exigé.
- Les points MSE visent une moyenne, pas un **P50**. Les anciens P10/P90 appartiennent à NYX ; aucune nouvelle couverture probabiliste n'est revendiquée et la moyenne peut sortir de cet intervalle.
- L'EVA reprend la politique déjà utilisée : portefeuille total de 100 MW, 25 MW par pays, référence observée de la veille à heure civile identique, seuil et coûts inchangés. Cette référence n'est pas un prix exécutable : il s'agit d'un diagnostic hypothétique, pas d'un P&L négociable.
- Les vintages historiques des prévisions NYX/Storm de cette source ne sont pas certifiés PIT. Le correcteur respecte la chronologie des labels sans améliorer rétroactivement cette certification.
- L'année a déjà été analysée. Tout gain obtenu ici reste exploratoire et doit être confirmé sur de nouvelles journées avant une éventuelle production. **Aucune promotion automatique n'existe.**

## Premier résultat — 15 septembre 2026

Backtest achevé sur le 15/09/2025–14/09/2026, FR/DE/BE/NL : 34 940 heures-pays communes, dont 20 924 avec expert prêt. Les 32 folds entraînés sont sauvegardés ; les 21 autres conservent la base. Les chiffres NYX, Storm et anciens P50 sont identiques au KPI antérieur.

| Annuel ALL | MAE EUR/MWh | RMSE EUR/MWh | Gain EVA simulée vs NYX |
|---|---:|---:|---:|
| NYX actuel | 11,36750 | 21,87553 | 0 EUR |
| `nyx_rmse` gouverné | 11,37360 | 21,88009 | −2 242,94 EUR |
| Résiduel MSE direct | 12,45990 | 22,92121 | −141 677,25 EUR |
| Moyenne du mélange gouvernée | 11,37014 | 21,87746 | −1 427,88 EUR |

**Essai non concluant : ne pas remplacer la production.** La gouvernance limite la dégradation, sans l'annuler. Le candidat principal intervient seulement en DE sur 743 heures (347 améliorées, 396 dégradées). Il ne modifie aucune prévision des 90 derniers jours : les spikes récents ne sont donc pas améliorés. Les versions directes corrigent trop d'heures sans signal assez généralisable. Aucun réglage n'a été modifié après lecture de ces résultats pour fabriquer un gain sur cette année.

Contrôles : 137 tests réussis, un test de permission des liens symboliques Windows ignoré ; identité exacte des 96 arbres du dernier fit avec les CDF antérieures ; contrôles des dates de disponibilité des labels ; filtres et thèmes vérifiés dans un navigateur headless isolé. Relecture d'un backtest terminé sans fit : environ 0,5 s hors démarrage Python et rapport sur cette machine.
