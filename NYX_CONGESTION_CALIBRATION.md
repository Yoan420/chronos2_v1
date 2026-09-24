# NYX Congestion — calibration chronologique à 90 jours

Ce laboratoire indépendant teste une calibration plus stable du **correcteur de prix aval**. Il ne modifie ni la production, ni l'ancien laboratoire congestion, ni les experts CNEC et régionaux déjà entraînés. Leurs signaux historiques OOF restent gelés et vérifiés par SHA.

## Commandes

```powershell
# Voir la commande sans créer de fichier ni entraîner
& 'C:\Users\BQ6757\chronos2_v1\NyxCongestionCalibration.ps1' -Action DryRun

# Nouveau snapshot, indépendant de l'ancien laboratoire
& 'C:\Users\BQ6757\chronos2_v1\NyxCongestionCalibration.ps1' -Action Prepare

# Réentraîner uniquement l'aval, backtester et produire le rapport
& 'C:\Users\BQ6757\chronos2_v1\NyxCongestionCalibration.ps1' -Action Run

# Vérifier l'état ou refaire seulement le rendu
& 'C:\Users\BQ6757\chronos2_v1\NyxCongestionCalibration.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\NyxCongestionCalibration.ps1' -Action Report
```

La configuration est `config/nyx_congestion_calibration.yaml`. Toutes les sorties restent sous `runs/experiments/nyx_congestion_calibration_v1`. `-RunDirectory` cible explicitement un snapshot pour `Run`, `Backtest`, `Status` ou `Report`. `Backtest` suit la même chaîne que `Run`.

Une reprise n'utilise que les checkpoints liés au même code, aux mêmes sources et aux mêmes paramètres. Si une entrée a changé, préparer une nouvelle expérience ; ne pas réécrire les sceaux de l'ancienne. Aucune collecte externe n'est lancée et aucun expert amont n'est réentraîné ici.

## Pourquoi cette variante ?

Dans l'expérience précédente, une calibration courte pouvait bloquer le correcteur faute d'événements positifs distincts, même si l'expert régional émettait un signal. La nouvelle hypothèse n'est pas de forcer une correction : elle est de conserver un calibrateur régularisé lorsque le bloc récent contient peu ou aucun positif, tout en maintenant les autres contrôles chronologiques.

La fenêtre de 90 jours et les pénalités sont fixées avant le replay de cette expérience. Elles ne sont pas choisies à partir des scores du 14 septembre.

## Découpage temporel précis

Pour une livraison D, seules les observations et leurs labels admissibles au cutoff **D−1 à 08 h Europe/Paris** peuvent être employés, dans les 365 jours passés au maximum.

| Composant | Données utilisées |
| --- | --- |
| Classifieur de risque | CORE antérieur à D−90 |
| Seuil d'erreur extrême par pays | Estimé dans ce même CORE, jamais dans CAL |
| Calibrateur de probabilité | Scores du classifieur sur les 90 derniers jours calendaires, non vus par ce classifieur |
| Distribution de sévérité / CDF | Passé antérieur à D−28, en conservant les seuils CORE90 |

La CDF peut donc utiliser les **62 premiers jours de CAL** pour conserver de l'information récente sur les amplitudes. Ses sorties ne sont jamais employées pour ajuster le calibrateur de probabilité. Le bloc CAL est séparé de l'entraînement du classifieur, **mais pas de l'entraînement de tous les composants de la chaîne**. Ce n'est pas un holdout intégral de 90 jours pour le modèle complet. Le respect des disponibilités et l'évaluation future restent indispensables.

Le principe d'apprendre le calibrateur sur des scores issus de données distinctes de celles ayant entraîné le classifieur est rappelé dans la [documentation officielle de calibration scikit-learn](https://scikit-learn.org/stable/modules/calibration.html). La variante hiérarchique régularisée de ce laboratoire est une extension empirique spécifique ; cette référence ne prouve pas sa fiabilité en présence de classes rares ou absentes.

## Calibration hiérarchique monotone

À partir de la probabilité brute `p`, on ajuste :

`p_calibrée = sigmoid(a × logit(p) + b + décalage_pays)`.

L'objectif est la **somme** des log-loss de CAL, plus une pénalité quadratique qui rapproche `a` de 1, `b` de 0 et les décalages pays de 0. Pénalités fixes : pente **10**, intercept **2**, pays **20**. La pente est contrainte à être non négative.

Le calibrateur partage ainsi de l'information entre pays tout en limitant les ajustements propres à chacun. La monotonie conserve le sens des scores dans un pays ; une pente nulle peut créer des ex æquo. Elle ne garantit pas un classement inchangé entre différents pays.

Ce sont des estimations ponctuelles régularisées, **pas des intervalles bayésiens d'incertitude**, ni une garantie de calibration. Les états `regularized_supported`, `regularized_sparse_support` ou `not_fitted` restent affichés. La régularisation n'invente pas des événements positifs ; un pays avec très peu d'événements fournit une preuve limitée.

## Contrôles toujours requis

- Au moins **118 jours calendaires** d'historique.
- Au moins **90 jours passés admissibles par pays**, dont **28 jours CORE** et **14 jours CAL** par pays.
- CORE et sévérité : au moins **30 observations de chaque classe** et **cinq dates positives distinctes**.
- Respect des disponibilités historiques, des signaux OOF, des partitions figées et de la convergence de l'optimisation.

Les minima concernent des dates avec des observations admissibles, pas nécessairement des journées complètes. Le bloc CAL peut contenir peu ou zéro positif ; cela ne contourne aucun des autres contrôles. Un fit peut encore se replier sur NYX.

## Modèles et comparaisons

| Nouveau modèle | Rôle |
| --- | --- |
| `calibrated_control_direct` | Fondamentaux/réseau, nouvelle calibration, sans signaux congestion prévus |
| `calibrated_control_governed` | Même contrôle avec gouvernance |
| `congestion_calibrated_direct` | Signaux CNEC et régionaux gelés en plus, nouvelle calibration |
| `nyx_congestion_calibrated` | Variante enrichie gouvernée, candidat primaire préfixé |

Les deux nouvelles chaînes utilisent les mêmes lignes admissibles et les mêmes seuils CORE90. Les probabilités de risque doivent être comparées **entre ces deux chaînes**. Les anciens détecteurs à CAL28 n'avaient pas le même CORE ni nécessairement le même événement : leurs probabilités ne mesurent pas automatiquement la même cible.

Le rapport de prix compare **douze modèles** : ces quatre nouveaux candidats, les quatre anciennes variantes congestion/contrôle, NYX opérationnel, Storm et les deux variantes physiques réseau+CGC. Les anciens modèles restent immuables. La comparaison de leurs prix est valide sur le support commun, mais elle mesure des chaînes complètes, pas seulement un calibrateur.

## Rapport et interprétation

Le rapport contient les KPI sur les **365 derniers jours** et les **7 derniers jours**, avec l'intersection exacte d'heures physiques des douze modèles. Les scores journaliers excluent les jours incomplets, y compris aux changements d'heure. Les replis sont inclus, et la livraison future n'est pas évaluée.

L'EVA conserve la politique économique existante, les volumes, seuils et coûts. Sa référence demeure le prix day-ahead observé D−1 : **proxy non exécutable à 08 h**, et non P&L réel démontré.

Les épisodes du 14 septembre et des 24–26 juin servent de diagnostics : prix et profils horaires, probabilités brutes/calibrées, état de calibration, disponibilité du correcteur, fit utilisé et raisons des propositions. Les statistiques d'entraînement et de repli proviennent des folds scellés du nouveau résultat.

Les **scores probabilistes hors échantillon** comparent Brier et log-loss, bruts et calibrés, sur les mêmes heures évaluées où les deux nouvelles chaînes sont prêtes. Le rapport vérifie l'égalité des seuils CORE90 avant de construire l'événement `observé − NYX ≥ seuil`. Il ne réévalue pas les lignes CAL ayant servi à ajuster un calibrateur. Les dix classes fixes de fiabilité affichent probabilité moyenne, fréquence observée, effectifs et événements ; une classe vide reste sans valeur. Les scores sont filtrables par pays et fenêtre 365/7 jours. Brier et log-loss reflètent aussi la discrimination, pas uniquement la calibration ; des événements rares et des heures dépendantes appellent à la prudence. Ce sous-échantillon prêt est distinct des KPI de prix, qui incluent les replis.

Une disponibilité accrue peut augmenter les fausses corrections. Il faut comparer l'enrichi au **nouveau contrôle apparié**, les erreurs annuelles, les pics, les jours ordinaires et l'EVA, sans choisir une règle par pays après observation. Le P50 reste le quantile 0,5 d'une distribution d'erreur ; aucune addition manuelle de `probabilité × amplitude` n'est introduite.

Cette année a déjà guidé des hypothèses : **l'expérience reste exploratoire et hors production**. Aucun gain n'est garanti, aucun modèle n'est promu automatiquement. Une période future gelée est nécessaire avant de remplacer le modèle opérationnel.

## Résultat du replay du 15 septembre 2026

Snapshot : `20260915T160622Z_2de43144`. Rapport : `reports/20260915T162012Z_40d15643/nyx_congestion_calibration_report.html` dans ce snapshot. Période : **15/09/2025–14/09/2026**, FR/DE/BE/NL, **34 940 heures-pays communes**. Les partitions reconstruites, les disponibilités des labels et les empreintes des sources ont été vérifiées ; aucun changement de production.

La nouvelle calibration permet **21 fits sur 53**, dont celui du 14 septembre. L'apprentissage commence le 27 avril : la réserve de 90 jours retarde le démarrage, même si elle permet de récupérer des semaines récentes. Le correcteur est prêt sur **8 232 heures-pays évaluées** ; les autres restent des replis sur NYX.

Sur ces 8 232 heures et 147 événements, le détecteur enrichi passe d'un Brier de **0,017459 à 0,017022** et d'une log-loss de **0,101737 à 0,083473** après calibration. Cela améliore les scores probabilistes, mais **ne suffit pas à améliorer les prix** :

| Modèle | MAE €/MWh | RMSE €/MWh | EVA simulée, écart à NYX |
| --- | ---: | ---: | ---: |
| NYX opérationnel | 11,36750 | 21,87553 | 0 € |
| Congestion recalibrée, gouvernée | 11,36976 | 21,87721 | −576,56 € |
| Congestion recalibrée, directe | 11,39863 | 21,86899 | −17 250,38 € |

Le candidat gouverné modifie 41 heures communes : 12 améliorées, 29 dégradées. L'EVA conserve le proxy D−1 non négociable décrit plus haut. Aucun changement sur les 24–26 juin pour les deux variantes congestion.

**14 septembre, Allemagne à 19 h :** le correcteur est entraîné, la probabilité d'une sous-estimation de NYX d'au moins 50 €/MWh passe de 3,51 % à 15,77 %, mais la médiane conditionnelle de l'erreur n'est pas positive. Le P50 reste donc **324,59**, contre **697,31 €/MWh observés**. Il ne s'agit plus d'un échec de calibration : le lien entre signal réseau, risque d'erreur local et distribution d'amplitude reste insuffisant. Cette probabilité d'erreur n'est pas la probabilité d'activation de congestion de l'expert amont.

**Décision : conserver cette variante au laboratoire, sans promotion.** La calibration résout le blocage technique récent ; elle ne résout pas les spikes. Les résultats ne justifient ni de forcer une hausse du P50 ni de modifier les poids par pays après lecture de cette année.
