# Variantes anti-spikes NYX : laboratoire séparé

Ce programme ne modifie ni Forecast.ps1, ni ses modes, modèles, dépendances,
caches ou exports. Il compare des variantes de l'expert Scarcity après le
`nuclear_kalman` déjà publié. Aucun mécanisme de promotion n'est présent.

## Utilisation

```powershell
# Déjà effectué ici : installation dans un répertoire privé, pas dans le venv.
& 'C:\Users\BQ6757\chronos2_v1\ScarcityVariants.ps1' -Action Install

# Fige la recette, calcule les quatre variantes et crée le rapport comparatif.
& 'C:\Users\BQ6757\chronos2_v1\ScarcityVariants.ps1' -Action Run

# Après interruption : réutilise les variantes terminées, recalcule les autres.
& 'C:\Users\BQ6757\chronos2_v1\ScarcityVariants.ps1' -Action Backtest
& 'C:\Users\BQ6757\chronos2_v1\ScarcityVariants.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\ScarcityVariants.ps1' -Action Report
```

Le YAML `config/nyx_scarcity_variants.yaml` fixe la source de comparaison,
les variantes et les paramètres. Le défaut conserve le snapshot HGB terminé
`20260914T125027Z_ed1dc982` : même période du 15/09/2025 au 14/09/2026,
mêmes observations, quantiles, NYX et Storm, plus livraison du 15/09/2026
séparée des scores. Ce n'est pas une actualisation automatique des sources.

Une autre référence s'utilise avec `-SourceSnapshot <snapshot Scarcity terminé>`
sur Run/Prepare. Les actions Backtest/Report/Status acceptent `-RunDirectory`
et ne permettent pas de changer la recette figée. Modifier le YAML puis lancer
Run crée une expérience distincte, sans écraser les résultats précédents.

Le moteur vérifie les empreintes des entrées, du contrôle HGB, du code, du
runtime privé et des résultats. Deux processus de deux threads au plus
peuvent tourner simultanément. Chaque variante possède son log et ses résultats
intermédiaires. Les variantes déjà terminées ne sont pas réentraînées. Un
changement de code avant la fin impose un nouveau Prepare ; une expérience
terminée reste consultable. Aucun appel Saturn ni téléchargement de données
n'est nécessaire au backtest de ces entrées déjà figées.

## Les quatre expériences prédéclarées

| Identifiant | Classification | Seuil de forte erreur |
| --- | --- | --- |
| xgb_unweighted_fixed | XGBoost non pondéré, recalibré | Constant par bloc, historique d'entraînement |
| xgb_weighted_fixed | XGBoost pondéré, recalibré | Identique au précédent |
| xgb_unweighted_dwt | XGBoost non pondéré, recalibré | DWT causal, actualisé chaque origine |
| xgb_weighted_dwt | XGBoost pondéré, recalibré | Même DWT causal |

Le contrôle `hgb_v1` reprend les prévisions déjà calculées lors du premier
test. NYX et Storm sont aussi affichés. Le passage HGB vers XGBoost n'est
pas un simple changement de poids : il change le moteur d'arbres. Les paires
XGBoost pondéré/non pondéré permettent d'isoler précisément la pondération.

### Pondération et calibration

Le poids positif vaut `min(30, sqrt(négatifs / positifs))`, calculé uniquement
sur le bloc d'entraînement initial. Il n'est pas optimisé sur le backtest et
ne reproduit pas les poids choisis dans le papier. La racine carrée représente
un choix de pondération modérée, déclaré avant les résultats.

Les 28 jours chronologiquement suivants sont exclus de l'entraînement des
arbres et servent à une calibration logistique non pondérée des marges brutes
XGBoost. Ainsi le score issu d'un apprentissage pondéré n'est pas directement
interprété comme une probabilité. L'efficacité de cette recalibration doit
néanmoins être mesurée hors échantillon, pas supposée.

La sévérité, le déclenchement à p>0,6, le plafond de 400 EUR/MWh, la banque
0/0,25/0,5/1, les contrôles sur la MAE et les intervalles restent ceux du premier
expert. La proposition ponctuelle reste une médiane du mélange simplifié
correction nulle/erreur positive, ensuite réduite par la gouvernance.

### DWT strictement antérieur à 08 h

La cible reste **la forte sous-estimation de NYX**, et non le simple niveau
élevé du prix. Le DWT est donc une adaptation sur les erreurs de NYX :

`u = max(50, w × quantile95(erreurs sur 365 jours) + (1-w) × quantile95(erreurs sur 30 jours))`.

Le poids w augmente avec la volatilité des erreurs sur 30 jours. Sa plage de
normalisation provient exclusivement des volatilités calculées aux origines
précédentes ; une volatilité élevée donne plus de poids à la référence globale.
Une plage dégénérée impose w=0,5, tracé dans l'audit. Toutes les erreurs utilisées
concernent des livraisons antérieures, avec labels disponibles avant le cutoff.
Le prix réalisé de l'heure à prévoir ne participe jamais au seuil ou au poids.

Le seuil est une feature déclarée et évolue quotidiennement, y compris entre
deux réentraînements hebdomadaires. Les premières 30 journées ne possèdent pas
ce seuil ; elles ne sont pas remplies artificiellement. Il faut ensuite au
moins 90 jours d'entraînement éligibles : le démarrage DWT est donc plus tardif,
en général après 120 jours de support source. Les abstentions restent incluses
dans les scores annuels, et la couverture d'entraînement est affichée.

## Comparaison et interprétation

Les scores de prix utilisent les mêmes heures disponibles pour NYX, Storm,
HGB et toutes les variantes. Le seuil natif DWT peut changer les événements
comptés ; les précisions/rappels natifs de cibles différentes ne sont donc pas
classés comme s'il s'agissait du même problème.

Un diagnostic de classement séparé utilise l'événement commun
`observé - NYX >= 50 EUR/MWh`, sur l'intersection des probabilités disponibles.
ROC-AUC et average precision/PR y mesurent la discrimination. Cela ne transforme
pas les probabilités de cibles différentes en probabilités calibrées du même
événement. Brier et courbes de fiabilité sont interprétés pour chaque cible
native. Fausses alertes, événements manqués, corrections effectivement appliquées
et dégradations sur les heures corrigées sont explicités.

Les queues top 1 % et 5 % de prix observés sont des cohortes de diagnostic
définies ex post, jamais des entrées de sélection des poids. L'année et les
épisodes de juin/septembre ont déjà été examinés ; un meilleur score après
ces essais reste exploratoire et devra être confirmé prospectivement.

## SHAP : ce qui est expliqué

TreeSHAP exact est calculé par le moteur XGBoost, sans dépendance SHAP ajoutée
au venv opérationnel. La somme des contributions, avec le terme de base, doit
reconstruire la marge brute. Pour Platt `logit(p)=a+b×marge`, les contributions
calibrées sont `b×SHAP` et la base est `a+b×base_brute`. Les reconstructions
de marge, log-odds et probabilité sont vérifiées numériquement.

Il s'agit des **contributions au log-odds du détecteur**, pas de contributions
en EUR/MWh au prix final, ni de causes physiques démontrées. Les décisions du
plafond et de la gouvernance sont expliquées séparément. Les contributions
signées sont nécessaires : une importance absolue ne donne pas le sens de l'effet.

L'échantillon est fixé par calendrier : dernières journées et observations
hebdomadaires à une heure prédéfinie, uniquement quand le modèle était disponible.
Il n'est pas choisi d'après les erreurs observées. Chaque explication emploie
le modèle disponible à l'origine de la prévision, jamais un modèle final ajusté
avec des observations futures. Le rapport expose la sélection et la couverture.

## Proposer un prix corrigé, pas seulement une alerte

```powershell
# Après une comparaison ScarcityVariants terminée : aucune nouvelle formation.
& 'C:\Users\BQ6757\chronos2_v1\ScarcityAdjustments.ps1' -Action Run
& 'C:\Users\BQ6757\chronos2_v1\ScarcityAdjustments.ps1' -Action Report
```

Ce deuxième rapport utilise les amplitudes déjà calculées à chaque origine
historique, jamais les observations ultérieures pour construire les propositions.
Pour une probabilité dépassant le seuil prévu, l'expert de sévérité estime une
forte erreur positive à partir des fondamentaux et de ses erreurs de calibration.
Le modèle combine cette sévérité avec la probabilité pour proposer une correction,
plafonnée à 400 EUR/MWh dans la recette actuelle.

Les colonnes distinguent : prix NYX, probabilité, correction proposée en EUR/MWh,
prix proposé, poids autorisé, correction appliquée, prix gouverné et raison
du refus éventuel. La proposition reste visible même si le contrôle refuse
de l'appliquer. Une absence de modèle disponible ne devient pas une alerte.

Le rapport teste trois applications fixes de la proposition existante :
25 %, 50 % et 100 %. Ces fractions sont déclarées avant leur comparaison,
pas sélectionnées heure par heure après lecture des observations. Chaque
fraction est évaluée sur la même période et les mêmes heures que NYX, Storm
et la prévision gouvernée ; heures améliorées et dégradées sont comptées.
Les quantiles de NYX ne sont pas réutilisés comme de faux intervalles calibrés
des propositions. Les propositions sont des scénarios de recherche non gouvernés,
et non des forecasts promus. Aucune garantie annuelle n'est relâchée dans le
pipeline actuel.

`-SourceSuite` choisit une autre suite ScarcityVariants **terminée et scellée**.
Les résultats de ce diagnostic sont séparés dans
`runs/experiments/nyx_scarcity_v1/variants/adjustments/snapshots/`.
`-RunDirectory` s'utilise uniquement avec Report/Status pour relire un diagnostic
existant. Les entrées, sorties et recettes sont contrôlées par empreintes SHA.

## Limites conservées

Les labels et les replays historiques ont les mêmes limites de provenance que
le premier test : disponibilité des labels supposée à D-1 18 h, entraînement
initial progressif (pas 365 jours avant chaque origine), historiques pas
certifiés comme véritables prévisions opérationnelles hors échantillon.
JAO reste exclu à 08 h ; les coûts gaz sont des proxies TTF/EUA, pas la série
CGC native Saturn. Aucun résultat ne garantit la non-dégradation future.

## Résultats exécutés le 14 septembre 2026

Suite complète : `20260914T135026Z_0efb0e05` ; diagnostic des amplitudes :
`20260914T135953Z_77df8eeb`. Les quatre variantes ont été entraînées et
évaluées. Toutes les reconstructions TreeSHAP sont vérifiées. L'export des
dictionnaires d'audit vides du DWT est corrigé : ils sont sérialisés en texte
JSON dans les Parquet d'audit, sans modifier l'apprentissage. Les deux variantes
fixes relancées ont reproduit des Parquet de prévisions strictement identiques.

Fenêtre du 15/09/2025 au 14/09/2026 : 365 jours représentés, mais **8 735 heures
communes par pays**. Les 24 heures du 15/09/2025 et une heure DST du 26/10/2025
manquent chez Storm et sont exclues de tous les scores comparatifs. Le
15/09/2026 reste hors scores, même lorsque son prix observé est déjà publié.

MAE regroupée FR/DE/BE/NL en EUR/MWh ; mêmes heures pour chaque cellule :

| Expert | Prix gouverné | Proposition 25 % | Proposition 50 % | Proposition 100 % |
| --- | ---: | ---: | ---: | ---: |
| HGB de référence | 11,368240 | 11,365612 | 11,366919 | 11,376294 |
| XGB non pondéré, fixe | 11,368902 | 11,347755 | 11,337564 | 11,333792 |
| XGB pondéré, fixe | 11,368968 | 11,349946 | 11,348469 | 11,365517 |
| XGB non pondéré, DWT | 11,367500 | 11,353089 | 11,356996 | 11,383580 |
| XGB pondéré, DWT | 11,367500 | 11,347810 | 11,349975 | 11,376508 |

NYX inchangé : **11,367500** ; Storm : **11,317589**. Aucune variante gouvernée
ne démontre ici de gain annuel contre NYX. Les amplitudes non gouvernées du
XGB fixe non pondéré sont plus intéressantes, mais leur meilleur résultat
global ne dépasse pas Storm sur ce support, et ne vaut pas validation prospective.

Le diagnostic commun de classement utilise 18 908 heures prêtes et 214 fortes
sous-estimations : l'average precision passe de 0,15599 pour HGB à 0,24666 pour
XGB non pondéré fixe. Les seuils natifs, y compris ceux du DWT, restent à leur
plancher de 50 EUR/MWh sur les heures évaluées. Le DWT ne fournit donc pas de
variation effective de seuil dans ce test ; ses différences reflètent aussi
son historique éligible plus court.

### Amplitude du XGB non pondéré fixe, par pays

| Pays | NYX, MAE | Proposition 25 % | Proposition 50 % | Proposition 100 % |
| --- | ---: | ---: | ---: | ---: |
| BE | 11,150389 | 11,124900 | 11,105595 | 11,083294 |
| DE | 11,078194 | 11,053620 | 11,035536 | 11,005802 |
| FR | 12,078891 | 12,073210 | 12,081700 | 12,104031 |
| NL | 11,162528 | 11,139289 | 11,127423 | 11,142043 |

L'application à 25 % améliore légèrement les quatre MAE annuelles dans ce
diagnostic ; 100 % améliore davantage BE/DE, mais dégrade FR. Les propositions
actives concernent seulement 53 heures pays cumulées (13 BE, 17 DE, 8 FR,
15 NL), pas 53 épisodes indépendants. Aucune fraction n'est activée.

Sur le top 1 % des prix observés, la proposition 100 % réduit la MAE BE
de 102,735 à 93,362, DE de 112,433 à 101,745 et NL de 106,987 à 100,882 ;
elle augmente la MAE FR de 25,090 à 26,426. Storm reste meilleur sur ces queues.

Exemple du 14/09/2026 à 19 h, **même proposition 100 % dans tous les pays** :

| Pays | NYX | Proposition | Observé | Storm |
| --- | ---: | ---: | ---: | ---: |
| BE | 286,78 | 443,05 | 441,74 | 370,58 |
| DE | 324,59 | 471,58 | 697,31 | 421,70 |
| FR | 284,41 | 436,37 | 298,01 | 296,47 |
| NL | 307,18 | 473,51 | 400,00 | 412,32 |

La correction approche bien le pic belge et réduit la sous-estimation
allemande, mais provoque une forte fausse correction française. Les alertes
sont dominées notamment par le spread CGC proxy/NYX et les fondamentaux
régionaux ; une meilleure différenciation zonale et une validation prospective
sont nécessaires avant d'envisager une application opérationnelle.

Les tests couvrent les recettes, la causalité des seuils et des propositions,
les métriques, la reconstruction SHAP, le confinement des fichiers, les
checksums, la reprise sans apprentissage et les contrôles interactifs des HTML.
Les fichiers opérationnels protégés sont inchangés.

## Références bibliographiques

- Ma, Chen et Meng (2026), [article fourni](https://doi.org/10.1080/01605682.2026.2660989).
- [Pondération XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).
- [Prédiction et contributions TreeSHAP](https://xgboost.readthedocs.io/en/stable/prediction.html).
- [Calibration des probabilités](https://scikit-learn.org/stable/modules/calibration.html).
- [Runtime XGBoost 3.2.0 compatible Python 3.11](https://pypi.org/project/xgboost/3.2.0/).
