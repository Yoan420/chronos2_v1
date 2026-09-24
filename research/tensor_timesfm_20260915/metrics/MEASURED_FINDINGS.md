# NYX — audit numérique des prévisions figées et prototypes résiduels

Analyse du 15 septembre 2026. **La sélection sur validation conserve NYX sans correction supplémentaire.** La faible dimension des erreurs simultanées est mesurée ; leur prévisibilité causale par les petits modèles testés ne l'est pas. Aucune activation de production, acquisition Saturn, réexécution Chronos/CatBoost/Kalman ou exécution Tensor-TimesFM n'a eu lieu pour cette analyse.

## Périmètre, unités et provenance

- Quatre bundles `runs/experiments/nuclear_forecast_v1/2026-09-16/{be,de,fr,nl}/civil_pit_v2`, chargés avec le validateur officiel des archives : identités, empreintes des entrées/sorties, alignement et invariants sémantiques validés.
- Référence : médiane **`residual_kalman__q50`**, chaîne opérationnelle NYX complète. P10/P90 proviennent du même objet. Les séries Chronos et après correcteur résiduel servent seulement à situer les étapes.
- Historique commun : **16/09/2025–15/09/2026**, 365 jours, 8 760 heures physiques par pays, 35 040 lignes. Le jour futur 16/09/2026 est exclu. Les audits du replay indiquent zéro violation de causalité détectée et zéro assimilation d'observation cible avant sa prévision.
- Il s'agit de **backtests préquentiels reconstruits**, pas d'une archive d'un an de prévisions réellement publiées à leur date. Ces contrôles internes ne prouvent pas la disponibilité historique de chaque vintage de donnée ni l'absence de sélection ultérieure de la configuration.
- Observations et Storm : derniers snapshots locaux de reporting validés par empreinte et contrat zone/date. Leurs chemins et empreintes exacts figurent dans `extraction_manifest.json`.
- Prix, MAE, RMSE, biais et scores d'intervalle en **EUR/MWh**. Biais = prévision − observation. Chaque comparaison NYX/Storm emploie exactement les mêmes heures.
- L'unique trou Storm par pays est le **26/10/2025 à 00:00 UTC**, première occurrence de 02h civile lors du changement d'heure : aucun remplissage. La comparaison annuelle utilise donc **8 759 heures par pays**. Les scores propres à NYX et ses intervalles gardent les 8 760 heures. Les journées physiques de 23/24/25 heures sont conservées.

## Résultats descriptifs sur l'année

| Pays | NYX MAE | NYX RMSE | NYX biais | Storm MAE | Storm RMSE | Storm biais |
|---|---:|---:|---:|---:|---:|---:|
| BE | 11,165 | 23,824 | −1,048 | 10,936 | 23,499 | −2,819 |
| DE | 11,120 | 22,071 | −0,712 | 11,444 | 20,174 | −1,798 |
| FR | 12,087 | 19,060 | −0,088 | 11,857 | 18,501 | −1,915 |
| NL | 11,172 | 22,486 | −0,757 | 11,000 | 20,594 | −1,775 |

NYX réduit le biais négatif de Storm sur les quatre pays. Il réduit sa MAE seulement en DE dans cet agrégat ; la RMSE reste plus élevée dans les quatre pays. Storm demeure un benchmark fondamental sérieux. Son instantané audité ne prouve cependant pas sa disponibilité stricte à 08h au cutoff des anciennes prévisions : il ne peut pas devenir automatiquement une covariable de combinaison causale.

MAE des étapes, mêmes 8 759 heures par pays :

| Pays | Chronos | Après correcteur résiduel | NYX complet |
|---|---:|---:|---:|
| BE | 12,029 | 11,264 | 11,165 |
| DE | 12,390 | 11,209 | 11,120 |
| FR | 12,977 | 12,234 | 12,087 |
| NL | 12,297 | 11,219 | 11,172 |

Le benchmark pertinent d'un nouveau module est bien NYX final. Un gain contre Chronos seul surestimerait ici l'apport additionnel.

Les heures les plus difficiles sont principalement **19–21h civiles** : MAE à 19h/20h de 18,38/18,24 en DE, 16,79/17,88 en BE, 17,74/17,81 en NL. En FR, 19h atteint 14,90 et 20h 14,31 ; 14h est également difficile, 13,46. Les biais des heures du soir sont généralement négatifs.

| Pays | MAE hiver DJF | Printemps MAM | Été JJA | Automne SON |
|---|---:|---:|---:|---:|
| BE | 6,86 | 12,06 | 14,94 | 10,70 |
| DE | 8,48 | 11,71 | 12,15 | 12,09 |
| FR | 9,26 | 15,07 | 13,29 | 10,65 |
| NL | 7,81 | 12,35 | 13,04 | 11,41 |

La saisonnalité est descriptive sur un seul cycle annuel ; elle ne démontre pas un effet stable d'une année à l'autre.

## Extrêmes et intervalles

Régime négatif : observation <0. Pics : observation supérieure au quantile initial d'entraînement, **jamais recalculé sur validation/test**. Les seuils q99 initiaux sont BE 183,818 ; DE 249,924 ; FR 158,461 ; NL 216,532 EUR/MWh. Comme le niveau de prix change, les occurrences futures au-dessus de ce seuil ne représentent pas nécessairement 1 % de chaque période.

| Pays | Heures négatives | MAE négatifs | Heures >q99 train | MAE >q99 | Biais >q99 | Couverture P10–P90 >q99 |
|---|---:|---:|---:|---:|---:|---:|
| BE | 330 | 17,79 | 388 | 39,81 | −26,44 | 61,34 % |
| DE | 527 | 13,81 | 104 | 102,26 | −87,46 | 33,65 % |
| FR | 577 | 13,36 | 626 | 14,96 | −7,15 | 70,61 % |
| NL | 439 | 15,64 | 155 | 73,23 | −55,05 | 52,26 % |

Les médianes sont trop hautes lors des prix négatifs en moyenne (biais +2,55 à +4,80) et trop basses lors des pics. Les amplitudes et divergences locales constituent donc un risque concret pour une représentation trop compressée. Ce constat ne dit pas qu'une décomposition particulière les lisserait : il définit un test obligatoire.

| Pays | Couverture annuelle, cible 80 % | Largeur moyenne | Score d'intervalle 80 | WIS à un intervalle 80 |
|---|---:|---:|---:|---:|
| BE | 74,46 % | 30,68 | 58,45 | 7,619 |
| DE | 74,34 % | 30,98 | 58,79 | 7,625 |
| FR | 72,99 % | 33,28 | 60,17 | 8,040 |
| NL | 75,87 % | 31,69 | 58,29 | 7,610 |

`IS80=(u−l)+10(l−y)+ si y<l +10(y−u)+ si y>u`, soit dans le code `(u-l)+10*max(l-y,0)+10*max(y-u,0)`. `WIS80=(0,5*|y−m|+0,1*IS80)/1,5` : WIS avec **un seul intervalle** et la médiane, pas un score sur une grille complète de quantiles. Les intervalles des corrections expérimentales ne sont pas fabriqués.

Les largeurs P10–P90 restent celles de Chronos, à l'arrondi machine près, après les deux décalages de médiane. Le correcteur résiduel atteint son cap absolu de 40 sur 0,114 % des heures BE, 0,502 % DE, 0,137 % FR, 0,411 % NL. Sur les pics >q99, il atteint ce cap sur 13,46 % des heures DE et 7,74 % NL. Le Kalman n'atteint pas son cap de 20 sur cette année. Ces fréquences ne justifient aucun relèvement automatique des caps.

## Dépendance simultanée, sans préjuger sa prévisibilité

Sur les 180 jours initiaux seulement, les résidus NYX horaires des quatre pays ont un premier axe expliquant **65,70 %** de la variance brute, deux axes **85,89 %**. Rang de participation `1/sum(p²)` = **2,07**, sur un espace initial à quatre dimensions. Après standardisation par pays : premier axe 63,65 %, rang 2,15.

Corrélations des erreurs : BE/NL 0,772 ; DE/NL 0,710 ; BE/DE 0,594 ; BE/FR 0,425 ; FR/NL 0,247 ; FR/DE 0,180. Le comportement français est moins partagé : l'écraser dans un facteur commun n'est pas neutre.

La matrice descriptive jour × (heure civile × pays), 96 colonnes, a un rang de participation **7,18** sur le train. Pour ce seul diagnostic, l'heure civile d'automne répétée est moyennée et le jour de printemps incomplet est exclu ; cette convention ne s'applique jamais aux scores physiques. Une reconstruction de résidus observés serait un **oracle de compression**, pas une prévision. L'éventuel audit oracle complémentaire est séparé.

Les erreurs NYX/Storm sont imparfaitement corrélées sur le train : BE 0,545 ; DE 0,419 ; FR 0,574 ; NL 0,482. Cela motive au plus un test de combinaison avec vintages compatibles. Nous n'avons aucune série d'erreurs Tensor-TimesFM permettant de conclure à sa complémentarité avec NYX.

## Protocole des expériences légères

1. Entraînement initial : **16/09/2025–14/03/2026**, 180 jours. Validation : **15/03–17/06/2026**, 95 jours. Test final verrouillé : **18/06–15/09/2026**, 90 jours. Les paramètres et la famille primaire sont choisis sur validation avant le calcul des prévisions du test ; `validation_selection_locked.json` enregistre la sélection.
2. Pour chaque livraison D, seules les erreurs des jours **≤D−2** sont permises. C'est une hypothèse de retard conservatrice face à l'incertitude sur la disponibilité des labels. Les modèles apprennent les erreurs des labels frozen ; ces labels ne sont pas prétendus vintages historiques.
3. EWMA pays et pays×heure, demi-vies 7/28 jours. Ridge univariée et multivariée, alpha 10/100 ; PCA-ridge, rang 1/2 et alpha 10/100. Zéro correction NYX inclus : 13 candidats au total. Ridge minimise L2, sélection finale par MAE de validation.
4. Caractéristiques : résidu D−2 à la même heure civile, EWMA pays et EWMA pays×heure. Pour ridge, demi-vie EWMA 14 jours fixée. Univarié : trois caractéristiques du pays ; multivarié : les douze caractéristiques des quatre pays. Aucun résidu contemporain, observation future ou Storm en entrée.
5. Ridge : jusqu'à 180 jours d'entraînement admissibles, refit tous les 7 jours. Le warm-up de 14 jours donne 165 jours au premier refit de validation, puis 172, 179 et 180. Imputation, normalisation et PCA apprises uniquement sur la fenêtre de fit. PCA sur les quatre résidus standardisés ; composantes estimées de nouveau à chaque refit et utilisées ensemble pour encodeur/décodeur de ce fit.
6. Prévision day-ahead chaque jour avec les nouvelles caractéristiques admissibles ; coefficients figés entre refits. Ce n'est pas une prévision à sept jours émise au refit. Les labels des premiers jours du test peuvent ensuite entrer dans les refits suivants selon la règle D−2 fixée, sans nouvelle sélection de paramètres.
7. Toutes les heures physiques cibles sont conservées ; l'heure d'automne répétée est moyennée seulement pour la caractéristique retardée. Une heure retardée manquante est imputée avec le train. Aucune imputation des observations utilisées pour scorer.
8. Incertitudes appariées : **2 000 rééchantillonnages de blocs mobiles contigus de 7 jours**, mêmes blocs pour tous les pays/modèles, seed 20260915. Les sommes des erreurs par jour servent à recalculer MAE et RMSE sur chaque tirage. Les IC95 sont conditionnels aux 90 jours disponibles ; pas de correction de multiplicité pour les comparaisons secondaires.

La revue indépendante du script n'a identifié aucune fuite temporelle dans ces opérations. Une vérification par perturbation des labels D−1 et futurs laisse les caractéristiques EWMA de D identiques. Les 151 refits respectent leur borne de labels enregistrée. Ces contrôles ne résolvent pas le manque de vintages source.

## Résultats du test final réservé

La validation choisit **NYX sans correction** : MAE 12,315, contre 12,411 EWMA pays28, 12,654 EWMA pays×heure28, 12,426 ridge univariée alpha100, 12,691 multivariée alpha100 et 12,494 PCA rang1 alpha100.

Les variantes ci-dessous sont les meilleurs paramètres de chaque famille sur validation. **Elles ne sont pas choisies après observation du test.** Test : 2 160 heures par pays, 8 640 points, sans heure Storm manquante. Un delta positif signifie une dégradation par rapport à NYX.

| Modèle | MAE | RMSE | Biais | Delta MAE [IC95 blocs 7j] |
|---|---:|---:|---:|---:|
| NYX | 14,191 | 30,196 | −1,852 | 0 |
| EWMA pays | 14,224 | 30,218 | −0,195 | +0,032 [−0,111 ; +0,192] |
| EWMA pays×heure | 14,463 | 30,262 | −0,194 | +0,271 [+0,107 ; +0,474] |
| Ridge univariée | 14,486 | 30,185 | −1,263 | +0,295 [+0,140 ; +0,482] |
| Ridge multivariée | 14,659 | 30,399 | −0,657 | +0,467 [+0,219 ; +0,741] |
| PCA-ridge | 14,366 | 30,139 | −1,090 | +0,175 [+0,026 ; +0,351] |
| Storm, benchmark séparé | 12,758 | 27,575 | −4,046 | −1,433 [−3,618 ; +0,199] |

PCA-ridge diminue ponctuellement la RMSE de 0,057, avec IC95 [−0,254 ; +0,289] : **aucun gain RMSE établi**. Le recentrage EWMA réduit fortement le biais sans améliorer MAE/RMSE. Le succès d'une correction du biais ne doit donc pas être assimilé à une meilleure prévision.

| Pays | NYX MAE test | Storm MAE test | PCA-ridge MAE test | Ridge multivariée MAE test |
|---|---:|---:|---:|---:|
| BE | 16,108 | 14,072 | 16,308 | 16,719 |
| DE | 13,069 | 12,017 | 13,225 | 13,284 |
| FR | 13,654 | 12,304 | 13,804 | 14,263 |
| NL | 13,936 | 12,640 | 14,127 | 14,369 |

L'avantage ponctuel de Storm sur ce test dominé par l'été n'est pas une preuve d'avantage moyen annuel : son IC agrégé traverse zéro et ses vintages strict-08 restent à vérifier. Les fichiers détaillent aussi les scores par heure et régime. Aucun résultat de ce test ne justifie l'ajout direct d'une correction multivariée ou factorielle de ces résidus.

## Labels frozen contre latest

Il ne faut pas compter les conversions float32 comme des révisions de marché. Sur 8 760 heures par pays, une seule paire présente un écart matériel ; les autres différences sont compatibles avec la même représentation float32 et un écart ≤0,00005 EUR/MWh. Critère d'écart non exact : >10⁻⁹ ; représentation vérifiée séparément.

L'écart matériel commun est **16/06/2026 à 21:00 UTC** :

| Pays | Frozen | Latest | Écart absolu | MAE des écarts sur les 8 760h |
|---|---:|---:|---:|---:|
| BE | 143,464996 | 134,16 | 9,304996 | 0,0010643 |
| DE | 143,107498 | 134,05 | 9,057498 | 0,0010362 |
| FR | 139,494995 | 132,54 | 6,954995 | 0,0007955 |
| NL | 143,520004 | 134,23 | 9,290004 | 0,0010626 |

Cela représente **0,0114 % par pays**. Il s'agit d'une divergence de versions observées, dont la cause exacte (révision amont ou transformation antérieure) n'est pas déterminée ici. Il n'y a aucun écart matériel sur les 90 jours du test ; ses métriques frozen/latest ne diffèrent que par les arrondis. L'effet maximal sur la MAE annuelle NYX est environ 0,001063 EUR/MWh. **Une faible sensibilité numérique ne démontre pas la disponibilité historique des données. Aucun gain prospectif ne peut être revendiqué avant un replay avec vintages horodatés ou une expérience prospective.**

## Coût, reproductibilité et portée de la conclusion

Sur cette machine, extraction/vérification locale environ **7,8 s** ; expériences, 151 refits et 2 000 bootstrap environ **3,5 s** internes, environ 3,3 s CPU et 178 Mo RSS en fin de processus. Le démarrage de l'environnement Python ajoute quelques secondes. Ces coûts ne sont ni le coût de NYX complet ni celui de Tensor-TimesFM. Aucun GPU utilisé.

Reproduction depuis la racine du projet, dans l'environnement Python du projet : `python research/tensor_timesfm_20260915/metrics/extract_metrics.py`, puis `python research/tensor_timesfm_20260915/metrics/experiments.py`. Ces scripts lisent les bundles et écrivent uniquement dans ce dossier. Ils ne contactent aucune API de marché et n'entraînent aucun modèle de production.

Artefacts : `verified_hourly_pairs.csv.gz` ; `extraction_manifest.json` ; `descriptive_metrics.csv` ; `nyx_interval_calibration.csv` ; `correction_saturation.csv` ; `label_revision_sensitivity.json` ; `validation_selection_locked.json` ; `locked_test_predictions.csv.gz` ; `locked_test_metrics.csv` ; `locked_test_group_metrics.csv` ; `locked_test_paired_bootstrap.csv` ; `rolling_fit_audit.json` ; `experiment_summary.json`.

**Conclusion mesurée :** les corrections résiduelles simples testées n'apportent pas de gain additionnel à NYX. Une corrélation spatiale élevée ne suffit pas à rendre les erreurs prévisibles. Les priorités démontrées sont la vérification des vintages, la calibration des intervalles et la robustesse aux pics/divergences entre pays. Cette petite expérience ne réfute pas tout modèle latent, toute perte robuste, ni l'apport de fondamentaux disponibles à temps ; elle impose qu'une solution plus complexe batte d'abord ces références sur une évaluation causale prospective. Elle n'évalue pas Tensor-TimesFM lui-même.
