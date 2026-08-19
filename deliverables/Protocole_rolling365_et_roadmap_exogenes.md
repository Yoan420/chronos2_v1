# Réentraînement quotidien sur 365 jours et feuille de route exogène

**Document de décision — 15 août 2026**  
**Périmètre :** forecasts day-ahead FR, DE, BE, NL et ES, avec un **cutoff d'information D−1 08:00 Europe/Paris**.  
**Contrainte de recherche :** aucune prévision Storm ou MKOnline comme variable d'entrée, cible auxiliaire, donnée de fit, signal de tuning ou poids appris. Storm n'intervient qu'après gel et hash du forecast, pour l'évaluation et la décision de promotion.

## 1. Décision recommandée

La prochaine version doit réentraîner **chaque jour le correcteur résiduel CatBoost + HistGradientBoosting**, à partir de zéro, sur les **365 derniers jours civils de livraison**. Le modèle fondation Chronos-2 reste préentraîné et utilisé en inférence : il **n'est pas réentraîné quotidiennement**.

Pour un jour de livraison `F`, la fenêtre de prix cible est exactement :

```text
F - 365 jours  ...  F - 1 jour, bornes incluses
```

La première activation théoriquement possible est le forecast du **16 août 2026**, car l'OOF scellé va jusqu'au 11 août et des replays/forecasts archivés existent du 12 au 15 août. Cette date reste **conditionnelle** : elle n'est acceptée que si les 365 journées, leurs origines 08:00, leurs snapshots exogènes et leurs hashes passent tous les contrôles décrits ci-dessous. Aucun rattrapage avec des données « latest » n'est permis.

Les premières variables à rechercher sont, dans cet ordre :

1. des **quantiles authentiques ou causalement recalibrés** de charge, éolien et solaire, puis de charge résiduelle ;
2. des prévisions météo numériques archivées par cycle, connues avant 08:00 ;
3. la capacité nucléaire disponible et les indisponibilités planifiées, surtout pour la France ;
4. des prix gaz/carbone connus avant 08:00 et des informations retardées des pays voisins ;
5. ensuite seulement, ensembles en ligne, filtrage robuste et recalibrage conformal ;
6. texte/news et adaptation LoRA de Chronos restent des pistes de recherche à faible priorité.

Le rolling-365 ne doit être promu qu'après un walk-forward sans fuite, un veto B2, puis **60 à 90 jours de shadow live**. Pour FR, DE, BE et NL, la condition métier principale est un win rate journalier MAE strictement supérieur à 50 % face au Storm officiel, sur des journées complètes et strictement appariées. Pour ES, aucune série Storm officielle n'est actuellement vérifiée : le tableau doit afficher `N/A`, jamais inventer une comparaison.

## 2. Ce qui est — et n'est pas — réentraîné

| Composant | Situation actuelle | Cible rolling-365 | Fréquence |
|---|---|---|---|
| Chronos-2 préentraîné | Chargé avec `from_pretrained`, puis utilisé en inférence | Inchangé ; contexte temporel actualisé | Inférence quotidienne |
| Correcteur résiduel autonome | CatBoost + HistGradientBoosting ajustés sur un corpus historique figé EXT223 + OOF730 | Ajustement à partir de zéro sur les 365 derniers jours causaux | Quotidienne |
| Hyperparamètres, schéma et clipping | Recette figée | Restent figés durant B2 et le shadow | Changement seulement après nouveau protocole A/B1/B2 |
| Poids MKOnline éventuels FR/NL | Poids historiques de la version de production | Aucun rôle dans le modèle autonome de recherche ; revalidation séparée si le blend final est conservé | Après gel du candidat autonome |
| Storm | Comparateur | Évaluation uniquement après hash du forecast candidat | Après production du forecast |

Preuves dans le dépôt :

- `chronos2_modular/forecasting.py` charge Chronos-2 via `Chronos2Pipeline.from_pretrained` ;
- `run_mkonline_live_hourly.py::_train_and_predict_extended` et `chronos2_hourly/multizone_live.py::_train_and_predict_autonomous` ajustent le correcteur résiduel ;
- `chronos2_hourly/models/residual_corrector.py` contient CatBoost et le fallback HistGradientBoosting ;
- le pipeline live actuel réajuste donc bien un modèle chaque jour, mais sur les mêmes données historiques figées : il ne constitue pas encore un vrai rolling-365 alimenté par les jours récents.

### Pourquoi ne pas fine-tuner Chronos tous les jours ?

Le bénéfice n'est pas établi, le coût et le risque opérationnel sont élevés, et l'identité du modèle changerait continuellement. Un benchmark européen de 2025 n'a trouvé aucun modèle fondation statistiquement supérieur à un fort benchmark MSTL biseasonnier ; à l'inverse, une étude belge récente trouve un intérêt à Chronos-2 en mode ARX, mais signale encore des difficultés sur les extrêmes. Ces résultats justifient d'abord l'enrichissement causal et le correcteur quotidien, non un fine-tuning quotidien du socle ([Hornek et al., 2025](https://arxiv.org/abs/2506.08113), [Bui et al., 2026, prépublication](https://arxiv.org/abs/2605.17045)).

Une adaptation LoRA de Chronos pourrait être étudiée plus tard, hors production et sur une cadence mensuelle ou ponctuelle. Une prépublication du 11 août 2026 rapporte des gains sur quatre marchés provinciaux chinois en modifiant environ 1 % des paramètres, mais l'étude est très récente, non évaluée sur les cinq zones européennes et son gain additionnel par rapport à un LoRA standard reste limité ([Fan et al., 2026, prépublication](https://arxiv.org/abs/2608.11359)).

## 3. Contrat temporel du rolling-365

### 3.1 Jour civil, pas « 365 × 24 heures »

Pour la zone `z` et le jour à prévoir `F` :

```text
W(F) = {F−365, F−364, ..., F−1}
```

`W(F)` contient exactement 365 dates civiles consécutives dans le fuseau de livraison de la zone. L'index horaire attendu est la concaténation des index physiques de ces 365 jours.

Conséquences :

- une journée de passage à l'heure d'été contient 23 heures ;
- une journée normale contient 24 heures ;
- une journée de retour à l'heure d'hiver contient 25 heures ;
- une fenêtre de 365 dates peut donc totaliser 8 759, 8 760 ou 8 761 heures ;
- le double horaire d'automne est conservé deux fois avec deux timestamps UTC différents ;
- aucune interpolation, moyenne ou déduplication sur l'heure locale n'est autorisée.

L'année bissextile ne change pas la règle : on soustrait 365 **jours civils**, pas une année calendaire (`DateOffset(years=1)` serait incorrect autour du 29 février).

### 3.2 Deux disponibilités différentes à ne pas confondre

À l'origine `F−1 08:00 Europe/Paris` :

- le prix day-ahead de `F−1` est déjà connu, car il a été fixé lors de l'enchère de `F−2` ; il peut donc être le dernier label prix de la fenêtre ;
- les réalisations physiques complètes de charge, éolien ou solaire de `F−1` ne sont pas encore disponibles ; une calibration des erreurs de fondamentaux doit s'arrêter à `F−2`.

Cette distinction est explicitement respectée dans l'étude 2026 sur les fondamentaux probabilistes : pour prévoir le jour `d`, les erreurs charge/éolien/solaire sont calibrées avec les réalisations de `d−N` à `d−2`, jamais avec la réalisation de `d−1` ([Uniejewski & Ziel, 2026](https://doi.org/10.1016/j.renene.2026.125844), [manuscrit et chronologie détaillée](https://arxiv.org/abs/2501.06180)).

### 3.3 Contrat d'une ligne d'entraînement

Pour chaque jour historique de livraison `T`, une ligne est recevable seulement si :

1. les q10/q50/q90 Chronos correspondent à une prévision réellement émise, un OOF scellé ou un replay PIT strict, d'origine `T−1 08:00 Europe/Paris` ;
2. la journée contient exactement l'index physique 23/24/25 attendu dans le fuseau de livraison ;
3. les quantiles sont finis et ordonnés `q10 <= q50 <= q90` ;
4. tout forecast exogène a un `snapshot_timestamp` et un `revision_timestamp` inférieurs ou égaux à l'origine ;
5. tout modèle/expert résiduel utilisé comme feature est lui-même cross-fitté et n'a pas appris sur `T` ;
6. le prix cible est le prix officiel du jour `T` ;
7. aucune colonne, dépendance ou transformation n'utilise Storm ou MKOnline ;
8. les transformations, imputations et normalisations sont ajustées uniquement sur la fenêtre d'entraînement.

Il est interdit de recalculer aujourd'hui une ancienne prévision Chronos avec les réalisations ou révisions actuelles : ce serait un backfill, pas un OOF.

Sources autorisées pour la prévision historique :

```text
sealed_oof | pit_replay | issued_live
```

Un `pit_replay` n'est admissible que s'il réutilise les snapshots immuables disponibles à l'origine historique. Un replay construit avec la dernière version d'une API est refusé.

## 4. Corpus initial et première date live

État observé dans le dépôt au 15 août 2026 :

- OOF scellé jusqu'au 11 août 2026 ;
- FR : replays PIT des 12–13 août, forecasts émis les 14–15 août ;
- DE, BE, ES : replays PIT des 12–14 août, forecast émis le 15 août ;
- NL : replays autonomes des 12–14 août dans `runs/live/nl/_replays`, puis forecast autonome émis le 15 août dans `runs/live/nl` ;
- les archives live incluent notamment le forecast Chronos, les inputs alignés, les covariables futures, les manifests et les checksums.

L'origine/cutoff à 08:00 prouve quelles informations étaient admissibles, pas l'heure d'exécution du programme. Les archives D14/D15 ont été générées plus tard dans la matinée pour plusieurs zones ; elles sont causales « as-of 08:00 », mais ne constituent pas encore une preuve de respect d'un SLA d'émission à 08:00.

Le forecast du 16 août 2026 peut donc être le premier candidat rolling-365 **si et seulement si** l'audit reconstruit, pour chaque pays, les 365 dates du 16 août 2025 au 15 août 2026 sans trou et confirme la causalité de chaque replay.

Audit local en lecture seule du 15 août : les cinq zones reconstruisent bien **365 jours et 8 760 heures**, dont 361 jours issus de l'OOF scellé et quatre jours issus des archives Chronos brutes des 12–15 août ; les checksums des forecasts et des inputs archivés concordent. Deux preuves manquent toutefois avant tout fit officiel. Premièrement, les caches de prix cibles s'arrêtent actuellement au 14 août : le label du 15 août doit être obtenu par la synchronisation Saturn puis validé. Deuxièmement, les archives historiques ne conservent que des résumés agrégés des snapshots/révisions exogènes, pas leurs maxima heure par heure. Le loader strict refuse donc aujourd'hui ce corpus ; l'OOF doit être rematérialisé une fois depuis les vintages PIT, ou rester bloqué si ces vintages ne permettent pas de reconstruire la preuve exacte. Pour NL, le corpus autonome homogène doit être chargé depuis `runs/live/nl` ; le run MKOnline parallèle n'est ni nécessaire ni admissible comme entrée de cette recherche.

En conséquence, le 16 août est une **date théorique de couverture**, pas encore une date d'activation opérationnelle. Le shadow rolling-365 ne démarre qu'après création et checksum d'un bloc d'entraînement par jour contenant explicitement actual, quantiles Chronos bruts, features, origine, maximum de snapshot et maximum de révision pour chaque heure.

Critère fail-closed : s'il manque un jour, une heure, une origine, un snapshot ou un hash, le candidat rolling n'est pas produit comme forecast officiel. La version courante reste disponible sous un nom de modèle explicite ; aucune substitution silencieuse n'est autorisée.

## 5. Artefacts à enregistrer chaque jour

Chaque run rolling doit écrire un manifeste immuable contenant au minimum :

- pays, fuseau de livraison, jour `F`, origine locale et UTC ;
- `window_start_day = F−365` et `window_end_day = F−1` ;
- nombre de jours (= 365), nombre d'heures attendu et observé ;
- distribution du nombre d'heures par jour : 23/24/25 ;
- pour chaque jour, `source_kind`, origine de forecast et cutoff exogène ;
- hash de chaque fichier source et hash du corpus concaténé ;
- hash du code, de la recette, du schéma de features et des hyperparamètres ;
- seed et nombre de threads ;
- audit des violations de cutoff, doublons, trous et valeurs non finies ;
- hash du forecast candidat **avant** tout chargement de Storm ;
- durée, statut et raison de fallback éventuelle.

Le corpus rolling doit être append-only. On ajoute un jour seulement lorsqu'il devient causalement utilisable ; on ne réécrit jamais une journée déjà scellée à partir d'une révision ultérieure.

## 6. Tests d'acceptation obligatoires

| Test | Condition d'acceptation |
|---|---|
| Fenêtre exacte | 365 dates consécutives, de `F−365` à `F−1`, incluses |
| Année bissextile | Toujours 365 dates ; aucun usage de « moins un an » |
| DST printemps | Le jour physique contient exactement 23 timestamps attendus |
| DST automne | Le jour physique contient exactement 25 timestamps UTC uniques ; les deux heures locales homonymes restent présentes |
| Total d'heures | Égal à la concaténation de l'index attendu ; aucune constante 8 760 |
| Complétude | Zéro trou, doublon, timestamp hors index ou infini ; un `NaN` de feature n'est recevable que s'il correspond à une absence PIT explicitement masquée, avec timestamps `NaT`, et n'est jamais rempli |
| Quantiles | `q10 <= q50 <= q90` pour chaque heure |
| Label prix | Aucun label postérieur à `F−1` |
| Origine historique | Pour chaque `T`, origine exactement `T−1 08:00 Europe/Paris` |
| Snapshot exogène | `snapshot <= origine` et `revision <= origine`; un dépassement d'une seconde doit faire échouer le test |
| Réalisation physique | Toute calibration d'erreur charge/RES s'arrête à `F−2` |
| Source OOF | Refus de toute source in-sample, backfill latest ou source non identifiée |
| Indépendance | Aucune feature/dépendance dont le nom ou la lignée contient Storm/MKOnline |
| Auto-inclusion | Le forecast de `F` ne peut pas entrer dans son propre entraînement |
| Déterminisme | Deux runs avec mêmes hashes/seed donnent le même forecast, à une tolérance numérique préfixée |
| Sensibilité de fenêtre | Modifier une ligne récente change le fit ; modifier une ligne antérieure à `F−365` ne le change pas |
| Sortie live | Exactement 23/24/25 heures du jour `F`, valeurs finies et quantiles ordonnés |
| Manifeste | Dates, heures, origines, cutoffs, sources et hashes présents |
| Première activation | `F=2026-08-16` échoue si la moindre journée du corpus est non vérifiée |
| Parité pays | Même contrat pour le runner FR et le runner multizone |
| Rapport | Libellé « rolling-365 » ; aucune métrique de l'ancien modèle figé présentée comme performance du nouveau |

Tests de non-régression conseillés :

```text
test_window_exact_365_consecutive_local_days
test_window_is_f_minus_365_through_f_minus_1
test_leap_year_still_has_365_local_days
test_dst_spring_and_autumn_preserve_23_25_hours
test_reject_missing_duplicate_nonfinite_or_crossed_quantiles
test_reject_price_label_after_f_minus_1
test_reject_forecast_origin_not_t_minus_1_0800_paris
test_reject_covariate_snapshot_one_second_after_origin
test_reject_in_sample_latest_storm_or_mkonline_sources
test_live_forecast_cannot_train_on_itself
test_identical_inputs_are_deterministic
test_manifest_contains_window_sources_cutoffs_and_hashes
test_first_live_activation_requires_complete_verified_corpus
test_fr_and_generic_runner_share_the_same_contract
```

## 7. Protocole de validation A / B1 / B2 / shadow

### 7.1 Walk-forward historique

Pour chaque jour évalué `E` :

1. construire uniquement le corpus `E−365 ... E−1` ;
2. réajuster le correcteur avec la recette figée ;
3. prévoir `E` ;
4. geler et hasher le candidat ;
5. charger ensuite les prix réels et, si disponible, Storm ;
6. scorer uniquement la journée physique complète.

Un seul fit sur une année suivi de 365 prédictions n'est pas un test du réentraînement quotidien. Le walk-forward doit réellement refaire 365 fits.

### 7.2 Blocs historiques existants

Le calendrier déjà utilisé dans le projet est cohérent et doit rester fixe :

| Bloc | Dates | Rôle |
|---|---|---|
| A | 12/08/2024–13/04/2025, 245 jours | cross-fit, mise au point et apprentissage autorisé |
| B1 | 14/04/2025–12/06/2025, 60 jours | sélection entre familles pré-déclarées |
| B2 | 13/06/2025–11/08/2025, 60 jours | veto gelé, aucun retuning |
| Final historique | 12/08/2025–11/08/2026, 365 jours | one-shot de confirmation |
| Shadow live | 60–90 jours futurs consécutifs | preuve prospective avant promotion |

Attention : plusieurs résultats de ces blocs ont déjà été consultés dans le projet. Ils sont utiles pour une analyse rétrospective, mais ne redeviennent pas « vierges » pour une nouvelle recherche massive. Si B2 ou Final a influencé le choix d'une feature, la seule confirmation honnête est le shadow prospectif suivant un protocole enregistré à l'avance.

### 7.3 Candidats à comparer

Les variantes doivent être peu nombreuses et pré-déclarées :

- **C0 — autonome actuel** : correcteur expanding/frozen actuel, évalué sans composante MKOnline ;
- **C1 — rolling365** : même schéma causal, seule la fenêtre change ;
- **C2 — rolling365 + quantiles fondamentaux** : quantiles charge/éolien/solaire/charge résiduelle ;
- **C3 — rolling365 + C2 + NWP/outages/commodities** : seulement après disponibilité PIT démontrée ;
- **C4 — ensemble multi-fenêtres** : 90/180/365/730 ou agrégation BOA, uniquement si pré-déclaré et validé séparément.

Chaque ajout doit battre C1 avant d'être combiné. Cela permet d'attribuer le gain et limite le biais de recherche.

## 8. Scoring et critères de promotion

### 8.1 Appariement strict

Une journée entre dans une comparaison seulement si elle contient :

- tous les prix réels ;
- tous les forecasts candidats ;
- tous les forecasts du comparateur ;
- exactement les 23/24/25 heures physiques attendues.

Pas d'interpolation, forward-fill, remplacement par un autre modèle ni journée partielle. Le rapport affiche le nombre de jours exclus et la raison. Pour Storm, viser 100 % de couverture ; en dessous de 95 % de journées complètes ou en présence d'un long trou contigu, ne pas conclure à une supériorité annuelle.

### 8.2 Win rate par statistique

Pour une métrique de perte `L` telle que MAE, RMSE ou MAPE :

```text
win_rate_L = moyenne_jour [ L(candidat, jour) < L(Storm, jour) ]
```

Pour une métrique de score telle que R² ou corrélation, le sens est inversé : le candidat gagne si son score est supérieur. Une égalité n'est pas une victoire et reste dans le dénominateur. La convention doit être identique dans le HTML, l'application et les fichiers CSV.

| Statistic | Règle de victoire journalière |
|---|---|
| MAE, RMSE, MAPE, sMAPE, pinball loss | valeur la plus basse |
| Biais/Mean Error | plus petite valeur absolue, pas le nombre signé le plus faible |
| Écart-type des erreurs | valeur la plus basse |
| R², corrélation | valeur la plus haute |
| Couverture q10–q90 | plus petit écart absolu à 80 % ; largeur comparée seulement à couverture acceptable |

Si R² ou corrélation est indéfini pour une courbe réelle constante, la journée est `N/A` pour cette seule statistique et son dénominateur apparié est affiché. Elle reste incluse pour MAE/RMSE.

MAPE est instable lorsque le prix réel est nul ou proche de zéro, situation possible sur le marché électrique. Il reste affiché pour continuité, mais son epsilon/règle de zéro doit être écrit dans le manifeste et il ne doit pas être un gate primaire. Ajouter sMAPE ou MASE comme contrôle robuste est recommandé.

### 8.3 Gate principal par pays

Pour FR, DE, BE et NL, sur B2/Final puis shadow :

1. **win rate journalier MAE vs Storm > 50 %** ;
2. MAE horaire agrégée du candidat strictement inférieure à celle de Storm sur les mêmes heures ;
3. MAE agrégée inférieure à la production actuelle et win rate MAE > 50 % face à la production actuelle ;
4. aucune dégradation supérieure à 5 % face à la production actuelle sur un trimestre, un mois critique ou une strate de prix pré-déclarée ;
5. pas de détérioration majeure sur jours DST, prix négatifs, top/bottom déciles et épisodes extrêmes ;
6. q10/q50/q90 évalués par pinball loss, avec couverture et largeur de l'intervalle 80 %.

Les contrôles 3 et 4 contre la production publiée sont exécutés uniquement après gel de la recette autonome. Ils servent à décider du déploiement et ne peuvent ni sélectionner une famille, ni régler un hyperparamètre, ni apprendre un poids à partir de MKOnline.

Pour ES : même gate contre la production actuelle et les baselines autonomes ; le comparateur Storm est `N/A` jusqu'à identification et audit d'une série native officielle.

Séries Storm autorisées, évaluation seulement :

```text
FR  power.price.fr.euromwh.h.fcst.3mv.storm
DE  power.price.de.euromwh.h.fcst.3mv.storm
BE  power.price.be.euromwh.h.fcst.3mv.storm
NL  power.price.nl.euromwh.h.fcst.3mv.storm
ES  aucune série officielle vérifiée
```

### 8.4 Force statistique de la conclusion

Le point estimate `win rate > 50 %` est nécessaire mais pas suffisant pour une affirmation forte. La conclusion « supériorité démontrée » requiert aussi :

- borne haute d'un IC unilatéral 95 % de `MAE_candidat − MAE_Storm` strictement inférieure à zéro ;
- borne basse d'un IC 95 % du win rate supérieure à 50 % ;
- bootstrap par blocs de jours, longueur 7 pré-déclarée, pour préserver l'autocorrélation ;
- correction de Holm sur les quatre comparaisons pays FR/DE/BE/NL si l'affirmation porte sur « tous les pays ».

Si seul le point estimate dépasse 50 %, écrire : **« win rate observé > 50 %, supériorité statistique non démontrée »**.

## 9. Résultats locaux déjà obtenus et décision

### 9.1 Point de départ autonome, sans MKOnline

Recalcul strict sur les artefacts réalisés disponibles au 15 août 2026, avec la colonne autonome `residual_corrected__q50`, Storm natif apparié heure par heure et journées physiques complètes uniquement :

| Zone | MAE journalière moyenne autonome | MAE journalière moyenne Storm | Jours gagnés / comparables | Win rate MAE quotidien | Statut |
|---|---:|---:|---:|---:|---|
| FR | 12,226 | 11,668 | 178 / 367 | 48,50 % | Sous 50 % sans MKOnline |
| DE | 11,363 | 11,162 | 199 / 367 | 54,22 % | Au-dessus de 50 %, mais MAE moyenne encore moins bonne |
| BE | 11,439 | 10,367 | 148 / 367 | 40,33 % | Écart prioritaire à réduire |
| NL | 11,230 | 10,638 | 182 / 365 | 49,86 % | Historique Statistics partiel ; résultat proche de 50 % |
| ES | 9,762 sur le snapshot global candidat | N/A | N/A | N/A | Aucun Storm officiel vérifié |

Ces chiffres sont un état daté, pas une promesse. Ils montrent aussi pourquoi le win rate ne suffit pas : DE gagne davantage de jours, mais perd plus fortement certains jours, ce qui maintient sa MAE moyenne au-dessus de Storm. L'objectif « tous les pays > 50 % sans MKOnline » n'est donc pas atteint ; seul DE dépasse actuellement le seuil observé, et ES reste non mesurable.

### 9.2 Expériences déjà rejetées

Ces résultats sont à conserver comme preuve négative ; ils ne doivent pas être relancés jusqu'à obtenir un résultat favorable.

| Essai | Résultat B1 | Décision |
|---|---|---|
| Revision-spread DE | meilleur gain vs référence figée : +0,0096 €/MWh | Non promu : très inférieur au gate de +0,10 |
| Revision-spread BE | +0,0016 €/MWh vs référence, mais −0,0005 vs contrôle calendrier | Non promu |
| Revision-spread NL | meilleur candidat encore négatif vs référence : −0,0077 €/MWh | Non promu |
| Nucléaire quotidien BE | référence 13,0740 ; candidat 13,0889 ; gain −0,0149 ; win rate 45 % ; IC bootstrap [−0,0504 ; +0,0216] | Non promu |
| OpenMeteo D−2 FR | référence 10,1853 ; candidat 10,2264 ; gain −0,0411 ; IC [−0,1037 ; +0,0202] | Non promu |
| Eco2Mix « revision latest » | historique non-PIT ; le même-day J−1 n'est pas exporté de manière exploitable | Exploratoire uniquement |

Artefacts :

- `runs/tmp/multizone_exogenous_de_b1.json`
- `runs/tmp/multizone_exogenous_be_b1.json`
- `runs/tmp/multizone_exogenous_nl_b1.json`
- `runs/tmp/multizone_exogenous_be_nuclear_b1.json`
- `runs/tmp/openmeteo_d2_ab1b2/screen_b1.json`
- `runs/tmp/eco2mix_screen_b1.json`

Conclusion : la priorité scientifique est de construire des **quantiles de fondamentaux à partir des point forecasts authentiques et de leurs erreurs historiques**, pas d'ajouter d'autres variantes du revision-spread déjà testé.

Un audit supplémentaire de la fenêtre 16/08/2025–15/08/2026 révèle aussi un défaut concret de qualité d'input : chacun des cinq parquets historiques de charge résiduelle omet environ **543 à 545 heures sur 8 760**, presque toutes les heures locales 22:00–23:00, sur 348–349 jours. Les matrices archivées portent les mêmes `NaN` et le correcteur sait les traiter ; aucune valeur n'a été remplie. Cela ne prouve toutefois pas que la donnée n'existait pas chez le fournisseur : il peut s'agir d'un défaut de collecte historique. Avant un modèle plus complexe, il faut donc auditer puis, si Saturn restitue les révisions exactes, rematérialiser causalement ces fins de journée et mesurer leur gain. C'est une priorité P0 de qualité des inputs.

## 10. Revue récente de la littérature primaire

### 10.1 Fondamentaux probabilistes — priorité la plus forte

Uniejewski et Ziel utilisent des prévisions probabilistes de charge, vent et solaire sur le marché allemand 2015–2023. L'ajout des distributions, particulièrement des quantiles extrêmes et de la charge résiduelle, améliore la prévision de prix jusqu'à environ 13 %. L'article fournit aussi la bonne chronologie opérationnelle : les erreurs physiques sont calibrées seulement jusqu'à `d−2` ([Renewable Energy, 2026](https://doi.org/10.1016/j.renene.2026.125844), [version ouverte](https://arxiv.org/abs/2501.06180)).

**Transfert au projet :** transformer les cinq point forecasts PIT de charge résiduelle déjà disponibles en quantiles causalement calibrés, puis — si les séries sources sont accessibles — faire de même séparément pour charge, éolien et solaire. Commencer par historique simulation/quantile regression avant tout réseau complexe.

### 10.2 NWP direct — preuve solide mais horaire de coupure à démontrer

Sgarlato et Ziel intègrent directement vitesse/direction du vent, irradiation, nuages et température de plusieurs lieux européens. Pour des horizons de 2 à 4 jours, ils rapportent une réduction de RMSE de 10 à 20 %, avec un rôle important du vent du nord de l'Allemagne ([IEEE Transactions on Power Systems, 2023](https://doi.org/10.1109/TPWRS.2022.3180119)).

**Transfert au projet :** tester des agrégats zonaux et voisins issus d'un cycle NWP archivé. La preuve publiée porte surtout au-delà du day-ahead ; le gain incrémental exact à D−1 08 reste à mesurer. Utiliser une réanalyse ou la dernière météo téléchargée après coup serait un oracle et donc une fuite.

### 10.3 Modèles avec exogènes et multi-pays

NBEATSx étend N-BEATS avec des variables exogènes et rapporte, sur plusieurs marchés et années, près de 20 % d'amélioration face à N-BEATS et jusqu'à 5 % face à des méthodes spécialisées établies ([Olivares et al., International Journal of Forecasting, 2023](https://doi.org/10.1016/j.ijforecast.2022.03.001)).

Tschora et al. trouvent jusqu'à 15 % de gain avec des datasets enrichis, incluant historiques de prix voisins et forecasts de consommation/production, et montrent l'intérêt du multi-pays ([Applied Energy, 2022](https://doi.org/10.1016/j.apenergy.2022.118752), [article ouvert](https://www.lrde.epita.fr/dload/papers/tschora.22.apen.pdf)). Une partie importante de leur gain utilise toutefois le prix suisse publié à 11:15 : il est **inutilisable** dans notre produit figé à 08:00. Seuls les prix voisins déjà publiés avant 08:00 ou leurs retards D−1/D−2/D−7 sont recevables.

Une étude 2026 exploite explicitement les horaires de fermeture asynchrones et obtient 22 % de gain en BE et 9 % en SE3. C'est une preuve de valeur des informations pré-clôture, mais aussi un exemple de feature à rejeter si elle n'existe qu'après notre cutoff 08:00 ([Applied Energy, 2026](https://doi.org/10.1016/j.apenergy.2025.127077)).

### 10.4 Gaz, nucléaire et pays voisins — drivers causaux, pas encore preuve forecast PIT

Une étude par modèles causaux structurels sur la France et l'Espagne 2018–2023 identifie le gaz comme principal driver des prix, la disponibilité nucléaire française comme facteur structurel important, ainsi que la charge résiduelle et la disponibilité nucléaire des voisins. Elle montre aussi que de simples corrélations peuvent être trompeuses à cause des confondeurs ([Schreyer et al., Nature Communications, 2026](https://doi.org/10.1038/s41467-026-75433-7)).

**Transfert au projet :** prioriser capacité nucléaire disponible et indisponibilités planifiées FR, prix TTF/EUA connu avant 08:00, charge résiduelle voisine et hydro. Cette étude est explicative, non un backtest strict D−1 08 : chaque signal doit encore prouver sa disponibilité PIT et son gain B1.

### 10.5 Fenêtre de calibration et ensembles

Fezzi et Mosetti montrent que la taille de la fenêtre peut modifier fortement la performance et que même des modèles simples deviennent compétitifs avec une fenêtre bien calibrée ([The Energy Journal, 2020](https://doi.org/10.5547/01956574.41.4.cfez)). Marcjasz, Serafin et Weron constatent qu'une longueur optimale varie selon marché/modèle et recommandent l'agrégation de plusieurs fenêtres comme solution robuste ([Energies, 2018](https://doi.org/10.3390/en11092364)). Hubicka et al. étudient également l'agrégation de forecasts issus de fenêtres distinctes ([IEEE Transactions on Sustainable Energy, 2019](https://doi.org/10.1109/TSTE.2018.2869557)).

**Transfert au projet :** 365 jours est une baseline gouvernable, pas une vérité universelle. Après validation de C1, comparer de façon pré-déclarée 90/180/365/730 ou leur agrégation, sans choisir chaque matin la meilleure fenêtre sur le résultat du jour.

Une prépublication 2026 combine apprentissage partiellement en ligne et Bernstein Online Aggregation sur plusieurs grands marchés européens et rapporte 14–17 % de réduction de MAE face aux benchmarks, avec moins de calcul ([El Mahtout & Ziel, 2026](https://doi.org/10.48550/arXiv.2601.02856)). Prometteur, mais à reproduire avant toute promotion.

### 10.6 Spikes et probabilités

Cerasa et Zani proposent un filtrage robuste glissant des spikes sur six marchés et rapportent jusqu'à 4 % de gain selon le modèle ([Applied Energy, 2025](https://doi.org/10.1016/j.apenergy.2025.125357)). Le filtre peut modifier la représentation d'entraînement, mais il ne doit jamais supprimer des heures de l'évaluation : tous les spikes réels restent scorés.

Brusaferri et al. utilisent une recalibration conformale en ligne d'ensembles neuronaux et améliorent la couverture horaire et la stabilité des scores probabilistes sur plusieurs marchés ([Applied Energy, 2025](https://doi.org/10.1016/j.apenergy.2025.126412)). C'est pertinent pour q10/q90 et le PICP, mais ce n'est pas en soi une preuve de baisse de MAE.

### 10.7 Texte, news et images — faible priorité

Le benchmark NSW-EPNews trouve un gain marginal des news pour les modèles classiques et seulement modeste pour les LLM, avec des séquences souvent hallucinées ou mal formées ([Bi et al., 2025, prépublication primaire](https://arxiv.org/abs/2506.11050)). Une étude australienne 2026 rapporte environ 0,39 point de NRMSE sur un cas five-minute, dans un marché et un horizon très différents des nôtres ([Applied Sciences, 2026](https://doi.org/10.3390/app16010200)).

**Transfert au projet :** préférer les annonces structurées d'indisponibilité et de maintenance aux embeddings de presse. Si un pilote news est tenté, archiver texte, URL, heure de publication et version avant 08:00 ; ne jamais autoriser un LLM à produire directement les 23/24/25 prix.

Il n'existe pas, dans les sources examinées, de preuve robuste que des **images brutes** améliorent notre forecast D−1 08. Les cartes météo, grilles ou satellites doivent être convertis en variables numériques auditables. Une image d'observation postérieure à 08:00, ou une mosaïque reconstruite avec des révisions futures, serait une fuite difficile à détecter.

## 11. Roadmap exogène classée

### P0 — à lancer d'abord

1. **Rolling365 causal sans nouvelle feature** : isoler le gain dû au rafraîchissement de fenêtre.
2. **Quantiles de charge résiduelle** à partir des cinq point forecasts PIT existants et des erreurs physiques jusqu'à D−2 : q01/q05/q10/q25/q50/q75/q90/q95/q99, dispersion, asymétrie, risques de rampes.
3. Si disponibles en PIT, **quantiles séparés charge/vent/solaire** : ils peuvent contenir plus d'information que leur seule combinaison résiduelle.
4. Baseline multi-fenêtres pré-déclarée 90/180/365/730, uniquement après C1.

### P1 — forte valeur attendue, travail de données nécessaire

1. **NWP archivé** : vent 100 m vitesse/direction, température 2 m, irradiation, nébulosité ; agrégats zone + voisins, cycle et publication <= 08:00.
2. **Nucléaire** : capacité disponible, indisponibilités planifiées et changements d'annonce, particulièrement FR/BE.
3. **Gaz et carbone** : TTF/EUA dernier prix utilisable avant 08:00 ou clôture D−2 ; charbon en complément DE/NL.
4. **Information inter-zones retardée** : prix D−1/D−2/D−7, écarts de charge résiduelle, rampes et spreads entre zones — jamais prix day-ahead non encore publié à 08:00.

### P2 — après les fondamentaux

1. hydro : remplissage réservoirs, ROR prévu, pompage et neige/hydrologie si forecast pré-cutoff ;
2. capacité d'interconnexion disponible publiée avant 08:00 ; les échanges réalisés ou schedules déterminés après enchère sont interdits ;
3. filtrage robuste de la représentation d'entraînement ;
4. ensemble BOA/online des fenêtres ou modèles ;
5. recalibration conformale des quantiles.

### P3 — recherche isolée

1. adaptation LoRA de Chronos, jamais quotidienne au départ ;
2. texte/news avec taxonomie d'événements et horodatage immuable ;
3. grilles ou images météo uniquement après conversion en features numériques et audit des licences.

## 12. Disponibilité à D−1 08:00, PIT et licences

| Signal | Statut probable à 08:00 | Preuve PIT exigée | Risque de licence/gouvernance | Priorité |
|---|---|---|---|---|
| Point forecasts Saturn de charge résiduelle | Oui pour les séries déjà auditées | snapshot + révision <= cutoff pour chaque jour | Faible à moyen, usage interne à documenter | P0 |
| Quantiles dérivés charge/vent/solaire | Conditionnel | point forecast disponible à 08:00 ; erreurs physiques seulement jusqu'à D−2 | Faible à moyen | P0 |
| NWP | Conditionnel | cycle, heure de publication et fichier brut archivés <= 08:00 | Moyen ; vérifier OpenMeteo/Météo-France/ECMWF pour usage production et redistribution | P1 |
| Eco2Mix forecast J−1 | RTE confirme « calculé la veille », pas une disponibilité garantie avant 08:00 | capture quotidienne horodatée ou historique de vintages | Moyen ; documenter licence et révisions | Bloqué tant que non-PIT |
| Eco2Mix réalisé | Oui après livraison, mais révisé plus tard | conserver la version observée à chaque date ou utiliser uniquement selon politique explicite | Moyen | Calibration jusqu'à D−2 |
| Nucléaire/outages | Conditionnel | publication et toutes révisions <= 08:00 | Moyen ; RTE/ENTSO-E/REMIT et droits de réutilisation à vérifier | P1 |
| TTF/EUA/charbon | Selon le feed | tick/clôture avec heure exacte <= 08:00 ; sinon D−2 | Élevé, souvent commercial et non redistribuable | P1 |
| Prix voisins retardés | Oui pour jours déjà clearés | timestamp de publication et contrat de marché | Moyen ; les données de prix peuvent avoir des restrictions fortes | P1 |
| Prix voisins du même day-ahead | Généralement non à 08:00 | ne promouvoir que si la zone source ferme et publie avant 08:00 | Moyen/élevé | Rejet par défaut |
| Capacité interconnexion | Conditionnel | publication du champ avant 08:00, pas seulement calcul ex post | Moyen | P2 |
| Flow/schedule réalisé | Souvent post-enchère ou post-livraison | preuve contraire champ par champ | Moyen/élevé | Rejet par défaut |
| News/texte | Oui seulement si publié avant 08:00 | archive immuable de l'article et de son timestamp | Élevé : copyright, stockage, provenance LLM | P3 |
| Storm/MKOnline | Disponible selon système | Sans objet | Interdit comme feature de recherche | Évaluation seulement |

RTE indique que le forecast de consommation day-ahead est calculé la veille, mais ne garantit pas sur cette page qu'il est publié avant 08:00 ; RTE précise aussi que les données réalisées sont ensuite consolidées puis finalisées, ce qui confirme le besoin de vintages ([RTE Eco2Mix consommation](https://www.rte-france.com/en/data-publications/eco2mix/electricity-consumption-france), [RTE téléchargements et révisions](https://www.rte-france.com/en/data-publications/eco2mix/download-indicators)). Les prix affichés dans Eco2Mix ont par ailleurs des conditions de réutilisation spécifiques liées aux bourses ([RTE, conditions des données de marché](https://www.rte-france.com/en/data-publications/eco2mix/market-data)). Certaines données ENTSO-E sont réutilisables sous CC BY 4.0, mais il faut vérifier la catégorie exacte et les conditions applicables au champ retenu ([ENTSO-E, liste officielle de données réutilisables du 18 octobre 2023](https://transparency.entsoe.eu/content/static_content/download?path=%2FStatic+content%2Fterms+and+conditions%2F231018_List_of_Data_available_for_reuse.pdf)).

## 13. Protocole expérimental concret sans MKOnline/Storm

### Phase A — ingénierie et cross-fit

- matérialiser le corpus PIT ;
- tester DST, cutoffs, valeurs manquantes et déterminisme ;
- apprendre sur A uniquement les choix, transformations et hyperparamètres qui peuvent être réglés ; chaque fit quotidien rolling conserve néanmoins sa fenêtre causale complète `E−365 ... E−1`, qui peut commencer avant A ;
- générer les prédictions cross-fittées de toutes les transformations apprises ;
- autoriser le debug et l'ablation, mais pas de revendication de performance.

### Phase B1 — sélection

- candidats et hyperparamètres enregistrés avant lecture des scores ;
- aucune donnée B2/Final chargée par le script ;
- comparer chaque famille à rolling365 simple et à l'autonome actuel sans MKOnline ; la production publiée, qui peut contenir MKOnline en FR/NL, n'est comparée qu'après gel de la recette comme contrôle de décision, jamais comme signal de sélection ou de poids ;
- gate minimal suggéré : gain MAE >= 0,10 €/MWh, gain positif sur les deux moitiés chronologiques de 30 jours et borne basse bootstrap > 0 ;
- une famille échouée est archivée comme non promue.

### Phase B2 — veto gelé

- un seul run par recette scellée ;
- aucun retuning, aucun changement de clipping, liste de features ou poids ;
- Storm n'est chargé qu'après hash du candidat ;
- échec d'un gate primaire = veto, sans « petite correction » puis nouveau calcul sur le même B2.

### Final historique et shadow

- one-shot final si le bloc était réellement intact pour cette recette ; sinon résultat descriptif uniquement ;
- 60–90 jours shadow consécutifs avec forecast enregistré à 08:00 ;
- score automatique dès disponibilité des prix/Storm, sans changer le forecast ;
- rapport hebdomadaire de couverture, drift, latence, fail-closed et métriques par strate ;
- promotion seulement après comité de validation et versionnage `rolling365_v1`, avec rollback immédiat vers la version précédente.

## 14. Risques de migration

| Risque | Effet | Mitigation |
|---|---|---|
| Perte de régimes rares au-delà d'un an | Moins bonne gestion des crises/spikes | multi-fenêtres ou features de régime ; comparaison 365/730 |
| Fenêtre unique non optimale par pays/saison | Gain instable | test pré-déclaré multi-fenêtres, jamais choix post-hoc quotidien |
| Replays non réellement PIT | Gain artificiel | snapshots/hashes/origines obligatoires ; fail-closed |
| Actual physique D−1 utilisée trop tôt | Fuite dans les quantiles fondamentaux | calibration des réalisations strictement <= D−2 |
| DST aplati à 24 h | heures manquantes ou dupliquées | index physique UTC 23/24/25 |
| Données API révisées | backtest non reproductible | stockage append-only des vintages bruts |
| Licence incompatible | blocage production/redistribution | revue juridique par source avant B2 |
| Hyperparamètres retunés chaque jour | surapprentissage et identité mouvante | recette/seed/schéma figés |
| Nondéterminisme CatBoost/threads | hashes de forecast instables | seed et threads enregistrés ; tolérance testée |
| Poids blend FR/NL devenus invalides | le meilleur autonome ne donne pas le meilleur forecast final | re-gate séparé du blend après gel de l'autonome |
| Multiplication des features/candidats | faux positifs B1/B2 | registre d'expériences, nombre de familles limité, correction multiple |
| Shadow trop court | win rate >50 % dû au hasard | 60–90 jours minimum, IC par blocs, poursuite si inconclusif |
| ES sans Storm officiel | faux objectif impossible à auditer | `N/A`, comparaison production/baselines jusqu'à série vérifiée |
| Fallback silencieux | rapport attribué au mauvais modèle | nom/version explicites et raison de fallback dans HTML |

## 15. Checklist de mise en service

Le rolling365 est prêt pour le shadow seulement si toutes les cases suivantes sont vraies :

- [ ] 365 dates civiles exactes par pays, sans trou ;
- [ ] toutes les journées 23/24/25 validées ;
- [ ] origines historiques exactement à T−1 08:00 Paris ;
- [ ] snapshots/révisions de toutes les features <= origine ;
- [ ] réalisations physiques utilisées seulement jusqu'à D−2 ;
- [ ] zéro dépendance Storm/MKOnline dans le candidat autonome ;
- [ ] code, recette, schéma, seed, corpus et forecast hashés ;
- [ ] walk-forward quotidien réellement exécuté ;
- [ ] B1 passé, B2 passé sans retuning ;
- [ ] métriques complètes contre production et, si officiel, Storm ;
- [ ] ES affiche Storm `N/A` ;
- [ ] 60–90 jours shadow complétés avec couverture suffisante ;
- [ ] rollback testé ;
- [ ] rapport HTML distingue clairement production actuelle, rolling backtest, PIT replay et issued live.

## 16. État de l'implémentation au 15 août 2026

Les briques de sécurité nécessaires au shadow sont maintenant présentes, sans activation du rolling-365 dans le forecast de production :

- `chronos2_hourly/rolling_refit.py` sélectionne une fenêtre exacte de 365 jours civils, contrôle les journées 23/24/25 heures, les origines, les timestamps PIT, les quantiles Chronos bruts et l'absence de Storm/MKOnline ;
- `chronos2_hourly/rolling_refit_loader.py` ne charge que des blocs immuables, checksumés et liés à la bonne zone, à la bonne configuration et au bon bundle autonome ;
- `chronos2_hourly/rolling_refit_backfill.py` permet de reconstruire une journée historique depuis les parquets de révisions locaux, sans interpolation et avec comparaison aux features archivées ;
- `chronos2_hourly/rolling_capture.py` capture désormais, après publication atomique du forecast officiel, le bloc causal du jour puis le finalise au run suivant lorsque le prix réalisé est disponible ;
- `run_mkonline_live_zone.py` active cette capture prospective par défaut sous `runs/rolling365_shadow/<pays>` ; l'option explicite `--no-rolling365-capture` permet de la désactiver ;
- `chronos2_hourly/exogenous_research_registry.py` et `exogenous_research_registry.example.yaml` empêchent l'essai d'un signal exogène tant que disponibilité à 08:00, historique PIT, licence, lignée, schéma numérique et couverture ne sont pas prouvés.

La capture intervient exclusivement **après** le renommage atomique de l'archive officielle. Son échec est non fatal et ne peut ni modifier ni annuler le forecast publié. Elle n'est pas exécutée lors d'un replay PIT. La suite ciblée rolling, live, historique et application totalise **204 tests réussis** lors de cette livraison.

Le réentraînement rolling-365 lui-même reste désactivé. Il ne pourra entrer en shadow qu'après le backfill checksumé des blocs historiques, l'arrivée du dernier prix réalisé requis et le passage complet du loader fail-closed. Aucun poids, hyperparamètre, registre de production ou forecast publié n'a été modifié par ce chantier.

## 17. Conclusion

Le réentraînement quotidien recommandé concerne le **correcteur résiduel**, pas Chronos-2. Une fenêtre exacte de 365 jours civils permettra au modèle de s'adapter aux régimes récents, mais elle ne doit être considérée que comme une baseline à valider : la littérature montre que la longueur optimale varie et que l'agrégation de fenêtres peut être plus robuste.

La piste exogène la mieux étayée et la plus compatible avec le cutoff de 08:00 est la distribution de charge/éolien/solaire — ou de charge résiduelle — construite à partir de forecasts PIT et d'erreurs physiques disponibles jusqu'à D−2. Elle doit précéder de nouveaux essais de revision-spread, de news ou d'images. NWP archivé, nucléaire, gaz/carbone et information inter-zones viennent ensuite, sous réserve de preuve PIT et de licence.

Enfin, Storm doit rester un adversaire de mesure, jamais un professeur caché : le candidat est produit, gelé et hashé avant que la série Storm soit chargée. C'est cette séparation qui rendra le win rate crédible.
