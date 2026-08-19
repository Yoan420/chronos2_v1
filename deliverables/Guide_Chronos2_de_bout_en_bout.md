# Chronos-2 multi-pays

## Guide de bout en bout du modèle de prévision day-ahead

**Version 1.0 — 14 août 2026**  
**Périmètre : France, Allemagne, Belgique, Pays-Bas et Espagne**  
**Public : utilisateurs métier, responsables, analystes et toute personne souhaitant comprendre le modèle sans être spécialiste de la data science**

---

## Comment utiliser ce document

Ce guide répond à quatre questions simples :

1. **Que prévoit exactement le système ?**
2. **Comment passe-t-on des données disponibles à 08:00 à une courbe de prix pour le lendemain ?**
3. **Comment lire P10, P50, P90 et toutes les Statistics ?**
4. **Quelles protections empêchent une comparaison trompeuse ou l'utilisation d'informations futures ?**

Pour une lecture rapide, lire le résumé, le schéma de bout en bout, la fiche par pays et la section « Lire les résultats ». Les autres chapitres servent de référence.

> **Idée essentielle** — Le système produit un forecast avant la livraison, le fige, puis seulement après compare cette prévision au prix réalisé et à Storm. Storm ne participe jamais à la fabrication du forecast.

---

## 1. Résumé en une minute

L'application prévoit le **prix day-ahead de l'électricité**, en **EUR/MWh**, pour chaque heure du prochain jour de livraison.

Elle couvre aujourd'hui cinq marchés :

- France (FR) ;
- Allemagne / Luxembourg (DE) ;
- Belgique (BE) ;
- Pays-Bas (NL) ;
- Espagne (ES).

Pour chaque heure, elle publie trois valeurs :

- **P50**, la prévision centrale ;
- **P10**, un scénario bas ;
- **P90**, un scénario haut.

Le cœur autonome est identique dans son principe pour tous les pays :

1. **Amazon Chronos-2** dessine une première courbe de prix probabiliste ;
2. un **correcteur d'erreurs historiques** ajuste cette courbe en fonction du contexte ;
3. en France et aux Pays-Bas, un forecast **MKOnline** validé apporte un second avis avec un poids fixe ;
4. le résultat est contrôlé, archivé et protégé par des empreintes SHA-256 ;
5. une fois les prix réels connus, le rapport mesure la performance et la compare à **Storm**.

Le forecast porte normalement sur 24 heures. Il en contient **23 ou 25 lors des changements d'heure**, ce qui est normal : le système suit la véritable journée physique de chaque marché.

---

## 2. La chaîne complète, en un regard

<div class="pipeline" role="img" aria-label="Chaîne de prévision : données disponibles, contrôles, Chronos-2, correcteur résiduel, mélange MKOnline optionnel, quantiles, archive, puis évaluation Storm">
  <div class="pipeline-step"><strong>1. Données disponibles</strong><br>Prix passés, calendriers et charges résiduelles prévues</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><strong>2. Gel à D−1 08:00</strong><br>Uniquement l'information réellement disponible</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><strong>3. Chronos-2</strong><br>Première courbe P10 / P50 / P90</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><strong>4. Correcteur résiduel</strong><br>Correction des biais récurrents</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><strong>5. MKOnline si validé</strong><br>FR et NL uniquement</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><strong>6. Forecast figé</strong><br>Contrôles, archive et checksums</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step evaluation"><strong>7. Évaluation après coup</strong><br>Prix réalisé + Storm + Statistics</div>
</div>

La partie située avant « Forecast figé » est la **chaîne de prédiction**. La partie Storm se trouve après : elle appartient exclusivement à l'**évaluation**.

---

## 3. Ce que le modèle prévoit — et ce qu'il ne prévoit pas

### 3.1 Le produit prévu

Le modèle prévoit le prix de l'enchère **day-ahead** pour chaque heure du jour de livraison suivant. Il ne prévoit pas :

- le prix intraday ;
- le prix d'équilibrage ;
- un volume échangé ;
- un flux physique entre pays ;
- une décision de trading automatique.

Le prix peut être positif, nul ou négatif. Un prix négatif est possible sur le marché et ne constitue pas, à lui seul, une erreur du modèle.

### 3.2 Le calendrier D et D−1

Dans ce document :

- **D** désigne le jour de livraison ;
- **D−1** désigne la veille ;
- **08:00** est l'heure civile de gel utilisée par la chaîne actuelle, dans le fuseau Europe/Paris.

À 08:00 le jour D−1, le système ne conserve que les données publiées ou révisées au plus tard à cette heure. Il refuse toute révision plus tardive, même si elle serait plus précise.

Cette règle simule honnêtement ce qu'un utilisateur pouvait savoir au moment du forecast.

### 3.3 Une nuance importante sur les prix historiques

Le timestamp d'un prix day-ahead correspond à son **heure de livraison**, pas à son heure de publication. Toute la courbe du jour D−1 a déjà été fixée lors du clearing de D−2. Elle est donc connue à 08:00 en D−1 et peut légitimement servir au contexte historique.

Le modèle accepte ainsi les prix de livraison de D−1, mais refuse les prix du jour D qu'il cherche précisément à prévoir.

---

## 4. Les données utilisées en production

### 4.1 L'historique des prix day-ahead

La série de prix passée donne à Chronos-2 le contexte du marché : niveaux habituels, alternance heures creuses / heures de pointe, comportements de semaine et changements de régime.

La cible est propre à chaque zone : FR, DE-LU, BE, NL ou ES. Les prix sont exprimés en EUR/MWh et alignés sur une grille horaire UTC, puis présentés dans l'heure locale du pays.

La cible horaire n'est pas interpolée : un trou non admissible provoque un refus, pas une valeur inventée.

### 4.2 Les cinq prévisions de charge résiduelle

Chaque modèle utilise le panel prévu de charge résiduelle de :

- la France ;
- l'Allemagne ;
- la Belgique ;
- les Pays-Bas ;
- l'Espagne.

La **charge résiduelle** représente, de façon simplifiée, la demande restant à satisfaire après déduction d'une partie de la production renouvelable. Un niveau élevé peut signaler un système plus tendu ; un niveau faible peut être associé à davantage d'offre renouvelable disponible.

Utiliser les cinq pays aide le modèle à tenir compte du caractère interconnecté des marchés européens, sans prétendre prévoir directement les flux transfrontaliers.

### 4.3 Le calendrier connu à l'avance

Le modèle connaît des informations sans ambiguïté avant la livraison :

- heure de la journée ;
- jour de la semaine ;
- week-end ou jour ouvré ;
- jours fériés et calendriers des cinq pays ;
- forme intra-journalière des courbes prévues.

### 4.4 Les variables dérivées

Le correcteur transforme les données brutes en indicateurs de contexte, par exemple :

- largeur et asymétrie de l'intervalle Chronos-2 ;
- niveau moyen, minimum, maximum et amplitude de la journée ;
- rampes entre heures successives ;
- dispersion des charges résiduelles entre pays ;
- écart entre la situation locale et la situation régionale.

Le schéma déployé comporte **175 variables explicatives**. Leur liste et leur ordre sont gelés. Une variable manquante, inattendue ou réordonnée provoque un refus au lieu d'une adaptation silencieuse.

### 4.5 Ce qui n'est pas une entrée du modèle

Le forecast de production n'utilise pas :

- Storm ;
- les prix futurs réellement observés ;
- des révisions arrivées après le cutoff ;
- des articles, images ou réseaux sociaux non audités ;
- un proxy inventé lorsqu'une série officielle manque ;
- des lags ou moyennes glissantes du prix observé ajoutés au correcteur résiduel.

Chronos-2 utilise bien l'historique causal du prix comme contexte. En revanche, le second étage ne rajoute pas artificiellement une longue collection de variables de prix passé susceptibles de rendre l'évaluation moins robuste.

---

## 5. La sélection causale des données : « point-in-time »

Une série énergétique peut être révisée plusieurs fois. Utiliser aujourd'hui sa dernière version pour simuler une prévision ancienne donnerait au modèle une information qu'il ne possédait pas encore.

Le système applique donc une logique **point-in-time**, ou PIT :

1. il fixe le jour D et le cutoff D−1 à 08:00 ;
2. il élimine tout snapshot ou toute révision postérieure ;
3. pour chaque heure de livraison, il conserve la dernière version admissible ;
4. il vérifie que les 23, 24 ou 25 heures physiques sont toutes couvertes ;
5. il refuse les doublons, les valeurs non numériques et les trous non autorisés.

> **Pourquoi c'est important** — Un backtest utilisant des données corrigées après coup peut sembler excellent tout en étant impossible à reproduire en réel. La sélection PIT réduit ce risque de « voir le futur ».

Les données live doivent également être suffisamment fraîches. Le contrat actuel contrôle la couverture exacte du jour D et refuse les vintages trop anciens au lieu de publier discrètement une prévision obsolète.

---

## 6. Étape 1 du modèle : Amazon Chronos-2

Chronos-2 est un modèle pré-entraîné spécialisé dans les séries temporelles. Il reçoit :

- l'historique causal du prix ;
- le contexte temporel ;
- les variables futures autorisées et déjà connues, notamment les charges résiduelles prévues.

La configuration actuelle utilise jusqu'à **2 048 heures de contexte**, soit environ douze semaines de données horaires récentes.

Il produit une première courbe de scénarios :

- `q10` : estimation basse ;
- `q50` : estimation centrale ;
- `q90` : estimation haute.

Chronos-2 joue le rôle du **prévisionniste principal** : il dessine la forme générale de la journée, les pointes, les creux et l'incertitude.

Il n'est toutefois pas supposé être parfait pour chaque zone et chaque régime de prix. Un deuxième étage apprend donc à corriger ses erreurs récurrentes.

---

## 7. Étape 2 : le correcteur résiduel

### 7.1 Ce qu'il apprend

Pour chaque ancienne heure, le système calcule l'erreur de la prévision centrale :

**erreur résiduelle = prix réalisé − P50 Chronos-2**

Le correcteur apprend dans quels contextes Chronos-2 tend à être trop haut ou trop bas : heure de pointe, week-end, jour férié, forte rampe de charge résiduelle, divergence entre pays, etc.

### 7.2 Deux modèles complémentaires

La recette autonome déployée combine à parts égales :

- un modèle **CatBoost** ;
- un modèle **HistGradientBoosting** de scikit-learn.

Ce sont deux familles de modèles à base d'arbres. Chacune cherche des relations non linéaires entre le contexte et l'erreur attendue. Leur correction est moyennée à **50 % / 50 %**.

La correction finale est bornée à **±40 EUR/MWh**. Cette limite évite qu'un second étage très confiant déplace excessivement la courbe de base.

### 7.3 La correction commune des quantiles

La même correction est ajoutée à P10, P50 et P90 :

- nouveau P10 = ancien P10 + correction ;
- nouveau P50 = ancien P50 + correction ;
- nouveau P90 = ancien P90 + correction.

Cette méthode conserve automatiquement :

- l'ordre **P10 ≤ P50 ≤ P90** ;
- la largeur de l'intervalle P10–P90.

Le correcteur déplace donc la bande sans prétendre recalculer une nouvelle incertitude complète.

---

## 8. Étape 3 : le mélange MKOnline, uniquement lorsqu'il est validé

La France et les Pays-Bas disposent d'un second forecast MKOnline ayant franchi un protocole de validation dédié.

Le système mélange uniquement les prévisions centrales :

**P50 final = poids autonome × P50 autonome + poids MKOnline × P50 MKOnline**

Puis il déplace P10 et P90 de la même quantité que P50. Là encore, la largeur et l'ordre des quantiles sont conservés.

### 8.1 Poids actuellement gelés

| Pays | Poids autonome | Poids MKOnline | Série MKOnline primaire |
|---|---:|---:|---|
| France | 49,7761 % | 50,2239 % | `41551_native` |
| Pays-Bas | 53,2274 % | 46,7726 % | `41554_native` |

Ces poids sont **fixes**. Ils ne sont pas recalculés chaque matin et ne peuvent pas être saisis librement dans l'application.

L'Allemagne, la Belgique et l'Espagne utilisent le modèle autonome corrigé : aucun poids français ou néerlandais n'est réutilisé par défaut.

### 8.2 MKOnline n'est pas Storm

- **MKOnline** peut contribuer à la prédiction en FR et NL.
- **Storm** est un benchmark d'évaluation et n'entre jamais dans la formule du forecast.

Storm peut servir à décider si une recette figée mérite d'être promue, mais ses valeurs ne sont ni des variables du modèle, ni une solution de secours en cas de donnée manquante.

### 8.3 Réserve de gouvernance

Les manifestes MKOnline indiquent que les droits commerciaux et de remplacement restent à confirmer avec les propriétaires de la donnée. C'est une question de gouvernance contractuelle distincte de la validation statistique et technique.

---

## 9. Les variantes par pays

| Pays | Fuseau de livraison | Modèle de production | Storm officiel dans Statistics | Particularité |
|---|---|---|---|---|
| **FR** | Europe/Paris | Chronos-2 + correcteur + blend MKOnline | Oui | MKOnline `41551_native` |
| **DE** | Europe/Berlin | Chronos-2 + correcteur autonome | Oui | Zone de prix DE-LU |
| **BE** | Europe/Brussels | Chronos-2 + correcteur autonome | Oui | Les améliorations expérimentales restent hors production tant qu'elles ne battent pas Storm de façon robuste |
| **NL** | Europe/Amsterdam | Chronos-2 + correcteur + blend MKOnline | Oui | MKOnline `41554_native` ; traitement explicite du fuseau de la charge résiduelle NL |
| **ES** | Europe/Madrid | Chronos-2 + correcteur autonome | Non | Aucune série Storm native équivalente n'est aujourd'hui vérifiée ; le rapport affiche N/A |

Le Royaume-Uni et l'Italie ne sont pas activés. Le Royaume-Uni manque encore d'un contrat complet cible / variables / devise. L'Italie nécessite d'abord de choisir et valider une zone de prix précise : IT_NORD, PUN et les autres zones ne sont pas interchangeables.

---

## 10. Comment le modèle a été entraîné sans tricher

### 10.1 Des prévisions historiques « comme en vrai »

Le correcteur ne doit pas apprendre sur des prévisions qui auraient déjà vu leur propre résultat. Les prévisions historiques utilisées sont donc **out-of-fold** : chaque bloc est prédit par un modèle entraîné uniquement sur une période antérieure admissible.

En langage simple, on rejoue le passé en respectant l'ordre du temps :

1. apprendre sur le passé ;
2. prévoir la période suivante ;
3. conserver cette prévision ;
4. avancer dans le temps ;
5. ne jamais réentraîner rétroactivement la prévision déjà conservée.

### 10.2 Les fenêtres gelées de la recette autonome

La version actuelle repose sur :

- **EXT223** : 223 jours historiques supplémentaires, du 2 janvier au 11 août 2024 ;
- **365 jours de calibration** ;
- **365 jours de final scellé** ;
- soit 730 jours de prévisions existantes, auxquels s'ajoutent EXT223.

Pour l'évaluation annuelle, le correcteur est ajusté sur EXT223 + les 365 premiers jours, puis testé sur les 365 derniers jours restés intacts.

Pour le forecast opérationnel ultérieur, une instance séparée est ajustée sur EXT223 + les 730 jours historiques désormais observés. Cette instance ne réécrit pas le résultat du backtest annuel.

### 10.3 Le final scellé

Le final actuel couvre du **12 août 2025 au 11 août 2026** :

- 365 jours civils ;
- 8 760 heures physiques ;
- une journée de 23 heures ;
- 363 journées de 24 heures ;
- une journée de 25 heures.

La recette, le schéma et les poids sont gelés avant l'ouverture de cette période. Le final ne sert pas à retoucher les hyperparamètres après lecture du résultat.

### 10.4 Le protocole A / B1 / B2 pour une amélioration

Une nouvelle idée ne remplace pas directement le modèle en production. Elle suit un parcours par étapes :

1. **A — construction et validation croisée chronologique** : choix du candidat sans regarder les périodes de veto ;
2. **B1 — premier test indépendant** : le candidat figé doit apporter un gain suffisant et cohérent ;
3. **B2 — second veto indépendant** : aucune nouvelle sélection ni retouche n'est autorisée ;
4. **scellement** : code, données, poids et paramètres reçoivent des checksums ;
5. **final** : une seule évaluation sur la période scellée ;
6. **promotion** : le registre de production change seulement si les critères sont réellement satisfaits.

Cette séparation protège contre le sur-ajustement, c'est-à-dire le fait de choisir par hasard une recette qui correspond très bien à une période déjà observée mais se dégrade ensuite.

---

## 11. Ce qui se passe après un clic dans l'application

### 11.1 Lancement

Le moyen le plus simple est de double-cliquer sur :

`C:\Users\BQ6757\chronos2_v1\launch_forecast_app.cmd`

L'équivalent PowerShell est :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Start-ForecastApp.ps1'
```

L'application Streamlit s'ouvre dans le navigateur local.

### 11.2 Préparation du run

L'utilisateur :

1. vérifie que le pays est marqué prêt ;
2. sélectionne un ou plusieurs pays, ou tous les pays disponibles ;
3. choisit le jour de livraison ;
4. clique sur le bouton de lancement ;
5. suit l'avancement et les logs.

Le jour proposé par défaut est le lendemain. Des options avancées permettent de choisir le device, le nombre de workers et de threads, ainsi que l'utilisation exclusive du cache local du modèle Hugging Face. Les réglages de recette, de schéma et de poids ne sont pas exposés comme des paramètres libres.

Les pays sont traités **séquentiellement** afin d'éviter la concurrence sur le modèle, le GPU et les caches.

### 11.3 Contrôles avant calcul

Pour chaque pays, le moteur vérifie notamment :

- l'identité de la zone, de la cible et du fuseau ;
- la présence de la recette et du benchmark scellés ;
- les checksums des données et du code de refit ;
- la date de livraison et son cutoff ;
- la disponibilité des covariables requises ;
- le mode explicite `autonomous_only` ou `mkonline_blend`.

Il n'existe pas de bascule silencieuse d'un mode vers un autre. Si MKOnline est requis mais absent, un modèle blend ne se transforme pas discrètement en modèle autonome.

### 11.4 Calcul du forecast

Le moteur :

1. synchronise les données admissibles ;
2. construit la journée locale exacte ;
3. exécute Chronos-2 ;
4. réajuste le correcteur avec la recette et l'historique gelés ;
5. applique éventuellement le blend MKOnline ;
6. valide P10, P50, P90, l'origine et la timeline ;
7. écrit le candidat dans un dossier privé de préparation.

### 11.5 Publication prioritaire et reporting

Le forecast validé est figé avant tout chargement de Storm. Ensuite seulement, le moteur met à jour l'historique réalisé, les Statistics et le rapport HTML.

Une panne de reporting postérieure au gel ne doit pas transformer un forecast valide en forecast perdu. Elle est signalée séparément. Les erreurs portant sur le candidat, son checksum ou sa publication restent, elles, bloquantes.

### 11.6 Historique manquant et replays

Si quelques journées récentes manquent dans l'historique des Statistics, le système peut reconstruire automatiquement jusqu'à **trois jours** de forecasts causaux, dans l'ordre chronologique. Ils sont étiquetés comme **replays PIT**, jamais comme forecasts réellement émis ce jour-là.

Si le trou ne peut pas être réparé, les Statistics utilisent uniquement le préfixe historique complet précédant le premier trou. Le statut devient partiel ; aucune journée postérieure n'est ajoutée artificiellement.

### 11.7 Archive déjà publiée

Une archive existante n'est pas écrasée. Si son identité, sa grille et tous ses checksums sont valides, l'application indique qu'elle est déjà publiée et l'ignore proprement. Si elle est incomplète ou altérée, le run est bloqué pour examen.

---

## 12. Les fichiers publiés et leur rôle

Un run contient notamment :

| Fichier | Rôle |
|---|---|
| `forecast_hourly_<pays>.csv` | P10, P50 et P90 de chaque heure, métadonnées et origine |
| `run_manifest.json` | identité du run, zone, date, cutoff, recette et provenance |
| `artifact_checksums.json` | taille et SHA-256 de chaque artefact |
| `statistics_history_hourly.csv.gz` | historique utilisé par la section Statistics |
| `statistics_history_audit.json` | couverture, jours manquants, nature live/replay et contrat Storm |
| rapport `.html` | visualisation mono-pays ou consolidée |

Les archives sont d'abord construites dans un dossier temporaire, contrôlées, puis publiées par renommage. Cette méthode limite le risque de voir un dossier à moitié écrit.

---

## 13. Comment lire P10, P50 et P90

### P50 — la courbe centrale

P50 est le scénario central. C'est cette série qui est comparée au prix réalisé pour calculer les principales métriques.

P50 ne signifie pas que le prix réel sera exactement égal à cette valeur. Il s'agit du milieu estimé de la distribution prévue.

### P10 — le scénario bas

P10 est le scénario bas produit par Chronos-2. Il indique la partie basse de la distribution interne du modèle, sans taux de couverture garanti par les artefacts actuels.

### P90 — le scénario haut

P90 est le scénario haut produit par Chronos-2. Il indique la partie haute de la distribution interne du modèle, sans constituer un plafond garanti.

### La bande P10–P90

La bande donne une indication de l'incertitude vue par Chronos-2. Le système ne publie aujourd'hui aucune étude de calibration démontrant qu'elle contient le prix réel avec une fréquence donnée. Elle n'est ni un minimum, ni un maximum, ni un intervalle de confiance garanti.

- bande large : incertitude annoncée plus forte ;
- bande étroite : incertitude annoncée plus faible ;
- prix réel hors bande : possible, notamment lors d'un événement exceptionnel.

Dans la recette actuelle, le correcteur et le blend déplacent la bande sans modifier sa largeur. La largeur provient donc principalement de Chronos-2.

---

## 14. Storm : le modèle à battre, jamais une entrée de prédiction

Storm est le **benchmark**, c'est-à-dire le forecast de référence. Le prix réellement observé reste la vérité de marché ; Storm n'est pas la vérité.

### 14.1 Séparation stricte

Storm est chargé uniquement après le gel du candidat :

- il ne figure pas dans les variables du modèle ;
- il ne modifie pas P10, P50 ou P90 ;
- il n'est pas utilisé comme fallback ;
- il sert à mesurer la performance relative.

### 14.2 Série officielle du dashboard

Pour FR, DE, BE et NL, le rapport utilise la série native officielle identifiée par pays. Cette série est extraite dans son état courant, normalisée selon le fuseau civil, puis figée avec son instant d'extraction et son SHA-256.

Elle peut être révisée à la source. Le snapshot conservé dans l'artefact rend le rapport reproductible, mais il faut garder à l'esprit qu'il s'agit d'un benchmark d'évaluation « état courant », pas nécessairement de la version exacte que Storm affichait à 08:00 chaque jour historique.

Un comparateur causal distinct « Storm disponible à 08:00 » peut exister pour certains audits. Il ne doit jamais être confondu avec le benchmark officiel du dashboard.

### 14.3 Cas de l'Espagne

Aucune série native officielle Storm équivalente n'est aujourd'hui vérifiée pour ES. Le rapport affiche donc **N/A**. Il ne remplace pas Storm par une basecase ou un proxy sous un nom trompeur.

### 14.4 Couverture appariée

Notre modèle et Storm sont comparés :

- aux mêmes prix réalisés ;
- sur les mêmes heures ;
- uniquement lorsque les deux valeurs sont finies ;
- sur des journées complètes pour les win rates journaliers stricts.

Lors du passage à l'heure d'hiver, la série Storm native peut ne contenir qu'un des deux folds de 02:00. La journée incomplète est alors exclue plutôt qu'interpolée.

---

## 15. Comprendre la section Statistics

Le tableau présente, pour chaque pays et chaque métrique :

- la performance globale de notre modèle ;
- la performance globale de Storm ;
- leur écart ;
- le nombre de périodes gagnées, à égalité ou perdues ;
- le nombre de périodes comparables ;
- le win rate ;
- le volume d'historique ;
- le statut complet, partiel ou indisponible.

Les sept métriques ne sont pas redondantes. Chacune éclaire une dimension différente.

### 15.1 Tableau des sept métriques

| Statistique | Ce qu'elle mesure | Sens souhaité | Unité | Principal piège |
|---|---|---|---|---|
| **MAE** | erreur absolue moyenne | plus faible | EUR/MWh | masque l'ampleur particulière des très gros ratés |
| **RMSE** | erreur avec forte pénalisation des gros ratés | plus faible | EUR/MWh | très sensible à quelques pointes extrêmes |
| **MAPE** | erreur relative au prix réalisé | plus faible | % | explose lorsque le prix est proche de zéro |
| **Explained Variance** | capacité à reproduire les variations | plus élevée | sans unité | peut rester bonne malgré un biais constant |
| **R²** | gain par rapport à une prévision égale à la moyenne | plus élevé | sans unité | peut être négatif ou instable sur une période peu variable |
| **Écart-type de l'erreur** | stabilité / dispersion des erreurs | plus faible | EUR/MWh | ne détecte pas un biais constant à lui seul |
| **Corrélation** | synchronisation des hausses et baisses | plus élevée | de −1 à 1 | peut être excellente avec un niveau de prix faux |

### 15.2 MAE — erreur absolue moyenne

Une MAE de 10 EUR/MWh signifie que la prévision se trouve en moyenne à environ 10 EUR/MWh du prix réel, sans distinguer erreur à la hausse ou à la baisse.

C'est l'indicateur le plus simple pour commencer la lecture.

### 15.3 RMSE — attention aux gros ratés

Le RMSE met au carré les erreurs avant de les moyenner. Il pénalise donc fortement les pointes mal prévues.

Si le RMSE est très supérieur à la MAE, quelques erreurs importantes pèsent probablement sur le résultat.

### 15.4 MAPE — un pourcentage à manier avec prudence

La MAPE divise l'erreur absolue par la valeur absolue du prix réalisé. Le code exclut uniquement les valeurs quasi nulles, avec **|prix| ≤ 10⁻⁹ EUR/MWh**.

Exemple : une erreur de 5 EUR/MWh représente 5 % si le prix vaut 100 EUR/MWh, mais 500 % s'il vaut 1 EUR/MWh.

Sur les marchés électriques, où des prix proches de zéro ou négatifs sont possibles, la MAPE ne doit jamais être utilisée seule.

### 15.5 Explained Variance

Elle mesure la capacité à reproduire les variations : pointes, creux et volatilité.

- 1 : variations parfaitement reproduites ;
- 0 : peu de variation expliquée ;
- négatif : mauvaise reproduction de la dynamique.

Un forecast toujours décalé de +10 EUR/MWh peut conserver une excellente variance expliquée. Il faut donc la lire avec la MAE ou le R².

### 15.6 R²

Le R² compare le forecast à une référence très simple qui utiliserait le prix moyen de la période.

- 1 : parfait ;
- 0 : équivalent à la référence moyenne ;
- négatif : moins bon que cette référence.

Une valeur négative n'est pas une panne informatique.

### 15.7 Écart-type de l'erreur

Il mesure la dispersion des erreurs autour de leur moyenne. Une valeur faible indique des erreurs régulières ; une valeur élevée signale un comportement instable.

Un modèle toujours trop haut de 10 EUR/MWh peut avoir un écart-type nul tout en restant mauvais en MAE.

### 15.8 Corrélation

La corrélation indique si la courbe monte et descend au même moment que le prix réel.

Une corrélation élevée ne garantit pas le bon niveau de prix. Un forecast deux fois trop élevé peut encore être parfaitement corrélé.

---

## 16. Le win rate face à Storm

Le win rate transforme chaque période en un match.

- **W — Win / gagné** : notre modèle est meilleur ;
- **T — Tie / égalité** : les valeurs sont égales à la tolérance numérique près ;
- **L — Loss / perdu** : Storm est meilleur.

Pour MAE, RMSE, MAPE et écart-type, la valeur la plus faible gagne. Pour Explained Variance, R² et corrélation, la valeur la plus élevée gagne.

**Win rate = W ÷ (W + T + L)**

Les égalités restent dans le dénominateur et ne valent pas une demi-victoire.

### Exemple

Sur 100 journées complètes :

- 56 gagnées ;
- 1 égalité ;
- 43 perdues.

Le win rate est de **56 %**.

### Ce que le win rate ne dit pas

Il mesure la fréquence des victoires, pas leur ampleur. Un modèle peut gagner souvent de peu et perdre rarement de beaucoup. Il peut alors afficher plus de 50 % de win rate tout en ayant une MAE globale légèrement supérieure à Storm.

Il faut donc toujours lire ensemble :

1. le win rate ;
2. W / T / L ;
3. le nombre de périodes ;
4. la MAE ou la métrique globale ;
5. la stabilité au fil du temps.

---

## 17. Daily, Weekly et Monthly

### Daily

Chaque journée locale est un match, avec 23, 24 ou 25 heures selon le calendrier. C'est la vue la plus opérationnelle et l'objectif principal actuel pour la MAE.

### Weekly

Les heures d'une semaine locale, du lundi au dimanche, sont regroupées. La vue est plus stable mais moins sensible à une journée particulière.

### Monthly

Les heures d'un mois civil sont regroupées. Cette vue convient au pilotage de long terme mais peut masquer des épisodes courts.

Une même recette peut gagner 18 jours sur 30 et perdre le mois si ses 12 défaites sont beaucoup plus fortes. Les trois vues sont donc complémentaires.

Les semaines ou mois situés au bord de l'historique peuvent être partiels. Le rapport expose le nombre d'heures et le statut afin d'éviter une fausse équivalence.

---

## 18. Lire le rapport HTML consolidé interactif

Le rapport consolidé est autonome : les données et Plotly sont inclus dans le fichier. Il peut être ouvert dans Edge ou Chrome sans serveur et sans CDN.

JavaScript doit être autorisé dans le navigateur. Le fichier est un **snapshot** : il ne s'actualise pas tout seul lorsque de nouveaux runs apparaissent ; il faut télécharger une nouvelle version depuis l'application.

La comparaison multi-pays charge les dernières archives portant le statut `issued_live`. Elle refuse par défaut d'assembler des jours de livraison différents. L'autorisation multi-dates existe pour un diagnostic explicite, mais ne doit pas être utilisée pour présenter une comparaison homogène.

### 18.1 Forecast consolidé

L'utilisateur peut :

- choisir un ou plusieurs pays ;
- filtrer les dates ;
- afficher ou masquer P10–P90 ;
- normaliser les courbes depuis H00 pour comparer leurs formes ;
- survoler une heure pour lire la valeur exacte ;
- zoomer, déplacer le graphique et masquer une série via la légende ;
- exporter le graphe en PNG.

Storm n'apparaît pas dans ce graphique de prédiction. Il reste confiné à la partie Statistics.

### 18.2 Historique et erreurs

Pour chaque pays disposant d'un historique, le rapport permet d'afficher :

- prix réalisé, notre P50 et Storm ;
- erreurs absolues de notre modèle et de Storm ;
- erreur moyenne selon l'heure ;
- plages récentes ou plage complète.

### 18.3 Statistics complètes

Le rapport contient :

- les 7 métriques pour tous les pays ;
- les vues Daily, Weekly et Monthly ;
- une heatmap des win rates ;
- une courbe d'évolution par période ;
- un tableau exhaustif de 35 lignes pour cinq pays et sept métriques ;
- le détail W / T / L ;
- l'export CSV des Statistics.

Pour ES, les valeurs de notre modèle restent visibles et le benchmark Storm est N/A.

### 18.4 Audit et provenance

La section d'audit indique les archives sources, dates, identifiants et empreintes. Elle permet de vérifier que le rapport n'a pas assemblé des pays ou des jours incompatibles.

---

## 19. État de performance au 14 août 2026

Le tableau ci-dessous est un **instantané des artefacts utilisés pour la livraison du 15 août 2026**, pas une promesse de performance future. L'historique réunit le benchmark scellé et les archives réalisées admissibles. Les comparaisons portent sur les périodes communes à notre modèle et à la série Storm officielle ; les heures Storm absentes ne sont pas interpolées.

| Pays | Modèle évalué | Win rate journalier MAE vs Storm | W / T / L | MAE modèle | MAE Storm | Lecture |
|---|---|---:|---:|---:|---:|---|
| FR | blend MKOnline | **54,62 %** | 201 / 0 / 167 | 11,250 | 11,653 | majorité de jours gagnés et MAE globale meilleure |
| DE | autonome corrigé | **54,08 %** | 199 / 0 / 169 | 11,348 | 11,147 | gagne plus souvent, mais perd plus fortement certains jours ; MAE globale moins bonne |
| BE | autonome corrigé | **40,49 %** | 149 / 0 / 219 | 11,423 | 10,357 | objectif > 50 % non atteint ; priorité d'amélioration |
| NL | blend MKOnline promu | **52,73 %** | 193 / 0 / 173 | 10,677¹ | 10,621¹ | objectif de fréquence franchi ; Statistics encore partielles |
| ES | autonome corrigé | N/A | N/A | 9,762 | N/A | aucune série Storm native officielle vérifiée ; aucun proxy n'est affiché |

¹ L'historique NL est actuellement marqué `partial_contiguous_prefix` : il s'arrête au 12 août 2026, car les archives des 13 et 14 août manquent dans ce nouveau périmètre blend. Il contient 8 784 heures et 366 périodes journalières comparables. FR, DE, BE et ES vont jusqu'au 14 août.

> **Conclusion métier** — Dépasser 50 % de win rate ne suffit pas à lui seul. DE et NL illustrent pourquoi la fréquence des victoires doit être lue avec la MAE globale et l'ampleur des défaites.

### Win rates quotidiens des sept métriques sur ce snapshot

| Pays | MAE | RMSE | MAPE | Explained Variance | R² | Écart-type erreur | Corrélation |
|---|---:|---:|---:|---:|---:|---:|---:|
| FR | 54,62 % | 54,89 % | 48,64 % | 61,14 % | 54,89 % | 61,14 % | 65,22 % |
| DE | 54,08 % | 52,45 % | 50,27 % | 50,27 % | 52,45 % | 50,27 % | 50,00 % |
| BE | 40,49 % | 39,95 % | 37,50 % | 47,01 % | 39,95 % | 47,01 % | 49,46 % |
| NL¹ | 52,73 % | 50,00 % | 44,81 % | 54,64 % | 50,00 % | 54,64 % | 56,56 % |
| ES | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Ces pourcentages ne comportent pas d'intervalle de confiance dans le rapport courant. Un écart proche de 50 % doit donc être considéré comme fragile tant qu'il n'est pas confirmé sur davantage de prévisions réellement émises.

Les chiffres du rapport interactif sélectionné restent la source opérationnelle : ils peuvent inclure des jours live ou des replays postérieurs au final scellé.

---

## 20. Améliorer le modèle : ce qui est en production et ce qui reste expérimental

### 20.1 Principe de promotion

Une source externe n'est pas retenue parce qu'elle paraît intuitivement intéressante. Elle doit prouver :

1. une heure de disponibilité antérieure au cutoff ;
2. un historique de versions ou des snapshots permettant un backtest honnête ;
3. une couverture 23/24/25 heures sans remplissage inventé ;
4. un gain sur B1 ;
5. la confirmation indépendante sur B2 ;
6. une amélioration robuste face à l'autonome et, lorsque disponible, face à Storm ;
7. des droits d'utilisation compatibles avec la production.

### 20.2 Sources explorées mais non promues

À la date de ce document :

- la météo Open-Meteo en previous-run D−2 n'a pas amélioré le gate FR ;
- les données Eco2Mix disponibles n'offrent pas encore un historique PIT suffisamment propre pour une promotion ;
- un prototype de vigilance Météo-France révision-aware existe, mais n'est pas intégré au modèle de production ;
- les expériences BE de correction non linéaire et de signaux cross-zone améliorent certaines références mais n'atteignent pas le critère robuste > 50 % face à Storm ;
- les indisponibilités de centrales, alertes REMIT, données ENTSO-E et signaux de texte restent prometteurs mais exigent un historique versionné vérifiable.

### 20.3 Images et texte

Le modèle de production n'analyse aujourd'hui ni images, ni PDF, ni articles.

Pour la vigilance météo, le JSON structuré est préférable à une image ou à un PDF représentant la même information : il est horodaté, plus fiable et évite les erreurs d'OCR.

Des images radar ou satellite ne seraient envisagées que si elles apportaient un signal supplémentaire démontré après les variables météo numériques. Les articles et annonces exigeraient une preuve de leur heure de première disponibilité, une déduplication et une validation multilingue.

### 20.4 Priorités actuelles

- consolider le gain FR et NL hors échantillon live ;
- réduire les grosses pertes DE lors des épisodes extrêmes ;
- améliorer en priorité BE, qui reste sous 50 % ;
- identifier une série Storm native officielle pour ES avant de prétendre mesurer son win rate ;
- conserver chaque nouvelle piste hors production jusqu'au passage complet des gates.

---

## 21. Garde-fous et raisons de confiance

Le système adopte une logique **fail-closed** : en cas d'ambiguïté, il refuse ou signale la partie concernée au lieu de fabriquer une valeur.

### Contrôles principaux

- timeline UTC unique, triée et horaire ;
- journée locale exacte de 23, 24 ou 25 heures ;
- absence de valeur non finie ;
- P10 ≤ P50 ≤ P90 ;
- origine de forecast antérieure à la livraison ;
- snapshots et révisions antérieurs au cutoff ;
- identité stricte pays / cible / modèle / mode ;
- absence de Storm dans les entrées de prédiction ;
- schéma de 175 variables exact ;
- vérification des fichiers gelés et du code algorithmique par SHA-256 ;
- publication par staging puis renommage ;
- archives existantes non écrasées silencieusement ;
- benchmark manquant affiché N/A, jamais remplacé par un faux équivalent.

### Ce que ces contrôles ne garantissent pas

Ils garantissent la cohérence, la causalité et la traçabilité de la chaîne. Ils ne garantissent pas :

- que le marché ne connaîtra jamais un événement inédit ;
- que P50 sera toujours proche du prix réel ;
- que le modèle battra Storm chaque jour ;
- que la licence commerciale d'une source externe est automatiquement acquise ;
- qu'une forte performance passée se répétera à l'identique.

---

## 22. Bonnes pratiques de lecture

1. Commencer par la **courbe P50** et la bande P10–P90.
2. Repérer les heures de pointe, les rampes et les zones d'incertitude.
3. Lire la **MAE**, puis le **RMSE** pour détecter les gros ratés.
4. Vérifier le **win rate MAE**, W/T/L et le nombre de périodes.
5. Examiner Daily, Weekly et Monthly.
6. Vérifier le statut complet / partiel et la couverture.
7. Utiliser MAPE, corrélation, R² et Explained Variance comme compléments, jamais isolément.
8. Pour une décision à risque, considérer P10/P90 et les scénarios métier, pas seulement P50.
9. Ne pas confondre backtest, replay PIT et forecast réellement émis.
10. Consulter la provenance lorsqu'un résultat paraît surprenant.

---

## 23. Dépannage courant

### L'application mentionne le proxy `127.0.0.1:9`

Ce proxy correspond à un environnement réseau isolé et empêche l'accès à Saturn.

1. fermer complètement les processus Streamlit / Python précédents ;
2. quitter entièrement Codex si le terminal en dépend ;
3. double-cliquer sur `launch_forecast_app.cmd` depuis l'Explorateur Windows ;
4. si le message persiste, faire vérifier les variables Windows `HTTP_PROXY`, `HTTPS_PROXY` et `ALL_PROXY` par le support, sans supprimer un éventuel proxy d'entreprise légitime.

### Une archive est indiquée « déjà publiée »

C'est normal si le forecast du pays et du jour existe déjà avec des checksums valides. Le système l'ignore pour préserver son immutabilité.

### Statistics est « partiel »

Un jour antérieur manque ou n'est pas complet. Le forecast courant peut rester valide. Le rapport s'arrête avant le premier trou et n'invente aucune performance.

### Storm vaut N/A pour ES

C'est le comportement attendu tant qu'aucune série native officielle équivalente n'a été vérifiée.

### Une journée comporte 23 ou 25 lignes

C'est normal lors du changement d'heure. Une journée artificiellement forcée à 24 heures serait au contraire suspecte.

### Le rapport consolidé semble volumineux

Il embarque Plotly, l'historique et les données de cinq pays afin de fonctionner hors ligne. Sa taille est donc supérieure à celle d'une simple page web.

---

## 24. Glossaire

| Terme | Définition simple |
|---|---|
| **Actual / prix réalisé** | prix de marché réellement observé après livraison |
| **Backtest** | simulation historique d'une prévision passée |
| **Benchmark** | modèle de référence utilisé pour comparer les performances |
| **Biais** | tendance à prévoir systématiquement trop haut ou trop bas |
| **Charge résiduelle** | demande restant à couvrir après prise en compte d'une partie des renouvelables |
| **Checksum / SHA-256** | empreinte numérique permettant de détecter une modification de fichier |
| **Chronos-2** | modèle de séries temporelles qui produit la première prévision probabiliste |
| **Cutoff** | dernière heure à laquelle une information est autorisée pour le forecast |
| **Day-ahead** | marché fixant aujourd'hui les prix horaires du lendemain |
| **DST** | changement d'heure donnant des journées de 23 ou 25 heures |
| **Fail-closed** | refuser une situation douteuse au lieu de deviner ou compléter silencieusement |
| **Final scellé** | période d'évaluation ouverte uniquement après gel de la recette |
| **Forecast origin** | instant officiel auquel la prévision est réputée produite |
| **MKOnline** | forecast externe utilisé dans le blend FR/NL après validation |
| **OOF / out-of-fold** | prévision historique produite sans utiliser son propre résultat |
| **P10 / P50 / P90** | quantiles bas, central et haut du forecast |
| **PIT / point-in-time** | sélection de la version réellement disponible à l'époque |
| **Replay PIT** | reconstruction causale a posteriori, clairement séparée d'un forecast réellement émis |
| **Résidu** | différence entre le prix réalisé et la prévision de base |
| **Storm** | forecast de référence utilisé seulement pour l'évaluation |
| **Win rate** | part des périodes comparables gagnées face à Storm |

---

## 25. Questions fréquentes

### Le modèle « sait-il » pourquoi le prix monte ?

Il apprend des relations statistiques entre historique, calendrier, charge résiduelle et courbe Chronos-2. Il ne produit pas automatiquement une explication causale complète d'un événement de marché.

### Peut-on lancer tous les pays d'un coup ?

Oui. L'application les place dans une file et les traite l'un après l'autre.

### Pourquoi ne pas donner directement Storm au modèle ?

Parce que Storm est le benchmark à battre. L'utiliser comme entrée brouillerait la comparaison et créerait une dépendance opérationnelle au concurrent.

### Pourquoi garder plusieurs métriques ?

Parce qu'un modèle peut être bon en moyenne mais mauvais sur les pointes, suivre la forme sans le niveau, ou gagner souvent tout en perdant très fortement certains jours.

### Plus de 50 % de win rate signifie-t-il que notre MAE globale est meilleure ?

Non. Le win rate compte les périodes gagnées ; la MAE globale tient compte de l'ampleur de toutes les erreurs. Il faut lire les deux.

### Pourquoi ne pas combler une heure Storm manquante ?

Une interpolation fabriquerait une valeur de benchmark et pourrait changer artificiellement le gagnant. Le système préfère exclure la période incomplète et l'indiquer.

### Le modèle se réentraîne-t-il tous les jours ?

Le correcteur opérationnel est refitté à partir de la recette, du code et de l'historique gelés. Les hyperparamètres, le schéma et les poids de blend ne sont pas retunés chaque matin.

### L'application exécute-t-elle des ordres ?

Non. Elle produit et évalue des forecasts. Toute décision ou action de marché reste sous contrôle humain et dans les processus autorisés.

---

## 26. Références internes pour audit

Les contrats de production sont notamment définis dans :

- `chronos2_hourly_live_zones.yaml` — registre des pays activés ;
- `run_extended_residual_hourly.py` — recette autonome étendue ;
- `chronos2_hourly/models/residual_corrector.py` — correcteur résiduel ;
- `chronos2_hourly/models/blended_residual_corrector.py` — combinaison CatBoost / HGB ;
- `mkonline_fr_blend_recipe_v1.json` — poids FR gelés ;
- `mkonline_nl_blend_recipe_v1.json` — poids NL gelés ;
- `chronos2_hourly/multizone_live.py` — moteur live générique ;
- `chronos2_hourly/app_service.py` — lecture des archives et calcul des Statistics ;
- `chronos2_hourly/consolidated_report.py` — rapport HTML interactif ;
- `app_multizone.py` — interface utilisateur.

Ces références sont destinées à l'audit. L'utilisateur quotidien n'a pas besoin de les modifier.

---

## Conclusion

Chronos-2 multi-pays est une chaîne complète, pas un unique algorithme :

- données causales gelées à 08:00 ;
- première prévision probabiliste Chronos-2 ;
- correction des erreurs récurrentes par deux modèles complémentaires ;
- second avis MKOnline uniquement en FR et NL ;
- contrôles stricts de timeline, de quantiles, de fraîcheur et d'identité ;
- archive immuable et traçable ;
- comparaison Storm séparée après gel ;
- lecture multi-métriques plutôt qu'un score isolé.

Son objectif n'est pas de prétendre connaître le prix avec certitude. Il vise à fournir une prévision day-ahead reproductible, auditée et progressivement meilleure, tout en montrant honnêtement son incertitude, ses échecs et le périmètre exact de chaque comparaison.
