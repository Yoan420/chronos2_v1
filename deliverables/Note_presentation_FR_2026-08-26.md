# Présentation courte — Forecast France du 26 août 2026

## Texte prêt à dire

« Le modèle prévoit le prix day-ahead français heure par heure. Le pipeline est simple : les données disponibles à l’heure du forecast alimentent Chronos-2, qui produit une première courbe probabiliste ; un correcteur ajuste ensuite ses erreurs récurrentes ; enfin, la prévision est contrôlée, figée et évaluée après coup. Le rapport présente une prévision centrale P50, encadrée par P10 et P90 pour matérialiser l’incertitude.

Pour le 26 août, le prix central prévu est de 144 EUR/MWh en moyenne. Il atteint son minimum vers 13 h, à 75,9 EUR/MWh, puis remonte jusqu’à un maximum de 191,5 EUR/MWh vers 20 h. L’incertitude est particulièrement forte autour de midi.

Historiquement, le correcteur réduit la MAE de Chronos-2 d’environ 5 %, à 12,44 EUR/MWh, avec une corrélation de 0,93. En revanche, Storm reste légèrement meilleur sur la fenêtre complète : MAE de 11,67 EUR/MWh et win rate du modèle de 47,56 %. Storm sert uniquement de benchmark après coup et n’entre jamais dans la prévision. »

## Pipeline dans les grandes lignes

**Données disponibles à D−1 → Chronos-2 → correcteur → contrôles et publication → évaluation après livraison**

1. **Entrées :** historique des prix, calendrier et prévisions de charge résiduelle de la France, de l’Allemagne, de la Belgique, des Pays-Bas et de l’Espagne. Seules les informations disponibles au moment du forecast sont utilisées.
2. **Chronos-2 :** produit la forme initiale de la journée et trois scénarios : P10, P50 et P90.
3. **Correcteur :** estime l’erreur probable de Chronos-2 selon le contexte et déplace la courbe pour réduire les biais récurrents.
4. **Contrôles :** vérification des 24 heures, de l’ordre P10 ≤ P50 ≤ P90 et de l’absence de données futures, puis gel du forecast.
5. **Évaluation :** une fois les prix réels connus, calcul des métriques et comparaison avec Storm. Storm n’est jamais une entrée du modèle.

## Comment fonctionnent les deux étages ?

### 1. Chronos-2 : construire la prévision de base

Chronos-2 est un modèle pré-entraîné spécialisé dans les séries temporelles. Il a appris, sur un grand nombre de séries, des comportements généraux comme les tendances, les cycles, les pics, les creux et les changements de régime. Il n’est donc pas entraîné depuis zéro uniquement sur le marché français.

Pour chaque forecast, il reçoit notamment :

- jusqu’à **2 048 heures d’historique récent du prix français**, soit environ douze semaines ;
- le calendrier : heure, jour de la semaine, week-end et jours fériés ;
- les prévisions de charge résiduelle des cinq pays, déjà connues pour le lendemain.

À partir de ce contexte, Chronos-2 produit une distribution de prix pour chaque heure :

- **P50** est la prévision centrale ;
- **P10** représente un scénario bas ;
- **P90** représente un scénario haut.

Chronos-2 construit ainsi la forme générale de la journée et quantifie l’incertitude. Il peut toutefois conserver des biais propres au marché français ou à certaines situations rares. C’est le rôle du deuxième étage.

### 2. Le correcteur résiduel : apprendre les erreurs récurrentes

Le correcteur ne cherche pas à prévoir directement le prix. Il apprend à prévoir l’erreur de Chronos-2, appelée **résidu** :

**résidu = prix réellement observé − P50 de Chronos-2**

- Un résidu positif signifie que Chronos-2 avait prévu trop bas.
- Un résidu négatif signifie que Chronos-2 avait prévu trop haut.

Pour comprendre dans quelles situations ces erreurs apparaissent, le correcteur utilise notamment :

- l’heure, le jour et les effets calendaires ;
- le niveau et la forme de la courbe Chronos-2 ;
- la largeur et l’asymétrie de sa bande d’incertitude ;
- les niveaux et les rampes des charges résiduelles ;
- les écarts entre la situation française et celle des pays voisins.

Deux modèles d’arbres complémentaires — **CatBoost** et **HistGradientBoosting** — estiment chacun la correction attendue. Leurs résultats sont moyennés à 50/50 afin de réduire la dépendance à un seul algorithme.

L’apprentissage est causal : pour chaque période historique, le correcteur travaille avec une prévision Chronos-2 qui n’a pas utilisé le prix qu’elle devait prévoir. Cette construction « out-of-fold » évite de lui montrer indirectement le futur et limite une performance artificiellement optimiste.

La correction finale est limitée à **±40 EUR/MWh** pour éviter un ajustement excessif. Elle est ajoutée de manière identique à P10, P50 et P90 :

**P50 corrigé = P50 Chronos-2 + correction estimée**

Par exemple, si Chronos-2 prévoit 150 EUR/MWh et que le correcteur anticipe une sous-estimation de 8 EUR/MWh, le P50 corrigé devient 158 EUR/MWh.

Appliquer le même déplacement aux trois quantiles conserve automatiquement l’ordre P10 ≤ P50 ≤ P90. En revanche, le correcteur **ne modifie pas la largeur de la bande d’incertitude** : il corrige principalement le niveau de la courbe, pas son degré d’incertitude.

## Message à retenir

Le modèle améliore Chronos-2 brut et reproduit bien la forme des prix, mais il reste encore un peu moins précis que Storm sur l’historique complet. La prévision du 26 août doit donc être lue comme une courbe centrale accompagnée d’une plage d’incertitude, et non comme une valeur certaine.
