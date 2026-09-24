# Résultats exécutés — Solar Ramp, 19 septembre 2026

**Conclusion : association partielle, amélioration du P50 non démontrée ; NYX
conservé.** Le solaire régional apporte un petit gain descriptif au classement
des heures à risque, mais ni une correction centrale meilleure, ni une
calibration opérationnelle suffisamment convaincante. Aucune production modifiée.

## Prix — mêmes 35 040 observations, 365 jours × quatre pays

Période : 18/09/2025–17/09/2026. Les 24 heures de livraison du 18/09 par pays
sont historiques et séparées. Spikes = prix horaire ≥ 300 EUR/MWh.

| Configuration | MAE EUR/MWh | RMSE EUR/MWh | MAE sur les 123 heures de spike |
|---|---:|---:|---:|
| NYX nucléaire + Kalman figé | 11,398 | 21,916 | 166,374 |
| Expert complet non gouverné | 11,449 | 22,071 | 168,023 |
| Combinaison gouvernée | 11,398 | 21,916 | 166,374 |

Le gouverneur n'a autorisé **aucune correction**. L'identité des scores est
donc attendue, pas une erreur du rapport. Les ablations non gouvernées restent
consultables dans le rapport principal.

Delta MAE expert complet moins NYX : +0,0507 EUR/MWh, IC bootstrap apparié 95 %
[-0,0430 ; +0,1782]. Sur spikes : +1,6484, IC [0,6972 ; 2,5015]. Ces IC sont
conditionnels à l'historique et à la recette, sans correction de recherche de
modèles. Ils ne fondent pas une garantie future.

Par pays, la MAE de l'expert complet contre NYX est :

| Pays | NYX | Expert complet |
|---|---:|---:|
| BE | 11,169 | 11,239 |
| DE | 11,090 | 11,135 |
| FR | 12,112 | 12,086 |
| NL | 11,223 | 11,336 |

La petite amélioration française ne suffit pas : seulement six heures ≥300,
RMSE et erreur sur spikes dégradées. Sélectionner maintenant une règle
« seulement FR » serait une nouvelle hypothèse, pas une validation indépendante.

## Signal de risque — support OOS commun

22 748 heures, dont 99 spikes, 47 épisodes. Comparaison entre le contrôle
(calendrier, fondamentaux non solaires explicites, baseline NYX) et la variante
complète avec solaire et interactions :

| Mesure | Contrôle | Complet |
|---|---:|---:|
| PR-AUC / average precision | 0,1799 | 0,2001 |
| Rappel horaire | 87,88 % | 86,87 % |
| Précision des alertes | 14,31 % | 17,23 % |
| Fausses alertes / 1 000 heures | 22,90 | 18,16 |
| Rappel des épisodes exact | 87,23 % | 89,36 % |
| Brier | 0,00576 | 0,00508 |

Le classement s'améliore descriptivement, mais la majorité des alertes restent
fausses. Le budget de 1 % a été réglé sur les négatifs de calibration, **pas
respecté automatiquement hors échantillon** : environ 1,82 % des non-spikes
sont alertés par la variante complète. Les taux effectifs des variantes sont
différents ; on ne présente pas leur rappel comme une comparaison à taux effectif
de fausses alertes identique. Certaines fenêtres n'ont pas assez d'événements
pour calibrer Platt ; leur probabilité brute est explicitement signalée :
17 468 heures (76,79 %) brutes contre 5 280 (23,21 %) calibrées Platt.
Ce n'est donc pas encore un module d'alerte validé pour la production.

La phase finale diagnostique de 90 jours ne confirme pas le P50 : MAE 14,207
contre 14,063 pour NYX. La meilleure variante descriptive sur les 60 jours de
sélection était `local`, mais son IC de gain MAE incluait déjà zéro.

## Épisode illustratif et contre-exemples

Le 14/09/2026 à **19 h Europe/Berlin = 17 h UTC**, intervalle 19–20 h :

- DE observé 697,3075, NYX figé 313,5523 EUR/MWh ; solaire prévu local
  0,5278 GW, baisse 1 h de 5,0133 GW, RL 54,1605 GW.
- Le détecteur complet émet une alerte DE (probabilité sauvegardée 9,95 %),
  mais ne démontre pas une amplitude adéquate. P50 gouverné inchangé.
- BE observé 441,735, NYX 290,4100 EUR/MWh ; alerte également, probabilité
  13,37 %, sans correction gouvernée.

Ces probabilités relativement faibles peuvent franchir un seuil de calibration
bas ; une alerte n'est ni une probabilité >50 %, ni un motif suffisant de
relever la médiane.

Après warm-up, l'analyse trouve **2 322 heures de forte baisse solaire sans
spike** et **5 spikes sans baisse solaire locale**, sur les lignes interprétables.
Les seuils de « forte baisse » sont par pays et figés sur le warm-up. Il ne
s'agit pas de 2 322 journées ou épisodes indépendants.

Dans les strates appariées heure/saison/RL/vent/pression, l'association moyenne
entre forte baisse et spike est positive en BE/DE/NL, mais faible et négative
en FR. Ces effectifs sont limités ; les facteurs confondants, notamment le
réseau et la flexibilité, restent incomplets. Aucune causalité établie.

## Limites et suite justifiée

Archives as-of sans publication fournisseur originale certifiée ; labels
historiques avec délai supposé ; baseline elle-même issue d'un replay publié.
Pas de réalisé solaire utilisé. Premier fit après au moins 120 jours ; fenêtre
maximale de 365, pas 365 jours d'entraînement disponibles dès le début. Suffixe
de fondamentaux manquant après le 15/09 : repli NYX, sans supprimer les heures.
Cette année déjà vue et le cas ayant motivé l'hypothèse ne forment pas un test vierge.

Les Statistics standard face à Storm emploient **8 735 paires horaires par pays**
sur cette archive, contre 8 760 heures par pays pour la comparaison NYX/expert.
Les valeurs Storm absentes ou exclues par le masque figé ne sont pas remplies.
Les MAE entre ces deux supports peuvent donc différer sans incohérence de calcul.

Priorité suivante : captures datées avant enchère des trajectoires renouvelables,
qualification des disponibilités/imports à 08 h, puis validation prospective
figée. Une nouvelle calibration du risque et une meilleure mesure de flexibilité
pourraient être testées ensuite ; elles ne doivent pas être optimisées sur le
seul épisode du 14/09. En attendant, garder le signal dans le laboratoire et
**ne pas changer le P50 de NYX**.

Le rapport principal, quatre rapports au format opérationnel, prévisions,
métriques et manifestes sont effectivement générés. Aucune nouvelle prévision
prospective n'est émise : qualification des entrées et vérification fraîche de
non-publication de la cible non disponibles ensemble.

## Reproduction effectivement contrôlée

Expérience finale :
`runs/experiments/nyx_solar_ramp_v1/snapshots/20260919T132418Z_a249e440`.
Le second lancement de la même recette a produit les mêmes SHA256 pour le
panel, les prévisions, les modèles, les folds, les décisions du gouverneur et
les métriques. Seuls le durcissement de la validation de configuration et
l'allègement sans perte de l'HTML ont changé entre ces deux exécutions ; aucun
paramètre de modèle n'a été ajusté après lecture du premier résultat.

Validation : **117 tests passent**, un test Windows de liens symboliques est
ignoré faute de permission système. Les décodages interactifs et la syntaxe
JavaScript sont testés ; pas de contrôle visuel navigateur revendiqué.
Les empreintes des fichiers opérationnels protégés restent inchangées.

Le launcher et toutes les explications de reproduction sont dans
[la note méthodologique](nyx_solar_ramp.md).
