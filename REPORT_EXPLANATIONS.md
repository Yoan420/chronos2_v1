# Explication du forecast et comparaison horaire

Les nouveaux rapports HTML des runs `Both` affichent :

- **Influence des variables et des prix passés** : contributions Shapley locales
  en EUR/MWh, parts relatives (%) et carte des contributions à chaque heure.
- **Du modèle de base au prix final** : prix Chronos-2, correction résiduelle,
  éventuel ajustement Kalman, puis prix final. Il s'agit des montants réellement
  émis, pas de poids de variables.
- **Performance par heure locale** : sélecteur MAE, prix moyens
  modèle/Storm/observé et biais. Le survol indique les effectifs et le gain de MAE.

## Ce que signifie une influence

Les prix historiques sont l'entrée cible de Chronos-2, distincte des covariables.
Leur groupe contient le contexte cible et ses dérivés lag/rolling lorsqu'ils
sont utilisés par le correcteur. Aucun prix observé du jour prédit n'est injecté.

Chaque groupe est comparé à une référence construite à partir de l'historique
fourni : médiane par heure de semaine sur les 56 derniers jours. Les paramètres
appris restent fixes. La somme des contributions signées et de la prévision
de référence reconstruit la P50 publiée (tolérance par défaut : 0,001 EUR/MWh).

La part relative est la somme des contributions absolues d'un groupe, divisée
par celle de tous les groupes, sur les heures physiques du jour. Elle ne mesure
pas une part du prix et dépend du jour, de la référence et des interactions.
Les groupes totalisent 100 %, ou 0 % si toutes les contributions sont nulles.
Ce n'est ni un coefficient constant du réseau ni un effet économique causal.
Le calendrier reste fixe et n'a pas de poids individuel dans cette décomposition.

Avec Kalman, les contributions expliquent **l'autonome en amont du filtre**.
L'ajustement Kalman est présenté séparément. Ses états internes, son historique
de réentraînement et son gouverneur ne sont pas attribués par variable.

## Comparaison avec Storm

Le profil horaire utilise la même fenêtre glissante de Statistics : les
365 derniers jours observés, plus un emplacement pour une livraison non observée.
Seuls les triplets finis (modèle, Storm, observation) participent aux moyennes.
La MAE est la moyenne des erreurs absolues, pas l'écart entre deux prix moyens.
Chaque heure physique compte une fois : les deux heures répétées d'automne
restent deux observations ; aucune heure fictive n'est créée au printemps.
Les données manquantes restent absentes, jamais remplacées par zéro.

Le rapport nucléaire peut charger le snapshot officiel audité du run FR de même
livraison pour ce graphique. Checksum, provenance, couverture et audit DST sont
vérifiés. Ses Statistics natives restent inchangées, sur leur support complet ;
les effectifs du graphique comparatif peuvent donc être légèrement inférieurs.
Ce chargement est local et n'effectue aucun appel Saturn/RTE.

Les labels du replay peuvent être stockés en float32, contrairement aux prix
canoniques lus pour le rapport. Seule une différence de représentation vérifiée
à cette précision est acceptée ; un changement réel de prix reste refusé.
Les métriques du rapport utilisent les décimales canoniques et le correcteur
conserve les représentations exactes du cache de calibration original.

## Calculs et réutilisation

Le forecast est gelé avant l'explication. Jusqu'à six groupes, le Shapley est
exact : cinq fondamentaux et les prix passés donnent 64 scénarios. Au-delà,
des permutations reproductibles sont utilisées (32 par défaut, budget maximal
de 256 scénarios). Le nucléaire comporte actuellement sept groupes.

Ces inférences supplémentaires portent uniquement sur la livraison affichée.
Le nucléaire reconstitue une fois le correcteur du dernier jour, pas les
730 jours de replay. Le calcul initial de l'explication peut donc prendre
plusieurs minutes ; la comparaison horaire ne lance aucun modèle.

Les explications d'archives enrichies sont mises en cache dans
`runs/cache/report_attribution/`. Le nucléaire a un cache séparé dans son
dossier d'expérience. Les archives live et les caches du backtest ne sont
ni réécrits ni rescellés. Une explication absente ou limitée aux seules variables
physiques est explicitement signalée ; aucun poids manquant n'est inventé.

Les scripts chargés par un processus déjà démarré ne changent pas en cours
d'exécution. Une fois ce run terminé, relancer la même commande applique les
nouveaux rapports en réutilisant les checkpoints et caches de prédiction valides.
