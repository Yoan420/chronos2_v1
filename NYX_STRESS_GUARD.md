# StressGuard : physique, intervalles et validation prospective

Laboratoire isolé. Aucun changement à Forecast.ps1, au mode Complete, aux
modèles actifs ou aux HTML déjà publiés dans runs/exports.

## Hypothèses déclarées avant ce nouveau replay

1. **Intervalle seul** : reprendre exactement les P50 de CoherentP50, conserver
   l'enveloppe contenant à la fois l'intervalle NYX et celui de l'expert, puis
   calibrer chronologiquement. Cette ablation ne change aucune décision de prix.
2. **Signal physique enrichi** : détecteur XGB non pondéré et calibration Platt
   sur les 28 jours chronologiquement les plus récents de la fenêtre d'apprentissage.
   Même hyperparamétrage que précédemment, mais représentation physique enrichie.
3. **Décision commune aux pays** : correction seulement lorsque p > 0,5, seuil
   structurel de la médiane. Suppression de l'ancienne porte de pression p90
   propre à chaque pays ; les pressions restent des entrées du détecteur.
4. **Amplitudes empiriques** : distribution des erreurs normales et des fortes
   sous-estimations, normalisées par le seuil historique propre au pays.
   Même construction cohérente de médiane que CoherentP50, sans forêt d'amplitude.
5. **Candidat principal : physics_governed**. Le gouverneur évalue les poids
   0 / 25 / 50 / 100 % sur les erreurs hors échantillon déjà publiées des 90 jours
   précédents. La sortie directe est un diagnostic, pas un remplacement choisi
   après lecture de l'année.

Ce sont de nouvelles hypothèses inspirées de l'année déjà étudiée, **pas une
validation indépendante**. Le nouveau replay sert à vérifier le fonctionnement
et à documenter les résultats, y compris les échecs. Aucun gagnant ou réglage
par pays n'est activé automatiquement.

## Ce qui est réellement nouveau dans les entrées

Les niveaux et rampes locales/voisines de charge résiduelle, vent et solaire
existaient déjà. Les nouvelles features ajoutent :

- le caractère simultané des hausses de charge résiduelle et des baisses
  d'ENR dans les quatre pays, leurs comptes et amplitudes régionales ;
- les interactions entre charge résiduelle montante et profils ENR décroissants,
  sans soustraire le solaire/vent une deuxième fois à la charge résiduelle ;
- la position de l'heure dans son profil prévu complet pour la journée : rang,
  distance au maximum/quantiles et évolution prévue des prochaines heures ;
- six rangs de rareté historique, ajustés uniquement sur le cœur d'apprentissage,
  par pays et heure civile si le support est suffisant, sinon par pays.

Toutes les heures du profil D+1 utilisé sont des **prévisions connues au même
cutoff de 08 h**, pas des réalisations futures. Cela ne prouve pas que toutes
les sources ont été publiées lors d'un même run fournisseur. Les jours incomplets
(23/24/25 heures selon DST) ou les constituants physiques essentiels manquants
provoquent une abstention ; les températures restent optionnelles.

212 features au total, indicateurs de valeurs manquantes compris. Les prix
électriques, NYX, Storm et leurs retards ne sont pas des entrées du détecteur.
L'erreur historique observé − NYX demeure sa **cible supervisée**, ce qui est distinct.
Les coûts gaz/CO2 sont des proxys fondamentaux autorisés.

Limitations conservées : Pmax journalier n'est pas une rampe de capacité horaire,
le parc sélectionné est incomplet et mélange capacités et production nucléaire
prévue. Ce n'est pas une marge de réserve ni une capacité d'import physiquement
réalisable. Aucun domaine JAO non interprétable au cutoff strict de 08 h n'est
injecté. Les révisions météo absentes restent absentes, pas des révisions nulles.

## Calibration chronologique des bornes

Avant calibrage, la borne basse est au plus celle de NYX ; la borne haute est
au moins celle de NYX. Cette union empêche une baisse de couverture par rapport
à NYX sur les mêmes observations, mais peut élargir sensiblement l'intervalle.
Elle ne garantit ni 80 % de couverture ni une amélioration du score d'intervalle.

Les scores historiques sont `borne_basse_initiale − observé` et
`observé − borne_haute_initiale`. À chaque cutoff, on prend séparément l'ordre
statistique de rang `ceil((n+1)*0,9)` et on n'applique que son excès positif.
Le P50 reste strictement inchangé. Les scores sont toujours calculés à partir
des bornes **avant** calibrage, pour éviter une double inflation.

Fenêtre glissante maximale : 365 jours. Seules les prévisions antérieures et
les observations éligibles déjà disponibles au cutoff sont prises en compte.
Hiérarchie fixée :

| Groupe | Minimum local | Repli autorisé |
| --- | --- | --- |
| Intervention active | 40 heures, 10 journées | Pays regroupés, actifs uniquement : 120 heures, 28 journées |
| Sans intervention | 120 heures, 28 journées | Pays regroupés, inactifs uniquement : 120 heures, 28 journées |
| Support insuffisant | Enveloppe conservée | Statut explicite, aucune calibration inventée |

Les groupes actif/inactif ne sont jamais mélangés silencieusement. Un indicateur
de dérive compare les taux de dépassement récents et plus anciens ; il ne change
pas les hyperparamètres après examen des résultats.

Cette calibration s'inspire de la
[Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222).
Les hypothèses d'échangeabilité ne sont pas garanties sur ces séries temporelles
et ces interventions sélectionnées : aucune garantie de couverture conditionnelle
n'est revendiquée. Les travaux sur les
[prédictions conformes adaptatives en séries temporelles](https://proceedings.mlr.press/v162/zaffran22a.html)
motivent le suivi de couverture et d'efficacité, pas une garantie transférée
automatiquement à notre implémentation.

Les rapports montrent couverture, largeur moyenne et score d'intervalle à80 %,
sur toutes les heures et sur les seules interventions. Une amélioration obtenue
uniquement en élargissant beaucoup les bornes doit rester visible.

## Lancer le laboratoire

```powershell
& 'C:\Users\BQ6757\chronos2_v1\StressGuard.ps1' -Action Run
& 'C:\Users\BQ6757\chronos2_v1\StressGuard.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\StressGuard.ps1' -Action Backtest
& 'C:\Users\BQ6757\chronos2_v1\StressGuard.ps1' -Action Report
```

Run prépare une nouvelle expérience puis rejoue l'année figée. Backtest reprend
le dernier snapshot : des résultats complets et vérifiés sont réutilisés, pas
réentraînés. Un calcul interrompu avant le scellement des résultats est relancé.
Report régénère les HTML à partir des sorties sauvegardées, sans API ni fit.
`-RunDirectory` permet de cibler un snapshot existant. `-DryRun` n'écrit rien
et ne fait aucun appel réseau.

## Valider sur de nouvelles journées

```powershell
# Une fois le replay terminé, figer ce candidat pour les journées à venir.
& 'C:\Users\BQ6757\chronos2_v1\StressGuard.ps1' -Action Freeze

# Après disponibilité des forecasts NYX et des inputs PIT du jour, avant la
# limite d'émission et tant que les observations ne sont pas publiées :
& 'C:\Users\BQ6757\chronos2_v1\StressGuard.ps1' -Action Issue -DeliveryDay YYYY-MM-DD

# Plus tard, récupérer les observations et évaluer les prévisions déjà émises.
& 'C:\Users\BQ6757\chronos2_v1\StressGuard.ps1' -Action Evaluate
```

Issue capture les entrées puis calcule le modèle gelé, sans nouvel entraînement.
Ce premier suivi prospectif évalue donc **le dernier modèle entraîné et figé**,
avec mise à jour causale du gouverneur et des intervalles. Il ne constitue pas
une validation d'un futur réentraînement hebdomadaire automatique. Un nouvel
artefact entraîné devra ouvrir une nouvelle expérience, pas remplacer celui du
journal en cours.
Le gouverneur et le calibrateur d'intervalles peuvent utiliser les nouvelles
erreurs du journal **uniquement après leur réception** et avant le cutoff suivant.
Les prévisions passées ne sont jamais recalculées. Aucun forecast NYX n'est généré
par ce launcher : il lit les runs existants. Si les sources n'ont pas la couverture
nécessaire, il s'arrête explicitement, sans remplir les trous ni actualiser la prod.

Une capture séparée peut être produite avec `-Action Capture`, puis fournie à
Issue avec `-InputDirectory`. Les commandes prospectives utilisent le dernier
journal gelé ; `-LedgerDirectory` en cible un autre. Pour son état :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\StressGuard.ps1' -Action Status -LedgerDirectory 'CHEMIN_REEL_DU_JOURNAL'
```

Deux notions sont distinguées :

- **Information à08 h** : cutoff inchangé pour les features et les labels
  d'apprentissage/calibration, y compris les changements d'heure.
- **Heure réelle d'émission** : le workflow usuel peut finir après08 h. Le grade
  par défaut est `pre_observation_asof08`, avec limite interne prudente à11:45,
  et vérification fraîche de la cible canonique avant ET après l'inférence.
  Ce n'est pas une affirmation de l'heure de publication de l'enchère, ni une
  émission à08 h. Le grade strict d'émission avant08 h reste distinct.

Le registre refuse les journées déjà examinées, même si elles étaient nommées
« live ». La source actuelle contient déjà les observations du15septembre2026.
La première nouvelle livraison est donc au plus tôt le16septembre, si le gel
précède son origine du15septembre08 h. Si le gel est plus tardif, le début est
repoussé automatiquement. Il est impossible de fabriquer cette validation
aujourd'hui en renommant une partie de l'ancien backtest.

Les événements sont append-only, liés aux hashes du candidat, des entrées et des
prévisions. Les labels sont attachés ensuite, par journée/pays complet ; les
observations partielles restent en attente, les journées d'émission manquées
sont signalées. Le premier jeu complet est figé ; une révision divergente ne
réécrit pas les scores antérieurs. Les grades ne sont pas mélangés.

Ce journal constitue une preuve locale vérifiable, pas un horodatage externe
infalsifiable ni une certification historique des publications fournisseur.
365 jours représentés dans l'ancien replay avec amorçage progressif ne signifient
pas365 jours de validation indépendante, ni365 jours d'apprentissage avant chaque
jour évalué. Aucune promotion automatique n'existe dans ce laboratoire.

## Résultats du premier replay, 14 septembre 2026

Snapshot : `runs/experiments/nyx_scarcity_v1/stress_guard/snapshots/20260914T171142Z_86e6f8f9`.
[Synthèse et huit rapports HTML](runs/experiments/nyx_scarcity_v1/stress_guard/snapshots/20260914T171142Z_86e6f8f9/index.html).

Période représentée : 15 septembre 2025 au 14 septembre 2026, soit 365 jours.
Support commun : 34 940 heures-pays (8 735 par pays). Les heures non appariées
à Storm ne sont pas inventées ; la livraison du 15 septembre est hors Statistics.
245 tests passent sans avertissement. Les huit HTML ont été contrôlés sur leurs
valeurs, axes horaires, changements d'heure, Statistics, calendrier et mode nuit
(JavaScript exécuté avec doublures DOM/Plotly, pas de contrôle pixel navigateur).
Les empreintes des quatre rapports opérationnels d'origine sont inchangées.

### Intervalles : amélioration mesurée, sans changer les anciens P50

Sur les mêmes 779 interventions de l'ancien P50 :

| Étape | Couverture | Largeur moyenne EUR/MWh | Score intervalle 80 %, plus bas = mieux |
| --- | ---: | ---: | ---: |
| Ancien intervalle | 65,21 % | 47,86 | 206,05 |
| Enveloppe NYX / expert seule | 75,22 % | 68,05 | 181,44 |
| Enveloppe + calibration chronologique | 78,18 % | 76,36 | 178,33 |

La calibration ajoute donc **2,95 points** de couverture au-delà de l'enveloppe,
pas l'intégralité des 12,97 points. 545 heures disposent d'une calibration active
(533 locale, 12 regroupées sur le même état) ; 234 restent en support insuffisant.
La couverture de 80 % n'est toujours pas atteinte sur toutes les interventions.
Sur l'ensemble des heures communes, la couverture de cet ancien P50 recalibré
atteint 79,44 %, avec une MAE strictement inchangée à 11,34645 EUR/MWh.

### Nouveau signal physique : progrès ponctuels, pas de supériorité validée

| Prévision | MAE annuelle commune EUR/MWh | Corrections annuelles actives |
| --- | ---: | ---: |
| NYX nucléaire + Kalman figé | 11,36750 | — |
| Storm officiel figé | 11,31759 | — |
| Ancien P50 empirique | 11,33371 | 563 |
| StressGuard direct, diagnostic | 11,34475 | 28 |
| StressGuard gouverné, candidat principal pré-déclaré | 11,36750 | 0 |

Le direct améliore 20 corrections et en aggrave 8, dont 3 sur des heures où
NYX était déjà trop haut. Son gain moyen face à NYX est de 0,02275 EUR/MWh ;
l'intervalle bootstrap apparié par journées à 95 % est [-0,00541 ; +0,06113],
donc il inclut zéro. Le bootstrap reste exploratoire sur une année déjà examinée.
Le gain hors 24/25/26 juin et 14 septembre est de 0,00992 EUR/MWh.

Le 14 septembre à 19 h, le direct améliore BE/DE/NL, mais surcorrige FR.
Le 24 juin à 19 h reste mal détecté : probabilités d'environ 3,36 % en BE,
13,30 % en DE et 16,39 % en NL. Les nouvelles transformations des séries
physiques disponibles ne suffisent donc pas à résoudre ces événements.

Les 28 interventions directes restent toutes en
`insufficient_history_same_state` : leur couverture (67,86 %, contre 39,29 %
pour NYX sur ces mêmes heures) provient de l'enveloppe, **pas d'une calibration
active disposant d'assez d'historique**. Le gouverneur n'applique aucune
correction sur l'année ; son absence de dégradation du P50 vient de cette
abstention, pas d'une preuve d'expertise supérieure.

Le log compte une intervention gouvernée supplémentaire sur le live du
15 septembre, DE à 19 h, exclu de l'année : 363,54 → 414,46 EUR/MWh pour
303,365 observé. Elle aggrave l'erreur de 50,91 EUR/MWh. Cette journée était
déjà connue et ne constitue pas une observation indépendante nouvelle.

### Journal prospectif effectivement gelé

Journal : `runs/experiments/nyx_scarcity_v1/stress_guard/prospective/20260914T172023Z_f5e4eaa0`.
Premier jour admissible : **16 septembre 2026**. État initial vérifié :
`armed_no_forecasts`, zéro prévision, zéro journée résolue. Un essai volontaire
d'émission pour le 15 septembre a été refusé avant toute collecte API.

Après le run NYX habituel et disponibilité des données physiques nécessaires,
le 15 septembre, entre 08 h et strictement avant 11 h 45 (Paris), et seulement
si les observations ne sont pas encore publiées :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\StressGuard.ps1' -Action Issue -DeliveryDay 2026-09-16
# Après publication des observations, sans refaire les prévisions :
& 'C:\Users\BQ6757\chronos2_v1\StressGuard.ps1' -Action Evaluate
```

Les quatre pays doivent être prêts. En cas de sources physiques manquantes,
le message précise la mise à jour privée existante à lancer ; aucune donnée
n'est inventée. Les métriques prospectives comparent pour l'instant le candidat
au NYX conservé lors de l'émission ; elles ne constituent pas un suivi Storm
prospectif apparié. Aucun lancement futur automatique n'a été programmé.

Conclusion : la protection et la calibration des intervalles sont implémentées
et mesurées ; le signal physique supplémentaire et la procédure d'évaluation
future sont implémentés, mais **les pics manqués ne sont pas tous résolus** et
la supériorité prédictive n'est pas démontrée. Aucun candidat n'est promu.
