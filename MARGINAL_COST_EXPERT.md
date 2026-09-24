# Expert indépendant de coût marginal zonal — laboratoire v1

## Statut et utilisation

Prototype de recherche **isolé et non activé**. `Forecast.ps1`, `-Mode Both`,
`-Mode Complete`, les modèles actuels et leurs caches ne sont pas modifiés.
Ce premier test porte sur une **pile résiduelle partielle gaz + nucléaire**,
pas encore sur un merit order complet couplé au réseau. Un coût simulé n'est
pas l'identification d'une centrale réelle.

Depuis n'importe quel dossier PowerShell :

```powershell
# Nouvelle expérience, sur les dates et comparateurs figés dans le YAML.
& 'C:\Users\BQ6757\chronos2_v1\MarginalCost.ps1' -Action Run -Countries FR,DE,BE,NL

# Refaire seulement le dernier rapport : aucun entraînement ni téléchargement.
& 'C:\Users\BQ6757\chronos2_v1\MarginalCost.ps1' -Action Report
```

`Prepare` crée un nouveau snapshot des données, références et paramètres.
`Backtest -RunDirectory '...\snapshots\identifiant'` évalue ce snapshot.
`Audit` effectue les contrôles et archive leur résultat, sans entraîner.
Un snapshot terminé n'est jamais écrasé par un nouveau backtest.

- **Paramètres physiques, scénarios, gouvernance, dates, références** :
  `config/marginal_cost_expert.yaml`.
- **Séries, unités, formules et hypothèses par pays** :
  `config/marginal_cost_sources.yaml`.
- **Résultats** : `runs/experiments/marginal_cost_expert_v1/snapshots/.../`.
- **Lien vers le dernier résultat** : `runs/experiments/marginal_cost_expert_v1/latest.json`.

Le dossier contient le rapport HTML interactif autonome, les prédictions
horaires, métriques annuelles et quotidiennes, scénarios, décisions et audits.
Le mode nuit et les contrôles pays/date fonctionnent sans connexion Internet.
L'expert brut est masqué initialement dans les graphiques pour ne pas écraser
l'échelle ; il reste affichable dans la légende et entièrement évalué.

### Résultat du test figé le 9 septembre 2026

MAE horaire sur les 8 760 heures de référence et d'intervention :

| Pays | Référence | Avec intervention | Heures d'intervention |
|---|---:|---:|---:|
| FR | 11,9410 | 11,9410 | 0 |
| DE | 11,6770 | 11,6206 | 558 |
| BE | 10,9736 | 10,9712 | 7 |
| NL | 11,1185 | 11,1185 | 0 |

L'Allemagne gagne environ 0,48 % de MAE, mais sa RMSE passe de 22,4837 à
22,5953 et sa MAE du prix moyen quotidien augmente légèrement. Sur les
558 heures d'intervention, 301 sont améliorées et 257 aggravées. La Belgique
ne gagne que 0,0023 EUR/MWh, sur sept heures : ce n'est pas une validation
robuste. **Aucun gain sur les 24–26 juin dans les quatre pays.**

L'expert brut est trop incomplet : le déficit de pile concerne environ 70,4 %
de ses heures belges, 68,5 % de ses heures allemandes disponibles, 12,5 % des
heures néerlandaises et 0,9 % des heures françaises. Ce n'est pas de la rareté
réelle mesurée. **Conclusion : conserver les modèles actuels ; compléter et
qualifier l'offre et le réseau avant de tester une promotion.**

Référence reproductible : snapshot `20260909T135218Z_afa10eaa`, observations
des rapports figés. Les deux snapshots intermédiaires précédents utilisaient
le cache canonique en évaluation ; ils sont conservés pour audit et ne sont
pas le benchmark principal. Aucun scénario ou poids n'a été optimisé sur
les résultats de juin. Suite technique : 114 tests réussis.

## Méthode

### 1. Une prédiction physique indépendante

À l'inférence, le moteur ne lit **ni prix électriques passés, ni Chronos,
ni correcteur résiduel, ni Kalman, ni Storm**. Il prend une demande résiduelle
prévue, une offre nucléaire bas coût, des capacités thermiques disponibles
et les derniers prix de combustible/CO₂ connus au cutoff D−1 08:00 civil.

Pour une technologie thermique :

```text
coût EUR/MWh électrique =
    (combustible EUR/MWh thermique + CO₂ EUR/t × émissions t/MWh thermique)
    / rendement
    + exploitation variable
    + prime d'offre explicitement supposée
```

Les blocs sont appelés par coût croissant. Le coût du bloc marginal donne
le prix simulé. Trois scénarios sont déclarés avant le test : central,
flexible et contraint. Les variations de disponibilité, les primes d'offre
et les offres négatives sont des **hypothèses**, pas des valeurs observées.

Le choix du scénario est recalibré chaque jour avec les prix historiques
observés des **365 jours calendaires précédents**, sans utiliser le label
du jour prédit. L'indépendance aux prix concerne les *entrées d'inférence* :
les labels électriques restent nécessaires pour évaluer/calibrer la banque.

La demande résiduelle conserve son signe. Pour une demande négative :

```text
demande de pile = max(charge résiduelle, 0)
offre bas coût = nucléaire + max(−charge résiduelle, 0)
```

Cette représentation conserve le bilan résiduel mais ne reconstitue pas
la totalité des productions et de leur écrêtement. En FR/DE, l'hydraulique
au fil de l'eau est déjà soustraite par la formule Saturn : pas de double
soustraction.

### 2. Une intervention conditionnelle limitée

```text
prévision finale = référence + poids × (expert − référence)
```

Les régimes « tendu » / « surplus » viennent exclusivement de la pile
prévue, jamais de la volatilité observée après coup. Le régime neutre conserve
la référence. Les poids possibles sont 0, 0,10, 0,25 et 0,50.

La politique est réévaluée toutes les semaines sur au plus 365 jours passés :

- démarrage à poids nul pendant au moins 60 jours **complets de prévisions
  expert hors échantillon** ;
- au moins 14 jours distincts du régime considéré ;
- gain passé minimal de `max(0,05 EUR/MWh, 0,5 % de MAE)` et contrôle de la
  non-régression moyenne passée ;
- contrôle de l'aggravation d'une journée passée, augmentation du poids
  limitée à 0,25 par décision ;
- abstention si données absentes, historique incomplet ou preuve de gain
  insuffisante. Aucune correction rétrospective d'une mauvaise intervention.

Ces contrôles ne garantissent **ni une amélioration future, ni une absence
de dégradation des jours extrêmes**. RMSE, MAE du prix moyen et pires journées
doivent aussi être examinées ; améliorer légèrement la MAE ne suffit pas.

### 3. Protocole de comparaison

Période principale : **10 septembre 2025 → 9 septembre 2026**, 365 jours.
Support initial : **10 septembre 2024 → 9 septembre 2026**, 730 jours.
Les heures UTC physiques préservent les journées civiles de 23/24/25 heures.

Références choisies une seule fois : nucléaire + Kalman pour FR/BE/NL,
Kalman ordinaire pour DE. Aucun choix du meilleur modèle à chaque journée.

Deux tableaux distincts :

1. Référence et intervention sur les **8 760 heures** de l'année, sans
   supprimer les abstentions. L'expert seul affiche sa propre couverture.
2. Modèles sur les **mêmes heures communes avec Storm et l'expert disponible**.
   L'unique heure Storm absente à l'automne n'est pas interpolée. En DE, les
   premiers jours sans expert ne figurent pas dans ce second tableau.

Les observations d'évaluation sont celles des **rapports figés**, pour
reproduire leur benchmark. Elles ne sont pas annoncées comme la dernière
révision disponible. Le scénario physique utilise les labels du snapshot
canonique d'entraînement ; un label manquant peut uniquement être complété
par la trace *observée* du rapport, avec provenance par ligne. La gouvernance
utilise les observations appariées à sa référence figée. Les désaccords entre
cache et rapport sont audités, pas arbitrés heure par heure selon le score.

Limite importante : les révisions historiques des observations ne sont pas
certifiées disponibles à chaque cutoff. Les archives du modèle de référence
ne certifient pas non plus un nouvel entraînement neuronal OOF. Il s'agit d'un
**diagnostic rétrospectif**, pas d'une preuve de performance prospective.
Les 24–26 juin, déjà identifiés avant ce travail, restent un diagnostic
post-hoc et non un jeu de test final indépendant.

## Données effectivement collectées et limites

Les nouvelles collectes sont isolées dans `data/pit/marginal_cost_expert/`.
Saturn fournit les capacités Pmax CCGT/GT et nucléaire BE/NL, interrogées au
cutoff historique civil. La collecte en bloc a été contrôlée contre des
requêtes individuelles, notamment autour des changements d'heure. Les
capacités **quotidiennes** sont diffusées sur les heures physiques de la
journée : elles ne constituent pas des prévisions horaires distinctes.

- FR/BE/NL : 730 jours des entrées utilisées.
- DE : 20 jours manquants, du 8 au 27 novembre 2024. Les NaN restent dans
  le calendrier. L'expert s'abstient jusqu'au 27 novembre 2025 inclus ; il
  devient évaluable le 28 novembre avec une fenêtre complète. La référence
  et l'intervention restent évaluées sur les 365 jours entiers.
- TTF M1 et EUA premier décembre : banque isolée resynchronisée jusqu'au
  9 septembre. Ce sont des proxies de coût disponibles avant le cutoff,
  pas des prix spot de combustible observés le jour futur.
- Nucléaire FR : prévision de génération. BE/NL : disponibilité Pmax utilisée
  comme bloc bas coût écrêtable, **pas une prévision de production**.
- Le périmètre du parc couvert par le fournisseur n'est pas qualifié. Un
  Pmax belge nul ne prouve pas que tout le nucléaire belge est arrêté.
- Le segment GT est approximé par un coût gaz OCGT ; son mix de combustibles
  par unité n'est pas vérifié.
- Charbon/lignite collectés partiellement mais **exclus du test**, faute de
  contrat de prix combustible et de couverture de parc validé. Autres
  omissions : plusieurs filières, stockage, hydraulique modulable et échanges.

La pénalité de 4 000 EUR/MWh représente un **déficit de cette pile incomplète**,
pas une estimation validée de la rareté réelle. Elle explique une grande
partie des erreurs de l'expert seul. Ne pas abaisser arbitrairement cette
pénalité pour masquer les données/technologies manquantes.

Le moteur contient aussi un solveur couplé PTDF/RAM (HiGHS), testé sur des
cas synthétiques : équilibre zonal, conservation des positions nettes,
contraintes CNEC complètes et prix duaux. **Il n'est pas raccordé aux données
JAO dans ce backtest réel**. Des RAM agrégées ne sont jamais converties en
capacités bilatérales fictives. Une version réelle exige la représentation
des zones/bords extérieurs et des vintages réseau valides, y compris les
changements de périmètre du couplage.

## Suite à privilégier

1. Qualifier le périmètre des disponibilités et reconstruire la pile complète,
   en priorité le nucléaire belge, le charbon/lignite allemand et néerlandais,
   puis hydraulique, CHP et stockage. Ajouter des prix combustibles historiques
   connus au cutoff avec unités et conversion vérifiées.
2. Raccorder les CNEC/PTDF/RAM originaux à un véritable bilan multi-zone ;
   documenter chaque frontière et s'abstenir si les données ne suffisent pas.
3. Garder cette gouvernance prudente, évaluer les fausses interventions,
   la MAE annuelle, les erreurs extrêmes et les prix moyens séparément.
4. Après fixation d'une recette, valider sur des journées **prospectives**
   nouvelles. Aucune promotion automatique n'existe dans ce laboratoire.

## Références méthodologiques

- [Ghelasi & Ziel — Learning the Merit Order](https://arxiv.org/abs/2501.02963)
  motive une structure de pile calibrable ; les résultats de cet article ne
  constituent pas une garantie pour nos données ou notre protocole.
- [Description publique EUPHEMIA](https://www.nemo-committee.eu/assets/files/euphemia-public-description.pdf)
  : le couplage réel est plus complexe qu'un empilement zonal convexe.
- [SciPy / HiGHS](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html)
  : optimisation linéaire et sensibilités/duaux des contraintes.

Tests ciblés :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pytest `
  tests/test_marginal_cost_model.py tests/test_marginal_cost_data.py `
  tests/test_marginal_cost_evaluation.py tests/test_marginal_cost_governance.py `
  tests/test_marginal_cost_runner.py -q
```
