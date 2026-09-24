# Expert de coût marginal V2 — offre enrichie, cutoff strict à 08 h

Cette version est **un laboratoire isolé**. `Forecast.ps1`, les modes `Both`
et `Complete`, les modèles opérationnels et les résultats V1 sont inchangés.
Elle améliore la description de l'offre et prépare le solveur réseau, mais
**n'active pas un expert couplé tant que le parc, la référence du RAM et les
frontières ne sont pas qualifiés**.

## Lancer et modifier l'expérience

```powershell
& 'C:\Users\BQ6757\chronos2_v1\MarginalCost.ps1' `
  -Action Run -Config 'config\marginal_cost_expert_v2.yaml' `
  -Countries FR,DE,BE,NL
```

Cette commande ne télécharge pas à nouveau les banques. Elle crée un nouveau
snapshot, vérifie les données, simule les scénarios et produit un rapport HTML
dans `runs/experiments/marginal_cost_expert_v2/snapshots/`. Le chemin du dernier
résultat se trouve dans `runs/experiments/marginal_cost_expert_v2/latest.json`.
Les snapshots terminés ne sont pas écrasés.

- **Paramètres/scénarios** : `config/marginal_cost_expert_v2.yaml`.
- **Séries, unités, hypothèses et périmètre** : `config/marginal_cost_sources_v2.yaml`.
- **Moteur physique réutilisable** : `marginal_cost_expert/dispatch.py`.
- **Qualification JAO** : `marginal_cost_expert/network.py`.
- **Orchestration** : `marginal_cost_expert/runner_v2.py`.

`Prepare` fige les entrées sans entraîner. `Backtest -RunDirectory <snapshot>`
évalue un snapshot préparé. `Report -RunDirectory <snapshot>` régénère seulement
l'HTML à partir des résultats archivés, sans requête externe ni entraînement.
Le schéma du snapshot choisit V1 ou V2 ; le lanceur V1 conserve son défaut.

## Données ajoutées

Support : **10 septembre 2024 au 9 septembre 2026**, soit 730 jours.

| Entrée | Construction | Couverture |
|---|---|---|
| Gaz naturel FR/BE/NL | Pmax Saturn par combustible `fuel.nat_gas` | 730 jours |
| Gaz naturel DE | Même recette | 710 jours ; 8–27 novembre 2024 manquants |
| Charbon DE/NL | Pmax par combustible, banque antérieure lue sans modification | DE 710, NL 730 jours |
| Lignite DE | Pmax distinct du charbon | 710 jours |
| API2 et EUR/USD | Dernière clôture connue avant le cutoff, jamais celle du même jour civil | 730 jours |

Aux Pays-Bas le gaz agrégé récupère environ 3 GW omis par CCGT+GT le 24 juin.
À l'inverse, GT n'est pas synonyme de gaz sur tous les périmètres : ne pas
additionner l'agrégat gaz et les séries par type. Le partage 85 % efficace /
15 % pointe est une **hypothèse de blocs de rendement**, pas une identification
des centrales. Les fractions totalisent exactement 100 % du même agrégat.

API2 est converti explicitement : `USD/t ÷ EURUSD (USD/EUR) ÷ 6,978 MWh_th/t`.
Le lignite n'utilise pas API2 : 2,3 EUR/MWh thermique est une hypothèse technique
[Fraunhofer ISE, juillet 2024, tableau 5](https://www.ise.fraunhofer.de/content/dam/ise/en/documents/publications/studies/EN2024_ISE_Study_Levelized_Cost_of_Electricity_Renewable_Energy_Technologies.pdf),
et non une cotation quotidienne. Les rendements, coûts variables, émissions et
offres nucléaires sont également des hypothèses explicites et modifiables.

Le nucléaire FR reste une **prévision de production**, employée comme bloc bas
coût écrêtable. BE/NL utilisent Pmax, qui n'est pas une prévision de production.
La valeur belge nulle en juin est cohérente avec les arrêts de Doel 4 et
Tihange 3 décrits dans le [monitoring officiel belge](https://economie.fgov.be/fr/themes/energie/securite-dapprovisionnement/monitoring-de/le-monitoring-backward-de).
Ce document publié après juin sert uniquement de contrôle explicatif a
posteriori, jamais d'entrée du modèle historique.

Des timestamps complets ne prouvent pas un parc national complet. Le registre
gaz belge testé est notamment plus restreint que le périmètre national.
Hydraulique flexible, stockage, certains CHP/biomasse/fioul et échanges restent
absents ou non qualifiés. Une simple puissance hydraulique ne fournit pas un
budget d'énergie ; elle n'est pas ajoutée comme production gratuite pendant 24 h.

Les alternatives Saturn `.fct` testées étaient vides. Les séries `.full`
mélangeant disponibilité réalisée et prévue ne remplacent pas un historique
de prévisions manquant. Une donnée attendue absente reste `NaN` ; une
disponibilité publiée de zéro reste zéro.

## Réseau : résultat de la qualification à 08 h

Sur 730 jours et 17 520 heures physiques : **425 journées complètes** et
**13 773 heures de données originales qualifiées pour la recherche**.
Les 6 journées du 4 au 9 septembre sont archivées dans la nouvelle banque ;
les archives JAO opérationnelles n'ont pas été modifiées. Deux sondes
historiques ont renvoyé exactement les mêmes enregistrements partiels : les
trous ne sont pas réparés par un nouveau téléchargement.

L'audit est `data/pit/marginal_cost_expert_v2/network/window_audit.json`.
Il conserve chaque journée, ses sources, ses empreintes SHA et ses motifs
d'inéligibilité. Les agrégats interpolés ne sont jamais utilisés comme CNEC.

Le [handbook JAO](https://publicationtool.jao.eu/core/CORE_PublicationHandbook)
distingue l'initial publié à 01 h 15 du D2CF/RefProg annoncé à 10 h 30.
**Ce dernier est exclu** conformément au cutoff demandé de 08 h.
Le RAM initial ne peut pas être posé arbitrairement comme le membre droit
d'un domaine à positions nettes nulles : la référence des flux et toutes les
frontières doivent être comprises et validées. Les hubs virtuels AHC actifs
depuis juin 2026 sont conservés ; aucune position externe n'est supposée nulle.

Les requêtes historiques as-of et les dates `lastModifiedOn` ne constituent
pas une preuve indépendante de capture publique avant chaque origine.
L'audit ne certifie donc aucune capture opérationnelle historique à 08 h.
Même une journée avec toutes ses contraintes demeure non qualifiée pour le
calcul couplé si sa référence et ses frontières ne le sont pas.

## Calibration, évaluation et intervention

1. Les simulations zonales n'utilisent aucun prix électrique passé ni prévision
   Chronos/Storm. Elles construisent une pile à partir de demande résiduelle,
   capacités et coûts de combustibles/CO₂.
2. Trois hypothèses de rendements sont calculées indépendamment des labels.
   À chaque origine D−1 08 h, le scénario est choisi uniquement sur les
   **365 jours calendaires précédant D**. Les trous ne sont pas compressés.
3. Les **365 jours évalués sont du 10 septembre 2025 au 9 septembre 2026**.
   Les labels de calibration et les observations/prévisions de comparaison
   proviennent du snapshot V1 final : mêmes heures, mêmes observations,
   référence figée par pays. Aucune sélection quotidienne du meilleur modèle.
4. Les révisions historiques des labels ne sont pas certifiées publiées à
   chaque origine ; la calibration reste rétrospective et diagnostique.
   Les jours du 24–26 juin sont un épisode déjà identifié, pas un test vierge.
5. Les simulations brutes sont présentées **séparément** des sorties qualifiées.
   Une pénalité de déficit de 4 000 EUR/MWh mesure un manque de la pile partielle,
   pas une rareté réelle certifiée. Tous ces extrêmes restent dans les scores.
6. L'expert couplé est indisponible : son poids reste nul sur toute la période,
   et le modèle actuel est intégralement conservé. Ce n'est ni un poids appris,
   ni une amélioration annuelle démontrée. On n'entraîne pas une gouvernance
   pour masquer l'absence de qualification physique.

L'HTML contient les scores annuels, une comparaison appariée sur les seules
heures communes, les journées de juin, les courbes quotidiennes et horaires,
la couverture réseau, ainsi qu'un mode nuit.

### Résultat obtenu le 9 septembre 2026

Le diagnostic reste insuffisant pour intervenir. Sur les heures communes
avec Storm et la simulation calibrée :

| Pays | Jours / heures communs | MAE référence | MAE simulation zonale non qualifiée |
|---|---:|---:|---:|
| FR | 365 / 8 759 | 11,942 | 105,950 |
| DE | 286 / 6 863 | 11,703 | 399,960 |
| BE | 365 / 8 759 | 10,973 | 2 821,246 |
| NL | 365 / 8 759 | 11,120 | 43,262 |

Unités : EUR/MWh. La comparaison allemande est plus courte uniquement pour le
diagnostic : les trous de calibration de novembre 2024 interdisent les premières
fenêtres. **La référence et l'abstention conservent bien 365 jours / 8 760 heures**
dans les Statistics annuelles. Le tableau apparié exclut aussi l'heure absente
de Storm.

Sur toute l'année évaluée, la pile centrale enrichie présente un déficit
simulé dans 0,046 % des heures NL, 10,902 % DE, 72,169 % BE et 1,233 % FR.
Ce n'est pas une fréquence de pénurie du marché réel. Les grandes erreurs
belges et allemandes confirment qu'une offre partielle sans imports ne peut
pas encore représenter le marché couplé. Aucune intervention n'a été activée,
y compris sur les 24–26 juin ; aucune amélioration annuelle n'est revendiquée.

## Étape suivante avant un véritable essai conditionnel

Il faut compléter/attester le parc et ses vintages, interpréter un domaine
initial avec sa référence connue avant 08 h, et représenter explicitement
chaque frontière. Le solveur couplé est prêt et testé sur des cas synthétiques,
mais ces preuves manquantes empêchent encore le test réseau réel. Changer
un booléen de configuration ne contourne pas cette restriction.

Après qualification seulement : réexécuter la calibration et le test sur la
même année, puis réutiliser la gouvernance préquentielle pour mesurer les
interventions bénéfiques/nuisibles, la MAE annuelle, les extrêmes et les prix
moyens. Une validation prospective nouvelle reste nécessaire avant activation.

## Reconstituer les banques séparément

Les collecteurs utilisent deux connexions maximum et des checkpoints.
Depuis la racine du projet, avec le Python de l'environnement :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m marginal_cost_expert.supply_sources coal-fx
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m marginal_cost_expert.supply_sources gas-capacities --zones FR DE BE NL
```

La requalification réseau est une lecture des archives, sans nouveau download :

```python
from marginal_cost_expert.network import audit_network_window
audit_network_window(
    "data/pit/jao_core_flowbased", "2024-09-10", "2026-09-09",
    overlay_root="data/pit/marginal_cost_expert_v2/network",
    output_path="data/pit/marginal_cost_expert_v2/network/window_audit_new.json",
)
```

Le fichier de sortie doit être nouveau ; sélectionner ensuite explicitement
cet audit dans la configuration V2. `Prepare` refusera un audit calculé avec
une ancienne version du code de qualification ou des fichiers sources modifiés.
