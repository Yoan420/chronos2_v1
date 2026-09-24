# Economic Value Added — laboratoire de décision Chronos-2

## Extension indépendante : expert des mouvements extrêmes

Pour entraîner et comparer la nouvelle **politique de position gouvernée** :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\EconomicExpert.ps1' -Action Run
```

Cette commande conserve le forecast et les quantiles `nuclear_kalman`, puis
teste des positions différentes sur le même snapshot EVA. Elle n'altère pas
les commandes ni les modèles opérationnels. Paramètres et méthode :
`config/economic_extreme_policy.yaml` et `ECONOMIC_EXTREME_POLICY.md`.
`EconomicValue.ps1` conserve le diagnostic initial à règle fixe décrit ci-dessous.

## Lancer le rapport

Depuis **n'importe quel dossier PowerShell** :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\EconomicValue.ps1' -Action Run
```

Le lanceur lit les rapports locaux déjà produits, puis crée un snapshot et un
rapport HTML dédié. **Il ne relance aucun forecast, aucun entraînement, aucune
synchronisation et ne passe aucun ordre.** `Forecast.ps1`, `Both` et `Complete`
ne changent pas.

Par défaut : FR, DE, BE, NL ; quatre candidats `autonomous`, `kalman`,
`nuclear_autonomous`, `nuclear_kalman` ; **100 MW au total** ; dernière date
d'export commune et 365 derniers jours complets d'observations communs.
Les candidats sont quatre alternatives pour le même portefeuille, pas quatre
portefeuilles cumulés. Avec quatre pays, l'allocation fixe est de 25 MW par pays.

```powershell
# Un seul candidat sur les quatre pays, toujours 100 MW au total
& 'C:\Users\BQ6757\chronos2_v1\EconomicValue.ps1' `
  -Action Run -Models nuclear_kalman -PortfolioMW 100

# Comparaison autonome / Kalman en France : 100 MW sur FR
& 'C:\Users\BQ6757\chronos2_v1\EconomicValue.ps1' `
  -Action Run -Countries FR -Models autonomous,kalman -PortfolioMW 100

# Figer explicitement les dates et utiliser une autre configuration
& 'C:\Users\BQ6757\chronos2_v1\EconomicValue.ps1' `
  -Action Run -DeliveryDay 2026-09-10 -EndDay 2026-09-10 `
  -Config 'config\economic_value.yaml'

# Afficher seulement la commande ; aucun fichier créé
& 'C:\Users\BQ6757\chronos2_v1\EconomicValue.ps1' -DryRun
```

Les résultats sont dans `runs/experiments/economic_value_v1/snapshots/<identifiant>/`.
Le chemin du dernier rapport terminé est dans `runs/experiments/economic_value_v1/latest.json`.
La console affiche aussi le chemin cliquable du rapport. Aucune ouverture de
navigateur n'est imposée par le script.

## Ce que mesure — et ne mesure pas — cette version

À votre demande, le prix de référence provisoire est le **prix day-ahead de la
veille à la même heure civile**. Ce prix porte sur une **autre livraison** :
ce n'est pas une cotation à laquelle on pouvait acheter ou vendre la livraison
prévue. Ce choix permet de comparer les signaux, **pas de prouver un P&L réalisable**.

Storm est la prévision de la stratégie benchmark, jamais le prix d'exécution.
Le rapport n'utilise donc pas Storm à la fois comme prix de référence et comme
signal benchmark, ce qui rendrait sa stratégie artificiellement neutre.

Ici « Economic Value Added » signifie **différence de P&L net simulé entre le
candidat et Storm**. Ce n'est pas l'EVA comptable après rémunération du capital.
L'absence de forecast correspond à une absence de position incrémentale, pas à
une modélisation complète du coût d'approvisionnement d'un portefeuille client.

Une formulation honnête pour présenter ce premier résultat est :

> Sur un portefeuille hypothétique de 100 MW, le signal produit une différence
> de score économique de X euros par rapport à Storm, sur les mêmes heures,
> après les coûts supposés. Le prix de référence n'étant pas exécutable,
> ce montant n'est pas un gain de trading démontré.

## Règle commune aux trois stratégies

Pour le pays z et l'heure h, soit `edge = forecast − reference`.
Le seuil effectif est le seuil de signal plus coûts et slippage supposés.
Par défaut : **5 + 0,5 + 0,5 = 6 EUR/MWh**.

```text
Position = +Qz si edge > 6 ; −Qz si edge < −6 ; 0 sinon
Énergie signée (MWh) = Position (MW) × durée réelle (h)
P&L brut = Énergie signée × (observé − référence)
Coûts = |Énergie signée| × (coût de transaction + slippage)
P&L net = P&L brut − Coûts
EVA = P&L net candidat − P&L net Storm
```

La stratégie 0 reste sans position. Les stratégies 1 (Storm) et 2 (candidat)
utilisent **exactement la même capacité, le même seuil et les mêmes coûts**.
Les seuils sont déclarés avant l'évaluation : aucune recherche du seuil le plus
rentable sur l'année affichée. Les coûts par MWh représentent une friction
totale hypothétique du chemin simulé ; ils ne sont pas des tarifs réels attestés.

L'allocation entre pays est constante et n'est pas redistribuée en cas de manque
de données. Pour le total portefeuille, une heure est retenue seulement si tous
les pays alloués sont évaluables à cette heure. Les positions ne sont pas
cumulées entre les variantes de modèle.

## Métriques et dénominateurs

| Indicateur | Définition |
|---|---|
| P&L brut, coûts, P&L net | Sommes en EUR sur les heures communes évaluables |
| EVA | P&L net candidat moins P&L net Storm sur le même échantillon |
| P&L/MWh | P&L net divisé par la somme des MWh absolus effectivement positionnés |
| EVA/MW, échantillon | EVA divisée par la capacité fixe allouée ; aucune extrapolation |
| EVA/MW/an | Disponible seulement si les 365 jours sont entièrement appariés |
| Sharpe quotidien de P&L | `sqrt(365) × moyenne(P&L quotidien) / écart-type(ddof=1)` sur journées complètes |
| Hit ratio directionnel | Part des positions actives dont le sens forecast−référence correspond au sens observé−référence |
| Transactions gagnantes | Part des positions actives avec P&L net strictement positif ; distinct du hit ratio |
| Drawdown maximal | Plus forte baisse du cumul quotidien de P&L, en incluant le point initial zéro |
| Drawdown en % | Seulement si un capital initial positif a été spécifié ; sinon non défini |

Une stratégie sans transactions n'a pas de P&L/MWh ou de hit ratio défini.
Un P&L quotidien constant n'a pas de Sharpe défini. Ces résultats restent vides,
pas artificiellement égaux à zéro ou infinis. Une première journée perdante
compte immédiatement dans le drawdown. Le drawdown quotidien ne capture pas
les creux intra-journaliers ni les appels de marge.

Le Sharpe est un **ratio descriptif de P&L**, pas un rendement sur capital investi.
Son annualisation repose sur des hypothèses de dépendance temporelle qui ne
sont pas garanties. [Sharpe (1994)](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)
précise le rôle des rendements différentiels ; [Lo (2002)](https://alo.mit.edu/publications/page/18/)
montre l'importance de l'autocorrélation pour l'interprétation et l'annualisation.
Le rapport ne présente ni une significativité statistique ni une garantie de
gain futur. Les queues de distribution doivent être examinées séparément.

## Confiance, quantiles et vue horaire

Les P10 et P90 sont ceux des rapports effectivement produits. Aucune borne
n'est inventée si elle manque. La confiance est une **heuristique ex ante**
combinant la largeur de l'intervalle et l'écart du forecast au proxy : ce n'est
pas une probabilité calibrée de gagner un trade.

La confiance ne filtre pas les positions dans la configuration initiale :
Storm n'a pas nécessairement les mêmes quantiles. Un filtre appliqué seulement
au candidat rendrait la comparaison asymétrique. Les analyses par confiance
classent les **mêmes heures selon la confiance du candidat** pour toutes les
stratégies ; la couverture observée de l'intervalle nominal à 80 % est indiquée.

Le tableau horaire expose forecast, référence, edge, P10/P90, confiance,
BUY/SELL/FLAT, position, observation et P&L. BUY/SELL sont des étiquettes de
simulation, pas des ordres ou recommandations. Les observations absentes
et les P&L non évaluables restent vides. Les lignes prospectives sont exclues
des métriques historiques, même si un label est fourni par erreur.

Le rapport propose aussi : cumul de P&L, drawdown quotidien, performance par
heure, par mois, par saison et pendant les spikes.

- Hiver : octobre à mars ; été : avril à septembre.
- Spikes positifs : observé ≥ 200 EUR/MWh ; négatifs : observé ≤ 0 EUR/MWh.
- Les catégories de spikes sont **des diagnostics ex post**, jamais un filtre
  permettant de prendre position avec la connaissance du prix futur.

## Point-in-time, observations et changements d'heure

Le cutoff cible reste D−1 **08 h Europe/Paris**. Le lanceur contrôle l'origine
civile et refuse une référence déclarée disponible après cette origine.
Pour le proxy veille, l'heure de disponibilité est une hypothèse D−2 à 18 h,
pas une preuve de publication authentifiée. Les caches canoniques et Storm
peuvent inclure des révisions postérieures ; les exports ne certifient pas un
replay neuronal hors-échantillon ou une disponibilité historique exacte à 08 h.
Ces limites sont présentes dans le rapport et les manifestes.

La sélection automatique retient la dernière date d'export **commune** à tous
les couples pays/modèle demandés. Une nouvelle journée disponible pour seulement
une partie du batch ne fait pas mélanger silencieusement les livraisons.
La fenêtre d'évaluation se termine à la dernière journée d'observations complète
commune. Un `EndDay` explicite au-delà de cette limite est refusé.

Chaque comparaison utilise un même observé et un même Storm par pays/heure,
priorisés selon la date de publication du rapport, sans sélection par performance.
Les différences de révision entre sources sont auditées.

L'identité des heures est UTC, avec calendriers civils de 23, 24 ou 25 heures.
Le proxy n'est **pas** calculé par un décalage aveugle de 24 heures UTC.
Si l'heure civile précédente manque ou est ambiguë, on s'abstient : ni moyenne,
ni duplication arbitraire. L'heure Storm manquante déclarée par son audit DST
reste absente. Un rapport portant sur 365 jours peut donc avoir quelques heures
non appariées : il affiche leur nombre et ne revendique pas un montant annuel
intégral en €/MW/an.

Les prévisions et prix utilisés sont horaires. Le marché couplé a introduit les
produits quart-horaires pour livraison le 1er octobre 2025 : ce laboratoire
horaire ne reproduit ni leur exécution ni leurs variations intra-horaires.
[Documentation EPEX SPOT](https://www.epexspot.com/en/new-15-minute-products-market-coupling).

## Paramètres, archives et reprise

Les principaux paramètres se règlent dans **`config/economic_value.yaml`** :
modèles/pays, capacité, seuil, coûts, seuils de spikes et dates. La définition
hiver/été est fixe dans cette version du moteur et explicitée ci-dessus.
Pour comparer une autre recette, copier la configuration puis lancer `Run`
avec `-Config`. Les résultats antérieurs restent disponibles.

| Action | Effet |
|---|---|
| Audit | Contrôle les sources en lecture seule ; ne produit pas de snapshot |
| Prepare | Fige panel, configuration, provenance, empreintes et calendrier |
| Backtest | Évalue le dernier snapshot préparé, ou `-RunDirectory` explicite |
| Run | Prepare puis Backtest et génération HTML |
| Report | Régénère l'HTML depuis les résultats scellés, sans collecte/calcul de stratégie |
| Status | Lit l'état du dernier snapshot préparé, ou d'un snapshot explicite |

```powershell
& 'C:\Users\BQ6757\chronos2_v1\EconomicValue.ps1' -Action Audit
& 'C:\Users\BQ6757\chronos2_v1\EconomicValue.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\EconomicValue.ps1' -Action Report
```

Chaque snapshot contient le panel d'entrée, les lignes des trois stratégies,
les métriques quotidiennes et agrégées, les ventilations, la configuration et
les manifestes SHA256. Les prévisions originales sont seulement lues, jamais
modifiées. Les résultats sont publiés par remplacement atomique ; le pointeur
du dernier rapport réussi n'est pas remplacé par un échec.

Les verrous système empêchent les doubles évaluations. Un processus terminé
libère son verrou ; une propriété de verrou inconnue n'est pas supprimée
aveuglément. Une modification concurrente des sources est détectée et demande
une nouvelle préparation. `Backtest` d'un snapshot déjà terminé vérifie ses
empreintes et régénère seulement le rapport. Un changement de code après
`Prepare` exige un nouveau snapshot avant calcul.

## Pour passer à une véritable évaluation de sourcing

Il faudra un historique du **prix négociable du même produit et de la même
livraison à l'instant de décision**, avec bid/ask, vintage et profondeur/volume
accessible ; des prévisions historiquement disponibles ; puis coûts,
contraintes physiques, obligations de sourcing, inventaire/hedges, financement
et collatéral. Le mode actuel refuse d'être déclaré exécutable par simple
changement d'un booléen. Il constitue une première évaluation reproductible
des signaux et une interface d'extension, pas une homologation trading.

## Premier diagnostic généré — 10 septembre 2026

Le snapshot `20260910T085138Z_13a598c8` couvre le 11 septembre 2025 au
10 septembre 2026. Il conserve 8 757 heures appariées sur 8 760 par pays :
une heure Storm et deux heures du proxy veille sont indisponibles/ambiguës.
Le portefeuille représente 100 MW au total, répartis à 25 MW par pays.

| Variante | EVA simulée contre Storm, après frictions |
|---|---:|
| autonomous | −988 350,25 EUR |
| kalman | −743 732,75 EUR |
| nuclear_autonomous | −508 247,31 EUR |
| nuclear_kalman | −412 867,50 EUR |

`nuclear_kalman` est le meilleur des quatre candidats avec cette règle,
mais il reste derrière Storm. Son EVA par MW alloué est de −4 128,675 EUR/MW
**sur l'échantillon**, pas une estimation annuelle complète. Les P&L et
Sharpes bruts élevés dépendent fortement du proxy non exécutable ; ils ne
doivent pas être présentés comme des opportunités de profit réalisables.
Ce résultat ne déclenche aucune modification des modèles ou des seuils.
