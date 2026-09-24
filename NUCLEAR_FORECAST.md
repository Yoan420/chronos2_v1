# Previsions nucleaires : lancement habituel et rapports comparatifs Storm

Le lancement habituel reste **strictement inchange** :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL -Mode Both
```

L'option `-WithNuclear` produit deux variantes distinctes dans les exports
habituels, sans remplacer les variantes autonomous / kalman existantes ni
leurs configurations ou contrats scelles :

1. `nuclear_autonomous` : Chronos-2 + prevision nucleaire FR + correcteur residuel.
2. `nuclear_kalman` : le meme upstream + Kalman gouverne, avec le nucleaire egalement dans ses
   variables de marche.

Il s'agit de challengers **sans LoRA**. Une activation LoRA sur une zone est
refusee dans ce parcours : ajouter une variable au schema d'un adaptateur
entraine demanderait un nouvel entrainement, pas un changement silencieux.
Le travail LoRA NOAA reste en pause.

## Tous les modeles avec une seule commande

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL -Mode Complete
```

`Complete` produit pour chaque pays `autonomous`, `kalman`,
`nuclear_autonomous` et `nuclear_kalman`, avec leurs rapports comparatifs et CSV
dans `runs/exports/YYYY-MM-DD/<pays>/<variante>/`. Pour la France seule, utiliser
`-Countries FR`.

Une seule livraison est determinee au debut (demain en heure de Paris, sauf
`-DeliveryDay YYYY-MM-DD` explicite) et transmise a toutes les etapes, meme si
le calcul passe minuit. Le Both habituel passe d'abord, puis les deux variantes
nucleaires de chaque pays. Les calculs sont successifs, sans concurrence GPU.
Observations et Storm sont actualises par chaque pipeline, selon leurs contrats
habituels; les instants d'extraction peuvent donc differer entre les etapes.

Le bilan final conserve tout echec; les autres etapes sont tentees par defaut.
Ajouter `-StopOnError` pour arreter au premier echec. Une interruption clavier
arrete l'enchainement. `-DryRun` affiche la commande sans rien lancer.

Ne pas ajouter `-WithNuclear` ou `-NuclearStage` avec Complete. Le mode utilise
les configurations actuelles et des poids deja presents localement; les
overrides Kalman/LoRA, download et SkipObservedSync ne sont pas acceptes.
`-NuclearConfig` peut pointer vers un autre fichier nucleaire du meme projet.
L'incompatibilite avec une activation LoRA nucleaire est controlee avant le
premier processus. Complete utilise **Both, pas All**; les anciens modes
restent inchanges et les variantes historiques abandonnees ne sont pas relancees.

Une premiere livraison nucleaire reste couteuse (730 jours de replay) et les
premiers runs DE/BE/NL peuvent donc prendre longtemps. Le mode incremental est
active par defaut : les livraisons suivantes reutilisent les journees valides
et calculent les nouvelles journees. Les relances de la meme livraison
reutilisent leurs resultats figes. Hors FR, le signal ajoute est
toujours la production nucleaire **francaise**, pas celle du pays cible.

## Lancement conseille : commencer par FR

### 1. Recuperer les previsions historiques (optionnel : Run le fait automatiquement)

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR -Mode Both -WithNuclear -NuclearStage Sync
```

Sans `-DeliveryDay`, la date de livraison est demain, heure de Paris. Utiliser
la meme `-DeliveryDay YYYY-MM-DD` sur toutes les commandes pour un replay.
Le cutoff D-1 08:00 Paris doit deja etre passe.

La collecte nucleaire couvre initialement 730 jours d'historique et la livraison.
Le mode incremental conserve ensuite le support historique de son initialisation,
dans la limite des 1095 jours bruts necessaires aux fenetres imbriquees. Elle est decoupee
en blocs de 31 jours : les blocs valides sont conserves et une relance ne
demande que les jours manquants. Un cache existant invalide est refuse, jamais
remplace silencieusement. Les charges residuelles viennent d'une copie du
backfill Saturn as-of audite, prolongee automatiquement (voir correction
ci-dessous). Run actualise aussi les prix observes et Storm pour le reporting.
Pour une nouvelle livraison, les observations recentes anterieures a D
completent une copie isolee du cache cible avant son gel; le cache partage
reste intact. Les observations de D ne sont jamais injectees dans les inputs.

### 2. Controler les donnees (optionnel, rapide, sans modele)

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR -Mode Both -WithNuclear -NuclearStage Audit
```

Cet audit verifie les sources nucleaire et charge residuelle, ainsi que la
presence du cache de prix.
Le run effectue ensuite les controles complets de preparation/couverture
des autres variables avant toute inference. Aucun remplissage par les
observations nucleaires n'est autorise.

### 3. Produire les deux variantes et leurs rapports

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR -Mode Both -WithNuclear
```

Les rapports HTML et forecasts CSV se trouvent sous :

- `runs/exports/YYYY-MM-DD/fr/nuclear_autonomous/`
- `runs/exports/YYYY-MM-DD/fr/nuclear_kalman/`
- `runs/exports/YYYY-MM-DD/fr/nuclear_index.html` : liens vers les deux rapports.

Le manifeste `current_nuclear_batch_manifest.json` du pays trace les fichiers
publies et leurs empreintes. Il ne remplace pas le manifeste du batch incumbent.
Les calculs et audits sources restent sous
`runs/experiments/nuclear_forecast_v1/YYYY-MM-DD/fr/civil_pit_v2/`.
Ajouter `DE,BE,NL,ES` a `-Countries` pour tester le meme signal francais dans
ces marches interconnectes ; ce ne sont pas leurs productions nucleaires.

Le premier run initialise l'historique nucleaire : il peut etre long. Il ne
recycle pas les anciennes predictions Chronos depourvues de nucleaire. Ensuite,
le cache partage sous `output_root/_daily_cache/<pays>/civil_pit_v2/` conserve les
predictions Chronos, les corrections residuelles et les refits Kalman par jour.
Les empreintes portent sur les donnees effectivement utilisees pour chaque jour,
la recette, le modele local et le code. Une revision d'entrees invalide les
calculs qui en dependent ; un changement de recette ouvre un nouvel espace de
cache. Les anciens resultats figes restent consultables.

Pour une livraison quotidienne consecutive avec des entrees inchangees,
l'objectif est une seule nouvelle inference Chronos, un refit du correcteur et
un refit Kalman. Le correcteur et le Kalman utilisent toujours leurs 365 jours
anterieurs ; aucun entrainement ne voit les observations de la livraison.
Le passage de la prevision de la veille dans l'historique ajoute ses observations
sans refaire un calcul dont les entrees de prevision sont identiques.
Une interruption peut reprendre les journees validees. Aucun delai de production
n'est garanti avant mesure sur la machine et les sources reelles.

Le debut du warm-up est ancre lors de l'initialisation. Il ne redemarre plus
chaque matin : les predictions historiques conservent ainsi leur trajectoire.
Les rapports gardent 365 jours evalues et les archives leur couverture de 730
jours ; jusqu'a 1095 jours bruts sont conserves pour fournir 365 jours de
calibration au plus ancien jour corrige. La prevision de la nouvelle livraison
utilise les memes fenetres de donnees que le replay complet de reference.
En revanche, certaines anciennes predictions Kalman des rapports peuvent
differer d'un replay qui deplacerait a nouveau le debut du warm-up ; cela ne
modifie pas retroactivement les livraisons deja figees.

`computation_mode: incremental` dans `config/nuclear_forecast.yaml` active ce
parcours pour les nouveaux snapshots. `computation_mode: full`, ou
`python run_nuclear_forecast.py --computation-mode full --delivery-day YYYY-MM-DD`,
demande le replay complet de reference. Pour comparer les deux modes sur une
meme date deja figee, utiliser une copie de la configuration avec un autre
`output_root` sous `runs/experiments/`, puis la passer avec `--config`.

Les runs historiques complets du moteur legacy `bda946d3...` peuvent etre
importes explicitement avec `migrate_nuclear_daily_cache.py --source-day
2026-09-09 --zones FR NL`. Cet outil controle les archives et leurs entrees,
reconstruit les quantiles float32 exacts des statistiques, compare une livraison
Chronos/CatBoost reelle puis verifie la relecture des corrections et du Kalman
sans entrainement. Il conserve l'ancre historique et publie les caches seulement
apres ces controles. Les recus sont dans `runs/tmp/ncm_*/<pays>/receipt.json`.
Un snapshot interrompu en ancien mode complet doit etre conserve en sauvegarde
et remplace par un nouveau snapshot incremental pour utiliser ces caches ; ne
pas modifier ses empreintes pour contourner le controle de compatibilite.
Un snapshot ne peut pas etre reinterprete dans un autre mode. Les anciens
snapshots sans mode explicite gardent leur comportement de replay complet.
Un processus Python deja lance continue avec le code qu'il a charge ; cette
amelioration prend effet aux prochains lancements.

### 4. Actualiser uniquement les rapports d'une livraison deja calculee

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR -Mode Both -WithNuclear -NuclearStage Report -DeliveryDay 2026-09-09
```

Cette action recharge le resultat fige, extrait les observations et Storm a
nouveau et republie les deux rapports. Aucun replay ni entrainement n'est
autorise. Le resultat fige est enregistre apres un Run termine; une premiere
relance Run migre les anciens resultats depuis leurs checkpoints valides.
Run reutilise egalement ce resultat lorsqu'il existe pour la meme livraison.
L'attribution deja calculee est conservee en mode Report; elle reste indiquee
indisponible si elle n'avait pas ete produite, sans relancer un modele.

`-SkipObservedSync` est un mode hors ligne explicite : observations du cache
local et snapshot Storm audite du run habituel du meme pays / meme jour.
Il n'affirme pas avoir actualise les sources. Une panne d'API en mode normal
echoue explicitement au lieu de publier silencieusement des donnees anciennes.

Les deux variantes d'une zone sont executees successivement ; plusieurs zones
sont aussi traitees successivement. `-Threads` limite les threads CatBoost et
`-Workers` le parallelisme des refits Kalman (et des requetes pour Sync).
`-DryRun` affiche seulement la commande. `All`, `Blend` et les overrides
`-KalmanConfig` / `-LoraActivationConfig` ne sont pas disponibles avec cette
option. Le parcours s'arrete au premier echec.

## Source et controles anti-fuite

### Correction du 08/09 : historique de charge residuelle

Le parcours utilise maintenant le fichier Saturn as-of audite
`data/pit/kalman_hybrid/residual_load_market_features.parquet` comme amorce
en lecture seule. Une copie est creee sous `data/pit/nuclear_forecast/`, puis
completee automatiquement jusqu'a la livraison demandee. Pour le 09/09/2026,
seuls les cinq jours du 05 au 09 septembre doivent etre demandes a Saturn.
Les cinq anciens fichiers `data/pit/vintages/*_residual_load_fcst.parquet`
ne constituent plus la source de ce challenger : ils omettaient notamment
des heures de fin de journee sur l'historique. Il ne suffit pas de remplir
ces trous, car le contrat de selection des autres valeurs differe aussi.

La copie conserve l'audit d'origine, les fuseaux propres a chaque serie
(notamment UTC pour NL) et les quatre heures NL de printemps deja derivees
par moyenne des voisins du meme forecast as-of. Ces quatre approximations
restent explicitement declarees ; aucune nouvelle interpolation generique,
aucune realisation et aucune revision posterieure au cutoff ne sont admises.

Toutes les covariables du challenger sont selectionnees selon le vrai
calendrier D-1 08:00, y compris le lendemain des changements d'heure. La
couverture physique complete des 730 jours et de D est controlee **avant**
Chronos. La selection est materialisee et auditee dans le dossier prepare.
Le contexte de prix est borne au support historique requis (730 jours au depart,
jusqu'a 1095 jours en incremental) plus les 2048 heures
necessaires a Chronos avant le premier jour. Le seuil de couverture n'est pas
abaisse et les annees de prix inutilisees ne faussent plus ce controle.

Un ancien snapshot incomplet n'est ni efface ni descelle : le nouveau
sous-dossier `civil_pit_v2` contient les inputs et caches corriges. Le mode
Both habituel et les archives de production ne sont pas modifies.

Pour verifier les donnees jusqu'au bout sans commencer le backtest :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR -Mode Both -WithNuclear -NuclearStage Prepare
```

`Prepare` et `Run` peuvent completer le cache residuel isole si son suffixe
manque. `Audit` reste strictement en lecture seule. `Sync` synchronise desormais
le nucleaire puis ce cache residuel. Un cache audite deja complet est reutilise.

### Prevision nucleaire

- Serie Saturn : `power.fr.generation.nuclear.gw.fcst`, production **prevue en
  GW**, distincte de la disponibilite REMIT en MW.
- Requete historique as-of D-1 08:00 **civil Europe/Paris**, avant l'enchere.
  Ce timestamp est celui de la requete, pas une preuve de publication initiale
  du fournisseur. Les rapports restent experimentaux/non promouvables sur
  cette seule preuve.
- Les libelles horaires naifs de cette source sont interpretes en Paris.
  L'ancien cache `fr_nuclear_generation_fcst_long.parquet`, construit avec une
  autre interpretation, n'est pas reutilise.
- Au retour a l'heure d'hiver, les sondages ont montre un seul libelle 02:00
  pour deux heures physiques. La politique par defaut `duplicate` reutilise
  explicitement cette valeur comme covariable pour les deux folds. Ce n'est
  pas une deuxieme observation independante. L'audit signale cette convention
  et les jours potentiellement concernes ; le nombre exact de reparations
  n'est pas fourni par le materialiseur generique. Pour refuser cette
  convention, choisir `incomplete_dst_policy: raise` dans le YAML.
- Aucun prix cible, aucune production realisee, aucune donnee post-enchere
  n'est utilise comme substitut de cette prevision.
- Inputs copies et empreintes verifies dans chaque dossier d'experience.
  Les retries gardent les memes donnees et le meme commit Chronos local.

Les sondages Saturn ont retourne des previsions anciennes et recentes ; la
collecte complete doit encore confirmer chaque heure des 731 jours. Il n'y a
pas de fallback RTE silencieux. L'API RTE Generation Forecast ne constitue pas
une source equivalente de prevision nucleaire ; les historiques RTE
d'indisponibilite peuvent servir ulterieurement de variable complementaire,
mais ne doivent pas etre renommes « production prevue ».

## Methode et lecture des performances

La variable est injectee dans le contexte et les covariables futures de
Chronos-2. `oracle` est le nom technique existant d'une strategie d'input connu :
ici les valeurs sont des **forecasts selectionnes as-of**, pas les realisations.
Le reseau Chronos reste gele : pas de nouvel entrainement neuronal.

Le correcteur utilise la recette CatBoost existante, reentrainee chaque jour
sur les erreurs Chronos des 365 jours civils anterieurs. Ses experts d'entree
sont ici les quantiles Chronos recalcules, sans reutiliser les experts
LEAR/ensemble d'une ancienne archive. Le Kalman conserve les parametres et la
banque gouvernee du fichier operationnel, avec une covariable nucleaire de
plus dans le groupe `market`. Les aggregats de charge residuelle restent
inchanges : on ne melange pas production et charge dans leur moyenne.

Le premier bloc de 365 jours initialise le correcteur et le Kalman. Durant le
demarrage, le correcteur laisse passer Chronos tant que le minimum de lignes
n'est pas atteint. Les 365 jours suivants constituent la periode evaluee
commune aux deux variantes ; chaque fit exclut la journee qu'il predit. Le
warmup ne remplace pas une validation imbriquee complete avec 1095 jours.

Les rapports standards indiquent les bornes FINAL365 (D-365 a D-1) et ajoutent
la livraison D, avec observation si publiee ou case vide sinon. Les rubriques
Storm sont les memes que dans les rapports habituels : bandeau du jour,
Statistics, prix moyens, calendrier, historique et profil horaire. Les
comparaisons utilisent les memes heures physiques disponibles. Une valeur
Storm absente reste vide et auditee; les observations du modele ne sont pas
effacees pour masquer ce trou. ES n'a pas de source Storm officielle verifiee.
Les observations canoniques revisees ne servent qu'aux nouveaux scores des
rapports, apres verification d'un snapshot recent; les labels d'entrainement
et predictions figees restent intacts. Sans cet audit de rafraichissement,
les differences historiques autres que l'arrondi float32 sont refusees.

**Ce n'est pas encore une mesure causale du gain de la seule variable** : le
protocole de refit et les experts du correcteur different de certaines
archives incumbent. Une promotion demanderait un controle sans nucleaire
refait sous le meme protocole, la verification des sources, puis une validation
prospective. Rien n'est active automatiquement dans `Both`.

## Parametrage et maintenance

Le fichier `config/nuclear_forecast.yaml` contient chemins, politique DST,
taille des blocs, configurations de base par pays et chemin du Kalman. Pour
modifier des hyperparametres sans toucher au live, copier les YAML de base /
Kalman dans des fichiers experimentaux, puis pointer vers ces copies depuis
ce fichier. Seuls les `filter_parameters` du Kalman sont repris, jamais les
prefixes de predictions incumbent.

Changer `output_root` (sous `runs/experiments/`) pour une nouvelle configuration.
Ne pas modifier les snapshots ni leurs empreintes. Un verrou persistant apres
un arret brutal demande de verifier que le processus est termine avant de
retirer uniquement ce verrou.

Code : `run_nuclear_forecast.py` (orchestration),
`chronos2_hourly/nuclear_sources.py` (source et audit),
`chronos2_hourly/nuclear_forecast.py` (inference/refits),
`chronos2_hourly/nuclear_reporting.py` (rapports).
