# NYX annuel CPU — FR, DE, NL et BE

Branche : `codex/nyx-regional-rmse-production`.

Cette chaîne reprend les recettes des rapports annuels au 23 septembre 2026 :
baseline Chronos/correcteur/Kalman, référence de rareté, experts prix propres
aux pays et probabilités de prix négatif. Les modèles sont réentraînés sur CPU.
Les scores historiques conservés dans NYX restent identifiés comme tels.

**État au 29 septembre 2026 : l'initialisation des historiques publics est
disponible ; l'activation de production reste verrouillée.** La préparation
peut récupérer les historiques JAO, hydro et échanges et conserver leur date
réelle de récupération. Il n'est pas nécessaire d'attendre 366 nouvelles
captures pour commencer à préparer les données. En revanche, un téléchargement
effectué aujourd'hui ne prouve pas ce qui était disponible lors d'une ancienne
prévision. La qualification de la chaîne CPU complète reste nécessaire.

Si le calcul s'arrête sur une archive de septembre 2025 absente, suivre
[les commandes de mise à jour et d'initialisation](nyx_annual_history_bootstrap.md).

## 1. Récupérer la branche sur le poste de travail

Fermer la fenêtre NYX avant la mise à jour. Dans PowerShell, à la racine du clone :

```powershell
Set-Location "$env:USERPROFILE\chronos2_v1"
git fetch origin
git switch codex/nyx-regional-rmse-production
git pull --ff-only origin codex/nyx-regional-rmse-production
```

Si le clone est ailleurs, adapter seulement le chemin de `Set-Location`.
Si Git signale des changements locaux, les conserver avant de changer de branche.
La branche principale reste disponible pour revenir à l'installation précédente.

## 2. Installer l'environnement CPU et le raccourci NYX

Python 3.11 est requis. L'installateur crée `.venv-annual`, installe PyTorch CPU
et les bibliothèques figées, télécharge le snapshot exact de Chronos-2 puis
crée le raccourci **NYX annuel CPU**. L'accès Internet aux dépôts Python et
Hugging Face est nécessaire pour cette installation.

```powershell
.\Setup-NYXAnnualCPU.ps1
```

Le client Saturn privé doit provenir de l'installation autorisée du poste.
L'installateur recherche notamment l'environnement `venvs\pricefm311`.
Pour donner explicitement son chemin :

```powershell
.\Setup-NYXAnnualCPU.ps1 -SaturnPython "$env:USERPROFILE\venvs\pricefm311\Scripts\python.exe"
```

Le client Saturn n'est pas publié dans GitHub. Si le poste ne l'a pas déjà,
utiliser le fichier wheel ou l'index interne fourni par l'équipe Saturn
(options `-SaturnWheel` ou `-SaturnIndexUrl` de l'installateur).
Les certificats, accès réseau et droits Saturn restent ceux du poste.
Ne pas désactiver la validation TLS pour contourner une erreur de certificat.

Le nouveau raccourci utilise un état de console séparé et le port 8767.
Le lanceur NYX vérifie l'identité du serveur et peut sélectionner un port libre
si celui-ci appartient déjà à une autre application.

## 3. Commencer les captures quotidiennes

```powershell
.\Install-NYXAnnualTasks.ps1
Get-ScheduledTask -TaskName 'NYX CWE PIT' | Get-ScheduledTaskInfo
```

La tâche capture JAO Initial Computation, l'hydro et les échanges à 07:00 Paris,
puis retente à 07:15, 07:30, 07:45 et 07:55 si nécessaire. Les captures existantes
sont vérifiées. Les réponses brutes et leur heure réelle sont conservées.
Après les trois captures, la tâche initialise ou actualise les historiques
publics afin de les récupérer avant la coupure. Elle ne contacte pas Saturn
et ne lance pas les entraînements CPU. Le premier téléchargement peut durer
au-delà de 08:00 ; dans ce cas, il ne qualifie pas les données pour cette coupure.
Le poste doit être allumé, éveillé, connecté au réseau, avec la session ouverte.

Dans l'app : **Modèles régionaux → Modèles annuels CPU → Vérifier**, puis
**Capturer les sources** pendant la fenêtre autorisée. Équivalent PowerShell :

```powershell
.\NYXAnnualCPU.ps1 -Action capture
```

La livraison par défaut est le lendemain à Paris. Les captures tardives
ne sont jamais présentées comme disponibles avant la coupure de 08:00.

## 4. Préparer les données et modèles

Une fois les trois captures de la livraison effectuées, les historiques publics
peuvent aussi être initialisés séparément, avant ou après 08:00 Paris. Cette
commande s'exécute indépendamment de Saturn et des entraînements :

```powershell
.\NYXAnnualCPU.ps1 -Action bootstrap
```

Elle vérifie les trois captures de la livraison, puis récupère les 365 jours
historiques JAO, hydro et échanges nécessaires à
la fenêtre d'entraînement et réutilise les archives compatibles déjà présentes,
y compris celles de JAO provenant du dépôt. Les heures réelles de récupération
et la provenance sont conservées. Cette commande ne reconstitue pas une capture
de la journée à prévoir qui aurait été manquée avant 08:00 Paris.
Dans NYX, le bouton correspondant est **Initialiser les historiques**.
Le statut `BOOTSTRAPPED` confirme seulement l'assemblage des historiques publics.

Après 08:00 Paris, la veille de la livraison, et une fois les captures de cette
livraison effectuées :

```powershell
& .\.venv-annual\Scripts\python.exe .\run_nyx_annual_pipeline.py --action inspect
.\NYXAnnualCPU.ps1 -Action prepare
```

Dans NYX, le bouton correspondant est **Préparer les modèles CPU**.
La préparation réalise successivement :

1. Initialisation, assemblage et vérification des historiques JAO, hydro et
   échanges ; contrôle séparé des captures de la journée à prévoir.
2. Mise à jour Saturn : profils prévus à leur coupure quotidienne, avec reprise
   des journées historiques manquantes dans la version disponible à la coupure
   de la livraison préparée ; combustible et disponibilités thermiques à leur
   coupure quotidienne. Les prix historiques utilisent la coupure de la
   livraison préparée ; les calculs internes restent limités aux prix antérieurs
   à leur journée. Les révisions réelles et les reprises sont tracées.
3. Chronos-2 CPU, correcteur résiduel CPU et Kalman pour les quatre pays.
4. Construction exacte des matrices 449/503 colonnes et projection 123 colonnes.
5. Réentraînement HGB/Test2, validation chronologique et référence de rareté.
6. Scellement du lot et de toutes ses preuves.

Le premier démarrage exige **834 jours de covariables Saturn**, 469 jours de
baselines, **365 jours d'historiques publics** et les captures de la livraison
à prévoir. Les historiques peuvent être récupérés au démarrage, sous réserve
de leur disponibilité chez les fournisseurs. Il peut prendre beaucoup plus
longtemps qu'un calcul quotidien. Les caches
sont réutilisés ensuite ; une interruption ne supprime pas les étapes terminées.
Sur le même volume, les archives immuables sont partagées par liens physiques
pour éviter de recopier chaque jour tout le démarrage historique. Sur un autre
volume, une copie vérifiée est utilisée ; prévoir davantage d'espace disque.
Les capacités thermiques ne requièrent ensuite que 13 nouveaux états quotidiens,
en plus du contrôle global des séries.

Si une source ne peut pas être récupérée ou vérifiée, ou si une capture de la
livraison à prévoir manque, la commande s'arrête en `BLOCKED` avant la collecte
Saturn et les entraînements, en indiquant les sources concernées.
Les captures quotidiennes restent disponibles séparément.

Un historique récupéré après la coupure du calcul peut servir à préparer les
matrices, mais ne rend pas le lot éligible à un backtest ou à une prévision
qualifiée pour cette coupure. Ce statut est conservé avec la provenance des
données ; les contrôles de qualification ne sont pas désactivés. Pour une
coupure future, le cache déjà récupéré peut être utilisable si tous les autres
contrôles passent.

`PREPARED` signifie que les entrées sont préparées ; aucune prévision n'est publiée
et ce statut ne vaut pas qualification.

La politique de profils `own_origin_with_current_fit_recovery_v1` conserve les
anciennes versions complètes et récupère, si possible, les journées historiques
manquantes à la coupure du calcul actuel. Elle ne remplace jamais le profil du
jour à prévoir et ne certifie pas les anciennes versions manquantes. Cette
politique est liée à la qualification CPU et contrôlée lors de l'activation.
Voir [les détails de reprise Saturn](nyx_annual_history_bootstrap.md).

La politique de prix `current_fit_origin_reconstruction_v1` utilise, pour un
entraînement donné, les historiques disponibles à sa coupure. Ses courbes
internes servent à l'apprentissage et ne certifient pas des prévisions émises
dans le passé. La qualification complète doit évaluer cette même politique à
chaque date extérieure ; une qualification de l'ancien régime ne l'active pas.
Voir aussi [le correctif Saturn et la commande de reprise](nyx_annual_history_bootstrap.md#correctif-des-anciennes-versions-de-prix-saturn).
Pour imposer une date, ajouter `-DeliveryDay YYYY-MM-DD`.

## 5. Évaluer toute la chaîne sur le CPU du poste

Cette étape nécessite des données vérifiées pour **chaque** fenêtre quotidienne
de la période évaluée. Une année de prévisions avec 365 jours d'entraînement
couvre 730 journées de données publiques. Ce nombre décrit la couverture de
données et n'impose pas d'attendre 730 jours pour initialiser un poste.

Pour qu'un score rétrospectif qualifie la production, il faut aussi démontrer
que les données utilisées étaient disponibles à la coupure de chaque prévision
évaluée. Un historique révisé téléchargé aujourd'hui peut convenir à un
entraînement futur ; il ne suffit pas à démontrer cette disponibilité passée.
La récupération initiale ne fabrique pas ces preuves. Tant qu'elles manquent,
un rejeu peut servir au diagnostic, mais ne débloque pas l'activation.

Une fois les archives disponibles, cette commande prépare les journées dans
l'ordre, exporte les comparaisons officielles EPEX/Storm, entraîne les modèles,
calcule les scores puis demande l'activation si tous les contrôles passent :

```powershell
.\Evaluate-NYXAnnualCPU.ps1 -FirstDay 2025-09-24 -StopDayExclusive 2026-09-24 -Activate
```

Ces dates correspondent aux rapports annuels ; elles ne rendent pas les preuves
de disponibilité historique manquantes disponibles. Une autre période de
365 jours consécutifs est possible.
Un essai plus court, sans `-Activate`, sert uniquement au diagnostic.
Le rejeu complet peut être long : chaque journée réentraîne trois experts prix
et quatre classifieurs. La même commande reprend les journées déjà scellées.

L'évaluation possède ses propres caches, distincts du lancement quotidien.
Les prix observés et Storm sont lus pour le score après le scellement des
prévisions. FR/BE/NL doivent battre Storm en RMSE et sur plus de 50 % des heures.
DE bénéficie de l'exception de performance demandée, avec ses écarts affichés ;
les contrôles de données et de calcul restent les mêmes pour les quatre pays.

Une qualification lie code, recette, poids Chronos et versions des bibliothèques
réellement utilisées. Le reçu d'un autre environnement Python n'active pas automatiquement
ce poste. Le détail des commandes individuelles est dans
[le guide d'évaluation](nyx_annual_cpu_full_chain_evaluation.md).

## 6. Lancer et automatiser la production après qualification

```powershell
.\NYXAnnualCPU.ps1 -Action forecast
.\Install-NYXAnnualTasks.ps1 -EnableQualifiedForecast
```

Dans NYX : **Vérifier → Lancer les quatre pays**. La tâche automatique lance
la chaîne à **08:05 Paris**. La mise à jour Saturn et le réentraînement font
partie de chaque lancement. Une qualification absente ou devenue invalide
empêche la publication.

Les sorties comportent le prix horaire en €/MWh, la probabilité de prix négatif
et le seuil de décision. Les 23/24/25 heures physiques sont conservées.
Les résultats manuels se consultent dans **Derniers lancements** ; les résultats
automatiques apparaissent dans **Prévisions annuelles publiées**, avec rapport
HTML et CSV par pays. Leurs fichiers sont dans `runs/nyx_annual_cpu_live/YYYY-MM-DD/`.

## Journaux et dépannage

- Tâches automatiques : `runs/logs/nyx_annual_cpu/YYYY-MM-DD/*.log`.
- Statut quotidien : `runs/nyx_annual_cpu_live/YYYY-MM-DD.pipeline.json`.
- Sources et preuves : `runs/live/nyx_annual_cpu/YYYY-MM-DD/`.
- Caches des collecteurs : `data/pit/nyx_annual_*`.
- Résultats du rejeu complet : `runs/evaluations/annual_cpu_full_chain/evaluation/`.

Le diagnostic peut retourner `ready: false` et un code 2 tant que la chaîne
n'est pas qualifiée. Cela ne signifie pas à lui seul que l'environnement CPU
ou Saturn est mal installé. Lire les champs de source et l'erreur d'activation.

Une erreur `residual_bank` ne provient pas du nouveau lanceur annuel : utiliser
le raccourci **NYX annuel CPU** et son panneau annuel. Pour une erreur de source,
lire le journal avant de relancer ; ne pas modifier les reçus ni remplacer les
valeurs manquantes pour forcer l'activation.
