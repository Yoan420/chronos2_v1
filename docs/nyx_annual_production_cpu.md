# NYX annuel CPU — FR, DE, NL et BE

Branche : `codex/nyx-regional-rmse-production`.

Cette chaîne reprend les recettes des rapports annuels au 23 septembre 2026 :
baseline Chronos/correcteur/Kalman, référence de rareté, experts prix propres
aux pays et probabilités de prix négatif. Les modèles sont réentraînés sur CPU.
Les scores historiques conservés dans NYX restent identifiés comme tels.

**État au 29 septembre 2026 : le logiciel est livré, l'activation de production
reste verrouillée.** Les captures historiques vérifiées JAO/hydro/échanges
nécessaires à la qualification complète ne sont pas disponibles dans le dépôt.
La collecte future ne peut pas reconstituer une ancienne version de ces sources.
Une commande d'installation réussie ou un rejeu des experts sur des matrices
historiques ne suffit pas à lever ce verrou.

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
Le poste doit être allumé, éveillé, connecté au réseau, avec la session ouverte.

Dans l'app : **Modèles régionaux → Modèles annuels CPU → Vérifier**, puis
**Capturer les sources** pendant la fenêtre autorisée. Équivalent PowerShell :

```powershell
.\NYXAnnualCPU.ps1 -Action capture
```

La livraison par défaut est le lendemain à Paris. Les captures tardives
ne sont jamais présentées comme disponibles avant la coupure de 08:00.

## 4. Préparer les données et modèles

Après 08:00 Paris, la veille de la livraison :

```powershell
.\NYXAnnualCPU.ps1 -Action inspect
.\NYXAnnualCPU.ps1 -Action prepare
```

Dans NYX, le bouton correspondant est **Préparer les modèles CPU**.
La préparation réalise successivement :

1. Assemblage et vérification des archives JAO, hydro et échanges.
2. Mise à jour Saturn à la date de coupure propre à chaque journée : profils
   prévus, historiques des prix, combustible et disponibilités thermiques.
3. Chronos-2 CPU, correcteur résiduel CPU et Kalman pour les quatre pays.
4. Construction exacte des matrices 449/503 colonnes et projection 123 colonnes.
5. Réentraînement HGB/Test2, validation chronologique et référence de rareté.
6. Scellement du lot et de toutes ses preuves.

Le premier démarrage exige **834 jours de covariables Saturn**, 469 jours de
baselines et **366 captures quotidiennes JAO/hydro/échanges** pour une livraison.
Il peut prendre beaucoup plus longtemps qu'un calcul quotidien. Les caches
sont réutilisés ensuite ; une interruption ne supprime pas les étapes terminées.
Sur le même volume, les archives immuables sont partagées par liens physiques
pour éviter de recopier chaque jour tout le démarrage historique. Sur un autre
volume, une copie vérifiée est utilisée ; prévoir davantage d'espace disque.
Les capacités thermiques ne requièrent ensuite que 13 nouveaux états quotidiens,
en plus du contrôle global des séries.

Si une archive manque ou est invalide, la commande s'arrête en `BLOCKED` avant
la collecte Saturn et les entraînements, en indiquant les sources concernées.
Les captures quotidiennes restent disponibles séparément.
`PREPARED` signifie que les entrées sont préparées ; aucune prévision n'est publiée.
Pour imposer une date, ajouter `-DeliveryDay YYYY-MM-DD`.

## 5. Évaluer toute la chaîne sur le CPU du poste

Cette étape nécessite les archives vérifiées pour **chaque** fenêtre quotidienne
de la période évaluée. Sur une année complète, cela couvre 730 journées de
captures publiques, fenêtres d'entraînement comprises. Les fichiers historiques
du PC personnel téléchargés après leurs coupures ne remplacent pas ces preuves.

Une fois les archives disponibles, cette commande prépare les journées dans
l'ordre, exporte les comparaisons officielles EPEX/Storm, entraîne les modèles,
calcule les scores puis demande l'activation si tous les contrôles passent :

```powershell
.\Evaluate-NYXAnnualCPU.ps1 -FirstDay 2025-09-24 -StopDayExclusive 2026-09-24 -Activate
```

Ces dates correspondent aux rapports annuels ; elles ne rendent pas les archives
manquantes disponibles. Une autre période de 365 jours consécutifs est possible.
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

Une erreur `residual_bank` ne provient pas du nouveau lanceur annuel : utiliser
le raccourci **NYX annuel CPU** et son panneau annuel. Pour une erreur de source,
lire le journal avant de relancer ; ne pas modifier les reçus ni remplacer les
valeurs manquantes pour forcer l'activation.
