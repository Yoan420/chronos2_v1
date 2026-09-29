# Débloquer la préparation des modèles annuels CPU

Ce correctif concerne l'erreur « 365 jours JAO manquants » ou une capture hydro /
échanges de septembre 2025 absente. Il initialise les historiques publics sur
le poste de travail. Il conserve la branche principale et les scores historiques.

**Il débloque l'initialisation, sous réserve des accès aux sources. Il n'active
pas les prévisions de production : la qualification CPU de la chaîne complète
reste nécessaire.**

## 1. Mettre à jour la branche

Fermer NYX. Dans l'Explorateur Windows, ouvrir le dossier `NYX_CWE_CPU` qui
contient `NYX.pyw`, taper `powershell` dans la barre d'adresse, puis Entrée.
Copier le bloc entier :

```powershell
& {
    $ErrorActionPreference = 'Stop'
    if (-not (Test-Path -LiteralPath '.\NYX.pyw')) {
        throw 'Ouvrir PowerShell dans le dossier NYX_CWE_CPU contenant NYX.pyw.'
    }
    git fetch origin
    if ($LASTEXITCODE -ne 0) { throw 'Echec de git fetch.' }
    git switch codex/nyx-regional-rmse-production
    if ($LASTEXITCODE -ne 0) { throw 'Echec du changement de branche.' }
    git pull --ff-only origin codex/nyx-regional-rmse-production
    if ($LASTEXITCODE -ne 0) { throw 'Echec de git pull.' }
    git branch --show-current
    git log -1 --oneline
}
```

Si Git signale des modifications locales, les conserver et lire son message ;
ne pas utiliser `git reset --hard`. L'environnement `.venv-annual` déjà installé
est réutilisé : cette mise à jour ne nécessite pas de relancer l'installation.

## 2. Vérifier les captures de la livraison à prévoir

Les historiques d'entraînement et les sources de la prochaine livraison sont
contrôlés séparément. Pour une livraison le lendemain, les trois captures
JAO Initial, hydro et échanges doivent exister avant **08:00, heure de Paris,
aujourd'hui**.

Calculer la date du lendemain à Paris et vérifier les captures existantes :

```powershell
$NyxDeliveryDay = & .\.venv-annual\Scripts\python.exe -c "from datetime import datetime, timedelta; from zoneinfo import ZoneInfo; print((datetime.now(ZoneInfo('Europe/Paris')).date() + timedelta(days=1)).isoformat())"
if ($LASTEXITCODE -ne 0) { throw 'Impossible de determiner la date de livraison.' }
& .\.venv-annual\Scripts\python.exe .\run_nyx_annual_daily_capture.py --verify-only --delivery-day $NyxDeliveryDay
```

Les trois sources `jao_initial`, `public_hydro` et `lagged_exchange` doivent
indiquer `COMPLETE`. Un code de sortie non nul signifie qu'au moins une source
n'est pas complète ; consulter son champ `error`.

- **Entre 01:15 et 08:00 Paris**, une capture manquante peut être demandée :

  ```powershell
  .\NYXAnnualCPU.ps1 -Action capture
  ```

- **Après 08:00 Paris**, une capture manquée pour la livraison du lendemain
  ne peut pas être reconstruite avec ce correctif. Installer ou vérifier la
  tâche pour le prochain matin :

  ```powershell
  .\Install-NYXAnnualTasks.ps1
  Get-ScheduledTask -TaskName 'NYX CWE PIT' | Get-ScheduledTaskInfo
  ```

La tâche s'exécute à 07:00, puis retente à 07:15, 07:30, 07:45 et 07:55 Paris.
Le poste doit être allumé, éveillé, connecté au réseau, avec la session ouverte.
Elle capture les sources pour la livraison du lendemain de son exécution.
Par exemple, une capture le 30 septembre au matin concerne le 1er octobre.
Après les trois captures, elle initialise ou actualise aussi les historiques
publics. Le bouton **Capturer les sources** fait la même chose. Aucun accès
Saturn ni entraînement CPU n'est lancé par cette tâche.

## 3. Initialiser les historiques publics

Lorsque les trois captures de la livraison sont complètes, l'initialisation
peut être exécutée avant ou après 08:00 Paris :

```powershell
.\NYXAnnualCPU.ps1 -Action bootstrap
```

La commande vérifie d'abord les trois captures de la livraison, puis récupère
les 365 jours d'historiques JAO, hydro et échanges de la fenêtre d'entraînement.
Les archives JAO compatibles déjà présentes sont réutilisées. Les réponses,
leur provenance et leur date réelle de récupération sont conservées dans les
caches locaux. Le premier téléchargement peut être long ; relancer la commande
après une interruption réutilise ce qui est valide. Si la tâche de capture a
déjà terminé cette initialisation, les archives valides sont réutilisées.

Cette étape ne contacte pas Saturn et ne réentraîne pas les modèles. Un message
d'erreur réseau ou fournisseur doit être résolu pour terminer l'initialisation.
Ne pas désactiver TLS ou modifier les heures dans les reçus.

Le résultat `BOOTSTRAPPED` confirme que les historiques publics sont assemblés.
Il ne publie aucune prévision. Si la première récupération dépasse 08:00 Paris,
ces données ne peuvent pas qualifier la prévision correspondant à cette coupure.
Les exécutions suivantes de la tâche du matin actualisent le cache avant la
nouvelle coupure, sous réserve de leur réussite dans les temps.

## 4. Préparer le calcul

Lorsque les captures de la livraison sont complètes, après la coupure de
08:00 Paris :

```powershell
.\NYXAnnualCPU.ps1 -Action prepare
```

La commande choisit automatiquement la livraison du lendemain à Paris et lance
aussi l'initialisation historique si elle n'a pas encore été faite. Elle
actualise Saturn puis prépare la baseline CPU, les matrices et la référence
de rareté pour les quatre pays. Le premier calcul est plus long que les
suivants ; les étapes valides sont conservées dans les caches.

Le résultat `PREPARED` confirme la préparation des entrées. Il ne publie aucun
prix ni aucune probabilité de prix négatif et ne vaut pas qualification.

Si le premier téléchargement des historiques se termine après la coupure de
la livraison préparée, le lot reste identifié comme impropre à une évaluation
ou une prévision qualifiée pour cette coupure. Ce cache peut servir lors d'une
coupure future, sous réserve des autres contrôles. Le correctif ne change ni
les horodatages réels ni les critères de qualification.

## 5. Rouvrir NYX et conserver un diagnostic

```powershell
& .\.venv-annual\Scripts\pythonw.exe .\NYX.pyw --settings .\runs\annual_desktop.json
```

Dans l'application : **Modèles régionaux → Modèles annuels CPU → Vérifier**.
Le bouton **Initialiser les historiques** exécute l'étape 3 séparément.
Le bouton **Préparer les modèles CPU** lance la même préparation que la commande
ci-dessus. L'initialisation historique est automatique.

Pour conserver un diagnostic dans un fichier :

```powershell
& .\.venv-annual\Scripts\python.exe .\run_nyx_annual_pipeline.py --action inspect | Tee-Object -FilePath .\runs\diagnostic_annuel_poste.txt
```

`forecast_enabled: false`, `ready: false` et un code 2 peuvent être attendus tant
que la qualification complète n'est pas obtenue. Les erreurs de source et
l'erreur d'activation donnent la raison précise. Le statut du dernier calcul
se trouve dans `runs/nyx_annual_cpu_live/YYYY-MM-DD.pipeline.json`.

Le lanceur `NYXAnnualCPU.ps1` conserve automatiquement la sortie complète et
les erreurs dans `runs/logs/nyx_annual_cpu/YYYY-MM-DD/`. Chaque tentative a son
propre journal `.log` et son diagnostic `.diagnostic.json`. Leurs chemins sont
affichés à la fin. Transmettre ce diagnostic permet d'identifier l'action, la
livraison et les sources en erreur ; le seul message « code 2 » ne suffit pas.
La commande `-Action inspect` signale une qualification absente comme un état
de diagnostic attendu, sans la présenter comme une panne du calcul.

Pour une erreur survenue avant cette amélioration du lanceur, lire les derniers
fichiers disponibles, sans relancer le calcul :

```powershell
Get-ChildItem .\runs\nyx_annual_cpu_live -Filter *.pipeline.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content
Get-Content .\runs\live\nyx_annual_cpu\daily_capture_log.jsonl -Tail 1
```

Vérifier les dates et l'action de ces fichiers : ils peuvent concerner une
tentative antérieure si le programme s'est arrêté avant de créer son statut.

### Saturn renvoie une réponse vide pour les anciens prix FR

Le journal du 29 septembre a identifié une requête vide pour
`power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh` : historique du 24 février
2023 au 17 juin 2024, demandé dans l'état Saturn du **17 juin 2024 à 06:00 UTC**.
Cette requête sert au premier jour de préparation, le 18 juin 2024. La réussite
de la collecte des prix à la coupure actuelle ne prouve pas la disponibilité de
cet ancien état.

Après la mise à jour Git, exécuter sur le poste ayant accès à Saturn :

```powershell
& .\.venv-annual\Scripts\python.exe .\diagnose_nyx_annual_saturn_targets.py --delivery-day 2026-09-30 --failed-day 2024-06-18 --zone FR
```

Le diagnostic compare les réponses pour l'ancien état, un intervalle plus court,
l'état actuel et la série officielle alternative déjà référencée dans le code.
Il enregistre un rapport JSON et affiche son chemin. Transmettre ce fichier.
Il ne modifie pas les caches de données et ne lance aucun entraînement.

Une concordance entre l'alternative ancienne et les prix actuels est seulement
un contrôle de compatibilité. Elle n'autorise pas automatiquement l'utilisation
de cette alternative et ne qualifie pas une prévision. La réponse vide reste
bloquante tant que la disponibilité des prix nécessaires n'est pas établie.

Pour lire la provenance dans les fichiers de diagnostic :

- `training_snapshot_max_retrieved_at_utc` indique la dernière récupération
  effective parmi les historiques utilisés.
- `public_history_before_forecast_cutoff` indique si cette récupération respecte
  la coupure de la prévision concernée. La valeur `false` empêche de qualifier
  ce lot pour cette coupure.
- `current_fit_snapshot_v1` identifie la politique qui conserve les observations
  historiques disponibles pour l'entraînement courant, avec leurs dates réelles.

## Ce qui reste nécessaire avant une prévision qualifiée

La récupération des historiques permet de préparer un entraînement futur.
Elle ne prouve pas, à elle seule, les versions qui étaient disponibles lors
de chacune des anciennes prévisions des rapports annuels. La chaîne complète
doit être évaluée avec des données dont la disponibilité à chaque coupure est
vérifiée. FR, BE et NL conservent leurs critères contre Storm ; DE conserve
l'exception de performance acceptée, avec les mêmes contrôles sur les données.

Voir [le guide principal](nyx_annual_production_cpu.md) pour l'évaluation,
l'activation et l'automatisation des prévisions après qualification.
