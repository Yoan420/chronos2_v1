# Démarrer les captures prospectives JAO et échanges sur le poste de travail

L'accès Saturn ne remplace pas les archives quotidiennes JAO Initial et
Energy Charts disponibles **avant 08 h, la veille de chaque livraison**.
`run_nyx_annual_daily_capture.py` lance les deux collecteurs existants pour
la livraison du lendemain, entre 01 h 15 et 08 h (heure de Paris). Il scelle
les réponses et leur horodatage local. Il refuse un jour passé et ne fabrique
aucune archive antérieure. Une erreur JAO n'empêche pas la capture des
échanges ; les exécutions suivantes réessaient JAO sans écraser une capture
déjà valide.

La première fenêtre encore possible au 28 septembre 2026 est **mardi
29 septembre, de 01 h 15 à 08 h**, pour la livraison du **30 septembre**.
L'ordinateur doit être allumé, connecté au réseau et la session de travail
ouverte pendant cette fenêtre. Les contrôles TLS de JAO restent actifs.

## Installer la tâche quotidienne

Ouvrir PowerShell dans le dossier `NYX_CWE_CPU` après avoir récupéré la branche
`codex/nyx-regional-rmse-production`, puis copier le bloc suivant. Il crée une
tâche pour le compte Windows courant, sans demander de mot de passe ni de
droits administrateur. Les cinq déclenchements à 07 h 00, 07 h 15, 07 h 30,
07 h 45 et 07 h 55 permettent de réessayer une publication JAO retardée ;
une capture complète est relue et vérifiée, jamais remplacée.

```powershell
$repo = (Get-Location).Path
$python = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
$script = Join-Path $repo 'run_nyx_annual_daily_capture.py'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Python introuvable : $python" }
if (-not (Test-Path -LiteralPath $script -PathType Leaf)) { throw "Script introuvable : $script" }
$action = New-ScheduledTaskAction -Execute $python -Argument ('-u "' + $script + '"') -WorkingDirectory $repo
$triggers = @(foreach ($minute in 0,15,30,45,55) {
    New-ScheduledTaskTrigger -Daily -At ([datetime]::Today.AddHours(7).AddMinutes($minute))
})
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName 'NYX CWE PIT' -Action $action -Trigger $triggers -Settings $settings -Principal $principal -Description 'Capture quotidienne JAO Initial et échanges avant 08 h Paris' -Force
Get-ScheduledTask -TaskName 'NYX CWE PIT' | Select-Object TaskName,State
```

Si la politique informatique empêche la création d'une tâche planifiée,
exécuter manuellement la commande suivante chaque matin **avant 08 h** et
après **01 h 15**. Une exécution à 07 h est recommandée ; réessayer avant
08 h en cas d'échec, sans changer la date :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' .\run_nyx_annual_daily_capture.py
```

## Contrôler les deux captures

La tâche écrit chaque résultat dans
`runs/live/nyx_annual_cpu/daily_capture_log.jsonl`. Après le premier matin,
ces deux commandes montrent le journal et vérifient les deux archives du
30 septembre **sans appel réseau** :

```powershell
Get-Content .\runs\live\nyx_annual_cpu\daily_capture_log.jsonl -Tail 5
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' .\run_nyx_annual_daily_capture.py --verify-only --delivery-day 2026-09-30
```

Dans le JSON, `sources.jao_initial.state` et
`sources.lagged_exchange.state` doivent tous deux valoir `COMPLETE`.
`ERROR` donne la cause précise. Pour les jours suivants, remplacer
`2026-09-30` par le jour de livraison à contrôler. Les fichiers sont
conservés localement sous `data/pit/nyx_annual_jao_initial_live/` et
`data/pit/nyx_annual_exchange_captures/`; ils sont exclus de Git et doivent
être sauvegardés par l'entreprise si le poste peut être remplacé.

`COMPLETE` dans ce journal signifie **uniquement deux captures du jour**.
La chaîne de prévision annuelle demande encore 365 jours antérieurs pour la
fenêtre d'entraînement, les autres sources, la baseline et les variables
ordonnées, puis une nouvelle évaluation chronologique. Cette commande ne
change ni la qualification ni l'activation des modèles annuels dans NYX.
