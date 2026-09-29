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

### Correctif des anciennes versions de prix Saturn

Le journal du 29 septembre a identifié une requête vide pour
`power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh` : historique du 24 février
2023 au 17 juin 2024, demandé dans l'état Saturn du **17 juin 2024 à 06:00 UTC**.
Cette requête servait au premier jour de préparation, le 18 juin 2024. Le
diagnostic du poste a confirmé que les 11 498 heures demandées sont complètes
dans la version du 29 septembre 2026 à 06:00 UTC. La requête à l'ancienne date
reste vide, y compris sur 2 048 heures. La série alternative est ambiguë au
changement d'heure du 29 octobre 2023 et n'est pas utilisée.

La préparation utilise désormais les prix canoniques disponibles à la coupure
de la livraison préparée. Elle reconstruit ses calculs internes en excluant,
pour chacun, tous les prix de sa journée et des journées suivantes. Les reçus
portent `target_history_policy: current_fit_origin_reconstruction_v1` et la
date réelle de référence Saturn dans `target_revision_utc`. Ils déclarent
explicitement que les anciennes versions quotidiennes ne sont pas certifiées.

Après la mise à jour Git, reprendre sur le poste ayant accès à Saturn :

```powershell
.\NYXAnnualCPU.ps1 -Action prepare -DeliveryDay '2026-09-30'
```

La collecte vérifie les quatre historiques de prix avant de reprendre les
profils Saturn. Les nouveaux caches `profiles_v2` et `targets_current_fit_v1`
sont créés sous `data/pit/nyx_annual_saturn` ; aucun ancien cache n'est supprimé.
La première collecte et les premiers calculs CPU peuvent être longs. Garder
PowerShell ouvert et le poste connecté. Une relance réutilise les éléments
valides. Les calculs déjà produits sont réutilisés seulement lorsque leurs
entrées et leurs dépendances numériques sont identiques.

`PREPARED` confirme la préparation, sans activer la production. Ce régime doit
être évalué sur la chaîne CPU complète : chaque journée extérieure évaluée
doit utiliser sa propre date de référence et respecter les contrôles des
autres sources. Les courbes internes reconstruites ne constituent pas à elles
seules des prévisions historiques qualifiées. L'activation refuse une politique
de prix différente de celle évaluée. Les meilleurs scores historiques restent
conservés dans leurs rapports d'origine.

Pour lire la provenance dans les fichiers de diagnostic :

- `training_snapshot_max_retrieved_at_utc` indique la dernière récupération
  effective parmi les historiques utilisés.
- `public_history_before_forecast_cutoff` indique si cette récupération respecte
  la coupure de la prévision concernée. La valeur `false` empêche de qualifier
  ce lot pour cette coupure.
- `current_fit_snapshot_v1` identifie la politique qui conserve les observations
  historiques disponibles pour l'entraînement courant, avec leurs dates réelles.

### Profils Saturn historiques vides

Le diagnostic du 29 septembre identifie aussi un profil de charge résiduelle FR
vide pour le **16 août 2024**, demandé à la coupure du 15 août à 08:00 Paris.
Les archives personnelles ne contiennent pas non plus ce profil pour les 16 et
17 août. Cela ne démontre pas sa disponibilité dans une version Saturn ultérieure.

Le diagnostic suivant, `prepare_20260929T111057Z_7d6550fc`, identifie un autre
cas : la charge résiduelle NL du **17 août 2024** est vide même dans la version
du 29 septembre 2026 à 06:00 UTC. La première récupération remplaçait les
14 séries de la journée dès qu'une seule manquait, sans tester les autres
à leur origine historique.

La collecte traite maintenant **chaque série complète séparément**. Elle
réessaie trois fois son origine historique, puis demande seulement les séries
manquantes à la coupure de la livraison préparée. Chaque succès est conservé
pour la reprise. Toutes les heures d'une série proviennent d'une même version ;
aucune journée n'est complétée en mélangeant des versions heure par heure.

Pour la charge résiduelle NL, un dernier secours vérifié utilise l'archive
déjà suivie dans Git : `data/pit/vintages/nl_residual_load_fcst.parquet`.
Elle contient les 24 heures des 16 et 17 août 2024 dans une version commune
du **5 août 2026 à 12:17 UTC**, récupérée le 10 août 2026. Le fichier est admis
uniquement si son empreinte correspond à celle auditée. La sélection exige
une version complète unique, avec dates de snapshot, révision et récupération
antérieures ou égales à la coupure du calcul. La pièce source est conservée
avec le paquet pour permettre une vérification indépendante.

Ce secours est réservé à l'historique d'entraînement. Il ne certifie pas une
disponibilité en août 2024 et ne peut pas être utilisé pour une évaluation dont
la coupure précède ses dates de disponibilité. Le profil de la journée
réellement prévue reste obligatoire à sa propre coupure Saturn. Si aucune
source admissible n'est complète, le calcul indique précisément la série
manquante et reste bloqué.

Les journées strictes déjà complètes sont réutilisées, y compris pour les
livraisons suivantes. Les profils récupérés
sont conservés séparément dans
`data/pit/nyx_annual_saturn/profiles_per_series_v2/<livraison>/<jour_historique>`.
Les anciens paquets complets restent lisibles avec leur politique d'origine.
La copie de secours NL est figée séparément sous
`data/pit/nyx_annual_saturn/repository_vintages/<empreinte>.parquet` avant
d'être jointe aux paquets ; une mise à jour du fichier d'origine ne modifie
donc pas les preuves déjà constituées.
Le lot enregistre les révisions réelles **par série** dans `profile_revisions.parquet` ;
`origins.parquet` contient les origines logiques des calculs internes.
La politique `own_origin_with_per_series_recovery_v2` et son plafond
`profile_revision_ceiling_utc` suivent les modèles et l'évaluation.
`profile_origin_snapshot_verified: false` indique explicitement que cette
recette ne certifie pas toutes les anciennes versions quotidiennes.

Après la mise à jour de la branche, reprendre avec la même commande `prepare`
ci-dessus. Une relance conserve les journées complètes. La qualification doit
évaluer cette politique sur la chaîne CPU complète, avec la coupure propre à
chaque livraison évaluée. Une qualification portant sur une autre politique
de profils ne peut pas activer celle-ci, y compris la précédente récupération
de journées entières `own_origin_with_current_fit_recovery_v1`.

### Heure manquante dans le vent NL au printemps

Le diagnostic `prepare_20260929T114549Z_5b609248` s'arrête sur
`nl_wind_generation_fcst` au **30 mars 2025 à 02:00 UTC**, soit 04:00 aux
Pays-Bas. Cette heure physique existe dans la journée de 23 heures. Les
nouveaux essais à la coupure actuelle ne la rétablissent pas.

La recette historique `solar_wind_v1`, utilisée par les rapports annuels,
traite déjà ce trou précis avec la politique
`nl_ecmwf_spring_2025_2026`. Ce traitement avait été omis dans le collecteur
annuel CPU. Il est maintenant raccordé avec les mêmes limites :

- Seulement `nl_wind_generation_fcst`, à **02:00 UTC le 30 mars 2025 ou le
  29 mars 2026**, et seulement si cette heure est la seule absente.
- Une valeur native finie reste prioritaire.
- La composante Saturn
  `power.nrjscan.nl.prod.total.wind.mw.ecmwf_avg.pointconnect.6h.cache`
  est demandée à la **même coupure propre J-1 08:00 locale**, soit 07:00 UTC
  pour ces deux journées ; la conversion est MW × 0,001 vers GW.
- Aucun recours à cette substitution lors d'une demande à la coupure
  actuelle du réentraînement. Aucun autre trou n'est interpolé ou rempli.
- La série composante, l'heure, la coupure, les unités et la valeur sont
  conservées dans les preuves de substitution et revérifiées à la reprise.

Les tests rejouent les deux journées avec les valeurs des archives suivies
dans Git : la courbe reconstituée retrouve exactement leurs 23 valeurs.
Ils simulent les réponses Saturn ; la collecte réelle doit être relancée
sur le poste ayant accès au service. Une composante absente ou invalide
reste bloquante.

Après mise à jour de la branche, relancer la même commande `prepare` sans
supprimer les caches. Les journées et séries déjà validées sont conservées.
Ce raccordement ne vaut pas qualification CPU et ne modifie pas les scores
historiques. Le nouveau code de collecte est lié aux empreintes de la
qualification complète.

## Ce qui reste nécessaire avant une prévision qualifiée

La récupération des historiques permet de préparer un entraînement futur.
Elle ne prouve pas, à elle seule, les versions qui étaient disponibles lors
de chacune des anciennes prévisions des rapports annuels. La chaîne complète
doit être évaluée avec des données dont la disponibilité à chaque coupure est
vérifiée. FR, BE et NL conservent leurs critères contre Storm ; DE conserve
l'exception de performance acceptée, avec les mêmes contrôles sur les données.

Voir [le guide principal](nyx_annual_production_cpu.md) pour l'évaluation,
l'activation et l'automatisation des prévisions après qualification.
