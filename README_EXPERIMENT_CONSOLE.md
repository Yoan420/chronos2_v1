# NYX — Prévisions et résultats locaux

Application fonctionnelle en français, dans une direction cyberpunk noir : fond
presque noir, accents ambre, cyan et magenta, schéma interactif et résultats au
premier plan. La navigation relie l'architecture du modèle, les résultats et les
publications par date. La page Résultats permet de lancer la prévision complète.
Aucun service cloud, Node, Docker ou
compte n'est nécessaire pour l'utiliser.

**NYX** est le nom de l'application et du système de prévision présenté.
Le moteur Chronos-2 conserve son nom scientifique dans l'architecture, les
configurations et les commandes existantes.

## Ouvrir NYX depuis le Bureau

Double-cliquer sur l'icône **NYX** du Bureau. Ce raccourci est déjà installé
sur le Bureau, dans le menu Démarrer et à la racine du dépôt. Le Bureau
Windows de ce compte est `C:\Users\BQ6757\OneDrive - ENGIE\Desktop`.
NYX ouvre sa propre fenêtre, sans barre d'adresse ni fenêtre PowerShell.
Aucune commande n'est nécessaire au quotidien. Pour un accès supplémentaire,
vous pouvez épingler le raccourci à la barre des tâches depuis son menu contextuel.

Le raccourci appelle directement le `pythonw.exe` de l'environnement configuré
avec `NYX.pyw` ; il ne dépend pas de l'association Windows des fichiers Python.
Le lanceur démarre le backend local si nécessaire, attend qu'il réponde, puis
ouvre une fenêtre d'application Edge, ou Chrome si Edge n'est pas disponible.
Son profil dédié se trouve dans `runs/.experiment_console/desktop_browser`.

Fermer la fenêtre laisse le backend et les calculs actifs. Un nouveau double clic
rouvre NYX et redémarre le backend s'il s'est arrêté. Un backend déjà présent
n'est réutilisé qu'après vérification du dépôt, du dossier d'état et du Python
déclarés par son API locale. Un verrou sérialise les démarrages simultanés.
Si le service démarre encore et tarde à répondre, le lanceur attend dans la limite
du délai prévu sans créer un deuxième backend concurrent.
En cas d'échec, une boîte de dialogue indique le problème ; les diagnostics de
démarrage sont conservés dans `runs/.experiment_console/desktop_startup.log`.

Cette installation utilise le dépôt existant, son environnement Python et
Edge ou Chrome. Elle ne crée ni service Windows ni démarrage automatique à
l'ouverture de session et ne nécessite pas de droits administrateur. Le
raccourci reste lié à ces chemins : ce n'est pas un exécutable portable autonome.

## Démarrage technique et maintenance (facultatif)

Le lanceur PowerShell reste disponible pour un démarrage explicite du serveur :

Depuis n'importe quel dossier :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Start-ExperimentConsole.ps1'
```

La commande utilise l'interpréteur explicite de
`config/experiment_console.json`, actuellement celui de `Forecast.ps1` :
`C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe`.
Elle ouvre le navigateur après la liaison du serveur à
**http://127.0.0.1:8765**. La fenêtre PowerShell héberge le serveur.

`Install-ChronosDesktop.ps1` conserve son nom technique et crée les trois
raccourcis **NYX**. La migration a été exécutée sur ce poste : les trois liens
NYX sont présents et les trois anciens liens Chronos reconnus ont été retirés.
Le script migre les anciens `Chronos.lnk` uniquement lorsque leur
cible Python et leurs arguments correspondent à l'installation précédente.
Un raccourci non reconnu est conservé ; une destination NYX déjà utilisée par
un autre raccourci bloque la migration avant toute modification. `Chronos.pyw`
reste un point d'entrée compatible pour un ancien lien conservé.
Ce script sert à installer ou recréer les liens après une maintenance ou un
changement de chemins ; il n'est pas nécessaire pour ouvrir l'application
chaque jour :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Install-ChronosDesktop.ps1'
```

Si les dépendances de la console manquent :

```powershell
Set-Location -LiteralPath 'C:\Users\BQ6757\chronos2_v1'
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pip install -r requirements_console.txt
& '.\Start-ExperimentConsole.ps1'
```

Sur ce poste, toutes ces dépendances étaient déjà présentes ; aucune installation
n'a été nécessaire. Lancer les pipelines scientifiques exige aussi leur
environnement et leurs données habituels : `requirements_console.txt` ne les
remplace pas. L'interface n'utilise jamais un Python résolu implicitement dans PATH.

Options du lanceur :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Start-ExperimentConsole.ps1' -NoOpen
& 'C:\Users\BQ6757\chronos2_v1\Start-ExperimentConsole.ps1' -Port 8766
& 'C:\Users\BQ6757\chronos2_v1\Start-ExperimentConsole.ps1' -Settings 'C:\Users\BQ6757\chronos2_v1\config\experiment_console.json'
```

Pour modifier Python, le port ou la concurrence, éditer le fichier JSON puis
redémarrer le backend. Fermer seulement la fenêtre NYX ne l'arrête pas.
Après changement de l'interpréteur, recréer aussi les raccourcis avec le script
d'installation. `max_concurrency` vaut **1** par défaut (1 à 8).
`state_root` doit rester dans le dépôt pour les adaptateurs actuels ; la valeur
par défaut est `runs/.experiment_console`.

## Premier parcours

1. L'accueil **Architecture du modèle** (`#architecture`) s'ouvre par défaut.
   Cliquer sur un module pour consulter son rôle, ou sur **Le bloc Transformer**
   pour explorer les opérations internes. **Explorer mes résultats** ouvre le
   tableau de bord centré sur les résultats disponibles.
2. Dans **Résultats**, cliquer sur **Lancer la prévision**. Le backend exécute
   le véritable `NuclearKalman.ps1`, avec ses réglages scientifiques par défaut.
   Seul `-NoOpen` est ajouté pour consulter les rapports dans NYX.
3. Le calcul actualise les sources communes, traite **BE, DE, FR et NL**, puis
   publie les rapports pays et CWE. La livraison est le lendemain à Paris,
   résolu par le script au démarrage ; le calcul utilise `auto`, 4 threads et
   4 workers. L'attribution coûteuse reste désactivée comme dans la commande.
4. Le statut du calcul reste visible, même si un rapport antérieur est disponible.
   Fermer la fenêtre ne l'arrête pas. Une nouvelle demande pendant un lancement
   actif retrouve ce lancement au lieu d'en démarrer un second.
   La barre suit les six étapes réelles du lot : sources, quatre pays, puis CWE.
   Elle compte les étapes traitées, y compris celles échouées ; le statut final
   indique séparément la réussite. Avant réception du statut du lot, une animation
   d'attente remplace le pourcentage. La cause d'un échec est résumée sans logs.
5. Choisir une livraison pour consulter le **rapport global CWE**, ouvrir les
   rapports par pays ou télécharger leurs fichiers HTML/CSV. **Publications**
   rassemble uniquement ces résultats par date.
   La date dorée dans le titre du rapport CWE est également un sélecteur :
   elle charge directement la livraison choisie et ses résultats par pays,
   y compris depuis le rapport agrandi. Seules les dates publiées sont proposées.

Le script conserve ses synchronisations, caches, résultats figés et sorties
canoniques : une relance peut réutiliser des prévisions déjà calculées, exactement
comme la commande. La console ne force ni nouveau calcul ni lecture exclusive
des caches. Le code de sortie du batch décide de son statut : un rapport existant
ou partiel ne transforme pas un échec en succès.

Les publications suivies sont exclusivement :

- `runs/reports/model_storm/CWE_Model_Storm_YYYY-MM-DD.html` ;
- `runs/exports/YYYY-MM-DD/{be,de,fr,nl}/nuclear_kalman/forecast_<pays>_<date>_nuclear_kalman.{html,csv}`.

La mention **Publication vérifiée** atteste l'identité du manifeste nucléaire et
l'intégrité des fichiers, pas la qualité scientifique ni la réussite du dernier
batch. Les publications partielles restent signalées. Les expériences, variantes
concurrentes, caches et journaux ne figurent pas dans ces vues. Leur historique
existant est conservé. L'index des publications ne dépend pas de la base des runs.

Les pages Résultats et Publications partagent un fond animé discret, avec des
flux ambre, cyan et magenta qui circulent dans les deux sens. L'aperçu
CWE utilise la palette NYX et un fond transparent ; ses données et les fichiers
publiés restent inchangés par cette adaptation visuelle. Les animations de fond
s'arrêtent lorsque la page est masquée et respectent la réduction des mouvements.

Un environnement de démarrage ayant hérité du proxy bloquant de l'outil de
développement est détecté avant calcul. Aucun proxy utilisateur n'est supprimé
ou remplacé : le service doit être démarré dans l'environnement Windows normal.

Les journaux ont été retirés de l'interface : aucun onglet, recherche, filtre ou
téléchargement de logs n'est proposé, et les fichiers de journal sont masqués dans
les listes d'artefacts. Le backend conserve `console.log` sur disque pour le
diagnostic ; cette modification de présentation ne supprime pas la traçabilité.

## Comprendre l'architecture affichée

L'en-tête compact **Architecture du modèle** présente **NYX**, le système
composite de prévision. Le badge **Moteur Transformer** identifie son composant
**Chronos-2**. Le schéma explique le parcours **Nuclear Kalman** configuré dans le
dépôt : séries et covariables → contrôle point-in-time → **Chronos-2** →
**correcteur résiduel CatBoost** → **Kalman nucléaire gouverné** → prévisions →
évaluation et rapports. Les contextes complémentaires alimentent les corrections
comme dans le code existant. **Storm et les observations de la livraison** forment
une branche indépendante qui rejoint l'évaluation ; ils ne sont pas représentés
comme des entrées de la chaîne de prévision.

Les détails viennent des configurations et du code local. La configuration
actuelle conserve **LoRA inactif** sur ce parcours ; une configuration absente,
inconnue ou incompatible est signalée. L'explication ne charge ni poids du modèle,
ni runtime de calcul scientifique. Le bouton de prévision utilise séparément le
lanceur nucléaire existant, sans modifier la chaîne décrite par le schéma.

Le zoom sur le Transformer présente un bloc conceptuel : **patches et projection**,
**attention temporelle**, **attention de groupe**, **feed-forward**, puis
**projection en quantiles**. Ces opérations sont vérifiées dans les sources du
paquet Chronos installé. Aucun nombre de couches, de têtes ou de dimensions n'est
inventé ; les formes et flux animés ne représentent pas des poids d'attention
mesurés. Les informations non vérifiables restent indiquées comme telles.
**Suspendre le flux** arrête uniquement l'animation du schéma, jamais un calcul.

Dans la vue détaillée, un réseau de **27 points illustratifs** réagit au survol
et au clic : des impulsions se propagent entre ses groupes. Le clavier permet
aussi de stimuler la cible avec **Entrée** ou **Espace**. Ce dessin n'indique
aucun nombre réel de neurones ou de couches du modèle. L'animation respecte la
réduction des mouvements, s'arrête lorsque la vue n'est plus visible et est
libérée en quittant cette page.

## Adaptateurs techniques conservés dans le backend

Ces quatre adaptateurs historiques ne sont plus proposés par l'interface NYX.
Les précautions de copie et de cache décrites ci-dessous concernent ces
adaptateurs ; elles ne changent pas le fonctionnement de `NuclearKalman.ps1`.

| Type proposé | Point d'entrée existant | Sortie isolée |
|---|---|---|
| Rapport Model / Storm | `run_model_storm_report.py` | `--output <nouveau run>/outputs/model_storm.html` |
| Rapport horaire | `generate_hourly_html_report.py` | `--output <nouveau run>/outputs/report.html` |
| Évaluation horaire appariée | `evaluate_hourly_backtest.py` | `--output-dir <nouveau run>/outputs` |
| Prévision / backtest horaire | `run_chronos2_hourly.py` | `--config <snapshot> --output-dir <nouveau run>/outputs` |

Le catalogue vient des scripts et configurations réellement présents. Cinq YAML
respectaient le contrat horaire lors de la validation. Les configurations de
l'orchestration live multi-pays suivent un autre contrat et ne sont pas proposées
comme YAML horaires. Le modèle proposé correspond au `report.native_model` du
YAML, et le moteur exécute les composants de cette configuration.

Pour la prévision horaire, les références relatives des fichiers/PIT sont rendues
absolues dans la copie, les sorties sont isolées et la lecture des caches est
imposée (`data.source=cache`, sources auto/Saturn vers cache,
`model.local_files_only=true`). Aucune synchronisation Saturn n'est déclenchée.
Un cache incomplet ou un modèle absent peut faire échouer ce run. Ces adaptations
portent sur les entrées/sorties ; les paramètres numériques, métriques, règles PIT
et découpages d'évaluation du moteur ne sont pas réécrits.

Les données et caches scientifiques existants peuvent encore être modifiés par
une commande PowerShell extérieure. Les runs horaires de la console réservent
la ressource `scientific-cache`, même si la concurrence globale dépasse 1.
L'interface expose cette limite dans le récapitulatif ; elle ne prétend pas
verrouiller un processus externe qui ne participe pas à ce mécanisme.

## Importer les anciennes exécutions

Depuis **Résultats** ou **Historique**, choisir **Importer des runs**, puis `runs/`,
`output/` ou un de leurs sous-dossiers. Les imports externes au dépôt sont refusés
dans cette première version. L'import s'effectue en arrière-plan ; sur un grand
arbre, prévoir plusieurs dizaines de secondes.

Les marqueurs incluent `run_manifest.json`, `experiment_manifest.json`,
`evaluation_manifest.json`, `metrics_hourly.json`, `evaluation_summary.json`,
`status.json`, `run_status.json` et les formats de prévisions reconnus. Les
alias `latest_status.json` ne créent pas un deuxième run. L'identité persistante
dépend du chemin canonique : réimporter actualise la même entrée, sans doublon.

Aucun dossier n'est déplacé, recalculé ou écrasé. Les fichiers incomplets et les
formats non reconnus sont signalés dans le bilan ; ils ne font pas échouer les
autres dossiers. Les liens symboliques/jonctions, caches, entrées, checkpoints
et métadonnées propres à la console sont exclus de l'exploration.

La présence de résultats ne prouve pas qu'un run a réussi. Une fin historique
non attestée reste **Inconnu**. Une référence à un YAML actuel n'est pas présentée
comme la configuration historique exacte. Le `run_status.json` nucléaire atteste
parfois seulement une étape auxiliaire ; `complete` n'est alors pas transformé
en preuve de réussite de toute la publication.

Les processus externes restent en **lecture seule**. Le détail affiche leur phase
déclarée et leurs résultats ; la liste actualise leurs métadonnées. Leurs journaux
restent destinés au diagnostic sur disque, sans vue dédiée dans l'interface. Aucun
contrôle ni rattachement à leur PID n'est proposé. L'ancien état déclaré peut être
obsolète : le fichier n'est pas une preuve de vie du processus.

## Dupliquer, annoter et comparer

**Dupliquer ce run** ouvre un formulaire ; cela ne lance rien. Pour un run horaire
géré, la copie scientifique conservée en base est réutilisée même si le YAML
source a changé. Un nouvel identifiant et de nouvelles sorties sont attribués.
Sans configuration historique effective, la duplication prépare explicitement
un rapport depuis les résultats, plutôt que de prétendre reconstruire le calcul.
Un format non compatible est refusé à la validation.

**Notes et tags** enregistre les annotations uniquement dans SQLite, y compris
pour les archives. L'historique filtre nom/identifiant/note, dates, statut, type,
modèle, tag et valeurs de métriques ; il se trie et se parcourt par pages.

Cocher **2 à 8 runs** dans l'historique puis ouvrir **Comparaison**. La console
affiche durées, métriques, différences de configuration et rapports/courbes.
Elle vérifie période, cible, zone, horizon, fréquence, couverture et protocole
lorsqu'ils sont documentés. Informations manquantes ou divergentes : avertissement,
sans classement. Les métriques finales restent celles exportées par le moteur.

Les courbes CSV/GZIP identifient les colonnes observées explicitement, jamais
`price_eur_mwh` dans un forecast live où elle peut recopier q50. Les erreurs sont
prévision moins observation sur la même ligne. Lorsqu'une fenêtre UTC d'évaluation
est documentée, l'aperçu backtest est filtré sur cette fenêtre. Les aperçus sont
bornés et signalent leur troncature ; aucune métrique finale n'est recalculée
sur cet extrait.

## Exécution, redémarrage et sécurité locale

Les états sont `queued`, `starting`, `running`, `succeeded`, `failed`,
`cancelling`, `cancelled`, `interrupted`, `unknown`. Les boutons et libellés sont
traduits. Une activité indéterminée remplace le pourcentage lorsque le moteur
ne publie pas de progression mesurable. Il n'y a pas de pause/reprise simulée.

- La fenêtre NYX ne possède pas les processus. La fermer laisse le backend
  et les calculs actifs ; rouvrir le raccourci réutilise le backend vérifié.
- Un superviseur Python détaché persiste logs, identité PID/date de création,
  activité, dates et code de retour. Il reste actif si le backend est fermé.
- Sous Windows, le calcul démarre suspendu, est assigné à un **Job Object**,
  puis reprend. L'arrêt ou la mort du superviseur nettoie les enfants de ce Job.
- **Demander l'arrêt** nécessite une confirmation dans l'interface. Les résultats
  déjà écrits restent sur disque ; ils peuvent être partiels.
- Fermer le serveur avec Ctrl+C laisse continuer les calculs déjà démarrés. Les
  runs en attente reprennent au prochain démarrage du backend.
- La réconciliation vérifie l'identité du superviseur : un PID réutilisé n'est
  pas considéré comme le bon processus. Un superviseur disparu donne
  **Interrompu**, avec code de retour et heure exacte de fin non disponibles.
  Si son identité est inaccessible, ses ressources restent réservées avec un
  avertissement ; aucune nouvelle exécution conflictuelle n'est lancée.
- Un verrou empêche deux backends de gérer la même base. Les plans et clés
  d'idempotence empêchent le double clic ou la répétition d'une requête de créer
  deux calculs. Les transitions de réconciliation sont transactionnelles.
- Le backend réserve son port avant de démarrer l'ordonnanceur. Un port occupé
  ne peut donc pas lancer des runs en attente depuis un serveur qui a échoué
  à démarrer ; les ressources de ce démarrage sont libérées.

Les commandes sont des listes d'arguments (`shell=False`), limitées aux quatre
adaptateurs ; aucun terminal arbitraire n'est exposé. Un récapitulatif devenu
obsolète après modification d'une source est refusé avant lancement.

La console écoute uniquement `127.0.0.1`. Elle vérifie Host/Origin et un nonce
CSRF pour les mutations. Cette configuration n'est pas un service Internet et
ne comporte pas d'authentification multi-utilisateur. Les rapports HTML sont
affichés dans une iframe sandboxée sans accès à l'API, avec ressources externes
bloquées. Certains rapports dépendant d'un CDN peuvent donc être incomplets.
Le script Plotly intégré est préservé seulement s'il correspond exactement à la
bibliothèque installée, avec sa sérialisation LF ou Windows CRLF. Les blocs JSON
des rapports sont assainis ; un autre script dont le masquage modifierait le code
est retiré avec un avertissement visible. Le rapport original sur disque reste
inchangé.

Les noms de champs sensibles, arguments de secrets, valeurs d'environnement
sensibles, URL avec identifiants et blocs de clés privées sont masqués. Les
configurations exécutables contenant des secrets reconnus sont refusées, plutôt
que modifiées silencieusement. Les exports JSON/YAML/CSV/GZIP sont assainis.
Ce masquage ne peut pas reconnaître une valeur confidentielle arbitraire sans
contexte : les pipelines doivent continuer à éviter l'impression de secrets et
à recevoir leurs identifiants via les mécanismes d'environnement existants.

## Organisation du code

| Fichier | Responsabilité |
|---|---|
| `NYX.pyw` | Entrée sans console et boîte de dialogue en cas d'erreur |
| `Chronos.pyw` | Compatibilité avec les raccourcis de l'ancien nom |
| `experiment_console/desktop.py` | Démarrage local vérifié et fenêtre Edge/Chrome dédiée |
| `Install-ChronosDesktop.ps1` | Installation des raccourcis Bureau, menu Démarrer et dépôt |
| `experiment_console/static/` | Interface HTML/CSS/JS, sans compilation ; Plotly servi localement |
| `experiment_console/server.py` | Serveur HTTP local, API, exports contrôlés |
| `experiment_console/manager.py` | Plans, queue, duplication, annotations et imports persistants |
| `experiment_console/worker.py` | Supervision détachée, journalisation et fin des processus |
| `experiment_console/processes.py` | Identités et Job Objects Windows |
| `experiment_console/store.py` | Transactions SQLite/WAL |
| `experiment_console/adapters.py` | Validation et adaptation des quatre CLI existants |
| `experiment_console/artifacts.py` | Lecture seule des formats historiques et comparabilité |
| `experiment_console/architecture.py` | Lecture des preuves locales et description du schéma interactif |
| `experiment_console/security.py` | Masquage partagé |

Le serveur HTTP de la bibliothèque standard évite une nouvelle dépendance
FastAPI/Node ; SQLite suffit à cette console locale. `psutil`, `filelock`, PyYAML
et Plotly sont déjà utilisés dans l'environnement. L'ancienne interface Streamlit
reste intacte ; sa file en session ne pouvait pas fournir à elle seule les
garanties de durée de vie demandées. Aucun moteur scientifique n'a été modifié.

Métadonnées : `runs/.experiment_console/console.sqlite3`. Pour chaque nouveau run :
`executions/<id>/config.yaml`, `metadata.json`, `console.log`, `outputs/`.
Le JSON de métadonnées décrit le lancement ; les états ultérieurs sont dans SQLite.
Sauvegarder l'ensemble du dossier d'état avec la base, son WAL éventuel et les
sorties, de préférence après arrêt du backend et des calculs. Les gros datasets
et modèles ne sont pas recopiés.

## Validation réalisée

Sur **Windows**, avec Python **3.11.9 / pricefm311**, les tests de la console ont
été exécutés avec de minuscules scripts de fixture explicitement nommés, sans
entraînement ni backtest complet. La qualification des six modules initiaux a
donné **107 tests réussis en 112,18 secondes**, code de sortie **0**, aucun test
ignoré. Après adaptation du démarrage sans console, les suites démarrage,
exécution et API ont été rejouées ensemble : **42 tests réussis en 83,08 secondes**,
code de sortie **0**, dont **5 nouveaux tests de démarrage**.
La suite du lanceur Bureau a ensuite donné **29 tests réussis en 9,36 secondes**,
code de sortie **0**. Cette étape ajoute donc deux exécutions de test totalisant
**71 réussites** ; elles sont distinctes de la qualification initiale.

```powershell
Set-Location -LiteralPath 'C:\Users\BQ6757\chronos2_v1'
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pytest tests\test_console_execution.py tests\test_console_adapters.py tests\test_console_artifacts.py tests\test_console_api.py tests\test_console_security.py tests\test_console_architecture.py -q
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pytest tests\test_console_startup.py tests\test_console_execution.py tests\test_console_api.py -q
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pytest tests\test_console_desktop.py -q
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pytest tests\test_console_architecture.py -q
```

Couverture : réussite/échec, persistance, annulation en file et arbre Windows,
concurrence/ressources, doubles clics, redémarrage avec worker vivant, disparition
du superviseur, course de réconciliation, PID incertain, redaction, traversée de
chemins, Host/Origin/CSRF, imports répétés, fichiers manquants/incomplets et
duplication fidèle. Les tests du schéma vérifient aussi la séparation des
références, les opérations du Transformer et la lecture sans exécuter le modèle.
Les régressions de démarrage couvrent le port occupé avant ordonnancement,
le démarrage idempotent, la libération des verrous et l'absence de sortie standard
sous `pythonw.exe`. La suite Bureau vérifie aussi l'identité du service local,
les réponses HTTP temporairement lentes et l'absence de démarrage concurrent.
Les détails chiffrés et le parcours de navigateur sont dans
`CONSOLE_VALIDATION.md`.

Le raccourci du Bureau a été ouvert par un **double clic réel dans Windows** :
fenêtre alors intitulée **Chronos · Expériences**, icône ambre dédiée, sans barre d'adresse ni
terminal. La réutilisation du backend existant et un **démarrage à froid** après
arrêt contrôlé ont été vérifiés. Les **267 entrées de l'historique** sont restées
présentes et aucun run supplémentaire n'a été lancé pendant ce contrôle. Cette
preuve précède le renommage de l'application en NYX.
Après renommage, les **29 tests du lanceur Bureau** ont été rejoués avec succès
en **9,86 secondes**, et les **5 tests d'architecture** en **3,26 secondes**
(codes `0`). La compilation des deux entrées Python et du lanceur, ainsi que
l'analyse syntaxique de l'installateur PowerShell, ont réussi. Les trois
raccourcis NYX ont été installés et une fenêtre Windows portant le nom **NYX**
a été vérifiée. Le schéma interactif a été contrôlé sur écran de bureau et à
**390 px**, sans débordement horizontal ni erreur JavaScript.

Un vrai **rapport Model / Storm du 11 septembre 2026** a également été lancé
depuis l'interface, terminé avec un code `0` en environ **11 secondes**, puis
consulté dans son historique et son aperçu HTML. Il utilise les résultats
locaux existants et porte le nom « Validation console · rapport local du
11 septembre ». Ce n'est pas une donnée de démonstration ni une prévision recalculée.

## Limites de cette première version

- Quatre familles intégrées. Les entraînements LoRA et l'orchestration de production
  multi-pays/nucléaire restent lancés par leurs commandes existantes : leurs effets
  sur caches, bundles promus et exports partagés nécessitent des adaptateurs dédiés.
  Leurs résultats et statuts reconnus peuvent être importés en lecture seule.
- Les lancements scientifiques lourds n'ont pas été exécutés pendant cette
  intégration. Il reste à les valider sur les caches/modèles requis, par votre
  action explicite. L'interface ne télécharge pas les données manquantes.
- Pas de suivi fiable d'un PID PowerShell extérieur, de pause/reprise du calcul,
  de reprise automatique d'un run interrompu, ni de métrique intermédiaire
  lorsque le pipeline ne l'exporte pas. Les journaux restent conservés sur disque
  pour le diagnostic et ne sont pas exposés dans l'interface.
- Import limité à 2 000 candidats, profondeur 12 ; exploration des artefacts
  limitée à 500 fichiers et deux niveaux par run, hors sous-runs identifiés.
  Les rapports parents n'agrègent pas automatiquement tous les sous-runs.
- Exports web limités à 64 Mo ; au-delà, utiliser le chemin de sortie affiché.
  Les formats binaires de modèles, les entrées sensibles et les journaux ne sont
  pas proposés au téléchargement dans l'interface.
- Pas de suppression depuis l'interface. Le classement des runs dont le périmètre
  diffère ou reste incomplet n'est jamais présenté comme une compétition valide.
