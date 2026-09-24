# Validation de NYX — 11 septembre 2026

Environnement réel : Windows, Python 3.11.9 dans
`C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe`.
La validation des processus n'a pas été réalisée uniquement sous Linux.

## Réparation et validation réelle du lancement — 11 septembre

L'exécution `1ab81961fcfa427f9e080541d836416c` avait échoué pour les quatre pays
lors de l'actualisation Saturn : le backend avait hérité du proxy de l'outil
vers `127.0.0.1:9`. Les paramètres scientifiques étaient corrects. Le service
identifié a été redémarré hors de cet environnement restreint, après vérification
qu'aucun calcul n'était actif. Aucun paramètre proxy utilisateur ou machine n'a
été modifié.

La relance réelle `d48df29243834fa4af3f8490d9694ee3` a terminé avec **code 0**,
le 11 septembre à **13:07:38 UTC**, en **153,719 secondes**. Les **6 étapes**
sont complètes : sources, BE, DE, FR, NL et CWE. Les quatre publications du
12 septembre sont à nouveau vérifiées après actualisation. Le pipeline a
réutilisé les prévisions figées, conformément à ses réglages, et actualisé les
observations, références et rapports. L'ancien échec reste dans l'historique.

La progression provient du fichier de statut du lot exact, lié à son propre
journal avec vérification de chemin, identité et horaires. L'API réelle a été
observée à **1/6 (17 %)**, **5/6 (83 %)** puis **6/6 (100 %, réussi)**. Les erreurs
affichent désormais leur cause au lieu du simple chemin de diagnostic.

Validation ciblée : 32 tests environnement, 25 tests progression (2 contrôles de
liens symboliques ignorés faute de droits Windows, protections reparse testées
séparément), 21 tests intégrés erreurs/worker/environnement, 24 régressions
exécution/lancement et 32 tests publications réussis. Les contrats JavaScript
passent aussi pour les barres déterminée/indéterminée et un lot terminé en échec.
Ces ensembles se recoupent ; aucun total de cas uniques n'est revendiqué.

L'aperçu CWE servi reprend le fond transparent et la palette NYX ; comparaison
directe des traces JSON avec le fichier source : valeurs identiques. Les fichiers
et le thème du téléchargement restent inchangés par l'aperçu. Le fond animé est
présent sur Résultats et Publications, avec suspension hors visibilité et respect
de la réduction des mouvements. Le contrôle automatique a refusé l'inspection
visuelle CUA, désactivée dans cet environnement ; vérifications effectuées via
les fichiers, les contrats JavaScript et les réponses HTTP du service réel.

## Résultats principaux et lancement NuclearKalman

L'interface courante propose **Accueil**, **Résultats** et **Publications**.
Le bouton **Lancer la prévision** demande le véritable `NuclearKalman.ps1`,
sans argument scientifique ajouté. Seul `-NoOpen` évite l'ouverture externe
du rapport. Le statut durable du processus reste indépendant de la présence
d'un rapport ancien ou partiel. Aucun calcul scientifique n'a été exécuté pour
valider cette intégration.

- **32 tests** des publications réussis en **7,06 s** : sélection canonique,
  dates et pays manquants, exclusion des expériences, manifestes et SHA,
  refus des chemins et jonctions Windows, lectures sans écriture et protection
  des exports HTML/CSV.
- **10 tests** du lancement primaire réussis : 9 en **36,67 s**, puis un contrôle
  du Python configuré en **4,56 s**. La parité des arguments est vérifiée avec
  le véritable PowerShell en mode `-DryRun`. Les exécutions succès/échec utilisent
  uniquement de petites fixtures Python isolées. Six demandes concurrentes
  produisent un seul run ; les clés restent valides après redémarrage et fin.
- Contrats frontend vérifiés avec **Node**, sans navigateur : sélection d'une
  livraison ancienne, rapport absent, statut échoué avec résultat disponible,
  POST vide, nonce, clé de répétition, double clic et échappement du contenu.
  `node --check` valide également les deux fichiers JavaScript.
- **25 régressions** adaptateurs, démarrage et architecture réussies ensemble
  en **9,50 s**.
- La qualification backend finale a rejoué **57 tests** adaptateurs, exécution,
  démarrage et API en **151,38 s**, tous réussis. Ce total recoupe certaines
  régressions précédentes ; il ne s'agit pas de tests supplémentaires distincts.

L'inventaire réel contient **4 dates**, **3 rapports CWE** et **15 publications
pays NYX** (33 fichiers HTML/CSV). Les 15 publications pays passent les contrôles
du manifeste nucléaire et des empreintes. Cela atteste l'intégrité des fichiers,
pas une certification scientifique.

Le service local sur le port **8765** a été redémarré après contrôle d'identité
et absence de calcul actif. Les contrôles HTTP réels confirment le bouton
disponible, BE/DE/FR/NL, synchronisation active, attribution désactivée, les
4 dates, les 3 rapports CWE et les 15 publications pays. Les **271 entrées**
historiques sont conservées. Le JavaScript servi correspond au fichier modifié ;
l'aperçu CWE est livré sous CSP sandbox et sa lecture ne modifie pas son SHA.

Le contrôle natif de la fenêtre Windows a été arrêté par l'outil, qui ne pouvait
pas confirmer son adresse. La nouvelle vue des publications et du bouton de
calcul n'a donc pas fait l'objet d'un nouveau contrôle visuel dans cette étape.
Les captures et contrôles visuels détaillés plus bas concernent les étapes
précédentes de l'interface.

## Tests automatisés

Qualification des six modules initiaux : **107 tests réussis en 112,18 s**, code de
sortie **0**, aucun test ignoré. Répartition :

| Suite | Résultat |
|---|---|
| Gestionnaire, worker et arbre Windows (`test_console_execution.py`) | 14 réussis |
| Adaptateurs CLI (`test_console_adapters.py`) | 15 réussis |
| Archives / prévisions (`test_console_artifacts.py`) | 13 réussis |
| API HTTP, import, duplication et exports (`test_console_api.py`) | 23 réussis |
| Masquage (`test_console_security.py`) | 37 réussis |
| Architecture documentée (`test_console_architecture.py`) | 5 réussis |

Après adaptation du démarrage sans console, les suites démarrage, exécution et
API ont été rejouées ensemble : **42 tests réussis en 83,08 s**, code de sortie
**0**. Elles comprennent les **5 nouveaux tests** de `test_console_startup.py`,
les 14 tests d'exécution et les 23 tests HTTP déjà comptés ci-dessus. Les nouveaux
cas couvrent le démarrage idempotent, le verrou libéré après échec, le port occupé
avant tout ordonnancement et le nettoyage après service réussi ou échoué lorsque
`sys.stdout` est absent, comme sous `pythonw.exe`.

La version finale du lanceur Bureau a été validée séparément : **29 tests réussis
en 9,36 s**, code de sortie **0**, dans `test_console_desktop.py`. Ils incluent
trois régressions HTTP sur un service occupé et la vérification de son identité,
ajoutées après le contrôle du démarrage à froid réel. Les deux exécutions de
validation de cette étape totalisent **71 réussites** (**42 + 29**). Ce total
reste distinct des 107 tests de la qualification initiale, dont certains ont
été rejoués ; aucun total combiné des huit modules n'est revendiqué.

Après renommage de l'application en **NYX**, les **29 tests du lanceur Bureau**
ont été rejoués : **29 réussis en 9,86 s**, code de sortie **0**. Les **5 tests
d'architecture** ont également réussi en **3,26 s**, code de sortie **0**.
Ces deux suites ont été exécutées séparément. La compilation
de `NYX.pyw`, de l'entrée compatible `Chronos.pyw` et de `desktop.py`, ainsi que
l'analyse syntaxique de l'installateur PowerShell ont réussi. Les noms de modules,
les chemins de stockage et le moteur scientifique restent inchangés.

Les assertions portent sur des effets réels et des fixtures explicitement
nommées, jamais sur un entraînement coûteux. Les
tests de diagnostic HTTP restent présents même si les journaux ne sont plus
proposés dans l'interface utilisateur.

```powershell
Set-Location -LiteralPath 'C:\Users\BQ6757\chronos2_v1'
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pytest tests\test_console_execution.py tests\test_console_adapters.py tests\test_console_artifacts.py tests\test_console_api.py tests\test_console_security.py tests\test_console_architecture.py -q
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pytest tests\test_console_startup.py tests\test_console_execution.py tests\test_console_api.py -q
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pytest tests\test_console_desktop.py -q
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pytest tests\test_console_architecture.py -q
```

Vérifications complémentaires : compilation des modules Python, syntaxe de
`app.js` et `architecture.js` avec `node --check`, analyse syntaxique PowerShell
du lanceur et vérification whitespace Git sur l'ajout au README. Node n'est pas
nécessaire au démarrage de l'application.

## Installation Bureau

Le point d'entrée quotidien porte désormais le nom **NYX**, comme le système
de prévision présenté. Le moteur scientifique conserve le nom **Chronos-2**.
Les trois raccourcis **NYX.lnk** sont effectivement installés dans le Bureau réel
de ce compte
(`C:\Users\BQ6757\OneDrive - ENGIE\Desktop`), les programmes du menu Démarrer
de l'utilisateur et la racine du dépôt. Son nom technique reste
`Install-ChronosDesktop.ps1` ; c'est un outil de maintenance qui n'est pas
nécessaire à chaque ouverture.

Chaque raccourci cible directement le `pythonw.exe` voisin de l'interpréteur
configuré et lui transmet le chemin absolu de `NYX.pyw`. L'association Windows
des fichiers Python n'intervient pas et aucune fenêtre PowerShell n'est requise.
Le lanceur ouvre Edge, avec Chrome en repli, en mode application `--app`, sans
barre d'adresse et avec un profil séparé dans
`runs/.experiment_console/desktop_browser`.

La migration prépare les trois liens `NYX.lnk` après vérification de toutes
les destinations. Elle ne retire les anciens `Chronos.lnk` qu'après création
des nouveaux liens et seulement si leur cible Python et leurs arguments exacts
correspondent à l'installation précédente. Les liens non reconnus sont conservés.
`Chronos.pyw` reste compatible et appelle le point d'entrée NYX.
L'installateur a été exécuté avec succès : la présence des trois nouveaux liens
et le retrait des seuls trois anciens liens Chronos reconnus ont été vérifiés.

Le backend local est démarré à la demande. Sa réutilisation vérifie les chemins
du dépôt, du dossier d'état et du Python renvoyés par `/api/bootstrap`, sans
publier le jeton de mutation. Un verrou de fichier sérialise les doubles clics
concurrents. Le serveur lie son port avant de démarrer la file d'exécution et
nettoie ses ressources en cas d'erreur. Un service temporairement occupé est
réessayé jusqu'à l'échéance du démarrage, sans lancer un backend concurrent.
Les erreurs du lanceur sont présentées
dans une boîte de dialogue et le diagnostic du backend est conservé dans
`runs/.experiment_console/desktop_startup.log`.

Fermer la fenêtre laisse le backend et les calculs actifs ; le raccourci permet
de rouvrir l'interface et redémarre le backend si nécessaire. Il est possible
d'épingler le raccourci à la barre des tâches depuis Windows. L'installation ne
crée ni service, ni démarrage automatique Windows, ni exécutable portable ; elle
requiert les chemins existants du dépôt et de l'environnement, ainsi qu'Edge ou
Chrome. Le démarrage PowerShell reste documenté comme option technique.

## Parcours réel depuis le Bureau Windows

Ce contrôle a été effectué avant le changement de nom en NYX ; les libellés
historiques ci-dessous décrivent la fenêtre effectivement observée.

- Sélection puis **double clic sur `Chronos.lnk`** dans le Bureau, avec les
  commandes natives de Windows : ouverture de la fenêtre dédiée
  **Chronos · Expériences**, icône ambre personnalisée, sans barre d'adresse
  ni fenêtre terminal. Le backend existant est réutilisé après vérification.
- Fermeture de la fenêtre, puis arrêt du seul backend d'origine après contrôle
  de l'historique : **267 entrées, aucun run actif**. Les calculs scientifiques
  et leurs fichiers n'ont pas été modifiés pour ce test.
- Nouveau double clic sur l'icône : **démarrage à froid réussi** avec `pythonw`,
  backend local prêt, **267 entrées conservées et toujours aucun run actif**.
  Le démarrage réel a été répété après correction d'une course où le service
  avait réservé son port mais ne répondait pas encore. Le lanceur attend
  désormais ce service dans le délai prévu sans créer de serveur concurrent.
- La dernière ouverture à froid a aussi été contrôlée visuellement : la fenêtre
  Chronos est restée ouverte avec son icône ambre et sans barre d'adresse.

Après la migration, Windows a confirmé le titre **NYX** de la nouvelle fenêtre
d'application. L'API renvoie `model_name=NYX`, conserve l'identifiant scientifique
`nuclear_kalman` et présente **Chronos-2** sous le badge **Moteur Transformer**.

## Parcours réel dans le navigateur

- Accueil vide utilisable, formulaire des quatre familles et affichage du Python.
- Saisie du run **Validation console · rapport local du 11 septembre** ; contrôle
  de la date, de la commande structurée et des sorties dans le récapitulatif.
- Lancement de `run_model_storm_report.py`, livraison `2026-09-11`, à partir des
  données déjà disponibles. Aucune synchronisation ni prévision recalculée.
- Run `6a8a62eb28fd4810aca755d611f12cf6` : statut **Terminé**, code **0**, durée
  affichée **11 s** ; `console.log` conservé côté backend et
  `outputs/model_storm.html` disponible dans les résultats.
- Le rapport présente les résultats locaux BE/DE/FR/NL ; consultation des artefacts.
- Sur la version finale, l'aperçu du rapport affiche les commandes Plotly sans
  nouvelle erreur JavaScript. La fiche propose uniquement **Résultats**,
  **Configuration** et **Notes et tags**, sans onglet de journal.
- **Dupliquer ce run** sur ce rapport réel ouvre le formulaire et conserve date,
  description, configuration et nom avec son suffixe de copie. Cette action
  n'exécute aucun nouveau run.
- Redémarrage réel du backend : le run terminé reste dans l'historique, sans
  second lancement. Les tests automatisés couvrent aussi un calcul encore actif
  au moment du redémarrage et ses enfants.
- Import depuis `runs/` : **266 nouvelles archives**, aucune actualisation à ce
  premier import, soit **267 entrées** avec le rapport de validation. Les formats
  non reconnus sont signalés et ignorés, sans interrompre l'import.
- Recherche `chronos2_hourly` : 11 résultats. Sélection des archives françaises
  `chronos2_hourly_fr_residual_extended_v1` et `chronos2_hourly_fr_mkonline_blend_v1`.
  Comparaison des métriques existantes ; **deux graphiques Plotly** rendus.
  Avertissement explicite « cible non disponible » et limitation des aperçus
  à 2 000 lignes. Aucun run désigné comme meilleur.
- Inspection de l'interface en largeur 1 440 px et à la largeur étroite du panneau
  intégré ; aucun débordement horizontal de page à **630 px** après
  réinitialisation du viewport.
- Depuis l'accueil, passage au bloc Transformer puis sélection de
  **Attention de groupe** : le panneau présente le bon module. Les commandes
  de suspension/reprise changent correctement l'état de l'animation. La page
  finale reste sur l'architecture complète, animation active.
- L'outil de lecture WebMCP `list_experiments` renvoie les **267 entrées** de la
  console, sans mutation.

Le contrôle de l'export du rapport réel, effectué sur les octets UTF-8 sans
normalisation préalable des retours à la ligne, confirme **3 blocs script**, le
code Plotly installé intact, un bloc JSON valide et aucun avertissement de script
retiré. Le SHA du fichier sur disque reste identique. Les régressions HTTP
vérifient séparément le masquage des secrets dans le JSON et le retrait d'un
script sensible, même s'il porte un nom ou attribut prétendument « trusted ».

## Présentation validée avant le recentrage sur les publications

La version courante adopte un **cyberpunk noir**, avec accents **ambre, cyan et
magenta**. Son titre compact est **Architecture du modèle**. L'accueil par défaut
est `#architecture` : modules sélectionnables,
inspection de leur rôle, flux animés et accès au bloc Transformer. La page
**Résultats** met les sorties récentes en avant ; les fiches de run s'ouvrent
directement sur les rapports, métriques et graphiques.

Le schéma décrit le parcours configuré **Chronos-2 → correcteur CatBoost → Kalman
nucléaire gouverné**. La branche **Storm / observations de la livraison** rejoint
uniquement l'évaluation. LoRA est **inactif** dans la configuration vérifiée. Le
zoom décrit patches/projection, attention temporelle, attention de groupe,
feed-forward et quantiles à partir des opérations reconnues dans le paquet
Chronos. Les tests contrôlent l'absence de runtime scientifique importé, de
dimensions inventées et d'exposition de valeurs sensibles. Les animations sont
conceptuelles : elles ne représentent ni couches dénombrées ni poids d'attention
mesurés et ne servent pas d'indicateur de progression d'un run.

La vue détaillée affiche **27 points illustratifs**, distribués en groupes de
**4, 6, 7, 6 et 4**. Ce choix graphique ne décrit ni le nombre de neurones ni les
couches réelles de Chronos-2. Des impulsions en nombre borné se propagent dans
ce réseau ; le survol, le clic et la cible clavier **Entrée/Espace** permettent
de le stimuler. La suspension, la réduction des mouvements et la visibilité de
la vue contrôlent l'animation ; quitter la route libère son rendu.

Le contrôle visuel final couvre le bureau et une largeur de **390 px** : aucun
débordement horizontal, commandes du réseau correctement disposées, aucune
erreur JavaScript. La stimulation au clic et la sélection **Attention de groupe**
fonctionnent. La suspension change la classe visuelle et l'état pressé du bouton.
Après navigation hors de l'architecture, **aucun canvas** de ce réseau ne reste
dans la page.

Les journaux ont été entièrement retirés de la présentation : aucun onglet,
recherche, filtre ou téléchargement de logs, et aucun journal dans les cartes
d'artefacts. Leur stockage backend reste actif pour le diagnostic et la
réconciliation des exécutions.

## Limites de la qualification

Les adaptateurs de prévision/backtest sont validés sur leurs arguments, snapshots,
contrats et protections de fichiers ; aucune exécution scientifique lourde n'a
été déclenchée. Leur validation numérique complète nécessite une action explicite
et les caches/modèles requis. Les formats non reconnus ne sont pas reconstruits.
Les conclusions scientifiques et la qualité des modèles ne sont pas évaluées
par cette intégration. Voir le README de la console pour toutes les limites.
