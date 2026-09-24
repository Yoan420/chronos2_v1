# Moniteur des calculs NYX

Petite fenêtre locale dédiée, indépendante de la console NYX principale,
reprenant son thème sombre noir, ambre et cyan.
Ouvrir `ProcessMonitor.ps1` avec PowerShell, ou le raccourci **Moniteur NYX**.
Adresse locale de secours : http://127.0.0.1:8766/.

Le programme affiche l'état, la phase, les pays terminés, la durée, l'activité
CPU/RAM des processus et de leurs workers, les journaux et les rapports déjà
présents. Seuls les calculs dont des processus identifiés sont encore présents
sont affichés ; les anciens lancements et les calculs terminés ont disparu de
la vue. Les workers encore actifs sont conservés même si leur parent a disparu.

Un collecteur partagé relève l'activité en arrière-plan avec une cadence cible
d'une seconde, sans scans simultanés. L'interface consulte le dernier relevé
chaque seconde et anime localement les durées et comptes à rebours. Sous forte
charge, un relevé peut prendre davantage de temps : sa date reste visible,
et les données anciennes ou hors ligne sont signalées, pas présentées comme
une observation actuelle.

Le collecteur ne découvre les lancements qu'une fois par relevé. Il vérifie
toujours chaque composant des chemins (liens et jonctions refusés), mais ne
résout plus répétitivement toutes leurs chaînes de parents. Les journaux des
lancements inactifs ne sont pas relus ; une fin de journal inchangée est
réutilisée après revalidation du chemin et de son identité de fichier, de sa
taille et de ses dates. Les statuts et inventaires de fin restent relus, sans
cache de confiance ni cache des preuves de complétude.

Un calcul par lot ne publie pas nécessairement de compteur intermédiaire.
Le moniteur affiche alors une progression indéterminée. « 1 pays sur 2 »
signifie un pays terminé, pas 50 % du temps de calcul.
Un journal immobile ne prouve pas un blocage. Un PID est identifié avec sa
date de création pour éviter de confondre deux processus réutilisant le même
numéro. Les parents du venv et les workers ne sont pas des batches distincts.

## Estimation de durée restante

Chaque calcul actif affiche une durée restante approximative, une heure de fin,
une fourchette indicative et la méthode utilisée. Les durées de pays déjà
terminés servent de référence lorsque les étapes sont comparables. La charge
de la machine et le pays peuvent modifier fortement le temps nécessaire.

Sans compteur ni référence exploitable, une **hypothèse non calibrée** est
affichée avec une confiance très faible : durée totale initiale supposée égale
au maximum d'une heure et du double du temps déjà écoulé. C'est une convention
de repli, pas une mesure de travail restant. L'hypothèse est ancrée puis révisée
si elle est dépassée ; aucune valeur zéro ne prétend que le calcul est fini
tant que ses processus sont encore actifs. Les fourchettes ne sont pas des
intervalles statistiques garantis.

## Aucun contrôle des calculs

Le moniteur est en lecture seule : pas de bouton d'arrêt, relance, suppression,
ni modification des sources, configurations ou checkpoints. Fermer sa fenêtre
ne termine aucun calcul. Le service de lecture reste disponible en arrière-plan
et une nouvelle ouverture le réutilise ; il ne démarre jamais une expérience.
La console NYX existante sur le port 8765 reste indépendante et inchangée.

Le service écoute seulement sur 127.0.0.1. Aucun hébergement ou compte externe,
aucune nouvelle dépendance téléchargée. Les journaux affichés sont limités en
taille et les secrets usuels masqués. Seuls des rapports identifiés à l'intérieur
des résultats du projet sont accessibles ; ils s'ouvrent dans un bac à sable.

## Découverte des lancements

Les métadonnées dans `runs/experiments/*/launcher_logs/*.launch.json` sont
découvertes automatiquement, notamment le batch SolarWind interaction DE/NL.
Les autres lanceurs `run_*.py` actifs du projet peuvent être présentés comme
« calcul détecté » : leur état de processus n'est pas une preuve de progression
scientifique et ne leur attribue pas un pourcentage inventé.

Pour les prochains lancements de l'assistant, conserver un fichier unique
`.launch.json`, sans secret, contenant l'heure, la commande, les PID et dates
de création, les chemins des journaux et les identités des workdirs par pays.
Une ancienne métadonnée sans date de création reste incertaine, jamais utilisée
pour attribuer silencieusement un autre processus au calcul historique.

Les journaux propres au moniteur et son profil de fenêtre se trouvent dans
`runs/.process_monitor/`. Ils ne sont pas utilisés par les modèles scientifiques.
