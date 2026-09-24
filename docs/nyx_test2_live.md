# NYX / Test2 — candidat reel isole

Reference choisie le 23 septembre 2026 : hybride `3a08985a6007d5b8`,
NYX SolarWind interaction +/-40 apres Kalman, avec Test2 hour-local sur les
seules heures selectionnees par le routage causal. Le registre sous
`runs/model_references/nyx_test2_hybrid` ne modifie aucune configuration active.

## Perimetre

La livraison initiale est le **24 septembre 2026**, comme le batch operationnel
en cours. Ordre : DE, NL, apprentissage/routage DE-NL et rapports, puis BE, FR
et validation/rapports propres a cette extension. Il s'agit d'une execution
reelle parallele (shadow), pas d'un remplacement de la production.

Les sources, snapshots, caches, checkpoints et rapports restent dans
`runs/experiments/nyx_test2_live_v1`. Les donneurs de production ne sont lus
qu'une fois leur bundle scelle et verifie. Aucune interruption, suppression de
verrou, modification de recette active ni ecriture dans leurs caches.

Les acquisitions sont limitees aux suffixes manquants (31 jours maximum),
avec les memes series Saturn, unite GW, coupure civile D-1 08:00 et politiques
DST documentees. Aucune generation realisee n'est substituee a une prevision.
Les seules substitutions historiques admises restent les deux heures NL
de printemps deja explicites dans la reference. Les donnees query-asof
historiques ne constituent pas une preuve de publication initiale.

## Apprentissage

Chronos reste le modele pretrained de la recette : ce lancement rejoue les
previsions et apprend les correcteurs, experts Test2 et leur routage ; il ne
fine-tune pas les poids de Chronos.

DE/NL conservent leurs fonctions scientifiques et seuils de reference.
Le routage n'observe ni le prix realise de l'heure courante, ni la sortie
Test2 pour decider du basculement. Les trois quantiles sont copies ensemble.
Les ajustements sont appris uniquement a partir des jours precedant leur
origine, avec tests sur les grilles physiques 23/24/25 heures.

Pour BE/FR, le voisin est defini explicitement BE <-> FR et l'indicateur pays
est propre a cette paire. Les vents propres BE/FR servent au score du
correcteur et aux experts/au routage. Ils n'ajoutent pas implicitement de
nouvelles entrees au Chronos ou au Kalman SolarWind (six courbes generation).
Cette extension est une nouvelle identite scientifique, non une validation
transferee de DE/NL. Ses experts et politiques sont appris separement.

La baseline fournit 365 jours de previsions prequentielles ; Test2 est rejoue
sur 184 jours, avec une origine hebdomadaire relative a la livraison. Le
premier routage attend au moins 90 jours complets de sorties Test2 hors
echantillon (premiere origine admissible a +91 jours). Les rapports hybrides
couvrent donc 93 jours, pas une annee. La livraison finale utilise un fit
frais, sans prix realise futur. Les scores ne certifient pas les performances
futures et les intervalles ne sont pas garantis calibres a 80 %.

## Exploitation

Lanceur : `run_nyx_test2_live.py`, Python pricefm311, `-B -u`, fenetre cachee,
priorite BelowNormal, CPU, un worker / un thread. Une garde attend si moins
de 3,5 GiB sont disponibles avant une etape ; ce n'est pas une reservation OS.
Ce compromis limite la concurrence avec la production, sans promettre zero
impact sur les ressources partagees.

Le plan epingle le code, la reference, les dates, l'ordre et les ressources.
Un changement du plan ou du code bloque la reprise. Les checkpoints sont
verifies par SHA avant reutilisation. Un processus distinct qui possede le
verrou interdit un doublon. Aucune reprise automatique d'erreur deterministe.
Les status/logs distinguent attente d'un parent, preparation, calcul, erreur
et achevement. Un rapport n'est publie COMPLETE qu'apres scellage et verification.
