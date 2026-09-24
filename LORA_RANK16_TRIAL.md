# Essai des deux chaines LoRA rang 16

Deux sorties, a partir du meme checkpoint rang 16 deja entraine :

- Chronos-2 + LoRA rang 16 + correcteur residuel neuf par pays.
- La meme chaine, suivie du Kalman gouverne standard.

LoRA brut n'est qu'une reference de diagnostic. Le candidat NOAA et la
production actuelle sont independants de cet essai. `Forecast.ps1` ne change pas.

## Utilisation

Dans PowerShell, depuis n'importe quel dossier :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\LoRATrial.ps1' -Action Prepare
& 'C:\Users\BQ6757\chronos2_v1\LoRATrial.ps1' -Action Status

# Optionnel, une seule fois : completer les jours passes et deja publies.
& 'C:\Users\BQ6757\chronos2_v1\LoRATrial.ps1' -Action Bootstrap -DeliveryDay 2026-09-08

# Pour le 9 septembre : lancer le 8 septembre apres 08:00 Paris.
# Les deux chaines doivent etre TERMINEES avant 11:45 et avant publication du prix.
& 'C:\Users\BQ6757\chronos2_v1\LoRATrial.ps1' -Action Run -DeliveryDay 2026-09-09 -Zones FR,DE,BE,NL

# Apres l'auction : rattacher les observations et regenerer un nouveau rapport.
& 'C:\Users\BQ6757\chronos2_v1\LoRATrial.ps1' -Action Resolve
```

## Comparer avec un run deja produit

### Rapports HTML complets sur 365 jours

```powershell
& 'C:\Users\BQ6757\chronos2_v1\LoRATrial.ps1' -Action FullReport -DeliveryDay 2026-09-08 -Zones FR
# Meme outil pour les quatre pays, lorsque les inputs et Bootstrap sont disponibles :
& 'C:\Users\BQ6757\chronos2_v1\LoRATrial.ps1' -Action FullReport -DeliveryDay 2026-09-08 -Zones FR,DE,BE,NL
```

`FullReport` produit deux rapports avec le moteur HTML habituel : forecast,
backtest et probabilites, Statistics, Statistics — prix moyens, calendrier,
comparateur Storm officiel et mode nuit. Les scores sont **recalcules sur les
predictions des deux chaines LoRA**, jamais recopies depuis les anciens modeles.
Pour cette livraison, la fenetre fixe est le **09/09/2025–08/09/2026** (365 jours).
Une observation absente reste vide, sans reculer la fenetre vers une autre date.

Le premier lancement reconstruit les 359 jours de predictions brutes manquants
du 09/09/2024 au 02/09/2025 avec le checkpoint rang 16 deja entraine, puis rejoue
les correcteurs sur 365 origines. Il ne relance pas l'entrainement LoRA. Les
calculs bruts et Kalman sont scelles jour par jour et reutilises apres verification
des empreintes. Une interruption peut donc etre reprise avec la meme commande.
Une relance cree une nouvelle version des rapports sans ecraser les precedentes.

Sorties exclusivement dans
`runs/experiments/chronos2_exogenous_rank16_rolling365_research_v1`.
Les exports de `Forecast.ps1`, l'essai prospectif et la comparaison d'une journee
restent inchanges. Le nouveau support historique commence en 2024 : le Kalman
du 8 septembre peut donc differer de celui du rapport d'une journee dont
l'amorcage commencait en 2025. Les deux protocoles restent distincts et traces.

**Limites methodologiques :** c'est un rejeu de recherche avec checkpoint fixe,
selectionne apres lecture de cette annee, pas un nouveau test independant. Le
prefixe 2024–2025 appartient a l'entrainement/validation LoRA ; ses predictions
sont in-sample pour le reseau et ne constituent pas un OOF neuronal. Le correcteur
historique commence avec 30 jours sans correction puis une fenetre croissante
jusqu'a 365 jours. Chaque jour **evalue** a exactement 365 jours anterieurs pour
son correcteur et pour son Kalman, mais l'historique corrige servant au Kalman
contient cet amorcage : ce n'est pas un historique doublement imbrique de 1 095
jours. Les anciennes covariables de contexte absentes gardent leurs NaN et le
masque natif Chronos utilise a l'entrainement ; les covariables futures restent
obligatoirement completes. Aucun label du jour predit ne sert a sa correction.

Pour comparer exactement aux rapports existants, les observations et Storm
sont lus dans leur snapshot HTML verifie par SHA. Ce n'est pas une nouvelle
lecture API des derniers prix. Ces observations de reporting n'entrent jamais
dans la calibration. Les heures Storm manquantes, notamment au changement
d'heure, ne sont pas interpolees ; la vue appariee utilise les memes heures des
deux cotes. P10/P50/P90 sont calcules, les autres deciles du moteur HTML sont
interpoles pour l'affichage et le CRPS approximatif. L'attribution des variables
n'est pas reutilisee depuis un autre modele. Aucune promotion automatique.

### Comparaison d'une seule livraison

La commande suivante fonctionne aussi apres l'enchere : `Run` annonce alors
explicitement le mode **RETROSPECTIF** (apres la limite D-1 11:45 Paris).

```powershell
& 'C:\Users\BQ6757\chronos2_v1\LoRATrial.ps1' -Action Run -DeliveryDay 2026-09-08 -Zones FR,DE,BE,NL
# Equivalent explicite pour une comparaison apres coup :
& 'C:\Users\BQ6757\chronos2_v1\LoRATrial.ps1' -Action Compare -DeliveryDay 2026-09-08 -Zones FR,DE,BE,NL
```

Ce calcul applique les deux chaines au 8 septembre avec calibration exclusivement
anterieure au 8 (les 365 jours jusqu'au 7). Les observations du 8 sont exclues de
l'apprentissage, meme si Bootstrap les contient deja. Les previsions existantes
du LoRA brut sont reutilisees ; les autres sont reconstruites avec les inputs
historiques verifies. Les forecasts `autonomous` et `kalman` du run existant
sont lus depuis leurs exports, avec verification des empreintes et des heures.

Le rapport affiche prix moyens, differences, MAE horaire et courbes pour cette
seule livraison. Les prix canoniques manquants restent vides et ne bloquent pas
le forecast ; aucun prix EPEX reserve au reporting n'est utilise pour calibrer.
Ce n'est ni une emission avant enchere ni un nouveau backtest de 365 jours.

Sorties distinctes : `runs/experiments/chronos2_exogenous_rank16_prospective_v1/retrospective/2026-09-08`.
Une relance de `Run`/`Compare` pour cette date reutilise les deux chaines deja
calculees et cree un nouveau rapport avec les observations actualisees, sans
modifier les anciens exports, le journal prospectif ou le manifeste initial.
`Resolve` reste reserve aux veritables emissions prospectives.

`Run` complete aussi les jours de calibration manquants, mais le faire en avance
avec `Bootstrap` evite d'allonger le calcul matinal. Une date deja scellee n'est
jamais recalculee. `Bootstrap` peut etre relance avec exactement les memes pays
et la meme date : il reutilise les journees terminees et les inputs historiques
deja verifies, et relit seulement les observations canoniques recentes.
Une journee dont les prix sont absents ou partiels reste `EN ATTENTE`
(`waiting_for_observations`), sans charger LoRA pour cette journee ; les autres
pays continuent. Aucun trou de calibration n'est saute. Relancer la meme
commande apres publication des prix pour terminer le suffixe manquant.
`Run` conserve son refus strict si la calibration anterieure est incomplete.
Dans la fenetre avant enchere, il conserve tous les controles prospectifs ;
aucune erreur de calibration ou de preuve n'est convertie en repli retrospectif.
Les tentatives incompletes sont conservees pour diagnostic.
Il n'y a pas de planification automatique : ces commandes sont manuelles.

Les captures, predictions et rapports restent dans
`runs/experiments/chronos2_exogenous_rank16_prospective_v1`.
Le reseau est necessaire pour les dernieres vintages Saturn/JAO et prix canoniques.
La premiere preparation ne charge pas le reseau neuronal et ne relance pas
les anciens backtests. Les checkpoints d'origine restent inchanges.

## Note methodologique

Les backtests bruts du 03/09/2025 au 02/09/2026 ont deja ete consultes pour
selectionner le rang 16. Ici ils servent exclusivement de **calibration**, pas
d'estimation independante des gains des deux nouvelles chaines.

La recette du correcteur est fixee avant le test : regression ridge (alpha 1)
de l'erreur du LoRA sur constante, sinus/cosinus de l'heure et du jour de semaine,
avec correction bornee a +/-20 EUR/MWh. Ses coefficients sont reestimes sur les
365 jours precedant chaque livraison. Les corrections historiques sont
reconstruites jour par jour avec seulement les labels anterieurs ; avant 30 jours
de support, la correction reste nulle. Les parametres d'echelle sont eux aussi
appris uniquement sur le passe. On ne reutilise pas le correcteur du Chronos
sans LoRA ni les coefficients de l'ancien test de validation de 30 jours.

Le Kalman apprend sur les erreurs **apres correction prequentielle** des 365
jours precedents : biais, profil harmonique, charge residuelle des cinq pays,
facteur d'echelle lineaire et UKF. La banque et sa gouvernance reprennent les
valeurs standard, figees dans le manifeste de l'essai. Un seul refit de l'origine
future est necessaire, pas un nouveau backtest Kalman de 365 origines.

Le support brut commence toujours au 03/09/2025, puis grandit : cela evite de
deplacer artificiellement la phase de demarrage du correcteur a chaque run.
Le dernier prix connu est rattache aux historiques, mais aucun prix du jour
predit ne doit etre present avant emission (controle avant et apres calcul).

Les metriques du nouveau rapport portent seulement sur les journees reellement
emises avant publication et ensuite observees, au maximum les 365 derniers jours
calendaires. Une journee non publiee reste vide ; une journee incomplete n'entre
pas dans les scores. Les jours de 23/25 heures sont conserves physiquement.

Cet essai ne constitue pas un OOF neuronal ni une attestation PIT de production :
les anciens inputs Saturn/JAO restent des reconstructions historiques. Il ne
contourne aucune regle de promotion et n'active aucun modele operationnel.
Les forecasts sont immuables ; `Resolve` conserve chaque version des observations
et du rapport, sans reexecuter LoRA, le correcteur ou le Kalman.

La correction de reprise du 07/09/2026 est tracee dans une revision de maintenance
separee, avec copie et SHA de l'ancien runner. Le manifeste initial, les journees
deja scellees, le checkpoint, les correcteurs et les recettes ne sont pas reecrits.
Cette revision ne peut pas etre enregistree apres une premiere emission prospective.
