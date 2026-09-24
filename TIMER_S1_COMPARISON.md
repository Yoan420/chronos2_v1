# Comparaison Timer-S1 / Chronos-2

Sources officielles: [checkpoint Timer-S1](https://huggingface.co/bytedance-research/Timer-S1)
et [article arXiv](https://arxiv.org/pdf/2603.04791).

Ce runner reproduit le protocole horaire FR gele (origines D-1 a 08:00 heure
de Paris, journees DST de 23/24/25 heures, contexte de 2 048 heures, memes
fenetres EXT223 + calibration 365 jours + evaluation scellee 365 jours).
Ces cinq parametres de protocole sont volontairement non configurables: le
runner refuse toute valeur differente afin de ne pas certifier silencieusement
un autre experiment comme une comparaison du systeme courant.

## Ce que signifie « memes features »

Timer-S1 publie une interface strictement univariee. Il ne peut donc pas
recevoir nativement les 17 covariables de contexte et les 12 covariables
connues dans le futur que le backbone Chronos-2 actuel utilise. Empiler ces
features comme des lignes du batch serait incorrect: Timer-S1 les traiterait
comme des series independantes.

Le runner distingue trois pistes conceptuelles:

1. **Comparaison native stricte**: Timer-S1 target-only contre Chronos-2
   target-only, avec exactement la meme cible, les memes 2 048 heures de
   contexte, origines, horizons et lignes d'evaluation.
2. **Information commune exacte**: les deux backbones target-only alimentent
   chacun un correcteur neuf partageant exactement les memes 40 features et
   le meme schema de 175 meta-features. Cette piste est le test symetrique le
   plus complet lorsque son artefact est publie.
3. **Challenger du systeme actuel**: le pipeline Timer-S1 est aussi presente
   face au systeme Chronos-2 gele actuellement exploite. Cette piste mesure
   l'interet operationnel du challenger, tout en signalant clairement
   l'asymetrie d'information native des backbones.

Les manifests et `feature_parity.json` conservent explicitement cette
distinction afin d'eviter une conclusion trompeuse.

La metrique primaire reste la fenetre scellee de 365 jours. Comme une partie
de cette fenetre precede la publication de Timer-S1, le runner publie aussi
une sensibilite secondaire limitee aux jours locaux a partir du 10 avril 2026
(le lendemain de la v3 arXiv datee du 9 avril). Elle contient le MAE q50, le
nombre d'heures/jours et les deltas MAE apparies par jour. Cette coupe reduit
le risque de contamination du benchmark anterieur a la publication; elle ne
prouve pas l'absence de donnees d'entrainement chevauchantes.

## Installation isolee

Timer-S1 demande `transformers>=4.57.1,<4.58`. Utiliser explicitement Python
3.11 (et non le `python` par defaut s'il pointe vers Python 3.14) dans un
environnement separe du runtime courant:

```powershell
py -3.11 -m venv .venv-timer-s1-py311
.\.venv-timer-s1-py311\Scripts\python.exe -m pip install --upgrade pip
.\.venv-timer-s1-py311\Scripts\python.exe -m pip install --no-cache-dir -r requirements_timer_s1.txt
```

Si un depot Artifactory renvoie une erreur de decodage gzip
(`UnicodeDecodeError`, octet `0x8b`), la cause est une page HTML compressee
sans en-tete HTTP `Content-Encoding: gzip`. Comme toutes les dependances de ce
fichier sont publiques, contourner ponctuellement les index additionnels avec
PyPI seul:

```powershell
.\.venv-timer-s1-py311\Scripts\python.exe -m pip install `
  --isolated --no-cache-dir --prefer-binary `
  --index-url https://pypi.org/simple `
  --cert "$env:USERPROFILE\certs\full-ca.pem" `
  -r requirements_timer_s1.txt
```

`--isolated` ignore les `extra-index-url` utilisateur pour cette seule
commande; il ne modifie aucune configuration persistante. Dans un
environnement qui interdit PyPI direct, utiliser une seule URL Artifactory
HTTPS canonique et authentifiee, puis faire corriger le depot/proxy s'il
continue a servir du gzip sans en-tete coherent. Ne pas desactiver TLS.

Le checkpoint BF16 est volumineux (environ 16,6 Go) et sa fiche recommande un
GPU d'au moins 40 Go. Le runner bloque avant chargement si aucune ressource
raisonnable n'est detectee. `--allow-low-memory` existe uniquement comme
contournement explicite et peut provoquer un OOM.

## Execution sans telechargement accidentel

Une action est obligatoire. `--help`, `--plan-only` et `--compare-only` ne
chargent aucun modele et ne telechargent rien.

Valider le protocole et produire le diagnostic materiel:

```powershell
python run_timer_s1_comparison.py --plan-only
```

Generer ou reprendre Timer-S1 (action qui peut telecharger le checkpoint):

```powershell
python run_timer_s1_comparison.py --generate-timer
```

Pour interdire le reseau et exiger un checkpoint deja en cache:

```powershell
python run_timer_s1_comparison.py --generate-timer --local-files-only
```

Generer le denominateur Chronos-2 target-only (configuration locale par
defaut):

```powershell
python run_timer_s1_comparison.py --generate-chronos-target-only
```

Publier la comparaison a partir des OOF existants:

```powershell
python run_timer_s1_comparison.py --compare-only
```

Chaque generation ecrit des checkpoints atomiques formes de journees locales
completes sous `runs/experiments/timer_s1_fr/checkpoints/`. Une relance valide
et reutilise les checkpoints existants; un checkpoint invalide provoque un
arret plutot qu'un ecrasement silencieux.

Chaque CSV de generation possede un sidecar de provenance obligatoire. Avant
toute reprise ou comparaison, le runner verifie son SHA-256, le modele et sa
revision, le contexte de 2 048 heures, le mode target-only, les options
d'inference (ReVIN/cache/quantiles ou cross-learning), le hash du manifest de
40 features et le hash de toute la cible UTC, y compris l'historique precedant
EXT223. Un CSV sans sidecar valide est refuse et n'est jamais « recertifie ».

## Artefacts principaux

- `timer_s1_plan.json`: contrat, parite des features, fenetres et ressources.
- `timer_s1_oof.csv.gz` et son manifest: OOF Timer-S1 target-only.
- `chronos2_target_only_oof.csv.gz` et son manifest: denominateur natif strict.
- `comparison/system_metrics.csv`: comparaison des systemes sur la fenetre
  scellee.
- `comparison/strict_target_only_metrics.csv`: comparaison native honnete.
- `comparison/strict_common_metrics.csv`: deux backbones target-only, puis
  correcteurs symetriques utilisant exactement les memes 40/175 features.
- `comparison/paired_tests.json`: bootstrap paire par jour de livraison.
- `comparison/post_publication_sensitivity_*.json`: sensibilites secondaires
  a partir du 10 avril 2026, avec avertissement de contamination explicite.
- `comparison/feature_parity.json`: affirmation exacte de parite aval et
  absence de parite native.
- `comparison/artifact_checksums.json`: sommes SHA-256 des resultats publies.

Si l'OOF Chronos-2 target-only n'existe pas, `--compare-only` publie quand meme
la comparaison systeme et marque le test natif strict comme incomplet dans
`strict_target_only_status.json`; il ne le remplace jamais par une comparaison
injuste.
