# Lancer Chronos-2 — fiche rapide

## Objectif

Lancer le modèle **Chronos-2**, afin de produire la prévision de prix du lendemain.

## Fichiers à partager

Le plus simple et le plus sûr est de transmettre une copie complète du dossier :

`chronos2_v1`

Il faut conserver la même arborescence. La copie doit notamment contenir :

- `Forecast.ps1` et les fichiers Python situés à la racine ;
- les dossiers `chronos2_hourly` et `chronos2_modular` ;
- tous les fichiers `.yaml` et `.json` situés à la racine ;
- le dossier `data`, qui contient les caches et les historiques PIT ;
- les dossiers de modèles figés dans `runs` :
  - `chronos2_hourly_fr_residual_extended_v1` ;
  - `chronos2_hourly_fr_mkonline_blend_v1` ;
  - `chronos2_hourly_de_residual_extended_v1` et `chronos2_hourly_de_sealed_benchmark_v1` ;
  - `chronos2_hourly_be_residual_extended_v1` et `chronos2_hourly_be_sealed_benchmark_v1` ;
  - `chronos2_hourly_nl_residual_extended_v1` et `chronos2_hourly_nl_mkonline_blend_v1` ;
  - `chronos2_hourly_es_residual_extended_v1` et `chronos2_hourly_es_sealed_benchmark_v1` ;
- `requirements.txt` et `requirements_hourly.txt`.

Les dossiers de résultats `runs\live`, `runs\exports` et `runs\tmp` ne sont pas nécessaires pour transmettre le modèle. Il ne faut pas modifier les fichiers `.yaml`, `.json` ou les artefacts figés, car leurs empreintes sont contrôlées au lancement.

### Poids Chronos-2

Deux possibilités :

- **avec accès à Hugging Face :** ne rien copier de plus et utiliser `-AllowModelDownload` lors du premier lancement ;
- **sans accès à Hugging Face :** partager aussi le dossier suivant, d'environ 450 Mo :

  `C:\Users\BQ6757\.cache\huggingface\hub\models--amazon--chronos-2`

  Le destinataire doit le copier dans :

  `C:\Users\<son_identifiant>\.cache\huggingface\hub\models--amazon--chronos-2`

Ne pas transmettre l'environnement Python `venv` : il doit être recréé sur l'autre ordinateur.

## Préparation du nouvel ordinateur

Pré-requis : **Python 3.11**, un accès au réseau/VPN ENGIE et les droits de lecture sur Saturn.

Chaque utilisateur doit employer ses propres droits Saturn. Aucun mot de passe ni identifiant de connexion ne doit être transmis avec le dossier.

Dans le dossier `chronos2_v1`, exécuter une seule fois :

```powershell
py -3.11 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r requirements_hourly.txt
& '.\.venv\Scripts\python.exe' -m pip install tshistory_lite==0.5
```

Avant chaque lancement, renseigner l'identifiant Saturn de l'utilisateur :

```powershell
$env:SATURN_AUTHOR = '<identifiant_utilisateur>'
$PythonChronos = (Resolve-Path '.\.venv\Scripts\python.exe').Path
```

## Procédure

1. Ouvrir une fenêtre **PowerShell Windows normale**.

2. Aller dans le dossier du projet :

```powershell
cd 'C:\Users\<son_identifiant>\chronos2_v1'
```

3. Définir Python et l'identifiant Saturn :

```powershell
$env:SATURN_AUTHOR = '<identifiant_utilisateur>'
$PythonChronos = (Resolve-Path '.\.venv\Scripts\python.exe').Path
```

4. Vérifier d'abord la commande, sans lancer le calcul :

```powershell
& '.\Forecast.ps1' -Action Run -Countries FR -Mode Autonomous -PythonExecutable $PythonChronos -DryRun
```

5. Si aucune erreur n'est affichée, lancer le test réel pour la France :

```powershell
& '.\Forecast.ps1' -Action Run -Countries FR -Mode Autonomous -PythonExecutable $PythonChronos
```

Si les poids Chronos-2 n'ont pas été copiés, ajouter `-AllowModelDownload` lors de ce premier lancement :

```powershell
& '.\Forecast.ps1' -Action Run -Countries FR -Mode Autonomous -PythonExecutable $PythonChronos -AllowModelDownload
```

Le jour de livraison est automatiquement fixé au lendemain. Pour choisir une autre date, ajouter par exemple :

```powershell
& '.\Forecast.ps1' -Action Run -Countries FR -Mode Autonomous -DeliveryDay 2026-08-22 -PythonExecutable $PythonChronos
```

6. Laisser la fenêtre PowerShell ouverte jusqu'à la fin du calcul.

7. Consulter les résultats dans :

`C:\Users\<son_identifiant>\chronos2_v1\runs\exports`

Le fichier CSV contient les valeurs horaires P10, P50 et P90. Le fichier HTML permet de visualiser la prévision.

## Commande principale : cinq pays et deux vues

La commande principale à utiliser est :

```powershell
& 'C:\Users\<son_identifiant>\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Both
```

Le mode `Both` produit les vues autonome et blend pour FR/NL, et la vue autonome pour DE/BE/ES. Les pays sont calculés l'un après l'autre.

## En cas de problème

- **Message « déjà publiée » :** c'est normal ; une prévision valide existante n'est pas écrasée.
- **Erreur Saturn :** vérifier le VPN, les droits Saturn et la valeur de `SATURN_AUTHOR`.
- **Python introuvable :** vérifier que `$PythonChronos` a bien été défini dans la fenêtre PowerShell courante.
- **Échec du calcul :** vérifier les dernières lignes des logs et les transmettre au support technique.
