# Lancer Chronos-2 sur un nouvel ordinateur

Le dossier `chronos2_v1` contient le code, les configurations, les données nécessaires, les artefacts figés et les poids Amazon Chronos-2.

## Pré-requis

- Windows avec **Python 3.11** installé ;
- accès au réseau ou VPN ENGIE ;
- droits de lecture sur Saturn.

Chaque utilisateur doit utiliser son propre accès Saturn. Aucun mot de passe n'est fourni dans ce paquet.

## Installation — une seule fois

1. Copier le dossier `chronos2_v1` sur l'ordinateur.
2. Ouvrir PowerShell dans ce dossier.
3. Exécuter :

```powershell
py -3.11 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r requirements_transfer.txt
```

## Commande principale — cinq pays et deux vues

Après l'installation, la commande principale est :

```powershell
& 'C:\Users\<son_identifiant>\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Both
```

Remplacer `<son_identifiant>` par le nom de l'utilisateur Windows. Le mode `Both` produit les vues autonome et blend pour FR/NL, et la vue autonome pour DE/BE/ES.

Le lanceur fourni utilise automatiquement :

- l'environnement Python local `.venv` ;
- les poids présents dans `huggingface_cache` ;
- l'identifiant Windows courant comme auteur Saturn.

Pour vérifier la commande sans lancer le calcul, ajouter `-DryRun` :

```powershell
& 'C:\Users\<son_identifiant>\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Both -DryRun
```

## Test France uniquement

Pour commencer par un seul pays :

```powershell
& 'C:\Users\<son_identifiant>\chronos2_v1\Forecast.ps1' -Action Run -Countries FR -Mode Both
```

Le jour de livraison est le lendemain par défaut. Pour imposer une date :

```powershell
& 'C:\Users\<son_identifiant>\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Both -DeliveryDay 2026-08-22
```

Les pays sont calculés l'un après l'autre.

## Résultats

Les fichiers sont écrits dans `runs\exports` :

- le CSV contient les prévisions horaires P10, P50 et P90 ;
- le rapport HTML permet de visualiser les courbes.

## En cas de problème

- **Erreur Saturn :** vérifier le VPN et les droits Saturn de l'utilisateur Windows.
- **Python introuvable :** vérifier que l'installation de `.venv` a été effectuée.
- **« Déjà publiée » :** une prévision valide existe déjà et n'est pas écrasée.
- **Échec du calcul :** conserver les dernières lignes des logs pour le support technique.

Ne pas modifier les configurations ou les artefacts figés : leurs empreintes sont contrôlées au lancement.
