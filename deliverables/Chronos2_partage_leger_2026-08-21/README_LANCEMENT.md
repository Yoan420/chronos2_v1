# Lancer Chronos-2 — paquet léger

Ce paquet contient uniquement le code, les configurations, les données causales et les artefacts figés nécessaires. Il ne contient aucun ancien forecast dans `runs\live` ou `runs\exports`.

Les poids Amazon Chronos-2 ne sont pas inclus afin d'alléger le transfert. Ils seront téléchargés une seule fois au premier lancement.

## Pré-requis

- Windows avec **Python 3.11** installé ;
- accès au réseau ou VPN ENGIE et droits de lecture sur Saturn ;
- accès à Hugging Face pour le premier téléchargement du modèle.

## Installation — une seule fois

1. Copier le dossier `chronos2_v1` dans le dossier utilisateur Windows.
2. Ouvrir PowerShell dans `chronos2_v1`.
3. Exécuter :

```powershell
py -3.11 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r requirements_transfer.txt
```

## Premier lancement

Le premier lancement télécharge les poids Chronos-2 dans le dossier local `huggingface_cache` :

```powershell
& 'C:\Users\<son_identifiant>\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Both -AllowModelDownload
```

Remplacer `<son_identifiant>` par le nom de l'utilisateur Windows.

## Commande principale

Après le premier téléchargement, utiliser normalement :

```powershell
& 'C:\Users\<son_identifiant>\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Both
```

Le mode `Both` produit les vues autonome et blend pour FR/NL, et la vue autonome pour DE/BE/ES. Les pays sont calculés l'un après l'autre.

Pour vérifier la commande sans lancer le calcul :

```powershell
& 'C:\Users\<son_identifiant>\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Both -DryRun
```

Le lanceur utilise automatiquement l'environnement `.venv`, le cache local du modèle et l'identifiant Windows courant comme auteur Saturn.

## Résultats

Les nouveaux résultats seront créés dans `runs\live` et `runs\exports`. Le CSV contient P10, P50 et P90 ; le rapport HTML contient les visualisations.

## À propos des dossiers conservés dans `runs`

Les dix dossiers `chronos2_hourly_*` fournis ne sont pas d'anciens forecasts live. Ils contiennent les artefacts de calibration, recettes et contrôles d'intégrité indispensables au modèle actuel. Les supprimer empêcherait le lancement.

## En cas de problème

- **Téléchargement impossible :** vérifier l'accès à Hugging Face ; sinon utiliser le paquet complet contenant déjà les poids.
- **Erreur Saturn :** vérifier le VPN et les droits Saturn de l'utilisateur Windows.
- **Python introuvable :** vérifier que `.venv` a bien été créé.
- **« Déjà publiée » :** une prévision valide existe déjà et n'est pas écrasée.
- **Échec du calcul :** conserver les dernières lignes des logs pour le support technique.

Ne pas modifier les configurations ou les artefacts figés : leurs empreintes sont contrôlées au lancement.
