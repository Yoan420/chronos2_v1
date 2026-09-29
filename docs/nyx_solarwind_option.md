# Choisir SolarWind interaction ±40 dans NYX

Dans **Résultats → Calculer avec NYX**, le champ **Modèle** propose :

- **NYX actuel · BE, DE, FR, NL** : sélection par défaut, avec le lancement
  `NuclearKalman.ps1` et ses paramètres existants.
- **SolarWind interaction ±40 · DE, NL** : variante séparée, sélectionnée
  explicitement pour la date de livraison choisie.

Après avoir récupéré la mise à jour de `main`, exécuter dans PowerShell :

```powershell
& .\Start-ExperimentConsole.ps1 -Restart
```

Cette commande recharge le backend déjà ouvert, uniquement sans calcul actif
ni en attente. Fermer la fenêtre NYX ne suffit pas à arrêter son backend.
Sans `-Restart`, le lanceur réutilise le serveur existant.

SolarWind conserve Chronos-2, le correcteur CatBoost et le Kalman. Les quatre
prévisions solaires FR/DE/BE/NL et les deux prévisions éoliennes DE/NL sont
des entrées explicites. L'interaction faible vent × faible solaire × forte
charge résiduelle entre dans CatBoost, avec une correction plafonnée entre
−40 et +40 €/MWh. Cette option ne lance pas Test2. Son périmètre est celui
de la variante initiale : Allemagne et Pays-Bas.

Le premier calcul doit reconstruire l'historique avec ces entrées et peut
être long. Les poids Chronos locaux et l'accès aux sources Saturn sont
nécessaires, ainsi que les historiques bruts audités nucléaire/résiduel et
solaire/éolien déjà préparés. Le moteur en copie les sources dans des caches
dédiés avant de synchroniser les jours manquants. Une source indisponible ou une erreur du calcul est signalée
sur ce lancement ; aucun autre modèle n'est substitué silencieusement.

Les rapports SolarWind et les CSV DE/NL apparaissent sous le suivi de cette
option après publication complète et vérification. Ils sont séparés des
rapports du modèle actuel, qui restent affichés dans la page. On peut
revenir à **NYX actuel** à tout moment hors soumission d'une demande.
La sélection revient au modèle actuel lors d'une nouvelle ouverture de l'app.

## Isolation

La nouvelle route de lancement est `/api/solarwind-run`. L'ancienne route
`/api/primary-run` et le lanceur `NuclearKalman.ps1` gardent leur contrat.
Le moteur facultatif est chargé à la demande. La file de calcul partage les
ressources scientifiques avec le modèle actuel pour éviter leur exécution
simultanée depuis NYX.

Les publications sont sous `runs/solarwind_interaction40/YYYY-MM-DD` :
`index.html`, `forecast_de.csv`, `forecast_nl.csv`, puis `manifest.json` écrit
en dernier. L'app ne présente comme résultat que les publications complètes
dont les empreintes correspondent au manifeste. Les paramètres scientifiques
et les publications nucléaires ne sont pas remplacés.

Le code et les configurations de `main` définissent cette nouvelle exécution
datée ; les anciens bundles de recherche du 22 septembre ne sont pas requis.
Cela ne reconstitue pas leurs anciens scores et ne constitue pas une nouvelle
validation de performance du modèle.
