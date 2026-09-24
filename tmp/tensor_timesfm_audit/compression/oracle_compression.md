# Diagnostic ORACLE de compression PCA — aucune prévision

Les coordonnées latentes du test sont calculées à partir des prix ou erreurs futurs effectivement observés. Ces reconstructions constituent un diagnostic de perte d’information dans une base figée ; elles ne sont ni des prévisions disponibles au cutoff, ni une mesure de gain de NYX.

## Protocole et alignement

- Quatre pays BE, DE, FR, NL ; 8 760 heures UTC communes, sans doublon ni heure supprimée.
- Train : 180 jours, 4 321 heures par pays ; validation : 95 jours, 2 279 heures, inutilisée ; test : 90 jours, 2160 heures, du 2026-06-18 au 2026-09-15.
- Les journées locales sont vérifiées contre leur grille physique ; le jour de 25 heures est conservé dans le train et celui de 23 heures dans la validation.
- Centrage et SVD uniquement sur train, aucun écart-type appliqué : unité EUR/MWh conservée. Les pays de plus forte variance peuvent donc dominer la base.
- Deux diagnostics principaux : prix révisés `observed`, et erreurs finales `observed − NYX`. Deux sensibilités indépendantes utilisent `frozen_actual` à la place des observations révisées.
- Seuils de pics positifs : quantiles 95 % / 99 % des prix du train, séparément par pays (bornés à zéro). Les régimes du test servent uniquement à mesurer, jamais à ajuster.

## Résultats ORACLE sur le test

| Espace | Rang | Variance train retenue | RMSE globale | MAE globale | RMSE prix négatifs | RMSE pics q95 | RMSE pics q99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| observed_price | 1 | 86.41 % | 20.731 | 14.221 | 12.409 | 26.550 | 38.893 |
| observed_price | 2 | 98.03 % | 11.737 | 4.948 | 6.418 | 15.325 | 26.045 |
| observed_price | 3 | 99.41 % | 6.271 | 2.211 | 3.447 | 7.037 | 11.630 |
| observed_price | 4 | 100.00 % | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| observed_minus_nyx | 1 | 65.70 % | 13.644 | 7.418 | 8.197 | 16.866 | 27.111 |
| observed_minus_nyx | 2 | 85.89 % | 10.886 | 4.971 | 5.686 | 14.776 | 24.945 |
| observed_minus_nyx | 3 | 95.83 % | 7.019 | 2.918 | 3.260 | 9.332 | 15.747 |
| observed_minus_nyx | 4 | 100.00 % | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Toutes les erreurs sont des **erreurs de reconstruction ORACLE**, en EUR/MWh. Le diagnostic résiduel fournit déjà les erreurs finales futures exactes au projecteur.

### Spreads et prix négatifs — reconstruction ORACLE

| Espace | Rang | RMSE spread FR−DE | RMSE spread BE−NL | Prix négatifs reconstruits ≥ 0 |
|---|---:|---:|---:|---:|
| observed_price | 1 | 44.600 | 20.040 | 228/522 |
| observed_price | 2 | 6.852 | 17.455 | 201/522 |
| observed_price | 3 | 3.739 | 17.307 | 151/522 |
| observed_price | 4 | 0.000 | 0.000 | 0/522 |
| observed_minus_nyx | 1 | 20.846 | 18.806 | 209/522 |
| observed_minus_nyx | 2 | 4.413 | 18.075 | 207/522 |
| observed_minus_nyx | 3 | 3.945 | 19.273 | 161/522 |
| observed_minus_nyx | 4 | 0.000 | 0.000 | 0/522 |

Pour les résidus, les spreads et le signe sont calculés sur `NYX + résidu reconstruit`. Les erreurs de spread sont identiques à celles de la différence des deux résidus reconstruits.

## Sensibilité aux observations révisées

| Rang | RMSE prix observed | RMSE prix frozen_actual | RMSE résidu observed | RMSE résidu frozen_actual |
|---|---:|---:|---:|---:|
| 1 | 20.730707 | 20.730707 | 13.643625 | 13.643625 |
| 2 | 11.736750 | 11.736750 | 10.886489 | 10.886489 |
| 3 | 6.270947 | 6.270947 | 7.019132 | 7.019132 |
| 4 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Les détails par pays, biais, effectifs, seuils train, charges des composantes, grandes erreurs résiduelles et grands spreads sont dans `oracle_compression.json`.

## Constats de compression, sans conclusion prévisionnelle

- Le rang 3 conserve 99.41 % de la variance des prix du train, mais son erreur de reconstruction ORACLE du spread BE−NL reste 17.31 EUR/MWh de RMSE. La petite composante écartée conserve donc une information utile pour cette divergence entre pays.
- Pour les erreurs finales NYX, le rang 3 laisse 7.02 EUR/MWh de RMSE de reconstruction ORACLE. Cela ne permet pas de déduire un gain réalisable : il faudrait ensuite prévoir les facteurs.
- Les régimes test comptent 522 cellules de prix négatifs, 3393 au-dessus du q95 train et 1025 au-dessus du q99 train, sur 8640 cellules pays×heure. « q99 train » ne signifie pas les 1 % les plus hauts du test ; les seuils restent ceux appris avant ce test.

## Interprétation autorisée

- Le rang 4 est la reconstruction exacte triviale d’un vecteur à quatre coordonnées : aucune compression et aucune performance prévisionnelle démontrée.
- Les rangs 1–3 mesurent ce que la projection supprime même avec des coordonnées futures parfaites. Un grand pourcentage de variance train retenue ne garantit pas une bonne préservation des pics, des prix négatifs ou des divergences de pays dans le test.
- Pour cette base et ce décodeur linéaire figés, la projection orthogonale est la meilleure reconstruction quadratique dans le sous-espace choisi. Une prévision imparfaite des facteurs ajoute de l’erreur ; ce diagnostic n’est pas une borne universelle pour tout décodeur non linéaire.
- La possibilité de compresser des erreurs contemporaines ne démontre pas que leurs facteurs soient prévisibles. Le test causal suivant doit estimer les facteurs sans aucune valeur future du test.
- Aucun rang n’a été choisi sur ces résultats ; un choix de rang ou de normalisation pour une future comparaison doit utiliser la validation, puis un nouveau test réellement réservé.

## Reproduction

```powershell
& 'C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe' 'tmp/tensor_timesfm_audit/compression/oracle_pca_compression.py'
```

Entrée SHA256 : `0548b72b6fadcb700d1b863db93fd95d0076670fb5e11bf4ad1c1fe1413a1188`. Le script contrôle que cette empreinte demeure inchangée après lecture. Il ne charge aucun modèle ni poids ; seuls les fichiers de ce diagnostic sont écrits.
