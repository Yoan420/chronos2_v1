# Référence annuelle des variantes FR, BE et NL

Les variantes de prix retenues dans les rapports CWE du 23 septembre 2026
emploient la même référence `scarcity_confirmed_pair`. Le replay CPU en cours
peut lire sa colonne historique scellée, mais cela ne fournit pas une
référence pour les nouvelles dates de livraison.

## Formule exacte

Pour chaque heure UTC du pays cible, la référence vaut `test2__q50` si les
trois conditions suivantes sont vraies, et `ensemble__q50` sinon :

1. `prior90_active` est vrai pour le pays cible ;
2. sa probabilité Test2 `P(prix observé − NYX > 50)` est au moins `0,95` ;
3. `max(nyx__q50 du pays, nyx__q50 du partenaire)` est au moins
   `300 EUR/MWh`.

Les partenaires sont `FR ↔ BE` et `NL ↔ DE`, appariés sur la **même heure
physique UTC**. Il n'y a ni interpolation, ni mélange de prix, ni décision
fondée sur Storm. Le sélecteur autonome
`chronos2_hourly/nyx_annual_reference_pair.py` reproduit bit à bit les
8 760 points archivés de FR, BE et NL quand il reçoit exactement les six
signaux historiques. Ces trois séries sont aussi identiques à la colonne
`reference` des fichiers OOF de `rmse_exchange_composition_v1` sur l'année.

## Producteurs indispensables avant la formule

| Signal | Producteur historique | Données et état pour une nouvelle livraison |
|---|---|---|
| `nyx__q50`, `partner_nyx__q50` | Baseline NYX : Chronos-2, correcteur résiduel interaction40, Kalman | Prévisions Saturn et historiques de prix des **deux pays** de la paire ; le replay local originel charge Chronos sur GPU et lit des caches et snapshots locaux. Le sync actuel `NuclearKalman.ps1` ne produit pas ces sorties appariées. |
| `test2__q50`, `spike_probability` | Test2 : modèles CatBoost CPU par paire `BE/FR` ou `DE/NL`, 120 arbres et réentraînement hebdomadaire | Jusqu'à 365 jours antérieurs de baseline NYX, de prix réalisés et de signaux de rareté. La probabilité est celle d'un résidu NYX > 50, pas celle de battre Storm. Pour NL, l'entraînement exige aussi les données DE. |
| `ensemble__q50` | Ensemble fixe des trois HGB de prix et de Test2 plafonné | `0,75 × ((HGB résiduel 400 + HGB absolu 400 + HGB résiduel augmenté 400) / 3) + 0,25 × (NYX + clip(Test2 − NYX, −20, 20))`. Les deux premiers HGB utilisent 292 variables, l'augmenté 334. Leurs prévisions et leurs producteurs de variables sont requis avant de calculer l'ensemble. |
| `prior90_active` | Politique `test2_spike_gate_prior90_v1` | Une règle choisie chaque semaine parmi 24 à partir des **90 jours civils précédents** de prévisions OOF, prix réalisés et sept signaux ; puis appliquée aux signaux de chaque journée. Il faut conserver les sorties historiques du modèle, les prix et la décision. Un simple seuil fixe ne reproduit pas cette colonne. |

Le constructeur historique de Test2 calcule notamment le déficit solaire et
éolien, le stress de charge résiduelle et l'écart du NYX au maximum de sa
journée. La règle `prior90_active` n'est sélectionnée qu'après vérification
des erreurs de ses candidats sur le passé : au moins cinq activations sur
trois jours, autant de gains que de pertes, MAE améliorée et RMSE non
dégradée. Les valeurs de la journée courante et Storm n'entrent pas dans la
sélection.

## Coupure temporelle et disponibilité

Chaque journée de livraison requiert ses prévisions et signaux publiés avant
**D−1 à 08 h locale**, ainsi que les prix passés disponibles à cette date.
Le modèle Test2 est entraîné à son origine hebdomadaire, puis ses signaux de
livraison sont fournis **jour par jour** à chaque coupure D−1 08 h. Il ne faut
pas calculer les sept journées dès l'origine hebdomadaire en utilisant des
données qui ne seraient publiées que les jours suivants. La décision
`prior90_active` emploie 90 journées OOF strictement antérieures. Les heures
des transitions été/hiver restent des heures UTC distinctes. Les archives
locales originales reconnaissent que la publication historique de certaines
sources n'est pas certifiée et que quelques trous ont été comblés par des
proxys documentés. Leur horodatage logique ne certifie pas une disponibilité
effective de fournisseur.

## Contrat de lancement sur un clone GitHub

La prévision annuelle reste **désactivée** tant que les producteurs ci-dessus
ne sont pas reliés à un flux futur et vérifiés sur le poste CPU. Le lancement
doit échouer explicitement si un des six signaux, un pays partenaire, une
heure, la provenance d'un producteur ou une fenêtre de calibrage manque. Il
ne doit pas remplacer silencieusement la référence par NYX, par le modèle
nucléaire en production, par zéro ou par une colonne archivée.

Les scripts de recherche qui ont produit ces signaux, leurs fichiers
`runs/experiments` et plusieurs caches `data/pit` ne sont actuellement pas
suivis dans le dépôt GitHub. `run_nyx_local_365.py` est borné au replay
historique et sa préparation indique elle-même qu'elle n'accède pas à
Saturn. Le sélecteur autonome ne masque pas ces absences : il assemble
seulement six prévisions **déjà produites** et refuse un schéma incomplet.

Le code autonome porte maintenant la formule de l'ensemble
(`nyx_annual_equal_ensemble.py`), l'entraînement CPU Test2 à une origine
hebdomadaire et son scoring quotidien (`nyx_annual_test2_cpu.py`), la règle
prior90 (`nyx_annual_prior90_cpu.py`) et la confirmation par paire
(`nyx_annual_reference_pair.py`). Les tests rapprochent leurs points ou
décisions des archives quand elles sont présentes. Ces modules consomment des
matrices et prévisions fournies ; ils ne collectent aucune source.

Pour rendre ce chemin prospectif, il faut relier la baseline NYX, les trois
HGB et leurs sources de variables à un collecteur
quotidien horodaté, conserver les prédictions OOF et prix de 90 jours, puis
enregistrer les modèles Test2 d'une semaine à l'autre et évaluer la chaîne
CPU entière sur des origines chronologiques. La nouvelle
évaluation des experts CPU avec la référence historique fixe mesure une
étape distincte ; elle ne valide pas seule cette chaîne future.
