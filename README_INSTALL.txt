CHRONOS-2 — CUTOFF PERSISTENCE

Principe
--------
Pour prévoir la journée D, la variable future vaut la dernière valeur de
da_cap_system_asymmetry dont le timestamp est inférieur ou égal à D-1 08:00.
Cette valeur est répétée sur les 24 heures de D.

Deux colonnes sont ajoutées :
- known_da_cap_system_asymmetry_cutoff_persistence
- known_da_cap_system_asymmetry_cutoff_age_hours

Important
---------
L'historique Saturn ne conserve pas les vraies dates de publication anciennes.
Cette implémentation est donc pseudo-causale : elle respecte strictement le
timestamp de livraison et le cutoff, mais utilise la valeur historique finale
associée à ce timestamp.

Installation
------------
Copier tous les fichiers à leur emplacement relatif dans chronos2_v1.

Test :
python -m pytest .\tests\test_cutoff_persistence.py -q

Run complet :
.\run_da_cap_cutoff_persistence.ps1
