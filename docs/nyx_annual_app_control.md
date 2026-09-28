# Contrôle NYX CWE annuel CPU dans l'application

L'espace **Runs et rapports** de `app_multizone.py` contient un panneau séparé
pour le réentraînement CPU et les prévisions FR, BE et NL. Il utilise le jour
de livraison choisi dans l'application. Le bundle attendu est
`runs/live/nyx_annual_cpu/<jour>`, comme dans les adaptateurs de sources
combustible et prix d'enchère. Le consommateur écrit dans
`runs/nyx_annual_cpu_live/<jour>`.

L'application appelle le même contrôle en lecture seule que
`run_nyx_annual_cpu_live.py --preflight`. Le bouton reste inactif si le
manifeste n'active pas les modèles CPU qualifiés, si le reçu de qualification
est absent, ou si une matrice, baseline, référence ou provenance de source
manque. Le service répète ce contrôle juste avant le lancement, sans option
de contournement. Un calcul annuel et un forecast classique ne sont pas
lancés simultanément depuis la même session.

Après un calcul achevé, le panneau affiche le statut, le journal et les prix
et probabilités négatives de chaque pays. Il vérifie le reçu complet et
l'empreinte du CSV avant d'afficher les prévisions.

Cette interface ne collecte pas les sources. La génération automatique du
bundle complet, y compris la mise à jour Saturn et toutes les autres sources,
n'est pas encore disponible dans ce dépôt. Les adaptateurs combustible et
prix d'enchère ne suffisent pas à activer ce bouton.
