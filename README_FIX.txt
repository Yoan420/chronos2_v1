CORRECTIF STRUCTURAL MARKET — missing residual_load + timezone validator

1. Corrige le merge du validateur en convertissant explicitement tous les
   timestamps en UTC avant comparaison.
2. Autorise une réparation intra-journalière de residual_load_gw pour au plus
   2 heures manquantes par jour.
3. La réparation n'utilise que la courbe du même jour déjà matérialisée par le
   pipeline point-in-time.
4. Plus de 2 heures manquantes => la journée reste rejetée.
5. Les diagnostics enregistrent le nombre et les timestamps imputés.

Configuration :
structural_model:
  input_repair:
    impute_residual_load: true
    max_residual_load_missing_hours_per_day: 2
