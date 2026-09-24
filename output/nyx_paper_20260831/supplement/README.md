# NYX research evidence through 31 August 2026

This supplement accompanies the revised English research manuscript. Evaluation delivery dates are 16 September 2025 through 31 August 2026 inclusive: 350 civil days, 33,600 country–hour observations, and 33,596 matched stage/comparator pairs. The additional residual experiment preserves the original 180-day initial sample and 95-day validation period, followed by 75 retained test days (7,200 country–hours). No configurations were reselected and no model was retrained.

The manuscript contains 29 numbered editable equations, 10 tables and five figures. The source Markdown uses `@table` and `@figure` inclusion directives; the completed Word and PDF files are supplied separately and linked by their checksums in `manuscript_release.json`.

## Contents

- `tables/`: the final publication tables and supporting aggregates. Filenames containing `annual` are retained for compatibility; all their revised contents concern 350 days. Explicit period identifiers are `full350` and `test75`.
- `figures/`: five data-derived figures, each as high-resolution PNG and vector PDF.
- `audit/`: cutoff-specific scores, intervals, provenance, parameter selection, mathematical implementation notes and source fingerprints. Source materialization dates after the cutoff describe provenance, not included evaluation observations.
- `reproduction/`: a tested portable rescoring script, a local table-layout seed, runtime requirements and numerical verification. It reads authorized local inputs and writes only to its own `recomputed/` subdirectory. It does not train models, fetch provider data or modify the source archives.

## Recompute

Use the Python environment and dependencies recorded under `reproduction/`. With access to the original authorized NYX workspace, run from this supplement directory:

```text
python reproduction/recompute_cutoff.py --workspace /path/to/authorized/nyx-workspace
```

The workspace must contain `research/tensor_timesfm_20260915/metrics/`, including the saved paired forecasts, residual test forecasts, extraction manifest, validation selection and rolling fit audit. It must also contain the referenced four-country frozen bundles and source snapshots under `runs/experiments/nuclear_forecast_v1/2026-09-16/`. Exact source paths and checksums are recorded in `audit/cutoff_manifest.json`. The script remaps historical source paths to the supplied workspace. It needs no copy of the previous manuscript's table file: the revised layout seed is included locally.

All revised moving-block intervals use 2,000 replicates, seven-day blocks and seed 20260915. The seed is a reproducibility integer, not the evaluated endpoint. Full precision is retained in CSV; rounding occurs only for manuscript presentation. Positive residual-test differences mean worse than NYX, whereas stage differences explicitly name both models.

## Scope and limitations

The retained test is a truncated subset of a previously examined test period, not a newly untouched holdout. Underlying forecasts and source snapshots were assembled retrospectively. Recorded as-of queries and rolling label checks do not certify provider publication vintages or availability on 31 August. The Chronos checkpoint also postdates the beginning of the evaluated period.

The optimized nonlinear filter candidate has a documented sigma-point covariance-orientation discrepancy. It was selected for 24 NL hours on 21 February 2026, and never during the retained test. This selection count does not bound a corrected rerun's impact on governance. The archived forecasts are unchanged; no corrected-UKF accuracy claim is made.

The package contains aggregate evidence, not licensed hourly observations, prediction matrices, model weights or credentials. Data access and redistribution rights must be handled separately. Reproducing aggregate scores does not recreate full NYX training, establish prospective accuracy, eliminate vintage uncertainty or demonstrate trading profitability.
