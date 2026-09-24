# NYX numerical evidence supplement

This supplement contains aggregate result tables, independent methods/results audits, the saved residual-selection record and protocol, input fingerprints, and two portable audit scripts. It accompanies the 16 September 2026 manuscript. It is **not a complete public replication dataset**: provider observations, hourly forecast matrices, model weights and production archives are not included. Access and redistribution rights must be obtained separately. No credentials are included.

## Contents

- `tables/`: manuscript aggregates and additional audit/sensitivity tables. MAE, RMSE and bias are in EUR/MWh; each CSV records its effective sample or support.
- `audits/`: scientific audit reports and aggregate checks. Individual provider observation values are omitted from the packaged label-revision checks.
- `protocol/`: the 13-candidate validation selection, residual experiment protocol, relative input paths and hashes, and the tested dependency versions.
- `scripts/`: independent rescoring and provenance-sensitivity scripts. They do not fit models, rerun NYX, fetch provider data or modify the source repository.
- `recomputed/`: outputs from the successful test of these portable scripts. Future executions write only here and replace prior recomputed outputs.
- `validation.json`: comparison of the portable outputs against the supplied aggregate tables.
- `sha256_manifest.json`: fingerprints of the supplement files at packaging, excluding the manifest itself. Regeneration or later added figures will require a new packaging manifest.

## Recompute using authorized local inputs

Use Python 3.11 with NumPy, pandas and PyArrow. Exact versions used for verification are in `requirements.txt` and `protocol/tested_environment.json`. In a suitable environment, install them with `python -m pip install -r requirements.txt`.

The chosen NYX workspace must contain the retained research inputs under:

```text
research/tensor_timesfm_20260915/metrics/
```

This includes `verified_hourly_pairs.csv.gz`, `locked_test_predictions.csv.gz`, `extraction_manifest.json`, the saved metrics/bootstrap CSVs, `validation_selection_locked.json`, `experiment_summary.json`, and `rolling_fit_audit.json`. It must also contain the four referenced frozen bundles and reporting source snapshots under `runs/experiments/nuclear_forecast_v1/2026-09-16/{be,de,fr,nl}/civil_pit_v2/`. The original source-manifest paths are remapped to this workspace, so another machine need not reproduce the author's Windows username or absolute directory. `protocol/input_provenance.json` identifies the expected artifacts and fingerprints.

From this supplement directory, run:

```text
python scripts/audit_results.py --workspace /path/to/authorized/nyx-workspace
python scripts/audit_provenance_sensitivity.py --workspace /path/to/authorized/nyx-workspace
```

Alternatively set the `NYX_WORKSPACE` environment variable and omit `--workspace`. If neither is set, the current working directory is used as the workspace. Outputs always go to this supplement's `recomputed/` directory; no market data are downloaded and no archived inputs are rewritten.

The first script verifies fingerprints and time support, rescores the three NYX stages and Storm on common hours, reproduces the saved residual-experiment scores and block intervals, and computes a weekly-naïve comparator without fitting. The second reconstructs the documented Storm fallback mask from complete affected-day sets and exact hour counts, checks that reconstruction, and calculates comparator/label-source sensitivities and interval metrics. The optional byte comparison with `tmp/tensor_timesfm_audit/metrics` only runs for files present in that workspace; its absence does not invalidate score replication.

## Interpretation and reproduction limits

Annual stage intervals are post hoc descriptive paired bootstrap intervals. The 90-day residual experiment selects on the preceding validation segment, but this does not erase prior NYX development or prove external preregistration. Temporal replay checks do not establish original provider publication vintages. Storm is a reporting composite with recomputed fallback hours; cache-only sensitivities still do not prove availability at the historical 08:00 cutoff. The current Chronos checkpoint is applied retrospectively.

The scripts reproduce calculations conditional on the authorized retained inputs. They do not reproduce original model training, confirm hardware/runtime claims, establish prospective accuracy, or resolve pretraining overlap and data-license questions. No model fitting was performed when constructing this supplement.

The tested script copies reproduce all eight newly computed CSV outputs exactly as parsed tables, and the 70 saved residual score rows and 70 bootstrap rows to floating-point precision. See `validation.json` for the measured differences. Platform and dependency changes may produce last-bit differences; substantive discrepancies require investigation rather than silent replacement of the packaged tables.

## Manuscript assets

The final manuscript contains 18 pages, 10 tables and five figures. The `figures/` directory contains the five data-derived figures in high-resolution PNG and vector PDF formats. The `tables/manuscript_tables.json` file records the exact formatted table content. `manuscript_release.json` links this evidence package to the final Word and PDF files by SHA256; those two manuscript files are delivered separately. Figure assets are not additional experimental runs.
