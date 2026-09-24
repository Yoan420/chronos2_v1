# Coverage audit: 31 August 2025–31 August 2026

Read-only audit performed on 16 September 2026. No model, forecast, original archive or previous analysis was modified; no training, replay or forecast generation was run.

## Conclusion

The requested inclusive study window contains **366 civil days**. The existing, consistently identified NYX evidence supports **350 evaluated civil days, 16 September 2025–31 August 2026**. The preceding 16 days, 31 August–15 September 2025, cannot be added to the complete NYX evaluation by changing dates alone.

The smallest scientifically valid revision is to state both dates explicitly: **the study window is 31 August 2025–31 August 2026, while quantitative NYX results use the available 350-day subset beginning 16 September 2025**. Preserve the current scores, matched supports, bootstrap calculations and 180/95/75-day residual experiment. Do not label those statistics as covering all 366 days.

The complete requested physical-hour grid would contain 8,784 hours per country and 35,136 country–hour pairs, because the one 25-hour day and one 23-hour day cancel. The current NYX sample contains 8,400 per country and 33,600 in total, with 33,596 pairs in the Storm-matched comparison. The absent prefix accounts for 384 hours per country and 1,536 country–hour pairs. A complete future matched-support count cannot be assumed without auditing the missing Storm dates.

## Evidence from the exact archives already used

The source identity is retained in `research/tensor_timesfm_20260915/metrics/extraction_manifest.json`. For each lower-case country code `z`, the frozen bundle is:

`runs/experiments/nuclear_forecast_v1/2026-09-16/z/civil_pit_v2/report_only/frozen_result/`

All four bundles have the same time coverage:

| Archived object | Civil-day range | Rows in 31 Aug–15 Sep 2025 per country | Interpretation |
|---|---|---:|---|
| `raw_history.parquet` | 16 Sep 2024–15 Sep 2026 | Prefix represented in the upstream history | Chronos historical quantiles and frozen observations; not final NYX output |
| `residual_statistics.parquet` | 16 Sep 2024–15 Sep 2026 | 384 | Frozen actual, Chronos P10/P50/P90 and CatBoost-corrected P10/P50/P90 are present |
| `kalman_backtest.parquet` | 16 Sep 2025–15 Sep 2026 | **0** | Final governed NYX P10/P50/P90, shift, weight and selected candidate are absent for the requested prefix |
| The selected `observed_latest.parquet` reporting snapshot | 16 Sep 2025–16 Sep 2026 | **0** | The refreshed observation version used in the published analysis does not cover the prefix |
| The selected `storm_dashboard_official_statistics.parquet` reporting snapshot | 16 Sep 2025–16 Sep 2026 | **0** | The same frozen Storm reporting composite does not cover the prefix |

The original `snapshot/target.parquet` contains older target history (the inspected BE snapshot begins in August 2022). Thus the issue is not that all historical prices are absent. The issue is the absence of the final NYX forecasts and of the selected aligned reporting snapshots for those dates. An older target series cannot be substituted for the audited refreshed label version without disclosing and checking the change.

The 16 prefix dates in `residual_daily_audit.parquet` are explicitly marked `phase=diagnostic_warmup`, with `generation_source=daily_prequential_refit`. These are real retained upstream outputs, but are not final governed NYX outputs. At 31 August 2025, the CatBoost audit records 8,520 training rows for BE/DE, starting 10 September 2024, and 8,544 for FR/NL, starting 9 September 2024. The available early upstream output therefore also has a shorter initial fitting history than a full 365-day sample. No claim of an unchanged complete-window experiment should be inferred from its availability.

## Other archives and caches inspected

All 27 retained `kalman_backtest.parquet` files under `runs/experiments/nuclear_forecast_v1/` were checked. Their run directories are dated 9, 10, 11, 12, 15, 16 and 17 September 2026. Earliest final-output dates are:

| Country | Earliest final NYX date in those bundles | Bundle run date | Earliest cached rolling-Kalman day |
|---|---|---|---|
| BE | 9 Sep 2025 | 9 Sep 2026 | 10 Sep 2025 |
| DE | 10 Sep 2025 | 10 Sep 2026 | 10 Sep 2025 |
| FR | 9 Sep 2025 | 9 Sep 2026 | 9 Sep 2025 |
| NL | 9 Sep 2025 | 9 Sep 2026 | 9 Sep 2025 |

The rolling cache is under `_daily_cache/{country}/civil_pit_v2/epochs/*/kalman_rolling/`. Its dated cache entries were inventoried without loading pickle payloads. These caches do not contain the missing 31 August–8 September 2025 forecasts; DE also lacks 9 September. Combining bundles would still fail to cover the whole interval, and would change archive identity and potentially input versions. No such combination was made or certified equivalent.

The oldest frozen upstream residual-statistics histories in this family begin 9 September 2024 for BE/FR/NL and 10 September 2024 for DE. The retained Chronos daily-cache histories begin 9 or 10 September 2024. They do not supply the complete missing August 2024 prefix needed by the strict rolling filter.

Separate native Storm materializations were found for BE/DE/NL under `runs/materialized/`, named `*_storm_dashboard_native_20250812_20260811.parquet`. The inspected BE file starts at 11 August 2025 22:00 UTC. These are different materializations, not the selected reporting-composite snapshots; they were not merged or certified as an equivalent benchmark. No exhaustive inventory of all possible external/provider data was attempted, because the complete-NYX coverage limitation is already decisive.

## Feasibility of replaying only the governed filter

Using only the **same 16 September frozen residual-statistics bundle** does not permit an unchanged replay for the missing 16 dates. For delivery 31 August 2025, the required D−365…D−1 history is 31 August 2024–30 August 2025. The stored statistics and prepared filter covariates start 16 September 2024, leaving **16 required civil days missing**. Only 349 prior days are present. The deficit shrinks to one day for delivery 15 September 2025.

The implementation enforces complete fixed-length windows:

- `chronos2_hourly/kalman_residual.py`, `_rolling_training_frame`, lines 1439–1459, raises `KalmanResidualError` when any required prior day is absent.
- The rolling-cache validation repeats that completeness requirement at lines 1910–1921.
- The operational wrapper documents 365 complete prior days at lines 2986–2987 and validates the contract before replay.
- The archived audit states `training_policy=fixed_length_rolling_local_days`, `training_lookback_days=365`, `warmup_days=365`, four rolling workers and `pykalman_version=0.11.2`.

The five recorded candidates are `linear_bias`, `linear_harmonic`, `linear_market`, `linear_scale` and `ukf_scale`. The source snapshot includes their parameter identity. The current inspected implementation contains the already disclosed UKF sigma-point orientation issue; no code correction was applied. Inspecting current code and recorded parameter identity does not independently establish a complete historical software lock.

A 349-day window, an identity correction for absent dates, an expanding-history fallback, or transplantation of another run's final outputs would change the evaluated procedure. None is an authorized date-only revision. A valid unchanged-protocol extension would first require compatible earlier upstream statistics and covariates, plus aligned observation and Storm provenance, and would then constitute a new explicitly documented experiment.

If all compatible missing inputs were supplied, 16 days across four countries would require 64 daily filter replays. At five candidates and roughly 8,760 hourly observations per replay, this is approximately 2.8 million candidate-hour update steps, excluding governance and feature processing. This is an operation-count estimate, not a measured runtime or a promise of a cheap run. No trustworthy wall-clock timing for that hypothetical task was established. No Chronos or CatBoost fitting would be necessary only if the missing compatible upstream outputs already existed; that condition is not met by the inspected archives.

## Suggested manuscript wording

“The study window extends from 31 August 2025 through 31 August 2026 inclusive (366 civil days). The retained, consistently identified NYX forecast archive covers 16 September 2025–31 August 2026 within that window, so the quantitative evaluation uses these 350 days. Complete governed NYX forecasts are unavailable in the selected archive for 31 August–15 September 2025; those dates are not imputed or included in the reported scores.”

Keep all results tied to that 350-day support. Retain the existing validation and test dates rather than relabelling the initial 180 days as beginning 31 August. The source snapshots remain retrospective September 2026 materializations; neither the new study-window wording nor retained archive coverage proves their availability at historical forecast origins.
