# Independent audit of the existing NYX experimental results

Audit date: 16 September 2026. Scope: existing archived predictions and source audits only. No NYX, Chronos, CatBoost, Kalman, TimesFM, or residual model was trained or rerun. All new calculations are descriptive rescoring, source-integrity checks, a deterministic weekly-naive comparator, or resampling of saved forecast errors. Existing inputs were not modified.

## 1. Overall assessment

The saved outputs support an empirical paper about a **retrospective, prequential electricity-price forecasting system**, its incremental stages, its remaining calibration and tail limitations, and a negative result for several additional residual corrections. They do not support a claim of a prospectively validated deployment, a globally untouched 365-day holdout, superiority to Storm, or demonstrated benefits from Tensor-TimesFM.

On identical annual hours, the complete NYX median improves over the archived Chronos stage by 1.037 EUR/MWh MAE (8.35%) and over the CatBoost residual stage by 0.095 EUR/MWh (0.83%). The comparison with the Storm reporting composite is essentially unresolved: NYX has slightly higher annual pooled MAE and RMSE, and the descriptive block-bootstrap intervals for their differences include zero. In the distinct 90-day residual experiment, validation selects **no additional correction**, and none of the five validation-selected correction families improves test MAE. This is a useful negative result, not evidence that all multivariate residual forecasting is impossible.

The most consequential provenance finding is that the published Storm comparator is **a reporting composite of a frozen day-ahead cache and a live-recomputed native-curve fallback**, with 1,968 fallback country-hours in the annual comparison and 432 in the 90-day period. This must be stated alongside the main comparator table, not only in a distant limitations paragraph.

## 2. Inputs, integrity and actual evaluation support

Primary sources:

- `research/tensor_timesfm_20260915/metrics/verified_hourly_pairs.csv.gz`, SHA256 `0548b72b6fadcb700d1b863db93fd95d0076670fb5e11bf4ad1c1fe1413a1188`.
- `research/tensor_timesfm_20260915/metrics/locked_test_predictions.csv.gz`, selection JSON, extraction manifest, rolling-fit audit, saved score and bootstrap CSVs.
- The four `runs/experiments/nuclear_forecast_v1/2026-09-16/{be,de,fr,nl}/civil_pit_v2/report_only/frozen_result` bundles, their `prepared/pit_selection_audit.json`, and the specific reporting snapshots named in the extraction manifest.
- The extraction and residual-experiment source scripts; `chronos2_hourly/storm_dashboard.py` for fallback-mask semantics.

The audit verified all four bundle-manifest, backtest, source-audit, observed-snapshot and Storm-snapshot hashes against the saved extraction manifest. Every matrix timestamp matches its corresponding bundle. NYX, residual and interval values agree with the bundle within ordinary floating-point precision; serialized Chronos/label values differ by at most 0.000030 EUR/MWh, consistent with CSV serialization of float32 values. These are not economically material discrepancies. Fifteen of the sixteen metrics-directory files shared by `tmp/tensor_timesfm_audit` and `research/tensor_timesfm_20260915` are byte-identical; the only difference is the explanatory `MEASURED_FINDINGS.md`, not a data, score, selection, or experiment-script file.

The annual matrix has 35,040 rows, four countries, 8,760 distinct physical UTC hours, and no duplicate country/timestamp. Its civil delivery dates are **16 September 2025 through 15 September 2026**; its UTC endpoints are 2025-09-15 22:00 and 2026-09-15 21:00. The future delivery day 16 September 2026 is excluded. There are 363 ordinary 24-hour days, one 25-hour autumn day and one 23-hour spring day per country. No target values are imputed for scoring.

The only missing Storm hour in each country is **26 October 2025 at 00:00 UTC**, the first 02:00 civil occurrence. Thus annual four-model comparisons use **8,759 hours per country / 35,036 country-hours**. Standalone NYX calibration uses all 8,760 hours per country. An important table-construction issue is that the original `descriptive_metrics.csv` places NYX/Storm on 8,759 annual hours but Chronos/residual on 8,760. The new `verified_stage_metrics.csv` provides an explicitly common four-model support and should be used for stage-comparison tables.

## 3. Verified annual stage comparison

Latest audited reporting labels; the same 8,759 hours in every row for a country; all errors in EUR/MWh. Bias means prediction minus observation.

| Country | Chronos MAE | Residual MAE | NYX MAE | Storm composite MAE |
|---|---:|---:|---:|---:|
| BE | 12.029471 | 11.264221 | 11.165423 | 10.936282 |
| DE | 12.389952 | 11.209457 | 11.120065 | 11.444472 |
| FR | 12.976877 | 12.234061 | 12.086988 | 11.856627 |
| NL | 12.297010 | 11.218653 | 11.172322 | 10.999551 |
| Pooled | 12.423327 | 11.481598 | 11.386200 | 11.309233 |

| Stage, pooled | RMSE | Bias |
|---|---:|---:|
| Chronos | 23.468025 | -0.397170 |
| Residual | 22.102026 | -0.882962 |
| NYX | 21.929489 | -0.651510 |
| Storm reporting composite | 20.770182 | -2.076905 |

NYX's annual MAE is lower than Storm only in DE. Storm's annual RMSE is lower in all four countries. NYX has smaller absolute bias than Storm in all four countries, but bias reduction alone is not a predictive-accuracy improvement. These results also do not demonstrate that every NYX stage improves every country/hour/regime.

Additional **post hoc, descriptive** paired moving-block intervals were calculated here from the saved forecasts: 2,000 resamples, seven contiguous civil days per block, seed 20260915, the same sampled dates for all countries, and nonlinear RMSE recomputed from squared-error totals. A negative difference favors the first named model.

| Annual comparison | Delta MAE | 95% block interval | Delta RMSE | 95% block interval |
|---|---:|---:|---:|---:|
| NYX minus Chronos | -1.037128 | [-1.269714, -0.759684] | -1.538536 | [-1.978838, -1.046480] |
| Residual minus Chronos | -0.941729 | [-1.162001, -0.668705] | -1.365999 | [-1.825206, -0.895786] |
| NYX minus Residual | -0.095398 | [-0.169442, -0.008679] | -0.172537 | [-0.286522, -0.052324] |
| NYX minus Storm composite | +0.076966 | [-0.531291, +0.759132] | +1.159307 | [-2.657896, +5.032519] |

Pooled block-length sensitivity at 14 and 28 days leaves the stage-improvement intervals below zero and the Storm-comparison intervals crossing zero. These intervals are conditional on the selected system, fixed observed year and saved predictions. They do not account for prior architecture/hyperparameter exploration, uncertain vintages, pretrained-model data overlap, or future seasonal and regime changes. They are not a retrospective conversion of the annual replay into a preregistered confirmatory experiment. The four correlated countries are resampled jointly, not treated as four independent replications.

## 4. Storm and observation provenance: quantified sensitivity

The Storm source audit states `cache_live_recomputation=false` but `fallback_live_recomputation=true`; cache values have precedence and the native curve fills missing cache hours. The native values were materialized during reporting refresh on 15 September 2026. This does not establish their availability at each historical D-1 08:00 cutoff.

| Country | Storm fallback in 365-day sample | Fraction of paired annual hours | Storm fallback in 90-day test | Observation fallback inside both samples |
|---|---:|---:|---:|---:|
| BE | 480 h / 20 days | 5.48% | 96 h | 0 h |
| DE | 480 h / 20 days | 5.48% | 96 h | 24 h |
| FR | 504 h / 21 days | 5.75% | 120 h | 24 h |
| NL | 504 h / 21 days | 5.75% | 120 h | 24 h |
| Total country-hours | 1,968 | 5.62% | 432 | 72 |

**Attribution method and its limit.** The Storm audit does not save an hourly fallback mask. It does save the exhaustive set of affected local dates; source code builds this set directly from the Boolean fallback mask without truncation. In every country, the recorded fallback-hour count exactly equals the full physical-hour capacity of those listed dates (20×24 or 21×24), and every date falls inside the annual evaluation. Therefore every hour on those dates must be a fallback hour; the assignment used in this audit is a deduction from the recorded count, complete date set and civil-hour calendar, rather than a directly stored timestamp flag. The assertions and complete dates are preserved in `source_fallback_detail.json` and `audit_provenance_sensitivity.py`.

The 90-day fallback dates are 13 July, 27 July, 2 August and 14 September 2026 in every country, plus 20 June in FR/NL. No missing or fallback target was interpolated in this audit.

The latest observed-label series uses a reporting-only EPEX fallback for **13 September 2026** in DE/FR/NL, 24 hours each. The source audit records agreement with canonical labels over 216 overlap hours to at most 5.69×10^-14 EUR/MWh. BE has no such fallback inside the evaluated year. Every source snapshot also contains a 24-hour future-label suffix for 16 September, excluded from all reported scores. Label-source substitution and a market-price revision are distinct concepts and should not be conflated.

| Period and retained support | Country-hours | NYX MAE | Storm MAE | NYX RMSE | Storm RMSE |
|---|---:|---:|---:|---:|---:|
| Annual, all common | 35,036 | 11.386200 | 11.309233 | 21.929489 | 20.770182 |
| Annual, Storm cache only | 33,068 | 11.212747 | 11.056475 | 21.192069 | 20.046670 |
| Annual, cache only and no label fallback | 32,996 | 11.196454 | 11.030193 | 21.186110 | 20.021694 |
| Test 90 days, all common | 8,640 | 14.191395 | 12.758226 | 30.196013 | 27.575068 |
| Test, Storm cache only | 8,208 | 14.013126 | 12.489843 | 30.012886 | 27.514777 |
| Test, cache only and no label fallback | 8,136 | 13.971832 | 12.395937 | 30.062373 | 27.498140 |

Excluding fallback hours does not reverse Storm's lower pooled point-estimate MAE. This is a sensitivity analysis on a selected subset, not an unbiased experiment on the effect of fallback data. Even the retained frozen-cache values lack independently demonstrated strict-08 historical availability in this audit. The benchmark must not be promoted to a causal NYX input on this evidence.

## 5. Weekly-naive baseline, newly rescored without fitting

The comparator predicts the frozen label from D-7 at the same civil hour. Both physical target hours are retained on the autumn transition. When the lag date contains a repeated civil hour, its two frozen labels are averaged; when the lag hour does not exist, the target is excluded. No target interpolation, parameter selection, or model fitting is performed.

The first seven dates are unavailable from the supplied one-year matrix. One further lag is missing on 5 April 2026 at 02:00 civil time because 29 March had no 02:00 hour. The naïve/NYX support thus contains 8,591 hours per country. Requiring Storm and all three NYX stages removes its additional missing autumn hour, leaving **358 dates / 8,590 physical hours per country / 34,360 country-hours**, 23 September 2025–15 September 2026.

| Model, identical five-model support | MAE | RMSE | Bias |
|---|---:|---:|---:|
| Weekly naïve | 34.000308 | 51.741954 | -2.420750 |
| Chronos | 12.377551 | 23.487949 | -0.494087 |
| Residual | 11.479873 | 22.144213 | -0.943195 |
| NYX | 11.385034 | 21.973684 | -0.708572 |
| Storm reporting composite | 11.316479 | 20.868270 | -2.088711 |

NYX minus weekly-naive MAE is -22.615274 EUR/MWh, descriptive paired seven-day block interval [-25.304492, -19.766668]. The baseline uses old delivery dates and is temporally admissible under the assumed label lag, but frozen labels are retrospective snapshots: it does not solve the historical-vintage limitation. It is a basic sanity comparator, not a substitute for strong statistical/econometric baselines. Its 358-day scores must not be mixed into the 365-day main table without the explicit support difference.

## 6. The separate 90-day residual experiment

The split is chronologically disjoint at selection time:

| Partition | Dates | Civil days | Physical hours per country | Country-hours |
|---|---|---:|---:|---:|
| Initial train | 2025-09-16–2026-03-14 | 180 | 4,321 | 17,284 |
| Validation | 2026-03-15–2026-06-17 | 95 | 2,279 | 9,116 |
| Test | 2026-06-18–2026-09-15 | 90 | 2,160 | 8,640 |

The implementation considers 13 candidates: unchanged NYX; country EWMA and country/hour EWMA with half-lives 7/28 days; univariate/multivariate ridge with alpha 10/100; PCA-ridge with ranks 1/2 and alpha 10/100. Validation minimizes pooled MAE against `frozen_actual`. A selection JSON is written before generating the test predictions; its SHA matches `experiment_summary.json`. The selected primary family is **NYX**, validation MAE 12.314574. The best candidate within each correction family has MAE 12.411052 (country EWMA, 28), 12.653821 (country/hour, 28), 12.425765 (univariate ridge, 100), 12.690870 (multivariate ridge, 100) and 12.493782 (PCA-ridge, rank 1 / alpha 100).

For delivery D, experiment features assimilate labels through D-2 only: same civil-hour lagged residual, country EWMA and country/hour EWMA. Ridge coefficients are refit weekly over at most 180 eligible days; daily features still update between refits. The first fit uses 165 days after the 14-day feature warm-up, then 172, 179 and 180. Imputation, scaling and PCA are fitted only on the eligible fit rows. All 151 saved refit records satisfy the logged cutoff; 112 concern validation and 39 the selected test families. Earlier test labels later enter the fixed rolling training rule. This is an **adaptive day-ahead prequential test**, not a fixed 90-day forecast from one origin and not an entirely unused block after its first day.

The source's perturbation check tests EWMA invariance at one probe day when D-1/future labels are altered. It is a useful unit-level check, not a universal proof of all source availability. Independent source-code review found the D-2 masks, training-only transforms and fit window coherent. No training was repeated during this audit.

All 70 saved test metric rows were independently reproduced from the saved predictions to a maximum absolute difference of 7.11×10^-15. All 70 saved paired bootstrap rows were reproduced to 1.77×10^-14 using the stated seed and sampling rule.

| Validation-selected family | MAE | RMSE | Bias | Delta MAE versus NYX, 95% blocks |
|---|---:|---:|---:|---:|
| NYX | 14.191395 | 30.196013 | -1.852417 | 0 |
| Country EWMA | 14.223554 | 30.217989 | -0.194787 | +0.032159 [-0.111435, +0.192463] |
| Country/hour EWMA | 14.462651 | 30.261641 | -0.194480 | +0.271256 [+0.106803, +0.474231] |
| Univariate ridge | 14.486237 | 30.185004 | -1.263120 | +0.294842 [+0.140164, +0.481582] |
| Multivariate ridge | 14.658718 | 30.398794 | -0.656582 | +0.467323 [+0.219166, +0.741331] |
| PCA-ridge | 14.366032 | 30.138966 | -1.089562 | +0.174637 [+0.026205, +0.350838] |
| Storm reporting composite, separate comparator | 12.758226 | 27.575068 | -4.045827 | -1.433169 [-3.617815, +0.198607] |

PCA-ridge's RMSE difference is -0.057047 with interval [-0.253983, +0.288563], so a demonstrated RMSE benefit must not be claimed. Country EWMA substantially reduces bias without improving MAE or RMSE. The secondary family/country/regime intervals have no multiplicity adjustment and the test covers June–September, not a full annual cycle. These already-inspected 90 days are no longer a fresh holdout for future hypotheses.

## 7. Calibration, revisions and dependence

Annual nominal P10–P90 coverage is **74.415% pooled**, with BE 74.463%, DE 74.338%, FR 72.991%, NL 75.868%. The 80% interval is undercovered in all four countries. Pooled mean width is 31.656634 EUR/MWh, interval score 58.924382, and weighted interval score 7.723490 using **one 80% interval plus the median**, not a full quantile grid. There are zero quantile crossings. The same additive median shifts preserve the inherited Chronos widths up to float rounding (maximum 3.06×10^-5); this is not a learned recalibration of dispersion. The residual prototypes have only point predictions, so no new interval-calibration claim is supported.

The source reports severe underprediction of high-price events; its fixed initial-train q99 threshold should always accompany the regime definition. For example, DE has 104 annual hours above its train q99 threshold, NYX MAE 102.26 EUR/MWh and coverage 33.65%; NL has 155 such hours, MAE 73.23 and coverage 52.26%. These are descriptive subsets defined from realized evaluation prices, not regimes identifiable in advance. Future exceedance frequencies are not constrained to be 1%.

There is one material frozen/latest label discrepancy per country, at 16 June 2026 21:00 UTC, in the validation partition. Approximate absolute differences are BE 9.3050, DE 9.0575, FR 6.9550, NL 9.2900 EUR/MWh. On the CSV matrix, annual NYX MAE changes by +0.001062, +0.001034, -0.000643 and -0.000292 respectively when switching from frozen to latest labels. The saved 90-day CSV labels agree and yield identical point scores; original float32-versus-decimal representation differences are immaterial. Small observed revision sensitivity does **not** establish correct historical input vintages or prove that no other revision process mattered.

The archived initial-train dependence analysis reports a first residual covariance axis explaining 65.70% of raw variance, two axes 85.89%, and participation rank 2.07 across four countries. This describes simultaneous errors. It is neither causal predictability nor evidence of useful information available before delivery. A projection receiving realized future errors is an oracle reconstruction diagnostic and must remain separate from forecast accuracy. No Tensor-TimesFM predictions exist in this evidence set.

## 8. Causality and selection qualifications required in the manuscript

1. **Algorithmic ordering versus data vintages.** All four replay audits record zero causality violations and zero target observations assimilated before forecasting. Their 365 logged Kalman windows each end before the target delivery date. This supports the coded replay ordering. However, every checked preparation audit explicitly reports `provider_revision_timestamp_available=false` and `production_pit_evidence=false`. Nuclear timestamps mean query-as-of cutoff, not an independently returned provider insertion time. The paper should say “cutoff-constrained retrospective replay” rather than “fully proven point-in-time historical simulation.”
2. **Annual replay versus untouched holdout.** Base-system residual/Kalman stages use earlier training history and rolling updates, but the run records `warmup_scope=fixed_epoch_diagnostic_warmup_not_full_nested_validation`. The local artifacts do not establish that the entire NYX architecture, its recipes, caps and governance thresholds were frozen before anyone inspected all 365 evaluation dates. Do not call this a sealed 365-day research holdout.
3. **Residual-selection lock has a limited scope.** Writing the JSON before this script's test-generation step is verifiable. External preregistration or proof that the researcher had never seen the last 90 days during earlier NYX work is not provided. Call it a “validation-selected residual experiment with a reserved chronological 90-day evaluation,” while explicitly acknowledging the broader retrospective setting.
4. **Pretrained weights and availability.** A pinned Chronos checkpoint and archived source hashes improve reproducibility; they do not establish that pretraining data are disjoint from every historical evaluation timestamp or that the same model existed at each simulated origin. Do not assert a contamination-free prospective baseline without additional evidence.
5. **Stage comparisons are not feature-causal experiments.** The data compare the complete archived stages. They do not isolate the causal value of the nuclear covariate, each residual-load covariate, pretraining, or governance. There is no matched “without nuclear” rerun here. The internal feature name containing `oracle` is not alone evidence of realized-future generation: the documented nuclear source is a forecast series. Availability must be assessed from its source contract, not its alias.
6. **Generalization scope.** Four connected CWE markets over one annual cycle do not establish cross-domain foundation-model generality. A weekly naïve is informative but does not replace a strong LEAR/ARX or other electricity forecasting baseline. Do not claim state of the art, trading profitability, operational savings or statistically proven superiority to Storm.

## 9. Suggested defensible manuscript claims

Recommended central statement: “We evaluate NYX, a frozen Chronos-2 backbone with supervised residual correction and a governed online Kalman stage, using archived retrospective prequential predictions for four European electricity markets. On 35,036 common country-hours, the complete system reduces pooled MAE from 12.423 to 11.386 EUR/MWh relative to its Chronos stage. A separately validation-selected experiment finds no MAE improvement from additional lagged EWMA, ridge or PCA-ridge residual corrections over a reserved 90-day period. Intervals remain undercovered, and comparison with a partly reconstructed Storm reporting benchmark is qualified by source-vintage uncertainty.”

Recommended results wording: “The Kalman stage contributes a modest additional reduction relative to the existing residual stage in this replay”; “Residual dependence did not translate into an improvement for the tested lightweight prediction recipes”; “The Storm reporting composite has lower pooled MAE, but comparative uncertainty and provenance limitations preclude a claim of reliable superiority in either direction.”

Avoid: “causally superior,” “a 365-day untouched sealed holdout,” “fully real-time PIT-verified,” “NYX outperforms Storm,” “factorization improves forecasting,” “Tensor-TimesFM complementary to NYX,” “zero-shot NYX” (the backbone is frozen, downstream components are trained), “calibrated 80% intervals,” and “no leakage” without restricting the claim to the specific algorithmic controls actually checked.

## 10. New audit outputs

All files below are in `tmp/nyx_paper_20260916/` and were newly created for this audit:

- `verified_stage_metrics.csv`: annual and split-wise stage scores, explicit all-NYX versus four-model-common supports.
- `stage_paired_block_bootstrap.csv`: new descriptive stage differences; 7-day country/pooled intervals and 14/28-day pooled sensitivity.
- `verified_residual_test_metrics.csv`: independently recalculated saved 90-day metrics and exact bootstrap results.
- `weekly_naive_matched_metrics.csv`, `weekly_naive_paired_bootstrap.csv`: deterministic weekly-naïve comparison with common-support definitions and both label versions.
- `source_fallback_counts.csv`, `source_fallback_detail.json`: source substitution counts and attribution evidence.
- `storm_provenance_sensitivity.csv`: NYX/Storm scores with and without comparator/label fallback hours.
- `verified_interval_metrics.csv`: independently recomputed annual coverage and interval scores.
- `results_audit_checks.json`: integrity, support, replay-window, refit and label-revision checks.
- `audit_results.py`, `audit_provenance_sensitivity.py`: reproducible read-only computations on existing artifacts. They require the existing `pricefm311` Python environment for Parquet support; neither script trains a model or fetches market data.

The analysis has deliberately not reopened model selection or tuned a new correction after reading the test. Future confirmatory work needs a genuinely new period or a separately specified historical evaluation with auditable vintages and a complete model-selection record.
