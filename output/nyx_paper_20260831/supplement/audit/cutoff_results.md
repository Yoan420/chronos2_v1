# Results restricted to delivery dates through 31 August 2026

This revision filters existing saved forecasts; it does not train models, rerun the forecasting pipeline, select new parameters, or alter source archives. The reporting cutoff is **31 August 2026 inclusive in each country's civil time**. The original beginning, 16 September 2025, is retained. Source forecasts belong to the 16 September 2026 archive and refreshed source snapshots were extracted on 15 September 2026. These are retrospective source versions: **nothing here establishes that the archives or values were available on 31 August**.

## Sample, chronology and filenames

The full retained evaluation has **350 delivery dates, 8,400 physical hours per country and 33,600 country–hour observations**. UTC endpoints are 2025-09-15 22:00 and 2026-08-31 21:00. It contains 348 ordinary 24-hour days, one 25-hour day and one 23-hour day. The original missing Storm autumn hour, 2025-10-26 00:00 UTC, remains excluded from matched comparisons, leaving **8,399 hours per country / 33,596 pairs**.

The residual experiment keeps the original split boundaries and original selected configurations:

| Partition | Inclusive delivery dates | Days | Hours per country | Country–hours |
|---|---|---:|---:|---:|
| Initial train | 16 September 2025–14 March 2026 | 180 | 4,321 | 17,284 |
| Validation | 15 March–17 June 2026 | 95 | 2,279 | 9,116 |
| Retained test | 18 June–31 August 2026 | 75 | 1,800 | 7,200 |

This is a truncated subset of the previously inspected 90-day test, not a newly untouched or preregistered holdout. The date restriction must not be presented as additional out-of-sample evidence. The primary validation choice remains identity/NYX, validation MAE 12.314574. The five correction settings remain country EWMA half-life 28, country/hour EWMA half-life 28, univariate ridge alpha 100, multivariate ridge alpha 100, and PCA-ridge rank 1 / alpha 100. None was reselected after truncation.

For compatibility, `annual_paired_metrics.csv` and table dictionary keys `annual_stages` / `annual_benchmark` retain their old names. **Their contents now concern 350 days; “annual” must not appear as the evaluation-period label in the revised article.** `verified_stage_metrics.csv`, bootstrap tables and sensitivity tables use `period=full350` and `period=test75`. `monthly_mae.csv` has 12 bins, September 2025–August 2026; only September 2025 is partial. The original 365-day training windows inside NYX remain actual model settings and must not be mechanically changed to 350.

## Main stage and comparator results

Latest audited reporting labels, identical four-model support; units EUR/MWh:

| Country | Chronos MAE | Residual MAE | NYX MAE | Storm composite MAE |
|---|---:|---:|---:|---:|
| BE | 11.760731 | 11.010942 | 10.907909 | 10.698148 |
| DE | 12.102428 | 10.990640 | 10.912411 | 11.295950 |
| FR | 12.841732 | 12.114677 | 11.966398 | 11.841861 |
| NL | 12.085095 | 11.018661 | 10.970221 | 10.812053 |
| Pooled | 12.197496 | 11.283730 | 11.189235 | 11.162003 |

Pooled RMSE is 23.188357 (Chronos), 21.875806 (residual), 21.707804 (NYX) and 20.685584 (Storm). Pooled bias is -0.446821, -0.901180, -0.661665 and -1.858191 respectively. The NYX reduction relative to Chronos is about **8.27%**. NYX still has a lower Storm-comparison MAE only in DE, a higher RMSE in every country, and a smaller absolute bias in every country.

All uncertainty intervals were **recomputed for the retained sample** using 2,000 paired moving-block resamples, seven contiguous civil dates per block, seed 20260915, common sampled dates across countries and models, percentile 95% endpoints. The original 365/90-day intervals are not reused. Pooled squared-error sums reconstruct RMSE in each draw.

| Full 350-day difference | Delta MAE | 95% block interval | Delta RMSE | 95% block interval |
|---|---:|---:|---:|---:|
| NYX minus Chronos | -1.008262 | [-1.239764, -0.715223] | -1.480553 | [-1.949967, -0.955358] |
| Residual minus Chronos | -0.913767 | [-1.119252, -0.640465] | -1.312551 | [-1.785389, -0.809005] |
| NYX minus Residual | -0.094495 | [-0.176187, -0.007471] | -0.168002 | [-0.292838, -0.038738] |
| NYX minus Storm | +0.027232 | [-0.561292, +0.735622] | +1.022220 | [-2.949918, +4.984116] |

These are conditional descriptive intervals, without adjustment for historical model development, vintage uncertainty, period selection or multiple comparisons. The date-restricted sample is shorter than a full annual cycle.

## Retained 75-day residual test

| Model | MAE | RMSE | Bias | Delta MAE versus NYX, 95% interval |
|---|---:|---:|---:|---:|
| NYX | 13.833375 | 30.868103 | -2.139981 | 0 |
| Country EWMA | 13.861383 | 30.887158 | -0.331201 | +0.028008 [-0.144434, +0.227613] |
| Country/hour EWMA | 14.136868 | 30.973884 | -0.330852 | +0.303493 [+0.105006, +0.535072] |
| Univariate ridge | 14.176796 | 30.866441 | -1.812311 | +0.343420 [+0.196109, +0.553763] |
| Multivariate ridge | 14.320605 | 31.038962 | -1.164609 | +0.487230 [+0.203132, +0.837584] |
| PCA-ridge | 14.070391 | 30.832374 | -1.566049 | +0.237016 [+0.086314, +0.426727] |
| Storm reporting composite | 12.361033 | 28.457278 | -3.419065 | -1.472342 [-4.154566, +0.263907] |

None of the five previously selected correction configurations improves MAE. The PCA-ridge RMSE difference is -0.035728 with interval [-0.274786, +0.358877], so it still establishes no RMSE benefit. The validation table remains unchanged. Both frozen and refreshed test labels are scored; they give identical point scores on the saved 75-day matrix.

The original fit audit has 151 records: 112 validation and 39 test. **145 records contribute predictions before the new cutoff: 112 validation and 33 test** (11 weekly fits for each of the three ridge families). The omitted six records are the three-family refits on 3 and 10 September. The last retained test refit is **27 August**, with labels through **25 August**. Its originally saved seven-day forecast block is retained only for 27–31 August, five days / 120 physical hours per country. The coefficients are the original coefficients; forecasts still use daily eligible D-2 features. Earlier test labels may enter later fits under the already fixed rule.

Each country's NYX archive contributes 350 retained Kalman rolling-window records. All end before their target date and record zero target observations assimilated. No new fitting was executed to create the restricted results.

## Weekly naïve, intervals and tails

The weekly naïve uses the frozen label at the same civil hour on D-7. Repeated lag hours are averaged; a missing lag is excluded and no target is filled. Its support is **343 delivery dates**, 23 September 2025–31 August 2026. Naïve/NYX alone have 8,231 hours per country; the five-model common support has **8,230 / country, 32,920 total** after removing Storm's missing autumn hour.

On this common support, pooled MAE is 33.645229 for weekly naïve, 12.145080 Chronos, 11.277866 residual, 11.183974 NYX and 11.166543 Storm. The two naïve CSVs include both label versions, country-specific scores and newly recomputed paired intervals. These numbers must not be mixed with the 350-day main table because support differs.

Nominal 80% interval coverage is **74.693% pooled**, with BE 74.667%, DE 74.643%, FR 73.226% and NL 76.238%. Pooled mean width is 31.391130, interval score 57.777111 and one-interval WIS 7.581350. Point shifts still preserve interval widths; no recalibration was performed.

The initial-training thresholds remain fixed at q99 = 183.817993 (BE), 249.923999 (DE), 158.460995 (FR), 216.531995 (NL) EUR/MWh. Exceedance metrics change materially when September is removed:

| Country | High-price hours | MAE | Bias | P10–P90 coverage |
|---|---:|---:|---:|---:|
| BE | 243 | 51.260225 | -34.015567 | 56.790% |
| DE | 85 | 109.510263 | -95.018456 | 29.412% |
| FR | 427 | 16.290533 | -9.418024 | 70.726% |
| NL | 105 | 95.786068 | -74.472971 | 43.810% |

Negative-price counts are BE 328, DE 507, FR 566 and NL 426. These regimes condition on realized prices. Their tail coverage and signed error are diagnostics, not independent evidence of conditional miscalibration or advance event detection. Full-period high-price summaries still include the initial training sample used to set q99.

Residual cap-hit rates over all hours are BE 0.119%, DE 0.524%, FR 0.143%, NL 0.429%; within high-price hours they are 0.823%, 16.471%, 0% and 11.429%. No retained final Kalman shift reaches its 20 EUR/MWh cap. Do not carry over the old 365-day saturation percentages.

The initial 180-day correlation analysis is unchanged by truncation. Recomputed covariance gives first-component variance share 65.70%, first two 85.89%, participation ratio 2.07, and standardized first share 63.65%. The preserved train sample is the same 4,321 simultaneous physical hours. No predictive factor model is inferred from this contemporaneous dependence.

## Cutoff-aware provenance

| Country | Storm fallback full 350 days | Storm fallback retained test | Observed-label EPEX fallback before cutoff |
|---|---:|---:|---:|
| BE | 456 h | 72 h | 0 |
| DE | 456 h | 72 h | 0 |
| FR | 480 h | 96 h | 0 |
| NL | 480 h | 96 h | 0 |
| Country–hours | 1,872 | 336 | 0 |

Full-period Storm fallback shares are 5.429% in BE/DE and 5.715% in FR/NL. Test shares are 4% and 5.333% respectively. The 14 September fallback day is removed in all countries. The 13 September EPEX label replacement lies outside the sample, so **there is no EPEX label-fallback sensitivity difference in the revised results**. The source snapshot still records these later observations, but they are never scored.

The original exhaustive fallback-date sets and total counts saturate those dates' physical-hour capacities, permitting reconstruction of the hourly mask. That assertion is checked before applying the cutoff. This does not validate their historical availability: the native fallback was still recomputed in September.

| Period/support | Pairs | NYX MAE | Storm MAE |
|---|---:|---:|---:|
| Full 350 days, common | 33,596 | 11.189235 | 11.162003 |
| Full 350 days, cache-only | 31,724 | 11.037799 | 10.923739 |
| Test 75 days, common | 7,200 | 13.833375 | 12.361033 |
| Test 75 days, cache-only | 6,864 | 13.752878 | 12.157021 |

The redundant `cache_only_no_label_fallback` rows remain in the sensitivity CSV for backward compatibility and exactly equal cache-only rows. The publication Table 9 omits the redundant rows and explains why. Some sensitivity supports have unequal country counts; pooled scores remain weighted by country–hour.

The material frozen/latest label discrepancy on 16 June 2026 remains in validation: one hour per country, maximum absolute differences 9.3050, 9.0575, 6.9550 and 9.2900 EUR/MWh. Its effect on full-period NYX MAE is +0.001108, +0.001078, -0.000671 and -0.000304 respectively. Source labels are retrospective even though these effects are small.

## Observed UKF selection exposure

The archived `kalman_selected_filter` is inspected without changing forecasts. Over the retained 350-day window, UKF scale is selected on **zero BE, DE or FR dates and one NL date: 21 February 2026**, 24 physical hours, weight 0.45. The applied shifts range from -0.735899 to -0.735300 EUR/MWh. No retained 75-day test date selects UKF scale. Exact country/filter counts and the selected date appear in `governance_selected_filter_counts.csv` and `ukf_selected_days.csv`.

A separate implementation review has raised a sigma-point covariance-orientation concern. These counts quantify direct selection in the existing outputs only. They do **not** show that correcting UKF would leave other governance choices unchanged; a changed candidate may win on dates on which it was originally rejected. No corrected-UKF performance claim or re-estimation is made here. The retained predictions continue to represent the actually archived implementation.

## Reproduction and publication inputs

Run `python recompute_cutoff.py --workspace <authorized NYX workspace>` using the existing Python environment with NumPy, pandas and PyArrow. Alternatively set `NYX_WORKSPACE`, or execute from the workspace root. The script writes only to this new `analysis/` directory, reads source archives, and never calls a model estimator.

`manuscript_tables.json` supplies all ten table dictionaries. Numerical Table 1 and Tables 4–9 are rebuilt for the new window; methodological Tables 2, 3 and 10 preserve the prior draft's verified settings and theory. Their equation cross-references may need adjustment in the newly expanded theoretical manuscript. Scores, stage bootstrap, residual bootstrap, naïve, intervals, extremes, saturation, provenance sensitivity and monthly results have compatible filenames. `verified_hourly_pairs.csv.gz` and `locked_test_predictions.csv.gz` are locally filtered derived matrices for figure/reproduction use, not a public redistribution authorization. `extraction_manifest.json` identifies their truncated evaluation while keeping original snapshot provenance explicit.

The source selection record is copied unchanged. `cutoff_manifest.json` records exact retained dates, counts, admissible fit records, original snapshot dates and source SHA256 values. `output_sha256_manifest.json` fingerprints generated artifacts. Hash integrity and temporal record checks establish reproducibility of this restricted rescoring, not provider-vintage certification, a fresh holdout, prospective deployment or correction of the separate UKF implementation concern.
